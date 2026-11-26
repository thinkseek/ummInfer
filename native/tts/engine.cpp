// SPDX-License-Identifier: Apache-2.0
#include "engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "codec.hpp"
#include "common/cuda_request_allocator.hpp"
#include "cuda_kernels.hpp"
#include "model_config.hpp"
#include "prompt.hpp"
#include "safetensors.hpp"
#include "speaker_encoder.hpp"
#include "tail_trim.hpp"

namespace umminfer::native {
namespace {

// Operational capacity, independent of the checkpoint size: KV cache holds up
// to 8192 generated frames + prompt headroom; prefill scratch only needs to
// cover max prompt rows (L <= 200), saving ~570 MB VRAM vs sizing scratch for
// the full decode length.
constexpr int kMaxSlots = 8704;  // 8192 frames + prompt headroom
constexpr int kMaxPrefillRows = 256;
constexpr int kCpSlots = 32;
constexpr int kCpRows = 32;  // prefill pair plus the 15 predictor steps

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::invalid_argument("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

struct DenseB {
  const Bf16* w = nullptr;
  const Bf16* bias = nullptr;
  int in = 0, out = 0;
};

// Pre-resolved layer weight pointers so decode loops perform zero string
// allocations or hash-map lookups.
struct DecoderLayerWeights {
  const Bf16* input_ln = nullptr;
  const Bf16* q_proj = nullptr;
  const Bf16* k_proj = nullptr;
  const Bf16* v_proj = nullptr;
  const Bf16* q_norm = nullptr;
  const Bf16* k_norm = nullptr;
  const Bf16* o_proj = nullptr;
  const Bf16* post_ln = nullptr;
  const Bf16* gate_proj = nullptr;
  const Bf16* up_proj = nullptr;
  const Bf16* down_proj = nullptr;
};

// Per-width scratch a decoder layer reads and writes.
struct LayerScratch {
  Bf16 *norm = nullptr, *q = nullptr, *k = nullptr, *v = nullptr,
       *attn = nullptr, *gate = nullptr, *up = nullptr;
};

bool env_flag_enabled(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && v[0] != '\0' && std::string(v) != "0";
}

}  // namespace

struct TtsEngine::Impl {
  TtsModelConfig cfg;
  cudaStream_t stream = nullptr;
  int gpu_id = 0;

  std::unordered_map<std::string, Bf16*> bw;  // talker BF16
  Bf16* d_bf = nullptr;
  std::unordered_map<std::string, const float*> fw;  // codec F32
  float* d_f32 = nullptr;
  std::unique_ptr<CodecDecoder> codec;
  std::unique_ptr<SpeakerEncoder> speaker_encoder;

  // Pre-resolved model weight pointers.
  std::vector<DecoderLayerWeights> talker_layers;
  std::vector<DecoderLayerWeights> cp_layers;
  const Bf16* text_emb_w = nullptr;
  DenseB text_fc1;
  DenseB text_fc2;
  const Bf16* talker_codec_emb_w = nullptr;
  const Bf16* talker_norm_w = nullptr;
  const Bf16* talker_codec_head_w = nullptr;
  const Bf16* cp_norm_w = nullptr;
  DenseB mtp_proj;
  std::vector<const Bf16*> cp_codec_emb_w;  // [groups - 1]
  std::vector<const Bf16*> cp_lm_head_w;    // [groups - 1]
  const Bf16** d_frame_emb_tables = nullptr;  // device pointer array [groups]

  // Talker-width scratch (sized for prefill <= kMaxPrefillRows).
  Bf16 *hidden = nullptr, *norm = nullptr, *q = nullptr, *k = nullptr,
       *v = nullptr, *attn = nullptr, *gate = nullptr, *up = nullptr,
       *proj = nullptr, *text_in = nullptr, *text_fc = nullptr,
       *text_out = nullptr;
  Bf16* decode_x = nullptr;  // fixed 1-row talker decode hidden state [H]
  LayerScratch tk;

  // Code-predictor-width scratch.
  Bf16 *cp_x = nullptr, *cp_norm = nullptr, *cp_q = nullptr, *cp_k = nullptr,
       *cp_v = nullptr, *cp_attn = nullptr, *cp_gate = nullptr,
       *cp_up = nullptr;
  LayerScratch cp;

  // Control/index buffers.
  int *d_ids = nullptr, *d_pos = nullptr, *d_codes = nullptr, *d_hist = nullptr;
  int* d_step_pos = nullptr;       // single device int holding current seq_len
  int* h_codes_pinned = nullptr;   // pinned host staging [16] for c0..c15
  float* logits = nullptr;
  float* d_inv = nullptr;
  Bf16* talker_kv = nullptr;
  Bf16* cp_kv = nullptr;
  Bf16 *d_pad = nullptr, *d_bos = nullptr, *d_eos = nullptr;
  Bf16* d_xvector = nullptr;
  Bf16* cp_past = nullptr;  // talker-width final-norm row for codec_head & CP

  // Captured CUDA graphs for zero-overhead per-frame execution.
  cudaGraphExec_t cp_graph_exec = nullptr;
  cudaGraphExec_t talker_graph_exec = nullptr;

  const Bf16* wb(const std::string& n) const {
    auto it = bw.find(n);
    if (it == bw.end()) throw std::runtime_error("missing BF16 tensor: " + n);
    return it->second;
  }
  DenseB dense(const std::string& pre, int in, int out, bool bias) const {
    return {wb(pre + ".weight"), bias ? wb(pre + ".bias") : nullptr, in, out};
  }
  void lin(Bf16* y, const Bf16* x, const DenseB& d, int m) {
    gemm_t<Bf16>(y, x, d.w, d.bias, m, d.in, d.out, stream);
  }

  void validate_shapes(const SafeTensors& st) const {
    for (const TensorShape& want : cfg.tensor_shape_plan()) {
      const TensorInfo& info = st.info(want.name);
      if (info.dtype != "BF16")
        throw std::runtime_error("expected BF16 tensor " + want.name +
                                 ", got " + info.dtype);
      if (info.shape != want.dims) {
        std::string got;
        for (size_t d : info.shape) got += " " + std::to_string(d);
        std::string expect;
        for (size_t d : want.dims) expect += " " + std::to_string(d);
        throw std::runtime_error("tensor " + want.name + " has shape [" + got +
                                 " ] but config.json implies [" + expect +
                                 " ]");
      }
    }
  }

  size_t kv_off(int layer, bool is_v, int slot, int n_kv_heads, int head_dim,
                size_t slot_cap) const {
    return ((((static_cast<size_t>(layer) * 2) + (is_v ? 1 : 0)) * slot_cap +
             slot) *
            n_kv_heads) *
           head_dim;
  }

  // Gather text ids, project to the talker width. Returns a stable device
  // pointer (the dedicated text_out buffer); do not alias fc2's input and
  // output.
  Bf16* project_text(const int* ids, int m) {
    UMM_CUDA_CHECK(cudaMemcpyAsync(d_ids, ids, m * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
    gather_rows_t<Bf16>(text_emb_w, d_ids, text_in, m, cfg.text_hidden, stream);
    lin(text_fc, text_in, text_fc1, m);
    silu_inplace_t<Bf16>(text_fc, m * cfg.text_hidden, stream);
    lin(text_out, text_fc, text_fc2, m);
    return text_out;
  }

  // Fused decoder layers with scalar position/history (prefill & code predictor).
  void run_layers(Bf16* x, int m, int history,
                  const std::vector<DecoderLayerWeights>& layers, Bf16* kv,
                  size_t slot_cap, int h, int inter, int n_q_heads,
                  int n_kv_heads, int head_dim, int mask, int window_len,
                  const LayerScratch& s) {
    const int qw = n_q_heads * head_dim;
    const int kw = n_kv_heads * head_dim;
    const int n_layers = static_cast<int>(layers.size());
    for (int l = 0; l < n_layers; ++l) {
      const DecoderLayerWeights& lw = layers[l];
      Bf16* k_cache_base = kv + kv_off(l, false, 0, n_kv_heads, head_dim, slot_cap);
      Bf16* v_cache_base = kv + kv_off(l, true, 0, n_kv_heads, head_dim, slot_cap);
      Bf16* v_cache_dst =
          kv + kv_off(l, true, history, n_kv_heads, head_dim, slot_cap);

      rms_norm_t<Bf16>(x, lw.input_ln, s.norm, m, h, cfg.rms_norm_eps, stream);
      // Fused Q/K/V projection writing V directly into v_cache[history..].
      QkvGemvDesc qkv_desc{{lw.q_proj, lw.k_proj, lw.v_proj},
                           {s.q, s.k, v_cache_dst},
                           {qw, kw, kw}};
      gemv_qkv_bf16(qkv_desc, s.norm, m, h, stream);
      // Fused Q-norm + K-norm + RoPE + K-cache append (V already in place).
      qk_norm_rope_kv_append_bf16(s.q, s.k, nullptr, lw.q_norm, lw.k_norm, d_inv,
                                  k_cache_base, nullptr, m, n_q_heads, n_kv_heads,
                                  head_dim, /*pos_base=*/history,
                                  /*history_base=*/history, cfg.rms_norm_eps,
                                  stream);
      gqa_attention_t<Bf16>(s.q, k_cache_base, v_cache_base, s.attn, m,
                            history + m, n_q_heads, n_kv_heads, head_dim, mask,
                            window_len, history, stream);
      // Fused O-proj + residual add directly into x.
      gemm_bf16_add_residual(x, s.attn, lw.o_proj, m, qw, h, stream);

      rms_norm_t<Bf16>(x, lw.post_ln, s.norm, m, h, cfg.rms_norm_eps, stream);
      // Fused SwiGLU (gate_proj + up_proj + silu*mul) into s.gate [m, inter].
      gemv_swiglu_bf16(lw.gate_proj, lw.up_proj, s.norm, s.gate, m, inter, h,
                       stream);
      // Fused down_proj + residual add directly into x.
      gemm_bf16_add_residual(x, s.gate, lw.down_proj, m, inter, h, stream);
    }
  }

  // Fixed-address single-token talker decode + final norm into cp_past.
  // Reads current position/history from *d_step_pos on device, making the
  // entire 28-layer step 100% capturable into a single CUDA Graph.
  void run_talker_decode_body() {
    const int h = cfg.hidden;
    const int inter = cfg.inter;
    const int n_q_heads = cfg.q_heads;
    const int n_kv_heads = cfg.kv_heads;
    const int head_dim = cfg.head_dim;
    const int qw = n_q_heads * head_dim;
    const int kw = n_kv_heads * head_dim;
    const int n_layers = static_cast<int>(talker_layers.size());
    for (int l = 0; l < n_layers; ++l) {
      const DecoderLayerWeights& lw = talker_layers[l];
      Bf16* k_cache_base =
          talker_kv + kv_off(l, false, 0, n_kv_heads, head_dim, kMaxSlots);
      Bf16* v_cache_base =
          talker_kv + kv_off(l, true, 0, n_kv_heads, head_dim, kMaxSlots);

      rms_norm_t<Bf16>(decode_x, lw.input_ln, tk.norm, 1, h, cfg.rms_norm_eps,
                       stream);
      QkvGemvDesc qkv_desc{{lw.q_proj, lw.k_proj, lw.v_proj},
                           {tk.q, tk.k, tk.v},
                           {qw, kw, kw}};
      gemv_qkv_bf16(qkv_desc, tk.norm, 1, h, stream);
      qk_norm_rope_kv_append_devpos_bf16(
          tk.q, tk.k, tk.v, lw.q_norm, lw.k_norm, d_inv, k_cache_base,
          v_cache_base, n_q_heads, n_kv_heads, head_dim, d_step_pos,
          cfg.rms_norm_eps, stream);
      gqa_attention_devpos_bf16(tk.q, k_cache_base, v_cache_base, tk.attn,
                                n_q_heads, n_kv_heads, head_dim, d_step_pos,
                                stream);
      gemm_bf16_add_residual(decode_x, tk.attn, lw.o_proj, 1, qw, h, stream);

      rms_norm_t<Bf16>(decode_x, lw.post_ln, tk.norm, 1, h, cfg.rms_norm_eps,
                       stream);
      gemv_swiglu_bf16(lw.gate_proj, lw.up_proj, tk.norm, tk.gate, 1, inter, h,
                       stream);
      gemm_bf16_add_residual(decode_x, tk.gate, lw.down_proj, 1, inter, h,
                             stream);
    }
    // Final RMSNorm directly into cp_past (reused by both select_c0 and CP).
    rms_norm_t<Bf16>(decode_x, talker_norm_w, cp_past, 1, h, cfg.rms_norm_eps,
                     stream);
  }

  void to_cp_width(Bf16* dst, const Bf16* src) {
    if (cfg.has_mtp_projection()) {
      lin(dst, src, mtp_proj, 1);
    } else {
      cudaMemcpyAsync(dst, src,
                      static_cast<size_t>(cfg.cp_hidden) * sizeof(Bf16),
                      cudaMemcpyDeviceToDevice, stream);
    }
  }

  // Fixed-address, fixed-shape 15-step Code Predictor (reads cp_past and
  // d_codes[0], writes d_codes[1..15]). 100% capturable into a single CUDA Graph.
  void run_cp_decode_body() {
    const int H = cfg.hidden;
    const int CPH = cfg.cp_hidden, CPI = cfg.cp_inter;
    const int CPQH = cfg.cp_q_heads, CPKH = cfg.cp_kv_heads,
              CPHD = cfg.cp_head_dim;
    const int groups = cfg.code_groups;

    to_cp_width(cp_x, cp_past);
    gather_rows_t<Bf16>(talker_codec_emb_w, d_codes, proj, 1, H, stream);
    to_cp_width(cp_x + CPH, proj);
    run_layers(cp_x, 2, 0, cp_layers, cp_kv, kCpSlots, CPH, CPI, CPQH, CPKH,
               CPHD, /*mask=*/0, /*window_len=*/0, cp);
    rms_norm_t<Bf16>(cp_x + CPH, cp_norm_w, cp_norm, 1, CPH, cfg.rms_norm_eps,
                     stream);
    gemv_t<Bf16>(cp_lm_head_w[0], cp_norm, logits, cfg.cp_vocab, CPH, stream);
    argmax_f32_dev(logits, cfg.cp_vocab, d_codes + 1, stream);

    int cp_len = 2;
    for (int j = 1; j < groups - 1; ++j) {
      gather_rows_t<Bf16>(cp_codec_emb_w[j - 1], d_codes + j, proj, 1, H,
                          stream);
      to_cp_width(cp_x, proj);
      run_layers(cp_x, 1, cp_len, cp_layers, cp_kv, kCpSlots, CPH, CPI, CPQH,
                 CPKH, CPHD, /*mask=*/0, /*window_len=*/0, cp);
      ++cp_len;
      rms_norm_t<Bf16>(cp_x, cp_norm_w, cp_norm, 1, CPH, cfg.rms_norm_eps,
                       stream);
      gemv_t<Bf16>(cp_lm_head_w[j], cp_norm, logits, cfg.cp_vocab, CPH, stream);
      argmax_f32_dev(logits, cfg.cp_vocab, d_codes + j + 1, stream);
    }
  }

  template <typename Fn>
  cudaGraphExec_t try_capture_graph(Fn&& body) {
    cudaGraph_t graph = nullptr;
    if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) !=
        cudaSuccess) {
      return nullptr;
    }
    body();
    if (cudaStreamEndCapture(stream, &graph) != cudaSuccess ||
        graph == nullptr) {
      return nullptr;
    }
    cudaGraphExec_t exec = nullptr;
#if CUDART_VERSION >= 12000
    const cudaError_t err = cudaGraphInstantiate(&exec, graph, 0);
#else
    const cudaError_t err =
        cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
#endif
    cudaGraphDestroy(graph);
    if (err != cudaSuccess) return nullptr;
    return exec;
  }
};

TtsEngine::TtsEngine(const std::string& model_dir, int gpu_id) {
  impl_ = std::make_unique<Impl>();
  Impl& im = *impl_;
  im.gpu_id = gpu_id;
  im.cfg = TtsModelConfig::from_model_dir(model_dir);
  const TtsModelConfig& cfg = im.cfg;
  if (cfg.language_id("chinese") != kLangChinese ||
      cfg.language_id("english") != kLangEnglish)
    throw std::runtime_error(
        "config.json language ids differ from the wired prompt");
  if (!cfg.is_base() && (cfg.speaker_id("vivian") != kSpkVivian ||
                         cfg.speaker_id("ryan") != kSpkRyan))
    throw std::runtime_error(
        "config.json speaker ids differ from the wired prompt");
  if (cfg.tts_pad_token_id != kTtsPad || cfg.tts_bos_token_id != kTtsBos ||
      cfg.tts_eos_token_id != kTtsEos)
    throw std::runtime_error(
        "config.json tts special ids differ from the wired prompt");
  if (cfg.codec_pad_id != kCodecPad || cfg.codec_bos_id != kCodecBos ||
      cfg.codec_eos_id != kCodecEos || cfg.codec_think_id != kCodecThink ||
      cfg.codec_think_bos_id != kCodecThinkBos ||
      cfg.codec_think_eos_id != kCodecThinkEos)
    throw std::runtime_error(
        "config.json codec control ids differ from the wired prompt");

  UMM_CUDA_CHECK(cudaSetDevice(gpu_id));
  UMM_CUDA_CHECK(cudaStreamCreate(&im.stream));

  const int H = cfg.hidden, INTER = cfg.inter, QW = cfg.q_width();
  const int KVW = cfg.kv_width(), HD = cfg.head_dim;
  const int CPH = cfg.cp_hidden, CPI = cfg.cp_inter;
  const int CPQW = cfg.cp_q_width(), CPKVW = cfg.cp_kv_width();

  // Talker BF16 region.
  {
    SafeTensors st(read_file(model_dir + "/model.safetensors"));
    im.validate_shapes(st);
    if (cfg.is_base())
      im.speaker_encoder = std::make_unique<SpeakerEncoder>(st, cfg);
    const size_t start = st.data_region_offset();
    const size_t bytes = st.bytes().size() - start;
    UMM_CUDA_CHECK(request_cuda_malloc(&im.d_bf, bytes));
    UMM_CUDA_CHECK(cudaMemcpy(im.d_bf, st.bytes().data() + start, bytes,
                              cudaMemcpyHostToDevice));
    for (const auto& [name, info] : st.tensors()) {
      if (info.dtype != "BF16")
        throw std::runtime_error("expected BF16 talker tensor: " + name);
      im.bw.emplace(name, im.d_bf + info.byte_offset / 2);
    }
  }
  // Codec F32 region (only decoder.* is ever dereferenced).
  {
    SafeTensors st(read_file(model_dir + "/speech_tokenizer/model.safetensors"));
    const size_t start = st.data_region_offset();
    const size_t bytes = st.bytes().size() - start;
    UMM_CUDA_CHECK(request_cuda_malloc(&im.d_f32, bytes));
    UMM_CUDA_CHECK(cudaMemcpy(im.d_f32, st.bytes().data() + start, bytes,
                              cudaMemcpyHostToDevice));
    for (const auto& [name, info] : st.tensors()) {
      if (info.dtype != "F32") continue;  // skip any non-F32 entry
      im.fw.emplace(name, im.d_f32 + info.byte_offset / 4);
    }
  }
  im.codec = std::make_unique<CodecDecoder>(im.fw, im.stream);

  // Pre-resolve all Talker and Code Predictor weight pointers once.
  auto resolve_layers = [&](const std::string& base, int count) {
    std::vector<DecoderLayerWeights> out(count);
    for (int l = 0; l < count; ++l) {
      const std::string p = base + ".layers." + std::to_string(l);
      DecoderLayerWeights& lw = out[l];
      lw.input_ln = im.wb(p + ".input_layernorm.weight");
      lw.q_proj = im.wb(p + ".self_attn.q_proj.weight");
      lw.k_proj = im.wb(p + ".self_attn.k_proj.weight");
      lw.v_proj = im.wb(p + ".self_attn.v_proj.weight");
      lw.q_norm = im.wb(p + ".self_attn.q_norm.weight");
      lw.k_norm = im.wb(p + ".self_attn.k_norm.weight");
      lw.o_proj = im.wb(p + ".self_attn.o_proj.weight");
      lw.post_ln = im.wb(p + ".post_attention_layernorm.weight");
      lw.gate_proj = im.wb(p + ".mlp.gate_proj.weight");
      lw.up_proj = im.wb(p + ".mlp.up_proj.weight");
      lw.down_proj = im.wb(p + ".mlp.down_proj.weight");
    }
    return out;
  };
  im.talker_layers = resolve_layers("talker.model", cfg.layers);
  im.cp_layers = resolve_layers("talker.code_predictor.model", cfg.cp_layers);
  im.text_emb_w = im.wb("talker.model.text_embedding.weight");
  im.text_fc1 = im.dense("talker.text_projection.linear_fc1", cfg.text_hidden,
                         cfg.text_hidden, true);
  im.text_fc2 =
      im.dense("talker.text_projection.linear_fc2", cfg.text_hidden, H, true);
  im.talker_codec_emb_w = im.wb("talker.model.codec_embedding.weight");
  im.talker_norm_w = im.wb("talker.model.norm.weight");
  im.talker_codec_head_w = im.wb("talker.codec_head.weight");
  im.cp_norm_w = im.wb("talker.code_predictor.model.norm.weight");
  if (cfg.has_mtp_projection()) {
    im.mtp_proj = im.dense("talker.code_predictor.small_to_mtp_projection", H,
                           CPH, true);
  }
  im.cp_codec_emb_w.resize(cfg.code_groups - 1);
  im.cp_lm_head_w.resize(cfg.code_groups - 1);
  std::vector<const Bf16*> host_frame_tables(cfg.code_groups);
  host_frame_tables[0] = im.talker_codec_emb_w;
  for (int j = 0; j < cfg.code_groups - 1; ++j) {
    const std::string s = std::to_string(j);
    im.cp_codec_emb_w[j] =
        im.wb("talker.code_predictor.model.codec_embedding." + s + ".weight");
    im.cp_lm_head_w[j] = im.wb("talker.code_predictor.lm_head." + s + ".weight");
    host_frame_tables[j + 1] = im.cp_codec_emb_w[j];
  }
  UMM_CUDA_CHECK(
      request_cuda_malloc(&im.d_frame_emb_tables, cfg.code_groups * sizeof(const Bf16*)));
  UMM_CUDA_CHECK(cudaMemcpy(im.d_frame_emb_tables, host_frame_tables.data(),
                            cfg.code_groups * sizeof(const Bf16*),
                            cudaMemcpyHostToDevice));

  // Workspaces (prefill scratch sized for kMaxPrefillRows; decode uses 1-row
  // im.decode_x).
  auto mk = [&](Bf16*& p, size_t n) {
    UMM_CUDA_CHECK(request_cuda_malloc(&p, n * sizeof(Bf16)));
  };
  const size_t W = kMaxPrefillRows;
  mk(im.hidden, W * H);
  mk(im.norm, W * H);
  mk(im.q, W * QW);
  mk(im.k, W * KVW);
  mk(im.v, W * KVW);
  mk(im.attn, W * QW);
  mk(im.gate, W * INTER);
  mk(im.up, W * INTER);
  mk(im.proj, W * H);
  mk(im.text_in, W * cfg.text_hidden);
  mk(im.text_fc, W * cfg.text_hidden);
  mk(im.text_out, W * H);
  mk(im.decode_x, H);
  im.tk = {im.norm, im.q, im.k, im.v, im.attn, im.gate, im.up};

  const size_t CW = kCpRows;
  mk(im.cp_x, CW * CPH);
  mk(im.cp_norm, CW * CPH);
  mk(im.cp_q, CW * CPQW);
  mk(im.cp_k, CW * CPKVW);
  mk(im.cp_v, CW * CPKVW);
  mk(im.cp_attn, CW * CPQW);
  mk(im.cp_gate, CW * CPI);
  mk(im.cp_up, CW * CPI);
  im.cp = {im.cp_norm, im.cp_q,    im.cp_k,    im.cp_v,
           im.cp_attn, im.cp_gate, im.cp_up};

  UMM_CUDA_CHECK(request_cuda_malloc(&im.logits, cfg.codec_vocab * sizeof(float)));
  UMM_CUDA_CHECK(request_cuda_malloc(&im.d_ids, W * sizeof(int)));
  UMM_CUDA_CHECK(request_cuda_malloc(&im.d_pos, W * sizeof(int)));
  UMM_CUDA_CHECK(request_cuda_malloc(&im.d_codes, cfg.code_groups * sizeof(int)));
  UMM_CUDA_CHECK(request_cuda_malloc(&im.d_hist, kMaxSlots * sizeof(int)));
  UMM_CUDA_CHECK(request_cuda_malloc(&im.d_step_pos, sizeof(int)));
  UMM_CUDA_CHECK(
      cudaMallocHost(&im.h_codes_pinned, cfg.code_groups * sizeof(int)));
  UMM_CUDA_CHECK(request_cuda_malloc(&im.d_inv, (HD / 2) * sizeof(float)));
  UMM_CUDA_CHECK(request_cuda_malloc(&im.d_pad, H * sizeof(Bf16)));
  UMM_CUDA_CHECK(request_cuda_malloc(&im.d_bos, H * sizeof(Bf16)));
  UMM_CUDA_CHECK(request_cuda_malloc(&im.d_eos, H * sizeof(Bf16)));
  if (cfg.is_base())
    UMM_CUDA_CHECK(request_cuda_malloc(&im.d_xvector, H * sizeof(Bf16)));
  UMM_CUDA_CHECK(request_cuda_malloc(&im.cp_past, H * sizeof(Bf16)));
  UMM_CUDA_CHECK(request_cuda_malloc(&im.talker_kv,
                            sizeof(Bf16) * cfg.layers * 2 * kMaxSlots * KVW));
  UMM_CUDA_CHECK(
      request_cuda_malloc(&im.cp_kv, sizeof(Bf16) * cfg.cp_layers * 2 * kCpSlots * CPKVW));

  std::vector<float> inv(HD / 2);
  for (int i = 0; i < HD / 2; ++i)
    inv[i] = 1.0f / std::pow(cfg.rope_theta, 2.0f * i / HD);
  UMM_CUDA_CHECK(cudaMemcpy(im.d_inv, inv.data(), inv.size() * sizeof(float),
                            cudaMemcpyHostToDevice));

  // Precompute tts_pad/bos/eos vectors: text embed -> text_projection MLP.
  const int tts_ids[3] = {cfg.tts_pad_token_id, cfg.tts_bos_token_id,
                          cfg.tts_eos_token_id};
  UMM_CUDA_CHECK(
      cudaMemcpy(im.d_ids, tts_ids, sizeof(tts_ids), cudaMemcpyHostToDevice));
  gather_rows_t<Bf16>(im.text_emb_w, im.d_ids, im.text_in, 3, cfg.text_hidden,
                      im.stream);
  if (const char* dump = std::getenv("UMM_TTS_DUMP_GATHER")) {
    std::vector<Bf16> hb(3 * cfg.text_hidden);
    cudaMemcpy(hb.data(), im.text_in, hb.size() * sizeof(Bf16),
               cudaMemcpyDeviceToHost);
    std::vector<float> hf(hb.size());
    for (size_t i = 0; i < hb.size(); ++i) hf[i] = __bfloat162float(hb[i]);
    std::ofstream f(dump, std::ios::binary);
    f.write(reinterpret_cast<const char*>(hf.data()), hf.size() * sizeof(float));
  }
  im.lin(im.text_fc, im.text_in, im.text_fc1, 3);
  if (const char* dump = std::getenv("UMM_TTS_DUMP_FC1")) {
    std::vector<Bf16> hb(3 * cfg.text_hidden);
    cudaMemcpy(hb.data(), im.text_fc, hb.size() * sizeof(Bf16),
               cudaMemcpyDeviceToHost);
    std::vector<float> hf(hb.size());
    for (size_t i = 0; i < hb.size(); ++i) hf[i] = __bfloat162float(hb[i]);
    std::ofstream f(dump, std::ios::binary);
    f.write(reinterpret_cast<const char*>(hf.data()), hf.size() * sizeof(float));
  }
  silu_inplace_t<Bf16>(im.text_fc, 3 * cfg.text_hidden, im.stream);
  if (const char* dump = std::getenv("UMM_TTS_DUMP_SILU")) {
    std::vector<Bf16> hb(3 * cfg.text_hidden);
    cudaMemcpy(hb.data(), im.text_fc, hb.size() * sizeof(Bf16),
               cudaMemcpyDeviceToHost);
    std::vector<float> hf(hb.size());
    for (size_t i = 0; i < hb.size(); ++i) hf[i] = __bfloat162float(hb[i]);
    std::ofstream f(dump, std::ios::binary);
    f.write(reinterpret_cast<const char*>(hf.data()), hf.size() * sizeof(float));
  }
  im.lin(im.hidden, im.text_fc, im.text_fc2, 3);
  UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_pad, im.hidden, H * sizeof(Bf16),
                                 cudaMemcpyDeviceToDevice, im.stream));
  UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_bos, im.hidden + H, H * sizeof(Bf16),
                                 cudaMemcpyDeviceToDevice, im.stream));
  UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_eos, im.hidden + 2 * H, H * sizeof(Bf16),
                                 cudaMemcpyDeviceToDevice, im.stream));
  UMM_CUDA_CHECK(
      cudaMemsetAsync(im.d_codes, 0, cfg.code_groups * sizeof(int), im.stream));
  UMM_CUDA_CHECK(cudaMemsetAsync(im.d_step_pos, 0, sizeof(int), im.stream));
  UMM_CUDA_CHECK(cudaStreamSynchronize(im.stream));

  if (const char* dump = std::getenv("UMM_TTS_DUMP_PAD")) {
    std::vector<Bf16> hb(3 * H);
    cudaMemcpy(hb.data(), im.hidden, hb.size() * sizeof(Bf16),
               cudaMemcpyDeviceToHost);
    std::vector<float> hf(hb.size());
    for (size_t i = 0; i < hb.size(); ++i) hf[i] = __bfloat162float(hb[i]);
    std::ofstream f(dump, std::ios::binary);
    f.write(reinterpret_cast<const char*>(hf.data()), hf.size() * sizeof(float));
  }

  // Capture CUDA graphs for the 15-step Code Predictor and 28-layer Talker
  // decode step. Both use fixed device addresses and zero host copies.
  if (!env_flag_enabled("UMM_TTS_NO_GRAPH")) {
    im.cp_graph_exec = im.try_capture_graph([&]() { im.run_cp_decode_body(); });
    im.talker_graph_exec =
        im.try_capture_graph([&]() { im.run_talker_decode_body(); });
  }
}

TtsEngine::~TtsEngine() {
  if (!impl_) return;
  Impl& im = *impl_;
  if (im.talker_graph_exec) cudaGraphExecDestroy(im.talker_graph_exec);
  if (im.cp_graph_exec) cudaGraphExecDestroy(im.cp_graph_exec);
  if (im.h_codes_pinned) cudaFreeHost(im.h_codes_pinned);
  request_cuda_free(im.d_step_pos);
  request_cuda_free(im.d_frame_emb_tables);
  request_cuda_free(im.decode_x);
  request_cuda_free(im.cp_up);
  request_cuda_free(im.cp_gate);
  request_cuda_free(im.cp_attn);
  request_cuda_free(im.cp_v);
  request_cuda_free(im.cp_k);
  request_cuda_free(im.cp_q);
  request_cuda_free(im.cp_norm);
  request_cuda_free(im.cp_x);
  request_cuda_free(im.cp_kv);
  request_cuda_free(im.talker_kv);
  request_cuda_free(im.d_eos);
  request_cuda_free(im.d_bos);
  request_cuda_free(im.d_pad);
  request_cuda_free(im.d_xvector);
  request_cuda_free(im.cp_past);
  request_cuda_free(im.d_inv);
  request_cuda_free(im.d_codes);
  request_cuda_free(im.d_hist);
  request_cuda_free(im.d_pos);
  request_cuda_free(im.d_ids);
  request_cuda_free(im.logits);
  request_cuda_free(im.text_out);
  request_cuda_free(im.text_fc);
  request_cuda_free(im.text_in);
  request_cuda_free(im.proj);
  request_cuda_free(im.up);
  request_cuda_free(im.gate);
  request_cuda_free(im.attn);
  request_cuda_free(im.v);
  request_cuda_free(im.k);
  request_cuda_free(im.q);
  request_cuda_free(im.norm);
  request_cuda_free(im.hidden);
  request_cuda_free(im.d_f32);
  request_cuda_free(im.d_bf);
  if (im.stream) cudaStreamDestroy(im.stream);
}

bool TtsEngine::is_base() const { return impl_->cfg.is_base(); }

std::vector<float> TtsEngine::extract_speaker_embedding(
    const std::vector<float>& ref_pcm_24k) {
  if (!impl_->cfg.is_base())
    throw std::invalid_argument("speaker extraction requires a Base model");
  return impl_->speaker_encoder->extract(ref_pcm_24k);
}

std::vector<float> TtsEngine::synthesize(const std::vector<int>& text_ids,
                                         int language_id, int speaker_id,
                                         int max_frames,
                                         const AbortCheck& abort,
                                         SynthTimings* timings,
                                         const StreamSink& on_chunk) {
  if (impl_->cfg.is_base())
    throw std::invalid_argument(
        "Base models require synthesize_clone with a speaker embedding");
  return synthesize_impl(text_ids, language_id, speaker_id, nullptr, max_frames,
                         abort, timings, on_chunk);
}

std::vector<float> TtsEngine::synthesize_clone(
    const std::vector<int>& text_ids, int language_id,
    const std::vector<float>& speaker_embedding, int max_frames,
    const AbortCheck& abort, SynthTimings* timings,
    const StreamSink& on_chunk) {
  if (!impl_->cfg.is_base())
    throw std::invalid_argument("voice cloning requires a Base model");
  if (speaker_embedding.size() != static_cast<size_t>(impl_->cfg.hidden))
    throw std::invalid_argument(
        "speaker embedding width does not match the talker");
  for (float v : speaker_embedding)
    if (!std::isfinite(v))
      throw std::invalid_argument("speaker embedding contains non-finite data");
  return synthesize_impl(text_ids, language_id, -1, &speaker_embedding,
                         max_frames, abort, timings, on_chunk);
}

std::vector<float> TtsEngine::synthesize_impl(
    const std::vector<int>& text_ids, int language_id, int speaker_id,
    const std::vector<float>* speaker_embedding, int max_frames,
    const AbortCheck& abort, SynthTimings* timings,
    const StreamSink& on_chunk) {
  Impl& im = *impl_;
  const TtsModelConfig& cfg = im.cfg;
  const int H = cfg.hidden, INTER = cfg.inter, HD = cfg.head_dim;
  const int QH = cfg.q_heads, KH = cfg.kv_heads;
  const int groups = cfg.code_groups;
  const auto started = std::chrono::steady_clock::now();
  const auto elapsed_ms = [&started] {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - started)
        .count();
  };
  if (speaker_embedding != nullptr) {
    std::vector<Bf16> host(speaker_embedding->size());
    for (size_t i = 0; i < host.size(); ++i)
      host[i] = __float2bfloat16((*speaker_embedding)[i]);
    UMM_CUDA_CHECK(cudaMemcpy(im.d_xvector, host.data(), H * sizeof(Bf16),
                              cudaMemcpyHostToDevice));
  }
  const PromptPlan plan =
      speaker_embedding != nullptr
          ? build_clone_prompt_plan(text_ids, language_id)
          : build_prompt_plan(text_ids, language_id, speaker_id);
  const int L = static_cast<int>(plan.rows.size());
  if (L > 200) throw std::runtime_error("prompt too long");

  // Build prefill embeddings row by row into im.hidden [L, H].
  for (int r = 0; r < L; ++r) {
    Bf16* dst = im.hidden + static_cast<size_t>(r) * H;
    bool first = true;
    for (const PromptTerm& term : plan.rows[r]) {
      const Bf16* src = nullptr;
      if (term.kind == PromptTerm::kTextProj) {
        src = im.project_text(&term.id, 1);
      } else if (term.kind == PromptTerm::kCodec) {
        UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_ids, &term.id, sizeof(int),
                                       cudaMemcpyHostToDevice, im.stream));
        gather_rows_t<Bf16>(im.talker_codec_emb_w, im.d_ids, im.proj, 1, H,
                            im.stream);
        src = im.proj;
      } else if (term.kind == PromptTerm::kXVector) {
        src = im.d_xvector;
      } else {
        src = term.kind == PromptTerm::kTtsPad  ? im.d_pad
              : term.kind == PromptTerm::kTtsBos ? im.d_bos
                                                 : im.d_eos;
      }
      if (first) {
        cudaMemcpyAsync(dst, src, H * sizeof(Bf16), cudaMemcpyDeviceToDevice,
                        im.stream);
        first = false;
      } else {
        add_residual_t<Bf16>(dst, src, H, im.stream);
      }
    }
    if (r < 3) {
      UMM_CUDA_CHECK(cudaStreamSynchronize(im.stream));
    }
  }

  if (const char* dump = std::getenv("UMM_TTS_DUMP")) {
    std::vector<Bf16> hb(static_cast<size_t>(L) * H);
    UMM_CUDA_CHECK(cudaMemcpy(hb.data(), im.hidden, hb.size() * sizeof(Bf16),
                              cudaMemcpyDeviceToHost));
    std::vector<float> hf(hb.size());
    for (size_t i = 0; i < hb.size(); ++i) hf[i] = __bfloat162float(hb[i]);
    std::ofstream f(dump, std::ios::binary);
    f.write(reinterpret_cast<const char*>(hf.data()), hf.size() * sizeof(float));
  }

  // Talker prompt prefill + final norm of the last prompt row into im.cp_past.
  im.run_layers(im.hidden, L, 0, im.talker_layers, im.talker_kv, kMaxSlots, H,
                INTER, QH, KH, HD, /*mask=*/0, /*window_len=*/0, im.tk);
  rms_norm_t<Bf16>(im.hidden + static_cast<size_t>(L - 1) * H, im.talker_norm_w,
                   im.cp_past, 1, H, cfg.rms_norm_eps, im.stream);
  int seq_len = L;

  constexpr float kRepPenalty = 1.05f;
  int kRunBan = 4;
  if (const char* e = std::getenv("UMM_TTS_RUN_BAN")) kRunBan = std::stoi(e);
  const char* dump_logits = std::getenv("UMM_TTS_DUMP_LOGITS");

  // Reads final-normed talker state from im.cp_past (computed by prefill or by
  // the end of the graphed talker decode step) and writes winning c0 to
  // im.d_codes[0].
  auto select_c0 = [&](bool allow_eos, int n_hist, int ban_id) {
    gemv_t<Bf16>(im.talker_codec_head_w, im.cp_past, im.logits, cfg.codec_vocab,
                 H, im.stream);
    repetition_penalty_f32(im.logits, im.d_hist, n_hist, kRepPenalty,
                           im.stream);
    suppress_logits(im.logits, cfg.cp_vocab, cfg.codec_vocab,
                    allow_eos ? cfg.codec_eos_id : -1, im.stream);
    if (ban_id >= 0) ban_logit_f32(im.logits, ban_id, im.stream);
    if (dump_logits != nullptr) {
      std::vector<float> h(cfg.codec_vocab);
      cudaMemcpy(h.data(), im.logits, h.size() * sizeof(float),
                 cudaMemcpyDeviceToHost);
      std::ofstream f(std::string(dump_logits) + "." + std::to_string(n_hist),
                      std::ios::binary);
      f.write(reinterpret_cast<const char*>(h.data()), h.size() * sizeof(float));
    }
    argmax_f32_dev(im.logits, cfg.codec_vocab, im.d_codes, im.stream);
  };

  auto run_predictor = [&]() {
    if (im.cp_graph_exec != nullptr) {
      UMM_CUDA_CHECK(cudaGraphLaunch(im.cp_graph_exec, im.stream));
    } else {
      im.run_cp_decode_body();
    }
  };

  auto run_talker_step = [&]() {
    if (im.talker_graph_exec != nullptr) {
      UMM_CUDA_CHECK(cudaGraphLaunch(im.talker_graph_exec, im.stream));
    } else {
      im.run_talker_decode_body();
    }
  };

  auto fetch_rest = [&]() {
    UMM_CUDA_CHECK(cudaMemcpyAsync(im.h_codes_pinned + 1, im.d_codes + 1,
                                   (groups - 1) * sizeof(int),
                                   cudaMemcpyDeviceToHost, im.stream));
    UMM_CUDA_CHECK(cudaStreamSynchronize(im.stream));
  };
  auto fetch_c0 = [&]() {
    UMM_CUDA_CHECK(cudaMemcpyAsync(im.h_codes_pinned, im.d_codes, sizeof(int),
                                   cudaMemcpyDeviceToHost, im.stream));
    UMM_CUDA_CHECK(cudaStreamSynchronize(im.stream));
  };

  select_c0(/*allow_eos=*/false, /*n_hist=*/0, /*ban_id=*/-1);
  fetch_c0();
  int c0 = im.h_codes_pinned[0];
  std::vector<int> frames;
  std::vector<int> c0_trace;

  constexpr int kSamplesPerFrame = 1920;
  constexpr int kSampleRate = 24000;
  constexpr int kFirstChunkFrames = 2;
  constexpr int kChunkFrames = 8;
  std::vector<float> samples;
  int decoded_frames = 0;
  int next_stream_frame = kFirstChunkFrames;
  if (on_chunk) im.codec->reset_stream();

  constexpr int kTrimGuardFrames = 13;
  constexpr size_t kTrimGuardSamples = kTrimGuardFrames * kSamplesPerFrame;
  std::vector<float> held;
  int held_frame = 0;
  bool emitted_any = false;
  auto emit = [&](const std::vector<float>& chunk, int first_frame,
                  int n_frames) {
    if (timings != nullptr && timings->first_audio_ms < 0)
      timings->first_audio_ms = elapsed_ms();
    on_chunk(chunk, first_frame, n_frames);
  };
  // Appends incrementally decoded samples for [decoded_frames, frame_count).
  auto take_chunk = [&](std::vector<float>&& chunk, int frame_count) {
    if (chunk.empty()) return;
    const int first_frame = decoded_frames;
    samples.insert(samples.end(), chunk.begin(), chunk.end());
    decoded_frames = frame_count;
    if (!emitted_any) {
      emit(chunk, first_frame, frame_count - first_frame);
      emitted_any = true;
      held_frame = frame_count;
      return;
    }
    held.insert(held.end(), chunk.begin(), chunk.end());
    const size_t keep = std::min(held.size(), kTrimGuardSamples);
    const size_t release = held.size() - keep;
    if (release > 0) {
      std::vector<float> out(held.begin(),
                             held.begin() + static_cast<long>(release));
      emit(out, held_frame, static_cast<int>(release / kSamplesPerFrame));
      held.erase(held.begin(), held.begin() + static_cast<long>(release));
      held_frame += static_cast<int>(release / kSamplesPerFrame);
    }
  };
  auto finish_stream = [&]() {
    const size_t cut =
        std::min(trailing_noise_cut(samples, kSampleRate), held.size());
    if (cut > 0) {
      held.resize(held.size() - cut);
      samples.resize(samples.size() - cut);
      fade_out(held, static_cast<size_t>(kSampleRate) * 5 / 1000);
      std::copy(held.begin(), held.end(), samples.end() - held.size());
    }
    if (!held.empty())
      emit(held, held_frame,
           static_cast<int>(held.size() / kSamplesPerFrame));
    held.clear();
  };

  c0_trace.push_back(c0);
  for (int step = 0; step < max_frames; ++step) {
    if (abort && abort()) {
      throw RequestAborted("aborted at frame " + std::to_string(step));
    }
    if (c0 < 0 || c0 >= cfg.codec_vocab)
      throw std::runtime_error("talker produced out-of-range codec id");
    if (seq_len + 1 > kMaxSlots)
      throw std::runtime_error("tts text too long for the talker buffers");

    // 1. Run all 15 Code Predictor steps (single CUDA Graph launch).
    run_predictor();
    fetch_rest();

    frames.push_back(c0);
    if (step == 0 && timings != nullptr) timings->first_frame_ms = elapsed_ms();
    for (int j = 0; j < groups - 1; ++j)
      frames.push_back(im.h_codes_pinned[1 + j]);
    const int frame_count = static_cast<int>(frames.size() / groups);
    if (on_chunk && frame_count >= next_stream_frame) {
      const int new_frames = frame_count - decoded_frames;
      take_chunk(im.codec->decode_chunk(
                     frames.data() + static_cast<size_t>(decoded_frames) * groups,
                     new_frames, abort),
                 frame_count);
      next_stream_frame = frame_count + kChunkFrames;
    }

    // 2. Single fused kernel: sum 16 codebook embeddings + d_pad into
    //    im.decode_x, record d_hist[step] = d_codes[0], and set *d_step_pos =
    //    seq_len on device (zero H2D staging copies).
    sum_frame_embeddings_bf16(im.d_frame_emb_tables, im.d_codes, im.d_pad,
                              im.decode_x, im.d_hist + step, im.d_step_pos,
                              seq_len, groups, H, im.stream);

    // 3. Run all 28 Talker layers + final norm into im.cp_past (single Graph).
    run_talker_step();
    ++seq_len;

    // 4. Codec head + repetition penalty + argmax -> d_codes[0].
    const bool allow_eos = (step + 1) >= 2;
    int ban_id = -1;
    const int ntr = static_cast<int>(c0_trace.size());
    if (ntr >= kRunBan) {
      bool run = true;
      for (int z = ntr - kRunBan; z < ntr; ++z)
        run = run && c0_trace[z] == c0_trace.back();
      if (run) ban_id = c0_trace.back();
    }
    select_c0(allow_eos, step + 1, ban_id);
    fetch_c0();
    const int next = im.h_codes_pinned[0];
    if (next == cfg.codec_eos_id) break;
    c0 = next;
    c0_trace.push_back(c0);
  }
  if (const char* dump = std::getenv("UMM_TTS_DUMP_C0")) {
    std::ofstream f(dump, std::ios::binary);
    f.write(reinterpret_cast<const char*>(c0_trace.data()),
            c0_trace.size() * sizeof(int));
  }
  if (const char* dump = std::getenv("UMM_TTS_DUMP_FRAMES")) {
    std::ofstream f(dump, std::ios::binary);
    f.write(reinterpret_cast<const char*>(frames.data()),
            static_cast<std::streamsize>(frames.size() * sizeof(int)));
  }
  UMM_CUDA_CHECK(cudaStreamSynchronize(im.stream));
  if (abort && abort()) {
    throw RequestAborted("aborted before codec decode at frame " +
                         std::to_string(frames.size() / groups));
  }
  if (frames.size() % groups != 0)
    throw std::runtime_error(
        "codec frames not divisible by the code group count");
  const int total_frames = static_cast<int>(frames.size() / groups);
  if (!on_chunk) {
    samples = trim_trailing_noise(im.codec->decode(frames, total_frames, abort),
                                  kSampleRate);
  } else {
    if (decoded_frames < total_frames) {
      const int rem = total_frames - decoded_frames;
      take_chunk(im.codec->decode_chunk(
                     frames.data() + static_cast<size_t>(decoded_frames) * groups,
                     rem, abort),
                 total_frames);
    }
    finish_stream();
  }
  if (on_chunk && std::getenv("UMM_TTS_REPEAT_CHECK")) {
    std::vector<float> again = im.codec->decode(frames, total_frames, abort);
    size_t fd = 0;
    while (fd < again.size() && fd < samples.size() && again[fd] == samples[fd])
      ++fd;
    float md = 0.0f;
    for (size_t i = 0; i < std::min(again.size(), samples.size()); ++i)
      md = std::max(md, std::fabs(again[i] - samples[i]));
    std::cerr << "tts repeat_check same_size="
              << (again.size() == samples.size()) << " first_diff=" << fd
              << " max_abs_diff=" << md << "\n";
  }
  if (timings != nullptr) {
    timings->total_ms = elapsed_ms();
    if (std::getenv("UMM_TTS_TIMINGS"))
      std::cerr << "tts timings first_frame_ms=" << timings->first_frame_ms
                << " first_audio_ms=" << timings->first_audio_ms
                << " total_ms=" << timings->total_ms << "\n";
  }
  return samples;
}

}  // namespace umminfer::native

// native/stage_cli enumerates engine.cpp directly; include this implementation
// here so both CUDA executables link without changing the Stage source list.
#include "speaker_encoder.cpp"
