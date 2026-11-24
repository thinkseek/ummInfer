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

#include "codec.hpp"
#include "cuda_kernels.hpp"
#include "model_config.hpp"
#include "prompt.hpp"
#include "safetensors.hpp"
#include "speaker_encoder.hpp"
#include "tail_trim.hpp"

namespace umminfer::native {
namespace {

// Operational capacity, independent of the checkpoint size: the decode loop
// writes one talker row per codec frame, so the KV cache and the activation
// buffers are sized for the longest utterance the CLI accepts.
constexpr int kMaxSlots = 8704;       // 8192 frames + prompt headroom
constexpr int kTalkerRows = kMaxSlots + 256;
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

// Per-width scratch a decoder layer reads and writes. The talker and the code
// predictor run the same layer topology at different widths, so each keeps its
// own set; the buffers are small next to the weights.
struct LayerScratch {
  Bf16 *norm = nullptr, *q = nullptr, *k = nullptr, *v = nullptr,
      *attn = nullptr, *gate = nullptr, *up = nullptr;
};

}  // namespace

struct TtsEngine::Impl {
  TtsModelConfig cfg;
  cudaStream_t stream = nullptr;
  int gpu_id = 0;

  std::unordered_map<std::string, Bf16*> bw;   // talker BF16
  Bf16* d_bf = nullptr;
  std::unordered_map<std::string, const float*> fw;  // codec F32
  float* d_f32 = nullptr;
  std::unique_ptr<CodecDecoder> codec;
  std::unique_ptr<SpeakerEncoder> speaker_encoder;

  // Talker-width scratch.
  Bf16 *hidden = nullptr, *norm = nullptr, *q = nullptr, *k = nullptr, *v = nullptr,
      *attn = nullptr, *gate = nullptr, *up = nullptr, *proj = nullptr,
      *text_in = nullptr, *text_fc = nullptr, *text_out = nullptr;
  LayerScratch tk;
  // Code-predictor-width scratch.
  Bf16 *cp_x = nullptr, *cp_norm = nullptr, *cp_q = nullptr, *cp_k = nullptr,
      *cp_v = nullptr, *cp_attn = nullptr, *cp_gate = nullptr, *cp_up = nullptr;
  LayerScratch cp;
  // Prompt lookups are neither width: indices, positions, emitted history.
  int *d_ids = nullptr, *d_pos = nullptr, *d_codes = nullptr, *d_hist = nullptr;
  float* logits = nullptr;
  float* d_inv = nullptr;
  Bf16* talker_kv = nullptr;
  Bf16* cp_kv = nullptr;
  Bf16 *d_pad = nullptr, *d_bos = nullptr, *d_eos = nullptr;
  Bf16* d_xvector = nullptr;
  Bf16* cp_past = nullptr;  // talker-width final-norm row for the predictor

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

  // Fail at load when the weight file does not match the parsed config, rather
  // than letting a 0.6B header pair with 1.7B weights and overrun a buffer.
  void validate_shapes(const SafeTensors& st) const {
    for (const TensorShape& want : cfg.tensor_shape_plan()) {
      const TensorInfo& info = st.info(want.name);
      if (info.dtype != "BF16")
        throw std::runtime_error("expected BF16 tensor " + want.name + ", got " +
                                 info.dtype);
      if (info.shape != want.dims) {
        std::string got;
        for (size_t d : info.shape) got += " " + std::to_string(d);
        std::string expect;
        for (size_t d : want.dims) expect += " " + std::to_string(d);
        throw std::runtime_error("tensor " + want.name + " has shape [" + got +
                                 " ] but config.json implies [" + expect + " ]");
      }
    }
  }

  size_t kv_off(int layer, bool is_v, int slot, int n_kv_heads, int head_dim,
                size_t slot_cap) const {
    return ((((static_cast<size_t>(layer) * 2) + (is_v ? 1 : 0)) * slot_cap + slot) *
            n_kv_heads) * head_dim;
  }

  // Gather text ids, project to the talker width. Returns a stable device
  // pointer (the dedicated text_out buffer); do not alias fc2's input and
  // output.
  Bf16* project_text(const int* ids, int m) {
    UMM_CUDA_CHECK(cudaMemcpyAsync(d_ids, ids, m * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
    gather_rows_t<Bf16>(wb("talker.model.text_embedding.weight"), d_ids, text_in, m,
                        cfg.text_hidden, stream);
    lin(text_fc, text_in,
        dense("talker.text_projection.linear_fc1", cfg.text_hidden, cfg.text_hidden,
              true),
        m);
    silu_inplace_t<Bf16>(text_fc, m * cfg.text_hidden, stream);
    lin(text_out, text_fc,
        dense("talker.text_projection.linear_fc2", cfg.text_hidden, cfg.hidden, true),
        m);
    return text_out;
  }

  // Generic decoder layer shared by the talker and the code predictor. `base`
  // names the tensor prefix, `h`/`inter` the widths, and `s` the scratch.
  void layer(Bf16* x, int m, int history, int n_layers, Bf16* kv, size_t slot_cap,
             const std::string& base, int h, int inter, int n_q_heads,
             int n_kv_heads, int head_dim, int mask, int window_len, int q_offset,
             const LayerScratch& s) {
    for (int l = 0; l < n_layers; ++l) {
      const std::string pre = base + ".layers." + std::to_string(l);
      const int qw = n_q_heads * head_dim, kw = n_kv_heads * head_dim;
      rms_norm_t<Bf16>(x, wb(pre + ".input_layernorm.weight"), s.norm, m, h,
                       cfg.rms_norm_eps, stream);
      lin(s.q, s.norm, dense(pre + ".self_attn.q_proj", h, qw, false), m);
      lin(s.k, s.norm, dense(pre + ".self_attn.k_proj", h, kw, false), m);
      lin(s.v, s.norm, dense(pre + ".self_attn.v_proj", h, kw, false), m);
      rms_norm_t<Bf16>(s.q, wb(pre + ".self_attn.q_norm.weight"), s.q,
                       m * n_q_heads, head_dim, cfg.rms_norm_eps, stream);
      rms_norm_t<Bf16>(s.k, wb(pre + ".self_attn.k_norm.weight"), s.k,
                       m * n_kv_heads, head_dim, cfg.rms_norm_eps, stream);
      rope_half_t<Bf16>(s.q, d_pos, d_inv, m, n_q_heads, head_dim, stream);
      rope_half_t<Bf16>(s.k, d_pos, d_inv, m, n_kv_heads, head_dim, stream);
      const size_t sb = static_cast<size_t>(m) * kw * sizeof(Bf16);
      cudaMemcpyAsync(kv + kv_off(l, false, history, n_kv_heads, head_dim, slot_cap),
                      s.k, sb, cudaMemcpyDeviceToDevice, stream);
      cudaMemcpyAsync(kv + kv_off(l, true, history, n_kv_heads, head_dim, slot_cap),
                      s.v, sb, cudaMemcpyDeviceToDevice, stream);
      gqa_attention_t<Bf16>(s.q,
                            kv + kv_off(l, false, 0, n_kv_heads, head_dim, slot_cap),
                            kv + kv_off(l, true, 0, n_kv_heads, head_dim, slot_cap),
                            s.attn, m, history + m, n_q_heads, n_kv_heads, head_dim,
                            mask, window_len, history, stream);
      lin(s.q, s.attn, dense(pre + ".self_attn.o_proj", qw, h, false), m);
      add_residual_t<Bf16>(x, s.q, m * h, stream);

      rms_norm_t<Bf16>(x, wb(pre + ".post_attention_layernorm.weight"), s.norm, m, h,
                       cfg.rms_norm_eps, stream);
      lin(s.gate, s.norm, dense(pre + ".mlp.gate_proj", h, inter, false), m);
      lin(s.up, s.norm, dense(pre + ".mlp.up_proj", h, inter, false), m);
      silu_mul_t<Bf16>(s.norm, s.gate, s.up, m * inter, stream);
      lin(s.gate, s.norm, dense(pre + ".mlp.down_proj", inter, h, false), m);
      add_residual_t<Bf16>(x, s.gate, m * h, stream);
    }
  }

  // Copies one talker-width row into the predictor width, through the
  // checkpoint's small_to_mtp_projection when the widths differ (1.7B).
  void to_cp_width(Bf16* dst, const Bf16* src) {
    if (cfg.has_mtp_projection()) {
      lin(dst, src,
          dense("talker.code_predictor.small_to_mtp_projection", cfg.hidden,
                cfg.cp_hidden, true),
          1);
    } else {
      cudaMemcpyAsync(dst, src,
                      static_cast<size_t>(cfg.cp_hidden) * sizeof(Bf16),
                      cudaMemcpyDeviceToDevice, stream);
    }
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
    throw std::runtime_error("config.json language ids differ from the wired prompt");
  if (!cfg.is_base() &&
      (cfg.speaker_id("vivian") != kSpkVivian || cfg.speaker_id("ryan") != kSpkRyan))
    throw std::runtime_error("config.json speaker ids differ from the wired prompt");
  if (cfg.tts_pad_token_id != kTtsPad || cfg.tts_bos_token_id != kTtsBos ||
      cfg.tts_eos_token_id != kTtsEos)
    throw std::runtime_error("config.json tts special ids differ from the wired prompt");
  if (cfg.codec_pad_id != kCodecPad || cfg.codec_bos_id != kCodecBos ||
      cfg.codec_eos_id != kCodecEos || cfg.codec_think_id != kCodecThink ||
      cfg.codec_think_bos_id != kCodecThinkBos ||
      cfg.codec_think_eos_id != kCodecThinkEos)
    throw std::runtime_error("config.json codec control ids differ from the wired prompt");

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
    if (cfg.is_base()) im.speaker_encoder = std::make_unique<SpeakerEncoder>(st, cfg);
    const size_t start = st.data_region_offset();
    const size_t bytes = st.bytes().size() - start;
    UMM_CUDA_CHECK(cudaMalloc(&im.d_bf, bytes));
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
    UMM_CUDA_CHECK(cudaMalloc(&im.d_f32, bytes));
    UMM_CUDA_CHECK(cudaMemcpy(im.d_f32, st.bytes().data() + start, bytes,
                              cudaMemcpyHostToDevice));
    for (const auto& [name, info] : st.tensors()) {
      if (info.dtype != "F32") continue;  // skip any non-F32 entry
      im.fw.emplace(name, im.d_f32 + info.byte_offset / 4);
    }
  }
  im.codec = std::make_unique<CodecDecoder>(im.fw, im.stream);

  // Workspaces (sized for prefill up to a few hundred tokens).
  auto mk = [&](Bf16*& p, size_t n) { UMM_CUDA_CHECK(cudaMalloc(&p, n * sizeof(Bf16))); };
  const size_t W = kTalkerRows;
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
  // Dedicated fc2 target: project_text must not write into its own input (the
  // fc1 output), or the in-place GEMM races across blocks.
  mk(im.text_out, W * H);
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
  im.cp = {im.cp_norm, im.cp_q, im.cp_k, im.cp_v, im.cp_attn, im.cp_gate, im.cp_up};

  UMM_CUDA_CHECK(cudaMalloc(&im.logits, cfg.codec_vocab * sizeof(float)));
  UMM_CUDA_CHECK(cudaMalloc(&im.d_ids, W * sizeof(int)));
  UMM_CUDA_CHECK(cudaMalloc(&im.d_pos, W * sizeof(int)));
  UMM_CUDA_CHECK(cudaMalloc(&im.d_codes, cfg.code_groups * sizeof(int)));
  UMM_CUDA_CHECK(cudaMalloc(&im.d_hist, kMaxSlots * sizeof(int)));
  UMM_CUDA_CHECK(cudaMalloc(&im.d_inv, (HD / 2) * sizeof(float)));
  UMM_CUDA_CHECK(cudaMalloc(&im.d_pad, H * sizeof(Bf16)));
  UMM_CUDA_CHECK(cudaMalloc(&im.d_bos, H * sizeof(Bf16)));
  UMM_CUDA_CHECK(cudaMalloc(&im.d_eos, H * sizeof(Bf16)));
  if (cfg.is_base())
    UMM_CUDA_CHECK(cudaMalloc(&im.d_xvector, H * sizeof(Bf16)));
  UMM_CUDA_CHECK(cudaMalloc(&im.cp_past, H * sizeof(Bf16)));
  UMM_CUDA_CHECK(cudaMalloc(&im.talker_kv,
                            sizeof(Bf16) * cfg.layers * 2 * kMaxSlots * KVW));
  UMM_CUDA_CHECK(cudaMalloc(&im.cp_kv,
                            sizeof(Bf16) * cfg.cp_layers * 2 * kCpSlots * CPKVW));

  std::vector<float> inv(HD / 2);
  for (int i = 0; i < HD / 2; ++i)
    inv[i] = 1.0f / std::pow(cfg.rope_theta, 2.0f * i / HD);
  UMM_CUDA_CHECK(cudaMemcpy(im.d_inv, inv.data(), inv.size() * sizeof(float),
                            cudaMemcpyHostToDevice));

  // Precompute tts_pad/bos/eos vectors: text embed -> text_projection MLP.
  const int tts_ids[3] = {cfg.tts_pad_token_id, cfg.tts_bos_token_id,
                          cfg.tts_eos_token_id};
  UMM_CUDA_CHECK(cudaMemcpy(im.d_ids, tts_ids, sizeof(tts_ids), cudaMemcpyHostToDevice));
  gather_rows_t<Bf16>(im.wb("talker.model.text_embedding.weight"), im.d_ids,
                      im.text_in, 3, cfg.text_hidden, im.stream);
  if (const char* dump = std::getenv("UMM_TTS_DUMP_GATHER")) {
    std::vector<Bf16> hb(3 * cfg.text_hidden);
    cudaMemcpy(hb.data(), im.text_in, hb.size() * sizeof(Bf16),
               cudaMemcpyDeviceToHost);
    std::vector<float> hf(hb.size());
    for (size_t i = 0; i < hb.size(); ++i) hf[i] = __bfloat162float(hb[i]);
    std::ofstream f(dump, std::ios::binary);
    f.write(reinterpret_cast<const char*>(hf.data()), hf.size() * sizeof(float));
  }
  im.lin(im.text_fc, im.text_in,
         im.dense("talker.text_projection.linear_fc1", cfg.text_hidden, cfg.text_hidden,
                  true),
         3);
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
  im.lin(im.hidden, im.text_fc,
         im.dense("talker.text_projection.linear_fc2", cfg.text_hidden, H, true), 3);
  UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_pad, im.hidden, H * sizeof(Bf16),
                                 cudaMemcpyDeviceToDevice, im.stream));
  UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_bos, im.hidden + H, H * sizeof(Bf16),
                                 cudaMemcpyDeviceToDevice, im.stream));
  UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_eos, im.hidden + 2 * H, H * sizeof(Bf16),
                                 cudaMemcpyDeviceToDevice, im.stream));
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
}

TtsEngine::~TtsEngine() {
  if (!impl_) return;
  Impl& im = *impl_;
  cudaFree(im.cp_up);
  cudaFree(im.cp_gate);
  cudaFree(im.cp_attn);
  cudaFree(im.cp_v);
  cudaFree(im.cp_k);
  cudaFree(im.cp_q);
  cudaFree(im.cp_norm);
  cudaFree(im.cp_x);
  cudaFree(im.cp_kv);
  cudaFree(im.talker_kv);
  cudaFree(im.d_eos);
  cudaFree(im.d_bos);
  cudaFree(im.d_pad);
  cudaFree(im.d_xvector);
  cudaFree(im.cp_past);
  cudaFree(im.d_inv);
  cudaFree(im.d_codes);
  cudaFree(im.d_hist);
  cudaFree(im.d_pos);
  cudaFree(im.d_ids);
  cudaFree(im.logits);
  cudaFree(im.text_out);
  cudaFree(im.text_fc);
  cudaFree(im.text_in);
  cudaFree(im.proj);
  cudaFree(im.up);
  cudaFree(im.gate);
  cudaFree(im.attn);
  cudaFree(im.v);
  cudaFree(im.k);
  cudaFree(im.q);
  cudaFree(im.norm);
  cudaFree(im.hidden);
  cudaFree(im.d_f32);
  cudaFree(im.d_bf);
  if (im.stream) cudaStreamDestroy(im.stream);
}

std::vector<float> TtsEngine::extract_speaker_embedding(
    const std::vector<float>& ref_pcm_24k) {
  if (!impl_->cfg.is_base())
    throw std::invalid_argument("speaker extraction requires a Base model");
  return impl_->speaker_encoder->extract(ref_pcm_24k);
}

std::vector<float> TtsEngine::synthesize(const std::vector<int>& text_ids,
                                         int language_id, int speaker_id,
                                         int max_frames, const AbortCheck& abort,
                                         SynthTimings* timings, const StreamSink& on_chunk) {
  if (impl_->cfg.is_base())
    throw std::invalid_argument("Base models require synthesize_clone with a speaker embedding");
  return synthesize_impl(text_ids, language_id, speaker_id, nullptr, max_frames, abort,
                         timings, on_chunk);
}

std::vector<float> TtsEngine::synthesize_clone(
    const std::vector<int>& text_ids, int language_id,
    const std::vector<float>& speaker_embedding, int max_frames, const AbortCheck& abort,
    SynthTimings* timings, const StreamSink& on_chunk) {
  if (!impl_->cfg.is_base())
    throw std::invalid_argument("voice cloning requires a Base model");
  if (speaker_embedding.size() != static_cast<size_t>(impl_->cfg.hidden))
    throw std::invalid_argument("speaker embedding width does not match the talker");
  for (float v : speaker_embedding)
    if (!std::isfinite(v)) throw std::invalid_argument("speaker embedding contains non-finite data");
  return synthesize_impl(text_ids, language_id, -1, &speaker_embedding, max_frames, abort,
                         timings, on_chunk);
}

std::vector<float> TtsEngine::synthesize_impl(
    const std::vector<int>& text_ids, int language_id, int speaker_id,
    const std::vector<float>* speaker_embedding, int max_frames, const AbortCheck& abort,
    SynthTimings* timings, const StreamSink& on_chunk) {
  Impl& im = *impl_;
  const TtsModelConfig& cfg = im.cfg;
  const int H = cfg.hidden, INTER = cfg.inter, HD = cfg.head_dim;
  const int QH = cfg.q_heads, KH = cfg.kv_heads;
  const int CPH = cfg.cp_hidden, CPI = cfg.cp_inter;
  const int CPQH = cfg.cp_q_heads, CPKH = cfg.cp_kv_heads, CPHD = cfg.cp_head_dim;
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
  const PromptPlan plan = speaker_embedding != nullptr
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
        gather_rows_t<Bf16>(im.wb("talker.model.codec_embedding.weight"), im.d_ids,
                            im.proj, 1, H, im.stream);
        src = im.proj;
      } else if (term.kind == PromptTerm::kXVector) {
        src = im.d_xvector;
      } else {
        src = term.kind == PromptTerm::kTtsPad ? im.d_pad
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

  // Optional bring-up dump of the prefill embedding matrix for parity checks
  // against the offline PyTorch reference (env UMM_TTS_DUMP=<path>.f32).
  if (const char* dump = std::getenv("UMM_TTS_DUMP")) {
    std::vector<Bf16> hb(static_cast<size_t>(L) * H);
    UMM_CUDA_CHECK(cudaMemcpy(hb.data(), im.hidden, hb.size() * sizeof(Bf16),
                              cudaMemcpyDeviceToHost));
    std::vector<float> hf(hb.size());
    for (size_t i = 0; i < hb.size(); ++i) hf[i] = __bfloat162float(hb[i]);
    std::ofstream f(dump, std::ios::binary);
    f.write(reinterpret_cast<const char*>(hf.data()), hf.size() * sizeof(float));
  }

  std::vector<int> positions(L);
  std::iota(positions.begin(), positions.end(), 0);
  UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_pos, positions.data(), L * sizeof(int),
                                 cudaMemcpyHostToDevice, im.stream));
  im.layer(im.hidden, L, 0, cfg.layers, im.talker_kv, kMaxSlots, "talker.model", H,
           INTER, QH, KH, HD, /*mask=*/0, /*window_len=*/0, /*q_offset=*/0, im.tk);
  UMM_CUDA_CHECK(cudaStreamSynchronize(im.stream));
  int seq_len = L;

  // Match the reference generation config: greedy codec head with HF
  // repetition penalty 1.05 over emitted c0 ids (pure greedy loops forever).
  constexpr float kRepPenalty = 1.05f;
  // BF16 reduction-order noise can flip a near-tie into a self-reinforcing
  // repeat; ban an id after kRunBan consecutive emissions to keep decoding
  // moving. ponytail: hard cap; replace with no-repeat-ngram if needed.
  int kRunBan = 4;
  if (const char* e = std::getenv("UMM_TTS_RUN_BAN")) kRunBan = std::stoi(e);
  auto select_c0 = [&](bool allow_eos, int n_hist, int ban_id) {
    rms_norm_t<Bf16>(im.hidden + static_cast<size_t>(seq_len - 1) * H,
                     im.wb("talker.model.norm.weight"), im.norm, 1, H,
                     cfg.rms_norm_eps, im.stream);
    gemv_t<Bf16>(im.wb("talker.codec_head.weight"), im.norm, im.logits,
                 cfg.codec_vocab, H, im.stream);
    repetition_penalty_f32(im.logits, im.d_hist, n_hist, kRepPenalty, im.stream);
    suppress_logits(im.logits, cfg.cp_vocab, cfg.codec_vocab,
                    allow_eos ? cfg.codec_eos_id : -1, im.stream);
    if (ban_id >= 0) ban_logit_f32(im.logits, ban_id, im.stream);
    if (const char* dump = std::getenv("UMM_TTS_DUMP_LOGITS")) {
      std::vector<float> h(cfg.codec_vocab);
      cudaMemcpy(h.data(), im.logits, h.size() * sizeof(float),
                 cudaMemcpyDeviceToHost);
      std::ofstream f(std::string(dump) + "." + std::to_string(n_hist),
                      std::ios::binary);
      f.write(reinterpret_cast<const char*>(h.data()), h.size() * sizeof(float));
    }
    argmax_f32_dev(im.logits, cfg.codec_vocab, im.d_codes, im.stream);
  };

  // Run the code predictor for one frame. c0 is read from d_codes[0] and
  // c1..c(groups-1) are written to the following slots. No host round-trip:
  // every sub-codebook argmax leaves its id in device memory, and the next
  // step's embedding gather reads it from there.
  auto run_predictor = [&](const Bf16* past_hidden) {
    // Prefill rows: [past_hidden, codec_embed(c0)], both talker-width in the
    // checkpoint and narrowed to the predictor width when they differ. The
    // reference passes the talker's final-normalized row here (modeling:
    // past_hidden = self.norm(...)[:, -1]), not the raw residual stream; the
    // 1.7B projection is trained on the normalized input.
    rms_norm_t<Bf16>(past_hidden, im.wb("talker.model.norm.weight"), im.cp_past, 1,
                     H, cfg.rms_norm_eps, im.stream);
    im.to_cp_width(im.cp_x, im.cp_past);
    gather_rows_t<Bf16>(im.wb("talker.model.codec_embedding.weight"), im.d_codes,
                        im.proj, 1, H, im.stream);
    im.to_cp_width(im.cp_x + CPH, im.proj);
    int pp[2] = {0, 1};
    UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_pos, pp, sizeof(pp), cudaMemcpyHostToDevice,
                                   im.stream));
    im.layer(im.cp_x, 2, 0, cfg.cp_layers, im.cp_kv, kCpSlots,
             "talker.code_predictor.model", CPH, CPI, CPQH, CPKH, CPHD,
             /*mask=*/0, /*window_len=*/0, /*q_offset=*/0, im.cp);
    rms_norm_t<Bf16>(im.cp_x + CPH, im.wb("talker.code_predictor.model.norm.weight"),
                     im.cp_norm, 1, CPH, cfg.rms_norm_eps, im.stream);
    gemv_t<Bf16>(im.wb("talker.code_predictor.lm_head.0.weight"), im.cp_norm,
                 im.logits, cfg.cp_vocab, CPH, im.stream);
    argmax_f32_dev(im.logits, cfg.cp_vocab, im.d_codes + 1, im.stream);
    int cp_len = 2;
    for (int j = 1; j < groups - 1; ++j) {
      gather_rows_t<Bf16>(
          im.wb("talker.code_predictor.model.codec_embedding." +
                std::to_string(j - 1) + ".weight"),
          im.d_codes + j, im.proj, 1, H, im.stream);
      im.to_cp_width(im.cp_x, im.proj);
      int pos = cp_len;
      UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_pos, &pos, sizeof(int),
                                     cudaMemcpyHostToDevice, im.stream));
      im.layer(im.cp_x, 1, cp_len, cfg.cp_layers, im.cp_kv, kCpSlots,
               "talker.code_predictor.model", CPH, CPI, CPQH, CPKH, CPHD,
               /*mask=*/0, /*window_len=*/0, /*q_offset=*/0, im.cp);
      ++cp_len;
      rms_norm_t<Bf16>(im.cp_x, im.wb("talker.code_predictor.model.norm.weight"),
                       im.cp_norm, 1, CPH, cfg.rms_norm_eps, im.stream);
      gemv_t<Bf16>(
          im.wb("talker.code_predictor.lm_head." + std::to_string(j) + ".weight"),
          im.cp_norm, im.logits, cfg.cp_vocab, CPH, im.stream);
      argmax_f32_dev(im.logits, cfg.cp_vocab, im.d_codes + j + 1, im.stream);
    }
  };

  // Two host snapshots per frame, each at a boundary the host must cross
  // anyway: the residual ids feed the host-side codec decode, and c0 decides
  // whether the loop continues. Both keep the frame's order of operations
  // unchanged, so time-to-first-audio does not move.
  int hc[16];
  auto fetch_rest = [&]() {
    UMM_CUDA_CHECK(cudaMemcpyAsync(hc + 1, im.d_codes + 1, (groups - 1) * sizeof(int),
                                   cudaMemcpyDeviceToHost, im.stream));
    UMM_CUDA_CHECK(cudaStreamSynchronize(im.stream));
  };
  auto fetch_c0 = [&]() {
    UMM_CUDA_CHECK(cudaMemcpyAsync(hc, im.d_codes, sizeof(int),
                                   cudaMemcpyDeviceToHost, im.stream));
    UMM_CUDA_CHECK(cudaStreamSynchronize(im.stream));
  };
  select_c0(/*allow_eos=*/false, /*n_hist=*/0, /*ban_id=*/-1);
  fetch_c0();
  int c0 = hc[0];
  std::vector<int> frames;
  std::vector<int> c0_trace;
  // Streaming policy: the first chunk is short so audio starts early, later
  // chunks are longer to keep the repeated prefix decode cheap. Every chunk
  // ends on a codec-frame boundary.
  //
  // One waveform copy serves the whole request: a chunk decode returns a
  // prefix of the waveform, and only the samples past `decoded_frames` are
  // kept, so `samples` is by construction the concatenation of the chunks in
  // order. The function returns that same vector. The codec is not
  // bit-repeatable across decodes, so the terminal waveform is never decoded
  // a second time.
  constexpr int kSamplesPerFrame = 1920;
  constexpr int kSampleRate = 24000;
  constexpr int kFirstChunkFrames = 2;
  constexpr int kChunkFrames = 8;
  std::vector<float> samples;
  int decoded_frames = 0;
  int next_stream_frame = kFirstChunkFrames;
  // The terminal waveform is trimmed of the decoder's trailing noise, so the
  // end of the stream cannot be emitted before the synthesis knows it is the
  // end. The last kTrimGuardFrames are kept back and released as whole frames
  // whenever a later chunk arrives. The guard holds 1.04 s: the 0.6 s noise
  // cap plus the silence the run may carry. A longer cut is clamped to the
  // guard. The first chunk is emitted at once so first_audio_ms does not
  // move.
  constexpr int kTrimGuardFrames = 13;
  constexpr size_t kTrimGuardSamples = kTrimGuardFrames * kSamplesPerFrame;
  std::vector<float> held;
  int held_frame = 0;
  bool emitted_any = false;
  auto emit = [&](const std::vector<float>& chunk, int first_frame, int frames) {
    if (timings != nullptr && timings->first_audio_ms < 0)
      timings->first_audio_ms = elapsed_ms();
    on_chunk(chunk, first_frame, frames);
  };
  // Takes a decode of the first `frame_count` frames and keeps the samples
  // that are past the frames already emitted.
  auto take = [&](std::vector<float>&& all, int frame_count) {
    const size_t start = static_cast<size_t>(decoded_frames) * kSamplesPerFrame;
    if (all.size() <= start) return;
    std::vector<float> chunk(all.begin() + static_cast<long>(start), all.end());
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
      std::vector<float> out(held.begin(), held.begin() + static_cast<long>(release));
      emit(out, held_frame, static_cast<int>(release / kSamplesPerFrame));
      held.erase(held.begin(), held.begin() + static_cast<long>(release));
      held_frame += static_cast<int>(release / kSamplesPerFrame);
    }
  };
  // Trims the terminal noise and emits the held tail. `samples` keeps the
  // trimmed tail, so the emitted chunks still concatenate to it exactly. A
  // single-chunk utterance has nothing held back to trim, and is left alone.
  auto finish_stream = [&]() {
    const size_t cut = std::min(trailing_noise_cut(samples, kSampleRate), held.size());
    if (cut > 0) {
      held.resize(held.size() - cut);
      samples.resize(samples.size() - cut);
      fade_out(held, static_cast<size_t>(kSampleRate) * 5 / 1000);
      std::copy(held.begin(), held.end(), samples.end() - held.size());
    }
    if (!held.empty()) emit(held, held_frame, static_cast<int>(held.size() / kSamplesPerFrame));
    held.clear();
  };
  c0_trace.push_back(c0);
  for (int step = 0; step < max_frames; ++step) {
    if (abort && abort()) {
      throw RequestAborted("aborted at frame " + std::to_string(step));
    }
    if (c0 < 0 || c0 >= cfg.codec_vocab)
      throw std::runtime_error("talker produced out-of-range codec id");
    if (seq_len + 1 > kTalkerRows)
      throw std::runtime_error("tts text too long for the talker buffers");
    UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_hist + step, &c0, sizeof(int),
                                   cudaMemcpyHostToDevice, im.stream));
    run_predictor(im.hidden + (seq_len - 1) * H);
    fetch_rest();
    frames.push_back(c0);
    if (step == 0 && timings != nullptr) timings->first_frame_ms = elapsed_ms();
    for (int j = 0; j < groups - 1; ++j) frames.push_back(hc[1 + j]);
    const int frame_count = static_cast<int>(frames.size() / groups);
    if (on_chunk && frame_count >= next_stream_frame) {
      take(im.codec->decode(frames, frame_count, abort), frame_count);
      next_stream_frame = frame_count + kChunkFrames;
    }

    // Frame vector: codec_embed(c0) + residual predictor embeddings + tts_pad.
    gather_rows_t<Bf16>(im.wb("talker.model.codec_embedding.weight"), im.d_codes,
                        im.proj, 1, H, im.stream);
    for (int j = 0; j < groups - 1; ++j) {
      gather_rows_t<Bf16>(
          im.wb("talker.code_predictor.model.codec_embedding." +
                std::to_string(j) + ".weight"),
          im.d_codes + 1 + j, im.hidden, 1, H, im.stream);
      add_residual_t<Bf16>(im.proj, im.hidden, H, im.stream);
    }
    add_residual_t<Bf16>(im.proj, im.d_pad, H, im.stream);
    cudaMemcpyAsync(im.hidden + seq_len * H, im.proj, H * sizeof(Bf16),
                    cudaMemcpyDeviceToDevice, im.stream);

    int pos = seq_len;
    UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_pos, &pos, sizeof(int),
                                   cudaMemcpyHostToDevice, im.stream));
    im.layer(im.hidden + seq_len * H, 1, seq_len, cfg.layers, im.talker_kv, kMaxSlots,
             "talker.model", H, INTER, QH, KH, HD, /*mask=*/0, /*window_len=*/0,
             /*q_offset=*/0, im.tk);
    ++seq_len;
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
    const int next = hc[0];
    if (next == cfg.codec_eos_id) break;
    c0 = next;
    c0_trace.push_back(c0);
  }
  if (const char* dump = std::getenv("UMM_TTS_DUMP_C0")) {
    std::ofstream f(dump, std::ios::binary);
    f.write(reinterpret_cast<const char*>(c0_trace.data()),
            c0_trace.size() * sizeof(int));
  }
  // Every codec id in emission order, one frame per `groups` ids. Diagnostic
  // only: the file is written from a vector that already exists, so decoding
  // is unchanged.
  if (const char* dump = std::getenv("UMM_TTS_DUMP_FRAMES")) {
    std::ofstream f(dump, std::ios::binary);
    f.write(reinterpret_cast<const char*>(frames.data()),
            static_cast<std::streamsize>(frames.size() * sizeof(int)));
  }
  UMM_CUDA_CHECK(cudaStreamSynchronize(im.stream));
  // The waveform decoder checks between its own stages too; this boundary
  // covers the hand-off.
  if (abort && abort()) {
    throw RequestAborted("aborted before codec decode at frame " +
                         std::to_string(frames.size() / groups));
  }
  if (frames.size() % groups != 0)
    throw std::runtime_error("codec frames not divisible by the code group count");
  const int total_frames = static_cast<int>(frames.size() / groups);
  if (!on_chunk) {
    samples = trim_trailing_noise(im.codec->decode(frames, total_frames, abort),
                                  kSampleRate);
  } else {
    if (decoded_frames < total_frames) {
      // Emit the tail frames the loop did not reach.
      take(im.codec->decode(frames, total_frames, abort), total_frames);
    }
    finish_stream();
  }
  if (on_chunk && std::getenv("UMM_TTS_REPEAT_CHECK")) {
    // Diagnostic only: decodes of the same frames are not bit-repeatable, so a
    // separate decode differs from the emitted waveform. This is a codec
    // property, not a streaming bug.
    std::vector<float> again = im.codec->decode(frames, total_frames, abort);
    size_t fd = 0;
    while (fd < again.size() && fd < samples.size() && again[fd] == samples[fd]) ++fd;
    float md = 0.0f;
    for (size_t i = 0; i < std::min(again.size(), samples.size()); ++i)
      md = std::max(md, std::fabs(again[i] - samples[i]));
    std::cerr << "tts repeat_check same_size=" << (again.size() == samples.size())
              << " first_diff=" << fd << " max_abs_diff=" << md << "\n";
  }
  if (timings != nullptr) {
    timings->total_ms = elapsed_ms();
    // The CLI reports waveform stats only; the verification runs need the
    // phase times, so print them on request without adding a flag.
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
