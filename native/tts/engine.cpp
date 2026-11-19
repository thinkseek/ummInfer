// SPDX-License-Identifier: Apache-2.0
#include "engine.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "codec.hpp"
#include "cuda_kernels.hpp"
#include "prompt.hpp"
#include "safetensors.hpp"
#include "tail_trim.hpp"

namespace umminfer::native {
namespace {

constexpr int kH = 1024, kQH = 16, kKH = 8, kHD = 128, kInter = 3072;
constexpr int kTalkerLayers = 28;
constexpr int kCpLayers = 5;
constexpr int kTextH = 2048;
constexpr int kTextVocab = 151936;
constexpr int kCodecVocab = 3072;
constexpr int kCpVocab = 2048;
constexpr int kGroups = 16;
constexpr int kEps = 1e-6;
constexpr float kRopeTheta = 1000000.0f;
constexpr int kMaxSlots = 8704;       // 8192 frames + prompt headroom
// Talker activation rows: the prompt (at most 200 rows) plus every decoded
// frame. Decode writes row seq_len of im.hidden after each frame, so this must
// match the KV capacity, not just the prefill length.
constexpr int kTalkerRows = kMaxSlots + 256;
constexpr int kCpSlots = 32;

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

}  // namespace

struct TtsEngine::Impl {
  cudaStream_t stream = nullptr;
  int gpu_id = 0;

  std::unordered_map<std::string, Bf16*> bw;   // talker BF16
  Bf16* d_bf = nullptr;
  std::unordered_map<std::string, const float*> fw;  // codec F32
  float* d_f32 = nullptr;
  std::unique_ptr<CodecDecoder> codec;

  // Talker scratch.
  Bf16 *hidden = nullptr, *norm = nullptr, *q = nullptr, *k = nullptr, *v = nullptr,
      *attn = nullptr, *gate = nullptr, *up = nullptr, *proj = nullptr,
      *text2048 = nullptr, *text_out = nullptr;
  float* logits = nullptr;
  int *d_ids = nullptr, *d_pos = nullptr, *d_argmax = nullptr;
  int* d_hist = nullptr;  // emitted c0 ids for repetition penalty
  float* d_inv = nullptr;
  Bf16* talker_kv = nullptr;
  Bf16* cp_kv = nullptr;
  Bf16 tts_pad_vec[1024];  // host copies uploaded once
  Bf16 *d_pad = nullptr, *d_bos = nullptr, *d_eos = nullptr;

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

  size_t kv_off(int layer, bool is_v, int slot) const {
    return ((((static_cast<size_t>(layer) * 2) + (is_v ? 1 : 0)) * kMaxSlots + slot) *
            kKH) * kHD;
  }
  size_t cp_off(int layer, bool is_v, int slot) const {
    return ((((static_cast<size_t>(layer) * 2) + (is_v ? 1 : 0)) * kCpSlots + slot) *
            kKH) * kHD;
  }

  // Gather text ids, project to 1024. Returns a stable device pointer (the
  // dedicated text_out buffer); do not alias fc2's input and output.
  Bf16* project_text(const int* ids, int m) {
    UMM_CUDA_CHECK(cudaMemcpyAsync(d_ids, ids, m * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
    gather_rows_t<Bf16>(wb("talker.model.text_embedding.weight"), d_ids, text2048, m,
                        kTextH, stream);
    lin(proj, text2048, dense("talker.text_projection.linear_fc1", kTextH, kTextH, true),
        m);
    silu_inplace_t<Bf16>(proj, m * kTextH, stream);
    lin(text_out, proj,
        dense("talker.text_projection.linear_fc2", kTextH, kH, true), m);
    return text_out;
  }

  // Generic 1024-wide decoder layer (talker and CP share topology). kv_base
  // selects the cache; slots/slot_count bound it.
  void layer(Bf16* x, int m, int history, int n_layers, Bf16* kv, size_t slots,
             bool cp) {
    for (int l = 0; l < n_layers; ++l) {
      const std::string pre =
          cp ? "talker.code_predictor.model.layers." + std::to_string(l)
             : "talker.model.layers." + std::to_string(l);
      rms_norm_t<Bf16>(x, wb(pre + ".input_layernorm.weight"), norm, m, kH, kEps,
                       stream);
      lin(q, norm, dense(pre + ".self_attn.q_proj", kH, kQH * kHD, false), m);
      lin(k, norm, dense(pre + ".self_attn.k_proj", kH, kKH * kHD, false), m);
      lin(v, norm, dense(pre + ".self_attn.v_proj", kH, kKH * kHD, false), m);
      rms_norm_t<Bf16>(q, wb(pre + ".self_attn.q_norm.weight"), q, m * kQH, kHD,
                       kEps, stream);
      rms_norm_t<Bf16>(k, wb(pre + ".self_attn.k_norm.weight"), k, m * kKH, kHD,
                       kEps, stream);
      rope_half_t<Bf16>(q, d_pos, d_inv, m, kQH, kHD, stream);
      rope_half_t<Bf16>(k, d_pos, d_inv, m, kKH, kHD, stream);
      const size_t sb = m * kKH * kHD * sizeof(Bf16);
      auto cpfx = [&](bool isv, int slot) -> size_t {
        return cp ? cp_off(l, isv, slot) : kv_off(l, isv, slot);
      };
      cudaMemcpyAsync(kv + cpfx(false, history), k, sb, cudaMemcpyDeviceToDevice,
                      stream);
      cudaMemcpyAsync(kv + cpfx(true, history), v, sb, cudaMemcpyDeviceToDevice,
                      stream);
      gqa_attention_t<Bf16>(q, kv + cpfx(false, 0), kv + cpfx(true, 0), attn, m,
                            history + m, kQH, kKH, kHD, 0, 0, history, stream);
      lin(q, attn, dense(pre + ".self_attn.o_proj", kQH * kHD, kH, false), m);
      add_residual_t<Bf16>(x, q, m * kH, stream);

      rms_norm_t<Bf16>(x, wb(pre + ".post_attention_layernorm.weight"), norm, m,
                       kH, kEps, stream);
      lin(gate, norm, dense(pre + ".mlp.gate_proj", kH, kInter, false), m);
      lin(up, norm, dense(pre + ".mlp.up_proj", kH, kInter, false), m);
      silu_mul_t<Bf16>(norm, gate, up, m * kInter, stream);
      lin(gate, norm, dense(pre + ".mlp.down_proj", kInter, kH, false), m);
      add_residual_t<Bf16>(x, gate, m * kH, stream);
    }
  }
};

TtsEngine::TtsEngine(const std::string& model_dir, int gpu_id) {
  impl_ = std::make_unique<Impl>();
  Impl& im = *impl_;
  im.gpu_id = gpu_id;
  UMM_CUDA_CHECK(cudaSetDevice(gpu_id));
  UMM_CUDA_CHECK(cudaStreamCreate(&im.stream));

  // Fail loudly on an unsupported checkpoint.
  const nlohmann::json cfg =
      nlohmann::json::parse(read_file(model_dir + "/config.json"));
  const auto& tc = cfg["talker_config"];
  auto need = [](const nlohmann::json& j, const char* key, long long want) {
    if (j.value(key, -1) != want)
      throw std::runtime_error(std::string("unsupported config: ") + key);
  };
  need(tc, "num_hidden_layers", kTalkerLayers);
  need(tc, "hidden_size", kH);
  need(tc, "num_attention_heads", kQH);
  need(tc, "num_key_value_heads", kKH);
  need(tc, "head_dim", kHD);
  need(tc, "intermediate_size", kInter);
  need(tc, "text_vocab_size", kTextVocab);
  need(tc, "vocab_size", kCodecVocab);
  need(tc["code_predictor_config"], "num_hidden_layers", kCpLayers);
  need(tc["code_predictor_config"], "vocab_size", kCpVocab);

  // Talker BF16 region.
  {
    SafeTensors st(read_file(model_dir + "/model.safetensors"));
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
  mk(im.hidden, W * kH);
  mk(im.norm, W * kH);
  mk(im.q, W * kQH * kHD);
  mk(im.k, W * kKH * kHD);
  mk(im.v, W * kKH * kHD);
  mk(im.attn, W * kQH * kHD);
  mk(im.gate, W * kInter);
  mk(im.up, W * kInter);
  mk(im.proj, W * kH);
  mk(im.text2048, W * kTextH);
  // Dedicated fc2 target: project_text's fc2 must not write into its own
  // input (im.proj), or the in-place GEMM races across blocks.
  mk(im.text_out, W * kH);
  UMM_CUDA_CHECK(cudaMalloc(&im.logits, kCodecVocab * sizeof(float)));
  UMM_CUDA_CHECK(cudaMalloc(&im.d_ids, W * sizeof(int)));
  UMM_CUDA_CHECK(cudaMalloc(&im.d_pos, W * sizeof(int)));
  UMM_CUDA_CHECK(cudaMalloc(&im.d_argmax, sizeof(int)));
  UMM_CUDA_CHECK(cudaMalloc(&im.d_hist, kMaxSlots * sizeof(int)));
  UMM_CUDA_CHECK(cudaMalloc(&im.d_inv, (kHD / 2) * sizeof(float)));
  UMM_CUDA_CHECK(cudaMalloc(&im.d_pad, kH * sizeof(Bf16)));
  UMM_CUDA_CHECK(cudaMalloc(&im.d_bos, kH * sizeof(Bf16)));
  UMM_CUDA_CHECK(cudaMalloc(&im.d_eos, kH * sizeof(Bf16)));
  UMM_CUDA_CHECK(cudaMalloc(&im.talker_kv,
                            sizeof(Bf16) * kTalkerLayers * 2 * kMaxSlots * kKH * kHD));
  UMM_CUDA_CHECK(cudaMalloc(&im.cp_kv,
                            sizeof(Bf16) * kCpLayers * 2 * kCpSlots * kKH * kHD));

  std::vector<float> inv(kHD / 2);
  for (int i = 0; i < kHD / 2; ++i)
    inv[i] = 1.0f / std::pow(kRopeTheta, 2.0f * i / kHD);
  UMM_CUDA_CHECK(cudaMemcpy(im.d_inv, inv.data(), inv.size() * sizeof(float),
                            cudaMemcpyHostToDevice));

  // Precompute tts_pad/bos/eos vectors: text embed -> text_projection MLP.
  const int tts_ids[3] = {kTtsPad, kTtsBos, kTtsEos};
  UMM_CUDA_CHECK(cudaMemcpy(im.d_ids, tts_ids, sizeof(tts_ids), cudaMemcpyHostToDevice));
  gather_rows_t<Bf16>(im.wb("talker.model.text_embedding.weight"), im.d_ids,
                      im.text2048, 3, kTextH, im.stream);
  if (const char* dump = std::getenv("UMM_TTS_DUMP_GATHER")) {
    Bf16 hb[3 * kTextH];
    cudaMemcpy(hb, im.text2048, sizeof(hb), cudaMemcpyDeviceToHost);
    std::vector<float> hf(3 * kTextH);
    for (int i = 0; i < 3 * kTextH; ++i) hf[i] = __bfloat162float(hb[i]);
    std::ofstream f(dump, std::ios::binary);
    f.write(reinterpret_cast<const char*>(hf.data()), hf.size() * sizeof(float));
  }
  im.lin(im.proj, im.text2048,
         im.dense("talker.text_projection.linear_fc1", kTextH, kTextH, true), 3);
  if (const char* dump = std::getenv("UMM_TTS_DUMP_FC1")) {
    Bf16 hb[3 * kTextH];
    cudaMemcpy(hb, im.proj, sizeof(hb), cudaMemcpyDeviceToHost);
    std::vector<float> hf(3 * kTextH);
    for (int i = 0; i < 3 * kTextH; ++i) hf[i] = __bfloat162float(hb[i]);
    std::ofstream f(dump, std::ios::binary);
    f.write(reinterpret_cast<const char*>(hf.data()), hf.size() * sizeof(float));
  }
  // SiLU in place on the 3x2048 buffer.
  silu_inplace_t<Bf16>(im.proj, 3 * kTextH, im.stream);
  if (const char* dump = std::getenv("UMM_TTS_DUMP_SILU")) {
    Bf16 hb[3 * kTextH];
    cudaMemcpy(hb, im.proj, sizeof(hb), cudaMemcpyDeviceToHost);
    std::vector<float> hf(3 * kTextH);
    for (int i = 0; i < 3 * kTextH; ++i) hf[i] = __bfloat162float(hb[i]);
    std::ofstream f(dump, std::ios::binary);
    f.write(reinterpret_cast<const char*>(hf.data()), hf.size() * sizeof(float));
  }
  im.lin(im.hidden, im.proj,
         im.dense("talker.text_projection.linear_fc2", kTextH, kH, true), 3);
  UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_pad, im.hidden, kH * sizeof(Bf16),
                                 cudaMemcpyDeviceToDevice, im.stream));
  UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_bos, im.hidden + kH, kH * sizeof(Bf16),
                                 cudaMemcpyDeviceToDevice, im.stream));
  UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_eos, im.hidden + 2 * kH, kH * sizeof(Bf16),
                                 cudaMemcpyDeviceToDevice, im.stream));
  UMM_CUDA_CHECK(cudaStreamSynchronize(im.stream));

  if (const char* dump = std::getenv("UMM_TTS_DUMP_PAD")) {
    Bf16 hb[3 * kH];
    cudaMemcpy(hb, im.hidden, sizeof(hb), cudaMemcpyDeviceToHost);
    std::vector<float> hf(3 * kH);
    for (int i = 0; i < 3 * kH; ++i) hf[i] = __bfloat162float(hb[i]);
    std::ofstream f(dump, std::ios::binary);
    f.write(reinterpret_cast<const char*>(hf.data()), hf.size() * sizeof(float));
  }
}

TtsEngine::~TtsEngine() {
  if (!impl_) return;
  Impl& im = *impl_;
  cudaFree(im.cp_kv);
  cudaFree(im.talker_kv);
  cudaFree(im.d_eos);
  cudaFree(im.d_bos);
  cudaFree(im.d_pad);
  cudaFree(im.d_inv);
  cudaFree(im.d_argmax);
  cudaFree(im.d_hist);
  cudaFree(im.d_pos);
  cudaFree(im.d_ids);
  cudaFree(im.logits);
  cudaFree(im.text2048);
  cudaFree(im.text_out);
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

std::vector<float> TtsEngine::synthesize(const std::vector<int>& text_ids,
                                         int language_id, int speaker_id,
                                         int max_frames,
                                         const AbortCheck& abort,
                                         SynthTimings* timings,
                                         const StreamSink& on_chunk) {
  Impl& im = *impl_;
  const auto started = std::chrono::steady_clock::now();
  const auto elapsed_ms = [&started] {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - started)
        .count();
  };
  const PromptPlan plan = build_prompt_plan(text_ids, language_id, speaker_id);
  const int L = static_cast<int>(plan.rows.size());
  if (L > 200) throw std::runtime_error("prompt too long");

  // Build prefill embeddings row by row into im.hidden [L,1024].
  for (int r = 0; r < L; ++r) {
    Bf16* dst = im.hidden + static_cast<size_t>(r) * kH;
    bool first = true;
    for (const PromptTerm& term : plan.rows[r]) {
      const Bf16* src = nullptr;
      if (term.kind == PromptTerm::kTextProj) {
        src = im.project_text(&term.id, 1);
      } else if (term.kind == PromptTerm::kCodec) {
        UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_ids, &term.id, sizeof(int),
                                       cudaMemcpyHostToDevice, im.stream));
        gather_rows_t<Bf16>(im.wb("talker.model.codec_embedding.weight"), im.d_ids,
                            im.proj, 1, kH, im.stream);
        src = im.proj;
      } else {
        src = term.kind == PromptTerm::kTtsPad ? im.d_pad
             : term.kind == PromptTerm::kTtsBos ? im.d_bos
                                                : im.d_eos;
      }
      if (first) {
        cudaMemcpyAsync(dst, src, kH * sizeof(Bf16), cudaMemcpyDeviceToDevice,
                        im.stream);
        first = false;
      } else {
        add_residual_t<Bf16>(dst, src, kH, im.stream);
      }
    }
    if (r < 3) {
      UMM_CUDA_CHECK(cudaStreamSynchronize(im.stream));
    }
  }

  // Optional bring-up dump of the prefill embedding matrix for parity checks
  // against the offline PyTorch reference (env UMM_TTS_DUMP=<path>.f32).
  if (const char* dump = std::getenv("UMM_TTS_DUMP")) {
    std::vector<Bf16> hb(static_cast<size_t>(L) * kH);
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
  im.layer(im.hidden, L, 0, kTalkerLayers, im.talker_kv, kMaxSlots, false);
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
  auto select_c0 = [&](bool allow_eos, int n_hist, int ban_id) -> int {
    rms_norm_t<Bf16>(im.hidden + static_cast<size_t>(seq_len - 1) * kH,
                     im.wb("talker.model.norm.weight"), im.norm, 1, kH, kEps,
                     im.stream);
    gemv_t<Bf16>(im.wb("talker.codec_head.weight"), im.norm, im.logits, kCodecVocab,
                 kH, im.stream);
    repetition_penalty_f32(im.logits, im.d_hist, n_hist, kRepPenalty, im.stream);
    suppress_logits(im.logits, 2048, 3072, allow_eos ? kCodecEos : -1, im.stream);
    if (ban_id >= 0) ban_logit_f32(im.logits, ban_id, im.stream);
    if (const char* dump = std::getenv("UMM_TTS_DUMP_LOGITS")) {
      std::vector<float> h(kCodecVocab);
      cudaMemcpy(h.data(), im.logits, h.size() * sizeof(float),
                 cudaMemcpyDeviceToHost);
      std::ofstream f(std::string(dump) + "." + std::to_string(n_hist),
                      std::ios::binary);
      f.write(reinterpret_cast<const char*>(h.data()), h.size() * sizeof(float));
    }
    return argmax_f32(im.logits, kCodecVocab, im.d_argmax, im.stream);
  };

  // Run the 5-layer code predictor for one frame; returns c1..c15.
  auto run_predictor = [&](const Bf16* past_hidden, int c0) -> std::vector<int> {
    Bf16 emb[1];
    (void)emb;
    // Prefill rows: [past_hidden, codec_embed(c0)].
    UMM_CUDA_CHECK(cudaMemcpyAsync(im.proj, past_hidden, kH * sizeof(Bf16),
                                   cudaMemcpyDeviceToDevice, im.stream));
    UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_ids, &c0, sizeof(int), cudaMemcpyHostToDevice,
                                   im.stream));
    gather_rows_t<Bf16>(im.wb("talker.model.codec_embedding.weight"), im.d_ids,
                        im.proj + kH, 1, kH, im.stream);
    int pp[2] = {0, 1};
    UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_pos, pp, sizeof(pp), cudaMemcpyHostToDevice,
                                   im.stream));
    im.layer(im.proj, 2, 0, kCpLayers, im.cp_kv, kCpSlots, true);
    rms_norm_t<Bf16>(im.proj + kH, im.wb("talker.code_predictor.model.norm.weight"),
                     im.norm, 1, kH, kEps, im.stream);
    gemv_t<Bf16>(im.wb("talker.code_predictor.lm_head.0.weight"), im.norm, im.logits,
                 kCpVocab, kH, im.stream);
    std::vector<int> codes(15);
    codes[0] = argmax_f32(im.logits, kCpVocab, im.d_argmax, im.stream);
    int cp_len = 2;
    for (int j = 1; j < 15; ++j) {
      const int prev = codes[j - 1];
      UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_ids, &prev, sizeof(int),
                                     cudaMemcpyHostToDevice, im.stream));
      gather_rows_t<Bf16>(
          im.wb("talker.code_predictor.model.codec_embedding." +
                std::to_string(j - 1) + ".weight"),
          im.d_ids, im.hidden, 1, kH, im.stream);
      int pos = cp_len;
      UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_pos, &pos, sizeof(int),
                                     cudaMemcpyHostToDevice, im.stream));
      im.layer(im.hidden, 1, cp_len, kCpLayers, im.cp_kv, kCpSlots, true);
      ++cp_len;
      rms_norm_t<Bf16>(im.hidden, im.wb("talker.code_predictor.model.norm.weight"),
                       im.norm, 1, kH, kEps, im.stream);
      gemv_t<Bf16>(
          im.wb("talker.code_predictor.lm_head." + std::to_string(j) + ".weight"),
          im.norm, im.logits, kCpVocab, kH, im.stream);
      codes[j] = argmax_f32(im.logits, kCpVocab, im.d_argmax, im.stream);
    }
    return codes;
  };

  int c0 = select_c0(/*allow_eos=*/false, /*n_hist=*/0, /*ban_id=*/-1);
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
    if (c0 < 0 || c0 >= kCodecVocab)
      throw std::runtime_error("talker produced out-of-range codec id");
    if (seq_len + 1 > kTalkerRows)
      throw std::runtime_error("tts text too long for the talker buffers");
    UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_hist + step, &c0, sizeof(int),
                                   cudaMemcpyHostToDevice, im.stream));
    const std::vector<int> rest = run_predictor(im.hidden + (seq_len - 1) * kH, c0);
    frames.push_back(c0);
    if (step == 0 && timings != nullptr) timings->first_frame_ms = elapsed_ms();
    for (int x : rest) frames.push_back(x);
    const int frame_count = static_cast<int>(frames.size() / kGroups);
    if (on_chunk && frame_count >= next_stream_frame) {
      take(im.codec->decode(frames, frame_count, abort), frame_count);
      next_stream_frame = frame_count + kChunkFrames;
    }

    // Frame vector: codec_embed(c0) + 15 predictor embeddings + tts_pad.
    UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_ids, &c0, sizeof(int), cudaMemcpyHostToDevice,
                                   im.stream));
    gather_rows_t<Bf16>(im.wb("talker.model.codec_embedding.weight"), im.d_ids,
                        im.proj, 1, kH, im.stream);
    for (int j = 0; j < 15; ++j) {
      UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_ids, &rest[j], sizeof(int),
                                     cudaMemcpyHostToDevice, im.stream));
      gather_rows_t<Bf16>(
          im.wb("talker.code_predictor.model.codec_embedding." +
                std::to_string(j) + ".weight"),
          im.d_ids, im.hidden, 1, kH, im.stream);
      add_residual_t<Bf16>(im.proj, im.hidden, kH, im.stream);
    }
    add_residual_t<Bf16>(im.proj, im.d_pad, kH, im.stream);
    cudaMemcpyAsync(im.hidden + seq_len * kH, im.proj, kH * sizeof(Bf16),
                    cudaMemcpyDeviceToDevice, im.stream);

    int pos = seq_len;
    UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_pos, &pos, sizeof(int),
                                   cudaMemcpyHostToDevice, im.stream));
    im.layer(im.hidden + seq_len * kH, 1, seq_len, kTalkerLayers, im.talker_kv,
             kMaxSlots, false);
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
    const int next = select_c0(allow_eos, step + 1, ban_id);
    if (next == kCodecEos) break;
    c0 = next;
    c0_trace.push_back(c0);
  }
  if (const char* dump = std::getenv("UMM_TTS_DUMP_C0")) {
    std::ofstream f(dump, std::ios::binary);
    f.write(reinterpret_cast<const char*>(c0_trace.data()),
            c0_trace.size() * sizeof(int));
  }
  UMM_CUDA_CHECK(cudaStreamSynchronize(im.stream));
  // The waveform decoder checks between its own stages too; this boundary
  // covers the hand-off.
  if (abort && abort()) {
    throw RequestAborted("aborted before codec decode at frame " +
                         std::to_string(frames.size() / kGroups));
  }
  if (frames.size() % kGroups != 0)
    throw std::runtime_error("codec frames not divisible by 16");
  const int total_frames = static_cast<int>(frames.size() / kGroups);
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
  if (timings != nullptr) timings->total_ms = elapsed_ms();
  return samples;
}

bool stream_matches(const std::vector<float>& streamed,
                    const std::vector<float>& terminal,
                    size_t* first_diff) {
  size_t i = 0;
  const size_t common = std::min(streamed.size(), terminal.size());
  while (i < common && streamed[i] == terminal[i]) ++i;
  if (first_diff != nullptr) *first_diff = i;
  return streamed.size() == terminal.size() && i == common;
}

}  // namespace umminfer::native
