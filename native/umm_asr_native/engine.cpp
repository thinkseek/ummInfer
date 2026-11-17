// SPDX-License-Identifier: Apache-2.0
#include "engine.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "cuda_kernels.hpp"
#include "safetensors.hpp"
#include "tokenizer.hpp"
#include "wav.hpp"

namespace umminfer::native {
namespace {

constexpr int kMels = 128;
constexpr int kMelFrames = 3000;
constexpr int kAudioD = 896;
constexpr int kAudioHeads = 14;
constexpr int kAudioHeadDim = 64;
constexpr int kAudioLayers = 18;
constexpr int kAudioFfn = 3584;
constexpr int kAudioDown = 480;
constexpr int kAudioOut = 1024;
constexpr int kTextLayers = 28;
constexpr int kTextH = 1024;
constexpr int kTextQHeads = 16;
constexpr int kTextKvHeads = 8;
constexpr int kTextHeadDim = 128;
constexpr int kTextInter = 3072;
constexpr int kVocab = 151936;
constexpr float kEps = 1e-6f;
constexpr float kRopeTheta = 1000000.0f;
constexpr int kMaxSeq = 1024;

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::invalid_argument("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

struct Dense {
  const Bf16* w = nullptr;  // [out, in], row-major
  const Bf16* b = nullptr;  // [out], optional
  int in = 0, out = 0;
};

}  // namespace

struct AsrEngine::Impl {
  cudaStream_t stream = nullptr;
  int gpu_id = 0;

  std::unordered_map<std::string, Bf16*> weights;
  Bf16* d_data = nullptr;  // single checkpoint data region on device
  size_t data_bytes = 0;

  // Scratch / state.
  float *d_mel = nullptr, *d_inv_freq = nullptr, *d_logits = nullptr;
  int *d_ids = nullptr, *d_positions = nullptr, *d_argmax = nullptr;
  Bf16 *d_hidden = nullptr, *d_norm = nullptr, *d_q = nullptr, *d_k = nullptr,
      *d_v = nullptr, *d_proj = nullptr, *d_gate = nullptr, *d_up = nullptr,
      *d_audio = nullptr;
  // Audio tower workspaces (896-wide encoder, 3584 FFN, 30 conv chunks).
  Bf16 *d_au_norm = nullptr, *d_au_q = nullptr, *d_au_k = nullptr, *d_au_v = nullptr,
      *d_au_ffn = nullptr;
  Bf16 *d_conv_in = nullptr, *d_conv1 = nullptr, *d_conv2 = nullptr, *d_conv3 = nullptr;
  float* d_pe = nullptr;  // sinusoidal audio position table [13, 896]
  Bf16* kv = nullptr;     // [L][2][kMaxSeq][kTextKvHeads][kTextHeadDim]
  int seq_len = 0;

  const Bf16* w(const std::string& name) const {
    auto it = weights.find(name);
    if (it == weights.end()) throw std::runtime_error("checkpoint missing tensor: " + name);
    return it->second;
  }

  Dense dense(const std::string& prefix, int in, int out, bool bias) const {
    Dense d;
    d.in = in;
    d.out = out;
    d.w = w(prefix + ".weight");
    d.b = bias ? w(prefix + ".bias") : nullptr;
    return d;
  }

  // y [m, n] = x [m, k] * W^T [n, k], then bias.
  // cuBLAS 12.4 on this host crashes (illegal memory access, verified on GPU2
  // with minimal repros) on the shapes this model uses, so the slice uses a
  // self-contained BF16 GEMM. Correctness first; tile with shared memory when
  // longer audio makes the naive version a bottleneck.
  void linear(Bf16* y, const Bf16* x, const Dense& layer, int m) {
    gemm_bf16_naive(y, x, layer.w, layer.b, m, layer.in, layer.out, stream);
  }

  size_t kv_offset(int layer, bool is_v, int slot) const {
    return ((((static_cast<size_t>(layer) * 2) + (is_v ? 1 : 0)) * kMaxSeq + slot) *
            kTextKvHeads) *
           kTextHeadDim;
  }

  // Whisper-style audio tower. Returns the token count N; d_audio holds [N,1024].
  int audio_forward(int valid_frames, const AbortCheck& abort) {
    const int chunks = std::max(1, (valid_frames + 99) / 100);
    const int valid_tokens = AsrEngine::audio_token_count(valid_frames);
    mel_to_chunks(d_mel, d_conv_in, chunks, valid_frames, stream);
    const std::string p = "thinker.audio_tower.";
    conv3x3s2_gelu(d_conv_in, w(p + "conv2d1.weight"), w(p + "conv2d1.bias"), d_conv1,
                   chunks, 1, kAudioDown, 128, 100, stream);
    conv3x3s2_gelu(d_conv1, w(p + "conv2d2.weight"), w(p + "conv2d2.bias"), d_conv2,
                   chunks, kAudioDown, kAudioDown, 64, 50, stream);
    conv3x3s2_gelu(d_conv2, w(p + "conv2d3.weight"), w(p + "conv2d3.bias"), d_conv3,
                   chunks, kAudioDown, kAudioDown, 32, 25, stream);
    // [c,480,16,13] -> [c,13,7680] -> conv_out linear -> [c,13,896].
    conv_permute(d_conv3, d_au_norm, chunks, stream);
    const int tokens = chunks * 13;
    linear(d_audio, d_au_norm, dense(p + "conv_out", kAudioDown * 16, kAudioD, false),
           tokens);
    add_sinusoidal(d_audio, d_pe, chunks, 13, kAudioD, stream);

    auto attn = [&](int layer) {
      const std::string lp = p + "layers." + std::to_string(layer);
      layer_norm(d_audio, w(lp + ".self_attn_layer_norm.weight"),
                 w(lp + ".self_attn_layer_norm.bias"), d_au_norm, tokens, kAudioD, 1e-5f,
                 stream);
      linear(d_au_q, d_au_norm, dense(lp + ".self_attn.q_proj", kAudioD, kAudioD, true),
             tokens);
      linear(d_au_k, d_au_norm, dense(lp + ".self_attn.k_proj", kAudioD, kAudioD, true),
             tokens);
      linear(d_au_v, d_au_norm, dense(lp + ".self_attn.v_proj", kAudioD, kAudioD, true),
             tokens);
      // HF window_aftercnn = 13 * (n_window_infer / (n_window*2)) = 104; a
      // 55-token clip is one full-bidirectional group, a 196-token clip is
      // split into [0,104)+[104,196). Fixed 104-wide windows reproduce it.
      gqa_attention(d_au_q, d_au_k, d_au_v, d_au_norm, valid_tokens, valid_tokens,
                    kAudioHeads, kAudioHeads, kAudioHeadDim, false, 104, 0, stream);
      linear(d_au_q, d_au_norm, dense(lp + ".self_attn.out_proj", kAudioD, kAudioD, true),
             valid_tokens);
      add_residual(d_audio, d_au_q, valid_tokens * kAudioD, stream);

      layer_norm(d_audio, w(lp + ".final_layer_norm.weight"),
                 w(lp + ".final_layer_norm.bias"), d_au_norm, valid_tokens, kAudioD,
                 1e-5f, stream);
      linear(d_au_ffn, d_au_norm, dense(lp + ".fc1", kAudioD, kAudioFfn, true),
             valid_tokens);
      gelu_inplace(d_au_ffn, valid_tokens * kAudioFfn, stream);
      linear(d_au_norm, d_au_ffn, dense(lp + ".fc2", kAudioFfn, kAudioD, true),
             valid_tokens);
      add_residual(d_audio, d_au_norm, valid_tokens * kAudioD, stream);
    };
    for (int l = 0; l < kAudioLayers; ++l) {
      if (abort && abort()) {
        throw RequestAborted("aborted in audio tower layer " + std::to_string(l));
      }
      attn(l);
    }

    layer_norm(d_audio, w(p + "ln_post.weight"), w(p + "ln_post.bias"), d_au_norm,
               valid_tokens, kAudioD, 1e-5f, stream);
    linear(d_audio, d_au_norm, dense(p + "proj1", kAudioD, kAudioD, true), valid_tokens);
    gelu_inplace(d_audio, valid_tokens * kAudioD, stream);
    linear(d_au_norm, d_audio, dense(p + "proj2", kAudioD, kAudioOut, true), valid_tokens);
    cudaMemcpyAsync(d_audio, d_au_norm, valid_tokens * kAudioOut * sizeof(Bf16),
                    cudaMemcpyDeviceToDevice, stream);
    return valid_tokens;
  }

  // One text decoder layer over m tokens (prefill m=S, decode m=1).
  void text_layer(int layer, int m, int history) {
    const std::string lp = "thinker.model.layers." + std::to_string(layer);
    rms_norm(d_hidden, w(lp + ".input_layernorm.weight"), d_norm, m, kTextH, kEps, stream);
    linear(d_q, d_norm, dense(lp + ".self_attn.q_proj", kTextH,
                              kTextQHeads * kTextHeadDim, false),
           m);
    linear(d_k, d_norm, dense(lp + ".self_attn.k_proj", kTextH,
                              kTextKvHeads * kTextHeadDim, false),
           m);
    linear(d_v, d_norm, dense(lp + ".self_attn.v_proj", kTextH,
                              kTextKvHeads * kTextHeadDim, false),
           m);
    rms_norm(d_q, w(lp + ".self_attn.q_norm.weight"), d_q, m * kTextQHeads, kTextHeadDim,
             kEps, stream);
    rms_norm(d_k, w(lp + ".self_attn.k_norm.weight"), d_k, m * kTextKvHeads, kTextHeadDim,
             kEps, stream);
    rope_half(d_q, d_positions, d_inv_freq, m, kTextQHeads, kTextHeadDim, stream);
    rope_half(d_k, d_positions, d_inv_freq, m, kTextKvHeads, kTextHeadDim, stream);

    const size_t slot_bytes = m * kTextKvHeads * kTextHeadDim * sizeof(Bf16);
    cudaMemcpyAsync(kv + kv_offset(layer, false, history), d_k, slot_bytes,
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(kv + kv_offset(layer, true, history), d_v, slot_bytes,
                    cudaMemcpyDeviceToDevice, stream);
    gqa_attention(d_q, kv + kv_offset(layer, false, 0),
                  kv + kv_offset(layer, true, 0), d_norm, m, history + m, kTextQHeads,
                  kTextKvHeads, kTextHeadDim, true, 0, history, stream);
    linear(d_q, d_norm,
           dense(lp + ".self_attn.o_proj", kTextQHeads * kTextHeadDim, kTextH, false), m);
    add_residual(d_hidden, d_q, m * kTextH, stream);

    rms_norm(d_hidden, w(lp + ".post_attention_layernorm.weight"), d_norm, m, kTextH,
             kEps, stream);
    linear(d_gate, d_norm, dense(lp + ".mlp.gate_proj", kTextH, kTextInter, false), m);
    linear(d_up, d_norm, dense(lp + ".mlp.up_proj", kTextH, kTextInter, false), m);
    silu_mul(d_norm, d_gate, d_up, m * kTextInter, stream);
    linear(d_gate, d_norm, dense(lp + ".mlp.down_proj", kTextInter, kTextH, false), m);
    add_residual(d_hidden, d_gate, m * kTextH, stream);
  }
};

int AsrEngine::audio_token_count(int valid_frames) {
  // transformers _get_feat_extract_output_lengths, frame length in mel frames.
  if (valid_frames <= 0) return 0;
  const int leave = valid_frames % 100;
  const int feat = (leave - 1) / 2 + 1;
  return ((feat - 1) / 2 + 1 - 1) / 2 + 1 + (valid_frames / 100) * 13;
}

AsrEngine::AsrEngine(const std::string& model_dir, int gpu_id) {
  impl_ = std::make_unique<Impl>();
  Impl& im = *impl_;
  im.gpu_id = gpu_id;
  UMM_CUDA_CHECK(cudaSetDevice(gpu_id));
  UMM_CUDA_CHECK(cudaStreamCreate(&im.stream));

  // Fail loudly on an unsupported checkpoint instead of silently mis-running.
  const nlohmann::json cfg = nlohmann::json::parse(read_file(model_dir + "/config.json"));
  const auto& audio = cfg["thinker_config"]["audio_config"];
  const auto& text = cfg["thinker_config"]["text_config"];
  auto need = [](const nlohmann::json& j, const char* k, long long want) {
    if (j.value(k, -1) != want)
      throw std::runtime_error(std::string("unsupported config value: ") + k);
  };
  need(audio, "d_model", kAudioD);
  need(audio, "encoder_layers", kAudioLayers);
  need(audio, "encoder_attention_heads", kAudioHeads);
  need(audio, "num_mel_bins", kMels);
  need(text, "hidden_size", kTextH);
  need(text, "num_hidden_layers", kTextLayers);
  need(text, "num_attention_heads", kTextQHeads);
  need(text, "num_key_value_heads", kTextKvHeads);
  need(text, "head_dim", kTextHeadDim);
  need(text, "intermediate_size", kTextInter);
  need(text, "vocab_size", kVocab);

  SafeTensors st(read_file(model_dir + "/model.safetensors"));
  // Tensor byte_offsets are relative to the data region which starts after
  // the 8-byte length prefix and the JSON header.
  const size_t data_start = st.data_region_offset();
  const size_t region_bytes = st.bytes().size() - data_start;
  UMM_CUDA_CHECK(cudaMalloc(&im.d_data, region_bytes));
  UMM_CUDA_CHECK(cudaMemcpy(im.d_data, st.bytes().data() + data_start, region_bytes,
                            cudaMemcpyHostToDevice));
  for (const auto& [name, info] : st.tensors()) {
    if (info.dtype != "BF16") throw std::runtime_error("only BF16 weights supported: " + name);
    if (info.byte_offset % 2 != 0)
      throw std::runtime_error("BF16 tensor not 2-byte aligned: " + name);
    im.weights.emplace(name, im.d_data + info.byte_offset / 2);
  }

  // Workspaces.
  auto mk = [&](Bf16*& p, size_t n) { UMM_CUDA_CHECK(cudaMalloc(&p, n * sizeof(Bf16))); };
  UMM_CUDA_CHECK(cudaMalloc(&im.d_mel, kMels * kMelFrames * sizeof(float)));
  UMM_CUDA_CHECK(cudaMalloc(&im.d_inv_freq, kTextHeadDim / 2 * sizeof(float)));
  UMM_CUDA_CHECK(cudaMalloc(&im.d_logits, kVocab * sizeof(float)));
  UMM_CUDA_CHECK(cudaMalloc(&im.d_ids, kMaxSeq * sizeof(int)));
  UMM_CUDA_CHECK(cudaMalloc(&im.d_positions, kMaxSeq * sizeof(int)));
  UMM_CUDA_CHECK(cudaMalloc(&im.d_argmax, sizeof(int)));
  mk(im.d_hidden, kMaxSeq * kTextH);
  mk(im.d_norm, kMaxSeq * kTextH);
  mk(im.d_q, kMaxSeq * kTextQHeads * kTextHeadDim);
  mk(im.d_k, kMaxSeq * kTextKvHeads * kTextHeadDim);
  mk(im.d_v, kMaxSeq * kTextKvHeads * kTextHeadDim);
  mk(im.d_proj, kMaxSeq * kTextH);
  mk(im.d_gate, kMaxSeq * kTextInter);
  mk(im.d_up, kMaxSeq * kTextInter);
  mk(im.d_audio, kMaxSeq * kAudioOut);
  mk(im.d_au_norm, 390 * kAudioD);
  mk(im.d_au_q, 390 * kAudioD);
  mk(im.d_au_k, 390 * kAudioD);
  mk(im.d_au_v, 390 * kAudioD);
  mk(im.d_au_ffn, 390 * kAudioFfn);
  constexpr int kMaxChunks = 30;
  mk(im.d_conv_in, kMaxChunks * 1 * 128 * 100);
  mk(im.d_conv1, kMaxChunks * kAudioDown * 64 * 50);
  mk(im.d_conv2, kMaxChunks * kAudioDown * 32 * 25);
  mk(im.d_conv3, kMaxChunks * kAudioDown * 16 * 13);
  UMM_CUDA_CHECK(cudaMalloc(&im.d_pe, 13 * kAudioD * sizeof(float)));
  // SinusoidsPositionEmbedding: sin/cos split, max_timescale 10000.
  std::vector<float> pe(13 * kAudioD);
  for (int t = 0; t < 13; ++t) {
    for (int i = 0; i < kAudioD / 2; ++i) {
      const float inv = std::exp(-std::log(10000.0f) / (kAudioD / 2 - 1) * i);
      const float angle = t * inv;
      pe[t * kAudioD + i] = std::sin(angle);
      pe[t * kAudioD + kAudioD / 2 + i] = std::cos(angle);
    }
  }
  UMM_CUDA_CHECK(cudaMemcpy(im.d_pe, pe.data(), pe.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
  UMM_CUDA_CHECK(cudaMalloc(&im.kv, sizeof(Bf16) * kTextLayers * 2 * kMaxSeq *
                                          kTextKvHeads * kTextHeadDim));

  std::vector<float> inv(kTextHeadDim / 2);
  for (int i = 0; i < kTextHeadDim / 2; ++i)
    // HF: arange(0, head_dim, 2) / head_dim, rope_theta = 1e6.
    inv[i] = 1.0f / std::pow(kRopeTheta, 2.0f * i / kTextHeadDim);
  UMM_CUDA_CHECK(cudaMemcpy(im.d_inv_freq, inv.data(), inv.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
}

AsrEngine::~AsrEngine() {
  if (!impl_) return;
  Impl& im = *impl_;
  cudaFree(im.kv);
  cudaFree(im.d_pe);
  cudaFree(im.d_conv3);
  cudaFree(im.d_conv2);
  cudaFree(im.d_conv1);
  cudaFree(im.d_conv_in);
  cudaFree(im.d_au_ffn);
  cudaFree(im.d_au_v);
  cudaFree(im.d_au_k);
  cudaFree(im.d_au_q);
  cudaFree(im.d_au_norm);
  cudaFree(im.d_audio);
  cudaFree(im.d_up);
  cudaFree(im.d_gate);
  cudaFree(im.d_proj);
  cudaFree(im.d_v);
  cudaFree(im.d_k);
  cudaFree(im.d_q);
  cudaFree(im.d_norm);
  cudaFree(im.d_hidden);
  cudaFree(im.d_argmax);
  cudaFree(im.d_positions);
  cudaFree(im.d_ids);
  cudaFree(im.d_logits);
  cudaFree(im.d_inv_freq);
  cudaFree(im.d_mel);
  cudaFree(im.d_data);
  if (im.stream) cudaStreamDestroy(im.stream);
}

std::vector<int> AsrEngine::transcribe_tokens(const MelFeatures& mel, int max_new_tokens,
                                              const AbortCheck& abort) {
  Impl& im = *impl_;
  UMM_CUDA_CHECK(cudaMemcpy(im.d_mel, mel.data.data(),
                            mel.data.size() * sizeof(float), cudaMemcpyHostToDevice));
  const int n_audio = im.audio_forward(mel.valid_frames, abort);
  const auto prompt = QwenTokenizer::build_asr_prompt(n_audio);
  const int S = static_cast<int>(prompt.size());
  if (S + max_new_tokens > kMaxSeq)
    throw std::runtime_error("sequence exceeds fixed KV capacity 1024");

  UMM_CUDA_CHECK(cudaMemcpy(im.d_ids, prompt.data(), S * sizeof(int),
                            cudaMemcpyHostToDevice));
  std::vector<int> positions(S);
  std::iota(positions.begin(), positions.end(), 0);
  UMM_CUDA_CHECK(cudaMemcpy(im.d_positions, positions.data(), S * sizeof(int),
                            cudaMemcpyHostToDevice));
  index_embeddings(im.w("thinker.model.embed_tokens.weight"), im.d_ids, im.d_hidden, S,
                   kTextH, im.stream);
  // prompt[3] is audio_start; placeholders start at offset 4.
  replace_rows(im.d_hidden, im.d_audio, 4, n_audio, kTextH, im.stream);
  im.seq_len = S;
  for (int l = 0; l < kTextLayers; ++l) {
    if (abort && abort()) {
      throw RequestAborted("aborted in prefill layer " + std::to_string(l));
    }
    im.text_layer(l, S, 0);
  }

  auto lm_head = [&](int row) {
    // Only the final token's logits are needed.
    rms_norm(im.d_hidden + static_cast<size_t>(row) * kTextH,
             im.w("thinker.model.norm.weight"), im.d_norm, 1, kTextH, kEps, im.stream);
    gemv_bf16(im.w("thinker.lm_head.weight"), im.d_norm, im.d_logits, kVocab, kTextH,
              im.stream);
  };
  lm_head(S - 1);
  int next = argmax_f32(im.d_logits, kVocab, im.d_argmax, im.stream);

  std::vector<int> generated;
  for (int step = 0; step < max_new_tokens && !QwenTokenizer::is_eos(next); ++step) {
    if (abort && abort()) {
      throw RequestAborted("aborted at token " + std::to_string(step));
    }
    generated.push_back(next);
    UMM_CUDA_CHECK(cudaMemcpy(im.d_ids, &next, sizeof(int), cudaMemcpyHostToDevice));
    const int pos = im.seq_len;
    UMM_CUDA_CHECK(cudaMemcpy(im.d_positions, &pos, sizeof(int), cudaMemcpyHostToDevice));
    index_embeddings(im.w("thinker.model.embed_tokens.weight"), im.d_ids, im.d_hidden, 1,
                     kTextH, im.stream);
    for (int l = 0; l < kTextLayers; ++l) im.text_layer(l, 1, im.seq_len);
    ++im.seq_len;
    lm_head(0);
    next = argmax_f32(im.d_logits, kVocab, im.d_argmax, im.stream);
  }
  UMM_CUDA_CHECK(cudaStreamSynchronize(im.stream));
  return generated;
}

}  // namespace umminfer::native
