// SPDX-License-Identifier: Apache-2.0
#include "engine.hpp"

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
#include "model_config.hpp"
#include "safetensors.hpp"
#include "tokenizer.hpp"
#include "wav.hpp"

namespace umminfer::native {
namespace {

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
  AsrModelConfig cfg;
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
  // Audio tower workspaces (audio_d-wide encoder, audio_ffn FFN, 30 conv
  // chunks), all sized from Impl::cfg.
  Bf16 *d_au_norm = nullptr, *d_au_q = nullptr, *d_au_k = nullptr, *d_au_v = nullptr,
      *d_au_ffn = nullptr;
  Bf16 *d_conv_in = nullptr, *d_conv1 = nullptr, *d_conv2 = nullptr, *d_conv3 = nullptr;
  float* d_pe = nullptr;  // sinusoidal audio position table [13, audio_d]
  Bf16* kv = nullptr;     // [L][2][max_seq][text_kv_heads][text_head_dim]
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
    return ((((static_cast<size_t>(layer) * 2) + (is_v ? 1 : 0)) * cfg.max_seq + slot) *
            cfg.text_kv_heads) *
           cfg.text_head_dim;
  }

  // Whisper-style audio tower. Returns the token count N; d_audio holds
  // [N, audio_out].
  int audio_forward(int valid_frames, const AbortCheck& abort) {
    const int chunks = std::max(1, (valid_frames + AsrModelConfig::kChunkFrames - 1) /
                                       AsrModelConfig::kChunkFrames);
    const int valid_tokens = AsrEngine::audio_token_count(valid_frames);
    const int per_chunk = AsrModelConfig::kConvTokensPerChunk;
    mel_to_chunks(d_mel, d_conv_in, chunks, valid_frames, stream);
    const std::string p = "thinker.audio_tower.";
    conv3x3s2_gelu(d_conv_in, w(p + "conv2d1.weight"), w(p + "conv2d1.bias"), d_conv1,
                   chunks, 1, cfg.audio_down, cfg.num_mel_bins,
                   AsrModelConfig::kChunkFrames, stream);
    conv3x3s2_gelu(d_conv1, w(p + "conv2d2.weight"), w(p + "conv2d2.bias"), d_conv2,
                   chunks, cfg.audio_down, cfg.audio_down, AsrModelConfig::kConv1H,
                   AsrModelConfig::kConv1W, stream);
    conv3x3s2_gelu(d_conv2, w(p + "conv2d3.weight"), w(p + "conv2d3.bias"), d_conv3,
                   chunks, cfg.audio_down, cfg.audio_down, AsrModelConfig::kConv2H,
                   AsrModelConfig::kConv2W, stream);
    // [c,down,16,13] -> [c,13,down*16] -> conv_out linear -> [c,13,audio_d].
    // conv_permute is unbounded by design; the workspace must hold its output.
    const int tokens = chunks * per_chunk;
    if (tokens > cfg.max_audio_tokens())
      throw std::runtime_error("audio tower tokens exceed workspace: " +
                               std::to_string(tokens) + " > " +
                               std::to_string(cfg.max_audio_tokens()));
    conv_permute(d_conv3, d_au_norm, chunks, cfg.audio_down, stream);
    linear(d_audio, d_au_norm, dense(p + "conv_out", cfg.audio_perm_row(), cfg.audio_d,
                                     false),
           tokens);
    add_sinusoidal(d_audio, d_pe, chunks, per_chunk, cfg.audio_d, stream);

    auto attn = [&](int layer) {
      const std::string lp = p + "layers." + std::to_string(layer);
      layer_norm(d_audio, w(lp + ".self_attn_layer_norm.weight"),
                 w(lp + ".self_attn_layer_norm.bias"), d_au_norm, tokens, cfg.audio_d,
                 cfg.audio_norm_eps, stream);
      linear(d_au_q, d_au_norm,
             dense(lp + ".self_attn.q_proj", cfg.audio_d, cfg.audio_d, true), tokens);
      linear(d_au_k, d_au_norm,
             dense(lp + ".self_attn.k_proj", cfg.audio_d, cfg.audio_d, true), tokens);
      linear(d_au_v, d_au_norm,
             dense(lp + ".self_attn.v_proj", cfg.audio_d, cfg.audio_d, true), tokens);
      // HF window_aftercnn = 13 * (n_window_infer / (n_window*2)) = 104; a
      // 55-token clip is one full-bidirectional group, a 196-token clip is
      // split into [0,104)+[104,196). Fixed 104-wide windows reproduce it.
      gqa_attention(d_au_q, d_au_k, d_au_v, d_au_norm, valid_tokens, valid_tokens,
                    cfg.audio_heads, cfg.audio_heads, cfg.audio_head_dim(), false,
                    cfg.window_len(), 0, stream);
      linear(d_au_q, d_au_norm,
             dense(lp + ".self_attn.out_proj", cfg.audio_d, cfg.audio_d, true),
             valid_tokens);
      add_residual(d_audio, d_au_q, valid_tokens * cfg.audio_d, stream);

      layer_norm(d_audio, w(lp + ".final_layer_norm.weight"),
                 w(lp + ".final_layer_norm.bias"), d_au_norm, valid_tokens, cfg.audio_d,
                 cfg.audio_norm_eps, stream);
      linear(d_au_ffn, d_au_norm, dense(lp + ".fc1", cfg.audio_d, cfg.audio_ffn, true),
             valid_tokens);
      gelu_inplace(d_au_ffn, valid_tokens * cfg.audio_ffn, stream);
      linear(d_au_norm, d_au_ffn,
             dense(lp + ".fc2", cfg.audio_ffn, cfg.audio_d, true), valid_tokens);
      add_residual(d_audio, d_au_norm, valid_tokens * cfg.audio_d, stream);
    };
    for (int l = 0; l < cfg.audio_layers; ++l) {
      if (abort && abort()) {
        throw RequestAborted("aborted in audio tower layer " + std::to_string(l));
      }
      attn(l);
    }

    layer_norm(d_audio, w(p + "ln_post.weight"), w(p + "ln_post.bias"), d_au_norm,
               valid_tokens, cfg.audio_d, cfg.audio_norm_eps, stream);
    linear(d_audio, d_au_norm, dense(p + "proj1", cfg.audio_d, cfg.audio_d, true),
           valid_tokens);
    gelu_inplace(d_audio, valid_tokens * cfg.audio_d, stream);
    linear(d_au_norm, d_audio, dense(p + "proj2", cfg.audio_d, cfg.audio_out, true),
           valid_tokens);
    cudaMemcpyAsync(d_audio, d_au_norm, valid_tokens * cfg.audio_out * sizeof(Bf16),
                    cudaMemcpyDeviceToDevice, stream);
    return valid_tokens;
  }

  // One text decoder layer over m tokens (prefill m=S, decode m=1).
  void text_layer(int layer, int m, int history) {
    const std::string lp = "thinker.model.layers." + std::to_string(layer);
    const int q_dim = cfg.text_q_heads * cfg.text_head_dim;
    const int kv_dim = cfg.text_kv_heads * cfg.text_head_dim;
    rms_norm(d_hidden, w(lp + ".input_layernorm.weight"), d_norm, m, cfg.text_h,
             cfg.rms_norm_eps, stream);
    linear(d_q, d_norm, dense(lp + ".self_attn.q_proj", cfg.text_h, q_dim, false), m);
    linear(d_k, d_norm, dense(lp + ".self_attn.k_proj", cfg.text_h, kv_dim, false), m);
    linear(d_v, d_norm, dense(lp + ".self_attn.v_proj", cfg.text_h, kv_dim, false), m);
    rms_norm(d_q, w(lp + ".self_attn.q_norm.weight"), d_q, m * cfg.text_q_heads,
             cfg.text_head_dim, cfg.rms_norm_eps, stream);
    rms_norm(d_k, w(lp + ".self_attn.k_norm.weight"), d_k, m * cfg.text_kv_heads,
             cfg.text_head_dim, cfg.rms_norm_eps, stream);
    rope_half(d_q, d_positions, d_inv_freq, m, cfg.text_q_heads, cfg.text_head_dim,
              stream);
    rope_half(d_k, d_positions, d_inv_freq, m, cfg.text_kv_heads, cfg.text_head_dim,
              stream);

    const size_t slot_bytes = m * cfg.text_kv_heads * cfg.text_head_dim * sizeof(Bf16);
    cudaMemcpyAsync(kv + kv_offset(layer, false, history), d_k, slot_bytes,
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(kv + kv_offset(layer, true, history), d_v, slot_bytes,
                    cudaMemcpyDeviceToDevice, stream);
    gqa_attention(d_q, kv + kv_offset(layer, false, 0),
                  kv + kv_offset(layer, true, 0), d_norm, m, history + m, cfg.text_q_heads,
                  cfg.text_kv_heads, cfg.text_head_dim, true, 0, history, stream);
    linear(d_q, d_norm, dense(lp + ".self_attn.o_proj", q_dim, cfg.text_h, false), m);
    add_residual(d_hidden, d_q, m * cfg.text_h, stream);

    rms_norm(d_hidden, w(lp + ".post_attention_layernorm.weight"), d_norm, m, cfg.text_h,
             cfg.rms_norm_eps, stream);
    linear(d_gate, d_norm, dense(lp + ".mlp.gate_proj", cfg.text_h, cfg.text_inter, false),
           m);
    linear(d_up, d_norm, dense(lp + ".mlp.up_proj", cfg.text_h, cfg.text_inter, false), m);
    silu_mul(d_norm, d_gate, d_up, m * cfg.text_inter, stream);
    linear(d_gate, d_norm, dense(lp + ".mlp.down_proj", cfg.text_inter, cfg.text_h, false),
           m);
    add_residual(d_hidden, d_gate, m * cfg.text_h, stream);
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
  im.cfg = AsrModelConfig::from_model_dir(model_dir);
  const AsrModelConfig& c = im.cfg;
  const int per_chunk = AsrModelConfig::kConvTokensPerChunk;
  UMM_CUDA_CHECK(cudaSetDevice(gpu_id));
  UMM_CUDA_CHECK(cudaStreamCreate(&im.stream));

  SafeTensors st(read_file(model_dir + "/model.safetensors"));
  // Fail loudly when the checkpoint does not match config.json: a wrong shape
  // would otherwise read past the tensor or silently mis-run.
  auto expect = [&](const std::string& name, std::vector<size_t> want) {
    const std::vector<size_t> got = st.info(name).shape;
    if (got == want) return;
    auto join = [](const std::vector<size_t>& v) {
      std::string s;
      for (size_t d : v) s += (s.empty() ? "" : "x") + std::to_string(d);
      return s;
    };
    throw std::runtime_error("weight shape mismatch for " + name + ": config.json implies " +
                             join(want) + ", checkpoint has " + join(got));
  };
  const size_t audio_d = static_cast<size_t>(c.audio_d);
  const size_t audio_perm = static_cast<size_t>(c.audio_perm_row());
  const size_t audio_down = static_cast<size_t>(c.audio_down);
  const size_t text_h = static_cast<size_t>(c.text_h);
  const size_t q_dim = static_cast<size_t>(c.text_q_heads) * c.text_head_dim;
  const size_t kv_dim = static_cast<size_t>(c.text_kv_heads) * c.text_head_dim;
  const std::string at = "thinker.audio_tower.";
  expect(at + "conv2d1.weight", {audio_down, 1, 3, 3});
  expect(at + "conv2d2.weight", {audio_down, audio_down, 3, 3});
  expect(at + "conv2d3.weight", {audio_down, audio_down, 3, 3});
  expect(at + "conv_out.weight", {audio_d, audio_perm});
  expect(at + "ln_post.weight", {audio_d});
  expect(at + "proj1.weight", {audio_d, audio_d});
  expect(at + "proj2.weight", {static_cast<size_t>(c.audio_out), audio_d});
  for (int l : {0, c.audio_layers - 1}) {
    const std::string lp = at + "layers." + std::to_string(l) + ".";
    expect(lp + "self_attn.q_proj.weight", {audio_d, audio_d});
    expect(lp + "self_attn.k_proj.weight", {audio_d, audio_d});
    expect(lp + "self_attn.v_proj.weight", {audio_d, audio_d});
    expect(lp + "self_attn.out_proj.weight", {audio_d, audio_d});
    expect(lp + "self_attn_layer_norm.weight", {audio_d});
    expect(lp + "final_layer_norm.weight", {audio_d});
    expect(lp + "fc1.weight", {static_cast<size_t>(c.audio_ffn), audio_d});
    expect(lp + "fc2.weight", {audio_d, static_cast<size_t>(c.audio_ffn)});
  }
  expect("thinker.model.embed_tokens.weight", {static_cast<size_t>(c.vocab), text_h});
  expect("thinker.model.norm.weight", {text_h});
  expect("thinker.lm_head.weight", {static_cast<size_t>(c.vocab), text_h});
  for (int l : {0, c.text_layers - 1}) {
    const std::string lp = "thinker.model.layers." + std::to_string(l) + ".";
    expect(lp + "input_layernorm.weight", {text_h});
    expect(lp + "self_attn.q_proj.weight", {q_dim, text_h});
    expect(lp + "self_attn.k_proj.weight", {kv_dim, text_h});
    expect(lp + "self_attn.v_proj.weight", {kv_dim, text_h});
    expect(lp + "self_attn.q_norm.weight", {static_cast<size_t>(c.text_head_dim)});
    expect(lp + "self_attn.k_norm.weight", {static_cast<size_t>(c.text_head_dim)});
    expect(lp + "self_attn.o_proj.weight", {text_h, q_dim});
    expect(lp + "post_attention_layernorm.weight", {text_h});
    expect(lp + "mlp.gate_proj.weight", {static_cast<size_t>(c.text_inter), text_h});
    expect(lp + "mlp.up_proj.weight", {static_cast<size_t>(c.text_inter), text_h});
    expect(lp + "mlp.down_proj.weight", {text_h, static_cast<size_t>(c.text_inter)});
  }

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

  // Workspaces, every size derived from config.json.
  auto mk = [&](Bf16*& p, size_t n) { UMM_CUDA_CHECK(cudaMalloc(&p, n * sizeof(Bf16))); };
  UMM_CUDA_CHECK(cudaMalloc(&im.d_mel, c.mel_elements() * sizeof(float)));
  UMM_CUDA_CHECK(cudaMalloc(&im.d_inv_freq, c.inv_freq_elements() * sizeof(float)));
  UMM_CUDA_CHECK(cudaMalloc(&im.d_logits, c.logits_elements() * sizeof(float)));
  UMM_CUDA_CHECK(cudaMalloc(&im.d_ids, c.max_seq * sizeof(int)));
  UMM_CUDA_CHECK(cudaMalloc(&im.d_positions, c.max_seq * sizeof(int)));
  UMM_CUDA_CHECK(cudaMalloc(&im.d_argmax, sizeof(int)));
  mk(im.d_hidden, c.text_row_elements());
  mk(im.d_norm, c.text_row_elements());
  mk(im.d_q, c.text_q_elements());
  mk(im.d_k, c.text_kv_elements());
  mk(im.d_v, c.text_kv_elements());
  mk(im.d_proj, c.text_row_elements());
  mk(im.d_gate, c.text_ffn_elements());
  mk(im.d_up, c.text_ffn_elements());
  mk(im.d_audio, c.audio_io_elements());
  mk(im.d_au_norm, c.au_norm_elements());
  mk(im.d_au_q, c.au_qkv_elements());
  mk(im.d_au_k, c.au_qkv_elements());
  mk(im.d_au_v, c.au_qkv_elements());
  mk(im.d_au_ffn, c.au_ffn_elements());
  mk(im.d_conv_in, c.conv_in_elements());
  mk(im.d_conv1, c.conv1_elements());
  mk(im.d_conv2, c.conv2_elements());
  mk(im.d_conv3, c.conv3_elements());
  UMM_CUDA_CHECK(cudaMalloc(&im.d_pe, c.pe_elements() * sizeof(float)));
  // SinusoidsPositionEmbedding: sin/cos split, max_timescale 10000.
  const int audio_dim = c.audio_d;
  std::vector<float> pe(static_cast<size_t>(per_chunk) * audio_dim);
  for (int t = 0; t < per_chunk; ++t) {
    for (int i = 0; i < audio_dim / 2; ++i) {
      const float inv = std::exp(-std::log(10000.0f) / (audio_dim / 2 - 1) * i);
      const float angle = t * inv;
      pe[t * audio_dim + i] = std::sin(angle);
      pe[t * audio_dim + audio_dim / 2 + i] = std::cos(angle);
    }
  }
  UMM_CUDA_CHECK(cudaMemcpy(im.d_pe, pe.data(), pe.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
  UMM_CUDA_CHECK(cudaMalloc(&im.kv, sizeof(Bf16) * c.kv_cache_elements()));

  std::vector<float> inv(static_cast<size_t>(c.inv_freq_elements()));
  for (int i = 0; i < c.inv_freq_elements(); ++i)
    // HF: arange(0, head_dim, 2) / head_dim, rope_theta from config.
    inv[i] = 1.0f / std::pow(c.rope_theta, 2.0f * i / c.text_head_dim);
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
  const AsrModelConfig& c = im.cfg;
  UMM_CUDA_CHECK(cudaMemcpy(im.d_mel, mel.data.data(),
                            mel.data.size() * sizeof(float), cudaMemcpyHostToDevice));
  const int n_audio = im.audio_forward(mel.valid_frames, abort);
  const auto prompt = QwenTokenizer::build_asr_prompt(n_audio, c.audio_start_id,
                                                      c.audio_token_id, c.audio_end_id);
  const int S = static_cast<int>(prompt.size());
  if (S + max_new_tokens > c.max_seq)
    throw std::runtime_error("sequence exceeds fixed KV capacity " +
                             std::to_string(c.max_seq));

  UMM_CUDA_CHECK(cudaMemcpy(im.d_ids, prompt.data(), S * sizeof(int),
                            cudaMemcpyHostToDevice));
  std::vector<int> positions(S);
  std::iota(positions.begin(), positions.end(), 0);
  UMM_CUDA_CHECK(cudaMemcpy(im.d_positions, positions.data(), S * sizeof(int),
                            cudaMemcpyHostToDevice));
  index_embeddings(im.w("thinker.model.embed_tokens.weight"), im.d_ids, im.d_hidden, S,
                   c.text_h, im.stream);
  // prompt[3] is audio_start; placeholders start at offset 4.
  replace_rows(im.d_hidden, im.d_audio, 4, n_audio, c.text_h, im.stream);
  im.seq_len = S;
  for (int l = 0; l < c.text_layers; ++l) {
    if (abort && abort()) {
      throw RequestAborted("aborted in prefill layer " + std::to_string(l));
    }
    im.text_layer(l, S, 0);
  }

  auto lm_head = [&](int row) {
    // Only the final token's logits are needed.
    rms_norm(im.d_hidden + static_cast<size_t>(row) * c.text_h,
             im.w("thinker.model.norm.weight"), im.d_norm, 1, c.text_h, c.rms_norm_eps,
             im.stream);
    gemv_bf16(im.w("thinker.lm_head.weight"), im.d_norm, im.d_logits, c.vocab, c.text_h,
              im.stream);
  };
  lm_head(S - 1);
  int next = argmax_f32(im.d_logits, c.vocab, im.d_argmax, im.stream);

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
                     c.text_h, im.stream);
    for (int l = 0; l < c.text_layers; ++l) im.text_layer(l, 1, im.seq_len);
    ++im.seq_len;
    lm_head(0);
    next = argmax_f32(im.d_logits, c.vocab, im.d_argmax, im.stream);
  }
  UMM_CUDA_CHECK(cudaStreamSynchronize(im.stream));
  return generated;
}

}  // namespace umminfer::native
