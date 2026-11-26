// SPDX-License-Identifier: Apache-2.0
#include "engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/cuda_request_allocator.hpp"
#include "cuda_kernels.hpp"
#include "model_config.hpp"
#include "safetensors.hpp"
#include "tokenizer.hpp"
#include "wav.hpp"

namespace umminfer::native {
namespace {

bool env_flag_enabled(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && v[0] != '\0' && std::string(v) != "0";
}

// Pre-resolved weight pointers for one Whisper-style audio encoder block.
struct AudioLayerWeights {
  const Bf16* attn_ln_w = nullptr;
  const Bf16* attn_ln_b = nullptr;
  const Bf16* q_w = nullptr;
  const Bf16* q_b = nullptr;
  const Bf16* k_w = nullptr;
  const Bf16* k_b = nullptr;
  const Bf16* v_w = nullptr;
  const Bf16* v_b = nullptr;
  const Bf16* out_w = nullptr;
  const Bf16* out_b = nullptr;
  const Bf16* final_ln_w = nullptr;
  const Bf16* final_ln_b = nullptr;
  const Bf16* fc1_w = nullptr;
  const Bf16* fc1_b = nullptr;
  const Bf16* fc2_w = nullptr;
  const Bf16* fc2_b = nullptr;
};

struct AudioTowerWeights {
  const Bf16* conv1_w = nullptr;
  const Bf16* conv1_b = nullptr;
  const Bf16* conv2_w = nullptr;
  const Bf16* conv2_b = nullptr;
  const Bf16* conv3_w = nullptr;
  const Bf16* conv3_b = nullptr;
  const Bf16* conv_out_w = nullptr;
  std::vector<AudioLayerWeights> layers;
  const Bf16* ln_post_w = nullptr;
  const Bf16* ln_post_b = nullptr;
  const Bf16* proj1_w = nullptr;
  const Bf16* proj1_b = nullptr;
  const Bf16* proj2_w = nullptr;
  const Bf16* proj2_b = nullptr;
};

// Pre-resolved weight pointers for one Qwen3 dense text decoder layer.
struct TextLayerWeights {
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

}  // namespace

struct AsrEngine::Impl {
  AsrModelConfig cfg;
  cudaStream_t stream = nullptr;
  int gpu_id = 0;

  std::unordered_map<std::string, Bf16*> weights;
  std::vector<Bf16*> d_shards;  // one checkpoint data region per shard on device

  // Pre-resolved weight tables (zero string/hash-map lookups in hot paths).
  AudioTowerWeights audio_w;
  std::vector<TextLayerWeights> text_layers;
  const Bf16* embed_tokens_w = nullptr;
  const Bf16* final_norm_w = nullptr;
  const Bf16* lm_head_w = nullptr;

  // Device scratch / state.
  float *d_mel = nullptr, *d_inv_freq = nullptr, *d_logits = nullptr;
  int *d_ids = nullptr, *d_positions = nullptr, *d_step_pos = nullptr,
      *d_argmax = nullptr;
  Bf16 *d_hidden = nullptr, *d_norm = nullptr, *d_q = nullptr, *d_k = nullptr,
      *d_v = nullptr, *d_gate = nullptr, *d_up = nullptr, *d_audio = nullptr;
  // Audio tower workspaces (audio_d-wide encoder, audio_ffn FFN, 30 conv
  // chunks), all sized from Impl::cfg.
  Bf16 *d_au_norm = nullptr, *d_au_q = nullptr, *d_au_k = nullptr, *d_au_v = nullptr,
      *d_au_ffn = nullptr;
  Bf16 *d_conv_in = nullptr, *d_conv1 = nullptr, *d_conv2 = nullptr, *d_conv3 = nullptr;
  float* d_pe = nullptr;  // sinusoidal audio position table [13, audio_d]
  Bf16* kv = nullptr;     // [L][2][max_seq][text_kv_heads][text_head_dim]
  int seq_len = 0;

  // Pinned host staging buffers (zero extra VRAM; eliminates pageable bounce
  // copies and implicit default-stream synchronizations).
  float* h_mel_pinned = nullptr;
  int* h_ids_pinned = nullptr;
  int* h_argmax_pinned = nullptr;
  int* h_step_pos_pinned = nullptr;

  // Captured CUDA Graph for the fixed-shape 28-layer decode step + LM head +
  // argmax. Null when UMM_ASR_NO_GRAPH=1, head_dim != 128, or capture failed.
  cudaGraphExec_t decode_graph_exec = nullptr;

  ~Impl() {
    if (decode_graph_exec) cudaGraphExecDestroy(decode_graph_exec);
    if (h_step_pos_pinned) cudaFreeHost(h_step_pos_pinned);
    if (h_argmax_pinned) cudaFreeHost(h_argmax_pinned);
    if (h_ids_pinned) cudaFreeHost(h_ids_pinned);
    if (h_mel_pinned) cudaFreeHost(h_mel_pinned);
    request_cuda_free(kv);
    request_cuda_free(d_pe);
    request_cuda_free(d_conv3);
    request_cuda_free(d_conv2);
    request_cuda_free(d_conv1);
    request_cuda_free(d_conv_in);
    request_cuda_free(d_au_ffn);
    request_cuda_free(d_au_v);
    request_cuda_free(d_au_k);
    request_cuda_free(d_au_q);
    request_cuda_free(d_au_norm);
    request_cuda_free(d_audio);
    request_cuda_free(d_up);
    request_cuda_free(d_gate);
    request_cuda_free(d_v);
    request_cuda_free(d_k);
    request_cuda_free(d_q);
    request_cuda_free(d_norm);
    request_cuda_free(d_hidden);
    request_cuda_free(d_argmax);
    request_cuda_free(d_step_pos);
    request_cuda_free(d_positions);
    request_cuda_free(d_ids);
    request_cuda_free(d_logits);
    request_cuda_free(d_inv_freq);
    request_cuda_free(d_mel);
    for (Bf16* shard : d_shards) request_cuda_free(shard);
    if (stream) cudaStreamDestroy(stream);
  }

  const Bf16* w(const std::string& name) const {
    auto it = weights.find(name);
    if (it == weights.end()) throw std::runtime_error("checkpoint missing tensor: " + name);
    return it->second;
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
    const AudioTowerWeights& au = audio_w;

    mel_to_chunks(d_mel, d_conv_in, chunks, valid_frames, stream);
    conv3x3s2_gelu(d_conv_in, au.conv1_w, au.conv1_b, d_conv1, chunks, 1,
                   cfg.audio_down, cfg.num_mel_bins,
                   AsrModelConfig::kChunkFrames, stream);
    conv3x3s2_gelu(d_conv1, au.conv2_w, au.conv2_b, d_conv2, chunks,
                   cfg.audio_down, cfg.audio_down, AsrModelConfig::kConv1H,
                   AsrModelConfig::kConv1W, stream);
    conv3x3s2_gelu(d_conv2, au.conv3_w, au.conv3_b, d_conv3, chunks,
                   cfg.audio_down, cfg.audio_down, AsrModelConfig::kConv2H,
                   AsrModelConfig::kConv2W, stream);
    // [c,down,16,13] -> [c,13,down*16] -> conv_out linear -> [c,13,audio_d].
    const int tokens = chunks * per_chunk;
    if (tokens > cfg.max_audio_tokens())
      throw std::runtime_error("audio tower tokens exceed workspace: " +
                               std::to_string(tokens) + " > " +
                               std::to_string(cfg.max_audio_tokens()));
    conv_permute(d_conv3, d_au_norm, chunks, cfg.audio_down, stream);
    gemm_bf16(d_audio, d_au_norm, au.conv_out_w, nullptr, tokens,
              cfg.audio_perm_row(), cfg.audio_d, stream);
    add_sinusoidal(d_audio, d_pe, chunks, per_chunk, cfg.audio_d, stream);

    const int audio_head_dim = cfg.audio_head_dim();
    const int window_len = cfg.window_len();
    for (int l = 0; l < cfg.audio_layers; ++l) {
      if (abort && abort()) {
        throw RequestAborted("aborted in audio tower layer " + std::to_string(l));
      }
      const AudioLayerWeights& al = au.layers[l];
      layer_norm(d_audio, al.attn_ln_w, al.attn_ln_b, d_au_norm, valid_tokens,
                 cfg.audio_d, cfg.audio_norm_eps, stream);
      QkvGemvDesc qkv_desc{{al.q_w, al.k_w, al.v_w},
                           {al.q_b, al.k_b, al.v_b},
                           {d_au_q, d_au_k, d_au_v},
                           {cfg.audio_d, cfg.audio_d, cfg.audio_d}};
      gemv_qkv_bf16(qkv_desc, d_au_norm, valid_tokens, cfg.audio_d, stream);
      gqa_attention(d_au_q, d_au_k, d_au_v, d_au_norm, valid_tokens,
                    valid_tokens, cfg.audio_heads, cfg.audio_heads,
                    audio_head_dim, false, window_len, 0, stream);
      gemm_bf16_add_residual(d_audio, d_au_norm, al.out_w, al.out_b,
                             valid_tokens, cfg.audio_d, cfg.audio_d, stream);

      layer_norm(d_audio, al.final_ln_w, al.final_ln_b, d_au_norm, valid_tokens,
                 cfg.audio_d, cfg.audio_norm_eps, stream);
      gemm_bf16_gelu(d_au_ffn, d_au_norm, al.fc1_w, al.fc1_b, valid_tokens,
                     cfg.audio_d, cfg.audio_ffn, stream);
      gemm_bf16_add_residual(d_audio, d_au_ffn, al.fc2_w, al.fc2_b,
                             valid_tokens, cfg.audio_ffn, cfg.audio_d, stream);
    }

    layer_norm(d_audio, au.ln_post_w, au.ln_post_b, d_au_norm, valid_tokens,
               cfg.audio_d, cfg.audio_norm_eps, stream);
    // Write proj1+GELU into d_au_q and proj2 directly into d_audio (no D2D copy).
    gemm_bf16_gelu(d_au_q, d_au_norm, au.proj1_w, au.proj1_b, valid_tokens,
                   cfg.audio_d, cfg.audio_d, stream);
    gemm_bf16(d_audio, d_au_q, au.proj2_w, au.proj2_b, valid_tokens,
              cfg.audio_d, cfg.audio_out, stream);
    return valid_tokens;
  }

  // One text decoder layer during prefill (m = S, history = 0).
  void text_layer_prefill(int layer, int m) {
    const TextLayerWeights& lw = text_layers[layer];
    const int q_dim = cfg.text_q_heads * cfg.text_head_dim;
    const int kv_dim = cfg.text_kv_heads * cfg.text_head_dim;
    Bf16* k_cache_base = kv + kv_offset(layer, false, 0);
    Bf16* v_cache_base = kv + kv_offset(layer, true, 0);

    rms_norm(d_hidden, lw.input_ln, d_norm, m, cfg.text_h, cfg.rms_norm_eps,
             stream);
    if (cfg.text_head_dim == 128) {
      // Write V directly into v_cache[0..m-1], eliminating D2D copy.
      QkvGemvDesc qkv_desc{{lw.q_proj, lw.k_proj, lw.v_proj},
                           {nullptr, nullptr, nullptr},
                           {d_q, d_k, v_cache_base},
                           {q_dim, kv_dim, kv_dim}};
      gemv_qkv_bf16(qkv_desc, d_norm, m, cfg.text_h, stream);
      qk_norm_rope_kv_append_bf16(
          d_q, d_k, nullptr, lw.q_norm, lw.k_norm, d_inv_freq, k_cache_base,
          nullptr, m, cfg.text_q_heads, cfg.text_kv_heads, cfg.text_head_dim,
          /*pos_base=*/0, /*history_base=*/0, cfg.rms_norm_eps, stream);
    } else {
      QkvGemvDesc qkv_desc{{lw.q_proj, lw.k_proj, lw.v_proj},
                           {nullptr, nullptr, nullptr},
                           {d_q, d_k, v_cache_base},
                           {q_dim, kv_dim, kv_dim}};
      gemv_qkv_bf16(qkv_desc, d_norm, m, cfg.text_h, stream);
      rms_norm(d_q, lw.q_norm, d_q, m * cfg.text_q_heads, cfg.text_head_dim,
               cfg.rms_norm_eps, stream);
      rms_norm(d_k, lw.k_norm, d_k, m * cfg.text_kv_heads, cfg.text_head_dim,
               cfg.rms_norm_eps, stream);
      rope_half(d_q, d_positions, d_inv_freq, m, cfg.text_q_heads,
                cfg.text_head_dim, stream);
      rope_half(d_k, d_positions, d_inv_freq, m, cfg.text_kv_heads,
                cfg.text_head_dim, stream);
      const size_t slot_bytes = static_cast<size_t>(m) * kv_dim * sizeof(Bf16);
      cudaMemcpyAsync(k_cache_base, d_k, slot_bytes, cudaMemcpyDeviceToDevice,
                      stream);
    }
    // Use d_gate (sized >= max_seq * max(text_inter, q_dim)) for attention out.
    gqa_attention(d_q, k_cache_base, v_cache_base, d_gate, m, m,
                  cfg.text_q_heads, cfg.text_kv_heads, cfg.text_head_dim, true,
                  0, 0, stream);
    gemm_bf16_add_residual(d_hidden, d_gate, lw.o_proj, nullptr, m, q_dim,
                           cfg.text_h, stream);

    rms_norm(d_hidden, lw.post_ln, d_norm, m, cfg.text_h, cfg.rms_norm_eps,
             stream);
    if (m < 16) {
      gemv_swiglu_bf16(lw.gate_proj, lw.up_proj, d_norm, d_gate, m,
                       cfg.text_inter, cfg.text_h, stream);
    } else {
      gemm_bf16(d_gate, d_norm, lw.gate_proj, nullptr, m, cfg.text_h,
                cfg.text_inter, stream);
      gemm_bf16(d_up, d_norm, lw.up_proj, nullptr, m, cfg.text_h,
                cfg.text_inter, stream);
      silu_mul(d_gate, d_gate, d_up, m * cfg.text_inter, stream);
    }
    gemm_bf16_add_residual(d_hidden, d_gate, lw.down_proj, nullptr, m,
                           cfg.text_inter, cfg.text_h, stream);
  }

  // Fixed-address, fixed-shape single-token (m == 1) decode step:
  // reads token id from *d_argmax and position from *d_step_pos, runs all
  // text layers + final norm + lm_head + argmax into *d_argmax.
  // 100% capturable into a single CUDA Graph when text_head_dim == 128.
  void run_decode_step_body() {
    const int q_dim = cfg.text_q_heads * cfg.text_head_dim;
    const int kv_dim = cfg.text_kv_heads * cfg.text_head_dim;
    index_embeddings(embed_tokens_w, d_argmax, d_hidden, 1, cfg.text_h, stream);
    for (int l = 0; l < cfg.text_layers; ++l) {
      const TextLayerWeights& lw = text_layers[l];
      Bf16* k_cache_base = kv + kv_offset(l, false, 0);
      Bf16* v_cache_base = kv + kv_offset(l, true, 0);

      rms_norm(d_hidden, lw.input_ln, d_norm, 1, cfg.text_h, cfg.rms_norm_eps,
               stream);
      QkvGemvDesc qkv_desc{{lw.q_proj, lw.k_proj, lw.v_proj},
                           {nullptr, nullptr, nullptr},
                           {d_q, d_k, d_v},
                           {q_dim, kv_dim, kv_dim}};
      gemv_qkv_bf16(qkv_desc, d_norm, 1, cfg.text_h, stream);
      qk_norm_rope_kv_append_devpos_bf16(
          d_q, d_k, d_v, lw.q_norm, lw.k_norm, d_inv_freq, k_cache_base,
          v_cache_base, cfg.text_q_heads, cfg.text_kv_heads, cfg.text_head_dim,
          d_step_pos, cfg.rms_norm_eps, stream);
      gqa_attention_devpos_bf16(d_q, k_cache_base, v_cache_base, d_gate,
                                cfg.text_q_heads, cfg.text_kv_heads,
                                cfg.text_head_dim, d_step_pos, stream);
      gemm_bf16_add_residual(d_hidden, d_gate, lw.o_proj, nullptr, 1, q_dim,
                             cfg.text_h, stream);

      rms_norm(d_hidden, lw.post_ln, d_norm, 1, cfg.text_h, cfg.rms_norm_eps,
               stream);
      gemv_swiglu_bf16(lw.gate_proj, lw.up_proj, d_norm, d_gate, 1,
                       cfg.text_inter, cfg.text_h, stream);
      gemm_bf16_add_residual(d_hidden, d_gate, lw.down_proj, nullptr, 1,
                             cfg.text_inter, cfg.text_h, stream);
    }
    rms_norm(d_hidden, final_norm_w, d_norm, 1, cfg.text_h, cfg.rms_norm_eps,
             stream);
    gemv_bf16(lm_head_w, d_norm, d_logits, cfg.vocab, cfg.text_h, stream);
    argmax_f32_dev(d_logits, cfg.vocab, d_argmax, stream);
  }

  // Fallback single-token decode for non-128 head_dim checkpoints.
  void run_decode_step_generic(int history) {
    const int q_dim = cfg.text_q_heads * cfg.text_head_dim;
    const int kv_dim = cfg.text_kv_heads * cfg.text_head_dim;
    const size_t slot_bytes = static_cast<size_t>(kv_dim) * sizeof(Bf16);
    index_embeddings(embed_tokens_w, d_argmax, d_hidden, 1, cfg.text_h, stream);
    for (int l = 0; l < cfg.text_layers; ++l) {
      const TextLayerWeights& lw = text_layers[l];
      Bf16* k_cache_base = kv + kv_offset(l, false, 0);
      Bf16* v_cache_base = kv + kv_offset(l, true, 0);
      Bf16* v_cache_dst = kv + kv_offset(l, true, history);
      rms_norm(d_hidden, lw.input_ln, d_norm, 1, cfg.text_h, cfg.rms_norm_eps,
               stream);
      QkvGemvDesc qkv_desc{{lw.q_proj, lw.k_proj, lw.v_proj},
                           {nullptr, nullptr, nullptr},
                           {d_q, d_k, v_cache_dst},
                           {q_dim, kv_dim, kv_dim}};
      gemv_qkv_bf16(qkv_desc, d_norm, 1, cfg.text_h, stream);
      rms_norm(d_q, lw.q_norm, d_q, cfg.text_q_heads, cfg.text_head_dim,
               cfg.rms_norm_eps, stream);
      rms_norm(d_k, lw.k_norm, d_k, cfg.text_kv_heads, cfg.text_head_dim,
               cfg.rms_norm_eps, stream);
      rope_half(d_q, d_step_pos, d_inv_freq, 1, cfg.text_q_heads,
                cfg.text_head_dim, stream);
      rope_half(d_k, d_step_pos, d_inv_freq, 1, cfg.text_kv_heads,
                cfg.text_head_dim, stream);
      cudaMemcpyAsync(kv + kv_offset(l, false, history), d_k, slot_bytes,
                      cudaMemcpyDeviceToDevice, stream);
      gqa_attention(d_q, k_cache_base, v_cache_base, d_gate, 1, history + 1,
                    cfg.text_q_heads, cfg.text_kv_heads, cfg.text_head_dim,
                    true, 0, history, stream);
      gemm_bf16_add_residual(d_hidden, d_gate, lw.o_proj, nullptr, 1, q_dim,
                             cfg.text_h, stream);

      rms_norm(d_hidden, lw.post_ln, d_norm, 1, cfg.text_h, cfg.rms_norm_eps,
               stream);
      gemv_swiglu_bf16(lw.gate_proj, lw.up_proj, d_norm, d_gate, 1,
                       cfg.text_inter, cfg.text_h, stream);
      gemm_bf16_add_residual(d_hidden, d_gate, lw.down_proj, nullptr, 1,
                             cfg.text_inter, cfg.text_h, stream);
    }
    rms_norm(d_hidden, final_norm_w, d_norm, 1, cfg.text_h, cfg.rms_norm_eps,
             stream);
    gemv_bf16(lm_head_w, d_norm, d_logits, cfg.vocab, cfg.text_h, stream);
    argmax_f32_dev(d_logits, cfg.vocab, d_argmax, stream);
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

  // Single-file model.safetensors or the shards named by
  // model.safetensors.index.json; both are validated by the loader.
  const SafeTensors st = SafeTensors::from_model_dir(model_dir);
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

  // One device buffer per shard. Tensor byte_offsets are relative to that
  // shard's data region, which starts after its 8-byte length prefix and JSON
  // header.
  im.d_shards.assign(st.shard_count(), nullptr);
  for (size_t s = 0; s < st.shard_count(); ++s) {
    const size_t data_start = st.data_region_offset(s);
    const size_t region_bytes = st.data_region_bytes(s);
    UMM_CUDA_CHECK(request_cuda_malloc(&im.d_shards[s], region_bytes));
    UMM_CUDA_CHECK(cudaMemcpy(im.d_shards[s], st.shard_bytes(s).data() + data_start,
                              region_bytes, cudaMemcpyHostToDevice));
  }
  for (const auto& [name, info] : st.tensors()) {
    if (info.dtype != "BF16") throw std::runtime_error("only BF16 weights supported: " + name);
    if (info.byte_offset % 2 != 0)
      throw std::runtime_error("BF16 tensor not 2-byte aligned: " + name);
    im.weights.emplace(name, im.d_shards[info.shard] + info.byte_offset / 2);
  }

  // Pre-resolve ALL Audio Tower and Text Decoder weight pointers once.
  {
    AudioTowerWeights& au = im.audio_w;
    au.conv1_w = im.w(at + "conv2d1.weight");
    au.conv1_b = im.w(at + "conv2d1.bias");
    au.conv2_w = im.w(at + "conv2d2.weight");
    au.conv2_b = im.w(at + "conv2d2.bias");
    au.conv3_w = im.w(at + "conv2d3.weight");
    au.conv3_b = im.w(at + "conv2d3.bias");
    au.conv_out_w = im.w(at + "conv_out.weight");
    au.ln_post_w = im.w(at + "ln_post.weight");
    au.ln_post_b = im.w(at + "ln_post.bias");
    au.proj1_w = im.w(at + "proj1.weight");
    au.proj1_b = im.w(at + "proj1.bias");
    au.proj2_w = im.w(at + "proj2.weight");
    au.proj2_b = im.w(at + "proj2.bias");
    au.layers.resize(c.audio_layers);
    for (int l = 0; l < c.audio_layers; ++l) {
      const std::string lp = at + "layers." + std::to_string(l);
      AudioLayerWeights& al = au.layers[l];
      al.attn_ln_w = im.w(lp + ".self_attn_layer_norm.weight");
      al.attn_ln_b = im.w(lp + ".self_attn_layer_norm.bias");
      al.q_w = im.w(lp + ".self_attn.q_proj.weight");
      al.q_b = im.w(lp + ".self_attn.q_proj.bias");
      al.k_w = im.w(lp + ".self_attn.k_proj.weight");
      al.k_b = im.w(lp + ".self_attn.k_proj.bias");
      al.v_w = im.w(lp + ".self_attn.v_proj.weight");
      al.v_b = im.w(lp + ".self_attn.v_proj.bias");
      al.out_w = im.w(lp + ".self_attn.out_proj.weight");
      al.out_b = im.w(lp + ".self_attn.out_proj.bias");
      al.final_ln_w = im.w(lp + ".final_layer_norm.weight");
      al.final_ln_b = im.w(lp + ".final_layer_norm.bias");
      al.fc1_w = im.w(lp + ".fc1.weight");
      al.fc1_b = im.w(lp + ".fc1.bias");
      al.fc2_w = im.w(lp + ".fc2.weight");
      al.fc2_b = im.w(lp + ".fc2.bias");
    }
  }
  im.embed_tokens_w = im.w("thinker.model.embed_tokens.weight");
  im.final_norm_w = im.w("thinker.model.norm.weight");
  im.lm_head_w = im.w("thinker.lm_head.weight");
  im.text_layers.resize(c.text_layers);
  for (int l = 0; l < c.text_layers; ++l) {
    const std::string lp = "thinker.model.layers." + std::to_string(l);
    TextLayerWeights& lw = im.text_layers[l];
    lw.input_ln = im.w(lp + ".input_layernorm.weight");
    lw.q_proj = im.w(lp + ".self_attn.q_proj.weight");
    lw.k_proj = im.w(lp + ".self_attn.k_proj.weight");
    lw.v_proj = im.w(lp + ".self_attn.v_proj.weight");
    lw.q_norm = im.w(lp + ".self_attn.q_norm.weight");
    lw.k_norm = im.w(lp + ".self_attn.k_norm.weight");
    lw.o_proj = im.w(lp + ".self_attn.o_proj.weight");
    lw.post_ln = im.w(lp + ".post_attention_layernorm.weight");
    lw.gate_proj = im.w(lp + ".mlp.gate_proj.weight");
    lw.up_proj = im.w(lp + ".mlp.up_proj.weight");
    lw.down_proj = im.w(lp + ".mlp.down_proj.weight");
  }

  // Workspaces, every size derived from config.json (strictly <= original VRAM:
  // unused d_proj removed, d_v shrunk to 1 row for decode, d_positions only
  // allocated when head_dim != 128).
  auto mk = [&](Bf16*& p, size_t n) { UMM_CUDA_CHECK(request_cuda_malloc(&p, n * sizeof(Bf16))); };
  UMM_CUDA_CHECK(request_cuda_malloc(&im.d_mel, c.mel_elements() * sizeof(float)));
  UMM_CUDA_CHECK(request_cuda_malloc(&im.d_inv_freq, c.inv_freq_elements() * sizeof(float)));
  UMM_CUDA_CHECK(request_cuda_malloc(&im.d_logits, c.logits_elements() * sizeof(float)));
  UMM_CUDA_CHECK(request_cuda_malloc(&im.d_ids, c.max_seq * sizeof(int)));
  if (c.text_head_dim != 128) {
    UMM_CUDA_CHECK(request_cuda_malloc(&im.d_positions, c.max_seq * sizeof(int)));
  }
  UMM_CUDA_CHECK(request_cuda_malloc(&im.d_step_pos, sizeof(int)));
  UMM_CUDA_CHECK(request_cuda_malloc(&im.d_argmax, sizeof(int)));
  UMM_CUDA_CHECK(cudaMemset(im.d_step_pos, 0, sizeof(int)));
  UMM_CUDA_CHECK(cudaMemset(im.d_argmax, 0, sizeof(int)));

  mk(im.d_hidden, c.text_row_elements());
  mk(im.d_norm, c.text_row_elements());
  mk(im.d_q, c.text_q_elements());
  mk(im.d_k, c.text_kv_elements());
  mk(im.d_v, kv_dim);  // m==1 decode only; prefill writes V directly to kv cache
  mk(im.d_gate, std::max(static_cast<size_t>(c.text_ffn_elements()),
                         static_cast<size_t>(c.text_q_elements())));
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
  UMM_CUDA_CHECK(request_cuda_malloc(&im.d_pe, c.pe_elements() * sizeof(float)));
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
  UMM_CUDA_CHECK(request_cuda_malloc(&im.kv, sizeof(Bf16) * c.kv_cache_elements()));

  std::vector<float> inv(static_cast<size_t>(c.inv_freq_elements()));
  for (int i = 0; i < c.inv_freq_elements(); ++i)
    // HF: arange(0, head_dim, 2) / head_dim, rope_theta from config.
    inv[i] = 1.0f / std::pow(c.rope_theta, 2.0f * i / c.text_head_dim);
  UMM_CUDA_CHECK(cudaMemcpy(im.d_inv_freq, inv.data(), inv.size() * sizeof(float),
                            cudaMemcpyHostToDevice));

  // Pinned host staging buffers (host RAM only, zero VRAM).
  UMM_CUDA_CHECK(cudaMallocHost(&im.h_mel_pinned, c.mel_elements() * sizeof(float)));
  UMM_CUDA_CHECK(cudaMallocHost(&im.h_ids_pinned, c.max_seq * sizeof(int)));
  UMM_CUDA_CHECK(cudaMallocHost(&im.h_argmax_pinned, sizeof(int)));
  UMM_CUDA_CHECK(cudaMallocHost(&im.h_step_pos_pinned, sizeof(int)));

  // Capture the fixed-shape 28-layer decode step into a single CUDA Graph.
  if (c.text_head_dim == 128 && !env_flag_enabled("UMM_ASR_NO_GRAPH")) {
    im.decode_graph_exec = im.try_capture_graph([&] { im.run_decode_step_body(); });
  }
}

AsrEngine::~AsrEngine() = default;

std::vector<int> AsrEngine::transcribe_tokens(const MelFeatures& mel, int max_new_tokens,
                                              const AbortCheck& abort) {
  Impl& im = *impl_;
  const AsrModelConfig& c = im.cfg;
  if (mel.data.size() > static_cast<size_t>(c.mel_elements())) {
    throw std::invalid_argument("mel feature buffer exceeds workspace capacity");
  }
  std::memcpy(im.h_mel_pinned, mel.data.data(), mel.data.size() * sizeof(float));
  UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_mel, im.h_mel_pinned,
                                 mel.data.size() * sizeof(float),
                                 cudaMemcpyHostToDevice, im.stream));

  const int n_audio = im.audio_forward(mel.valid_frames, abort);
  const auto prompt = QwenTokenizer::build_asr_prompt(n_audio, c.audio_start_id,
                                                      c.audio_token_id, c.audio_end_id);
  const int S = static_cast<int>(prompt.size());
  if (S + max_new_tokens > c.max_seq)
    throw std::runtime_error("sequence exceeds fixed KV capacity " +
                             std::to_string(c.max_seq));

  std::memcpy(im.h_ids_pinned, prompt.data(), static_cast<size_t>(S) * sizeof(int));
  UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_ids, im.h_ids_pinned,
                                 static_cast<size_t>(S) * sizeof(int),
                                 cudaMemcpyHostToDevice, im.stream));
  if (im.d_positions) {
    std::vector<int> positions(S);
    std::iota(positions.begin(), positions.end(), 0);
    UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_positions, positions.data(),
                                   static_cast<size_t>(S) * sizeof(int),
                                   cudaMemcpyHostToDevice, im.stream));
  }
  index_embeddings(im.embed_tokens_w, im.d_ids, im.d_hidden, S, c.text_h,
                   im.stream);
  // prompt[3] is audio_start; placeholders start at offset 4.
  replace_rows(im.d_hidden, im.d_audio, 4, n_audio, c.text_h, im.stream);
  im.seq_len = S;
  for (int l = 0; l < c.text_layers; ++l) {
    if (abort && abort()) {
      throw RequestAborted("aborted in prefill layer " + std::to_string(l));
    }
    im.text_layer_prefill(l, S);
  }

  // Final norm + LM head + argmax for the last prefill token (S - 1).
  rms_norm(im.d_hidden + static_cast<size_t>(S - 1) * c.text_h, im.final_norm_w,
           im.d_norm, 1, c.text_h, c.rms_norm_eps, im.stream);
  gemv_bf16(im.lm_head_w, im.d_norm, im.d_logits, c.vocab, c.text_h, im.stream);
  argmax_f32_dev(im.d_logits, c.vocab, im.d_argmax, im.stream);
  UMM_CUDA_CHECK(cudaMemcpyAsync(im.h_argmax_pinned, im.d_argmax, sizeof(int),
                                 cudaMemcpyDeviceToHost, im.stream));
  UMM_CUDA_CHECK(cudaStreamSynchronize(im.stream));
  int next = *im.h_argmax_pinned;

  std::vector<int> generated;
  for (int step = 0; step < max_new_tokens && !QwenTokenizer::is_eos(next); ++step) {
    if (abort && abort()) {
      throw RequestAborted("aborted at token " + std::to_string(step));
    }
    generated.push_back(next);
    // *im.d_argmax already holds `next` on device from the previous argmax.
    *im.h_step_pos_pinned = im.seq_len;
    UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_step_pos, im.h_step_pos_pinned,
                                   sizeof(int), cudaMemcpyHostToDevice,
                                   im.stream));
    if (im.decode_graph_exec) {
      UMM_CUDA_CHECK(cudaGraphLaunch(im.decode_graph_exec, im.stream));
    } else if (c.text_head_dim == 128) {
      im.run_decode_step_body();
    } else {
      im.run_decode_step_generic(im.seq_len);
    }
    ++im.seq_len;
    UMM_CUDA_CHECK(cudaMemcpyAsync(im.h_argmax_pinned, im.d_argmax, sizeof(int),
                                   cudaMemcpyDeviceToHost, im.stream));
    UMM_CUDA_CHECK(cudaStreamSynchronize(im.stream));
    next = *im.h_argmax_pinned;
  }
  return generated;
}

}  // namespace umminfer::native
