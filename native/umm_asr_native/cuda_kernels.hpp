// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace umminfer::native {

using Bf16 = __nv_bfloat16;

#define UMM_CUDA_CHECK(expr)                                                          \
  do {                                                                                \
    cudaError_t _e = (expr);                                                          \
    if (_e != cudaSuccess)                                                            \
      throw std::runtime_error(std::string("CUDA error ") + cudaGetErrorString(_e) + \
                               " at " + __FILE__ + ":" + std::to_string(__LINE__));  \
  } while (0)

// Descriptor for fused 3-way Q/K/V BF16 projection sharing the same input x.
// bias[p] may be nullptr for bias-free decoder layers.
struct QkvGemvDesc {
  const Bf16* w[3];
  const Bf16* bias[3];
  Bf16* out[3];
  int n_out[3];
};

// Elementwise / norm kernels. Row-major tensors, `dim` columns per row.
void rms_norm(const Bf16* x, const Bf16* weight, Bf16* out, int rows, int dim,
              float eps, cudaStream_t stream);
void layer_norm(const Bf16* x, const Bf16* weight, const Bf16* bias, Bf16* out,
                int rows, int dim, float eps, cudaStream_t stream);
void add_bias(Bf16* y, const Bf16* bias, int rows, int dim, cudaStream_t stream);
void add_residual(Bf16* y, const Bf16* residual, int n, cudaStream_t stream);
void gelu_inplace(Bf16* x, int n, cudaStream_t stream);
void silu_mul(Bf16* out, const Bf16* gate, const Bf16* up, int n, cudaStream_t stream);
void f32_to_bf16(const float* src, Bf16* dst, int n, cudaStream_t stream);
void bf16_to_f32(const Bf16* src, float* dst, int n, cudaStream_t stream);

// Gather embedding rows: out[m, dim] = table[ids[m], dim].
void index_embeddings(const Bf16* table, const int* ids, Bf16* out, int m, int dim,
                      cudaStream_t stream);
// Replace rows [start, start+n) of dst with src [n, dim].
void replace_rows(Bf16* dst, const Bf16* src, int start, int n, int dim,
                  cudaStream_t stream);

// Half-rotation RoPE (Qwen3 rotate_half). inv_freq (device, head_dim/2
// entries) is precomputed once. x is [n, num_heads, head_dim].
void rope_half(Bf16* x, const int* positions, const float* inv_freq, int n,
               int num_heads, int head_dim, cudaStream_t stream);

// Fused Q-norm + K-norm + half-rotation RoPE + K/V cache append for head_dim=128.
// Scalar-position variant (prefill): positions are [pos_base .. pos_base + m - 1],
// cache slots start at history_base. When v_src != nullptr && v_cache != nullptr,
// V is also copied into v_cache.
void qk_norm_rope_kv_append_bf16(Bf16* q, const Bf16* k_src, const Bf16* v_src,
                                 const Bf16* q_norm_w, const Bf16* k_norm_w,
                                 const float* inv_freq, Bf16* k_cache,
                                 Bf16* v_cache, int m, int n_q_heads,
                                 int n_kv_heads, int head_dim, int pos_base,
                                 int history_base, float eps,
                                 cudaStream_t stream);

// Device-position variant for CUDA-graphed single-token (m == 1) decode:
// reads pos = *d_pos_ptr on device and appends K/V at slot `pos`.
void qk_norm_rope_kv_append_devpos_bf16(Bf16* q, const Bf16* k_src,
                                        const Bf16* v_src, const Bf16* q_norm_w,
                                        const Bf16* k_norm_w,
                                        const float* inv_freq, Bf16* k_cache,
                                        Bf16* v_cache, int n_q_heads,
                                        int n_kv_heads, int head_dim,
                                        const int* d_pos_ptr, float eps,
                                        cudaStream_t stream);

// Conv2d 3x3, stride 2, padding 1, then GELU. Input [b, ci, h, w],
// weight [co, ci, 3, 3], bias [co]. Uses SM90 BF16 Tensor Core implicit-GEMM
// when ci >= 16 && co >= 16, and direct kernel for Conv1 (ci == 1).
void conv3x3s2_gelu(const Bf16* in, const Bf16* weight, const Bf16* bias, Bf16* out,
                    int b, int ci, int co, int h, int w, cudaStream_t stream);

// Gather 100-frame mel chunks: out [chunks, 1, 128, 100] BF16, frames beyond
// valid_frames are zero padding.
void mel_to_chunks(const float* mel, Bf16* out, int chunks, int valid_frames,
                   cudaStream_t stream);

// [chunks, down, 16, 13] -> [chunks, 13, down*16] (permute 0,3,1,2 + flatten).
// `down` is config.json downsample_hidden_size (480 for both ASR sizes).
void conv_permute(const Bf16* in, Bf16* out, int chunks, int down,
                  cudaStream_t stream);

// Add the Whisper-style sinusoidal position table [13, dim] to each chunk's
// 13 rows. hidden is [chunks, 13, dim].
void add_sinusoidal(Bf16* hidden, const float* pe, int chunks, int rows, int dim,
                    cudaStream_t stream);

// GQA attention. q [nq, nqh, d], k/v [nkv, nkh, d], out [nq, nqh, d].
// causal=true: decoder mask; q_offset is the key-space position of q row 0.
// causal=false: audio encoder, bidirectional inside non-overlapping windows
// of window_len (104 for both Qwen3-ASR sizes, from config.json).
void gqa_attention(const Bf16* q, const Bf16* k, const Bf16* v, Bf16* out, int nq,
                   int nkv, int n_q_heads, int n_kv_heads, int head_dim, bool causal,
                   int window_len, int q_offset, cudaStream_t stream);

// Device-position causal GQA attention (nq == 1, head_dim == 128) for
// CUDA-graphed decode: attends over [0, *d_pos_ptr + 1).
void gqa_attention_devpos_bf16(const Bf16* q, const Bf16* k, const Bf16* v,
                               Bf16* out, int n_q_heads, int n_kv_heads,
                               int head_dim, const int* d_pos_ptr,
                               cudaStream_t stream);

// Greedy argmax over one logits row. scratch_idx is a device int owned by
// the caller; the chosen index is returned synchronously.
int argmax_f32(const float* logits, int n, int* scratch_idx, cudaStream_t stream);

// Device-side argmax: writes the winning index into `out` without host sync.
void argmax_f32_dev(const float* logits, int n, int* out, cudaStream_t stream);

// Vectorized SM90 BF16->F32 matrix-vector for the LM head:
// out[j] = sum_k W[j,k] x[k], fp32 accumulators.
void gemv_bf16(const Bf16* w, const Bf16* x, float* out, int n_out, int k,
               cudaStream_t stream);

// SM90 BF16 linear: y[m, n] = x[m, k] * W[n, k]^T (+ bias).
// Auto-dispatches to 64x64x64 BF16 Tensor Core MMA for m >= 16 and 1-warp/row
// vectorized uint4 GEMV for small m.
void gemm_bf16(Bf16* y, const Bf16* x, const Bf16* w, const Bf16* bias, int m,
               int k, int n, cudaStream_t stream);

// Fused BF16 linear + residual add with exact BF16 intermediate rounding:
// y[m, n] = to_bf16(to_float(y[m, n]) + to_float(to_bf16(x * W^T + bias))).
void gemm_bf16_add_residual(Bf16* y, const Bf16* x, const Bf16* w,
                            const Bf16* bias, int m, int k, int n,
                            cudaStream_t stream);

// Fused BF16 linear + exact GELU with exact BF16 intermediate rounding:
// y[m, n] = to_bf16(gelu(to_float(to_bf16(x * W^T + bias)))).
void gemm_bf16_gelu(Bf16* y, const Bf16* x, const Bf16* w, const Bf16* bias,
                    int m, int k, int n, cudaStream_t stream);

// Fused 3-way Q/K/V BF16 projection sharing input `x`.
void gemv_qkv_bf16(const QkvGemvDesc& desc, const Bf16* x, int m, int k,
                   cudaStream_t stream);

// Fused dual-output SwiGLU BF16 projection:
// out = to_bf16(silu(to_bf16(x*W_gate^T)) * to_bf16(x*W_up^T)).
void gemv_swiglu_bf16(const Bf16* w_gate, const Bf16* w_up, const Bf16* x,
                      Bf16* out, int m, int n_out, int k, cudaStream_t stream);

// Legacy alias kept for compatibility; routes directly to gemm_bf16.
void gemm_bf16_naive(Bf16* y, const Bf16* x, const Bf16* w, const Bf16* bias,
                     int n_tokens, int k, int m_out, cudaStream_t stream);

}  // namespace umminfer::native
