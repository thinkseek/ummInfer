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

// Conv2d 3x3, stride 2, padding 1, then GELU. Input [b, ci, h, w],
// weight [co, ci, 3, 3], bias [co]. Output dims returned via out_h/out_w.
void conv3x3s2_gelu(const Bf16* in, const Bf16* weight, const Bf16* bias, Bf16* out,
                    int b, int ci, int co, int h, int w, cudaStream_t stream);

// Gather 100-frame mel chunks: out [chunks, 1, 128, 100] BF16, frames beyond
// valid_frames are zero padding.
void mel_to_chunks(const float* mel, Bf16* out, int chunks, int valid_frames,
                   cudaStream_t stream);

// [chunks, 480, 16, 13] -> [chunks, 13, 7680] (permute 0,3,1,2 + flatten).
void conv_permute(const Bf16* in, Bf16* out, int chunks, cudaStream_t stream);

// Add the Whisper-style sinusoidal position table [13, 896] to each chunk's
// 13 rows. hidden is [chunks, 13, 896].
void add_sinusoidal(Bf16* hidden, const float* pe, int chunks, int rows, int dim,
                    cudaStream_t stream);

// GQA attention. q [nq, nqh, d], k/v [nkv, nkh, d], out [nq, nqh, d].
// causal=true: decoder mask; q_offset is the key-space position of q row 0.
// causal=false: audio encoder, bidirectional inside non-overlapping windows
// of window_len (104 for this model).
void gqa_attention(const Bf16* q, const Bf16* k, const Bf16* v, Bf16* out, int nq,
                   int nkv, int n_q_heads, int n_kv_heads, int head_dim, bool causal,
                   int window_len, int q_offset, cudaStream_t stream);

// Greedy argmax over one logits row. scratch_idx is a device int owned by
// the caller; the chosen index is returned synchronously.
int argmax_f32(const float* logits, int n, int* scratch_idx, cudaStream_t stream);

// Matrix-vector for the tied lm head: out[j] = sum_k W[j,k] x[k], fp32
// accumulators. Used instead of cuBLAS GEMM because cuBLAS 12.4 on this host
// returns CUBLAS_STATUS_NOT_SUPPORTED for GEMMs with m above 1024 (no usable
// precompiled kernel; libcuLTO absent), and the head is 151936x1024.
void gemv_bf16(const Bf16* w, const Bf16* x, float* out, int n_out, int k,
               cudaStream_t stream);

// General fallback GEMM for shapes cuBLAS rejects here:
// y[n, m_out] = x[n, k] * W^T[m_out, k] + bias[m_out], BF16 throughout with
// fp32 accumulation. One thread per output element; the model is 0.6B and the
// widest failing dimension is 3584, so L2 reuse is sufficient.
// ponytail: naive global-memory GEMM, replace with a tiled kernel if the
// decoder for longer audio needs the throughput.
void gemm_bf16_naive(Bf16* y, const Bf16* x, const Bf16* w, const Bf16* bias,
                     int n_tokens, int k, int m_out, cudaStream_t stream);

}  // namespace umminfer::native
