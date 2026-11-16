// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "bf16.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace umminfer::omni {

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

// Text mRoPE, GPT-NeoX half rotation. cos/sin are precomputed [n, head_dim]
// fp32 tables (duplicated halves already included); x is [n, heads, head_dim].
void mrope_half(Bf16* x, const float* cos, const float* sin, int n,
                int num_heads, int head_dim, cudaStream_t stream);

// Vision rotary, same half-rotation math, cos/sin are BF16 [n, head_dim] but
// the reference upcasts q/k/cos/sin to fp32 inside the op.
void vision_rope(Bf16* x, const Bf16* cos, const Bf16* sin, int n,
                 int num_heads, int head_dim, cudaStream_t stream);

// gelu_pytorch_tanh: 0.5 x (1 + tanh(sqrt(2/pi)(x + 0.044715 x^3))).
void gelu_tanh_inplace(Bf16* x, int n, cudaStream_t stream);

// Router: logits [rows, 128] fp32 -> softmax, stable top-8 (lower index on
// ties), renormalize to sum 1. Writes selected experts [rows, 8] int32 and
// weights [rows, 8] fp32.
void moe_route(const float* logits, int* top_idx, float* top_w, int rows,
               int num_experts, int top_k, cudaStream_t stream);

// One expert GEMV chain for one token, fp32 accumulation.
//   gate[k768] = Wg[768,2048] x[2048]; up likewise;
//   a = silu(gate)*up; y[2048] = Wd[2048,768] a; acc += routing_w * y
void expert_gemv_acc(const Bf16* x, const Bf16* wg, const Bf16* wu,
                     const Bf16* wd, Bf16* acc, float routing_w, int hidden,
                     int inter, cudaStream_t stream);

// Tiled BF16 GEMM with shared-memory tiles and fp32 accumulation.
// y[n, m_out] = x[n, k] * W^T[m_out, k] + optional bias.
void gemm_bf16_tiled(Bf16* y, const Bf16* x, const Bf16* w, const Bf16* bias,
                     int n_tokens, int k, int m_out, cudaStream_t stream);

// Add a deepstack feature into the visual placeholder rows of hidden.
// hidden [seq, 2048], feature [n_vis, 2048]; vis_pos[0..n_vis-1] are the
// sequence indices of the image_pad rows.
void add_deepstack(Bf16* hidden, const Bf16* feature, const int* vis_pos,
                   int n_vis, int dim, cudaStream_t stream);

// Split a [tokens, 3*width] row-major qkv buffer into three contiguous
// [tokens, width] buffers (the Linear output interleaves q/k/v per token).
void split_qkv(const Bf16* qkv, Bf16* q, Bf16* k, Bf16* v, int tokens,
               int width, cudaStream_t stream);

}  // namespace umminfer::omni
