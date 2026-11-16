// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace umminfer::native {

using Bf16 = __nv_bfloat16;

#define UMM_CUDA_CHECK(expr)                                                          \
  do {                                                                                \
    cudaError_t _e = (expr);                                                          \
    if (_e != cudaSuccess)                                                            \
      throw std::runtime_error(std::string("CUDA error ") + cudaGetErrorString(_e) + \
                               " at " + __FILE__ + ":" + std::to_string(__LINE__));  \
  } while (0)

// ---- elementwise / norms (templated on scalar T = float or Bf16) -------------

template <typename T>
void rms_norm_t(const T* x, const T* weight, T* out, int rows, int dim, float eps,
                cudaStream_t stream);
template <typename T>
void layer_norm_t(const T* x, const T* weight, const T* bias, T* out, int rows,
                  int dim, float eps, cudaStream_t stream);
template <typename T>
void add_residual_t(T* y, const T* r, int n, cudaStream_t stream);
template <typename T>
void silu_mul_t(T* out, const T* gate, const T* up, int n, cudaStream_t stream);
// Plain SiLU x*sigmoid(x) in place; distinct from the gated product.
template <typename T>
void silu_inplace_t(T* x, int n, cudaStream_t stream);
template <typename T>
void gelu_inplace_t(T* x, int n, cudaStream_t stream);
template <typename T>
void layer_scale_t(T* y, const T* x, const T* scale, int rows, int dim,
                   cudaStream_t stream);  // y = x * scale[row-independent]
// SnakeBeta over channel-first [channels, length] data; alpha/beta per channel.
void snake_beta_f32(float* x, const float* alpha, const float* beta, int channels,
                    int length, cudaStream_t stream);
void f32_to_bf16(const float* src, Bf16* dst, int n, cudaStream_t stream);
void bf16_to_f32(const Bf16* src, float* dst, int n, cudaStream_t stream);
void clamp_f32(float* x, int n, float lo, float hi, cudaStream_t stream);

// ---- gather / matmul ---------------------------------------------------------

template <typename T>
void gather_rows_t(const T* table, const int* ids, T* out, int m, int dim,
                   cudaStream_t stream);
// y[m,out] = x[m,k] * W[out,k]^T (+ bias). Naive one-thread-per-output GEMM;
// ponytail: global-memory GEMM, tile with shared memory if codec conv throughput
// becomes the bottleneck.
template <typename T>
void gemm_t(T* y, const T* x, const T* w, const T* bias, int m, int k, int out,
            cudaStream_t stream);
template <typename T>
void gemv_t(const T* w, const T* x, float* out, int n_out, int k, cudaStream_t stream);
int argmax_f32(const float* logits, int n, int* scratch, cudaStream_t stream);
// Set logits[idx] = -inf for every idx in [lo, hi) except keep.
void suppress_logits(float* logits, int lo, int hi, int keep, cudaStream_t stream);
// HF repetition penalty over the emitted id history (applied before argmax).
void repetition_penalty_f32(float* logits, const int* history, int n_history,
                            float penalty, cudaStream_t stream);
// Force one id to -inf (anti-degeneracy consecutive-run ban).
void ban_logit_f32(float* logits, int id, cudaStream_t stream);

// ---- rotary / attention ------------------------------------------------------

// Half-rotation (GPT-NeoX) RoPE, standard 1D positions. Equal mRoPE axes in
// this model make the talker path numerically identical to this kernel.
template <typename T>
void rope_half_t(T* x, const int* positions, const float* inv_freq, int n,
                 int num_heads, int head_dim, cudaStream_t stream);

// GQA attention. mask: 0 causal full, 1 causal sliding window (window_len),
// 2 bidirectional fixed windows (ASR-style, unused by TTS). q [nq,nqh,d],
// k/v [nkv,nkh,d].
template <typename T>
void gqa_attention_t(const T* q, const T* k, const T* v, T* out, int nq, int nkv,
                     int n_q_heads, int n_kv_heads, int head_dim, int mask,
                     int window_len, int q_offset, cudaStream_t stream);

// ---- 1-D convolution family (all F32) ----------------------------------------

// Causal Conv1d stride 1. Left zero pad = (kernel-1)*dilation, length preserved.
// groups=1 (dense) or groups=channels (depthwise; weight [c,1,k]).
void conv1d_causal_f32(const float* in, const float* w, const float* bias,
                       float* out, int channels_in, int channels_out, int length,
                       int kernel, int dilation, int groups, cudaStream_t stream);
// ConvTranspose1d with right trim of (kernel-stride) samples; net length L*stride.
void conv_transpose1d_f32(const float* in, const float* w, const float* bias,
                          float* out, int ch_in, int ch_out, int length, int kernel,
                          int stride, cudaStream_t stream);
// RVQ codebook dequant: emb[c,d] = embedding_sum[c,d] / max(usage[c], eps).
void codebook_dequant_f32(const float* embedding_sum, const float* cluster_usage,
                          float* emb, int codes, int dim, float eps,
                          cudaStream_t stream);
// Sum 16 per-group gathered [T,256] tables into one [T,256] buffer.
void sum_tables_f32(float* dst, const float* const* srcs, int count, int rows,
                    int dim, cudaStream_t stream);
// out[t, c] = in[c, t], shapes in [channels, length].
void transpose_f32(const float* in, float* out, int channels, int length,
                   cudaStream_t stream);
// out[c, t] = in[t, c], the inverse layout (channel-last -> channel-first).
void transpose_inv_f32(const float* in, float* out, int channels, int length,
                       cudaStream_t stream);
// y = x + bias broadcast over rows; y/x are [rows, dim].
void add_bias_rows_f32(float* y, const float* bias, int rows, int dim,
                       cudaStream_t stream);
// out[t] = codes[t * stride + group], extracting one codebook column.
void gather_code_column(const int* codes, int* out, int frames, int stride,
                        int group, cudaStream_t stream);

}  // namespace umminfer::native
