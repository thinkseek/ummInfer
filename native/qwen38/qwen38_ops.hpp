// SPDX-License-Identifier: Apache-2.0
// CUDA entry points for the native Qwen3.8-27B executor.
//
// Derived from native/omni (same repository, Apache-2.0): the tensor-core GEMM,
// the warp GEMV and several elementwise kernels keep that implementation and
// its precision choices.  The hybrid-attention kernels (partial mRoPE, fused
// output gate, depthwise causal conv, gated delta rule, gated RMSNorm) are new.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

#include "bf16.hpp"

namespace umminfer::qwen38 {

#define UMM_QWEN38_CUDA_CHECK(expr)                                               \
  do {                                                                            \
    cudaError_t _e = (expr);                                                      \
    if (_e != cudaSuccess)                                                        \
      throw std::runtime_error(std::string("CUDA error ") +                       \
                               cudaGetErrorString(_e) + " at " + __FILE__ + ":" + \
                               std::to_string(__LINE__));                         \
  } while (0)

// ---- GEMM / GEMV ----------------------------------------------------------
// y[m,n] = x[m,k] * w[n,k]^T (+ bias).  Bias and output dtype vary by call.
void gemm_bf16(Bf16* y, const Bf16* x, const Bf16* w, const Bf16* bias, int m,
               int k, int n, cudaStream_t stream);
void gemm_bf16_f32(float* y, const Bf16* x, const Bf16* w, int m, int k, int n,
                   cudaStream_t stream);
void gemv_bf16_f32(const Bf16* w, const Bf16* x, float* out, int n_out, int k,
                   cudaStream_t stream);
void gemv_bf16_bf16(const Bf16* w, const Bf16* x, Bf16* out, int n_out, int k,
                    cudaStream_t stream);

// ---- Normalisation and elementwise ----------------------------------------
// Plain RMSNorm (weight applied as-is), fp32 math, bf16 output.
void rms_norm(const Bf16* x, const Bf16* weight, Bf16* out, int rows, int dim,
              float eps, cudaStream_t stream);
// Qwen3.5 RMSNorm: weight is zero-centred, the scale is (1 + weight).
void rms_norm_zero_centered(const Bf16* x, const Bf16* weight, Bf16* out,
                            int rows, int dim, float eps, cudaStream_t stream);
void layer_norm(const Bf16* x, const Bf16* weight, const Bf16* bias, Bf16* out,
                int rows, int dim, float eps, cudaStream_t stream);
void add_bias(Bf16* y, const Bf16* bias, int rows, int dim, cudaStream_t stream);
void add_residual(Bf16* y, const Bf16* residual, int n, cudaStream_t stream);
// act = bf16(silu(float(bf16(gate)))) * up, each product rounded once.
void silu_mul(Bf16* out, const Bf16* gate, const Bf16* up, int n,
              cudaStream_t stream);
void gelu_tanh_inplace(Bf16* x, int n, cudaStream_t stream);   // gelu_pytorch_tanh
void gelu_erf_inplace(Bf16* x, int n, cudaStream_t stream);    // exact erf GELU
void f32_to_bf16(const float* src, Bf16* dst, int n, cudaStream_t stream);
void bf16_to_f32(const Bf16* src, float* dst, int n, cudaStream_t stream);

// ---- Embedding / gather ---------------------------------------------------
void index_embeddings(const Bf16* table, const int* ids, Bf16* out, int m,
                      int dim, cudaStream_t stream);
// dst[start .. start+n) = src[0 .. n).
void replace_rows(Bf16* dst, const Bf16* src, int start, int n, int dim,
                  cudaStream_t stream);

// ---- Text attention -------------------------------------------------------
// Partial mRoPE: rotates the first `rotary_dim` channels, half-rotation pairs
// (i, i + rotary_dim/2).  cos/sin are bf16-rounded reference tables [n, rotary_dim].
void mrope_partial(Bf16* x, const Bf16* cos, const Bf16* sin, int n, int heads,
                   int head_dim, int rotary_dim, cudaStream_t stream);
// Causal GQA over a KV buffer that already holds history + m new rows.
// q [m, n_q_heads, head_dim], k/v [nkv, n_kv_heads, head_dim], out [m, n_q_heads,
// head_dim].  When `gate` is non-null the output is multiplied by
// bf16(sigmoid(float(gate))) before rounding.
void gqa_attention(const Bf16* q, const Bf16* k, const Bf16* v, const Bf16* gate,
                   Bf16* out, int m, int nkv, int n_q_heads, int n_kv_heads,
                   int head_dim, int q_offset, cudaStream_t stream);
// Bidirectional MHA for the vision tower (one packed image per call).
void vision_attention(const Bf16* q, const Bf16* k, const Bf16* v, Bf16* out,
                      int n, int heads, int head_dim, cudaStream_t stream);
// Vision rope: full-dim half rotation, fp32 cos/sin tables [n, head_dim].
void vision_rope(Bf16* x, const float* cos, const float* sin, int n, int heads,
                 int head_dim, cudaStream_t stream);
// Split the per-head interleaved q_proj output [m, heads, 2*head_dim] into
// contiguous query [m, heads, head_dim] and gate [m, heads, head_dim].
void split_q_gate(const Bf16* qg, Bf16* q, Bf16* gate, int m, int heads,
                  int head_dim, cudaStream_t stream);
// Split fused qkv [m, 3*width] into three contiguous [m, width] buffers.
void split_qkv(const Bf16* qkv, Bf16* q, Bf16* k, Bf16* v, int m, int width,
               cudaStream_t stream);
// Linear-attention layout: q [m, wq], k [m, wk], v [m, wv] concatenated.
void split_qkv_widths(const Bf16* qkv, Bf16* q, Bf16* k, Bf16* v, int m, int wq,
                      int wk, int wv, cudaStream_t stream);

// ---- Linear attention -----------------------------------------------------
// Depthwise causal conv over time with kernel 4, then SiLU.  x [m, channels],
// w [channels, 4], state [channels, 3] keeps the last three inputs (oldest
// first).  The state is updated in place.
void conv1d_causal_silu(const Bf16* x, const Bf16* w, Bf16* state, Bf16* out,
                        int m, int channels, cudaStream_t stream);
// beta = bf16(sigmoid(float(b))); g = -exp(A_log) * softplus(a + dt_bias) in fp32.
void beta_and_decay(float* beta, float* g, const Bf16* b, const Bf16* a,
                    const float* A_log, const float* dt_bias, int m, int heads,
                    cudaStream_t stream);
// L2-normalise q/k, scale q by 1/sqrt(head_dim).  Inputs bf16 [m, k_heads, dim].
void l2norm_qk(const Bf16* q, const Bf16* k, float* qn, float* kn, int m,
               int k_heads, int dim, cudaStream_t stream);
// Recurrent gated delta rule, one step per token.  qn/kn fp32 [m, k_heads, dim],
// v bf16 [m, v_heads, dim], beta/g fp32 [m, v_heads], state fp32
// [v_heads, dim, dim] (k-major), out bf16 [m, v_heads, dim].
void gated_delta_rule(float* state, const float* qn, const float* kn,
                      const Bf16* v, const float* beta, const float* g,
                      Bf16* out, int m, int k_heads, int v_heads, int dim,
                      cudaStream_t stream);
// RMSNorm over each head row, then multiply by weight * silu(z); matches
// Qwen3_5RMSNormGated's rounding (norm -> bf16, gate -> fp32).
void gated_rms_norm(const Bf16* x, const Bf16* z, const Bf16* weight, Bf16* out,
                    int rows, int dim, float eps, cudaStream_t stream);

// ---- Sampling -------------------------------------------------------------
int argmax_f32(const float* logits, int n, int* scratch_idx, cudaStream_t stream);

}  // namespace umminfer::qwen38
