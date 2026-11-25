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

// Descriptor for up to 4 BF16 GEMV projections sharing the same input x[m, k]
// in one launch. Every active entry is bitwise equal to calling
// gemv_bf16_bf16_batch independently.
struct MultiGemvDesc {
  const Bf16* w[4];
  Bf16* out[4];
  int n_out[4];
  int count;
};
void gemv_bf16_bf16_multi_batch(const MultiGemvDesc& desc, const Bf16* x, int m,
                                int k, cudaStream_t stream);

// Fused SwiGLU decode GEMV: reads x[m, k] once, computes both gate_proj and
// up_proj rows, and writes bf16(bf16(silu(bf16(gate))) * bf16(up)) directly.
// Bitwise equal to gemv_bf16_bf16_batch(gate) + gemv_bf16_bf16_batch(up) +
// silu_mul.
void gemv_bf16_swiglu_batch(const Bf16* w_gate, const Bf16* w_up, const Bf16* x,
                            Bf16* out, int m, int n_out, int k,
                            cudaStream_t stream);

// ---- Normalisation and elementwise ----------------------------------------
// Plain RMSNorm (weight applied as-is), fp32 math, bf16 output.
void rms_norm(const Bf16* x, const Bf16* weight, Bf16* out, int rows, int dim,
              float eps, cudaStream_t stream);
// Qwen3.5 RMSNorm: weight is zero-centred, the scale is (1 + weight).
void rms_norm_zero_centered(const Bf16* x, const Bf16* weight, Bf16* out,
                            int rows, int dim, float eps, cudaStream_t stream);
// Fused residual add + zero-centred RMSNorm:
//   hidden[i] = bf16(hidden[i] + residual[i]);
//   norm_out[r, :] = rms_norm_zero_centered(hidden[r, :], weight, eps).
void add_residual_rms_norm_zero_centered(Bf16* hidden, const Bf16* residual,
                                         const Bf16* weight, Bf16* norm_out,
                                         int rows, int dim, float eps,
                                         cudaStream_t stream);
void layer_norm(const Bf16* x, const Bf16* weight, const Bf16* bias, Bf16* out,
                int rows, int dim, float eps, cudaStream_t stream);
void add_bias(Bf16* y, const Bf16* bias, int rows, int dim, cudaStream_t stream);
void add_residual(Bf16* y, const Bf16* residual, int n, cudaStream_t stream);
// Adds a repeating `tile_n`-element residual across `total_n` elements:
// y[i] += residual[i % tile_n]. Dispatches to add_residual when total_n <= tile_n.
void add_residual_tiled(Bf16* y, const Bf16* residual, int total_n, int tile_n,
                        cudaStream_t stream);
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
// Computes one decode row [kTextRotaryDim] of partial mRoPE cos/sin tables on
// device from precomputed fp64 inverse frequencies and *d_rope_pos (shared
// across T/H/W for autoregressive text decode) for CUDA Graph capture.
void compute_text_mrope_devpos(const double* inv_freq, const int* d_rope_pos,
                               Bf16* cos_out, Bf16* sin_out,
                               cudaStream_t stream);
// Causal GQA over a KV buffer that already holds history + m new rows.
// q [m, n_q_heads, head_dim], k/v [nkv, n_kv_heads, head_dim], out [m, n_q_heads,
// head_dim].  When `gate` is non-null the output is multiplied by
// bf16(sigmoid(float(gate))) before rounding.
void gqa_attention(const Bf16* q, const Bf16* k, const Bf16* v, const Bf16* gate,
                   Bf16* out, int m, int nkv, int n_q_heads, int n_kv_heads,
                   int head_dim, int q_offset, cudaStream_t stream);
// Single-token (m == 1) device-position variant of gqa_attention for CUDA Graph
// capture: attends over KV positions [0, *d_kv_pos] inclusive (hi = *d_kv_pos + 1).
void gqa_attention_devpos(const Bf16* q, const Bf16* k, const Bf16* v,
                          const Bf16* gate, Bf16* out, const int* d_kv_pos,
                          int n_q_heads, int n_kv_heads, int head_dim,
                          cudaStream_t stream);
// Bidirectional MHA for the vision tower (one packed image per call).
void vision_attention(const Bf16* q, const Bf16* k, const Bf16* v, Bf16* out,
                      int n, int heads, int head_dim, cudaStream_t stream);
// Batched bidirectional MHA for multi-slice video: `num_slices` independent
// slices of `slice_tokens` packed contiguously in [num_slices, slice_tokens,
// heads, head_dim]. Tokens in slice s attend strictly within slice s.
void vision_attention_batched(const Bf16* q, const Bf16* k, const Bf16* v,
                              Bf16* out, int slice_tokens, int num_slices,
                              int heads, int head_dim, cudaStream_t stream);
// Vision rope: full-dim half rotation, fp32 cos/sin tables [n, head_dim].
void vision_rope(Bf16* x, const float* cos, const float* sin, int n, int heads,
                 int head_dim, cudaStream_t stream);
// Batched vision rope: applies a [slice_tokens, head_dim] spatial table across
// `n` contiguous tokens (spatial index = token % slice_tokens).
void vision_rope_tiled(Bf16* x, const float* cos, const float* sin, int n,
                       int slice_tokens, int heads, int head_dim,
                       cudaStream_t stream);
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
// Fused conv1d + SiLU + split directly into contiguous q/k/v buffers.
void conv1d_causal_silu_split(const Bf16* x, const Bf16* w, Bf16* state,
                              Bf16* q, Bf16* k, Bf16* v, int m, int wq, int wk,
                              int wv, cudaStream_t stream);
// beta = bf16(sigmoid(float(b))); g = -exp(A_log) * softplus(a + dt_bias) in fp32.
void beta_and_decay(float* beta, float* g, const Bf16* b, const Bf16* a,
                    const float* A_log, const float* dt_bias, int m, int heads,
                    cudaStream_t stream);
// L2-normalise q/k, scale q by 1/sqrt(head_dim).  Inputs bf16 [m, k_heads, dim].
void l2norm_qk(const Bf16* q, const Bf16* k, float* qn, float* kn, int m,
               int k_heads, int dim, cudaStream_t stream);
// Fused l2norm_qk + beta_and_decay in a single kernel launch.
void l2norm_qk_beta_decay(const Bf16* q, const Bf16* k, float* qn, float* kn,
                          float* beta, float* g, const Bf16* b, const Bf16* a,
                          const float* A_log, const float* dt_bias, int m,
                          int k_heads, int v_heads, int dim,
                          cudaStream_t stream);
// Fuses split_q_gate + q_norm + k_norm + partial mRoPE(q, k) + KV cache append
// for full-attention layers into a single kernel launch.
void prep_qkv_rope_kvcache(const Bf16* qg, const Bf16* k_in, const Bf16* v_in,
                           const Bf16* q_norm_w, const Bf16* k_norm_w,
                           const Bf16* cos, const Bf16* sin, Bf16* q_out,
                           Bf16* gate_out, Bf16* k_cache, Bf16* v_cache, int m,
                           int history, int n_q_heads, int n_kv_heads,
                           int head_dim, int rotary_dim, float eps,
                           cudaStream_t stream);
// Single-token (m == 1) device-position variant of prep_qkv_rope_kvcache for
// CUDA Graph capture: reads the KV cache write slot from *d_kv_pos on device.
void prep_qkv_rope_kvcache_devpos(const Bf16* qg, const Bf16* k_in,
                                  const Bf16* v_in, const Bf16* q_norm_w,
                                  const Bf16* k_norm_w, const Bf16* cos,
                                  const Bf16* sin, Bf16* q_out, Bf16* gate_out,
                                  Bf16* k_cache, Bf16* v_cache,
                                  const int* d_kv_pos, int n_q_heads,
                                  int n_kv_heads, int head_dim, int rotary_dim,
                                  float eps, cudaStream_t stream);
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

// ---- Continuous batching --------------------------------------------------
// One decode or prefill call covers at most this many sequences. A sequence
// owns rows [off, off+len) of the shared workspace, has `hist` tokens already
// in its KV cache, and uses per-sequence cache slot `slot`.
constexpr int kMaxBatch = 8;

struct SeqSpans {
  int off[kMaxBatch];
  int len[kMaxBatch];
  int hist[kMaxBatch];
  int slot[kMaxBatch];
};

// Batched GEMV. Each of the m rows is reduced in exactly the order the
// single-row gemv_* uses, so a row's result is bitwise equal to running that
// row alone. The weight matrix is read once per four output rows.
void gemv_bf16_bf16_batch(const Bf16* w, const Bf16* x, Bf16* out, int m,
                          int n_out, int k, cudaStream_t stream);
void gemv_bf16_f32_batch(const Bf16* w, const Bf16* x, float* out, int m,
                         int n_out, int k, cudaStream_t stream);
// GEMV output is [m, n_out]; this samples one argmax per row.
void argmax_rows(const float* logits, int rows, int n, int* idx,
                 cudaStream_t stream);
// dst[s, :] = src[rows[s], :].
void gather_rows(Bf16* dst, const Bf16* src, const int* rows, int m, int dim,
                 cudaStream_t stream);

// Per-sequence versions of the recurrent/attention kernels. The math is the
// row math of the single-sequence kernels; only the state base and the KV
// prefix length become per-span.
// state is the layer base of sequence 0; sequence slot s starts at
// state + s * seq_stride elements.
void conv1d_causal_silu_seg(const Bf16* x, const Bf16* w, Bf16* state, Bf16* out,
                            const SeqSpans& spans, int nseq, int channels,
                            size_t seq_stride, cudaStream_t stream);
void conv1d_causal_silu_split_seg(const Bf16* x, const Bf16* w, Bf16* state,
                                  Bf16* q, Bf16* k, Bf16* v,
                                  const SeqSpans& spans, int nseq, int wq,
                                  int wk, int wv, size_t seq_stride,
                                  cudaStream_t stream);
void prep_qkv_rope_kvcache_seg(const Bf16* qg, const Bf16* k_in,
                               const Bf16* v_in, const Bf16* q_norm_w,
                               const Bf16* k_norm_w, const Bf16* cos,
                               const Bf16* sin, Bf16* q_out, Bf16* gate_out,
                               Bf16* k_cache, Bf16* v_cache,
                               const SeqSpans& spans, int nseq, int rows,
                               size_t seq_stride, int n_q_heads, int n_kv_heads,
                               int head_dim, int rotary_dim, float eps,
                               cudaStream_t stream);
void gated_delta_rule_seg(float* state, const float* qn, const float* kn,
                          const Bf16* v, const float* beta, const float* g,
                          Bf16* out, const SeqSpans& spans, int nseq, int k_heads,
                          int v_heads, int dim, size_t seq_stride,
                          cudaStream_t stream);
// q/out rows are workspace rows. k_cache holds one layer slot for every
// sequence; sequence `slot` starts at k_cache + slot * seq_stride (elements).
void gqa_attention_seg(const Bf16* q, const Bf16* k_cache, const Bf16* v_cache,
                       const Bf16* gate, Bf16* out, const SeqSpans& spans,
                       int nseq, size_t seq_stride, int n_q_heads,
                       int n_kv_heads, int head_dim, cudaStream_t stream);

}  // namespace umminfer::qwen38
