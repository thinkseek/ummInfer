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

// Descriptor for fused 3-way Q/K/V BF16 projection sharing the same input x.
struct QkvGemvDesc {
  const Bf16* w[3];
  Bf16* out[3];
  int n_out[3];
};

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
// Out-of-place SnakeBeta: out[c,t] = snake_beta(in[c,t]), leaving `in` intact
// as a residual without an extra cudaMemcpyAsync.
void snake_beta_out_f32(const float* in, float* out, const float* alpha,
                        const float* beta, int channels, int length,
                        cudaStream_t stream);
void f32_to_bf16(const float* src, Bf16* dst, int n, cudaStream_t stream);
void bf16_to_f32(const Bf16* src, float* dst, int n, cudaStream_t stream);
void clamp_f32(float* x, int n, float lo, float hi, cudaStream_t stream);

// ---- gather / matmul / fused decode projections ------------------------------

template <typename T>
void gather_rows_t(const T* table, const int* ids, T* out, int m, int dim,
                   cudaStream_t stream);
// Fuses the 16 per-codebook embedding lookups + d_pad residual sum into `dst`,
// and optionally records d_hist[0] = d_codes[0] and *d_step_pos = seq_len on
// device (eliminating host-to-device staging copies in the talker loop).
void sum_frame_embeddings_bf16(const Bf16* const* tables, const int* d_codes,
                               const Bf16* d_pad, Bf16* dst, int* d_hist_slot,
                               int* d_step_pos, int seq_len, int groups, int dim,
                               cudaStream_t stream);
// y[m,out] = x[m,k] * W[out,k]^T (+ bias). Vectorized SM90 warp-per-row path.
template <typename T>
void gemm_t(T* y, const T* x, const T* w, const T* bias, int m, int k, int out,
            cudaStream_t stream);
// Fused BF16 linear + residual add: y[m,out] = to_bf(y[m,out] + to_bf(x*W^T)).
void gemm_bf16_add_residual(Bf16* y, const Bf16* x, const Bf16* w, int m, int k,
                            int out, cudaStream_t stream);
// Fused 3-way Q/K/V BF16 projection sharing `x`.
void gemv_qkv_bf16(const QkvGemvDesc& desc, const Bf16* x, int m, int k,
                   cudaStream_t stream);
// Fused dual-output SwiGLU BF16 projection: out = silu(x*W_gate^T) * (x*W_up^T)
// with exact BF16 intermediate rounding.
void gemv_swiglu_bf16(const Bf16* w_gate, const Bf16* w_up, const Bf16* x,
                      Bf16* out, int m, int n_out, int k, cudaStream_t stream);
template <typename T>
void gemv_t(const T* w, const T* x, float* out, int n_out, int k, cudaStream_t stream);
int argmax_f32(const float* logits, int n, int* scratch, cudaStream_t stream);
// Same kernel and same tie-breaking, but the winning index stays on the device:
// it is written to `out` and the caller reads it after its own synchronisation.
// Used by the talker frame loop, which must not stall per sub-codebook.
void argmax_f32_dev(const float* logits, int n, int* out, cudaStream_t stream);
// Set logits[idx] = -inf for every idx in [lo, hi) except keep.
void suppress_logits(float* logits, int lo, int hi, int keep, cudaStream_t stream);
// HF repetition penalty over the emitted id history (applied before argmax).
// Each distinct logit id is penalized at most once, regardless of duplicates.
void repetition_penalty_f32(float* logits, const int* history, int n_history,
                            int n_logits, float penalty, cudaStream_t stream);
// Force one id to -inf (anti-degeneracy consecutive-run ban).
void ban_logit_f32(float* logits, int id, cudaStream_t stream);

// ---- rotary / attention ------------------------------------------------------

// Half-rotation (GPT-NeoX) RoPE, standard 1D positions. Equal mRoPE axes in
// this model make the talker path numerically identical to this kernel.
template <typename T>
void rope_half_t(T* x, const int* positions, const float* inv_freq, int n,
                 int num_heads, int head_dim, cudaStream_t stream);
// Half-rotation RoPE with contiguous positions [pos_base, pos_base + n),
// eliminating host-to-device position array copies.
void rope_half_offset_f32(float* x, int pos_base, const float* inv_freq, int n,
                          int num_heads, int head_dim, cudaStream_t stream);

// Fused Q-norm + K-norm + half-rotation RoPE + K/V cache append for head_dim=128.
// Scalar-position variant (prefill & fixed-shape code predictor):
// positions are [pos_base .. pos_base + m - 1], cache slots start at history_base.
// When v_src != nullptr && v_cache != nullptr, V is also copied into v_cache.
void qk_norm_rope_kv_append_bf16(Bf16* q, const Bf16* k_src, const Bf16* v_src,
                                 const Bf16* q_norm_w, const Bf16* k_norm_w,
                                 const float* inv_freq, Bf16* k_cache,
                                 Bf16* v_cache, int m, int n_q_heads,
                                 int n_kv_heads, int head_dim, int pos_base,
                                 int history_base, float eps,
                                 cudaStream_t stream);

// Device-position variant for CUDA-graphed single-token (m == 1) talker decode:
// reads pos = *d_pos_ptr on device and appends K/V at slot `pos`.
void qk_norm_rope_kv_append_devpos_bf16(Bf16* q, const Bf16* k_src,
                                        const Bf16* v_src, const Bf16* q_norm_w,
                                        const Bf16* k_norm_w,
                                        const float* inv_freq, Bf16* k_cache,
                                        Bf16* v_cache, int n_q_heads,
                                        int n_kv_heads, int head_dim,
                                        const int* d_pos_ptr, float eps,
                                        cudaStream_t stream);

// GQA attention. mask: 0 causal full, 1 causal sliding window (window_len),
// 2 bidirectional fixed windows (ASR-style, unused by TTS). q [nq,nqh,d],
// k/v [nkv,nkh,d].
template <typename T>
void gqa_attention_t(const T* q, const T* k, const T* v, T* out, int nq, int nkv,
                     int n_q_heads, int n_kv_heads, int head_dim, int mask,
                     int window_len, int q_offset, cudaStream_t stream);

// Device-position causal GQA attention (nq == 1, head_dim == 128) for
// CUDA-graphed talker decode: attends over [0, *d_pos_ptr + 1).
void gqa_attention_devpos_bf16(const Bf16* q, const Bf16* k, const Bf16* v,
                               Bf16* out, int n_q_heads, int n_kv_heads,
                               int head_dim, const int* d_pos_ptr,
                               cudaStream_t stream);

// ---- 1-D convolution family & codec helpers (all F32) ------------------------

// Causal Conv1d stride 1. Left zero pad = (kernel-1)*dilation, length preserved.
// groups=1 (dense) or groups=channels (depthwise; weight [c,1,k]).
void conv1d_causal_f32(const float* in, const float* w, const float* bias,
                       float* out, int channels_in, int channels_out, int length,
                       int kernel, int dilation, int groups, cudaStream_t stream);
// Stateful causal Conv1d: when hist != nullptr, reads left-pad taps from
// hist [channels_in, (kernel-1)*dilation] instead of zeros.
void conv1d_causal_stateful_f32(const float* in, const float* hist,
                                const float* w, const float* bias, float* out,
                                int channels_in, int channels_out, int length,
                                int kernel, int dilation, int groups,
                                cudaStream_t stream);
// Updates causal Conv1d history [channels, pad] in-place after processing `in`
// [channels, length]. Race-free for both length >= pad and length < pad.
void update_conv_history_f32(float* hist, const float* in, int channels, int pad,
                             int length, cudaStream_t stream);
// ConvTranspose1d with right trim of (kernel-stride) samples; net length L*stride.
void conv_transpose1d_f32(const float* in, const float* w, const float* bias,
                          float* out, int ch_in, int ch_out, int length, int kernel,
                          int stride, cudaStream_t stream);
// Stateful ConvTranspose1d (kernel == 2*stride): when hist != nullptr, reads
// the t_local - 1 tap at t_local == 0 from hist [ch_in].
void conv_transpose1d_stateful_f32(const float* in, const float* hist,
                                   const float* w, const float* bias, float* out,
                                   int ch_in, int ch_out, int length, int kernel,
                                   int stride, cudaStream_t stream);
// Copies the last sample in[c, length - 1] into hist[c] for ConvTranspose1d state.
void save_last_sample_f32(float* hist, const float* in, int channels, int length,
                          cudaStream_t stream);
// RVQ codebook dequant: emb[c,d] = embedding_sum[c,d] / max(usage[c], eps).
void codebook_dequant_f32(const float* embedding_sum, const float* cluster_usage,
                          float* emb, int codes, int dim, float eps,
                          cudaStream_t stream);
// Fused RVQ gather + sum across [first_group, first_group + count) + transpose
// directly into channel-first out [vq_dim, frames].
void rvq_gather_sum_transposed_f32(const float* codebooks, const int* codes,
                                   float* out, int first_group, int count,
                                   int frames, int groups, int codebook_size,
                                   int vq_dim, cudaStream_t stream);
// Compacts non-overlapping rolling KV cache slabs [src_slot, src_slot + keep_len)
// down to [0, keep_len) across all (n_layers * 2) K/V buffers in one launch.
void shift_kv_window_f32(float* kv, int num_slabs, int slot_cap, int width,
                         int src_slot, int keep_len, cudaStream_t stream);
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
