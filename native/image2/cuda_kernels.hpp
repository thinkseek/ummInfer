// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "tensor.hpp"

#include <cstdint>

namespace umminfer::image2 {

// ---- elementwise / norms -------------------------------------------------

// RMSNorm over the last dim, fp32 stats: out = x * rsqrt(mean(x^2)+eps) * w.
void rms_norm_lastdim(const Bf16* x, const Bf16* weight, Bf16* out, int rows, int dim,
                      float eps, cudaStream_t s);
// Affine-free LayerNorm over the last dim (DiT block norms, eps 1e-6).
void layer_norm_no_affine(const Bf16* x, Bf16* out, int rows, int dim, float eps,
                          cudaStream_t s);
// DiT AdaLN modulate: out = normed * (1+scale) + shift; rows broadcast over
// one modulation vector (mod is per-batch, applied to every token row).
void adaln_modulate(const Bf16* normed, const Bf16* shift, const Bf16* scale, Bf16* out,
                    int rows, int dim, cudaStream_t s);
// VAE RMS norm over channel dim of NCHW: x/||x||_2 * sqrt(C) * gamma[c].
void rms_norm_nchw(const Bf16* x, const Bf16* gamma, Bf16* out, int n, int c, int hw,
                   cudaStream_t s);

void silu_inplace(Bf16* x, int n, cudaStream_t s);
void gelu_tanh_inplace(Bf16* x, int n, cudaStream_t s);
// out = silu(gate) * up.
void silu_mul(Bf16* out, const Bf16* gate, const Bf16* up, int n, cudaStream_t s);
void add_residual(Bf16* y, const Bf16* r, int n, cudaStream_t s);
// out = y + gate * x (DiT gated residuals; gate is a per-row vector).
void gated_residual(Bf16* y, const Bf16* x, const Bf16* gate, int rows, int dim,
                    cudaStream_t s);
void f32_to_bf16(const float* src, Bf16* dst, int n, cudaStream_t s);
void bf16_to_f32(const Bf16* src, float* dst, int n, cudaStream_t s);

// ---- RoPE ----------------------------------------------------------------

// Qwen2 text-encoder 1-D RoPE, rotate_half convention:
// q' = q*cos + rotate_half(q)*sin. x is [rows, n_heads, head_dim], positions
// are plain 0..rows-1 (text-only mrope has three identical axes).
void rope_rotate_half(Bf16* x, const float* cos, const float* sin, int rows, int n_heads,
                      int head_dim, cudaStream_t s);

// DiT complex-polar RoPE, adjacent-pair convention: channel pair (2k,2k+1)
// rotates by (cos_k, sin_k) as a complex multiply. freqs layout [rows, d/2]
// complex given as separate cos/sin arrays of length rows*d/2 row-major.
void rope_complex_pairs(Bf16* x, const float* cos, const float* sin, int rows,
                        int n_heads, int head_dim, cudaStream_t s);

// ---- attention ------------------------------------------------------------

// Non-causal multi-head attention with full (non-repeated) heads, used for
// the DiT joint stream and the VAE spatial block. q/k/v [B,S,H,D] row-major
// in (S,H,D) order; online-softmax flash kernel, fp32 accumulation.
void flash_attn(const Bf16* q, const Bf16* k, const Bf16* v, Bf16* out, int batch, int heads,
                int sq, int sk, int head_dim, cudaStream_t s);

// Causal GQA attention for the text encoder: 28 q heads / 4 kv heads (group
// 7), lower-triangular mask, seq <= ~600 so a one-block-per-(head,token)
// two-pass kernel suffices. q [B,Sq,Hq,D], k/v [B,Sk,Hkv,D].
void gqa_causal_attn(const Bf16* q, const Bf16* k, const Bf16* v, Bf16* out, int batch,
                     int hq, int hkv, int sq, int sk, int head_dim, cudaStream_t s);

// ---- VAE 2D convs (t=1 causal-conv3d reduction) --------------------------

// 3x3 conv, padding 1, stride 1 + bias: in [n,ci,h,w], weight [co,ci,3,3] (the
// last time slice of the CausalConv3d kernel), out [n,co,h,w].
void conv2d_3x3(const Bf16* in, const Bf16* weight, const Bf16* bias, Bf16* out, int n,
                int ci, int co, int h, int width, cudaStream_t s);
// 1x1 conv + bias.
void conv2d_1x1(const Bf16* in, const Bf16* weight, const Bf16* bias, Bf16* out, int n,
                int ci, int co, int hw, cudaStream_t s);
// Nearest-neighbor 2x upsample (nearest-exact, output i reads floor(i/2)).
void upsample_nearest_2x(const Bf16* in, Bf16* out, int n, int c, int h, int w,
                         cudaStream_t s);
// fp32 nearest 2x upsample matching diffusers QwenImageUpsample (interpolate
// in float32, cast back to bf16). in/out are BF16; the interpolation itself
// runs in fp32 to avoid error accumulation across the three decoder stages.
void upsample_nearest_2x_fp32(const Bf16* in, Bf16* out, int n, int c, int h, int w,
                              cudaStream_t s);

// ---- misc -----------------------------------------------------------------

// ATen-style Philox4x32 Box-Muller normal sampler, fp32 then bf16 RNE.
// Self-consistent across runs (fixed seed -> identical bytes); bit-parity
// with torch.randn is not guaranteed, stage tests use golden latent blobs.
void randn_bf16_philox(Bf16* out, uint64_t seed, size_t n, cudaStream_t s);

// True-CFG combine: comb = neg + scale*(cond-neg), then per-token norm
// renormalization comb * ||cond|| / ||comb|| over the feature dim.
void cfg_combine(Bf16* out, const Bf16* cond, const Bf16* neg, float scale, int tokens,
                 int features, cudaStream_t s);

// VAE latent de-normalization z = lat * inv_std[c] + mean[c], 16 channels.
void denorm_latents(Bf16* z, const Bf16* lat, const float* mean, const float* inv_std,
                    int hw, cudaStream_t s);
// Clamp every element to [-1, 1].
void clamp_n1p1(Bf16* x, int n, cudaStream_t s);
// One Euler step entirely on device: sample += dt * velocity (both BF16),
// accumulating in fp32 and rounding back to BF16 like the fp32-upcast scheduler.
void euler_step_bf16(Bf16* sample, const Bf16* velocity, float dt, int n,
                     cudaStream_t s);
// fp32 Euler accumulation (sample + dt*velocity), matching the diffusers
// scheduler that upcasts the latent to fp32 for every step.
void euler_f32(float* sample, const float* velocity, float dt, int n, cudaStream_t s);
// Add a per-column bias to every row of [rows, cols] row-major tensor.
void bias_broadcast(Bf16* out, const Bf16* bias, int rows, int cols, cudaStream_t s);
// Gather embedding rows: out[r, :] = table[ids[r], :].
void gather_rows(const Bf16* table, const int* ids, Bf16* out, int rows, int dim,
                 cudaStream_t s);
// Split an interleaved NCHW QKV tensor [n, 3c, h, w] into three [n, c, h, w].
void split_qkv_nchw(const Bf16* qkv, Bf16* q, Bf16* k, Bf16* v, int n, int c, int hw,
                    cudaStream_t s);
// Transpose a single-channel-group NCHW [c, hw] into token-major [hw, c] and
// back, used to bridge NCHW convolutions and the (token,head,d) attention.
void nchw_to_tokens(const Bf16* nchw, Bf16* tokens, int c, int hw, cudaStream_t s);
void tokens_to_nchw(const Bf16* tokens, Bf16* nchw, int c, int hw, cudaStream_t s);
// Inverse interleave is not needed; attention output projects straight back.


// ---- Qwen-Image 2.1 additions ---------------------------------------------

// Zero-centered RMSNorm: out = (x * rsqrt(mean(x^2)+eps)) * (w + 1), fp32 math.
void zero_center_rms_norm(const Bf16* x, const Bf16* weight, Bf16* out, int rows, int dim,
                          float eps, cudaStream_t s);

// LayerNorm (no affine) then per-token scale from two modulation rows:
// out[r] = ln(x[r]) * (1 + (is_target[r] ? scale_real : scale_zero)).
void layer_norm_mod_select(const Bf16* x, const Bf16* scale_real, const Bf16* scale_zero,
                           const int* is_target, Bf16* out, int rows, int dim, float eps,
                           cudaStream_t s);

// out[r] = x[r] * (1 + selected scale).
void scale_apply_select(const Bf16* x, const Bf16* scale_real, const Bf16* scale_zero,
                        const int* is_target, Bf16* out, int rows, int dim,
                        cudaStream_t s);

// y[r] += tanh(selected gate) * x[r].
void gated_residual_tanh(Bf16* y, const Bf16* x, const Bf16* gate_real,
                         const Bf16* gate_zero, const int* is_target, int rows, int dim,
                         cudaStream_t s);

// Spatial 2x pixel-shuffle shortcut (QwenImage21DupUp3D at T=1, first chunk).
// in [cin,H,W] NCHW -> out [cout,2H,2W] NCHW. factor_t is 1 or 2; for 2 the
// first duplicated frame is dropped, leaving temporal slice index 1.
void dupup_2x(const Bf16* in, Bf16* out, int cin, int cout, int h, int w, int factor_t,
              cudaStream_t s);

// Single-head spatial self-attention of the VAE mid block, fp32 math.
// q/k/v are NCHW [c, hw]; scores and out are workspace buffers.
void vae_attn_f32(const Bf16* q, const Bf16* k, const Bf16* v, float* scores,
                  float* out_tokens, Bf16* out_nchw, int c, int hw, cudaStream_t s);

// z = bf16(bf16(lat * std) + mean), 64 latent channels.
void denorm_latents_std(const Bf16* lat, const float* mean, const float* std, Bf16* z,
                        int channels, int hw, cudaStream_t s);

}  // namespace umminfer::image2
