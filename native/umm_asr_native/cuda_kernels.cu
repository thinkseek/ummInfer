// SPDX-License-Identifier: Apache-2.0
#include "cuda_kernels.hpp"

#include <cfloat>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "decode_helpers.hpp"

namespace umminfer::native {
namespace {

constexpr int kBlock = 256;

inline bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0u;
}
inline bool aligned8(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 7u) == 0u;
}
inline bool aligned4(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 3u) == 0u;
}

__device__ __forceinline__ float bf(const Bf16& x) { return __bfloat162float(x); }

__device__ __forceinline__ float gelu_f32(float x) {
  // transformers default gelu: 0.5 * x * (1 + erf(x / sqrt(2)))
  return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

__device__ __forceinline__ uint4 ld_cs_u4(const void* p) {
  return __ldcs(reinterpret_cast<const uint4*>(p));
}
__device__ __forceinline__ uint4 ld_ro_u4(const void* p) {
  return __ldg(reinterpret_cast<const uint4*>(p));
}

__device__ __forceinline__ float block_reduce_sum(float partial, float* scratch) {
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int nwarps = (blockDim.x + 31) / 32;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    partial += __shfl_down_sync(0xffffffffu, partial, off);
  if (lane == 0) scratch[wid] = partial;
  __syncthreads();
  float total = 0.0f;
  if (wid == 0) {
    total = lane < nwarps ? scratch[lane] : 0.0f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      total += __shfl_down_sync(0xffffffffu, total, off);
  }
  __shared__ float s_total;
  if (threadIdx.x == 0) s_total = total;
  __syncthreads();
  return s_total;
}

// ---- Norms (vectorized uint4 + register-cached + fallback) ------------------

__global__ void rms_norm_scalar_kernel(const Bf16* __restrict__ x,
                                       const Bf16* __restrict__ w,
                                       Bf16* __restrict__ out, int dim,
                                       float eps) {
  const int row = blockIdx.x;
  const Bf16* xr = x + static_cast<size_t>(row) * dim;
  Bf16* orow = out + static_cast<size_t>(row) * dim;
  __shared__ float warp_sums[32];
  float sum = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = bf(xr[i]);
    sum += v * v;
  }
  const float scale = rsqrtf(block_reduce_sum(sum, warp_sums) / dim + eps);
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    orow[i] = __float2bfloat16(bf(xr[i]) * scale * bf(w[i]));
}

// Single-warp vectorized BF16 RMSNorm for dim <= 256 (dim % 8 == 0, 16B aligned).
__global__ void __launch_bounds__(32) rms_norm_warp_vec_bf16_kernel(
    const Bf16* __restrict__ x, const Bf16* __restrict__ w,
    Bf16* __restrict__ out, int dim, float eps) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  const int vecs = dim >> 3;
  const uint4* xv =
      reinterpret_cast<const uint4*>(x + static_cast<size_t>(row) * dim);
  const uint4* wv = reinterpret_cast<const uint4*>(w);
  uint4* ov = reinterpret_cast<uint4*>(out + static_cast<size_t>(row) * dim);

  float xf[8];
  float sum = 0.0f;
  if (lane < vecs) {
    const uint4 raw = __ldg(xv + lane);
    const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&raw);
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 f2 = __bfloat1622float2(xb[e]);
      xf[2 * e] = f2.x;
      xf[2 * e + 1] = f2.y;
      sum += f2.x * f2.x + f2.y * f2.y;
    }
  }
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    sum += __shfl_xor_sync(0xffffffffu, sum, off);
  const float inv = rsqrtf(sum / dim + eps);

  if (lane < vecs) {
    const uint4 wraw = __ldg(wv + lane);
    const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wraw);
    uint4 oraw;
    __nv_bfloat162* ob = reinterpret_cast<__nv_bfloat162*>(&oraw);
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 wf = __bfloat1622float2(wb[e]);
      ob[e] = __floats2bfloat162_rn(xf[2 * e] * inv * wf.x,
                                    xf[2 * e + 1] * inv * wf.y);
    }
    ov[lane] = oraw;
  }
}

// 256-thread vectorized BF16 RMSNorm (dim % 8 == 0, 16B aligned).
// Caches up to 3 uint4s per thread (dim <= 6144) in registers so x is read once.
__global__ void __launch_bounds__(256) rms_norm_vec_bf16_kernel(
    const Bf16* __restrict__ x, const Bf16* __restrict__ w,
    Bf16* __restrict__ out, int dim, float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31, wid = tid >> 5;
  const int vecs = dim >> 3;
  const uint4* xv =
      reinterpret_cast<const uint4*>(x + static_cast<size_t>(row) * dim);
  const uint4* wv = reinterpret_cast<const uint4*>(w);
  uint4* ov = reinterpret_cast<uint4*>(out + static_cast<size_t>(row) * dim);

  __shared__ float warp_sums[8];
  __shared__ float s_inv;

  uint4 reg_x[3];
  const bool cached = (vecs <= 768);
  float sum = 0.0f;
  int idx_reg = 0;
  for (int i = tid; i < vecs; i += 256) {
    const uint4 raw = __ldg(xv + i);
    if (cached) reg_x[idx_reg++] = raw;
    const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&raw);
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 f2 = __bfloat1622float2(xb[e]);
      sum += f2.x * f2.x + f2.y * f2.y;
    }
  }
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    sum += __shfl_down_sync(0xffffffffu, sum, off);
  if (lane == 0) warp_sums[wid] = sum;
  __syncthreads();
  if (wid == 0) {
    sum = lane < 8 ? warp_sums[lane] : 0.0f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) s_inv = rsqrtf(sum / dim + eps);
  }
  __syncthreads();
  const float inv = s_inv;

  idx_reg = 0;
  for (int i = tid; i < vecs; i += 256) {
    const uint4 raw = cached ? reg_x[idx_reg++] : __ldg(xv + i);
    const uint4 wraw = __ldg(wv + i);
    const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&raw);
    const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wraw);
    uint4 oraw;
    __nv_bfloat162* ob = reinterpret_cast<__nv_bfloat162*>(&oraw);
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 xf = __bfloat1622float2(xb[e]);
      const float2 wf = __bfloat1622float2(wb[e]);
      ob[e] = __floats2bfloat162_rn(xf.x * inv * wf.x, xf.y * inv * wf.y);
    }
    ov[i] = oraw;
  }
}

// 256-thread vectorized BF16 LayerNorm (dim % 8 == 0, 16B aligned).
// Caches up to 2 uint4s per thread (dim <= 4096) in registers so x is read once
// across mean, variance, and output passes.
__global__ void __launch_bounds__(256) layer_norm_vec_bf16_kernel(
    const Bf16* __restrict__ x, const Bf16* __restrict__ w,
    const Bf16* __restrict__ b, Bf16* __restrict__ out, int dim, float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31, wid = tid >> 5;
  const int vecs = dim >> 3;
  const uint4* xv =
      reinterpret_cast<const uint4*>(x + static_cast<size_t>(row) * dim);
  const uint4* wv = reinterpret_cast<const uint4*>(w);
  const uint4* bv = reinterpret_cast<const uint4*>(b);
  uint4* ov = reinterpret_cast<uint4*>(out + static_cast<size_t>(row) * dim);

  __shared__ float warp_sums[8];
  __shared__ float s_mean, s_inv;

  uint4 reg_x[2];
  const bool cached = (vecs <= 512);
  float acc = 0.0f;
  int idx_reg = 0;
  for (int i = tid; i < vecs; i += 256) {
    const uint4 raw = __ldg(xv + i);
    if (cached) reg_x[idx_reg++] = raw;
    const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&raw);
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 f2 = __bfloat1622float2(xb[e]);
      acc += f2.x + f2.y;
    }
  }
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) warp_sums[wid] = acc;
  __syncthreads();
  if (wid == 0) {
    acc = lane < 8 ? warp_sums[lane] : 0.0f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) s_mean = acc / dim;
  }
  __syncthreads();
  const float mean = s_mean;

  acc = 0.0f;
  idx_reg = 0;
  for (int i = tid; i < vecs; i += 256) {
    const uint4 raw = cached ? reg_x[idx_reg++] : __ldg(xv + i);
    const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&raw);
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 f2 = __bfloat1622float2(xb[e]);
      const float d0 = f2.x - mean;
      const float d1 = f2.y - mean;
      acc += d0 * d0 + d1 * d1;
    }
  }
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) warp_sums[wid] = acc;
  __syncthreads();
  if (wid == 0) {
    acc = lane < 8 ? warp_sums[lane] : 0.0f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) s_inv = rsqrtf(acc / dim + eps);
  }
  __syncthreads();
  const float inv = s_inv;

  idx_reg = 0;
  for (int i = tid; i < vecs; i += 256) {
    const uint4 raw = cached ? reg_x[idx_reg++] : __ldg(xv + i);
    const uint4 wraw = __ldg(wv + i);
    const uint4 braw = __ldg(bv + i);
    const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&raw);
    const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wraw);
    const __nv_bfloat162* bb = reinterpret_cast<const __nv_bfloat162*>(&braw);
    uint4 oraw;
    __nv_bfloat162* ob = reinterpret_cast<__nv_bfloat162*>(&oraw);
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 xf = __bfloat1622float2(xb[e]);
      const float2 wf = __bfloat1622float2(wb[e]);
      const float2 bf2 = __bfloat1622float2(bb[e]);
      ob[e] = __floats2bfloat162_rn((xf.x - mean) * inv * wf.x + bf2.x,
                                    (xf.y - mean) * inv * wf.y + bf2.y);
    }
    ov[i] = oraw;
  }
}

__global__ void layer_norm_scalar_kernel(const Bf16* __restrict__ x,
                                         const Bf16* __restrict__ w,
                                         const Bf16* __restrict__ b,
                                         Bf16* __restrict__ out, int dim,
                                         float eps) {
  const int row = blockIdx.x;
  const Bf16* xr = x + static_cast<size_t>(row) * dim;
  Bf16* orow = out + static_cast<size_t>(row) * dim;
  __shared__ float warp_sums[32];
  float acc = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) acc += bf(xr[i]);
  const float mean = block_reduce_sum(acc, warp_sums) / dim;
  acc = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float d = bf(xr[i]) - mean;
    acc += d * d;
  }
  const float inv = rsqrtf(block_reduce_sum(acc, warp_sums) / dim + eps);
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    orow[i] = __float2bfloat16((bf(xr[i]) - mean) * inv * bf(w[i]) + bf(b[i]));
}

// ---- Elementwise / Gather / Frontend ---------------------------------------

__global__ void gelu_scalar_kernel(Bf16* x, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = __float2bfloat16(gelu_f32(bf(x[i])));
}

__global__ void gelu_vec_bf16_kernel(Bf16* __restrict__ x, int vecs) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= vecs) return;
  uint4* xv = reinterpret_cast<uint4*>(x);
  const uint4 raw = xv[i];
  const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&raw);
  uint4 oraw;
  __nv_bfloat162* ob = reinterpret_cast<__nv_bfloat162*>(&oraw);
  #pragma unroll
  for (int e = 0; e < 4; ++e) {
    const float2 f2 = __bfloat1622float2(xb[e]);
    ob[e] = __floats2bfloat162_rn(gelu_f32(f2.x), gelu_f32(f2.y));
  }
  xv[i] = oraw;
}

__global__ void bias_kernel(Bf16* y, const Bf16* bias, int dim) {
  const int r = blockIdx.x;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const size_t idx = static_cast<size_t>(r) * dim + i;
    y[idx] = __float2bfloat16(bf(y[idx]) + bf(bias[i]));
  }
}

__global__ void residual_scalar_kernel(Bf16* y, const Bf16* r, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __float2bfloat16(bf(y[i]) + bf(r[i]));
}

__global__ void residual_vec_bf16_kernel(Bf16* __restrict__ y,
                                         const Bf16* __restrict__ r, int vecs) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= vecs) return;
  uint4* yv = reinterpret_cast<uint4*>(y);
  const uint4* rv = reinterpret_cast<const uint4*>(r);
  const uint4 yraw = yv[i];
  const uint4 rraw = __ldg(rv + i);
  const __nv_bfloat162* yb = reinterpret_cast<const __nv_bfloat162*>(&yraw);
  const __nv_bfloat162* rb = reinterpret_cast<const __nv_bfloat162*>(&rraw);
  uint4 oraw;
  __nv_bfloat162* ob = reinterpret_cast<__nv_bfloat162*>(&oraw);
  #pragma unroll
  for (int e = 0; e < 4; ++e) {
    const float2 yf = __bfloat1622float2(yb[e]);
    const float2 rf = __bfloat1622float2(rb[e]);
    ob[e] = __floats2bfloat162_rn(yf.x + rf.x, yf.y + rf.y);
  }
  yv[i] = oraw;
}

__global__ void silu_mul_scalar_kernel(Bf16* out, const Bf16* gate,
                                       const Bf16* up, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float g = bf(gate[i]), u = bf(up[i]);
    out[i] = __float2bfloat16(u * g / (1.0f + expf(-g)));
  }
}

__global__ void silu_mul_vec_bf16_kernel(Bf16* __restrict__ out,
                                         const Bf16* __restrict__ gate,
                                         const Bf16* __restrict__ up,
                                         int vecs) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= vecs) return;
  uint4* ov = reinterpret_cast<uint4*>(out);
  const uint4* gv = reinterpret_cast<const uint4*>(gate);
  const uint4* uv = reinterpret_cast<const uint4*>(up);
  const uint4 graw = gv[i];
  const uint4 uraw = __ldg(uv + i);
  const __nv_bfloat162* gb = reinterpret_cast<const __nv_bfloat162*>(&graw);
  const __nv_bfloat162* ub = reinterpret_cast<const __nv_bfloat162*>(&uraw);
  uint4 oraw;
  __nv_bfloat162* ob = reinterpret_cast<__nv_bfloat162*>(&oraw);
  #pragma unroll
  for (int e = 0; e < 4; ++e) {
    const float2 gf = __bfloat1622float2(gb[e]);
    const float2 uf = __bfloat1622float2(ub[e]);
    ob[e] = __floats2bfloat162_rn(uf.x * gf.x / (1.0f + expf(-gf.x)),
                                  uf.y * gf.y / (1.0f + expf(-gf.y)));
  }
  ov[i] = oraw;
}

__global__ void f32_to_bf16_kernel(const float* src, Bf16* dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2bfloat16(src[i]);
}

__global__ void bf16_to_f32_kernel(const Bf16* src, float* dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = bf(src[i]);
}

__global__ void embed_scalar_kernel(const Bf16* table, const int* ids,
                                    Bf16* out, int m, int dim) {
  const int row = blockIdx.y;
  if (row >= m) return;
  const Bf16* src = table + static_cast<size_t>(ids[row]) * dim;
  Bf16* dst = out + static_cast<size_t>(row) * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) dst[i] = src[i];
}

__global__ void embed_vec_kernel(const Bf16* __restrict__ table,
                                 const int* __restrict__ ids,
                                 Bf16* __restrict__ out, int m, int dim) {
  const int row = blockIdx.y;
  if (row >= m) return;
  const int vecs = dim >> 3;
  const uint4* src = reinterpret_cast<const uint4*>(
      table + static_cast<size_t>(ids[row]) * dim);
  uint4* dst =
      reinterpret_cast<uint4*>(out + static_cast<size_t>(row) * dim);
  for (int i = threadIdx.x; i < vecs; i += blockDim.x) dst[i] = __ldg(src + i);
}

__global__ void replace_rows_scalar_kernel(Bf16* dst, const Bf16* src,
                                           int start, int n, int dim) {
  const int row = blockIdx.y;
  if (row >= n) return;
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    dst[static_cast<size_t>(start + row) * dim + i] =
        src[static_cast<size_t>(row) * dim + i];
}

__global__ void replace_rows_vec_kernel(Bf16* __restrict__ dst,
                                        const Bf16* __restrict__ src, int start,
                                        int n, int dim) {
  const int row = blockIdx.y;
  if (row >= n) return;
  const int vecs = dim >> 3;
  uint4* drow =
      reinterpret_cast<uint4*>(dst + static_cast<size_t>(start + row) * dim);
  const uint4* srow =
      reinterpret_cast<const uint4*>(src + static_cast<size_t>(row) * dim);
  for (int i = threadIdx.x; i < vecs; i += blockDim.x)
    drow[i] = __ldg(srow + i);
}

// ---- RoPE & Fused QK-Norm + RoPE + KV-Append --------------------------------

__global__ void rope_kernel(Bf16* x, const int* positions,
                            const float* inv_freq, int n, int num_heads,
                            int head_dim) {
  const int token = blockIdx.y;
  if (token >= n) return;
  const int half = head_dim / 2;
  const int pos = positions[token];
  for (int h = threadIdx.y; h < num_heads; h += blockDim.y) {
    Bf16* row = x + (static_cast<size_t>(token) * num_heads + h) * head_dim;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
      float s, c;
      sincosf(pos * __ldg(inv_freq + i), &s, &c);
      const float x1 = bf(row[i]), x2 = bf(row[i + half]);
      row[i] = __float2bfloat16(x1 * c - x2 * s);
      row[i + half] = __float2bfloat16(x2 * c + x1 * s);
    }
  }
}

// Single-warp-per-head fused Q-norm + K-norm + RoPE + K/V cache append for
// head_dim == 128. Zero shared memory, zero __syncthreads().
template <bool kDevPos, bool kCopyV>
__global__ void __launch_bounds__(32) qk_norm_rope_kv_append_128_kernel(
    Bf16* __restrict__ q, const Bf16* __restrict__ k_src,
    const Bf16* __restrict__ v_src, const Bf16* __restrict__ q_norm_w,
    const Bf16* __restrict__ k_norm_w, const float* __restrict__ inv_freq,
    Bf16* __restrict__ k_cache, Bf16* __restrict__ v_cache, int n_q_heads,
    int n_kv_heads, int pos_base_arg, int history_base_arg,
    const int* __restrict__ d_pos_ptr, float eps) {
  constexpr int kHeadDim = 128;
  const int b = blockIdx.x;
  const int tok = blockIdx.y;
  const int lane = threadIdx.x;
  const int pos_base = kDevPos ? *d_pos_ptr : pos_base_arg;
  const int hist_base = kDevPos ? pos_base : history_base_arg;

  if constexpr (kCopyV) {
    if (b >= n_q_heads + n_kv_heads) {
      const int kvh = b - (n_q_heads + n_kv_heads);
      const Bf16* src =
          v_src + (static_cast<size_t>(tok) * n_kv_heads + kvh) * kHeadDim;
      Bf16* dst = v_cache +
                  (static_cast<size_t>(hist_base + tok) * n_kv_heads + kvh) *
                      kHeadDim;
      reinterpret_cast<uint2*>(dst)[lane] =
          __ldg(reinterpret_cast<const uint2*>(src) + lane);
      return;
    }
  }

  const bool is_q = (b < n_q_heads);
  const int h = is_q ? b : (b - n_q_heads);
  const Bf16* in_row =
      is_q ? (q + (static_cast<size_t>(tok) * n_q_heads + h) * kHeadDim)
           : (k_src + (static_cast<size_t>(tok) * n_kv_heads + h) * kHeadDim);
  Bf16* out_row =
      is_q ? (q + (static_cast<size_t>(tok) * n_q_heads + h) * kHeadDim)
           : (k_cache +
              (static_cast<size_t>(hist_base + tok) * n_kv_heads + h) *
                  kHeadDim);
  const Bf16* nw = is_q ? q_norm_w : k_norm_w;

  const int i0 = lane;
  const int i1 = lane + 32;
  const int i2 = lane + 64;
  const int i3 = lane + 96;

  const float x0 = bf(in_row[i0]);
  const float x1 = bf(in_row[i1]);
  const float x2 = bf(in_row[i2]);
  const float x3 = bf(in_row[i3]);

  float ss = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    ss += __shfl_xor_sync(0xffffffffu, ss, off);
  const float inv = rsqrtf(ss / static_cast<float>(kHeadDim) + eps);

  // Exact BF16 rounding between RMSNorm and RoPE, matching unfused path.
  const float n0 = bf(__float2bfloat16(x0 * inv * bf(__ldg(nw + i0))));
  const float n1 = bf(__float2bfloat16(x1 * inv * bf(__ldg(nw + i1))));
  const float n2 = bf(__float2bfloat16(x2 * inv * bf(__ldg(nw + i2))));
  const float n3 = bf(__float2bfloat16(x3 * inv * bf(__ldg(nw + i3))));

  const int pos = pos_base + tok;
  float s0, c0, s1, c1;
  sincosf(pos * __ldg(inv_freq + i0), &s0, &c0);
  sincosf(pos * __ldg(inv_freq + i1), &s1, &c1);

  out_row[i0] = __float2bfloat16(n0 * c0 - n2 * s0);
  out_row[i2] = __float2bfloat16(n2 * c0 + n0 * s0);
  out_row[i1] = __float2bfloat16(n1 * c1 - n3 * s1);
  out_row[i3] = __float2bfloat16(n3 * c1 + n1 * s1);
}

// ---- SM90 PTX Tensor Core MMA primitives -----------------------------------

__device__ __forceinline__ void cp_async16(void* dst, const void* src,
                                           bool pred) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  const unsigned bytes = pred ? 16u : 0u;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s),
               "l"(src), "r"(bytes));
}
__device__ __forceinline__ void cp_commit() {
  asm volatile("cp.async.commit_group;\n");
}
template <int kPending>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(kPending));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* p) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(a));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], const void* p) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(a));
}
__device__ __forceinline__ void mma_m16n8k16(float (&d)[4],
                                             const uint32_t (&a)[4],
                                             const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---- Audio Conv2d 3x3 stride 2 pad 1 + GELU --------------------------------

// Direct kernel for Conv2d1 (ci == 1) and small-channel fallback.
__global__ void conv3x3s2_direct_kernel(const Bf16* __restrict__ in,
                                        const Bf16* __restrict__ weight,
                                        const Bf16* __restrict__ bias,
                                        Bf16* __restrict__ out, int b, int ci,
                                        int co, int h, int w, int oh, int ow) {
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = b * co * oh * ow;
  if (o >= total) return;
  const int owx = o % ow;
  const int ohy = (o / ow) % oh;
  const int oc = (o / (ow * oh)) % co;
  const int ob = o / (ow * oh * co);
  float acc = bias ? bf(__ldg(bias + oc)) : 0.0f;
  for (int ic = 0; ic < ci; ++ic) {
    const size_t in_base = (static_cast<size_t>(ob) * ci + ic) * h * w;
    const size_t w_base = (static_cast<size_t>(oc) * ci + ic) * 9;
    #pragma unroll
    for (int kh = 0; kh < 3; ++kh) {
      const int ih = ohy * 2 + kh - 1;
      if (ih < 0 || ih >= h) continue;
      #pragma unroll
      for (int kw = 0; kw < 3; ++kw) {
        const int iw = owx * 2 + kw - 1;
        if (iw < 0 || iw >= w) continue;
        acc += bf(__ldg(in + in_base + static_cast<size_t>(ih) * w + iw)) *
               bf(__ldg(weight + w_base + kh * 3 + kw));
      }
    }
  }
  out[o] = __float2bfloat16(gelu_f32(acc));
}

// SM90 64x64x64 BF16 Tensor Core implicit-GEMM Conv2d 3x3s2p1 + GELU.
// Zero extra VRAM (no im2col buffer), 18 KiB static shared memory.
// Accelerates conv2d2 (480->480) and conv2d3 (480->480) by 30-50x.
__global__ void __launch_bounds__(128, 2) conv3x3s2_gelu_mma_kernel(
    const Bf16* __restrict__ in, const Bf16* __restrict__ weight,
    const Bf16* __restrict__ bias, Bf16* __restrict__ out, int b, int ci,
    int co, int h, int w, int oh, int ow) {
  constexpr int kBM = 64;
  constexpr int kBN = 64;
  constexpr int kBKP = 72;

  alignas(16) __shared__ uint16_t as_raw[kBM * kBKP];
  alignas(16) __shared__ uint16_t bs_raw[kBN * kBKP];
  Bf16* as = reinterpret_cast<Bf16*>(as_raw);
  Bf16* bs = reinterpret_cast<Bf16*>(bs_raw);

  const AsrGemmTileCoord coord = asr_swizzle_gemm_cta(
      static_cast<int>(blockIdx.x), static_cast<int>(blockIdx.y),
      static_cast<int>(gridDim.x), static_cast<int>(gridDim.y));
  const int m0 = coord.tile_m * kBM;
  const int n0 = coord.tile_n * kBN;
  const int oho = oh * ow;
  const int total_m = b * oho;
  const int hw = h * w;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int wm = warp >> 1;
  const int wn = warp & 1;

  // Precompute spatial pixel coordinates for this thread's A-tile row.
  const int a_m = tid & 63;
  const int a_k0 = tid >> 6;
  const int p_a = m0 + a_m;
  const bool p_a_valid = (p_a < total_m);
  const int a_ob = p_a_valid ? (p_a / oho) : 0;
  const int a_rem = p_a_valid ? (p_a - a_ob * oho) : 0;
  const int a_ohy = a_rem / ow;
  const int a_owx = a_rem - a_ohy * ow;

  // Precompute B-tile thread mapping (32 consecutive ic per warp).
  const int b_k = tid & 31;
  const int b_n0 = (tid >> 5) * 16;

  // MMA ldmatrix lane coordinates.
  const int a_row = (lane & 7) + ((lane & 8) ? 8 : 0);
  const int a_col = (lane & 16) ? 8 : 0;
  const int b_row = lane & 7;
  const int b_col = (lane & 8) ? 8 : 0;

  float acc[2][4][4];
  #pragma unroll
  for (int mt = 0; mt < 2; ++mt)
    #pragma unroll
    for (int nt = 0; nt < 4; ++nt)
      #pragma unroll
      for (int q = 0; q < 4; ++q) acc[mt][nt][q] = 0.0f;

  const Bf16 zero_bf = __float2bfloat16(0.0f);

  for (int kh = 0; kh < 3; ++kh) {
    const int ih = a_ohy * 2 + kh - 1;
    const bool h_ok = p_a_valid && (ih >= 0) && (ih < h);
    for (int kw = 0; kw < 3; ++kw) {
      const int iw = a_owx * 2 + kw - 1;
      const bool hw_ok = h_ok && (iw >= 0) && (iw < w);
      const int tap = kh * 3 + kw;
      const size_t in_spatial =
          hw_ok ? (static_cast<size_t>(a_ob) * ci * hw +
                   static_cast<size_t>(ih) * w + iw)
                : 0;
      for (int ic0 = 0; ic0 < ci; ic0 += 64) {
        #pragma unroll
        for (int s = 0; s < 32; ++s) {
          const int kk = a_k0 + s * 2;
          const int ic = ic0 + kk;
          as[a_m * kBKP + kk] =
              (hw_ok && ic < ci)
                  ? __ldg(in + in_spatial + static_cast<size_t>(ic) * hw)
                  : zero_bf;
        }
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
          const int oc = n0 + b_n0 + r;
          const bool oc_ok = (oc < co);
          const Bf16* w_oc =
              oc_ok ? (weight + static_cast<size_t>(oc) * ci * 9 + tap)
                    : nullptr;
          #pragma unroll
          for (int c2 = 0; c2 < 2; ++c2) {
            const int kk = c2 * 32 + b_k;
            const int ic = ic0 + kk;
            bs[(b_n0 + r) * kBKP + kk] =
                (oc_ok && ic < ci) ? __ldg(w_oc + static_cast<size_t>(ic) * 9)
                                   : zero_bf;
          }
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < 64; kk += 16) {
          uint32_t af[2][4], bfr[4][2];
          #pragma unroll
          for (int mt = 0; mt < 2; ++mt)
            ldmatrix_x4(af[mt],
                        as + (wm * 32 + mt * 16 + a_row) * kBKP + kk + a_col);
          #pragma unroll
          for (int nt = 0; nt < 4; ++nt)
            ldmatrix_x2(bfr[nt],
                        bs + (wn * 32 + nt * 8 + b_row) * kBKP + kk + b_col);
          #pragma unroll
          for (int mt = 0; mt < 2; ++mt)
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt)
              mma_m16n8k16(acc[mt][nt], af[mt], bfr[nt]);
        }
        __syncthreads();
      }
    }
  }

  const int r0 = m0 + wm * 32 + (lane >> 2);
  const int c0 = n0 + wn * 32 + (lane & 3) * 2;
  #pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    const int r_a = r0 + mt * 16;
    const int r_b = r_a + 8;
    const bool va = (r_a < total_m);
    const bool vb = (r_b < total_m);
    const int ob_a = va ? (r_a / oho) : 0;
    const int rem_a = va ? (r_a - ob_a * oho) : 0;
    const int ob_b = vb ? (r_b / oho) : 0;
    const int rem_b = vb ? (r_b - ob_b * oho) : 0;
    #pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int c = c0 + nt * 8;
      if (c >= co) continue;
      const bool c1_ok = (c + 1 < co);
      const float b0 = bias ? bf(__ldg(bias + c)) : 0.0f;
      const float b1 = (bias && c1_ok) ? bf(__ldg(bias + c + 1)) : 0.0f;
      if (va) {
        out[(static_cast<size_t>(ob_a) * co + c) * oho + rem_a] =
            __float2bfloat16(gelu_f32(acc[mt][nt][0] + b0));
        if (c1_ok)
          out[(static_cast<size_t>(ob_a) * co + (c + 1)) * oho + rem_a] =
              __float2bfloat16(gelu_f32(acc[mt][nt][1] + b1));
      }
      if (vb) {
        out[(static_cast<size_t>(ob_b) * co + c) * oho + rem_b] =
            __float2bfloat16(gelu_f32(acc[mt][nt][2] + b0));
        if (c1_ok)
          out[(static_cast<size_t>(ob_b) * co + (c + 1)) * oho + rem_b] =
              __float2bfloat16(gelu_f32(acc[mt][nt][3] + b1));
      }
    }
  }
}

// ---- Multi-Warp 1-Pass Online Softmax GQA (head_dim=128 & head_dim=64) -----

// Multi-warp single-pass Online Softmax for head_dim == 128 BF16.
// Each lane holds 4 contiguous BF16 elements (uint2 coalesced loads).
// Only ONE __syncthreads() in the entire kernel (at final warp merge).
template <int kWarps, bool kDevPos>
__global__ void __launch_bounds__(kWarps * 32) gqa_attention_128_bf16_kernel(
    const Bf16* __restrict__ q, const Bf16* __restrict__ k,
    const Bf16* __restrict__ v, Bf16* __restrict__ out, int nkv, int nqh,
    int nkh, bool causal, int window_len, int q_offset,
    const int* __restrict__ d_pos_ptr) {
  constexpr int kHeadDim = 128;
  const int qi = kDevPos ? 0 : blockIdx.y;
  const int qh = blockIdx.x;
  const int tid = threadIdx.x;
  const int wid = tid >> 5;
  const int lane = tid & 31;
  const int kvh = qh * nkh / nqh;
  const float scale = rsqrtf(static_cast<float>(kHeadDim));

  int lo = 0, hi = 0;
  if constexpr (kDevPos) {
    lo = 0;
    hi = (*d_pos_ptr) + 1;
  } else if (causal) {
    lo = 0;
    hi = min(nkv, q_offset + qi + 1);
  } else {
    const AsrAttnWindow win = asr_audio_attn_window_bounds(qi, nkv, window_len);
    lo = win.lo;
    hi = win.hi;
  }

  const Bf16* qrow = q + (static_cast<size_t>(qi) * nqh + qh) * kHeadDim;
  const uint2 qv = __ldg(reinterpret_cast<const uint2*>(qrow) + lane);
  const __nv_bfloat162* qb = reinterpret_cast<const __nv_bfloat162*>(&qv);
  const float2 q01 = __bfloat1622float2(qb[0]);
  const float2 q23 = __bfloat1622float2(qb[1]);

  float w_max = -INFINITY;
  float w_sum = 0.0f;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

  for (int kj = lo + wid; kj < hi; kj += kWarps) {
    const size_t kv_off = (static_cast<size_t>(kj) * nkh + kvh) * kHeadDim;
    const uint2 kv = __ldg(reinterpret_cast<const uint2*>(k + kv_off) + lane);
    const __nv_bfloat162* kb = reinterpret_cast<const __nv_bfloat162*>(&kv);
    const float2 k01 = __bfloat1622float2(kb[0]);
    const float2 k23 = __bfloat1622float2(kb[1]);
    float dot = q01.x * k01.x + q01.y * k01.y + q23.x * k23.x + q23.y * k23.y;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      dot += __shfl_xor_sync(0xffffffffu, dot, off);
    const float score = dot * scale;
    const float new_max = fmaxf(w_max, score);
    const float alpha = (w_sum > 0.0f) ? expf(w_max - new_max) : 0.0f;
    const float p = expf(score - new_max);
    w_sum = w_sum * alpha + p;
    w_max = new_max;

    const uint2 vv = __ldg(reinterpret_cast<const uint2*>(v + kv_off) + lane);
    const __nv_bfloat162* vb = reinterpret_cast<const __nv_bfloat162*>(&vv);
    const float2 v01 = __bfloat1622float2(vb[0]);
    const float2 v23 = __bfloat1622float2(vb[1]);
    a0 = a0 * alpha + p * v01.x;
    a1 = a1 * alpha + p * v01.y;
    a2 = a2 * alpha + p * v23.x;
    a3 = a3 * alpha + p * v23.y;
  }

  __shared__ float sm_max[kWarps];
  __shared__ float sm_sum[kWarps];
  __shared__ float sm_acc[kWarps][kHeadDim];
  if (lane == 0) {
    sm_max[wid] = w_max;
    sm_sum[wid] = w_sum;
  }
  sm_acc[wid][lane * 4 + 0] = a0;
  sm_acc[wid][lane * 4 + 1] = a1;
  sm_acc[wid][lane * 4 + 2] = a2;
  sm_acc[wid][lane * 4 + 3] = a3;
  __syncthreads();

  if (wid == 0) {
    float g_max = -INFINITY;
    #pragma unroll
    for (int w = 0; w < kWarps; ++w)
      if (sm_sum[w] > 0.0f) g_max = fmaxf(g_max, sm_max[w]);
    float g_sum = 0.0f;
    float scale_w[kWarps];
    #pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      if (sm_sum[w] > 0.0f) {
        scale_w[w] = expf(sm_max[w] - g_max);
        g_sum += sm_sum[w] * scale_w[w];
      } else {
        scale_w[w] = 0.0f;
      }
    }
    const float inv = (g_sum > 0.0f) ? (1.0f / g_sum) : 0.0f;
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
    #pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      r0 += sm_acc[w][lane * 4 + 0] * scale_w[w];
      r1 += sm_acc[w][lane * 4 + 1] * scale_w[w];
      r2 += sm_acc[w][lane * 4 + 2] * scale_w[w];
      r3 += sm_acc[w][lane * 4 + 3] * scale_w[w];
    }
    Bf16* orow = out + (static_cast<size_t>(qi) * nqh + qh) * kHeadDim;
    uint2 ov;
    __nv_bfloat162* ob = reinterpret_cast<__nv_bfloat162*>(&ov);
    ob[0] = __floats2bfloat162_rn(r0 * inv, r1 * inv);
    ob[1] = __floats2bfloat162_rn(r2 * inv, r3 * inv);
    reinterpret_cast<uint2*>(orow)[lane] = ov;
  }
}

// 4-warp single-pass Online Softmax for head_dim == 64 BF16 (Audio Encoder).
// Each lane holds 2 contiguous BF16 elements (__nv_bfloat162 coalesced loads).
// Only ONE __syncthreads() in the entire kernel.
__global__ void __launch_bounds__(128) gqa_attention_64_bf16_kernel(
    const Bf16* __restrict__ q, const Bf16* __restrict__ k,
    const Bf16* __restrict__ v, Bf16* __restrict__ out, int nkv, int nqh,
    int nkh, bool causal, int window_len, int q_offset) {
  constexpr int kHeadDim = 64;
  constexpr int kWarps = 4;
  const int qi = blockIdx.y;
  const int qh = blockIdx.x;
  const int tid = threadIdx.x;
  const int wid = tid >> 5;
  const int lane = tid & 31;
  const int kvh = qh * nkh / nqh;
  const float scale = rsqrtf(static_cast<float>(kHeadDim));

  int lo = 0, hi = 0;
  if (causal) {
    lo = 0;
    hi = min(nkv, q_offset + qi + 1);
  } else {
    const AsrAttnWindow win = asr_audio_attn_window_bounds(qi, nkv, window_len);
    lo = win.lo;
    hi = win.hi;
  }

  const Bf16* qrow = q + (static_cast<size_t>(qi) * nqh + qh) * kHeadDim;
  const __nv_bfloat162 qv =
      __ldg(reinterpret_cast<const __nv_bfloat162*>(qrow) + lane);
  const float2 q01 = __bfloat1622float2(qv);

  float w_max = -INFINITY;
  float w_sum = 0.0f;
  float a0 = 0.0f, a1 = 0.0f;

  for (int kj = lo + wid; kj < hi; kj += kWarps) {
    const size_t kv_off = (static_cast<size_t>(kj) * nkh + kvh) * kHeadDim;
    const __nv_bfloat162 kv =
        __ldg(reinterpret_cast<const __nv_bfloat162*>(k + kv_off) + lane);
    const float2 k01 = __bfloat1622float2(kv);
    float dot = q01.x * k01.x + q01.y * k01.y;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      dot += __shfl_xor_sync(0xffffffffu, dot, off);
    const float score = dot * scale;
    const float new_max = fmaxf(w_max, score);
    const float alpha = (w_sum > 0.0f) ? expf(w_max - new_max) : 0.0f;
    const float p = expf(score - new_max);
    w_sum = w_sum * alpha + p;
    w_max = new_max;

    const __nv_bfloat162 vv =
        __ldg(reinterpret_cast<const __nv_bfloat162*>(v + kv_off) + lane);
    const float2 v01 = __bfloat1622float2(vv);
    a0 = a0 * alpha + p * v01.x;
    a1 = a1 * alpha + p * v01.y;
  }

  __shared__ float sm_max[kWarps];
  __shared__ float sm_sum[kWarps];
  __shared__ float sm_acc[kWarps][kHeadDim];
  if (lane == 0) {
    sm_max[wid] = w_max;
    sm_sum[wid] = w_sum;
  }
  sm_acc[wid][lane * 2 + 0] = a0;
  sm_acc[wid][lane * 2 + 1] = a1;
  __syncthreads();

  if (wid == 0) {
    float g_max = -INFINITY;
    #pragma unroll
    for (int w = 0; w < kWarps; ++w)
      if (sm_sum[w] > 0.0f) g_max = fmaxf(g_max, sm_max[w]);
    float g_sum = 0.0f;
    float scale_w[kWarps];
    #pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      if (sm_sum[w] > 0.0f) {
        scale_w[w] = expf(sm_max[w] - g_max);
        g_sum += sm_sum[w] * scale_w[w];
      } else {
        scale_w[w] = 0.0f;
      }
    }
    const float inv = (g_sum > 0.0f) ? (1.0f / g_sum) : 0.0f;
    float r0 = 0.0f, r1 = 0.0f;
    #pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      r0 += sm_acc[w][lane * 2 + 0] * scale_w[w];
      r1 += sm_acc[w][lane * 2 + 1] * scale_w[w];
    }
    Bf16* orow = out + (static_cast<size_t>(qi) * nqh + qh) * kHeadDim;
    reinterpret_cast<__nv_bfloat162*>(orow)[lane] =
        __floats2bfloat162_rn(r0 * inv, r1 * inv);
  }
}

// Generic fallback GQA kernel (no device lambdas).
__device__ float gqa_block_dot(const float* qs, const Bf16* row, int head_dim,
                               float* scratch) {
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const int nwarps = blockDim.x >> 5;
  float partial = 0.0f;
  for (int i = tid; i < head_dim; i += blockDim.x)
    partial += qs[i] * bf(row[i]);
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    partial += __shfl_down_sync(0xffffffffu, partial, off);
  if (lane == 0) scratch[wid] = partial;
  __syncthreads();
  __shared__ float s_dot;
  if (tid == 0) {
    float total = 0.0f;
    for (int w = 0; w < nwarps; ++w) total += scratch[w];
    s_dot = total;
  }
  __syncthreads();
  return s_dot;
}

__global__ void gqa_fallback_kernel(const Bf16* q, const Bf16* k, const Bf16* v,
                                    Bf16* out, int nq, int nkv, int n_q_heads,
                                    int n_kv_heads, int head_dim, bool causal,
                                    int window_len, int q_offset) {
  extern __shared__ float smem[];
  float* qs = smem;
  float* acc = smem + head_dim;
  float* reduce = smem + 2 * head_dim;
  __shared__ float s_max, s_sum, s_w;
  const int qi = blockIdx.y, qh = blockIdx.x;
  if (qi >= nq) return;
  const int tid = threadIdx.x;
  const int kvh = qh * n_kv_heads / n_q_heads;
  const Bf16* qrow = q + (static_cast<size_t>(qi) * n_q_heads + qh) * head_dim;
  if (tid < head_dim) qs[tid] = bf(qrow[tid]);
  __syncthreads();

  int lo = 0, hi = 0;
  if (causal) {
    lo = 0;
    hi = min(nkv, q_offset + qi + 1);
  } else {
    const AsrAttnWindow win = asr_audio_attn_window_bounds(qi, nkv, window_len);
    lo = win.lo;
    hi = win.hi;
  }
  const float scale = rsqrtf(static_cast<float>(head_dim));

  if (tid == 0) s_max = -INFINITY;
  __syncthreads();
  for (int kj = lo; kj < hi; ++kj) {
    const Bf16* krow = k + (static_cast<size_t>(kj) * n_kv_heads + kvh) * head_dim;
    const float dot = gqa_block_dot(qs, krow, head_dim, reduce) * scale;
    if (tid == 0) s_max = fmaxf(s_max, dot);
    __syncthreads();
  }
  const float mx = s_max;

  if (tid == 0) s_sum = 0.0f;
  if (tid < head_dim) acc[tid] = 0.0f;
  __syncthreads();
  for (int kj = lo; kj < hi; ++kj) {
    const Bf16* krow = k + (static_cast<size_t>(kj) * n_kv_heads + kvh) * head_dim;
    const float weight = expf(gqa_block_dot(qs, krow, head_dim, reduce) * scale - mx);
    if (tid == 0) {
      s_w = weight;
      s_sum += weight;
    }
    __syncthreads();
    const Bf16* vrow = v + (static_cast<size_t>(kj) * n_kv_heads + kvh) * head_dim;
    if (tid < head_dim) acc[tid] += s_w * bf(vrow[tid]);
    __syncthreads();
  }
  if (tid < head_dim) {
    Bf16* orow = out + (static_cast<size_t>(qi) * n_q_heads + qh) * head_dim;
    orow[tid] = __float2bfloat16(acc[tid] / s_sum);
  }
}

// ---- Argmax -----------------------------------------------------------------

__global__ void argmax_kernel(const float* __restrict__ x, int n,
                              int* __restrict__ best_idx) {
  float local_val = -INFINITY;
  int local_idx = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = __ldg(x + i);
    if (v > local_val) {
      local_val = v;
      local_idx = i;
    }
  }
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float ov = __shfl_down_sync(0xffffffffu, local_val, off);
    const int oi = __shfl_down_sync(0xffffffffu, local_idx, off);
    if (ov > local_val) {
      local_val = ov;
      local_idx = oi;
    }
  }
  __shared__ float wv[32];
  __shared__ int wi[32];
  if (lane == 0) {
    wv[wid] = local_val;
    wi[wid] = local_idx;
  }
  __syncthreads();
  if (wid == 0) {
    const int nw = (blockDim.x + 31) / 32;
    if (lane < nw) {
      local_val = wv[lane];
      local_idx = wi[lane];
    } else {
      local_val = -INFINITY;
      local_idx = 0;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_down_sync(0xffffffffu, local_val, off);
      const int oi = __shfl_down_sync(0xffffffffu, local_idx, off);
      if (ov > local_val) {
        local_val = ov;
        local_idx = oi;
      }
    }
    if (lane == 0) *best_idx = local_idx;
  }
}

// ---- SM90 64x64x64 BF16 Tensor Core GEMM (prefill & audio encoder) ---------

constexpr int kMmaBM = 64;
constexpr int kMmaBN = 64;
constexpr int kMmaBK = 64;
constexpr int kMmaBKP = kMmaBK + 8;  // 72 Bf16 = 144B (16B aligned, bank-free)
constexpr int kMmaThreads = 128;     // 4 warps (2x2)
constexpr int kMmaStages = 2;

__device__ __forceinline__ void mma_stage_tile64(Bf16* dst, const Bf16* src,
                                                 int k0, int k, int valid_rows,
                                                 int row_base) {
  constexpr int kVec = kMmaBK / 8;  // 8 uint4s per row
  constexpr int kSteps = (64 * kVec) / kMmaThreads;  // 4 steps/thread
  #pragma unroll
  for (int i = 0; i < kSteps; ++i) {
    const int idx = threadIdx.x + i * kMmaThreads;
    const int r = idx / kVec;
    const int c = (idx % kVec) * 8;
    const int kk = k0 + c;
    const bool ok = (r < valid_rows) && (kk + 8 <= k);
    const Bf16* g =
        ok ? (src + static_cast<size_t>(row_base + r) * k + kk) : src;
    cp_async16(dst + r * kMmaBKP + c, g, ok);
  }
}

__device__ __forceinline__ void mma_stage_ab(Bf16* as, Bf16* bs, const Bf16* a,
                                             const Bf16* b, int kb, int s,
                                             int m0, int rows, int n0, int n,
                                             int k) {
  mma_stage_tile64(as + s * kMmaBM * kMmaBKP, a, kb * kMmaBK, k, rows - m0, m0);
  mma_stage_tile64(bs + s * kMmaBN * kMmaBKP, b, kb * kMmaBK, k, n - n0, n0);
}

// kEpilogue: 0 = store, 1 = add_residual (exact BF16), 2 = gelu (exact BF16)
template <int kEpilogue>
__global__ void __launch_bounds__(kMmaThreads, 2) gemm_mma_64x64_kernel(
    Bf16* __restrict__ c_out, const Bf16* __restrict__ a,
    const Bf16* __restrict__ b, const Bf16* __restrict__ bias, int rows, int k,
    int n) {
  alignas(16) __shared__ uint16_t as_raw[kMmaStages * kMmaBM * kMmaBKP];
  alignas(16) __shared__ uint16_t bs_raw[kMmaStages * kMmaBN * kMmaBKP];
  Bf16* as = reinterpret_cast<Bf16*>(as_raw);
  Bf16* bs = reinterpret_cast<Bf16*>(bs_raw);

  const AsrGemmTileCoord coord = asr_swizzle_gemm_cta(
      static_cast<int>(blockIdx.x), static_cast<int>(blockIdx.y),
      static_cast<int>(gridDim.x), static_cast<int>(gridDim.y));
  const int m0 = coord.tile_m * kMmaBM;
  const int n0 = coord.tile_n * kMmaBN;

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int wm = warp >> 1;
  const int wn = warp & 1;

  float acc[2][4][4];
  #pragma unroll
  for (int mt = 0; mt < 2; ++mt)
    #pragma unroll
    for (int nt = 0; nt < 4; ++nt)
      #pragma unroll
      for (int q = 0; q < 4; ++q) acc[mt][nt][q] = 0.0f;

  const int nblk = (k + kMmaBK - 1) / kMmaBK;
  mma_stage_ab(as, bs, a, b, 0, 0, m0, rows, n0, n, k);
  cp_commit();

  const int a_row = (lane & 7) + ((lane & 8) ? 8 : 0);
  const int a_col = (lane & 16) ? 8 : 0;
  const int b_row = lane & 7;
  const int b_col = (lane & 8) ? 8 : 0;

  for (int kb = 0; kb < nblk; ++kb) {
    const bool ahead = (kb + 1 < nblk);
    if (ahead) mma_stage_ab(as, bs, a, b, kb + 1, (kb + 1) & 1, m0, rows, n0, n, k);
    cp_commit();
    if (ahead) cp_wait<1>();
    else cp_wait<0>();
    __syncthreads();

    const Bf16* asb = as + (kb & 1) * kMmaBM * kMmaBKP;
    const Bf16* bsb = bs + (kb & 1) * kMmaBN * kMmaBKP;
    #pragma unroll
    for (int kk = 0; kk < kMmaBK; kk += 16) {
      uint32_t af[2][4], bfr[4][2];
      #pragma unroll
      for (int mt = 0; mt < 2; ++mt)
        ldmatrix_x4(af[mt],
                    asb + (wm * 32 + mt * 16 + a_row) * kMmaBKP + kk + a_col);
      #pragma unroll
      for (int nt = 0; nt < 4; ++nt)
        ldmatrix_x2(bfr[nt],
                    bsb + (wn * 32 + nt * 8 + b_row) * kMmaBKP + kk + b_col);
      #pragma unroll
      for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) mma_m16n8k16(acc[mt][nt], af[mt], bfr[nt]);
    }
    __syncthreads();
  }

  const int r0 = m0 + wm * 32 + (lane >> 2);
  const int c0 = n0 + wn * 32 + (lane & 3) * 2;
  #pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    #pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int c = c0 + nt * 8;
      if (c >= n) continue;
      float v0 = acc[mt][nt][0], v1 = acc[mt][nt][1];
      float v2 = acc[mt][nt][2], v3 = acc[mt][nt][3];
      if (bias) {
        const float b0 = bf(__ldg(bias + c));
        const float b1 = bf(__ldg(bias + c + 1));
        v0 += b0;
        v1 += b1;
        v2 += b0;
        v3 += b1;
      }
      const int r = r0 + mt * 16;
      if constexpr (kEpilogue == 1) {
        if (r < rows) {
          __nv_bfloat162* dst = reinterpret_cast<__nv_bfloat162*>(
              c_out + static_cast<size_t>(r) * n + c);
          const float2 cur = __bfloat1622float2(*dst);
          const float2 add = __bfloat1622float2(__floats2bfloat162_rn(v0, v1));
          *dst = __floats2bfloat162_rn(cur.x + add.x, cur.y + add.y);
        }
        if (r + 8 < rows) {
          __nv_bfloat162* dst = reinterpret_cast<__nv_bfloat162*>(
              c_out + static_cast<size_t>(r + 8) * n + c);
          const float2 cur = __bfloat1622float2(*dst);
          const float2 add = __bfloat1622float2(__floats2bfloat162_rn(v2, v3));
          *dst = __floats2bfloat162_rn(cur.x + add.x, cur.y + add.y);
        }
      } else if constexpr (kEpilogue == 2) {
        if (r < rows) {
          const float2 p = __bfloat1622float2(__floats2bfloat162_rn(v0, v1));
          *reinterpret_cast<__nv_bfloat162*>(
              c_out + static_cast<size_t>(r) * n + c) =
              __floats2bfloat162_rn(gelu_f32(p.x), gelu_f32(p.y));
        }
        if (r + 8 < rows) {
          const float2 p = __bfloat1622float2(__floats2bfloat162_rn(v2, v3));
          *reinterpret_cast<__nv_bfloat162*>(
              c_out + static_cast<size_t>(r + 8) * n + c) =
              __floats2bfloat162_rn(gelu_f32(p.x), gelu_f32(p.y));
        }
      } else {
        if (r < rows)
          *reinterpret_cast<__nv_bfloat162*>(
              c_out + static_cast<size_t>(r) * n + c) =
              __floats2bfloat162_rn(v0, v1);
        if (r + 8 < rows)
          *reinterpret_cast<__nv_bfloat162*>(
              c_out + static_cast<size_t>(r + 8) * n + c) =
              __floats2bfloat162_rn(v2, v3);
      }
    }
  }
}

// ---- SM90 Vectorized 1-Warp/Row BF16 GEMV / Small-M / QKV / SwiGLU ---------

template <int kM, int kUnroll>
__device__ __forceinline__ void gemv_warp_dot_bf16(const Bf16* __restrict__ wr,
                                                   const Bf16* __restrict__ x,
                                                   int k, int lane,
                                                   float (&acc)[kM]) {
  constexpr int kStr = 32 * 8;
  #pragma unroll
  for (int b = 0; b < kM; ++b) acc[b] = 0.0f;

  int i = lane * 8;
  const int group_end = k - ((kUnroll - 1) * kStr + 8);
  for (; i <= group_end; i += kUnroll * kStr) {
    uint4 wv[kUnroll];
    #pragma unroll
    for (int u = 0; u < kUnroll; ++u) wv[u] = ld_cs_u4(wr + i + u * kStr);
    #pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      uint4 xv[kM];
      #pragma unroll
      for (int b = 0; b < kM; ++b)
        xv[b] = ld_ro_u4(x + static_cast<size_t>(b) * k + i + u * kStr);
      const __nv_bfloat162* vb = reinterpret_cast<const __nv_bfloat162*>(&wv[u]);
      float fw[8];
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 fv = __bfloat1622float2(vb[e]);
        fw[2 * e] = fv.x;
        fw[2 * e + 1] = fv.y;
      }
      #pragma unroll
      for (int b = 0; b < kM; ++b) {
        const __nv_bfloat162* ub = reinterpret_cast<const __nv_bfloat162*>(&xv[b]);
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 fu = __bfloat1622float2(ub[e]);
          acc[b] += fw[2 * e] * fu.x + fw[2 * e + 1] * fu.y;
        }
      }
    }
  }
  for (; i + 8 <= k; i += kStr) {
    const uint4 v = ld_cs_u4(wr + i);
    const __nv_bfloat162* vb = reinterpret_cast<const __nv_bfloat162*>(&v);
    #pragma unroll
    for (int b = 0; b < kM; ++b) {
      const uint4 xu = ld_ro_u4(x + static_cast<size_t>(b) * k + i);
      const __nv_bfloat162* ub = reinterpret_cast<const __nv_bfloat162*>(&xu);
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 fv = __bfloat1622float2(vb[e]);
        const float2 fu = __bfloat1622float2(ub[e]);
        acc[b] += fv.x * fu.x + fv.y * fu.y;
      }
    }
  }
  for (; i < k; ++i) {
    const float wf = bf(wr[i]);
    #pragma unroll
    for (int b = 0; b < kM; ++b)
      acc[b] += wf * bf(x[static_cast<size_t>(b) * k + i]);
  }
  #pragma unroll
  for (int b = 0; b < kM; ++b) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc[b] += __shfl_down_sync(0xffffffffu, acc[b], off);
  }
}

// kEpilogue: 0 = store, 1 = add_residual (exact BF16), 2 = gelu (exact BF16)
template <int kM, int kUnroll, typename OutT, int kEpilogue>
__global__ void __launch_bounds__(128) gemv_bf16_vec_kernel(
    const Bf16* __restrict__ w, const Bf16* __restrict__ x,
    const Bf16* __restrict__ bias, OutT* __restrict__ out, int n_out, int k) {
  const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (row >= n_out) return;
  const int m0 = blockIdx.y * kM;
  const int lane = threadIdx.x & 31;
  const Bf16* wr = w + static_cast<size_t>(row) * k;
  const Bf16* xb = x + static_cast<size_t>(m0) * k;

  float acc[kM];
  gemv_warp_dot_bf16<kM, kUnroll>(wr, xb, k, lane, acc);

  if (lane == 0) {
    const float bv = bias ? bf(__ldg(bias + row)) : 0.0f;
    #pragma unroll
    for (int b = 0; b < kM; ++b) {
      const size_t o = static_cast<size_t>(m0 + b) * n_out + row;
      const float val = acc[b] + bv;
      if constexpr (sizeof(OutT) == 4) {
        out[o] = val;
      } else if constexpr (kEpilogue == 1) {
        const float cur = bf(out[o]);
        const float prod = bf(__float2bfloat16(val));
        out[o] = __float2bfloat16(cur + prod);
      } else if constexpr (kEpilogue == 2) {
        const float prod = bf(__float2bfloat16(val));
        out[o] = __float2bfloat16(gelu_f32(prod));
      } else {
        out[o] = __float2bfloat16(val);
      }
    }
  }
}

// Fused 3-way Q/K/V BF16 projection sharing input x (optional per-proj bias).
template <int kM, int kUnroll>
__global__ void __launch_bounds__(128) gemv_qkv_bf16_kernel(
    QkvGemvDesc desc, const Bf16* __restrict__ x, int k) {
  const int global_row = blockIdx.x * 4 + (threadIdx.x >> 5);
  const AsrMultiGemvRowLoc loc =
      asr_multi_gemv_locate_row(global_row, desc.n_out, 3);
  if (loc.proj < 0) return;

  const int m0 = blockIdx.y * kM;
  const int lane = threadIdx.x & 31;
  const Bf16* wr = desc.w[loc.proj] + static_cast<size_t>(loc.local_row) * k;
  const Bf16* xb = x + static_cast<size_t>(m0) * k;
  Bf16* out_base = desc.out[loc.proj];
  const int out_stride = desc.n_out[loc.proj];

  float acc[kM];
  gemv_warp_dot_bf16<kM, kUnroll>(wr, xb, k, lane, acc);

  if (lane == 0) {
    const Bf16* bptr = desc.bias[loc.proj];
    const float bv = bptr ? bf(__ldg(bptr + loc.local_row)) : 0.0f;
    #pragma unroll
    for (int b = 0; b < kM; ++b) {
      out_base[static_cast<size_t>(m0 + b) * out_stride + loc.local_row] =
          __float2bfloat16(acc[b] + bv);
    }
  }
}

// Fused dual-output SwiGLU BF16 GEMV: computes gate_proj and up_proj sharing x,
// then applies exact BF16-rounded silu(gate) * up.
template <int kM, int kUnroll>
__global__ void __launch_bounds__(128) gemv_swiglu_bf16_kernel(
    const Bf16* __restrict__ w_gate, const Bf16* __restrict__ w_up,
    const Bf16* __restrict__ x, Bf16* __restrict__ out, int n_out, int k) {
  constexpr int kStr = 32 * 8;
  const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (row >= n_out) return;
  const int m0 = blockIdx.y * kM;
  const int lane = threadIdx.x & 31;
  const Bf16* wgr = w_gate + static_cast<size_t>(row) * k;
  const Bf16* wur = w_up + static_cast<size_t>(row) * k;
  const Bf16* xb = x + static_cast<size_t>(m0) * k;

  float acc_g[kM];
  float acc_u[kM];
  #pragma unroll
  for (int b = 0; b < kM; ++b) {
    acc_g[b] = 0.0f;
    acc_u[b] = 0.0f;
  }

  int i = lane * 8;
  const int group_end = k - ((kUnroll - 1) * kStr + 8);
  for (; i <= group_end; i += kUnroll * kStr) {
    uint4 wgv[kUnroll];
    uint4 wuv[kUnroll];
    #pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      wgv[u] = ld_cs_u4(wgr + i + u * kStr);
      wuv[u] = ld_cs_u4(wur + i + u * kStr);
    }
    #pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      uint4 xv[kM];
      #pragma unroll
      for (int b = 0; b < kM; ++b)
        xv[b] = ld_ro_u4(xb + static_cast<size_t>(b) * k + i + u * kStr);
      const __nv_bfloat162* vgb =
          reinterpret_cast<const __nv_bfloat162*>(&wgv[u]);
      const __nv_bfloat162* vub =
          reinterpret_cast<const __nv_bfloat162*>(&wuv[u]);
      float fg[8], fu_w[8];
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 f2g = __bfloat1622float2(vgb[e]);
        const float2 f2u = __bfloat1622float2(vub[e]);
        fg[2 * e] = f2g.x;
        fg[2 * e + 1] = f2g.y;
        fu_w[2 * e] = f2u.x;
        fu_w[2 * e + 1] = f2u.y;
      }
      #pragma unroll
      for (int b = 0; b < kM; ++b) {
        const __nv_bfloat162* xbb =
            reinterpret_cast<const __nv_bfloat162*>(&xv[b]);
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 fx = __bfloat1622float2(xbb[e]);
          acc_g[b] += fg[2 * e] * fx.x + fg[2 * e + 1] * fx.y;
          acc_u[b] += fu_w[2 * e] * fx.x + fu_w[2 * e + 1] * fx.y;
        }
      }
    }
  }
  for (; i + 8 <= k; i += kStr) {
    const uint4 wg = ld_cs_u4(wgr + i);
    const uint4 wu = ld_cs_u4(wur + i);
    const __nv_bfloat162* vgb = reinterpret_cast<const __nv_bfloat162*>(&wg);
    const __nv_bfloat162* vub = reinterpret_cast<const __nv_bfloat162*>(&wu);
    #pragma unroll
    for (int b = 0; b < kM; ++b) {
      const uint4 xu = ld_ro_u4(xb + static_cast<size_t>(b) * k + i);
      const __nv_bfloat162* xbb = reinterpret_cast<const __nv_bfloat162*>(&xu);
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 f2g = __bfloat1622float2(vgb[e]);
        const float2 f2u = __bfloat1622float2(vub[e]);
        const float2 fx = __bfloat1622float2(xbb[e]);
        acc_g[b] += f2g.x * fx.x + f2g.y * fx.y;
        acc_u[b] += f2u.x * fx.x + f2u.y * fx.y;
      }
    }
  }
  for (; i < k; ++i) {
    const float wgf = bf(wgr[i]);
    const float wuf = bf(wur[i]);
    #pragma unroll
    for (int b = 0; b < kM; ++b) {
      const float xf = bf(xb[static_cast<size_t>(b) * k + i]);
      acc_g[b] += wgf * xf;
      acc_u[b] += wuf * xf;
    }
  }
  #pragma unroll
  for (int b = 0; b < kM; ++b) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      acc_g[b] += __shfl_down_sync(0xffffffffu, acc_g[b], off);
      acc_u[b] += __shfl_down_sync(0xffffffffu, acc_u[b], off);
    }
  }
  if (lane == 0) {
    #pragma unroll
    for (int b = 0; b < kM; ++b) {
      const float g = bf(__float2bfloat16(acc_g[b]));
      const float u = bf(__float2bfloat16(acc_u[b]));
      out[static_cast<size_t>(m0 + b) * n_out + row] =
          __float2bfloat16(u * g / (1.0f + expf(-g)));
    }
  }
}

// Scalar fallbacks for unaligned inputs.
__global__ void gemv_scalar_f32_kernel(const Bf16* w, const Bf16* x, float* out,
                                       int n_out, int k) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  if (row >= n_out) return;
  const Bf16* wr = w + static_cast<size_t>(row) * k;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < k; i += blockDim.x) acc += bf(wr[i]) * bf(x[i]);
  const float total = block_reduce_sum(acc, scratch);
  if (threadIdx.x == 0) out[row] = total;
}

template <int kEpilogue>
__global__ void gemm_scalar_bf16_kernel(Bf16* y, const Bf16* x, const Bf16* w,
                                        const Bf16* bias, int n_tokens, int k,
                                        int m_out) {
  __shared__ float scratch[32];
  const int col = blockIdx.x;
  const int row = blockIdx.y;
  if (col >= m_out || row >= n_tokens) return;
  const Bf16* wr = w + static_cast<size_t>(col) * k;
  const Bf16* xr = x + static_cast<size_t>(row) * k;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < k; i += blockDim.x)
    acc += bf(wr[i]) * bf(xr[i]);
  acc = block_reduce_sum(acc, scratch);
  if (threadIdx.x == 0) {
    if (bias) acc += bf(bias[col]);
    const size_t o = static_cast<size_t>(row) * m_out + col;
    if constexpr (kEpilogue == 1) {
      y[o] = __float2bfloat16(bf(y[o]) + bf(__float2bfloat16(acc)));
    } else if constexpr (kEpilogue == 2) {
      y[o] = __float2bfloat16(gelu_f32(bf(__float2bfloat16(acc))));
    } else {
      y[o] = __float2bfloat16(acc);
    }
  }
}

__global__ void gemv_swiglu_scalar_kernel(const Bf16* w_gate, const Bf16* w_up,
                                          const Bf16* x, Bf16* out, int m,
                                          int n_out, int k) {
  __shared__ float scratch_g[32];
  __shared__ float scratch_u[32];
  const int col = blockIdx.x;
  const int row = blockIdx.y;
  if (col >= n_out || row >= m) return;
  const Bf16* wgr = w_gate + static_cast<size_t>(col) * k;
  const Bf16* wur = w_up + static_cast<size_t>(col) * k;
  const Bf16* xr = x + static_cast<size_t>(row) * k;
  float ag = 0.0f, au = 0.0f;
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    const float xv = bf(xr[i]);
    ag += bf(wgr[i]) * xv;
    au += bf(wur[i]) * xv;
  }
  const float g_tot = block_reduce_sum(ag, scratch_g);
  const float u_tot = block_reduce_sum(au, scratch_u);
  if (threadIdx.x == 0) {
    const float g = bf(__float2bfloat16(g_tot));
    const float u = bf(__float2bfloat16(u_tot));
    out[static_cast<size_t>(row) * n_out + col] =
        __float2bfloat16(u * g / (1.0f + expf(-g)));
  }
}

// ---- Audio Frontend / Permute / Sinusoidal ----------------------------------

constexpr int kConvTokensPerChunk = 13;
constexpr int kConvPosPerToken = 16;

__global__ void conv_permute_kernel(const Bf16* __restrict__ in,
                                    Bf16* __restrict__ out, int row,
                                    int total) {
  // out [c,13,down*16], in [c,down,16,13].
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int flat = idx % row;
  const int t = (idx / row) % kConvTokensPerChunk;
  const int c = idx / (kConvTokensPerChunk * row);
  const int down = row / kConvPosPerToken;
  const int ch = flat / kConvPosPerToken, fr = flat % kConvPosPerToken;
  const size_t src =
      ((static_cast<size_t>(c) * down + ch) * kConvPosPerToken + fr) *
          kConvTokensPerChunk +
      t;
  out[idx] = __ldg(in + src);
}

__global__ void mel_chunks_kernel(const float* __restrict__ mel,
                                  Bf16* __restrict__ out, int chunks,
                                  int valid_frames) {
  // out [chunks, 1, 128, 100], mel [128, 3000].
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = chunks * 128 * 100;
  if (idx >= total) return;
  const int t = idx % 100;
  const int m = (idx / 100) % 128;
  const int c = idx / (128 * 100);
  const int frame = c * 100 + t;
  out[idx] = frame < valid_frames
                 ? __float2bfloat16(__ldg(mel + static_cast<size_t>(m) * 3000 + frame))
                 : __float2bfloat16(0.0f);
}

__global__ void sinusoidal_kernel(Bf16* __restrict__ hidden,
                                  const float* __restrict__ pe, int rows,
                                  int dim) {
  const int c = blockIdx.z, r = blockIdx.y;
  Bf16* hrow = hidden + (static_cast<size_t>(c) * rows + r) * dim;
  const float* prow = pe + static_cast<size_t>(r) * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    hrow[i] = __float2bfloat16(bf(hrow[i]) + __ldg(prow + i));
}

// ---- Dispatch Helpers -------------------------------------------------------

template <int kM, typename OutT, int kEpilogue>
void launch_bf16_gemv_exact(const Bf16* w, const Bf16* x, const Bf16* bias,
                            OutT* out, int n_groups, int n_out, int k,
                            cudaStream_t st) {
  dim3 grid((n_out + 3) / 4, n_groups);
  if constexpr (asr_gemv_unroll_for(kM) == 4) {
    gemv_bf16_vec_kernel<kM, 4, OutT, kEpilogue>
        <<<grid, 128, 0, st>>>(w, x, bias, out, n_out, k);
  } else {
    gemv_bf16_vec_kernel<kM, 1, OutT, kEpilogue>
        <<<grid, 128, 0, st>>>(w, x, bias, out, n_out, k);
  }
}

template <typename OutT, int kEpilogue>
void launch_bf16_gemv_tail(const Bf16* w, const Bf16* x, const Bf16* bias,
                           OutT* out, int rem, int n_out, int k,
                           cudaStream_t st) {
  switch (rem) {
    case 1: launch_bf16_gemv_exact<1, OutT, kEpilogue>(w, x, bias, out, 1, n_out, k, st); break;
    case 2: launch_bf16_gemv_exact<2, OutT, kEpilogue>(w, x, bias, out, 1, n_out, k, st); break;
    case 3: launch_bf16_gemv_exact<3, OutT, kEpilogue>(w, x, bias, out, 1, n_out, k, st); break;
    case 4: launch_bf16_gemv_exact<4, OutT, kEpilogue>(w, x, bias, out, 1, n_out, k, st); break;
    case 5: launch_bf16_gemv_exact<5, OutT, kEpilogue>(w, x, bias, out, 1, n_out, k, st); break;
    case 6: launch_bf16_gemv_exact<6, OutT, kEpilogue>(w, x, bias, out, 1, n_out, k, st); break;
    case 7: launch_bf16_gemv_exact<7, OutT, kEpilogue>(w, x, bias, out, 1, n_out, k, st); break;
    default: break;
  }
}

template <int kEpilogue>
void gemm_bf16_dispatch(Bf16* y, const Bf16* x, const Bf16* w, const Bf16* bias,
                        int m, int k, int n, cudaStream_t st) {
  if (m <= 0 || k <= 0 || n <= 0) return;
  if (asr_use_mma_gemm(m, n, k) && aligned16(x) && aligned16(w) && aligned4(y)) {
    dim3 grid((n + kMmaBN - 1) / kMmaBN, (m + kMmaBM - 1) / kMmaBM);
    gemm_mma_64x64_kernel<kEpilogue><<<grid, kMmaThreads, 0, st>>>(y, x, w, bias, m, k, n);
    return;
  }
  if ((k & 7) == 0 && aligned16(w) && aligned16(x)) {
    const int full = m / 8;
    const int rem = m % 8;
    if (full > 0) {
      launch_bf16_gemv_exact<8, Bf16, kEpilogue>(w, x, bias, y, full, n, k, st);
    }
    if (rem > 0) {
      const size_t off_rows = static_cast<size_t>(full) * 8;
      launch_bf16_gemv_tail<Bf16, kEpilogue>(
          w, x + off_rows * k, bias, y + off_rows * n, rem, n, k, st);
    }
    return;
  }
  dim3 grid(n, m);
  gemm_scalar_bf16_kernel<kEpilogue><<<grid, kBlock, 0, st>>>(y, x, w, bias, m, k, n);
}

template <int kM>
void launch_qkv_exact(const QkvGemvDesc& desc, const Bf16* x, int n_groups,
                      int total_n, int k, cudaStream_t st) {
  dim3 grid((total_n + 3) / 4, n_groups);
  if constexpr (asr_gemv_unroll_for(kM) == 4) {
    gemv_qkv_bf16_kernel<kM, 4><<<grid, 128, 0, st>>>(desc, x, k);
  } else {
    gemv_qkv_bf16_kernel<kM, 1><<<grid, 128, 0, st>>>(desc, x, k);
  }
}

void launch_qkv_tail(const QkvGemvDesc& desc, const Bf16* x, int rem,
                     int total_n, int k, cudaStream_t st) {
  switch (rem) {
    case 1: launch_qkv_exact<1>(desc, x, 1, total_n, k, st); break;
    case 2: launch_qkv_exact<2>(desc, x, 1, total_n, k, st); break;
    case 3: launch_qkv_exact<3>(desc, x, 1, total_n, k, st); break;
    case 4: launch_qkv_exact<4>(desc, x, 1, total_n, k, st); break;
    case 5: launch_qkv_exact<5>(desc, x, 1, total_n, k, st); break;
    case 6: launch_qkv_exact<6>(desc, x, 1, total_n, k, st); break;
    case 7: launch_qkv_exact<7>(desc, x, 1, total_n, k, st); break;
    default: break;
  }
}

template <int kM>
void launch_swiglu_exact(const Bf16* w_gate, const Bf16* w_up, const Bf16* x,
                         Bf16* out, int n_groups, int n_out, int k,
                         cudaStream_t st) {
  dim3 grid((n_out + 3) / 4, n_groups);
  if constexpr (asr_swiglu_unroll_for(kM) == 2) {
    gemv_swiglu_bf16_kernel<kM, 2>
        <<<grid, 128, 0, st>>>(w_gate, w_up, x, out, n_out, k);
  } else {
    gemv_swiglu_bf16_kernel<kM, 1>
        <<<grid, 128, 0, st>>>(w_gate, w_up, x, out, n_out, k);
  }
}

void launch_swiglu_tail(const Bf16* w_gate, const Bf16* w_up, const Bf16* x,
                        Bf16* out, int rem, int n_out, int k, cudaStream_t st) {
  switch (rem) {
    case 1: launch_swiglu_exact<1>(w_gate, w_up, x, out, 1, n_out, k, st); break;
    case 2: launch_swiglu_exact<2>(w_gate, w_up, x, out, 1, n_out, k, st); break;
    case 3: launch_swiglu_exact<3>(w_gate, w_up, x, out, 1, n_out, k, st); break;
    case 4: launch_swiglu_exact<4>(w_gate, w_up, x, out, 1, n_out, k, st); break;
    case 5: launch_swiglu_exact<5>(w_gate, w_up, x, out, 1, n_out, k, st); break;
    case 6: launch_swiglu_exact<6>(w_gate, w_up, x, out, 1, n_out, k, st); break;
    case 7: launch_swiglu_exact<7>(w_gate, w_up, x, out, 1, n_out, k, st); break;
    default: break;
  }
}

}  // namespace

// ---- Public API -------------------------------------------------------------

void rms_norm(const Bf16* x, const Bf16* weight, Bf16* out, int rows, int dim,
              float eps, cudaStream_t stream) {
  if (rows <= 0 || dim <= 0) return;
  if ((dim & 7) == 0 && aligned16(x) && aligned16(weight) && aligned16(out)) {
    if (dim <= 256) {
      rms_norm_warp_vec_bf16_kernel<<<rows, 32, 0, stream>>>(x, weight, out,
                                                             dim, eps);
    } else {
      rms_norm_vec_bf16_kernel<<<rows, 256, 0, stream>>>(x, weight, out, dim,
                                                         eps);
    }
    return;
  }
  rms_norm_scalar_kernel<<<rows, kBlock, 0, stream>>>(x, weight, out, dim, eps);
}

void layer_norm(const Bf16* x, const Bf16* weight, const Bf16* bias, Bf16* out,
                int rows, int dim, float eps, cudaStream_t stream) {
  if (rows <= 0 || dim <= 0) return;
  if ((dim & 7) == 0 && aligned16(x) && aligned16(weight) && aligned16(bias) &&
      aligned16(out)) {
    layer_norm_vec_bf16_kernel<<<rows, 256, 0, stream>>>(x, weight, bias, out,
                                                         dim, eps);
    return;
  }
  layer_norm_scalar_kernel<<<rows, kBlock, 0, stream>>>(x, weight, bias, out,
                                                        dim, eps);
}

void add_bias(Bf16* y, const Bf16* bias, int rows, int dim, cudaStream_t stream) {
  if (rows <= 0 || dim <= 0) return;
  bias_kernel<<<rows, kBlock, 0, stream>>>(y, bias, dim);
}

void add_residual(Bf16* y, const Bf16* residual, int n, cudaStream_t stream) {
  if (n <= 0) return;
  if ((n & 7) == 0 && aligned16(y) && aligned16(residual)) {
    const int vecs = n >> 3;
    residual_vec_bf16_kernel<<<(vecs + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
        y, residual, vecs);
    return;
  }
  residual_scalar_kernel<<<(n + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
      y, residual, n);
}

void gelu_inplace(Bf16* x, int n, cudaStream_t stream) {
  if (n <= 0) return;
  if ((n & 7) == 0 && aligned16(x)) {
    const int vecs = n >> 3;
    gelu_vec_bf16_kernel<<<(vecs + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
        x, vecs);
    return;
  }
  gelu_scalar_kernel<<<(n + kBlock - 1) / kBlock, kBlock, 0, stream>>>(x, n);
}

void silu_mul(Bf16* out, const Bf16* gate, const Bf16* up, int n,
              cudaStream_t stream) {
  if (n <= 0) return;
  if ((n & 7) == 0 && aligned16(out) && aligned16(gate) && aligned16(up)) {
    const int vecs = n >> 3;
    silu_mul_vec_bf16_kernel<<<(vecs + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
        out, gate, up, vecs);
    return;
  }
  silu_mul_scalar_kernel<<<(n + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
      out, gate, up, n);
}

void f32_to_bf16(const float* src, Bf16* dst, int n, cudaStream_t stream) {
  if (n <= 0) return;
  f32_to_bf16_kernel<<<(n + kBlock - 1) / kBlock, kBlock, 0, stream>>>(src, dst,
                                                                       n);
}

void bf16_to_f32(const Bf16* src, float* dst, int n, cudaStream_t stream) {
  if (n <= 0) return;
  bf16_to_f32_kernel<<<(n + kBlock - 1) / kBlock, kBlock, 0, stream>>>(src, dst,
                                                                       n);
}

void index_embeddings(const Bf16* table, const int* ids, Bf16* out, int m,
                      int dim, cudaStream_t stream) {
  if (m <= 0 || dim <= 0) return;
  dim3 grid(1, m);
  if ((dim & 7) == 0 && aligned16(table) && aligned16(out)) {
    embed_vec_kernel<<<grid, kBlock, 0, stream>>>(table, ids, out, m, dim);
    return;
  }
  embed_scalar_kernel<<<grid, kBlock, 0, stream>>>(table, ids, out, m, dim);
}

void replace_rows(Bf16* dst, const Bf16* src, int start, int n, int dim,
                  cudaStream_t stream) {
  if (n <= 0 || dim <= 0) return;
  dim3 grid(1, n);
  if ((dim & 7) == 0 && aligned16(dst) && aligned16(src)) {
    replace_rows_vec_kernel<<<grid, kBlock, 0, stream>>>(dst, src, start, n,
                                                         dim);
    return;
  }
  replace_rows_scalar_kernel<<<grid, kBlock, 0, stream>>>(dst, src, start, n,
                                                          dim);
}

void rope_half(Bf16* x, const int* positions, const float* inv_freq, int n,
               int num_heads, int head_dim, cudaStream_t stream) {
  if (n <= 0) return;
  dim3 grid(1, n);
  dim3 block(64, 4);
  rope_kernel<<<grid, block, 0, stream>>>(x, positions, inv_freq, n, num_heads,
                                          head_dim);
}

void qk_norm_rope_kv_append_bf16(Bf16* q, const Bf16* k_src, const Bf16* v_src,
                                 const Bf16* q_norm_w, const Bf16* k_norm_w,
                                 const float* inv_freq, Bf16* k_cache,
                                 Bf16* v_cache, int m, int n_q_heads,
                                 int n_kv_heads, int head_dim, int pos_base,
                                 int history_base, float eps,
                                 cudaStream_t stream) {
  if (m <= 0) return;
  if (head_dim != 128)
    throw std::runtime_error(
        "qk_norm_rope_kv_append_bf16 expects head_dim==128");
  const bool copy_v = (v_src != nullptr && v_cache != nullptr);
  if (copy_v) {
    dim3 grid(n_q_heads + 2 * n_kv_heads, m);
    qk_norm_rope_kv_append_128_kernel<false, true><<<grid, 32, 0, stream>>>(
        q, k_src, v_src, q_norm_w, k_norm_w, inv_freq, k_cache, v_cache,
        n_q_heads, n_kv_heads, pos_base, history_base, nullptr, eps);
  } else {
    dim3 grid(n_q_heads + n_kv_heads, m);
    qk_norm_rope_kv_append_128_kernel<false, false><<<grid, 32, 0, stream>>>(
        q, k_src, nullptr, q_norm_w, k_norm_w, inv_freq, k_cache, nullptr,
        n_q_heads, n_kv_heads, pos_base, history_base, nullptr, eps);
  }
}

void qk_norm_rope_kv_append_devpos_bf16(Bf16* q, const Bf16* k_src,
                                        const Bf16* v_src, const Bf16* q_norm_w,
                                        const Bf16* k_norm_w,
                                        const float* inv_freq, Bf16* k_cache,
                                        Bf16* v_cache, int n_q_heads,
                                        int n_kv_heads, int head_dim,
                                        const int* d_pos_ptr, float eps,
                                        cudaStream_t stream) {
  if (head_dim != 128)
    throw std::runtime_error(
        "qk_norm_rope_kv_append_devpos_bf16 expects head_dim==128");
  dim3 grid(n_q_heads + 2 * n_kv_heads, 1);
  qk_norm_rope_kv_append_128_kernel<true, true><<<grid, 32, 0, stream>>>(
      q, k_src, v_src, q_norm_w, k_norm_w, inv_freq, k_cache, v_cache,
      n_q_heads, n_kv_heads, 0, 0, d_pos_ptr, eps);
}

void conv3x3s2_gelu(const Bf16* in, const Bf16* weight, const Bf16* bias,
                    Bf16* out, int b, int ci, int co, int h, int w,
                    cudaStream_t stream) {
  if (b <= 0 || ci <= 0 || co <= 0 || h <= 0 || w <= 0) return;
  const int oh = (h + 1) / 2, ow = (w + 1) / 2;  // padding 1, stride 2
  if (asr_use_mma_conv(ci, co)) {
    const int total_m = b * oh * ow;
    dim3 grid((co + 63) / 64, (total_m + 63) / 64);
    conv3x3s2_gelu_mma_kernel<<<grid, 128, 0, stream>>>(
        in, weight, bias, out, b, ci, co, h, w, oh, ow);
    return;
  }
  const int total = b * co * oh * ow;
  conv3x3s2_direct_kernel<<<(total + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
      in, weight, bias, out, b, ci, co, h, w, oh, ow);
}

void conv_permute(const Bf16* in, Bf16* out, int chunks, int down,
                  cudaStream_t stream) {
  const int row = down * kConvPosPerToken;
  const int total = chunks * kConvTokensPerChunk * row;
  if (total <= 0) return;
  conv_permute_kernel<<<(total + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
      in, out, row, total);
}

void mel_to_chunks(const float* mel, Bf16* out, int chunks, int valid_frames,
                   cudaStream_t stream) {
  const int total = chunks * 128 * 100;
  if (total <= 0) return;
  mel_chunks_kernel<<<(total + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
      mel, out, chunks, valid_frames);
}

void add_sinusoidal(Bf16* hidden, const float* pe, int chunks, int rows,
                    int dim, cudaStream_t stream) {
  if (chunks <= 0 || rows <= 0 || dim <= 0) return;
  dim3 grid(1, rows, chunks);
  sinusoidal_kernel<<<grid, kBlock, 0, stream>>>(hidden, pe, rows, dim);
}

void gqa_attention(const Bf16* q, const Bf16* k, const Bf16* v, Bf16* out,
                   int nq, int nkv, int n_q_heads, int n_kv_heads, int head_dim,
                   bool causal, int window_len, int q_offset,
                   cudaStream_t stream) {
  if (nq <= 0 || nkv <= 0) return;
  dim3 grid(n_q_heads, nq);
  if (head_dim == 128 && aligned8(q) && aligned8(k) && aligned8(v) &&
      aligned8(out)) {
    if (nkv <= 32) {
      gqa_attention_128_bf16_kernel<4, false><<<grid, 128, 0, stream>>>(
          q, k, v, out, nkv, n_q_heads, n_kv_heads, causal, window_len,
          q_offset, nullptr);
    } else {
      gqa_attention_128_bf16_kernel<8, false><<<grid, 256, 0, stream>>>(
          q, k, v, out, nkv, n_q_heads, n_kv_heads, causal, window_len,
          q_offset, nullptr);
    }
    return;
  }
  if (head_dim == 64 && aligned4(q) && aligned4(k) && aligned4(v) &&
      aligned4(out)) {
    gqa_attention_64_bf16_kernel<<<grid, 128, 0, stream>>>(
        q, k, v, out, nkv, n_q_heads, n_kv_heads, causal, window_len, q_offset);
    return;
  }
  const size_t smem = (2 * head_dim + 32) * sizeof(float);
  gqa_fallback_kernel<<<grid, head_dim, smem, stream>>>(
      q, k, v, out, nq, nkv, n_q_heads, n_kv_heads, head_dim, causal,
      window_len, q_offset);
}

void gqa_attention_devpos_bf16(const Bf16* q, const Bf16* k, const Bf16* v,
                               Bf16* out, int n_q_heads, int n_kv_heads,
                               int head_dim, const int* d_pos_ptr,
                               cudaStream_t stream) {
  if (head_dim != 128)
    throw std::runtime_error(
        "gqa_attention_devpos_bf16 expects head_dim==128");
  dim3 grid(n_q_heads, 1);
  gqa_attention_128_bf16_kernel<8, true><<<grid, 256, 0, stream>>>(
      q, k, v, out, 0, n_q_heads, n_kv_heads, true, 0, 0, d_pos_ptr);
}

int argmax_f32(const float* logits, int n, int* scratch_idx,
               cudaStream_t stream) {
  argmax_f32_dev(logits, n, scratch_idx, stream);
  int idx = 0;
  cudaMemcpyAsync(&idx, scratch_idx, sizeof(int), cudaMemcpyDeviceToHost,
                  stream);
  cudaStreamSynchronize(stream);
  return idx;
}

void argmax_f32_dev(const float* logits, int n, int* out, cudaStream_t stream) {
  if (n <= 0) return;
  const int threads = n < 1024 ? (((n + 31) / 32) * 32) : 1024;
  argmax_kernel<<<1, threads, 0, stream>>>(logits, n, out);
}

void gemv_bf16(const Bf16* w, const Bf16* x, float* out, int n_out, int k,
               cudaStream_t stream) {
  if (n_out <= 0 || k <= 0) return;
  if ((k & 7) == 0 && aligned16(w) && aligned16(x)) {
    launch_bf16_gemv_exact<1, float, 0>(w, x, nullptr, out, 1, n_out, k,
                                        stream);
    return;
  }
  gemv_scalar_f32_kernel<<<n_out, kBlock, 0, stream>>>(w, x, out, n_out, k);
}

void gemm_bf16(Bf16* y, const Bf16* x, const Bf16* w, const Bf16* bias, int m,
               int k, int n, cudaStream_t stream) {
  gemm_bf16_dispatch<0>(y, x, w, bias, m, k, n, stream);
}

void gemm_bf16_add_residual(Bf16* y, const Bf16* x, const Bf16* w,
                            const Bf16* bias, int m, int k, int n,
                            cudaStream_t stream) {
  gemm_bf16_dispatch<1>(y, x, w, bias, m, k, n, stream);
}

void gemm_bf16_gelu(Bf16* y, const Bf16* x, const Bf16* w, const Bf16* bias,
                    int m, int k, int n, cudaStream_t stream) {
  gemm_bf16_dispatch<2>(y, x, w, bias, m, k, n, stream);
}

void gemv_qkv_bf16(const QkvGemvDesc& desc, const Bf16* x, int m, int k,
                   cudaStream_t stream) {
  if (m <= 0 || k <= 0) return;
  const int total_n = desc.n_out[0] + desc.n_out[1] + desc.n_out[2];
  if (total_n <= 0) return;
  // For m >= 16, each projection uses Tensor Core 64x64x64 MMA GEMM.
  if (m >= 16) {
    for (int p = 0; p < 3; ++p) {
      gemm_bf16(desc.out[p], x, desc.w[p], desc.bias[p], m, k, desc.n_out[p],
                stream);
    }
    return;
  }
  if ((k & 7) == 0 && aligned16(x) && aligned16(desc.w[0]) &&
      aligned16(desc.w[1]) && aligned16(desc.w[2])) {
    const int full = m / 8;
    const int rem = m % 8;
    if (full > 0) {
      launch_qkv_exact<8>(desc, x, full, total_n, k, stream);
    }
    if (rem > 0) {
      const size_t off_rows = static_cast<size_t>(full) * 8;
      QkvGemvDesc tail = desc;
      for (int p = 0; p < 3; ++p)
        tail.out[p] = desc.out[p] + off_rows * desc.n_out[p];
      launch_qkv_tail(tail, x + off_rows * k, rem, total_n, k, stream);
    }
    return;
  }
  for (int p = 0; p < 3; ++p) {
    gemm_bf16(desc.out[p], x, desc.w[p], desc.bias[p], m, k, desc.n_out[p],
              stream);
  }
}

void gemv_swiglu_bf16(const Bf16* w_gate, const Bf16* w_up, const Bf16* x,
                      Bf16* out, int m, int n_out, int k, cudaStream_t stream) {
  if (m <= 0 || n_out <= 0 || k <= 0) return;
  if ((k & 7) == 0 && aligned16(w_gate) && aligned16(w_up) && aligned16(x)) {
    const int full = m / 8;
    const int rem = m % 8;
    if (full > 0) {
      launch_swiglu_exact<8>(w_gate, w_up, x, out, full, n_out, k, stream);
    }
    if (rem > 0) {
      const size_t off_rows = static_cast<size_t>(full) * 8;
      launch_swiglu_tail(w_gate, w_up, x + off_rows * k, out + off_rows * n_out,
                         rem, n_out, k, stream);
    }
    return;
  }
  dim3 grid(n_out, m);
  gemv_swiglu_scalar_kernel<<<grid, kBlock, 0, stream>>>(w_gate, w_up, x, out,
                                                         m, n_out, k);
}

void gemm_bf16_naive(Bf16* y, const Bf16* x, const Bf16* w, const Bf16* bias,
                     int n_tokens, int k, int m_out, cudaStream_t stream) {
  gemm_bf16(y, x, w, bias, n_tokens, k, m_out, stream);
}

}  // namespace umminfer::native
