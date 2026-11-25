// SPDX-License-Identifier: Apache-2.0
// Elementwise, attention and hybrid-layer kernels for Qwen3.8-27B.
#include "qwen38_ops.hpp"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace umminfer::qwen38 {
namespace {

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float bf(const Bf16& x) { return __bfloat162float(x); }
__device__ __forceinline__ Bf16 to_bf(float x) { return __float2bfloat16(x); }

inline bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0u;
}
inline bool aligned8(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 7u) == 0u;
}

// ---- RMSNorm (plain, zero-centred, and fused residual+RMSNorm) -------------
template <bool kZeroCentered>
__global__ void rms_norm_kernel(const Bf16* __restrict__ x,
                                const Bf16* __restrict__ w, Bf16* __restrict__ out,
                                int dim, float eps) {
  const int row = blockIdx.x;
  const Bf16* xr = x + static_cast<size_t>(row) * dim;
  Bf16* orow = out + static_cast<size_t>(row) * dim;
  __shared__ float warp_sums[32];
  __shared__ float s_inv;
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int nwarps = blockDim.x >> 5;
  float sum = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = bf(xr[i]);
    sum += v * v;
  }
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
  if (lane == 0) warp_sums[wid] = sum;
  __syncthreads();
  if (wid == 0) {
    sum = lane < nwarps ? warp_sums[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) s_inv = rsqrtf(sum / dim + eps);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float scale = kZeroCentered ? 1.0f + bf(w[i]) : bf(w[i]);
    orow[i] = to_bf(bf(xr[i]) * s_inv * scale);
  }
}

// Single-warp vectorized RMSNorm for dim <= 256 (dim % 8 == 0, 16B aligned).
// Zero shared memory, zero __syncthreads(), single 128-bit load & store.
template <bool kZeroCentered>
__global__ void rms_norm_warp_vec_kernel(const Bf16* __restrict__ x,
                                         const Bf16* __restrict__ w,
                                         Bf16* __restrict__ out, int dim,
                                         float eps) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  const int vecs = dim >> 3;
  const uint4* xv = reinterpret_cast<const uint4*>(x + static_cast<size_t>(row) * dim);
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
      const float s0 = kZeroCentered ? (1.0f + wf.x) : wf.x;
      const float s1 = kZeroCentered ? (1.0f + wf.y) : wf.y;
      ob[e] = __floats2bfloat162_rn(xf[2 * e] * inv * s0, xf[2 * e + 1] * inv * s1);
    }
    ov[lane] = oraw;
  }
}

// 256-thread vectorized RMSNorm for dim % 8 == 0, 16B aligned.
// Caches up to 3 uint4s per thread (dim <= 6144, covering kHidden = 5120) in
// registers so x is read only once from memory.
template <bool kZeroCentered>
__global__ void __launch_bounds__(256) rms_norm_vec_kernel(
    const Bf16* __restrict__ x, const Bf16* __restrict__ w,
    Bf16* __restrict__ out, int dim, float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31, wid = tid >> 5;
  const int vecs = dim >> 3;
  const uint4* xv = reinterpret_cast<const uint4*>(x + static_cast<size_t>(row) * dim);
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
      const float s0 = kZeroCentered ? (1.0f + wf.x) : wf.x;
      const float s1 = kZeroCentered ? (1.0f + wf.y) : wf.y;
      ob[e] = __floats2bfloat162_rn(xf.x * inv * s0, xf.y * inv * s1);
    }
    ov[i] = oraw;
  }
}

// Fused residual add + zero-centred RMSNorm (vectorized uint4 path).
// Exact bf16 rounding: hidden[i] = to_bf(bf(hidden[i]) + bf(residual[i])) is
// computed and stored once, kept in registers across the reduction, and then
// normalized directly into norm_out without reloading hidden.
__global__ void __launch_bounds__(256) add_residual_rms_norm_vec_kernel(
    Bf16* __restrict__ hidden, const Bf16* __restrict__ residual,
    const Bf16* __restrict__ w, Bf16* __restrict__ norm_out, int dim,
    float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31, wid = tid >> 5;
  const int vecs = dim >> 3;
  uint4* hv = reinterpret_cast<uint4*>(hidden + static_cast<size_t>(row) * dim);
  const uint4* rv =
      reinterpret_cast<const uint4*>(residual + static_cast<size_t>(row) * dim);
  const uint4* wv = reinterpret_cast<const uint4*>(w);
  uint4* ov =
      reinterpret_cast<uint4*>(norm_out + static_cast<size_t>(row) * dim);

  __shared__ float warp_sums[8];
  __shared__ float s_inv;

  uint4 reg_h[3];
  const bool cached = (vecs <= 768);
  float sum = 0.0f;
  int idx_reg = 0;
  for (int i = tid; i < vecs; i += 256) {
    const uint4 hraw = hv[i];
    const uint4 rraw = __ldg(rv + i);
    const __nv_bfloat162* hb = reinterpret_cast<const __nv_bfloat162*>(&hraw);
    const __nv_bfloat162* rb = reinterpret_cast<const __nv_bfloat162*>(&rraw);
    uint4 updated;
    __nv_bfloat162* ub = reinterpret_cast<__nv_bfloat162*>(&updated);
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 hf = __bfloat1622float2(hb[e]);
      const float2 rf = __bfloat1622float2(rb[e]);
      const __nv_bfloat162 h2 = __floats2bfloat162_rn(hf.x + rf.x, hf.y + rf.y);
      ub[e] = h2;
      const float2 nf = __bfloat1622float2(h2);
      sum += nf.x * nf.x + nf.y * nf.y;
    }
    hv[i] = updated;
    if (cached) reg_h[idx_reg++] = updated;
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
    const uint4 raw = cached ? reg_h[idx_reg++] : hv[i];
    const uint4 wraw = __ldg(wv + i);
    const __nv_bfloat162* hb = reinterpret_cast<const __nv_bfloat162*>(&raw);
    const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wraw);
    uint4 oraw;
    __nv_bfloat162* ob = reinterpret_cast<__nv_bfloat162*>(&oraw);
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 hf = __bfloat1622float2(hb[e]);
      const float2 wf = __bfloat1622float2(wb[e]);
      ob[e] = __floats2bfloat162_rn(hf.x * inv * (1.0f + wf.x),
                                    hf.y * inv * (1.0f + wf.y));
    }
    ov[i] = oraw;
  }
}

__global__ void add_residual_rms_norm_scalar_kernel(
    Bf16* __restrict__ hidden, const Bf16* __restrict__ residual,
    const Bf16* __restrict__ w, Bf16* __restrict__ norm_out, int dim,
    float eps) {
  const int row = blockIdx.x;
  Bf16* hr = hidden + static_cast<size_t>(row) * dim;
  const Bf16* rr = residual + static_cast<size_t>(row) * dim;
  Bf16* orow = norm_out + static_cast<size_t>(row) * dim;
  __shared__ float warp_sums[32];
  __shared__ float s_inv;
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int nwarps = blockDim.x >> 5;
  float sum = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const Bf16 h = to_bf(bf(hr[i]) + bf(rr[i]));
    hr[i] = h;
    const float v = bf(h);
    sum += v * v;
  }
  for (int off = 16; off > 0; off >>= 1)
    sum += __shfl_down_sync(0xffffffffu, sum, off);
  if (lane == 0) warp_sums[wid] = sum;
  __syncthreads();
  if (wid == 0) {
    sum = lane < nwarps ? warp_sums[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1)
      sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) s_inv = rsqrtf(sum / dim + eps);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    orow[i] = to_bf(bf(hr[i]) * s_inv * (1.0f + bf(w[i])));
  }
}

__global__ void layer_norm_kernel(const Bf16* __restrict__ x,
                                  const Bf16* __restrict__ w,
                                  const Bf16* __restrict__ b, Bf16* __restrict__ out,
                                  int dim, float eps) {
  const int row = blockIdx.x;
  const Bf16* xr = x + static_cast<size_t>(row) * dim;
  Bf16* orow = out + static_cast<size_t>(row) * dim;
  __shared__ float s_mean, s_inv;
  __shared__ float warp_sums[32];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int nwarps = blockDim.x >> 5;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) acc += bf(xr[i]);
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) warp_sums[wid] = acc;
  __syncthreads();
  if (wid == 0) {
    acc = lane < nwarps ? warp_sums[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) s_mean = acc / dim;
  }
  __syncthreads();
  acc = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float d = bf(xr[i]) - s_mean;
    acc += d * d;
  }
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) warp_sums[wid] = acc;
  __syncthreads();
  if (wid == 0) {
    acc = lane < nwarps ? warp_sums[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) s_inv = rsqrtf(acc / dim + eps);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    orow[i] = to_bf((bf(xr[i]) - s_mean) * s_inv * bf(w[i]) + bf(b[i]));
}

template <typename Op>
__global__ void elementwise_kernel(Bf16* x, int n, Op op) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = to_bf(op(bf(x[i])));
}

struct GeluTanhOp {
  __device__ float operator()(float x) const {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(kSqrt2OverPi * (x + 0.044715f * x * x * x)));
  }
};
struct GeluErfOp {
  __device__ float operator()(float x) const {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
  }
};

__global__ void bias_kernel(Bf16* y, const Bf16* bias, int rows, int dim) {
  const int r = blockIdx.x;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const int idx = r * dim + i;
    y[idx] = to_bf(bf(y[idx]) + bf(bias[i]));
  }
}

__global__ void residual_kernel(Bf16* y, const Bf16* r, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = to_bf(bf(y[i]) + bf(r[i]));
}

__global__ void residual_vec_kernel(Bf16* __restrict__ y,
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

__global__ void residual_tiled_kernel(Bf16* y, const Bf16* r, int total_n,
                                      int tile_n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total_n) y[i] = to_bf(bf(y[i]) + bf(r[i % tile_n]));
}

__global__ void silu_mul_kernel(Bf16* out, const Bf16* gate, const Bf16* up, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float g = bf(gate[i]);
  const float sg = g / (1.0f + expf(-g));
  out[i] = to_bf(bf(to_bf(sg)) * bf(up[i]));
}

__global__ void silu_mul_vec_kernel(Bf16* __restrict__ out,
                                    const Bf16* __restrict__ gate,
                                    const Bf16* __restrict__ up, int vecs) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= vecs) return;
  const uint4 graw = __ldg(reinterpret_cast<const uint4*>(gate) + i);
  const uint4 uraw = __ldg(reinterpret_cast<const uint4*>(up) + i);
  const __nv_bfloat162* gb = reinterpret_cast<const __nv_bfloat162*>(&graw);
  const __nv_bfloat162* ub = reinterpret_cast<const __nv_bfloat162*>(&uraw);
  uint4 oraw;
  __nv_bfloat162* ob = reinterpret_cast<__nv_bfloat162*>(&oraw);
  #pragma unroll
  for (int e = 0; e < 4; ++e) {
    const float2 gf = __bfloat1622float2(gb[e]);
    const float2 uf = __bfloat1622float2(ub[e]);
    const float sg0 = bf(to_bf(gf.x / (1.0f + expf(-gf.x))));
    const float sg1 = bf(to_bf(gf.y / (1.0f + expf(-gf.y))));
    ob[e] = __floats2bfloat162_rn(sg0 * uf.x, sg1 * uf.y);
  }
  reinterpret_cast<uint4*>(out)[i] = oraw;
}

__global__ void f32_to_bf16_kernel(const float* src, Bf16* dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = to_bf(src[i]);
}

__global__ void bf16_to_f32_kernel(const Bf16* src, float* dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = bf(src[i]);
}

__global__ void embed_kernel(const Bf16* table, const int* ids, Bf16* out, int m,
                             int dim) {
  const int row = blockIdx.y;
  if (row >= m) return;
  const Bf16* src = table + static_cast<size_t>(ids[row]) * dim;
  Bf16* dst = out + static_cast<size_t>(row) * dim;
  if ((dim & 7) == 0 && (reinterpret_cast<uintptr_t>(src) & 15u) == 0u &&
      (reinterpret_cast<uintptr_t>(dst) & 15u) == 0u) {
    const int vecs = dim >> 3;
    const uint4* sv = reinterpret_cast<const uint4*>(src);
    uint4* dv = reinterpret_cast<uint4*>(dst);
    for (int i = threadIdx.x; i < vecs; i += blockDim.x) dv[i] = __ldg(sv + i);
  } else {
    for (int i = threadIdx.x; i < dim; i += blockDim.x) dst[i] = src[i];
  }
}

__global__ void replace_rows_kernel(Bf16* dst, const Bf16* src, int start, int dim) {
  const int row = blockIdx.y;
  const Bf16* s = src + static_cast<size_t>(row) * dim;
  Bf16* d = dst + static_cast<size_t>(start + row) * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) d[i] = s[i];
}

// ---- GQA / MHA ------------------------------------------------------------
// Multi-warp single-pass Online Softmax for head_dim == 256 (text GQA).
// 8 warps per block split the KV sequence (kj = wid, wid+8, ...); each lane
// holds 8 contiguous head elements in registers and loads K/V with coalesced
// 128-bit uint4 transactions. Only ONE __syncthreads() per block (at the final
// warp merge) instead of 7 per KV token.
template <bool kGated>
__global__ void __launch_bounds__(256) gqa_attention_256_kernel(
    const Bf16* __restrict__ q, const Bf16* __restrict__ k,
    const Bf16* __restrict__ v, const Bf16* __restrict__ gate,
    Bf16* __restrict__ out, int nkv, int n_q_heads, int n_kv_heads,
    int q_offset) {
  constexpr int kHeadDim = 256;
  constexpr int kWarps = 8;
  const int qi = blockIdx.y;
  const int qh = blockIdx.x;
  const int tid = threadIdx.x;
  const int wid = tid >> 5;
  const int lane = tid & 31;
  const int kvh = qh * n_kv_heads / n_q_heads;
  const int hi = min(nkv, q_offset + qi + 1);
  const float scale = rsqrtf(static_cast<float>(kHeadDim));

  const Bf16* qrow = q + (static_cast<size_t>(qi) * n_q_heads + qh) * kHeadDim;
  const uint4 qv = __ldg(reinterpret_cast<const uint4*>(qrow) + lane);
  const __nv_bfloat162* qb = reinterpret_cast<const __nv_bfloat162*>(&qv);
  float qf[8];
  #pragma unroll
  for (int e = 0; e < 4; ++e) {
    const float2 f2 = __bfloat1622float2(qb[e]);
    qf[2 * e] = f2.x;
    qf[2 * e + 1] = f2.y;
  }

  float w_max = -INFINITY;
  float w_sum = 0.0f;
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int kj = wid; kj < hi; kj += kWarps) {
    const size_t kv_off = (static_cast<size_t>(kj) * n_kv_heads + kvh) * kHeadDim;
    const uint4 kv = __ldg(reinterpret_cast<const uint4*>(k + kv_off) + lane);
    const __nv_bfloat162* kb = reinterpret_cast<const __nv_bfloat162*>(&kv);
    float dot = 0.0f;
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 f2 = __bfloat1622float2(kb[e]);
      dot += qf[2 * e] * f2.x + qf[2 * e + 1] * f2.y;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      dot += __shfl_xor_sync(0xffffffffu, dot, off);
    const float score = dot * scale;
    const float new_max = fmaxf(w_max, score);
    const float alpha = (w_sum > 0.0f) ? expf(w_max - new_max) : 0.0f;
    const float p = expf(score - new_max);
    w_sum = w_sum * alpha + p;
    w_max = new_max;
    const uint4 vv = __ldg(reinterpret_cast<const uint4*>(v + kv_off) + lane);
    const __nv_bfloat162* vb = reinterpret_cast<const __nv_bfloat162*>(&vv);
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 f2 = __bfloat1622float2(vb[e]);
      acc[2 * e] = acc[2 * e] * alpha + p * f2.x;
      acc[2 * e + 1] = acc[2 * e + 1] * alpha + p * f2.y;
    }
  }

  __shared__ float sm_max[kWarps];
  __shared__ float sm_sum[kWarps];
  __shared__ float sm_acc[kWarps][kHeadDim];
  if (lane == 0) {
    sm_max[wid] = w_max;
    sm_sum[wid] = w_sum;
  }
  #pragma unroll
  for (int e = 0; e < 8; ++e) sm_acc[wid][lane * 8 + e] = acc[e];
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
    Bf16* orow = out + (static_cast<size_t>(qi) * n_q_heads + qh) * kHeadDim;
    uint4 gv = make_uint4(0, 0, 0, 0);
    if constexpr (kGated) {
      gv = __ldg(reinterpret_cast<const uint4*>(
                     gate + (static_cast<size_t>(qi) * n_q_heads + qh) * kHeadDim) +
                 lane);
    }
    const __nv_bfloat162* gb = reinterpret_cast<const __nv_bfloat162*>(&gv);
    uint4 ov;
    __nv_bfloat162* ob = reinterpret_cast<__nv_bfloat162*>(&ov);
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      float a0 = 0.0f, a1 = 0.0f;
      #pragma unroll
      for (int w = 0; w < kWarps; ++w) {
        a0 += sm_acc[w][lane * 8 + 2 * e] * scale_w[w];
        a1 += sm_acc[w][lane * 8 + 2 * e + 1] * scale_w[w];
      }
      float o0 = bf(to_bf(a0 * inv));
      float o1 = bf(to_bf(a1 * inv));
      if constexpr (kGated) {
        const float2 gf = __bfloat1622float2(gb[e]);
        o0 *= bf(to_bf(1.0f / (1.0f + expf(-gf.x))));
        o1 *= bf(to_bf(1.0f / (1.0f + expf(-gf.y))));
      }
      ob[e] = __floats2bfloat162_rn(o0, o1);
    }
    reinterpret_cast<uint4*>(orow)[lane] = ov;
  }
}

template <bool kGated>
__global__ void __launch_bounds__(256) gqa_attention_256_seg_kernel(
    const Bf16* __restrict__ q, const Bf16* __restrict__ kbase,
    const Bf16* __restrict__ vbase, const Bf16* __restrict__ gate,
    Bf16* __restrict__ out, SeqSpans spans, int nseq, size_t seq_stride,
    int n_q_heads, int n_kv_heads) {
  constexpr int kHeadDim = 256;
  constexpr int kWarps = 8;
  const int row = blockIdx.y;
  const int qh = blockIdx.x;
  const int tid = threadIdx.x;
  const int wid = tid >> 5;
  const int lane = tid & 31;

  int s = 0;
  while (s + 1 < nseq && row >= spans.off[s] + spans.len[s]) ++s;
  const int hi = spans.hist[s] + (row - spans.off[s]) + 1;
  const Bf16* k = kbase + static_cast<size_t>(spans.slot[s]) * seq_stride;
  const Bf16* v = vbase + static_cast<size_t>(spans.slot[s]) * seq_stride;
  const int kvh = qh * n_kv_heads / n_q_heads;
  const float scale = rsqrtf(static_cast<float>(kHeadDim));

  const Bf16* qrow = q + (static_cast<size_t>(row) * n_q_heads + qh) * kHeadDim;
  const uint4 qv = __ldg(reinterpret_cast<const uint4*>(qrow) + lane);
  const __nv_bfloat162* qb = reinterpret_cast<const __nv_bfloat162*>(&qv);
  float qf[8];
  #pragma unroll
  for (int e = 0; e < 4; ++e) {
    const float2 f2 = __bfloat1622float2(qb[e]);
    qf[2 * e] = f2.x;
    qf[2 * e + 1] = f2.y;
  }

  float w_max = -INFINITY;
  float w_sum = 0.0f;
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int kj = wid; kj < hi; kj += kWarps) {
    const size_t kv_off = (static_cast<size_t>(kj) * n_kv_heads + kvh) * kHeadDim;
    const uint4 kv = __ldg(reinterpret_cast<const uint4*>(k + kv_off) + lane);
    const __nv_bfloat162* kb = reinterpret_cast<const __nv_bfloat162*>(&kv);
    float dot = 0.0f;
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 f2 = __bfloat1622float2(kb[e]);
      dot += qf[2 * e] * f2.x + qf[2 * e + 1] * f2.y;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      dot += __shfl_xor_sync(0xffffffffu, dot, off);
    const float score = dot * scale;
    const float new_max = fmaxf(w_max, score);
    const float alpha = (w_sum > 0.0f) ? expf(w_max - new_max) : 0.0f;
    const float p = expf(score - new_max);
    w_sum = w_sum * alpha + p;
    w_max = new_max;
    const uint4 vv = __ldg(reinterpret_cast<const uint4*>(v + kv_off) + lane);
    const __nv_bfloat162* vb = reinterpret_cast<const __nv_bfloat162*>(&vv);
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 f2 = __bfloat1622float2(vb[e]);
      acc[2 * e] = acc[2 * e] * alpha + p * f2.x;
      acc[2 * e + 1] = acc[2 * e + 1] * alpha + p * f2.y;
    }
  }

  __shared__ float sm_max[kWarps];
  __shared__ float sm_sum[kWarps];
  __shared__ float sm_acc[kWarps][kHeadDim];
  if (lane == 0) {
    sm_max[wid] = w_max;
    sm_sum[wid] = w_sum;
  }
  #pragma unroll
  for (int e = 0; e < 8; ++e) sm_acc[wid][lane * 8 + e] = acc[e];
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
    Bf16* orow = out + (static_cast<size_t>(row) * n_q_heads + qh) * kHeadDim;
    uint4 gv = make_uint4(0, 0, 0, 0);
    if constexpr (kGated) {
      gv = __ldg(reinterpret_cast<const uint4*>(
                     gate + (static_cast<size_t>(row) * n_q_heads + qh) * kHeadDim) +
                 lane);
    }
    const __nv_bfloat162* gb = reinterpret_cast<const __nv_bfloat162*>(&gv);
    uint4 ov;
    __nv_bfloat162* ob = reinterpret_cast<__nv_bfloat162*>(&ov);
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      float a0 = 0.0f, a1 = 0.0f;
      #pragma unroll
      for (int w = 0; w < kWarps; ++w) {
        a0 += sm_acc[w][lane * 8 + 2 * e] * scale_w[w];
        a1 += sm_acc[w][lane * 8 + 2 * e + 1] * scale_w[w];
      }
      float o0 = bf(to_bf(a0 * inv));
      float o1 = bf(to_bf(a1 * inv));
      if constexpr (kGated) {
        const float2 gf = __bfloat1622float2(gb[e]);
        o0 *= bf(to_bf(1.0f / (1.0f + expf(-gf.x))));
        o1 *= bf(to_bf(1.0f / (1.0f + expf(-gf.y))));
      }
      ob[e] = __floats2bfloat162_rn(o0, o1);
    }
    reinterpret_cast<uint4*>(orow)[lane] = ov;
  }
}

// Fast 4-warp single-pass Online Softmax for vision attention (head_dim == 72).
__global__ void __launch_bounds__(128) vision_attention_72_kernel(
    const Bf16* __restrict__ q, const Bf16* __restrict__ k,
    const Bf16* __restrict__ v, Bf16* __restrict__ out, int slice_tokens,
    int heads) {
  constexpr int kDim = 72;
  constexpr int kWarps = 4;
  const int qi = blockIdx.y;
  const int qh = blockIdx.x;
  const int tid = threadIdx.x;
  const int wid = tid >> 5;
  const int lane = tid & 31;
  const size_t slice_elems =
      static_cast<size_t>(blockIdx.z) * slice_tokens * heads * kDim;
  q += slice_elems;
  k += slice_elems;
  v += slice_elems;
  out += slice_elems;

  const float scale = rsqrtf(static_cast<float>(kDim));
  const Bf16* qrow = q + (static_cast<size_t>(qi) * heads + qh) * kDim;
  const float q0 = bf(qrow[lane]);
  const float q1 = bf(qrow[lane + 32]);
  const float q2 = (lane < 8) ? bf(qrow[lane + 64]) : 0.0f;

  float w_max = -INFINITY;
  float w_sum = 0.0f;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;

  for (int kj = wid; kj < slice_tokens; kj += kWarps) {
    const size_t kv_off = (static_cast<size_t>(kj) * heads + qh) * kDim;
    const Bf16* krow = k + kv_off;
    float dot = q0 * bf(krow[lane]) + q1 * bf(krow[lane + 32]);
    if (lane < 8) dot += q2 * bf(krow[lane + 64]);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      dot += __shfl_xor_sync(0xffffffffu, dot, off);
    const float score = dot * scale;
    const float new_max = fmaxf(w_max, score);
    const float alpha = (w_sum > 0.0f) ? expf(w_max - new_max) : 0.0f;
    const float p = expf(score - new_max);
    w_sum = w_sum * alpha + p;
    w_max = new_max;
    const Bf16* vrow = v + kv_off;
    a0 = a0 * alpha + p * bf(vrow[lane]);
    a1 = a1 * alpha + p * bf(vrow[lane + 32]);
    if (lane < 8) a2 = a2 * alpha + p * bf(vrow[lane + 64]);
  }

  __shared__ float sm_max[kWarps];
  __shared__ float sm_sum[kWarps];
  __shared__ float sm_acc[kWarps][kDim];
  if (lane == 0) {
    sm_max[wid] = w_max;
    sm_sum[wid] = w_sum;
  }
  sm_acc[wid][lane] = a0;
  sm_acc[wid][lane + 32] = a1;
  if (lane < 8) sm_acc[wid][lane + 64] = a2;
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
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f;
    #pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      r0 += sm_acc[w][lane] * scale_w[w];
      r1 += sm_acc[w][lane + 32] * scale_w[w];
      if (lane < 8) r2 += sm_acc[w][lane + 64] * scale_w[w];
    }
    Bf16* orow = out + (static_cast<size_t>(qi) * heads + qh) * kDim;
    orow[lane] = to_bf(r0 * inv);
    orow[lane + 32] = to_bf(r1 * inv);
    if (lane < 8) orow[lane + 64] = to_bf(r2 * inv);
  }
}

// Generic fallback attention kernel for non-standard head_dim or alignment.
__device__ __forceinline__ float block_dot(const float* qs, const bf16* row,
                                           int dim, float* reduce) {
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const int nwarps = blockDim.x >> 5;
  float partial = 0.0f;
  for (int i = tid; i < dim; i += blockDim.x) partial += qs[i] * __bfloat162float(row[i]);
  for (int off = 16; off > 0; off >>= 1)
    partial += __shfl_down_sync(0xffffffffu, partial, off);
  if (lane == 0) reduce[wid] = partial;
  __syncthreads();
  __shared__ float s_total;
  if (wid == 0) {
    float total = lane < nwarps ? reduce[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1)
      total += __shfl_down_sync(0xffffffffu, total, off);
    if (lane == 0) s_total = total;
  }
  __syncthreads();
  return s_total;
}

template <bool kCausal, bool kGated>
__global__ void attention_kernel(const Bf16* __restrict__ q,
                                 const Bf16* __restrict__ k,
                                 const Bf16* __restrict__ v,
                                 const Bf16* __restrict__ gate,
                                 Bf16* __restrict__ out, int m, int nkv,
                                 int n_q_heads, int n_kv_heads, int head_dim,
                                 int q_offset) {
  extern __shared__ float smem[];
  float* qs = smem;
  float* acc = smem + head_dim;
  float* reduce = smem + 2 * head_dim;
  __shared__ float s_max, s_sum, s_w;
  const int qi = blockIdx.y;
  const int qh = blockIdx.x;
  const int tid = threadIdx.x;
  if constexpr (!kCausal && !kGated) {
    const size_t slice_elems =
        static_cast<size_t>(blockIdx.z) * m * n_q_heads * head_dim;
    q += slice_elems;
    k += slice_elems;
    v += slice_elems;
    out += slice_elems;
  }
  const int kvh = qh * n_kv_heads / n_q_heads;
  const Bf16* qrow = q + (static_cast<size_t>(qi) * n_q_heads + qh) * head_dim;
  for (int i = tid; i < head_dim; i += blockDim.x) qs[i] = bf(qrow[i]);
  __syncthreads();

  int lo = 0, hi = nkv;
  if (kCausal) hi = min(nkv, q_offset + qi + 1);
  const float scale = rsqrtf(static_cast<float>(head_dim));

  if (tid == 0) s_max = -INFINITY;
  __syncthreads();
  for (int kj = lo; kj < hi; ++kj) {
    const bf16* krow =
        reinterpret_cast<const bf16*>(k) +
        (static_cast<size_t>(kj) * n_kv_heads + kvh) * head_dim;
    const float dot = block_dot(qs, krow, head_dim, reduce) * scale;
    if (tid == 0) s_max = fmaxf(s_max, dot);
    __syncthreads();
  }
  const float mx = s_max;

  if (tid == 0) s_sum = 0.0f;
  for (int i = tid; i < head_dim; i += blockDim.x) acc[i] = 0.0f;
  __syncthreads();
  for (int kj = lo; kj < hi; ++kj) {
    const bf16* krow =
        reinterpret_cast<const bf16*>(k) +
        (static_cast<size_t>(kj) * n_kv_heads + kvh) * head_dim;
    const float weight = expf(block_dot(qs, krow, head_dim, reduce) * scale - mx);
    if (tid == 0) {
      s_w = weight;
      s_sum += weight;
    }
    __syncthreads();
    const bf16* vrow =
        reinterpret_cast<const bf16*>(v) +
        (static_cast<size_t>(kj) * n_kv_heads + kvh) * head_dim;
    const float wgt = s_w;
    for (int i = tid; i < head_dim; i += blockDim.x)
      acc[i] += wgt * __bfloat162float(vrow[i]);
    __syncthreads();
  }
  const float inv = 1.0f / s_sum;
  Bf16* orow = out + (static_cast<size_t>(qi) * n_q_heads + qh) * head_dim;
  const Bf16* grow =
      kGated ? gate + (static_cast<size_t>(qi) * n_q_heads + qh) * head_dim : nullptr;
  for (int i = tid; i < head_dim; i += blockDim.x) {
    float o = bf(to_bf(acc[i] * inv));
    if (kGated) {
      const float gv = bf(grow[i]);
      const float sg = 1.0f / (1.0f + expf(-gv));
      o *= bf(to_bf(sg));
    }
    orow[i] = to_bf(o);
  }
}

// ---- mRoPE & fused QKV prep + KV-cache append -----------------------------
__global__ void mrope_partial_kernel(Bf16* x, const Bf16* cos, const Bf16* sin,
                                     int n, int heads, int head_dim,
                                     int rotary_dim) {
  const int token = blockIdx.y;
  const int half = rotary_dim / 2;
  for (int h = threadIdx.y; h < heads; h += blockDim.y) {
    Bf16* row = x + (static_cast<size_t>(token) * heads + h) * head_dim;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
      const float c = bf(cos[static_cast<size_t>(token) * rotary_dim + i]);
      const float s = bf(sin[static_cast<size_t>(token) * rotary_dim + i]);
      const float x1 = bf(row[i]), x2 = bf(row[i + half]);
      row[i] = to_bf(x1 * c - x2 * s);
      row[i + half] = to_bf(x2 * c + x1 * s);
    }
  }
}

// Fuses split_q_gate + q_norm + k_norm + mrope_partial(q, k) + KV-cache write
// for head_dim == 256 into a single kernel launch.
// Grid: dim3(n_q_heads + 2 * n_kv_heads, rows), Block: 256 threads.
template <bool kSeg>
__global__ void __launch_bounds__(256) prep_qkv_rope_kvcache_256_kernel(
    const Bf16* __restrict__ qg, const Bf16* __restrict__ k_in,
    const Bf16* __restrict__ v_in, const Bf16* __restrict__ q_norm_w,
    const Bf16* __restrict__ k_norm_w, const Bf16* __restrict__ cos,
    const Bf16* __restrict__ sin, Bf16* __restrict__ q_out,
    Bf16* __restrict__ gate_out, Bf16* __restrict__ k_cache,
    Bf16* __restrict__ v_cache, int history, SeqSpans spans, int nseq,
    size_t seq_stride, int n_q_heads, int n_kv_heads, int rotary_dim,
    float eps) {
  constexpr int kHeadDim = 256;
  const int b = blockIdx.x;
  const int t = blockIdx.y;
  const int i = threadIdx.x;  // 0..255

  int cache_pos = history + t;
  size_t slot_off = 0;
  if constexpr (kSeg) {
    int s = 0;
    while (s + 1 < nseq && t >= spans.off[s] + spans.len[s]) ++s;
    cache_pos = spans.hist[s] + (t - spans.off[s]);
    slot_off = static_cast<size_t>(spans.slot[s]) * seq_stride;
  }

  // V head: direct copy into v_cache.
  if (b >= n_q_heads + n_kv_heads) {
    const int h = b - n_q_heads - n_kv_heads;
    const Bf16* v_src = v_in + (static_cast<size_t>(t) * n_kv_heads + h) * kHeadDim;
    Bf16* v_dst =
        v_cache + slot_off + (static_cast<size_t>(cache_pos) * n_kv_heads + h) * kHeadDim;
    v_dst[i] = v_src[i];
    return;
  }

  __shared__ float warp_sums[8];
  __shared__ float s_inv;
  __shared__ float s_rot[256];

  const int lane = i & 31, wid = i >> 5;
  const bool is_q = (b < n_q_heads);
  float v = 0.0f;
  const Bf16* norm_w = nullptr;
  Bf16* dst = nullptr;

  if (is_q) {
    const int h = b;
    const Bf16* qg_row =
        qg + (static_cast<size_t>(t) * n_q_heads + h) * (2 * kHeadDim);
    Bf16* g_dst = gate_out + (static_cast<size_t>(t) * n_q_heads + h) * kHeadDim;
    g_dst[i] = qg_row[kHeadDim + i];
    v = bf(qg_row[i]);
    norm_w = q_norm_w;
    dst = q_out + (static_cast<size_t>(t) * n_q_heads + h) * kHeadDim;
  } else {
    const int h = b - n_q_heads;
    const Bf16* k_src = k_in + (static_cast<size_t>(t) * n_kv_heads + h) * kHeadDim;
    v = bf(k_src[i]);
    norm_w = k_norm_w;
    dst = k_cache + slot_off +
          (static_cast<size_t>(cache_pos) * n_kv_heads + h) * kHeadDim;
  }

  float sum = v * v;
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
    if (lane == 0) s_inv = rsqrtf(sum / kHeadDim + eps);
  }
  __syncthreads();

  const Bf16 normed = to_bf(v * s_inv * (1.0f + bf(norm_w[i])));
  if (i < rotary_dim) {
    s_rot[i] = bf(normed);
  } else {
    dst[i] = normed;
  }
  __syncthreads();

  const int half = rotary_dim >> 1;
  if (i < half) {
    const float c = bf(cos[static_cast<size_t>(t) * rotary_dim + i]);
    const float s_val = bf(sin[static_cast<size_t>(t) * rotary_dim + i]);
    const float x1 = s_rot[i];
    const float x2 = s_rot[i + half];
    dst[i] = to_bf(x1 * c - x2 * s_val);
    dst[i + half] = to_bf(x2 * c + x1 * s_val);
  }
}

__global__ void vision_rope_kernel(Bf16* x, const float* cos, const float* sin,
                                   int n, int slice_tokens, int heads,
                                   int head_dim) {
  const int token = blockIdx.y;
  if (token >= n) return;
  const int pos_tok =
      (slice_tokens > 0 && slice_tokens < n) ? (token % slice_tokens) : token;
  const int half = head_dim / 2;
  for (int h = threadIdx.y; h < heads; h += blockDim.y) {
    Bf16* row = x + (static_cast<size_t>(token) * heads + h) * head_dim;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
      const float c = cos[static_cast<size_t>(pos_tok) * head_dim + i];
      const float s = sin[static_cast<size_t>(pos_tok) * head_dim + i];
      const float x1 = bf(row[i]), x2 = bf(row[i + half]);
      row[i] = to_bf(x1 * c - x2 * s);
      row[i + half] = to_bf(x2 * c + x1 * s);
    }
  }
}

__global__ void split_q_gate_kernel(const Bf16* qg, Bf16* q, Bf16* gate, int m,
                                    int heads, int head_dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = m * heads * head_dim;
  if (idx >= total) return;
  const int i = idx % head_dim;
  const int h = (idx / head_dim) % heads;
  const int t = idx / (head_dim * heads);
  const size_t src = (static_cast<size_t>(t) * heads + h) * 2 * head_dim;
  q[idx] = qg[src + i];
  gate[idx] = qg[src + head_dim + i];
}

__global__ void split_qkv_kernel(const Bf16* qkv, Bf16* q, Bf16* k, Bf16* v,
                                 int m, int width) {
  const int t = blockIdx.y;
  if (t >= m) return;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= width) return;
  const Bf16* src = qkv + static_cast<size_t>(t) * 3 * width;
  const size_t dst = static_cast<size_t>(t) * width + j;
  q[dst] = src[j];
  k[dst] = src[width + j];
  v[dst] = src[2 * width + j];
}

__global__ void split_qkv_widths_kernel(const Bf16* qkv, Bf16* q, Bf16* k, Bf16* v,
                                        int m, int wq, int wk, int wv) {
  const int t = blockIdx.y;
  if (t >= m) return;
  const int total = wq + wk + wv;
  const Bf16* src = qkv + static_cast<size_t>(t) * total;
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < total;
       j += blockDim.x * gridDim.x) {
    if (j < wq)
      q[static_cast<size_t>(t) * wq + j] = src[j];
    else if (j < wq + wk)
      k[static_cast<size_t>(t) * wk + j - wq] = src[j];
    else
      v[static_cast<size_t>(t) * wv + j - wq - wk] = src[j];
  }
}

// ---- Linear attention -----------------------------------------------------
__global__ void conv1d_causal_silu_kernel(const Bf16* x, const Bf16* w, Bf16* state,
                                          Bf16* out, int m, int channels) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) return;
  float h0 = bf(state[c]);
  float h1 = bf(state[channels + c]);
  float h2 = bf(state[2 * channels + c]);
  const float w0 = bf(w[c * 4 + 0]), w1 = bf(w[c * 4 + 1]);
  const float w2 = bf(w[c * 4 + 2]), w3 = bf(w[c * 4 + 3]);
  for (int t = 0; t < m; ++t) {
    const float xt = bf(x[static_cast<size_t>(t) * channels + c]);
    const float acc = bf(to_bf(w0 * h0 + w1 * h1 + w2 * h2 + w3 * xt));
    out[static_cast<size_t>(t) * channels + c] = to_bf(acc / (1.0f + expf(-acc)));
    h0 = h1;
    h1 = h2;
    h2 = xt;
  }
  state[c] = to_bf(h0);
  state[channels + c] = to_bf(h1);
  state[2 * channels + c] = to_bf(h2);
}

// Fused conv1d + SiLU writing split q, k, v outputs directly.
__global__ void conv1d_causal_silu_split_kernel(const Bf16* __restrict__ x,
                                                const Bf16* __restrict__ w,
                                                Bf16* __restrict__ state,
                                                Bf16* __restrict__ q,
                                                Bf16* __restrict__ k,
                                                Bf16* __restrict__ v, int m,
                                                int wq, int wk, int wv) {
  const int channels = wq + wk + wv;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) return;

  float h0 = bf(state[c]);
  float h1 = bf(state[channels + c]);
  float h2 = bf(state[2 * channels + c]);
  const uint2 wraw = __ldg(reinterpret_cast<const uint2*>(w) + c);
  const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wraw);
  const float2 wf01 = __bfloat1622float2(wb[0]);
  const float2 wf23 = __bfloat1622float2(wb[1]);
  const float w0 = wf01.x, w1 = wf01.y, w2 = wf23.x, w3 = wf23.y;

  Bf16* dst;
  int dst_stride;
  int dst_col;
  if (c < wq) {
    dst = q;
    dst_stride = wq;
    dst_col = c;
  } else if (c < wq + wk) {
    dst = k;
    dst_stride = wk;
    dst_col = c - wq;
  } else {
    dst = v;
    dst_stride = wv;
    dst_col = c - wq - wk;
  }

  for (int t = 0; t < m; ++t) {
    const float xt = bf(x[static_cast<size_t>(t) * channels + c]);
    const float acc = bf(to_bf(w0 * h0 + w1 * h1 + w2 * h2 + w3 * xt));
    dst[static_cast<size_t>(t) * dst_stride + dst_col] =
        to_bf(acc / (1.0f + expf(-acc)));
    h0 = h1;
    h1 = h2;
    h2 = xt;
  }
  state[c] = to_bf(h0);
  state[channels + c] = to_bf(h1);
  state[2 * channels + c] = to_bf(h2);
}

// Per-sequence fused conv1d + SiLU + split, parallelized across both channels
// (blockIdx.x) and sequences (blockIdx.y).
__global__ void conv1d_causal_silu_split_seg_kernel(
    const Bf16* __restrict__ x, const Bf16* __restrict__ w,
    Bf16* __restrict__ state, Bf16* __restrict__ q, Bf16* __restrict__ k,
    Bf16* __restrict__ v, SeqSpans spans, int wq, int wk, int wv,
    size_t seq_stride) {
  const int channels = wq + wk + wv;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) return;
  const int s = blockIdx.y;
  const int len = spans.len[s];
  if (len <= 0) return;

  const uint2 wraw = __ldg(reinterpret_cast<const uint2*>(w) + c);
  const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wraw);
  const float2 wf01 = __bfloat1622float2(wb[0]);
  const float2 wf23 = __bfloat1622float2(wb[1]);
  const float w0 = wf01.x, w1 = wf01.y, w2 = wf23.x, w3 = wf23.y;

  Bf16* dst;
  int dst_stride;
  int dst_col;
  if (c < wq) {
    dst = q;
    dst_stride = wq;
    dst_col = c;
  } else if (c < wq + wk) {
    dst = k;
    dst_stride = wk;
    dst_col = c - wq;
  } else {
    dst = v;
    dst_stride = wv;
    dst_col = c - wq - wk;
  }

  Bf16* st = state + static_cast<size_t>(spans.slot[s]) * seq_stride;
  float h0 = bf(st[c]);
  float h1 = bf(st[channels + c]);
  float h2 = bf(st[2 * channels + c]);
  const int off = spans.off[s];
  for (int i = 0; i < len; ++i) {
    const int t = off + i;
    const float xt = bf(x[static_cast<size_t>(t) * channels + c]);
    const float acc = bf(to_bf(w0 * h0 + w1 * h1 + w2 * h2 + w3 * xt));
    dst[static_cast<size_t>(t) * dst_stride + dst_col] =
        to_bf(acc / (1.0f + expf(-acc)));
    h0 = h1;
    h1 = h2;
    h2 = xt;
  }
  st[c] = to_bf(h0);
  st[channels + c] = to_bf(h1);
  st[2 * channels + c] = to_bf(h2);
}

__global__ void beta_decay_kernel(float* beta, float* g, const Bf16* b, const Bf16* a,
                                  const float* A_log, const float* dt_bias, int m,
                                  int heads) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= m * heads) return;
  const int h = idx % heads;
  const float bv = bf(b[idx]);
  beta[idx] = bf(to_bf(1.0f / (1.0f + expf(-bv))));
  const float x = bf(a[idx]) + dt_bias[h];
  const float sp = x > 20.0f ? x : log1pf(expf(x));
  g[idx] = -expf(A_log[h]) * sp;
}

__global__ void l2norm_qk_kernel(const Bf16* q, const Bf16* k, float* qn, float* kn,
                                 int rows, int dim, float scale) {
  const int row = blockIdx.x;
  const Bf16* qr = q + static_cast<size_t>(row) * dim;
  const Bf16* kr = k + static_cast<size_t>(row) * dim;
  __shared__ float warp_sums[32];
  __shared__ float s_iq, s_ik;
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int nwarps = blockDim.x >> 5;
  float sq = 0.0f, sk = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float qv = bf(qr[i]), kv = bf(kr[i]);
    sq += qv * qv;
    sk += kv * kv;
  }
  for (int off = 16; off > 0; off >>= 1) {
    sq += __shfl_down_sync(0xffffffffu, sq, off);
    sk += __shfl_down_sync(0xffffffffu, sk, off);
  }
  if (lane == 0) {
    warp_sums[wid] = sq;
    warp_sums[16 + wid] = sk;
  }
  __syncthreads();
  if (wid == 0) {
    sq = lane < nwarps ? warp_sums[lane] : 0.0f;
    sk = lane < nwarps ? warp_sums[16 + lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      sq += __shfl_down_sync(0xffffffffu, sq, off);
      sk += __shfl_down_sync(0xffffffffu, sk, off);
    }
    if (lane == 0) {
      s_iq = rsqrtf(sq + 1e-6f) * scale;
      s_ik = rsqrtf(sk + 1e-6f);
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    qn[static_cast<size_t>(row) * dim + i] = bf(qr[i]) * s_iq;
    kn[static_cast<size_t>(row) * dim + i] = bf(kr[i]) * s_ik;
  }
}

// Fused l2norm_qk + beta_and_decay in one launch.
// Blocks [0, qk_blocks) compute L2-norm of q & k (keeping qv/kv in registers
// when dim == 128); blocks [qk_blocks, ...) compute beta and decay.
__global__ void __launch_bounds__(128) l2norm_qk_beta_decay_kernel(
    const Bf16* __restrict__ q, const Bf16* __restrict__ k,
    float* __restrict__ qn, float* __restrict__ kn, float* __restrict__ beta,
    float* __restrict__ g, const Bf16* __restrict__ b, const Bf16* __restrict__ a,
    const float* __restrict__ A_log, const float* __restrict__ dt_bias,
    int qk_blocks, int bd_elems, int v_heads, int dim, float scale) {
  const int blk = blockIdx.x;
  if (blk >= qk_blocks) {
    const int idx = (blk - qk_blocks) * 128 + threadIdx.x;
    if (idx >= bd_elems) return;
    const int h = idx % v_heads;
    const float bv = bf(b[idx]);
    beta[idx] = bf(to_bf(1.0f / (1.0f + expf(-bv))));
    const float x = bf(a[idx]) + dt_bias[h];
    const float sp = x > 20.0f ? x : log1pf(expf(x));
    g[idx] = -expf(A_log[h]) * sp;
    return;
  }

  const int row = blk;
  const Bf16* qr = q + static_cast<size_t>(row) * dim;
  const Bf16* kr = k + static_cast<size_t>(row) * dim;
  __shared__ float warp_sums[8];
  __shared__ float s_iq, s_ik;
  const int tid = threadIdx.x;
  const int lane = tid & 31, wid = tid >> 5;

  float sq = 0.0f, sk = 0.0f;
  float q_reg = 0.0f, k_reg = 0.0f;
  if (dim == 128) {
    q_reg = bf(qr[tid]);
    k_reg = bf(kr[tid]);
    sq = q_reg * q_reg;
    sk = k_reg * k_reg;
  } else {
    for (int i = tid; i < dim; i += 128) {
      const float qv = bf(qr[i]), kv = bf(kr[i]);
      sq += qv * qv;
      sk += kv * kv;
    }
  }
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    sq += __shfl_down_sync(0xffffffffu, sq, off);
    sk += __shfl_down_sync(0xffffffffu, sk, off);
  }
  if (lane == 0) {
    warp_sums[wid] = sq;
    warp_sums[4 + wid] = sk;
  }
  __syncthreads();
  if (wid == 0) {
    sq = lane < 4 ? warp_sums[lane] : 0.0f;
    sk = lane < 4 ? warp_sums[4 + lane] : 0.0f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      sq += __shfl_down_sync(0xffffffffu, sq, off);
      sk += __shfl_down_sync(0xffffffffu, sk, off);
    }
    if (lane == 0) {
      s_iq = rsqrtf(sq + 1e-6f) * scale;
      s_ik = rsqrtf(sk + 1e-6f);
    }
  }
  __syncthreads();
  if (dim == 128) {
    qn[static_cast<size_t>(row) * 128 + tid] = q_reg * s_iq;
    kn[static_cast<size_t>(row) * 128 + tid] = k_reg * s_ik;
  } else {
    for (int i = tid; i < dim; i += 128) {
      qn[static_cast<size_t>(row) * dim + i] = bf(qr[i]) * s_iq;
      kn[static_cast<size_t>(row) * dim + i] = bf(kr[i]) * s_ik;
    }
  }
}

// Fast 1-warp (32-column) gated delta rule for dim == 128.
// Splits each value head across 4 blocks (192 blocks total for nseq=1, 192*nseq
// for batched decode) so all 132 SM90 SMs stay busy. Uses 17 KiB static shared
// memory with zero bank conflicts, float4 qn/kn loads, fused state-update +
// output accumulation, and direct HBM write on single-token decode (len == 1).
// Bitwise identical to gated_delta_rule_kernel.
template <bool kSeg>
__global__ void __launch_bounds__(32) gated_delta_rule_128_kernel(
    float* __restrict__ state, const float* __restrict__ qn,
    const float* __restrict__ kn, const Bf16* __restrict__ v,
    const float* __restrict__ beta, const float* __restrict__ g,
    Bf16* __restrict__ out, int m, SeqSpans spans, int k_heads, int v_heads,
    size_t seq_stride) {
  constexpr int kDim = 128;
  constexpr int kCols = 32;
  const int hv = blockIdx.x >> 2;
  const int col_tile = blockIdx.x & 3;
  const int lane = threadIdx.x;  // 0..31
  const int vid = col_tile * kCols + lane;
  const int kh = hv / (v_heads / k_heads);

  int off = 0;
  int len = m;
  float* sbase = state + static_cast<size_t>(hv) * kDim * kDim;
  if constexpr (kSeg) {
    const int s = blockIdx.y;
    off = spans.off[s];
    len = spans.len[s];
    sbase = state + static_cast<size_t>(spans.slot[s]) * seq_stride +
            static_cast<size_t>(hv) * kDim * kDim;
  }
  if (len <= 0) return;

  __shared__ float S[kDim][kCols];           // 16 KiB, bank(S[k][lane]) == lane
  alignas(16) __shared__ float s_kn[kDim];   // 512 B
  alignas(16) __shared__ float s_qn[kDim];   // 512 B

  if (len == 1) {
    const int t = off;
    const size_t qrow = (static_cast<size_t>(t) * k_heads + kh) * kDim;
    reinterpret_cast<float4*>(s_kn)[lane] =
        __ldg(reinterpret_cast<const float4*>(kn + qrow) + lane);
    reinterpret_cast<float4*>(s_qn)[lane] =
        __ldg(reinterpret_cast<const float4*>(qn + qrow) + lane);
    __syncwarp(0xffffffffu);

    const float decay = expf(g[static_cast<size_t>(t) * v_heads + hv]);
    const float bt = beta[static_cast<size_t>(t) * v_heads + hv];
    float kv_mem = 0.0f;
    #pragma unroll 4
    for (int k = 0; k < kDim; ++k) {
      const float sv = sbase[k * kDim + vid] * decay;
      S[k][lane] = sv;
      kv_mem += sv * s_kn[k];
    }
    const float delta =
        (bf(v[(static_cast<size_t>(t) * v_heads + hv) * kDim + vid]) - kv_mem) * bt;
    float acc = 0.0f;
    #pragma unroll 4
    for (int k = 0; k < kDim; ++k) {
      const float s_new = S[k][lane] + s_kn[k] * delta;
      sbase[k * kDim + vid] = s_new;
      acc += s_new * s_qn[k];
    }
    out[(static_cast<size_t>(t) * v_heads + hv) * kDim + vid] = to_bf(acc);
    return;
  }

  #pragma unroll 4
  for (int k = 0; k < kDim; ++k) S[k][lane] = sbase[k * kDim + vid];

  for (int i = 0; i < len; ++i) {
    const int t = off + i;
    const size_t qrow = (static_cast<size_t>(t) * k_heads + kh) * kDim;
    reinterpret_cast<float4*>(s_kn)[lane] =
        __ldg(reinterpret_cast<const float4*>(kn + qrow) + lane);
    reinterpret_cast<float4*>(s_qn)[lane] =
        __ldg(reinterpret_cast<const float4*>(qn + qrow) + lane);
    __syncwarp(0xffffffffu);

    const float decay = expf(g[static_cast<size_t>(t) * v_heads + hv]);
    const float bt = beta[static_cast<size_t>(t) * v_heads + hv];
    float kv_mem = 0.0f;
    #pragma unroll 4
    for (int k = 0; k < kDim; ++k) {
      const float sv = S[k][lane] * decay;
      S[k][lane] = sv;
      kv_mem += sv * s_kn[k];
    }
    const float delta =
        (bf(v[(static_cast<size_t>(t) * v_heads + hv) * kDim + vid]) - kv_mem) * bt;
    float acc = 0.0f;
    #pragma unroll 4
    for (int k = 0; k < kDim; ++k) {
      const float s_new = S[k][lane] + s_kn[k] * delta;
      S[k][lane] = s_new;
      acc += s_new * s_qn[k];
    }
    out[(static_cast<size_t>(t) * v_heads + hv) * kDim + vid] = to_bf(acc);
    __syncwarp(0xffffffffu);
  }

  #pragma unroll 4
  for (int k = 0; k < kDim; ++k) sbase[k * kDim + vid] = S[k][lane];
}

// Generic fallback gated delta rule for dim != 128.
__global__ void gated_delta_rule_kernel(float* state, const float* qn,
                                        const float* kn, const Bf16* v,
                                        const float* beta, const float* g, Bf16* out,
                                        int m, int k_heads, int v_heads, int dim) {
  extern __shared__ float sm[];
  const int stride = dim + 1;
  const int hv = blockIdx.x;
  const int vid = threadIdx.x;
  const int group = v_heads / k_heads;
  const int kh = hv / group;
  float* S = sm;
  for (int k = 0; k < dim; ++k)
    S[k * stride + vid] = state[static_cast<size_t>(hv) * dim * dim + k * dim + vid];
  __syncthreads();

  for (int t = 0; t < m; ++t) {
    const float decay = expf(g[t * v_heads + hv]);
    const float bt = beta[t * v_heads + hv];
    const size_t qrow = (static_cast<size_t>(t) * k_heads + kh) * dim;
    float kv_mem = 0.0f;
    for (int k = 0; k < dim; ++k) {
      const float s = S[k * stride + vid] * decay;
      S[k * stride + vid] = s;
      kv_mem += s * kn[qrow + k];
    }
    const float delta =
        (bf(v[(static_cast<size_t>(t) * v_heads + hv) * dim + vid]) - kv_mem) * bt;
    float acc = 0.0f;
    for (int k = 0; k < dim; ++k) {
      const float s_new = S[k * stride + vid] + kn[qrow + k] * delta;
      S[k * stride + vid] = s_new;
      acc += s_new * qn[qrow + k];
    }
    out[(static_cast<size_t>(t) * v_heads + hv) * dim + vid] = to_bf(acc);
  }

  for (int k = 0; k < dim; ++k)
    state[static_cast<size_t>(hv) * dim * dim + k * dim + vid] = S[k * stride + vid];
}

__device__ __forceinline__ float gated_rms_elem(float xv, float wv, float zv,
                                                float inv) {
  const float wb_val = bf(to_bf(wv * bf(to_bf(xv * inv))));
  const float silu = zv / (1.0f + expf(-zv));
  return wb_val * silu;
}

// Single-warp vectorized gated RMSNorm for dim == 128 (4 Bf16 / lane via uint2).
// Zero shared memory, zero __syncthreads(), exact reference bf16 rounding.
__global__ void __launch_bounds__(32) gated_rms_norm_128_kernel(
    const Bf16* __restrict__ x, const Bf16* __restrict__ z,
    const Bf16* __restrict__ weight, Bf16* __restrict__ out, float eps) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  const uint2 xraw = __ldg(reinterpret_cast<const uint2*>(x + static_cast<size_t>(row) * 128) + lane);
  const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&xraw);
  const float2 xf0 = __bfloat1622float2(xb[0]);
  const float2 xf1 = __bfloat1622float2(xb[1]);
  float sum = xf0.x * xf0.x + xf0.y * xf0.y + xf1.x * xf1.x + xf1.y * xf1.y;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    sum += __shfl_xor_sync(0xffffffffu, sum, off);
  const float inv = rsqrtf(sum / 128.0f + eps);

  const uint2 zraw = __ldg(reinterpret_cast<const uint2*>(z + static_cast<size_t>(row) * 128) + lane);
  const uint2 wraw = __ldg(reinterpret_cast<const uint2*>(weight) + lane);
  const __nv_bfloat162* zb = reinterpret_cast<const __nv_bfloat162*>(&zraw);
  const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wraw);
  const float2 zf0 = __bfloat1622float2(zb[0]);
  const float2 zf1 = __bfloat1622float2(zb[1]);
  const float2 wf0 = __bfloat1622float2(wb[0]);
  const float2 wf1 = __bfloat1622float2(wb[1]);

  uint2 oraw;
  __nv_bfloat162* ob = reinterpret_cast<__nv_bfloat162*>(&oraw);
  ob[0] = __floats2bfloat162_rn(gated_rms_elem(xf0.x, wf0.x, zf0.x, inv),
                                gated_rms_elem(xf0.y, wf0.y, zf0.y, inv));
  ob[1] = __floats2bfloat162_rn(gated_rms_elem(xf1.x, wf1.x, zf1.x, inv),
                                gated_rms_elem(xf1.y, wf1.y, zf1.y, inv));
  reinterpret_cast<uint2*>(out + static_cast<size_t>(row) * 128)[lane] = oraw;
}

__global__ void gated_rms_norm_kernel(const Bf16* x, const Bf16* z,
                                      const Bf16* weight, Bf16* out, int dim,
                                      float eps) {
  const int row = blockIdx.x;
  const Bf16* xr = x + static_cast<size_t>(row) * dim;
  const Bf16* zr = z + static_cast<size_t>(row) * dim;
  Bf16* orow = out + static_cast<size_t>(row) * dim;
  __shared__ float warp_sums[32];
  __shared__ float s_inv;
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int nwarps = blockDim.x >> 5;
  float sum = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = bf(xr[i]);
    sum += v * v;
  }
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
  if (lane == 0) warp_sums[wid] = sum;
  __syncthreads();
  if (wid == 0) {
    sum = lane < nwarps ? warp_sums[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) s_inv = rsqrtf(sum / dim + eps);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float normed = bf(xr[i]) * s_inv;
    const float wb = bf(to_bf(bf(weight[i]) * bf(to_bf(normed))));
    const float zv = bf(zr[i]);
    const float silu = zv / (1.0f + expf(-zv));
    orow[i] = to_bf(wb * silu);
  }
}

__device__ __forceinline__ void argmax_update(float v, int idx, float& best_v,
                                              int& best_i) {
  if (v > best_v || (v == best_v && idx < best_i)) {
    best_v = v;
    best_i = idx;
  }
}

__global__ void __launch_bounds__(1024) argmax_rows_kernel(
    const float* __restrict__ x, int n, int* __restrict__ best_idx) {
  const int row = blockIdx.x;
  const float* xr = x + static_cast<size_t>(row) * n;
  __shared__ int s_idx[32];
  __shared__ float wv[32];
  float local_val = -INFINITY;
  int local_idx = 0;

  if ((n & 3) == 0 && (reinterpret_cast<uintptr_t>(xr) & 15u) == 0u) {
    const int n4 = n >> 2;
    const float4* x4 = reinterpret_cast<const float4*>(xr);
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
      const float4 v = __ldg(x4 + i);
      const int base = i << 2;
      argmax_update(v.x, base + 0, local_val, local_idx);
      argmax_update(v.y, base + 1, local_val, local_idx);
      argmax_update(v.z, base + 2, local_val, local_idx);
      argmax_update(v.w, base + 3, local_val, local_idx);
    }
  } else {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      argmax_update(xr[i], i, local_val, local_idx);
    }
  }

  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float ov = __shfl_down_sync(0xffffffffu, local_val, off);
    const int oi = __shfl_down_sync(0xffffffffu, local_idx, off);
    argmax_update(ov, oi, local_val, local_idx);
  }
  if (lane == 0) {
    wv[wid] = local_val;
    s_idx[wid] = local_idx;
  }
  __syncthreads();
  if (wid == 0) {
    const int nw = (blockDim.x + 31) / 32;
    if (lane < nw) {
      local_val = wv[lane];
      local_idx = s_idx[lane];
    } else {
      local_val = -INFINITY;
      local_idx = n;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_down_sync(0xffffffffu, local_val, off);
      const int oi = __shfl_down_sync(0xffffffffu, local_idx, off);
      argmax_update(ov, oi, local_val, local_idx);
    }
    if (lane == 0) best_idx[row] = local_idx;
  }
}

// ---- Continuous batching --------------------------------------------------
__global__ void gather_rows_kernel(Bf16* dst, const Bf16* src, const int* rows,
                                   int m, int dim) {
  const int s = blockIdx.y;
  if (s >= m) return;
  const Bf16* sr = src + static_cast<size_t>(rows[s]) * dim;
  Bf16* dr = dst + static_cast<size_t>(s) * dim;
  if ((dim & 7) == 0 && (reinterpret_cast<uintptr_t>(sr) & 15u) == 0u &&
      (reinterpret_cast<uintptr_t>(dr) & 15u) == 0u) {
    const int vecs = dim >> 3;
    const uint4* sv = reinterpret_cast<const uint4*>(sr);
    uint4* dv = reinterpret_cast<uint4*>(dr);
    for (int i = threadIdx.x; i < vecs; i += blockDim.x) dv[i] = __ldg(sv + i);
  } else {
    for (int i = threadIdx.x; i < dim; i += blockDim.x) dr[i] = sr[i];
  }
}

__global__ void conv1d_causal_silu_seg_kernel(const Bf16* x, const Bf16* w,
                                              Bf16* state, Bf16* out,
                                              SeqSpans spans, int nseq,
                                              int channels, size_t seq_stride) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) return;
  const float w0 = bf(w[c * 4 + 0]), w1 = bf(w[c * 4 + 1]);
  const float w2 = bf(w[c * 4 + 2]), w3 = bf(w[c * 4 + 3]);
  for (int s = 0; s < nseq; ++s) {
    Bf16* st = state + static_cast<size_t>(spans.slot[s]) * seq_stride;
    float h0 = bf(st[c]);
    float h1 = bf(st[channels + c]);
    float h2 = bf(st[2 * channels + c]);
    for (int i = 0; i < spans.len[s]; ++i) {
      const int t = spans.off[s] + i;
      const float xt = bf(x[static_cast<size_t>(t) * channels + c]);
      const float acc = bf(to_bf(w0 * h0 + w1 * h1 + w2 * h2 + w3 * xt));
      out[static_cast<size_t>(t) * channels + c] =
          to_bf(acc / (1.0f + expf(-acc)));
      h0 = h1;
      h1 = h2;
      h2 = xt;
    }
    st[c] = to_bf(h0);
    st[channels + c] = to_bf(h1);
    st[2 * channels + c] = to_bf(h2);
  }
}

__global__ void gated_delta_rule_seg_kernel(float* state, const float* qn,
                                            const float* kn, const Bf16* v,
                                            const float* beta, const float* g,
                                            Bf16* out, SeqSpans spans, int nseq,
                                            int k_heads, int v_heads, int dim,
                                            size_t seq_stride) {
  extern __shared__ float sm[];
  const int stride = dim + 1;
  const int hv = blockIdx.x;
  const int vid = threadIdx.x;
  const int group = v_heads / k_heads;
  const int kh = hv / group;
  float* S = sm;
  for (int s = 0; s < nseq; ++s) {
    float* sbase = state + static_cast<size_t>(spans.slot[s]) * seq_stride +
                   static_cast<size_t>(hv) * dim * dim;
    for (int k = 0; k < dim; ++k) S[k * stride + vid] = sbase[k * dim + vid];
    __syncthreads();
    for (int i = 0; i < spans.len[s]; ++i) {
      const int t = spans.off[s] + i;
      const float decay = expf(g[t * v_heads + hv]);
      const float bt = beta[t * v_heads + hv];
      const size_t qrow = (static_cast<size_t>(t) * k_heads + kh) * dim;
      float kv_mem = 0.0f;
      for (int k = 0; k < dim; ++k) {
        const float sv = S[k * stride + vid] * decay;
        S[k * stride + vid] = sv;
        kv_mem += sv * kn[qrow + k];
      }
      const float delta =
          (bf(v[(static_cast<size_t>(t) * v_heads + hv) * dim + vid]) - kv_mem) * bt;
      float acc = 0.0f;
      for (int k = 0; k < dim; ++k) {
        const float s_new = S[k * stride + vid] + kn[qrow + k] * delta;
        S[k * stride + vid] = s_new;
        acc += s_new * qn[qrow + k];
      }
      out[(static_cast<size_t>(t) * v_heads + hv) * dim + vid] = to_bf(acc);
    }
    for (int k = 0; k < dim; ++k) sbase[k * dim + vid] = S[k * stride + vid];
    __syncthreads();
  }
}

__global__ void attention_seg_kernel(const Bf16* __restrict__ q,
                                     const Bf16* __restrict__ kbase,
                                     const Bf16* __restrict__ vbase,
                                     const Bf16* __restrict__ gate,
                                     Bf16* __restrict__ out, SeqSpans spans,
                                     int nseq, size_t seq_stride, int n_q_heads,
                                     int n_kv_heads, int head_dim) {
  extern __shared__ float smem[];
  float* qs = smem;
  float* acc = smem + head_dim;
  float* reduce = smem + 2 * head_dim;
  __shared__ float s_max, s_sum, s_w;
  const int row = blockIdx.y;
  const int qh = blockIdx.x;
  const int tid = threadIdx.x;
  int s = 0;
  while (s + 1 < nseq && row >= spans.off[s] + spans.len[s]) ++s;
  const int hi = spans.hist[s] + (row - spans.off[s]) + 1;
  const bf16* k = reinterpret_cast<const bf16*>(kbase) +
                  static_cast<size_t>(spans.slot[s]) * seq_stride;
  const bf16* v = reinterpret_cast<const bf16*>(vbase) +
                  static_cast<size_t>(spans.slot[s]) * seq_stride;
  const int kvh = qh * n_kv_heads / n_q_heads;
  const Bf16* qrow = q + (static_cast<size_t>(row) * n_q_heads + qh) * head_dim;
  for (int i = tid; i < head_dim; i += blockDim.x) qs[i] = bf(qrow[i]);
  __syncthreads();
  const float scale = rsqrtf(static_cast<float>(head_dim));

  if (tid == 0) s_max = -INFINITY;
  __syncthreads();
  for (int kj = 0; kj < hi; ++kj) {
    const bf16* krow =
        k + (static_cast<size_t>(kj) * n_kv_heads + kvh) * head_dim;
    const float dot = block_dot(qs, krow, head_dim, reduce) * scale;
    if (tid == 0) s_max = fmaxf(s_max, dot);
    __syncthreads();
  }
  const float mx = s_max;

  if (tid == 0) s_sum = 0.0f;
  for (int i = tid; i < head_dim; i += blockDim.x) acc[i] = 0.0f;
  __syncthreads();
  for (int kj = 0; kj < hi; ++kj) {
    const bf16* krow =
        k + (static_cast<size_t>(kj) * n_kv_heads + kvh) * head_dim;
    const float weight = expf(block_dot(qs, krow, head_dim, reduce) * scale - mx);
    if (tid == 0) {
      s_w = weight;
      s_sum += weight;
    }
    __syncthreads();
    const bf16* vrow =
        v + (static_cast<size_t>(kj) * n_kv_heads + kvh) * head_dim;
    const float wgt = s_w;
    for (int i = tid; i < head_dim; i += blockDim.x)
      acc[i] += wgt * __bfloat162float(vrow[i]);
    __syncthreads();
  }
  const float inv = 1.0f / s_sum;
  Bf16* orow = out + (static_cast<size_t>(row) * n_q_heads + qh) * head_dim;
  const Bf16* grow =
      gate + (static_cast<size_t>(row) * n_q_heads + qh) * head_dim;
  for (int i = tid; i < head_dim; i += blockDim.x) {
    float o = bf(to_bf(acc[i] * inv));
    const float gv = bf(grow[i]);
    const float sg = 1.0f / (1.0f + expf(-gv));
    o *= bf(to_bf(sg));
    orow[i] = to_bf(o);
  }
}

bool g_dgr_configured = false;
void configure_dgr_smem(int dim) {
  if (g_dgr_configured) return;
  const size_t bytes = static_cast<size_t>(dim) * (dim + 1) * sizeof(float);
  cudaError_t e = cudaFuncSetAttribute(
      gated_delta_rule_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(bytes));
  if (e == cudaSuccess)
    e = cudaFuncSetAttribute(
        gated_delta_rule_seg_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(bytes));
  if (e != cudaSuccess)
    throw std::runtime_error(std::string("cudaFuncSetAttribute: ") + cudaGetErrorString(e));
  g_dgr_configured = true;
}

template <bool kZeroCentered>
void launch_rms_norm(const Bf16* x, const Bf16* weight, Bf16* out, int rows,
                     int dim, float eps, cudaStream_t stream) {
  if (rows <= 0 || dim <= 0) return;
  if ((dim & 7) == 0 && aligned16(x) && aligned16(weight) && aligned16(out)) {
    if (dim <= 256) {
      rms_norm_warp_vec_kernel<kZeroCentered>
          <<<rows, 32, 0, stream>>>(x, weight, out, dim, eps);
    } else {
      rms_norm_vec_kernel<kZeroCentered>
          <<<rows, 256, 0, stream>>>(x, weight, out, dim, eps);
    }
    return;
  }
  rms_norm_kernel<kZeroCentered><<<rows, 256, 0, stream>>>(x, weight, out, dim, eps);
}

}  // namespace

// ---- host entry points ----------------------------------------------------
void rms_norm(const Bf16* x, const Bf16* weight, Bf16* out, int rows, int dim,
              float eps, cudaStream_t stream) {
  launch_rms_norm<false>(x, weight, out, rows, dim, eps, stream);
}

void rms_norm_zero_centered(const Bf16* x, const Bf16* weight, Bf16* out, int rows,
                            int dim, float eps, cudaStream_t stream) {
  launch_rms_norm<true>(x, weight, out, rows, dim, eps, stream);
}

void add_residual_rms_norm_zero_centered(Bf16* hidden, const Bf16* residual,
                                         const Bf16* weight, Bf16* norm_out,
                                         int rows, int dim, float eps,
                                         cudaStream_t stream) {
  if (rows <= 0 || dim <= 0) return;
  if ((dim & 7) == 0 && aligned16(hidden) && aligned16(residual) &&
      aligned16(weight) && aligned16(norm_out)) {
    add_residual_rms_norm_vec_kernel<<<rows, 256, 0, stream>>>(
        hidden, residual, weight, norm_out, dim, eps);
    return;
  }
  add_residual_rms_norm_scalar_kernel<<<rows, 256, 0, stream>>>(
      hidden, residual, weight, norm_out, dim, eps);
}

void layer_norm(const Bf16* x, const Bf16* weight, const Bf16* bias, Bf16* out,
                int rows, int dim, float eps, cudaStream_t stream) {
  layer_norm_kernel<<<rows, 256, 0, stream>>>(x, weight, bias, out, dim, eps);
}

void add_bias(Bf16* y, const Bf16* bias, int rows, int dim, cudaStream_t stream) {
  bias_kernel<<<rows, 256, 0, stream>>>(y, bias, rows, dim);
}

void add_residual(Bf16* y, const Bf16* residual, int n, cudaStream_t stream) {
  if (n <= 0) return;
  if ((n & 7) == 0 && aligned16(y) && aligned16(residual)) {
    const int vecs = n >> 3;
    residual_vec_kernel<<<(vecs + 255) / 256, 256, 0, stream>>>(y, residual, vecs);
    return;
  }
  residual_kernel<<<(n + 255) / 256, 256, 0, stream>>>(y, residual, n);
}

void add_residual_tiled(Bf16* y, const Bf16* residual, int total_n, int tile_n,
                        cudaStream_t stream) {
  if (total_n <= 0 || tile_n <= 0) return;
  if (total_n <= tile_n) {
    add_residual(y, residual, total_n, stream);
    return;
  }
  residual_tiled_kernel<<<(total_n + 255) / 256, 256, 0, stream>>>(
      y, residual, total_n, tile_n);
}

void silu_mul(Bf16* out, const Bf16* gate, const Bf16* up, int n, cudaStream_t stream) {
  if (n <= 0) return;
  if ((n & 7) == 0 && aligned16(out) && aligned16(gate) && aligned16(up)) {
    const int vecs = n >> 3;
    silu_mul_vec_kernel<<<(vecs + 255) / 256, 256, 0, stream>>>(out, gate, up, vecs);
    return;
  }
  silu_mul_kernel<<<(n + 255) / 256, 256, 0, stream>>>(out, gate, up, n);
}

void gelu_tanh_inplace(Bf16* x, int n, cudaStream_t stream) {
  elementwise_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, n, GeluTanhOp{});
}

void gelu_erf_inplace(Bf16* x, int n, cudaStream_t stream) {
  elementwise_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, n, GeluErfOp{});
}

void f32_to_bf16(const float* src, Bf16* dst, int n, cudaStream_t stream) {
  f32_to_bf16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(src, dst, n);
}

void bf16_to_f32(const Bf16* src, float* dst, int n, cudaStream_t stream) {
  bf16_to_f32_kernel<<<(n + 255) / 256, 256, 0, stream>>>(src, dst, n);
}

void index_embeddings(const Bf16* table, const int* ids, Bf16* out, int m, int dim,
                      cudaStream_t stream) {
  dim3 grid(1, m);
  embed_kernel<<<grid, 256, 0, stream>>>(table, ids, out, m, dim);
}

void replace_rows(Bf16* dst, const Bf16* src, int start, int n, int dim,
                  cudaStream_t stream) {
  dim3 grid(1, n);
  replace_rows_kernel<<<grid, 256, 0, stream>>>(dst, src, start, dim);
}

void mrope_partial(Bf16* x, const Bf16* cos, const Bf16* sin, int n, int heads,
                   int head_dim, int rotary_dim, cudaStream_t stream) {
  dim3 grid(1, n);
  dim3 block(64, 4);
  mrope_partial_kernel<<<grid, block, 0, stream>>>(x, cos, sin, n, heads, head_dim,
                                                   rotary_dim);
}

void prep_qkv_rope_kvcache(const Bf16* qg, const Bf16* k_in, const Bf16* v_in,
                           const Bf16* q_norm_w, const Bf16* k_norm_w,
                           const Bf16* cos, const Bf16* sin, Bf16* q_out,
                           Bf16* gate_out, Bf16* k_cache, Bf16* v_cache, int m,
                           int history, int n_q_heads, int n_kv_heads,
                           int head_dim, int rotary_dim, float eps,
                           cudaStream_t stream) {
  if (m <= 0) return;
  if (head_dim == 256 && rotary_dim <= 256) {
    dim3 grid(n_q_heads + 2 * n_kv_heads, m);
    SeqSpans empty{};
    prep_qkv_rope_kvcache_256_kernel<false><<<grid, 256, 0, stream>>>(
        qg, k_in, v_in, q_norm_w, k_norm_w, cos, sin, q_out, gate_out, k_cache,
        v_cache, history, empty, 1, 0, n_q_heads, n_kv_heads, rotary_dim, eps);
    return;
  }
  split_q_gate(qg, q_out, gate_out, m, n_q_heads, head_dim, stream);
  rms_norm_zero_centered(q_out, q_norm_w, q_out, m * n_q_heads, head_dim, eps, stream);
  Bf16* k_dst = k_cache + static_cast<size_t>(history) * n_kv_heads * head_dim;
  Bf16* v_dst = v_cache + static_cast<size_t>(history) * n_kv_heads * head_dim;
  rms_norm_zero_centered(k_in, k_norm_w, k_dst, m * n_kv_heads, head_dim, eps, stream);
  mrope_partial(q_out, cos, sin, m, n_q_heads, head_dim, rotary_dim, stream);
  mrope_partial(k_dst, cos, sin, m, n_kv_heads, head_dim, rotary_dim, stream);
  UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(
      v_dst, v_in, static_cast<size_t>(m) * n_kv_heads * head_dim * sizeof(Bf16),
      cudaMemcpyDeviceToDevice, stream));
}

void prep_qkv_rope_kvcache_seg(const Bf16* qg, const Bf16* k_in,
                               const Bf16* v_in, const Bf16* q_norm_w,
                               const Bf16* k_norm_w, const Bf16* cos,
                               const Bf16* sin, Bf16* q_out, Bf16* gate_out,
                               Bf16* k_cache, Bf16* v_cache,
                               const SeqSpans& spans, int nseq, int rows,
                               size_t seq_stride, int n_q_heads, int n_kv_heads,
                               int head_dim, int rotary_dim, float eps,
                               cudaStream_t stream) {
  if (rows <= 0 || nseq <= 0) return;
  if (head_dim == 256 && rotary_dim <= 256) {
    dim3 grid(n_q_heads + 2 * n_kv_heads, rows);
    prep_qkv_rope_kvcache_256_kernel<true><<<grid, 256, 0, stream>>>(
        qg, k_in, v_in, q_norm_w, k_norm_w, cos, sin, q_out, gate_out, k_cache,
        v_cache, 0, spans, nseq, seq_stride, n_q_heads, n_kv_heads, rotary_dim,
        eps);
    return;
  }
  throw std::runtime_error("prep_qkv_rope_kvcache_seg requires head_dim == 256");
}

void vision_rope(Bf16* x, const float* cos, const float* sin, int n, int heads,
                 int head_dim, cudaStream_t stream) {
  vision_rope_tiled(x, cos, sin, n, n, heads, head_dim, stream);
}

void vision_rope_tiled(Bf16* x, const float* cos, const float* sin, int n,
                       int slice_tokens, int heads, int head_dim,
                       cudaStream_t stream) {
  if (n <= 0) return;
  dim3 grid(1, n);
  dim3 block(64, 4);
  vision_rope_kernel<<<grid, block, 0, stream>>>(x, cos, sin, n, slice_tokens,
                                                 heads, head_dim);
}

void split_q_gate(const Bf16* qg, Bf16* q, Bf16* gate, int m, int heads, int head_dim,
                  cudaStream_t stream) {
  const int total = m * heads * head_dim;
  split_q_gate_kernel<<<(total + 255) / 256, 256, 0, stream>>>(qg, q, gate, m, heads,
                                                               head_dim);
}

void split_qkv(const Bf16* qkv, Bf16* q, Bf16* k, Bf16* v, int m, int width,
               cudaStream_t stream) {
  dim3 grid((width + 255) / 256, m);
  split_qkv_kernel<<<grid, 256, 0, stream>>>(qkv, q, k, v, m, width);
}

void gqa_attention(const Bf16* q, const Bf16* k, const Bf16* v, const Bf16* gate,
                   Bf16* out, int m, int nkv, int n_q_heads, int n_kv_heads,
                   int head_dim, int q_offset, cudaStream_t stream) {
  if (m <= 0) return;
  dim3 grid(n_q_heads, m);
  if (head_dim == 256 && aligned16(q) && aligned16(k) && aligned16(v) &&
      aligned16(out) && (gate == nullptr || aligned16(gate))) {
    if (gate != nullptr) {
      gqa_attention_256_kernel<true><<<grid, 256, 0, stream>>>(
          q, k, v, gate, out, nkv, n_q_heads, n_kv_heads, q_offset);
    } else {
      gqa_attention_256_kernel<false><<<grid, 256, 0, stream>>>(
          q, k, v, nullptr, out, nkv, n_q_heads, n_kv_heads, q_offset);
    }
    return;
  }
  const size_t smem = (2 * head_dim + 32) * sizeof(float);
  attention_kernel<true, true><<<grid, 128, smem, stream>>>(
      q, k, v, gate, out, m, nkv, n_q_heads, n_kv_heads, head_dim, q_offset);
}

void vision_attention(const Bf16* q, const Bf16* k, const Bf16* v, Bf16* out, int n,
                      int heads, int head_dim, cudaStream_t stream) {
  vision_attention_batched(q, k, v, out, n, 1, heads, head_dim, stream);
}

void vision_attention_batched(const Bf16* q, const Bf16* k, const Bf16* v,
                              Bf16* out, int slice_tokens, int num_slices,
                              int heads, int head_dim, cudaStream_t stream) {
  if (slice_tokens <= 0 || num_slices <= 0) return;
  dim3 grid(heads, slice_tokens, num_slices);
  if (head_dim == 72) {
    vision_attention_72_kernel<<<grid, 128, 0, stream>>>(
        q, k, v, out, slice_tokens, heads);
    return;
  }
  const size_t smem = (2 * head_dim + 32) * sizeof(float);
  attention_kernel<false, false><<<grid, 128, smem, stream>>>(
      q, k, v, nullptr, out, slice_tokens, slice_tokens, heads, heads, head_dim,
      0);
}

void split_qkv_widths(const Bf16* qkv, Bf16* q, Bf16* k, Bf16* v, int m, int wq,
                      int wk, int wv, cudaStream_t stream) {
  const int total = wq + wk + wv;
  dim3 grid((total + 255) / 256, m);
  split_qkv_widths_kernel<<<grid, 256, 0, stream>>>(qkv, q, k, v, m, wq, wk, wv);
}

void conv1d_causal_silu(const Bf16* x, const Bf16* w, Bf16* state, Bf16* out, int m,
                        int channels, cudaStream_t stream) {
  conv1d_causal_silu_kernel<<<(channels + 255) / 256, 256, 0, stream>>>(x, w, state,
                                                                       out, m, channels);
}

void conv1d_causal_silu_split(const Bf16* x, const Bf16* w, Bf16* state,
                              Bf16* q, Bf16* k, Bf16* v, int m, int wq, int wk,
                              int wv, cudaStream_t stream) {
  const int channels = wq + wk + wv;
  if (m <= 0 || channels <= 0) return;
  conv1d_causal_silu_split_kernel<<<(channels + 255) / 256, 256, 0, stream>>>(
      x, w, state, q, k, v, m, wq, wk, wv);
}

void conv1d_causal_silu_split_seg(const Bf16* x, const Bf16* w, Bf16* state,
                                  Bf16* q, Bf16* k, Bf16* v,
                                  const SeqSpans& spans, int nseq, int wq,
                                  int wk, int wv, size_t seq_stride,
                                  cudaStream_t stream) {
  const int channels = wq + wk + wv;
  if (nseq <= 0 || channels <= 0) return;
  dim3 grid((channels + 255) / 256, nseq);
  conv1d_causal_silu_split_seg_kernel<<<grid, 256, 0, stream>>>(
      x, w, state, q, k, v, spans, wq, wk, wv, seq_stride);
}

void beta_and_decay(float* beta, float* g, const Bf16* b, const Bf16* a,
                    const float* A_log, const float* dt_bias, int m, int heads,
                    cudaStream_t stream) {
  const int n = m * heads;
  beta_decay_kernel<<<(n + 255) / 256, 256, 0, stream>>>(beta, g, b, a, A_log,
                                                          dt_bias, m, heads);
}

void l2norm_qk(const Bf16* q, const Bf16* k, float* qn, float* kn, int m, int k_heads,
               int dim, cudaStream_t stream) {
  const int rows = m * k_heads;
  l2norm_qk_kernel<<<rows, 128, 0, stream>>>(q, k, qn, kn, rows, dim,
                                             rsqrtf(static_cast<float>(dim)));
}

void l2norm_qk_beta_decay(const Bf16* q, const Bf16* k, float* qn, float* kn,
                          float* beta, float* g, const Bf16* b, const Bf16* a,
                          const float* A_log, const float* dt_bias, int m,
                          int k_heads, int v_heads, int dim,
                          cudaStream_t stream) {
  if (m <= 0) return;
  const int qk_blocks = m * k_heads;
  const int bd_elems = m * v_heads;
  const int bd_blocks = (bd_elems + 127) / 128;
  l2norm_qk_beta_decay_kernel<<<qk_blocks + bd_blocks, 128, 0, stream>>>(
      q, k, qn, kn, beta, g, b, a, A_log, dt_bias, qk_blocks, bd_elems, v_heads,
      dim, rsqrtf(static_cast<float>(dim)));
}

void gated_delta_rule(float* state, const float* qn, const float* kn, const Bf16* v,
                      const float* beta, const float* g, Bf16* out, int m,
                      int k_heads, int v_heads, int dim, cudaStream_t stream) {
  if (m <= 0) return;
  if (dim == 128 && aligned16(qn) && aligned16(kn)) {
    SeqSpans empty{};
    gated_delta_rule_128_kernel<false><<<v_heads * 4, 32, 0, stream>>>(
        state, qn, kn, v, beta, g, out, m, empty, k_heads, v_heads, 0);
    return;
  }
  configure_dgr_smem(dim);
  const size_t smem = static_cast<size_t>(dim) * (dim + 1) * sizeof(float);
  gated_delta_rule_kernel<<<v_heads, dim, smem, stream>>>(
      state, qn, kn, v, beta, g, out, m, k_heads, v_heads, dim);
}

void gated_rms_norm(const Bf16* x, const Bf16* z, const Bf16* weight, Bf16* out,
                    int rows, int dim, float eps, cudaStream_t stream) {
  if (rows <= 0) return;
  if (dim == 128 && aligned8(x) && aligned8(z) && aligned8(weight) &&
      aligned8(out)) {
    gated_rms_norm_128_kernel<<<rows, 32, 0, stream>>>(x, z, weight, out, eps);
    return;
  }
  gated_rms_norm_kernel<<<rows, 128, 0, stream>>>(x, z, weight, out, dim, eps);
}

void gather_rows(Bf16* dst, const Bf16* src, const int* rows, int m, int dim,
                 cudaStream_t stream) {
  dim3 grid(1, m);
  gather_rows_kernel<<<grid, 256, 0, stream>>>(dst, src, rows, m, dim);
}

void argmax_rows(const float* logits, int rows, int n, int* idx,
                 cudaStream_t stream) {
  argmax_rows_kernel<<<rows, 1024, 0, stream>>>(logits, n, idx);
}

void conv1d_causal_silu_seg(const Bf16* x, const Bf16* w, Bf16* state, Bf16* out,
                            const SeqSpans& spans, int nseq, int channels,
                            size_t seq_stride, cudaStream_t stream) {
  conv1d_causal_silu_seg_kernel<<<(channels + 255) / 256, 256, 0, stream>>>(
      x, w, state, out, spans, nseq, channels, seq_stride);
}

void gated_delta_rule_seg(float* state, const float* qn, const float* kn,
                          const Bf16* v, const float* beta, const float* g,
                          Bf16* out, const SeqSpans& spans, int nseq, int k_heads,
                          int v_heads, int dim, size_t seq_stride,
                          cudaStream_t stream) {
  if (nseq <= 0) return;
  if (dim == 128 && aligned16(qn) && aligned16(kn)) {
    dim3 grid(v_heads * 4, nseq);
    gated_delta_rule_128_kernel<true><<<grid, 32, 0, stream>>>(
        state, qn, kn, v, beta, g, out, 0, spans, k_heads, v_heads, seq_stride);
    return;
  }
  configure_dgr_smem(dim);
  const size_t smem = static_cast<size_t>(dim) * (dim + 1) * sizeof(float);
  gated_delta_rule_seg_kernel<<<v_heads, dim, smem, stream>>>(
      state, qn, kn, v, beta, g, out, spans, nseq, k_heads, v_heads, dim,
      seq_stride);
}

void gqa_attention_seg(const Bf16* q, const Bf16* k_cache, const Bf16* v_cache,
                       const Bf16* gate, Bf16* out, const SeqSpans& spans,
                       int nseq, size_t seq_stride, int n_q_heads,
                       int n_kv_heads, int head_dim, cudaStream_t stream) {
  int rows = 0;
  for (int s = 0; s < nseq; ++s) {
    const int end = spans.off[s] + spans.len[s];
    rows = end > rows ? end : rows;
  }
  if (rows <= 0) return;
  dim3 grid(n_q_heads, rows);
  if (head_dim == 256 && aligned16(q) && aligned16(k_cache) && aligned16(v_cache) &&
      aligned16(out) && (gate == nullptr || aligned16(gate)) &&
      (seq_stride & 7u) == 0u) {
    if (gate != nullptr) {
      gqa_attention_256_seg_kernel<true><<<grid, 256, 0, stream>>>(
          q, k_cache, v_cache, gate, out, spans, nseq, seq_stride, n_q_heads,
          n_kv_heads);
    } else {
      gqa_attention_256_seg_kernel<false><<<grid, 256, 0, stream>>>(
          q, k_cache, v_cache, nullptr, out, spans, nseq, seq_stride, n_q_heads,
          n_kv_heads);
    }
    return;
  }
  const size_t smem = (2 * head_dim + 32) * sizeof(float);
  attention_seg_kernel<<<grid, 128, smem, stream>>>(
      q, k_cache, v_cache, gate, out, spans, nseq, seq_stride, n_q_heads,
      n_kv_heads, head_dim);
}

int argmax_f32(const float* logits, int n, int* scratch_idx, cudaStream_t stream) {
  argmax_rows_kernel<<<1, 1024, 0, stream>>>(logits, n, scratch_idx);
  int idx = -1;
  cudaMemcpyAsync(&idx, scratch_idx, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  return idx;
}
}  // namespace umminfer::qwen38
