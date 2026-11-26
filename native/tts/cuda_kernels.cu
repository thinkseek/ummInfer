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
constexpr int kReduceBlock = 1024;

inline bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0u;
}
inline bool aligned8(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 7u) == 0u;
}

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(Bf16 x) { return __bfloat162float(x); }

template <typename T>
__device__ __forceinline__ T from_float(float x);
template <>
__device__ __forceinline__ float from_float<float>(float x) {
  return x;
}
template <>
__device__ __forceinline__ Bf16 from_float<Bf16>(float x) {
  return __float2bfloat16(x);
}

__device__ __forceinline__ uint4 ld_cs_u4(const void* p) {
  return __ldcs(reinterpret_cast<const uint4*>(p));
}
__device__ __forceinline__ uint4 ld_ro_u4(const void* p) {
  return __ldg(reinterpret_cast<const uint4*>(p));
}
__device__ __forceinline__ float4 ld_cs_f4(const void* p) {
  return __ldcs(reinterpret_cast<const float4*>(p));
}
__device__ __forceinline__ float4 ld_ro_f4(const void* p) {
  return __ldg(reinterpret_cast<const float4*>(p));
}

__device__ float block_reduce(float v) {
  __shared__ float warp[32];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if (lane == 0) warp[wid] = v;
  __syncthreads();
  if (wid == 0) {
    v = lane < (blockDim.x + 31) / 32 ? warp[lane] : 0.0f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  }
  __shared__ float total;
  if (threadIdx.x == 0) total = v;
  __syncthreads();
  return total;
}

// ---- Norms ------------------------------------------------------------------

template <typename T>
__global__ void rms_kernel(const T* x, const T* w, T* out, int dim, float eps) {
  const int row = blockIdx.x;
  const T* xr = x + static_cast<size_t>(row) * dim;
  T* orow = out + static_cast<size_t>(row) * dim;
  float sum = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = to_float(xr[i]);
    sum += v * v;
  }
  const float scale = rsqrtf(block_reduce(sum) / dim + eps);
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    orow[i] = from_float<T>(to_float(xr[i]) * scale * to_float(w[i]));
}

// Single-warp vectorized BF16 RMSNorm for dim <= 256 (dim % 8 == 0, 16B aligned).
__global__ void rms_norm_warp_vec_bf16_kernel(const Bf16* __restrict__ x,
                                              const Bf16* __restrict__ w,
                                              Bf16* __restrict__ out, int dim,
                                              float eps) {
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

template <typename T>
__global__ void ln_kernel(const T* x, const T* w, const T* b, T* out, int dim,
                          float eps) {
  const int row = blockIdx.x;
  const T* xr = x + static_cast<size_t>(row) * dim;
  T* orow = out + static_cast<size_t>(row) * dim;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) acc += to_float(xr[i]);
  const float mean = block_reduce(acc) / dim;
  acc = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float d = to_float(xr[i]) - mean;
    acc += d * d;
  }
  const float inv = rsqrtf(block_reduce(acc) / dim + eps);
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    orow[i] = from_float<T>((to_float(xr[i]) - mean) * inv * to_float(w[i]) +
                            to_float(b[i]));
}

// ---- Elementwise ------------------------------------------------------------

template <typename T>
__global__ void residual_kernel(T* y, const T* r, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = from_float<T>(to_float(y[i]) + to_float(r[i]));
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

template <typename T>
__global__ void silu_mul_kernel(T* out, const T* gate, const T* up, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float g = to_float(gate[i]), u = to_float(up[i]);
    out[i] = from_float<T>(u * g / (1.0f + expf(-g)));
  }
}

template <typename T>
__global__ void silu_inplace_kernel(T* x, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float v = to_float(x[i]);
    x[i] = from_float<T>(v / (1.0f + expf(-v)));
  }
}

template <typename T>
__global__ void gelu_kernel(T* x, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float v = to_float(x[i]);
    x[i] = from_float<T>(0.5f * v * (1.0f + erff(v * 0.7071067811865476f)));
  }
}

template <typename T>
__global__ void layer_scale_kernel(T* y, const T* x, const T* scale, int rows,
                                   int dim) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows * dim)
    y[i] = from_float<T>(to_float(x[i]) * to_float(scale[i % dim]));
}

// Channel-first [channels, length]; one thread per element, coalesced along t.
__global__ void snake_kernel(const float* __restrict__ in,
                             float* __restrict__ out,
                             const float* __restrict__ alpha,
                             const float* __restrict__ beta, int channels,
                             int length) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y;
  if (t >= length || c >= channels) return;
  const float a = expf(__ldg(alpha + c));
  const float inv_b = 1.0f / (expf(__ldg(beta + c)) + 1e-9f);
  const size_t idx = static_cast<size_t>(c) * length + t;
  const float v = in[idx];
  const float s = sinf(v * a);
  out[idx] = v + inv_b * s * s;
}

__global__ void clamp_kernel(float* x, int n, float lo, float hi) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = fminf(fmaxf(x[i], lo), hi);
}

template <typename T>
__global__ void gather_kernel(const T* table, const int* ids, T* out, int m,
                              int dim) {
  const int row = blockIdx.y;
  if (row >= m) return;
  const T* src = table + static_cast<size_t>(ids[row]) * dim;
  T* dst = out + static_cast<size_t>(row) * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) dst[i] = src[i];
}

// Fuses 16 codebook row gathers + sequential BF16 residual adds + d_pad, and
// optionally updates d_hist[step] and *d_step_pos on device.
__global__ void sum_frame_embeddings_bf16_kernel(
    const Bf16* const* __restrict__ tables, const int* __restrict__ d_codes,
    const Bf16* __restrict__ d_pad, Bf16* __restrict__ dst,
    int* __restrict__ d_hist_slot, int* __restrict__ d_step_pos, int seq_len,
    int groups, int dim) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) {
    if (d_hist_slot) *d_hist_slot = d_codes[0];
    if (d_step_pos) *d_step_pos = seq_len;
  }
  if (i >= dim) return;
  const int c0 = d_codes[0];
  float acc = __bfloat162float(tables[0][static_cast<size_t>(c0) * dim + i]);
  for (int g = 1; g < groups; ++g) {
    const int cg = d_codes[g];
    const float v =
        __bfloat162float(tables[g][static_cast<size_t>(cg) * dim + i]);
    acc = __bfloat162float(__float2bfloat16(acc + v));
  }
  acc = __bfloat162float(
      __float2bfloat16(acc + __bfloat162float(__ldg(d_pad + i))));
  dst[i] = __float2bfloat16(acc);
}

// ---- Scalar fallback GEMM / GEMV --------------------------------------------

template <typename T>
__global__ void gemm_scalar_kernel(T* y, const T* x, const T* w, const T* bias,
                                   int m, int k, int out) {
  const int col = blockIdx.x, row = blockIdx.y;
  if (col >= out || row >= m) return;
  const T* wr = w + static_cast<size_t>(col) * k;
  const T* xr = x + static_cast<size_t>(row) * k;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < k; i += blockDim.x)
    acc += to_float(wr[i]) * to_float(xr[i]);
  acc = block_reduce(acc);
  if (threadIdx.x == 0) {
    if (bias) acc += to_float(bias[col]);
    y[static_cast<size_t>(row) * out + col] = from_float<T>(acc);
  }
}

template <typename T>
__global__ void gemv_scalar_kernel(const T* w, const T* x, float* out,
                                   int n_out, int k) {
  const int row = blockIdx.x;
  if (row >= n_out) return;
  const T* wr = w + static_cast<size_t>(row) * k;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < k; i += blockDim.x)
    acc += to_float(wr[i]) * to_float(x[i]);
  const float total = block_reduce(acc);
  if (threadIdx.x == 0) out[row] = total;
}

// ---- Proven SM90 Vectorized BF16 GEMV / Small-M / Multi-Proj / SwiGLU -------

// One warp per output row, 128-bit streaming weight loads (__ldcs) + read-only
// cached activation loads (__ldg), K-unrolled by kUnroll.
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
    const float wf = __bfloat162float(wr[i]);
    #pragma unroll
    for (int b = 0; b < kM; ++b)
      acc[b] += wf * __bfloat162float(x[static_cast<size_t>(b) * k + i]);
  }
  #pragma unroll
  for (int b = 0; b < kM; ++b) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc[b] += __shfl_down_sync(0xffffffffu, acc[b], off);
  }
}

template <int kM, int kUnroll, typename OutT, bool kAddResidual>
__global__ void __launch_bounds__(128) gemv_bf16_kernel(
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
    const float bv = bias ? __bfloat162float(bias[row]) : 0.0f;
    #pragma unroll
    for (int b = 0; b < kM; ++b) {
      const size_t o = static_cast<size_t>(m0 + b) * n_out + row;
      const float val = acc[b] + bv;
      if constexpr (sizeof(OutT) == 4) {
        out[o] = val;
      } else if constexpr (kAddResidual) {
        const float cur = __bfloat162float(out[o]);
        const float prod = __bfloat162float(__float2bfloat16(val));
        out[o] = __float2bfloat16(cur + prod);
      } else {
        out[o] = __float2bfloat16(val);
      }
    }
  }
}

// Fused 3-way Q/K/V BF16 projection sharing input x.
template <int kM, int kUnroll>
__global__ void __launch_bounds__(128) gemv_qkv_bf16_kernel(
    QkvGemvDesc desc, const Bf16* __restrict__ x, int k) {
  const int global_row = blockIdx.x * 4 + (threadIdx.x >> 5);
  const MultiGemvRowLoc loc = multi_gemv_locate_row(global_row, desc.n_out, 3);
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
    #pragma unroll
    for (int b = 0; b < kM; ++b) {
      out_base[static_cast<size_t>(m0 + b) * out_stride + loc.local_row] =
          __float2bfloat16(acc[b]);
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
      const __nv_bfloat162* vgb = reinterpret_cast<const __nv_bfloat162*>(&wgv[u]);
      const __nv_bfloat162* vub = reinterpret_cast<const __nv_bfloat162*>(&wuv[u]);
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
        const __nv_bfloat162* xbb = reinterpret_cast<const __nv_bfloat162*>(&xv[b]);
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
    const float wgf = __bfloat162float(wgr[i]);
    const float wuf = __bfloat162float(wur[i]);
    #pragma unroll
    for (int b = 0; b < kM; ++b) {
      const float xf = __bfloat162float(xb[static_cast<size_t>(b) * k + i]);
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
      const float g = __bfloat162float(__float2bfloat16(acc_g[b]));
      const float u = __bfloat162float(__float2bfloat16(acc_u[b]));
      out[static_cast<size_t>(m0 + b) * n_out + row] =
          __float2bfloat16(u * g / (1.0f + expf(-g)));
    }
  }
}

// ---- Vectorized FP32 warp-per-row batched GEMM for CodecDecoder -------------

template <int kM>
__global__ void __launch_bounds__(128) gemm_f32_vec_kernel(
    const float* __restrict__ w, const float* __restrict__ x,
    const float* __restrict__ bias, float* __restrict__ out, int n_out, int k) {
  constexpr int kStr = 32 * 4;
  const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (row >= n_out) return;
  const int m0 = blockIdx.y * kM;
  const int lane = threadIdx.x & 31;
  const float* wr = w + static_cast<size_t>(row) * k;
  const float* xb = x + static_cast<size_t>(m0) * k;

  float acc[kM];
  #pragma unroll
  for (int b = 0; b < kM; ++b) acc[b] = 0.0f;

  int i = lane * 4;
  for (; i + 4 <= k; i += kStr) {
    const float4 wv = ld_cs_f4(wr + i);
    #pragma unroll
    for (int b = 0; b < kM; ++b) {
      const float4 xv = ld_ro_f4(xb + static_cast<size_t>(b) * k + i);
      acc[b] += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
    }
  }
  for (; i < k; ++i) {
    const float wf = wr[i];
    #pragma unroll
    for (int b = 0; b < kM; ++b) acc[b] += wf * xb[static_cast<size_t>(b) * k + i];
  }
  #pragma unroll
  for (int b = 0; b < kM; ++b) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc[b] += __shfl_down_sync(0xffffffffu, acc[b], off);
  }
  if (lane == 0) {
    const float bv = bias ? bias[row] : 0.0f;
    #pragma unroll
    for (int b = 0; b < kM; ++b) {
      out[static_cast<size_t>(m0 + b) * n_out + row] = acc[b] + bv;
    }
  }
}

// ---- Logits / Argmax --------------------------------------------------------

__global__ void argmax_kernel(const float* x, int n, int* best) {
  __shared__ int s_idx[1024];
  float lv = -FLT_MAX;
  int li = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    if (x[i] > lv) {
      lv = x[i];
      li = i;
    }
  }
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_down_sync(0xffffffffu, lv, o);
    const int oi = __shfl_down_sync(0xffffffffu, li, o);
    if (ov > lv) {
      lv = ov;
      li = oi;
    }
  }
  __shared__ float wv[32];
  if (lane == 0) {
    wv[wid] = lv;
    s_idx[wid] = li;
  }
  __syncthreads();
  if (wid == 0) {
    const int nw = (blockDim.x + 31) / 32;
    if (threadIdx.x < nw) {
      lv = wv[lane];
      li = s_idx[lane];
    }
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_down_sync(0xffffffffu, lv, o);
      const int oi = __shfl_down_sync(0xffffffffu, li, o);
      if (ov > lv) {
        lv = ov;
        li = oi;
      }
    }
    if (lane == 0) *best = li;
  }
}

__global__ void suppress_kernel(float* logits, int lo, int hi, int keep) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= lo && i < hi && i != keep) logits[i] = -INFINITY;
}

__global__ void repetition_penalty_kernel(float* logits, const int* history,
                                          int n_history, int n_logits,
                                          float penalty) {
  extern __shared__ unsigned int seen[];
  const int words = (n_logits + 31) / 32;
  for (int i = threadIdx.x; i < words; i += blockDim.x) seen[i] = 0;
  __syncthreads();
  for (int i = threadIdx.x; i < n_history; i += blockDim.x) {
    const int id = history[i];
    if (id >= 0 && id < n_logits) {
      // Duplicate atomic bit sets are idempotent; no float logit is shared.
      atomicOr(&seen[id / 32], 1u << (id % 32));
    }
  }
  __syncthreads();
  for (int id = threadIdx.x; id < n_logits; id += blockDim.x) {
    if ((seen[id / 32] & (1u << (id % 32))) != 0) {
      logits[id] = apply_repetition_penalty(logits[id], penalty);
    }
  }
}

__global__ void ban_logit_kernel(float* logits, int id) {
  logits[id] = -INFINITY;
}

// ---- RoPE & Fused QK-Norm + RoPE + KV-Append --------------------------------

template <typename T>
__global__ void rope_kernel(T* x, const int* positions, const float* inv_freq,
                            int n, int heads, int head_dim) {
  const int token = blockIdx.y;
  const int half = head_dim / 2;
  const int pos = positions[token];
  for (int h = threadIdx.y; h < heads; h += blockDim.y) {
    T* row = x + (static_cast<size_t>(token) * heads + h) * head_dim;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
      const float angle = pos * inv_freq[i];
      float s, c;
      sincosf(angle, &s, &c);
      const float x1 = to_float(row[i]), x2 = to_float(row[i + half]);
      row[i] = from_float<T>(x1 * c - x2 * s);
      row[i + half] = from_float<T>(x2 * c + x1 * s);
    }
  }
}

__global__ void rope_offset_f32_kernel(float* __restrict__ x, int pos_base,
                                       const float* __restrict__ inv_freq,
                                       int n, int heads, int head_dim) {
  const int token = blockIdx.y;
  const int half = head_dim / 2;
  const int pos = pos_base + token;
  for (int h = threadIdx.y; h < heads; h += blockDim.y) {
    float* row = x + (static_cast<size_t>(token) * heads + h) * head_dim;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
      float s, c;
      sincosf(pos * __ldg(inv_freq + i), &s, &c);
      const float x1 = row[i], x2 = row[i + half];
      row[i] = x1 * c - x2 * s;
      row[i + half] = x2 * c + x1 * s;
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

  const float x0 = __bfloat162float(in_row[i0]);
  const float x1 = __bfloat162float(in_row[i1]);
  const float x2 = __bfloat162float(in_row[i2]);
  const float x3 = __bfloat162float(in_row[i3]);

  float ss = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    ss += __shfl_xor_sync(0xffffffffu, ss, off);
  const float inv = rsqrtf(ss / static_cast<float>(kHeadDim) + eps);

  // Exact BF16 rounding between RMSNorm and RoPE, matching unfused path.
  const float n0 =
      __bfloat162float(__float2bfloat16(x0 * inv * __bfloat162float(__ldg(nw + i0))));
  const float n1 =
      __bfloat162float(__float2bfloat16(x1 * inv * __bfloat162float(__ldg(nw + i1))));
  const float n2 =
      __bfloat162float(__float2bfloat16(x2 * inv * __bfloat162float(__ldg(nw + i2))));
  const float n3 =
      __bfloat162float(__float2bfloat16(x3 * inv * __bfloat162float(__ldg(nw + i3))));

  const int pos = pos_base + tok;
  float s0, c0, s1, c1;
  sincosf(pos * __ldg(inv_freq + i0), &s0, &c0);
  sincosf(pos * __ldg(inv_freq + i1), &s1, &c1);

  out_row[i0] = __float2bfloat16(n0 * c0 - n2 * s0);
  out_row[i2] = __float2bfloat16(n2 * c0 + n0 * s0);
  out_row[i1] = __float2bfloat16(n1 * c1 - n3 * s1);
  out_row[i3] = __float2bfloat16(n3 * c1 + n1 * s1);
}

// ---- Attention (Fast Online-Softmax for head_dim=128 BF16 & head_dim=64 F32)

// Multi-warp single-pass Online Softmax for head_dim == 128 BF16.
// Each lane holds 4 contiguous BF16 elements (uint2 coalesced loads).
// Only ONE __syncthreads() in the entire kernel (at final warp merge).
template <int kWarps, bool kDevPos>
__global__ void __launch_bounds__(kWarps * 32) gqa_attention_128_bf16_kernel(
    const Bf16* __restrict__ q, const Bf16* __restrict__ k,
    const Bf16* __restrict__ v, Bf16* __restrict__ out, int nkv, int nqh,
    int nkh, int mask, int window, int q_offset,
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
    hi = (*d_pos_ptr) + 1;
  } else if (mask == 0) {
    lo = 0;
    hi = min(nkv, q_offset + qi + 1);
  } else {
    lo = max(0, q_offset + qi - window + 1);
    hi = min(nkv, q_offset + qi + 1);
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

// 4-warp single-pass Online Softmax for head_dim == 64 FP32 (Codec pre-transformer).
__global__ void __launch_bounds__(128) gqa_attention_64_f32_kernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, float* __restrict__ out, int nkv, int nqh,
    int nkh, int mask, int window, int q_offset) {
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
  if (mask == 0) {
    lo = 0;
    hi = min(nkv, q_offset + qi + 1);
  } else {
    lo = max(0, q_offset + qi - window + 1);
    hi = min(nkv, q_offset + qi + 1);
  }

  const float* qrow = q + (static_cast<size_t>(qi) * nqh + qh) * kHeadDim;
  const float2 q01 = __ldg(reinterpret_cast<const float2*>(qrow) + lane);

  float w_max = -INFINITY;
  float w_sum = 0.0f;
  float a0 = 0.0f, a1 = 0.0f;

  for (int kj = lo + wid; kj < hi; kj += kWarps) {
    const size_t kv_off = (static_cast<size_t>(kj) * nkh + kvh) * kHeadDim;
    const float2 k01 = __ldg(reinterpret_cast<const float2*>(k + kv_off) + lane);
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

    const float2 v01 = __ldg(reinterpret_cast<const float2*>(v + kv_off) + lane);
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
    float* orow = out + (static_cast<size_t>(qi) * nqh + qh) * kHeadDim;
    reinterpret_cast<float2*>(orow)[lane] = make_float2(r0 * inv, r1 * inv);
  }
}

// Generic fallback GQA kernel (no device lambdas).
template <typename T>
__device__ float gqa_block_dot(const float* qs, const T* row, int head_dim,
                               float* scratch) {
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const int nwarps = blockDim.x >> 5;
  float p = 0.0f;
  for (int i = tid; i < head_dim; i += blockDim.x) p += qs[i] * to_float(row[i]);
  for (int o = 16; o > 0; o >>= 1) p += __shfl_down_sync(0xffffffffu, p, o);
  if (lane == 0) scratch[wid] = p;
  __syncthreads();
  __shared__ float sd;
  if (tid == 0) {
    float z = 0.0f;
    for (int w = 0; w < nwarps; ++w) z += scratch[w];
    sd = z;
  }
  __syncthreads();
  return sd;
}

template <typename T>
__global__ void gqa_fallback_kernel(const T* q, const T* k, const T* v, T* out,
                                    int nq, int nkv, int nqh, int nkh,
                                    int head_dim, int mask, int window,
                                    int q_offset) {
  extern __shared__ float smem[];
  float* qs = smem;
  float* acc = smem + head_dim;
  float* scratch = smem + 2 * head_dim;
  __shared__ float s_max, s_sum, s_w;
  const int qi = blockIdx.y, qh = blockIdx.x;
  const int tid = threadIdx.x;
  const int kvh = qh * nkh / nqh;
  const T* qrow = q + (static_cast<size_t>(qi) * nqh + qh) * head_dim;
  if (tid < head_dim) qs[tid] = to_float(qrow[tid]);
  __syncthreads();

  int lo, hi;
  if (mask == 0) {
    lo = 0;
    hi = min(nkv, q_offset + qi + 1);
  } else if (mask == 1) {
    lo = max(0, q_offset + qi - window + 1);
    hi = min(nkv, q_offset + qi + 1);
  } else {
    const int ws = (qi / window) * window;
    lo = ws;
    hi = min(nkv, ws + window);
  }
  const float scale = rsqrtf(static_cast<float>(head_dim));
  if (tid == 0) s_max = -FLT_MAX;
  __syncthreads();
  for (int j = lo; j < hi; ++j) {
    const float d =
        gqa_block_dot(qs, k + (static_cast<size_t>(j) * nkh + kvh) * head_dim,
                      head_dim, scratch) *
        scale;
    if (tid == 0) s_max = fmaxf(s_max, d);
    __syncthreads();
  }
  const float mx = s_max;
  if (tid == 0) s_sum = 0.0f;
  if (tid < head_dim) acc[tid] = 0.0f;
  __syncthreads();
  for (int j = lo; j < hi; ++j) {
    const float wgt = expf(
        gqa_block_dot(qs, k + (static_cast<size_t>(j) * nkh + kvh) * head_dim,
                      head_dim, scratch) *
            scale -
        mx);
    if (tid == 0) {
      s_w = wgt;
      s_sum += wgt;
    }
    __syncthreads();
    if (tid < head_dim)
      acc[tid] +=
          s_w *
          to_float(v[(static_cast<size_t>(j) * nkh + kvh) * head_dim + tid]);
    __syncthreads();
  }
  if (tid < head_dim)
    out[(static_cast<size_t>(qi) * nqh + qh) * head_dim + tid] =
        from_float<T>(acc[tid] / s_sum);
}

// ---- 1-D Convolutions & Codec Helpers (F32, Stateful & Register-Tiled) ------

// 8-output-channels-per-thread dense causal Conv1d (groups == 1, co >= 8).
// Shares each input load across 8 output channels (8x less HBM/L2 traffic)
// while keeping exact (kk, ic) FP32 accumulation order.
__global__ void conv1d_dense_8ch_kernel(const float* __restrict__ in,
                                        const float* __restrict__ hist,
                                        const float* __restrict__ w,
                                        const float* __restrict__ bias,
                                        float* __restrict__ out, int ci, int co,
                                        int L, int K, int dilation) {
  const int t = blockIdx.x * 128 + threadIdx.x;
  const int c0 = blockIdx.y * 8;
  if (t >= L || c0 >= co) return;
  const int pad = (K - 1) * dilation;
  int ch[8];
  float acc[8];
  #pragma unroll
  for (int r = 0; r < 8; ++r) {
    ch[r] = (c0 + r < co) ? (c0 + r) : (co - 1);
    acc[r] = bias ? __ldg(bias + ch[r]) : 0.0f;
  }
  for (int kk = 0; kk < K; ++kk) {
    const CausalSrcLoc loc = causal_conv_src_lookup(t, kk, dilation, pad);
    if (loc.from_hist) {
      if (!hist) continue;
      for (int ic = 0; ic < ci; ++ic) {
        const float x = __ldg(hist + static_cast<size_t>(ic) * pad + loc.idx);
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
          const float wr =
              __ldg(w + ((static_cast<size_t>(ch[r]) * ci + ic) * K) + kk);
          acc[r] += x * wr;
        }
      }
    } else if (loc.idx < L) {
      for (int ic = 0; ic < ci; ++ic) {
        const float x = __ldg(in + static_cast<size_t>(ic) * L + loc.idx);
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
          const float wr =
              __ldg(w + ((static_cast<size_t>(ch[r]) * ci + ic) * K) + kk);
          acc[r] += x * wr;
        }
      }
    }
  }
  #pragma unroll
  for (int r = 0; r < 8; ++r) {
    if (c0 + r < co) out[static_cast<size_t>(c0 + r) * L + t] = acc[r];
  }
}

// Generic / depthwise causal Conv1d (coalesced along t, 1 channel per blockIdx.y).
__global__ void conv1d_generic_kernel(const float* __restrict__ in,
                                      const float* __restrict__ hist,
                                      const float* __restrict__ w,
                                      const float* __restrict__ bias,
                                      float* __restrict__ out, int ci, int co,
                                      int L, int K, int dilation, int groups) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y;
  if (t >= L || c >= co) return;
  const int pad = (K - 1) * dilation;
  const int in_per_group = ci / groups;
  const int g = c / (co / groups);
  float acc = bias ? __ldg(bias + c) : 0.0f;
  for (int kk = 0; kk < K; ++kk) {
    const CausalSrcLoc loc = causal_conv_src_lookup(t, kk, dilation, pad);
    if (loc.from_hist) {
      if (!hist) continue;
      if (groups == ci) {
        acc += __ldg(hist + static_cast<size_t>(c) * pad + loc.idx) *
               __ldg(w + static_cast<size_t>(c) * K + kk);
      } else {
        for (int ic = g * in_per_group; ic < (g + 1) * in_per_group; ++ic) {
          acc += __ldg(hist + static_cast<size_t>(ic) * pad + loc.idx) *
                 __ldg(w + ((static_cast<size_t>(c) * in_per_group +
                             (ic - g * in_per_group)) *
                            K) +
                       kk);
        }
      }
    } else if (loc.idx < L) {
      if (groups == ci) {
        acc += __ldg(in + static_cast<size_t>(c) * L + loc.idx) *
               __ldg(w + static_cast<size_t>(c) * K + kk);
      } else {
        for (int ic = g * in_per_group; ic < (g + 1) * in_per_group; ++ic) {
          acc += __ldg(in + static_cast<size_t>(ic) * L + loc.idx) *
                 __ldg(w + ((static_cast<size_t>(c) * in_per_group +
                             (ic - g * in_per_group)) *
                            K) +
                       kk);
        }
      }
    }
  }
  out[static_cast<size_t>(c) * L + t] = acc;
}

__global__ void update_conv_history_kernel(float* __restrict__ hist,
                                           const float* __restrict__ in,
                                           int channels, int pad, int length) {
  const int c = blockIdx.x;
  const int j = threadIdx.x;
  if (c >= channels) return;
  float val = 0.0f;
  if (j < pad) {
    const CausalSrcLoc src = causal_conv_hist_update_src(j, pad, length);
    val = src.from_hist ? hist[static_cast<size_t>(c) * pad + src.idx]
                        : in[static_cast<size_t>(c) * length + src.idx];
  }
  __syncthreads();
  if (j < pad) hist[static_cast<size_t>(c) * pad + j] = val;
}

// 8-output-channels-per-thread ConvTranspose1d with direct stride stepping.
__global__ void transconv_8ch_kernel(const float* __restrict__ in,
                                     const float* __restrict__ hist,
                                     const float* __restrict__ w,
                                     const float* __restrict__ bias,
                                     float* __restrict__ out, int ci, int co,
                                     int L, int K, int stride) {
  const int out_len = L * stride;
  const int o = blockIdx.x * 128 + threadIdx.x;
  const int c0 = blockIdx.y * 8;
  if (o >= out_len || c0 >= co) return;
  const int t_local = o / stride;
  const int p = o - t_local * stride;
  int ch[8];
  float acc[8];
  #pragma unroll
  for (int r = 0; r < 8; ++r) {
    ch[r] = (c0 + r < co) ? (c0 + r) : (co - 1);
    acc[r] = bias ? __ldg(bias + ch[r]) : 0.0f;
  }
  for (int step = 0, kk = p; kk < K; ++step, kk += stride) {
    const CausalSrcLoc loc = transconv_step_src(t_local, step);
    if (loc.from_hist) {
      if (!hist || (t_local - step) != -1) continue;
      for (int ic = 0; ic < ci; ++ic) {
        const float x = __ldg(hist + ic);
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
          const float wr =
              __ldg(w + ((static_cast<size_t>(ic) * co + ch[r]) * K) + kk);
          acc[r] += x * wr;
        }
      }
    } else if (loc.idx < L) {
      for (int ic = 0; ic < ci; ++ic) {
        const float x = __ldg(in + static_cast<size_t>(ic) * L + loc.idx);
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
          const float wr =
              __ldg(w + ((static_cast<size_t>(ic) * co + ch[r]) * K) + kk);
          acc[r] += x * wr;
        }
      }
    }
  }
  #pragma unroll
  for (int r = 0; r < 8; ++r) {
    if (c0 + r < co) out[static_cast<size_t>(c0 + r) * out_len + o] = acc[r];
  }
}

__global__ void transconv_generic_kernel(const float* __restrict__ in,
                                         const float* __restrict__ hist,
                                         const float* __restrict__ w,
                                         const float* __restrict__ bias,
                                         float* __restrict__ out, int ci,
                                         int co, int L, int K, int stride) {
  const int out_len = L * stride;
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y;
  if (o >= out_len || c >= co) return;
  const int t_local = o / stride;
  const int p = o - t_local * stride;
  float acc = bias ? __ldg(bias + c) : 0.0f;
  for (int step = 0, kk = p; kk < K; ++step, kk += stride) {
    const CausalSrcLoc loc = transconv_step_src(t_local, step);
    if (loc.from_hist) {
      if (!hist || (t_local - step) != -1) continue;
      for (int ic = 0; ic < ci; ++ic)
        acc += __ldg(hist + ic) *
               __ldg(w + ((static_cast<size_t>(ic) * co + c) * K) + kk);
    } else if (loc.idx < L) {
      for (int ic = 0; ic < ci; ++ic)
        acc += __ldg(in + static_cast<size_t>(ic) * L + loc.idx) *
               __ldg(w + ((static_cast<size_t>(ic) * co + c) * K) + kk);
    }
  }
  out[static_cast<size_t>(c) * out_len + o] = acc;
}

__global__ void save_last_sample_kernel(float* __restrict__ hist,
                                        const float* __restrict__ in,
                                        int channels, int length) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < channels) hist[c] = in[static_cast<size_t>(c) * length + (length - 1)];
}

__global__ void dequant_kernel(const float* sum, const float* usage, float* emb,
                               int rows, int dim, float eps) {
  const int r = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows || d >= dim) return;
  const float u = usage[r];
  emb[static_cast<size_t>(r) * dim + d] =
      sum[static_cast<size_t>(r) * dim + d] / fmaxf(u, eps);
}

// Fused RVQ gather + sum across groups + transpose directly to [vq_dim, T].
__global__ void rvq_gather_sum_transposed_kernel(
    const float* __restrict__ codebooks, const int* __restrict__ codes,
    float* __restrict__ out, int first_group, int count, int frames, int groups,
    int codebook_size, int vq_dim) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int d = blockIdx.y;
  if (t >= frames || d >= vq_dim) return;
  float acc = 0.0f;
  for (int g = 0; g < count; ++g) {
    const int grp = first_group + g;
    const int id = codes[t * groups + grp];
    acc += __ldg(
        codebooks +
        (static_cast<size_t>(grp) * codebook_size + id) * vq_dim + d);
  }
  out[static_cast<size_t>(d) * frames + t] = acc;
}

__global__ void shift_kv_window_kernel(float* __restrict__ kv, int num_slabs,
                                       int slot_cap, int width, int src_slot,
                                       int keep_len) {
  const size_t elems_per_slab = static_cast<size_t>(keep_len) * width;
  const size_t idx =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int slab = blockIdx.y;
  if (slab >= num_slabs || idx >= elems_per_slab) return;
  float* base = kv + static_cast<size_t>(slab) * slot_cap * width;
  base[idx] = base[static_cast<size_t>(src_slot) * width + idx];
}

__global__ void sum_tables_kernel(float* dst, const float* const* srcs,
                                  int count, int rows, int dim) {
  const int r = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows || d >= dim) return;
  float acc = 0.0f;
  for (int g = 0; g < count; ++g)
    acc += srcs[g][static_cast<size_t>(r) * dim + d];
  dst[static_cast<size_t>(r) * dim + d] = acc;
}

// 32x32 tiled bank-conflict-free 2D transpose: in [rows, cols] -> out [cols, rows].
__global__ void transpose_2d_tiled_kernel(const float* __restrict__ in,
                                          float* __restrict__ out, int rows,
                                          int cols) {
  __shared__ float tile[32][33];
  const int x = blockIdx.x * 32 + threadIdx.x;
  const int y0 = blockIdx.y * 32 + threadIdx.y;
  #pragma unroll
  for (int k = 0; k < 32; k += 8) {
    const int y = y0 + k;
    if (x < cols && y < rows)
      tile[threadIdx.y + k][threadIdx.x] =
          in[static_cast<size_t>(y) * cols + x];
  }
  __syncthreads();
  const int tx = blockIdx.y * 32 + threadIdx.x;
  const int ty0 = blockIdx.x * 32 + threadIdx.y;
  #pragma unroll
  for (int k = 0; k < 32; k += 8) {
    const int ty = ty0 + k;
    if (tx < rows && ty < cols)
      out[static_cast<size_t>(ty) * rows + tx] =
          tile[threadIdx.x][threadIdx.y + k];
  }
}

__global__ void bias_rows_kernel(float* y, const float* bias, int rows,
                                 int dim) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows * dim) y[i] += bias[i % dim];
}

__global__ void gather_col_kernel(const int* codes, int* out, int T, int stride,
                                  int group) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < T) out[t] = codes[t * stride + group];
}

// ---- Dispatch Helpers -------------------------------------------------------

template <int kM, typename OutT, bool kAddResidual>
void launch_bf16_gemv_exact(const Bf16* w, const Bf16* x, const Bf16* bias,
                            OutT* out, int n_groups, int n_out, int k,
                            cudaStream_t st) {
  dim3 grid((n_out + 3) / 4, n_groups);
  if constexpr (tts_gemv_unroll_for(kM) == 4) {
    gemv_bf16_kernel<kM, 4, OutT, kAddResidual>
        <<<grid, 128, 0, st>>>(w, x, bias, out, n_out, k);
  } else {
    gemv_bf16_kernel<kM, 1, OutT, kAddResidual>
        <<<grid, 128, 0, st>>>(w, x, bias, out, n_out, k);
  }
}

template <typename OutT, bool kAddResidual>
void launch_bf16_gemv_tail(const Bf16* w, const Bf16* x, const Bf16* bias,
                           OutT* out, int rem, int n_out, int k,
                           cudaStream_t st) {
  switch (rem) {
    case 1:
      launch_bf16_gemv_exact<1, OutT, kAddResidual>(w, x, bias, out, 1, n_out,
                                                    k, st);
      break;
    case 2:
      launch_bf16_gemv_exact<2, OutT, kAddResidual>(w, x, bias, out, 1, n_out,
                                                    k, st);
      break;
    case 3:
      launch_bf16_gemv_exact<3, OutT, kAddResidual>(w, x, bias, out, 1, n_out,
                                                    k, st);
      break;
    case 4:
      launch_bf16_gemv_exact<4, OutT, kAddResidual>(w, x, bias, out, 1, n_out,
                                                    k, st);
      break;
    case 5:
      launch_bf16_gemv_exact<5, OutT, kAddResidual>(w, x, bias, out, 1, n_out,
                                                    k, st);
      break;
    case 6:
      launch_bf16_gemv_exact<6, OutT, kAddResidual>(w, x, bias, out, 1, n_out,
                                                    k, st);
      break;
    case 7:
      launch_bf16_gemv_exact<7, OutT, kAddResidual>(w, x, bias, out, 1, n_out,
                                                    k, st);
      break;
    default:
      break;
  }
}

template <int kM>
void launch_qkv_exact(const QkvGemvDesc& desc, const Bf16* x, int n_groups,
                      int total_n, int k, cudaStream_t st) {
  dim3 grid((total_n + 3) / 4, n_groups);
  if constexpr (tts_gemv_unroll_for(kM) == 4) {
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
  if constexpr (tts_swiglu_unroll_for(kM) == 2) {
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

void launch_f32_gemm_tail(const float* w, const float* x, const float* bias,
                          float* out, int rem, int n_out, int k,
                          cudaStream_t st) {
  dim3 grid((n_out + 3) / 4, 1);
  switch (rem) {
    case 1: gemm_f32_vec_kernel<1><<<grid, 128, 0, st>>>(w, x, bias, out, n_out, k); break;
    case 2: gemm_f32_vec_kernel<2><<<grid, 128, 0, st>>>(w, x, bias, out, n_out, k); break;
    case 3: gemm_f32_vec_kernel<3><<<grid, 128, 0, st>>>(w, x, bias, out, n_out, k); break;
    default: break;
  }
}

}  // namespace

// ---- Public API -------------------------------------------------------------

template <typename T>
void rms_norm_t(const T* x, const T* weight, T* out, int rows, int dim,
                float eps, cudaStream_t st) {
  if (rows <= 0 || dim <= 0) return;
  if constexpr (sizeof(T) == 2) {
    if ((dim & 7) == 0 && aligned16(x) && aligned16(weight) && aligned16(out)) {
      if (dim <= 256) {
        rms_norm_warp_vec_bf16_kernel<<<rows, 32, 0, st>>>(x, weight, out, dim,
                                                           eps);
      } else {
        rms_norm_vec_bf16_kernel<<<rows, 256, 0, st>>>(x, weight, out, dim, eps);
      }
      return;
    }
  }
  rms_kernel<T><<<rows, kReduceBlock, 0, st>>>(x, weight, out, dim, eps);
}

template <typename T>
void layer_norm_t(const T* x, const T* w, const T* b, T* out, int rows, int dim,
                  float eps, cudaStream_t st) {
  if (rows <= 0 || dim <= 0) return;
  ln_kernel<T><<<rows, kBlock, 0, st>>>(x, w, b, out, dim, eps);
}

template <typename T>
void add_residual_t(T* y, const T* r, int n, cudaStream_t st) {
  if (n <= 0) return;
  if constexpr (sizeof(T) == 2) {
    if ((n & 7) == 0 && aligned16(y) && aligned16(r)) {
      const int vecs = n >> 3;
      residual_vec_bf16_kernel<<<(vecs + kBlock - 1) / kBlock, kBlock, 0, st>>>(
          y, r, vecs);
      return;
    }
  }
  residual_kernel<T><<<(n + kBlock - 1) / kBlock, kBlock, 0, st>>>(y, r, n);
}

template <typename T>
void silu_mul_t(T* out, const T* g, const T* u, int n, cudaStream_t st) {
  if (n <= 0) return;
  silu_mul_kernel<T><<<(n + kBlock - 1) / kBlock, kBlock, 0, st>>>(out, g, u, n);
}

template <typename T>
void silu_inplace_t(T* x, int n, cudaStream_t st) {
  if (n <= 0) return;
  silu_inplace_kernel<T><<<(n + kBlock - 1) / kBlock, kBlock, 0, st>>>(x, n);
}

template <typename T>
void gelu_inplace_t(T* x, int n, cudaStream_t st) {
  if (n <= 0) return;
  gelu_kernel<T><<<(n + kBlock - 1) / kBlock, kBlock, 0, st>>>(x, n);
}

template <typename T>
void layer_scale_t(T* y, const T* x, const T* s, int rows, int dim,
                   cudaStream_t st) {
  if (rows <= 0 || dim <= 0) return;
  layer_scale_kernel<T><<<(rows * dim + kBlock - 1) / kBlock, kBlock, 0, st>>>(
      y, x, s, rows, dim);
}

void snake_beta_f32(float* x, const float* a, const float* b, int channels,
                    int length, cudaStream_t st) {
  if (channels <= 0 || length <= 0) return;
  dim3 grid((length + kBlock - 1) / kBlock, channels);
  snake_kernel<<<grid, kBlock, 0, st>>>(x, x, a, b, channels, length);
}

void snake_beta_out_f32(const float* in, float* out, const float* a,
                        const float* b, int channels, int length,
                        cudaStream_t st) {
  if (channels <= 0 || length <= 0) return;
  dim3 grid((length + kBlock - 1) / kBlock, channels);
  snake_kernel<<<grid, kBlock, 0, st>>>(in, out, a, b, channels, length);
}

void clamp_f32(float* x, int n, float lo, float hi, cudaStream_t st) {
  if (n <= 0) return;
  clamp_kernel<<<(n + kBlock - 1) / kBlock, kBlock, 0, st>>>(x, n, lo, hi);
}

__global__ void f32_to_bf16_k(const float* s, Bf16* d, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = __float2bfloat16(s[i]);
}
__global__ void bf16_to_f32_k(const Bf16* s, float* d, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = __bfloat162float(s[i]);
}
void f32_to_bf16(const float* s, Bf16* d, int n, cudaStream_t st) {
  f32_to_bf16_k<<<(n + kBlock - 1) / kBlock, kBlock, 0, st>>>(s, d, n);
}
void bf16_to_f32(const Bf16* s, float* d, int n, cudaStream_t st) {
  bf16_to_f32_k<<<(n + kBlock - 1) / kBlock, kBlock, 0, st>>>(s, d, n);
}

template <typename T>
void gather_rows_t(const T* table, const int* ids, T* out, int m, int dim,
                   cudaStream_t st) {
  if (m <= 0 || dim <= 0) return;
  dim3 grid(1, m);
  gather_kernel<T><<<grid, kBlock, 0, st>>>(table, ids, out, m, dim);
}

void sum_frame_embeddings_bf16(const Bf16* const* tables, const int* d_codes,
                               const Bf16* d_pad, Bf16* dst, int* d_hist_slot,
                               int* d_step_pos, int seq_len, int groups,
                               int dim, cudaStream_t st) {
  if (dim <= 0) return;
  sum_frame_embeddings_bf16_kernel<<<(dim + kBlock - 1) / kBlock, kBlock, 0,
                                     st>>>(tables, d_codes, d_pad, dst,
                                           d_hist_slot, d_step_pos, seq_len,
                                           groups, dim);
}

template <typename T>
void gemm_t(T* y, const T* x, const T* w, const T* bias, int m, int k, int out,
            cudaStream_t st) {
  if (m <= 0 || k <= 0 || out <= 0) return;
  if constexpr (sizeof(T) == 2) {
    if ((k & 7) == 0 && aligned16(w) && aligned16(x)) {
      const int full = m / 8;
      const int rem = m % 8;
      if (full > 0) {
        launch_bf16_gemv_exact<8, Bf16, false>(w, x, bias, y, full, out, k, st);
      }
      if (rem > 0) {
        const size_t off_rows = static_cast<size_t>(full) * 8;
        launch_bf16_gemv_tail<Bf16, false>(
            w, x + off_rows * k, bias, y + off_rows * out, rem, out, k, st);
      }
      return;
    }
  } else {
    if ((k & 3) == 0 && aligned16(w) && aligned16(x)) {
      const int full = m / 4;
      const int rem = m % 4;
      if (full > 0) {
        dim3 grid((out + 3) / 4, full);
        gemm_f32_vec_kernel<4><<<grid, 128, 0, st>>>(w, x, bias, y, out, k);
      }
      if (rem > 0) {
        const size_t off_rows = static_cast<size_t>(full) * 4;
        launch_f32_gemm_tail(w, x + off_rows * k, bias, y + off_rows * out, rem,
                             out, k, st);
      }
      return;
    }
  }
  dim3 grid(out, m);
  gemm_scalar_kernel<T><<<grid, kReduceBlock, 0, st>>>(y, x, w, bias, m, k, out);
}

void gemm_bf16_add_residual(Bf16* y, const Bf16* x, const Bf16* w, int m, int k,
                            int out, cudaStream_t st) {
  if (m <= 0 || k <= 0 || out <= 0) return;
  if ((k & 7) == 0 && aligned16(w) && aligned16(x)) {
    const int full = m / 8;
    const int rem = m % 8;
    if (full > 0) {
      launch_bf16_gemv_exact<8, Bf16, true>(w, x, nullptr, y, full, out, k, st);
    }
    if (rem > 0) {
      const size_t off_rows = static_cast<size_t>(full) * 8;
      launch_bf16_gemv_tail<Bf16, true>(
          w, x + off_rows * k, nullptr, y + off_rows * out, rem, out, k, st);
    }
    return;
  }
  // Fallback: unaligned path not expected in Qwen3-TTS (all dims multiple of 128).
  throw std::runtime_error("gemm_bf16_add_residual requires k%8==0 and 16B alignment");
}

void gemv_qkv_bf16(const QkvGemvDesc& desc, const Bf16* x, int m, int k,
                   cudaStream_t st) {
  if (m <= 0 || k <= 0) return;
  const int total_n = desc.n_out[0] + desc.n_out[1] + desc.n_out[2];
  if (total_n <= 0) return;
  if ((k & 7) == 0 && aligned16(x) && aligned16(desc.w[0]) &&
      aligned16(desc.w[1]) && aligned16(desc.w[2])) {
    const int full = m / 8;
    const int rem = m % 8;
    if (full > 0) {
      launch_qkv_exact<8>(desc, x, full, total_n, k, st);
    }
    if (rem > 0) {
      const size_t off_rows = static_cast<size_t>(full) * 8;
      QkvGemvDesc tail = desc;
      for (int p = 0; p < 3; ++p)
        tail.out[p] = desc.out[p] + off_rows * desc.n_out[p];
      launch_qkv_tail(tail, x + off_rows * k, rem, total_n, k, st);
    }
    return;
  }
  for (int p = 0; p < 3; ++p) {
    gemm_t<Bf16>(desc.out[p], x, desc.w[p], nullptr, m, k, desc.n_out[p], st);
  }
}

void gemv_swiglu_bf16(const Bf16* w_gate, const Bf16* w_up, const Bf16* x,
                      Bf16* out, int m, int n_out, int k, cudaStream_t st) {
  if (m <= 0 || n_out <= 0 || k <= 0) return;
  if ((k & 7) == 0 && aligned16(w_gate) && aligned16(w_up) && aligned16(x)) {
    const int full = m / 8;
    const int rem = m % 8;
    if (full > 0) {
      launch_swiglu_exact<8>(w_gate, w_up, x, out, full, n_out, k, st);
    }
    if (rem > 0) {
      const size_t off_rows = static_cast<size_t>(full) * 8;
      launch_swiglu_tail(w_gate, w_up, x + off_rows * k, out + off_rows * n_out,
                         rem, n_out, k, st);
    }
    return;
  }
  throw std::runtime_error("gemv_swiglu_bf16 requires k%8==0 and 16B alignment");
}

template <typename T>
void gemv_t(const T* w, const T* x, float* out, int n_out, int k,
            cudaStream_t st) {
  if (n_out <= 0 || k <= 0) return;
  if constexpr (sizeof(T) == 2) {
    if ((k & 7) == 0 && aligned16(w) && aligned16(x)) {
      launch_bf16_gemv_exact<1, float, false>(w, x, nullptr, out, 1, n_out, k,
                                              st);
      return;
    }
  }
  gemv_scalar_kernel<T><<<n_out, kReduceBlock, 0, st>>>(w, x, out, n_out, k);
}

int argmax_f32(const float* x, int n, int* scratch, cudaStream_t st) {
  argmax_kernel<<<1, min(1024, n), 0, st>>>(x, n, scratch);
  int idx;
  cudaMemcpyAsync(&idx, scratch, sizeof(int), cudaMemcpyDeviceToHost, st);
  cudaStreamSynchronize(st);
  return idx;
}
void argmax_f32_dev(const float* x, int n, int* out, cudaStream_t st) {
  argmax_kernel<<<1, min(1024, n), 0, st>>>(x, n, out);
}
void suppress_logits(float* logits, int lo, int hi, int keep, cudaStream_t st) {
  suppress_kernel<<<(hi - lo + kBlock - 1) / kBlock, kBlock, 0, st>>>(
      logits, lo, hi, keep);
}
void repetition_penalty_f32(float* logits, const int* history, int n_history,
                            int n_logits, float penalty, cudaStream_t st) {
  if (n_history <= 0 || n_logits <= 0) return;
  const size_t seen_bytes = static_cast<size_t>((n_logits + 31) / 32) *
                            sizeof(unsigned int);
  repetition_penalty_kernel<<<1, min(1024, n_logits), seen_bytes, st>>>(
      logits, history, n_history, n_logits, penalty);
}
void ban_logit_f32(float* logits, int id, cudaStream_t st) {
  ban_logit_kernel<<<1, 1, 0, st>>>(logits, id);
}

template <typename T>
void rope_half_t(T* x, const int* positions, const float* inv_freq, int n,
                 int heads, int head_dim, cudaStream_t st) {
  if (n <= 0) return;
  dim3 grid(1, n);
  dim3 block(64, 4);
  rope_kernel<T>
      <<<grid, block, 0, st>>>(x, positions, inv_freq, n, heads, head_dim);
}

void rope_half_offset_f32(float* x, int pos_base, const float* inv_freq, int n,
                          int heads, int head_dim, cudaStream_t st) {
  if (n <= 0) return;
  dim3 grid(1, n);
  dim3 block(32, 8);
  rope_offset_f32_kernel<<<grid, block, 0, st>>>(x, pos_base, inv_freq, n,
                                                 heads, head_dim);
}

void qk_norm_rope_kv_append_bf16(Bf16* q, const Bf16* k_src, const Bf16* v_src,
                                 const Bf16* q_norm_w, const Bf16* k_norm_w,
                                 const float* inv_freq, Bf16* k_cache,
                                 Bf16* v_cache, int m, int n_q_heads,
                                 int n_kv_heads, int head_dim, int pos_base,
                                 int history_base, float eps, cudaStream_t st) {
  if (m <= 0) return;
  if (head_dim != 128)
    throw std::runtime_error("qk_norm_rope_kv_append_bf16 expects head_dim==128");
  const bool copy_v = (v_src != nullptr && v_cache != nullptr);
  if (copy_v) {
    dim3 grid(n_q_heads + 2 * n_kv_heads, m);
    qk_norm_rope_kv_append_128_kernel<false, true><<<grid, 32, 0, st>>>(
        q, k_src, v_src, q_norm_w, k_norm_w, inv_freq, k_cache, v_cache,
        n_q_heads, n_kv_heads, pos_base, history_base, nullptr, eps);
  } else {
    dim3 grid(n_q_heads + n_kv_heads, m);
    qk_norm_rope_kv_append_128_kernel<false, false><<<grid, 32, 0, st>>>(
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
                                        cudaStream_t st) {
  if (head_dim != 128)
    throw std::runtime_error(
        "qk_norm_rope_kv_append_devpos_bf16 expects head_dim==128");
  dim3 grid(n_q_heads + 2 * n_kv_heads, 1);
  qk_norm_rope_kv_append_128_kernel<true, true><<<grid, 32, 0, st>>>(
      q, k_src, v_src, q_norm_w, k_norm_w, inv_freq, k_cache, v_cache,
      n_q_heads, n_kv_heads, 0, 0, d_pos_ptr, eps);
}

template <typename T>
void gqa_attention_t(const T* q, const T* k, const T* v, T* out, int nq,
                     int nkv, int nqh, int nkh, int head_dim, int mask,
                     int window, int q_offset, cudaStream_t st) {
  if (nq <= 0) return;
  dim3 grid(nqh, nq);
  if constexpr (sizeof(T) == 2) {
    if (head_dim == 128 && (mask == 0 || mask == 1) && aligned8(q) &&
        aligned8(k) && aligned8(v) && aligned8(out)) {
      if (nkv <= 32) {
        gqa_attention_128_bf16_kernel<4, false><<<grid, 128, 0, st>>>(
            q, k, v, out, nkv, nqh, nkh, mask, window, q_offset, nullptr);
      } else {
        gqa_attention_128_bf16_kernel<8, false><<<grid, 256, 0, st>>>(
            q, k, v, out, nkv, nqh, nkh, mask, window, q_offset, nullptr);
      }
      return;
    }
  } else {
    if (head_dim == 64 && (mask == 0 || mask == 1) && aligned8(q) &&
        aligned8(k) && aligned8(v) && aligned8(out)) {
      gqa_attention_64_f32_kernel<<<grid, 128, 0, st>>>(
          q, k, v, out, nkv, nqh, nkh, mask, window, q_offset);
      return;
    }
  }
  const size_t smem = (2 * head_dim + 32) * sizeof(float);
  gqa_fallback_kernel<T><<<grid, head_dim, smem, st>>>(
      q, k, v, out, nq, nkv, nqh, nkh, head_dim, mask, window, q_offset);
}

void gqa_attention_devpos_bf16(const Bf16* q, const Bf16* k, const Bf16* v,
                               Bf16* out, int n_q_heads, int n_kv_heads,
                               int head_dim, const int* d_pos_ptr,
                               cudaStream_t st) {
  if (head_dim != 128)
    throw std::runtime_error("gqa_attention_devpos_bf16 expects head_dim==128");
  dim3 grid(n_q_heads, 1);
  gqa_attention_128_bf16_kernel<8, true><<<grid, 256, 0, st>>>(
      q, k, v, out, 0, n_q_heads, n_kv_heads, 0, 0, 0, d_pos_ptr);
}

void conv1d_causal_stateful_f32(const float* in, const float* hist,
                                const float* w, const float* bias, float* out,
                                int ci, int co, int L, int kernel, int dilation,
                                int groups, cudaStream_t st) {
  if (co <= 0 || L <= 0) return;
  if (groups == 1 && co >= 8) {
    dim3 grid((L + 127) / 128, (co + 7) / 8);
    conv1d_dense_8ch_kernel<<<grid, 128, 0, st>>>(in, hist, w, bias, out, ci,
                                                  co, L, kernel, dilation);
    return;
  }
  dim3 grid((L + kBlock - 1) / kBlock, co);
  conv1d_generic_kernel<<<grid, kBlock, 0, st>>>(in, hist, w, bias, out, ci, co,
                                                 L, kernel, dilation, groups);
}

void conv1d_causal_f32(const float* in, const float* w, const float* bias,
                       float* out, int ci, int co, int L, int kernel,
                       int dilation, int groups, cudaStream_t st) {
  conv1d_causal_stateful_f32(in, nullptr, w, bias, out, ci, co, L, kernel,
                             dilation, groups, st);
}

void update_conv_history_f32(float* hist, const float* in, int channels,
                             int pad, int length, cudaStream_t st) {
  if (channels <= 0 || pad <= 0 || length <= 0) return;
  if (pad > 1024)
    throw std::runtime_error("update_conv_history_f32 pad exceeds 1024");
  const int threads = ((pad + 31) / 32) * 32;
  update_conv_history_kernel<<<channels, threads, 0, st>>>(hist, in, channels,
                                                           pad, length);
}

void conv_transpose1d_stateful_f32(const float* in, const float* hist,
                                   const float* w, const float* bias,
                                   float* out, int ci, int co, int L,
                                   int kernel, int stride, cudaStream_t st) {
  if (co <= 0 || L <= 0) return;
  const int out_len = L * stride;
  if (co >= 8) {
    dim3 grid((out_len + 127) / 128, (co + 7) / 8);
    transconv_8ch_kernel<<<grid, 128, 0, st>>>(in, hist, w, bias, out, ci, co,
                                               L, kernel, stride);
    return;
  }
  dim3 grid((out_len + kBlock - 1) / kBlock, co);
  transconv_generic_kernel<<<grid, kBlock, 0, st>>>(in, hist, w, bias, out, ci,
                                                    co, L, kernel, stride);
}

void conv_transpose1d_f32(const float* in, const float* w, const float* bias,
                          float* out, int ci, int co, int L, int kernel,
                          int stride, cudaStream_t st) {
  conv_transpose1d_stateful_f32(in, nullptr, w, bias, out, ci, co, L, kernel,
                                stride, st);
}

void save_last_sample_f32(float* hist, const float* in, int channels,
                          int length, cudaStream_t st) {
  if (channels <= 0 || length <= 0) return;
  save_last_sample_kernel<<<(channels + kBlock - 1) / kBlock, kBlock, 0, st>>>(
      hist, in, channels, length);
}

void codebook_dequant_f32(const float* sum, const float* usage, float* emb,
                          int codes, int dim, float eps, cudaStream_t st) {
  dim3 grid((dim + kBlock - 1) / kBlock, codes);
  dequant_kernel<<<grid, kBlock, 0, st>>>(sum, usage, emb, codes, dim, eps);
}

void rvq_gather_sum_transposed_f32(const float* codebooks, const int* codes,
                                   float* out, int first_group, int count,
                                   int frames, int groups, int codebook_size,
                                   int vq_dim, cudaStream_t st) {
  if (frames <= 0 || vq_dim <= 0 || count <= 0) return;
  dim3 grid((frames + kBlock - 1) / kBlock, vq_dim);
  rvq_gather_sum_transposed_kernel<<<grid, kBlock, 0, st>>>(
      codebooks, codes, out, first_group, count, frames, groups, codebook_size,
      vq_dim);
}

void shift_kv_window_f32(float* kv, int num_slabs, int slot_cap, int width,
                         int src_slot, int keep_len, cudaStream_t st) {
  if (num_slabs <= 0 || keep_len <= 0 || width <= 0) return;
  const size_t elems = static_cast<size_t>(keep_len) * width;
  dim3 grid(static_cast<unsigned>((elems + kBlock - 1) / kBlock), num_slabs);
  shift_kv_window_kernel<<<grid, kBlock, 0, st>>>(kv, num_slabs, slot_cap,
                                                  width, src_slot, keep_len);
}

void sum_tables_f32(float* dst, const float* const* srcs, int count, int rows,
                    int dim, cudaStream_t st) {
  dim3 grid((dim + kBlock - 1) / kBlock, rows);
  sum_tables_kernel<<<grid, kBlock, 0, st>>>(dst, srcs, count, rows, dim);
}

void transpose_f32(const float* in, float* out, int ch, int L,
                   cudaStream_t st) {
  if (ch <= 0 || L <= 0) return;
  // in [ch, L] -> out [L, ch]
  dim3 grid((L + 31) / 32, (ch + 31) / 32);
  dim3 block(32, 8);
  transpose_2d_tiled_kernel<<<grid, block, 0, st>>>(in, out, ch, L);
}

void transpose_inv_f32(const float* in, float* out, int ch, int L,
                       cudaStream_t st) {
  if (ch <= 0 || L <= 0) return;
  // in [L, ch] -> out [ch, L]
  dim3 grid((ch + 31) / 32, (L + 31) / 32);
  dim3 block(32, 8);
  transpose_2d_tiled_kernel<<<grid, block, 0, st>>>(in, out, L, ch);
}

void add_bias_rows_f32(float* y, const float* bias, int rows, int dim,
                       cudaStream_t st) {
  bias_rows_kernel<<<(rows * dim + kBlock - 1) / kBlock, kBlock, 0, st>>>(
      y, bias, rows, dim);
}

void gather_code_column(const int* codes, int* out, int frames, int stride,
                        int group, cudaStream_t st) {
  gather_col_kernel<<<(frames + kBlock - 1) / kBlock, kBlock, 0, st>>>(
      codes, out, frames, stride, group);
}

// Explicit instantiations.
template void rms_norm_t<float>(const float*, const float*, float*, int, int,
                                float, cudaStream_t);
template void rms_norm_t<Bf16>(const Bf16*, const Bf16*, Bf16*, int, int, float,
                               cudaStream_t);
template void layer_norm_t<float>(const float*, const float*, const float*,
                                  float*, int, int, float, cudaStream_t);
template void add_residual_t<float>(float*, const float*, int, cudaStream_t);
template void add_residual_t<Bf16>(Bf16*, const Bf16*, int, cudaStream_t);
template void silu_mul_t<float>(float*, const float*, const float*, int,
                                cudaStream_t);
template void silu_mul_t<Bf16>(Bf16*, const Bf16*, const Bf16*, int,
                               cudaStream_t);
template void silu_inplace_t<float>(float*, int, cudaStream_t);
template void silu_inplace_t<Bf16>(Bf16*, int, cudaStream_t);
template void gelu_inplace_t<float>(float*, int, cudaStream_t);
template void layer_scale_t<float>(float*, const float*, const float*, int, int,
                                   cudaStream_t);
template void gather_rows_t<float>(const float*, const int*, float*, int, int,
                                   cudaStream_t);
template void gather_rows_t<Bf16>(const Bf16*, const int*, Bf16*, int, int,
                                  cudaStream_t);
template void gemm_t<float>(float*, const float*, const float*, const float*,
                            int, int, int, cudaStream_t);
template void gemm_t<Bf16>(Bf16*, const Bf16*, const Bf16*, const Bf16*, int,
                           int, int, cudaStream_t);
template void gemv_t<Bf16>(const Bf16*, const Bf16*, float*, int, int,
                           cudaStream_t);
template void rope_half_t<float>(float*, const int*, const float*, int, int,
                                 int, cudaStream_t);
template void rope_half_t<Bf16>(Bf16*, const int*, const float*, int, int, int,
                                cudaStream_t);
template void gqa_attention_t<float>(const float*, const float*, const float*,
                                     float*, int, int, int, int, int, int, int,
                                     int, cudaStream_t);
template void gqa_attention_t<Bf16>(const Bf16*, const Bf16*, const Bf16*,
                                    Bf16*, int, int, int, int, int, int, int,
                                    int, cudaStream_t);

}  // namespace umminfer::native
