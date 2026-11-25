// SPDX-License-Identifier: Apache-2.0
// BF16 GEMM (prefill) and GEMV (decode) for Qwen3.8-27B.
//
// Same design as native/omni/omni_gemm.cu: 128x128 CTA tiles, 64-deep K
// stages, cp.async double buffering, mma.sync.m16n8k16 with fp32 accumulation;
// decode uses one warp per output row with 128-bit streaming weight loads,
// read-only cached activation loads, K-unrolling, fused multi-projection GEMV,
// and fused SwiGLU dual-output GEMV.
#include "qwen38_ops.hpp"

#include <cuda_bf16.h>

#include <stdexcept>
#include <string>

#include "decode_helpers.hpp"

namespace umminfer::qwen38 {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kBM = 128;
constexpr int kBN = 128;
constexpr int kBK = 64;
constexpr int kBKP = kBK + 8;
constexpr int kThreads = 256;
constexpr int kStages = 2;
constexpr int kTileElems = kBM * kBKP + kBN * kBKP;
constexpr size_t kSmemBytes = kStages * kTileElems * sizeof(bf16);  // 72 KiB

__device__ __forceinline__ uint4 ld_cs_u4(const void* p) {
  return __ldcs(reinterpret_cast<const uint4*>(p));
}
__device__ __forceinline__ uint4 ld_ro_u4(const void* p) {
  return __ldg(reinterpret_cast<const uint4*>(p));
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool pred) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  const unsigned bytes = pred ? 16u : 0u;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(src),
               "r"(bytes));
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
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], const void* p) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(a));
}
__device__ __forceinline__ void mma_m16n8k16(float (&d)[4], const uint32_t (&a)[4],
                                             const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int kRows>
__device__ __forceinline__ void stage_tile(bf16* dst, const bf16* src, int k0,
                                           int k, int valid_rows, int row_base) {
  constexpr int kVec = kBK / 8;
  constexpr int kSteps = kRows * kVec / kThreads;
  #pragma unroll
  for (int i = 0; i < kSteps; ++i) {
    const int idx = threadIdx.x + i * kThreads;
    const int r = idx / kVec;
    const int c = (idx % kVec) * 8;
    const int kk = k0 + c;
    const bool ok = (r < valid_rows) && (kk + 8 <= k);
    const bf16* g = ok ? src + static_cast<size_t>(row_base + r) * k + kk : src;
    cp_async16(dst + r * kBKP + c, g, ok);
  }
}

template <bool kF32Out>
__device__ __forceinline__ void gemm_tile_body(void* cv, const bf16* a,
                                               const bf16* b, const bf16* bias,
                                               int m0, int rows, int n0, int n,
                                               int k, char* smem) {
  bf16* as = reinterpret_cast<bf16*>(smem);
  bf16* bs = as + kStages * kBM * kBKP;
  const int warp = threadIdx.x >> 5;
  const int wm = warp >> 2, wn = warp & 3;
  float acc[4][4][4];
  #pragma unroll
  for (int i = 0; i < 4; ++i)
    #pragma unroll
    for (int j = 0; j < 4; ++j)
      #pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.0f;

  const int nblk = (k + kBK - 1) / kBK;
  auto stage = [&](int kb, int s) {
    stage_tile<kBM>(as + s * kBM * kBKP, a, kb * kBK, k, rows - m0, m0);
    stage_tile<kBN>(bs + s * kBN * kBKP, b, kb * kBK, k, n - n0, n0);
  };
  stage(0, 0);
  cp_commit();
  const int lane = threadIdx.x & 31;
  const int a_row = (lane & 7) + ((lane & 8) ? 8 : 0);
  const int a_col = (lane & 16) ? 8 : 0;
  const int b_row = lane & 7;
  const int b_col = (lane & 8) ? 8 : 0;

  for (int kb = 0; kb < nblk; ++kb) {
    const bool ahead = (kb + 1 < nblk);
    if (ahead) stage(kb + 1, (kb + 1) & 1);
    cp_commit();
    if (ahead) cp_wait<1>(); else cp_wait<0>();
    __syncthreads();
    const bf16* asb = as + (kb & 1) * kBM * kBKP;
    const bf16* bsb = bs + (kb & 1) * kBN * kBKP;
    #pragma unroll
    for (int kk = 0; kk < kBK; kk += 16) {
      uint32_t af[4][4], bfr[4][2];
      #pragma unroll
      for (int mt = 0; mt < 4; ++mt)
        ldmatrix_x4(af[mt], asb + (wm * 64 + mt * 16 + a_row) * kBKP + kk + a_col);
      #pragma unroll
      for (int nt = 0; nt < 4; ++nt)
        ldmatrix_x2(bfr[nt], bsb + (wn * 32 + nt * 8 + b_row) * kBKP + kk + b_col);
      #pragma unroll
      for (int mt = 0; mt < 4; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) mma_m16n8k16(acc[mt][nt], af[mt], bfr[nt]);
    }
    __syncthreads();
  }

  const int r0 = m0 + wm * 64 + (lane >> 2);
  const int c0 = n0 + wn * 32 + (lane & 3) * 2;
  #pragma unroll
  for (int mt = 0; mt < 4; ++mt) {
    #pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int c = c0 + nt * 8;
      if (c >= n) continue;
      float v0 = acc[mt][nt][0], v1 = acc[mt][nt][1];
      float v2 = acc[mt][nt][2], v3 = acc[mt][nt][3];
      if (bias) {
        v0 += __bfloat162float(bias[c]);
        v1 += __bfloat162float(bias[c + 1]);
        v2 += __bfloat162float(bias[c]);
        v3 += __bfloat162float(bias[c + 1]);
      }
      const int r = r0 + mt * 16;
      if (kF32Out) {
        float* out = reinterpret_cast<float*>(cv);
        if (r < rows) {
          out[static_cast<size_t>(r) * n + c] = v0;
          out[static_cast<size_t>(r) * n + c + 1] = v1;
        }
        if (r + 8 < rows) {
          out[static_cast<size_t>(r + 8) * n + c] = v2;
          out[static_cast<size_t>(r + 8) * n + c + 1] = v3;
        }
      } else {
        bf16* out = reinterpret_cast<bf16*>(cv);
        if (r < rows)
          *reinterpret_cast<__nv_bfloat162*>(out + static_cast<size_t>(r) * n + c) =
              __floats2bfloat162_rn(v0, v1);
        if (r + 8 < rows)
          *reinterpret_cast<__nv_bfloat162*>(
              out + static_cast<size_t>(r + 8) * n + c) =
              __floats2bfloat162_rn(v2, v3);
      }
    }
  }
}

__global__ void __launch_bounds__(kThreads, 2) gemm_mma_kernel(
    bf16* __restrict__ c, const bf16* __restrict__ a, const bf16* __restrict__ b,
    const bf16* __restrict__ bias, int m, int k, int n) {
  extern __shared__ char smem[];
  const GemmTileCoord coord = swizzle_gemm_cta(
      static_cast<int>(blockIdx.x), static_cast<int>(blockIdx.y),
      static_cast<int>(gridDim.x), static_cast<int>(gridDim.y));
  const int m0 = coord.tile_m * kBM, n0 = coord.tile_n * kBN;
  gemm_tile_body<false>(c, a, b, bias, m0, m, n0, n, k, smem);
}

__global__ void __launch_bounds__(kThreads, 2) gemm_mma_f32_kernel(
    float* __restrict__ c, const bf16* __restrict__ a, const bf16* __restrict__ b,
    int m, int k, int n) {
  extern __shared__ char smem[];
  const GemmTileCoord coord = swizzle_gemm_cta(
      static_cast<int>(blockIdx.x), static_cast<int>(blockIdx.y),
      static_cast<int>(gridDim.x), static_cast<int>(gridDim.y));
  const int m0 = coord.tile_m * kBM, n0 = coord.tile_n * kBN;
  gemm_tile_body<true>(c, a, b, nullptr, m0, m, n0, n, k, smem);
}

bool g_smem_configured = false;
void configure_smem() {
  if (g_smem_configured) return;
  int dev = 0;
  cudaGetDevice(&dev);
  int max_smem = 0;
  cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  if (static_cast<size_t>(max_smem) < kSmemBytes)
    throw std::runtime_error("device cannot host the 72 KiB GEMM tile");
  for (const void* f :
       {reinterpret_cast<const void*>(&gemm_mma_kernel),
        reinterpret_cast<const void*>(&gemm_mma_f32_kernel)}) {
    cudaError_t err = cudaFuncSetAttribute(
        f, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmemBytes));
    if (err != cudaSuccess)
      throw std::runtime_error(std::string("cudaFuncSetAttribute: ") +
                               cudaGetErrorString(err));
  }
  g_smem_configured = true;
}

// Warp-level K-dot helper shared by single-matrix and multi-projection GEMV.
// Uses streaming cache hints (__ldcs) on weights so large weight matrices do
// not evict the small shared activation rows x[0..kM-1] from L1/L2, and
// read-only cache loads (__ldg) on x.
//
// Bitwise equality with the naive warp GEMV is preserved: lane L accumulates
// chunks (L*8 + 256*t) in strictly increasing order with the exact same
// `acc[b] += a*b + c*d` expression and warp shuffle reduction tree.
template <int kM, int kUnroll>
__device__ __forceinline__ void gemv_warp_dot(const bf16* __restrict__ wr,
                                              const bf16* __restrict__ x, int k,
                                              int lane, float (&acc)[kM]) {
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

template <int kM, int kUnroll, typename OutT>
__global__ void __launch_bounds__(128) gemv_small_m_kernel(
    const bf16* __restrict__ w, const bf16* __restrict__ x,
    OutT* __restrict__ out, int n_out, int k) {
  const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (row >= n_out) return;
  const int lane = threadIdx.x & 31;
  const bf16* wr = w + static_cast<size_t>(row) * k;

  float acc[kM];
  gemv_warp_dot<kM, kUnroll>(wr, x, k, lane, acc);

  if (lane == 0) {
    #pragma unroll
    for (int b = 0; b < kM; ++b) {
      const size_t o = static_cast<size_t>(b) * n_out + row;
      if constexpr (sizeof(OutT) == 4) out[o] = acc[b];
      else out[o] = static_cast<OutT>(__float2bfloat16(acc[b]));
    }
  }
}

// Multi-projection GEMV: fuses up to 4 projections sharing `x` into one grid so
// tiny projections (e.g. in_proj_a/b with 48 rows, k/v with 1024 rows) ride
// alongside large projections with zero tail-effect and zero extra weight VRAM.
template <int kM, int kUnroll>
__global__ void __launch_bounds__(128) gemv_multi_small_m_kernel(
    MultiGemvDesc desc, const bf16* __restrict__ x, int k) {
  const int global_row = blockIdx.x * 4 + (threadIdx.x >> 5);
  const MultiGemvRowLoc loc =
      multi_gemv_locate_row(global_row, desc.n_out, desc.count);
  if (loc.proj < 0) return;

  const int lane = threadIdx.x & 31;
  const bf16* wr = reinterpret_cast<const bf16*>(desc.w[loc.proj]) +
                   static_cast<size_t>(loc.local_row) * k;
  bf16* out_base = reinterpret_cast<bf16*>(desc.out[loc.proj]);
  const int out_stride = desc.n_out[loc.proj];

  float acc[kM];
  gemv_warp_dot<kM, kUnroll>(wr, x, k, lane, acc);

  if (lane == 0) {
    #pragma unroll
    for (int b = 0; b < kM; ++b) {
      out_base[static_cast<size_t>(b) * out_stride + loc.local_row] =
          __float2bfloat16(acc[b]);
    }
  }
}

// Fused dual-output SwiGLU GEMV: computes both gate_proj and up_proj rows while
// loading activation vector x[b] only once, then applies the exact reference
// bf16 rounding for silu(gate) * up.
template <int kM, int kUnroll>
__global__ void __launch_bounds__(128) gemv_swiglu_small_m_kernel(
    const bf16* __restrict__ w_gate, const bf16* __restrict__ w_up,
    const bf16* __restrict__ x, bf16* __restrict__ out, int n_out, int k) {
  constexpr int kStr = 32 * 8;
  const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (row >= n_out) return;
  const int lane = threadIdx.x & 31;
  const bf16* wgr = w_gate + static_cast<size_t>(row) * k;
  const bf16* wur = w_up + static_cast<size_t>(row) * k;

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
        xv[b] = ld_ro_u4(x + static_cast<size_t>(b) * k + i + u * kStr);
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
        const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&xv[b]);
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 fx = __bfloat1622float2(xb[e]);
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
      const uint4 xu = ld_ro_u4(x + static_cast<size_t>(b) * k + i);
      const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&xu);
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 f2g = __bfloat1622float2(vgb[e]);
        const float2 f2u = __bfloat1622float2(vub[e]);
        const float2 fx = __bfloat1622float2(xb[e]);
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
      const float xf = __bfloat162float(x[static_cast<size_t>(b) * k + i]);
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
      const float sg = g / (1.0f + expf(-g));
      out[static_cast<size_t>(b) * n_out + row] =
          __float2bfloat16(__bfloat162float(__float2bfloat16(sg)) * u);
    }
  }
}

template <int kM, typename OutT>
void launch_small_m(const bf16* w, const bf16* x, OutT* out, int n_out, int k,
                    cudaStream_t stream) {
  const int blocks = (n_out + 3) / 4;
  if constexpr (gemv_unroll_for(kM) == 4)
    gemv_small_m_kernel<kM, 4, OutT>
        <<<blocks, 128, 0, stream>>>(w, x, out, n_out, k);
  else
    gemv_small_m_kernel<kM, 1, OutT>
        <<<blocks, 128, 0, stream>>>(w, x, out, n_out, k);
}

template <typename OutT>
void launch_small_m_dyn(const bf16* w, const bf16* x, OutT* out, int m,
                        int n_out, int k, cudaStream_t stream) {
  switch (m) {
    case 1: launch_small_m<1, OutT>(w, x, out, n_out, k, stream); break;
    case 2: launch_small_m<2, OutT>(w, x, out, n_out, k, stream); break;
    case 3: launch_small_m<3, OutT>(w, x, out, n_out, k, stream); break;
    case 4: launch_small_m<4, OutT>(w, x, out, n_out, k, stream); break;
    case 5: launch_small_m<5, OutT>(w, x, out, n_out, k, stream); break;
    case 6: launch_small_m<6, OutT>(w, x, out, n_out, k, stream); break;
    case 7: launch_small_m<7, OutT>(w, x, out, n_out, k, stream); break;
    default: launch_small_m<8, OutT>(w, x, out, n_out, k, stream); break;
  }
}

template <int kM>
void launch_multi_m(const MultiGemvDesc& desc, const bf16* x, int total_n,
                    int k, cudaStream_t stream) {
  const int blocks = (total_n + 3) / 4;
  if constexpr (gemv_unroll_for(kM) == 4)
    gemv_multi_small_m_kernel<kM, 4><<<blocks, 128, 0, stream>>>(desc, x, k);
  else
    gemv_multi_small_m_kernel<kM, 1><<<blocks, 128, 0, stream>>>(desc, x, k);
}

template <int kM>
void launch_swiglu_m(const bf16* w_gate, const bf16* w_up, const bf16* x,
                     bf16* out, int n_out, int k, cudaStream_t stream) {
  const int blocks = (n_out + 3) / 4;
  if constexpr (swiglu_gemv_unroll_for(kM) == 2)
    gemv_swiglu_small_m_kernel<kM, 2>
        <<<blocks, 128, 0, stream>>>(w_gate, w_up, x, out, n_out, k);
  else
    gemv_swiglu_small_m_kernel<kM, 1>
        <<<blocks, 128, 0, stream>>>(w_gate, w_up, x, out, n_out, k);
}

}  // namespace

void gemm_bf16(Bf16* y, const Bf16* x, const Bf16* w, const Bf16* bias, int m,
               int k, int n, cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0) return;
  configure_smem();
  dim3 grid((n + kBN - 1) / kBN, (m + kBM - 1) / kBM);
  gemm_mma_kernel<<<grid, kThreads, kSmemBytes, stream>>>(
      reinterpret_cast<bf16*>(y), reinterpret_cast<const bf16*>(x),
      reinterpret_cast<const bf16*>(w), reinterpret_cast<const bf16*>(bias), m, k,
      n);
}

void gemm_bf16_f32(float* y, const Bf16* x, const Bf16* w, int m, int k, int n,
                   cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0) return;
  configure_smem();
  dim3 grid((n + kBN - 1) / kBN, (m + kBM - 1) / kBM);
  gemm_mma_f32_kernel<<<grid, kThreads, kSmemBytes, stream>>>(
      y, reinterpret_cast<const bf16*>(x), reinterpret_cast<const bf16*>(w), m, k, n);
}

void gemv_bf16_f32(const Bf16* w, const Bf16* x, float* out, int n_out, int k,
                   cudaStream_t stream) {
  if (n_out <= 0 || k <= 0) return;
  launch_small_m<1, float>(reinterpret_cast<const bf16*>(w),
                           reinterpret_cast<const bf16*>(x), out, n_out, k,
                           stream);
}

void gemv_bf16_bf16(const Bf16* w, const Bf16* x, Bf16* out, int n_out, int k,
                    cudaStream_t stream) {
  if (n_out <= 0 || k <= 0) return;
  launch_small_m<1, bf16>(reinterpret_cast<const bf16*>(w),
                          reinterpret_cast<const bf16*>(x),
                          reinterpret_cast<bf16*>(out), n_out, k, stream);
}

void gemv_bf16_f32_batch(const Bf16* w, const Bf16* x, float* out, int m,
                         int n_out, int k, cudaStream_t stream) {
  if (m <= 0 || n_out <= 0 || k <= 0) return;
  launch_small_m_dyn(reinterpret_cast<const bf16*>(w),
                     reinterpret_cast<const bf16*>(x), out, m, n_out, k,
                     stream);
}

void gemv_bf16_bf16_batch(const Bf16* w, const Bf16* x, Bf16* out, int m,
                          int n_out, int k, cudaStream_t stream) {
  if (m <= 0 || n_out <= 0 || k <= 0) return;
  launch_small_m_dyn(reinterpret_cast<const bf16*>(w),
                     reinterpret_cast<const bf16*>(x),
                     reinterpret_cast<bf16*>(out), m, n_out, k, stream);
}

void gemv_bf16_bf16_multi_batch(const MultiGemvDesc& desc, const Bf16* x, int m,
                                int k, cudaStream_t stream) {
  if (m <= 0 || k <= 0 || desc.count <= 0) return;
  int total_n = 0;
  for (int p = 0; p < desc.count; ++p) total_n += desc.n_out[p];
  if (total_n <= 0) return;
  const bf16* xb = reinterpret_cast<const bf16*>(x);
  switch (m) {
    case 1: launch_multi_m<1>(desc, xb, total_n, k, stream); break;
    case 2: launch_multi_m<2>(desc, xb, total_n, k, stream); break;
    case 3: launch_multi_m<3>(desc, xb, total_n, k, stream); break;
    case 4: launch_multi_m<4>(desc, xb, total_n, k, stream); break;
    case 5: launch_multi_m<5>(desc, xb, total_n, k, stream); break;
    case 6: launch_multi_m<6>(desc, xb, total_n, k, stream); break;
    case 7: launch_multi_m<7>(desc, xb, total_n, k, stream); break;
    default: launch_multi_m<8>(desc, xb, total_n, k, stream); break;
  }
}

void gemv_bf16_swiglu_batch(const Bf16* w_gate, const Bf16* w_up, const Bf16* x,
                            Bf16* out, int m, int n_out, int k,
                            cudaStream_t stream) {
  if (m <= 0 || n_out <= 0 || k <= 0) return;
  const bf16* wg = reinterpret_cast<const bf16*>(w_gate);
  const bf16* wu = reinterpret_cast<const bf16*>(w_up);
  const bf16* xb = reinterpret_cast<const bf16*>(x);
  bf16* ob = reinterpret_cast<bf16*>(out);
  switch (m) {
    case 1: launch_swiglu_m<1>(wg, wu, xb, ob, n_out, k, stream); break;
    case 2: launch_swiglu_m<2>(wg, wu, xb, ob, n_out, k, stream); break;
    case 3: launch_swiglu_m<3>(wg, wu, xb, ob, n_out, k, stream); break;
    case 4: launch_swiglu_m<4>(wg, wu, xb, ob, n_out, k, stream); break;
    case 5: launch_swiglu_m<5>(wg, wu, xb, ob, n_out, k, stream); break;
    case 6: launch_swiglu_m<6>(wg, wu, xb, ob, n_out, k, stream); break;
    case 7: launch_swiglu_m<7>(wg, wu, xb, ob, n_out, k, stream); break;
    default: launch_swiglu_m<8>(wg, wu, xb, ob, n_out, k, stream); break;
  }
}

}  // namespace umminfer::qwen38
