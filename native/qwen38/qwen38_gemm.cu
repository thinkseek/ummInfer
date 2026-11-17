// SPDX-License-Identifier: Apache-2.0
// BF16 GEMM (prefill) and GEMV (decode) for Qwen3.8-27B.
//
// Same design as native/omni/omni_gemm.cu: 128x128 CTA tiles, 64-deep K
// stages, cp.async double buffering, mma.sync.m16n8k16 with fp32 accumulation;
// batch-1 uses one warp per output row with 16-byte loads.  The MoE-specific
// kernels of the Omni slice are not needed here and were dropped.
#include "qwen38_ops.hpp"

#include <cuda_bf16.h>

#include <stdexcept>
#include <string>

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
  const int m0 = blockIdx.y * kBM, n0 = blockIdx.x * kBN;
  gemm_tile_body<false>(c, a, b, bias, m0, m, n0, n, k, smem);
}

__global__ void __launch_bounds__(kThreads, 2) gemm_mma_f32_kernel(
    float* __restrict__ c, const bf16* __restrict__ a, const bf16* __restrict__ b,
    int m, int k, int n) {
  extern __shared__ char smem[];
  const int m0 = blockIdx.y * kBM, n0 = blockIdx.x * kBN;
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

template <typename OutT>
__global__ void __launch_bounds__(128) gemv_kernel(const bf16* __restrict__ w,
                                                   const bf16* __restrict__ x,
                                                   OutT* __restrict__ out,
                                                   int n_out, int k) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = blockIdx.x * 4 + warp;
  if (row >= n_out) return;
  const bf16* wr = w + static_cast<size_t>(row) * k;
  float acc = 0.0f;
  const int step = 32 * 8;
  int i = lane * 8;
  for (; i + 8 <= k; i += step) {
    const uint4 v = *reinterpret_cast<const uint4*>(wr + i);
    const uint4 u = *reinterpret_cast<const uint4*>(x + i);
    const __nv_bfloat162* vb = reinterpret_cast<const __nv_bfloat162*>(&v);
    const __nv_bfloat162* ub = reinterpret_cast<const __nv_bfloat162*>(&u);
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 fv = __bfloat1622float2(vb[e]);
      const float2 fu = __bfloat1622float2(ub[e]);
      acc += fv.x * fu.x + fv.y * fu.y;
    }
  }
  for (; i < k; ++i) acc += __bfloat162float(wr[i]) * __bfloat162float(x[i]);
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) {
    if (sizeof(OutT) == 4) out[row] = acc;
    else out[row] = static_cast<OutT>(__float2bfloat16(acc));
  }
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
  gemv_kernel<float><<<(n_out + 3) / 4, 128, 0, stream>>>(
      reinterpret_cast<const bf16*>(w), reinterpret_cast<const bf16*>(x), out,
      n_out, k);
}

void gemv_bf16_bf16(const Bf16* w, const Bf16* x, Bf16* out, int n_out, int k,
                    cudaStream_t stream) {
  gemv_kernel<bf16><<<(n_out + 3) / 4, 128, 0, stream>>>(
      reinterpret_cast<const bf16*>(w), reinterpret_cast<const bf16*>(x),
      reinterpret_cast<bf16*>(out), n_out, k);
}

}  // namespace umminfer::qwen38
