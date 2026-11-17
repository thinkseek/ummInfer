// SPDX-License-Identifier: Apache-2.0
// BF16 matmul kernels for the native Qwen3-Omni slice.
//
// Prefill: warp-level mma.sync.m16n8k16 (BF16 in, FP32 out) on 128x128 CTA
// tiles with 64-deep K stages, cp.async double buffering and ldmatrix operand
// loads.  Measured instruction peak on this H20 is 95 TFLOPS; the kernels here
// are written to reach the same code path for every projection in the model.
//
// Decode: batch-1 runs are bound by weight bytes, not FLOPs, so they use a
// dedicated GEMV (one warp per output row, 16-byte loads, fp32 accumulation)
// instead of padding a 128-row MMA tile.
//
// MoE: the router already round-trips to the host once per layer, so the
// (token, expert) pairs are sorted by ascending expert id there.  Each
// expert's pairs become one contiguous row block, which turns the per-expert
// GEMV chain into a grouped GEMM whose weights are streamed exactly once per
// prefill pass.  Pair outputs are kept in fp32 and combined in the reference
// order (ascending expert, bf16 rounding after each add).
#include "omni_gemm.hpp"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "cuda_kernels_omni.hpp"

namespace umminfer::omni {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kBM = 128;
constexpr int kBN = 128;
constexpr int kBK = 64;
constexpr int kBKP = kBK + 8;  // 144-byte row stride: 16B aligned, conflict-free
constexpr int kThreads = 256;
constexpr int kStages = 2;
constexpr int kTileElems = kBM * kBKP + kBN * kBKP;
constexpr size_t kSmemBytes = kStages * kTileElems * sizeof(bf16);  // 73728 B

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

// Stage one [ROWS, kBK] tile of a K-contiguous matrix into shared memory.
// Rows >= valid_rows and K columns past the end are zero-filled, so the MMA
// never needs a tail path.
template <int kRows>
__device__ __forceinline__ void stage_tile(bf16* dst, const bf16* src, int k0,
                                           int k, int valid_rows, int row_base) {
  constexpr int kVec = kBK / 8;                 // 8 bf16 per 16B vector
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

// Full tile body: one 128x128 output tile at (m0, n0) with `rows` valid rows.
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
  // `rows` is the absolute valid row count; stage_tile wants rows from the tile
  // base so that nothing past the matrix end is ever read.
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

  // Store.  Thread lane holds cols (lane&3)*2, +1 of rows lane>>2 (+8) of each
  // 16x8 fragment.
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

// Grouped MoE tile: the flat CTA list supplies (expert, m-tile, n-tile).
template <bool kF32Out>
__global__ void __launch_bounds__(kThreads, 2) moe_gemm_kernel(
    void* __restrict__ c, const bf16* __restrict__ a,
    const bf16* const* __restrict__ wtbl, const int* __restrict__ g_off,
    const int* __restrict__ g_size, const int* __restrict__ blk_e,
    const int* __restrict__ blk_m, const int* __restrict__ blk_n, int base,
    int blk_off, int n, int k) {
  extern __shared__ char smem[];
  const int bid = blk_off + blockIdx.x;
  const int e = blk_e[bid];
  const int rows = g_size[e];
  const int m0 = g_off[e] + blk_m[bid] * kBM;
  const int n0 = blk_n[bid] * kBN;
  gemm_tile_body<kF32Out>(c, a, wtbl[base + e], nullptr, m0, g_off[e] + rows, n0,
                          n, k, smem);
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
        reinterpret_cast<const void*>(&gemm_mma_f32_kernel),
        reinterpret_cast<const void*>(&moe_gemm_kernel<true>)}) {
    cudaError_t err = cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           static_cast<int>(kSmemBytes));
    if (err != cudaSuccess)
      throw std::runtime_error(std::string("cudaFuncSetAttribute: ") +
                               cudaGetErrorString(err));
  }
  g_smem_configured = true;
}

// ---- GEMV -----------------------------------------------------------------

// One warp per output row.  lane strides 16-byte vectors; fp32 partials per
// lane are reduced across the warp.
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

__global__ void __launch_bounds__(128) gemv_grouped_kernel(
    const bf16* const* __restrict__ wtbl, const bf16* __restrict__ x,
    float* __restrict__ out, int n_out, int k, int rows_per_blk) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = blockIdx.x / rows_per_blk;
  const int local = blockIdx.x % rows_per_blk;
  const int row = local * 4 + warp;
  if (row >= n_out) return;
  const bf16* wr = wtbl[g] + static_cast<size_t>(row) * k;
  const bf16* xg = x + static_cast<size_t>(g) * k;
  float acc = 0.0f;
  const int step = 32 * 8;
  int i = lane * 8;
  for (; i + 8 <= k; i += step) {
    const uint4 v = *reinterpret_cast<const uint4*>(wr + i);
    const uint4 u = *reinterpret_cast<const uint4*>(xg + i);
    const __nv_bfloat162* vb = reinterpret_cast<const __nv_bfloat162*>(&v);
    const __nv_bfloat162* ub = reinterpret_cast<const __nv_bfloat162*>(&u);
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 fv = __bfloat1622float2(vb[e]);
      const float2 fu = __bfloat1622float2(ub[e]);
      acc += fv.x * fu.x + fv.y * fu.y;
    }
  }
  for (; i < k; ++i) acc += __bfloat162float(wr[i]) * __bfloat162float(xg[i]);
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) out[static_cast<size_t>(g) * n_out + row] = acc;
}

// ---- FP32-activation down projection -------------------------------------
// The reference down projection multiplies the expert activation in FP32.  A
// tensor-core mma needs BF16 operands, so this projection uses CUDA-core FP32
// FMA instead: the products are bit-identical to the reference and only the
// accumulation order differs.  Weights are upcast from BF16 (lossless).

// One CTA computes a [kDownTileM pairs x 128 outputs] tile for one expert.
__global__ void __launch_bounds__(256) moe_down_kernel(
    float* __restrict__ c, const float* __restrict__ a,
    const bf16* const* __restrict__ wtbl, const int* __restrict__ g_off,
    const int* __restrict__ g_size, const int* __restrict__ blk_e,
    const int* __restrict__ blk_m, const int* __restrict__ blk_n, int base,
    int n, int k) {
  constexpr int kMT = kDownTileM;
  constexpr int kNT = 128;
  constexpr int kKT = 64;
  __shared__ float as[kKT][kMT];
  __shared__ bf16 bs[kKT][kNT];

  const int bid = blockIdx.x;
  const int e = blk_e[bid];
  const int mt = blk_m[bid];
  const int rows = g_size[e] - mt * kMT;  // valid pairs from this tile's base
  const int m0 = g_off[e] + mt * kMT;
  const int n0 = blk_n[bid] * kNT;

  const int tm = threadIdx.x >> 4;   // pair row within the tile
  const int tn = threadIdx.x & 15;   // 8-output group
  const int ncol = n0 + tn * 8;

  float acc[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = 0.0f;

  const bf16* w = wtbl[base + e];
  const int nb = (k + kKT - 1) / kKT;
  for (int kb = 0; kb < nb; ++kb) {
    const int k0 = kb * kKT;
    for (int i = threadIdx.x; i < kMT * kKT; i += 256) {
      const int r = i / kKT, cc = i % kKT;
      as[cc][r] = (r < rows && k0 + cc < k)
                      ? a[static_cast<size_t>(m0 + r) * k + k0 + cc]
                      : 0.0f;
    }
    for (int i = threadIdx.x; i < kNT * (kKT / 8); i += 256) {
      const int rn = i / (kKT / 8), kg = (i % (kKT / 8)) * 8;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (n0 + rn < n && k0 + kg < k)
        v = *reinterpret_cast<const uint4*>(w + static_cast<size_t>(n0 + rn) * k +
                                            k0 + kg);
      const bf16* vb = reinterpret_cast<const bf16*>(&v);
      #pragma unroll
      for (int j = 0; j < 8; ++j) bs[kg + j][rn] = vb[j];
    }
    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < kKT; ++kk) {
      const float av = as[kk][tm];
      const uint4 bv = *reinterpret_cast<const uint4*>(&bs[kk][tn * 8]);
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&bv);
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(h[j]);
        acc[2 * j] += av * f.x;
        acc[2 * j + 1] += av * f.y;
      }
    }
    __syncthreads();
  }

  const int r = m0 + tm;
  if (tm < rows)
    #pragma unroll
    for (int j = 0; j < 8; ++j)
      if (ncol + j < n) c[static_cast<size_t>(r) * n + ncol + j] = acc[j];
}

// Batch-1 variant: one warp per output row, FP32 activation vector.
__global__ void __launch_bounds__(128) gemv_grouped_f32act_kernel(
    const bf16* const* __restrict__ wtbl, const float* __restrict__ x,
    float* __restrict__ out, int n_out, int k, int rows_per_blk) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = blockIdx.x / rows_per_blk;
  const int local = blockIdx.x % rows_per_blk;
  const int row = local * 4 + warp;
  if (row >= n_out) return;
  const bf16* wr = wtbl[g] + static_cast<size_t>(row) * k;
  const float* xg = x + static_cast<size_t>(g) * k;
  float acc = 0.0f;
  const int step = 32 * 8;
  int i = lane * 8;
  for (; i + 8 <= k; i += step) {
    const uint4 v = *reinterpret_cast<const uint4*>(wr + i);
    const __nv_bfloat162* vb = reinterpret_cast<const __nv_bfloat162*>(&v);
    const float4 xa = *reinterpret_cast<const float4*>(xg + i);
    const float4 xb = *reinterpret_cast<const float4*>(xg + i + 4);
    const float xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 fv = __bfloat1622float2(vb[e]);
      acc += fv.x * xs[2 * e] + fv.y * xs[2 * e + 1];
    }
  }
  for (; i < k; ++i) acc += __bfloat162float(wr[i]) * xg[i];
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) out[static_cast<size_t>(g) * n_out + row] = acc;
}

// ---- small elementwise helpers -------------------------------------------

__global__ void gather_rows_kernel(bf16* __restrict__ out,
                                   const bf16* __restrict__ in,
                                   const int* __restrict__ ids, int dim) {
  const int row = blockIdx.x;
  const bf16* src = in + static_cast<size_t>(ids[row]) * dim;
  bf16* dst = out + static_cast<size_t>(row) * dim;
  for (int i = threadIdx.x * 8; i + 8 <= dim; i += blockDim.x * 8)
    *reinterpret_cast<uint4*>(dst + i) = *reinterpret_cast<const uint4*>(src + i);
}

__global__ void moe_act_kernel(float* __restrict__ act, const float* __restrict__ g,
                               const float* __restrict__ u, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Exactly the reference expression, kept in FP32 for the down projection.
  if (i < n) act[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
}

// Reference MoE combine: acc[o] = bf16(bf16(acc[o]) + w_k * y_k[o]) over the
// token's experts in ascending id order, acc starting at zero.
__global__ void moe_accumulate_kernel(bf16* __restrict__ acc,
                                      const float* __restrict__ y,
                                      const int* __restrict__ pair_of,
                                      const float* __restrict__ pair_w, int top_k,
                                      int dim) {
  const int row = blockIdx.y;
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= dim) return;
  float a = 0.0f;
  for (int k = 0; k < top_k; ++k) {
    const int p = pair_of[row * top_k + k];
    a = __bfloat162float(
        __float2bfloat16(a + pair_w[row * top_k + k] * y[static_cast<size_t>(p) * dim + o]));
  }
  acc[static_cast<size_t>(row) * dim + o] = __float2bfloat16(a);
}

}  // namespace

// ---- host entry points ----------------------------------------------------

void gemm_bf16_mma(Bf16* y, const Bf16* x, const Bf16* w, const Bf16* bias,
                   int m, int k, int n, cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0) return;
  configure_smem();
  dim3 grid((n + kBN - 1) / kBN, (m + kBM - 1) / kBM);
  gemm_mma_kernel<<<grid, kThreads, kSmemBytes, stream>>>(
      reinterpret_cast<bf16*>(y), reinterpret_cast<const bf16*>(x),
      reinterpret_cast<const bf16*>(w), reinterpret_cast<const bf16*>(bias), m, k,
      n);
}

void gemm_bf16_mma_f32(float* y, const Bf16* x, const Bf16* w, int m, int k,
                       int n, cudaStream_t stream) {
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

void gemv_grouped_f32(const Bf16* const* wtbl, const Bf16* x, float* out,
                      int n_out, int k, int n_groups, cudaStream_t stream) {
  const int rpb = (n_out + 3) / 4;
  gemv_grouped_kernel<<<rpb * n_groups, 128, 0, stream>>>(
      reinterpret_cast<const bf16* const*>(wtbl), reinterpret_cast<const bf16*>(x),
      out, n_out, k, rpb);
}

void MoeGroupPlan::build(const int* top_idx, const float* top_w, int n_tokens,
                         int k_top, int num_experts) {
  tokens = n_tokens;
  top_k = k_top;
  std::vector<std::pair<int, int>> pairs;  // (expert, token)
  pairs.reserve(static_cast<size_t>(n_tokens) * k_top);
  for (int t = 0; t < n_tokens; ++t)
    for (int k = 0; k < k_top; ++k) pairs.emplace_back(top_idx[t * k_top + k], t);
  // Ascending expert id then ascending token: gives contiguous per-expert row
  // blocks and the reference ascending-expert accumulation order.
  std::sort(pairs.begin(), pairs.end());
  total_pairs = static_cast<int>(pairs.size());
  group_off.assign(num_experts, 0);
  group_size.assign(num_experts, 0);
  pair_token.assign(total_pairs, 0);
  pair_expert.assign(total_pairs, 0);
  pair_of.assign(static_cast<size_t>(n_tokens) * k_top, 0);
  pair_w.assign(static_cast<size_t>(n_tokens) * k_top, 0.0f);
  std::vector<int> filled(static_cast<size_t>(n_tokens) * k_top, 0);
  for (int p = 0; p < total_pairs; ++p) {
    const int e = pairs[p].first, t = pairs[p].second;
    group_size[e]++;
    pair_token[p] = t;
    pair_expert[p] = e;
    const int slot = filled[t * k_top]++;
    pair_of[t * k_top + slot] = p;
    for (int k = 0; k < k_top; ++k)
      if (top_idx[t * k_top + k] == e) {
        pair_w[t * k_top + slot] = top_w[t * k_top + k];
        break;
      }
  }
  int off = 0;
  for (int e = 0; e < num_experts; ++e) {
    group_off[e] = off;
    off += group_size[e];
  }
}

void MoeGroupPlan::build_blocks(int n_tiles) {
  blk_expert.clear();
  blk_m.clear();
  blk_n.clear();
  const int n_experts = static_cast<int>(group_size.size());
  for (int e = 0; e < n_experts; ++e) {
    const int mt = (group_size[e] + kBM - 1) / kBM;
    for (int m = 0; m < mt; ++m)
      for (int t = 0; t < n_tiles; ++t) {
        blk_expert.push_back(e);
        blk_m.push_back(m);
        blk_n.push_back(t);
      }
  }
}

void MoeGroupPlan::build_down_blocks(int n_tiles) {
  down_e.clear();
  down_m.clear();
  down_n.clear();
  const int n_experts = static_cast<int>(group_size.size());
  for (int e = 0; e < n_experts; ++e) {
    const int mt = (group_size[e] + kDownTileM - 1) / kDownTileM;
    for (int m = 0; m < mt; ++m)
      for (int t = 0; t < n_tiles; ++t) {
        down_e.push_back(e);
        down_m.push_back(m);
        down_n.push_back(t);
      }
  }
}

void moe_upload_tables(const MoeGroupPlan& plan, const Bf16* const wg[],
                       const Bf16* const wu[], const Bf16* const wd[],
                       int num_experts, MoeDeviceTables* t, cudaStream_t stream) {
  std::vector<const Bf16*> w(static_cast<size_t>(3) * num_experts);
  for (int e = 0; e < num_experts; ++e) {
    w[e] = wg[e];
    w[num_experts + e] = wu[e];
    w[2 * num_experts + e] = wd[e];
  }
  cudaMemcpyAsync(t->weights, w.data(), w.size() * sizeof(const Bf16*),
                  cudaMemcpyHostToDevice, stream);
  // Pair-indexed view for the batch-1 GEMV path, which must touch only the
  // experts this token actually routes to.
  std::vector<const Bf16*> pw(static_cast<size_t>(3) * plan.total_pairs);
  const int np = plan.total_pairs;
  for (int p = 0; p < np; ++p) {
    const int e = plan.pair_expert[p];
    pw[p] = wg[e];
    pw[np + p] = wu[e];
    pw[2 * np + p] = wd[e];
  }
  cudaMemcpyAsync(t->pair_weights, pw.data(), pw.size() * sizeof(const Bf16*),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(t->pair_token, plan.pair_token.data(),
                  plan.pair_token.size() * sizeof(int), cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(t->pair_of, plan.pair_of.data(),
                  plan.pair_of.size() * sizeof(int), cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(t->pair_w, plan.pair_w.data(),
                  plan.pair_w.size() * sizeof(float), cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(t->group_off, plan.group_off.data(),
                  plan.group_off.size() * sizeof(int), cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(t->group_size, plan.group_size.data(),
                  plan.group_size.size() * sizeof(int), cudaMemcpyHostToDevice,
                  stream);
  const size_t nb = plan.blk_expert.size();
  if (static_cast<int>(nb) > t->blk_cap || 
      static_cast<int>(plan.down_e.size()) > t->down_cap)
    throw std::runtime_error("MoE CTA list exceeds its device buffer");
  cudaMemcpyAsync(t->down_e, plan.down_e.data(),
                  plan.down_e.size() * sizeof(int), cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(t->down_m, plan.down_m.data(),
                  plan.down_m.size() * sizeof(int), cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(t->down_n, plan.down_n.data(),
                  plan.down_n.size() * sizeof(int), cudaMemcpyHostToDevice,
                  stream);
  if (nb) {
    cudaMemcpyAsync(t->blk_e, plan.blk_expert.data(), nb * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(t->blk_m, plan.blk_m.data(), nb * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(t->blk_n, plan.blk_n.data(), nb * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
  }
}

void moe_gemm_grouped_f32(float* y, const Bf16* x, const MoeDeviceTables& t,
                          int base, int n_blocks, int n, int k,
                          cudaStream_t stream) {
  if (n_blocks <= 0) return;
  configure_smem();
  moe_gemm_kernel<true><<<n_blocks, kThreads, kSmemBytes, stream>>>(
      reinterpret_cast<void*>(y), reinterpret_cast<const bf16*>(x),
      reinterpret_cast<const bf16* const*>(t.weights), t.group_off, t.group_size,
      t.blk_e, t.blk_m, t.blk_n, base, 0, n, k);
}

void moe_down_f32(float* y, const float* act, const MoeDeviceTables& t, int base,
                  int n_blocks, int n, int k, cudaStream_t stream) {
  if (n_blocks <= 0) return;
  moe_down_kernel<<<n_blocks, 256, 0, stream>>>(
      y, act, reinterpret_cast<const bf16* const*>(t.weights), t.group_off,
      t.group_size, t.down_e, t.down_m, t.down_n, base, n, k);
}

void gather_rows(Bf16* out, const Bf16* in, const int* ids, int n_rows, int dim,
                 cudaStream_t stream) {
  gather_rows_kernel<<<n_rows, 128, 0, stream>>>(
      reinterpret_cast<bf16*>(out), reinterpret_cast<const bf16*>(in), ids, dim);
}

void moe_act(float* act, const float* gate, const float* up, int n,
             cudaStream_t stream) {
  moe_act_kernel<<<(n + 255) / 256, 256, 0, stream>>>(act, gate, up, n);
}

void gemv_grouped_f32act(const Bf16* const* wtbl, const float* x, float* out,
                         int n_out, int k, int n_groups, cudaStream_t stream) {
  const int rpb = (n_out + 3) / 4;
  gemv_grouped_f32act_kernel<<<rpb * n_groups, 128, 0, stream>>>(
      reinterpret_cast<const bf16* const*>(wtbl), x, out, n_out, k, rpb);
}

void moe_accumulate(Bf16* acc, const float* y, const int* pair_of,
                    const float* pair_w, int tokens, int top_k, int dim,
                    cudaStream_t stream) {
  dim3 grid((dim + 255) / 256, tokens);
  moe_accumulate_kernel<<<grid, 256, 0, stream>>>(
      reinterpret_cast<bf16*>(acc), y, pair_of, pair_w, top_k, dim);
}

}  // namespace umminfer::omni
