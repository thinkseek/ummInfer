// SPDX-License-Identifier: Apache-2.0
#include "cuda_kernels.hpp"

#include <cuda.h>
#include <math.h>

namespace umminfer::image2 {
namespace {

constexpr int kBlock = 256;
__device__ __forceinline__ float bf(const Bf16& x) { return __bfloat162float(x); }

__device__ __forceinline__ float warp_reduce(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
  return v;
}

// Block sum over blockDim.x; broadcasts result through warp_sums[0].
__device__ float block_sum(float v, float* warp_sums) {
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
  // Entry barrier: the previous block_sum returned warp_sums[0] after its final
  // barrier, but a slow thread may still be reading it when a fast thread
  // re-enters and overwrites warp_sums[0]. Wait for all readers first.
  __syncthreads();
  v = warp_reduce(v);
  if (lane == 0) warp_sums[wid] = v;
  __syncthreads();
  if (wid == 0) {
    v = lane < nw ? warp_sums[lane] : 0.0f;
    v = warp_reduce(v);
    if (lane == 0) warp_sums[0] = v;
  }
  __syncthreads();
  return warp_sums[0];
}

}  // namespace

// ---------------- norms / elementwise -------------------------------------

__global__ void rms_norm_lastdim_k(const Bf16* __restrict__ x, const Bf16* __restrict__ w,
                                   Bf16* __restrict__ out, int dim, float eps) {
  const int row = blockIdx.x;
  const Bf16* xr = x + (size_t)row * dim;
  Bf16* orow = out + (size_t)row * dim;
  __shared__ float smem[32];
  float ss = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    float v = bf(xr[i]);
    ss += v * v;
  }
  const float scale = rsqrtf(block_sum(ss, smem) / dim + eps);
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    orow[i] = __float2bfloat16(bf(xr[i]) * scale * bf(w[i]));
}

void rms_norm_lastdim(const Bf16* x, const Bf16* weight, Bf16* out, int rows, int dim,
                      float eps, cudaStream_t s) {
  rms_norm_lastdim_k<<<rows, kBlock, 32 * sizeof(float), s>>>(x, weight, out, dim, eps);
}

__global__ void ln_no_affine_k(const Bf16* __restrict__ x, Bf16* __restrict__ out, int dim,
                               float eps) {
  const int row = blockIdx.x;
  const Bf16* xr = x + (size_t)row * dim;
  Bf16* orow = out + (size_t)row * dim;
  __shared__ float scratch[32], s_mean, s_inv;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) acc += bf(xr[i]);
  const float mean = block_sum(acc, scratch) / dim;
  acc = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    float d = bf(xr[i]) - mean;
    acc += d * d;
  }
  // block_sum contains __syncthreads, so every thread must call it; broadcast
  // the result through shared memory instead of guarding the call by lane 0.
  const float inv = rsqrtf(block_sum(acc, scratch) / dim + eps);
  if (threadIdx.x == 0) { s_mean = mean; s_inv = inv; }
  __syncthreads();
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    orow[i] = __float2bfloat16((bf(xr[i]) - s_mean) * s_inv);
}

void layer_norm_no_affine(const Bf16* x, Bf16* out, int rows, int dim, float eps,
                          cudaStream_t s) {
  ln_no_affine_k<<<rows, kBlock, 32 * sizeof(float), s>>>(x, out, dim, eps);
}

__global__ void adaln_modulate_k(const Bf16* normed, const Bf16* shift, const Bf16* scale,
                                 Bf16* out, int dim) {
  const int row = blockIdx.x;
  const Bf16* nr = normed + (size_t)row * dim;
  Bf16* orow = out + (size_t)row * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    orow[i] = __float2bfloat16(bf(nr[i]) * (1.0f + bf(scale[i])) + bf(shift[i]));
}

void adaln_modulate(const Bf16* normed, const Bf16* shift, const Bf16* scale, Bf16* out,
                    int rows, int dim, cudaStream_t s) {
  adaln_modulate_k<<<rows, kBlock, 0, s>>>(normed, shift, scale, out, dim);
}

// RMS norm over the channel dimension of an NCHW tensor. One block per
// (n, spatial pixel); channel c for pixel p is at c*hw + p in NCHW layout.
__global__ void rms_nchw_k(const Bf16* __restrict__ x, const Bf16* __restrict__ gamma,
                           Bf16* __restrict__ out, int c, int hw, float sqrt_c) {
  const size_t pixel = blockIdx.x;  // flat over n*hw
  __shared__ float smem[32];
  float ss = 0.0f;
  for (int ch = threadIdx.x; ch < c; ch += blockDim.x) {
    float v = bf(x[(size_t)ch * hw + pixel]);
    ss += v * v;
  }
  float norm = fmaxf(sqrtf(block_sum(ss, smem)), 1e-12f);
  for (int ch = threadIdx.x; ch < c; ch += blockDim.x) {
    float v = bf(x[(size_t)ch * hw + pixel]) / norm * sqrt_c * bf(gamma[ch]);
    out[(size_t)ch * hw + pixel] = __float2bfloat16(v);
  }
}

void rms_norm_nchw(const Bf16* x, const Bf16* gamma, Bf16* out, int n, int c, int hw,
                   cudaStream_t s) {
  const int threads = c <= 32 ? 32 : c <= 256 ? 256 : kBlock;
  rms_nchw_k<<<n * hw, threads, 32 * sizeof(float), s>>>(x, gamma, out, c, hw, sqrtf((float)c));
}

__global__ void silu_k(Bf16* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = bf(x[i]);
  x[i] = __float2bfloat16(v / (1.0f + expf(-v)));
}
void silu_inplace(Bf16* x, int n, cudaStream_t s) {
  silu_k<<<(n + kBlock - 1) / kBlock, kBlock, 0, s>>>(x, n);
}

__global__ void gelu_tanh_k(Bf16* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = bf(x[i]);
  float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
  x[i] = __float2bfloat16(0.5f * v * (1.0f + tanhf(inner)));
}
void gelu_tanh_inplace(Bf16* x, int n, cudaStream_t s) {
  gelu_tanh_k<<<(n + kBlock - 1) / kBlock, kBlock, 0, s>>>(x, n);
}

__global__ void silu_mul_k(Bf16* out, const Bf16* gate, const Bf16* up, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = bf(gate[i]);
  out[i] = __float2bfloat16(bf(up[i]) * (g / (1.0f + expf(-g))));
}
void silu_mul(Bf16* out, const Bf16* gate, const Bf16* up, int n, cudaStream_t s) {
  silu_mul_k<<<(n + kBlock - 1) / kBlock, kBlock, 0, s>>>(out, gate, up, n);
}

__global__ void add_res_k(Bf16* y, const Bf16* r, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __float2bfloat16(bf(y[i]) + bf(r[i]));
}
void add_residual(Bf16* y, const Bf16* r, int n, cudaStream_t s) {
  add_res_k<<<(n + kBlock - 1) / kBlock, kBlock, 0, s>>>(y, r, n);
}

__global__ void gated_res_k(Bf16* y, const Bf16* x, const Bf16* gate, int dim) {
  const int row = blockIdx.x;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const size_t idx = (size_t)row * dim + i;
    y[idx] = __float2bfloat16(bf(y[idx]) + bf(gate[i]) * bf(x[idx]));
  }
}
void gated_residual(Bf16* y, const Bf16* x, const Bf16* gate, int rows, int dim,
                    cudaStream_t s) {
  gated_res_k<<<rows, kBlock, 0, s>>>(y, x, gate, dim);
}

__global__ void f32_bf16_k(const float* src, Bf16* dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2bfloat16(src[i]);
}
void f32_to_bf16(const float* src, Bf16* dst, int n, cudaStream_t s) {
  f32_bf16_k<<<(n + kBlock - 1) / kBlock, kBlock, 0, s>>>(src, dst, n);
}
__global__ void bf16_f32_k(const Bf16* src, float* dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = bf(src[i]);
}
void bf16_to_f32(const Bf16* src, float* dst, int n, cudaStream_t s) {
  bf16_f32_k<<<(n + kBlock - 1) / kBlock, kBlock, 0, s>>>(src, dst, n);
}

// ---------------- RoPE -----------------------------------------------------

__global__ void rope_half_k(Bf16* x, const float* cos, const float* sin, int rows,
                            int n_heads, int half) {
  const int token = blockIdx.y, head = blockIdx.x;
  Bf16* row = x + ((size_t)token * n_heads + head) * (half * 2);
  for (int i = threadIdx.x; i < half; i += blockDim.x) {
    const float c = cos[token * half + i], sn = sin[token * half + i];
    float a = bf(row[i]), b = bf(row[i + half]);
    row[i] = __float2bfloat16(a * c - b * sn);
    row[i + half] = __float2bfloat16(b * c + a * sn);
  }
}
void rope_rotate_half(Bf16* x, const float* cos, const float* sin, int rows, int n_heads,
                      int head_dim, cudaStream_t s) {
  dim3 grid(n_heads, rows);
  rope_half_k<<<grid, 64, 0, s>>>(x, cos, sin, rows, n_heads, head_dim / 2);
}

__global__ void rope_complex_k(Bf16* x, const float* cos, const float* sin, int rows,
                               int n_heads, int half) {
  const int token = blockIdx.y, head = blockIdx.x;
  Bf16* row = x + ((size_t)token * n_heads + head) * (half * 2);
  for (int k = threadIdx.x; k < half; k += blockDim.x) {
    const float c = cos[token * half + k], sn = sin[token * half + k];
    float re = bf(row[2 * k]), im = bf(row[2 * k + 1]);
    row[2 * k] = __float2bfloat16(re * c - im * sn);
    row[2 * k + 1] = __float2bfloat16(re * sn + im * c);
  }
}
void rope_complex_pairs(Bf16* x, const float* cos, const float* sin, int rows,
                        int n_heads, int head_dim, cudaStream_t s) {
  dim3 grid(n_heads, rows);
  rope_complex_k<<<grid, 64, 0, s>>>(x, cos, sin, rows, n_heads, head_dim / 2);
}

// ---------------- unified flash attention ---------------------------------
// One block per (q head, query token); threads == head_dim (128). Online
// softmax over KV tiles in shared memory. Supports GQA via head/group and an
// optional causal mask. q/k/v are [B,S,H,D] in (token,head,channel) order.
template <int D, int BLOCK_K>
__global__ void flash_attn_k(const Bf16* __restrict__ q, const Bf16* __restrict__ k,
                             const Bf16* __restrict__ v, Bf16* __restrict__ out, int sq,
                             int sk, int heads, int kv_heads, int group, bool causal) {
  extern __shared__ float smem[];
  float* qs = smem;
  float* acc = qs + D;
  float* red = acc + D;
  Bf16* kt = reinterpret_cast<Bf16*>(red + 32);
  Bf16* vt = kt + BLOCK_K * D;
  const int head = blockIdx.x;
  const int qi = blockIdx.y;
  const int tid = threadIdx.x;
  const int kv_head = head / group;
  const Bf16* qrow = q + ((size_t)qi * heads + head) * D;
  if (tid < D) qs[tid] = bf(qrow[tid]);
  if (tid < D) acc[tid] = 0.0f;
  __syncthreads();
  const float scale = rsqrtf((float)D);
  const int k_hi = causal ? min(sk, qi + 1) : sk;

  auto dot_q = [&](const Bf16* kr) {
    float partial = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) partial += qs[i] * bf(kr[i]);
    return block_sum(partial, red);
  };

  // Running softmax state in shared scalars (single-writer, barrier-guarded).
  __shared__ float run_max_s, run_sum_s;
  if (tid == 0) { run_max_s = -INFINITY; run_sum_s = 0.0f; }
  __syncthreads();

  for (int start = 0; start < k_hi; start += BLOCK_K) {
    const int tile = min(BLOCK_K, k_hi - start);
    for (int i = tid; i < tile * D; i += blockDim.x) {
      kt[i] = k[((size_t)(start + i / D) * kv_heads + kv_head) * D + i % D];
      vt[i] = v[((size_t)(start + i / D) * kv_heads + kv_head) * D + i % D];
    }
    __syncthreads();
    float tile_max = -INFINITY;
    for (int j = 0; j < tile; ++j)
      tile_max = fmaxf(tile_max, dot_q(kt + (size_t)j * D) * scale);
    __shared__ float old_max_s, new_max_s;
    if (tid == 0) {
      old_max_s = run_max_s;
      new_max_s = fmaxf(run_max_s, tile_max);
      run_max_s = new_max_s;
    }
    __syncthreads();
    if (old_max_s != -INFINITY) {
      const float r = expf(old_max_s - new_max_s);
      if (tid < D) acc[tid] *= r;
    }
    __syncthreads();
    if (tid == 0)
      run_sum_s = (old_max_s == -INFINITY ? 0.0f : run_sum_s * expf(old_max_s - new_max_s));
    __syncthreads();
    float wsum = 0.0f;
    for (int j = 0; j < tile; ++j) {
      const float w = expf(dot_q(kt + (size_t)j * D) * scale - new_max_s);
      wsum += w;
      if (tid < D) {
        const Bf16* vr = vt + (size_t)j * D;
        for (int i = tid; i < D; i += blockDim.x) acc[i] += w * bf(vr[i]);
      }
    }
    __syncthreads();
    if (tid == 0) run_sum_s += wsum;
    __syncthreads();  // before the next tile reuses kt/vt
  }
  if (tid < D)
    out[((size_t)qi * heads + head) * D + tid] =
        __float2bfloat16(acc[tid] / fmaxf(run_sum_s, 1e-30f));
}

template <int D, int BLOCK_K>
static void launch_flash(const Bf16* q, const Bf16* k, const Bf16* v, Bf16* out, int batch,
                         int heads, int kv_heads, int sq, int sk, bool causal,
                         cudaStream_t s) {
  const int group = heads / kv_heads;
  const size_t smem =
      (2 * D + 32) * sizeof(float) + 2 * BLOCK_K * D * sizeof(Bf16);
  dim3 grid(heads, sq);
  dim3 block(D, 1);
  for (int b = 0; b < batch; ++b) {
    const size_t qoff = (size_t)b * sq * heads * D;
    const size_t kvoff = (size_t)b * sk * kv_heads * D;
    flash_attn_k<D, BLOCK_K><<<grid, block, smem, s>>>(
        q + qoff, k + kvoff, v + kvoff, out + qoff, sq, sk, heads, kv_heads, group, causal);
  }
}

void flash_attn(const Bf16* q, const Bf16* k, const Bf16* v, Bf16* out, int batch, int heads,
                int sq, int sk, int head_dim, cudaStream_t s) {
  // DiT joint (D=128, sk up to ~4608) and VAE spatial (D=384, heads=1).
  if (head_dim == 128) launch_flash<128, 16>(q, k, v, out, batch, heads, heads, sq, sk, false, s);
  else if (head_dim == 384)
    launch_flash<384, 8>(q, k, v, out, batch, heads, heads, sq, sk, false, s);
  else
    throw std::runtime_error("flash_attn: unsupported head_dim " + std::to_string(head_dim));
}

void gqa_causal_attn(const Bf16* q, const Bf16* k, const Bf16* v, Bf16* out, int batch,
                     int hq, int hkv, int sq, int sk, int head_dim, cudaStream_t s) {
  if (head_dim != 128)
    throw std::runtime_error("gqa_causal_attn: expected head_dim 128");
  launch_flash<128, 16>(q, k, v, out, batch, hq, hkv, sq, sk, true, s);
}

// ---------------- VAE 2D convs ---------------------------------------------

__global__ void conv3x3_k(const Bf16* __restrict__ in, const Bf16* __restrict__ weight,
                          const Bf16* __restrict__ bias, Bf16* __restrict__ out, int n,
                          int ci, int co, int h, int width) {
  // One thread per output voxel.
  const size_t total = (size_t)n * co * h * width;
  size_t o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= total) return;
  const int ox = o % width;
  const int oy = (o / width) % h;
  const int oc = (o / ((size_t)width * h)) % co;
  const int nn = o / ((size_t)width * h * co);
  float acc = bf(bias[oc]);
  for (int ic = 0; ic < ci; ++ic)
    for (int ky = 0; ky < 3; ++ky) {
      const int iy = oy + ky - 1;
      if (iy < 0 || iy >= h) continue;
      for (int kx = 0; kx < 3; ++kx) {
        const int ix = ox + kx - 1;
        if (ix < 0 || ix >= width) continue;
        acc += bf(in[((size_t)nn * ci + ic) * h * width + iy * width + ix]) *
               bf(weight[(((size_t)oc * ci + ic) * 3 + ky) * 3 + kx]);
      }
    }
  out[o] = __float2bfloat16(acc);
}
void conv2d_3x3(const Bf16* in, const Bf16* weight, const Bf16* bias, Bf16* out, int n,
                int ci, int co, int h, int width, cudaStream_t s) {
  const size_t total = (size_t)n * co * h * width;
  conv3x3_k<<<(total + kBlock - 1) / kBlock, kBlock, 0, s>>>(in, weight, bias, out, n, ci, co,
                                                              h, width);
}

// 1x1 convolution over an NCHW tensor. One block per (output channel, spatial
// pixel); threads reduce over input channels. Input layout is [n, ci, hw], so
// channel c for pixel p lives at c*hw + p. One block per (pixel, out channel).
__global__ void conv1x1_k(const Bf16* __restrict__ in, const Bf16* __restrict__ weight,
                          const Bf16* __restrict__ bias, Bf16* __restrict__ out, int n,
                          int ci, int co, int hw) {
  // Pixels go on x: n*hw reaches 65536 at 256x256, which exceeds the
  // gridDim.y limit of 65535 and silently skipped the launch.
  const int oc = blockIdx.y;
  const size_t pixel = blockIdx.x;  // flat over n*hw
  const Bf16* wr = weight + (size_t)oc * ci;
  // Only lanes that own an input channel accumulate; idle lanes stay zero so
  // the warp reduction does not double-count the bias below.
  float acc = 0.0f;
  for (int c = threadIdx.x; c < ci; c += blockDim.x)
    acc += bf(in[(size_t)c * hw + pixel]) * bf(wr[c]);
  const int nwarps = blockDim.x >> 5;
  __shared__ float partials[8];  // kBlock=256 -> at most 8 warps
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_down_sync(0xffffffff, acc, off);
  if (lane == 0) partials[wid] = acc;
  __syncthreads();
  if (wid == 0) {
    acc = lane < nwarps ? partials[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1)
      acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) out[(size_t)oc * hw + pixel] =
                       __float2bfloat16(acc + bf(bias[oc]));
  }
}
void conv2d_1x1(const Bf16* in, const Bf16* weight, const Bf16* bias, Bf16* out, int n,
                int ci, int co, int hw, cudaStream_t s) {
  dim3 grid(n * hw, co);
  conv1x1_k<<<grid, kBlock, 32 * sizeof(float), s>>>(in, weight, bias, out, n, ci, co, hw);
}

__global__ void upsample2x_k(const Bf16* in, Bf16* out, int n, int c, int h, int w) {
  const size_t total = (size_t)n * c * h * w;
  size_t o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= total) return;
  const int x = o % w, y = (o / w) % h, ch = (o / ((size_t)w * h)) % c,
            nn = o / ((size_t)w * h * c);
  const int sx = x / 2, sy = y / 2;  // nearest-exact at integer 2x
  out[o] = in[(((size_t)nn * c + ch) * (h / 2) + sy) * (w / 2) + sx];
}
void upsample_nearest_2x(const Bf16* in, Bf16* out, int n, int c, int h, int w,
                         cudaStream_t s) {
  const size_t total = (size_t)n * c * h * w;
  upsample2x_k<<<(total + kBlock - 1) / kBlock, kBlock, 0, s>>>(in, out, n, c, h, w);
}

__global__ void upsample2x_fp32_k(const Bf16* in, Bf16* out, int n, int c, int h, int w) {
  const size_t total = (size_t)n * c * h * w;
  size_t o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= total) return;
  const int x = o % w, y = (o / w) % h, ch = (o / ((size_t)w * h)) % c,
            nn = o / ((size_t)w * h * c);
  const int sx = x / 2, sy = y / 2;
  float v = bf(in[(((size_t)nn * c + ch) * (h / 2) + sy) * (w / 2) + sx]);
  out[o] = __float2bfloat16(v);
}
void upsample_nearest_2x_fp32(const Bf16* in, Bf16* out, int n, int c, int h, int w,
                              cudaStream_t s) {
  const size_t total = (size_t)n * c * h * w;
  upsample2x_fp32_k<<<(total + kBlock - 1) / kBlock, kBlock, 0, s>>>(in, out, n, c, h, w);
}

// ---------------- Philox Box-Muller ----------------------------------------
// Philox4x32-10 (Salmon et al.), keyed by a 64-bit seed. Produces four uint32
// per round; pairs feed the ATen normal Box-Muller: u1 = 1 - u0_uni,
// z0 = sqrt(-2 ln u1) cos(2 pi u2), z1 = ... sin(...).
__device__ __forceinline__ void philox4x32_round(uint32_t ctr[4], uint32_t key[2]) {
  const uint64_t M0 = 0xD2511F53ULL, M1 = 0xCD9E8D57ULL;
  uint64_t p0 = M0 * ctr[0];
  uint64_t p1 = M1 * ctr[2];
  uint32_t lo0 = (uint32_t)p0, hi0 = p0 >> 32;
  uint32_t lo1 = (uint32_t)p1, hi1 = p1 >> 32;
  ctr[0] = hi1 ^ ctr[1] ^ key[0];
  ctr[1] = lo1;
  ctr[2] = hi0 ^ ctr[3] ^ key[1];
  ctr[3] = lo0;
  key[0] += 0x9E3779B9u;
  key[1] += 0xBB67AE85u;
}
__device__ __forceinline__ void philox4x32(uint64_t seed, uint64_t ctr_full,
                                           uint32_t out[4]) {
  uint32_t ctr[4] = {(uint32_t)ctr_full, (uint32_t)(ctr_full >> 32), 0, 0};
  uint32_t key[2] = {(uint32_t)seed, (uint32_t)(seed >> 32)};
#pragma unroll
  for (int r = 0; r < 10; ++r) philox4x32_round(ctr, key);
  out[0] = ctr[0]; out[1] = ctr[1]; out[2] = ctr[2]; out[3] = ctr[3];
}
__global__ void randn_philox_k(Bf16* out, uint64_t seed, size_t n) {
  // Each thread group consumes Philox 4-packs; two uniforms -> two normals.
  const size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t packs = (n + 1) / 2;           // one normal pair per two outputs
  const size_t pid = gid;
  if (pid >= packs) return;
  // Two Philox 4-packs give the four uniforms needed for one Box-Muller pair;
  // element order follows a flat row-major counter over the allocation.
  uint32_t a[4], b[4];
  philox4x32(seed, pid * 2, a);
  philox4x32(seed, pid * 2 + 1, b);
  auto uni = [](uint32_t u) { return (u >> 8) * (1.0f / 16777216.0f) + (1.0f / 33554432.0f); };
  const float u0 = uni(a[0]), u2 = uni(a[1]);
  const float u1 = 1.0f - u0;
  const float radius = sqrtf(-2.0f * logf(u1));
  const float z0 = radius * cosf(6.283185307179586f * u2);
  const float z1 = radius * sinf(6.283185307179586f * u2);
  if (pid * 2 < n) out[pid * 2] = __float2bfloat16(z0);
  if (pid * 2 + 1 < n) out[pid * 2 + 1] = __float2bfloat16(z1);
  (void)b;
}
void randn_bf16_philox(Bf16* out, uint64_t seed, size_t n, cudaStream_t s) {
  const size_t packs = (n + 1) / 2;
  randn_philox_k<<<(packs + kBlock - 1) / kBlock, kBlock, 0, s>>>(out, seed, n);
}

// ---------------- CFG / latent helpers ------------------------------------

__global__ void cfg_k(Bf16* out, const Bf16* cond, const Bf16* neg, float scale,
                      int features) {
  const int row = blockIdx.x;
  extern __shared__ float smem[];
  float* cn = smem;
  float* nn = cn + blockDim.x;
  // Per-token L2 norms over the feature dimension via block reduction.
  float cacc = 0.0f, nacc = 0.0f;
  for (int i = threadIdx.x; i < features; i += blockDim.x) {
    float c = bf(cond[(size_t)row * features + i]);
    float g = bf(neg[(size_t)row * features + i]);
    float comb = g + scale * (c - g);
    out[(size_t)row * features + i] = __float2bfloat16(comb);
    cacc += c * c;
    nacc += comb * comb;
  }
  __shared__ float red[32];
  const float cnorm = sqrtf(block_sum(cacc, red));
  const float nnorm = sqrtf(block_sum(nacc, red));
  const float ratio = cnorm / fmaxf(nnorm, 1e-30f);
  for (int i = threadIdx.x; i < features; i += blockDim.x) {
    const size_t idx = (size_t)row * features + i;
    out[idx] = __float2bfloat16(bf(out[idx]) * ratio);
  }
}
void cfg_combine(Bf16* out, const Bf16* cond, const Bf16* neg, float scale, int tokens,
                 int features, cudaStream_t s) {
  cfg_k<<<tokens, kBlock, 2 * kBlock * sizeof(float) + 32 * sizeof(float), s>>>(
      out, cond, neg, scale, features);
}

__global__ void denorm_k(Bf16* z, const Bf16* lat, const float* mean,
                         const float* inv_std, int hw) {
  const int c = blockIdx.x;
  const size_t base = (size_t)c * hw;
  const float m = mean[c], istd = inv_std[c];
  for (size_t i = threadIdx.x; i < (size_t)hw; i += blockDim.x)
    z[base + i] = __float2bfloat16(bf(lat[base + i]) * istd + m);
}
void denorm_latents(Bf16* z, const Bf16* lat, const float* mean, const float* inv_std,
                    int hw, cudaStream_t s) {
  denorm_k<<<16, kBlock, 0, s>>>(z, lat, mean, inv_std, hw);
}

__global__ void clamp_k(Bf16* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = fminf(1.0f, fmaxf(-1.0f, bf(x[i])));
    x[i] = __float2bfloat16(v);
  }
}
void clamp_n1p1(Bf16* x, int n, cudaStream_t s) {
  clamp_k<<<(n + kBlock - 1) / kBlock, kBlock, 0, s>>>(x, n);
}

__global__ void euler_k(Bf16* sample, const Bf16* velocity, float dt, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = bf(sample[i]) + dt * bf(velocity[i]);
    sample[i] = __float2bfloat16(v);
  }
}
void euler_step_bf16(Bf16* sample, const Bf16* velocity, float dt, int n,
                     cudaStream_t s) {
  euler_k<<<(n + kBlock - 1) / kBlock, kBlock, 0, s>>>(sample, velocity, dt, n);
}

__global__ void euler_f32_k(float* sample, const float* velocity, float dt, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) sample[i] = sample[i] + dt * velocity[i];
}
void euler_f32(float* sample, const float* velocity, float dt, int n, cudaStream_t s) {
  euler_f32_k<<<(n + kBlock - 1) / kBlock, kBlock, 0, s>>>(sample, velocity, dt, n);
}

__global__ void bias_bcast_k(Bf16* out, const Bf16* bias, int rows, int cols) {
  const int col = blockIdx.x;
  for (int r = threadIdx.y; r < rows; r += blockDim.y) {
    const size_t idx = (size_t)r * cols + col;
    out[idx] = __float2bfloat16(bf(out[idx]) + bf(bias[col]));
  }
}
void bias_broadcast(Bf16* out, const Bf16* bias, int rows, int cols, cudaStream_t s) {
  dim3 grid(cols);
  dim3 block(32, min(8, rows));
  bias_bcast_k<<<grid, block, 0, s>>>(out, bias, rows, cols);
}

__global__ void gather_k(const Bf16* table, const int* ids, Bf16* out, int dim) {
  const int row = blockIdx.y;
  const Bf16* src = table + (size_t)ids[row] * dim;
  Bf16* dst = out + (size_t)row * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) dst[i] = src[i];
}
void gather_rows(const Bf16* table, const int* ids, Bf16* out, int rows, int dim,
                 cudaStream_t s) {
  dim3 grid(1, rows);
  gather_k<<<grid, kBlock, 0, s>>>(table, ids, out, dim);
}

__global__ void split_qkv_k(const Bf16* qkv, Bf16* q, Bf16* k, Bf16* v, int c, int hw) {
  // One thread per output element of one part; total n*c*hw per destination.
  const int part = blockIdx.z;
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)c * hw) return;
  const int ch = idx / hw, pix = idx % hw;
  const Bf16 val = qkv[((size_t)part * c + ch) * hw + pix];
  Bf16* dst = part == 0 ? q : part == 1 ? k : v;
  dst[idx] = val;
}
void split_qkv_nchw(const Bf16* qkv, Bf16* q, Bf16* k, Bf16* v, int n, int c, int hw,
                    cudaStream_t s) {
  dim3 grid(((size_t)c * hw + kBlock - 1) / kBlock, 1, 3);
  split_qkv_k<<<grid, kBlock, 0, s>>>(qkv, q, k, v, c, hw);
}

__global__ void nchw_tokens_k(const Bf16* src, Bf16* dst, int c, int hw, bool to_tokens) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)c * hw) return;
  const int ch = i / hw, pix = i % hw;
  if (to_tokens)
    dst[(size_t)pix * c + ch] = src[(size_t)ch * hw + pix];
  else
    dst[(size_t)ch * hw + pix] = src[(size_t)pix * c + ch];
}
void nchw_to_tokens(const Bf16* nchw, Bf16* tokens, int c, int hw, cudaStream_t s) {
  nchw_tokens_k<<<((size_t)c * hw + kBlock - 1) / kBlock, kBlock, 0, s>>>(nchw, tokens, c,
                                                                           hw, true);
}
void tokens_to_nchw(const Bf16* tokens, Bf16* nchw, int c, int hw, cudaStream_t s) {
  nchw_tokens_k<<<((size_t)c * hw + kBlock - 1) / kBlock, kBlock, 0, s>>>(tokens, nchw, c,
                                                                           hw, false);
}


// ---------------- Qwen-Image 2.1 additions ---------------------------------

namespace {

__device__ float block_max(float v, float* warp_sums) {
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
  __syncthreads();
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
  if (lane == 0) warp_sums[wid] = v;
  __syncthreads();
  if (wid == 0) {
    v = lane < nw ? warp_sums[lane] : -INFINITY;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    if (lane == 0) warp_sums[0] = v;
  }
  __syncthreads();
  return warp_sums[0];
}

}  // namespace

__global__ void zero_center_rms_k(const Bf16* __restrict__ x, const Bf16* __restrict__ w,
                                  Bf16* __restrict__ out, int dim, float eps) {
  const int row = blockIdx.x;
  const Bf16* xr = x + (size_t)row * dim;
  Bf16* orow = out + (size_t)row * dim;
  __shared__ float smem[32];
  float ss = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    float v = bf(xr[i]);
    ss += v * v;
  }
  const float scale = rsqrtf(block_sum(ss, smem) / dim + eps);
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    orow[i] = __float2bfloat16(bf(xr[i]) * scale * (bf(w[i]) + 1.0f));
}

void zero_center_rms_norm(const Bf16* x, const Bf16* weight, Bf16* out, int rows, int dim,
                          float eps, cudaStream_t s) {
  zero_center_rms_k<<<rows, kBlock, 32 * sizeof(float), s>>>(x, weight, out, dim, eps);
}

__global__ void ln_mod_select_k(const Bf16* __restrict__ x, const Bf16* __restrict__ sr,
                                const Bf16* __restrict__ sz, const int* __restrict__ tgt,
                                Bf16* __restrict__ out, int dim, float eps) {
  const int row = blockIdx.x;
  const Bf16* xr = x + (size_t)row * dim;
  Bf16* orow = out + (size_t)row * dim;
  const Bf16* sc = tgt[row] ? sr : sz;
  __shared__ float smem[32], s_mean, s_inv;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) acc += bf(xr[i]);
  const float mean = block_sum(acc, smem) / dim;
  acc = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    float d = bf(xr[i]) - mean;
    acc += d * d;
  }
  const float inv = rsqrtf(block_sum(acc, smem) / dim + eps);
  if (threadIdx.x == 0) { s_mean = mean; s_inv = inv; }
  __syncthreads();
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    orow[i] = __float2bfloat16((bf(xr[i]) - s_mean) * s_inv * (1.0f + bf(sc[i])));
}

void layer_norm_mod_select(const Bf16* x, const Bf16* scale_real, const Bf16* scale_zero,
                           const int* is_target, Bf16* out, int rows, int dim, float eps,
                           cudaStream_t s) {
  ln_mod_select_k<<<rows, kBlock, 32 * sizeof(float), s>>>(x, scale_real, scale_zero,
                                                          is_target, out, dim, eps);
}

__global__ void scale_apply_k(const Bf16* __restrict__ x, const Bf16* __restrict__ sr,
                              const Bf16* __restrict__ sz, const int* __restrict__ tgt,
                              Bf16* __restrict__ out, int dim) {
  const int row = blockIdx.x;
  const Bf16* xr = x + (size_t)row * dim;
  Bf16* orow = out + (size_t)row * dim;
  const Bf16* sc = tgt[row] ? sr : sz;
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    orow[i] = __float2bfloat16(bf(xr[i]) * (1.0f + bf(sc[i])));
}

void scale_apply_select(const Bf16* x, const Bf16* scale_real, const Bf16* scale_zero,
                        const int* is_target, Bf16* out, int rows, int dim,
                        cudaStream_t s) {
  scale_apply_k<<<rows, kBlock, 0, s>>>(x, scale_real, scale_zero, is_target, out, dim);
}

__global__ void gated_tanh_k(Bf16* __restrict__ y, const Bf16* __restrict__ x,
                             const Bf16* __restrict__ gr, const Bf16* __restrict__ gz,
                             const int* __restrict__ tgt, int dim) {
  const int row = blockIdx.x;
  Bf16* yr = y + (size_t)row * dim;
  const Bf16* xr = x + (size_t)row * dim;
  const Bf16* g = tgt[row] ? gr : gz;
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    yr[i] = __float2bfloat16(bf(yr[i]) + tanhf(bf(g[i])) * bf(xr[i]));
}

void gated_residual_tanh(Bf16* y, const Bf16* x, const Bf16* gate_real,
                         const Bf16* gate_zero, const int* is_target, int rows, int dim,
                         cudaStream_t s) {
  gated_tanh_k<<<rows, kBlock, 0, s>>>(y, x, gate_real, gate_zero, is_target, dim);
}

__global__ void dupup_k(const Bf16* __restrict__ in, Bf16* __restrict__ out, int cin,
                        int cout, int h, int w, int factor_t, size_t total) {
  size_t o = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (o >= total) return;
  const int ow = o % (2 * w);
  const int oh = (o / (2 * w)) % (2 * h);
  const int co = o / ((size_t)2 * w * 2 * h);
  const int b = oh & 1, d = ow & 1;
  const int hh = oh >> 1, ww = ow >> 1;
  const int a = (factor_t == 2) ? 1 : 0;  // first_chunk drops temporal slice 0
  const int c = ((co * factor_t + a) * 2 + b) * 2 + d;
  const int r = cout * factor_t * 4 / cin;
  const int ci = c / r;
  out[o] = in[((size_t)ci * h + hh) * w + ww];
}

void dupup_2x(const Bf16* in, Bf16* out, int cin, int cout, int h, int w, int factor_t,
              cudaStream_t s) {
  const size_t total = (size_t)cout * (2 * h) * (2 * w);
  dupup_k<<<(total + kBlock - 1) / kBlock, kBlock, 0, s>>>(in, out, cin, cout, h, w,
                                                           factor_t, total);
}

// One thread per key, one block per (query, 64-key tile). Q is staged in shared.
__global__ void vae_scores_k(const Bf16* __restrict__ q, const Bf16* __restrict__ k,
                             float* __restrict__ scores, int c, int hw) {
  extern __shared__ float qs[];
  const int t = blockIdx.x;
  for (int i = threadIdx.x; i < c; i += blockDim.x) qs[i] = bf(q[(size_t)i * hw + t]);
  __syncthreads();
  const int s = blockIdx.y * blockDim.x + threadIdx.x;
  if (s >= hw) return;
  float acc = 0.0f;
  for (int ci = 0; ci < c; ++ci) acc += qs[ci] * bf(k[(size_t)ci * hw + s]);
  scores[(size_t)t * hw + s] = acc * rsqrtf((float)c);
}

__global__ void vae_softmax_k(float* __restrict__ scores, int hw) {
  const int t = blockIdx.x;
  float* r = scores + (size_t)t * hw;
  __shared__ float smem[32];
  float m = -INFINITY;
  for (int i = threadIdx.x; i < hw; i += blockDim.x) m = fmaxf(m, r[i]);
  m = block_max(m, smem);
  float ssum = 0.0f;
  for (int i = threadIdx.x; i < hw; i += blockDim.x) ssum += expf(r[i] - m);
  ssum = block_sum(ssum, smem);
  const float inv = 1.0f / ssum;
  for (int i = threadIdx.x; i < hw; i += blockDim.x) r[i] = expf(r[i] - m) * inv;
}

__global__ void vae_values_k(const float* __restrict__ p, const Bf16* __restrict__ v,
                             float* __restrict__ out, int c, int hw) {
  const int t = blockIdx.x;
  const int ci = blockIdx.y * blockDim.x + threadIdx.x;
  if (ci >= c) return;
  const float* pr = p + (size_t)t * hw;
  const Bf16* vr = v + (size_t)ci * hw;
  float acc = 0.0f;
  for (int s = 0; s < hw; ++s) acc += pr[s] * bf(vr[s]);
  out[(size_t)t * c + ci] = acc;
}

__global__ void tokens_to_nchw_f32_k(const float* __restrict__ tok, Bf16* __restrict__ out,
                                     int c, int hw) {
  size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (idx >= (size_t)c * hw) return;
  const int p = idx % hw;
  const int ch = idx / hw;
  out[idx] = __float2bfloat16(tok[(size_t)p * c + ch]);
}

void vae_attn_f32(const Bf16* q, const Bf16* k, const Bf16* v, float* scores,
                  float* out_tokens, Bf16* out_nchw, int c, int hw, cudaStream_t s) {
  const int tile = 64;
  dim3 sgrid(hw, (hw + tile - 1) / tile);
  vae_scores_k<<<sgrid, tile, (size_t)c * sizeof(float), s>>>(q, k, scores, c, hw);
  vae_softmax_k<<<hw, kBlock, 32 * sizeof(float), s>>>(scores, hw);
  dim3 vgrid(hw, (c + tile - 1) / tile);
  vae_values_k<<<vgrid, tile, 0, s>>>(scores, v, out_tokens, c, hw);
  const size_t total = (size_t)c * hw;
  tokens_to_nchw_f32_k<<<(total + kBlock - 1) / kBlock, kBlock, 0, s>>>(out_tokens, out_nchw,
                                                                       c, hw);
}

__global__ void denorm_std_k(const Bf16* __restrict__ lat, const float* __restrict__ mean,
                             const float* __restrict__ std, Bf16* __restrict__ z, int c,
                             int hw, size_t total) {
  size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int ch = idx / hw;
  const Bf16 sb = __float2bfloat16(std[ch]);
  const Bf16 mb = __float2bfloat16(mean[ch]);
  const Bf16 t1 = __float2bfloat16(bf(lat[idx]) * bf(sb));
  z[idx] = __float2bfloat16(bf(t1) + bf(mb));
}

void denorm_latents_std(const Bf16* lat, const float* mean, const float* std, Bf16* z,
                        int channels, int hw, cudaStream_t s) {
  const size_t total = (size_t)channels * hw;
  denorm_std_k<<<(total + kBlock - 1) / kBlock, kBlock, 0, s>>>(lat, mean, std, z, channels,
                                                                hw, total);
}

}  // namespace umminfer::image2
