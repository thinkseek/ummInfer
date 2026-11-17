// SPDX-License-Identifier: Apache-2.0
#include "cuda_kernels_omni.hpp"

#include <stdexcept>
#include <string>

namespace umminfer::omni {
namespace {

constexpr int kBlock = 256;

__device__ inline float bf(const Bf16& x) { return __bfloat162float(x); }

__global__ void rms_norm_kernel(const Bf16* __restrict__ x, const Bf16* __restrict__ w,
                                Bf16* __restrict__ out, int dim, float eps) {
  const int row = blockIdx.x;
  const Bf16* xr = x + static_cast<size_t>(row) * dim;
  Bf16* orow = out + static_cast<size_t>(row) * dim;
  extern __shared__ float smem[];
  float sum = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = bf(xr[i]);
    sum += v * v;
  }
  __shared__ float s_var;
  // Warp-block reduction.
  __shared__ float warp_sums[32];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
  if (lane == 0) warp_sums[wid] = sum;
  __syncthreads();
  if (wid == 0) {
    sum = lane < (blockDim.x + 31) / 32 ? warp_sums[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) s_var = rsqrtf(sum / dim + eps);
  }
  __syncthreads();
  const float scale = s_var;
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    orow[i] = __float2bfloat16(bf(xr[i]) * scale * bf(w[i]));
}

__global__ void layer_norm_kernel(const Bf16* __restrict__ x, const Bf16* __restrict__ w,
                                  const Bf16* __restrict__ b, Bf16* __restrict__ out,
                                  int dim, float eps) {
  const int row = blockIdx.x;
  const Bf16* xr = x + static_cast<size_t>(row) * dim;
  Bf16* orow = out + static_cast<size_t>(row) * dim;
  __shared__ float s_mean, s_inv;
  __shared__ float warp_sums[32];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) acc += bf(xr[i]);
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
  if (lane == 0) warp_sums[wid] = acc;
  __syncthreads();
  if (wid == 0) {
    acc = lane < (blockDim.x + 31) / 32 ? warp_sums[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) s_mean = acc / dim;
  }
  __syncthreads();
  acc = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float d = bf(xr[i]) - s_mean;
    acc += d * d;
  }
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
  if (lane == 0) warp_sums[wid] = acc;
  __syncthreads();
  if (wid == 0) {
    acc = lane < (blockDim.x + 31) / 32 ? warp_sums[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) s_inv = rsqrtf(acc / dim + eps);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    orow[i] = __float2bfloat16((bf(xr[i]) - s_mean) * s_inv * bf(w[i]) + bf(b[i]));
}

template <typename Op>
__global__ void elementwise_kernel(Bf16* x, int n, Op op) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = __float2bfloat16(op(bf(x[i])));
}

struct GeluOp {
  __device__ float operator()(float x) const {
    // transformers default gelu: 0.5 x (1 + erf(x / sqrt(2)))
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
  }
};

__global__ void bias_kernel(Bf16* y, const Bf16* bias, int rows, int dim) {
  const int r = blockIdx.x;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const int idx = r * dim + i;
    y[idx] = __float2bfloat16(bf(y[idx]) + bf(bias[i]));
  }
}

__global__ void residual_kernel(Bf16* y, const Bf16* r, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __float2bfloat16(bf(y[i]) + bf(r[i]));
}

__global__ void silu_mul_kernel(Bf16* out, const Bf16* gate, const Bf16* up, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float g = bf(gate[i]), u = bf(up[i]);
    out[i] = __float2bfloat16(u * g / (1.0f + expf(-g)));
  }
}

__global__ void f32_to_bf16_kernel(const float* src, Bf16* dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2bfloat16(src[i]);
}

__global__ void bf16_to_f32_kernel(const Bf16* src, float* dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = bf(src[i]);
}

__global__ void embed_kernel(const Bf16* table, const int* ids, Bf16* out, int m, int dim) {
  const int row = blockIdx.y;
  if (row >= m) return;
  const Bf16* src = table + static_cast<size_t>(ids[row]) * dim;
  Bf16* dst = out + static_cast<size_t>(row) * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) dst[i] = src[i];
}

__global__ void replace_rows_kernel(Bf16* dst, const Bf16* src, int start, int n, int dim) {
  const int row = blockIdx.y;
  if (row >= n) return;
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    dst[static_cast<size_t>(start + row) * dim + i] = src[static_cast<size_t>(row) * dim + i];
}

// Half-rotation RoPE (Qwen3 rotate_half). inv_freq has head_dim/2 entries.
// x is [n, num_heads, head_dim].
__global__ void rope_kernel(Bf16* x, const int* positions, const float* inv_freq,
                            int n, int num_heads, int head_dim) {
  const int token = blockIdx.y;
  const int half = head_dim / 2;
  const int pos = positions[token];
  for (int h = threadIdx.y; h < num_heads; h += blockDim.y) {
    Bf16* row = x + (static_cast<size_t>(token) * num_heads + h) * head_dim;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
      // inv_freq is indexed by within-half channel, independent of head/token.
      const float angle = pos * inv_freq[i];
      const float c = cosf(angle), s = sinf(angle);
      const float x1 = bf(row[i]), x2 = bf(row[i + half]);
      row[i] = __float2bfloat16(x1 * c - x2 * s);
      row[i + half] = __float2bfloat16(x2 * c + x1 * s);
    }
  }
}

__global__ void conv_kernel(const Bf16* in, const Bf16* weight, const Bf16* bias,
                            Bf16* out, int b, int ci, int co, int h, int w, int oh,
                            int ow) {
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = b * co * oh * ow;
  if (o >= total) return;
  const int owx = o % ow;
  const int ohy = (o / ow) % oh;
  const int oc = (o / (ow * oh)) % co;
  const int ob = o / (ow * oh * co);
  float acc = bf(bias[oc]);
  for (int ic = 0; ic < ci; ++ic) {
    for (int kh = 0; kh < 3; ++kh) {
      const int ih = ohy * 2 + kh - 1;
      if (ih < 0 || ih >= h) continue;
      for (int kw = 0; kw < 3; ++kw) {
        const int iw = owx * 2 + kw - 1;
        if (iw < 0 || iw >= w) continue;
        acc += bf(in[((ob * ci + ic) * h + ih) * w + iw]) *
               bf(weight[((oc * ci + ic) * 3 + kh) * 3 + kw]);
      }
    }
  }
  out[o] = __float2bfloat16(GeluOp{}(acc));
}

// One block per (query head, query token), blockDim.x == head_dim so each
// thread owns one output coordinate. Two-pass exact softmax.
// Shared memory: q[head_dim] + acc[head_dim] + reduction scratch.
__global__ void gqa_attn_kernel(const Bf16* q, const Bf16* k, const Bf16* v,
                                Bf16* out, int nq, int nkv, int n_q_heads,
                                int n_kv_heads, int head_dim, bool causal,
                                int window_len, int q_offset) {
  extern __shared__ float smem[];
  float* qs = smem;
  float* acc = smem + head_dim;
  float* reduce = smem + 2 * head_dim;  // one slot per warp
  __shared__ float s_max, s_sum, s_w;
  const int qi = blockIdx.y, qh = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31, wid = tid >> 5;
  const int nwarps = blockDim.x >> 5;
  const int kvh = qh * n_kv_heads / n_q_heads;
  const Bf16* qrow = q + (static_cast<size_t>(qi) * n_q_heads + qh) * head_dim;
  if (tid < head_dim) qs[tid] = bf(qrow[tid]);
  __syncthreads();

  // Block-wide sum of per-thread partial products -> one scalar in every lane.
  auto block_dot = [&](const Bf16* row) {
    float partial = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) partial += qs[i] * bf(row[i]);
    for (int off = 16; off > 0; off >>= 1)
      partial += __shfl_down_sync(0xffffffff, partial, off);
    if (lane == 0) reduce[wid] = partial;
    __syncthreads();
    float total = 0.0f;
    if (tid < 32) {
      total = tid < nwarps ? reduce[tid] : 0.0f;
      for (int off = 16; off > 0; off >>= 1)
        total += __shfl_down_sync(0xffffffff, total, off);
    }
    __shared__ float s_dot;
    if (tid == 0) s_dot = total;
    __syncthreads();
    return s_dot;
  };

  int lo, hi;
  if (causal) {
    lo = 0;
    hi = min(nkv, q_offset + qi + 1);
  } else {
    const int wstart = (qi / window_len) * window_len;
    lo = wstart;
    hi = min(nkv, wstart + window_len);
  }
  const float scale = rsqrtf(static_cast<float>(head_dim));

  if (tid == 0) s_max = -INFINITY;
  __syncthreads();
  for (int kj = lo; kj < hi; ++kj) {
    const Bf16* krow = k + (static_cast<size_t>(kj) * n_kv_heads + kvh) * head_dim;
    const float dot = block_dot(krow) * scale;
    if (tid == 0) s_max = fmaxf(s_max, dot);
    __syncthreads();
  }
  const float mx = s_max;

  if (tid == 0) s_sum = 0.0f;
  if (tid < head_dim) acc[tid] = 0.0f;
  __syncthreads();
  for (int kj = lo; kj < hi; ++kj) {
    const Bf16* krow = k + (static_cast<size_t>(kj) * n_kv_heads + kvh) * head_dim;
    const float weight = expf(block_dot(krow) * scale - mx);
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

__global__ void argmax_kernel(const float* x, int n, int* best_idx) {
  __shared__ int s_idx[1024];
  float local_val = -INFINITY;
  int local_idx = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = x[i];
    if (v > local_val) { local_val = v; local_idx = i; }
  }
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  for (int off = 16; off > 0; off >>= 1) {
    const float ov = __shfl_down_sync(0xffffffff, local_val, off);
    const int oi = __shfl_down_sync(0xffffffff, local_idx, off);
    if (ov > local_val) { local_val = ov; local_idx = oi; }
  }
  __shared__ float wv[32];
  if (lane == 0) { wv[wid] = local_val; s_idx[wid] = local_idx; }
  __syncthreads();
  if (wid == 0) {
    const int nw = (blockDim.x + 31) / 32;
    if (threadIdx.x < nw) { local_val = wv[lane]; local_idx = s_idx[lane]; }
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_down_sync(0xffffffff, local_val, off);
      const int oi = __shfl_down_sync(0xffffffff, local_idx, off);
      if (ov > local_val) { local_val = ov; local_idx = oi; }
    }
    if (lane == 0) *best_idx = local_idx;
  }
}

}  // namespace

void rms_norm(const Bf16* x, const Bf16* weight, Bf16* out, int rows, int dim,
              float eps, cudaStream_t stream) {
  rms_norm_kernel<<<rows, kBlock, 0, stream>>>(x, weight, out, dim, eps);
}

void layer_norm(const Bf16* x, const Bf16* weight, const Bf16* bias, Bf16* out,
                int rows, int dim, float eps, cudaStream_t stream) {
  layer_norm_kernel<<<rows, kBlock, 0, stream>>>(x, weight, bias, out, dim, eps);
}

void add_bias(Bf16* y, const Bf16* bias, int rows, int dim, cudaStream_t stream) {
  bias_kernel<<<rows, kBlock, 0, stream>>>(y, bias, rows, dim);
}

void add_residual(Bf16* y, const Bf16* residual, int n, cudaStream_t stream) {
  residual_kernel<<<(n + kBlock - 1) / kBlock, kBlock, 0, stream>>>(y, residual, n);
}

void gelu_inplace(Bf16* x, int n, cudaStream_t stream) {
  elementwise_kernel<<<(n + kBlock - 1) / kBlock, kBlock, 0, stream>>>(x, n, GeluOp{});
}

void silu_mul(Bf16* out, const Bf16* gate, const Bf16* up, int n, cudaStream_t stream) {
  silu_mul_kernel<<<(n + kBlock - 1) / kBlock, kBlock, 0, stream>>>(out, gate, up, n);
}

void f32_to_bf16(const float* src, Bf16* dst, int n, cudaStream_t stream) {
  f32_to_bf16_kernel<<<(n + kBlock - 1) / kBlock, kBlock, 0, stream>>>(src, dst, n);
}

void bf16_to_f32(const Bf16* src, float* dst, int n, cudaStream_t stream) {
  bf16_to_f32_kernel<<<(n + kBlock - 1) / kBlock, kBlock, 0, stream>>>(src, dst, n);
}

void index_embeddings(const Bf16* table, const int* ids, Bf16* out, int m, int dim,
                      cudaStream_t stream) {
  dim3 grid(1, m);
  embed_kernel<<<grid, kBlock, 0, stream>>>(table, ids, out, m, dim);
}

void replace_rows(Bf16* dst, const Bf16* src, int start, int n, int dim,
                  cudaStream_t stream) {
  dim3 grid(1, n);
  replace_rows_kernel<<<grid, kBlock, 0, stream>>>(dst, src, start, n, dim);
}

void rope_half(Bf16* x, const int* positions, const float* inv_freq, int n,
               int num_heads, int head_dim, cudaStream_t stream) {
  dim3 grid(1, n);
  dim3 block(64, 4);
  rope_kernel<<<grid, block, 0, stream>>>(x, positions, inv_freq, n, num_heads, head_dim);
}

void conv3x3s2_gelu(const Bf16* in, const Bf16* weight, const Bf16* bias, Bf16* out,
                    int b, int ci, int co, int h, int w, cudaStream_t stream) {
  const int oh = (h + 1) / 2, ow = (w + 1) / 2;  // padding 1, stride 2
  const int total = b * co * oh * ow;
  conv_kernel<<<(total + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
      in, weight, bias, out, b, ci, co, h, w, oh, ow);
}

__global__ void conv_permute_kernel(const Bf16* in, Bf16* out, int chunks) {
  // out [c,13,7680], in [c,480,16,13].
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = chunks * 13 * 7680;
  if (idx >= total) return;
  const int flat = idx % 7680;
  const int t = (idx / 7680) % 13;
  const int c = idx / (13 * 7680);
  const int ch = flat / 16, fr = flat % 16;
  out[idx] = in[(((static_cast<size_t>(c) * 480 + ch) * 16 + fr) * 13 + t)];
}

void conv_permute(const Bf16* in, Bf16* out, int chunks, cudaStream_t stream) {
  const int total = chunks * 13 * 7680;
  conv_permute_kernel<<<(total + kBlock - 1) / kBlock, kBlock, 0, stream>>>(in, out,
                                                                             chunks);
}

__global__ void mel_chunks_kernel(const float* mel, Bf16* out, int chunks,
                                  int valid_frames) {
  // out [chunks, 1, 128, 100], mel [128, 3000].
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = chunks * 128 * 100;
  if (idx >= total) return;
  const int t = idx % 100;
  const int m = (idx / 100) % 128;
  const int c = idx / (128 * 100);
  const int frame = c * 100 + t;
  out[idx] = frame < valid_frames ? __float2bfloat16(mel[m * 3000 + frame]) : Bf16(0.0f);
}

__global__ void sinusoidal_kernel(Bf16* hidden, const float* pe, int chunks,
                                  int rows, int dim) {
  const int c = blockIdx.z, r = blockIdx.y;
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    hidden[((static_cast<size_t>(c) * rows + r) * dim) + i] =
        __float2bfloat16(bf(hidden[((static_cast<size_t>(c) * rows + r) * dim) + i]) + pe[r * dim + i]);
}

void mel_to_chunks(const float* mel, Bf16* out, int chunks, int valid_frames,
                   cudaStream_t stream) {
  const int total = chunks * 128 * 100;
  mel_chunks_kernel<<<(total + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
      mel, out, chunks, valid_frames);
}

void add_sinusoidal(Bf16* hidden, const float* pe, int chunks, int rows, int dim,
                    cudaStream_t stream) {
  dim3 grid(1, rows, chunks);
  sinusoidal_kernel<<<grid, kBlock, 0, stream>>>(hidden, pe, chunks, rows, dim);
}

void gqa_attention(const Bf16* q, const Bf16* k, const Bf16* v, Bf16* out, int nq,
                   int nkv, int n_q_heads, int n_kv_heads, int head_dim, bool causal,
                   int window_len, int q_offset, cudaStream_t stream) {
  // Two-pass softmax: block = (query head, query token), shared memory holds
  // the head_dim query and fp32 accumulator. Max sequence here is ~400.
  // ponytail: O(nq*nkv*d) naive attention, no FlashAttention; fine at the
  // 0.6B scale and 30 s clips, replace if longer audio is needed.
  dim3 grid(n_q_heads, nq);
  // One thread per head_dim coordinate; head_dim is 64 (audio) or 128 (text).
  const int block = head_dim;
  const size_t smem = (2 * head_dim + 32) * sizeof(float);
  gqa_attn_kernel<<<grid, block, smem, stream>>>(q, k, v, out, nq, nkv, n_q_heads,
                                                 n_kv_heads, head_dim, causal,
                                                 window_len, q_offset);
}

int argmax_f32(const float* logits, int n, int* scratch_idx, cudaStream_t stream) {
  argmax_kernel<<<1, min(1024, n), 0, stream>>>(logits, n, scratch_idx);
  int idx;
  cudaMemcpyAsync(&idx, scratch_idx, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  return idx;
}

// ---- Qwen3-Omni additions -------------------------------------------------

__global__ void mrope_half_kernel(Bf16* x, const float* cos, const float* sin,
                                  int n, int num_heads, int head_dim) {
  const int token = blockIdx.y;
  const int half = head_dim / 2;
  for (int h = threadIdx.y; h < num_heads; h += blockDim.y) {
    Bf16* row = x + (static_cast<size_t>(token) * num_heads + h) * head_dim;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
      const float c = cos[static_cast<size_t>(token) * head_dim + i];
      const float s = sin[static_cast<size_t>(token) * head_dim + i];
      const float x1 = bf(row[i]), x2 = bf(row[i + half]);
      row[i] = __float2bfloat16(x1 * c - x2 * s);
      row[i + half] = __float2bfloat16(x2 * c + x1 * s);
    }
  }
}

void mrope_half(Bf16* x, const float* cos, const float* sin, int n,
                int num_heads, int head_dim, cudaStream_t stream) {
  dim3 grid(1, n);
  dim3 block(64, 4);
  mrope_half_kernel<<<grid, block, 0, stream>>>(x, cos, sin, n, num_heads, head_dim);
}

__global__ void vision_rope_kernel(Bf16* x, const Bf16* cos, const Bf16* sin,
                                   int n, int num_heads, int head_dim) {
  const int token = blockIdx.y;
  const int half = head_dim / 2;
  for (int h = threadIdx.y; h < num_heads; h += blockDim.y) {
    Bf16* row = x + (static_cast<size_t>(token) * num_heads + h) * head_dim;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
      const float c = bf(cos[static_cast<size_t>(token) * head_dim + i]);
      const float s = bf(sin[static_cast<size_t>(token) * head_dim + i]);
      const float x1 = bf(row[i]), x2 = bf(row[i + half]);
      row[i] = __float2bfloat16(x1 * c - x2 * s);
      row[i + half] = __float2bfloat16(x2 * c + x1 * s);
    }
  }
}

void vision_rope(Bf16* x, const Bf16* cos, const Bf16* sin, int n,
                 int num_heads, int head_dim, cudaStream_t stream) {
  dim3 grid(1, n);
  dim3 block(64, 4);
  vision_rope_kernel<<<grid, block, 0, stream>>>(x, cos, sin, n, num_heads, head_dim);
}

struct GeluTanhOp {
  __device__ float operator()(float x) const {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(kSqrt2OverPi * (x + 0.044715f * x * x * x)));
  }
};

void gelu_tanh_inplace(Bf16* x, int n, cudaStream_t stream) {
  elementwise_kernel<<<(n + kBlock - 1) / kBlock, kBlock, 0, stream>>>(x, n, GeluTanhOp{});
}

// One block per token row. Softmax over 128 logits in fp32, then stable top-8.
__global__ void moe_route_kernel(const float* logits, int* top_idx, float* top_w,
                                 int rows, int num_experts, int top_k) {
  extern __shared__ float smem[];  // num_experts floats
  const int row = blockIdx.x;
  const float* lr = logits + static_cast<size_t>(row) * num_experts;
  for (int i = threadIdx.x; i < num_experts; i += blockDim.x) smem[i] = lr[i];
  __syncthreads();
  __shared__ float s_max;
  if (threadIdx.x == 0) {
    float m = -INFINITY;
    for (int i = 0; i < num_experts; ++i) m = fmaxf(m, smem[i]);
    s_max = m;
  }
  __syncthreads();
  __shared__ float s_sum;
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (int i = 0; i < num_experts; ++i) {
      smem[i] = expf(smem[i] - s_max);
      sum += smem[i];
    }
    s_sum = sum;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < num_experts; i += blockDim.x) smem[i] /= s_sum;
  __syncthreads();
  if (threadIdx.x != 0) return;
  // Sequential top-k on 128 values; strict > keeps the lower index on ties.
  float vals[8];
  int idxs[8];
  for (int k = 0; k < top_k; ++k) {
    vals[k] = -1.0f;
    idxs[k] = -1;
  }
  for (int e = 0; e < num_experts; ++e) {
    const float v = smem[e];
    for (int k = 0; k < top_k; ++k) {
      if (v > vals[k]) {
        for (int j = top_k - 1; j > k; --j) {
          vals[j] = vals[j - 1];
          idxs[j] = idxs[j - 1];
        }
        vals[k] = v;
        idxs[k] = e;
        break;
      }
    }
  }
  float wsum = 0.0f;
  for (int k = 0; k < top_k; ++k) wsum += vals[k];
  int* ti = top_idx + static_cast<size_t>(row) * top_k;
  float* tw = top_w + static_cast<size_t>(row) * top_k;
  for (int k = 0; k < top_k; ++k) {
    ti[k] = idxs[k];
    tw[k] = vals[k] / wsum;
  }
}

void moe_route(const float* logits, int* top_idx, float* top_w, int rows,
               int num_experts, int top_k, cudaStream_t stream) {
  moe_route_kernel<<<rows, 128, num_experts * sizeof(float), stream>>>(
      logits, top_idx, top_w, rows, num_experts, top_k);
}

__global__ void add_deepstack_kernel(Bf16* hidden, const Bf16* feature,
                                     const int* vis_pos, int n_vis, int dim) {
  const int v = blockIdx.y;
  if (v >= n_vis) return;
  const int row = vis_pos[v];
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    Bf16* dst = hidden + static_cast<size_t>(row) * dim + i;
    *dst = __float2bfloat16(bf(*dst) + bf(feature[static_cast<size_t>(v) * dim + i]));
  }
}

void add_deepstack(Bf16* hidden, const Bf16* feature, const int* vis_pos,
                   int n_vis, int dim, cudaStream_t stream) {
  dim3 grid(1, n_vis);
  add_deepstack_kernel<<<grid, kBlock, 0, stream>>>(hidden, feature, vis_pos, n_vis,
                                                     dim);
}

__global__ void split_qkv_kernel(const Bf16* qkv, Bf16* q, Bf16* k, Bf16* v,
                                 int tokens, int width) {
  const int t = blockIdx.y;
  if (t >= tokens) return;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= width) return;
  const Bf16* src = qkv + static_cast<size_t>(t) * 3 * width;
  const size_t dst = static_cast<size_t>(t) * width + j;
  q[dst] = src[j];
  k[dst] = src[width + j];
  v[dst] = src[2 * width + j];
}

void split_qkv(const Bf16* qkv, Bf16* q, Bf16* k, Bf16* v, int tokens,
               int width, cudaStream_t stream) {
  dim3 grid((width + kBlock - 1) / kBlock, tokens);
  split_qkv_kernel<<<grid, kBlock, 0, stream>>>(qkv, q, k, v, tokens, width);
}

}  // namespace umminfer::omni
