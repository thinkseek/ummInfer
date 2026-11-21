// SPDX-License-Identifier: Apache-2.0
#include "cuda_kernels.hpp"

#include <cfloat>
#include <stdexcept>
#include <string>

namespace umminfer::native {
namespace {

constexpr int kBlock = 256;
// Matmul reductions use a full warp-widened block so each thread holds ~1-3
// terms on the 1024-wide dims; the smaller summation order noise keeps the
// talker's near-tie argmax on the same attractor as the FP32-accumulating
// reference. Elementwise/conv kernels keep kBlock.
constexpr int kReduceBlock = 1024;

__device__ inline float to_float(float x) { return x; }
__device__ inline float to_float(Bf16 x) { return __bfloat162float(x); }
__device__ inline float from_float_f(float x) { return x; }
__device__ inline Bf16 from_float_b(float x) { return __float2bfloat16(x); }

template <typename T>
__device__ inline T from_float(float x);
template <>
__device__ inline float from_float<float>(float x) { return x; }
template <>
__device__ inline Bf16 from_float<Bf16>(float x) { return __float2bfloat16(x); }

__device__ float block_reduce(float v) {
  __shared__ float warp[32];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
  if (lane == 0) warp[wid] = v;
  __syncthreads();
  if (wid == 0) {
    v = lane < (blockDim.x + 31) / 32 ? warp[lane] : 0.0f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
  }
  __shared__ float total;
  if (threadIdx.x == 0) total = v;
  __syncthreads();
  return total;
}

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

template <typename T>
__global__ void residual_kernel(T* y, const T* r, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = from_float<T>(to_float(y[i]) + to_float(r[i]));
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
  if (i < rows * dim) y[i] = from_float<T>(to_float(x[i]) * to_float(scale[i % dim]));
}

// Channel-first layout [channels, length]; alpha/beta are per-channel.
__global__ void snake_kernel(float* x, const float* alpha, const float* beta,
                             int channels, int length) {
  const int t = blockIdx.x, c = blockIdx.y;
  if (t >= length || c >= channels) return;
  const float a = expf(alpha[c]), b = expf(beta[c]);
  const float v = x[static_cast<size_t>(c) * length + t];
  x[static_cast<size_t>(c) * length + t] =
      v + (1.0f / (b + 1e-9f)) * sinf(v * a) * sinf(v * a);
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

template <typename T>
__global__ void gemm_kernel(T* y, const T* x, const T* w, const T* bias, int m,
                            int k, int out) {
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
__global__ void gemv_kernel(const T* w, const T* x, float* out, int n_out, int k) {
  const int row = blockIdx.x;
  if (row >= n_out) return;
  const T* wr = w + static_cast<size_t>(row) * k;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < k; i += blockDim.x)
    acc += to_float(wr[i]) * to_float(x[i]);
  out[row] = block_reduce(acc);
}

__global__ void argmax_kernel(const float* x, int n, int* best) {
  __shared__ int s_idx[1024];
  float lv = -FLT_MAX;
  int li = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    if (x[i] > lv) { lv = x[i]; li = i; }
  }
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_down_sync(0xffffffff, lv, o);
    const int oi = __shfl_down_sync(0xffffffff, li, o);
    if (ov > lv) { lv = ov; li = oi; }
  }
  __shared__ float wv[32];
  if (lane == 0) { wv[wid] = lv; s_idx[wid] = li; }
  __syncthreads();
  if (wid == 0) {
    const int nw = (blockDim.x + 31) / 32;
    if (threadIdx.x < nw) { lv = wv[lane]; li = s_idx[lane]; }
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_down_sync(0xffffffff, lv, o);
      const int oi = __shfl_down_sync(0xffffffff, li, o);
      if (ov > lv) { lv = ov; li = oi; }
    }
    if (lane == 0) *best = li;
  }
}

__global__ void suppress_kernel(float* logits, int lo, int hi, int keep) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= lo && i < hi && i != keep) logits[i] = -INFINITY;
}

// HF repetition penalty: divide positive logits / multiply negatives.
__global__ void repetition_penalty_kernel(float* logits, const int* history,
                                          int n_history, float penalty) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_history) return;
  const int id = history[i];
  logits[id] = logits[id] > 0.0f ? logits[id] / penalty
                                 : logits[id] * penalty;
}

// Force one id to -inf (anti-degeneracy run ban).
__global__ void ban_logit_kernel(float* logits, int id) { logits[id] = -INFINITY; }

// Half-rotation RoPE. x [n, heads, head_dim], positions per token.
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
      const float c = cosf(angle), s = sinf(angle);
      const float x1 = to_float(row[i]), x2 = to_float(row[i + half]);
      row[i] = from_float<T>(x1 * c - x2 * s);
      row[i + half] = from_float<T>(x2 * c + x1 * s);
    }
  }
}

// One block per (q head, q token). Two-pass softmax, fp32 accumulation.
template <typename T>
__global__ void gqa_kernel(const T* q, const T* k, const T* v, T* out, int nq,
                           int nkv, int nqh, int nkh, int head_dim, int mask,
                           int window, int q_offset) {
  extern __shared__ float smem[];
  float* qs = smem;
  float* acc = smem + head_dim;
  float* scratch = smem + 2 * head_dim;
  __shared__ float s_max, s_sum, s_w;
  const int qi = blockIdx.y, qh = blockIdx.x;
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const int nwarps = blockDim.x >> 5;
  const int kvh = qh * nkh / nqh;
  const T* qrow = q + (static_cast<size_t>(qi) * nqh + qh) * head_dim;
  if (tid < head_dim) qs[tid] = to_float(qrow[tid]);
  __syncthreads();

  auto dot = [&](const T* row) {
    float p = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) p += qs[i] * to_float(row[i]);
    for (int o = 16; o > 0; o >>= 1) p += __shfl_down_sync(0xffffffff, p, o);
    if (lane == 0) scratch[wid] = p;
    __syncthreads();
    // Single-thread second stage: all warp partials are visible after the
    // barrier. No divergent-warp shuffle, which stalls when blockDim has more
    // warps than the partial-merge width.
    __shared__ float sd;
    if (tid == 0) {
      float z = 0.0f;
      for (int w = 0; w < nwarps; ++w) z += scratch[w];
      sd = z;
    }
    __syncthreads();
    return sd;
  };

  int lo, hi;
  if (mask == 0) {  // causal full
    lo = 0;
    hi = min(nkv, q_offset + qi + 1);
  } else if (mask == 1) {  // causal sliding window
    lo = max(0, q_offset + qi - window + 1);
    hi = min(nkv, q_offset + qi + 1);
  } else {  // bidirectional fixed windows
    const int ws = (qi / window) * window;
    lo = ws;
    hi = min(nkv, ws + window);
  }
  const float scale = rsqrtf(static_cast<float>(head_dim));
  if (tid == 0) s_max = -FLT_MAX;
  __syncthreads();
  for (int j = lo; j < hi; ++j) {
    const float d = dot(k + (static_cast<size_t>(j) * nkh + kvh) * head_dim) * scale;
    if (tid == 0) s_max = fmaxf(s_max, d);
    __syncthreads();
  }
  const float mx = s_max;
  if (tid == 0) s_sum = 0.0f;
  if (tid < head_dim) acc[tid] = 0.0f;
  __syncthreads();
  for (int j = lo; j < hi; ++j) {
    const float wgt = expf(dot(k + (static_cast<size_t>(j) * nkh + kvh) * head_dim) *
                           scale - mx);
    if (tid == 0) { s_w = wgt; s_sum += wgt; }
    __syncthreads();
    if (tid < head_dim)
      acc[tid] += s_w * to_float(v[(static_cast<size_t>(j) * nkh + kvh) * head_dim + tid]);
    __syncthreads();
  }
  if (tid < head_dim)
    out[(static_cast<size_t>(qi) * nqh + qh) * head_dim + tid] =
        from_float<T>(acc[tid] / s_sum);
}

// in [ci, L], w [co, ci/g, K], out [co, L].
// One thread per output element.  The previous launch gave each element its own
// block and let every thread of that block run the same serial reduction, so
// kBlock threads computed and stored identical values.  The arithmetic below is
// unchanged: each element still accumulates in (k, then input channel) order.
__global__ void conv1d_kernel(const float* in, const float* w, const float* bias,
                              float* out, int ci, int co, int L, int K, int dilation,
                              int groups) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= co * L) return;
  const int c = idx / L, t = idx - c * L;
  const int pad = (K - 1) * dilation;
  const int in_per_group = ci / groups;
  const int g = c / (co / groups);
  float acc = bias ? bias[c] : 0.0f;
  for (int kk = 0; kk < K; ++kk) {
    const int src_t = t - pad + kk * dilation;
    if (src_t < 0 || src_t >= L) continue;
    if (groups == ci) {  // depthwise: w [c, 0, kk]
      acc += in[static_cast<size_t>(c) * L + src_t] *
             w[(static_cast<size_t>(c) * 1 + 0) * K + kk];
    } else {
      for (int ic = g * in_per_group; ic < (g + 1) * in_per_group; ++ic)
        acc += in[static_cast<size_t>(ic) * L + src_t] *
               w[((static_cast<size_t>(c) * ci + ic) * K) + kk];
    }
  }
  out[static_cast<size_t>(c) * L + t] = acc;
}

// ConvTranspose1d: trimmed output length L*stride. w [in, out, K].
__global__ void transconv_kernel(const float* in, const float* w, const float* bias,
                                 float* out, int ci, int co, int L, int K,
                                 int stride) {
  const int out_len = L * stride;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= co * out_len) return;
  const int c = idx / out_len, o = idx - c * out_len;
  float acc = bias ? bias[c] : 0.0f;
  for (int kk = 0; kk < K; ++kk) {
    const int rem = o - kk;
    if (rem < 0 || rem % stride != 0) continue;
    const int t = rem / stride;
    if (t >= L) continue;
    for (int ic = 0; ic < ci; ++ic)
      acc += in[static_cast<size_t>(ic) * L + t] *
             w[((static_cast<size_t>(ic) * co + c) * K) + kk];
  }
  out[static_cast<size_t>(c) * L * stride + o] = acc;
}

__global__ void dequant_kernel(const float* sum, const float* usage, float* emb,
                               int rows, int dim, float eps) {
  const int r = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows || d >= dim) return;
  const float u = usage[r];
  emb[static_cast<size_t>(r) * dim + d] =
      sum[static_cast<size_t>(r) * dim + d] / fmaxf(u, eps);
}

__global__ void sum_tables_kernel(float* dst, const float* const* srcs, int count,
                                  int rows, int dim) {
  const int r = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows || d >= dim) return;
  float acc = 0.0f;
  for (int g = 0; g < count; ++g) acc += srcs[g][static_cast<size_t>(r) * dim + d];
  dst[static_cast<size_t>(r) * dim + d] = acc;
}

__global__ void transpose_kernel(const float* in, float* out, int ch, int L) {
  const int t = blockIdx.x, c = blockIdx.y;
  if (t >= L || c >= ch) return;
  out[static_cast<size_t>(t) * ch + c] = in[static_cast<size_t>(c) * L + t];
}

__global__ void transpose_inv_kernel(const float* in, float* out, int ch, int L) {
  // in [L, ch] channel-last -> out [ch, L] channel-first.
  const int t = blockIdx.x, c = blockIdx.y;
  if (t >= L || c >= ch) return;
  out[static_cast<size_t>(c) * L + t] = in[static_cast<size_t>(t) * ch + c];
}

__global__ void bias_rows_kernel(float* y, const float* bias, int rows, int dim) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows * dim) y[i] += bias[i % dim];
}

__global__ void gather_col_kernel(const int* codes, int* out, int T, int stride,
                                  int group) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < T) out[t] = codes[t * stride + group];
}

}  // namespace

template <typename T>
void rms_norm_t(const T* x, const T* weight, T* out, int rows, int dim, float eps,
                cudaStream_t st) {
  rms_kernel<T><<<rows, kReduceBlock, 0, st>>>(x, weight, out, dim, eps);
}
template <typename T>
void layer_norm_t(const T* x, const T* w, const T* b, T* out, int rows, int dim,
                  float eps, cudaStream_t st) {
  ln_kernel<T><<<rows, kReduceBlock, 0, st>>>(x, w, b, out, dim, eps);
}
template <typename T>
void add_residual_t(T* y, const T* r, int n, cudaStream_t st) {
  residual_kernel<T><<<(n + kBlock - 1) / kBlock, kBlock, 0, st>>>(y, r, n);
}
template <typename T>
void silu_mul_t(T* out, const T* g, const T* u, int n, cudaStream_t st) {
  silu_mul_kernel<T><<<(n + kBlock - 1) / kBlock, kBlock, 0, st>>>(out, g, u, n);
}
template <typename T>
void silu_inplace_t(T* x, int n, cudaStream_t st) {
  silu_inplace_kernel<T><<<(n + kBlock - 1) / kBlock, kBlock, 0, st>>>(x, n);
}
template <typename T>
void gelu_inplace_t(T* x, int n, cudaStream_t st) {
  gelu_kernel<T><<<(n + kBlock - 1) / kBlock, kBlock, 0, st>>>(x, n);
}
template <typename T>
void layer_scale_t(T* y, const T* x, const T* s, int rows, int dim,
                   cudaStream_t st) {
  layer_scale_kernel<T><<<(rows * dim + kBlock - 1) / kBlock, kBlock, 0, st>>>(
      y, x, s, rows, dim);
}
void snake_beta_f32(float* x, const float* a, const float* b, int channels,
                    int length, cudaStream_t st) {
  dim3 grid(length, channels);
  snake_kernel<<<grid, kBlock, 0, st>>>(x, a, b, channels, length);
}
void clamp_f32(float* x, int n, float lo, float hi, cudaStream_t st) {
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
  dim3 grid(1, m);
  gather_kernel<T><<<grid, kBlock, 0, st>>>(table, ids, out, m, dim);
}
template <typename T>
void gemm_t(T* y, const T* x, const T* w, const T* bias, int m, int k, int out,
            cudaStream_t st) {
  dim3 grid(out, m);
  gemm_kernel<T><<<grid, kReduceBlock, 0, st>>>(y, x, w, bias, m, k, out);
}
template <typename T>
void gemv_t(const T* w, const T* x, float* out, int n_out, int k, cudaStream_t st) {
  gemv_kernel<T><<<n_out, kReduceBlock, 0, st>>>(w, x, out, n_out, k);
}
int argmax_f32(const float* x, int n, int* scratch, cudaStream_t st) {
  argmax_kernel<<<1, min(1024, n), 0, st>>>(x, n, scratch);
  int idx;
  cudaMemcpyAsync(&idx, scratch, sizeof(int), cudaMemcpyDeviceToHost, st);
  cudaStreamSynchronize(st);
  return idx;
}
void suppress_logits(float* logits, int lo, int hi, int keep, cudaStream_t st) {
  suppress_kernel<<<(hi - lo + kBlock - 1) / kBlock, kBlock, 0, st>>>(logits, lo, hi,
                                                                       keep);
}
void repetition_penalty_f32(float* logits, const int* history, int n_history,
                            float penalty, cudaStream_t st) {
  if (n_history <= 0) return;
  repetition_penalty_kernel<<<(n_history + kBlock - 1) / kBlock, kBlock, 0, st>>>(
      logits, history, n_history, penalty);
}
void ban_logit_f32(float* logits, int id, cudaStream_t st) {
  ban_logit_kernel<<<1, 1, 0, st>>>(logits, id);
}

template <typename T>
void rope_half_t(T* x, const int* positions, const float* inv_freq, int n,
                 int heads, int head_dim, cudaStream_t st) {
  dim3 grid(1, n);
  dim3 block(64, 4);
  rope_kernel<T><<<grid, block, 0, st>>>(x, positions, inv_freq, n, heads, head_dim);
}
template <typename T>
void gqa_attention_t(const T* q, const T* k, const T* v, T* out, int nq, int nkv,
                     int nqh, int nkh, int head_dim, int mask, int window,
                     int q_offset, cudaStream_t st) {
  dim3 grid(nqh, nq);
  const size_t smem = (2 * head_dim + 32) * sizeof(float);
  gqa_kernel<T><<<grid, head_dim, smem, st>>>(q, k, v, out, nq, nkv, nqh, nkh,
                                              head_dim, mask, window, q_offset);
}

void conv1d_causal_f32(const float* in, const float* w, const float* bias,
                       float* out, int ci, int co, int L, int kernel, int dilation,
                       int groups, cudaStream_t st) {
  const int n = co * L;
  conv1d_kernel<<<(n + kBlock - 1) / kBlock, kBlock, 0, st>>>(
      in, w, bias, out, ci, co, L, kernel, dilation, groups);
}
void conv_transpose1d_f32(const float* in, const float* w, const float* bias,
                          float* out, int ci, int co, int L, int kernel, int stride,
                          cudaStream_t st) {
  const int n = co * L * stride;
  transconv_kernel<<<(n + kBlock - 1) / kBlock, kBlock, 0, st>>>(
      in, w, bias, out, ci, co, L, kernel, stride);
}
void codebook_dequant_f32(const float* sum, const float* usage, float* emb,
                          int codes, int dim, float eps, cudaStream_t st) {
  dim3 grid((dim + kBlock - 1) / kBlock, codes);
  dequant_kernel<<<grid, kBlock, 0, st>>>(sum, usage, emb, codes, dim, eps);
}
void sum_tables_f32(float* dst, const float* const* srcs, int count, int rows,
                    int dim, cudaStream_t st) {
  dim3 grid((dim + kBlock - 1) / kBlock, rows);
  sum_tables_kernel<<<grid, kBlock, 0, st>>>(dst, srcs, count, rows, dim);
}
void transpose_f32(const float* in, float* out, int ch, int L, cudaStream_t st) {
  dim3 grid(L, ch);
  transpose_kernel<<<grid, kBlock, 0, st>>>(in, out, ch, L);
}
void transpose_inv_f32(const float* in, float* out, int ch, int L,
                       cudaStream_t st) {
  dim3 grid(L, ch);
  transpose_inv_kernel<<<grid, kBlock, 0, st>>>(in, out, ch, L);
}
void add_bias_rows_f32(float* y, const float* bias, int rows, int dim,
                       cudaStream_t st) {
  bias_rows_kernel<<<(rows * dim + kBlock - 1) / kBlock, kBlock, 0, st>>>(y, bias,
                                                                           rows, dim);
}
void gather_code_column(const int* codes, int* out, int frames, int stride,
                        int group, cudaStream_t st) {
  gather_col_kernel<<<(frames + kBlock - 1) / kBlock, kBlock, 0, st>>>(
      codes, out, frames, stride, group);
}

// Explicit instantiations.
template void rms_norm_t<float>(const float*, const float*, float*, int, int, float,
                                cudaStream_t);
template void rms_norm_t<Bf16>(const Bf16*, const Bf16*, Bf16*, int, int, float,
                               cudaStream_t);
template void layer_norm_t<float>(const float*, const float*, const float*, float*,
                                  int, int, float, cudaStream_t);
template void add_residual_t<float>(float*, const float*, int, cudaStream_t);
template void add_residual_t<Bf16>(Bf16*, const Bf16*, int, cudaStream_t);
template void silu_mul_t<float>(float*, const float*, const float*, int, cudaStream_t);
template void silu_mul_t<Bf16>(Bf16*, const Bf16*, const Bf16*, int, cudaStream_t);
template void silu_inplace_t<float>(float*, int, cudaStream_t);
template void silu_inplace_t<Bf16>(Bf16*, int, cudaStream_t);
template void gelu_inplace_t<float>(float*, int, cudaStream_t);
template void layer_scale_t<float>(float*, const float*, const float*, int, int,
                                   cudaStream_t);
template void gather_rows_t<float>(const float*, const int*, float*, int, int,
                                   cudaStream_t);
template void gather_rows_t<Bf16>(const Bf16*, const int*, Bf16*, int, int,
                                  cudaStream_t);
template void gemm_t<float>(float*, const float*, const float*, const float*, int, int,
                            int, cudaStream_t);
template void gemm_t<Bf16>(Bf16*, const Bf16*, const Bf16*, const Bf16*, int, int,
                           int, cudaStream_t);
template void gemv_t<Bf16>(const Bf16*, const Bf16*, float*, int, int, cudaStream_t);
template void rope_half_t<float>(float*, const int*, const float*, int, int, int,
                                 cudaStream_t);
template void rope_half_t<Bf16>(Bf16*, const int*, const float*, int, int, int,
                                cudaStream_t);
template void gqa_attention_t<float>(const float*, const float*, const float*, float*,
                                     int, int, int, int, int, int, int, int,
                                     cudaStream_t);
template void gqa_attention_t<Bf16>(const Bf16*, const Bf16*, const Bf16*, Bf16*,
                                    int, int, int, int, int, int, int, int,
                                    cudaStream_t);

}  // namespace umminfer::native
