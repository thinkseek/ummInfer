// SPDX-License-Identifier: Apache-2.0
// Elementwise, attention and hybrid-layer kernels for Qwen3.8-27B.
#include "qwen38_ops.hpp"

#include <cuda_bf16.h>

#include <stdexcept>
#include <string>

namespace umminfer::qwen38 {
namespace {

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float bf(const Bf16& x) { return __bfloat162float(x); }
__device__ __forceinline__ Bf16 to_bf(float x) { return __float2bfloat16(x); }

// ---- RMSNorm (plain and zero-centred) ------------------------------------
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
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
  if (lane == 0) warp_sums[wid] = sum;
  __syncthreads();
  if (wid == 0) {
    sum = lane < nwarps ? warp_sums[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) s_inv = rsqrtf(sum / dim + eps);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float scale = kZeroCentered ? 1.0f + bf(w[i]) : bf(w[i]);
    orow[i] = to_bf(bf(xr[i]) * s_inv * scale);
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
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
  if (lane == 0) warp_sums[wid] = acc;
  __syncthreads();
  if (wid == 0) {
    acc = lane < nwarps ? warp_sums[lane] : 0.0f;
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
    acc = lane < nwarps ? warp_sums[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
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

__global__ void silu_mul_kernel(Bf16* out, const Bf16* gate, const Bf16* up, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float g = bf(gate[i]);
  const float sg = g / (1.0f + expf(-g));
  out[i] = to_bf(bf(to_bf(sg)) * bf(up[i]));
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
  for (int i = threadIdx.x; i < dim; i += blockDim.x) dst[i] = src[i];
}

__global__ void replace_rows_kernel(Bf16* dst, const Bf16* src, int start, int dim) {
  const int row = blockIdx.y;
  const Bf16* s = src + static_cast<size_t>(row) * dim;
  Bf16* d = dst + static_cast<size_t>(start + row) * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) d[i] = s[i];
}

// ---- GQA / MHA ------------------------------------------------------------
// One block per (query head, query token).  blockDim is 128 for every head
// size; threads stride over head_dim.  Two-pass exact softmax in fp32.
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
  float* qs = smem;                 // head_dim
  float* acc = smem + head_dim;     // head_dim
  float* reduce = smem + 2 * head_dim;  // 32
  __shared__ float s_max, s_sum, s_w;
  const int qi = blockIdx.y;
  const int qh = blockIdx.x;
  const int tid = threadIdx.x;
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
    // The reference stores the attention output as bf16 before the gate.
    float o = bf(to_bf(acc[i] * inv));
    if (kGated) {
      const float gv = bf(grow[i]);
      const float sg = 1.0f / (1.0f + expf(-gv));
      o *= bf(to_bf(sg));
    }
    orow[i] = to_bf(o);
  }
}

// ---- mRoPE (partial rotary, half rotation) --------------------------------
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

__global__ void vision_rope_kernel(Bf16* x, const float* cos, const float* sin,
                                   int n, int heads, int head_dim) {
  const int token = blockIdx.y;
  const int half = head_dim / 2;
  for (int h = threadIdx.y; h < heads; h += blockDim.y) {
    Bf16* row = x + (static_cast<size_t>(token) * heads + h) * head_dim;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
      const float c = cos[static_cast<size_t>(token) * head_dim + i];
      const float s = sin[static_cast<size_t>(token) * head_dim + i];
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
// One thread per channel; the three previous inputs per channel live in
// `state`, which is updated in place.  Layout state[k * channels + c].
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
    // conv1d writes bf16, and the reference applies SiLU to that rounded value.
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

// One block per value head.  The 128x128 state tile is staged in shared memory
// (dim * (dim + 1) floats) and walked once per token, as in the reference
// recurrent form.
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
    for (int k = 0; k < dim; ++k) S[k * stride + vid] += kn[qrow + k] * delta;
    __syncthreads();
    float acc = 0.0f;
    for (int k = 0; k < dim; ++k) acc += S[k * stride + vid] * qn[qrow + k];
    out[(static_cast<size_t>(t) * v_heads + hv) * dim + vid] = to_bf(acc);
    __syncthreads();
  }

  for (int k = 0; k < dim; ++k)
    state[static_cast<size_t>(hv) * dim * dim + k * dim + vid] = S[k * stride + vid];
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
    // weight * hidden.to(bf16) is a bf16 tensor in the reference, so the
    // product is rounded before the fp32 gate multiply.
    const float wb = bf(to_bf(bf(weight[i]) * bf(to_bf(normed))));
    const float zv = bf(zr[i]);
    const float silu = zv / (1.0f + expf(-zv));
    orow[i] = to_bf(wb * silu);
  }
}

__global__ void argmax_kernel(const float* x, int n, int* best_idx) {
  __shared__ int s_idx[1024];
  __shared__ float wv[32];
  float local_val = -INFINITY;
  int local_idx = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = x[i];
    if (v > local_val) { local_val = v; local_idx = i; }
  }
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  for (int off = 16; off > 0; off >>= 1) {
    const float ov = __shfl_down_sync(0xffffffffu, local_val, off);
    const int oi = __shfl_down_sync(0xffffffffu, local_idx, off);
    if (ov > local_val) { local_val = ov; local_idx = oi; }
  }
  if (lane == 0) { wv[wid] = local_val; s_idx[wid] = local_idx; }
  __syncthreads();
  if (wid == 0) {
    const int nw = (blockDim.x + 31) / 32;
    if (threadIdx.x < nw) { local_val = wv[lane]; local_idx = s_idx[lane]; }
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_down_sync(0xffffffffu, local_val, off);
      const int oi = __shfl_down_sync(0xffffffffu, local_idx, off);
      if (ov > local_val) { local_val = ov; local_idx = oi; }
    }
    if (lane == 0) *best_idx = local_idx;
  }
}

// ---- Continuous batching --------------------------------------------------
// The per-sequence kernels below keep the row math of their single-sequence
// counterpart. Only the state base and the visible KV prefix become per-span,
// so a row's result does not depend on which other rows share the call.
__global__ void gather_rows_kernel(Bf16* dst, const Bf16* src, const int* rows,
                                   int m, int dim) {
  const int s = blockIdx.y;
  if (s >= m) return;
  const Bf16* sr = src + static_cast<size_t>(rows[s]) * dim;
  Bf16* dr = dst + static_cast<size_t>(s) * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) dr[i] = sr[i];
}

__global__ void argmax_rows_kernel(const float* x, int n, int* best_idx) {
  const int row = blockIdx.x;
  const float* xr = x + static_cast<size_t>(row) * n;
  __shared__ int s_idx[1024];
  __shared__ float wv[32];
  float local_val = -INFINITY;
  int local_idx = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = xr[i];
    if (v > local_val) { local_val = v; local_idx = i; }
  }
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  for (int off = 16; off > 0; off >>= 1) {
    const float ov = __shfl_down_sync(0xffffffffu, local_val, off);
    const int oi = __shfl_down_sync(0xffffffffu, local_idx, off);
    if (ov > local_val) { local_val = ov; local_idx = oi; }
  }
  if (lane == 0) { wv[wid] = local_val; s_idx[wid] = local_idx; }
  __syncthreads();
  if (wid == 0) {
    const int nw = (blockDim.x + 31) / 32;
    if (threadIdx.x < nw) { local_val = wv[lane]; local_idx = s_idx[lane]; }
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_down_sync(0xffffffffu, local_val, off);
      const int oi = __shfl_down_sync(0xffffffffu, local_idx, off);
      if (ov > local_val) { local_val = ov; local_idx = oi; }
    }
    if (lane == 0) best_idx[row] = local_idx;
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
      for (int k = 0; k < dim; ++k) S[k * stride + vid] += kn[qrow + k] * delta;
      __syncthreads();
      float acc = 0.0f;
      for (int k = 0; k < dim; ++k) acc += S[k * stride + vid] * qn[qrow + k];
      out[(static_cast<size_t>(t) * v_heads + hv) * dim + vid] = to_bf(acc);
      __syncthreads();
    }
    for (int k = 0; k < dim; ++k) sbase[k * dim + vid] = S[k * stride + vid];
    // The next span reloads S from its own state; no thread may still read.
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
  float* qs = smem;                     // head_dim
  float* acc = smem + head_dim;         // head_dim
  float* reduce = smem + 2 * head_dim;  // 32
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

}  // namespace

// ---- host entry points ----------------------------------------------------
void rms_norm(const Bf16* x, const Bf16* weight, Bf16* out, int rows, int dim,
              float eps, cudaStream_t stream) {
  rms_norm_kernel<false><<<rows, 256, 0, stream>>>(x, weight, out, dim, eps);
}

void rms_norm_zero_centered(const Bf16* x, const Bf16* weight, Bf16* out, int rows,
                            int dim, float eps, cudaStream_t stream) {
  rms_norm_kernel<true><<<rows, 256, 0, stream>>>(x, weight, out, dim, eps);
}

void layer_norm(const Bf16* x, const Bf16* weight, const Bf16* bias, Bf16* out,
                int rows, int dim, float eps, cudaStream_t stream) {
  layer_norm_kernel<<<rows, 256, 0, stream>>>(x, weight, bias, out, dim, eps);
}

void add_bias(Bf16* y, const Bf16* bias, int rows, int dim, cudaStream_t stream) {
  bias_kernel<<<rows, 256, 0, stream>>>(y, bias, rows, dim);
}

void add_residual(Bf16* y, const Bf16* residual, int n, cudaStream_t stream) {
  residual_kernel<<<(n + 255) / 256, 256, 0, stream>>>(y, residual, n);
}

void silu_mul(Bf16* out, const Bf16* gate, const Bf16* up, int n, cudaStream_t stream) {
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

void vision_rope(Bf16* x, const float* cos, const float* sin, int n, int heads,
                 int head_dim, cudaStream_t stream) {
  dim3 grid(1, n);
  dim3 block(64, 4);
  vision_rope_kernel<<<grid, block, 0, stream>>>(x, cos, sin, n, heads, head_dim);
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
  dim3 grid(n_q_heads, m);
  const size_t smem = (2 * head_dim + 32) * sizeof(float);
  attention_kernel<true, true><<<grid, 128, smem, stream>>>(
      q, k, v, gate, out, m, nkv, n_q_heads, n_kv_heads, head_dim, q_offset);
}

void vision_attention(const Bf16* q, const Bf16* k, const Bf16* v, Bf16* out, int n,
                      int heads, int head_dim, cudaStream_t stream) {
  dim3 grid(heads, n);
  const size_t smem = (2 * head_dim + 32) * sizeof(float);
  attention_kernel<false, false><<<grid, 128, smem, stream>>>(
      q, k, v, nullptr, out, n, n, heads, heads, head_dim, 0);
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

void gated_delta_rule(float* state, const float* qn, const float* kn, const Bf16* v,
                      const float* beta, const float* g, Bf16* out, int m,
                      int k_heads, int v_heads, int dim, cudaStream_t stream) {
  configure_dgr_smem(dim);
  const size_t smem = static_cast<size_t>(dim) * (dim + 1) * sizeof(float);
  gated_delta_rule_kernel<<<v_heads, dim, smem, stream>>>(
      state, qn, kn, v, beta, g, out, m, k_heads, v_heads, dim);
}

void gated_rms_norm(const Bf16* x, const Bf16* z, const Bf16* weight, Bf16* out,
                    int rows, int dim, float eps, cudaStream_t stream) {
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
  dim3 grid(n_q_heads, rows);
  const size_t smem = (2 * head_dim + 32) * sizeof(float);
  attention_seg_kernel<<<grid, 128, smem, stream>>>(
      q, k_cache, v_cache, gate, out, spans, nseq, seq_stride, n_q_heads,
      n_kv_heads, head_dim);
}

int argmax_f32(const float* logits, int n, int* scratch_idx, cudaStream_t stream) {
  argmax_kernel<<<1, 1024, 0, stream>>>(logits, n, scratch_idx);
  int idx = -1;
  cudaMemcpyAsync(&idx, scratch_idx, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  return idx;
}
}  // namespace umminfer::qwen38
