// SPDX-License-Identifier: Apache-2.0
// Wan2.2-TI2V-5B DiT: BF16 tensor-core GEMMs with FP32 normalization/modulation.
#include "wan_dit.hpp"

#ifdef UMM_VIDEO_CUDA

#include "safetensors_mmap.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cublas_v2.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace umminfer::video {
namespace {

using Bf16 = __nv_bfloat16;
using Bf16x2 = __nv_bfloat162;
constexpr int kLayers = 30;
constexpr int kDim = 3072;
constexpr int kHeads = 24;
constexpr int kHeadDim = 128;
constexpr int kFf = 14336;
constexpr int kTextDim = 4096;
constexpr int kFreqDim = 256;
constexpr int kChannels = 48;
constexpr int kPatchVolume = 4;
constexpr int kProjection = kChannels * kPatchVolume;
constexpr int kThreads = 256;
constexpr int kQueryTile = kWanDitQueryTile;
constexpr size_t kCrossKvCacheSlots = 2;

static_assert(kDim == kThreads * 12, "Wan DiT norm kernels assume kDim == 12 * kThreads");
static_assert(kHeadDim == kWanDitFlashHeadDim, "Wan DiT FlashAttention assumes head_dim == 128");
static_assert(kWanDitFlashBr == 64 && kWanDitFlashBc == 64 && kWanDitFlashThreads == 128,
              "Wan DiT FlashAttention assumes 64x64 tiles with 4 warps (128 threads)");

#define UMM_DIT_CUDA(expr)                                                     \
  do {                                                                         \
    const cudaError_t status = (expr);                                          \
    if (status != cudaSuccess) {                                                \
      throw std::runtime_error(std::string(#expr) + ": " +                    \
                               cudaGetErrorString(status));                      \
    }                                                                           \
  } while (0)

#define UMM_DIT_CUBLAS(expr)                                                   \
  do {                                                                         \
    const cublasStatus_t status = (expr);                                       \
    if (status != CUBLAS_STATUS_SUCCESS) {                                      \
      throw std::runtime_error(std::string(#expr) + ": cuBLAS status " +      \
                               std::to_string(static_cast<int>(status)));       \
    }                                                                           \
  } while (0)

size_t checked_bytes(size_t count, size_t element_size) {
  if (element_size != 0 && count > std::numeric_limits<size_t>::max() / element_size) {
    throw std::overflow_error("Wan DiT allocation size overflows size_t");
  }
  return count * element_size;
}

int grid_for(size_t count) {
  const size_t blocks = (count + kThreads - 1) / kThreads;
  if (blocks > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("Wan DiT CUDA grid is too large");
  }
  return static_cast<int>(std::max<size_t>(blocks, 1));
}

uint32_t load_u32(const std::byte* source) {
  return static_cast<uint32_t>(source[0]) |
         (static_cast<uint32_t>(source[1]) << 8) |
         (static_cast<uint32_t>(source[2]) << 16) |
         (static_cast<uint32_t>(source[3]) << 24);
}

float bits_to_f32(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint16_t f32_to_bf16_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0) {
    return static_cast<uint16_t>((bits >> 16) | 0x0040u);
  }
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

float bf16_bits_to_f32(uint16_t value) {
  return bits_to_f32(static_cast<uint32_t>(value) << 16);
}

struct AllocationTracker {
  size_t current = 0;
  size_t peak = 0;
};

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { reset(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&&) = delete;
  DeviceBuffer& operator=(DeviceBuffer&&) = delete;

  void reset() noexcept {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
      if (tracker_ != nullptr) {
        tracker_->current -= bytes_;
      }
      data_ = nullptr;
      bytes_ = 0;
      tracker_ = nullptr;
    }
  }

  void allocate(size_t bytes, AllocationTracker& tracker) {
    if (data_ != nullptr) throw std::logic_error("Wan DiT buffer allocated twice");
    if (bytes == 0) return;
    void* data = nullptr;
    UMM_DIT_CUDA(cudaMalloc(&data, bytes));
    data_ = data;
    bytes_ = bytes;
    tracker_ = &tracker;
    tracker.current += bytes;
    tracker.peak = std::max(tracker.peak, tracker.current);
  }

  void resize(size_t bytes, AllocationTracker& tracker) {
    if (bytes == bytes_ && (bytes == 0 || data_ != nullptr)) return;
    reset();
    allocate(bytes, tracker);
  }

  void* data() const { return data_; }
  size_t bytes() const { return bytes_; }
  template <typename T>
  T* as() const { return static_cast<T*>(data_); }

 private:
  void* data_ = nullptr;
  size_t bytes_ = 0;
  AllocationTracker* tracker_ = nullptr;
};

class PinnedHostBuffer {
 public:
  PinnedHostBuffer() = default;
  ~PinnedHostBuffer() { reset(); }
  PinnedHostBuffer(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer(PinnedHostBuffer&&) = delete;
  PinnedHostBuffer& operator=(PinnedHostBuffer&&) = delete;

  void reset() noexcept {
    if (data_ != nullptr) {
      (void)cudaFreeHost(data_);
      data_ = nullptr;
      bytes_ = 0;
    }
  }

  void resize(size_t bytes) {
    if (bytes <= bytes_ && (bytes == 0 || data_ != nullptr)) return;
    reset();
    if (bytes == 0) return;
    void* data = nullptr;
    UMM_DIT_CUDA(cudaMallocHost(&data, bytes));
    data_ = data;
    bytes_ = bytes;
  }

  void* data() const { return data_; }
  size_t bytes() const { return bytes_; }
  template <typename T>
  T* as() const { return static_cast<T*>(data_); }

 private:
  void* data_ = nullptr;
  size_t bytes_ = 0;
};

class CudaContext {
 public:
  explicit CudaContext(int gpu) {
    UMM_DIT_CUDA(cudaSetDevice(gpu));
    UMM_DIT_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    try {
      UMM_DIT_CUBLAS(cublasCreate(&handle_));
      UMM_DIT_CUBLAS(cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH));
      UMM_DIT_CUBLAS(cublasSetStream(handle_, stream_));
    } catch (...) {
      if (handle_ != nullptr) {
        cublasDestroy(handle_);
        handle_ = nullptr;
      }
      if (stream_ != nullptr) {
        (void)cudaStreamDestroy(stream_);
        stream_ = nullptr;
      }
      throw;
    }
  }
  ~CudaContext() {
    if (handle_ != nullptr) cublasDestroy(handle_);
    if (stream_ != nullptr) (void)cudaStreamDestroy(stream_);
  }
  cublasHandle_t get() const { return handle_; }
  cudaStream_t stream() const { return stream_; }

 private:
  cublasHandle_t handle_ = nullptr;
  cudaStream_t stream_ = nullptr;
};

struct DeviceTensor {
  const void* data = nullptr;
  bool fp32 = false;
  std::vector<int64_t> shape;
};

struct LoadSpec {
  std::string name;
  std::vector<int64_t> shape;
  size_t offset = 0;
  bool fp32 = false;
  bool reorder_patch = false;
};

bool keep_fp32(const std::string& name) {
  return name.starts_with("condition_embedder.time_embedder.") ||
         name.ends_with(".norm2.weight") || name.ends_with(".norm2.bias") ||
         name.ends_with("scale_shift_table");
}

__device__ __forceinline__ float to_float(Bf16 value) { return __bfloat162float(value); }
__device__ __forceinline__ Bf16 to_bf16(float value) { return __float2bfloat16_rn(value); }

__device__ __forceinline__ float gelu_tanh_scalar(float x) {
  const float cube = x * x * x;
  return 0.5f * x *
         (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * cube)));
}

__device__ __forceinline__ uint32_t pack_bf162_u32(float lo, float hi) {
  uint32_t out;
  // PTX cvt.rn.bf16x2.f32 d, a, b packs a -> d[31:16] (hi) and b -> d[15:0] (lo).
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(out) : "f"(lo), "f"(hi));
  return out;
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool pred) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  const unsigned bytes = pred ? 16u : 0u;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(src),
               "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void cp_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int kPending>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(kPending) : "memory");
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

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t (&r)[2], const void* p) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
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

__device__ __forceinline__ float warp_reduce_sum(float val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffffu, val, offset);
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, offset));
  }
  return val;
}

// Warp-shuffle block reductions for kThreads == 256 (8 warps).
// smem must have 9 floats; slot 8 holds the broadcast result so back-to-back
// calls are race-free with only 2 barriers per reduction.
__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  val = warp_reduce_sum(val);
  if (lane == 0) smem[warp] = val;
  __syncthreads();
  if (warp == 0) {
    val = (lane < (kThreads / 32)) ? smem[lane] : 0.0f;
    val = warp_reduce_sum(val);
    if (lane == 0) smem[8] = val;
  }
  __syncthreads();
  return smem[8];
}

__device__ __forceinline__ float block_reduce_max(float val, float* smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  val = warp_reduce_max(val);
  if (lane == 0) smem[warp] = val;
  __syncthreads();
  if (warp == 0) {
    val = (lane < (kThreads / 32)) ? smem[lane] : -CUDART_INF_F;
    val = warp_reduce_max(val);
    if (lane == 0) smem[8] = val;
  }
  __syncthreads();
  return smem[8];
}

__global__ void float_to_bf16_kernel(const float* __restrict__ input,
                                     Bf16* __restrict__ output, size_t count) {
  const size_t pairs = count >> 1;
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < pairs) {
    const float2 v = reinterpret_cast<const float2*>(input)[i];
    reinterpret_cast<Bf16x2*>(output)[i] = __floats2bfloat162_rn(v.x, v.y);
  }
  if (i == 0 && (count & 1u) != 0) {
    output[count - 1] = to_bf16(input[count - 1]);
  }
}

// GPU patchify: scatters NCFHW [kChannels, frames, height, width] FP32 into
// patchified [tokens, 4, kChannels] BF16 with coalesced output writes.
__global__ void patchify_fp32_to_bf16_kernel(
    const float* __restrict__ latent_ncfhw, Bf16* __restrict__ patch_tokens,
    int frames, int height, int width) {
  const size_t total =
      static_cast<size_t>(kChannels) * frames * height * width;
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const size_t src =
      wan_dit_patch_index_to_ncfhw(i, kChannels, frames, height, width);
  patch_tokens[i] = to_bf16(latent_ncfhw[src]);
}

// GPU unpatchify (B=1): reads projected [tokens, 4, kChannels] BF16 and writes
// NCFHW [kChannels, frames, height, width] FP32 with coalesced output writes.
__global__ void unpatchify_bf16_to_fp32_kernel(
    const Bf16* __restrict__ projected, float* __restrict__ output_ncfhw,
    int frames, int height, int width) {
  const size_t total =
      static_cast<size_t>(kChannels) * frames * height * width;
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const size_t src =
      wan_dit_ncfhw_to_patch_index(i, kChannels, frames, height, width);
  output_ncfhw[i] = to_float(projected[src]);
}

// Fused CFG combine + GPU unpatchify (B=2): reads cond/uncond [tokens, 4,
// kChannels] BF16 projections, combines uncond + gs * (cond - uncond) in FP32,
// and writes NCFHW [kChannels, frames, height, width] FP32 directly.
__global__ void cfg_unpatchify_bf16_to_fp32_kernel(
    const Bf16* __restrict__ cond_projected,
    const Bf16* __restrict__ uncond_projected,
    float* __restrict__ output_ncfhw, int frames, int height, int width,
    float guidance_scale) {
  const size_t total =
      static_cast<size_t>(kChannels) * frames * height * width;
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const size_t src =
      wan_dit_ncfhw_to_patch_index(i, kChannels, frames, height, width);
  const float cond_val = to_float(cond_projected[src]);
  const float uncond_val = to_float(uncond_projected[src]);
  output_ncfhw[i] = uncond_val + guidance_scale * (cond_val - uncond_val);
}

__global__ void bias_bf16_vec2_kernel(Bf16* __restrict__ output,
                                      const float* __restrict__ bias,
                                      int output_features_half, size_t pairs) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= pairs) return;
  const int channel_pair = static_cast<int>(i % output_features_half);
  const float2 val = __bfloat1622float2(reinterpret_cast<const Bf16x2*>(output)[i]);
  const float2 b = reinterpret_cast<const float2*>(bias)[channel_pair];
  reinterpret_cast<Bf16x2*>(output)[i] =
      __floats2bfloat162_rn(val.x + b.x, val.y + b.y);
}

__global__ void bias_bf16_kernel(Bf16* __restrict__ output,
                                 const float* __restrict__ bias,
                                 int output_features, size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    const int channel = static_cast<int>(i % output_features);
    output[i] = to_bf16(to_float(output[i]) + bias[channel]);
  }
}

__global__ void bias_f32_kernel(float* __restrict__ output,
                                const float* __restrict__ bias,
                                int output_features, size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) output[i] += bias[i % output_features];
}

__global__ void silu_f32_kernel(float* __restrict__ values, size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) values[i] = values[i] / (1.0f + expf(-values[i]));
}

__global__ void silu_bf16_kernel(Bf16* __restrict__ values, size_t count) {
  const size_t pairs = count >> 1;
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < pairs) {
    const float2 x = __bfloat1622float2(reinterpret_cast<const Bf16x2*>(values)[i]);
    reinterpret_cast<Bf16x2*>(values)[i] = __floats2bfloat162_rn(
        x.x / (1.0f + expf(-x.x)), x.y / (1.0f + expf(-x.y)));
  }
  if (i == 0 && (count & 1u) != 0) {
    const float x = to_float(values[count - 1]);
    values[count - 1] = to_bf16(x / (1.0f + expf(-x)));
  }
}

// Fused bias + approximate tanh GELU for BF16 GEMM outputs with even width.
// Preserves exact post-bias BF16 rounding before GELU for bit-exact parity.
__global__ void bias_gelu_tanh_bf16_kernel(Bf16* __restrict__ values,
                                           const float* __restrict__ bias,
                                           int output_features_half,
                                           size_t pairs) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= pairs) return;
  const int channel_pair = static_cast<int>(i % output_features_half);
  const float2 raw = __bfloat1622float2(reinterpret_cast<const Bf16x2*>(values)[i]);
  const float2 b = reinterpret_cast<const float2*>(bias)[channel_pair];
  const float2 x =
      __bfloat1622float2(__floats2bfloat162_rn(raw.x + b.x, raw.y + b.y));
  reinterpret_cast<Bf16x2*>(values)[i] =
      __floats2bfloat162_rn(gelu_tanh_scalar(x.x), gelu_tanh_scalar(x.y));
}

// One block per token (kDim == 3072 == 256 threads * 6 pairs). Reads input
// once into registers (12 floats/thread) and uses warp-shuffle reductions.
__global__ void layer_norm_kernel(const Bf16* __restrict__ input,
                                  Bf16* __restrict__ output,
                                  const float* __restrict__ gamma,
                                  const float* __restrict__ beta,
                                  const float* __restrict__ shift,
                                  const float* __restrict__ scale,
                                  int tokens, float eps) {
  const int token = blockIdx.x;
  if (token >= tokens) return;
  const Bf16x2* in_pairs =
      reinterpret_cast<const Bf16x2*>(input + static_cast<size_t>(token) * kDim);
  Bf16x2* out_pairs =
      reinterpret_cast<Bf16x2*>(output + static_cast<size_t>(token) * kDim);
  const float2* gamma2 = reinterpret_cast<const float2*>(gamma);
  const float2* beta2 = reinterpret_cast<const float2*>(beta);
  const float2* shift2 = reinterpret_cast<const float2*>(shift);
  const float2* scale2 = reinterpret_cast<const float2*>(scale);

  __shared__ float smem[9];
  float vals[12];
  float local = 0.0f;
  #pragma unroll
  for (int j = 0; j < 6; ++j) {
    const int p = threadIdx.x + j * kThreads;
    const float2 v = __bfloat1622float2(in_pairs[p]);
    vals[j * 2] = v.x;
    vals[j * 2 + 1] = v.y;
    local += v.x + v.y;
  }
  const float mean = block_reduce_sum(local, smem) / static_cast<float>(kDim);

  float local_var = 0.0f;
  #pragma unroll
  for (int j = 0; j < 12; ++j) {
    const float centered = vals[j] - mean;
    local_var += centered * centered;
  }
  const float inverse =
      rsqrtf(block_reduce_sum(local_var, smem) / static_cast<float>(kDim) + eps);

  #pragma unroll
  for (int j = 0; j < 6; ++j) {
    const int p = threadIdx.x + j * kThreads;
    float v0 = (vals[j * 2] - mean) * inverse;
    float v1 = (vals[j * 2 + 1] - mean) * inverse;
    if (gamma != nullptr) {
      const float2 g = gamma2[p];
      const float2 b = beta2[p];
      v0 = v0 * g.x + b.x;
      v1 = v1 * g.y + b.y;
    }
    if (shift != nullptr) {
      const float2 sc = scale2[p];
      const float2 sh = shift2[p];
      v0 = v0 * (1.0f + sc.x) + sh.x;
      v1 = v1 * (1.0f + sc.y) + sh.y;
    }
    out_pairs[p] = __floats2bfloat162_rn(v0, v1);
  }
}

// Fused bias + qk_norm ("rms_norm_across_heads") + optional 3D RoPE.
// One block per token (256 threads * 6 even/odd pairs == 3072 channels).
// Because 256 is a multiple of 64 (kHeadDim / 2), all 6 pairs owned by a
// thread share the exact same RoPE frequency index (threadIdx.x & 63).
template <bool kApplyRope>
__global__ void rms_norm_bias_kernel(Bf16* __restrict__ data,
                                     const float* __restrict__ bias,
                                     const float* __restrict__ weight,
                                     const float* __restrict__ cosine,
                                     const float* __restrict__ sine,
                                     int total_tokens, int rope_tokens,
                                     float eps) {
  const int token = blockIdx.x;
  if (token >= total_tokens) return;
  Bf16x2* data_pairs =
      reinterpret_cast<Bf16x2*>(data + static_cast<size_t>(token) * kDim);
  const float2* bias2 = reinterpret_cast<const float2*>(bias);
  const float2* weight2 = reinterpret_cast<const float2*>(weight);

  __shared__ float smem[9];
  float vals[12];
  float local_sq = 0.0f;
  #pragma unroll
  for (int j = 0; j < 6; ++j) {
    const int p = threadIdx.x + j * kThreads;
    const float2 raw = __bfloat1622float2(data_pairs[p]);
    const float2 b = bias2[p];
    const float2 biased =
        __bfloat1622float2(__floats2bfloat162_rn(raw.x + b.x, raw.y + b.y));
    vals[j * 2] = biased.x;
    vals[j * 2 + 1] = biased.y;
    local_sq += biased.x * biased.x + biased.y * biased.y;
  }
  const float inverse =
      rsqrtf(block_reduce_sum(local_sq, smem) / static_cast<float>(kDim) +
             to_float(to_bf16(eps)));

  float c = 1.0f;
  float s = 0.0f;
  if constexpr (kApplyRope) {
    const int rp = threadIdx.x & ((kHeadDim / 2) - 1);
    const int rope_token = wan_dit_batched_token_in_batch(token, rope_tokens);
    const size_t rope_idx = static_cast<size_t>(rope_token) * kHeadDim + rp * 2;
    c = cosine[rope_idx];
    s = sine[rope_idx + 1];
  }

  #pragma unroll
  for (int j = 0; j < 6; ++j) {
    const int p = threadIdx.x + j * kThreads;
    const float2 w = weight2[p];
    const float2 norm = __bfloat1622float2(
        __floats2bfloat162_rn(vals[j * 2] * inverse, vals[j * 2 + 1] * inverse));
    const Bf16x2 scaled_bf =
        __floats2bfloat162_rn(norm.x * w.x, norm.y * w.y);
    if constexpr (kApplyRope) {
      const float2 scaled = __bfloat1622float2(scaled_bf);
      const float re = __fsub_rn(__fmul_rn(scaled.x, c), __fmul_rn(scaled.y, s));
      const float ro = __fadd_rn(__fmul_rn(scaled.x, s), __fmul_rn(scaled.y, c));
      data_pairs[p] = __floats2bfloat162_rn(re, ro);
    } else {
      data_pairs[p] = scaled_bf;
    }
  }
}

// Stage one [kWanDitFlashBc, kHeadDim] BF16 tile for a single head into shared
// memory using 16-byte async copies. Rows past total_rows are zero-filled.
__device__ __forceinline__ void stage_fa_tile(Bf16* dst, const Bf16* src_head,
                                              int row_base, int total_rows) {
  constexpr int kFaVec = kHeadDim / 8;  // 16 x 16B chunks per row
  constexpr int kFaSteps = (kWanDitFlashBc * kFaVec) / kWanDitFlashThreads;  // 8
  #pragma unroll
  for (int i = 0; i < kFaSteps; ++i) {
    const int idx = static_cast<int>(threadIdx.x) + i * kWanDitFlashThreads;
    const int r = idx >> 4;
    const int c = (idx & 15) << 3;
    const int global_row = row_base + r;
    const bool ok = global_row < total_rows;
    const Bf16* g =
        ok ? (src_head + static_cast<size_t>(global_row) * kDim + c) : src_head;
    cp_async16(dst + r * kWanDitFlashStride + c, g, ok);
  }
}

// Fused BF16 Tensor-Core FlashAttention-2 kernel for Wan2.2-TI2V-5B (D=128).
// Layout: Q [batch, tokens, kHeads, 128], K/V [batch, context_tokens, kHeads, 128],
// O [batch, tokens, kHeads, 128] in BF16 (token stride kDim == 3072).
// Grid: (ceil(tokens / 64), kHeads, batch), Block: 128 threads (4 warps, 16 Q rows/warp).
// Uses mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32, cp.async K/V
// overlap in 34,816 B static shared memory, and warp-shuffle online softmax.
__global__ void __launch_bounds__(kWanDitFlashThreads, 2)
flash_attention_bf16_kernel(const Bf16* __restrict__ query,
                            const Bf16* __restrict__ key,
                            const Bf16* __restrict__ value,
                            Bf16* __restrict__ output, int tokens,
                            int context_tokens) {
  static_assert((kWanDitFlashStride % 8) == 0,
                "FlashAttention shared memory stride must be a multiple of 8 bf16 (16B)");
  __shared__ uint4 s_raw[2 * kWanDitFlashBc * (kWanDitFlashStride / 8)];
  Bf16* s_buf0 = reinterpret_cast<Bf16*>(s_raw);
  Bf16* s_buf1 = s_buf0 + kWanDitFlashBc * kWanDitFlashStride;

  const int q_block_start = blockIdx.x * kWanDitFlashBr;
  const int head = blockIdx.y;
  const int batch = blockIdx.z;
  const size_t q_batch_offset = static_cast<size_t>(batch) * tokens * kDim;
  const size_t kv_batch_offset =
      static_cast<size_t>(batch) * context_tokens * kDim;
  const int head_offset = head * kHeadDim;
  const Bf16* q_head = query + q_batch_offset + head_offset;
  const Bf16* k_head = key + kv_batch_offset + head_offset;
  const Bf16* v_head = value + kv_batch_offset + head_offset;
  Bf16* out_batch = output + q_batch_offset;

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int group_id = lane >> 2;
  const int tid_in_group = lane & 3;
  const int a_row = lane & 15;
  const int a_col = (lane & 16) ? 8 : 0;
  const int b_row = lane & 7;
  const int b_col = (lane & 8) ? 8 : 0;
  const int v_row = lane & 15;

  // Stage Q tile into s_buf0 and load this warp's 16x128 slice into registers.
  stage_fa_tile(s_buf0, q_head, q_block_start, tokens);
  cp_commit();
  cp_wait<0>();
  __syncthreads();

  uint32_t q_frag[8][4];
  #pragma unroll
  for (int kt = 0; kt < 8; ++kt) {
    ldmatrix_x4(q_frag[kt],
                s_buf0 + (warp * 16 + a_row) * kWanDitFlashStride + kt * 16 + a_col);
  }
  __syncthreads();

  // Running online-softmax state and FP32 output accumulator [16, 128] per warp.
  constexpr float kLog2E = 1.4426950408889634f;
  const float scale_log2 = rsqrtf(static_cast<float>(kHeadDim)) * kLog2E;
  float row_max0 = -CUDART_INF_F;
  float row_max1 = -CUDART_INF_F;
  float row_sum0 = 0.0f;
  float row_sum1 = 0.0f;
  float o_acc[16][4];
  #pragma unroll
  for (int dt = 0; dt < 16; ++dt) {
    o_acc[dt][0] = 0.0f;
    o_acc[dt][1] = 0.0f;
    o_acc[dt][2] = 0.0f;
    o_acc[dt][3] = 0.0f;
  }

  // Prefetch K[0] into s_buf0.
  stage_fa_tile(s_buf0, k_head, 0, context_tokens);
  cp_commit();

  const int num_tiles = (context_tokens + kWanDitFlashBc - 1) / kWanDitFlashBc;
  for (int tile = 0; tile < num_tiles; ++tile) {
    const int tile_start = tile * kWanDitFlashBc;
    const int next_start = tile_start + kWanDitFlashBc;

    // Issue async load of V[tile] into s_buf1 while K[tile] completes.
    stage_fa_tile(s_buf1, v_head, tile_start, context_tokens);
    cp_commit();
    cp_wait<1>();
    __syncthreads();

    // 1. S = Q @ K[tile]^T -> s_acc[8][4] ([16, 64] FP32 per warp).
    float s_acc[8][4];
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      s_acc[nt][0] = 0.0f;
      s_acc[nt][1] = 0.0f;
      s_acc[nt][2] = 0.0f;
      s_acc[nt][3] = 0.0f;
    }
    #pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
      #pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        uint32_t k_frag[2];
        ldmatrix_x2(k_frag,
                    s_buf0 + (nt * 8 + b_row) * kWanDitFlashStride + kt * 16 + b_col);
        mma_m16n8k16(s_acc[nt], q_frag[kt], k_frag);
      }
    }

    // 2. Scale scores (in log2 domain) and mask out-of-bounds keys on tail tile.
    if (next_start <= context_tokens) {
      #pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        s_acc[nt][0] *= scale_log2;
        s_acc[nt][1] *= scale_log2;
        s_acc[nt][2] *= scale_log2;
        s_acc[nt][3] *= scale_log2;
      }
    } else {
      #pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int col0 = tile_start + nt * 8 + tid_in_group * 2;
        const int col1 = col0 + 1;
        const bool ok0 = col0 < context_tokens;
        const bool ok1 = col1 < context_tokens;
        s_acc[nt][0] = ok0 ? (s_acc[nt][0] * scale_log2) : -CUDART_INF_F;
        s_acc[nt][1] = ok1 ? (s_acc[nt][1] * scale_log2) : -CUDART_INF_F;
        s_acc[nt][2] = ok0 ? (s_acc[nt][2] * scale_log2) : -CUDART_INF_F;
        s_acc[nt][3] = ok1 ? (s_acc[nt][3] * scale_log2) : -CUDART_INF_F;
      }
    }

    // 3. Online softmax: per-row max over 4-lane quad, rescale O, exp2 & pack P.
    float m0 = -CUDART_INF_F;
    float m1 = -CUDART_INF_F;
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      m0 = fmaxf(m0, fmaxf(s_acc[nt][0], s_acc[nt][1]));
      m1 = fmaxf(m1, fmaxf(s_acc[nt][2], s_acc[nt][3]));
    }
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));

    const float new_max0 = fmaxf(row_max0, m0);
    const float new_max1 = fmaxf(row_max1, m1);
    if (tile > 0) {
      const float rescale0 = exp2f(row_max0 - new_max0);
      const float rescale1 = exp2f(row_max1 - new_max1);
      row_sum0 *= rescale0;
      row_sum1 *= rescale1;
      #pragma unroll
      for (int dt = 0; dt < 16; ++dt) {
        o_acc[dt][0] *= rescale0;
        o_acc[dt][1] *= rescale0;
        o_acc[dt][2] *= rescale1;
        o_acc[dt][3] *= rescale1;
      }
    }
    row_max0 = new_max0;
    row_max1 = new_max1;

    float tile_sum0 = 0.0f;
    float tile_sum1 = 0.0f;
    uint32_t p_frag[4][4];
    #pragma unroll
    for (int pk = 0; pk < 4; ++pk) {
      const int nt0 = pk * 2;
      const int nt1 = nt0 + 1;
      const float p00 = exp2f(s_acc[nt0][0] - row_max0);
      const float p01 = exp2f(s_acc[nt0][1] - row_max0);
      const float p02 = exp2f(s_acc[nt0][2] - row_max1);
      const float p03 = exp2f(s_acc[nt0][3] - row_max1);
      const float p10 = exp2f(s_acc[nt1][0] - row_max0);
      const float p11 = exp2f(s_acc[nt1][1] - row_max0);
      const float p12 = exp2f(s_acc[nt1][2] - row_max1);
      const float p13 = exp2f(s_acc[nt1][3] - row_max1);
      tile_sum0 += (p00 + p01) + (p10 + p11);
      tile_sum1 += (p02 + p03) + (p12 + p13);
      p_frag[pk][0] = pack_bf162_u32(p00, p01);
      p_frag[pk][1] = pack_bf162_u32(p02, p03);
      p_frag[pk][2] = pack_bf162_u32(p10, p11);
      p_frag[pk][3] = pack_bf162_u32(p12, p13);
    }
    tile_sum0 += __shfl_xor_sync(0xffffffffu, tile_sum0, 1);
    tile_sum0 += __shfl_xor_sync(0xffffffffu, tile_sum0, 2);
    tile_sum1 += __shfl_xor_sync(0xffffffffu, tile_sum1, 1);
    tile_sum1 += __shfl_xor_sync(0xffffffffu, tile_sum1, 2);
    row_sum0 += tile_sum0;
    row_sum1 += tile_sum1;

    // 4. Prefetch K[tile+1] into s_buf0 while waiting for V[tile] in s_buf1.
    __syncthreads();
    if (next_start < context_tokens) {
      stage_fa_tile(s_buf0, k_head, next_start, context_tokens);
      cp_commit();
      cp_wait<1>();
    } else {
      cp_wait<0>();
    }
    __syncthreads();

    // 5. O += P @ V[tile] using transposed ldmatrix.x2 on s_buf1.
    #pragma unroll
    for (int pk = 0; pk < 4; ++pk) {
      #pragma unroll
      for (int dt = 0; dt < 16; ++dt) {
        uint32_t v_frag[2];
        ldmatrix_x2_trans(v_frag,
                          s_buf1 + (pk * 16 + v_row) * kWanDitFlashStride + dt * 8);
        mma_m16n8k16(o_acc[dt], p_frag[pk], v_frag);
      }
    }
    if (next_start < context_tokens) {
      __syncthreads();
    }
  }

  // Epilogue: normalize by row_sum and store BF16 directly to [tokens, kDim].
  const float inv_sum0 = 1.0f / fmaxf(row_sum0, 1e-30f);
  const float inv_sum1 = 1.0f / fmaxf(row_sum1, 1e-30f);
  const int q_row0 = q_block_start + warp * 16 + group_id;
  const int q_row1 = q_row0 + 8;
  const int col_in_tile = tid_in_group * 2;
  #pragma unroll
  for (int dt = 0; dt < 16; ++dt) {
    const int d = dt * 8 + col_in_tile;
    if (q_row0 < tokens) {
      *reinterpret_cast<Bf16x2*>(
          out_batch + static_cast<size_t>(q_row0) * kDim + head_offset + d) =
          __floats2bfloat162_rn(o_acc[dt][0] * inv_sum0, o_acc[dt][1] * inv_sum0);
    }
    if (q_row1 < tokens) {
      *reinterpret_cast<Bf16x2*>(
          out_batch + static_cast<size_t>(q_row1) * kDim + head_offset + d) =
          __floats2bfloat162_rn(o_acc[dt][2] * inv_sum1, o_acc[dt][3] * inv_sum1);
    }
  }
}

// Reference cuBLAS attention softmax (used when UMM_DIT_ATTN_REF=1).
// Caches up to 4096 keys in registers (16 floats/thread) so scores are read
// once and expf is evaluated once per key instead of 3 reads + 2 expf calls.
__global__ void attention_softmax_kernel(const float* __restrict__ scores,
                                         Bf16* __restrict__ probabilities,
                                         int keys, int queries) {
  const int row = blockIdx.x;
  const int query = row % queries;
  const int head = row / queries;
  if (head >= kHeads) return;
  const size_t base = (static_cast<size_t>(head) * queries + query) * keys;
  const float attention_scale = rsqrtf(static_cast<float>(kHeadDim));
  __shared__ float smem[9];

  constexpr int kMaxCached = 16;
  if (keys <= kMaxCached * kThreads) {
    float cached[kMaxCached];
    float local_max = -CUDART_INF_F;
    int count = 0;
    for (int key = threadIdx.x; key < keys; key += kThreads) {
      const float s = scores[base + key] * attention_scale;
      cached[count++] = s;
      local_max = fmaxf(local_max, s);
    }
    const float maximum = block_reduce_max(local_max, smem);
    float local_sum = 0.0f;
    for (int j = 0; j < count; ++j) {
      const float w = expf(cached[j] - maximum);
      cached[j] = w;
      local_sum += w;
    }
    const float inverse = 1.0f / block_reduce_sum(local_sum, smem);
    int idx = 0;
    for (int key = threadIdx.x; key < keys; key += kThreads) {
      probabilities[base + key] = to_bf16(cached[idx++] * inverse);
    }
    return;
  }

  float local_max = -CUDART_INF_F;
  for (int key = threadIdx.x; key < keys; key += kThreads) {
    local_max = fmaxf(local_max, scores[base + key] * attention_scale);
  }
  const float maximum = block_reduce_max(local_max, smem);
  float local_sum = 0.0f;
  for (int key = threadIdx.x; key < keys; key += kThreads) {
    local_sum += expf(scores[base + key] * attention_scale - maximum);
  }
  const float inverse = 1.0f / block_reduce_sum(local_sum, smem);
  for (int key = threadIdx.x; key < keys; key += kThreads) {
    const float weight = expf(scores[base + key] * attention_scale - maximum);
    probabilities[base + key] = to_bf16(weight * inverse);
  }
}

__global__ void attention_output_kernel(const float* __restrict__ tiles,
                                        Bf16* __restrict__ output,
                                        int query_begin, int queries) {
  const size_t pairs = static_cast<size_t>(queries) * (kDim / 2);
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= pairs) return;
  const int query = static_cast<int>(i / (kDim / 2));
  const int channel_pair = static_cast<int>(i % (kDim / 2));
  const int head = channel_pair / (kHeadDim / 2);
  const int d_pair = channel_pair % (kHeadDim / 2);
  const size_t source_pair =
      (static_cast<size_t>(head) * queries + query) * (kHeadDim / 2) + d_pair;
  const float2 v = reinterpret_cast<const float2*>(tiles)[source_pair];
  reinterpret_cast<Bf16x2*>(output)[static_cast<size_t>(query_begin + query) *
                                        (kDim / 2) +
                                    channel_pair] =
      __floats2bfloat162_rn(v.x, v.y);
}

// Fused linear output bias + optional gated residual add (vectorized Bf16x2).
__global__ void residual_bias_kernel(Bf16* __restrict__ hidden,
                                     const Bf16* __restrict__ update,
                                     const float* __restrict__ bias,
                                     const float* __restrict__ gate,
                                     size_t pairs) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= pairs) return;
  const int channel_pair = static_cast<int>(i % (kDim / 2));
  const float2 h = __bfloat1622float2(reinterpret_cast<const Bf16x2*>(hidden)[i]);
  const float2 u = __bfloat1622float2(reinterpret_cast<const Bf16x2*>(update)[i]);
  const float2 b = reinterpret_cast<const float2*>(bias)[channel_pair];
  const float2 ub =
      __bfloat1622float2(__floats2bfloat162_rn(u.x + b.x, u.y + b.y));
  if (gate != nullptr) {
    const float2 g = reinterpret_cast<const float2*>(gate)[channel_pair];
    reinterpret_cast<Bf16x2*>(hidden)[i] =
        __floats2bfloat162_rn(h.x + ub.x * g.x, h.y + ub.y * g.y);
  } else {
    reinterpret_cast<Bf16x2*>(hidden)[i] =
        __floats2bfloat162_rn(h.x + ub.x, h.y + ub.y);
  }
}

__global__ void modulation_kernel(const float* __restrict__ table,
                                  const Bf16* __restrict__ temb,
                                  float* __restrict__ output, int rows) {
  const size_t pairs = static_cast<size_t>(rows) * (kDim / 2);
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= pairs) return;
  const float2 t = reinterpret_cast<const float2*>(table)[i];
  const float2 e = __bfloat1622float2(reinterpret_cast<const Bf16x2*>(temb)[i]);
  reinterpret_cast<float2*>(output)[i] = make_float2(t.x + e.x, t.y + e.y);
}

__global__ void output_modulation_kernel(const float* __restrict__ table,
                                         const Bf16* __restrict__ temb,
                                         float* __restrict__ output) {
  const size_t pairs = static_cast<size_t>(kDim);  // 2 * kDim / 2
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= pairs) return;
  const float2 t = reinterpret_cast<const float2*>(table)[i];
  const float2 e =
      __bfloat1622float2(reinterpret_cast<const Bf16x2*>(temb)[i % (kDim / 2)]);
  reinterpret_cast<float2*>(output)[i] = make_float2(t.x + e.x, t.y + e.y);
}


// Wan2.2-TI2V-5B expand_timesteps conditioning: tokens below
// `first_frame_tokens` belong to latent frame 0 and carry timestep 0, all other
// tokens carry the current step timestep. Two embedding groups are therefore
// enough, and these kernels select the group per token exactly like Diffusers'
// per-token `scale_shift_table + temb` broadcast.
__global__ void modulation_pair_kernel(const float* __restrict__ table,
                                       const Bf16* __restrict__ time_proj,
                                       float* __restrict__ output,
                                       int group_stride) {
  const size_t group_pairs = static_cast<size_t>(group_stride / 2);
  const size_t pairs = static_cast<size_t>(2) * group_pairs;
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= pairs) return;
  const float2 t = reinterpret_cast<const float2*>(table)[i % group_pairs];
  const float2 e =
      __bfloat1622float2(reinterpret_cast<const Bf16x2*>(time_proj)[i]);
  reinterpret_cast<float2*>(output)[i] = make_float2(t.x + e.x, t.y + e.y);
}

__global__ void output_modulation_pair_kernel(const float* __restrict__ table,
                                              const Bf16* __restrict__ temb,
                                              float* __restrict__ output) {
  const size_t pairs = static_cast<size_t>(2) * kDim;  // 2 * 2 * kDim / 2
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= pairs) return;
  const size_t group = i / kDim;
  const size_t offset = i % kDim;
  const float2 t = reinterpret_cast<const float2*>(table)[offset];
  const float2 e = __bfloat1622float2(
      reinterpret_cast<const Bf16x2*>(temb)[group * (kDim / 2) + offset % (kDim / 2)]);
  reinterpret_cast<float2*>(output)[i] = make_float2(t.x + e.x, t.y + e.y);
}

__global__ void layer_norm_pair_kernel(const Bf16* __restrict__ input,
                                       Bf16* __restrict__ output,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ beta,
                                       const float* __restrict__ modulation,
                                       int rows, int scale_row, int shift_row,
                                       int total_tokens, int tokens_per_batch,
                                       float eps, int first_frame_tokens) {
  const int token = blockIdx.x;
  if (token >= total_tokens) return;
  const int group =
      wan_dit_batched_per_frame_group(token, tokens_per_batch, first_frame_tokens);
  const float* mod = modulation + static_cast<size_t>(group) * rows * kDim;
  const float2* scale2 = reinterpret_cast<const float2*>(mod + scale_row * kDim);
  const float2* shift2 = reinterpret_cast<const float2*>(mod + shift_row * kDim);
  const float2* gamma2 = reinterpret_cast<const float2*>(gamma);
  const float2* beta2 = reinterpret_cast<const float2*>(beta);
  const Bf16x2* in_pairs =
      reinterpret_cast<const Bf16x2*>(input + static_cast<size_t>(token) * kDim);
  Bf16x2* out_pairs =
      reinterpret_cast<Bf16x2*>(output + static_cast<size_t>(token) * kDim);

  __shared__ float smem[9];
  float vals[12];
  float local = 0.0f;
  #pragma unroll
  for (int j = 0; j < 6; ++j) {
    const int p = threadIdx.x + j * kThreads;
    const float2 v = __bfloat1622float2(in_pairs[p]);
    vals[j * 2] = v.x;
    vals[j * 2 + 1] = v.y;
    local += v.x + v.y;
  }
  const float mean = block_reduce_sum(local, smem) / static_cast<float>(kDim);

  float local_var = 0.0f;
  #pragma unroll
  for (int j = 0; j < 12; ++j) {
    const float centered = vals[j] - mean;
    local_var += centered * centered;
  }
  const float inverse =
      rsqrtf(block_reduce_sum(local_var, smem) / static_cast<float>(kDim) + eps);

  #pragma unroll
  for (int j = 0; j < 6; ++j) {
    const int p = threadIdx.x + j * kThreads;
    float v0 = (vals[j * 2] - mean) * inverse;
    float v1 = (vals[j * 2 + 1] - mean) * inverse;
    if (gamma != nullptr) {
      const float2 g = gamma2[p];
      const float2 b = beta2[p];
      v0 = v0 * g.x + b.x;
      v1 = v1 * g.y + b.y;
    }
    const float2 sc = scale2[p];
    const float2 sh = shift2[p];
    v0 = v0 * (1.0f + sc.x) + sh.x;
    v1 = v1 * (1.0f + sc.y) + sh.y;
    out_pairs[p] = __floats2bfloat162_rn(v0, v1);
  }
}

__global__ void residual_bias_pair_kernel(Bf16* __restrict__ hidden,
                                          const Bf16* __restrict__ update,
                                          const float* __restrict__ bias,
                                          const float* __restrict__ modulation,
                                          int rows, int gate_row, size_t pairs,
                                          int tokens_per_batch,
                                          int first_frame_tokens) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= pairs) return;
  const int global_token = static_cast<int>(i / (kDim / 2));
  const int channel_pair = static_cast<int>(i % (kDim / 2));
  const int group = wan_dit_batched_per_frame_group(
      global_token, tokens_per_batch, first_frame_tokens);
  const float2 h = __bfloat1622float2(reinterpret_cast<const Bf16x2*>(hidden)[i]);
  const float2 u = __bfloat1622float2(reinterpret_cast<const Bf16x2*>(update)[i]);
  const float2 b = reinterpret_cast<const float2*>(bias)[channel_pair];
  const float2 ub =
      __bfloat1622float2(__floats2bfloat162_rn(u.x + b.x, u.y + b.y));
  const float2 g = reinterpret_cast<const float2*>(
      modulation + (static_cast<size_t>(group) * rows + gate_row) * kDim)[channel_pair];
  reinterpret_cast<Bf16x2*>(hidden)[i] =
      __floats2bfloat162_rn(h.x + ub.x * g.x, h.y + ub.y * g.y);
}

// TeaCache residual kernels:
//   - sub_residual_bf16_kernel: residual_inout holds x_in before the 30 blocks;
//     in-place computes residual = x_after - x_in.
//   - add_residual_bf16_kernel: on a TeaCache hit, skips all 30 blocks and
//     adds x_inout = x_in + prev_residual.
__global__ void sub_residual_bf16_kernel(Bf16* __restrict__ residual_inout,
                                         const Bf16* __restrict__ x_after,
                                         size_t count) {
  const size_t pairs = count >> 1;
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < pairs) {
    const float2 a = __bfloat1622float2(reinterpret_cast<const Bf16x2*>(x_after)[i]);
    const float2 r =
        __bfloat1622float2(reinterpret_cast<const Bf16x2*>(residual_inout)[i]);
    reinterpret_cast<Bf16x2*>(residual_inout)[i] =
        __floats2bfloat162_rn(a.x - r.x, a.y - r.y);
  }
  if (i == 0 && (count & 1u) != 0) {
    residual_inout[count - 1] =
        to_bf16(to_float(x_after[count - 1]) - to_float(residual_inout[count - 1]));
  }
}

__global__ void add_residual_bf16_kernel(Bf16* __restrict__ x_inout,
                                         const Bf16* __restrict__ residual,
                                         size_t count) {
  const size_t pairs = count >> 1;
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < pairs) {
    const float2 x = __bfloat1622float2(reinterpret_cast<const Bf16x2*>(x_inout)[i]);
    const float2 r =
        __bfloat1622float2(reinterpret_cast<const Bf16x2*>(residual)[i]);
    reinterpret_cast<Bf16x2*>(x_inout)[i] =
        __floats2bfloat162_rn(x.x + r.x, x.y + r.y);
  }
  if (i == 0 && (count & 1u) != 0) {
    x_inout[count - 1] =
        to_bf16(to_float(x_inout[count - 1]) + to_float(residual[count - 1]));
  }
}

}  // namespace

struct WanDit::Impl {
  struct WorkspaceBuffers {
    int frames = 0;
    int height = 0;
    int width = 0;
    int text_len = 0;
    int batch_capacity = 0;

    // Fixed-capacity buffers (max 2 rows: scalar T2V uses 1 row, per-frame I2V uses 2).
    DeviceBuffer time_input;
    DeviceBuffer time_a;
    DeviceBuffer time_b;
    DeviceBuffer temb;
    DeviceBuffer temb_act;
    DeviceBuffer time_proj;
    DeviceBuffer modulation;

    // Text-projection scratch (used on cross-attention K/V cache miss).
    DeviceBuffer context_fp32;
    DeviceBuffer text_input;
    DeviceBuffer text_a;
    DeviceBuffer text_projected;

    // Latent and transformer activations sized for (frames, height, width, text_len).
    DeviceBuffer patch_fp32;
    DeviceBuffer patch_tokens;
    DeviceBuffer x;
    DeviceBuffer normed;
    DeviceBuffer q;
    DeviceBuffer k;
    DeviceBuffer v;
    DeviceBuffer attended;
    DeviceBuffer ff;
    DeviceBuffer scores;
    DeviceBuffer probabilities;
    DeviceBuffer output_tiles;
  };

  struct RopeCache {
    int frames = 0;
    int height = 0;
    int width = 0;
    DeviceBuffer cos;
    DeviceBuffer sin;
  };

  struct CrossAttnKvEntry {
    WanTextEmbedCacheKey key{};
    bool valid = false;
    uint64_t last_used = 0;
    // Contiguous [kLayers, text_len, kDim] BF16 post-RMSNorm K and V tensors.
    DeviceBuffer k_all;
    DeviceBuffer v_all;
  };

  struct LayerWeights {
    const float* scale_shift_table = nullptr;
    const Bf16* attn1_q_w = nullptr;
    const float* attn1_q_b = nullptr;
    const Bf16* attn1_k_w = nullptr;
    const float* attn1_k_b = nullptr;
    const Bf16* attn1_v_w = nullptr;
    const float* attn1_v_b = nullptr;
    const float* attn1_norm_q_w = nullptr;
    const float* attn1_norm_k_w = nullptr;
    const Bf16* attn1_out_w = nullptr;
    const float* attn1_out_b = nullptr;

    const float* norm2_w = nullptr;
    const float* norm2_b = nullptr;
    const Bf16* attn2_q_w = nullptr;
    const float* attn2_q_b = nullptr;
    const Bf16* attn2_k_w = nullptr;
    const float* attn2_k_b = nullptr;
    const Bf16* attn2_v_w = nullptr;
    const float* attn2_v_b = nullptr;
    const float* attn2_norm_q_w = nullptr;
    const float* attn2_norm_k_w = nullptr;
    const Bf16* attn2_out_w = nullptr;
    const float* attn2_out_b = nullptr;

    const Bf16* ff0_w = nullptr;
    const float* ff0_b = nullptr;
    const Bf16* ff2_w = nullptr;
    const float* ff2_b = nullptr;
  };

  Impl(const std::filesystem::path& model_dir, int gpu)
      : context_(gpu),
        gpu_(gpu),
        use_flash_attn_(wan_dit_use_flash_attn(std::getenv("UMM_DIT_ATTN_REF"))) {
    config_ = load_wan_dit_config(model_dir.string());
    const auto started = std::chrono::steady_clock::now();
    load_weights(model_dir / "transformer");
    load_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
  }

  ~Impl() {
    (void)cudaSetDevice(gpu_);
    (void)cudaStreamSynchronize(context_.stream());
  }

  void add_spec(std::vector<LoadSpec>& specs, const std::string& name,
                std::vector<int64_t> shape) {
    specs.push_back({name, std::move(shape), 0, keep_fp32(name), false});
  }

  void load_weights(const std::filesystem::path& directory) {
    std::vector<LoadSpec> specs;
    const auto add = [&](const std::string& name, std::vector<int64_t> shape) {
      add_spec(specs, name, std::move(shape));
    };
    add("patch_embedding.weight", {kDim, kChannels, 1, 2, 2});
    specs.back().reorder_patch = true;
    add("patch_embedding.bias", {kDim});
    add("condition_embedder.time_embedder.linear_1.weight", {kDim, kFreqDim});
    add("condition_embedder.time_embedder.linear_1.bias", {kDim});
    add("condition_embedder.time_embedder.linear_2.weight", {kDim, kDim});
    add("condition_embedder.time_embedder.linear_2.bias", {kDim});
    add("condition_embedder.time_proj.weight", {6 * kDim, kDim});
    add("condition_embedder.time_proj.bias", {6 * kDim});
    add("condition_embedder.text_embedder.linear_1.weight", {kDim, kTextDim});
    add("condition_embedder.text_embedder.linear_1.bias", {kDim});
    add("condition_embedder.text_embedder.linear_2.weight", {kDim, kDim});
    add("condition_embedder.text_embedder.linear_2.bias", {kDim});
    for (int layer = 0; layer < kLayers; ++layer) {
      const std::string prefix = "blocks." + std::to_string(layer) + ".";
      for (const char* attention : {"attn1", "attn2"}) {
        const std::string a = prefix + attention + ".";
        for (const char* projection : {"to_q", "to_k", "to_v", "to_out.0"}) {
          add(a + projection + ".weight", {kDim, kDim});
          add(a + projection + ".bias", {kDim});
        }
        add(a + "norm_q.weight", {kDim});
        add(a + "norm_k.weight", {kDim});
      }
      add(prefix + "ffn.net.0.proj.weight", {kFf, kDim});
      add(prefix + "ffn.net.0.proj.bias", {kFf});
      add(prefix + "ffn.net.2.weight", {kDim, kFf});
      add(prefix + "ffn.net.2.bias", {kDim});
      add(prefix + "norm2.weight", {kDim});
      add(prefix + "norm2.bias", {kDim});
      add(prefix + "scale_shift_table", {1, 6, kDim});
    }
    add("proj_out.weight", {kProjection, kDim});
    add("proj_out.bias", {kProjection});
    add("scale_shift_table", {1, 2, kDim});
    if (specs.size() != 825) throw std::logic_error("Wan DiT tensor specification is incomplete");

    ShardedSafetensorsMmap shards(directory / "diffusion_pytorch_model.safetensors.index.json");
    size_t bf16_total = 0;
    size_t f32_total = 0;
    for (auto& spec : specs) {
      const auto& view = shards.tensor(spec.name);
      if (view.shape != spec.shape || view.dtype != SafetensorsDType::kF32) {
        throw std::invalid_argument("unexpected Wan DiT tensor shape or dtype: " + spec.name);
      }
      const bool matrix = spec.name.ends_with(".weight") && spec.shape.size() >= 2 &&
                          !spec.name.ends_with("scale_shift_table");
      spec.fp32 = spec.fp32 || !matrix;
      size_t& total = spec.fp32 ? f32_total : bf16_total;
      const size_t element_size = spec.fp32 ? sizeof(float) : sizeof(Bf16);
      if (total > std::numeric_limits<size_t>::max() - 255) {
        throw std::overflow_error("Wan DiT weight offset overflow");
      }
      spec.offset = (total + 255) & ~size_t{255};
      total = spec.offset;
      const size_t bytes = checked_bytes(view.numel(), element_size);
      if (bytes > std::numeric_limits<size_t>::max() - total) {
        throw std::overflow_error("Wan DiT weight arena overflow");
      }
      total += bytes;
    }
    bf16_weights_.allocate(bf16_total, tracker_);
    f32_weights_.allocate(f32_total, tracker_);

    for (const auto& spec : specs) {
      const auto& view = shards.tensor(spec.name);
      const size_t count = view.numel();
      void* destination = spec.fp32
          ? static_cast<void*>(f32_weights_.as<std::byte>() + spec.offset)
          : static_cast<void*>(bf16_weights_.as<std::byte>() + spec.offset);
      if (spec.fp32 && keep_fp32(spec.name) && !spec.reorder_patch) {
        UMM_DIT_CUDA(cudaMemcpy(destination, view.data, view.size_bytes,
                                cudaMemcpyHostToDevice));
      } else if (spec.fp32) {
        std::vector<float> converted(count);
        for (size_t i = 0; i < count; ++i) {
          const float value = bits_to_f32(load_u32(view.data + i * 4));
          converted[i] = bf16_bits_to_f32(f32_to_bf16_bits(value));
        }
        UMM_DIT_CUDA(cudaMemcpy(destination, converted.data(), checked_bytes(count, 4),
                                cudaMemcpyHostToDevice));
      } else {
        std::vector<uint16_t> converted(count);
        if (spec.reorder_patch) {
          for (int output = 0; output < kDim; ++output) {
            for (int dy = 0; dy < 2; ++dy) {
              for (int dx = 0; dx < 2; ++dx) {
                for (int channel = 0; channel < kChannels; ++channel) {
                  const size_t target = static_cast<size_t>(output) * kChannels * 4 +
                                        (dy * 2 + dx) * kChannels + channel;
                  const size_t source = static_cast<size_t>(output) * kChannels * 4 +
                                        (channel * 2 + dy) * 2 + dx;
                  converted[target] = f32_to_bf16_bits(
                      bits_to_f32(load_u32(view.data + source * 4)));
                }
              }
            }
          }
        } else {
          for (size_t i = 0; i < count; ++i) {
            converted[i] = f32_to_bf16_bits(bits_to_f32(load_u32(view.data + i * 4)));
          }
        }
        UMM_DIT_CUDA(cudaMemcpy(destination, converted.data(), checked_bytes(count, 2),
                                cudaMemcpyHostToDevice));
      }
      tensors_.emplace(spec.name, DeviceTensor{destination, spec.fp32, spec.shape});
    }
    for (int layer = 0; layer < kLayers; ++layer) {
      const std::string p = "blocks." + std::to_string(layer) + ".";
      auto& lw = layers_[static_cast<size_t>(layer)];
      lw.scale_shift_table = static_cast<const float*>(tensor(p + "scale_shift_table").data);
      lw.attn1_q_w = static_cast<const Bf16*>(tensor(p + "attn1.to_q.weight").data);
      lw.attn1_q_b = static_cast<const float*>(tensor(p + "attn1.to_q.bias").data);
      lw.attn1_k_w = static_cast<const Bf16*>(tensor(p + "attn1.to_k.weight").data);
      lw.attn1_k_b = static_cast<const float*>(tensor(p + "attn1.to_k.bias").data);
      lw.attn1_v_w = static_cast<const Bf16*>(tensor(p + "attn1.to_v.weight").data);
      lw.attn1_v_b = static_cast<const float*>(tensor(p + "attn1.to_v.bias").data);
      lw.attn1_norm_q_w = static_cast<const float*>(tensor(p + "attn1.norm_q.weight").data);
      lw.attn1_norm_k_w = static_cast<const float*>(tensor(p + "attn1.norm_k.weight").data);
      lw.attn1_out_w = static_cast<const Bf16*>(tensor(p + "attn1.to_out.0.weight").data);
      lw.attn1_out_b = static_cast<const float*>(tensor(p + "attn1.to_out.0.bias").data);

      lw.norm2_w = static_cast<const float*>(tensor(p + "norm2.weight").data);
      lw.norm2_b = static_cast<const float*>(tensor(p + "norm2.bias").data);
      lw.attn2_q_w = static_cast<const Bf16*>(tensor(p + "attn2.to_q.weight").data);
      lw.attn2_q_b = static_cast<const float*>(tensor(p + "attn2.to_q.bias").data);
      lw.attn2_k_w = static_cast<const Bf16*>(tensor(p + "attn2.to_k.weight").data);
      lw.attn2_k_b = static_cast<const float*>(tensor(p + "attn2.to_k.bias").data);
      lw.attn2_v_w = static_cast<const Bf16*>(tensor(p + "attn2.to_v.weight").data);
      lw.attn2_v_b = static_cast<const float*>(tensor(p + "attn2.to_v.bias").data);
      lw.attn2_norm_q_w = static_cast<const float*>(tensor(p + "attn2.norm_q.weight").data);
      lw.attn2_norm_k_w = static_cast<const float*>(tensor(p + "attn2.norm_k.weight").data);
      lw.attn2_out_w = static_cast<const Bf16*>(tensor(p + "attn2.to_out.0.weight").data);
      lw.attn2_out_b = static_cast<const float*>(tensor(p + "attn2.to_out.0.bias").data);

      lw.ff0_w = static_cast<const Bf16*>(tensor(p + "ffn.net.0.proj.weight").data);
      lw.ff0_b = static_cast<const float*>(tensor(p + "ffn.net.0.proj.bias").data);
      lw.ff2_w = static_cast<const Bf16*>(tensor(p + "ffn.net.2.weight").data);
      lw.ff2_b = static_cast<const float*>(tensor(p + "ffn.net.2.bias").data);
    }
    UMM_DIT_CUDA(cudaDeviceSynchronize());
  }

  const DeviceTensor& tensor(const std::string& name) const { return tensors_.at(name); }

  void gemm_bf16(const void* weight, int input_features, int output_features,
                 int rows, const void* input, void* output) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    UMM_DIT_CUBLAS(cublasGemmEx(
        context_.get(), CUBLAS_OP_T, CUBLAS_OP_N, output_features, rows,
        input_features, &alpha, weight, CUDA_R_16BF, input_features,
        input, CUDA_R_16BF, input_features, &beta, output, CUDA_R_16BF,
        output_features, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  }

  void linear_bf16(const Bf16* weight, const float* bias, int input_features,
                   int output_features, int rows, const void* input,
                   void* output) {
    gemm_bf16(weight, input_features, output_features, rows, input, output);
    const size_t count = static_cast<size_t>(rows) * output_features;
    const cudaStream_t stream = context_.stream();
    if ((output_features & 1) == 0) {
      const size_t pairs = count >> 1;
      bias_bf16_vec2_kernel<<<grid_for(pairs), kThreads, 0, stream>>>(
          static_cast<Bf16*>(output), bias, output_features >> 1, pairs);
    } else {
      bias_bf16_kernel<<<grid_for(count), kThreads, 0, stream>>>(
          static_cast<Bf16*>(output), bias, output_features, count);
    }
    UMM_DIT_CUDA(cudaGetLastError());
  }

  void linear(const std::string& name, int input_features, int output_features,
              int rows, const void* input, void* output) {
    const auto& weight = tensor(name + ".weight");
    const auto& bias = tensor(name + ".bias");
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const size_t count = static_cast<size_t>(rows) * output_features;
    if (weight.fp32) {
      UMM_DIT_CUBLAS(cublasSgemm(
          context_.get(), CUBLAS_OP_T, CUBLAS_OP_N, output_features, rows,
          input_features, &alpha, static_cast<const float*>(weight.data),
          input_features, static_cast<const float*>(input), input_features, &beta,
          static_cast<float*>(output), output_features));
      bias_f32_kernel<<<grid_for(count), kThreads, 0, context_.stream()>>>(
          static_cast<float*>(output), static_cast<const float*>(bias.data),
          output_features, count);
      UMM_DIT_CUDA(cudaGetLastError());
    } else {
      linear_bf16(static_cast<const Bf16*>(weight.data),
                  static_cast<const float*>(bias.data), input_features,
                  output_features, rows, input, output);
    }
  }

  void attention(const Bf16* query, const Bf16* key, const Bf16* value,
                 int tokens, int context_tokens, int batch_size, Bf16* output,
                 DeviceBuffer& scores, DeviceBuffer& probabilities,
                 DeviceBuffer& output_tiles) {
    const cudaStream_t stream = context_.stream();
    if (use_flash_attn_ &&
        wan_dit_flash_attn_supported(kHeadDim, tokens, context_tokens)) {
      const dim3 grid(static_cast<unsigned>(wan_dit_flash_grid_q(tokens)),
                      static_cast<unsigned>(kHeads),
                      static_cast<unsigned>(batch_size));
      flash_attention_bf16_kernel<<<grid, kWanDitFlashThreads, 0, stream>>>(
          query, key, value, output, tokens, context_tokens);
      UMM_DIT_CUDA(cudaGetLastError());
      return;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const size_t q_batch_stride = static_cast<size_t>(tokens) * kDim;
    const size_t kv_batch_stride = static_cast<size_t>(context_tokens) * kDim;
    for (int b = 0; b < batch_size; ++b) {
      const Bf16* q_b = query + static_cast<size_t>(b) * q_batch_stride;
      const Bf16* k_b = key + static_cast<size_t>(b) * kv_batch_stride;
      const Bf16* v_b = value + static_cast<size_t>(b) * kv_batch_stride;
      Bf16* out_b = output + static_cast<size_t>(b) * q_batch_stride;
      for (int begin = 0; begin < tokens; begin += kQueryTile) {
        const int queries = wan_attention_tile_len(tokens, begin, kQueryTile);
        UMM_DIT_CUBLAS(cublasGemmStridedBatchedEx(
            context_.get(), CUBLAS_OP_T, CUBLAS_OP_N, context_tokens, queries,
            kHeadDim, &alpha, k_b, CUDA_R_16BF, kDim, kHeadDim,
            q_b + static_cast<size_t>(begin) * kDim, CUDA_R_16BF, kDim, kHeadDim,
            &beta, scores.as<float>(), CUDA_R_32F, context_tokens,
            static_cast<int64_t>(context_tokens) * queries, kHeads,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        attention_softmax_kernel<<<kHeads * queries, kThreads, 0, stream>>>(
            scores.as<float>(), probabilities.as<Bf16>(), context_tokens, queries);
        UMM_DIT_CUDA(cudaGetLastError());
        UMM_DIT_CUBLAS(cublasGemmStridedBatchedEx(
            context_.get(), CUBLAS_OP_N, CUBLAS_OP_N, kHeadDim, queries,
            context_tokens, &alpha, v_b, CUDA_R_16BF, kDim, kHeadDim,
            probabilities.as<Bf16>(), CUDA_R_16BF, context_tokens,
            static_cast<int64_t>(context_tokens) * queries, &beta,
            output_tiles.as<float>(), CUDA_R_32F, kHeadDim,
            static_cast<int64_t>(kHeadDim) * queries, kHeads,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        const size_t out_pairs = static_cast<size_t>(queries) * (kDim / 2);
        attention_output_kernel<<<grid_for(out_pairs), kThreads, 0, stream>>>(
            output_tiles.as<float>(), out_b, begin, queries);
        UMM_DIT_CUDA(cudaGetLastError());
      }
    }
  }

  void ensure_workspace(int frames, int height, int width, int text_len,
                        int batch_size) {
    if (ws_.frames == frames && ws_.height == height && ws_.width == width &&
        ws_.text_len >= text_len && ws_.batch_capacity >= batch_size) {
      return;
    }
    const bool same_shape =
        (ws_.frames == frames && ws_.height == height && ws_.width == width);
    const int alloc_text_len =
        same_shape ? std::max(ws_.text_len, text_len) : text_len;
    const int alloc_batch =
        same_shape ? std::max(ws_.batch_capacity, batch_size) : batch_size;

    ws_.frames = 0;
    ws_.height = 0;
    ws_.width = 0;
    ws_.text_len = 0;
    ws_.batch_capacity = 0;

    constexpr size_t kMaxTimeRows = 2;
    const int tokens = frames * (height / 2) * (width / 2);
    const size_t batched_tokens =
        static_cast<size_t>(alloc_batch) * static_cast<size_t>(tokens);
    const size_t model_count = batched_tokens * kDim;
    const size_t text_count = static_cast<size_t>(alloc_text_len) * kDim;
    const size_t context_count = static_cast<size_t>(alloc_text_len) * kTextDim;
    const size_t patch_count = static_cast<size_t>(tokens) * kProjection;
    const int query_tile = std::min(kQueryTile, tokens);
    const int attention_keys = std::max(tokens, alloc_text_len);

    ws_.time_input.resize(checked_bytes(kFreqDim * kMaxTimeRows, sizeof(float)), tracker_);
    ws_.time_a.resize(checked_bytes(kDim * kMaxTimeRows, sizeof(float)), tracker_);
    ws_.time_b.resize(checked_bytes(kDim * kMaxTimeRows, sizeof(float)), tracker_);
    ws_.temb.resize(checked_bytes(kDim * kMaxTimeRows, sizeof(Bf16)), tracker_);
    ws_.temb_act.resize(checked_bytes(kDim * kMaxTimeRows, sizeof(Bf16)), tracker_);
    ws_.time_proj.resize(checked_bytes(6 * kDim * kMaxTimeRows, sizeof(Bf16)), tracker_);
    ws_.modulation.resize(checked_bytes(6 * kDim * kMaxTimeRows, sizeof(float)), tracker_);

    ws_.context_fp32.resize(checked_bytes(context_count, sizeof(float)), tracker_);
    ws_.text_input.resize(checked_bytes(context_count, sizeof(Bf16)), tracker_);
    ws_.text_a.resize(checked_bytes(text_count, sizeof(Bf16)), tracker_);
    ws_.text_projected.resize(checked_bytes(text_count, sizeof(Bf16)), tracker_);

    ws_.patch_fp32.resize(checked_bytes(patch_count, sizeof(float)), tracker_);
    ws_.patch_tokens.resize(checked_bytes(patch_count, sizeof(Bf16)), tracker_);
    ws_.x.resize(checked_bytes(model_count, sizeof(Bf16)), tracker_);
    ws_.normed.resize(checked_bytes(model_count, sizeof(Bf16)), tracker_);
    ws_.q.resize(checked_bytes(model_count, sizeof(Bf16)), tracker_);
    ws_.k.resize(checked_bytes(model_count, sizeof(Bf16)), tracker_);
    ws_.v.resize(checked_bytes(model_count, sizeof(Bf16)), tracker_);
    ws_.attended.resize(checked_bytes(model_count, sizeof(Bf16)), tracker_);
    ws_.ff.resize(checked_bytes(batched_tokens * kFf, sizeof(Bf16)), tracker_);
    if (!use_flash_attn_) {
      ws_.scores.resize(
          checked_bytes(static_cast<size_t>(kHeads) * query_tile * attention_keys, sizeof(float)),
          tracker_);
      ws_.probabilities.resize(
          checked_bytes(static_cast<size_t>(kHeads) * query_tile * attention_keys, sizeof(Bf16)),
          tracker_);
      ws_.output_tiles.resize(
          checked_bytes(static_cast<size_t>(query_tile) * kDim, sizeof(float)), tracker_);
    } else {
      ws_.scores.reset();
      ws_.probabilities.reset();
      ws_.output_tiles.reset();
    }

    ws_.frames = frames;
    ws_.height = height;
    ws_.width = width;
    ws_.text_len = alloc_text_len;
    ws_.batch_capacity = alloc_batch;
  }

  void ensure_pinned_host(size_t latent_count) {
    h_time_pinned_.resize(checked_bytes(static_cast<size_t>(2) * kFreqDim, sizeof(float)));
    h_latent_pinned_.resize(checked_bytes(latent_count, sizeof(float)));
  }

  void ensure_rope(int frames, int height, int width) {
    if (rope_.frames == frames && rope_.height == height && rope_.width == width) {
      return;
    }
    rope_.frames = 0;
    rope_.height = 0;
    rope_.width = 0;

    const auto rope = wan_build_rope_3d(frames, height, width, kHeadDim);
    const size_t cos_bytes = checked_bytes(rope.cos.size(), sizeof(float));
    const size_t sin_bytes = checked_bytes(rope.sin.size(), sizeof(float));
    rope_.cos.resize(cos_bytes, tracker_);
    rope_.sin.resize(sin_bytes, tracker_);
    const cudaStream_t stream = context_.stream();
    UMM_DIT_CUDA(cudaMemcpyAsync(rope_.cos.data(), rope.cos.data(), cos_bytes,
                                 cudaMemcpyHostToDevice, stream));
    UMM_DIT_CUDA(cudaMemcpyAsync(rope_.sin.data(), rope.sin.data(), sin_bytes,
                                 cudaMemcpyHostToDevice, stream));
    UMM_DIT_CUDA(cudaStreamSynchronize(stream));

    rope_.frames = frames;
    rope_.height = height;
    rope_.width = width;
  }

  const CrossAttnKvEntry& get_or_build_cross_kv(const float* context, int text_len) {
    const WanTextEmbedCacheKey key{
        wan_hash_text_embeds(context, text_len, kTextDim), text_len, kTextDim};
    for (auto& entry : cross_kv_cache_) {
      if (entry.valid && entry.key == key) {
        entry.last_used = ++cross_kv_clock_;
        return entry;
      }
    }

    size_t victim = 0;
    for (size_t i = 0; i < kCrossKvCacheSlots; ++i) {
      if (!cross_kv_cache_[i].valid) {
        victim = i;
        break;
      }
      if (cross_kv_cache_[i].last_used < cross_kv_cache_[victim].last_used) {
        victim = i;
      }
    }

    CrossAttnKvEntry& entry = cross_kv_cache_[victim];
    entry.valid = false;

    const size_t context_count = static_cast<size_t>(text_len) * kTextDim;
    const size_t text_count = static_cast<size_t>(text_len) * kDim;
    const size_t all_layers_bytes =
        checked_bytes(static_cast<size_t>(kLayers) * text_count, sizeof(Bf16));
    entry.k_all.resize(all_layers_bytes, tracker_);
    entry.v_all.resize(all_layers_bytes, tracker_);

    const cudaStream_t stream = context_.stream();
    UMM_DIT_CUDA(cudaMemcpyAsync(ws_.context_fp32.data(), context,
                                 checked_bytes(context_count, sizeof(float)),
                                 cudaMemcpyHostToDevice, stream));
    float_to_bf16_kernel<<<grid_for((context_count + 1) / 2), kThreads, 0, stream>>>(
        ws_.context_fp32.as<float>(), ws_.text_input.as<Bf16>(), context_count);
    UMM_DIT_CUDA(cudaGetLastError());
    gemm_bf16(tensor("condition_embedder.text_embedder.linear_1.weight").data,
              kTextDim, kDim, text_len, ws_.text_input.data(), ws_.text_a.data());
    bias_gelu_tanh_bf16_kernel<<<grid_for(text_count / 2), kThreads, 0, stream>>>(
        ws_.text_a.as<Bf16>(),
        static_cast<const float*>(
            tensor("condition_embedder.text_embedder.linear_1.bias").data),
        kDim / 2, text_count / 2);
    UMM_DIT_CUDA(cudaGetLastError());
    linear("condition_embedder.text_embedder.linear_2", kDim, kDim, text_len,
           ws_.text_a.data(), ws_.text_projected.data());

    for (int layer = 0; layer < kLayers; ++layer) {
      const auto& lw = layers_[static_cast<size_t>(layer)];
      Bf16* layer_k = entry.k_all.as<Bf16>() + static_cast<size_t>(layer) * text_count;
      Bf16* layer_v = entry.v_all.as<Bf16>() + static_cast<size_t>(layer) * text_count;
      gemm_bf16(lw.attn2_k_w, kDim, kDim, text_len, ws_.text_projected.data(), layer_k);
      rms_norm_bias_kernel<false><<<text_len, kThreads, 0, stream>>>(
          layer_k, lw.attn2_k_b, lw.attn2_norm_k_w, nullptr, nullptr,
          text_len, text_len, config_.eps);
      UMM_DIT_CUDA(cudaGetLastError());
      linear_bf16(lw.attn2_v_w, lw.attn2_v_b, kDim, kDim, text_len,
                  ws_.text_projected.data(), layer_v);
    }

    entry.key = key;
    entry.last_used = ++cross_kv_clock_;
    entry.valid = true;
    return entry;
  }

  void set_teacache_thresh(float thresh) {
    teacache_.set_threshold(thresh);
    if (!teacache_.enabled()) {
      (void)cudaSetDevice(gpu_);
      (void)cudaStreamSynchronize(context_.stream());
      for (auto& buf : teacache_residuals_) {
        buf.reset();
      }
      teacache_e0_bits_.clear();
      teacache_e0_f32_.clear();
    }
  }

  float teacache_thresh() const { return teacache_.threshold(); }

  void reset_teacache() { teacache_.reset(); }

  int teacache_hits() const { return teacache_.hits(); }

  int teacache_total_calls() const { return teacache_.total_calls(); }

  struct ResidualRollbackGuard {
    TeaCacheController* controller = nullptr;
    int branch = 0;
    bool active = false;
    ~ResidualRollbackGuard() {
      if (active && controller != nullptr) {
        controller->invalidate_branch(branch);
      }
    }
  };

  void compute_time_embeddings(float timestep, bool per_frame) {
    const cudaStream_t stream = context_.stream();
    const int temb_rows = per_frame ? 2 : 1;
    const size_t time_rows = static_cast<size_t>(temb_rows);
    float* h_time = h_time_pinned_.as<float>();
    if (per_frame) {
      const auto zero = wan_sinusoidal_timestep_embedding(0.0f, kFreqDim);
      const auto step = wan_sinusoidal_timestep_embedding(timestep, kFreqDim);
      std::memcpy(h_time, zero.data(), checked_bytes(kFreqDim, sizeof(float)));
      std::memcpy(h_time + kFreqDim, step.data(),
                  checked_bytes(kFreqDim, sizeof(float)));
    } else {
      const auto step = wan_sinusoidal_timestep_embedding(timestep, kFreqDim);
      std::memcpy(h_time, step.data(), checked_bytes(kFreqDim, sizeof(float)));
    }

    UMM_DIT_CUDA(cudaMemcpyAsync(
        ws_.time_input.data(), h_time,
        checked_bytes(kFreqDim * time_rows, sizeof(float)),
        cudaMemcpyHostToDevice, stream));
    linear("condition_embedder.time_embedder.linear_1", kFreqDim, kDim,
           temb_rows, ws_.time_input.data(), ws_.time_a.data());
    silu_f32_kernel<<<grid_for(kDim * time_rows), kThreads, 0, stream>>>(
        ws_.time_a.as<float>(), kDim * time_rows);
    UMM_DIT_CUDA(cudaGetLastError());
    linear("condition_embedder.time_embedder.linear_2", kDim, kDim,
           temb_rows, ws_.time_a.data(), ws_.time_b.data());
    float_to_bf16_kernel<<<grid_for((kDim * time_rows + 1) / 2), kThreads, 0, stream>>>(
        ws_.time_b.as<float>(), ws_.temb.as<Bf16>(), kDim * time_rows);
    UMM_DIT_CUDA(cudaGetLastError());
    if (per_frame) {
      // time_proj consumes the activated embedding; the output norm keeps
      // the unactivated one, so the activated copy gets its own buffer.
      UMM_DIT_CUDA(cudaMemcpyAsync(ws_.temb_act.data(), ws_.temb.data(),
                                   checked_bytes(kDim * time_rows, sizeof(Bf16)),
                                   cudaMemcpyDeviceToDevice, stream));
      silu_bf16_kernel<<<grid_for((kDim * time_rows + 1) / 2), kThreads, 0, stream>>>(
          ws_.temb_act.as<Bf16>(), kDim * time_rows);
      UMM_DIT_CUDA(cudaGetLastError());
      linear("condition_embedder.time_proj", kDim, 6 * kDim, temb_rows,
             ws_.temb_act.data(), ws_.time_proj.data());
    } else {
      silu_bf16_kernel<<<grid_for(kDim / 2), kThreads, 0, stream>>>(
          ws_.temb.as<Bf16>(), kDim);
      UMM_DIT_CUDA(cudaGetLastError());
      linear("condition_embedder.time_proj", kDim, 6 * kDim, 1,
             ws_.temb.data(), ws_.time_proj.data());
      // The output norm uses the unactivated time embedding, so restore temb after time_proj.
      float_to_bf16_kernel<<<grid_for(kDim / 2), kThreads, 0, stream>>>(
          ws_.time_b.as<float>(), ws_.temb.as<Bf16>(), kDim);
      UMM_DIT_CUDA(cudaGetLastError());
    }
  }

  void read_active_teacache_e0(int temb_rows) {
    constexpr size_t kE0Count = static_cast<size_t>(6) * kDim;
    const cudaStream_t stream = context_.stream();
    teacache_e0_bits_.resize(kE0Count);
    teacache_e0_f32_.resize(kE0Count);
    const Bf16* active_e0 =
        ws_.time_proj.as<Bf16>() + static_cast<size_t>(temb_rows - 1) * kE0Count;
    UMM_DIT_CUDA(cudaMemcpyAsync(teacache_e0_bits_.data(), active_e0,
                                 checked_bytes(kE0Count, sizeof(Bf16)),
                                 cudaMemcpyDeviceToHost, stream));
    UMM_DIT_CUDA(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < kE0Count; ++i) {
      teacache_e0_f32_[i] = bf16_bits_to_f32(teacache_e0_bits_[i]);
    }
  }

  void patchify_and_embed_latent(const float* latent, int frames, int height,
                                 int width, int tokens, size_t latent_count) {
    const cudaStream_t stream = context_.stream();
    const size_t latent_bytes = checked_bytes(latent_count, sizeof(float));
    std::memcpy(h_latent_pinned_.data(), latent, latent_bytes);
    UMM_DIT_CUDA(cudaMemcpyAsync(ws_.patch_fp32.data(), h_latent_pinned_.data(),
                                 latent_bytes, cudaMemcpyHostToDevice, stream));
    patchify_fp32_to_bf16_kernel<<<grid_for(latent_count), kThreads, 0, stream>>>(
        ws_.patch_fp32.as<float>(), ws_.patch_tokens.as<Bf16>(), frames, height,
        width);
    UMM_DIT_CUDA(cudaGetLastError());
    linear("patch_embedding", kChannels * kPatchVolume, kDim, tokens,
           ws_.patch_tokens.data(), ws_.x.data());
  }

  void run_transformer_blocks(Bf16* x_buf, int tokens, int batch_size,
                              int first_frame_tokens, bool per_frame,
                              const CrossAttnKvEntry& kv0, int len0,
                              const CrossAttnKvEntry* kv1, int len1) {
    const cudaStream_t stream = context_.stream();
    const int total_tokens = batch_size * tokens;
    const size_t model_count_per_batch = static_cast<size_t>(tokens) * kDim;
    const size_t total_model_pairs =
        static_cast<size_t>(total_tokens) * (kDim / 2);
    const size_t total_ff_pairs = static_cast<size_t>(total_tokens) * (kFf / 2);
    const size_t text_count0 = static_cast<size_t>(len0) * kDim;
    const size_t text_count1 = static_cast<size_t>(len1) * kDim;

    // Per-token modulation differs only between latent frame 0 and the
    // remaining frames, so two embedding groups drive the pair kernels.
    const auto apply_norm = [&](const Bf16* input, Bf16* output,
                                const float* gamma, const float* beta,
                                int shift_row, int scale_row, int rows) {
      if (per_frame) {
        layer_norm_pair_kernel<<<total_tokens, kThreads, 0, stream>>>(
            input, output, gamma, beta, ws_.modulation.as<float>(), rows,
            scale_row, shift_row, total_tokens, tokens, config_.eps,
            first_frame_tokens);
      } else {
        const float* mod = ws_.modulation.as<float>();
        layer_norm_kernel<<<total_tokens, kThreads, 0, stream>>>(
            input, output, gamma, beta, mod + shift_row * kDim,
            mod + scale_row * kDim, total_tokens, config_.eps);
      }
      UMM_DIT_CUDA(cudaGetLastError());
    };
    const auto apply_residual_bias = [&](const Bf16* update, const float* bias,
                                         int gate_row) {
      if (per_frame) {
        residual_bias_pair_kernel<<<grid_for(total_model_pairs), kThreads, 0, stream>>>(
            x_buf, update, bias, ws_.modulation.as<float>(), 6, gate_row,
            total_model_pairs, tokens, first_frame_tokens);
      } else {
        residual_bias_kernel<<<grid_for(total_model_pairs), kThreads, 0, stream>>>(
            x_buf, update, bias, ws_.modulation.as<float>() + gate_row * kDim,
            total_model_pairs);
      }
      UMM_DIT_CUDA(cudaGetLastError());
    };

    for (int layer = 0; layer < kLayers; ++layer) {
      const auto& lw = layers_[static_cast<size_t>(layer)];
      if (per_frame) {
        modulation_pair_kernel<<<grid_for(6 * kDim), kThreads, 0, stream>>>(
            lw.scale_shift_table, ws_.time_proj.as<Bf16>(),
            ws_.modulation.as<float>(), 6 * kDim);
      } else {
        modulation_kernel<<<grid_for(3 * kDim), kThreads, 0, stream>>>(
            lw.scale_shift_table, ws_.time_proj.as<Bf16>(),
            ws_.modulation.as<float>(), 6);
      }
      UMM_DIT_CUDA(cudaGetLastError());

      // Self-attention: fused bias + RMSNorm + 3D RoPE on Q/K, FlashAttention-2
      // across all batch slices in one launch, and fused bias + gated residual.
      apply_norm(x_buf, ws_.normed.as<Bf16>(), nullptr, nullptr, 0, 1, 6);
      gemm_bf16(lw.attn1_q_w, kDim, kDim, total_tokens, ws_.normed.data(),
                ws_.q.data());
      rms_norm_bias_kernel<true><<<total_tokens, kThreads, 0, stream>>>(
          ws_.q.as<Bf16>(), lw.attn1_q_b, lw.attn1_norm_q_w,
          rope_.cos.as<float>(), rope_.sin.as<float>(), total_tokens, tokens,
          config_.eps);
      gemm_bf16(lw.attn1_k_w, kDim, kDim, total_tokens, ws_.normed.data(),
                ws_.k.data());
      rms_norm_bias_kernel<true><<<total_tokens, kThreads, 0, stream>>>(
          ws_.k.as<Bf16>(), lw.attn1_k_b, lw.attn1_norm_k_w,
          rope_.cos.as<float>(), rope_.sin.as<float>(), total_tokens, tokens,
          config_.eps);
      UMM_DIT_CUDA(cudaGetLastError());
      linear_bf16(lw.attn1_v_w, lw.attn1_v_b, kDim, kDim, total_tokens,
                  ws_.normed.data(), ws_.v.data());
      attention(ws_.q.as<Bf16>(), ws_.k.as<Bf16>(), ws_.v.as<Bf16>(), tokens,
                tokens, batch_size, ws_.attended.as<Bf16>(), ws_.scores,
                ws_.probabilities, ws_.output_tiles);
      gemm_bf16(lw.attn1_out_w, kDim, kDim, total_tokens, ws_.attended.data(),
                ws_.q.data());
      apply_residual_bias(ws_.q.as<Bf16>(), lw.attn1_out_b, 2);

      // Cross-attention: fused bias + RMSNorm on Q, FlashAttention-2 against
      // cached text K/V (per batch slice), and fused bias + residual on out.
      layer_norm_kernel<<<total_tokens, kThreads, 0, stream>>>(
          x_buf, ws_.normed.as<Bf16>(), lw.norm2_w, lw.norm2_b, nullptr, nullptr,
          total_tokens, config_.eps);
      UMM_DIT_CUDA(cudaGetLastError());
      gemm_bf16(lw.attn2_q_w, kDim, kDim, total_tokens, ws_.normed.data(),
                ws_.q.data());
      rms_norm_bias_kernel<false><<<total_tokens, kThreads, 0, stream>>>(
          ws_.q.as<Bf16>(), lw.attn2_q_b, lw.attn2_norm_q_w, nullptr, nullptr,
          total_tokens, tokens, config_.eps);
      UMM_DIT_CUDA(cudaGetLastError());
      const Bf16* layer_k0 =
          kv0.k_all.as<Bf16>() + static_cast<size_t>(layer) * text_count0;
      const Bf16* layer_v0 =
          kv0.v_all.as<Bf16>() + static_cast<size_t>(layer) * text_count0;
      attention(ws_.q.as<Bf16>(), layer_k0, layer_v0, tokens, len0, 1,
                ws_.attended.as<Bf16>(), ws_.scores, ws_.probabilities,
                ws_.output_tiles);
      if (batch_size == 2 && kv1 != nullptr) {
        const Bf16* layer_k1 =
            kv1->k_all.as<Bf16>() + static_cast<size_t>(layer) * text_count1;
        const Bf16* layer_v1 =
            kv1->v_all.as<Bf16>() + static_cast<size_t>(layer) * text_count1;
        attention(ws_.q.as<Bf16>() + model_count_per_batch, layer_k1, layer_v1,
                  tokens, len1, 1, ws_.attended.as<Bf16>() + model_count_per_batch,
                  ws_.scores, ws_.probabilities, ws_.output_tiles);
      }
      gemm_bf16(lw.attn2_out_w, kDim, kDim, total_tokens, ws_.attended.data(),
                ws_.q.data());
      residual_bias_kernel<<<grid_for(total_model_pairs), kThreads, 0, stream>>>(
          x_buf, ws_.q.as<Bf16>(), lw.attn2_out_b, nullptr, total_model_pairs);
      UMM_DIT_CUDA(cudaGetLastError());

      // Feed-forward: fused bias + GELU on [total_tokens, 14336], and fused
      // bias + gated residual on down-projection.
      apply_norm(x_buf, ws_.normed.as<Bf16>(), nullptr, nullptr, 3, 4, 6);
      gemm_bf16(lw.ff0_w, kDim, kFf, total_tokens, ws_.normed.data(),
                ws_.ff.data());
      bias_gelu_tanh_bf16_kernel<<<grid_for(total_ff_pairs), kThreads, 0, stream>>>(
          ws_.ff.as<Bf16>(), lw.ff0_b, kFf / 2, total_ff_pairs);
      UMM_DIT_CUDA(cudaGetLastError());
      gemm_bf16(lw.ff2_w, kFf, kDim, total_tokens, ws_.ff.data(), ws_.q.data());
      apply_residual_bias(ws_.q.as<Bf16>(), lw.ff2_b, 5);
    }
  }

  void run_output_projection(int tokens, int batch_size, int first_frame_tokens,
                             bool per_frame) {
    const cudaStream_t stream = context_.stream();
    const int total_tokens = batch_size * tokens;
    if (per_frame) {
      output_modulation_pair_kernel<<<grid_for(2 * kDim), kThreads, 0, stream>>>(
          static_cast<const float*>(tensor("scale_shift_table").data),
          ws_.temb.as<Bf16>(), ws_.modulation.as<float>());
      UMM_DIT_CUDA(cudaGetLastError());
      layer_norm_pair_kernel<<<total_tokens, kThreads, 0, stream>>>(
          ws_.x.as<Bf16>(), ws_.normed.as<Bf16>(), nullptr, nullptr,
          ws_.modulation.as<float>(), 2, 1, 0, total_tokens, tokens,
          config_.eps, first_frame_tokens);
    } else {
      output_modulation_kernel<<<grid_for(kDim), kThreads, 0, stream>>>(
          static_cast<const float*>(tensor("scale_shift_table").data),
          ws_.temb.as<Bf16>(), ws_.modulation.as<float>());
      UMM_DIT_CUDA(cudaGetLastError());
      layer_norm_kernel<<<total_tokens, kThreads, 0, stream>>>(
          ws_.x.as<Bf16>(), ws_.normed.as<Bf16>(), nullptr, nullptr,
          ws_.modulation.as<float>(), ws_.modulation.as<float>() + kDim,
          total_tokens, config_.eps);
    }
    UMM_DIT_CUDA(cudaGetLastError());
    linear("proj_out", kDim, kProjection, total_tokens, ws_.normed.data(),
           ws_.q.data());
  }

  WanDitOutput forward(const float* latent, int frames, int height, int width,
                       float timestep, const float* context, int text_len,
                       bool first_frame_zero_timestep,
                       WanDitTeaCacheHint teacache_hint) {
    if (latent == nullptr || context == nullptr || frames <= 0 || height <= 0 ||
        width <= 0 || height % 2 != 0 || width % 2 != 0 || text_len <= 0 ||
        text_len > 512 || !std::isfinite(timestep)) {
      throw std::invalid_argument("invalid Wan DiT forward inputs");
    }
    const auto started = std::chrono::steady_clock::now();
    tracker_.peak = tracker_.current;
    UMM_DIT_CUDA(cudaSetDevice(gpu_));
    const cudaStream_t stream = context_.stream();
    const int64_t tokens64 =
        static_cast<int64_t>(frames) * (height / 2) * (width / 2);
    if (tokens64 <= 0 || tokens64 > std::numeric_limits<int>::max()) {
      throw std::overflow_error("Wan DiT token count overflows int");
    }
    const int tokens = static_cast<int>(tokens64);
    // Wan2.2-TI2V-5B expand_timesteps: latent frame 0 tokens get t=0.
    const bool per_frame = first_frame_zero_timestep;
    const int temb_rows = per_frame ? 2 : 1;
    const int first_frame_tokens = (height / 2) * (width / 2);
    const size_t model_count = static_cast<size_t>(tokens) * kDim;
    const size_t latent_count =
        static_cast<size_t>(kChannels) * frames * height * width;
    const size_t latent_bytes = checked_bytes(latent_count, sizeof(float));

    ensure_workspace(frames, height, width, text_len, /*batch_size=*/1);
    ensure_rope(frames, height, width);
    ensure_pinned_host(latent_count);

    compute_time_embeddings(timestep, per_frame);

    // Opt-in TeaCache decision using the active timestep's 6*kDim time_proj
    // signal (36 KiB D2H). When threshold == 0.0f, zero D2H copy or GPU
    // allocation occurs and execution stays 100% bit-exact with uncached DiT.
    TeaCacheDecision tc_decision{};
    if (teacache_.enabled()) {
      read_active_teacache_e0(temb_rows);
      tc_decision = teacache_.evaluate(
          teacache_hint.branch, teacache_hint.step_index,
          teacache_hint.total_steps, frames, height, width, per_frame,
          teacache_e0_f32_.data(), teacache_e0_f32_.size());
    } else {
      teacache_.record_disabled_call();
    }

    ResidualRollbackGuard rollback_guard{&teacache_, teacache_hint.branch,
                                         tc_decision.save_residual};

    patchify_and_embed_latent(latent, frames, height, width, tokens,
                              latent_count);

    const size_t residual_bytes = checked_bytes(model_count, sizeof(Bf16));
    if (tc_decision.use_cached_residual) {
      const DeviceBuffer& cached_residual =
          teacache_residuals_[static_cast<size_t>(teacache_hint.branch)];
      if (cached_residual.bytes() != residual_bytes) {
        throw std::logic_error("Wan DiT TeaCache residual buffer size mismatch");
      }
      add_residual_bf16_kernel<<<grid_for((model_count + 1) / 2), kThreads, 0, stream>>>(
          ws_.x.as<Bf16>(), cached_residual.as<Bf16>(), model_count);
      UMM_DIT_CUDA(cudaGetLastError());
    } else {
      const CrossAttnKvEntry& cross_kv = get_or_build_cross_kv(context, text_len);
      DeviceBuffer* residual_slot = nullptr;
      if (tc_decision.save_residual) {
        residual_slot =
            &teacache_residuals_[static_cast<size_t>(teacache_hint.branch)];
        residual_slot->resize(residual_bytes, tracker_);
        UMM_DIT_CUDA(cudaMemcpyAsync(residual_slot->data(), ws_.x.data(),
                                     residual_bytes, cudaMemcpyDeviceToDevice,
                                     stream));
      }
      run_transformer_blocks(ws_.x.as<Bf16>(), tokens, /*batch_size=*/1,
                             first_frame_tokens, per_frame, cross_kv, text_len,
                             nullptr, 0);
      if (tc_decision.save_residual) {
        sub_residual_bf16_kernel<<<grid_for((model_count + 1) / 2), kThreads, 0, stream>>>(
            residual_slot->as<Bf16>(), ws_.x.as<Bf16>(), model_count);
        UMM_DIT_CUDA(cudaGetLastError());
      }
    }

    run_output_projection(tokens, /*batch_size=*/1, first_frame_tokens,
                          per_frame);

    // GPU unpatchify directly into ws_.patch_fp32 (exact size latent_count),
    // then async D2H into reusable pinned host memory.
    unpatchify_bf16_to_fp32_kernel<<<grid_for(latent_count), kThreads, 0, stream>>>(
        ws_.q.as<Bf16>(), ws_.patch_fp32.as<float>(), frames, height, width);
    UMM_DIT_CUDA(cudaGetLastError());
    UMM_DIT_CUDA(cudaMemcpyAsync(h_latent_pinned_.data(), ws_.patch_fp32.data(),
                                 latent_bytes, cudaMemcpyDeviceToHost, stream));
    UMM_DIT_CUDA(cudaStreamSynchronize(stream));
    rollback_guard.active = false;

    WanDitOutput result;
    result.velocity.resize(latent_count);
    std::memcpy(result.velocity.data(), h_latent_pinned_.data(), latent_bytes);
    result.stats.load_ms = load_ms_;
    result.stats.forward_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    result.stats.peak_vram_bytes = tracker_.peak;
    result.stats.tensor_count = tensors_.size();
    result.stats.teacache_hit = tc_decision.use_cached_residual;
    return result;
  }

  WanDitCfgOutput forward_cfg(const float* latent, int frames, int height,
                              int width, float timestep,
                              const float* pos_context, int pos_len,
                              const float* neg_context, int neg_len,
                              float guidance_scale,
                              bool first_frame_zero_timestep,
                              WanDitTeaCacheHint cond_hint,
                              WanDitTeaCacheHint uncond_hint) {
    if (latent == nullptr || pos_context == nullptr || neg_context == nullptr ||
        frames <= 0 || height <= 0 || width <= 0 || height % 2 != 0 ||
        width % 2 != 0 || pos_len <= 0 || pos_len > 512 || neg_len <= 0 ||
        neg_len > 512 || !std::isfinite(timestep) ||
        !std::isfinite(guidance_scale)) {
      throw std::invalid_argument("invalid Wan DiT forward_cfg inputs");
    }
    const auto started = std::chrono::steady_clock::now();
    tracker_.peak = tracker_.current;
    UMM_DIT_CUDA(cudaSetDevice(gpu_));
    const cudaStream_t stream = context_.stream();
    const int64_t tokens64 =
        static_cast<int64_t>(frames) * (height / 2) * (width / 2);
    if (tokens64 <= 0 || tokens64 > std::numeric_limits<int>::max() / 2) {
      throw std::overflow_error("Wan DiT batched token count overflows int");
    }
    const int tokens = static_cast<int>(tokens64);
    const bool per_frame = first_frame_zero_timestep;
    const int temb_rows = per_frame ? 2 : 1;
    const int first_frame_tokens = (height / 2) * (width / 2);
    const size_t model_count = static_cast<size_t>(tokens) * kDim;
    const size_t residual_bytes = checked_bytes(model_count, sizeof(Bf16));
    const size_t latent_count =
        static_cast<size_t>(kChannels) * frames * height * width;
    const size_t latent_bytes = checked_bytes(latent_count, sizeof(float));

    ensure_workspace(frames, height, width, std::max(pos_len, neg_len),
                     /*batch_size=*/2);
    ensure_rope(frames, height, width);
    ensure_pinned_host(latent_count);

    // 1. Timestep embedding computed ONCE (shared across cond & uncond).
    compute_time_embeddings(timestep, per_frame);

    // 2. Evaluate TeaCache decisions for both branches from that single time_proj
    // (zero D2H copy when TeaCache is disabled).
    WanDitCfgTeaCachePlan tc_plan{};
    if (teacache_.enabled()) {
      read_active_teacache_e0(temb_rows);
      tc_plan = wan_dit_evaluate_cfg_teacache(
          teacache_, cond_hint, uncond_hint, frames, height, width, per_frame,
          teacache_e0_f32_.data(), teacache_e0_f32_.size());
    } else {
      tc_plan = wan_dit_evaluate_cfg_teacache(
          teacache_, cond_hint, uncond_hint, frames, height, width, per_frame,
          nullptr, 0);
    }

    ResidualRollbackGuard cond_guard{&teacache_, cond_hint.branch,
                                     tc_plan.cond.save_residual};
    ResidualRollbackGuard uncond_guard{&teacache_, uncond_hint.branch,
                                       tc_plan.uncond.save_residual};

    // 3. Patchify and embed latent ONCE, then duplicate into batch slice 1.
    patchify_and_embed_latent(latent, frames, height, width, tokens,
                              latent_count);
    Bf16* x_cond = ws_.x.as<Bf16>();
    Bf16* x_uncond = ws_.x.as<Bf16>() + model_count;
    UMM_DIT_CUDA(cudaMemcpyAsync(x_uncond, x_cond, residual_bytes,
                                 cudaMemcpyDeviceToDevice, stream));

    const auto apply_cached_residual = [&](Bf16* x_slice, int branch) {
      const DeviceBuffer& cached =
          teacache_residuals_[static_cast<size_t>(branch)];
      if (cached.bytes() != residual_bytes) {
        throw std::logic_error("Wan DiT TeaCache residual buffer size mismatch");
      }
      add_residual_bf16_kernel<<<grid_for((model_count + 1) / 2), kThreads, 0, stream>>>(
          x_slice, cached.as<Bf16>(), model_count);
      UMM_DIT_CUDA(cudaGetLastError());
    };
    const auto prepare_save_slot = [&](Bf16* x_slice, int branch,
                                       bool save) -> DeviceBuffer* {
      if (!save) return nullptr;
      DeviceBuffer* slot = &teacache_residuals_[static_cast<size_t>(branch)];
      slot->resize(residual_bytes, tracker_);
      UMM_DIT_CUDA(cudaMemcpyAsync(slot->data(), x_slice, residual_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
      return slot;
    };
    const auto finalize_save_slot = [&](DeviceBuffer* slot, const Bf16* x_slice) {
      if (slot == nullptr) return;
      sub_residual_bf16_kernel<<<grid_for((model_count + 1) / 2), kThreads, 0, stream>>>(
          slot->as<Bf16>(), x_slice, model_count);
      UMM_DIT_CUDA(cudaGetLastError());
    };

    // 4. Execute all 30 transformer layers according to the dual-branch plan.
    switch (tc_plan.mode) {
      case WanDitCfgDispatchMode::kBatchedBothMiss: {
        const CrossAttnKvEntry& pos_kv =
            get_or_build_cross_kv(pos_context, pos_len);
        const CrossAttnKvEntry& neg_kv =
            get_or_build_cross_kv(neg_context, neg_len);
        DeviceBuffer* cond_slot = prepare_save_slot(
            x_cond, cond_hint.branch, tc_plan.cond.save_residual);
        DeviceBuffer* uncond_slot = prepare_save_slot(
            x_uncond, uncond_hint.branch, tc_plan.uncond.save_residual);
        run_transformer_blocks(ws_.x.as<Bf16>(), tokens, /*batch_size=*/2,
                               first_frame_tokens, per_frame, pos_kv, pos_len,
                               &neg_kv, neg_len);
        finalize_save_slot(cond_slot, x_cond);
        finalize_save_slot(uncond_slot, x_uncond);
        break;
      }
      case WanDitCfgDispatchMode::kBothCachedHit: {
        apply_cached_residual(x_cond, cond_hint.branch);
        apply_cached_residual(x_uncond, uncond_hint.branch);
        break;
      }
      case WanDitCfgDispatchMode::kCondOnlyMiss: {
        apply_cached_residual(x_uncond, uncond_hint.branch);
        const CrossAttnKvEntry& pos_kv =
            get_or_build_cross_kv(pos_context, pos_len);
        DeviceBuffer* cond_slot = prepare_save_slot(
            x_cond, cond_hint.branch, tc_plan.cond.save_residual);
        run_transformer_blocks(x_cond, tokens, /*batch_size=*/1,
                               first_frame_tokens, per_frame, pos_kv, pos_len,
                               nullptr, 0);
        finalize_save_slot(cond_slot, x_cond);
        break;
      }
      case WanDitCfgDispatchMode::kUncondOnlyMiss: {
        apply_cached_residual(x_cond, cond_hint.branch);
        const CrossAttnKvEntry& neg_kv =
            get_or_build_cross_kv(neg_context, neg_len);
        DeviceBuffer* uncond_slot = prepare_save_slot(
            x_uncond, uncond_hint.branch, tc_plan.uncond.save_residual);
        run_transformer_blocks(x_uncond, tokens, /*batch_size=*/1,
                               first_frame_tokens, per_frame, neg_kv, neg_len,
                               nullptr, 0);
        finalize_save_slot(uncond_slot, x_uncond);
        break;
      }
    }

    // 5. Final norm + proj_out across both batch slices in one shot, then
    // fused CFG combine + unpatchify directly into FP32 NCFHW output.
    run_output_projection(tokens, /*batch_size=*/2, first_frame_tokens,
                          per_frame);
    const Bf16* cond_proj = ws_.q.as<Bf16>();
    const Bf16* uncond_proj =
        ws_.q.as<Bf16>() + static_cast<size_t>(tokens) * kProjection;
    cfg_unpatchify_bf16_to_fp32_kernel<<<grid_for(latent_count), kThreads, 0, stream>>>(
        cond_proj, uncond_proj, ws_.patch_fp32.as<float>(), frames, height,
        width, guidance_scale);
    UMM_DIT_CUDA(cudaGetLastError());

    UMM_DIT_CUDA(cudaMemcpyAsync(h_latent_pinned_.data(), ws_.patch_fp32.data(),
                                 latent_bytes, cudaMemcpyDeviceToHost, stream));
    UMM_DIT_CUDA(cudaStreamSynchronize(stream));
    cond_guard.active = false;
    uncond_guard.active = false;

    WanDitCfgOutput result;
    result.velocity.resize(latent_count);
    std::memcpy(result.velocity.data(), h_latent_pinned_.data(), latent_bytes);
    result.stats.load_ms = load_ms_;
    result.stats.forward_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    result.stats.peak_vram_bytes = tracker_.peak;
    result.stats.tensor_count = tensors_.size();
    result.cond_teacache_hit = tc_plan.cond.use_cached_residual;
    result.uncond_teacache_hit = tc_plan.uncond.use_cached_residual;
    result.stats.teacache_hit =
        result.cond_teacache_hit && result.uncond_teacache_hit;
    return result;
  }

  CudaContext context_;
  int gpu_ = 0;
  bool use_flash_attn_ = true;
  WanDitConfig config_;
  AllocationTracker tracker_;
  DeviceBuffer bf16_weights_;
  DeviceBuffer f32_weights_;
  WorkspaceBuffers ws_;
  RopeCache rope_;
  std::array<CrossAttnKvEntry, kCrossKvCacheSlots> cross_kv_cache_{};
  uint64_t cross_kv_clock_ = 0;
  TeaCacheController teacache_;
  std::array<DeviceBuffer, kWanTeaCacheBranches> teacache_residuals_{};
  PinnedHostBuffer h_time_pinned_;
  PinnedHostBuffer h_latent_pinned_;
  std::vector<uint16_t> teacache_e0_bits_;
  std::vector<float> teacache_e0_f32_;
  std::unordered_map<std::string, DeviceTensor> tensors_;
  std::array<LayerWeights, kLayers> layers_{};
  double load_ms_ = 0.0;
};

WanDit::WanDit(const std::filesystem::path& model_dir, int gpu)
    : impl_(std::make_unique<Impl>(model_dir, gpu)) {}
WanDit::~WanDit() = default;
void WanDit::set_teacache_thresh(float thresh) {
  impl_->set_teacache_thresh(thresh);
}
float WanDit::teacache_thresh() const { return impl_->teacache_thresh(); }
void WanDit::reset_teacache() { impl_->reset_teacache(); }
int WanDit::teacache_hits() const { return impl_->teacache_hits(); }
int WanDit::teacache_total_calls() const {
  return impl_->teacache_total_calls();
}
WanDitOutput WanDit::forward(const float* latent, int frames, int height,
                             int width, float timestep, const float* context,
                             int text_len, bool first_frame_zero_timestep,
                             WanDitTeaCacheHint teacache_hint) {
  return impl_->forward(latent, frames, height, width, timestep, context,
                        text_len, first_frame_zero_timestep, teacache_hint);
}
WanDitCfgOutput WanDit::forward_cfg(
    const float* latent, int frames, int height, int width, float timestep,
    const float* pos_context, int pos_len, const float* neg_context,
    int neg_len, float guidance_scale, bool first_frame_zero_timestep,
    WanDitTeaCacheHint cond_hint, WanDitTeaCacheHint uncond_hint) {
  return impl_->forward_cfg(latent, frames, height, width, timestep,
                            pos_context, pos_len, neg_context, neg_len,
                            guidance_scale, first_frame_zero_timestep,
                            cond_hint, uncond_hint);
}

}  // namespace umminfer::video

#endif  // UMM_VIDEO_CUDA
