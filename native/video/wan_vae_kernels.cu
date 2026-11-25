// SPDX-License-Identifier: Apache-2.0
// Wan2.2 AutoencoderKLWan decoder in FP32 CUDA.
//
// The decoder is a causal 3D convolutional network. Like Diffusers
// AutoencoderKLWan._decode, this file decodes one latent frame at a time and
// keeps a 2-frame causal feature cache in every kd=3 convolution. That keeps
// peak VRAM O(1) in clip length (e.g. 1280x704 77-frame fits comfortably on
// one GPU without OOM) while remaining mathematically identical to whole-clip
// causal decoding with zero spatial seams.
#include "wan_vae.hpp"

#ifdef UMM_VIDEO_CUDA

#include "safetensors_mmap.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace umminfer::video {
namespace {

void cuda_check(cudaError_t status, const char* expression) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA call failed: ") + expression +
                             ": " + cudaGetErrorString(status));
  }
}

void cublas_check(cublasStatus_t status, const char* expression) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("cuBLAS call failed: ") + expression +
                             " (status " +
                             std::to_string(static_cast<int>(status)) + ")");
  }
}

#define UMM_CUDA(expr) cuda_check((expr), #expr)
#define UMM_CUBLAS(expr) cublas_check((expr), #expr)

size_t g_device_bytes = 0;
size_t g_peak_device_bytes = 0;

size_t checked_product(std::initializer_list<int64_t> values) {
  size_t result = 1;
  for (const int64_t value : values) {
    if (value <= 0 ||
        result > std::numeric_limits<size_t>::max() / static_cast<uint64_t>(value)) {
      throw std::invalid_argument("invalid or overflowing CUDA tensor shape");
    }
    result *= static_cast<size_t>(value);
  }
  return result;
}

size_t checked_bytes(size_t elements, size_t elem_bytes) {
  if (elem_bytes != 0 &&
      elements > std::numeric_limits<size_t>::max() / elem_bytes) {
    throw std::overflow_error("CUDA allocation byte size overflow");
  }
  return elements * elem_bytes;
}

inline bool is_aligned_16(const void* ptr) noexcept {
  return (reinterpret_cast<uintptr_t>(ptr) & 15u) == 0u;
}

inline bool is_aligned_8(const void* ptr) noexcept {
  return (reinterpret_cast<uintptr_t>(ptr) & 7u) == 0u;
}

unsigned grid_for(size_t count) {
  const size_t blocks = (count + 255) / 256;
  if (blocks > static_cast<size_t>(std::numeric_limits<unsigned>::max())) {
    throw std::overflow_error("CUDA grid dimension overflow");
  }
  return static_cast<unsigned>(std::max<size_t>(blocks, 1));
}

dim3 im2col_grid(size_t positions, int64_t channels) {
  const unsigned gx = grid_for(positions);
  const unsigned gy = static_cast<unsigned>(
      std::max<int64_t>(1, std::min<int64_t>(channels, 65535)));
  return dim3(gx, gy, 1);
}

struct alignas(8) Bf16x4 {
  __nv_bfloat162 lo;
  __nv_bfloat162 hi;
};
static_assert(sizeof(Bf16x4) == 8, "Bf16x4 must be 8 bytes");

struct alignas(8) Fp16x4 {
  __half2 lo;
  __half2 hi;
};
static_assert(sizeof(Fp16x4) == 8, "Fp16x4 must be 8 bytes");

__global__ void fp32_to_bf16_vec4_kernel(const float* __restrict__ src,
                                         __nv_bfloat16* __restrict__ dst,
                                         size_t count) {
  const size_t vec_count = count / 4;
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < vec_count) {
    const float4 v = reinterpret_cast<const float4*>(src)[idx];
    Bf16x4 out;
    out.lo = __float22bfloat162_rn(make_float2(v.x, v.y));
    out.hi = __float22bfloat162_rn(make_float2(v.z, v.w));
    reinterpret_cast<Bf16x4*>(dst)[idx] = out;
  } else if (idx == vec_count) {
    for (size_t i = vec_count * 4; i < count; ++i) {
      dst[i] = __float2bfloat16(src[i]);
    }
  }
}

__global__ void fp32_to_fp16_vec4_kernel(const float* __restrict__ src,
                                         __half* __restrict__ dst,
                                         size_t count) {
  const size_t vec_count = count / 4;
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < vec_count) {
    const float4 v = reinterpret_cast<const float4*>(src)[idx];
    Fp16x4 out;
    out.lo = __float22half2_rn(make_float2(v.x, v.y));
    out.hi = __float22half2_rn(make_float2(v.z, v.w));
    reinterpret_cast<Fp16x4*>(dst)[idx] = out;
  } else if (idx == vec_count) {
    for (size_t i = vec_count * 4; i < count; ++i) {
      dst[i] = __float2half_rn(src[i]);
    }
  }
}

__global__ void fp32_to_bf16_scalar_kernel(const float* __restrict__ src,
                                           __nv_bfloat16* __restrict__ dst,
                                           size_t count) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) {
    dst[idx] = __float2bfloat16(src[idx]);
  }
}

__global__ void fp32_to_fp16_scalar_kernel(const float* __restrict__ src,
                                           __half* __restrict__ dst,
                                           size_t count) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) {
    dst[idx] = __float2half_rn(src[idx]);
  }
}

void convert_fp32_to_lowp(const float* src, void* dst, size_t count,
                          WanVaeGemmPrec prec) {
  if (count == 0) return;
  const size_t vec_count = count / 4;
  const size_t work_items = vec_count + ((count & 3u) != 0 ? 1u : 0u);
  if (prec == WanVaeGemmPrec::kBf16) {
    auto* out = static_cast<__nv_bfloat16*>(dst);
    if (is_aligned_16(src) && is_aligned_8(out)) {
      fp32_to_bf16_vec4_kernel<<<grid_for(work_items), 256>>>(src, out, count);
    } else {
      fp32_to_bf16_scalar_kernel<<<grid_for(count), 256>>>(src, out, count);
    }
  } else if (prec == WanVaeGemmPrec::kFp16) {
    auto* out = static_cast<__half*>(dst);
    if (is_aligned_16(src) && is_aligned_8(out)) {
      fp32_to_fp16_vec4_kernel<<<grid_for(work_items), 256>>>(src, out, count);
    } else {
      fp32_to_fp16_scalar_kernel<<<grid_for(count), 256>>>(src, out, count);
    }
  } else {
    throw std::logic_error("convert_fp32_to_lowp called for non-16-bit mode");
  }
  UMM_CUDA(cudaGetLastError());
}

// Context-scoped GPU scratch buffer pool. Eliminates synchronous per-layer
// cudaMalloc/cudaFree stalls while bounding idle memory with WanVaePoolPolicy
// and releasing 100% of allocations in ~BufferPool() / on OOM.
class BufferPool {
 public:
  explicit BufferPool(WanVaePoolPolicy policy = {}) : policy_(policy) {}
  ~BufferPool() { clear(); }
  BufferPool(const BufferPool&) = delete;
  BufferPool& operator=(const BufferPool&) = delete;

  struct Allocation {
    float* ptr = nullptr;
    size_t capacity_bytes = 0;
  };

  Allocation acquire(size_t requested_bytes) {
    if (requested_bytes == 0) return {};
    if (const auto idx = policy_.select_best_fit(idle_sizes_, requested_bytes);
        idx.has_value()) {
      const size_t i = *idx;
      float* ptr = idle_ptrs_[i];
      const size_t cap = idle_sizes_[i];
      idle_ptrs_.erase(idle_ptrs_.begin() + static_cast<std::ptrdiff_t>(i));
      idle_sizes_.erase(idle_sizes_.begin() + static_cast<std::ptrdiff_t>(i));
      cached_bytes_ -= cap;
      return {ptr, cap};
    }
    return {allocate_device_bytes(requested_bytes), requested_bytes};
  }

  void release(float* ptr, size_t capacity_bytes) noexcept {
    if (ptr == nullptr || capacity_bytes == 0) return;
    const auto plan =
        policy_.plan_release(idle_sizes_, cached_bytes_, capacity_bytes);
    if (!plan.cache_incoming) {
      free_device_bytes(ptr, capacity_bytes);
      return;
    }
    if (plan.evict_prefix_count > 0) {
      for (size_t i = 0; i < plan.evict_prefix_count; ++i) {
        free_device_bytes(idle_ptrs_[i], idle_sizes_[i]);
        cached_bytes_ -= idle_sizes_[i];
      }
      idle_ptrs_.erase(
          idle_ptrs_.begin(),
          idle_ptrs_.begin() +
              static_cast<std::ptrdiff_t>(plan.evict_prefix_count));
      idle_sizes_.erase(
          idle_sizes_.begin(),
          idle_sizes_.begin() +
              static_cast<std::ptrdiff_t>(plan.evict_prefix_count));
    }
    try {
      idle_ptrs_.push_back(ptr);
      idle_sizes_.push_back(capacity_bytes);
      cached_bytes_ += capacity_bytes;
    } catch (...) {
      if (idle_ptrs_.size() > idle_sizes_.size()) idle_ptrs_.pop_back();
      free_device_bytes(ptr, capacity_bytes);
    }
  }

  float* allocate_raw_with_oom_flush(size_t bytes) {
    return allocate_device_bytes(bytes);
  }

  void clear() noexcept {
    for (size_t i = 0; i < idle_ptrs_.size(); ++i) {
      free_device_bytes(idle_ptrs_[i], idle_sizes_[i]);
    }
    idle_ptrs_.clear();
    idle_sizes_.clear();
    cached_bytes_ = 0;
  }

 private:
  float* allocate_device_bytes(size_t bytes) {
    float* ptr = nullptr;
    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&ptr), bytes);
    if (status == cudaErrorMemoryAllocation && !idle_ptrs_.empty()) {
      (void)cudaGetLastError();
      clear();
      status = cudaMalloc(reinterpret_cast<void**>(&ptr), bytes);
    }
    UMM_CUDA(status);
    g_device_bytes += bytes;
    g_peak_device_bytes = std::max(g_peak_device_bytes, g_device_bytes);
    return ptr;
  }

  static void free_device_bytes(float* ptr, size_t bytes) noexcept {
    if (ptr != nullptr) {
      (void)cudaFree(ptr);
      g_device_bytes -= bytes;
    }
  }

  WanVaePoolPolicy policy_;
  std::vector<float*> idle_ptrs_;
  std::vector<size_t> idle_sizes_;
  size_t cached_bytes_ = 0;
};

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t elements) : elements_(elements) {
    if (elements == 0) return;
    if (elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
      throw std::overflow_error("device allocation size overflow");
    }
    const size_t bytes = elements * sizeof(float);
    UMM_CUDA(cudaMalloc(reinterpret_cast<void**>(&data_), bytes));
    capacity_bytes_ = bytes;
    g_device_bytes += bytes;
    g_peak_device_bytes = std::max(g_peak_device_bytes, g_device_bytes);
  }
  DeviceBuffer(BufferPool& pool, size_t elements)
      : elements_(elements), pool_(&pool) {
    if (elements == 0) {
      pool_ = nullptr;
      return;
    }
    if (elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
      throw std::overflow_error("device allocation size overflow");
    }
    const auto alloc = pool.acquire(elements * sizeof(float));
    data_ = alloc.ptr;
    capacity_bytes_ = alloc.capacity_bytes;
  }
  static DeviceBuffer adopt_raw(float* ptr, size_t elements,
                                size_t capacity_bytes) noexcept {
    DeviceBuffer buffer;
    buffer.data_ = ptr;
    buffer.elements_ = elements;
    buffer.capacity_bytes_ = capacity_bytes;
    return buffer;
  }
  ~DeviceBuffer() { release(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(other.data_),
        elements_(other.elements_),
        capacity_bytes_(other.capacity_bytes_),
        pool_(other.pool_) {
    other.data_ = nullptr;
    other.elements_ = 0;
    other.capacity_bytes_ = 0;
    other.pool_ = nullptr;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      data_ = other.data_;
      elements_ = other.elements_;
      capacity_bytes_ = other.capacity_bytes_;
      pool_ = other.pool_;
      other.data_ = nullptr;
      other.elements_ = 0;
      other.capacity_bytes_ = 0;
      other.pool_ = nullptr;
    }
    return *this;
  }
  float* data() const { return data_; }
  size_t size() const { return elements_; }

 private:
  void release() noexcept {
    if (data_ != nullptr) {
      if (pool_ != nullptr) {
        pool_->release(data_, capacity_bytes_);
      } else {
        (void)cudaFree(data_);
        g_device_bytes -= capacity_bytes_;
      }
      data_ = nullptr;
      elements_ = 0;
      capacity_bytes_ = 0;
      pool_ = nullptr;
    }
  }
  float* data_ = nullptr;
  size_t elements_ = 0;
  size_t capacity_bytes_ = 0;
  BufferPool* pool_ = nullptr;
};

class PooledRawBuffer {
 public:
  PooledRawBuffer() = default;
  PooledRawBuffer(BufferPool& pool, size_t bytes) : pool_(&pool) {
    if (bytes == 0) {
      pool_ = nullptr;
      return;
    }
    const auto alloc = pool.acquire(bytes);
    data_ = alloc.ptr;
    capacity_bytes_ = alloc.capacity_bytes;
  }
  ~PooledRawBuffer() { release(); }
  PooledRawBuffer(const PooledRawBuffer&) = delete;
  PooledRawBuffer& operator=(const PooledRawBuffer&) = delete;
  PooledRawBuffer(PooledRawBuffer&& other) noexcept
      : data_(other.data_),
        capacity_bytes_(other.capacity_bytes_),
        pool_(other.pool_) {
    other.data_ = nullptr;
    other.capacity_bytes_ = 0;
    other.pool_ = nullptr;
  }
  PooledRawBuffer& operator=(PooledRawBuffer&& other) noexcept {
    if (this != &other) {
      release();
      data_ = other.data_;
      capacity_bytes_ = other.capacity_bytes_;
      pool_ = other.pool_;
      other.data_ = nullptr;
      other.capacity_bytes_ = 0;
      other.pool_ = nullptr;
    }
    return *this;
  }
  void* data() const noexcept { return static_cast<void*>(data_); }

 private:
  void release() noexcept {
    if (data_ != nullptr) {
      if (pool_ != nullptr) {
        pool_->release(data_, capacity_bytes_);
      }
      data_ = nullptr;
      capacity_bytes_ = 0;
      pool_ = nullptr;
    }
  }
  float* data_ = nullptr;
  size_t capacity_bytes_ = 0;
  BufferPool* pool_ = nullptr;
};

// Reusable pinned host staging buffer for fast PCIe DMA D2H frame transfers.
// Falls back gracefully to pageable memory if cudaHostAlloc fails.
class PinnedHostBuffer {
 public:
  PinnedHostBuffer() = default;
  ~PinnedHostBuffer() { reset(); }
  PinnedHostBuffer(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

  float* acquire(size_t elements) {
    if (elements == 0) return nullptr;
    if (elements <= capacity_elements_) {
      return pinned_ptr_;
    }
    if (pinned_failed_) {
      return nullptr;
    }
    reset();
    const size_t bytes = checked_bytes(elements, sizeof(float));
    void* ptr = nullptr;
    const cudaError_t status = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
    if (status == cudaSuccess && ptr != nullptr) {
      pinned_ptr_ = static_cast<float*>(ptr);
      capacity_elements_ = elements;
      return pinned_ptr_;
    }
    (void)cudaGetLastError();
    pinned_failed_ = true;
    return nullptr;
  }

  bool is_pinned() const noexcept { return pinned_ptr_ != nullptr; }

  void reset() noexcept {
    if (pinned_ptr_ != nullptr) {
      (void)cudaFreeHost(pinned_ptr_);
      pinned_ptr_ = nullptr;
    }
    capacity_elements_ = 0;
  }

 private:
  float* pinned_ptr_ = nullptr;
  size_t capacity_elements_ = 0;
  bool pinned_failed_ = false;
};

DeviceBuffer upload(const std::vector<float>& values) {
  DeviceBuffer result(values.size());
  if (!values.empty()) {
    UMM_CUDA(cudaMemcpy(result.data(), values.data(), values.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  }
  return result;
}

class CudaContext {
 public:
  explicit CudaContext(int gpu) {
    UMM_CUDA(cudaSetDevice(gpu));
    UMM_CUBLAS(cublasCreate(&handle_));
    try {
      gemm_prec_ = parse_wan_vae_gemm_prec(
          std::getenv("UMM_VAE_GEMM_PREC"),
          std::getenv("UMM_VAE_PEDANTIC_FP32"));
      const cublasMath_t math_mode =
          (gemm_prec_ == WanVaeGemmPrec::kFp32) ? CUBLAS_PEDANTIC_MATH
                                                : CUBLAS_TF32_TENSOR_OP_MATH;
      UMM_CUBLAS(cublasSetMathMode(handle_, math_mode));
    } catch (...) {
      cublasDestroy(handle_);
      handle_ = nullptr;
      throw;
    }
  }
  ~CudaContext() {
    clear_lowp_weight_cache();
    columns_ = DeviceBuffer{};
    columns_bytes_ = 0;
    pinned_staging_.reset();
    pool_.clear();
    if (handle_ != nullptr) cublasDestroy(handle_);
  }
  CudaContext(const CudaContext&) = delete;
  CudaContext& operator=(const CudaContext&) = delete;

  cublasHandle_t handle() const { return handle_; }
  BufferPool& pool() { return pool_; }
  WanVaeGemmPrec gemm_prec() const noexcept { return gemm_prec_; }
  bool is_lowp_gemm() const noexcept {
    return gemm_prec_ == WanVaeGemmPrec::kBf16 ||
           gemm_prec_ == WanVaeGemmPrec::kFp16;
  }

  // Persistent high-water-mark im2col buffer reused across all conv3d /
  // conv2d_stride2 calls without repeated up-to-1 GiB cudaMalloc/cudaFree.
  void* acquire_columns_bytes(size_t bytes) {
    if (bytes == 0) return nullptr;
    if (bytes > columns_bytes_) {
      columns_ = DeviceBuffer{};
      columns_bytes_ = 0;
      if (bytes > std::numeric_limits<size_t>::max() - (sizeof(float) - 1)) {
        throw std::overflow_error("im2col columns allocation size overflow");
      }
      const size_t aligned_bytes =
          ((bytes + sizeof(float) - 1) / sizeof(float)) * sizeof(float);
      float* ptr = pool_.allocate_raw_with_oom_flush(aligned_bytes);
      columns_ = DeviceBuffer::adopt_raw(ptr, aligned_bytes / sizeof(float),
                                         aligned_bytes);
      columns_bytes_ = aligned_bytes;
    }
    return static_cast<void*>(columns_.data());
  }

  // Lazily converts and caches a conv weight matrix in BF16 or FP16 when lowp
  // GEMM precision is active. Default kTf32 / kFp32 allocates 0 extra bytes.
  const void* get_or_convert_weight(const float* weight_fp32, size_t elements) {
    if (!is_lowp_gemm()) {
      return static_cast<const void*>(weight_fp32);
    }
    const auto it = lowp_weight_cache_.find(weight_fp32);
    if (it != lowp_weight_cache_.end()) {
      return it->second.ptr;
    }
    const size_t bytes = checked_bytes(elements, sizeof(uint16_t));
    void* lowp_ptr =
        static_cast<void*>(pool_.allocate_raw_with_oom_flush(bytes));
    try {
      convert_fp32_to_lowp(weight_fp32, lowp_ptr, elements, gemm_prec_);
      lowp_weight_cache_.emplace(weight_fp32, CachedWeight{lowp_ptr, bytes});
    } catch (...) {
      (void)cudaFree(lowp_ptr);
      g_device_bytes -= bytes;
      throw;
    }
    return lowp_ptr;
  }

  float* acquire_pinned_staging(size_t elements) {
    return pinned_staging_.acquire(elements);
  }
  bool has_pinned_staging() const noexcept {
    return pinned_staging_.is_pinned();
  }

 private:
  struct CachedWeight {
    void* ptr = nullptr;
    size_t bytes = 0;
  };

  void clear_lowp_weight_cache() noexcept {
    for (auto& kv : lowp_weight_cache_) {
      if (kv.second.ptr != nullptr) {
        (void)cudaFree(kv.second.ptr);
        g_device_bytes -= kv.second.bytes;
      }
    }
    lowp_weight_cache_.clear();
  }

  cublasHandle_t handle_ = nullptr;
  WanVaeGemmPrec gemm_prec_ = WanVaeGemmPrec::kTf32;
  BufferPool pool_;
  DeviceBuffer columns_;
  size_t columns_bytes_ = 0;
  std::unordered_map<const float*, CachedWeight> lowp_weight_cache_;
  PinnedHostBuffer pinned_staging_;
};

struct Tensor {
  int64_t channels = 0;
  int64_t time = 0;
  int64_t height = 0;
  int64_t width = 0;
  DeviceBuffer storage;

  Tensor() = default;
  Tensor(CudaContext& context, int64_t c, int64_t t, int64_t h, int64_t w)
      : channels(c), time(t), height(h), width(w),
        storage(context.pool(), checked_product({c, t, h, w})) {}
  Tensor(Tensor&&) = default;
  Tensor& operator=(Tensor&&) = default;
  int64_t positions() const { return time * height * width; }
  size_t numel() const { return storage.size(); }
  float* data() const { return storage.data(); }
};

Tensor clone_tensor(CudaContext& context, const Tensor& source) {
  Tensor copy(context, source.channels, source.time, source.height,
              source.width);
  UMM_CUDA(cudaMemcpy(copy.data(), source.data(), source.numel() * sizeof(float),
                      cudaMemcpyDeviceToDevice));
  return copy;
}

struct Conv {
  int64_t input_channels = 0;
  int64_t output_channels = 0;
  int64_t kd = 1;
  int64_t kh = 1;
  int64_t kw = 1;
  DeviceBuffer weight;
  DeviceBuffer bias;
};

struct Norm {
  int64_t channels = 0;
  DeviceBuffer gamma;
};

Conv load_conv(const SafetensorsMmap& file, const std::string& prefix) {
  Conv conv;
  const auto& weight = file.tensor(prefix + ".weight");
  if (weight.shape.size() == 5) {
    conv.output_channels = weight.shape[0];
    conv.input_channels = weight.shape[1];
    conv.kd = weight.shape[2];
    conv.kh = weight.shape[3];
    conv.kw = weight.shape[4];
  } else if (weight.shape.size() == 4) {
    conv.output_channels = weight.shape[0];
    conv.input_channels = weight.shape[1];
    conv.kd = 1;
    conv.kh = weight.shape[2];
    conv.kw = weight.shape[3];
  } else {
    throw std::invalid_argument("unexpected convolution weight rank at " + prefix);
  }
  conv.weight = upload(weight.to_f32());
  conv.bias = upload(
      file.tensor(prefix + ".bias", {conv.output_channels}).to_f32());
  return conv;
}

Norm load_norm(const SafetensorsMmap& file, const std::string& prefix,
               int64_t channels) {
  Norm norm;
  norm.channels = channels;
  const auto& gamma = file.tensor(prefix + ".gamma");
  if (gamma.numel() != static_cast<size_t>(channels)) {
    throw std::invalid_argument("Wan RMSNorm gamma size mismatch at " + prefix);
  }
  norm.gamma = upload(gamma.to_f32());
  return norm;
}

__global__ void denormalize_kernel(float* x, size_t positions, int64_t channels,
                                   const float* mean, const float* std) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= positions * static_cast<size_t>(channels)) return;
  const int64_t channel = static_cast<int64_t>(index / positions);
  x[index] = x[index] * std[channel] + mean[channel];
}

template <typename OutT>
__device__ __forceinline__ OutT cast_from_f32(float v);

template <>
__device__ __forceinline__ float cast_from_f32<float>(float v) {
  return v;
}

template <>
__device__ __forceinline__ __nv_bfloat16 cast_from_f32<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

template <>
__device__ __forceinline__ __half cast_from_f32<__half>(float v) {
  return __float2half_rn(v);
}

// Specialized branch-free spatial-only 1x3x3 im2col (kd == 1, kh == 3, kw == 3)
// with fused FP32 -> float / BF16 / FP16 output conversion.
template <typename OutT>
__global__ void im2col_spatial_3x3_kernel(const float* __restrict__ input,
                                          OutT* __restrict__ columns,
                                          int64_t channels, int64_t time,
                                          int64_t height, int64_t width,
                                          int64_t position_begin,
                                          int64_t chunk_positions) {
  const int local = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                    static_cast<int>(threadIdx.x);
  if (local >= static_cast<int>(chunk_positions)) return;
  const int n = static_cast<int>(position_begin) + local;
  const int w = static_cast<int>(width);
  const int h = static_cast<int>(height);
  const int spatial = h * w;
  const int ow = n % w;
  const int rem = n / w;
  const int oh = rem % h;
  const int ot = rem / h;

  const bool has_top = (oh > 0);
  const bool has_bot = (oh + 1 < h);
  const bool has_left = (ow > 0);
  const bool has_right = (ow + 1 < w);
  const OutT zero = cast_from_f32<OutT>(0.0f);

  const int64_t time_spatial = time * static_cast<int64_t>(spatial);
  const int64_t frame_off = static_cast<int64_t>(ot) * spatial;
  const int64_t mid_row_off = static_cast<int64_t>(oh) * w;
  const int64_t top_row_off = mid_row_off - w;
  const int64_t bot_row_off = mid_row_off + w;

  for (int64_t c = static_cast<int64_t>(blockIdx.y); c < channels;
       c += static_cast<int64_t>(gridDim.y)) {
    const float* frame = input + c * time_spatial + frame_off;
    OutT* col_base = columns + (c * 9) * chunk_positions + local;

    if (has_top) {
      const float* row = frame + top_row_off;
      col_base[0 * chunk_positions] =
          has_left ? cast_from_f32<OutT>(row[ow - 1]) : zero;
      col_base[1 * chunk_positions] = cast_from_f32<OutT>(row[ow]);
      col_base[2 * chunk_positions] =
          has_right ? cast_from_f32<OutT>(row[ow + 1]) : zero;
    } else {
      col_base[0 * chunk_positions] = zero;
      col_base[1 * chunk_positions] = zero;
      col_base[2 * chunk_positions] = zero;
    }

    {
      const float* row = frame + mid_row_off;
      col_base[3 * chunk_positions] =
          has_left ? cast_from_f32<OutT>(row[ow - 1]) : zero;
      col_base[4 * chunk_positions] = cast_from_f32<OutT>(row[ow]);
      col_base[5 * chunk_positions] =
          has_right ? cast_from_f32<OutT>(row[ow + 1]) : zero;
    }

    if (has_bot) {
      const float* row = frame + bot_row_off;
      col_base[6 * chunk_positions] =
          has_left ? cast_from_f32<OutT>(row[ow - 1]) : zero;
      col_base[7 * chunk_positions] = cast_from_f32<OutT>(row[ow]);
      col_base[8 * chunk_positions] =
          has_right ? cast_from_f32<OutT>(row[ow + 1]) : zero;
    } else {
      col_base[6 * chunk_positions] = zero;
      col_base[7 * chunk_positions] = zero;
      col_base[8 * chunk_positions] = zero;
    }
  }
}

// Specialized unrolled causal 3x3x3 im2col with hoisted spatial/temporal
// bounds and fused FP32 -> float / BF16 / FP16 output conversion.
template <typename OutT>
__global__ void im2col_causal_3x3x3_kernel(
    const float* __restrict__ input, const float* __restrict__ cache,
    int64_t cache_time, OutT* __restrict__ columns, int64_t channels,
    int64_t time, int64_t height, int64_t width, int64_t position_begin,
    int64_t chunk_positions) {
  const int local = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                    static_cast<int>(threadIdx.x);
  if (local >= static_cast<int>(chunk_positions)) return;
  const int n = static_cast<int>(position_begin) + local;
  const int w = static_cast<int>(width);
  const int h = static_cast<int>(height);
  const int spatial = h * w;
  const int ow = n % w;
  const int rem = n / w;
  const int oh = rem % h;
  const int ot = rem / h;

  const bool has_top = (oh > 0);
  const bool has_bot = (oh + 1 < h);
  const bool has_left = (ow > 0);
  const bool has_right = (ow + 1 < w);
  const OutT zero = cast_from_f32<OutT>(0.0f);

  // mode: 0 = zero pad, 1 = current input frame, 2 = causal cache frame.
  int src_mode[3];
  int64_t src_off[3];
#pragma unroll
  for (int dt = 0; dt < 3; ++dt) {
    const int64_t it = static_cast<int64_t>(ot) + dt - 2;
    if (it >= 0 && it < time) {
      src_mode[dt] = 1;
      src_off[dt] = it * spatial;
    } else if (it < 0 && cache != nullptr) {
      const int64_t ct = cache_time + it;
      if (ct >= 0 && ct < cache_time) {
        src_mode[dt] = 2;
        src_off[dt] = ct * spatial;
      } else {
        src_mode[dt] = 0;
        src_off[dt] = 0;
      }
    } else {
      src_mode[dt] = 0;
      src_off[dt] = 0;
    }
  }

  const int64_t in_stride = time * static_cast<int64_t>(spatial);
  const int64_t cache_stride = cache_time * static_cast<int64_t>(spatial);
  const int64_t mid_row_off = static_cast<int64_t>(oh) * w;
  const int64_t top_row_off = mid_row_off - w;
  const int64_t bot_row_off = mid_row_off + w;

  for (int64_t c = static_cast<int64_t>(blockIdx.y); c < channels;
       c += static_cast<int64_t>(gridDim.y)) {
    const float* input_c = input + c * in_stride;
    const float* cache_c =
        cache != nullptr ? (cache + c * cache_stride) : nullptr;
    OutT* col_c = columns + (c * 27) * chunk_positions + local;

#pragma unroll
    for (int dt = 0; dt < 3; ++dt) {
      OutT* col_t = col_c + (dt * 9) * chunk_positions;
      const int mode = src_mode[dt];
      if (mode == 0) {
#pragma unroll
        for (int s = 0; s < 9; ++s) {
          col_t[s * chunk_positions] = zero;
        }
        continue;
      }
      const float* frame = ((mode == 1) ? input_c : cache_c) + src_off[dt];
      if (has_top) {
        const float* row = frame + top_row_off;
        col_t[0 * chunk_positions] =
            has_left ? cast_from_f32<OutT>(row[ow - 1]) : zero;
        col_t[1 * chunk_positions] = cast_from_f32<OutT>(row[ow]);
        col_t[2 * chunk_positions] =
            has_right ? cast_from_f32<OutT>(row[ow + 1]) : zero;
      } else {
        col_t[0 * chunk_positions] = zero;
        col_t[1 * chunk_positions] = zero;
        col_t[2 * chunk_positions] = zero;
      }
      {
        const float* row = frame + mid_row_off;
        col_t[3 * chunk_positions] =
            has_left ? cast_from_f32<OutT>(row[ow - 1]) : zero;
        col_t[4 * chunk_positions] = cast_from_f32<OutT>(row[ow]);
        col_t[5 * chunk_positions] =
            has_right ? cast_from_f32<OutT>(row[ow + 1]) : zero;
      }
      if (has_bot) {
        const float* row = frame + bot_row_off;
        col_t[6 * chunk_positions] =
            has_left ? cast_from_f32<OutT>(row[ow - 1]) : zero;
        col_t[7 * chunk_positions] = cast_from_f32<OutT>(row[ow]);
        col_t[8 * chunk_positions] =
            has_right ? cast_from_f32<OutT>(row[ow + 1]) : zero;
      } else {
        col_t[6 * chunk_positions] = zero;
        col_t[7 * chunk_positions] = zero;
        col_t[8 * chunk_positions] = zero;
      }
    }
  }
}

// Generic causal 3D im2col fallback (e.g. 3x1x1 temporal upsample conv) with
// fused FP32 -> float / BF16 / FP16 output conversion.
template <typename OutT>
__global__ void im2col_causal_kernel(const float* __restrict__ input,
                                     const float* __restrict__ cache,
                                     int64_t cache_time,
                                     OutT* __restrict__ columns,
                                     int64_t channels, int64_t time,
                                     int64_t height, int64_t width, int64_t kd,
                                     int64_t kh, int64_t kw,
                                     int64_t position_begin,
                                     int64_t chunk_positions) {
  const int64_t spatial = height * width;
  const int64_t local = static_cast<int64_t>(blockIdx.x) *
                            static_cast<int64_t>(blockDim.x) +
                        static_cast<int64_t>(threadIdx.x);
  if (local >= chunk_positions) return;
  const int64_t n = position_begin + local;
  const int64_t ow = n % width;
  const int64_t oh = (n / width) % height;
  const int64_t ot = n / spatial;
  const int64_t k_per_channel = kd * kh * kw;
  const int64_t kh_half = kh / 2;
  const int64_t kw_half = kw / 2;
  for (int64_t c = static_cast<int64_t>(blockIdx.y); c < channels;
       c += static_cast<int64_t>(gridDim.y)) {
    const float* input_c = input + c * time * spatial;
    const float* cache_c =
        cache != nullptr ? (cache + c * cache_time * spatial) : nullptr;
    int64_t k = c * k_per_channel;
    for (int64_t dt = 0; dt < kd; ++dt) {
      const int64_t it = ot + dt - (kd - 1);
      const float* frame_ptr = nullptr;
      if (it >= 0 && it < time) {
        frame_ptr = input_c + it * spatial;
      } else if (it < 0 && cache_c != nullptr) {
        const int64_t ct = cache_time + it;
        if (ct >= 0 && ct < cache_time) {
          frame_ptr = cache_c + ct * spatial;
        }
      }
      for (int64_t dy = 0; dy < kh; ++dy) {
        const int64_t ih = oh + dy - kh_half;
        const float* row_ptr = (frame_ptr != nullptr && ih >= 0 && ih < height)
                                   ? (frame_ptr + ih * width)
                                   : nullptr;
        for (int64_t dx = 0; dx < kw; ++dx, ++k) {
          const int64_t iw = ow + dx - kw_half;
          const float value = (row_ptr != nullptr && iw >= 0 && iw < width)
                                  ? row_ptr[iw]
                                  : 0.0f;
          columns[k * chunk_positions + local] = cast_from_f32<OutT>(value);
        }
      }
    }
  }
}

template <typename OutT>
void launch_im2col_causal(const float* input, const float* cache,
                          int64_t cache_time, OutT* columns, int64_t channels,
                          int64_t time, int64_t height, int64_t width,
                          int64_t kd, int64_t kh, int64_t kw,
                          int64_t position_begin, int64_t chunk_positions) {
  const dim3 grid = im2col_grid(static_cast<size_t>(chunk_positions), channels);
  if (wan_conv3d_is_spatial_3x3(kd, kh, kw)) {
    im2col_spatial_3x3_kernel<OutT><<<grid, 256>>>(
        input, columns, channels, time, height, width, position_begin,
        chunk_positions);
  } else if (wan_conv3d_is_causal_3x3x3(kd, kh, kw)) {
    im2col_causal_3x3x3_kernel<OutT><<<grid, 256>>>(
        input, cache, cache_time, columns, channels, time, height, width,
        position_begin, chunk_positions);
  } else {
    im2col_causal_kernel<OutT><<<grid, 256>>>(
        input, cache, cache_time, columns, channels, time, height, width, kd,
        kh, kw, position_begin, chunk_positions);
  }
  UMM_CUDA(cudaGetLastError());
}

__global__ void update_causal_cache_kernel(const float* old_cache,
                                           const float* input,
                                           float* next_cache, int64_t channels,
                                           int64_t cache_time, int64_t in_time,
                                           int64_t spatial) {
  const size_t total = static_cast<size_t>(channels * cache_time * spatial);
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int64_t s = index % spatial;
  const int64_t ct = (index / spatial) % cache_time;
  const int64_t c = index / (cache_time * spatial);
  const int64_t src = in_time + ct;
  if (src < cache_time) {
    next_cache[index] = old_cache[(c * cache_time + src) * spatial + s];
  } else {
    next_cache[index] = input[(c * in_time + (src - cache_time)) * spatial + s];
  }
}

struct CausalCache {
  Tensor frames;
};

__global__ void add_kernel(float* __restrict__ x,
                           const float* __restrict__ residual, size_t count) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) x[index] += residual[index];
}

__global__ void add_vec4_kernel(float* __restrict__ x,
                                const float* __restrict__ residual,
                                size_t count) {
  const size_t vec_count = count / 4;
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < vec_count) {
    float4* x4 = reinterpret_cast<float4*>(x);
    const float4* r4 = reinterpret_cast<const float4*>(residual);
    float4 v = x4[idx];
    const float4 r = r4[idx];
    v.x += r.x;
    v.y += r.y;
    v.z += r.z;
    v.w += r.w;
    x4[idx] = v;
  } else if (idx == vec_count) {
    for (size_t i = vec_count * 4; i < count; ++i) {
      x[i] += residual[i];
    }
  }
}

void add_inplace(Tensor& tensor, const Tensor& residual) {
  if (tensor.numel() != residual.numel()) {
    throw std::invalid_argument("residual tensor size mismatch");
  }
  const size_t count = tensor.numel();
  if (count == 0) return;
  if (is_aligned_16(tensor.data()) && is_aligned_16(residual.data())) {
    const size_t vec_count = count / 4;
    const size_t work_items = vec_count + ((count & 3u) != 0 ? 1u : 0u);
    add_vec4_kernel<<<grid_for(work_items), 256>>>(tensor.data(),
                                                   residual.data(), count);
  } else {
    add_kernel<<<grid_for(count), 256>>>(tensor.data(), residual.data(), count);
  }
  UMM_CUDA(cudaGetLastError());
}

// Vectorized 2D bias add with optional fused residual in exact IEEE-754 order:
//   x[c, n] = (x[c, n] + bias[c]) + residual[c, n]
template <bool HasResidual>
__global__ void bias_add_vec4_kernel(float* __restrict__ x,
                                     const float* __restrict__ bias,
                                     const float* __restrict__ residual,
                                     int64_t positions_vec4,
                                     int64_t channels) {
  const int64_t n4 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (n4 >= positions_vec4) return;
  float4* x4 = reinterpret_cast<float4*>(x);
  const float4* r4 =
      HasResidual ? reinterpret_cast<const float4*>(residual) : nullptr;
  for (int64_t c = static_cast<int64_t>(blockIdx.y); c < channels;
       c += static_cast<int64_t>(gridDim.y)) {
    const float b = bias[c];
    const int64_t off = c * positions_vec4 + n4;
    float4 v = x4[off];
    v.x += b;
    v.y += b;
    v.z += b;
    v.w += b;
    if constexpr (HasResidual) {
      const float4 r = r4[off];
      v.x += r.x;
      v.y += r.y;
      v.z += r.z;
      v.w += r.w;
    }
    x4[off] = v;
  }
}

template <bool HasResidual>
__global__ void bias_add_scalar_kernel(float* __restrict__ x,
                                       const float* __restrict__ bias,
                                       const float* __restrict__ residual,
                                       int64_t positions,
                                       int64_t channels) {
  const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (n >= positions) return;
  for (int64_t c = static_cast<int64_t>(blockIdx.y); c < channels;
       c += static_cast<int64_t>(gridDim.y)) {
    const float b = bias[c];
    const int64_t off = c * positions + n;
    float v = x[off] + b;
    if constexpr (HasResidual) {
      v += residual[off];
    }
    x[off] = v;
  }
}

void apply_conv_bias_and_residual(Tensor& output, const DeviceBuffer& bias,
                                  const Tensor* residual, bool add_bias) {
  if (residual != nullptr && residual->numel() != output.numel()) {
    throw std::invalid_argument("residual tensor size mismatch");
  }
  if (!add_bias) {
    if (residual != nullptr) {
      add_inplace(output, *residual);
    }
    return;
  }
  const int64_t positions = output.positions();
  const int64_t channels = output.channels;
  if (positions == 0 || channels == 0) return;
  const bool can_vec4 =
      ((positions & 3) == 0) && is_aligned_16(output.data()) &&
      (residual == nullptr || is_aligned_16(residual->data()));
  if (residual != nullptr) {
    if (can_vec4) {
      const int64_t pos4 = positions / 4;
      bias_add_vec4_kernel<true>
          <<<im2col_grid(static_cast<size_t>(pos4), channels), 256>>>(
              output.data(), bias.data(), residual->data(), pos4, channels);
    } else {
      bias_add_scalar_kernel<true>
          <<<im2col_grid(static_cast<size_t>(positions), channels), 256>>>(
              output.data(), bias.data(), residual->data(), positions, channels);
    }
  } else {
    if (can_vec4) {
      const int64_t pos4 = positions / 4;
      bias_add_vec4_kernel<false>
          <<<im2col_grid(static_cast<size_t>(pos4), channels), 256>>>(
              output.data(), bias.data(), nullptr, pos4, channels);
    } else {
      bias_add_scalar_kernel<false>
          <<<im2col_grid(static_cast<size_t>(positions), channels), 256>>>(
              output.data(), bias.data(), nullptr, positions, channels);
    }
  }
  UMM_CUDA(cudaGetLastError());
}

void run_conv_gemm(CudaContext& context, int m, int n, int k,
                   const void* a_ptr, int lda, const float* weight_fp32,
                   size_t weight_elements, int ldb, float* c_ptr, int ldc) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const WanVaeGemmPrec prec = context.gemm_prec();
  if (prec == WanVaeGemmPrec::kTf32 || prec == WanVaeGemmPrec::kFp32) {
    UMM_CUBLAS(cublasSgemm(context.handle(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                           &alpha, static_cast<const float*>(a_ptr), lda,
                           weight_fp32, ldb, &beta, c_ptr, ldc));
    return;
  }
  const cudaDataType_t lowp_type =
      (prec == WanVaeGemmPrec::kBf16) ? CUDA_R_16BF : CUDA_R_16F;
  const void* b_ptr =
      context.get_or_convert_weight(weight_fp32, weight_elements);
  UMM_CUBLAS(cublasGemmEx(context.handle(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                          &alpha, a_ptr, lowp_type, lda, b_ptr, lowp_type, ldb,
                          &beta, c_ptr, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F,
                          CUBLAS_GEMM_DEFAULT));
}

// out[o, n] = sum_k weight[o, k] * columns[k, n]. Position chunks cap the
// temporary im2col allocation at 1 GiB; the output keeps full NCTHW strides.
// Pointwise 1x1x1 convolutions bypass im2col completely and feed input directly
// to cuBLAS. When `cache` is provided for a kd > 1 convolution, the left
// temporal context is read from cache->frames and cache->frames is updated to
// the trailing kd-1 input frames. Optional `residual` is fused into the bias
// epilogue; `add_bias=false` defers bias addition to a downstream fused norm.
Tensor conv3d(CudaContext& context, const Tensor& input, const Conv& conv,
              CausalCache* cache = nullptr, const Tensor* residual = nullptr,
              bool add_bias = true) {
  if (input.channels != conv.input_channels) {
    throw std::invalid_argument("convolution input channel mismatch");
  }
  const int64_t positions = input.positions();
  const int64_t kernel_size = conv.input_channels * conv.kd * conv.kh * conv.kw;
  if (positions > std::numeric_limits<int>::max() ||
      kernel_size > std::numeric_limits<int>::max() ||
      conv.output_channels > std::numeric_limits<int>::max()) {
    throw std::overflow_error("convolution dimensions exceed cuBLAS limits");
  }
  Tensor output(context, conv.output_channels, input.time, input.height,
                input.width);
  if (wan_conv3d_is_pointwise(conv.kd, conv.kh, conv.kw)) {
    const void* act_ptr = static_cast<const void*>(input.data());
    PooledRawBuffer lowp_act;
    if (context.is_lowp_gemm()) {
      const size_t act_elems = input.numel();
      lowp_act = PooledRawBuffer(context.pool(),
                                 checked_bytes(act_elems, sizeof(uint16_t)));
      convert_fp32_to_lowp(input.data(), lowp_act.data(), act_elems,
                           context.gemm_prec());
      act_ptr = lowp_act.data();
    }
    run_conv_gemm(context, static_cast<int>(positions),
                  static_cast<int>(conv.output_channels),
                  static_cast<int>(kernel_size), act_ptr,
                  static_cast<int>(positions), conv.weight.data(),
                  conv.weight.size(), static_cast<int>(kernel_size),
                  output.data(), static_cast<int>(positions));
    apply_conv_bias_and_residual(output, conv.bias, residual, add_bias);
    return output;
  }

  const bool use_cache = cache != nullptr && conv.kd > 1;
  const int64_t cache_time = use_cache ? (conv.kd - 1) : 0;
  if (use_cache && cache->frames.numel() == 0) {
    cache->frames =
        Tensor(context, input.channels, cache_time, input.height, input.width);
    UMM_CUDA(cudaMemset(cache->frames.data(), 0,
                        cache->frames.numel() * sizeof(float)));
  }
  const float* cache_ptr = use_cache ? cache->frames.data() : nullptr;

  constexpr int64_t kMaxColumnBytes = 1LL << 30;  // 1 GiB cap
  const int64_t col_elem_bytes =
      context.is_lowp_gemm() ? static_cast<int64_t>(sizeof(uint16_t))
                             : static_cast<int64_t>(sizeof(float));
  const int64_t max_column_elements = kMaxColumnBytes / col_elem_bytes;
  int64_t chunk_capacity = std::max<int64_t>(
      1, std::min(positions, max_column_elements / kernel_size));
  if (context.is_lowp_gemm() && chunk_capacity < positions &&
      chunk_capacity >= 8) {
    chunk_capacity = (chunk_capacity / 8) * 8;
  }
  void* columns = context.acquire_columns_bytes(checked_bytes(
      checked_product({kernel_size, chunk_capacity}),
      static_cast<size_t>(col_elem_bytes)));
  const WanVaeGemmPrec prec = context.gemm_prec();
  for (int64_t begin = 0; begin < positions; begin += chunk_capacity) {
    const int64_t chunk = std::min(chunk_capacity, positions - begin);
    if (prec == WanVaeGemmPrec::kBf16) {
      launch_im2col_causal<__nv_bfloat16>(
          input.data(), cache_ptr, cache_time,
          static_cast<__nv_bfloat16*>(columns), input.channels, input.time,
          input.height, input.width, conv.kd, conv.kh, conv.kw, begin, chunk);
    } else if (prec == WanVaeGemmPrec::kFp16) {
      launch_im2col_causal<__half>(
          input.data(), cache_ptr, cache_time, static_cast<__half*>(columns),
          input.channels, input.time, input.height, input.width, conv.kd,
          conv.kh, conv.kw, begin, chunk);
    } else {
      launch_im2col_causal<float>(
          input.data(), cache_ptr, cache_time, static_cast<float*>(columns),
          input.channels, input.time, input.height, input.width, conv.kd,
          conv.kh, conv.kw, begin, chunk);
    }
    run_conv_gemm(context, static_cast<int>(chunk),
                  static_cast<int>(conv.output_channels),
                  static_cast<int>(kernel_size), columns,
                  static_cast<int>(chunk), conv.weight.data(),
                  conv.weight.size(), static_cast<int>(kernel_size),
                  output.data() + begin, static_cast<int>(positions));
  }
  apply_conv_bias_and_residual(output, conv.bias, residual, add_bias);
  if (use_cache) {
    if (input.time >= cache_time) {
      // Every output slot reads exclusively from `input`, so cache->frames can
      // be updated in-place without allocating a temporary cache tensor.
      update_causal_cache_kernel<<<grid_for(cache->frames.numel()), 256>>>(
          nullptr, input.data(), cache->frames.data(), input.channels,
          cache_time, input.time, input.height * input.width);
      UMM_CUDA(cudaGetLastError());
    } else {
      Tensor next_cache(context, input.channels, cache_time, input.height,
                        input.width);
      update_causal_cache_kernel<<<grid_for(next_cache.numel()), 256>>>(
          cache->frames.data(), input.data(), next_cache.data(), input.channels,
          cache_time, input.time, input.height * input.width);
      UMM_CUDA(cudaGetLastError());
      cache->frames = std::move(next_cache);
    }
  }
  return output;
}

template <bool ApplySilu, bool HasBias>
__global__ void rms_norm_vec4_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ in_bias,
                                     int64_t channels, size_t positions_vec4,
                                     bool use_shared) {
  extern __shared__ float smem[];
  float* s_gamma = smem;
  float* s_bias = HasBias ? (smem + channels) : nullptr;
  if (use_shared) {
    for (int64_t c = threadIdx.x; c < channels; c += blockDim.x) {
      s_gamma[c] = gamma[c];
      if constexpr (HasBias) s_bias[c] = in_bias[c];
    }
    __syncthreads();
  }

  const size_t n4 = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (n4 >= positions_vec4) return;

  const float4* in4 = reinterpret_cast<const float4*>(input);
  float4* out4 = reinterpret_cast<float4*>(output);

  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;
  for (int64_t c = 0; c < channels; ++c) {
    float4 v = in4[static_cast<size_t>(c) * positions_vec4 + n4];
    if constexpr (HasBias) {
      const float b = use_shared ? s_bias[c] : in_bias[c];
      v.x += b;
      v.y += b;
      v.z += b;
      v.w += b;
    }
    sum0 += v.x * v.x;
    sum1 += v.y * v.y;
    sum2 += v.z * v.z;
    sum3 += v.w * v.w;
  }

  const float sqrt_c = sqrtf(static_cast<float>(channels));
  const float scale0 = sqrt_c / fmaxf(sqrtf(sum0), 1e-12f);
  const float scale1 = sqrt_c / fmaxf(sqrtf(sum1), 1e-12f);
  const float scale2 = sqrt_c / fmaxf(sqrtf(sum2), 1e-12f);
  const float scale3 = sqrt_c / fmaxf(sqrtf(sum3), 1e-12f);

  for (int64_t c = 0; c < channels; ++c) {
    const size_t index = static_cast<size_t>(c) * positions_vec4 + n4;
    float4 v = in4[index];
    if constexpr (HasBias) {
      const float b = use_shared ? s_bias[c] : in_bias[c];
      v.x += b;
      v.y += b;
      v.z += b;
      v.w += b;
    }
    const float g = use_shared ? s_gamma[c] : gamma[c];
    float o0 = v.x * scale0 * g;
    float o1 = v.y * scale1 * g;
    float o2 = v.z * scale2 * g;
    float o3 = v.w * scale3 * g;
    if constexpr (ApplySilu) {
      o0 = o0 / (1.0f + expf(-o0));
      o1 = o1 / (1.0f + expf(-o1));
      o2 = o2 / (1.0f + expf(-o2));
      o3 = o3 / (1.0f + expf(-o3));
    }
    out4[index] = make_float4(o0, o1, o2, o3);
  }
}

template <bool ApplySilu, bool HasBias>
__global__ void rms_norm_scalar_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ in_bias,
                                       int64_t channels, size_t positions,
                                       bool use_shared) {
  extern __shared__ float smem[];
  float* s_gamma = smem;
  float* s_bias = HasBias ? (smem + channels) : nullptr;
  if (use_shared) {
    for (int64_t c = threadIdx.x; c < channels; c += blockDim.x) {
      s_gamma[c] = gamma[c];
      if constexpr (HasBias) s_bias[c] = in_bias[c];
    }
    __syncthreads();
  }

  const size_t n = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (n >= positions) return;

  float sum = 0.0f;
  for (int64_t c = 0; c < channels; ++c) {
    float value = input[static_cast<size_t>(c) * positions + n];
    if constexpr (HasBias) {
      value += use_shared ? s_bias[c] : in_bias[c];
    }
    sum += value * value;
  }
  const float denominator = fmaxf(sqrtf(sum), 1e-12f);
  const float scale = sqrtf(static_cast<float>(channels)) / denominator;
  for (int64_t c = 0; c < channels; ++c) {
    const size_t index = static_cast<size_t>(c) * positions + n;
    float value = input[index];
    if constexpr (HasBias) {
      value += use_shared ? s_bias[c] : in_bias[c];
    }
    const float g = use_shared ? s_gamma[c] : gamma[c];
    value = value * scale * g;
    if constexpr (ApplySilu) {
      value = value / (1.0f + expf(-value));
    }
    output[index] = value;
  }
}

template <bool ApplySilu, bool HasBias>
void launch_rms_norm(const Tensor& input, Tensor& output, const Norm& norm,
                     const float* in_bias) {
  const size_t positions = static_cast<size_t>(input.positions());
  const int64_t channels = input.channels;
  const bool use_shared = (channels <= 2048);
  const size_t smem_bytes =
      use_shared ? static_cast<size_t>(channels) * sizeof(float) *
                       (HasBias ? 2u : 1u)
                 : 0u;
  const bool use_vec4 = (positions >= 65536) && ((positions & 3u) == 0u) &&
                        is_aligned_16(input.data()) &&
                        is_aligned_16(output.data());
  if (use_vec4) {
    const size_t pos4 = positions / 4;
    rms_norm_vec4_kernel<ApplySilu, HasBias>
        <<<grid_for(pos4), 256, smem_bytes>>>(input.data(), output.data(),
                                              norm.gamma.data(), in_bias,
                                              channels, pos4, use_shared);
  } else {
    rms_norm_scalar_kernel<ApplySilu, HasBias>
        <<<grid_for(positions), 256, smem_bytes>>>(
            input.data(), output.data(), norm.gamma.data(), in_bias, channels,
            positions, use_shared);
  }
  UMM_CUDA(cudaGetLastError());
}

Tensor rms_norm(CudaContext& context, const Tensor& input, const Norm& norm,
                bool apply_silu = false, const float* in_bias = nullptr) {
  if (input.channels != norm.channels) {
    throw std::invalid_argument("RMSNorm channel mismatch");
  }
  Tensor output(context, input.channels, input.time, input.height, input.width);
  if (apply_silu) {
    if (in_bias != nullptr) {
      launch_rms_norm<true, true>(input, output, norm, in_bias);
    } else {
      launch_rms_norm<true, false>(input, output, norm, nullptr);
    }
  } else {
    if (in_bias != nullptr) {
      launch_rms_norm<false, true>(input, output, norm, in_bias);
    } else {
      launch_rms_norm<false, false>(input, output, norm, nullptr);
    }
  }
  return output;
}

__global__ void nearest_spatial_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int64_t channels, int64_t time,
                                       int64_t height, int64_t width) {
  const int64_t out_height = height * 2;
  const int64_t out_width = width * 2;
  const size_t count =
      static_cast<size_t>(channels * time * out_height * out_width);
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const int64_t ow = index % out_width;
  const int64_t oh = (index / out_width) % out_height;
  const int64_t t = (index / (out_height * out_width)) % time;
  const int64_t c = index / (time * out_height * out_width);
  output[index] = input[((c * time + t) * height + oh / 2) * width + ow / 2];
}

// Vectorized 2x spatial nearest upsample: reads 2 adjacent input pixels as
// float2 and writes two coalesced 128-bit float4 rows to output.
__global__ void nearest_spatial_2x_vec_kernel(const float* __restrict__ input,
                                              float* __restrict__ output,
                                              int64_t ct_total, int64_t height,
                                              int64_t width_pairs) {
  const size_t count = static_cast<size_t>(ct_total * height * width_pairs);
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  const int64_t pw = static_cast<int64_t>(idx % static_cast<size_t>(width_pairs));
  const int64_t rem = static_cast<int64_t>(idx / static_cast<size_t>(width_pairs));
  const int64_t h = rem % height;
  const int64_t ct = rem / height;
  const float2 pair = reinterpret_cast<const float2*>(input)[idx];
  const float4 dup = make_float4(pair.x, pair.x, pair.y, pair.y);
  float4* out4 = reinterpret_cast<float4*>(output);
  const int64_t out_row0 = (ct * (height * 2) + h * 2) * width_pairs + pw;
  out4[out_row0] = dup;
  out4[out_row0 + width_pairs] = dup;
}

Tensor nearest_spatial(CudaContext& context, const Tensor& input) {
  Tensor output(context, input.channels, input.time, input.height * 2,
                input.width * 2);
  if (((input.width & 1) == 0) && is_aligned_8(input.data()) &&
      is_aligned_16(output.data())) {
    const int64_t ct_total = input.channels * input.time;
    const int64_t width_pairs = input.width / 2;
    const size_t work =
        static_cast<size_t>(ct_total * input.height * width_pairs);
    nearest_spatial_2x_vec_kernel<<<grid_for(work), 256>>>(
        input.data(), output.data(), ct_total, input.height, width_pairs);
  } else {
    nearest_spatial_kernel<<<grid_for(output.numel()), 256>>>(
        input.data(), output.data(), input.channels, input.time, input.height,
        input.width);
  }
  UMM_CUDA(cudaGetLastError());
  return output;
}

// Temporal upsample for non-first streaming chunks: each input step j expands
// into two output frames from the two channel halves of the cached causal
// time_conv output. For the first chunk (latent step 0), Diffusers emits frame
// 0 unchanged without running or updating time_conv.
__global__ void temporal_interleave_chunk_kernel(const float* __restrict__ convolved,
                                                 float* __restrict__ output,
                                                 int64_t channels, int64_t time,
                                                 int64_t spatial) {
  const int64_t out_time = time * 2;
  const size_t count = static_cast<size_t>(channels * out_time * spatial);
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const int64_t n = index % spatial;
  const int64_t ot = (index / spatial) % out_time;
  const int64_t c = index / (out_time * spatial);
  const int64_t j = ot / 2;
  const int64_t source_channel = (ot % 2 == 0) ? c : channels + c;
  output[index] = convolved[(source_channel * time + j) * spatial + n];
}

__global__ void temporal_interleave_vec4_kernel(const float* __restrict__ convolved,
                                                float* __restrict__ output,
                                                int64_t channels, int64_t time,
                                                int64_t spatial_vec4) {
  const int64_t out_time = time * 2;
  const size_t count = static_cast<size_t>(channels * out_time * spatial_vec4);
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  const int64_t n4 = static_cast<int64_t>(idx % static_cast<size_t>(spatial_vec4));
  const int64_t rem = static_cast<int64_t>(idx / static_cast<size_t>(spatial_vec4));
  const int64_t ot = rem % out_time;
  const int64_t c = rem / out_time;
  const int64_t j = ot / 2;
  const int64_t source_channel = ((ot & 1) == 0) ? c : (channels + c);
  const float4* in4 = reinterpret_cast<const float4*>(convolved);
  float4* out4 = reinterpret_cast<float4*>(output);
  out4[idx] = in4[(source_channel * time + j) * spatial_vec4 + n4];
}

Tensor temporal_upsample(CudaContext& context, const Tensor& input,
                         const Conv& time_conv, CausalCache& time_cache,
                         bool first_chunk) {
  if (time_conv.kd != 3 || time_conv.kh != 1 || time_conv.kw != 1 ||
      time_conv.input_channels != input.channels ||
      time_conv.output_channels != input.channels * 2) {
    throw std::invalid_argument("time convolution must be C -> 2C with a 3x1x1 kernel");
  }
  if (first_chunk) {
    return clone_tensor(context, input);
  }
  Tensor convolved = conv3d(context, input, time_conv, &time_cache);
  Tensor expanded(context, input.channels, input.time * 2, input.height,
                  input.width);
  const int64_t spatial = input.height * input.width;
  if (((spatial & 3) == 0) && is_aligned_16(convolved.data()) &&
      is_aligned_16(expanded.data())) {
    const int64_t spatial_vec4 = spatial / 4;
    const size_t vec_count =
        static_cast<size_t>(input.channels * (input.time * 2) * spatial_vec4);
    temporal_interleave_vec4_kernel<<<grid_for(vec_count), 256>>>(
        convolved.data(), expanded.data(), input.channels, input.time,
        spatial_vec4);
  } else {
    temporal_interleave_chunk_kernel<<<grid_for(expanded.numel()), 256>>>(
        convolved.data(), expanded.data(), input.channels, input.time, spatial);
  }
  UMM_CUDA(cudaGetLastError());
  return expanded;
}

// Fused DupUp3D shortcut + in-place residual add: eliminates allocating,
// writing, and re-reading the temporary `shortcut` tensor.
__global__ void dup_up_add_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int64_t out_channels, int64_t in_time,
                                  int64_t out_time, int64_t height,
                                  int64_t width, int64_t factor_t,
                                  int64_t factor_s, int64_t repeats,
                                  int64_t drop_first) {
  const int64_t out_height = height * factor_s;
  const int64_t out_width = width * factor_s;
  const size_t count =
      static_cast<size_t>(out_channels * out_time * out_height * out_width);
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const int64_t ow = static_cast<int64_t>(index % static_cast<size_t>(out_width));
  const int64_t rem1 = static_cast<int64_t>(index / static_cast<size_t>(out_width));
  const int64_t oh = rem1 % out_height;
  const int64_t rem2 = rem1 / out_height;
  const int64_t ot = rem2 % out_time;
  const int64_t oc = rem2 / out_time;
  const int64_t shifted = ot + drop_first;
  const int64_t ft = shifted % factor_t;
  const int64_t st = shifted / factor_t;
  const int64_t fs1 = oh % factor_s;
  const int64_t h = oh / factor_s;
  const int64_t fs2 = ow % factor_s;
  const int64_t w = ow / factor_s;
  const int64_t p =
      ((oc * factor_t + ft) * factor_s + fs1) * factor_s + fs2;
  const int64_t source_channel = p / repeats;
  output[index] +=
      input[((source_channel * in_time + st) * height + h) * width + w];
}

void dup_up3d_add_inplace(Tensor& output, const Tensor& input,
                          int64_t out_channels, int64_t factor_t,
                          int64_t factor_s, bool first_chunk) {
  const int64_t factor = factor_t * factor_s * factor_s;
  const int64_t numerator = out_channels * factor;
  if (numerator % input.channels != 0) {
    throw std::invalid_argument("DupUp3D channel ratio is not integral");
  }
  const int64_t repeats = numerator / input.channels;
  const int64_t drop_first = first_chunk ? (factor_t - 1) : 0;
  const int64_t out_time = factor_t * input.time - drop_first;
  if (output.channels != out_channels || output.time != out_time ||
      output.height != input.height * factor_s ||
      output.width != input.width * factor_s) {
    throw std::invalid_argument("DupUp3D shortcut shape mismatch");
  }
  dup_up_add_kernel<<<grid_for(output.numel()), 256>>>(
      input.data(), output.data(), out_channels, input.time, out_time,
      input.height, input.width, factor_t, factor_s, repeats, drop_first);
  UMM_CUDA(cudaGetLastError());
}

__global__ void softmax_key_kernel(float* scores, int64_t spatial) {
  extern __shared__ float shared[];
  const int64_t row = blockIdx.x;
  const int64_t frame = row / spatial;
  const int64_t query = row % spatial;
  float* base = scores + frame * spatial * spatial + query;
  float local_max = -INFINITY;
  for (int64_t key = threadIdx.x; key < spatial; key += blockDim.x) {
    local_max = fmaxf(local_max, base[key * spatial]);
  }
  shared[threadIdx.x] = local_max;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float maximum = shared[0];
  float local_sum = 0.0f;
  for (int64_t key = threadIdx.x; key < spatial; key += blockDim.x) {
    const float value = expf(base[key * spatial] - maximum);
    base[key * spatial] = value;
    local_sum += value;
  }
  __syncthreads();
  shared[threadIdx.x] = local_sum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  const float inverse = 1.0f / shared[0];
  for (int64_t key = threadIdx.x; key < spatial; key += blockDim.x) {
    base[key * spatial] *= inverse;
  }
}

// Per-frame single-head spatial self-attention. Q, K and V are stored column
// major as [positions, channels] per frame with leading dimension positions.
Tensor attention(CudaContext& context, const Tensor& input, const Norm& norm,
                 const Conv& qkv, const Conv& projection) {
  Tensor normalized = rms_norm(context, input, norm, /*apply_silu=*/false);
  Tensor packed = conv3d(context, normalized, qkv);
  normalized = Tensor{};
  const int64_t channels = input.channels;
  const int64_t spatial = input.height * input.width;
  const int64_t positions = input.positions();
  const int64_t frames = input.time;
  if (packed.channels != channels * 3) {
    throw std::invalid_argument("attention QKV projection shape mismatch");
  }
  const int64_t stride = spatial * spatial;
  if (spatial > std::numeric_limits<int>::max() ||
      channels > std::numeric_limits<int>::max() ||
      positions > std::numeric_limits<int>::max() ||
      stride > std::numeric_limits<int>::max()) {
    throw std::overflow_error("attention dimensions exceed cuBLAS limits");
  }
  const float* query = packed.data();
  const float* key = query + channels * positions;
  const float* value = key + channels * positions;
  DeviceBuffer scores(context.pool(), checked_product({frames, spatial, spatial}));
  const float scale = 1.0f / sqrtf(static_cast<float>(channels));
  const float alpha = 1.0f;
  const float beta = 0.0f;
  // scores[f] = query[f] * key[f]^T
  UMM_CUBLAS(cublasSgemmStridedBatched(
      context.handle(), CUBLAS_OP_N, CUBLAS_OP_T, static_cast<int>(spatial),
      static_cast<int>(spatial), static_cast<int>(channels), &scale, query,
      static_cast<int>(positions), static_cast<int>(spatial), key,
      static_cast<int>(positions), static_cast<int>(spatial), &beta, scores.data(),
      static_cast<int>(spatial), static_cast<int>(stride),
      static_cast<int>(frames)));
  softmax_key_kernel<<<static_cast<unsigned>(frames * spatial), 256,
                       256 * sizeof(float)>>>(scores.data(), spatial);
  UMM_CUDA(cudaGetLastError());

  Tensor attended(context, channels, input.time, input.height, input.width);
  UMM_CUBLAS(cublasSgemmStridedBatched(
      context.handle(), CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(spatial),
      static_cast<int>(channels), static_cast<int>(spatial), &alpha, scores.data(),
      static_cast<int>(spatial), static_cast<int>(stride), value,
      static_cast<int>(positions), static_cast<int>(spatial), &beta, attended.data(),
      static_cast<int>(positions), static_cast<int>(spatial),
      static_cast<int>(frames)));
  scores = DeviceBuffer{};
  packed = Tensor{};
  Tensor output = conv3d(context, attended, projection, /*cache=*/nullptr,
                         /*residual=*/&input);
  attended = Tensor{};
  return output;
}

struct ResBlock {
  Norm norm1;
  Conv conv1;
  Norm norm2;
  Conv conv2;
  bool has_shortcut = false;
  Conv shortcut;
};

struct ResBlockCache {
  CausalCache conv1;
  CausalCache conv2;
};

ResBlock load_res_block(const SafetensorsMmap& file, const std::string& prefix) {
  ResBlock block;
  Conv conv1 = load_conv(file, prefix + ".conv1");
  const int64_t in_channels = conv1.input_channels;
  const int64_t out_channels = conv1.output_channels;
  block.norm1 = load_norm(file, prefix + ".norm1", in_channels);
  block.conv1 = std::move(conv1);
  block.conv2 = load_conv(file, prefix + ".conv2");
  if (block.conv2.output_channels != out_channels) {
    throw std::invalid_argument("residual block conv2 channel mismatch at " + prefix);
  }
  block.norm2 = load_norm(file, prefix + ".norm2", out_channels);
  if (file.contains(prefix + ".conv_shortcut.weight")) {
    block.has_shortcut = true;
    block.shortcut = load_conv(file, prefix + ".conv_shortcut");
    if (block.shortcut.kd != 1) {
      throw std::invalid_argument("residual shortcut must have kd=1 at " + prefix);
    }
  } else if (in_channels != out_channels) {
    throw std::invalid_argument("residual block needs a shortcut at " + prefix);
  }
  return block;
}

Tensor res_block(CudaContext& context, const Tensor& input, const ResBlock& block,
                 ResBlockCache& cache) {
  Tensor output = rms_norm(context, input, block.norm1, /*apply_silu=*/true);
  output = conv3d(context, output, block.conv1, &cache.conv1,
                  /*residual=*/nullptr, /*add_bias=*/false);
  output = rms_norm(context, output, block.norm2, /*apply_silu=*/true,
                    block.conv1.bias.data());
  if (block.has_shortcut) {
    output = conv3d(context, output, block.conv2, &cache.conv2);
    return conv3d(context, input, block.shortcut, /*cache=*/nullptr,
                  /*residual=*/&output);
  }
  return conv3d(context, output, block.conv2, &cache.conv2,
                /*residual=*/&input);
}

struct UpBlock {
  std::vector<ResBlock> resnets;
  bool has_upsample = false;
  bool temporal = false;
  int64_t factor_t = 1;
  int64_t input_channels = 0;
  int64_t output_channels = 0;
  Conv resample_conv;
  Conv time_conv;
};

struct UpBlockCache {
  std::vector<ResBlockCache> resnets;
  CausalCache time_conv;
  CausalCache resample_conv;
};

struct Decoder {
  Conv conv_in;
  ResBlock mid_res0;
  ResBlock mid_res1;
  Norm mid_norm;
  Conv mid_qkv;
  Conv mid_proj;
  std::vector<UpBlock> up_blocks;
  Norm norm_out;
  Conv conv_out;
};

struct DecoderCache {
  CausalCache conv_in;
  ResBlockCache mid_res0;
  ResBlockCache mid_res1;
  std::vector<UpBlockCache> up_blocks;
  CausalCache conv_out;

  explicit DecoderCache(const Decoder& decoder) {
    up_blocks.resize(decoder.up_blocks.size());
    for (size_t i = 0; i < decoder.up_blocks.size(); ++i) {
      up_blocks[i].resnets.resize(decoder.up_blocks[i].resnets.size());
    }
  }
};

Decoder load_decoder(const SafetensorsMmap& file, const WanVaeConfig& config) {
  Decoder decoder;
  decoder.conv_in = load_conv(file, "decoder.conv_in");
  decoder.mid_res0 = load_res_block(file, "decoder.mid_block.resnets.0");
  decoder.mid_res1 = load_res_block(file, "decoder.mid_block.resnets.1");
  const int64_t mid_channels = decoder.mid_res0.conv1.output_channels;
  decoder.mid_norm = load_norm(file, "decoder.mid_block.attentions.0.norm",
                               mid_channels);
  decoder.mid_qkv = load_conv(file, "decoder.mid_block.attentions.0.to_qkv");
  decoder.mid_proj = load_conv(file, "decoder.mid_block.attentions.0.proj");

  if (decoder.conv_in.output_channels != mid_channels) {
    throw std::invalid_argument("decoder conv_in must feed the mid block");
  }

  std::vector<int64_t> dims;
  dims.push_back(config.decoder_base_dim * config.dim_mult.back());
  for (auto it = config.dim_mult.rbegin(); it != config.dim_mult.rend(); ++it) {
    dims.push_back(config.decoder_base_dim * *it);
  }

  const int64_t up_blocks = static_cast<int64_t>(config.dim_mult.size());
  for (int64_t i = 0; i < up_blocks; ++i) {
    UpBlock block;
    block.input_channels = dims[static_cast<size_t>(i)];
    block.output_channels = dims[static_cast<size_t>(i + 1)];
    for (int64_t j = 0; j <= config.num_res_blocks; ++j) {
      block.resnets.push_back(load_res_block(
          file, "decoder.up_blocks." + std::to_string(i) + ".resnets." +
                    std::to_string(j)));
    }
    const bool up_flag = i != up_blocks - 1;
    if (up_flag) {
      const int64_t reversed_index =
          static_cast<int64_t>(config.temperal_downsample.size()) - 1 - i;
      block.temporal = config.temperal_downsample[static_cast<size_t>(reversed_index)];
      block.factor_t = block.temporal ? 2 : 1;
      const std::string prefix =
          "decoder.up_blocks." + std::to_string(i) + ".upsampler";
      block.resample_conv = load_conv(file, prefix + ".resample.1");
      if (block.temporal) {
        block.time_conv = load_conv(file, prefix + ".time_conv");
      }
      block.has_upsample = true;
    }
    decoder.up_blocks.push_back(std::move(block));
  }

  const int64_t final_channels = dims.back();
  decoder.norm_out = load_norm(file, "decoder.norm_out", final_channels);
  decoder.conv_out = load_conv(file, "decoder.conv_out");
  if (decoder.conv_out.input_channels != final_channels) {
    throw std::invalid_argument("decoder conv_out channel mismatch");
  }
  return decoder;
}

__global__ void unpatchify_clamp_kernel(const float* input, float* output,
                                        int64_t output_channels, int64_t time,
                                        int64_t height, int64_t width,
                                        int64_t patch) {
  const int64_t output_height = height * patch;
  const int64_t output_width = width * patch;
  const size_t count = static_cast<size_t>(output_channels * time * output_height *
                                           output_width);
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const int64_t x = index % output_width;
  const int64_t y = (index / output_width) % output_height;
  const int64_t t = (index / (output_height * output_width)) % time;
  const int64_t c = index / (time * output_height * output_width);
  const int64_t r = y % patch;
  const int64_t s = x % patch;
  // Diffusers view/permute order: patches split the channel axis as
  // (channel, x-patch, y-patch), so the x patch index is the outer one.
  const int64_t input_channel = (c * patch + s) * patch + r;
  const float value =
      input[((input_channel * time + t) * height + y / patch) * width + x / patch];
  output[index] = value < -1.0f ? -1.0f : (value > 1.0f ? 1.0f : value);
}

__device__ __forceinline__ float clamp_unit_f32(float v) {
  return v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
}

__global__ void unpatchify_clamp_p2_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int64_t output_channels, int64_t time,
                                           int64_t height, int64_t width) {
  const size_t total =
      static_cast<size_t>(output_channels * time * height * width);
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int64_t x = static_cast<int64_t>(idx % static_cast<size_t>(width));
  const int64_t rem1 = static_cast<int64_t>(idx / static_cast<size_t>(width));
  const int64_t y = rem1 % height;
  const int64_t rem2 = rem1 / height;
  const int64_t t = rem2 % time;
  const int64_t c = rem2 / time;

  const size_t plane = static_cast<size_t>(time * height * width);
  const size_t in_off = static_cast<size_t>((t * height + y) * width + x);
  const size_t c_base = static_cast<size_t>(c) * 4;
  const float v00 = clamp_unit_f32(input[(c_base + 0) * plane + in_off]);
  const float v01 = clamp_unit_f32(input[(c_base + 1) * plane + in_off]);
  const float v10 = clamp_unit_f32(input[(c_base + 2) * plane + in_off]);
  const float v11 = clamp_unit_f32(input[(c_base + 3) * plane + in_off]);

  float2* out2 = reinterpret_cast<float2*>(output);
  const size_t out_row0 =
      static_cast<size_t>(((c * time + t) * (height * 2) + y * 2) * width + x);
  out2[out_row0] = make_float2(v00, v10);
  out2[out_row0 + static_cast<size_t>(width)] = make_float2(v01, v11);
}

Tensor unpatchify_clamp(CudaContext& context, const Tensor& input,
                        int64_t output_channels, int64_t patch) {
  if (input.channels != output_channels * patch * patch) {
    throw std::invalid_argument("unpatchify channel mismatch");
  }
  Tensor output(context, output_channels, input.time, input.height * patch,
                input.width * patch);
  if (patch == 2 && is_aligned_8(output.data())) {
    const size_t work = static_cast<size_t>(
        output_channels * input.time * input.height * input.width);
    unpatchify_clamp_p2_kernel<<<grid_for(work), 256>>>(
        input.data(), output.data(), output_channels, input.time, input.height,
        input.width);
  } else {
    unpatchify_clamp_kernel<<<grid_for(output.numel()), 256>>>(
        input.data(), output.data(), output_channels, input.time, input.height,
        input.width, patch);
  }
  UMM_CUDA(cudaGetLastError());
  return output;
}

Tensor upsample_block(CudaContext& context, const Tensor& input,
                      const UpBlock& block, UpBlockCache& cache,
                      bool first_chunk) {
  if (input.channels != block.input_channels || block.resnets.empty()) {
    throw std::invalid_argument("up block input channel mismatch");
  }
  Tensor current = res_block(context, input, block.resnets.front(), cache.resnets[0]);
  for (size_t i = 1; i < block.resnets.size(); ++i) {
    current = res_block(context, current, block.resnets[i], cache.resnets[i]);
  }
  if (block.has_upsample) {
    if (block.temporal) {
      Tensor expanded = temporal_upsample(context, current, block.time_conv,
                                          cache.time_conv, first_chunk);
      current = Tensor{};
      Tensor spatial = nearest_spatial(context, expanded);
      expanded = Tensor{};
      current = conv3d(context, spatial, block.resample_conv, &cache.resample_conv);
    } else {
      Tensor spatial = nearest_spatial(context, current);
      current = Tensor{};
      current = conv3d(context, spatial, block.resample_conv, &cache.resample_conv);
    }
    // Fused DupUp3D shortcut + in-place add reads block input without
    // allocating a temporary shortcut tensor.
    dup_up3d_add_inplace(current, input, block.output_channels, block.factor_t,
                         2, first_chunk);
  }
  return current;
}

}  // namespace

WanVaeOutput decode_wan_vae_fp32(const std::filesystem::path& vae_dir,
                                 const std::vector<float>& latent,
                                 const std::vector<int64_t>& latent_shape,
                                 int gpu) {
  if (latent_shape.size() != 5 || latent_shape[0] != 1) {
    throw std::invalid_argument("latent shape must be [1,C,T,H,W]");
  }
  const WanVaeConfig config = load_wan_vae_config(vae_dir);
  if (!config.is_residual) {
    throw std::invalid_argument("only the Wan2.2 residual decoder is supported");
  }
  if (latent_shape[1] != config.z_dim) {
    throw std::invalid_argument("latent channel count does not match z_dim");
  }
  const size_t expected = checked_product({latent_shape[0], latent_shape[1],
                                           latent_shape[2], latent_shape[3],
                                           latent_shape[4]});
  if (latent.size() != expected) {
    throw std::invalid_argument("latent buffer size does not match its shape");
  }

  g_peak_device_bytes = 0;
  CudaContext context(gpu);
  const auto safetensors = vae_dir / "diffusion_pytorch_model.safetensors";
  SafetensorsMmap file(safetensors);
  Decoder decoder = load_decoder(file, config);
  if (file.tensor("post_quant_conv.weight").numel() !=
      static_cast<size_t>(config.z_dim * config.z_dim)) {
    throw std::invalid_argument("post_quant_conv weight does not match z_dim");
  }
  Conv post_quant = load_conv(file, "post_quant_conv");
  DeviceBuffer mean = upload(config.latents_mean);
  DeviceBuffer stddev = upload(config.latents_std);

  const int64_t latent_c = latent_shape[1];
  const int64_t latent_t = latent_shape[2];
  const int64_t latent_h = latent_shape[3];
  const int64_t latent_w = latent_shape[4];
  const int64_t latent_spatial = latent_h * latent_w;

  const int64_t patch_area = static_cast<int64_t>(config.patch_size) * config.patch_size;
  const int64_t output_channels = config.out_channels / patch_area;
  const int64_t temporal_factor = wan_vae_temporal_factor(config);
  const int64_t total_out_t = wan_vae_decoded_frame_count(latent_t, temporal_factor);
  int64_t spatial_factor = config.patch_size;
  for (const UpBlock& block : decoder.up_blocks) {
    if (block.has_upsample) spatial_factor *= 2;
  }
  const int64_t out_h = latent_h * spatial_factor;
  const int64_t out_w = latent_w * spatial_factor;
  const int64_t out_spatial = out_h * out_w;

  WanVaeOutput output;
  output.shape = {1, output_channels, total_out_t, out_h, out_w};
  output.data.resize(checked_product({output_channels, total_out_t, out_h, out_w}));

  DecoderCache cache(decoder);
  std::vector<float> frame_host(checked_product({latent_c, latent_h, latent_w}));
  const int64_t max_chunk_t = latent_t > 1 ? temporal_factor : 1;
  (void)context.acquire_pinned_staging(
      checked_product({output_channels, max_chunk_t, out_h, out_w}));

  // Exact causal streaming decode: one latent frame per step with persistent
  // 2-frame causal caches, matching Diffusers AutoencoderKLWan._decode while
  // bounding peak VRAM independently of clip length.
  const auto begin = std::chrono::steady_clock::now();
  for (int64_t t = 0; t < latent_t; ++t) {
    for (int64_t c = 0; c < latent_c; ++c) {
      const float* src =
          latent.data() + static_cast<size_t>((c * latent_t + t) * latent_spatial);
      float* dst = frame_host.data() + static_cast<size_t>(c * latent_spatial);
      std::copy(src, src + latent_spatial, dst);
    }

    Tensor input(context, latent_c, 1, latent_h, latent_w);
    UMM_CUDA(cudaMemcpy(input.data(), frame_host.data(),
                        frame_host.size() * sizeof(float), cudaMemcpyHostToDevice));
    denormalize_kernel<<<grid_for(input.numel()), 256>>>(
        input.data(), static_cast<size_t>(input.positions()), input.channels,
        mean.data(), stddev.data());
    UMM_CUDA(cudaGetLastError());

    const bool first_chunk = (t == 0);
    Tensor current = conv3d(context, input, post_quant);
    input = Tensor{};
    current = conv3d(context, current, decoder.conv_in, &cache.conv_in);
    current = res_block(context, current, decoder.mid_res0, cache.mid_res0);
    current = attention(context, current, decoder.mid_norm, decoder.mid_qkv,
                        decoder.mid_proj);
    current = res_block(context, current, decoder.mid_res1, cache.mid_res1);
    for (size_t i = 0; i < decoder.up_blocks.size(); ++i) {
      current = upsample_block(context, current, decoder.up_blocks[i],
                               cache.up_blocks[i], first_chunk);
    }
    current = rms_norm(context, current, decoder.norm_out, /*apply_silu=*/true);
    current = conv3d(context, current, decoder.conv_out, &cache.conv_out);
    current = unpatchify_clamp(context, current, output_channels,
                               config.patch_size);

    const int64_t expected_chunk_t = wan_vae_chunk_frame_count(t, temporal_factor);
    const int64_t out_t_offset = wan_vae_chunk_frame_offset(t, temporal_factor);
    if (current.channels != output_channels || current.time != expected_chunk_t ||
        current.height != out_h || current.width != out_w) {
      throw std::runtime_error("unexpected streamed Wan2.2 VAE chunk shape");
    }
    const size_t chunk_plane = static_cast<size_t>(current.time * out_spatial);
    const size_t chunk_elements = current.numel();
    float* staging = context.acquire_pinned_staging(chunk_elements);
    if (context.has_pinned_staging() && staging != nullptr) {
      UMM_CUDA(cudaMemcpy(staging, current.data(),
                          chunk_elements * sizeof(float),
                          cudaMemcpyDeviceToHost));
      for (int64_t c = 0; c < output_channels; ++c) {
        const float* src_plane =
            staging + static_cast<size_t>(c) * chunk_plane;
        float* host_plane =
            output.data.data() +
            static_cast<size_t>((c * total_out_t + out_t_offset) * out_spatial);
        std::memcpy(host_plane, src_plane, chunk_plane * sizeof(float));
      }
    } else {
      for (int64_t c = 0; c < output_channels; ++c) {
        const float* dev_plane =
            current.data() + static_cast<size_t>(c) * chunk_plane;
        float* host_plane =
            output.data.data() +
            static_cast<size_t>((c * total_out_t + out_t_offset) * out_spatial);
        UMM_CUDA(cudaMemcpy(host_plane, dev_plane, chunk_plane * sizeof(float),
                            cudaMemcpyDeviceToHost));
      }
    }
  }

  UMM_CUDA(cudaDeviceSynchronize());
  const auto end = std::chrono::steady_clock::now();

  output.decode_ms =
      std::chrono::duration<double, std::milli>(end - begin).count();
  output.peak_native_device_bytes = g_peak_device_bytes;
  return output;
}

// ---------------------------------------------------------------------------
// Single-frame encoder (Diffusers AutoencoderKLWan.encode at T=1).
//
// At T=1 every causal feature cache is still empty, so a cache-free causal
// convolution is bit-identical to the streaming path: left temporal padding of
// kd-1 zeros and symmetric spatial padding. WanResample.downsample3d only runs
// its 2D half because the temporal convolution is skipped for the first chunk.
// ---------------------------------------------------------------------------
namespace {

// ResBlock without a causal cache. Equivalent to res_block() at T=1.
Tensor res_block_plain(CudaContext& context, const Tensor& input,
                       const ResBlock& block) {
  Tensor output = rms_norm(context, input, block.norm1, /*apply_silu=*/true);
  output = conv3d(context, output, block.conv1, /*cache=*/nullptr,
                  /*residual=*/nullptr, /*add_bias=*/false);
  output = rms_norm(context, output, block.norm2, /*apply_silu=*/true,
                    block.conv1.bias.data());
  if (block.has_shortcut) {
    output = conv3d(context, output, block.conv2);
    return conv3d(context, input, block.shortcut, /*cache=*/nullptr,
                  /*residual=*/&output);
  }
  return conv3d(context, output, block.conv2, /*cache=*/nullptr,
                /*residual=*/&input);
}

// im2col for a 2D stride-2 convolution whose input was ZeroPad2d((0,1,0,1))'d,
// which is exactly WanResample's "downsample2d" pre-processing, with fused
// FP32 -> float / BF16 / FP16 output conversion.
template <typename OutT>
__global__ void im2col_stride2_kernel(const float* __restrict__ input,
                                      OutT* __restrict__ columns,
                                      int64_t channels, int64_t height,
                                      int64_t width, int64_t kh, int64_t kw,
                                      int64_t out_height, int64_t out_width) {
  const int64_t positions = out_height * out_width;
  const int64_t n = static_cast<int64_t>(blockIdx.x) *
                        static_cast<int64_t>(blockDim.x) +
                    static_cast<int64_t>(threadIdx.x);
  if (n >= positions) return;
  const int64_t ow = n % out_width;
  const int64_t oh = n / out_width;
  const int64_t k_per_channel = kh * kw;
  const int64_t in_spatial = height * width;
  for (int64_t c = static_cast<int64_t>(blockIdx.y); c < channels;
       c += static_cast<int64_t>(gridDim.y)) {
    const float* input_c = input + c * in_spatial;
    int64_t k = c * k_per_channel;
    for (int64_t dy = 0; dy < kh; ++dy) {
      const int64_t ih = oh * 2 + dy;
      const float* row_ptr = (ih < height) ? (input_c + ih * width) : nullptr;
      for (int64_t dx = 0; dx < kw; ++dx, ++k) {
        const int64_t iw = ow * 2 + dx;
        const float value =
            (row_ptr != nullptr && iw < width) ? row_ptr[iw] : 0.0f;
        columns[k * positions + n] = cast_from_f32<OutT>(value);
      }
    }
  }
}

// ZeroPad2d((0,1,0,1)) + Conv2d(3x3, stride 2), applied per frame. Requires T=1.
Tensor conv2d_stride2(CudaContext& context, const Tensor& input,
                      const Conv& conv) {
  if (input.time != 1 || input.channels != conv.input_channels) {
    throw std::invalid_argument("stride-2 convolution expects a single frame");
  }
  const int64_t out_h = (input.height + 1 - conv.kh) / 2 + 1;
  const int64_t out_w = (input.width + 1 - conv.kw) / 2 + 1;
  const int64_t positions = out_h * out_w;
  const int64_t kernel_size = conv.input_channels * conv.kh * conv.kw;
  if (positions > std::numeric_limits<int>::max() ||
      kernel_size > std::numeric_limits<int>::max() ||
      conv.output_channels > std::numeric_limits<int>::max()) {
    throw std::overflow_error("stride-2 convolution exceeds cuBLAS limits");
  }
  const size_t col_elem_bytes =
      context.is_lowp_gemm() ? sizeof(uint16_t) : sizeof(float);
  void* columns = context.acquire_columns_bytes(
      checked_bytes(checked_product({kernel_size, positions}), col_elem_bytes));
  Tensor output(context, conv.output_channels, 1, out_h, out_w);
  const dim3 grid = im2col_grid(static_cast<size_t>(positions), input.channels);
  const WanVaeGemmPrec prec = context.gemm_prec();
  if (prec == WanVaeGemmPrec::kBf16) {
    im2col_stride2_kernel<__nv_bfloat16><<<grid, 256>>>(
        input.data(), static_cast<__nv_bfloat16*>(columns), input.channels,
        input.height, input.width, conv.kh, conv.kw, out_h, out_w);
  } else if (prec == WanVaeGemmPrec::kFp16) {
    im2col_stride2_kernel<__half><<<grid, 256>>>(
        input.data(), static_cast<__half*>(columns), input.channels,
        input.height, input.width, conv.kh, conv.kw, out_h, out_w);
  } else {
    im2col_stride2_kernel<float><<<grid, 256>>>(
        input.data(), static_cast<float*>(columns), input.channels,
        input.height, input.width, conv.kh, conv.kw, out_h, out_w);
  }
  UMM_CUDA(cudaGetLastError());
  run_conv_gemm(context, static_cast<int>(positions),
                static_cast<int>(conv.output_channels),
                static_cast<int>(kernel_size), columns,
                static_cast<int>(positions), conv.weight.data(),
                conv.weight.size(), static_cast<int>(kernel_size),
                output.data(), static_cast<int>(positions));
  apply_conv_bias_and_residual(output, conv.bias, /*residual=*/nullptr,
                               /*add_bias=*/true);
  return output;
}

// AvgDown3D: zero-pad the front of the time axis, then average every
// (factor_t, factor_s, factor_s) group of the flattened (channel, t, y, x) axis
// into one output channel.
__global__ void avg_down3d_kernel(const float* input, float* output,
                                  int64_t out_channels, int64_t time,
                                  int64_t height, int64_t width,
                                  int64_t out_time, int64_t out_height,
                                  int64_t out_width, int64_t factor_t,
                                  int64_t factor_s, int64_t group_size,
                                  int64_t pad_t) {
  const int64_t factor = factor_t * factor_s * factor_s;
  const size_t count = static_cast<size_t>(out_channels * out_time * out_height *
                                           out_width);
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const int64_t ow = index % out_width;
  const int64_t oh = (index / out_width) % out_height;
  const int64_t ot = (index / (out_width * out_height)) % out_time;
  const int64_t oc = index / (out_width * out_height * out_time);
  float sum = 0.0f;
  for (int64_t g = 0; g < group_size; ++g) {
    const int64_t p = oc * group_size + g;
    const int64_t c = p / factor;
    const int64_t remainder = p % factor;
    const int64_t it = remainder / (factor_s * factor_s);
    const int64_t ih = (remainder / factor_s) % factor_s;
    const int64_t iw = remainder % factor_s;
    const int64_t src_t = ot * factor_t + it - pad_t;
    if (src_t < 0) continue;  // front zero padding
    sum += input[((c * time + src_t) * height + oh * factor_s + ih) * width +
                 ow * factor_s + iw];
  }
  output[index] = sum / static_cast<float>(group_size);
}

Tensor avg_down3d(CudaContext& context, const Tensor& input,
                  int64_t out_channels, int64_t factor_t, int64_t factor_s) {
  const int64_t factor = factor_t * factor_s * factor_s;
  const int64_t numerator = input.channels * factor;
  if (numerator % out_channels != 0) {
    throw std::invalid_argument("AvgDown3D channel ratio is not integral");
  }
  const int64_t group_size = numerator / out_channels;
  const int64_t pad_t = (factor_t - input.time % factor_t) % factor_t;
  const int64_t out_time = (input.time + pad_t) / factor_t;
  const int64_t out_height = input.height / factor_s;
  const int64_t out_width = input.width / factor_s;
  Tensor output(context, out_channels, out_time, out_height, out_width);
  avg_down3d_kernel<<<grid_for(output.numel()), 256>>>(
      input.data(), output.data(), out_channels, input.time, input.height,
      input.width, out_time, out_height, out_width, factor_t, factor_s,
      group_size, pad_t);
  UMM_CUDA(cudaGetLastError());
  return output;
}

struct EncoderDownBlock {
  std::vector<ResBlock> resnets;
  bool has_downsample = false;
  bool temporal = false;
  int64_t factor_t = 1;
  int64_t factor_s = 1;
  int64_t in_channels = 0;
  int64_t out_channels = 0;
  Conv resample_conv;
};

struct Encoder {
  Conv conv_in;
  std::vector<EncoderDownBlock> down_blocks;
  ResBlock mid_res0;
  ResBlock mid_res1;
  Norm mid_norm;
  Conv mid_qkv;
  Conv mid_proj;
  Norm norm_out;
  Conv conv_out;
  int64_t latent_channels = 0;
};

Encoder load_encoder(const SafetensorsMmap& file, const WanVaeConfig& config) {
  Encoder encoder;
  encoder.conv_in = load_conv(file, "encoder.conv_in");

  std::vector<int64_t> dims;
  dims.push_back(config.dim);
  for (const int multiplier : config.dim_mult) {
    dims.push_back(static_cast<int64_t>(config.dim) * multiplier);
  }
  const int64_t blocks = static_cast<int64_t>(config.dim_mult.size());
  for (int64_t i = 0; i < blocks; ++i) {
    EncoderDownBlock block;
    block.in_channels = dims[static_cast<size_t>(i)];
    block.out_channels = dims[static_cast<size_t>(i + 1)];
    const bool last = i == blocks - 1;
    // WanEncoder3d passes temperal_downsample[i] only for non-final blocks.
    block.temporal =
        !last && config.temperal_downsample[static_cast<size_t>(i)];
    block.factor_t = block.temporal ? 2 : 1;
    block.factor_s = last ? 1 : 2;
    for (int64_t j = 0; j < config.num_res_blocks; ++j) {
      block.resnets.push_back(load_res_block(
          file, "encoder.down_blocks." + std::to_string(i) + ".resnets." +
                    std::to_string(j)));
    }
    if (!last) {
      block.has_downsample = true;
      block.resample_conv = load_conv(
          file, "encoder.down_blocks." + std::to_string(i) +
                    ".downsampler.resample.1");
      if (block.temporal &&
          !file.contains("encoder.down_blocks." + std::to_string(i) +
                         ".downsampler.time_conv.weight")) {
        throw std::invalid_argument("encoder temporal downsampler is missing");
      }
    }
    encoder.down_blocks.push_back(std::move(block));
  }

  encoder.mid_res0 = load_res_block(file, "encoder.mid_block.resnets.0");
  encoder.mid_res1 = load_res_block(file, "encoder.mid_block.resnets.1");
  const int64_t mid_channels = encoder.mid_res0.conv1.output_channels;
  encoder.mid_norm =
      load_norm(file, "encoder.mid_block.attentions.0.norm", mid_channels);
  encoder.mid_qkv = load_conv(file, "encoder.mid_block.attentions.0.to_qkv");
  encoder.mid_proj = load_conv(file, "encoder.mid_block.attentions.0.proj");
  encoder.norm_out = load_norm(file, "encoder.norm_out", mid_channels);
  encoder.conv_out = load_conv(file, "encoder.conv_out");
  encoder.latent_channels = encoder.conv_out.output_channels;
  if (encoder.conv_out.input_channels != mid_channels ||
      encoder.latent_channels != 2 * config.z_dim) {
    throw std::invalid_argument("encoder output channels must be 2 * z_dim");
  }
  return encoder;
}

// Diffusers patchify(p=2) for [C, 1, H, W]: channel index is
// (c, x-patch, y-patch) with the x patch index outer.
std::vector<float> patchify_image(const std::vector<float>& rgb, int channels,
                                  int height, int width, int patch) {
  const int out_h = height / patch;
  const int out_w = width / patch;
  std::vector<float> output(static_cast<size_t>(channels) * patch * patch *
                            out_h * out_w);
  const size_t plane = static_cast<size_t>(height) * width;
  for (int c = 0; c < channels; ++c) {
    for (int py = 0; py < patch; ++py) {
      for (int px = 0; px < patch; ++px) {
        const size_t out_channel =
            static_cast<size_t>(c) * patch * patch + px * patch + py;
        float* dst = output.data() + out_channel * out_h * out_w;
        for (int y = 0; y < out_h; ++y) {
          for (int x = 0; x < out_w; ++x) {
            dst[static_cast<size_t>(y) * out_w + x] =
                rgb[static_cast<size_t>(c) * plane +
                    static_cast<size_t>(y * patch + py) * width +
                    (x * patch + px)];
          }
        }
      }
    }
  }
  return output;
}

}  // namespace

WanVaeEncoded encode_wan_vae_image_fp32(const std::filesystem::path& vae_dir,
                                        const std::vector<float>& rgb_nchw,
                                        int height, int width, int gpu) {
  if (height <= 0 || width <= 0 || height % 16 != 0 || width % 16 != 0) {
    throw std::invalid_argument("encoder input size must be divisible by 16");
  }
  const WanVaeConfig config = load_wan_vae_config(vae_dir);
  if (!config.is_residual || config.patch_size <= 0) {
    throw std::invalid_argument("only the Wan2.2 residual encoder is supported");
  }
  const int64_t patch = config.patch_size;
  if (height % patch != 0 || width % patch != 0) {
    throw std::invalid_argument("encoder input size must be divisible by patch_size");
  }
  const size_t expected = checked_product({3, 1, height, width});
  if (rgb_nchw.size() != expected) {
    throw std::invalid_argument("encoder RGB buffer size does not match its size");
  }

  g_peak_device_bytes = 0;
  CudaContext context(gpu);
  SafetensorsMmap file(vae_dir / "diffusion_pytorch_model.safetensors");
  Encoder encoder = load_encoder(file, config);
  if (file.tensor("quant_conv.weight").shape[0] != encoder.latent_channels) {
    throw std::invalid_argument("quant_conv does not match the encoder output");
  }
  Conv quant_conv = load_conv(file, "quant_conv");

  const auto begin = std::chrono::steady_clock::now();

  const auto patchified =
      patchify_image(rgb_nchw, 3, height, width, static_cast<int>(patch));
  Tensor x(context, static_cast<int64_t>(encoder.conv_in.input_channels), 1,
           height / patch, width / patch);
  UMM_CUDA(cudaMemcpy(x.data(), patchified.data(),
                      patchified.size() * sizeof(float),
                      cudaMemcpyHostToDevice));

  x = conv3d(context, x, encoder.conv_in);
  for (const EncoderDownBlock& block : encoder.down_blocks) {
    Tensor shortcut_input = clone_tensor(context, x);
    for (const ResBlock& resnet : block.resnets) {
      x = res_block_plain(context, x, resnet);
    }
    if (block.has_downsample) {
      x = conv2d_stride2(context, x, block.resample_conv);
    }
    Tensor shortcut =
        avg_down3d(context, shortcut_input, block.out_channels, block.factor_t,
                   block.factor_s);
    shortcut_input = Tensor{};
    if (shortcut.channels != x.channels || shortcut.time != x.time ||
        shortcut.height != x.height || shortcut.width != x.width) {
      throw std::runtime_error("AvgDown3D shortcut shape mismatch");
    }
    add_inplace(x, shortcut);
  }

  x = res_block_plain(context, x, encoder.mid_res0);
  x = attention(context, x, encoder.mid_norm, encoder.mid_qkv,
                encoder.mid_proj);
  x = res_block_plain(context, x, encoder.mid_res1);
  x = rms_norm(context, x, encoder.norm_out, /*apply_silu=*/true);
  x = conv3d(context, x, encoder.conv_out);
  x = conv3d(context, x, quant_conv);

  const int64_t latent_h = height / 16;
  const int64_t latent_w = width / 16;
  if (x.channels != 2 * config.z_dim || x.time != 1 || x.height != latent_h ||
      x.width != latent_w) {
    throw std::runtime_error("unexpected Wan2.2 VAE encoder output shape");
  }

  // posterior mean = first z_dim channels of the 2 * z_dim statistics.
  const size_t plane = static_cast<size_t>(latent_h * latent_w);
  std::vector<float> raw(static_cast<size_t>(config.z_dim) * plane);
  UMM_CUDA(cudaMemcpy(raw.data(), x.data(), raw.size() * sizeof(float),
                      cudaMemcpyDeviceToHost));
  UMM_CUDA(cudaDeviceSynchronize());

  WanVaeEncoded output;
  output.shape = {1, config.z_dim, 1, latent_h, latent_w};
  output.data.resize(raw.size());
  for (int c = 0; c < config.z_dim; ++c) {
    const float mean_value = config.latents_mean[static_cast<size_t>(c)];
    const float std_value = config.latents_std[static_cast<size_t>(c)];
    const float* src = raw.data() + static_cast<size_t>(c) * plane;
    float* dst = output.data.data() + static_cast<size_t>(c) * plane;
    for (size_t i = 0; i < plane; ++i) {
      dst[i] = (src[i] - mean_value) / std_value;
    }
  }
  output.encode_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - begin)
                         .count();
  output.peak_native_device_bytes = g_peak_device_bytes;
  return output;
}

}  // namespace umminfer::video

#endif  // UMM_VIDEO_CUDA
