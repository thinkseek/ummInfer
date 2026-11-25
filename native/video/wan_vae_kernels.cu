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
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
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

unsigned grid_for(size_t count) {
  const size_t blocks = (count + 255) / 256;
  if (blocks > static_cast<size_t>(std::numeric_limits<unsigned>::max())) {
    throw std::overflow_error("CUDA grid dimension overflow");
  }
  return static_cast<unsigned>(std::max<size_t>(blocks, 1));
}

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t elements) : elements_(elements) {
    if (elements == 0) return;
    if (elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
      throw std::overflow_error("device allocation size overflow");
    }
    UMM_CUDA(cudaMalloc(reinterpret_cast<void**>(&data_), elements * sizeof(float)));
    g_device_bytes += elements * sizeof(float);
    g_peak_device_bytes = std::max(g_peak_device_bytes, g_device_bytes);
  }
  ~DeviceBuffer() { release(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(other.data_), elements_(other.elements_) {
    other.data_ = nullptr;
    other.elements_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      data_ = other.data_;
      elements_ = other.elements_;
      other.data_ = nullptr;
      other.elements_ = 0;
    }
    return *this;
  }
  float* data() const { return data_; }
  size_t size() const { return elements_; }

 private:
  void release() {
    if (data_ != nullptr) {
      cudaFree(data_);
      g_device_bytes -= elements_ * sizeof(float);
      data_ = nullptr;
      elements_ = 0;
    }
  }
  float* data_ = nullptr;
  size_t elements_ = 0;
};

DeviceBuffer upload(const std::vector<float>& values) {
  DeviceBuffer result(values.size());
  if (!values.empty()) {
    UMM_CUDA(cudaMemcpy(result.data(), values.data(), values.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  }
  return result;
}

struct Tensor {
  int64_t channels = 0;
  int64_t time = 0;
  int64_t height = 0;
  int64_t width = 0;
  DeviceBuffer storage;

  Tensor() = default;
  Tensor(int64_t c, int64_t t, int64_t h, int64_t w)
      : channels(c), time(t), height(h), width(w),
        storage(checked_product({c, t, h, w})) {}
  Tensor(Tensor&&) = default;
  Tensor& operator=(Tensor&&) = default;
  int64_t positions() const { return time * height * width; }
  size_t numel() const { return storage.size(); }
  float* data() const { return storage.data(); }
};

Tensor clone_tensor(const Tensor& source) {
  Tensor copy(source.channels, source.time, source.height, source.width);
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

// im2col for a causal 3D convolution: left temporal padding of kd-1 (reading
// from `cache` when provided, otherwise zeros) and symmetric spatial padding
// of kh/2 and kw/2. Output matrix is [K, positions] row-major with
// K = channels * kd * kh * kw.
__global__ void im2col_causal_kernel(const float* input, const float* cache,
                                     int64_t cache_time, float* columns,
                                     int64_t channels, int64_t time, int64_t height,
                                     int64_t width, int64_t kd, int64_t kh,
                                     int64_t kw, int64_t position_begin,
                                     int64_t chunk_positions) {
  const int64_t spatial = height * width;
  const int64_t local = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (local >= chunk_positions) return;
  const int64_t n = position_begin + local;
  const int64_t ow = n % width;
  const int64_t oh = (n / width) % height;
  const int64_t ot = n / spatial;
  int64_t k = 0;
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t dt = 0; dt < kd; ++dt) {
      const int64_t it = ot + dt - (kd - 1);
      for (int64_t dy = 0; dy < kh; ++dy) {
        const int64_t ih = oh + dy - kh / 2;
        for (int64_t dx = 0; dx < kw; ++dx, ++k) {
          const int64_t iw = ow + dx - kw / 2;
          float value = 0.0f;
          if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            if (it >= 0 && it < time) {
              value = input[((c * time + it) * height + ih) * width + iw];
            } else if (it < 0 && cache != nullptr) {
              const int64_t ct = cache_time + it;
              if (ct >= 0 && ct < cache_time) {
                value = cache[((c * cache_time + ct) * height + ih) * width + iw];
              }
            }
          }
          columns[k * chunk_positions + local] = value;
        }
      }
    }
  }
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

__global__ void bias_chunk_kernel(float* x, int64_t positions,
                                  int64_t position_begin, int64_t chunk_positions,
                                  int64_t channels, const float* bias) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= chunk_positions * channels) return;
  const int64_t channel = index / chunk_positions;
  const int64_t local = index % chunk_positions;
  x[channel * positions + position_begin + local] += bias[channel];
}

class CudaContext {
 public:
  explicit CudaContext(int gpu) {
    UMM_CUDA(cudaSetDevice(gpu));
    UMM_CUBLAS(cublasCreate(&handle_));
    const char* pedantic = std::getenv("UMM_VAE_PEDANTIC_FP32");
    const bool use_pedantic =
        pedantic != nullptr && pedantic[0] != '\0' && pedantic[0] != '0';
    UMM_CUBLAS(cublasSetMathMode(
        handle_,
        use_pedantic ? CUBLAS_PEDANTIC_MATH : CUBLAS_TF32_TENSOR_OP_MATH));
  }
  ~CudaContext() {
    if (handle_ != nullptr) cublasDestroy(handle_);
  }
  cublasHandle_t handle() const { return handle_; }

 private:
  cublasHandle_t handle_ = nullptr;
};

// out[o, n] = sum_k weight[o, k] * columns[k, n]. Position chunks cap the
// temporary im2col allocation at 1 GiB; the output keeps full NCTHW strides.
// When `cache` is provided for a kd > 1 convolution, the left temporal context
// is read from cache->frames and cache->frames is updated to the trailing kd-1
// input frames.
Tensor conv3d(CudaContext& context, const Tensor& input, const Conv& conv,
              CausalCache* cache = nullptr) {
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
  const bool use_cache = cache != nullptr && conv.kd > 1;
  const int64_t cache_time = use_cache ? (conv.kd - 1) : 0;
  if (use_cache && cache->frames.numel() == 0) {
    cache->frames = Tensor(input.channels, cache_time, input.height, input.width);
    UMM_CUDA(cudaMemset(cache->frames.data(), 0,
                        cache->frames.numel() * sizeof(float)));
  }
  const float* cache_ptr = use_cache ? cache->frames.data() : nullptr;

  constexpr int64_t kMaxColumnElements = 1LL << 28;
  const int64_t chunk_capacity =
      std::max<int64_t>(1, std::min(positions, kMaxColumnElements / kernel_size));
  DeviceBuffer columns(checked_product({kernel_size, chunk_capacity}));
  Tensor output(conv.output_channels, input.time, input.height, input.width);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  for (int64_t begin = 0; begin < positions; begin += chunk_capacity) {
    const int64_t chunk = std::min(chunk_capacity, positions - begin);
    im2col_causal_kernel<<<grid_for(chunk), 256>>>(
        input.data(), cache_ptr, cache_time, columns.data(), input.channels,
        input.time, input.height, input.width, conv.kd, conv.kh, conv.kw, begin,
        chunk);
    UMM_CUDA(cudaGetLastError());
    UMM_CUBLAS(cublasSgemm(
        context.handle(), CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(chunk),
        static_cast<int>(conv.output_channels), static_cast<int>(kernel_size), &alpha,
        columns.data(), static_cast<int>(chunk), conv.weight.data(),
        static_cast<int>(kernel_size), &beta, output.data() + begin,
        static_cast<int>(positions)));
    bias_chunk_kernel<<<grid_for(chunk * conv.output_channels), 256>>>(
        output.data(), positions, begin, chunk, conv.output_channels,
        conv.bias.data());
    UMM_CUDA(cudaGetLastError());
  }
  if (use_cache) {
    Tensor next_cache(input.channels, cache_time, input.height, input.width);
    update_causal_cache_kernel<<<grid_for(next_cache.numel()), 256>>>(
        cache->frames.data(), input.data(), next_cache.data(), input.channels,
        cache_time, input.time, input.height * input.width);
    UMM_CUDA(cudaGetLastError());
    cache->frames = std::move(next_cache);
  }
  return output;
}

__global__ void rms_norm_kernel(const float* input, float* output,
                                const float* gamma, int64_t channels,
                                size_t positions) {
  const size_t n = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (n >= positions) return;
  float sum = 0.0f;
  for (int64_t c = 0; c < channels; ++c) {
    const float value = input[static_cast<size_t>(c) * positions + n];
    sum += value * value;
  }
  const float denominator = fmaxf(sqrtf(sum), 1e-12f);
  const float scale = sqrtf(static_cast<float>(channels)) / denominator;
  for (int64_t c = 0; c < channels; ++c) {
    const size_t index = static_cast<size_t>(c) * positions + n;
    output[index] = input[index] * scale * gamma[c];
  }
}

Tensor rms_norm(const Tensor& input, const Norm& norm) {
  if (input.channels != norm.channels) {
    throw std::invalid_argument("RMSNorm channel mismatch");
  }
  Tensor output(input.channels, input.time, input.height, input.width);
  rms_norm_kernel<<<grid_for(input.positions()), 256>>>(
      input.data(), output.data(), norm.gamma.data(), input.channels,
      input.positions());
  UMM_CUDA(cudaGetLastError());
  return output;
}

__global__ void silu_kernel(float* x, size_t count) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const float value = x[index];
  x[index] = value / (1.0f + expf(-value));
}

void silu_inplace(Tensor& tensor) {
  silu_kernel<<<grid_for(tensor.numel()), 256>>>(tensor.data(), tensor.numel());
  UMM_CUDA(cudaGetLastError());
}

__global__ void add_kernel(float* x, const float* residual, size_t count) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) x[index] += residual[index];
}

void add_inplace(Tensor& tensor, const Tensor& residual) {
  if (tensor.numel() != residual.numel()) {
    throw std::invalid_argument("residual tensor size mismatch");
  }
  add_kernel<<<grid_for(tensor.numel()), 256>>>(tensor.data(), residual.data(),
                                                tensor.numel());
  UMM_CUDA(cudaGetLastError());
}

__global__ void nearest_spatial_kernel(const float* input, float* output,
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

Tensor nearest_spatial(const Tensor& input) {
  Tensor output(input.channels, input.time, input.height * 2, input.width * 2);
  nearest_spatial_kernel<<<grid_for(output.numel()), 256>>>(
      input.data(), output.data(), input.channels, input.time, input.height,
      input.width);
  UMM_CUDA(cudaGetLastError());
  return output;
}

// Temporal upsample for non-first streaming chunks: each input step j expands
// into two output frames from the two channel halves of the cached causal
// time_conv output. For the first chunk (latent step 0), Diffusers emits frame
// 0 unchanged without running or updating time_conv.
__global__ void temporal_interleave_chunk_kernel(const float* convolved,
                                                 float* output,
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

Tensor temporal_upsample(CudaContext& context, const Tensor& input,
                         const Conv& time_conv, CausalCache& time_cache,
                         bool first_chunk) {
  if (time_conv.kd != 3 || time_conv.kh != 1 || time_conv.kw != 1 ||
      time_conv.input_channels != input.channels ||
      time_conv.output_channels != input.channels * 2) {
    throw std::invalid_argument("time convolution must be C -> 2C with a 3x1x1 kernel");
  }
  if (first_chunk) {
    return clone_tensor(input);
  }
  Tensor convolved = conv3d(context, input, time_conv, &time_cache);
  Tensor expanded(input.channels, input.time * 2, input.height, input.width);
  temporal_interleave_chunk_kernel<<<grid_for(expanded.numel()), 256>>>(
      convolved.data(), expanded.data(), input.channels, input.time,
      input.height * input.width);
  UMM_CUDA(cudaGetLastError());
  return expanded;
}

// DupUp3D shortcut: nearest duplication over time and space, dropping the
// leading factor_t-1 frames only on the first streaming chunk.
__global__ void dup_up_kernel(const float* input, float* output, int64_t out_channels,
                              int64_t in_time, int64_t out_time, int64_t height,
                              int64_t width, int64_t factor_t, int64_t factor_s,
                              int64_t repeats, int64_t drop_first) {
  const int64_t out_height = height * factor_s;
  const int64_t out_width = width * factor_s;
  const size_t count =
      static_cast<size_t>(out_channels * out_time * out_height * out_width);
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const int64_t ow = index % out_width;
  const int64_t oh = (index / out_width) % out_height;
  const int64_t ot = (index / (out_height * out_width)) % out_time;
  const int64_t oc = index / (out_time * out_height * out_width);
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
  output[index] =
      input[((source_channel * in_time + st) * height + h) * width + w];
}

Tensor dup_up3d(const Tensor& input, int64_t out_channels, int64_t factor_t,
                int64_t factor_s, bool first_chunk) {
  const int64_t factor = factor_t * factor_s * factor_s;
  const int64_t numerator = out_channels * factor;
  if (numerator % input.channels != 0) {
    throw std::invalid_argument("DupUp3D channel ratio is not integral");
  }
  const int64_t repeats = numerator / input.channels;
  const int64_t drop_first = first_chunk ? (factor_t - 1) : 0;
  const int64_t out_time = factor_t * input.time - drop_first;
  Tensor output(out_channels, out_time, input.height * factor_s,
                input.width * factor_s);
  dup_up_kernel<<<grid_for(output.numel()), 256>>>(
      input.data(), output.data(), out_channels, input.time, out_time,
      input.height, input.width, factor_t, factor_s, repeats, drop_first);
  UMM_CUDA(cudaGetLastError());
  return output;
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
  Tensor normalized = rms_norm(input, norm);
  Tensor packed = conv3d(context, normalized, qkv);
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
  DeviceBuffer scores(checked_product({frames, spatial, spatial}));
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

  Tensor attended(channels, input.time, input.height, input.width);
  UMM_CUBLAS(cublasSgemmStridedBatched(
      context.handle(), CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(spatial),
      static_cast<int>(channels), static_cast<int>(spatial), &alpha, scores.data(),
      static_cast<int>(spatial), static_cast<int>(stride), value,
      static_cast<int>(positions), static_cast<int>(spatial), &beta, attended.data(),
      static_cast<int>(positions), static_cast<int>(spatial),
      static_cast<int>(frames)));
  Tensor output = conv3d(context, attended, projection);
  add_inplace(output, input);
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
  Tensor output = rms_norm(input, block.norm1);
  silu_inplace(output);
  output = conv3d(context, output, block.conv1, &cache.conv1);
  output = rms_norm(output, block.norm2);
  silu_inplace(output);
  output = conv3d(context, output, block.conv2, &cache.conv2);
  if (block.has_shortcut) {
    Tensor identity = conv3d(context, input, block.shortcut);
    add_inplace(output, identity);
  } else {
    add_inplace(output, input);
  }
  return output;
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

Tensor unpatchify_clamp(const Tensor& input, int64_t output_channels, int64_t patch) {
  if (input.channels != output_channels * patch * patch) {
    throw std::invalid_argument("unpatchify channel mismatch");
  }
  Tensor output(output_channels, input.time, input.height * patch,
                input.width * patch);
  unpatchify_clamp_kernel<<<grid_for(output.numel()), 256>>>(
      input.data(), output.data(), output_channels, input.time, input.height,
      input.width, patch);
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
      Tensor spatial = nearest_spatial(expanded);
      current = conv3d(context, spatial, block.resample_conv, &cache.resample_conv);
    } else {
      Tensor spatial = nearest_spatial(current);
      current = conv3d(context, spatial, block.resample_conv, &cache.resample_conv);
    }
    // DupUp3D shortcut reads the block input, before any residual block ran.
    Tensor shortcut = dup_up3d(input, block.output_channels, block.factor_t, 2,
                               first_chunk);
    add_inplace(current, shortcut);
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

    Tensor input(latent_c, 1, latent_h, latent_w);
    UMM_CUDA(cudaMemcpy(input.data(), frame_host.data(),
                        frame_host.size() * sizeof(float), cudaMemcpyHostToDevice));
    denormalize_kernel<<<grid_for(input.numel()), 256>>>(
        input.data(), input.positions(), input.channels, mean.data(), stddev.data());
    UMM_CUDA(cudaGetLastError());

    const bool first_chunk = (t == 0);
    Tensor current = conv3d(context, input, post_quant);
    current = conv3d(context, current, decoder.conv_in, &cache.conv_in);
    current = res_block(context, current, decoder.mid_res0, cache.mid_res0);
    current = attention(context, current, decoder.mid_norm, decoder.mid_qkv,
                        decoder.mid_proj);
    current = res_block(context, current, decoder.mid_res1, cache.mid_res1);
    for (size_t i = 0; i < decoder.up_blocks.size(); ++i) {
      current = upsample_block(context, current, decoder.up_blocks[i],
                               cache.up_blocks[i], first_chunk);
    }
    current = rms_norm(current, decoder.norm_out);
    silu_inplace(current);
    current = conv3d(context, current, decoder.conv_out, &cache.conv_out);
    current = unpatchify_clamp(current, output_channels, config.patch_size);

    const int64_t expected_chunk_t = wan_vae_chunk_frame_count(t, temporal_factor);
    const int64_t out_t_offset = wan_vae_chunk_frame_offset(t, temporal_factor);
    if (current.channels != output_channels || current.time != expected_chunk_t ||
        current.height != out_h || current.width != out_w) {
      throw std::runtime_error("unexpected streamed Wan2.2 VAE chunk shape");
    }
    const size_t chunk_plane = static_cast<size_t>(current.time * out_spatial);
    for (int64_t c = 0; c < output_channels; ++c) {
      const float* dev_plane = current.data() + static_cast<size_t>(c) * chunk_plane;
      float* host_plane =
          output.data.data() +
          static_cast<size_t>((c * total_out_t + out_t_offset) * out_spatial);
      UMM_CUDA(cudaMemcpy(host_plane, dev_plane, chunk_plane * sizeof(float),
                          cudaMemcpyDeviceToHost));
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
  Tensor output = rms_norm(input, block.norm1);
  silu_inplace(output);
  output = conv3d(context, output, block.conv1);
  output = rms_norm(output, block.norm2);
  silu_inplace(output);
  output = conv3d(context, output, block.conv2);
  if (block.has_shortcut) {
    Tensor identity = conv3d(context, input, block.shortcut);
    add_inplace(output, identity);
  } else {
    add_inplace(output, input);
  }
  return output;
}

// im2col for a 2D stride-2 convolution whose input was ZeroPad2d((0,1,0,1))'d,
// which is exactly WanResample's "downsample2d" pre-processing.
__global__ void im2col_stride2_kernel(const float* input, float* columns,
                                      int64_t channels, int64_t height,
                                      int64_t width, int64_t kh, int64_t kw,
                                      int64_t out_height, int64_t out_width) {
  const int64_t positions = out_height * out_width;
  const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (n >= positions) return;
  const int64_t ow = n % out_width;
  const int64_t oh = n / out_width;
  int64_t k = 0;
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t dy = 0; dy < kh; ++dy) {
      const int64_t ih = oh * 2 + dy;
      for (int64_t dx = 0; dx < kw; ++dx, ++k) {
        const int64_t iw = ow * 2 + dx;
        columns[k * positions + n] =
            (ih < height && iw < width)
                ? input[(c * height + ih) * width + iw]
                : 0.0f;
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
  DeviceBuffer columns(checked_product({kernel_size, positions}));
  Tensor output(conv.output_channels, 1, out_h, out_w);
  im2col_stride2_kernel<<<grid_for(positions), 256>>>(
      input.data(), columns.data(), input.channels, input.height, input.width,
      conv.kh, conv.kw, out_h, out_w);
  UMM_CUDA(cudaGetLastError());
  const float alpha = 1.0f;
  const float beta = 0.0f;
  UMM_CUBLAS(cublasSgemm(
      context.handle(), CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(positions),
      static_cast<int>(conv.output_channels), static_cast<int>(kernel_size),
      &alpha, columns.data(), static_cast<int>(positions), conv.weight.data(),
      static_cast<int>(kernel_size), &beta, output.data(),
      static_cast<int>(positions)));
  bias_chunk_kernel<<<grid_for(positions * conv.output_channels), 256>>>(
      output.data(), positions, 0, positions, conv.output_channels,
      conv.bias.data());
  UMM_CUDA(cudaGetLastError());
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

Tensor avg_down3d(const Tensor& input, int64_t out_channels, int64_t factor_t,
                  int64_t factor_s) {
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
  Tensor output(out_channels, out_time, out_height, out_width);
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
  Tensor x(static_cast<int64_t>(encoder.conv_in.input_channels), 1,
           height / patch, width / patch);
  UMM_CUDA(cudaMemcpy(x.data(), patchified.data(),
                      patchified.size() * sizeof(float),
                      cudaMemcpyHostToDevice));

  x = conv3d(context, x, encoder.conv_in);
  for (const EncoderDownBlock& block : encoder.down_blocks) {
    Tensor shortcut_input = clone_tensor(x);
    for (const ResBlock& resnet : block.resnets) {
      x = res_block_plain(context, x, resnet);
    }
    if (block.has_downsample) {
      x = conv2d_stride2(context, x, block.resample_conv);
    }
    Tensor shortcut =
        avg_down3d(shortcut_input, block.out_channels, block.factor_t,
                   block.factor_s);
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
  x = rms_norm(x, encoder.norm_out);
  silu_inplace(x);
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
