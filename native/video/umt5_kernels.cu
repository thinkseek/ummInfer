// SPDX-License-Identifier: Apache-2.0
// UMT5-XXL BF16 weights, FP32 accumulation, and bidirectional attention.
#include "umt5_encoder.hpp"

#include "safetensors_mmap.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace umminfer::video {
namespace {

using Bf16 = __nv_bfloat16;
constexpr int kLayers = 24;
constexpr int kModel = 4096;
constexpr int kFf = 10240;
constexpr int kHeads = 64;
constexpr int kHeadDim = 64;
constexpr int kVocab = 256384;
constexpr int kThreads = 256;

#define UMM_CUDA(expr)                                                        \
  do {                                                                        \
    const cudaError_t status = (expr);                                         \
    if (status != cudaSuccess) {                                               \
      throw std::runtime_error(std::string(#expr) + ": " +                   \
                               cudaGetErrorString(status));                     \
    }                                                                          \
  } while (0)

#define UMM_CUBLAS(expr)                                                      \
  do {                                                                        \
    const cublasStatus_t status = (expr);                                      \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      throw std::runtime_error(std::string(#expr) + ": cuBLAS status " +     \
                               std::to_string(static_cast<int>(status)));       \
    }                                                                          \
  } while (0)

size_t checked_bytes(size_t count, size_t element_size) {
  if (element_size != 0 && count > std::numeric_limits<size_t>::max() / element_size) {
    throw std::overflow_error("UMT5 allocation size overflows size_t");
  }
  return count * element_size;
}

int grid_for(size_t count) {
  const size_t blocks = (count + kThreads - 1) / kThreads;
  if (blocks > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("UMT5 CUDA grid is too large");
  }
  return static_cast<int>(std::max<size_t>(blocks, 1));
}

struct AllocationTracker {
  size_t current = 0;
  size_t peak = 0;
};

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(size_t bytes, AllocationTracker& tracker) { allocate(bytes, tracker); }
  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
      tracker_->current -= bytes_;
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void allocate(size_t bytes, AllocationTracker& tracker) {
    if (data_ != nullptr) throw std::logic_error("DeviceBuffer allocated twice");
    if (bytes == 0) return;
    void* data = nullptr;
    UMM_CUDA(cudaMalloc(&data, bytes));
    data_ = data;
    bytes_ = bytes;
    tracker_ = &tracker;
    tracker.current += bytes;
    tracker.peak = std::max(tracker.peak, tracker.current);
  }

  void* data() const { return data_; }
  template <typename T>
  T* as() const { return static_cast<T*>(data_); }

 private:
  void* data_ = nullptr;
  size_t bytes_ = 0;
  AllocationTracker* tracker_ = nullptr;
};

class CudaContext {
 public:
  explicit CudaContext(int gpu) {
    UMM_CUDA(cudaSetDevice(gpu));
    UMM_CUBLAS(cublasCreate(&handle_));
    try {
      UMM_CUBLAS(cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH));
    } catch (...) {
      cublasDestroy(handle_);
      handle_ = nullptr;
      throw;
    }
  }
  ~CudaContext() {
    if (handle_ != nullptr) cublasDestroy(handle_);
  }
  cublasHandle_t get() const { return handle_; }

 private:
  cublasHandle_t handle_ = nullptr;
};

struct LayerWeights {
  const Bf16* q;
  const Bf16* k;
  const Bf16* v;
  const Bf16* o;
  const Bf16* relative_bias;
  const Bf16* attention_norm;
  const Bf16* wi_0;
  const Bf16* wi_1;
  const Bf16* wo;
  const Bf16* ffn_norm;
};

struct WeightSpec {
  std::string name;
  std::vector<int64_t> shape;
  size_t offset = 0;
};

__global__ void embedding_kernel(const int32_t* ids,
                                 const Bf16* embedding,
                                 Bf16* output,
                                 int length) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= length * kModel) return;
  const int token = index / kModel;
  const int channel = index - token * kModel;
  output[index] = embedding[static_cast<int64_t>(ids[token]) * kModel + channel];
}

__global__ void rms_norm_kernel(const Bf16* input,
                                const Bf16* weight,
                                Bf16* output,
                                int length) {
  const int token = blockIdx.x;
  if (token >= length) return;
  const size_t base = static_cast<size_t>(token) * kModel;
  float sum = 0.0f;
  for (int i = threadIdx.x; i < kModel; i += blockDim.x) {
    const float value = __bfloat162float(input[base + i]);
    sum += value * value;
  }
  __shared__ float reduction[kThreads];
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = kThreads / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inverse = rsqrtf(reduction[0] / kModel + 1e-6f);
  for (int i = threadIdx.x; i < kModel; i += blockDim.x) {
    const float value = __bfloat162float(input[base + i]);
    const float normalized = __bfloat162float(__float2bfloat16_rn(value * inverse));
    output[base + i] = __float2bfloat16_rn(
        __bfloat162float(weight[i]) * normalized);
  }
}

__global__ void residual_kernel(const Bf16* input,
                                const Bf16* residual,
                                Bf16* output,
                                size_t count) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  output[index] = __float2bfloat16_rn(
      __bfloat162float(input[index]) + __bfloat162float(residual[index]));
}

__global__ void attention_softmax_kernel(const Bf16* scores,
                                         const Bf16* relative_bias,
                                         const int32_t* buckets,
                                         const int32_t* mask,
                                         float* probabilities,
                                         int length) {
  const int row = blockIdx.x;
  const int query = row % length;
  const int head = row / length;
  const size_t score_base = static_cast<size_t>(row) * length;
  float local_max = -3.4028234663852886e38f;
  for (int key = threadIdx.x; key < length; key += blockDim.x) {
    const float bias = __bfloat162float(
        relative_bias[buckets[query * length + key] * kHeads + head]);
    constexpr float kMask = -3.3895313892515355e38f;
    const float mask_value = mask[key] != 0 ? 0.0f : kMask;
    const float position = __bfloat162float(__float2bfloat16_rn(bias + mask_value));
    const float score = __bfloat162float(scores[score_base + key]);
    const float biased = __bfloat162float(__float2bfloat16_rn(score + position));
    local_max = fmaxf(local_max, biased);
  }
  __shared__ float reduction[kThreads];
  reduction[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = kThreads / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x],
                                     reduction[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float maximum = reduction[0];
  __syncthreads();

  float local_sum = 0.0f;
  for (int key = threadIdx.x; key < length; key += blockDim.x) {
    const float bias = __bfloat162float(
        relative_bias[buckets[query * length + key] * kHeads + head]);
    constexpr float kMask = -3.3895313892515355e38f;
    const float mask_value = mask[key] != 0 ? 0.0f : kMask;
    const float position = __bfloat162float(__float2bfloat16_rn(bias + mask_value));
    const float score = __bfloat162float(scores[score_base + key]);
    const float biased = __bfloat162float(__float2bfloat16_rn(score + position));
    const float weight = expf(biased - maximum);
    probabilities[score_base + key] = weight;
    local_sum += weight;
  }
  reduction[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = kThreads / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }
  const float inverse = 1.0f / reduction[0];
  for (int key = threadIdx.x; key < length; key += blockDim.x) {
    probabilities[score_base + key] *= inverse;
  }
}

__global__ void float_to_bf16_kernel(const float* input,
                                     Bf16* output,
                                     size_t count) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) output[index] = __float2bfloat16_rn(input[index]);
}

__device__ float rounded(float value) {
  return __bfloat162float(__float2bfloat16_rn(value));
}

__global__ void gated_gelu_kernel(const Bf16* linear,
                                  Bf16* gated,
                                  size_t count) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const float input = __bfloat162float(gated[index]);
  // Match NewGELUActivation's eager BF16 rounding between tensor operations.
  const float cube = rounded(input * input * input);
  const float polynomial = rounded(input + rounded(0.044715f * cube));
  const float scaled = rounded(0.7978845608028654f * polynomial);
  const float hyperbolic = rounded(tanhf(scaled));
  const float gate = rounded(rounded(0.5f * input) * rounded(1.0f + hyperbolic));
  gated[index] = __float2bfloat16_rn(
      gate * __bfloat162float(linear[index]));
}

__global__ void output_kernel(const Bf16* input,
                              const int32_t* mask,
                              float* output,
                              int length) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= length * kModel) return;
  const int token = index / kModel;
  output[index] = mask[token] != 0 ? __bfloat162float(input[index]) : 0.0f;
}

void linear(cublasHandle_t handle,
            const Bf16* weight,
            const Bf16* input,
            Bf16* output,
            int output_features,
            int input_features,
            int length) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  UMM_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                          output_features, length, input_features,
                          &alpha, weight, CUDA_R_16BF, input_features,
                          input, CUDA_R_16BF, input_features,
                          &beta, output, CUDA_R_16BF, output_features,
                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

void validate_config(const nlohmann::json& config) {
  const auto expect_int = [&](const char* key, int expected) {
    if (config.at(key).get<int>() != expected) {
      throw std::invalid_argument(std::string("unsupported UMT5 config ") + key);
    }
  };
  expect_int("d_model", kModel);
  expect_int("d_ff", kFf);
  expect_int("num_layers", kLayers);
  expect_int("num_heads", kHeads);
  expect_int("d_kv", kHeadDim);
  expect_int("vocab_size", kVocab);
  expect_int("relative_attention_num_buckets", 32);
  expect_int("relative_attention_max_distance", 128);
  if (std::fabs(config.at("layer_norm_epsilon").get<double>() - 1e-6) > 1e-12 ||
      config.at("feed_forward_proj").get<std::string>() != "gated-gelu") {
    throw std::invalid_argument("unsupported UMT5 normalization or activation");
  }
}

}  // namespace

struct Umt5Encoder::Impl {
  Impl(const std::filesystem::path& directory, int gpu) : context_(gpu) {
    const auto start = std::chrono::steady_clock::now();
    std::ifstream config_input(directory / "config.json");
    if (!config_input) throw std::runtime_error("failed to open UMT5 config.json");
    validate_config(nlohmann::json::parse(config_input));

    std::vector<WeightSpec> specs;
    const auto add = [&](const std::string& name, std::vector<int64_t> shape) {
      specs.push_back({name, std::move(shape), 0});
    };
    add("shared.weight", {kVocab, kModel});
    for (int layer = 0; layer < kLayers; ++layer) {
      const std::string prefix = "encoder.block." + std::to_string(layer) + ".layer.";
      add(prefix + "0.SelfAttention.q.weight", {kModel, kModel});
      add(prefix + "0.SelfAttention.k.weight", {kModel, kModel});
      add(prefix + "0.SelfAttention.v.weight", {kModel, kModel});
      add(prefix + "0.SelfAttention.o.weight", {kModel, kModel});
      add(prefix + "0.SelfAttention.relative_attention_bias.weight", {32, kHeads});
      add(prefix + "0.layer_norm.weight", {kModel});
      add(prefix + "1.DenseReluDense.wi_0.weight", {kFf, kModel});
      add(prefix + "1.DenseReluDense.wi_1.weight", {kFf, kModel});
      add(prefix + "1.DenseReluDense.wo.weight", {kModel, kFf});
      add(prefix + "1.layer_norm.weight", {kModel});
    }
    add("encoder.final_layer_norm.weight", {kModel});

    const std::filesystem::path index = directory / "model.safetensors.index.json";
    ShardedSafetensorsMmap shards(std::filesystem::exists(index)
                                      ? index
                                      : directory / "model.safetensors");
    size_t total = 0;
    for (auto& spec : specs) {
      const auto& view = shards.tensor(spec.name);
      if (view.shape != spec.shape) {
        throw std::invalid_argument("unexpected UMT5 tensor shape: " + spec.name);
      }
      if (view.dtype != SafetensorsDType::kBF16) {
        throw std::invalid_argument("UMT5 tensor is not BF16: " + spec.name);
      }
      if (total > std::numeric_limits<size_t>::max() - 255) {
        throw std::overflow_error("UMT5 weight offsets overflow");
      }
      spec.offset = (total + 255) & ~size_t{255};
      total = spec.offset;
      if (view.size_bytes > std::numeric_limits<size_t>::max() - total) {
        throw std::overflow_error("UMT5 weight allocation overflows");
      }
      total += view.size_bytes;
    }
    weights_.allocate(total, tracker_);
    std::unordered_map<std::string, const Bf16*> loaded;
    for (const auto& spec : specs) {
      const auto& view = shards.tensor(spec.name);
      void* destination = weights_.as<std::byte>() + spec.offset;
      UMM_CUDA(cudaMemcpy(destination, view.data, view.size_bytes, cudaMemcpyHostToDevice));
      loaded.emplace(spec.name, static_cast<const Bf16*>(destination));
    }
    embedding_ = loaded.at("shared.weight");
    final_norm_ = loaded.at("encoder.final_layer_norm.weight");
    layers_.reserve(kLayers);
    for (int layer = 0; layer < kLayers; ++layer) {
      const std::string prefix = "encoder.block." + std::to_string(layer) + ".layer.";
      layers_.push_back({
          loaded.at(prefix + "0.SelfAttention.q.weight"),
          loaded.at(prefix + "0.SelfAttention.k.weight"),
          loaded.at(prefix + "0.SelfAttention.v.weight"),
          loaded.at(prefix + "0.SelfAttention.o.weight"),
          loaded.at(prefix + "0.SelfAttention.relative_attention_bias.weight"),
          loaded.at(prefix + "0.layer_norm.weight"),
          loaded.at(prefix + "1.DenseReluDense.wi_0.weight"),
          loaded.at(prefix + "1.DenseReluDense.wi_1.weight"),
          loaded.at(prefix + "1.DenseReluDense.wo.weight"),
          loaded.at(prefix + "1.layer_norm.weight"),
      });
    }
    UMM_CUDA(cudaDeviceSynchronize());
    load_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
  }

  Umt5EncoderOutput encode(const Umt5TokenBatch& tokens) {
    const auto start = std::chrono::steady_clock::now();
    const size_t length_size = tokens.ids.size();
    if (length_size == 0 || length_size > 512 ||
        length_size != tokens.attention_mask.size()) {
      throw std::invalid_argument("invalid UMT5 token batch dimensions");
    }
    const int length = static_cast<int>(length_size);
    bool has_active_token = false;
    for (size_t i = 0; i < length_size; ++i) {
      if (tokens.ids[i] < 0 || tokens.ids[i] >= kVocab ||
          (tokens.attention_mask[i] != 0 && tokens.attention_mask[i] != 1)) {
        throw std::invalid_argument("invalid UMT5 token id or attention mask");
      }
      has_active_token = has_active_token || tokens.attention_mask[i] == 1;
    }
    if (!has_active_token) throw std::invalid_argument("UMT5 batch has no active token");

    std::vector<int32_t> bucket_host(length_size * length_size);
    for (int query = 0; query < length; ++query) {
      for (int key = 0; key < length; ++key) {
        bucket_host[static_cast<size_t>(query) * length + key] =
            umt5_relative_position_bucket(key - query);
      }
    }

    const size_t model_count = length_size * kModel;
    const size_t ff_count = length_size * kFf;
    const size_t score_count = length_size * length_size * kHeads;
    DeviceBuffer ids(checked_bytes(length_size, sizeof(int32_t)), tracker_);
    DeviceBuffer mask(checked_bytes(length_size, sizeof(int32_t)), tracker_);
    DeviceBuffer buckets(checked_bytes(length_size * length_size, sizeof(int32_t)), tracker_);
    DeviceBuffer x(checked_bytes(model_count, sizeof(Bf16)), tracker_);
    DeviceBuffer normed(checked_bytes(model_count, sizeof(Bf16)), tracker_);
    DeviceBuffer q(checked_bytes(model_count, sizeof(Bf16)), tracker_);
    DeviceBuffer k(checked_bytes(model_count, sizeof(Bf16)), tracker_);
    DeviceBuffer v(checked_bytes(model_count, sizeof(Bf16)), tracker_);
    DeviceBuffer context(checked_bytes(model_count, sizeof(Bf16)), tracker_);
    DeviceBuffer scores(checked_bytes(score_count, sizeof(Bf16)), tracker_);
    DeviceBuffer probabilities(checked_bytes(score_count, sizeof(float)), tracker_);
    DeviceBuffer wi_0(checked_bytes(ff_count, sizeof(Bf16)), tracker_);
    DeviceBuffer wi_1(checked_bytes(ff_count, sizeof(Bf16)), tracker_);
    DeviceBuffer output(checked_bytes(model_count, sizeof(float)), tracker_);

    UMM_CUDA(cudaMemcpy(ids.data(), tokens.ids.data(), checked_bytes(length_size, 4),
                        cudaMemcpyHostToDevice));
    UMM_CUDA(cudaMemcpy(mask.data(), tokens.attention_mask.data(),
                        checked_bytes(length_size, 4), cudaMemcpyHostToDevice));
    UMM_CUDA(cudaMemcpy(buckets.data(), bucket_host.data(),
                        checked_bytes(length_size * length_size, 4),
                        cudaMemcpyHostToDevice));
    embedding_kernel<<<grid_for(model_count), kThreads>>>(
        ids.as<int32_t>(), embedding_, x.as<Bf16>(), length);
    UMM_CUDA(cudaGetLastError());

    for (const auto& layer : layers_) {
      rms_norm_kernel<<<length, kThreads>>>(x.as<Bf16>(), layer.attention_norm,
                                            normed.as<Bf16>(), length);
      UMM_CUDA(cudaGetLastError());
      linear(context_.get(), layer.q, normed.as<Bf16>(), q.as<Bf16>(), kModel, kModel, length);
      linear(context_.get(), layer.k, normed.as<Bf16>(), k.as<Bf16>(), kModel, kModel, length);
      linear(context_.get(), layer.v, normed.as<Bf16>(), v.as<Bf16>(), kModel, kModel, length);

      const float alpha = 1.0f;
      const float beta = 0.0f;
      UMM_CUBLAS(cublasGemmStridedBatchedEx(
          context_.get(), CUBLAS_OP_T, CUBLAS_OP_N, length, length, kHeadDim,
          &alpha, k.as<Bf16>(), CUDA_R_16BF, kModel, kHeadDim,
          q.as<Bf16>(), CUDA_R_16BF, kModel, kHeadDim,
          &beta, scores.as<Bf16>(), CUDA_R_16BF, length,
          length_size * length_size, kHeads, CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT));
      attention_softmax_kernel<<<kHeads * length, kThreads>>>(
          scores.as<Bf16>(), layer.relative_bias, buckets.as<int32_t>(),
          mask.as<int32_t>(), probabilities.as<float>(), length);
      UMM_CUDA(cudaGetLastError());
      float_to_bf16_kernel<<<grid_for(score_count), kThreads>>>(
          probabilities.as<float>(), scores.as<Bf16>(), score_count);
      UMM_CUDA(cudaGetLastError());
      UMM_CUBLAS(cublasGemmStridedBatchedEx(
          context_.get(), CUBLAS_OP_N, CUBLAS_OP_N, kHeadDim, length, length,
          &alpha, v.as<Bf16>(), CUDA_R_16BF, kModel, kHeadDim,
          scores.as<Bf16>(), CUDA_R_16BF, length, length_size * length_size,
          &beta, context.as<Bf16>(), CUDA_R_16BF, kModel, kHeadDim,
          kHeads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      linear(context_.get(), layer.o, context.as<Bf16>(), q.as<Bf16>(), kModel, kModel, length);
      residual_kernel<<<grid_for(model_count), kThreads>>>(
          q.as<Bf16>(), x.as<Bf16>(), x.as<Bf16>(), model_count);
      UMM_CUDA(cudaGetLastError());

      rms_norm_kernel<<<length, kThreads>>>(x.as<Bf16>(), layer.ffn_norm,
                                            normed.as<Bf16>(), length);
      UMM_CUDA(cudaGetLastError());
      linear(context_.get(), layer.wi_0, normed.as<Bf16>(), wi_0.as<Bf16>(), kFf, kModel, length);
      linear(context_.get(), layer.wi_1, normed.as<Bf16>(), wi_1.as<Bf16>(), kFf, kModel, length);
      gated_gelu_kernel<<<grid_for(ff_count), kThreads>>>(
          wi_1.as<Bf16>(), wi_0.as<Bf16>(), ff_count);
      UMM_CUDA(cudaGetLastError());
      linear(context_.get(), layer.wo, wi_0.as<Bf16>(), q.as<Bf16>(), kModel, kFf, length);
      residual_kernel<<<grid_for(model_count), kThreads>>>(
          q.as<Bf16>(), x.as<Bf16>(), x.as<Bf16>(), model_count);
      UMM_CUDA(cudaGetLastError());
    }
    rms_norm_kernel<<<length, kThreads>>>(x.as<Bf16>(), final_norm_, x.as<Bf16>(), length);
    UMM_CUDA(cudaGetLastError());
    output_kernel<<<grid_for(model_count), kThreads>>>(
        x.as<Bf16>(), mask.as<int32_t>(), output.as<float>(), length);
    UMM_CUDA(cudaGetLastError());

    Umt5EncoderOutput result;
    result.shape = {length, kModel};
    result.hidden_states.resize(model_count);
    UMM_CUDA(cudaMemcpy(result.hidden_states.data(), output.data(),
                        checked_bytes(model_count, sizeof(float)),
                        cudaMemcpyDeviceToHost));
    UMM_CUDA(cudaDeviceSynchronize());
    result.encode_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    result.peak_native_device_bytes = tracker_.peak;
    return result;
  }

  CudaContext context_;
  AllocationTracker tracker_;
  DeviceBuffer weights_;
  const Bf16* embedding_ = nullptr;
  const Bf16* final_norm_ = nullptr;
  std::vector<LayerWeights> layers_;
  double load_ms_ = 0.0;
};

Umt5Encoder::Umt5Encoder(const std::filesystem::path& directory, int gpu)
    : impl_(std::make_unique<Impl>(directory, gpu)) {}
Umt5Encoder::~Umt5Encoder() = default;
Umt5EncoderOutput Umt5Encoder::encode(const Umt5TokenBatch& tokens) {
  return impl_->encode(tokens);
}
double Umt5Encoder::load_ms() const { return impl_->load_ms_; }

}  // namespace umminfer::video
