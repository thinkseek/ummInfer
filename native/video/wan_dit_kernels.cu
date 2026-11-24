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
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
constexpr int kQueryTile = 64;

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
    UMM_DIT_CUDA(cudaSetDevice(gpu));
    UMM_DIT_CUBLAS(cublasCreate(&handle_));
    try {
      UMM_DIT_CUBLAS(cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH));
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

__device__ float to_float(Bf16 value) { return __bfloat162float(value); }
__device__ Bf16 to_bf16(float value) { return __float2bfloat16_rn(value); }

__global__ void float_to_bf16_kernel(const float* input, Bf16* output, size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) output[i] = to_bf16(input[i]);
}

__global__ void bias_bf16_kernel(Bf16* output, const float* bias,
                                 int output_features, size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    const int channel = static_cast<int>(i % output_features);
    output[i] = to_bf16(to_float(output[i]) + bias[channel]);
  }
}

__global__ void bias_f32_kernel(float* output, const float* bias,
                                int output_features, size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) output[i] += bias[i % output_features];
}

__global__ void silu_f32_kernel(float* values, size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) values[i] = values[i] / (1.0f + expf(-values[i]));
}

__global__ void silu_bf16_kernel(Bf16* values, size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    const float x = to_float(values[i]);
    values[i] = to_bf16(x / (1.0f + expf(-x)));
  }
}

__global__ void gelu_tanh_kernel(Bf16* values, size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    const float x = to_float(values[i]);
    const float cube = x * x * x;
    const float y = 0.5f * x *
        (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * cube)));
    values[i] = to_bf16(y);
  }
}

__device__ void reduce_sum(float* reduction) {
  for (int stride = kThreads / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }
}

__device__ void reduce_max(float* reduction) {
  for (int stride = kThreads / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
    }
    __syncthreads();
  }
}

// One block per token. scale/shift are optional AdaLN vectors; gamma/beta are
// optional LayerNorm affine parameters. Inputs and outputs have BF16 storage.
__global__ void layer_norm_kernel(const Bf16* input, Bf16* output,
                                  const float* gamma, const float* beta,
                                  const float* shift, const float* scale,
                                  int tokens, float eps) {
  const int token = blockIdx.x;
  if (token >= tokens) return;
  const size_t base = static_cast<size_t>(token) * kDim;
  __shared__ float reduction[kThreads];
  float local = 0.0f;
  for (int d = threadIdx.x; d < kDim; d += kThreads) {
    local += to_float(input[base + d]);
  }
  reduction[threadIdx.x] = local;
  __syncthreads();
  reduce_sum(reduction);
  const float mean = reduction[0] / kDim;
  __syncthreads();
  local = 0.0f;
  for (int d = threadIdx.x; d < kDim; d += kThreads) {
    const float centered = to_float(input[base + d]) - mean;
    local += centered * centered;
  }
  reduction[threadIdx.x] = local;
  __syncthreads();
  reduce_sum(reduction);
  const float inverse = rsqrtf(reduction[0] / kDim + eps);
  for (int d = threadIdx.x; d < kDim; d += kThreads) {
    float value = (to_float(input[base + d]) - mean) * inverse;
    if (gamma != nullptr) value = value * gamma[d] + beta[d];
    if (shift != nullptr) value = value * (1.0f + scale[d]) + shift[d];
    output[base + d] = to_bf16(value);
  }
}

// qk_norm="rms_norm_across_heads": one RMS reduction over all 3072 channels.
__global__ void rms_norm_kernel(const Bf16* input, const float* weight,
                                Bf16* output, int tokens, float eps) {
  const int token = blockIdx.x;
  if (token >= tokens) return;
  const size_t base = static_cast<size_t>(token) * kDim;
  __shared__ float reduction[kThreads];
  float local = 0.0f;
  for (int d = threadIdx.x; d < kDim; d += kThreads) {
    const float value = to_float(input[base + d]);
    local += value * value;
  }
  reduction[threadIdx.x] = local;
  __syncthreads();
  reduce_sum(reduction);
  const float inverse = rsqrtf(reduction[0] / kDim + to_float(to_bf16(eps)));
  for (int d = threadIdx.x; d < kDim; d += kThreads) {
    const Bf16 normalized = to_bf16(to_float(input[base + d]) * inverse);
    output[base + d] = to_bf16(to_float(normalized) * weight[d]);
  }
}

__global__ void rope_kernel(Bf16* values, const float* cosine,
                            const float* sine, int tokens) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t pairs = static_cast<size_t>(tokens) * kHeads * (kHeadDim / 2);
  if (i >= pairs) return;
  const int pair = static_cast<int>(i % (kHeadDim / 2));
  const int head = static_cast<int>((i / (kHeadDim / 2)) % kHeads);
  const int token = static_cast<int>(i / (kHeads * (kHeadDim / 2)));
  const size_t index = (static_cast<size_t>(token) * kHeads + head) * kHeadDim + pair * 2;
  const size_t rope = static_cast<size_t>(token) * kHeadDim + pair * 2;
  const float even = to_float(values[index]);
  const float odd = to_float(values[index + 1]);
  const float c = cosine[rope];
  const float s = sine[rope + 1];
  values[index] = to_bf16(__fsub_rn(__fmul_rn(even, c), __fmul_rn(odd, s)));
  values[index + 1] = to_bf16(__fadd_rn(__fmul_rn(even, s), __fmul_rn(odd, c)));
}

// scores are column-major [key, query] within each head. Output probabilities
// have the same layout and are normalized in FP32 before BF16 tensor-core PV.
__global__ void attention_softmax_kernel(const float* scores, Bf16* probabilities,
                                         int keys, int queries) {
  const int row = blockIdx.x;
  const int query = row % queries;
  const int head = row / queries;
  if (head >= kHeads) return;
  const size_t base = (static_cast<size_t>(head) * queries + query) * keys;
  const float attention_scale = rsqrtf(static_cast<float>(kHeadDim));
  __shared__ float reduction[kThreads];
  float local = -CUDART_INF_F;
  for (int key = threadIdx.x; key < keys; key += kThreads) {
    local = fmaxf(local, scores[base + key] * attention_scale);
  }
  reduction[threadIdx.x] = local;
  __syncthreads();
  reduce_max(reduction);
  const float maximum = reduction[0];
  __syncthreads();
  local = 0.0f;
  for (int key = threadIdx.x; key < keys; key += kThreads) {
    const float weight = expf(scores[base + key] * attention_scale - maximum);
    local += weight;
  }
  reduction[threadIdx.x] = local;
  __syncthreads();
  reduce_sum(reduction);
  const float inverse = 1.0f / reduction[0];
  for (int key = threadIdx.x; key < keys; key += kThreads) {
    const float weight = expf(scores[base + key] * attention_scale - maximum);
    probabilities[base + key] = to_bf16(weight * inverse);
  }
}

__global__ void attention_output_kernel(const float* tiles, Bf16* output,
                                        int query_begin, int queries) {
  const size_t count = static_cast<size_t>(queries) * kDim;
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const int query = static_cast<int>(i / kDim);
  const int channel = static_cast<int>(i % kDim);
  const int head = channel / kHeadDim;
  const int d = channel % kHeadDim;
  const size_t source = (static_cast<size_t>(head) * queries + query) * kHeadDim + d;
  output[static_cast<size_t>(query_begin + query) * kDim + channel] = to_bf16(tiles[source]);
}

__global__ void residual_kernel(Bf16* hidden, const Bf16* update,
                                const float* gate, size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    const float multiplier = gate == nullptr ? 1.0f : gate[i % kDim];
    hidden[i] = to_bf16(to_float(hidden[i]) + to_float(update[i]) * multiplier);
  }
}

__global__ void modulation_kernel(const float* table, const Bf16* temb,
                                  float* output, int rows) {
  const size_t count = static_cast<size_t>(rows) * kDim;
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) output[i] = table[i] + to_float(temb[i]);
}

__global__ void output_modulation_kernel(const float* table, const Bf16* temb,
                                         float* output) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < 2 * kDim) output[i] = table[i] + to_float(temb[i % kDim]);
}

}  // namespace

struct WanDit::Impl {
  Impl(const std::filesystem::path& model_dir, int gpu) : context_(gpu) {
    config_ = load_wan_dit_config(model_dir.string());
    const auto started = std::chrono::steady_clock::now();
    load_weights(model_dir / "transformer");
    load_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
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
    UMM_DIT_CUDA(cudaDeviceSynchronize());
  }

  const DeviceTensor& tensor(const std::string& name) const { return tensors_.at(name); }

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
      bias_f32_kernel<<<grid_for(count), kThreads>>>(
          static_cast<float*>(output), static_cast<const float*>(bias.data),
          output_features, count);
    } else {
      UMM_DIT_CUBLAS(cublasGemmEx(
          context_.get(), CUBLAS_OP_T, CUBLAS_OP_N, output_features, rows,
          input_features, &alpha, weight.data, CUDA_R_16BF, input_features,
          input, CUDA_R_16BF, input_features, &beta, output, CUDA_R_16BF,
          output_features, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      bias_bf16_kernel<<<grid_for(count), kThreads>>>(
          static_cast<Bf16*>(output), static_cast<const float*>(bias.data),
          output_features, count);
    }
    UMM_DIT_CUDA(cudaGetLastError());
  }

  void attention(const Bf16* query, const Bf16* key, const Bf16* value,
                 int tokens, int context_tokens, Bf16* output,
                 DeviceBuffer& scores, DeviceBuffer& probabilities,
                 DeviceBuffer& output_tiles) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    for (int begin = 0; begin < tokens; begin += kQueryTile) {
      const int queries = std::min(kQueryTile, tokens - begin);
      UMM_DIT_CUBLAS(cublasGemmStridedBatchedEx(
          context_.get(), CUBLAS_OP_T, CUBLAS_OP_N, context_tokens, queries,
          kHeadDim, &alpha, key, CUDA_R_16BF, kDim, kHeadDim,
          query + static_cast<size_t>(begin) * kDim, CUDA_R_16BF, kDim, kHeadDim,
          &beta, scores.as<float>(), CUDA_R_32F, context_tokens,
          static_cast<int64_t>(context_tokens) * queries, kHeads,
          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      attention_softmax_kernel<<<kHeads * queries, kThreads>>>(
          scores.as<float>(), probabilities.as<Bf16>(), context_tokens, queries);
      UMM_DIT_CUDA(cudaGetLastError());
      UMM_DIT_CUBLAS(cublasGemmStridedBatchedEx(
          context_.get(), CUBLAS_OP_N, CUBLAS_OP_N, kHeadDim, queries,
          context_tokens, &alpha, value, CUDA_R_16BF, kDim, kHeadDim,
          probabilities.as<Bf16>(), CUDA_R_16BF, context_tokens,
          static_cast<int64_t>(context_tokens) * queries, &beta,
          output_tiles.as<float>(), CUDA_R_32F, kHeadDim,
          kHeadDim * queries, kHeads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      attention_output_kernel<<<grid_for(static_cast<size_t>(queries) * kDim), kThreads>>>(
          output_tiles.as<float>(), output, begin, queries);
      UMM_DIT_CUDA(cudaGetLastError());
    }
  }

  WanDitOutput forward(const float* latent, int frames, int height, int width,
                       float timestep, const float* context, int text_len) {
    if (latent == nullptr || context == nullptr || frames <= 0 || height <= 0 ||
        width <= 0 || height % 2 != 0 || width % 2 != 0 || text_len <= 0 ||
        text_len > 512 || !std::isfinite(timestep)) {
      throw std::invalid_argument("invalid Wan DiT forward inputs");
    }
    const auto started = std::chrono::steady_clock::now();
    tracker_.peak = tracker_.current;
    UMM_DIT_CUDA(cudaSetDevice(gpu_));
    const int tokens = frames * (height / 2) * (width / 2);
    const size_t model_count = static_cast<size_t>(tokens) * kDim;
    const size_t kv_count = static_cast<size_t>(std::max(tokens, text_len)) * kDim;
    const size_t text_count = static_cast<size_t>(text_len) * kDim;
    const size_t latent_count = static_cast<size_t>(kChannels) * frames * height * width;
    const int attention_keys = std::max(tokens, text_len);

    const auto time_embedding = wan_sinusoidal_timestep_embedding(timestep, kFreqDim);
    const auto rope = wan_build_rope_3d(frames, height, width, kHeadDim);
    const auto patch = wan_patchify_host(latent, kChannels, frames, height, width);

    DeviceBuffer time_input(checked_bytes(kFreqDim, 4), tracker_);
    DeviceBuffer time_a(checked_bytes(kDim, 4), tracker_);
    DeviceBuffer time_b(checked_bytes(kDim, 4), tracker_);
    DeviceBuffer temb(checked_bytes(kDim, 2), tracker_);
    DeviceBuffer time_proj(checked_bytes(6 * kDim, 2), tracker_);
    DeviceBuffer modulation(checked_bytes(6 * kDim, 4), tracker_);
    DeviceBuffer text_input(checked_bytes(static_cast<size_t>(text_len) * kTextDim, 2), tracker_);
    DeviceBuffer text_a(checked_bytes(text_count, 2), tracker_);
    DeviceBuffer text_projected(checked_bytes(text_count, 2), tracker_);
    DeviceBuffer rope_cos(checked_bytes(rope.cos.size(), 4), tracker_);
    DeviceBuffer rope_sin(checked_bytes(rope.sin.size(), 4), tracker_);
    DeviceBuffer patch_tokens(checked_bytes(patch.size(), 2), tracker_);
    DeviceBuffer x(checked_bytes(model_count, 2), tracker_);
    DeviceBuffer normed(checked_bytes(model_count, 2), tracker_);
    DeviceBuffer q(checked_bytes(model_count, 2), tracker_);
    DeviceBuffer k(checked_bytes(kv_count, 2), tracker_);
    DeviceBuffer v(checked_bytes(kv_count, 2), tracker_);
    DeviceBuffer attended(checked_bytes(model_count, 2), tracker_);
    DeviceBuffer ff(checked_bytes(static_cast<size_t>(tokens) * kFf, 2), tracker_);
    DeviceBuffer scores(checked_bytes(static_cast<size_t>(kHeads) * kQueryTile * attention_keys, 4), tracker_);
    DeviceBuffer probabilities(checked_bytes(static_cast<size_t>(kHeads) * kQueryTile * attention_keys, 2), tracker_);
    DeviceBuffer output_tiles(checked_bytes(static_cast<size_t>(kQueryTile) * kDim, 4), tracker_);

    UMM_DIT_CUDA(cudaMemcpy(time_input.data(), time_embedding.data(),
                            checked_bytes(kFreqDim, 4), cudaMemcpyHostToDevice));
    linear("condition_embedder.time_embedder.linear_1", kFreqDim, kDim, 1,
           time_input.data(), time_a.data());
    silu_f32_kernel<<<grid_for(kDim), kThreads>>>(time_a.as<float>(), kDim);
    UMM_DIT_CUDA(cudaGetLastError());
    linear("condition_embedder.time_embedder.linear_2", kDim, kDim, 1,
           time_a.data(), time_b.data());
    float_to_bf16_kernel<<<grid_for(kDim), kThreads>>>(time_b.as<float>(), temb.as<Bf16>(), kDim);
    UMM_DIT_CUDA(cudaGetLastError());
    silu_bf16_kernel<<<grid_for(kDim), kThreads>>>(temb.as<Bf16>(), kDim);
    UMM_DIT_CUDA(cudaGetLastError());
    linear("condition_embedder.time_proj", kDim, 6 * kDim, 1, temb.data(), time_proj.data());
    // The output norm uses the unactivated time embedding, so restore temb after time_proj.
    float_to_bf16_kernel<<<grid_for(kDim), kThreads>>>(time_b.as<float>(), temb.as<Bf16>(), kDim);
    UMM_DIT_CUDA(cudaGetLastError());

    {
      std::vector<float> context_f32(context, context + static_cast<size_t>(text_len) * kTextDim);
      DeviceBuffer context_fp32(checked_bytes(context_f32.size(), 4), tracker_);
      UMM_DIT_CUDA(cudaMemcpy(context_fp32.data(), context_f32.data(),
                              checked_bytes(context_f32.size(), 4), cudaMemcpyHostToDevice));
      float_to_bf16_kernel<<<grid_for(context_f32.size()), kThreads>>>(
          context_fp32.as<float>(), text_input.as<Bf16>(), context_f32.size());
      UMM_DIT_CUDA(cudaGetLastError());
    }
    linear("condition_embedder.text_embedder.linear_1", kTextDim, kDim, text_len,
           text_input.data(), text_a.data());
    gelu_tanh_kernel<<<grid_for(text_count), kThreads>>>(text_a.as<Bf16>(), text_count);
    UMM_DIT_CUDA(cudaGetLastError());
    linear("condition_embedder.text_embedder.linear_2", kDim, kDim, text_len,
           text_a.data(), text_projected.data());

    UMM_DIT_CUDA(cudaMemcpy(rope_cos.data(), rope.cos.data(), checked_bytes(rope.cos.size(), 4),
                            cudaMemcpyHostToDevice));
    UMM_DIT_CUDA(cudaMemcpy(rope_sin.data(), rope.sin.data(), checked_bytes(rope.sin.size(), 4),
                            cudaMemcpyHostToDevice));
    {
      DeviceBuffer patch_fp32(checked_bytes(patch.size(), 4), tracker_);
      UMM_DIT_CUDA(cudaMemcpy(patch_fp32.data(), patch.data(), checked_bytes(patch.size(), 4),
                              cudaMemcpyHostToDevice));
      float_to_bf16_kernel<<<grid_for(patch.size()), kThreads>>>(
          patch_fp32.as<float>(), patch_tokens.as<Bf16>(), patch.size());
      UMM_DIT_CUDA(cudaGetLastError());
    }
    linear("patch_embedding", kChannels * kPatchVolume, kDim, tokens,
           patch_tokens.data(), x.data());

    for (int layer = 0; layer < kLayers; ++layer) {
      const std::string prefix = "blocks." + std::to_string(layer) + ".";
      const auto& table = tensor(prefix + "scale_shift_table");
      modulation_kernel<<<grid_for(6 * kDim), kThreads>>>(
          static_cast<const float*>(table.data), time_proj.as<Bf16>(),
          modulation.as<float>(), 6);
      UMM_DIT_CUDA(cudaGetLastError());
      const float* mod = modulation.as<float>();
      layer_norm_kernel<<<tokens, kThreads>>>(
          x.as<Bf16>(), normed.as<Bf16>(), nullptr, nullptr, mod, mod + kDim,
          tokens, config_.eps);
      UMM_DIT_CUDA(cudaGetLastError());
      linear(prefix + "attn1.to_q", kDim, kDim, tokens, normed.data(), q.data());
      linear(prefix + "attn1.to_k", kDim, kDim, tokens, normed.data(), k.data());
      linear(prefix + "attn1.to_v", kDim, kDim, tokens, normed.data(), v.data());
      rms_norm_kernel<<<tokens, kThreads>>>(
          q.as<Bf16>(), static_cast<const float*>(tensor(prefix + "attn1.norm_q.weight").data),
          q.as<Bf16>(), tokens, config_.eps);
      rms_norm_kernel<<<tokens, kThreads>>>(
          k.as<Bf16>(), static_cast<const float*>(tensor(prefix + "attn1.norm_k.weight").data),
          k.as<Bf16>(), tokens, config_.eps);
      UMM_DIT_CUDA(cudaGetLastError());
      rope_kernel<<<grid_for(static_cast<size_t>(tokens) * kDim / 2), kThreads>>>(
          q.as<Bf16>(), rope_cos.as<float>(), rope_sin.as<float>(), tokens);
      rope_kernel<<<grid_for(static_cast<size_t>(tokens) * kDim / 2), kThreads>>>(
          k.as<Bf16>(), rope_cos.as<float>(), rope_sin.as<float>(), tokens);
      UMM_DIT_CUDA(cudaGetLastError());
      attention(q.as<Bf16>(), k.as<Bf16>(), v.as<Bf16>(), tokens, tokens,
                attended.as<Bf16>(), scores, probabilities, output_tiles);
      linear(prefix + "attn1.to_out.0", kDim, kDim, tokens, attended.data(), q.data());
      residual_kernel<<<grid_for(model_count), kThreads>>>(
          x.as<Bf16>(), q.as<Bf16>(), mod + 2 * kDim, model_count);
      UMM_DIT_CUDA(cudaGetLastError());

      layer_norm_kernel<<<tokens, kThreads>>>(
          x.as<Bf16>(), normed.as<Bf16>(),
          static_cast<const float*>(tensor(prefix + "norm2.weight").data),
          static_cast<const float*>(tensor(prefix + "norm2.bias").data),
          nullptr, nullptr, tokens, config_.eps);
      UMM_DIT_CUDA(cudaGetLastError());
      linear(prefix + "attn2.to_q", kDim, kDim, tokens, normed.data(), q.data());
      linear(prefix + "attn2.to_k", kDim, kDim, text_len, text_projected.data(), k.data());
      linear(prefix + "attn2.to_v", kDim, kDim, text_len, text_projected.data(), v.data());
      rms_norm_kernel<<<tokens, kThreads>>>(
          q.as<Bf16>(), static_cast<const float*>(tensor(prefix + "attn2.norm_q.weight").data),
          q.as<Bf16>(), tokens, config_.eps);
      rms_norm_kernel<<<text_len, kThreads>>>(
          k.as<Bf16>(), static_cast<const float*>(tensor(prefix + "attn2.norm_k.weight").data),
          k.as<Bf16>(), text_len, config_.eps);
      UMM_DIT_CUDA(cudaGetLastError());
      attention(q.as<Bf16>(), k.as<Bf16>(), v.as<Bf16>(), tokens, text_len,
                attended.as<Bf16>(), scores, probabilities, output_tiles);
      linear(prefix + "attn2.to_out.0", kDim, kDim, tokens, attended.data(), q.data());
      residual_kernel<<<grid_for(model_count), kThreads>>>(
          x.as<Bf16>(), q.as<Bf16>(), nullptr, model_count);
      UMM_DIT_CUDA(cudaGetLastError());

      layer_norm_kernel<<<tokens, kThreads>>>(
          x.as<Bf16>(), normed.as<Bf16>(), nullptr, nullptr,
          mod + 3 * kDim, mod + 4 * kDim, tokens, config_.eps);
      UMM_DIT_CUDA(cudaGetLastError());
      linear(prefix + "ffn.net.0.proj", kDim, kFf, tokens, normed.data(), ff.data());
      gelu_tanh_kernel<<<grid_for(static_cast<size_t>(tokens) * kFf), kThreads>>>(
          ff.as<Bf16>(), static_cast<size_t>(tokens) * kFf);
      UMM_DIT_CUDA(cudaGetLastError());
      linear(prefix + "ffn.net.2", kFf, kDim, tokens, ff.data(), q.data());
      residual_kernel<<<grid_for(model_count), kThreads>>>(
          x.as<Bf16>(), q.as<Bf16>(), mod + 5 * kDim, model_count);
      UMM_DIT_CUDA(cudaGetLastError());
    }

    output_modulation_kernel<<<grid_for(2 * kDim), kThreads>>>(
        static_cast<const float*>(tensor("scale_shift_table").data), temb.as<Bf16>(),
        modulation.as<float>());
    UMM_DIT_CUDA(cudaGetLastError());
    layer_norm_kernel<<<tokens, kThreads>>>(
        x.as<Bf16>(), normed.as<Bf16>(), nullptr, nullptr,
        modulation.as<float>(), modulation.as<float>() + kDim, tokens, config_.eps);
    UMM_DIT_CUDA(cudaGetLastError());
    linear("proj_out", kDim, kProjection, tokens, normed.data(), q.data());

    std::vector<uint16_t> projected_bits(static_cast<size_t>(tokens) * kProjection);
    UMM_DIT_CUDA(cudaMemcpy(projected_bits.data(), q.data(),
                            checked_bytes(projected_bits.size(), 2), cudaMemcpyDeviceToHost));
    std::vector<float> projected(projected_bits.size());
    for (size_t i = 0; i < projected.size(); ++i) projected[i] = bf16_bits_to_f32(projected_bits[i]);

    WanDitOutput result;
    result.velocity = wan_unpatchify_host(projected.data(), kChannels, frames, height, width);
    if (result.velocity.size() != latent_count) throw std::logic_error("Wan DiT output size mismatch");
    UMM_DIT_CUDA(cudaDeviceSynchronize());
    result.stats.load_ms = load_ms_;
    result.stats.forward_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    result.stats.peak_vram_bytes = tracker_.peak;
    result.stats.tensor_count = tensors_.size();
    return result;
  }

  CudaContext context_;
  int gpu_ = 0;
  WanDitConfig config_;
  AllocationTracker tracker_;
  DeviceBuffer bf16_weights_;
  DeviceBuffer f32_weights_;
  std::unordered_map<std::string, DeviceTensor> tensors_;
  double load_ms_ = 0.0;
};

WanDit::WanDit(const std::filesystem::path& model_dir, int gpu)
    : impl_(std::make_unique<Impl>(model_dir, gpu)) {
  impl_->gpu_ = gpu;
}
WanDit::~WanDit() = default;
WanDitOutput WanDit::forward(const float* latent, int frames, int height,
                             int width, float timestep, const float* context,
                             int text_len) {
  return impl_->forward(latent, frames, height, width, timestep, context, text_len);
}

}  // namespace umminfer::video

#endif  // UMM_VIDEO_CUDA
