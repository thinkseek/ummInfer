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
constexpr int kQueryTile = kWanDitQueryTile;
constexpr size_t kCrossKvCacheSlots = 2;

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


// Wan2.2-TI2V-5B expand_timesteps conditioning: tokens below
// `first_frame_tokens` belong to latent frame 0 and carry timestep 0, all other
// tokens carry the current step timestep. Two embedding groups are therefore
// enough, and these kernels select the group per token exactly like Diffusers'
// per-token `scale_shift_table + temb` broadcast.
__global__ void modulation_pair_kernel(const float* table,
                                       const Bf16* time_proj, float* output,
                                       int group_stride) {
  const size_t count = static_cast<size_t>(2) * group_stride;
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  output[i] = table[i % group_stride] + to_float(time_proj[i]);
}

__global__ void output_modulation_pair_kernel(const float* table,
                                              const Bf16* temb, float* output) {
  const size_t count = static_cast<size_t>(2) * 2 * kDim;
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const size_t group = i / (2 * kDim);
  const size_t offset = i % (2 * kDim);
  output[i] = table[offset] + to_float(temb[group * kDim + offset % kDim]);
}

__global__ void layer_norm_pair_kernel(const Bf16* input, Bf16* output,
                                       const float* gamma, const float* beta,
                                       const float* modulation, int rows,
                                       int scale_row, int shift_row, int tokens,
                                       float eps, int first_frame_tokens) {
  const int token = blockIdx.x;
  if (token >= tokens) return;
  const int group = token < first_frame_tokens ? 0 : 1;
  const float* mod = modulation + static_cast<size_t>(group) * rows * kDim;
  const float* scale = mod + scale_row * kDim;
  const float* shift = mod + shift_row * kDim;
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
    value = value * (1.0f + scale[d]) + shift[d];
    output[base + d] = to_bf16(value);
  }
}

__global__ void residual_pair_kernel(Bf16* hidden, const Bf16* update,
                                     const float* modulation, int rows,
                                     int gate_row, size_t count,
                                     int first_frame_tokens) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const int token = static_cast<int>(i / kDim);
  const int group = token < first_frame_tokens ? 0 : 1;
  const float multiplier =
      modulation[static_cast<size_t>(group) * rows * kDim + gate_row * kDim +
                 (i % kDim)];
  hidden[i] = to_bf16(to_float(hidden[i]) + to_float(update[i]) * multiplier);
}

// TeaCache residual kernels:
//   - sub_residual_bf16_kernel: residual_inout holds x_in before the 30 blocks;
//     in-place computes residual = x_after - x_in.
//   - add_residual_bf16_kernel: on a TeaCache hit, skips all 30 blocks and
//     adds x_inout = x_in + prev_residual.
__global__ void sub_residual_bf16_kernel(Bf16* residual_inout,
                                         const Bf16* x_after, size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    residual_inout[i] =
        to_bf16(to_float(x_after[i]) - to_float(residual_inout[i]));
  }
}

__global__ void add_residual_bf16_kernel(Bf16* x_inout, const Bf16* residual,
                                         size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    x_inout[i] = to_bf16(to_float(x_inout[i]) + to_float(residual[i]));
  }
}

}  // namespace

struct WanDit::Impl {
  struct WorkspaceBuffers {
    int frames = 0;
    int height = 0;
    int width = 0;
    int text_len = 0;

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

  Impl(const std::filesystem::path& model_dir, int gpu)
      : context_(gpu), gpu_(gpu) {
    config_ = load_wan_dit_config(model_dir.string());
    const auto started = std::chrono::steady_clock::now();
    load_weights(model_dir / "transformer");
    load_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
  }

  ~Impl() {
    (void)cudaSetDevice(gpu_);
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
      const int queries = wan_attention_tile_len(tokens, begin, kQueryTile);
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
          static_cast<int64_t>(kHeadDim) * queries, kHeads,
          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      attention_output_kernel<<<grid_for(static_cast<size_t>(queries) * kDim), kThreads>>>(
          output_tiles.as<float>(), output, begin, queries);
      UMM_DIT_CUDA(cudaGetLastError());
    }
  }

  void ensure_workspace(int frames, int height, int width, int text_len) {
    if (ws_.frames == frames && ws_.height == height && ws_.width == width &&
        ws_.text_len == text_len) {
      return;
    }
    ws_.frames = 0;
    ws_.height = 0;
    ws_.width = 0;
    ws_.text_len = 0;

    constexpr size_t kMaxTimeRows = 2;
    const int tokens = frames * (height / 2) * (width / 2);
    const size_t model_count = static_cast<size_t>(tokens) * kDim;
    const size_t text_count = static_cast<size_t>(text_len) * kDim;
    const size_t context_count = static_cast<size_t>(text_len) * kTextDim;
    const size_t patch_count = static_cast<size_t>(tokens) * kProjection;
    const int query_tile = std::min(kQueryTile, tokens);
    const int attention_keys = std::max(tokens, text_len);

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
    ws_.ff.resize(checked_bytes(static_cast<size_t>(tokens) * kFf, sizeof(Bf16)), tracker_);
    ws_.scores.resize(
        checked_bytes(static_cast<size_t>(kHeads) * query_tile * attention_keys, sizeof(float)),
        tracker_);
    ws_.probabilities.resize(
        checked_bytes(static_cast<size_t>(kHeads) * query_tile * attention_keys, sizeof(Bf16)),
        tracker_);
    ws_.output_tiles.resize(
        checked_bytes(static_cast<size_t>(query_tile) * kDim, sizeof(float)), tracker_);

    ws_.frames = frames;
    ws_.height = height;
    ws_.width = width;
    ws_.text_len = text_len;
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
    UMM_DIT_CUDA(cudaMemcpy(rope_.cos.data(), rope.cos.data(), cos_bytes,
                            cudaMemcpyHostToDevice));
    UMM_DIT_CUDA(cudaMemcpy(rope_.sin.data(), rope.sin.data(), sin_bytes,
                            cudaMemcpyHostToDevice));

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

    UMM_DIT_CUDA(cudaMemcpy(ws_.context_fp32.data(), context,
                            checked_bytes(context_count, sizeof(float)),
                            cudaMemcpyHostToDevice));
    float_to_bf16_kernel<<<grid_for(context_count), kThreads>>>(
        ws_.context_fp32.as<float>(), ws_.text_input.as<Bf16>(), context_count);
    UMM_DIT_CUDA(cudaGetLastError());
    linear("condition_embedder.text_embedder.linear_1", kTextDim, kDim, text_len,
           ws_.text_input.data(), ws_.text_a.data());
    gelu_tanh_kernel<<<grid_for(text_count), kThreads>>>(ws_.text_a.as<Bf16>(), text_count);
    UMM_DIT_CUDA(cudaGetLastError());
    linear("condition_embedder.text_embedder.linear_2", kDim, kDim, text_len,
           ws_.text_a.data(), ws_.text_projected.data());

    for (int layer = 0; layer < kLayers; ++layer) {
      const std::string prefix = "blocks." + std::to_string(layer) + ".";
      Bf16* layer_k = entry.k_all.as<Bf16>() + static_cast<size_t>(layer) * text_count;
      Bf16* layer_v = entry.v_all.as<Bf16>() + static_cast<size_t>(layer) * text_count;
      linear(prefix + "attn2.to_k", kDim, kDim, text_len, ws_.text_projected.data(), layer_k);
      linear(prefix + "attn2.to_v", kDim, kDim, text_len, ws_.text_projected.data(), layer_v);
      rms_norm_kernel<<<text_len, kThreads>>>(
          layer_k, static_cast<const float*>(tensor(prefix + "attn2.norm_k.weight").data),
          layer_k, text_len, config_.eps);
      UMM_DIT_CUDA(cudaGetLastError());
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
    const int64_t tokens64 =
        static_cast<int64_t>(frames) * (height / 2) * (width / 2);
    if (tokens64 <= 0 || tokens64 > std::numeric_limits<int>::max()) {
      throw std::overflow_error("Wan DiT token count overflows int");
    }
    const int tokens = static_cast<int>(tokens64);
    // Wan2.2-TI2V-5B expand_timesteps: latent frame 0 tokens get t=0.
    const bool per_frame = first_frame_zero_timestep;
    const int temb_rows = per_frame ? 2 : 1;
    const size_t time_rows = static_cast<size_t>(temb_rows);
    const int first_frame_tokens = (height / 2) * (width / 2);
    const size_t model_count = static_cast<size_t>(tokens) * kDim;
    const size_t text_count = static_cast<size_t>(text_len) * kDim;
    const size_t latent_count = static_cast<size_t>(kChannels) * frames * height * width;

    ensure_workspace(frames, height, width, text_len);
    ensure_rope(frames, height, width);

    std::vector<float> time_embedding;
    if (per_frame) {
      const auto zero = wan_sinusoidal_timestep_embedding(0.0f, kFreqDim);
      const auto step = wan_sinusoidal_timestep_embedding(timestep, kFreqDim);
      time_embedding.resize(static_cast<size_t>(2) * kFreqDim);
      std::copy(zero.begin(), zero.end(), time_embedding.begin());
      std::copy(step.begin(), step.end(), time_embedding.begin() + kFreqDim);
    } else {
      time_embedding = wan_sinusoidal_timestep_embedding(timestep, kFreqDim);
    }
    const auto patch = wan_patchify_host(latent, kChannels, frames, height, width);

    UMM_DIT_CUDA(cudaMemcpy(
        ws_.time_input.data(), time_embedding.data(),
        checked_bytes(kFreqDim * time_rows, 4), cudaMemcpyHostToDevice));
    linear("condition_embedder.time_embedder.linear_1", kFreqDim, kDim,
           temb_rows, ws_.time_input.data(), ws_.time_a.data());
    silu_f32_kernel<<<grid_for(kDim * time_rows), kThreads>>>(
        ws_.time_a.as<float>(), kDim * time_rows);
    UMM_DIT_CUDA(cudaGetLastError());
    linear("condition_embedder.time_embedder.linear_2", kDim, kDim,
           temb_rows, ws_.time_a.data(), ws_.time_b.data());
    float_to_bf16_kernel<<<grid_for(kDim * time_rows), kThreads>>>(
        ws_.time_b.as<float>(), ws_.temb.as<Bf16>(), kDim * time_rows);
    UMM_DIT_CUDA(cudaGetLastError());
    if (per_frame) {
      // time_proj consumes the activated embedding; the output norm keeps
      // the unactivated one, so the activated copy gets its own buffer.
      UMM_DIT_CUDA(cudaMemcpy(ws_.temb_act.data(), ws_.temb.data(),
                              checked_bytes(kDim * time_rows, 2),
                              cudaMemcpyDeviceToDevice));
      silu_bf16_kernel<<<grid_for(kDim * time_rows), kThreads>>>(
          ws_.temb_act.as<Bf16>(), kDim * time_rows);
      UMM_DIT_CUDA(cudaGetLastError());
      linear("condition_embedder.time_proj", kDim, 6 * kDim, temb_rows,
             ws_.temb_act.data(), ws_.time_proj.data());
    } else {
      silu_bf16_kernel<<<grid_for(kDim), kThreads>>>(ws_.temb.as<Bf16>(), kDim);
      UMM_DIT_CUDA(cudaGetLastError());
      linear("condition_embedder.time_proj", kDim, 6 * kDim, 1,
             ws_.temb.data(), ws_.time_proj.data());
      // The output norm uses the unactivated time embedding, so restore temb after time_proj.
      float_to_bf16_kernel<<<grid_for(kDim), kThreads>>>(
          ws_.time_b.as<float>(), ws_.temb.as<Bf16>(), kDim);
      UMM_DIT_CUDA(cudaGetLastError());
    }

    // Opt-in TeaCache decision using the active timestep's 6*kDim time_proj
    // signal (36 KiB D2H). When threshold == 0.0f, zero D2H copy or GPU
    // allocation occurs and execution stays 100% bit-exact with uncached DiT.
    TeaCacheDecision tc_decision{};
    if (teacache_.enabled()) {
      constexpr size_t kE0Count = static_cast<size_t>(6) * kDim;
      teacache_e0_bits_.resize(kE0Count);
      teacache_e0_f32_.resize(kE0Count);
      const Bf16* active_e0 =
          ws_.time_proj.as<Bf16>() + static_cast<size_t>(temb_rows - 1) * kE0Count;
      UMM_DIT_CUDA(cudaMemcpy(teacache_e0_bits_.data(), active_e0,
                              checked_bytes(kE0Count, sizeof(Bf16)),
                              cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < kE0Count; ++i) {
        teacache_e0_f32_[i] = bf16_bits_to_f32(teacache_e0_bits_[i]);
      }
      tc_decision = teacache_.evaluate(
          teacache_hint.branch, teacache_hint.step_index,
          teacache_hint.total_steps, frames, height, width, per_frame,
          teacache_e0_f32_.data(), kE0Count);
    } else {
      teacache_.record_disabled_call();
    }

    UMM_DIT_CUDA(cudaMemcpy(ws_.patch_fp32.data(), patch.data(),
                            checked_bytes(patch.size(), 4),
                            cudaMemcpyHostToDevice));
    float_to_bf16_kernel<<<grid_for(patch.size()), kThreads>>>(
        ws_.patch_fp32.as<float>(), ws_.patch_tokens.as<Bf16>(), patch.size());
    UMM_DIT_CUDA(cudaGetLastError());
    linear("patch_embedding", kChannels * kPatchVolume, kDim, tokens,
           ws_.patch_tokens.data(), ws_.x.data());

    const size_t residual_bytes = checked_bytes(model_count, sizeof(Bf16));
    if (tc_decision.use_cached_residual) {
      const DeviceBuffer& cached_residual =
          teacache_residuals_[static_cast<size_t>(teacache_hint.branch)];
      if (cached_residual.bytes() != residual_bytes) {
        throw std::logic_error("Wan DiT TeaCache residual buffer size mismatch");
      }
      add_residual_bf16_kernel<<<grid_for(model_count), kThreads>>>(
          ws_.x.as<Bf16>(), cached_residual.as<Bf16>(), model_count);
      UMM_DIT_CUDA(cudaGetLastError());
    } else {
      // Cache hit skips text projection and all 30 layers' attn2 K/V GEMMs + RMSNorms.
      const CrossAttnKvEntry& cross_kv = get_or_build_cross_kv(context, text_len);

      struct ResidualRollbackGuard {
        TeaCacheController* controller = nullptr;
        int branch = 0;
        bool active = false;
        ~ResidualRollbackGuard() {
          if (active && controller != nullptr) {
            controller->invalidate_branch(branch);
          }
        }
      } rollback_guard{&teacache_, teacache_hint.branch,
                       tc_decision.save_residual};

      DeviceBuffer* residual_slot = nullptr;
      if (tc_decision.save_residual) {
        residual_slot =
            &teacache_residuals_[static_cast<size_t>(teacache_hint.branch)];
        residual_slot->resize(residual_bytes, tracker_);
        UMM_DIT_CUDA(cudaMemcpy(residual_slot->data(), ws_.x.data(),
                                residual_bytes, cudaMemcpyDeviceToDevice));
      }

      // Per-token modulation differs only between latent frame 0 and the
      // remaining frames, so two embedding groups drive the pair kernels.
      const auto apply_norm = [&](const Bf16* input, Bf16* output,
                                  const float* gamma, const float* beta,
                                  int shift_row, int scale_row, int rows) {
        if (per_frame) {
          layer_norm_pair_kernel<<<tokens, kThreads>>>(
              input, output, gamma, beta, ws_.modulation.as<float>(), rows, scale_row,
              shift_row, tokens, config_.eps, first_frame_tokens);
        } else {
          const float* mod = ws_.modulation.as<float>();
          layer_norm_kernel<<<tokens, kThreads>>>(
              input, output, gamma, beta, mod + shift_row * kDim,
              mod + scale_row * kDim, tokens, config_.eps);
        }
        UMM_DIT_CUDA(cudaGetLastError());
      };
      const auto apply_residual = [&](const Bf16* update, int gate_row) {
        if (per_frame) {
          residual_pair_kernel<<<grid_for(model_count), kThreads>>>(
              ws_.x.as<Bf16>(), update, ws_.modulation.as<float>(), 6, gate_row,
              model_count, first_frame_tokens);
        } else {
          residual_kernel<<<grid_for(model_count), kThreads>>>(
              ws_.x.as<Bf16>(), update, ws_.modulation.as<float>() + gate_row * kDim,
              model_count);
        }
        UMM_DIT_CUDA(cudaGetLastError());
      };

      for (int layer = 0; layer < kLayers; ++layer) {
        const std::string prefix = "blocks." + std::to_string(layer) + ".";
        const auto& table = tensor(prefix + "scale_shift_table");
        if (per_frame) {
          modulation_pair_kernel<<<grid_for(2 * 6 * kDim), kThreads>>>(
              static_cast<const float*>(table.data), ws_.time_proj.as<Bf16>(),
              ws_.modulation.as<float>(), 6 * kDim);
        } else {
          modulation_kernel<<<grid_for(6 * kDim), kThreads>>>(
              static_cast<const float*>(table.data), ws_.time_proj.as<Bf16>(),
              ws_.modulation.as<float>(), 6);
        }
        UMM_DIT_CUDA(cudaGetLastError());
        apply_norm(ws_.x.as<Bf16>(), ws_.normed.as<Bf16>(), nullptr, nullptr, 0, 1, 6);
        linear(prefix + "attn1.to_q", kDim, kDim, tokens, ws_.normed.data(), ws_.q.data());
        linear(prefix + "attn1.to_k", kDim, kDim, tokens, ws_.normed.data(), ws_.k.data());
        linear(prefix + "attn1.to_v", kDim, kDim, tokens, ws_.normed.data(), ws_.v.data());
        rms_norm_kernel<<<tokens, kThreads>>>(
            ws_.q.as<Bf16>(), static_cast<const float*>(tensor(prefix + "attn1.norm_q.weight").data),
            ws_.q.as<Bf16>(), tokens, config_.eps);
        rms_norm_kernel<<<tokens, kThreads>>>(
            ws_.k.as<Bf16>(), static_cast<const float*>(tensor(prefix + "attn1.norm_k.weight").data),
            ws_.k.as<Bf16>(), tokens, config_.eps);
        UMM_DIT_CUDA(cudaGetLastError());
        rope_kernel<<<grid_for(static_cast<size_t>(tokens) * kDim / 2), kThreads>>>(
            ws_.q.as<Bf16>(), rope_.cos.as<float>(), rope_.sin.as<float>(), tokens);
        rope_kernel<<<grid_for(static_cast<size_t>(tokens) * kDim / 2), kThreads>>>(
            ws_.k.as<Bf16>(), rope_.cos.as<float>(), rope_.sin.as<float>(), tokens);
        UMM_DIT_CUDA(cudaGetLastError());
        attention(ws_.q.as<Bf16>(), ws_.k.as<Bf16>(), ws_.v.as<Bf16>(), tokens, tokens,
                  ws_.attended.as<Bf16>(), ws_.scores, ws_.probabilities, ws_.output_tiles);
        linear(prefix + "attn1.to_out.0", kDim, kDim, tokens, ws_.attended.data(), ws_.q.data());
        apply_residual(ws_.q.as<Bf16>(), 2);

        layer_norm_kernel<<<tokens, kThreads>>>(
            ws_.x.as<Bf16>(), ws_.normed.as<Bf16>(),
            static_cast<const float*>(tensor(prefix + "norm2.weight").data),
            static_cast<const float*>(tensor(prefix + "norm2.bias").data),
            nullptr, nullptr, tokens, config_.eps);
        UMM_DIT_CUDA(cudaGetLastError());
        linear(prefix + "attn2.to_q", kDim, kDim, tokens, ws_.normed.data(), ws_.q.data());
        rms_norm_kernel<<<tokens, kThreads>>>(
            ws_.q.as<Bf16>(), static_cast<const float*>(tensor(prefix + "attn2.norm_q.weight").data),
            ws_.q.as<Bf16>(), tokens, config_.eps);
        UMM_DIT_CUDA(cudaGetLastError());
        const Bf16* layer_k_cross =
            cross_kv.k_all.as<Bf16>() + static_cast<size_t>(layer) * text_count;
        const Bf16* layer_v_cross =
            cross_kv.v_all.as<Bf16>() + static_cast<size_t>(layer) * text_count;
        attention(ws_.q.as<Bf16>(), layer_k_cross, layer_v_cross, tokens, text_len,
                  ws_.attended.as<Bf16>(), ws_.scores, ws_.probabilities, ws_.output_tiles);
        linear(prefix + "attn2.to_out.0", kDim, kDim, tokens, ws_.attended.data(), ws_.q.data());
        residual_kernel<<<grid_for(model_count), kThreads>>>(
            ws_.x.as<Bf16>(), ws_.q.as<Bf16>(), nullptr, model_count);
        UMM_DIT_CUDA(cudaGetLastError());

        apply_norm(ws_.x.as<Bf16>(), ws_.normed.as<Bf16>(), nullptr, nullptr, 3, 4, 6);
        linear(prefix + "ffn.net.0.proj", kDim, kFf, tokens, ws_.normed.data(), ws_.ff.data());
        gelu_tanh_kernel<<<grid_for(static_cast<size_t>(tokens) * kFf), kThreads>>>(
            ws_.ff.as<Bf16>(), static_cast<size_t>(tokens) * kFf);
        UMM_DIT_CUDA(cudaGetLastError());
        linear(prefix + "ffn.net.2", kFf, kDim, tokens, ws_.ff.data(), ws_.q.data());
        apply_residual(ws_.q.as<Bf16>(), 5);
      }

      if (tc_decision.save_residual) {
        sub_residual_bf16_kernel<<<grid_for(model_count), kThreads>>>(
            residual_slot->as<Bf16>(), ws_.x.as<Bf16>(), model_count);
        UMM_DIT_CUDA(cudaGetLastError());
        rollback_guard.active = false;
      }
    }

    if (per_frame) {
      output_modulation_pair_kernel<<<grid_for(4 * kDim), kThreads>>>(
          static_cast<const float*>(tensor("scale_shift_table").data),
          ws_.temb.as<Bf16>(), ws_.modulation.as<float>());
      UMM_DIT_CUDA(cudaGetLastError());
      layer_norm_pair_kernel<<<tokens, kThreads>>>(
          ws_.x.as<Bf16>(), ws_.normed.as<Bf16>(), nullptr, nullptr,
          ws_.modulation.as<float>(), 2, 1, 0, tokens, config_.eps,
          first_frame_tokens);
    } else {
      output_modulation_kernel<<<grid_for(2 * kDim), kThreads>>>(
          static_cast<const float*>(tensor("scale_shift_table").data),
          ws_.temb.as<Bf16>(), ws_.modulation.as<float>());
      UMM_DIT_CUDA(cudaGetLastError());
      layer_norm_kernel<<<tokens, kThreads>>>(
          ws_.x.as<Bf16>(), ws_.normed.as<Bf16>(), nullptr, nullptr,
          ws_.modulation.as<float>(), ws_.modulation.as<float>() + kDim, tokens,
          config_.eps);
    }
    UMM_DIT_CUDA(cudaGetLastError());
    linear("proj_out", kDim, kProjection, tokens, ws_.normed.data(), ws_.q.data());

    std::vector<uint16_t> projected_bits(static_cast<size_t>(tokens) * kProjection);
    UMM_DIT_CUDA(cudaMemcpy(projected_bits.data(), ws_.q.data(),
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
    result.stats.teacache_hit = tc_decision.use_cached_residual;
    return result;
  }

  CudaContext context_;
  int gpu_ = 0;
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
  std::vector<uint16_t> teacache_e0_bits_;
  std::vector<float> teacache_e0_f32_;
  std::unordered_map<std::string, DeviceTensor> tensors_;
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

}  // namespace umminfer::video

#endif  // UMM_VIDEO_CUDA
