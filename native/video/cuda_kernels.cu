// SPDX-License-Identifier: Apache-2.0
#include "cuda_kernels.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace umminfer::video {

#ifdef UMM_VIDEO_CUDA

namespace {

__global__ void euler_step_f32_kernel(float* latents, const float* velocity,
                                      float dt, size_t count) {
  const size_t idx =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) {
    latents[idx] = fmaf(dt, velocity[idx], latents[idx]);
  }
}

__global__ void latent_denorm_f32_kernel(float* latents, const float* mean,
                                         const float* std_dev, int channels,
                                         size_t spatial_temporal) {
  const size_t idx =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = static_cast<size_t>(channels) * spatial_temporal;
  if (idx < total) {
    const int c = static_cast<int>(idx / spatial_temporal);
    latents[idx] = fmaf(latents[idx], std_dev[c], mean[c]);
  }
}

__device__ __forceinline__ uint8_t float_n1p1_to_u8(float v) {
  const float clamped = fminf(1.0f, fmaxf(-1.0f, v));
  const float scaled = (clamped * 0.5f + 0.5f) * 255.0f;
  return static_cast<uint8_t>(rintf(scaled));
}

__global__ void frames_n1p1_to_rgb24_kernel(const float* frames_tchw,
                                            uint8_t* rgb24_out, int num_frames,
                                            int height, int width) {
  const size_t hw = static_cast<size_t>(height) * static_cast<size_t>(width);
  const size_t total_pixels = static_cast<size_t>(num_frames) * hw;
  const size_t idx =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_pixels) return;

  const size_t t = idx / hw;
  const size_t pix = idx % hw;
  const size_t frame_base = t * 3u * hw;

  const float r = frames_tchw[frame_base + 0u * hw + pix];
  const float g = frames_tchw[frame_base + 1u * hw + pix];
  const float b = frames_tchw[frame_base + 2u * hw + pix];

  const size_t out_base = idx * 3u;
  rgb24_out[out_base + 0u] = float_n1p1_to_u8(r);
  rgb24_out[out_base + 1u] = float_n1p1_to_u8(g);
  rgb24_out[out_base + 2u] = float_n1p1_to_u8(b);
}

}  // namespace

void cuda_check_video(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(err));
  }
}

DeviceBuffer::DeviceBuffer(size_t bytes) : bytes_(bytes) {
  if (bytes_ == 0) return;
  cuda_check_video(cudaMalloc(&ptr_, bytes_), "cudaMalloc");
}

DeviceBuffer::~DeviceBuffer() { reset(); }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : ptr_(other.ptr_), bytes_(other.bytes_) {
  other.ptr_ = nullptr;
  other.bytes_ = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this != &other) {
    reset();
    ptr_ = other.ptr_;
    bytes_ = other.bytes_;
    other.ptr_ = nullptr;
    other.bytes_ = 0;
  }
  return *this;
}

void DeviceBuffer::reset() {
  if (ptr_ != nullptr) {
    (void)cudaFree(ptr_);
    ptr_ = nullptr;
    bytes_ = 0;
  }
}

void euler_step_f32_cuda(float* latents, const float* velocity, float dt,
                         size_t count, cudaStream_t stream) {
  if (count == 0) return;
  constexpr int kThreads = 256;
  const unsigned int blocks =
      static_cast<unsigned int>((count + kThreads - 1) / kThreads);
  euler_step_f32_kernel<<<blocks, kThreads, 0, stream>>>(latents, velocity, dt,
                                                         count);
  cuda_check_video(cudaGetLastError(), "euler_step_f32_kernel");
}

void latent_denorm_f32_cuda(float* latents, const float* mean,
                            const float* std_dev, int channels,
                            size_t spatial_temporal, cudaStream_t stream) {
  const size_t total = static_cast<size_t>(channels) * spatial_temporal;
  if (total == 0) return;
  constexpr int kThreads = 256;
  const unsigned int blocks =
      static_cast<unsigned int>((total + kThreads - 1) / kThreads);
  latent_denorm_f32_kernel<<<blocks, kThreads, 0, stream>>>(
      latents, mean, std_dev, channels, spatial_temporal);
  cuda_check_video(cudaGetLastError(), "latent_denorm_f32_kernel");
}

void frames_n1p1_to_rgb24_cuda(const float* frames_tchw, uint8_t* rgb24_out,
                               int num_frames, int height, int width,
                               cudaStream_t stream) {
  const size_t total_pixels = static_cast<size_t>(num_frames) *
                              static_cast<size_t>(height) *
                              static_cast<size_t>(width);
  if (total_pixels == 0) return;
  constexpr int kThreads = 256;
  const unsigned int blocks =
      static_cast<unsigned int>((total_pixels + kThreads - 1) / kThreads);
  frames_n1p1_to_rgb24_kernel<<<blocks, kThreads, 0, stream>>>(
      frames_tchw, rgb24_out, num_frames, height, width);
  cuda_check_video(cudaGetLastError(), "frames_n1p1_to_rgb24_kernel");
}

#endif  // UMM_VIDEO_CUDA

}  // namespace umminfer::video
