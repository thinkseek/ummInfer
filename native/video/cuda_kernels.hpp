// SPDX-License-Identifier: Apache-2.0
// CUDA kernels and RAII device buffer helpers for the native C++/CUDA video
// vertical slice (flow-matching Euler step, latent denormalization, and
// [-1, 1] TCHW -> RGB24 frame conversion).
#pragma once

#include <cstddef>
#include <cstdint>

#ifdef UMM_VIDEO_CUDA
#include <cuda_runtime.h>
#endif

namespace umminfer::video {

#ifdef UMM_VIDEO_CUDA

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t bytes);
  ~DeviceBuffer();

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

  void reset();
  void* data() const { return ptr_; }
  template <typename T>
  T* as() const {
    return static_cast<T*>(ptr_);
  }
  size_t bytes() const { return bytes_; }

 private:
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
};

void cuda_check_video(cudaError_t err, const char* what);

void euler_step_f32_cuda(float* latents, const float* velocity, float dt,
                         size_t count, cudaStream_t stream);

void latent_denorm_f32_cuda(float* latents, const float* mean,
                            const float* std_dev, int channels,
                            size_t spatial_temporal, cudaStream_t stream);

void frames_n1p1_to_rgb24_cuda(const float* frames_tchw, uint8_t* rgb24_out,
                               int num_frames, int height, int width,
                               cudaStream_t stream);

#endif  // UMM_VIDEO_CUDA

}  // namespace umminfer::video
