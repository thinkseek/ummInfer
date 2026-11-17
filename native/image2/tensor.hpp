// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace umminfer::image2 {

using Bf16 = __nv_bfloat16;

#define IMG_CUDA_CHECK(expr)                                                            \
  do {                                                                                  \
    cudaError_t _e = (expr);                                                            \
    if (_e != cudaSuccess)                                                              \
      throw std::runtime_error(std::string("CUDA error ") + cudaGetErrorString(_e) +    \
                               " at " + __FILE__ + ":" + std::to_string(__LINE__));     \
  } while (0)

// Thin owning row-major device tensor. BF16 storage, fp32 scratch via separate
// buffers. Shape is informational; kernels take explicit dims.
struct DeviceTensor {
  std::vector<int64_t> shape;
  Bf16* data = nullptr;
  size_t numel = 0;

  DeviceTensor() = default;
  explicit DeviceTensor(std::vector<int64_t> dims);
  ~DeviceTensor();
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;
  DeviceTensor(DeviceTensor&& o) noexcept { *this = std::move(o); }
  DeviceTensor& operator=(DeviceTensor&& o) noexcept;

  void zeros_();
  // Copy BF16 host bytes (numel*2) into the allocation.
  void copy_from_host(const void* bf16_bytes);
  std::vector<uint16_t> to_host_u16() const;
};

size_t product(const std::vector<int64_t>& shape);

// Host-side BF16 bit conversions (the CUDA __nv_bfloat16 type has no .bits
// member, so all host code goes through these helpers).
inline uint16_t bf16_to_bits(float f) {
  uint32_t u;
  __builtin_memcpy(&u, &f, 4);
  // round-to-nearest-even on the truncated 16-bit value.
  uint32_t lsb = (u >> 16) & 1;
  uint32_t rounded = (u + 0x7fffu + lsb) >> 16;
  return static_cast<uint16_t>(rounded);
}
inline float bits_to_bf16_float(uint16_t bits) {
  uint32_t u = static_cast<uint32_t>(bits) << 16;
  float f;
  __builtin_memcpy(&f, &u, 4);
  return f;
}

// Contiguous fp32 device buffer for RoPE tables and fp32 scratch.
struct DeviceTensorF {
  float* data = nullptr;
  size_t numel = 0;
  DeviceTensorF() = default;
  explicit DeviceTensorF(size_t n) : numel(n) { IMG_CUDA_CHECK(cudaMalloc(&data, n * sizeof(float))); }
  ~DeviceTensorF() { if (data) cudaFree(data); }
  DeviceTensorF(const DeviceTensorF&) = delete;
  DeviceTensorF& operator=(const DeviceTensorF&) = delete;
  DeviceTensorF(DeviceTensorF&& o) noexcept : data(o.data), numel(o.numel) {
    o.data = nullptr; o.numel = 0;
  }
  DeviceTensorF& operator=(DeviceTensorF&& o) noexcept {
    if (this != &o) {
      if (data) cudaFree(data);
      data = o.data; numel = o.numel;
      o.data = nullptr; o.numel = 0;
    }
    return *this;
  }
};

// One-process-per-GPU runtime state: stream + cuBLAS handle created once.
struct CudaRuntime {
  cudaStream_t stream = nullptr;
  void* cublas = nullptr;  // cublasHandle_t, opaque to keep cublas out of headers
  int device = 0;

  explicit CudaRuntime(int gpu_id);
  ~CudaRuntime();
  void sync() const;
};

// Row-major BF16 GEMM with fp32 accumulation:
//   out[m, n] = in[m, k] * w_t[k, n] + bias[n]   (bias optional)
// `w` is the PyTorch [n, k] row-major weight (out_features first); it is
// fed to cuBLAS transposed without any host-side copy.
void linear_forward(CudaRuntime& rt, const Bf16* in, const Bf16* w, const Bf16* bias,
                    Bf16* out, int m, int n, int k);

}  // namespace umminfer::image2
