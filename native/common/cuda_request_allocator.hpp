// SPDX-License-Identifier: Apache-2.0
// CUDA runtime allocation adapters. Only explicit native cudaMalloc/cudaFree
// calls in instrumented translation units are covered; no driver or external
// library memory is inferred. Standalone native CLI builds keep the original
// direct CUDA calls because they have no request scheduler or public telemetry.
#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

#ifdef UMM_NATIVE_REQUEST_ALLOCATOR

#include "umminfer/obs/request_telemetry.hpp"

namespace umminfer::native {

cudaError_t request_cuda_malloc(void** ptr, size_t bytes) noexcept;
cudaError_t request_cuda_free(void* ptr) noexcept;

template <typename T>
cudaError_t request_cuda_malloc(T** ptr, size_t bytes) noexcept {
  return request_cuda_malloc(reinterpret_cast<void**>(ptr), bytes);
}

}  // namespace umminfer::native

using umminfer::native::request_cuda_malloc;
using umminfer::native::request_cuda_free;

#else

template <typename T>
cudaError_t request_cuda_malloc(T** ptr, size_t bytes) noexcept {
  return cudaMalloc(reinterpret_cast<void**>(ptr), bytes);
}

inline cudaError_t request_cuda_free(void* ptr) noexcept { return cudaFree(ptr); }

#endif
