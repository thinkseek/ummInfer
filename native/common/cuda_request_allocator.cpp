// SPDX-License-Identifier: Apache-2.0
#include "cuda_request_allocator.hpp"

namespace umminfer::native {

cudaError_t request_cuda_malloc(void** ptr, size_t bytes) noexcept {
  const cudaError_t status = cudaMalloc(ptr, bytes);
  if (status == cudaSuccess && ptr != nullptr) {
    obs::note_native_allocation(*ptr, static_cast<uint64_t>(bytes));
  }
  return status;
}

cudaError_t request_cuda_free(void* ptr) noexcept {
  // Remove ownership only after CUDA accepts the free. A failed free must not
  // make the epoch under-report its live request-owned block.
  const cudaError_t status = cudaFree(ptr);
  if (status == cudaSuccess) obs::note_native_free(ptr);
  return status;
}

}  // namespace umminfer::native
