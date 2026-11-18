// SPDX-License-Identifier: Apache-2.0
// Measures the batched decode GEMV against the single-row GEMV on one layer
// sized matrix, and checks that every row is bitwise equal.
//
// Build (on the GPU host):
//   nvcc -std=c++17 -O3 -arch=sm_90 -I native \
//     native/qwen38_tests/test_gemv_batch.cu \
//     native/qwen38/qwen38_gemm.cu native/qwen38/qwen38_kernels.cu \
//     -o gemv_batch_bench
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "qwen38/qwen38_ops.hpp"

using namespace umminfer::qwen38;

namespace {

void check(cudaError_t error, const char* what) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(error));
    std::exit(1);
  }
}

float time_gemv_batch(const Bf16* w, const Bf16* x, Bf16* y, int m, int n, int k,
                      int iters) {
  cudaEvent_t start, stop;
  check(cudaEventCreate(&start), "event");
  check(cudaEventCreate(&stop), "event");
  for (int i = 0; i < 3; ++i) gemv_bf16_bf16_batch(w, x, y, m, n, k, nullptr);
  check(cudaDeviceSynchronize(), "warmup");
  check(cudaEventRecord(start), "record");
  for (int i = 0; i < iters; ++i) gemv_bf16_bf16_batch(w, x, y, m, n, k, nullptr);
  check(cudaEventRecord(stop), "record");
  check(cudaEventSynchronize(stop), "sync");
  float ms = 0.0f;
  check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms / iters;
}

}  // namespace

int main() {
  // One Qwen3.8 MLP gate projection: the widest decode matrix.
  const int n = 17408;
  const int k = 5120;
  const int kMax = 8;
  const size_t weight_bytes = static_cast<size_t>(n) * k * sizeof(Bf16);
  Bf16* w = nullptr;
  Bf16* x = nullptr;
  Bf16* y_batch = nullptr;
  Bf16* y_one = nullptr;
  check(cudaMalloc(&w, weight_bytes), "malloc w");
  check(cudaMalloc(&x, static_cast<size_t>(kMax) * k * sizeof(Bf16)), "malloc x");
  check(cudaMalloc(&y_batch, static_cast<size_t>(kMax) * n * sizeof(Bf16)), "malloc y");
  check(cudaMalloc(&y_one, static_cast<size_t>(n) * sizeof(Bf16)), "malloc y1");
  // Any nonzero payload works for timing; 0x3e3e is a small normal bf16.
  check(cudaMemset(w, 0x3e, weight_bytes), "fill w");
  check(cudaMemset(x, 0x3e, static_cast<size_t>(kMax) * k * sizeof(Bf16)), "fill x");

  std::printf("matrix n=%d k=%d weight=%.3f GiB\n", n, k,
              weight_bytes / 1073741824.0);
  std::printf("rows  ms/step  effective_GBps  GFLOP/s\n");
  for (int m = 1; m <= kMax; ++m) {
    const float ms = time_gemv_batch(w, x, y_batch, m, n, k, 20);
    const double gbps = weight_bytes / (ms * 1e6);
    const double gflops = 2.0 * n * k * m / (ms * 1e6);
    std::printf("%4d  %7.3f  %13.1f  %7.1f\n", m, ms, gbps, gflops);
  }

  float ms_one = 0.0f;
  {
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event");
    check(cudaEventCreate(&stop), "event");
    for (int i = 0; i < 3; ++i) gemv_bf16_bf16(w, x, y_one, n, k, nullptr);
    check(cudaDeviceSynchronize(), "warmup");
    check(cudaEventRecord(start), "record");
    for (int i = 0; i < 20; ++i) gemv_bf16_bf16(w, x, y_one, n, k, nullptr);
    check(cudaEventRecord(stop), "record");
    check(cudaEventSynchronize(stop), "sync");
    check(cudaEventElapsedTime(&ms_one, start, stop), "elapsed");
  }
  ms_one /= 20.0f;
  std::printf("single-row gemv_bf16_bf16: %.3f ms  %.1f GB/s\n", ms_one,
              weight_bytes / (ms_one * 1e6));

  // The tensor-core GEMM reads the weight once per step. Its fp32 accumulation
  // order differs from the GEMV, so it is timed for comparison only.
  std::printf("gemm_bf16 tensor-core path:\n");
  for (int m = 1; m <= kMax; m *= 2) {
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event");
    check(cudaEventCreate(&stop), "event");
    for (int i = 0; i < 3; ++i) gemm_bf16(y_batch, x, w, nullptr, m, k, n, nullptr);
    check(cudaDeviceSynchronize(), "warmup");
    check(cudaEventRecord(start), "record");
    for (int i = 0; i < 20; ++i) gemm_bf16(y_batch, x, w, nullptr, m, k, n, nullptr);
    check(cudaEventRecord(stop), "record");
    check(cudaEventSynchronize(stop), "sync");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    ms /= 20.0f;
    const double gbps = weight_bytes / (ms * 1e6);
    const double gflops = 2.0 * n * k * m / (ms * 1e6);
    std::printf("%4d  %7.3f  %13.1f  %7.1f\n", m, ms, gbps, gflops);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  // Row equivalence: the batched kernel must reproduce the single-row output
  // bit for bit for every row.
  int mismatches = 0;
  for (int m = 1; m <= kMax; ++m) {
    gemv_bf16_bf16(w, x, y_one, n, k, nullptr);
    gemv_bf16_bf16_batch(w, x, y_batch, m, n, k, nullptr);
    check(cudaDeviceSynchronize(), "sync");
    std::vector<Bf16> one(static_cast<size_t>(n));
    check(cudaMemcpy(one.data(), y_one, one.size() * sizeof(Bf16),
                     cudaMemcpyDeviceToHost),
          "copy");
    for (int b = 0; b < m; ++b) {
      std::vector<Bf16> many(static_cast<size_t>(n));
      check(cudaMemcpy(many.data(),
                       y_batch + static_cast<size_t>(b) * n,
                       many.size() * sizeof(Bf16), cudaMemcpyDeviceToHost),
            "copy");
      if (std::memcmp(one.data(), many.data(), one.size() * sizeof(Bf16)) != 0) {
        ++mismatches;
      }
    }
  }
  std::printf("row_equivalence mismatches=%d rows_checked=%d\n", mismatches,
              kMax * (kMax + 1) / 2);
  return mismatches == 0 ? 0 : 1;
}
