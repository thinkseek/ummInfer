// SPDX-License-Identifier: Apache-2.0
// Real-CUDA regression for deterministic repetition penalties with duplicate
// emitted ids. No model files are needed.
#include "cuda_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    const cudaError_t status = (expr);                                          \
    if (status != cudaSuccess) {                                                \
      std::cerr << "CUDA failure: " << cudaGetErrorString(status) << " at "     \
                << __FILE__ << ":" << __LINE__ << "\n";                       \
      return false;                                                             \
    }                                                                           \
  } while (false)

bool run_case(const std::string& name, const std::vector<int>& history,
              cudaStream_t stream, float* device_logits, int* device_history) {
  constexpr int kLogits = 3072;
  constexpr float kPenalty = 2.0f;
  constexpr int kRounds = 16;
  std::vector<float> initial(kLogits);
  for (int i = 0; i < kLogits; ++i) {
    initial[i] = static_cast<float>((i % 9) - 4) * 0.5f;
  }
  std::vector<float> expected = initial;
  for (int id = 0; id < kLogits; ++id) {
    if (std::find(history.begin(), history.end(), id) == history.end()) continue;
    expected[id] = expected[id] > 0.0f ? expected[id] / kPenalty
                                       : expected[id] * kPenalty;
  }
  std::vector<float> previous;
  std::vector<float> actual(kLogits);
  int semantic_failures = 0;
  int repeat_failures = 0;
  for (int round = 0; round < kRounds; ++round) {
    CUDA_CHECK(cudaMemcpyAsync(device_logits, initial.data(),
                               initial.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(device_history, history.data(),
                               history.size() * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    umminfer::native::repetition_penalty_f32(
        device_logits, device_history, static_cast<int>(history.size()), kLogits,
        kPenalty, stream);
    CUDA_CHECK(cudaMemcpyAsync(actual.data(), device_logits,
                               actual.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (actual != expected) {
      ++semantic_failures;
      if (semantic_failures == 1) {
        const auto it = std::mismatch(actual.begin(), actual.end(), expected.begin());
        const int id = static_cast<int>(it.first - actual.begin());
        std::cerr << "duplicate-history semantic mismatch case=" << name
                  << " round=" << round << " id=" << id
                  << " actual=" << *it.first << " expected=" << *it.second << "\n";
      }
    }
    if (round > 0 && actual != previous) ++repeat_failures;
    previous = actual;
  }
  std::cout << "case=" << name << " history=" << history.size()
            << " rounds=" << kRounds
            << " semantic_failures=" << semantic_failures
            << " repeat_failures=" << repeat_failures << "\n";
  return semantic_failures == 0 && repeat_failures == 0;
}

}  // namespace

int main() {
  constexpr int kLogits = 3072;
  constexpr int kHistory = 4096;
  cudaStream_t stream = nullptr;
  float* device_logits = nullptr;
  int* device_history = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess ||
      cudaMalloc(&device_logits, kLogits * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&device_history, kHistory * sizeof(int)) != cudaSuccess) {
    std::cerr << "failed to allocate CUDA determinism fixture\n";
    return 2;
  }
  std::vector<int> mixed(kHistory);
  for (int i = 0; i < kHistory; ++i) mixed[i] = (i * 37) % kLogits;
  std::vector<int> repeated(kHistory, 19);
  const bool mixed_ok = run_case("mixed_duplicates", mixed, stream, device_logits,
                                 device_history);
  const bool repeated_ok = run_case("single_id_dense_duplicates", repeated, stream,
                                    device_logits, device_history);
  cudaFree(device_history);
  cudaFree(device_logits);
  cudaStreamDestroy(stream);
  if (!mixed_ok || !repeated_ok) return 1;
  std::cout << "tts CUDA repetition determinism passed\n";
  return 0;
}
