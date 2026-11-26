// SPDX-License-Identifier: Apache-2.0
// CUDA-only adapter checks using small real device allocations. Run with one
// explicitly isolated CUDA device; no model weights or inference are loaded.
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "common/cuda_request_allocator.hpp"
#include "umminfer/obs/request_telemetry.hpp"

namespace {

using umminfer::Value;
using umminfer::obs::NativeRequestTelemetry;
using umminfer::obs::NativeRequestTelemetryConfig;

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

int64_t integer(const Value& value, const char* key) {
  const Value* field = value.find(key);
  check(field != nullptr && field->is_int(), key);
  return field->as_int();
}

std::string text(const Value& value, const char* key) {
  const Value* field = value.find(key);
  check(field != nullptr && field->is_str(), key);
  return field->as_str();
}

int64_t epoch_id(const Value& value) {
  const Value* epoch = value.find("epoch");
  check(epoch != nullptr, "epoch missing");
  return integer(*epoch, "id");
}

int64_t sample(const Value& value, const char* key) {
  const Value* samples = value.find("sample_coverage");
  check(samples != nullptr, "sample_coverage missing");
  return integer(*samples, key);
}

NativeRequestTelemetryConfig config(const std::string& request_id,
                                    const char* lifecycle_stage = "tts") {
  NativeRequestTelemetryConfig out;
  out.request_id = request_id;
  out.stage = lifecycle_stage;
  out.backend = "tts";
  out.device_index = 0;
  out.instrumented = true;
  return out;
}

void* allocate(size_t bytes) {
  void* ptr = nullptr;
  check(request_cuda_malloc(&ptr, bytes) == cudaSuccess, "cudaMalloc failed");
  check(ptr != nullptr, "cudaMalloc returned null");
  return ptr;
}

void release(void* ptr) {
  check(request_cuda_free(ptr) == cudaSuccess, "cudaFree failed");
}

}  // namespace

int main() {
  void* resident = nullptr;
  void* cold_retained = nullptr;
  void* failed_ptr = nullptr;
  void* cancelled_ptr = nullptr;
  try {
    // A load-time allocation is deliberately outside every request epoch.
    resident = allocate(128);
    check(umminfer::obs::native_tracked_allocation_count_for_test() == 0,
          "resident allocation entered request ownership");

    NativeRequestTelemetry cold(config("tts-cold"));
    void* cold_scratch = allocate(256);
    cold_retained = allocate(512);
    release(cold_scratch);
    Value cold_result = cold.finish("completed");
    // Retain the second block across this terminal and free it during the next
    // epoch. The terminal remains an immutable snapshot of the original owner.
    check(text(cold_result, "coverage") == "COVERED", "cold TTS not covered");
    check(integer(cold_result, "allocated_bytes_peak") == 768,
          "cold peak does not equal direct device bytes");
    check(integer(cold_result, "reserved_bytes_peak") == 768,
          "direct reserved peak differs from allocated");
    check(integer(cold_result, "allocated_bytes_final") == 512,
          "cold terminal final does not subtract its completed free");
    check(sample(cold_result, "allocation_events") == 2,
          "cold allocation count wrong");
    const int64_t cold_epoch = epoch_id(cold_result);

    NativeRequestTelemetry warm(config("tts-warm-late-free"));
    release(cold_retained);
    cold_retained = nullptr;
    Value warm_result = warm.finish("completed");
    check(integer(warm_result, "allocated_bytes_peak") == 0 &&
              integer(warm_result, "allocated_bytes_final") == 0 &&
              sample(warm_result, "allocation_events") == 0 &&
              sample(warm_result, "free_events") == 0,
          "late free contaminated warm preallocated-pool request");
    check(epoch_id(warm_result) > cold_epoch, "warm epoch not monotonic");
    check(integer(cold.finish("completed"), "allocated_bytes_final") == 512 &&
              sample(cold.finish("completed"), "free_events") == 1,
          "late free rewrote the terminal snapshot");
    check(umminfer::obs::native_tracked_allocation_count_for_test() == 0,
          "late free did not release original owner pointer");

    NativeRequestTelemetry repeated(config("tts-warm-repeat"));
    Value repeated_result = repeated.finish("completed");
    check(epoch_id(repeated_result) > epoch_id(warm_result) &&
              integer(repeated_result, "allocated_bytes_peak") == 0 &&
              sample(repeated_result, "allocation_events") == 0,
          "repeated warm request inherited resident bytes or epoch");

    NativeRequestTelemetry failed(config("tts-failed"));
    failed_ptr = allocate(128);
    Value failed_result = failed.finish("failed");
    check(text(failed_result, "lifecycle_status") == "failed" &&
              integer(failed_result, "allocated_bytes_peak") == 128,
          "failed TTS lifecycle/peak not retained");
    release(failed_ptr);
    failed_ptr = nullptr;

    NativeRequestTelemetry cancelled(config("tts-cancelled"));
    cancelled_ptr = allocate(64);
    Value cancelled_result = cancelled.finish("cancelled");
    check(text(cancelled_result, "lifecycle_status") == "cancelled" &&
              integer(cancelled_result, "allocated_bytes_peak") == 64,
          "cancelled TTS lifecycle/peak not retained");
    release(cancelled_ptr);
    cancelled_ptr = nullptr;

    NativeRequestTelemetryConfig unsupported_config = config("tts-unsupported");
    unsupported_config.instrumented = false;
    NativeRequestTelemetry unsupported(std::move(unsupported_config));
    Value unsupported_result = unsupported.finish("completed");
    check(text(unsupported_result, "coverage") == "NOT_COVERED" &&
              text(unsupported_result, "reason") == "backend_not_instrumented" &&
              unsupported_result.find("allocated_bytes_peak")->is_nil(),
          "unsupported TTS fabricated covered zero bytes");

    release(resident);
    resident = nullptr;
    check(umminfer::obs::native_tracked_allocation_count_for_test() == 0,
          "tracked device blocks remained at test exit");
    std::cout << "tts_request_allocator_test: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    if (cancelled_ptr) cudaFree(cancelled_ptr);
    if (failed_ptr) cudaFree(failed_ptr);
    if (cold_retained) cudaFree(cold_retained);
    if (resident) cudaFree(resident);
    std::cerr << "tts_request_allocator_test: FAIL: " << error.what() << "\n";
    return 1;
  }
}
