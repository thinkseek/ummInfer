// SPDX-License-Identifier: Apache-2.0
// CUDA-only ASR adapter checks using small real device allocations. Run with
// one explicitly isolated CUDA device; no model weights or inference are loaded.
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include "common/cuda_request_allocator.hpp"
#include "umminfer/obs/request_telemetry.hpp"

namespace {

using umminfer::Value;
using umminfer::obs::NativeRequestTelemetry;
using umminfer::obs::NativeRequestTelemetryConfig;
using umminfer::obs::NativeTelemetryExecution;

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

NativeRequestTelemetryConfig config(const std::string& request_id) {
  NativeRequestTelemetryConfig out;
  out.request_id = request_id;
  out.stage = "asr";
  out.backend = "asr";
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

void check_null_bytes(const Value& value) {
  for (const char* key : {"allocated_bytes_peak", "reserved_bytes_peak",
                          "allocated_bytes_final", "reserved_bytes_final"}) {
    const Value* field = value.find(key);
    check(field != nullptr && field->is_nil(), "unsupported byte field is not null");
  }
}

}  // namespace

int main() {
  void* resident = nullptr;
  void* cold_retained = nullptr;
  void* failed_ptr = nullptr;
  void* cancelled_ptr = nullptr;
  try {
    // Model weights, fixed encoder/decoder/KV workspaces and graph setup are
    // created before requests. They must not enter the first request epoch.
    resident = allocate(128);
    check(umminfer::obs::native_tracked_allocation_count_for_test() == 0,
          "resident ASR allocation entered request ownership");

    NativeRequestTelemetry cold(config("asr-cold"));
    void* cold_scratch = allocate(256);
    cold_retained = allocate(512);
    release(cold_scratch);
    Value cold_result = cold.finish("completed");
    check(text(cold_result, "coverage") == "COVERED", "cold ASR not covered");
    check(integer(cold_result, "allocated_bytes_peak") == 768 &&
              integer(cold_result, "reserved_bytes_peak") == 768,
          "cold direct allocated/reserved peaks differ from 768");
    check(integer(cold_result, "allocated_bytes_final") == 512 &&
              integer(cold_result, "reserved_bytes_final") == 512,
          "cold direct final does not subtract the completed free");
    check(sample(cold_result, "allocation_events") == 2 &&
              sample(cold_result, "free_events") == 1 &&
              sample(cold_result, "terminal_snapshots") == 1,
          "cold direct allocation/free/terminal counts are wrong");
    const int64_t cold_epoch = epoch_id(cold_result);

    NativeRequestTelemetry warm(config("asr-warm-late-free"));
    release(cold_retained);
    cold_retained = nullptr;
    Value warm_result = warm.finish("completed");
    check(integer(warm_result, "allocated_bytes_peak") == 0 &&
              integer(warm_result, "allocated_bytes_final") == 0 &&
              sample(warm_result, "allocation_events") == 0 &&
              sample(warm_result, "free_events") == 0,
          "late free contaminated a warm preallocated-workspace request");
    check(epoch_id(warm_result) > cold_epoch, "warm epoch is not monotonic");
    check(integer(cold.finish("completed"), "allocated_bytes_final") == 512 &&
              sample(cold.finish("completed"), "free_events") == 1,
          "late free rewrote the original terminal snapshot");
    check(umminfer::obs::native_tracked_allocation_count_for_test() == 0,
          "late free did not release the original owner pointer");

    NativeRequestTelemetry repeated(config("asr-warm-repeat"));
    Value repeated_result = repeated.finish("completed");
    check(epoch_id(repeated_result) > epoch_id(warm_result) &&
              integer(repeated_result, "allocated_bytes_peak") == 0 &&
              sample(repeated_result, "allocation_events") == 0 &&
              sample(repeated_result, "free_events") == 0,
          "warm repeat inherited resident bytes or an earlier epoch");

    NativeRequestTelemetry failed(config("asr-failed"));
    failed_ptr = allocate(128);
    Value failed_result = failed.finish("failed");
    check(text(failed_result, "lifecycle_status") == "failed" &&
              integer(failed_result, "allocated_bytes_peak") == 128 &&
              integer(failed_result, "allocated_bytes_final") == 128 &&
              sample(failed_result, "allocation_events") == 1 &&
              sample(failed_result, "free_events") == 0,
          "failed ASR lifecycle/counts/peak/final were not retained");
    release(failed_ptr);
    failed_ptr = nullptr;

    NativeRequestTelemetry cancelled(config("asr-cancelled"));
    cancelled_ptr = allocate(64);
    Value cancelled_result = cancelled.finish("cancelled");
    check(text(cancelled_result, "lifecycle_status") == "cancelled" &&
              integer(cancelled_result, "allocated_bytes_peak") == 64 &&
              sample(cancelled_result, "allocation_events") == 1 &&
              sample(cancelled_result, "free_events") == 0,
          "cancelled ASR lifecycle/counts/peak were not retained");
    release(cancelled_ptr);
    cancelled_ptr = nullptr;
    check(epoch_id(cancelled_result) > epoch_id(failed_result) &&
              epoch_id(failed_result) > epoch_id(repeated_result),
          "terminal lifecycle epochs are not unique and monotonic");

    NativeRequestTelemetryConfig unsupported_config = config("asr-unsupported");
    unsupported_config.instrumented = false;
    NativeRequestTelemetry unsupported(std::move(unsupported_config));
    Value unsupported_result = unsupported.finish("completed");
    check(text(unsupported_result, "coverage") == "NOT_COVERED" &&
              text(unsupported_result, "reason") == "backend_not_instrumented",
          "unsupported ASR did not fail closed");
    check_null_bytes(unsupported_result);

    NativeRequestTelemetryConfig batched_config = config("asr-batched");
    batched_config.execution = NativeTelemetryExecution::kBatched;
    batched_config.batch_capacity = 2;
    NativeRequestTelemetry batched(std::move(batched_config));
    Value batched_result = batched.finish("completed");
    check(text(batched_result, "coverage") == "NOT_COVERED" &&
              text(batched_result, "reason") ==
                  "batched_request_attribution_unavailable",
          "batched ASR did not remain explicitly unsupported");
    check_null_bytes(batched_result);

    release(resident);
    resident = nullptr;
    check(umminfer::obs::native_tracked_allocation_count_for_test() == 0,
          "tracked ASR device blocks remained at test exit");
    std::cout << "asr_request_allocator_test: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    if (cancelled_ptr) cudaFree(cancelled_ptr);
    if (failed_ptr) cudaFree(failed_ptr);
    if (cold_retained) cudaFree(cold_retained);
    if (resident) cudaFree(resident);
    std::cerr << "asr_request_allocator_test: FAIL: " << error.what() << "\n";
    return 1;
  }
}
