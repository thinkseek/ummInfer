// SPDX-License-Identifier: Apache-2.0
// Real-CUDA Qwen-Image-2.1 adapter checks. This fixture deliberately uses the
// engine's DeviceTensor/DeviceTensorF owners on one explicitly selected device;
// it loads no model weights and performs no image inference.
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "image2/tensor.hpp"
#include "umminfer/obs/request_telemetry.hpp"

namespace {

using umminfer::Value;
using umminfer::image2::DeviceTensor;
using umminfer::image2::DeviceTensorF;
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
  out.stage = "image";
  out.backend = "qwen-image-2.1";
  out.device_index = 1;
  out.instrumented = true;
  return out;
}

void check_zero_event_epoch(const Value& value) {
  check(text(value, "coverage") == "COVERED" &&
            integer(value, "allocated_bytes_peak") == 0 &&
            integer(value, "reserved_bytes_peak") == 0 &&
            integer(value, "allocated_bytes_final") == 0 &&
            integer(value, "reserved_bytes_final") == 0 &&
            sample(value, "allocation_events") == 0 &&
            sample(value, "free_events") == 0 &&
            sample(value, "terminal_snapshots") == 1,
        "zero-event epoch is not a covered no-direct-allocation observation");
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
  try {
    check(cudaSetDevice(1) == cudaSuccess, "cannot select physical CUDA device 1");

    // These allocations model the engine's pre-request weights/workspaces. No
    // active request epoch exists, so they cannot enter request ownership.
    DeviceTensor resident_weights({128});
    DeviceTensorF resident_workspace(32);
    check(umminfer::obs::native_tracked_allocation_count_for_test() == 0,
          "pre-request image2 allocations entered request ownership");

    NativeRequestTelemetry cold(config("image2-cold"));
    auto cold_retained = std::make_unique<DeviceTensorF>(128);  // 512 bytes
    {
      DeviceTensor cold_scratch({128});  // 256 bytes, freed before terminal
    }
    Value cold_result = cold.finish("completed");
    check(text(cold_result, "coverage") == "COVERED" &&
              integer(cold_result, "allocated_bytes_peak") == 768 &&
              integer(cold_result, "reserved_bytes_peak") == 768 &&
              integer(cold_result, "allocated_bytes_final") == 512 &&
              sample(cold_result, "allocation_events") == 2 &&
              sample(cold_result, "free_events") == 1,
          "image2 cold direct allocation peak/final/events are wrong");
    const int64_t cold_epoch = epoch_id(cold_result);

    NativeRequestTelemetry warm_late_free(config("image2-warm-late-free"));
    cold_retained.reset();
    Value warm_result = warm_late_free.finish("completed");
    check_zero_event_epoch(warm_result);
    check(epoch_id(warm_result) > cold_epoch, "warm epoch is not monotonic");
    check(integer(cold.finish("completed"), "allocated_bytes_final") == 512 &&
              sample(cold.finish("completed"), "free_events") == 1,
          "late free rewrote the original immutable terminal snapshot");
    check(umminfer::obs::native_tracked_allocation_count_for_test() == 0,
          "late free did not release the original image2 owner");

    NativeRequestTelemetry warm_repeat(config("image2-warm-repeat"));
    Value repeated_result = warm_repeat.finish("completed");
    check_zero_event_epoch(repeated_result);
    check(epoch_id(repeated_result) > epoch_id(warm_result),
          "warm repeat inherited an earlier epoch");

    NativeRequestTelemetry warm_owned(config("image2-warm-owned"));
    {
      DeviceTensor bf16({32});   // 64 bytes
      DeviceTensorF fp32(8);    // 32 bytes
    }
    Value warm_owned_result = warm_owned.finish("completed");
    check(integer(warm_owned_result, "allocated_bytes_peak") == 96 &&
              integer(warm_owned_result, "allocated_bytes_final") == 0 &&
              sample(warm_owned_result, "allocation_events") == 2 &&
              sample(warm_owned_result, "free_events") == 2,
          "warm image2 request allocations were not independently measured");

    NativeRequestTelemetry failed(config("image2-failed"));
    auto failed_block = std::make_unique<DeviceTensor>(std::vector<int64_t>{64});
    Value failed_result = failed.finish("failed");
    check(text(failed_result, "lifecycle_status") == "failed" &&
              integer(failed_result, "allocated_bytes_peak") == 128 &&
              integer(failed_result, "allocated_bytes_final") == 128 &&
              sample(failed_result, "free_events") == 0,
          "failed image2 lifecycle/peak/final were not retained");
    failed_block.reset();

    NativeRequestTelemetry cancelled(config("image2-cancelled"));
    auto cancelled_block = std::make_unique<DeviceTensorF>(16);
    Value cancelled_result = cancelled.finish("cancelled");
    check(text(cancelled_result, "lifecycle_status") == "cancelled" &&
              integer(cancelled_result, "allocated_bytes_peak") == 64 &&
              sample(cancelled_result, "allocation_events") == 1 &&
              sample(cancelled_result, "free_events") == 0,
          "cancelled image2 lifecycle/peak/events were not retained");
    cancelled_block.reset();
    check(epoch_id(cancelled_result) > epoch_id(failed_result) &&
              epoch_id(failed_result) > epoch_id(warm_owned_result),
          "image2 terminal lifecycle epochs are not unique and monotonic");

    NativeRequestTelemetryConfig unsupported_config = config("image2-unsupported");
    unsupported_config.instrumented = false;
    NativeRequestTelemetry unsupported(std::move(unsupported_config));
    Value unsupported_result = unsupported.finish("completed");
    check(text(unsupported_result, "coverage") == "NOT_COVERED" &&
              text(unsupported_result, "reason") == "backend_not_instrumented",
          "unsupported image backend did not fail closed");
    check_null_bytes(unsupported_result);

    NativeRequestTelemetryConfig batched_config = config("image2-batched");
    batched_config.execution = NativeTelemetryExecution::kBatched;
    batched_config.batch_capacity = 2;
    NativeRequestTelemetry batched(std::move(batched_config));
    Value batched_result = batched.finish("completed");
    check(text(batched_result, "coverage") == "NOT_COVERED" &&
              text(batched_result, "reason") ==
                  "batched_request_attribution_unavailable",
          "batched image2 attribution did not remain explicitly unsupported");
    check_null_bytes(batched_result);

    check(umminfer::obs::native_tracked_allocation_count_for_test() == 0,
          "tracked image2 device blocks remained at fixture exit");
    std::cout << "image2_request_allocator_test: PASS cold_peak=768 "
                 "warm_owned_peak=96 final=0\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "image2_request_allocator_test: FAIL: " << error.what() << "\n";
    return 1;
  }
}
