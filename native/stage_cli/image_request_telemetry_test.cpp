// SPDX-License-Identifier: Apache-2.0
// CPU-only image-backend coverage selection and protocol tests.
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include "request_telemetry_stage.hpp"
#include "umminfer/obs/request_telemetry.hpp"
#include "umminfer/proto/wire.hpp"
#include "umminfer/scheduling/simple_scheduler.hpp"

namespace umminfer::stage {
const char* compiled_kind() { return "image"; }
}  // namespace umminfer::stage

namespace {

using umminfer::IncomingMessage;
using umminfer::OutgoingMessage;
using umminfer::SimpleScheduler;
using umminfer::Value;

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

void check_null_bytes(const Value& value) {
  for (const char* key : {"allocated_bytes_peak", "reserved_bytes_peak",
                          "allocated_bytes_final", "reserved_bytes_final"}) {
    const Value* field = value.find(key);
    check(field != nullptr && field->is_nil(), "unsupported byte field is not null");
  }
}

OutgoingMessage run(SimpleScheduler& scheduler, IncomingMessage request) {
  std::thread worker([&scheduler] { scheduler.start(); });
  scheduler.enqueue(std::move(request));
  auto result = scheduler.outbox().get(5.0);
  scheduler.stop();
  worker.join();
  check(result.has_value(), "image scheduler terminal result timeout");
  return std::move(*result);
}

OutgoingMessage run_image(umminfer::stage::StageOptions options,
                          bool telemetry, const char* request_id) {
  auto compute = umminfer::stage::with_native_request_telemetry(
      [](const Value&, const umminfer::AbortCheck&) {
        umminfer::obs::note_native_allocation(reinterpret_cast<void*>(0x2100), 256);
        return Value(Value::Map{{"media_type", Value("image")}});
      },
      options, "image", true);
  SimpleScheduler scheduler(nullptr);
  scheduler.set_cancellable_compute(compute);
  Value::Map params;
  if (telemetry) {
    params[umminfer::obs::kNativeAllocatorTelemetryParam] = Value(true);
  }
  return run(scheduler, IncomingMessage{request_id, "new_request", Value(), params});
}

}  // namespace

int main() {
  try {
    umminfer::stage::StageOptions options;
    options.gpu = 1;
    options.image_backend = "2.1";
    OutgoingMessage covered = run_image(options, true, "image21-covered");
    check(covered.type == "result", "image2.1 result type changed");
    const Value* telemetry =
        covered.data.find(umminfer::obs::kRequestTelemetryResultKey);
    check(telemetry != nullptr, "image2.1 telemetry missing");
    check(text(*telemetry, "coverage") == "COVERED" &&
              text(*telemetry, "backend") == "qwen-image-2.1" &&
              text(*telemetry, "stage") == "image" &&
              integer(*telemetry, "device_index") == 1 &&
              integer(*telemetry, "allocated_bytes_peak") == 256 &&
              integer(*telemetry, "allocated_bytes_final") == 256,
          "image2.1 backend identity or owned bytes are wrong");
    umminfer::obs::note_native_free(reinterpret_cast<void*>(0x2100));

    options.image_backend = "qwen-image-2.1";
    OutgoingMessage alias = run_image(options, true, "image21-alias-covered");
    telemetry = alias.data.find(umminfer::obs::kRequestTelemetryResultKey);
    check(telemetry != nullptr && text(*telemetry, "coverage") == "COVERED" &&
              text(*telemetry, "backend") == "qwen-image-2.1",
          "accepted full image2.1 alias was not instrumented");
    umminfer::obs::note_native_free(reinterpret_cast<void*>(0x2100));

    options.image_backend = "2512";
    OutgoingMessage unsupported = run_image(options, true, "image2512-unsupported");
    telemetry = unsupported.data.find(umminfer::obs::kRequestTelemetryResultKey);
    check(telemetry != nullptr && text(*telemetry, "coverage") == "NOT_COVERED" &&
              text(*telemetry, "reason") == "backend_not_instrumented" &&
              text(*telemetry, "backend") == "qwen-image-2512",
          "image2512 must remain explicitly uninstrumented");
    check_null_bytes(*telemetry);

    options.image_backend = "2.1";
    OutgoingMessage default_off = run_image(options, false, "image21-default");
    const Value original(Value::Map{{"media_type", Value("image")}});
    check(umminfer::pack(default_off.data) == umminfer::pack(original),
          "default-off image result bytes changed");

    options.batch_size = 2;
    OutgoingMessage batched = run_image(options, true, "image21-batched");
    telemetry = batched.data.find(umminfer::obs::kRequestTelemetryResultKey);
    check(telemetry != nullptr && text(*telemetry, "coverage") == "NOT_COVERED" &&
              text(*telemetry, "reason") ==
                  "batched_request_attribution_unavailable",
          "batched image attribution did not fail closed");
    const Value* execution = telemetry->find("execution");
    check(execution != nullptr && integer(*execution, "batch_capacity") == 2,
          "batched image capacity was not retained");
    check_null_bytes(*telemetry);
    check(umminfer::obs::native_tracked_allocation_count_for_test() == 0,
          "image telemetry retained a fake pointer");
    std::cout << "image_request_telemetry_test: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "image_request_telemetry_test: FAIL: " << error.what() << "\n";
    return 1;
  }
}
