// SPDX-License-Identifier: Apache-2.0
// CPU-only fake-allocator and scheduler tests for request-scoped telemetry.
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include "request_telemetry_stage.hpp"
#include "umminfer/obs/request_telemetry.hpp"
#include "umminfer/proto/wire.hpp"

namespace umminfer::stage {
const char* compiled_kind() { return "qwen38"; }
}  // namespace umminfer::stage

namespace {

using umminfer::AbortCheck;
using umminfer::IncomingMessage;
using umminfer::OutgoingMessage;
using umminfer::RequestAborted;
using umminfer::SimpleScheduler;
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

Value* mutable_field(Value& value, const char* key) {
  check(value.is_map(), "map expected");
  return &value[key];
}

NativeRequestTelemetryConfig config(std::string id) {
  NativeRequestTelemetryConfig out;
  out.request_id = std::move(id);
  out.stage = "native-test";
  out.backend = "qwen38";
  out.device_index = 0;
  out.instrumented = true;
  return out;
}

void* fake_pointer(uintptr_t value) {
  return reinterpret_cast<void*>(value);
}

void test_serial_epochs_and_retained_free_isolation() {
  auto first = std::make_unique<NativeRequestTelemetry>(config("req-first"));
  umminfer::obs::note_native_allocation(fake_pointer(0x1000), 100);
  umminfer::obs::note_native_allocation(fake_pointer(0x2000), 50);
  check(umminfer::obs::native_tracked_allocation_count_for_test() == 2,
        "two fake allocations tracked");
  umminfer::obs::note_native_free(fake_pointer(0x1000));
  Value first_result = first->finish("completed");
  check(text(first_result, "coverage") == "COVERED", "first request covered");
  check(integer(first_result, "allocated_bytes_peak") == 150, "first peak 150");
  check(integer(first_result, "reserved_bytes_peak") == 150, "first reserved peak");
  check(integer(first_result, "allocated_bytes_final") == 50, "first final 50");
  check(integer(first_result, "schema_version") == 1, "schema v1");
  check(integer(*mutable_field(first_result, "epoch"), "id") > 0, "first epoch id");
  const int64_t first_epoch = integer(*mutable_field(first_result, "epoch"), "id");
  check(umminfer::pack(first->finish("completed")) == umminfer::pack(first_result),
        "terminal finish idempotent");

  auto second = std::make_unique<NativeRequestTelemetry>(config("req-second"));
  umminfer::obs::note_native_allocation(fake_pointer(0x3000), 25);
  // A retained first-request block freed during request two updates only its
  // original owner. Request two must neither inherit nor subtract those bytes.
  umminfer::obs::note_native_free(fake_pointer(0x2000));
  Value second_result = second->finish("cancelled");
  check(integer(second_result, "allocated_bytes_peak") == 25, "second peak reset");
  check(integer(second_result, "allocated_bytes_final") == 25,
        "old free does not contaminate new final");
  check(text(second_result, "lifecycle_status") == "cancelled",
        "cancelled lifecycle retained");
  check(integer(*mutable_field(second_result, "epoch"), "id") > first_epoch,
        "epoch ids are monotonic");
  umminfer::obs::note_native_free(fake_pointer(0x3000));
  check(umminfer::obs::native_tracked_allocation_count_for_test() == 0,
        "all fake allocations released");

  auto failed = std::make_unique<NativeRequestTelemetry>(config("req-failed"));
  umminfer::obs::note_native_allocation(fake_pointer(0x4000), 70);
  Value failed_result = failed->finish("failed");
  check(text(failed_result, "lifecycle_status") == "failed", "failed status");
  check(integer(failed_result, "allocated_bytes_peak") == 70,
        "failed request keeps peak");
  umminfer::obs::note_native_free(fake_pointer(0x4000));
}

void test_overlap_batch_and_unsupported_are_not_covered() {
  std::mutex overlap_mu;
  std::condition_variable overlap_cv;
  bool first_active = false;
  bool second_active = false;
  Value first_result;
  Value second_result;
  std::thread first_thread([&] {
    NativeRequestTelemetry first(config("req-overlap-a"));
    std::unique_lock<std::mutex> lock(overlap_mu);
    first_active = true;
    overlap_cv.notify_all();
    check(overlap_cv.wait_for(lock, std::chrono::seconds(5),
                              [&] { return second_active; }),
          "second overlapping epoch did not start");
    lock.unlock();
    umminfer::obs::note_native_allocation(fake_pointer(0x5000), 100);
    first_result = first.finish("completed");
  });
  {
    std::unique_lock<std::mutex> lock(overlap_mu);
    check(overlap_cv.wait_for(lock, std::chrono::seconds(5),
                              [&] { return first_active; }),
          "first overlapping epoch did not start");
  }
  std::thread second_thread([&] {
    NativeRequestTelemetry second(config("req-overlap-b"));
    {
      std::lock_guard<std::mutex> lock(overlap_mu);
      second_active = true;
      overlap_cv.notify_all();
    }
    second_result = second.finish("completed");
  });
  second_thread.join();
  first_thread.join();
  check(text(first_result, "coverage") == "NOT_COVERED", "overlap poisons first");
  check(text(second_result, "coverage") == "NOT_COVERED", "overlap poisons second");
  check(text(first_result, "reason") == "overlapping_serial_epochs",
        "overlap reason is explicit");
  check(first_result.find("allocated_bytes_peak")->is_nil(),
        "no shared-counter subtraction");
  check(umminfer::obs::native_tracked_allocation_count_for_test() == 0,
        "poisoned epochs do not retain fake pointers");

  NativeRequestTelemetryConfig batched = config("req-batch-a");
  batched.execution = NativeTelemetryExecution::kBatched;
  batched.batch_capacity = 4;
  NativeRequestTelemetry batch(std::move(batched));
  umminfer::obs::note_native_allocation(fake_pointer(0x6000), 999);
  Value batch_result = batch.finish("completed");
  check(text(batch_result, "coverage") == "NOT_COVERED", "batch not attributed");
  check(text(batch_result, "request_id") == "req-batch-a", "batch request id");
  check(integer(*mutable_field(batch_result, "execution"), "batch_capacity") == 4,
        "batch capacity retained");
  check(batch_result.find("allocated_bytes_peak")->is_nil(), "batch peak null");

  NativeRequestTelemetryConfig concurrent = config("req-concurrent");
  concurrent.execution = NativeTelemetryExecution::kConcurrent;
  NativeRequestTelemetry concurrent_scope(std::move(concurrent));
  check(text(concurrent_scope.finish("completed"), "reason") ==
            "concurrent_request_attribution_unavailable",
        "concurrent stages fail closed");

  NativeRequestTelemetryConfig unsupported = config("req-external");
  unsupported.instrumented = false;
  unsupported.unsupported_reason = "backend_not_instrumented";
  NativeRequestTelemetry unsupported_scope(std::move(unsupported));
  Value unsupported_result = unsupported_scope.finish("completed");
  check(text(unsupported_result, "coverage") == "NOT_COVERED",
        "unsupported backend not covered");
  check(unsupported_result.find("reserved_bytes_peak")->is_nil(),
        "unsupported bytes null");
}

void test_sanitization_redaction_and_bounded_envelope() {
  NativeRequestTelemetry scope(config("req-sanitize"));
  umminfer::obs::note_native_allocation(fake_pointer(0x7000), 64);
  Value raw = scope.finish("completed");
  umminfer::obs::note_native_free(fake_pointer(0x7000));
  raw["model_path"] = Value("/secret/internal/model");
  raw["unexpected"] = Value(std::string(10000, 'x'));
  Value clean = umminfer::obs::sanitize_native_request_telemetry(
      &raw, "req-sanitize", "vlm", "completed");
  check(text(clean, "coverage") == "COVERED", "valid metrics survive allow-list");
  check(clean.find("model_path") == nullptr, "model path removed");
  check(clean.find("unexpected") == nullptr, "unknown long field removed");
  check(text(clean, "stage") == "vlm", "public stage attached");
  check(text(clean, "native_stage") == "native-test", "native stage retained");
  check(umminfer::pack(clean).size() < 2048, "single stage serialization bounded");

  Value malformed = raw;
  (*mutable_field(malformed, "allocated_bytes_peak")) = Value(int64_t{-1});
  Value rejected = umminfer::obs::sanitize_native_request_telemetry(
      &malformed, "req-sanitize", "vlm", "completed");
  check(text(rejected, "coverage") == "NOT_COVERED", "negative bytes rejected");
  check(text(rejected, "reason") == "malformed_native_telemetry",
        "malformed reason fixed");
  check(rejected.find("model_path") == nullptr, "malformed path not leaked");

  Value wrong_id = umminfer::obs::sanitize_native_request_telemetry(
      &raw, "req-other", "vlm", "completed");
  check(text(wrong_id, "coverage") == "NOT_COVERED", "request mismatch rejected");

  std::vector<Value> stages;
  for (size_t i = 0; i < 65; ++i) stages.push_back(clean);
  Value envelope = umminfer::obs::request_telemetry_envelope(
      "req-sanitize", "completed", std::move(stages));
  check(envelope.find("stages")->as_array().size() == 64, "stage array capped");
  check(mutable_field(envelope, "sample_coverage")->find("truncated")->as_bool(),
        "truncation explicit");
}

OutgoingMessage run_scheduler(SimpleScheduler& scheduler, IncomingMessage request) {
  std::thread worker([&scheduler] { scheduler.start(); });
  scheduler.enqueue(std::move(request));
  auto result = scheduler.outbox().get(5.0);
  scheduler.stop();
  worker.join();
  check(result.has_value(), "scheduler terminal result timeout");
  return std::move(*result);
}

void test_scheduler_default_success_failure_and_cancel() {
  umminfer::stage::StageOptions options;
  options.gpu = 0;
  const Value original = Value(Value::Map{{"content", Value("ok")}});
  auto success = umminfer::stage::with_native_request_telemetry(
      [](const Value&, const AbortCheck&) {
        umminfer::obs::note_native_allocation(fake_pointer(0x8000), 80);
        return Value(Value::Map{{"content", Value("ok")}});
      },
      options, "native-test", true);
  {
    SimpleScheduler scheduler(nullptr);
    scheduler.set_cancellable_compute(success);
    OutgoingMessage out = run_scheduler(
        scheduler, IncomingMessage{"req-default", "new_request", Value(), {}});
    check(out.type == "result", "default result");
    check(umminfer::pack(out.data) == umminfer::pack(original),
          "default protocol bytes identical");
  }
  {
    SimpleScheduler scheduler(nullptr);
    scheduler.set_cancellable_compute(success);
    OutgoingMessage out = run_scheduler(
        scheduler,
        IncomingMessage{"req-success", "new_request", Value(),
                        {{umminfer::obs::kNativeAllocatorTelemetryParam, Value(true)}}});
    check(out.type == "result", "covered result");
    const Value* telemetry = out.data.find(umminfer::obs::kRequestTelemetryResultKey);
    check(telemetry != nullptr && text(*telemetry, "lifecycle_status") == "completed",
          "success telemetry attached");
    check(integer(*telemetry, "allocated_bytes_peak") == 80, "scheduler peak 80");
    umminfer::obs::note_native_free(fake_pointer(0x8000));
  }

  auto failure = umminfer::stage::with_native_request_telemetry(
      [](const Value&, const AbortCheck&) -> Value {
        umminfer::obs::note_native_allocation(fake_pointer(0x9000), 90);
        throw std::runtime_error("injected failure");
      },
      options, "native-test", true);
  {
    SimpleScheduler scheduler(nullptr);
    scheduler.set_cancellable_compute(failure);
    OutgoingMessage out = run_scheduler(
        scheduler,
        IncomingMessage{"req-failure", "new_request", Value(),
                        {{umminfer::obs::kNativeAllocatorTelemetryParam, Value(true)}}});
    check(out.type == "error", "failure remains error");
    check(text(out.data, "error") == "injected failure", "failure text retained");
    const Value* telemetry = out.data.find(umminfer::obs::kRequestTelemetryResultKey);
    check(telemetry != nullptr && text(*telemetry, "lifecycle_status") == "failed",
          "failed telemetry retained");
    check(integer(*telemetry, "allocated_bytes_peak") == 90, "failed peak 90");
    umminfer::obs::note_native_free(fake_pointer(0x9000));
  }

  std::mutex mutex;
  std::condition_variable cv;
  bool started = false;
  auto cancelled = umminfer::stage::with_native_request_telemetry(
      [&](const Value&, const AbortCheck& aborted) -> Value {
        umminfer::obs::note_native_allocation(fake_pointer(0xa000), 110);
        std::unique_lock<std::mutex> lock(mutex);
        started = true;
        cv.notify_all();
        while (!aborted()) cv.wait_for(lock, std::chrono::milliseconds(2));
        throw RequestAborted("injected cancel");
      },
      options, "native-test", true);
  SimpleScheduler scheduler(nullptr);
  scheduler.set_cancellable_compute(cancelled);
  std::thread worker([&scheduler] { scheduler.start(); });
  scheduler.enqueue(
      IncomingMessage{"req-cancel", "new_request", Value(),
                      {{umminfer::obs::kNativeAllocatorTelemetryParam, Value(true)}}});
  {
    std::unique_lock<std::mutex> lock(mutex);
    check(cv.wait_for(lock, std::chrono::seconds(5), [&] { return started; }),
          "cancel compute started");
  }
  scheduler.abort("req-cancel");
  auto out = scheduler.outbox().get(5.0);
  scheduler.stop();
  worker.join();
  check(out.has_value() && out->type == "cancelled", "cancel terminal retained");
  const Value* telemetry = out->data.find(umminfer::obs::kRequestTelemetryResultKey);
  check(telemetry != nullptr && text(*telemetry, "lifecycle_status") == "cancelled",
        "cancel status accurate");
  check(integer(*telemetry, "allocated_bytes_peak") == 110, "cancel peak retained");
  umminfer::obs::note_native_free(fake_pointer(0xa000));
}

}  // namespace

int main() {
  try {
    test_serial_epochs_and_retained_free_isolation();
    test_overlap_batch_and_unsupported_are_not_covered();
    test_sanitization_redaction_and_bounded_envelope();
    test_scheduler_default_success_failure_and_cancel();
    std::cout << "request_telemetry_test: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "request_telemetry_test: FAIL: " << error.what() << "\n";
    return 1;
  }
}
