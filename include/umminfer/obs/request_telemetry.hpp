// SPDX-License-Identifier: Apache-2.0
// Opt-in request-scoped native allocator telemetry.
//
// This is deliberately separate from whole-device NVML observations. A covered
// serial epoch accounts only for explicit native allocations first owned by that
// request. It never subtracts a resident model baseline or another request.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "umminfer/proto/wire.hpp"

namespace umminfer::obs {

inline constexpr const char* kRequestTelemetryResultKey = "request_telemetry";
inline constexpr const char* kNativeAllocatorTelemetryParam =
    "native_allocator_telemetry";
inline constexpr int64_t kRequestTelemetrySchemaVersion = 1;
inline constexpr size_t kMaxRequestTelemetryStages = 64;

bool native_allocator_telemetry_requested(const Value::Map& params);

enum class NativeTelemetryExecution { kSerial, kConcurrent, kBatched };

struct NativeRequestTelemetryConfig {
  std::string request_id;
  std::string stage;
  std::string backend;
  int device_index = -1;
  NativeTelemetryExecution execution = NativeTelemetryExecution::kSerial;
  int batch_capacity = 1;
  // True only when every explicit native runtime allocation in this backend is
  // routed through note_native_allocation/note_native_free.
  bool instrumented = false;
  std::string unsupported_reason;
};

// One request owns one epoch. Construction never resets CUDA's process/global
// counters. Serial epochs instead start a new allocation-ownership epoch with a
// monotonically increasing id. Overlapping serial scopes poison both attributions
// rather than reset or divide a shared high-water mark.
class NativeRequestTelemetry {
 public:
  explicit NativeRequestTelemetry(NativeRequestTelemetryConfig config);
  ~NativeRequestTelemetry();
  NativeRequestTelemetry(const NativeRequestTelemetry&) = delete;
  NativeRequestTelemetry& operator=(const NativeRequestTelemetry&) = delete;

  // Idempotent. lifecycle_status is completed, failed, or cancelled. Covered
  // cancelled/failed requests retain the peak observed before that terminal.
  Value finish(const char* lifecycle_status);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Allocation hooks used by the CUDA wrapper and by host fake-allocator tests.
// They never dereference ptr. Calls outside a covered serial request epoch are
// no-ops. A freed block updates its original owner, even on another thread or
// after that owner's terminal snapshot; it can never affect the current request.
void note_native_allocation(void* ptr, uint64_t bytes) noexcept;
void note_native_free(void* ptr) noexcept;
size_t native_tracked_allocation_count_for_test();

// Projects a stage result through a strict allow-list. Expected identity and
// lifecycle come from the Coordinator, not from untrusted result bytes. Unknown
// fields are dropped; malformed values become a fixed NOT_COVERED record.
Value sanitize_native_request_telemetry(
    const Value* candidate, const std::string& expected_request_id,
    const std::string& expected_stage, const char* expected_lifecycle);

// Builds the public request envelope from already sanitized stage entries.
// Missing entries remain NOT_COVERED. No whole-device sample is folded in.
Value request_telemetry_envelope(const std::string& request_id,
                                 const char* lifecycle_status,
                                 std::vector<Value> stages);

// Single-stage HTTP helper. raw_result may be nil or lack telemetry.
Value request_telemetry_from_result(const Value& raw_result,
                                    const std::string& request_id,
                                    const std::string& stage,
                                    const char* lifecycle_status);

// Changes only an already present telemetry payload, for the race where compute
// finishes while its abort is being committed. Returns false when absent.
bool rewrite_result_telemetry_lifecycle(Value& result, const char* lifecycle);

}  // namespace umminfer::obs
