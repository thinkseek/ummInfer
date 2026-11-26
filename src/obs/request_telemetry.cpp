// SPDX-License-Identifier: Apache-2.0
#include "umminfer/obs/request_telemetry.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <string_view>
#include <utility>

namespace umminfer::obs {
namespace {

constexpr const char* kScope = "request-native-cuda-runtime";
constexpr const char* kAttribution =
    "explicit native cudaMalloc/cudaFree blocks first owned by one serial request "
    "epoch; excludes pre-request resident allocations, CUDA context/driver and "
    "external-library allocations; no baseline or peer-request subtraction";

struct EpochState {
  uint64_t id = 0;
  bool covered = true;
  std::string reason;
  uint64_t current_bytes = 0;
  uint64_t peak_bytes = 0;
  uint64_t allocation_events = 0;
  uint64_t free_events = 0;
};

struct AllocationOwner {
  std::shared_ptr<EpochState> epoch;
  uint64_t bytes = 0;
};

std::mutex g_mu;
std::map<void*, AllocationOwner> g_allocations;
std::map<uint64_t, std::shared_ptr<EpochState>> g_active_serial;
std::atomic<uint64_t> g_next_epoch{0};
std::atomic<size_t> g_tracked_count{0};
thread_local std::shared_ptr<EpochState> g_thread_epoch;

bool valid_identity(std::string_view value) {
  if (value.empty() || value.size() > 160) return false;
  for (const unsigned char c : value) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') || c == '-' || c == '_' || c == ':' ||
        c == '.' || c == '@') {
      continue;
    }
    return false;
  }
  return true;
}

bool valid_backend(std::string_view value) {
  return valid_identity(value) && value.size() <= 64 &&
         value.find(':') == std::string_view::npos &&
         value.find('@') == std::string_view::npos;
}

bool valid_reason(std::string_view value) {
  static const std::set<std::string_view> kReasons = {
      "backend_not_instrumented",
      "contract_stub_has_no_cuda_allocator",
      "concurrent_request_attribution_unavailable",
      "batched_request_attribution_unavailable",
      "missing_request_identity",
      "invalid_device_identity",
      "overlapping_serial_epochs",
      "allocation_tracking_failed",
      "stage_terminal_snapshot_missing",
      "malformed_native_telemetry",
  };
  return kReasons.count(value) != 0;
}

const char* execution_name(NativeTelemetryExecution execution) {
  switch (execution) {
    case NativeTelemetryExecution::kSerial:
      return "serial";
    case NativeTelemetryExecution::kConcurrent:
      return "concurrent";
    case NativeTelemetryExecution::kBatched:
      return "batched";
  }
  return "unknown";
}

Value integer_or_null(bool present, uint64_t value) {
  if (!present || value > static_cast<uint64_t>(
                              std::numeric_limits<int64_t>::max())) {
    return Value();
  }
  return Value(static_cast<int64_t>(value));
}

Value uncovered_stage(const std::string& request_id, const std::string& stage,
                      const char* lifecycle, const std::string& reason) {
  Value::Map out;
  out["schema_version"] = Value(kRequestTelemetrySchemaVersion);
  out["coverage"] = Value("NOT_COVERED");
  out["scope"] = Value(kScope);
  out["request_id"] = Value(valid_identity(request_id) ? request_id : "redacted");
  out["stage"] = Value(valid_identity(stage) ? stage : "unknown");
  out["native_stage"] = Value("unknown");
  out["backend"] = Value("unknown");
  out["device_index"] = Value();
  out["lifecycle_status"] = Value(lifecycle);
  out["execution"] = Value(Value::Map{{"mode", Value("unknown")},
                                      {"batch_capacity", Value()}});
  out["epoch"] = Value(Value::Map{{"id", Value()},
                                   {"reset", Value(false)},
                                   {"semantics", Value("no_request_epoch")}});
  for (const char* key : {"allocated_bytes_peak", "reserved_bytes_peak",
                          "allocated_bytes_final", "reserved_bytes_final"}) {
    out[key] = Value();
  }
  out["sample_coverage"] =
      Value(Value::Map{{"allocation_events", Value(int64_t{0})},
                       {"free_events", Value(int64_t{0})},
                       {"terminal_snapshots", Value(int64_t{0})}});
  out["reason"] = Value(valid_reason(reason) ? reason
                                             : "malformed_native_telemetry");
  out["attribution"] = Value(kAttribution);
  return Value(std::move(out));
}

bool read_int(const Value& map, const char* key, int64_t minimum, int64_t* out) {
  const Value* value = map.find(key);
  if (value == nullptr || !value->is_int() || value->as_int() < minimum) {
    return false;
  }
  *out = value->as_int();
  return true;
}

bool read_string(const Value& map, const char* key, std::string* out) {
  const Value* value = map.find(key);
  if (value == nullptr || !value->is_str()) return false;
  *out = value->as_str();
  return true;
}

bool read_nonnegative_counter(const Value& map, const char* key, int64_t* out) {
  return read_int(map, key, 0, out);
}

}  // namespace

bool native_allocator_telemetry_requested(const Value::Map& params) {
  const auto it = params.find(kNativeAllocatorTelemetryParam);
  return it != params.end() && it->second.is_bool() && it->second.as_bool();
}

struct NativeRequestTelemetry::Impl {
  NativeRequestTelemetryConfig config;
  uint64_t epoch_id = ++g_next_epoch;
  std::shared_ptr<EpochState> epoch;
  bool reset = false;
  bool finished = false;
  std::string reason;
  Value snapshot;

  Impl(NativeRequestTelemetryConfig value) : config(std::move(value)) {
    if (!valid_identity(config.request_id) || !valid_identity(config.stage)) {
      reason = "missing_request_identity";
    } else if (!config.instrumented) {
      reason = config.unsupported_reason.empty()
                   ? "backend_not_instrumented"
                   : config.unsupported_reason;
      if (!valid_reason(reason)) reason = "backend_not_instrumented";
    } else if (config.device_index < 0 || config.device_index > 1024) {
      reason = "invalid_device_identity";
    } else if (config.execution == NativeTelemetryExecution::kBatched) {
      reason = "batched_request_attribution_unavailable";
    } else if (config.execution == NativeTelemetryExecution::kConcurrent) {
      reason = "concurrent_request_attribution_unavailable";
    }
    if (!reason.empty()) return;

    epoch = std::make_shared<EpochState>();
    epoch->id = epoch_id;
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_active_serial.empty() || g_thread_epoch) {
      // Never reset a shared allocator under another active request. Poison the
      // existing epoch too: its supposedly exclusive interval has ended.
      for (auto& [id, active] : g_active_serial) {
        (void)id;
        active->covered = false;
        active->reason = "overlapping_serial_epochs";
      }
      if (g_thread_epoch) {
        g_thread_epoch->covered = false;
        g_thread_epoch->reason = "overlapping_serial_epochs";
      }
      reason = "overlapping_serial_epochs";
      epoch->covered = false;
      epoch->reason = reason;
      return;
    }
    reset = true;
    g_active_serial.emplace(epoch_id, epoch);
    g_thread_epoch = epoch;
  }
};

NativeRequestTelemetry::NativeRequestTelemetry(NativeRequestTelemetryConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

NativeRequestTelemetry::~NativeRequestTelemetry() {
  if (!impl_->finished) (void)finish("failed");
}

Value NativeRequestTelemetry::finish(const char* lifecycle_status) {
  if (impl_->finished) return impl_->snapshot;
  impl_->finished = true;
  const std::string lifecycle =
      std::string_view(lifecycle_status) == "completed" ||
              std::string_view(lifecycle_status) == "failed" ||
              std::string_view(lifecycle_status) == "cancelled"
          ? lifecycle_status
          : "failed";

  if (impl_->epoch) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_active_serial.erase(impl_->epoch_id);
    if (g_thread_epoch == impl_->epoch) g_thread_epoch.reset();
  }
  if (impl_->reason.empty() && impl_->epoch && !impl_->epoch->covered) {
    impl_->reason = impl_->epoch->reason.empty()
                        ? "allocation_tracking_failed"
                        : impl_->epoch->reason;
  }
  if (!impl_->reason.empty() || !impl_->epoch) {
    Value value = uncovered_stage(impl_->config.request_id, impl_->config.stage,
                                  lifecycle.c_str(),
                                  impl_->reason.empty() ? "allocation_tracking_failed"
                                                        : impl_->reason);
    Value::Map out = value.as_map();
    out["backend"] = Value(valid_backend(impl_->config.backend)
                               ? impl_->config.backend
                               : "unknown");
    out["device_index"] = Value(static_cast<int64_t>(impl_->config.device_index));
    out["execution"] = Value(Value::Map{
        {"mode", Value(execution_name(impl_->config.execution))},
        {"batch_capacity", Value(static_cast<int64_t>(
                                  std::max(impl_->config.batch_capacity, 1)))}});
    out["epoch"] = Value(Value::Map{
        {"id", Value(static_cast<int64_t>(impl_->epoch_id))},
        {"reset", Value(impl_->reset)},
        {"semantics", Value(impl_->reset ? "new_allocation_ownership_epoch"
                                         : "not_reset_shared_or_unsupported")}});
    impl_->snapshot = Value(std::move(out));
    return impl_->snapshot;
  }

  std::lock_guard<std::mutex> lock(g_mu);
  const auto& epoch = *impl_->epoch;
  const bool representable =
      epoch.current_bytes <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) &&
      epoch.peak_bytes <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) &&
      epoch.allocation_events <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) &&
      epoch.free_events <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  if (!representable) {
    impl_->snapshot = uncovered_stage(impl_->config.request_id, impl_->config.stage,
                                      lifecycle.c_str(), "allocation_tracking_failed");
    return impl_->snapshot;
  }
  Value::Map out;
  out["schema_version"] = Value(kRequestTelemetrySchemaVersion);
  out["coverage"] = Value("COVERED");
  out["scope"] = Value(kScope);
  out["request_id"] = Value(impl_->config.request_id);
  out["stage"] = Value(impl_->config.stage);
  out["backend"] = Value(impl_->config.backend);
  out["device_index"] = Value(static_cast<int64_t>(impl_->config.device_index));
  out["lifecycle_status"] = Value(lifecycle);
  out["execution"] = Value(Value::Map{{"mode", Value("serial")},
                                      {"batch_capacity", Value(int64_t{1})}});
  out["epoch"] = Value(Value::Map{
      {"id", Value(static_cast<int64_t>(impl_->epoch_id))},
      {"reset", Value(true)},
      {"semantics", Value("new_allocation_ownership_epoch")}});
  out["allocated_bytes_peak"] = Value(static_cast<int64_t>(epoch.peak_bytes));
  out["reserved_bytes_peak"] = Value(static_cast<int64_t>(epoch.peak_bytes));
  out["allocated_bytes_final"] = Value(static_cast<int64_t>(epoch.current_bytes));
  out["reserved_bytes_final"] = Value(static_cast<int64_t>(epoch.current_bytes));
  out["sample_coverage"] = Value(Value::Map{
      {"allocation_events", Value(static_cast<int64_t>(epoch.allocation_events))},
      {"free_events", Value(static_cast<int64_t>(epoch.free_events))},
      {"terminal_snapshots", Value(int64_t{1})}});
  out["attribution"] = Value(kAttribution);
  impl_->snapshot = Value(std::move(out));
  return impl_->snapshot;
}

void note_native_allocation(void* ptr, uint64_t bytes) noexcept {
  if (ptr == nullptr || !g_thread_epoch) return;
  try {
    std::lock_guard<std::mutex> lock(g_mu);
    auto epoch = g_thread_epoch;
    if (!epoch || !epoch->covered) return;
    if (g_allocations.count(ptr) != 0 ||
        epoch->current_bytes > std::numeric_limits<uint64_t>::max() - bytes) {
      epoch->covered = false;
      epoch->reason = "allocation_tracking_failed";
      return;
    }
    g_allocations.emplace(ptr, AllocationOwner{epoch, bytes});
    ++g_tracked_count;
    ++epoch->allocation_events;
    epoch->current_bytes += bytes;
    epoch->peak_bytes = std::max(epoch->peak_bytes, epoch->current_bytes);
  } catch (...) {
    if (g_thread_epoch) {
      g_thread_epoch->covered = false;
      g_thread_epoch->reason = "allocation_tracking_failed";
    }
  }
}

void note_native_free(void* ptr) noexcept {
  if (ptr == nullptr || g_tracked_count.load(std::memory_order_acquire) == 0) return;
  try {
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_allocations.find(ptr);
    if (it == g_allocations.end()) return;
    AllocationOwner owner = std::move(it->second);
    g_allocations.erase(it);
    --g_tracked_count;
    if (owner.bytes > owner.epoch->current_bytes) {
      owner.epoch->covered = false;
      owner.epoch->reason = "allocation_tracking_failed";
      owner.epoch->current_bytes = 0;
    } else {
      owner.epoch->current_bytes -= owner.bytes;
    }
    ++owner.epoch->free_events;
  } catch (...) {
    // Telemetry must never turn a successful cudaFree into a model failure.
  }
}

size_t native_tracked_allocation_count_for_test() {
  return g_tracked_count.load(std::memory_order_acquire);
}

Value sanitize_native_request_telemetry(const Value* candidate,
                                        const std::string& expected_request_id,
                                        const std::string& expected_stage,
                                        const char* expected_lifecycle) {
  const auto malformed = [&] {
    return uncovered_stage(expected_request_id, expected_stage, expected_lifecycle,
                           candidate == nullptr ? "stage_terminal_snapshot_missing"
                                                : "malformed_native_telemetry");
  };
  if (candidate == nullptr) return malformed();
  if (!candidate->is_map()) return malformed();
  const Value& in = *candidate;
  int64_t schema = 0;
  std::string coverage, scope, request_id, stage, backend, lifecycle;
  if (!read_int(in, "schema_version", 1, &schema) ||
      schema != kRequestTelemetrySchemaVersion ||
      !read_string(in, "coverage", &coverage) ||
      (coverage != "COVERED" && coverage != "NOT_COVERED") ||
      !read_string(in, "scope", &scope) || scope != kScope ||
      !read_string(in, "request_id", &request_id) ||
      request_id != expected_request_id || !valid_identity(request_id) ||
      !read_string(in, "stage", &stage) || !valid_identity(stage) ||
      !read_string(in, "backend", &backend) || !valid_backend(backend) ||
      !read_string(in, "lifecycle_status", &lifecycle) ||
      lifecycle != expected_lifecycle) {
    return malformed();
  }
  const Value* execution = in.find("execution");
  const Value* epoch = in.find("epoch");
  const Value* samples = in.find("sample_coverage");
  if (execution == nullptr || !execution->is_map() || epoch == nullptr ||
      !epoch->is_map() || samples == nullptr || !samples->is_map()) {
    return malformed();
  }
  std::string mode;
  int64_t batch_capacity = 0;
  if (!read_string(*execution, "mode", &mode) ||
      (mode != "serial" && mode != "concurrent" && mode != "batched") ||
      !read_int(*execution, "batch_capacity", 1, &batch_capacity) ||
      batch_capacity > 4096) {
    return malformed();
  }
  const Value* reset = epoch->find("reset");
  int64_t epoch_id = 0;
  std::string reset_semantics;
  if (reset == nullptr || !reset->is_bool() ||
      !read_int(*epoch, "id", 1, &epoch_id) ||
      !read_string(*epoch, "semantics", &reset_semantics) ||
      (reset_semantics != "new_allocation_ownership_epoch" &&
       reset_semantics != "not_reset_shared_or_unsupported")) {
    return malformed();
  }
  int64_t allocation_events = 0, free_events = 0, terminal_snapshots = 0;
  if (!read_nonnegative_counter(*samples, "allocation_events", &allocation_events) ||
      !read_nonnegative_counter(*samples, "free_events", &free_events) ||
      !read_nonnegative_counter(*samples, "terminal_snapshots", &terminal_snapshots) ||
      free_events > allocation_events || terminal_snapshots > 1) {
    return malformed();
  }

  Value::Map out;
  out["schema_version"] = Value(kRequestTelemetrySchemaVersion);
  out["coverage"] = Value(coverage);
  out["scope"] = Value(kScope);
  out["request_id"] = Value(expected_request_id);
  out["stage"] = Value(expected_stage);
  out["native_stage"] = Value(stage);
  out["backend"] = Value(backend);
  out["lifecycle_status"] = Value(expected_lifecycle);
  out["execution"] = Value(Value::Map{
      {"mode", Value(mode)}, {"batch_capacity", Value(batch_capacity)}});
  out["epoch"] = Value(Value::Map{{"id", Value(epoch_id)},
                                   {"reset", Value(reset->as_bool())},
                                   {"semantics", Value(reset_semantics)}});
  out["sample_coverage"] = Value(Value::Map{
      {"allocation_events", Value(allocation_events)},
      {"free_events", Value(free_events)},
      {"terminal_snapshots", Value(terminal_snapshots)}});
  out["attribution"] = Value(kAttribution);

  if (coverage == "NOT_COVERED") {
    std::string reason;
    if (!read_string(in, "reason", &reason) || !valid_reason(reason)) {
      return malformed();
    }
    for (const char* key : {"allocated_bytes_peak", "reserved_bytes_peak",
                            "allocated_bytes_final", "reserved_bytes_final"}) {
      const Value* value = in.find(key);
      if (value == nullptr || !value->is_nil()) return malformed();
      out[key] = Value();
    }
    if (allocation_events != 0 || free_events != 0 || terminal_snapshots != 0) {
      return malformed();
    }
    out["device_index"] = Value();
    out["reason"] = Value(reason);
    return Value(std::move(out));
  }

  int64_t device = -1, allocated_peak = -1, reserved_peak = -1;
  int64_t allocated_final = -1, reserved_final = -1;
  if (mode != "serial" || batch_capacity != 1 || !reset->as_bool() ||
      reset_semantics != "new_allocation_ownership_epoch" ||
      !read_int(in, "device_index", 0, &device) || device > 1024 ||
      !read_nonnegative_counter(in, "allocated_bytes_peak", &allocated_peak) ||
      !read_nonnegative_counter(in, "reserved_bytes_peak", &reserved_peak) ||
      !read_nonnegative_counter(in, "allocated_bytes_final", &allocated_final) ||
      !read_nonnegative_counter(in, "reserved_bytes_final", &reserved_final) ||
      allocated_peak != reserved_peak || allocated_final != reserved_final ||
      allocated_final > allocated_peak || terminal_snapshots != 1) {
    return malformed();
  }
  out["device_index"] = Value(device);
  out["allocated_bytes_peak"] = Value(allocated_peak);
  out["reserved_bytes_peak"] = Value(reserved_peak);
  out["allocated_bytes_final"] = Value(allocated_final);
  out["reserved_bytes_final"] = Value(reserved_final);
  return Value(std::move(out));
}

Value request_telemetry_envelope(const std::string& request_id,
                                 const char* lifecycle_status,
                                 std::vector<Value> stages) {
  const bool truncated = stages.size() > kMaxRequestTelemetryStages;
  if (truncated) stages.resize(kMaxRequestTelemetryStages);
  size_t covered = 0;
  Value::Array entries;
  entries.reserve(stages.size());
  for (auto& stage : stages) {
    if (!stage.is_map()) continue;
    const Value* coverage = stage.find("coverage");
    if (coverage != nullptr && coverage->is_str() && coverage->as_str() == "COVERED") {
      ++covered;
    }
    entries.push_back(std::move(stage));
  }
  const std::string coverage = entries.empty() || covered == 0
                                   ? "NOT_COVERED"
                                   : covered == entries.size() ? "COVERED" : "PARTIAL";
  Value::Map out;
  out["schema_version"] = Value(kRequestTelemetrySchemaVersion);
  out["coverage"] = Value(coverage);
  out["scope"] = Value("request-native-allocator");
  out["request_id"] = Value(valid_identity(request_id) ? request_id : "redacted");
  out["lifecycle_status"] = Value(lifecycle_status);
  out["sample_coverage"] = Value(Value::Map{
      {"stage_count", Value(static_cast<int64_t>(entries.size()))},
      {"covered_stage_count", Value(static_cast<int64_t>(covered))},
      {"truncated", Value(truncated)}});
  out["stages"] = Value(std::move(entries));
  out["whole_device_nvml"] = Value("separate_measurement_resources");
  return Value(std::move(out));
}

Value request_telemetry_from_result(const Value& raw_result,
                                    const std::string& request_id,
                                    const std::string& stage,
                                    const char* lifecycle_status) {
  const Value* candidate = raw_result.is_map()
                               ? raw_result.find(kRequestTelemetryResultKey)
                               : nullptr;
  std::vector<Value> stages;
  stages.push_back(sanitize_native_request_telemetry(
      candidate, request_id, stage, lifecycle_status));
  return request_telemetry_envelope(request_id, lifecycle_status,
                                    std::move(stages));
}

bool rewrite_result_telemetry_lifecycle(Value& result, const char* lifecycle) {
  if (!result.is_map()) return false;
  Value& telemetry = result[kRequestTelemetryResultKey];
  if (!telemetry.is_map()) return false;
  Value::Map copy = telemetry.as_map();
  copy["lifecycle_status"] = Value(lifecycle);
  telemetry = Value(std::move(copy));
  return true;
}

}  // namespace umminfer::obs
