// SPDX-License-Identifier: Apache-2.0
// Process-local metrics, GPU sampling and request tracing.
//
// Dependency-free by design: the Prometheus text form is assembled here and
// NVML is loaded with dlopen. A host without libnvidia-ml still serves the same
// endpoint and reports umm_gpu_nvml_available 0.
//
// No metric label carries request content. Labels are limited to stage, leg,
// request type, status and source, all of which are fixed identifier sets.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "umminfer/proto/wire.hpp"

namespace umminfer::obs {

// Fixed-bucket histogram in Prometheus cumulative form. Buckets are in seconds.
struct Histogram {
  static constexpr size_t kBuckets = 14;
  static const std::array<double, kBuckets>& bounds();

  void observe(double value);
  void merge(const Histogram& other);
  // Appends the _bucket, _count and _sum samples. labels is already escaped
  // and may be empty.
  void render(const std::string& name, const std::string& labels,
              std::string& out) const;

  std::array<uint64_t, kBuckets> cumulative{};
  uint64_t count = 0;
  double sum = 0;
};

struct GpuSample {
  unsigned index = 0;
  std::string name;
  double utilization_percent = 0;
  double memory_used_bytes = 0;
  double memory_total_bytes = 0;
  double temperature_celsius = 0;
  double power_watts = 0;
  double sm_clock_mhz = 0;
  double memory_clock_mhz = 0;
};

// Optional NVML client. available() is false when libnvidia-ml cannot be
// loaded or lacks a required symbol; sample() then returns an empty vector.
// Every call is read-only: this class never changes a device setting.
class Nvml {
 public:
  // library_name overrides the driver library. The test passes a name that does
  // not exist to exercise the degrade path on a host that has NVML.
  explicit Nvml(const char* library_name = nullptr);
  ~Nvml();
  Nvml(const Nvml&) = delete;
  Nvml& operator=(const Nvml&) = delete;

  bool available() const { return available_; }
  std::vector<GpuSample> sample();

 private:
  void* library_ = nullptr;
  bool available_ = false;
  bool initialized_ = false;
};

// One structured request record. It carries identifiers, a stage name, two
// epoch timestamps, a terminal status and leg durations. It has no field that
// can hold prompt text, request bodies or credentials.
struct TraceLegSnapshot {
  std::string modality;
  std::string stage;
  std::string state;
  std::string error;
};

struct TraceRecord {
  std::string request_id;
  std::string stage;
  std::string request_type;
  int64_t start_epoch_ms = 0;
  int64_t end_epoch_ms = 0;
  std::string status;
  double duration_ms = 0;
  // Negative means the leg did not run for this request.
  double asr_ms = -1;
  double vlm_ms = -1;
  double image_ms = -1;
  double tts_ms = -1;
  double video_ms = -1;
};

// Process-wide registry. The server is one process, so one instance is the
// whole scrape surface.
class Metrics {
 public:
  static Metrics& instance();

  // Coordinator view of a stage: submissions, in-flight and terminal states.
  void stage_submit(const std::string& stage, const std::string& request_type);
  void stage_finish(const std::string& stage, const std::string& request_type,
                    const std::string& status, double seconds);
  void stage_in_flight(const std::string& stage, int delta);
  // Admission or scheduling queue depth, updated on every state change.
  void queue_depth(const std::string& queue, size_t depth);

  // Engine-level end to end: submit to terminal.
  void request(const std::string& request_type, const std::string& status,
               double seconds);
  // One leg of a unified request.
  void leg(const std::string& leg_name, const std::string& request_type,
           const std::string& status, double seconds);
  // source: client, timeout or shutdown.
  void abort(const std::string& source, double latency_seconds);
  // phase: prefill or decode.
  void vlm_phase(const char* phase, double seconds);
  // phase: first_frame or total.
  void tts_phase(const char* phase, double seconds);
  void image_denoise_step(double seconds);
  // HTTP boundary counter: route and status code only.
  void http(const std::string& route, const std::string& status);

  bool trace_enabled();
  void trace_start(const std::string& request_id,
                   const std::string& request_type, const std::string& stage,
                   int64_t start_epoch_ms);
  void trace_leg(const std::string& request_id, const std::string& leg_name,
                 const std::string& status, double duration_ms);
  void trace_legs_snapshot(const std::string& request_id,
                           const std::string& phase,
                           const std::vector<TraceLegSnapshot>& legs);
  void trace(const TraceRecord& record);

  // Full Prometheus text exposition, including a live GPU sample.
  std::string render();

 private:
  Metrics() = default;
  void write_trace_line(const std::string& line);

  struct StageState {
    uint64_t in_flight = 0;
    std::map<std::string, uint64_t> requests;  // status -> count
    std::map<std::string, Histogram> duration;  // request_type -> histogram
  };

  std::mutex mu_;
  std::map<std::string, StageState> stages_;
  std::map<std::string, uint64_t> requests_;  // "request_type|status"
  std::map<std::string, Histogram> request_duration_;
  std::map<std::string, Histogram> leg_duration_;
  std::map<std::string, std::string> leg_status_;  // "leg|type|status" -> ""
  std::map<std::string, uint64_t> leg_total_;
  std::map<std::string, uint64_t> queue_depth_;
  std::map<std::string, uint64_t> aborts_;
  Histogram abort_latency_;
  Histogram vlm_prefill_;
  Histogram vlm_decode_;
  Histogram tts_first_frame_;
  Histogram tts_total_;
  Histogram image_denoise_step_;
  std::map<std::string, uint64_t> http_;  // "route|status"

  Nvml nvml_;
  std::string trace_path_;
  std::string trace_open_path_;
  std::mutex trace_mu_;
  std::FILE* trace_file_ = nullptr;
  size_t trace_bytes_ = 0;
};

// The value a stage result carries under key, in milliseconds. Returns a
// negative value when the key is absent or not numeric, so a caller never
// records a fabricated zero.
double result_ms(const struct Value& result, const char* key);

}  // namespace umminfer::obs
