// SPDX-License-Identifier: Apache-2.0
// Metrics aggregation, Prometheus rendering, NVML degradation and tracing.
#include "umminfer/obs/metrics.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "umminfer/proto/wire.hpp"

namespace {

int g_failures = 0;

void check(bool condition, const char* expression, int line) {
  if (condition) return;
  std::fprintf(stderr, "FAIL line %d: %s\n", line, expression);
  ++g_failures;
}

#define CHECK(expr) check((expr), #expr, __LINE__)

bool contains(const std::string& text, const std::string& needle) {
  return text.find(needle) != std::string::npos;
}

std::string read_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}

long file_size(const std::string& path) {
  struct stat info{};
  if (::stat(path.c_str(), &info) != 0) return -1;
  return static_cast<long>(info.st_size);
}

void test_histogram() {
  umminfer::obs::Histogram histogram;
  histogram.observe(0.002);
  histogram.observe(0.3);
  histogram.observe(100.0);

  CHECK(histogram.count == 3);
  CHECK(histogram.sum > 100.29 && histogram.sum < 100.31);
  // Cumulative buckets: 0.002 lands in le=0.005, 0.3 in le=0.5, 100 is +Inf
  // only. A bucket counts every sample at or below its bound.
  CHECK(histogram.cumulative[0] == 0);
  CHECK(histogram.cumulative[1] == 1);
  CHECK(histogram.cumulative[7] == 2);
  CHECK(histogram.cumulative[umminfer::obs::Histogram::kBuckets - 1] == 2);

  umminfer::obs::Histogram merged;
  merged.observe(4.0);
  merged.merge(histogram);
  CHECK(merged.count == 4);
  CHECK(merged.sum > 104.29 && merged.sum < 104.31);

  std::string out;
  histogram.render("umm_test_seconds", "stage=\"vlm\"", out);
  CHECK(contains(out, "umm_test_seconds_bucket{stage=\"vlm\",le=\"+Inf\"} 3"));
  CHECK(contains(out, "umm_test_seconds_count{stage=\"vlm\"} 3"));
  CHECK(contains(out, "umm_test_seconds_sum{stage=\"vlm\"} 100.302"));
}

void test_aggregation() {
  using umminfer::obs::Metrics;
  Metrics& metrics = Metrics::instance();

  metrics.stage_submit("vlm", "chat");
  metrics.stage_in_flight("vlm", 1);
  metrics.stage_finish("vlm", "chat", "completed", 0.25);
  // A negative delta must clamp at zero, never wrap to a huge gauge value.
  metrics.stage_in_flight("vlm", -3);
  metrics.stage_submit("asr", "transcription");
  metrics.stage_finish("asr", "transcription", "failed", 0.5);
  metrics.stage_in_flight("asr", -1);

  metrics.request("chat", "completed", 1.5);
  metrics.leg("asr", "chat", "completed", 0.2);
  metrics.leg("vlm", "chat", "failed", 0.8);
  metrics.queue_depth("unified_admission", 3);
  metrics.abort("client", 0.4);
  metrics.vlm_phase("prefill", 0.05);
  metrics.vlm_phase("decode", 0.75);
  metrics.tts_phase("first_frame", 0.1);
  metrics.tts_phase("total", 1.0);
  metrics.image_denoise_step(0.02);
  metrics.http("chat", "200");

  const std::string text = metrics.render();
  CHECK(contains(text,
                 "umm_stage_requests_total{stage=\"vlm\",request_type=\"chat\","
                 "status=\"submitted\"} 1"));
  CHECK(contains(text,
                 "umm_stage_requests_total{stage=\"vlm\",request_type=\"chat\","
                 "status=\"completed\"} 1"));
  CHECK(contains(text,
                 "umm_stage_requests_total{stage=\"asr\","
                 "request_type=\"transcription\",status=\"failed\"} 1"));
  CHECK(contains(text, "umm_stage_in_flight{stage=\"vlm\"} 0"));
  CHECK(contains(text, "umm_stage_in_flight{stage=\"asr\"} 0"));
  CHECK(contains(text,
                 "umm_stage_duration_seconds_count{stage=\"vlm\","
                 "request_type=\"chat\"} 1"));
  CHECK(contains(text,
                 "umm_request_total{request_type=\"chat\",status=\"completed\"} 1"));
  CHECK(contains(text,
                 "umm_request_duration_seconds_count{request_type=\"chat\","
                 "status=\"completed\"} 1"));
  CHECK(contains(text,
                 "umm_leg_requests_total{leg=\"asr\",request_type=\"chat\","
                 "status=\"completed\"} 1"));
  CHECK(contains(text, "umm_leg_duration_seconds_count{leg=\"vlm\","
                       "request_type=\"chat\"} 1"));
  CHECK(contains(text, "umm_queue_depth{queue=\"unified_admission\"} 3"));
  CHECK(contains(text, "umm_aborts_total{source=\"client\"} 1"));
  CHECK(contains(text, "umm_abort_latency_seconds_count 1"));
  CHECK(contains(text, "umm_vlm_prefill_seconds_count 1"));
  CHECK(contains(text, "umm_vlm_decode_seconds_count 1"));
  CHECK(contains(text, "umm_tts_first_frame_seconds_count 1"));
  CHECK(contains(text, "umm_tts_total_seconds_count 1"));
  CHECK(contains(text, "umm_image_denoise_step_seconds_count 1"));
  CHECK(contains(text, "umm_http_requests_total{route=\"chat\",status=\"200\"} 1"));
  CHECK(contains(text, "umm_gpu_nvml_available "));
}

void test_nvml_degrades() {
  // The build host may have no driver library at all. The contract is: an
  // unavailable client returns no samples instead of throwing or inventing.
  umminfer::obs::Nvml nvml;
  if (!nvml.available()) {
    CHECK(nvml.sample().empty());
    const std::string text = umminfer::obs::Metrics::instance().render();
    CHECK(contains(text, "umm_gpu_nvml_available 0"));
    CHECK(!contains(text, "umm_gpu_utilization_percent{"));
    return;
  }
  const std::string text = umminfer::obs::Metrics::instance().render();
  CHECK(contains(text, "umm_gpu_nvml_available 1"));
  for (const auto& gpu : nvml.sample()) {
    CHECK(gpu.memory_total_bytes >= gpu.memory_used_bytes);
    CHECK(contains(text, "umm_gpu_memory_total_bytes{index=\"" +
                             std::to_string(gpu.index) + "\"}"));
  }
}

void test_result_ms() {
  umminfer::Value::Map map;
  map["prefill_ms"] = umminfer::Value(12.5);
  map["decode_ms"] = umminfer::Value(static_cast<int64_t>(300));
  map["text"] = umminfer::Value("not a number");
  const umminfer::Value result(std::move(map));
  CHECK(umminfer::obs::result_ms(result, "prefill_ms") == 12.5);
  CHECK(umminfer::obs::result_ms(result, "decode_ms") == 300.0);
  CHECK(umminfer::obs::result_ms(result, "text") < 0);
  CHECK(umminfer::obs::result_ms(result, "absent") < 0);
}

void test_trace_gate() {
  const std::string path = "/tmp/umm-metrics-test-trace.jsonl";
  std::remove(path.c_str());
  std::remove((path + ".1").c_str());
  umminfer::obs::Metrics& metrics = umminfer::obs::Metrics::instance();
  umminfer::obs::TraceRecord record;
  record.request_id = "req-7";
  record.stage = "vlm";
  record.request_type = "chat";
  record.start_epoch_ms = 1700000000000;
  record.end_epoch_ms = 1700000000125;
  record.status = "completed";
  record.duration_ms = 125.0;
  record.asr_ms = 20.0;
  record.vlm_ms = 100.0;

  // Off by default: no file is created at all.
  ::unsetenv("UMM_TRACE");
  metrics.reset_trace_for_test();
  metrics.trace(record);
  CHECK(file_size(path) < 0);

  // On: exactly one JSON line with the id, the status and every leg.
  ::setenv("UMM_TRACE", "1", 1);
  ::setenv("UMM_TRACE_FILE", path.c_str(), 1);
  metrics.reset_trace_for_test();
  metrics.trace(record);
  const std::string line = read_file(path);
  CHECK(contains(line, "\"request_id\":\"req-7\""));
  CHECK(contains(line, "\"status\":\"completed\""));
  CHECK(contains(line, "\"asr_ms\":20"));
  CHECK(contains(line, "\"vlm_ms\":100"));
  CHECK(contains(line, "\"start_epoch_ms\":1700000000000"));
  CHECK(!contains(line, "image_ms"));
  CHECK(line.size() > 0 && line.back() == '\n');

  // A leg with no duration is omitted, never reported as -1.
  umminfer::obs::TraceRecord bare = record;
  bare.asr_ms = -1;
  bare.vlm_ms = -1;
  metrics.reset_trace_for_test();
  std::remove(path.c_str());
  metrics.trace(bare);
  CHECK(!contains(read_file(path), "asr_ms"));
}

void test_trace_cap() {
  const std::string path = "/tmp/umm-metrics-test-cap.jsonl";
  std::remove(path.c_str());
  std::remove((path + ".1").c_str());
  ::setenv("UMM_TRACE", "1", 1);
  ::setenv("UMM_TRACE_FILE", path.c_str(), 1);
  umminfer::obs::Metrics& metrics = umminfer::obs::Metrics::instance();
  metrics.reset_trace_for_test();

  umminfer::obs::TraceRecord record;
  record.request_id = "req-0123456789012345678901234567890123456789";
  record.stage = "vlm";
  record.request_type = "chat";
  record.status = "completed";
  record.duration_ms = 1.0;
  for (int index = 0; index < 4000; ++index) metrics.trace(record);

  const long active = file_size(path);
  const long previous = file_size(path + ".1");
  CHECK(active >= 0 && active <= 65536);
  CHECK(previous >= 0 && previous <= 65536);
  CHECK(active + previous > 0);

  ::unsetenv("UMM_TRACE");
  ::unsetenv("UMM_TRACE_FILE");
  metrics.reset_trace_for_test();
}

}  // namespace

int main() {
  test_histogram();
  test_aggregation();
  test_nvml_degrades();
  test_result_ms();
  test_trace_gate();
  test_trace_cap();
  if (g_failures != 0) {
    std::fprintf(stderr, "%d metric check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "metrics_test: all checks passed\n");
  return 0;
}
