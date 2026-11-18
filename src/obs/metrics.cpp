// SPDX-License-Identifier: Apache-2.0
#include "umminfer/obs/metrics.hpp"

#include <dlfcn.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

#include "umminfer/proto/wire.hpp"

namespace umminfer::obs {
namespace {

constexpr size_t kTraceCapBytes = 65536;

void append_label_value(std::string& out, const std::string& value) {
  for (const char ch : value) {
    if (ch == '\\' || ch == '"') out.push_back('\\');
    out.push_back(ch);
  }
}

std::string labels(std::initializer_list<std::pair<const char*, std::string>> items) {
  std::string out;
  for (const auto& [name, value] : items) {
    if (!out.empty()) out.push_back(',');
    out.append(name);
    out.append("=\"");
    append_label_value(out, value);
    out.append("\"");
  }
  return out;
}

std::string number(double value) {
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%.10g", value);
  return buffer;
}

int64_t epoch_millis() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

}  // namespace

// ---- histogram -----------------------------------------------------------

const std::array<double, Histogram::kBuckets>& Histogram::bounds() {
  static const std::array<double, kBuckets> value = {
      0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25,
      0.5,   1.0,   2.5,  5.0,   10.0, 30.0, 60.0};
  return value;
}

void Histogram::observe(double value) {
  if (!(value >= 0)) value = 0;  // also rejects NaN
  const auto& limits = bounds();
  size_t index = 0;
  while (index < limits.size() && value > limits[index]) ++index;
  for (size_t slot = index; slot < limits.size(); ++slot) ++cumulative[slot];
  ++count;
  sum += value;
}

void Histogram::merge(const Histogram& other) {
  for (size_t index = 0; index < kBuckets; ++index) {
    cumulative[index] += other.cumulative[index];
  }
  count += other.count;
  sum += other.sum;
}

void Histogram::render(const std::string& name, const std::string& label_set,
                       std::string& out) const {
  const auto& limits = bounds();
  for (size_t index = 0; index < limits.size(); ++index) {
    out += name;
    out += "_bucket{";
    out += label_set;
    if (!label_set.empty()) out.push_back(',');
    out += "le=\"";
    out += number(limits[index]);
    out += "\"} ";
    out += std::to_string(cumulative[index]);
    out.push_back('\n');
  }
  out += name;
  out += "_bucket{";
  out += label_set;
  if (!label_set.empty()) out.push_back(',');
  out += "le=\"+Inf\"} ";
  out += std::to_string(count);
  out.push_back('\n');
  // A metric without labels is written "name_count value". Empty braces are
  // not valid Prometheus text, so they are omitted rather than left blank.
  out += name;
  out += "_count";
  if (!label_set.empty()) {
    out += "{";
    out += label_set;
    out += "}";
  }
  out += " ";
  out += std::to_string(count);
  out.push_back('\n');
  out += name;
  out += "_sum";
  if (!label_set.empty()) {
    out += "{";
    out += label_set;
    out += "}";
  }
  out += " ";
  out += number(sum);
  out.push_back('\n');
}

// ---- NVML ----------------------------------------------------------------

namespace {

// Minimal mirrors of the NVML ABI. Declared here so no header from the driver
// package is needed at build time.
using NvmlReturn = int;
using NvmlDevice = void*;
struct NvmlUtilization {
  unsigned gpu;
  unsigned memory;
};
struct NvmlMemory {
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
};
constexpr int kNvmlSuccess = 0;
constexpr int kNvmlTemperatureGpu = 0;
constexpr int kNvmlClockGraphics = 0;
constexpr int kNvmlClockSm = 1;
constexpr int kNvmlClockMemory = 2;

struct NvmlApi {
  NvmlReturn (*init)();
  NvmlReturn (*shutdown)();
  NvmlReturn (*count)(unsigned*);
  NvmlReturn (*handle)(unsigned, NvmlDevice*);
  NvmlReturn (*name)(NvmlDevice, char*, unsigned);
  NvmlReturn (*utilization)(NvmlDevice, NvmlUtilization*);
  NvmlReturn (*memory)(NvmlDevice, NvmlMemory*);
  NvmlReturn (*temperature)(NvmlDevice, int, unsigned*);
  NvmlReturn (*power)(NvmlDevice, unsigned*);
  NvmlReturn (*clock)(NvmlDevice, int, unsigned*);
};

template <typename Fn>
Fn resolve(void* library, const char* primary, const char* fallback = nullptr) {
  void* symbol = dlsym(library, primary);
  if (symbol == nullptr && fallback != nullptr) symbol = dlsym(library, fallback);
  return reinterpret_cast<Fn>(symbol);
}

}  // namespace

Nvml::Nvml(const char* library_name) {
  // The runtime driver library only exists on a host with an NVIDIA driver.
  if (library_name != nullptr) {
    library_ = dlopen(library_name, RTLD_NOW | RTLD_LOCAL);
  } else {
    library_ = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (library_ == nullptr) {
      library_ = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_LOCAL);
    }
  }
  if (library_ == nullptr) return;
  auto init = resolve<NvmlReturn (*)()>(library_, "nvmlInit_v2", "nvmlInit");
  auto shutdown = resolve<NvmlReturn (*)()>(library_, "nvmlShutdown");
  auto count = resolve<NvmlReturn (*)(unsigned*)>(library_, "nvmlDeviceGetCount_v2",
                                                   "nvmlDeviceGetCount");
  auto handle = resolve<NvmlReturn (*)(unsigned, NvmlDevice*)>(
      library_, "nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetHandleByIndex");
  if (init == nullptr || shutdown == nullptr || count == nullptr || handle == nullptr) {
    dlclose(library_);
    library_ = nullptr;
    return;
  }
  if (init() != kNvmlSuccess) {
    dlclose(library_);
    library_ = nullptr;
    return;
  }
  initialized_ = true;
  available_ = true;
}

Nvml::~Nvml() {
  if (initialized_) {
    auto shutdown = resolve<NvmlReturn (*)()>(library_, "nvmlShutdown");
    if (shutdown != nullptr) shutdown();
  }
  if (library_ != nullptr) dlclose(library_);
}

std::vector<GpuSample> Nvml::sample() {
  std::vector<GpuSample> samples;
  if (!available_) return samples;
  auto count = resolve<NvmlReturn (*)(unsigned*)>(library_, "nvmlDeviceGetCount_v2",
                                                   "nvmlDeviceGetCount");
  auto handle = resolve<NvmlReturn (*)(unsigned, NvmlDevice*)>(
      library_, "nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetHandleByIndex");
  auto get_name = resolve<NvmlReturn (*)(NvmlDevice, char*, unsigned)>(
      library_, "nvmlDeviceGetName");
  auto utilization = resolve<NvmlReturn (*)(NvmlDevice, NvmlUtilization*)>(
      library_, "nvmlDeviceGetUtilizationRates");
  auto memory = resolve<NvmlReturn (*)(NvmlDevice, NvmlMemory*)>(
      library_, "nvmlDeviceGetMemoryInfo");
  auto temperature = resolve<NvmlReturn (*)(NvmlDevice, int, unsigned*)>(
      library_, "nvmlDeviceGetTemperature");
  auto power = resolve<NvmlReturn (*)(NvmlDevice, unsigned*)>(
      library_, "nvmlDeviceGetPowerUsage");
  auto clock = resolve<NvmlReturn (*)(NvmlDevice, int, unsigned*)>(
      library_, "nvmlDeviceGetClockInfo");
  unsigned total = 0;
  if (count == nullptr || handle == nullptr || count(&total) != kNvmlSuccess) {
    return samples;
  }
  for (unsigned index = 0; index < total; ++index) {
    NvmlDevice device = nullptr;
    if (handle(index, &device) != kNvmlSuccess || device == nullptr) continue;
    GpuSample sample;
    sample.index = index;
    char name[96] = {0};
    if (get_name != nullptr && get_name(device, name, sizeof(name) - 1) == kNvmlSuccess) {
      sample.name = name;
    }
    NvmlUtilization use{};
    if (utilization != nullptr && utilization(device, &use) == kNvmlSuccess) {
      sample.utilization_percent = use.gpu;
    }
    NvmlMemory mem{};
    if (memory != nullptr && memory(device, &mem) == kNvmlSuccess) {
      sample.memory_used_bytes = static_cast<double>(mem.used);
      sample.memory_total_bytes = static_cast<double>(mem.total);
    }
    unsigned temp = 0;
    if (temperature != nullptr &&
        temperature(device, kNvmlTemperatureGpu, &temp) == kNvmlSuccess) {
      sample.temperature_celsius = temp;
    }
    unsigned milliwatts = 0;
    if (power != nullptr && power(device, &milliwatts) == kNvmlSuccess) {
      sample.power_watts = milliwatts / 1000.0;
    }
    unsigned mhz = 0;
    if (clock != nullptr &&
        clock(device, kNvmlClockSm, &mhz) == kNvmlSuccess) {
      sample.sm_clock_mhz = mhz;
    }
    if (clock != nullptr &&
        clock(device, kNvmlClockMemory, &mhz) == kNvmlSuccess) {
      sample.memory_clock_mhz = mhz;
    }
    samples.push_back(std::move(sample));
  }
  return samples;
}

// ---- metrics -------------------------------------------------------------

Metrics& Metrics::instance() {
  static Metrics metrics;
  return metrics;
}

void Metrics::stage_submit(const std::string& stage, const std::string& request_type) {
  std::lock_guard<std::mutex> lock(mu_);
  stages_[stage].requests[request_type + "|submitted"]++;
}

void Metrics::stage_finish(const std::string& stage, const std::string& request_type,
                           const std::string& status, double seconds) {
  std::lock_guard<std::mutex> lock(mu_);
  StageState& state = stages_[stage];
  state.requests[request_type + "|" + status]++;
  state.duration[request_type].observe(seconds);
}

void Metrics::stage_in_flight(const std::string& stage, int delta) {
  std::lock_guard<std::mutex> lock(mu_);
  StageState& state = stages_[stage];
  if (delta < 0) {
    const auto drop = static_cast<uint64_t>(-delta);
    state.in_flight = state.in_flight > drop ? state.in_flight - drop : 0;
  } else {
    state.in_flight += static_cast<uint64_t>(delta);
  }
}

void Metrics::queue_depth(const std::string& queue, size_t depth) {
  std::lock_guard<std::mutex> lock(mu_);
  queue_depth_[queue] = depth;
}

void Metrics::request(const std::string& request_type, const std::string& status,
                      double seconds) {
  std::lock_guard<std::mutex> lock(mu_);
  requests_[request_type + "|" + status]++;
  request_duration_[request_type + "|" + status].observe(seconds);
}

void Metrics::leg(const std::string& leg_name, const std::string& request_type,
                  const std::string& status, double seconds) {
  std::lock_guard<std::mutex> lock(mu_);
  leg_total_[leg_name + "|" + request_type + "|" + status]++;
  leg_duration_[leg_name + "|" + request_type].observe(seconds);
}

void Metrics::abort(const std::string& source, double latency_seconds) {
  std::lock_guard<std::mutex> lock(mu_);
  aborts_[source]++;
  abort_latency_.observe(latency_seconds);
}

void Metrics::vlm_phase(const char* phase, double seconds) {
  std::lock_guard<std::mutex> lock(mu_);
  if (std::strcmp(phase, "prefill") == 0) vlm_prefill_.observe(seconds);
  else if (std::strcmp(phase, "decode") == 0) vlm_decode_.observe(seconds);
}

void Metrics::tts_phase(const char* phase, double seconds) {
  std::lock_guard<std::mutex> lock(mu_);
  if (std::strcmp(phase, "first_frame") == 0) tts_first_frame_.observe(seconds);
  else if (std::strcmp(phase, "total") == 0) tts_total_.observe(seconds);
}

void Metrics::image_denoise_step(double seconds) {
  std::lock_guard<std::mutex> lock(mu_);
  image_denoise_step_.observe(seconds);
}

void Metrics::http(const std::string& route, const std::string& status) {
  std::lock_guard<std::mutex> lock(mu_);
  http_[route + "|" + status]++;
}

// The environment is read per call: a getenv is cheaper than the trace write it
// guards, and an operator can turn tracing on without a restart.
bool Metrics::trace_enabled() {
  const char* value = std::getenv("UMM_TRACE");
  if (value == nullptr || std::strcmp(value, "1") != 0) {
    std::lock_guard<std::mutex> lock(trace_mu_);
    if (trace_file_ != nullptr) {
      std::fclose(trace_file_);
      trace_file_ = nullptr;
      trace_open_path_.clear();
      trace_bytes_ = 0;
    }
    return false;
  }
  const char* path = std::getenv("UMM_TRACE_FILE");
  std::lock_guard<std::mutex> lock(trace_mu_);
  trace_path_ = (path != nullptr && path[0] != '\0') ? path : "umm-trace.jsonl";
  return true;
}

// Rotation keeps every file at or below kTraceCapBytes: the active file and
// its single predecessor. A hard kill after rotation leaves two bounded files.
void Metrics::write_trace_line(const std::string& line) {
  std::lock_guard<std::mutex> lock(trace_mu_);
  if (line.size() + 1 > kTraceCapBytes) return;  // one line never exceeds the cap
  if (trace_file_ != nullptr && trace_open_path_ != trace_path_) {
    std::fclose(trace_file_);
    trace_file_ = nullptr;
    trace_open_path_.clear();
    trace_bytes_ = 0;
  }
  if (trace_file_ == nullptr) {
    trace_file_ = std::fopen(trace_path_.c_str(), "a");
    if (trace_file_ == nullptr) return;
    trace_open_path_ = trace_path_;
    std::fseek(trace_file_, 0, SEEK_END);
    const long size = std::ftell(trace_file_);
    trace_bytes_ = size > 0 ? static_cast<size_t>(size) : 0;
  }
  if (trace_bytes_ + line.size() + 1 > kTraceCapBytes) {
    std::fclose(trace_file_);
    trace_file_ = nullptr;
    trace_open_path_.clear();
    const std::string previous = trace_path_ + ".1";
    std::remove(previous.c_str());
    std::rename(trace_path_.c_str(), previous.c_str());
    trace_file_ = std::fopen(trace_path_.c_str(), "a");
    trace_bytes_ = 0;
    if (trace_file_ == nullptr) return;
    trace_open_path_ = trace_path_;
  }
  std::fwrite(line.data(), 1, line.size(), trace_file_);
  std::fputc('\n', trace_file_);
  std::fflush(trace_file_);
  trace_bytes_ += line.size() + 1;
}

void Metrics::trace_start(const std::string& request_id,
                          const std::string& request_type,
                          const std::string& stage, int64_t start_epoch_ms) {
  if (!trace_enabled()) return;
  std::string line = "{\"request_id\":\"";
  append_label_value(line, request_id);
  line += "\",\"event\":\"request_start\",\"request_type\":\"";
  append_label_value(line, request_type);
  line += "\",\"stage\":\"";
  append_label_value(line, stage);
  line += "\",\"start_epoch_ms\":";
  line += std::to_string(start_epoch_ms);
  line += "}";
  write_trace_line(line);
}

void Metrics::trace_leg(const std::string& request_id,
                        const std::string& leg_name,
                        const std::string& status, double duration_ms) {
  if (!trace_enabled()) return;
  std::string line = "{\"request_id\":\"";
  append_label_value(line, request_id);
  line += "\",\"event\":\"leg_end\",\"leg\":\"";
  append_label_value(line, leg_name);
  line += "\",\"status\":\"";
  append_label_value(line, status);
  line += "\",\"duration_ms\":";
  line += number(duration_ms);
  line += "}";
  write_trace_line(line);
}

void Metrics::trace(const TraceRecord& record) {
  if (!trace_enabled()) return;
  std::string line = "{\"request_id\":\"";
  append_label_value(line, record.request_id);
  line += "\",\"event\":\"request_end\",\"request_type\":\"";
  append_label_value(line, record.request_type);
  line += "\",\"stage\":\"";
  append_label_value(line, record.stage);
  line += "\",\"start_epoch_ms\":";
  line += std::to_string(record.start_epoch_ms);
  line += ",\"end_epoch_ms\":";
  line += std::to_string(record.end_epoch_ms);
  line += ",\"status\":\"";
  append_label_value(line, record.status);
  line += "\",\"duration_ms\":";
  line += number(record.duration_ms);
  const auto append_leg = [&line](const char* name, double ms) {
    if (ms < 0) return;
    line += ",\"";
    line += name;
    line += "\":";
    line += number(ms);
  };
  append_leg("asr_ms", record.asr_ms);
  append_leg("vlm_ms", record.vlm_ms);
  append_leg("image_ms", record.image_ms);
  append_leg("tts_ms", record.tts_ms);
  line += "}";
  write_trace_line(line);
}

std::string Metrics::render() {
  std::string out;
  out.reserve(16 << 10);

  // The GPU sample runs outside the registry lock: NVML calls can block and
  // must not stall the request path that updates the counters.
  const std::vector<GpuSample> gpus = nvml_.sample();

  std::lock_guard<std::mutex> lock(mu_);

  out += "# HELP umm_build_info Build marker.\n# TYPE umm_build_info gauge\n";
  out += "umm_build_info{component=\"umm-infer\"} 1\n";

  out += "# HELP umm_trace_enabled 1 when UMM_TRACE=1 enables request tracing.\n"
         "# TYPE umm_trace_enabled gauge\n";
  {
    const char* value = std::getenv("UMM_TRACE");
    const bool on = value != nullptr && std::strcmp(value, "1") == 0;
    out += "umm_trace_enabled ";
    out += on ? "1\n" : "0\n";
  }

  out += "# HELP umm_stage_requests_total Stage request transitions seen by the "
         "coordinator.\n# TYPE umm_stage_requests_total counter\n";
  for (const auto& [stage, state] : stages_) {
    for (const auto& [key, count] : state.requests) {
      const size_t split = key.find('|');
      out += "umm_stage_requests_total{";
      out += labels({{"stage", stage},
                     {"request_type", key.substr(0, split)},
                     {"status", key.substr(split + 1)}});
      out += "} ";
      out += std::to_string(count);
      out.push_back('\n');
    }
  }

  out += "# HELP umm_stage_in_flight Requests submitted to a stage and not yet "
         "terminal.\n# TYPE umm_stage_in_flight gauge\n";
  for (const auto& [stage, state] : stages_) {
    out += "umm_stage_in_flight{";
    out += labels({{"stage", stage}});
    out += "} ";
    out += std::to_string(state.in_flight);
    out.push_back('\n');
  }

  out += "# HELP umm_stage_duration_seconds Submit-to-terminal time per stage.\n"
         "# TYPE umm_stage_duration_seconds histogram\n";
  for (const auto& [stage, state] : stages_) {
    for (const auto& [request_type, histogram] : state.duration) {
      histogram.render("umm_stage_duration_seconds",
                       labels({{"stage", stage}, {"request_type", request_type}}),
                       out);
    }
  }

  out += "# HELP umm_request_duration_seconds End-to-end engine request time.\n"
         "# TYPE umm_request_duration_seconds histogram\n";
  for (const auto& [key, histogram] : request_duration_) {
    const size_t split = key.find('|');
    histogram.render("umm_request_duration_seconds",
                     labels({{"request_type", key.substr(0, split)},
                             {"status", key.substr(split + 1)}}),
                     out);
  }

  out += "# HELP umm_request_total Engine requests by terminal status.\n"
         "# TYPE umm_request_total counter\n";
  for (const auto& [key, count] : requests_) {
    const size_t split = key.find('|');
    out += "umm_request_total{";
    out += labels({{"request_type", key.substr(0, split)},
                   {"status", key.substr(split + 1)}});
    out += "} ";
    out += std::to_string(count);
    out.push_back('\n');
  }

  out += "# HELP umm_leg_duration_seconds One leg of a unified request.\n"
         "# TYPE umm_leg_duration_seconds histogram\n";
  for (const auto& [key, histogram] : leg_duration_) {
    const size_t split = key.find('|');
    histogram.render("umm_leg_duration_seconds",
                     labels({{"leg", key.substr(0, split)},
                             {"request_type", key.substr(split + 1)}}),
                     out);
  }

  out += "# HELP umm_leg_requests_total Unified leg terminals.\n"
         "# TYPE umm_leg_requests_total counter\n";
  for (const auto& [key, count] : leg_total_) {
    const size_t first = key.find('|');
    const size_t second = key.find('|', first + 1);
    out += "umm_leg_requests_total{";
    out += labels({{"leg", key.substr(0, first)},
                   {"request_type", key.substr(first + 1, second - first - 1)},
                   {"status", key.substr(second + 1)}});
    out += "} ";
    out += std::to_string(count);
    out.push_back('\n');
  }

  out += "# HELP umm_queue_depth Admission queue depth.\n"
         "# TYPE umm_queue_depth gauge\n";
  for (const auto& [queue, depth] : queue_depth_) {
    out += "umm_queue_depth{";
    out += labels({{"queue", queue}});
    out += "} ";
    out += std::to_string(depth);
    out.push_back('\n');
  }

  out += "# HELP umm_aborts_total Abort broadcasts by source.\n"
         "# TYPE umm_aborts_total counter\n";
  for (const auto& [source, count] : aborts_) {
    out += "umm_aborts_total{";
    out += labels({{"source", source}});
    out += "} ";
    out += std::to_string(count);
    out.push_back('\n');
  }

  out += "# HELP umm_abort_latency_seconds Request start to observed abort.\n"
         "# TYPE umm_abort_latency_seconds histogram\n";
  abort_latency_.render("umm_abort_latency_seconds", "", out);

  out += "# HELP umm_vlm_prefill_seconds VLM prefill time reported by the stage.\n"
         "# TYPE umm_vlm_prefill_seconds histogram\n";
  vlm_prefill_.render("umm_vlm_prefill_seconds", "", out);
  out += "# HELP umm_vlm_decode_seconds VLM decode time reported by the stage.\n"
         "# TYPE umm_vlm_decode_seconds histogram\n";
  vlm_decode_.render("umm_vlm_decode_seconds", "", out);

  out += "# HELP umm_tts_first_frame_seconds Time to the first generated codec "
         "frame.\n# TYPE umm_tts_first_frame_seconds histogram\n";
  tts_first_frame_.render("umm_tts_first_frame_seconds", "", out);
  out += "# HELP umm_tts_total_seconds TTS synthesis time reported by the stage.\n"
         "# TYPE umm_tts_total_seconds histogram\n";
  tts_total_.render("umm_tts_total_seconds", "", out);

  out += "# HELP umm_image_denoise_step_seconds Average denoise time per step.\n"
         "# TYPE umm_image_denoise_step_seconds histogram\n";
  image_denoise_step_.render("umm_image_denoise_step_seconds", "", out);

  out += "# HELP umm_http_requests_total HTTP responses by route and status.\n"
         "# TYPE umm_http_requests_total counter\n";
  for (const auto& [key, count] : http_) {
    const size_t split = key.find('|');
    out += "umm_http_requests_total{";
    out += labels({{"route", key.substr(0, split)},
                   {"status", key.substr(split + 1)}});
    out += "} ";
    out += std::to_string(count);
    out.push_back('\n');
  }

  out += "# HELP umm_gpu_nvml_available 1 when NVML loaded and initialised.\n"
         "# TYPE umm_gpu_nvml_available gauge\n";
  out += "umm_gpu_nvml_available ";
  out += nvml_.available() ? "1\n" : "0\n";

  if (!gpus.empty()) {
    out += "# HELP umm_gpu_utilization_percent GPU utilisation reported by NVML.\n"
           "# TYPE umm_gpu_utilization_percent gauge\n";
    for (const GpuSample& gpu : gpus) {
      out += "umm_gpu_utilization_percent{";
      out += labels({{"index", std::to_string(gpu.index)}, {"name", gpu.name}});
      out += "} ";
      out += number(gpu.utilization_percent);
      out.push_back('\n');
    }
    out += "# HELP umm_gpu_memory_used_bytes Device memory in use.\n"
           "# TYPE umm_gpu_memory_used_bytes gauge\n";
    for (const GpuSample& gpu : gpus) {
      out += "umm_gpu_memory_used_bytes{";
      out += labels({{"index", std::to_string(gpu.index)}});
      out += "} ";
      out += number(gpu.memory_used_bytes);
      out.push_back('\n');
    }
    out += "# HELP umm_gpu_memory_total_bytes Device memory installed.\n"
           "# TYPE umm_gpu_memory_total_bytes gauge\n";
    for (const GpuSample& gpu : gpus) {
      out += "umm_gpu_memory_total_bytes{";
      out += labels({{"index", std::to_string(gpu.index)}});
      out += "} ";
      out += number(gpu.memory_total_bytes);
      out.push_back('\n');
    }
    out += "# HELP umm_gpu_temperature_celsius GPU temperature.\n"
           "# TYPE umm_gpu_temperature_celsius gauge\n";
    for (const GpuSample& gpu : gpus) {
      out += "umm_gpu_temperature_celsius{";
      out += labels({{"index", std::to_string(gpu.index)}});
      out += "} ";
      out += number(gpu.temperature_celsius);
      out.push_back('\n');
    }
    out += "# HELP umm_gpu_power_watts GPU board power draw.\n"
           "# TYPE umm_gpu_power_watts gauge\n";
    for (const GpuSample& gpu : gpus) {
      out += "umm_gpu_power_watts{";
      out += labels({{"index", std::to_string(gpu.index)}});
      out += "} ";
      out += number(gpu.power_watts);
      out.push_back('\n');
    }
    out += "# HELP umm_gpu_sm_clock_mhz Streaming multiprocessor clock.\n"
           "# TYPE umm_gpu_sm_clock_mhz gauge\n";
    for (const GpuSample& gpu : gpus) {
      out += "umm_gpu_sm_clock_mhz{";
      out += labels({{"index", std::to_string(gpu.index)}});
      out += "} ";
      out += number(gpu.sm_clock_mhz);
      out.push_back('\n');
    }
    out += "# HELP umm_gpu_memory_clock_mhz Device memory clock.\n"
           "# TYPE umm_gpu_memory_clock_mhz gauge\n";
    for (const GpuSample& gpu : gpus) {
      out += "umm_gpu_memory_clock_mhz{";
      out += labels({{"index", std::to_string(gpu.index)}});
      out += "} ";
      out += number(gpu.memory_clock_mhz);
      out.push_back('\n');
    }
  }
  return out;
}

double result_ms(const Value& result, const char* key) {
  const Value* value = result.find(key);
  if (value == nullptr) return -1;
  if (value->is_int()) return static_cast<double>(value->as_int());
  if (value->is_float()) return value->as_double();
  return -1;
}

}  // namespace umminfer::obs
