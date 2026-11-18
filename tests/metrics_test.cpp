// SPDX-License-Identifier: Apache-2.0
// Blackbox API tests for Prometheus GET /metrics exposition, NVML graceful
// degradation, and UMM_TRACE=1 JSONL request/leg tracing + 64 KiB rotation.
// Exercises real HTTP requests through OpenAIApi + UnifiedEngine + Coordinator
// without calling Metrics internal mutators or test-only reset hooks.
#include <httplib.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "umminfer/audio/audio.hpp"
#include "umminfer/pipeline/pipeline.hpp"
#include "umminfer/pipeline/unified.hpp"
#include "umminfer/serve/openai_api.hpp"

namespace {

using json = nlohmann::json;

int g_failures = 0;

void check(bool condition, const char* expr, int line) {
  if (condition) return;
  std::fprintf(stderr, "FAIL line %d: %s\n", line, expr);
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

struct MetricsHarness {
  std::string ipc_dir;
  std::unique_ptr<umminfer::Coordinator> coordinator;
  std::unique_ptr<umminfer::Stage> asr_stage;
  std::unique_ptr<umminfer::Stage> vlm_stage;
  std::unique_ptr<umminfer::Stage> tts_stage;
  std::unique_ptr<umminfer::Stage> image_stage;
  std::unique_ptr<umminfer::UnifiedEngine> unified;
  std::unique_ptr<umminfer::OpenAIApi> api;
  std::thread asr_thread;
  std::thread vlm_thread;
  std::thread tts_thread;
  std::thread image_thread;
  std::thread coord_thread;
  std::thread api_thread;

  std::atomic<bool> fail_asr{false};
  std::atomic<bool> hold_vlm{false};
  std::atomic<bool> vlm_entered{false};

  MetricsHarness() {
    static std::atomic<int> seq{0};
    ipc_dir = "/tmp/umm-metrics-bb-" + std::to_string(::getpid()) + "-" +
              std::to_string(++seq);
    std::filesystem::create_directories(ipc_dir);
    const std::string done_ep = "ipc://" + ipc_dir + "/done";
    const std::string abort_ep = "ipc://" + ipc_dir + "/abort";
    const std::string asr_ep = "ipc://" + ipc_dir + "/asr";
    const std::string vlm_ep = "ipc://" + ipc_dir + "/vlm";
    const std::string tts_ep = "ipc://" + ipc_dir + "/tts";
    const std::string img_ep = "ipc://" + ipc_dir + "/image";

    coordinator = std::make_unique<umminfer::Coordinator>(done_ep, abort_ep);
    coordinator->register_stage("asr", asr_ep, true);
    coordinator->register_stage("vlm", vlm_ep, true);
    coordinator->register_stage("tts", tts_ep, true);
    coordinator->register_stage("image", img_ep, true);

    umminfer::ApiConfig acfg;
    acfg.host = "127.0.0.1";
    acfg.port = 0;
    acfg.model = "served-model";
    acfg.entry_stage = "asr";
    acfg.request_timeout_s = 10.0;
    api = std::make_unique<umminfer::OpenAIApi>(*coordinator, acfg);

    umminfer::UnifiedConfig ucfg;
    ucfg.asr_stage = "asr";
    ucfg.vlm_stage = "vlm";
    ucfg.tts_stage = "tts";
    ucfg.image_stage = "image";
    unified = std::make_unique<umminfer::UnifiedEngine>(*coordinator, ucfg);
    api->set_unified_engine(*unified, "vlm");
    coordinator->start();
    coord_thread = std::thread([this] { coordinator->run(); });

    auto make_cfg = [&](const std::string& name, const std::string& ep) {
      umminfer::Stage::Config scfg;
      scfg.name = name;
      scfg.recv_endpoint = ep;
      scfg.coordinator_endpoint = done_ep;
      scfg.abort_endpoint = abort_ep;
      scfg.announce_ready = true;
      return scfg;
    };

    asr_stage = std::make_unique<umminfer::Stage>(
        make_cfg("asr", asr_ep),
        [this](const umminfer::Value&, const umminfer::AbortCheck&) {
          if (fail_asr.load()) throw std::runtime_error("asr failed");
          umminfer::Value::Map out;
          out["text"] = umminfer::Value("transcribed hello");
          out["generated_token_count"] = umminfer::Value(int64_t{3});
          out["finish_reason"] = umminfer::Value("stop");
          return umminfer::Value(std::move(out));
        });

    vlm_stage = std::make_unique<umminfer::Stage>(
        make_cfg("vlm", vlm_ep),
        [this](const umminfer::Value&, const umminfer::AbortCheck& aborted) {
          vlm_entered.store(true);
          while (hold_vlm.load()) {
            if (aborted()) throw umminfer::RequestAborted("aborted");
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
          }
          umminfer::Value::Map out;
          out["content"] = umminfer::Value("assistant answer.");
          out["finish_reason"] = umminfer::Value("stop");
          out["prefill_ms"] = umminfer::Value(12.5);
          out["decode_ms"] = umminfer::Value(int64_t{80});
          return umminfer::Value(std::move(out));
        });

    tts_stage = std::make_unique<umminfer::Stage>(
        make_cfg("tts", tts_ep),
        [](const umminfer::Value&, const umminfer::AbortCheck&) {
          const std::string wav =
              umminfer::audio::wav_from_pcm16(std::string(480, '\x01'), 24000);
          umminfer::Value::Map out;
          out["media_type"] = umminfer::Value("audio");
          out["mime_type"] = umminfer::Value("audio/wav");
          out["format"] = umminfer::Value("wav");
          out["sample_rate"] = umminfer::Value(int64_t{24000});
          out["channels"] = umminfer::Value(int64_t{1});
          out["first_frame_ms"] = umminfer::Value(15.0);
          out["total_ms"] = umminfer::Value(45.0);
          out["data"] = umminfer::Value::bin(wav);
          return umminfer::Value(std::move(out));
        });

    image_stage = std::make_unique<umminfer::Stage>(
        make_cfg("image", img_ep),
        [](const umminfer::Value&, const umminfer::AbortCheck&) {
          umminfer::Value::Map out;
          out["media_type"] = umminfer::Value("image");
          out["mime_type"] = umminfer::Value("image/png");
          out["format"] = umminfer::Value("png");
          out["size"] = umminfer::Value("512x512");
          out["steps"] = umminfer::Value(int64_t{2});
          out["denoise_ms"] = umminfer::Value(40.0);
          out["denoise_step_ms"] = umminfer::Value(20.0);
          out["data"] = umminfer::Value::bin("\x89PNG\r\n\x1a\npayload");
          return umminfer::Value(std::move(out));
        });

    asr_stage->start();
    vlm_stage->start();
    tts_stage->start();
    image_stage->start();
    asr_thread = std::thread([this] { asr_stage->run(); });
    vlm_thread = std::thread([this] { vlm_stage->run(); });
    tts_thread = std::thread([this] { tts_stage->run(); });
    image_thread = std::thread([this] { image_stage->run(); });

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
      if (coordinator->stage_ready("asr") &&
          coordinator->stage_ready("vlm") &&
          coordinator->stage_ready("tts") &&
          coordinator->stage_ready("image")) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!api->bind()) throw std::runtime_error("OpenAIApi bind failed");
    api_thread = std::thread([this] { api->listen(); });
    httplib::Client cli("127.0.0.1", api->bound_port());
    for (int i = 0; i < 50; ++i) {
      if (auto r = cli.Get("/health"); r && r->status == 200) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  ~MetricsHarness() {
    hold_vlm.store(false);
    if (api) api->stop();
    if (api_thread.joinable()) api_thread.join();
    for (auto* st : {asr_stage.get(), vlm_stage.get(), tts_stage.get(),
                     image_stage.get()}) {
      if (st) st->stop();
    }
    for (auto* th : {&asr_thread, &vlm_thread, &tts_thread, &image_thread}) {
      if (th->joinable()) th->join();
    }
    if (coordinator) coordinator->stop();
    if (coord_thread.joinable()) coord_thread.join();
    std::error_code ec;
    std::filesystem::remove_all(ipc_dir, ec);
  }
};

void test_http_metrics_and_nvml() {
  MetricsHarness h;
  httplib::Client cli("127.0.0.1", h.api->bound_port());
  cli.set_read_timeout(10, 0);

  // 1. Non-streaming chat completion -> records chat request, vlm prefill/decode.
  {
    const json req = {{"model", "served-model"},
                      {"messages", {{{"role", "user"}, {"content", "hi"}}}}};
    auto res = cli.Post("/v1/chat/completions", req.dump(), "application/json");
    CHECK(res && res->status == 200);
  }

  // 2. Multimodal streaming chat (audio input -> ASR -> VLM -> image + TTS).
  {
    const std::string wav =
        umminfer::audio::wav_from_pcm16(std::string(3200, '\x02'), 16000);
    std::string b64;
    // Encode WAV via httplib base64 or simple lookup:
    static const char* kLut =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (size_t i = 0; i < wav.size(); i += 3) {
      const uint32_t a = static_cast<unsigned char>(wav[i]);
      const uint32_t b =
          i + 1 < wav.size() ? static_cast<unsigned char>(wav[i + 1]) : 0;
      const uint32_t c =
          i + 2 < wav.size() ? static_cast<unsigned char>(wav[i + 2]) : 0;
      const uint32_t t = (a << 16) | (b << 8) | c;
      b64.push_back(kLut[(t >> 18) & 63]);
      b64.push_back(kLut[(t >> 12) & 63]);
      b64.push_back(i + 1 < wav.size() ? kLut[(t >> 6) & 63] : '=');
      b64.push_back(i + 2 < wav.size() ? kLut[t & 63] : '=');
    }
    const json req = {
        {"model", "served-model"},
        {"stream", true},
        {"modalities", {"text", "image", "audio"}},
        {"image", {{"size", "512x512"}}},
        {"audio", {{"voice", "vivian"}, {"format", "wav"}}},
        {"messages",
         {{{"role", "user"},
           {"content",
            {{{"type", "input_audio"},
              {"input_audio", {{"data", b64}, {"format", "wav"}}}}}}}}}};
    auto res = cli.Post("/v1/chat/completions", req.dump(), "application/json");
    CHECK(res && res->status == 200);
    CHECK(contains(res->body, "[DONE]"));
  }

  // 3. ASR transcription success + failure.
  {
    const std::string wav =
        umminfer::audio::wav_from_pcm16(std::string(640, '\x01'), 16000);
    httplib::UploadFormDataItems items = {
        {"model", "served-model", "", ""},
        {"file", wav, "a.wav", "audio/wav"}};
    auto ok = cli.Post("/v1/audio/transcriptions", items);
    CHECK(ok && ok->status == 200);

    h.fail_asr.store(true);
    auto bad = cli.Post("/v1/audio/transcriptions", items);
    CHECK(bad && bad->status == 500);
    h.fail_asr.store(false);
  }

  // 4. Client abort via POST /abort while VLM is in flight.
  {
    h.vlm_entered.store(false);
    h.hold_vlm.store(true);
    std::string stream_req_id;
    std::mutex id_mu;
    std::thread streamer([&] {
      httplib::Client scli("127.0.0.1", h.api->bound_port());
      scli.set_read_timeout(5, 0);
      const json req = {{"model", "served-model"},
                        {"stream", true},
                        {"messages", {{{"role", "user"}, {"content", "hold"}}}}};
      std::string accum;
      (void)scli.Post(
          "/v1/chat/completions", httplib::Headers{}, req.dump(),
          "application/json", [&](const char* data, size_t len) {
            accum.append(data, len);
            const auto pos = accum.find("\"id\":\"");
            if (pos != std::string::npos) {
              const size_t start = pos + 6;
              const size_t end = accum.find('"', start);
              if (end != std::string::npos) {
                std::lock_guard<std::mutex> lock(id_mu);
                if (stream_req_id.empty()) {
                  stream_req_id = accum.substr(start, end - start);
                }
              }
            }
            return true;
          });
    });

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < deadline) {
      std::lock_guard<std::mutex> lock(id_mu);
      if (!stream_req_id.empty() && h.vlm_entered.load()) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::string id_copy;
    {
      std::lock_guard<std::mutex> lock(id_mu);
      id_copy = stream_req_id;
    }
    CHECK(!id_copy.empty());
    auto aborted = cli.Post("/abort", json{{"request_id", id_copy}}.dump(),
                            "application/json");
    CHECK(aborted && aborted->status == 200);
    h.hold_vlm.store(false);
    streamer.join();
  }

  // 5. Scrape GET /metrics over HTTP and verify Prometheus counters, gauges,
  // histograms, phase timings, and NVML graceful degradation.
  auto metrics_res = cli.Get("/metrics");
  CHECK(metrics_res && metrics_res->status == 200);
  const std::string text = metrics_res ? metrics_res->body : "";

  CHECK(contains(text, "umm_http_requests_total{route=\"chat\",status=\"200\"}"));
  CHECK(contains(
      text, "umm_http_requests_total{route=\"transcriptions\",status=\"200\"}"));
  CHECK(contains(
      text, "umm_http_requests_total{route=\"transcriptions\",status=\"500\"}"));
  CHECK(contains(text, "umm_request_total{request_type=\"chat\",status=\"completed\"}"));
  CHECK(contains(text, "umm_stage_in_flight{stage=\"vlm\"} 0"));
  CHECK(contains(text, "umm_stage_in_flight{stage=\"asr\"} 0"));
  CHECK(contains(
      text,
      "umm_leg_requests_total{leg=\"asr\",request_type=\"chat\",status=\"completed\"}"));
  CHECK(contains(
      text,
      "umm_leg_requests_total{leg=\"image\",request_type=\"chat\",status=\"completed\"}"));
  CHECK(contains(
      text,
      "umm_leg_requests_total{leg=\"tts\",request_type=\"chat\",status=\"completed\"}"));
  CHECK(contains(text, "umm_aborts_total{source=\"client\"} 1"));
  CHECK(contains(text, "umm_vlm_prefill_seconds_count "));
  CHECK(contains(text, "umm_vlm_decode_seconds_count "));
  CHECK(contains(text, "umm_tts_first_frame_seconds_count 1"));
  CHECK(contains(text, "umm_tts_total_seconds_count 1"));
  CHECK(contains(text, "umm_image_denoise_step_seconds_count 1"));
  CHECK(contains(text, "le=\"+Inf\"}"));
  CHECK(contains(text, "umm_gpu_nvml_available 0") ||
        contains(text, "umm_gpu_nvml_available 1"));
  if (contains(text, "umm_gpu_nvml_available 0")) {
    CHECK(!contains(text, "umm_gpu_utilization_percent{"));
  }
}

int run_child_mode(const char* mode, const char* trace_path) {
  MetricsHarness h;
  httplib::Client cli("127.0.0.1", h.api->bound_port());
  cli.set_read_timeout(10, 0);
  const json req = {{"model", "served-model"},
                    {"messages", {{{"role", "user"}, {"content", "trace me"}}}}};
  const int count = std::strcmp(mode, "--child-trace-cap") == 0 ? 450 : 1;
  for (int i = 0; i < count; ++i) {
    auto res = cli.Post("/v1/chat/completions", req.dump(), "application/json");
    if (!res || res->status != 200) return 2;
  }
  (void)trace_path;
  return 0;
}

int spawn_self(const char* self_bin, const char* mode, const char* trace_env,
               const std::string& trace_path) {
  const pid_t pid = ::fork();
  if (pid < 0) return -1;
  if (pid == 0) {
    if (trace_env == nullptr) {
      ::unsetenv("UMM_TRACE");
      ::unsetenv("UMM_TRACE_FILE");
    } else {
      ::setenv("UMM_TRACE", trace_env, 1);
      ::setenv("UMM_TRACE_FILE", trace_path.c_str(), 1);
    }
    ::execl(self_bin, self_bin, mode, trace_path.c_str(), nullptr);
    std::_Exit(127);
  }
  int status = 0;
  if (::waitpid(pid, &status, 0) < 0) return -1;
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

void test_trace_via_blackbox_subprocesses(const char* self_bin) {
  const std::string base =
      "/tmp/umm-metrics-trace-" + std::to_string(::getpid());
  const std::string off_path = base + "-off.jsonl";
  const std::string on_path = base + "-on.jsonl";
  const std::string cap_path = base + "-cap.jsonl";
  for (const auto& p : {off_path, on_path, cap_path, cap_path + ".1"}) {
    std::remove(p.c_str());
  }

  // 1. Default (UMM_TRACE unset): no trace file is created.
  CHECK(spawn_self(self_bin, "--child-trace-off", nullptr, off_path) == 0);
  CHECK(file_size(off_path) < 0);

  // 2. UMM_TRACE=1: emits request_start, leg_end, and terminal trace JSONL
  // while omitting unexecuted optional leg durations (no image_ms/tts_ms).
  CHECK(spawn_self(self_bin, "--child-trace-on", "1", on_path) == 0);
  const std::string lines = read_file(on_path);
  CHECK(contains(lines, "\"event\":\"request_start\""));
  CHECK(contains(lines, "\"event\":\"leg_end\""));
  CHECK(contains(lines, "\"status\":\"completed\""));
  CHECK(contains(lines, "\"vlm_ms\":"));
  CHECK(!contains(lines, "\"image_ms\":"));
  CHECK(!lines.empty() && lines.back() == '\n');

  // 3. 64 KiB cap rotation keeps both active and .1 files <= 65536 bytes.
  CHECK(spawn_self(self_bin, "--child-trace-cap", "1", cap_path) == 0);
  const long active = file_size(cap_path);
  const long prev = file_size(cap_path + ".1");
  CHECK(active >= 0 && active <= 65536);
  CHECK(prev >= 0 && prev <= 65536);
  CHECK(active + prev > 0);

  for (const auto& p : {off_path, on_path, cap_path, cap_path + ".1"}) {
    std::remove(p.c_str());
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 3 && std::strncmp(argv[1], "--child-trace-", 14) == 0) {
    return run_child_mode(argv[1], argv[2]);
  }
  test_http_metrics_and_nvml();
  test_trace_via_blackbox_subprocesses(argv[0]);
  if (g_failures != 0) {
    std::fprintf(stderr, "%d blackbox metric check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "metrics_test: all blackbox API checks passed\n");
  return 0;
}
