// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include "serve/upstream_proxy.hpp"
#include "umminfer/obs/metrics.hpp"
#include "umminfer/pipeline/unified.hpp"
#include "umminfer/serve/openai_api.hpp"

namespace umminfer::openai_detail {

using json = nlohmann::json;

int64_t now_epoch_ms();
double elapsed_seconds(std::chrono::steady_clock::time_point started);

struct RouteMetric {
  const char* route;
  const httplib::Response* response;
  ~RouteMetric();
};

struct AsrRequestMetric {
  std::chrono::steady_clock::time_point started{std::chrono::steady_clock::now()};
  int64_t start_epoch_ms = now_epoch_ms();
  std::string request_id;
  std::string status;
  bool counted = false;

  ~AsrRequestMetric();
};

class AdmissionGuard {
 public:
  AdmissionGuard(std::atomic<size_t>& active, size_t limit) : active_(active) {
    if (limit == 0) {
      admitted_ = false;
      return;
    }
    size_t cur = active_.load(std::memory_order_relaxed);
    while (cur < limit) {
      if (active_.compare_exchange_weak(cur, cur + 1,
                                        std::memory_order_acq_rel)) {
        admitted_ = true;
        return;
      }
    }
  }
  ~AdmissionGuard() { release(); }
  AdmissionGuard(const AdmissionGuard&) = delete;
  AdmissionGuard& operator=(const AdmissionGuard&) = delete;

  bool admitted() const { return admitted_; }
  void release() {
    if (!admitted_) return;
    admitted_ = false;
    active_.fetch_sub(1, std::memory_order_acq_rel);
  }

 private:
  std::atomic<size_t>& active_;
  bool admitted_ = false;
};

bool valid_request_id_chars(std::string_view id);
Value json_to_value(const json& j);
json value_to_json(const Value& v);
std::string result_text(const Value& result);
std::string result_finish_reason(const Value& result);
json result_usage(const Value& result);

using SseItem = std::variant<Value, UnifiedEvent>;

struct SseSink {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<SseItem> chunks;
  bool done = false;
  bool has_deltas = false;
  std::string error;
  std::string finish_reason;
  std::optional<UnifiedEvent> terminal;
  // Non-unified terminal stage result; the unified path keeps it in `terminal`.
  std::optional<Value> result;

  void push(Value v, bool delta = false) {
    {
      std::lock_guard<std::mutex> lock(mu);
      has_deltas = has_deltas || delta;
      chunks.emplace_back(std::move(v));
    }
    cv.notify_all();
  }
  void push(UnifiedEvent event) {
    {
      std::lock_guard<std::mutex> lock(mu);
      chunks.emplace_back(std::move(event));
    }
    cv.notify_all();
  }
  void finish(std::string error_message = "") {
    {
      std::lock_guard<std::mutex> lock(mu);
      error = std::move(error_message);
      done = true;
    }
    cv.notify_all();
  }
  bool pop(SseItem& out, double timeout_s) {
    std::unique_lock<std::mutex> lock(mu);
    if (!cv.wait_for(lock, std::chrono::duration<double>(timeout_s),
                     [this] { return !chunks.empty() || done; })) {
      return false;
    }
    if (chunks.empty()) return false;
    out = std::move(chunks.front());
    chunks.pop_front();
    return true;
  }
};

json unified_event_json(const UnifiedEvent& event);
bool decode_wav_ref_audio(std::string_view encoded, std::string& wav_bytes,
                          std::string& error);
bool unified_input_parts(const json& body, std::string& pcm, bool& has_audio,
                         std::string& error);
bool valid_mixed_output(const json& body, bool image_enabled, bool tts_enabled,
                        bool video_enabled, Value::Map& engine_inputs,
                        std::string& error);
void reject_overloaded(httplib::Response& res);

// Synchronous Coordinator request outcome shared by single-stage HTTP routes
// (/v1/audio/transcriptions and /v1/audio/speech). Guarantees abort + release
// on disconnect and timeout, and release before returning on completion.
struct SyncCoordinatorResult {
  enum class Status { kSubmitError, kClientClosed, kTimeout, kSettled };
  Status status = Status::kSubmitError;
  std::string request_id;
  std::string submit_error;
  bool settled_before_close = false;
  std::optional<RequestInfo> info;
};

SyncCoordinatorResult run_sync_coordinator_request(
    Coordinator& coordinator, const httplib::Request& req, StagePayload payload,
    const std::string& stage_name, double timeout_s,
    const std::function<void(const std::string&)>& on_submitted = {});

}  // namespace umminfer::openai_detail

namespace umminfer {

class OpenAIApi::Impl {
 public:
  Impl(Coordinator& coordinator, ApiConfig config);

  void install_routes();
  bool bind();
  bool listen();
  void stop();
  int bound_port() const;

  void set_unified_forwarder(std::function<void(const UnifiedEvent&)> forwarder);
  void set_unified_engine(UnifiedEngine& engine, std::string vlm_stage);

  void on_unified_event(const UnifiedEvent& event);
  void on_stream(const StreamMessage& msg);
  void on_complete(const RequestInfo& info);

 private:
  std::string submit_chat(StagePayload payload);
  bool wait_chat(const std::string& request_id, double timeout_s);
  std::optional<RequestInfo> info_chat(const std::string& request_id);
  void cancel_chat(const std::string& request_id, const char* source = "client");
  void release_chat(const std::string& request_id);

  // stream=false with optional output legs. Collects the same unified events
  // the SSE path forwards, reassembles each complete artifact, and answers with one
  // chat.completion JSON body. A failed optional leg keeps HTTP 200; a failed
  // primary leg answers the same error shape the text-only path uses.
  void handle_nonstream_mixed(const httplib::Request& req,
                              httplib::Response& res, StagePayload payload,
                              int64_t created);
  void handle_transcription(const httplib::Request& req, httplib::Response& res);
  void handle_speech(const httplib::Request& req, httplib::Response& res);
  void handle_chat(const httplib::Request& req, httplib::Response& res);

  Coordinator& coordinator_;
  UnifiedEngine* unified_engine_ = nullptr;
  std::function<void(const UnifiedEvent&)> unified_forwarder_;
  std::string unified_vlm_stage_;
  ApiConfig config_;
  serve_detail::UpstreamProxy proxy_;
  httplib::Server server_;
  int bound_port_ = 0;
  bool bound_ = false;
  std::atomic<size_t> active_http_requests_{0};
  std::mutex streams_mu_;
  std::map<std::string, std::shared_ptr<openai_detail::SseSink>> streams_;
  std::atomic<uint64_t> next_request_id_{0};
};

}  // namespace umminfer
