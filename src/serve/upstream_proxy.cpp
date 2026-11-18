// SPDX-License-Identifier: Apache-2.0
#include "serve/upstream_proxy.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include "client/http.hpp"
#include "umminfer/util/codec.hpp"

namespace umminfer {
namespace {

using json = nlohmann::json;

constexpr size_t kRelayQueueBytes = 4u << 20;
constexpr size_t kMaxJsonUpstreamBytes = 64u << 20;
constexpr size_t kMaxBinaryUpstreamBytes = 512u << 20;

class UpstreamRelay {
 public:
  using HttpRequest = client::detail::HttpRequest;

  UpstreamRelay() = default;
  UpstreamRelay(const UpstreamRelay&) = delete;
  UpstreamRelay& operator=(const UpstreamRelay&) = delete;

  ~UpstreamRelay() {
    cancel();
    if (worker_.joinable()) worker_.join();
  }

  void start(HttpRequest request, std::function<void()> on_transfer_done = {}) {
    worker_ = std::thread([this, request = std::move(request),
                           done_hook = std::move(on_transfer_done)]() mutable {
      request.cancel = &cancel_flag_;
      try {
        client::detail::perform(request);
      } catch (const client::Error& error) {
        std::lock_guard<std::mutex> lock(mu_);
        error_ = error.what();
        error_code_ = error.code();
        status_ = error.status();
        retry_after_ = error.retry_after();
      } catch (const std::exception& error) {
        std::lock_guard<std::mutex> lock(mu_);
        error_ = error.what();
      }
      try {
        if (done_hook) done_hook();
      } catch (const std::exception& error) {
        std::lock_guard<std::mutex> lock(mu_);
        if (error_.empty()) error_ = error.what();
      } catch (...) {
        std::lock_guard<std::mutex> lock(mu_);
        if (error_.empty()) error_ = "upstream stream finalization failed";
      }
      {
        std::lock_guard<std::mutex> lock(mu_);
        done_ = true;
      }
      cv_.notify_all();
    });
  }

  bool push(std::string bytes) {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [this] {
      return cancel_flag_.load() || queued_bytes_ < kRelayQueueBytes;
    });
    if (cancel_flag_.load()) return false;
    started_ = true;
    queued_bytes_ += bytes.size();
    queue_.push_back(std::move(bytes));
    cv_.notify_all();
    return true;
  }

  void cancel() {
    cancel_flag_.store(true);
    cv_.notify_all();
  }

  bool pop_wait(std::string& out, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait_for(lock, timeout, [this] { return !queue_.empty() || done_; });
    if (queue_.empty()) return false;
    out = std::move(queue_.front());
    queue_.pop_front();
    queued_bytes_ -= out.size();
    cv_.notify_all();
    return true;
  }

  bool wait_started(double seconds) {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait_for(lock, std::chrono::duration<double>(seconds),
                 [this] { return started_ || done_; });
    return started_;
  }

  bool settled_without_start() {
    std::lock_guard<std::mutex> lock(mu_);
    return done_ && !started_;
  }
  bool finished() {
    std::lock_guard<std::mutex> lock(mu_);
    return done_ && queue_.empty();
  }
  bool has_error() {
    std::lock_guard<std::mutex> lock(mu_);
    return !error_.empty();
  }
  std::string error_text() {
    std::lock_guard<std::mutex> lock(mu_);
    return error_;
  }
  client::ErrorCode error_code() {
    std::lock_guard<std::mutex> lock(mu_);
    return error_code_;
  }
  client::Milliseconds retry_after() {
    std::lock_guard<std::mutex> lock(mu_);
    return retry_after_;
  }
  long upstream_status() {
    std::lock_guard<std::mutex> lock(mu_);
    return status_;
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::string> queue_;
  size_t queued_bytes_ = 0;
  bool started_ = false;
  bool done_ = false;
  std::string error_;
  client::ErrorCode error_code_ = client::ErrorCode::kBadResponse;
  long status_ = 0;
  client::Milliseconds retry_after_{0};
  std::atomic<bool> cancel_flag_{false};
  std::thread worker_;
};

class SseFramer {
 public:
  explicit SseFramer(UpstreamRelay* relay) : relay_(relay) {}

  bool feed(std::string_view bytes) {
    return parser_.feed(bytes,
                        [this](std::string_view data) { return emit(data); });
  }
  void finish() {
    parser_.finish([this](std::string_view data) { return emit(data); });
  }
  bool saw_done() const { return saw_done_.load(); }

 private:
  bool emit(std::string_view data) {
    if (data == "[DONE]") saw_done_.store(true);
    std::string framed;
    size_t start = 0;
    while (true) {
      const size_t newline = data.find('\n', start);
      const std::string_view line =
          newline == std::string_view::npos
              ? data.substr(start)
              : data.substr(start, newline - start);
      framed += "data: ";
      framed.append(line);
      framed += "\n";
      if (newline == std::string_view::npos) break;
      start = newline + 1;
    }
    framed += "\n";
    return relay_->push(std::move(framed));
  }

  client::detail::SSEParser parser_;
  UpstreamRelay* relay_;
  std::atomic<bool> saw_done_{false};
};

int upstream_error_status(client::ErrorCode code, long status) {
  if (status >= 400 && status <= 599) return static_cast<int>(status);
  if (code == client::ErrorCode::kTimeout) return 504;
  if (code == client::ErrorCode::kAuthentication) return 502;
  return 502;
}

bool is_breaker_trip_error(client::ErrorCode code, long status) {
  if (code == client::ErrorCode::kCancelled) return false;
  if (status >= 400 && status < 500) return false;
  return true;
}

client::detail::HttpRequest make_upstream_request(
    const ApiConfig& config, const std::string& upstream_base_url,
    const std::string& path) {
  client::detail::HttpRequest out;
  out.url = upstream_base_url + path;
  out.bearer = config.upstream_api_key;
  out.timeout = client::Milliseconds{static_cast<int64_t>(
      std::max(1.0, config.request_timeout_s) * 1000.0)};
  out.connect_timeout = client::Milliseconds{10000};
  out.body_limit = kMaxJsonUpstreamBytes;
  out.no_proxy = out.url.rfind("http://", 0) == 0;
  return out;
}

std::string scrubbed_text(const ApiConfig& config, const std::string& text) {
  return util::utf8_sanitized(
      client::detail::redact(text, config.upstream_api_key));
}

bool served_model_matches(const ApiConfig& config, const json& body,
                          std::string& error) {
  if (!body.contains("model") || body["model"].is_null()) return true;
  if (!body["model"].is_string()) {
    error = "model must be a string";
    return false;
  }
  const std::string requested = body["model"].get<std::string>();
  if (requested != config.served_model_name) {
    error = "unsupported model: " + requested + "; this server serves " +
            config.served_model_name;
    return false;
  }
  return true;
}

bool valid_image_generation(const ApiConfig& config, const json& body,
                            std::string& error) {
  if (!body.is_object()) {
    error = "body must be a JSON object";
    return false;
  }
  if (!body.contains("prompt") || !body["prompt"].is_string() ||
      body["prompt"].get_ref<const std::string&>().empty()) {
    error = "prompt must be a non-empty string";
    return false;
  }
  static const std::set<std::string> kAllowed = {
      "model", "prompt", "n", "size", "response_format", "output_format",
      "background", "seed", "generator_device", "user", "negative_prompt",
      "guidance_scale", "true_cfg_scale", "num_inference_steps", "quality",
      "output_quality", "output_compression", "enhance_prompt", "task_type",
      "style"};
  for (auto it = body.begin(); it != body.end(); ++it) {
    if (!it.value().is_null() && !kAllowed.count(it.key())) {
      error = "unsupported field: " + it.key();
      return false;
    }
  }
  if (!served_model_matches(config, body, error)) return false;
  if (body.contains("response_format") && !body["response_format"].is_null()) {
    if (!body["response_format"].is_string() ||
        body["response_format"].get<std::string>() != "b64_json") {
      error = "response_format must be b64_json: url responses would expose "
              "upstream storage";
      return false;
    }
  }
  return true;
}

bool valid_video_create(const ApiConfig& config, const json& body,
                        std::string& error) {
  if (!body.is_object()) {
    error = "body must be a JSON object";
    return false;
  }
  if (!body.contains("prompt") || !body["prompt"].is_string() ||
      body["prompt"].get_ref<const std::string&>().empty()) {
    error = "prompt must be a non-empty string";
    return false;
  }
  static const std::set<std::string> kAllowed = {
      "model", "prompt", "task", "conditions", "target", "seed", "size",
      "seconds", "n", "fps", "num_frames", "negative_prompt", "enhance_prompt",
      "input_reference", "reference_url", "video_url", "video_path",
      "generator_device", "task_type", "num_outputs_per_prompt",
      "num_inference_steps", "flow_shift", "audio_flow_shift",
      "reference_resize_mode"};
  for (auto it = body.begin(); it != body.end(); ++it) {
    if (!it.value().is_null() && !kAllowed.count(it.key())) {
      error = "unsupported field: " + it.key();
      return false;
    }
  }
  if (!served_model_matches(config, body, error)) return false;
  if (body.contains("task") && !body["task"].is_null()) {
    if (!body["task"].is_string()) {
      error = "task must be a string";
      return false;
    }
    const std::string task = body["task"].get<std::string>();
    if (task != "t2va" && task != "fl2va" && task != "ref2va") {
      error = "task must be t2va, fl2va or ref2va";
      return false;
    }
  }
  if (body.contains("conditions") && !body["conditions"].is_null()) {
    if (!body["conditions"].is_array()) {
      error = "conditions must be an array";
      return false;
    }
    for (const auto& condition : body["conditions"]) {
      if (!condition.is_object() || !condition.contains("type") ||
          !condition["type"].is_string() || !condition.contains("uri") ||
          !condition["uri"].is_string()) {
        error = "each condition requires a string type and uri";
        return false;
      }
      const std::string type = condition["type"].get<std::string>();
      if (type != "image" && type != "video" && type != "audio" &&
          type != "video_audio") {
        error = "condition type must be image, video, audio or video_audio";
        return false;
      }
      if (condition.contains("role") && !condition["role"].is_null()) {
        if (!condition["role"].is_string()) {
          error = "condition role must be a string";
          return false;
        }
        const std::string role = condition["role"].get<std::string>();
        if (role != "keyframe" && role != "reference") {
          error = "condition role must be keyframe or reference";
          return false;
        }
      }
    }
  }
  return true;
}

bool valid_qwen_chat(const ApiConfig& config, const json& body,
                     std::string& error) {
  if (!body.is_object()) {
    error = "body must be a JSON object";
    return false;
  }
  if (!served_model_matches(config, body, error)) return false;
  if (!body.contains("messages") || !body["messages"].is_array() ||
      body["messages"].empty()) {
    error = "messages must be a non-empty array";
    return false;
  }
  for (const auto& message : body["messages"]) {
    if (!message.is_object() || !message.contains("role") ||
        !message["role"].is_string() || !message.contains("content")) {
      error = "each message requires a string role and content";
      return false;
    }
    const std::string role = message["role"].get<std::string>();
    if (role != "system" && role != "user" && role != "assistant") {
      error = "message role must be system, user or assistant";
      return false;
    }
    const json& content = message["content"];
    if (content.is_string()) continue;
    if (!content.is_array()) {
      error = "message content must be a string or a content part array";
      return false;
    }
    for (const auto& part : content) {
      if (!part.is_object() || !part.contains("type") ||
          !part["type"].is_string()) {
        error = "each content part requires a string type";
        return false;
      }
      const std::string type = part["type"].get<std::string>();
      if (type == "text") {
        if (!part.contains("text") || !part["text"].is_string()) {
          error = "text part requires a string text";
          return false;
        }
      } else if (type == "image_url") {
        if (!part.contains("image_url") || !part["image_url"].is_object() ||
            !part["image_url"].contains("url") ||
            !part["image_url"]["url"].is_string()) {
          error = "image_url part requires image_url.url";
          return false;
        }
      } else if (type == "input_audio") {
        if (!part.contains("input_audio") || !part["input_audio"].is_object() ||
            !part["input_audio"].contains("data") ||
            !part["input_audio"]["data"].is_string() ||
            !part["input_audio"].contains("format") ||
            !part["input_audio"]["format"].is_string()) {
          error = "input_audio part requires input_audio.data and format";
          return false;
        }
      } else if (type == "video") {
        if (!part.contains("video") || !part["video"].is_array() ||
            part["video"].size() < 2 || part["video"].size() > 2048) {
          error = "video part requires 2 to 2048 frame urls";
          return false;
        }
        for (const auto& url : part["video"]) {
          if (!url.is_string()) {
            error = "video part urls must be strings";
            return false;
          }
        }
      } else if (type == "video_url") {
        if (!part.contains("video_url") || !part["video_url"].is_object() ||
            !part["video_url"].contains("url") ||
            !part["video_url"]["url"].is_string()) {
          error = "video_url part requires video_url.url";
          return false;
        }
        if (part.contains("fps") && !part["fps"].is_null()) {
          if (!part["fps"].is_number()) {
            error = "video fps must be a number";
            return false;
          }
          const double fps = part["fps"].get<double>();
          if (fps < 0.1 || fps > 10) {
            error = "video fps must be between 0.1 and 10";
            return false;
          }
        }
      } else {
        error = "unsupported content part type: " + type;
        return false;
      }
    }
  }
  static const std::set<std::string> kAllowed = {
      "model", "messages", "stream", "stream_options", "modalities", "audio",
      "temperature", "top_p", "max_tokens", "max_completion_tokens", "stop",
      "seed", "response_format"};
  for (auto it = body.begin(); it != body.end(); ++it) {
    if (!it.value().is_null() && !kAllowed.count(it.key())) {
      error = "unsupported field: " + it.key();
      return false;
    }
  }
  if (body.contains("stream") && !body["stream"].is_boolean()) {
    error = "stream must be a boolean";
    return false;
  }
  const bool stream = body.value("stream", false);
  if (body.contains("stream_options") && !body["stream_options"].is_null()) {
    const json& options = body["stream_options"];
    if (!options.is_object()) {
      error = "stream_options must be an object";
      return false;
    }
    for (auto it = options.begin(); it != options.end(); ++it) {
      if (it.key() != "include_usage") {
        error = "unsupported stream_options field: " + it.key();
        return false;
      }
    }
    if (options.contains("include_usage") &&
        !options["include_usage"].is_boolean()) {
      error = "stream_options.include_usage must be a boolean";
      return false;
    }
  }
  bool audio_output = false;
  if (body.contains("modalities") && !body["modalities"].is_null()) {
    const json& modalities = body["modalities"];
    if (!modalities.is_array() || modalities.empty()) {
      error = "modalities must be a non-empty array";
      return false;
    }
    for (const auto& modality : modalities) {
      if (!modality.is_string()) {
        error = "modalities values must be strings";
        return false;
      }
      const std::string name = modality.get<std::string>();
      if (name != "text" && name != "audio") {
        error = "modalities values must be text or audio";
        return false;
      }
      if (name == "audio") audio_output = true;
    }
  }
  if (body.contains("audio") && !body["audio"].is_null()) {
    const json& audio = body["audio"];
    if (!audio_output) {
      error = "audio requires modalities to include audio";
      return false;
    }
    if (!audio.is_object() || !audio.contains("voice") ||
        !audio["voice"].is_string() ||
        audio["voice"].get_ref<const std::string&>().empty() ||
        !audio.contains("format") || !audio["format"].is_string()) {
      error = "audio requires a non-empty voice and a format";
      return false;
    }
    if (audio["format"].get<std::string>() != "wav") {
      error = "audio.format must be wav: it is the only documented format";
      return false;
    }
  } else if (audio_output) {
    error = "audio output requires the audio object with voice and format";
    return false;
  }
  if (audio_output && !stream) {
    error = "audio output requires stream=true";
    return false;
  }
  if (body.contains("response_format") && !body["response_format"].is_null()) {
    const json& format = body["response_format"];
    if (!format.is_object() || !format.contains("type") ||
        !format["type"].is_string() ||
        format["type"].get<std::string>() != "text") {
      error = "only response_format type text is supported";
      return false;
    }
  }
  return true;
}

}  // namespace

std::string validate_upstream_config(const ApiConfig& config) {
  const bool any = !config.upstream_kind.empty() ||
                   !config.upstream_base_url.empty() ||
                   !config.upstream_model.empty() ||
                   !config.upstream_api_key.empty() ||
                   !config.upstream_variant.empty();
  if (!any) return {};
  if (config.upstream_kind.empty()) {
    return "upstream_kind is required with the other upstream fields";
  }
  if (config.upstream_base_url.empty()) return "upstream_base_url is required";
  if (config.upstream_model.empty()) return "upstream_model is required";
  if (config.upstream_variant.empty()) return "upstream_variant is required";
  if (config.upstream_kind != "qwen" && config.upstream_kind != "sglang") {
    return "upstream_kind must be qwen or sglang";
  }
  if (config.upstream_kind == "qwen") {
    if (config.upstream_variant != "omni") {
      return "upstream_variant must be omni for a qwen upstream";
    }
    if (config.upstream_api_key.empty()) {
      return "upstream_api_key is required for a qwen upstream";
    }
  } else if (config.upstream_variant != "image" &&
             config.upstream_variant != "video") {
    return "upstream_variant must be image or video for an sglang upstream";
  }
  if (!config.upstream_api_key.empty() &&
      config.upstream_api_key == config.api_key) {
    return "upstream_api_key must differ from api_key";
  }
  try {
    client::detail::validate_base_url(config.upstream_base_url, false);
  } catch (const std::exception& error) {
    return std::string("invalid upstream_base_url: ") + error.what();
  }
  return {};
}

namespace serve_detail {

UpstreamProxy::UpstreamProxy(const ApiConfig& config) : config_(config) {
  if (config_.served_model_name.empty()) {
    config_.served_model_name = config_.model;
  }
  upstream_base_url_ = config_.upstream_base_url;
  while (upstream_base_url_.size() > 8 && upstream_base_url_.back() == '/') {
    upstream_base_url_.pop_back();
  }
}

void UpstreamProxy::reject(httplib::Response& res, int status,
                           const std::string& message,
                           const char* type) const {
  res.status = status;
  res.set_content(
      json{{"error", {{"message", util::utf8_sanitized(message)}, {"type", type}}}}
          .dump(),
      "application/json");
}

void UpstreamProxy::reject_route(httplib::Response& res,
                                 const char* route) const {
  const std::string variant =
      config_.upstream_variant.empty() ? "none" : config_.upstream_variant;
  reject(res, 501,
         std::string(route) +
             " is not served by this process: the configured upstream variant is '" +
             variant + "'",
         "unsupported_route");
}

bool UpstreamProxy::check_circuit_breaker(httplib::Response& res) {
  if (config_.upstream_circuit_breaker_failures <= 0) return true;
  std::lock_guard<std::mutex> lock(cb_mu_);
  const auto now = std::chrono::steady_clock::now();
  if (now < cb_open_until_) {
    const auto remaining_s = std::max<int64_t>(
        1, std::chrono::duration_cast<std::chrono::seconds>(cb_open_until_ - now)
                   .count() +
               1);
    res.set_header("Retry-After", std::to_string(remaining_s));
    reject(res, 503,
           "upstream circuit breaker is open after repeated upstream failures",
           "upstream_circuit_open");
    return false;
  }
  return true;
}

void UpstreamProxy::record_upstream_outcome(bool trip_failure) {
  if (config_.upstream_circuit_breaker_failures <= 0) return;
  std::lock_guard<std::mutex> lock(cb_mu_);
  if (!trip_failure) {
    consecutive_failures_ = 0;
    cb_open_until_ = {};
    return;
  }
  if (++consecutive_failures_ >= config_.upstream_circuit_breaker_failures) {
    cb_open_until_ = std::chrono::steady_clock::now() +
                     std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                         std::chrono::duration<double>(
                             std::max(0.1, config_.upstream_circuit_breaker_cooldown_s)));
  }
}

namespace {

void reject_upstream_err(const UpstreamProxy& self, const ApiConfig& config,
                         httplib::Response& res, client::ErrorCode code,
                         long status, const std::string& text,
                         client::Milliseconds retry_after = client::Milliseconds{0}) {
  self.reject(res, upstream_error_status(code, status),
              scrubbed_text(config, text), "upstream_error");
  if (status == 503 && retry_after.count() > 0) {
    res.set_header("Retry-After",
                   std::to_string((retry_after.count() + 999) / 1000));
  }
}

}  // namespace

void UpstreamProxy::proxy_qwen_chat(const httplib::Request& req,
                                    httplib::Response& res) {
  const json body = json::parse(req.body, nullptr, false);
  std::string error;
  if (!valid_qwen_chat(config_, body, error)) {
    reject(res, 400, error, "invalid_request");
    return;
  }
  if (!check_circuit_breaker(res)) return;
  const bool stream = body.value("stream", false);
  json forwarded = body;
  forwarded["model"] = config_.upstream_model;
  auto out = make_upstream_request(config_, upstream_base_url_,
                                   "/chat/completions");
  out.method = client::detail::HttpMethod::kPost;
  out.body = forwarded.dump();
  out.headers.emplace("Content-Type", "application/json");
  if (!stream) {
    out.accept = "application/json";
    out.require_content_type = "application/json";
    std::atomic<bool> cancel{false};
    std::atomic<bool> done{false};
    out.cancel = &cancel;
    std::thread watcher([&req, &cancel, &done] {
      while (!done.load() && !req.is_connection_closed()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
      if (!done.load()) cancel.store(true);
    });
    try {
      auto response = client::detail::perform(out);
      done.store(true);
      watcher.join();
      record_upstream_outcome(false);
      res.status = static_cast<int>(response.status);
      res.set_content(response.body, "application/json");
    } catch (const client::Error& err) {
      done.store(true);
      watcher.join();
      if (err.code() == client::ErrorCode::kCancelled) return;
      record_upstream_outcome(is_breaker_trip_error(err.code(), err.status()));
      reject_upstream_err(*this, config_, res, err.code(), err.status(),
                          err.what(), err.retry_after());
    } catch (const std::exception& err) {
      done.store(true);
      watcher.join();
      record_upstream_outcome(true);
      reject(res, 502, scrubbed_text(config_, err.what()), "upstream_error");
    }
    return;
  }

  out.accept = "text/event-stream";
  out.require_content_type = "text/event-stream";
  out.stream_limit = kMaxJsonUpstreamBytes;
  auto relay = std::make_shared<UpstreamRelay>();
  auto framer = std::make_shared<SseFramer>(relay.get());
  out.on_body = [framer](std::string_view bytes) { return framer->feed(bytes); };
  relay->start(std::move(out), [framer] { framer->finish(); });

  const auto deadline =
      std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(config_.request_timeout_s));
  while (!relay->wait_started(0.2)) {
    if (relay->settled_without_start()) {
      relay->cancel();
      if (relay->has_error()) {
        record_upstream_outcome(
            is_breaker_trip_error(relay->error_code(), relay->upstream_status()));
        reject_upstream_err(*this, config_, res, relay->error_code(),
                            relay->upstream_status(), relay->error_text(),
                            relay->retry_after());
      } else {
        record_upstream_outcome(true);
        reject(res, 502, "upstream ended before sending a response body",
               "upstream_error");
      }
      return;
    }
    if (req.is_connection_closed()) {
      relay->cancel();
      return;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      relay->cancel();
      record_upstream_outcome(true);
      if (relay->has_error()) {
        reject_upstream_err(*this, config_, res, relay->error_code(),
                            relay->upstream_status(), relay->error_text(),
                            relay->retry_after());
      } else {
        reject(res, 504, "upstream did not start a response", "upstream_error");
      }
      return;
    }
  }
  record_upstream_outcome(false);
  res.set_chunked_content_provider(
      "text/event-stream", [relay, framer](size_t, httplib::DataSink& sink) {
        while (true) {
          if (!sink.is_writable()) {
            relay->cancel();
            return false;
          }
          std::string piece;
          if (relay->pop_wait(piece, std::chrono::milliseconds(200))) {
            if (!sink.write(piece.data(), piece.size())) {
              relay->cancel();
              return false;
            }
            continue;
          }
          if (relay->finished()) break;
        }
        if (relay->has_error()) return false;
        if (!framer->saw_done()) return false;
        sink.done();
        return true;
      });
}

void UpstreamProxy::handle_image_generation(const httplib::Request& req,
                                            httplib::Response& res) {
  const json body = json::parse(req.body, nullptr, false);
  std::string error;
  if (!valid_image_generation(config_, body, error)) {
    reject(res, 400, error, "invalid_request");
    return;
  }
  if (!check_circuit_breaker(res)) return;
  json forwarded = body;
  forwarded["model"] = config_.upstream_model;
  auto out = make_upstream_request(config_, upstream_base_url_,
                                   "/v1/images/generations");
  out.method = client::detail::HttpMethod::kPost;
  out.body = forwarded.dump();
  out.headers.emplace("Content-Type", "application/json");
  out.accept = "application/json";
  out.require_content_type = "application/json";

  std::atomic<bool> cancel{false};
  std::atomic<bool> done{false};
  out.cancel = &cancel;
  std::thread watcher([&req, &cancel, &done] {
    while (!done.load() && !req.is_connection_closed()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!done.load()) cancel.store(true);
  });
  try {
    auto response = client::detail::perform(out);
    done.store(true);
    watcher.join();
    record_upstream_outcome(false);
    res.status = static_cast<int>(response.status);
    res.set_content(response.body, "application/json");
  } catch (const client::Error& err) {
    done.store(true);
    watcher.join();
    if (err.code() == client::ErrorCode::kCancelled) return;
    record_upstream_outcome(is_breaker_trip_error(err.code(), err.status()));
    reject_upstream_err(*this, config_, res, err.code(), err.status(),
                        err.what(), err.retry_after());
  } catch (const std::exception& err) {
    done.store(true);
    watcher.join();
    record_upstream_outcome(true);
    reject(res, 502, scrubbed_text(config_, err.what()), "upstream_error");
  }
}

void UpstreamProxy::handle_image_edit(const httplib::Request& req,
                                      httplib::Response& res) {
  if (!req.is_multipart_form_data()) {
    reject(res, 400, "one multipart form is required", "invalid_request");
    return;
  }
  for (const auto& part : req.form.files) {
    if (part.first != "image") {
      reject(res, 400, "unsupported multipart file part: " + part.first,
             "invalid_request");
      return;
    }
  }
  if (req.form.get_file_count("image") == 0) {
    reject(res, 400, "at least one image part is required", "invalid_request");
    return;
  }
  static const std::set<std::string> kAllowed = {
      "prompt", "model", "n", "response_format", "size", "output_format",
      "background", "seed", "generator_device", "user", "negative_prompt",
      "guidance_scale", "true_cfg_scale", "num_inference_steps", "quality",
      "output_quality", "output_compression", "enhance_prompt", "url",
      "task_type"};
  std::map<std::string, int> counts;
  for (const auto& part : req.form.fields) {
    ++counts[part.first];
    if (!kAllowed.count(part.first)) {
      reject(res, 400, "unsupported multipart field: " + part.first,
             "invalid_request");
      return;
    }
  }
  for (const auto& [name, count] : counts) {
    if (count > 1 && name != "url") {
      reject(res, 400, "duplicate multipart field: " + name, "invalid_request");
      return;
    }
  }
  const std::string prompt = req.form.get_field("prompt");
  if (prompt.empty()) {
    reject(res, 400, "prompt is required", "invalid_request");
    return;
  }
  const std::string requested_model = req.form.get_field("model");
  if (!requested_model.empty() && requested_model != config_.served_model_name) {
    reject(res, 400,
           "unsupported model: " + requested_model + "; this server serves " +
               config_.served_model_name,
           "invalid_request");
    return;
  }
  const std::string format = req.form.get_field("response_format");
  if (!format.empty() && format != "b64_json") {
    reject(res, 400,
           "response_format must be b64_json: url responses would expose "
           "upstream storage",
           "invalid_request");
    return;
  }
  if (!check_circuit_breaker(res)) return;
  auto out = make_upstream_request(config_, upstream_base_url_,
                                   "/v1/images/edits");
  out.method = client::detail::HttpMethod::kPost;
  out.require_content_type = "application/json";
  for (const auto& [name, part] : req.form.files) {
    client::detail::MimePart mime;
    mime.name = "image";
    mime.data = part.content;
    mime.filename = "image";
    mime.content_type =
        part.content_type.empty() ? "application/octet-stream" : part.content_type;
    out.mime_parts.push_back(std::move(mime));
  }
  for (const auto& [name, field] : req.form.fields) {
    if (name == "model") continue;
    client::detail::MimePart mime;
    mime.name = name;
    mime.data = field.content;
    out.mime_parts.push_back(std::move(mime));
  }
  client::detail::MimePart model_part;
  model_part.name = "model";
  model_part.data = config_.upstream_model;
  out.mime_parts.push_back(std::move(model_part));

  std::atomic<bool> cancel{false};
  std::atomic<bool> done{false};
  out.cancel = &cancel;
  std::thread watcher([&req, &cancel, &done] {
    while (!done.load() && !req.is_connection_closed()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!done.load()) cancel.store(true);
  });
  try {
    auto response = client::detail::perform(out);
    done.store(true);
    watcher.join();
    record_upstream_outcome(false);
    res.status = static_cast<int>(response.status);
    res.set_content(response.body, "application/json");
  } catch (const client::Error& err) {
    done.store(true);
    watcher.join();
    if (err.code() == client::ErrorCode::kCancelled) return;
    record_upstream_outcome(is_breaker_trip_error(err.code(), err.status()));
    reject_upstream_err(*this, config_, res, err.code(), err.status(),
                        err.what(), err.retry_after());
  } catch (const std::exception& err) {
    done.store(true);
    watcher.join();
    record_upstream_outcome(true);
    reject(res, 502, scrubbed_text(config_, err.what()), "upstream_error");
  }
}

void UpstreamProxy::handle_video_create(const httplib::Request& req,
                                        httplib::Response& res) {
  const json body = json::parse(req.body, nullptr, false);
  std::string error;
  if (!valid_video_create(config_, body, error)) {
    reject(res, 400, error, "invalid_request");
    return;
  }
  if (!check_circuit_breaker(res)) return;
  json forwarded = body;
  forwarded["model"] = config_.upstream_model;
  auto out =
      make_upstream_request(config_, upstream_base_url_, "/v1/videos");
  out.method = client::detail::HttpMethod::kPost;
  out.body = forwarded.dump();
  out.headers.emplace("Content-Type", "application/json");
  out.accept = "application/json";
  out.require_content_type = "application/json";

  std::atomic<bool> cancel{false};
  std::atomic<bool> done{false};
  out.cancel = &cancel;
  std::thread watcher([&req, &cancel, &done] {
    while (!done.load() && !req.is_connection_closed()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!done.load()) cancel.store(true);
  });
  try {
    auto response = client::detail::perform(out);
    done.store(true);
    watcher.join();
    record_upstream_outcome(false);
    res.status = static_cast<int>(response.status);
    res.set_content(response.body, "application/json");
  } catch (const client::Error& err) {
    done.store(true);
    watcher.join();
    if (err.code() == client::ErrorCode::kCancelled) return;
    record_upstream_outcome(is_breaker_trip_error(err.code(), err.status()));
    reject_upstream_err(*this, config_, res, err.code(), err.status(),
                        err.what(), err.retry_after());
  } catch (const std::exception& err) {
    done.store(true);
    watcher.join();
    record_upstream_outcome(true);
    reject(res, 502, scrubbed_text(config_, err.what()), "upstream_error");
  }
}

void UpstreamProxy::handle_video_status(const httplib::Request& req,
                                        const std::string& video_id,
                                        httplib::Response& res) {
  if (!check_circuit_breaker(res)) return;
  auto out = make_upstream_request(config_, upstream_base_url_,
                                   "/v1/videos/" + video_id);
  out.method = client::detail::HttpMethod::kGet;
  out.accept = "application/json";
  out.require_content_type = "application/json";

  std::atomic<bool> cancel{false};
  std::atomic<bool> done{false};
  out.cancel = &cancel;
  std::thread watcher([&req, &cancel, &done] {
    while (!done.load() && !req.is_connection_closed()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!done.load()) cancel.store(true);
  });
  try {
    auto response = client::detail::perform(out);
    done.store(true);
    watcher.join();
    record_upstream_outcome(false);
    res.status = static_cast<int>(response.status);
    res.set_content(response.body, "application/json");
  } catch (const client::Error& err) {
    done.store(true);
    watcher.join();
    if (err.code() == client::ErrorCode::kCancelled) return;
    record_upstream_outcome(is_breaker_trip_error(err.code(), err.status()));
    reject_upstream_err(*this, config_, res, err.code(), err.status(),
                        err.what(), err.retry_after());
  } catch (const std::exception& err) {
    done.store(true);
    watcher.join();
    record_upstream_outcome(true);
    reject(res, 502, scrubbed_text(config_, err.what()), "upstream_error");
  }
}

void UpstreamProxy::handle_video_content(const httplib::Request& req,
                                         const std::string& video_id,
                                         httplib::Response& res) {
  bool seen_variant = false;
  std::string variant;
  for (const auto& [name, value] : req.params) {
    if (name != "variant") {
      reject(res, 400, "unsupported query parameter: " + name, "invalid_request");
      return;
    }
    if (seen_variant) {
      reject(res, 400, "duplicate query parameter: variant", "invalid_request");
      return;
    }
    seen_variant = true;
    int parsed = 0;
    const char* begin = value.data();
    const char* end = begin + value.size();
    const auto result = std::from_chars(begin, end, parsed);
    if (value.empty() || result.ec != std::errc{} || result.ptr != end ||
        parsed < 0) {
      reject(res, 400, "variant must be a non-negative integer",
             "invalid_request");
      return;
    }
    variant = value;
  }
  if (!check_circuit_breaker(res)) return;
  std::string path = "/v1/videos/" + video_id + "/content";
  if (seen_variant) path += "?variant=" + variant;
  auto out = make_upstream_request(config_, upstream_base_url_, path);
  out.method = client::detail::HttpMethod::kGet;
  out.accept = "video/mp4";
  out.require_content_type = "video/mp4";
  out.stream_limit = kMaxBinaryUpstreamBytes;

  auto relay = std::make_shared<UpstreamRelay>();
  UpstreamRelay* raw = relay.get();
  out.on_body = [raw](std::string_view bytes) {
    return raw->push(std::string(bytes));
  };
  relay->start(std::move(out));

  const auto deadline =
      std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(config_.request_timeout_s));
  while (!relay->wait_started(0.2)) {
    if (relay->settled_without_start()) {
      relay->cancel();
      if (relay->has_error()) {
        record_upstream_outcome(
            is_breaker_trip_error(relay->error_code(), relay->upstream_status()));
        reject_upstream_err(*this, config_, res, relay->error_code(),
                            relay->upstream_status(), relay->error_text(),
                            relay->retry_after());
      } else {
        record_upstream_outcome(true);
        reject(res, 502, "upstream ended before sending a response body",
               "upstream_error");
      }
      return;
    }
    if (req.is_connection_closed()) {
      relay->cancel();
      return;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      relay->cancel();
      record_upstream_outcome(true);
      if (relay->has_error()) {
        reject_upstream_err(*this, config_, res, relay->error_code(),
                            relay->upstream_status(), relay->error_text(),
                            relay->retry_after());
      } else {
        reject(res, 504, "upstream did not start a response", "upstream_error");
      }
      return;
    }
  }
  record_upstream_outcome(false);
  res.set_chunked_content_provider(
      "video/mp4", [relay](size_t, httplib::DataSink& sink) {
        while (true) {
          if (!sink.is_writable()) {
            relay->cancel();
            return false;
          }
          std::string piece;
          if (relay->pop_wait(piece, std::chrono::milliseconds(200))) {
            if (!sink.write(piece.data(), piece.size())) {
              relay->cancel();
              return false;
            }
            continue;
          }
          if (relay->finished()) break;
        }
        if (relay->has_error()) return false;
        sink.done();
        return true;
      });
}

void UpstreamProxy::handle_video_delete(const httplib::Request& req,
                                        const std::string& video_id,
                                        httplib::Response& res) {
  if (!check_circuit_breaker(res)) return;
  auto out = make_upstream_request(config_, upstream_base_url_,
                                   "/v1/videos/" + video_id);
  out.method = client::detail::HttpMethod::kDelete;
  out.accept = "application/json";
  out.require_content_type = "application/json";

  std::atomic<bool> cancel{false};
  std::atomic<bool> done{false};
  out.cancel = &cancel;
  std::thread watcher([&req, &cancel, &done] {
    while (!done.load() && !req.is_connection_closed()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!done.load()) cancel.store(true);
  });
  try {
    auto response = client::detail::perform(out);
    done.store(true);
    watcher.join();
    record_upstream_outcome(false);
    res.status = static_cast<int>(response.status);
    res.set_content(response.body, "application/json");
  } catch (const client::Error& err) {
    done.store(true);
    watcher.join();
    if (err.code() == client::ErrorCode::kCancelled) return;
    record_upstream_outcome(is_breaker_trip_error(err.code(), err.status()));
    reject_upstream_err(*this, config_, res, err.code(), err.status(),
                        err.what(), err.retry_after());
  } catch (const std::exception& err) {
    done.store(true);
    watcher.join();
    record_upstream_outcome(true);
    reject(res, 502, scrubbed_text(config_, err.what()), "upstream_error");
  }
}

}  // namespace serve_detail
}  // namespace umminfer
