// SPDX-License-Identifier: Apache-2.0
#include "umminfer/serve/openai_api.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <ctime>
#include <deque>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <variant>

#include "asr_common.hpp"
#include "serve/upstream_proxy.hpp"
#include "umminfer/client/client.hpp"
#include "umminfer/obs/metrics.hpp"
#include "umminfer/pipeline/unified.hpp"
#include "umminfer/util/codec.hpp"

namespace umminfer {
namespace {

using json = nlohmann::json;

constexpr size_t kMaxHttpPayloadBytes = 64u << 20;
// How long one socket read may wait for the next byte. The body is read before
// any handler runs, so this value alone decides whether a slow upload survives.
// httplib's 5 s default aborts a 1 MB audio upload on any link that stalls for
// longer than that: measured on the H20B window, the server logged
// "Failed to read" with recv returning EAGAIN while the client's send had
// already completed. 30 s exceeds a worst-case TCP retransmission backoff and
// still reclaims a dead connection.
constexpr time_t kHttpReadStallSeconds = 30;

int64_t now_epoch_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

double elapsed_seconds(std::chrono::steady_clock::time_point started) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
      .count();
}

struct RouteMetric {
  const char* route;
  const httplib::Response* response;
  ~RouteMetric() {
    const int status = response->status > 0 ? response->status : 200;
    obs::Metrics::instance().http(route, std::to_string(status));
  }
};

struct AsrRequestMetric {
  std::chrono::steady_clock::time_point started{std::chrono::steady_clock::now()};
  int64_t start_epoch_ms = now_epoch_ms();
  std::string request_id;
  std::string status;
  bool counted = false;

  ~AsrRequestMetric() {
    if (!counted) return;
    const double seconds = elapsed_seconds(started);
    obs::Metrics::instance().request("transcription", status, seconds);
    obs::TraceRecord trace;
    trace.request_id = request_id;
    trace.stage = "asr";
    trace.request_type = "transcription";
    trace.start_epoch_ms = start_epoch_ms;
    trace.end_epoch_ms = now_epoch_ms();
    trace.status = status;
    trace.duration_ms = seconds * 1000.0;
    trace.asr_ms = seconds * 1000.0;
    obs::Metrics::instance().trace(trace);
  }
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

bool valid_request_id_chars(std::string_view id) {
  if (id.empty() || id.size() > 160) return false;
  for (unsigned char ch : id) {
    if (ch < 0x21 || ch > 0x7e) return false;
  }
  return true;
}

Value json_to_value(const json& j) {
  if (j.is_null()) return Value();
  if (j.is_boolean()) return Value(j.get<bool>());
  if (j.is_number_unsigned()) {
    const uint64_t value = j.get<uint64_t>();
    if (value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      throw ProtocolError("unsigned integer exceeds signed int64");
    }
    return Value(value);
  }
  if (j.is_number_integer()) return Value(j.get<int64_t>());
  if (j.is_number()) return Value(j.get<double>());
  if (j.is_string()) return Value(j.get<std::string>());
  if (j.is_array()) {
    Value::Array out;
    out.reserve(j.size());
    for (const auto& item : j) out.push_back(json_to_value(item));
    return Value(std::move(out));
  }
  Value::Map out;
  for (auto it = j.begin(); it != j.end(); ++it) {
    out[it.key()] = json_to_value(it.value());
  }
  return Value(std::move(out));
}

json value_to_json(const Value& v) {
  switch (v.type()) {
    case Value::Type::Nil:
      return nullptr;
    case Value::Type::Bool:
      return v.as_bool();
    case Value::Type::Int:
      return v.as_int();
    case Value::Type::Float:
      return v.as_double();
    case Value::Type::Str:
      return v.as_str();
    case Value::Type::Bin:
      return v.as_bin();
    case Value::Type::Array: {
      json out = json::array();
      for (const auto& item : v.as_array()) out.push_back(value_to_json(item));
      return out;
    }
    case Value::Type::Map: {
      json out = json::object();
      for (const auto& [key, item] : v.as_map()) out[key] = value_to_json(item);
      return out;
    }
    case Value::Type::Tensor_:
      return {{"dtype", v.as_tensor().dtype},
              {"shape", v.as_tensor().shape},
              {"nbytes", v.as_tensor().bytes.size()}};
  }
  return nullptr;
}

std::string result_text(const Value& result) {
  if (result.is_str()) return result.as_str();
  if (const Value* text = result.find("text"); text != nullptr && text->is_str()) {
    return text->as_str();
  }
  if (const Value* text = result.find("content"); text != nullptr && text->is_str()) {
    return text->as_str();
  }
  return value_to_json(result).dump();
}

std::string result_finish_reason(const Value& result) {
  const Value* reason = result.find("finish_reason");
  if (reason == nullptr || !reason->is_str()) return {};
  return reason->as_str();
}

// Token counts the stage reported in its terminal result. Every field is copied,
// never estimated: a stage that cannot count a leg leaves the key out and the
// key stays absent on the wire. total_tokens is derived only when both halves
// are present, so a partial sum is never shown as a total. Images and audio
// have no comparable token count and contribute no field.
json result_usage(const Value& result) {
  json usage = json::object();
  const auto read = [&result](const char* key) -> std::optional<int64_t> {
    const Value* value = result.find(key);
    if (value == nullptr || !value->is_int() || value->as_int() < 0) {
      return std::nullopt;
    }
    return value->as_int();
  };
  const auto prompt = read("prompt_tokens");
  const auto completion = read("generated_token_count");
  if (prompt) usage["prompt_tokens"] = *prompt;
  if (completion) usage["completion_tokens"] = *completion;
  if (prompt && completion) usage["total_tokens"] = *prompt + *completion;
  return usage.empty() ? json(nullptr) : usage;
}

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

json unified_event_json(const UnifiedEvent& event) {
  json out = {{"kind", to_string(event.kind)},
              {"stage", event.stage_id},
              {"seq", event.seq},
              {"session_id", event.session_id},
              {"is_final", event.is_final},
              {"text", event.text}};
  if (!event.error.empty()) out["error"] = event.error;
  if (event.kind == UnifiedEvent::Kind::kArtifact) {
    out["artifact"] = value_to_json(event.artifact);
  }
  if (event.legs.type() == Value::Type::Array) {
    out["legs"] = value_to_json(event.legs);
  }
  return out;
}

bool unified_input_parts(const json& body, std::string& pcm, bool& has_audio,
                         std::string& error) {
  size_t audio_count = 0;
  size_t image_count = 0;
  size_t text_count = 0;
  has_audio = false;
  try {
    for (const auto& message : body["messages"]) {
      const json& content = message["content"];
      if (content.is_string()) {
        if (content.get<std::string>().empty()) {
          error = "message content string must be nonempty";
          return false;
        }
        if (++text_count > 1) {
          error = "unified chat accepts at most one text part";
          return false;
        }
        continue;
      }
      if (!content.is_array()) {
        error = "unified chat content must be a text string or content part array";
        return false;
      }
      for (const auto& part : content) {
        if (!part.is_object() || !part.contains("type") ||
            !part["type"].is_string()) {
          error = "unified chat supports input_audio, image_url and text parts only";
          return false;
        }
        const std::string type = part["type"].get<std::string>();
        if (type == "image_url") {
          if (!part.contains("image_url") || !part["image_url"].is_object() ||
              !part["image_url"].contains("url") ||
              !part["image_url"]["url"].is_string() ||
              part["image_url"]["url"].get<std::string>().empty()) {
            error = "image_url part requires a nonempty image_url.url";
            return false;
          }
          if (++image_count > 1) {
            error = "unified chat accepts at most one image_url part";
            return false;
          }
          continue;
        }
        if (type == "text") {
          if (!part.contains("text") || !part["text"].is_string() ||
              part["text"].get<std::string>().empty()) {
            error = "text part requires a nonempty text string";
            return false;
          }
          if (++text_count > 1) {
            error = "unified chat accepts at most one text part";
            return false;
          }
          continue;
        }
        if (type != "input_audio") {
          error = "unified chat supports input_audio, image_url and text parts only";
          return false;
        }
        ++audio_count;
        if (!part.contains("input_audio") || !part["input_audio"].is_object() ||
            !part["input_audio"].contains("data") ||
            !part["input_audio"]["data"].is_string() ||
            !part["input_audio"].contains("format") ||
            !part["input_audio"]["format"].is_string()) {
          error = "input_audio part requires input_audio.data and format";
          return false;
        }
        if (part["input_audio"]["format"] != "wav") {
          error = "unified ASR accepts WAV input_audio only";
          return false;
        }
        const std::string wav = util::base64_decode(
            part["input_audio"]["data"].get<std::string>(), 32u << 20);
        pcm = client::pcm16_from_wav(wav);
      }
    }
  } catch (const std::exception& exc) {
    error = exc.what();
    return false;
  }
  if (audio_count > 1) {
    error = "unified chat accepts at most one input_audio part";
    return false;
  }
  if (audio_count == 1) {
    if (pcm.empty()) {
      error = "unified chat input_audio part must be nonempty";
      return false;
    }
    has_audio = true;
  } else if (text_count == 0 && image_count == 0) {
    error = "unified chat without input_audio requires text or image content";
    return false;
  }
  return true;
}

bool valid_mixed_output(const json& body, bool image_enabled, bool tts_enabled,
                        Value::Map& engine_inputs, std::string& error) {
  bool image_output = false;
  bool audio_output = false;
  if (body.contains("modalities") && !body["modalities"].is_null()) {
    const json& modalities = body["modalities"];
    if (!modalities.is_array() || modalities.empty()) {
      error = "modalities must be a non-empty array";
      return false;
    }
    for (const auto& token : modalities) {
      if (!token.is_string()) {
        error = "modalities values must be strings";
        return false;
      }
      const std::string name = token.get<std::string>();
      if (name == "text") continue;
      if (name == "image") image_output = true;
      else if (name == "audio") audio_output = true;
      else {
        error = "modalities values must be text, image or audio";
        return false;
      }
    }
  }
  if (body.contains("image") && !body["image"].is_null()) {
    const json& image = body["image"];
    if (!image.is_object()) {
      error = "image must be an object";
      return false;
    }
    if (image.contains("prompt") && !image["prompt"].is_null()) {
      if (!image["prompt"].is_string() ||
          image["prompt"].get_ref<const std::string&>().empty()) {
        error = "image.prompt must be a non-empty string";
        return false;
      }
    }
    if (image.contains("size") && !image["size"].is_null()) {
      if (!image["size"].is_string() || image["size"].get<std::string>().empty()) {
        error = "image.size must be a non-empty string";
        return false;
      }
    }
    if (image.contains("seed") && !image["seed"].is_null() &&
        !image["seed"].is_number_integer()) {
      error = "image.seed must be an integer";
      return false;
    }
    if (image.contains("output_format") && !image["output_format"].is_null()) {
      if (!image["output_format"].is_string() ||
          (image["output_format"].get<std::string>() != "png" &&
           image["output_format"].get<std::string>() != "jpeg")) {
        error = "image.output_format must be png or jpeg";
        return false;
      }
    }
    image_output = true;
  }
  if (body.contains("audio") && !body["audio"].is_null()) {
    const json& audio = body["audio"];
    if (!audio.is_object() || !audio.contains("voice") ||
        !audio["voice"].is_string() ||
        audio["voice"].get_ref<const std::string&>().empty()) {
      error = "audio requires a non-empty voice";
      return false;
    }
    if (!audio.contains("format") || !audio["format"].is_string() ||
        audio["format"].get<std::string>() != "wav") {
      error = "audio.format must be wav: it is the only documented format";
      return false;
    }
    if (audio.contains("sample_rate") && !audio["sample_rate"].is_null()) {
      if (!audio["sample_rate"].is_number_integer() ||
          audio["sample_rate"].get<int64_t>() != 24000) {
        error = "audio.sample_rate must be 24000";
        return false;
      }
    }
    audio_output = true;
  }
  if (!image_output && !audio_output) return true;
  if (!body.value("stream", false)) {
    error = "image and audio output require stream=true";
    return false;
  }
  if (image_output && !image_enabled) {
    error = "image output is not enabled on this server";
    return false;
  }
  if (audio_output && !tts_enabled) {
    error = "audio output is not enabled on this server";
    return false;
  }
  Value::Array selected;
  if (image_output) selected.push_back(Value("image"));
  if (audio_output) selected.push_back(Value("audio"));
  engine_inputs["output_modalities"] = Value(std::move(selected));
  if (image_output) {
    Value::Map options;
    const json empty_image = json::object();
    const json& image = body.contains("image") && body["image"].is_object()
                            ? body["image"]
                            : empty_image;
    if (image.contains("prompt") && !image["prompt"].is_null()) {
      options["prompt"] = Value(image["prompt"].get<std::string>());
    }
    options["size"] = Value(image.value("size", std::string("512x512")));
    if (image.contains("seed") && !image["seed"].is_null()) {
      options["seed"] = Value(image["seed"].get<int64_t>());
    }
    options["output_format"] =
        Value(image.value("output_format", std::string("png")));
    engine_inputs["image_options"] = Value(std::move(options));
  }
  if (audio_output) {
    Value::Map options;
    const json& audio = body["audio"];
    if (audio.contains("stream") && !audio["stream"].is_null() &&
        !audio["stream"].is_boolean()) {
      error = "audio.stream must be a boolean";
      return false;
    }
    options["voice"] = Value(audio["voice"].get<std::string>());
    options["format"] = Value("wav");
    options["sample_rate"] = Value(int64_t{24000});
    if (audio.value("stream", false)) options["stream"] = Value(true);
    engine_inputs["audio_options"] = Value(std::move(options));
  }
  return true;
}

void reject_overloaded(httplib::Response& res) {
  res.status = 503;
  res.set_header("Retry-After", "1");
  res.set_content(
      json{{"error", {{"message", "server is at capacity"}, {"type", "overloaded"}}}}
          .dump(),
      "application/json");
}

}  // namespace

class OpenAIApi::Impl {
 public:
  Impl(Coordinator& coordinator, ApiConfig config)
      : coordinator_(coordinator),
        config_(std::move(config)),
        proxy_(config_) {
    if (config_.served_model_name.empty()) {
      config_.served_model_name = config_.model;
    }
    server_.set_payload_max_length(kMaxHttpPayloadBytes);
    server_.set_read_timeout(kHttpReadStallSeconds, 0);
    // Transport faults were invisible in the log: a connection dropped before
    // a response left no trace, so a client-seen EOF could not be attributed.
    // One line names the httplib error and the route; never the body.
    server_.set_error_logger([](const httplib::Error& error,
                                const httplib::Request* request) {
      std::cerr << "http error=" << httplib::to_string(error);
      if (request != nullptr) {
        std::cerr << " method=" << request->method << " path=" << request->path
                  << " body_bytes=" << request->body.size();
      }
      std::cerr << "\n";
    });
  }

  void install_routes() {
    server_.set_pre_routing_handler(
        [this](const httplib::Request& req, httplib::Response& res) {
          if (config_.api_key.empty()) {
            return httplib::Server::HandlerResponse::Unhandled;
          }
          const std::string expected = "Bearer " + config_.api_key;
          const auto auth = req.get_header_value("Authorization");
          if (util::constant_time_equals(auth, expected)) {
            return httplib::Server::HandlerResponse::Unhandled;
          }
          res.status = 401;
          res.set_content(
              R"({"error":{"message":"invalid api key","type":"auth_error"}})",
              "application/json");
          return httplib::Server::HandlerResponse::Handled;
        });

    server_.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
      const size_t in_flight =
          unified_engine_
              ? unified_engine_->in_flight()
              : (!config_.upstream_kind.empty()
                     ? active_http_requests_.load(std::memory_order_relaxed)
                     : coordinator_.in_flight());
      json body = {{"status", "ok"}, {"in_flight", in_flight}};
      res.set_content(body.dump(), "application/json");
    });

    server_.Get("/metrics", [](const httplib::Request&, httplib::Response& res) {
      res.set_content(obs::Metrics::instance().render(),
                      "text/plain; version=0.0.4; charset=utf-8");
    });

    server_.Get("/ready", [this](const httplib::Request&, httplib::Response& res) {
      const bool entry_ready = !config_.external_entry_stage ||
                               coordinator_.stage_ready(config_.entry_stage);
      const bool ready = entry_ready &&
                         (!unified_engine_ ||
                          coordinator_.stage_ready(unified_vlm_stage_));
      res.status = ready ? 200 : 503;
      res.set_content(ready ? R"({"status":"ready"})" : R"({"status":"starting"})",
                      "application/json");
    });

    server_.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
      json body = {{"object", "list"},
                   {"data", {{{"id", config_.served_model_name},
                              {"object", "model"},
                              {"created", 0},
                              {"owned_by", "ummInfer"}}}}};
      res.set_content(body.dump(), "application/json");
    });

    server_.Post("/v1/chat/completions",
                 [this](const httplib::Request& req, httplib::Response& res) {
                   if (config_.upstream_kind == "qwen") {
                     AdmissionGuard guard(active_http_requests_,
                                          config_.max_in_flight);
                     if (!guard.admitted()) {
                       reject_overloaded(res);
                       return;
                     }
                     proxy_.proxy_qwen_chat(req, res);
                     return;
                   }
                   if (!config_.upstream_kind.empty()) {
                     proxy_.reject_route(res, "chat completions");
                     return;
                   }
                   handle_chat(req, res);
                 });

    server_.Post("/v1/audio/transcriptions",
                 [this](const httplib::Request& req, httplib::Response& res) {
                   if (!config_.upstream_kind.empty()) {
                     proxy_.reject_route(res, "audio transcriptions");
                     return;
                   }
                   handle_transcription(req, res);
                 });

    server_.Post("/v1/images/generations",
                 [this](const httplib::Request& req, httplib::Response& res) {
                   if (config_.upstream_variant != "image") {
                     proxy_.reject_route(res, "image generation");
                     return;
                   }
                   AdmissionGuard guard(active_http_requests_,
                                        config_.max_in_flight);
                   if (!guard.admitted()) {
                     reject_overloaded(res);
                     return;
                   }
                   proxy_.handle_image_generation(req, res);
                 });

    server_.Post("/v1/images/edits",
                 [this](const httplib::Request& req, httplib::Response& res) {
                   if (config_.upstream_variant != "image") {
                     proxy_.reject_route(res, "image editing");
                     return;
                   }
                   AdmissionGuard guard(active_http_requests_,
                                        config_.max_in_flight);
                   if (!guard.admitted()) {
                     reject_overloaded(res);
                     return;
                   }
                   proxy_.handle_image_edit(req, res);
                 });

    server_.Post("/v1/videos",
                 [this](const httplib::Request& req, httplib::Response& res) {
                   if (config_.upstream_variant != "video") {
                     proxy_.reject_route(res, "video creation");
                     return;
                   }
                   AdmissionGuard guard(active_http_requests_,
                                        config_.max_in_flight);
                   if (!guard.admitted()) {
                     reject_overloaded(res);
                     return;
                   }
                   proxy_.handle_video_create(req, res);
                 });

    server_.Get(R"(/v1/videos/([A-Za-z0-9_\-]{1,128})/content)",
                [this](const httplib::Request& req, httplib::Response& res) {
                  if (config_.upstream_variant != "video") {
                    proxy_.reject_route(res, "video content");
                    return;
                  }
                  AdmissionGuard guard(active_http_requests_,
                                       config_.max_in_flight);
                  if (!guard.admitted()) {
                    reject_overloaded(res);
                    return;
                  }
                  proxy_.handle_video_content(req, req.matches[1].str(), res);
                });

    server_.Get(R"(/v1/videos/([A-Za-z0-9_\-]{1,128}))",
                [this](const httplib::Request& req, httplib::Response& res) {
                  if (config_.upstream_variant != "video") {
                    proxy_.reject_route(res, "video status");
                    return;
                  }
                  AdmissionGuard guard(active_http_requests_,
                                       config_.max_in_flight);
                  if (!guard.admitted()) {
                    reject_overloaded(res);
                    return;
                  }
                  proxy_.handle_video_status(req, req.matches[1].str(), res);
                });

    server_.Delete(R"(/v1/videos/([A-Za-z0-9_\-]{1,128}))",
                   [this](const httplib::Request& req, httplib::Response& res) {
                     if (config_.upstream_variant != "video") {
                       proxy_.reject_route(res, "video deletion");
                       return;
                     }
                     AdmissionGuard guard(active_http_requests_,
                                          config_.max_in_flight);
                     if (!guard.admitted()) {
                       reject_overloaded(res);
                       return;
                     }
                     proxy_.handle_video_delete(req, req.matches[1].str(), res);
                   });

    server_.Post("/abort", [this](const httplib::Request& req, httplib::Response& res) {
      if (!config_.upstream_kind.empty()) {
        proxy_.reject(
            res, 501,
            "abort is not supported: requests are served by a fixed upstream",
            "unsupported_route");
        return;
      }
      const json body = json::parse(req.body, nullptr, false);
      if (body.is_discarded() || !body.is_object() ||
          !body.contains("request_id") || !body["request_id"].is_string() ||
          !valid_request_id_chars(body["request_id"].get_ref<const std::string&>())) {
        res.status = 400;
        res.set_content(R"({"error":{"message":"request_id is required"}})",
                        "application/json");
        return;
      }
      std::string session_id;
      if (body.contains("session_id") && !body["session_id"].is_null()) {
        if (!body["session_id"].is_string() ||
            body["session_id"].get_ref<const std::string&>().empty() ||
            body["session_id"].get_ref<const std::string&>().size() > 128) {
          res.status = 400;
          res.set_content(
              R"({"error":{"message":"session_id must be a nonempty string of at most 128 characters"}})",
              "application/json");
          return;
        }
        session_id = body["session_id"].get<std::string>();
      }
      const std::string request_id = body["request_id"].get<std::string>();
      if (unified_engine_) {
        const auto status =
            unified_engine_->cancel_for_session(request_id, session_id);
        if (status == UnifiedEngine::CancelStatus::kSessionMismatch) {
          res.status = 403;
          res.set_content(
              R"({"error":{"message":"request_id does not belong to session_id","type":"permission_error"}})",
              "application/json");
          return;
        }
        if (status == UnifiedEngine::CancelStatus::kNotFoundOrTerminal &&
            session_id.empty()) {
          coordinator_.abort(request_id);
        }
      } else {
        coordinator_.abort(request_id);
      }
      res.set_content(R"({"success":true})", "application/json");
    });

    server_.Post("/v1/audio/speech",
                 [](const httplib::Request&, httplib::Response& res) {
                   res.status = 501;
                   res.set_content(
                       R"({"error":{"message":"speech is not implemented by this server"}})",
                       "application/json");
                 });

    // Opt-in static hosting for the bundled browser client. cpp-httplib's mount
    // rejects a ".." segment in the URL and any path that resolves outside the
    // directory (a symlink escape), so no request reads past web_root.
    if (!config_.web_root.empty() &&
        !server_.set_mount_point("/", config_.web_root)) {
      std::fprintf(stderr, "web root is not a readable directory: %s\n",
                   config_.web_root.c_str());
    }
  }

  bool bind() {
    if (bound_) return true;
    const std::string config_error = validate_upstream_config(config_);
    if (!config_error.empty()) {
      std::fprintf(stderr, "invalid upstream configuration: %s\n",
                   config_error.c_str());
      return false;
    }
    if (config_.port == 0) {
      bound_port_ = server_.bind_to_any_port(config_.host);
      if (bound_port_ < 0) return false;
    } else {
      if (!server_.bind_to_port(config_.host, config_.port)) return false;
      bound_port_ = config_.port;
    }
    bound_ = true;
    return true;
  }
  bool listen() {
    if (!bound_ && !bind()) return false;
    return server_.listen_after_bind();
  }
  void stop() { server_.stop(); }

  int bound_port() const { return bound_port_; }

  void set_unified_forwarder(std::function<void(const UnifiedEvent&)> forwarder) {
    unified_forwarder_ = std::move(forwarder);
  }

  void set_unified_engine(UnifiedEngine& engine, std::string vlm_stage) {
    unified_engine_ = &engine;
    unified_vlm_stage_ = std::move(vlm_stage);
    engine.on_event = [this](const UnifiedEvent& event) { on_unified_event(event); };
  }

 private:
  std::string submit_chat(StagePayload payload) {
    if (unified_engine_) return unified_engine_->submit(std::move(payload));
    return coordinator_.submit(std::move(payload), config_.entry_stage);
  }
  bool wait_chat(const std::string& request_id, double timeout_s) {
    return unified_engine_ ? unified_engine_->wait(request_id, timeout_s)
                           : coordinator_.wait(request_id, timeout_s);
  }
  std::optional<RequestInfo> info_chat(const std::string& request_id) {
    return unified_engine_ ? unified_engine_->info(request_id)
                           : coordinator_.info(request_id);
  }
  void cancel_chat(const std::string& request_id, const char* source = "client") {
    if (unified_engine_) unified_engine_->cancel(request_id, source);
    else coordinator_.abort(request_id, source);
  }
  void release_chat(const std::string& request_id) {
    if (unified_engine_) unified_engine_->release(request_id);
    else coordinator_.release(request_id);
  }

  void handle_transcription(const httplib::Request& req, httplib::Response& res) {
    RouteMetric route_metric{"transcriptions", &res};
    AsrRequestMetric request_metric;
    // The handler starts after httplib has read the whole body, so this clock
    // separates upload time (client total minus this) from engine time.
    const auto handler_started = std::chrono::steady_clock::now();
    auto reject = [&res](int status, const std::string& message) {
      res.status = status;
      res.set_content(json{{"error", {{"message", util::utf8_sanitized(message)},
                                      {"type", "transcription_error"}}}}
                          .dump(),
                      "application/json");
    };
    if (!req.is_multipart_form_data()) {
      reject(400, "one multipart WAV file is required");
      return;
    }
    for (const auto& part : req.form.files) {
      if (part.first != "file") {
        reject(400, "unsupported multipart file part: " + part.first);
        return;
      }
    }
    if (req.form.get_file_count("file") != 1) {
      reject(400, "one multipart WAV file is required");
      return;
    }
    std::map<std::string, int> field_counts;
    for (const auto& part : req.form.fields) ++field_counts[part.first];
    for (const auto& [name, count] : field_counts) {
      if (count > 1) {
        reject(400, "duplicate multipart field: " + name);
        return;
      }
    }
    static const std::set<std::string> kAllowedFields = {
        "model", "stream", "response_format", "include_generation_metadata",
        "language", "prompt", "max_new_tokens", "temperature"};
    for (const auto& [name, count] : field_counts) {
      if (!kAllowedFields.count(name)) {
        reject(400, "unsupported multipart field: " + name);
        return;
      }
    }
    const std::string requested_model = req.form.get_field("model");
    if (requested_model.empty()) {
      reject(400, "model is required");
      return;
    }
    if (requested_model != config_.served_model_name) {
      reject(400, "unsupported model: " + requested_model +
                      "; this server serves " + config_.served_model_name);
      return;
    }
    if (req.form.has_field("language")) {
      reject(400, "language is not supported by this server");
      return;
    }
    if (req.form.has_field("prompt")) {
      reject(400, "prompt is not supported by this server");
      return;
    }
    if (req.form.has_field("temperature") && req.form.get_field("temperature") != "0") {
      reject(400, "temperature is fixed at 0 by this server");
      return;
    }
    bool stream = false;
    if (req.form.has_field("stream")) {
      const std::string value = req.form.get_field("stream");
      if (value != "true" && value != "false") {
        reject(400, "stream must be true or false");
        return;
      }
      stream = value == "true";
    }
    const std::string response_format = req.form.has_field("response_format")
                                            ? req.form.get_field("response_format")
                                            : std::string("json");
    if (response_format != "json") {
      reject(400, "unsupported response_format: " + response_format +
                      "; only json is supported, for both streaming and non-streaming");
      return;
    }
    bool want_metadata = false;
    if (req.form.has_field("include_generation_metadata")) {
      const std::string value = req.form.get_field("include_generation_metadata");
      if (value != "true" && value != "false") {
        reject(400, "include_generation_metadata must be true or false");
        return;
      }
      want_metadata = value == "true";
    }
    int64_t max_tokens = 128;
    if (req.form.has_field("max_new_tokens")) {
      const std::string text = req.form.get_field("max_new_tokens");
      const auto parsed =
          std::from_chars(text.data(), text.data() + text.size(), max_tokens);
      if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
          max_tokens < 1 || max_tokens > 4096) {
        reject(400, "max_new_tokens must be between 1 and 4096");
        return;
      }
    }
    std::string pcm;
    try {
      pcm = client::pcm16_from_wav(req.form.get_file("file").content);
    } catch (const std::exception& error) {
      reject(400, error.what());
      return;
    }
    StagePayload payload;
    try {
      payload = serve_detail::make_asr_payload(std::move(pcm), max_tokens, "", "");
    } catch (const std::exception& error) {
      reject(400, error.what());
      return;
    }
    AdmissionGuard guard(active_http_requests_, config_.max_in_flight);
    if (!guard.admitted()) {
      reject_overloaded(res);
      return;
    }
    std::string request_id;
    try {
      request_id = coordinator_.submit(std::move(payload), config_.entry_stage);
    } catch (const std::exception& error) {
      reject(503, error.what());
      return;
    }
    // Submit is synchronous only in the sense that it enqueues; the split
    // between this and the wait below is what tells a slow engine from a slow
    // client-to-server transfer.
    const auto submit_done = std::chrono::steady_clock::now();
    obs::Metrics::instance().trace_start(request_id, "transcription", "asr",
                                         request_metric.start_epoch_ms);
    request_metric.request_id = request_id;
    request_metric.counted = true;
    request_metric.status = "aborted";
    bool settled = false;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration<double>(config_.request_timeout_s);
    while (!settled && !req.is_connection_closed()) {
      const auto remaining = deadline - std::chrono::steady_clock::now();
      if (remaining <= std::chrono::steady_clock::duration::zero()) break;
      settled = coordinator_.wait(
          request_id,
          std::min(0.1, std::chrono::duration<double>(remaining).count()));
    }
    if (req.is_connection_closed()) {
      std::cerr << "transcription client_closed request_id=" << request_id
                << " body_bytes=" << req.body.size() << " submit_ms="
                << static_cast<long>(
                       std::chrono::duration<double, std::milli>(
                           submit_done - handler_started).count())
                << " handler_ms="
                << static_cast<long>(elapsed_seconds(handler_started) * 1000.0)
                << " settled=" << (settled ? 1 : 0) << "\n";
      coordinator_.abort(request_id, "client");
      coordinator_.release(request_id);
      // An abandoned request must not leave res.status unset: httplib fills a
      // missing status with 200, so a client that is still reading would see a
      // successful transcription with an empty body.
      reject(503, "client closed the connection before the transcription finished");
      return;
    }
    auto info = coordinator_.info(request_id);
    if (!settled || !info) {
      std::cerr << "transcription timeout request_id=" << request_id
                << " handler_ms="
                << static_cast<long>(elapsed_seconds(handler_started) * 1000.0)
                << "\n";
      request_metric.status = "timeout";
      coordinator_.abort(request_id, "timeout");
      coordinator_.release(request_id);
      reject(504, "transcription timed out");
      return;
    }
    coordinator_.release(request_id);
    request_metric.status = info->state;
    std::cerr << "transcription done request_id=" << request_id
              << " body_bytes=" << req.body.size() << " handler_ms="
              << static_cast<long>(elapsed_seconds(handler_started) * 1000.0)
              << " state=" << info->state << "\n";
    if (info->state != "completed") {
      reject(500, info->error);
      return;
    }
    std::string text;
    try {
      text = serve_detail::asr_result_text(info->result);
    } catch (const std::exception& error) {
      reject(500, error.what());
      return;
    }
    serve_detail::AsrGenerationMetadata metadata;
    const bool has_metadata =
        serve_detail::asr_generation_metadata(info->result, metadata);
    if (want_metadata && !has_metadata) {
      reject(500, "the ASR stage did not report generation metadata");
      return;
    }
    json out = {{"text", text}};
    if (has_metadata) {
      out["generation_metadata"] = {
          {"generated_token_count", metadata.generated_token_count},
          {"finish_reason", metadata.finish_reason}};
    }
    if (!stream) {
      res.set_content(out.dump(), "application/json");
      return;
    }
    json event = {{"type", "transcript.text.done"}, {"text", text}};
    if (has_metadata) event["generation_metadata"] = out["generation_metadata"];
    res.set_content("data: " + event.dump() + "\n\ndata: [DONE]\n\n",
                    "text/event-stream");
  }

  void handle_chat(const httplib::Request& req, httplib::Response& res) {
    RouteMetric route_metric{"chat", &res};
    const json body = json::parse(req.body, nullptr, false);
    auto reject = [&res](const std::string& message) {
      res.status = 400;
      res.set_content(
          json{{"error", {{"message", message}, {"type", "invalid_request"}}}}
              .dump(),
          "application/json");
    };
    if (body.is_discarded() || !body.is_object() || !body.contains("messages") ||
        !body["messages"].is_array() || body["messages"].empty()) {
      reject("messages must be a non-empty array");
      return;
    }
    for (const auto& message : body["messages"]) {
      if (!message.is_object() || !message.contains("role") ||
          !message["role"].is_string() || !message.contains("content")) {
        reject("each message requires a string role and content");
        return;
      }
      if (!message["content"].is_string() && !message["content"].is_array()) {
        reject("message content must be a text string or content part array");
        return;
      }
    }
    if (body.contains("stream") && !body["stream"].is_boolean()) {
      reject("stream must be a boolean");
      return;
    }
    const bool stream = body.value("stream", false);
    const int64_t created = static_cast<int64_t>(std::time(nullptr));
    if (body.contains("model") && !body["model"].is_null()) {
      if (!body["model"].is_string()) {
        reject("model must be a string");
        return;
      }
      if (body["model"].get<std::string>() != config_.served_model_name) {
        reject("unsupported model: " + body["model"].get<std::string>() +
               "; this server serves " + config_.served_model_name);
        return;
      }
    }
    if (body.contains("top_p") && !body["top_p"].is_null()) {
      reject("top_p is not implemented by this server");
      return;
    }
    if (body.contains("seed") && !body["seed"].is_null()) {
      reject("seed is not implemented by this server");
      return;
    }
    if (body.contains("stop") && !body["stop"].is_null()) {
      const json& stop = body["stop"];
      const bool empty = (stop.is_string() && stop.get<std::string>().empty()) ||
                         (stop.is_array() && stop.empty());
      if (!empty) {
        reject("stop is not implemented by this server");
        return;
      }
    }
    if (body.contains("response_format") && !body["response_format"].is_null()) {
      const json& format = body["response_format"];
      if (!format.is_object() || !format.contains("type") ||
          !format["type"].is_string() ||
          format["type"].get<std::string>() != "text") {
        reject("only response_format type text is supported");
        return;
      }
    }
    static const std::set<std::string> kAllowedFields = {
        "model", "messages", "stream", "temperature", "max_tokens",
        "max_completion_tokens", "top_p", "seed", "stop", "response_format",
        "stream_options", "session_id", "session_history", "modalities",
        "image", "audio"};
    for (auto it = body.begin(); it != body.end(); ++it) {
      if (!kAllowedFields.count(it.key()) && !it.value().is_null()) {
        reject("unsupported field: " + it.key());
        return;
      }
    }
    if (body.contains("session_id") && !body["session_id"].is_null() &&
        !unified_engine_) {
      reject("session_id requires unified mode");
      return;
    }
    if (body.contains("session_history") && !body["session_history"].is_null()) {
      if (!body["session_history"].is_boolean()) {
        reject("session_history must be a boolean");
        return;
      }
      if (!unified_engine_) {
        reject("session_history requires unified mode");
        return;
      }
      if (body["session_history"].get<bool>() &&
          !unified_engine_->history_enabled()) {
        reject("session_history is not enabled on this server");
        return;
      }
    }
    if (body.contains("stream_options") && !body["stream_options"].is_null()) {
      const json& options = body["stream_options"];
      if (!stream) {
        reject("stream_options requires stream=true");
        return;
      }
      if (!options.is_object()) {
        reject("stream_options must be an object");
        return;
      }
      for (auto it = options.begin(); it != options.end(); ++it) {
        if (it.key() != "include_usage" &&
            it.key() != "include_asr_partials") {
          reject("unsupported stream_options field: " + it.key());
          return;
        }
      }
      if (options.contains("include_usage") &&
          !options["include_usage"].is_boolean()) {
        reject("stream_options.include_usage must be a boolean");
        return;
      }
      if (options.value("include_usage", false)) {
        reject("stream_options.include_usage is not implemented by this server");
        return;
      }
      if (options.contains("include_asr_partials") &&
          !options["include_asr_partials"].is_boolean()) {
        reject("stream_options.include_asr_partials must be a boolean");
        return;
      }
      if (options.value("include_asr_partials", false) && !unified_engine_) {
        reject("stream_options.include_asr_partials requires unified mode");
        return;
      }
    }
    if (body.contains("temperature") && !body["temperature"].is_null()) {
      if (!body["temperature"].is_number()) {
        reject("temperature must be a number");
        return;
      }
      const double temperature = body["temperature"].get<double>();
      if (temperature < 0 || temperature > 2) {
        reject("temperature must be between 0 and 2");
        return;
      }
    }
    const json* token_limit = nullptr;
    for (const char* key : {"max_tokens", "max_completion_tokens"}) {
      if (!body.contains(key) || body[key].is_null()) continue;
      const json& value = body[key];
      if (!value.is_number_integer() ||
          (value.is_number_unsigned() &&
           value.get<uint64_t>() >
               static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) ||
          value.get<int64_t>() < 1) {
        reject(std::string(key) + " must be a positive integer");
        return;
      }
      token_limit = &value;
    }
    StagePayload payload;
    Value::Map inputs;
    Value::Map mixed_engine_inputs;
    if (unified_engine_) {
      std::string mixed_error;
      if (!valid_mixed_output(
              body, unified_engine_->image_output_enabled(),
              unified_engine_->tts_output_enabled(), mixed_engine_inputs,
              mixed_error)) {
        reject(mixed_error);
        return;
      }
    } else if (body.contains("modalities") || body.contains("image") ||
               body.contains("audio")) {
      reject("mixed output requires unified mode");
      return;
    }
    try {
      inputs["messages"] = json_to_value(body["messages"]);
      if (unified_engine_) {
        std::string pcm;
        bool has_audio = false;
        std::string error;
        if (!unified_input_parts(body, pcm, has_audio, error)) {
          reject(error);
          return;
        }
        if (has_audio) inputs["audio"] = Value::bin(std::move(pcm));
        if (body.contains("stream_options") &&
            body["stream_options"].value("include_asr_partials", false)) {
          inputs["asr_partials"] = Value(true);
        }
        if (body.contains("session_id") && !body["session_id"].is_null()) {
          if (!body["session_id"].is_string() ||
              body["session_id"].get<std::string>().empty() ||
              body["session_id"].get<std::string>().size() > 128) {
            reject("session_id must be a nonempty string of at most 128 characters");
            return;
          }
          inputs["session_id"] = Value(body["session_id"].get<std::string>());
        }
        if (body.contains("session_history") &&
            !body["session_history"].is_null()) {
          if (body["session_history"].get<bool>() &&
              !(body.contains("session_id") &&
                body["session_id"].is_string() &&
                !body["session_id"].get<std::string>().empty())) {
            reject("session_history requires a nonempty session_id");
            return;
          }
          inputs["session_history"] =
              Value(body["session_history"].get<bool>());
        }
      }
    } catch (const std::exception& exc) {
      reject(exc.what());
      return;
    }
    for (auto& [key, value] : mixed_engine_inputs) {
      inputs[key] = std::move(value);
    }
    payload.request.inputs = Value(std::move(inputs));
    Value::Map params;
    if (token_limit) params["max_new_tokens"] = Value(token_limit->get<int64_t>());
    if (body.contains("temperature") && !body["temperature"].is_null()) {
      params["temperature"] = Value(body["temperature"].get<double>());
    }
    params["stream"] = Value(stream);
    payload.request.params = std::move(params);

    std::shared_ptr<AdmissionGuard> non_unified_guard;
    if (!unified_engine_) {
      non_unified_guard = std::make_shared<AdmissionGuard>(
          active_http_requests_, config_.max_in_flight);
      if (!non_unified_guard->admitted()) {
        reject_overloaded(res);
        return;
      }
    }

    if (!stream) {
      std::string request_id;
      try {
        request_id = submit_chat(std::move(payload));
      } catch (const QueueFullError&) {
        reject_overloaded(res);
        return;
      } catch (const std::exception& exc) {
        res.status = 503;
        res.set_content(
            json{{"error", {{"message", exc.what()}, {"type", "pipeline_error"}}}}
                .dump(),
            "application/json");
        return;
      }
      bool settled = false;
      const auto deadline = std::chrono::steady_clock::now() +
                            std::chrono::duration<double>(config_.request_timeout_s);
      while (!settled && !req.is_connection_closed()) {
        const auto remaining = deadline - std::chrono::steady_clock::now();
        if (remaining <= std::chrono::steady_clock::duration::zero()) break;
        const double wait_s = std::min(
            0.1, std::chrono::duration<double>(remaining).count());
        settled = wait_chat(request_id, wait_s);
      }
      if (req.is_connection_closed()) {
        cancel_chat(request_id);
        release_chat(request_id);
        return;
      }
      auto info = info_chat(request_id);
      if (!settled || !info) {
        cancel_chat(request_id, "timeout");
        release_chat(request_id);
        res.status = 504;
        res.set_content(json{{"error", {{"message", "request timed out"}}}}.dump(),
                        "application/json");
        return;
      }
      release_chat(request_id);
      if (info->state != "completed") {
        res.status = 500;
        res.set_content(
            json{{"error", {{"message", info->error}, {"type", "generation_error"}}}}
                .dump(),
            "application/json");
        return;
      }
      const std::string finish = result_finish_reason(info->result);
      json out = {{"id", request_id},
                  {"object", "chat.completion"},
                  {"created", created},
                  {"model", config_.served_model_name},
                  {"choices",
                   {{{"index", 0},
                     {"message", {{"role", "assistant"},
                                  {"content", result_text(info->result)}}},
                     {"finish_reason", finish.empty() ? json(nullptr) : json(finish)}}}}};
      res.set_content(out.dump(), "application/json");
      return;
    }

    const std::string request_id = "http-" + std::to_string(++next_request_id_);
    payload.request_id = request_id;
    auto sink = std::make_shared<SseSink>();
    {
      std::lock_guard<std::mutex> lock(streams_mu_);
      streams_[request_id] = sink;
    }
    try {
      submit_chat(std::move(payload));
    } catch (const QueueFullError&) {
      {
        std::lock_guard<std::mutex> lock(streams_mu_);
        streams_.erase(request_id);
      }
      reject_overloaded(res);
      return;
    } catch (const std::exception& exc) {
      {
        std::lock_guard<std::mutex> lock(streams_mu_);
        streams_.erase(request_id);
      }
      res.status = 503;
      res.set_content(
          json{{"error", {{"message", exc.what()}, {"type", "pipeline_error"}}}}
              .dump(),
          "application/json");
      return;
    }
    auto cleanup = std::shared_ptr<void>(
        nullptr, [this, request_id, non_unified_guard](void*) {
          cancel_chat(request_id);
          {
            std::lock_guard<std::mutex> lock(streams_mu_);
            streams_.erase(request_id);
          }
          release_chat(request_id);
        });
    const std::string model = config_.served_model_name;
    res.set_chunked_content_provider(
        "text/event-stream",
        [this, sink, request_id, model, created, cleanup](size_t,
                                                          httplib::DataSink& out_sink) {
          auto send_chunk = [&](const json& payload) {
            const std::string data = "data: " + payload.dump() + "\n\n";
            return out_sink.write(data.data(), data.size());
          };
          if (!send_chunk({{"id", request_id},
                           {"object", "chat.completion.chunk"},
                           {"created", created},
                           {"model", model},
                           {"choices", {{{"index", 0},
                                        {"delta", {{"role", "assistant"}}},
                                        {"finish_reason", nullptr}}}}})) {
            cancel_chat(request_id);
            return false;
          }
          std::string stream_error;
          std::string finish;
          std::optional<UnifiedEvent> terminal;
          std::optional<Value> sink_result;
          bool settled = false;
          bool disconnected = false;
          const auto timeout = std::chrono::duration<double>(config_.request_timeout_s);
          auto deadline = std::chrono::steady_clock::now() + timeout;
          while (true) {
            if (!out_sink.is_writable()) {
              disconnected = true;
              break;
            }
            const double remaining =
                std::chrono::duration<double>(
                    deadline - std::chrono::steady_clock::now())
                    .count();
            if (remaining <= 0) {
              stream_error = "stream timed out";
              break;
            }
            SseItem item;
            if (sink->pop(item, std::min(0.2, remaining))) {
              json chunk_body = {{"id", request_id},
                                 {"object", "chat.completion.chunk"},
                                 {"created", created},
                                 {"model", model}};
              if (const auto* chunk = std::get_if<Value>(&item)) {
                chunk_body["choices"] = {
                    {{"index", 0},
                     {"delta", {{"content", result_text(*chunk)}}},
                     {"finish_reason", nullptr}}};
              } else {
                const auto& event = std::get<UnifiedEvent>(item);
                json delta = json::object();
                if (event.kind == UnifiedEvent::Kind::kVlmDelta) {
                  delta["content"] = event.text;
                }
                chunk_body["choices"] = {{{"index", 0},
                                          {"delta", std::move(delta)},
                                          {"finish_reason", nullptr}}};
                chunk_body["unified"] = unified_event_json(event);
                if (event.kind == UnifiedEvent::Kind::kArtifact) {
                  json& artifact = chunk_body["unified"]["artifact"];
                  const std::string& raw = *event.artifact_bytes;
                  const std::string_view slice(
                      raw.data() + event.artifact_offset,
                      event.artifact_length);
                  artifact["encoding"] = "base64";
                  artifact["data"] = util::base64_encode(slice);
                  const int64_t index = artifact.value("index", int64_t{-1});
                  const int64_t total = artifact.value("total", int64_t{-1});
                  if (index + 1 == total) {
                    artifact["bytes"] = static_cast<int64_t>(raw.size());
                    artifact["sha256"] = util::sha256_hex(raw);
                  }
                }
              }
              const std::string frame_payload = chunk_body.dump();
              if (frame_payload.size() + 8 > kMaxSseFrameWireBytes) {
                stream_error = "artifact SSE frame exceeds its size limit";
                break;
              }
              if (!send_chunk(chunk_body)) {
                disconnected = true;
                break;
              }
              deadline = std::chrono::steady_clock::now() + timeout;
              continue;
            }
            {
              std::lock_guard<std::mutex> lock(sink->mu);
              settled = sink->done;
              if (settled) {
                stream_error = sink->error;
                finish = sink->finish_reason;
                terminal = sink->terminal;
                sink_result = sink->result;
              }
            }
            if (settled) break;
          }
          if (!settled) cancel_chat(request_id);
          if (!disconnected) {
            if (stream_error.empty()) {
              json final_chunk = {
                  {"id", request_id},
                  {"object", "chat.completion.chunk"},
                  {"created", created},
                  {"model", model},
                  {"choices",
                   {{{"index", 0},
                     {"delta", json::object()},
                     {"finish_reason",
                      finish.empty() ? json(nullptr) : json(finish)}}}}};
              if (terminal) final_chunk["unified"] = unified_event_json(*terminal);
              // The usage carrier rides on the last chunk, before [DONE]. A
              // client that does not know the key ignores it.
              const Value* usage_source =
                  terminal ? &terminal->result
                           : (sink_result ? &*sink_result : nullptr);
              if (usage_source != nullptr) {
                json usage = result_usage(*usage_source);
                if (!usage.is_null()) final_chunk["usage"] = std::move(usage);
              }
              send_chunk(final_chunk);
            } else {
              json error_chunk = {{"error", {{"message", stream_error},
                                             {"type", "generation_error"},
                                             {"code", 500}}}};
              if (terminal) error_chunk["unified"] = unified_event_json(*terminal);
              send_chunk(error_chunk);
            }
            static const std::string kDone = "data: [DONE]\n\n";
            out_sink.write(kDone.data(), kDone.size());
            out_sink.done();
          }
          return !disconnected;
        });
  }

 public:
  void on_unified_event(const UnifiedEvent& event) {
    if (unified_forwarder_) {
      try {
        unified_forwarder_(event);
      } catch (...) {
      }
    }
    std::shared_ptr<SseSink> sink;
    {
      std::lock_guard<std::mutex> lock(streams_mu_);
      auto it = streams_.find(event.request_id);
      if (it != streams_.end()) sink = it->second;
      const bool terminal = event.kind == UnifiedEvent::Kind::kComplete ||
                            event.kind == UnifiedEvent::Kind::kError ||
                            event.kind == UnifiedEvent::Kind::kCancelled;
      if (terminal && it != streams_.end()) streams_.erase(it);
    }
    if (!sink) return;
    if (event.kind == UnifiedEvent::Kind::kAsrPartial ||
        event.kind == UnifiedEvent::Kind::kAsrFinal ||
        event.kind == UnifiedEvent::Kind::kVlmDelta ||
        event.kind == UnifiedEvent::Kind::kArtifact ||
        event.kind == UnifiedEvent::Kind::kLegError) {
      if (event.kind == UnifiedEvent::Kind::kVlmDelta) {
        std::lock_guard<std::mutex> lock(sink->mu);
        sink->has_deltas = true;
      }
      sink->push(event);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(sink->mu);
      sink->terminal = event;
    }
    if (event.kind == UnifiedEvent::Kind::kComplete) {
      bool has_deltas;
      {
        std::lock_guard<std::mutex> lock(sink->mu);
        has_deltas = sink->has_deltas;
        sink->finish_reason = result_finish_reason(event.result);
      }
      if (!has_deltas) sink->push(event.result);
      sink->finish();
      return;
    }
    if (event.kind == UnifiedEvent::Kind::kCancelled) {
      sink->finish();
      return;
    }
    sink->finish(event.error.empty()
                     ? std::string("request ") + to_string(event.kind)
                     : event.error);
  }

  void on_stream(const StreamMessage& msg) {
    if (unified_engine_ && unified_engine_->info(msg.request_id)) return;
    std::shared_ptr<SseSink> sink;
    {
      std::lock_guard<std::mutex> lock(streams_mu_);
      auto it = streams_.find(msg.request_id);
      if (it != streams_.end()) sink = it->second;
    }
    if (sink) sink->push(msg.chunk, true);
  }

  void on_complete(const RequestInfo& info) {
    if (unified_engine_ && unified_engine_->info(info.request_id)) return;
    std::shared_ptr<SseSink> sink;
    {
      std::lock_guard<std::mutex> lock(streams_mu_);
      auto it = streams_.find(info.request_id);
      if (it != streams_.end()) {
        sink = it->second;
        streams_.erase(it);
      }
    }
    if (!sink) return;
    if (info.state == "completed") {
      bool has_deltas;
      {
        std::lock_guard<std::mutex> lock(sink->mu);
        has_deltas = sink->has_deltas;
        sink->finish_reason = result_finish_reason(info.result);
        sink->result = info.result;
      }
      if (!has_deltas) sink->push(info.result);
      sink->finish();
      return;
    }
    sink->finish(info.error.empty() ? "request " + info.state : info.error);
  }

 private:
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
  std::map<std::string, std::shared_ptr<SseSink>> streams_;
  std::atomic<uint64_t> next_request_id_{0};
};

OpenAIApi::OpenAIApi(Coordinator& coordinator, ApiConfig config)
    : impl_(std::make_unique<Impl>(coordinator, std::move(config))) {
  impl_->install_routes();
  coordinator.on_stream = [this](const StreamMessage& msg) { impl_->on_stream(msg); };
  coordinator.on_complete = [this](const RequestInfo& info) { impl_->on_complete(info); };
}

OpenAIApi::~OpenAIApi() = default;

void OpenAIApi::set_unified_forwarder(
    std::function<void(const UnifiedEvent&)> forwarder) {
  impl_->set_unified_forwarder(std::move(forwarder));
}

void OpenAIApi::set_unified_engine(UnifiedEngine& engine, std::string vlm_stage) {
  impl_->set_unified_engine(engine, std::move(vlm_stage));
}

bool OpenAIApi::bind() { return impl_->bind(); }
bool OpenAIApi::listen() { return impl_->listen(); }
void OpenAIApi::stop() { impl_->stop(); }
int OpenAIApi::bound_port() const { return impl_->bound_port(); }

}  // namespace umminfer
