// SPDX-License-Identifier: Apache-2.0
#include "http.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>

namespace umminfer::client {
namespace {

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

constexpr size_t kMaxJsonResponseBytes = 64u << 20;
constexpr size_t kMaxImageBytes = 64u << 20;
constexpr size_t kMaxImageTotalBytes = 256u << 20;
constexpr size_t kMaxAudioBytes = 128u << 20;
constexpr size_t kMaxVideoBytes = 512u << 20;
// Mixed-output chat mirrors the server's frozen decoded-byte contract.
constexpr size_t kMixedMaxImageBytes = 48u << 20;
constexpr size_t kMixedMaxAudioBytes = 48u << 20;
constexpr size_t kMixedMaxAggregateBytes = 60u << 20;
// The largest legal mixed stream on the wire: 60 MiB decoded is ~80 MiB of
// base64; this leaves envelope headroom. Text-only streams keep the 64 MiB
// transport cap, so their behavior is unchanged.
constexpr size_t kMixedStreamWireBytes = 96u << 20;
constexpr size_t kMixedChunkDecodedBytes = (2u << 20) + 16;

bool blank(std::string_view text) {
  return std::all_of(text.begin(), text.end(), [](unsigned char c) { return std::isspace(c); });
}

std::string lower(std::string_view text) {
  std::string out(text);
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return out;
}

[[noreturn]] void bad_response(std::string_view raw, std::string_view secret) {
  throw Error(ErrorCode::kBadResponse,
              "invalid JSON response: " + detail::clip(detail::redact(raw, secret), 200));
}

json response_json(std::string_view raw, std::string_view secret) {
  json value = json::parse(raw, nullptr, false);
  if (value.is_discarded() || !value.is_object()) bad_response(raw, secret);
  if (value.contains("error") && !value["error"].is_null()) {
    detail::throw_http(500, raw, {}, secret);
  }
  return value;
}

std::string response_string(const json& object, const char* name, std::string_view raw,
                            std::string_view secret) {
  const auto it = object.find(name);
  if (it == object.end() || it->is_null()) return {};
  if (!it->is_string()) bad_response(raw, secret);
  return it->get<std::string>();
}

Usage response_usage(const json& root, std::string_view raw, std::string_view secret) {
  const auto it = root.find("usage");
  if (it == root.end() || it->is_null()) return {};
  if (!it->is_object()) bad_response(raw, secret);
  Usage usage;
  auto read = [&](const char* name) {
    const auto field = it->find(name);
    if (field == it->end() || field->is_null()) return int64_t{0};
    if (!field->is_number_integer() ||
        (field->is_number_unsigned() &&
         field->get<uint64_t>() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))) {
      bad_response(raw, secret);
    }
    return field->get<int64_t>();
  };
  usage.prompt_tokens = read("prompt_tokens");
  usage.completion_tokens = read("completion_tokens");
  usage.total_tokens = read("total_tokens");
  if (usage.total_tokens == 0) {
    if ((usage.completion_tokens > 0 &&
         usage.prompt_tokens > std::numeric_limits<int64_t>::max() - usage.completion_tokens) ||
        (usage.completion_tokens < 0 &&
         usage.prompt_tokens < std::numeric_limits<int64_t>::min() - usage.completion_tokens)) {
      bad_response(raw, secret);
    }
    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
  }
  return usage;
}

std::string normalized_finish(std::string_view reason) {
  if (reason == "stop" || reason == "end_turn" || reason == "stop_sequence" || reason == "eos") {
    return "stop";
  }
  if (reason == "length" || reason == "max_tokens") return "length";
  if (reason == "content_filter" || reason == "refusal" || reason == "safety") {
    return "content_filter";
  }
  if (reason.empty()) return {};
  return "other";
}

// The same identifier class the serve route accepts. A rejected id fails here
// instead of turning into a 404 or an escaped path.
bool valid_media_id(std::string_view id) {
  if (id.empty() || id.size() > 128) return false;
  for (const char c : id) {
    const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                    (c >= '0' && c <= '9') || c == '_' || c == '-';
    if (!ok) return false;
  }
  return true;
}

std::string checked_media_id(std::string_view id) {
  if (!valid_media_id(id)) throw Error(ErrorCode::kConfig, "invalid video id");
  return std::string(id);
}

json content_part_json(const ContentPart& part) {
  switch (part.kind) {
    case ContentPart::Kind::kText:
      return {{"type", "text"}, {"text", part.text}};
    case ContentPart::Kind::kImageUrl:
      return {{"type", "image_url"}, {"image_url", {{"url", part.url}}}};
    case ContentPart::Kind::kInputAudio:
      return {{"type", "input_audio"},
              {"input_audio", {{"data", part.audio}, {"format", part.audio_format}}}};
    case ContentPart::Kind::kVideo:
      return {{"type", "video"}, {"video", part.video_urls}};
    case ContentPart::Kind::kVideoUrl: {
      json out = {{"type", "video_url"}, {"video_url", {{"url", part.url}}}};
      if (part.fps) out["fps"] = *part.fps;
      return out;
    }
  }
  throw Error(ErrorCode::kConfig, "unsupported content part kind");
}

void validate_message_parts(const Message& message, size_t index) {
  const std::string where = "message " + std::to_string(index);
  if (!message.parts.empty() && !message.content.empty()) {
    throw Error(ErrorCode::kConfig, where + " has both content and parts");
  }
  for (const auto& part : message.parts) {
    switch (part.kind) {
      case ContentPart::Kind::kText:
        break;
      case ContentPart::Kind::kImageUrl:
        if (part.url.empty()) {
          throw Error(ErrorCode::kConfig, where + " has an image_url part without a url");
        }
        break;
      case ContentPart::Kind::kInputAudio:
        if (part.audio.empty() || part.audio_format.empty()) {
          throw Error(ErrorCode::kConfig,
                      where + " has an input_audio part without data or format");
        }
        break;
      case ContentPart::Kind::kVideo:
        if (part.video_urls.empty()) {
          throw Error(ErrorCode::kConfig, where + " has an empty video part");
        }
        for (const auto& url : part.video_urls) {
          if (url.empty()) throw Error(ErrorCode::kConfig, where + " has an empty video url");
        }
        break;
      case ContentPart::Kind::kVideoUrl:
        if (part.url.empty()) {
          throw Error(ErrorCode::kConfig, where + " has a video_url part without a url");
        }
        if (part.fps && (!std::isfinite(*part.fps) || *part.fps < 0.1 || *part.fps > 10)) {
          throw Error(ErrorCode::kConfig, where + " video fps must be between 0.1 and 10");
        }
        break;
    }
  }
}

// Appends the decoded audio of one response object. Stream chunks carry a
// complete base64 value each, so each is decoded on its own.
void read_audio_field(const json& object, std::string& audio, std::string& transcript,
                      std::string_view raw, std::string_view secret) {
  const auto it = object.find("audio");
  if (it == object.end() || it->is_null()) return;
  if (!it->is_object()) bad_response(raw, secret);
  const std::string data = response_string(*it, "data", raw, secret);
  const std::string transcript_delta = response_string(*it, "transcript", raw, secret);
  if (transcript.size() + transcript_delta.size() > kMaxJsonResponseBytes) {
    throw Error(ErrorCode::kBadResponse, "transcript exceeds its size limit");
  }
  transcript += transcript_delta;
  if (data.empty()) return;
  const size_t remaining =
      kMaxAudioBytes - std::min(kMaxAudioBytes, audio.size());
  if (remaining == 0) throw Error(ErrorCode::kBadResponse, "audio response exceeds its size limit");
  audio += detail::base64_decode(data, remaining);
}

detail::HttpRequest media_request(const OpenAIConfig& config, const std::string& base_url,
                                  const std::string& path, detail::HttpMethod method,
                                  Milliseconds timeout, CancelFlag cancel) {
  detail::HttpRequest http;
  http.url = base_url + path;
  http.method = method;
  http.bearer = config.api_key;
  http.timeout = timeout.count() > 0 ? timeout : config.timeout;
  http.connect_timeout = config.connect_timeout;
  http.cancel = cancel;
  http.no_proxy = lower(base_url).rfind("http://", 0) == 0;
  http.body_limit = kMaxJsonResponseBytes;
  return http;
}

std::string encoded_body(const json& body) {
  try {
    return body.dump();
  } catch (const json::exception& error) {
    throw Error(ErrorCode::kBadRequest,
                std::string("cannot encode request: ") + error.what());
  }
}

void reserve_media_fields(const json& body, std::initializer_list<const char*> names,
                          const char* container) {
  for (const char* name : names) {
    if (body.contains(name) && !body[name].is_null()) {
      throw Error(ErrorCode::kConfig,
                  std::string(container) + "." + name + " is reserved");
    }
  }
}

ImageResponse parse_image_response(const detail::HttpResponse& response,
                                   std::string_view secret) {
  const json root = response_json(response.body, secret);
  const auto data = root.find("data");
  if (data == root.end() || !data->is_array() || data->empty()) {
    throw Error(ErrorCode::kBadResponse, "response has no images");
  }
  ImageResponse out;
  size_t remaining = kMaxImageTotalBytes;
  for (const json& item : *data) {
    if (!item.is_object()) bad_response(response.body, secret);
    const std::string encoded = response_string(item, "b64_json", response.body, secret);
    if (encoded.empty()) {
      const auto url = item.find("url");
      if (url != item.end() && !url->is_null()) {
        throw Error(ErrorCode::kBadResponse,
                    "url image responses are not supported; request b64_json");
      }
      throw Error(ErrorCode::kBadResponse, "image response has no b64_json");
    }
    if (remaining == 0) {
      throw Error(ErrorCode::kBadResponse, "image response exceeds its size limit");
    }
    std::string bytes =
        detail::base64_decode(encoded, std::min(kMaxImageBytes, remaining));
    remaining -= bytes.size();
    out.images.push_back(std::move(bytes));
  }
  return out;
}

VideoJob parse_video_job(const detail::HttpResponse& response, std::string_view secret) {
  const json root = response_json(response.body, secret);
  VideoJob job;
  job.raw = root;
  job.id = response_string(root, "id", response.body, secret);
  job.status = response_string(root, "status", response.body, secret);
  if (job.id.empty()) throw Error(ErrorCode::kBadResponse, "video response has no id");
  return job;
}

}  // namespace

OpenAIClient::OpenAIClient(OpenAIConfig config) : config_(std::move(config)) {
  base_url_ = detail::validate_base_url(config_.base_url, false);
  endpoint_ = base_url_ + "/chat/completions";
  const auto first = std::find_if_not(config_.api_key.begin(), config_.api_key.end(),
                                      [](unsigned char c) { return std::isspace(c); });
  const auto last = std::find_if_not(config_.api_key.rbegin(), config_.api_key.rend(),
                                     [](unsigned char c) { return std::isspace(c); })
                        .base();
  config_.api_key = first < last ? std::string(first, last) : std::string{};
  if (config_.timeout.count() < 0 || config_.connect_timeout.count() < 0 ||
      config_.retry_base.count() < 0 || config_.retry_max.count() < 0 ||
      config_.max_retry_after.count() < 0 || config_.max_attempts < 0) {
    throw Error(ErrorCode::kConfig, "retry and timeout settings must not be negative");
  }
  const OpenAIConfig defaults;
  if (config_.timeout.count() == 0) config_.timeout = defaults.timeout;
  if (config_.connect_timeout.count() == 0) config_.connect_timeout = defaults.connect_timeout;
  if (config_.max_attempts == 0) config_.max_attempts = defaults.max_attempts;
  if (config_.retry_base.count() == 0) config_.retry_base = defaults.retry_base;
  if (config_.retry_max.count() == 0) config_.retry_max = defaults.retry_max;
  if (config_.max_retry_after.count() == 0) config_.max_retry_after = defaults.max_retry_after;
}

ChatResponse OpenAIClient::chat(const ChatRequest& request) const {
  const std::string& model = request.model.empty() ? config_.model : request.model;
  if (blank(model)) throw Error(ErrorCode::kConfig, "model is required");
  if (request.messages.empty()) throw Error(ErrorCode::kConfig, "messages are required");
  for (size_t i = 0; i < request.messages.size(); ++i) {
    if (blank(request.messages[i].role)) {
      throw Error(ErrorCode::kConfig, "message " + std::to_string(i) + " has no role");
    }
    validate_message_parts(request.messages[i], i);
  }
  if (request.max_tokens < 0) throw Error(ErrorCode::kConfig, "max_tokens must not be negative");
  if (request.temperature &&
      (!std::isfinite(*request.temperature) || *request.temperature < 0 || *request.temperature > 2)) {
    throw Error(ErrorCode::kConfig, "temperature must be between 0 and 2");
  }
  if (request.top_p &&
      (!std::isfinite(*request.top_p) || *request.top_p < 0 || *request.top_p > 1)) {
    throw Error(ErrorCode::kConfig, "top_p must be between 0 and 1");
  }
  if (request.stop.size() > 4) throw Error(ErrorCode::kConfig, "at most four stop strings are allowed");
  if (!request.response_format.is_null() && !request.response_format.is_object()) {
    throw Error(ErrorCode::kConfig, "response_format must be an object");
  }
  if (!request.extra_body.is_object()) throw Error(ErrorCode::kConfig, "extra_body must be an object");
  if (request.extra_body.contains("stream")) {
    throw Error(ErrorCode::kConfig,
                "extra_body.stream is reserved; use ChatRequest::stream");
  }
  if (request.timeout.count() < 0 || request.first_token_timeout.count() < 0 ||
      request.stall_timeout.count() < 0) {
    throw Error(ErrorCode::kConfig, "request timeouts must not be negative");
  }
  detail::check_cancelled(request.cancel);

  const bool stream = request.stream || static_cast<bool>(request.on_delta) ||
                      request.first_token_timeout.count() > 0 || request.stall_timeout.count() > 0;
  bool audio_output = false;
  bool image_output = false;
  for (const auto& modality : request.modalities) {
    if (modality != "text" && modality != "audio" && modality != "image") {
      throw Error(ErrorCode::kConfig,
                  "modalities values must be text, image or audio");
    }
    audio_output = audio_output || modality == "audio";
    image_output = image_output || modality == "image";
  }
  if (request.image) {
    if (!image_output) {
      throw Error(ErrorCode::kConfig,
                  "image output requires modalities to include image");
    }
    if (request.image->output_format != "png" &&
        request.image->output_format != "jpeg") {
      throw Error(ErrorCode::kConfig, "image.output_format must be png or jpeg");
    }
    if (request.image->size.empty()) {
      throw Error(ErrorCode::kConfig, "image.size must be non-empty");
    }
    if (request.image->seed && *request.image->seed < 0) {
      throw Error(ErrorCode::kConfig, "image.seed must not be negative");
    }
    if (!stream) {
      throw Error(ErrorCode::kConfig, "image output requires stream");
    }
  }
  if (request.audio) {
    if (!audio_output) {
      throw Error(ErrorCode::kConfig, "audio output requires modalities to include audio");
    }
    if (blank(request.audio->voice)) {
      throw Error(ErrorCode::kConfig, "audio.voice is required");
    }
    if (request.audio->format != "wav") {
      throw Error(ErrorCode::kConfig, "audio.format must be wav");
    }
    if (request.audio->sample_rate != 24000) {
      throw Error(ErrorCode::kConfig, "audio.sample_rate must be 24000");
    }
    if (!stream) {
      throw Error(ErrorCode::kConfig, "audio output requires stream");
    }
  } else if (audio_output) {
    throw Error(ErrorCode::kConfig, "audio output requires audio.voice");
  }
  const bool mixed_output = image_output || audio_output;
  const Milliseconds timeout = request.timeout.count() > 0 ? request.timeout : config_.timeout;
  bool emitted_delta = false;

  // The client does not add stream_options. The server does not report usage,
  // and the caller's own fields are sent unchanged.
  auto build_body = [&]() {
    json body = request.extra_body;
    body["model"] = model;
    body["messages"] = json::array();
    for (const auto& message : request.messages) {
      json content = message.content;
      if (!message.parts.empty()) {
        content = json::array();
        for (const auto& part : message.parts) content.push_back(content_part_json(part));
      }
      body["messages"].push_back({{"role", message.role}, {"content", std::move(content)}});
    }
    if (!request.modalities.empty()) body["modalities"] = request.modalities;
    if (request.audio) {
      body["audio"] = {{"voice", request.audio->voice},
                       {"format", request.audio->format},
                       {"sample_rate", request.audio->sample_rate}};
    }
    if (request.image) {
      json image = {{"size", request.image->size},
                    {"output_format", request.image->output_format}};
      if (!request.image->prompt.empty()) {
        image["prompt"] = request.image->prompt;
      }
      if (request.image->seed) image["seed"] = *request.image->seed;
      body["image"] = std::move(image);
    }
    if (request.temperature) body["temperature"] = *request.temperature;
    if (request.top_p) body["top_p"] = *request.top_p;
    if (request.max_tokens > 0) body["max_tokens"] = request.max_tokens;
    if (!request.stop.empty()) body["stop"] = request.stop;
    if (request.seed) body["seed"] = *request.seed;
    if (!request.response_format.is_null()) body["response_format"] = request.response_format;
    if (stream) {
      body["stream"] = true;
    } else {
      // A caller-supplied stream_options stays in the body so the server can
      // reject it. Silently dropping it would report success for a field the
      // server never saw.
      if (!request.extra_body.contains("stream_options")) body.erase("stream_options");
    }
    return body;
  };

  auto complete = [&](ChatResponse out, int attempt) {
    if (out.model.empty()) out.model = model;
    out.attempts = attempt;
    const bool empty = out.text.empty() && out.audio.empty() &&
                       out.audio_transcript.empty() && out.artifacts.empty();
    if (out.finish_reason == "content_filter" && empty) {
      throw Error(ErrorCode::kContentFilter, "the model refused the request");
    }
    if (empty && out.finish_reason != "length") {
      throw Error(ErrorCode::kEmpty, "the model returned no text");
    }
    return out;
  };

  for (int attempt = 1;; ++attempt) {
    detail::check_cancelled(request.cancel);
    auto attempt_start = Clock::now();
    detail::Progress progress;
    progress.first_token = request.first_token_timeout;
    progress.stall = request.stall_timeout;
    Milliseconds retry_delay{0};

    try {
      for (;;) {
        const auto elapsed = std::chrono::duration_cast<Milliseconds>(Clock::now() - attempt_start);
        if (elapsed >= timeout) throw Error(ErrorCode::kTimeout, "request timeout", true);
        json body = build_body();
        detail::HttpRequest http;
        http.url = endpoint_;
        http.bearer = config_.api_key;
        http.timeout = timeout - elapsed;
        http.connect_timeout = config_.connect_timeout;
        http.cancel = request.cancel;
        // Both paths share one bound, but a stream limit requires a body
        // callback, so each branch sets only its own.
        if (stream) {
          http.stream_limit = mixed_output ? kMixedStreamWireBytes
                                           : kMaxJsonResponseBytes;
        } else {
          http.body_limit = kMaxJsonResponseBytes;
        }
        http.progress = &progress;
        http.no_proxy = lower(endpoint_).rfind("http://", 0) == 0;
        http.headers.emplace("Content-Type", "application/json");
        try {
          http.body = body.dump();
        } catch (const json::exception& error) {
          throw Error(ErrorCode::kBadRequest, std::string("cannot encode request: ") + error.what());
        }

        try {
          if (!stream) {
            const detail::HttpResponse response = detail::perform(http);
            const json root = response_json(response.body, config_.api_key);
            ChatResponse out;
            out.model = response_string(root, "model", response.body, config_.api_key);
            out.usage = response_usage(root, response.body, config_.api_key);
            const auto choices = root.find("choices");
            if (choices == root.end() || !choices->is_array() || choices->empty()) {
              throw Error(ErrorCode::kBadResponse, "response has no choices");
            }
            const json& choice = (*choices)[0];
            if (!choice.is_object()) bad_response(response.body, config_.api_key);
            const auto message = choice.find("message");
            if (message == choice.end() || message->is_null()) {
              throw Error(ErrorCode::kBadResponse, "response has no choices");
            }
            if (!message->is_object()) bad_response(response.body, config_.api_key);
            out.text = response_string(*message, "content", response.body, config_.api_key);
            read_audio_field(*message, out.audio, out.audio_transcript, response.body,
                             config_.api_key);
            const std::string refusal =
                response_string(*message, "refusal", response.body, config_.api_key);
            const std::string reason =
                response_string(choice, "finish_reason", response.body, config_.api_key);
            out.finish_reason = normalized_finish(reason);
            if (out.text.empty() && !refusal.empty()) out.finish_reason = "content_filter";
            return complete(std::move(out), attempt);
          }

          ChatResponse out;
          detail::SSEParser parser;
          std::string refusal;
          // Every aggregated field shares one bound, so a stream cannot grow
          // past the non-stream JSON limit.
          size_t aggregated = 0;
          bool done = false;
          // Mixed-output artifact reassembly. Chunks of one artifact_id must
          // arrive in index order; the final chunk carries bytes and sha256.
          struct IncomingArtifact {
            Artifact artifact;
            std::string bytes;
            int64_t next_index = 0;
            int64_t total = -1;
          };
          std::map<std::string, IncomingArtifact> incoming;
          size_t artifact_decoded_total = 0;
          int64_t last_unified_seq = -1;
          bool seq_started = false;
          auto fail_artifact = [](const std::string& detail) {
            throw Error(ErrorCode::kBadResponse, detail);
          };
          auto event = [&](std::string_view data) {
            if (data == "[DONE]") {
              done = true;
              return false;
            }
            const json root = response_json(data, config_.api_key);
            const std::string reported_model = response_string(root, "model", data, config_.api_key);
            if (!reported_model.empty()) out.model = reported_model;
            if (root.contains("usage") && !root["usage"].is_null()) {
              out.usage = response_usage(root, data, config_.api_key);
            }
            const auto choices = root.find("choices");
            if (choices != root.end()) {
              if (!choices->is_array()) bad_response(data, config_.api_key);
              for (const json& choice : *choices) {
                if (!choice.is_object()) bad_response(data, config_.api_key);
                const auto delta = choice.find("delta");
                if (delta != choice.end() && !delta->is_null()) {
                  if (!delta->is_object()) bad_response(data, config_.api_key);
                  const std::string text = response_string(*delta, "content", data, config_.api_key);
                  if (!text.empty()) {
                    progress.token();
                    aggregated += text.size();
                    out.text += text;
                    if (request.on_delta) {
                      emitted_delta = true;
                      request.on_delta(text);
                    }
                    detail::check_cancelled(request.cancel);
                  }
                  // A refusal and an audio chunk are output too. They reset the
                  // first-token and stall timers like a text delta.
                  const std::string refusal_delta =
                      response_string(*delta, "refusal", data, config_.api_key);
                  if (!refusal_delta.empty()) {
                    progress.token();
                    aggregated += refusal_delta.size();
                    refusal += refusal_delta;
                  }
                  const size_t before = out.audio.size() + out.audio_transcript.size();
                  read_audio_field(*delta, out.audio, out.audio_transcript, data,
                                   config_.api_key);
                  const size_t gained = out.audio.size() + out.audio_transcript.size() - before;
                  if (gained > 0) {
                    progress.token();
                    aggregated += gained;
                  }
                  if (aggregated > kMaxJsonResponseBytes) {
                    throw Error(ErrorCode::kBadResponse,
                                "streamed response exceeds its size limit");
                  }
                }
                const std::string reason =
                    response_string(choice, "finish_reason", data, config_.api_key);
                if (!reason.empty()) out.finish_reason = normalized_finish(reason);
              }
            }
            const auto unified = root.find("unified");
            if (unified != root.end() && !unified->is_null()) {
              if (!unified->is_object()) bad_response(data, config_.api_key);
              const auto kind_it = unified->find("kind");
              if (kind_it == unified->end() || !kind_it->is_string()) {
                bad_response(data, config_.api_key);
              }
              const auto seq_it = unified->find("seq");
              if (seq_it == unified->end() || !seq_it->is_number_integer()) {
                bad_response(data, config_.api_key);
              }
              const int64_t seq = seq_it->get<int64_t>();
              if (seq_started && seq != last_unified_seq + 1) {
                fail_artifact("unified event seq is not gap-free");
              }
              seq_started = true;
              last_unified_seq = seq;
              if (unified->contains("legs") && !(*unified)["legs"].is_null()) {
                out.legs = (*unified)["legs"];
              }
              if (kind_it->get<std::string>() == "artifact") {
                const auto& a = (*unified)["artifact"];
                if (!a.is_object()) bad_response(data, config_.api_key);
                for (const char* key :
                     {"artifact_id", "media_type", "mime_type", "encoding",
                      "data", "index", "total"}) {
                  if (!a.contains(key)) fail_artifact(
                      std::string("artifact missing field: ") + key);
                }
                const std::string id = a["artifact_id"].get<std::string>();
                const std::string media = a["media_type"].get<std::string>();
                const std::string mime = a["mime_type"].get<std::string>();
                if (media != "image" && media != "audio") {
                  fail_artifact("artifact media_type must be image or audio");
                }
                if (a["encoding"].get<std::string>() != "base64") {
                  fail_artifact("artifact encoding must be base64");
                }
                const int64_t index = a["index"].get<int64_t>();
                const int64_t total = a["total"].get<int64_t>();
                if (index < 0 || total < 1 || index >= total) {
                  fail_artifact("artifact chunk index is out of range");
                }
                auto& slot = incoming[id];
                if (slot.artifact.artifact_id.empty()) {
                  slot.artifact.artifact_id = id;
                  slot.artifact.media_type = media;
                  slot.artifact.mime_type = mime;
                } else if (slot.artifact.media_type != media ||
                           slot.artifact.mime_type != mime) {
                  fail_artifact("artifact descriptor changed between chunks");
                }
                if (index != slot.next_index) {
                  fail_artifact("artifact chunks arrived out of order");
                }
                const std::string encoded = a["data"].get<std::string>();
                if (encoded.size() % 4 != 0) {
                  fail_artifact("artifact chunk is not legal base64");
                }
                std::string decoded = detail::base64_decode(
                    encoded, kMixedChunkDecodedBytes);
                const size_t per_leg = media == "image"
                                           ? kMixedMaxImageBytes
                                           : kMixedMaxAudioBytes;
                if (slot.bytes.size() + decoded.size() > per_leg) {
                  throw Error(ErrorCode::kBadResponse,
                              media + " artifact exceeds its size limit");
                }
                if (artifact_decoded_total + decoded.size() >
                    kMixedMaxAggregateBytes) {
                  throw Error(ErrorCode::kBadResponse,
                              "mixed output artifacts exceed the aggregate limit");
                }
                artifact_decoded_total += decoded.size();
                slot.bytes += decoded;
                slot.next_index = index + 1;
                if (slot.total == -1) slot.total = total;
                else if (slot.total != total) {
                  fail_artifact("artifact total changed between chunks");
                }
                aggregated += encoded.size();
                if (aggregated > kMaxJsonResponseBytes && !mixed_output) {
                  throw Error(ErrorCode::kBadResponse,
                              "streamed response exceeds its size limit");
                }
                if (index + 1 == total) {
                  if (!a.contains("sha256") || !a["sha256"].is_string() ||
                      !a.contains("bytes") || !a["bytes"].is_number_integer()) {
                    fail_artifact(
                        "final artifact chunk requires sha256 and bytes");
                  }
                  if (a["bytes"].get<int64_t>() !=
                      static_cast<int64_t>(slot.bytes.size())) {
                    fail_artifact("artifact byte count does not match");
                  }
                  const std::string digest =
                      detail::sha256_hex(slot.bytes);
                  if (digest != a["sha256"].get<std::string>()) {
                    fail_artifact("artifact sha256 mismatch");
                  }
                  slot.artifact.data = std::move(slot.bytes);
                  slot.artifact.sha256 = digest;
                  slot.artifact.descriptor = a;
                  out.artifacts.push_back(std::move(slot.artifact));
                  incoming.erase(id);
                }
              }
            }
            return true;
          };
          http.on_body = [&](std::string_view bytes) { return parser.feed(bytes, event); };
          detail::perform(http);
          if (!parser.stopped()) parser.finish(event);
          detail::check_cancelled(request.cancel);
          // [DONE] is the terminator. A finish_reason alone is not: the stream
          // can still be cut after it, so a missing terminator is a truncation.
          if (!done) {
            throw Error(ErrorCode::kNetwork, "stream ended before [DONE]", true);
          }
          if (!incoming.empty()) {
            throw Error(ErrorCode::kBadResponse,
                        "an artifact stream ended before its final chunk");
          }
          if (out.text.empty() && !refusal.empty()) out.finish_reason = "content_filter";
          return complete(std::move(out), attempt);
        } catch (const Error& error) {
          // No field is dropped and retried. An unsupported field is reported
          // to the caller as an error.
          throw;
        }
      }
    } catch (const Error& error) {
      if (!error.retryable() || emitted_delta || attempt >= config_.max_attempts ||
          error.retry_after() > config_.max_retry_after) {
        throw;
      }
      if (error.retry_after().count() > 0) {
        retry_delay = error.retry_after();
      } else {
        int64_t cap = config_.retry_base.count();
        for (int retry = 1; retry < attempt && cap < config_.retry_max.count(); ++retry) {
          cap = cap > config_.retry_max.count() / 2 ? config_.retry_max.count() : cap * 2;
        }
        cap = std::min<int64_t>(cap, config_.retry_max.count());
        const int64_t floor = cap / 2;
        thread_local std::mt19937_64 generator(std::random_device{}());
        const double jitter = std::uniform_real_distribution<double>(0.0, 1.0)(generator);
        retry_delay = Milliseconds{floor + static_cast<int64_t>(jitter * static_cast<double>(floor))};
      }
    }

    // The server's Retry-After wait is never shortened by this attempt's
    // timeout. The next attempt starts with a fresh attempt timeout.
    const auto sleep_until = Clock::now() + retry_delay;
    while (Clock::now() < sleep_until) {
      detail::check_cancelled(request.cancel);
      const auto remaining = std::chrono::duration_cast<Milliseconds>(sleep_until - Clock::now());
      std::this_thread::sleep_for(std::min(remaining, Milliseconds{50}));
    }
    detail::check_cancelled(request.cancel);
  }
}

ImageResponse OpenAIClient::generate_image(const ImageRequest& request) const {
  const std::string& model = request.model.empty() ? config_.model : request.model;
  if (blank(model)) throw Error(ErrorCode::kConfig, "model is required");
  if (request.prompt.empty()) throw Error(ErrorCode::kConfig, "prompt is required");
  if (request.n < 0) throw Error(ErrorCode::kConfig, "n must not be negative");
  if (!request.extra_body.is_object()) {
    throw Error(ErrorCode::kConfig, "extra_body must be an object");
  }
  reserve_media_fields(request.extra_body,
                       {"model", "prompt", "response_format", "stream"}, "extra_body");
  detail::check_cancelled(request.cancel);

  json body = request.extra_body;
  body["model"] = model;
  body["prompt"] = request.prompt;
  // Inline bytes only. A url response would point at upstream storage.
  body["response_format"] = "b64_json";
  if (request.n > 0) body["n"] = request.n;
  if (!request.size.empty()) body["size"] = request.size;

  auto http = media_request(config_, base_url_, "/images/generations",
                            detail::HttpMethod::kPost, request.timeout, request.cancel);
  http.headers.emplace("Content-Type", "application/json");
  http.accept = "application/json";
  http.body = encoded_body(body);
  return parse_image_response(detail::perform(http), config_.api_key);
}

ImageResponse OpenAIClient::edit_image(const ImageEditRequest& request) const {
  const std::string& model = request.model.empty() ? config_.model : request.model;
  if (blank(model)) throw Error(ErrorCode::kConfig, "model is required");
  if (request.prompt.empty()) throw Error(ErrorCode::kConfig, "prompt is required");
  if (request.images.empty()) throw Error(ErrorCode::kConfig, "at least one image is required");
  for (const auto& image : request.images) {
    if (image.data.empty()) throw Error(ErrorCode::kConfig, "an image part is empty");
  }
  std::vector<std::string> seen;
  for (const auto& [name, value] : request.fields) {
    if (name.empty()) throw Error(ErrorCode::kConfig, "a multipart field has no name");
    if (name == "model" || name == "prompt" || name == "image" ||
        name == "response_format") {
      throw Error(ErrorCode::kConfig, "multipart field is reserved: " + name);
    }
    if (name != "url") {
      for (const auto& previous : seen) {
        if (previous == name) {
          throw Error(ErrorCode::kConfig, "duplicate multipart field: " + name);
        }
      }
      seen.push_back(name);
    }
  }
  detail::check_cancelled(request.cancel);

  auto http = media_request(config_, base_url_, "/images/edits",
                            detail::HttpMethod::kPost, request.timeout, request.cancel);
  http.accept = "application/json";
  // Image order defines the prompt placeholders, so it is preserved exactly.
  for (const auto& image : request.images) {
    detail::MimePart part;
    part.name = "image";
    part.data = image.data;
    part.filename = image.filename.empty() ? "image" : image.filename;
    part.content_type =
        image.content_type.empty() ? "application/octet-stream" : image.content_type;
    http.mime_parts.push_back(std::move(part));
  }
  detail::MimePart prompt_part;
  prompt_part.name = "prompt";
  prompt_part.data = request.prompt;
  http.mime_parts.push_back(std::move(prompt_part));
  for (const auto& [name, value] : request.fields) {
    detail::MimePart part;
    part.name = name;
    part.data = value;
    http.mime_parts.push_back(std::move(part));
  }
  detail::MimePart model_part;
  model_part.name = "model";
  model_part.data = model;
  http.mime_parts.push_back(std::move(model_part));
  // The parser accepts inline bytes only, so the request forces b64_json
  // instead of inheriting an upstream default of url.
  detail::MimePart format_part;
  format_part.name = "response_format";
  format_part.data = "b64_json";
  http.mime_parts.push_back(std::move(format_part));
  return parse_image_response(detail::perform(http), config_.api_key);
}

VideoJob OpenAIClient::create_video(const VideoRequest& request) const {
  const std::string& model = request.model.empty() ? config_.model : request.model;
  if (blank(model)) throw Error(ErrorCode::kConfig, "model is required");
  if (request.prompt.empty()) throw Error(ErrorCode::kConfig, "prompt is required");
  if (!request.body.is_object()) throw Error(ErrorCode::kConfig, "body must be an object");
  reserve_media_fields(request.body, {"model", "prompt"}, "body");
  detail::check_cancelled(request.cancel);

  json body = request.body;
  body["model"] = model;
  body["prompt"] = request.prompt;
  auto http = media_request(config_, base_url_, "/videos", detail::HttpMethod::kPost,
                            request.timeout, request.cancel);
  http.headers.emplace("Content-Type", "application/json");
  http.accept = "application/json";
  http.body = encoded_body(body);
  // Sent once. A retry would submit a second job.
  return parse_video_job(detail::perform(http), config_.api_key);
}

VideoJob OpenAIClient::video_status(std::string_view video_id, CancelFlag cancel) const {
  const std::string id = checked_media_id(video_id);
  detail::check_cancelled(cancel);
  auto http = media_request(config_, base_url_, "/videos/" + id,
                            detail::HttpMethod::kGet, Milliseconds{0}, cancel);
  http.accept = "application/json";
  return parse_video_job(detail::perform(http), config_.api_key);
}

nlohmann::json OpenAIClient::delete_video(std::string_view video_id,
                                          CancelFlag cancel) const {
  const std::string id = checked_media_id(video_id);
  detail::check_cancelled(cancel);
  auto http = media_request(config_, base_url_, "/videos/" + id,
                            detail::HttpMethod::kDelete, Milliseconds{0}, cancel);
  http.accept = "application/json";
  const detail::HttpResponse response = detail::perform(http);
  return response_json(response.body, config_.api_key);
}

void OpenAIClient::download_video(std::string_view video_id,
                                  const std::string& output_path,
                                  std::optional<int> variant,
                                  CancelFlag cancel) const {
  const std::string id = checked_media_id(video_id);
  if (output_path.empty()) throw Error(ErrorCode::kConfig, "output path is required");
  if (variant && *variant < 0) {
    throw Error(ErrorCode::kConfig, "variant must not be negative");
  }
  std::error_code status_error;
  if (std::filesystem::exists(output_path, status_error)) {
    throw Error(ErrorCode::kBadRequest, "output file already exists: " + output_path);
  }
  if (status_error) {
    throw Error(ErrorCode::kBadRequest,
                "cannot inspect output path: " + status_error.message());
  }
  detail::check_cancelled(cancel);

  std::unique_ptr<detail::PartWriter> writer;
  try {
    writer = std::make_unique<detail::PartWriter>(output_path);
  } catch (const std::exception& error) {
    throw Error(ErrorCode::kBadRequest,
                std::string("cannot create output: ") + error.what());
  }
  std::string content_path = "/videos/" + id + "/content";
  if (variant) content_path += "?variant=" + std::to_string(*variant);
  auto http = media_request(config_, base_url_, content_path,
                            detail::HttpMethod::kGet, Milliseconds{0}, cancel);
  http.accept = "video/mp4";
  http.require_content_type = "video/mp4";
  http.stream_limit = kMaxVideoBytes;
  size_t received = 0;
  http.on_body = [&writer, &received](std::string_view bytes) {
    try {
      writer->write(bytes);
    } catch (const std::exception& error) {
      throw Error(ErrorCode::kBadRequest,
                  std::string("cannot write output: ") + error.what());
    }
    received += bytes.size();
    return true;
  };
  detail::perform(http);
  // A 200 with an empty body is not a video. The part file is removed by the
  // writer destructor and the final path never appears.
  if (received == 0) {
    throw Error(ErrorCode::kBadResponse, "video response is empty");
  }
  try {
    writer->publish();
  } catch (const std::exception& error) {
    throw Error(ErrorCode::kBadRequest,
                std::string("cannot publish output: ") + error.what());
  }
}

}  // namespace umminfer::client
