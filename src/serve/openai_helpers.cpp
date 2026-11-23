// SPDX-License-Identifier: Apache-2.0
#include "openai_internal.hpp"

#include <limits>
#include <utility>

#include "umminfer/client/client.hpp"
#include "umminfer/util/codec.hpp"

namespace umminfer::openai_detail {

int64_t now_epoch_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

double elapsed_seconds(std::chrono::steady_clock::time_point started) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
      .count();
}

RouteMetric::~RouteMetric() {
  const int status = response->status > 0 ? response->status : 200;
  obs::Metrics::instance().http(route, std::to_string(status));
}

AsrRequestMetric::~AsrRequestMetric() {
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
                        bool video_enabled, Value::Map& engine_inputs,
                        std::string& error) {
  bool image_output = false;
  bool audio_output = false;
  bool video_output = false;
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
      else if (name == "video") video_output = true;
      else {
        error = "modalities values must be text, image, audio or video";
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
      // The unified image leg is the native image stage, and it emits PNG only.
      // Accepting jpeg here would promise bytes the stage cannot produce.
      if (!image["output_format"].is_string() ||
          image["output_format"].get<std::string>() != "png") {
        error = "image.output_format must be png: the image stage emits PNG only";
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
  if (body.contains("video") && !body["video"].is_null()) {
    const json& video = body["video"];
    if (!video.is_object()) {
      error = "video must be an object";
      return false;
    }
    if (video.contains("prompt") && !video["prompt"].is_null()) {
      if (!video["prompt"].is_string() ||
          video["prompt"].get_ref<const std::string&>().empty()) {
        error = "video.prompt must be a non-empty string";
        return false;
      }
    }
    if (video.contains("size") && !video["size"].is_null()) {
      if (!video["size"].is_string() || video["size"].get<std::string>().empty()) {
        error = "video.size must be a non-empty string";
        return false;
      }
    }
    if (video.contains("seconds") && !video["seconds"].is_null()) {
      if (!video["seconds"].is_number_integer() ||
          video["seconds"].get<int64_t>() < 1) {
        error = "video.seconds must be a positive integer";
        return false;
      }
    }
    if (video.contains("fps") && !video["fps"].is_null()) {
      if (!video["fps"].is_number_integer() ||
          video["fps"].get<int64_t>() < 1) {
        error = "video.fps must be a positive integer";
        return false;
      }
    }
    if (video.contains("seed") && !video["seed"].is_null() &&
        !video["seed"].is_number_integer()) {
      error = "video.seed must be an integer";
      return false;
    }
    if (video.contains("output_format") && !video["output_format"].is_null()) {
      if (!video["output_format"].is_string() ||
          video["output_format"].get<std::string>() != "mp4") {
        error = "video.output_format must be mp4";
        return false;
      }
    }
    video_output = true;
  }
  if (!image_output && !audio_output && !video_output) return true;
  if (image_output && !image_enabled) {
    error = "image output is not enabled on this server";
    return false;
  }
  if (audio_output && !tts_enabled) {
    error = "audio output is not enabled on this server";
    return false;
  }
  if (video_output && !video_enabled) {
    error = "video output is not enabled on this server";
    return false;
  }
  // Streamed PCM chunks are SSE-only, so they cannot ride a single JSON body.
  if (!body.value("stream", false) && audio_output &&
      body["audio"].is_object() && body["audio"].value("stream", false)) {
    error = "audio.stream=true requires stream=true";
    return false;
  }
  Value::Array selected;
  if (image_output) selected.push_back(Value("image"));
  if (audio_output) selected.push_back(Value("audio"));
  if (video_output) selected.push_back(Value("video"));
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
  if (video_output) {
    Value::Map options;
    const json empty_video = json::object();
    const json& video = body.contains("video") && body["video"].is_object()
                            ? body["video"]
                            : empty_video;
    if (video.contains("prompt") && !video["prompt"].is_null()) {
      options["prompt"] = Value(video["prompt"].get<std::string>());
    }
    options["size"] = Value(video.value("size", std::string("1280x720")));
    if (video.contains("seconds") && !video["seconds"].is_null()) {
      options["seconds"] = Value(video["seconds"].get<int64_t>());
    }
    if (video.contains("fps") && !video["fps"].is_null()) {
      options["fps"] = Value(video["fps"].get<int64_t>());
    }
    if (video.contains("seed") && !video["seed"].is_null()) {
      options["seed"] = Value(video["seed"].get<int64_t>());
    }
    options["output_format"] =
        Value(video.value("output_format", std::string("mp4")));
    engine_inputs["video_options"] = Value(std::move(options));
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

}  // namespace umminfer::openai_detail
