// SPDX-License-Identifier: Apache-2.0
#include "openai_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <utility>

#include "umminfer/audio/audio.hpp"
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
  if (event.kind == UnifiedEvent::Kind::kVisualSlot) {
    out["request_id"] = event.request_id;
    out["slot_id"] = event.slot_id;
    out["slot_index"] = event.slot_index;
    out["media_type"] = event.stage_id;
    out["prompt"] = event.prompt;
    out["text_offset"] = event.text_offset;
    if (!event.continuity_mode.empty()) {
      out["continuity_mode"] = event.continuity_mode;
    }
    if (event.anchor_slot_index >= 0) {
      out["anchor_slot_index"] = event.anchor_slot_index;
    }
    if (event.anchor_explicit) out["anchor_explicit"] = true;
  } else if (event.slot_index >= 0 && !event.slot_id.empty()) {
    // Slot-attributed leg_error events carry the same identity as the
    // artifact they describe.
    out["slot_id"] = event.slot_id;
    out["slot_index"] = event.slot_index;
    if (event.kind == UnifiedEvent::Kind::kLegError) {
      if (event.anchor_slot_index >= 0) {
        out["anchor_slot_index"] = event.anchor_slot_index;
      }
      if (event.anchor_explicit) out["anchor_explicit"] = true;
      if (!event.anchor_sha256.empty()) {
        out["anchor_sha256"] = event.anchor_sha256;
      }
    }
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
  size_t video_count = 0;
  has_audio = false;
  try {
    const json& messages = body["messages"];
    if (!messages.is_array() || messages.empty()) {
      error = "messages must be a non-empty array";
      return false;
    }
    bool saw_conversation_turn = false;
    bool final_turn_has_non_audio = false;
    for (size_t i = 0; i < messages.size(); ++i) {
      const json& message = messages[i];
      const bool is_last = (i + 1 == messages.size());
      const std::string role = message["role"].get<std::string>();
      if (role != "system" && role != "user" && role != "assistant") {
        error = "unified chat supports system, user and assistant roles only";
        return false;
      }
      if (role == "system") {
        if (saw_conversation_turn) {
          error = "system messages must appear before conversation turns";
          return false;
        }
      } else {
        saw_conversation_turn = true;
      }
      if (is_last && role != "user") {
        error = "unified chat messages must end with a user message";
        return false;
      }

      const json& content = message["content"];
      if (content.is_string()) {
        if (content.get<std::string>().empty()) {
          error = "message content string must be nonempty";
          return false;
        }
        if (is_last) final_turn_has_non_audio = true;
        continue;
      }
      if (!content.is_array()) {
        error = "unified chat content must be a text string or content part array";
        return false;
      }
      if (content.empty()) {
        error = "unified chat message content array must be nonempty";
        return false;
      }
      size_t msg_text_count = 0;
      for (const auto& part : content) {
        if (!part.is_object() || !part.contains("type") ||
            !part["type"].is_string()) {
          error = "unified chat supports input_audio, image_url, video_url and text parts only";
          return false;
        }
        const std::string type = part["type"].get<std::string>();
        if (type == "image_url") {
          if (role != "user") {
            error = "image_url is only supported on user messages";
            return false;
          }
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
          if (is_last) final_turn_has_non_audio = true;
          continue;
        }
        if (type == "video_url") {
          if (role != "user") {
            error = "video_url is only supported on user messages";
            return false;
          }
          if (!part.contains("video_url") || !part["video_url"].is_object() ||
              !part["video_url"].contains("url") ||
              !part["video_url"]["url"].is_string() ||
              part["video_url"]["url"].get<std::string>().empty()) {
            error = "video_url part requires a nonempty video_url.url";
            return false;
          }
          const std::string& url = part["video_url"]["url"].get_ref<const std::string&>();
          if (part["video_url"].contains("frame_selection")) {
            const json& selection = part["video_url"]["frame_selection"];
            if (!selection.is_object() || !selection.contains("policy") ||
                !selection["policy"].is_string() ||
                (selection["policy"] != "uniform" && selection["policy"] != "temporal")) {
              error = "video_url.frame_selection.policy must be uniform or temporal";
              return false;
            }
            for (const auto& item : selection.items()) {
              if (item.key() != "policy" && item.key() != "motion_threshold" &&
                  item.key() != "scene_threshold") {
                error = "unsupported video_url.frame_selection field: " + item.key();
                return false;
              }
            }
            if (selection["policy"] == "uniform" && selection.size() != 1) {
              error = "uniform video frame_selection does not accept temporal thresholds";
              return false;
            }
            for (const char* key : {"motion_threshold", "scene_threshold"}) {
              if (!selection.contains(key)) continue;
              if (!selection[key].is_number()) {
                error = std::string("video_url.frame_selection.") + key + " must be a number";
                return false;
              }
              const double threshold = selection[key].get<double>();
              if (!std::isfinite(threshold) || threshold < 0.0 || threshold > 1.0) {
                error = std::string("video_url.frame_selection.") + key + " must be in [0,1]";
                return false;
              }
            }
          }
          const size_t comma = url.find(',');
          if (url.rfind("data:video/", 0) != 0 || comma == std::string::npos ||
              url.substr(0, comma).find(";base64") == std::string::npos) {
            error = "video_url.url must be a data:video/...;base64,... URL";
            return false;
          }
          const std::string decoded =
              util::base64_decode(url.substr(comma + 1), 32u << 20);
          if (decoded.empty()) {
            error = "video_url.url payload must be non-empty";
            return false;
          }
          if (++video_count > 1) {
            error = "unified chat accepts at most one video_url part";
            return false;
          }
          if (is_last) final_turn_has_non_audio = true;
          continue;
        }
        if (type == "text") {
          if (!part.contains("text") || !part["text"].is_string() ||
              part["text"].get<std::string>().empty()) {
            error = "text part requires a nonempty text string";
            return false;
          }
          if (++msg_text_count > 1) {
            error = "unified chat accepts at most one text part per message";
            return false;
          }
          if (is_last) final_turn_has_non_audio = true;
          continue;
        }
        if (type != "input_audio") {
          error = "unified chat supports input_audio, image_url, video_url and text parts only";
          return false;
        }
        if (!is_last || role != "user") {
          error = "input_audio is only supported on the final user message";
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
    if (image_count > 0 && video_count > 0) {
      error = "unified chat cannot combine image_url and video_url in one turn";
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
    } else if (!final_turn_has_non_audio) {
      error = "unified chat without input_audio requires text, image or video content";
      return false;
    }
  } catch (const std::exception& exc) {
    error = exc.what();
    return false;
  }
  return true;
}

namespace {

constexpr size_t kMaxMediaInputBytes = 16u << 20;  // 16 MiB decoded image/WAV

// One validation rule for every image data URL on the wire: image.image,
// video.image, and /v1/visual/render image (image i2i and video i2v).
bool validate_image_data_url(const json& value, const char* field,
                             std::string& out_url, std::string& error) {
  if (!value.is_string()) {
    error = std::string(field) + " must be a data:image/...;base64,... string";
    return false;
  }
  const std::string& url = value.get_ref<const std::string&>();
  const size_t comma = url.find(',');
  if (url.rfind("data:image/", 0) != 0 || comma == std::string::npos ||
      url.substr(0, comma).find(";base64") == std::string::npos) {
    error = std::string(field) + " must be a data:image/...;base64,... URL";
    return false;
  }
  try {
    const std::string decoded =
        util::base64_decode(url.substr(comma + 1), kMaxMediaInputBytes);
    if (decoded.empty()) {
      error = std::string(field) + " payload must be non-empty";
      return false;
    }
  } catch (const std::exception& exc) {
    error = std::string("invalid ") + field + " base64: " + exc.what();
    return false;
  }
  out_url = url;
  return true;
}

bool parse_image_options(const json& body, bool& enabled, Value::Map& out,
                         std::string& error) {
  if (!body.contains("image") || body["image"].is_null()) {
    if (!enabled) return true;
    out["size"] = Value("512x512");
    out["output_format"] = Value("png");
    return true;
  }
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
    out["prompt"] = Value(image["prompt"].get<std::string>());
  }
  if (image.contains("size") && !image["size"].is_null()) {
    if (!image["size"].is_string() || image["size"].get<std::string>().empty()) {
      error = "image.size must be a non-empty string";
      return false;
    }
    out["size"] = Value(image["size"].get<std::string>());
  } else {
    out["size"] = Value("512x512");
  }
  if (image.contains("seed") && !image["seed"].is_null()) {
    if (!image["seed"].is_number_integer()) {
      error = "image.seed must be an integer";
      return false;
    }
    out["seed"] = Value(image["seed"].get<int64_t>());
  }
  if (image.contains("steps") && !image["steps"].is_null()) {
    if (!image["steps"].is_number_integer() ||
        image["steps"].get<int64_t>() < 1 ||
        image["steps"].get<int64_t>() > 100) {
      error = "image.steps must be an integer between 1 and 100";
      return false;
    }
    out["steps"] = Value(image["steps"].get<int64_t>());
  }
  if (image.contains("image") && !image["image"].is_null()) {
    std::string url;
    if (!validate_image_data_url(image["image"], "image.image", url, error)) {
      return false;
    }
    out["image"] = Value(std::move(url));
  }
  if (image.contains("output_format") && !image["output_format"].is_null()) {
    // The unified image leg emits PNG only.
    if (!image["output_format"].is_string() ||
        image["output_format"].get<std::string>() != "png") {
      error = "image.output_format must be png: the image stage emits PNG only";
      return false;
    }
  }
  out["output_format"] = Value("png");
  enabled = true;
  return true;
}

bool parse_audio_options(const json& body, bool& enabled, Value::Map& out,
                         std::string& error) {
  if (!body.contains("audio") || body["audio"].is_null()) {
    if (!enabled) return true;
    error = "audio requires a non-empty voice";
    return false;
  }
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
  if (audio.contains("stream") && !audio["stream"].is_null() &&
      !audio["stream"].is_boolean()) {
    error = "audio.stream must be a boolean";
    return false;
  }
  out["voice"] = Value(audio["voice"].get<std::string>());
  out["format"] = Value("wav");
  out["sample_rate"] = Value(int64_t{24000});
  if (audio.value("stream", false)) out["stream"] = Value(true);
  if (audio.contains("ref_audio") && !audio["ref_audio"].is_null()) {
    if (!audio["ref_audio"].is_string()) {
      error = "audio.ref_audio must be a base64 or data:audio/wav;base64,... string";
      return false;
    }
    std::string wav_bytes;
    if (!decode_wav_ref_audio(audio["ref_audio"].get_ref<const std::string&>(),
                              wav_bytes, error)) {
      return false;
    }
    out["ref_audio"] = Value::bin(std::move(wav_bytes));
  }
  enabled = true;
  return true;
}

bool parse_video_options(const json& body, bool& enabled, Value::Map& out,
                         std::string& error) {
  if (!body.contains("video") || body["video"].is_null()) {
    if (!enabled) return true;
    out["size"] = Value("1280x720");
    out["output_format"] = Value("mp4");
    return true;
  }
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
    out["prompt"] = Value(video["prompt"].get<std::string>());
  }
  if (video.contains("size") && !video["size"].is_null()) {
    if (!video["size"].is_string() || video["size"].get<std::string>().empty()) {
      error = "video.size must be a non-empty string";
      return false;
    }
    out["size"] = Value(video["size"].get<std::string>());
  } else {
    out["size"] = Value("1280x720");
  }
  if (video.contains("seconds") && !video["seconds"].is_null()) {
    if (!video["seconds"].is_number_integer() ||
        video["seconds"].get<int64_t>() < 1) {
      error = "video.seconds must be a positive integer";
      return false;
    }
    out["seconds"] = Value(video["seconds"].get<int64_t>());
  }
  if (video.contains("fps") && !video["fps"].is_null()) {
    if (!video["fps"].is_number_integer() ||
        video["fps"].get<int64_t>() < 1) {
      error = "video.fps must be a positive integer";
      return false;
    }
    out["fps"] = Value(video["fps"].get<int64_t>());
  }
  if (video.contains("num_frames") && !video["num_frames"].is_null()) {
    if (!video["num_frames"].is_number_integer() ||
        video["num_frames"].get<int64_t>() < 1) {
      error = "video.num_frames must be a positive integer";
      return false;
    }
    out["num_frames"] = Value(video["num_frames"].get<int64_t>());
  }
  if (video.contains("steps") && !video["steps"].is_null()) {
    if (!video["steps"].is_number_integer() ||
        video["steps"].get<int64_t>() < 1 ||
        video["steps"].get<int64_t>() > 100) {
      error = "video.steps must be an integer between 1 and 100";
      return false;
    }
    out["steps"] = Value(video["steps"].get<int64_t>());
  }
  if (video.contains("guidance_scale") && !video["guidance_scale"].is_null()) {
    if (!video["guidance_scale"].is_number() ||
        !std::isfinite(video["guidance_scale"].get<double>()) ||
        video["guidance_scale"].get<double>() < 0.0) {
      error = "video.guidance_scale must be a finite non-negative number";
      return false;
    }
    out["guidance_scale"] = Value(video["guidance_scale"].get<double>());
  }
  if (video.contains("teacache_thresh") && !video["teacache_thresh"].is_null()) {
    if (!video["teacache_thresh"].is_number() ||
        !std::isfinite(video["teacache_thresh"].get<double>()) ||
        video["teacache_thresh"].get<double>() < 0.0 ||
        video["teacache_thresh"].get<double>() > 0.25) {
      error = "video.teacache_thresh must be a finite number between 0.0 and 0.25";
      return false;
    }
    out["teacache_thresh"] = Value(video["teacache_thresh"].get<double>());
  }
  if (video.contains("seed") && !video["seed"].is_null()) {
    if (!video["seed"].is_number_integer()) {
      error = "video.seed must be an integer";
      return false;
    }
    out["seed"] = Value(video["seed"].get<int64_t>());
  }
  if (video.contains("image") && !video["image"].is_null()) {
    std::string url;
    if (!validate_image_data_url(video["image"], "video.image", url, error)) {
      return false;
    }
    out["image"] = Value(std::move(url));
  }
  if (video.contains("output_format") && !video["output_format"].is_null()) {
    if (!video["output_format"].is_string() ||
        video["output_format"].get<std::string>() != "mp4") {
      error = "video.output_format must be mp4";
      return false;
    }
  }
  out["output_format"] = Value("mp4");
  enabled = true;
  return true;
}

}  // namespace

bool decode_wav_ref_audio(std::string_view encoded, std::string& wav_bytes,
                          std::string& error) {
  if (encoded.empty()) {
    error = "ref_audio must be non-empty";
    return false;
  }
  std::string_view b64 = encoded;
  if (encoded.rfind("data:", 0) == 0) {
    const size_t comma = encoded.find(',');
    if (comma == std::string_view::npos ||
        encoded.substr(0, comma).find(";base64") == std::string_view::npos) {
      error = "ref_audio data URL must use base64 encoding";
      return false;
    }
    b64 = encoded.substr(comma + 1);
  }
  try {
    wav_bytes = util::base64_decode(std::string(b64), kMaxMediaInputBytes);
    const auto view = audio::parse_wav_pcm16(wav_bytes);
    if (view.pcm.empty()) {
      error = "ref_audio WAV must contain non-empty PCM16 audio";
      return false;
    }
  } catch (const std::exception& exc) {
    error = std::string("invalid ref_audio WAV: ") + exc.what();
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
  // Optional interleaved-narrative control. Type validation happens here for
  // every request; the modality cross-checks follow the output flags because
  // they depend on modalities.
  bool narrative_enabled = false;
  std::string narrative_mode = "auto";
  int64_t narrative_slots = 3;
  bool narrative_allow_text_only = false;
  std::string narrative_continuity_mode = "off";
  bool explicit_continuity_mode = false;
  std::string narrative_style_anchor;
  // The contract gates narrative on the explicit `modalities` array, not on a
  // top-level image/video object that also enables an output leg.
  bool modalities_image = false;
  bool modalities_video = false;
  if (body.contains("narrative") && !body["narrative"].is_null()) {
    const json& narrative = body["narrative"];
    if (!narrative.is_object()) {
      error = "narrative must be an object";
      return false;
    }
    if (narrative.contains("enabled") && !narrative["enabled"].is_null()) {
      if (!narrative["enabled"].is_boolean()) {
        error = "narrative.enabled must be a boolean";
        return false;
      }
      narrative_enabled = narrative["enabled"].get<bool>();
    }
    if (narrative.contains("visual_mode") && !narrative["visual_mode"].is_null()) {
      if (!narrative["visual_mode"].is_string()) {
        error = "narrative.visual_mode must be a string";
        return false;
      }
      const std::string mode = narrative["visual_mode"].get<std::string>();
      if (mode != "auto" && mode != "image" && mode != "video") {
        error = "narrative.visual_mode must be auto, image or video";
        return false;
      }
      narrative_mode = mode;
    }
    if (narrative.contains("max_slots") && !narrative["max_slots"].is_null()) {
      if (!narrative["max_slots"].is_number_integer() ||
          narrative["max_slots"].get<int64_t>() < 1 ||
          narrative["max_slots"].get<int64_t>() > 6) {
        error = "narrative.max_slots must be an integer between 1 and 6";
        return false;
      }
      narrative_slots = narrative["max_slots"].get<int64_t>();
    }
    if (narrative.contains("allow_text_only") &&
        !narrative["allow_text_only"].is_null()) {
      if (!narrative["allow_text_only"].is_boolean()) {
        error = "narrative.allow_text_only must be a boolean";
        return false;
      }
      narrative_allow_text_only = narrative["allow_text_only"].get<bool>();
    }
    if (narrative.contains("continuity_mode") &&
        !narrative["continuity_mode"].is_null()) {
      if (!narrative["continuity_mode"].is_string()) {
        error =
            "narrative.continuity_mode must be 'off', 'prompt_anchor', or 'i2v_chain'";
        return false;
      }
      const std::string cmode = narrative["continuity_mode"].get<std::string>();
      if (cmode != "off" && cmode != "prompt_anchor" && cmode != "i2v_chain") {
        error =
            "narrative.continuity_mode must be 'off', 'prompt_anchor', or 'i2v_chain'";
        return false;
      }
      narrative_continuity_mode = cmode;
      explicit_continuity_mode = true;
    }
    if (narrative.contains("style_anchor") &&
        !narrative["style_anchor"].is_null()) {
      if (!narrative["style_anchor"].is_string()) {
        error = "narrative.style_anchor must be a string";
        return false;
      }
      const std::string& raw =
          narrative["style_anchor"].get_ref<const std::string&>();
      const size_t first = raw.find_first_not_of(" \t\r\n");
      if (first != std::string::npos) {
        const size_t last = raw.find_last_not_of(" \t\r\n");
        std::string trimmed = raw.substr(first, last - first + 1);
        if (trimmed.size() > 256) {
          error = "narrative.style_anchor must be at most 256 UTF-8 bytes";
          return false;
        }
        narrative_style_anchor = std::move(trimmed);
      }
    }
    if (!explicit_continuity_mode && !narrative_style_anchor.empty()) {
      narrative_continuity_mode = "prompt_anchor";
    }
  }
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
      if (name == "image") {
        image_output = true;
        modalities_image = true;
      } else if (name == "audio") {
        audio_output = true;
      } else if (name == "video") {
        video_output = true;
        modalities_video = true;
      }
      else {
        error = "modalities values must be text, image, audio or video";
        return false;
      }
    }
  }

  Value::Map image_opts;
  Value::Map audio_opts;
  Value::Map video_opts;
  if (!parse_image_options(body, image_output, image_opts, error)) return false;
  if (!parse_audio_options(body, audio_output, audio_opts, error)) return false;
  if (!parse_video_options(body, video_output, video_opts, error)) return false;
  if (narrative_enabled) {
    if (!body.value("stream", false)) {
      error = "narrative.enabled=true requires stream=true";
      return false;
    }
    if (!modalities_image && !modalities_video) {
      error = "narrative requires an image or video modality";
      return false;
    }
    if (narrative_mode == "image" && !modalities_image) {
      error = "narrative.visual_mode=image requires the image modality";
      return false;
    }
    if (narrative_mode == "video" && !modalities_video) {
      error = "narrative.visual_mode=video requires the video modality";
      return false;
    }
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
  const bool audio_stream =
      audio_opts.count("stream") > 0 && audio_opts["stream"].as_bool();
  if (!body.value("stream", false) && audio_output && audio_stream) {
    error = "audio.stream=true requires stream=true";
    return false;
  }

  Value::Array selected;
  if (image_output) {
    selected.push_back(Value("image"));
    engine_inputs["image_options"] = Value(std::move(image_opts));
  }
  if (audio_output) {
    selected.push_back(Value("audio"));
    engine_inputs["audio_options"] = Value(std::move(audio_opts));
  }
  if (video_output) {
    selected.push_back(Value("video"));
    engine_inputs["video_options"] = Value(std::move(video_opts));
  }
  if (narrative_enabled) {
    Value::Map narrative;
    narrative["enabled"] = Value(true);
    narrative["visual_mode"] = Value(narrative_mode);
    narrative["max_slots"] = Value(narrative_slots);
    if (narrative_allow_text_only) {
      narrative["allow_text_only"] = Value(true);
    }
    if (narrative_continuity_mode != "off" || !narrative_style_anchor.empty()) {
      narrative["continuity_mode"] = Value(narrative_continuity_mode);
      if (!narrative_style_anchor.empty()) {
        narrative["style_anchor"] = Value(narrative_style_anchor);
      }
    }
    engine_inputs["narrative"] = Value(std::move(narrative));
  }
  engine_inputs["output_modalities"] = Value(std::move(selected));
  return true;
}

SyncCoordinatorResult run_sync_coordinator_request(
    Coordinator& coordinator, const httplib::Request& req, StagePayload payload,
    const std::string& stage_name, double timeout_s,
    const std::function<void(const std::string&)>& on_submitted) {
  SyncCoordinatorResult out;
  try {
    out.request_id = coordinator.submit(std::move(payload), stage_name);
  } catch (const std::exception& error) {
    out.status = SyncCoordinatorResult::Status::kSubmitError;
    out.submit_error = error.what();
    return out;
  }
  if (on_submitted) on_submitted(out.request_id);

  bool settled = false;
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::duration<double>(timeout_s);
  while (!settled && !req.is_connection_closed()) {
    const auto remaining = deadline - std::chrono::steady_clock::now();
    if (remaining <= std::chrono::steady_clock::duration::zero()) break;
    settled = coordinator.wait(
        out.request_id,
        std::min(0.1, std::chrono::duration<double>(remaining).count()));
  }
  if (req.is_connection_closed()) {
    out.settled_before_close = settled;
    coordinator.abort(out.request_id, "client");
    coordinator.release(out.request_id);
    out.status = SyncCoordinatorResult::Status::kClientClosed;
    return out;
  }
  out.info = coordinator.info(out.request_id);
  if (!settled || !out.info) {
    coordinator.abort(out.request_id, "timeout");
    coordinator.release(out.request_id);
    out.status = SyncCoordinatorResult::Status::kTimeout;
    return out;
  }
  coordinator.release(out.request_id);
  out.status = SyncCoordinatorResult::Status::kSettled;
  return out;
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

namespace umminfer {

using namespace openai_detail;

// POST /v1/visual/render. One direct image or video render against the stage
// the unified engine already registered: the same Coordinator submit/abort
// path and the same status conventions as /v1/audio/speech.
void OpenAIApi::Impl::handle_visual_render(const httplib::Request& req,
                                           httplib::Response& res) {
  RouteMetric route_metric{"visual_render", &res};
  auto reject = [&res](int status, const std::string& message,
                       const char* type) {
    res.status = status;
    res.set_content(
        json{{"error", {{"message", util::utf8_sanitized(message)},
                        {"type", type}}}}
            .dump(),
        "application/json");
  };
  if (!config_.upstream_kind.empty()) {
    reject(501,
           "visual render is not supported: requests are served by a fixed upstream",
           "unsupported_route");
    return;
  }
  if (unified_engine_ == nullptr) {
    reject(501,
           "visual render is not implemented by this server: no visual stage is registered",
           "unsupported_route");
    return;
  }
  const json body = json::parse(req.body, nullptr, false);
  if (body.is_discarded() || !body.is_object()) {
    reject(400, "request body must be a JSON object", "invalid_request");
    return;
  }
  static const std::set<std::string> kAllowedFields = {
      "kind", "prompt", "size", "steps", "num_frames", "fps", "seconds",
      "guidance_scale", "seed", "image", "teacache_thresh"};
  for (auto it = body.begin(); it != body.end(); ++it) {
    if (!kAllowedFields.count(it.key())) {
      reject(400, "unsupported field: " + it.key(), "invalid_request");
      return;
    }
  }
  if (!body.contains("kind") || !body["kind"].is_string() ||
      (body["kind"].get<std::string>() != "image" &&
       body["kind"].get<std::string>() != "video")) {
    reject(400, "kind must be image or video", "invalid_request");
    return;
  }
  const std::string kind = body["kind"].get<std::string>();
  const bool image = kind == "image";
  if (image ? !unified_engine_->image_enabled()
            : !unified_engine_->video_enabled()) {
    reject(501,
           kind +
               " render is not implemented by this server: no stage is registered",
           "unsupported_route");
    return;
  }
  if (!body.contains("prompt") || !body["prompt"].is_string() ||
      body["prompt"].get_ref<const std::string&>().empty()) {
    reject(400, "prompt must be a non-empty string", "invalid_request");
    return;
  }
  Value::Map inputs;
  inputs["prompt"] = Value(body["prompt"].get<std::string>());
  if (body.contains("size") && !body["size"].is_null()) {
    if (!body["size"].is_string() ||
        body["size"].get_ref<const std::string&>().empty()) {
      reject(400, "size must be a non-empty string", "invalid_request");
      return;
    }
    inputs["size"] = Value(body["size"].get<std::string>());
  } else {
    inputs["size"] = Value(image ? "512x512" : "480x320");
  }
  if (body.contains("steps") && !body["steps"].is_null()) {
    if (!body["steps"].is_number_integer() ||
        body["steps"].get<int64_t>() < 1 ||
        body["steps"].get<int64_t>() > 100) {
      reject(400, "steps must be an integer between 1 and 100",
             "invalid_request");
      return;
    }
    inputs["steps"] = Value(body["steps"].get<int64_t>());
  }
  if (body.contains("seed") && !body["seed"].is_null()) {
    if (!body["seed"].is_number_integer() ||
        (body["seed"].is_number_unsigned() &&
         body["seed"].get<uint64_t>() >
             static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))) {
      reject(400, "seed must be an integer", "invalid_request");
      return;
    }
    inputs["seed"] = Value(body["seed"].get<int64_t>());
  }
  if (image) {
    if (body.contains("num_frames") || body.contains("fps") ||
        body.contains("seconds") || body.contains("guidance_scale") ||
        body.contains("teacache_thresh")) {
      reject(400, "video-only fields are not valid for kind image",
             "invalid_request");
      return;
    }
  } else {
    for (const char* key : {"num_frames", "fps", "seconds"}) {
      if (!body.contains(key) || body[key].is_null()) continue;
      if (!body[key].is_number_integer() || body[key].get<int64_t>() < 1) {
        reject(400, std::string(key) + " must be a positive integer",
               "invalid_request");
        return;
      }
      inputs[key] = Value(body[key].get<int64_t>());
    }
    if (body.contains("guidance_scale") && !body["guidance_scale"].is_null()) {
      const json& value = body["guidance_scale"];
      if (!value.is_number() || !std::isfinite(value.get<double>()) ||
          value.get<double>() < 0.0) {
        reject(400, "guidance_scale must be a finite non-negative number",
               "invalid_request");
        return;
      }
      inputs["guidance_scale"] = Value(value.get<double>());
    }
    if (body.contains("teacache_thresh") && !body["teacache_thresh"].is_null()) {
      const json& value = body["teacache_thresh"];
      if (!value.is_number() || !std::isfinite(value.get<double>()) ||
          value.get<double>() < 0.0 || value.get<double>() > 0.25) {
        reject(400,
               "teacache_thresh must be a finite number between 0.0 and 0.25",
               "invalid_request");
        return;
      }
      inputs["teacache_thresh"] = Value(value.get<double>());
    }
  }
  if (body.contains("image") && !body["image"].is_null()) {
    std::string url;
    std::string error;
    if (!validate_image_data_url(body["image"], "image", url, error)) {
      reject(400, error, "invalid_request");
      return;
    }
    inputs["image"] = Value(std::move(url));
  }
  inputs["output_format"] = Value(image ? "png" : "mp4");

  AdmissionGuard guard(active_http_requests_, config_.max_in_flight);
  if (!guard.admitted()) {
    reject_overloaded(res);
    return;
  }
  StagePayload payload;
  payload.request.inputs = Value(std::move(inputs));
  const std::string& stage = image ? unified_engine_->image_stage_name()
                                   : unified_engine_->video_stage_name();
  const SyncCoordinatorResult sync = run_sync_coordinator_request(
      coordinator_, req, std::move(payload), stage, config_.request_timeout_s);
  if (sync.status == SyncCoordinatorResult::Status::kSubmitError) {
    reject(503, sync.submit_error, "pipeline_error");
    return;
  }
  if (sync.status == SyncCoordinatorResult::Status::kClientClosed) {
    reject(503, "client closed the connection before the render finished",
           "render_error");
    return;
  }
  if (sync.status == SyncCoordinatorResult::Status::kTimeout) {
    reject(504, "render timed out", "render_error");
    return;
  }
  const RequestInfo& info = *sync.info;
  if (info.state != "completed") {
    reject(500, info.error.empty() ? "render failed" : info.error,
           "render_error");
    return;
  }
  const Value* data =
      info.result.is_map() ? info.result.find("data") : nullptr;
  if (data == nullptr || !data->is_bin() || data->as_bin().empty()) {
    reject(500, "the " + kind + " stage returned no bytes", "render_error");
    return;
  }
  res.set_header("X-UmmInfer-Kind", kind);
  res.set_content(data->as_bin(), image ? "image/png" : "video/mp4");
}

}  // namespace umminfer
