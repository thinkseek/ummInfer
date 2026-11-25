// SPDX-License-Identifier: Apache-2.0
#include "pipeline/unified_schema.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>

namespace umminfer::unified_detail {

UnifiedModality modality_of_part_type(const std::string& type) {
  if (type == "input_audio") return UnifiedModality::kAudio;
  if (type == "image_url") return UnifiedModality::kImage;
  if (type == "video_url" || type == "video") return UnifiedModality::kVideo;
  return UnifiedModality::kText;
}

std::string_view chunk_text(const Value& chunk) {
  if (!chunk.is_map()) return {};
  const Value* text = chunk.find("text");
  if (text != nullptr && text->is_str()) return text->as_str();
  return {};
}

bool chunk_is_final(const Value& chunk) {
  if (!chunk.is_map()) return false;
  const Value* flag = chunk.find("is_final");
  return flag != nullptr && flag->is_bool() && flag->as_bool();
}

std::optional<std::string> asr_result_text(const Value& result) {
  if (result.is_str()) return result.as_str();
  if (result.is_map()) {
    const Value* text = result.find("text");
    if (text != nullptr && text->is_str()) return text->as_str();
  }
  return std::nullopt;
}

std::optional<std::string> vlm_result_text(const Value& result) {
  if (auto text = asr_result_text(result)) return text;
  if (result.is_map()) {
    const Value* content = result.find("content");
    if (content != nullptr && content->is_str()) return content->as_str();
  }
  return std::nullopt;
}

std::string user_text_from_inputs(const Value& original_inputs) {
  std::string joined;
  if (!original_inputs.is_map()) return joined;
  const Value* messages = original_inputs.find("messages");
  if (messages == nullptr || !messages->is_array()) return joined;
  const auto& arr = messages->as_array();
  for (size_t idx = arr.size(); idx > 0; --idx) {
    const Value& message = arr[idx - 1];
    if (!message.is_map()) continue;
    const Value* role = message.find("role");
    if (role != nullptr && role->is_str() && role->as_str() != "user") {
      continue;
    }
    const Value* content = message.find("content");
    if (content == nullptr) return joined;
    if (content->is_str()) {
      return content->as_str();
    }
    if (!content->is_array()) return joined;
    for (const Value& part : content->as_array()) {
      const Value* type = part.find("type");
      if (type == nullptr || !type->is_str() ||
          modality_of_part_type(type->as_str()) != UnifiedModality::kText) {
        continue;
      }
      const Value* text = part.find("text");
      if (text != nullptr && text->is_str() && !text->as_str().empty()) {
        if (!joined.empty()) joined += " ";
        joined += text->as_str();
      }
    }
    return joined;
  }
  return joined;
}

ValidatedSubmission validate_submission(const Value& inputs,
                                        const UnifiedConfig& config) {
  if (!inputs.is_map() || inputs.find("messages") == nullptr ||
      !inputs.find("messages")->is_array() ||
      inputs.find("messages")->as_array().empty()) {
    throw ProtocolError("unified request inputs must contain a non-empty messages array");
  }

  if (const Value* declared = inputs.find("modalities"); declared != nullptr) {
    if (!declared->is_array()) {
      throw ProtocolError("unified request modalities must be an array");
    }
    for (const Value& token : declared->as_array()) {
      if (!token.is_str()) {
        throw ProtocolError("unified request modality must be a string");
      }
      const auto modality = modality_from_string(token.as_str());
      if (!modality) throw ProtocolError("unknown modality: " + token.as_str());
      if (*modality == UnifiedModality::kVideo) {
        throw ProtocolError(
            "modality video is not supported as an input modality; use output_modalities for video output");
      }
    }
  }

  size_t image_count = 0;
  size_t video_count = 0;
  size_t audio_part_count = 0;
  bool saw_conversation_turn = false;
  const auto& messages = inputs.find("messages")->as_array();
  for (size_t i = 0; i < messages.size(); ++i) {
    const Value& message = messages[i];
    const bool is_last = (i + 1 == messages.size());
    if (!message.is_map()) {
      throw ProtocolError("each message must be a map");
    }
    const Value* role_val = message.find("role");
    const std::string role =
        (role_val != nullptr && role_val->is_str()) ? role_val->as_str()
                                                    : std::string();
    if (role != "system" && role != "user" && role != "assistant") {
      throw ProtocolError(
          "unified request supports system, user and assistant roles only");
    }
    if (role == "system") {
      if (saw_conversation_turn) {
        throw ProtocolError(
            "system messages must appear before conversation turns");
      }
    } else {
      saw_conversation_turn = true;
    }
    if (is_last && role != "user") {
      throw ProtocolError("unified request messages must end with a user message");
    }

    const Value* content = message.find("content");
    if (content == nullptr) {
      throw ProtocolError("each message requires content");
    }
    if (content->is_str()) {
      if (content->as_str().empty()) {
        throw ProtocolError("message content string must be nonempty");
      }
      continue;
    }
    if (!content->is_array()) {
      throw ProtocolError(
          "message content must be a text string or a content part array");
    }
    if (content->as_array().empty()) {
      throw ProtocolError("message content array must be nonempty");
    }
    size_t msg_text_count = 0;
    for (const Value& part : content->as_array()) {
      if (!part.is_map()) continue;
      const Value* type = part.find("type");
      if (type == nullptr || !type->is_str()) continue;
      const UnifiedModality modality = modality_of_part_type(type->as_str());
      if (modality == UnifiedModality::kVideo) {
        if (role != "user") {
          throw ProtocolError("video_url is only supported on user messages");
        }
        if (type->as_str() != "video_url") {
          throw ProtocolError(std::string("content part ") + type->as_str() +
                              " is not supported; use video_url");
        }
        const Value* video_url = part.find("video_url");
        if (video_url == nullptr || !video_url->is_map() ||
            video_url->find("url") == nullptr ||
            !video_url->find("url")->is_str() ||
            video_url->find("url")->as_str().empty()) {
          throw ProtocolError("video_url part requires a nonempty video_url.url");
        }
        if (const Value* selection = video_url->find("frame_selection");
            selection != nullptr) {
          if (!selection->is_map()) {
            throw ProtocolError("video_url.frame_selection must be an object");
          }
          const Value* policy = selection->find("policy");
          if (policy == nullptr || !policy->is_str() ||
              (policy->as_str() != "uniform" && policy->as_str() != "temporal")) {
            throw ProtocolError(
                "video_url.frame_selection.policy must be uniform or temporal");
          }
          for (const auto& [key, value] : selection->as_map()) {
            if (key != "policy" && key != "motion_threshold" &&
                key != "scene_threshold") {
              throw ProtocolError("unsupported video_url.frame_selection field: " + key);
            }
            (void)value;
          }
          if (policy->as_str() == "uniform" && selection->as_map().size() != 1) {
            throw ProtocolError(
                "uniform video frame_selection does not accept temporal thresholds");
          }
          for (const char* key : {"motion_threshold", "scene_threshold"}) {
            const Value* threshold = selection->find(key);
            if (threshold == nullptr) continue;
            if (!threshold->is_int() && !threshold->is_float()) {
              throw ProtocolError(std::string("video_url.frame_selection.") + key +
                                  " must be a number");
            }
            const double value = threshold->as_double();
            if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
              throw ProtocolError(std::string("video_url.frame_selection.") + key +
                                  " must be in [0,1]");
            }
          }
        }
        if (++video_count > 1) {
          throw ProtocolError("unified request accepts at most one video_url part");
        }
      } else if (modality == UnifiedModality::kText) {
        // The HTTP boundary and this schema apply one rule: at most one text
        // part per message.
        if (++msg_text_count > 1) {
          throw ProtocolError(
              "unified request accepts at most one text part per message");
        }
      } else if (modality == UnifiedModality::kAudio) {
        if (!is_last || role != "user") {
          throw ProtocolError(
              "input_audio is only supported on the final user message");
        }
        if (++audio_part_count > 1) {
          throw ProtocolError(
              "unified request accepts at most one input_audio part");
        }
      } else if (modality == UnifiedModality::kImage) {
        if (role != "user") {
          throw ProtocolError("image_url is only supported on user messages");
        }
        const Value* image_url = part.find("image_url");
        if (image_url == nullptr || !image_url->is_map() ||
            image_url->find("url") == nullptr ||
            !image_url->find("url")->is_str() ||
            image_url->find("url")->as_str().empty()) {
          throw ProtocolError("image_url part requires a nonempty image_url.url");
        }
        if (++image_count > 1) {
          throw ProtocolError("unified request accepts at most one image_url part");
        }
      }
    }
  }
  if (image_count > 0 && video_count > 0) {
    throw ProtocolError(
        "unified request cannot combine image_url and video_url in one turn");
  }

  ValidatedSubmission out;
  out.original_inputs = inputs;
  if (const Value* requested = out.original_inputs.find("asr_partials");
      requested != nullptr) {
    if (!requested->is_bool()) {
      throw ProtocolError("asr_partials must be a boolean");
    }
    out.asr_partials = requested->as_bool();
  }
  if (const Value* requested = out.original_inputs.find("session_history");
      requested != nullptr) {
    if (!requested->is_bool()) {
      throw ProtocolError("session_history must be a boolean");
    }
    out.history_requested = requested->as_bool();
    if (out.history_requested && !config.session_history.enabled) {
      throw ProtocolError(
          "session_history requires engine session history to be enabled");
    }
  }
  if (const Value* requested = out.original_inputs.find("realtime_playback_ack");
      requested != nullptr) {
    if (!requested->is_bool()) {
      throw ProtocolError("realtime_playback_ack must be a boolean");
    }
    out.realtime_playback_ack = requested->as_bool();
  }

  // Optional interleaved-narrative object. Type validation happens here for
  // every request; the modality cross-checks run after output_modalities is
  // parsed, because they depend on it.
  if (const Value* narrative = out.original_inputs.find("narrative");
      narrative != nullptr && !narrative->is_nil()) {
    if (!narrative->is_map()) {
      throw ProtocolError("narrative must be an object");
    }
    if (const Value* enabled = narrative->find("enabled");
        enabled != nullptr && !enabled->is_nil()) {
      if (!enabled->is_bool()) {
        throw ProtocolError("narrative.enabled must be a boolean");
      }
      out.narrative_enabled = enabled->as_bool();
    }
    if (const Value* mode = narrative->find("visual_mode");
        mode != nullptr && !mode->is_nil()) {
      if (!mode->is_str() || (mode->as_str() != "auto" &&
                              mode->as_str() != "image" &&
                              mode->as_str() != "video")) {
        throw ProtocolError(
            "narrative.visual_mode must be auto, image or video");
      }
      out.narrative_visual_mode = mode->as_str();
    }
    if (const Value* slots = narrative->find("max_slots");
        slots != nullptr && !slots->is_nil()) {
      if (!slots->is_int() || slots->as_int() < 1 || slots->as_int() > 6) {
        throw ProtocolError("narrative.max_slots must be an integer in [1, 6]");
      }
      out.narrative_max_slots = static_cast<int>(slots->as_int());
    }
    if (const Value* allow_text_only = narrative->find("allow_text_only");
        allow_text_only != nullptr && !allow_text_only->is_nil()) {
      if (!allow_text_only->is_bool()) {
        throw ProtocolError("narrative.allow_text_only must be a boolean");
      }
      out.narrative_allow_text_only = allow_text_only->as_bool();
    }
  }

  if (const Value* requested = out.original_inputs.find("output_modalities");
      requested != nullptr) {
    if (!requested->is_array()) {
      throw ProtocolError("output_modalities must be an array");
    }
    for (const Value& token : requested->as_array()) {
      if (!token.is_str()) {
        throw ProtocolError("output_modalities values must be strings");
      }
      const std::string& name = token.as_str();
      if (name == "text") continue;
      if (name == "image") out.image_output = true;
      else if (name == "audio") out.audio_output = true;
      else if (name == "video") out.video_output = true;
      else throw ProtocolError("unknown output modality: " + name);
    }
  }
  if (const Value* options = out.original_inputs.find("image_options");
      options != nullptr) {
    if (!options->is_map()) throw ProtocolError("image_options must be a map");
    if (const Value* format = options->find("output_format");
        format != nullptr && !format->is_nil()) {
      if (!format->is_str() || format->as_str() != "png") {
        throw ProtocolError(
            "image_options.output_format must be png: the image stage emits PNG only");
      }
    }
    out.image_options = options->as_map();
  }
  if (const Value* options = out.original_inputs.find("audio_options");
      options != nullptr) {
    if (!options->is_map()) throw ProtocolError("audio_options must be a map");
    if (options->find("text") != nullptr) {
      throw ProtocolError("audio_options must not carry text");
    }
    out.audio_options = options->as_map();
  }
  if (const Value* options = out.original_inputs.find("video_options");
      options != nullptr) {
    if (!options->is_map()) throw ProtocolError("video_options must be a map");
    if (const Value* tc = options->find("teacache_thresh");
        tc != nullptr && !tc->is_nil()) {
      if ((!tc->is_float() && !tc->is_int()) ||
          !std::isfinite(tc->as_double()) ||
          tc->as_double() < 0.0 || tc->as_double() > 0.25) {
        throw ProtocolError(
            "video_options.teacache_thresh must be a finite number in [0.0, 0.25]");
      }
    }
    out.video_options = options->as_map();
  }
  if (out.image_output && config.image_stage.empty()) {
    throw ProtocolError("image output is not enabled on this server");
  }
  if (out.audio_output && config.tts_stage.empty()) {
    throw ProtocolError("audio output is not enabled on this server");
  }
  if (out.video_output && config.video_stage.empty()) {
    throw ProtocolError("video output is not enabled on this server");
  }
  if (out.audio_output) {
    auto voice = out.audio_options.find("voice");
    if (voice == out.audio_options.end() || !voice->second.is_str() ||
        voice->second.as_str().empty()) {
      throw ProtocolError("audio output requires a non-empty voice");
    }
  }
  if (out.realtime_playback_ack) {
    const auto stream = out.audio_options.find("stream");
    if (!out.audio_output || stream == out.audio_options.end() ||
        !stream->second.is_bool() || !stream->second.as_bool()) {
      throw ProtocolError(
          "realtime_playback_ack requires streaming audio output");
    }
  }
  if (out.narrative_enabled) {
    if (!out.image_output && !out.video_output) {
      throw ProtocolError(
          "narrative requires image or video in output_modalities");
    }
    if (out.narrative_visual_mode == "image" && !out.image_output) {
      throw ProtocolError(
          "narrative.visual_mode image requires image output");
    }
    if (out.narrative_visual_mode == "video" && !out.video_output) {
      throw ProtocolError(
          "narrative.visual_mode video requires video output");
    }
  }

  out.has_audio = audio_part_count > 0;
  if (out.has_audio) {
    const Value* audio = inputs.find("audio");
    if (audio != nullptr &&
        (!audio->is_bin() || audio->as_bin().empty() ||
         audio->as_bin().size() % 2 != 0)) {
      throw ProtocolError(
          "unified audio must be nonempty PCM16 with complete samples");
    }
    Value::Map asr_inputs;
    if (audio != nullptr) asr_inputs["audio"] = *audio;
    out.asr_inputs = Value(std::move(asr_inputs));
  }

  if (const Value* session = out.original_inputs.find("session_id");
      session != nullptr && session->is_str() && !session->as_str().empty()) {
    out.session_id = session->as_str();
    out.explicit_session = true;
  }
  if (out.history_requested && !out.explicit_session) {
    throw ProtocolError("session_history requires an explicit session_id");
  }
  out.user_text = user_text_from_inputs(out.original_inputs);
  return out;
}

Value build_vlm_inputs(const std::string& transcript,
                       const Value& original_inputs) {
  Value::Map out;
  if (original_inputs.is_map()) {
    for (const auto& [key, value] : original_inputs.as_map()) {
      if (key != "messages" && key != "modalities" && key != "session_id" &&
          key != "audio" && key != "asr_partials" && key != "session_history" &&
          key != "realtime_playback_ack" && key != "output_modalities" &&
          key != "image_options" &&
          key != "audio_options" && key != "video_options" &&
          key != "narrative") {
        out[key] = value;
      }
    }
  }
  Value::Array out_messages;
  bool has_image = false;
  if (original_inputs.is_map()) {
    const Value* messages = original_inputs.find("messages");
    if (messages != nullptr && messages->is_array() &&
        !messages->as_array().empty()) {
      const auto& src = messages->as_array();
      out_messages.reserve(src.size());
      for (size_t i = 0; i < src.size(); ++i) {
        const Value& message = src[i];
        if (!message.is_map()) continue;
        const bool is_last = (i + 1 == src.size());
        const Value* role_val = message.find("role");
        const std::string role =
            (role_val != nullptr && role_val->is_str() &&
             !role_val->as_str().empty())
                ? role_val->as_str()
                : std::string("user");
        const Value* content = message.find("content");
        Value::Map out_msg;
        out_msg["role"] = Value(role);
        if (!is_last) {
          if (content != nullptr && content->is_str()) {
            out_msg["content"] = Value(content->as_str());
          } else if (content != nullptr && content->is_array()) {
            Value::Array parts;
            for (const Value& part : content->as_array()) {
              const Value* type = part.find("type");
              if (type == nullptr || !type->is_str()) continue;
              const UnifiedModality modality =
                  modality_of_part_type(type->as_str());
              if (modality == UnifiedModality::kImage) {
                has_image = true;
                parts.push_back(part);
              } else if (modality == UnifiedModality::kVideo &&
                         type->as_str() == "video_url") {
                parts.push_back(part);
              } else if (modality == UnifiedModality::kText) {
                parts.push_back(part);
              }
            }
            out_msg["content"] = Value(std::move(parts));
          } else {
            out_msg["content"] = Value("");
          }
          out_messages.push_back(Value(std::move(out_msg)));
          continue;
        }

        // Final user turn: preserve non-audio parts and append any ASR
        // transcript. Single-turn wire shape stays unchanged: string content
        // becomes a one-element text part array and an audio-only turn becomes
        // a plain transcript string.
        Value::Array extra_parts;
        if (content != nullptr && content->is_str()) {
          if (!content->as_str().empty()) {
            Value::Map text_part;
            text_part["type"] = Value("text");
            text_part["text"] = Value(content->as_str());
            extra_parts.push_back(Value(std::move(text_part)));
          }
        } else if (content != nullptr && content->is_array()) {
          for (const Value& part : content->as_array()) {
            const Value* type = part.find("type");
            if (type == nullptr || !type->is_str()) continue;
            const UnifiedModality modality =
                modality_of_part_type(type->as_str());
            if (modality == UnifiedModality::kImage) {
              has_image = true;
              extra_parts.push_back(part);
            } else if (modality == UnifiedModality::kVideo &&
                       type->as_str() == "video_url") {
              extra_parts.push_back(part);
            } else if (modality == UnifiedModality::kText) {
              extra_parts.push_back(part);
            }
          }
        }
        if (!extra_parts.empty()) {
          if (!transcript.empty()) {
            Value::Map text_part;
            text_part["type"] = Value("text");
            text_part["text"] = Value(transcript);
            extra_parts.push_back(Value(std::move(text_part)));
          }
          out_msg["content"] = Value(std::move(extra_parts));
        } else {
          out_msg["content"] = Value(transcript);
        }
        out_messages.push_back(Value(std::move(out_msg)));
      }
    }
  }
  if (out_messages.empty()) {
    Value::Map user_message;
    user_message["role"] = Value("user");
    user_message["content"] = Value(transcript);
    out_messages.push_back(Value(std::move(user_message)));
  }
  out["messages"] = Value(std::move(out_messages));
  Value::Array modalities;
  modalities.push_back(Value("text"));
  if (has_image) modalities.push_back(Value("image"));
  out["modalities"] = Value(std::move(modalities));
  return Value(std::move(out));
}

std::string narrative_director_prompt(const std::string& visual_mode,
                                      int max_slots) {
  std::string out;
  if (visual_mode == "auto") {
    out =
        "You are a multimodal assistant that chooses between pure text and "
        "interleaved visuals. Write coherent, well-structured prose in the "
        "user's language. Only emit visual markers when a visual genuinely "
        "clarifies a scene, diagram, or motion; for pure math, code, logic "
        "proofs, short factual questions, or text-only requests, emit zero "
        "markers and respond in clean prose only. When visuals genuinely help, "
        "insert up to " +
        std::to_string(max_slots) +
        " standalone markers on their own lines (use [[image: ...]] for still "
        "scenes or diagrams, and [[video: ...]] for motion or dynamics); a "
        "generated visual replaces each marker, so never describe or mention "
        "the marker itself.\n"
        "[[image: <vivid concrete visual scene description in English>]]\n"
        "[[video: <vivid concrete motion and camera description in English>]]";
  } else {
    out =
        "You are a film director narrating with interleaved visuals. Write "
        "coherent, well-structured prose in the user's language. At up to " +
        std::to_string(max_slots) +
        " natural narrative beats, insert one standalone visual marker on its "
        "own line; a generated visual replaces it, so never describe or mention "
        "the marker itself.";
    if (visual_mode == "image") {
      out +=
          "\n[[image: <vivid concrete visual scene description in English>]]";
    } else {
      out +=
          "\n[[video: <vivid concrete motion and camera description in English>]]";
    }
  }
  out += "\nDo not use any other bracket syntax.";
  return out;
}

Value replace_result_text(const Value& result, const std::string& text) {
  if (result.is_str()) return Value(text);
  if (result.is_map()) {
    Value::Map out = result.as_map();
    if (const Value* existing = result.find("text");
        existing != nullptr && existing->is_str()) {
      out["text"] = Value(text);
    } else if (const Value* content = result.find("content");
               content != nullptr && content->is_str()) {
      out["content"] = Value(text);
    } else {
      out["text"] = Value(text);
    }
    return Value(std::move(out));
  }
  return result;
}

void prepend_narrative_instructions(Value& vlm_inputs,
                                    const std::string& visual_mode,
                                    int max_slots) {
  if (!vlm_inputs.is_map()) return;
  const Value* messages = vlm_inputs.find("messages");
  if (messages == nullptr || !messages->is_array()) return;
  Value::Map system;
  system["role"] = Value("system");
  system["content"] = Value(narrative_director_prompt(visual_mode, max_slots));
  Value::Array out;
  out.reserve(messages->as_array().size() + 1);
  out.push_back(Value(std::move(system)));
  for (const Value& message : messages->as_array()) out.push_back(message);
  vlm_inputs["messages"] = Value(std::move(out));
}

Value build_image_inputs(const Value::Map& image_options,
                         const std::string& vlm_text) {
  Value::Map out = image_options;
  auto prompt = out.find("prompt");
  if (prompt == out.end() || !prompt->second.is_str() ||
      prompt->second.as_str().empty()) {
    out["prompt"] = Value(vlm_text);
  }
  return Value(std::move(out));
}

Value build_video_inputs(const Value::Map& video_options,
                         const std::string& vlm_text) {
  Value::Map out = video_options;
  auto prompt = out.find("prompt");
  if (prompt == out.end() || !prompt->second.is_str() ||
      prompt->second.as_str().empty()) {
    out["prompt"] = Value(vlm_text);
  }
  return Value(std::move(out));
}

Value build_tts_inputs(const Value::Map& audio_options,
                       const std::string& text, bool tts_stream) {
  Value::Map out = audio_options;
  out["text"] = Value(text);
  if (tts_stream) out["stream"] = Value(true);
  return Value(std::move(out));
}

std::vector<UnifiedEvent> slice_artifact_events(
    const std::string& request_id, const std::string& session_id,
    const std::string& stage_id, const Value& result,
    size_t& artifact_bytes_total, int& next_artifact, int64_t& next_seq) {
  if (!result.is_map()) {
    throw ProtocolError(stage_id + " stage returned a non-map artifact result");
  }
  const Value* data = result.find("data");
  if (data == nullptr || !data->is_bin() || data->as_bin().empty()) {
    throw ProtocolError(stage_id + " stage returned no artifact bytes");
  }
  const Value* media_type = result.find("media_type");
  if (media_type == nullptr || !media_type->is_str()) {
    throw ProtocolError(stage_id + " stage returned no media_type");
  }
  const std::string& media = media_type->as_str();
  const size_t size = data->as_bin().size();
  size_t per_leg_limit = 0;
  if (media == "image") {
    per_leg_limit = kMaxArtifactImageBytes;
  } else if (media == "audio") {
    per_leg_limit = kMaxArtifactAudioBytes;
  } else if (media == "video") {
    per_leg_limit = kMaxArtifactVideoBytes;
  } else {
    throw ProtocolError("artifact media_type must be image, audio or video");
  }
  if (size > per_leg_limit) {
    throw ProtocolError(stage_id + " artifact exceeds its decoded size limit");
  }
  if (artifact_bytes_total + size > kMaxArtifactAggregateBytes) {
    throw ProtocolError("mixed output artifacts exceed the aggregate limit");
  }
  artifact_bytes_total += size;

  Value::Map descriptor;
  for (const auto& [key, value] : result.as_map()) {
    if (key != "data") descriptor[key] = value;
  }
  const std::string artifact_id = "art-" + std::to_string(next_artifact++);
  descriptor["artifact_id"] = Value(artifact_id);
  const size_t total =
      (size + kArtifactChunkTargetBytes - 1) / kArtifactChunkTargetBytes;
  descriptor["total"] = Value(static_cast<int64_t>(total));

  auto bytes = std::make_shared<const std::string>(data->as_bin());
  std::vector<UnifiedEvent> events;
  events.reserve(total);
  for (size_t index = 0; index < total; ++index) {
    const size_t offset = index * kArtifactChunkTargetBytes;
    const size_t length = std::min(kArtifactChunkTargetBytes, size - offset);
    Value::Map chunk_descriptor = descriptor;
    chunk_descriptor["index"] = Value(static_cast<int64_t>(index));
    UnifiedEvent event;
    event.request_id = request_id;
    event.session_id = session_id;
    event.stage_id = stage_id;
    event.kind = UnifiedEvent::Kind::kArtifact;
    event.seq = next_seq++;
    event.is_final = true;
    event.artifact = Value(std::move(chunk_descriptor));
    event.artifact_bytes = bytes;
    event.artifact_offset = offset;
    event.artifact_length = length;
    events.push_back(std::move(event));
  }
  return events;
}

}  // namespace umminfer::unified_detail
