// SPDX-License-Identifier: Apache-2.0
// One native engine behind the existing C++ Stage protocol.
//
// ASR and TTS both define umminfer::native::SafeTensors/TensorInfo in their
// accepted trees, so one executable cannot link both. Each kind is therefore
// built as its own executable from this shared main. --kind is validated
// against the compiled kind.
#pragma once

#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>

#include "umminfer/pipeline/pipeline.hpp"

namespace umminfer::stage {

struct StageOptions {
  std::string model_dir;
  int gpu = 0;
  // 0 selects the kind's default.
  int max_new_tokens = 0;
  int max_frames = 0;  // TTS and Video
  int max_seq = 0;     // Omni and Qwen3.8
  int steps = 0;       // Image and Video
  std::string image_backend;  // Image only: "" / "2512" (default) or "2.1"
  std::string image_size;     // Image and Video: optional default/gate WxH
  std::string video_backend;  // Video only: "" / "cuda" | "cuda-scaffold" | "contract-stub"
  int video_fps = 0;          // Video only: default fps (0 -> 16)
  int video_max_seconds = 0;  // Video only: max seconds (0 -> 10)
  int video_step_delay_ms = 0;  // Video only: optional per-step delay for abort tests
  std::string ffmpeg_bin;     // Video only: ffmpeg binary path (empty -> "ffmpeg")
  float video_teacache_thresh = 0.0f;  // Video only: opt-in TeaCache threshold in [0.0, 1.0] (0 = off)
  std::string default_ref_audio;  // TTS Base only: optional default 24 kHz WAV
  // Requests decoded together. 0 or 1 keeps the single-request executor.
  int batch_size = 0;
};

// The kind this executable was built for: "asr", "omni", "qwen38", "tts",
// "image" or "video".
const char* compiled_kind();

// Stage-specific CLI hooks. Each *_stage.cpp implements both for its compiled
// kind, so an executable accepts only its own flags. main.cpp parses the
// transport flags.

// The stage's optional-flag fragment for usage(). The fragment is empty or
// ends with a space.
const char* stage_usage_flags();

// Parses one stage-specific option into options.
// Returns true when key is a flag of this stage.
// Returns false when key is not a flag of this stage; the caller reports it.
// Throws std::runtime_error when the value is invalid.
bool parse_stage_flag(const std::string& key, const std::string& value,
                      StageOptions& options);

// Shared integer parsing for stage flags: a non-negative decimal int that
// covers the whole value. Throws std::runtime_error otherwise.
inline int parse_stage_int(const std::string& text, const std::string& option) {
  size_t used = 0;
  int value = 0;
  try {
    value = std::stoi(text, &used);
  } catch (const std::exception&) {
    throw std::runtime_error("invalid " + option + ": " + text);
  }
  if (used != text.size() || value < 0) {
    throw std::runtime_error("invalid " + option + ": " + text);
  }
  return value;
}

// Shared finite floating-point parsing for stage flags bounded to [min_value, max_value].
inline float parse_stage_float_range(const std::string& text,
                                     const std::string& option,
                                     float min_value, float max_value) {
  if (text.empty() || std::isspace(static_cast<unsigned char>(text.front()))) {
    throw std::runtime_error("invalid " + option + ": " + text);
  }
  size_t used = 0;
  float value = 0.0f;
  try {
    value = std::stof(text, &used);
  } catch (const std::exception&) {
    throw std::runtime_error("invalid " + option + ": " + text);
  }
  if (used != text.size() || !std::isfinite(value) || value < min_value ||
      value > max_value) {
    throw std::runtime_error("invalid " + option + ": " + text);
  }
  return value;
}

// Loads the model synchronously and returns the function the Stage runs. The
// function receives the per-request abort check and must stop at its next
// boundary by throwing umminfer::RequestAborted.
CancellableComputeFn make_compute(const StageOptions& options);

}  // namespace umminfer::stage
