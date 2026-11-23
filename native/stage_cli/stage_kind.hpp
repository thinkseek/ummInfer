// SPDX-License-Identifier: Apache-2.0
// One native engine behind the existing C++ Stage protocol.
//
// ASR and TTS both define umminfer::native::SafeTensors/TensorInfo in their
// accepted trees, so one executable cannot link both. Each kind is therefore
// built as its own executable from this shared main. --kind is validated
// against the compiled kind.
#pragma once

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
  // Requests decoded together. 0 or 1 keeps the single-request executor.
  int batch_size = 0;
};

// The kind this executable was built for: "asr", "omni", "qwen38", "tts",
// "image" or "video".
const char* compiled_kind();

// Loads the model synchronously and returns the function the Stage runs. The
// function receives the per-request abort check and must stop at its next
// boundary by throwing umminfer::RequestAborted.
CancellableComputeFn make_compute(const StageOptions& options);

}  // namespace umminfer::stage
