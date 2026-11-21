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
  int max_frames = 0;  // TTS only
  int max_seq = 0;     // Omni and Qwen3.8
  int steps = 0;       // Image only
  std::string image_backend;  // Image only: "" / "2512" (default) or "2.1"
  std::string image_size;     // Image only: optional startup WxH gate
  // Requests decoded together. 0 or 1 keeps the single-request executor.
  int batch_size = 0;
};

// The kind this executable was built for: "asr", "omni", "qwen38", "tts" or
// "image".
const char* compiled_kind();

// Loads the model synchronously and returns the function the Stage runs. The
// function receives the per-request abort check and must stop at its next
// boundary by throwing umminfer::RequestAborted.
CancellableComputeFn make_compute(const StageOptions& options);

}  // namespace umminfer::stage
