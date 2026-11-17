// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "abort.hpp"

namespace umminfer::native {

// Loads the two Qwen3-TTS checkpoints onto one GPU and runs greedy
// CustomVoice synthesis to 24 kHz float samples. No framework: hand-written
// CUDA kernels only; talker in BF16 with FP32 norm/softmax, codec in FP32.
class TtsEngine {
 public:
  explicit TtsEngine(const std::string& model_dir, int gpu_id = 0);
  ~TtsEngine();
  TtsEngine(const TtsEngine&) = delete;
  TtsEngine& operator=(const TtsEngine&) = delete;

  // Optional phase times, measured by the engine in milliseconds. first_frame
  // is the time to the first generated codec frame; the decoder is
  // non-streaming, so it is not a time to first audio sample.
  struct SynthTimings {
    double first_frame_ms = -1;
    double total_ms = -1;
  };

  // text_ids is the full wrapped prompt; language/speaker are codec ids.
  // Greedy synthesis returning 24 kHz mono float samples in [-1, 1]. The abort
  // check runs at every frame boundary; an aborted request raises
  // RequestAborted. An empty check keeps the original behaviour bit for bit.
  std::vector<float> synthesize(const std::vector<int>& text_ids, int language_id,
                                int speaker_id, int max_frames,
                                const AbortCheck& abort = {},
                                SynthTimings* timings = nullptr);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace umminfer::native
