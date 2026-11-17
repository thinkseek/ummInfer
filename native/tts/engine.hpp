// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <functional>
#include <memory>
#include <vector>
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
  // is the time to the first generated codec frame; first_audio is the time to
  // the first emitted sample when a stream sink is installed.
  struct SynthTimings {
    double first_frame_ms = -1;
    double first_audio_ms = -1;
    double total_ms = -1;
  };

  // Receives the newly decoded samples at a codec-frame boundary. first_frame
  // is the index of the first codec frame in this chunk; the concatenation of
  // every chunk equals the vector synthesize returns, bit for bit.
  using StreamSink = std::function<void(const std::vector<float>& samples,
                                        int first_frame, int frames)>;

  // text_ids is the full wrapped prompt; language/speaker are codec ids.
  // Greedy synthesis returning 24 kHz mono float samples in [-1, 1]. The abort
  // check runs at every frame boundary; an aborted request raises
  // RequestAborted. An empty check keeps the original behaviour bit for bit.
  // With a stream sink, each chunk decodes the frame prefix produced so far;
  // the decoder is causal, so a prefix decode equals the head of the full one.
  std::vector<float> synthesize(const std::vector<int>& text_ids, int language_id,
                                int speaker_id, int max_frames,
                                const AbortCheck& abort = {},
                                SynthTimings* timings = nullptr,
                                const StreamSink& on_chunk = {});

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace umminfer::native
