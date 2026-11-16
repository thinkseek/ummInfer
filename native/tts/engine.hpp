// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <string>
#include <vector>

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

  // text_ids is the full wrapped prompt; language/speaker are codec ids.
  // Greedy synthesis returning 24 kHz mono float samples in [-1, 1].
  std::vector<float> synthesize(const std::vector<int>& text_ids, int language_id,
                                int speaker_id, int max_frames);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace umminfer::native
