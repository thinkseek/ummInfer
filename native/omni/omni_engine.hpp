// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "image_preprocess.hpp"

namespace umminfer::omni {

// Loads the real Qwen3-Omni thinker checkpoint (BF16, 15 shards) onto one GPU
// and runs greedy image+text generation. No Python/torch/SGLang at runtime:
// hand-written CUDA kernels only. Audio/video inputs are unsupported.
class OmniEngine {
 public:
  OmniEngine(const std::string& model_dir, int gpu_id, int max_seq = 2048);
  ~OmniEngine();
  OmniEngine(const OmniEngine&) = delete;
  OmniEngine& operator=(const OmniEngine&) = delete;

  struct Result {
    std::vector<int> token_ids;          // generated ids, excluding EOS
    std::string text;                    // decoded, specials skipped
    int prefill_tokens = 0;
    int image_tokens = 0;
  };

  // Full prompt + optional one image. When image is null the run is pure text.
  Result generate(const std::vector<int>& prompt_ids,
                  const PreprocessedImage* image, int max_new_tokens);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace umminfer::omni
