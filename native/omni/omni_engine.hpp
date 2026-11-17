// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "image_preprocess.hpp"

namespace umminfer::omni {

// True once the caller's request was aborted. Empty means never abort.
using AbortCheck = std::function<bool()>;

// Raised at a cancellation boundary. The caller settles the request on its
// existing abort path.
class RequestAborted : public std::runtime_error {
 public:
  explicit RequestAborted(std::string message)
      : std::runtime_error(std::move(message)) {}
};

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
  // The abort check runs at every vision block, every prefill layer and every
  // generated token; an aborted request raises RequestAborted. An empty check
  // keeps the original behaviour bit for bit.
  Result generate(const std::vector<int>& prompt_ids,
                  const PreprocessedImage* image, int max_new_tokens,
                  const AbortCheck& abort = {});

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace umminfer::omni
