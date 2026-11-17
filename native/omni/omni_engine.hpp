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
// Receives each generated token id before the next forward pass.
using TokenCallback = std::function<void(int)>;

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
    // prompt_tokens is the whole prompt; prefill_tokens is what this call
    // actually computed; reused_tokens came from the KV cache of the previous
    // request in the same session.
    int prefill_tokens = 0;
    int reused_tokens = 0;
    int image_tokens = 0;
    // Wall-clock phases (host timed, every phase boundary is already a stream
    // synchronisation point).  prefill covers the text decoder only; vision is
    // the image tower ahead of it.
    double vision_ms = 0.0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
  };

  // Full prompt + optional one image. When image is null the run is pure text.
  // The abort check runs at every vision block, every prefill layer and every
  // generated token; an aborted request raises RequestAborted. An empty check
  // keeps the original behaviour bit for bit.
  //
  // prefix_tokens > 0 reuses the KV cache written by the previous call on this
  // engine: the first prefix_tokens of prompt_ids must be identical to that
  // call's prompt prefix, and no image may be involved. The cached region was
  // produced by prefill, so a reused run is bit-identical to a full prefill.
  Result generate(const std::vector<int>& prompt_ids,
                  const PreprocessedImage* image, int max_new_tokens,
                  const AbortCheck& abort = {},
                  const TokenCallback& on_token = {},
                  int prefix_tokens = 0);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace umminfer::omni
