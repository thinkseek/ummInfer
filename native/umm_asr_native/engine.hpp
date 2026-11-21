// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "feature_frontend.hpp"

namespace umminfer::native {

// True once the caller's request was aborted. Empty means never abort.
using AbortCheck = std::function<bool()>;

// Raised at a cancellation boundary. The caller settles the request on its
// existing abort path.
class RequestAborted : public std::runtime_error {
 public:
  explicit RequestAborted(std::string message)
      : std::runtime_error(std::move(message)) {}
};

// Loads the real Qwen3-ASR checkpoint (BF16 safetensors) onto one GPU and
// runs greedy transcription. No framework: cuBLAS GEMM plus hand-written
// kernels in cuda_kernels.cu.
class AsrEngine {
 public:
  explicit AsrEngine(const std::string& model_dir, int gpu_id = 0);
  ~AsrEngine();
  AsrEngine(const AsrEngine&) = delete;
  AsrEngine& operator=(const AsrEngine&) = delete;

  // Runs the audio tower + decoder and returns generated token ids. The abort
  // check runs at every audio-tower layer, every prefill layer and every
  // generated token; an aborted request raises RequestAborted at that point.
  // An empty check keeps the original behaviour bit for bit.
  std::vector<int> transcribe_tokens(const MelFeatures& mel, int max_new_tokens = 256,
                                     const AbortCheck& abort = {});

  // Number of audio placeholder tokens for a valid-frame count, matching
  // transformers _get_feat_extract_output_lengths.
  static int audio_token_count(int valid_frames);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace umminfer::native
