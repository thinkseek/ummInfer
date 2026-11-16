// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "cuda_kernels.hpp"

namespace umminfer::native {

// F32 speech codec decoder (speech_tokenizer/model.safetensors, decoder.*
// tensors only). One-shot causal decode over the whole utterance; equivalent
// to the Python chunked_decode because every layer is left-causal.
class CodecDecoder {
 public:
  // weights maps suffix name -> device pointer inside the F32 checkpoint region.
  CodecDecoder(std::unordered_map<std::string, const float*> weights,
               cudaStream_t stream);
  ~CodecDecoder();
  CodecDecoder(const CodecDecoder&) = delete;
  CodecDecoder& operator=(const CodecDecoder&) = delete;

  // codes [frames, 16] row-major, values in [0,2047]. Returns frames*1920
  // mono samples.
  std::vector<float> decode(const std::vector<int>& codes, int frames);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace umminfer::native
