// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "abort.hpp"
#include "cuda_kernels.hpp"

namespace umminfer::native {

// F32 speech codec decoder (speech_tokenizer/model.safetensors, decoder.*
// tensors only). Every layer is strictly causal, so stateful chunked decoding
// produces exact O(T) streaming output matching one-shot decoding with bounded
// scratch VRAM.
class CodecDecoder {
 public:
  // weights maps suffix name -> device pointer inside the F32 checkpoint region.
  CodecDecoder(std::unordered_map<std::string, const float*> weights,
               cudaStream_t stream);
  ~CodecDecoder();
  CodecDecoder(const CodecDecoder&) = delete;
  CodecDecoder& operator=(const CodecDecoder&) = delete;

  // Reset streaming causal state (conv/deconv history and pre-transformer
  // rolling KV cache) before a new utterance.
  void reset_stream();

  // Incrementally decode `frames` newly generated codec frames (`codes` is
  // [frames, 16] row-major, values in [0, 2047]) continuing from the current
  // causal state. Returns `frames * 1920` mono 24 kHz samples.
  std::vector<float> decode_chunk(const int* codes, int frames,
                                  const AbortCheck& abort = {});

  // codes [frames, 16] row-major, values in [0,2047]. Returns frames*1920
  // mono samples. Resets stream state and decodes in bounded chunks.
  std::vector<float> decode(const std::vector<int>& codes, int frames,
                            const AbortCheck& abort = {});

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace umminfer::native
