// SPDX-License-Identifier: Apache-2.0
// Real-weight BF16 UMT5-XXL bidirectional text encoder.
#pragma once

#include "umt5_tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace umminfer::video {

// relative_position is key_position - query_position. The defaults are UMT5-XXL.
int umt5_relative_position_bucket(int relative_position,
                                  int num_buckets = 32,
                                  int max_distance = 128);

struct Umt5EncoderOutput {
  std::vector<int64_t> shape;
  std::vector<float> hidden_states;
  double encode_ms = 0.0;
  size_t peak_native_device_bytes = 0;
};

class Umt5Encoder {
 public:
  Umt5Encoder(const std::filesystem::path& text_encoder_dir, int gpu);
  ~Umt5Encoder();

  Umt5Encoder(const Umt5Encoder&) = delete;
  Umt5Encoder& operator=(const Umt5Encoder&) = delete;

  Umt5EncoderOutput encode(const Umt5TokenBatch& tokens);
  double load_ms() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace umminfer::video
