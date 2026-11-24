// SPDX-License-Identifier: Apache-2.0
// HuggingFace Unigram/Metaspace tokenizer used by Wan2.2's UMT5-XXL encoder.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace umminfer::video {

struct Umt5TokenBatch {
  std::vector<int32_t> ids;
  std::vector<int32_t> attention_mask;
};

class Umt5Tokenizer {
 public:
  explicit Umt5Tokenizer(const std::filesystem::path& tokenizer_json);

  // Applies Wan prompt cleaning, appends </s>, truncates to max_length, and
  // optionally pads with id 0. The mask is one for every non-padding token.
  Umt5TokenBatch encode(const std::string& text,
                        size_t max_length = 512,
                        bool pad_to_max = false) const;

 private:
  struct AddedToken {
    std::string text;
    int32_t id;
  };

  std::vector<int32_t> encode_unigram(const std::string& input) const;

  std::unordered_map<std::string, int32_t> vocab_;
  std::vector<std::string> pieces_;
  std::vector<double> scores_;
  std::vector<AddedToken> added_tokens_;
  int32_t unk_id_ = 2;
  double unk_score_ = 0.0;
  bool byte_fallback_ = false;
  size_t max_piece_bytes_ = 0;
};

}  // namespace umminfer::video
