// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace umminfer::native {

// Qwen2 byte-level BPE encoder, matching the HuggingFace fast tokenizer as
// loaded with fix_mistral_regex=True on transformers 4.57.3 (the pinned
// qwen-tts runtime). Reads vocab.json, merges.txt and the
// added_tokens_decoder table in tokenizer_config.json. No decode path.
class Qwen2BpeTokenizer {
 public:
  // model_dir must contain vocab.json, merges.txt and tokenizer_config.json.
  explicit Qwen2BpeTokenizer(const std::string& model_dir);

  std::vector<int> encode(const std::string& text) const;

 private:
  std::unordered_map<std::string, int> vocab_;           // BPE symbol -> id
  std::unordered_map<std::string, int> merge_rank_;     // "a\x1fb" -> rank
  std::vector<std::pair<std::string, int>> added_;      // content -> id, long first
  std::unordered_map<std::string, int> added_fast_;
  unsigned char byte_to_utf8_[256][3];                  // GPT-2 symbol table
  int byte_to_utf8_len_[256];

  std::vector<std::string> pretokenize(const std::string& text) const;
  std::vector<int> bpe_chunk(const std::string& chunk) const;
};

}  // namespace umminfer::native
