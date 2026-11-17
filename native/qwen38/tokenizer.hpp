// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace umminfer::qwen38 {

// Qwen2 byte-level BPE tokenizer, encode + decode slice.
// Vocabulary and merges come from the checkpoint's vocab.json / merges.txt.
class Qwen2BpeTokenizer {
 public:
  Qwen2BpeTokenizer(const std::string& vocab_json, const std::string& merges_txt,
                    const std::string& tokenizer_config_json);

  // Token ids come from tokenizer_config.json, so this class works for any
  // checkpoint that shares the Qwen2 byte-level BPE vocabulary.
  int id_of(const std::string& token) const;

  // Full text -> ids. Special tokens (<|...|>) are recognized and protected.
  std::vector<int> encode(const std::string& text) const;
  // Inverse byte-level decode; special ids are skipped.
  std::string decode(const std::vector<int>& ids) const;

  bool is_eos(int id) const;

 private:
  std::unordered_map<std::string, int> token_to_id_;
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int> merge_rank_;
  std::unordered_map<std::string, int> specials_;  // literal -> id
  std::unordered_map<uint32_t, unsigned char> unicode_to_byte_;
  std::unordered_map<unsigned char, uint32_t> byte_to_unicode_;
};

}  // namespace umminfer::qwen38
