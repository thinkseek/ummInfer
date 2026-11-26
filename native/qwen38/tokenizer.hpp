// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace umminfer::qwen38 {

struct Qwen2BpeTokenizerSource {
  std::string vocab_json;
  std::string merges_txt;
  std::string tokenizer_config_json;

  // Reads each file once in binary mode. Constructing from this object parses
  // these exact buffers, allowing the stage to bind a digest without a second
  // filesystem read or a time-of-check/time-of-use window.
  static Qwen2BpeTokenizerSource from_files(
      const std::string& vocab_json_path, const std::string& merges_txt_path,
      const std::string& tokenizer_config_json_path);
};

// Qwen2 byte-level BPE tokenizer, encode + decode slice.
// Vocabulary and merges come from the checkpoint's vocab.json / merges.txt.
class Qwen2BpeTokenizer {
 public:
  Qwen2BpeTokenizer(const std::string& vocab_json, const std::string& merges_txt,
                    const std::string& tokenizer_config_json);
  explicit Qwen2BpeTokenizer(Qwen2BpeTokenizerSource source);

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
