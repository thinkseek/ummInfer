// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace umminfer::omni {

// Qwen2 byte-level BPE tokenizer, encode + decode slice.
// Vocabulary and merges come from the checkpoint's vocab.json / merges.txt.
class Qwen2BpeTokenizer {
 public:
  Qwen2BpeTokenizer(const std::string& vocab_json, const std::string& merges_txt,
                    const std::string& tokenizer_config_json);

  static constexpr int kImStart = 151644;
  static constexpr int kImEnd = 151645;
  static constexpr int kEndOfText = 151643;
  static constexpr int kVisionStart = 151652;
  static constexpr int kVisionEnd = 151653;
  static constexpr int kImagePad = 151655;
  static constexpr int kUserToken = 872;
  static constexpr int kAssistantToken = 77091;
  static constexpr int kSystemToken = 8948;
  static constexpr int kNewline = 198;

  // Full text -> ids. Special tokens (<|...|>) are recognized and protected.
  std::vector<int> encode(const std::string& text) const;
  // Inverse byte-level decode; special ids are skipped.
  std::string decode(const std::vector<int>& ids) const;

  bool is_eos(int id) const { return id == kImEnd || id == kEndOfText; }

 private:
  std::unordered_map<std::string, int> token_to_id_;
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int> merge_rank_;
  std::unordered_map<std::string, int> specials_;  // literal -> id
  std::unordered_map<uint32_t, unsigned char> unicode_to_byte_;
  std::unordered_map<unsigned char, uint32_t> byte_to_unicode_;
};

}  // namespace umminfer::omni
