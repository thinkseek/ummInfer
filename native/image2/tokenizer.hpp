// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace umminfer::image2 {

// Qwen3 byte-level BPE tokenizer, runtime slice needed for image prompts.
// Loads vocab.json, merges.txt and added_tokens.json from the processor dir.
// Qwen3 uses the same byte-level BPE and special-token table as Qwen2.
class Qwen2BpeTokenizer {
 public:
  explicit Qwen2BpeTokenizer(const std::string& tokenizer_dir);

  // Raw template used by QwenImage21Pipeline for text-to-image (no condition
  // image). The system block is tokenized on its own to derive the drop count.
  static const char* kSystemPrompt();
  static std::string prompt_template();

  // Tokenize a raw user prompt: applies the template, splits specials, BPE.
  // No truncation or padding for a single sequence.
  std::vector<int> encode_prompt(const std::string& user_prompt) const;

  // Number of leading tokens the pipeline drops (the system block, including
  // its trailing newline), derived from tokenizing that block.
  int system_prefix_tokens() const;

  // Debug: regex pretokenization of a raw string (no template, no specials).
  std::vector<std::string> debug_pretokens(const std::string& text) const {
    return pretokenize(text);
  }

  const std::unordered_set<std::string>& special_tokens() const { return specials_; }

 private:
  std::vector<int> encode_text(const std::string& text) const;
  std::vector<int> encode_with_specials(const std::string& text) const;
  std::vector<int> bpe_word(const std::string& word_utf8) const;
  std::vector<std::string> pretokenize(const std::string& chunk) const;

  std::unordered_map<std::string, int> vocab_;        // unicode-token string -> id
  std::unordered_map<std::string, int> added_ids_;    // special tokens
  std::unordered_map<std::string, int> merge_ranks_;  // "A B" -> rank
  std::vector<std::string> byte_to_uni_;              // 0..255 -> utf8 char
  std::unordered_set<std::string> specials_;
};

}  // namespace umminfer::image2
