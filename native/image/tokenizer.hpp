// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace umminfer::image {

// Qwen2 byte-level BPE tokenizer, runtime slice needed for image prompts.
// Loads the three HuggingFace files: vocab.json, merges.txt,
// added_tokens.json (22 special ids 151643..151664).
class Qwen2BpeTokenizer {
 public:
  explicit Qwen2BpeTokenizer(const std::string& tokenizer_dir);

  // Full pipeline template used by QwenImagePipeline._get_qwen_prompt_embeds.
  static constexpr const char* kPromptTemplate =
      "<|im_start|>system\nDescribe the image by detailing the color, shape, "
      "size, texture, quantity, text, spatial relationships of the objects and "
      "background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n";
  // prompt_template_encode_start_idx: hidden states of the first 34 tokens
  // (the fixed system/user prefix) are discarded by the pipeline.
  static constexpr int kTemplatePrefixTokens = 34;
  static constexpr int kMaxSeqLen = 1024;  // post-drop cap; truncation at 1058 raw

  // Tokenize a raw user prompt: applies the template, splits specials, BPE.
  // Returns raw template token ids (length includes the 34 prefix rows).
  std::vector<int> encode_prompt(const std::string& user_prompt) const;

  // Debug: regex pretokenization of a raw string (no template, no specials).
  std::vector<std::string> debug_pretokens(const std::string& text) const {
    return pretokenize(text);
  }

  const std::unordered_set<std::string>& special_tokens() const { return specials_; }

 private:
  std::vector<int> encode_text(const std::string& text, bool split_specials) const;
  std::vector<int> bpe_word(const std::string& word_utf8) const;
  std::vector<std::string> pretokenize(const std::string& chunk) const;

  std::unordered_map<std::string, int> vocab_;          // unicode-token string -> id
  std::unordered_map<std::string, int> added_ids_;      // special tokens from added_tokens.json
  std::unordered_map<std::string, int> merge_ranks_;    // "A B" -> rank
  std::unordered_map<int, std::string> byte_to_uni_;    // 0..255 -> utf8 char
  std::unordered_set<std::string> specials_;
};

}  // namespace umminfer::image
