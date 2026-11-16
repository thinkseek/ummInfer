// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
namespace umminfer::native {

// Qwen2 BPE tokenizer, runtime slice:
//  - encoding is the fixed ASR prompt whose ids are built from special tokens;
//    general text encoding is not needed for this vertical slice.
//  - decoding uses the full vocab plus the GPT-2 byte-level inverse map.
// vocab.json maps token strings to ids (151936 entries for this checkpoint).
class QwenTokenizer {
 public:
  explicit QwenTokenizer(const std::string& vocab_json);

  static constexpr int kEndOfText = 151643;
  static constexpr int kImStart = 151644;
  static constexpr int kImEnd = 151645;
  static constexpr int kAudioStart = 151669;
  static constexpr int kAudioPad = 151676;
  static constexpr int kAudioEnd = 151670;
  // First generated token prefixing the language name ("language").
  static constexpr int kLanguage = 11528;
  static constexpr int kUserToken = 872;
  static constexpr int kAssistantToken = 77091;
  static constexpr int kNewline = 198;

  // <|im_start|>user\n <|audio_start|> audio_pad*N <|audio_end|>
  // <|im_end|>\n<|im_start|>assistant\n
  static std::vector<int> build_asr_prompt(int audio_token_count);

  static bool is_eos(int id) { return id == kEndOfText || id == kImEnd; }
  static bool is_special(int id);

  // Inverse byte-level BPE decode. Special tokens are skipped.
  std::string decode(const std::vector<int>& ids) const;

 private:
  std::vector<std::string> id_to_token_;  // indexed by token id
  std::unordered_map<uint32_t, unsigned char> unicode_to_byte_;
};

}  // namespace umminfer::native
