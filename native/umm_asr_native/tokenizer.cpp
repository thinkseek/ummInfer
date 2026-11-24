// SPDX-License-Identifier: Apache-2.0
#include "tokenizer.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace umminfer::native {
namespace {

// GPT-2 byte-level BPE printable-unicode mapping (transformers
// tokenizers added_tokens / bytes_to_unicode).
std::unordered_map<uint32_t, unsigned char> build_unicode_to_byte() {
  std::vector<int> bs;
  for (int c = '!'; c <= '~'; ++c) bs.push_back(c);
  for (int c = 0xa1; c <= 0xac; ++c) bs.push_back(c);
  for (int c = 0xae; c <= 0xff; ++c) bs.push_back(c);
  std::vector<int> cs = bs;
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
      bs.push_back(b);
      cs.push_back(256 + n);
      ++n;
    }
  }
  std::unordered_map<uint32_t, unsigned char> out;
  for (size_t i = 0; i < bs.size(); ++i)
    out.emplace(static_cast<uint32_t>(cs[i]), static_cast<unsigned char>(bs[i]));
  return out;
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::invalid_argument("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

}  // namespace

QwenTokenizer::QwenTokenizer(const std::string& vocab_json) {
  unicode_to_byte_ = build_unicode_to_byte();
  const nlohmann::json vocab = nlohmann::json::parse(read_file(vocab_json));
  int max_id = 0;
  for (auto it = vocab.begin(); it != vocab.end(); ++it)
    max_id = std::max(max_id, it.value().get<int>());
  id_to_token_.resize(max_id + 1);
  for (auto it = vocab.begin(); it != vocab.end(); ++it)
    id_to_token_[it.value().get<int>()] = it.key();
}

std::vector<int> QwenTokenizer::build_asr_prompt(int audio_token_count) {
  return build_asr_prompt(audio_token_count, kAudioStart, kAudioPad, kAudioEnd);
}

std::vector<int> QwenTokenizer::build_asr_prompt(int audio_token_count, int audio_start,
                                                 int audio_pad, int audio_end) {
  if (audio_token_count <= 0)
    throw std::invalid_argument("audio token count must be positive");
  std::vector<int> ids;
  ids.reserve(11 + audio_token_count);
  ids.insert(ids.end(), {kImStart, kUserToken, kNewline, audio_start});
  ids.insert(ids.end(), audio_token_count, audio_pad);
  ids.insert(ids.end(),
             {audio_end, kImEnd, kNewline, kImStart, kAssistantToken, kNewline});
  return ids;
}

bool QwenTokenizer::is_special(int id) {
  // Every Qwen2 added token (62 entries) sits at or above 151643.
  return id >= kEndOfText;
}

std::string QwenTokenizer::decode(const std::vector<int>& ids) const {
  std::string bytes;
  for (int id : ids) {
    if (id < 0 || id >= static_cast<int>(id_to_token_.size()) || is_special(id)) continue;
    const std::string& tok = id_to_token_[id];
    // UTF-8 walk: each codepoint maps back to one original byte.
    size_t i = 0;
    while (i < tok.size()) {
      uint32_t cp = 0;
      int len = 0;
      unsigned char lead = static_cast<unsigned char>(tok[i]);
      if (lead < 0x80) { cp = lead; len = 1; }
      else if ((lead & 0xe0) == 0xc0) { cp = lead & 0x1f; len = 2; }
      else if ((lead & 0xf0) == 0xe0) { cp = lead & 0x0f; len = 3; }
      else { cp = lead & 0x07; len = 4; }
      for (int j = 1; j < len && i + j < tok.size(); ++j)
        cp = (cp << 6) | (static_cast<unsigned char>(tok[i + j]) & 0x3f);
      auto it = unicode_to_byte_.find(cp);
      if (it != unicode_to_byte_.end()) bytes.push_back(static_cast<char>(it->second));
      i += len;
    }
  }
  return bytes;
}

}  // namespace umminfer::native
