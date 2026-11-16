// SPDX-License-Identifier: Apache-2.0
#include "bpe_tokenizer.hpp"

#include <nlohmann/json.hpp>
#include <unicode/normalizer2.h>
#include <unicode/regex.h>
#include <unicode/unistr.h>

#include <algorithm>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace umminfer::native {
namespace {

// The patched pre-tokenizer regex P (fix_mistral_regex=True on
// transformers 4.57.3). See TTS_NATIVE.md and spec 4.
// Source is UTF-8; plain literals stay char[] so ICU's fromUTF8 accepts them.
constexpr const char* kPretokenRegex =
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
    "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+"
    "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+"
    "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*"
    "|\\p{N}"
    "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*"
    "|\\s*[\\r\\n]+"
    "|\\s+(?!\\S)"
    "|\\s+";

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::invalid_argument("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// GPT-2 bytes_to_unicode, output as the UTF-8 encoding of each mapped
// codepoint. Byte b -> byte_to_utf8_[b].
void build_byte_table(unsigned char out[256][3], int len[256]) {
  std::vector<int> bs;
  for (int c = 33; c <= 126; ++c) bs.push_back(c);
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
  for (size_t i = 0; i < bs.size(); ++i) {
    const int cp = cs[i];
    if (cp < 0x80) {
      out[bs[i]][0] = static_cast<unsigned char>(cp);
      len[bs[i]] = 1;
    } else if (cp < 0x800) {
      out[bs[i]][0] = static_cast<unsigned char>(0xc0 | (cp >> 6));
      out[bs[i]][1] = static_cast<unsigned char>(0x80 | (cp & 0x3f));
      len[bs[i]] = 2;
    } else {
      out[bs[i]][0] = static_cast<unsigned char>(0xe0 | (cp >> 12));
      out[bs[i]][1] = static_cast<unsigned char>(0x80 | ((cp >> 6) & 0x3f));
      out[bs[i]][2] = static_cast<unsigned char>(0x80 | (cp & 0x3f));
      len[bs[i]] = 3;
    }
  }
}

}  // namespace

Qwen2BpeTokenizer::Qwen2BpeTokenizer(const std::string& model_dir) {
  build_byte_table(byte_to_utf8_, byte_to_utf8_len_);

  const nlohmann::json vocab = nlohmann::json::parse(read_file(model_dir + "/vocab.json"));
  vocab_.reserve(vocab.size() * 2);
  for (auto it = vocab.begin(); it != vocab.end(); ++it)
    vocab_.emplace(it.key(), it.value().get<int>());

  // merges.txt: rank = 0-based line index, no header.
  std::istringstream merges(read_file(model_dir + "/merges.txt"));
  std::string line;
  int rank = 0;
  while (std::getline(merges, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    const size_t sp = line.find(' ');
    if (sp == std::string::npos)
      throw std::invalid_argument("bad merge line: " + line);
    merge_rank_.emplace(line.substr(0, sp) + '\x1f' + line.substr(sp + 1), rank++);
  }

  const nlohmann::json cfg =
      nlohmann::json::parse(read_file(model_dir + "/tokenizer_config.json"));
  for (auto it = cfg["added_tokens_decoder"].begin();
       it != cfg["added_tokens_decoder"].end(); ++it) {
    const int id = std::stoi(it.key());
    added_.emplace_back(it.value()["content"].get<std::string>(), id);
  }
  // Longest content wins at every scan position.
  std::sort(added_.begin(), added_.end(), [](const auto& a, const auto& b) {
    return a.first.size() > b.first.size();
  });
  added_fast_.reserve(added_.size());
  for (const auto& [content, id] : added_) added_fast_.emplace(content, id);
}

std::vector<std::string> Qwen2BpeTokenizer::pretokenize(const std::string& text) const {
  UErrorCode st = U_ZERO_ERROR;
  const icu::Normalizer2* nfc = icu::Normalizer2::getNFCInstance(st);
  if (U_FAILURE(st)) throw std::runtime_error("icu NFC unavailable");
  icu::UnicodeString u = icu::UnicodeString::fromUTF8(text);
  u = nfc->normalize(u, st);
  if (U_FAILURE(st)) throw std::runtime_error("icu NFC normalize failed");

  std::unique_ptr<icu::RegexPattern> pat(
      icu::RegexPattern::compile(icu::UnicodeString::fromUTF8(kPretokenRegex), 0, st));
  if (U_FAILURE(st)) throw std::runtime_error("regex compile failed");
  std::unique_ptr<icu::RegexMatcher> m(pat->matcher(u, st));
  if (U_FAILURE(st)) throw std::runtime_error("regex matcher failed");

  std::vector<std::string> chunks;
  int32_t last = 0;
  while (m->find(st)) {
    const int32_t start = m->start(st);
    if (start != last)
      throw std::runtime_error("pretokenizer left a gap; regex does not partition input");
    icu::UnicodeString piece(m->group(st));
    std::string utf8;
    piece.toUTF8String(utf8);
    chunks.push_back(std::move(utf8));
    last = m->end(st);
  }
  if (last != u.length())
    throw std::runtime_error("pretokenizer did not cover the whole input");
  return chunks;
}

std::vector<int> Qwen2BpeTokenizer::bpe_chunk(const std::string& chunk) const {
  std::vector<std::string> sym;
  sym.reserve(chunk.size());
  for (unsigned char b : chunk) {
    sym.emplace_back(reinterpret_cast<const char*>(byte_to_utf8_[b]),
                     byte_to_utf8_len_[b]);
  }
  // ponytail: O(n^2) per-word merge scan; chunks are word-sized, replace with
  // a priority-queue BPE only if profiling shows it matters.
  while (sym.size() >= 2) {
    int best_rank = std::numeric_limits<int>::max();
    size_t best_i = 0;
    bool any = false;
    for (size_t i = 0; i + 1 < sym.size(); ++i) {
      auto it = merge_rank_.find(sym[i] + '\x1f' + sym[i + 1]);
      if (it != merge_rank_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_i = i;
        any = true;
      }
    }
    if (!any) break;
    sym[best_i] += sym[best_i + 1];
    sym.erase(sym.begin() + static_cast<ptrdiff_t>(best_i) + 1);
  }
  std::vector<int> ids;
  ids.reserve(sym.size());
  for (const std::string& s : sym) {
    auto it = vocab_.find(s);
    if (it == vocab_.end())
      throw std::runtime_error("BPE symbol missing from vocab: " + s);
    ids.push_back(it->second);
  }
  return ids;
}

std::vector<int> Qwen2BpeTokenizer::encode(const std::string& text) const {
  std::vector<int> ids;
  size_t pos = 0;
  while (pos < text.size()) {
    // Step 2: longest added-token match at this position.
    bool matched = false;
    for (const auto& [content, id] : added_) {
      if (text.compare(pos, content.size(), content) == 0) {
        ids.push_back(id);
        pos += content.size();
        matched = true;
        break;
      }
    }
    if (matched) continue;

    size_t next = pos + 1;
    while (next < text.size()) {
      bool is_added = false;
      for (const auto& [content, id] : added_) {
        if (text.compare(next, content.size(), content) == 0) {
          is_added = true;
          break;
        }
      }
      if (is_added) break;
      ++next;
    }
    const std::string run = text.substr(pos, next - pos);
    for (const std::string& chunk : pretokenize(run)) {
      auto part = bpe_chunk(chunk);
      ids.insert(ids.end(), part.begin(), part.end());
    }
    pos = next;
  }
  return ids;
}

}  // namespace umminfer::native
