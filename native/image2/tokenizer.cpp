// SPDX-License-Identifier: Apache-2.0
#include "tokenizer.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace umminfer::image2 {
namespace {

// ---- UTF-8 + Unicode predicates ------------------------------------------
// Hand-rolled category ranges for the scripts our prompts use (ASCII, Latin
// extensions, CJK, kana, hangul, common digits/whitespace). Parity with the
// Python tokenizer is checked by image2_tests against reference ids.

int decode_utf8(const std::string& s, size_t& i, uint32_t& cp) {
  const uint8_t u = static_cast<uint8_t>(s[i]);
  if (u < 0x80) { cp = u; i += 1; return 1; }
  int len = u >= 0xF0 ? 4 : u >= 0xE0 ? 3 : 2;
  if (i + len > s.size()) { cp = u; i += 1; return 1; }
  cp = u & (len == 2 ? 0x1f : len == 3 ? 0x0f : 0x07);
  for (int k = 1; k < len; ++k) cp = (cp << 6) | (static_cast<uint8_t>(s[i + k]) & 0x3f);
  i += len;
  return len;
}

void encode_utf8(uint32_t cp, std::string& out) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  } else {
    out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  }
}

bool is_letter(uint32_t cp) {
  return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z') ||
         (cp >= 0x00C0 && cp <= 0x024F) || (cp >= 0x1D00 && cp <= 0x1D7F) ||
         (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
         (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x3040 && cp <= 0x309F) ||
         (cp >= 0x30A0 && cp <= 0x30FF) || (cp >= 0xAC00 && cp <= 0xD7AF) ||
         (cp >= 0xFF21 && cp <= 0xFF3A) || (cp >= 0xFF41 && cp <= 0xFF5A);
}

bool is_number(uint32_t cp) {
  return (cp >= '0' && cp <= '9') || cp == 0x00B2 || cp == 0x00B3 || cp == 0x00B9 ||
         (cp >= 0x00BC && cp <= 0x00BE) || (cp >= 0x0660 && cp <= 0x0669) ||
         (cp >= 0xFF10 && cp <= 0xFF19);
}

bool is_ws(uint32_t cp) {
  return cp == ' ' || cp == '\t' || cp == '\n' || cp == 0x0b || cp == '\f' || cp == '\r' ||
         cp == 0x0085 || cp == 0x00A0 || (cp >= 0x2000 && cp <= 0x200A) ||
         cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000 ||
         cp == 0xFEFF;
}

// PRETOKENIZE_REGEX, scanned left-to-right with the same alternation order and
// greediness as Python re.findall (first matching alternative wins).
struct RegexScanner {
  const std::string& s;
  size_t pos = 0;

  bool next(std::string& tok) {
    if (pos >= s.size()) return false;
    const size_t start = pos;
    uint32_t cp = 0;
    size_t after = pos;
    decode_utf8(s, after, cp);
    // Alternative 1: 's|'t|'re|'ve|'m|'ll|'d (case-insensitive).
    if (cp == '\'') {
      static const char* contracts[] = {"s", "t", "re", "ve", "m", "ll", "d"};
      for (const char* c : contracts) {
        const size_t L = std::char_traits<char>::length(c);
        if (pos + 1 + L <= s.size()) {
          bool match = true;
          for (size_t k = 0; k < L; ++k) {
            const char a = static_cast<char>(std::tolower(static_cast<unsigned char>(s[pos + 1 + k])));
            if (a != c[k]) { match = false; break; }
          }
          if (match) { pos += 1 + L; tok = s.substr(start, pos - start); return true; }
        }
      }
    }
    // Alternative 2: [^\r\n L N]? L+  (the optional prefix may be a space).
    if (is_letter(cp)) {
      size_t r = pos;
      while (r < s.size()) {
        size_t nx = r;
        uint32_t cn = 0;
        decode_utf8(s, nx, cn);
        if (!is_letter(cn)) break;
        r = nx;
      }
      pos = r;
      tok = s.substr(start, pos - start);
      return true;
    }
    if (cp != '\r' && cp != '\n' && !is_letter(cp) && !is_number(cp)) {
      size_t q = after;
      uint32_t next_cp = 0;
      if (q < s.size()) decode_utf8(s, q, next_cp);
      if (is_letter(next_cp)) {
        size_t r = q;
        while (r < s.size()) {
          size_t nx = r;
          uint32_t cn = 0;
          decode_utf8(s, nx, cn);
          if (!is_letter(cn)) break;
          r = nx;
        }
        pos = r;
        tok = s.substr(start, pos - start);
        return true;
      }
    }
    // Alternative 3: N (single number codepoint).
    if (is_number(cp)) { pos = after; tok = s.substr(start, after - start); return true; }
    // Alternative 4: " ?[^\s L N]+[\r\n]*" — literal space prefix only.
    bool alt4 = !is_ws(cp) && !is_letter(cp) && !is_number(cp);
    size_t pcur = pos;
    if (cp == ' ') {
      size_t q = after;
      uint32_t next_cp = 0;
      if (q < s.size()) decode_utf8(s, q, next_cp);
      if (!is_ws(next_cp) && !is_letter(next_cp) && !is_number(next_cp)) {
        pcur = after;
        alt4 = true;
      }
    }
    if (alt4) {
      size_t p = pcur;
      while (p < s.size()) {
        size_t nx = p;
        uint32_t cn = 0;
        decode_utf8(s, nx, cn);
        if (is_ws(cn) || is_letter(cn) || is_number(cn)) break;
        p = nx;
      }
      while (p < s.size() && (s[p] == '\r' || s[p] == '\n')) ++p;
      pos = p;
      tok = s.substr(start, pos - start);
      return true;
    }
    // Whitespace alternatives 5-7.
    if (is_ws(cp)) {
      size_t p = pos;
      bool has_newline = false;
      int ws_codepoints = 0;
      while (p < s.size()) {
        size_t nx = p;
        uint32_t cn = 0;
        decode_utf8(s, nx, cn);
        if (!is_ws(cn)) break;
        if (cn == '\r' || cn == '\n') has_newline = true;
        ++ws_codepoints;
        p = nx;
      }
      if (has_newline) {
        size_t cut = pos;
        size_t q2 = pos;
        while (q2 < p) {
          size_t nx = q2;
          uint32_t cn = 0;
          decode_utf8(s, nx, cn);
          if (cn == '\r' || cn == '\n') cut = nx;
          q2 = nx;
        }
        pos = cut;
        tok = s.substr(start, pos - start);
        return true;
      }
      if (p >= s.size() || ws_codepoints == 1) {
        pos = p;
      } else {
        size_t q2 = pos;
        for (int k = 0; k < ws_codepoints - 1; ++k) {
          size_t nx = q2;
          uint32_t cn = 0;
          decode_utf8(s, nx, cn);
          q2 = nx;
        }
        pos = q2;
      }
      tok = s.substr(start, pos - start);
      return true;
    }
    pos = after;
    tok = s.substr(start, after - start);
    return true;
  }
};

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::invalid_argument("cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

}  // namespace

const char* Qwen2BpeTokenizer::kSystemPrompt() {
  return "Comprehend and analyze the provided prompt.";
}

std::string Qwen2BpeTokenizer::prompt_template() {
  return std::string("<|im_start|>system\n") + kSystemPrompt() +
         "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n";
}

Qwen2BpeTokenizer::Qwen2BpeTokenizer(const std::string& dir) {
  const nlohmann::json vocab = nlohmann::json::parse(read_file(dir + "/vocab.json"));
  for (auto it = vocab.begin(); it != vocab.end(); ++it)
    vocab_.emplace(it.key(), it.value().get<int>());

  std::ifstream merges(dir + "/merges.txt");
  if (!merges) throw std::invalid_argument("cannot open merges.txt");
  std::string line;
  int rank = 0;
  while (std::getline(merges, line)) {
    if (line.empty() || line[0] == '#') continue;
    merge_ranks_.emplace(line, rank++);
  }

  const nlohmann::json added = nlohmann::json::parse(read_file(dir + "/added_tokens.json"));
  for (auto it = added.begin(); it != added.end(); ++it) {
    specials_.insert(it.key());
    added_ids_.emplace(it.key(), it.value().get<int>());
  }

  // GPT-2 bytes_to_unicode.
  byte_to_uni_.resize(256);
  for (int b = 0; b < 256; ++b) {
    bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
    uint32_t cp = static_cast<uint32_t>(b);
    if (!printable) {
      int n = 0;
      for (int x = 0; x < b; ++x) {
        bool px = (x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174 && x <= 255);
        if (!px) ++n;
      }
      cp = 256u + n;
    }
    encode_utf8(cp, byte_to_uni_[b]);
  }
}

std::vector<std::string> Qwen2BpeTokenizer::pretokenize(const std::string& chunk) const {
  std::vector<std::string> words;
  RegexScanner scanner{chunk};
  std::string w;
  while (scanner.next(w)) words.push_back(w);
  return words;
}

std::vector<int> Qwen2BpeTokenizer::bpe_word(const std::string& word_utf8) const {
  std::string mapped;
  for (unsigned char b : word_utf8) mapped += byte_to_uni_.at(b);
  std::vector<std::string> syms;
  size_t p = 0;
  while (p < mapped.size()) {
    size_t q = p;
    uint32_t cp = 0;
    decode_utf8(mapped, q, cp);
    syms.push_back(mapped.substr(p, q - p));
    p = q;
  }
  while (syms.size() > 1) {
    int best_rank = INT32_MAX;
    size_t best_i = 0;
    bool found = false;
    for (size_t i = 0; i + 1 < syms.size(); ++i) {
      auto it = merge_ranks_.find(syms[i] + " " + syms[i + 1]);
      if (it != merge_ranks_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_i = i;
        found = true;
      }
    }
    if (!found) break;
    syms[best_i] += syms[best_i + 1];
    syms.erase(syms.begin() + static_cast<long>(best_i) + 1);
  }
  std::vector<int> ids;
  for (const auto& sym : syms) {
    auto it = vocab_.find(sym);
    if (it == vocab_.end()) throw std::runtime_error("BPE token missing from vocab: " + sym);
    ids.push_back(it->second);
  }
  return ids;
}

std::vector<int> Qwen2BpeTokenizer::encode_text(const std::string& text) const {
  std::vector<int> ids;
  for (const auto& word : pretokenize(text)) {
    auto w = bpe_word(word);
    ids.insert(ids.end(), w.begin(), w.end());
  }
  return ids;
}

std::vector<int> Qwen2BpeTokenizer::encode_with_specials(const std::string& text) const {
  std::vector<int> ids;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t best = std::string::npos;
    const std::string* best_sp = nullptr;
    for (const auto& sp : specials_) {
      const size_t hit = text.find(sp, pos);
      if (hit != std::string::npos &&
          (best == std::string::npos || hit < best ||
           (hit == best && sp.size() > best_sp->size()))) {
        best = hit;
        best_sp = &sp;
      }
    }
    if (best == std::string::npos) {
      auto rest = encode_text(text.substr(pos));
      ids.insert(ids.end(), rest.begin(), rest.end());
      break;
    }
    if (best > pos) {
      auto mid = encode_text(text.substr(pos, best - pos));
      ids.insert(ids.end(), mid.begin(), mid.end());
    }
    ids.push_back(added_ids_.at(*best_sp));
    pos = best + best_sp->size();
  }
  return ids;
}

std::vector<int> Qwen2BpeTokenizer::encode_prompt(const std::string& user_prompt) const {
  std::string templated = prompt_template();
  const size_t ph = templated.find("{}");
  if (ph == std::string::npos) throw std::runtime_error("template placeholder missing");
  templated.replace(ph, 2, user_prompt);
  return encode_with_specials(templated);
}

int Qwen2BpeTokenizer::system_prefix_tokens() const {
  // apply_chat_template renders the system message as
  // "<|im_start|>system\n<content><|im_end|>\n".
  return static_cast<int>(encode_with_specials(
      std::string("<|im_start|>system\n") + kSystemPrompt() + "<|im_end|>\n").size());
}

}  // namespace umminfer::image2
