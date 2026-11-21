// SPDX-License-Identifier: Apache-2.0
#include "tokenizer.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>

#include "unicode_ranges.hpp"

namespace umminfer::omni {
namespace {

std::string read_text(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::invalid_argument("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

bool in_table(uint32_t cp, const uint32_t (&table)[][2], size_t n) {
  size_t lo = 0, hi = n;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if (cp < table[mid][0])
      hi = mid;
    else if (cp > table[mid][1])
      lo = mid + 1;
    else
      return true;
  }
  return false;
}

bool is_letter(uint32_t cp) {
  return in_table(cp, kLetterRanges, sizeof(kLetterRanges) / sizeof(kLetterRanges[0]));
}
bool is_number(uint32_t cp) {
  return in_table(cp, kNumberRanges, sizeof(kNumberRanges) / sizeof(kNumberRanges[0]));
}
bool is_space(uint32_t cp) {
  return in_table(cp, kSpaceRanges, sizeof(kSpaceRanges) / sizeof(kSpaceRanges[0]));
}
bool is_crlf(uint32_t cp) { return cp == '\r' || cp == '\n'; }

// Decode one UTF-8 codepoint at s[i]; advances i. U+FFFD on malformed input.
uint32_t next_cp(const std::string& s, size_t* i) {
  const unsigned char c = static_cast<unsigned char>(s[*i]);
  uint32_t cp = 0;
  int len = 1;
  if (c < 0x80) {
    cp = c;
  } else if ((c & 0xe0) == 0xc0) {
    cp = c & 0x1f;
    len = 2;
  } else if ((c & 0xf0) == 0xe0) {
    cp = c & 0x0f;
    len = 3;
  } else if ((c & 0xf0) == 0xf0) {
    cp = c & 0x07;
    len = 4;
  } else {
    ++*i;
    return 0xFFFD;
  }
  for (int j = 1; j < len && *i + j < s.size(); ++j)
    cp = (cp << 6) | (static_cast<unsigned char>(s[*i + j]) & 0x3f);
  *i += len;
  return cp;
}

void append_cp(std::string& out, uint32_t cp) {
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

bool ci2(const std::string& s, size_t i, const char* lit) {
  if (i + 2 > s.size()) return false;
  return std::tolower(static_cast<unsigned char>(s[i])) == lit[0] &&
         std::tolower(static_cast<unsigned char>(s[i + 1])) == lit[1];
}

// Leftmost-first alternation of PRETOKENIZE_REGEX:
// 1 (?i:'s|'t|'re|'ve|'m|'ll|'d)
// 2 [^\r\n\p{L}\p{N}]?\p{L}+
// 3 \p{N}
// 4  ?[^\s\p{L}\p{N}]+[\r\n]*
// 5 \s*[\r\n]+
// 6 \s+(?!\S)
// 7 \s+
// Alternatives are tried in this order at every position; the first match
// wins. Validated against tokenizers 0.22.2 golden vectors (see omni_tests).
std::vector<std::string> pretokenize(const std::string& s) {
  static constexpr std::array<const char*, 7> kContractions = {
      "'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
  std::vector<std::string> out;
  size_t i = 0;
  while (i < s.size()) {
    const size_t start = i;
    size_t nx = i;
    const uint32_t cp0 = next_cp(s, &nx);

    // 1. contractions
    bool contracted = false;
    for (const char* lit : kContractions) {
      if (ci2(s, i, lit)) {
        out.push_back(s.substr(i, 2));
        i += 2;
        contracted = true;
        break;
      }
    }
    if (contracted) continue;

    // 2. optional one-codepoint lead (not CR/LF/L/N) + one or more letters.
    {
      size_t lead_end = i;
      if (is_letter(cp0)) {
        lead_end = i;
      } else if (!is_crlf(cp0) && !is_number(cp0)) {
        lead_end = nx;
      }
      if (lead_end < s.size()) {
        size_t q = lead_end;
        if (is_letter(next_cp(s, &q))) {
          while (q < s.size()) {
            size_t r = q;
            if (!is_letter(next_cp(s, &r))) break;
            q = r;
          }
          out.push_back(s.substr(start, q - start));
          i = q;
          continue;
        }
      }
    }

    // 3. one number codepoint
    if (is_number(cp0)) {
      out.push_back(s.substr(start, nx - start));
      i = nx;
      continue;
    }

    // 4. optional ASCII space + punctuation/symbol run + trailing CR/LF*.
    {
      size_t k = i;
      if (cp0 == ' ') k = nx;
      const size_t body = k;
      while (k < s.size()) {
        size_t q = k;
        const uint32_t cp = next_cp(s, &q);
        if (is_space(cp) || is_letter(cp) || is_number(cp)) break;
        k = q;
      }
      if (k > body) {
        while (k < s.size()) {
          size_t q = k;
          if (!is_crlf(next_cp(s, &q))) break;
          k = q;
        }
        out.push_back(s.substr(start, k - start));
        i = k;
        continue;
      }
    }

    // Whitespace alternatives; cp0 is known to be whitespace here.
    if (is_space(cp0)) {
      // End of the whitespace run starting at i.
      size_t run_end = nx;
      while (run_end < s.size()) {
        size_t q = run_end;
        if (!is_space(next_cp(s, &q))) break;
        run_end = q;
      }
      // 5. \s*[\r\n]+: first CR/LF in the run, then consecutive CR/LF.
      size_t p = i;
      while (p < run_end) {
        size_t q = p;
        if (is_crlf(next_cp(s, &q))) break;
        p = q;
      }
      if (p < run_end) {
        size_t q = p;
        while (q < s.size()) {
          size_t r = q;
          if (!is_crlf(next_cp(s, &r))) break;
          q = r;
        }
        out.push_back(s.substr(start, q - start));
        i = q;
        continue;
      }
      // 6. \s+(?!\S): whole run when nothing non-whitespace follows.
      if (run_end >= s.size()) {
        out.push_back(s.substr(start, run_end - start));
        i = run_end;
        continue;
      }
      // 6 with backtrack: leave the final whitespace codepoint as an optional
      // lead for rule 2/4 (e.g. "a  b" -> "a", " ", " b").
      if (run_end - i > 1) {
        size_t prev = i, scan = i;
        while (scan < run_end) {
          size_t q = scan;
          next_cp(s, &q);
          if (q == run_end) break;
          prev = q;
          scan = q;
        }
        out.push_back(s.substr(start, prev - start));
        i = prev;
        continue;
      }
      // 7. single-codepoint run before non-whitespace content.
      out.push_back(s.substr(start, run_end - start));
      i = run_end;
      continue;
    }

    // Unreachable for valid UTF-8; advance to avoid a spin.
    out.push_back(s.substr(start, nx - start));
    i = nx;
  }
  return out;
}

}  // namespace

Qwen2BpeTokenizer::Qwen2BpeTokenizer(const std::string& vocab_json,
                                     const std::string& merges_txt,
                                     const std::string& tokenizer_config_json) {
  const nlohmann::json vocab = nlohmann::json::parse(read_text(vocab_json));
  int max_id = 0;
  for (auto it = vocab.begin(); it != vocab.end(); ++it) {
    const int id = it.value().get<int>();
    token_to_id_.emplace(it.key(), id);
    max_id = std::max(max_id, id);
  }
  // The 33 added tokens live in tokenizer_config.json, not vocab.json.
  const nlohmann::json cfg = nlohmann::json::parse(read_text(tokenizer_config_json));
  for (auto it = cfg.at("added_tokens_decoder").begin();
       it != cfg.at("added_tokens_decoder").end(); ++it) {
    const int id = std::stoi(it.key());
    const std::string content = it.value().at("content").get<std::string>();
    token_to_id_.emplace(content, id);
    specials_.emplace(content, id);
    max_id = std::max(max_id, id);
  }
  id_to_token_.resize(max_id + 1);
  for (const auto& [tok, id] : token_to_id_) id_to_token_[id] = tok;
  // Real checkpoint: 151643 base entries plus 33 added tokens ending 151675.
  if (token_to_id_.size() > 150000 && max_id != 151675)
    throw std::invalid_argument("unexpected Qwen2 vocab max id: " + std::to_string(max_id));

  std::istringstream ms(read_text(merges_txt));
  std::string line;
  int rank = 0;
  while (std::getline(ms, line)) {
    if (line.empty()) continue;
    const size_t sp = line.find(' ');
    if (sp == std::string::npos) continue;
    merge_rank_.emplace(line.substr(0, sp) + '\x01' + line.substr(sp + 1), rank++);
  }

  // GPT-2 byte_to_unicode.
  std::vector<int> bs;
  for (int c = '!'; c <= '~'; ++c) bs.push_back(c);
  for (int c = 0xa1; c <= 0xac; ++c) bs.push_back(c);
  for (int c = 0xae; c <= 0xff; ++c) bs.push_back(c);
  std::vector<int> cs = bs;
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
      bs.push_back(b);
      cs.push_back(256 + n++);
    }
  }
  for (size_t i = 0; i < bs.size(); ++i) {
    unicode_to_byte_.emplace(static_cast<uint32_t>(cs[i]), static_cast<unsigned char>(bs[i]));
    byte_to_unicode_.emplace(static_cast<unsigned char>(bs[i]), static_cast<uint32_t>(cs[i]));
  }
}

std::vector<int> Qwen2BpeTokenizer::encode(const std::string& text) const {
  // ponytail: NFC normalization is skipped; prompts are already NFC. Golden
  // vectors pin the supported input set; add a mini-UNFC pass if needed.
  std::vector<int> ids;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t next_special = std::string::npos;
    const std::string* which = nullptr;
    for (const auto& [lit, id] : specials_) {
      const size_t p = text.find(lit, pos);
      if (p != std::string::npos && p < next_special) {
        next_special = p;
        which = &lit;
      }
    }
    const size_t seg_end = next_special == std::string::npos ? text.size() : next_special;
    if (seg_end > pos) {
      const std::string seg = text.substr(pos, seg_end - pos);
      for (const std::string& piece : pretokenize(seg)) {
        std::vector<std::string> syms;
        syms.reserve(piece.size());
        for (unsigned char b : piece) {
          std::string sym;
          append_cp(sym, byte_to_unicode_.at(b));
          syms.push_back(sym);
        }
        // Lowest-rank adjacent pair wins; earliest position on rank ties.
        while (syms.size() > 1) {
          int best_rank = std::numeric_limits<int>::max();
          size_t best_pos = 0;
          for (size_t k = 0; k + 1 < syms.size(); ++k) {
            auto it = merge_rank_.find(syms[k] + '\x01' + syms[k + 1]);
            if (it != merge_rank_.end() && it->second < best_rank) {
              best_rank = it->second;
              best_pos = k;
            }
          }
          if (best_rank == std::numeric_limits<int>::max()) break;
          syms[best_pos] += syms[best_pos + 1];
          syms.erase(syms.begin() + static_cast<long>(best_pos) + 1);
        }
        for (const std::string& sym : syms) {
          auto it = token_to_id_.find(sym);
          if (it == token_to_id_.end())
            throw std::invalid_argument("BPE symbol missing from vocab: " + sym);
          ids.push_back(it->second);
        }
      }
    }
    if (next_special == std::string::npos) break;
    ids.push_back(specials_.at(*which));
    pos = seg_end + which->size();
  }
  return ids;
}

std::string Qwen2BpeTokenizer::decode(const std::vector<int>& ids) const {
  std::string bytes;
  for (int id : ids) {
    if (id < 0 || id >= static_cast<int>(id_to_token_.size()) ||
        specials_.count(id_to_token_[id]))
      continue;
    const std::string& tok = id_to_token_[id];
    size_t i = 0;
    while (i < tok.size()) {
      size_t nx = i;
      const uint32_t cp = next_cp(tok, &nx);
      auto it = unicode_to_byte_.find(cp);
      if (it != unicode_to_byte_.end()) bytes.push_back(static_cast<char>(it->second));
      i = nx;
    }
  }
  return bytes;
}

}  // namespace umminfer::omni
