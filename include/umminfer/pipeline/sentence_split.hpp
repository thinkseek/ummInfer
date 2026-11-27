// SPDX-License-Identifier: Apache-2.0
// Streaming sentence splitter for the sentence-level TTS pipeline.
//
// A VLM delta stream is cut into sentences so each sentence can start TTS
// before the answer ends. The rules are intentionally small and explicit:
//
//   strong terminators   。！？!?；;\n and the ellipsis character …
//   weak breaks          ，,、:：  and ASCII space (Latin word boundary)
//   closing absorption   after a cut, trailing closing quotes/brackets and
//                        repeated terminators stay with the sentence
//   '.'                  a terminator only when the next codepoint is
//                        whitespace, end of text or a CJK codepoint, and the
//                        previous codepoint is not a digit (so "3.14" and
//                        "v1.2" stay whole)
//   length threshold     max_codepoints codepoints; a weak break in the second
//                        half of the window wins, otherwise the window edge is
//                        the cut
//
// Uncertain boundaries, not handled here:
//   - abbreviations ("Mr. Smith", "e.g. this") are cut after the period;
//   - "U.S. Army" becomes "U.S." + " Army";
//   - a streaming delta that ends mid-sentence is kept until a later delta
//     completes the boundary, so a long sentence delays TTS by up to
//     max_codepoints codepoints;
//   - no semantic end-of-turn model: this is punctuation only (L3).
// A cut never splits a UTF-8 sequence.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

namespace umminfer {

namespace sentence_detail {

inline size_t utf8_length(unsigned char lead) {
  if (lead < 0x80) return 1;
  if ((lead & 0xE0) == 0xC0) return 2;
  if ((lead & 0xF0) == 0xE0) return 3;
  if ((lead & 0xF8) == 0xF0) return 4;
  return 1;
}

inline uint32_t decode_cp(const std::string& text, size_t offset, size_t length) {
  const auto b0 = static_cast<unsigned char>(text[offset]);
  if (length == 1) return b0;
  uint32_t code = b0 & (0xFFu >> (length + 1));
  for (size_t k = 1; k < length && offset + k < text.size(); ++k) {
    code = (code << 6) | (static_cast<unsigned char>(text[offset + k]) & 0x3Fu);
  }
  return code;
}

inline bool is_cjk(uint32_t cp) {
  return (cp >= 0x3000 && cp <= 0x303F) || (cp >= 0x3400 && cp <= 0x9FFF) ||
         (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0xFF00 && cp <= 0xFFEF) ||
         (cp >= 0x20000 && cp <= 0x2FA1F);
}

inline bool is_closer(uint32_t cp) {
  switch (cp) {
    case U'"': case U'\'': case U')': case U']': case U'}':
    case 0x300D: case 0x300F: case 0x3011: case 0x3015:  // 」』〕〗
    case 0x3009: case 0x300B: case 0xFF09: case 0xFF3D:  // 〉》  ）］
    case 0x201C: case 0x201D: case 0x2019: case 0x00BB:
      return true;
    default:
      return false;
  }
}

inline bool is_strong(uint32_t cp) {
  switch (cp) {
    case U'\n':
    case 0x3002:  // 。
    case 0xFF01:  // ！
    case 0xFF1F:  // ？
    case 0x2026:  // …
    case 0xFF1B:  // ；
    case U'!': case U'?': case U';':
      return true;
    default:
      return false;
  }
}

inline bool weak_break(uint32_t cp) {
  switch (cp) {
    case U',': case U':': case U' ': case U'\t':
    case 0xFF0C:  // ，
    case 0x3001:  // 、
    case 0xFF1A:  // ：
      return true;
    default:
      return false;
  }
}

}  // namespace sentence_detail

// Byte length of the first sentence in `text`, or 0 when no cut is due.
inline size_t sentence_cut(const std::string& text, size_t max_codepoints) {
  using namespace sentence_detail;
  if (text.empty() || max_codepoints == 0) return 0;
  size_t codepoints = 0;
  size_t last_weak_end = 0;
  size_t last_weak_cp = 0;
  size_t offset = 0;
  uint32_t prev = 0;
  while (offset < text.size()) {
    const size_t length =
        std::min(utf8_length(static_cast<unsigned char>(text[offset])),
                 text.size() - offset);
    const uint32_t cp = decode_cp(text, offset, length);
    size_t next_offset = offset + length;
    uint32_t next = 0;
    if (next_offset < text.size()) {
      const size_t next_length =
          std::min(utf8_length(static_cast<unsigned char>(text[next_offset])),
                   text.size() - next_offset);
      next = decode_cp(text, next_offset, next_length);
    }
    bool terminator = is_strong(cp);
    if (cp == U'.') {
      const bool after_digit = prev >= U'0' && prev <= U'9';
      const bool at_end = next_offset >= text.size();
      const bool spaced = next == U' ' || next == U'\t' || next == U'\n' ||
                          next == U'\r';
      terminator = !after_digit && (at_end || spaced || is_cjk(next));
    }
    if (terminator) {
      ++codepoints;
      // Absorb closing quotes/brackets and a run of terminators.
      size_t end = next_offset;
      while (end < text.size()) {
        const size_t l =
            std::min(utf8_length(static_cast<unsigned char>(text[end])),
                     text.size() - end);
        const uint32_t c = decode_cp(text, end, l);
        if (is_closer(c) || is_strong(c) || c == U'.') {
          end += l;
          continue;
        }
        break;
      }
      return end;
    }
    ++codepoints;
    if (weak_break(cp)) {
      last_weak_end = next_offset;
      last_weak_cp = codepoints;
    }
    prev = cp;
    if (codepoints >= max_codepoints) {
      if (last_weak_cp * 2 >= codepoints) return last_weak_end;
      return next_offset;
    }
    offset = next_offset;
  }
  return 0;
}

// Byte length of the first clause cut: the last weak break at or after
// `min_codepoints` codepoints, or 0 when no such break exists yet. Used only
// for the first TTS leg, so a long opening sentence does not delay all audio.
inline size_t clause_cut(const std::string& text, size_t min_codepoints) {
  using namespace sentence_detail;
  size_t codepoints = 0;
  size_t cut = 0;
  size_t offset = 0;
  while (offset < text.size()) {
    const size_t length =
        std::min(utf8_length(static_cast<unsigned char>(text[offset])),
                 text.size() - offset);
    const uint32_t cp = decode_cp(text, offset, length);
    offset += length;
    ++codepoints;
    if (codepoints >= min_codepoints && weak_break(cp)) cut = offset;
  }
  // A cut at the very end is not useful: the sentence may continue.
  return cut;
}

// Converts one already-cut public Markdown span to the text intended for
// speech. Public prose and playback cursors remain in the original byte space.
// The two bools carry single-delimiter emphasis state across adjacent spans.
// Presentation-only emphasis/code delimiters, heading/list/quote prefixes, and
// horizontal rules are not Talker input. Ordinary star arithmetic and intraword
// underscores are preserved rather than silently dropped.
inline std::string speech_text_from_markdown(const std::string& text,
                                             bool& single_star_open,
                                             bool& single_underscore_open) {
  const auto ascii_space = [](char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
  };
  const auto ascii_alnum = [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') ||
           (c >= 'a' && c <= 'z');
  };
  // A horizontal rule is a complete markup-only span, not narration.
  char rule = 0;
  size_t rule_count = 0;
  bool horizontal_rule = true;
  for (char c : text) {
    if (ascii_space(c)) continue;
    if (c != '-' && c != '*' && c != '_') {
      horizontal_rule = false;
      break;
    }
    if (rule == 0) rule = c;
    if (c != rule) {
      horizontal_rule = false;
      break;
    }
    ++rule_count;
  }
  if (horizontal_rule && rule_count >= 3) return {};

  std::string out;
  out.reserve(text.size());
  bool line_start = true;
  size_t i = 0;
  while (i < text.size()) {
    if (line_start) {
      // Up to three spaces of Markdown indentation precede a block marker.
      size_t marker = i;
      size_t indentation = 0;
      while (marker < text.size() && text[marker] == ' ' && indentation < 3) {
        ++marker;
        ++indentation;
      }
      size_t end = marker;
      if (end < text.size() && text[end] == '#') {
        while (end < text.size() && text[end] == '#') ++end;
        if (end < text.size() && (text[end] == ' ' || text[end] == '\t')) {
          i = end + 1;
          continue;
        }
      }
      if (end < text.size() && text[end] == '>') {
        i = end + 1;
        if (i < text.size() && (text[i] == ' ' || text[i] == '\t')) ++i;
        continue;
      }
      if (end < text.size() && (text[end] == '*' || text[end] == '-' ||
                                text[end] == '+') &&
          end + 1 < text.size() &&
          (text[end + 1] == ' ' || text[end + 1] == '\t')) {
        i = end + 2;
        while (i < text.size() && (text[i] == ' ' || text[i] == '\t')) ++i;
        continue;
      }
      line_start = false;
    }
    if (text[i] == '\n') {
      out.push_back(text[i++]);
      line_start = true;
      continue;
    }
    if (text[i] == '*' || text[i] == '_' || text[i] == '~' || text[i] == '`') {
      const char marker = text[i];
      size_t end = i + 1;
      while (end < text.size() && text[end] == marker) ++end;
      const size_t count = end - i;
      bool remove = marker == '`' || (marker == '~' && count >= 2) ||
                    ((marker == '*' || marker == '_') && count >= 2);
      const bool left_alnum = i > 0 && ascii_alnum(text[i - 1]);
      const bool right_alnum = end < text.size() && ascii_alnum(text[end]);
      if (marker == '_' && count >= 2) {
        remove = !(left_alnum && right_alnum);
      }
      if ((marker == '*' || marker == '_') && count == 1) {
        const bool left_space = i == 0 || ascii_space(text[i - 1]);
        const bool right_space = end == text.size() || ascii_space(text[end]);
        const bool digit_arithmetic =
            i > 0 && end < text.size() && text[i - 1] >= '0' && text[i - 1] <= '9' &&
            text[end] >= '0' && text[end] <= '9';
        const bool intraword_underscore = marker == '_' && left_alnum && right_alnum;
        bool& open = marker == '*' ? single_star_open : single_underscore_open;
        const bool can_open = !right_space && !digit_arithmetic && !intraword_underscore;
        const bool can_close = !left_space || (i == 0 && open);
        if (!digit_arithmetic && !intraword_underscore &&
            ((open && can_close) || (!open && can_open))) {
          open = !open;
          remove = true;
        } else {
          remove = false;
        }
      }
      if (!remove) out.append(text, i, count);
      i = end;
      continue;
    }
    out.push_back(text[i++]);
  }
  return out;
}

}  // namespace umminfer
