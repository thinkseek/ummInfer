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

}  // namespace umminfer
