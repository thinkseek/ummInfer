// SPDX-License-Identifier: Apache-2.0
#include "pipeline/narrative_parser.hpp"

#include <array>
#include <utility>

namespace umminfer {
namespace {

bool is_space(char c) {
  return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

char ascii_lower(char c) {
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

std::string trim(std::string_view text) {
  size_t begin = 0;
  size_t end = text.size();
  while (begin < end && is_space(text[begin])) ++begin;
  while (end > begin && is_space(text[end - 1])) --end;
  return std::string(text.substr(begin, end - begin));
}

constexpr std::array<std::string_view, 3> kTags = {"image", "video",
                                                   "visual"};

bool ascii_iequals(std::string_view text, size_t at, std::string_view expected) {
  if (at + expected.size() > text.size()) return false;
  for (size_t i = 0; i < expected.size(); ++i) {
    if (ascii_lower(text[at + i]) != expected[i]) return false;
  }
  return true;
}

// Returns the pipe introducing a source-like suffix, or npos. Only a suffix is
// reserved; prose such as "the source=slot-0 metaphor" remains an ordinary
// prompt. Once `| source` appears, malformed syntax is fail-closed rather than
// silently becoming an unconditioned video prompt.
size_t source_suffix_pipe(std::string_view body, size_t from = 0) {
  for (size_t pipe = body.find('|', from); pipe != std::string_view::npos;
       pipe = body.find('|', pipe + 1)) {
    size_t cursor = pipe + 1;
    while (cursor < body.size() && is_space(body[cursor])) ++cursor;
    if (!ascii_iequals(body, cursor, "source")) continue;
    cursor += 6;
    if (cursor == body.size() || body[cursor] == '=' || is_space(body[cursor])) {
      return pipe;
    }
  }
  return std::string_view::npos;
}

bool canonical_slot_id(std::string_view slot) {
  if (slot.size() < 6 || slot.substr(0, 5) != "slot-") return false;
  const std::string_view digits = slot.substr(5);
  if (digits.empty() || (digits.size() > 1 && digits.front() == '0')) return false;
  for (char c : digits) {
    if (c < '0' || c > '9') return false;
  }
  return true;
}

// Parses the single source suffix in `body` and updates `directive`.
void parse_reference(NarrativeDirective& directive, std::string body) {
  directive.prompt = body;
  const size_t first = source_suffix_pipe(body);
  if (first == std::string::npos) return;
  directive.prompt = trim(std::string_view(body).substr(0, first));
  if (source_suffix_pipe(body, first + 1) != std::string::npos) {
    directive.reference_error = "duplicate source reference";
    return;
  }
  size_t cursor = first + 1;
  while (cursor < body.size() && is_space(body[cursor])) ++cursor;
  cursor += 6;  // source_suffix_pipe already matched "source".
  while (cursor < body.size() && is_space(body[cursor])) ++cursor;
  if (cursor >= body.size() || body[cursor] != '=') {
    directive.reference_error = "source reference must use source=slot-N";
    return;
  }
  ++cursor;
  while (cursor < body.size() && is_space(body[cursor])) ++cursor;
  size_t end = body.size();
  while (end > cursor && is_space(body[end - 1])) --end;
  const std::string slot = body.substr(cursor, end - cursor);
  if (!canonical_slot_id(slot)) {
    directive.reference_error = "source reference must name slot-N";
    return;
  }
  if (directive.prompt.empty()) {
    directive.reference_error = "referenced visual requires a non-empty prompt";
    return;
  }
  directive.source_slot = slot;
}

// Match one tag at `cursor`. Returns 1 when a tag matched (cursor advances),
// 0 when no tag can match, -1 when the buffer ends inside a possible tag.
int match_tag(const std::string& buf, size_t& cursor, std::string& kind) {
  bool need_more = false;
  for (const std::string_view tag : kTags) {
    size_t i = 0;
    for (; i < tag.size(); ++i) {
      const size_t at = cursor + i;
      if (at >= buf.size()) {
        need_more = true;
        break;
      }
      if (ascii_lower(buf[at]) != tag[i]) break;
    }
    if (i == tag.size()) {
      cursor += tag.size();
      kind.assign(tag);
      return 1;
    }
  }
  return need_more ? -1 : 0;
}

}  // namespace

std::string NarrativeStreamParser::feed(std::string_view chunk,
                                        std::vector<NarrativeDirective>& out) {
  buf_.append(chunk.data(), chunk.size());
  return scan(false, out);
}

std::string NarrativeStreamParser::flush(std::vector<NarrativeDirective>& out) {
  return scan(true, out);
}

std::string NarrativeStreamParser::scan(bool eof,
                                        std::vector<NarrativeDirective>& out) {
  std::string ret;
  size_t pos = 0;
  while (pos < buf_.size()) {
    const size_t open = buf_.find("[[", pos);
    if (open == std::string::npos) {
      // A trailing lone '[' may begin a directive on the next chunk.
      size_t end = buf_.size();
      if (!eof && buf_.back() == '[') --end;
      ret.append(buf_, pos, end - pos);
      pos = end;
      break;
    }
    ret.append(buf_, pos, open - pos);

    size_t cursor = open + 2;
    while (cursor < buf_.size() && is_space(buf_[cursor])) ++cursor;
    std::string kind;
    const int tag = match_tag(buf_, cursor, kind);
    if (tag < 0) {
      if (eof) {
        // A reserved directive prefix is not narration. Finalize its exact raw
        // span instead of leaking a truncated tag into public prose or TTS.
        removed_spans_.push_back(
            NarrativeRemovedSpan{buf_raw_begin_ + open, buf_raw_begin_ + buf_.size()});
        pos = buf_.size();
      } else {
        pos = open;  // hold the undecidable suffix
      }
      break;
    }
    if (tag == 0) {
      // Not one of the tags: this '[' is prose. Advance one byte so an
      // overlapping "[[[" can still open a directive at the next index.
      ret.push_back('[');
      pos = open + 1;
      continue;
    }
    while (cursor < buf_.size() && is_space(buf_[cursor])) ++cursor;
    if (cursor >= buf_.size()) {
      if (!eof) {
        pos = open;
        break;
      }
      removed_spans_.push_back(
          NarrativeRemovedSpan{buf_raw_begin_ + open, buf_raw_begin_ + buf_.size()});
      pos = buf_.size();
      break;
    }
    if (buf_[cursor] != ':') {
      ret.push_back('[');
      pos = open + 1;
      continue;
    }
    ++cursor;
    const size_t close = buf_.find("]]", cursor);
    if (close == std::string::npos) {
      if (!eof && buf_.size() - open <= kMaxDirectiveBytes) {
        pos = open;
        break;
      }
      if (eof) {
        removed_spans_.push_back(
            NarrativeRemovedSpan{buf_raw_begin_ + open, buf_raw_begin_ + buf_.size()});
        pos = buf_.size();
        break;
      }
      // An over-limit candidate is not allowed to stall the stream. This legacy
      // non-EOF fallback is unchanged; a final reserved candidate is suppressed.
      ret.push_back('[');
      pos = open + 1;
      continue;
    }
    removed_spans_.push_back(
        NarrativeRemovedSpan{buf_raw_begin_ + open, buf_raw_begin_ + close + 2});
    NarrativeDirective directive;
    directive.kind = std::move(kind);
    parse_reference(directive,
                    trim(std::string_view(buf_).substr(cursor, close - cursor)));
    if (!directive.prompt.empty() || !directive.source_slot.empty() ||
        !directive.reference_error.empty()) {
      directive.prose_offset = prose_emitted_ + ret.size();
      out.push_back(std::move(directive));
    }
    pos = close + 2;
  }
  buf_.erase(0, pos);
  buf_raw_begin_ += pos;
  prose_emitted_ += ret.size();
  return ret;
}

std::optional<std::vector<int64_t>> narrative_behavior_masks(
    const std::vector<int>& token_ids,
    const std::function<std::string(int)>& decode_token) {
  if (token_ids.empty() || !decode_token) return std::nullopt;
  NarrativeStreamParser parser;
  std::vector<NarrativeDirective> directives;
  std::vector<std::string> token_bytes;
  token_bytes.reserve(token_ids.size());
  std::string clean;
  std::string raw;
  for (const int id : token_ids) {
    token_bytes.push_back(decode_token(id));
    raw += token_bytes.back();
    clean += parser.feed(token_bytes.back(), directives);
  }
  clean += parser.flush(directives);

  std::vector<int64_t> mask(token_ids.size(), 1);
  size_t token_begin = 0;
  for (size_t i = 0; i < token_bytes.size(); ++i) {
    const size_t token_end = token_begin + token_bytes[i].size();
    if (token_bytes[i].empty()) mask[i] = 0;
    for (const NarrativeRemovedSpan& span : parser.removed_spans()) {
      if (span.begin > span.end || span.end > raw.size()) return std::nullopt;
      if (token_begin < span.end && token_end > span.begin) mask[i] = 0;
    }
    token_begin = token_end;
  }
  std::string masked;
  for (size_t i = 0; i < token_bytes.size(); ++i) {
    if (mask[i] == 1) masked += token_bytes[i];
  }
  if (masked != clean) return std::nullopt;
  return mask;
}

}  // namespace umminfer
