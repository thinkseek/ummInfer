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

// Match one tag at `cursor`. Returns 1 when a tag matched (cursor advances),
// 0 when no tag can match, -1 when the buffer ends inside a possible tag.
int match_tag(const std::string& buf, size_t& cursor, std::string& kind,
              bool eof) {
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
  return (need_more && !eof) ? -1 : 0;
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
    const int tag = match_tag(buf_, cursor, kind, eof);
    if (tag < 0) {
      pos = open;  // hold the undecidable suffix
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
      ret.push_back('[');
      pos = open + 1;
      continue;
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
      // End of input or an unbounded candidate: release it as prose rather
      // than stall the narrative indefinitely.
      ret.push_back('[');
      pos = open + 1;
      continue;
    }
    std::string prompt =
        trim(std::string_view(buf_).substr(cursor, close - cursor));
    if (!prompt.empty()) {
      NarrativeDirective directive;
      directive.kind = std::move(kind);
      directive.prompt = std::move(prompt);
      directive.prose_offset = prose_emitted_ + ret.size();
      out.push_back(std::move(directive));
    }
    pos = close + 2;
  }
  buf_.erase(0, pos);
  prose_emitted_ += ret.size();
  return ret;
}

}  // namespace umminfer
