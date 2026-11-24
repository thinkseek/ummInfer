// SPDX-License-Identifier: Apache-2.0
#include "umt5_tokenizer.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace umminfer::video {
namespace {

struct Utf8Char {
  uint32_t codepoint;
  size_t size;
};

Utf8Char decode_utf8(const std::string& text, size_t offset) {
  const auto byte = [&](size_t i) {
    return static_cast<unsigned char>(text[offset + i]);
  };
  const uint8_t first = byte(0);
  if (first < 0x80) return {first, 1};

  size_t size = 0;
  uint32_t codepoint = 0;
  if ((first & 0xe0) == 0xc0) {
    size = 2;
    codepoint = first & 0x1f;
  } else if ((first & 0xf0) == 0xe0) {
    size = 3;
    codepoint = first & 0x0f;
  } else if ((first & 0xf8) == 0xf0) {
    size = 4;
    codepoint = first & 0x07;
  } else {
    throw std::invalid_argument("invalid UTF-8 leading byte in prompt");
  }
  if (offset + size > text.size()) {
    throw std::invalid_argument("truncated UTF-8 codepoint in prompt");
  }
  for (size_t i = 1; i < size; ++i) {
    if ((byte(i) & 0xc0) != 0x80) {
      throw std::invalid_argument("invalid UTF-8 continuation byte in prompt");
    }
    codepoint = (codepoint << 6) | (byte(i) & 0x3f);
  }
  const uint32_t minimum = size == 2 ? 0x80 : size == 3 ? 0x800 : 0x10000;
  if (codepoint < minimum || codepoint > 0x10ffff ||
      (codepoint >= 0xd800 && codepoint <= 0xdfff)) {
    throw std::invalid_argument("invalid UTF-8 codepoint in prompt");
  }
  return {codepoint, size};
}

void append_utf8(std::string& output, uint32_t codepoint) {
  if (codepoint > 0x10ffff || (codepoint >= 0xd800 && codepoint <= 0xdfff)) {
    codepoint = 0xfffd;
  }
  if (codepoint < 0x80) {
    output.push_back(static_cast<char>(codepoint));
  } else if (codepoint < 0x800) {
    output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
    output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
  } else if (codepoint < 0x10000) {
    output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
    output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
    output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
  } else {
    output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
    output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
    output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
    output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
  }
}

std::string html_unescape_once(const std::string& text) {
  static const std::unordered_map<std::string, uint32_t> named = {
      {"amp", '&'}, {"lt", '<'}, {"gt", '>'},
      {"quot", '"'}, {"apos", '\''}, {"nbsp", 0xa0},
  };
  std::string output;
  for (size_t i = 0; i < text.size();) {
    if (text[i] != '&') {
      output.push_back(text[i++]);
      continue;
    }
    const size_t semicolon = text.find(';', i + 1);
    if (semicolon == std::string::npos || semicolon - i > 64) {
      output.push_back(text[i++]);
      continue;
    }
    const std::string entity = text.substr(i + 1, semicolon - i - 1);
    const auto name = named.find(entity);
    if (name != named.end()) {
      append_utf8(output, name->second);
      i = semicolon + 1;
      continue;
    }
    if (!entity.empty() && entity[0] == '#') {
      try {
        size_t used = 0;
        const bool hexadecimal = entity.size() > 2 &&
            (entity[1] == 'x' || entity[1] == 'X');
        const std::string digits = entity.substr(hexadecimal ? 2 : 1);
        const unsigned long codepoint = std::stoul(digits, &used, hexadecimal ? 16 : 10);
        if (used == digits.size() && !digits.empty()) {
          append_utf8(output, static_cast<uint32_t>(codepoint));
          i = semicolon + 1;
          continue;
        }
      } catch (const std::exception&) {
        // Preserve malformed entities verbatim, as ordinary prompt text.
      }
    }
    output.append(text, i, semicolon - i + 1);
    i = semicolon + 1;
  }
  return output;
}

bool is_unicode_whitespace(uint32_t c) {
  return (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x20) ||
         c == 0x85 || c == 0xa0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 ||
         c == 0x202f || c == 0x205f || c == 0x3000;
}

std::string clean_prompt(const std::string& text) {
  const std::string unescaped = html_unescape_once(html_unescape_once(text));
  std::string cleaned;
  bool pending_space = false;
  for (size_t i = 0; i < unescaped.size();) {
    const Utf8Char c = decode_utf8(unescaped, i);
    if (is_unicode_whitespace(c.codepoint)) {
      pending_space = !cleaned.empty();
    } else {
      if (pending_space) cleaned.push_back(' ');
      cleaned.append(unescaped, i, c.size);
      pending_space = false;
    }
    i += c.size;
  }
  return cleaned;
}

std::string byte_piece(unsigned char value) {
  constexpr char hex[] = "0123456789ABCDEF";
  std::string piece = "<0x00>";
  piece[3] = hex[value >> 4];
  piece[4] = hex[value & 0x0f];
  return piece;
}

}  // namespace

Umt5Tokenizer::Umt5Tokenizer(const std::filesystem::path& tokenizer_json) {
  std::ifstream input(tokenizer_json);
  if (!input) {
    throw std::runtime_error("failed to open UMT5 tokenizer: " +
                             tokenizer_json.string());
  }
  const auto root = nlohmann::json::parse(input);
  const auto& model = root.at("model");
  if (model.at("type").get<std::string>() != "Unigram") {
    throw std::invalid_argument("UMT5 tokenizer model is not Unigram");
  }
  unk_id_ = model.at("unk_id").get<int32_t>();
  byte_fallback_ = model.value("byte_fallback", false);
  const auto& vocab = model.at("vocab");
  pieces_.reserve(vocab.size());
  scores_.reserve(vocab.size());
  for (const auto& entry : vocab) {
    const std::string piece = entry.at(0).get<std::string>();
    const int32_t id = static_cast<int32_t>(pieces_.size());
    if (!vocab_.emplace(piece, id).second) {
      throw std::invalid_argument("duplicate piece in UMT5 tokenizer: " + piece);
    }
    pieces_.push_back(piece);
    scores_.push_back(entry.at(1).get<double>());
    if (piece != "<unk>") max_piece_bytes_ = std::max(max_piece_bytes_, piece.size());
  }
  if (unk_id_ < 0 || static_cast<size_t>(unk_id_) >= pieces_.size()) {
    throw std::invalid_argument("UMT5 tokenizer has an invalid unk_id");
  }
  if (pieces_.size() <= 1 || pieces_[0] != "<pad>" || pieces_[1] != "</s>") {
    throw std::invalid_argument("UMT5 tokenizer requires <pad>=0 and </s>=1");
  }
  unk_score_ = *std::min_element(scores_.begin(), scores_.end()) - 10.0;

  if (const auto it = root.find("added_tokens"); it != root.end()) {
    for (const auto& token : *it) {
      const int32_t id = token.at("id").get<int32_t>();
      const std::string content = token.at("content").get<std::string>();
      if (id < 0 || static_cast<size_t>(id) >= pieces_.size() ||
          pieces_[id] != content) {
        throw std::invalid_argument("invalid UMT5 added token: " + content);
      }
      added_tokens_.push_back({content, id});
    }
    std::sort(added_tokens_.begin(), added_tokens_.end(),
              [](const AddedToken& a, const AddedToken& b) {
                return a.text.size() > b.text.size();
              });
  }
  if (const auto it = root.find("pre_tokenizer"); it != root.end() && !it->is_null()) {
    if (it->at("type").get<std::string>() != "Metaspace" ||
        it->value("replacement", std::string("▁")) != "▁" ||
        it->value("prepend_scheme", std::string("always")) != "always" ||
        !it->value("split", true)) {
      throw std::invalid_argument("unsupported UMT5 pre-tokenizer");
    }
  }
}

std::vector<int32_t> Umt5Tokenizer::encode_unigram(const std::string& input) const {
  const size_t n = input.size();
  constexpr double kNegativeInfinity = -std::numeric_limits<double>::infinity();
  std::vector<double> best(n + 1, kNegativeInfinity);
  std::vector<size_t> previous(n + 1, n + 1);
  std::vector<int32_t> previous_id(n + 1, -1);
  std::vector<bool> continuation(n + 1, false);
  best[0] = 0.0;

  const auto relax = [&](size_t from, size_t to, int32_t id, double score) {
    const double candidate = best[from] + score;
    if (candidate > best[to]) {
      best[to] = candidate;
      previous[to] = from;
      previous_id[to] = id;
    }
  };

  for (size_t i = 0; i < n; ++i) {
    if (!std::isfinite(best[i]) || continuation[i]) continue;
    const Utf8Char c = decode_utf8(input, i);
    const auto single = vocab_.find(input.substr(i, c.size));
    const bool has_single_node = single != vocab_.end() && single->second != unk_id_;
    const size_t longest = std::min(n - i, max_piece_bytes_);
    for (size_t length = 1; length <= longest; ++length) {
      const auto it = vocab_.find(input.substr(i, length));
      if (it == vocab_.end() || it->second == unk_id_) continue;
      relax(i, i + length, it->second, scores_[it->second]);
    }
    if (has_single_node) continue;

    if (byte_fallback_) {
      size_t from = i;
      for (size_t j = 0; j < c.size; ++j) {
        const auto byte = vocab_.find(
            byte_piece(static_cast<unsigned char>(input[i + j])));
        if (byte == vocab_.end()) break;
        relax(from, i + j + 1, byte->second, scores_[byte->second]);
        if (j + 1 < c.size) continuation[i + j + 1] = true;
        from = i + j + 1;
      }
    } else {
      relax(i, i + c.size, unk_id_, unk_score_);
    }
  }
  if (n != 0 && previous[n] == n + 1) {
    throw std::runtime_error("UMT5 Unigram segmentation failed");
  }

  std::vector<int32_t> reversed;
  for (size_t at = n; at != 0; at = previous[at]) {
    if (previous[at] > at || previous_id[at] < 0) {
      throw std::runtime_error("invalid UMT5 Unigram backtrace");
    }
    reversed.push_back(previous_id[at]);
  }
  std::reverse(reversed.begin(), reversed.end());
  std::vector<int32_t> result;
  result.reserve(reversed.size());
  for (const int32_t id : reversed) {
    if (id == unk_id_ && !result.empty() && result.back() == unk_id_) continue;
    result.push_back(id);
  }
  return result;
}

Umt5TokenBatch Umt5Tokenizer::encode(const std::string& text,
                                     size_t max_length,
                                     bool pad_to_max) const {
  if (max_length == 0) throw std::invalid_argument("UMT5 max_length must be positive");
  const std::string cleaned = clean_prompt(text);
  std::vector<int32_t> ids;

  const auto append_normal = [&](const std::string& chunk) {
    if (chunk.empty()) return;
    std::string transformed;
    if (!chunk.starts_with("▁")) transformed = "▁";
    for (const char c : chunk) {
      if (c == ' ') {
        transformed += "▁";
      } else {
        transformed.push_back(c);
      }
    }
    size_t begin = 0;
    size_t search = 3;  // Every non-empty transformed chunk starts with U+2581.
    while (true) {
      const size_t next = transformed.find("▁", search);
      const size_t end = next == std::string::npos ? transformed.size() : next;
      const auto part = encode_unigram(transformed.substr(begin, end - begin));
      ids.insert(ids.end(), part.begin(), part.end());
      if (next == std::string::npos) break;
      begin = next;
      search = next + 3;
    }
  };

  size_t normal_begin = 0;
  size_t position = 0;
  while (position < cleaned.size()) {
    const AddedToken* match = nullptr;
    for (const auto& token : added_tokens_) {
      if (position + token.text.size() <= cleaned.size() &&
          cleaned.compare(position, token.text.size(), token.text) == 0) {
        match = &token;
        break;
      }
    }
    if (match == nullptr) {
      ++position;
      continue;
    }
    append_normal(cleaned.substr(normal_begin, position - normal_begin));
    ids.push_back(match->id);
    position += match->text.size();
    normal_begin = position;
  }
  append_normal(cleaned.substr(normal_begin));

  if (ids.size() >= max_length) ids.resize(max_length - 1);
  ids.push_back(1);
  Umt5TokenBatch result;
  result.ids = std::move(ids);
  result.attention_mask.assign(result.ids.size(), 1);
  if (pad_to_max) {
    result.ids.resize(max_length, 0);
    result.attention_mask.resize(max_length, 0);
  }
  return result;
}

}  // namespace umminfer::video
