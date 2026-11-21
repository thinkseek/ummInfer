// SPDX-License-Identifier: Apache-2.0
// Host-only prompt construction shared by the CLI and the host tests.
#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine.hpp"

namespace umminfer::qwen38 {
namespace {
constexpr int kImagePad = 248056;

std::string_view trim_ascii(std::string_view s) {
  size_t b = 0;
  size_t e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

}  // namespace

std::string render_chat_prompt(const std::vector<ChatTurn>& turns,
                               bool enable_thinking) {
  if (turns.empty()) {
    throw std::invalid_argument("render_chat_prompt requires at least one turn");
  }
  std::string out;
  for (const ChatTurn& turn : turns) {
    const std::string& role = turn.role.empty() ? std::string("user") : turn.role;
    out += "<|im_start|>" + role + "\n";
    if (turn.has_image) out += "<|vision_start|><|image_pad|><|vision_end|>";
    // The Jinja template applies |trim to the rendered content of each turn.
    const std::string_view trimmed = trim_ascii(turn.text);
    out.append(trimmed.data(), trimmed.size());
    out += "<|im_end|>\n";
  }
  out += "<|im_start|>assistant\n";
  const std::string think_open = "<" "think" ">";
  const std::string think_close = "</" "think" ">";
  out += enable_thinking ? think_open + "\n"
                         : think_open + "\n\n" + think_close + "\n\n";
  return out;
}

std::string render_chat_prompt(const std::string& user_text, bool has_image,
                               bool enable_thinking) {
  return render_chat_prompt(
      std::vector<ChatTurn>{ChatTurn{"user", user_text, has_image}},
      enable_thinking);
}

std::vector<int> expand_image_pads(const std::vector<int>& ids, int n_tokens) {
  std::vector<int> out;
  out.reserve(ids.size() + n_tokens);
  bool inserted = false;
  for (int id : ids) {
    if (!inserted && id == kImagePad) {
      // Replace the single placeholder with one id per merged vision token.
      out.insert(out.end(), n_tokens, kImagePad);
      inserted = true;
      continue;
    }
    out.push_back(id);
  }
  if (!inserted) throw std::runtime_error("no image_pad token in rendered prompt");
  return out;
}

}  // namespace umminfer::qwen38
