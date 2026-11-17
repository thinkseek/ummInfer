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
}

std::string render_chat_prompt(const std::string& user_text, bool has_image,
                               bool enable_thinking) {
  // The Jinja template applies |trim to the rendered content.
  size_t b = 0, e = user_text.size();
  while (b < e && std::isspace(static_cast<unsigned char>(user_text[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(user_text[e - 1]))) --e;
  std::string out = "<|im_start|>user\n";
  if (has_image) out += "<|vision_start|><|image_pad|><|vision_end|>";
  out += user_text.substr(b, e - b);
  out += "<|im_end|>\n<|im_start|>assistant\n";
  const std::string think_open = "<" "think" ">";
  const std::string think_close = "</" "think" ">";
  out += enable_thinking ? think_open + "\n"
                         : think_open + "\n\n" + think_close + "\n\n";
  return out;
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
