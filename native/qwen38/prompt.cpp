// SPDX-License-Identifier: Apache-2.0
// Host-only prompt construction shared by the CLI and the host tests.
#include <cctype>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine.hpp"

namespace umminfer::qwen38 {
namespace {
constexpr int kImagePad = 248056;
constexpr int kVideoPad = 248057;

std::string_view trim_ascii(std::string_view s) {
  size_t b = 0;
  size_t e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

}  // namespace

RenderedChatPrompt render_chat_prompt_parts(const std::vector<ChatTurn>& turns,
                                            bool enable_thinking) {
  if (turns.empty()) {
    throw std::invalid_argument("render_chat_prompt requires at least one turn");
  }
  RenderedChatPrompt parts;
  for (const ChatTurn& turn : turns) {
    const std::string& role = turn.role.empty() ? std::string("user") : turn.role;
    parts.prefix += "<|im_start|>" + role + "\n";
    if (turn.has_image) {
      parts.prefix += "<|vision_start|><|image_pad|><|vision_end|>";
    } else if (turn.video_slices > 0) {
      // Qwen3.5/Qwen3.8 separates video temporal slices with timestamp markers
      // matching Qwen3_5Model.get_rope_index.
      for (int s = 0; s < turn.video_slices; ++s) {
        const double sec =
            static_cast<size_t>(s) < turn.video_timestamps.size()
                ? turn.video_timestamps[static_cast<size_t>(s)]
                : static_cast<double>(s);
        char stamp[48];
        std::snprintf(stamp, sizeof(stamp), "<%.1f seconds>", sec);
        parts.prefix += stamp;
        parts.prefix += "<|vision_start|><|video_pad|><|vision_end|>";
      }
    }
    // The Jinja template applies |trim to the rendered content of each turn.
    const std::string_view trimmed = trim_ascii(turn.text);
    parts.prefix.append(trimmed.data(), trimmed.size());
    parts.prefix += "<|im_end|>\n";
  }
  const std::string think_open = "<" "think" ">";
  const std::string think_close = "</" "think" ">";
  parts.suffix = "<|im_start|>assistant\n";
  parts.suffix += enable_thinking ? think_open + "\n"
                                 : think_open + "\n\n" + think_close + "\n\n";
  return parts;
}

std::string render_chat_prompt(const std::vector<ChatTurn>& turns,
                               bool enable_thinking) {
  return render_chat_prompt_parts(turns, enable_thinking).full();
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

std::vector<int> expand_video_pads(const std::vector<int>& ids,
                                   int n_tokens_per_slice,
                                   int expected_slices) {
  if (n_tokens_per_slice <= 0) {
    throw std::invalid_argument("n_tokens_per_slice must be positive");
  }
  std::vector<int> out;
  out.reserve(ids.size() +
              static_cast<size_t>(std::max(1, expected_slices)) *
                  static_cast<size_t>(n_tokens_per_slice));
  int slices = 0;
  for (int id : ids) {
    if (id == kVideoPad) {
      out.insert(out.end(), static_cast<size_t>(n_tokens_per_slice), kVideoPad);
      ++slices;
      continue;
    }
    out.push_back(id);
  }
  if (slices == 0) {
    throw std::runtime_error("no video_pad token in rendered prompt");
  }
  if (expected_slices > 0 && slices != expected_slices) {
    throw std::runtime_error("video_pad placeholder count does not match grid_t");
  }
  return out;
}

}  // namespace umminfer::qwen38
