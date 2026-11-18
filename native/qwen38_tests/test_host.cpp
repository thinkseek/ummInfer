// SPDX-License-Identifier: Apache-2.0
// Host-side checks: tokenizer round trip, chat rendering, image preprocessing.
// CHECK is always active: Release builds define NDEBUG, which would silently
// empty assert().
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "qwen38/engine.hpp"
#include "qwen38/image_preprocess.hpp"
#include "qwen38/tokenizer.hpp"

using namespace umminfer::qwen38;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

int main() {
  // smart_resize keeps multiples of 32 and the 512x512 reference size.
  int h = 0, w = 0;
  smart_resize(512, 512, 32, 65536, 16777216, &h, &w);
  CHECK(h == 512 && w == 512);
  smart_resize(300, 200, 32, 65536, 16777216, &h, &w);
  CHECK(h % 32 == 0 && w % 32 == 0);

  // Synthetic 512x512 image: 32x32 pre-merge grid, 1024 rows of 1536 values.
  std::vector<uint8_t> rgb(512 * 512 * 3);
  for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = static_cast<uint8_t>(i * 31);
  const PreprocessedImage img = preprocess_rgb(rgb.data(), 512, 512, 65536, 16777216);
  CHECK(img.grid_t == 1 && img.grid_h == 32 && img.grid_w == 32);
  CHECK(img.rows() == 1024);
  CHECK(img.pixel_values.size() == 1024u * 1536u);
  for (float v : img.pixel_values) CHECK(v >= -1.0f && v <= 1.0f);

  // Chat rendering matches the template with enable_thinking=false.
  const std::string text = render_chat_prompt("hello", false, false);
  CHECK(text ==
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        "<" "think" ">\n\n</" "think" ">\n\n");
  const std::string with_img = render_chat_prompt("hi", true, false);
  CHECK(with_img.rfind("<|im_start|>user\n<|vision_start|><|image_pad|>"
                       "<|vision_end|>hi<|im_end|>\n",
                       0) == 0);
  const std::string multi_turn = render_chat_prompt(
      {{"system", "be brief", false},
       {"user", "first", false},
       {"assistant", "ok", false},
       {"user", "second", true}},
      false);
  CHECK(multi_turn ==
        "<|im_start|>system\nbe brief<|im_end|>\n"
        "<|im_start|>user\nfirst<|im_end|>\n"
        "<|im_start|>assistant\nok<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>second<|im_end|>\n"
        "<|im_start|>assistant\n<" "think" ">\n\n</" "think" ">\n\n");

  // One placeholder becomes exactly n tokens (not n + 1).
  const std::vector<int> ids = expand_image_pads({1, 248056, 2}, 4);
  CHECK((ids == std::vector<int>{1, 248056, 248056, 248056, 248056, 2}));

  std::printf("host tests passed\n");
  return 0;
}
