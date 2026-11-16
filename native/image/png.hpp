// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace umminfer::image {

// Minimal PNG (RGB8, 8-bit, no interlace) writer. Scanline filtering uses
// filter type 0 (None); compression is zlib deflate (zlib license, linked
// dynamically on both build hosts — no third-party source vendored).
struct RgbImage {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> rgb;  // width*height*3, row-major, top-left origin

  RgbImage() = default;
  RgbImage(int w, int h) : width(w), height(h), rgb(static_cast<size_t>(w) * h * 3) {}
};

// Encodes to PNG bytes; throws std::invalid_argument on bad dimensions.
std::string encode_png(const RgbImage& img);

// Tensor [3, H, W] fp32 with values in [-1, 1] (VAE output convention) ->
// clamped, scaled to [0,255] with round-half-away-from-zero-free std::lrintf.
RgbImage tensor_n1p1_to_rgb(const float* chw, int height, int width);

}  // namespace umminfer::image
