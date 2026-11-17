// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace umminfer::image2 {

// Minimal PNG writer (8-bit RGBA, no interlace, filter 0, zlib deflate).
struct RgbaImage {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> rgba;  // width*height*4, row-major, top-left origin

  RgbaImage() = default;
  RgbaImage(int w, int h) : width(w), height(h), rgba(static_cast<size_t>(w) * h * 4) {}
};

std::string encode_png(const RgbaImage& img);

// Tensor [4, H, W] with values in [-1, 1] (VAE output convention) -> RGBA8.
// Denormalization is (x/2+0.5), clamped, then *255 and rounded half-to-even,
// matching diffusers VaeImageProcessor + numpy round.
RgbaImage tensor_n1p1_to_rgba(const float* chw, int height, int width);

}  // namespace umminfer::image2
