// SPDX-License-Identifier: Apache-2.0
#include "png.hpp"

#include <zlib.h>

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace umminfer::image {
namespace {

uint32_t be32(uint32_t v) {
  return ((v & 0xffu) << 24) | (((v >> 8) & 0xffu) << 16) | (((v >> 16) & 0xffu) << 8) |
         ((v >> 24) & 0xffu);
}

void append_u32(std::string& out, uint32_t v) {
  uint32_t b = be32(v);
  out.append(reinterpret_cast<const char*>(&b), 4);
}

void append_chunk(std::string& out, const char type[4], const std::vector<uint8_t>& data) {
  append_u32(out, static_cast<uint32_t>(data.size()));
  const size_t pos = out.size();
  out.append(type, 4);
  out.append(reinterpret_cast<const char*>(data.data()), data.size());
  uLong crc = crc32(0L, Z_NULL, 0);
  crc = crc32(crc, reinterpret_cast<const Bytef*>(out.data() + pos), 4 + data.size());
  append_u32(out, static_cast<uint32_t>(crc));
}

}  // namespace

std::string encode_png(const RgbImage& img) {
  if (img.width <= 0 || img.height <= 0) throw std::invalid_argument("png: bad dimensions");
  if (img.rgb.size() != static_cast<size_t>(img.width) * img.height * 3)
    throw std::invalid_argument("png: rgb buffer size mismatch");

  // Raw scanlines with a leading filter byte (0 = None).
  const size_t row_bytes = static_cast<size_t>(img.width) * 3;
  std::vector<uint8_t> raw((row_bytes + 1) * img.height);
  for (int y = 0; y < img.height; ++y) {
    raw[y * (row_bytes + 1)] = 0;
    std::memcpy(raw.data() + y * (row_bytes + 1) + 1, img.rgb.data() + y * row_bytes, row_bytes);
  }

  uLong bound = compressBound(raw.size());
  std::vector<uint8_t> compressed(bound);
  // zlib default level 6 matches libpng default compression.
  int rc = compress2(compressed.data(), &bound, raw.data(), raw.size(), Z_DEFAULT_COMPRESSION);
  if (rc != Z_OK) throw std::runtime_error("png: zlib compress failed");
  compressed.resize(bound);

  std::string out;
  static const uint8_t signature[8] = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
  out.append(reinterpret_cast<const char*>(signature), 8);

  // IHDR: width, height, bit depth 8, color type 2 (RGB), rest zero.
  std::vector<uint8_t> ihdr(13, 0);
  uint32_t w = be32(static_cast<uint32_t>(img.width));
  uint32_t h = be32(static_cast<uint32_t>(img.height));
  std::memcpy(ihdr.data(), &w, 4);
  std::memcpy(ihdr.data() + 4, &h, 4);
  ihdr[8] = 8;  // bit depth
  ihdr[9] = 2;  // color type: truecolor RGB
  append_chunk(out, "IHDR", ihdr);
  append_chunk(out, "IDAT", compressed);

  std::vector<uint8_t> empty;
  append_chunk(out, "IEND", empty);
  return out;
}

RgbImage tensor_n1p1_to_rgb(const float* chw, int height, int width) {
  RgbImage out(width, height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < 3; ++c) {
        // VAE clamps to [-1, 1]; diffusers postprocess maps (x*0.5+0.5).
        float v = chw[(static_cast<size_t>(c) * height + y) * width + x] * 0.5f + 0.5f;
        v = std::fmin(1.0f, std::fmax(0.0f, v));
        out.rgb[(static_cast<size_t>(y) * width + x) * 3 + c] =
            static_cast<uint8_t>(std::lrintf(v * 255.0f));
      }
    }
  }
  return out;
}

}  // namespace umminfer::image
