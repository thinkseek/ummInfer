// SPDX-License-Identifier: Apache-2.0
#include "packing.hpp"

#include <stdexcept>

namespace umminfer::image {

PackedLatents pack_latents(const float* chw, int channels, int height, int width) {
  if (height % 2 || width % 2) throw std::invalid_argument("latent H/W must be even");
  PackedLatents out;
  out.h2 = height / 2;
  out.w2 = width / 2;
  out.features = channels * 4;
  out.tokens = out.h2 * out.w2;
  out.data.resize(static_cast<size_t>(out.tokens) * out.features);
  // view(b,C,h/2,2,w/2,2).permute(0,2,4,1,3,5): token (iy,ix), feat c*4+dy*2+dx.
  for (int iy = 0; iy < out.h2; ++iy)
    for (int ix = 0; ix < out.w2; ++ix)
      for (int c = 0; c < channels; ++c)
        for (int dy = 0; dy < 2; ++dy)
          for (int dx = 0; dx < 2; ++dx) {
            const size_t src =
                (((static_cast<size_t>(c) * height) + iy * 2 + dy) * width) + ix * 2 + dx;
            const size_t dst =
                ((static_cast<size_t>(iy) * out.w2 + ix) * out.features) + c * 4 + dy * 2 + dx;
            out.data[dst] = chw[src];
          }
  return out;
}

std::vector<float> unpack_latents(const PackedLatents& packed, int channels, int height,
                                  int width) {
  if (packed.features != channels * 4 || packed.h2 != height / 2 ||
      packed.w2 != width / 2)
    throw std::invalid_argument("unpack shape mismatch");
  std::vector<float> chw(static_cast<size_t>(channels) * height * width);
  for (int iy = 0; iy < packed.h2; ++iy)
    for (int ix = 0; ix < packed.w2; ++ix)
      for (int c = 0; c < channels; ++c)
        for (int dy = 0; dy < 2; ++dy)
          for (int dx = 0; dx < 2; ++dx) {
            const size_t src =
                ((static_cast<size_t>(iy) * packed.w2 + ix) * packed.features) +
                c * 4 + dy * 2 + dx;
            const size_t dst =
                (((static_cast<size_t>(c) * height) + iy * 2 + dy) * width) + ix * 2 + dx;
            chw[dst] = packed.data[src];
          }
  return chw;
}

}  // namespace umminfer::image
