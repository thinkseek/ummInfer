// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <vector>

namespace umminfer::image {

// Latent packing, exact match of QwenImagePipeline._pack_latents /
// _unpack_latents. Latents are [C, H, W] with C=16; 2x2 patches flatten to
// tokens of C*4=64 features. Token (iy, ix) feature layout:
//   feature = c*4 + dy*2 + dx, token = iy * (W/2) + ix.
struct PackedLatents {
  int tokens = 0;   // (H/2) * (W/2)
  int features = 0; // C * 4
  int h2 = 0;
  int w2 = 0;
  std::vector<float> data;  // [tokens, features]
};

// H and W must be even; channels must equal the VAE z_dim (16).
PackedLatents pack_latents(const float* chw, int channels, int height, int width);
// Inverse: returns [channels, height, width].
std::vector<float> unpack_latents(const PackedLatents& packed, int channels, int height, int width);

}  // namespace umminfer::image
