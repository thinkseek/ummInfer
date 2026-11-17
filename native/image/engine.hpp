// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "tensor.hpp"

#include <string>
#include <vector>

namespace umminfer::image {

struct GenerateParams {
  std::string prompt;
  std::string negative_prompt = " ";  // required to activate true CFG
  uint64_t seed = 0;
  int steps = 50;
  float guidance = 4.0f;
  int width = 512;
  int height = 512;
};

struct GenerateResult {
  std::vector<uint16_t> rgb_u16;  // HWC RGB8 bit patterns in low 8 bits
  int width = 0, height = 0;
  int load_ms = 0, encode_ms = 0, denoise_ms = 0, vae_ms = 0, total_ms = 0;
  size_t peak_gpu_bytes = 0;
};

// Owns the three model components and runs the full pipeline on one GPU.
class ImageEngine {
 public:
  ImageEngine(int gpu_id, const std::string& model_dir);
  GenerateResult generate(const GenerateParams& p);

 private:
  int gpu_;
  std::string model_dir_;
};

}  // namespace umminfer::image
