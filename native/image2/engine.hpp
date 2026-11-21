// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "png.hpp"
#include "tensor.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace umminfer::image2 {

using AbortCheck = std::function<bool()>;

class RequestAborted : public std::runtime_error {
 public:
  explicit RequestAborted(const std::string& boundary)
      : std::runtime_error(boundary) {}
};

struct GenerateParams {
  std::string prompt;
  uint64_t seed = 0;
  int steps = 40;
  int width = 512;
  int height = 512;
};

struct GenerateResult {
  RgbaImage image;
  int load_ms = 0;
  int encode_ms = 0;
  int denoise_ms = 0;
  int vae_ms = 0;
  int total_ms = 0;
  size_t peak_gpu_bytes = 0;
};

// Owns the three components and runs the full text-to-image pipeline.
class Image2Engine {
 public:
  Image2Engine(int gpu_id, const std::string& model_dir);
  ~Image2Engine();
  Image2Engine(const Image2Engine&) = delete;
  Image2Engine& operator=(const Image2Engine&) = delete;
  GenerateResult generate(const GenerateParams& p, const AbortCheck& abort = {});

 private:
  int gpu_;
  std::string model_dir_;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace umminfer::image2
