// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "safetensors_shard.hpp"
#include "tensor.hpp"

#include <memory>

namespace umminfer::image {

// AutoencoderKLQwenImage decoder slice. Only the t=1 single-frame decode path
// is implemented: every causal 3D conv reduces to a 2D conv on the last time
// slice. Input is a de-normalized [16, h, w] latent; output is [3, H, W].
class VaeDecoder {
 public:
  VaeDecoder(CudaRuntime& rt, const std::string& component_dir);

  // lat_denorm [16,h,w] (already mean/std de-normalized). Returns RGB CHW
  // [3, 8h, 8w] clamped to [-1,1] in a freshly allocated BF16 tensor.
  DeviceTensor decode(const DeviceTensor& lat_denorm, int h, int w);

  static constexpr int kZ = 16;
  static constexpr int kBase = 96;

 private:
  CudaRuntime& rt_;
  std::unique_ptr<ShardedSafeTensors> st_;
};

}  // namespace umminfer::image
