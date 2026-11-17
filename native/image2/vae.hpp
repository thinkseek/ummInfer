// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "safetensors_shard.hpp"
#include "tensor.hpp"

#include <memory>

namespace umminfer::image2 {

// AutoencoderKLQwenImage21 decoder slice. At one frame every causal 3D conv
// is a plain 2D conv and the temporal branches are skipped, so only the 2D
// path is implemented. Input is a de-normalized [64, h, w] latent; output is
// [4, 16h, 16w] (RGBA) clamped to [-1, 1].
class VaeDecoder {
 public:
  VaeDecoder(CudaRuntime& rt, const std::string& component_dir);

  DeviceTensor decode(const DeviceTensor& lat_denorm, int h, int w);

  static constexpr int kZ = 64;
  static constexpr int kBase = 144;

  // Loads a conv weight, taking the last temporal slice if the checkpoint
  // stores a 5D CausalConv3d kernel.
  DeviceTensor load_conv_weight(const std::string& name) const;

 private:
  CudaRuntime& rt_;
  std::unique_ptr<ShardedSafeTensors> st_;
};

}  // namespace umminfer::image2
