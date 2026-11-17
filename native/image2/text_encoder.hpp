// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "safetensors_shard.hpp"
#include "tensor.hpp"

#include <memory>
#include <vector>

namespace umminfer::image2 {

// Qwen3-VL text decoder slice used by Qwen-Image 2.1: 36 layers, hidden 4096,
// 32 q / 8 kv heads, head_dim 128, rope theta 5e6, SwiGLU 12288. The vision
// tower and lm_head are never loaded. Returns the last decoder layer's output
// before the final RMSNorm, which is what the DiT was trained on.
class TextEncoder {
 public:
  TextEncoder(CudaRuntime& rt, const std::string& component_dir);

  // Unpadded token ids -> [seq, kHidden] bf16 on the device.
  DeviceTensor forward(const std::vector<int>& ids);

  static constexpr int kLayers = 36;
  static constexpr int kHidden = 4096;
  static constexpr int kInter = 12288;
  static constexpr int kQHeads = 32;
  static constexpr int kKvHeads = 8;
  static constexpr int kHeadDim = 128;

 private:
  CudaRuntime& rt_;
  std::unique_ptr<ShardedSafeTensors> st_;
  DeviceTensor embed_;
  struct LayerW {
    DeviceTensor q_w, k_w, v_w, o_w;
    DeviceTensor q_norm, k_norm;
    DeviceTensor norm1, norm2;
    DeviceTensor gate_w, up_w, down_w;
  };
  std::vector<LayerW> layers_;
};

}  // namespace umminfer::image2
