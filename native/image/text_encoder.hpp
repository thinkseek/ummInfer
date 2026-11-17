// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "safetensors_shard.hpp"
#include "tensor.hpp"

#include <memory>
#include <vector>

namespace umminfer::image {

// Qwen2.5-VL language-model slice used by the image pipeline: 28 decoder
// layers, hidden 3584, 28 q / 4 kv heads, head_dim 128. The vision tower and
// lm_head are never loaded. Produces the post-final-norm hidden states.
class TextEncoder {
 public:
  TextEncoder(CudaRuntime& rt, const std::string& component_dir);

  // Runs the unpadded sequence of token ids and returns hidden states
  // [seq_len, 3584] on the device (caller keeps the DeviceTensor).
  DeviceTensor forward(const std::vector<int>& ids);

  static constexpr int kLayers = 28;
  static constexpr int kHidden = 3584;
  static constexpr int kInter = 18944;
  static constexpr int kQHeads = 28;
  static constexpr int kKvHeads = 4;
  static constexpr int kHeadDim = 128;

 private:
  CudaRuntime& rt_;
  std::unique_ptr<ShardedSafeTensors> st_;
  DeviceTensor embed_;  // [152064, 3584]
  DeviceTensor final_norm_;  // [3584]
  struct LayerW {
    DeviceTensor q_w, q_b, k_w, k_b, v_w, v_b, o_w;
    DeviceTensor norm1, norm2;
    DeviceTensor gate_w, up_w, down_w;
  };
  std::vector<LayerW> layers_;
};

}  // namespace umminfer::image
