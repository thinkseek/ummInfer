// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "safetensors_shard.hpp"
#include "tensor.hpp"

#include <memory>
#include <vector>

namespace umminfer::image2 {

// QwenImage21Transformer2DModel: 32 single-stream blocks, inner 4096,
// 32 heads x 128, SwiGLU 12288, one shared modulation projection.
//
// Text-to-image only: the joint sequence is [text tokens, target image
// tokens]. Text tokens attend causally among themselves and are modulated
// from t=0 (causal_condition); target tokens attend the whole sequence and
// use the sampled timestep.
class DiT {
 public:
  DiT(CudaRuntime& rt, const std::string& component_dir);

  // img [gh*gw, 64] packed latents, txt [txt_len, 4096] encoder states.
  // timestep_scaled is t/1000. Returns the target-image velocity [gh*gw, 64].
  DeviceTensor forward(const DeviceTensor& img_packed, const DeviceTensor& txt_embed,
                       float timestep_scaled, int gh, int gw, int txt_len);

  static constexpr int kBlocks = 32;
  static constexpr int kInner = 4096;
  static constexpr int kHeads = 32;
  static constexpr int kHeadDim = 128;
  static constexpr int kMlp = 12288;
  static constexpr int kChannels = 64;
  static constexpr int kTextDim = 4096;

 private:
  CudaRuntime& rt_;
  std::unique_ptr<ShardedSafeTensors> st_;
  DeviceTensor img_in_w;
  DeviceTensor txt_norm_w, txt_in_w, txt_out_w;
  DeviceTensor time_l1_w, time_l2_w;
  DeviceTensor modulation_w;
  DeviceTensor norm_out_w;
  DeviceTensor proj_out_w;
  struct BlockW {
    DeviceTensor q_w, k_w, v_w, out_w;
    DeviceTensor nq, nk;
    DeviceTensor mlp_proj_w, mlp_gate_w, mlp_out_w;
  };
  std::vector<BlockW> blocks_;
};

}  // namespace umminfer::image2
