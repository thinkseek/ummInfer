// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "safetensors_shard.hpp"
#include "tensor.hpp"

#include <memory>
#include <vector>

namespace umminfer::image {

// QwenImageTransformer2DModel: 60 joint image/text blocks, inner 3072,
// 24 heads x 128. Inputs are the packed image tokens and projected text
// embeddings; output is the packed velocity prediction [s_img, 64].
class DiT {
 public:
  DiT(CudaRuntime& rt, const std::string& component_dir);

  // img [s_img,64], txt [s_txt,3072] already txt_in-projected is NOT done by
  // caller: pass raw 3584 text embeddings. timestep_scaled is t/1000 (bf16
  // scale already applied by the caller value range). Grid dims gh, gw are the
  // packed latent height/width (latent H/2).
  DeviceTensor forward(DeviceTensor& img_packed, const DeviceTensor& txt_embed3584,
                       float timestep_scaled, int gh, int gw, int txt_len);

  static constexpr int kBlocks = 60;
  static constexpr int kInner = 3072;
  static constexpr int kHeads = 24;
  static constexpr int kHeadDim = 128;
  static constexpr int kMlp = 12288;
  static constexpr int kPacked = 64;
  static constexpr int kJoint = 3584;

 private:
  CudaRuntime& rt_;
  std::unique_ptr<ShardedSafeTensors> st_;
  // Top-level modules.
  DeviceTensor img_in_w, img_in_b, txt_in_w, txt_in_b, txt_norm_w;
  DeviceTensor time_l1_w, time_l1_b, time_l2_w, time_l2_b;
  DeviceTensor norm_out_w, norm_out_b, proj_out_w, proj_out_b;
  struct BlockW {
    DeviceTensor img_mod_w, img_mod_b, txt_mod_w, txt_mod_b;
    DeviceTensor q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b;
    DeviceTensor aq_w, aq_b, ak_w, ak_b, av_w, av_b, aout_w, aout_b;
    DeviceTensor nq, nk, anq, ank;
    DeviceTensor imlp0_w, imlp0_b, imlp2_w, imlp2_b;
    DeviceTensor tmlp0_w, tmlp0_b, tmlp2_w, tmlp2_b;
  };
  std::vector<BlockW> blocks_;
};

}  // namespace umminfer::image
