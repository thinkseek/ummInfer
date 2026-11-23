// SPDX-License-Identifier: Apache-2.0
// Wan2.2 AutoencoderKLWan decoder configuration and native FP32 CUDA entry point.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace umminfer::video {

struct WanVaeConfig {
  int z_dim = 0;
  int dim = 0;
  int decoder_base_dim = 0;
  int num_res_blocks = 0;
  int out_channels = 0;
  int patch_size = 0;
  bool is_residual = false;
  std::vector<int> dim_mult;
  std::vector<bool> temperal_downsample;
  std::vector<float> attn_scales;
  std::vector<float> latents_mean;
  std::vector<float> latents_std;
};

WanVaeConfig load_wan_vae_config(const std::filesystem::path& vae_dir);

struct WanVaeOutput {
  // FP32 NCHW video with one batch item, ordered channel, time, height, width.
  std::vector<int64_t> shape;
  std::vector<float> data;
  double decode_ms = 0.0;
  size_t peak_native_device_bytes = 0;
};

#ifdef UMM_VIDEO_CUDA
// Loads decoder.* and post_quant_conv.* from diffusion_pytorch_model.safetensors.
// latent_shape must be [1, z_dim, time, height, width]. No TF32 is used.
WanVaeOutput decode_wan_vae_fp32(const std::filesystem::path& vae_dir,
                                 const std::vector<float>& latent,
                                 const std::vector<int64_t>& latent_shape,
                                 int gpu);
#endif

}  // namespace umminfer::video
