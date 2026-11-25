// SPDX-License-Identifier: Apache-2.0
// Wan2.2-TI2V-5B transformer configuration, layout helpers, and CUDA forward.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace umminfer::video {

struct WanDitConfig {
  int num_layers = 30;
  int num_attention_heads = 24;
  int attention_head_dim = 128;
  int inner_dim = 3072;
  int ffn_dim = 14336;
  int in_channels = 48;
  int out_channels = 48;
  int patch_t = 1;
  int patch_h = 2;
  int patch_w = 2;
  int text_dim = 4096;
  int freq_dim = 256;
  int rope_max_seq_len = 1024;
  float eps = 1e-6f;
  bool cross_attn_norm = true;
  std::string qk_norm = "rms_norm_across_heads";
};

// model_dir is the Diffusers model root containing transformer/config.json.
WanDitConfig load_wan_dit_config(const std::string& model_dir);

// Diffusers Timesteps: cosine half followed by sine half.
std::vector<float> wan_sinusoidal_timestep_embedding(float timestep,
                                                      int freq_dim);

struct WanRope3D {
  int frames = 0;
  int height = 0;
  int width = 0;
  int head_dim = 0;
  // [frame, y, x, head_dim], with each axis frequency repeated for a complex pair.
  std::vector<float> cos;
  std::vector<float> sin;
};

WanRope3D wan_build_rope_3d(int frames, int height, int width, int head_dim,
                            float theta = 10000.0f);

// Raw patch token layout is [token, patch_t, patch_y, patch_x, channel], exactly
// the proj_out component order before Diffusers' final permute/flatten operations.
std::vector<float> wan_patchify_host(const float* latent_ncfhw, int channels,
                                     int frames, int height, int width);
std::vector<float> wan_unpatchify_host(const float* tokens, int channels,
                                       int frames, int height, int width);

// Attention query tile size used by the CUDA Wan DiT attention loop.
inline constexpr int kWanDitQueryTile = 512;

// Returns min(tile_size, tokens - q_start) for attention query tiling.
int wan_attention_tile_len(int tokens, int q_start,
                           int tile_size = kWanDitQueryTile);

struct WanTextEmbedCacheKey {
  uint64_t hash = 0;
  int text_len = 0;
  int text_dim = 0;
  bool operator==(const WanTextEmbedCacheKey& other) const {
    return hash == other.hash && text_len == other.text_len &&
           text_dim == other.text_dim;
  }
};

// Fast 64-bit FNV-1a content hash over IEEE-754 bytes of [text_len, text_dim]
// plus shape dimensions, used to key the cross-attention K/V cache across steps.
uint64_t wan_hash_text_embeds(const float* data, int text_len, int text_dim);

struct WanDitStats {
  double load_ms = 0.0;
  double forward_ms = 0.0;
  size_t peak_vram_bytes = 0;
  size_t tensor_count = 0;
};

struct WanDitOutput {
  std::vector<float> velocity;
  WanDitStats stats;
};

#ifdef UMM_VIDEO_CUDA
class WanDit {
 public:
  WanDit(const std::filesystem::path& model_dir, int gpu);
  ~WanDit();

  WanDit(const WanDit&) = delete;
  WanDit& operator=(const WanDit&) = delete;

  // latent is [1,48,F,H,W], context is [text_len,4096]. The result is [1,48,F,H,W].
  // With first_frame_zero_timestep the Wan2.2-TI2V-5B per-token timestep
  // conditioning applies: tokens of latent frame 0 use t=0 and every other
  // token uses `timestep`. The default keeps the scalar T2V path unchanged.
  WanDitOutput forward(const float* latent_ncfhw, int frames, int height,
                       int width, float timestep, const float* context,
                       int text_len,
                       bool first_frame_zero_timestep = false);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
#endif

}  // namespace umminfer::video
