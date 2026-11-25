// SPDX-License-Identifier: Apache-2.0
// Wan2.2 AutoencoderKLWan decoder configuration and native FP32 CUDA entry point.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
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

// Exact temporal upsampling factor from temperal_downsample (4 for Wan2.2).
inline int64_t wan_vae_temporal_factor(const WanVaeConfig& config) {
  int64_t factor = 1;
  for (bool up : config.temperal_downsample) {
    if (up) factor *= 2;
  }
  return factor;
}

// Causal Wan2.2 VAE emits 1 frame for latent step 0 and `temporal_factor`
// frames for every subsequent latent step.
inline int64_t wan_vae_decoded_frame_count(int64_t latent_t,
                                           int64_t temporal_factor) {
  if (latent_t <= 0 || temporal_factor <= 0) return 0;
  return 1 + (latent_t - 1) * temporal_factor;
}

inline int64_t wan_vae_chunk_frame_count(int64_t latent_index,
                                         int64_t temporal_factor) {
  if (latent_index < 0 || temporal_factor <= 0) return 0;
  return latent_index == 0 ? 1 : temporal_factor;
}

inline int64_t wan_vae_chunk_frame_offset(int64_t latent_index,
                                          int64_t temporal_factor) {
  if (latent_index <= 0 || temporal_factor <= 0) return 0;
  return 1 + (latent_index - 1) * temporal_factor;
}

// Returns true when a 3D convolution is strictly 1x1x1 pointwise, in which
// case its [C, T*H*W] input is already identical to the im2col column matrix
// and can be passed directly to cuBLAS SGEMM without allocating `columns`.
inline constexpr bool wan_conv3d_is_pointwise(int64_t kd, int64_t kh,
                                              int64_t kw) {
  return kd == 1 && kh == 1 && kw == 1;
}

// Default cap on idle cached blocks in the Wan2.2 3D-VAE GPU scratch pool.
inline constexpr size_t kWanVaePoolDefaultMaxCachedBytes = 1536ULL << 20;

// Deterministic best-fit reuse and bounded-cap eviction policy for Wan2.2
// 3D-VAE GPU scratch allocations. Shared between CUDA runtime and host tests.
struct WanVaePoolPolicy {
  size_t max_cached_bytes = kWanVaePoolDefaultMaxCachedBytes;
  size_t max_oversize_num = 5;  // 1.25x upper bound (5 / 4)
  size_t max_oversize_den = 4;

  // Overflow-safe upper bound floor(requested_bytes * num / den), clamped to
  // at least requested_bytes when requested_bytes > 0.
  size_t max_reusable_bytes(size_t requested_bytes) const {
    if (requested_bytes == 0 || max_oversize_den == 0) return 0;
    if (max_oversize_num <= max_oversize_den) return requested_bytes;
    const size_t max_sz = std::numeric_limits<size_t>::max();
    const size_t q = requested_bytes / max_oversize_den;
    const size_t r = requested_bytes % max_oversize_den;
    if (q > max_sz / max_oversize_num) return max_sz;
    const size_t high = q * max_oversize_num;
    if (r > max_sz / max_oversize_num) return max_sz;
    const size_t low = (r * max_oversize_num) / max_oversize_den;
    if (high > max_sz - low) return max_sz;
    return high + low;
  }

  bool can_reuse(size_t requested_bytes, size_t block_bytes) const {
    if (requested_bytes == 0) return false;
    return block_bytes >= requested_bytes &&
           block_bytes <= max_reusable_bytes(requested_bytes);
  }

  // Picks the smallest idle block within [requested_bytes, 1.25 * requested].
  // Ties prefer the earliest index.
  std::optional<size_t> select_best_fit(
      const std::vector<size_t>& idle_block_bytes,
      size_t requested_bytes) const {
    if (requested_bytes == 0) return std::nullopt;
    std::optional<size_t> best;
    size_t best_bytes = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < idle_block_bytes.size(); ++i) {
      const size_t candidate = idle_block_bytes[i];
      if (can_reuse(requested_bytes, candidate) && candidate < best_bytes) {
        best = i;
        best_bytes = candidate;
      }
    }
    return best;
  }

  struct ReleasePlan {
    bool cache_incoming = false;
    size_t evict_prefix_count = 0;
  };

  // Plans whether to cache a freed block of `incoming_bytes` and how many
  // oldest idle blocks (leading prefix) to evict so cached bytes <= cap.
  ReleasePlan plan_release(const std::vector<size_t>& idle_block_bytes,
                           size_t current_cached_bytes,
                           size_t incoming_bytes) const {
    ReleasePlan plan;
    if (incoming_bytes == 0 || incoming_bytes > max_cached_bytes) {
      return plan;
    }
    plan.cache_incoming = true;
    size_t projected = current_cached_bytes;
    while (plan.evict_prefix_count < idle_block_bytes.size() &&
           projected > max_cached_bytes - incoming_bytes) {
      const size_t evicted = idle_block_bytes[plan.evict_prefix_count];
      projected = evicted >= projected ? 0 : (projected - evicted);
      ++plan.evict_prefix_count;
    }
    return plan;
  }
};

struct WanVaeOutput {
  // FP32 NCHW video with one batch item, ordered channel, time, height, width.
  std::vector<int64_t> shape;
  std::vector<float> data;
  double decode_ms = 0.0;
  size_t peak_native_device_bytes = 0;
};

struct WanVaeEncoded {
  // Normalized FP32 NCHW latent, always [1, z_dim, 1, height/16, width/16].
  std::vector<int64_t> shape;
  std::vector<float> data;
  double encode_ms = 0.0;
  size_t peak_native_device_bytes = 0;
};

#ifdef UMM_VIDEO_CUDA
// Loads decoder.* and post_quant_conv.* from diffusion_pytorch_model.safetensors.
// latent_shape must be [1, z_dim, time, height, width]. No TF32 is used.
WanVaeOutput decode_wan_vae_fp32(const std::filesystem::path& vae_dir,
                                 const std::vector<float>& latent,
                                 const std::vector<int64_t>& latent_shape,
                                 int gpu);

// Single-frame encoder, mathematically identical to Diffusers
// AutoencoderKLWan.encode on a T=1 input: patchify -> encoder -> quant_conv ->
// posterior mean (first z_dim channels) -> (z - latents_mean) / latents_std.
// rgb_nchw is FP32 [1, 3, 1, height, width] in [-1, 1] and both spatial sizes
// must be divisible by 16. No TF32 is used.
WanVaeEncoded encode_wan_vae_image_fp32(const std::filesystem::path& vae_dir,
                                        const std::vector<float>& rgb_nchw,
                                        int height, int width, int gpu);
#endif

}  // namespace umminfer::video
