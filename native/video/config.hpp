// SPDX-License-Identifier: Apache-2.0
// Configuration, checkpoint inspection, and Wan2.2/DiT video request geometry
// alignment for the native C++/CUDA video stage vertical slice.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace umminfer::video {

constexpr size_t kMaxVideoArtifactBytes = 48u << 20;  // 48 MiB (UnifiedEngine cap)

struct WanModelConfig {
  std::string model_dir;
  std::string model_id;
  std::string model_class = "WanPipeline";
  std::string revision = "UNKNOWN";

  int spatial_align = 32;     // vae_scale_factor_spatial * patch_w
  int temporal_stride = 4;    // vae_scale_factor_temporal
  int latent_channels = 16;   // Wan2.2-TI2V-5B / Wan2.1 latent channels
  int patch_t = 1;
  int patch_h = 2;
  int patch_w = 2;
  int in_channels = 16;
  int out_channels = 16;
  int num_layers = 30;
  double flow_shift = 5.0;
  int num_train_timesteps = 1000;

  std::vector<float> latents_mean;
  std::vector<float> latents_std;
};

// Detects an explicit revision, HF snapshot 40-hex directory name, or UNKNOWN.
std::string detect_model_revision(const std::string& model_dir,
                                  const std::string& explicit_revision = "");

// Parses Wan Diffusers-layout metadata (model_index.json, vae/config.json,
// transformer/config.json, scheduler/scheduler_config.json) when present.
// Throws std::invalid_argument if model_dir does not exist or JSON is malformed.
WanModelConfig load_wan_model_config(const std::string& model_dir,
                                     const std::string& explicit_revision = "");

// Validates that required Wan checkpoint subdirectories/files exist on disk
// before attempting CUDA execution.
void validate_wan_checkpoint_paths(const std::string& model_dir);

// Parses "WIDTHxHEIGHT" (case-insensitive 'x') and validates [min_dim, max_dim]
// and divisibility by `align`. Throws std::invalid_argument on mismatch.
std::pair<int, int> parse_video_size(const std::string& size_str,
                                     int default_width, int default_height,
                                     int align = 32, int min_dim = 128,
                                     int max_dim = 1280);

// Computes a valid Wan/DiT frame count satisfying (n - 1) % temporal_stride == 0
// within [temporal_stride + 1, max_frames].
int align_video_frames(int seconds, int fps, int max_frames,
                       int temporal_stride = 4);

}  // namespace umminfer::video
