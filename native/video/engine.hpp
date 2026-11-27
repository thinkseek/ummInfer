// SPDX-License-Identifier: Apache-2.0
// Native C++/CUDA video engine vertical slice API. Reuses the unified video
// stage contract without introducing a separate modality protocol.
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "config.hpp"
#include "mp4_encoder.hpp"

namespace umminfer::video {

// Aspect-preserving center-crop (cover) + half-pixel bilinear resize of
// interleaved RGB8 pixels into [-1, 1] NCHW layout [1, 3, 1, dst_h, dst_w].
// When src and dst share the same aspect ratio, this maps border-to-border
// identically to plain bilinear scaling with zero distortion.
std::vector<float> resize_center_crop_rgb_to_nchw(const uint8_t* rgb,
                                                  int src_width, int src_height,
                                                  int dst_width, int dst_height);

enum class VideoBackendMode {
  // Default: real native Wan2.2 C++/CUDA pipeline (UMT5-XXL + DiT + 3D VAE + MP4).
  // Fails loudly when built without CUDA or when weights are missing; never silently
  // falls back to synthetic frames.
  kCuda,
  // Explicit CUDA kernel/lifecycle/abort/VRAM-cleanup vertical slice on GPU.
  kCudaScaffold,
  // Explicit host-only offline contract testing mode (no GPU required).
  kContractStub,
};

VideoBackendMode parse_video_backend_mode(const std::string& value);
const char* video_backend_mode_name(VideoBackendMode mode);

struct VideoEngineConfig {
  int gpu = 0;
  std::string model_dir;
  VideoBackendMode mode = VideoBackendMode::kCuda;
  std::string ffmpeg_bin = "ffmpeg";
  // Opt-in TeaCache L1 threshold in [0.0, 1.0]. 0.0 disables caching and keeps
  // the uncached execution path 100% bit-exact.
  float teacache_thresh = 0.0f;
};

struct GenerateParams {
  std::string prompt;
  std::string negative_prompt;
  std::string output_format = "mp4";
  int width = 480;
  int height = 320;
  int seconds = 1;
  int fps = 16;
  int num_frames = 17;
  int steps = 20;
  float guidance_scale = 5.0f;
  uint64_t seed = 0;
  int step_delay_ms = 0;  // Optional per-step delay for deterministic abort tests.
  // Optional per-request TeaCache threshold override in [0.0, 1.0]. When unset,
  // inherits VideoEngineConfig::teacache_thresh (default 0.0 = off).
  std::optional<float> teacache_thresh;
  // Optional PNG/JPEG/WebP bytes for Wan2.2-TI2V-5B image-to-video. When set,
  // the image is VAE-encoded and latent frame 0 is pinned to it for every
  // DiT evaluation.
  std::string init_image_bytes;
};

struct GenerateResult {
  std::string mp4_bytes;
  int width = 0;
  int height = 0;
  int fps = 0;
  int frames = 0;
  double seconds = 0.0;
  int steps = 0;
  uint64_t seed = 0;
  std::string model;
  std::string revision;
  std::string backend;
  double umt5_ms = 0.0;
  double dit_ms = 0.0;
  double vae_ms = 0.0;
  double denoise_ms = 0.0;
  double encode_ms = 0.0;
  // Internal phase diagnostics. These do not alter the public video result
  // schema; the Stage logs them for controlled latency decomposition.
  double image_decode_ms = 0.0;
  double image_resize_ms = 0.0;
  double vae_encode_ms = 0.0;
  double vae_decode_ms = 0.0;
  double latent_init_ms = 0.0;
  int init_image_width = 0;
  int init_image_height = 0;
  size_t peak_gpu_bytes = 0;
  float teacache_thresh = 0.0f;
  int teacache_hits = 0;
  int teacache_total_calls = 0;
};

class VideoEngine {
 public:
  VideoEngine(int gpu, const std::string& model_dir, VideoBackendMode mode,
              std::string ffmpeg_bin = "ffmpeg",
              float teacache_thresh = 0.0f);
  explicit VideoEngine(const VideoEngineConfig& config);
  ~VideoEngine();

  VideoEngine(const VideoEngine&) = delete;
  VideoEngine& operator=(const VideoEngine&) = delete;

  const WanModelConfig& config() const { return config_; }
  VideoBackendMode mode() const { return mode_; }
  int gpu() const { return gpu_; }
  float teacache_thresh() const { return teacache_thresh_; }

  GenerateResult generate(const GenerateParams& params,
                          const AbortCheck& abort = {});

 private:
  struct WanCudaState;
  int gpu_ = 0;
  std::string model_dir_;
  VideoBackendMode mode_ = VideoBackendMode::kCuda;
  std::string ffmpeg_bin_;
  float teacache_thresh_ = 0.0f;
  WanModelConfig config_;
  std::unique_ptr<WanCudaState> wan_cuda_;
  std::mutex mu_;
};

}  // namespace umminfer::video
