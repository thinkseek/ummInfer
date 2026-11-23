// SPDX-License-Identifier: Apache-2.0
// Native C++/CUDA video engine vertical slice API. Reuses the unified video
// stage contract without introducing a separate modality protocol.
#pragma once

#include <cstdint>
#include <mutex>
#include <string>

#include "config.hpp"
#include "mp4_encoder.hpp"

namespace umminfer::video {

enum class VideoBackendMode {
  // Default: real native Wan2.2 CUDA weight inference (currently unimplemented
  // and fails loudly; never silently falls back to synthetic frames).
  kCuda,
  // Explicit CUDA kernel/lifecycle/abort/VRAM-cleanup vertical slice on GPU.
  kCudaScaffold,
  // Explicit host-only offline contract testing mode (no GPU required).
  kContractStub,
};

VideoBackendMode parse_video_backend_mode(const std::string& value);
const char* video_backend_mode_name(VideoBackendMode mode);

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
  double denoise_ms = 0.0;
  double encode_ms = 0.0;
  size_t peak_gpu_bytes = 0;
};

class VideoEngine {
 public:
  VideoEngine(int gpu, const std::string& model_dir, VideoBackendMode mode,
              std::string ffmpeg_bin = "ffmpeg");
  ~VideoEngine() = default;

  VideoEngine(const VideoEngine&) = delete;
  VideoEngine& operator=(const VideoEngine&) = delete;

  const WanModelConfig& config() const { return config_; }
  VideoBackendMode mode() const { return mode_; }
  int gpu() const { return gpu_; }

  GenerateResult generate(const GenerateParams& params,
                          const AbortCheck& abort = {});

 private:
  int gpu_ = 0;
  VideoBackendMode mode_ = VideoBackendMode::kCuda;
  std::string ffmpeg_bin_;
  WanModelConfig config_;
  std::mutex mu_;
};

}  // namespace umminfer::video
