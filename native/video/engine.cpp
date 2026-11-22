// SPDX-License-Identifier: Apache-2.0
#include "engine.hpp"

#include "cuda_kernels.hpp"
#include "scheduler.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace umminfer::video {
namespace {

std::string to_lower_copy(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return text;
}

uint64_t fnv1a_hash(const std::string& text, uint64_t seed) {
  uint64_t h = 14695981039346656037ull ^ seed;
  for (unsigned char c : text) {
    h ^= static_cast<uint64_t>(c);
    h *= 1099511628211ull;
  }
  return h;
}

void check_abort(const AbortCheck& abort, const std::string& boundary) {
  if (abort && abort()) {
    throw RequestAborted(boundary);
  }
}

std::vector<float> synthesize_tchw_from_seed(int num_frames, int height,
                                             int width, uint64_t hash,
                                             float latent_energy) {
  const size_t hw = static_cast<size_t>(height) * static_cast<size_t>(width);
  std::vector<float> tchw(static_cast<size_t>(num_frames) * 3u * hw, 0.0f);
  const float phase_r = static_cast<float>((hash & 0xFFu)) / 255.0f * 6.283185f;
  const float phase_g =
      static_cast<float>(((hash >> 8) & 0xFFu)) / 255.0f * 6.283185f;
  const float phase_b =
      static_cast<float>(((hash >> 16) & 0xFFu)) / 255.0f * 6.283185f;
  const float energy_bias = std::tanh(latent_energy * 0.05f) * 0.25f;

  for (int t = 0; t < num_frames; ++t) {
    const float tf =
        num_frames > 1 ? static_cast<float>(t) / static_cast<float>(num_frames - 1)
                       : 0.0f;
    const size_t base = static_cast<size_t>(t) * 3u * hw;
    for (int y = 0; y < height; ++y) {
      const float yf = static_cast<float>(y) / static_cast<float>(height);
      for (int x = 0; x < width; ++x) {
        const float xf = static_cast<float>(x) / static_cast<float>(width);
        const size_t pix = static_cast<size_t>(y) * static_cast<size_t>(width) +
                           static_cast<size_t>(x);
        tchw[base + 0u * hw + pix] =
            std::clamp(std::sin(phase_r + 3.14159f * (xf + tf)) + energy_bias,
                       -1.0f, 1.0f);
        tchw[base + 1u * hw + pix] =
            std::clamp(std::cos(phase_g + 3.14159f * (yf + tf)) - energy_bias,
                       -1.0f, 1.0f);
        tchw[base + 2u * hw + pix] =
            std::clamp(std::sin(phase_b + 3.14159f * (xf - yf + 2.0f * tf)),
                       -1.0f, 1.0f);
      }
    }
  }
  return tchw;
}

std::vector<std::vector<uint8_t>> tchw_to_rgb24_host(
    const std::vector<float>& tchw, int num_frames, int height, int width) {
  const size_t hw = static_cast<size_t>(height) * static_cast<size_t>(width);
  std::vector<std::vector<uint8_t>> frames(static_cast<size_t>(num_frames));
  auto to_u8 = [](float v) -> uint8_t {
    const float clamped = std::clamp(v, -1.0f, 1.0f);
    return static_cast<uint8_t>(std::lrintf((clamped * 0.5f + 0.5f) * 255.0f));
  };
  for (int t = 0; t < num_frames; ++t) {
    auto& out = frames[static_cast<size_t>(t)];
    out.resize(hw * 3u);
    const size_t base = static_cast<size_t>(t) * 3u * hw;
    for (size_t pix = 0; pix < hw; ++pix) {
      out[pix * 3u + 0u] = to_u8(tchw[base + 0u * hw + pix]);
      out[pix * 3u + 1u] = to_u8(tchw[base + 1u * hw + pix]);
      out[pix * 3u + 2u] = to_u8(tchw[base + 2u * hw + pix]);
    }
  }
  return frames;
}

#ifdef UMM_VIDEO_CUDA
struct CudaStreamGuard {
  cudaStream_t stream = nullptr;
  CudaStreamGuard() {
    cuda_check_video(cudaStreamCreate(&stream), "cudaStreamCreate");
  }
  ~CudaStreamGuard() {
    if (stream != nullptr) {
      (void)cudaStreamDestroy(stream);
    }
  }
};
#endif

}  // namespace

VideoBackendMode parse_video_backend_mode(const std::string& value) {
  const std::string norm = to_lower_copy(value);
  if (norm.empty() || norm == "cuda" || norm == "wan-cuda") {
    return VideoBackendMode::kCuda;
  }
  if (norm == "cuda-scaffold" || norm == "cuda_scaffold") {
    return VideoBackendMode::kCudaScaffold;
  }
  if (norm == "contract-stub" || norm == "contract_stub") {
    return VideoBackendMode::kContractStub;
  }
  throw std::invalid_argument(
      "unsupported --video-backend '" + value +
      "': expected cuda, cuda-scaffold, or contract-stub");
}

const char* video_backend_mode_name(VideoBackendMode mode) {
  switch (mode) {
    case VideoBackendMode::kCuda:
      return "native-cuda-wan";
    case VideoBackendMode::kCudaScaffold:
      return "native-cuda-scaffold";
    case VideoBackendMode::kContractStub:
      return "native-cpp-contract-stub";
  }
  return "native-cuda-wan";
}

VideoEngine::VideoEngine(int gpu, const std::string& model_dir,
                         VideoBackendMode mode, std::string ffmpeg_bin)
    : gpu_(gpu), mode_(mode), ffmpeg_bin_(std::move(ffmpeg_bin)) {
  config_ = load_wan_model_config(model_dir);

  if (mode_ == VideoBackendMode::kCuda) {
#ifndef UMM_VIDEO_CUDA
    throw std::runtime_error(
        "umm-native-stage-video was built without CUDA; rebuild with "
        "-DUMM_STAGE_CUDA=ON on a CUDA GPU host, or pass --video-backend "
        "contract-stub for offline contract testing only");
#else
    validate_wan_checkpoint_paths(model_dir);
    throw std::runtime_error(
        "native C++/CUDA Wan2.2 DiT/UMT5/3D-VAE weight forward kernels are not "
        "yet implemented; use --video-backend cuda-scaffold for GPU "
        "lifecycle/kernel/abort verification or python/umminfer_video_stage.py "
        "--backend diffusers for real Wan2.2 generation");
#endif
  }

  if (mode_ == VideoBackendMode::kCudaScaffold) {
#ifndef UMM_VIDEO_CUDA
    throw std::runtime_error(
        "--video-backend cuda-scaffold requires a CUDA build "
        "(-DUMM_STAGE_CUDA=ON / -DUMM_VIDEO_CUDA=ON)");
#else
    validate_wan_checkpoint_paths(model_dir);
    cuda_check_video(cudaSetDevice(gpu_), "cudaSetDevice");
#endif
  }
}

GenerateResult VideoEngine::generate(const GenerateParams& params,
                                     const AbortCheck& abort) {
  std::lock_guard<std::mutex> lock(mu_);
  if (params.prompt.empty()) {
    throw std::invalid_argument("video.prompt must not be empty");
  }
  if (params.output_format != "mp4") {
    throw std::invalid_argument("video.output_format must be mp4");
  }
  if (params.width % config_.spatial_align != 0 ||
      params.height % config_.spatial_align != 0) {
    throw std::invalid_argument(
        "video.size " + std::to_string(params.width) + "x" +
        std::to_string(params.height) + " must be divisible by " +
        std::to_string(config_.spatial_align));
  }
  if (params.num_frames < 1 ||
      (config_.temporal_stride > 1 &&
       (params.num_frames - 1) % config_.temporal_stride != 0)) {
    throw std::invalid_argument(
        "video.num_frames " + std::to_string(params.num_frames) +
        " must satisfy (n - 1) % " + std::to_string(config_.temporal_stride) +
        " == 0");
  }

  FlowMatchConfig sched_cfg;
  sched_cfg.num_train_timesteps = config_.num_train_timesteps;
  sched_cfg.flow_shift = config_.flow_shift;
  sched_cfg.steps = params.steps;
  const FlowMatchSchedule schedule = build_flow_match_schedule(sched_cfg);

  const uint64_t hash = fnv1a_hash(params.prompt, params.seed);
  const auto denoise_started = std::chrono::steady_clock::now();

  std::vector<std::vector<uint8_t>> rgb_frames;
  size_t peak_gpu_bytes = 0;

  if (mode_ == VideoBackendMode::kCudaScaffold) {
#ifndef UMM_VIDEO_CUDA
    throw std::runtime_error("cuda-scaffold requires CUDA build");
#else
    cuda_check_video(cudaSetDevice(gpu_), "cudaSetDevice");
    CudaStreamGuard stream_guard;

    const int channels = std::max(1, config_.latent_channels);
    const int latent_t =
        1 + (params.num_frames - 1) / std::max(1, config_.temporal_stride);
    const int latent_h = std::max(1, params.height / 16);
    const int latent_w = std::max(1, params.width / 16);
    const size_t spatial_temporal =
        static_cast<size_t>(latent_t) * static_cast<size_t>(latent_h) *
        static_cast<size_t>(latent_w);
    const size_t latent_count = static_cast<size_t>(channels) * spatial_temporal;

    std::vector<float> h_latents(latent_count, 0.25f);
    std::vector<float> h_velocity(latent_count, -0.1f);
    std::vector<float> h_mean = config_.latents_mean;
    std::vector<float> h_std = config_.latents_std;
    h_mean.resize(static_cast<size_t>(channels), 0.0f);
    h_std.resize(static_cast<size_t>(channels), 1.0f);

    DeviceBuffer d_latents(latent_count * sizeof(float));
    DeviceBuffer d_velocity(latent_count * sizeof(float));
    DeviceBuffer d_mean(static_cast<size_t>(channels) * sizeof(float));
    DeviceBuffer d_std(static_cast<size_t>(channels) * sizeof(float));

    cuda_check_video(cudaMemcpyAsync(d_latents.data(), h_latents.data(),
                                     d_latents.bytes(), cudaMemcpyHostToDevice,
                                     stream_guard.stream),
                     "cudaMemcpyAsync(latents)");
    cuda_check_video(cudaMemcpyAsync(d_velocity.data(), h_velocity.data(),
                                     d_velocity.bytes(), cudaMemcpyHostToDevice,
                                     stream_guard.stream),
                     "cudaMemcpyAsync(velocity)");
    cuda_check_video(cudaMemcpyAsync(d_mean.data(), h_mean.data(), d_mean.bytes(),
                                     cudaMemcpyHostToDevice,
                                     stream_guard.stream),
                     "cudaMemcpyAsync(mean)");
    cuda_check_video(cudaMemcpyAsync(d_std.data(), h_std.data(), d_std.bytes(),
                                     cudaMemcpyHostToDevice,
                                     stream_guard.stream),
                     "cudaMemcpyAsync(std)");

    for (int step = 0; step < params.steps; ++step) {
      check_abort(abort, "aborted before step " + std::to_string(step + 1) +
                             "/" + std::to_string(params.steps));
      euler_step_f32_cuda(d_latents.as<float>(), d_velocity.as<float>(),
                          schedule.dt[static_cast<size_t>(step)], latent_count,
                          stream_guard.stream);
      cuda_check_video(cudaStreamSynchronize(stream_guard.stream),
                       "cudaStreamSynchronize(step)");
      if (params.step_delay_ms > 0) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(params.step_delay_ms));
      }
      check_abort(abort, "aborted at step " + std::to_string(step + 1) + "/" +
                             std::to_string(params.steps));
    }

    latent_denorm_f32_cuda(d_latents.as<float>(), d_mean.as<float>(),
                           d_std.as<float>(), channels, spatial_temporal,
                           stream_guard.stream);
    cuda_check_video(cudaMemcpyAsync(h_latents.data(), d_latents.data(),
                                     d_latents.bytes(), cudaMemcpyDeviceToHost,
                                     stream_guard.stream),
                     "cudaMemcpyAsync(latents d2h)");
    cuda_check_video(cudaStreamSynchronize(stream_guard.stream),
                     "cudaStreamSynchronize(denorm)");

    const float energy = h_latents.empty() ? 0.0f : h_latents[0];
    const std::vector<float> h_tchw = synthesize_tchw_from_seed(
        params.num_frames, params.height, params.width, hash, energy);
    const size_t rgb_total = static_cast<size_t>(params.num_frames) *
                             static_cast<size_t>(params.height) *
                             static_cast<size_t>(params.width) * 3u;
    DeviceBuffer d_tchw(h_tchw.size() * sizeof(float));
    DeviceBuffer d_rgb(rgb_total);
    cuda_check_video(cudaMemcpyAsync(d_tchw.data(), h_tchw.data(),
                                     d_tchw.bytes(), cudaMemcpyHostToDevice,
                                     stream_guard.stream),
                     "cudaMemcpyAsync(tchw)");
    frames_n1p1_to_rgb24_cuda(d_tchw.as<float>(), d_rgb.as<uint8_t>(),
                              params.num_frames, params.height, params.width,
                              stream_guard.stream);
    std::vector<uint8_t> h_rgb(rgb_total);
    cuda_check_video(cudaMemcpyAsync(h_rgb.data(), d_rgb.data(), d_rgb.bytes(),
                                     cudaMemcpyDeviceToHost,
                                     stream_guard.stream),
                     "cudaMemcpyAsync(rgb d2h)");
    cuda_check_video(cudaStreamSynchronize(stream_guard.stream),
                     "cudaStreamSynchronize(rgb)");

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess &&
        total_bytes >= free_bytes) {
      peak_gpu_bytes = total_bytes - free_bytes;
    }

    const size_t frame_bytes = static_cast<size_t>(params.height) *
                               static_cast<size_t>(params.width) * 3u;
    rgb_frames.resize(static_cast<size_t>(params.num_frames));
    for (int t = 0; t < params.num_frames; ++t) {
      const size_t off = static_cast<size_t>(t) * frame_bytes;
      rgb_frames[static_cast<size_t>(t)].assign(
          h_rgb.begin() + static_cast<std::ptrdiff_t>(off),
          h_rgb.begin() + static_cast<std::ptrdiff_t>(off + frame_bytes));
    }
#endif
  } else {
    // Explicit kContractStub host execution: runs real Wan flow-match schedule
    // and step-level abort checks before encoding real H.264 MP4 via ffmpeg.
    const size_t latent_count =
        static_cast<size_t>(std::max(1, config_.latent_channels)) * 64u;
    std::vector<float> latents(latent_count, 0.25f);
    std::vector<float> velocity(latent_count, -0.1f);
    for (int step = 0; step < params.steps; ++step) {
      check_abort(abort, "aborted before step " + std::to_string(step + 1) +
                             "/" + std::to_string(params.steps));
      euler_step_host(latents.data(), velocity.data(),
                      schedule.dt[static_cast<size_t>(step)], latent_count);
      if (params.step_delay_ms > 0) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(params.step_delay_ms));
      }
      check_abort(abort, "aborted at step " + std::to_string(step + 1) + "/" +
                             std::to_string(params.steps));
    }
    const std::vector<float> tchw = synthesize_tchw_from_seed(
        params.num_frames, params.height, params.width, hash, latents[0]);
    rgb_frames =
        tchw_to_rgb24_host(tchw, params.num_frames, params.height, params.width);
  }

  const auto denoise_done = std::chrono::steady_clock::now();
  const double denoise_ms =
      std::chrono::duration<double, std::milli>(denoise_done - denoise_started)
          .count();

  const auto encode_started = std::chrono::steady_clock::now();
  std::string mp4_bytes = encode_rgb24_frames_mp4(
      rgb_frames, params.width, params.height, params.fps, abort, ffmpeg_bin_);
  const double encode_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - encode_started)
          .count();

  GenerateResult result;
  result.mp4_bytes = std::move(mp4_bytes);
  result.width = params.width;
  result.height = params.height;
  result.fps = params.fps;
  result.frames = params.num_frames;
  result.seconds =
      static_cast<double>(params.num_frames) / static_cast<double>(params.fps);
  result.steps = params.steps;
  result.seed = params.seed;
  result.model = config_.model_id;
  result.revision = config_.revision;
  result.backend = video_backend_mode_name(mode_);
  result.denoise_ms = denoise_ms;
  result.encode_ms = encode_ms;
  result.peak_gpu_bytes = peak_gpu_bytes;
  return result;
}

}  // namespace umminfer::video
