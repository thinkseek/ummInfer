// SPDX-License-Identifier: Apache-2.0
#include "engine.hpp"

#include "cuda_kernels.hpp"
#include "scheduler.hpp"
#if defined(UMM_VIDEO_CUDA) && defined(UMM_VIDEO_WAN_FULL)
#include "umt5_encoder.hpp"
#include "umt5_tokenizer.hpp"
#include "wan_dit.hpp"
#include "wan_vae.hpp"
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// stb_image is header-only; the single implementation lives here.
#define STB_IMAGE_IMPLEMENTATION
#include "common/stb_image.h"

namespace umminfer::video {
namespace {

constexpr size_t kMaxInitImageBytes = 64u << 20;

struct RgbImage {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> pixels;  // interleaved RGB
};

RgbImage decode_init_image(const std::string& bytes) {
  if (bytes.empty() || bytes.size() > kMaxInitImageBytes) {
    throw std::invalid_argument("init image bytes are empty or too large");
  }
  int width = 0;
  int height = 0;
  int components = 0;
  stbi_uc* data = stbi_load_from_memory(
      reinterpret_cast<const stbi_uc*>(bytes.data()),
      static_cast<int>(bytes.size()), &width, &height, &components, 3);
  if (data == nullptr || width <= 0 || height <= 0) {
    throw std::invalid_argument(
        std::string("failed to decode init image: ") +
        (stbi_failure_reason() == nullptr ? "unknown error"
                                          : stbi_failure_reason()));
  }
  RgbImage image;
  image.width = width;
  image.height = height;
  image.pixels.assign(data, data + static_cast<size_t>(width) * height * 3);
  stbi_image_free(data);
  return image;
}

// Bilinear resize with half-pixel centres and clamped edges, then scaled to
// [-1, 1] and written as [1, 3, 1, height, width].
std::vector<float> resize_image_to_nchw(const RgbImage& image, int width,
                                        int height) {
  const size_t plane = static_cast<size_t>(width) * height;
  std::vector<float> output(3 * plane);
  const float scale_x = static_cast<float>(image.width) / width;
  const float scale_y = static_cast<float>(image.height) / height;
  for (int y = 0; y < height; ++y) {
    float source_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
    source_y = std::clamp(source_y, 0.0f, static_cast<float>(image.height - 1));
    const int y0 = static_cast<int>(source_y);
    const int y1 = std::min(y0 + 1, image.height - 1);
    const float wy = source_y - static_cast<float>(y0);
    for (int x = 0; x < width; ++x) {
      float source_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
      source_x = std::clamp(source_x, 0.0f, static_cast<float>(image.width - 1));
      const int x0 = static_cast<int>(source_x);
      const int x1 = std::min(x0 + 1, image.width - 1);
      const float wx = source_x - static_cast<float>(x0);
      const size_t base = static_cast<size_t>(y) * width + x;
      for (int c = 0; c < 3; ++c) {
        const auto at = [&](int sy, int sx) {
          return static_cast<float>(
              image.pixels[(static_cast<size_t>(sy) * image.width + sx) * 3 +
                           c]);
        };
        const float top = at(y0, x0) * (1.0f - wx) + at(y0, x1) * wx;
        const float bottom = at(y1, x0) * (1.0f - wx) + at(y1, x1) * wx;
        const float value = (top * (1.0f - wy) + bottom * wy) / 127.5f - 1.0f;
        output[static_cast<size_t>(c) * plane + base] = value;
      }
    }
  }
  return output;
}

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

struct VideoEngine::WanCudaState {
#if defined(UMM_VIDEO_CUDA) && defined(UMM_VIDEO_WAN_FULL)
  Umt5Tokenizer tokenizer;
  Umt5Encoder text_encoder;
  WanDit dit;

  WanCudaState(const std::filesystem::path& root, int gpu)
      : tokenizer(root / "tokenizer" / "tokenizer.json"),
        text_encoder(root / "text_encoder", gpu),
        dit(root, gpu) {}
#endif
};

VideoEngine::~VideoEngine() = default;

VideoEngine::VideoEngine(int gpu, const std::string& model_dir,
                         VideoBackendMode mode, std::string ffmpeg_bin)
    : gpu_(gpu),
      model_dir_(model_dir),
      mode_(mode),
      ffmpeg_bin_(std::move(ffmpeg_bin)) {
  config_ = load_wan_model_config(model_dir);

  if (mode_ == VideoBackendMode::kCuda) {
#if !defined(UMM_VIDEO_CUDA)
    throw std::runtime_error(
        "umm-native-stage-video was built without CUDA; rebuild with "
        "-DUMM_STAGE_CUDA=ON on a CUDA GPU host, or pass --video-backend "
        "contract-stub for offline contract testing only");
#elif !defined(UMM_VIDEO_WAN_FULL)
    validate_wan_checkpoint_paths(model_dir);
    throw std::runtime_error(
        "native C++/CUDA Wan2.2 full pipeline requires UMM_VIDEO_WAN_FULL=1");
#else
    try {
      validate_wan_checkpoint_paths(model_dir);
    } catch (const std::invalid_argument& err) {
      throw std::runtime_error(err.what());
    }
    cuda_check_video(cudaSetDevice(gpu_), "cudaSetDevice");
    wan_cuda_ = std::make_unique<WanCudaState>(
        std::filesystem::path(model_dir_), gpu_);
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

  if (mode_ == VideoBackendMode::kCuda) {
#if !defined(UMM_VIDEO_CUDA) || !defined(UMM_VIDEO_WAN_FULL)
    throw std::runtime_error("native Wan2.2 CUDA pipeline is not enabled in this build");
#else
    cuda_check_video(cudaSetDevice(gpu_), "cudaSetDevice");
    const auto update_peak_vram = [&peak_gpu_bytes](size_t candidate) {
      if (candidate > peak_gpu_bytes) peak_gpu_bytes = candidate;
      size_t free_bytes = 0;
      size_t total_bytes = 0;
      if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess &&
          total_bytes >= free_bytes) {
        const size_t used = total_bytes - free_bytes;
        if (used > peak_gpu_bytes) peak_gpu_bytes = used;
      }
    };

    check_abort(abort, "aborted before text encoding");
    const Umt5TokenBatch pos_tokens =
        wan_cuda_->tokenizer.encode(params.prompt, 512, true);
    const Umt5EncoderOutput pos_ctx = wan_cuda_->text_encoder.encode(pos_tokens);
    update_peak_vram(pos_ctx.peak_native_device_bytes);

    const bool use_cfg = params.guidance_scale > 1.0f;
    std::vector<float> neg_hidden;
    int neg_len = 512;
    if (use_cfg) {
      check_abort(abort, "aborted before negative text encoding");
      const Umt5TokenBatch neg_tokens =
          wan_cuda_->tokenizer.encode(params.negative_prompt, 512, true);
      const Umt5EncoderOutput neg_ctx = wan_cuda_->text_encoder.encode(neg_tokens);
      update_peak_vram(neg_ctx.peak_native_device_bytes);
      neg_hidden = neg_ctx.hidden_states;
      neg_len = static_cast<int>(neg_tokens.ids.size());
    }
    const int pos_len = static_cast<int>(pos_tokens.ids.size());

    const int channels = std::max(1, config_.latent_channels);
    const int latent_t =
        1 + (params.num_frames - 1) / std::max(1, config_.temporal_stride);
    const int latent_h = std::max(1, params.height / 16);
    const int latent_w = std::max(1, params.width / 16);
    const size_t latent_count = static_cast<size_t>(channels) *
                                static_cast<size_t>(latent_t) *
                                static_cast<size_t>(latent_h) *
                                static_cast<size_t>(latent_w);

    // Wan2.2-TI2V-5B image conditioning: encode the init image once, then pin
    // latent frame 0 to it before every DiT evaluation.
    const size_t latent_plane =
        static_cast<size_t>(latent_h) * static_cast<size_t>(latent_w);
    const bool has_init_image = !params.init_image_bytes.empty();
    std::vector<float> image_latent;
    if (has_init_image) {
      check_abort(abort, "aborted before init image encode");
      const RgbImage image = decode_init_image(params.init_image_bytes);
      const std::vector<float> rgb =
          resize_image_to_nchw(image, params.width, params.height);
      const WanVaeEncoded encoded = encode_wan_vae_image_fp32(
          std::filesystem::path(model_dir_) / "vae", rgb, params.height,
          params.width, gpu_);
      update_peak_vram(encoded.peak_native_device_bytes);
      if (encoded.shape.size() != 5 || encoded.shape[0] != 1 ||
          encoded.shape[1] != channels || encoded.shape[2] != 1 ||
          encoded.shape[3] != latent_h || encoded.shape[4] != latent_w) {
        throw std::runtime_error("unexpected Wan2.2 VAE encoder output shape");
      }
      image_latent = encoded.data;
    }
    const auto pin_first_frame = [&](std::vector<float>& target) {
      for (int c = 0; c < channels; ++c) {
        std::copy(
            image_latent.begin() + static_cast<std::ptrdiff_t>(c) * latent_plane,
            image_latent.begin() +
                static_cast<std::ptrdiff_t>(c + 1) * latent_plane,
            target.begin() +
                static_cast<std::ptrdiff_t>(c) * latent_t * latent_plane);
      }
    };

    std::mt19937_64 rng(params.seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> latents(latent_count);
    for (size_t i = 0; i < latent_count; ++i) {
      latents[i] = dist(rng);
    }
    if (has_init_image) pin_first_frame(latents);

    // Wan2.2-TI2V-5B ships UniPCMultistepScheduler with solver_order=2,
    // solver_type="bh2", predict_x0=true, lower_order_final=true and
    // final_sigmas_type="zero"; the solver below replays that update exactly.
    UniPCFlowMatchSolver solver;
    solver.reset(schedule);

    for (int step = 0; step < params.steps; ++step) {
      check_abort(abort, "aborted before step " + std::to_string(step + 1) +
                             "/" + std::to_string(params.steps));
      const float t_val = schedule.timesteps[static_cast<size_t>(step)];
      if (has_init_image) pin_first_frame(latents);
      WanDitOutput cond = wan_cuda_->dit.forward(
          latents.data(), latent_t, latent_h, latent_w, t_val,
          pos_ctx.hidden_states.data(), pos_len, has_init_image);
      update_peak_vram(cond.stats.peak_vram_bytes);

      std::vector<float> velocity = std::move(cond.velocity);
      if (use_cfg) {
        check_abort(abort, "aborted mid-step CFG " + std::to_string(step + 1) +
                               "/" + std::to_string(params.steps));
        WanDitOutput uncond = wan_cuda_->dit.forward(
            latents.data(), latent_t, latent_h, latent_w, t_val,
            neg_hidden.data(), neg_len, has_init_image);
        update_peak_vram(uncond.stats.peak_vram_bytes);
        const float gs = params.guidance_scale;
        for (size_t i = 0; i < latent_count; ++i) {
          velocity[i] = uncond.velocity[i] + gs * (velocity[i] - uncond.velocity[i]);
        }
      }

      solver.step(step, latents.data(), velocity.data(), latent_count);
      if (has_init_image) pin_first_frame(latents);
      if (params.step_delay_ms > 0) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(params.step_delay_ms));
      }
      check_abort(abort, "aborted at step " + std::to_string(step + 1) + "/" +
                             std::to_string(params.steps));
    }

    if (has_init_image) pin_first_frame(latents);
    check_abort(abort, "aborted before VAE decode");
    const std::vector<int64_t> latent_shape = {
        1, channels, latent_t, latent_h, latent_w};
    const WanVaeOutput decoded = decode_wan_vae_fp32(
        std::filesystem::path(model_dir_) / "vae", latents, latent_shape, gpu_);
    update_peak_vram(decoded.peak_native_device_bytes);
    if (decoded.shape.size() != 5 || decoded.shape[0] != 1 ||
        decoded.shape[1] != 3 || decoded.shape[2] != params.num_frames ||
        decoded.shape[3] != params.height || decoded.shape[4] != params.width) {
      throw std::runtime_error("unexpected Wan2.2 VAE decoded output shape");
    }

    const int64_t out_t = decoded.shape[2];
    const int64_t out_h = decoded.shape[3];
    const int64_t out_w = decoded.shape[4];
    rgb_frames.resize(static_cast<size_t>(out_t));
    for (int64_t t = 0; t < out_t; ++t) {
      auto& frame = rgb_frames[static_cast<size_t>(t)];
      frame.resize(static_cast<size_t>(out_h * out_w * 3));
      for (int64_t y = 0; y < out_h; ++y) {
        for (int64_t x = 0; x < out_w; ++x) {
          for (int64_t c = 0; c < 3; ++c) {
            const size_t src = static_cast<size_t>(
                ((c * out_t + t) * out_h + y) * out_w + x);
            const float clamped =
                std::max(-1.0f, std::min(1.0f, decoded.data[src]));
            frame[static_cast<size_t>((y * out_w + x) * 3 + c)] =
                static_cast<uint8_t>(std::lround((clamped + 1.0f) * 127.5f));
          }
        }
      }
    }
#endif
  } else if (mode_ == VideoBackendMode::kCudaScaffold) {
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
