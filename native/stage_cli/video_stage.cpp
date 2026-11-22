// SPDX-License-Identifier: Apache-2.0
// Video kind: native C++/CUDA video vertical slice behind the existing C++
// Stage and UnifiedEngine protocol. Reuses the unified video leg inputs,
// artifact map, abort propagation, and resource cleanup without introducing a
// separate modality protocol.
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "stage_common.hpp"
#include "stage_kind.hpp"
#include "video/config.hpp"
#include "video/engine.hpp"

namespace umminfer::stage {

const char* compiled_kind() { return "video"; }

CancellableComputeFn make_compute(const StageOptions& options) {
  const video::VideoBackendMode mode =
      video::parse_video_backend_mode(options.video_backend);
  const std::string ffmpeg_bin =
      options.ffmpeg_bin.empty() ? "ffmpeg" : options.ffmpeg_bin;
  auto engine = std::make_shared<video::VideoEngine>(
      options.gpu, options.model_dir, mode, ffmpeg_bin);

  const int spatial_align = engine->config().spatial_align;
  const int temporal_stride = engine->config().temporal_stride;

  int default_width = 480;
  int default_height = 320;
  if (!options.image_size.empty()) {
    const auto [w, h] =
        video::parse_video_size(options.image_size, default_width,
                                default_height, spatial_align);
    default_width = w;
    default_height = h;
  }

  const int default_fps = options.video_fps > 0 ? options.video_fps : 16;
  const int max_seconds =
      options.video_max_seconds > 0 ? options.video_max_seconds : 10;
  const int max_frames = options.max_frames > 0 ? options.max_frames : 81;
  const int default_steps = options.steps > 0 ? options.steps : 20;
  const int step_delay_ms = options.video_step_delay_ms;

  return [engine, default_width, default_height, spatial_align, default_fps,
          max_seconds, max_frames, temporal_stride, default_steps,
          step_delay_ms](const Value& work, const AbortCheck& aborted) {
    const auto started = std::chrono::steady_clock::now();
    const std::string req_id = bound_request_id();
    AbortTracker tracker(aborted);

    const VideoWork input = parse_video_work(
        work, default_width, default_height, spatial_align,
        /*default_seconds=*/1, max_seconds, default_fps, max_frames,
        temporal_stride, default_steps, /*max_steps=*/100);

    video::GenerateParams params;
    params.prompt = input.prompt;
    params.negative_prompt = input.negative_prompt;
    params.output_format = input.output_format;
    params.width = input.width;
    params.height = input.height;
    params.seconds = input.seconds;
    params.fps = input.fps;
    params.num_frames = input.num_frames;
    params.steps = input.steps;
    params.guidance_scale = input.guidance_scale;
    params.seed = input.seed;
    params.step_delay_ms = step_delay_ms;

    video::GenerateResult generated;
    try {
      generated = engine->generate(params, [&tracker] { return tracker(); });
    } catch (const video::RequestAborted& error) {
      const auto elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - started);
      std::cerr << "video request aborted backend="
                << video::video_backend_mode_name(engine->mode())
                << " request_id=" << req_id << " boundary=\"" << error.what()
                << "\" abort_epoch_ms=" << tracker.detected_epoch_ms()
                << " exit_epoch_ms=" << epoch_millis()
                << " compute_ms=" << elapsed.count() << "\n";
      throw RequestAborted(error.what());
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started);
    std::cerr << "video request backend=" << generated.backend
              << " request_id=" << req_id << " size=" << generated.width << "x"
              << generated.height << " frames=" << generated.frames
              << " fps=" << generated.fps << " steps=" << generated.steps
              << " seed=" << generated.seed
              << " mp4_bytes=" << generated.mp4_bytes.size()
              << " denoise_ms=" << generated.denoise_ms
              << " encode_ms=" << generated.encode_ms
              << " elapsed_ms=" << elapsed.count() << "\n";

    return video_result(std::move(generated.mp4_bytes), generated.width,
                        generated.height, generated.fps, generated.frames,
                        generated.seconds, generated.seed, generated.steps,
                        generated.model, generated.revision, generated.backend,
                        generated.denoise_ms, generated.encode_ms);
  };
}

}  // namespace umminfer::stage
