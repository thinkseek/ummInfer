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

const char* stage_usage_flags() {
  return "[--video-backend cuda|cuda-scaffold|contract-stub] [--size WxH] "
         "[--fps N] [--max-seconds N] [--max-frames N] [--steps N] "
         "[--step-delay-ms N] [--ffmpeg-bin PATH] [--teacache-thresh F] ";
}

bool parse_stage_flag(const std::string& key, const std::string& value,
                      StageOptions& options) {
  if (key == "video-backend") {
    options.video_backend = value;
    return true;
  }
  if (key == "size") {
    options.image_size = value;
    return true;
  }
  if (key == "fps") {
    options.video_fps = parse_stage_int(value, "--fps");
    return true;
  }
  if (key == "max-seconds") {
    options.video_max_seconds = parse_stage_int(value, "--max-seconds");
    return true;
  }
  if (key == "max-frames") {
    options.max_frames = parse_stage_int(value, "--max-frames");
    return true;
  }
  if (key == "steps") {
    options.steps = parse_stage_int(value, "--steps");
    return true;
  }
  if (key == "step-delay-ms") {
    options.video_step_delay_ms = parse_stage_int(value, "--step-delay-ms");
    return true;
  }
  if (key == "ffmpeg-bin") {
    options.ffmpeg_bin = value;
    return true;
  }
  if (key == "teacache-thresh") {
    options.video_teacache_thresh =
        parse_stage_float_range(value, "--teacache-thresh", 0.0f, 1.0f);
    return true;
  }
  return false;
}

CancellableComputeFn make_compute(const StageOptions& options) {
  const video::VideoBackendMode mode =
      video::parse_video_backend_mode(options.video_backend);
  const std::string ffmpeg_bin =
      options.ffmpeg_bin.empty() ? "ffmpeg" : options.ffmpeg_bin;
  video::VideoEngineConfig engine_cfg;
  engine_cfg.gpu = options.gpu;
  engine_cfg.model_dir = options.model_dir;
  engine_cfg.mode = mode;
  engine_cfg.ffmpeg_bin = ffmpeg_bin;
  engine_cfg.teacache_thresh = options.video_teacache_thresh;
  auto engine = std::make_shared<video::VideoEngine>(engine_cfg);

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
    params.init_image_bytes = input.init_image_bytes;

    video::GenerateResult generated;
    try {
      generated = engine->generate(params, [&tracker] { return tracker(); });
    } catch (const video::RequestAborted& error) {
      log_stage_abort("video", video::video_backend_mode_name(engine->mode()),
                      req_id, error.what(), tracker, started);
      throw RequestAborted(error.what());
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started);
    std::cerr << "video request backend=" << generated.backend
              << " request_id=" << req_id << " size=" << generated.width << "x"
              << generated.height << " frames=" << generated.frames
              << " fps=" << generated.fps << " steps=" << generated.steps
              << " seed=" << generated.seed
              << " i2v=" << (input.init_image_bytes.empty() ? 0 : 1)
              << " teacache_thresh=" << generated.teacache_thresh
              << " teacache_hits=" << generated.teacache_hits << "/"
              << generated.teacache_total_calls
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
