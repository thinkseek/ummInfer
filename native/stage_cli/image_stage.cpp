// SPDX-License-Identifier: Apache-2.0
// Image kind: Qwen-Image-2512 (default) or Qwen-Image-2.1 behind the Stage
// protocol. Both backends use stage id "image" and emit the same artifact map.
#include <cuda_runtime.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "image/engine.hpp"
#include "image/png.hpp"
#include "image2/engine.hpp"
#include "image2/png.hpp"
#include "stage_common.hpp"
#include "stage_kind.hpp"

namespace umminfer::stage {
namespace {

void require_model_paths(const std::string& model_dir, ImageBackend backend,
                         std::initializer_list<const char*> rel_paths) {
  const std::filesystem::path root(model_dir);
  for (const char* rel : rel_paths) {
    std::error_code ec;
    if (!std::filesystem::exists(root / rel, ec)) {
      throw std::runtime_error(
          std::string("image backend ") + image_backend_name(backend) +
          " missing required path '" + (root / rel).string() + "'");
    }
  }
}

}  // namespace

const char* compiled_kind() { return "image"; }

const char* stage_usage_flags() {
  return "[--steps N] [--image-backend 2512|2.1] [--size WxH] ";
}

bool parse_stage_flag(const std::string& key, const std::string& value,
                      StageOptions& options) {
  if (key == "steps") {
    options.steps = parse_stage_int(value, "--steps");
    return true;
  }
  if (key == "image-backend") {
    options.image_backend = value;
    return true;
  }
  if (key == "size") {
    options.image_size = value;
    return true;
  }
  return false;
}

CancellableComputeFn make_compute(const StageOptions& options) {
  const ImageBackend backend = parse_image_backend(options.image_backend);
  if (!options.image_size.empty()) {
    (void)parse_supported_image_size(options.image_size);
  }
  const int gpu = options.gpu;

  if (backend == ImageBackend::kQwen21) {
    require_model_paths(options.model_dir, backend,
                        {"processor", "text_encoder", "transformer",
                         "vae/config.json"});
    auto engine =
        std::make_shared<image2::Image2Engine>(options.gpu, options.model_dir);
    const int default_steps = options.steps > 0 ? options.steps : 40;
    return [engine, default_steps, gpu](const Value& work,
                                        const AbortCheck& aborted) {
      const auto started = std::chrono::steady_clock::now();
      const std::string req_id = bound_request_id();
      if (cudaSetDevice(gpu) != cudaSuccess) {
        throw ProtocolError("cudaSetDevice(" + std::to_string(gpu) + ") failed");
      }
      AbortTracker tracker(aborted);
      const ImageWork input = parse_image_work(work);
      image2::GenerateParams params;
      params.prompt = input.prompt;
      params.seed = input.seed;
      params.steps = input.steps > 0 ? input.steps : default_steps;
      params.width = input.width;
      params.height = input.height;

      image2::GenerateResult generated;
      try {
        generated = engine->generate(params, [&tracker] { return tracker(); });
      } catch (const image2::RequestAborted& error) {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - started);
        std::cerr << "image request aborted backend=2.1 request_id=" << req_id
                  << " boundary=\"" << error.what() << "\""
                  << " abort_epoch_ms=" << tracker.detected_epoch_ms()
                  << " exit_epoch_ms=" << epoch_millis()
                  << " compute_ms=" << elapsed.count() << "\n";
        throw RequestAborted(error.what());
      }

      std::string png = image2::encode_png(generated.image);
      const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - started);
      std::cerr << "image request backend=2.1 request_id=" << req_id
                << " size=" << generated.image.width << "x"
                << generated.image.height
                << " steps=" << params.steps << " seed=" << params.seed
                << " png_bytes=" << png.size()
                << " load_ms=" << generated.load_ms
                << " encode_ms=" << generated.encode_ms
                << " denoise_ms=" << generated.denoise_ms
                << " vae_ms=" << generated.vae_ms
                << " elapsed_ms=" << elapsed.count() << "\n";
      return image_result(std::move(png), generated.image.width,
                          generated.image.height, params.seed, params.steps,
                          generated.denoise_ms);
    };
  }

  require_model_paths(options.model_dir, backend,
                      {"tokenizer", "text_encoder", "transformer", "vae"});
  auto engine = std::make_shared<image::ImageEngine>(options.gpu, options.model_dir);
  const int default_steps = options.steps > 0 ? options.steps : 50;
  return [engine, default_steps, gpu](const Value& work, const AbortCheck& aborted) {
    const auto started = std::chrono::steady_clock::now();
    const std::string req_id = bound_request_id();
    if (cudaSetDevice(gpu) != cudaSuccess) {
      throw ProtocolError("cudaSetDevice(" + std::to_string(gpu) + ") failed");
    }
    AbortTracker tracker(aborted);
    const ImageWork input = parse_image_work(work);
    image::GenerateParams params;
    params.prompt = input.prompt;
    params.negative_prompt = input.negative_prompt;
    params.seed = input.seed;
    params.steps = input.steps > 0 ? input.steps : default_steps;
    params.guidance = input.guidance;
    params.width = input.width;
    params.height = input.height;

    image::GenerateResult generated;
    try {
      generated = engine->generate(params, [&tracker] { return tracker(); });
    } catch (const image::RequestAborted& error) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - started);
      std::cerr << "image request aborted backend=2512 request_id=" << req_id
                << " boundary=\"" << error.what() << "\""
                << " abort_epoch_ms=" << tracker.detected_epoch_ms()
                << " exit_epoch_ms=" << epoch_millis()
                << " compute_ms=" << elapsed.count() << "\n";
      throw RequestAborted(error.what());
    }

    image::RgbImage rgb(generated.width, generated.height);
    for (size_t index = 0; index < generated.rgb_u16.size(); ++index) {
      rgb.rgb[index] = static_cast<uint8_t>(generated.rgb_u16[index]);
    }
    std::string png = image::encode_png(rgb);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started);
    std::cerr << "image request backend=2512 request_id=" << req_id
              << " size=" << generated.width << "x" << generated.height
              << " steps=" << params.steps << " seed=" << params.seed
              << " png_bytes=" << png.size()
              << " load_ms=" << generated.load_ms
              << " encode_ms=" << generated.encode_ms
              << " denoise_ms=" << generated.denoise_ms
              << " vae_ms=" << generated.vae_ms
              << " elapsed_ms=" << elapsed.count() << "\n";
    return image_result(std::move(png), generated.width, generated.height,
                        params.seed, params.steps, generated.denoise_ms);
  };
}

}  // namespace umminfer::stage
