// SPDX-License-Identifier: Apache-2.0
// Image kind: Qwen-Image-2512 behind the Stage protocol. The result is the
// image artifact map the unified engine turns into kArtifact events.
#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <string>

#include "engine.hpp"
#include "png.hpp"
#include "stage_common.hpp"
#include "stage_kind.hpp"

namespace umminfer::stage {

const char* compiled_kind() { return "image"; }

CancellableComputeFn make_compute(const StageOptions& options) {
  auto engine = std::make_shared<image::ImageEngine>(options.gpu, options.model_dir);
  const int default_steps = options.steps > 0 ? options.steps : 50;
  const int gpu = options.gpu;
  return [engine, default_steps, gpu](const Value& work, const AbortCheck& aborted) {
    const auto started = std::chrono::steady_clock::now();
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
      std::cerr << "image request aborted boundary=\"" << error.what() << "\""
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
    std::cerr << "image request size=" << generated.width << "x" << generated.height
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
