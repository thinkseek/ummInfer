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

struct EncodedImageOutput {
  std::string png;
  int width = 0;
  int height = 0;
  double load_ms = 0;
  double encode_ms = 0;
  double denoise_ms = 0;
  double vae_ms = 0;
};

template <typename AbortEx, typename RenderFn>
CancellableComputeFn make_image_compute_envelope(const char* backend_label,
                                                 int gpu, int default_steps,
                                                 RenderFn render) {
  return [backend_label, gpu, default_steps, render = std::move(render)](
             const Value& work, const AbortCheck& aborted) {
    const auto started = std::chrono::steady_clock::now();
    const std::string req_id = bound_request_id();
    if (cudaSetDevice(gpu) != cudaSuccess) {
      throw ProtocolError("cudaSetDevice(" + std::to_string(gpu) + ") failed");
    }
    AbortTracker tracker(aborted);
    const ImageWork input = parse_image_work(work);
    const int steps = input.steps > 0 ? input.steps : default_steps;
    EncodedImageOutput out;
    try {
      out = render(input, steps, [&tracker] { return tracker(); });
    } catch (const AbortEx& error) {
      log_stage_abort("image", backend_label, req_id, error.what(), tracker,
                      started);
      throw RequestAborted(error.what());
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started);
    std::cerr << "image request backend=" << backend_label
              << " request_id=" << req_id << " size=" << out.width << "x"
              << out.height << " steps=" << steps << " seed=" << input.seed
              << " png_bytes=" << out.png.size() << " load_ms=" << out.load_ms
              << " encode_ms=" << out.encode_ms
              << " denoise_ms=" << out.denoise_ms << " vae_ms=" << out.vae_ms
              << " elapsed_ms=" << elapsed.count() << "\n";
    return image_result(std::move(out.png), out.width, out.height, input.seed,
                        steps, out.denoise_ms);
  };
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
    return make_image_compute_envelope<image2::RequestAborted>(
        "2.1", gpu, default_steps,
        [engine](const ImageWork& input, int steps, const auto& abort_fn) {
          image2::GenerateParams params;
          params.prompt = input.prompt;
          params.seed = input.seed;
          params.steps = steps;
          params.width = input.width;
          params.height = input.height;
          image2::GenerateResult generated = engine->generate(params, abort_fn);
          EncodedImageOutput out;
          out.png = image2::encode_png(generated.image);
          out.width = generated.image.width;
          out.height = generated.image.height;
          out.load_ms = generated.load_ms;
          out.encode_ms = generated.encode_ms;
          out.denoise_ms = generated.denoise_ms;
          out.vae_ms = generated.vae_ms;
          return out;
        });
  }

  require_model_paths(options.model_dir, backend,
                      {"tokenizer", "text_encoder", "transformer", "vae"});
  auto engine = std::make_shared<image::ImageEngine>(options.gpu, options.model_dir);
  const int default_steps = options.steps > 0 ? options.steps : 50;
  return make_image_compute_envelope<image::RequestAborted>(
      "2512", gpu, default_steps,
      [engine](const ImageWork& input, int steps, const auto& abort_fn) {
        image::GenerateParams params;
        params.prompt = input.prompt;
        params.negative_prompt = input.negative_prompt;
        params.seed = input.seed;
        params.steps = steps;
        params.guidance = input.guidance;
        params.width = input.width;
        params.height = input.height;
        image::GenerateResult generated = engine->generate(params, abort_fn);
        image::RgbImage rgb(generated.width, generated.height);
        for (size_t index = 0; index < generated.rgb_u16.size(); ++index) {
          rgb.rgb[index] = static_cast<uint8_t>(generated.rgb_u16[index]);
        }
        EncodedImageOutput out;
        out.png = image::encode_png(rgb);
        out.width = generated.width;
        out.height = generated.height;
        out.load_ms = generated.load_ms;
        out.encode_ms = generated.encode_ms;
        out.denoise_ms = generated.denoise_ms;
        out.vae_ms = generated.vae_ms;
        return out;
      });
}

}  // namespace umminfer::stage
