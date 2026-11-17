// SPDX-License-Identifier: Apache-2.0
// Omni kind: Qwen3-Omni thinker (text + one optional image) behind the Stage
// protocol. Prompt rendering and image preprocessing follow native/omni_cli.
#include <cuda_runtime.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Same subset the accepted image_preprocess.cpp compiles in, so the decoder
// declarations match its implementation.
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_NO_THREAD_LOCALS
#include "omni/stb_image.h"

#include "omni/image_preprocess.hpp"
#include "omni/omni_engine.hpp"
#include "omni/tokenizer.hpp"
#include "stage_common.hpp"
#include "stage_kind.hpp"

namespace umminfer::stage {
namespace {

constexpr int kDefaultMaxNewTokens = 512;
constexpr int kMinPixels = 3136;
constexpr int kMaxPixels = 12845056;

omni::PreprocessedImage decode_image(const std::string& url) {
  const std::string bytes = load_image_bytes(url);
  int width = 0;
  int height = 0;
  int channels = 0;
  stbi_uc* pixels = stbi_load_from_memory(
      reinterpret_cast<const stbi_uc*>(bytes.data()),
      static_cast<int>(bytes.size()), &width, &height, &channels, 3);
  if (pixels == nullptr) {
    throw ProtocolError(std::string("image decode failed: ") + stbi_failure_reason());
  }
  std::vector<uint8_t> rgb(pixels, pixels + static_cast<size_t>(width) * height * 3);
  stbi_image_free(pixels);
  return omni::preprocess_rgb(rgb.data(), height, width, kMinPixels, kMaxPixels);
}

// Expands the single image_pad between vision_start and vision_end, exactly as
// native/omni_cli does.
std::vector<int> expand_image_pads(std::vector<int> ids, int image_tokens) {
  std::vector<int> expanded;
  expanded.reserve(ids.size() + static_cast<size_t>(image_tokens));
  bool inserted = false;
  for (int id : ids) {
    expanded.push_back(id);
    if (!inserted && id == omni::Qwen2BpeTokenizer::kVisionStart) {
      expanded.insert(expanded.end(), image_tokens,
                      omni::Qwen2BpeTokenizer::kImagePad);
      inserted = true;
    }
  }
  if (!inserted) throw ProtocolError("vision_start not found in the rendered prompt");
  return expanded;
}

}  // namespace

const char* compiled_kind() { return "omni"; }

CancellableComputeFn make_compute(const StageOptions& options) {
  const int max_new_tokens =
      options.max_new_tokens > 0 ? options.max_new_tokens : kDefaultMaxNewTokens;
  auto tokenizer = std::make_shared<omni::Qwen2BpeTokenizer>(
      options.model_dir + "/vocab.json", options.model_dir + "/merges.txt",
      options.model_dir + "/tokenizer_config.json");
  auto engine = std::make_shared<omni::OmniEngine>(
      options.model_dir, options.gpu,
      options.max_seq > 0 ? options.max_seq : 2048);

  const int gpu = options.gpu;
  return [tokenizer, engine, max_new_tokens, gpu](
             const Value& work, const AbortCheck& aborted) {
    const auto started = std::chrono::steady_clock::now();
    // The compute function runs on the Stage scheduler thread, whose CUDA
    // current device defaults to 0. The engine's buffers belong to --gpu.
    if (cudaSetDevice(gpu) != cudaSuccess) {
      throw ProtocolError("cudaSetDevice(" + std::to_string(gpu) + ") failed");
    }
    AbortTracker tracker(aborted);
    const OmniWork input = parse_omni_work(work);
    std::unique_ptr<omni::PreprocessedImage> image;
    if (!input.image_url.empty()) {
      image = std::make_unique<omni::PreprocessedImage>(decode_image(input.image_url));
    }
    std::vector<int> ids =
        tokenizer->encode(render_omni_chat(input.text, image != nullptr));
    if (image != nullptr) {
      ids = expand_image_pads(std::move(ids), image->rows() / 4);
    }
    omni::OmniEngine::Result result;
    try {
      result = engine->generate(ids, image.get(), max_new_tokens,
                                [&tracker] { return tracker(); });
    } catch (const omni::RequestAborted& error) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - started);
      std::cerr << "omni request aborted boundary=\"" << error.what() << "\""
                << " abort_epoch_ms=" << tracker.detected_epoch_ms()
                << " exit_epoch_ms=" << epoch_millis()
                << " compute_ms=" << elapsed.count() << "\n";
      throw RequestAborted(error.what());
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started);
    std::cerr << "omni request prompt_tokens=" << result.prefill_tokens
              << " image_tokens=" << result.image_tokens
              << " generated_tokens=" << result.token_ids.size()
              << " chars=" << result.text.size()
              << " prefill_ms=" << std::fixed << std::setprecision(1)
              << result.prefill_ms
              << " decode_ms=" << std::fixed << std::setprecision(1)
              << result.decode_ms
              << " elapsed_ms=" << elapsed.count() << "\n";
    return omni_result(result.text, static_cast<int>(result.token_ids.size()),
                       result.prefill_ms, result.decode_ms);
  };
}

}  // namespace umminfer::stage
