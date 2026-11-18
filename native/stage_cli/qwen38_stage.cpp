// SPDX-License-Identifier: Apache-2.0
// qwen38 kind: Qwen3.8-27B (text + one optional image) behind the Stage
// protocol. Prompt rendering and image preprocessing follow native/qwen38_cli.
//
// Deltas: every generated token is detokenized and the new suffix is sent as a
// stream chunk {"text": ...}. The HTTP layer forwards them as SSE deltas and
// still receives the aggregate result at completion.
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
#include "qwen38/stb_image.h"

#include "qwen38/engine.hpp"
#include "qwen38/image_preprocess.hpp"
#include "qwen38/tokenizer.hpp"
#include "stage_common.hpp"
#include "stage_kind.hpp"

namespace umminfer::stage {
namespace {

constexpr int kDefaultMaxNewTokens = 256;
constexpr int kMinPixels = 65536;
constexpr int kMaxPixels = 16777216;

qwen38::PreprocessedImage decode_image(const std::string& url) {
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
  return qwen38::preprocess_rgb(rgb.data(), height, width, kMinPixels, kMaxPixels);
}

}  // namespace

const char* compiled_kind() { return "qwen38"; }

CancellableComputeFn make_compute(const StageOptions& options) {
  const int max_new_tokens =
      options.max_new_tokens > 0 ? options.max_new_tokens : kDefaultMaxNewTokens;
  auto tokenizer = std::make_shared<qwen38::Qwen2BpeTokenizer>(
      options.model_dir + "/vocab.json", options.model_dir + "/merges.txt",
      options.model_dir + "/tokenizer_config.json");
  // batch_size > 1 enables continuous batching; the engine owns one batch
  // thread and is safe to call from several scheduler workers.
  const int batch_size = options.batch_size > 1 ? options.batch_size : 1;
  auto engine = std::make_shared<qwen38::Engine>(
      options.model_dir, options.gpu,
      options.max_seq > 0 ? options.max_seq : 8192, batch_size);

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
    const OmniWork input = parse_qwen38_work(work);
    std::unique_ptr<qwen38::PreprocessedImage> image;
    int image_tokens = 0;
    if (!input.image_url.empty()) {
      image = std::make_unique<qwen38::PreprocessedImage>(decode_image(input.image_url));
      if (image->grid_t != 1) {
        throw ProtocolError("qwen38 stage accepts still images only");
      }
      image_tokens = image->rows() / 4;
    }
    std::vector<int> ids =
        tokenizer->encode(qwen38::render_chat_prompt(input.text, image != nullptr,
                                                     /*enable_thinking=*/false));
    if (image != nullptr) {
      ids = qwen38::expand_image_pads(ids, image_tokens);
    }

    // decode() rebuilds the whole text, so the emitted suffix is exactly the
    // new bytes. Byte-level BPE only appends, which keeps this monotonic.
    std::vector<int> generated;
    std::string emitted;
    // Copy the sink and id bound to this compute call. In batched mode the
    // callback fires on the engine's batch thread, where the scheduler's
    // thread-locals are not bound.
    const StreamFn stream = bound_stream_fn();
    const std::string stream_id = bound_request_id();
    auto emit_token = [&tokenizer, &generated, &emitted, stream,
                       stream_id](int id) {
      generated.push_back(id);
      const std::string text = tokenizer->decode(generated);
      const size_t safe = utf8_complete_prefix(text);
      if (safe <= emitted.size()) return;
      Value::Map chunk;
      chunk["text"] = Value(text.substr(emitted.size(), safe - emitted.size()));
      Value value(std::move(chunk));
      if (stream) {
        stream(stream_id, std::move(value));
      } else {
        emit_stage_stream(std::move(value));
      }
      emitted = text.substr(0, safe);
    };

    qwen38::GenResult result;
    try {
      if (engine->batch_capacity() > 1) {
        qwen38::BatchRequest request;
        request.prompt_ids = std::move(ids);
        request.image = image.get();
        request.max_new_tokens = max_new_tokens;
        request.abort = [&tracker] { return tracker(); };
        request.on_token = emit_token;
        result = engine->generate_batched(request);
      } else {
        result = engine->generate(ids, image.get(), max_new_tokens, "",
                                  [&tracker] { return tracker(); }, emit_token);
      }
    } catch (const qwen38::GenerationAborted& error) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - started);
      std::cerr << "qwen38 request aborted boundary=\"" << error.what() << "\""
                << " abort_epoch_ms=" << tracker.detected_epoch_ms()
                << " exit_epoch_ms=" << epoch_millis()
                << " compute_ms=" << elapsed.count() << "\n";
      throw RequestAborted(error.what());
    }
    const std::string text = tokenizer->decode(result.ids);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started);
    const double decode_tok_s = result.decode_ms > 0.0
                                    ? result.decode_steps * 1000.0 / result.decode_ms
                                    : 0.0;
    std::cerr << "qwen38 request prompt_tokens=" << result.prompt_tokens
              << " image_tokens=" << result.image_tokens
              << " generated_tokens=" << result.ids.size()
              << " chars=" << text.size()
              << " prefill_ms=" << std::fixed << std::setprecision(1)
              << result.prefill_ms
              << " decode_ms=" << std::fixed << std::setprecision(1)
              << result.decode_ms
              << " decode_tok_per_s=" << std::fixed << std::setprecision(1)
              << decode_tok_s
              << " elapsed_ms=" << elapsed.count() << "\n";
    return omni_result(text, static_cast<int>(result.ids.size()));
  };
}

}  // namespace umminfer::stage
