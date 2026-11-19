// SPDX-License-Identifier: Apache-2.0
// Omni kind: Qwen3-Omni thinker (text + one optional image) behind the Stage
// protocol. Prompt rendering and image preprocessing follow native/omni_cli.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
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
constexpr int kMaxPixels = omni::kMaxImagePixels;

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
  const int max_seq = options.max_seq > 0 ? options.max_seq : 2048;
  auto engine = std::make_shared<omni::OmniEngine>(
      options.model_dir, options.gpu, max_seq);

  const int gpu = options.gpu;
  // One engine owns one KV cache, so requests run one at a time. The cache
  // holds the prompt region of the most recent pure-text prefill; any pure-text
  // request (same session, another session sharing a system prompt, or a repeat
  // prompt) reuses the common leading token prefix up to ids.size() - 1 so
  // reuse stays bit-identical to a full prefill.
  struct PrefixCache {
    bool enabled = true;
    std::string last_session;
    std::vector<int> ids;
    int valid = 0;
  };
  auto cache = std::make_shared<PrefixCache>();
  auto cache_mu = std::make_shared<std::mutex>();
  if (const char* env = std::getenv("UMM_OMNI_PREFIX_CACHE")) {
    cache->enabled = std::atoi(env) != 0;
  }
  return [tokenizer, engine, max_new_tokens, max_seq, gpu, cache, cache_mu](
             const Value& work, const AbortCheck& aborted) {
    std::lock_guard<std::mutex> serialize(*cache_mu);
    const auto started = std::chrono::steady_clock::now();
    const std::string req_id = bound_request_id();
    // The compute function runs on the Stage scheduler thread, whose CUDA
    // current device defaults to 0. The engine's buffers belong to --gpu.
    if (cudaSetDevice(gpu) != cudaSuccess) {
      throw ProtocolError("cudaSetDevice(" + std::to_string(gpu) + ") failed");
    }
    AbortTracker tracker(aborted);
    const OmniWork input = parse_omni_work(work);
    const int req_max_new_tokens =
        resolve_max_new_tokens(bound_request_params(), max_new_tokens, max_seq);

    std::unique_ptr<omni::PreprocessedImage> image;
    if (!input.image_url.empty()) {
      image = std::make_unique<omni::PreprocessedImage>(decode_image(input.image_url));
    }
    std::vector<int> ids = tokenizer->encode(
        render_omni_chat(input.messages, image != nullptr));
    if (image != nullptr) {
      ids = expand_image_pads(std::move(ids), image->rows() / 4);
    }
    int prefix = 0;
    if (cache->enabled && image == nullptr && cache->valid > 0 &&
        ids.size() > 1) {
      size_t common = 0;
      const size_t limit = std::min(cache->ids.size(), ids.size());
      while (common < limit && cache->ids[common] == ids[common]) ++common;
      const size_t max_reusable =
          std::min<size_t>(static_cast<size_t>(cache->valid), ids.size() - 1);
      prefix = static_cast<int>(std::min(common, max_reusable));
    }
    // Token streaming: byte-level BPE is additive per token, so buffer only
    // incomplete trailing UTF-8 bytes in O(1) per token.
    std::string pending_utf8;
    auto emit_token = [&tokenizer, &pending_utf8](int id) {
      pending_utf8 += tokenizer->decode({id});
      const size_t safe = utf8_complete_prefix(pending_utf8);
      if (safe == 0) return;
      Value::Map chunk;
      chunk["text"] = Value(pending_utf8.substr(0, safe));
      emit_stage_stream(Value(std::move(chunk)));
      pending_utf8.erase(0, safe);
    };
    omni::OmniEngine::Result result;
    try {
      result = engine->generate(ids, image.get(), req_max_new_tokens,
                                [&tracker] { return tracker(); }, emit_token,
                                prefix);
      if (image == nullptr) {
        cache->last_session = input.session_id;
        cache->ids = ids;
        cache->valid = static_cast<int>(ids.size());
      } else {
        cache->ids.clear();
        cache->valid = 0;
      }
    } catch (const omni::RequestAborted& error) {
      cache->valid = 0;
      const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - started);
      std::cerr << "omni request aborted request_id=" << req_id
                << " boundary=\"" << error.what() << "\""
                << " abort_epoch_ms=" << tracker.detected_epoch_ms()
                << " exit_epoch_ms=" << epoch_millis()
                << " compute_ms=" << elapsed.count() << "\n";
      throw RequestAborted(error.what());
    } catch (const std::exception&) {
      // A failed run may have written part of the KV cache; drop the prefix.
      cache->valid = 0;
      throw;
    }
    {
      const size_t safe = utf8_complete_prefix(pending_utf8);
      if (safe > 0) {
        Value::Map chunk;
        chunk["text"] = Value(pending_utf8.substr(0, safe));
        emit_stage_stream(Value(std::move(chunk)));
      }
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started);
    const char* finish_reason = result.stopped_by_eos ? "stop" : "length";
    std::cerr << "omni request request_id=" << req_id
              << " prompt_tokens="
              << result.prefill_tokens + result.reused_tokens
              << " prefill_tokens=" << result.prefill_tokens
              << " prefix_reused=" << result.reused_tokens
              << " image_tokens=" << result.image_tokens
              << " generated_tokens=" << result.token_ids.size()
              << " finish_reason=" << finish_reason
              << " chars=" << result.text.size()
              << " prefill_ms=" << std::fixed << std::setprecision(1)
              << result.prefill_ms
              << " decode_ms=" << std::fixed << std::setprecision(1)
              << result.decode_ms
              << " elapsed_ms=" << elapsed.count() << "\n";
    return omni_result(result.text, static_cast<int>(result.token_ids.size()),
                       result.prefill_ms, result.decode_ms, finish_reason);
  };
}

}  // namespace umminfer::stage
