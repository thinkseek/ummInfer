// SPDX-License-Identifier: Apache-2.0
// ASR kind: Qwen3-ASR 0.6B behind the Stage protocol.
//
// Cancellation: the Coordinator broadcast aborts the request record and the
// stage settles it as failed immediately; the accepted AsrEngine exposes no
// per-token abort hook, so an in-flight decode runs to its own end and its
// result is discarded. Modifying native/umm_asr_native is out of scope here.
#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "stage_common.hpp"
#include "stage_kind.hpp"
#include "umm_asr_native/engine.hpp"
#include "umm_asr_native/feature_frontend.hpp"
#include "umm_asr_native/tokenizer.hpp"

namespace umminfer::stage {
namespace {

constexpr int kDefaultMaxNewTokens = 256;

}  // namespace

const char* compiled_kind() { return "asr"; }

CancellableComputeFn make_compute(const StageOptions& options) {
  const int max_new_tokens =
      options.max_new_tokens > 0 ? options.max_new_tokens : kDefaultMaxNewTokens;
  auto engine = std::make_shared<native::AsrEngine>(options.model_dir, options.gpu);
  auto tokenizer =
      std::make_shared<native::QwenTokenizer>(options.model_dir + "/vocab.json");
  auto frontend = std::make_shared<native::WhisperMel>();

  const int gpu = options.gpu;
  return [engine, tokenizer, frontend, max_new_tokens, gpu](
             const Value& work, const AbortCheck& aborted) {
    const auto started = std::chrono::steady_clock::now();
    // The compute function runs on the Stage scheduler thread, whose CUDA
    // current device defaults to 0. The engine's buffers belong to --gpu.
    if (cudaSetDevice(gpu) != cudaSuccess) {
      throw ProtocolError("cudaSetDevice(" + std::to_string(gpu) + ") failed");
    }
    AbortTracker tracker(aborted);
    const AsrWork input = parse_asr_work(work);
    const native::MelFeatures mel = frontend->compute(input.samples);
    std::vector<int> ids;
    try {
      ids = engine->transcribe_tokens(mel, max_new_tokens,
                                      [&tracker] { return tracker(); });
    } catch (const native::RequestAborted& error) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - started);
      std::cerr << "asr request aborted boundary=\"" << error.what() << "\""
                << " abort_epoch_ms=" << tracker.detected_epoch_ms()
                << " exit_epoch_ms=" << epoch_millis()
                << " compute_ms=" << elapsed.count() << "\n";
      throw RequestAborted(error.what());
    }
    const int generated = static_cast<int>(ids.size());
    // Same prefix strip as the accepted CLI: the model emits
    // <language><name> before the transcript.
    if (ids.size() >= 2 && ids[0] == native::QwenTokenizer::kLanguage) {
      ids.erase(ids.begin(), ids.begin() + 2);
    }
    const std::string text = tokenizer->decode(ids);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started);
    std::cerr << "asr request audio_samples=" << input.samples.size()
              << " generated_tokens=" << generated << " chars=" << text.size()
              << " elapsed_ms=" << elapsed.count() << "\n";
    return asr_result(text, generated,
                      generated >= max_new_tokens ? "length" : "stop");
  };
}

}  // namespace umminfer::stage
