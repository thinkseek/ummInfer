// SPDX-License-Identifier: Apache-2.0
// TTS kind: Qwen3-TTS 0.6B CustomVoice behind the Stage protocol. The result
// is the audio artifact map the unified engine turns into kArtifact events.
#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "bpe_tokenizer.hpp"
#include "engine.hpp"
#include "prompt.hpp"
#include "stage_common.hpp"
#include "stage_kind.hpp"
#include "wav_io.hpp"

namespace umminfer::stage {
namespace {

constexpr int kDefaultMaxFrames = 8192;
constexpr int kSampleRate = 24000;

}  // namespace

const char* compiled_kind() { return "tts"; }

CancellableComputeFn make_compute(const StageOptions& options) {
  const int max_frames =
      options.max_frames > 0 ? options.max_frames : kDefaultMaxFrames;
  auto tokenizer = std::make_shared<native::Qwen2BpeTokenizer>(options.model_dir);
  auto engine = std::make_shared<native::TtsEngine>(options.model_dir, options.gpu);

  const int gpu = options.gpu;
  return [tokenizer, engine, max_frames, gpu](
             const Value& work, const AbortCheck& aborted) {
    const auto started = std::chrono::steady_clock::now();
    // The compute function runs on the Stage scheduler thread, whose CUDA
    // current device defaults to 0. The engine's buffers belong to --gpu.
    if (cudaSetDevice(gpu) != cudaSuccess) {
      throw ProtocolError("cudaSetDevice(" + std::to_string(gpu) + ") failed");
    }
    AbortTracker tracker(aborted);
    const TtsWork input = parse_tts_work(work);
    // The public "voice" field is a free label; only the two wired presets
    // exist here. The artifact descriptor names the voice actually used.
    std::string voice = input.voice.empty() ? "vivian" : input.voice;
    int speaker = 0;
    try {
      speaker = native::speaker_id_from_name(voice);
    } catch (const std::exception&) {
      std::cerr << "tts unsupported voice '" << voice << "', falling back to vivian\n";
      voice = "vivian";
      speaker = native::speaker_id_from_name(voice);
    }
    const int language = text_is_mostly_cjk(input.text) ? native::kLangChinese
                                                        : native::kLangEnglish;
    const std::string wrapped = "<|im_start|>assistant\n" + input.text +
                                "<|im_end|>\n<|im_start|>assistant\n";
    const std::vector<int> text_ids = tokenizer->encode(wrapped);
    std::vector<float> samples;
    try {
      samples = engine->synthesize(text_ids, language, speaker, max_frames,
                                   [&tracker] { return tracker(); });
    } catch (const native::RequestAborted& error) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - started);
      std::cerr << "tts request aborted boundary=\"" << error.what() << "\""
                << " abort_epoch_ms=" << tracker.detected_epoch_ms()
                << " exit_epoch_ms=" << epoch_millis()
                << " compute_ms=" << elapsed.count() << "\n";
      throw RequestAborted(error.what());
    }
    std::string wav = native::write_wav_pcm16_mono(samples, kSampleRate);
    const native::WavStats stats = native::pcm_stats(samples, kSampleRate);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started);
    std::cerr << "tts request voice=" << voice
              << " language=" << (language == native::kLangChinese ? "chinese" : "english")
              << " samples=" << stats.samples << " duration_s=" << stats.duration_s
              << " rms=" << stats.rms << " peak=" << stats.peak
              << " non_silence=" << stats.non_silence_ratio
              << " wav_bytes=" << wav.size()
              << " elapsed_ms=" << elapsed.count() << "\n";
    if (stats.samples == 0) throw ProtocolError("TTS produced no samples");
    return tts_result(std::move(wav), kSampleRate, 1, voice);
  };
}

}  // namespace umminfer::stage
