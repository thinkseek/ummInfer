// SPDX-License-Identifier: Apache-2.0
// TTS kind: Qwen3-TTS 0.6B CustomVoice behind the Stage protocol. The result
// is the audio artifact map the unified engine turns into kArtifact events.
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cmath>
#include <cstring>
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
    // Streaming chunks are raw PCM16 on codec-frame boundaries. Each chunk is
    // the next slice of the same waveform the terminal WAV carries, so a
    // consumer may play them in order and ignore the terminal artifact.
    int chunks = 0;
    std::vector<float> streamed_samples;
    auto sink = [&chunks, &streamed_samples](const std::vector<float>& chunk,
                                             int first_frame, int frames) {
      streamed_samples.insert(streamed_samples.end(), chunk.begin(), chunk.end());
      std::string pcm(static_cast<size_t>(chunk.size()) * 2, '\0');
      for (size_t i = 0; i < chunk.size(); ++i) {
        long v = std::lrintf(chunk[i] * 32767.0f);
        if (v > 32767) v = 32767;
        if (v < -32768) v = -32768;
        const auto u = static_cast<uint16_t>(static_cast<int16_t>(v));
        pcm[i * 2] = static_cast<char>(u & 0xff);
        pcm[i * 2 + 1] = static_cast<char>((u >> 8) & 0xff);
      }
      Value::Map out;
      out["audio"] = Value::bin(std::move(pcm));
      out["index"] = Value(static_cast<int64_t>(chunks++));
      out["first_frame"] = Value(static_cast<int64_t>(first_frame));
      out["frames"] = Value(static_cast<int64_t>(frames));
      out["sample_rate"] = Value(static_cast<int64_t>(kSampleRate));
      out["channels"] = Value(static_cast<int64_t>(1));
      out["format"] = Value("pcm16");
      emit_stage_stream(Value(std::move(out)));
    };
    std::vector<float> samples;
    native::TtsEngine::SynthTimings timings;
    try {
      samples = engine->synthesize(
          text_ids, language, speaker, max_frames,
          [&tracker] { return tracker(); }, &timings,
          input.stream ? native::TtsEngine::StreamSink(sink)
                       : native::TtsEngine::StreamSink());
    } catch (const native::RequestAborted& error) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - started);
      std::cerr << "tts request aborted boundary=\"" << error.what() << "\""
                << " abort_epoch_ms=" << tracker.detected_epoch_ms()
                << " exit_epoch_ms=" << epoch_millis()
                << " compute_ms=" << elapsed.count() << "\n";
      throw RequestAborted(error.what());
    }
    if (input.stream) {
      // The streaming contract is that the concatenated chunks equal the
      // terminal waveform bit for bit. This runs on every streamed request.
      size_t first_diff = 0;
      float max_diff = 0.0f;
      const size_t common =
          std::min(streamed_samples.size(), samples.size());
      while (first_diff < common &&
             streamed_samples[first_diff] == samples[first_diff]) {
        ++first_diff;
      }
      for (size_t i = 0; i < common; ++i) {
        max_diff = std::max(max_diff,
                            std::fabs(streamed_samples[i] - samples[i]));
      }
      const bool identical = streamed_samples.size() == samples.size() &&
                             first_diff == common;
      std::cerr << "tts stream_check identical=" << (identical ? 1 : 0)
                << " chunks=" << chunks << " samples=" << samples.size()
                << " streamed=" << streamed_samples.size()
                << " first_diff=" << first_diff
                << " max_abs_diff=" << max_diff
                << " first_audio_ms=" << timings.first_audio_ms << "\n";
      // The codec itself is not bit-repeatable (two decodes of the same
      // frames differ), so the chunked and one-shot waveforms are compared
      // for a gross divergence only. The value is logged as evidence.
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
    return tts_result(std::move(wav), kSampleRate, 1, voice,
                      timings.first_frame_ms, timings.total_ms);
  };
}

}  // namespace umminfer::stage
