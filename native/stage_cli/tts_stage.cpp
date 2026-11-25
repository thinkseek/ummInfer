// SPDX-License-Identifier: Apache-2.0
// TTS kind: Qwen3-TTS 0.6B CustomVoice behind the Stage protocol. The result
// is the audio artifact map the unified engine turns into kArtifact events.
#include <cuda_runtime.h>

#include <chrono>
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

const char* stage_usage_flags() {
  return "[--max-frames N] [--default-ref-audio PATH] ";
}

bool parse_stage_flag(const std::string& key, const std::string& value,
                      StageOptions& options) {
  if (key == "max-frames") {
    options.max_frames = parse_stage_int(value, "--max-frames");
    return true;
  }
  if (key == "default-ref-audio") {
    if (value.empty()) {
      throw std::runtime_error("--default-ref-audio requires a path");
    }
    options.default_ref_audio = value;
    return true;
  }
  return false;
}

CancellableComputeFn make_compute(const StageOptions& options) {
  const int max_frames =
      options.max_frames > 0 ? options.max_frames : kDefaultMaxFrames;
  auto tokenizer = std::make_shared<native::Qwen2BpeTokenizer>(options.model_dir);
  auto engine = std::make_shared<native::TtsEngine>(options.model_dir, options.gpu);
  const bool is_base = engine->is_base();
  auto default_embedding = std::make_shared<std::vector<float>>();
  if (!options.default_ref_audio.empty()) {
    if (!is_base) {
      throw std::runtime_error(
          "--default-ref-audio requires a Qwen3-TTS Base checkpoint");
    }
    const std::vector<float> ref =
        native::read_wav_pcm16_mono(options.default_ref_audio, kSampleRate);
    *default_embedding = engine->extract_speaker_embedding(ref);
  }

  const int gpu = options.gpu;
  return [tokenizer, engine, is_base, default_embedding, max_frames, gpu](
             const Value& work, const AbortCheck& aborted) {
    const auto started = std::chrono::steady_clock::now();
    const std::string req_id = bound_request_id();
    // The compute function runs on the Stage scheduler thread, whose CUDA
    // current device defaults to 0. The engine's buffers belong to --gpu.
    if (cudaSetDevice(gpu) != cudaSuccess) {
      throw ProtocolError("cudaSetDevice(" + std::to_string(gpu) + ") failed");
    }
    AbortTracker tracker(aborted);
    const TtsWork input = parse_tts_work(work);
    std::string voice = input.voice.empty() ? "vivian" : input.voice;
    int speaker = 0;
    std::vector<float> request_embedding;
    const std::vector<float>* clone_embedding = nullptr;
    if (is_base) {
      if (!input.ref_audio_wav.empty()) {
        const std::vector<float> ref =
            native::read_wav_pcm16_mono_bytes(input.ref_audio_wav, kSampleRate);
        request_embedding = engine->extract_speaker_embedding(ref);
        clone_embedding = &request_embedding;
      } else if (!default_embedding->empty()) {
        clone_embedding = default_embedding.get();
      } else {
        throw ProtocolError(
            "Base TTS checkpoint requires inputs.ref_audio (24 kHz PCM16 mono "
            "WAV) or stage --default-ref-audio");
      }
      if (input.voice.empty()) voice = "clone";
    } else {
      if (!input.ref_audio_wav.empty()) {
        throw ProtocolError(
            "CustomVoice TTS checkpoint does not accept ref_audio; load a "
            "Qwen3-TTS Base checkpoint for zero-shot voice cloning");
      }
      // The public "voice" field is a free label; only the wired presets exist
      // here. The artifact descriptor names the voice actually used.
      try {
        speaker = native::speaker_id_from_name(voice);
      } catch (const std::exception&) {
        std::cerr << "tts unsupported voice '" << voice
                  << "', falling back to vivian\n";
        voice = "vivian";
        speaker = native::speaker_id_from_name(voice);
      }
    }
    const int language = text_is_mostly_cjk(input.text) ? native::kLangChinese
                                                        : native::kLangEnglish;
    const std::string wrapped = "<|im_start|>assistant\n" + input.text +
                                "<|im_end|>\n<|im_start|>assistant\n";
    const std::vector<int> text_ids = tokenizer->encode(wrapped);
    // Streaming chunks are raw PCM16 on codec-frame boundaries. The engine
    // keeps one waveform per request, so the chunks concatenated in order are
    // that waveform: a consumer may play them and ignore the terminal
    // artifact. That equality is checked below and a mismatch fails the
    // request.
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
    const auto stream_sink = input.stream ? native::TtsEngine::StreamSink(sink)
                                          : native::TtsEngine::StreamSink();
    try {
      samples = is_base
                    ? engine->synthesize_clone(text_ids, language, *clone_embedding,
                                               max_frames,
                                               [&tracker] { return tracker(); },
                                               &timings, stream_sink)
                    : engine->synthesize(text_ids, language, speaker, max_frames,
                                         [&tracker] { return tracker(); },
                                         &timings, stream_sink);
    } catch (const native::RequestAborted& error) {
      log_stage_abort("tts", nullptr, req_id, error.what(), tracker, started);
      throw RequestAborted(error.what());
    }
    if (input.stream) {
      // Contract: the concatenated chunks equal the terminal waveform bit for
      // bit, because both come from the same buffer. A mismatch is a bug, not
      // a float tolerance, so it fails the request. This runs on every
      // streamed request.
      size_t first_diff = 0;
      const bool identical =
          native::stream_matches(streamed_samples, samples, &first_diff);
      std::cerr << "tts stream_check request_id=" << req_id
                << " identical=" << (identical ? 1 : 0)
                << " chunks=" << chunks << " samples=" << samples.size()
                << " streamed=" << streamed_samples.size()
                << " first_diff=" << first_diff
                << " first_audio_ms=" << timings.first_audio_ms << "\n";
      if (!identical) {
        throw ProtocolError(
            "stream chunk concatenation differs from the terminal waveform at "
            "sample " + std::to_string(first_diff) + " (streamed=" +
            std::to_string(streamed_samples.size()) + " terminal=" +
            std::to_string(samples.size()) + ")");
      }
    }
    std::string wav = native::write_wav_pcm16_mono(samples, kSampleRate);
    const native::WavStats stats = native::pcm_stats(samples, kSampleRate);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started);
    std::cerr << "tts request request_id=" << req_id
              << " voice=" << voice
              << " language=" << (language == native::kLangChinese ? "chinese" : "english")
              << " samples=" << stats.samples << " duration_s=" << stats.duration_s
              << " rms=" << stats.rms << " peak=" << stats.peak
              << " non_silence=" << stats.non_silence_ratio
              << " wav_bytes=" << wav.size()
              << " elapsed_ms=" << elapsed.count() << "\n";
    if (stats.samples == 0) throw ProtocolError("TTS produced no samples");
    return tts_result(std::move(wav), kSampleRate, 1, voice,
                      timings.first_frame_ms, timings.total_ms, input.stream);
  };
}

}  // namespace umminfer::stage
