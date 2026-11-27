// SPDX-License-Identifier: Apache-2.0
// umm-tts-native: native Qwen3-TTS CustomVoice and Base voice-cloning CLI.
//   umm-tts-native --model DIR --text "..." --out out.wav
//                 [--speaker vivian|ryan] [--ref-audio ref_24k.wav]
//                 [--language chinese|english] [--gpu N] [--max-frames N]
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "bpe_tokenizer.hpp"
#include "engine.hpp"
#include "prompt.hpp"
#include "wav_io.hpp"

using namespace umminfer::native;

namespace {

std::string flag(int argc, char** argv, const std::string& name,
                 const std::string& dflt) {
  for (int i = 1; i + 1 < argc; ++i)
    if (name == argv[i]) return argv[i + 1];
  return dflt;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string model = flag(argc, argv, "--model", "");
  const std::string text = flag(argc, argv, "--text", "");
  const std::string out = flag(argc, argv, "--out", "");
  const std::string speaker = flag(argc, argv, "--speaker", "vivian");
  const std::string ref_audio = flag(argc, argv, "--ref-audio", "");
  const std::string embedding_dump = flag(argc, argv, "--dump-speaker-embedding", "");
  const std::string language = flag(argc, argv, "--language", "chinese");
  const int gpu = std::stoi(flag(argc, argv, "--gpu", "0"));
  const int max_frames = std::stoi(flag(argc, argv, "--max-frames", "8192"));
  if (model.empty() || text.empty() || out.empty()) {
    std::cerr << "usage: umm-tts-native --model DIR --text TEXT --out WAV "
                 "[--speaker vivian|ryan] [--ref-audio WAV_24K] "
                 "[--language chinese|english] [--gpu N] [--max-frames N] [--stream 1] "
                 "[--dump-speaker-embedding PATH]\n";
    return 2;
  }
  try {
    if (!embedding_dump.empty() && ref_audio.empty())
      throw std::invalid_argument("--dump-speaker-embedding requires --ref-audio");
    const int spk = ref_audio.empty() ? speaker_id_from_name(speaker) : 0;
    const int lang = language_id_from_name(language);
    Qwen2BpeTokenizer tok(model);
    const std::string wrapped = "<|im_start|>assistant\n" + text +
                                "<|im_end|>\n<|im_start|>assistant\n";
    const std::vector<int> text_ids = tok.encode(wrapped);

    TtsEngine engine(model, gpu);
    // --stream installs a sink that accumulates the chunks, then asserts the
    // streamed contract: the chunks in order equal the returned waveform.
    const bool stream = flag(argc, argv, "--stream", "0") == "1";
    std::vector<float> streamed;
    TtsEngine::SynthTimings timings;
    TtsEngine::StreamSink sink;
    if (stream) {
      sink = [&streamed](const std::vector<float>& chunk, int, int) {
        streamed.insert(streamed.end(), chunk.begin(), chunk.end());
      };
    }
    std::vector<float> wav;
    if (ref_audio.empty()) {
      wav = engine.synthesize(text_ids, lang, spk, max_frames, {}, &timings,
                              stream ? sink : TtsEngine::StreamSink());
    } else {
      const std::vector<float> ref = read_wav_pcm16_mono(ref_audio);
      const auto speaker_started = std::chrono::steady_clock::now();
      const std::vector<float> embedding = engine.extract_speaker_embedding(ref);
      if (std::getenv("UMM_TTS_TIMINGS"))
        std::cerr << "tts speaker_encoder_ms="
                  << std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - speaker_started)
                         .count()
                  << "\n";
      if (!embedding_dump.empty()) {
        std::ofstream f(embedding_dump, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open embedding output " + embedding_dump);
        f.write(reinterpret_cast<const char*>(embedding.data()),
                static_cast<std::streamsize>(embedding.size() * sizeof(float)));
      }
      wav = engine.synthesize_clone(text_ids, lang, embedding, max_frames, {}, &timings,
                                    stream ? sink : TtsEngine::StreamSink());
    }
    if (stream) {
      size_t first_diff = 0;
      const bool identical = stream_matches(streamed, wav, &first_diff);
      std::cerr << "tts stream_check identical=" << (identical ? 1 : 0)
                << " samples=" << wav.size()
                << " streamed=" << streamed.size()
                << " first_diff=" << first_diff
                << " first_audio_ms=" << timings.first_audio_ms << "\n";
      if (!identical) {
        std::cerr << "stream chunks differ from the terminal waveform\n";
        return 1;
      }
    }
    const std::string bytes = write_wav_pcm16_mono(wav, 24000);
    std::ofstream f(out, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open output " + out);
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));

    const WavStats st = pcm_stats(wav, 24000);
    std::cout << "codec_frames=" << timings.generated_frames
              << " natural_eos=" << (timings.natural_eos ? 1 : 0)
              << " samples=" << st.samples << " duration_s=" << st.duration_s
              << " rms=" << st.rms << " peak=" << st.peak
              << " non_silence=" << st.non_silence_ratio << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
