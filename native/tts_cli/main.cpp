// SPDX-License-Identifier: Apache-2.0
// umm-tts-native: native Qwen3-TTS CustomVoice CLI.
//   umm-tts-native --model DIR --text "..." --out out.wav
//                 [--speaker vivian|ryan] [--language chinese|english]
//                 [--gpu N] [--max-frames N]
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
  const std::string language = flag(argc, argv, "--language", "chinese");
  const int gpu = std::stoi(flag(argc, argv, "--gpu", "0"));
  const int max_frames = std::stoi(flag(argc, argv, "--max-frames", "8192"));
  if (model.empty() || text.empty() || out.empty()) {
    std::cerr << "usage: umm-tts-native --model DIR --text TEXT --out WAV "
                 "[--speaker vivian|ryan] [--language chinese|english] [--gpu N] "
                 "[--max-frames N]\n";
    return 2;
  }
  try {
    const int spk = speaker_id_from_name(speaker);
    const int lang = language_id_from_name(language);
    Qwen2BpeTokenizer tok(model);
    const std::string wrapped = "<|im_start|>assistant\n" + text +
                                "<|im_end|>\n<|im_start|>assistant\n";
    const std::vector<int> text_ids = tok.encode(wrapped);

    TtsEngine engine(model, gpu);
    const std::vector<float> wav =
        engine.synthesize(text_ids, lang, spk, max_frames);
    const std::string bytes = write_wav_pcm16_mono(wav, 24000);
    std::ofstream f(out, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open output " + out);
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));

    const WavStats st = pcm_stats(wav, 24000);
    std::cout << "samples=" << st.samples << " duration_s=" << st.duration_s
              << " rms=" << st.rms << " peak=" << st.peak
              << " non_silence=" << st.non_silence_ratio << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
