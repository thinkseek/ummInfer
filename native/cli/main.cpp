// SPDX-License-Identifier: Apache-2.0
// umm-asr-native: model directory + WAV -> transcript on stdout.
// No Python, no SGLang, no inference engine; cuBLAS + hand-written CUDA.
#include <cuda_runtime.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

#include "umm_asr_native/engine.hpp"
#include "umm_asr_native/feature_frontend.hpp"
#include "umm_asr_native/tokenizer.hpp"
#include "umm_asr_native/wav.hpp"

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "usage: umm-asr-native <model_dir> <wav> [gpu_id]\n";
    return 2;
  }
  const int gpu_id = argc == 4 ? std::stoi(argv[3]) : 0;
  std::ifstream f(argv[2], std::ios::binary);
  if (!f) {
    std::cerr << "cannot open wav: " << argv[2] << "\n";
    return 2;
  }
  std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

  try {
    auto t0 = std::chrono::steady_clock::now();
    auto audio = umminfer::native::load_wav_pcm16(bytes);
    umminfer::native::WhisperMel frontend;
    auto mel = frontend.compute(audio.samples);
    auto t1 = std::chrono::steady_clock::now();

    umminfer::native::AsrEngine engine(argv[1], gpu_id);
    auto t2 = std::chrono::steady_clock::now();
    auto ids = engine.transcribe_tokens(mel);
    auto t3 = std::chrono::steady_clock::now();

    umminfer::native::QwenTokenizer tokenizer(std::string(argv[1]) + "/vocab.json");
    // The model emits a "language" token (11528) followed by the language name
    // token, matching the SGLang adapter's <asr_text> prefix that gets stripped
    // before the actual transcript. Drop both when present.
    if (ids.size() >= 2 && ids[0] == umminfer::native::QwenTokenizer::kLanguage)
      ids.erase(ids.begin(), ids.begin() + 2);
    const std::string text = tokenizer.decode(ids);
    std::cout << text << "\n";

    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    std::cerr << "audio_tokens=" << umminfer::native::AsrEngine::audio_token_count(
                                         mel.valid_frames)
              << " generated_tokens=" << ids.size()
              << " frontend_ms="
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " load_ms="
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " infer_ms="
              << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
              << " gpu_used_mb=" << (total_b - free_b) / (1024 * 1024) << "\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
