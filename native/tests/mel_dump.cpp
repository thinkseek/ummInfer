// SPDX-License-Identifier: Apache-2.0
// Dumps log-mel features as raw float32 [128][3000] plus a header line, for a
// numeric cross-check against the transformers WhisperFeatureExtractor.
#include <cstdio>
#include <fstream>
#include <iostream>

#include "umm_asr_native/feature_frontend.hpp"
#include "umm_asr_native/wav.hpp"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: mel_dump <wav> <out.f32>\n";
    return 2;
  }
  std::ifstream f(argv[1], std::ios::binary);
  std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  auto audio = umminfer::native::load_wav_pcm16(bytes);
  umminfer::native::WhisperMel mel;
  auto features = mel.compute(audio.samples);
  std::printf("samples=%zu valid_frames=%d shape=%dx%d\n", audio.samples.size(),
              features.valid_frames, features.num_mels, features.num_frames);
  std::ofstream out(argv[2], std::ios::binary);
  out.write(reinterpret_cast<const char*>(features.data.data()),
            static_cast<std::streamsize>(features.data.size() * sizeof(float)));
  return out ? 0 : 1;
}
