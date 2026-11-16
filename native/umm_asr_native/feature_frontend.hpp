// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <vector>

#include "wav.hpp"

namespace umminfer::native {

// Whisper-compatible log-mel frontend, matching transformers
// WhisperFeatureExtractor (128 mels, n_fft 400, hop 160, 30 s chunk).
struct MelFeatures {
  int num_mels = 128;
  int num_frames = 0;       // always 3000 for a padded 30 s chunk
  int valid_frames = 0;     // frames covered by real samples (attention mask)
  std::vector<float> data;  // row-major [num_mels][num_frames]
};

class WhisperMel {
 public:
  WhisperMel();

  // Pads/truncates to 30 s and computes the 128x3000 log-mel matrix.
  MelFeatures compute(const std::vector<float>& samples) const;

  // Visible for tests: number of mel frames for a raw sample count.
  static int valid_frame_count(int num_samples);

 private:
  static constexpr int kNfft = 400;
  static constexpr int kHop = 160;
  static constexpr int kMels = 128;
  static constexpr int kFftBins = kNfft / 2 + 1;  // 201
  static constexpr int kChunkSamples = 30 * 16000;
  static constexpr int kChunkFrames = kChunkSamples / kHop;  // 3000

  std::vector<float> window_;                       // 400 periodic Hann
  std::vector<float> mel_bank_;                     // [201][128], slaney norm
  std::vector<float> dft_cos_;                      // [201][400]
  std::vector<float> dft_sin_;
};

}  // namespace umminfer::native
