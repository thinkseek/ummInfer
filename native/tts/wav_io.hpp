// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace umminfer::native {

// Serializes mono float samples in [-1, 1] to a PCM16 RIFF WAVE byte string.
// Samples are rounded to nearest with the 32767 scale (libsndfile convention),
// clipped to the int16 range. Any sample rate is accepted; the TTS CLI passes
// 24000.
std::string write_wav_pcm16_mono(const std::vector<float>& samples, uint32_t sample_rate);

// Parses a RIFF PCM16 mono WAV byte buffer and rejects a sample rate other than
// expected. Shared by file loading and Stage in-memory ref_audio decoding.
std::vector<float> read_wav_pcm16_mono_bytes(std::string_view bytes,
                                             uint32_t expected_sample_rate = 24000);

// Reads a RIFF PCM16 mono file and rejects a sample rate other than expected.
std::vector<float> read_wav_pcm16_mono(const std::string& path,
                                       uint32_t expected_sample_rate = 24000);

// PCM16 statistics used by the verification harness.
struct WavStats {
  uint32_t sample_rate = 0;
  size_t samples = 0;
  double duration_s = 0.0;
  double rms = 0.0;
  float peak = 0.0f;
  double non_silence_ratio = 0.0;  // fraction of |x| > 1e-3
};

WavStats pcm_stats(const std::vector<float>& samples, uint32_t sample_rate);

}  // namespace umminfer::native
