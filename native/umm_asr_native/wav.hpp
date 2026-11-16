// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace umminfer::native {

struct PcmAudio {
  uint32_t sample_rate = 0;
  std::vector<float> samples;  // mono, range [-1.0f, 1.0f]
};

// Parses a RIFF WAVE file. Only uncompressed PCM (format 1), mono, 16-bit is
// accepted; anything else throws std::invalid_argument with the reason.
// ponytail: 16 kHz only is accepted for inference; add linear resampling if a
// non-16k WAV must be served instead of rejecting it.
PcmAudio load_wav_pcm16(std::string_view wav);

}  // namespace umminfer::native
