// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace umminfer::audio {

struct WavPcm16View {
  std::string_view pcm;
  uint32_t sample_rate = 0;
  uint16_t channels = 0;
};

// Headerless PCM16 mono sample bytes. wav_from_pcm16() writes 16 kHz for the
// existing ASR path; the overload accepts 1 to 2147483647 Hz.
// parse_wav_pcm16() validates a RIFF/WAVE PCM16 container of any valid sample
// rate and channel count without copying its sample payload; pcm16_from_wav()
// enforces the 16 kHz mono ASR input contract.
// Invalid or unsupported input throws std::invalid_argument.
std::string wav_from_pcm16(std::string_view pcm);
std::string wav_from_pcm16(std::string_view pcm, uint32_t sample_rate);
WavPcm16View parse_wav_pcm16(std::string_view wav);
std::string pcm16_from_wav(std::string_view wav);

}  // namespace umminfer::audio
