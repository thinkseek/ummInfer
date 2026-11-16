// SPDX-License-Identifier: Apache-2.0
#include "wav.hpp"

#include <limits>
#include <stdexcept>

namespace umminfer::native {
namespace {

uint16_t read_u16(std::string_view b, size_t off) {
  return static_cast<uint16_t>(static_cast<uint8_t>(b[off])) |
         (static_cast<uint16_t>(static_cast<uint8_t>(b[off + 1])) << 8);
}

uint32_t read_u32(std::string_view b, size_t off) {
  return static_cast<uint32_t>(static_cast<uint8_t>(b[off])) |
         (static_cast<uint32_t>(static_cast<uint8_t>(b[off + 1])) << 8) |
         (static_cast<uint32_t>(static_cast<uint8_t>(b[off + 2])) << 16) |
         (static_cast<uint32_t>(static_cast<uint8_t>(b[off + 3])) << 24);
}

}  // namespace

PcmAudio load_wav_pcm16(std::string_view wav) {
  if (wav.size() < 12 || wav.substr(0, 4) != "RIFF" || wav.substr(8, 4) != "WAVE") {
    throw std::invalid_argument("not a RIFF WAVE file");
  }
  const size_t riff_end = static_cast<size_t>(read_u32(wav, 4)) + 8;
  if (riff_end > wav.size()) throw std::invalid_argument("truncated RIFF container");

  bool have_fmt = false;
  uint16_t format = 0, channels = 0, bits = 0;
  uint32_t sample_rate = 0;
  size_t data_off = 0, data_size = 0;

  size_t off = 12;
  while (off + 8 <= riff_end) {
    const std::string_view id = wav.substr(off, 4);
    const uint32_t chunk_size = read_u32(wav, off + 4);
    const size_t body = off + 8;
    if (body + chunk_size > riff_end) throw std::invalid_argument("truncated WAV chunk");
    if (id == "fmt " && chunk_size >= 16) {
      if (have_fmt) throw std::invalid_argument("duplicate fmt chunk");
      format = read_u16(wav, body);
      channels = read_u16(wav, body + 2);
      sample_rate = read_u32(wav, body + 4);
      bits = read_u16(wav, body + 14);
      have_fmt = true;
    } else if (id == "data") {
      if (data_size != 0) throw std::invalid_argument("duplicate data chunk");
      data_off = body;
      data_size = chunk_size;
    }
    off = body + chunk_size + (chunk_size & 1);
  }
  if (!have_fmt) throw std::invalid_argument("WAV has no fmt chunk");
  if (data_size == 0) throw std::invalid_argument("WAV has no data chunk");
  if (format != 1) throw std::invalid_argument("only uncompressed PCM WAV is supported");
  if (channels != 1) throw std::invalid_argument("only mono WAV is supported");
  if (bits != 16) throw std::invalid_argument("only 16-bit PCM WAV is supported");
  if (sample_rate != 16000)
    throw std::invalid_argument("only 16 kHz WAV is supported by the native slice");
  if (data_size % 2 != 0) throw std::invalid_argument("odd PCM16 byte count");

  PcmAudio out;
  out.sample_rate = sample_rate;
  out.samples.resize(data_size / 2);
  constexpr float kScale = 1.0f / 32768.0f;
  for (size_t i = 0; i < out.samples.size(); ++i) {
    int16_t s = static_cast<int16_t>(read_u16(wav, data_off + i * 2));
    out.samples[i] = static_cast<float>(s) * kScale;
  }
  return out;
}

}  // namespace umminfer::native
