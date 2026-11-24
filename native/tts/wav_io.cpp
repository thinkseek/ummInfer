// SPDX-License-Identifier: Apache-2.0
#include "wav_io.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace umminfer::native {
namespace {

void put_u16(std::string& out, uint16_t v) {
  out.push_back(static_cast<char>(v & 0xff));
  out.push_back(static_cast<char>((v >> 8) & 0xff));
}

void put_u32(std::string& out, uint32_t v) {
  out.push_back(static_cast<char>(v & 0xff));
  out.push_back(static_cast<char>((v >> 8) & 0xff));
  out.push_back(static_cast<char>((v >> 16) & 0xff));
  out.push_back(static_cast<char>((v >> 24) & 0xff));
}

uint16_t get_u16(const std::string& in, size_t off) {
  return static_cast<uint16_t>(static_cast<uint8_t>(in[off])) |
         (static_cast<uint16_t>(static_cast<uint8_t>(in[off + 1])) << 8);
}

uint32_t get_u32(const std::string& in, size_t off) {
  return static_cast<uint32_t>(get_u16(in, off)) |
         (static_cast<uint32_t>(get_u16(in, off + 2)) << 16);
}

}  // namespace

std::string write_wav_pcm16_mono(const std::vector<float>& samples, uint32_t sample_rate) {
  if (sample_rate == 0) throw std::invalid_argument("sample rate must be positive");
  const uint32_t data_bytes = static_cast<uint32_t>(samples.size() * 2);
  std::string out;
  out.reserve(44 + data_bytes);
  out += "RIFF";
  put_u32(out, 36 + data_bytes);
  out += "WAVE";
  out += "fmt ";
  put_u32(out, 16);          // PCM fmt chunk size
  put_u16(out, 1);           // format tag: PCM
  put_u16(out, 1);           // channels: mono
  put_u32(out, sample_rate);
  put_u32(out, sample_rate * 2);  // byte rate
  put_u16(out, 2);           // block align
  put_u16(out, 16);          // bits per sample
  out += "data";
  put_u32(out, data_bytes);
  for (float x : samples) {
    long s = std::lrintf(x * 32767.0f);
    if (s > 32767) s = 32767;
    if (s < -32768) s = -32768;
    put_u16(out, static_cast<uint16_t>(static_cast<int16_t>(s)));
  }
  return out;
}

std::vector<float> read_wav_pcm16_mono(const std::string& path,
                                       uint32_t expected_sample_rate) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::invalid_argument("cannot open reference WAV " + path);
  const std::string bytes((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
  if (bytes.size() < 12 || bytes.compare(0, 4, "RIFF") != 0 ||
      bytes.compare(8, 4, "WAVE") != 0)
    throw std::invalid_argument("reference WAV is not RIFF/WAVE");
  if (static_cast<uint64_t>(get_u32(bytes, 4)) + 8 != bytes.size())
    throw std::invalid_argument("reference WAV has an inconsistent RIFF size");

  bool have_fmt = false, have_data = false;
  uint32_t rate = 0;
  size_t data_off = 0, data_size = 0;
  size_t pos = 12;
  while (pos + 8 <= bytes.size()) {
    const uint32_t chunk_size = get_u32(bytes, pos + 4);
    const size_t payload = pos + 8;
    if (chunk_size > bytes.size() - payload)
      throw std::invalid_argument("reference WAV has a truncated chunk");
    if (bytes.compare(pos, 4, "fmt ") == 0) {
      if (chunk_size < 16 || get_u16(bytes, payload) != 1 ||
          get_u16(bytes, payload + 2) != 1 || get_u16(bytes, payload + 14) != 16 ||
          get_u16(bytes, payload + 12) != 2)
        throw std::invalid_argument("reference WAV must be PCM16 mono");
      rate = get_u32(bytes, payload + 4);
      have_fmt = true;
    } else if (bytes.compare(pos, 4, "data") == 0 && !have_data) {
      data_off = payload;
      data_size = chunk_size;
      have_data = true;
    }
    pos = payload + chunk_size + (chunk_size & 1);
    if (pos < payload || pos > bytes.size())
      throw std::invalid_argument("reference WAV chunk exceeds the file");
  }
  if (pos != bytes.size())
    throw std::invalid_argument("reference WAV has a truncated trailing chunk header");
  if (!have_fmt || !have_data) throw std::invalid_argument("reference WAV lacks fmt or data");
  if (rate != expected_sample_rate)
    throw std::invalid_argument("reference WAV must be " +
                                std::to_string(expected_sample_rate) + " Hz");
  if (data_size % 2 != 0) throw std::invalid_argument("reference WAV has odd PCM16 data size");
  std::vector<float> out(data_size / 2);
  for (size_t i = 0; i < out.size(); ++i) {
    const int16_t s = static_cast<int16_t>(get_u16(bytes, data_off + i * 2));
    out[i] = static_cast<float>(s) / 32768.0f;
  }
  return out;
}

WavStats pcm_stats(const std::vector<float>& samples, uint32_t sample_rate) {
  WavStats st;
  st.sample_rate = sample_rate;
  st.samples = samples.size();
  st.duration_s = static_cast<double>(samples.size()) / sample_rate;
  double sum_sq = 0.0;
  size_t voiced = 0;
  for (float x : samples) {
    const double a = std::fabs(static_cast<double>(x));
    sum_sq += static_cast<double>(x) * x;
    if (a > 1e-3) ++voiced;
    if (a > st.peak) st.peak = static_cast<float>(a);
  }
  if (!samples.empty()) {
    st.rms = std::sqrt(sum_sq / samples.size());
    st.non_silence_ratio = static_cast<double>(voiced) / samples.size();
  }
  return st;
}

}  // namespace umminfer::native
