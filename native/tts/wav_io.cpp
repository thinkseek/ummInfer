// SPDX-License-Identifier: Apache-2.0
#include "wav_io.hpp"

#include <cmath>
#include <cstring>
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
