// SPDX-License-Identifier: Apache-2.0
#include "speaker_encoder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "safetensors.hpp"

namespace umminfer::native {
namespace {

constexpr double kPi = 3.1415926535897932384626433832795;

int reflect_index(int i, int n) {
  while (i < 0 || i >= n) {
    if (i < 0)
      i = -i;
    else
      i = 2 * n - 2 - i;
  }
  return i;
}

struct Tensor {
  int channels = 0;
  int frames = 0;
  std::vector<float> data;

  Tensor() = default;
  Tensor(int c, int t) : channels(c), frames(t), data(static_cast<size_t>(c) * t) {}
  Tensor(int c, int t, std::vector<float>&& values)
      : channels(c), frames(t), data(std::move(values)) {}
};

void fft1024(std::vector<std::complex<float>>& a) {
  static const std::array<std::complex<float>, 512> roots = [] {
    std::array<std::complex<float>, 512> r;
    for (int i = 0; i < 512; ++i) {
      const double angle = -2.0 * kPi * i / 1024.0;
      r[i] = {static_cast<float>(std::cos(angle)), static_cast<float>(std::sin(angle))};
    }
    return r;
  }();
  for (size_t i = 1, j = 0; i < a.size(); ++i) {
    size_t bit = a.size() >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }
  for (size_t len = 2; len <= a.size(); len <<= 1) {
    const size_t stride = a.size() / len;
    for (size_t i = 0; i < a.size(); i += len) {
      for (size_t j = 0; j < len / 2; ++j) {
        const std::complex<float> even = a[i + j];
        const std::complex<float> odd = a[i + j + len / 2] * roots[j * stride];
        a[i + j] = even + odd;
        a[i + j + len / 2] = even - odd;
      }
    }
  }
}

double hz_to_mel(double hz) {
  return hz < 1000.0
             ? 3.0 * hz / 200.0
             : 15.0 + std::log(hz / 1000.0) * 27.0 / std::log(6.4);
}

double mel_to_hz(double mel) {
  return mel < 15.0
             ? 200.0 * mel / 3.0
             : 1000.0 * std::exp((mel - 15.0) * std::log(6.4) / 27.0);
}

}  // namespace

SpeakerMel compute_speaker_log_mel(const std::vector<float>& ref_pcm_24k) {
  if (ref_pcm_24k.size() <= 384)
    throw std::invalid_argument("reference audio is too short for 384-sample reflect padding");
  const int n = static_cast<int>(ref_pcm_24k.size());
  const int frames = n / 256;
  std::array<float, 1024> window{};
  for (int i = 0; i < 1024; ++i)
    window[i] = static_cast<float>(0.5 * (1.0 - std::cos(2.0 * kPi * i / 1024.0)));

  std::array<double, 130> mel_hz{};
  const double mel_min = hz_to_mel(0.0), mel_max = hz_to_mel(12000.0);
  for (int i = 0; i < 130; ++i)
    mel_hz[i] = mel_to_hz(mel_min + (mel_max - mel_min) * i / 129.0);
  std::vector<float> filters(static_cast<size_t>(128) * 513, 0.0f);
  for (int m = 0; m < 128; ++m) {
    for (int k = 0; k < 513; ++k) {
      const double freq = 24000.0 * k / 1024.0;
      const double lower = (freq - mel_hz[m]) / (mel_hz[m + 1] - mel_hz[m]);
      const double upper = (mel_hz[m + 2] - freq) / (mel_hz[m + 2] - mel_hz[m + 1]);
      const double weight = std::max(0.0, std::min(lower, upper)) *
                            (2.0 / (mel_hz[m + 2] - mel_hz[m]));
      filters[static_cast<size_t>(m) * 513 + k] = static_cast<float>(weight);
    }
  }

  SpeakerMel out{frames, std::vector<float>(static_cast<size_t>(128) * frames)};
  std::vector<std::complex<float>> fft(1024);
  std::array<float, 513> magnitude{};
  for (int t = 0; t < frames; ++t) {
    for (int i = 0; i < 1024; ++i) {
      const int source = reflect_index(t * 256 + i - 384, n);
      fft[i] = std::complex<float>(ref_pcm_24k[source] * window[i], 0.0f);
    }
    fft1024(fft);
    for (int k = 0; k < 513; ++k) {
      const float re = fft[k].real(), im = fft[k].imag();
      magnitude[k] = std::sqrt(re * re + im * im + 1e-9f);
    }
    for (int m = 0; m < 128; ++m) {
      const float* filter = filters.data() + static_cast<size_t>(m) * 513;
      float sum = 0.0f;
      for (int k = 0; k < 513; ++k) sum += filter[k] * magnitude[k];
      out.log_mel[static_cast<size_t>(m) * frames + t] =
          std::log(std::max(sum, 1e-5f));
    }
  }
  return out;
}

struct SpeakerEncoder::Impl {
  SpeakerEncoderConfig cfg;
  std::unordered_map<std::string, std::vector<float>> weights;

  Impl(const SafeTensors& st, const TtsModelConfig& model_cfg) : cfg(model_cfg.speaker_encoder) {
    if (!model_cfg.is_base()) throw std::invalid_argument("speaker encoder requires a Base config");
    for (const TensorShape& want : model_cfg.tensor_shape_plan()) {
      if (want.name.rfind("speaker_encoder.", 0) != 0) continue;
      const TensorInfo& info = st.info(want.name);
      if (info.dtype != "BF16" || info.shape != want.dims)
        throw std::invalid_argument("invalid speaker encoder tensor: " + want.name);
      const std::string_view bytes = st.tensor_bytes(want.name);
      std::vector<float> values(bytes.size() / 2);
      for (size_t i = 0; i < values.size(); ++i) {
        const uint16_t bits = static_cast<uint8_t>(bytes[i * 2]) |
                              (static_cast<uint16_t>(static_cast<uint8_t>(bytes[i * 2 + 1])) << 8);
        const uint32_t bits32 = static_cast<uint32_t>(bits) << 16;
        __builtin_memcpy(&values[i], &bits32, sizeof(values[i]));
      }
      weights.emplace(want.name, std::move(values));
    }
    if (weights.size() != 76)
      throw std::invalid_argument("Base checkpoint must contain 76 speaker encoder tensors");
  }

  Tensor conv(const Tensor& x, const std::string& prefix, int out_channels, int kernel,
              int dilation = 1, bool relu = true) const {
    if (kernel % 2 == 0 || dilation <= 0)
      throw std::invalid_argument("speaker conv requires an odd kernel and positive dilation");
    const auto wi = weights.find(prefix + ".weight");
    const auto bi = weights.find(prefix + ".bias");
    if (wi == weights.end() || bi == weights.end())
      throw std::runtime_error("missing speaker encoder weights: " + prefix);
    const std::vector<float>& w = wi->second;
    const std::vector<float>& bias = bi->second;
    const size_t expected = static_cast<size_t>(out_channels) * x.channels * kernel;
    if (w.size() != expected || bias.size() != static_cast<size_t>(out_channels))
      throw std::runtime_error("speaker encoder weight size mismatch: " + prefix);

    const int pad = (kernel - 1) / 2 * dilation;
    if (pad > 0 && x.frames <= pad)
      throw std::invalid_argument("reference audio has too few frames for speaker convolution");
    Tensor padded(x.channels, x.frames + 2 * pad);
    for (int c = 0; c < x.channels; ++c) {
      const float* src = x.data.data() + static_cast<size_t>(c) * x.frames;
      float* dst = padded.data.data() + static_cast<size_t>(c) * padded.frames;
      for (int t = 0; t < padded.frames; ++t)
        dst[t] = src[reflect_index(t - pad, x.frames)];
    }

    Tensor out(out_channels, x.frames);
    for (int o = 0; o < out_channels; ++o) {
      float* dst = out.data.data() + static_cast<size_t>(o) * x.frames;
      std::fill(dst, dst + x.frames, bias[o]);
      for (int i = 0; i < x.channels; ++i) {
        const float* src = padded.data.data() + static_cast<size_t>(i) * padded.frames;
        const float* weights_oc = w.data() + (static_cast<size_t>(o) * x.channels + i) * kernel;
        for (int k = 0; k < kernel; ++k) {
          const float weight = weights_oc[k];
          const float* shifted = src + k * dilation;
          for (int t = 0; t < x.frames; ++t) dst[t] += weight * shifted[t];
        }
      }
      if (relu)
        for (int t = 0; t < x.frames; ++t) dst[t] = std::max(dst[t], 0.0f);
    }
    return out;
  }

  Tensor res2net(const Tensor& x, const std::string& prefix, int dilation) const {
    if (x.channels != 512) throw std::runtime_error("Res2Net expects 512 channels");
    Tensor out(512, x.frames);
    for (int c = 0; c < 64; ++c) {
      const auto first = x.data.begin() + static_cast<size_t>(c) * x.frames;
      std::copy(first, first + x.frames,
                out.data.begin() + static_cast<size_t>(c) * x.frames);
    }
    Tensor previous;
    for (int j = 0; j < 7; ++j) {
      Tensor part(64, x.frames);
      for (int c = 0; c < 64; ++c) {
        const float* src = x.data.data() + static_cast<size_t>((j + 1) * 64 + c) * x.frames;
        float* dst = part.data.data() + static_cast<size_t>(c) * x.frames;
        if (j == 0) {
          std::copy(src, src + x.frames, dst);
        } else {
          const float* add = previous.data.data() + static_cast<size_t>(c) * x.frames;
          for (int t = 0; t < x.frames; ++t) dst[t] = src[t] + add[t];
        }
      }
      previous = conv(part, prefix + ".blocks." + std::to_string(j) + ".conv", 64, 3,
                      dilation);
      for (int c = 0; c < 64; ++c) {
        const float* src = previous.data.data() + static_cast<size_t>(c) * x.frames;
        std::copy(src, src + x.frames,
                  out.data.begin() + static_cast<size_t>((j + 1) * 64 + c) * x.frames);
      }
    }
    return out;
  }

  Tensor se_res2net(const Tensor& residual, int block, int dilation) const {
    const std::string prefix = "speaker_encoder.blocks." + std::to_string(block);
    Tensor x = conv(residual, prefix + ".tdnn1.conv", 512, 1);
    x = res2net(x, prefix + ".res2net_block", dilation);
    x = conv(x, prefix + ".tdnn2.conv", 512, 1);

    Tensor mean(512, 1);
    for (int c = 0; c < 512; ++c) {
      const float* row = x.data.data() + static_cast<size_t>(c) * x.frames;
      float sum = 0.0f;
      for (int t = 0; t < x.frames; ++t) sum += row[t];
      mean.data[c] = sum / x.frames;
    }
    Tensor scale = conv(mean, prefix + ".se_block.conv1", 128, 1);
    scale = conv(scale, prefix + ".se_block.conv2", 512, 1, 1, false);
    for (float& v : scale.data) v = 1.0f / (1.0f + std::exp(-v));
    Tensor out(512, x.frames);
    for (int c = 0; c < 512; ++c) {
      const float* h = x.data.data() + static_cast<size_t>(c) * x.frames;
      const float* r = residual.data.data() + static_cast<size_t>(c) * x.frames;
      float* dst = out.data.data() + static_cast<size_t>(c) * x.frames;
      for (int t = 0; t < x.frames; ++t) dst[t] = r[t] + h[t] * scale.data[c];
    }
    return out;
  }

  std::vector<float> extract(const std::vector<float>& pcm) const {
    SpeakerMel mel = compute_speaker_log_mel(pcm);
    if (mel.frames <= 4) throw std::invalid_argument("reference audio is too short for ECAPA-TDNN");
    Tensor initial = conv(Tensor(128, mel.frames, std::move(mel.log_mel)),
                          "speaker_encoder.blocks.0.conv", 512, 5);
    std::array<Tensor, 3> blocks;
    const Tensor* current = &initial;
    const std::array<int, 3> dilations = {2, 3, 4};
    for (int b = 0; b < 3; ++b) {
      blocks[b] = se_res2net(*current, b + 1, dilations[b]);
      current = &blocks[b];
    }

    Tensor joined(1536, initial.frames);
    for (int b = 0; b < 3; ++b)
      for (int c = 0; c < 512; ++c) {
        const float* src = blocks[b].data.data() + static_cast<size_t>(c) * initial.frames;
        std::copy(src, src + initial.frames,
                  joined.data.begin() + static_cast<size_t>(b * 512 + c) * initial.frames);
      }
    Tensor x = conv(joined, "speaker_encoder.mfa.conv", 1536, 1);
    const int frames = x.frames;

    Tensor attention_input(4608, frames);
    for (int c = 0; c < 1536; ++c) {
      const float* row = x.data.data() + static_cast<size_t>(c) * frames;
      float sum = 0.0f;
      for (int t = 0; t < frames; ++t) sum += row[t];
      const float mean = sum / frames;
      float variance = 0.0f;
      for (int t = 0; t < frames; ++t) {
        const float d = row[t] - mean;
        variance += d * d;
      }
      const float stddev = std::sqrt(std::max(variance / frames, 1e-12f));
      float* dst = attention_input.data.data() + static_cast<size_t>(c) * frames;
      std::copy(row, row + frames, dst);
      std::fill(attention_input.data.begin() + static_cast<size_t>(1536 + c) * frames,
                attention_input.data.begin() + static_cast<size_t>(1536 + c + 1) * frames,
                mean);
      std::fill(attention_input.data.begin() + static_cast<size_t>(3072 + c) * frames,
                attention_input.data.begin() + static_cast<size_t>(3072 + c + 1) * frames,
                stddev);
    }
    Tensor attention = conv(attention_input, "speaker_encoder.asp.tdnn.conv", 128, 1);
    for (float& v : attention.data) v = std::tanh(v);
    attention = conv(attention, "speaker_encoder.asp.conv", 1536, 1, 1, false);

    Tensor pooled(3072, 1);
    std::vector<float> probability(frames);
    for (int c = 0; c < 1536; ++c) {
      const float* row = x.data.data() + static_cast<size_t>(c) * frames;
      const float* logits = attention.data.data() + static_cast<size_t>(c) * frames;
      float maximum = -std::numeric_limits<float>::infinity();
      for (int t = 0; t < frames; ++t) maximum = std::max(maximum, logits[t]);
      float denominator = 0.0f;
      for (int t = 0; t < frames; ++t) {
        probability[t] = std::exp(logits[t] - maximum);
        denominator += probability[t];
      }
      float mean = 0.0f;
      for (int t = 0; t < frames; ++t) mean += probability[t] * row[t];
      mean /= denominator;
      float variance = 0.0f;
      for (int t = 0; t < frames; ++t) {
        const float d = row[t] - mean;
        variance += probability[t] * d * d;
      }
      pooled.data[c] = mean;
      pooled.data[1536 + c] = std::sqrt(std::max(variance / denominator, 1e-12f));
    }
    Tensor out = conv(pooled, "speaker_encoder.fc", cfg.enc_dim, 1, 1, false);
    return std::move(out.data);
  }
};

SpeakerEncoder::SpeakerEncoder(const SafeTensors& weights, const TtsModelConfig& config)
    : impl_(std::make_unique<Impl>(weights, config)) {}
SpeakerEncoder::~SpeakerEncoder() = default;

std::vector<float> SpeakerEncoder::extract(const std::vector<float>& ref_pcm_24k) const {
  return impl_->extract(ref_pcm_24k);
}

int SpeakerEncoder::output_dim() const { return impl_->cfg.enc_dim; }

}  // namespace umminfer::native
