// SPDX-License-Identifier: Apache-2.0
#include "feature_frontend.hpp"

#include <algorithm>
#include <cmath>
#include <thread>

namespace umminfer::native {
namespace {

constexpr double kPi = 3.14159265358979323846;

// Slaney mel scale, matching transformers.audio_utils hertz_to_mel.
double hz_to_mel(double hz) {
  constexpr double kMinLogHz = 1000.0;
  constexpr double kMinLogMel = 15.0;
  const double logstep = 27.0 / std::log(6.4);
  double mel = 3.0 * hz / 200.0;
  if (hz >= kMinLogHz) mel = kMinLogMel + std::log(hz / kMinLogHz) * logstep;
  return mel;
}

double mel_to_hz(double mel) {
  constexpr double kMinLogMel = 15.0;
  const double logstep = std::log(6.4) / 27.0;
  double hz = 200.0 * mel / 3.0;
  if (mel >= kMinLogMel) hz = 1000.0 * std::exp(logstep * (mel - kMinLogMel));
  return hz;
}

// np.linspace over doubles.
std::vector<double> linspace(double lo, double hi, int n) {
  std::vector<double> out(n);
  const double step = (hi - lo) / static_cast<double>(n - 1);
  for (int i = 0; i < n; ++i) out[i] = lo + step * i;
  return out;
}

}  // namespace

WhisperMel::WhisperMel() {
  window_.resize(kNfft);
  for (int n = 0; n < kNfft; ++n)
    window_[n] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * n / kNfft));

  // Slaney-normalized triangular mel filter bank, shape [201][128].
  const double mel_min = hz_to_mel(0.0);
  const double mel_max = hz_to_mel(8000.0);
  const std::vector<double> mel_points = linspace(mel_min, mel_max, kMels + 2);
  std::vector<double> filter_freqs(kMels + 2);
  for (int i = 0; i < kMels + 2; ++i) filter_freqs[i] = mel_to_hz(mel_points[i]);
  const std::vector<double> fft_freqs = linspace(0.0, 8000.0, kFftBins);

  std::vector<double> filter_diff(kMels + 1);
  for (int i = 0; i < kMels + 1; ++i)
    filter_diff[i] = filter_freqs[i + 1] - filter_freqs[i];

  mel_bank_.assign(kFftBins * kMels, 0.0f);
  for (int b = 0; b < kFftBins; ++b) {
    for (int m = 0; m < kMels; ++m) {
      // Mirrors transformers _create_triangular_filter_bank:
      // down = (fft - ff[m]) / (ff[m+1] - ff[m])
      // up   = (ff[m+2] - fft) / (ff[m+2] - ff[m+1])
      const double down = (fft_freqs[b] - filter_freqs[m]) / filter_diff[m];
      const double up = (filter_freqs[m + 2] - fft_freqs[b]) / filter_diff[m + 1];
      double weight = std::min(down, up);
      if (weight < 0.0) weight = 0.0;
      mel_bank_[b * kMels + m] = static_cast<float>(weight);
    }
  }
  // Slaney area normalization per mel channel.
  for (int m = 0; m < kMels; ++m) {
    const double enorm = 2.0 / (filter_freqs[m + 2] - filter_freqs[m]);
    for (int b = 0; b < kFftBins; ++b) mel_bank_[b * kMels + m] *= static_cast<float>(enorm);
  }

  // DFT basis, one-sided: 201 bins, 400 taps.
  dft_cos_.resize(kFftBins * kNfft);
  dft_sin_.resize(kFftBins * kNfft);
  for (int k = 0; k < kFftBins; ++k) {
    for (int n = 0; n < kNfft; ++n) {
      const double angle = 2.0 * kPi * k * n / kNfft;
      dft_cos_[k * kNfft + n] = static_cast<float>(std::cos(angle));
      dft_sin_[k * kNfft + n] = static_cast<float>(std::sin(angle));
    }
  }
}

int WhisperMel::valid_frame_count(int num_samples) {
  if (num_samples <= 0) return 0;
  // Mask is the 30 s-padded waveform subsampled at hop 160; last STFT frame
  // is dropped afterwards, so the mask is capped at 3000 as well.
  const int frames = (num_samples - 1) / kHop + 1;
  return std::min(frames, kChunkFrames);
}

MelFeatures WhisperMel::compute(const std::vector<float>& samples_in) const {
  std::vector<double> wave(kChunkSamples, 0.0);
  const int real = std::min<int>(samples_in.size(), kChunkSamples);
  for (int i = 0; i < real; ++i) wave[i] = static_cast<double>(samples_in[i]);

  // STFT with center=True, reflect padding (numpy convention: edge not repeated).
  const int padded_len = kChunkSamples + 2 * (kNfft / 2);
  auto at = [&](int idx) -> double {
    if (idx >= kNfft / 2 && idx < kNfft / 2 + kChunkSamples)
      return wave[idx - kNfft / 2];
    if (idx < kNfft / 2) return wave[kNfft / 2 - idx];          // left reflect
    const int d = idx - kNfft / 2;
    return wave[2 * kChunkSamples - 2 - d];                    // right reflect
  };

  const int raw_frames = 1 + (padded_len - kNfft) / kHop;  // 3001 for a full chunk
  std::vector<double> power(static_cast<size_t>(raw_frames) * kFftBins, 0.0);

  const int threads = std::min(8, static_cast<int>(std::thread::hardware_concurrency()));
  auto worker = [&](int t0, int t1) {
    std::vector<double> frame(kNfft);
    for (int t = t0; t < t1; ++t) {
      const int center = t * kHop;
      for (int n = 0; n < kNfft; ++n)
        frame[n] = at(center + n) * static_cast<double>(window_[n]);
      for (int k = 0; k < kFftBins; ++k) {
        double re = 0.0, im = 0.0;
        const int base = k * kNfft;
        for (int n = 0; n < kNfft; ++n) {
          re += frame[n] * dft_cos_[base + n];
          im -= frame[n] * dft_sin_[base + n];
        }
        power[static_cast<size_t>(t) * kFftBins + k] = re * re + im * im;
      }
    }
  };
  std::vector<std::thread> pool;
  for (int i = 0; i < threads; ++i) {
    const int t0 = raw_frames * i / threads;
    const int t1 = raw_frames * (i + 1) / threads;
    if (t1 > t0) pool.emplace_back(worker, t0, t1);
  }
  for (auto& th : pool) th.join();

  // Mel projection + log, dropping the final STFT frame like Whisper.
  MelFeatures out;
  out.num_frames = kChunkFrames;
  out.valid_frames = valid_frame_count(real);
  out.data.assign(static_cast<size_t>(kMels) * kChunkFrames, 0.0f);

  double log_max = -std::numeric_limits<double>::infinity();
  std::vector<double> logmel(static_cast<size_t>(kMels) * kChunkFrames);
  for (int t = 0; t < kChunkFrames; ++t) {
    for (int m = 0; m < kMels; ++m) {
      double acc = 0.0;
      for (int b = 0; b < kFftBins; ++b)
        acc += static_cast<double>(mel_bank_[b * kMels + m]) *
               power[static_cast<size_t>(t) * kFftBins + b];
      if (acc < 1e-10) acc = 1e-10;
      const double v = std::log10(acc);
      logmel[static_cast<size_t>(m) * kChunkFrames + t] = v;
      log_max = std::max(log_max, v);
    }
  }
  const double floor = log_max - 8.0;
  for (size_t i = 0; i < logmel.size(); ++i) {
    const double v = std::max(logmel[i], floor);
    out.data[i] = static_cast<float>((v + 4.0) / 4.0);
  }
  return out;
}

}  // namespace umminfer::native
