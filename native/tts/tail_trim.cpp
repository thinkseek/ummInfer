// SPDX-License-Identifier: Apache-2.0
#include "tail_trim.hpp"

#include <algorithm>
#include <cmath>

namespace umminfer::native {
namespace {

// Thresholds are stated in PCM16 units, the domain audio_analyze.py and the
// listening evidence use. The codec returns floats in [-1, 1], so each window
// is scaled before it is compared.
constexpr double kSilentRmsPcm16 = 60.0;
constexpr double kPcm16Scale = 32768.0;
constexpr double kNoiseZcrRatio = 0.4;
constexpr int kWindowMs = 10;
// A real fricative measured at up to 0.16 s and the decoder's noise run at
// 0.24 s, so 0.20 s keeps every fricative and still removes the noise.
constexpr int kMinNoiseMs = 200;
constexpr int kMaxNoiseMs = 600;

double window_rms(const float* p, std::size_t n) {
  if (n == 0) return 0.0;
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) sum += double(p[i]) * p[i];
  return std::sqrt(sum / double(n));
}

double window_zcr(const float* p, std::size_t n, int sample_rate) {
  if (n < 2) return 0.0;
  std::size_t crossings = 0;
  for (std::size_t i = 1; i < n; ++i)
    if ((p[i - 1] < 0.0f) != (p[i] < 0.0f)) ++crossings;
  return double(crossings) * double(sample_rate) / double(n);
}

}  // namespace

std::size_t trailing_noise_cut(const std::vector<float>& samples,
                               int sample_rate) {
  if (sample_rate <= 0 || samples.empty()) return 0;
  const std::size_t window =
      std::max<std::size_t>(1, std::size_t(sample_rate) * kWindowMs / 1000);
  const std::size_t min_noise = std::size_t(sample_rate) * kMinNoiseMs / 1000;
  const std::size_t max_noise = std::size_t(sample_rate) * kMaxNoiseMs / 1000;
  const double zcr_threshold = kNoiseZcrRatio * double(sample_rate);

  const std::size_t end = samples.size();
  std::size_t cursor = end;
  std::size_t noise_only = 0;
  std::size_t budget_end = 0;  // where the noise budget runs out
  bool reached_voice = false;
  // One walk back from the end over a single run: silence and noise windows,
  // silence between them skipped. The first other window ends the run.
  while (cursor > 0) {
    const std::size_t begin = cursor > window ? cursor - window : 0;
    const std::size_t n = cursor - begin;
    const double rms = window_rms(samples.data() + begin, n) * kPcm16Scale;
    const bool silent = rms < kSilentRmsPcm16;
    const bool noise =
        !silent && window_zcr(samples.data() + begin, n, sample_rate) >= zcr_threshold;
    if (!silent && !noise) {
      reached_voice = true;
      break;
    }
    if (noise) {
      noise_only += n;
      // The cap bounds how much noise is removed. The silence on either side
      // of it is not speech, so it does not consume the budget; counting it
      // left the noise in place whenever an utterance ended with a pause.
      if (budget_end == 0 && noise_only >= max_noise) budget_end = begin;
    }
    cursor = begin;
  }
  // An output with no speech window at all is degenerate, not a tail to trim.
  if (!reached_voice) return 0;
  // A run shorter than 0.20 s could be a final fricative; keep the audio.
  if (noise_only < min_noise) return 0;
  if (budget_end != 0 && budget_end > cursor) cursor = budget_end;
  return end - cursor;
}

void fade_out(std::vector<float>& samples, std::size_t n) {
  const std::size_t count = std::min(n, samples.size());
  if (count == 0) return;
  for (std::size_t i = 0; i < count; ++i) {
    const float gain = 1.0f - float(i + 1) / float(count);
    samples[samples.size() - count + i] *= gain;
  }
}

std::vector<float> trim_trailing_noise(std::vector<float> samples,
                                       int sample_rate) {
  const std::size_t cut = trailing_noise_cut(samples, sample_rate);
  if (cut == 0 || cut >= samples.size()) return samples;
  samples.resize(samples.size() - cut);
  fade_out(samples, std::size_t(sample_rate) * 5 / 1000);
  return samples;
}

}  // namespace umminfer::native
