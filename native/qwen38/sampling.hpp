// SPDX-License-Identifier: Apache-2.0
// Host-side token sampler, shared by the serial and the batched CUDA paths.
// Header-only so the CUDA targets pick it up without another translation unit.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include "engine.hpp"

namespace umminfer::qwen38 {

namespace detail {

// The probability key is the IEEE bit pattern of a non-negative float, which
// orders exactly like the value. The histogram keeps its top 12 bits: one sign
// bit, the 8 exponent bits and 3 mantissa bits, so consecutive buckets are 9%
// apart in value and a 248k row lands in a few hundred of them.
constexpr int kNucleusBucketShift = 20;
constexpr int kNucleusBuckets = 1 << (32 - kNucleusBucketShift);

inline uint32_t prob_key(float p) {
  return std::bit_cast<uint32_t>(p);
}

// Packs a candidate so that a descending sort of the packed words is a
// descending sort by probability with ascending index as the tie-break, which
// is the order the full sort used.
inline uint64_t nucleus_key(float p, int index) {
  return (static_cast<uint64_t>(prob_key(p)) << 32) |
         (0xFFFFFFFFu - static_cast<uint32_t>(index));
}

inline float key_prob(uint64_t key) {
  return std::bit_cast<float>(static_cast<uint32_t>(key >> 32));
}

inline int key_index(uint64_t key) {
  return static_cast<int>(0xFFFFFFFFu - static_cast<uint32_t>(key));
}

}  // namespace detail

// Full FP32-row log-softmax for behavior evidence. Sampling transformations are
// intentionally not reapplied: the value is always the selected id's raw model
// log-probability under the exact pre-sampling logits, for greedy and sampled
// decode alike. Degenerate rows fail closed instead of producing placeholders.
inline bool selected_token_logprob(const float* logits, int n, int id,
                                   double* out) {
  if (out == nullptr) return false;
  *out = std::numeric_limits<double>::quiet_NaN();
  if (logits == nullptr || n <= 0 || id < 0 || id >= n) return false;
  double maxv = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < n; ++i) {
    const double value = logits[i];
    if (!std::isfinite(value)) return false;
    maxv = std::max(maxv, value);
  }
  double total = 0.0;
  for (int i = 0; i < n; ++i) {
    total += std::exp(static_cast<double>(logits[i]) - maxv);
  }
  if (!(total > 0.0) || !std::isfinite(total)) return false;
  const double value = static_cast<double>(logits[id]) - maxv - std::log(total);
  if (!std::isfinite(value) || value > 0.0) return false;
  *out = value;
  return true;
}

// Draws one token id from `n` fp32 logits.
//
// temperature <= 0 keeps the exact argmax, so the default request is bitwise
// greedy. temperature > 0 scales the logits, applies softmax, keeps the
// smallest set of most-likely tokens whose probability mass reaches top_p, and
// draws once from `rng`.
//
// Non-finite logits never reach the arithmetic: NaN becomes the lowest finite
// value and +/-infinity is clamped, so a degenerate row returns a token instead
// of producing a NaN probability.
//
// Nucleus selection. A full sort of the 248k vocabulary costs ~15 ms per token.
// Instead the code builds one histogram of the probability mass over the
// exponent buckets of the row, then takes the smallest floor whose buckets
// already hold top_p of the mass. Every index below that floor has probability
// strictly below every index at or above it, so the sorted candidate list is a
// prefix of the full sorted order with the same addends in the same order: the
// cutoff, the cumulative sum and the draw are identical to sorting the whole
// row, while the sort only sees the nucleus. A row whose mass is spread over
// every bucket (a flat temperature, or top_p == 1 over a heavy tail) keeps the
// whole row as its candidate list, which is the original behaviour.
inline int sample_token(const float* logits, int n, const SamplingParams& params,
                        std::mt19937_64& rng) {
  if (logits == nullptr || n <= 0) return 0;
  constexpr double kLimit =
      static_cast<double>(std::numeric_limits<float>::max());
  double maxv = -kLimit;
  for (int i = 0; i < n; ++i) {
    const double v = logits[i];
    if (std::isnan(v)) continue;
    if (v > maxv) maxv = std::min(v, kLimit);
  }
  const bool greedy = !(params.temperature > 0.0f) ||
                      !std::isfinite(params.temperature);
  if (greedy) {
    // First index of the maximum, matching the strict-greater argmax kernel.
    int best = 0;
    double bestv = -kLimit;
    for (int i = 0; i < n; ++i) {
      const double v = logits[i];
      if (std::isnan(v)) continue;
      if (v > bestv) {
        bestv = std::min(v, kLimit);
        best = i;
      }
    }
    return best;
  }

  double inv_t = 1.0 / static_cast<double>(params.temperature);
  if (!std::isfinite(inv_t)) inv_t = 1e30;  // subnormal temperature
  // A subnormal temperature keeps inv_t finite in fp64 but saturates it to
  // +inf once narrowed to fp32, and the argmax row would then evaluate
  // 0.0f * inf as NaN. Saturate before narrowing.
  const float inv_tf = static_cast<float>(std::min(inv_t, 1e30));

  // fp32 exp, fp64 sum. The nucleus only needs the ordering the probabilities
  // induce, and a float exponential is several times cheaper than a double one
  // over a 248k vocabulary; the sum stays fp64 so the normaliser does not drift.
  // The clamp anchor is either a logit or FLT_MAX, both exactly representable
  // as float, so the difference below rounds to the same float either way.
  constexpr float kLimitF = std::numeric_limits<float>::max();
  const float maxf = static_cast<float>(maxv);
  std::vector<float> prob(static_cast<size_t>(n));
  double total = 0.0;
  for (int i = 0; i < n; ++i) {
    const float v = logits[i];
    float p = 0.0f;
    if (!std::isnan(v)) p = std::exp((std::min(v, kLimitF) - maxf) * inv_tf);
    prob[static_cast<size_t>(i)] = p;
    total += p;
  }
  if (!(total > 0.0) || !std::isfinite(total)) return 0;

  double top_p = static_cast<double>(params.top_p);
  if (!std::isfinite(top_p) || top_p <= 0.0 || top_p > 1.0) top_p = 1.0;

  // Normalise and bucket in one pass: the row is then read only once more.
  std::array<double, detail::kNucleusBuckets> mass{};
  for (int i = 0; i < n; ++i) {
    const float p = static_cast<float>(
        static_cast<double>(prob[static_cast<size_t>(i)]) / total);
    prob[static_cast<size_t>(i)] = p;
    mass[detail::prob_key(p) >> detail::kNucleusBucketShift] += p;
  }
  double covered = 0.0;
  int bucket = detail::kNucleusBuckets;
  while (bucket > 0) {
    --bucket;
    covered += mass[static_cast<size_t>(bucket)];
    if (covered >= top_p) break;
  }
  const float floor_p = std::bit_cast<float>(
      static_cast<uint32_t>(bucket) << detail::kNucleusBucketShift);

  std::vector<uint64_t> keys;
  keys.reserve(static_cast<size_t>(n) / 8 + 1);
  for (int i = 0; i < n; ++i) {
    const float p = prob[static_cast<size_t>(i)];
    if (p >= floor_p) keys.push_back(detail::nucleus_key(p, i));
  }
  if (keys.empty()) return 0;
  std::sort(keys.begin(), keys.end(), std::greater<uint64_t>());

  const int ncand = static_cast<int>(keys.size());
  int keep = 0;
  double cumulative = 0.0;
  while (keep < ncand) {
    cumulative += detail::key_prob(keys[static_cast<size_t>(keep)]);
    ++keep;
    if (cumulative >= top_p) break;
  }
  if (cumulative <= 0.0) return detail::key_index(keys[0]);

  std::uniform_real_distribution<double> draw(0.0, cumulative);
  const double r = draw(rng);
  double acc = 0.0;
  for (int k = 0; k < keep; ++k) {
    acc += detail::key_prob(keys[static_cast<size_t>(k)]);
    if (r < acc) return detail::key_index(keys[static_cast<size_t>(k)]);
  }
  return detail::key_index(keys[static_cast<size_t>(keep - 1)]);
}

}  // namespace umminfer::qwen38
