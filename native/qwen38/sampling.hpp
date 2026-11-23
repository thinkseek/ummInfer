// SPDX-License-Identifier: Apache-2.0
// Host-side token sampler, shared by the serial and the batched CUDA paths.
// Header-only so the CUDA targets pick it up without another translation unit.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "engine.hpp"

namespace umminfer::qwen38 {

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

  std::vector<double> prob(static_cast<size_t>(n));
  double total = 0.0;
  for (int i = 0; i < n; ++i) {
    const double v = logits[i];
    double p = 0.0;
    if (!std::isnan(v)) {
      const double d = std::min(v, kLimit) - maxv;  // <= 0
      p = std::exp(d * inv_t);
    }
    prob[static_cast<size_t>(i)] = p;
    total += p;
  }
  if (!(total > 0.0) || !std::isfinite(total)) return 0;
  for (double& p : prob) p /= total;

  double top_p = static_cast<double>(params.top_p);
  if (!std::isfinite(top_p) || top_p <= 0.0 || top_p > 1.0) top_p = 1.0;

  // ponytail: full sort per token. A 248k vocabulary costs a few milliseconds;
  // replace with a bucket/partial selection if decode latency matters.
  std::vector<int> order(static_cast<size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    const double pa = prob[static_cast<size_t>(a)];
    const double pb = prob[static_cast<size_t>(b)];
    return pa != pb ? pa > pb : a < b;
  });

  int keep = 0;
  double cumulative = 0.0;
  while (keep < n) {
    cumulative += prob[static_cast<size_t>(order[static_cast<size_t>(keep)])];
    ++keep;
    if (cumulative >= top_p) break;
  }
  if (cumulative <= 0.0) return order[0];

  std::uniform_real_distribution<double> draw(0.0, cumulative);
  const double r = draw(rng);
  double acc = 0.0;
  for (int k = 0; k < keep; ++k) {
    acc += prob[static_cast<size_t>(order[static_cast<size_t>(k)])];
    if (r < acc) return order[static_cast<size_t>(k)];
  }
  return order[static_cast<size_t>(keep - 1)];
}

}  // namespace umminfer::qwen38
