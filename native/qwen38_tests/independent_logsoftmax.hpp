// SPDX-License-Identifier: Apache-2.0
// Independent validation arithmetic for raw Qwen3.8 pre-sampling FP32 logits.
// Deliberately does not include or call qwen38/sampling.hpp.
#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

namespace umminfer::qwen38_validation {

struct IndependentLogSoftmax {
  double selected_raw_logit = 0.0;
  double max_raw_logit = 0.0;
  double log_normalizer = 0.0;
  double logprob = 0.0;
};

// The input row and selected raw logit are FP32. Accumulation deliberately uses
// extended precision rather than the producer's FP64 loop, providing a distinct
// numerical implementation for fresh-process agreement checks.
inline bool independent_presampling_logsoftmax(const float* logits, int n, int id,
                                               IndependentLogSoftmax* out) {
  if (out == nullptr) return false;
  *out = {};
  if (logits == nullptr || n <= 0 || id < 0 || id >= n) return false;
  long double maximum = -std::numeric_limits<long double>::infinity();
  for (int i = 0; i < n; ++i) {
    const long double value = static_cast<long double>(logits[i]);
    if (!std::isfinite(value)) return false;
    if (value > maximum) maximum = value;
  }
  long double normalizer = 0.0L;
  for (int i = 0; i < n; ++i) {
    normalizer += std::exp(static_cast<long double>(logits[i]) - maximum);
  }
  if (!(normalizer > 0.0L) || !std::isfinite(normalizer)) return false;
  const long double log_normalizer = std::log(normalizer);
  const long double selected = static_cast<long double>(logits[id]);
  const long double logprob = selected - maximum - log_normalizer;
  if (!std::isfinite(log_normalizer) || !std::isfinite(logprob) || logprob > 0.0L) {
    return false;
  }
  out->selected_raw_logit = static_cast<double>(selected);
  out->max_raw_logit = static_cast<double>(maximum);
  out->log_normalizer = static_cast<double>(log_normalizer);
  out->logprob = static_cast<double>(logprob);
  return std::isfinite(out->selected_raw_logit) &&
         std::isfinite(out->max_raw_logit) && std::isfinite(out->log_normalizer) &&
         std::isfinite(out->logprob) && out->logprob <= 0.0;
}

}  // namespace umminfer::qwen38_validation
