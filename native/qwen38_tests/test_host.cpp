// SPDX-License-Identifier: Apache-2.0
// Assert-based host checks for the Qwen3.8-27B sampling and prefix-reuse
// helpers. No GPU, no framework.
#include "qwen38/prefix_cache.hpp"
#include "qwen38/sampling.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

namespace {

using umminfer::qwen38::match_prefix;
using umminfer::qwen38::SamplingParams;
using umminfer::qwen38::sample_token;

int g_failures = 0;

void check(bool cond, const char* what) {
  if (!cond) {
    std::printf("FAIL: %s\n", what);
    ++g_failures;
  }
}

SamplingParams sampling(float temperature, float top_p, uint64_t seed) {
  SamplingParams params;
  params.temperature = temperature;
  params.top_p = top_p;
  params.seed = seed;
  params.has_seed = true;
  return params;
}

std::vector<int> draw(const std::vector<float>& logits, const SamplingParams& params,
                      int count) {
  std::mt19937_64 rng(params.seed);
  std::vector<int> out;
  out.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i)
    out.push_back(sample_token(logits.data(), static_cast<int>(logits.size()),
                               params, rng));
  return out;
}

void test_greedy_is_argmax() {
  const std::vector<float> logits{0.1f, 5.0f, -1.0f, 3.2f};
  std::mt19937_64 rng(7);
  check(sample_token(logits.data(), 4, sampling(0.0f, 1.0f, 1), rng) == 1,
        "temperature=0 picks the argmax");
  check(sample_token(logits.data(), 4, sampling(-1.0f, 1.0f, 1), rng) == 1,
        "negative temperature picks the argmax");
  SamplingParams nan_temp = sampling(1.0f, 1.0f, 1);
  nan_temp.temperature = std::numeric_limits<float>::quiet_NaN();
  check(sample_token(logits.data(), 4, nan_temp, rng) == 1,
        "NaN temperature falls back to greedy");
}

void test_sampling_is_seed_deterministic() {
  const std::vector<float> logits{2.0f, 1.0f, 0.5f, 0.0f, -1.0f};
  const auto a = draw(logits, sampling(0.7f, 0.9f, 42), 200);
  const auto b = draw(logits, sampling(0.7f, 0.9f, 42), 200);
  const auto c = draw(logits, sampling(0.7f, 0.9f, 43), 200);
  check(a == b, "same seed reproduces the same token sequence");
  check(a != c, "a different seed produces a different sequence");
  bool saw_second = false;
  for (int id : a) saw_second = saw_second || id == 1;
  check(saw_second, "temperature>0 does not collapse to greedy");
}

void test_top_p_excludes_tail() {
  // p = {0.5, 0.25, 0.25}; a nucleus of 0.4 keeps only the first token.
  const std::vector<float> logits{static_cast<float>(std::log(2.0)), 0.0f, 0.0f};
  for (int id : draw(logits, sampling(1.0f, 0.4f, 11), 5000))
    check(id == 0, "top_p drops the low-probability tail");
  // top_p = 1 keeps the whole row, so the tail does appear.
  bool saw_tail = false;
  for (int id : draw(logits, sampling(1.0f, 1.0f, 11), 5000))
    saw_tail = saw_tail || id != 0;
  check(saw_tail, "top_p=1 keeps the full distribution");
}

void test_extreme_logits() {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<float> logits{inf, 0.0f, -inf, nan};
  std::mt19937_64 rng(3);
  check(sample_token(logits.data(), 4, sampling(1.0f, 0.9f, 3), rng) == 0,
        "an infinite logit dominates without a NaN probability");
  const std::vector<float> dead{-inf, nan, -inf};
  const int id = sample_token(dead.data(), 3, sampling(1.0f, 1.0f, 3), rng);
  check(id >= 0 && id < 3, "an all-degenerate row returns a valid index");
  check(sample_token(dead.data(), 3, sampling(0.0f, 1.0f, 3), rng) == 0,
        "an all-degenerate greedy row returns a valid index");
}

void test_prefix_match() {
  const std::vector<int> cached{5, 6, 7};
  check(match_prefix(cached, {5, 6, 7, 8}, false, true) == 3,
        "exact prefix is reused");
  check(match_prefix(cached, {5, 6, 9, 8}, false, true) == 0,
        "a mid-prefix mismatch disables reuse");
  check(match_prefix(cached, {5, 6}, false, true) == 0,
        "a shorter prompt disables reuse");
  check(match_prefix(cached, {5, 6, 7}, false, true) == 0,
        "an equal-length prompt disables reuse");
  check(match_prefix(cached, {5, 6, 7, 8}, true, true) == 0,
        "an image request bypasses the snapshot");
  check(match_prefix(cached, {5, 6, 7, 8}, false, false) == 0,
        "a disabled cache returns no reuse");
  check(match_prefix({}, {1, 2}, false, true) == 0,
        "an empty cache returns no reuse");
}

}  // namespace

int main() {
  test_greedy_is_argmax();
  test_sampling_is_seed_deterministic();
  test_top_p_excludes_tail();
  test_extreme_logits();
  test_prefix_match();
  if (g_failures != 0) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("qwen38 host tests passed\n");
  return 0;
}
