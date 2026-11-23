// SPDX-License-Identifier: Apache-2.0
// Assert-based host checks for the Qwen3.8-27B sampling and prefix-reuse
// helpers. No GPU, no framework.
#include "qwen38/prefix_cache.hpp"
#include "qwen38/sampling.hpp"

#include <algorithm>
#include <algorithm>
#include <algorithm>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
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

// The full-sort nucleus selector this header replaced. It shares the softmax
// with the fast path and differs only in ordering the whole row, so it is the
// reference the fast path must reproduce bit for bit, including the exact rng
// consumption.
int reference_sample(const float* logits, int n, const SamplingParams& params,
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
  if (!std::isfinite(inv_t)) inv_t = 1e30;
  // Same softmax as sample_token, so the only thing under test is the selection:
  // which indices the sorter sees, in which order, and the draw over them.
  const float inv_tf = static_cast<float>(std::min(inv_t, 1e30));
  std::vector<float> prob(static_cast<size_t>(n));
  double total = 0.0;
  for (int i = 0; i < n; ++i) {
    const double v = logits[i];
    float p = 0.0f;
    if (!std::isnan(v)) {
      const double d = std::min(v, kLimit) - maxv;
      p = std::exp(static_cast<float>(d) * inv_tf);
    }
    prob[static_cast<size_t>(i)] = p;
    total += p;
  }
  if (!(total > 0.0) || !std::isfinite(total)) return 0;
  for (float& p : prob) p = static_cast<float>(static_cast<double>(p) / total);
  double top_p = static_cast<double>(params.top_p);
  if (!std::isfinite(top_p) || top_p <= 0.0 || top_p > 1.0) top_p = 1.0;
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
  std::uniform_real_distribution<double> d0(0.0, cumulative);
  const double r = d0(rng);
  double acc = 0.0;
  for (int k = 0; k < keep; ++k) {
    acc += prob[static_cast<size_t>(order[static_cast<size_t>(k)])];
    if (r < acc) return order[static_cast<size_t>(k)];
  }
  return order[static_cast<size_t>(keep - 1)];
}

// ---- fast-path equivalence -------------------------------------------------

struct Row {
  std::vector<float> logits;
  const char* name;
};

std::vector<Row> equivalence_rows() {
  std::vector<Row> rows;
  std::mt19937_64 gen(2024);
  std::normal_distribution<float> normal(0.0f, 2.5f);

  // A realistic tail: a few strong tokens over a long low-probability body.
  std::vector<float> tail(4096);
  for (auto& v : tail) v = normal(gen);
  tail[17] = 9.5f;
  tail[900] = 8.0f;
  rows.push_back({std::move(tail), "random"});

  // Exact ties everywhere: only the index tie-break can order them.
  rows.push_back({std::vector<float>(2048, 0.0f), "flat"});
  std::vector<float> tied(2048);
  for (size_t i = 0; i < tied.size(); ++i)
    tied[i] = static_cast<float>(static_cast<int>(i % 7) - 3);
  rows.push_back({std::move(tied), "tied"});

  // One dominant token, so most probabilities underflow to zero.
  std::vector<float> peaked(5000);
  for (auto& v : peaked) v = normal(gen) - 40.0f;
  peaked[321] = 25.0f;
  rows.push_back({std::move(peaked), "peaked"});

  // Degenerate values must not perturb the ordering either.
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  rows.push_back({{inf, 0.0f, -inf, nan, 2.0f, 2.0f}, "degenerate"});
  rows.push_back({{nan, nan, nan}, "all-nan"});
  rows.push_back({{-inf, -inf, -inf}, "all-neginf"});

  // Long rows exercise the floor search and the full-sort fallback.
  std::vector<float> wide(50000);
  for (auto& v : wide) v = normal(gen);
  rows.push_back({std::move(wide), "wide"});
  return rows;
}

void test_fast_matches_reference() {
  const float temperatures[] = {0.1f, 0.7f, 1.0f, 2.0f};
  const float top_ps[] = {0.1f, 0.5f, 0.9f, 0.99f, 1.0f};
  const std::vector<Row> rows = equivalence_rows();
  for (const Row& row : rows) {
    for (float temperature : temperatures) {
      for (float top_p : top_ps) {
        const SamplingParams params = sampling(temperature, top_p, 424242);
        std::mt19937_64 fast_rng(params.seed);
        std::mt19937_64 ref_rng(params.seed);
        const int n = static_cast<int>(row.logits.size());
        bool same = true;
        for (int i = 0; i < 24 && same; ++i) {
          const int a = sample_token(row.logits.data(), n, params, fast_rng);
          const int b = reference_sample(row.logits.data(), n, params, ref_rng);
          same = (a == b);
        }
        if (!same) {
          std::printf("FAIL: %s t=%.2f top_p=%.2f diverges from full sort\n",
                      row.name, static_cast<double>(temperature),
                      static_cast<double>(top_p));
          ++g_failures;
        }
      }
    }
  }
  check(true, "fast nucleus selector matches the full-sort reference");
}

// A 248320-row vocabulary is the production LM head. The fast path must stay
// far under the several milliseconds a full sort costs.
void test_large_vocab_latency() {
  constexpr int n = 248320;
  constexpr int kDraws = 30;
  const SamplingParams params = sampling(0.7f, 0.9f, 42);

  // Both rows are synthetic, so the numbers do not depend on a checkpoint: one
  // looks like a trained LM head (a few peaks over a steep body), the other is
  // a flat Gaussian whose 0.9 nucleus spans ~50k tokens.
  std::mt19937_64 gen(77);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<float> realistic(static_cast<size_t>(n));
  for (auto& v : realistic) v = normal(gen);
  realistic[1234] = 14.0f;
  realistic[55555] = 12.5f;
  realistic[90001] = 11.0f;
  std::vector<float> flat(static_cast<size_t>(n));
  std::normal_distribution<float> wide(0.0f, 1.5f);
  for (auto& v : flat) v = wide(gen);

  auto measure = [&](const std::vector<float>& logits, bool fast) {
    std::mt19937_64 rng(params.seed);
    int sink = 0;
    for (int i = 0; i < 3; ++i)  // warm the allocator and the caches
      sink += fast ? sample_token(logits.data(), n, params, rng)
                   : reference_sample(logits.data(), n, params, rng);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kDraws; ++i)
      sink += fast ? sample_token(logits.data(), n, params, rng)
                   : reference_sample(logits.data(), n, params, rng);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count() /
                      kDraws;
    check(sink >= 0, "every draw returns a valid token id");
    return ms;
  };

  const double lm_ref = measure(realistic, false);
  const double lm_fast = measure(realistic, true);
  const double flat_ref = measure(flat, false);
  const double flat_fast = measure(flat, true);
  std::printf("n=%d top_p=0.9 ms/token: lm-shaped %.3f -> %.3f, flat %.3f -> %.3f\n",
              n, lm_ref, lm_fast, flat_ref, flat_fast);
  check(lm_fast < 2.0, "fast nucleus selection stays under 2 ms/token");
  check(lm_fast < lm_ref, "fast nucleus selection beats the full sort");
  check(flat_fast < flat_ref, "the flat row also beats the full sort");
}

// ---- greedy and cache-free behaviour ---------------------------------------

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

void test_subnormal_temperature() {
  // 1e-40f is subnormal, so 1/temperature is finite in fp64 but overflows fp32.
  // The nucleus must still collapse onto the argmax instead of turning the
  // largest row into 0 * inf = NaN.
  const std::vector<float> logits{0.1f, 5.0f, -1.0f, 3.2f};
  std::mt19937_64 rng(7);
  for (float temperature : {1e-40f, 1e-38f, 1e-30f,
                            std::numeric_limits<float>::denorm_min()}) {
    for (float top_p : {1.0f, 0.9f, 0.5f}) {
      const int id =
          sample_token(logits.data(), 4, sampling(temperature, top_p, 5), rng);
      check(id == 1, "a subnormal temperature still picks the argmax");
    }
  }
  // The same row must agree with the full-sort reference.
  const SamplingParams params = sampling(1e-40f, 1.0f, 9);
  std::mt19937_64 fast_rng(params.seed);
  std::mt19937_64 ref_rng(params.seed);
  for (int i = 0; i < 16; ++i) {
    check(sample_token(logits.data(), 4, params, fast_rng) ==
              reference_sample(logits.data(), 4, params, ref_rng),
          "a subnormal temperature matches the full-sort reference");
  }
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
  test_subnormal_temperature();
  test_sampling_is_seed_deterministic();
  test_top_p_excludes_tail();
  test_extreme_logits();
  test_prefix_match();
  test_fast_matches_reference();
  test_large_vocab_latency();
  if (g_failures != 0) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("qwen38 host tests passed\n");
  return 0;
}
