// SPDX-License-Identifier: Apache-2.0
// Assert-based host checks for the Qwen3.8-27B sampling and prefix-reuse
// helpers. No GPU, no framework.
#include "qwen38/decode_helpers.hpp"
#include "qwen38/engine.hpp"
#include "qwen38/prefix_cache.hpp"
#include "qwen38/sampling.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace {

using umminfer::qwen38::ChatTurn;
using umminfer::qwen38::MRoPEPositions;
using umminfer::qwen38::PreprocessedImage;
using umminfer::qwen38::build_mrope_positions;
using umminfer::qwen38::build_suffix_mrope_positions;
using umminfer::qwen38::can_snapshot_multimodal_prefix;
using umminfer::qwen38::compute_media_hash;
using umminfer::qwen38::kImagePadToken;
using umminfer::qwen38::kVideoPadToken;
using umminfer::qwen38::match_multimodal_prefix;
using umminfer::qwen38::match_prefix;
using umminfer::qwen38::mrope_pos_next_at;
using umminfer::qwen38::render_chat_prompt;
using umminfer::qwen38::render_chat_prompt_parts;
using umminfer::qwen38::RenderedChatPrompt;
using umminfer::qwen38::resolve_snapshot_len;
using umminfer::qwen38::SamplingParams;
using umminfer::qwen38::sample_token;
using umminfer::qwen38::selected_token_logprob;

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

void test_behavior_logprobs() {
  const std::vector<float> row{static_cast<float>(std::log(2.0)), 0.0f, 0.0f};
  const auto reference = [&row](int id) {
    const long double maxv = *std::max_element(row.begin(), row.end());
    long double total = 0.0;
    for (const float value : row) total += std::exp(static_cast<long double>(value) - maxv);
    return static_cast<double>(static_cast<long double>(row[id]) - maxv - std::log(total));
  };
  const double expected[] = {reference(0), reference(1), reference(2)};
  for (int id = 0; id < 3; ++id) {
    double value = 1.0;
    check(selected_token_logprob(row.data(), 3, id, &value),
          "behavior logprob accepts a finite row");
    check(std::abs(value - expected[id]) < 1e-15,
          "behavior logprob matches an independent known probability");
    check(std::isfinite(value) && value <= 0.0,
          "behavior logprob is finite and non-positive");
  }

  const std::vector<float> extreme{10000.0f, 9999.0f, -10000.0f};
  double dominant = 1.0;
  check(selected_token_logprob(extreme.data(), 3, 0, &dominant) &&
            std::abs(dominant + std::log1p(std::exp(-1.0))) < 1e-15,
        "behavior logprob is stable across extreme finite logits");

  const SamplingParams params = sampling(0.7f, 0.9f, 42);
  std::mt19937_64 first_rng(params.seed);
  std::mt19937_64 second_rng(params.seed);
  for (int i = 0; i < 128; ++i) {
    const int first = sample_token(row.data(), 3, params, first_rng);
    const int second = sample_token(row.data(), 3, params, second_rng);
    double value = 1.0;
    check(first == second && selected_token_logprob(row.data(), 3, first, &value) &&
              std::abs(value - expected[first]) < 1e-15,
          "evidence preserves the sampled id and consumes no additional RNG");
  }

  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<float> bad_rows[] = {
      {inf, 0.0f}, {nan, 0.0f}, {-inf, -inf}};
  for (const auto& bad : bad_rows) {
    double value = 1.0;
    check(!selected_token_logprob(bad.data(), 2, 0, &value) && std::isnan(value),
          "degenerate logits fail evidence closed");
  }
  double value = 1.0;
  check(!selected_token_logprob(row.data(), 3, 3, &value) &&
            !selected_token_logprob(nullptr, 3, 0, &value),
        "invalid row and token bounds fail evidence closed");
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

// ---- turn-boundary prompt split --------------------------------------------

// The split renderer must be a pure refactor of the whole-prompt renderer.
void test_render_parts_matches_full() {
  const std::vector<std::vector<ChatTurn>> cases = {
      {{"user", "hello", false}},
      {{"system", "be brief", false}, {"user", "  hi there \n", false}},
      {{"user", "q1", false}, {"assistant", "a1", false}, {"user", "q2", false}},
      {{"user", "", false}},
      {{"user", "with image", true}},
      {{"", "empty role", false}},
  };
  for (const auto& turns : cases) {
    for (bool thinking : {false, true}) {
      const RenderedChatPrompt parts = render_chat_prompt_parts(turns, thinking);
      check(parts.full() == render_chat_prompt(turns, thinking),
            "parts.full() equals render_chat_prompt byte for byte");
      check(!parts.prefix.empty() && !parts.suffix.empty(),
            "both prompt parts are non-empty");
      check(parts.prefix.size() + parts.suffix.size() == parts.full().size(),
            "the parts concatenate without a gap");
    }
  }
  bool threw = false;
  try {
    render_chat_prompt_parts({}, false);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  check(threw, "an empty turn list still throws");
}

// Turn 1's cacheable prefix must survive every assistant reply and every
// later turn, because the boundary sits on a special token.
void test_turn_boundary_prefix_is_stable() {
  const std::vector<std::string> replies = {
      "",
      "ok",
      "Sure, here is the answer.",
      "\xE4\xB8\xAD\xE6\x96\x87\xE5\x9B\x9E\xE5\xA4\x8D\xEF\xBC\x9A\xE4\xBD\xA0\xE5\xA5\xBD",
      "\n\nleading newlines",
      "```cpp\nint main() { return 0; }\n```",
      "line one\nline two\n",
      "  trailing spaces  ",
  };
  for (bool thinking : {false, true}) {
    const std::vector<ChatTurn> turn1 = {{"user", "u1", false}};
    const RenderedChatPrompt p1 = render_chat_prompt_parts(turn1, thinking);
    for (const std::string& reply : replies) {
      const std::vector<ChatTurn> turn2 = {{"user", "u1", false},
                                           {"assistant", reply, false},
                                           {"user", "u2", false}};
      const std::vector<ChatTurn> turn3 = {{"user", "u1", false},
                                           {"assistant", reply, false},
                                           {"user", "u2", false},
                                           {"assistant", "a2", false},
                                           {"user", "u3", false}};
      const RenderedChatPrompt p2 = render_chat_prompt_parts(turn2, thinking);
      const RenderedChatPrompt p3 = render_chat_prompt_parts(turn3, thinking);
      check(p2.prefix.compare(0, p1.prefix.size(), p1.prefix) == 0,
            "turn 1 prefix prefixes turn 2 prefix");
      check(p2.full().compare(0, p1.prefix.size(), p1.prefix) == 0,
            "turn 1 prefix prefixes turn 2 full prompt");
      check(p3.prefix.compare(0, p2.prefix.size(), p2.prefix) == 0,
            "each turn prefix extends the previous one");
      check(p1.suffix == p2.suffix && p2.suffix == p3.suffix,
            "the assistant header suffix never changes");
    }
  }
}

void test_resolve_snapshot_len() {
  check(resolve_snapshot_len(100, -1) == 100, "-1 snapshots the whole prompt");
  check(resolve_snapshot_len(100, 0) == 100, "0 snapshots the whole prompt");
  check(resolve_snapshot_len(100, 100) == 100,
        "a cut at the prompt length snapshots the whole prompt");
  check(resolve_snapshot_len(100, 250) == 100,
        "a cut past the prompt length is clamped to the prompt length");
  check(resolve_snapshot_len(100, 7) == 7, "an interior cut is kept");
  check(resolve_snapshot_len(1, 0) == 1, "a one-token prompt stays whole");
}

// The bug this wave fixes: a snapshot at the full prompt never matches the next
// turn, while a snapshot at the turn boundary always does.
void test_prefix_boundary_reuses_next_turn() {
  // ids[:cut] is the turn-boundary prefix; the two suffixes differ.
  const std::vector<int> full_turn1{1, 2, 3, 4, 5, 6, 7, 8};
  const std::vector<int> turn1_prefix{1, 2, 3, 4, 5};
  const std::vector<int> turn2{1, 2, 3, 4, 5, 90, 91, 20, 21, 22, 6, 7, 8};
  const int cut = static_cast<int>(turn1_prefix.size());
  check(match_prefix(turn1_prefix, turn2, false, true) == cut,
        "a turn-boundary snapshot hits the next turn");
  check(match_prefix(full_turn1, turn2, false, true) == 0,
        "a whole-prompt snapshot misses the next turn");
  // The split is only safe when it is an interior cut with rows on both sides.
  check(resolve_snapshot_len(static_cast<int>(turn2.size()), cut) == cut,
        "the boundary cut is a valid snapshot length");
}

void test_video_preprocess_and_prompt() {
  using umminfer::qwen38::PreprocessedImage;
  using umminfer::qwen38::expand_video_pads;
  using umminfer::qwen38::preprocess_video_frames;
  // 5 solid-color 64x64 frames -> padded to 6 even frames -> grid_t = 3.
  std::vector<std::vector<uint8_t>> frames(5, std::vector<uint8_t>(64 * 64 * 3, 0));
  for (size_t f = 0; f < frames.size(); ++f) {
    for (size_t i = 0; i < frames[f].size(); i += 3) {
      frames[f][i + 0] = static_cast<uint8_t>(f * 50);
      frames[f][i + 1] = 128;
      frames[f][i + 2] = 255;
    }
  }
  const PreprocessedImage vid =
      preprocess_video_frames(frames, 64, 64, 4096, 16384, 3584);
  check(vid.grid_t == 3, "5 frames pack into grid_t == 3 temporal slices");
  check(vid.grid_h >= 2 && vid.grid_w >= 2 && (vid.grid_h % 2 == 0) &&
            (vid.grid_w % 2 == 0),
        "video grid dimensions are positive multiples of spatial merge size");
  check(vid.rows() == vid.grid_t * vid.grid_h * vid.grid_w &&
            vid.rows() <= 3584,
        "video patch rows match grid_t*grid_h*grid_w and fit workspace cap");
  check(vid.pixel_values.size() ==
            static_cast<size_t>(vid.rows()) * PreprocessedImage::kPatchDim,
        "video pixel_values size matches rows * kPatchDim");

  ChatTurn vturn;
  vturn.role = "user";
  vturn.text = "Describe this clip.";
  vturn.video_slices = 3;
  vturn.video_timestamps = {0.0, 1.0, 2.0};
  const RenderedChatPrompt rendered = render_chat_prompt_parts({vturn}, false);
  check(rendered.prefix.find("<0.0 seconds><|vision_start|><|video_pad|><|vision_end|>") !=
            std::string::npos,
        "video prompt emits first timestamp + video_pad marker");
  check(rendered.prefix.find("<2.0 seconds><|vision_start|><|video_pad|><|vision_end|>") !=
            std::string::npos,
        "video prompt emits third timestamp + video_pad marker");

  // Expanding 3 video_pad placeholders by 4 merged tokens each yields 12 video_pads.
  const std::vector<int> ids{10, 248057, 11, 248057, 12, 248057, 13};
  const std::vector<int> expanded = expand_video_pads(ids, 4, 3);
  check(expanded.size() == 4 + 12, "expand_video_pads expands every slice");
  int count = 0;
  for (int id : expanded)
    if (id == 248057) ++count;
  check(count == 12, "expand_video_pads emits exact total video_pad count");

  // Exact-size frames (src_h == dh && src_w == dw) skip bicubic_resize and
  // produce identical normalized patch values to the resized path.
  constexpr int kExactH = 64;
  constexpr int kExactW = 96;
  int dh = 0, dw = 0;
  umminfer::qwen38::smart_resize(kExactH, kExactW, 32, 4096, 16384, &dh, &dw);
  check(dh == kExactH && dw == kExactW,
        "smart_resize preserves exact-size multiple-of-32 frame dimensions");
  std::vector<std::vector<uint8_t>> exact_frames(
      3, std::vector<uint8_t>(static_cast<size_t>(kExactH) * kExactW * 3, 0));
  for (size_t f = 0; f < exact_frames.size(); ++f) {
    for (int y = 0; y < kExactH; ++y) {
      for (int x = 0; x < kExactW; ++x) {
        const size_t idx = (static_cast<size_t>(y) * kExactW + x) * 3;
        exact_frames[f][idx + 0] = static_cast<uint8_t>((y * 3 + x + f * 37) % 256);
        exact_frames[f][idx + 1] = static_cast<uint8_t>((y + x * 5 + f * 19) % 256);
        exact_frames[f][idx + 2] = static_cast<uint8_t>((y * 7 + x * 2 + f * 11) % 256);
      }
    }
  }
  const PreprocessedImage exact =
      preprocess_video_frames(exact_frames, kExactH, kExactW, 4096, 16384, 3584);
  check(exact.resized_h == kExactH && exact.resized_w == kExactW,
        "exact-size video frames keep target dimensions");
  check(exact.grid_t == 2 && exact.grid_h == 4 && exact.grid_w == 6,
        "exact-size video grid matches (2, 4, 6)");
  std::vector<std::vector<float>> ref_resized;
  ref_resized.reserve(4);
  for (int i = 0; i < 4; ++i) {
    const auto& raw = exact_frames[std::min<size_t>(static_cast<size_t>(i),
                                                    exact_frames.size() - 1)];
    ref_resized.push_back(
        umminfer::qwen38::resize_image_rgb(raw.data(), kExactH, kExactW, dh, dw));
  }
  bool exact_matches_bicubic = true;
  bool exact_matches_direct = true;
  const int gh2 = exact.grid_h / 2;
  const int gw2 = exact.grid_w / 2;
  for (int t = 0; t < exact.grid_t && exact_matches_bicubic && exact_matches_direct; ++t) {
    const auto& raw0 = exact_frames[std::min<size_t>(static_cast<size_t>(t * 2),
                                                     exact_frames.size() - 1)];
    const auto& raw1 = exact_frames[std::min<size_t>(static_cast<size_t>(t * 2 + 1),
                                                     exact_frames.size() - 1)];
    const auto& bic0 = ref_resized[static_cast<size_t>(t * 2)];
    const auto& bic1 = ref_resized[static_cast<size_t>(t * 2 + 1)];
    for (int gh = 0; gh < gh2; ++gh) {
      for (int gw = 0; gw < gw2; ++gw) {
        for (int msh = 0; msh < 2; ++msh) {
          for (int msw = 0; msw < 2; ++msw) {
            const int row = (((t * gh2 + gh) * gw2 + gw) * 2 + msh) * 2 + msw;
            for (int c = 0; c < 3; ++c) {
              for (int ts = 0; ts < 2; ++ts) {
                const auto& raw = (ts == 0) ? raw0 : raw1;
                const auto& bic = (ts == 0) ? bic0 : bic1;
                for (int py = 0; py < 16; ++py) {
                  for (int px = 0; px < 16; ++px) {
                    const int y = gh * 32 + msh * 16 + py;
                    const int x = gw * 32 + msw * 16 + px;
                    const size_t pidx = (static_cast<size_t>(y) * kExactW + x) * 3 + c;
                    const int fidx = (((c * 2 + ts) * 16 + py) * 16 + px);
                    const float got =
                        exact.pixel_values[static_cast<size_t>(row) *
                                               PreprocessedImage::kPatchDim +
                                           fidx];
                    const float want_direct = static_cast<float>(raw[pidx]) / 127.5f - 1.0f;
                    const float want_bicubic = bic[pidx] / 127.5f - 1.0f;
                    if (got != want_direct) exact_matches_direct = false;
                    if (got != want_bicubic) exact_matches_bicubic = false;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  check(exact_matches_direct,
        "exact-size video path converts RGB24 directly to [-1, 1] patches");
  check(exact_matches_bicubic,
        "exact-size video path matches bicubic_resize output bit for bit");
}

// ---- cross-turn multimodal prefix cache & MRoPE continuity -----------------

PreprocessedImage make_synthetic_media(int grid_t, int grid_h, int grid_w,
                                       bool is_video, float seed_val) {
  PreprocessedImage img;
  img.grid_t = grid_t;
  img.grid_h = grid_h;
  img.grid_w = grid_w;
  img.resized_h = grid_h * 16;
  img.resized_w = grid_w * 16;
  img.is_video = is_video;
  const size_t n =
      static_cast<size_t>(img.rows()) * PreprocessedImage::kPatchDim;
  img.pixel_values.resize(n);
  for (size_t i = 0; i < n; ++i) {
    img.pixel_values[i] =
        seed_val + static_cast<float>((i * 131u + 17u) % 256u) / 127.5f - 1.0f;
  }
  return img;
}

void test_compute_media_hash() {
  check(compute_media_hash(nullptr) == 0, "nullptr media hashes to 0");

  const PreprocessedImage a = make_synthetic_media(1, 4, 6, false, 0.0f);
  const PreprocessedImage a_copy = make_synthetic_media(1, 4, 6, false, 0.0f);
  const uint64_t ha = compute_media_hash(&a);
  check(ha != 0, "non-null media hash is never 0");
  check(ha == compute_media_hash(&a_copy),
        "compute_media_hash is deterministic across identical copies");

  // Sensitivity to grid_t, grid_h, grid_w, is_video.
  PreprocessedImage diff_t = a;
  diff_t.grid_t = 2;
  check(compute_media_hash(&diff_t) != ha,
        "compute_media_hash changes when grid_t changes");

  PreprocessedImage diff_h = a;
  diff_h.grid_h = 6;
  diff_h.grid_w = 4;  // same row count (24), transposed spatial grid
  check(compute_media_hash(&diff_h) != ha,
        "compute_media_hash changes when grid_h/grid_w change");

  PreprocessedImage diff_w = a;
  diff_w.grid_w = 8;
  check(compute_media_hash(&diff_w) != ha,
        "compute_media_hash changes when grid_w changes");

  PreprocessedImage diff_video = a;
  diff_video.is_video = true;
  check(compute_media_hash(&diff_video) != ha,
        "compute_media_hash changes when is_video toggles");

  // Sensitivity to single-pixel edits (first, middle, last float).
  for (size_t idx : {size_t{0}, a.pixel_values.size() / 2,
                     a.pixel_values.size() - 1}) {
    PreprocessedImage edited = a;
    edited.pixel_values[idx] += 1e-4f;
    check(compute_media_hash(&edited) != ha,
          "compute_media_hash changes when a single pixel value is edited");
  }

  // Real preprocessor sets is_video=false for RGB image and true for video.
  std::vector<uint8_t> rgb(64 * 64 * 3, 120);
  const PreprocessedImage real_img =
      umminfer::qwen38::preprocess_rgb(rgb.data(), 64, 64, 4096, 16384);
  check(!real_img.is_video, "preprocess_rgb sets is_video=false");
  const PreprocessedImage real_vid = umminfer::qwen38::preprocess_video_frames(
      {rgb, rgb}, 64, 64, 4096, 16384, 3584);
  check(real_vid.is_video, "preprocess_video_frames sets is_video=true");
  check(compute_media_hash(&real_img) != compute_media_hash(&real_vid),
        "still image and 2-frame identical video hash distinctly");
}

void test_can_snapshot_multimodal_prefix() {
  // Pure text accepts any snapshot_len in (0, n].
  const std::vector<int> text_ids{10, 11, 12, 13, 14};
  check(!can_snapshot_multimodal_prefix(text_ids, 0, false),
        "text snapshot rejects len=0");
  check(!can_snapshot_multimodal_prefix(text_ids, -1, false),
        "text snapshot rejects negative len");
  check(!can_snapshot_multimodal_prefix(text_ids, 6, false),
        "text snapshot rejects len > size");
  check(can_snapshot_multimodal_prefix(text_ids, 3, false),
        "text snapshot accepts interior boundary");
  check(can_snapshot_multimodal_prefix(text_ids, 5, false),
        "text snapshot accepts full prompt");

  // Single image: pads at indices [2, 6).
  const std::vector<int> img_ids{
      1, 2, kImagePadToken, kImagePadToken, kImagePadToken, kImagePadToken,
      3, 4, 5, 6};
  check(!can_snapshot_multimodal_prefix(img_ids, 1, true),
        "image snapshot rejects boundary before image pads");
  check(!can_snapshot_multimodal_prefix(img_ids, 2, true),
        "image snapshot rejects boundary right at start of image pads");
  check(!can_snapshot_multimodal_prefix(img_ids, 4, true),
        "image snapshot rejects boundary cutting inside image pads");
  check(!can_snapshot_multimodal_prefix(img_ids, 5, true),
        "image snapshot rejects boundary leaving one image pad outside");
  check(can_snapshot_multimodal_prefix(img_ids, 6, true),
        "image snapshot accepts boundary right after all image pads");
  check(can_snapshot_multimodal_prefix(img_ids, 8, true),
        "image snapshot accepts turn boundary after all image pads");
  check(can_snapshot_multimodal_prefix(img_ids, 10, true),
        "image snapshot accepts whole prompt");

  // Multi-slice video: pads at [2, 4) and [6, 8).
  const std::vector<int> vid_ids{
      1, 2, kVideoPadToken, kVideoPadToken, 100, 101,
      kVideoPadToken, kVideoPadToken, 3, 4, 5};
  check(!can_snapshot_multimodal_prefix(vid_ids, 4, true),
        "video snapshot rejects boundary between slice 0 and slice 1");
  check(!can_snapshot_multimodal_prefix(vid_ids, 6, true),
        "video snapshot rejects boundary before slice 1");
  check(!can_snapshot_multimodal_prefix(vid_ids, 7, true),
        "video snapshot rejects boundary cutting inside slice 1");
  check(can_snapshot_multimodal_prefix(vid_ids, 8, true),
        "video snapshot accepts boundary right after final video slice");
  check(can_snapshot_multimodal_prefix(vid_ids, 9, true),
        "video snapshot accepts turn boundary strictly after all video slices");

  // Malformed multimodal prompt with zero vision pads is rejected.
  check(!can_snapshot_multimodal_prefix(text_ids, 5, true),
        "multimodal snapshot rejects prompt without any vision pad");
}

void test_match_multimodal_prefix() {
  const PreprocessedImage img_a = make_synthetic_media(1, 4, 4, false, 0.0f);
  const PreprocessedImage img_b = make_synthetic_media(1, 4, 4, false, 0.25f);
  const PreprocessedImage vid_a = make_synthetic_media(2, 4, 4, true, 0.1f);
  const uint64_t hash_a = compute_media_hash(&img_a);
  const uint64_t hash_b = compute_media_hash(&img_b);
  const uint64_t hash_v = compute_media_hash(&vid_a);

  // Pure-text hit and miss.
  const std::vector<int> txt_cached{10, 20, 30};
  check(match_multimodal_prefix(txt_cached, 0, {10, 20, 30, 40}, 0, true) == 3,
        "text-only prompt hits text-only cache");
  check(match_multimodal_prefix(txt_cached, 0, {10, 20, 99, 40}, 0, true) == 0,
        "text-only mid-prefix mismatch misses");
  check(match_multimodal_prefix(txt_cached, 0, {10, 20, 30}, 0, true) == 0,
        "equal-length text prompt misses");
  check(match_multimodal_prefix(txt_cached, 0, {10, 20}, 0, true) == 0,
        "shorter text prompt misses");
  check(match_multimodal_prefix(txt_cached, 0, {10, 20, 30, 40}, 0, false) == 0,
        "disabled cache misses for text");

  // Same-image hit on follow-up turn.
  const std::vector<int> img_turn1_prefix{
      1, 2, kImagePadToken, kImagePadToken, kImagePadToken, kImagePadToken, 3, 4};
  const std::vector<int> img_turn2{
      1, 2, kImagePadToken, kImagePadToken, kImagePadToken, kImagePadToken,
      3, 4, 50, 51, 52, 53};
  check(match_multimodal_prefix(img_turn1_prefix, hash_a, img_turn2, hash_a,
                                true) ==
            static_cast<int>(img_turn1_prefix.size()),
        "same-image follow-up turn hits multimodal prefix cache");

  // Same-video hit on follow-up turn.
  const std::vector<int> vid_turn1_prefix{
      1, 2, kVideoPadToken, kVideoPadToken, 9, kVideoPadToken, kVideoPadToken,
      3, 4};
  const std::vector<int> vid_turn2{
      1, 2, kVideoPadToken, kVideoPadToken, 9, kVideoPadToken, kVideoPadToken,
      3, 4, 60, 61, 62};
  check(match_multimodal_prefix(vid_turn1_prefix, hash_v, vid_turn2, hash_v,
                                true) ==
            static_cast<int>(vid_turn1_prefix.size()),
        "same-video follow-up turn hits multimodal prefix cache");

  // Different-image miss.
  check(match_multimodal_prefix(img_turn1_prefix, hash_a, img_turn2, hash_b,
                                true) == 0,
        "different-image follow-up misses");

  // Image-to-text and text-to-image misses.
  check(match_multimodal_prefix(img_turn1_prefix, hash_a, img_turn2, 0, true) ==
            0,
        "image-to-text request misses image snapshot");
  check(match_multimodal_prefix(txt_cached, 0, {10, 20, 30, 40}, hash_a,
                                true) == 0,
        "text-to-image request misses text snapshot");

  // Disabled cache and non-extending prompt misses.
  check(match_multimodal_prefix(img_turn1_prefix, hash_a, img_turn2, hash_a,
                                false) == 0,
        "disabled cache misses for multimodal");
  check(match_multimodal_prefix(img_turn1_prefix, hash_a, img_turn1_prefix,
                                hash_a, true) == 0,
        "non-extending equal-length multimodal prompt misses");
  const std::vector<int> img_shorter(img_turn1_prefix.begin(),
                                     img_turn1_prefix.end() - 1);
  check(match_multimodal_prefix(img_turn1_prefix, hash_a, img_shorter, hash_a,
                                true) == 0,
        "shorter multimodal prompt misses");

  // Follow-up prompt introducing new vision pads in the suffix must miss.
  std::vector<int> img_turn2_extra_pad = img_turn2;
  img_turn2_extra_pad.push_back(kImagePadToken);
  check(match_multimodal_prefix(img_turn1_prefix, hash_a, img_turn2_extra_pad,
                                hash_a, true) == 0,
        "follow-up with new vision pad in suffix misses");
}

void test_mrope_suffix_continuity() {
  // 1) Still image: grid_t=1, grid_h=6, grid_w=8 -> gh=3, gw=4, slice=12,
  //    compressed MRoPE advance = max(3, 4) = 4 (saves 8 positions vs 12 tokens).
  const int img_gt = 1, img_gh = 6, img_gw = 8;
  const int img_slice = (img_gh / 2) * (img_gw / 2);  // 12
  std::vector<int> img_t1_prefix{101, 102, 103};
  img_t1_prefix.insert(img_t1_prefix.end(), static_cast<size_t>(img_slice),
                       kImagePadToken);
  img_t1_prefix.insert(img_t1_prefix.end(), {201, 202, 203, 204});
  std::vector<int> img_t1_full = img_t1_prefix;
  img_t1_full.insert(img_t1_full.end(), {301, 302, 303, 304, 305, 306});

  const int img_snap1 = static_cast<int>(img_t1_prefix.size());
  check(can_snapshot_multimodal_prefix(img_t1_full, img_snap1, true),
        "image turn 1 snapshot boundary is valid");
  const MRoPEPositions pos_t1 =
      build_mrope_positions(img_t1_full, img_gt, img_gh, img_gw, true);
  const int cached_pos_next1 =
      mrope_pos_next_at(pos_t1, static_cast<int>(img_t1_full.size()), img_snap1);
  // 3 leading text + max(3,4)=4 image + 4 trailing text = 11 (< img_snap1=19).
  check(cached_pos_next1 == 11 && cached_pos_next1 < img_snap1,
        "image cached_pos_next reflects 3D MRoPE compression");

  // Turn 2 extends Turn 1 prefix with assistant reply + follow-up user turn.
  std::vector<int> img_t2_prefix = img_t1_prefix;
  img_t2_prefix.insert(img_t2_prefix.end(), {401, 402, 403, 404, 405, 406, 407});
  std::vector<int> img_t2_full = img_t2_prefix;
  img_t2_full.insert(img_t2_full.end(), {301, 302, 303, 304, 305, 306});
  const MRoPEPositions full_t2 =
      build_mrope_positions(img_t2_full, img_gt, img_gh, img_gw, true);
  const MRoPEPositions reused_t2 = build_suffix_mrope_positions(
      static_cast<int>(img_t2_full.size()), img_snap1, cached_pos_next1);
  bool img_t2_exact = (reused_t2.next == full_t2.next);
  for (int i = img_snap1; i < static_cast<int>(img_t2_full.size()); ++i) {
    const size_t k = static_cast<size_t>(i);
    if (reused_t2.t[k] != full_t2.t[k] || reused_t2.h[k] != full_t2.h[k] ||
        reused_t2.w[k] != full_t2.w[k]) {
      img_t2_exact = false;
      break;
    }
  }
  check(img_t2_exact,
        "image turn 2 suffix MRoPE positions match full-prompt MRoPE exactly");

  // Turn 3 reuses Turn 2's updated snapshot boundary.
  const int img_snap2 = static_cast<int>(img_t2_prefix.size());
  const int cached_pos_next2 = mrope_pos_next_at(
      reused_t2, static_cast<int>(img_t2_full.size()), img_snap2);
  std::vector<int> img_t3_full = img_t2_prefix;
  img_t3_full.insert(img_t3_full.end(), {501, 502, 503, 301, 302, 303});
  const MRoPEPositions full_t3 =
      build_mrope_positions(img_t3_full, img_gt, img_gh, img_gw, true);
  const MRoPEPositions reused_t3 = build_suffix_mrope_positions(
      static_cast<int>(img_t3_full.size()), img_snap2, cached_pos_next2);
  bool img_t3_exact = (reused_t3.next == full_t3.next);
  for (int i = img_snap2; i < static_cast<int>(img_t3_full.size()); ++i) {
    const size_t k = static_cast<size_t>(i);
    if (reused_t3.t[k] != full_t3.t[k] || reused_t3.h[k] != full_t3.h[k] ||
        reused_t3.w[k] != full_t3.w[k]) {
      img_t3_exact = false;
      break;
    }
  }
  check(img_t3_exact,
        "image turn 3 chained suffix MRoPE matches full-prompt MRoPE exactly");

  // 2) Multi-slice video: grid_t=3, grid_h=4, grid_w=6 -> gh=2, gw=3, slice=6,
  //    3 slices separated by timestamp tokens.
  const int vid_gt = 3, vid_gh = 4, vid_gw = 6;
  const int vid_slice = (vid_gh / 2) * (vid_gw / 2);  // 6
  std::vector<int> vid_t1_prefix{10, 11};
  for (int s = 0; s < vid_gt; ++s) {
    vid_t1_prefix.push_back(700 + s);  // timestamp token
    vid_t1_prefix.push_back(710);      // vision_start
    vid_t1_prefix.insert(vid_t1_prefix.end(), static_cast<size_t>(vid_slice),
                         kVideoPadToken);
    vid_t1_prefix.push_back(711);      // vision_end
  }
  vid_t1_prefix.insert(vid_t1_prefix.end(), {800, 801, 802});
  std::vector<int> vid_t1_full = vid_t1_prefix;
  vid_t1_full.insert(vid_t1_full.end(), {900, 901, 902, 903});

  const int vid_snap1 = static_cast<int>(vid_t1_prefix.size());
  const MRoPEPositions vid_pos1 =
      build_mrope_positions(vid_t1_full, vid_gt, vid_gh, vid_gw, true);
  const int vid_cached_pos1 = mrope_pos_next_at(
      vid_pos1, static_cast<int>(vid_t1_full.size()), vid_snap1);
  // 2 lead + 3*(1 ts + 1 start + max(2,3)=3 + 1 end) + 3 trail = 2 + 18 + 3 = 23
  check(vid_cached_pos1 == 23 && vid_cached_pos1 < vid_snap1,
        "video cached_pos_next reflects multi-slice 3D MRoPE compression");

  std::vector<int> vid_t2_full = vid_t1_prefix;
  vid_t2_full.insert(vid_t2_full.end(), {810, 811, 812, 813, 900, 901, 902, 903});
  const MRoPEPositions vid_full2 =
      build_mrope_positions(vid_t2_full, vid_gt, vid_gh, vid_gw, true);
  const MRoPEPositions vid_reused2 = build_suffix_mrope_positions(
      static_cast<int>(vid_t2_full.size()), vid_snap1, vid_cached_pos1);
  bool vid_t2_exact = (vid_reused2.next == vid_full2.next);
  for (int i = vid_snap1; i < static_cast<int>(vid_t2_full.size()); ++i) {
    const size_t k = static_cast<size_t>(i);
    if (vid_reused2.t[k] != vid_full2.t[k] ||
        vid_reused2.h[k] != vid_full2.h[k] ||
        vid_reused2.w[k] != vid_full2.w[k]) {
      vid_t2_exact = false;
      break;
    }
  }
  check(vid_t2_exact,
        "video turn 2 suffix MRoPE positions match full-prompt MRoPE exactly");
}

void test_plan_vision_slice_chunks() {
  using umminfer::qwen38::VisionSliceChunk;
  using umminfer::qwen38::plan_vision_slice_chunks;

  auto verify_coverage = [](const std::vector<VisionSliceChunk>& chunks,
                            int grid_t, int slice_tokens, int token_cap) {
    int expected_start = 0;
    int total_slices = 0;
    int total_tokens = 0;
    for (const VisionSliceChunk& c : chunks) {
      check(c.start_slice == expected_start,
            "vision chunk start_slice is contiguous");
      check(c.num_slices > 0, "vision chunk has positive slice count");
      check(c.tokens == c.num_slices * slice_tokens,
            "vision chunk tokens equals num_slices * slice_tokens");
      check(c.tokens <= token_cap,
            "vision chunk tokens stays within token_cap");
      expected_start += c.num_slices;
      total_slices += c.num_slices;
      total_tokens += c.tokens;
    }
    check(total_slices == grid_t,
          "vision chunks cover every temporal slice exactly once");
    check(total_tokens == grid_t * slice_tokens,
          "vision chunks cover exact total token count");
  };

  // 1) Still image (grid_t == 1): 1 chunk of 1 slice.
  {
    const auto chunks = plan_vision_slice_chunks(1, 864, 4096);
    check(chunks.size() == 1 && chunks[0].start_slice == 0 &&
              chunks[0].num_slices == 1 && chunks[0].tokens == 864,
          "still image produces a single 1-slice chunk");
    verify_coverage(chunks, 1, 864, 4096);
  }

  // 2) 4-slice video fitting inside a single kVisTokenCap=4096 chunk.
  {
    const auto chunks = plan_vision_slice_chunks(4, 468, 4096);
    check(chunks.size() == 1 && chunks[0].start_slice == 0 &&
              chunks[0].num_slices == 4 && chunks[0].tokens == 1872,
          "4-slice 468-token video batches into a single chunk");
    verify_coverage(chunks, 4, 468, 4096);
  }

  // 3) Exact capacity boundary (4 * 1024 == 4096): still 1 chunk.
  {
    const auto chunks = plan_vision_slice_chunks(4, 1024, 4096);
    check(chunks.size() == 1 && chunks[0].num_slices == 4 &&
              chunks[0].tokens == 4096,
          "exact token_cap multiple fits in a single chunk");
    verify_coverage(chunks, 4, 1024, 4096);
  }

  // 4) Large video splitting cleanly across multiple chunks with tail remainder.
  {
    const auto chunks = plan_vision_slice_chunks(5, 1024, 4096);
    check(chunks.size() == 2 && chunks[0].start_slice == 0 &&
              chunks[0].num_slices == 4 && chunks[0].tokens == 4096 &&
              chunks[1].start_slice == 4 && chunks[1].num_slices == 1 &&
              chunks[1].tokens == 1024,
          "5x1024 video splits into [4, 1] chunks under cap 4096");
    verify_coverage(chunks, 5, 1024, 4096);
  }
  {
    const auto chunks = plan_vision_slice_chunks(7, 1500, 4096);
    check(chunks.size() == 4 && chunks[0].num_slices == 2 &&
              chunks[1].num_slices == 2 && chunks[2].num_slices == 2 &&
              chunks[3].num_slices == 1 && chunks[3].start_slice == 6,
          "7x1500 video splits into [2, 2, 2, 1] chunks under cap 4096");
    verify_coverage(chunks, 7, 1500, 4096);
  }
  {
    const auto chunks = plan_vision_slice_chunks(3, 4096, 4096);
    check(chunks.size() == 3 && chunks[0].num_slices == 1 &&
              chunks[1].num_slices == 1 && chunks[2].num_slices == 1,
          "full-cap slices fall back cleanly to 1 slice per chunk");
    verify_coverage(chunks, 3, 4096, 4096);
  }

  // 5) Invalid input guards.
  auto expect_invalid = [](int grid_t, int slice_tokens, int token_cap,
                           const char* label) {
    bool threw = false;
    try {
      (void)plan_vision_slice_chunks(grid_t, slice_tokens, token_cap);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    check(threw, label);
  };
  expect_invalid(0, 468, 4096, "plan_vision_slice_chunks rejects grid_t == 0");
  expect_invalid(-2, 468, 4096, "plan_vision_slice_chunks rejects grid_t < 0");
  expect_invalid(4, 0, 4096,
                 "plan_vision_slice_chunks rejects slice_tokens == 0");
  expect_invalid(4, -16, 4096,
                 "plan_vision_slice_chunks rejects slice_tokens < 0");
  expect_invalid(4, 468, 0, "plan_vision_slice_chunks rejects token_cap == 0");
  expect_invalid(4, 468, -1, "plan_vision_slice_chunks rejects token_cap < 0");
  expect_invalid(2, 4097, 4096,
                 "plan_vision_slice_chunks rejects slice_tokens > token_cap");
}

void test_decode_helpers() {
  using umminfer::qwen38::Bf16;
  using umminfer::qwen38::GemmTileCoord;
  using umminfer::qwen38::MultiGemvRowLoc;
  using umminfer::qwen38::fill_text_rope_row;
  using umminfer::qwen38::gemv_unroll_for;
  using umminfer::qwen38::init_text_rope_inv_freq;
  using umminfer::qwen38::kTextRopeTheta;
  using umminfer::qwen38::kTextRotaryDim;
  using umminfer::qwen38::kTextRotaryHalf;
  using umminfer::qwen38::multi_gemv_locate_row;
  using umminfer::qwen38::swiglu_gemv_unroll_for;
  using umminfer::qwen38::swizzle_gemm_cta;

  // 1) Unroll schedules match the latency-bound vs issue-bound split.
  check(gemv_unroll_for(1) == 4 && gemv_unroll_for(2) == 4 &&
            gemv_unroll_for(3) == 1 && gemv_unroll_for(8) == 1,
        "gemv_unroll_for uses 4x unroll for m<=2 and 1x for m>=3");
  check(swiglu_gemv_unroll_for(1) == 2 && swiglu_gemv_unroll_for(2) == 2 &&
            swiglu_gemv_unroll_for(3) == 1 && swiglu_gemv_unroll_for(8) == 1,
        "swiglu_gemv_unroll_for uses 2x dual-weight unroll for m<=2 and 1x for m>=3");

  // 2) swizzle_gemm_cta must be an exact bijection over [0, gx) x [0, gy) for
  //    every combination of grid dimensions (including non-multiples of group_m).
  const int dims[][2] = {
      {1, 1}, {40, 1}, {1, 13}, {5, 7}, {40, 8}, {136, 17}, {17, 25}};
  for (const auto& d : dims) {
    const int gx = d[0], gy = d[1];
    std::vector<int> seen(static_cast<size_t>(gx) * gy, 0);
    bool ok = true;
    for (int by = 0; by < gy; ++by) {
      for (int bx = 0; bx < gx; ++bx) {
        const GemmTileCoord c = swizzle_gemm_cta(bx, by, gx, gy, 8);
        if (c.tile_n < 0 || c.tile_n >= gx || c.tile_m < 0 || c.tile_m >= gy) {
          ok = false;
          break;
        }
        ++seen[static_cast<size_t>(c.tile_m) * gx + c.tile_n];
      }
    }
    for (int cnt : seen)
      if (cnt != 1) ok = false;
    check(ok, "swizzle_gemm_cta is an exact bijection over grid tiles");
  }

  // 3) multi_gemv_locate_row maps linear-attention (4-way) and full-attention
  //    (3-way) fused projection grids to exact (proj, local_row) pairs.
  {
    const int lin_n[4] = {10240, 6144, 48, 48};
    const int total = 10240 + 6144 + 48 + 48;
    bool ok = true;
    int expected_p = 0, expected_r = 0;
    for (int g = 0; g < total; ++g) {
      const MultiGemvRowLoc loc = multi_gemv_locate_row(g, lin_n, 4);
      if (loc.proj != expected_p || loc.local_row != expected_r) {
        ok = false;
        break;
      }
      ++expected_r;
      if (expected_r == lin_n[expected_p]) {
        ++expected_p;
        expected_r = 0;
      }
    }
    const MultiGemvRowLoc oob_lo = multi_gemv_locate_row(-1, lin_n, 4);
    const MultiGemvRowLoc oob_hi = multi_gemv_locate_row(total, lin_n, 4);
    check(ok && oob_lo.proj == -1 && oob_hi.proj == -1,
          "multi_gemv_locate_row covers all 4 linear projections with exact bounds");
  }
  {
    const int qkv_n[3] = {12288, 1024, 1024};
    const MultiGemvRowLoc r0 = multi_gemv_locate_row(0, qkv_n, 3);
    const MultiGemvRowLoc r1 = multi_gemv_locate_row(12288, qkv_n, 3);
    const MultiGemvRowLoc r2 = multi_gemv_locate_row(13312, qkv_n, 3);
    const MultiGemvRowLoc r3 = multi_gemv_locate_row(14336, qkv_n, 3);
    check(r0.proj == 0 && r0.local_row == 0 && r1.proj == 1 &&
              r1.local_row == 0 && r2.proj == 2 && r2.local_row == 0 &&
              r3.proj == -1,
          "multi_gemv_locate_row maps 3-way QKV projections accurately");
  }

  // 4) Precomputed text RoPE inv_freq reproduces direct std::pow bit-for-bit
  //    for both pure-text and 3D multimodal (t, h, w) coordinates.
  double inv[kTextRotaryHalf];
  init_text_rope_inv_freq(inv);
  const int coords[][3] = {{0, 0, 0}, {17, 17, 17}, {5, 11, 23}, {4095, 4095, 4095}};
  bool rope_exact = true;
  for (const auto& c : coords) {
    Bf16 cos_row[kTextRotaryDim];
    Bf16 sin_row[kTextRotaryDim];
    fill_text_rope_row(inv, c[0], c[1], c[2], cos_row, sin_row);
    for (int i = 0; i < kTextRotaryHalf; ++i) {
      int pos = c[0];
      if ((i % 3) == 1 && i < 33)
        pos = c[1];
      else if ((i % 3) == 2 && i < 30)
        pos = c[2];
      const double ref_inv =
          1.0 / std::pow(static_cast<double>(kTextRopeTheta),
                         static_cast<double>(2 * i) / kTextRotaryDim);
      const double angle = ref_inv * pos;
      const Bf16 want_c(static_cast<float>(std::cos(angle)));
      const Bf16 want_s(static_cast<float>(std::sin(angle)));
      if (cos_row[i].bits != want_c.bits || sin_row[i].bits != want_s.bits) {
        rope_exact = false;
        break;
      }
    }
    for (int i = kTextRotaryHalf; i < kTextRotaryDim; ++i) {
      if (cos_row[i].bits != 0 || sin_row[i].bits != 0) rope_exact = false;
    }
  }
  check(rope_exact,
        "precomputed RoPE inv_freq matches reference std::pow bit for bit");

  // 5) CUDA Graph decode policy helper (UMM_QWEN38_NO_GRAPH + dump_dir guard).
  using umminfer::qwen38::fill_text_rope_row_scalar_pos;
  using umminfer::qwen38::qwen38_use_cuda_graph;
  check(qwen38_use_cuda_graph(nullptr, false),
        "CUDA Graph enabled by default when env unset and dump disabled");
  check(qwen38_use_cuda_graph("", false),
        "CUDA Graph enabled when UMM_QWEN38_NO_GRAPH is empty");
  check(qwen38_use_cuda_graph("0", false),
        "CUDA Graph enabled when UMM_QWEN38_NO_GRAPH=0");
  check(!qwen38_use_cuda_graph("1", false),
        "CUDA Graph disabled when UMM_QWEN38_NO_GRAPH=1");
  check(!qwen38_use_cuda_graph("true", false),
        "CUDA Graph disabled when UMM_QWEN38_NO_GRAPH=true");
  check(!qwen38_use_cuda_graph("00", false),
        "CUDA Graph disabled for non-'0' truthy env strings");
  check(!qwen38_use_cuda_graph(nullptr, true),
        "CUDA Graph disabled when parity dump_dir is enabled");
  check(!qwen38_use_cuda_graph("0", true),
        "dump_enabled overrides UMM_QWEN38_NO_GRAPH=0");
  check(!qwen38_use_cuda_graph("1", true),
        "CUDA Graph stays disabled when both env opt-out and dump are set");

  // 6) Scalar-position decode mRoPE helper (matching compute_text_mrope_devpos)
  //    must match fill_text_rope_row(inv, pos, pos, pos) bit-for-bit across
  //    text and multimodal-compressed decode positions.
  const int decode_positions[] = {0, 1, 7, 11, 17, 63, 128, 1024, 4095, 8191};
  bool devpos_rope_exact = true;
  for (int pos : decode_positions) {
    Bf16 ref_cos[kTextRotaryDim], ref_sin[kTextRotaryDim];
    Bf16 dev_cos[kTextRotaryDim], dev_sin[kTextRotaryDim];
    fill_text_rope_row(inv, pos, pos, pos, ref_cos, ref_sin);
    fill_text_rope_row_scalar_pos(inv, pos, dev_cos, dev_sin);
    for (int i = 0; i < kTextRotaryDim; ++i) {
      if (ref_cos[i].bits != dev_cos[i].bits ||
          ref_sin[i].bits != dev_sin[i].bits) {
        devpos_rope_exact = false;
        break;
      }
    }
  }
  check(devpos_rope_exact,
        "scalar-position decode mRoPE matches fill_text_rope_row bit for bit");
}

}  // namespace

int main() {
  test_greedy_is_argmax();
  test_subnormal_temperature();
  test_sampling_is_seed_deterministic();
  test_top_p_excludes_tail();
  test_extreme_logits();
  test_behavior_logprobs();
  test_prefix_match();
  test_render_parts_matches_full();
  test_turn_boundary_prefix_is_stable();
  test_resolve_snapshot_len();
  test_prefix_boundary_reuses_next_turn();
  test_fast_matches_reference();
  test_large_vocab_latency();
  test_video_preprocess_and_prompt();
  test_compute_media_hash();
  test_can_snapshot_multimodal_prefix();
  test_match_multimodal_prefix();
  test_mrope_suffix_continuity();
  test_plan_vision_slice_chunks();
  test_decode_helpers();
  if (g_failures != 0) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("qwen38 host tests passed\n");
  return 0;
}
