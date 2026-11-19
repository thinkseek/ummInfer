// SPDX-License-Identifier: Apache-2.0
// Self-check for the trailing-noise trim. Host only, no CUDA:
//   cc -std=c++20 -I native/tts native/tts/tail_trim.cpp native/tts/tail_trim_test.cpp
// Amplitudes are in the codec's float domain; 0.03 is about -31 dBFS.
#include <cmath>
#include <cstdio>
#include <vector>

#include "tail_trim.hpp"

namespace {

constexpr int kRate = 24000;
constexpr float kLevel = 0.03f;

std::vector<float> voiced(int n) {
  std::vector<float> v(n);
  for (int i = 0; i < n; ++i)
    v[i] = kLevel * std::sin(2.0f * 3.14159265f * 200 * i / kRate);
  return v;
}

std::vector<float> noise(int n) {
  std::vector<float> v(n);
  for (int i = 0; i < n; ++i) v[i] = (i % 2) ? -kLevel : kLevel;
  return v;
}

std::vector<float> silence(int n) { return std::vector<float>(n, 0.0f); }

void append(std::vector<float>& dst, const std::vector<float>& src) {
  dst.insert(dst.end(), src.begin(), src.end());
}

int failures = 0;

void check(const char* name, bool ok) {
  std::printf("%-40s %s\n", name, ok ? "ok" : "FAIL");
  if (!ok) ++failures;
}

}  // namespace

int main() {
  {
    std::vector<float> s = voiced(kRate);
    check("clean speech is untouched",
          umminfer::native::trailing_noise_cut(s, kRate) == 0);
  }
  {
    std::vector<float> s = voiced(kRate);
    append(s, noise(kRate * 3 / 10));
    append(s, silence(kRate * 2 / 10));
    const auto cut = umminfer::native::trailing_noise_cut(s, kRate);
    check("noise plus trailing silence is cut", cut == size_t(kRate / 2));
    const auto out = umminfer::native::trim_trailing_noise(s, kRate);
    check("trimmed length", out.size() == size_t(kRate));
    check("fade reaches zero", out.back() == 0.0f);
    check("speech before the cut is kept", out[kRate / 2] == s[kRate / 2]);
  }
  {
    std::vector<float> s = voiced(kRate);
    append(s, noise(kRate * 20 / 1000));
    check("20 ms of noise is below the floor",
          umminfer::native::trailing_noise_cut(s, kRate) == 0);
  }
  {
    std::vector<float> s = voiced(kRate);
    append(s, noise(kRate));
    check("cut is capped at 600 ms",
          umminfer::native::trailing_noise_cut(s, kRate) == size_t(kRate * 6 / 10));
  }
  {
    std::vector<float> s = silence(kRate);
    check("pure silence is kept",
          umminfer::native::trailing_noise_cut(s, kRate) == 0);
  }
  {
    std::vector<float> s = noise(kRate);
    check("all-noise output is kept",
          umminfer::native::trailing_noise_cut(s, kRate) == 0);
  }
  {
    std::vector<float> s;
    check("empty input is safe",
          umminfer::native::trim_trailing_noise(s, kRate).empty());
  }
  std::printf("%s\n", failures ? "FAILED" : "all checks passed");
  return failures ? 1 : 0;
}
