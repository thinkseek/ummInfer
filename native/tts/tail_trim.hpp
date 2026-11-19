// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <vector>

namespace umminfer::native {

// Qwen3-TTS leaves 0.2-0.5 s of near-Nyquist noise after the last phoneme.
// The zero-crossing rate separates it from voiced speech; it does not separate
// it from a fricative, which reaches the same rate for at most 0.16 s, so the
// rule is duration-bounded and conservative.

// Returns how many trailing samples to remove. Walks 10 ms windows backwards,
// skipping silent windows (rms < 60) and noise windows (zcr >= 0.4 *
// sample_rate), plus the silence between them, and stops at the first other
// window. The run must carry at least 200 ms of noise, because a real
// fricative measures up to 160 ms and the decoder's run 240 ms; a shorter run
// is left in place. The cut stays inside that one run. At most 0.6 s of noise
// is removed; the
// silence around it is removed with it but does not consume that budget, so
// an utterance ending in a pause does not keep its noise. Trailing silence on
// its own is kept.
std::size_t trailing_noise_cut(const std::vector<float>& samples, int sample_rate);

// Linear ramp to zero over the last `n` samples (0.5 ms at 24 kHz).
void fade_out(std::vector<float>& samples, std::size_t n);

// trailing_noise_cut followed by a 5 ms fade_out at the new end. Returns the
// input unchanged when the cut rule does not fire.
std::vector<float> trim_trailing_noise(std::vector<float> samples, int sample_rate);

}  // namespace umminfer::native
