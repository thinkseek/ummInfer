// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <vector>

namespace umminfer::native {

// Qwen3-TTS leaves 0.1-0.5 s of high-frequency hiss after the last phoneme.
// The zero-crossing rate separates it from voiced speech but not from a
// fricative, which reaches the same rate; the energy above 8 kHz separates
// those two, so a trailing run is cut only when it is both high-rate and dark.

// Returns how many trailing samples to remove. Walks 10 ms windows backwards,
// skipping silent windows (rms < 60) and noise windows (zcr >= 0.4 *
// sample_rate), plus the silence between them, and stops at the first other
// window. The cut stays inside that one run.
//
// A final fricative passes the zero-crossing test, so the run must also be
// dark: at least 0.79 of its energy below 8 kHz. Measured on a Linux SM90 GPU
// host, the decoder's run holds 0.81-0.85 below 8 kHz and a final /s/, /f/ or
// /S/ only 0.54-0.76, so a bright run is left in place. The run must carry at
// least 30 ms of noise. At most 0.6 s of noise is removed; the silence around it
// is removed with it but does not consume that budget. Trailing silence on its
// own is kept.
std::size_t trailing_noise_cut(const std::vector<float>& samples, int sample_rate);

// Linear ramp to zero over the last `n` samples (0.5 ms at 24 kHz).
void fade_out(std::vector<float>& samples, std::size_t n);

// trailing_noise_cut followed by a 5 ms fade_out at the new end. Returns the
// input unchanged when the cut rule does not fire.
std::vector<float> trim_trailing_noise(std::vector<float> samples, int sample_rate);

}  // namespace umminfer::native
