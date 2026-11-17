// SPDX-License-Identifier: Apache-2.0
#include "scheduler.hpp"

#include <cmath>
#include <stdexcept>

namespace umminfer::image2 {

FlowMatchSchedule build_schedule(const SchedulerConfig& cfg, int image_seq_len,
                                 int num_steps) {
  if (num_steps <= 0) throw std::invalid_argument("num_steps must be positive");
  if (image_seq_len <= 0) throw std::invalid_argument("image_seq_len must be positive");

  // pipeline.calculate_shift() with the scheduler config's own limits.
  const double m = (cfg.max_shift - cfg.base_shift) /
                   (cfg.max_image_seq_len - cfg.base_image_seq_len);
  const double b = cfg.base_shift - m * cfg.base_image_seq_len;
  const double mu = image_seq_len * m + b;
  const float exp_mu = static_cast<float>(std::exp(mu));

  std::vector<float> sig(num_steps);
  for (int i = 0; i < num_steps; ++i) {
    // np.linspace(1.0, 1/n, n) in float64, cast to float32.
    const double t64 = 1.0 - static_cast<double>(i) * (1.0 - 1.0 / num_steps) /
                                 static_cast<double>(num_steps - 1);
    const float t = static_cast<float>(t64);
    // _time_shift_exponential evaluated in float32 like numpy.
    sig[i] = exp_mu / (exp_mu + (1.0f / t - 1.0f));
  }

  // stretch_shift_to_terminal in float32.
  const float scale = (1.0f - sig.back()) / (1.0f - static_cast<float>(cfg.shift_terminal));
  for (float& s : sig) s = 1.0f - (1.0f - s) / scale;

  FlowMatchSchedule out;
  out.sigmas.reserve(num_steps + 1);
  out.timesteps.reserve(num_steps);
  for (float s : sig) {
    out.sigmas.push_back(s);
    out.timesteps.push_back(s * static_cast<float>(cfg.num_train_timesteps));
  }
  out.sigmas.push_back(0.0f);
  return out;
}

void euler_step_f32(float* sample, const float* velocity, double sigma, double sigma_next,
                    int n) {
  const float dt = static_cast<float>(sigma_next - sigma);
  for (int i = 0; i < n; ++i) sample[i] += dt * velocity[i];
}

}  // namespace umminfer::image2
