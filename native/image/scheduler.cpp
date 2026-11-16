// SPDX-License-Identifier: Apache-2.0
#include "scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace umminfer::image {

FlowMatchSchedule build_schedule(const SchedulerConfig& cfg, int image_seq_len, int num_steps) {
  if (num_steps <= 0) throw std::invalid_argument("num_steps must be positive");
  if (image_seq_len <= 0) throw std::invalid_argument("image_seq_len must be positive");

  // pipeline.calculate_shift(): linear mu over the packed sequence length.
  const double m = (cfg.max_shift - cfg.base_shift) /
                   (cfg.max_image_seq_len - cfg.base_image_seq_len);
  const double b = cfg.base_shift - m * cfg.base_image_seq_len;
  const double mu = image_seq_len * m + b;
  const double exp_mu = std::exp(mu);

  std::vector<double> sig(num_steps);
  for (int i = 0; i < num_steps; ++i) {
    // np.linspace(1.0, 1/num_steps, num_steps)
    const double t = 1.0 - static_cast<double>(i) * (1.0 - 1.0 / num_steps) /
                               static_cast<double>(num_steps - 1);
    // _time_shift_exponential: e^mu / (e^mu + (1/t - 1))
    sig[i] = exp_mu / (exp_mu + (1.0 / t - 1.0));
  }

  // stretch_shift_to_terminal: final sigma becomes shift_terminal.
  const double scale = (1.0 - sig.back()) / (1.0 - cfg.shift_terminal);
  for (double& s : sig) s = 1.0 - (1.0 - s) / scale;

  FlowMatchSchedule out;
  out.sigmas.reserve(num_steps + 1);
  out.timesteps.reserve(num_steps);
  for (double s : sig) {
    out.sigmas.push_back(static_cast<float>(s));
    out.timesteps.push_back(static_cast<float>(s * cfg.num_train_timesteps));
  }
  out.sigmas.push_back(0.0f);
  return out;
}

void euler_step_f32(float* sample, const float* velocity, double sigma, double sigma_next, int n) {
  const float dt = static_cast<float>(sigma_next - sigma);
  for (int i = 0; i < n; ++i) sample[i] += dt * velocity[i];
}

}  // namespace umminfer::image
