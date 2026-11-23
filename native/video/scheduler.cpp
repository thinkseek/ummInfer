// SPDX-License-Identifier: Apache-2.0
#include "scheduler.hpp"

#include <cmath>
#include <stdexcept>

namespace umminfer::video {

FlowMatchSchedule build_flow_match_schedule(const FlowMatchConfig& cfg) {
  if (cfg.steps < 1 || cfg.steps > 500) {
    throw std::invalid_argument("flow-match steps must be in [1, 500]");
  }
  if (cfg.num_train_timesteps < 1) {
    throw std::invalid_argument("num_train_timesteps must be positive");
  }
  if (!std::isfinite(cfg.flow_shift) || cfg.flow_shift <= 0.0) {
    throw std::invalid_argument("flow_shift must be a finite positive number");
  }

  FlowMatchSchedule out;
  out.sigmas.resize(static_cast<size_t>(cfg.steps) + 1);
  out.timesteps.resize(static_cast<size_t>(cfg.steps));
  out.dt.resize(static_cast<size_t>(cfg.steps));

  for (int i = 0; i < cfg.steps; ++i) {
    const double t = 1.0 - static_cast<double>(i) / static_cast<double>(cfg.steps);
    const double shifted = (cfg.flow_shift * t) /
                           (1.0 + (cfg.flow_shift - 1.0) * t);
    out.sigmas[static_cast<size_t>(i)] = static_cast<float>(shifted);
    out.timesteps[static_cast<size_t>(i)] =
        static_cast<float>(shifted * static_cast<double>(cfg.num_train_timesteps));
  }
  out.sigmas[static_cast<size_t>(cfg.steps)] = 0.0f;

  for (int i = 0; i < cfg.steps; ++i) {
    out.dt[static_cast<size_t>(i)] =
        out.sigmas[static_cast<size_t>(i) + 1] - out.sigmas[static_cast<size_t>(i)];
  }
  return out;
}

void euler_step_host(float* latents, const float* velocity, float dt,
                     size_t count) {
  for (size_t i = 0; i < count; ++i) {
    latents[i] += dt * velocity[i];
  }
}

}  // namespace umminfer::video
