// SPDX-License-Identifier: Apache-2.0
// Flow-matching shifted timestep/sigma schedule for Wan2.2 / Wan2.1 video DiT.
#pragma once

#include <cstddef>
#include <vector>

namespace umminfer::video {

struct FlowMatchConfig {
  int num_train_timesteps = 1000;
  double flow_shift = 5.0;
  int steps = 20;
};

struct FlowMatchSchedule {
  std::vector<float> sigmas;     // length steps + 1, sigmas[steps] == 0.0f
  std::vector<float> timesteps;  // length steps, sigmas[i] * num_train_timesteps
  std::vector<float> dt;         // length steps, sigmas[i + 1] - sigmas[i] (< 0)
};

// Builds the Wan flow-matching schedule:
//   t_i = 1.0 - i / steps  (for i = 0..steps-1)
//   sigma_i = (shift * t_i) / (1.0 + (shift - 1.0) * t_i)
//   sigma_steps = 0.0
// Throws std::invalid_argument on invalid steps/shift/num_train_timesteps.
FlowMatchSchedule build_flow_match_schedule(const FlowMatchConfig& cfg);

// Host reference Euler step: latents[j] += dt * velocity[j].
void euler_step_host(float* latents, const float* velocity, float dt,
                     size_t count);

}  // namespace umminfer::video
