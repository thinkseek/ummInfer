// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <vector>

namespace umminfer::image {

// FlowMatchEulerDiscreteScheduler, diffusers 0.36.0 configuration of
// Qwen-Image-2512: dynamic exponential shifting + terminal stretch.
// All math is fp64 locally; the pipeline consumes sigmas/timesteps as fp32.
// Reference: diffusers/schedulers/scheduling_flow_match_euler_discrete.py.
struct FlowMatchSchedule {
  std::vector<float> timesteps;  // N entries, = sigmas[:N] * 1000
  std::vector<float> sigmas;     // N+1 entries, last one is 0.0
};

struct SchedulerConfig {
  int num_train_timesteps = 1000;
  double base_image_seq_len = 256.0;
  double max_image_seq_len = 8192.0;
  double base_shift = 0.5;
  double max_shift = 0.9;
  double shift_terminal = 0.02;
};

// image_seq_len is the PACKED latent token count: (H/16) * (W/16) for a
// WxH image (VAE 8x + 2x patch packing). num_steps defaults to 50 upstream.
FlowMatchSchedule build_schedule(const SchedulerConfig& cfg, int image_seq_len, int num_steps);

// One deterministic Euler step in fp32, matching scheduler.step():
// prev = sample + (sigma_next - sigma) * velocity. n = element count.
void euler_step_f32(float* sample, const float* velocity, double sigma, double sigma_next, int n);

}  // namespace umminfer::image
