// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>

namespace umminfer::image2 {

// Values from Qwen-Image-2.1 scheduler/scheduler_config.json.
struct SchedulerConfig {
  int num_train_timesteps = 1000;
  int base_image_seq_len = 256;
  int max_image_seq_len = 8192;
  double base_shift = 0.5;
  double max_shift = 0.9;
  double shift_terminal = 0.02;
};

// FlowMatchEulerDiscreteScheduler with dynamic exponential shifting and
// stretch-to-terminal, exactly as the pipeline configures it.
struct FlowMatchSchedule {
  std::vector<float> sigmas;     // num_steps + 1, last is 0
  std::vector<float> timesteps;  // sigmas * 1000 (no terminal row)
};

FlowMatchSchedule build_schedule(const SchedulerConfig& cfg, int image_seq_len,
                                 int num_steps);

// prev = sample + (sigma_next - sigma) * velocity, computed in fp32.
void euler_step_f32(float* sample, const float* velocity, double sigma, double sigma_next,
                    int n);

}  // namespace umminfer::image2
