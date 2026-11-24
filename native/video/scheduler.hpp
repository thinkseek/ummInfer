// SPDX-License-Identifier: Apache-2.0
// Flow-matching shifted timestep/sigma schedule for Wan2.2 / Wan2.1 video DiT,
// plus the UniPC-BH2 multistep solver used by Wan2.2-TI2V-5B.
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
  std::vector<float> timesteps;  // length steps, truncated sigma_i * num_train_timesteps
  std::vector<float> dt;         // length steps, sigmas[i + 1] - sigmas[i] (< 0)
};

// Builds the Wan flow-matching schedule exactly as Diffusers'
// UniPCMultistepScheduler with use_flow_sigmas=True, flow_shift, and
// final_sigmas_type="zero":
//   s_i      = linspace(1.0, 1/num_train_timesteps, steps + 1)[i]  (i < steps)
//   sigma_i  = (shift * s_i) / (1 + (shift - 1) * s_i)
//   sigma_0 -= 1e-6 when sigma_0 == 1 (avoids log(0) in the UniPC solver)
//   sigma_steps = 0.0
//   timestep_i  = int64(sigma_i * num_train_timesteps)   (Diffusers truncates)
// sigmas are stored as float32, matching the Diffusers tensor dtype.
// Throws std::invalid_argument on invalid steps/shift/num_train_timesteps.
FlowMatchSchedule build_flow_match_schedule(const FlowMatchConfig& cfg);

// Host reference Euler step: latents[j] += dt * velocity[j].
void euler_step_host(float* latents, const float* velocity, float dt,
                     size_t count);

// Second-order UniPC-BH2 flow-matching solver (UniP + UniC corrector),
// numerically equivalent to Diffusers' UniPCMultistepScheduler configured with
// solver_order=2, solver_type="bh2", predict_x0=true,
// prediction_type="flow_prediction", lower_order_final=true and an empty
// disable_corrector list. `model_output` is the raw flow velocity; the sample
// is updated in place.
//
// Usage mirrors the Diffusers loop: one model evaluation per step, with
// step() doing the corrector for the previous step and the predictor for the
// next one. The corrector needs the model output of the current step evaluated
// at the sample produced by the previous predictor, which is exactly the
// model_output passed here.
class UniPCFlowMatchSolver {
 public:
  void reset(const FlowMatchSchedule& schedule);

  // step_index is 0-based; count is the number of elements in sample and
  // model_output.
  void step(int step_index, float* sample, const float* model_output,
            size_t count);

  // Order used by the most recent predictor (1 on the first step, 2 after
  // warm-up, 1 on the final step when lower_order_final applies).
  int last_order() const { return last_order_; }

 private:
  void apply_corrector(int step_index, float* sample, const float* model_t,
                       size_t count);

  std::vector<float> sigmas_;
  std::vector<float> prev_x0_;    // model output (x0) at step_index - 1
  std::vector<float> prev2_x0_;   // model output (x0) at step_index - 2
  std::vector<float> last_sample_;  // sample fed to the previous predictor
  std::vector<float> x0_scratch_;   // current converted model output
  bool has_prev_ = false;
  bool has_prev2_ = false;
  int last_order_ = 1;
  int lower_order_nums_ = 0;
};

}  // namespace umminfer::video
