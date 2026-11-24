// SPDX-License-Identifier: Apache-2.0
#include "scheduler.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace umminfer::video {
namespace {

// Diffusers subtracts this from sigma_0 when it is exactly 1.0 so that the
// UniPC update never evaluates log(0).
constexpr double kSigmaEpsilon = 1e-6;

inline double lambda_of(double sigma) {
  return std::log(1.0 - sigma) - std::log(sigma);
}

}  // namespace

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

  const int steps = cfg.steps;
  const double nts = static_cast<double>(cfg.num_train_timesteps);

  FlowMatchSchedule out;
  out.sigmas.resize(static_cast<size_t>(steps) + 1);
  out.timesteps.resize(static_cast<size_t>(steps));
  out.dt.resize(static_cast<size_t>(steps));

  // np.linspace(1.0, 1 / num_train_timesteps, steps + 1)[:-1]
  const double lin_start = 1.0;
  const double lin_stop = 1.0 / nts;
  const double lin_step = (lin_stop - lin_start) / static_cast<double>(steps);

  std::vector<double> shifted(static_cast<size_t>(steps));
  for (int i = 0; i < steps; ++i) {
    const double s = lin_start + static_cast<double>(i) * lin_step;
    shifted[static_cast<size_t>(i)] =
        (cfg.flow_shift * s) / (1.0 + (cfg.flow_shift - 1.0) * s);
  }
  if (std::fabs(shifted[0] - 1.0) < kSigmaEpsilon) {
    shifted[0] -= kSigmaEpsilon;
  }

  for (int i = 0; i < steps; ++i) {
    const double sigma = shifted[static_cast<size_t>(i)];
    out.sigmas[static_cast<size_t>(i)] = static_cast<float>(sigma);
    // Diffusers casts timesteps to int64, which truncates toward zero.
    out.timesteps[static_cast<size_t>(i)] =
        static_cast<float>(static_cast<int64_t>(sigma * nts));
  }
  out.sigmas[static_cast<size_t>(steps)] = 0.0f;

  for (int i = 0; i < steps; ++i) {
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

void UniPCFlowMatchSolver::reset(const FlowMatchSchedule& schedule) {
  if (schedule.sigmas.size() < 2) {
    throw std::invalid_argument("UniPC solver requires at least one step");
  }
  sigmas_ = schedule.sigmas;
  prev_x0_.clear();
  prev2_x0_.clear();
  last_sample_.clear();
  x0_scratch_.clear();
  has_prev_ = false;
  has_prev2_ = false;
  last_order_ = 1;
  lower_order_nums_ = 0;
}

void UniPCFlowMatchSolver::apply_corrector(int step_index, float* sample,
                                           const float* model_t, size_t count) {
  // Diffusers' multistep_uni_c_bh_update with predict_x0=True, bh2 and
  // flow_prediction. Histories hold x0 predictions, not raw velocities.
  const int order = last_order_;
  const double sigma_prev = static_cast<double>(sigmas_[step_index - 1]);
  const double sigma_cur = static_cast<double>(sigmas_[step_index]);
  const double alpha_prev = 1.0 - sigma_prev;
  const double alpha_cur = 1.0 - sigma_cur;
  const double h = lambda_of(sigma_cur) - lambda_of(sigma_prev);
  const double hh = -h;
  const double h_phi_1 = std::expm1(hh);
  const double b_h = std::expm1(hh);

  double rk = 0.0;
  double rho_d1 = 0.0;
  double rho_t = 0.5;
  if (order >= 2) {
    const double sigma_si = static_cast<double>(sigmas_[step_index - 2]);
    rk = (lambda_of(sigma_si) - lambda_of(sigma_prev)) / h;
    // R = [[1, 1], [rk, 1]], b = [b1, b2]; solve for rhos_c.
    const double h_phi_k = h_phi_1 / hh - 1.0;
    double factorial = 1.0;
    const double b1 = h_phi_k * factorial / b_h;
    factorial *= 2.0;
    const double b2 = (h_phi_k / hh - 1.0 / factorial) * factorial / b_h;
    rho_d1 = (b1 - b2) / (1.0 - rk);
    rho_t = b2 - rk * rho_d1;
  }

  const double scale_x = sigma_cur / sigma_prev;
  const double scale_m0 = alpha_cur * h_phi_1;
  const double scale_corr = alpha_cur * b_h;

  for (size_t j = 0; j < count; ++j) {
    const double x = static_cast<double>(last_sample_[j]);
    const double m0 = static_cast<double>(prev_x0_[j]);
    double corr = 0.0;
    if (order >= 2) {
      const double d1 =
          (static_cast<double>(prev2_x0_[j]) - m0) / rk;
      corr = rho_d1 * d1;
    }
    corr += rho_t * (static_cast<double>(model_t[j]) - m0);
    sample[j] = static_cast<float>(scale_x * x - scale_m0 * m0 -
                                   scale_corr * corr);
  }
}

void UniPCFlowMatchSolver::step(int step_index, float* sample,
                                const float* model_output, size_t count) {
  const int total_steps = static_cast<int>(sigmas_.size()) - 1;
  if (step_index < 0 || step_index >= total_steps) {
    throw std::out_of_range("UniPC step_index out of range");
  }
  if (count == 0) {
    return;
  }
  if (x0_scratch_.size() != count) {
    x0_scratch_.resize(count);
    prev_x0_.resize(count);
    prev2_x0_.resize(count);
    last_sample_.resize(count);
    has_prev_ = false;
    has_prev2_ = false;
  }

  // convert_model_output: x0 = sample - sigma_t * model_output.
  const float sigma_t = sigmas_[static_cast<size_t>(step_index)];
  for (size_t j = 0; j < count; ++j) {
    x0_scratch_[j] = sample[j] - sigma_t * model_output[j];
  }

  if (step_index > 0 && has_prev_) {
    apply_corrector(step_index, sample, x0_scratch_.data(), count);
  }

  if (has_prev_) {
    prev2_x0_ = prev_x0_;
    has_prev2_ = true;
  }
  prev_x0_ = x0_scratch_;
  has_prev_ = true;
  // Diffusers stores `last_sample` before the predictor updates it; the next
  // step's corrector needs the sample that fed this predictor.
  last_sample_.assign(sample, sample + count);

  // lower_order_final: the last step drops to first order.
  int this_order = total_steps - step_index;
  if (this_order > 2) {
    this_order = 2;
  }
  if (this_order > lower_order_nums_ + 1) {
    this_order = lower_order_nums_ + 1;
  }

  const double sigma_s0 = static_cast<double>(sigmas_[static_cast<size_t>(step_index)]);
  const double sigma_next =
      static_cast<double>(sigmas_[static_cast<size_t>(step_index) + 1]);
  const double alpha_s0 = 1.0 - sigma_s0;
  const double alpha_next = 1.0 - sigma_next;
  const double h = lambda_of(sigma_next) - lambda_of(sigma_s0);
  const double hh = -h;
  const double h_phi_1 = std::expm1(hh);
  const double b_h = std::expm1(hh);

  double rho_p = 0.0;
  double inv_rk = 0.0;
  if (this_order >= 2) {
    // D1s use x0 at step_index - 1, which after the history shift is prev2_x0_.
    const double sigma_si = static_cast<double>(sigmas_[static_cast<size_t>(step_index) - 1]);
    const double rk = (lambda_of(sigma_si) - lambda_of(sigma_s0)) / h;
    inv_rk = 1.0 / rk;
    rho_p = 0.5;  // rhos_p is hard-coded for order 2 in Diffusers.
  }

  const double scale_x = sigma_next / sigma_s0;
  const double scale_m0 = alpha_next * h_phi_1;
  const double scale_corr = alpha_next * b_h;

  for (size_t j = 0; j < count; ++j) {
    const double m0 = static_cast<double>(prev_x0_[j]);
    double pred_res = 0.0;
    if (this_order >= 2) {
      const double d1 = (static_cast<double>(prev2_x0_[j]) - m0) * inv_rk;
      pred_res = rho_p * d1;
    }
    sample[j] =
        static_cast<float>(scale_x * static_cast<double>(sample[j]) -
                           scale_m0 * m0 - scale_corr * pred_res);
  }

  last_order_ = this_order;
  if (lower_order_nums_ < 2) {
    ++lower_order_nums_;
  }
}

}  // namespace umminfer::video
