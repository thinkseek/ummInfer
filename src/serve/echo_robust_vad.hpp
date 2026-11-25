// SPDX-License-Identifier: Apache-2.0
// Reference-driven echo suppression for server-side realtime VAD.
//
// This is deliberately a small deterministic DSP slice, not a learned AEC.
// Each 10 ms near-end frame is aligned against a bounded ring of far-end PCM.
// A normalized-correlation match fits one gain coefficient and removes that
// component before the existing EnergyVad sees the frame. Echo-dominated
// nonlinear residuals (for example, moderate clipping) can be gated as echo;
// uncorrelated near-end speech remains and can start a turn immediately.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "energy_vad.hpp"

namespace umminfer::serve_detail {

struct EchoVadConfig {
  // Defaults: 500 ms search + one second of history = 8000 samples of lag
  // storage headroom and 16000 float samples (64,000 bytes) per session.
  int max_delay_ms = 500;
  int reference_history_ms = 1000;
  float min_correlation = 0.55f;
  float tracking_correlation = 0.98f;
  float min_reference_dbfs = -60.0f;
  float max_abs_gain = 8.0f;
  // A highly correlated frame with little unexplained energy is treated as
  // echo even when clipping leaves a nonlinear residual. Double talk with
  // sufficient independent near-end energy goes through the residual path.
  float echo_dominant_correlation = 0.82f;
  float echo_dominant_residual_ratio = 0.20f;
};

class EchoRobustVad {
 public:
  using Event = EnergyVad::Event;
  using FrameEvent = EnergyVad::FrameEvent;

  struct Alignment {
    bool aligned = false;
    size_t delay_samples = 0;
    double correlation = 0.0;
    double residual_ratio = 1.0;
    size_t candidate_evaluations = 0;
  };

  explicit EchoRobustVad(const VadConfig& vad_config = VadConfig{},
                         const EchoVadConfig& echo_config = EchoVadConfig{})
      : echo_config_(echo_config),
        vad_(vad_config),
        frame_samples_(vad_.frame_bytes() / 2),
        max_delay_samples_(static_cast<size_t>(
            std::clamp(echo_config.max_delay_ms, 0, 2000) * 16)),
        reference_(static_cast<size_t>(std::clamp(
                       echo_config.reference_history_ms,
                       std::clamp(echo_config.max_delay_ms, 0, 2000) +
                           vad_config.frame_ms,
                       5000) *
                   16),
                   0.0f),
        pending_(frame_samples_, 0.0f),
        residual_(frame_samples_, 0.0f),
        frame_pcm_(frame_samples_ * 2, '\0') {
    echo_config_.min_correlation =
        std::clamp(echo_config_.min_correlation, 0.0f, 1.0f);
    echo_config_.echo_dominant_correlation =
        std::clamp(echo_config_.echo_dominant_correlation,
                   echo_config_.min_correlation, 1.0f);
    echo_config_.tracking_correlation =
        std::clamp(echo_config_.tracking_correlation,
                   echo_config_.echo_dominant_correlation, 1.0f);
    echo_config_.echo_dominant_residual_ratio =
        std::clamp(echo_config_.echo_dominant_residual_ratio, 0.0f, 1.0f);
    echo_config_.max_abs_gain =
        std::clamp(echo_config_.max_abs_gain, 1.0f, 16.0f);
  }

  // Adds server-emitted far-end PCM. Multi-channel input is averaged to mono;
  // 8..96 kHz is linearly resampled to the near-end contract's 16 kHz. Invalid
  // or mid-response format changes clear the reference rather than risking a
  // false alignment. No samples are retained beyond reference_capacity().
  bool push_reference(const char* data, size_t bytes, int sample_rate = 16000,
                      int channels = 1) {
    if (bytes == 0) return true;
    const size_t frame_bytes = static_cast<size_t>(std::max(channels, 1)) * 2;
    if (data == nullptr || sample_rate < 8000 || sample_rate > 96000 ||
        channels < 1 || channels > 32 || bytes % frame_bytes != 0) {
      reset_reference();
      return false;
    }
    if (reference_rate_ != 0 &&
        (reference_rate_ != sample_rate || reference_channels_ != channels)) {
      reset_reference();
    }
    reference_rate_ = sample_rate;
    reference_channels_ = channels;
    for (size_t offset = 0; offset < bytes; offset += frame_bytes) {
      int64_t mixed = 0;
      for (int channel = 0; channel < channels; ++channel) {
        mixed += decode_pcm16(data + offset + static_cast<size_t>(channel) * 2);
      }
      const auto mono = static_cast<int16_t>(mixed / channels);
      if (sample_rate == 16000) {
        write_reference(static_cast<float>(mono));
      } else {
        push_resampled(mono, sample_rate);
      }
    }
    return true;
  }

  // Response boundaries call this so one assistant response can never supply
  // reference samples for another response. The VAD/noise-floor state remains.
  void reset_reference() {
    reference_total_ = 0;
    reference_write_ = 0;
    reference_rate_ = 0;
    reference_channels_ = 0;
    resampler_phase_ = 0;
    resampler_previous_ = 0;
    resampler_have_previous_ = false;
    tracking_ = false;
    tracked_lag_ = 0;
    last_alignment_ = Alignment{};
  }

  // Arbitrary even-byte chunking is accepted. Event offsets always point to
  // the last byte of the completed frame in this particular push() call.
  std::vector<FrameEvent> push(const char* data, size_t bytes) {
    std::vector<FrameEvent> events;
    for (size_t offset = 0; offset + 1 < bytes; offset += 2) {
      pending_[pending_samples_++] =
          static_cast<float>(decode_pcm16(data + offset));
      if (pending_samples_ < frame_samples_) continue;
      pending_samples_ = 0;
      auto frame_events = vad_.push(process_frame().data(), frame_pcm_.size());
      for (auto& event : frame_events) {
        event.bytes_consumed = offset + 2;
        events.push_back(event);
      }
    }
    return events;
  }

  bool speech() const { return vad_.speech(); }
  int silence_ms() const { return vad_.silence_ms(); }
  size_t frame_bytes() const { return vad_.frame_bytes(); }
  size_t reference_size() const {
    return std::min<size_t>(reference_total_, reference_.size());
  }
  size_t reference_capacity() const { return reference_.size(); }
  size_t max_delay_samples() const { return max_delay_samples_; }
  const Alignment& last_alignment() const { return last_alignment_; }

 private:
  struct Match {
    size_t lag = 0;
    double dot = 0.0;
    double reference_energy = 0.0;
    double correlation = 0.0;
  };

  static int16_t decode_pcm16(const char* data) {
    const auto low = static_cast<uint16_t>(static_cast<uint8_t>(data[0]));
    const auto high = static_cast<uint16_t>(static_cast<uint8_t>(data[1]));
    return static_cast<int16_t>(low | (high << 8));
  }

  static void append_pcm16(std::string& out, size_t offset, int value) {
    const auto sample = static_cast<uint16_t>(static_cast<int16_t>(
        std::clamp(value, static_cast<int>(std::numeric_limits<int16_t>::min()),
                   static_cast<int>(std::numeric_limits<int16_t>::max()))));
    out[offset] = static_cast<char>(sample & 0xff);
    out[offset + 1] = static_cast<char>((sample >> 8) & 0xff);
  }

  float reference_at(int64_t absolute_index) const {
    // The leading edge of a response has digital silence before its first
    // emitted sample. Treat that pre-response region as zero so a delay that
    // splits the first echo frame does not become a false speech onset.
    if (absolute_index < 0) return 0.0f;
    return reference_[static_cast<size_t>(absolute_index) % reference_.size()];
  }

  void write_reference(float sample) {
    reference_[reference_write_] = sample;
    reference_write_ = (reference_write_ + 1) % reference_.size();
    ++reference_total_;
  }

  // Integer phase accumulation avoids drift at chunk boundaries. One input
  // sample is held so interpolation never looks into the future.
  void push_resampled(int16_t sample, int sample_rate) {
    if (!resampler_have_previous_) {
      resampler_previous_ = sample;
      resampler_have_previous_ = true;
      resampler_phase_ = 0;
      return;
    }
    while (resampler_phase_ < 16000) {
      const double fraction = static_cast<double>(resampler_phase_) / 16000.0;
      write_reference(static_cast<float>(
          resampler_previous_ +
          (static_cast<int32_t>(sample) - resampler_previous_) * fraction));
      resampler_phase_ += sample_rate;
    }
    resampler_phase_ -= 16000;
    resampler_previous_ = sample;
  }

  Match evaluate(size_t lag, size_t stride, double near_energy) {
    Match match;
    match.lag = lag;
    const int64_t reference_start = static_cast<int64_t>(reference_total_) -
                                    static_cast<int64_t>(lag) -
                                    static_cast<int64_t>(frame_samples_);
    for (size_t i = 0; i < frame_samples_; i += stride) {
      const float reference = reference_at(reference_start + i);
      match.dot += static_cast<double>(pending_[i]) * reference;
      match.reference_energy += static_cast<double>(reference) * reference;
    }
    if (match.reference_energy > 1e-9 && near_energy > 1e-9) {
      match.correlation =
          std::abs(match.dot) / std::sqrt(match.reference_energy * near_energy);
    }
    ++last_alignment_.candidate_evaluations;
    return match;
  }

  static bool better(const Match& candidate, const Match& current) {
    return candidate.correlation > current.correlation;
  }

  const std::string& process_frame() {
    last_alignment_ = Alignment{};
    double near_energy = 0.0;
    for (float sample : pending_) {
      near_energy += static_cast<double>(sample) * sample;
    }
    if (reference_total_ == 0 || near_energy <= 1e-9) {
      for (size_t i = 0; i < frame_samples_; ++i) {
        append_pcm16(frame_pcm_, i * 2,
                     static_cast<int>(std::lround(pending_[i])));
      }
      return frame_pcm_;
    }

    const size_t max_lag = std::min(max_delay_samples_, reference_total_ - 1);
    Match best;
    // Once acquired, real acoustic delay normally changes by much less than
    // one millisecond per 10 ms frame. Search that neighborhood exhaustively.
    if (tracking_) {
      const size_t begin = tracked_lag_ > 16 ? tracked_lag_ - 16 : 0;
      const size_t end = std::min(max_lag, tracked_lag_ + 16);
      for (size_t lag = begin; lag <= end; ++lag) {
        Match candidate = evaluate(lag, 1, near_energy);
        if (better(candidate, best)) best = candidate;
      }
    }
    // Acquisition and larger delay jumps use a bounded full-lag screen. Only
    // 16 taps per candidate are examined (or the full frame for a sparse onset
    // missed by those taps); the best 16 finalists are then evaluated over the
    // complete frame. This handles arbitrary integer delay without the aliasing
    // of a 10 ms coarse grid on broadband audio.
    if (!tracking_ || best.correlation < echo_config_.tracking_correlation) {
      double coarse_near_energy = 0.0;
      for (size_t i = 0; i < frame_samples_; i += 10) {
        coarse_near_energy += static_cast<double>(pending_[i]) * pending_[i];
      }
      const size_t acquisition_stride = coarse_near_energy > 1e-9 ? 10 : 1;
      if (coarse_near_energy <= 1e-9) coarse_near_energy = near_energy;
      std::array<Match, 16> finalists{};
      for (size_t lag = 0; lag <= max_lag; ++lag) {
        Match candidate =
            evaluate(lag, acquisition_stride, coarse_near_energy);
        if (candidate.correlation <= finalists.back().correlation) continue;
        size_t position = finalists.size() - 1;
        while (position > 0 &&
               finalists[position - 1].correlation < candidate.correlation) {
          finalists[position] = finalists[position - 1];
          --position;
        }
        finalists[position] = candidate;
      }
      for (const Match& finalist : finalists) {
        if (finalist.correlation <= 0.0) continue;
        Match candidate = evaluate(finalist.lag, 1, near_energy);
        if (better(candidate, best)) best = candidate;
      }
    }

    constexpr double kFullScale = 32768.0 * 32768.0;
    const double min_reference_energy =
        kFullScale *
        std::pow(10.0, static_cast<double>(echo_config_.min_reference_dbfs) /
                           10.0) *
        frame_samples_;
    last_alignment_.delay_samples = best.lag;
    last_alignment_.correlation = best.correlation;
    if (best.correlation < echo_config_.min_correlation ||
        best.reference_energy < min_reference_energy ||
        best.reference_energy <= 1e-9) {
      tracking_ = false;
      for (size_t i = 0; i < frame_samples_; ++i) {
        append_pcm16(frame_pcm_, i * 2,
                     static_cast<int>(std::lround(pending_[i])));
      }
      return frame_pcm_;
    }

    last_alignment_.aligned = true;
    tracking_ = true;
    tracked_lag_ = best.lag;
    double gain = best.dot / best.reference_energy;
    gain = std::clamp(gain, -static_cast<double>(echo_config_.max_abs_gain),
                      static_cast<double>(echo_config_.max_abs_gain));
    const int64_t reference_start = static_cast<int64_t>(reference_total_) -
                                    static_cast<int64_t>(best.lag) -
                                    static_cast<int64_t>(frame_samples_);
    double residual_energy = 0.0;
    for (size_t i = 0; i < frame_samples_; ++i) {
      residual_[i] = pending_[i] - static_cast<float>(gain) *
                                       reference_at(reference_start + i);
      residual_energy += static_cast<double>(residual_[i]) * residual_[i];
    }
    const double residual_ratio = residual_energy / near_energy;
    last_alignment_.residual_ratio = residual_ratio;
    // A clamped or otherwise poor subtraction must not amplify noise. Fall
    // back to the unmodified near-end frame in that case.
    const bool use_residual = residual_ratio <= 1.0;
    const bool echo_dominant =
        best.correlation >= echo_config_.echo_dominant_correlation &&
        residual_ratio <= echo_config_.echo_dominant_residual_ratio;
    for (size_t i = 0; i < frame_samples_; ++i) {
      const float output = echo_dominant
                               ? 0.0f
                               : (use_residual ? residual_[i] : pending_[i]);
      append_pcm16(frame_pcm_, i * 2, static_cast<int>(std::lround(output)));
    }
    return frame_pcm_;
  }

  EchoVadConfig echo_config_;
  EnergyVad vad_;
  size_t frame_samples_;
  size_t max_delay_samples_;
  std::vector<float> reference_;
  size_t reference_total_ = 0;
  size_t reference_write_ = 0;
  int reference_rate_ = 0;
  int reference_channels_ = 0;
  int resampler_phase_ = 0;
  int16_t resampler_previous_ = 0;
  bool resampler_have_previous_ = false;
  std::vector<float> pending_;
  std::vector<float> residual_;
  size_t pending_samples_ = 0;
  std::string frame_pcm_;
  bool tracking_ = false;
  size_t tracked_lag_ = 0;
  Alignment last_alignment_;
};

}  // namespace umminfer::serve_detail
