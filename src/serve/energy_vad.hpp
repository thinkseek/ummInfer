// SPDX-License-Identifier: Apache-2.0
// Server-side endpoint detection for the realtime WebSocket.
//
// Energy VAD over 10 ms frames of PCM16 mono 16 kHz. Speech is a frame whose
// RMS level exceeds an adaptive noise floor by margin_db, and also exceeds
// abs_floor_dbfs. The floor learns only from non-speech frames, so a long
// utterance cannot drag it up. A turn ends after silence_ms of continuous
// non-speech frames.
//
// ponytail: energy only, no spectral or model VAD. It cannot separate loud
// steady noise from speech; add a spectral gate when noise rejection matters.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace umminfer::serve_detail {

struct VadConfig {
  int silence_ms = 500;            // continuous non-speech before speech_stopped
  int frame_ms = 10;               // analysis frame
  float abs_floor_dbfs = -55.0f;   // never report speech below this level
  float margin_db = 6.0f;          // speech must exceed the noise floor by this
};

class EnergyVad {
 public:
  enum class Event { kNone, kSpeechStarted, kSpeechStopped };

  // bytes_consumed counts from the start of the buffer passed to push(), so a
  // caller can cut a turn at the frame where speech started even when one
  // append carries several seconds of audio.
  struct FrameEvent {
    Event event;
    size_t bytes_consumed;
  };

  explicit EnergyVad(const VadConfig& config = VadConfig{})
      : config_(config),
        frame_samples_(std::max(16, config.frame_ms * 16)),
        silence_frames_(
            std::max(1, config.silence_ms / std::max(1, config.frame_ms))) {}

  // Consumes little-endian PCM16 bytes. Returns one event per completed frame;
  // a single call can report start and stop when the caller sends a long block.
  std::vector<FrameEvent> push(const char* data, size_t bytes) {
    std::vector<FrameEvent> events;
    for (size_t offset = 0; offset + 1 < bytes; offset += 2) {
      const auto low = static_cast<uint16_t>(static_cast<uint8_t>(data[offset]));
      const auto high =
          static_cast<uint16_t>(static_cast<uint8_t>(data[offset + 1]));
      const auto sample = static_cast<int16_t>(low | (high << 8));
      energy_ += static_cast<double>(sample) * static_cast<double>(sample);
      if (++pending_ < frame_samples_) continue;
      const Event event = classify(energy_ / static_cast<double>(pending_));
      energy_ = 0.0;
      pending_ = 0;
      if (event != Event::kNone) {
        events.push_back(FrameEvent{event, offset + 2});
      }
    }
    return events;
  }

  bool speech() const { return speech_; }
  int silence_ms() const { return config_.silence_ms; }
  size_t frame_bytes() const { return frame_samples_ * 2; }

 private:
  Event classify(double mean_square) {
    constexpr double kFullScale = 32768.0 * 32768.0;
    const double db = 10.0 * std::log10(mean_square / kFullScale + 1e-12);
    const double threshold = std::max<double>(
        config_.abs_floor_dbfs, noise_dbfs_ + config_.margin_db);
    const bool voiced = db > threshold;
    if (!speech_) {
      if (voiced) {
        speech_ = true;
        silent_frames_ = 0;
        return Event::kSpeechStarted;
      }
      noise_dbfs_ = noise_dbfs_ * 0.95 + db * 0.05;
      return Event::kNone;
    }
    if (voiced) {
      silent_frames_ = 0;
      return Event::kNone;
    }
    noise_dbfs_ = noise_dbfs_ * 0.98 + db * 0.02;
    if (++silent_frames_ >= silence_frames_) {
      speech_ = false;
      silent_frames_ = 0;
      return Event::kSpeechStopped;
    }
    return Event::kNone;
  }

  VadConfig config_;
  size_t frame_samples_;
  int silence_frames_;
  double energy_ = 0.0;
  size_t pending_ = 0;
  int silent_frames_ = 0;
  bool speech_ = false;
  // Starts low so a cold start does not mistake its own initial noise for a
  // high floor. Silent frames pull it toward the measured level.
  double noise_dbfs_ = -60.0;
};

}  // namespace umminfer::serve_detail
