// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <vector>

#include "model_config.hpp"

namespace umminfer::native {

class SafeTensors;

struct SpeakerMel {
  int frames = 0;
  std::vector<float> log_mel;  // [128, frames], channel-major
};

SpeakerMel compute_speaker_log_mel(const std::vector<float>& ref_pcm_24k);

class SpeakerEncoder {
 public:
  SpeakerEncoder(const SafeTensors& weights, const TtsModelConfig& config);
  ~SpeakerEncoder();
  SpeakerEncoder(const SpeakerEncoder&) = delete;
  SpeakerEncoder& operator=(const SpeakerEncoder&) = delete;

  std::vector<float> extract(const std::vector<float>& ref_pcm_24k) const;
  int output_dim() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace umminfer::native
