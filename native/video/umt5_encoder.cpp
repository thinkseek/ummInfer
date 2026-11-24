// SPDX-License-Identifier: Apache-2.0
#include "umt5_encoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace umminfer::video {

int umt5_relative_position_bucket(int relative_position,
                                  int num_buckets,
                                  int max_distance) {
  if (num_buckets < 2 || num_buckets % 2 != 0 || max_distance <= 0) {
    throw std::invalid_argument("invalid UMT5 relative-position configuration");
  }
  const int half = num_buckets / 2;
  const int max_exact = half / 2;
  const int64_t distance = relative_position == std::numeric_limits<int>::min()
                               ? static_cast<int64_t>(relative_position) * -1
                               : std::abs(relative_position);
  int bucket = relative_position > 0 ? half : 0;
  if (distance < max_exact) {
    return bucket + static_cast<int>(distance);
  }
  const float logarithmic = static_cast<float>(max_exact) +
      std::log(static_cast<float>(distance) / max_exact) /
          std::log(static_cast<float>(max_distance) / max_exact) *
          (half - max_exact);
  const int large_bucket = std::min(static_cast<int>(logarithmic), half - 1);
  return bucket + large_bucket;
}

#ifndef UMM_VIDEO_CUDA

struct Umt5Encoder::Impl {};

Umt5Encoder::Umt5Encoder(const std::filesystem::path&, int) {
  throw std::runtime_error("native UMT5 encoder requires UMM_VIDEO_CUDA=ON");
}
Umt5Encoder::~Umt5Encoder() = default;
Umt5EncoderOutput Umt5Encoder::encode(const Umt5TokenBatch&) {
  throw std::runtime_error("native UMT5 encoder requires UMM_VIDEO_CUDA=ON");
}
double Umt5Encoder::load_ms() const { return 0.0; }

#endif

}  // namespace umminfer::video
