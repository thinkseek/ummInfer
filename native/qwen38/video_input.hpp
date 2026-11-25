// SPDX-License-Identifier: Apache-2.0
// Bounded FFmpeg decode and temporal keyframe preprocessing for Qwen3.8.
#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

#include "image_preprocess.hpp"
#include "video_sampling.hpp"

namespace umminfer::qwen38 {

struct DecodedVideo {
  PreprocessedImage preprocessed;
  // One timestamp per native temporal-2 slice: the first selected source frame
  // in that pair. This feeds the existing video prompt and 3D MRoPE path.
  std::vector<double> timestamps;
  // Host-test/live diagnostics. Indices refer to the bounded decoder output,
  // not to timestamps sorted in time.
  std::vector<size_t> selected_frame_indices;
  std::vector<double> selected_frame_timestamps;
  size_t decoded_frame_count = 0;
};

// Applies temporal selection to an already decoded, bounded candidate list.
// Empty lists, malformed RGB buffers, and non-finite timestamps are rejected;
// non-monotonic finite timestamps are retained in decode order.
DecodedVideo preprocess_temporal_candidates(
    const std::vector<VideoCandidate>& candidates, int src_h, int src_w,
    const VideoFrameSelection& options);

// Decodes a data-video payload already accepted by the serving 32 MiB contract.
// Default/explicit uniform preserves the historical FFmpeg filter, 8-frame
// limit, synthetic slice timestamps, and preprocessing bytes. Temporal opt-in
// alone uses the bounded candidate decoder and source presentation timestamps.
DecodedVideo decode_video_bytes(std::string_view bytes,
                                const VideoFrameSelection& options = {});

}  // namespace umminfer::qwen38
