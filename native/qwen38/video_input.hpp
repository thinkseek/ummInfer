// SPDX-License-Identifier: Apache-2.0
// Bounded FFmpeg decode and temporal keyframe preprocessing for Qwen3.8.
#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "image_preprocess.hpp"
#include "video_sampling.hpp"

namespace umminfer::qwen38 {

struct VideoDecodeProfile {
  double input_write_ms = 0.0;
  double probe_ms = 0.0;
  double ffmpeg_decode_rgb_ms = 0.0;
  double rgb_split_ms = 0.0;
  double content_fingerprint_ms = 0.0;
  double temporal_analysis_ms = 0.0;
  double frame_selection_ms = 0.0;
  double selection_copy_ms = 0.0;
  double temporal_pack_ms = 0.0;
  double total_ms = 0.0;
};

struct DecodedVideo {
  PreprocessedImage preprocessed;
  // One timestamp per native temporal-2 slice: the first selected source frame
  // in that pair. This feeds the existing video prompt and 3D MRoPE path.
  std::vector<double> timestamps;
  // Host-test/live diagnostics. Candidate indices refer to the bounded decoder
  // output, not to timestamps sorted in time. Source indices are exact only
  // when source_indices_exact is true; otherwise they are candidate positions
  // accompanying nominal timestamps.
  std::vector<size_t> selected_frame_indices;
  std::vector<int64_t> selected_source_frame_indices;
  std::vector<double> selected_frame_timestamps;
  // Non-cryptographic FNV-1a fingerprints of the exact scaled RGB24 bytes fed
  // to temporal-2 preprocessing. They let an external fixture harness bind the
  // native selection to independently decoded source content without changing
  // the historical uniform decode or exposing frame bytes in the response.
  std::vector<std::string> selected_frame_content_fnv1a64;
  bool source_indices_exact = false;
  bool has_nominal_timestamps = true;
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

// Diagnostic-only variant. The production entry point above passes no profile
// and performs no timing-clock reads. Stage boundaries match the implementation;
// FFmpeg demux/decode/RGB conversion/scaling share one child-process boundary.
DecodedVideo decode_video_bytes_profiled(std::string_view bytes,
                                         const VideoFrameSelection& options,
                                         VideoDecodeProfile* profile);

}  // namespace umminfer::qwen38
