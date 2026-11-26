// SPDX-License-Identifier: Apache-2.0
// Deterministic, bounded keyframe selection for native Qwen3.8 video input.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace umminfer::qwen38 {

// These are hard ceilings, not allocation hints. The public API cannot raise
// them. Temporal mode may decode more candidates than it selects, but both the
// candidate RGB allocation and the final temporal-2 patch allocation are capped.
inline constexpr int kVideoFrameBudget = 8;  // Unchanged default.
inline constexpr int kVideoMaxFrameBudget = 16;
inline constexpr int kVideoCandidateFrameBudget = 64;
inline constexpr double kDefaultVideoMotionThreshold = 0.02;
inline constexpr double kDefaultVideoSceneThreshold = 0.25;

enum class VideoFramePolicy { kUniform, kTemporal };

struct VideoFrameSelection {
  VideoFramePolicy policy = VideoFramePolicy::kUniform;
  double motion_threshold = kDefaultVideoMotionThreshold;
  double scene_threshold = kDefaultVideoSceneThreshold;
  // The public explicit selected-frame budget is exactly 4, 8, or 16. Omission
  // retains the historical 8-frame path. Unsupported values are rejected before
  // any decode output buffer is allocated; max_candidate_frames stays a
  // separately bounded native/test setting and is not public API.
  int max_frames = kVideoFrameBudget;
  int max_candidate_frames = kVideoCandidateFrameBudget;
};

struct VideoCandidate {
  std::vector<uint8_t> rgb;
  // Original presentation timestamp in seconds. Decode order, not this value,
  // defines chronology. Non-monotonic finite timestamps are preserved.
  double timestamp_seconds = 0.0;
  // Zero-based source frame number when container metadata makes the exact
  // select(n) mapping known; -1 for the nominal-timestamp fallback.
  int64_t source_frame_index = -1;
};

struct TemporalFrameScore {
  // Mean absolute luma difference from the preceding decoded candidate, [0,1].
  double motion = 0.0;
  // Normalized L1 distance between 16-bin luma histograms, [0,1].
  double scene = 0.0;
};

void validate_video_frame_selection(const VideoFrameSelection& options);

// Computes deterministic scores on a fixed 16x16 luma analysis grid. All
// candidates must have the same tightly packed RGB dimensions.
std::vector<TemporalFrameScore> analyze_temporal_frames(
    const std::vector<VideoCandidate>& candidates, int src_h, int src_w);

// Selects at most options.max_frames candidate positions. The first and last
// candidates are coverage anchors. Adjacent scene detections count as one
// boundary; the strongest detection represents the run (motion, then earliest
// index, breaks ties). Scene representatives rank before other motion peaks.
// Unused slots are filled by deterministic uniform coverage, so static clips,
// repeated frames, high thresholds, and short clips always have a valid result.
// The returned positions are unique and strictly increasing in decode order.
std::vector<size_t> select_temporal_keyframes(
    const std::vector<TemporalFrameScore>& scores,
    const VideoFrameSelection& options);

}  // namespace umminfer::qwen38
