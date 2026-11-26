// SPDX-License-Identifier: Apache-2.0
#include "video_sampling.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace umminfer::qwen38 {
namespace {

constexpr int kAnalysisSide = 16;
constexpr int kAnalysisCells = kAnalysisSide * kAnalysisSide;
constexpr int kHistogramBins = 16;

struct AnalysisFrame {
  std::array<uint8_t, kAnalysisCells> luma{};
  std::array<uint16_t, kHistogramBins> histogram{};
};

AnalysisFrame analyze_frame(const VideoCandidate& candidate, int src_h, int src_w) {
  AnalysisFrame out;
  for (int ay = 0; ay < kAnalysisSide; ++ay) {
    const int y0 = ay * src_h / kAnalysisSide;
    const int y1 = (ay + 1) * src_h / kAnalysisSide;
    for (int ax = 0; ax < kAnalysisSide; ++ax) {
      const int x0 = ax * src_w / kAnalysisSide;
      const int x1 = (ax + 1) * src_w / kAnalysisSide;
      uint64_t sum = 0;
      uint64_t count = 0;
      for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
          const size_t offset = (static_cast<size_t>(y) * src_w + x) * 3;
          // Integer Rec.601 luma is deterministic on every host and avoids a
          // floating-point accumulator over source-sized frames.
          const uint8_t luma = static_cast<uint8_t>(
              (77u * candidate.rgb[offset] + 150u * candidate.rgb[offset + 1] +
               29u * candidate.rgb[offset + 2]) >>
              8);
          sum += luma;
          ++count;
        }
      }
      const uint8_t mean = static_cast<uint8_t>(sum / count);
      out.luma[static_cast<size_t>(ay * kAnalysisSide + ax)] = mean;
      ++out.histogram[static_cast<size_t>(mean) * kHistogramBins / 256];
    }
  }
  return out;
}

bool higher_motion(const TemporalFrameScore& a, const TemporalFrameScore& b) {
  if (a.motion != b.motion) return a.motion > b.motion;
  if (a.scene != b.scene) return a.scene > b.scene;
  return false;  // std::sort comparator below adds the earliest-index tie-break.
}

bool higher_scene(const TemporalFrameScore& a, const TemporalFrameScore& b) {
  if (a.scene != b.scene) return a.scene > b.scene;
  if (a.motion != b.motion) return a.motion > b.motion;
  return false;
}

void stable_score_sort(std::vector<size_t>* indices,
                       const std::vector<TemporalFrameScore>& scores,
                       bool scene_primary) {
  std::sort(indices->begin(), indices->end(), [&](size_t a, size_t b) {
    if (scene_primary) {
      if (higher_scene(scores[a], scores[b])) return true;
      if (higher_scene(scores[b], scores[a])) return false;
    } else {
      if (higher_motion(scores[a], scores[b])) return true;
      if (higher_motion(scores[b], scores[a])) return false;
    }
    return a < b;
  });
}

}  // namespace

void validate_video_frame_selection(const VideoFrameSelection& options) {
  if (options.policy != VideoFramePolicy::kUniform &&
      options.policy != VideoFramePolicy::kTemporal) {
    throw std::invalid_argument("unknown video frame selection policy");
  }
  if (!std::isfinite(options.motion_threshold) ||
      options.motion_threshold < 0.0 || options.motion_threshold > 1.0) {
    throw std::invalid_argument("video motion_threshold must be finite and in [0,1]");
  }
  if (!std::isfinite(options.scene_threshold) || options.scene_threshold < 0.0 ||
      options.scene_threshold > 1.0) {
    throw std::invalid_argument("video scene_threshold must be finite and in [0,1]");
  }
  if (options.max_frames != 4 && options.max_frames != 8 &&
      options.max_frames != kVideoMaxFrameBudget) {
    throw std::invalid_argument("video max_frames must be exactly 4, 8, or 16");
  }
  if (options.max_candidate_frames <= 0 ||
      options.max_candidate_frames > kVideoCandidateFrameBudget) {
    throw std::invalid_argument("video max_candidate_frames exceeds the 64-frame budget");
  }
  if (options.max_frames > options.max_candidate_frames) {
    throw std::invalid_argument("video max_frames cannot exceed max_candidate_frames");
  }
}

std::vector<TemporalFrameScore> analyze_temporal_frames(
    const std::vector<VideoCandidate>& candidates, int src_h, int src_w) {
  if (candidates.empty() || src_h < kAnalysisSide || src_w < kAnalysisSide) {
    throw std::invalid_argument(
        "temporal analysis requires candidates and dimensions of at least 16x16");
  }
  const size_t expected = static_cast<size_t>(src_h) * src_w * 3;
  std::vector<AnalysisFrame> analyzed;
  analyzed.reserve(candidates.size());
  for (const VideoCandidate& candidate : candidates) {
    if (candidate.rgb.size() != expected) {
      throw std::invalid_argument("video candidate RGB buffer size mismatch");
    }
    if (!std::isfinite(candidate.timestamp_seconds)) {
      throw std::invalid_argument("video candidate timestamp must be finite");
    }
    analyzed.push_back(analyze_frame(candidate, src_h, src_w));
  }

  std::vector<TemporalFrameScore> scores(candidates.size());
  for (size_t i = 1; i < candidates.size(); ++i) {
    uint64_t motion_sum = 0;
    uint64_t histogram_sum = 0;
    for (int cell = 0; cell < kAnalysisCells; ++cell) {
      motion_sum += std::abs(
          static_cast<int>(analyzed[i].luma[static_cast<size_t>(cell)]) -
          static_cast<int>(analyzed[i - 1].luma[static_cast<size_t>(cell)]));
    }
    for (int bin = 0; bin < kHistogramBins; ++bin) {
      histogram_sum += std::abs(
          static_cast<int>(analyzed[i].histogram[static_cast<size_t>(bin)]) -
          static_cast<int>(analyzed[i - 1].histogram[static_cast<size_t>(bin)]));
    }
    scores[i].motion = static_cast<double>(motion_sum) /
                       (255.0 * static_cast<double>(kAnalysisCells));
    scores[i].scene = static_cast<double>(histogram_sum) /
                      (2.0 * static_cast<double>(kAnalysisCells));
  }
  return scores;
}

std::vector<size_t> select_temporal_keyframes(
    const std::vector<TemporalFrameScore>& scores,
    const VideoFrameSelection& options) {
  validate_video_frame_selection(options);
  if (scores.empty()) return {};
  const size_t count = scores.size();
  const size_t target = std::min(count, static_cast<size_t>(options.max_frames));
  if (count <= target) {
    std::vector<size_t> all(count);
    for (size_t i = 0; i < count; ++i) all[i] = i;
    return all;
  }

  std::vector<bool> selected(count, false);
  std::vector<size_t> out;
  out.reserve(target);
  auto select = [&](size_t index) {
    if (out.size() >= target || selected[index]) return;
    selected[index] = true;
    out.push_back(index);
  };
  select(0);
  select(count - 1);

  // A run of adjacent detections is one boundary. This prevents repeated
  // decoder/scene notifications from consuming several keyframe slots. Genuine
  // high-motion members of the run remain eligible in the motion pass below.
  std::vector<size_t> scene_representatives;
  for (size_t i = 1; i < count;) {
    if (!(scores[i].scene > options.scene_threshold)) {
      ++i;
      continue;
    }
    size_t end = i + 1;
    size_t best = i;
    while (end < count && scores[end].scene > options.scene_threshold) {
      if (higher_scene(scores[end], scores[best]) ||
          (!higher_scene(scores[best], scores[end]) && end < best)) {
        best = end;
      }
      ++end;
    }
    scene_representatives.push_back(best);
    i = end;
  }
  stable_score_sort(&scene_representatives, scores, /*scene_primary=*/true);
  for (size_t index : scene_representatives) select(index);

  std::vector<size_t> motion_peaks;
  for (size_t i = 1; i < count;) {
    // Strict comparison makes repeated frames (score zero) ineligible even at
    // a zero threshold. Coalesce each contiguous above-threshold motion run to
    // one strongest representative so one event cannot consume several slots.
    if (!(scores[i].motion > options.motion_threshold)) {
      ++i;
      continue;
    }
    size_t end = i + 1;
    size_t best = i;
    while (end < count && scores[end].motion > options.motion_threshold) {
      if (higher_motion(scores[end], scores[best]) ||
          (!higher_motion(scores[best], scores[end]) && end < best)) {
        best = end;
      }
      ++end;
    }
    motion_peaks.push_back(best);
    i = end;
  }
  stable_score_sort(&motion_peaks, scores, /*scene_primary=*/false);
  for (size_t index : motion_peaks) select(index);

  // Uniform fill covers static videos and clips whose scores never cross the
  // thresholds. For count >= target these rounded positions are unique.
  for (size_t slot = 0; slot < target && out.size() < target; ++slot) {
    const size_t index = target > 1
                             ? (slot * (count - 1) + (target - 1) / 2) / (target - 1)
                             : 0;
    select(index);
  }
  // Priority selections can collide with the uniform positions. Fill any such
  // holes by earliest source position, never by dropping below the valid bound.
  for (size_t i = 0; i < count && out.size() < target; ++i) select(i);

  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace umminfer::qwen38
