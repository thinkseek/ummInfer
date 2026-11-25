// SPDX-License-Identifier: Apache-2.0
// Prefix-reuse decision, media fingerprinting, and MRoPE continuity helpers
// for Qwen3.8-27B. Host-only and engine-free.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "image_preprocess.hpp"

namespace umminfer::qwen38 {

inline constexpr int kImagePadToken = 248056;
inline constexpr int kVideoPadToken = 248057;

inline bool is_vision_pad(int id) {
  return id == kImagePadToken || id == kVideoPadToken;
}

struct VisionRun {
  int start = 0;
  int length = 0;
};

inline std::vector<VisionRun> find_vision_runs(const std::vector<int>& ids) {
  std::vector<VisionRun> runs;
  const int n = static_cast<int>(ids.size());
  for (int i = 0; i < n;) {
    if (!is_vision_pad(ids[static_cast<size_t>(i)])) {
      ++i;
      continue;
    }
    const int begin = i;
    while (i < n && is_vision_pad(ids[static_cast<size_t>(i)])) ++i;
    runs.push_back(VisionRun{begin, i - begin});
  }
  return runs;
}

// Deterministic 64-bit fingerprint over a PreprocessedImage (grid dimensions,
// video flag, and normalized pixel float bytes). Returns 0 if and only if
// `image == nullptr`; non-null media hashes equal to 0 are remapped to 1.
inline uint64_t compute_media_hash(const PreprocessedImage* image) {
  if (image == nullptr) return 0;
  uint64_t h = 0xcbf29ce484222325ULL;  // FNV-1a 64-bit offset basis
  auto absorb_u64 = [&h](uint64_t v) {
    h ^= v;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
  };
  absorb_u64(static_cast<uint64_t>(static_cast<uint32_t>(image->grid_t)));
  absorb_u64(static_cast<uint64_t>(static_cast<uint32_t>(image->grid_h)));
  absorb_u64(static_cast<uint64_t>(static_cast<uint32_t>(image->grid_w)));
  absorb_u64(image->is_video ? 1ULL : 0ULL);
  absorb_u64(static_cast<uint64_t>(image->pixel_values.size()));

  const float* data = image->pixel_values.data();
  const size_t n = image->pixel_values.size();
  size_t i = 0;
  for (; i + 1 < n; i += 2) {
    uint64_t word = 0;
    std::memcpy(&word, data + i, sizeof(word));
    absorb_u64(word);
  }
  if (i < n) {
    uint32_t tail = 0;
    std::memcpy(&tail, data + i, sizeof(tail));
    absorb_u64(static_cast<uint64_t>(tail));
  }
  // Final avalanche (MurmurHash3 fmix64).
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53ULL;
  h ^= h >> 33;
  return h == 0 ? 1ULL : h;
}

// Whether `[0, snapshot_len)` is a safe prefix-snapshot boundary for `ids`.
// Pure-text prompts accept any snapshot_len in (0, ids.size()]. Multimodal
// prompts (`has_image == true`) require at least one vision pad in
// `[0, snapshot_len)` and zero vision pads in `[snapshot_len, ids.size())`, so
// visual runs are never cut across the snapshot boundary and any follow-up turn
// extending this prefix needs zero ViT computation for its suffix.
inline bool can_snapshot_multimodal_prefix(const std::vector<int>& ids,
                                           int snapshot_len, bool has_image) {
  const int n = static_cast<int>(ids.size());
  if (snapshot_len <= 0 || snapshot_len > n) return false;
  if (!has_image) return true;
  bool has_vision_in_prefix = false;
  for (int i = 0; i < snapshot_len; ++i) {
    if (is_vision_pad(ids[static_cast<size_t>(i)])) {
      has_vision_in_prefix = true;
      break;
    }
  }
  if (!has_vision_in_prefix) return false;
  for (int i = snapshot_len; i < n; ++i) {
    if (is_vision_pad(ids[static_cast<size_t>(i)])) return false;
  }
  return true;
}

// Number of leading prompt tokens this request may take from the snapshot the
// previous successful text-only prefill left in the engine.
//
// Reuse needs four things at once: the cache is enabled, the request is
// pure text (the vision tower writes its own rows), the new prompt extends the
// cached ids strictly, and every cached id equals the new prompt id at the same
// index. Anything else returns 0, which means a full prefill.
inline int match_prefix(const std::vector<int>& cached_ids,
                        const std::vector<int>& ids, bool has_image,
                        bool enabled) {
  if (!enabled || has_image) return 0;
  if (cached_ids.empty() || ids.size() <= cached_ids.size()) return 0;
  if (!std::equal(cached_ids.begin(), cached_ids.end(), ids.begin())) return 0;
  return static_cast<int>(cached_ids.size());
}

// Number of leading prompt tokens this request (pure-text or multimodal) may
// take from the engine's prefix snapshot.
//
// Reuse requires:
//   1. `enabled == true`;
//   2. `cached_media_hash == media_hash` (0 for pure text, non-zero matching
//      fingerprint for identical image/video);
//   3. `ids` strictly extends `cached_ids` with identical leading token ids;
//   4. When `media_hash != 0`, every image_pad/video_pad in `ids` lies strictly
//      within `[0, cached_ids.size())`, leaving zero vision pads in the suffix.
inline int match_multimodal_prefix(const std::vector<int>& cached_ids,
                                   uint64_t cached_media_hash,
                                   const std::vector<int>& ids,
                                   uint64_t media_hash, bool enabled) {
  if (!enabled) return 0;
  if (cached_media_hash != media_hash) return 0;
  if (cached_ids.empty() || ids.size() <= cached_ids.size()) return 0;
  if (!std::equal(cached_ids.begin(), cached_ids.end(), ids.begin())) return 0;
  if (media_hash != 0 &&
      !can_snapshot_multimodal_prefix(ids, static_cast<int>(cached_ids.size()),
                                      /*has_image=*/true)) {
    return 0;
  }
  return static_cast<int>(cached_ids.size());
}

// Snapshot boundary for a request that may cut its prompt at a turn boundary.
// -1, 0 and any value outside (0, prompt_len) keep the historical behaviour of
// snapshotting the whole prompt; an interior value is used as given.
inline int resolve_snapshot_len(int prompt_len, int cache_prefix_len) {
  if (cache_prefix_len <= 0 || cache_prefix_len >= prompt_len) return prompt_len;
  return cache_prefix_len;
}

// 3D MRoPE coordinates matching Qwen3_5Model.get_rope_index.
struct MRoPEPositions {
  std::vector<int> t, h, w;
  int next = 0;
};

inline MRoPEPositions build_mrope_positions(const std::vector<int>& ids,
                                            int grid_t, int grid_h, int grid_w,
                                            bool has_vision) {
  const int n = static_cast<int>(ids.size());
  MRoPEPositions p;
  p.t.resize(static_cast<size_t>(n));
  p.h.resize(static_cast<size_t>(n));
  p.w.resize(static_cast<size_t>(n));
  if (!has_vision) {
    for (int i = 0; i < n; ++i) {
      p.t[static_cast<size_t>(i)] = p.h[static_cast<size_t>(i)] =
          p.w[static_cast<size_t>(i)] = i;
    }
    p.next = n;
    return p;
  }
  const int gh = grid_h / 2;
  const int gw = grid_w / 2;
  const int slice_tokens = gh * gw;
  const std::vector<VisionRun> runs = find_vision_runs(ids);
  if (runs.empty()) {
    throw std::runtime_error("vision prompt has no image_pad/video_pad run");
  }
  if (static_cast<int>(runs.size()) != grid_t) {
    throw std::runtime_error("vision pad runs " + std::to_string(runs.size()) +
                             " do not match grid_t " + std::to_string(grid_t));
  }
  // Matches Qwen3_5Model.get_rope_index: each temporal slice is a separate
  // grid (1, grid_h, grid_w) separated by timestamp text tokens; for still
  // images (grid_t == 1) this is bit-identical to the single-image layout.
  int counter = 0;
  size_t run_idx = 0;
  for (int i = 0; i < n;) {
    if (run_idx < runs.size() && i == runs[run_idx].start) {
      const VisionRun& run = runs[run_idx];
      if (run.length != slice_tokens) {
        throw std::runtime_error("vision tokens " + std::to_string(run.length) +
                                 " do not match merged slice grid " +
                                 std::to_string(slice_tokens));
      }
      for (int k = 0; k < slice_tokens; ++k) {
        p.t[static_cast<size_t>(i + k)] = counter;
        p.h[static_cast<size_t>(i + k)] = counter + k / gw;
        p.w[static_cast<size_t>(i + k)] = counter + k % gw;
      }
      counter += std::max(gh, gw);
      i += run.length;
      ++run_idx;
    } else {
      p.t[static_cast<size_t>(i)] = p.h[static_cast<size_t>(i)] =
          p.w[static_cast<size_t>(i)] = counter++;
      ++i;
    }
  }
  p.next = counter;
  return p;
}

// MRoPE positions for a suffix `[reused, prompt_len)` continuing from a cached
// prefix whose MRoPE position counter ended at `cached_pos_next`. All tokens in
// `[reused, prompt_len)` must be text tokens (guaranteed by
// `match_multimodal_prefix`), so each suffix token advances `(t, h, w)` by +1.
inline MRoPEPositions build_suffix_mrope_positions(int prompt_len, int reused,
                                                   int cached_pos_next) {
  MRoPEPositions p;
  p.t.assign(static_cast<size_t>(prompt_len), 0);
  p.h.assign(static_cast<size_t>(prompt_len), 0);
  p.w.assign(static_cast<size_t>(prompt_len), 0);
  for (int i = reused; i < prompt_len; ++i) {
    const int pos = cached_pos_next + (i - reused);
    p.t[static_cast<size_t>(i)] = pos;
    p.h[static_cast<size_t>(i)] = pos;
    p.w[static_cast<size_t>(i)] = pos;
  }
  p.next = cached_pos_next + (prompt_len - reused);
  return p;
}

// Exact MRoPE position counter immediately after `snapshot_len` tokens when
// `[snapshot_len, prompt_len)` contains only text tokens.
inline int mrope_pos_next_at(const MRoPEPositions& p, int prompt_len,
                             int snapshot_len) {
  if (snapshot_len <= 0) return 0;
  if (snapshot_len >= prompt_len) return p.next;
  return p.t[static_cast<size_t>(snapshot_len)];
}

}  // namespace umminfer::qwen38
