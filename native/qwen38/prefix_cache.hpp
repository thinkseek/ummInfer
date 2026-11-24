// SPDX-License-Identifier: Apache-2.0
// Prefix-reuse decision for Qwen3.8-27B. Host-only and engine-free.
#pragma once

#include <algorithm>
#include <vector>

namespace umminfer::qwen38 {

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

// Snapshot boundary for a request that may cut its prompt at a turn boundary.
// -1, 0 and any value outside (0, prompt_len) keep the historical behaviour of
// snapshotting the whole prompt; an interior value is used as given.
inline int resolve_snapshot_len(int prompt_len, int cache_prefix_len) {
  if (cache_prefix_len <= 0 || cache_prefix_len >= prompt_len) return prompt_len;
  return cache_prefix_len;
}

}  // namespace umminfer::qwen38
