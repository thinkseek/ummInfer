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

}  // namespace umminfer::qwen38
