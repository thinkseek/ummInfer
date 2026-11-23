// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>

namespace umminfer::omni {

// Per-token mRoPE grids for the thinker prompt.
struct PositionIds {
  std::vector<int> t, h, w;
  int next_pos;
};

// Build [3][n] position ids for the full prompt. The image region (if any) is
// the contiguous run of image_pad tokens; grid gives its (grid_h, grid_w).
// Host-only so the host test can check it without CUDA.
PositionIds build_positions(const std::vector<int>& ids, int image_pad_id,
                            int vision_start_id, int vision_end_id,
                            int grid_h, int grid_w, bool has_image);

}  // namespace umminfer::omni
