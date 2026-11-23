// SPDX-License-Identifier: Apache-2.0
#include "mrope.hpp"

#include <algorithm>
#include <stdexcept>

namespace umminfer::omni {

PositionIds build_positions(const std::vector<int>& ids, int image_pad_id,
                            int vision_start_id, int vision_end_id,
                            int grid_h, int grid_w, bool has_image) {
  (void)vision_start_id;
  (void)vision_end_id;
  const int n = static_cast<int>(ids.size());
  PositionIds p{std::vector<int>(n), std::vector<int>(n), std::vector<int>(n), 0};
  if (!has_image) {
    for (int i = 0; i < n; ++i) p.t[i] = p.h[i] = p.w[i] = i;
    p.next_pos = n;
    return p;
  }
  const int img_start =
      std::find(ids.begin(), ids.end(), image_pad_id) - ids.begin();
  int img_end = img_start;
  while (img_end < n && ids[img_end] == image_pad_id) ++img_end;
  const int n_img = img_end - img_start;
  const int llm_gh = grid_h / 2, llm_gw = grid_w / 2;
  if (n_img != llm_gh * llm_gw)
    throw std::runtime_error("image token count does not match grid");
  int counter = 0;
  for (int i = 0; i < img_start; ++i) p.t[i] = p.h[i] = p.w[i] = counter++;
  // vision_start is the token immediately before the pads and already counted.
  const int st_idx = counter;  // first image position
  for (int k = 0; k < n_img; ++k) {
    p.t[img_start + k] = st_idx;
    p.h[img_start + k] = st_idx + k / llm_gw;
    p.w[img_start + k] = st_idx + k % llm_gw;
  }
  int grid_max = st_idx + std::max(llm_gh - 1, llm_gw - 1);
  counter = grid_max + 1;
  for (int i = img_end; i < n; ++i) p.t[i] = p.h[i] = p.w[i] = counter++;
  p.next_pos = counter;
  return p;
}

}  // namespace umminfer::omni
