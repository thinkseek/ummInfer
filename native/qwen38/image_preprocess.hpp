// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace umminfer::qwen38 {

struct PreprocessedImage {
  // Pre-merge patch rows: grid_t * grid_h * grid_w rows of 1536 (864 for the
  // reference image). The vision tower merger does the 2x2 merge.
  std::vector<float> pixel_values;
  int grid_t = 0, grid_h = 0, grid_w = 0;
  int resized_h = 0, resized_w = 0;
  int rows() const { return grid_t * grid_h * grid_w; }
  static constexpr int kPatchDim = 3 * 2 * 16 * 16;  // 1536
};

// Qwen2-VL/Qwen3-Omni image path: decode (RGB), smart_resize to a multiple of
// 32 within the configured pixel bounds, bicubic resize, rescale/normalize by
// 0.5, reorder into merged 16x16 temporal-2 patches.
PreprocessedImage preprocess_image(const std::string& path, int min_pixels = 65536,
                                   int max_pixels = 16777216);

// Exposed for tests: resize/normalize/reorder on an in-memory RGB image.
PreprocessedImage preprocess_rgb(const uint8_t* rgb, int src_h, int src_w,
                                 int min_pixels, int max_pixels);

// Native Qwen3.5/Qwen3.8 video preprocessor: packs 2*grid_t uniformly sampled
// RGB frames (odd counts duplicate the final frame) into temporal-2 16x16
// merged patch slices bounded by max_total_patches <= 4096.
PreprocessedImage preprocess_video_frames(
    const std::vector<std::vector<uint8_t>>& frames, int src_h, int src_w,
    int min_pixels = 32768, int max_frame_pixels = 131072,
    int max_total_patches = 3584);

// Decode + bicubic resize only; returns HxWx3 uint8-equivalent floats.
std::vector<float> resize_image_rgb(const uint8_t* rgb, int src_h, int src_w,
                                    int dst_h, int dst_w);

// Decode an image file to tightly packed HxWx3 RGB bytes.
std::vector<uint8_t> decode_image_rgb(const std::string& path, int* h, int* w);

// smart_resize from image_processing_qwen2_vl.py: factor = patch * merge = 32.
void smart_resize(int height, int width, int factor, int min_pixels, int max_pixels,
                  int* out_h, int* out_w);

}  // namespace umminfer::qwen38
