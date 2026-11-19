// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace umminfer::omni {

// Image capacity of the thinker, one source of truth for three consumers:
// smart_resize clamps the resized area to kMaxImagePixels, the vision tower
// scratch spans kMaxImagePatchRows, and the decoder KV holds kMaxImageTokens
// image pads. A 16x16 patch and a 2x2 merge fix the ratios: rows = pixels/256
// and pads = pixels/1024. 1024x1024 leaves half of the default 2048-token
// context for text and generated tokens. Before this bound existed, the
// preprocessor accepted 12845056 pixels while the scratch held 1296 patch
// rows (576x576), so a 640-pixel-tall portrait image failed deterministically.
constexpr int kMaxImagePixels = 1024 * 1024;
constexpr int kMaxImagePatchRows = kMaxImagePixels / (16 * 16);
constexpr int kMaxImageTokens = kMaxImagePixels / (32 * 32);

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
PreprocessedImage preprocess_image(const std::string& path, int min_pixels = 3136,
                                   int max_pixels = kMaxImagePixels);

// Exposed for tests: resize/normalize/reorder on an in-memory RGB image.
PreprocessedImage preprocess_rgb(const uint8_t* rgb, int src_h, int src_w,
                                 int min_pixels, int max_pixels);

// Decode + bicubic resize only; returns HxWx3 uint8-equivalent floats.
std::vector<float> resize_image_rgb(const uint8_t* rgb, int src_h, int src_w,
                                    int dst_h, int dst_w);

// Decode an image file to tightly packed HxWx3 RGB bytes.
std::vector<uint8_t> decode_image_rgb(const std::string& path, int* h, int* w);

// smart_resize from image_processing_qwen2_vl.py: factor = patch * merge = 32.
void smart_resize(int height, int width, int factor, int min_pixels, int max_pixels,
                  int* out_h, int* out_w);

}  // namespace umminfer::omni
