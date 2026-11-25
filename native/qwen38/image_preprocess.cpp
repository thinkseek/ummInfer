// SPDX-License-Identifier: Apache-2.0
#include "image_preprocess.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_NO_THREAD_LOCALS
#include "stb_image.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace umminfer::qwen38 {
namespace {

// Torchvision antialias bicubic convolution. Verified by delta-probe against
// torch 2.11 interpolate(mode="bicubic", antialias=True): this is the Keys
// cubic with a = -0.5 (PIL's a=-0.75 BICUBIC differs; torchvision uses -0.5).
float bicubic_kernel(float x) {
  const float ax = std::fabs(x);
  if (ax <= 1.0f) return 1.5f * ax * ax * ax - 2.5f * ax * ax + 1.0f;
  if (ax < 2.0f) return -0.5f * ax * ax * ax + 2.5f * ax * ax - 4.0f * ax + 2.0f;
  return 0.0f;
}

struct Weights {
  std::vector<std::pair<int, float>> taps;  // clamped source index, weight
};

// Half-pixel coordinate transform (torchvision/PIL antialias resize).
Weights sample_weights(float in_coord, int in_size, float scale) {
  // Downsampling widens the filter support and divides the argument by scale.
  const bool down = scale > 1.0f;
  const float support = down ? 2.0f * scale : 2.0f;
  const int lo = std::ceil(in_coord - support);
  const int hi = std::floor(in_coord + support);
  Weights out;
  float sum = 0.0f;
  for (int pos = lo; pos <= hi; ++pos) {
    const float arg = down ? (in_coord - pos) / scale : (in_coord - pos);
    const float wgt = bicubic_kernel(arg);
    if (wgt == 0.0f) continue;
    out.taps.emplace_back(std::clamp(pos, 0, in_size - 1), wgt);
    sum += wgt;
  }
  for (auto& t : out.taps) t.second /= sum;
  return out;
}

std::vector<Weights> axis_weights(int out_size, int in_size) {
  const float scale = static_cast<float>(in_size) / static_cast<float>(out_size);
  std::vector<Weights> ws(out_size);
  for (int o = 0; o < out_size; ++o) {
    const float in_coord = (o + 0.5f) * scale - 0.5f;
    ws[o] = sample_weights(in_coord, in_size, scale);
  }
  return ws;
}

// Separable bicubic over an HxWx3 RGB float buffer (0..255), clamp + round
// at the end to match the uint8 PIL output boundary before /255 rescale.
std::vector<float> bicubic_resize(const std::vector<float>& src, int sh, int sw,
                                  int dh, int dw) {
  const auto hws = axis_weights(dh, sh);
  const auto wws = axis_weights(dw, sw);
  std::vector<float> tmp(static_cast<size_t>(dh) * sw * 3);
  for (int y = 0; y < dh; ++y) {
    for (int x = 0; x < sw; ++x) {
      for (int c = 0; c < 3; ++c) {
        float acc = 0.0f;
        for (const auto& [src_y, wgt] : hws[y].taps)
          acc += wgt * src[(static_cast<size_t>(src_y) * sw + x) * 3 + c];
        tmp[(static_cast<size_t>(y) * sw + x) * 3 + c] = acc;
      }
    }
  }
  std::vector<float> dst(static_cast<size_t>(dh) * dw * 3);
  for (int y = 0; y < dh; ++y) {
    for (int x = 0; x < dw; ++x) {
      for (int c = 0; c < 3; ++c) {
        float acc = 0.0f;
        for (const auto& [src_x, wgt] : wws[x].taps)
          acc += wgt * tmp[(static_cast<size_t>(y) * sw + src_x) * 3 + c];
        acc = std::clamp(acc, 0.0f, 255.0f);
        dst[(static_cast<size_t>(y) * dw + x) * 3 + c] = std::nearbyint(acc);
      }
    }
  }
  return dst;
}

}  // namespace

void smart_resize(int height, int width, int factor, int min_pixels, int max_pixels,
                  int* out_h, int* out_w) {
  if (std::max(height, width) / std::min(height, width) > 200)
    throw std::invalid_argument("image aspect ratio must be <= 200");
  // Python round() is banker's rounding (half-to-even); std::lround rounds
  // half away from zero. No current test image hits a .5 tie, but match the
  // reference policy exactly.
  auto py_round = [](double v) {
    double fl = std::floor(v);
    double frac = v - fl;
    if (frac > 0.5) return static_cast<int>(fl + 1);
    if (frac < 0.5) return static_cast<int>(fl);
    return static_cast<int>(fl) % 2 == 0 ? static_cast<int>(fl)
                                         : static_cast<int>(fl + 1);
  };
  int h_bar = py_round(static_cast<double>(height) / factor) * factor;
  int w_bar = py_round(static_cast<double>(width) / factor) * factor;
  if (static_cast<long long>(h_bar) * w_bar > max_pixels) {
    const double beta = std::sqrt(static_cast<double>(height) * width / max_pixels);
    h_bar = std::max(factor,
                     static_cast<int>(std::floor(height / beta / factor)) * factor);
    w_bar = std::max(factor,
                     static_cast<int>(std::floor(width / beta / factor)) * factor);
  } else if (static_cast<long long>(h_bar) * w_bar < min_pixels) {
    const double beta = std::sqrt(static_cast<double>(min_pixels) / (height * width));
    h_bar = static_cast<int>(std::ceil(height * beta / factor)) * factor;
    w_bar = static_cast<int>(std::ceil(width * beta / factor)) * factor;
  }
  *out_h = h_bar;
  *out_w = w_bar;
}

PreprocessedImage preprocess_rgb(const uint8_t* rgb, int src_h, int src_w,
                                 int min_pixels, int max_pixels) {
  constexpr int kPatch = 16, kMerge = 2, kTemporal = 2, kFactor = 32;
  int dh = 0, dw = 0;
  smart_resize(src_h, src_w, kFactor, min_pixels, max_pixels, &dh, &dw);
  if (dh % kPatch || dw % kPatch)
    throw std::runtime_error("smart_resize output not divisible by patch size");

  std::vector<float> src(static_cast<size_t>(src_h) * src_w * 3);
  for (size_t i = 0; i < src.size(); ++i) src[i] = rgb[i];
  const std::vector<float> resized = bicubic_resize(src, src_h, src_w, dh, dw);

  const int grid_h = dh / kPatch, grid_w = dw / kPatch, grid_t = 1;
  const int gh2 = grid_h / kMerge, gw2 = grid_w / kMerge;
  PreprocessedImage out;
  out.grid_t = grid_t;
  out.grid_h = grid_h;
  out.grid_w = grid_w;
  out.resized_h = dh;
  out.resized_w = dw;
  out.is_video = false;
  // pixel_values keeps the full pre-merge patch grid: grid_t*grid_h*grid_w
  // rows (864 for the reference image). The 2x2 merge happens in the vision
  // tower merger, not in the preprocessor.
  out.pixel_values.resize(static_cast<size_t>(grid_t) * grid_h * grid_w *
                          PreprocessedImage::kPatchDim);

  // Match image_processing_qwen2_vl view+permute. Row order after permute:
  // (t, gh2, gw2, merge_h, merge_w); within a row: (c, temporal, py, px).
  // The temporal dim is a repeat of the single still frame, so both ts values
  // read the same source pixel.
  for (int t = 0; t < grid_t; ++t) {
    for (int gh = 0; gh < gh2; ++gh) {
      for (int gw = 0; gw < gw2; ++gw) {
        for (int msh = 0; msh < kMerge; ++msh) {
          for (int msw = 0; msw < kMerge; ++msw) {
            const int row = (((t * gh2 + gh) * gw2 + gw) * kMerge + msh) * kMerge + msw;
            for (int c = 0; c < 3; ++c) {
              for (int ts = 0; ts < kTemporal; ++ts) {
                for (int py = 0; py < kPatch; ++py) {
                  for (int px = 0; px < kPatch; ++px) {
                    const int y = gh * kMerge * kPatch + msh * kPatch + py;
                    const int x = gw * kMerge * kPatch + msw * kPatch + px;
                    const int fidx = (((c * kTemporal + ts) * kPatch + py) * kPatch + px);
                    const float v = resized[(static_cast<size_t>(y) * dw + x) * 3 + c];
                    // (x/255 - 0.5) / 0.5 == x/127.5 - 1.
                    out.pixel_values[static_cast<size_t>(row) *
                                         PreprocessedImage::kPatchDim +
                                     fidx] = v / 127.5f - 1.0f;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return out;
}

PreprocessedImage preprocess_video_frames(
    const std::vector<std::vector<uint8_t>>& frames, int src_h, int src_w,
    int min_pixels, int max_frame_pixels, int max_total_patches) {
  constexpr int kPatch = 16, kMerge = 2, kTemporal = 2, kFactor = 32;
  if (frames.empty() || src_h <= 0 || src_w <= 0) {
    throw std::invalid_argument("preprocess_video_frames requires non-empty frames and positive dimensions");
  }
  const size_t expected_bytes = static_cast<size_t>(src_h) * src_w * 3;
  for (const auto& f : frames) {
    if (f.size() != expected_bytes) {
      throw std::invalid_argument("video frame RGB buffer size mismatch");
    }
  }
  const int n_even = static_cast<int>((frames.size() + 1) / 2) * 2;
  const int grid_t = n_even / kTemporal;
  if (grid_t <= 0) {
    throw std::invalid_argument("video frame count must produce at least 1 temporal slice");
  }
  const int per_slice_patches = std::max(4, max_total_patches / grid_t);
  const int effective_max_pixels =
      std::min(max_frame_pixels, per_slice_patches * kPatch * kPatch);
  const int effective_min_pixels = std::min(min_pixels, effective_max_pixels);
  int dh = 0, dw = 0;
  smart_resize(src_h, src_w, kFactor, effective_min_pixels,
               effective_max_pixels, &dh, &dw);
  if (dh % kPatch || dw % kPatch) {
    throw std::runtime_error("smart_resize video output not divisible by patch size");
  }

  const int grid_h = dh / kPatch, grid_w = dw / kPatch;
  const int gh2 = grid_h / kMerge, gw2 = grid_w / kMerge;
  const bool exact_size = (src_h == dh && src_w == dw);
  std::vector<std::vector<float>> resized_frames;
  if (!exact_size) {
    resized_frames.reserve(static_cast<size_t>(n_even));
    std::vector<float> src(expected_bytes);
    for (int i = 0; i < n_even; ++i) {
      const auto& raw =
          frames[std::min<size_t>(static_cast<size_t>(i), frames.size() - 1)];
      for (size_t k = 0; k < expected_bytes; ++k) src[k] = raw[k];
      resized_frames.push_back(bicubic_resize(src, src_h, src_w, dh, dw));
    }
  }

  PreprocessedImage out;
  out.grid_t = grid_t;
  out.grid_h = grid_h;
  out.grid_w = grid_w;
  out.resized_h = dh;
  out.resized_w = dw;
  out.is_video = true;
  out.pixel_values.resize(static_cast<size_t>(grid_t) * grid_h * grid_w *
                          PreprocessedImage::kPatchDim);

  auto pack_slice = [&](int t, auto read_pixel) {
    for (int gh = 0; gh < gh2; ++gh) {
      for (int gw = 0; gw < gw2; ++gw) {
        for (int msh = 0; msh < kMerge; ++msh) {
          for (int msw = 0; msw < kMerge; ++msw) {
            const int row =
                (((t * gh2 + gh) * gw2 + gw) * kMerge + msh) * kMerge + msw;
            for (int c = 0; c < 3; ++c) {
              for (int ts = 0; ts < kTemporal; ++ts) {
                for (int py = 0; py < kPatch; ++py) {
                  for (int px = 0; px < kPatch; ++px) {
                    const int y = gh * kMerge * kPatch + msh * kPatch + py;
                    const int x = gw * kMerge * kPatch + msw * kPatch + px;
                    const int fidx =
                        (((c * kTemporal + ts) * kPatch + py) * kPatch + px);
                    const float v =
                        read_pixel(ts, (static_cast<size_t>(y) * dw + x) * 3 + c);
                    out.pixel_values[static_cast<size_t>(row) *
                                         PreprocessedImage::kPatchDim +
                                     fidx] = v / 127.5f - 1.0f;
                  }
                }
              }
            }
          }
        }
      }
    }
  };

  for (int t = 0; t < grid_t; ++t) {
    if (exact_size) {
      const auto& f0 =
          frames[std::min<size_t>(static_cast<size_t>(t * 2), frames.size() - 1)];
      const auto& f1 = frames[std::min<size_t>(static_cast<size_t>(t * 2 + 1),
                                               frames.size() - 1)];
      pack_slice(t, [&](int ts, size_t idx) -> float {
        return static_cast<float>((ts == 0 ? f0 : f1)[idx]);
      });
    } else {
      const std::vector<float>& f0 = resized_frames[static_cast<size_t>(t * 2)];
      const std::vector<float>& f1 = resized_frames[static_cast<size_t>(t * 2 + 1)];
      pack_slice(t, [&](int ts, size_t idx) -> float {
        return (ts == 0 ? f0 : f1)[idx];
      });
    }
  }
  return out;
}

PreprocessedImage preprocess_image(const std::string& path, int min_pixels,
                                   int max_pixels) {
  int w = 0, h = 0;
  std::vector<uint8_t> rgb = decode_image_rgb(path, &h, &w);
  return preprocess_rgb(rgb.data(), h, w, min_pixels, max_pixels);
}

std::vector<uint8_t> decode_image_rgb(const std::string& path, int* h, int* w) {
  int comp = 0;
  stbi_uc* pixels = stbi_load(path.c_str(), w, h, &comp, 3);
  if (!pixels)
    throw std::invalid_argument(std::string("stbi_image failed: ") + stbi_failure_reason() +
                                " for " + path);
  std::vector<uint8_t> out(pixels, pixels + static_cast<size_t>((*h) * (*w) * 3));
  stbi_image_free(pixels);
  return out;
}

std::vector<float> resize_image_rgb(const uint8_t* rgb, int src_h, int src_w,
                                    int dst_h, int dst_w) {
  std::vector<float> src(static_cast<size_t>(src_h) * src_w * 3);
  for (size_t i = 0; i < src.size(); ++i) src[i] = rgb[i];
  return bicubic_resize(src, src_h, src_w, dst_h, dst_w);
}

}  // namespace umminfer::qwen38
