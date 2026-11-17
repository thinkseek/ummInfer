// SPDX-License-Identifier: Apache-2.0
// Debug tool: write the C++ image preprocessing result as raw float32 and
// print the patch grid, for comparison with the HF image processor.
#include <cstdio>
#include <fstream>
#include <iostream>

#include "qwen38/image_preprocess.hpp"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: qwen38_preprocess IMAGE OUT.f32\n";
    return 2;
  }
  const umminfer::qwen38::PreprocessedImage img =
      umminfer::qwen38::preprocess_image(argv[1]);
  std::ofstream f(argv[2], std::ios::binary);
  if (!f) {
    std::cerr << "cannot write " << argv[2] << "\n";
    return 1;
  }
  f.write(reinterpret_cast<const char*>(img.pixel_values.data()),
          static_cast<std::streamsize>(img.pixel_values.size() * sizeof(float)));
  std::printf("grid %d %d %d rows %d dim %zu\n", img.grid_t, img.grid_h,
              img.grid_w, img.rows(), img.pixel_values.size() / img.rows());
  return 0;
}
