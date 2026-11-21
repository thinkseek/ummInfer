// SPDX-License-Identifier: Apache-2.0
// umm-image-native: Qwen-Image-2512 text-to-image, no Python at runtime.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include "image/engine.hpp"
#include "image/png.hpp"

namespace {
struct Args {
  std::string model, prompt, out;
  uint64_t seed = 0;
  int steps = 50, width = 512, height = 512;
};

void usage() {
  std::fprintf(stderr,
               "usage: umm-image-native --model DIR --prompt TEXT --out PNG "
               "[--seed N] [--steps N] [--size WxH]\n");
}
}  // namespace

int main(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto need = [&](std::string& dst) {
      if (i + 1 >= argc) { usage(); std::exit(2); }
      dst = argv[++i];
    };
    if (k == "--model") need(a.model);
    else if (k == "--prompt") need(a.prompt);
    else if (k == "--out") need(a.out);
    else if (k == "--seed") a.seed = std::stoull(argv[++i]);
    else if (k == "--steps") a.steps = std::stoi(argv[++i]);
    else if (k == "--size") {
      std::string sz = argv[++i];
      auto x = sz.find('x');
      if (x == std::string::npos) { usage(); return 2; }
      a.width = std::stoi(sz.substr(0, x));
      a.height = std::stoi(sz.substr(x + 1));
    } else { usage(); return 2; }
  }
  if (a.model.empty() || a.prompt.empty() || a.out.empty()) { usage(); return 2; }

  try {
    umminfer::image::ImageEngine engine(0, a.model);
    umminfer::image::GenerateParams gp;
    gp.prompt = a.prompt; gp.seed = a.seed; gp.steps = a.steps;
    gp.width = a.width; gp.height = a.height;
    auto r = engine.generate(gp);
    umminfer::image::RgbImage img(r.width, r.height);
    for (size_t i = 0; i < r.rgb_u16.size(); ++i) img.rgb[i] = static_cast<uint8_t>(r.rgb_u16[i]);
    std::string png = umminfer::image::encode_png(img);
    std::ofstream f(a.out, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot write %s\n", a.out.c_str()); return 2; }
    f.write(png.data(), static_cast<std::streamsize>(png.size()));
    std::fprintf(stderr,
                 "size=%dx%d steps=%d seed=%llu load_ms=%d encode_ms=%d denoise_ms=%d "
                 "vae_ms=%d total_ms=%d gpu_used_mb=%zu\n",
                 r.width, r.height, a.steps, (unsigned long long)a.seed, r.load_ms,
                 r.encode_ms, r.denoise_ms, r.vae_ms, r.total_ms,
                 r.peak_gpu_bytes / (1024 * 1024));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
