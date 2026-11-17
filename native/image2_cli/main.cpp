// SPDX-License-Identifier: Apache-2.0
// umm-image2-native: native C++/CUDA Qwen-Image 2.1 text-to-image.
#include "image2/engine.hpp"
#include "image2/png.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

void usage() {
  std::fprintf(stderr,
               "usage: umm-image2-native --model DIR [--gpu N] --prompt TEXT\n"
               "       [--size WxH] [--seed S] [--steps K] --out PATH\n");
}

int parse_int(const char* s, const char* what) {
  char* end = nullptr;
  long v = std::strtol(s, &end, 10);
  if (!end || *end) throw std::invalid_argument(std::string("bad ") + what);
  return static_cast<int>(v);
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, prompt, out;
  int gpu = 0, seed = 0, steps = 40, width = 512, height = 512;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> const char* {
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
        return argv[++i];
      };
      if (a == "--model") model = next();
      else if (a == "--prompt") prompt = next();
      else if (a == "--out") out = next();
      else if (a == "--gpu") gpu = parse_int(next(), "gpu");
      else if (a == "--seed") seed = parse_int(next(), "seed");
      else if (a == "--steps") steps = parse_int(next(), "steps");
      else if (a == "--size") {
        const char* s = next();
        const char* x = std::strchr(s, 'x');
        if (!x) throw std::invalid_argument("--size expects WxH");
        width = parse_int(std::string(s, x - s).c_str(), "width");
        height = parse_int(x + 1, "height");
      } else if (a == "--help" || a == "-h") {
        usage();
        return 0;
      } else {
        throw std::invalid_argument("unknown argument " + a);
      }
    }
    if (model.empty() || prompt.empty() || out.empty()) {
      usage();
      return 2;
    }

    umminfer::image2::Image2Engine engine(gpu, model);
    umminfer::image2::GenerateParams p;
    p.prompt = prompt;
    p.seed = static_cast<uint64_t>(seed);
    p.steps = steps;
    p.width = width;
    p.height = height;
    auto r = engine.generate(p);

    const std::string png = umminfer::image2::encode_png(r.image);
    std::ofstream f(out, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + out);
    f.write(png.data(), static_cast<std::streamsize>(png.size()));

    std::printf(
        "image: %dx%d\nload_ms: %d\nencode_ms: %d\ndenoise_ms: %d\nvae_ms: %d\n"
        "total_ms: %d\npeak_gpu_MiB: %.1f\nout: %s\n",
        r.image.width, r.image.height, r.load_ms, r.encode_ms, r.denoise_ms, r.vae_ms,
        r.total_ms, static_cast<double>(r.peak_gpu_bytes) / (1024.0 * 1024.0), out.c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
