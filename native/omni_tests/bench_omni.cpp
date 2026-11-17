// SPDX-License-Identifier: Apache-2.0
// Repeatable benchmark for the native Qwen3-Omni thinker slice.
//
// Fixed inputs: the accepted real image (/tmp/man_ironing.png, SHA256
// e0691718...060e, 570x380) and the accepted prompt string below.  Both cases
// are the ones the slice was accepted on, so the answers can be compared
// verbatim against the reference run.
//
//   umm-omni-bench --model DIR [--image PATH] [--gpu N]
//
// Prints one line per case with the phase timings and the answer, then the
// resident device memory.  Build with -DUMM_OMNI_CUDA=ON.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "omni/image_preprocess.hpp"
#include "omni/omni_engine.hpp"
#include "omni/tokenizer.hpp"

namespace {

constexpr const char* kPrompt =
    "What is the person in this image doing? Answer in one short sentence.";
constexpr const char* kDefaultImage = "/tmp/man_ironing.png";
constexpr int kMaxNewTokens = 24;

struct Case {
  const char* name;
  bool with_image;
};

}  // namespace

int main(int argc, char** argv) {
  using umminfer::omni::OmniEngine;
  using umminfer::omni::PreprocessedImage;
  using umminfer::omni::Qwen2BpeTokenizer;

  std::string model_dir, image_path = kDefaultImage;
  int gpu = 6;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    auto need = [&](const char* n) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", n);
        std::exit(2);
      }
      return argv[++i];
    };
    if (flag == "--model") model_dir = need("--model");
    else if (flag == "--image") image_path = need("--image");
    else if (flag == "--gpu") gpu = std::stoi(need("--gpu"));
    else {
      std::fprintf(stderr, "unknown argument: %s\n", flag.c_str());
      return 2;
    }
  }
  if (model_dir.empty()) {
    std::fprintf(stderr,
                 "usage: umm-omni-bench --model DIR [--image PATH] [--gpu N]\n");
    return 2;
  }

  Qwen2BpeTokenizer tokenizer(model_dir + "/vocab.json",
                              model_dir + "/merges.txt",
                              model_dir + "/tokenizer_config.json");
  std::unique_ptr<PreprocessedImage> image;
  try {
    image = std::make_unique<PreprocessedImage>(
        umminfer::omni::preprocess_image(image_path));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "image load failed (%s), image case skipped\n", e.what());
  }

  std::unique_ptr<OmniEngine> engine;
  try {
    engine = std::make_unique<OmniEngine>(model_dir, gpu);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "engine init failed: %s\n", e.what());
    return 1;
  }

  // Render the same chat turn the CLI does, expanding the image_pad run.
  auto render = [&](bool with_image) {
    std::string s = "<|im_start|>user\n";
    if (with_image) s += "<|vision_start|><|vision_end|>";
    s += std::string(kPrompt) + "<|im_end|>\n<|im_start|>assistant\n";
    std::vector<int> ids = tokenizer.encode(s);
    if (with_image) {
      const int n_img = image->rows() / 4;
      std::vector<int> out;
      out.reserve(ids.size() + n_img);
      bool done = false;
      for (int id : ids) {
        out.push_back(id);
        if (!done && id == Qwen2BpeTokenizer::kVisionStart) {
          out.insert(out.end(), n_img, Qwen2BpeTokenizer::kImagePad);
          done = true;
        }
      }
      ids = std::move(out);
    }
    return ids;
  };

  const Case cases[] = {{"image", true}, {"text", false}};
  for (const Case& c : cases) {
    if (c.with_image && !image) continue;
    const std::vector<int> ids = render(c.with_image);
    const OmniEngine::Result r =
        engine->generate(ids, c.with_image ? image.get() : nullptr, kMaxNewTokens);
    const double n = static_cast<double>(r.token_ids.size());
    std::printf(
        "case=%s prefill_tokens=%d image_tokens=%d vision_s=%.3f prefill_s=%.3f "
        "decode_tokens=%d decode_s=%.3f decode_tok_s=%.3f answer=\"%s\"\n",
        c.name, r.prefill_tokens, r.image_tokens, r.vision_ms / 1000.0,
        r.prefill_ms / 1000.0, static_cast<int>(r.token_ids.size()),
        r.decode_ms / 1000.0, n > 0 ? n / (r.decode_ms / 1000.0) : 0.0,
        r.text.c_str());
  }

  size_t free_b = 0, total_b = 0;
  cudaMemGetInfo(&free_b, &total_b);
  std::printf("resident_mb=%zu total_mb=%zu\n", (total_b - free_b) / (1024 * 1024),
              total_b / (1024 * 1024));
  return 0;
}
