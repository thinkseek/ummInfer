// SPDX-License-Identifier: Apache-2.0
// umm-omni-native: model directory + prompt + optional image -> answer.
// No Python, no PyTorch, no SGLang; self-contained CUDA kernels only.
#include <cuda_runtime.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "omni/image_preprocess.hpp"
#include "omni/omni_engine.hpp"
#include "omni/tokenizer.hpp"

namespace {

struct Args {
  std::string model_dir;
  std::string prompt;
  std::string image;
  int gpu = 0;
  int max_new = 512;
};

void usage() {
  std::cerr << "usage: umm-omni-native --model DIR --prompt TEXT "
               "[--image PATH] [--gpu N] [--max-new-tokens N]\n";
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << "\n";
        usage();
        std::exit(2);
      }
      return argv[++i];
    };
    if (flag == "--model")
      args.model_dir = need("--model");
    else if (flag == "--prompt")
      args.prompt = need("--prompt");
    else if (flag == "--image")
      args.image = need("--image");
    else if (flag == "--gpu")
      args.gpu = std::stoi(need("--gpu"));
    else if (flag == "--max-new-tokens")
      args.max_new = std::stoi(need("--max-new-tokens"));
    else {
      std::cerr << "unknown argument: " << flag << "\n";
      usage();
      return 2;
    }
  }
  if (args.model_dir.empty() || args.prompt.empty()) {
    usage();
    return 2;
  }

  try {
    using umminfer::omni::PreprocessedImage;
    using umminfer::omni::Qwen2BpeTokenizer;
    Qwen2BpeTokenizer tokenizer(args.model_dir + "/vocab.json",
                                args.model_dir + "/merges.txt",
                                args.model_dir + "/tokenizer_config.json");

    std::unique_ptr<PreprocessedImage> image;
    int n_img = 0;
    if (!args.image.empty()) {
      image = std::make_unique<PreprocessedImage>(
          umminfer::omni::preprocess_image(args.image));
      n_img = image->rows() / 4;
    }

    // Render the chat turn: <|im_start|>user\n[vision block]PROMPT<|im_end|>\n
    // <|im_start|>assistant\n, expanding the single image_pad to n_img copies.
    std::string rendered = "<|im_start|>user\n";
    if (image) rendered += "<|vision_start|><|vision_end|>";
    rendered += args.prompt + "<|im_end|>\n<|im_start|>assistant\n";
    std::vector<int> ids = tokenizer.encode(rendered);
    if (image) {
      // Insert the expanded image_pad run between vision_start and vision_end.
      std::vector<int> expanded;
      expanded.reserve(ids.size() + n_img);
      bool inserted = false;
      for (int id : ids) {
        expanded.push_back(id);
        if (!inserted && id == Qwen2BpeTokenizer::kVisionStart) {
          expanded.insert(expanded.end(), n_img,
                          Qwen2BpeTokenizer::kImagePad);
          inserted = true;
        }
      }
      if (!inserted) throw std::runtime_error("vision_start not found in prompt");
      ids = std::move(expanded);
    }

    auto t0 = std::chrono::steady_clock::now();
    umminfer::omni::OmniEngine engine(args.model_dir, args.gpu);
    auto t1 = std::chrono::steady_clock::now();
    auto result = engine.generate(ids, image.get(), args.max_new);
    auto t2 = std::chrono::steady_clock::now();

    std::cout << result.text << "\n";
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    std::cerr << "prefill_tokens=" << result.prefill_tokens
              << " image_tokens=" << result.image_tokens
              << " generated_tokens=" << result.token_ids.size()
              << " load_ms="
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " infer_ms="
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " gpu_used_mb=" << (total_b - free_b) / (1024 * 1024) << "\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
