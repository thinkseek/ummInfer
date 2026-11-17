// SPDX-License-Identifier: Apache-2.0
// umm-qwen38-native: model directory + prompt + optional image -> answer.
// No Python, no PyTorch, no SGLang, no cuBLAS; only the CUDA runtime.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "qwen38/engine.hpp"
#include "qwen38/image_preprocess.hpp"
#include "qwen38/tokenizer.hpp"

namespace {

struct Args {
  std::string model_dir;
  std::string prompt;
  std::string image;
  std::string dump_dir;
  int gpu = 0;
  int max_new = 256;
  bool thinking = false;
};

void usage() {
  std::cerr << "usage: umm-qwen38-native --model DIR --prompt TEXT [--image PATH] "
               "[--gpu N] [--max-new-tokens M] [--thinking] [--dump-dir DIR]\n";
}

using umminfer::qwen38::PreprocessedImage;

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
    else if (flag == "--dump-dir")
      args.dump_dir = need("--dump-dir");
    else if (flag == "--thinking")
      args.thinking = true;
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
    using umminfer::qwen38::Qwen2BpeTokenizer;
    Qwen2BpeTokenizer tokenizer(args.model_dir + "/vocab.json",
                                args.model_dir + "/merges.txt",
                                args.model_dir + "/tokenizer_config.json");

    std::unique_ptr<PreprocessedImage> image;
    int n_img = 0;
    if (!args.image.empty()) {
      image = std::make_unique<PreprocessedImage>(
          umminfer::qwen38::preprocess_image(args.image));
      if (image->grid_t != 1)
        throw std::runtime_error("only still images are supported");
      n_img = image->rows() / 4;
    }

    const std::string rendered =
        umminfer::qwen38::render_chat_prompt(args.prompt, image != nullptr,
                                             args.thinking);
    std::vector<int> ids = tokenizer.encode(rendered);
    if (image) ids = umminfer::qwen38::expand_image_pads(ids, n_img);

    const auto t0 = std::chrono::steady_clock::now();
    auto engine = std::make_unique<umminfer::qwen38::Engine>(args.model_dir, args.gpu);
    const auto t1 = std::chrono::steady_clock::now();
    auto result = engine->generate(ids, image.get(), args.max_new, args.dump_dir);
    const auto t2 = std::chrono::steady_clock::now();

    std::cout << tokenizer.decode(result.ids) << "\n";

    const double load_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double wall_ms =
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    const double decode_tok_s =
        result.decode_ms > 0.0 ? result.decode_steps * 1000.0 / result.decode_ms : 0.0;
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    std::cerr << "load_ms=" << static_cast<long long>(load_ms) << "\n"
              << "prompt_tokens=" << result.prompt_tokens << "\n"
              << "image_tokens=" << result.image_tokens << "\n"
              << "vision_ms=" << static_cast<long long>(result.vision_ms) << "\n"
              << "prefill_ms=" << static_cast<long long>(result.prefill_ms) << "\n"
              << "decode_steps=" << result.decode_steps << "\n"
              << "decode_ms=" << static_cast<long long>(result.decode_ms) << "\n"
              << "decode_tok_per_s=" << decode_tok_s << "\n"
              << "wall_ms=" << static_cast<long long>(wall_ms) << "\n"
              << "gpu_used_mb=" << (total_b - free_b) / (1024 * 1024) << "\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
