// SPDX-License-Identifier: Apache-2.0
// Real-weight native UMT5 encoder verification. All outputs are raw binaries.
#include "video/umt5_encoder.hpp"
#include "video/umt5_tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename T>
void write_binary(const std::filesystem::path& path, const std::vector<T>& values) {
  std::ofstream output(path, std::ios::binary);
  if (!output) throw std::runtime_error("failed to open output: " + path.string());
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(T)));
  output.close();
  if (!output) throw std::runtime_error("failed to close output: " + path.string());
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path model_dir;
    std::filesystem::path output_prefix;
    std::string prompt;
    bool prompt_seen = false;
    int gpu = 0;
    bool pad_to_max = true;
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      const auto value = [&]() -> std::string {
        if (++i >= argc) throw std::invalid_argument("missing value after " + argument);
        return argv[i];
      };
      if (argument == "--model-dir") {
        model_dir = value();
      } else if (argument == "--prompt") {
        prompt = value();
        prompt_seen = true;
      } else if (argument == "--output-prefix") {
        output_prefix = value();
      } else if (argument == "--gpu") {
        gpu = std::stoi(value());
      } else if (argument == "--no-pad") {
        pad_to_max = false;
      } else {
        throw std::invalid_argument("unknown argument: " + argument);
      }
    }
    if (model_dir.empty() || output_prefix.empty() || !prompt_seen || gpu < 0) {
      throw std::invalid_argument(
          "usage: umm-video-umt5-test --model-dir DIR --prompt TEXT "
          "--output-prefix PREFIX [--gpu 0] [--no-pad]");
    }

    umminfer::video::Umt5Tokenizer tokenizer(model_dir / "tokenizer/tokenizer.json");
    const auto tokens = tokenizer.encode(prompt, 512, pad_to_max);
    umminfer::video::Umt5Encoder encoder(model_dir / "text_encoder", gpu);
    const auto result = encoder.encode(tokens);

    const std::string prefix = output_prefix.string();
    write_binary(prefix + ".tokens.bin", tokens.ids);
    write_binary(prefix + ".mask.bin", tokens.attention_mask);
    write_binary(prefix + ".fp32.bin", result.hidden_states);
    const int active = std::accumulate(tokens.attention_mask.begin(),
                                       tokens.attention_mask.end(), 0);
    std::cout << std::fixed << std::setprecision(3)
              << "active_tokens=" << active << "\n"
              << "shape=" << result.shape[0] << "," << result.shape[1] << "\n"
              << "load_ms=" << encoder.load_ms() << "\n"
              << "encode_ms=" << result.encode_ms << "\n"
              << "peak_native_device_mib="
              << result.peak_native_device_bytes / 1048576.0 << "\n"
              << "tokens_bin=" << prefix << ".tokens.bin\n"
              << "embeddings_bin=" << prefix << ".fp32.bin\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_umt5: " << error.what() << "\n";
    return 2;
  }
}
