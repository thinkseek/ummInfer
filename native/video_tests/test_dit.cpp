// SPDX-License-Identifier: Apache-2.0
// Real-weight Wan2.2 DiT parity and native end-to-end video verification.
#include "video/engine.hpp"
#include "video/wan_dit.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::vector<float> read_f32(const std::filesystem::path& path, size_t count) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("failed to open FP32 input: " + path.string());
  const std::streamsize bytes = input.tellg();
  if (bytes < 0 || static_cast<uint64_t>(bytes) != count * sizeof(float)) {
    throw std::invalid_argument("FP32 input size mismatch: " + path.string());
  }
  input.seekg(0);
  std::vector<float> values(count);
  if (!input.read(reinterpret_cast<char*>(values.data()), bytes)) {
    throw std::runtime_error("failed to read FP32 input: " + path.string());
  }
  return values;
}

void write_f32(const std::filesystem::path& path, const std::vector<float>& values) {
  std::ofstream output(path, std::ios::binary);
  if (!output) throw std::runtime_error("failed to create FP32 output: " + path.string());
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!output) throw std::runtime_error("failed to write FP32 output: " + path.string());
}

struct Metrics {
  double cosine = 0.0;
  double psnr = 0.0;
  double max_abs = 0.0;
  double mean_abs = 0.0;
};

Metrics compare(const std::vector<float>& actual, const std::vector<float>& expected) {
  if (actual.size() != expected.size() || actual.empty()) {
    throw std::invalid_argument("parity tensors must have the same nonzero size");
  }
  long double dot = 0.0L;
  long double actual_norm = 0.0L;
  long double expected_norm = 0.0L;
  long double squared_error = 0.0L;
  float minimum = std::numeric_limits<float>::infinity();
  float maximum = -std::numeric_limits<float>::infinity();
  Metrics metrics;
  for (size_t i = 0; i < actual.size(); ++i) {
    const double a = actual[i];
    const double b = expected[i];
    const double difference = a - b;
    dot += static_cast<long double>(a) * b;
    actual_norm += static_cast<long double>(a) * a;
    expected_norm += static_cast<long double>(b) * b;
    squared_error += static_cast<long double>(difference) * difference;
    metrics.max_abs = std::max(metrics.max_abs, std::fabs(difference));
    metrics.mean_abs += std::fabs(difference);
    minimum = std::min(minimum, expected[i]);
    maximum = std::max(maximum, expected[i]);
  }
  metrics.mean_abs /= actual.size();
  metrics.cosine = static_cast<double>(dot / std::sqrt(actual_norm * expected_norm));
  const double mse = static_cast<double>(squared_error / actual.size());
  const double range = static_cast<double>(maximum) - minimum;
  metrics.psnr = mse == 0.0 ? std::numeric_limits<double>::infinity()
                            : 10.0 * std::log10(range * range / mse);
  return metrics;
}

int run_parity(const std::unordered_map<std::string, std::string>& args) {
  for (const char* required : {"--model-dir", "--latent-bin", "--context-bin",
                               "--expected-bin", "--output-bin"}) {
    if (!args.contains(required)) throw std::invalid_argument(std::string("required argument: ") + required);
  }
  const int frames = args.contains("--frames") ? std::stoi(args.at("--frames")) : 3;
  const int height = args.contains("--height") ? std::stoi(args.at("--height")) : 16;
  const int width = args.contains("--width") ? std::stoi(args.at("--width")) : 22;
  const int text_len = args.contains("--text-len") ? std::stoi(args.at("--text-len")) : 512;
  const int gpu = args.contains("--gpu") ? std::stoi(args.at("--gpu")) : 0;
  const float timestep = args.contains("--timestep") ? std::stof(args.at("--timestep")) : 500.0f;
  if (frames <= 0 || height <= 0 || width <= 0 || text_len <= 0 || gpu < 0) {
    throw std::invalid_argument("invalid parity dimensions or GPU index");
  }
  const size_t latent_count = static_cast<size_t>(48) * frames * height * width;
  const auto latent = read_f32(args.at("--latent-bin"), latent_count);
  const auto context = read_f32(args.at("--context-bin"), static_cast<size_t>(text_len) * 4096);
  const auto expected = read_f32(args.at("--expected-bin"), latent_count);

  umminfer::video::WanDit dit(args.at("--model-dir"), gpu);
  const bool first_frame_zero = args.contains("--first-frame-zero");
  const auto output = dit.forward(latent.data(), frames, height, width,
                                  timestep, context.data(), text_len,
                                  first_frame_zero);
  write_f32(args.at("--output-bin"), output.velocity);
  const Metrics metrics = compare(output.velocity, expected);
  std::cout << std::fixed << std::setprecision(9)
            << "tensor_count=" << output.stats.tensor_count << '\n'
            << "cosine=" << metrics.cosine << '\n'
            << "psnr_db=" << metrics.psnr << '\n'
            << "max_abs_diff=" << metrics.max_abs << '\n'
            << "mean_abs_diff=" << metrics.mean_abs << '\n'
            << "load_ms=" << output.stats.load_ms << '\n'
            << "forward_ms=" << output.stats.forward_ms << '\n'
            << "peak_vram_mib=" << output.stats.peak_vram_bytes / 1048576.0 << '\n'
            << "output_bin=" << args.at("--output-bin") << '\n';
  if (output.stats.tensor_count != 825 || metrics.cosine <= 0.999 || metrics.psnr <= 55.0) {
    std::cerr << "DiT parity threshold failed: cosine>0.999 and PSNR>55 dB required\n";
    return 3;
  }
  return 0;
}

int run_e2e(const std::unordered_map<std::string, std::string>& args) {
  for (const char* required : {"--model-dir", "--output-mp4"}) {
    if (!args.contains(required)) {
      throw std::invalid_argument(std::string("required argument: ") + required);
    }
  }
  const int gpu = args.contains("--gpu") ? std::stoi(args.at("--gpu")) : 0;
  umminfer::video::GenerateParams params;
  params.prompt = args.contains("--prompt")
                      ? args.at("--prompt")
                      : "A cinematic wide shot of a red paper boat drifting on a calm mountain lake at sunrise";
  params.negative_prompt = args.contains("--negative-prompt")
                               ? args.at("--negative-prompt")
                               : "";
  params.width = args.contains("--width") ? std::stoi(args.at("--width")) : 352;
  params.height = args.contains("--height") ? std::stoi(args.at("--height")) : 256;
  params.num_frames = args.contains("--frames") ? std::stoi(args.at("--frames")) : 9;
  params.fps = args.contains("--fps") ? std::stoi(args.at("--fps")) : 16;
  params.steps = args.contains("--steps") ? std::stoi(args.at("--steps")) : 12;
  params.guidance_scale =
      args.contains("--guidance-scale") ? std::stof(args.at("--guidance-scale")) : 5.0f;
  params.seed = args.contains("--seed") ? std::stoull(args.at("--seed")) : 42ULL;
  if (args.contains("--init-image")) {
    std::ifstream image(args.at("--init-image"), std::ios::binary);
    if (!image) throw std::runtime_error("failed to open init image");
    params.init_image_bytes.assign(std::istreambuf_iterator<char>(image),
                                   std::istreambuf_iterator<char>());
  }

  umminfer::video::VideoEngine engine(
      gpu, args.at("--model-dir"), umminfer::video::VideoBackendMode::kCuda);
  const auto result = engine.generate(params);
  {
    std::ofstream out(args.at("--output-mp4"), std::ios::binary);
    if (!out) throw std::runtime_error("failed to create output MP4");
    out.write(result.mp4_bytes.data(),
              static_cast<std::streamsize>(result.mp4_bytes.size()));
    if (!out) throw std::runtime_error("failed to write output MP4");
  }
  std::cout << std::fixed << std::setprecision(3)
            << "backend=" << result.backend << '\n'
            << "width=" << result.width << '\n'
            << "height=" << result.height << '\n'
            << "frames=" << result.frames << '\n'
            << "fps=" << result.fps << '\n'
            << "steps=" << result.steps << '\n'
            << "i2v=" << (params.init_image_bytes.empty() ? 0 : 1) << '\n'
            << "seed=" << result.seed << '\n'
            << "denoise_ms=" << result.denoise_ms << '\n'
            << "encode_ms=" << result.encode_ms << '\n'
            << "peak_vram_mib=" << result.peak_gpu_bytes / 1048576.0 << '\n'
            << "mp4_bytes=" << result.mp4_bytes.size() << '\n'
            << "output_mp4=" << args.at("--output-mp4") << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::unordered_map<std::string, std::string> args;
    for (int i = 1; i < argc; ) {
      const std::string key = argv[i];
      if (key == "--e2e" || key == "--first-frame-zero") {
        args.emplace(key, "1");
        i += 1;
        continue;
      }
      if (i + 1 >= argc) throw std::invalid_argument("missing argument value");
      args.emplace(key, argv[i + 1]);
      i += 2;
    }
    if (args.contains("--e2e")) {
      return run_e2e(args);
    }
    return run_parity(args);
  } catch (const std::exception& error) {
    std::cerr << "umm-video-dit-test: " << error.what() << '\n';
    return 2;
  }
}
