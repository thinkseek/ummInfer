// SPDX-License-Identifier: Apache-2.0
// Numerical parity utility for the native Wan2.2 FP32 CUDA VAE decoder.
#include "video/mp4_encoder.hpp"
#include "video/wan_vae.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::vector<int64_t> parse_shape(const std::string& text) {
  std::vector<int64_t> shape;
  size_t begin = 0;
  while (begin < text.size()) {
    const size_t end = text.find(',', begin);
    const std::string part = text.substr(begin, end - begin);
    size_t used = 0;
    const int64_t value = std::stoll(part, &used);
    if (used != part.size() || value <= 0) {
      throw std::invalid_argument("latent shape must contain positive integers");
    }
    shape.push_back(value);
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  if (shape.size() != 5) {
    throw std::invalid_argument("latent shape must be N,C,T,H,W");
  }
  return shape;
}

size_t element_count(const std::vector<int64_t>& shape) {
  size_t count = 1;
  for (const int64_t dim : shape) {
    if (dim <= 0 || count > std::numeric_limits<size_t>::max() /
                               static_cast<uint64_t>(dim)) {
      throw std::invalid_argument("tensor element count overflow");
    }
    count *= static_cast<size_t>(dim);
  }
  return count;
}

std::vector<float> read_f32(const std::filesystem::path& path, size_t count) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input.is_open()) throw std::runtime_error("failed to open latent file");
  const std::streamsize bytes = input.tellg();
  if (bytes < 0 || static_cast<uint64_t>(bytes) != count * sizeof(float)) {
    throw std::invalid_argument("latent file size does not match latent shape");
  }
  input.seekg(0);
  std::vector<float> values(count);
  if (!input.read(reinterpret_cast<char*>(values.data()), bytes)) {
    throw std::runtime_error("failed to read latent file");
  }
  return values;
}

void write_f32(const std::filesystem::path& path, const std::vector<float>& values) {
  std::ofstream output(path, std::ios::binary);
  if (!output.is_open()) throw std::runtime_error("failed to create FP32 output");
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!output) throw std::runtime_error("failed to write FP32 output");
}

std::vector<std::vector<uint8_t>> to_rgb24(const umminfer::video::WanVaeOutput& output) {
  if (output.shape.size() != 5 || output.shape[0] != 1 || output.shape[1] != 3) {
    throw std::runtime_error("decoder output is not [1,3,T,H,W]");
  }
  const int64_t frames = output.shape[2];
  const int64_t height = output.shape[3];
  const int64_t width = output.shape[4];
  std::vector<std::vector<uint8_t>> rgb(
      static_cast<size_t>(frames),
      std::vector<uint8_t>(static_cast<size_t>(height * width * 3)));
  for (int64_t t = 0; t < frames; ++t) {
    for (int64_t y = 0; y < height; ++y) {
      for (int64_t x = 0; x < width; ++x) {
        for (int64_t c = 0; c < 3; ++c) {
          const size_t source = static_cast<size_t>(
              ((c * frames + t) * height + y) * width + x);
          const float clamped = std::max(-1.0f, std::min(1.0f, output.data[source]));
          const auto value = static_cast<uint8_t>(
              std::lround((clamped + 1.0f) * 127.5f));
          const size_t target = static_cast<size_t>((y * width + x) * 3 + c);
          rgb[static_cast<size_t>(t)][target] = value;
        }
      }
    }
  }
  return rgb;
}

int run_encode(const std::unordered_map<std::string, std::string>& args) {
  for (const char* required : {"--vae-dir", "--image-bin", "--out-f32",
                               "--width", "--height", "--gpu"}) {
    if (!args.contains(required)) {
      throw std::invalid_argument(std::string("required argument: ") + required);
    }
  }
  const int width = std::stoi(args.at("--width"));
  const int height = std::stoi(args.at("--height"));
  const int gpu = std::stoi(args.at("--gpu"));
  const size_t count = static_cast<size_t>(3) * height * width;
  const auto image = read_f32(args.at("--image-bin"), count);
  const auto encoded = umminfer::video::encode_wan_vae_image_fp32(
      args.at("--vae-dir"), image, height, width, gpu);
  write_f32(args.at("--out-f32"), encoded.data);
  std::cout << "output_shape=";
  for (size_t i = 0; i < encoded.shape.size(); ++i) {
    if (i != 0) std::cout << ',';
    std::cout << encoded.shape[i];
  }
  std::cout << "\nencode_ms=" << encoded.encode_ms
            << "\npeak_native_device_bytes="
            << encoded.peak_native_device_bytes << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::unordered_map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 >= argc) throw std::invalid_argument("missing argument value");
      args.emplace(argv[i], argv[i + 1]);
    }
    if (args.contains("--image-bin")) {
      return run_encode(args);
    }
    for (const char* required :
         {"--vae-dir", "--latent-bin", "--latent-shape", "--out-f32", "--out-mp4", "--gpu"}) {
      if (!args.contains(required)) {
        throw std::invalid_argument(std::string("required argument: ") + required);
      }
    }
    const auto shape = parse_shape(args.at("--latent-shape"));
    const auto latent = read_f32(args.at("--latent-bin"), element_count(shape));
    const int gpu = std::stoi(args.at("--gpu"));
    const int fps = args.contains("--fps") ? std::stoi(args.at("--fps")) : 24;
    auto output = umminfer::video::decode_wan_vae_fp32(
        args.at("--vae-dir"), latent, shape, gpu);
    write_f32(args.at("--out-f32"), output.data);
    const auto rgb = to_rgb24(output);
    const std::string mp4 = umminfer::video::encode_rgb24_frames_mp4(
        rgb, static_cast<int>(output.shape[4]), static_cast<int>(output.shape[3]), fps);
    {
      std::ofstream file(args.at("--out-mp4"), std::ios::binary);
      file.write(mp4.data(), static_cast<std::streamsize>(mp4.size()));
      if (!file) throw std::runtime_error("failed to write MP4 output");
    }
    std::cout << "output_shape=";
    for (size_t i = 0; i < output.shape.size(); ++i) {
      if (i != 0) std::cout << ',';
      std::cout << output.shape[i];
    }
    std::cout << "\ndecode_ms=" << output.decode_ms
              << "\npeak_native_device_bytes=" << output.peak_native_device_bytes
              << "\nmp4_bytes=" << mp4.size() << '\n';
    return 0;
  } catch (const std::exception& err) {
    std::cerr << "umm-video-vae-test: " << err.what() << '\n';
    return 2;
  }
}
