// SPDX-License-Identifier: Apache-2.0
// Deterministic host-only natural-video decoder profiler. No CUDA/model code.
#include "qwen38/video_input.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using umminfer::qwen38::DecodedVideo;
using umminfer::qwen38::VideoDecodeProfile;
using umminfer::qwen38::VideoFramePolicy;
using umminfer::qwen38::VideoFrameSelection;

struct Arm {
  std::string name;
  VideoFrameSelection selection;
};

std::string hex_u64(uint64_t value) {
  std::ostringstream out;
  out << std::hex << std::setfill('0') << std::setw(16) << value;
  return out.str();
}

void fnv_byte(uint64_t* hash, uint8_t byte) {
  *hash ^= byte;
  *hash *= 1099511628211ull;
}

template <typename T>
void fnv_value(uint64_t* hash, const T& value) {
  static_assert(std::is_trivially_copyable_v<T>);
  const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
  for (size_t i = 0; i < sizeof(T); ++i) fnv_byte(hash, bytes[i]);
}

std::string semantic_digest(const DecodedVideo& decoded) {
  uint64_t hash = 14695981039346656037ull;
  for (int value : {decoded.preprocessed.grid_t, decoded.preprocessed.grid_h,
                    decoded.preprocessed.grid_w, decoded.preprocessed.resized_h,
                    decoded.preprocessed.resized_w}) {
    fnv_value(&hash, value);
  }
  for (float value : decoded.preprocessed.pixel_values) fnv_value(&hash, value);
  for (double value : decoded.timestamps) fnv_value(&hash, value);
  for (double value : decoded.selected_frame_timestamps) fnv_value(&hash, value);
  for (size_t value : decoded.selected_frame_indices) fnv_value(&hash, value);
  for (int64_t value : decoded.selected_source_frame_indices) fnv_value(&hash, value);
  for (const std::string& value : decoded.selected_frame_content_fnv1a64) {
    for (uint8_t byte : value) fnv_byte(&hash, byte);
  }
  fnv_value(&hash, decoded.decoded_frame_count);
  fnv_byte(&hash, decoded.source_indices_exact ? 1 : 0);
  fnv_byte(&hash, decoded.has_nominal_timestamps ? 1 : 0);
  return hex_u64(hash);
}

nlohmann::json profile_json(const VideoDecodeProfile& p) {
  return {{"input_write_ms", p.input_write_ms},
          {"probe_ms", p.probe_ms},
          {"ffmpeg_decode_rgb_ms", p.ffmpeg_decode_rgb_ms},
          {"rgb_split_ms", p.rgb_split_ms},
          {"content_fingerprint_ms", p.content_fingerprint_ms},
          {"temporal_analysis_ms", p.temporal_analysis_ms},
          {"frame_selection_ms", p.frame_selection_ms},
          {"selection_copy_ms", p.selection_copy_ms},
          {"temporal_pack_ms", p.temporal_pack_ms},
          {"total_ms", p.total_ms}};
}

nlohmann::json semantics_json(const DecodedVideo& decoded) {
  return {{"semantic_fnv1a64", semantic_digest(decoded)},
          {"pixel_values_count", decoded.preprocessed.pixel_values.size()},
          {"grid_t", decoded.preprocessed.grid_t},
          {"grid_h", decoded.preprocessed.grid_h},
          {"grid_w", decoded.preprocessed.grid_w},
          {"resized_h", decoded.preprocessed.resized_h},
          {"resized_w", decoded.preprocessed.resized_w},
          {"decoded_frame_count", decoded.decoded_frame_count},
          {"selected_candidate_indices", decoded.selected_frame_indices},
          {"selected_source_frame_indices", decoded.selected_source_frame_indices},
          {"selected_frame_timestamps", decoded.selected_frame_timestamps},
          {"temporal2_pair_first_timestamps", decoded.timestamps},
          {"selected_frame_content_fnv1a64", decoded.selected_frame_content_fnv1a64},
          {"source_indices_exact", decoded.source_indices_exact},
          {"has_nominal_timestamps", decoded.has_nominal_timestamps}};
}

void write_semantics(const std::filesystem::path& path, const DecodedVideo& decoded) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot create semantic output");
  const auto& image = decoded.preprocessed;
  for (int value : {image.grid_t, image.grid_h, image.grid_w, image.resized_h,
                    image.resized_w, image.is_video ? 1 : 0}) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
  }
  const uint64_t pixels = image.pixel_values.size();
  out.write(reinterpret_cast<const char*>(&pixels), sizeof(pixels));
  out.write(reinterpret_cast<const char*>(image.pixel_values.data()),
            static_cast<std::streamsize>(pixels * sizeof(float)));
  for (const auto* values : {&decoded.timestamps, &decoded.selected_frame_timestamps}) {
    const uint64_t count = values->size();
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    out.write(reinterpret_cast<const char*>(values->data()),
              static_cast<std::streamsize>(count * sizeof(double)));
  }
  const uint64_t selected = decoded.selected_frame_indices.size();
  out.write(reinterpret_cast<const char*>(&selected), sizeof(selected));
  for (size_t value : decoded.selected_frame_indices) {
    const uint64_t fixed = value;
    out.write(reinterpret_cast<const char*>(&fixed), sizeof(fixed));
  }
  for (int64_t value : decoded.selected_source_frame_indices) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
  }
  for (const std::string& value : decoded.selected_frame_content_fnv1a64) {
    if (value.size() != 16) throw std::runtime_error("invalid RGB fingerprint");
    out.write(value.data(), value.size());
  }
  const uint64_t decoded_count = decoded.decoded_frame_count;
  out.write(reinterpret_cast<const char*>(&decoded_count), sizeof(decoded_count));
  const uint8_t flags = static_cast<uint8_t>(decoded.source_indices_exact) |
                        (static_cast<uint8_t>(decoded.has_nominal_timestamps) << 1);
  out.write(reinterpret_cast<const char*>(&flags), sizeof(flags));
  if (!out) throw std::runtime_error("failed to write semantic output");
}

Arm parse_arm(const std::string& text) {
  const size_t colon = text.find(':');
  if (colon == std::string::npos) throw std::invalid_argument("arm must be policy:budget");
  const std::string policy = text.substr(0, colon);
  const int budget = std::stoi(text.substr(colon + 1));
  Arm arm;
  arm.name = policy + "-" + std::to_string(budget);
  arm.selection.max_frames = budget;
  if (policy == "temporal") arm.selection.policy = VideoFramePolicy::kTemporal;
  else if (policy != "uniform" && policy != "default") {
    throw std::invalid_argument("arm policy must be default, uniform, or temporal");
  }
  return arm;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path video, output, semantic_dir;
    int repetitions = 9, warmup = 2;
    uint64_t seed = 510075;
    std::vector<Arm> arms;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      auto next = [&]() -> std::string {
        if (++i >= argc) throw std::invalid_argument("missing value for " + arg);
        return argv[i];
      };
      if (arg == "--video") video = next();
      else if (arg == "--output") output = next();
      else if (arg == "--semantic-dir") semantic_dir = next();
      else if (arg == "--arm") arms.push_back(parse_arm(next()));
      else if (arg == "--repetitions") repetitions = std::stoi(next());
      else if (arg == "--warmup") warmup = std::stoi(next());
      else if (arg == "--seed") seed = std::stoull(next());
      else throw std::invalid_argument("unknown argument: " + arg);
    }
    if (video.empty() || output.empty() || repetitions <= 0 || warmup < 0) {
      throw std::invalid_argument("--video/--output and positive --repetitions are required");
    }
    if (arms.empty()) {
      arms = {parse_arm("default:8"), parse_arm("uniform:4"), parse_arm("uniform:8"),
              parse_arm("uniform:16"), parse_arm("temporal:4"),
              parse_arm("temporal:8"), parse_arm("temporal:16")};
    }
    std::ifstream input(video, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open video");
    const std::string bytes((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
    if (bytes.empty()) throw std::runtime_error("empty video");
    if (!semantic_dir.empty()) std::filesystem::create_directories(semantic_dir);

    std::mt19937_64 rng(seed);
    nlohmann::json records = nlohmann::json::array();
    for (int phase = 0; phase < 2; ++phase) {
      const int count = phase == 0 ? warmup : repetitions;
      for (int repetition = 0; repetition < count; ++repetition) {
        std::vector<size_t> order(arms.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::shuffle(order.begin(), order.end(), rng);
        for (size_t index : order) {
          VideoDecodeProfile profile;
          const DecodedVideo decoded = umminfer::qwen38::decode_video_bytes_profiled(
              bytes, arms[index].selection, &profile);
          const nlohmann::json semantics = semantics_json(decoded);
          if (phase == 0) {
            if (repetition == 0 && !semantic_dir.empty()) {
              write_semantics(semantic_dir / (arms[index].name + ".bin"), decoded);
            }
            continue;
          }
          records.push_back({{"arm", arms[index].name},
                             {"repetition", repetition},
                             {"profile", profile_json(profile)},
                             {"semantics", semantics}});
        }
      }
    }
    nlohmann::json report = {{"schema_version", 1},
                             {"seed", seed},
                             {"warmup_repetitions", warmup},
                             {"measured_repetitions", repetitions},
                             {"records", records}};
    std::ofstream out(output, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create profile output");
    out << report.dump(2) << "\n";
    if (!out) throw std::runtime_error("failed to write profile output");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "video_decode_profile: " << error.what() << "\n";
    return 2;
  }
}
