// SPDX-License-Identifier: Apache-2.0
#include "config.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace umminfer::video {
namespace {

bool is_40_hex(const std::string& s) {
  if (s.size() != 40) return false;
  for (unsigned char c : s) {
    if (!std::isxdigit(c)) return false;
  }
  return true;
}

nlohmann::json read_json_file(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    throw std::invalid_argument("failed to open JSON file: " + path.string());
  }
  try {
    nlohmann::json doc;
    in >> doc;
    if (!doc.is_object()) {
      throw std::invalid_argument("JSON root must be an object: " + path.string());
    }
    return doc;
  } catch (const nlohmann::json::exception& err) {
    throw std::invalid_argument("malformed JSON in " + path.string() + ": " +
                                err.what());
  }
}

}  // namespace

std::string detect_model_revision(const std::string& model_dir,
                                  const std::string& explicit_revision) {
  if (!explicit_revision.empty()) return explicit_revision;
  const std::filesystem::path path(model_dir);
  std::error_code ec;
  if (std::filesystem::exists(path, ec)) {
    const std::string name = path.filename().string();
    const std::string parent = path.parent_path().filename().string();
    if (parent == "snapshots" && is_40_hex(name)) {
      return name;
    }
  }
  return "UNKNOWN";
}

WanModelConfig load_wan_model_config(const std::string& model_dir,
                                     const std::string& explicit_revision) {
  if (model_dir.empty()) {
    throw std::invalid_argument("video model directory must not be empty");
  }
  const std::filesystem::path root(model_dir);
  std::error_code ec;
  if (!std::filesystem::exists(root, ec) ||
      !std::filesystem::is_directory(root, ec)) {
    throw std::invalid_argument("video model directory does not exist: " +
                                model_dir);
  }

  WanModelConfig cfg;
  cfg.model_dir = model_dir;
  cfg.model_id = root.filename().string();
  if (cfg.model_id.empty()) {
    cfg.model_id = root.parent_path().filename().string();
  }
  cfg.revision = detect_model_revision(model_dir, explicit_revision);

  const std::filesystem::path model_index = root / "model_index.json";
  if (std::filesystem::is_regular_file(model_index, ec)) {
    const nlohmann::json meta = read_json_file(model_index);
    if (meta.contains("_class_name") && meta["_class_name"].is_string()) {
      cfg.model_class = meta["_class_name"].get<std::string>();
    }
  }

  int vae_spatial_scale = 8;
  const std::filesystem::path vae_cfg = root / "vae" / "config.json";
  if (std::filesystem::is_regular_file(vae_cfg, ec)) {
    const nlohmann::json vae = read_json_file(vae_cfg);
    if (vae.contains("z_dim") && vae["z_dim"].is_number_integer()) {
      cfg.latent_channels = vae["z_dim"].get<int>();
    }
    if (vae.contains("scale_factor_spatial") &&
        vae["scale_factor_spatial"].is_number_integer()) {
      vae_spatial_scale = vae["scale_factor_spatial"].get<int>();
    } else if (vae.contains("temperal_downsample") &&
               vae["temperal_downsample"].is_array()) {
      // Wan2.1/2.2 VAE encodes temporal downsample booleans; spatial is 8x or 16x.
      int t_stride = 1;
      for (const auto& flag : vae["temperal_downsample"]) {
        if (flag.is_boolean() && flag.get<bool>()) t_stride *= 2;
      }
      if (t_stride > 1) cfg.temporal_stride = t_stride;
    }
    if (vae.contains("scale_factor_temporal") &&
        vae["scale_factor_temporal"].is_number_integer()) {
      cfg.temporal_stride = vae["scale_factor_temporal"].get<int>();
    }
    if (vae.contains("latents_mean") && vae["latents_mean"].is_array()) {
      cfg.latents_mean.clear();
      for (const auto& v : vae["latents_mean"]) {
        if (v.is_number()) cfg.latents_mean.push_back(v.get<float>());
      }
    }
    if (vae.contains("latents_std") && vae["latents_std"].is_array()) {
      cfg.latents_std.clear();
      for (const auto& v : vae["latents_std"]) {
        if (v.is_number()) cfg.latents_std.push_back(v.get<float>());
      }
    }
  }

  const std::filesystem::path tx_cfg = root / "transformer" / "config.json";
  if (std::filesystem::is_regular_file(tx_cfg, ec)) {
    const nlohmann::json tx = read_json_file(tx_cfg);
    if (tx.contains("patch_size") && tx["patch_size"].is_array() &&
        tx["patch_size"].size() == 3) {
      cfg.patch_t = tx["patch_size"][0].get<int>();
      cfg.patch_h = tx["patch_size"][1].get<int>();
      cfg.patch_w = tx["patch_size"][2].get<int>();
    }
    if (tx.contains("in_channels") && tx["in_channels"].is_number_integer()) {
      cfg.in_channels = tx["in_channels"].get<int>();
    }
    if (tx.contains("out_channels") && tx["out_channels"].is_number_integer()) {
      cfg.out_channels = tx["out_channels"].get<int>();
    }
    if (tx.contains("num_layers") && tx["num_layers"].is_number_integer()) {
      cfg.num_layers = tx["num_layers"].get<int>();
    }
  }

  const std::filesystem::path sched_cfg =
      root / "scheduler" / "scheduler_config.json";
  if (std::filesystem::is_regular_file(sched_cfg, ec)) {
    const nlohmann::json sched = read_json_file(sched_cfg);
    if (sched.contains("flow_shift") && sched["flow_shift"].is_number()) {
      cfg.flow_shift = sched["flow_shift"].get<double>();
    }
    if (sched.contains("num_train_timesteps") &&
        sched["num_train_timesteps"].is_number_integer()) {
      cfg.num_train_timesteps = sched["num_train_timesteps"].get<int>();
    }
  }

  // Wan2.2-TI2V-5B has 16x spatial VAE + 2x2 patch = 32; Wan2.1 has 8x * 2 = 16.
  // Keep 32 as the default conservative Wan2.2 alignment unless config explicitly
  // declares a larger alignment.
  const int computed_align = std::max(16, vae_spatial_scale * std::max(1, cfg.patch_w));
  cfg.spatial_align = std::max(32, computed_align);

  if (cfg.latents_mean.empty()) {
    cfg.latents_mean.assign(static_cast<size_t>(cfg.latent_channels), 0.0f);
  }
  if (cfg.latents_std.empty()) {
    cfg.latents_std.assign(static_cast<size_t>(cfg.latent_channels), 1.0f);
  }
  return cfg;
}

void validate_wan_checkpoint_paths(const std::string& model_dir) {
  const std::filesystem::path root(model_dir);
  for (const char* rel : {"model_index.json", "transformer", "vae", "scheduler"}) {
    std::error_code ec;
    if (!std::filesystem::exists(root / rel, ec)) {
      throw std::runtime_error(
          "native video checkpoint missing required path '" +
          (root / rel).string() + "'");
    }
  }
}

std::pair<int, int> parse_video_size(const std::string& size_str,
                                     int default_width, int default_height,
                                     int align, int min_dim, int max_dim) {
  int width = default_width;
  int height = default_height;
  if (!size_str.empty()) {
    std::string lower = size_str;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const size_t x_pos = lower.find('x');
    if (x_pos == std::string::npos || x_pos == 0 || x_pos + 1 == lower.size() ||
        lower.find('x', x_pos + 1) != std::string::npos) {
      throw std::invalid_argument("video.size must be formatted as WIDTHxHEIGHT");
    }
    const std::string w_part = lower.substr(0, x_pos);
    const std::string h_part = lower.substr(x_pos + 1);
    auto all_digits = [](const std::string& s) {
      return !s.empty() &&
             std::all_of(s.begin(), s.end(),
                         [](unsigned char c) { return std::isdigit(c) != 0; });
    };
    if (!all_digits(w_part) || !all_digits(h_part)) {
      throw std::invalid_argument(
          "video.size must contain positive integers WIDTHxHEIGHT");
    }
    try {
      width = std::stoi(w_part);
      height = std::stoi(h_part);
    } catch (const std::exception&) {
      throw std::invalid_argument(
          "video.size must contain positive integers WIDTHxHEIGHT");
    }
  }

  if (width < min_dim || height < min_dim || width > max_dim || height > max_dim) {
    throw std::invalid_argument(
        "video.size " + std::to_string(width) + "x" + std::to_string(height) +
        " is outside [" + std::to_string(min_dim) + ", " +
        std::to_string(max_dim) + "]");
  }
  if (align > 1 && (width % align != 0 || height % align != 0)) {
    throw std::invalid_argument(
        "video.size " + std::to_string(width) + "x" + std::to_string(height) +
        " must be divisible by " + std::to_string(align));
  }
  return {width, height};
}

int align_video_frames(int seconds, int fps, int max_frames, int temporal_stride) {
  if (seconds < 1) {
    throw std::invalid_argument("video.seconds must be a positive integer");
  }
  if (fps < 1 || fps > 60) {
    throw std::invalid_argument("video.fps must be between 1 and 60");
  }
  if (max_frames < 1) {
    throw std::invalid_argument("max_frames must be positive");
  }
  const int raw = seconds * fps;
  int target = std::max(temporal_stride + 1, std::min(raw, max_frames));
  if (temporal_stride > 1) {
    int steps = std::max(1, (target - 1) / temporal_stride);
    target = steps * temporal_stride + 1;
    while (target > max_frames && steps > 1) {
      --steps;
      target = steps * temporal_stride + 1;
    }
  }
  return target;
}

}  // namespace umminfer::video
