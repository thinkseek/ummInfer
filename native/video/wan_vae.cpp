// SPDX-License-Identifier: Apache-2.0
#include "wan_vae.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>

namespace umminfer::video {

WanVaeConfig load_wan_vae_config(const std::filesystem::path& vae_dir) {
  const auto path = vae_dir / "config.json";
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::invalid_argument("failed to open Wan VAE config: " + path.string());
  }
  nlohmann::json json;
  try {
    input >> json;
  } catch (const nlohmann::json::exception& err) {
    throw std::invalid_argument("malformed Wan VAE config " + path.string() +
                                ": " + err.what());
  }
  WanVaeConfig config;
  try {
    config.z_dim = json.at("z_dim").get<int>();
    // Wan2.2 configs name the encoder width "base_dim"; older configs use "dim".
    config.dim = json.value("base_dim", json.value("dim", 0));
    config.decoder_base_dim = json.value("decoder_base_dim", config.dim);
    config.num_res_blocks = json.at("num_res_blocks").get<int>();
    config.out_channels = json.value("out_channels", 3);
    config.patch_size = json.value("patch_size", 1);
    config.is_residual = json.value("is_residual", false);
    config.dim_mult = json.at("dim_mult").get<std::vector<int>>();
    config.temperal_downsample =
        json.at("temperal_downsample").get<std::vector<bool>>();
    config.attn_scales = json.at("attn_scales").get<std::vector<float>>();
    config.latents_mean = json.at("latents_mean").get<std::vector<float>>();
    config.latents_std = json.at("latents_std").get<std::vector<float>>();
  } catch (const nlohmann::json::exception& err) {
    throw std::invalid_argument("invalid Wan VAE config " + path.string() +
                                ": " + err.what());
  }
  if (config.z_dim <= 0 || config.decoder_base_dim <= 0 ||
      config.num_res_blocks <= 0 || config.out_channels <= 0 ||
      config.patch_size <= 0 || config.dim_mult.empty() ||
      config.dim_mult.size() != config.temperal_downsample.size() + 1 ||
      config.out_channels % (config.patch_size * config.patch_size) != 0) {
    throw std::invalid_argument("Wan VAE dimensions or temporal scales are invalid");
  }
  for (const int multiplier : config.dim_mult) {
    if (multiplier <= 0) throw std::invalid_argument("Wan VAE dim_mult must be positive");
  }
  if (config.latents_mean.size() != static_cast<size_t>(config.z_dim) ||
      config.latents_std.size() != static_cast<size_t>(config.z_dim)) {
    throw std::invalid_argument("Wan VAE latent statistics must have z_dim entries");
  }
  return config;
}

}  // namespace umminfer::video
