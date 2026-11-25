// SPDX-License-Identifier: Apache-2.0
#include "wan_dit.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace umminfer::video {
namespace {

size_t checked_count(int frames, int height, int width, int channels) {
  if (frames <= 0 || height <= 0 || width <= 0 || channels <= 0 ||
      height % 2 != 0 || width % 2 != 0) {
    throw std::invalid_argument("Wan DiT patch geometry must be positive with even H/W");
  }
  const uint64_t count = static_cast<uint64_t>(frames) * height * width * channels;
  if (count > std::numeric_limits<size_t>::max()) {
    throw std::overflow_error("Wan DiT patch buffer size overflows size_t");
  }
  return static_cast<size_t>(count);
}

}  // namespace

WanDitConfig load_wan_dit_config(const std::string& model_dir) {
  const std::filesystem::path path =
      std::filesystem::path(model_dir) / "transformer" / "config.json";
  std::ifstream input(path);
  if (!input) throw std::invalid_argument("failed to open Wan DiT config: " + path.string());
  nlohmann::json json;
  try {
    input >> json;
    WanDitConfig c;
    c.num_layers = json.at("num_layers").get<int>();
    c.num_attention_heads = json.at("num_attention_heads").get<int>();
    c.attention_head_dim = json.at("attention_head_dim").get<int>();
    c.inner_dim = c.num_attention_heads * c.attention_head_dim;
    c.ffn_dim = json.at("ffn_dim").get<int>();
    c.in_channels = json.at("in_channels").get<int>();
    c.out_channels = json.at("out_channels").get<int>();
    const auto patch = json.at("patch_size").get<std::vector<int>>();
    if (patch.size() != 3) throw std::invalid_argument("Wan patch_size must have three axes");
    c.patch_t = patch[0];
    c.patch_h = patch[1];
    c.patch_w = patch[2];
    c.text_dim = json.at("text_dim").get<int>();
    c.freq_dim = json.at("freq_dim").get<int>();
    c.rope_max_seq_len = json.at("rope_max_seq_len").get<int>();
    c.eps = json.at("eps").get<float>();
    c.cross_attn_norm = json.at("cross_attn_norm").get<bool>();
    c.qk_norm = json.at("qk_norm").get<std::string>();
    if (json.at("image_dim") != nullptr ||
        (json.contains("added_kv_proj_dim") && json.at("added_kv_proj_dim") != nullptr)) {
      throw std::invalid_argument("image-conditioned Wan DiT is not supported");
    }
    const bool valid = c.num_layers == 30 && c.num_attention_heads == 24 &&
                       c.attention_head_dim == 128 && c.inner_dim == 3072 &&
                       c.ffn_dim == 14336 && c.in_channels == 48 &&
                       c.out_channels == 48 && c.patch_t == 1 && c.patch_h == 2 &&
                       c.patch_w == 2 && c.text_dim == 4096 && c.freq_dim == 256 &&
                       c.rope_max_seq_len == 1024 && c.cross_attn_norm &&
                       c.qk_norm == "rms_norm_across_heads" &&
                       std::fabs(c.eps - 1e-6f) < 1e-12f;
    if (!valid) throw std::invalid_argument("unsupported Wan2.2-TI2V-5B DiT dimensions");
    return c;
  } catch (const nlohmann::json::exception& error) {
    throw std::invalid_argument("invalid Wan DiT config " + path.string() + ": " + error.what());
  }
}

std::vector<float> wan_sinusoidal_timestep_embedding(float timestep, int freq_dim) {
  if (!std::isfinite(timestep) || freq_dim <= 0 || freq_dim % 2 != 0) {
    throw std::invalid_argument("timestep must be finite and freq_dim must be positive and even");
  }
  const int half = freq_dim / 2;
  std::vector<float> output(static_cast<size_t>(freq_dim));
  const float log_period = std::log(10000.0f);
  for (int i = 0; i < half; ++i) {
    const float exponent = -log_period * static_cast<float>(i) /
                           static_cast<float>(half);
    const float angle = timestep * std::exp(exponent);
    output[static_cast<size_t>(i)] = std::cos(angle);
    output[static_cast<size_t>(i + half)] = std::sin(angle);
  }
  return output;
}

WanRope3D wan_build_rope_3d(int frames, int height, int width, int head_dim,
                            float theta) {
  if (frames <= 0 || height <= 0 || width <= 0 || frames % 1 != 0 ||
      height % 2 != 0 || width % 2 != 0 || head_dim != 128 ||
      !std::isfinite(theta) || theta <= 0.0f) {
    throw std::invalid_argument("unsupported Wan 3D RoPE geometry");
  }
  const int pf = frames;
  const int ph = height / 2;
  const int pw = width / 2;
  WanRope3D result;
  result.frames = pf;
  result.height = ph;
  result.width = pw;
  result.head_dim = head_dim;
  const size_t tokens = static_cast<size_t>(pf) * ph * pw;
  result.cos.resize(tokens * head_dim);
  result.sin.resize(tokens * head_dim);
  const int dims[3] = {44, 42, 42};
  for (int axis = 0; axis < 3; ++axis) {
    const int dim = dims[axis];
    const int offset = axis == 0 ? 0 : (axis == 1 ? 44 : 86);
    for (int pair = 0; pair < dim / 2; ++pair) {
      const double frequency =
          1.0 / std::pow(static_cast<double>(theta),
                         static_cast<double>(2 * pair) / static_cast<double>(dim));
      for (int f = 0; f < pf; ++f) {
        for (int y = 0; y < ph; ++y) {
          for (int x = 0; x < pw; ++x) {
            const int position = axis == 0 ? f : (axis == 1 ? y : x);
            const double angle = static_cast<double>(position) * frequency;
            const float cosine = static_cast<float>(std::cos(angle));
            const float sine = static_cast<float>(std::sin(angle));
            const size_t token = (static_cast<size_t>(f) * ph + y) * pw + x;
            const size_t index = token * head_dim + offset + pair * 2;
            result.cos[index] = cosine;
            result.cos[index + 1] = cosine;
            result.sin[index] = sine;
            result.sin[index + 1] = sine;
          }
        }
      }
    }
  }
  return result;
}

std::vector<float> wan_patchify_host(const float* latent, int channels,
                                     int frames, int height, int width) {
  if (latent == nullptr) throw std::invalid_argument("null Wan latent buffer");
  checked_count(frames, height, width, channels);
  const int ph = height / 2;
  const int pw = width / 2;
  const size_t tokens = static_cast<size_t>(frames) * ph * pw;
  std::vector<float> output(tokens * channels * 4);
  for (int f = 0; f < frames; ++f) {
    for (int y = 0; y < ph; ++y) {
      for (int x = 0; x < pw; ++x) {
        const size_t token = (static_cast<size_t>(f) * ph + y) * pw + x;
        for (int dy = 0; dy < 2; ++dy) {
          for (int dx = 0; dx < 2; ++dx) {
            for (int c = 0; c < channels; ++c) {
              const size_t source =
                  (static_cast<size_t>(c) * frames + f) * height * width +
                  static_cast<size_t>(y * 2 + dy) * width + x * 2 + dx;
              const size_t target = (token * 4 + dy * 2 + dx) * channels + c;
              output[target] = latent[source];
            }
          }
        }
      }
    }
  }
  return output;
}

std::vector<float> wan_unpatchify_host(const float* tokens, int channels,
                                       int frames, int height, int width) {
  if (tokens == nullptr) throw std::invalid_argument("null Wan token buffer");
  const size_t count = checked_count(frames, height, width, channels);
  const int ph = height / 2;
  const int pw = width / 2;
  std::vector<float> output(count);
  for (int f = 0; f < frames; ++f) {
    for (int y = 0; y < ph; ++y) {
      for (int x = 0; x < pw; ++x) {
        const size_t token = (static_cast<size_t>(f) * ph + y) * pw + x;
        for (int dy = 0; dy < 2; ++dy) {
          for (int dx = 0; dx < 2; ++dx) {
            for (int c = 0; c < channels; ++c) {
              const size_t source = (token * 4 + dy * 2 + dx) * channels + c;
              const size_t target =
                  (static_cast<size_t>(c) * frames + f) * height * width +
                  static_cast<size_t>(y * 2 + dy) * width + x * 2 + dx;
              output[target] = tokens[source];
            }
          }
        }
      }
    }
  }
  return output;
}

int wan_attention_tile_len(int tokens, int q_start, int tile_size) {
  if (tokens <= 0 || tile_size <= 0 || q_start < 0 || q_start >= tokens) {
    throw std::invalid_argument("invalid Wan DiT attention tile bounds");
  }
  return std::min(tile_size, tokens - q_start);
}

uint64_t wan_hash_text_embeds(const float* data, int text_len, int text_dim) {
  if (data == nullptr || text_len <= 0 || text_dim <= 0) {
    throw std::invalid_argument("invalid Wan DiT text embedding buffer for hashing");
  }
  const uint64_t count =
      static_cast<uint64_t>(text_len) * static_cast<uint64_t>(text_dim);
  if (count > std::numeric_limits<size_t>::max() / sizeof(float)) {
    throw std::overflow_error("Wan DiT text embedding size overflows size_t");
  }
  const size_t bytes = static_cast<size_t>(count) * sizeof(float);
  const auto* raw = reinterpret_cast<const uint8_t*>(data);

  constexpr uint64_t kFnvOffset = 14695981039346656037ull;
  constexpr uint64_t kFnvPrime = 1099511628211ull;

  uint64_t h0 = kFnvOffset ^ static_cast<uint64_t>(text_len);
  h0 *= kFnvPrime;
  h0 ^= static_cast<uint64_t>(text_dim);
  h0 *= kFnvPrime;
  uint64_t h1 = h0 ^ 0x9e3779b97f4a7c15ull;
  uint64_t h2 = h0 ^ 0xbf58476d1ce4e5b9ull;
  uint64_t h3 = h0 ^ 0x94d049bb133111ebull;

  size_t offset = 0;
  for (; offset + 32 <= bytes; offset += 32) {
    uint64_t w0 = 0;
    uint64_t w1 = 0;
    uint64_t w2 = 0;
    uint64_t w3 = 0;
    std::memcpy(&w0, raw + offset + 0, sizeof(w0));
    std::memcpy(&w1, raw + offset + 8, sizeof(w1));
    std::memcpy(&w2, raw + offset + 16, sizeof(w2));
    std::memcpy(&w3, raw + offset + 24, sizeof(w3));
    h0 = (h0 ^ w0) * kFnvPrime;
    h1 = (h1 ^ w1) * kFnvPrime;
    h2 = (h2 ^ w2) * kFnvPrime;
    h3 = (h3 ^ w3) * kFnvPrime;
  }
  for (; offset + 8 <= bytes; offset += 8) {
    uint64_t w = 0;
    std::memcpy(&w, raw + offset, sizeof(w));
    h0 = (h0 ^ w) * kFnvPrime;
  }
  for (; offset < bytes; ++offset) {
    h0 = (h0 ^ static_cast<uint64_t>(raw[offset])) * kFnvPrime;
  }

  uint64_t h = h0;
  h = (h ^ h1) * kFnvPrime;
  h = (h ^ h2) * kFnvPrime;
  h = (h ^ h3) * kFnvPrime;
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdull;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53ull;
  h ^= h >> 33;
  return h;
}

}  // namespace umminfer::video
