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

std::vector<float> wan_cfg_unpatchify_host(const float* cond_tokens,
                                           const float* uncond_tokens,
                                           int channels, int frames, int height,
                                           int width, float guidance_scale) {
  if (cond_tokens == nullptr || uncond_tokens == nullptr) {
    throw std::invalid_argument("null Wan CFG token buffer");
  }
  if (!std::isfinite(guidance_scale)) {
    throw std::invalid_argument("Wan CFG guidance_scale must be finite");
  }
  const size_t count = checked_count(frames, height, width, channels);
  std::vector<float> output(count);
  for (size_t i = 0; i < count; ++i) {
    const size_t src =
        wan_dit_ncfhw_to_patch_index(i, channels, frames, height, width);
    const float c = cond_tokens[src];
    const float u = uncond_tokens[src];
    output[i] = u + guidance_scale * (c - u);
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

void wan_validate_teacache_thresh(float thresh) {
  if (!std::isfinite(thresh) || thresh < 0.0f || thresh > 1.0f) {
    throw std::invalid_argument(
        "Wan TeaCache threshold must be finite and in [0.0, 1.0]");
  }
}

float wan_teacache_relative_l1(const float* cur, const float* prev,
                               size_t count) {
  if (cur == nullptr || prev == nullptr || count == 0) {
    throw std::invalid_argument(
        "Wan TeaCache relative L1 requires non-empty buffers");
  }
  double diff_sum = 0.0;
  double prev_sum = 0.0;
  for (size_t i = 0; i < count; ++i) {
    diff_sum +=
        std::fabs(static_cast<double>(cur[i]) - static_cast<double>(prev[i]));
    prev_sum += std::fabs(static_cast<double>(prev[i]));
  }
  const double denom = std::max(prev_sum, 1e-6);
  const double ratio = diff_sum / denom;
  if (!std::isfinite(ratio)) {
    return std::numeric_limits<float>::infinity();
  }
  return static_cast<float>(ratio);
}

float wan_teacache_rescale(float rel_l1) {
  if (!std::isfinite(rel_l1)) {
    return std::numeric_limits<float>::infinity();
  }
  return std::max(0.0f, rel_l1);
}

TeaCacheController::TeaCacheController(float threshold) {
  set_threshold(threshold);
}

void TeaCacheController::set_threshold(float threshold) {
  wan_validate_teacache_thresh(threshold);
  threshold_ = threshold;
  reset();
}

void TeaCacheController::reset() {
  for (auto& branch : branches_) {
    branch = BranchState{};
  }
  hits_ = 0;
  total_calls_ = 0;
}

void TeaCacheController::invalidate_branch(int branch) {
  if (branch < 0 || branch >= kWanTeaCacheBranches) {
    throw std::invalid_argument("Wan TeaCache branch must be 0 or 1");
  }
  branches_[static_cast<size_t>(branch)] = BranchState{};
}

void TeaCacheController::record_disabled_call() { ++total_calls_; }

TeaCacheDecision TeaCacheController::evaluate(
    int branch, int step_index, int total_steps, int frames, int height,
    int width, bool first_frame_zero_timestep, const float* cur_e0,
    size_t e0_len) {
  if (branch < 0 || branch >= kWanTeaCacheBranches) {
    throw std::invalid_argument("Wan TeaCache branch must be 0 or 1");
  }
  ++total_calls_;
  TeaCacheDecision decision;
  if (threshold_ <= 0.0f) {
    return decision;
  }
  if (cur_e0 == nullptr || e0_len == 0) {
    throw std::invalid_argument(
        "Wan TeaCache modulation signal must be non-empty");
  }
  if (frames <= 0 || height <= 0 || width <= 0) {
    throw std::invalid_argument("Wan TeaCache geometry must be positive");
  }

  BranchState& state = branches_[static_cast<size_t>(branch)];

  // Unscheduled / standalone forward or <= 2 steps: always compute full blocks.
  if (step_index < 0 || total_steps <= 2 || step_index >= total_steps) {
    state = BranchState{};
    return decision;
  }

  // First step: always compute full blocks and populate the branch residual.
  if (step_index == 0) {
    state.frames = frames;
    state.height = height;
    state.width = width;
    state.first_frame_zero_timestep = first_frame_zero_timestep;
    state.accumulated_rel_l1 = 0.0f;
    state.prev_e0.assign(cur_e0, cur_e0 + e0_len);
    state.has_prev_e0 = true;
    state.has_residual = true;
    decision.use_cached_residual = false;
    decision.save_residual = true;
    return decision;
  }

  // Final step: always compute full blocks; no later step will reuse it.
  if (step_index == total_steps - 1) {
    state.accumulated_rel_l1 = 0.0f;
    state.prev_e0.assign(cur_e0, cur_e0 + e0_len);
    state.has_prev_e0 = true;
    state.has_residual = false;
    decision.use_cached_residual = false;
    decision.save_residual = false;
    return decision;
  }

  // Intermediate step: verify branch cache exists and matches shape/mode.
  const bool cache_compatible =
      state.has_residual && state.has_prev_e0 && state.frames == frames &&
      state.height == height && state.width == width &&
      state.first_frame_zero_timestep == first_frame_zero_timestep &&
      state.prev_e0.size() == e0_len;
  if (!cache_compatible) {
    state.frames = frames;
    state.height = height;
    state.width = width;
    state.first_frame_zero_timestep = first_frame_zero_timestep;
    state.accumulated_rel_l1 = 0.0f;
    state.prev_e0.assign(cur_e0, cur_e0 + e0_len);
    state.has_prev_e0 = true;
    state.has_residual = true;
    decision.use_cached_residual = false;
    decision.save_residual = true;
    return decision;
  }

  const float rel_l1 =
      wan_teacache_relative_l1(cur_e0, state.prev_e0.data(), e0_len);
  const float delta = wan_teacache_rescale(rel_l1);
  state.accumulated_rel_l1 += delta;
  state.prev_e0.assign(cur_e0, cur_e0 + e0_len);
  decision.rel_l1 = rel_l1;

  if (state.accumulated_rel_l1 < threshold_) {
    ++hits_;
    decision.use_cached_residual = true;
    decision.save_residual = false;
    decision.accumulated_rel_l1 = state.accumulated_rel_l1;
    return decision;
  }

  state.accumulated_rel_l1 = 0.0f;
  state.has_residual = true;
  decision.use_cached_residual = false;
  decision.save_residual = true;
  decision.accumulated_rel_l1 = 0.0f;
  return decision;
}

WanDitCfgTeaCachePlan wan_dit_evaluate_cfg_teacache(
    TeaCacheController& controller, WanDitTeaCacheHint cond_hint,
    WanDitTeaCacheHint uncond_hint, int frames, int height, int width,
    bool first_frame_zero_timestep, const float* cur_e0, size_t e0_len) {
  WanDitCfgTeaCachePlan plan;
  if (!controller.enabled()) {
    controller.record_disabled_call();
    controller.record_disabled_call();
    plan.mode = WanDitCfgDispatchMode::kBatchedBothMiss;
    return plan;
  }
  if (cond_hint.branch == uncond_hint.branch) {
    throw std::invalid_argument(
        "Wan CFG TeaCache cond and uncond branches must be distinct");
  }
  plan.cond = controller.evaluate(cond_hint.branch, cond_hint.step_index,
                                  cond_hint.total_steps, frames, height, width,
                                  first_frame_zero_timestep, cur_e0, e0_len);
  plan.uncond = controller.evaluate(
      uncond_hint.branch, uncond_hint.step_index, uncond_hint.total_steps,
      frames, height, width, first_frame_zero_timestep, cur_e0, e0_len);
  plan.mode = wan_dit_plan_cfg_dispatch(plan.cond.use_cached_residual,
                                        plan.uncond.use_cached_residual);
  return plan;
}

}  // namespace umminfer::video
