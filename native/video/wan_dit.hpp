// SPDX-License-Identifier: Apache-2.0
// Wan2.2-TI2V-5B transformer configuration, layout helpers, and CUDA forward.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace umminfer::video {

struct WanDitConfig {
  int num_layers = 30;
  int num_attention_heads = 24;
  int attention_head_dim = 128;
  int inner_dim = 3072;
  int ffn_dim = 14336;
  int in_channels = 48;
  int out_channels = 48;
  int patch_t = 1;
  int patch_h = 2;
  int patch_w = 2;
  int text_dim = 4096;
  int freq_dim = 256;
  int rope_max_seq_len = 1024;
  float eps = 1e-6f;
  bool cross_attn_norm = true;
  std::string qk_norm = "rms_norm_across_heads";
};

// model_dir is the Diffusers model root containing transformer/config.json.
WanDitConfig load_wan_dit_config(const std::string& model_dir);

// Diffusers Timesteps: cosine half followed by sine half.
std::vector<float> wan_sinusoidal_timestep_embedding(float timestep,
                                                      int freq_dim);

struct WanRope3D {
  int frames = 0;
  int height = 0;
  int width = 0;
  int head_dim = 0;
  // [frame, y, x, head_dim], with each axis frequency repeated for a complex pair.
  std::vector<float> cos;
  std::vector<float> sin;
};

WanRope3D wan_build_rope_3d(int frames, int height, int width, int head_dim,
                            float theta = 10000.0f);

// Raw patch token layout is [token, patch_t, patch_y, patch_x, channel], exactly
// the proj_out component order before Diffusers' final permute/flatten operations.
std::vector<float> wan_patchify_host(const float* latent_ncfhw, int channels,
                                     int frames, int height, int width);
std::vector<float> wan_unpatchify_host(const float* tokens, int channels,
                                       int frames, int height, int width);

// Attention query tile size used by the CUDA Wan DiT reference attention loop.
inline constexpr int kWanDitQueryTile = 512;

// Returns min(tile_size, tokens - q_start) for attention query tiling.
int wan_attention_tile_len(int tokens, int q_start,
                           int tile_size = kWanDitQueryTile);

// BF16 Tensor-Core FlashAttention-2 tile geometry for Wan2.2-TI2V-5B (D=128).
// Two [kWanDitFlashBc, kWanDitFlashStride] BF16 shared-memory tiles occupy
// 34,816 B (< 48 KiB static smem limit), with 16-byte aligned rows and
// conflict-free ldmatrix.x2 / ldmatrix.x2.trans access.
inline constexpr int kWanDitFlashBr = 64;
inline constexpr int kWanDitFlashBc = 64;
inline constexpr int kWanDitFlashHeadDim = 128;
inline constexpr int kWanDitFlashStride = kWanDitFlashHeadDim + 8;
inline constexpr int kWanDitFlashThreads = 128;
inline constexpr size_t kWanDitFlashSmemBytes =
    static_cast<size_t>(2) * kWanDitFlashBc * kWanDitFlashStride * sizeof(uint16_t);

inline constexpr bool wan_dit_flash_attn_supported(
    int head_dim, int query_tokens, int key_tokens) noexcept {
  return head_dim == kWanDitFlashHeadDim && query_tokens > 0 && key_tokens > 0;
}

inline constexpr int wan_dit_flash_grid_q(int query_tokens) noexcept {
  return query_tokens > 0
             ? (query_tokens + kWanDitFlashBr - 1) / kWanDitFlashBr
             : 0;
}

inline constexpr int wan_dit_flash_kv_tiles(int key_tokens) noexcept {
  return key_tokens > 0
             ? (key_tokens + kWanDitFlashBc - 1) / kWanDitFlashBc
             : 0;
}

// Selects the fused BF16 Tensor-Core FlashAttention-2 kernel (default: true)
// vs the cuBLAS tiled attention reference path (false when UMM_DIT_ATTN_REF is
// set to a truthy value: "1", "true", "yes", or "on", case-insensitive).
inline bool wan_dit_use_flash_attn(const char* ref_env) noexcept {
  if (ref_env == nullptr) return true;
  const char* begin = ref_env;
  while (*begin == ' ' || *begin == '\t' || *begin == '\r' || *begin == '\n') {
    ++begin;
  }
  const char* end = begin;
  while (*end != '\0') ++end;
  while (end > begin &&
         (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r' ||
          end[-1] == '\n')) {
    --end;
  }
  const size_t len = static_cast<size_t>(end - begin);
  if (len == 0) return true;
  const auto lower = [](char c) noexcept -> char {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
  };
  if (len == 1 && begin[0] == '1') return false;
  if (len == 2 && lower(begin[0]) == 'o' && lower(begin[1]) == 'n') return false;
  if (len == 3 && lower(begin[0]) == 'y' && lower(begin[1]) == 'e' &&
      lower(begin[2]) == 's') {
    return false;
  }
  if (len == 4 && lower(begin[0]) == 't' && lower(begin[1]) == 'r' &&
      lower(begin[2]) == 'u' && lower(begin[3]) == 'e') {
    return false;
  }
  return true;
}

struct WanTextEmbedCacheKey {
  uint64_t hash = 0;
  int text_len = 0;
  int text_dim = 0;
  bool operator==(const WanTextEmbedCacheKey& other) const {
    return hash == other.hash && text_len == other.text_len &&
           text_dim == other.text_dim;
  }
};

// Fast 64-bit FNV-1a content hash over IEEE-754 bytes of [text_len, text_dim]
// plus shape dimensions, used to key the cross-attention K/V cache across steps.
uint64_t wan_hash_text_embeds(const float* data, int text_len, int text_dim);

// Opt-in TeaCache (Timestep-Embedding-Aware Residual Caching) for Wan2.2 DiT.
// Tracks conditional (branch 0) and unconditional (branch 1) trajectories
// independently so CFG branches never mix residuals.
inline constexpr int kWanTeaCacheBranches = 2;

void wan_validate_teacache_thresh(float thresh);
float wan_teacache_relative_l1(const float* cur, const float* prev,
                               size_t count);
float wan_teacache_rescale(float rel_l1);

struct TeaCacheDecision {
  bool use_cached_residual = false;
  bool save_residual = false;
  float rel_l1 = 0.0f;
  float accumulated_rel_l1 = 0.0f;
};

struct WanDitTeaCacheHint {
  int step_index = -1;
  int total_steps = 0;
  int branch = 0;  // 0 = conditional (or unguided), 1 = unconditional (CFG)
};

class TeaCacheController {
 public:
  TeaCacheController() = default;
  explicit TeaCacheController(float threshold);

  void set_threshold(float threshold);
  float threshold() const { return threshold_; }
  bool enabled() const { return threshold_ > 0.0f; }

  // Clears per-branch cached history and hit/call telemetry while preserving
  // the configured threshold.
  void reset();

  // Invalidates a single branch's cached residual (e.g., if a GPU forward
  // aborts or throws before updating the residual buffer).
  void invalidate_branch(int branch);

  // Records a forward call when TeaCache is disabled (threshold == 0.0f) so
  // telemetry reports accurate 0/N totals with zero GPU synchronization.
  void record_disabled_call();

  // Decides whether to skip all 30 transformer blocks and reuse the branch's
  // cached residual, or run full blocks and refresh the residual.
  TeaCacheDecision evaluate(int branch, int step_index, int total_steps,
                            int frames, int height, int width,
                            bool first_frame_zero_timestep,
                            const float* cur_e0, size_t e0_len);

  int hits() const { return hits_; }
  int total_calls() const { return total_calls_; }

 private:
  struct BranchState {
    bool has_prev_e0 = false;
    bool has_residual = false;
    int frames = 0;
    int height = 0;
    int width = 0;
    bool first_frame_zero_timestep = false;
    float accumulated_rel_l1 = 0.0f;
    std::vector<float> prev_e0;
  };

  float threshold_ = 0.0f;
  std::array<BranchState, kWanTeaCacheBranches> branches_{};
  int hits_ = 0;
  int total_calls_ = 0;
};

using TeaCacheState = TeaCacheController;

struct WanDitStats {
  double load_ms = 0.0;
  double forward_ms = 0.0;
  size_t peak_vram_bytes = 0;
  size_t tensor_count = 0;
  bool teacache_hit = false;
};

struct WanDitOutput {
  std::vector<float> velocity;
  WanDitStats stats;
};

#ifdef UMM_VIDEO_CUDA
class WanDit {
 public:
  WanDit(const std::filesystem::path& model_dir, int gpu);
  ~WanDit();

  WanDit(const WanDit&) = delete;
  WanDit& operator=(const WanDit&) = delete;

  // Opt-in TeaCache threshold in [0.0, 1.0]. Default 0.0 disables caching
  // completely (zero extra VRAM, zero extra sync, bit-identical uncached path).
  void set_teacache_thresh(float thresh);
  float teacache_thresh() const;
  void reset_teacache();
  int teacache_hits() const;
  int teacache_total_calls() const;

  // latent is [1,48,F,H,W], context is [text_len,4096]. The result is [1,48,F,H,W].
  // With first_frame_zero_timestep the Wan2.2-TI2V-5B per-token timestep
  // conditioning applies: tokens of latent frame 0 use t=0 and every other
  // token uses `timestep`. The default keeps the scalar T2V path unchanged.
  WanDitOutput forward(const float* latent_ncfhw, int frames, int height,
                       int width, float timestep, const float* context,
                       int text_len,
                       bool first_frame_zero_timestep = false,
                       WanDitTeaCacheHint teacache_hint = {});

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
#endif

}  // namespace umminfer::video
