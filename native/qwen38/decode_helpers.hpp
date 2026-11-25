// SPDX-License-Identifier: Apache-2.0
// Pure host/device dispatch, swizzle, and RoPE helpers for Qwen3.8-27B.
// Kept free of CUDA runtime headers so host unit tests can verify every
// index bijection, row mapping, and RoPE table calculation directly.
#pragma once

#include <cmath>
#include <cstddef>

#include "bf16.hpp"

#ifdef __CUDACC__
#define UMM_QWEN38_HD __host__ __device__ __forceinline__
#else
#define UMM_QWEN38_HD inline
#endif

namespace umminfer::qwen38 {

inline constexpr int kTextRotaryDim = 64;
inline constexpr int kTextRotaryHalf = kTextRotaryDim / 2;
inline constexpr float kTextRopeTheta = 1e7f;

// Measured K-unroll split for decode GEMV: m = 1..2 are latency-bound and need
// four 128-bit weight loads in flight; m >= 3 are issue-bound.
UMM_QWEN38_HD constexpr int gemv_unroll_for(int m) { return m <= 2 ? 4 : 1; }

// Dual-weight SwiGLU GEMV loads both gate and up rows per K step, so kUnroll=2
// already places four 128-bit weight loads in flight at m = 1..2.
UMM_QWEN38_HD constexpr int swiglu_gemv_unroll_for(int m) {
  return m <= 2 ? 2 : 1;
}

struct GemmTileCoord {
  int tile_n;
  int tile_m;
};

// Bijective L2-friendly CTA swizzle over [0, grid_x) x [0, grid_y).
// Groups up to `group_m` M-tiles together before advancing along N so each
// weight tile loaded into L2 is reused across multiple M-tiles during prefill.
UMM_QWEN38_HD GemmTileCoord swizzle_gemm_cta(int block_x, int block_y,
                                             int grid_x, int grid_y,
                                             int group_m = 8) {
  if (grid_y <= 1 || group_m <= 1) return GemmTileCoord{block_x, block_y};
  const int linear = block_y * grid_x + block_x;
  const int group_size = group_m * grid_x;
  const int group_id = linear / group_size;
  const int first_m = group_id * group_m;
  const int rem_m = grid_y - first_m;
  const int actual_gm = rem_m < group_m ? rem_m : group_m;
  const int in_group = linear - group_id * group_size;
  return GemmTileCoord{in_group / actual_gm, first_m + (in_group % actual_gm)};
}

struct MultiGemvRowLoc {
  int proj;
  int local_row;
};

// Maps a flat output row in [0, sum(n_outs)) across up to 4 fused projections
// sharing the same input vector x. Returns {-1, -1} if out of range.
UMM_QWEN38_HD MultiGemvRowLoc multi_gemv_locate_row(int global_row,
                                                    const int* n_outs,
                                                    int count) {
  if (global_row < 0) return MultiGemvRowLoc{-1, -1};
  int r = global_row;
  for (int p = 0; p < count; ++p) {
    if (r < n_outs[p]) return MultiGemvRowLoc{p, r};
    r -= n_outs[p];
  }
  return MultiGemvRowLoc{-1, -1};
}

// Precomputes the 32 inverse frequencies for Qwen3.8 partial mRoPE once per
// engine instead of calling std::pow 32 times on every decode token.
inline void init_text_rope_inv_freq(double (&inv)[kTextRotaryHalf]) {
  for (int i = 0; i < kTextRotaryHalf; ++i) {
    inv[i] = 1.0 / std::pow(static_cast<double>(kTextRopeTheta),
                            static_cast<double>(2 * i) / kTextRotaryDim);
  }
}

// Computes one BF16 (cos, sin) pair for Qwen3.8 partial mRoPE from a
// precomputed fp64 inverse frequency and integer position. Shared by both host
// prefill/fallback tables and the on-device CUDA Graph decode mRoPE kernel so
// both paths match bit for bit.
UMM_QWEN38_HD void text_rope_cos_sin_elem(double inv_i, int pos, Bf16* cos_out,
                                          Bf16* sin_out) {
  const double angle = inv_i * static_cast<double>(pos);
#ifdef __CUDA_ARCH__
  *cos_out = __float2bfloat16(static_cast<float>(cos(angle)));
  *sin_out = __float2bfloat16(static_cast<float>(sin(angle)));
#else
  *cos_out = Bf16(static_cast<float>(std::cos(angle)));
  *sin_out = Bf16(static_cast<float>(std::sin(angle)));
#endif
}

// Fills one [kTextRotaryDim] row of cos/sin tables using precomputed inv_freq.
// Channel rule matches build_text_rope: H occupies i%3==1 (i<33), W occupies
// i%3==2 (i<30), everything else uses T; upper half [32..64) is zeroed.
inline void fill_text_rope_row(const double (&inv)[kTextRotaryHalf], int pos_t,
                               int pos_h, int pos_w, Bf16* cos_row,
                               Bf16* sin_row) {
  for (int i = 0; i < kTextRotaryHalf; ++i) {
    int pos = pos_t;
    if ((i % 3) == 1 && i < 33)
      pos = pos_h;
    else if ((i % 3) == 2 && i < 30)
      pos = pos_w;
    text_rope_cos_sin_elem(inv[i], pos, &cos_row[i], &sin_row[i]);
  }
  for (int i = kTextRotaryHalf; i < kTextRotaryDim; ++i) {
    cos_row[i] = Bf16(0.0f);
    sin_row[i] = Bf16(0.0f);
  }
}

// Scalar-position decode helper (pos_t == pos_h == pos_w == pos) mirroring
// compute_text_mrope_devpos_kernel for host-side parity tests.
inline void fill_text_rope_row_scalar_pos(const double (&inv)[kTextRotaryHalf],
                                          int pos, Bf16* cos_row,
                                          Bf16* sin_row) {
  for (int i = 0; i < kTextRotaryHalf; ++i) {
    text_rope_cos_sin_elem(inv[i], pos, &cos_row[i], &sin_row[i]);
  }
  for (int i = kTextRotaryHalf; i < kTextRotaryDim; ++i) {
    cos_row[i] = Bf16(0.0f);
    sin_row[i] = Bf16(0.0f);
  }
}

// Policy helper deciding whether m == 1 CUDA Graph decode capture should be
// enabled. Opt-out via UMM_QWEN38_NO_GRAPH (any non-empty value other than
// "0") or whenever parity dumping is active.
inline bool qwen38_use_cuda_graph(const char* no_graph_env, bool dump_enabled) {
  if (dump_enabled) return false;
  if (no_graph_env != nullptr && no_graph_env[0] != '\0' &&
      !(no_graph_env[0] == '0' && no_graph_env[1] == '\0')) {
    return false;
  }
  return true;
}

}  // namespace umminfer::qwen38

#undef UMM_QWEN38_HD
