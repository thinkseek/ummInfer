// SPDX-License-Identifier: Apache-2.0
// Pure host/device dispatch, swizzle, window-bounds, and frontend helpers for
// Qwen3-ASR. Kept free of CUDA runtime headers so offline host unit tests can
// verify every CTA swizzle bijection, row mapping, and boundary condition.
#pragma once

#include <cstddef>

#ifdef __CUDACC__
#define UMM_ASR_HD __host__ __device__ __forceinline__
#else
#define UMM_ASR_HD inline
#endif

namespace umminfer::native {

// Proven SM90 K-unroll split for decode GEMV: m = 1..2 are latency-bound and
// benefit from four 128-bit streaming weight loads in flight; larger m are
// issue-bound.
UMM_ASR_HD constexpr int asr_gemv_unroll_for(int m) { return m <= 2 ? 4 : 1; }

// Dual-weight SwiGLU GEMV loads both gate and up rows per K step, so kUnroll=2
// already keeps four 128-bit weight loads in flight at m = 1..2.
UMM_ASR_HD constexpr int asr_swiglu_unroll_for(int m) { return m <= 2 ? 2 : 1; }

// Use SM90 64x64x64 BF16 Tensor Core MMA when m >= 16 and shapes allow 128-bit
// cp.async K-staging plus bf162 stores. Smaller m stays on 1-warp/row GEMV.
UMM_ASR_HD constexpr bool asr_use_mma_gemm(int m, int n, int k) {
  return m >= 16 && (k & 7) == 0 && (n & 1) == 0 && k > 0 && n > 0;
}

// Use SM90 implicit-GEMM BF16 Tensor Core Conv2d 3x3s2 when both channel dims
// are large enough to amortize MMA tile staging (conv2d2/conv2d3: 480->480).
UMM_ASR_HD constexpr bool asr_use_mma_conv(int ci, int co) {
  return ci >= 16 && co >= 16;
}

struct AsrGemmTileCoord {
  int tile_n;
  int tile_m;
};

// Bijective L2-friendly CTA swizzle over [0, grid_x) x [0, grid_y).
// Groups up to `group_m` M-tiles together before advancing along N so each
// weight tile loaded into L2 is reused across multiple M-tiles during prefill
// and audio-encoder forward passes.
UMM_ASR_HD AsrGemmTileCoord asr_swizzle_gemm_cta(int block_x, int block_y,
                                                 int grid_x, int grid_y,
                                                 int group_m = 8) {
  if (grid_y <= 1 || group_m <= 1) return AsrGemmTileCoord{block_x, block_y};
  const int linear = block_y * grid_x + block_x;
  const int group_size = group_m * grid_x;
  const int group_id = linear / group_size;
  const int first_m = group_id * group_m;
  const int rem_m = grid_y - first_m;
  const int actual_gm = rem_m < group_m ? rem_m : group_m;
  const int in_group = linear - group_id * group_size;
  return AsrGemmTileCoord{in_group / actual_gm, first_m + (in_group % actual_gm)};
}

struct AsrMultiGemvRowLoc {
  int proj;
  int local_row;
};

// Maps a flat output row in [0, sum(n_outs)) across up to 3 fused projections
// sharing the same input vector x (e.g. Q, K, V). Returns {-1, -1} out of range.
UMM_ASR_HD AsrMultiGemvRowLoc asr_multi_gemv_locate_row(int global_row,
                                                        const int* n_outs,
                                                        int count) {
  if (global_row < 0) return AsrMultiGemvRowLoc{-1, -1};
  int r = global_row;
  for (int p = 0; p < count; ++p) {
    if (r < n_outs[p]) return AsrMultiGemvRowLoc{p, r};
    r -= n_outs[p];
  }
  return AsrMultiGemvRowLoc{-1, -1};
}

struct AsrAttnWindow {
  int lo;
  int hi;
};

// Bidirectional non-overlapping fixed-window attention bounds for the Whisper-
// style audio tower (window_len = 104 for both Qwen3-ASR 0.6B and 1.7B).
UMM_ASR_HD AsrAttnWindow asr_audio_attn_window_bounds(int qi, int nkv,
                                                      int window_len) {
  if (window_len <= 0 || nkv <= 0) return AsrAttnWindow{0, 0};
  const int wstart = (qi / window_len) * window_len;
  const int wend = wstart + window_len;
  return AsrAttnWindow{wstart, wend < nkv ? wend : nkv};
}

// Number of leading STFT frames (out of k_chunk_frames) that can touch any
// non-zero sample in a zero-padded 30s waveform of `real_samples` samples.
// Frames at t >= active_frames see exclusively zero samples and therefore
// produce exact log10(1e-10) = -10.0 across all mel bands.
UMM_ASR_HD int asr_mel_active_frames(int real_samples, int k_chunk_frames,
                                     int k_nfft, int k_hop) {
  if (real_samples <= 0 || k_chunk_frames <= 0 || k_hop <= 0) return 0;
  const int n = (real_samples + (k_nfft / 2) + k_hop - 1) / k_hop;
  return n < k_chunk_frames ? n : k_chunk_frames;
}

}  // namespace umminfer::native

#undef UMM_ASR_HD
