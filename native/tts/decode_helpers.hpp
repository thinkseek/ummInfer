// SPDX-License-Identifier: Apache-2.0
// Pure host/device dispatch, index-mapping, and stateful-codec helpers for
// Qwen3-TTS. Kept free of CUDA runtime headers so offline host unit tests can
// verify every row mapping, causal history boundary, and rolling KV plan.
#pragma once

#include <cstddef>

#ifdef __CUDACC__
#define UMM_TTS_HD __host__ __device__ __forceinline__
#else
#define UMM_TTS_HD inline
#endif

namespace umminfer::native {

// Proven SM90 K-unroll split for decode GEMV: m = 1..2 are latency-bound and
// benefit from four 128-bit streaming weight loads in flight; larger m are
// issue-bound.
UMM_TTS_HD constexpr int tts_gemv_unroll_for(int m) { return m <= 2 ? 4 : 1; }

// Dual-weight SwiGLU GEMV loads both gate and up rows per K step, so kUnroll=2
// already keeps four 128-bit weight loads in flight at m = 1..2.
UMM_TTS_HD constexpr int tts_swiglu_unroll_for(int m) { return m <= 2 ? 2 : 1; }

struct MultiGemvRowLoc {
  int proj;
  int local_row;
};

// Maps a flat output row in [0, sum(n_outs)) across up to 3 fused projections
// sharing the same input vector x (e.g. Q, K, V). Returns {-1, -1} out of range.
UMM_TTS_HD MultiGemvRowLoc multi_gemv_locate_row(int global_row,
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

struct CausalSrcLoc {
  bool from_hist;  // true -> read hist[idx], false -> read in[idx]
  int idx;
};

// Maps causal Conv1d tap (t, kk) with left pad = (kernel - 1) * dilation to
// either the saved history buffer [0, pad) or the current chunk [0, L).
// Returns {-1} when out of range (only possible if pad == 0 or t >= L).
UMM_TTS_HD CausalSrcLoc causal_conv_src_lookup(int t, int kk, int dilation,
                                               int pad) {
  const int src_t = t - pad + kk * dilation;
  if (src_t < 0) {
    return CausalSrcLoc{true, pad + src_t};
  }
  return CausalSrcLoc{false, src_t};
}

// For updating causal Conv1d history [0, pad) after processing a chunk of length
// `chunk_len` (>= 1): returns whether new_hist[j] comes from current chunk `in`
// (from_hist == false) or shifts from old `hist` (from_hist == true, when
// chunk_len < pad).
UMM_TTS_HD CausalSrcLoc causal_conv_hist_update_src(int j, int pad,
                                                    int chunk_len) {
  const int s = chunk_len - pad + j;
  if (s >= 0) {
    return CausalSrcLoc{false, s};
  }
  return CausalSrcLoc{true, pad + s};
}

// For stateful causal ConvTranspose1d with kernel = 2 * stride (or stride):
// output index `o` in [0, L * stride) decomposes as o = t_local * stride + p,
// and tap `step` (kk = p + step * stride) reads relative input index
// `t_local - step`. For step == 1 and t_local == 0, reads the single saved
// previous input sample (hist[0]).
UMM_TTS_HD CausalSrcLoc transconv_step_src(int t_local, int step) {
  const int t_rel = t_local - step;
  if (t_rel < 0) {
    return CausalSrcLoc{true, 0};
  }
  return CausalSrcLoc{false, t_rel};
}

struct RollingKvPlan {
  bool need_shift;
  int keep_len;   // number of trailing slots preserved after shift
  int shift_src;  // starting slot copied down to slot 0
};

// Plans whether the pre-transformer rolling KV cache (capacity `cap`, sliding
// window `window`) needs to compact before appending `chunk_len` new frames.
// When `cap >= 2 * (window - 1) + max_chunk`, the compacted copy [shift_src,
// shift_src + keep_len) -> [0, keep_len) is strictly non-overlapping.
UMM_TTS_HD RollingKvPlan rolling_kv_plan(int cached, int chunk_len, int window,
                                         int cap) {
  if (cached + chunk_len <= cap) {
    return RollingKvPlan{false, cached, 0};
  }
  const int keep = (window > 1 && cached > window - 1) ? (window - 1) : cached;
  const int src = cached - keep;
  return RollingKvPlan{true, keep, src};
}

// Exact peak F32 elements across all Qwen3-TTS 12Hz codec decoder stages for a
// chunk of `chunk_frames` frames: reached at decoder block 3 (96 ch x 1920*T).
UMM_TTS_HD constexpr size_t codec_peak_scratch_elems(int chunk_frames) {
  return static_cast<size_t>(chunk_frames) * 1920u * 96u;
}

}  // namespace umminfer::native

#undef UMM_TTS_HD
