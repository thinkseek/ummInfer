// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

#include <vector>

#include "bf16.hpp"

namespace umminfer::omni {

// CTA tile widths of the prefill kernels; callers derive tile counts from them.
constexpr int kGemmTileN = 128;   // tensor-core GEMM / gate-up MoE tile
constexpr int kDownTileM = 16;    // FP32-activation down-projection row tile

// ---- Prefill path: tensor-core GEMM ---------------------------------------
// y[m, n] = x[m, k] * w[n, k]^T + bias[n].  `bias` may be null.  y is BF16.
// 128x128 CTA tiles, 64-deep K stages, cp.async double buffering, warp-level
// mma.sync.m16n8k16 with fp32 accumulation.
void gemm_bf16_mma(Bf16* y, const Bf16* x, const Bf16* w, const Bf16* bias,
                   int m, int k, int n, cudaStream_t stream);
// Same, fp32 output (router logits).
void gemm_bf16_mma_f32(float* y, const Bf16* x, const Bf16* w, int m, int k,
                       int n, cudaStream_t stream);

// ---- Decode path: memory-bound GEMV --------------------------------------
// One warp per output row, 16-byte vector loads, fp32 accumulation.
void gemv_bf16_f32(const Bf16* w, const Bf16* x, float* out, int n_out, int k,
                   cudaStream_t stream);
void gemv_bf16_bf16(const Bf16* w, const Bf16* x, Bf16* out, int n_out, int k,
                    cudaStream_t stream);
// Grouped GEMV: group g computes out[g*n_out + j] = wtbl[g][j] . (x + g*k).
void gemv_grouped_f32(const Bf16* const* wtbl, const Bf16* x, float* out,
                      int n_out, int k, int n_groups, cudaStream_t stream);
// Same with an FP32 activation vector (expert down projection).
void gemv_grouped_f32act(const Bf16* const* wtbl, const float* x, float* out,
                         int n_out, int k, int n_groups, cudaStream_t stream);

// ---- MoE grouped launch plan --------------------------------------------
// Routing is resolved on the host (it already is: the router logits are copied
// back per layer).  Pair p = (token, expert) with p ordered by ascending
// expert id, so each expert's pairs form one contiguous row block and the
// per-token accumulation order matches the reference ascending-expert order.
struct MoeGroupPlan {
  std::vector<int> group_off;    // first pair row of each expert
  std::vector<int> group_size;   // pairs for each expert
  std::vector<int> pair_token;   // source token row of each pair
  std::vector<int> pair_expert;  // expert of each pair
  std::vector<int> pair_of;      // [tokens * top_k] pair index, teacher-forced
  std::vector<float> pair_w;     // [tokens * top_k] routing weight
  // Flat CTA lists: gate/up use 128-row tiles, the FP32 down projection uses
  // kDownTileM-row tiles, so they cannot share one list.
  std::vector<int> blk_expert, blk_m, blk_n;
  std::vector<int> down_e, down_m, down_n;
  int total_pairs = 0;
  int tokens = 0;
  int top_k = 0;

  // Fill blk_* / down_* for a launch with `n_tiles` column tiles per expert.
  void build_blocks(int n_tiles);
  void build_down_blocks(int n_tiles);
  // Fill group_off/group_size/pair_token/pair_of/pair_w from host routing.
  void build(const int* top_idx, const float* top_w, int tokens, int top_k,
             int num_experts);
};

// Device-side copies of the plan tables.  Owned by the engine, reused.
struct MoeDeviceTables {
  const Bf16** weights = nullptr;       // [3][num_experts]: gate, up, down
  const Bf16** pair_weights = nullptr;  // [3][pairs]: same, indexed by pair
  int *group_off = nullptr, *group_size = nullptr;
  int *pair_token = nullptr, *pair_of = nullptr;
  float* pair_w = nullptr;
  int *blk_e = nullptr, *blk_m = nullptr, *blk_n = nullptr;
  int *down_e = nullptr, *down_m = nullptr, *down_n = nullptr;
  int blk_cap = 0, down_cap = 0;  // entries allocated in each list
};

// Upload the variable tables of `plan`, the expert-indexed weight pointer table
// (gate at 0, up at num_experts, down at 2*num_experts) and the pair-indexed
// one used by the batch-1 GEMV path.
void moe_upload_tables(const MoeGroupPlan& plan, const Bf16* const wg[],
                       const Bf16* const wu[], const Bf16* const wd[],
                       int num_experts, MoeDeviceTables* t, cudaStream_t stream);

// Grouped expert gate/up GEMM: y[pair, n] = x[pair, k] * weights[base+expert]^T.
void moe_gemm_grouped_f32(float* y, const Bf16* x, const MoeDeviceTables& t,
                          int base, int n_blocks, int n, int k,
                          cudaStream_t stream);
// Grouped expert down projection with the FP32 activation as the A operand.
// Same product and accumulation precision as the reference.
void moe_down_f32(float* y, const float* act, const MoeDeviceTables& t,
                  int base, int n_blocks, int n, int k, cudaStream_t stream);

// Data movement helpers used by the MoE block.
void gather_rows(Bf16* out, const Bf16* in, const int* ids, int n_rows, int dim,
                 cudaStream_t stream);
// act = silu(gate) * up, kept in FP32 (the reference down projection reads it
// in FP32; rounding it to BF16 would change the numeric semantics).
void moe_act(float* act, const float* gate, const float* up, int n,
             cudaStream_t stream);
void moe_accumulate(Bf16* acc, const float* y, const int* pair_of,
                    const float* pair_w, int tokens, int top_k, int dim,
                    cudaStream_t stream);

}  // namespace umminfer::omni
