// SPDX-License-Identifier: Apache-2.0
#include "engine_internal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace umminfer::omni {
// Build the text mRoPE cos/sin table [n, 128] from the three position grids.
// Channel rule over the 64 half-slots: i<60 i%3==1 -> H, i%3==2 -> W, else T.
void build_text_rope_table(const int pos_t[], const int pos_h[],
                           const int pos_w[], int n, EngineScratch& s) {
  cudaStream_t st = s.stream;
  float* cos = s.mrope_cos;
  float* sin = s.mrope_sin;
  std::vector<float> inv(kHeadDim / 2);
  for (int i = 0; i < kHeadDim / 2; ++i)
    inv[i] = 1.0f / std::pow(kTextRopeTheta, 2.0f * i / kHeadDim);
  std::vector<float> hcos(static_cast<size_t>(n) * kHeadDim);
  std::vector<float> hsin(hcos.size());
  for (int p = 0; p < n; ++p) {
    for (int i = 0; i < kHeadDim / 2; ++i) {
      const int pid = (i < 60 && i % 3 == 1) ? pos_h[p]
                    : (i < 60 && i % 3 == 2) ? pos_w[p]
                                             : pos_t[p];
      const float angle = inv[i] * pid;
      hcos[static_cast<size_t>(p) * kHeadDim + i] = std::cos(angle);
      hsin[static_cast<size_t>(p) * kHeadDim + i] = std::sin(angle);
    }
    for (int i = 0; i < kHeadDim / 2; ++i) {
      hcos[static_cast<size_t>(p) * kHeadDim + 64 + i] =
          hcos[static_cast<size_t>(p) * kHeadDim + i];
      hsin[static_cast<size_t>(p) * kHeadDim + 64 + i] =
          hsin[static_cast<size_t>(p) * kHeadDim + i];
    }
  }
  cudaMemcpyAsync(cos, hcos.data(), hcos.size() * sizeof(float),
                  cudaMemcpyHostToDevice, st);
  cudaMemcpyAsync(sin, hsin.data(), hsin.size() * sizeof(float),
                  cudaMemcpyHostToDevice, st);
}

// One text decoder layer over m tokens (prefill m=S, decode m=1).
// history: KV slots already occupied. vis_pos/n_vis/deep_idx drive deepstack.
void text_layer(const EngineWeights& w, EngineScratch& s, int layer, int m,
                int history) {
  cudaStream_t st = s.stream;
  const std::string lp = "thinker.model.layers." + std::to_string(layer);
  const bool one = (m == 1);

  // Batch-1 tokens are bound by weight bytes, so they take the GEMV path;
  // prefill takes the tensor-core GEMM.
  auto lin = [&](Bf16* y, const Bf16* x, const Bf16* wt, int k, int n) {
    if (one)
      gemv_bf16_bf16(wt, x, y, n, k, st);
    else
      gemm_bf16_mma(y, x, wt, nullptr, m, k, n, st);
  };

  rms_norm(s.hidden, w.get(lp + ".input_layernorm.weight"), s.norm, m,
           kTextH, kRmsEps, st);
  lin(s.q, s.norm, w.get(lp + ".self_attn.q_proj.weight"), kTextH,
      kQHeads * kHeadDim);
  lin(s.k, s.norm, w.get(lp + ".self_attn.k_proj.weight"), kTextH,
      kKvHeads * kHeadDim);
  lin(s.v, s.norm, w.get(lp + ".self_attn.v_proj.weight"), kTextH,
      kKvHeads * kHeadDim);
  rms_norm(s.q, w.get(lp + ".self_attn.q_norm.weight"), s.q,
           m * kQHeads, kHeadDim, kRmsEps, st);
  rms_norm(s.k, w.get(lp + ".self_attn.k_norm.weight"), s.k,
           m * kKvHeads, kHeadDim, kRmsEps, st);
  mrope_half(s.q, s.mrope_cos, s.mrope_sin, m, kQHeads, kHeadDim, st);
  mrope_half(s.k, s.mrope_cos, s.mrope_sin, m, kKvHeads, kHeadDim, st);

  const size_t slot_bytes =
      static_cast<size_t>(m) * kKvHeads * kHeadDim * sizeof(Bf16);
  cudaMemcpyAsync(s.kv + kv_cache_offset(layer, false, history), s.k, slot_bytes,
                  cudaMemcpyDeviceToDevice, st);
  cudaMemcpyAsync(s.kv + kv_cache_offset(layer, true, history), s.v, slot_bytes,
                  cudaMemcpyDeviceToDevice, st);
  gqa_attention(s.q, s.kv + kv_cache_offset(layer, false, 0),
                s.kv + kv_cache_offset(layer, true, 0), s.norm, m,
                history + m, kQHeads, kKvHeads, kHeadDim, true, 0, history, st);
  lin(s.q, s.norm, w.get(lp + ".self_attn.o_proj.weight"), kQHeads * kHeadDim,
      kTextH);
  add_residual(s.hidden, s.q, m * kTextH, st);

  // ---- MoE block (every layer) ----
  rms_norm(s.hidden, w.get(lp + ".post_attention_layernorm.weight"), s.norm, m,
           kTextH, kRmsEps, st);
  const Bf16* gate_w = w.get(lp + ".mlp.gate.weight");
  if (one)
    gemv_bf16_f32(gate_w, s.norm, s.router_logits, kNumExperts, kTextH, st);
  else
    gemm_bf16_mma_f32(s.router_logits, s.norm, gate_w, m, kTextH, kNumExperts,
                      st);
  moe_route(s.router_logits, s.router_idx, s.router_w, m, kNumExperts, kTopK, st);

  // The reference implementation resolves experts on the host per layer, so
  // this round trip is not new; it is what makes the grouped launch possible.
  // The route buffers live in the scratch object: reallocating two vectors per
  // layer per token was measurable host time.
  const size_t route_n = static_cast<size_t>(m) * kTopK;
  if (s.h_route_idx.size() < route_n) {
    s.h_route_idx.resize(route_n);
    s.h_route_w.resize(route_n);
  }
  int* h_idx = s.h_route_idx.data();
  float* h_w = s.h_route_w.data();
  cudaMemcpyAsync(h_idx, s.router_idx, route_n * sizeof(int),
                  cudaMemcpyDeviceToHost, st);
  cudaMemcpyAsync(h_w, s.router_w, route_n * sizeof(float),
                  cudaMemcpyDeviceToHost, st);
  cudaStreamSynchronize(st);

  MoeGroupPlan& plan = s.moe_plan;
  plan.build(h_idx, h_w, m, kTopK, kNumExperts);
  // Expert pointers are resolved once at load time.
  const size_t eb = static_cast<size_t>(layer) * kNumExperts;
  const Bf16* const* wg = w.expert_gate.data() + eb;
  const Bf16* const* wu = w.expert_up.data() + eb;
  const Bf16* const* wd = w.expert_down.data() + eb;
  const int pairs = plan.total_pairs;
  const int gate_blocks = (plan.build_blocks(kExpertInter / kGemmTileN),
                           static_cast<int>(plan.blk_expert.size()));
  const int down_blocks = (plan.build_down_blocks(kTextH / kGemmTileN),
                           static_cast<int>(plan.down_e.size()));
  moe_upload_tables(plan, wg, wu, wd, kNumExperts, &s.moe_tab, st);

  // Expert inputs in pair order (rows of the same token repeat once per expert).
  gather_rows(s.moe_x, s.norm, s.moe_tab.pair_token, pairs, kTextH, st);
  if (one) {
    gemv_grouped_f32(s.moe_tab.pair_weights, s.moe_x, s.moe_gate, kExpertInter,
                     kTextH, pairs, st);
    gemv_grouped_f32(s.moe_tab.pair_weights + pairs, s.moe_x, s.moe_up,
                     kExpertInter, kTextH, pairs, st);
    moe_act(s.moe_act, s.moe_gate, s.moe_up, pairs * kExpertInter, st);
    gemv_grouped_f32act(s.moe_tab.pair_weights + 2 * pairs, s.moe_act,
                        s.moe_y, kTextH, kExpertInter, pairs, st);
  } else {
    moe_gemm_grouped_f32(s.moe_gate, s.moe_x, s.moe_tab, 0, gate_blocks,
                         kExpertInter, kTextH, st);
    moe_gemm_grouped_f32(s.moe_up, s.moe_x, s.moe_tab, kNumExperts, gate_blocks,
                         kExpertInter, kTextH, st);
    moe_act(s.moe_act, s.moe_gate, s.moe_up, pairs * kExpertInter, st);
    moe_down_f32(s.moe_y, s.moe_act, s.moe_tab, 2 * kNumExperts, down_blocks,
                 kTextH, kExpertInter, st);
  }
  // Combine in ascending expert order, rounding to bf16 after each add, exactly
  // as the reference single-expert accumulator chain did.
  moe_accumulate(s.moe_acc, s.moe_y, s.moe_tab.pair_of, s.moe_tab.pair_w, m,
                 kTopK, kTextH, st);
  add_residual(s.hidden, s.moe_acc, m * kTextH, st);
}

// Build [3][n] position ids for the full prompt. Image region (if any) is the
// contiguous run of image_pad tokens; grid gives its (grid_h, grid_w).
PositionIds build_positions(const std::vector<int>& ids, int image_pad_id,
                            int vision_start_id, int vision_end_id,
                            int grid_h, int grid_w, bool has_image) {
  const int n = static_cast<int>(ids.size());
  PositionIds p{std::vector<int>(n), std::vector<int>(n), std::vector<int>(n), 0};
  if (!has_image) {
    for (int i = 0; i < n; ++i) p.t[i] = p.h[i] = p.w[i] = i;
    p.next_pos = n;
    return p;
  }
  const int img_start =
      std::find(ids.begin(), ids.end(), image_pad_id) - ids.begin();
  int img_end = img_start;
  while (img_end < n && ids[img_end] == image_pad_id) ++img_end;
  const int n_img = img_end - img_start;
  const int llm_gh = grid_h / 2, llm_gw = grid_w / 2;
  if (n_img != llm_gh * llm_gw)
    throw std::runtime_error("image token count does not match grid");
  int counter = 0;
  for (int i = 0; i < img_start; ++i) p.t[i] = p.h[i] = p.w[i] = counter++;
  // vision_start is the token immediately before the pads and already counted.
  const int st_idx = counter;  // first image position
  for (int k = 0; k < n_img; ++k) {
    p.t[img_start + k] = st_idx;
    p.h[img_start + k] = st_idx + k / llm_gw;
    p.w[img_start + k] = st_idx + k % llm_gw;
  }
  int grid_max = st_idx + std::max(llm_gh - 1, llm_gw - 1);
  counter = grid_max + 1;
  for (int i = img_end; i < n; ++i) p.t[i] = p.h[i] = p.w[i] = counter++;
  p.next_pos = counter;
  return p;
}

}  // namespace umminfer::omni
