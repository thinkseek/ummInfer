// SPDX-License-Identifier: Apache-2.0
#include "engine_internal.hpp"

#include <stdexcept>

namespace umminfer::omni {

const Bf16* EngineWeights::get(const std::string& name) const {
  auto it = ptr.find(name);
  if (it == ptr.end())
    throw std::runtime_error("checkpoint missing tensor: " + name);
  return it->second;
}

// The MoE layer resolves three projection pointers for each of 128 experts on
// every token. Doing that through the name map cost a string build and a hash
// lookup per pointer; the addresses never change, so resolve them once.
void EngineWeights::cache_expert_pointers() {
  const size_t n = static_cast<size_t>(kTextLayers) * kNumExperts;
  expert_gate.assign(n, nullptr);
  expert_up.assign(n, nullptr);
  expert_down.assign(n, nullptr);
  for (int l = 0; l < kTextLayers; ++l) {
    const std::string lp =
        "thinker.model.layers." + std::to_string(l) + ".mlp.experts.";
    for (int e = 0; e < kNumExperts; ++e) {
      const std::string ep = lp + std::to_string(e);
      const size_t i = static_cast<size_t>(l) * kNumExperts + e;
      expert_gate[i] = get(ep + ".gate_proj.weight");
      expert_up[i] = get(ep + ".up_proj.weight");
      expert_down[i] = get(ep + ".down_proj.weight");
    }
  }
}

size_t kv_cache_offset(int layer, bool is_v, int slot) {
  return ((((static_cast<size_t>(layer) * 2) + (is_v ? 1 : 0)) * 2048 + slot) *
          kKvHeads) * kHeadDim;
}

// Shuffle four consecutive 1152-token rows into one 4608 row, then merger MLP.
// postshuffle=true (deepstack): view first, LayerNorm over 4608.
// postshuffle=false (main): LayerNorm over 1152 first, then view.
void run_merger(const EngineWeights& w, EngineScratch& s, const Bf16* hidden,
                Bf16* out, const std::string& prefix, int tokens,
                bool postshuffle, cudaStream_t st) {
  const int merged = tokens / 4;
  Bf16* sh = s.v_shuffle;
  if (!postshuffle) {
    // Main merger: LayerNorm(1152) on the unshuffled rows first, then view.
    layer_norm(hidden, w.get(prefix + ".ln_q.weight"),
               w.get(prefix + ".ln_q.bias"), s.v_norm, tokens, kVisionH, 1e-6f, st);
    for (int b = 0; b < merged; ++b)
      cudaMemcpyAsync(sh + static_cast<size_t>(b) * 4 * kVisionH,
                      s.v_norm + static_cast<size_t>(b * 4) * kVisionH,
                      4 * kVisionH * sizeof(Bf16), cudaMemcpyDeviceToDevice, st);
  } else {
    // Deepstack: pixel-shuffle first, then LayerNorm over the full 4608.
    for (int b = 0; b < merged; ++b)
      cudaMemcpyAsync(sh + static_cast<size_t>(b) * 4 * kVisionH,
                      hidden + static_cast<size_t>(b * 4) * kVisionH,
                      4 * kVisionH * sizeof(Bf16), cudaMemcpyDeviceToDevice, st);
    layer_norm(sh, w.get(prefix + ".ln_q.weight"),
               w.get(prefix + ".ln_q.bias"), sh, merged, 4 * kVisionH, 1e-6f, st);
  }
  constexpr int kH4 = 4 * kVisionH;  // 4608
  gemm_bf16_mma(s.v_proj, s.v_shuffle, w.get(prefix + ".mlp.0.weight"),
                w.get(prefix + ".mlp.0.bias"), merged, kH4, kH4, st);
  gelu_inplace(s.v_proj, merged * kH4, st);  // merger uses exact erf GELU
  gemm_bf16_mma(out, s.v_proj, w.get(prefix + ".mlp.2.weight"),
                w.get(prefix + ".mlp.2.bias"), merged, kH4, kVisionOut, st);
}

}  // namespace umminfer::omni
