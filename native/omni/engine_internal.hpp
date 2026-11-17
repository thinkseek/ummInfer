// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "bf16.hpp"
#include "cuda_kernels_omni.hpp"
#include "image_preprocess.hpp"
#include "omni_engine.hpp"
#include "safetensors_mmap.hpp"

namespace umminfer::omni {

// Frozen architecture of the Qwen3-Omni thinker checkpoint this slice supports.
constexpr int kTextLayers = 48;
constexpr int kTextH = 2048;
constexpr int kQHeads = 32;
constexpr int kKvHeads = 4;
constexpr int kHeadDim = 128;
constexpr int kVocab = 152064;
constexpr int kNumExperts = 128;
constexpr int kTopK = 8;
constexpr int kExpertInter = 768;
constexpr float kRmsEps = 1e-6f;
constexpr float kTextRopeTheta = 1000000.0f;

constexpr int kVisionBlocks = 27;
constexpr int kVisionH = 1152;
constexpr int kVisionHeads = 16;
constexpr int kVisionHeadDim = 72;
constexpr int kVisionInter = 4304;
constexpr int kVisionOut = 2048;
constexpr int kVisionGridSide = 48;
constexpr float kVisionRopeTheta = 10000.0f;
constexpr int kDeepstack[] = {8, 16, 24};

struct EngineWeights {
  // Names are exactly the checkpoint keys; pointers stay valid for the
  // process lifetime because the mmap is held by the ShardedSafetensors owner.
  std::unordered_map<std::string, const Bf16*> ptr;
  const Bf16* get(const std::string& name) const;
};

// Device-side scratch buffers and caches, sized once at construction.
struct EngineScratch {
  cudaStream_t stream = nullptr;
  int max_seq = 2048;

  // Text decoder workspaces.
  Bf16 *hidden = nullptr, *norm = nullptr;
  Bf16 *q = nullptr, *k = nullptr, *v = nullptr;
  Bf16 *gate = nullptr, *up = nullptr, *moe_acc = nullptr;
  float *router_logits = nullptr, *router_w = nullptr, *logits = nullptr;
  int *router_idx = nullptr, *argmax_scratch = nullptr, *d_vis_pos = nullptr;
  float *mrope_cos = nullptr, *mrope_sin = nullptr;  // [max_seq, 128]
  Bf16 *kv = nullptr;  // [layers][2][max_seq][kv_heads][head_dim]

  // Vision tower workspaces (864-token image).
  Bf16 *v_hidden = nullptr, *v_norm = nullptr, *v_qkv = nullptr, *v_proj = nullptr;
  Bf16* v_qp = nullptr, *v_kp = nullptr, *v_vp = nullptr;  // split q/k/v
  Bf16 *v_mlp = nullptr, *v_rope_cos = nullptr, *v_rope_sin = nullptr;
  Bf16 *v_shuffle = nullptr;  // 216 x 4608
  Bf16 *pooler = nullptr;     // [216, 2048]
  Bf16 *deepstack[3] = {nullptr, nullptr, nullptr};
};

size_t kv_cache_offset(int layer, bool is_v, int slot);

void run_merger(const EngineWeights& w, EngineScratch& s, const Bf16* hidden,
                Bf16* out, const std::string& prefix, int tokens,
                bool postshuffle, cudaStream_t st);

// The abort check runs before every vision block; RequestAborted is raised
// there. An empty check keeps the original behaviour.
void vision_forward(const EngineWeights& w, EngineScratch& s,
                    const PreprocessedImage& img, Bf16* d_pixels_bf16,
                    const AbortCheck& abort = {});

void text_layer(const EngineWeights& w, EngineScratch& s, int layer, int m,
                int history);

struct PositionIds {
  std::vector<int> t, h, w;
  int next_pos;
};

PositionIds build_positions(const std::vector<int>& ids, int image_pad_id,
                            int vision_start_id, int vision_end_id,
                            int grid_h, int grid_w, bool has_image);

void build_text_rope_table(const int pos_t[], const int pos_h[],
                           const int pos_w[], int n, EngineScratch& s);

}  // namespace umminfer::omni
