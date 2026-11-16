// SPDX-License-Identifier: Apache-2.0
#include "engine_internal.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace umminfer::omni {
namespace {

constexpr int kMaxImgTokens = 1296;  // 36x36 pre-merge upper bound for 768px cap

// Build the [tokens, 72] vision cos/sin tables for one image grid.
void build_vision_rope(int grid_h, int grid_w, Bf16* cos, Bf16* sin,
                       cudaStream_t stream) {
  const int half = kVisionHeadDim / 2;  // 36
  std::vector<float> inv(half / 2);     // 18
  for (int i = 0; i < half / 2; ++i)
    inv[i] = 1.0f / std::pow(kVisionRopeTheta, 2.0f * i / half);
  const int gh2 = grid_h / 2, gw2 = grid_w / 2;
  const int tokens = grid_h * grid_w;
  std::vector<float> hcos(tokens * kVisionHeadDim, 0.0f);
  std::vector<float> hsin(tokens * kVisionHeadDim, 0.0f);
  int row = 0;
  // Token order matches the processor: (block_r, block_c, intra_r, intra_c).
  for (int br = 0; br < gh2; ++br)
    for (int bc = 0; bc < gw2; ++bc)
      for (int ir = 0; ir < 2; ++ir)
        for (int ic = 0; ic < 2; ++ic, ++row) {
          const int r = br * 2 + ir, c = bc * 2 + ic;
          for (int j = 0; j < half / 2; ++j) {
            const float fr = r * inv[j], fc = c * inv[j];
            hcos[row * kVisionHeadDim + j] = std::cos(fr);
            hsin[row * kVisionHeadDim + j] = std::sin(fr);
            hcos[row * kVisionHeadDim + half / 2 + j] = std::cos(fc);
            hsin[row * kVisionHeadDim + half / 2 + j] = std::sin(fc);
          }
          for (int j = 0; j < half; ++j) {  // duplicate halves
            hcos[row * kVisionHeadDim + half + j] = hcos[row * kVisionHeadDim + j];
            hsin[row * kVisionHeadDim + half + j] = hsin[row * kVisionHeadDim + j];
          }
        }
  if (row != tokens) throw std::runtime_error("vision rope token count mismatch");
  std::vector<Bf16> bcos(hcos.size()), bsin(hsin.size());
  for (size_t i = 0; i < hcos.size(); ++i) {
    bcos[i] = Bf16(hcos[i]);
    bsin[i] = Bf16(hsin[i]);
  }
  cudaMemcpyAsync(cos, bcos.data(), bcos.size() * sizeof(Bf16),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(sin, bsin.data(), bsin.size() * sizeof(Bf16),
                  cudaMemcpyHostToDevice, stream);
}

}  // namespace

// Vision tower forward. Input pixels [tokens, 1536] f32 already on host.
// Writes pooler [tokens/4, 2048] and the three deepstack matrices.
void vision_forward(const EngineWeights& w, EngineScratch& s,
                    const PreprocessedImage& img, Bf16* d_pixels_bf16) {
  const int tokens = img.rows();
  if (tokens > kMaxImgTokens) throw std::runtime_error("image too large for slice");
  const std::string p = "thinker.visual.";
  cudaStream_t st = s.stream;

  // Patch embed: conv3d is a dense linear over the 1536-vector.
  gemm_bf16_tiled(s.v_hidden, d_pixels_bf16, w.get(p + "patch_embed.proj.weight"),
                  w.get(p + "patch_embed.proj.bias"), tokens, 1536, kVisionH, st);

  // Absolute position embedding with bilinear interpolation.
  // Computed on host for clarity; table is only 2304x1152.
  const Bf16* pos_table = w.get(p + "pos_embed.weight");
  std::vector<Bf16> h_table(kVisionGridSide * kVisionGridSide * kVisionH);
  cudaMemcpy(h_table.data(), pos_table, h_table.size() * sizeof(Bf16),
             cudaMemcpyDeviceToHost);
  const int gh = img.grid_h, gw = img.grid_w, gh2 = gh / 2, gw2 = gw / 2;
  std::vector<float> h_idx(gh), w_idx(gw);
  for (int i = 0; i < gh; ++i)
    h_idx[i] = i * static_cast<float>(kVisionGridSide - 1) / (gh - 1);
  for (int i = 0; i < gw; ++i)
    w_idx[i] = i * static_cast<float>(kVisionGridSide - 1) / (gw - 1);
  std::vector<Bf16> pos_add(static_cast<size_t>(tokens) * kVisionH);
  int row = 0;
  for (int t = 0; t < img.grid_t; ++t)
    for (int br = 0; br < gh2; ++br)
      for (int bc = 0; bc < gw2; ++bc)
        for (int ir = 0; ir < 2; ++ir)
          for (int ic = 0; ic < 2; ++ic, ++row) {
            const int h = br * 2 + ir, wd = bc * 2 + ic;
            const int hf = static_cast<int>(h_idx[h]);
            const int hc = std::min(hf + 1, kVisionGridSide - 1);
            const int wf = static_cast<int>(w_idx[wd]);
            const int wc = std::min(wf + 1, kVisionGridSide - 1);
            const float dh = h_idx[h] - hf, dw = w_idx[wd] - wf;
            const float cw[4] = {(1 - dh) * (1 - dw), (1 - dh) * dw,
                                 dh * (1 - dw), dh * dw};
            const int idx[4] = {hf * 48 + wf, hf * 48 + wc, hc * 48 + wf,
                                hc * 48 + wc};
            for (int d = 0; d < kVisionH; ++d) {
              float v = 0.0f;
              for (int k = 0; k < 4; ++k)
                v += cw[k] * float(h_table[static_cast<size_t>(idx[k]) * kVisionH + d]);
              pos_add[static_cast<size_t>(row) * kVisionH + d] = Bf16(v);
            }
          }
  Bf16* d_pos;
  cudaMalloc(&d_pos, pos_add.size() * sizeof(Bf16));
  cudaMemcpyAsync(d_pos, pos_add.data(), pos_add.size() * sizeof(Bf16),
                  cudaMemcpyHostToDevice, st);
  add_residual(s.v_hidden, d_pos, tokens * kVisionH, st);
  cudaFree(d_pos);

  build_vision_rope(gh, gw, s.v_rope_cos, s.v_rope_sin, st);

  for (int blk = 0; blk < kVisionBlocks; ++blk) {
    const std::string bp = p + "blocks." + std::to_string(blk);
    layer_norm(s.v_hidden, w.get(bp + ".norm1.weight"),
               w.get(bp + ".norm1.bias"), s.v_norm, tokens, kVisionH, 1e-6f, st);
    // qkv [tokens, 3456], split into contiguous q/k/v [tokens, 1152].
    gemm_bf16_tiled(s.v_qkv, s.v_norm, w.get(bp + ".attn.qkv.weight"),
                    w.get(bp + ".attn.qkv.bias"), tokens, kVisionH,
                    3 * kVisionH, st);
    split_qkv(s.v_qkv, s.v_qp, s.v_kp, s.v_vp, tokens, kVisionH, st);
    Bf16 *vq = s.v_qp, *vk = s.v_kp, *vv = s.v_vp;
    vision_rope(vq, s.v_rope_cos, s.v_rope_sin, tokens, kVisionHeads,
                kVisionHeadDim, st);
    vision_rope(vk, s.v_rope_cos, s.v_rope_sin, tokens, kVisionHeads,
                kVisionHeadDim, st);
    // Bidirectional MHA over the single image: window_len = tokens.
    gqa_attention(vq, vk, vv, s.v_proj, tokens, tokens, kVisionHeads,
                  kVisionHeads, kVisionHeadDim, false, tokens, 0, st);
    gemm_bf16_tiled(s.v_qkv, s.v_proj, w.get(bp + ".attn.proj.weight"),
                    w.get(bp + ".attn.proj.bias"), tokens, kVisionH, kVisionH,
                    st);
    add_residual(s.v_hidden, s.v_qkv, tokens * kVisionH, st);

    layer_norm(s.v_hidden, w.get(bp + ".norm2.weight"),
               w.get(bp + ".norm2.bias"), s.v_norm, tokens, kVisionH, 1e-6f, st);
    gemm_bf16_tiled(s.v_mlp, s.v_norm, w.get(bp + ".mlp.linear_fc1.weight"),
                    w.get(bp + ".mlp.linear_fc1.bias"), tokens, kVisionH,
                    kVisionInter, st);
    gelu_tanh_inplace(s.v_mlp, tokens * kVisionInter, st);
    gemm_bf16_tiled(s.v_norm, s.v_mlp, w.get(bp + ".mlp.linear_fc2.weight"),
                    w.get(bp + ".mlp.linear_fc2.bias"), tokens, kVisionInter,
                    kVisionH, st);
    add_residual(s.v_hidden, s.v_norm, tokens * kVisionH, st);

    // Deepstack mergers after blocks 8, 16, 24.
    for (int k = 0; k < 3; ++k)
      if (blk == kDeepstack[k])
        run_merger(w, s, s.v_hidden, s.deepstack[k],
                   p + "merger_list." + std::to_string(k), tokens, true, st);
  }
  run_merger(w, s, s.v_hidden, s.pooler, p + "merger", tokens, false, st);
}

}  // namespace umminfer::omni
