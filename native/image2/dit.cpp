// SPDX-License-Identifier: Apache-2.0
#include "dit.hpp"

#include "cuda_kernels.hpp"
#include "weights.hpp"

#include <cmath>

namespace umminfer::image2 {
namespace {
constexpr float kEps = 1e-6f;
constexpr int kTheta = 10000;

// Complex-pair RoPE table for the joint [text, image] sequence.
// axes_dims_rope = [16, 56, 56] -> 8/28/28 complex components.
struct RopeTable {
  DeviceTensorF cos, sin;  // [joint_rows * 64]
};

void fill_row(float* c, float* s, int row, int frame, int height, int width) {
  for (int k = 0; k < 64; ++k) {
    int pos, comp, axis_dim;
    if (k < 8) {
      pos = frame; comp = k; axis_dim = 16;
    } else if (k < 36) {
      pos = height; comp = k - 8; axis_dim = 56;
    } else {
      pos = width; comp = k - 36; axis_dim = 56;
    }
    // rope_params: angle = index * theta^(-2*comp/axis_dim). Negative indices
    // use the same formula; the reference's negative table is exactly that.
    const float angle =
        static_cast<float>(pos) /
        std::pow(static_cast<float>(kTheta), (2.0f * comp) / static_cast<float>(axis_dim));
    c[static_cast<size_t>(row) * 64 + k] = std::cos(angle);
    s[static_cast<size_t>(row) * 64 + k] = std::sin(angle);
  }
}

void build_rope(int gh, int gw, int txt_len, RopeTable& out) {
  const int s_img = gh * gw;
  const int rows = txt_len + s_img;
  std::vector<float> c(static_cast<size_t>(rows) * 64), sn(static_cast<size_t>(rows) * 64);
  for (int j = 0; j < txt_len; ++j) fill_row(c.data(), sn.data(), j, j, j, j);
  const int h_half = gh / 2, w_half = gw / 2;
  for (int iy = 0; iy < gh; ++iy)
    for (int ix = 0; ix < gw; ++ix)
      fill_row(c.data(), sn.data(), txt_len + iy * gw + ix, txt_len, iy - h_half,
               ix - w_half);
  out.cos = DeviceTensorF(c.size());
  out.sin = DeviceTensorF(sn.size());
  IMG_CUDA_CHECK(cudaMemcpy(out.cos.data, c.data(), c.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
  IMG_CUDA_CHECK(cudaMemcpy(out.sin.data, sn.data(), sn.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
}

// Sinusoid 256: cos on the first half, sin on the second, 1000x time factor.
void timestep_embedding(float t_scaled, std::vector<float>& h) {
  h.assign(256, 0.0f);
  const float t = 1000.0f * t_scaled;
  for (int i = 0; i < 128; ++i) {
    const float freq = std::exp(-std::log(10000.0f) * i / 128.0f);
    const float arg = t * freq;
    h[i] = std::cos(arg);
    h[128 + i] = std::sin(arg);
  }
}

}  // namespace

DiT::DiT(CudaRuntime& rt, const std::string& dir)
    : rt_(rt), st_(std::make_unique<ShardedSafeTensors>(dir)) {
  auto L = [&](DeviceTensor& dst, const std::string& n) { dst = load_bf16(*st_, n); };
  L(img_in_w, "img_in.weight");
  L(txt_norm_w, "txt_in.text_norm.weight");
  L(txt_in_w, "txt_in.in_layer.weight");
  L(txt_out_w, "txt_in.out_layer.weight");
  L(time_l1_w, "time_text_embed.timestep_embedder.linear_1.weight");
  L(time_l2_w, "time_text_embed.timestep_embedder.linear_2.weight");
  L(modulation_w, "modulation.1.weight");
  L(norm_out_w, "norm_out.linear.weight");
  L(proj_out_w, "proj_out.weight");
  blocks_.reserve(kBlocks);
  for (int i = 0; i < kBlocks; ++i) {
    const std::string p = "transformer_blocks." + std::to_string(i) + ".";
    BlockW b;
    L(b.q_w, p + "attn.to_q.weight");
    L(b.k_w, p + "attn.to_k.weight");
    L(b.v_w, p + "attn.to_v.weight");
    L(b.out_w, p + "attn.to_out.0.weight");
    L(b.nq, p + "attn.norm_q.weight");
    L(b.nk, p + "attn.norm_k.weight");
    L(b.mlp_proj_w, p + "img_mlp.proj.weight");
    L(b.mlp_gate_w, p + "img_mlp.gate_layer.weight");
    L(b.mlp_out_w, p + "img_mlp.out.weight");
    blocks_.push_back(std::move(b));
  }
}

DeviceTensor DiT::forward(const DeviceTensor& img_packed, const DeviceTensor& txt_embed,
                          float timestep_scaled, int gh, int gw, int txt_len) {
  auto& s = rt_.stream;
  const int s_img = gh * gw;
  const int joint = txt_len + s_img;
  const int inner = kInner;

  // --- input projections ---------------------------------------------------
  DeviceTensor img({s_img, inner});
  linear_forward(rt_, img_packed.data, img_in_w.data, nullptr, img.data, s_img, inner,
                 kChannels);
  DeviceTensor txt_normed({txt_len, kTextDim});
  zero_center_rms_norm(txt_embed.data, txt_norm_w.data, txt_normed.data, txt_len, kTextDim,
                       kEps, s);
  DeviceTensor txt_mid({txt_len, inner});
  linear_forward(rt_, txt_normed.data, txt_in_w.data, nullptr, txt_mid.data, txt_len, inner,
                 kTextDim);
  gelu_tanh_inplace(txt_mid.data, txt_len * inner, s);
  DeviceTensor txt({txt_len, inner});
  linear_forward(rt_, txt_mid.data, txt_out_w.data, nullptr, txt.data, txt_len, inner,
                 inner);

  // --- timestep embedding and the shared modulation -------------------------
  std::vector<float> sin_real, sin_zero;
  timestep_embedding(timestep_scaled, sin_real);
  timestep_embedding(0.0f, sin_zero);
  DeviceTensorF sinus_real(256), sinus_zero(256);
  IMG_CUDA_CHECK(cudaMemcpy(sinus_real.data, sin_real.data(), 256 * sizeof(float),
                            cudaMemcpyHostToDevice));
  IMG_CUDA_CHECK(cudaMemcpy(sinus_zero.data, sin_zero.data(), 256 * sizeof(float),
                            cudaMemcpyHostToDevice));
  DeviceTensor sbf({2, 256});
  f32_to_bf16(sinus_real.data, sbf.data, 256, s);
  f32_to_bf16(sinus_zero.data, sbf.data + 256, 256, s);

  DeviceTensor temb1({2, inner}), temb({2, inner});
  linear_forward(rt_, sbf.data, time_l1_w.data, nullptr, temb1.data, 2, inner, 256);
  silu_inplace(temb1.data, 2 * inner, s);
  linear_forward(rt_, temb1.data, time_l2_w.data, nullptr, temb.data, 2, inner, inner);
  DeviceTensor temb_silu({2, inner});
  IMG_CUDA_CHECK(cudaMemcpyAsync(temb_silu.data, temb.data, 2 * (size_t)inner * sizeof(Bf16),
                                 cudaMemcpyDeviceToDevice, s));
  silu_inplace(temb_silu.data, 2 * inner, s);
  // row 0 = sampled timestep, row 1 = t=0 (used by text tokens).
  DeviceTensor modulation({2, 4 * inner});
  linear_forward(rt_, temb_silu.data, modulation_w.data, nullptr, modulation.data, 2,
                 4 * inner, inner);
  DeviceTensor final_scale({2, inner});
  linear_forward(rt_, temb_silu.data, norm_out_w.data, nullptr, final_scale.data, 2, inner,
                 inner);

  // --- joint sequence -------------------------------------------------------
  DeviceTensor joint_hidden({joint, inner});
  IMG_CUDA_CHECK(cudaMemcpyAsync(joint_hidden.data, txt.data, (size_t)txt_len * inner * sizeof(Bf16),
                                 cudaMemcpyDeviceToDevice, s));
  IMG_CUDA_CHECK(cudaMemcpyAsync(joint_hidden.data + (size_t)txt_len * inner, img.data,
                                 (size_t)s_img * inner * sizeof(Bf16),
                                 cudaMemcpyDeviceToDevice, s));
  std::vector<int> target_host(joint, 0);
  for (int i = txt_len; i < joint; ++i) target_host[i] = 1;
  int* is_target = nullptr;
  IMG_CUDA_CHECK(request_cuda_malloc(&is_target, joint * sizeof(int)));
  IMG_CUDA_CHECK(cudaMemcpy(is_target, target_host.data(), joint * sizeof(int),
                            cudaMemcpyHostToDevice));

  RopeTable rope;
  build_rope(gh, gw, txt_len, rope);

  // Persistent buffers.
  DeviceTensor normed({joint, inner});
  DeviceTensor q({joint, inner}), k({joint, inner}), v({joint, inner});
  DeviceTensor attn_out({joint, inner});
  DeviceTensor proj({joint, inner});
  DeviceTensor mlp_a({joint, kMlp}), mlp_b({joint, kMlp});
  const Bf16* mod_real = modulation.data;
  const Bf16* mod_zero = modulation.data + (size_t)4 * inner;

  for (const BlockW& w : blocks_) {
    // ---- attention sub-block ----
    layer_norm_mod_select(joint_hidden.data, mod_real + 0, mod_zero + 0, is_target,
                          normed.data, joint, inner, kEps, s);
    linear_forward(rt_, normed.data, w.q_w.data, nullptr, q.data, joint, inner, inner);
    linear_forward(rt_, normed.data, w.k_w.data, nullptr, k.data, joint, inner, inner);
    linear_forward(rt_, normed.data, w.v_w.data, nullptr, v.data, joint, inner, inner);
    rms_norm_lastdim(q.data, w.nq.data, q.data, joint * kHeads, kHeadDim, kEps, s);
    rms_norm_lastdim(k.data, w.nk.data, k.data, joint * kHeads, kHeadDim, kEps, s);
    rope_complex_pairs(q.data, rope.cos.data, rope.sin.data, joint, kHeads, kHeadDim, s);
    rope_complex_pairs(k.data, rope.cos.data, rope.sin.data, joint, kHeads, kHeadDim, s);
    // Text rows: causal over text keys only. Image rows: full attention.
    gqa_causal_attn(q.data, k.data, v.data, attn_out.data, 1, kHeads, kHeads, txt_len,
                    txt_len, kHeadDim, s);
    flash_attn(q.data + (size_t)txt_len * inner, k.data, v.data,
               attn_out.data + (size_t)txt_len * inner, 1, kHeads, s_img, joint, kHeadDim, s);
    linear_forward(rt_, attn_out.data, w.out_w.data, nullptr, proj.data, joint, inner, inner);
    gated_residual_tanh(joint_hidden.data, proj.data, mod_real + inner, mod_zero + inner,
                        is_target, joint, inner, s);

    // ---- MLP sub-block (SwiGLU) ----
    layer_norm_mod_select(joint_hidden.data, mod_real + 2 * inner, mod_zero + 2 * inner,
                          is_target, normed.data, joint, inner, kEps, s);
    linear_forward(rt_, normed.data, w.mlp_gate_w.data, nullptr, mlp_a.data, joint, kMlp,
                   inner);
    linear_forward(rt_, normed.data, w.mlp_proj_w.data, nullptr, mlp_b.data, joint, kMlp,
                   inner);
    silu_mul(mlp_a.data, mlp_a.data, mlp_b.data, joint * kMlp, s);
    linear_forward(rt_, mlp_a.data, w.mlp_out_w.data, nullptr, proj.data, joint, inner,
                   kMlp);
    gated_residual_tanh(joint_hidden.data, proj.data, mod_real + 3 * inner,
                        mod_zero + 3 * inner, is_target, joint, inner, s);
  }

  // --- head: target rows only ----------------------------------------------
  DeviceTensor img_hidden({s_img, inner});
  IMG_CUDA_CHECK(cudaMemcpyAsync(img_hidden.data,
                                 joint_hidden.data + (size_t)txt_len * inner,
                                 (size_t)s_img * inner * sizeof(Bf16),
                                 cudaMemcpyDeviceToDevice, s));
  DeviceTensor img_normed({s_img, inner});
  layer_norm_no_affine(img_hidden.data, img_normed.data, s_img, inner, kEps, s);
  DeviceTensor scaled({s_img, inner});
  scale_apply_select(img_normed.data, final_scale.data, final_scale.data, is_target + txt_len,
                     scaled.data, s_img, inner, s);
  DeviceTensor velocity({s_img, kChannels});
  linear_forward(rt_, scaled.data, proj_out_w.data, nullptr, velocity.data, s_img, kChannels,
                 inner);

  IMG_CUDA_CHECK(request_cuda_free(is_target));
  rt_.sync();
  return velocity;
}

}  // namespace umminfer::image2
