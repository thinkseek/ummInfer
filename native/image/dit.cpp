// SPDX-License-Identifier: Apache-2.0
#include "dit.hpp"

#include "cuda_kernels.hpp"
#include "weights.hpp"

#include <cmath>

namespace umminfer::image {
namespace {
constexpr float kEps = 1e-6f;
constexpr int kTheta = 10000;

// Build the complex-polar RoPE cos/sin tables (real view) for image grid and
// text rows. axes dims 16/56/56 -> 8/28/28 complex components per axis.
struct RopeTables {
  DeviceTensorF vid_cos, vid_sin, txt_cos, txt_sin;  // each [rows, 64]
};
}  // namespace

DiT::DiT(CudaRuntime& rt, const std::string& dir)
    : rt_(rt), st_(std::make_unique<ShardedSafeTensors>(dir)) {
  auto L = [&](DeviceTensor& dst, const std::string& n) { dst = load_bf16(*st_, n); };
  L(img_in_w, "img_in.weight"); L(img_in_b, "img_in.bias");
  L(txt_in_w, "txt_in.weight"); L(txt_in_b, "txt_in.bias");
  L(txt_norm_w, "txt_norm.weight");
  L(time_l1_w, "time_text_embed.timestep_embedder.linear_1.weight");
  L(time_l1_b, "time_text_embed.timestep_embedder.linear_1.bias");
  L(time_l2_w, "time_text_embed.timestep_embedder.linear_2.weight");
  L(time_l2_b, "time_text_embed.timestep_embedder.linear_2.bias");
  L(norm_out_w, "norm_out.linear.weight"); L(norm_out_b, "norm_out.linear.bias");
  L(proj_out_w, "proj_out.weight"); L(proj_out_b, "proj_out.bias");
  blocks_.reserve(kBlocks);
  for (int i = 0; i < kBlocks; ++i) {
    const std::string p = "transformer_blocks." + std::to_string(i) + ".";
    BlockW b;
    L(b.img_mod_w, p + "img_mod.1.weight"); L(b.img_mod_b, p + "img_mod.1.bias");
    L(b.txt_mod_w, p + "txt_mod.1.weight"); L(b.txt_mod_b, p + "txt_mod.1.bias");
    L(b.q_w, p + "attn.to_q.weight"); L(b.q_b, p + "attn.to_q.bias");
    L(b.k_w, p + "attn.to_k.weight"); L(b.k_b, p + "attn.to_k.bias");
    L(b.v_w, p + "attn.to_v.weight"); L(b.v_b, p + "attn.to_v.bias");
    L(b.out_w, p + "attn.to_out.0.weight"); L(b.out_b, p + "attn.to_out.0.bias");
    L(b.aq_w, p + "attn.add_q_proj.weight"); L(b.aq_b, p + "attn.add_q_proj.bias");
    L(b.ak_w, p + "attn.add_k_proj.weight"); L(b.ak_b, p + "attn.add_k_proj.bias");
    L(b.av_w, p + "attn.add_v_proj.weight"); L(b.av_b, p + "attn.add_v_proj.bias");
    L(b.aout_w, p + "attn.to_add_out.weight"); L(b.aout_b, p + "attn.to_add_out.bias");
    L(b.nq, p + "attn.norm_q.weight"); L(b.nk, p + "attn.norm_k.weight");
    L(b.anq, p + "attn.norm_added_q.weight"); L(b.ank, p + "attn.norm_added_k.weight");
    L(b.imlp0_w, p + "img_mlp.net.0.proj.weight"); L(b.imlp0_b, p + "img_mlp.net.0.proj.bias");
    L(b.imlp2_w, p + "img_mlp.net.2.weight"); L(b.imlp2_b, p + "img_mlp.net.2.bias");
    L(b.tmlp0_w, p + "txt_mlp.net.0.proj.weight"); L(b.tmlp0_b, p + "txt_mlp.net.0.proj.bias");
    L(b.tmlp2_w, p + "txt_mlp.net.2.weight"); L(b.tmlp2_b, p + "txt_mlp.net.2.bias");
    blocks_.push_back(std::move(b));
  }
}

namespace {

// Frequency angle for one position and one complex component index.
float rope_angle(int position, int comp, int axis_complex_dim) {
  const float f = 1.0f / std::pow((float)kTheta,
                                  (2.0f * comp) / (2.0f * axis_complex_dim));
  return position * f;
}

// Fill a [rows,64] cos/sin pair; component groups are frame(8),h(28),w(28).
void make_rope_tables(int gh, int gw, int txt_len, RopeTables& out) {
  const int s_img = gh * gw;
  std::vector<float> vc(s_img * 64), vs(s_img * 64), tc(txt_len * 64), ts(txt_len * 64);
  // Centered negative H/W coordinates for even grid dims.
  const int h_half = gh / 2, w_half = gw / 2;
  for (int iy = 0; iy < gh; ++iy) {
    const int hp = iy - h_half;  // centered: [-h/2, h/2)
    for (int ix = 0; ix < gw; ++ix) {
      const int wp = ix - w_half;  // centered: [-w/2, w/2)
      const int row = iy * gw + ix;
      for (int cpx = 0; cpx < 64; ++cpx) {
        int pos;
        int axis_dim;
        if (cpx < 8) { pos = 0; axis_dim = 8; }
        else if (cpx < 36) { pos = hp; axis_dim = 28; }
        else { pos = wp; axis_dim = 28; }
        const float a = rope_angle(pos, cpx < 8 ? cpx : cpx < 36 ? cpx - 8 : cpx - 36,
                                   axis_dim);
        vc[row * 64 + cpx] = std::cos(a);
        vs[row * 64 + cpx] = std::sin(a);
      }
    }
  }
  // Text rows: absolute positions starting at max(gh/2, gw/2).
  const int txt_base = std::max(gh / 2, gw / 2);
  for (int j = 0; j < txt_len; ++j) {
    const int pos = txt_base + j;
    for (int cpx = 0; cpx < 64; ++cpx) {
      const int axis_dim = cpx < 8 ? 8 : cpx < 36 ? 28 : 28;
      const float a = rope_angle(pos, cpx < 8 ? cpx : cpx < 36 ? cpx - 8 : cpx - 36,
                                 axis_dim);
      tc[j * 64 + cpx] = std::cos(a);
      ts[j * 64 + cpx] = std::sin(a);
    }
  }
  auto upload = [](const std::vector<float>& h, DeviceTensorF& d) {
    d = DeviceTensorF(h.size());
    cudaMemcpy(d.data, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
  };
  upload(vc, out.vid_cos); upload(vs, out.vid_sin);
  upload(tc, out.txt_cos); upload(ts, out.txt_sin);
}

}  // namespace

DeviceTensor DiT::forward(DeviceTensor& img_packed, const DeviceTensor& txt_embed3584,
                          float timestep_scaled, int gh, int gw, int txt_len) {
  auto& s = rt_.stream;
  const int s_img = static_cast<int>(img_packed.numel / kPacked);
  const int s_joint = s_img + txt_len;

  DeviceTensor img({s_img, kInner}), txt({txt_len, kInner});
  linear_forward(rt_, img_packed.data, img_in_w.data, img_in_b.data, img.data, s_img,
                 kInner, kPacked);
  // txt: RMSNorm(3584) then Linear 3584->3072.
  DeviceTensor txt_normed({txt_len, kJoint});
  rms_norm_lastdim(txt_embed3584.data, txt_norm_w.data, txt_normed.data, txt_len, kJoint,
                   kEps, s);
  linear_forward(rt_, txt_normed.data, txt_in_w.data, txt_in_b.data, txt.data, txt_len,
                 kInner, kJoint);

  // Timestep sinusoid 256 (flip_sin_to_cos, scale 1000) then SiLU MLP.
  DeviceTensorF sinus(256);
  {
    std::vector<float> h(256);
    const float t = timestep_scaled;  // already t/1000; time_proj re-applies 1000
    for (int i = 0; i < 128; ++i) {
      const float freq = std::exp(-std::log(10000.0f) * i / 128.0f);
      const float arg = 1000.0f * t * freq;
      h[128 + i] = std::sin(arg);  // flipped layout: cos block first
      h[i] = std::cos(arg);
    }
    cudaMemcpy(sinus.data, h.data(), 256 * sizeof(float), cudaMemcpyHostToDevice);
  }
  DeviceTensor sin_bf({256});
  f32_to_bf16(sinus.data, sin_bf.data, 256, s);
  DeviceTensor temb1({1, kInner}), temb({1, kInner});
  linear_forward(rt_, sin_bf.data, time_l1_w.data, time_l1_b.data, temb1.data, 1, kInner,
                 256);
  silu_inplace(temb1.data, kInner, s);
  linear_forward(rt_, temb1.data, time_l2_w.data, time_l2_b.data, temb.data, 1, kInner,
                 kInner);

  RopeTables rope;
  make_rope_tables(gh, gw, txt_len, rope);

  // Persistent work buffers.
  DeviceTensor work_i({s_img, kInner}), work_t({txt_len, kInner});
  DeviceTensor qj({s_joint, kInner}), kj({s_joint, kInner}), vj({s_joint, kInner});
  DeviceTensor attn_out({s_joint, kInner});
  DeviceTensor img_mod({1, 6 * kInner}), txt_mod({1, 6 * kInner});
  // img_mod/txt_mod are Sequential(SiLU, Linear); the SiLU acts on temb once.
  DeviceTensor temb_silu({1, kInner});
  cudaMemcpy(temb_silu.data, temb.data, (size_t)kInner * sizeof(Bf16),
             cudaMemcpyDeviceToDevice);
  silu_inplace(temb_silu.data, kInner, s);
  DeviceTensor mlp0({s_img, kMlp}), mlp2({s_img, kMlp});
  DeviceTensor norm2({1, 2 * kInner});

  for (int bi = 0; bi < kBlocks; ++bi) {
    const BlockW& w = blocks_[bi];
    // Each modulation linear is Sequential(SiLU,Linear); input is silu(temb).
    linear_forward(rt_, temb_silu.data, w.img_mod_w.data, w.img_mod_b.data, img_mod.data, 1,
                   6 * kInner, kInner);
    linear_forward(rt_, temb_silu.data, w.txt_mod_w.data, w.txt_mod_b.data, txt_mod.data, 1,
                   6 * kInner, kInner);
    // --- stream modulation 1: shift1, scale1, gate1 ---
    layer_norm_no_affine(img.data, work_i.data, s_img, kInner, kEps, s);
    adaln_modulate(work_i.data, img_mod.data, img_mod.data + kInner, work_i.data, s_img,
                   kInner, s);
    layer_norm_no_affine(txt.data, work_t.data, txt_len, kInner, kEps, s);
    adaln_modulate(work_t.data, txt_mod.data, txt_mod.data + kInner, work_t.data, txt_len,
                   kInner, s);

    // QKV projections for both streams.
    linear_forward(rt_, work_i.data, w.q_w.data, w.q_b.data, qj.data + (size_t)txt_len * kInner,
                   s_img, kInner, kInner);
    linear_forward(rt_, work_i.data, w.k_w.data, w.k_b.data, kj.data + (size_t)txt_len * kInner,
                   s_img, kInner, kInner);
    linear_forward(rt_, work_i.data, w.v_w.data, w.v_b.data, vj.data + (size_t)txt_len * kInner,
                   s_img, kInner, kInner);
    linear_forward(rt_, work_t.data, w.aq_w.data, w.aq_b.data, qj.data, txt_len, kInner,
                   kInner);
    linear_forward(rt_, work_t.data, w.ak_w.data, w.ak_b.data, kj.data, txt_len, kInner,
                   kInner);
    linear_forward(rt_, work_t.data, w.av_w.data, w.av_b.data, vj.data, txt_len, kInner,
                   kInner);

    // QK RMS norms per head, then RoPE on each stream before the joint concat.
    // Gamma is a single shared [head_dim] vector applied to every (token,head)
    // row, so the flattened last-dim norm indexes it by channel only.
    auto per_head_norm = [&](Bf16* x, const Bf16* weight, int rows) {
      rms_norm_lastdim(x, weight, x, rows * kHeads, kHeadDim, kEps, s);
    };
    per_head_norm(qj.data, w.anq.data, txt_len);
    per_head_norm(kj.data, w.ank.data, txt_len);
    per_head_norm(qj.data + (size_t)txt_len * kInner, w.nq.data, s_img);
    per_head_norm(kj.data + (size_t)txt_len * kInner, w.nk.data, s_img);
    rope_complex_pairs(qj.data, rope.txt_cos.data, rope.txt_sin.data, txt_len, kHeads,
                       kHeadDim, s);
    rope_complex_pairs(kj.data, rope.txt_cos.data, rope.txt_sin.data, txt_len, kHeads,
                       kHeadDim, s);
    rope_complex_pairs(qj.data + (size_t)txt_len * kInner, rope.vid_cos.data,
                       rope.vid_sin.data, s_img, kHeads, kHeadDim, s);
    rope_complex_pairs(kj.data + (size_t)txt_len * kInner, rope.vid_cos.data,
                       rope.vid_sin.data, s_img, kHeads, kHeadDim, s);

    // Joint non-causal attention over [text, image].
    flash_attn(qj.data, kj.data, vj.data, attn_out.data, 1, kHeads, s_joint, s_joint,
               kHeadDim, s);
    // Split and output projections.
    linear_forward(rt_, attn_out.data, w.aout_w.data, w.aout_b.data, work_t.data, txt_len,
                   kInner, kInner);
    linear_forward(rt_, attn_out.data + (size_t)txt_len * kInner, w.out_w.data, w.out_b.data,
                   work_i.data, s_img, kInner, kInner);

    // Gated attention residuals (gate1 is modulation block 2 per stream).
    gated_residual(img.data, work_i.data, img_mod.data + 2 * kInner, s_img, kInner, s);
    gated_residual(txt.data, work_t.data, txt_mod.data + 2 * kInner, txt_len, kInner, s);

    // --- MLP sub-block (modulation 2: shift2, scale2, gate2) ---
    layer_norm_no_affine(img.data, work_i.data, s_img, kInner, kEps, s);
    adaln_modulate(work_i.data, img_mod.data + 3 * kInner, img_mod.data + 4 * kInner,
                   work_i.data, s_img, kInner, s);
    linear_forward(rt_, work_i.data, w.imlp0_w.data, w.imlp0_b.data, mlp0.data, s_img, kMlp,
                   kInner);
    gelu_tanh_inplace(mlp0.data, s_img * kMlp, s);
    linear_forward(rt_, mlp0.data, w.imlp2_w.data, w.imlp2_b.data, mlp2.data, s_img, kInner,
                   kMlp);
    gated_residual(img.data, mlp2.data, img_mod.data + 5 * kInner, s_img, kInner, s);

    layer_norm_no_affine(txt.data, work_t.data, txt_len, kInner, kEps, s);
    adaln_modulate(work_t.data, txt_mod.data + 3 * kInner, txt_mod.data + 4 * kInner,
                   work_t.data, txt_len, kInner, s);
    linear_forward(rt_, work_t.data, w.tmlp0_w.data, w.tmlp0_b.data, mlp0.data, txt_len, kMlp,
                   kInner);
    gelu_tanh_inplace(mlp0.data, txt_len * kMlp, s);
    linear_forward(rt_, mlp0.data, w.tmlp2_w.data, w.tmlp2_b.data, mlp2.data, txt_len, kInner,
                   kMlp);
    gated_residual(txt.data, mlp2.data, txt_mod.data + 5 * kInner, txt_len, kInner, s);
  }

  // Final AdaLN on image tokens only, then proj_out. AdaLayerNormContinuous
  // applies SiLU to temb before the (scale, shift) linear.
  DeviceTensor normed({s_img, kInner});
  layer_norm_no_affine(img.data, normed.data, s_img, kInner, kEps, s);
  linear_forward(rt_, temb_silu.data, norm_out_w.data, norm_out_b.data, norm2.data, 1,
                 2 * kInner, kInner);
  // norm_out.linear yields (scale, shift); AdaLN order is scale then shift.
  adaln_modulate(normed.data, norm2.data + kInner, norm2.data, normed.data, s_img, kInner, s);
  DeviceTensor velocity({s_img, kPacked});
  linear_forward(rt_, normed.data, proj_out_w.data, proj_out_b.data, velocity.data, s_img,
                 kPacked, kInner);
  rt_.sync();
  return velocity;
}

}  // namespace umminfer::image
