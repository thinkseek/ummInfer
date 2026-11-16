// SPDX-License-Identifier: Apache-2.0
#include "vae.hpp"

#include "cuda_kernels.hpp"
#include "weights.hpp"

#include <algorithm>
#include <utility>

namespace umminfer::image {
namespace {
constexpr int kMid = 384;  // base 96 * dim_mult 4

// A loaded 2D conv pair (weight + bias); 3D causal kernels use time slice 2.
struct Conv2dW {
  DeviceTensor w, b;
};
struct ResnetW {
  DeviceTensor g1, g2;
  Conv2dW c1, c2, shortcut;  // shortcut empty when in==out
  bool has_shortcut = false;
};

Conv2dW load_conv3d(const ShardedSafeTensors& st, const std::string& prefix) {
  // Causal Conv3d weights are 5D [co,ci,kd,kh,kw]. At T=1 only one time slice
  // sees real data: index 2 for kd=3 (causal pad 2), index 0 for kd=1.
  DeviceTensor w5 = load_bf16(st, prefix + ".weight");
  const auto& shape5 = w5.shape;  // [co,ci,kd,kh,kw]
  const int co = shape5[0], ci = shape5[1], kd = shape5[2], kh = shape5[3],
            kw = shape5[4];
  const int time_idx = kd == 1 ? 0 : 2;
  DeviceTensor w4({co, ci, kh, kw});
  auto host5 = w5.to_host_u16();
  std::vector<uint16_t> host4((size_t)co * ci * kh * kw);
  const size_t plane = (size_t)kh * kw;
  for (int o = 0; o < co; ++o)
    for (int c = 0; c < ci; ++c) {
      const size_t src = (((size_t)o * ci + c) * kd + time_idx) * plane;
      const size_t dst = ((size_t)o * ci + c) * plane;
      std::copy_n(host5.begin() + src, plane, host4.begin() + dst);
    }
  w4.copy_from_host(host4.data());
  return {std::move(w4), load_bf16(st, prefix + ".bias")};
}

Conv2dW load_conv2d(const ShardedSafeTensors& st, const std::string& prefix) {
  return {load_bf16(st, prefix + ".weight"), load_bf16(st, prefix + ".bias")};
}

ResnetW load_resnet(const ShardedSafeTensors& st, const std::string& p, bool shortcut) {
  ResnetW r;
  r.g1 = load_bf16(st, p + ".norm1.gamma");
  r.g2 = load_bf16(st, p + ".norm2.gamma");
  r.c1 = load_conv3d(st, p + ".conv1");
  r.c2 = load_conv3d(st, p + ".conv2");
  r.has_shortcut = shortcut;
  if (shortcut) r.shortcut = load_conv3d(st, p + ".conv_shortcut");
  return r;
}

// Runs one residual block; returns a fresh [n,co,hh,ww] tensor. The input x is
// not modified beyond the residual identity semantics.
DeviceTensor run_resnet(CudaRuntime& rt, const ResnetW& w, const DeviceTensor& x, int n,
                        int ci, int co, int hh, int ww) {
  auto& s = rt.stream;
  const size_t in_n = (size_t)n * ci * hh * ww;
  const size_t out_n = (size_t)n * co * hh * ww;
  DeviceTensor shortcut({n, co, hh, ww});
  if (w.has_shortcut)
    conv2d_1x1(x.data, w.shortcut.w.data, w.shortcut.b.data, shortcut.data, n, ci, co,
               hh * ww, s);
  DeviceTensor normed({n, ci, hh, ww});
  rms_norm_nchw(x.data, w.g1.data, normed.data, n, ci, hh * ww, s);
  silu_inplace(normed.data, in_n, s);
  DeviceTensor h1({n, co, hh, ww});
  conv2d_3x3(normed.data, w.c1.w.data, w.c1.b.data, h1.data, n, ci, co, hh, ww, s);
  DeviceTensor normed2({n, co, hh, ww});
  rms_norm_nchw(h1.data, w.g2.data, normed2.data, n, co, hh * ww, s);
  silu_inplace(normed2.data, out_n, s);
  DeviceTensor h2({n, co, hh, ww});
  conv2d_3x3(normed2.data, w.c2.w.data, w.c2.b.data, h2.data, n, co, co, hh, ww, s);
  if (w.has_shortcut)
    add_residual(h2.data, shortcut.data, out_n, s);
  else
    add_residual(h2.data, x.data, out_n, s);  // ci==co when no shortcut
  return h2;
}

}  // namespace

VaeDecoder::VaeDecoder(CudaRuntime& rt, const std::string& dir)
    : rt_(rt), st_(std::make_unique<ShardedSafeTensors>(dir)) {}

DeviceTensor VaeDecoder::decode(const DeviceTensor& lat, int h, int w) {
  auto& s = rt_.stream;
  const int n = 1;
  const int H8 = h * 8, W8 = w * 8;
  // post_quant_conv: 1x1x1 causal conv, 16->16.
  DeviceTensor pq_w = load_bf16(*st_, "post_quant_conv.weight");
  DeviceTensor pq_b = load_bf16(*st_, "post_quant_conv.bias");
  DeviceTensor x({n, kZ, h, w});
  conv2d_1x1(lat.data, pq_w.data, pq_b.data, x.data, n, kZ, kZ, h * w, s);

  // conv_in 16 -> 384, 3x3.
  Conv2dW conv_in = load_conv3d(*st_, "decoder.conv_in");
  DeviceTensor feat({n, kMid, h, w});
  conv2d_3x3(x.data, conv_in.w.data, conv_in.b.data, feat.data, n, kZ, kMid, h, w, s);

  DeviceTensor out_buf({n, kMid, h, w});
  auto resnet_at = [&](const std::string& p, int ci, int co, int hh, int ww, bool sc) {
    ResnetW rw = load_resnet(*st_, p, sc);
    feat = run_resnet(rt_, rw, feat, n, ci, co, hh, ww);
  };

  // Mid block at 384 channels, latent resolution.
  resnet_at("decoder.mid_block.resnets.0", kMid, kMid, h, w, false);
  {
    DeviceTensor gamma = load_bf16(*st_, "decoder.mid_block.attentions.0.norm.gamma");
    Conv2dW qkv = load_conv2d(*st_, "decoder.mid_block.attentions.0.to_qkv");
    Conv2dW proj = load_conv2d(*st_, "decoder.mid_block.attentions.0.proj");
    const int hw = h * w;
    rms_norm_nchw(feat.data, gamma.data, out_buf.data, n, kMid, hw, s);
    DeviceTensor qkv_t({n, 3 * kMid, h, w});
    conv2d_1x1(out_buf.data, qkv.w.data, qkv.b.data, qkv_t.data, n, kMid, 3 * kMid, hw, s);
    DeviceTensor q({n, kMid, h, w}), kk({n, kMid, h, w}), vv({n, kMid, h, w});
    split_qkv_nchw(qkv_t.data, q.data, kk.data, vv.data, n, kMid, hw, s);
    // flash kernel treats leading dims as (tokens, heads, d): use [hw,1,384].
    DeviceTensor attn({hw, kMid});
    flash_attn(q.data, kk.data, vv.data, attn.data, 1, 1, hw, hw, kMid, s);
    DeviceTensor attn_nchw({n, kMid, h, w});
    IMG_CUDA_CHECK(cudaMemcpyAsync(attn_nchw.data, attn.data,
                                   (size_t)kMid * hw * sizeof(Bf16),
                                   cudaMemcpyDeviceToDevice, s));
    conv2d_1x1(attn_nchw.data, proj.w.data, proj.b.data, out_buf.data, n, kMid, kMid, hw, s);
    add_residual(feat.data, out_buf.data, (size_t)kMid * hw, s);
  }
  resnet_at("decoder.mid_block.resnets.1", kMid, kMid, h, w, false);

  // Up blocks. State per spec table:
  //  block0: 384->384 x3 at r,    upsample 384->192 to 2r
  //  block1: res0 192->384 (shortcut), then 384 x2 at 2r, upsample 384->192 to 4r
  //  block2: 192->192 x3 at 4r,   upsample 192->96  to 8r
  //  block3: 96->96 x3 at 8r, no upsampler.
  struct Stage { int block, ci0, co, r; bool sc0; };
  // sc0 must be set explicitly: an omitted trailing aggregate member defaults
  // to false, which would skip block1 resnets.0's channel-change shortcut.
  const Stage stages[3] = {{0, kMid, kMid, 1, false},
                           {1, 192, kMid, 2, true},
                           {2, 192, 192, 4, false}};
  for (const Stage& stg : stages) {
    const std::string bp = "decoder.up_blocks." + std::to_string(stg.block);
    const int hh = h * stg.r, ww = w * stg.r;
    resnet_at(bp + ".resnets.0", stg.ci0, stg.co, hh, ww, stg.sc0);
    resnet_at(bp + ".resnets.1", stg.co, stg.co, hh, ww, false);
    resnet_at(bp + ".resnets.2", stg.co, stg.co, hh, ww, false);
    const int ch = stg.co, out_ch = ch / 2;
    DeviceTensor up({n, ch, hh * 2, ww * 2});
    upsample_nearest_2x(feat.data, up.data, n, ch, hh * 2, ww * 2, s);
    Conv2dW uc = load_conv2d(*st_, bp + ".upsamplers.0.resample.1");
    DeviceTensor up_conv({n, out_ch, hh * 2, ww * 2});
    conv2d_3x3(up.data, uc.w.data, uc.b.data, up_conv.data, n, ch, out_ch, hh * 2, ww * 2, s);
    feat = std::move(up_conv);
  }
  {
    const std::string bp = "decoder.up_blocks.3";
    resnet_at(bp + ".resnets.0", 96, 96, H8, W8, false);
    resnet_at(bp + ".resnets.1", 96, 96, H8, W8, false);
    resnet_at(bp + ".resnets.2", 96, 96, H8, W8, false);
  }

  // Head: RMS norm -> SiLU -> conv_out 96 -> 3.
  DeviceTensor gamma = load_bf16(*st_, "decoder.norm_out.gamma");
  DeviceTensor normed({n, 96, H8, W8});
  rms_norm_nchw(feat.data, gamma.data, normed.data, n, 96, H8 * W8, s);
  silu_inplace(normed.data, (size_t)96 * H8 * W8, s);
  DeviceTensor rgb({n, 3, H8, W8});
  Conv2dW conv_out = load_conv3d(*st_, "decoder.conv_out");
  conv2d_3x3(normed.data, conv_out.w.data, conv_out.b.data, rgb.data, n, 96, 3, H8, W8, s);
  clamp_n1p1(rgb.data, rgb.numel, s);
  rt_.sync();
  return rgb;
}

}  // namespace umminfer::image
