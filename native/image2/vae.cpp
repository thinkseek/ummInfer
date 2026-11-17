// SPDX-License-Identifier: Apache-2.0
#include "vae.hpp"

#include "cuda_kernels.hpp"
#include "weights.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>

namespace umminfer::image2 {
namespace {

struct ConvW {
  DeviceTensor w, b;
};
struct ResnetW {
  DeviceTensor g1, g2;
  ConvW c1, c2, shortcut;
  bool has_shortcut = false;
};

}  // namespace

DeviceTensor VaeDecoder::load_conv_weight(const std::string& name) const {
  DeviceTensor raw = load_bf16(*st_, name);
  if (raw.shape.size() == 4) return raw;
  if (raw.shape.size() != 5) throw std::invalid_argument("conv weight rank: " + name);
  const auto& sh = raw.shape;  // [co, ci, kd, kh, kw]
  const int co = static_cast<int>(sh[0]), ci = static_cast<int>(sh[1]);
  const int kd = static_cast<int>(sh[2]), kh = static_cast<int>(sh[3]),
            kw = static_cast<int>(sh[4]);
  const int time_idx = kd == 1 ? 0 : kd - 1;  // causal pad: last slice
  DeviceTensor out({co, ci, kh, kw});
  auto src = raw.to_host_u16();
  std::vector<uint16_t> dst(static_cast<size_t>(co) * ci * kh * kw);
  const size_t plane = static_cast<size_t>(kh) * kw;
  for (int o = 0; o < co; ++o)
    for (int c = 0; c < ci; ++c) {
      const size_t s = ((static_cast<size_t>(o) * ci + c) * kd + time_idx) * plane;
      std::copy_n(src.begin() + s, plane, dst.begin() + (static_cast<size_t>(o) * ci + c) * plane);
    }
  out.copy_from_host(dst.data());
  return out;
}

VaeDecoder::VaeDecoder(CudaRuntime& rt, const std::string& dir)
    : rt_(rt), st_(std::make_unique<ShardedSafeTensors>(dir)) {}

namespace {
void maybe_dump(const char* name, const DeviceTensor& t, int c, int h, int w) {
  const char* dir = std::getenv("IMAGE2_VAE_DUMP_DIR");
  if (!dir) return;
  auto host = t.to_host_u16();
  const std::string path = std::string(dir) + "/" + name + ".bin";
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot open " + path);
  int32_t dims[3] = {c, h, w};
  std::fwrite(dims, 4, 3, f);
  std::fwrite(host.data(), 2, host.size(), f);
  std::fclose(f);
}

ConvW load_conv(const ShardedSafeTensors& st, const VaeDecoder& vae, const std::string& p,
                bool five_d) {
  ConvW c;
  c.w = five_d ? vae.load_conv_weight(p + ".weight") : load_bf16(st, p + ".weight");
  c.b = load_bf16(st, p + ".bias");
  return c;
}

// ResidualBlock with NCHW tensors; returns a fresh output.
DeviceTensor run_resnet(CudaRuntime& rt, const ShardedSafeTensors& st, const VaeDecoder& vae,
                        const std::string& p, const DeviceTensor& x, int n, int ci, int co,
                        int hh, int ww, const char* tag = "") {
  auto& s = rt.stream;
  ConvW c1 = load_conv(st, vae, p + ".conv1", true);
  ConvW c2 = load_conv(st, vae, p + ".conv2", true);
  DeviceTensor g1 = load_bf16(st, p + ".norm1.gamma");
  DeviceTensor g2 = load_bf16(st, p + ".norm2.gamma");
  const bool shortcut = ci != co;

  DeviceTensor shortcut_t;
  if (shortcut) {
    shortcut_t = DeviceTensor({n, co, hh, ww});
    ConvW sc = load_conv(st, vae, p + ".conv_shortcut", true);
    conv2d_1x1(x.data, sc.w.data, sc.b.data, shortcut_t.data, n, ci, co, hh * ww, s);
    if (tag[0]) maybe_dump((std::string(tag) + "_sc").c_str(), shortcut_t, co, hh, ww);
  }

  DeviceTensor normed({n, ci, hh, ww});
  rms_norm_nchw(x.data, g1.data, normed.data, n, ci, hh * ww, s);
  silu_inplace(normed.data, (size_t)n * ci * hh * ww, s);
  if (tag[0]) maybe_dump((std::string(tag) + "_n1").c_str(), normed, ci, hh, ww);
  DeviceTensor h1({n, co, hh, ww});
  conv2d_3x3(normed.data, c1.w.data, c1.b.data, h1.data, n, ci, co, hh, ww, s);
  if (tag[0]) maybe_dump((std::string(tag) + "_c1").c_str(), h1, co, hh, ww);

  DeviceTensor normed2({n, co, hh, ww});
  rms_norm_nchw(h1.data, g2.data, normed2.data, n, co, hh * ww, s);
  silu_inplace(normed2.data, (size_t)n * co * hh * ww, s);
  if (tag[0]) maybe_dump((std::string(tag) + "_n2").c_str(), normed2, co, hh, ww);
  DeviceTensor conv_out({n, co, hh, ww});
  conv2d_3x3(normed2.data, c2.w.data, c2.b.data, conv_out.data, n, co, co, hh, ww, s);
  if (tag[0]) maybe_dump((std::string(tag) + "_c2").c_str(), conv_out, co, hh, ww);
  add_residual(conv_out.data, shortcut ? shortcut_t.data : x.data,
               (size_t)n * co * hh * ww, s);
  return conv_out;
}

// One ResidualUpBlock: resnets, spatial upsample + 3x3 conv, DupUp shortcut.
DeviceTensor run_up_block(CudaRuntime& rt, const ShardedSafeTensors& st,
                          const VaeDecoder& vae, const std::string& bp,
                          const DeviceTensor& x, int ci, int co, int hh, int ww,
                          bool up_flag, int factor_t, const char* tag) {
  auto& s = rt.stream;
  const std::string r0tag = std::string(tag) + "_r0";
  DeviceTensor cur = run_resnet(rt, st, vae, bp + ".resnets.0", x, 1, ci, co, hh, ww,
                                r0tag.c_str());
  {
    std::string n = std::string(tag) + "_r0";
    maybe_dump(n.c_str(), cur, co, hh, ww);
  }
  cur = run_resnet(rt, st, vae, bp + ".resnets.1", cur, 1, co, co, hh, ww);
  {
    std::string n = std::string(tag) + "_r1";
    maybe_dump(n.c_str(), cur, co, hh, ww);
  }
  cur = run_resnet(rt, st, vae, bp + ".resnets.2", cur, 1, co, co, hh, ww);
  {
    std::string n = std::string(tag) + "_res";
    maybe_dump(n.c_str(), cur, co, hh, ww);
  }
  if (!up_flag) return cur;
  // main path: nearest 2x then 3x3 conv (dim -> dim)
  DeviceTensor up({1, co, hh * 2, ww * 2});
  upsample_nearest_2x_fp32(cur.data, up.data, 1, co, hh * 2, ww * 2, s);
  ConvW uc = load_conv(st, vae, bp + ".upsampler.resample.1", false);
  DeviceTensor conv_up({1, co, hh * 2, ww * 2});
  conv2d_3x3(up.data, uc.w.data, uc.b.data, conv_up.data, 1, co, co, hh * 2, ww * 2, s);

  {
    std::string n = std::string(tag) + "_conv";
    maybe_dump(n.c_str(), conv_up, co, hh * 2, ww * 2);
  }
  DeviceTensor shortcut({1, co, hh * 2, ww * 2});
  dupup_2x(x.data, shortcut.data, ci, co, hh, ww, factor_t, s);
  add_residual(conv_up.data, shortcut.data, (size_t)co * hh * 2 * ww * 2, s);
  return conv_up;
}

}  // namespace

DeviceTensor VaeDecoder::decode(const DeviceTensor& lat, int h, int w) {
  auto& s = rt_.stream;
  const int n = 1;

  ConvW pq = load_conv(*st_, *this, "post_quant_conv", true);
  DeviceTensor x({n, kZ, h, w});
  conv2d_1x1(lat.data, pq.w.data, pq.b.data, x.data, n, kZ, kZ, h * w, s);
  maybe_dump("post_quant", x, kZ, h, w);

  const int c0 = kBase * 8;  // 1152
  ConvW ci = load_conv(*st_, *this, "decoder.conv_in", true);
  DeviceTensor feat({n, c0, h, w});
  conv2d_3x3(x.data, ci.w.data, ci.b.data, feat.data, n, kZ, c0, h, w, s);
  maybe_dump("conv_in", feat, c0, h, w);

  feat = run_resnet(rt_, *st_, *this, "decoder.mid_block.resnets.0", feat, n, c0, c0, h, w);
  maybe_dump("mid_res0", feat, c0, h, w);

  // Spatial attention at the latent resolution.
  {
    DeviceTensor gamma = load_bf16(*st_, "decoder.mid_block.attentions.0.norm.gamma");
    ConvW qkv = load_conv(*st_, *this, "decoder.mid_block.attentions.0.to_qkv", false);
    ConvW proj = load_conv(*st_, *this, "decoder.mid_block.attentions.0.proj", false);
    const int hw = h * w;
    DeviceTensor normed({n, c0, h, w});
    rms_norm_nchw(feat.data, gamma.data, normed.data, n, c0, hw, s);
    DeviceTensor qkv_t({n, 3 * c0, h, w});
    conv2d_1x1(normed.data, qkv.w.data, qkv.b.data, qkv_t.data, n, c0, 3 * c0, hw, s);
    DeviceTensor q({n, c0, h, w}), k({n, c0, h, w}), v({n, c0, h, w});
    split_qkv_nchw(qkv_t.data, q.data, k.data, v.data, n, c0, hw, s);
    DeviceTensorF scores(static_cast<size_t>(hw) * hw);
    DeviceTensorF attn_tok(static_cast<size_t>(hw) * c0);
    DeviceTensor attn({n, c0, h, w});
    vae_attn_f32(q.data, k.data, v.data, scores.data, attn_tok.data, attn.data, c0, hw, s);
    DeviceTensor proj_out({n, c0, h, w});
    conv2d_1x1(attn.data, proj.w.data, proj.b.data, proj_out.data, n, c0, c0, hw, s);
    add_residual(feat.data, proj_out.data, (size_t)n * c0 * hw, s);
    maybe_dump("mid_attn", feat, c0, h, w);
  }

  feat = run_resnet(rt_, *st_, *this, "decoder.mid_block.resnets.1", feat, n, c0, c0, h, w);
  maybe_dump("mid_res1", feat, c0, h, w);

  struct UpCfg {
    const char* name;
    int ci, co, scale, factor_t;
    bool up_flag;
  };
  // Checkpoint channel progression: 1152 -> 1152 -> 1152 -> 576 -> 288 -> 144.
  // Blocks 0..3 upsample spatially; the first three also carry the temporal
  // DupUp factor, block 3 is upsample2d. Block 4 has no upsampler.
  const UpCfg ups[5] = {
      {"decoder.up_blocks.0", c0, c0, 1, 2, true},
      {"decoder.up_blocks.1", c0, c0, 2, 2, true},
      {"decoder.up_blocks.2", c0, c0 / 2, 4, 2, true},
      {"decoder.up_blocks.3", c0 / 2, c0 / 4, 8, 1, true},
      {"decoder.up_blocks.4", c0 / 4, c0 / 8, 16, 0, false},
  };
  for (const UpCfg& u : ups) {
    char nm[16];
    std::snprintf(nm, sizeof(nm), "up%d", static_cast<int>(&u - ups));
    feat = run_up_block(rt_, *st_, *this, u.name, feat, u.ci, u.co, h * u.scale,
                        w * u.scale, u.up_flag, u.factor_t, nm);
    maybe_dump(nm, feat, u.co, u.up_flag ? h * u.scale * 2 : h * u.scale,
               u.up_flag ? w * u.scale * 2 : w * u.scale);
  }

  const int H = h * 16, W = w * 16;
  DeviceTensor gamma = load_bf16(*st_, "decoder.norm_out.gamma");
  DeviceTensor normed({n, c0 / 8, H, W});
  rms_norm_nchw(feat.data, gamma.data, normed.data, n, c0 / 8, H * W, s);
  maybe_dump("norm_out", normed, c0 / 8, H, W);
  silu_inplace(normed.data, (size_t)c0 / 8 * H * W, s);
  ConvW co = load_conv(*st_, *this, "decoder.conv_out", true);
  DeviceTensor out({n, 4, H, W});
  conv2d_3x3(normed.data, co.w.data, co.b.data, out.data, n, c0 / 8, 4, H, W, s);
  maybe_dump("conv_out", out, 4, H, W);
  clamp_n1p1(out.data, out.numel, s);
  rt_.sync();
  return out;
}

}  // namespace umminfer::image2
