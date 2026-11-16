// SPDX-License-Identifier: Apache-2.0
#include "codec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include "cuda_kernels.hpp"

namespace umminfer::native {
namespace {

constexpr int kCodebook = 2048;
constexpr int kVqDim = 256;
constexpr int kLatent = 512;
constexpr int kHidden = 1024;     // conv / upsample channels
constexpr int kTrHidden = 512;   // pre-transformer hidden
constexpr int kTrHeads = 16;
constexpr int kTrKvHeads = 16;  // MHA: q/k/v proj all output 1024 = 16*64
constexpr int kHeadDim = 64;
constexpr int kTrLayers = 8;
constexpr int kWindow = 72;
constexpr int kGroups = 16;
constexpr int kDecoderDim = 1536;
constexpr int kUpsample = 1920;
constexpr int kMaxFrames = 8192;

constexpr float kRmsEps = 1e-5f;
constexpr float kLnEps = 1e-6f;
constexpr float kVqEps = 1e-5f;

const float* need(const std::unordered_map<std::string, const float*>& w,
                  const std::string& name) {
  auto it = w.find(name);
  if (it == w.end()) throw std::runtime_error("codec missing tensor: " + name);
  return it->second;
}

}  // namespace

struct CodecDecoder::Impl {
  std::unordered_map<std::string, const float*> w;
  cudaStream_t stream;

  float* codebooks = nullptr;  // 16 dequantized tables, each [2048,256]
  // Generic channel-last / channel-first scratch, grown to the waveform size.
  float *a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr, *e = nullptr;
  float *rvq_first = nullptr, *rvq_rest = nullptr;  // dedicated [512,T] outputs
  float *q = nullptr, *k = nullptr, *v = nullptr, *attn = nullptr;
  float *gate = nullptr, *up = nullptr, *ffn_out = nullptr;
  int* d_codes_all = nullptr;
  int* d_ids = nullptr;
  size_t cap = 0;

  const float* wt(const std::string& n) const { return need(w, n); }

  void ensure(size_t elements) {
    if (elements <= cap) return;
    for (float** p : {&a, &b, &c, &d, &e}) {
      if (*p) UMM_CUDA_CHECK(cudaFree(*p));
      UMM_CUDA_CHECK(cudaMalloc(p, elements * sizeof(float)));
    }
    cap = elements;
  }

  void linear(float* y, const float* x, const float* weight, const float* bias,
              int m, int in, int out) {
    gemm_t<float>(y, x, weight, bias, m, in, out, stream);
  }
};

CodecDecoder::CodecDecoder(std::unordered_map<std::string, const float*> weights,
                           cudaStream_t stream) {
  impl_ = std::make_unique<Impl>();
  Impl& im = *impl_;
  im.w = std::move(weights);
  im.stream = stream;

  UMM_CUDA_CHECK(cudaMalloc(&im.codebooks,
                            kGroups * kCodebook * kVqDim * sizeof(float)));
  auto dequant = [&](const std::string& qname, int layer, float* dst) {
    const std::string base =
        "decoder.quantizer." + qname + ".vq.layers." + std::to_string(layer);
    codebook_dequant_f32(im.wt(base + "._codebook.embedding_sum"),
                         im.wt(base + "._codebook.cluster_usage"), dst, kCodebook,
                         kVqDim, kVqEps, stream);
  };
  dequant("rvq_first", 0, im.codebooks);
  for (int j = 0; j < 15; ++j)
    dequant("rvq_rest", j, im.codebooks + (j + 1) * kCodebook * kVqDim);
  if (const char* dump = std::getenv("UMM_CODEC_DUMP_CB")) {
    cudaStreamSynchronize(stream);
    std::vector<float> h(kCodebook * kVqDim);
    cudaMemcpy(h.data(), im.codebooks, h.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    std::ofstream f(dump, std::ios::binary);
    f.write(reinterpret_cast<const char*>(h.data()), h.size() * sizeof(float));
  }

  const size_t tr = static_cast<size_t>(kMaxFrames) * kHidden;
  UMM_CUDA_CHECK(cudaMalloc(&im.d_codes_all,
                            kMaxFrames * kGroups * sizeof(int)));
  UMM_CUDA_CHECK(cudaMalloc(&im.d_ids, kMaxFrames * sizeof(int)));
  UMM_CUDA_CHECK(cudaMalloc(&im.rvq_first, kMaxFrames * kLatent * sizeof(float)));
  UMM_CUDA_CHECK(cudaMalloc(&im.rvq_rest, kMaxFrames * kLatent * sizeof(float)));
  UMM_CUDA_CHECK(cudaMalloc(&im.q, kMaxFrames * kTrHeads * kHeadDim * sizeof(float)));
  UMM_CUDA_CHECK(cudaMalloc(&im.k, kMaxFrames * kTrKvHeads * kHeadDim * sizeof(float)));
  UMM_CUDA_CHECK(cudaMalloc(&im.v, kMaxFrames * kTrKvHeads * kHeadDim * sizeof(float)));
  UMM_CUDA_CHECK(cudaMalloc(&im.attn, kMaxFrames * kTrHeads * kHeadDim * sizeof(float)));
  UMM_CUDA_CHECK(cudaMalloc(&im.gate, tr * sizeof(float)));
  UMM_CUDA_CHECK(cudaMalloc(&im.up, tr * sizeof(float)));
  UMM_CUDA_CHECK(cudaMalloc(&im.ffn_out, tr * sizeof(float)));
}

CodecDecoder::~CodecDecoder() {
  if (!impl_) return;
  Impl& im = *impl_;
  cudaFree(im.ffn_out);
  cudaFree(im.up);
  cudaFree(im.gate);
  cudaFree(im.attn);
  cudaFree(im.v);
  cudaFree(im.k);
  cudaFree(im.q);
  cudaFree(im.d_ids);
  cudaFree(im.d_codes_all);
  cudaFree(im.rvq_first);
  cudaFree(im.rvq_rest);
  cudaFree(im.codebooks);
  for (float* p : {im.a, im.b, im.c, im.d, im.e}) cudaFree(p);
}

std::vector<float> CodecDecoder::decode(const std::vector<int>& codes, int T) {
  Impl& im = *impl_;
  cudaStream_t st = im.stream;
  if (T <= 0 || T > kMaxFrames)
    throw std::invalid_argument("frame count out of range");
  if (static_cast<int>(codes.size()) != T * kGroups)
    throw std::invalid_argument("codes must be [T,16]");
  for (int id : codes)
    if (id < 0 || id >= kCodebook)
      throw std::runtime_error("codec id outside [0,2047]");
  im.ensure(static_cast<size_t>(kDecoderDim) * T * 480);  // biggest conv stream
  UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_codes_all, codes.data(),
                                 T * kGroups * sizeof(int), cudaMemcpyHostToDevice,
                                 st));

  // ---- RVQ dequantization ----------------------------------------------------
  // Each branch gathers its group codebooks [T,256] into im.a, transposes to
  // im.b [256,T], applies the 1x1 output projection into a dedicated [512,T]
  // buffer. Dedicated outputs avoid aliasing the generic a..e scratch.
  auto branch = [&](int first_group, int count, const std::string& proj,
                    float* out512) {
    for (int g = 0; g < count; ++g) {
      const int grp = first_group + g;
      gather_code_column(im.d_codes_all, im.d_ids, T, kGroups, grp, st);
      gather_rows_t<float>(im.codebooks + grp * kCodebook * kVqDim, im.d_ids,
                           im.e, T, kVqDim, st);  // [T,256]
      if (grp == 0 && std::getenv("UMM_CODEC_DUMP_G0")) {
        std::vector<float> h(T * kVqDim);
        cudaMemcpy(h.data(), im.e, h.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::ofstream o(std::getenv("UMM_CODEC_DUMP_G0"), std::ios::binary);
        o.write(reinterpret_cast<const char*>(h.data()), h.size() * sizeof(float));
      }
      if (g == 0)
        cudaMemcpyAsync(im.a, im.e, T * kVqDim * sizeof(float),
                        cudaMemcpyDeviceToDevice, st);
      else
        add_residual_t<float>(im.a, im.e, T * kVqDim, st);
    }
    transpose_inv_f32(im.a, im.b, kVqDim, T, st);  // [T,256] -> [256,T]
    if (first_group == 0 && std::getenv("UMM_CODEC_DUMP_SUM")) {
      std::vector<float> h(T * kVqDim);
      cudaMemcpy(h.data(), im.a, h.size() * sizeof(float), cudaMemcpyDeviceToHost);
      std::ofstream o(std::string(std::getenv("UMM_CODEC_DUMP_SUM")) +
                          (count == 1 ? "_first" : "_rest"),
                      std::ios::binary);
      o.write(reinterpret_cast<const char*>(h.data()), h.size() * sizeof(float));
    }
    conv1d_causal_f32(im.b, im.wt("decoder.quantizer." + proj + ".output_proj.weight"),
                      nullptr, out512, kVqDim, kLatent, T, 1, 1, 1, st);  // [512,T]
    if (std::getenv("UMM_CODEC_DUMP_PROJ")) {
      std::vector<float> h(T * kLatent);
      cudaMemcpy(h.data(), out512, h.size() * sizeof(float), cudaMemcpyDeviceToHost);
      std::ofstream o(std::string(std::getenv("UMM_CODEC_DUMP_PROJ")) + "_" + proj,
                      std::ios::binary);
      o.write(reinterpret_cast<const char*>(h.data()), h.size() * sizeof(float));
    }
  };
  branch(0, 1, "rvq_first", im.rvq_first);
  branch(1, 15, "rvq_rest", im.rvq_rest);
  // Latent [512,T] in im.d = first + rest.
  UMM_CUDA_CHECK(cudaMemcpyAsync(im.d, im.rvq_first,
                                 T * kLatent * sizeof(float),
                                 cudaMemcpyDeviceToDevice, st));
  add_residual_t<float>(im.d, im.rvq_rest, T * kLatent, st);

  if (const char* dump = std::getenv("UMM_CODEC_DUMP_Q")) {
    std::vector<float> h(T * kLatent);
    cudaMemcpy(h.data(), im.d, h.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::ofstream f(dump, std::ios::binary);
    f.write(reinterpret_cast<const char*>(h.data()), h.size() * sizeof(float));
  }

  // ---- pre_conv then pre-transformer ----------------------------------------
  conv1d_causal_f32(im.d, im.wt("decoder.pre_conv.conv.weight"),
                    im.wt("decoder.pre_conv.conv.bias"), im.c, kLatent, kHidden, T,
                    3, 1, 1, st);  // [1024,T]
  if (const char* dump = std::getenv("UMM_CODEC_DUMP_PRECONV")) {
    std::vector<float> h(T * kHidden);
    cudaMemcpy(h.data(), im.c, h.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::ofstream f(dump, std::ios::binary);
    f.write(reinterpret_cast<const char*>(h.data()), h.size() * sizeof(float));
  }
  transpose_f32(im.c, im.a, kHidden, T, st);
  im.linear(im.ffn_out, im.a, im.wt("decoder.pre_transformer.input_proj.weight"),
            im.wt("decoder.pre_transformer.input_proj.bias"), T, kHidden, kTrHidden);

  std::vector<float> inv(kHeadDim / 2);
  for (int i = 0; i < kHeadDim / 2; ++i)
    inv[i] = 1.0f / std::pow(10000.0f, 2.0f * i / kHeadDim);
  float* d_inv;
  int* d_pos;
  UMM_CUDA_CHECK(cudaMalloc(&d_inv, inv.size() * sizeof(float)));
  UMM_CUDA_CHECK(cudaMalloc(&d_pos, T * sizeof(int)));
  std::vector<int> pos(T);
  for (int i = 0; i < T; ++i) pos[i] = i;
  UMM_CUDA_CHECK(cudaMemcpyAsync(d_inv, inv.data(), inv.size() * sizeof(float),
                                 cudaMemcpyHostToDevice, st));
  UMM_CUDA_CHECK(cudaMemcpyAsync(d_pos, pos.data(), T * sizeof(int),
                                 cudaMemcpyHostToDevice, st));

  float* hidden = im.ffn_out;  // reuse as the transformer residual stream
  for (int l = 0; l < kTrLayers; ++l) {
    const std::string p = "decoder.pre_transformer.layers." + std::to_string(l);
    rms_norm_t<float>(hidden, im.wt(p + ".input_layernorm.weight"), im.a, T,
                      kTrHidden, kRmsEps, st);
    im.linear(im.q, im.a, im.wt(p + ".self_attn.q_proj.weight"), nullptr, T,
              kTrHidden, kTrHeads * kHeadDim);
    im.linear(im.k, im.a, im.wt(p + ".self_attn.k_proj.weight"), nullptr, T,
              kTrHidden, kTrKvHeads * kHeadDim);
    im.linear(im.v, im.a, im.wt(p + ".self_attn.v_proj.weight"), nullptr, T,
              kTrHidden, kTrKvHeads * kHeadDim);
    rope_half_t<float>(im.q, d_pos, d_inv, T, kTrHeads, kHeadDim, st);
    rope_half_t<float>(im.k, d_pos, d_inv, T, kTrKvHeads, kHeadDim, st);
    gqa_attention_t<float>(im.q, im.k, im.v, im.attn, T, T, kTrHeads, kTrKvHeads,
                           kHeadDim, 1, kWindow, 0, st);
    if (l == 0 && std::getenv("UMM_CODEC_DUMP_ATTN0")) {
      std::vector<float> h(T * kTrHeads * kHeadDim);
      cudaMemcpy(h.data(), im.attn, h.size() * sizeof(float),
                 cudaMemcpyDeviceToHost);
      std::ofstream o(std::getenv("UMM_CODEC_DUMP_ATTN0"), std::ios::binary);
      o.write(reinterpret_cast<const char*>(h.data()), h.size() * sizeof(float));
    }
    im.linear(im.a, im.attn, im.wt(p + ".self_attn.o_proj.weight"), nullptr, T,
              kTrHeads * kHeadDim, kTrHidden);
    layer_scale_t<float>(im.a, im.a, im.wt(p + ".self_attn_layer_scale.scale"), T,
                         kTrHidden, st);
    add_residual_t<float>(hidden, im.a, T * kTrHidden, st);
    if (l == 0 && std::getenv("UMM_CODEC_DUMP_ATTRES")) {
      std::vector<float> h(T * kTrHidden);
      cudaMemcpy(h.data(), hidden, h.size() * sizeof(float),
                 cudaMemcpyDeviceToHost);
      std::ofstream o(std::getenv("UMM_CODEC_DUMP_ATTRES"), std::ios::binary);
      o.write(reinterpret_cast<const char*>(h.data()), h.size() * sizeof(float));
    }

    rms_norm_t<float>(hidden, im.wt(p + ".post_attention_layernorm.weight"), im.a, T,
                      kTrHidden, kRmsEps, st);
    im.linear(im.gate, im.a, im.wt(p + ".mlp.gate_proj.weight"), nullptr, T,
              kTrHidden, 2 * kTrHidden);
    im.linear(im.up, im.a, im.wt(p + ".mlp.up_proj.weight"), nullptr, T, kTrHidden,
              2 * kTrHidden);
    silu_mul_t<float>(im.gate, im.gate, im.up, T * 2 * kTrHidden, st);
    im.linear(im.a, im.gate, im.wt(p + ".mlp.down_proj.weight"), nullptr, T,
              2 * kTrHidden, kTrHidden);
    layer_scale_t<float>(im.a, im.a, im.wt(p + ".mlp_layer_scale.scale"), T,
                         kTrHidden, st);
    add_residual_t<float>(hidden, im.a, T * kTrHidden, st);
    if (l == 0 && std::getenv("UMM_CODEC_DUMP_L0")) {
      std::vector<float> h(T * kTrHidden);
      cudaMemcpy(h.data(), hidden, h.size() * sizeof(float),
                 cudaMemcpyDeviceToHost);
      std::ofstream o(std::getenv("UMM_CODEC_DUMP_L0"), std::ios::binary);
      o.write(reinterpret_cast<const char*>(h.data()), h.size() * sizeof(float));
    }
  }
  rms_norm_t<float>(hidden, im.wt("decoder.pre_transformer.norm.weight"), im.a, T,
                    kTrHidden, kRmsEps, st);
  im.linear(im.ffn_out, im.a, im.wt("decoder.pre_transformer.output_proj.weight"),
            im.wt("decoder.pre_transformer.output_proj.bias"), T, kTrHidden, kHidden);
  cudaFree(d_inv);
  cudaFree(d_pos);
  if (const char* dump = std::getenv("UMM_CODEC_DUMP_TR")) {
    std::vector<float> h(T * kHidden);
    cudaMemcpy(h.data(), im.ffn_out, h.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    std::ofstream f(dump, std::ios::binary);
    f.write(reinterpret_cast<const char*>(h.data()), h.size() * sizeof(float));
  }
  transpose_inv_f32(im.ffn_out, im.a, kHidden, T, st);  // [T,1024] -> [1024,T]

  // ---- two ConvNeXt x2 upsample stages --------------------------------------
  int len = T;
  for (int s = 0; s < 2; ++s) {
    const std::string up = "decoder.upsample." + std::to_string(s) + ".0";
    const std::string cx = "decoder.upsample." + std::to_string(s) + ".1";
    conv_transpose1d_f32(im.a, im.wt(up + ".conv.weight"),
                         im.wt(up + ".conv.bias"), im.b, kHidden, kHidden, len, 2, 2,
                         st);
    len *= 2;
    UMM_CUDA_CHECK(cudaMemcpyAsync(im.c, im.b,
                                   static_cast<size_t>(len) * kHidden * sizeof(float),
                                   cudaMemcpyDeviceToDevice, st));  // residual
    conv1d_causal_f32(im.c, im.wt(cx + ".dwconv.conv.weight"),
                      im.wt(cx + ".dwconv.conv.bias"), im.d, kHidden, kHidden, len,
                      7, 1, kHidden, st);
    transpose_f32(im.d, im.a, kHidden, len, st);
    layer_norm_t<float>(im.a, im.wt(cx + ".norm.weight"), im.wt(cx + ".norm.bias"),
                        im.a, len, kHidden, kLnEps, st);
    im.linear(im.gate, im.a, im.wt(cx + ".pwconv1.weight"),
              im.wt(cx + ".pwconv1.bias"), len, kHidden, 4 * kHidden);
    gelu_inplace_t<float>(im.gate, len * 4 * kHidden, st);
    im.linear(im.a, im.gate, im.wt(cx + ".pwconv2.weight"),
              im.wt(cx + ".pwconv2.bias"), len, 4 * kHidden, kHidden);
    layer_scale_t<float>(im.a, im.a, im.wt(cx + ".gamma"), len, kHidden, st);
    transpose_inv_f32(im.a, im.d, kHidden, len, st);  // [len,1024] -> [1024,len]
    add_residual_t<float>(im.b, im.d, len * kHidden, st);
    std::swap(im.a, im.b);
  }

  // ---- waveform decoder ------------------------------------------------------
  if (const char* dump = std::getenv("UMM_CODEC_DUMP_UP")) {
    std::vector<float> h(static_cast<size_t>(len) * kHidden);
    cudaMemcpy(h.data(), im.a, h.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::ofstream f(dump, std::ios::binary);
    f.write(reinterpret_cast<const char*>(h.data()), h.size() * sizeof(float));
  }
  conv1d_causal_f32(im.a, im.wt("decoder.decoder.0.conv.weight"),
                    im.wt("decoder.decoder.0.conv.bias"), im.b, kHidden, kDecoderDim,
                    len, 7, 1, 1, st);
  if (const char* dump = std::getenv("UMM_CODEC_DUMP_D0")) {
    std::vector<float> h(static_cast<size_t>(len) * kDecoderDim);
    cudaMemcpy(h.data(), im.b, h.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::ofstream f(dump, std::ios::binary);
    f.write(reinterpret_cast<const char*>(h.data()), h.size() * sizeof(float));
  }
  const int rates[4] = {8, 5, 4, 3};
  int in_ch = kDecoderDim;
  for (int i = 0; i < 4; ++i) {
    const int out_ch = kDecoderDim / (1 << (i + 1));
    const std::string blk = "decoder.decoder." + std::to_string(i + 1);
    snake_beta_f32(im.b, im.wt(blk + ".block.0.alpha"),
                   im.wt(blk + ".block.0.beta"), in_ch, len, st);
    conv_transpose1d_f32(im.b, im.wt(blk + ".block.1.conv.weight"),
                         im.wt(blk + ".block.1.conv.bias"), im.c, in_ch, out_ch, len,
                         2 * rates[i], rates[i], st);
    len *= rates[i];
    const int dilations[3] = {1, 3, 9};
    for (int u = 0; u < 3; ++u) {
      const std::string ru = blk + ".block." + std::to_string(u + 2);
      // Preserve the residual in im.d; snake_beta is in-place so the unit path
      // works on copies: im.c -> im.a (act1+conv1) -> im.e (act2+conv2).
      cudaMemcpyAsync(im.d, im.c,
                      static_cast<size_t>(len) * out_ch * sizeof(float),
                      cudaMemcpyDeviceToDevice, st);  // residual
      snake_beta_f32(im.c, im.wt(ru + ".act1.alpha"), im.wt(ru + ".act1.beta"),
                     out_ch, len, st);
      conv1d_causal_f32(im.c, im.wt(ru + ".conv1.conv.weight"),
                        im.wt(ru + ".conv1.conv.bias"), im.a, out_ch, out_ch, len, 7,
                        dilations[u], 1, st);
      snake_beta_f32(im.a, im.wt(ru + ".act2.alpha"), im.wt(ru + ".act2.beta"),
                     out_ch, len, st);
      conv1d_causal_f32(im.a, im.wt(ru + ".conv2.conv.weight"),
                        im.wt(ru + ".conv2.conv.bias"), im.e, out_ch, out_ch, len, 1,
                        1, 1, st);
      add_residual_t<float>(im.e, im.d, len * out_ch, st);  // e = unit + residual
      std::swap(im.c, im.e);  // result back in im.c
    }
    std::swap(im.b, im.c);
    in_ch = out_ch;
    if (const char* dump = std::getenv("UMM_CODEC_DUMP_DB")) {
      std::vector<float> h(static_cast<size_t>(len) * out_ch);
      cudaMemcpy(h.data(), im.b, h.size() * sizeof(float), cudaMemcpyDeviceToHost);
      std::ofstream f(std::string(dump) + "_" + std::to_string(i),
                      std::ios::binary);
      f.write(reinterpret_cast<const char*>(h.data()), h.size() * sizeof(float));
    }
  }
  snake_beta_f32(im.b, im.wt("decoder.decoder.5.alpha"),
                 im.wt("decoder.decoder.5.beta"), 96, len, st);
  conv1d_causal_f32(im.b, im.wt("decoder.decoder.6.conv.weight"),
                    im.wt("decoder.decoder.6.conv.bias"), im.a, 96, 1, len, 7, 1, 1,
                    st);
  const int samples = T * kUpsample;
  clamp_f32(im.a, samples, -1.0f, 1.0f, st);

  std::vector<float> wav(samples);
  UMM_CUDA_CHECK(cudaMemcpyAsync(wav.data(), im.a, samples * sizeof(float),
                                 cudaMemcpyDeviceToHost, st));
  UMM_CUDA_CHECK(cudaStreamSynchronize(st));
  return wav;
}

}  // namespace umminfer::native
