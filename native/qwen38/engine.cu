// SPDX-License-Identifier: Apache-2.0
// Qwen3.8-27B hybrid decoder: 64 dense layers, 3:1 gated-delta-net to full
// attention, partial mRoPE, dense SwiGLU, 27-block ViT with patch merger.
#include "engine.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <cstring>

#include "qwen38_ops.hpp"
#include "safetensors_mmap.hpp"

namespace umminfer::qwen38 {
namespace {

constexpr int kLayers = 64;
constexpr int kHidden = 5120;
constexpr int kQHeads = 24;
constexpr int kKvHeads = 4;
constexpr int kHeadDim = 256;
constexpr int kInter = 17408;
constexpr int kVocab = 248320;
constexpr float kEps = 1e-6f;
constexpr int kRotaryDim = 64;  // 256 * partial_rotary_factor 0.25
constexpr float kRopeTheta = 1e7f;

constexpr int kLKH = 16;   // linear key heads
constexpr int kLVH = 48;   // linear value heads
constexpr int kLKD = 128;
constexpr int kLVD = 128;
constexpr int kKeyDim = kLKH * kLKD;      // 2048
constexpr int kValueDim = kLVH * kLVD;    // 6144
constexpr int kConvDim = 2 * kKeyDim + kValueDim;  // 10240
constexpr int kConvK = 4;

constexpr int kVisBlocks = 27;
constexpr int kVisH = 1152;
constexpr int kVisHeads = 16;
constexpr int kVisHeadDim = 72;
constexpr int kVisInter = 4304;
constexpr int kVisOut = 5120;
constexpr int kPatchDim = 3 * 2 * 16 * 16;  // 1536
constexpr int kPosSide = 48;
constexpr int kVisTokenCap = 4096;
constexpr float kVisRopeTheta = 10000.0f;

constexpr int kImagePad = 248056;
constexpr int kEos = 248046;       // <|im_end|>
constexpr int kEndOfText = 248044; // <|endoftext|>

bool is_eos(int id) { return id == kEos || id == kEndOfText; }

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct Tensor {
  const void* ptr = nullptr;
  std::string dtype;
};

void write_file(const std::string& dir, const std::string& name, const void* host,
                size_t bytes) {
  std::ofstream f(dir + "/" + name, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + dir + "/" + name);
  f.write(static_cast<const char*>(host), static_cast<std::streamsize>(bytes));
}

struct Positions {
  std::vector<int> t, h, w;
  int next = 0;
};

Positions build_positions(const std::vector<int>& ids, int grid_h, int grid_w,
                          bool has_image) {
  const int n = static_cast<int>(ids.size());
  Positions p;
  p.t.resize(n);
  p.h.resize(n);
  p.w.resize(n);
  if (!has_image) {
    for (int i = 0; i < n; ++i) p.t[i] = p.h[i] = p.w[i] = i;
    p.next = n;
    return p;
  }
  int img_start = -1, img_end = -1;
  for (int i = 0; i < n; ++i)
    if (ids[i] == kImagePad) {
      if (img_start < 0) img_start = i;
      img_end = i + 1;
    }
  if (img_start < 0) throw std::runtime_error("image prompt has no image_pad run");
  const int n_img = img_end - img_start;
  const int gh = grid_h / 2, gw = grid_w / 2;
  if (n_img != gh * gw)
    throw std::runtime_error("image tokens " + std::to_string(n_img) +
                             " do not match merged grid " + std::to_string(gh * gw));
  for (int i = 0; i < img_start; ++i) p.t[i] = p.h[i] = p.w[i] = i;
  const int st = img_start;
  for (int k = 0; k < n_img; ++k) {
    p.t[img_start + k] = st;
    p.h[img_start + k] = st + k / gw;
    p.w[img_start + k] = st + k % gw;
  }
  int counter = st + std::max(gh, gw);
  for (int i = img_end; i < n; ++i) p.t[i] = p.h[i] = p.w[i] = counter++;
  p.next = counter;
  return p;
}

// mRoPE channel rule for the 32 half-slots: H occupies i%3==1 up to 33,
// W occupies i%3==2 up to 30, everything else takes the text position.
void build_text_rope(const Positions& p, int n, std::vector<Bf16>* cos,
                     std::vector<Bf16>* sin) {
  const int half = kRotaryDim / 2;  // 32
  double inv[32];
  for (int i = 0; i < half; ++i)
    inv[i] = 1.0 / std::pow(static_cast<double>(kRopeTheta),
                            static_cast<double>(2 * i) / kRotaryDim);
  cos->resize(static_cast<size_t>(n) * kRotaryDim);
  sin->resize(cos->size());
  for (int row = 0; row < n; ++row) {
    for (int i = 0; i < half; ++i) {
      int pos = p.t[row];
      if ((i % 3) == 1 && i < 33)
        pos = p.h[row];
      else if ((i % 3) == 2 && i < 30)
        pos = p.w[row];
      const double angle = inv[i] * pos;
      (*cos)[static_cast<size_t>(row) * kRotaryDim + i] = Bf16(std::cos(angle));
      (*sin)[static_cast<size_t>(row) * kRotaryDim + i] = Bf16(std::sin(angle));
    }
  }
}

// Vision axial 2D rope, fp32 table [tokens, 72], merge-block token order.
void build_vision_rope(int grid_h, int grid_w, std::vector<float>* cos,
                       std::vector<float>* sin) {
  const int half = kVisHeadDim / 2;  // 36
  const int freqs = half / 2;        // 18
  std::vector<double> inv(freqs);
  for (int i = 0; i < freqs; ++i)
    inv[i] = 1.0 / std::pow(static_cast<double>(kVisRopeTheta),
                            static_cast<double>(2 * i) / half);
  const int gh2 = grid_h / 2, gw2 = grid_w / 2;
  const int tokens = grid_h * grid_w;
  cos->assign(static_cast<size_t>(tokens) * kVisHeadDim, 0.0f);
  sin->assign(cos->size(), 0.0f);
  int row = 0;
  for (int br = 0; br < gh2; ++br)
    for (int bc = 0; bc < gw2; ++bc)
      for (int ir = 0; ir < 2; ++ir)
        for (int ic = 0; ic < 2; ++ic, ++row) {
          const int r = br * 2 + ir, c = bc * 2 + ic;
          float* cr = cos->data() + static_cast<size_t>(row) * kVisHeadDim;
          float* sr = sin->data() + static_cast<size_t>(row) * kVisHeadDim;
          for (int j = 0; j < freqs; ++j) {
            const double ah = r * inv[j], aw = c * inv[j];
            cr[j] = static_cast<float>(std::cos(ah));
            sr[j] = static_cast<float>(std::sin(ah));
            cr[freqs + j] = static_cast<float>(std::cos(aw));
            sr[freqs + j] = static_cast<float>(std::sin(aw));
          }
          for (int j = 0; j < half; ++j) {
            cr[half + j] = cr[j];
            sr[half + j] = sr[j];
          }
        }
  if (row != tokens) throw std::runtime_error("vision rope token count mismatch");
}

// Bilinear resampling of the learned 48x48 position grid with
// align_corners=True, in the same merge-block order as the patch rows.
std::vector<Bf16> interpolate_pos_embed(const std::vector<Bf16>& table, int grid_h,
                                        int grid_w) {
  const int tokens = grid_h * grid_w;
  const int gh2 = grid_h / 2, gw2 = grid_w / 2;
  std::vector<Bf16> out(static_cast<size_t>(tokens) * kVisH);
  std::vector<float> idx_h(grid_h), idx_w(grid_w);
  for (int i = 0; i < grid_h; ++i)
    idx_h[i] = grid_h > 1 ? static_cast<float>(i) * (kPosSide - 1) / (grid_h - 1) : 0.0f;
  for (int i = 0; i < grid_w; ++i)
    idx_w[i] = grid_w > 1 ? static_cast<float>(i) * (kPosSide - 1) / (grid_w - 1) : 0.0f;
  int row = 0;
  for (int br = 0; br < gh2; ++br)
    for (int bc = 0; bc < gw2; ++bc)
      for (int ir = 0; ir < 2; ++ir)
        for (int ic = 0; ic < 2; ++ic, ++row) {
          const int h = br * 2 + ir, w = bc * 2 + ic;
          const float sh = idx_h[h], sw = idx_w[w];
          const int h0 = static_cast<int>(std::floor(sh));
          const int w0 = static_cast<int>(std::floor(sw));
          const int h1 = std::min(h0 + 1, kPosSide - 1);
          const int w1 = std::min(w0 + 1, kPosSide - 1);
          const float fh = sh - h0, fw = sw - w0;
          Bf16* orow = out.data() + static_cast<size_t>(row) * kVisH;
          for (int d = 0; d < kVisH; ++d) {
            const float v00 = float(table[static_cast<size_t>(h0 * kPosSide + w0) * kVisH + d]);
            const float v01 = float(table[static_cast<size_t>(h0 * kPosSide + w1) * kVisH + d]);
            const float v10 = float(table[static_cast<size_t>(h1 * kPosSide + w0) * kVisH + d]);
            const float v11 = float(table[static_cast<size_t>(h1 * kPosSide + w1) * kVisH + d]);
            const float v = (1 - fh) * (1 - fw) * v00 + (1 - fh) * fw * v01 +
                            fh * (1 - fw) * v10 + fh * fw * v11;
            orow[d] = Bf16(v);
          }
        }
  return out;
}

}  // namespace

struct Engine::Impl {
  int gpu = 0;
  int max_seq = 8192;
  std::unique_ptr<ShardedSafetensors> shards;
  std::unordered_map<std::string, Tensor> weights;
  std::vector<Bf16> pos_table_host;  // [2304, 1152] learned vision position grid
  cudaStream_t stream = nullptr;

  // Text workspaces.
  Bf16 *hidden = nullptr, *norm = nullptr, *qg = nullptr, *q = nullptr,
       *gate = nullptr, *k = nullptr, *v = nullptr, *o = nullptr;
  Bf16 *mlp_gate = nullptr, *mlp_up = nullptr, *mlp_act = nullptr;
  Bf16 *mixed = nullptr, *mixed_conv = nullptr, *z = nullptr, *b = nullptr,
       *a = nullptr, *core = nullptr, *gated = nullptr;
  float *qn = nullptr, *kn = nullptr, *beta = nullptr, *decay = nullptr;
  Bf16 *mrope_cos = nullptr, *mrope_sin = nullptr;
  float* logits = nullptr;
  int *d_ids = nullptr, *argmax_scratch = nullptr;

  // Caches.
  Bf16* kv = nullptr;            // [16 slots][2][max_seq][4][256]
  float* rec_state = nullptr;    // [48 linear layers][48][128][128]
  Bf16* conv_state = nullptr;    // [48][10240][3]

  // Vision workspaces.
  Bf16 *d_pixels = nullptr, *v_hidden = nullptr, *v_norm = nullptr,
       *v_qkv = nullptr, *v_q = nullptr, *v_k = nullptr, *v_v = nullptr,
       *v_proj = nullptr, *v_mlp = nullptr, *v_pos = nullptr,
       *pooler = nullptr;
  float *v_rope_cos = nullptr, *v_rope_sin = nullptr;

  const Bf16* bf(const std::string& name) const {
    auto it = weights.find(name);
    if (it == weights.end()) throw std::runtime_error("missing tensor: " + name);
    if (it->second.dtype != "BF16")
      throw std::runtime_error("expected BF16: " + name + " got " + it->second.dtype);
    return static_cast<const Bf16*>(it->second.ptr);
  }
  const float* fp32(const std::string& name) const {
    auto it = weights.find(name);
    if (it == weights.end()) throw std::runtime_error("missing tensor: " + name);
    if (it->second.dtype != "F32")
      throw std::runtime_error("expected F32: " + name + " got " + it->second.dtype);
    return static_cast<const float*>(it->second.ptr);
  }

  void load_weights(const std::string& dir) {
    shards = std::make_unique<ShardedSafetensors>(dir);
    size_t bytes = 0;
    for (const auto& [name, info] : shards->tensors()) {
      const bool wanted = name.rfind("model.language_model.", 0) == 0 ||
                          name.rfind("model.visual.", 0) == 0 ||
                          name == "lm_head.weight";
      if (!wanted) continue;  // mtp.* is unused by plain autoregressive decoding
      if (info.dtype != "BF16" && info.dtype != "F32")
        throw std::runtime_error("unsupported dtype " + info.dtype + ": " + name);
      // A_log and dt_bias are consumed in fp32 by the reference; promote a
      // bf16 checkpoint tensor once instead of per token.
      auto ends_with = [](const std::string& s, const char* suf) {
        const size_t n = std::strlen(suf);
        return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
      };
      const bool promote =
          info.dtype == "BF16" && (ends_with(name, "A_log") || ends_with(name, "dt_bias"));
      void* d = nullptr;
      if (promote) {
        const size_t n = info.byte_size / sizeof(Bf16);
        std::vector<float> host(n);
        const Bf16* src = static_cast<const Bf16*>(shards->tensor_bytes(name));
        for (size_t i = 0; i < n; ++i) host[i] = float(src[i]);
        UMM_QWEN38_CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
        UMM_QWEN38_CUDA_CHECK(cudaMemcpy(d, host.data(), n * sizeof(float),
                                         cudaMemcpyHostToDevice));
        weights.emplace(name, Tensor{d, "F32"});
        bytes += n * sizeof(float);
      } else {
        UMM_QWEN38_CUDA_CHECK(cudaMalloc(&d, info.byte_size));
        UMM_QWEN38_CUDA_CHECK(cudaMemcpy(d, shards->tensor_bytes(name), info.byte_size,
                                         cudaMemcpyHostToDevice));
        weights.emplace(name, Tensor{d, info.dtype});
        bytes += info.byte_size;
      }
    }
    std::fprintf(stderr, "loaded %zu tensors, %.2f GiB\n", weights.size(),
                 bytes / 1073741824.0);
  }

  void alloc() {
    UMM_QWEN38_CUDA_CHECK(cudaStreamCreate(&stream));
    auto mb = [](Bf16** p, size_t n) {
      UMM_QWEN38_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(p),
                                       n * sizeof(Bf16)));
    };
    auto mf = [](float** p, size_t n) {
      UMM_QWEN38_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(p),
                                       n * sizeof(float)));
    };
    const size_t S = static_cast<size_t>(max_seq);
    mb(&hidden, S * kHidden);
    mb(&norm, S * kHidden);
    mb(&qg, S * kQHeads * kHeadDim * 2);
    mb(&q, S * kQHeads * kHeadDim);
    mb(&gate, S * kQHeads * kHeadDim);
    // The same buffers serve both attention kinds: full attention needs 1024
    // k/v values per token, a linear layer needs 2048 / 6144.  Each path uses
    // its own compact stride and stays inside its bound.
    mb(&k, S * std::max(kKvHeads * kHeadDim, kKeyDim));
    mb(&v, S * std::max(kKvHeads * kHeadDim, kValueDim));
    mb(&o, S * kHidden);
    mb(&mlp_gate, S * kInter);
    mb(&mlp_up, S * kInter);
    mb(&mlp_act, S * kInter);
    mb(&mixed, S * kConvDim);
    mb(&mixed_conv, S * kConvDim);
    mb(&z, S * kValueDim);
    mb(&b, S * kLVH);
    mb(&a, S * kLVH);
    mb(&core, S * kValueDim);
    mb(&gated, S * kValueDim);
    mf(&qn, S * kLKH * kLKD);
    mf(&kn, S * kLKH * kLKD);
    mf(&beta, S * kLVH);
    mf(&decay, S * kLVH);
    mb(&mrope_cos, S * kRotaryDim);
    mb(&mrope_sin, S * kRotaryDim);
    mf(&logits, kVocab);
    UMM_QWEN38_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ids),
                                     S * sizeof(int)));
    UMM_QWEN38_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&argmax_scratch),
                                     sizeof(int)));
    UMM_QWEN38_CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&kv),
                   sizeof(Bf16) * 16 * 2 * S * kKvHeads * kHeadDim));
    UMM_QWEN38_CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&rec_state),
                   sizeof(float) * kLayers * kLVH * kLVD * kLVD));
    UMM_QWEN38_CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&conv_state),
                   sizeof(Bf16) * kLayers * kConvDim * (kConvK - 1)));
    UMM_QWEN38_CUDA_CHECK(
        cudaMemset(rec_state, 0, sizeof(float) * kLayers * kLVH * kLVD * kLVD));
    UMM_QWEN38_CUDA_CHECK(
        cudaMemset(conv_state, 0, sizeof(Bf16) * kLayers * kConvDim * (kConvK - 1)));

    const size_t T = kVisTokenCap;
    mb(&d_pixels, T * kPatchDim);
    mb(&v_hidden, T * kVisH);
    mb(&v_norm, T * kVisH);
    mb(&v_qkv, T * 3 * kVisH);
    mb(&v_q, T * kVisH);
    mb(&v_k, T * kVisH);
    mb(&v_v, T * kVisH);
    mb(&v_proj, T * kVisH);
    mb(&v_mlp, T * kVisInter);
    mb(&v_pos, T * kVisH);
    mb(&pooler, (T / 4) * kVisOut);
    mf(&v_rope_cos, T * kVisHeadDim);
    mf(&v_rope_sin, T * kVisHeadDim);
  }
};

Engine::Engine(const std::string& model_dir, int gpu, int max_seq) {
  impl_ = std::make_unique<Impl>();
  impl_->gpu = gpu;
  impl_->max_seq = max_seq;
  UMM_QWEN38_CUDA_CHECK(cudaSetDevice(gpu));
  impl_->load_weights(model_dir);
  impl_->alloc();

  // The learned position table is small; interpolation runs on the host.
  const Bf16* table = impl_->bf("model.visual.pos_embed.weight");
  impl_->pos_table_host.resize(static_cast<size_t>(kPosSide) * kPosSide * kVisH);
  UMM_QWEN38_CUDA_CHECK(cudaMemcpy(impl_->pos_table_host.data(), table,
                                   impl_->pos_table_host.size() * sizeof(Bf16),
                                   cudaMemcpyDeviceToHost));
}

Engine::~Engine() {
  if (!impl_) return;
  for (const auto& [name, t] : impl_->weights)
    cudaFree(const_cast<void*>(t.ptr));
  Impl& im = *impl_;
  auto fb = [](Bf16*& p) {
    if (p) cudaFree(p);
  };
  auto ff = [](float*& p) {
    if (p) cudaFree(p);
  };
  fb(im.hidden); fb(im.norm); fb(im.qg); fb(im.q); fb(im.gate); fb(im.k);
  fb(im.v); fb(im.o); fb(im.mlp_gate); fb(im.mlp_up); fb(im.mlp_act);
  fb(im.mixed); fb(im.mixed_conv); fb(im.z); fb(im.b); fb(im.a); fb(im.core);
  fb(im.gated); fb(im.mrope_cos); fb(im.mrope_sin); fb(im.kv); fb(im.conv_state);
  ff(im.qn); ff(im.kn); ff(im.beta); ff(im.decay); ff(im.logits); ff(im.rec_state);
  fb(im.d_pixels); fb(im.v_hidden); fb(im.v_norm); fb(im.v_qkv); fb(im.v_q);
  fb(im.v_k); fb(im.v_v); fb(im.v_proj); fb(im.v_mlp); fb(im.v_pos); fb(im.pooler);
  ff(im.v_rope_cos); ff(im.v_rope_sin);
  if (im.d_ids) cudaFree(im.d_ids);
  if (im.argmax_scratch) cudaFree(im.argmax_scratch);
  if (im.stream) cudaStreamDestroy(im.stream);
}

namespace {

void dump_dev(Engine::Impl& im, const std::string& dir, const std::string& name,
              const void* dev, size_t bytes);

void vision_forward(Engine::Impl& im, const PreprocessedImage& img,
                    const std::string& dump_dir) {
  const int tokens = img.rows();
  if (tokens > kVisTokenCap)
    throw std::runtime_error("image exceeds the bounded vision workspace");
  if (img.grid_t != 1) throw std::runtime_error("only still images are supported");
  const std::string p = "model.visual.";
  cudaStream_t st = im.stream;

  std::vector<Bf16> pix(img.pixel_values.size());
  for (size_t i = 0; i < pix.size(); ++i) pix[i] = Bf16(img.pixel_values[i]);
  UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(im.d_pixels, pix.data(),
                                        pix.size() * sizeof(Bf16),
                                        cudaMemcpyHostToDevice, st));

  gemm_bf16(im.v_hidden, im.d_pixels, im.bf(p + "patch_embed.proj.weight"),
            im.bf(p + "patch_embed.proj.bias"), tokens, kPatchDim, kVisH, st);

  const std::vector<Bf16> pos = interpolate_pos_embed(im.pos_table_host, img.grid_h,
                                                     img.grid_w);
  UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(im.v_pos, pos.data(),
                                        pos.size() * sizeof(Bf16),
                                        cudaMemcpyHostToDevice, st));
  add_residual(im.v_hidden, im.v_pos, tokens * kVisH, st);
  if (!dump_dir.empty())
    dump_dev(im, dump_dir, "vpatch.bf16", im.v_hidden,
             static_cast<size_t>(tokens) * kVisH * sizeof(Bf16));

  std::vector<float> vcos, vsin;
  build_vision_rope(img.grid_h, img.grid_w, &vcos, &vsin);
  UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(im.v_rope_cos, vcos.data(),
                                        vcos.size() * sizeof(float),
                                        cudaMemcpyHostToDevice, st));
  UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(im.v_rope_sin, vsin.data(),
                                        vsin.size() * sizeof(float),
                                        cudaMemcpyHostToDevice, st));

  for (int blk = 0; blk < kVisBlocks; ++blk) {
    const std::string bp = p + "blocks." + std::to_string(blk);
    layer_norm(im.v_hidden, im.bf(bp + ".norm1.weight"),
               im.bf(bp + ".norm1.bias"), im.v_norm, tokens, kVisH, kEps, st);
    gemm_bf16(im.v_qkv, im.v_norm, im.bf(bp + ".attn.qkv.weight"),
              im.bf(bp + ".attn.qkv.bias"), tokens, kVisH, 3 * kVisH, st);
    split_qkv(im.v_qkv, im.v_q, im.v_k, im.v_v, tokens, kVisH, st);
    vision_rope(im.v_q, im.v_rope_cos, im.v_rope_sin, tokens, kVisHeads,
                kVisHeadDim, st);
    vision_rope(im.v_k, im.v_rope_cos, im.v_rope_sin, tokens, kVisHeads,
                kVisHeadDim, st);
    vision_attention(im.v_q, im.v_k, im.v_v, im.v_proj, tokens, kVisHeads,
                     kVisHeadDim, st);
    gemm_bf16(im.v_qkv, im.v_proj, im.bf(bp + ".attn.proj.weight"),
              im.bf(bp + ".attn.proj.bias"), tokens, kVisH, kVisH, st);
    add_residual(im.v_hidden, im.v_qkv, tokens * kVisH, st);

    layer_norm(im.v_hidden, im.bf(bp + ".norm2.weight"),
               im.bf(bp + ".norm2.bias"), im.v_norm, tokens, kVisH, kEps, st);
    gemm_bf16(im.v_mlp, im.v_norm, im.bf(bp + ".mlp.linear_fc1.weight"),
              im.bf(bp + ".mlp.linear_fc1.bias"), tokens, kVisH, kVisInter, st);
    gelu_tanh_inplace(im.v_mlp, tokens * kVisInter, st);
    gemm_bf16(im.v_proj, im.v_mlp, im.bf(bp + ".mlp.linear_fc2.weight"),
              im.bf(bp + ".mlp.linear_fc2.bias"), tokens, kVisInter, kVisH, st);
    add_residual(im.v_hidden, im.v_proj, tokens * kVisH, st);
    if (!dump_dir.empty())
      dump_dev(im, dump_dir, "vblock_" + std::to_string(blk) + ".bf16", im.v_hidden,
               static_cast<size_t>(tokens) * kVisH * sizeof(Bf16));
  }

  const int merged = tokens / 4;
  layer_norm(im.v_hidden, im.bf(p + "merger.norm.weight"),
             im.bf(p + "merger.norm.bias"), im.v_norm, tokens, kVisH, kEps, st);
  gemm_bf16(im.v_mlp, im.v_norm, im.bf(p + "merger.linear_fc1.weight"),
            im.bf(p + "merger.linear_fc1.bias"), merged, 4 * kVisH, 4 * kVisH, st);
  gelu_erf_inplace(im.v_mlp, merged * 4 * kVisH, st);
  gemm_bf16(im.pooler, im.v_mlp, im.bf(p + "merger.linear_fc2.weight"),
            im.bf(p + "merger.linear_fc2.bias"), merged, 4 * kVisH, kVisOut, st);
  UMM_QWEN38_CUDA_CHECK(cudaStreamSynchronize(st));

  if (!dump_dir.empty()) {
    std::vector<Bf16> host(static_cast<size_t>(merged) * kVisOut);
    UMM_QWEN38_CUDA_CHECK(cudaMemcpy(host.data(), im.pooler,
                                     host.size() * sizeof(Bf16),
                                     cudaMemcpyDeviceToHost));
    write_file(dump_dir, "pooler.bf16", host.data(), host.size() * sizeof(Bf16));
  }
}

void dump_dev(Engine::Impl& im, const std::string& dir, const std::string& name,
              const void* dev, size_t bytes) {
  std::vector<char> host(bytes);
  UMM_QWEN38_CUDA_CHECK(cudaMemcpy(host.data(), dev, bytes, cudaMemcpyDeviceToHost));
  write_file(dir, name, host.data(), bytes);
}

void linear_layer(Engine::Impl& im, int layer, int m, int history,
                  const std::string& dump_dir) {
  cudaStream_t st = im.stream;
  const std::string lp = "model.language_model.layers." + std::to_string(layer);
  const bool one = (m == 1);
  auto lin = [&](Bf16* y, const Bf16* x, const Bf16* w, int k, int n) {
    if (one)
      gemv_bf16_bf16(w, x, y, n, k, st);
    else
      gemm_bf16(y, x, w, nullptr, m, k, n, st);
  };

  rms_norm_zero_centered(im.hidden, im.bf(lp + ".input_layernorm.weight"), im.norm,
                         m, kHidden, kEps, st);
  lin(im.mixed, im.norm, im.bf(lp + ".linear_attn.in_proj_qkv.weight"), kHidden,
      kConvDim);
  lin(im.z, im.norm, im.bf(lp + ".linear_attn.in_proj_z.weight"), kHidden,
      kValueDim);
  lin(im.b, im.norm, im.bf(lp + ".linear_attn.in_proj_b.weight"), kHidden, kLVH);
  lin(im.a, im.norm, im.bf(lp + ".linear_attn.in_proj_a.weight"), kHidden, kLVH);

  conv1d_causal_silu(im.mixed, im.bf(lp + ".linear_attn.conv1d.weight"),
                     im.conv_state + static_cast<size_t>(layer) * kConvDim * (kConvK - 1),
                     im.mixed_conv, m, kConvDim, st);
  split_qkv_widths(im.mixed_conv, im.q, im.k, im.v, m, kKeyDim, kKeyDim, kValueDim,
                   st);
  l2norm_qk(im.q, im.k, im.qn, im.kn, m, kLKH, kLKD, st);
  beta_and_decay(im.beta, im.decay, im.b, im.a,
                 im.fp32(lp + ".linear_attn.A_log"),
                 im.fp32(lp + ".linear_attn.dt_bias"), m, kLVH, st);
  gated_delta_rule(im.rec_state + static_cast<size_t>(layer) * kLVH * kLVD * kLVD,
                   im.qn, im.kn, im.v, im.beta, im.decay, im.core, m, kLKH, kLVH,
                   kLVD, st);
  gated_rms_norm(im.core, im.z, im.bf(lp + ".linear_attn.norm.weight"), im.gated,
                 m * kLVH, kLVD, kEps, st);
  lin(im.o, im.gated, im.bf(lp + ".linear_attn.out_proj.weight"), kValueDim,
      kHidden);
  if (!dump_dir.empty())
    dump_dev(im, dump_dir, "mixer_" + std::to_string(layer) + ".bf16", im.o,
             static_cast<size_t>(m) * kHidden * sizeof(Bf16));
  add_residual(im.hidden, im.o, m * kHidden, st);
}

void full_layer(Engine::Impl& im, int layer, int m, int history,
                const std::string& dump_dir) {
  cudaStream_t st = im.stream;
  const std::string lp = "model.language_model.layers." + std::to_string(layer);
  const bool one = (m == 1);
  auto lin = [&](Bf16* y, const Bf16* x, const Bf16* w, int k, int n) {
    if (one)
      gemv_bf16_bf16(w, x, y, n, k, st);
    else
      gemm_bf16(y, x, w, nullptr, m, k, n, st);
  };

  rms_norm_zero_centered(im.hidden, im.bf(lp + ".input_layernorm.weight"), im.norm,
                         m, kHidden, kEps, st);
  lin(im.qg, im.norm, im.bf(lp + ".self_attn.q_proj.weight"), kHidden,
      2 * kQHeads * kHeadDim);
  lin(im.k, im.norm, im.bf(lp + ".self_attn.k_proj.weight"), kHidden,
      kKvHeads * kHeadDim);
  lin(im.v, im.norm, im.bf(lp + ".self_attn.v_proj.weight"), kHidden,
      kKvHeads * kHeadDim);
  split_q_gate(im.qg, im.q, im.gate, m, kQHeads, kHeadDim, st);
  rms_norm_zero_centered(im.q, im.bf(lp + ".self_attn.q_norm.weight"), im.q,
                         m * kQHeads, kHeadDim, kEps, st);
  rms_norm_zero_centered(im.k, im.bf(lp + ".self_attn.k_norm.weight"), im.k,
                         m * kKvHeads, kHeadDim, kEps, st);
  mrope_partial(im.q, im.mrope_cos, im.mrope_sin, m, kQHeads, kHeadDim, kRotaryDim,
                st);
  mrope_partial(im.k, im.mrope_cos, im.mrope_sin, m, kKvHeads, kHeadDim, kRotaryDim,
                st);

  const int slot = layer / 4;
  Bf16* kbase = im.kv + ((static_cast<size_t>(slot) * 2) * im.max_seq) * kKvHeads * kHeadDim;
  Bf16* vbase = kbase + static_cast<size_t>(im.max_seq) * kKvHeads * kHeadDim;
  const size_t row = static_cast<size_t>(kKvHeads) * kHeadDim;
  UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(kbase + static_cast<size_t>(history) * row,
                                        im.k, m * row * sizeof(Bf16),
                                        cudaMemcpyDeviceToDevice, st));
  UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(vbase + static_cast<size_t>(history) * row,
                                        im.v, m * row * sizeof(Bf16),
                                        cudaMemcpyDeviceToDevice, st));
  gqa_attention(im.q, kbase, vbase, im.gate, im.o, m, history + m, kQHeads, kKvHeads,
                kHeadDim, history, st);
  lin(im.qg, im.o, im.bf(lp + ".self_attn.o_proj.weight"), kQHeads * kHeadDim,
      kHidden);
  if (!dump_dir.empty())
    dump_dev(im, dump_dir, "mixer_" + std::to_string(layer) + ".bf16", im.qg,
             static_cast<size_t>(m) * kHidden * sizeof(Bf16));
  add_residual(im.hidden, im.qg, m * kHidden, st);
}

void mlp_block(Engine::Impl& im, int layer, int m) {
  cudaStream_t st = im.stream;
  const std::string lp = "model.language_model.layers." + std::to_string(layer);
  const bool one = (m == 1);
  auto lin = [&](Bf16* y, const Bf16* x, const Bf16* w, int k, int n) {
    if (one)
      gemv_bf16_bf16(w, x, y, n, k, st);
    else
      gemm_bf16(y, x, w, nullptr, m, k, n, st);
  };
  rms_norm_zero_centered(im.hidden, im.bf(lp + ".post_attention_layernorm.weight"),
                         im.norm, m, kHidden, kEps, st);
  lin(im.mlp_gate, im.norm, im.bf(lp + ".mlp.gate_proj.weight"), kHidden, kInter);
  lin(im.mlp_up, im.norm, im.bf(lp + ".mlp.up_proj.weight"), kHidden, kInter);
  silu_mul(im.mlp_act, im.mlp_gate, im.mlp_up, m * kInter, st);
  lin(im.o, im.mlp_act, im.bf(lp + ".mlp.down_proj.weight"), kInter, kHidden);
  add_residual(im.hidden, im.o, m * kHidden, st);
}

void run_layer(Engine::Impl& im, int layer, int m, int history,
               const std::string& dump_dir) {
  const bool full = (layer % 4) == 3;
  if (full)
    full_layer(im, layer, m, history, dump_dir);
  else
    linear_layer(im, layer, m, history, dump_dir);
  mlp_block(im, layer, m);
  if (!dump_dir.empty())
    dump_dev(im, dump_dir, "layer_" + std::to_string(layer) + ".bf16", im.hidden,
             static_cast<size_t>(m) * kHidden * sizeof(Bf16));
}

}  // namespace

GenResult Engine::generate(const std::vector<int>& prompt_ids,
                           const PreprocessedImage* image, int max_new_tokens,
                           const std::string& dump_dir) {
  Impl& im = *impl_;
  auto& st = im.stream;
  const int P = static_cast<int>(prompt_ids.size());
  if (P <= 0) throw std::runtime_error("empty prompt");
  if (P + max_new_tokens > im.max_seq)
    throw std::runtime_error("prompt + max_new_tokens exceeds max_seq");
  if (!dump_dir.empty()) {
    write_file(dump_dir, "prompt_ids.i32", prompt_ids.data(),
               prompt_ids.size() * sizeof(int));
  }

  GenResult result;
  result.prompt_tokens = P;

  double vision_ms = 0.0;
  int n_img = 0;
  int img_start = -1;
  if (image) {
    const auto t0 = Clock::now();
    vision_forward(im, *image, dump_dir);
    vision_ms = ms_since(t0);
    n_img = image->rows() / 4;
    for (int i = 0; i < P; ++i)
      if (prompt_ids[i] == kImagePad) {
        if (img_start < 0) img_start = i;
      }
    if (img_start < 0) throw std::runtime_error("image supplied but no image_pad id");
  }
  result.image_tokens = n_img;
  result.vision_ms = vision_ms;

  const auto t_prefill = Clock::now();
  UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(im.d_ids, prompt_ids.data(),
                                        P * sizeof(int), cudaMemcpyHostToDevice,
                                        st));
  index_embeddings(im.bf("model.language_model.embed_tokens.weight"), im.d_ids,
                   im.hidden, P, kHidden, st);
  if (image) {
    UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(
        im.hidden + static_cast<size_t>(img_start) * kHidden, im.pooler,
        static_cast<size_t>(n_img) * kHidden * sizeof(Bf16),
        cudaMemcpyDeviceToDevice, st));
  }

  const Positions pos = build_positions(
      prompt_ids, image ? image->grid_h : 0, image ? image->grid_w : 0, image != nullptr);
  std::vector<Bf16> cos, sin;
  build_text_rope(pos, P, &cos, &sin);
  UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(im.mrope_cos, cos.data(),
                                        cos.size() * sizeof(Bf16),
                                        cudaMemcpyHostToDevice, st));
  UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(im.mrope_sin, sin.data(),
                                        sin.size() * sizeof(Bf16),
                                        cudaMemcpyHostToDevice, st));

  if (!dump_dir.empty()) {
    dump_dev(im, dump_dir, "embed.bf16", im.hidden,
             static_cast<size_t>(P) * kHidden * sizeof(Bf16));
    std::string meta = "prompt_tokens=" + std::to_string(P) +
                       "\nimage_tokens=" + std::to_string(n_img) +
                       "\ngrid_h=" + std::to_string(image ? image->grid_h : 0) +
                       "\ngrid_w=" + std::to_string(image ? image->grid_w : 0) +
                       "\nnext_pos=" + std::to_string(pos.next) + "\n";
    write_file(dump_dir, "meta.txt", meta.data(), meta.size());
  }

  for (int l = 0; l < kLayers; ++l) run_layer(im, l, P, 0, dump_dir);

  rms_norm_zero_centered(im.hidden + static_cast<size_t>(P - 1) * kHidden,
                         im.bf("model.language_model.norm.weight"), im.norm, 1,
                         kHidden, kEps, st);
  gemv_bf16_f32(im.bf("lm_head.weight"), im.norm, im.logits, kVocab, kHidden, st);
  int next = argmax_f32(im.logits, kVocab, im.argmax_scratch, st);
  UMM_QWEN38_CUDA_CHECK(cudaStreamSynchronize(st));
  result.prefill_ms = ms_since(t_prefill);
  if (!dump_dir.empty())
    dump_dev(im, dump_dir, "logits0.f32", im.logits,
             static_cast<size_t>(kVocab) * sizeof(float));

  int seq_len = P;
  int next_pos = pos.next;
  size_t decode_forwards = 0;
  const auto t_decode = Clock::now();
  while (static_cast<int>(result.ids.size()) < max_new_tokens && !is_eos(next)) {
    result.ids.push_back(next);
    if (static_cast<int>(result.ids.size()) >= max_new_tokens) break;
    UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(im.d_ids, &next, sizeof(int),
                                          cudaMemcpyHostToDevice, st));
    index_embeddings(im.bf("model.language_model.embed_tokens.weight"), im.d_ids,
                     im.hidden, 1, kHidden, st);
    const Positions one_pos{{next_pos}, {next_pos}, {next_pos}, next_pos + 1};
    build_text_rope(one_pos, 1, &cos, &sin);
    UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(im.mrope_cos, cos.data(),
                                          cos.size() * sizeof(Bf16),
                                          cudaMemcpyHostToDevice, st));
    UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(im.mrope_sin, sin.data(),
                                          sin.size() * sizeof(Bf16),
                                          cudaMemcpyHostToDevice, st));
    for (int l = 0; l < kLayers; ++l) run_layer(im, l, 1, seq_len, "");
    rms_norm_zero_centered(im.hidden, im.bf("model.language_model.norm.weight"),
                           im.norm, 1, kHidden, kEps, st);
    gemv_bf16_f32(im.bf("lm_head.weight"), im.norm, im.logits, kVocab, kHidden, st);
    next = argmax_f32(im.logits, kVocab, im.argmax_scratch, st);
    ++seq_len;
    ++next_pos;
    ++decode_forwards;
  }
  UMM_QWEN38_CUDA_CHECK(cudaStreamSynchronize(st));
  result.decode_ms = ms_since(t_decode);
  result.decode_steps = static_cast<int>(decode_forwards);

  if (!dump_dir.empty()) {
    write_file(dump_dir, "gen_ids.i32", result.ids.data(),
               result.ids.size() * sizeof(int));
  }
  return result;
}

}  // namespace umminfer::qwen38
