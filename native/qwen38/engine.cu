// SPDX-License-Identifier: Apache-2.0
// Qwen3.8-27B hybrid decoder: 64 dense layers, 3:1 gated-delta-net to full
// attention, partial mRoPE, dense SwiGLU, 27-block ViT with patch merger.
#include "engine.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "prefix_cache.hpp"
#include "qwen38_ops.hpp"
#include "safetensors_mmap.hpp"
#include "sampling.hpp"

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

// One queued batch request plus its decode progress. The batch thread owns
// every field after `req`; the calling thread reads `result` and `error` only
// after `done` is set under `mu`.
struct BatchJob {
  BatchRequest req;
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  GenResult result;
  std::exception_ptr error;
  int slot = -1;
  // Tokens this job takes from the prefix snapshot instead of prefilling.
  // Resolved once, before the job joins a group, so grouping can key on the
  // rows it will actually compute.
  int prefix = 0;
  int seq_len = 0;   // tokens already in the KV cache
  int next_pos = 0;  // mRoPE position of next_token
  int next_token = 0;
  // Seeded on the calling thread before the job is queued, so a request's
  // sampling sequence depends only on its own seed. The batch thread is the
  // only reader.
  std::mt19937_64 rng{0};
  std::chrono::steady_clock::time_point decode_started;
};

struct Engine::Impl {
  int gpu = 0;
  int max_seq = 8192;
  int batch_capacity = 1;
  std::unique_ptr<ShardedSafetensors> shards;
  std::unordered_map<std::string, Tensor> weights;
  std::vector<Bf16> pos_table_host;  // [2304, 1152] learned vision position grid
  cudaStream_t stream = nullptr;

  // Continuous batching. Only the batch thread touches these after start.
  std::thread batch_thread;
  std::mutex queue_mu;
  std::condition_variable queue_cv;
  std::deque<std::shared_ptr<BatchJob>> queue;
  bool stopping = false;
  std::vector<std::shared_ptr<BatchJob>> active;
  std::vector<int> free_slots;
  Bf16* last_hidden = nullptr;  // [kMaxBatch][kHidden]
  int* argmax_idx = nullptr;    // [kMaxBatch]
  int* d_logit_rows = nullptr;  // [kMaxBatch] gather rows for the LM head

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

  // Prefix snapshot: the state a successful text-only prefill leaves at its
  // prompt boundary. Allocated on the first snapshot so a disabled cache and
  // image-only workloads never pay for it.
  bool prefix_cache = true;
  std::vector<int> cached_ids;
  int cached_len = 0;
  Bf16* kv_snap = nullptr;    // same layout as `kv` for one sequence
  float* rec_snap = nullptr;  // same size as `rec_state`
  Bf16* conv_snap = nullptr;  // same size as `conv_state`

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
    // The text caches are indexed [sequence][layer][...]. Sequence 0 keeps the
    // exact layout the single-request executor used, so its kernels are
    // unchanged. batch_capacity == 1 allocates exactly one sequence.
    const size_t B = static_cast<size_t>(batch_capacity);
    const size_t rec_bytes = sizeof(float) * kLayers * kLVH * kLVD * kLVD;
    const size_t conv_bytes = sizeof(Bf16) * kLayers * kConvDim * (kConvK - 1);
    const size_t kv_bytes = sizeof(Bf16) * 16 * 2 * S * kKvHeads * kHeadDim;
    mf(&logits, B * kVocab);
    UMM_QWEN38_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ids),
                                     S * sizeof(int)));
    UMM_QWEN38_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&argmax_scratch),
                                     sizeof(int)));
    UMM_QWEN38_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&argmax_idx),
                                     B * sizeof(int)));
    UMM_QWEN38_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_logit_rows),
                                     kMaxBatch * sizeof(int)));
    mb(&last_hidden, static_cast<size_t>(kMaxBatch) * kHidden);
    UMM_QWEN38_CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&kv), B * kv_bytes));
    UMM_QWEN38_CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&rec_state), B * rec_bytes));
    UMM_QWEN38_CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&conv_state), B * conv_bytes));
    UMM_QWEN38_CUDA_CHECK(cudaMemset(rec_state, 0, B * rec_bytes));
    UMM_QWEN38_CUDA_CHECK(cudaMemset(conv_state, 0, B * conv_bytes));

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

Engine::Engine(const std::string& model_dir, int gpu, int max_seq,
               int batch_capacity) {
  impl_ = std::make_unique<Impl>();
  impl_->gpu = gpu;
  impl_->max_seq = max_seq;
  impl_->batch_capacity = std::max(batch_capacity, 1);
  if (impl_->batch_capacity > kMaxBatch)
    throw std::runtime_error("batch_capacity exceeds kMaxBatch");
  if (const char* env = std::getenv("UMM_QWEN38_PREFIX_CACHE")) {
    impl_->prefix_cache = std::atoi(env) != 0;
  }
  UMM_QWEN38_CUDA_CHECK(cudaSetDevice(gpu));
  impl_->load_weights(model_dir);
  impl_->alloc();
  impl_->free_slots.resize(static_cast<size_t>(impl_->batch_capacity));
  for (int i = 0; i < impl_->batch_capacity; ++i) impl_->free_slots[i] = i;

  // The learned position table is small; interpolation runs on the host.
  const Bf16* table = impl_->bf("model.visual.pos_embed.weight");
  impl_->pos_table_host.resize(static_cast<size_t>(kPosSide) * kPosSide * kVisH);
  UMM_QWEN38_CUDA_CHECK(cudaMemcpy(impl_->pos_table_host.data(), table,
                                   impl_->pos_table_host.size() * sizeof(Bf16),
                                   cudaMemcpyDeviceToHost));
}

Engine::~Engine() {
  if (!impl_) return;
  if (impl_->batch_thread.joinable()) {
    {
      std::lock_guard<std::mutex> lock(impl_->queue_mu);
      impl_->stopping = true;
    }
    impl_->queue_cv.notify_all();
    impl_->batch_thread.join();
  }
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
  fb(im.kv_snap); fb(im.conv_snap); ff(im.rec_snap);
  fb(im.last_hidden);
  ff(im.qn); ff(im.kn); ff(im.beta); ff(im.decay); ff(im.logits); ff(im.rec_state);
  fb(im.d_pixels); fb(im.v_hidden); fb(im.v_norm); fb(im.v_qkv); fb(im.v_q);
  fb(im.v_k); fb(im.v_v); fb(im.v_proj); fb(im.v_mlp); fb(im.v_pos); fb(im.pooler);
  ff(im.v_rope_cos); ff(im.v_rope_sin);
  if (im.d_ids) cudaFree(im.d_ids);
  if (im.argmax_scratch) cudaFree(im.argmax_scratch);
  if (im.argmax_idx) cudaFree(im.argmax_idx);
  if (im.d_logit_rows) cudaFree(im.d_logit_rows);
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

// ---- continuous batching --------------------------------------------------
constexpr size_t kKvRow = static_cast<size_t>(kKvHeads) * kHeadDim;

// ---- prefix snapshot ------------------------------------------------------
// The snapshot copies exactly what carries across a prefill boundary: the KV
// rows of every full-attention layer and the recurrent + conv state the linear
// layers leave behind. Decode only ever writes beyond the prompt boundary, so
// the copy stays valid for the rest of the request.
constexpr int kKvSlots = kLayers / 4;

size_t rec_state_bytes() {
  return sizeof(float) * kLayers * kLVH * kLVD * kLVD;
}
size_t conv_state_bytes() {
  return sizeof(Bf16) * kLayers * kConvDim * (kConvK - 1);
}

// Element strides between two batch slots. The batched path stacks one full
// serial-sized cache per slot, so slot s owns
// [s * stride, (s + 1) * stride) of each cache.
size_t kv_slot_stride(const Engine::Impl& im) {
  return static_cast<size_t>(kKvSlots) * 2 * im.max_seq * kKvRow;
}
size_t rec_slot_stride() { return static_cast<size_t>(kLayers) * kLVH * kLVD * kLVD; }
size_t conv_slot_stride() {
  return static_cast<size_t>(kLayers) * kConvDim * (kConvK - 1);
}

void ensure_snapshot_buffers(Engine::Impl& im) {
  if (im.kv_snap != nullptr) return;
  const size_t kv_bytes = static_cast<size_t>(kKvSlots) * 2 * im.max_seq * kKvRow;
  UMM_QWEN38_CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&im.kv_snap), kv_bytes * sizeof(Bf16)));
  UMM_QWEN38_CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&im.rec_snap), rec_state_bytes()));
  UMM_QWEN38_CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&im.conv_snap), conv_state_bytes()));
}

void copy_kv_rows(Bf16* dst, const Bf16* src, int len, size_t stride,
                  cudaStream_t st) {
  const size_t bytes = static_cast<size_t>(len) * kKvRow * sizeof(Bf16);
  for (int s = 0; s < kKvSlots; ++s) {
    UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(
        dst + static_cast<size_t>(s) * 2 * stride, src + static_cast<size_t>(s) * 2 * stride,
        bytes, cudaMemcpyDeviceToDevice, st));
    UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(
        dst + static_cast<size_t>(s) * 2 * stride + stride,
        src + static_cast<size_t>(s) * 2 * stride + stride, bytes,
        cudaMemcpyDeviceToDevice, st));
  }
}

// The snapshot holds one sequence's worth of KV rows, recurrent state and conv
// history, so it is copied from and into exactly one batch slot. The serial
// executor owns slot 0.
void snapshot_prefix(Engine::Impl& im, int slot, const std::vector<int>& ids,
                     cudaStream_t st) {
  ensure_snapshot_buffers(im);
  const size_t stride = static_cast<size_t>(im.max_seq) * kKvRow;
  copy_kv_rows(im.kv_snap, im.kv + static_cast<size_t>(slot) * kv_slot_stride(im),
               static_cast<int>(ids.size()), stride, st);
  UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(
      im.rec_snap, im.rec_state + static_cast<size_t>(slot) * rec_slot_stride(),
      rec_state_bytes(), cudaMemcpyDeviceToDevice, st));
  UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(
      im.conv_snap, im.conv_state + static_cast<size_t>(slot) * conv_slot_stride(),
      conv_state_bytes(), cudaMemcpyDeviceToDevice, st));
  im.cached_ids = ids;
  im.cached_len = static_cast<int>(ids.size());
}

void restore_prefix(Engine::Impl& im, int slot, int len, cudaStream_t st) {
  const size_t stride = static_cast<size_t>(im.max_seq) * kKvRow;
  copy_kv_rows(im.kv + static_cast<size_t>(slot) * kv_slot_stride(im), im.kv_snap,
               len, stride, st);
  UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(
      im.rec_state + static_cast<size_t>(slot) * rec_slot_stride(), im.rec_snap,
      rec_state_bytes(), cudaMemcpyDeviceToDevice, st));
  UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(
      im.conv_state + static_cast<size_t>(slot) * conv_slot_stride(), im.conv_snap,
      conv_state_bytes(), cudaMemcpyDeviceToDevice, st));
}

void finish_job(const std::shared_ptr<BatchJob>& job) {
  {
    std::lock_guard<std::mutex> lock(job->mu);
    job->done = true;
  }
  job->cv.notify_all();
}

void release_slot(Engine::Impl& im, const std::shared_ptr<BatchJob>& job) {
  if (job->slot < 0) return;
  im.free_slots.push_back(job->slot);
  job->slot = -1;
}

void retire_job(Engine::Impl& im, const std::shared_ptr<BatchJob>& job) {
  job->result.decode_ms = ms_since(job->decode_started);
  release_slot(im, job);
  finish_job(job);
}

// Emits the current token. Returns true when the request reached EOS or its
// token limit, which matches the single-request loop's terminal condition.
bool emit_token(const std::shared_ptr<BatchJob>& job) {
  if (is_eos(job->next_token)) {
    job->result.stopped_by_eos = true;
    return true;
  }
  job->result.ids.push_back(job->next_token);
  if (job->req.on_token) job->req.on_token(job->next_token);
  return static_cast<int>(job->result.ids.size()) >= job->req.max_new_tokens;
}

void seg_linear(Bf16* y, const Bf16* x, const Bf16* w, int rows, int k, int n,
                bool use_gemv, cudaStream_t st) {
  if (use_gemv)
    gemv_bf16_bf16_batch(w, x, y, rows, n, k, st);
  else
    gemm_bf16(y, x, w, nullptr, rows, k, n, st);
}

void linear_layer_seg(Engine::Impl& im, int layer, int rows, const SeqSpans& sp,
                      int nseq, bool use_gemv) {
  cudaStream_t st = im.stream;
  const std::string lp = "model.language_model.layers." + std::to_string(layer);
  rms_norm_zero_centered(im.hidden, im.bf(lp + ".input_layernorm.weight"), im.norm,
                         rows, kHidden, kEps, st);
  seg_linear(im.mixed, im.norm, im.bf(lp + ".linear_attn.in_proj_qkv.weight"),
             rows, kHidden, kConvDim, use_gemv, st);
  seg_linear(im.z, im.norm, im.bf(lp + ".linear_attn.in_proj_z.weight"), rows,
             kHidden, kValueDim, use_gemv, st);
  seg_linear(im.b, im.norm, im.bf(lp + ".linear_attn.in_proj_b.weight"), rows,
             kHidden, kLVH, use_gemv, st);
  seg_linear(im.a, im.norm, im.bf(lp + ".linear_attn.in_proj_a.weight"), rows,
             kHidden, kLVH, use_gemv, st);
  conv1d_causal_silu_seg(
      im.mixed, im.bf(lp + ".linear_attn.conv1d.weight"),
      im.conv_state + static_cast<size_t>(layer) * kConvDim * (kConvK - 1),
      im.mixed_conv, sp, nseq, kConvDim,
      static_cast<size_t>(kLayers) * kConvDim * (kConvK - 1), st);
  split_qkv_widths(im.mixed_conv, im.q, im.k, im.v, rows, kKeyDim, kKeyDim,
                   kValueDim, st);
  l2norm_qk(im.q, im.k, im.qn, im.kn, rows, kLKH, kLKD, st);
  beta_and_decay(im.beta, im.decay, im.b, im.a, im.fp32(lp + ".linear_attn.A_log"),
                 im.fp32(lp + ".linear_attn.dt_bias"), rows, kLVH, st);
  gated_delta_rule_seg(
      im.rec_state + static_cast<size_t>(layer) * kLVH * kLVD * kLVD, im.qn,
      im.kn, im.v, im.beta, im.decay, im.core, sp, nseq, kLKH, kLVH, kLVD,
      static_cast<size_t>(kLayers) * kLVH * kLVD * kLVD, st);
  gated_rms_norm(im.core, im.z, im.bf(lp + ".linear_attn.norm.weight"), im.gated,
                 rows * kLVH, kLVD, kEps, st);
  seg_linear(im.o, im.gated, im.bf(lp + ".linear_attn.out_proj.weight"), rows,
             kValueDim, kHidden, use_gemv, st);
  add_residual(im.hidden, im.o, rows * kHidden, st);
}

void full_layer_seg(Engine::Impl& im, int layer, int rows, const SeqSpans& sp,
                    int nseq, bool use_gemv) {
  cudaStream_t st = im.stream;
  const std::string lp = "model.language_model.layers." + std::to_string(layer);
  rms_norm_zero_centered(im.hidden, im.bf(lp + ".input_layernorm.weight"), im.norm,
                         rows, kHidden, kEps, st);
  seg_linear(im.qg, im.norm, im.bf(lp + ".self_attn.q_proj.weight"), rows, kHidden,
             2 * kQHeads * kHeadDim, use_gemv, st);
  seg_linear(im.k, im.norm, im.bf(lp + ".self_attn.k_proj.weight"), rows, kHidden,
             kKvHeads * kHeadDim, use_gemv, st);
  seg_linear(im.v, im.norm, im.bf(lp + ".self_attn.v_proj.weight"), rows, kHidden,
             kKvHeads * kHeadDim, use_gemv, st);
  split_q_gate(im.qg, im.q, im.gate, rows, kQHeads, kHeadDim, st);
  rms_norm_zero_centered(im.q, im.bf(lp + ".self_attn.q_norm.weight"), im.q,
                         rows * kQHeads, kHeadDim, kEps, st);
  rms_norm_zero_centered(im.k, im.bf(lp + ".self_attn.k_norm.weight"), im.k,
                         rows * kKvHeads, kHeadDim, kEps, st);
  mrope_partial(im.q, im.mrope_cos, im.mrope_sin, rows, kQHeads, kHeadDim,
                kRotaryDim, st);
  mrope_partial(im.k, im.mrope_cos, im.mrope_sin, rows, kKvHeads, kHeadDim,
                kRotaryDim, st);

  const int slot = layer / 4;
  const size_t strand = static_cast<size_t>(16) * 2 * im.max_seq * kKvRow;
  Bf16* kbase = im.kv + static_cast<size_t>(slot) * 2 * im.max_seq * kKvRow;
  Bf16* vbase = kbase + static_cast<size_t>(im.max_seq) * kKvRow;
  for (int s = 0; s < nseq; ++s) {
    const size_t dst = static_cast<size_t>(sp.slot[s]) * strand +
                       static_cast<size_t>(sp.hist[s]) * kKvRow;
    const size_t src = static_cast<size_t>(sp.off[s]) * kKvRow;
    const size_t bytes = static_cast<size_t>(sp.len[s]) * kKvRow * sizeof(Bf16);
    UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(kbase + dst, im.k + src, bytes,
                                          cudaMemcpyDeviceToDevice, st));
    UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(vbase + dst, im.v + src, bytes,
                                          cudaMemcpyDeviceToDevice, st));
  }
  gqa_attention_seg(im.q, kbase, vbase, im.gate, im.o, sp, nseq, strand, kQHeads,
                    kKvHeads, kHeadDim, st);
  seg_linear(im.qg, im.o, im.bf(lp + ".self_attn.o_proj.weight"), rows,
             kQHeads * kHeadDim, kHidden, use_gemv, st);
  add_residual(im.hidden, im.qg, rows * kHidden, st);
}

void mlp_block_seg(Engine::Impl& im, int layer, int rows, bool use_gemv) {
  cudaStream_t st = im.stream;
  const std::string lp = "model.language_model.layers." + std::to_string(layer);
  rms_norm_zero_centered(im.hidden,
                         im.bf(lp + ".post_attention_layernorm.weight"), im.norm,
                         rows, kHidden, kEps, st);
  seg_linear(im.mlp_gate, im.norm, im.bf(lp + ".mlp.gate_proj.weight"), rows,
             kHidden, kInter, use_gemv, st);
  seg_linear(im.mlp_up, im.norm, im.bf(lp + ".mlp.up_proj.weight"), rows, kHidden,
             kInter, use_gemv, st);
  silu_mul(im.mlp_act, im.mlp_gate, im.mlp_up, rows * kInter, st);
  seg_linear(im.o, im.mlp_act, im.bf(lp + ".mlp.down_proj.weight"), rows, kInter,
             kHidden, use_gemv, st);
  add_residual(im.hidden, im.o, rows * kHidden, st);
}

void run_layer_seg(Engine::Impl& im, int layer, int rows, const SeqSpans& sp,
                   int nseq, bool use_gemv) {
  if ((layer % 4) == 3)
    full_layer_seg(im, layer, rows, sp, nseq, use_gemv);
  else
    linear_layer_seg(im, layer, rows, sp, nseq, use_gemv);
  mlp_block_seg(im, layer, rows, use_gemv);
}

// Runs every layer over `rows` workspace rows and returns one sampled token per
// span, taken from the last row of that span.
std::vector<int> batch_forward(
    Engine::Impl& im, const SeqSpans& sp, int nseq, int rows, bool use_gemv,
    const int* logit_rows, const std::vector<std::shared_ptr<BatchJob>>& jobs) {
  cudaStream_t st = im.stream;
  for (int l = 0; l < kLayers; ++l) run_layer_seg(im, l, rows, sp, nseq, use_gemv);
  // A launch configuration error is reported by the launch itself, not by the
  // following stream call, so check it before the first dependent copy.
  UMM_QWEN38_CUDA_CHECK(cudaGetLastError());
  UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(im.d_logit_rows, logit_rows,
                                        static_cast<size_t>(nseq) * sizeof(int),
                                        cudaMemcpyHostToDevice, st));
  gather_rows(im.last_hidden, im.hidden, im.d_logit_rows, nseq, kHidden, st);
  rms_norm_zero_centered(im.last_hidden, im.bf("model.language_model.norm.weight"),
                         im.last_hidden, nseq, kHidden, kEps, st);
  gemv_bf16_f32_batch(im.bf("lm_head.weight"), im.last_hidden, im.logits, nseq,
                      kVocab, kHidden, st);
  bool sample_any = false;
  for (int i = 0; i < nseq; ++i)
    if (jobs[i]->req.sampling.temperature > 0.0f) sample_any = true;
  std::vector<int> out(static_cast<size_t>(nseq));
  if (!sample_any) {
    argmax_rows(im.logits, nseq, kVocab, im.argmax_idx, st);
    UMM_QWEN38_CUDA_CHECK(cudaGetLastError());
    UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(out.data(), im.argmax_idx,
                                          static_cast<size_t>(nseq) * sizeof(int),
                                          cudaMemcpyDeviceToHost, st));
    UMM_QWEN38_CUDA_CHECK(cudaStreamSynchronize(st));
    return out;
  }
  // A sampled row needs its whole logit vector on the host. Greedy rows in the
  // same group take the same host path, so a row's result never depends on
  // which other requests share its forward.
  std::vector<float> host(static_cast<size_t>(nseq) * kVocab);
  UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(host.data(), im.logits,
                                        host.size() * sizeof(float),
                                        cudaMemcpyDeviceToHost, st));
  UMM_QWEN38_CUDA_CHECK(cudaStreamSynchronize(st));
  for (int i = 0; i < nseq; ++i)
    out[static_cast<size_t>(i)] =
        sample_token(host.data() + static_cast<size_t>(i) * kVocab, kVocab,
                     jobs[i]->req.sampling, jobs[i]->rng);
  return out;
}

void prefill_group(Engine::Impl& im,
                   const std::vector<std::shared_ptr<BatchJob>>& jobs) {
  cudaStream_t st = im.stream;
  const int nseq = static_cast<int>(jobs.size());
  std::vector<int> prompt_len(static_cast<size_t>(nseq));
  std::vector<int> prefix(static_cast<size_t>(nseq), 0);
  // Token index where this job publishes its snapshot. Equal to prompt_len
  // when the job takes the historical whole-prompt boundary.
  std::vector<int> snap(static_cast<size_t>(nseq), 0);
  std::vector<Positions> pos(static_cast<size_t>(nseq));
  std::vector<int> img_start(static_cast<size_t>(nseq), -1);
  std::vector<int> n_img(static_cast<size_t>(nseq), 0);
  std::vector<double> vision_ms(static_cast<size_t>(nseq), 0.0);
  bool want_split = false;
  for (int i = 0; i < nseq; ++i) {
    const BatchJob& job = *jobs[i];
    const int P = static_cast<int>(job.req.prompt_ids.size());
    prompt_len[i] = P;
    // The batch loop resolved `prefix` before it grouped the jobs, so the
    // snapshot this group publishes cannot change what a later job in the same
    // group reuses.
    prefix[i] = job.prefix;
    const bool text = job.req.image == nullptr && im.prefix_cache;
    snap[i] = text ? resolve_snapshot_len(P, job.req.cache_prefix_len) : P;
    // A split keeps the code path of both phases identical to the one-pass
    // path only when each phase still takes the multi-row GEMM kernel. A
    // one-row phase would switch to GEMV, whose reduction order differs.
    if (snap[i] < P && !(snap[i] - prefix[i] > 1 && P - snap[i] > 1)) snap[i] = P;
    if (snap[i] < P) want_split = true;
    const PreprocessedImage* image = job.req.image;
    if (image != nullptr) {
      if (image->grid_t != 1)
        throw std::runtime_error("only still images are supported");
      const auto t0 = Clock::now();
      vision_forward(im, *image, "");
      vision_ms[i] = ms_since(t0);
      n_img[i] = image->rows() / 4;
      for (int k = 0; k < P; ++k) {
        if (job.req.prompt_ids[k] == kImagePad) {
          img_start[i] = k;
          break;
        }
      }
      if (img_start[i] < 0)
        throw std::runtime_error("image supplied but no image_pad id");
    }
    pos[i] = build_positions(job.req.prompt_ids, image ? image->grid_h : 0,
                             image ? image->grid_w : 0, image != nullptr);
  }

  // Every job in one phase must take the same linear-op path. The grouping
  // already enforces that for the whole prompt; a split can reintroduce a
  // mix, so it falls back to the single pass when that happens.
  if (want_split) {
    auto consistent = [&](bool first) {
      int single = 0, multi = 0;
      for (int i = 0; i < nseq; ++i) {
        const int a = first ? prefix[i] : snap[i];
        const int b = first ? snap[i] : prompt_len[i];
        if (b <= a) continue;
        if (b - a == 1) ++single;
        else ++multi;
      }
      return single == 0 || multi == 0;
    };
    if (!consistent(true) || !consistent(false)) {
      for (int i = 0; i < nseq; ++i) snap[i] = prompt_len[i];
      want_split = false;
    }
  }

  // One forward over the chosen span of every job. `first` selects the phase:
  // the lead [prefix, snap) or the tail [snap, prompt_len). Jobs with an empty
  // span sit out. Returns the sampled token of each participating job and the
  // job index it belongs to.
  auto run_phase = [&](bool first, std::vector<int>* job_index,
                       std::vector<int>* tokens) {
    SeqSpans sp{};
    std::vector<int> all_ids;
    std::vector<std::shared_ptr<BatchJob>> phase_jobs;
    std::vector<int> logit_rows;
    job_index->clear();
    int rows = 0;
    for (int i = 0; i < nseq; ++i) {
      const int a = first ? prefix[i] : snap[i];
      const int b = first ? snap[i] : prompt_len[i];
      if (b <= a) continue;
      const int s = static_cast<int>(job_index->size());
      sp.off[s] = rows;
      sp.len[s] = b - a;
      sp.hist[s] = a;
      sp.slot[s] = jobs[i]->slot;
      rows += b - a;
      all_ids.insert(all_ids.end(), jobs[i]->req.prompt_ids.begin() + a,
                     jobs[i]->req.prompt_ids.begin() + b);
      job_index->push_back(i);
      phase_jobs.push_back(jobs[i]);
      logit_rows.push_back(rows - 1);
    }
    const int n = static_cast<int>(job_index->size());
    tokens->clear();
    if (n == 0) return;
    if (rows > im.max_seq) throw std::runtime_error("batch rows exceed max_seq");

    // The recurrent state is cumulative, so a sequence that starts from an
    // empty cache begins at zero; a sequence that reuses a prefix starts from
    // the state the snapshot holds at its boundary. Only the first phase
    // initialises it, because the second continues in the same slot.
    if (first) {
      const size_t rec_bytes = sizeof(float) * kLayers * kLVH * kLVD * kLVD;
      const size_t conv_bytes = sizeof(Bf16) * kLayers * kConvDim * (kConvK - 1);
      for (int s = 0; s < n; ++s) {
        const int i = (*job_index)[s];
        if (prefix[i] > 0) {
          restore_prefix(im, sp.slot[s], prefix[i], st);
          continue;
        }
        UMM_QWEN38_CUDA_CHECK(cudaMemsetAsync(
            im.rec_state + static_cast<size_t>(sp.slot[s]) * rec_slot_stride(), 0,
            rec_bytes, st));
        UMM_QWEN38_CUDA_CHECK(cudaMemsetAsync(
            im.conv_state + static_cast<size_t>(sp.slot[s]) * conv_slot_stride(),
            0, conv_bytes, st));
      }
    }
    UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(im.d_ids, all_ids.data(),
                                          static_cast<size_t>(rows) * sizeof(int),
                                          cudaMemcpyHostToDevice, st));
    index_embeddings(im.bf("model.language_model.embed_tokens.weight"), im.d_ids,
                     im.hidden, rows, kHidden, st);
    for (int s = 0; s < n; ++s) {
      const int i = (*job_index)[s];
      if (n_img[i] <= 0) continue;
      UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(
          im.hidden +
              static_cast<size_t>(sp.off[s] + img_start[i] - sp.hist[s]) * kHidden,
          im.pooler, static_cast<size_t>(n_img[i]) * kHidden * sizeof(Bf16),
          cudaMemcpyDeviceToDevice, st));
    }
    std::vector<Bf16> cos, sin;
    cos.reserve(static_cast<size_t>(rows) * kRotaryDim);
    sin.reserve(static_cast<size_t>(rows) * kRotaryDim);
    for (int s = 0; s < n; ++s) {
      // Text positions are absolute, so a reused or split prefix shifts the
      // rope window rather than the positions inside it.
      const int i = (*job_index)[s];
      std::vector<Bf16> c, s2;
      build_text_rope(pos[i], prompt_len[i], &c, &s2);
      const size_t from = static_cast<size_t>(sp.hist[s]) * kRotaryDim;
      cos.insert(cos.end(), c.begin() + from, c.end());
      sin.insert(sin.end(), s2.begin() + from, s2.end());
    }
    UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(im.mrope_cos, cos.data(),
                                          cos.size() * sizeof(Bf16),
                                          cudaMemcpyHostToDevice, st));
    UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(im.mrope_sin, sin.data(),
                                          sin.size() * sizeof(Bf16),
                                          cudaMemcpyHostToDevice, st));

    // A one-token prompt takes the GEMV path, exactly as the single-request
    // executor does for m == 1. Each phase is homogeneous, so no call mixes
    // paths.
    const bool use_gemv = (rows == n);
    *tokens = batch_forward(im, sp, n, rows, use_gemv, logit_rows.data(),
                            phase_jobs);
  };

  const auto t0 = Clock::now();
  std::vector<int> next(static_cast<size_t>(nseq), 0);
  std::vector<int> job_index, tokens;
  // The lead phase samples a token for a split job too. That token is dropped,
  // so keep its rng where it was: a sampled request must consume the same draw
  // sequence as the one-pass prefill would.
  std::vector<int> saved_rng_job;
  std::vector<std::mt19937_64> saved_rng;
  if (want_split)
    for (int i = 0; i < nseq; ++i)
      if (snap[i] < prompt_len[i]) {
        saved_rng_job.push_back(i);
        saved_rng.push_back(jobs[i]->rng);
      }
  run_phase(/*first=*/true, &job_index, &tokens);
  for (size_t s = 0; s < job_index.size(); ++s) next[job_index[s]] = tokens[s];
  for (size_t k = 0; k < saved_rng_job.size(); ++k)
    jobs[saved_rng_job[k]]->rng = saved_rng[k];
  if (want_split) {
    // Publish the snapshot between the phases. The KV rows [0, snap) and the
    // recurrent state at snap are exactly what a prefill of only those tokens
    // would leave, so the next turn reuses them and prefills just the tail.
    if (im.prefix_cache)
      for (int i = nseq - 1; i >= 0; --i) {
        if (jobs[i]->req.image != nullptr || snap[i] >= prompt_len[i]) continue;
        snapshot_prefix(im, jobs[i]->slot,
                        std::vector<int>(jobs[i]->req.prompt_ids.begin(),
                                         jobs[i]->req.prompt_ids.begin() +
                                             snap[i]),
                        st);
        break;
      }
    run_phase(/*first=*/false, &job_index, &tokens);
    for (size_t s = 0; s < job_index.size(); ++s) next[job_index[s]] = tokens[s];
  } else if (im.prefix_cache) {
    // Publish the snapshot at the prompt boundary, before the first decode row
    // is written into these slots. The snapshot holds one sequence, so the last
    // pure-text job of the group refreshes it; an image job leaves it alone
    // because the vision tower writes its own hidden rows.
    for (int i = nseq - 1; i >= 0; --i) {
      if (jobs[i]->req.image != nullptr) continue;
      snapshot_prefix(im, jobs[i]->slot, jobs[i]->req.prompt_ids, st);
      break;
    }
  }
  const double prefill_ms = ms_since(t0);
  for (int i = 0; i < nseq; ++i) {
    auto& job = jobs[i];
    job->result.prompt_tokens = prompt_len[i];
    job->result.prefill_tokens = prompt_len[i] - prefix[i];
    job->result.reused_tokens = prefix[i];
    job->result.image_tokens = n_img[i];
    job->result.vision_ms = vision_ms[i];
    job->result.prefill_ms = prefill_ms;
    job->seq_len = prompt_len[i];
    job->next_pos = pos[i].next;
    job->next_token = next[i];
  }
}

void step_decode(Engine::Impl& im,
                 const std::vector<std::shared_ptr<BatchJob>>& jobs) {
  cudaStream_t st = im.stream;
  const int m = static_cast<int>(jobs.size());
  SeqSpans sp{};
  std::vector<int> ids(static_cast<size_t>(m));
  std::vector<int> logit_rows(static_cast<size_t>(m));
  std::vector<Bf16> cos, sin;
  cos.reserve(static_cast<size_t>(m) * kRotaryDim);
  sin.reserve(static_cast<size_t>(m) * kRotaryDim);
  for (int i = 0; i < m; ++i) {
    const BatchJob& job = *jobs[i];
    sp.off[i] = i;
    sp.len[i] = 1;
    sp.hist[i] = job.seq_len;
    sp.slot[i] = job.slot;
    ids[i] = job.next_token;
    logit_rows[i] = i;
    const Positions one{{job.next_pos}, {job.next_pos}, {job.next_pos},
                        job.next_pos + 1};
    std::vector<Bf16> c, s2;
    build_text_rope(one, 1, &c, &s2);
    cos.insert(cos.end(), c.begin(), c.end());
    sin.insert(sin.end(), s2.begin(), s2.end());
  }
  UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(im.d_ids, ids.data(),
                                        static_cast<size_t>(m) * sizeof(int),
                                        cudaMemcpyHostToDevice, st));
  index_embeddings(im.bf("model.language_model.embed_tokens.weight"), im.d_ids,
                   im.hidden, m, kHidden, st);
  UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(im.mrope_cos, cos.data(),
                                        cos.size() * sizeof(Bf16),
                                        cudaMemcpyHostToDevice, st));
  UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(im.mrope_sin, sin.data(),
                                        sin.size() * sizeof(Bf16),
                                        cudaMemcpyHostToDevice, st));
  const auto t0 = Clock::now();
  const std::vector<int> tokens =
      batch_forward(im, sp, m, m, /*use_gemv=*/true, logit_rows.data(), jobs);
  // One line per decode step: the batch size and the wall time of the forward.
  std::fprintf(stderr, "qwen38 batch step m=%d rows=%d ms=%.2f\n", m, m,
               ms_since(t0));
  for (int i = 0; i < m; ++i) {
    jobs[i]->next_token = tokens[i];
    jobs[i]->seq_len += 1;
    jobs[i]->next_pos += 1;
    jobs[i]->result.decode_steps += 1;
  }
}

// The single batch thread. Requests queue behind it; each iteration admits
// whatever fits, prefills it, then advances every active request by one step.
void batch_loop(Engine::Impl& im) {
  // A new thread starts on device 0. The engine's stream and buffers belong to
  // im.gpu, so bind this thread before the first launch; otherwise every
  // launch fails with "invalid resource handle".
  try {
    UMM_QWEN38_CUDA_CHECK(cudaSetDevice(im.gpu));
  } catch (...) {
    // No forward can run; each admitted request reports the CUDA error below.
  }
  while (true) {
    std::vector<std::shared_ptr<BatchJob>> incoming;
    {
      std::unique_lock<std::mutex> lock(im.queue_mu);
      if (im.active.empty() && im.queue.empty())
        im.queue_cv.wait(lock, [&] { return im.stopping || !im.queue.empty(); });
      if (im.stopping) break;
      const int room = static_cast<int>(im.free_slots.size());
      for (int i = 0; i < room && !im.queue.empty(); ++i) {
        incoming.push_back(std::move(im.queue.front()));
        im.queue.pop_front();
      }
    }

    std::vector<std::shared_ptr<BatchJob>> admitted;
    for (auto& job : incoming) {
      if (job->req.abort && job->req.abort()) {
        job->error = std::make_exception_ptr(GenerationAborted("queued"));
        finish_job(job);
        continue;
      }
      const int P = static_cast<int>(job->req.prompt_ids.size());
      if (P + job->req.max_new_tokens > im.max_seq) {
        job->error = std::make_exception_ptr(
            std::runtime_error("prompt + max_new_tokens exceeds max_seq"));
        finish_job(job);
        continue;
      }
      // Resolve the prefix here, before any group prefills: the group a job
      // joins depends on how many rows it will actually compute, and every job
      // admitted in one iteration must see the snapshot as it stood before any
      // of them touched it.
      job->prefix = match_prefix(im.cached_ids, job->req.prompt_ids,
                                 job->req.image != nullptr, im.prefix_cache);
      job->slot = im.free_slots.back();
      im.free_slots.pop_back();
      admitted.push_back(job);
    }

    // One forward per group. A group is homogeneous in the linear-op path the
    // single-request executor would choose, and stays inside max_seq rows.
    std::vector<std::vector<std::shared_ptr<BatchJob>>> groups;
    std::vector<int> group_rows;
    for (auto& job : admitted) {
      const int P = static_cast<int>(job->req.prompt_ids.size());
      const int rows_i = P - job->prefix;
      const bool single = (rows_i == 1);
      // The vision tower writes one shared workspace and the pooler is copied
      // after the embedding pass, so an image request prefills alone.
      const bool loner = (job->req.image != nullptr);
      bool placed = false;
      if (!loner)
        for (size_t g = 0; g < groups.size(); ++g) {
          if (groups[g].front()->req.image != nullptr) continue;
          const int first = static_cast<int>(groups[g].front()->req.prompt_ids.size()) -
                            groups[g].front()->prefix;
          if ((first == 1) != single) continue;
          if (group_rows[g] + rows_i > im.max_seq) continue;
          groups[g].push_back(job);
          group_rows[g] += rows_i;
          placed = true;
          break;
        }
      if (!placed) {
        groups.push_back({job});
        group_rows.push_back(rows_i);
      }
    }

    for (size_t g = 0; g < groups.size(); ++g) {
      try {
        prefill_group(im, groups[g]);
      } catch (...) {
        for (auto& job : groups[g]) {
          job->error = std::current_exception();
          release_slot(im, job);
          finish_job(job);
        }
        continue;
      }
      for (auto& job : groups[g]) {
        job->decode_started = Clock::now();
        if (emit_token(job)) {
          retire_job(im, job);
        } else {
          im.active.push_back(job);
        }
      }
    }

    if (im.active.empty()) continue;

    std::vector<std::shared_ptr<BatchJob>> step_jobs;
    for (auto& job : im.active) {
      if (job->req.abort && job->req.abort()) {
        job->error = std::make_exception_ptr(GenerationAborted("decode"));
        release_slot(im, job);
        finish_job(job);
        continue;
      }
      step_jobs.push_back(job);
    }
    im.active.clear();
    if (step_jobs.empty()) continue;
    try {
      step_decode(im, step_jobs);
    } catch (...) {
      for (auto& job : step_jobs) {
        job->error = std::current_exception();
        release_slot(im, job);
        finish_job(job);
      }
      continue;
    }
    for (auto& job : step_jobs) {
      if (emit_token(job)) {
        retire_job(im, job);
      } else {
        im.active.push_back(job);
      }
    }
  }

  // Shutdown: every queued and active request gets a terminal error.
  std::vector<std::shared_ptr<BatchJob>> leftover;
  {
    std::lock_guard<std::mutex> lock(im.queue_mu);
    while (!im.queue.empty()) {
      leftover.push_back(std::move(im.queue.front()));
      im.queue.pop_front();
    }
  }
  for (auto& job : im.active) leftover.push_back(job);
  im.active.clear();
  for (auto& job : leftover) {
    job->error = std::make_exception_ptr(std::runtime_error("engine is stopping"));
    release_slot(im, job);
    finish_job(job);
  }
}

}  // namespace

GenResult Engine::generate(const std::vector<int>& prompt_ids,
                           const PreprocessedImage* image, int max_new_tokens,
                           const std::string& dump_dir, const AbortCheck& abort,
                           const TokenCallback& on_token,
                           const SamplingParams& sampling,
                           int cache_prefix_len) {
  Impl& im = *impl_;
  if (im.batch_capacity > 1)
    throw std::runtime_error(
        "generate() cannot run on an engine built for continuous batching");
  auto& st = im.stream;
  // Checked on the host between kernel launches, so it changes no result.
  auto check_abort = [&abort](const char* boundary) {
    if (abort && abort()) throw GenerationAborted(boundary);
  };
  const int P = static_cast<int>(prompt_ids.size());
  if (P <= 0) throw std::runtime_error("empty prompt");
  if (P + max_new_tokens > im.max_seq)
    throw std::runtime_error("prompt + max_new_tokens exceeds max_seq");
  if (!dump_dir.empty()) {
    write_file(dump_dir, "prompt_ids.i32", prompt_ids.data(),
               prompt_ids.size() * sizeof(int));
  }

  // Prefix reuse: continue from the snapshot the previous successful pure-text
  // prefill left behind when this prompt extends it exactly. A parity dump
  // always runs the full prompt.
  int prefix = 0;
  if (dump_dir.empty() && image == nullptr)
    prefix = match_prefix(im.cached_ids, prompt_ids, /*has_image=*/false,
                          im.prefix_cache);
  im.cached_len = 0;
  im.cached_ids.clear();
  // Any failure leaves the live caches in an unknown state, so the snapshot is
  // published only when the call reaches its normal return.
  struct SnapshotGuard {
    Engine::Impl& im;
    bool keep = false;
    ~SnapshotGuard() {
      if (!keep) {
        im.cached_len = 0;
        im.cached_ids.clear();
      }
    }
  } snapshot_guard{im};
  const int M = P - prefix;

  std::mt19937_64 rng(sampling.has_seed ? sampling.seed : std::random_device{}());
  auto pick_token = [&]() -> int {
    if (!(sampling.temperature > 0.0f))
      return argmax_f32(im.logits, kVocab, im.argmax_scratch, st);
    std::vector<float> host(static_cast<size_t>(kVocab));
    UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(host.data(), im.logits,
                                          host.size() * sizeof(float),
                                          cudaMemcpyDeviceToHost, st));
    UMM_QWEN38_CUDA_CHECK(cudaStreamSynchronize(st));
    return sample_token(host.data(), kVocab, sampling, rng);
  };

  GenResult result;
  result.prompt_tokens = P;
  result.prefill_tokens = M;
  result.reused_tokens = prefix;

  double vision_ms = 0.0;
  int n_img = 0;
  int img_start = -1;
  if (image) {
    check_abort("vision");
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

  // Snapshot boundary. -1 keeps the whole prompt; an interior cut at the turn
  // boundary lets the next turn of the same conversation reuse it.
  const int snap_len = (image == nullptr && im.prefix_cache && dump_dir.empty())
                           ? resolve_snapshot_len(P, cache_prefix_len)
                           : P;
  // Two-phase prefill is only bit-identical to one pass when both phases take
  // the multi-row GEMM path: a single-row phase would switch the linear ops to
  // the GEMV kernel, whose reduction order differs. A one-row phase therefore
  // keeps the original single pass.
  const bool split = snap_len > prefix + 1 && P - snap_len > 1;

  const auto t_prefill = Clock::now();
  // The linear-attention recurrent state and the conv history are cumulative.
  // They are allocated once for the process, so every request must start from
  // zero: otherwise request N+1 continues from request N's history. A reused
  // prefix instead restores the state the snapshot holds at its boundary. The
  // KV cache needs no reset because prefill rewrites positions [0, P).
  if (prefix == 0) {
    UMM_QWEN38_CUDA_CHECK(cudaMemsetAsync(im.rec_state, 0, rec_state_bytes(), st));
    UMM_QWEN38_CUDA_CHECK(
        cudaMemsetAsync(im.conv_state, 0, conv_state_bytes(), st));
  } else {
    restore_prefix(im, /*slot=*/0, prefix, st);
  }

  const Positions pos = build_positions(
      prompt_ids, image ? image->grid_h : 0, image ? image->grid_w : 0, image != nullptr);
  std::vector<Bf16> cos, sin;
  build_text_rope(pos, P, &cos, &sin);

  // One prefill pass over the prompt rows [begin, end). Everything before
  // `begin` is already in the KV cache and the recurrent state.
  auto prefill_rows = [&](int begin, int end) {
    const int m = end - begin;
    UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(im.d_ids, prompt_ids.data() + begin,
                                          m * sizeof(int), cudaMemcpyHostToDevice,
                                          st));
    index_embeddings(im.bf("model.language_model.embed_tokens.weight"), im.d_ids,
                     im.hidden, m, kHidden, st);
    if (image) {
      UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(
          im.hidden + static_cast<size_t>(img_start) * kHidden, im.pooler,
          static_cast<size_t>(n_img) * kHidden * sizeof(Bf16),
          cudaMemcpyDeviceToDevice, st));
    }
    const size_t rope_off = static_cast<size_t>(begin) * kRotaryDim;
    const size_t rope_bytes = static_cast<size_t>(m) * kRotaryDim * sizeof(Bf16);
    UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(im.mrope_cos, cos.data() + rope_off,
                                          rope_bytes, cudaMemcpyHostToDevice, st));
    UMM_QWEN38_CUDA_CHECK(cudaMemcpyAsync(im.mrope_sin, sin.data() + rope_off,
                                          rope_bytes, cudaMemcpyHostToDevice, st));
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
    for (int l = 0; l < kLayers; ++l) {
      check_abort("prefill");
      run_layer(im, l, m, begin, dump_dir);
    }
  };

  // The last pass owns the row the first generated token is read from.
  const int last_begin = split ? snap_len : prefix;
  if (!split) {
    prefill_rows(prefix, P);
  } else {
    prefill_rows(prefix, snap_len);
    // Publish the snapshot at the turn boundary before the suffix is
    // prefilled. The KV rows [0, snap_len) and the recurrent state at
    // snap_len are exactly what a prefill of only those tokens would leave.
    snapshot_prefix(im, /*slot=*/0,
                    std::vector<int>(prompt_ids.begin(),
                                     prompt_ids.begin() + snap_len),
                    st);
    prefill_rows(snap_len, P);
  }

  // Snapshot at the prompt boundary, before the first decode row is written.
  if (!split && image == nullptr && im.prefix_cache)
    snapshot_prefix(im, /*slot=*/0, prompt_ids, st);

  rms_norm_zero_centered(im.hidden + static_cast<size_t>(P - last_begin - 1) * kHidden,
                         im.bf("model.language_model.norm.weight"), im.norm, 1,
                         kHidden, kEps, st);
  gemv_bf16_f32(im.bf("lm_head.weight"), im.norm, im.logits, kVocab, kHidden, st);
  int next = pick_token();
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
    if (on_token) on_token(next);
    if (static_cast<int>(result.ids.size()) >= max_new_tokens) break;
    check_abort("decode");
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
    next = pick_token();
    ++seq_len;
    ++next_pos;
    ++decode_forwards;
  }
  result.stopped_by_eos = (max_new_tokens > 0) && is_eos(next);
  UMM_QWEN38_CUDA_CHECK(cudaStreamSynchronize(st));
  result.decode_ms = ms_since(t_decode);
  result.decode_steps = static_cast<int>(decode_forwards);

  if (!dump_dir.empty()) {
    write_file(dump_dir, "gen_ids.i32", result.ids.data(),
               result.ids.size() * sizeof(int));
  }
  snapshot_guard.keep = true;
  return result;
}

GenResult Engine::generate_batched(const BatchRequest& request) {
  Impl& im = *impl_;
  if (im.batch_capacity <= 1)
    throw std::runtime_error("generate_batched requires batch_capacity > 1");
  if (request.prompt_ids.empty()) throw std::runtime_error("empty prompt");
  if (request.max_new_tokens <= 0)
    throw std::runtime_error("max_new_tokens must be positive");
  auto job = std::make_shared<BatchJob>();
  job->req = request;
  // Seed here, on the calling thread: the batch thread only advances the rng,
  // and a request's draw sequence must depend on its own seed alone.
  job->rng.seed(request.sampling.has_seed ? request.sampling.seed
                                          : std::random_device{}());
  {
    std::lock_guard<std::mutex> lock(im.queue_mu);
    if (im.stopping) throw std::runtime_error("engine is stopping");
    // The batch thread starts on the first request, so the serial engine keeps
    // its original single-thread shape and no work happens at construction.
    if (!im.batch_thread.joinable()) {
      im.batch_thread = std::thread([this] { batch_loop(*impl_); });
    }
    im.queue.push_back(job);
  }
  im.queue_cv.notify_one();
  std::unique_lock<std::mutex> lock(job->mu);
  job->cv.wait(lock, [&job] { return job->done; });
  if (job->error) std::rethrow_exception(job->error);
  return job->result;
}

int Engine::batch_capacity() const { return impl_->batch_capacity; }

}  // namespace umminfer::qwen38
