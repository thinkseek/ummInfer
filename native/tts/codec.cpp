// SPDX-License-Identifier: Apache-2.0
#include "codec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include "common/cuda_request_allocator.hpp"
#include "cuda_kernels.hpp"
#include "decode_helpers.hpp"

namespace umminfer::native {
namespace {

constexpr int kCodebook = 2048;
constexpr int kVqDim = 256;
constexpr int kLatent = 512;
constexpr int kHidden = 1024;    // conv / upsample channels
constexpr int kTrHidden = 512;   // pre-transformer hidden
constexpr int kTrHeads = 16;
constexpr int kTrKvHeads = 16;   // MHA: q/k/v proj all output 1024 = 16*64
constexpr int kHeadDim = 64;
constexpr int kTrLayers = 8;
constexpr int kWindow = 72;
constexpr int kGroups = 16;
constexpr int kDecoderDim = 1536;
constexpr int kUpsample = 1920;
constexpr int kMaxFrames = 8192;

// Bounded chunk size for stateful streaming & chunked one-shot decode.
// Keeps peak scratch VRAM constant (~118 MB for a..e instead of up to 8.8 GB).
constexpr int kCodecChunk = 32;
// Rolling KV cache capacity for the 8-layer pre-transformer:
// 192 >= 2*(kWindow - 1) + kCodecChunk = 174, so compaction copies are strictly
// non-overlapping.
constexpr int kTrKvCap = 192;

constexpr float kRmsEps = 1e-5f;
constexpr float kLnEps = 1e-6f;
constexpr float kVqEps = 1e-5f;

const float* need(const std::unordered_map<std::string, const float*>& w,
                  const std::string& name) {
  auto it = w.find(name);
  if (it == w.end()) throw std::runtime_error("codec missing tensor: " + name);
  return it->second;
}

struct TrLayerWeights {
  const float* input_ln = nullptr;
  const float* q_w = nullptr;
  const float* k_w = nullptr;
  const float* v_w = nullptr;
  const float* o_w = nullptr;
  const float* attn_scale = nullptr;
  const float* post_ln = nullptr;
  const float* gate_w = nullptr;
  const float* up_w = nullptr;
  const float* down_w = nullptr;
  const float* mlp_scale = nullptr;
};

struct UpsampleStageWeights {
  const float* up_w = nullptr;
  const float* up_b = nullptr;
  const float* dw_w = nullptr;
  const float* dw_b = nullptr;
  const float* ln_w = nullptr;
  const float* ln_b = nullptr;
  const float* pw1_w = nullptr;
  const float* pw1_b = nullptr;
  const float* pw2_w = nullptr;
  const float* pw2_b = nullptr;
  const float* gamma = nullptr;
};

struct ResUnitWeights {
  const float* act1_a = nullptr;
  const float* act1_b = nullptr;
  const float* conv1_w = nullptr;
  const float* conv1_b = nullptr;
  const float* act2_a = nullptr;
  const float* act2_b = nullptr;
  const float* conv2_w = nullptr;
  const float* conv2_b = nullptr;
};

struct DecoderBlockWeights {
  const float* act_a = nullptr;
  const float* act_b = nullptr;
  const float* tconv_w = nullptr;
  const float* tconv_b = nullptr;
  ResUnitWeights units[3];
};

}  // namespace

struct CodecDecoder::Impl {
  cudaStream_t stream = nullptr;

  // Pre-resolved weight pointers (zero std::string / map lookups in decode).
  const float* rvq_first_proj_w = nullptr;
  const float* rvq_rest_proj_w = nullptr;
  const float* pre_conv_w = nullptr;
  const float* pre_conv_b = nullptr;
  const float* tr_in_w = nullptr;
  const float* tr_in_b = nullptr;
  const float* tr_norm_w = nullptr;
  const float* tr_out_w = nullptr;
  const float* tr_out_b = nullptr;
  TrLayerWeights tr_layers[kTrLayers];
  UpsampleStageWeights upsample[2];
  const float* dec0_w = nullptr;
  const float* dec0_b = nullptr;
  DecoderBlockWeights dec_blocks[4];
  const float* dec5_a = nullptr;
  const float* dec5_b = nullptr;
  const float* dec6_w = nullptr;
  const float* dec6_b = nullptr;

  // Device buffers (fixed-size, allocated once in constructor).
  float* codebooks = nullptr;  // 16 dequantized tables, each [2048, 256]
  float *a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr, *e = nullptr;
  float *rvq_first = nullptr, *rvq_rest = nullptr;
  float *q = nullptr, *k = nullptr, *v = nullptr, *attn = nullptr;
  float *gate = nullptr, *up = nullptr, *ffn_out = nullptr;
  float* d_inv = nullptr;      // RoPE inv_freq [32]
  float* tr_kv = nullptr;      // [kTrLayers * 2, kTrKvCap, kTrKvHeads * kHeadDim]
  int* d_codes_all = nullptr;  // [kCodecChunk * kGroups]

  // Single contiguous buffer for all causal Conv1d / ConvTranspose1d history.
  float* hist_buf = nullptr;
  size_t hist_floats = 0;
  float* h_pre_conv = nullptr;
  float* h_upsample_dw[2] = {nullptr, nullptr};
  float* h_dec0 = nullptr;
  float* h_tconv[4] = {nullptr, nullptr, nullptr, nullptr};
  float* h_res_conv1[4][3] = {};
  float* h_dec6 = nullptr;

  // Pinned host staging buffers for fast async DMA transfers.
  int* h_codes_pinned = nullptr;   // [kCodecChunk * kGroups]
  float* h_pcm_pinned = nullptr;   // [kCodecChunk * kUpsample]

  // Streaming state.
  int tr_cached = 0;
  int total_frames = 0;

  // Cached diagnostic env flag so production decode never calls getenv().
  bool any_dump = false;

  void linear(float* y, const float* x, const float* weight, const float* bias,
              int m, int in, int out) {
    gemm_t<float>(y, x, weight, bias, m, in, out, stream);
  }

  float* kv_slab(int layer, bool is_v) const {
    const size_t slab = static_cast<size_t>(layer) * 2u + (is_v ? 1u : 0u);
    return tr_kv + slab * kTrKvCap * (kTrKvHeads * kHeadDim);
  }
};

CodecDecoder::CodecDecoder(std::unordered_map<std::string, const float*> weights,
                           cudaStream_t stream) {
  impl_ = std::make_unique<Impl>();
  Impl& im = *impl_;
  im.stream = stream;

  auto wt = [&weights](const std::string& n) { return need(weights, n); };

  // Pre-resolve every codec weight tensor once.
  im.rvq_first_proj_w = wt("decoder.quantizer.rvq_first.output_proj.weight");
  im.rvq_rest_proj_w = wt("decoder.quantizer.rvq_rest.output_proj.weight");
  im.pre_conv_w = wt("decoder.pre_conv.conv.weight");
  im.pre_conv_b = wt("decoder.pre_conv.conv.bias");
  im.tr_in_w = wt("decoder.pre_transformer.input_proj.weight");
  im.tr_in_b = wt("decoder.pre_transformer.input_proj.bias");
  im.tr_norm_w = wt("decoder.pre_transformer.norm.weight");
  im.tr_out_w = wt("decoder.pre_transformer.output_proj.weight");
  im.tr_out_b = wt("decoder.pre_transformer.output_proj.bias");
  for (int l = 0; l < kTrLayers; ++l) {
    const std::string p = "decoder.pre_transformer.layers." + std::to_string(l);
    TrLayerWeights& tw = im.tr_layers[l];
    tw.input_ln = wt(p + ".input_layernorm.weight");
    tw.q_w = wt(p + ".self_attn.q_proj.weight");
    tw.k_w = wt(p + ".self_attn.k_proj.weight");
    tw.v_w = wt(p + ".self_attn.v_proj.weight");
    tw.o_w = wt(p + ".self_attn.o_proj.weight");
    tw.attn_scale = wt(p + ".self_attn_layer_scale.scale");
    tw.post_ln = wt(p + ".post_attention_layernorm.weight");
    tw.gate_w = wt(p + ".mlp.gate_proj.weight");
    tw.up_w = wt(p + ".mlp.up_proj.weight");
    tw.down_w = wt(p + ".mlp.down_proj.weight");
    tw.mlp_scale = wt(p + ".mlp_layer_scale.scale");
  }
  for (int s = 0; s < 2; ++s) {
    const std::string up = "decoder.upsample." + std::to_string(s) + ".0";
    const std::string cx = "decoder.upsample." + std::to_string(s) + ".1";
    UpsampleStageWeights& uw = im.upsample[s];
    uw.up_w = wt(up + ".conv.weight");
    uw.up_b = wt(up + ".conv.bias");
    uw.dw_w = wt(cx + ".dwconv.conv.weight");
    uw.dw_b = wt(cx + ".dwconv.conv.bias");
    uw.ln_w = wt(cx + ".norm.weight");
    uw.ln_b = wt(cx + ".norm.bias");
    uw.pw1_w = wt(cx + ".pwconv1.weight");
    uw.pw1_b = wt(cx + ".pwconv1.bias");
    uw.pw2_w = wt(cx + ".pwconv2.weight");
    uw.pw2_b = wt(cx + ".pwconv2.bias");
    uw.gamma = wt(cx + ".gamma");
  }
  im.dec0_w = wt("decoder.decoder.0.conv.weight");
  im.dec0_b = wt("decoder.decoder.0.conv.bias");
  for (int i = 0; i < 4; ++i) {
    const std::string blk = "decoder.decoder." + std::to_string(i + 1);
    DecoderBlockWeights& db = im.dec_blocks[i];
    db.act_a = wt(blk + ".block.0.alpha");
    db.act_b = wt(blk + ".block.0.beta");
    db.tconv_w = wt(blk + ".block.1.conv.weight");
    db.tconv_b = wt(blk + ".block.1.conv.bias");
    for (int u = 0; u < 3; ++u) {
      const std::string ru = blk + ".block." + std::to_string(u + 2);
      ResUnitWeights& rw = db.units[u];
      rw.act1_a = wt(ru + ".act1.alpha");
      rw.act1_b = wt(ru + ".act1.beta");
      rw.conv1_w = wt(ru + ".conv1.conv.weight");
      rw.conv1_b = wt(ru + ".conv1.conv.bias");
      rw.act2_a = wt(ru + ".act2.alpha");
      rw.act2_b = wt(ru + ".act2.beta");
      rw.conv2_w = wt(ru + ".conv2.conv.weight");
      rw.conv2_b = wt(ru + ".conv2.conv.bias");
    }
  }
  im.dec5_a = wt("decoder.decoder.5.alpha");
  im.dec5_b = wt("decoder.decoder.5.beta");
  im.dec6_w = wt("decoder.decoder.6.conv.weight");
  im.dec6_b = wt("decoder.decoder.6.conv.bias");

  // Dequantize the 16 RVQ codebooks once.
  UMM_CUDA_CHECK(request_cuda_malloc(&im.codebooks,
                            static_cast<size_t>(kGroups) * kCodebook * kVqDim *
                                sizeof(float)));
  auto dequant = [&](const std::string& qname, int layer, float* dst) {
    const std::string base =
        "decoder.quantizer." + qname + ".vq.layers." + std::to_string(layer);
    codebook_dequant_f32(wt(base + "._codebook.embedding_sum"),
                         wt(base + "._codebook.cluster_usage"), dst, kCodebook,
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

  // Fixed bounded scratch buffers for chunks of up to kCodecChunk frames.
  const size_t peak = codec_peak_scratch_elems(kCodecChunk);
  for (float** p : {&im.a, &im.b, &im.c, &im.d, &im.e}) {
    UMM_CUDA_CHECK(request_cuda_malloc(p, peak * sizeof(float)));
  }
  UMM_CUDA_CHECK(request_cuda_malloc(&im.d_codes_all,
                            static_cast<size_t>(kCodecChunk) * kGroups * sizeof(int)));
  UMM_CUDA_CHECK(request_cuda_malloc(&im.rvq_first,
                            static_cast<size_t>(kCodecChunk) * kLatent * sizeof(float)));
  UMM_CUDA_CHECK(request_cuda_malloc(&im.rvq_rest,
                            static_cast<size_t>(kCodecChunk) * kLatent * sizeof(float)));
  const size_t tr_q_elems = static_cast<size_t>(kCodecChunk) * kTrHeads * kHeadDim;
  UMM_CUDA_CHECK(request_cuda_malloc(&im.q, tr_q_elems * sizeof(float)));
  UMM_CUDA_CHECK(request_cuda_malloc(&im.k, tr_q_elems * sizeof(float)));
  UMM_CUDA_CHECK(request_cuda_malloc(&im.v, tr_q_elems * sizeof(float)));
  UMM_CUDA_CHECK(request_cuda_malloc(&im.attn, tr_q_elems * sizeof(float)));
  // upsample.1 pwconv1 outputs [4*kCodecChunk, 4*kHidden] floats.
  const size_t gate_elems = static_cast<size_t>(4 * kCodecChunk) * (4 * kHidden);
  UMM_CUDA_CHECK(request_cuda_malloc(&im.gate, gate_elems * sizeof(float)));
  UMM_CUDA_CHECK(request_cuda_malloc(&im.up,
                            static_cast<size_t>(kCodecChunk) * (2 * kTrHidden) *
                                sizeof(float)));
  UMM_CUDA_CHECK(request_cuda_malloc(&im.ffn_out,
                            static_cast<size_t>(kCodecChunk) * kHidden * sizeof(float)));

  // Pre-transformer rolling KV cache and RoPE inv_freq.
  const size_t kv_elems = static_cast<size_t>(kTrLayers) * 2u * kTrKvCap *
                          (kTrKvHeads * kHeadDim);
  UMM_CUDA_CHECK(request_cuda_malloc(&im.tr_kv, kv_elems * sizeof(float)));

  std::vector<float> inv(kHeadDim / 2);
  for (int i = 0; i < kHeadDim / 2; ++i)
    inv[i] = 1.0f / std::pow(10000.0f, 2.0f * i / kHeadDim);
  UMM_CUDA_CHECK(request_cuda_malloc(&im.d_inv, inv.size() * sizeof(float)));
  UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_inv, inv.data(), inv.size() * sizeof(float),
                                 cudaMemcpyHostToDevice, stream));

  // Single contiguous causal history buffer (~541 KB total).
  size_t off = 0;
  auto reserve = [&off](size_t n) {
    const size_t cur = off;
    off += n;
    return cur;
  };
  const size_t off_pre_conv = reserve(static_cast<size_t>(kLatent) * 2);
  const size_t off_up_dw0 = reserve(static_cast<size_t>(kHidden) * 6);
  const size_t off_up_dw1 = reserve(static_cast<size_t>(kHidden) * 6);
  const size_t off_dec0 = reserve(static_cast<size_t>(kHidden) * 6);
  size_t off_tconv[4];
  size_t off_res[4][3];
  const int dilations[3] = {1, 3, 9};
  int in_ch = kDecoderDim;
  for (int i = 0; i < 4; ++i) {
    const int out_ch = kDecoderDim / (1 << (i + 1));
    off_tconv[i] = reserve(static_cast<size_t>(in_ch));
    for (int u = 0; u < 3; ++u) {
      off_res[i][u] = reserve(static_cast<size_t>(out_ch) * (6 * dilations[u]));
    }
    in_ch = out_ch;
  }
  const size_t off_dec6 = reserve(96u * 6u);
  im.hist_floats = off;
  UMM_CUDA_CHECK(request_cuda_malloc(&im.hist_buf, im.hist_floats * sizeof(float)));
  im.h_pre_conv = im.hist_buf + off_pre_conv;
  im.h_upsample_dw[0] = im.hist_buf + off_up_dw0;
  im.h_upsample_dw[1] = im.hist_buf + off_up_dw1;
  im.h_dec0 = im.hist_buf + off_dec0;
  for (int i = 0; i < 4; ++i) {
    im.h_tconv[i] = im.hist_buf + off_tconv[i];
    for (int u = 0; u < 3; ++u) im.h_res_conv1[i][u] = im.hist_buf + off_res[i][u];
  }
  im.h_dec6 = im.hist_buf + off_dec6;

  // Pinned host staging buffers for DMA transfers.
  UMM_CUDA_CHECK(cudaMallocHost(&im.h_codes_pinned,
                                static_cast<size_t>(kCodecChunk) * kGroups *
                                    sizeof(int)));
  UMM_CUDA_CHECK(cudaMallocHost(&im.h_pcm_pinned,
                                static_cast<size_t>(kCodecChunk) * kUpsample *
                                    sizeof(float)));

  im.any_dump = (std::getenv("UMM_CODEC_DUMP_Q") != nullptr ||
                 std::getenv("UMM_CODEC_DUMP_PRECONV") != nullptr ||
                 std::getenv("UMM_CODEC_DUMP_ATTN0") != nullptr ||
                 std::getenv("UMM_CODEC_DUMP_ATTRES") != nullptr ||
                 std::getenv("UMM_CODEC_DUMP_L0") != nullptr ||
                 std::getenv("UMM_CODEC_DUMP_TR") != nullptr ||
                 std::getenv("UMM_CODEC_DUMP_UP") != nullptr ||
                 std::getenv("UMM_CODEC_DUMP_D0") != nullptr ||
                 std::getenv("UMM_CODEC_DUMP_DB") != nullptr);

  reset_stream();
}

CodecDecoder::~CodecDecoder() {
  if (!impl_) return;
  Impl& im = *impl_;
  if (im.h_pcm_pinned) cudaFreeHost(im.h_pcm_pinned);
  if (im.h_codes_pinned) cudaFreeHost(im.h_codes_pinned);
  request_cuda_free(im.hist_buf);
  request_cuda_free(im.d_inv);
  request_cuda_free(im.tr_kv);
  request_cuda_free(im.ffn_out);
  request_cuda_free(im.up);
  request_cuda_free(im.gate);
  request_cuda_free(im.attn);
  request_cuda_free(im.v);
  request_cuda_free(im.k);
  request_cuda_free(im.q);
  request_cuda_free(im.d_codes_all);
  request_cuda_free(im.rvq_first);
  request_cuda_free(im.rvq_rest);
  request_cuda_free(im.codebooks);
  for (float* p : {im.a, im.b, im.c, im.d, im.e}) request_cuda_free(p);
}

void CodecDecoder::reset_stream() {
  Impl& im = *impl_;
  im.tr_cached = 0;
  im.total_frames = 0;
  UMM_CUDA_CHECK(cudaMemsetAsync(im.hist_buf, 0, im.hist_floats * sizeof(float),
                                 im.stream));
}

std::vector<float> CodecDecoder::decode_chunk(const int* codes, int frames,
                                              const AbortCheck& abort) {
  Impl& im = *impl_;
  cudaStream_t st = im.stream;
  if (frames <= 0 || im.total_frames + frames > kMaxFrames)
    throw std::invalid_argument("frame count out of range");
  for (int i = 0; i < frames * kGroups; ++i)
    if (codes[i] < 0 || codes[i] >= kCodebook)
      throw std::runtime_error("codec id outside [0,2047]");

  std::vector<float> wav(static_cast<size_t>(frames) * kUpsample);
  int done = 0;
  while (done < frames) {
    if (abort && abort())
      throw RequestAborted("aborted in codec chunk");
    const int T = std::min(frames - done, kCodecChunk);
    const size_t code_count = static_cast<size_t>(T) * kGroups;
    std::memcpy(im.h_codes_pinned,
                codes + static_cast<size_t>(done) * kGroups,
                code_count * sizeof(int));
    UMM_CUDA_CHECK(cudaMemcpyAsync(im.d_codes_all, im.h_codes_pinned,
                                   code_count * sizeof(int),
                                   cudaMemcpyHostToDevice, st));

    // ---- RVQ dequantization (fused gather + sum + transpose to [256, T]) ----
    rvq_gather_sum_transposed_f32(im.codebooks, im.d_codes_all, im.b,
                                  /*first_group=*/0, /*count=*/1, T, kGroups,
                                  kCodebook, kVqDim, st);
    conv1d_causal_f32(im.b, im.rvq_first_proj_w, nullptr, im.rvq_first, kVqDim,
                      kLatent, T, 1, 1, 1, st);
    rvq_gather_sum_transposed_f32(im.codebooks, im.d_codes_all, im.b,
                                  /*first_group=*/1, /*count=*/15, T, kGroups,
                                  kCodebook, kVqDim, st);
    conv1d_causal_f32(im.b, im.rvq_rest_proj_w, nullptr, im.rvq_rest, kVqDim,
                      kLatent, T, 1, 1, 1, st);
    // Latent [512, T] in im.d = rvq_first + rvq_rest.
    UMM_CUDA_CHECK(cudaMemcpyAsync(im.d, im.rvq_first,
                                   static_cast<size_t>(T) * kLatent * sizeof(float),
                                   cudaMemcpyDeviceToDevice, st));
    add_residual_t<float>(im.d, im.rvq_rest, T * kLatent, st);

    if (im.any_dump) {
      if (const char* dump = std::getenv("UMM_CODEC_DUMP_Q")) {
        std::vector<float> h(static_cast<size_t>(T) * kLatent);
        cudaMemcpy(h.data(), im.d, h.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);
        std::ofstream f(dump, std::ios::binary);
        f.write(reinterpret_cast<const char*>(h.data()),
                h.size() * sizeof(float));
      }
    }

    // ---- Stateful pre_conv (K=3, pad=2) -------------------------------------
    conv1d_causal_stateful_f32(im.d, im.h_pre_conv, im.pre_conv_w, im.pre_conv_b,
                               im.c, kLatent, kHidden, T, 3, 1, 1, st);
    update_conv_history_f32(im.h_pre_conv, im.d, kLatent, 2, T, st);
    if (im.any_dump) {
      if (const char* dump = std::getenv("UMM_CODEC_DUMP_PRECONV")) {
        std::vector<float> h(static_cast<size_t>(T) * kHidden);
        cudaMemcpy(h.data(), im.c, h.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);
        std::ofstream f(dump, std::ios::binary);
        f.write(reinterpret_cast<const char*>(h.data()),
                h.size() * sizeof(float));
      }
    }

    // ---- Stateful pre-transformer with rolling KV window --------------------
    transpose_f32(im.c, im.a, kHidden, T, st);  // [1024, T] -> [T, 1024]
    im.linear(im.ffn_out, im.a, im.tr_in_w, im.tr_in_b, T, kHidden, kTrHidden);

    const RollingKvPlan kv_plan =
        rolling_kv_plan(im.tr_cached, T, kWindow, kTrKvCap);
    if (kv_plan.need_shift) {
      shift_kv_window_f32(im.tr_kv, kTrLayers * 2, kTrKvCap,
                          kTrKvHeads * kHeadDim, kv_plan.shift_src,
                          kv_plan.keep_len, st);
      im.tr_cached = kv_plan.keep_len;
    }

    const int pos_base = im.total_frames;
    const int kvw = kTrKvHeads * kHeadDim;
    float* hidden = im.ffn_out;
    for (int l = 0; l < kTrLayers; ++l) {
      const TrLayerWeights& tw = im.tr_layers[l];
      float* k_slab = im.kv_slab(l, false);
      float* v_slab = im.kv_slab(l, true);
      float* k_dst = k_slab + static_cast<size_t>(im.tr_cached) * kvw;
      float* v_dst = v_slab + static_cast<size_t>(im.tr_cached) * kvw;

      rms_norm_t<float>(hidden, tw.input_ln, im.a, T, kTrHidden, kRmsEps, st);
      im.linear(im.q, im.a, tw.q_w, nullptr, T, kTrHidden, kTrHeads * kHeadDim);
      im.linear(k_dst, im.a, tw.k_w, nullptr, T, kTrHidden, kvw);
      im.linear(v_dst, im.a, tw.v_w, nullptr, T, kTrHidden, kvw);
      rope_half_offset_f32(im.q, pos_base, im.d_inv, T, kTrHeads, kHeadDim, st);
      rope_half_offset_f32(k_dst, pos_base, im.d_inv, T, kTrKvHeads, kHeadDim,
                           st);
      gqa_attention_t<float>(im.q, k_slab, v_slab, im.attn, T,
                             im.tr_cached + T, kTrHeads, kTrKvHeads, kHeadDim,
                             /*mask=*/1, kWindow, /*q_offset=*/im.tr_cached, st);
      if (im.any_dump && l == 0 && std::getenv("UMM_CODEC_DUMP_ATTN0")) {
        std::vector<float> h(static_cast<size_t>(T) * kTrHeads * kHeadDim);
        cudaMemcpy(h.data(), im.attn, h.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);
        std::ofstream o(std::getenv("UMM_CODEC_DUMP_ATTN0"), std::ios::binary);
        o.write(reinterpret_cast<const char*>(h.data()),
                h.size() * sizeof(float));
      }
      im.linear(im.a, im.attn, tw.o_w, nullptr, T, kTrHeads * kHeadDim,
                kTrHidden);
      layer_scale_t<float>(im.a, im.a, tw.attn_scale, T, kTrHidden, st);
      add_residual_t<float>(hidden, im.a, T * kTrHidden, st);

      rms_norm_t<float>(hidden, tw.post_ln, im.a, T, kTrHidden, kRmsEps, st);
      im.linear(im.gate, im.a, tw.gate_w, nullptr, T, kTrHidden, 2 * kTrHidden);
      im.linear(im.up, im.a, tw.up_w, nullptr, T, kTrHidden, 2 * kTrHidden);
      silu_mul_t<float>(im.gate, im.gate, im.up, T * 2 * kTrHidden, st);
      im.linear(im.a, im.gate, tw.down_w, nullptr, T, 2 * kTrHidden, kTrHidden);
      layer_scale_t<float>(im.a, im.a, tw.mlp_scale, T, kTrHidden, st);
      add_residual_t<float>(hidden, im.a, T * kTrHidden, st);
    }
    im.tr_cached += T;
    im.total_frames += T;

    rms_norm_t<float>(hidden, im.tr_norm_w, im.a, T, kTrHidden, kRmsEps, st);
    im.linear(im.ffn_out, im.a, im.tr_out_w, im.tr_out_b, T, kTrHidden, kHidden);
    if (im.any_dump) {
      if (const char* dump = std::getenv("UMM_CODEC_DUMP_TR")) {
        std::vector<float> h(static_cast<size_t>(T) * kHidden);
        cudaMemcpy(h.data(), im.ffn_out, h.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);
        std::ofstream f(dump, std::ios::binary);
        f.write(reinterpret_cast<const char*>(h.data()),
                h.size() * sizeof(float));
      }
    }
    transpose_inv_f32(im.ffn_out, im.a, kHidden, T, st);  // [T, 1024] -> [1024, T]

    // ---- Two ConvNeXt x2 upsample stages ------------------------------------
    int len = T;
    for (int s = 0; s < 2; ++s) {
      const UpsampleStageWeights& uw = im.upsample[s];
      // ConvTranspose1d(K=2, stride=2) has zero cross-frame history.
      conv_transpose1d_f32(im.a, uw.up_w, uw.up_b, im.b, kHidden, kHidden, len,
                           2, 2, st);
      len *= 2;
      // Depthwise causal Conv1d (K=7, pad=6); im.b is read-only so kept as residual.
      conv1d_causal_stateful_f32(im.b, im.h_upsample_dw[s], uw.dw_w, uw.dw_b,
                                 im.d, kHidden, kHidden, len, 7, 1, kHidden, st);
      update_conv_history_f32(im.h_upsample_dw[s], im.b, kHidden, 6, len, st);
      transpose_f32(im.d, im.a, kHidden, len, st);
      layer_norm_t<float>(im.a, uw.ln_w, uw.ln_b, im.a, len, kHidden, kLnEps,
                          st);
      im.linear(im.gate, im.a, uw.pw1_w, uw.pw1_b, len, kHidden, 4 * kHidden);
      gelu_inplace_t<float>(im.gate, len * 4 * kHidden, st);
      im.linear(im.a, im.gate, uw.pw2_w, uw.pw2_b, len, 4 * kHidden, kHidden);
      layer_scale_t<float>(im.a, im.a, uw.gamma, len, kHidden, st);
      transpose_inv_f32(im.a, im.d, kHidden, len, st);
      add_residual_t<float>(im.b, im.d, len * kHidden, st);
      std::swap(im.a, im.b);
    }

    // ---- Waveform decoder ---------------------------------------------------
    conv1d_causal_stateful_f32(im.a, im.h_dec0, im.dec0_w, im.dec0_b, im.b,
                               kHidden, kDecoderDim, len, 7, 1, 1, st);
    update_conv_history_f32(im.h_dec0, im.a, kHidden, 6, len, st);

    const int rates[4] = {8, 5, 4, 3};
    const int dilations[3] = {1, 3, 9};
    int in_ch = kDecoderDim;
    for (int i = 0; i < 4; ++i) {
      const int out_ch = kDecoderDim / (1 << (i + 1));
      const DecoderBlockWeights& db = im.dec_blocks[i];
      snake_beta_f32(im.b, db.act_a, db.act_b, in_ch, len, st);
      conv_transpose1d_stateful_f32(im.b, im.h_tconv[i], db.tconv_w, db.tconv_b,
                                    im.c, in_ch, out_ch, len, 2 * rates[i],
                                    rates[i], st);
      save_last_sample_f32(im.h_tconv[i], im.b, in_ch, len, st);
      len *= rates[i];
      for (int u = 0; u < 3; ++u) {
        const ResUnitWeights& rw = db.units[u];
        const int pad = 6 * dilations[u];
        // Out-of-place act1 keeps im.c intact as the residual (no D2D copy).
        snake_beta_out_f32(im.c, im.d, rw.act1_a, rw.act1_b, out_ch, len, st);
        conv1d_causal_stateful_f32(im.d, im.h_res_conv1[i][u], rw.conv1_w,
                                   rw.conv1_b, im.a, out_ch, out_ch, len, 7,
                                   dilations[u], 1, st);
        update_conv_history_f32(im.h_res_conv1[i][u], im.d, out_ch, pad, len,
                                st);
        snake_beta_f32(im.a, rw.act2_a, rw.act2_b, out_ch, len, st);
        conv1d_causal_f32(im.a, rw.conv2_w, rw.conv2_b, im.e, out_ch, out_ch,
                          len, 1, 1, 1, st);
        add_residual_t<float>(im.c, im.e, len * out_ch, st);
      }
      std::swap(im.b, im.c);
      in_ch = out_ch;
    }

    snake_beta_f32(im.b, im.dec5_a, im.dec5_b, 96, len, st);
    conv1d_causal_stateful_f32(im.b, im.h_dec6, im.dec6_w, im.dec6_b, im.a, 96,
                               1, len, 7, 1, 1, st);
    update_conv_history_f32(im.h_dec6, im.b, 96, 6, len, st);

    const int chunk_samples = T * kUpsample;
    clamp_f32(im.a, chunk_samples, -1.0f, 1.0f, st);
    UMM_CUDA_CHECK(cudaMemcpyAsync(im.h_pcm_pinned, im.a,
                                   static_cast<size_t>(chunk_samples) *
                                       sizeof(float),
                                   cudaMemcpyDeviceToHost, st));
    UMM_CUDA_CHECK(cudaStreamSynchronize(st));
    std::memcpy(wav.data() + static_cast<size_t>(done) * kUpsample,
                im.h_pcm_pinned,
                static_cast<size_t>(chunk_samples) * sizeof(float));
    done += T;
  }
  return wav;
}

std::vector<float> CodecDecoder::decode(const std::vector<int>& codes, int T,
                                        const AbortCheck& abort) {
  if (T <= 0 || T > kMaxFrames)
    throw std::invalid_argument("frame count out of range");
  if (static_cast<int>(codes.size()) != T * kGroups)
    throw std::invalid_argument("codes must be [T,16]");
  reset_stream();
  return decode_chunk(codes.data(), T, abort);
}

}  // namespace umminfer::native
