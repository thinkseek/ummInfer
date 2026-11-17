// SPDX-License-Identifier: Apache-2.0
#include "omni_engine.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "cuda_kernels_omni.hpp"
#include "engine_internal.hpp"
#include "tokenizer.hpp"

namespace umminfer::omni {
namespace {

std::string read_text(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::invalid_argument("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

}  // namespace

struct OmniEngine::Impl {
  int gpu_id = 0;
  int max_seq = 2048;
  std::unique_ptr<ShardedSafetensors> shards;
  EngineWeights weights;
  EngineScratch scratch;
  std::unique_ptr<Qwen2BpeTokenizer> tokenizer;

  void load_weights(const std::string& model_dir) {
    shards = std::make_unique<ShardedSafetensors>(model_dir);
    // Copy every thinker visual/text/lm_head tensor onto the device. Skip
    // audio_tower, talker and code2wav entirely (~8.5 GB not resident).
    size_t bytes = 0;
    for (const auto& [name, info] : shards->tensors()) {
      if (info.dtype != "BF16")
        throw std::runtime_error("only BF16 supported: " + name);
      const bool wanted =
          name.rfind("thinker.visual.", 0) == 0 ||
          name.rfind("thinker.model.", 0) == 0 ||
          name.rfind("thinker.lm_head.", 0) == 0;
      if (!wanted) continue;
      Bf16* d;
      UMM_CUDA_CHECK(cudaMalloc(&d, info.byte_size));
      UMM_CUDA_CHECK(cudaMemcpy(d, shards->tensor_bytes(name), info.byte_size,
                                cudaMemcpyHostToDevice));
      weights.ptr.emplace(name, d);
      bytes += info.byte_size;
    }
    std::fprintf(stderr, "loaded %zu thinker tensors, %.2f GiB weights\n",
                 weights.ptr.size(), bytes / 1073741824.0);
  }

  void alloc_scratch() {
    auto& s = scratch;
    s.max_seq = max_seq;
    UMM_CUDA_CHECK(cudaStreamCreate(&s.stream));
    auto mk = [&](Bf16*& p, size_t n) {
      UMM_CUDA_CHECK(cudaMalloc(&p, n * sizeof(Bf16)));
    };
    mk(s.hidden, static_cast<size_t>(max_seq) * kTextH);
    mk(s.norm, static_cast<size_t>(max_seq) * kTextH);
    mk(s.q, static_cast<size_t>(max_seq) * kQHeads * kHeadDim);
    mk(s.k, static_cast<size_t>(max_seq) * kKvHeads * kHeadDim);
    mk(s.v, static_cast<size_t>(max_seq) * kKvHeads * kHeadDim);
    mk(s.gate, static_cast<size_t>(max_seq) * kExpertInter);
    mk(s.up, static_cast<size_t>(max_seq) * kExpertInter);
    mk(s.moe_acc, static_cast<size_t>(max_seq) * kTextH);
    UMM_CUDA_CHECK(cudaMalloc(&s.router_logits,
                              static_cast<size_t>(max_seq) * kNumExperts *
                                  sizeof(float)));
    UMM_CUDA_CHECK(cudaMalloc(&s.router_w,
                              static_cast<size_t>(max_seq) * kTopK * sizeof(float)));
    UMM_CUDA_CHECK(cudaMalloc(&s.logits, kVocab * sizeof(float)));
    UMM_CUDA_CHECK(cudaMalloc(&s.router_idx,
                              static_cast<size_t>(max_seq) * kTopK * sizeof(int)));
    UMM_CUDA_CHECK(cudaMalloc(&s.argmax_scratch, sizeof(int)));
    UMM_CUDA_CHECK(cudaMalloc(&s.d_vis_pos, max_seq * sizeof(int)));
    UMM_CUDA_CHECK(cudaMalloc(&s.mrope_cos,
                              static_cast<size_t>(max_seq) * kHeadDim * sizeof(float)));
    UMM_CUDA_CHECK(cudaMalloc(&s.mrope_sin,
                              static_cast<size_t>(max_seq) * kHeadDim * sizeof(float)));
    // KV cache: [48][2][max_seq][4][128].
    UMM_CUDA_CHECK(cudaMalloc(&s.kv, sizeof(Bf16) * kTextLayers * 2 * max_seq *
                                          kKvHeads * kHeadDim));
    // Vision workspaces (bounded by the 768px preprocessor cap).
    constexpr int kVT = 1296;
    mk(s.v_hidden, static_cast<size_t>(kVT) * kVisionH);
    mk(s.v_norm, static_cast<size_t>(kVT) * kVisionH);
    mk(s.v_qkv, static_cast<size_t>(kVT) * 3 * kVisionH);
    mk(s.v_qp, static_cast<size_t>(kVT) * kVisionH);
    mk(s.v_kp, static_cast<size_t>(kVT) * kVisionH);
    mk(s.v_vp, static_cast<size_t>(kVT) * kVisionH);
    mk(s.v_proj, static_cast<size_t>(kVT) * std::max(kVisionH, 4 * kVisionH));
    mk(s.v_mlp, static_cast<size_t>(kVT) * kVisionInter);
    mk(s.v_rope_cos, static_cast<size_t>(kVT) * kVisionHeadDim);
    mk(s.v_rope_sin, static_cast<size_t>(kVT) * kVisionHeadDim);
    mk(s.v_shuffle, static_cast<size_t>(kVT / 4) * 4 * kVisionH);
    mk(s.pooler, static_cast<size_t>(kVT / 4) * kVisionOut);
    for (int i = 0; i < 3; ++i)
      mk(s.deepstack[i], static_cast<size_t>(kVT / 4) * kVisionOut);
  }
};

OmniEngine::OmniEngine(const std::string& model_dir, int gpu_id, int max_seq) {
  impl_ = std::make_unique<Impl>();
  impl_->gpu_id = gpu_id;
  impl_->max_seq = max_seq;
  UMM_CUDA_CHECK(cudaSetDevice(gpu_id));
  impl_->load_weights(model_dir);
  impl_->alloc_scratch();
  impl_->tokenizer = std::make_unique<Qwen2BpeTokenizer>(
      model_dir + "/vocab.json", model_dir + "/merges.txt",
      model_dir + "/tokenizer_config.json");
}

OmniEngine::~OmniEngine() {
  if (!impl_) return;
  for (const auto& [name, p] : impl_->weights.ptr)
    cudaFree(const_cast<Bf16*>(p));
  auto& s = impl_->scratch;
  cudaFree(s.kv);
  cudaFree(s.mrope_sin);
  cudaFree(s.mrope_cos);
  cudaFree(s.d_vis_pos);
  cudaFree(s.argmax_scratch);
  cudaFree(s.router_idx);
  cudaFree(s.logits);
  cudaFree(s.router_w);
  cudaFree(s.router_logits);
  cudaFree(s.moe_acc);
  cudaFree(s.up);
  cudaFree(s.gate);
  cudaFree(s.v);
  cudaFree(s.k);
  cudaFree(s.q);
  cudaFree(s.norm);
  cudaFree(s.hidden);
  for (int i = 0; i < 3; ++i) cudaFree(s.deepstack[i]);
  cudaFree(s.pooler);
  cudaFree(s.v_shuffle);
  cudaFree(s.v_rope_sin);
  cudaFree(s.v_rope_cos);
  cudaFree(s.v_mlp);
  cudaFree(s.v_proj);
  cudaFree(s.v_vp);
  cudaFree(s.v_kp);
  cudaFree(s.v_qp);
  cudaFree(s.v_qkv);
  cudaFree(s.v_norm);
  cudaFree(s.v_hidden);
  if (s.stream) cudaStreamDestroy(s.stream);
}

OmniEngine::Result OmniEngine::generate(const std::vector<int>& prompt_ids,
                                        const PreprocessedImage* image,
                                        int max_new_tokens,
                                        const AbortCheck& abort) {
  Impl& im = *impl_;
  auto& s = im.scratch;
  const int S = static_cast<int>(prompt_ids.size());
  if (S + max_new_tokens > im.max_seq)
    throw std::runtime_error("sequence exceeds bounded KV capacity");

  // Vision tower, then replace the image_pad rows.
  int img_start = -1, n_img = 0;
  if (image) {
    Bf16* d_pix;
    const size_t pix_n =
        static_cast<size_t>(image->rows()) * PreprocessedImage::kPatchDim;
    std::vector<Bf16> h_pix(pix_n);
    for (size_t i = 0; i < pix_n; ++i) h_pix[i] = Bf16(image->pixel_values[i]);
    UMM_CUDA_CHECK(cudaMalloc(&d_pix, pix_n * sizeof(Bf16)));
    UMM_CUDA_CHECK(cudaMemcpy(d_pix, h_pix.data(), pix_n * sizeof(Bf16),
                              cudaMemcpyHostToDevice));
    vision_forward(im.weights, s, *image, d_pix, abort);
    cudaFree(d_pix);
    // Optional parity dump: UMM_OMNI_DUMP_DIR writes pooler/deepstack as f32.
    if (const char* dump_dir = std::getenv("UMM_OMNI_DUMP_DIR")) {
      const int n_vis = image->rows() / 4;
      auto dump_dev = [&](const char* name, const Bf16* dev) {
        std::vector<float> hf(static_cast<size_t>(n_vis) * kVisionOut);
        std::vector<Bf16> hb(hf.size());
        cudaMemcpy(hb.data(), dev, hb.size() * sizeof(Bf16),
                   cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < hf.size(); ++i) hf[i] = float(hb[i]);
        std::ofstream f(std::string(dump_dir) + "/" + name + ".bin",
                        std::ios::binary);
        f.write(reinterpret_cast<const char*>(hf.data()),
                hf.size() * sizeof(float));
      };
      dump_dev("pooler", s.pooler);
      for (int k = 0; k < 3; ++k)
        dump_dev(("deepstack_" + std::to_string(k)).c_str(), s.deepstack[k]);
      std::fprintf(stderr, "dumped vision tensors to %s\n", dump_dir);
    }
    for (int i = 0; i < S; ++i)
      if (prompt_ids[i] == Qwen2BpeTokenizer::kImagePad) {
        if (img_start < 0) img_start = i;
        ++n_img;
      }
    if (n_img != image->rows() / 4)
      throw std::runtime_error("image placeholder/feature count mismatch");
  }

  // Embedding gather.
  UMM_CUDA_CHECK(cudaMemcpy(s.d_vis_pos, prompt_ids.data(),
                            S * sizeof(int), cudaMemcpyHostToDevice));
  index_embeddings(im.weights.get("thinker.model.embed_tokens.weight"),
                   s.d_vis_pos, s.hidden, S, kTextH, s.stream);
  if (image)
    replace_rows(s.hidden, s.pooler, img_start, n_img, kTextH, s.stream);

  // Position ids and the prefill RoPE table.
  const bool has_image = image != nullptr;
  PositionIds pos = build_positions(
      prompt_ids, Qwen2BpeTokenizer::kImagePad,
      Qwen2BpeTokenizer::kVisionStart, Qwen2BpeTokenizer::kVisionEnd,
      has_image ? image->grid_h : 0, has_image ? image->grid_w : 0, has_image);
  build_text_rope_table(pos.t.data(), pos.h.data(), pos.w.data(), S, s);

  // Visual row indices for deepstack injection.
  std::vector<int> vis_rows;
  if (has_image)
    for (int i = img_start; i < img_start + n_img; ++i) vis_rows.push_back(i);
  if (has_image)
    cudaMemcpyAsync(s.d_vis_pos, vis_rows.data(),
                    vis_rows.size() * sizeof(int), cudaMemcpyHostToDevice,
                    s.stream);

  // ---- prefill ----
  for (int l = 0; l < kTextLayers; ++l) {
    if (abort && abort()) {
      throw RequestAborted("aborted in prefill layer " + std::to_string(l));
    }
    text_layer(im.weights, s, l, S, 0);
    if (has_image && l < 3)
      add_deepstack(s.hidden, s.deepstack[l], s.d_vis_pos, n_img, kTextH,
                    s.stream);
  }
  rms_norm(s.hidden + static_cast<size_t>(S - 1) * kTextH,
           im.weights.get("thinker.model.norm.weight"), s.norm, 1, kTextH,
           kRmsEps, s.stream);
  gemv_bf16(im.weights.get("thinker.lm_head.weight"), s.norm, s.logits,
             kVocab, kTextH, s.stream);
  int next = argmax_f32(s.logits, kVocab, s.argmax_scratch, s.stream);

  Result result;
  result.prefill_tokens = S;
  result.image_tokens = n_img;
  int seq_len = S;
  int next_pos = pos.next_pos;
  for (int step = 0;
       step < max_new_tokens && !im.tokenizer->is_eos(next); ++step) {
    if (abort && abort()) {
      throw RequestAborted("aborted at token " + std::to_string(step));
    }
    result.token_ids.push_back(next);
    UMM_CUDA_CHECK(cudaMemcpy(s.d_vis_pos, &next, sizeof(int),
                              cudaMemcpyHostToDevice));
    index_embeddings(im.weights.get("thinker.model.embed_tokens.weight"),
                     s.d_vis_pos, s.hidden, 1, kTextH, s.stream);
    const int p3[1] = {next_pos};
    build_text_rope_table(p3, p3, p3, 1, s);
    for (int l = 0; l < kTextLayers; ++l) text_layer(im.weights, s, l, 1, seq_len);
    ++seq_len;
    ++next_pos;
    rms_norm(s.hidden, im.weights.get("thinker.model.norm.weight"), s.norm, 1,
             kTextH, kRmsEps, s.stream);
    gemv_bf16(im.weights.get("thinker.lm_head.weight"), s.norm, s.logits,
               kVocab, kTextH, s.stream);
    next = argmax_f32(s.logits, kVocab, s.argmax_scratch, s.stream);
  }
  UMM_CUDA_CHECK(cudaStreamSynchronize(s.stream));
  result.text = im.tokenizer->decode(result.token_ids);
  return result;
}

}  // namespace umminfer::omni
