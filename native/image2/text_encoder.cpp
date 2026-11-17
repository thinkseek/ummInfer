// SPDX-License-Identifier: Apache-2.0
#include "text_encoder.hpp"

#include "cuda_kernels.hpp"
#include "weights.hpp"

#include <cmath>
#include <stdexcept>

namespace umminfer::image2 {
namespace {
constexpr float kEps = 1e-6f;
constexpr float kTheta = 5000000.0f;
constexpr const char* kPrefix = "model.language_model.";
}  // namespace

TextEncoder::TextEncoder(CudaRuntime& rt, const std::string& dir)
    : rt_(rt), st_(std::make_unique<ShardedSafeTensors>(dir)) {
  embed_ = load_bf16(*st_, std::string(kPrefix) + "embed_tokens.weight");
  layers_.reserve(kLayers);
  for (int i = 0; i < kLayers; ++i) {
    const std::string p = std::string(kPrefix) + "layers." + std::to_string(i) + ".";
    LayerW w;
    w.q_w = load_bf16(*st_, p + "self_attn.q_proj.weight");
    w.k_w = load_bf16(*st_, p + "self_attn.k_proj.weight");
    w.v_w = load_bf16(*st_, p + "self_attn.v_proj.weight");
    w.o_w = load_bf16(*st_, p + "self_attn.o_proj.weight");
    w.q_norm = load_bf16(*st_, p + "self_attn.q_norm.weight");
    w.k_norm = load_bf16(*st_, p + "self_attn.k_norm.weight");
    w.norm1 = load_bf16(*st_, p + "input_layernorm.weight");
    w.norm2 = load_bf16(*st_, p + "post_attention_layernorm.weight");
    w.gate_w = load_bf16(*st_, p + "mlp.gate_proj.weight");
    w.up_w = load_bf16(*st_, p + "mlp.up_proj.weight");
    w.down_w = load_bf16(*st_, p + "mlp.down_proj.weight");
    layers_.push_back(std::move(w));
  }
}

DeviceTensor TextEncoder::forward(const std::vector<int>& ids) {
  const int seq = static_cast<int>(ids.size());
  if (seq <= 0) throw std::invalid_argument("empty prompt token sequence");
  auto& s = rt_.stream;

  int* id_dev = nullptr;
  IMG_CUDA_CHECK(cudaMalloc(&id_dev, seq * sizeof(int)));
  IMG_CUDA_CHECK(cudaMemcpy(id_dev, ids.data(), seq * sizeof(int), cudaMemcpyHostToDevice));
  DeviceTensor hidden(std::vector<int64_t>{seq, kHidden});
  gather_rows(embed_.data, id_dev, hidden.data, seq, kHidden, s);
  IMG_CUDA_CHECK(cudaFree(id_dev));

  // Text-only Qwen3-VL mRoPE: all three axes share the same position, so the
  // interleaved layout degenerates to standard 1-D rotate_half RoPE.
  const int half = kHeadDim / 2;
  std::vector<float> cos_h(static_cast<size_t>(seq) * half);
  std::vector<float> sin_h(static_cast<size_t>(seq) * half);
  for (int pos = 0; pos < seq; ++pos)
    for (int i = 0; i < half; ++i) {
      const float freq = 1.0f / std::pow(kTheta, (2.0f * i) / kHeadDim);
      const float angle = pos * freq;
      cos_h[static_cast<size_t>(pos) * half + i] = std::cos(angle);
      sin_h[static_cast<size_t>(pos) * half + i] = std::sin(angle);
    }
  DeviceTensorF cos_dev(cos_h.size()), sin_dev(sin_h.size());
  IMG_CUDA_CHECK(cudaMemcpy(cos_dev.data, cos_h.data(), cos_h.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
  IMG_CUDA_CHECK(cudaMemcpy(sin_dev.data, sin_h.data(), sin_h.size() * sizeof(float),
                            cudaMemcpyHostToDevice));

  DeviceTensor normed({seq, kHidden});
  DeviceTensor q({seq, kHidden});
  DeviceTensor k_t({seq, kKvHeads * kHeadDim});
  DeviceTensor v_t({seq, kKvHeads * kHeadDim});
  DeviceTensor attn({seq, kHidden});
  DeviceTensor gate({seq, kInter});
  DeviceTensor up({seq, kInter});
  DeviceTensor down({seq, kHidden});
  DeviceTensor proj({seq, kHidden});

  for (const LayerW& w : layers_) {
    rms_norm_lastdim(hidden.data, w.norm1.data, normed.data, seq, kHidden, kEps, s);
    linear_forward(rt_, normed.data, w.q_w.data, nullptr, q.data, seq, kHidden, kHidden);
    linear_forward(rt_, normed.data, w.k_w.data, nullptr, k_t.data, seq,
                   kKvHeads * kHeadDim, kHidden);
    linear_forward(rt_, normed.data, w.v_w.data, nullptr, v_t.data, seq,
                   kKvHeads * kHeadDim, kHidden);
    // q/k RMSNorm runs over each head's 128 channels.
    rms_norm_lastdim(q.data, w.q_norm.data, q.data, seq * kQHeads, kHeadDim, kEps, s);
    rms_norm_lastdim(k_t.data, w.k_norm.data, k_t.data, seq * kKvHeads, kHeadDim, kEps, s);
    rope_rotate_half(q.data, cos_dev.data, sin_dev.data, seq, kQHeads, kHeadDim, s);
    rope_rotate_half(k_t.data, cos_dev.data, sin_dev.data, seq, kKvHeads, kHeadDim, s);
    gqa_causal_attn(q.data, k_t.data, v_t.data, attn.data, 1, kQHeads, kKvHeads, seq, seq,
                    kHeadDim, s);
    linear_forward(rt_, attn.data, w.o_w.data, nullptr, proj.data, seq, kHidden, kHidden);
    add_residual(hidden.data, proj.data, seq * kHidden, s);

    rms_norm_lastdim(hidden.data, w.norm2.data, normed.data, seq, kHidden, kEps, s);
    linear_forward(rt_, normed.data, w.gate_w.data, nullptr, gate.data, seq, kInter, kHidden);
    linear_forward(rt_, normed.data, w.up_w.data, nullptr, up.data, seq, kInter, kHidden);
    silu_mul(gate.data, gate.data, up.data, seq * kInter, s);
    linear_forward(rt_, gate.data, w.down_w.data, nullptr, down.data, seq, kHidden, kInter);
    add_residual(hidden.data, down.data, seq * kHidden, s);
  }
  // No final RMSNorm: the pipeline disables it for Qwen-Image 2.1.
  rt_.sync();
  return hidden;
}

}  // namespace umminfer::image2
