// SPDX-License-Identifier: Apache-2.0
#include "text_encoder.hpp"

#include "cuda_kernels.hpp"
#include "weights.hpp"

#include <cmath>

namespace umminfer::image {
namespace {
constexpr float kEps = 1e-6f;
constexpr float kTheta = 1000000.0f;
}  // namespace

TextEncoder::TextEncoder(CudaRuntime& rt, const std::string& dir)
    : rt_(rt), st_(std::make_unique<ShardedSafeTensors>(dir)) {
  embed_ = load_bf16(*st_, "model.embed_tokens.weight");
  final_norm_ = load_bf16(*st_, "model.norm.weight");
  layers_.reserve(kLayers);
  for (int i = 0; i < kLayers; ++i) {
    const std::string p = "model.layers." + std::to_string(i) + ".";
    LayerW w;
    w.q_w = load_bf16(*st_, p + "self_attn.q_proj.weight");
    w.q_b = load_bf16(*st_, p + "self_attn.q_proj.bias");
    w.k_w = load_bf16(*st_, p + "self_attn.k_proj.weight");
    w.k_b = load_bf16(*st_, p + "self_attn.k_proj.bias");
    w.v_w = load_bf16(*st_, p + "self_attn.v_proj.weight");
    w.v_b = load_bf16(*st_, p + "self_attn.v_proj.bias");
    w.o_w = load_bf16(*st_, p + "self_attn.o_proj.weight");
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
  auto& s = rt_.stream;
  // Embedding lookup via a gather kernel (index_embeddings style): build an
  // id buffer and select rows.
  int* id_dev = nullptr;
  IMG_CUDA_CHECK(cudaMalloc(&id_dev, seq * sizeof(int)));
  IMG_CUDA_CHECK(cudaMemcpy(id_dev, ids.data(), seq * sizeof(int), cudaMemcpyHostToDevice));
  DeviceTensor hidden(std::vector<int64_t>{seq, kHidden});
  gather_rows(embed_.data, id_dev, hidden.data, seq, kHidden, s);
  IMG_CUDA_CHECK(cudaFree(id_dev));

  // Precompute RoPE cos/sin (fp32), standard 1-D since all three mrope axes
  // coincide for text-only input.
  const int half = kHeadDim / 2;
  std::vector<float> cos_h(seq * half), sin_h(seq * half);
  for (int pos = 0; pos < seq; ++pos)
    for (int i = 0; i < half; ++i) {
      const float freq = 1.0f / std::pow(kTheta, (2.0f * i) / kHeadDim);
      const float angle = pos * freq;
      cos_h[pos * half + i] = std::cos(angle);
      sin_h[pos * half + i] = std::sin(angle);
    }
  DeviceTensorF cos_dev(seq * half), sin_dev(seq * half);
  IMG_CUDA_CHECK(cudaMemcpy(cos_dev.data, cos_h.data(), cos_h.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
  IMG_CUDA_CHECK(cudaMemcpy(sin_dev.data, sin_h.data(), sin_h.size() * sizeof(float),
                            cudaMemcpyHostToDevice));

  DeviceTensor normed(std::vector<int64_t>{seq, kHidden});
  DeviceTensor q(std::vector<int64_t>{seq, kHidden});
  DeviceTensor k_t(std::vector<int64_t>{seq, kKvHeads * kHeadDim});
  DeviceTensor v_t(std::vector<int64_t>{seq, kKvHeads * kHeadDim});
  DeviceTensor attn(std::vector<int64_t>{seq, kHidden});
  DeviceTensor gate(std::vector<int64_t>{seq, kInter});
  DeviceTensor up(std::vector<int64_t>{seq, kInter});
  DeviceTensor down(std::vector<int64_t>{seq, kHidden});

  for (const LayerW& w : layers_) {
    // Attention block.
    rms_norm_lastdim(hidden.data, w.norm1.data, normed.data, seq, kHidden, kEps, s);
    linear_forward(rt_, normed.data, w.q_w.data, w.q_b.data, q.data, seq, kHidden, kHidden);
    linear_forward(rt_, normed.data, w.k_w.data, w.k_b.data, k_t.data, seq,
                   kKvHeads * kHeadDim, kHidden);
    linear_forward(rt_, normed.data, w.v_w.data, w.v_b.data, v_t.data, seq,
                   kKvHeads * kHeadDim, kHidden);
    // q/k are stored [seq, heads, head_dim]; apply rotate_half RoPE.
    rope_rotate_half(q.data, cos_dev.data, sin_dev.data, seq, kQHeads, kHeadDim, s);
    rope_rotate_half(k_t.data, cos_dev.data, sin_dev.data, seq, kKvHeads, kHeadDim, s);
    gqa_causal_attn(q.data, k_t.data, v_t.data, attn.data, 1, kQHeads, kKvHeads, seq, seq,
                    kHeadDim, s);
    linear_forward(rt_, attn.data, w.o_w.data, nullptr, q.data, seq, kHidden, kHidden);
    add_residual(hidden.data, q.data, seq * kHidden, s);

    // MLP block: SwiGLU.
    rms_norm_lastdim(hidden.data, w.norm2.data, normed.data, seq, kHidden, kEps, s);
    linear_forward(rt_, normed.data, w.gate_w.data, nullptr, gate.data, seq, kInter, kHidden);
    linear_forward(rt_, normed.data, w.up_w.data, nullptr, up.data, seq, kInter, kHidden);
    silu_mul(gate.data, gate.data, up.data, seq * kInter, s);
    linear_forward(rt_, gate.data, w.down_w.data, nullptr, down.data, seq, kHidden, kInter);
    add_residual(hidden.data, down.data, seq * kHidden, s);
  }
  // Final norm in place into a fresh output to keep the pre-norm buffer free.
  DeviceTensor out(std::vector<int64_t>{seq, kHidden});
  rms_norm_lastdim(hidden.data, final_norm_.data, out.data, seq, kHidden, kEps, s);
  rt_.sync();
  return out;
}

}  // namespace umminfer::image
