// SPDX-License-Identifier: Apache-2.0
#include "engine.hpp"

#include "cuda_kernels.hpp"
#include "dit.hpp"
#include "packing.hpp"
#include "png.hpp"
#include "scheduler.hpp"
#include "text_encoder.hpp"
#include "tokenizer.hpp"
#include "vae.hpp"

#include <chrono>
#include <cmath>

namespace umminfer::image {
namespace {
constexpr int kLatentCh = 16;
constexpr int kMaxSeq = 512;  // pipeline max_sequence_length default

// Reinterpret a BF16 bit pattern as a host float.
float bf16_bits_to_float(uint16_t bits) {
  uint32_t u = uint32_t(bits) << 16;
  float f;
  __builtin_memcpy(&f, &u, sizeof(f));
  return f;
}
uint16_t float_to_bf16_bits(float f) { return bf16_to_bits(f); }

// VAE latent mean / inverse-std constants (vae/config.json).
const float kLatMean[16] = {-0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f, 0.9653f,
                            -0.1517f, 1.5508f, 0.4134f, -0.0715f, 0.5517f, -0.3632f,
                            -0.1922f, -0.9497f, 0.2503f, -0.2921f};
const float kLatStd[16] = {2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f,
                           2.0743f, 3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f,
                           2.8251f, 1.9160f};
}  // namespace

ImageEngine::ImageEngine(int gpu_id, const std::string& model_dir)
    : gpu_(gpu_id), model_dir_(model_dir) {}

GenerateResult ImageEngine::generate(const GenerateParams& gp) {
  using clock = std::chrono::steady_clock;
  const auto t_start = clock::now();
  CudaRuntime rt(gpu_);

  // ---- tokenize (host) ----
  Qwen2BpeTokenizer tokenizer(model_dir_ + "/tokenizer");
  auto cond_ids = tokenizer.encode_prompt(gp.prompt);
  auto neg_ids = tokenizer.encode_prompt(gp.negative_prompt);
  if (static_cast<int>(cond_ids.size()) > Qwen2BpeTokenizer::kTemplatePrefixTokens + kMaxSeq)
    cond_ids.resize(Qwen2BpeTokenizer::kTemplatePrefixTokens + kMaxSeq);
  if (static_cast<int>(neg_ids.size()) > Qwen2BpeTokenizer::kTemplatePrefixTokens + kMaxSeq)
    neg_ids.resize(Qwen2BpeTokenizer::kTemplatePrefixTokens + kMaxSeq);
  // Drop the fixed 34-token template prefix; keep the prompt + suffix rows.
  cond_ids.erase(cond_ids.begin(),
                 cond_ids.begin() + Qwen2BpeTokenizer::kTemplatePrefixTokens);
  neg_ids.erase(neg_ids.begin(),
                neg_ids.begin() + Qwen2BpeTokenizer::kTemplatePrefixTokens);

  auto t_loaded_components = clock::now();
  TextEncoder te(rt, model_dir_ + "/text_encoder");
  DiT dit(rt, model_dir_ + "/transformer");
  VaeDecoder vae(rt, model_dir_ + "/vae");
  const auto t_after_load = clock::now();

  // ---- text encode ----
  DeviceTensor cond_embed = te.forward(cond_ids);
  DeviceTensor neg_embed = te.forward(neg_ids);
  const auto t_after_te = clock::now();

  // ---- initial packed latent ----
  const int latent_h = 2 * (gp.height / 16);
  const int latent_w = 2 * (gp.width / 16);
  const int grid_h = latent_h / 2, grid_w = latent_w / 2;
  const int packed_tokens = grid_h * grid_w;
  DeviceTensor latent({packed_tokens, 64});
  randn_bf16_philox(latent.data, gp.seed, (size_t)packed_tokens * 64, rt.stream);

  // ---- schedule ----
  SchedulerConfig scfg;
  FlowMatchSchedule sched = build_schedule(scfg, packed_tokens, gp.steps);

  // CFG work buffers.
  DeviceTensor neg_v({packed_tokens, 64});

  const auto t_denoise_start = clock::now();
  for (int step = 0; step < gp.steps; ++step) {
    const float t_scaled = sched.timesteps[step] / 1000.0f;
    DeviceTensor vc = dit.forward(latent, cond_embed, t_scaled, grid_h, grid_w,
                                  static_cast<int>(cond_ids.size()));
    DeviceTensor vn = dit.forward(latent, neg_embed, t_scaled, grid_h, grid_w,
                                  static_cast<int>(neg_ids.size()));
    neg_v = std::move(vn);
    // cfg_combine writes the guided prediction into vc.
    cfg_combine(vc.data, vc.data, neg_v.data, gp.guidance, packed_tokens, 64, rt.stream);
    const float dt = sched.sigmas[step + 1] - sched.sigmas[step];
    euler_step_bf16(latent.data, vc.data, dt, packed_tokens * 64, rt.stream);
  }
  rt.sync();
  const auto t_after_denoise = clock::now();

  // ---- unpack + de-normalize + VAE ----
  auto packed_host = latent.to_host_u16();
  std::vector<float> packed_f32(packed_host.size());
  for (size_t i = 0; i < packed_host.size(); ++i)
    packed_f32[i] = bf16_bits_to_float(packed_host[i]);
  PackedLatents pl;
  pl.tokens = packed_tokens; pl.features = 64; pl.h2 = grid_h; pl.w2 = grid_w;
  pl.data = packed_f32;
  auto chw = unpack_latents(pl, kLatentCh, latent_h, latent_w);

  DeviceTensor lat_denorm({kLatentCh, latent_h, latent_w});
  {
    DeviceTensor lat_bf({kLatentCh, latent_h, latent_w});
    std::vector<uint16_t> bits(chw.size());
    for (size_t i = 0; i < chw.size(); ++i) bits[i] = float_to_bf16_bits(chw[i]);
    lat_bf.copy_from_host(bits.data());
    float mean[16], inv_std[16];
    for (int c = 0; c < 16; ++c) { mean[c] = kLatMean[c]; inv_std[c] = 1.0f / kLatStd[c]; }
    DeviceTensorF dm(16), dis(16);
    IMG_CUDA_CHECK(cudaMemcpy(dm.data, mean, 16 * 4, cudaMemcpyHostToDevice));
    IMG_CUDA_CHECK(cudaMemcpy(dis.data, inv_std, 16 * 4, cudaMemcpyHostToDevice));
    denorm_latents(lat_denorm.data, lat_bf.data, dm.data, dis.data, latent_h * latent_w,
                   rt.stream);
  }
  DeviceTensor rgb = vae.decode(lat_denorm, latent_h, latent_w);
  const auto t_after_vae = clock::now();

  // ---- to uint8 RGB HWC ----
  auto rgb_host = rgb.to_host_u16();
  RgbImage img(gp.width, gp.height);
  for (int y = 0; y < gp.height; ++y)
    for (int x = 0; x < gp.width; ++x)
      for (int c = 0; c < 3; ++c) {
        float v = bf16_bits_to_float(
            rgb_host[((size_t)c * gp.height + y) * gp.width + x]);
        v = std::fmin(1.0f, std::fmax(0.0f, v * 0.5f + 0.5f));
        img.rgb[((size_t)y * gp.width + x) * 3 + c] =
            static_cast<uint8_t>(std::lrint(v * 255.0f));
      }

  GenerateResult r;
  r.rgb_u16.assign(img.rgb.begin(), img.rgb.end());
  r.width = gp.width; r.height = gp.height;
  r.load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_load -
                                                                    t_loaded_components).count();
  r.encode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_te -
                                                                       t_after_load).count();
  r.denoise_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_denoise -
                                                                        t_denoise_start).count();
  r.vae_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_vae -
                                                                   t_after_denoise).count();
  r.total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t_start).count();
  size_t free_b = 0, total_b = 0;
  cudaMemGetInfo(&free_b, &total_b);
  r.peak_gpu_bytes = total_b - free_b;
  return r;
}

}  // namespace umminfer::image
