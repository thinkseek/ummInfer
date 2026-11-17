// SPDX-License-Identifier: Apache-2.0
#include "engine.hpp"

#include "cuda_kernels.hpp"
#include "dit.hpp"
#include "scheduler.hpp"
#include "text_encoder.hpp"
#include "tokenizer.hpp"
#include "vae.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>

namespace umminfer::image2 {
namespace {

using Clock = std::chrono::steady_clock;

int ms_since(Clock::time_point t0) {
  return static_cast<int>(
      std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count());
}

size_t free_bytes() {
  size_t free_mem = 0, total = 0;
  IMG_CUDA_CHECK(cudaMemGetInfo(&free_mem, &total));
  return free_mem;
}

std::vector<float> read_float_array(const nlohmann::json& j) {
  return j.get<std::vector<float>>();
}

}  // namespace

Image2Engine::Image2Engine(int gpu_id, const std::string& model_dir)
    : gpu_(gpu_id), model_dir_(model_dir) {}

GenerateResult Image2Engine::generate(const GenerateParams& p, const AbortCheck& abort) {
  auto t_start = Clock::now();
  auto t = Clock::now();
  GenerateResult res;

  int width = (p.width / 32) * 32;
  int height = (p.height / 32) * 32;
  if (width <= 0 || height <= 0) throw std::invalid_argument("bad image size");
  const int gh = height / 16, gw = width / 16;

  CudaRuntime rt(gpu_);
  size_t min_free = free_bytes();

  // ---- tokenizer + text encoder -------------------------------------------
  Qwen2BpeTokenizer tok(model_dir_ + "/processor");
  const std::vector<int> ids = tok.encode_prompt(p.prompt);
  const int drop = tok.system_prefix_tokens();
  if (drop < 0 || drop >= static_cast<int>(ids.size()))
    throw std::runtime_error("prompt shorter than the system prefix");

  DeviceTensor txt;
  int txt_len = 0;
  {
    auto t_load = Clock::now();
    TextEncoder te(rt, model_dir_ + "/text_encoder");
    res.load_ms += ms_since(t_load);
    auto t_enc = Clock::now();
    DeviceTensor hidden = te.forward(ids);
    txt_len = static_cast<int>(ids.size()) - drop;
    txt = DeviceTensor(std::vector<int64_t>{txt_len, TextEncoder::kHidden});
    IMG_CUDA_CHECK(cudaMemcpyAsync(txt.data, hidden.data + (size_t)drop * TextEncoder::kHidden,
                                   (size_t)txt_len * TextEncoder::kHidden * sizeof(Bf16),
                                   cudaMemcpyDeviceToDevice, rt.stream));
    rt.sync();
    res.encode_ms = ms_since(t_enc);
    min_free = std::min(min_free, free_bytes());
  }

  // ---- scheduler + noise ---------------------------------------------------
  SchedulerConfig scfg;
  const FlowMatchSchedule sched = build_schedule(scfg, gh * gw, p.steps);

  DeviceTensor packed({gh * gw, 64});
  randn_bf16_philox(packed.data, p.seed, packed.numel, rt.stream);
  rt.sync();

  // ---- DiT denoising -------------------------------------------------------
  {
    auto t_load = Clock::now();
    DiT dit(rt, model_dir_ + "/transformer");
    res.load_ms += ms_since(t_load);
    min_free = std::min(min_free, free_bytes());

    auto t_den = Clock::now();
    for (int i = 0; i < p.steps; ++i) {
      if (abort && abort()) throw std::runtime_error("aborted");
      // diffusers rounds the timestep to bf16, divides by 1000 in bf16, then
      // the model multiplies it back by 1000 in fp32. Mirror both roundings.
      const float t_bf = bits_to_bf16_float(bf16_to_bits(sched.timesteps[i]));
      const float t_scaled = bits_to_bf16_float(bf16_to_bits(t_bf / 1000.0f));
      DeviceTensor vel = dit.forward(packed, txt, t_scaled, gh, gw, txt_len);
      euler_step_bf16(packed.data, vel.data, sched.sigmas[i + 1] - sched.sigmas[i],
                      static_cast<int>(packed.numel), rt.stream);
      if ((i + 1) % 10 == 0)
        std::fprintf(stderr, "  step %d/%d\n", i + 1, p.steps);
    }
    rt.sync();
    res.denoise_ms = ms_since(t_den);
    min_free = std::min(min_free, free_bytes());
  }

  // ---- VAE decode ----------------------------------------------------------
  {
    DeviceTensor lat({64, gh * gw});
    tokens_to_nchw(packed.data, lat.data, 64, gh * gw, rt.stream);

    std::ifstream cfg_file(model_dir_ + "/vae/config.json");
    if (!cfg_file) throw std::runtime_error("cannot read vae/config.json");
    nlohmann::json cfg;
    cfg_file >> cfg;
    const std::vector<float> mean = read_float_array(cfg.at("latents_mean"));
    const std::vector<float> stdv = read_float_array(cfg.at("latents_std"));
    DeviceTensorF mean_d(mean.size()), std_d(stdv.size());
    IMG_CUDA_CHECK(cudaMemcpy(mean_d.data, mean.data(), mean.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    IMG_CUDA_CHECK(cudaMemcpy(std_d.data, stdv.data(), stdv.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    DeviceTensor denorm({64, gh * gw});
    denorm_latents_std(lat.data, mean_d.data, std_d.data, denorm.data, 64, gh * gw,
                       rt.stream);

    auto t_load = Clock::now();
    VaeDecoder vae(rt, model_dir_ + "/vae");
    res.load_ms += ms_since(t_load);
    auto t_vae = Clock::now();
    DeviceTensor rgb = vae.decode(denorm, gh, gw);
    rt.sync();

    const int H = gh * 16, W = gw * 16;
    auto host = rgb.to_host_u16();
    std::vector<float> chw(static_cast<size_t>(4) * H * W);
    for (size_t i = 0; i < chw.size(); ++i) chw[i] = bits_to_bf16_float(host[i]);
    res.image = tensor_n1p1_to_rgba(chw.data(), H, W);
    res.vae_ms = ms_since(t_vae);
    min_free = std::min(min_free, free_bytes());
  }

  size_t total_mem = 0, free_mem = 0;
  IMG_CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
  res.peak_gpu_bytes = total_mem - min_free;
  res.total_ms = ms_since(t_start);
  (void)t;
  return res;
}

}  // namespace umminfer::image2
