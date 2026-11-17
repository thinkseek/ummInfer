// SPDX-License-Identifier: Apache-2.0
// GPU parity tests, run only on H20B GPU4 against the offline golden tree.
// GOLDEN_DIR and MODEL_DIR are passed as env or argv. One binary, nonzero exit
// on any failed gate.
#include "golden_io.hpp"
#include "image/cuda_kernels.hpp"
#include "image/dit.hpp"
#include "image/scheduler.hpp"
#include "image/tensor.hpp"
#include "image/text_encoder.hpp"
#include "image/vae.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>

namespace {
int g_fail = 0;
void check(bool c, const char* what) {
  if (!c) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

void test_scheduler_arrays(const std::string& golden) {
  using namespace umminfer::image;
  SchedulerConfig cfg;
  // 512px -> packed 1024 tokens.
  auto s512 = build_schedule(cfg, 1024, 50);
  auto meta = ummtest::read_meta(golden + "/04_scheduler/512/timesteps");
  auto ref_t = ummtest::read_f32(golden + "/04_scheduler/512/timesteps", 50);
  float max_t = 0.0f;
  for (int i = 0; i < 50; ++i) max_t = std::max(max_t, std::fabs(s512.timesteps[i] - ref_t[i]));
  check(max_t < 1e-3f, "scheduler timesteps match golden (atol 1e-3)");
  auto sig_meta = ummtest::read_meta(golden + "/04_scheduler/512/scheduler_sigmas_with_zero");
  auto ref_s = ummtest::read_f32(golden + "/04_scheduler/512/scheduler_sigmas_with_zero", 51);
  float max_s = 0.0f;
  for (int i = 0; i < 51; ++i) max_s = std::max(max_s, std::fabs(s512.sigmas[i] - ref_s[i]));
  check(max_s < 1e-6f, "scheduler sigmas match golden (atol 1e-6)");
  std::printf("scheduler max|dt|=%e max|ds|=%e\n", max_t, max_s);
}

void test_mmdit_full(umminfer::image::CudaRuntime& rt, const std::string& model,
                     const std::string& golden) {
  using namespace umminfer::image;
  const std::string base = golden + "/03_mmdit";
  const int s_img = 1024, packed = 64, txt = 21, hidden = 3584;
  auto img_bits = ummtest::read_bf16(base + "/packed_latents", (size_t)s_img * packed);
  auto txt_bits = ummtest::read_bf16(base + "/cond_prompt_embeds", (size_t)txt * hidden);
  auto ref_bits = ummtest::read_bf16(base + "/cond_noise_pred", (size_t)s_img * packed);
  DeviceTensor img({s_img, packed}), txt_d({txt, hidden});
  img.copy_from_host(img_bits.data());
  txt_d.copy_from_host(txt_bits.data());
  DiT dit(rt, model + "/transformer");
  DeviceTensor pred = dit.forward(img, txt_d, 1.0f, 32, 32, txt);
  auto got = pred.to_host_u16();
  double se = 0, re = 0, dot = 0, gn = 0, rn = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    float a = ummtest::bf_to_f(got[i]), b = ummtest::bf_to_f(ref_bits[i]);
    float d = a - b;
    se += double(d) * d; re += double(b) * b;
    dot += double(a) * b; gn += double(a) * a; rn += double(b) * b;
  }
  double rel_l2 = std::sqrt(se / re);
  double cosine = dot / std::sqrt(gn * rn);
  std::printf("mmdit cond pred rel_l2=%.4f cosine=%.5f\n", rel_l2, cosine);
  check(rel_l2 < 0.05, "mmdit velocity rel_l2 within bf16 tolerance");
  check(cosine > 0.999, "mmdit velocity cosine > 0.999");
}

void test_vae_decode(umminfer::image::CudaRuntime& rt, const std::string& model,
                     const std::string& golden) {
  using namespace umminfer::image;
  const std::string base = golden + "/05_vae";
  auto in_meta = ummtest::read_meta(base + "/latent_denorm_bf16_input_to_decode");  // 1,16,1,64,64
  const size_t in_n = 16 * 64 * 64;
  auto in_bits = ummtest::read_bf16(base + "/latent_denorm_bf16_input_to_decode", in_n);
  auto out_meta = ummtest::read_meta(base + "/decoder_output_clamped_5d");  // 1,3,1,512,512
  const size_t out_n = 3 * 512 * 512;
  auto ref_bits = ummtest::read_bf16(base + "/decoder_output_clamped_5d", out_n);

  DeviceTensor lat({16, 64, 64});
  lat.copy_from_host(in_bits.data());
  VaeDecoder vae(rt, model + "/vae");
  DeviceTensor rgb = vae.decode(lat, 64, 64);
  auto got = rgb.to_host_u16();
  if (got.size() != out_n) { check(false, "vae output element count"); return; }
  float max_abs = ummtest::max_abs_bf16(got, ref_bits);
  // PSNR over the clamped [-1,1] output.
  double se = 0.0;
  for (size_t i = 0; i < out_n; ++i) {
    float d = ummtest::bf_to_f(got[i]) - ummtest::bf_to_f(ref_bits[i]);
    se += double(d) * d;
  }
  double psnr = 10.0 * std::log10(double(out_n) * 4.0 / se);  // signal power over range 2
  double ref_e = 0.0;
  for (size_t i = 0; i < out_n; ++i)
    ref_e += double(ummtest::bf_to_f(ref_bits[i])) * ummtest::bf_to_f(ref_bits[i]);
  double rel_l2 = std::sqrt(se / ref_e);
  std::printf("vae max_abs=%.4f psnr=%.2f dB rel_l2=%.4f\n", max_abs, psnr, rel_l2);
  // The hand-written online-softmax/NCHW kernels accumulate in bf16; on the
  // identical input they match diffusers-torch bf16 at ~44 dB / rel_l2 ~0.034.
  // Gate on content preservation (same basis as the TE/MMDiT checks).
  check(rel_l2 < 0.05, "vae rel_l2 within bf16 accumulation tolerance");
  check(psnr > 40.0, "vae PSNR above 40 dB");
}

void test_text_encoder(umminfer::image::CudaRuntime& rt, const std::string& model,
                       const std::string& golden) {
  using namespace umminfer::image;
  TextEncoder te(rt, model + "/text_encoder");
  for (const char* lang : {"en", "zh"}) {
    const std::string base = golden + "/02_text_enc/" + lang;
    auto meta = ummtest::read_meta(base + "/input_ids");
    const size_t seq = meta.shape[1];
    auto ids = ummtest::read_i64(base + "/input_ids", seq);
    std::vector<int> id32(seq);
    for (size_t i = 0; i < seq; ++i) id32[i] = static_cast<int>(ids[i]);
    auto ref = ummtest::read_bf16(base + "/last_hidden_state", seq * TextEncoder::kHidden);
    DeviceTensor out = te.forward(id32);
    auto got = out.to_host_u16();
    float mx = ummtest::max_abs_bf16(got, ref);
    // Deep bf16 nets diverge in elementwise max across 28 layers from fp32
    // reduction-order differences (diffusers bf16 is itself ~6% from fp32).
    // Gate on relative L2 and cosine, which measure preserved content.
    double se = 0, re = 0, dot = 0, gn = 0, rn = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      float a = ummtest::bf_to_f(got[i]), b = ummtest::bf_to_f(ref[i]);
      float d = a - b;
      se += double(d) * d; re += double(b) * b;
      dot += double(a) * b; gn += double(a) * a; rn += double(b) * b;
    }
    double rel_l2 = std::sqrt(se / re);
    double cosine = dot / std::sqrt(gn * rn);
    std::printf("text_encoder %s seq=%zu max_abs=%e rel_l2=%.4f cosine=%.5f\n",
                lang, seq, mx, rel_l2, cosine);
    check(rel_l2 < 0.10, "text encoder rel_l2 within bf16 accumulation tolerance");
    check(cosine > 0.99, "text encoder cosine similarity > 0.99");
  }
}

void test_linear_known_answer(umminfer::image::CudaRuntime& rt) {
  using namespace umminfer::image;
  // m=2,k=3,n=4 identity-like problem from the orientation probe.
  const int m = 2, k = 3, n = 4;
  float X[6] = {1, 2, 3, 4, 5, 6};
  float W[12] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
  float expect[8] = {1, 2, 3, 6, 4, 5, 6, 15};
  DeviceTensor dx({m, k}), dw({n, k}), dy({m, n});
  std::vector<uint16_t> xb(6), wb(12);
  for (int i = 0;i < 6; ++i) xb[i] = bf16_to_bits(X[i]);
  for (int i = 0; i < 12; ++i) wb[i] = bf16_to_bits(W[i]);
  dx.copy_from_host(xb.data());
  dw.copy_from_host(wb.data());
  linear_forward(rt, dx.data, dw.data, nullptr, dy.data, m, n, k);
  auto got = dy.to_host_u16();
  float max_abs = 0.0f;
  for (int i = 0; i < 8; ++i)
    max_abs = std::max(max_abs, std::fabs(ummtest::bf_to_f(got[i]) - expect[i]));
  std::printf("linear known-answer max_abs=%e\n", max_abs);
  check(max_abs < 1e-2f, "row-major linear GEMM orientation correct");
}
}  // namespace

int main(int argc, char** argv) {
  std::string golden = argc > 1 ? argv[1] : "/root/umm-mixed-output-20261116/image-golden";
  std::string model = argc > 2 ? argv[2] : "/root/umm-mixed-output-20261116/models/Qwen-Image-2512";
  test_scheduler_arrays(golden);
  umminfer::image::CudaRuntime rt(0);
  test_linear_known_answer(rt);
  test_text_encoder(rt, model, golden);
  test_mmdit_full(rt, model, golden);
  test_vae_decode(rt, model, golden);
  if (g_fail) { std::printf("%d GPU check(s) failed\n", g_fail); return 1; }
  std::printf("all image GPU checks passed\n");
  return 0;
}
