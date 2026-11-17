// SPDX-License-Identifier: Apache-2.0
// GPU test: the MoE down projection must consume the expert activation in FP32.
//
// The reference implementation accumulates `y[o] = sum_k w[o,k] * act[k]` in
// FP32 against a BF16 weight matrix.  Rounding `act` to BF16 first is a real
// numeric change (up to 2^-9 relative per element), so this test builds
// activations that BF16 cannot represent exactly and asserts two things:
//
//   1. the kernel matches a float64 reference over the exact FP32 activation
//      to within FP32 accumulation-order noise (only reduction order may
//      differ), and
//   2. the kernel is *further* from a reference that pre-rounds the activation
//      to BF16, by three orders of magnitude above that noise floor.
//
// Check 2 is what fails if a BF16 activation buffer is reintroduced.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "omni/omni_gemm.hpp"

using namespace umminfer::omni;

namespace {

int g_fail = 0;
void check(bool cond, const char* what) {
  if (!cond) {
    std::printf("FAIL: %s\n", what);
    ++g_fail;
  }
}

inline float round_bf16(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}

// Deterministic LCG; no library RNG.
struct Lcg {
  unsigned s = 12345u;
  float next() {
    s = s * 1664525u + 1013904223u;
    return static_cast<float>((s >> 8) & 0xffffff) / 16777216.0f;
  }
};

// BF16 has 7 stored mantissa bits, so its ulp around 1.0 is 2^-7 and the
// midpoint between 1.0 and the next BF16 value is 1 + 2^-8.  Both patterns
// below sit strictly inside (1.0, 1 + 2^-8), so BF16 rounding moves every one
// of them down to 1.0 by 0.2%..0.34%.  They must round the *same* way: values
// that straddle a midpoint cancel and hide the regression.
float act_value(int i) {
  const float two_minus_9 = 1.0f / 512.0f;
  return 1.0f + two_minus_9 * ((i & 1) ? 1.75f : 1.0f);
}

double rel_err_dd(const std::vector<double>& a, const std::vector<double>& b) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = a[i] - b[i];
    num += d * d;
    den += b[i] * b[i];
  }
  return std::sqrt(num / den);
}

double rel_err(const std::vector<float>& got, const std::vector<double>& ref) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double d = static_cast<double>(got[i]) - ref[i];
    num += d * d;
    den += ref[i] * ref[i];
  }
  return std::sqrt(num / den);
}

}  // namespace

int main() {
  if (cudaSetDevice(0) != cudaSuccess) {
    std::printf("FAIL: no CUDA device\n");
    return 1;
  }

  constexpr int kK = 768;   // expert inter size
  constexpr int kN = 2048;  // hidden size
  constexpr int kExperts = 2;
  const int kGroup[2] = {5, 3};
  const int kPairs = kGroup[0] + kGroup[1];

  // ---- expert weights -----------------------------------------------------
  __nv_bfloat16* d_wd[2] = {nullptr, nullptr};
  std::vector<__nv_bfloat16> h_wd[2];
  Lcg rng;
  for (int e = 0; e < kExperts; ++e) {
    h_wd[e].resize(static_cast<size_t>(kN) * kK);
    // Positive weights so a BF16 activation bias cannot cancel out.
    for (size_t i = 0; i < h_wd[e].size(); ++i)
      h_wd[e][i] = __float2bfloat16_rn(0.5f + rng.next());
    check(cudaMalloc(reinterpret_cast<void**>(&d_wd[e]),
                     h_wd[e].size() * sizeof(__nv_bfloat16)) == cudaSuccess,
          "cudaMalloc wd");
    check(cudaMemcpy(d_wd[e], h_wd[e].data(),
                     h_wd[e].size() * sizeof(__nv_bfloat16),
                     cudaMemcpyHostToDevice) == cudaSuccess,
          "copy wd");
  }

  // ---- activations --------------------------------------------------------
  std::vector<float> h_act(static_cast<size_t>(kPairs) * kK);
  for (int p = 0; p < kPairs; ++p)
    for (int k = 0; k < kK; ++k) h_act[p * kK + k] = act_value(k);
  check(h_act[0] != round_bf16(h_act[0]),
        "test activation is BF16-representable (test would be vacuous)");

  float* d_act = nullptr;
  float* d_y = nullptr;
  check(cudaMalloc(reinterpret_cast<void**>(&d_act),
                   h_act.size() * sizeof(float)) == cudaSuccess,
        "malloc act");
  check(cudaMalloc(reinterpret_cast<void**>(&d_y),
                   static_cast<size_t>(kPairs) * kN * sizeof(float)) ==
            cudaSuccess,
        "malloc y");
  check(cudaMemcpy(d_act, h_act.data(), h_act.size() * sizeof(float),
                   cudaMemcpyHostToDevice) == cudaSuccess,
        "copy act");
  if (g_fail) return 1;

  // ---- launch plan --------------------------------------------------------
  MoeGroupPlan plan;
  plan.tokens = kPairs;
  plan.top_k = 1;
  plan.total_pairs = kPairs;
  plan.group_off = {0, kGroup[0]};
  plan.group_size = {kGroup[0], kGroup[1]};
  plan.pair_expert.resize(kPairs);
  plan.pair_token.resize(kPairs);
  plan.pair_of.resize(kPairs);
  plan.pair_w.assign(kPairs, 1.0f);
  for (int p = 0; p < kPairs; ++p) {
    plan.pair_expert[p] = p < kGroup[0] ? 0 : 1;
    plan.pair_token[p] = p;
    plan.pair_of[p] = p;
  }
  plan.build_down_blocks(kN / kGemmTileN);
  plan.build_blocks(1);

  MoeDeviceTables t;
  const size_t pairs = kPairs;
  check(cudaMalloc(reinterpret_cast<void**>(&t.weights),
                   3 * kExperts * sizeof(const Bf16*)) == cudaSuccess,
        "alloc wtbl");
  check(cudaMalloc(reinterpret_cast<void**>(&t.pair_weights),
                   3 * pairs * sizeof(const Bf16*)) == cudaSuccess,
        "alloc pwtbl");
  check(cudaMalloc(reinterpret_cast<void**>(&t.group_off),
                   kExperts * sizeof(int)) == cudaSuccess,
        "alloc goff");
  check(cudaMalloc(reinterpret_cast<void**>(&t.group_size),
                   kExperts * sizeof(int)) == cudaSuccess,
        "alloc gsize");
  check(cudaMalloc(reinterpret_cast<void**>(&t.pair_token),
                   pairs * sizeof(int)) == cudaSuccess,
        "alloc ptok");
  check(cudaMalloc(reinterpret_cast<void**>(&t.pair_of), pairs * sizeof(int)) ==
            cudaSuccess,
        "alloc pof");
  check(cudaMalloc(reinterpret_cast<void**>(&t.pair_w), pairs * sizeof(float)) ==
            cudaSuccess,
        "alloc pw");
  t.blk_cap = static_cast<int>(plan.blk_expert.size());
  t.down_cap = static_cast<int>(plan.down_e.size());
  const size_t cap =
      static_cast<size_t>(std::max(t.blk_cap, t.down_cap)) + 1;
  check(cudaMalloc(reinterpret_cast<void**>(&t.blk_e), cap * sizeof(int)) ==
            cudaSuccess,
        "alloc blk_e");
  check(cudaMalloc(reinterpret_cast<void**>(&t.blk_m), cap * sizeof(int)) ==
            cudaSuccess,
        "alloc blk_m");
  check(cudaMalloc(reinterpret_cast<void**>(&t.blk_n), cap * sizeof(int)) ==
            cudaSuccess,
        "alloc blk_n");
  check(cudaMalloc(reinterpret_cast<void**>(&t.down_e), cap * sizeof(int)) ==
            cudaSuccess,
        "alloc down_e");
  check(cudaMalloc(reinterpret_cast<void**>(&t.down_m), cap * sizeof(int)) ==
            cudaSuccess,
        "alloc down_m");
  check(cudaMalloc(reinterpret_cast<void**>(&t.down_n), cap * sizeof(int)) ==
            cudaSuccess,
        "alloc down_n");
  if (g_fail) return 1;

  const Bf16* wg[kExperts] = {d_wd[0], d_wd[1]};
  const Bf16* wu[kExperts] = {d_wd[0], d_wd[1]};
  const Bf16* wd[kExperts] = {d_wd[0], d_wd[1]};
  moe_upload_tables(plan, wg, wu, wd, kExperts, &t, nullptr);
  check(cudaDeviceSynchronize() == cudaSuccess, "upload tables");

  // ---- float64 references over the exact FP32 and the BF16-rounded act -----
  std::vector<std::vector<double>> ref(kPairs), ref_bf16(kPairs);
  for (int p = 0; p < kPairs; ++p) {
    const __nv_bfloat16* w = h_wd[p < kGroup[0] ? 0 : 1].data();
    ref[p].resize(kN);
    ref_bf16[p].resize(kN);
    for (int o = 0; o < kN; ++o) {
      double acc = 0.0, acc_b = 0.0;
      for (int k = 0; k < kK; ++k) {
        const double wv = static_cast<double>(__bfloat162float(w[o * kK + k]));
        const float a = h_act[p * kK + k];
        acc += wv * static_cast<double>(a);
        acc_b += wv * static_cast<double>(round_bf16(a));
      }
      ref[p][o] = acc;
      ref_bf16[p][o] = acc_b;
    }
  }

  // Guard the fixture itself: if the activation stops being BF16-inexact, or
  // the two references converge, the regression check below becomes vacuous.
  const double sep = rel_err_dd(ref_bf16[0], ref[0]);
  std::printf("fixture: rel gap between FP32-act and BF16-act references = %.3e\n",
              sep);
  check(sep > 1e-3, "fixture references are distinguishable (test is not vacuous)");

  std::vector<float> h_y(static_cast<size_t>(kPairs) * kN);
  auto report = [&](const char* tag, int p, double e_fp32, double e_bf16) {
    std::printf("%s pair %d: rel_err_vs_fp32=%.3e rel_err_vs_bf16act=%.3e\n",
                tag, p, e_fp32, e_bf16);
    check(e_fp32 < 1e-4, "matches the FP32 activation reference");
    check(e_bf16 > 5e-4, "distinct from a BF16-rounded activation");
  };

  // ---- case 1: grouped down projection (prefill shape) --------------------
  moe_down_f32(d_y, d_act, t, 2 * kExperts,
               static_cast<int>(plan.down_e.size()), kN, kK, nullptr);
  check(cudaDeviceSynchronize() == cudaSuccess, "moe_down_f32 launch");
  check(cudaMemcpy(h_y.data(), d_y, h_y.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) == cudaSuccess,
        "copy y");
  for (int p = 0; p < kPairs; ++p) {
    const std::vector<float> row(h_y.begin() + static_cast<size_t>(p) * kN,
                                 h_y.begin() + static_cast<size_t>(p + 1) * kN);
    report("grouped", p, rel_err(row, ref[p]), rel_err(row, ref_bf16[p]));
  }

  // ---- case 2: batch-1 GEMV path (decode shape) ---------------------------
  gemv_grouped_f32act(t.pair_weights + 2 * kPairs, d_act, d_y, kN, kK, kPairs,
                      nullptr);
  check(cudaDeviceSynchronize() == cudaSuccess, "gemv_grouped_f32act launch");
  check(cudaMemcpy(h_y.data(), d_y, h_y.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) == cudaSuccess,
        "copy y (gemv)");
  for (int p = 0; p < kPairs; ++p) {
    const std::vector<float> row(h_y.begin() + static_cast<size_t>(p) * kN,
                                 h_y.begin() + static_cast<size_t>(p + 1) * kN);
    report("gemv   ", p, rel_err(row, ref[p]), rel_err(row, ref_bf16[p]));
  }

  if (g_fail) {
    std::printf("RESULT: FAIL (%d failed checks)\n", g_fail);
    return 1;
  }
  std::printf("RESULT: PASS\n");
  return 0;
}
