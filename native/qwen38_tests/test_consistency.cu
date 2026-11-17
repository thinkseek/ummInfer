// SPDX-License-Identifier: Apache-2.0
// Prefill (one call, m rows) and decode (m = 1 per call) must produce the same
// linear-attention outputs and the same cached state.  This pins the conv
// state and the fp32 recurrent state across both paths.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "qwen38/qwen38_ops.hpp"

using namespace umminfer::qwen38;

namespace {
Bf16* dev(const std::vector<Bf16>& h) {
  Bf16* d = nullptr;
  cudaMalloc(&d, h.size() * sizeof(Bf16));
  cudaMemcpy(d, h.data(), h.size() * sizeof(Bf16), cudaMemcpyHostToDevice);
  return d;
}
float* devf(const std::vector<float>& h) {
  float* d = nullptr;
  cudaMalloc(&d, h.size() * sizeof(float));
  cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
  return d;
}
unsigned bits_of(Bf16 v) {
  unsigned u = 0;
  std::memcpy(&u, &v, sizeof(u));
  return u & 0xffffu;
}
bool same_bf16(const std::vector<Bf16>& a, const std::vector<Bf16>& b, const char* what) {
  for (size_t i = 0; i < a.size(); ++i)
    if (bits_of(a[i]) != bits_of(b[i])) {
      std::printf("%s differs at %zu: %04x vs %04x\n", what, i, bits_of(a[i]),
                  bits_of(b[i]));
      return false;
    }
  return true;
}
}  // namespace

int main() {
  const int m = 8, channels = 16;
  std::vector<Bf16> x(static_cast<size_t>(m) * channels), w(static_cast<size_t>(channels) * 4);
  unsigned seed = 7;
  auto rnd = [&]() {
    seed = seed * 1664525u + 1013904223u;
    return (static_cast<float>(seed >> 8) / 16777216.0f) - 0.5f;
  };
  for (auto& v : x) v = Bf16(rnd());
  for (auto& v : w) v = Bf16(rnd());

  Bf16 *dx = dev(x), *dw = dev(w);
  Bf16 *state_a, *state_b, *out_a, *out_b;
  cudaMalloc(&state_a, channels * 3 * sizeof(Bf16));
  cudaMalloc(&state_b, channels * 3 * sizeof(Bf16));
  cudaMalloc(&out_a, x.size() * sizeof(Bf16));
  cudaMalloc(&out_b, x.size() * sizeof(Bf16));
  cudaMemset(state_a, 0, channels * 3 * sizeof(Bf16));
  cudaMemset(state_b, 0, channels * 3 * sizeof(Bf16));

  conv1d_causal_silu(dx, dw, state_a, out_a, m, channels, 0);
  for (int t = 0; t < m; ++t)
    conv1d_causal_silu(dx + static_cast<size_t>(t) * channels, dw, state_b,
                       out_b + static_cast<size_t>(t) * channels, 1, channels, 0);
  cudaDeviceSynchronize();
  std::vector<Bf16> ha(x.size()), hb(x.size()), sa(channels * 3), sb(channels * 3);
  cudaMemcpy(ha.data(), out_a, ha.size() * sizeof(Bf16), cudaMemcpyDeviceToHost);
  cudaMemcpy(hb.data(), out_b, hb.size() * sizeof(Bf16), cudaMemcpyDeviceToHost);
  cudaMemcpy(sa.data(), state_a, sa.size() * sizeof(Bf16), cudaMemcpyDeviceToHost);
  cudaMemcpy(sb.data(), state_b, sb.size() * sizeof(Bf16), cudaMemcpyDeviceToHost);
  if (!same_bf16(ha, hb, "conv output") || !same_bf16(sa, sb, "conv state")) {
    std::printf("FAIL: conv prefill/decode mismatch\n");
    return 1;
  }
  std::printf("conv prefill == decode (bit-exact), state layout %d x 3\n", channels);

  // Gated delta rule: one 8-row call vs eight 1-row calls, fp32 state.
  const int kh = 2, vh = 4, d = 16;
  std::vector<float> qn(static_cast<size_t>(m) * kh * d), kn(qn.size());
  std::vector<Bf16> v(static_cast<size_t>(m) * vh * d);
  std::vector<float> beta(static_cast<size_t>(m) * vh), g(beta.size());
  for (auto& t : qn) t = rnd();
  for (auto& t : kn) t = rnd();
  for (auto& t : v) t = Bf16(rnd());
  for (auto& t : beta) t = 0.1f + 0.8f * std::fabs(rnd());
  for (auto& t : g) t = -std::fabs(rnd());

  float* dqn = devf(qn);
  float* dkn = devf(kn);
  Bf16* dv = dev(v);
  float* dbeta = devf(beta);
  float* dg = devf(g);
  float *st_a, *st_b;
  Bf16 *o_a, *o_b;
  cudaMalloc(&st_a, static_cast<size_t>(vh) * d * d * sizeof(float));
  cudaMalloc(&st_b, static_cast<size_t>(vh) * d * d * sizeof(float));
  cudaMalloc(&o_a, v.size() * sizeof(Bf16));
  cudaMalloc(&o_b, v.size() * sizeof(Bf16));
  cudaMemset(st_a, 0, static_cast<size_t>(vh) * d * d * sizeof(float));
  cudaMemset(st_b, 0, static_cast<size_t>(vh) * d * d * sizeof(float));

  gated_delta_rule(st_a, dqn, dkn, dv, dbeta, dg, o_a, m, kh, vh, d, 0);
  for (int t = 0; t < m; ++t)
    gated_delta_rule(st_b, dqn + static_cast<size_t>(t) * kh * d,
                     dkn + static_cast<size_t>(t) * kh * d,
                     dv + static_cast<size_t>(t) * vh * d,
                     dbeta + static_cast<size_t>(t) * vh,
                     dg + static_cast<size_t>(t) * vh,
                     o_b + static_cast<size_t>(t) * vh * d, 1, kh, vh, d, 0);
  cudaDeviceSynchronize();
  std::vector<Bf16> vo_a(v.size()), vo_b(v.size());
  std::vector<float> fs_a(static_cast<size_t>(vh) * d * d), fs_b(fs_a.size());
  cudaMemcpy(vo_a.data(), o_a, vo_a.size() * sizeof(Bf16), cudaMemcpyDeviceToHost);
  cudaMemcpy(vo_b.data(), o_b, vo_b.size() * sizeof(Bf16), cudaMemcpyDeviceToHost);
  cudaMemcpy(fs_a.data(), st_a, fs_a.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(fs_b.data(), st_b, fs_b.size() * sizeof(float), cudaMemcpyDeviceToHost);
  bool state_equal = true;
  for (size_t i = 0; i < fs_a.size(); ++i)
    if (fs_a[i] != fs_b[i]) {
      state_equal = false;
      std::printf("state differs at %zu: %.9g vs %.9g\n", i, fs_a[i], fs_b[i]);
      break;
    }
  if (!same_bf16(vo_a, vo_b, "delta output") || !state_equal) {
    std::printf("FAIL: delta-rule prefill/decode mismatch\n");
    return 1;
  }
  std::printf("delta-rule prefill == decode (bit-exact), state fp32 %d x %d x %d\n",
              vh, d, d);
  std::printf("consistency tests passed\n");
  return 0;
}
