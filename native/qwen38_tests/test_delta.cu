// SPDX-License-Identifier: Apache-2.0
// GPU check for the two riskiest linear-attention kernels: the depthwise
// causal conv and the recurrent gated delta rule, against a CPU reference that
// follows the same recurrence as transformers (fp32).
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "qwen38/qwen38_ops.hpp"

using namespace umminfer::qwen38;

namespace {

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

std::vector<float> cpu_delta_rule(const std::vector<float>& qn,
                                  const std::vector<float>& kn,
                                  const std::vector<float>& v,
                                  const std::vector<float>& beta,
                                  const std::vector<float>& g, int m, int kh,
                                  int vh, int d) {
  std::vector<float> state(static_cast<size_t>(vh) * d * d, 0.0f);
  std::vector<float> out(static_cast<size_t>(m) * vh * d, 0.0f);
  const int group = vh / kh;
  for (int t = 0; t < m; ++t) {
    for (int h = 0; h < vh; ++h) {
      const int k_head = h / group;
      float* S = state.data() + static_cast<size_t>(h) * d * d;
      const float decay = std::exp(g[static_cast<size_t>(t) * vh + h]);
      const float bt = beta[static_cast<size_t>(t) * vh + h];
      for (int k = 0; k < d; ++k)
        for (int j = 0; j < d; ++j) S[k * d + j] *= decay;
      std::vector<float> delta(d, 0.0f);
      for (int j = 0; j < d; ++j) {
        float mem = 0.0f;
        for (int k = 0; k < d; ++k)
          mem += S[k * d + j] *
                 kn[(static_cast<size_t>(t) * kh + k_head) * d + k];
        delta[j] = (v[(static_cast<size_t>(t) * vh + h) * d + j] - mem) * bt;
      }
      for (int k = 0; k < d; ++k) {
        const float kk = kn[(static_cast<size_t>(t) * kh + k_head) * d + k];
        for (int j = 0; j < d; ++j) S[k * d + j] += kk * delta[j];
      }
      for (int j = 0; j < d; ++j) {
        float acc = 0.0f;
        for (int k = 0; k < d; ++k)
          acc += S[k * d + j] * qn[(static_cast<size_t>(t) * kh + k_head) * d + k];
        out[(static_cast<size_t>(t) * vh + h) * d + j] = acc;
      }
    }
  }
  return out;
}

}  // namespace

int main() {
  const int m = 5, kh = 2, vh = 4, d = 8;
  std::vector<float> qn(static_cast<size_t>(m) * kh * d);
  std::vector<float> kn(qn.size());
  std::vector<Bf16> v(static_cast<size_t>(m) * vh * d);
  std::vector<float> beta(static_cast<size_t>(m) * vh), g(beta.size());
  unsigned seed = 1;
  auto rnd = [&]() {
    seed = seed * 1664525u + 1013904223u;
    return (static_cast<float>(seed >> 8) / 16777216.0f) - 0.5f;
  };
  for (auto& x : qn) x = rnd();
  for (auto& x : kn) x = rnd();
  std::vector<float> vf(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    vf[i] = rnd();
    v[i] = Bf16(vf[i]);
    vf[i] = float(v[i]);
  }
  for (int t = 0; t < m; ++t)
    for (int h = 0; h < vh; ++h) {
      beta[t * vh + h] = sigmoid(rnd() * 4.0f);
      g[t * vh + h] = -std::exp(rnd()) * std::log1p(std::exp(rnd() * 3.0f));
    }

  const std::vector<float> expect = cpu_delta_rule(qn, kn, vf, beta, g, m, kh, vh, d);

  float *dqn, *dkn, *dbeta, *dg, *dstate;
  Bf16 *dv, *dout;
  cudaMalloc(&dqn, qn.size() * sizeof(float));
  cudaMalloc(&dkn, kn.size() * sizeof(float));
  cudaMalloc(&dv, v.size() * sizeof(Bf16));
  cudaMalloc(&dbeta, beta.size() * sizeof(float));
  cudaMalloc(&dg, g.size() * sizeof(float));
  cudaMalloc(&dstate, static_cast<size_t>(vh) * d * d * sizeof(float));
  cudaMalloc(&dout, v.size() * sizeof(Bf16));
  cudaMemcpy(dqn, qn.data(), qn.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dkn, kn.data(), kn.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dv, v.data(), v.size() * sizeof(Bf16), cudaMemcpyHostToDevice);
  cudaMemcpy(dbeta, beta.data(), beta.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dg, g.data(), g.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(dstate, 0, static_cast<size_t>(vh) * d * d * sizeof(float));

  gated_delta_rule(dstate, dqn, dkn, dv, dbeta, dg, dout, m, kh, vh, d, 0);
  cudaDeviceSynchronize();
  std::vector<Bf16> got(v.size());
  cudaMemcpy(got.data(), dout, v.size() * sizeof(Bf16), cudaMemcpyDeviceToHost);

  double max_rel = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double e = expect[i];
    const double a = float(got[i]);
    const double rel = std::fabs(a - e) / (std::fabs(e) + 1e-3);
    if (rel > max_rel) max_rel = rel;
  }
  std::printf("delta rule max relative error %.3e\n", max_rel);
  // The kernel output is bf16 by contract, so ~0.4% rounding is expected.
  if (max_rel > 1.5e-2) {
    std::printf("FAIL: gated delta rule diverges from the reference recurrence\n");
    return 1;
  }

  // Depthwise causal conv with kernel 4: out[t] = silu(sum_j w[j] x[t-3+j]).
  const int mm = 3, ch = 4;
  std::vector<Bf16> x(static_cast<size_t>(mm) * ch), w(static_cast<size_t>(ch) * 4),
      state(static_cast<size_t>(ch) * 3, Bf16(0.0f)), out(x.size());
  for (int t = 0; t < mm; ++t)
    for (int c = 0; c < ch; ++c) x[t * ch + c] = Bf16(0.1f * (t + 1) + 0.01f * c);
  for (size_t i = 0; i < w.size(); ++i) w[i] = Bf16(0.05f * (i + 1));
  Bf16 *dx, *dw, *dstate2, *dout2;
  cudaMalloc(&dx, x.size() * sizeof(Bf16));
  cudaMalloc(&dw, w.size() * sizeof(Bf16));
  cudaMalloc(&dstate2, state.size() * sizeof(Bf16));
  cudaMalloc(&dout2, out.size() * sizeof(Bf16));
  cudaMemcpy(dx, x.data(), x.size() * sizeof(Bf16), cudaMemcpyHostToDevice);
  cudaMemcpy(dw, w.data(), w.size() * sizeof(Bf16), cudaMemcpyHostToDevice);
  cudaMemcpy(dstate2, state.data(), state.size() * sizeof(Bf16),
             cudaMemcpyHostToDevice);
  conv1d_causal_silu(dx, dw, dstate2, dout2, mm, ch, 0);
  cudaDeviceSynchronize();
  cudaMemcpy(out.data(), dout2, out.size() * sizeof(Bf16), cudaMemcpyDeviceToHost);

  double conv_err = 0.0;
  for (int t = 0; t < mm; ++t)
    for (int c = 0; c < ch; ++c) {
      float acc = 0.0f;
      for (int j = 0; j < 4; ++j) {
        const int idx = t - 3 + j;
        const float xv = idx >= 0 ? float(x[idx * ch + c]) : 0.0f;
        acc += float(w[c * 4 + j]) * xv;
      }
      const float e = acc / (1.0f + std::exp(-acc));
      conv_err = std::fmax(conv_err, std::fabs(e - float(out[t * ch + c])));
    }
  std::printf("conv max abs error %.3e\n", conv_err);
  if (conv_err > 1e-3f) {
    std::printf("FAIL: causal conv does not match the definition\n");
    return 1;
  }

  std::printf("gpu kernel tests passed\n");
  return 0;
}
