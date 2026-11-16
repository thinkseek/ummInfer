// SPDX-License-Identifier: Apache-2.0
// Known-answer test for gqa_causal_attn: small random tensors, brute-force
// fp32 causal GQA reference computed on the host. Removes torch from the loop.
#include "golden_io.hpp"
#include "image/cuda_kernels.hpp"
#include "image/tensor.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

int main() {
  using namespace umminfer::image;
  CudaRuntime rt(0);
  const int sq = 55, hq = 28, hkv = 4, D = 128, group = hq / hkv;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 0.3f);
  std::vector<uint16_t> qb((size_t)sq * hq * D), kb((size_t)sq * hkv * D),
      vb((size_t)sq * hkv * D);
  for (auto& x : qb) x = bf16_to_bits(nd(rng));
  for (auto& x : kb) x = bf16_to_bits(nd(rng));
  for (auto& x : vb) x = bf16_to_bits(nd(rng));
  DeviceTensor q({sq, hq, D}), k({sq, hkv, D}), v({sq, hkv, D}),
      out({sq, hq, D});
  q.copy_from_host(qb.data()); k.copy_from_host(kb.data()); v.copy_from_host(vb.data());
  gqa_causal_attn(q.data, k.data, v.data, out.data, 1, hq, hkv, sq, sq, D, rt.stream);
  rt.sync();
  auto got = out.to_host_u16();

  float worst = 0.0f;
  for (int qi = 0; qi < sq; ++qi)
    for (int head = 0; head < hq; ++head) {
      int kvh = head / group;
      std::vector<float> logits(qi + 1);
      float mx = -1e30f;
      for (int j = 0; j <= qi; ++j) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d)
          dot += ummtest::bf_to_f(qb[((size_t)qi * hq + head) * D + d]) *
                 ummtest::bf_to_f(kb[((size_t)j * hkv + kvh) * D + d]);
        logits[j] = dot / std::sqrt(float(D));
        mx = std::max(mx, logits[j]);
      }
      float sum = 0.0f;
      for (int j = 0; j <= qi; ++j) { logits[j] = std::exp(logits[j] - mx); sum += logits[j]; }
      for (int d = 0; d < D; ++d) {
        float ref = 0.0f;
        for (int j = 0; j <= qi; ++j)
          ref += (logits[j] / sum) *
                 ummtest::bf_to_f(vb[((size_t)j * hkv + kvh) * D + d]);
        float g = ummtest::bf_to_f(got[((size_t)qi * hq + head) * D + d]);
        worst = std::max(worst, std::fabs(g - ref));
      }
    }
  std::printf("gqa_causal_attn host-ref max_abs=%e\n", worst);
  return worst < 0.05f ? 0 : 1;
}
