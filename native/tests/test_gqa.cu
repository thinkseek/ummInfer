// GPU unit test for the GQA attention kernel against a manual fp32 reference.
// Covers both full prefill (causal over S tokens) and one decode step reading
// a populated KV cache. Built only inside the CUDA tree on the H20B host.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../umm_asr_native/cuda_kernels.hpp"

using umminfer::native::Bf16;
using umminfer::native::gqa_attention;

namespace {
int failures = 0;
void check(bool c, const char* w) {
  if (!c) { printf("FAIL: %s\n", w); ++failures; }
}

std::vector<Bf16> to_bf(const std::vector<float>& x) {
  std::vector<Bf16> o(x.size());
  for (size_t i = 0; i < x.size(); ++i) o[i] = __float2bfloat16(x[i]);
  return o;
}

// Reference causal attention for one query position qi with GQA.
float ref_attn(const std::vector<float>& q, const std::vector<float>& k,
               const std::vector<float>& v, int qi, int nqh, int nkh, int d,
               int out_head, int coord) {
  const int kvh = out_head * nkh / nqh;
  float mx = -1e30f;
  std::vector<float> scores(qi + 1);
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));
  for (int j = 0; j <= qi; ++j) {
    float dot = 0;
    for (int i = 0; i < d; ++i)
      dot += q[(qi * nqh + out_head) * d + i] * k[(j * nkh + kvh) * d + i];
    scores[j] = dot * scale;
    mx = std::max(mx, scores[j]);
  }
  float sum = 0;
  for (float& s : scores) { s = std::exp(s - mx); sum += s; }
  float acc = 0;
  for (int j = 0; j <= qi; ++j)
    acc += (scores[j] / sum) * v[(j * nkh + kvh) * d + coord];
  return acc;
}
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const int S = 8, nqh = 4, nkh = 2, d = 64;
  std::vector<float> hq(S * nqh * d), hk(S * nkh * d), hv(S * nkh * d);
  srand(7);
  for (auto& x : hq) x = (rand() % 200 - 100) / 100.0f;
  for (auto& x : hk) x = (rand() % 200 - 100) / 100.0f;
  for (auto& x : hv) x = (rand() % 200 - 100) / 100.0f;
  auto dq = to_bf(hq), dk = to_bf(hk), dv = to_bf(hv);
  Bf16 *cq, *ck, *cv, *co;
  cudaMalloc(&cq, dq.size() * 2); cudaMalloc(&ck, dk.size() * 2);
  cudaMalloc(&cv, dv.size() * 2); cudaMalloc(&co, dq.size() * 2);
  cudaMemcpy(cq, dq.data(), dq.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(ck, dk.data(), dk.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(cv, dv.data(), dv.size() * 2, cudaMemcpyHostToDevice);

  // Prefill: all S query positions, causal.
  gqa_attention(cq, ck, cv, co, S, S, nqh, nkh, d, true, 0, 0, 0);
  cudaError_t e = cudaDeviceSynchronize();
  check(e == cudaSuccess, "prefill kernel launches cleanly");
  if (e != cudaSuccess) { printf("prefill cuda err %d\n", (int)e); return 1; }
  std::vector<Bf16> ho(dq.size());
  cudaMemcpy(ho.data(), co, ho.size() * 2, cudaMemcpyDeviceToHost);
  double maxerr = 0;
  for (int qi = 0; qi < S; ++qi)
    for (int h = 0; h < nqh; ++h)
      for (int i = 0; i < d; ++i) {
        float want = ref_attn(hq, hk, hv, qi, nqh, nkh, d, h, i);
        maxerr = std::max(maxerr, static_cast<double>(std::fabs(
                                      want - __bfloat162float(
                                                 ho[(qi * nqh + h) * d + i]))));
      }
  check(maxerr < 0.05, "prefill causal GQA matches reference");
  printf("prefill max abs err %.5f\n", maxerr);

  // Decode: one new query at position S against S cached keys.
  std::vector<float> nq(nqh * d), nk(nkh * d), nv(nkh * d);
  for (auto& x : nq) x = (rand() % 200 - 100) / 100.0f;
  for (auto& x : nk) x = (rand() % 200 - 100) / 100.0f;
  for (auto& x : nv) x = (rand() % 200 - 100) / 100.0f;
  // Extend cache: [cached S ; new 1]. fq needs room for S+1 query tokens.
  std::vector<float> fk = hk, fv = hv, fq((S + 1) * nqh * d, 0);
  fk.insert(fk.end(), nk.begin(), nk.end());
  fv.insert(fv.end(), nv.begin(), nv.end());
  for (int h = 0; h < nqh; ++h)
    for (int i = 0; i < d; ++i) fq[(S * nqh + h) * d + i] = nq[h * d + i];
  auto bq = to_bf(fq), bk = to_bf(fk), bv = to_bf(fv);
  Bf16 *q2, *k2, *v2, *o2;
  cudaMalloc(&q2, bq.size() * 2); cudaMalloc(&k2, bk.size() * 2);
  cudaMalloc(&v2, bv.size() * 2); cudaMalloc(&o2, nq.size() * 2);
  cudaMemcpy(q2, bq.data(), bq.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(k2, bk.data(), bk.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(v2, bv.data(), bv.size() * 2, cudaMemcpyHostToDevice);
  // Query pointer starts at the last token; nkv = S+1, q_offset = S.
  gqa_attention(q2 + S * nqh * d, k2, v2, o2, 1, S + 1, nqh, nkh, d, true, 0, S, 0);
  e = cudaDeviceSynchronize();
  check(e == cudaSuccess, "decode kernel launches cleanly");
  if (e != cudaSuccess) { printf("decode cuda err %d\n", (int)e); return 1; }
  std::vector<Bf16> oo(nq.size());
  cudaMemcpy(oo.data(), o2, oo.size() * 2, cudaMemcpyDeviceToHost);
  maxerr = 0;
  for (int h = 0; h < nqh; ++h)
    for (int i = 0; i < d; ++i) {
      float want = ref_attn(fq, fk, fv, S, nqh, nkh, d, h, i);
      maxerr = std::max(maxerr, static_cast<double>(std::fabs(
                                    want - __bfloat162float(oo[h * d + i]))));
    }
  check(maxerr < 0.05, "decode GQA with KV cache matches reference");
  printf("decode max abs err %.5f\n", maxerr);

  if (failures) { printf("%d failure(s)\n", failures); return 1; }
  printf("all GQA checks passed\n");
  return 0;
}
