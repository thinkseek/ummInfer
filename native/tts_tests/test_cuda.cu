// SPDX-License-Identifier: Apache-2.0
// CUDA kernel smoke tests with hand-computed values. No checkpoint required.
#include "cuda_kernels.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace umminfer::native;

#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::cerr << "CHECK failed: " #cond " at " << __FILE__ << ":"       \
                << __LINE__ << "\n";                                     \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

static bool close(float a, float b, float tol = 1e-4f) {
  return std::fabs(a - b) <= tol * (1.0f + std::fabs(b));
}

int main() {
  cudaStream_t st;
  UMM_CUDA_CHECK(cudaStreamCreate(&st));

  // RMSNorm over one 4-vector with all-ones weight.
  {
    const float x[4] = {1, 2, 2, 1};
    float *dx, *dw, *dy;
    cudaMalloc(&dx, 16);
    cudaMalloc(&dw, 16);
    cudaMalloc(&dy, 16);
    const float w[4] = {1, 1, 1, 1};
    cudaMemcpy(dx, x, 16, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w, 16, cudaMemcpyHostToDevice);
    rms_norm_t<float>(dx, dw, dy, 1, 4, 1e-5f, st);
    float y[4];
    cudaMemcpy(y, dy, 16, cudaMemcpyDeviceToHost);
    const float rms = std::sqrt((1.f + 4 + 4 + 1) / 4);  // sqrt(2.5)
    for (int i = 0; i < 4; ++i) CHECK(close(y[i], x[i] / rms, 1e-5f));
    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dy);
  }

  // GEMM: identity-like W=I(2), x = [1,2] -> [1,2].
  {
    const float x[2] = {1, 2};
    const float w[4] = {1, 0, 0, 1};
    float *dx, *dw, *dy;
    cudaMalloc(&dx, 8);
    cudaMalloc(&dw, 16);
    cudaMalloc(&dy, 8);
    cudaMemcpy(dx, x, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w, 16, cudaMemcpyHostToDevice);
    gemm_t<float>(dy, dx, dw, nullptr, 1, 2, 2, st);
    float y[2];
    cudaMemcpy(y, dy, 8, cudaMemcpyDeviceToHost);
    CHECK(close(y[0], 1) && close(y[1], 2));
    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dy);
  }

  // Causal conv1d k3, single channel, left pad 2: out[t] = x[t]+x[t-1]+x[t-2].
  {
    const int L = 4;
    const float x[L] = {1, 2, 3, 4};
    const float w[3] = {1, 1, 1};
    const float b[1] = {0};
    float *dx, *dw, *db, *dy;
    cudaMalloc(&dx, L * 4);
    cudaMalloc(&dw, 12);
    cudaMalloc(&db, 4);
    cudaMalloc(&dy, L * 4);
    cudaMemcpy(dx, x, L * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w, 12, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, 4, cudaMemcpyHostToDevice);
    conv1d_causal_f32(dx, dw, db, dy, 1, 1, L, 3, 1, 1, st);
    float y[L];
    cudaMemcpy(y, dy, L * 4, cudaMemcpyDeviceToHost);
    CHECK(close(y[0], 1) && close(y[1], 3) && close(y[2], 6) && close(y[3], 9));
    cudaFree(dx);
    cudaFree(dw);
    cudaFree(db);
    cudaFree(dy);
  }

  // ConvTranspose1d k2 stride2 doubles length: x [a,b] -> [a,0,b,0] for the
  // diagonal weight w=I, bias 0, no trim.
  {
    const int L = 2;
    const float x[L] = {3, 5};
    const float w[4] = {1, 0, 0, 1};  // [in1,out1,k2]
    const float b[1] = {0};
    float *dx, *dw, *db, *dy;
    cudaMalloc(&dx, L * 4);
    cudaMalloc(&dw, 16);
    cudaMalloc(&db, 4);
    cudaMalloc(&dy, L * 2 * 4);
    cudaMemcpy(dx, x, L * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w, 16, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, 4, cudaMemcpyHostToDevice);
    conv_transpose1d_f32(dx, dw, db, dy, 1, 1, L, 2, 2, st);
    float y[4];
    cudaMemcpy(y, dy, 16, cudaMemcpyDeviceToHost);
    CHECK(close(y[0], 3) && close(y[1], 0) && close(y[2], 5) && close(y[3], 0));
    cudaFree(dx);
    cudaFree(dw);
    cudaFree(db);
    cudaFree(dy);
  }

  // argmax and suppress.
  {
    const float x[5] = {0.1f, 2, 0.5f, 9, -1};
    float* dx;
    int* di;
    cudaMalloc(&dx, 20);
    cudaMalloc(&di, 4);
    cudaMemcpy(dx, x, 20, cudaMemcpyHostToDevice);
    CHECK(argmax_f32(dx, 5, di, st) == 3);
    suppress_logits(dx, 3, 5, -1, st);
    float y[5];
    cudaMemcpy(y, dx, 20, cudaMemcpyDeviceToHost);
    CHECK(y[3] == -INFINITY && y[4] == -INFINITY && y[2] == 0.5f &&
          y[0] == 0.1f);
    cudaFree(dx);
    cudaFree(di);
  }

  // repetition penalty: positive logit divided, negative multiplied.
  {
    const float x[4] = {2.0f, -4.0f, 0.5f, 1.0f};
    const int hist[2] = {0, 1};
    float* dx;
    int* dh;
    cudaMalloc(&dx, 16);
    cudaMalloc(&dh, 8);
    cudaMemcpy(dx, x, 16, cudaMemcpyHostToDevice);
    cudaMemcpy(dh, hist, 8, cudaMemcpyHostToDevice);
    repetition_penalty_f32(dx, dh, 2, 2.0f, st);
    float y[4];
    cudaMemcpy(y, dx, 16, cudaMemcpyDeviceToHost);
    CHECK(fabsf(y[0] - 1.0f) < 1e-6f && fabsf(y[1] + 8.0f) < 1e-6f &&
          y[2] == 0.5f && y[3] == 1.0f);
    cudaFree(dx);
    cudaFree(dh);
  }

  cudaStreamDestroy(st);
  std::cout << "all tts cuda tests passed\n";
  return 0;
}
