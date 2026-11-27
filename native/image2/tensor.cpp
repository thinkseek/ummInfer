// SPDX-License-Identifier: Apache-2.0
#include "tensor.hpp"

#include "cuda_kernels.hpp"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cstdio>

namespace umminfer::image2 {

size_t product(const std::vector<int64_t>& shape) {
  size_t n = 1;
  for (int64_t d : shape) n *= static_cast<size_t>(d);
  return n;
}

DeviceTensor::DeviceTensor(std::vector<int64_t> dims) : shape(std::move(dims)) {
  numel = product(shape);
  IMG_CUDA_CHECK(request_cuda_malloc(&data, numel * sizeof(Bf16)));
}

DeviceTensor& DeviceTensor::operator=(DeviceTensor&& o) noexcept {
  if (this != &o) {
    if (data) request_cuda_free(data);
    shape = std::move(o.shape);
    data = o.data;
    numel = o.numel;
    o.data = nullptr;
    o.numel = 0;
  }
  return *this;
}

DeviceTensor::~DeviceTensor() {
  if (data) request_cuda_free(data);
}

void DeviceTensor::zeros_() {
  IMG_CUDA_CHECK(cudaMemsetAsync(data, 0, numel * sizeof(Bf16)));
}

void DeviceTensor::copy_from_host(const void* bf16_bytes) {
  IMG_CUDA_CHECK(cudaMemcpy(data, bf16_bytes, numel * sizeof(Bf16), cudaMemcpyHostToDevice));
}

std::vector<uint16_t> DeviceTensor::to_host_u16() const {
  std::vector<uint16_t> host(numel);
  IMG_CUDA_CHECK(cudaMemcpy(host.data(), data, numel * sizeof(uint16_t),
                           cudaMemcpyDeviceToHost));
  return host;
}

CudaRuntime::CudaRuntime(int gpu_id) : device(gpu_id) {
  IMG_CUDA_CHECK(cudaSetDevice(gpu_id));
  IMG_CUDA_CHECK(cudaStreamCreate(&stream));
  cublasLtHandle_t h;
  if (cublasLtCreate(&h) != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cublasLtCreate failed");
  cublas = h;
}

CudaRuntime::~CudaRuntime() {
  if (cublas) cublasLtDestroy(static_cast<cublasLtHandle_t>(cublas));
  if (stream) cudaStreamDestroy(stream);
}

void CudaRuntime::sync() const { IMG_CUDA_CHECK(cudaStreamSynchronize(stream)); }

// Row-major via cublasLt ROW order:
//   out[m,n] = in[m,k] * w[n,k]^T + bias[n], fp32 compute, BF16 I/O.
void linear_forward(CudaRuntime& rt, const Bf16* in, const Bf16* w, const Bf16* bias,
                    Bf16* out, int m, int n, int k) {
  cublasLtHandle_t lt = static_cast<cublasLtHandle_t>(rt.cublas);
  cublasLtMatmulDesc_t opDesc = nullptr;
  cublasLtMatrixLayout_t aDesc = nullptr, bDesc = nullptr, cDesc = nullptr;
  auto cleanup = [&] {
    if (opDesc) cublasLtMatmulDescDestroy(opDesc);
    if (aDesc) cublasLtMatrixLayoutDestroy(aDesc);
    if (bDesc) cublasLtMatrixLayoutDestroy(bDesc);
    if (cDesc) cublasLtMatrixLayoutDestroy(cDesc);
  };
  auto fail = [&](const char* what) {
    cleanup();
    throw std::runtime_error(std::string("cublasLt ") + what);
  };

  if (cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS)
    fail("matmul desc");
  cublasOperation_t opN = CUBLAS_OP_N, opT = CUBLAS_OP_T;
  cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
  cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opT, sizeof(opT));

  if (cublasLtMatrixLayoutCreate(&aDesc, CUDA_R_16BF, m, k, k) != CUBLAS_STATUS_SUCCESS)
    fail("A layout");  // in [m,k]
  if (cublasLtMatrixLayoutCreate(&bDesc, CUDA_R_16BF, n, k, k) != CUBLAS_STATUS_SUCCESS)
    fail("B layout");  // w [n,k], transposed
  if (cublasLtMatrixLayoutCreate(&cDesc, CUDA_R_16BF, m, n, n) != CUBLAS_STATUS_SUCCESS)
    fail("C layout");  // out [m,n]
  const cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;
  for (auto d : {aDesc, bDesc, cDesc})
    cublasLtMatrixLayoutSetAttribute(d, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder,
                                     sizeof(rowOrder));

  const float alpha = 1.0f, beta = 0.0f;
  cublasStatus_t st = cublasLtMatmul(lt, opDesc, &alpha, in, aDesc, w, bDesc, &beta, out,
                                     cDesc, out, cDesc, nullptr, nullptr, 0, rt.stream);
  cleanup();
  if (st != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cublasLt GEMM fail m=%d n=%d k=%d status=%d(%s)\n", m, n, k,
                 (int)st, cublasGetStatusString(st));
    throw std::runtime_error("cublasLt matmul failed");
  }
  if (bias) bias_broadcast(out, bias, m, n, rt.stream);
}

}  // namespace umminfer::image2
