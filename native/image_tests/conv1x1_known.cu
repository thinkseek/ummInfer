// Isolate conv2d_1x1 NCHW against a host reference.
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
  const int n=1, ci=384, co=384, hw=64;  // VAE resnet-like
  std::mt19937 rng(1); std::normal_distribution<float> nd(0,0.1f);
  std::vector<uint16_t> xb((size_t)n*ci*hw), wb((size_t)co*ci), bb(co), ob((size_t)n*co*hw);
  for(auto&x:xb)x=bf16_to_bits(nd(rng));
  for(auto&x:wb)x=bf16_to_bits(nd(rng)*0.05f);
  for(auto&x:bb)x=bf16_to_bits(nd(rng)*0.05f);
  DeviceTensor x({n,ci,hw}),w({co,ci}),b({co}),o({n,co,hw});
  x.copy_from_host(xb.data());w.copy_from_host(wb.data());b.copy_from_host(bb.data());
  conv2d_1x1(x.data,w.data,b.data,o.data,n,ci,co,hw,rt.stream);
  rt.sync();
  auto got=o.to_host_u16();
  float worst=0;
  for(int oc=0;oc<co;++oc)for(int p=0;p<hw;++p){
    float acc=ummtest::bf_to_f(bb[oc]);
    for(int c=0;c<ci;++c) acc+=ummtest::bf_to_f(xb[(size_t)c*hw+p])*ummtest::bf_to_f(wb[(size_t)oc*ci+c]);
    float g=ummtest::bf_to_f(got[(size_t)oc*hw+p]);
    worst=std::fmax(worst,std::fabs(g-acc));
  }
  std::printf("conv2d_1x1 NCHW host-ref max_abs=%e\n",worst);
  return worst<0.05f?0:1;
}
