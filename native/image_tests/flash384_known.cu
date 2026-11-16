// Host-reference test for the D=384 single-head non-causal flash path (VAE).
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
  const int s=64, heads=1, D=384;  // VAE mid-attention geometry, multi-tile (BLOCK_K=8)
  std::mt19937 rng(7); std::normal_distribution<float> nd(0,0.2f);
  std::vector<uint16_t> qb((size_t)s*heads*D),kb(qb.size()),vb(qb.size());
  for(auto&x:qb)x=bf16_to_bits(nd(rng));
  for(auto&x:kb)x=bf16_to_bits(nd(rng));
  for(auto&x:vb)x=bf16_to_bits(nd(rng));
  DeviceTensor q({s,heads,D}),k({s,heads,D}),v({s,heads,D}),o({s,heads,D});
  q.copy_from_host(qb.data());k.copy_from_host(kb.data());v.copy_from_host(vb.data());
  flash_attn(q.data,k.data,v.data,o.data,1,heads,s,s,D,rt.stream);
  rt.sync();
  auto got=o.to_host_u16();
  float worst=0;
  for(int qi=0;qi<s;++qi)for(int head=0;head<heads;++head){
    std::vector<float> logits(s); float mx=-1e30f;
    for(int j=0;j<s;++j){float dot=0;for(int d=0;d<D;++d)
      dot+=ummtest::bf_to_f(qb[((size_t)qi*heads+head)*D+d])*ummtest::bf_to_f(kb[((size_t)j*heads+head)*D+d]);
      logits[j]=dot/std::sqrt((float)D); mx=std::fmax(mx,logits[j]);}
    float sum=0; for(float&z:logits){z=std::exp(z-mx);sum+=z;}
    for(int d=0;d<D;++d){float ref=0;for(int j=0;j<s;++j)
      ref+=(logits[j]/sum)*ummtest::bf_to_f(vb[((size_t)j*heads+head)*D+d]);
      worst=std::fmax(worst,std::fabs(ref-ummtest::bf_to_f(got[((size_t)qi*heads+head)*D+d])));}
  }
  std::printf("flash_attn D=384 host-ref max_abs=%e\n",worst);
  return worst<0.05f?0:1;
}
