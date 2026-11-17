// SPDX-License-Identifier: Apache-2.0
#include "weights.hpp"

#include <cstring>
#include <stdexcept>
#include <vector>

namespace umminfer::image2 {

DeviceTensor load_bf16(const ShardedSafeTensors& st, std::string_view name) {
  const StTensor& info = st.info(name);
  DeviceTensor t(std::vector<int64_t>(info.shape.begin(), info.shape.end()));
  if (info.dtype == "BF16") {
    t.copy_from_host(st.host_ptr(info));
    return t;
  }
  // The VAE checkpoint ships fp32 and diffusers casts it to bf16 on load.
  std::vector<uint16_t> dst(t.numel);
  if (info.dtype == "F32") {
    const float* src = static_cast<const float*>(st.host_ptr(info));
    for (size_t i = 0; i < t.numel; ++i) dst[i] = bf16_to_bits(src[i]);
  } else if (info.dtype == "F16") {
    const uint16_t* src = static_cast<const uint16_t*>(st.host_ptr(info));
    for (size_t i = 0; i < t.numel; ++i) {
      const uint32_t sign = (src[i] & 0x8000u) << 16;
      uint32_t exp = (src[i] >> 10) & 0x1fu;
      uint32_t mant = src[i] & 0x3ffu;
      float f;
      uint32_t bits;
      if (exp == 0) {
        // subnormal: value = mant * 2^-24
        const float v = static_cast<float>(mant) * 5.960464477539063e-08f;
        f = (sign ? -v : v);
      } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
        std::memcpy(&f, &bits, 4);
      } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
        std::memcpy(&f, &bits, 4);
      }
      dst[i] = bf16_to_bits(f);
    }
  } else {
    throw std::invalid_argument("expected BF16/F16/F32 tensor: " + std::string(name));
  }
  t.copy_from_host(dst.data());
  return t;
}

}  // namespace umminfer::image2
