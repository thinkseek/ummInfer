// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#ifdef __CUDACC__
#include <cuda_bf16.h>
#endif

namespace umminfer::qwen38 {

#ifdef __CUDACC__
// CUDA translation units use the native nv type (same 16-bit layout).
using Bf16 = __nv_bfloat16;
#else

// Brain float 16: upper 16 bits of an IEEE-754 float32. Host-only storage type.
struct Bf16 {
  uint16_t bits;

  Bf16() = default;
  explicit Bf16(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, sizeof(u));
    // Round-to-nearest-even, matching the bf16 truncation libraries use.
    u += 0x7fff + ((u >> 16) & 1);
    bits = static_cast<uint16_t>(u >> 16);
  }

  float to_float() const {
    uint32_t u = static_cast<uint32_t>(bits) << 16;
    float f;
    __builtin_memcpy(&f, &u, sizeof(f));
    return f;
  }
  operator float() const { return to_float(); }
};

#endif

}  // namespace umminfer::qwen38
