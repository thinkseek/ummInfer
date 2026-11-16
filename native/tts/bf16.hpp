// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace umminfer::native {

// Brain float 16: upper 16 bits of an IEEE-754 float32. Weight storage type.
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
};

}  // namespace umminfer::native
