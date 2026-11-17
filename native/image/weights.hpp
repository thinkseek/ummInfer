// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "safetensors_shard.hpp"
#include "tensor.hpp"

#include <string>

namespace umminfer::image {

// Copies one named BF16 tensor from an mmap'd shard set onto the device.
// Throws if absent or not BF16.
DeviceTensor load_bf16(const ShardedSafeTensors& st, std::string_view name);

// Device pointers for a fixed set of per-layer weights, resolved once.
struct TextEncoderLayer {
  const Bf16 *q_w, *q_b, *k_w, *k_b, *v_w, *v_b, *o_w;
  const Bf16 *norm1, *norm2;
  const Bf16 *gate_w, *up_w, *down_w;
};

}  // namespace umminfer::image
