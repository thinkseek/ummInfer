// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "safetensors_shard.hpp"
#include "tensor.hpp"

#include <string>

namespace umminfer::image2 {

// Copies one named BF16 tensor from an mmap'd shard set onto the device.
// Throws if absent or not BF16.
DeviceTensor load_bf16(const ShardedSafeTensors& st, std::string_view name);

}  // namespace umminfer::image2
