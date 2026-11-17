// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace umminfer::qwen38 {

struct OmniTensorInfo {
  std::string dtype;
  std::vector<size_t> shape;
  size_t byte_offset = 0;  // relative to the shard data region
  size_t byte_size = 0;
  int shard = 0;
};

// Multi-shard safetensors reader. Shards are mmaped read-only; tensor bytes
// are served straight from the mapping, no whole-file copies. The checkpoint
// is 66 GiB across 15 shards and the host has ample virtual address space.
class ShardedSafetensors {
 public:
  // model_dir holds model.safetensors.index.json and model-NNNNN-of-*.safetensors.
  explicit ShardedSafetensors(const std::string& model_dir);
  ~ShardedSafetensors();
  ShardedSafetensors(const ShardedSafetensors&) = delete;
  ShardedSafetensors& operator=(const ShardedSafetensors&) = delete;

  const std::unordered_map<std::string, OmniTensorInfo>& tensors() const { return infos_; }
  const OmniTensorInfo& info(const std::string& name) const;

  // Pointer to a tensor's raw bytes in the mmaped shard.
  const void* tensor_bytes(const std::string& name) const;

  struct ShardMapping {
    int fd = -1;
    void* addr = nullptr;
    size_t length = 0;
    size_t data_region_offset = 0;
  };

 private:
  std::string dir_;
  std::vector<ShardMapping> shards_;
  std::unordered_map<std::string, OmniTensorInfo> infos_;
};

}  // namespace umminfer::qwen38
