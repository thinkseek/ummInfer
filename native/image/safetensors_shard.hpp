// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace umminfer::image {

struct StTensor {
  std::string dtype;          // "BF16" for every Qwen-Image-2512 tensor
  std::vector<size_t> shape;  // row-major
  int shard = 0;              // index into ShardedSafeTensors::shards()
  size_t byte_offset = 0;     // relative to the shard's data region
  size_t byte_size = 0;
};

struct MmapShard {
  std::string path;
  int fd = -1;
  size_t mapped_size = 0;
  const uint8_t* data = nullptr;  // mmap base
  size_t data_region = 0;         // absolute offset of tensor bytes within the mapping
};

// Multi-shard safetensors model directory. Shard files are mmap'd; only the
// JSON headers are parsed into memory, tensor bytes stay page-cached and are
// copied straight from the mapping onto the device. A component may have an
// index.json (weight_map) or be a single standalone .safetensors file (VAE).
class ShardedSafeTensors {
 public:
  explicit ShardedSafeTensors(const std::string& component_dir);
  ~ShardedSafeTensors();
  ShardedSafeTensors(const ShardedSafeTensors&) = delete;
  ShardedSafeTensors& operator=(const ShardedSafeTensors&) = delete;

  bool contains(std::string_view name) const { return tensors_.count(std::string(name)) != 0; }
  const StTensor& info(std::string_view name) const;
  // Host pointer to a tensor's first byte inside the shard mapping.
  const void* host_ptr(const StTensor& t) const;
  const std::vector<MmapShard>& shards() const { return shards_; }
  size_t size() const { return tensors_.size(); }

  // Element count from a shape.
  static size_t numel(const std::vector<size_t>& shape);

 private:
  void open_shard(const std::string& path);
  void parse_header(void* header_json, size_t len, int shard_idx);

  std::vector<MmapShard> shards_;
  std::unordered_map<std::string, StTensor> tensors_;
};

}  // namespace umminfer::image
