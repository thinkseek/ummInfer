// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace umminfer::native {

struct TensorInfo {
  std::string dtype;          // safetensors dtype string, e.g. "BF16"
  std::vector<size_t> shape;  // row-major, same order as the header
  size_t byte_offset = 0;     // relative to the start of this shard's data region
  size_t byte_size = 0;
  size_t shard = 0;           // index into shard_bytes()
};

// Minimal safetensors reader over one file or several shards. Whole files are
// held in memory and tensor bytes are returned as views; no copies. Every
// header, offset, shape and cross-shard name is validated on construction.
class SafeTensors {
 public:
  // One already-read file.
  explicit SafeTensors(std::string file_bytes);

  // A model directory: <dir>/model.safetensors when present, otherwise the
  // shards named by <dir>/model.safetensors.index.json. Index shard names must
  // be plain file names inside <dir>; absolute paths and parent references are
  // rejected, as is any tensor missing from, duplicated across, or mapped to a
  // different shard than the index says.
  static SafeTensors from_model_dir(const std::string& dir);

  const std::unordered_map<std::string, TensorInfo>& tensors() const { return infos_; }
  const TensorInfo& info(std::string_view name) const;
  std::string_view tensor_bytes(std::string_view name) const;

  size_t shard_count() const { return shards_.size(); }
  const std::string& shard_bytes(size_t shard) const;
  const std::string& shard_name(size_t shard) const;
  // Absolute file offset where the shard's tensor data begins; tensor
  // byte_offsets are relative to this point.
  size_t data_region_offset(size_t shard = 0) const;
  size_t data_region_bytes(size_t shard = 0) const;

  // Writes a tiny single-file safetensors blob; test helper. dtype must be
  // "BF16" or "F32"; data length must match shape * element size.
  static std::string write(const std::vector<std::pair<std::string, TensorInfo>>& tensors,
                           const std::string& data);

 private:
  SafeTensors() = default;
  void init(std::vector<std::string> shards, std::vector<std::string> names);
  void parse_shard(size_t shard);

  std::vector<std::string> shards_;
  std::vector<std::string> names_;
  std::vector<size_t> data_starts_;
  std::unordered_map<std::string, TensorInfo> infos_;
};

}  // namespace umminfer::native
