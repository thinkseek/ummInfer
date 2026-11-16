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
  size_t byte_offset = 0;     // relative to the start of the data region
  size_t byte_size = 0;
};

// Minimal safetensors reader: whole file in memory (the ASR checkpoint is
// 1.8 GiB and the host has ample RAM). Header is validated; tensor bytes are
// returned as views into the file buffer, no copies.
class SafeTensors {
 public:
  explicit SafeTensors(std::string file_bytes);

  const std::unordered_map<std::string, TensorInfo>& tensors() const { return infos_; }
  const TensorInfo& info(std::string_view name) const;
  std::string_view tensor_bytes(std::string_view name) const;
  const std::string& bytes() const { return file_; }
  // Absolute file offset where tensor data begins; tensor byte_offsets are
  // relative to this point.
  size_t data_region_offset() const { return data_start_; }

  // Writes a tiny safetensors file; test helper. dtype must be "BF16" or
  // "F32"; data length must match shape * element size.
  static std::string write(const std::vector<std::pair<std::string, TensorInfo>>& tensors,
                           const std::string& data);

 private:
  std::string file_;
  size_t data_start_ = 0;
  std::unordered_map<std::string, TensorInfo> infos_;
};

}  // namespace umminfer::native
