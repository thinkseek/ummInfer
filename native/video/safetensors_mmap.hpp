// SPDX-License-Identifier: Apache-2.0
// Read-only memory-mapped Safetensors tensors with checked shapes and offsets.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace umminfer::video {

enum class SafetensorsDType { kF32, kBF16, kF16 };

struct SafetensorView {
  SafetensorsDType dtype = SafetensorsDType::kF32;
  std::vector<int64_t> shape;
  const std::byte* data = nullptr;
  size_t size_bytes = 0;

  size_t numel() const;
  std::vector<float> to_f32() const;
  void expect_shape(std::initializer_list<int64_t> expected) const;
};

class SafetensorsMmap {
 public:
  explicit SafetensorsMmap(const std::filesystem::path& path);
  ~SafetensorsMmap();

  SafetensorsMmap(const SafetensorsMmap&) = delete;
  SafetensorsMmap& operator=(const SafetensorsMmap&) = delete;

  bool contains(const std::string& name) const;
  const SafetensorView& tensor(const std::string& name) const;
  const SafetensorView& tensor(
      const std::string& name,
      std::initializer_list<int64_t> expected_shape) const;

 private:
  int fd_ = -1;
  const std::byte* mapping_ = nullptr;
  size_t mapping_size_ = 0;
  std::unordered_map<std::string, SafetensorView> tensors_;
};

class ShardedSafetensorsMmap {
 public:
  // Accepts a single .safetensors file or a model.safetensors.index.json path.
  explicit ShardedSafetensorsMmap(const std::filesystem::path& path);

  bool contains(const std::string& name) const;
  const SafetensorView& tensor(const std::string& name) const;
  const SafetensorView& tensor(
      const std::string& name,
      std::initializer_list<int64_t> expected_shape) const;

 private:
  const SafetensorsMmap& owner(const std::string& name) const;

  std::vector<std::unique_ptr<SafetensorsMmap>> files_;
  std::unordered_map<std::string, size_t> tensor_files_;
};

}  // namespace umminfer::video
