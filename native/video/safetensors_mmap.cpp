// SPDX-License-Identifier: Apache-2.0
#include "safetensors_mmap.hpp"

#include <nlohmann/json.hpp>

#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace umminfer::video {
namespace {

size_t dtype_size(SafetensorsDType dtype) {
  return dtype == SafetensorsDType::kF32 ? 4u : 2u;
}

uint16_t load_u16(const std::byte* p) {
  return static_cast<uint16_t>(p[0]) |
         (static_cast<uint16_t>(p[1]) << 8);
}

uint32_t load_u32(const std::byte* p) {
  return static_cast<uint32_t>(p[0]) |
         (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

float f16_to_f32(uint16_t half) {
  const uint32_t sign = static_cast<uint32_t>(half & 0x8000u) << 16;
  const uint32_t exponent = (half >> 10) & 0x1fu;
  const uint32_t mantissa = half & 0x03ffu;
  if (exponent == 0 && mantissa != 0) {
    const float value = std::ldexp(static_cast<float>(mantissa), -24);
    return sign == 0 ? value : -value;
  }
  uint32_t bits = sign;
  if (exponent == 0) {
    // Zero already has the correct representation after sign extraction.
  } else if (exponent == 31) {
    bits |= 0x7f800000u | (mantissa << 13);
  } else {
    bits |= ((exponent + 112u) << 23) | (mantissa << 13);
  }
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::string shape_string(const std::vector<int64_t>& shape) {
  std::ostringstream out;
  out << '[';
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) out << ',';
    out << shape[i];
  }
  out << ']';
  return out.str();
}

}  // namespace

size_t SafetensorView::numel() const {
  size_t count = 1;
  for (const int64_t dim : shape) {
    if (dim < 0) throw std::runtime_error("negative tensor dimension");
    if (dim != 0 && count > std::numeric_limits<size_t>::max() /
                               static_cast<uint64_t>(dim)) {
      throw std::overflow_error("tensor element count overflows size_t");
    }
    count *= static_cast<size_t>(dim);
  }
  return count;
}

std::vector<float> SafetensorView::to_f32() const {
  const size_t count = numel();
  std::vector<float> result(count);
  for (size_t i = 0; i < count; ++i) {
    if (dtype == SafetensorsDType::kF32) {
      const uint32_t bits = load_u32(data + i * 4);
      std::memcpy(&result[i], &bits, sizeof(result[i]));
    } else if (dtype == SafetensorsDType::kBF16) {
      const uint32_t bits = static_cast<uint32_t>(load_u16(data + i * 2)) << 16;
      std::memcpy(&result[i], &bits, sizeof(result[i]));
    } else {
      result[i] = f16_to_f32(load_u16(data + i * 2));
    }
  }
  return result;
}

void SafetensorView::expect_shape(
    std::initializer_list<int64_t> expected) const {
  const std::vector<int64_t> wanted(expected);
  if (shape != wanted) {
    throw std::invalid_argument("tensor shape " + shape_string(shape) +
                                " does not match expected " +
                                shape_string(wanted));
  }
}

SafetensorsMmap::SafetensorsMmap(const std::filesystem::path& path) {
  try {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
      throw std::runtime_error("open failed: " + std::string(std::strerror(errno)));
    }
    struct stat info {};
    if (::fstat(fd_, &info) != 0) {
      throw std::runtime_error("fstat failed: " +
                               std::string(std::strerror(errno)));
    }
    if (info.st_size < 8) {
      throw std::runtime_error("file is shorter than the 8-byte header length");
    }
    mapping_size_ = static_cast<size_t>(info.st_size);
    void* mapped = ::mmap(nullptr, mapping_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped == MAP_FAILED) {
      throw std::runtime_error("mmap failed: " + std::string(std::strerror(errno)));
    }
    mapping_ = static_cast<const std::byte*>(mapped);

    uint64_t header_size = 0;
    for (unsigned i = 0; i < 8; ++i) {
      header_size |= static_cast<uint64_t>(mapping_[i]) << (i * 8);
    }
    if (header_size > mapping_size_ - 8) {
      throw std::runtime_error("JSON header extends beyond end of file");
    }
    const auto* header_chars = reinterpret_cast<const char*>(mapping_ + 8);
    const auto header = nlohmann::json::parse(
        std::string_view(header_chars, static_cast<size_t>(header_size)));
    if (!header.is_object()) throw std::runtime_error("JSON header is not an object");

    const size_t data_begin = 8 + static_cast<size_t>(header_size);
    for (const auto& [name, entry] : header.items()) {
      if (name == "__metadata__") continue;
      if (!entry.is_object()) {
        throw std::runtime_error("tensor '" + name + "' is not an object");
      }
      SafetensorView view;
      const std::string dtype = entry.at("dtype").get<std::string>();
      if (dtype == "F32") {
        view.dtype = SafetensorsDType::kF32;
      } else if (dtype == "BF16") {
        view.dtype = SafetensorsDType::kBF16;
      } else if (dtype == "F16") {
        view.dtype = SafetensorsDType::kF16;
      } else {
        throw std::runtime_error("tensor '" + name + "' has unsupported dtype " +
                                 dtype);
      }
      for (const auto& dim : entry.at("shape")) {
        const int64_t value = dim.get<int64_t>();
        if (value < 0) {
          throw std::runtime_error("tensor '" + name + "' has a negative dimension");
        }
        view.shape.push_back(value);
      }
      const auto& offsets = entry.at("data_offsets");
      if (!offsets.is_array() || offsets.size() != 2) {
        throw std::runtime_error("tensor '" + name + "' needs two data offsets");
      }
      const uint64_t begin = offsets[0].get<uint64_t>();
      const uint64_t end = offsets[1].get<uint64_t>();
      const size_t count = view.numel();
      if (count > std::numeric_limits<size_t>::max() / dtype_size(view.dtype)) {
        throw std::overflow_error("tensor '" + name + "' byte size overflows");
      }
      const size_t expected_bytes = count * dtype_size(view.dtype);
      if (end < begin || end - begin != expected_bytes) {
        throw std::runtime_error("tensor '" + name +
                                 "' byte range does not match its shape and dtype");
      }
      if (begin > mapping_size_ - data_begin ||
          end > mapping_size_ - data_begin) {
        throw std::runtime_error("tensor '" + name + "' extends beyond end of file");
      }
      view.size_bytes = expected_bytes;
      view.data = mapping_ + data_begin + begin;
      tensors_.emplace(name, std::move(view));
    }
  } catch (const std::exception& err) {
    if (mapping_ != nullptr) ::munmap(const_cast<std::byte*>(mapping_), mapping_size_);
    if (fd_ >= 0) ::close(fd_);
    mapping_ = nullptr;
    fd_ = -1;
    throw std::runtime_error("failed to load Safetensors '" + path.string() +
                             "': " + err.what());
  } catch (...) {
    if (mapping_ != nullptr) ::munmap(const_cast<std::byte*>(mapping_), mapping_size_);
    if (fd_ >= 0) ::close(fd_);
    throw;
  }
}

SafetensorsMmap::~SafetensorsMmap() {
  if (mapping_ != nullptr) {
    ::munmap(const_cast<std::byte*>(mapping_), mapping_size_);
  }
  if (fd_ >= 0) ::close(fd_);
}

bool SafetensorsMmap::contains(const std::string& name) const {
  return tensors_.find(name) != tensors_.end();
}

const SafetensorView& SafetensorsMmap::tensor(const std::string& name) const {
  const auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    throw std::out_of_range("Safetensors tensor not found: " + name);
  }
  return it->second;
}

const SafetensorView& SafetensorsMmap::tensor(
    const std::string& name,
    std::initializer_list<int64_t> expected_shape) const {
  const auto& view = tensor(name);
  try {
    view.expect_shape(expected_shape);
  } catch (const std::invalid_argument& err) {
    throw std::invalid_argument("tensor '" + name + "': " + err.what());
  }
  return view;
}

ShardedSafetensorsMmap::ShardedSafetensorsMmap(
    const std::filesystem::path& path) {
  if (path.extension() != ".json") {
    files_.push_back(std::make_unique<SafetensorsMmap>(path));
    return;
  }

  std::ifstream input(path);
  if (!input) throw std::runtime_error("failed to open Safetensors index: " + path.string());
  const auto index = nlohmann::json::parse(input);
  const auto& weight_map = index.at("weight_map");
  if (!weight_map.is_object() || weight_map.empty()) {
    throw std::runtime_error("Safetensors index has no weight_map entries");
  }

  std::map<std::string, size_t> shard_files;
  for (const auto& [name, shard] : weight_map.items()) {
    const std::string filename = shard.get<std::string>();
    auto [it, inserted] = shard_files.emplace(filename, files_.size());
    if (inserted) {
      files_.push_back(
          std::make_unique<SafetensorsMmap>(path.parent_path() / filename));
    }
    if (!tensor_files_.emplace(name, it->second).second) {
      throw std::runtime_error("duplicate tensor in Safetensors index: " + name);
    }
  }
  for (const auto& [name, file_index] : tensor_files_) {
    if (!files_[file_index]->contains(name)) {
      throw std::runtime_error("indexed tensor '" + name +
                               "' is absent from its Safetensors shard");
    }
  }
}

bool ShardedSafetensorsMmap::contains(const std::string& name) const {
  if (tensor_files_.empty() && files_.size() == 1) {
    return files_[0]->contains(name);
  }
  return tensor_files_.find(name) != tensor_files_.end();
}

const SafetensorsMmap& ShardedSafetensorsMmap::owner(
    const std::string& name) const {
  if (tensor_files_.empty() && files_.size() == 1) {
    if (files_[0]->contains(name)) return *files_[0];
  } else {
    const auto it = tensor_files_.find(name);
    if (it != tensor_files_.end()) return *files_[it->second];
  }
  throw std::out_of_range("Safetensors tensor not found: " + name);
}

const SafetensorView& ShardedSafetensorsMmap::tensor(
    const std::string& name) const {
  return owner(name).tensor(name);
}

const SafetensorView& ShardedSafetensorsMmap::tensor(
    const std::string& name,
    std::initializer_list<int64_t> expected_shape) const {
  return owner(name).tensor(name, expected_shape);
}

}  // namespace umminfer::video
