// SPDX-License-Identifier: Apache-2.0
#include "safetensors_shard.hpp"

#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace umminfer::image {
namespace {

size_t dtype_size(std::string_view dtype) {
  if (dtype == "BF16" || dtype == "F16") return 2;
  if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
  if (dtype == "I64" || dtype == "U64" || dtype == "F64") return 8;
  if (dtype == "I8" || dtype == "U8" || dtype == "BOOL") return 1;
  throw std::invalid_argument("unsupported safetensors dtype: " + std::string(dtype));
}

}  // namespace

ShardedSafeTensors::ShardedSafeTensors(const std::string& component_dir) {
  std::vector<std::string> files;
  for (const auto& e : std::filesystem::directory_iterator(component_dir)) {
    if (e.path().extension() == ".safetensors") files.push_back(e.path().string());
  }
  if (files.empty()) throw std::invalid_argument("no safetensors in " + component_dir);
  std::sort(files.begin(), files.end());
  for (const auto& f : files) open_shard(f);
  if (tensors_.empty()) throw std::invalid_argument("no tensors parsed in " + component_dir);
}

void ShardedSafeTensors::open_shard(const std::string& path) {
  const int shard_idx = static_cast<int>(shards_.size());
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::invalid_argument("cannot open " + path);
  struct stat st {};
  if (::fstat(fd, &st) != 0 || st.st_size < 8) {
    ::close(fd);
    throw std::invalid_argument("bad safetensors file " + path);
  }
  void* map = ::mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED) {
    ::close(fd);
    throw std::runtime_error("mmap failed for " + path);
  }
  const auto* base = static_cast<const uint8_t*>(map);
  uint64_t header_len = 0;
  for (int i = 0; i < 8; ++i) header_len |= static_cast<uint64_t>(base[i]) << (8 * i);
  if (8 + header_len > static_cast<size_t>(st.st_size) ||
      header_len > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / 2)) {
    ::munmap(map, static_cast<size_t>(st.st_size));
    ::close(fd);
    throw std::invalid_argument("safetensors header out of range in " + path);
  }
  MmapShard shard{path, fd, static_cast<size_t>(st.st_size), base, 8 + static_cast<size_t>(header_len)};
  shards_.push_back(shard);

  nlohmann::json header;
  try {
    header = nlohmann::json::parse(base + 8, base + 8 + header_len);
  } catch (const nlohmann::json::exception& e) {
    throw std::invalid_argument("bad safetensors header JSON in " + path + ": " + e.what());
  }
  if (!header.is_object()) throw std::invalid_argument("header not an object in " + path);
  for (auto it = header.begin(); it != header.end(); ++it) {
    if (it.key() == "__metadata__") continue;
    const auto& entry = it.value();
    if (!entry.is_object() || !entry.contains("dtype") || !entry.contains("shape") ||
        !entry.contains("data_offsets"))
      throw std::invalid_argument("malformed tensor header: " + it.key());
    StTensor t;
    t.dtype = entry["dtype"].get<std::string>();
    t.shape = entry["shape"].get<std::vector<size_t>>();
    const auto offs = entry["data_offsets"].get<std::vector<size_t>>();
    if (offs.size() != 2 || offs[1] < offs[0])
      throw std::invalid_argument("bad data_offsets: " + it.key());
    t.byte_offset = offs[0];
    t.byte_size = offs[1] - offs[0];
    t.shard = shard_idx;
    size_t expected = dtype_size(t.dtype);
    for (size_t d : t.shape) {
      if (d == 0) throw std::invalid_argument("zero dim tensor: " + it.key());
      expected *= d;
    }
    if (expected != t.byte_size)
      throw std::invalid_argument("size/shape mismatch: " + it.key());
    if (!tensors_.emplace(it.key(), std::move(t)).second)
      throw std::invalid_argument("duplicate tensor across shards: " + it.key());
  }
}

const StTensor& ShardedSafeTensors::info(std::string_view name) const {
  auto it = tensors_.find(std::string(name));
  if (it == tensors_.end()) throw std::invalid_argument("missing tensor: " + std::string(name));
  return it->second;
}

const void* ShardedSafeTensors::host_ptr(const StTensor& t) const {
  const MmapShard& s = shards_.at(t.shard);
  return s.data + s.data_region + t.byte_offset;
}

size_t ShardedSafeTensors::numel(const std::vector<size_t>& shape) {
  size_t n = 1;
  for (size_t d : shape) n *= d;
  return n;
}

ShardedSafeTensors::~ShardedSafeTensors() {
  for (const auto& s : shards_) {
    if (s.data) ::munmap(const_cast<uint8_t*>(s.data), s.mapped_size);
    if (s.fd >= 0) ::close(s.fd);
  }
}

}  // namespace umminfer::image
