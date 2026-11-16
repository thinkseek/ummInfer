// SPDX-License-Identifier: Apache-2.0
#include "safetensors_mmap.hpp"

#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace umminfer::omni {
namespace {

std::string read_text(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::invalid_argument("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

size_t dtype_width(const std::string& dtype) {
  if (dtype == "BF16" || dtype == "F16") return 2;
  if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
  if (dtype == "I64" || dtype == "U64" || dtype == "F64") return 8;
  if (dtype == "I8" || dtype == "U8" || dtype == "BOOL") return 1;
  throw std::invalid_argument("unsupported safetensors dtype: " + dtype);
}

}  // namespace

ShardedSafetensors::ShardedSafetensors(const std::string& model_dir) : dir_(model_dir) {
  const nlohmann::json index =
      nlohmann::json::parse(read_text(dir_ + "/model.safetensors.index.json"));
  const auto& weight_map = index.at("weight_map");

  // Stable, unique shard file list in index order of first appearance.
  std::unordered_map<std::string, int> shard_id;
  for (auto it = weight_map.begin(); it != weight_map.end(); ++it) {
    const std::string file = it.value().get<std::string>();
    if (!shard_id.emplace(file, static_cast<int>(shard_id.size())).second) continue;
  }
  std::vector<std::string> files(shard_id.size());
  for (const auto& [file, id] : shard_id) files[id] = file;
  shards_.resize(files.size());

  for (int sid = 0; sid < static_cast<int>(files.size()); ++sid) {
    const std::string path = dir_ + "/" + files[sid];
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::invalid_argument("cannot open " + path);
    struct stat st {};
    if (fstat(fd, &st) != 0 || st.st_size < 8) {
      close(fd);
      throw std::invalid_argument("bad safetensors shard: " + path);
    }
    void* addr = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);  // mapping stays valid after close.
    if (addr == MAP_FAILED) throw std::runtime_error("mmap failed for " + path);

    const auto* base = static_cast<const unsigned char*>(addr);
    uint64_t header_len = 0;
    for (int i = 0; i < 8; ++i)
      header_len |= static_cast<uint64_t>(base[i]) << (8 * i);
    if (8 + header_len > static_cast<uint64_t>(st.st_size))
      throw std::invalid_argument("bad header length in " + path);
    const nlohmann::json header =
        nlohmann::json::parse(reinterpret_cast<const char*>(base + 8),
                              reinterpret_cast<const char*>(base + 8 + header_len));

    shards_[sid] = {fd, addr, static_cast<size_t>(st.st_size), 8 + header_len};
    for (auto it = header.begin(); it != header.end(); ++it) {
      if (it.key() == "__metadata__") continue;
      const auto& entry = it.value();
      OmniTensorInfo info;
      info.dtype = entry.at("dtype").get<std::string>();
      info.shape = entry.at("shape").get<std::vector<size_t>>();
      const auto offsets = entry.at("data_offsets").get<std::vector<size_t>>();
      if (offsets.size() != 2 || offsets[1] < offsets[0])
        throw std::invalid_argument("bad data_offsets: " + it.key());
      info.byte_offset = offsets[0];
      info.byte_size = offsets[1] - offsets[0];
      info.shard = sid;
      size_t expected = dtype_width(info.dtype);
      for (size_t d : info.shape) {
        if (d == 0) throw std::invalid_argument("zero-dim tensor: " + it.key());
        expected *= d;
      }
      if (expected != info.byte_size)
        throw std::invalid_argument("tensor size mismatch: " + it.key());
      infos_.emplace(it.key(), std::move(info));
    }
  }

  // Every index entry must resolve to a real tensor header.
  for (auto it = weight_map.begin(); it != weight_map.end(); ++it) {
    if (infos_.find(it.key()) == infos_.end())
      throw std::invalid_argument("index tensor missing from shard: " + it.key());
  }
}

ShardedSafetensors::~ShardedSafetensors() {
  for (const auto& s : shards_)
    if (s.addr && s.addr != MAP_FAILED) munmap(s.addr, s.length);
}

const OmniTensorInfo& ShardedSafetensors::info(const std::string& name) const {
  auto it = infos_.find(name);
  if (it == infos_.end()) throw std::invalid_argument("missing tensor: " + name);
  return it->second;
}

const void* ShardedSafetensors::tensor_bytes(const std::string& name) const {
  const OmniTensorInfo& t = info(name);
  const ShardMapping& s = shards_.at(t.shard);
  return static_cast<const unsigned char*>(s.addr) + s.data_region_offset + t.byte_offset;
}

}  // namespace umminfer::omni
