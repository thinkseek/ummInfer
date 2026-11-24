// SPDX-License-Identifier: Apache-2.0
#include "safetensors.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace umminfer::native {
namespace {

constexpr const char* kSingleFile = "model.safetensors";
constexpr const char* kIndexFile = "model.safetensors.index.json";

size_t dtype_size(std::string_view dtype) {
  if (dtype == "BF16" || dtype == "F16") return 2;
  if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
  if (dtype == "I64" || dtype == "U64" || dtype == "F64") return 8;
  if (dtype == "I8" || dtype == "U8" || dtype == "BOOL") return 1;
  throw std::invalid_argument("unsupported safetensors dtype: " + std::string(dtype));
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::invalid_argument("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// A shard name taken from the index must name one file inside the model
// directory. Separators, parent references and a leading '-' are rejected so
// a downloaded index cannot point the loader outside the directory.
bool plain_file_name(const std::string& name) {
  return !name.empty() && name != "." && name != ".." && name[0] != '-' &&
         name.find('/') == std::string::npos && name.find('\\') == std::string::npos &&
         name.find('\0') == std::string::npos;
}

}  // namespace

SafeTensors::SafeTensors(std::string file_bytes) {
  init({std::move(file_bytes)}, {kSingleFile});
}

SafeTensors SafeTensors::from_model_dir(const std::string& dir) {
  const std::string single = dir + "/" + kSingleFile;
  if (std::ifstream probe(single, std::ios::binary); probe) {
    SafeTensors st;
    st.init({read_file(single)}, {kSingleFile});
    return st;
  }

  const std::string index_path = dir + "/" + kIndexFile;
  if (!std::ifstream(index_path, std::ios::binary))
    throw std::invalid_argument("no checkpoint in " + dir + ": expected " + kSingleFile +
                                " or " + kIndexFile);

  nlohmann::json index;
  try {
    index = nlohmann::json::parse(read_file(index_path));
  } catch (const nlohmann::json::exception& e) {
    throw std::invalid_argument(std::string("invalid ") + kIndexFile + ": " + e.what());
  }
  if (!index.is_object() || !index.contains("weight_map") || !index["weight_map"].is_object())
    throw std::invalid_argument(std::string(kIndexFile) + " has no weight_map object");
  const nlohmann::json& weight_map = index["weight_map"];
  if (weight_map.empty())
    throw std::invalid_argument(std::string(kIndexFile) + " weight_map is empty");

  // Shard order follows first appearance in the index.
  std::vector<std::string> names;
  std::unordered_map<std::string, std::string> shard_of;
  for (auto it = weight_map.begin(); it != weight_map.end(); ++it) {
    if (!it.value().is_string())
      throw std::invalid_argument(std::string(kIndexFile) +
                                  " weight_map value is not a string: " + it.key());
    const std::string shard = it.value().get<std::string>();
    if (!plain_file_name(shard))
      throw std::invalid_argument(std::string(kIndexFile) +
                                  " shard name is not a plain file name: " + shard);
    if (std::find(names.begin(), names.end(), shard) == names.end()) names.push_back(shard);
    shard_of.emplace(it.key(), shard);
  }

  std::vector<std::string> shards;
  shards.reserve(names.size());
  for (const std::string& name : names) shards.push_back(read_file(dir + "/" + name));

  SafeTensors st;
  st.init(std::move(shards), names);

  // The index and the shard headers must name exactly the same tensors.
  for (const auto& [tensor, info] : st.infos_) {
    const auto it = shard_of.find(tensor);
    if (it == shard_of.end())
      throw std::invalid_argument("shard " + st.names_[info.shard] + " holds tensor " + tensor +
                                  " that " + kIndexFile + " does not list");
    if (it->second != st.names_[info.shard])
      throw std::invalid_argument("tensor " + tensor + " is in shard " + st.names_[info.shard] +
                                  " but " + kIndexFile + " names " + it->second);
  }
  for (const auto& [tensor, shard] : shard_of) {
    if (st.infos_.find(tensor) == st.infos_.end())
      throw std::invalid_argument(std::string(kIndexFile) + " lists tensor " + tensor +
                                  " missing from " + shard);
  }
  return st;
}

void SafeTensors::init(std::vector<std::string> shards, std::vector<std::string> names) {
  if (shards.empty()) throw std::invalid_argument("safetensors: no shards");
  if (shards.size() != names.size())
    throw std::invalid_argument("safetensors: shard/name count mismatch");
  shards_ = std::move(shards);
  names_ = std::move(names);
  data_starts_.assign(shards_.size(), 0);
  infos_.clear();
  for (size_t i = 0; i < shards_.size(); ++i) parse_shard(i);
  if (infos_.empty()) throw std::invalid_argument("safetensors: no tensors");
}

void SafeTensors::parse_shard(size_t shard) {
  const std::string& file = shards_[shard];
  if (file.size() < 8)
    throw std::invalid_argument("safetensors shard " + names_[shard] + " shorter than 8 bytes");
  const uint64_t header_len =
      static_cast<uint64_t>(static_cast<uint8_t>(file[0])) |
      (static_cast<uint64_t>(static_cast<uint8_t>(file[1])) << 8) |
      (static_cast<uint64_t>(static_cast<uint8_t>(file[2])) << 16) |
      (static_cast<uint64_t>(static_cast<uint8_t>(file[3])) << 24) |
      (static_cast<uint64_t>(static_cast<uint8_t>(file[4])) << 32) |
      (static_cast<uint64_t>(static_cast<uint8_t>(file[5])) << 40) |
      (static_cast<uint64_t>(static_cast<uint8_t>(file[6])) << 48) |
      (static_cast<uint64_t>(static_cast<uint8_t>(file[7])) << 56);
  if (header_len < 2 || header_len > file.size() - 8)
    throw std::invalid_argument("safetensors header length out of range in " + names_[shard]);
  const size_t data_start = 8 + static_cast<size_t>(header_len);
  const size_t region = file.size() - data_start;

  nlohmann::json header;
  try {
    header = nlohmann::json::parse(file.data() + 8, file.data() + data_start);
  } catch (const nlohmann::json::exception& e) {
    throw std::invalid_argument("invalid safetensors header JSON in " + names_[shard] + ": " +
                                e.what());
  }
  if (!header.is_object())
    throw std::invalid_argument("safetensors header is not an object in " + names_[shard]);

  for (auto it = header.begin(); it != header.end(); ++it) {
    if (it.key() == "__metadata__") continue;
    const auto& entry = it.value();
    if (!entry.is_object() || !entry.contains("dtype") || !entry.contains("shape") ||
        !entry.contains("data_offsets"))
      throw std::invalid_argument("malformed tensor header: " + it.key());
    TensorInfo info;
    info.dtype = entry["dtype"].get<std::string>();
    info.shape = entry["shape"].get<std::vector<size_t>>();
    const auto offsets = entry["data_offsets"].get<std::vector<size_t>>();
    if (offsets.size() != 2) throw std::invalid_argument("bad data_offsets: " + it.key());
    const size_t start = offsets[0];
    const size_t end = offsets[1];
    if (end < start || start > region || end > region)
      throw std::invalid_argument("tensor data exceeds shard " + names_[shard] + ": " + it.key());
    info.byte_offset = start;
    info.byte_size = end - start;
    info.shard = shard;

    size_t expected = dtype_size(info.dtype);
    for (size_t dim : info.shape) {
      if (dim == 0) throw std::invalid_argument("zero-dim tensor: " + it.key());
      if (dim > std::numeric_limits<size_t>::max() / expected)
        throw std::invalid_argument("tensor element count overflows: " + it.key());
      expected *= dim;
    }
    if (expected != info.byte_size)
      throw std::invalid_argument("tensor size does not match shape/dtype: " + it.key());
    // Header keys are unique inside one JSON object, so a collision here is the
    // same tensor present in two shards.
    if (!infos_.emplace(it.key(), std::move(info)).second)
      throw std::invalid_argument("tensor duplicated across shards: " + it.key());
  }
  data_starts_[shard] = data_start;
}

const TensorInfo& SafeTensors::info(std::string_view name) const {
  auto it = infos_.find(std::string(name));
  if (it == infos_.end()) throw std::invalid_argument("missing tensor: " + std::string(name));
  return it->second;
}

std::string_view SafeTensors::tensor_bytes(std::string_view name) const {
  const TensorInfo& t = info(name);
  const std::string& file = shards_.at(t.shard);
  const size_t begin = data_starts_[t.shard] + t.byte_offset;
  if (begin > file.size() || t.byte_size > file.size() - begin)
    throw std::invalid_argument("tensor data exceeds file: " + std::string(name));
  return std::string_view(file.data() + begin, t.byte_size);
}

const std::string& SafeTensors::shard_bytes(size_t shard) const {
  if (shard >= shards_.size()) throw std::invalid_argument("shard index out of range");
  return shards_[shard];
}

const std::string& SafeTensors::shard_name(size_t shard) const {
  if (shard >= names_.size()) throw std::invalid_argument("shard index out of range");
  return names_[shard];
}

size_t SafeTensors::data_region_offset(size_t shard) const {
  if (shard >= data_starts_.size()) throw std::invalid_argument("shard index out of range");
  return data_starts_[shard];
}

size_t SafeTensors::data_region_bytes(size_t shard) const {
  return shard_bytes(shard).size() - data_region_offset(shard);
}

std::string SafeTensors::write(
    const std::vector<std::pair<std::string, TensorInfo>>& tensors, const std::string& data) {
  nlohmann::json header = nlohmann::json::object();
  for (const auto& [name, t] : tensors) {
    header[name] = {{"dtype", t.dtype},
                    {"shape", t.shape},
                    {"data_offsets", {t.byte_offset, t.byte_offset + t.byte_size}}};
  }
  std::string header_bytes = header.dump();
  std::string out;
  out.resize(8);
  const uint64_t n = header_bytes.size();
  for (int i = 0; i < 8; ++i) out[i] = static_cast<char>((n >> (8 * i)) & 0xff);
  out += header_bytes;
  out += data;
  return out;
}

}  // namespace umminfer::native
