// SPDX-License-Identifier: Apache-2.0
#include "safetensors.hpp"

#include <nlohmann/json.hpp>

#include <limits>
#include <stdexcept>

namespace umminfer::native {
namespace {

size_t dtype_size(std::string_view dtype) {
  if (dtype == "BF16" || dtype == "F16") return 2;
  if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
  if (dtype == "I64" || dtype == "U64" || dtype == "F64") return 8;
  if (dtype == "I8" || dtype == "U8" || dtype == "BOOL") return 1;
  throw std::invalid_argument("unsupported safetensors dtype: " + std::string(dtype));
}

}  // namespace

SafeTensors::SafeTensors(std::string file_bytes) : file_(std::move(file_bytes)) {
  if (file_.size() < 8) throw std::invalid_argument("safetensors file shorter than 8 bytes");
  const uint64_t header_len =
      static_cast<uint64_t>(static_cast<uint8_t>(file_[0])) |
      (static_cast<uint64_t>(static_cast<uint8_t>(file_[1])) << 8) |
      (static_cast<uint64_t>(static_cast<uint8_t>(file_[2])) << 16) |
      (static_cast<uint64_t>(static_cast<uint8_t>(file_[3])) << 24) |
      (static_cast<uint64_t>(static_cast<uint8_t>(file_[4])) << 32) |
      (static_cast<uint64_t>(static_cast<uint8_t>(file_[5])) << 40) |
      (static_cast<uint64_t>(static_cast<uint8_t>(file_[6])) << 48) |
      (static_cast<uint64_t>(static_cast<uint8_t>(file_[7])) << 56);
  if (header_len > file_.size() - 8 || header_len > std::numeric_limits<size_t>::max() / 2)
    throw std::invalid_argument("safetensors header length out of range");
  data_start_ = 8 + static_cast<size_t>(header_len);

  nlohmann::json header;
  try {
    header = nlohmann::json::parse(file_.data() + 8, file_.data() + data_start_);
  } catch (const nlohmann::json::exception& e) {
    throw std::invalid_argument(std::string("invalid safetensors header JSON: ") + e.what());
  }
  if (!header.is_object()) throw std::invalid_argument("safetensors header is not an object");

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
    if (offsets.size() != 2)
      throw std::invalid_argument("bad data_offsets: " + it.key());
    const size_t start = offsets[0];
    const size_t end = offsets[1];
    if (end < start) throw std::invalid_argument("tensor end < start: " + it.key());
    info.byte_offset = start;
    info.byte_size = end - start;

    size_t expected = dtype_size(info.dtype);
    for (size_t dim : info.shape) {
      if (dim == 0) throw std::invalid_argument("zero-dim tensor: " + it.key());
      expected *= dim;
    }
    if (expected != info.byte_size)
      throw std::invalid_argument("tensor size does not match shape/dtype: " + it.key());
    infos_.emplace(it.key(), std::move(info));
  }
  if (infos_.empty()) throw std::invalid_argument("safetensors file contains no tensors");
}

const TensorInfo& SafeTensors::info(std::string_view name) const {
  auto it = infos_.find(std::string(name));
  if (it == infos_.end()) throw std::invalid_argument("missing tensor: " + std::string(name));
  return it->second;
}

std::string_view SafeTensors::tensor_bytes(std::string_view name) const {
  const TensorInfo& t = info(name);
  if (data_start_ + t.byte_offset + t.byte_size > file_.size())
    throw std::invalid_argument("tensor data exceeds file: " + std::string(name));
  return std::string_view(file_.data() + data_start_ + t.byte_offset, t.byte_size);
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
