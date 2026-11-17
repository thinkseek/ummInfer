// SPDX-License-Identifier: Apache-2.0
#pragma once
// Minimal golden-tensor IO for GPU parity tests against the offline diffusers
// dumps (.bin raw little-endian + .json shape/dtype). Test-only helper.
#include <nlohmann/json.hpp>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ummtest {

struct GoldenMeta {
  std::vector<int64_t> shape;
  std::string dtype;
};

inline GoldenMeta read_meta(const std::string& path_no_ext) {
  std::ifstream f(path_no_ext + ".json");
  if (!f) throw std::runtime_error("missing meta " + path_no_ext + ".json");
  nlohmann::json j;
  f >> j;
  GoldenMeta m;
  for (auto d : j["shape"]) m.shape.push_back(d.get<int64_t>());
  m.dtype = j["dtype"].get<std::string>();
  return m;
}

inline std::vector<uint16_t> read_bf16(const std::string& path_no_ext, size_t expect) {
  std::ifstream f(path_no_ext + ".bin", std::ios::binary);
  if (!f) throw std::runtime_error("missing bin " + path_no_ext + ".bin");
  std::ostringstream ss; ss << f.rdbuf();
  const std::string& b = ss.str();
  if (b.size() != expect * 2)
    throw std::runtime_error("golden size mismatch " + path_no_ext);
  std::vector<uint16_t> out(expect);
  std::memcpy(out.data(), b.data(), b.size());
  return out;
}

inline std::vector<float> read_f32(const std::string& path_no_ext, size_t expect) {
  std::ifstream f(path_no_ext + ".bin", std::ios::binary);
  std::ostringstream ss; ss << f.rdbuf();
  const std::string& b = ss.str();
  if (b.size() != expect * 4)
    throw std::runtime_error("golden f32 size mismatch " + path_no_ext);
  std::vector<float> out(expect);
  std::memcpy(out.data(), b.data(), b.size());
  return out;
}

inline std::vector<int64_t> read_i64(const std::string& path_no_ext, size_t expect) {
  std::ifstream f(path_no_ext + ".bin", std::ios::binary);
  if (!f) throw std::runtime_error("missing bin " + path_no_ext + ".bin");
  std::ostringstream ss; ss << f.rdbuf();
  const std::string& b = ss.str();
  if (b.size() != expect * 8)
    throw std::runtime_error("golden i64 size mismatch " + path_no_ext);
  std::vector<int64_t> out(expect);
  std::memcpy(out.data(), b.data(), b.size());
  return out;
}

inline float bf_to_f(uint16_t bits) {
  uint32_t u = uint32_t(bits) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Allclose over BF16 bit patterns, comparing as fp32. Returns max abs error.
inline float max_abs_bf16(const std::vector<uint16_t>& a,
                          const std::vector<uint16_t>& b) {
  float mx = 0.0f;
  for (size_t i = 0; i < a.size(); ++i)
    mx = std::max(mx, std::fabs(bf_to_f(a[i]) - bf_to_f(b[i])));
  return mx;
}

}  // namespace ummtest
