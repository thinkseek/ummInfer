// SPDX-License-Identifier: Apache-2.0
#include "umminfer/util/codec.hpp"

#include <cstdint>
#include <cstring>

#include "umminfer/proto/wire.hpp"

namespace umminfer::util {
namespace {

int base64_char_value(unsigned char c) noexcept {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+') return 62;
  if (c == '/') return 63;
  return -1;
}

uint32_t sha_rotr(uint32_t value, int bits) noexcept {
  return (value >> bits) | (value << (32 - bits));
}

void sha256_compress(uint32_t state[8], const unsigned char block[64]) noexcept {
  static constexpr uint32_t k[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
      0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
      0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
      0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
      0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
      0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
      0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
      0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
      0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
      0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
      0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
  uint32_t w[64];
  for (int i = 0; i < 16; ++i) {
    w[i] = (uint32_t(block[i * 4]) << 24) |
           (uint32_t(block[i * 4 + 1]) << 16) |
           (uint32_t(block[i * 4 + 2]) << 8) |
           uint32_t(block[i * 4 + 3]);
  }
  for (int i = 16; i < 64; ++i) {
    const uint32_t s0 =
        sha_rotr(w[i - 15], 7) ^ sha_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const uint32_t s1 =
        sha_rotr(w[i - 2], 17) ^ sha_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
  uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
  for (int i = 0; i < 64; ++i) {
    const uint32_t s1 = sha_rotr(e, 6) ^ sha_rotr(e, 11) ^ sha_rotr(e, 25);
    const uint32_t ch = (e & f) ^ (~e & g);
    const uint32_t t1 = h + s1 + ch + k[i] + w[i];
    const uint32_t s0 = sha_rotr(a, 2) ^ sha_rotr(a, 13) ^ sha_rotr(a, 22);
    const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t t2 = s0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  state[5] += f;
  state[6] += g;
  state[7] += h;
}

}  // namespace

std::string base64_encode(std::string_view bytes) {
  static constexpr char alphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve((bytes.size() + 2) / 3 * 4);
  for (size_t i = 0; i < bytes.size(); i += 3) {
    const uint32_t a = static_cast<unsigned char>(bytes[i]);
    const uint32_t b =
        i + 1 < bytes.size() ? static_cast<unsigned char>(bytes[i + 1]) : 0;
    const uint32_t c =
        i + 2 < bytes.size() ? static_cast<unsigned char>(bytes[i + 2]) : 0;
    const uint32_t value = (a << 16) | (b << 8) | c;
    out.push_back(alphabet[(value >> 18) & 63]);
    out.push_back(alphabet[(value >> 12) & 63]);
    out.push_back(i + 1 < bytes.size() ? alphabet[(value >> 6) & 63] : '=');
    out.push_back(i + 2 < bytes.size() ? alphabet[value & 63] : '=');
  }
  return out;
}

std::string base64_decode(std::string_view text, size_t max_bytes,
                          bool require_nonempty) {
  if (require_nonempty && text.empty()) {
    throw ProtocolError("base64 payload must not be empty");
  }
  if (text.size() % 4 != 0) {
    throw ProtocolError("base64 length is not a multiple of 4");
  }
  size_t padding = 0;
  if (!text.empty() && text.back() == '=') ++padding;
  if (text.size() >= 2 && text[text.size() - 2] == '=') ++padding;
  const size_t decoded_size = text.size() / 4 * 3 - padding;
  if (decoded_size > max_bytes) {
    throw ProtocolError("base64 output exceeds its size limit");
  }
  std::string out;
  out.reserve(decoded_size);
  for (size_t i = 0; i < text.size(); i += 4) {
    const bool last = i + 4 == text.size();
    int v[4] = {0, 0, 0, 0};
    for (size_t j = 0; j < 4; ++j) {
      const char c = text[i + j];
      if (c == '=') {
        if (!last || j + padding < 4) {
          throw ProtocolError("invalid base64 padding");
        }
        continue;
      }
      v[j] = base64_char_value(static_cast<unsigned char>(c));
      if (v[j] < 0) {
        throw ProtocolError("invalid base64 character");
      }
    }
    if (last && padding == 2 && (v[1] & 0x0f) != 0) {
      throw ProtocolError("non-canonical base64 padding");
    }
    if (last && padding == 1 && (v[2] & 0x03) != 0) {
      throw ProtocolError("non-canonical base64 padding");
    }
    const uint32_t chunk = (static_cast<uint32_t>(v[0]) << 18) |
                           (static_cast<uint32_t>(v[1]) << 12) |
                           (static_cast<uint32_t>(v[2]) << 6) |
                           static_cast<uint32_t>(v[3]);
    const size_t bytes_here = 3 - (last ? padding : 0);
    if (bytes_here >= 1) out.push_back(static_cast<char>((chunk >> 16) & 0xff));
    if (bytes_here >= 2) out.push_back(static_cast<char>((chunk >> 8) & 0xff));
    if (bytes_here >= 3) out.push_back(static_cast<char>(chunk & 0xff));
  }
  return out;
}

std::string sha256_hex(std::string_view bytes) {
  uint32_t state[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                       0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  const uint64_t bit_length = uint64_t(bytes.size()) * 8u;
  size_t offset = 0;
  unsigned char block[64];
  auto process = [&](size_t length) {
    std::memset(block + length, 0, 64 - length);
    sha256_compress(state, block);
  };
  while (bytes.size() - offset >= 64) {
    std::memcpy(block, bytes.data() + offset, 64);
    sha256_compress(state, block);
    offset += 64;
  }
  const size_t tail = bytes.size() - offset;
  std::memset(block, 0, sizeof(block));
  if (tail > 0) std::memcpy(block, bytes.data() + offset, tail);
  block[tail] = 0x80;
  if (tail >= 56) {
    process(tail + 1);
    std::memset(block, 0, 56);
  }
  for (int i = 0; i < 8; ++i) {
    block[63 - i] = static_cast<unsigned char>(bit_length >> (i * 8));
  }
  sha256_compress(state, block);
  static constexpr char hex[] = "0123456789abcdef";
  std::string out;
  out.resize(64);
  for (int i = 0; i < 8; ++i) {
    out[i * 8] = hex[(state[i] >> 28) & 0xf];
    out[i * 8 + 1] = hex[(state[i] >> 24) & 0xf];
    out[i * 8 + 2] = hex[(state[i] >> 20) & 0xf];
    out[i * 8 + 3] = hex[(state[i] >> 16) & 0xf];
    out[i * 8 + 4] = hex[(state[i] >> 12) & 0xf];
    out[i * 8 + 5] = hex[(state[i] >> 8) & 0xf];
    out[i * 8 + 6] = hex[(state[i] >> 4) & 0xf];
    out[i * 8 + 7] = hex[state[i] & 0xf];
  }
  return out;
}

std::string utf8_sanitized(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  size_t index = 0;
  while (index < text.size()) {
    const unsigned char lead = static_cast<unsigned char>(text[index]);
    size_t length = 0;
    if (lead < 0x80) length = 1;
    else if (lead >= 0xC2 && lead <= 0xDF) length = 2;
    else if (lead >= 0xE0 && lead <= 0xEF) length = 3;
    else if (lead >= 0xF0 && lead <= 0xF4) length = 4;
    bool valid = length > 0 && index + length <= text.size();
    for (size_t offset = 1; valid && offset < length; ++offset) {
      valid = (static_cast<unsigned char>(text[index + offset]) & 0xC0) == 0x80;
    }
    if (valid && length > 1) {
      const unsigned char second = static_cast<unsigned char>(text[index + 1]);
      if (lead == 0xE0) valid = second >= 0xA0;
      else if (lead == 0xED) valid = second <= 0x9F;
      else if (lead == 0xF0) valid = second >= 0x90;
      else if (lead == 0xF4) valid = second <= 0x8F;
    }
    if (!valid) {
      out.push_back('?');
      ++index;
      continue;
    }
    out.append(text.data() + index, length);
    index += length;
  }
  return out;
}

bool constant_time_equals(std::string_view a, std::string_view b) noexcept {
  volatile unsigned char diff =
      static_cast<unsigned char>((a.size() ^ b.size()) != 0 ? 1 : 0);
  const size_t n = a.size() < b.size() ? a.size() : b.size();
  for (size_t i = 0; i < n; ++i) {
    diff |= static_cast<unsigned char>(a[i]) ^ static_cast<unsigned char>(b[i]);
  }
  return diff == 0 && a.size() == b.size();
}

}  // namespace umminfer::util
