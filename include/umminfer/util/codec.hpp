// SPDX-License-Identifier: Apache-2.0
// Shared binary/text codecs and constant-time secret comparison.
//
// Centralizes Base64, FIPS 180-4 SHA-256, UTF-8 sanitization, and constant-time
// equality so serve, realtime, client, and native stages share one audited
// implementation with no reverse dependency on src/client/http.hpp.
#pragma once

#include <cstddef>
#include <limits>
#include <string>
#include <string_view>

namespace umminfer::util {

// Canonical RFC 4648 Base64 encoder.
std::string base64_encode(std::string_view bytes);

// Strict RFC 4648 Base64 decoder. Rejects non-multiple-of-4 lengths (including
// empty when require_nonempty is true), invalid characters, misplaced padding,
// non-canonical trailing bits, and decoded outputs larger than max_bytes by
// throwing ProtocolError.
std::string base64_decode(
    std::string_view text,
    size_t max_bytes = std::numeric_limits<size_t>::max(),
    bool require_nonempty = false);

// FIPS 180-4 SHA-256 hex digest (lowercase, 64 characters).
std::string sha256_hex(std::string_view bytes);

// Replaces invalid/overlong/surrogate UTF-8 sequences with '?' so untrusted raw
// bytes can be embedded safely in JSON error payloads.
std::string utf8_sanitized(std::string_view text);

// Constant-time byte string equality for API keys and bearer credentials.
bool constant_time_equals(std::string_view a, std::string_view b) noexcept;

}  // namespace umminfer::util
