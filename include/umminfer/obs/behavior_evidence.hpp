// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "umminfer/proto/wire.hpp"

namespace umminfer::obs {

inline constexpr const char* kBehaviorLogprobsParam = "behavior_logprobs";
inline constexpr const char* kBehaviorNarrativeParam = "behavior_narrative";
inline constexpr const char* kBehaviorLogprobsResultKey = "behavior";
inline constexpr const char* kBehaviorLogprobsFormat =
    "umminfer-behavior-logprobs-v1";

// True only for the explicit boolean request-parameter opt-in.
bool behavior_logprobs_requested(const Value::Map& params);

// Canonical same-buffer identity for the three files parsed by native Qwen.
std::string loaded_tokenizer_sha256(
    const std::vector<std::pair<std::string, std::string_view>>& files);

// Builds the accepted five-field block, or nil when any value is invalid. The
// arrays cover the complete emitted non-EOS native sequence; empty evidence is
// not a valid COVERED response.
Value behavior_logprobs_value(const std::vector<int64_t>& token_ids,
                              const std::vector<double>& logprobs,
                              const std::vector<int64_t>& mask,
                              const std::string& tokenizer_sha256);

// Validates and copies a terminal native result's behavior block. Missing,
// malformed, truncated, empty, or count-mismatched evidence returns nullopt;
// callers must attest not_exposed and omit the public member in that case.
std::optional<Value> behavior_logprobs_projection(const Value& terminal_result);

}  // namespace umminfer::obs
