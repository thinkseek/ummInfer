// SPDX-License-Identifier: Apache-2.0
#include "umminfer/obs/behavior_evidence.hpp"

#include <algorithm>
#include <cmath>
#include <string_view>

#include "umminfer/util/codec.hpp"

namespace umminfer::obs {
namespace {

constexpr size_t kMaxBehaviorTokens = 65536;

bool valid_sha256(std::string_view digest) {
  if (digest.size() != 64) return false;
  for (const char c : digest) {
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
  }
  return true;
}

bool valid_arrays(const Value::Array& ids, const Value::Array& logprobs,
                  const Value::Array& mask) {
  if (ids.empty() || ids.size() != logprobs.size() || ids.size() != mask.size() ||
      ids.size() > kMaxBehaviorTokens) {
    return false;
  }
  bool supervised = false;
  for (size_t i = 0; i < ids.size(); ++i) {
    if (!ids[i].is_int() || ids[i].as_int() < 0 ||
        // Qwen3.8 terminal markers are deliberately absent, never masked-in.
        ids[i].as_int() == 248044 || ids[i].as_int() == 248046 ||
        !logprobs[i].is_float() || !std::isfinite(logprobs[i].as_double()) ||
        logprobs[i].as_double() > 0.0 || !mask[i].is_int() ||
        (mask[i].as_int() != 0 && mask[i].as_int() != 1)) {
      return false;
    }
    supervised = supervised || mask[i].as_int() == 1;
  }
  return supervised;
}

}  // namespace

std::string loaded_tokenizer_sha256(
    const std::vector<std::pair<std::string, std::string_view>>& files) {
  if (files.empty()) return {};
  auto sorted = files;
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
    return a.first < b.first;
  });
  std::string manifest = "umminfer-qwen38-loaded-tokenizer-v1\n";
  std::string previous;
  for (const auto& [name, bytes] : sorted) {
    if (name.empty() || name == previous || name.find('\n') != std::string::npos) {
      return {};
    }
    manifest += name + "\n" + std::to_string(bytes.size()) + "\n" +
                util::sha256_hex(bytes) + "\n";
    previous = name;
  }
  return util::sha256_hex(manifest);
}

bool behavior_logprobs_requested(const Value::Map& params) {
  const auto it = params.find(kBehaviorLogprobsParam);
  return it != params.end() && it->second.is_bool() && it->second.as_bool();
}

Value behavior_logprobs_value(const std::vector<int64_t>& token_ids,
                              const std::vector<double>& logprobs,
                              const std::vector<int64_t>& mask,
                              const std::string& tokenizer_sha256) {
  Value::Array ids;
  Value::Array scores;
  Value::Array masks;
  ids.reserve(token_ids.size());
  scores.reserve(logprobs.size());
  masks.reserve(mask.size());
  for (const int64_t id : token_ids) ids.emplace_back(id);
  for (const double value : logprobs) scores.emplace_back(value);
  for (const int64_t bit : mask) masks.emplace_back(bit);
  if (!valid_arrays(ids, scores, masks) || !valid_sha256(tokenizer_sha256)) {
    return {};
  }
  return Value(Value::Map{
      {"format", Value(kBehaviorLogprobsFormat)},
      {"tokenizer_sha256", Value(tokenizer_sha256)},
      {"token_ids", Value(std::move(ids))},
      {"logprobs", Value(std::move(scores))},
      {"mask", Value(std::move(masks))},
  });
}

std::optional<Value> behavior_logprobs_projection(const Value& terminal_result) {
  if (!terminal_result.is_map()) return std::nullopt;
  const Value* finish = terminal_result.find("finish_reason");
  if (finish == nullptr || !finish->is_str() || finish->as_str() != "stop") {
    return std::nullopt;
  }
  const Value* count = terminal_result.find("generated_token_count");
  if (count == nullptr || !count->is_int() || count->as_int() <= 0) {
    return std::nullopt;
  }
  const Value* evidence = terminal_result.find(kBehaviorLogprobsResultKey);
  if (evidence == nullptr || !evidence->is_map() ||
      evidence->as_map().size() != 5) {
    return std::nullopt;
  }
  const Value* format = evidence->find("format");
  const Value* tokenizer = evidence->find("tokenizer_sha256");
  const Value* ids = evidence->find("token_ids");
  const Value* logprobs = evidence->find("logprobs");
  const Value* mask = evidence->find("mask");
  if (format == nullptr || !format->is_str() ||
      format->as_str() != kBehaviorLogprobsFormat || tokenizer == nullptr ||
      !tokenizer->is_str() || !valid_sha256(tokenizer->as_str()) ||
      ids == nullptr || !ids->is_array() || logprobs == nullptr ||
      !logprobs->is_array() || mask == nullptr || !mask->is_array() ||
      !valid_arrays(ids->as_array(), logprobs->as_array(), mask->as_array()) ||
      static_cast<int64_t>(ids->as_array().size()) != count->as_int()) {
    return std::nullopt;
  }
  return *evidence;
}

}  // namespace umminfer::obs
