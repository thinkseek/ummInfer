// SPDX-License-Identifier: Apache-2.0
#include "serve/online_attestation.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <system_error>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "umminfer/util/codec.hpp"

namespace umminfer::openai_detail {
namespace {

namespace fs = std::filesystem;
constexpr size_t kMaxDocumentBytes = 4u << 20;
constexpr int64_t kMaxFutureIdentitySkewMs = 60000;
constexpr int64_t kMaxIdentityAgeLimitMs = 7LL * 24 * 60 * 60 * 1000;

class AttestationError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

[[noreturn]] void fail(const std::string& message) {
  throw AttestationError("online rollout attestation: " + message);
}

void require(bool condition, const std::string& message) {
  if (!condition) fail(message);
}

void exact_keys(const OnlineJson& value, const std::set<std::string>& keys,
                const std::string& label) {
  require(value.is_object(), label + " must be an object");
  std::set<std::string> actual;
  for (auto it = value.begin(); it != value.end(); ++it) actual.insert(it.key());
  require(actual == keys, label + " has missing or unknown fields");
}

// Matches online_rollout._text: one to 256 Unicode scalar values, excluding
// C0 controls and DEL. Header values and ids therefore cannot inject framing.
void validate_text(const OnlineJson& value, const std::string& label) {
  require(value.is_string(), label + " must be a string");
  const std::string& text = value.get_ref<const std::string&>();
  size_t count = 0;
  for (size_t i = 0; i < text.size();) {
    const unsigned char lead = static_cast<unsigned char>(text[i]);
    size_t length = 0;
    uint32_t codepoint = 0;
    if (lead < 0x80) {
      length = 1;
      codepoint = lead;
    } else if (lead >= 0xC2 && lead <= 0xDF) {
      length = 2;
      codepoint = lead & 0x1F;
    } else if (lead >= 0xE0 && lead <= 0xEF) {
      length = 3;
      codepoint = lead & 0x0F;
    } else if (lead >= 0xF0 && lead <= 0xF4) {
      length = 4;
      codepoint = lead & 0x07;
    } else {
      fail(label + " must contain valid UTF-8");
    }
    require(i + length <= text.size(), label + " must contain valid UTF-8");
    for (size_t offset = 1; offset < length; ++offset) {
      const unsigned char next = static_cast<unsigned char>(text[i + offset]);
      require((next & 0xC0) == 0x80, label + " must contain valid UTF-8");
      codepoint = (codepoint << 6) | (next & 0x3F);
    }
    if ((length == 3 && codepoint < 0x800) ||
        (length == 4 && codepoint < 0x10000) ||
        (codepoint >= 0xD800 && codepoint <= 0xDFFF) || codepoint > 0x10FFFF) {
      fail(label + " must contain valid UTF-8");
    }
    require(codepoint >= 32 && codepoint != 127,
            label + " must not contain control characters");
    ++count;
    i += length;
  }
  require(count >= 1 && count <= 256,
          label + " must contain between 1 and 256 Unicode characters");
}

bool sha256_text(const std::string& value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](unsigned char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}

void validate_sha(const OnlineJson& value, const std::string& label) {
  require(value.is_string() &&
              sha256_text(value.get_ref<const std::string&>()),
          label + " must be one lowercase SHA-256 digest");
}

bool integer_value(const OnlineJson& value, int64_t& out) {
  if (!value.is_number_integer()) return false;
  if (value.is_number_unsigned()) {
    const uint64_t raw = value.get<uint64_t>();
    if (raw > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return false;
    }
  }
  out = value.get<int64_t>();
  return true;
}

bool finite_number(const OnlineJson& value, double& out) {
  if (!value.is_number()) return false;
  out = value.get<double>();
  return std::isfinite(out);
}

std::string python_float(double value) {
  if (!std::isfinite(value)) fail("canonical JSON requires finite numbers");
  if (value == 0.0) return std::signbit(value) ? "-0.0" : "0.0";

  const bool negative = std::signbit(value);
  if (negative) value = -value;
  char storage[128];
  const auto result = std::to_chars(storage, storage + sizeof(storage), value,
                                    std::chars_format::scientific);
  require(result.ec == std::errc(), "could not format a finite JSON number");
  const std::string scientific(storage, result.ptr);
  const size_t exponent_pos = scientific.find('e');
  require(exponent_pos != std::string::npos,
          "scientific JSON number has no exponent");
  std::string digits = scientific.substr(0, exponent_pos);
  digits.erase(std::remove(digits.begin(), digits.end(), '.'), digits.end());
  int exponent = 0;
  const std::string exponent_text = scientific.substr(exponent_pos + 1);
  const char* exponent_begin = exponent_text.data();
  const char* exponent_end = exponent_begin + exponent_text.size();
  if (exponent_begin != exponent_end && *exponent_begin == '+') ++exponent_begin;
  const auto parsed = std::from_chars(exponent_begin, exponent_end, exponent);
  require(parsed.ec == std::errc() && parsed.ptr == exponent_end &&
              exponent_begin != exponent_end,
          "scientific JSON number has an invalid exponent");

  std::string out;
  // Python repr uses fixed notation for decimal exponents [-4, 16). The
  // nlohmann default uses a different upper threshold, so dump() is insufficient.
  if (exponent >= -4 && exponent < 16) {
    if (exponent < 0) {
      out = "0." + std::string(static_cast<size_t>(-exponent - 1), '0') + digits;
    } else {
      const size_t integer_digits = static_cast<size_t>(exponent + 1);
      if (digits.size() <= integer_digits) {
        out = digits + std::string(integer_digits - digits.size(), '0') + ".0";
      } else {
        out = digits.substr(0, integer_digits) + "." +
              digits.substr(integer_digits);
      }
    }
  } else {
    out.assign(1, digits.front());
    if (digits.size() > 1) out += "." + digits.substr(1);
    out += "e";
    out += exponent < 0 ? "-" : "+";
    const int absolute = exponent < 0 ? -exponent : exponent;
    const std::string exponent_digits = std::to_string(absolute);
    if (exponent_digits.size() < 2) out += "0";
    out += exponent_digits;
  }
  return negative ? "-" + out : out;
}

void append_canonical(const OnlineJson& value, std::string& out, int depth) {
  require(depth <= 256, "canonical JSON nesting exceeds 256 levels");
  if (value.is_null()) {
    out += "null";
  } else if (value.is_boolean()) {
    out += value.get<bool>() ? "true" : "false";
  } else if (value.is_number_integer() || value.is_number_unsigned()) {
    out += value.dump();
  } else if (value.is_number_float()) {
    out += python_float(value.get<double>());
  } else if (value.is_string()) {
    out += value.dump(-1, ' ', false, OnlineJson::error_handler_t::strict);
  } else if (value.is_array()) {
    out.push_back('[');
    for (size_t i = 0; i < value.size(); ++i) {
      if (i != 0) out.push_back(',');
      append_canonical(value[i], out, depth + 1);
    }
    out.push_back(']');
  } else if (value.is_object()) {
    out.push_back('{');
    bool first = true;
    for (auto it = value.begin(); it != value.end(); ++it) {
      if (!first) out.push_back(',');
      first = false;
      out += OnlineJson(it.key()).dump(-1, ' ', false,
                                       OnlineJson::error_handler_t::strict);
      out.push_back(':');
      append_canonical(it.value(), out, depth + 1);
    }
    out.push_back('}');
  } else {
    fail("canonical JSON received a non-JSON value");
  }
}

class FileDescriptor {
 public:
  explicit FileDescriptor(const fs::path& path) {
    fd_ = ::open(path.c_str(), O_RDONLY | O_NOFOLLOW);
    if (fd_ < 0) fail("required local evidence file is missing or not a regular file");
    struct stat info {};
    if (::fstat(fd_, &info) != 0 || !S_ISREG(info.st_mode)) {
      ::close(fd_);
      fd_ = -1;
      fail("required local evidence path is not a regular file");
    }
  }
  ~FileDescriptor() {
    if (fd_ >= 0) ::close(fd_);
  }
  int get() const { return fd_; }

 private:
  int fd_ = -1;
};

std::string read_document(const fs::path& path) {
  FileDescriptor file(path);
  std::string out;
  std::array<char, 16384> buffer{};
  while (true) {
    const ssize_t count = ::read(file.get(), buffer.data(), buffer.size());
    require(count >= 0, "could not read local evidence");
    if (count == 0) break;
    out.append(buffer.data(), static_cast<size_t>(count));
    require(out.size() <= kMaxDocumentBytes, "evidence document exceeds 4 MiB");
  }
  return out;
}

struct VerifiedFile {
  std::string sha256;
  int64_t bytes = 0;
};

VerifiedFile verify_file(const fs::path& path, const OnlineJson& declared,
                         const std::string& label) {
  exact_keys(declared, {"sha256", "bytes"}, label);
  validate_sha(declared["sha256"], label + ".sha256");
  int64_t expected_bytes = 0;
  require(integer_value(declared["bytes"], expected_bytes) && expected_bytes >= 0,
          label + ".bytes must be a non-negative int64");
  FileDescriptor file(path);
  util::Sha256 digest;
  std::array<char, 1u << 16> buffer{};
  int64_t actual_bytes = 0;
  while (true) {
    const ssize_t count = ::read(file.get(), buffer.data(), buffer.size());
    require(count >= 0, "could not read model evidence");
    if (count == 0) break;
    require(actual_bytes <= std::numeric_limits<int64_t>::max() - count,
            "model evidence size overflows int64");
    actual_bytes += count;
    digest.update(std::string_view(buffer.data(), static_cast<size_t>(count)));
  }
  require(actual_bytes == expected_bytes, label + " byte length changed");
  const std::string actual_sha = digest.finish_hex();
  require(actual_sha == declared["sha256"].get<std::string>(),
          label + " SHA-256 changed");
  return {actual_sha, actual_bytes};
}

OnlineJson parse_document(const std::string& bytes, const std::string& label) {
  const auto value = OnlineJson::parse(bytes, nullptr, false);
  require(!value.is_discarded(), label + " is not valid JSON");
  return value;
}

fs::path resolve_document_path(const fs::path& config_dir,
                               const OnlineJson& value,
                               const std::string& label) {
  require(value.is_string() && !value.get_ref<const std::string&>().empty(),
          label + " must be a non-empty path string");
  fs::path path(value.get<std::string>());
  if (path.is_relative()) path = config_dir / path;
  return path.lexically_normal();
}

fs::path resolve_model_file(const fs::path& root, const std::string& relative,
                            const std::string& label) {
  require(!relative.empty(), label + " path must not be empty");
  const fs::path path(relative);
  require(!path.is_absolute() && path.lexically_normal() == path,
          label + " path must be normalized and relative");
  for (const auto& component : path) {
    require(component != ".." && component != "." && component != "/",
            label + " path escapes the model root");
  }
  return root / path;
}

void validate_plan(const OnlineJson& plan) {
  exact_keys(plan,
             {"schema", "batch_id", "behavior_policy", "sampling",
              "request_template", "reward", "split_seed", "eval_ratio",
              "transport", "groups"},
             "plan");
  require(plan["schema"] == "umminfer-online-grpo-plan-v1",
          "plan schema must be umminfer-online-grpo-plan-v1");
  validate_text(plan["batch_id"], "batch_id");
  (void)online_canonical_json(plan);

  const auto& behavior = plan["behavior_policy"];
  exact_keys(behavior,
             {"policy_id", "revision", "model_id", "model_revision",
              "model_sha256", "model_config_sha256"},
             "behavior_policy");
  for (const char* key : {"policy_id", "revision", "model_id", "model_revision"}) {
    validate_text(behavior[key], std::string("behavior_policy.") + key);
  }
  validate_sha(behavior["model_sha256"], "behavior_policy.model_sha256");
  validate_sha(behavior["model_config_sha256"],
               "behavior_policy.model_config_sha256");

  const auto& sampling = plan["sampling"];
  exact_keys(sampling, {"max_tokens", "temperature", "top_p"}, "sampling");
  int64_t max_tokens = 0;
  require(integer_value(sampling["max_tokens"], max_tokens) && max_tokens > 0,
          "sampling.max_tokens must be a positive int64");
  double number = 0.0;
  require(finite_number(sampling["temperature"], number) && number >= 0.0 &&
              number <= 2.0,
          "sampling.temperature is outside [0, 2]");
  require(finite_number(sampling["top_p"], number) && number > 0.0 && number <= 1.0,
          "sampling.top_p is outside (0, 1]");

  require(plan["request_template"].is_object(),
          "request_template must be an object");
  static const std::set<std::string> reserved = {
      "model", "messages", "stream", "session_id", "seed",
      "max_tokens", "temperature", "top_p"};
  for (auto it = plan["request_template"].begin();
       it != plan["request_template"].end(); ++it) {
    require(!reserved.count(it.key()),
            "request_template cannot replace scheduled request fields");
  }
  exact_keys(plan["reward"], {"spec"}, "reward");
  validate_text(plan["reward"]["spec"], "reward.spec");
  require(plan["reward"]["spec"].get<std::string>().find(':') != std::string::npos,
          "reward.spec must use module:attribute form");
  validate_text(plan["split_seed"], "split_seed");
  require(finite_number(plan["eval_ratio"], number) && number >= 0.0 && number <= 1.0,
          "eval_ratio must be in [0, 1]");

  const auto& transport = plan["transport"];
  exact_keys(transport,
             {"timeout_s", "max_attempts", "retry_statuses", "retry_transport",
              "retry_timeouts"},
             "transport");
  require(finite_number(transport["timeout_s"], number) && number > 0.0,
          "transport.timeout_s must be finite and positive");
  int64_t max_attempts = 0;
  require(integer_value(transport["max_attempts"], max_attempts) &&
              max_attempts >= 1 && max_attempts <= 8,
          "transport.max_attempts must be in [1, 8]");
  require(transport["retry_statuses"].is_array(),
          "transport.retry_statuses must be an array");
  std::set<int64_t> statuses;
  for (const auto& status : transport["retry_statuses"]) {
    int64_t code = 0;
    require(integer_value(status, code) && code >= 400 && code <= 599,
            "retry statuses must be HTTP error integers");
    require(statuses.insert(code).second, "retry statuses must be unique");
  }
  require(transport["retry_transport"].is_boolean() &&
              transport["retry_timeouts"].is_boolean(),
          "transport retry switches must be boolean");

  require(plan["groups"].is_array() && !plan["groups"].empty(),
          "plan requires at least one prompt group");
  std::set<std::string> prompt_ids;
  size_t group_size = 0;
  for (const auto& group : plan["groups"]) {
    exact_keys(group, {"prompt_id", "messages", "expected_visual_modalities", "seeds"},
               "group");
    validate_text(group["prompt_id"], "group.prompt_id");
    require(prompt_ids.insert(group["prompt_id"].get<std::string>()).second,
            "prompt_id values must be unique");
    require(group["messages"].is_array() && !group["messages"].empty(),
            "group.messages must be non-empty");
    for (const auto& message : group["messages"]) {
      require(message.is_object() && message.contains("role") &&
                  message.contains("content") && message["role"].is_string() &&
                  !message["role"].get_ref<const std::string&>().empty(),
              "group contains an invalid message");
    }
    require(group["expected_visual_modalities"].is_array(),
            "expected_visual_modalities must be an array");
    for (const auto& modality : group["expected_visual_modalities"]) {
      require(modality.is_string() &&
                  (modality == "image" || modality == "video"),
              "expected_visual_modalities accepts only image/video");
    }
    require(group["seeds"].is_array() && group["seeds"].size() >= 2,
            "each group needs at least two seeds");
    if (group_size == 0) group_size = group["seeds"].size();
    require(group["seeds"].size() == group_size,
            "all online GRPO groups must have the same group size");
    std::set<int64_t> seeds;
    for (const auto& seed : group["seeds"]) {
      int64_t parsed = 0;
      require(integer_value(seed, parsed), "seeds must be signed int64 values");
      require(seeds.insert(parsed).second, "seeds must be unique within a group");
    }
  }
}

std::string field_for_header(const std::string& key,
                             const OnlineScheduleEntry& entry, int attempt) {
  if (key == "contract") return "1";
  if (key == "generation") return "fresh";
  if (key == "batch_id") return entry.batch_id;
  if (key == "group_id") return entry.group_id;
  if (key == "rollout_id") return entry.rollout_id;
  if (key == "rollout_index") return std::to_string(entry.rollout_index);
  if (key == "attempt") return std::to_string(attempt);
  if (key == "request_sha256") return entry.request_sha256;
  if (key == "behavior_policy_sha256") return entry.behavior_policy_sha256;
  if (key == "behavior_policy_revision") return entry.behavior_policy_revision;
  if (key == "model_id") return entry.model_id;
  if (key == "model_revision") return entry.model_revision;
  if (key == "model_sha256") return entry.model_sha256;
  if (key == "model_config_sha256") return entry.model_config_sha256;
  if (key == "sampling_sha256") return entry.sampling_sha256;
  if (key == "behavior_evidence") return "not_exposed";
  fail("unknown response identity field");
}

bool reject_request(std::string& error, int& status, int code,
                    const std::string& message) {
  status = code;
  error = message;
  return false;
}

}  // namespace

const std::map<std::string, std::string>& online_response_headers() {
  static const std::map<std::string, std::string> headers = {
      {"contract", "x-umminfer-online-rollout-contract"},
      {"generation", "x-umminfer-rollout-generation"},
      {"batch_id", "x-umminfer-rollout-batch-id"},
      {"group_id", "x-umminfer-rollout-group-id"},
      {"rollout_id", "x-umminfer-rollout-id"},
      {"rollout_index", "x-umminfer-rollout-index"},
      {"attempt", "x-umminfer-rollout-attempt"},
      {"request_sha256", "x-umminfer-request-sha256"},
      {"behavior_policy_sha256", "x-umminfer-behavior-policy-sha256"},
      {"behavior_policy_revision", "x-umminfer-behavior-policy-revision"},
      {"model_id", "x-umminfer-behavior-model-id"},
      {"model_revision", "x-umminfer-behavior-model-revision"},
      {"model_sha256", "x-umminfer-behavior-model-sha256"},
      {"model_config_sha256", "x-umminfer-behavior-model-config-sha256"},
      {"sampling_sha256", "x-umminfer-sampling-sha256"},
      {"behavior_evidence", "x-umminfer-behavior-evidence"},
  };
  return headers;
}

std::string online_canonical_json(const OnlineJson& value) {
  std::string out;
  append_canonical(value, out, 0);
  return out;
}

std::string online_request_sha256(const OnlineJson& body) {
  return util::sha256_hex(online_canonical_json(body));
}

std::vector<OnlineScheduleEntry> online_derive_schedule(const OnlineJson& plan) {
  validate_plan(plan);
  const auto& behavior = plan["behavior_policy"];
  const std::string behavior_sha = util::sha256_hex(online_canonical_json(behavior));
  const std::string sampling_sha =
      util::sha256_hex(online_canonical_json(plan["sampling"]));
  std::vector<OnlineScheduleEntry> schedule;
  for (size_t group_index = 0; group_index < plan["groups"].size(); ++group_index) {
    require(group_index <= static_cast<size_t>(std::numeric_limits<int>::max()),
            "too many rollout groups");
    const auto& group = plan["groups"][group_index];
    const OnlineJson group_identity = {
        {"batch_id", plan["batch_id"]},
        {"group_index", group_index},
        {"prompt_id", group["prompt_id"]},
        {"messages", group["messages"]},
        {"seeds", group["seeds"]},
        {"request_template", plan["request_template"]},
        {"behavior_policy_sha256", behavior_sha},
    };
    const std::string group_id =
        util::sha256_hex(online_canonical_json(group_identity)).substr(0, 32);
    for (size_t rollout_index = 0; rollout_index < group["seeds"].size();
         ++rollout_index) {
      require(rollout_index <= static_cast<size_t>(std::numeric_limits<int>::max()),
              "too many rollouts in a group");
      const std::string session_id =
          "online-" + group_id + "-" + std::to_string(rollout_index);
      OnlineJson body = plan["request_template"];
      body["model"] = behavior["model_id"];
      body["messages"] = group["messages"];
      body["stream"] = true;
      body["session_id"] = session_id;
      for (auto it = plan["sampling"].begin(); it != plan["sampling"].end(); ++it) {
        body[it.key()] = it.value();
      }
      body["seed"] = group["seeds"][rollout_index];
      const std::string request_sha = online_request_sha256(body);
      const OnlineJson rollout_identity = {
          {"group_id", group_id},
          {"rollout_index", rollout_index},
          {"seed", group["seeds"][rollout_index]},
          {"request_sha256", request_sha},
          {"behavior_policy_sha256", behavior_sha},
      };
      OnlineScheduleEntry entry;
      entry.batch_id = plan["batch_id"].get<std::string>();
      entry.group_id = group_id;
      entry.rollout_id =
          util::sha256_hex(online_canonical_json(rollout_identity)).substr(0, 32);
      entry.rollout_index = static_cast<int>(rollout_index);
      entry.body = std::move(body);
      entry.request_sha256 = request_sha;
      entry.behavior_policy_sha256 = behavior_sha;
      entry.behavior_policy_revision = behavior["revision"].get<std::string>();
      entry.model_id = behavior["model_id"].get<std::string>();
      entry.model_revision = behavior["model_revision"].get<std::string>();
      entry.model_sha256 = behavior["model_sha256"].get<std::string>();
      entry.model_config_sha256 = behavior["model_config_sha256"].get<std::string>();
      entry.sampling_sha256 = sampling_sha;
      schedule.push_back(std::move(entry));
    }
  }
  return schedule;
}

std::unique_ptr<OnlineRolloutServer> OnlineRolloutServer::load(
    const std::string& config_path, const std::string& served_model_name,
    int64_t now_epoch_ms) {
  const fs::path config_file(config_path);
  const fs::path config_dir = config_file.parent_path();
  const OnlineJson config = parse_document(read_document(config_file), "server config");
  exact_keys(config,
             {"schema", "plan_path", "plan_sha256", "native_identity_path",
              "native_identity_sha256", "native_stage_name", "max_identity_age_ms"},
             "server config");
  require(config["schema"] == "umminfer-online-rollout-server-v1",
          "server config schema is unsupported");
  validate_sha(config["plan_sha256"], "plan_sha256");
  validate_sha(config["native_identity_sha256"], "native_identity_sha256");
  validate_text(config["native_stage_name"], "native_stage_name");
  int64_t max_identity_age = 0;
  require(integer_value(config["max_identity_age_ms"], max_identity_age) &&
              max_identity_age >= 1 && max_identity_age <= kMaxIdentityAgeLimitMs,
          "max_identity_age_ms must be in [1, 604800000]");

  const fs::path plan_path =
      resolve_document_path(config_dir, config["plan_path"], "plan_path");
  const std::string plan_bytes = read_document(plan_path);
  require(util::sha256_hex(plan_bytes) == config["plan_sha256"].get<std::string>(),
          "trusted plan exact-byte SHA-256 mismatch");
  const OnlineJson plan = parse_document(plan_bytes, "trusted plan");

  const fs::path identity_path = resolve_document_path(
      config_dir, config["native_identity_path"], "native_identity_path");
  const std::string identity_bytes = read_document(identity_path);
  require(util::sha256_hex(identity_bytes) ==
              config["native_identity_sha256"].get<std::string>(),
          "native identity exact-byte SHA-256 mismatch");
  const OnlineJson identity = parse_document(identity_bytes, "native identity");
  exact_keys(identity,
             {"schema", "stage_name", "state", "loaded_at_epoch_ms", "model_id",
              "model_revision", "model_root", "config", "weight_files",
              "model_sha256"},
             "native identity");
  require(identity["schema"] == "umminfer-native-stage-identity-v1",
          "native identity schema is unsupported");
  require(identity["stage_name"] == config["native_stage_name"],
          "native identity stage name mismatch");
  require(identity["state"] == "loaded",
          "native stage has not declared a completed model load");
  int64_t loaded_at = 0;
  require(integer_value(identity["loaded_at_epoch_ms"], loaded_at) && loaded_at >= 0,
          "loaded_at_epoch_ms must be a non-negative int64");
  require(loaded_at <= now_epoch_ms + kMaxFutureIdentitySkewMs,
          "native identity is from the future");
  require(now_epoch_ms >= loaded_at && now_epoch_ms - loaded_at <= max_identity_age,
          "native identity is stale");
  validate_text(identity["model_id"], "native identity.model_id");
  validate_text(identity["model_revision"], "native identity.model_revision");
  require(identity["model_root"].is_string() &&
              !identity["model_root"].get_ref<const std::string&>().empty(),
          "native identity.model_root must be a non-empty path");
  fs::path model_root(identity["model_root"].get<std::string>());
  if (model_root.is_relative()) model_root = identity_path.parent_path() / model_root;
  model_root = model_root.lexically_normal();

  exact_keys(identity["config"], {"path", "sha256", "bytes"}, "native identity.config");
  require(identity["config"]["path"].is_string(),
          "native identity.config.path must be a string");
  const fs::path model_config = resolve_model_file(
      model_root, identity["config"]["path"].get<std::string>(), "model config");
  const OnlineJson config_file_identity = {
      {"sha256", identity["config"]["sha256"]},
      {"bytes", identity["config"]["bytes"]},
  };
  verify_file(model_config, config_file_identity, "model config");

  require(identity["weight_files"].is_object() && !identity["weight_files"].empty() &&
              identity["weight_files"].size() <= 4096,
          "native identity requires 1..4096 available weight files");
  for (auto it = identity["weight_files"].begin();
       it != identity["weight_files"].end(); ++it) {
    const fs::path weight = resolve_model_file(model_root, it.key(), "weight file");
    verify_file(weight, it.value(), "weight file");
  }
  const std::string computed_weights_sha =
      util::sha256_hex(online_canonical_json(identity["weight_files"]));
  validate_sha(identity["model_sha256"], "native identity.model_sha256");
  require(computed_weights_sha == identity["model_sha256"].get<std::string>(),
          "native identity weight-set SHA-256 mismatch");

  auto server = std::unique_ptr<OnlineRolloutServer>(new OnlineRolloutServer());
  server->schedule_ = online_derive_schedule(plan);
  server->native_stage_name_ = config["native_stage_name"].get<std::string>();
  int64_t max_attempts = 0;
  require(integer_value(plan["transport"]["max_attempts"], max_attempts),
          "invalid max_attempts");
  server->max_attempts_ = static_cast<int>(max_attempts);
  const auto& behavior = plan["behavior_policy"];
  require(behavior["model_id"] == identity["model_id"] &&
              behavior["model_id"].get<std::string>() == served_model_name,
          "plan, served model, and native model id do not match");
  require(behavior["model_revision"] == identity["model_revision"],
          "plan and native model revision do not match");
  require(behavior["model_config_sha256"] == identity["config"]["sha256"],
          "plan and native model config SHA-256 do not match");
  require(behavior["model_sha256"] == identity["model_sha256"],
          "plan and native weight-set SHA-256 do not match");
  server->behavior_policy_sha256_ =
      util::sha256_hex(online_canonical_json(behavior));
  server->behavior_policy_revision_ = behavior["revision"].get<std::string>();
  server->model_id_ = behavior["model_id"].get<std::string>();
  server->model_revision_ = behavior["model_revision"].get<std::string>();
  server->model_sha256_ = behavior["model_sha256"].get<std::string>();
  server->model_config_sha256_ = behavior["model_config_sha256"].get<std::string>();
  server->sampling_sha256_ =
      util::sha256_hex(online_canonical_json(plan["sampling"]));
  for (size_t index = 0; index < server->schedule_.size(); ++index) {
    const auto& entry = server->schedule_[index];
    require(server->by_rollout_id_.emplace(entry.rollout_id, index).second,
            "derived rollout ids are not unique");
  }
  return server;
}

bool OnlineRolloutServer::request_marked(const httplib::Request& req) {
  for (const auto& [key, header] : online_response_headers()) {
    (void)key;
    if (req.get_header_value_count(header) != 0) return true;
  }
  return false;
}

bool OnlineRolloutServer::bind_request(const httplib::Request& req,
                                       const OnlineJson& body,
                                       OnlineRequestBinding& binding,
                                       std::string& error, int& http_status) {
  for (const auto& [key, header] : online_response_headers()) {
    const size_t expected_count = key == "behavior_evidence" ? 0 : 1;
    if (req.get_header_value_count(header) != expected_count) {
      return reject_request(error, http_status, 400,
                            "missing or duplicate online rollout header: " + header);
    }
  }
  if (req.get_header_value("x-umminfer-online-rollout-contract") != "1" ||
      req.get_header_value("x-umminfer-rollout-generation") != "fresh") {
    return reject_request(error, http_status, 400,
                          "online rollout requires contract=1 and generation=fresh");
  }
  const std::string rollout_id = req.get_header_value("x-umminfer-rollout-id");
  const auto found = by_rollout_id_.find(rollout_id);
  if (found == by_rollout_id_.end()) {
    return reject_request(error, http_status, 400,
                          "online rollout id is not present in the trusted schedule");
  }
  const std::string attempt_text = req.get_header_value("x-umminfer-rollout-attempt");
  int attempt = -1;
  const auto parsed = std::from_chars(attempt_text.data(),
                                      attempt_text.data() + attempt_text.size(), attempt);
  if (parsed.ec != std::errc() ||
      parsed.ptr != attempt_text.data() + attempt_text.size() || attempt < 0 ||
      attempt >= max_attempts_ || attempt_text != std::to_string(attempt)) {
    return reject_request(error, http_status, 400,
                          "online rollout attempt is outside the trusted retry policy");
  }
  const auto& entry = schedule_[found->second];
  for (const auto& [key, header] : online_response_headers()) {
    if (key == "behavior_evidence") continue;
    if (req.get_header_value(header) != field_for_header(key, entry, attempt)) {
      return reject_request(error, http_status, 400,
                            "online rollout request identity mismatch: " + key);
    }
  }
  if (online_request_sha256(body) != entry.request_sha256 || body != entry.body) {
    return reject_request(error, http_status, 400,
                          "canonical request body or sampling drift");
  }
  {
    std::lock_guard<std::mutex> lock(attempts_mu_);
    if (!consumed_attempts_.emplace(entry.rollout_id, attempt).second) {
      binding = OnlineRequestBinding{entry, attempt};
      http_status = 409;
      error = "duplicate online rollout attempt binding";
      return false;
    }
  }
  binding = OnlineRequestBinding{entry, attempt};
  return true;
}

void OnlineRolloutServer::write_response_headers(
    httplib::Response& res, const OnlineRequestBinding& binding) const {
  for (const auto& [key, header] : online_response_headers()) {
    res.set_header(header, field_for_header(key, binding.entry, binding.attempt));
  }
}

}  // namespace umminfer::openai_detail
