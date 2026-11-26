// SPDX-License-Identifier: Apache-2.0
// Server-derived online-rollout identity and HTTP response attestation.
#pragma once

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace umminfer::openai_detail {

using OnlineJson = nlohmann::json;

// Exact public response-header names shared with
// python/umminfer_online_rollout.py. attestation_sha256 is deliberately not a
// wire header in the accepted contract: the trainer recomputes it using the
// observed HTTP status.
const std::map<std::string, std::string>& online_response_headers();

// Canonical compact, key-sorted, unescaped-Unicode UTF-8 JSON used by the
// trainer. Float rendering follows Python repr/json semantics, including its
// fixed/scientific thresholds and signed zero. Non-finite numbers fail.
std::string online_canonical_json(const OnlineJson& value);
std::string online_request_sha256(const OnlineJson& body);

struct OnlineScheduleEntry {
  std::string batch_id;
  std::string group_id;
  std::string rollout_id;
  int rollout_index = 0;
  OnlineJson body;
  std::string request_sha256;
  std::string behavior_policy_sha256;
  std::string behavior_policy_revision;
  std::string model_id;
  std::string model_revision;
  std::string model_sha256;
  std::string model_config_sha256;
  std::string sampling_sha256;
};

// Derive the accepted group/candidate order and request bodies from a trusted
// plan. This does not read client headers and never loads a reward or model.
std::vector<OnlineScheduleEntry> online_derive_schedule(const OnlineJson& plan);

struct OnlineRequestBinding {
  OnlineScheduleEntry entry;
  int attempt = 0;
};

class OnlineRolloutServer {
 public:
  // Loads and validates all trusted configuration before bind(). The model
  // config and every available weight file are independently re-hashed. The
  // injectable clock makes stale-evidence tests deterministic.
  static std::unique_ptr<OnlineRolloutServer> load(
      const std::string& config_path, const std::string& served_model_name,
      int64_t now_epoch_ms);

  const std::string& native_stage_name() const { return native_stage_name_; }

  // True when any contract-owned request header is present. A marked request
  // cannot silently fall back to an unattested default response.
  static bool request_marked(const httplib::Request& req);

  // Validates every request header against a server-derived schedule entry and
  // atomically consumes its (rollout_id, attempt) slot. No identity is copied
  // from the request into the returned binding.
  bool bind_request(const httplib::Request& req, const OnlineJson& body,
                    OnlineRequestBinding& binding, std::string& error,
                    int& http_status);

  void write_response_headers(httplib::Response& res,
                              const OnlineRequestBinding& binding) const;

 private:
  std::vector<OnlineScheduleEntry> schedule_;
  std::map<std::string, size_t> by_rollout_id_;
  std::string behavior_policy_sha256_;
  std::string behavior_policy_revision_;
  std::string model_id_;
  std::string model_revision_;
  std::string model_sha256_;
  std::string model_config_sha256_;
  std::string sampling_sha256_;
  std::string native_stage_name_;
  int max_attempts_ = 1;
  std::mutex attempts_mu_;
  std::set<std::pair<std::string, int>> consumed_attempts_;
};

}  // namespace umminfer::openai_detail
