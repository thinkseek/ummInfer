// SPDX-License-Identifier: Apache-2.0
// Fixed-upstream HTTP/SSE proxy (DashScope Qwen-Omni and SGLang image/video)
// extracted from OpenAIApi so the local stage/unified HTTP surface has zero
// dependency on client/http.hpp.
#pragma once

#include <httplib.h>

#include <chrono>
#include <mutex>
#include <string>

#include "umminfer/serve/openai_api.hpp"

namespace umminfer::serve_detail {

class UpstreamProxy {
 public:
  explicit UpstreamProxy(const ApiConfig& config);

  void reject_route(httplib::Response& res, const char* route) const;
  void reject(httplib::Response& res, int status, const std::string& message,
              const char* type) const;

  void proxy_qwen_chat(const httplib::Request& req, httplib::Response& res);
  void handle_image_generation(const httplib::Request& req,
                               httplib::Response& res);
  void handle_image_edit(const httplib::Request& req, httplib::Response& res);
  void handle_video_create(const httplib::Request& req, httplib::Response& res);
  void handle_video_status(const httplib::Request& req,
                           const std::string& video_id, httplib::Response& res);
  void handle_video_content(const httplib::Request& req,
                            const std::string& video_id,
                            httplib::Response& res);
  void handle_video_delete(const httplib::Request& req,
                           const std::string& video_id, httplib::Response& res);

 private:
  bool check_circuit_breaker(httplib::Response& res);
  void record_upstream_outcome(bool trip_failure);

  ApiConfig config_;
  std::string upstream_base_url_;

  std::mutex cb_mu_;
  int consecutive_failures_ = 0;
  std::chrono::steady_clock::time_point cb_open_until_{};
};

}  // namespace umminfer::serve_detail
