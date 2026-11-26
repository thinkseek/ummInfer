// SPDX-License-Identifier: Apache-2.0
#include "openai_internal.hpp"

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <utility>

#include "umminfer/util/codec.hpp"

namespace umminfer {
namespace {

constexpr size_t kMaxHttpPayloadBytes = 64u << 20;
// How long one socket read may wait for the next byte. The body is read before
// any handler runs, so this value alone decides whether a slow upload survives.
// httplib's 5 s default aborts a 1 MB audio upload on any link that stalls for
// longer than that: measured on a loaded deployment, the server logged
// "Failed to read" with recv returning EAGAIN while the client's send had
// already completed. 30 s exceeds a worst-case TCP retransmission backoff and
// still reclaims a dead connection.
constexpr time_t kHttpReadStallSeconds = 30;

}  // namespace

using namespace openai_detail;

OpenAIApi::Impl::Impl(Coordinator& coordinator, ApiConfig config)
    : coordinator_(coordinator),
      config_(std::move(config)),
      proxy_(config_) {
  if (config_.served_model_name.empty()) {
    config_.served_model_name = config_.model;
  }
  if (!config_.online_rollout_config.empty()) {
    online_rollout_ = openai_detail::OnlineRolloutServer::load(
        config_.online_rollout_config, config_.served_model_name,
        openai_detail::now_epoch_ms());
  }
  server_.set_payload_max_length(kMaxHttpPayloadBytes);
  server_.set_read_timeout(kHttpReadStallSeconds, 0);
  const size_t bounded_in_flight = std::min(config_.max_in_flight, size_t{128});
  const size_t worker_count =
      std::clamp(bounded_in_flight * 2 + 8, size_t{16}, size_t{128});
  server_.new_task_queue = [worker_count] {
    return new httplib::ThreadPool(worker_count);
  };
  // Transport faults were invisible in the log: a connection dropped before
  // a response left no trace, so a client-seen EOF could not be attributed.
  // One line names the httplib error and the route; never the body.
  server_.set_error_logger([](const httplib::Error& error,
                              const httplib::Request* request) {
    std::cerr << "http error=" << httplib::to_string(error);
    if (request != nullptr) {
      std::cerr << " method=" << request->method << " path=" << request->path
                << " body_bytes=" << request->body.size();
    }
    std::cerr << "\n";
  });
}

void OpenAIApi::Impl::install_routes() {
  server_.set_pre_routing_handler(
      [this](const httplib::Request& req, httplib::Response& res) {
        if (config_.api_key.empty()) {
          return httplib::Server::HandlerResponse::Unhandled;
        }
        const std::string expected = "Bearer " + config_.api_key;
        const auto auth = req.get_header_value("Authorization");
        if (util::constant_time_equals(auth, expected)) {
          return httplib::Server::HandlerResponse::Unhandled;
        }
        res.status = 401;
        res.set_content(
            R"({"error":{"message":"invalid api key","type":"auth_error"}})",
            "application/json");
        return httplib::Server::HandlerResponse::Handled;
      });

  server_.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
    const size_t in_flight =
        unified_engine_
            ? unified_engine_->in_flight()
            : (!config_.upstream_kind.empty()
                   ? active_http_requests_.load(std::memory_order_relaxed)
                   : coordinator_.in_flight());
    json body = {{"status", "ok"}, {"in_flight", in_flight}};
    res.set_content(body.dump(), "application/json");
  });

  server_.Get("/metrics", [](const httplib::Request&, httplib::Response& res) {
    res.set_content(obs::Metrics::instance().render(),
                    "text/plain; version=0.0.4; charset=utf-8");
  });

  server_.Get("/ready", [this](const httplib::Request&, httplib::Response& res) {
    const bool entry_ready = !config_.external_entry_stage ||
                             coordinator_.stage_ready(config_.entry_stage);
    const bool ready = entry_ready &&
                       (!unified_engine_ ||
                        coordinator_.stage_ready(unified_vlm_stage_));
    res.status = ready ? 200 : 503;
    res.set_content(ready ? R"({"status":"ready"})" : R"({"status":"starting"})",
                    "application/json");
  });

  server_.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
    json body = {{"object", "list"},
                 {"data", {{{"id", config_.served_model_name},
                            {"object", "model"},
                            {"created", 0},
                            {"owned_by", "ummInfer"}}}}};
    res.set_content(body.dump(), "application/json");
  });

  server_.Post("/v1/chat/completions",
               [this](const httplib::Request& req, httplib::Response& res) {
                 if (config_.upstream_kind == "qwen") {
                   AdmissionGuard guard(active_http_requests_,
                                        config_.max_in_flight);
                   if (!guard.admitted()) {
                     reject_overloaded(res);
                     return;
                   }
                   proxy_.proxy_qwen_chat(req, res);
                   return;
                 }
                 if (!config_.upstream_kind.empty()) {
                   proxy_.reject_route(res, "chat completions");
                   return;
                 }
                 handle_chat(req, res);
               });

  server_.Post("/v1/audio/transcriptions",
               [this](const httplib::Request& req, httplib::Response& res) {
                 if (!config_.upstream_kind.empty()) {
                   proxy_.reject_route(res, "audio transcriptions");
                   return;
                 }
                 handle_transcription(req, res);
               });

  server_.Post("/v1/images/generations",
               [this](const httplib::Request& req, httplib::Response& res) {
                 if (config_.upstream_variant != "image") {
                   proxy_.reject_route(res, "image generation");
                   return;
                 }
                 AdmissionGuard guard(active_http_requests_,
                                      config_.max_in_flight);
                 if (!guard.admitted()) {
                   reject_overloaded(res);
                   return;
                 }
                 proxy_.handle_image_generation(req, res);
               });

  server_.Post("/v1/images/edits",
               [this](const httplib::Request& req, httplib::Response& res) {
                 if (config_.upstream_variant != "image") {
                   proxy_.reject_route(res, "image editing");
                   return;
                 }
                 AdmissionGuard guard(active_http_requests_,
                                      config_.max_in_flight);
                 if (!guard.admitted()) {
                   reject_overloaded(res);
                   return;
                 }
                 proxy_.handle_image_edit(req, res);
               });

  server_.Post("/v1/videos",
               [this](const httplib::Request& req, httplib::Response& res) {
                 if (config_.upstream_variant != "video") {
                   proxy_.reject_route(res, "video creation");
                   return;
                 }
                 AdmissionGuard guard(active_http_requests_,
                                      config_.max_in_flight);
                 if (!guard.admitted()) {
                   reject_overloaded(res);
                   return;
                 }
                 proxy_.handle_video_create(req, res);
               });

  server_.Get(R"(/v1/videos/([A-Za-z0-9_\-]{1,128})/content)",
              [this](const httplib::Request& req, httplib::Response& res) {
                if (config_.upstream_variant != "video") {
                  proxy_.reject_route(res, "video content");
                  return;
                }
                AdmissionGuard guard(active_http_requests_,
                                     config_.max_in_flight);
                if (!guard.admitted()) {
                  reject_overloaded(res);
                  return;
                }
                proxy_.handle_video_content(req, req.matches[1].str(), res);
              });

  server_.Get(R"(/v1/videos/([A-Za-z0-9_\-]{1,128}))",
              [this](const httplib::Request& req, httplib::Response& res) {
                if (config_.upstream_variant != "video") {
                  proxy_.reject_route(res, "video status");
                  return;
                }
                AdmissionGuard guard(active_http_requests_,
                                     config_.max_in_flight);
                if (!guard.admitted()) {
                  reject_overloaded(res);
                  return;
                }
                proxy_.handle_video_status(req, req.matches[1].str(), res);
              });

  server_.Delete(R"(/v1/videos/([A-Za-z0-9_\-]{1,128}))",
                 [this](const httplib::Request& req, httplib::Response& res) {
                   if (config_.upstream_variant != "video") {
                     proxy_.reject_route(res, "video deletion");
                     return;
                   }
                   AdmissionGuard guard(active_http_requests_,
                                        config_.max_in_flight);
                   if (!guard.admitted()) {
                     reject_overloaded(res);
                     return;
                   }
                   proxy_.handle_video_delete(req, req.matches[1].str(), res);
                 });

  server_.Post("/abort", [this](const httplib::Request& req, httplib::Response& res) {
    if (!config_.upstream_kind.empty()) {
      proxy_.reject(
          res, 501,
          "abort is not supported: requests are served by a fixed upstream",
          "unsupported_route");
      return;
    }
    const json body = json::parse(req.body, nullptr, false);
    if (body.is_discarded() || !body.is_object() ||
        !body.contains("request_id") || !body["request_id"].is_string() ||
        !valid_request_id_chars(body["request_id"].get_ref<const std::string&>())) {
      res.status = 400;
      res.set_content(R"({"error":{"message":"request_id is required"}})",
                      "application/json");
      return;
    }
    std::string session_id;
    if (body.contains("session_id") && !body["session_id"].is_null()) {
      if (!body["session_id"].is_string() ||
          body["session_id"].get_ref<const std::string&>().empty() ||
          body["session_id"].get_ref<const std::string&>().size() > 128) {
        res.status = 400;
        res.set_content(
            R"({"error":{"message":"session_id must be a nonempty string of at most 128 characters"}})",
            "application/json");
        return;
      }
      session_id = body["session_id"].get<std::string>();
    }
    const std::string request_id = body["request_id"].get<std::string>();
    if (unified_engine_) {
      const auto status =
          unified_engine_->cancel_for_session(request_id, session_id);
      if (status == UnifiedEngine::CancelStatus::kSessionMismatch) {
        res.status = 403;
        res.set_content(
            R"({"error":{"message":"request_id does not belong to session_id","type":"permission_error"}})",
            "application/json");
        return;
      }
      if (status == UnifiedEngine::CancelStatus::kNotFoundOrTerminal &&
          session_id.empty()) {
        coordinator_.abort(request_id);
      }
    } else {
      coordinator_.abort(request_id);
    }
    res.set_content(R"({"success":true})", "application/json");
  });

  server_.Post("/v1/audio/speech",
               [this](const httplib::Request& req, httplib::Response& res) {
                 handle_speech(req, res);
               });

  server_.Post("/v1/visual/render",
               [this](const httplib::Request& req, httplib::Response& res) {
                 handle_visual_render(req, res);
               });

  // Opt-in static hosting for the bundled browser client. cpp-httplib's mount
  // rejects a ".." segment in the URL and any path that resolves outside the
  // directory (a symlink escape), so no request reads past web_root.
  if (!config_.web_root.empty() &&
      !server_.set_mount_point("/", config_.web_root)) {
    std::fprintf(stderr, "web root is not a readable directory: %s\n",
                 config_.web_root.c_str());
  }
}

bool OpenAIApi::Impl::bind() {
  if (bound_) return true;
  if (online_rollout_ && !config_.upstream_kind.empty()) {
    std::fprintf(stderr,
                 "online rollout attestation cannot run behind a fixed upstream\n");
    return false;
  }
  const std::string config_error = validate_upstream_config(config_);
  if (!config_error.empty()) {
    std::fprintf(stderr, "invalid upstream configuration: %s\n",
                 config_error.c_str());
    return false;
  }
  if (config_.port == 0) {
    bound_port_ = server_.bind_to_any_port(config_.host);
    if (bound_port_ < 0) return false;
  } else {
    if (!server_.bind_to_port(config_.host, config_.port)) return false;
    bound_port_ = config_.port;
  }
  bound_ = true;
  return true;
}

bool OpenAIApi::Impl::listen() {
  if (!bound_ && !bind()) return false;
  return server_.listen_after_bind();
}

void OpenAIApi::Impl::stop() { server_.stop(); }

int OpenAIApi::Impl::bound_port() const { return bound_port_; }

void OpenAIApi::Impl::set_unified_forwarder(std::function<void(const UnifiedEvent&)> forwarder) {
  unified_forwarder_ = std::move(forwarder);
}

void OpenAIApi::Impl::set_unified_engine(UnifiedEngine& engine, std::string vlm_stage) {
  if (online_rollout_ && online_rollout_->native_stage_name() != vlm_stage) {
    throw ProtocolError(
        "online rollout attestation native stage does not match the unified VLM stage");
  }
  unified_engine_ = &engine;
  unified_vlm_stage_ = std::move(vlm_stage);
  engine.on_event = [this](const UnifiedEvent& event) { on_unified_event(event); };
}

void OpenAIApi::Impl::on_unified_event(const UnifiedEvent& event) {
  if (unified_forwarder_) {
    try {
      unified_forwarder_(event);
    } catch (...) {
    }
  }
  std::shared_ptr<SseSink> sink;
  {
    std::lock_guard<std::mutex> lock(streams_mu_);
    auto it = streams_.find(event.request_id);
    if (it != streams_.end()) sink = it->second;
    const bool terminal = event.kind == UnifiedEvent::Kind::kComplete ||
                          event.kind == UnifiedEvent::Kind::kError ||
                          event.kind == UnifiedEvent::Kind::kCancelled;
    if (terminal && it != streams_.end()) streams_.erase(it);
  }
  if (!sink) return;
  if (event.kind == UnifiedEvent::Kind::kAsrPartial ||
      event.kind == UnifiedEvent::Kind::kAsrFinal ||
      event.kind == UnifiedEvent::Kind::kVlmDelta ||
      event.kind == UnifiedEvent::Kind::kVisualSlot ||
      event.kind == UnifiedEvent::Kind::kArtifact ||
      event.kind == UnifiedEvent::Kind::kLegError) {
    if (event.kind == UnifiedEvent::Kind::kVlmDelta) {
      std::lock_guard<std::mutex> lock(sink->mu);
      sink->has_deltas = true;
    }
    sink->push(event);
    return;
  }
  {
    std::lock_guard<std::mutex> lock(sink->mu);
    sink->terminal = event;
  }
  if (event.kind == UnifiedEvent::Kind::kComplete) {
    bool has_deltas;
    {
      std::lock_guard<std::mutex> lock(sink->mu);
      has_deltas = sink->has_deltas;
      sink->finish_reason = result_finish_reason(event.result);
    }
    if (!has_deltas) sink->push(event.result);
    sink->finish();
    return;
  }
  if (event.kind == UnifiedEvent::Kind::kCancelled) {
    sink->finish();
    return;
  }
  sink->finish(event.error.empty()
                   ? std::string("request ") + to_string(event.kind)
                   : event.error);
}

void OpenAIApi::Impl::on_stream(const StreamMessage& msg) {
  if (unified_engine_ && unified_engine_->info(msg.request_id)) return;
  std::shared_ptr<SseSink> sink;
  {
    std::lock_guard<std::mutex> lock(streams_mu_);
    auto it = streams_.find(msg.request_id);
    if (it != streams_.end()) sink = it->second;
  }
  if (sink) sink->push(msg.chunk, true);
}

void OpenAIApi::Impl::on_complete(const RequestInfo& info) {
  if (unified_engine_ && unified_engine_->info(info.request_id)) return;
  std::shared_ptr<SseSink> sink;
  {
    std::lock_guard<std::mutex> lock(streams_mu_);
    auto it = streams_.find(info.request_id);
    if (it != streams_.end()) {
      sink = it->second;
      streams_.erase(it);
    }
  }
  if (!sink) return;
  if (!info.result.is_nil()) {
    std::lock_guard<std::mutex> lock(sink->mu);
    sink->result = info.result;
  }
  if (info.state == "completed") {
    bool has_deltas;
    {
      std::lock_guard<std::mutex> lock(sink->mu);
      has_deltas = sink->has_deltas;
      sink->finish_reason = result_finish_reason(info.result);
      sink->result = info.result;
    }
    if (!has_deltas) sink->push(info.result);
    sink->finish();
    return;
  }
  sink->finish(info.error.empty() ? "request " + info.state : info.error);
}

OpenAIApi::OpenAIApi(Coordinator& coordinator, ApiConfig config)
    : impl_(std::make_unique<Impl>(coordinator, std::move(config))) {
  impl_->install_routes();
  coordinator.on_stream = [this](const StreamMessage& msg) { impl_->on_stream(msg); };
  coordinator.on_complete = [this](const RequestInfo& info) { impl_->on_complete(info); };
}

OpenAIApi::~OpenAIApi() = default;

void OpenAIApi::set_unified_forwarder(
    std::function<void(const UnifiedEvent&)> forwarder) {
  impl_->set_unified_forwarder(std::move(forwarder));
}

void OpenAIApi::set_unified_engine(UnifiedEngine& engine, std::string vlm_stage) {
  impl_->set_unified_engine(engine, std::move(vlm_stage));
}

bool OpenAIApi::bind() { return impl_->bind(); }
bool OpenAIApi::listen() { return impl_->listen(); }
void OpenAIApi::stop() { impl_->stop(); }
int OpenAIApi::bound_port() const { return impl_->bound_port(); }

}  // namespace umminfer
