// SPDX-License-Identifier: Apache-2.0
// OpenAI-compatible HTTP surface.
//
// Served routes: health, ready, models, chat completions, audio
// transcriptions, audio speech, and abort. /v1/audio/speech submits one text
// to the TTS stage the unified engine already registered and answers wav or
// pcm bytes; it answers 501 when no TTS stage is registered. Realtime uses a
// separate WebSocket route.
//
// The surface accepts only behavior this server implements. Unsupported
// request fields are rejected with HTTP 400 instead of being ignored.
#pragma once

#include <functional>
#include <memory>
#include <string>

#include "umminfer/pipeline/pipeline.hpp"

namespace umminfer {

class UnifiedEngine;
struct UnifiedEvent;

struct ApiConfig {
  std::string host = "0.0.0.0";
  int port = 30000;
  std::string model = "omni";
  std::string served_model_name;  // defaults to model
  std::string entry_stage = "entry";
  std::string api_key;            // empty disables auth
  // True when the entry stage runs in another process. /ready then answers 503
  // until that stage announces stage_ready. The in-process demo keeps false.
  bool external_entry_stage = false;
  double request_timeout_s = 600.0;
  // Admission cap for non-unified local routes (/v1/audio/transcriptions,
  // non-unified /v1/chat/completions) and fixed-upstream proxy routes. Requests
  // past this cap answer 503 with Retry-After: 1 instead of spawning unbounded
  // worker threads.
  size_t max_in_flight = 64;
  // Opt-in static hosting of a directory (normally the repository `web/`) at
  // "/". Empty keeps the route absent, so a default deployment is unchanged.
  std::string web_root;

  // Fixed upstream. One process serves one upstream, one model and one
  // variant; there is no routing or fallback between upstreams. All five
  // fields empty keeps every route on its local behavior.
  //   upstream_kind    "qwen" (official DashScope compatible API) or "sglang"
  //   upstream_base_url  HTTP(S) origin plus API prefix, no credentials
  //   upstream_model   the model ID sent to the upstream
  //   upstream_api_key sent to the upstream only, never logged
  //   upstream_variant "omni" for qwen; "image" or "video" for sglang
  // A route outside the configured variant answers 501.
  std::string upstream_kind;
  std::string upstream_base_url;
  std::string upstream_model;
  std::string upstream_api_key;
  std::string upstream_variant;
  // Opt-in upstream circuit breaker (0 disables). When enabled, N consecutive
  // upstream transport or 5xx failures open the circuit for cooldown_s and
  // immediately answer 503 so dead paid APIs are not hammered.
  int upstream_circuit_breaker_failures = 0;
  double upstream_circuit_breaker_cooldown_s = 10.0;
};

// Returns an empty string when the upstream configuration is usable, otherwise
// the reason. Network-free; bind() calls it before reserving the port.
std::string validate_upstream_config(const ApiConfig& config);

class OpenAIApi {
 public:
  OpenAIApi(Coordinator& coordinator, ApiConfig config);
  ~OpenAIApi();

  // Call after construction and before Coordinator::run. Chat then uses the
  // unified ASR -> VLM engine; transcription and realtime stay on the ASR stage.
  void set_unified_engine(UnifiedEngine& engine, std::string vlm_stage);
  // Chains an extra consumer in front of the HTTP SSE path. The realtime
  // listener uses it to forward VLM text and streamed audio to its WebSocket.
  void set_unified_forwarder(std::function<void(const UnifiedEvent&)> forwarder);

  bool bind();     // reserves the configured port without accepting requests
  bool listen();   // binds if needed, then blocks
  void stop();
  int bound_port() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace umminfer
