// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "umminfer/serve/openai_api.hpp"

namespace umminfer {

class UnifiedEngine;
struct UnifiedEvent;

class RealtimeApi {
 public:
  RealtimeApi(Coordinator& coordinator, ApiConfig config);
  ~RealtimeApi();

  // Chains this listener in front of the sinks already installed by OpenAIApi
  // and the unified engine. Must be called before Coordinator::run.
  void install_stream_sink();
  // Enables the spoken response path. Without it, session.update rejects
  // audio_output and the listener stays transcription-only.
  void attach_unified(UnifiedEngine& engine);
  // Called on the Coordinator thread; queues the event for the WebSocket
  // thread. Safe to call after the listener stopped.
  void forward_event(const UnifiedEvent& event);

  bool bind();
  bool listen();
  void stop();
  int bound_port() const;

  class Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace umminfer
