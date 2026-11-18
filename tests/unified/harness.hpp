// SPDX-License-Identifier: Apache-2.0
// Shared test harness and builders for the unified multimodal engine contract suite.
#pragma once

#include <atomic>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "umminfer/comm/comm.hpp"
#include "umminfer/audio/audio.hpp"
#include "umminfer/obs/metrics.hpp"
#include "umminfer/pipeline/pipeline.hpp"
#include "umminfer/pipeline/unified.hpp"
#include "umminfer/serve/openai_api.hpp"
#include "umminfer/util/codec.hpp"
#include "client/http.hpp"
#include <nlohmann/json.hpp>

using namespace umminfer;

namespace {

int g_failures = 0;

void check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++g_failures;
  }
}

void wait_until(const std::function<bool()>& condition, const char* message) {
  for (int i = 0; i < 200; ++i) {
    if (condition()) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  check(false, message);
}

// Callbacks run on the coordinator pump thread; the lock covers the handoff.
struct EventLog {
  std::mutex mu;
  std::vector<UnifiedEvent> events;
  std::map<std::string, UnifiedEvent> terminals;
  // Every on_terminal invocation, counted separately from the first-only
  // terminal map. Without this, emplace makes terminal_count() always <= 1 and
  // a duplicate-terminal bug could not be observed.
  std::map<std::string, int> terminal_callbacks;

  void add(const UnifiedEvent& event) {
    std::lock_guard<std::mutex> lock(mu);
    events.push_back(event);
  }
  void finish(const UnifiedEvent& event) {
    std::lock_guard<std::mutex> lock(mu);
    // Keep the first terminal for state checks, but count every callback.
    terminals.emplace(event.request_id, event);
    ++terminal_callbacks[event.request_id];
  }
  bool has_terminal(const std::string& id) {
    std::lock_guard<std::mutex> lock(mu);
    return terminals.count(id) != 0;
  }
  std::optional<UnifiedEvent> terminal_for(const std::string& id) {
    std::lock_guard<std::mutex> lock(mu);
    auto it = terminals.find(id);
    return it == terminals.end() ? std::nullopt : std::optional(it->second);
  }
  size_t terminal_count(const std::string& id) {
    std::lock_guard<std::mutex> lock(mu);
    auto it = terminal_callbacks.find(id);
    return it == terminal_callbacks.end() ? 0 : static_cast<size_t>(it->second);
  }
  // Backwards-compatible single-request view used by the other tests.
  bool has_terminal() {
    std::lock_guard<std::mutex> lock(mu);
    return !terminals.empty();
  }
  std::optional<UnifiedEvent> terminal_event() {
    std::lock_guard<std::mutex> lock(mu);
    return terminals.empty() ? std::nullopt
                             : std::optional(terminals.begin()->second);
  }
  size_t size() {
    std::lock_guard<std::mutex> lock(mu);
    return events.size();
  }
  std::vector<UnifiedEvent> snapshot() {
    std::lock_guard<std::mutex> lock(mu);
    return events;
  }
};

// One mkdtemp(3) directory per Harness: a unique 0700 directory this run
// owns. Socket files keep their basenames inside it, and teardown removes only
// those paths plus the owned directory. No predictable global path is ever
// pre-deleted, so a foreign process cannot lose its sockets.
struct IpcDir {
  std::string path;
  std::vector<std::string> files;

  static IpcDir create(const char* tag) {
    std::string templ =
        std::string("/tmp/omni-unified-") + tag + "-XXXXXX";
    std::vector<char> buffer(templ.begin(), templ.end());
    buffer.push_back('\0');
    char* made = mkdtemp(buffer.data());
    if (made == nullptr) {
      throw std::runtime_error(std::string("mkdtemp failed: ") +
                               std::strerror(errno));
    }
    IpcDir dir;
    dir.path = made;  // mkdtemp itself applies 0700
    for (const char* name : {"asr", "vlm", "image", "tts", "done", "abort"}) {
      dir.files.push_back(dir.path + "/" + name);
    }
    return dir;
  }

  ~IpcDir() {
    for (const auto& file : files) std::remove(file.c_str());
    // rmdir only succeeds when the directory holds nothing this run did not
    // create; it never recurses into a foreign path.
    ::rmdir(path.c_str());
  }

  std::string endpoint(const char* name) const {
    return "ipc://" + path + "/" + name;
  }
};

Value::Map part(const std::string& type) {
  Value::Map out;
  out["type"] = Value(type);
  return out;
}

StagePayload audio_request(const std::string& modality = "audio") {
  StagePayload payload;
  Value::Map message;
  message["role"] = Value("user");
  Value::Array content;
  content.push_back(Value(part("input_audio")));
  message["content"] = Value(std::move(content));
  Value::Array messages;
  messages.push_back(Value(std::move(message)));
  Value::Array modalities;
  modalities.push_back(Value(modality));
  Value::Map inputs;
  inputs["messages"] = Value(std::move(messages));
  inputs["modalities"] = Value(std::move(modalities));
  payload.request.inputs = Value(std::move(inputs));
  return payload;
}

Value image_part(const std::string& url = "data:image/png;base64,AA") {
  Value::Map image_url;
  image_url["url"] = Value(url);
  Value::Map out = part("image_url");
  out["image_url"] = Value(std::move(image_url));
  return Value(std::move(out));
}

StagePayload image_part_request(const std::string& url = "data:image/png;base64,AA") {
  StagePayload payload;
  Value::Map message;
  message["role"] = Value("user");
  Value::Array content;
  content.push_back(image_part(url));
  message["content"] = Value(std::move(content));
  Value::Array messages;
  messages.push_back(Value(std::move(message)));
  Value::Map inputs;
  inputs["messages"] = Value(std::move(messages));
  payload.request.inputs = Value(std::move(inputs));
  return payload;
}

// One request carrying real PCM for the ASR leg and one image part for the VLM
// leg. The image rides on the original messages; the ASR leg never sees it.
StagePayload joint_request(const std::string& session_id = "",
                           const std::string& image_url = "") {
  StagePayload payload;
  Value::Map message;
  message["role"] = Value("user");
  Value::Array content;
  content.push_back(image_part(image_url.empty() ? "data:image/png;base64,AA"
                                                 : image_url));
  content.push_back(Value(part("input_audio")));
  message["content"] = Value(std::move(content));
  Value::Array messages;
  messages.push_back(Value(std::move(message)));
  Value::Array modalities;
  modalities.push_back(Value("audio"));
  modalities.push_back(Value("image"));
  Value::Map inputs;
  inputs["messages"] = Value(std::move(messages));
  inputs["modalities"] = Value(std::move(modalities));
  inputs["audio"] = Value::bin(std::string(800, '\0'));
  if (!session_id.empty()) inputs["session_id"] = Value(session_id);
  payload.request.inputs = Value(std::move(inputs));
  return payload;
}

// A joint request that explicitly opts into multi-turn history. The engine
// must have session history enabled or submit() rejects it.
StagePayload history_request(const std::string& session_id) {
  StagePayload payload = joint_request(session_id);
  Value::Map inputs = payload.request.inputs.as_map();
  inputs["session_history"] = Value(true);
  payload.request.inputs = Value(std::move(inputs));
  return payload;
}


// Real PCM selects the ASR leg. The optional flag is the documented opt-in for
// incremental ASR events through the unified chat route.
StagePayload audio_leg_request(bool asr_partials) {
  StagePayload payload = audio_request();
  Value::Map inputs = payload.request.inputs.as_map();
  inputs["audio"] = Value::bin(std::string(800, '\0'));
  if (asr_partials) inputs["asr_partials"] = Value(true);
  payload.request.inputs = Value(std::move(inputs));
  return payload;
}

Value::Map text_chunk(const std::string& text, bool is_final) {
  Value::Map chunk;
  chunk["text"] = Value(text);
  if (is_final) chunk["is_final"] = Value(true);
  return chunk;
}

Value::Map text_result(const std::string& text) {
  Value::Map result;
  result["text"] = Value(text);
  return result;
}

// A VLM-only request that opts into image and/or tts output legs.
StagePayload mixed_request(bool want_image, bool want_audio,
                           const std::string& image_prompt = "",
                           const std::string& voice = "Tina") {
  StagePayload payload;
  Value::Map message;
  message["role"] = Value("user");
  message["content"] = Value("describe the scene");
  Value::Array messages;
  messages.push_back(Value(std::move(message)));
  Value::Array selected;
  selected.push_back(Value("text"));
  if (want_image) selected.push_back(Value("image"));
  if (want_audio) selected.push_back(Value("audio"));
  Value::Map inputs;
  inputs["messages"] = Value(std::move(messages));
  inputs["output_modalities"] = Value(std::move(selected));
  if (want_image) {
    Value::Map options;
    if (!image_prompt.empty()) options["prompt"] = Value(image_prompt);
    options["size"] = Value("1024x1024");
    options["output_format"] = Value("png");
    inputs["image_options"] = Value(std::move(options));
  }
  if (want_audio) {
    Value::Map options;
    options["voice"] = Value(voice);
    options["format"] = Value("wav");
    options["sample_rate"] = Value(int64_t{24000});
    inputs["audio_options"] = Value(std::move(options));
  }
  payload.request.inputs = Value(std::move(inputs));
  return payload;
}

// A worker result carrying an inline Bin artifact, as the frozen contract
// specifies. The bytes are fake stage output, never a real model artifact.
Value artifact_result(const std::string& media_type,
                      const std::string& mime_type, std::string bytes,
                      Value::Map extra = {}) {
  Value::Map result = std::move(extra);
  result["media_type"] = Value(media_type);
  result["mime_type"] = Value(mime_type);
  result["data"] = Value::bin(std::move(bytes));
  return Value(std::move(result));
}

// Real Coordinator with two fake stage inboxes and one event pusher. No
// compute runs anywhere.
struct Harness {
  IpcDir dir;
  PullSocket asr_recv;
  PullSocket vlm_recv;
  PullSocket image_recv;
  PullSocket tts_recv;
  PushSocket event_push;
  Coordinator coordinator;
  std::thread pump;
  // broadcast_abort pushes AbortMessage on the same stage PULL sockets, so the
  // fake stages observe cancels here. The PUSH path is the reliable ordered
  // one; the test deliberately does not depend on PUB slow-joiner timing.
  std::mutex abort_mu;
  std::vector<std::string> aborted_legs;

  explicit Harness(const char* tag, bool start_pump_thread = false)
      : dir(IpcDir::create(tag)),
        asr_recv(dir.endpoint("asr"), true),
        vlm_recv(dir.endpoint("vlm"), true),
        image_recv(dir.endpoint("image"), true),
        tts_recv(dir.endpoint("tts"), true),
        event_push(dir.endpoint("done")),
        coordinator(dir.endpoint("done"), dir.endpoint("abort")) {
    asr_recv.start();
    vlm_recv.start();
    image_recv.start();
    tts_recv.start();
    coordinator.register_stage("asr", dir.endpoint("asr"));
    coordinator.register_stage("vlm", dir.endpoint("vlm"));
    // Optional legs stay out of the readiness gate, matching production wiring.
    coordinator.register_stage("image", dir.endpoint("image"));
    coordinator.register_stage("tts", dir.endpoint("tts"));
    coordinator.start();
    if (start_pump_thread) start_pump();
    event_push.connect();
  }

  // Config with both optional output stages enabled.
  UnifiedConfig mixed_config() const {
    UnifiedConfig config;
    config.image_stage = "image";
    config.tts_stage = "tts";
    return config;
  }

  void start_pump() {
    pump = std::thread([this] { coordinator.run(); });
  }

  ~Harness() {
    coordinator.stop();
    if (pump.joinable()) pump.join();
    event_push.close();
    asr_recv.close();
    vlm_recv.close();
    image_recv.close();
    tts_recv.close();
  }

  std::optional<SubmitMessage> next_submit(PullSocket& socket) {
    // Abort frames share the stage PULL socket: broadcast_abort pushes to every
    // stage socket, and the real Stage drains them in handle_message. The fake
    // stage records them and skips to the next submit frame.
    while (true) {
      auto msg = socket.recv_timeout(5000);
      if (!msg) return std::nullopt;
      if (auto* abort = std::get_if<AbortMessage>(&*msg)) {
        std::lock_guard<std::mutex> lock(abort_mu);
        aborted_legs.push_back(abort->request_id);
        continue;
      }
      if (auto* submit = std::get_if<SubmitMessage>(&*msg)) return *submit;
    }
  }

  bool saw_abort(const std::string& leg_id) {
    std::lock_guard<std::mutex> lock(abort_mu);
    for (const auto& id : aborted_legs) {
      if (id == leg_id) return true;
    }
    return false;
  }

  void record_abort(const std::string& leg_id) {
    std::lock_guard<std::mutex> lock(abort_mu);
    aborted_legs.push_back(leg_id);
  }

  // Negative assertion: only the PUSH abort path can arrive, never a submit.
  bool no_submit(PullSocket& socket) {
    while (true) {
      auto msg = socket.recv_timeout(200);
      if (!msg) return true;
      if (auto* abort = std::get_if<AbortMessage>(&*msg)) {
        std::lock_guard<std::mutex> lock(abort_mu);
        aborted_legs.push_back(abort->request_id);
        continue;
      }
      if (auto* submit = std::get_if<SubmitMessage>(&*msg)) {
        (void)submit;
        return false;
      }
    }
  }

  // Drains all stage inboxes until the abort for leg_id is observed.
  bool wait_abort(const std::string& leg_id) {
    for (int i = 0; i < 200 && !saw_abort(leg_id); ++i) {
      for (PullSocket* socket : {&asr_recv, &vlm_recv, &image_recv, &tts_recv}) {
        auto msg = socket->recv_timeout(10);
        if (!msg) continue;
        if (auto* abort = std::get_if<AbortMessage>(&*msg)) {
          std::lock_guard<std::mutex> lock(abort_mu);
          aborted_legs.push_back(abort->request_id);
        }
      }
    }
    return saw_abort(leg_id);
  }

  void stream(const std::string& id, const std::string& stage, Value chunk) {
    StreamMessage msg;
    msg.request_id = id;
    msg.from_stage = stage;
    msg.chunk = std::move(chunk);
    event_push.send(Message{std::move(msg)});
  }

  void complete(const std::string& id, const std::string& stage, bool success,
                Value result,
                std::optional<std::string> error = std::nullopt) {
    CompleteMessage msg;
    msg.request_id = id;
    msg.from_stage = stage;
    msg.success = success;
    msg.result = std::move(result);
    msg.error = std::move(error);
    event_push.send(Message{std::move(msg)});
  }
};

// Drives one request through the fake ASR and VLM stages to a successful
// terminal, returning the VLM leg submit payload for prompt inspection.
StagePayload complete_turn(Harness& h, UnifiedEngine& engine,
                           const StagePayload& request,
                           const std::string& transcript,
                           const std::string& answer) {
  const std::string id = engine.submit(request);
  check(h.next_submit(h.asr_recv).has_value(), "history turn: no ASR submit");
  h.complete(id, "asr", true, text_result(transcript));
  auto vlm = h.next_submit(h.vlm_recv);
  check(vlm.has_value(), "history turn: no VLM submit");
  StagePayload vlm_payload;
  if (vlm) vlm_payload = StagePayload::from_value(vlm->data);
  h.complete(id + ":vlm", "vlm", true, text_result(answer));
  check(engine.wait(id, 5.0), "history turn did not complete");
  engine.release(id);
  return vlm_payload;
}

// The role/content pairs of a VLM prompt, in wire order.
std::vector<std::pair<std::string, std::string>> prompt_roles(
    const StagePayload& vlm_payload) {
  std::vector<std::pair<std::string, std::string>> out;
  const Value* messages = vlm_payload.request.inputs.find("messages");
  if (messages == nullptr) return out;
  for (const Value& message : messages->as_array()) {
    const Value& content = *message.find("content");
    // History turns carry plain strings; the current turn may carry the
    // content-part array, whose last text part is the transcript.
    std::string text;
    if (content.is_str()) {
      text = content.as_str();
    } else {
      for (const Value& part : content.as_array()) {
        const Value* part_text = part.find("text");
        if (part_text != nullptr) text = part_text->as_str();
      }
    }
    out.emplace_back(message.find("role")->as_str(), text);
  }
  return out;
}

// Streaming ASR partials/final -> VLM: ordering, gap-free seq, aggregation.

template <typename Fut>
bool wait_future_ready(Fut& fut, const char* what) {
  const bool ok =
      fut.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
  check(ok, what);
  return ok;
}

inline void wait_http_ready(int port) {
  const std::string url =
      "http://127.0.0.1:" + std::to_string(port) + "/health";
  for (int i = 0; i < 250; ++i) {
    try {
      client::detail::HttpRequest req;
      req.url = url;
      req.method = client::detail::HttpMethod::kGet;
      req.timeout = client::Milliseconds{200};
      req.connect_timeout = client::Milliseconds{200};
      req.no_proxy = true;
      auto res = client::detail::perform(req);
      if (res.status == 200) return;
    } catch (...) {
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  check(false, "HTTP server did not answer /health within startup window");
}

inline UnifiedConfig history_config(size_t max_turns, size_t max_chars) {
  UnifiedConfig config;
  config.session_history.enabled = true;
  config.session_history.max_turns = max_turns;
  config.session_history.max_chars = max_chars;
  return config;
}

}  // namespace
