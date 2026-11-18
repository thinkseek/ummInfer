// SPDX-License-Identifier: Apache-2.0
#include "umminfer/serve/realtime_api.hpp"

#include <libwebsockets.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "asr_common.hpp"
#include "energy_vad.hpp"
#include "umminfer/pipeline/unified.hpp"
#include "umminfer/util/codec.hpp"

namespace umminfer {
namespace {

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

constexpr size_t kMaxMessageBytes = 1u << 20;
constexpr size_t kMaxAudioBytes = 64u << 20;
// 200 ms of PCM16 mono 16 kHz. A preview request is submitted for every window
// of new audio while no request is in flight.
constexpr size_t kPreviewWindowBytes = 6400;
// Bound preview re-transcription to the most recent 15 s (16 kHz mono PCM16)
// so a long monologue stays O(1) per preview instead of O(N^2); final commit
// still transcribes the full turn.
constexpr size_t kMaxPreviewSpanBytes = 32000u * 15u;
constexpr size_t kMaxRealtimeSessions = 64;
constexpr size_t kMaxOutboundFrames = 1024;

struct RealtimeSession {
  std::string incoming;
  // PCM appended by the client. With server VAD the completed turns are
  // erased, so this holds the current turn only.
  std::string audio;
  std::string request_id;
  // Cumulative text the stage has streamed so far. Partial events carry the
  // cumulative value: the client Assembler replaces its provisional segment.
  std::string partial_text;
  std::deque<std::string> outbound;
  Clock::time_point deadline;
  int64_t max_tokens = 128;
  bool ready = false;
  bool close_after_write = false;

  // Incremental submission and interruption state.
  bool request_is_final = false;
  size_t committed_audio = 0;  // bytes already sent in the last submit
  // Bytes owned by the turn that was just committed. reset_turn erases
  // exactly these; audio appended after the final submit starts the next
  // turn. Without this the buffer grew across turns (turn_audio_start is 0
  // on a normal turn, so erase(0, 0) kept everything).
  size_t final_committed = 0;
  size_t turn_audio_start = 0; // offset of the current turn in `audio`
  int64_t segment_id = 0;      // increments per turn under server VAD
  int64_t partial_count = 0;
  bool speech_open = false;
  std::unique_ptr<serve_detail::EnergyVad> vad;
  Clock::time_point first_audio_at;
  bool have_first_audio = false;
  bool first_partial_logged = false;
  long long turn_started_ms = 0;
  long long last_interrupt_ms = 0;

  // Spoken response. The realtime listener stays a pure ASR surface unless
  // session.update enables audio_output and a unified engine is attached.
  bool audio_output = false;
  std::string voice = "vivian";
  std::string unified_session;
  std::string response_request_id;
  Clock::time_point response_deadline{};
  int64_t response_segment = -1;
};

std::string header(lws* wsi, lws_token_indexes token) {
  const int length = lws_hdr_total_length(wsi, token);
  if (length <= 0 || length > 8192) return {};
  std::string value(static_cast<size_t>(length) + 1, '\0');
  if (lws_hdr_copy(wsi, value.data(), length + 1, token) != length) return {};
  value.resize(static_cast<size_t>(length));
  return value;
}

// libwebsockets stores the request path without the query string. The query
// values are available separately, already percent-decoded.
bool query_argument(lws* wsi, const char* name, std::string& out) {
  char buffer[8192];
  const int length =
      lws_get_urlarg_by_name_safe(wsi, name, buffer, sizeof(buffer));
  if (length < 0 || length >= static_cast<int>(sizeof(buffer))) return false;
  out.assign(buffer, static_cast<size_t>(length));
  return true;
}

// The URI lws exposes is normalized: it collapses //, resolves dot segments
// and decodes percent escapes. A different path is rejected, but a normalized
// alias of /v1/realtime is indistinguishable here. That is a known limitation.
bool valid_handshake(lws* wsi, const std::string& served_model,
                     std::string& error) {
  const std::string path = header(wsi, WSI_TOKEN_GET_URI);
  if (path != "/v1/realtime") {
    error = "realtime WebSocket path must be exactly /v1/realtime";
    return false;
  }
  std::string intent;
  if (!query_argument(wsi, "intent", intent) || intent != "transcription") {
    error = "realtime WebSocket intent must be transcription";
    return false;
  }
  std::string model;
  if (!query_argument(wsi, "model", model)) {
    error = "realtime WebSocket requires a model query parameter";
    return false;
  }
  if (model != served_model) {
    error = "realtime WebSocket model does not match the served model '" +
            served_model + "'";
    return false;
  }
  return true;
}

// The session.update reply must describe the configuration the server now
// enforces, not echo fields that were ignored.
json effective_session(const RealtimeSession& session) {
  json turn = nullptr;
  if (session.vad) {
    turn = {{"type", "server_vad"}, {"silence_ms", session.vad->silence_ms()}};
  }
  json effective = {{"input_audio_format", "pcm16"},
                    {"turn_detection", turn},
                    {"max_new_tokens", session.max_tokens}};
  if (session.audio_output) effective["audio_output"] = {{"voice", session.voice}};
  return effective;
}

long long epoch_millis() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

// Copies only metadata the stage actually reported. A stage that reports a
// nested generation_metadata map takes precedence over top-level fields.
std::optional<json> live_generation_metadata(const Value& result) {
  const Value* nested = result.find("generation_metadata");
  const Value& source =
      nested != nullptr && nested->is_map() ? *nested : result;
  json out = json::object();
  if (const Value* count = source.find("generated_token_count");
      count != nullptr && count->is_int() && count->as_int() >= 0) {
    out["generated_token_count"] = count->as_int();
  }
  if (const Value* finish = source.find("finish_reason");
      finish != nullptr && finish->is_str() && !finish->as_str().empty()) {
    out["finish_reason"] = finish->as_str();
  }
  if (out.empty()) return std::nullopt;
  return out;
}

int realtime_callback(lws* wsi, lws_callback_reasons reason, void* user,
                      void* in, size_t len);

const lws_protocols protocols[] = {
    {"umminfer-realtime", realtime_callback, sizeof(RealtimeSession), 16384, 0, nullptr, 0},
    LWS_PROTOCOL_LIST_TERM,
};

}  // namespace

class RealtimeApi::Impl {
 public:
  Impl(Coordinator& coordinator, ApiConfig config)
      : coordinator_(coordinator), config_(std::move(config)) {
    if (config_.served_model_name.empty()) {
      config_.served_model_name = config_.model;
    }
  }

  ~Impl() {
    if (context_) lws_context_destroy(context_);
  }

  // Chains this listener in front of the sinks already installed by OpenAIApi
  // (and the unified engine, when present). Must run before Coordinator::run;
  // exchange_sinks_before_run enforces that. Every other consumer still
  // receives the same messages.
  void install_stream_sink() {
    auto previous = coordinator_.exchange_sinks_before_run(
        [this](const StreamMessage& msg) {
          enqueue_partial(msg);
          if (previous_stream_) previous_stream_(msg);
        },
        [this](const RequestInfo& info) {
          if (previous_complete_) previous_complete_(info);
        });
    previous_stream_ = std::move(previous.first);
    previous_complete_ = std::move(previous.second);
  }

  // Called on the coordinator thread. WebSocket session state belongs to the
  // libwebsockets thread, so this only queues the delta.
  void enqueue_partial(const StreamMessage& msg) {
    if (!msg.chunk.is_map()) return;
    if (const Value* final = msg.chunk.find("is_final");
        final != nullptr && final->is_bool() && final->as_bool()) {
      return;
    }
    const Value* text = msg.chunk.find("text");
    if (text == nullptr || !text->is_str() || text->as_str().empty()) return;
    std::lock_guard<std::mutex> lock(partial_mu_);
    // ponytail: fixed cap; a stalled WebSocket must not grow the control plane.
    if (partial_queue_.size() >= 4096) return;
    partial_queue_.emplace_back(msg.request_id, text->as_str());
  }

  // Called on the libwebsockets thread only. Converts stage deltas into
  // cumulative, non-final segments for the matching session.
  void drain_partials() {
    std::deque<std::pair<std::string, std::string>> batch;
    {
      std::lock_guard<std::mutex> lock(partial_mu_);
      batch.swap(partial_queue_);
    }
    for (auto& [request_id, delta] : batch) {
      for (lws* wsi : sessions_) {
        auto* session = static_cast<RealtimeSession*>(lws_wsi_user(wsi));
        if (session == nullptr || session->request_id != request_id ||
            session->close_after_write) {
          continue;
        }
        session->partial_text += delta;
        // ponytail: drop preview events under outbound backpressure; the final
        // segment still carries the complete text.
        if (session->outbound.size() >= 512) break;
        send(wsi, session, {{"type", "transcription.segment"},
                            {"event_index", 0},
                            {"segment_id", session->segment_id},
                            {"text", session->partial_text},
                            {"is_final", false}});
        break;
      }
    }
  }

  bool bind() {
    if (context_) return true;
    lws_context_creation_info info{};
    info.port = config_.port;
    info.iface = config_.host.empty() ? nullptr : config_.host.c_str();
    info.protocols = protocols;
    info.user = this;
    info.vhost_name = "umminfer";
    info.options = LWS_SERVER_OPTION_DISABLE_IPV6;
    context_ = lws_create_context(&info);
    if (!context_) return false;
    lws_vhost* vhost = lws_get_vhost_by_name(context_, "umminfer");
    if (!vhost) return false;
    bound_port_ = config_.port == 0 ? lws_get_vhost_listen_port(vhost)
                                    : config_.port;
    return bound_port_ >= 0;
  }

  bool listen() {
    if (!context_ && !bind()) return false;
    running_ = true;
    schedule_poll();
    bool ok = true;
    while (running_) {
      if (lws_service(context_, 0) < 0) {
        ok = false;
        break;
      }
    }
    lws_sul_cancel(&poll_sul_);
    return ok;
  }

  void schedule_poll() {
    lws_sul_schedule(context_, 0, &poll_sul_, poll_timer, 10 * LWS_US_PER_MS);
  }

  static void poll_timer(lws_sorted_usec_list_t* sul) {
    auto* self = lws_container_of(sul, RealtimeApi::Impl, poll_sul_);
    self->poll_sessions();
    if (self->running_) self->schedule_poll();
  }

  void stop() {
    running_ = false;
    if (context_) lws_cancel_service(context_);
  }

  int bound_port() const { return bound_port_; }

  bool authorized(lws* wsi) const {
    if (config_.api_key.empty()) return true;
    const std::string expected = "Bearer " + config_.api_key;
    return util::constant_time_equals(header(wsi, WSI_TOKEN_HTTP_AUTHORIZATION),
                                      expected);
  }

  bool at_session_capacity() const {
    return active_session_count_.load(std::memory_order_relaxed) >=
           kMaxRealtimeSessions;
  }

  void cancel_and_release_response(RealtimeSession* session,
                                   const char* reason) {
    if (session->response_request_id.empty()) return;
    const std::string id = std::move(session->response_request_id);
    session->response_request_id.clear();
    if (unified_ != nullptr) {
      unified_->cancel(id, reason);
      unified_->release(id);
    }
  }

  void established(lws* wsi, RealtimeSession* session) {
    new (session) RealtimeSession;
    session->unified_session = "realtime-" + std::to_string(++next_session_no_);
    sessions_.insert(wsi);
    active_session_count_.store(sessions_.size(), std::memory_order_relaxed);
  }

  void closed(lws* wsi, RealtimeSession* session) {
    if (!session->request_id.empty()) {
      coordinator_.abort(session->request_id);
      coordinator_.release(session->request_id);
    }
    cancel_and_release_response(session, "shutdown");
    sessions_.erase(wsi);
    active_session_count_.store(sessions_.size(), std::memory_order_relaxed);
    session->~RealtimeSession();
  }

  void send(lws* wsi, RealtimeSession* session, const json& event) {
    if (session->outbound.size() >= kMaxOutboundFrames) {
      session->close_after_write = true;
      return;
    }
    session->outbound.push_back(event.dump());
    lws_callback_on_writable(wsi);
  }

  void fail(lws* wsi, RealtimeSession* session, const std::string& message) {
    if (session->close_after_write) return;
    if (session->outbound.size() >= kMaxOutboundFrames &&
        !session->outbound.empty()) {
      session->outbound.pop_back();
    }
    send(wsi, session,
         {{"type", "error"}, {"error", {{"code", "request_error"}, {"message", message}}}});
    session->close_after_write = true;
  }

  // Stops the in-flight request at the stage's next boundary and drops its
  // record. libwebsockets thread only.
  void interrupt_request(RealtimeSession* session, const char* reason) {
    cancel_and_release_response(session, "barge_in");
    if (session->request_id.empty()) return;
    const std::string request_id = session->request_id;
    session->last_interrupt_ms = epoch_millis();
    coordinator_.abort(request_id);
    coordinator_.release(request_id);
    session->request_id.clear();
    session->request_is_final = false;
    std::cerr << "realtime interrupt reason=" << reason
              << " request=" << request_id
              << " epoch_ms=" << session->last_interrupt_ms << "\n";
  }

  // transcription.done: the caller explicitly ends the turn. This stays the
  // compatibility path for clients that do not enable server VAD.
  void submit_final(lws* wsi, RealtimeSession* session) {
    const size_t length = session->audio.size() - session->turn_audio_start;
    if (!session->ready || length == 0 || length % 2 != 0 ||
        session->request_is_final || session->close_after_write) {
      fail(wsi, session,
           "transcription.done requires an accepted session and PCM16 audio");
      return;
    }
    submit_request(wsi, session, true);
  }

  // Starts one ASR request over the current turn's audio. Returns false when
  // the session cannot accept it. A final request preempts a preview.
  bool submit_request(lws* wsi, RealtimeSession* session, bool final) {
    if (!session->ready || session->close_after_write) return false;
    if (!session->request_id.empty()) {
      if (!final) return false;
      interrupt_request(session, "final_commit");
    }
    const size_t length = session->audio.size() - session->turn_audio_start;
    if (length == 0 || length % 2 != 0) return false;
    try {
      size_t slice_offset = session->turn_audio_start;
      if (!final && length > kMaxPreviewSpanBytes) {
        slice_offset = session->audio.size() - kMaxPreviewSpanBytes;
        if ((slice_offset - session->turn_audio_start) % 2 != 0) ++slice_offset;
      }
      StagePayload payload = serve_detail::make_asr_payload(
          session->audio.substr(slice_offset), session->max_tokens, "", "");
      // Ask the stage to stream. A stage that ignores stream sends no chunk;
      // this listener then emits only the final segment, as before.
      payload.request.params["stream"] = Value(true);
      // Previews are internal and are aborted independently, so they use
      // their own id space. Turn ids stay consecutive for clients and logs.
      session->request_id = final
                                ? "realtime-" + std::to_string(++next_request_id_)
                                : "realtime-preview-" + std::to_string(++next_preview_id_);
      payload.request_id = session->request_id;
      session->request_is_final = final;
      session->committed_audio = session->audio.size();
      if (final) session->final_committed = session->audio.size();
      session->deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(
          std::chrono::duration<double>(config_.request_timeout_s));
      coordinator_.submit(std::move(payload), config_.entry_stage);
      return true;
    } catch (const std::exception& error) {
      if (!session->request_id.empty()) coordinator_.release(session->request_id);
      session->request_id.clear();
      session->request_is_final = false;
      if (final) fail(wsi, session, error.what());
      return false;
    }
  }

  // Emits one cumulative provisional segment for a completed preview.
  void emit_preview(lws* wsi, RealtimeSession* session, const Value& result) {
    session->partial_text = serve_detail::asr_result_text(result);
    ++session->partial_count;
    if (!session->first_partial_logged) {
      session->first_partial_logged = true;
      const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          Clock::now() - session->first_audio_at);
      std::cerr << "realtime first_partial_ms=" << elapsed.count()
                << " segment=" << session->segment_id << "\n";
    }
    send(wsi, session, {{"type", "transcription.segment"},
                        {"event_index", 0},
                        {"segment_id", session->segment_id},
                        {"text", session->partial_text},
                        {"is_final", false}});
  }

  // Prepares the session for the next turn. Audio that arrived after the final
  // submit belongs to that turn and is kept.
  void reset_turn(RealtimeSession* session) {
    session->partial_text.clear();
    session->partial_count = 0;
    session->speech_open = false;
    session->first_partial_logged = false;
    session->segment_id += 1;
    const size_t drop = std::min(session->final_committed, session->audio.size());
    session->audio.erase(0, drop);
    session->final_committed = 0;
    session->turn_audio_start = 0;
    session->committed_audio = 0;
    if (!session->audio.empty()) {
      session->first_audio_at = Clock::now();
      session->have_first_audio = true;
    } else {
      session->have_first_audio = false;
    }
  }

  // Submits a preview once a new 200 ms window accumulated. Only one request
  // runs at a time; a slower stage therefore coalesces windows instead of
  // queueing them.
  void maybe_submit_preview(lws* wsi, RealtimeSession* session) {
    if (!session->ready || session->close_after_write) return;
    if (!session->request_id.empty()) return;
    if (session->vad && !session->speech_open) return;
    const size_t turn_bytes = session->audio.size() - session->turn_audio_start;
    if (turn_bytes < kPreviewWindowBytes) return;
    if (session->audio.size() - session->committed_audio < kPreviewWindowBytes) {
      return;
    }
    submit_request(wsi, session, false);
  }

  // Clears the provisional segment the client shows for the abandoned turn.
  void clear_preview(lws* wsi, RealtimeSession* session) {
    session->partial_text.clear();
    session->partial_count = 0;
    send(wsi, session, {{"type", "transcription.segment"},
                        {"event_index", 0},
                        {"segment_id", session->segment_id},
                        {"text", ""},
                        {"is_final", false}});
  }

  void on_speech_started(lws* wsi, RealtimeSession* session,
                         size_t speech_offset) {
    // Barge-in: a new utterance while the previous turn still computes.
    if (!session->request_id.empty()) {
      interrupt_request(session, "barge_in");
      session->segment_id += 1;
      // Keep the whole onset frame: it is the first audio of the new turn.
      session->turn_audio_start =
          speech_offset >= session->vad->frame_bytes()
              ? speech_offset - session->vad->frame_bytes()
              : 0;
      session->committed_audio = session->audio.size();
      session->first_audio_at = Clock::now();
      session->have_first_audio = true;
      clear_preview(wsi, session);
    }
    session->speech_open = true;
    send(wsi, session, {{"type", "input_audio_buffer.speech_started"},
                        {"segment_id", session->segment_id}});
  }

  void on_speech_stopped(lws* wsi, RealtimeSession* session) {
    session->speech_open = false;
    send(wsi, session, {{"type", "input_audio_buffer.speech_stopped"},
                        {"segment_id", session->segment_id},
                        {"audio_ms", static_cast<int64_t>(
                             (session->audio.size() - session->turn_audio_start) /
                             32)}});
    // Endpoint: commit the turn without waiting for transcription.done.
    submit_request(wsi, session, true);
  }

  void append_audio(lws* wsi, RealtimeSession* session, const std::string& pcm) {
    if (pcm.size() > kMaxAudioBytes - std::min(session->audio.size(), kMaxAudioBytes)) {
      throw ProtocolError("realtime audio exceeds the server limit");
    }
    if (!session->have_first_audio) {
      session->first_audio_at = Clock::now();
      session->have_first_audio = true;
      session->turn_started_ms = epoch_millis();
    }
    session->audio.append(pcm);
    if (session->vad) {
      const size_t base = session->audio.size() - pcm.size();
      for (const auto& frame : session->vad->push(pcm.data(), pcm.size())) {
        if (frame.event == serve_detail::EnergyVad::Event::kSpeechStarted) {
          on_speech_started(wsi, session, base + frame.bytes_consumed);
        } else if (frame.event ==
                   serve_detail::EnergyVad::Event::kSpeechStopped) {
          on_speech_stopped(wsi, session);
        }
      }
    }
    maybe_submit_preview(wsi, session);
  }

  // Accepts only fields this server actually applies. Validation completes
  // before any state changes, so a rejected update leaves the session intact.
  void update_session(lws* wsi, RealtimeSession* session, const json& settings) {
    if (!session->request_id.empty()) {
      throw ProtocolError("session.update is not accepted after transcription.done");
    }
    static const std::set<std::string> kAllowed = {
        "input_audio_format", "turn_detection", "max_new_tokens", "audio_output"};
    for (auto it = settings.begin(); it != settings.end(); ++it) {
      // The stage never reads these, so accepting them would report a
      // configuration that is not in effect. Presence alone is rejected,
      // including an empty string or null.
      if (it.key() == "language") {
        throw ProtocolError("language is not supported by this server");
      }
      if (it.key() == "prompt") {
        throw ProtocolError("prompt is not supported by this server");
      }
      if (it.key() == "audio_layout") {
        throw ProtocolError(
            "audio_layout is not supported: this backend consumes headerless "
            "PCM16 mono 16 kHz audio and reports no named layout; omit the field");
      }
      if (!kAllowed.count(it.key())) {
        throw ProtocolError("unsupported session field: " + it.key());
      }
    }
    if (settings.contains("input_audio_format")) {
      const json& format = settings["input_audio_format"];
      if (!format.is_string() || format.get<std::string>() != "pcm16") {
        throw ProtocolError("input_audio_format must be pcm16");
      }
    }
    // turn_detection: null disables the server endpoint detector; a
    // {"type":"server_vad","silence_ms":N} object enables it. Absence keeps the
    // current setting, so a max_new_tokens-only update does not turn VAD off.
    const bool apply_turn_detection = settings.contains("turn_detection");
    bool vad_enabled = false;
    serve_detail::VadConfig vad_config;
    if (apply_turn_detection && !settings["turn_detection"].is_null()) {
      const json& turn = settings["turn_detection"];
      if (!turn.is_object()) {
        throw ProtocolError("turn_detection must be null or a server_vad object");
      }
      for (auto it = turn.begin(); it != turn.end(); ++it) {
        if (it.key() != "type" && it.key() != "silence_ms") {
          throw ProtocolError("unsupported turn_detection field: " + it.key());
        }
      }
      if (!turn.contains("type") || !turn["type"].is_string() ||
          turn["type"].get<std::string>() != "server_vad") {
        throw ProtocolError("turn_detection type must be server_vad");
      }
      if (turn.contains("silence_ms")) {
        const json& silence = turn["silence_ms"];
        const bool valid =
            silence.is_number_integer() &&
            (silence.is_number_unsigned()
                 ? silence.get<uint64_t>() >= 100 && silence.get<uint64_t>() <= 5000
                 : silence.get<int64_t>() >= 100 && silence.get<int64_t>() <= 5000);
        if (!valid) {
          throw ProtocolError(
              "turn_detection silence_ms must be between 100 and 5000");
        }
        vad_config.silence_ms = static_cast<int>(
            silence.is_number_unsigned()
                ? silence.get<uint64_t>()
                : static_cast<uint64_t>(silence.get<int64_t>()));
      }
      vad_enabled = true;
    }
    int64_t max_tokens = session->max_tokens;
    if (settings.contains("max_new_tokens")) {
      const json& limit = settings["max_new_tokens"];
      // nlohmann parses a non-negative JSON integer as number_unsigned, so both
      // signed and unsigned values must be accepted when they are in range.
      const bool valid =
          limit.is_number_integer() &&
          (limit.is_number_unsigned()
               ? limit.get<uint64_t>() >= 1 && limit.get<uint64_t>() <= 4096
               : limit.get<int64_t>() >= 1 && limit.get<int64_t>() <= 4096);
      if (!valid) {
        throw ProtocolError("max_new_tokens must be between 1 and 4096");
      }
      max_tokens = limit.is_number_unsigned()
                       ? static_cast<int64_t>(limit.get<uint64_t>())
                       : limit.get<int64_t>();
    }
    // audio_output: null disables the spoken response; an object enables it
    // and may name a voice. It is rejected when this process has no TTS stage.
    bool audio_output = session->audio_output;
    std::string voice = session->voice;
    if (settings.contains("audio_output")) {
      const json& audio = settings["audio_output"];
      if (audio.is_null()) {
        audio_output = false;
      } else {
        if (!audio.is_object()) {
          throw ProtocolError("audio_output must be null or an object");
        }
        for (auto it = audio.begin(); it != audio.end(); ++it) {
          if (it.key() != "voice") {
            throw ProtocolError("unsupported audio_output field: " + it.key());
          }
        }
        if (audio.contains("voice")) {
          if (!audio["voice"].is_string() ||
              audio["voice"].get<std::string>().empty()) {
            throw ProtocolError("audio_output voice must be a non-empty string");
          }
          voice = audio["voice"].get<std::string>();
        }
        if (unified_ == nullptr || !unified_->tts_enabled()) {
          throw ProtocolError("audio output is not enabled on this server");
        }
        audio_output = true;
      }
    }
    // A spoken response needs the server to know where the turn ended; the
    // legacy transcription.done path closes the socket after one result.
    if (audio_output &&
        !(apply_turn_detection ? vad_enabled : session->vad != nullptr)) {
      throw ProtocolError("audio_output requires turn_detection server_vad");
    }
    session->max_tokens = max_tokens;
    session->audio_output = audio_output;
    session->voice = voice;
    if (apply_turn_detection) {
      if (vad_enabled) {
        session->vad = std::make_unique<serve_detail::EnergyVad>(vad_config);
      } else {
        session->vad.reset();
        session->speech_open = false;
      }
    }
    session->ready = true;
    send(wsi, session,
         {{"type", "transcription_session.updated"},
          {"session", effective_session(*session)}});
  }

  void receive(lws* wsi, RealtimeSession* session, const void* data, size_t length) {
    if (session->close_after_write) return;
    if (length > kMaxMessageBytes - std::min(session->incoming.size(), kMaxMessageBytes)) {
      fail(wsi, session, "realtime message exceeds 1 MiB");
      return;
    }
    session->incoming.append(static_cast<const char*>(data), length);
    if (!lws_is_final_fragment(wsi) || lws_remaining_packet_payload(wsi) > 0) return;
    std::string message = std::move(session->incoming);
    session->incoming.clear();
    try {
      const json event = json::parse(message);
      if (!event.is_object() || !event.contains("type") || !event["type"].is_string()) {
        throw ProtocolError("realtime event requires a type");
      }
      const std::string type = event["type"].get<std::string>();
      if (type == "session.update") {
        if (!event.contains("session") || !event["session"].is_object()) {
          throw ProtocolError("session.update requires a session object");
        }
        update_session(wsi, session, event["session"]);
      } else if (type == "input_audio_buffer.append") {
        // A preview in flight does not block appends: that is what makes the
        // input continuous. With server VAD a final turn must also keep
        // reading, because the next utterance is what interrupts it. Only the
        // legacy no-VAD session refuses audio after its final commit.
        if (session->request_is_final && !session->vad) {
          throw ProtocolError(
              "audio append is not accepted after transcription.done");
        }
        if (!session->ready || !event.contains("audio") || !event["audio"].is_string()) {
          throw ProtocolError("audio append requires a string after session.update");
        }
        std::string pcm = util::base64_decode(
            event["audio"].get_ref<const std::string&>(), kMaxMessageBytes,
            /*require_nonempty=*/true);
        if (pcm.empty() || pcm.size() % 2 != 0) throw ProtocolError("invalid PCM16 audio append");
        append_audio(wsi, session, pcm);
      } else if (type == "transcription.done") {
        submit_final(wsi, session);
      } else {
        throw ProtocolError("unsupported realtime event type");
      }
    } catch (const std::exception& error) {
      fail(wsi, session, error.what());
    }
  }

  void poll_sessions() {
    drain_partials();
    drain_unified();
    const auto now = Clock::now();
    for (lws* wsi : sessions_) {
      auto* session = static_cast<RealtimeSession*>(lws_wsi_user(wsi));
      if (!session) continue;
      if (!session->response_request_id.empty() &&
          now >= session->response_deadline) {
        const int64_t segment = session->response_segment;
        cancel_and_release_response(session, "timeout");
        send(wsi, session, {{"type", "response.failed"},
                            {"segment_id", segment},
                            {"error", "spoken response timed out"}});
      }
      if (session->request_id.empty()) continue;
      const std::string request_id = session->request_id;
      if (!coordinator_.wait(request_id, 0)) {
        if (now >= session->deadline) {
          coordinator_.abort(request_id);
          coordinator_.release(request_id);
          session->request_id.clear();
          // A preview that overruns its deadline is dropped; only a committed
          // final turn times the session out.
          if (session->request_is_final) {
            session->request_is_final = false;
            fail(wsi, session, "realtime transcription timed out");
          }
        }
        continue;
      }
      auto info = coordinator_.info(request_id);
      coordinator_.release(request_id);
      session->request_id.clear();
      if (!session->request_is_final) {
        // A preview failure is not a turn failure: the next window retries.
        if (info && info->state == "completed") {
          try {
            emit_preview(wsi, session, info->result);
          } catch (const std::exception& error) {
            std::cerr << "realtime preview dropped: " << error.what() << "\n";
          }
        }
        maybe_submit_preview(wsi, session);
        continue;
      }
      session->request_is_final = false;
      if (!info || info->state != "completed") {
        fail(wsi, session, info ? info->error : "request result disappeared");
        continue;
      }
      try {
        const std::string text = serve_detail::asr_result_text(info->result);
        send(wsi, session, {{"type", "transcription.segment"},
                            {"event_index", 1},
                            {"segment_id", session->segment_id},
                            {"text", text},
                            {"is_final", true}});
        json completed = {{"type", "transcription.completed"},
                          {"text", text},
                          {"segment_id", session->segment_id}};
        if (const auto metadata = live_generation_metadata(info->result)) {
          completed["generation_metadata"] = *metadata;
        }
        send(wsi, session, completed);
        start_response(session, text);
        std::cerr << "realtime turn_done segment=" << session->segment_id
                  << " previews=" << session->partial_count
                  << " chars=" << text.size() << "\n";
      } catch (const std::exception& error) {
        fail(wsi, session, error.what());
        continue;
      }
      if (session->vad) {
        // Server VAD keeps the session open; the next utterance is a new turn.
        reset_turn(session);
      } else {
        session->close_after_write = true;
      }
    }
  }

  // Starts the VLM -> TTS turn for the transcript just completed. The audio
  // arrives as audio.delta events through drain_unified().
  void start_response(RealtimeSession* session, const std::string& text) {
    if (unified_ == nullptr || !session->audio_output || text.empty()) return;
    cancel_and_release_response(session, "barge_in");
    Value::Map inputs;
    Value::Array messages;
    Value::Map message;
    message["role"] = Value("user");
    message["content"] = Value(text);
    messages.push_back(Value(std::move(message)));
    inputs["messages"] = Value(std::move(messages));
    Value::Array modalities;
    modalities.push_back(Value("text"));
    modalities.push_back(Value("audio"));
    inputs["output_modalities"] = Value(std::move(modalities));
    Value::Map audio;
    audio["voice"] = Value(session->voice);
    audio["stream"] = Value(true);
    inputs["audio_options"] = Value(std::move(audio));
    inputs["session_id"] = Value(session->unified_session);
    StagePayload payload;
    payload.request.inputs = Value(std::move(inputs));
    payload.request.metadata["request_type"] = Value("chat");
    try {
      session->response_request_id = unified_->submit(std::move(payload));
      session->response_segment = session->segment_id;
      session->response_deadline =
          Clock::now() + std::chrono::duration_cast<Clock::duration>(
                             std::chrono::duration<double>(config_.request_timeout_s));
      std::cerr << "realtime response_start segment=" << session->segment_id
                << " request=" << session->response_request_id << "\n";
    } catch (const std::exception& error) {
      session->response_request_id.clear();
      std::cerr << "realtime response_start failed: " << error.what() << "\n";
    }
  }

  // Converts queued unified events into WebSocket events. libwebsockets
  // thread only.
  void drain_unified() {
    std::deque<UnifiedEvent> batch;
    {
      std::lock_guard<std::mutex> lock(unified_mu_);
      batch.swap(unified_queue_);
    }
    const auto timeout_span = std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(config_.request_timeout_s));
    for (const UnifiedEvent& event : batch) {
      const bool terminal = event.kind == UnifiedEvent::Kind::kComplete ||
                            event.kind == UnifiedEvent::Kind::kError ||
                            event.kind == UnifiedEvent::Kind::kCancelled;
      bool matched = false;
      for (lws* wsi : sessions_) {
        auto* session = static_cast<RealtimeSession*>(lws_wsi_user(wsi));
        if (session == nullptr ||
            session->response_request_id != event.request_id) {
          continue;
        }
        matched = true;
        if (session->close_after_write) {
          if (terminal) {
            if (unified_ != nullptr) unified_->release(event.request_id);
            session->response_request_id.clear();
          }
          break;
        }
        const int64_t segment = session->response_segment;
        switch (event.kind) {
          case UnifiedEvent::Kind::kVlmDelta:
            session->response_deadline = Clock::now() + timeout_span;
            if (!event.text.empty()) {
              send(wsi, session, {{"type", "response.text.delta"},
                                  {"segment_id", segment},
                                  {"text", event.text}});
            }
            break;
          case UnifiedEvent::Kind::kArtifact: {
            session->response_deadline = Clock::now() + timeout_span;
            if (event.artifact_bytes == nullptr || event.artifact_length == 0) break;
            const std::string_view pcm(
                event.artifact_bytes->data() + event.artifact_offset,
                event.artifact_length);
            send(wsi, session, {{"type", "audio.delta"},
                                {"segment_id", segment},
                                {"format", "pcm16"},
                                {"sample_rate", 24000},
                                {"channels", 1},
                                {"audio", util::base64_encode(pcm)}});
            break;
          }
          case UnifiedEvent::Kind::kComplete:
            send(wsi, session, {{"type", "audio.done"}, {"segment_id", segment}});
            send(wsi, session,
                 {{"type", "response.completed"}, {"segment_id", segment}});
            if (unified_ != nullptr) unified_->release(event.request_id);
            session->response_request_id.clear();
            break;
          case UnifiedEvent::Kind::kError:
            send(wsi, session, {{"type", "response.failed"},
                                {"segment_id", segment},
                                {"error", event.error}});
            if (unified_ != nullptr) unified_->release(event.request_id);
            session->response_request_id.clear();
            break;
          case UnifiedEvent::Kind::kCancelled:
            if (unified_ != nullptr) unified_->release(event.request_id);
            session->response_request_id.clear();
            break;
          default:
            break;
        }
        break;
      }
      if (!matched && terminal && unified_ != nullptr &&
          event.request_id.rfind("http-", 0) != 0) {
        // Release an orphaned realtime response whose session closed or moved
        // on before the terminal event was drained; never touch HTTP SSE ids.
        if (const auto info = unified_->info(event.request_id);
            info && info->state != "running" && info->state != "pending") {
          unified_->release(event.request_id);
        }
      }
    }
  }

  void forward_event(const UnifiedEvent& event) {
    const bool terminal = event.kind == UnifiedEvent::Kind::kComplete ||
                          event.kind == UnifiedEvent::Kind::kError ||
                          event.kind == UnifiedEvent::Kind::kCancelled ||
                          event.kind == UnifiedEvent::Kind::kLegError;
    std::lock_guard<std::mutex> lock(unified_mu_);
    // Drop only incremental frames under extreme backpressure; terminal events
    // must always be delivered so UnifiedEngine records are released cleanly.
    if (!terminal && unified_queue_.size() >= 4096) return;
    unified_queue_.push_back(event);
  }

  void attach_unified(UnifiedEngine& engine) { unified_ = &engine; }

  Coordinator& coordinator_;
  ApiConfig config_;
  lws_context* context_ = nullptr;
  lws_sorted_usec_list_t poll_sul_{};
  std::atomic<bool> running_{false};
  std::atomic<size_t> active_session_count_{0};
  std::set<lws*> sessions_;
  Coordinator::StreamSink previous_stream_;
  Coordinator::CompleteSink previous_complete_;
  std::mutex partial_mu_;
  std::deque<std::pair<std::string, std::string>> partial_queue_;
  uint64_t next_request_id_ = 0;
  uint64_t next_preview_id_ = 0;
  uint64_t next_session_no_ = 0;
  UnifiedEngine* unified_ = nullptr;
  std::mutex unified_mu_;
  std::deque<UnifiedEvent> unified_queue_;
  int bound_port_ = -1;
};

namespace {

int realtime_callback(lws* wsi, lws_callback_reasons reason, void* user,
                      void* in, size_t len) {
  auto* self = static_cast<RealtimeApi::Impl*>(lws_context_user(lws_get_context(wsi)));
  auto* session = static_cast<RealtimeSession*>(user);
  switch (reason) {
    case LWS_CALLBACK_FILTER_PROTOCOL_CONNECTION:
      if (!self->authorized(wsi)) {
        lws_return_http_status(wsi, HTTP_STATUS_UNAUTHORIZED, "invalid api key");
        return -1;
      }
      if (self->at_session_capacity()) {
        lws_return_http_status(wsi, HTTP_STATUS_SERVICE_UNAVAILABLE,
                               "realtime session capacity reached");
        return -1;
      }
      {
        std::string error;
        if (!valid_handshake(wsi, self->config_.served_model_name, error)) {
          lws_return_http_status(wsi, HTTP_STATUS_BAD_REQUEST, error.c_str());
          return -1;
        }
      }
      return 0;
    case LWS_CALLBACK_HTTP:
      lws_return_http_status(wsi, HTTP_STATUS_NOT_FOUND, "realtime WebSocket only");
      return -1;
    case LWS_CALLBACK_ESTABLISHED:
      self->established(wsi, session);
      return 0;
    case LWS_CALLBACK_CLOSED:
      self->closed(wsi, session);
      return 0;
    case LWS_CALLBACK_RECEIVE:
      self->receive(wsi, session, in, len);
      return 0;
    case LWS_CALLBACK_SERVER_WRITEABLE: {
      if (!session || session->outbound.empty()) return 0;
      std::string message = std::move(session->outbound.front());
      session->outbound.pop_front();
      std::vector<unsigned char> buffer(LWS_PRE + message.size());
      std::memcpy(buffer.data() + LWS_PRE, message.data(), message.size());
      if (lws_write(wsi, buffer.data() + LWS_PRE, message.size(), LWS_WRITE_TEXT) <
          static_cast<int>(message.size())) {
        return -1;
      }
      if (!session->outbound.empty()) lws_callback_on_writable(wsi);
      if (session->close_after_write && session->outbound.empty()) {
        lws_close_reason(wsi, LWS_CLOSE_STATUS_NORMAL, nullptr, 0);
        return -1;
      }
      return 0;
    }
    default:
      return 0;
  }
}

}  // namespace

RealtimeApi::RealtimeApi(Coordinator& coordinator, ApiConfig config)
    : impl_(std::make_unique<Impl>(coordinator, std::move(config))) {}
RealtimeApi::~RealtimeApi() = default;
void RealtimeApi::install_stream_sink() { impl_->install_stream_sink(); }
void RealtimeApi::attach_unified(UnifiedEngine& engine) {
  impl_->attach_unified(engine);
}
void RealtimeApi::forward_event(const UnifiedEvent& event) {
  impl_->forward_event(event);
}
bool RealtimeApi::bind() { return impl_->bind(); }
bool RealtimeApi::listen() { return impl_->listen(); }
void RealtimeApi::stop() { impl_->stop(); }
int RealtimeApi::bound_port() const { return impl_->bound_port(); }

}  // namespace umminfer
