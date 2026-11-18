// SPDX-License-Identifier: Apache-2.0
#include "umminfer/pipeline/unified.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <string_view>
#include <utility>
#include <vector>

#include "pipeline/unified_schema.hpp"
#include "umminfer/obs/metrics.hpp"
#include "umminfer/pipeline/sentence_split.hpp"

namespace umminfer {
namespace {

double ms_since(std::chrono::steady_clock::time_point started) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - started)
      .count();
}

int64_t epoch_millis() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

SessionHistoryPolicy to_history_policy(const UnifiedConfig::SessionHistory& cfg) {
  SessionHistoryPolicy policy;
  policy.enabled = cfg.enabled;
  policy.max_turns = cfg.max_turns;
  policy.max_chars = cfg.max_chars;
  policy.max_sessions = cfg.max_sessions;
  return policy;
}

}  // namespace

struct UnifiedEngine::CallbackGate {
  explicit CallbackGate(UnifiedEngine* owner) : engine(owner) {}

  template <typename Arg>
  void dispatch(const Arg& arg, void (UnifiedEngine::*method)(const Arg&)) {
    UnifiedEngine* target = nullptr;
    {
      std::lock_guard<std::mutex> lock(mu);
      target = engine;
      if (target != nullptr) {
        ++active;
      } else {
        fallback_queue.emplace_back(arg);
      }
    }
    if (target == nullptr) {
      drain_fallback();
      return;
    }
    struct ActiveGuard {
      CallbackGate& gate;
      ~ActiveGuard() {
        std::lock_guard<std::mutex> lock(gate.mu);
        if (--gate.active == 0) gate.cv.notify_all();
      }
    } guard{*this};
    (target->*method)(arg);
  }

  void stream(const StreamMessage& msg) {
    dispatch(msg, &UnifiedEngine::handle_stream);
  }
  void complete(const RequestInfo& info) {
    dispatch(info, &UnifiedEngine::handle_complete);
  }
  void retire() noexcept {
    {
      std::unique_lock<std::mutex> lock(mu);
      engine = nullptr;
      cv.wait(lock, [&] { return active == 0; });
    }
    drain_fallback(true);
  }

  void drain_fallback(bool suppress_exceptions = false) {
    std::unique_lock<std::mutex> lock(mu);
    if (active != 0 || fallback_dispatching) return;
    fallback_dispatching = true;
    std::exception_ptr first_error;
    while (!fallback_queue.empty()) {
      FallbackItem item = std::move(fallback_queue.front());
      fallback_queue.pop_front();
      lock.unlock();
      try {
        if (const auto* msg = std::get_if<StreamMessage>(&item)) {
          if (previous_stream) previous_stream(*msg);
        } else {
          const auto& info = std::get<RequestInfo>(item);
          if (previous_complete) previous_complete(info);
        }
      } catch (...) {
        if (!first_error) first_error = std::current_exception();
      }
      lock.lock();
    }
    fallback_dispatching = false;
    if (first_error && !suppress_exceptions) std::rethrow_exception(first_error);
  }

  using FallbackItem = std::variant<StreamMessage, RequestInfo>;
  std::mutex mu;
  std::condition_variable cv;
  UnifiedEngine* engine;
  size_t active = 0;
  std::deque<FallbackItem> fallback_queue;
  bool fallback_dispatching = false;
  std::function<void(const StreamMessage&)> previous_stream;
  std::function<void(const RequestInfo&)> previous_complete;
};

bool unified_retirement_waiting_for_test(const UnifiedEngine& engine) {
  std::lock_guard<std::mutex> lock(engine.callback_gate_->mu);
  return engine.callback_gate_->engine == nullptr &&
         engine.callback_gate_->active != 0;
}

const char* to_string(UnifiedModality modality) {
  switch (modality) {
    case UnifiedModality::kAudio: return "audio";
    case UnifiedModality::kText: return "text";
    case UnifiedModality::kImage: return "image";
    case UnifiedModality::kVideo: return "video";
  }
  return "unknown";
}

std::optional<UnifiedModality> modality_from_string(const std::string& token) {
  if (token == "audio") return UnifiedModality::kAudio;
  if (token == "text") return UnifiedModality::kText;
  if (token == "image") return UnifiedModality::kImage;
  if (token == "video") return UnifiedModality::kVideo;
  return std::nullopt;
}

const char* to_string(UnifiedEvent::Kind kind) {
  switch (kind) {
    case UnifiedEvent::Kind::kAsrPartial: return "asr_partial";
    case UnifiedEvent::Kind::kAsrFinal: return "asr_final";
    case UnifiedEvent::Kind::kVlmDelta: return "vlm_delta";
    case UnifiedEvent::Kind::kArtifact: return "artifact";
    case UnifiedEvent::Kind::kLegError: return "leg_error";
    case UnifiedEvent::Kind::kComplete: return "complete";
    case UnifiedEvent::Kind::kError: return "error";
    case UnifiedEvent::Kind::kCancelled: return "cancelled";
  }
  return "unknown";
}

UnifiedEngine::UnifiedEngine(Coordinator& coordinator, UnifiedConfig config)
    : coordinator_(coordinator),
      config_(std::move(config)),
      session_store_(to_history_policy(config_.session_history)) {
  if (config_.asr_stage == config_.vlm_stage) {
    throw ProtocolError("unified engine requires distinct asr and vlm stages");
  }
  callback_gate_ = std::make_shared<CallbackGate>(this);
  auto previous = coordinator_.exchange_sinks_before_run(
      [gate = callback_gate_](const StreamMessage& msg) { gate->stream(msg); },
      [gate = callback_gate_](const RequestInfo& info) { gate->complete(info); });
  callback_gate_->previous_stream = std::move(previous.first);
  callback_gate_->previous_complete = std::move(previous.second);
}

UnifiedEngine::~UnifiedEngine() noexcept {
  callback_gate_->retire();
}

std::string UnifiedEngine::submit(StagePayload payload) {
  unified_detail::ValidatedSubmission parsed =
      unified_detail::validate_submission(payload.request.inputs, config_);

  auto vlm_params = payload.request.params;
  auto vlm_metadata = payload.request.metadata;
  if (parsed.has_audio) {
    payload.request.inputs = std::move(parsed.asr_inputs);
    payload.request.params["stream"] = Value(parsed.asr_partials);
    payload.request.params["temperature"] = Value(0.0);
  }
  std::string session_id =
      parsed.explicit_session
          ? std::move(parsed.session_id)
          : "session-" + std::to_string(session_counter_++);

  std::lock_guard<std::mutex> lock(mu_);
  if (admission_.size() >= config_.max_in_flight) {
    throw QueueFullError();
  }
  Record record;
  record.session_id = std::move(session_id);
  record.started = std::chrono::steady_clock::now();
  record.start_epoch_ms = epoch_millis();
  if (parsed.has_audio) record.asr_started = record.started;
  record.history_requested = parsed.history_requested;
  record.original_inputs = std::move(parsed.original_inputs);
  record.vlm_params = std::move(vlm_params);
  record.vlm_metadata = std::move(vlm_metadata);
  record.user_text = std::move(parsed.user_text);
  record.image_options = std::move(parsed.image_options);
  record.audio_options = std::move(parsed.audio_options);
  record.image_leg.enabled = parsed.image_output;
  record.image_leg.stage_name = config_.image_stage;
  record.image_leg.stage_id = kImageStageId;
  record.tts_enabled = parsed.audio_output;
  if (parsed.audio_output) {
    const auto voice = record.audio_options.find("voice");
    if (voice != record.audio_options.end() && voice->second.is_str()) {
      record.tts_voice = voice->second.as_str();
    }
    const auto stream = record.audio_options.find("stream");
    record.tts_stream = stream != record.audio_options.end() &&
                        stream->second.is_bool() && stream->second.as_bool() &&
                        !parsed.image_output;
  }
  for (Value::Map* metadata : {&record.vlm_metadata, &payload.request.metadata}) {
    (*metadata)["request_type"] = Value("chat");
  }

  std::string request_id;
  if (parsed.has_audio) {
    request_id = coordinator_.submit(payload, config_.asr_stage);
  } else {
    StagePayload vlm_payload;
    vlm_payload.request_id = payload.request_id;
    vlm_payload.request.inputs =
        unified_detail::build_vlm_inputs("", record.original_inputs);
    vlm_payload.request.inputs["session_id"] = Value(record.session_id);
    if (record.history_requested) {
      vlm_payload.request.inputs["messages"] = Value(
          session_store_.prepend_history(
              record.session_id,
              vlm_payload.request.inputs["messages"].as_array()));
    }
    vlm_payload.request.params = record.vlm_params;
    vlm_payload.request.metadata = record.vlm_metadata;
    record.phase = Phase::kVlm;
    request_id = coordinator_.submit(vlm_payload, config_.vlm_stage);
    record.vlm_id = request_id;
    record.vlm_started = record.started;
  }
  admission_.push_back(request_id);
  record.public_info.request_id = request_id;
  record.public_info.state = "running";
  record.public_info.current_stage =
      parsed.has_audio ? kAsrStageId : kVlmStageId;
  const std::string initial_stage = record.public_info.current_stage;
  const std::string req_type = record.request_type;
  const int64_t start_epoch = record.start_epoch_ms;
  records_[request_id] = std::move(record);
  obs::Metrics::instance().queue_depth("unified_admission", admission_.size());
  obs::Metrics::instance().trace_start(request_id, req_type, initial_stage,
                                       start_epoch);
  return request_id;
}

UnifiedEngine::CancelStatus UnifiedEngine::cancel_for_session(
    const std::string& request_id, std::string_view expected_session_id,
    const char* source) {
  std::vector<std::string> active_ids;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = records_.find(request_id);
    if (it == records_.end()) return CancelStatus::kNotFoundOrTerminal;
    if (!expected_session_id.empty() &&
        it->second.session_id != expected_session_id) {
      return CancelStatus::kSessionMismatch;
    }
    if (it->second.phase == Phase::kDone) {
      return CancelStatus::kNotFoundOrTerminal;
    }
    Record& record = it->second;
    record.cancel_requested = true;
    if (!record.vlm_id.empty()) active_ids.push_back(record.vlm_id);
    if (record.image_leg.submitted) active_ids.push_back(record.image_leg.child_id);
    for (const OutputLeg& leg : record.tts_legs) {
      if (leg.submitted) active_ids.push_back(leg.child_id);
    }
    if (active_ids.empty()) {
      active_ids.push_back(request_id);
    }
  }
  for (const std::string& id : active_ids) coordinator_.abort(id, source);
  return CancelStatus::kCancelled;
}

bool UnifiedEngine::cancel(const std::string& request_id, const char* source) {
  return cancel_for_session(request_id, {}, source) == CancelStatus::kCancelled;
}

bool UnifiedEngine::wait(const std::string& request_id, double timeout_s) {
  std::unique_lock<std::mutex> lock(mu_);
  return cv_.wait_for(lock, std::chrono::duration<double>(timeout_s), [&] {
    auto it = records_.find(request_id);
    return it != records_.end() && it->second.terminal_dispatched;
  });
}

std::optional<RequestInfo> UnifiedEngine::info(const std::string& request_id) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = records_.find(request_id);
  if (it == records_.end()) return std::nullopt;
  return it->second.public_info;
}

bool UnifiedEngine::release(const std::string& request_id) {
  std::vector<std::string> child_ids;
  bool removed = false;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = records_.find(request_id);
    if (it == records_.end()) return false;
    if (!it->second.vlm_id.empty()) {
      child_ids.push_back(it->second.vlm_id);
      if (it->second.vlm_id != request_id) {
        child_to_unified_.erase(it->second.vlm_id);
      }
    }
    if (it->second.image_leg.submitted) {
      child_ids.push_back(it->second.image_leg.child_id);
      child_to_unified_.erase(it->second.image_leg.child_id);
    }
    for (const OutputLeg& leg : it->second.tts_legs) {
      if (leg.submitted) {
        child_ids.push_back(leg.child_id);
        child_to_unified_.erase(leg.child_id);
      }
    }
    records_.erase(it);
    for (auto queue_it = admission_.begin(); queue_it != admission_.end();
         ++queue_it) {
      if (*queue_it == request_id) {
        admission_.erase(queue_it);
        break;
      }
    }
    removed = true;
  }
  coordinator_.release(request_id);
  for (const std::string& id : child_ids) {
    if (!id.empty()) coordinator_.release(id);
  }
  return removed;
}

size_t UnifiedEngine::in_flight() const {
  std::lock_guard<std::mutex> lock(mu_);
  return admission_.size();
}

std::map<std::string, UnifiedEngine::Record>::iterator
UnifiedEngine::find_record_locked(const std::string& event_request_id) {
  auto it = records_.find(event_request_id);
  if (it != records_.end()) return it;
  const auto mapped = child_to_unified_.find(event_request_id);
  if (mapped != child_to_unified_.end()) {
    return records_.find(mapped->second);
  }
  return records_.end();
}

void UnifiedEngine::handle_stream(const StreamMessage& msg) {
  std::vector<std::pair<std::string, const char*>> deferred_aborts;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto record_it = find_record_locked(msg.request_id);
    if (record_it != records_.end() && record_it->second.phase != Phase::kDone) {
      Record& record = record_it->second;
      const std::string& unified_id = record_it->first;
      if (msg.from_stage == config_.vlm_stage) {
        if (record.phase == Phase::kVlm) {
          const std::string delta(unified_detail::chunk_text(msg.chunk));
          record.vlm_text += delta;
          if (record.tts_stream && !record.tts_failed) {
            record.sentence_buf += delta;
            drain_sentences_locked(unified_id, record);
          }
          UnifiedEvent event;
          event.request_id = unified_id;
          event.session_id = record.session_id;
          event.stage_id = kVlmStageId;
          event.kind = UnifiedEvent::Kind::kVlmDelta;
          event.seq = record.next_seq++;
          event.text = delta;
          dispatch_queue_.push_back(std::move(event));
        }
      } else if (msg.from_stage == config_.tts_stage && record.tts_enabled) {
        if (record.tts_stream) {
          emit_tts_chunk_locked(unified_id, record, msg.chunk, deferred_aborts);
        }
      } else if (msg.from_stage == record.image_leg.stage_name) {
        // Image leg delivers its PNG bytes in complete.result.
      } else if (record.phase != Phase::kVlm) {
        const bool is_final = unified_detail::chunk_is_final(msg.chunk);
        bool emit = true;
        if (is_final) {
          if (record.asr_final_seen) {
            emit = false;
          } else {
            const auto text = unified_detail::asr_result_text(msg.chunk);
            if (!text) {
              emit = false;
              dispatch_queue_.push_back(finish_locked(
                  unified_id, record, UnifiedEvent::Kind::kError,
                  "ASR stage returned no text", {}));
            } else {
              record.asr_final_seen = true;
              record.transcript = *text;
            }
          }
        } else {
          record.transcript += unified_detail::chunk_text(msg.chunk);
        }
        if (emit) {
          UnifiedEvent event;
          event.request_id = unified_id;
          event.session_id = record.session_id;
          event.stage_id = kAsrStageId;
          event.kind = is_final ? UnifiedEvent::Kind::kAsrFinal
                                : UnifiedEvent::Kind::kAsrPartial;
          event.seq = record.next_seq++;
          event.is_final = is_final;
          event.text = unified_detail::chunk_text(msg.chunk);
          dispatch_queue_.push_back(std::move(event));
        }
      }
    }
    dispatch_queue_.push_back(msg);
  }
  for (const auto& [child_id, reason] : deferred_aborts) {
    coordinator_.abort(child_id, reason);
  }
  drain_dispatch();
}

void UnifiedEngine::drain_dispatch() {
  std::unique_lock<std::mutex> lock(mu_);
  if (dispatching_) return;
  dispatching_ = true;
  std::exception_ptr first_error;
  while (!dispatch_queue_.empty()) {
    DispatchItem item = std::move(dispatch_queue_.front());
    dispatch_queue_.pop_front();
    lock.unlock();
    bool notify_terminal = false;
    std::string terminal_request_id;
    if (const auto* event = std::get_if<UnifiedEvent>(&item)) {
      const bool terminal = event->kind == UnifiedEvent::Kind::kComplete ||
                            event->kind == UnifiedEvent::Kind::kError ||
                            event->kind == UnifiedEvent::Kind::kCancelled;
      if (terminal) {
        notify_terminal = true;
        terminal_request_id = event->request_id;
      }
      try {
        if (on_event) on_event(*event);
      } catch (...) {
        if (!first_error) first_error = std::current_exception();
      }
      try {
        if (terminal && on_terminal) on_terminal(*event);
      } catch (...) {
        if (!first_error) first_error = std::current_exception();
      }
    } else if (const auto* msg = std::get_if<StreamMessage>(&item)) {
      try {
        if (callback_gate_->previous_stream) {
          callback_gate_->previous_stream(*msg);
        }
      } catch (...) {
        if (!first_error) first_error = std::current_exception();
      }
    } else {
      const auto& info = std::get<RequestInfo>(item);
      try {
        if (callback_gate_->previous_complete) {
          callback_gate_->previous_complete(info);
        }
      } catch (...) {
        if (!first_error) first_error = std::current_exception();
      }
    }
    lock.lock();
    if (notify_terminal) {
      auto it = records_.find(terminal_request_id);
      if (it != records_.end()) it->second.terminal_dispatched = true;
      cv_.notify_all();
    }
  }
  dispatching_ = false;
  if (first_error) std::rethrow_exception(first_error);
}

void UnifiedEngine::start_vlm_locked(const std::string& request_id,
                                     Record& record) {
  StagePayload vlm_payload;
  record.vlm_id = request_id + ":vlm";
  child_to_unified_[record.vlm_id] = request_id;
  vlm_payload.request_id = record.vlm_id;
  vlm_payload.request.inputs =
      unified_detail::build_vlm_inputs(record.transcript, record.original_inputs);
  vlm_payload.request.inputs["session_id"] = Value(record.session_id);
  if (record.history_requested) {
    vlm_payload.request.inputs["messages"] = Value(
        session_store_.prepend_history(
            record.session_id,
            vlm_payload.request.inputs["messages"].as_array()));
  }
  vlm_payload.request.params = record.vlm_params;
  vlm_payload.request.metadata = record.vlm_metadata;
  record.phase = Phase::kVlm;
  record.public_info.current_stage = kVlmStageId;
  record.vlm_started = std::chrono::steady_clock::now();
  coordinator_.submit(vlm_payload, config_.vlm_stage);
}

void UnifiedEngine::record_session_turn_locked(const Record& record) {
  const std::string& user =
      record.transcript.empty() ? record.user_text : record.transcript;
  session_store_.record_turn(record.session_id, user, record.vlm_text);
}

UnifiedEvent UnifiedEngine::finish_locked(const std::string& request_id,
                                          Record& record,
                                          UnifiedEvent::Kind kind,
                                          std::string error, Value result) {
  record.phase = Phase::kDone;
  UnifiedEvent event;
  event.request_id = request_id;
  event.session_id = record.session_id;
  event.kind = kind;
  event.is_final = true;
  event.seq = record.next_seq++;
  event.error = std::move(error);
  event.result = std::move(result);
  if (kind == UnifiedEvent::Kind::kComplete) {
    event.stage_id = kVlmStageId;
    event.text = record.vlm_text;
  } else if (kind == UnifiedEvent::Kind::kError) {
    event.stage_id = record.vlm_id.empty() ? kAsrStageId : kVlmStageId;
  }
  record.public_info.state = kind == UnifiedEvent::Kind::kComplete
                                 ? "completed"
                                 : kind == UnifiedEvent::Kind::kError ? "failed"
                                                                      : "aborted";
  record.public_info.current_stage = event.stage_id;
  record.public_info.result = kind == UnifiedEvent::Kind::kComplete
                                  ? event.result
                                  : Value{};
  record.public_info.error = event.error;
  for (auto it = admission_.begin(); it != admission_.end(); ++it) {
    if (*it == request_id) {
      admission_.erase(it);
      break;
    }
  }
  const double total_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - record.started)
          .count();
  obs::Metrics::instance().request(record.request_type, record.public_info.state,
                                   total_seconds);
  obs::Metrics::instance().queue_depth("unified_admission", admission_.size());
  if (obs::Metrics::instance().trace_enabled()) {
    obs::TraceRecord trace;
    trace.request_id = request_id;
    trace.stage = event.stage_id;
    trace.request_type = record.request_type;
    trace.start_epoch_ms = record.start_epoch_ms;
    trace.end_epoch_ms = epoch_millis();
    trace.status = record.public_info.state;
    trace.duration_ms = total_seconds * 1000.0;
    trace.asr_ms = record.asr_ms;
    trace.vlm_ms = record.vlm_ms;
    trace.image_ms = record.image_ms;
    trace.tts_ms = record.tts_ms;
    obs::Metrics::instance().trace(trace);
  }
  return event;
}

void UnifiedEngine::handle_complete(const RequestInfo& info) {
  std::vector<std::pair<std::string, const char*>> deferred_aborts;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = find_record_locked(info.request_id);
    if (it != records_.end() && it->second.phase != Phase::kDone) {
      Record& record = it->second;
      const std::string& unified_id = it->first;
      const bool image_child =
          record.image_leg.submitted &&
          info.request_id == record.image_leg.child_id;
      OutputLeg* output_leg = image_child ? &record.image_leg : nullptr;
      if (output_leg == nullptr) {
        for (OutputLeg& leg : record.tts_legs) {
          if (leg.submitted && leg.child_id == info.request_id) {
            output_leg = &leg;
            break;
          }
        }
      }
      const auto record_primary_leg = [&record, &unified_id](
                                          const std::string& status) {
        const bool vlm = !record.vlm_id.empty();
        const double ms =
            ms_since(vlm ? record.vlm_started : record.asr_started);
        if (vlm) {
          record.vlm_ms = ms;
        } else {
          record.asr_ms = ms;
        }
        const char* leg_name = vlm ? "vlm" : "asr";
        obs::Metrics::instance().leg(leg_name, record.request_type, status,
                                     ms / 1000.0);
        obs::Metrics::instance().trace_leg(unified_id, leg_name, status, ms);
      };
      const auto record_output_phases = [](const OutputLeg& leg,
                                           const Value& result) {
        if (leg.stage_id == kImageStageId) {
          const double denoise = obs::result_ms(result, "denoise_ms");
          const double steps = obs::result_ms(result, "steps");
          if (denoise >= 0 && steps > 0) {
            obs::Metrics::instance().image_denoise_step(denoise / steps / 1000.0);
          }
        } else if (leg.stage_id == kTtsStageId) {
          const double first = obs::result_ms(result, "first_frame_ms");
          if (first >= 0) {
            obs::Metrics::instance().tts_phase("first_frame", first / 1000.0);
          }
          const double total = obs::result_ms(result, "total_ms");
          if (total >= 0) {
            obs::Metrics::instance().tts_phase("total", total / 1000.0);
          }
        }
      };
      if (output_leg != nullptr) {
        if (info.state == "aborted" || record.cancel_requested) {
          settle_output_leg_locked(
              unified_id, record, *output_leg,
              UnifiedEvent::Kind::kCancelled,
              info.state == "failed"
                  ? (info.error.empty() ? "stage failure" : info.error)
                  : "request aborted");
        } else if (info.state == "failed") {
          settle_output_leg_locked(
              unified_id, record, *output_leg,
              UnifiedEvent::Kind::kLegError,
              info.error.empty() ? "stage failure" : info.error);
        } else {
          try {
            record_output_phases(*output_leg, info.result);
            const Value* streamed = info.result.find("streamed");
            const bool was_streamed =
                streamed != nullptr && streamed->is_bool() && streamed->as_bool();
            if (!was_streamed) {
              emit_artifact_locked(unified_id, record, *output_leg, info.result);
            }
            if (output_leg->stage_id == kTtsStageId && record.tts_failed) {
              settle_output_leg_locked(
                  unified_id, record, *output_leg,
                  UnifiedEvent::Kind::kLegError,
                  record.tts_error.empty()
                      ? "streamed audio exceeds the artifact limit"
                      : record.tts_error);
            } else {
              settle_output_leg_locked(unified_id, record, *output_leg,
                                       UnifiedEvent::Kind::kComplete, {});
            }
          } catch (const std::exception& error) {
            settle_output_leg_locked(
                unified_id, record, *output_leg,
                UnifiedEvent::Kind::kLegError, error.what());
          }
        }
      } else if (info.state == "aborted") {
        record_primary_leg("aborted");
        collect_tts_aborts_locked(record, "client", deferred_aborts);
        dispatch_queue_.push_back(finish_locked(
            unified_id, record, UnifiedEvent::Kind::kCancelled, {}, {}));
      } else if (info.state == "failed") {
        record_primary_leg("failed");
        collect_tts_aborts_locked(record, "vlm_failure", deferred_aborts);
        dispatch_queue_.push_back(finish_locked(
            unified_id, record, UnifiedEvent::Kind::kError,
            info.error.empty() ? "stage failure" : info.error, {}));
      } else if (record.cancel_requested) {
        collect_tts_aborts_locked(record, "client", deferred_aborts);
        dispatch_queue_.push_back(finish_locked(
            unified_id, record, UnifiedEvent::Kind::kCancelled, {}, {}));
      } else if (record.vlm_id.empty()) {
        record_primary_leg("completed");
        if (!record.asr_final_seen) {
          const auto text = unified_detail::asr_result_text(info.result);
          if (!text) {
            dispatch_queue_.push_back(finish_locked(
                unified_id, record, UnifiedEvent::Kind::kError,
                "ASR stage returned no text", {}));
          } else {
            record.asr_final_seen = true;
            record.transcript = *text;
            UnifiedEvent event;
            event.request_id = unified_id;
            event.session_id = record.session_id;
            event.stage_id = kAsrStageId;
            event.kind = UnifiedEvent::Kind::kAsrFinal;
            event.seq = record.next_seq++;
            event.is_final = true;
            event.text = *text;
            dispatch_queue_.push_back(std::move(event));
          }
        }
        if (record.phase != Phase::kDone) {
          try {
            start_vlm_locked(unified_id, record);
          } catch (const std::exception& error) {
            dispatch_queue_.push_back(finish_locked(
                unified_id, record, UnifiedEvent::Kind::kError,
                error.what(), {}));
          }
        }
      } else {
        const auto text = unified_detail::vlm_result_text(info.result);
        record_primary_leg(text ? "completed" : "failed");
        if (!text) {
          collect_tts_aborts_locked(record, "vlm_failure", deferred_aborts);
          dispatch_queue_.push_back(finish_locked(
              unified_id, record, UnifiedEvent::Kind::kError,
              "VLM stage returned no text", {}));
        } else {
          const double prefill = obs::result_ms(info.result, "prefill_ms");
          if (prefill >= 0) {
            obs::Metrics::instance().vlm_phase("prefill", prefill / 1000.0);
          }
          const double decode = obs::result_ms(info.result, "decode_ms");
          if (decode >= 0) {
            obs::Metrics::instance().vlm_phase("decode", decode / 1000.0);
          }
          if (record.vlm_text.empty()) record.vlm_text = *text;
          record.vlm_result = info.result;
          record.vlm_done = true;
          if (record.image_leg.enabled || record.tts_enabled) {
            try {
              start_outputs_locked(unified_id, record);
            } catch (const std::exception& error) {
              collect_tts_aborts_locked(record, "vlm_failure", deferred_aborts);
              dispatch_queue_.push_back(finish_locked(
                  unified_id, record, UnifiedEvent::Kind::kError,
                  error.what(), {}));
            }
          } else {
            dispatch_queue_.push_back(finish_locked(
                unified_id, record, UnifiedEvent::Kind::kComplete, {},
                info.result));
            if (record.history_requested) record_session_turn_locked(record);
          }
        }
      }
    }
    dispatch_queue_.push_back(info);
  }
  for (const auto& [child_id, reason] : deferred_aborts) {
    coordinator_.abort(child_id, reason);
  }
  drain_dispatch();
}

void UnifiedEngine::start_image_locked(const std::string& request_id,
                                       Record& record) {
  OutputLeg& leg = record.image_leg;
  leg.child_id = request_id + kImageSuffix;
  child_to_unified_[leg.child_id] = request_id;
  StagePayload payload;
  payload.request_id = leg.child_id;
  payload.request.inputs =
      unified_detail::build_image_inputs(record.image_options, record.vlm_text);
  payload.request.params = record.vlm_params;
  payload.request.metadata = record.vlm_metadata;
  leg.submitted = true;
  leg.started = std::chrono::steady_clock::now();
  try {
    coordinator_.submit(payload, leg.stage_name);
  } catch (...) {
    leg.submitted = false;
    child_to_unified_.erase(leg.child_id);
    throw;
  }
  ++record.legs_remaining;
}

void UnifiedEngine::start_tts_locked(const std::string& request_id,
                                     Record& record, std::string text,
                                     bool first) {
  OutputLeg leg;
  leg.enabled = true;
  leg.stage_name = config_.tts_stage;
  leg.stage_id = kTtsStageId;
  leg.child_id = (!record.tts_stream && first)
                     ? request_id + kTtsSuffix
                     : request_id + kTtsSuffix + ":" +
                           std::to_string(record.tts_legs.size());
  child_to_unified_[leg.child_id] = request_id;
  StagePayload payload;
  payload.request_id = leg.child_id;
  payload.request.inputs = unified_detail::build_tts_inputs(
      record.audio_options, text, record.tts_stream);
  payload.request.params = record.vlm_params;
  payload.request.metadata = record.vlm_metadata;
  leg.submitted = true;
  leg.started = std::chrono::steady_clock::now();
  try {
    coordinator_.submit(payload, leg.stage_name);
  } catch (...) {
    child_to_unified_.erase(leg.child_id);
    throw;
  }
  record.tts_legs.push_back(std::move(leg));
  ++record.legs_remaining;
}

void UnifiedEngine::drain_sentences_locked(const std::string& request_id,
                                           Record& record) {
  constexpr size_t kMaxSentenceCodepoints = 48;
  constexpr size_t kFirstClauseCodepoints = 8;
  // Reserve the final slot for flush_sentences_locked so streamed TTS fan-out
  // never exceeds kMaxStreamedTtsLegs child requests per unified request.
  while (record.tts_legs.size() + 1 < kMaxStreamedTtsLegs) {
    size_t cut = sentence_cut(record.sentence_buf, kMaxSentenceCodepoints);
    if (cut == 0 && record.tts_legs.empty()) {
      cut = clause_cut(record.sentence_buf, kFirstClauseCodepoints);
    }
    if (cut == 0) break;
    std::string sentence = record.sentence_buf.substr(0, cut);
    record.sentence_buf.erase(0, cut);
    const size_t begin = sentence.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) continue;
    sentence = sentence.substr(
        begin, sentence.find_last_not_of(" \t\r\n") - begin + 1);
    start_tts_locked(request_id, record, std::move(sentence), false);
  }
}

void UnifiedEngine::flush_sentences_locked(const std::string& request_id,
                                           Record& record) {
  if (record.tts_failed) {
    record.sentence_buf.clear();
    return;
  }
  std::string tail = record.sentence_buf;
  record.sentence_buf.clear();
  const size_t begin = tail.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    if (!record.tts_legs.empty()) return;
    tail = record.vlm_text;
  } else {
    tail = tail.substr(begin, tail.find_last_not_of(" \t\r\n") - begin + 1);
  }
  if (tail.empty()) return;
  start_tts_locked(request_id, record, std::move(tail), record.tts_legs.empty());
}

void UnifiedEngine::start_outputs_locked(const std::string& request_id,
                                         Record& record) {
  record.phase = Phase::kOutputs;
  record.public_info.current_stage =
      record.image_leg.enabled ? kImageStageId : kTtsStageId;
  if (record.image_leg.enabled) start_image_locked(request_id, record);
  if (record.tts_enabled) {
    if (record.tts_stream) {
      flush_sentences_locked(request_id, record);
    } else {
      start_tts_locked(request_id, record, record.vlm_text, true);
    }
  }
  if (record.legs_remaining == 0) on_legs_drained_locked(request_id, record);
}

void UnifiedEngine::emit_tts_chunk_locked(
    const std::string& request_id, Record& record, const Value& chunk,
    std::vector<std::pair<std::string, const char*>>& deferred_aborts) {
  if (record.tts_failed) return;
  const Value* audio = chunk.find("audio");
  if (audio == nullptr || !audio->is_bin() || audio->as_bin().empty()) return;
  const size_t size = audio->as_bin().size();
  if (size > kMaxArtifactAudioBytes ||
      record.tts_bytes + size > kMaxArtifactAggregateBytes) {
    record.tts_failed = true;
    record.tts_error = "streamed audio exceeds the artifact limit";
    collect_tts_aborts_locked(record, "artifact_limit", deferred_aborts);
    return;
  }
  record.tts_bytes += size;
  Value::Map descriptor;
  descriptor["artifact_id"] =
      Value("art-" + std::to_string(record.next_artifact++));
  descriptor["media_type"] = Value("audio");
  descriptor["mime_type"] = Value("audio/pcm");
  descriptor["format"] = Value("pcm16");
  descriptor["sample_rate"] = Value(static_cast<int64_t>(24000));
  descriptor["channels"] = Value(static_cast<int64_t>(1));
  descriptor["index"] = Value(static_cast<int64_t>(0));
  descriptor["total"] = Value(static_cast<int64_t>(1));
  if (const Value* index = chunk.find("index")) {
    descriptor["stream_index"] = *index;
  }
  UnifiedEvent event;
  event.request_id = request_id;
  event.session_id = record.session_id;
  event.stage_id = kTtsStageId;
  event.kind = UnifiedEvent::Kind::kArtifact;
  event.seq = record.next_seq++;
  event.is_final = true;
  event.artifact = Value(std::move(descriptor));
  event.artifact_bytes = std::make_shared<const std::string>(audio->as_bin());
  event.artifact_offset = 0;
  event.artifact_length = size;
  dispatch_queue_.push_back(std::move(event));
}

void UnifiedEngine::collect_tts_aborts_locked(
    const Record& record, const char* reason,
    std::vector<std::pair<std::string, const char*>>& out) {
  for (const OutputLeg& leg : record.tts_legs) {
    if (leg.submitted && leg.state.empty()) {
      out.emplace_back(leg.child_id, reason);
    }
  }
}

void UnifiedEngine::emit_artifact_locked(const std::string& request_id,
                                         Record& record,
                                         const OutputLeg& leg,
                                         const Value& result) {
  std::vector<UnifiedEvent> events = unified_detail::slice_artifact_events(
      request_id, record.session_id, leg.stage_id, result,
      record.artifact_bytes_total, record.next_artifact, record.next_seq);
  for (UnifiedEvent& event : events) {
    dispatch_queue_.push_back(std::move(event));
  }
}

void UnifiedEngine::settle_output_leg_locked(
    const std::string& request_id, Record& record, OutputLeg& leg,
    UnifiedEvent::Kind kind, std::string error) {
  const bool aborted_for_limit =
      kind == UnifiedEvent::Kind::kCancelled && !record.cancel_requested &&
      record.tts_failed;
  leg.state = kind == UnifiedEvent::Kind::kComplete ? "completed"
            : (kind == UnifiedEvent::Kind::kCancelled && !aborted_for_limit)
                ? "aborted"
                : "failed";
  if (aborted_for_limit && !record.tts_error.empty()) {
    error = record.tts_error;
  }
  leg.error = error;
  {
    const double ms = ms_since(leg.started);
    const bool is_tts = leg.stage_id == kTtsStageId;
    if (is_tts) record.tts_ms = ms;
    else record.image_ms = ms;
    const char* leg_name = is_tts ? "tts" : "image";
    obs::Metrics::instance().leg(leg_name, record.request_type, leg.state,
                                 ms / 1000.0);
    obs::Metrics::instance().trace_leg(request_id, leg_name, leg.state, ms);
  }
  if (kind != UnifiedEvent::Kind::kComplete) {
    if (leg.stage_id == kTtsStageId) {
      record.tts_failed = true;
      if (record.tts_error.empty()) record.tts_error = error;
    }
    UnifiedEvent event;
    event.request_id = request_id;
    event.session_id = record.session_id;
    event.stage_id = leg.stage_id;
    event.kind = UnifiedEvent::Kind::kLegError;
    event.seq = record.next_seq++;
    event.error = std::move(error);
    dispatch_queue_.push_back(std::move(event));
  }
  --record.legs_remaining;
  if (record.vlm_done && record.legs_remaining == 0) {
    on_legs_drained_locked(request_id, record);
  }
}

Value UnifiedEngine::legs_value_locked(const Record& record) const {
  Value::Array legs;
  const auto add = [&](const OutputLeg& leg) {
    if (!leg.enabled) return;
    Value::Map item;
    item["stage"] = Value(leg.stage_id);
    item["state"] = Value(leg.state);
    if (!leg.error.empty()) item["error"] = Value(leg.error);
    legs.push_back(Value(std::move(item)));
  };
  add(record.image_leg);
  if (record.tts_enabled) {
    Value::Map item;
    item["stage"] = Value(kTtsStageId);
    bool all_aborted = !record.tts_legs.empty();
    for (const OutputLeg& leg : record.tts_legs) {
      if (leg.state != "aborted") all_aborted = false;
    }
    item["state"] = Value(record.tts_failed ? "failed"
                        : all_aborted       ? "aborted"
                                            : "completed");
    if (record.tts_failed && !record.tts_error.empty()) {
      item["error"] = Value(record.tts_error);
    }
    legs.push_back(Value(std::move(item)));
  }
  return Value(std::move(legs));
}

void UnifiedEngine::on_legs_drained_locked(const std::string& request_id,
                                           Record& record) {
  const UnifiedEvent::Kind kind = record.cancel_requested
                                      ? UnifiedEvent::Kind::kCancelled
                                      : UnifiedEvent::Kind::kComplete;
  UnifiedEvent event =
      finish_locked(request_id, record, kind, {}, record.vlm_result);
  event.legs = legs_value_locked(record);
  dispatch_queue_.push_back(std::move(event));
  if (kind == UnifiedEvent::Kind::kComplete && record.history_requested) {
    record_session_turn_locked(record);
  }
}

}  // namespace umminfer
