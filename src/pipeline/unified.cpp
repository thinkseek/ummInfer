// SPDX-License-Identifier: Apache-2.0
#include "umminfer/pipeline/unified.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <string_view>
#include <utility>
#include <vector>

#include "pipeline/narrative_parser.hpp"
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

const char* to_string(LegState state) {
  switch (state) {
    case LegState::kPending: return "pending";
    case LegState::kRunning: return "running";
    case LegState::kCompleted: return "completed";
    case LegState::kFailed: return "failed";
    case LegState::kAborted: return "aborted";
  }
  return "unknown";
}

const char* to_string(UnifiedEvent::Kind kind) {
  switch (kind) {
    case UnifiedEvent::Kind::kAsrPartial: return "asr_partial";
    case UnifiedEvent::Kind::kAsrFinal: return "asr_final";
    case UnifiedEvent::Kind::kVlmDelta: return "vlm_delta";
    case UnifiedEvent::Kind::kVisualSlot: return "visual_slot";
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
    // Sampling knobs belong to the VLM leg only. The ASR leg is deterministic
    // and must not see them. `record.vlm_params` already holds the copy above.
    payload.request.params.erase("top_p");
    payload.request.params.erase("seed");
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
  record.has_audio = parsed.has_audio;
  record.asr_state = parsed.has_audio ? LegState::kRunning : LegState::kPending;
  record.vlm_state = parsed.has_audio ? LegState::kPending : LegState::kRunning;
  record.started = std::chrono::steady_clock::now();
  record.start_epoch_ms = epoch_millis();
  if (parsed.has_audio) record.asr_started = record.started;
  record.history_requested = parsed.history_requested;
  record.realtime_playback_ack = parsed.realtime_playback_ack;
  record.original_inputs = std::move(parsed.original_inputs);
  record.vlm_params = std::move(vlm_params);
  record.vlm_metadata = std::move(vlm_metadata);
  record.user_text = std::move(parsed.user_text);
  record.image_options = std::move(parsed.image_options);
  record.audio_options = std::move(parsed.audio_options);
  record.video_options = std::move(parsed.video_options);
  record.narrative_enabled = parsed.narrative_enabled;
  record.narrative_visual_mode = parsed.narrative_visual_mode;
  record.narrative_max_slots = parsed.narrative_max_slots;
  record.narrative_allow_text_only = parsed.narrative_allow_text_only;
  if (record.narrative_enabled) {
    record.narrative_parser = std::make_shared<NarrativeStreamParser>();
  }
  record.image_leg.modality = UnifiedModality::kImage;
  record.image_leg.enabled = parsed.image_output;
  record.image_leg.stage_name = config_.image_stage;
  record.image_leg.stage_id = kImageStageId;
  record.image_leg.state = LegState::kPending;
  record.video_leg.modality = UnifiedModality::kVideo;
  record.video_leg.enabled = parsed.video_output;
  record.video_leg.stage_name = config_.video_stage;
  record.video_leg.stage_id = kVideoStageId;
  record.video_leg.state = LegState::kPending;
  record.tts_enabled = parsed.audio_output;
  if (parsed.audio_output) {
    const auto voice = record.audio_options.find("voice");
    if (voice != record.audio_options.end() && voice->second.is_str()) {
      record.tts_voice = voice->second.as_str();
    }
    const auto stream = record.audio_options.find("stream");
    // Global streaming voice: sentences are spoken from the first delta while
    // the VLM generates and visual legs render. Prose is cleaned of narrative
    // directives before it reaches the sentence pipeline.
    record.tts_stream = stream != record.audio_options.end() &&
                        stream->second.is_bool() && stream->second.as_bool();
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
    if (record.narrative_enabled) {
      unified_detail::prepend_narrative_instructions(
          vlm_payload.request.inputs, record.narrative_visual_mode,
          record.narrative_max_slots);
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
  trace_legs_locked(request_id, "submit", records_[request_id]);
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
    record.for_each_single_output_leg([&](const OutputLeg& leg) {
      if (leg.submitted) active_ids.push_back(leg.child_id);
    });
    for (const OutputLeg& leg : record.slot_legs) {
      if (leg.submitted) active_ids.push_back(leg.child_id);
    }
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

UnifiedEngine::BargeInResult UnifiedEngine::cancel_realtime_barge_in(
    const std::string& request_id, std::string_view expected_session_id,
    size_t acknowledged_text_cursor) {
  BargeInResult result;
  result.acknowledged_text_cursor = acknowledged_text_cursor;
  std::vector<std::string> active_ids;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = records_.find(request_id);
    if (it == records_.end() || it->second.session_id != expected_session_id ||
        !it->second.realtime_playback_ack) {
      return result;
    }
    Record& record = it->second;
    result.accepted = true;
    if (record.phase != Phase::kDone) {
      result.generation_cancelled = true;
      record.cancel_requested = true;
      if (!record.vlm_id.empty()) active_ids.push_back(record.vlm_id);
      record.for_each_single_output_leg([&](const OutputLeg& leg) {
        if (leg.submitted) active_ids.push_back(leg.child_id);
      });
      for (const OutputLeg& leg : record.slot_legs) {
        if (leg.submitted) active_ids.push_back(leg.child_id);
      }
      for (const OutputLeg& leg : record.tts_legs) {
        if (leg.submitted) active_ids.push_back(leg.child_id);
      }
      if (active_ids.empty()) active_ids.push_back(request_id);
    }
    result.history_recorded = record_realtime_history_locked(
        record, acknowledged_text_cursor, result.assistant_prefix);
  }
  for (const std::string& id : active_ids) coordinator_.abort(id, "barge_in");
  return result;
}

bool UnifiedEngine::complete_realtime_playback(
    const std::string& request_id, std::string_view expected_session_id,
    size_t acknowledged_text_cursor) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = records_.find(request_id);
  if (it == records_.end() || it->second.session_id != expected_session_id ||
      !it->second.realtime_playback_ack ||
      it->second.phase != Phase::kDone ||
      acknowledged_text_cursor != it->second.vlm_text.size()) {
    return false;
  }
  std::string prefix;
  record_realtime_history_locked(it->second, acknowledged_text_cursor, prefix);
  return it->second.realtime_history_prefix.size() == acknowledged_text_cursor;
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

std::vector<LegSnapshot> UnifiedEngine::legs_snapshot_locked(
    const Record& record) const {
  std::vector<LegSnapshot> out;
  if (record.has_audio) {
    out.push_back({UnifiedModality::kAudio, kAsrStageId, record.asr_state,
                   record.asr_error});
  }
  out.push_back({UnifiedModality::kText, kVlmStageId, record.vlm_state,
                 record.vlm_error});
  if (record.image_leg.enabled && !record.narrative_enabled) {
    out.push_back({record.image_leg.modality, record.image_leg.stage_id,
                   record.image_leg.state, record.image_leg.error});
  }
  if (record.tts_enabled) {
    LegState tts_state = LegState::kPending;
    if (record.tts_failed) {
      tts_state = LegState::kFailed;
    } else if (record.tts_legs.empty()) {
      tts_state = record.phase == Phase::kDone && record.cancel_requested
                      ? LegState::kAborted
                      : LegState::kPending;
    } else {
      bool any_running = false;
      bool all_aborted = true;
      for (const OutputLeg& leg : record.tts_legs) {
        if (leg.state == LegState::kRunning || leg.state == LegState::kPending) {
          any_running = true;
        }
        if (leg.state != LegState::kAborted) all_aborted = false;
      }
      tts_state = any_running ? LegState::kRunning
                : all_aborted ? LegState::kAborted
                              : LegState::kCompleted;
    }
    out.push_back({UnifiedModality::kAudio, kTtsStageId, tts_state,
                   record.tts_failed ? record.tts_error : std::string{}});
  }
  if (record.video_leg.enabled && !record.narrative_enabled) {
    out.push_back({record.video_leg.modality, record.video_leg.stage_id,
                   record.video_leg.state, record.video_leg.error});
  }
  for (const OutputLeg& leg : record.slot_legs) {
    out.push_back({leg.modality, leg.stage_id, leg.state, leg.error,
                   leg.slot_id});
  }
  return out;
}

void UnifiedEngine::trace_legs_locked(const std::string& request_id,
                                      const char* phase,
                                      const Record& record) const {
  if (!obs::Metrics::instance().trace_enabled()) return;
  const std::vector<LegSnapshot> legs = legs_snapshot_locked(record);
  std::vector<obs::TraceLegSnapshot> out;
  out.reserve(legs.size());
  for (const LegSnapshot& leg : legs) {
    out.push_back({to_string(leg.modality), leg.stage_id, to_string(leg.state),
                   leg.error});
  }
  obs::Metrics::instance().trace_legs_snapshot(request_id, phase, out);
}

std::vector<LegSnapshot> UnifiedEngine::legs_snapshot(
    const std::string& request_id) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = records_.find(request_id);
  if (it == records_.end()) return {};
  return legs_snapshot_locked(it->second);
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
    it->second.for_each_single_output_leg([&](const OutputLeg& leg) {
      if (leg.submitted) {
        child_ids.push_back(leg.child_id);
        child_to_unified_.erase(leg.child_id);
      }
    });
    for (const OutputLeg& leg : it->second.slot_legs) {
      if (leg.submitted) {
        child_ids.push_back(leg.child_id);
        child_to_unified_.erase(leg.child_id);
      }
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
    obs::Metrics::instance().queue_depth("unified_admission", admission_.size());
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
          const std::string raw(unified_detail::chunk_text(msg.chunk));
          std::string delta = raw;
          std::vector<NarrativeDirective> directives;
          if (record.narrative_enabled && record.narrative_parser) {
            record.vlm_stream_seen = true;
            delta = record.narrative_parser->feed(raw, directives);
          }
          record.vlm_text += delta;
          if (record.tts_stream && !record.tts_failed) {
            record.sentence_buf += delta;
            try {
              drain_sentences_locked(unified_id, record);
            } catch (const std::exception& error) {
              record.tts_failed = true;
              if (record.tts_error.empty()) record.tts_error = error.what();
              record.sentence_buf.clear();
              collect_tts_aborts_locked(record, "vlm_failure", deferred_aborts);
              UnifiedEvent leg_error;
              leg_error.request_id = unified_id;
              leg_error.session_id = record.session_id;
              leg_error.stage_id = kTtsStageId;
              leg_error.kind = UnifiedEvent::Kind::kLegError;
              leg_error.seq = record.next_seq++;
              leg_error.error = record.tts_error;
              dispatch_queue_.push_back(std::move(leg_error));
            }
          }
          if (!delta.empty() || !record.narrative_enabled) {
            UnifiedEvent event;
            event.request_id = unified_id;
            event.session_id = record.session_id;
            event.stage_id = kVlmStageId;
            event.kind = UnifiedEvent::Kind::kVlmDelta;
            event.seq = record.next_seq++;
            event.text = delta;
            dispatch_queue_.push_back(std::move(event));
          }
          // A directive closes after the prose that precedes it: dispatch each
          // slot immediately, still inside the VLM stream.
          for (const NarrativeDirective& directive : directives) {
            const std::string kind = slot_kind_locked(record, directive.kind);
            if (kind.empty()) continue;
            dispatch_narrative_slot_locked(unified_id, record, kind,
                                           directive.prompt,
                                           directive.prose_offset);
          }
        }
      } else if (msg.from_stage == config_.tts_stage && record.tts_enabled) {
        if (record.tts_stream) {
          emit_tts_chunk_locked(unified_id, record, msg.request_id, msg.chunk,
                                deferred_aborts);
        }
      } else if (msg.from_stage == record.image_leg.stage_name ||
                 msg.from_stage == record.video_leg.stage_name) {
        // Image and video legs deliver their artifact bytes in complete.result.
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
  if (record.narrative_enabled) {
    unified_detail::prepend_narrative_instructions(
        vlm_payload.request.inputs, record.narrative_visual_mode,
        record.narrative_max_slots);
  }
  vlm_payload.request.params = record.vlm_params;
  vlm_payload.request.metadata = record.vlm_metadata;
  record.phase = Phase::kVlm;
  record.vlm_state = LegState::kRunning;
  record.public_info.current_stage = kVlmStageId;
  record.vlm_started = std::chrono::steady_clock::now();
  coordinator_.submit(vlm_payload, config_.vlm_stage);
}

std::string UnifiedEngine::slot_kind_locked(
    const Record& record, const std::string& directive_kind) const {
  if (record.narrative_visual_mode == "image") {
    return record.image_leg.enabled ? std::string(kImageStageId)
                                    : std::string{};
  }
  if (record.narrative_visual_mode == "video") {
    return record.video_leg.enabled ? std::string(kVideoStageId)
                                    : std::string{};
  }
  if (directive_kind == "image" && record.image_leg.enabled) {
    return kImageStageId;
  }
  if (directive_kind == "video" && record.video_leg.enabled) {
    return kVideoStageId;
  }
  if (record.image_leg.enabled) return kImageStageId;
  if (record.video_leg.enabled) return kVideoStageId;
  return {};
}

void UnifiedEngine::dispatch_narrative_slot_locked(
    const std::string& request_id, Record& record, std::string kind,
    const std::string& prompt, size_t prose_offset) {
  // A cancel broadcast has already gone out; a slot created now would never
  // be aborted and would outlive the request.
  if (record.cancel_requested) return;
  if (record.narrative_next_slot >= record.narrative_max_slots) return;
  if (prompt.empty()) return;
  const bool image = kind == kImageStageId;
  if (!image && kind != kVideoStageId) return;
  const int index = record.narrative_next_slot;
  OutputLeg leg;
  leg.modality = image ? UnifiedModality::kImage : UnifiedModality::kVideo;
  leg.enabled = true;
  leg.stage_name = image ? config_.image_stage : config_.video_stage;
  leg.stage_id = image ? kImageStageId : kVideoStageId;
  leg.slot_id = "slot-" + std::to_string(index);
  leg.slot_index = index;
  leg.child_id = request_id + ":slot:" + std::to_string(index);
  Value::Map options = image ? record.image_options : record.video_options;
  options["prompt"] = Value(prompt);
  StagePayload payload;
  payload.request_id = leg.child_id;
  payload.request.inputs = Value(std::move(options));
  payload.request.params = record.vlm_params;
  payload.request.metadata = record.vlm_metadata;

  // The slot event goes out first, at the exact point the directive closed in
  // the prose stream, before the child is even submitted.
  UnifiedEvent slot_event;
  slot_event.request_id = request_id;
  slot_event.session_id = record.session_id;
  slot_event.stage_id = leg.stage_id;
  slot_event.kind = UnifiedEvent::Kind::kVisualSlot;
  slot_event.seq = record.next_seq++;
  slot_event.slot_id = leg.slot_id;
  slot_event.slot_index = index;
  slot_event.prompt = prompt;
  slot_event.text_offset = static_cast<int64_t>(prose_offset);
  dispatch_queue_.push_back(std::move(slot_event));

  ++record.narrative_next_slot;
  leg.submitted = true;
  leg.state = LegState::kRunning;
  leg.started = std::chrono::steady_clock::now();
  child_to_unified_[leg.child_id] = request_id;
  try {
    coordinator_.submit(payload, leg.stage_name);
  } catch (const std::exception& error) {
    child_to_unified_.erase(leg.child_id);
    leg.submitted = false;
    leg.state = LegState::kFailed;
    leg.error = error.what();
    const std::string slot_id = leg.slot_id;
    record.slot_legs.push_back(std::move(leg));
    UnifiedEvent leg_error;
    leg_error.request_id = request_id;
    leg_error.session_id = record.session_id;
    leg_error.stage_id = image ? kImageStageId : kVideoStageId;
    leg_error.kind = UnifiedEvent::Kind::kLegError;
    leg_error.seq = record.next_seq++;
    leg_error.error = error.what();
    leg_error.slot_id = slot_id;
    leg_error.slot_index = index;
    dispatch_queue_.push_back(std::move(leg_error));
    return;
  }
  record.slot_legs.push_back(std::move(leg));
  ++record.legs_remaining;
}

void UnifiedEngine::record_session_turn_locked(const Record& record) {
  const std::string& user =
      record.transcript.empty() ? record.user_text : record.transcript;
  session_store_.record_turn(record.session_id, user, record.vlm_text);
}

bool UnifiedEngine::record_realtime_history_locked(
    Record& record, size_t acknowledged_text_cursor,
    std::string& assistant_prefix) {
  if (record.realtime_history_recorded) {
    assistant_prefix = record.realtime_history_prefix;
    return false;
  }
  // Cursors originate at UTF-8-safe VLM/TTS boundaries. Reject a direct caller
  // that tries to split a codepoint rather than writing malformed history.
  if (acknowledged_text_cursor > record.vlm_text.size() ||
      (acknowledged_text_cursor < record.vlm_text.size() &&
       (static_cast<unsigned char>(
            record.vlm_text[acknowledged_text_cursor]) &
        0xc0) == 0x80)) {
    return false;
  }
  assistant_prefix = record.vlm_text.substr(0, acknowledged_text_cursor);
  record.realtime_history_prefix = assistant_prefix;
  if (!record.history_requested) return false;
  const std::string& user =
      record.transcript.empty() ? record.user_text : record.transcript;
  if (user.empty()) return false;
  session_store_.record_turn(record.session_id, user, assistant_prefix);
  record.realtime_history_recorded = true;
  record.realtime_history_prefix = assistant_prefix;
  return true;
}

UnifiedEvent UnifiedEngine::finish_locked(const std::string& request_id,
                                          Record& record,
                                          UnifiedEvent::Kind kind,
                                          std::string error, Value result) {
  record.phase = Phase::kDone;
  if (dispatching_) {
    record.terminal_dispatched = true;
    cv_.notify_all();
  }
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
  if (kind == UnifiedEvent::Kind::kCancelled) {
    if (record.has_audio && record.asr_state == LegState::kRunning) {
      record.asr_state = LegState::kAborted;
    }
    if (record.vlm_state == LegState::kPending ||
        record.vlm_state == LegState::kRunning) {
      record.vlm_state = LegState::kAborted;
    }
    if (record.image_leg.enabled && record.image_leg.state == LegState::kPending) {
      record.image_leg.state = LegState::kAborted;
    }
    if (record.video_leg.enabled && record.video_leg.state == LegState::kPending) {
      record.video_leg.state = LegState::kAborted;
    }
    for (OutputLeg& leg : record.slot_legs) {
      if (leg.state == LegState::kPending) leg.state = LegState::kAborted;
    }
  } else if (kind == UnifiedEvent::Kind::kError) {
    if (record.vlm_id.empty()) {
      if (record.has_audio && record.asr_state == LegState::kRunning) {
        record.asr_state = LegState::kFailed;
        record.asr_error = event.error;
      }
    } else if (record.vlm_state == LegState::kRunning) {
      record.vlm_state = LegState::kFailed;
      record.vlm_error = event.error;
    }
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
    trace.video_ms = record.video_ms;
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
      OutputLeg* output_leg = nullptr;
      if (record.narrative_enabled) {
        for (OutputLeg& leg : record.slot_legs) {
          if (leg.submitted && leg.child_id == info.request_id) {
            output_leg = &leg;
            break;
          }
        }
      } else {
        const bool image_child =
            record.image_leg.submitted &&
            info.request_id == record.image_leg.child_id;
        const bool video_child =
            record.video_leg.submitted &&
            info.request_id == record.video_leg.child_id;
        output_leg = image_child   ? &record.image_leg
                     : video_child ? &record.video_leg
                                   : nullptr;
      }
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
        collect_output_aborts_locked(record, "client", deferred_aborts);
        dispatch_queue_.push_back(finish_locked(
            unified_id, record, UnifiedEvent::Kind::kCancelled, {}, {}));
      } else if (info.state == "failed") {
        record_primary_leg("failed");
        collect_output_aborts_locked(record, "vlm_failure", deferred_aborts);
        dispatch_queue_.push_back(finish_locked(
            unified_id, record, UnifiedEvent::Kind::kError,
            info.error.empty() ? "stage failure" : info.error, {}));
      } else if (record.cancel_requested) {
        // The stage finished before it saw the abort. The request still ends
        // aborted, so the primary leg must be counted as aborted, not dropped.
        record_primary_leg("aborted");
        collect_output_aborts_locked(record, "client", deferred_aborts);
        dispatch_queue_.push_back(finish_locked(
            unified_id, record, UnifiedEvent::Kind::kCancelled, {}, {}));
      } else if (record.vlm_id.empty()) {
        record_primary_leg("completed");
        record.asr_state = LegState::kCompleted;
        if (!record.asr_final_seen) {
          const auto text = unified_detail::asr_result_text(info.result);
          if (!text) {
            record.asr_state = LegState::kFailed;
            record.asr_error = "ASR stage returned no text";
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
          record.vlm_state = LegState::kFailed;
          record.vlm_error = "VLM stage returned no text";
          collect_output_aborts_locked(record, "vlm_failure", deferred_aborts);
          dispatch_queue_.push_back(finish_locked(
              unified_id, record, UnifiedEvent::Kind::kError,
              "VLM stage returned no text", {}));
        } else {
          record.vlm_state = LegState::kCompleted;
          const double prefill = obs::result_ms(info.result, "prefill_ms");
          if (prefill >= 0) {
            obs::Metrics::instance().vlm_phase("prefill", prefill / 1000.0);
          }
          const double decode = obs::result_ms(info.result, "decode_ms");
          if (decode >= 0) {
            obs::Metrics::instance().vlm_phase("decode", decode / 1000.0);
          }
          if (record.narrative_enabled && record.narrative_parser) {
            std::vector<NarrativeDirective> late_directives;
            std::string narrative_tail;
            if (!record.vlm_stream_seen) {
              // The VLM never streamed: parse the terminal text as one chunk.
              narrative_tail =
                  record.narrative_parser->feed(*text, late_directives);
              narrative_tail +=
                  record.narrative_parser->flush(late_directives);
            } else {
              // The stream already consumed the text. EOF releases only the
              // suffix the parser withheld (a trailing '[' or an unclosed
              // candidate); a directive can no longer close at EOF.
              narrative_tail = record.narrative_parser->flush(late_directives);
            }
            if (!narrative_tail.empty()) {
              record.vlm_text += narrative_tail;
              if (record.tts_stream && !record.tts_failed) {
                record.sentence_buf += narrative_tail;
              }
              UnifiedEvent tail_event;
              tail_event.request_id = unified_id;
              tail_event.session_id = record.session_id;
              tail_event.stage_id = kVlmStageId;
              tail_event.kind = UnifiedEvent::Kind::kVlmDelta;
              tail_event.seq = record.next_seq++;
              tail_event.text = narrative_tail;
              dispatch_queue_.push_back(std::move(tail_event));
            }
            for (const NarrativeDirective& directive : late_directives) {
              const std::string kind =
                  slot_kind_locked(record, directive.kind);
              if (kind.empty()) continue;
              dispatch_narrative_slot_locked(unified_id, record, kind,
                                             directive.prompt,
                                             directive.prose_offset);
            }
            record.vlm_result = unified_detail::replace_result_text(
                info.result, record.vlm_text);
          } else {
            if (record.vlm_text.empty()) record.vlm_text = *text;
            record.vlm_result = info.result;
          }
          record.vlm_done = true;
          // Contract fallback: narrative enabled but the VLM produced no
          // marker. Render one visual from the clean prose unless the caller
          // opted into autonomous text-only routing (`allow_text_only: true`).
          if (record.narrative_enabled && record.slot_legs.empty() &&
              !record.vlm_text.empty() &&
              !(record.narrative_allow_text_only &&
                record.narrative_visual_mode == "auto")) {
            const std::string kind =
                record.narrative_visual_mode == "video"
                    ? kVideoStageId
                    : (record.image_leg.enabled ? kImageStageId : kVideoStageId);
            dispatch_narrative_slot_locked(unified_id, record, kind,
                                           record.vlm_text, 0);
          }
          if (record.image_leg.enabled || record.tts_enabled ||
              record.video_leg.enabled) {
            try {
              start_outputs_locked(unified_id, record);
            } catch (const std::exception& error) {
              collect_output_aborts_locked(record, "vlm_failure", deferred_aborts);
              dispatch_queue_.push_back(finish_locked(
                  unified_id, record, UnifiedEvent::Kind::kError,
                  error.what(), {}));
            }
          } else {
            dispatch_queue_.push_back(finish_locked(
                unified_id, record, UnifiedEvent::Kind::kComplete, {},
                info.result));
            if (record.history_requested && !record.realtime_playback_ack) {
              record_session_turn_locked(record);
            }
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

void UnifiedEngine::submit_single_output_leg_locked(
    const std::string& request_id, Record& record, OutputLeg& leg,
    const char* suffix, Value inputs) {
  leg.child_id = request_id + suffix;
  child_to_unified_[leg.child_id] = request_id;
  StagePayload payload;
  payload.request_id = leg.child_id;
  payload.request.inputs = std::move(inputs);
  payload.request.params = record.vlm_params;
  payload.request.metadata = record.vlm_metadata;
  leg.submitted = true;
  leg.state = LegState::kRunning;
  leg.started = std::chrono::steady_clock::now();
  try {
    coordinator_.submit(payload, leg.stage_name);
  } catch (...) {
    leg.submitted = false;
    leg.state = LegState::kFailed;
    child_to_unified_.erase(leg.child_id);
    throw;
  }
  ++record.legs_remaining;
}

void UnifiedEngine::start_tts_locked(const std::string& request_id,
                                     Record& record, std::string text,
                                     bool first, size_t text_begin,
                                     size_t text_end) {
  OutputLeg leg;
  leg.modality = UnifiedModality::kAudio;
  leg.enabled = true;
  leg.stage_name = config_.tts_stage;
  leg.stage_id = kTtsStageId;
  leg.text_begin = text_begin;
  leg.text_end = text_end;
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
  leg.state = LegState::kRunning;
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
    const size_t text_begin = record.tts_text_cursor;
    record.tts_text_cursor += cut;
    const size_t begin = sentence.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) continue;
    sentence = sentence.substr(
        begin, sentence.find_last_not_of(" \t\r\n") - begin + 1);
    start_tts_locked(request_id, record, std::move(sentence), false, text_begin,
                     record.tts_text_cursor);
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
  size_t text_begin = record.tts_text_cursor;
  record.tts_text_cursor += tail.size();
  const size_t begin = tail.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    if (!record.tts_legs.empty()) return;
    tail = record.vlm_text;
    text_begin = 0;
    record.tts_text_cursor = tail.size();
  } else {
    tail = tail.substr(begin, tail.find_last_not_of(" \t\r\n") - begin + 1);
  }
  if (tail.empty()) return;
  start_tts_locked(request_id, record, std::move(tail), record.tts_legs.empty(),
                   text_begin, record.tts_text_cursor);
}

void UnifiedEngine::start_outputs_locked(const std::string& request_id,
                                         Record& record) {
  record.phase = Phase::kOutputs;
  record.public_info.current_stage =
      record.image_leg.enabled   ? kImageStageId
      : record.video_leg.enabled ? kVideoStageId
                                 : kTtsStageId;
  // In narrative mode visuals come from parsed slots, not from the single
  // image/video legs; those modalities only enable the slot stages.
  if (!record.narrative_enabled) {
    if (record.image_leg.enabled) {
      submit_single_output_leg_locked(
          request_id, record, record.image_leg, kImageSuffix,
          unified_detail::build_image_inputs(record.image_options,
                                             record.vlm_text));
    }
    if (record.video_leg.enabled) {
      submit_single_output_leg_locked(
          request_id, record, record.video_leg, kVideoSuffix,
          unified_detail::build_video_inputs(record.video_options,
                                             record.vlm_text));
    }
  }
  if (record.tts_enabled) {
    if (record.tts_stream) {
      flush_sentences_locked(request_id, record);
    } else {
      start_tts_locked(request_id, record, record.vlm_text, true, 0,
                       record.vlm_text.size());
    }
  }
  trace_legs_locked(request_id, "outputs_started", record);
  if (record.legs_remaining == 0) on_legs_drained_locked(request_id, record);
}

void UnifiedEngine::emit_tts_chunk_locked(
    const std::string& request_id, Record& record,
    const std::string& child_id, const Value& chunk,
    std::vector<std::pair<std::string, const char*>>& deferred_aborts) {
  if (record.tts_failed) return;
  const OutputLeg* source_leg = nullptr;
  for (const OutputLeg& leg : record.tts_legs) {
    if (leg.submitted && leg.child_id == child_id) {
      source_leg = &leg;
      break;
    }
  }
  // A chunk for an unknown or already replaced child must never become audio
  // for the current response, nor lend it an acknowledgement cursor.
  if (source_leg == nullptr) return;
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
  int64_t sample_rate = 24000;
  if (const Value* rate = chunk.find("sample_rate");
      rate != nullptr && rate->is_int() && rate->as_int() > 0) {
    sample_rate = rate->as_int();
  }
  int64_t channels = 1;
  if (const Value* ch = chunk.find("channels");
      ch != nullptr && ch->is_int() && ch->as_int() > 0) {
    channels = ch->as_int();
  }
  descriptor["sample_rate"] = Value(sample_rate);
  descriptor["channels"] = Value(channels);
  if (record.realtime_playback_ack) {
    descriptor["text_cursor"] =
        Value(static_cast<int64_t>(source_leg->text_begin));
    descriptor["text_end_cursor"] =
        Value(static_cast<int64_t>(source_leg->text_end));
  }
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
    if (leg.submitted && leg.state == LegState::kRunning) {
      out.emplace_back(leg.child_id, reason);
    }
  }
}

void UnifiedEngine::collect_output_aborts_locked(
    const Record& record, const char* reason,
    std::vector<std::pair<std::string, const char*>>& out) {
  record.for_each_single_output_leg([&](const OutputLeg& leg) {
    if (leg.submitted && leg.state == LegState::kRunning) {
      out.emplace_back(leg.child_id, reason);
    }
  });
  for (const OutputLeg& leg : record.slot_legs) {
    if (leg.submitted && leg.state == LegState::kRunning) {
      out.emplace_back(leg.child_id, reason);
    }
  }
  collect_tts_aborts_locked(record, reason, out);
}

void UnifiedEngine::emit_artifact_locked(const std::string& request_id,
                                         Record& record,
                                         const OutputLeg& leg,
                                         const Value& result) {
  std::vector<UnifiedEvent> events = unified_detail::slice_artifact_events(
      request_id, record.session_id, leg.stage_id, result,
      record.artifact_bytes_total, record.next_artifact, record.next_seq);
  for (UnifiedEvent& event : events) {
    if (record.realtime_playback_ack && leg.stage_id == kTtsStageId &&
        event.artifact.is_map()) {
      Value::Map descriptor = event.artifact.as_map();
      descriptor["text_cursor"] = Value(static_cast<int64_t>(leg.text_begin));
      descriptor["text_end_cursor"] = Value(static_cast<int64_t>(leg.text_end));
      event.artifact = Value(std::move(descriptor));
    }
    if (leg.slot_index >= 0 && event.artifact.is_map()) {
      Value::Map descriptor = event.artifact.as_map();
      descriptor["slot_id"] = Value(leg.slot_id);
      descriptor["slot_index"] = Value(static_cast<int64_t>(leg.slot_index));
      event.artifact = Value(std::move(descriptor));
    }
    dispatch_queue_.push_back(std::move(event));
  }
}

void UnifiedEngine::settle_output_leg_locked(
    const std::string& request_id, Record& record, OutputLeg& leg,
    UnifiedEvent::Kind kind, std::string error) {
  const bool aborted_for_limit =
      kind == UnifiedEvent::Kind::kCancelled && !record.cancel_requested &&
      record.tts_failed;
  leg.state = kind == UnifiedEvent::Kind::kComplete ? LegState::kCompleted
            : (kind == UnifiedEvent::Kind::kCancelled && !aborted_for_limit)
                ? LegState::kAborted
                : LegState::kFailed;
  if (aborted_for_limit && !record.tts_error.empty()) {
    error = record.tts_error;
  }
  leg.error = error;
  {
    const double ms = ms_since(leg.started);
    const char* leg_name = leg.stage_id.c_str();
    if (leg.stage_id == kTtsStageId) {
      record.tts_ms = ms;
    } else if (leg.stage_id == kVideoStageId) {
      record.video_ms = ms;
    } else {
      record.image_ms = ms;
    }
    obs::Metrics::instance().leg(leg_name, record.request_type,
                                 to_string(leg.state), ms / 1000.0);
    obs::Metrics::instance().trace_leg(request_id, leg_name,
                                       to_string(leg.state), ms);
  }
  trace_legs_locked(request_id, "leg_settled", record);
  if (kind != UnifiedEvent::Kind::kComplete) {
    if (leg.stage_id == kTtsStageId && leg.state == LegState::kFailed) {
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
    event.slot_id = leg.slot_id;
    event.slot_index = leg.slot_index;
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
    item["state"] = Value(to_string(leg.state));
    if (!leg.error.empty()) item["error"] = Value(leg.error);
    if (leg.slot_index >= 0) {
      item["slot_id"] = Value(leg.slot_id);
      item["slot_index"] = Value(static_cast<int64_t>(leg.slot_index));
    }
    legs.push_back(Value(std::move(item)));
  };
  if (!record.narrative_enabled) add(record.image_leg);
  if (record.tts_enabled) {
    Value::Map item;
    item["stage"] = Value(kTtsStageId);
    bool all_aborted = !record.tts_legs.empty();
    for (const OutputLeg& leg : record.tts_legs) {
      if (leg.state != LegState::kAborted) all_aborted = false;
    }
    item["state"] = Value(record.tts_failed ? "failed"
                        : all_aborted       ? "aborted"
                                            : "completed");
    if (record.tts_failed && !record.tts_error.empty()) {
      item["error"] = Value(record.tts_error);
    }
    legs.push_back(Value(std::move(item)));
  }
  if (!record.narrative_enabled) add(record.video_leg);
  for (const OutputLeg& leg : record.slot_legs) add(leg);
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
  if (kind == UnifiedEvent::Kind::kComplete && record.history_requested &&
      !record.realtime_playback_ack) {
    record_session_turn_locked(record);
  }
}

}  // namespace umminfer
