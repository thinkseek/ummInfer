// SPDX-License-Identifier: Apache-2.0
#include "umminfer/pipeline/unified.hpp"

#include <algorithm>
#include <exception>
#include <string_view>
#include <vector>

namespace umminfer {
namespace {

// Content-part "type" values the client contract uses (client.hpp ContentPart).
UnifiedModality modality_of_part_type(const std::string& type) {
  if (type == "input_audio") return UnifiedModality::kAudio;
  if (type == "image_url") return UnifiedModality::kImage;
  if (type == "video_url" || type == "video") return UnifiedModality::kVideo;
  return UnifiedModality::kText;
}

std::string_view chunk_text(const Value& chunk) {
  if (!chunk.is_map()) return {};
  const Value* text = chunk.find("text");
  if (text != nullptr && text->is_str()) return text->as_str();
  return {};
}

// The current turn's real user text, extracted with the same shape handling as
// build_vlm_inputs: a plain string content is one text unit, and text parts in
// a content array keep their wire order. Used for session history on VLM-only
// turns, where there is no ASR transcript to stand in for the user input.
std::string user_text_from_inputs(const Value& original_inputs) {
  std::string joined;
  if (!original_inputs.is_map()) return joined;
  const Value* messages = original_inputs.find("messages");
  if (messages == nullptr || !messages->is_array()) return joined;
  for (const Value& message : messages->as_array()) {
    if (!message.is_map()) continue;
    const Value* content = message.find("content");
    if (content == nullptr) continue;
    if (content->is_str()) {
      if (!content->as_str().empty()) {
        if (!joined.empty()) joined += " ";
        joined += content->as_str();
      }
      continue;
    }
    if (!content->is_array()) continue;
    for (const Value& part : content->as_array()) {
      const Value* type = part.find("type");
      if (type == nullptr || !type->is_str() ||
          modality_of_part_type(type->as_str()) != UnifiedModality::kText) {
        continue;
      }
      const Value* text = part.find("text");
      if (text != nullptr && text->is_str() && !text->as_str().empty()) {
        if (!joined.empty()) joined += " ";
        joined += text->as_str();
      }
    }
  }
  return joined;
}

bool chunk_is_final(const Value& chunk) {
  if (!chunk.is_map()) return false;
  const Value* flag = chunk.find("is_final");
  return flag != nullptr && flag->is_bool() && flag->as_bool();
}

// Extracts ASR result text under the same contract as
// serve_detail::asr_result_text (src/serve/asr_common.hpp): a plain string or a
// map with a string "text". Kept local so the pipeline layer does not include a
// serve-private header. Nullopt means the stage returned no usable text; the
// caller must fail the request, not start the VLM leg with an empty prompt.
std::optional<std::string> asr_result_text(const Value& result) {
  if (result.is_str()) return result.as_str();
  if (result.is_map()) {
    const Value* text = result.find("text");
    if (text != nullptr && text->is_str()) return text->as_str();
  }
  return std::nullopt;
}

// Matches serve_detail::result_text's three valid shapes. Unlike the HTTP
// compatibility fallback, a malformed stage result is an orchestration error.
std::optional<std::string> vlm_result_text(const Value& result) {
  if (auto text = asr_result_text(result)) return text;
  if (result.is_map()) {
    const Value* content = result.find("content");
    if (content != nullptr && content->is_str()) return content->as_str();
  }
  return std::nullopt;
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
        // Copied sinks outlive the engine. Queue their work in the gate so
        // retirement and reentry cannot bypass the active-engine callbacks.
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
        // Drain later callbacks before reporting the first error. Otherwise a
        // reentrant completion would remain stranded behind a throwing stream.
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
    : coordinator_(coordinator), config_(std::move(config)) {
  if (config_.asr_stage == config_.vlm_stage) {
    throw ProtocolError("unified engine requires distinct asr and vlm stages");
  }
  // Chain sinks the embedded server installed instead of replacing them: the
  // HTTP SSE path and this engine share one Coordinator. The Coordinator rejects
  // this exchange if a pump has already started.
  callback_gate_ = std::make_shared<CallbackGate>(this);
  auto previous = coordinator_.exchange_sinks_before_run(
      [gate = callback_gate_](const StreamMessage& msg) { gate->stream(msg); },
      [gate = callback_gate_](const RequestInfo& info) { gate->complete(info); });
  callback_gate_->previous_stream = std::move(previous.first);
  callback_gate_->previous_complete = std::move(previous.second);
}

UnifiedEngine::~UnifiedEngine() noexcept {
  // ponytail: leave a retired forwarding shell in the Coordinator instead of
  // racing its unsynchronized std::function reads. Repeated engine recreation
  // chains shells; add synchronized sink exchange if that becomes a workload.
  callback_gate_->retire();
}

std::string UnifiedEngine::submit(StagePayload payload) {
  const Value& inputs = payload.request.inputs;
  if (!inputs.is_map() || inputs.find("messages") == nullptr ||
      !inputs.find("messages")->is_array()) {
    throw ProtocolError("unified request inputs must contain a messages array");
  }

  // Image reaches the VLM leg unchanged. Video still has no worker: reject an
  // explicit video modality claim or a matching content part before any stage
  // sees the work. At most one image part is accepted so the VLM prompt is
  // unambiguous.
  if (const Value* declared = inputs.find("modalities"); declared != nullptr) {
    if (!declared->is_array()) {
      throw ProtocolError("unified request modalities must be an array");
    }
    for (const Value& token : declared->as_array()) {
      if (!token.is_str()) {
        throw ProtocolError("unified request modality must be a string");
      }
      const auto modality = modality_from_string(token.as_str());
      if (!modality) throw ProtocolError("unknown modality: " + token.as_str());
      if (*modality == UnifiedModality::kVideo) {
        throw ProtocolError("modality " + token.as_str() +
                            " is declared but has no worker in unified engine v1");
      }
    }
  }
  size_t image_count = 0;
  size_t audio_part_count = 0;
  for (const Value& message : inputs.find("messages")->as_array()) {
    if (!message.is_map()) continue;
    const Value* content = message.find("content");
    if (content == nullptr) continue;
    // A plain string content is the OpenAI text shape and carries no parts.
    if (content->is_str()) continue;
    if (!content->is_array()) {
      throw ProtocolError(
          "message content must be a text string or a content part array");
    }
    for (const Value& part : content->as_array()) {
      if (!part.is_map()) continue;
      const Value* type = part.find("type");
      if (type == nullptr || !type->is_str()) continue;
      const UnifiedModality modality = modality_of_part_type(type->as_str());
      if (modality == UnifiedModality::kVideo) {
        throw ProtocolError(std::string("content part ") + type->as_str() +
                            " is declared but has no worker in unified engine v1");
      }
      if (modality == UnifiedModality::kAudio) {
        ++audio_part_count;
      }
      if (modality == UnifiedModality::kImage) {
        const Value* image_url = part.find("image_url");
        if (image_url == nullptr || !image_url->is_map() ||
            image_url->find("url") == nullptr ||
            !image_url->find("url")->is_str() ||
            image_url->find("url")->as_str().empty()) {
          throw ProtocolError("image_url part requires a nonempty image_url.url");
        }
        if (++image_count > 1) {
          throw ProtocolError("unified request accepts at most one image_url part");
        }
      }
    }
  }

  Value original_inputs = inputs;
  auto vlm_params = payload.request.params;
  auto vlm_metadata = payload.request.metadata;
  // ASR partials are opt-in per request: inputs.asr_partials is set only when
  // the caller asked for increment output. Without it the ASR leg keeps the
  // aggregated request and the event stream stays asr_final -> vlm_delta.
  bool asr_partials = false;
  if (const Value* requested = original_inputs.find("asr_partials");
      requested != nullptr) {
    if (!requested->is_bool()) {
      throw ProtocolError("asr_partials must be a boolean");
    }
    asr_partials = requested->as_bool();
  }
  bool history_requested = false;
  if (const Value* requested = original_inputs.find("session_history");
      requested != nullptr) {
    if (!requested->is_bool()) {
      throw ProtocolError("session_history must be a boolean");
    }
    history_requested = requested->as_bool();
    if (history_requested && !config_.session_history.enabled) {
      throw ProtocolError(
          "session_history requires engine session history to be enabled");
    }
  }
  // Optional generated outputs. output_modalities selects the legs;
  // image_options/audio_options are engine-only keys, stripped from the VLM
  // prompt in build_vlm_inputs. The HTTP boundary validates the public shape;
  // these checks are the engine-level backstop for direct submit() callers.
  bool image_output = false;
  bool audio_output = false;
  if (const Value* requested = original_inputs.find("output_modalities");
      requested != nullptr) {
    if (!requested->is_array()) {
      throw ProtocolError("output_modalities must be an array");
    }
    for (const Value& token : requested->as_array()) {
      if (!token.is_str()) {
        throw ProtocolError("output_modalities values must be strings");
      }
      const std::string& name = token.as_str();
      // Text is the implicit primary output; only the two optional legs are
      // tracked here.
      if (name == "text") continue;
      if (name == "image") image_output = true;
      else if (name == "audio") audio_output = true;
      else throw ProtocolError("unknown output modality: " + name);
    }
  }
  Value::Map image_options;
  Value::Map audio_options;
  if (const Value* options = original_inputs.find("image_options");
      options != nullptr) {
    if (!options->is_map()) throw ProtocolError("image_options must be a map");
    image_options = options->as_map();
  }
  if (const Value* options = original_inputs.find("audio_options");
      options != nullptr) {
    if (!options->is_map()) throw ProtocolError("audio_options must be a map");
    // TTS input is always the aggregated VLM answer; a client text here would
    // bypass that rule and is rejected rather than ignored.
    if (options->find("text") != nullptr) {
      throw ProtocolError("audio_options must not carry text");
    }
    audio_options = options->as_map();
  }
  if (image_output && config_.image_stage.empty()) {
    throw ProtocolError("image output is not enabled on this server");
  }
  if (audio_output && config_.tts_stage.empty()) {
    throw ProtocolError("audio output is not enabled on this server");
  }
  if (audio_output) {
    auto voice = audio_options.find("voice");
    if (voice == audio_options.end() || !voice->second.is_str() ||
        voice->second.as_str().empty()) {
      throw ProtocolError("audio output requires a non-empty voice");
    }
  }
  // The route signal is the public content shape: an input_audio part selects
  // the ASR -> VLM path, a text- or image-only request runs the VLM leg
  // directly. The binary PCM is attached only by the real HTTP boundary;
  // harness requests may omit it while still exercising the ASR leg.
  const bool has_audio = audio_part_count > 0;
  const Value* audio = inputs.find("audio");
  if (has_audio) {
    if (audio != nullptr &&
        (!audio->is_bin() || audio->as_bin().empty() ||
         audio->as_bin().size() % 2 != 0)) {
      throw ProtocolError(
          "unified audio must be nonempty PCM16 with complete samples");
    }
    Value::Map asr_inputs;
    if (audio != nullptr) asr_inputs["audio"] = *audio;
    payload.request.inputs = Value(std::move(asr_inputs));
    // The ASR backend streams only when the caller opted in. A partial event
    // is a delta; the final transcript still comes from the completion and
    // replaces the concatenated partials.
    payload.request.params["stream"] = Value(asr_partials);
    payload.request.params["temperature"] = Value(0.0);
  }

  // Read from original_inputs: the audio branch above replaced inputs with the
  // PCM-only ASR map, so inputs no longer holds the session label.
  std::string session_id;
  bool explicit_session = false;
  if (const Value* session = original_inputs.find("session_id");
      session != nullptr && session->is_str() && !session->as_str().empty()) {
    session_id = session->as_str();
    explicit_session = true;
  } else {
    session_id = "session-" + std::to_string(session_counter_++);
  }
  // A unique per-request label can never match another request, so accepting
  // the opt-in there would silently do nothing.
  if (history_requested && !explicit_session) {
    throw ProtocolError("session_history requires an explicit session_id");
  }

  std::lock_guard<std::mutex> lock(mu_);
  if (admission_.size() >= config_.max_in_flight) {
    throw QueueFullError();
  }
  Record record;
  record.session_id = std::move(session_id);
  record.history_requested = history_requested;
  record.original_inputs = std::move(original_inputs);
  record.vlm_params = std::move(vlm_params);
  record.vlm_metadata = std::move(vlm_metadata);
  record.user_text = user_text_from_inputs(record.original_inputs);
  record.image_options = std::move(image_options);
  record.audio_options = std::move(audio_options);
  record.image_leg.enabled = image_output;
  record.image_leg.stage_name = config_.image_stage;
  record.image_leg.stage_id = kImageStageId;
  record.tts_leg.enabled = audio_output;
  record.tts_leg.stage_name = config_.tts_stage;
  record.tts_leg.stage_id = kTtsStageId;

  std::string request_id;
  if (has_audio) {
    request_id = coordinator_.submit(payload, config_.asr_stage);
  } else {
    // VLM-only turn: build the same prompt shape the ASR -> VLM handoff would
    // produce, with an empty transcript, and submit straight to the VLM stage.
    // Keep the entry request id: the HTTP SSE sink and every public event are
    // keyed by it. A generated id would orphan the sink and hang the stream.
    StagePayload vlm_payload;
    vlm_payload.request_id = payload.request_id;
    vlm_payload.request.inputs =
        build_vlm_inputs("", record.original_inputs);
    if (record.history_requested) {
      vlm_payload.request.inputs["messages"] = Value(
          with_session_history_locked(
              record, vlm_payload.request.inputs["messages"].as_array()));
    }
    vlm_payload.request.params = record.vlm_params;
    vlm_payload.request.metadata = record.vlm_metadata;
    record.phase = Phase::kVlm;
    request_id = coordinator_.submit(vlm_payload, config_.vlm_stage);
    // The only leg is the VLM leg, keyed by the Coordinator id itself.
    record.vlm_id = request_id;
  }
  admission_.push_back(request_id);
  record.public_info.request_id = request_id;
  record.public_info.state = "running";
  record.public_info.current_stage =
      has_audio ? kAsrStageId : kVlmStageId;
  records_[request_id] = std::move(record);
  return request_id;
}

bool UnifiedEngine::cancel(const std::string& request_id) {
  std::vector<std::string> active_ids;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = records_.find(request_id);
    if (it == records_.end() || it->second.phase == Phase::kDone) return false;
    // Record the intent under mu_ first. Every fan-out decision is made under
    // this lock, so a leg not submitted yet is blocked by the flag instead of
    // an abort broadcast for an unknown id.
    Record& record = it->second;
    record.cancel_requested = true;
    if (!record.vlm_id.empty()) active_ids.push_back(record.vlm_id);
    if (record.image_leg.submitted) active_ids.push_back(record.image_leg.child_id);
    if (record.tts_leg.submitted) active_ids.push_back(record.tts_leg.child_id);
    if (active_ids.empty()) {
      // ASR/VLM-only entry leg, still keyed by the unified id.
      active_ids.push_back(request_id);
    }
  }
  // No engine lock held: Coordinator::abort can synchronously invoke
  // on_complete, which needs mu_. abort() on an unknown or already terminal
  // id is a silent no-op, so only submitted non-empty ids are listed.
  for (const std::string& id : active_ids) coordinator_.abort(id);
  return true;
}

bool UnifiedEngine::wait(const std::string& request_id, double timeout_s) {
  std::unique_lock<std::mutex> lock(mu_);
  return cv_.wait_for(lock, std::chrono::duration<double>(timeout_s), [&] {
    auto it = records_.find(request_id);
    return it != records_.end() && it->second.phase == Phase::kDone;
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
    child_ids.push_back(it->second.vlm_id);
    if (it->second.image_leg.submitted) {
      child_ids.push_back(it->second.image_leg.child_id);
    }
    if (it->second.tts_leg.submitted) {
      child_ids.push_back(it->second.tts_leg.child_id);
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
  // Every coordinator leg record belongs to this engine request. Releasing an
  // already-released id is a harmless no-op, which covers a late completion
  // racing this cleanup.
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

// Records are keyed by the ASR leg id; VLM events carry the derived child id.
// ponytail: linear fallback scan, max_in_flight bounds the record count.
std::map<std::string, UnifiedEngine::Record>::iterator
UnifiedEngine::find_record_locked(const std::string& event_request_id) {
  auto it = records_.find(event_request_id);
  if (it != records_.end()) return it;
  for (auto candidate = records_.begin(); candidate != records_.end();
       ++candidate) {
    const Record& record = candidate->second;
    if ((!record.vlm_id.empty() && record.vlm_id == event_request_id) ||
        (record.image_leg.submitted &&
         record.image_leg.child_id == event_request_id) ||
        (record.tts_leg.submitted &&
         record.tts_leg.child_id == event_request_id)) {
      return candidate;
    }
  }
  return records_.end();
}

void UnifiedEngine::handle_stream(const StreamMessage& msg) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto record_it = find_record_locked(msg.request_id);
    if (record_it != records_.end() && record_it->second.phase != Phase::kDone) {
      Record& record = record_it->second;
      const std::string& unified_id = record_it->first;
      if (msg.from_stage == config_.vlm_stage) {
        if (record.phase == Phase::kVlm) {
          record.vlm_text += chunk_text(msg.chunk);
          UnifiedEvent event;
          event.request_id = unified_id;  // contract id, never the child id
          event.session_id = record.session_id;
          event.stage_id = kVlmStageId;
          event.kind = UnifiedEvent::Kind::kVlmDelta;
          event.seq = record.next_seq++;
          event.text = chunk_text(msg.chunk);
          dispatch_queue_.push_back(std::move(event));
        }
        // A VLM chunk before the VLM leg starts is out of order: drop it.
      } else if (msg.from_stage == record.image_leg.stage_name ||
                 msg.from_stage == record.tts_leg.stage_name) {
        // Output legs deliver their Bin in complete.result; they emit no text
        // stream. A stray chunk must not fall into the ASR branch below and
        // appear as a fake transcript. It is still forwarded to any previous
        // sink (the realtime chain) after this block.
      } else if (record.phase != Phase::kVlm) {
        const bool is_final = chunk_is_final(msg.chunk);
        bool emit = true;
        if (is_final) {
          // A repeated final is idempotent: the first owns the transcript.
          if (record.asr_final_seen) {
            emit = false;
          } else {
            const auto text = asr_result_text(msg.chunk);
            if (!text) {
              emit = false;
              dispatch_queue_.push_back(finish_locked(
                  unified_id, record, UnifiedEvent::Kind::kError,
                  "ASR stage returned no text", {}));
            } else {
              record.asr_final_seen = true;
              record.transcript = *text;  // final replaces partials
            }
          }
        } else {
          record.transcript += chunk_text(msg.chunk);  // partials are deltas
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
          event.text = chunk_text(msg.chunk);
          dispatch_queue_.push_back(std::move(event));
        }
      }
      // An ASR chunk after the VLM leg started is late and out of order.
    }
    // Forwarding is queued behind this chunk's unified event. Unknown or
    // terminal records still forward without creating a unified event.
    dispatch_queue_.push_back(msg);
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
    if (const auto* event = std::get_if<UnifiedEvent>(&item)) {
      const bool terminal = event->kind == UnifiedEvent::Kind::kComplete ||
                            event->kind == UnifiedEvent::Kind::kError ||
                            event->kind == UnifiedEvent::Kind::kCancelled;
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
      if (terminal) cv_.notify_all();
    } else if (const auto* msg = std::get_if<StreamMessage>(&item)) {
      // Never call a foreign sink under mu_: it may reenter info() or cancel().
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
  }
  // The queue is empty and mu_ is held, so no producer can race this reset.
  dispatching_ = false;
  if (first_error) std::rethrow_exception(first_error);
}

Value UnifiedEngine::build_vlm_inputs(const std::string& transcript,
                                     const Value& original_inputs) {
  Value::Map out;
  if (original_inputs.is_map()) {
    for (const auto& [key, value] : original_inputs.as_map()) {
      if (key != "messages" && key != "modalities" && key != "session_id" &&
          key != "audio" && key != "asr_partials" && key != "session_history" &&
          key != "output_modalities" && key != "image_options" &&
          key != "audio_options") {
        out[key] = value;
      }
    }
  }
  // The VLM sees the original image and text parts next to the ASR transcript.
  // A text part is an optional client instruction; the transcript is appended
  // last so the spoken request stays the final turn. A plain string content is
  // the OpenAI text shape and becomes one text part. With no instruction the
  // content stays the transcript string, which older workers already accept.
  Value::Array extra_parts;
  bool has_image = false;
  if (original_inputs.is_map()) {
    const Value* messages = original_inputs.find("messages");
    if (messages != nullptr && messages->is_array()) {
      for (const Value& message : messages->as_array()) {
        if (!message.is_map()) continue;
        const Value* content = message.find("content");
        if (content == nullptr) continue;
        if (content->is_str()) {
          if (!content->as_str().empty()) {
            Value::Map text_part;
            text_part["type"] = Value("text");
            text_part["text"] = Value(content->as_str());
            extra_parts.push_back(Value(std::move(text_part)));
          }
          continue;
        }
        if (!content->is_array()) continue;
        for (const Value& part : content->as_array()) {
          const Value* type = part.find("type");
          if (type == nullptr || !type->is_str()) continue;
          const UnifiedModality modality = modality_of_part_type(type->as_str());
          if (modality == UnifiedModality::kImage) {
            has_image = true;
            extra_parts.push_back(part);
          } else if (modality == UnifiedModality::kText) {
            extra_parts.push_back(part);
          }
        }
      }
    }
  }
  Value::Map user_message;
  user_message["role"] = Value("user");
  if (!extra_parts.empty()) {
    // A VLM-only turn has no transcript: do not append an empty text part.
    if (!transcript.empty()) {
      Value::Map text_part;
      text_part["type"] = Value("text");
      text_part["text"] = Value(transcript);
      extra_parts.push_back(Value(std::move(text_part)));
    }
    user_message["content"] = Value(std::move(extra_parts));
  } else {
    user_message["content"] = Value(transcript);
  }
  Value::Array messages;
  messages.push_back(Value(std::move(user_message)));
  out["messages"] = Value(std::move(messages));
  Value::Array modalities;
  modalities.push_back(Value("text"));
  if (has_image) modalities.push_back(Value("image"));
  out["modalities"] = Value(std::move(modalities));
  return Value(std::move(out));
}

void UnifiedEngine::start_vlm_locked(const std::string& request_id,
                                     Record& record) {
  StagePayload vlm_payload;
  // The Coordinator rejects a duplicate request_id, so the VLM leg runs under a
  // derived child id. The suffix shape is not "req-N" and therefore never
  // advances the Coordinator's id counter.
  record.vlm_id = request_id + ":vlm";
  vlm_payload.request_id = record.vlm_id;
  vlm_payload.request.inputs =
      build_vlm_inputs(record.transcript, record.original_inputs);
  if (record.history_requested) {
    vlm_payload.request.inputs["messages"] = Value(
        with_session_history_locked(
            record, vlm_payload.request.inputs["messages"].as_array()));
  }
  vlm_payload.request.params = record.vlm_params;
  vlm_payload.request.metadata = record.vlm_metadata;
  record.phase = Phase::kVlm;
  record.public_info.current_stage = kVlmStageId;
  coordinator_.submit(vlm_payload, config_.vlm_stage);
}

Value::Array UnifiedEngine::with_session_history_locked(
    const Record& record, const Value::Array& current_messages) {
  Value::Array messages;
  const auto it = session_history_.find(record.session_id);
  if (it != session_history_.end()) {
    for (const SessionTurn& turn : it->second) {
      Value::Map user;
      user["role"] = Value("user");
      user["content"] = Value(turn.user);
      Value::Map assistant;
      assistant["role"] = Value("assistant");
      assistant["content"] = Value(turn.assistant);
      messages.push_back(Value(std::move(user)));
      messages.push_back(Value(std::move(assistant)));
    }
  }
  for (const Value& message : current_messages) messages.push_back(message);
  return messages;
}

void UnifiedEngine::record_session_turn_locked(const Record& record) {
  const bool is_new_session =
      session_history_.find(record.session_id) == session_history_.end();
  std::deque<SessionTurn>& turns = session_history_[record.session_id];
  // An audio turn's user side is the ASR transcript. A VLM-only turn has no
  // transcript, so use the real string/text-part input instead of an empty
  // user message.
  const std::string& user = record.transcript.empty() ? record.user_text
                                                       : record.transcript;
  turns.push_back(SessionTurn{user, record.vlm_text});
  size_t chars = 0;
  for (const SessionTurn& turn : turns) {
    chars += turn.user.size() + turn.assistant.size();
  }
  // Oldest turns leave first until both bounds hold. A single turn larger than
  // the character budget is itself evicted: an unbounded turn must never stay.
  while ((turns.size() > config_.session_history.max_turns ||
          chars > config_.session_history.max_chars) &&
         !turns.empty()) {
    chars -= turns.front().user.size() + turns.front().assistant.size();
    turns.pop_front();
  }
  if (turns.empty()) {
    session_history_.erase(record.session_id);
    if (!is_new_session) {
      // ponytail: linear deque scan; max_sessions (256) bounds the cost.
      for (auto it = session_order_.begin(); it != session_order_.end(); ++it) {
        if (*it == record.session_id) {
          session_order_.erase(it);
          break;
        }
      }
    }
    return;
  }
  // Track most-recent-turn order: an existing session moves to the back.
  if (!is_new_session) {
    // ponytail: linear deque scan; max_sessions (256) bounds the cost.
    for (auto it = session_order_.begin(); it != session_order_.end(); ++it) {
      if (*it == record.session_id) {
        session_order_.erase(it);
        break;
      }
    }
  }
  session_order_.push_back(record.session_id);
  // Global session cap: drop the session with the oldest last turn.
  while (session_history_.size() > config_.session_history.max_sessions &&
         !session_order_.empty()) {
    const std::string oldest = std::move(session_order_.front());
    session_order_.pop_front();
    session_history_.erase(oldest);
  }
}

UnifiedEvent UnifiedEngine::finish_locked(const std::string& request_id,
                                          Record& record,
                                          UnifiedEvent::Kind kind,
                                          std::string error, Value result) {
  record.phase = Phase::kDone;
  UnifiedEvent event;
  event.request_id = request_id;  // always the unified id, never the child id
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
  return event;
}

void UnifiedEngine::handle_complete(const RequestInfo& info) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    // The completion carries the ASR/VLM id or an output-leg child id.
    auto it = find_record_locked(info.request_id);
    if (it != records_.end() && it->second.phase != Phase::kDone) {
      Record& record = it->second;
      const std::string& unified_id = it->first;
      const bool image_child =
          record.image_leg.submitted &&
          info.request_id == record.image_leg.child_id;
      const bool tts_child =
          record.tts_leg.submitted &&
          info.request_id == record.tts_leg.child_id;
      OutputLeg* output_leg = image_child ? &record.image_leg
                              : tts_child ? &record.tts_leg : nullptr;
      if (output_leg != nullptr) {
        // Output-leg join. A failed or aborted leg never fails the whole
        // request after a successful VLM: it becomes a leg_error and the
        // request still settles with one kComplete (or kCancelled when the
        // client asked to cancel).
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
            emit_artifact_locked(unified_id, record, *output_leg, info.result);
            settle_output_leg_locked(unified_id, record, *output_leg,
                                     UnifiedEvent::Kind::kComplete, {});
          } catch (const std::exception& error) {
            settle_output_leg_locked(
                unified_id, record, *output_leg,
                UnifiedEvent::Kind::kLegError, error.what());
          }
        }
      } else if (info.state == "aborted") {
        dispatch_queue_.push_back(finish_locked(
            unified_id, record, UnifiedEvent::Kind::kCancelled, {}, {}));
      } else if (info.state == "failed") {
        dispatch_queue_.push_back(finish_locked(
            unified_id, record, UnifiedEvent::Kind::kError,
            info.error.empty() ? "stage failure" : info.error, {}));
      } else if (record.cancel_requested) {
        // A successful completion beat the abort frame through the
        // Coordinator. The intent was recorded under mu_ first, so this path is
        // deterministic: settle as kCancelled and never run (or trust) the VLM
        // leg. A record that already failed is kDone and cannot reach here.
        dispatch_queue_.push_back(finish_locked(
            unified_id, record, UnifiedEvent::Kind::kCancelled, {}, {}));
      } else if (record.vlm_id.empty()) {
        // ASR leg succeeded. A non-streaming stage sends no final chunk, so
        // synthesize the ASR final from the result text.
        if (!record.asr_final_seen) {
          const auto text = asr_result_text(info.result);
          if (!text) {
            // No usable text means a malformed stage result: fail here and
            // never start the VLM leg with an empty or invented prompt.
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
        const auto text = vlm_result_text(info.result);
        if (!text) {
          dispatch_queue_.push_back(finish_locked(
              unified_id, record, UnifiedEvent::Kind::kError,
              "VLM stage returned no text", {}));
        } else {
          if (record.vlm_text.empty()) record.vlm_text = *text;
          record.vlm_result = info.result;
          if (record.image_leg.enabled || record.tts_leg.enabled) {
            try {
              start_outputs_locked(unified_id, record);
            } catch (const std::exception& error) {
              dispatch_queue_.push_back(finish_locked(
                  unified_id, record, UnifiedEvent::Kind::kError,
                  error.what(), {}));
            }
          } else {
            dispatch_queue_.push_back(finish_locked(
                unified_id, record, UnifiedEvent::Kind::kComplete, {},
                info.result));
            // Only a successful terminal enters history. Cancelled and failed
            // requests never reach here, so no failed output becomes context.
            if (record.history_requested) record_session_turn_locked(record);
          }
        }
      }
    }
    // Completion forwarding follows every event generated by this completion.
    // A nested or concurrent caller only queues; the active drainer calls sinks.
    dispatch_queue_.push_back(info);
  }

  drain_dispatch();
}

Value UnifiedEngine::build_image_inputs(const Record& record) {
  Value::Map out = record.image_options;
  // An explicit prompt wins; otherwise the leg gets the same VLM answer.
  auto prompt = out.find("prompt");
  if (prompt == out.end() || !prompt->second.is_str() ||
      prompt->second.as_str().empty()) {
    out["prompt"] = Value(record.vlm_text);
  }
  return Value(std::move(out));
}

Value UnifiedEngine::build_tts_inputs(const Record& record) {
  Value::Map out = record.audio_options;
  // The TTS input is always the aggregated VLM answer, never a delta.
  out["text"] = Value(record.vlm_text);
  return Value(std::move(out));
}

void UnifiedEngine::start_image_locked(const std::string& request_id,
                                       Record& record) {
  OutputLeg& leg = record.image_leg;
  leg.child_id = request_id + kImageSuffix;
  StagePayload payload;
  payload.request_id = leg.child_id;
  payload.request.inputs = build_image_inputs(record);
  payload.request.params = record.vlm_params;
  payload.request.metadata = record.vlm_metadata;
  leg.submitted = true;
  coordinator_.submit(payload, leg.stage_name);
}

void UnifiedEngine::start_tts_locked(const std::string& request_id,
                                     Record& record) {
  OutputLeg& leg = record.tts_leg;
  leg.child_id = request_id + kTtsSuffix;
  StagePayload payload;
  payload.request_id = leg.child_id;
  payload.request.inputs = build_tts_inputs(record);
  payload.request.params = record.vlm_params;
  payload.request.metadata = record.vlm_metadata;
  leg.submitted = true;
  coordinator_.submit(payload, leg.stage_name);
}

void UnifiedEngine::start_outputs_locked(const std::string& request_id,
                                         Record& record) {
  record.phase = Phase::kOutputs;
  record.public_info.current_stage = kImageStageId;
  if (record.image_leg.enabled) start_image_locked(request_id, record);
  if (record.tts_leg.enabled) start_tts_locked(request_id, record);
  record.legs_remaining =
      (record.image_leg.submitted ? 1 : 0) +
      (record.tts_leg.submitted ? 1 : 0);
}

void UnifiedEngine::emit_artifact_locked(const std::string& request_id,
                                         Record& record,
                                         const OutputLeg& leg,
                                         const Value& result) {
  if (!result.is_map()) {
    throw ProtocolError(std::string(leg.stage_id) +
                        " stage returned a non-map artifact result");
  }
  const Value* data = result.find("data");
  if (data == nullptr || !data->is_bin() || data->as_bin().empty()) {
    throw ProtocolError(std::string(leg.stage_id) +
                        " stage returned no artifact bytes");
  }
  const Value* media_type = result.find("media_type");
  if (media_type == nullptr || !media_type->is_str()) {
    throw ProtocolError(std::string(leg.stage_id) +
                        " stage returned no media_type");
  }
  const std::string& media = media_type->as_str();
  const size_t size = data->as_bin().size();
  const size_t per_leg_limit = media == "image" ? kMaxArtifactImageBytes
                                                : kMaxArtifactAudioBytes;
  if (media != "image" && media != "audio") {
    throw ProtocolError("artifact media_type must be image or audio");
  }
  if (size > per_leg_limit) {
    throw ProtocolError(std::string(leg.stage_id) +
                        " artifact exceeds its decoded size limit");
  }
  if (record.artifact_bytes_total + size > kMaxArtifactAggregateBytes) {
    throw ProtocolError("mixed output artifacts exceed the aggregate limit");
  }
  record.artifact_bytes_total += size;

  // Descriptor: every worker-provided key except the inline Bin, plus delivery
  // ordering and chunk bookkeeping.
  Value::Map descriptor;
  for (const auto& [key, value] : result.as_map()) {
    if (key != "data") descriptor[key] = value;
  }
  const std::string artifact_id =
      "art-" + std::to_string(record.next_artifact++);
  descriptor["artifact_id"] = Value(artifact_id);
  const size_t total =
      (size + kArtifactChunkTargetBytes - 1) / kArtifactChunkTargetBytes;
  descriptor["total"] = Value(static_cast<int64_t>(total));

  auto bytes = std::make_shared<const std::string>(data->as_bin());
  for (size_t index = 0; index < total; ++index) {
    const size_t offset = index * kArtifactChunkTargetBytes;
    const size_t length = std::min(kArtifactChunkTargetBytes, size - offset);
    Value::Map chunk_descriptor = descriptor;
    chunk_descriptor["index"] = Value(static_cast<int64_t>(index));
    UnifiedEvent event;
    event.request_id = request_id;
    event.session_id = record.session_id;
    event.stage_id = leg.stage_id;
    event.kind = UnifiedEvent::Kind::kArtifact;
    event.seq = record.next_seq++;
    event.is_final = true;
    event.artifact = Value(std::move(chunk_descriptor));
    event.artifact_bytes = bytes;
    event.artifact_offset = offset;
    event.artifact_length = length;
    dispatch_queue_.push_back(std::move(event));
  }
}

void UnifiedEngine::settle_output_leg_locked(
    const std::string& request_id, Record& record, OutputLeg& leg,
    UnifiedEvent::Kind kind, std::string error) {
  leg.state = kind == UnifiedEvent::Kind::kComplete ? "completed"
            : kind == UnifiedEvent::Kind::kCancelled ? "aborted"
                                                     : "failed";
  leg.error = error;
  if (kind != UnifiedEvent::Kind::kComplete) {
    UnifiedEvent event;
    event.request_id = request_id;
    event.session_id = record.session_id;
    event.stage_id = leg.stage_id;
    event.kind = kind == UnifiedEvent::Kind::kCancelled
                     ? UnifiedEvent::Kind::kLegError
                     : kind;
    event.seq = record.next_seq++;
    event.error = std::move(error);
    dispatch_queue_.push_back(std::move(event));
  }
  --record.legs_remaining;
  if (record.legs_remaining == 0) on_legs_drained_locked(request_id, record);
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
  add(record.tts_leg);
  return Value(std::move(legs));
}

void UnifiedEngine::on_legs_drained_locked(const std::string& request_id,
                                           Record& record) {
  // A client-requested cancel still settles as one kCancelled even when some
  // artifacts were delivered; ordinary leg failures keep kComplete.
  const UnifiedEvent::Kind kind = record.cancel_requested
                                      ? UnifiedEvent::Kind::kCancelled
                                      : UnifiedEvent::Kind::kComplete;
  UnifiedEvent event = finish_locked(request_id, record, kind, {},
                                     record.vlm_result);
  event.legs = legs_value_locked(record);
  dispatch_queue_.push_back(std::move(event));
  if (kind == UnifiedEvent::Kind::kComplete && record.history_requested) {
    record_session_turn_locked(record);
  }
}

}  // namespace umminfer
