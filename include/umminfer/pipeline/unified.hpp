// SPDX-License-Identifier: Apache-2.0
// Unified multimodal engine v1: in-process ASR -> VLM composition.
//
// This is a value object plus a thin orchestrator over the existing
// Coordinator. It adds no second engine: the ASR leg is one StagePayload
// submitted to the registered ASR stage, the VLM leg reuses the same
// Coordinator against the registered VLM stage, every stage event stays a
// StreamMessage or CompleteMessage, and cancellation is the Coordinator abort
// broadcast the stages already implement. The orchestrator adds only:
//   - the unified request/event contract (modality, sequence, terminal state),
//   - deterministic aggregation of ASR output into one VLM prompt,
//   - a bounded admission queue (backpressure).
//
// One request may carry one image_url content part and one client text
// instruction next to the audio. The ASR leg still receives PCM only; the VLM
// leg receives the instruction, the image part, and the final ASR transcript as
// the last text part. A request without the binary audio input skips the ASR
// leg and runs the VLM leg directly against the original messages, so plain
// string text and image+text requests use the same public chat route. Video
// has no worker and is rejected at submission.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

#include "umminfer/pipeline/pipeline.hpp"
#include "umminfer/proto/messages.hpp"

namespace umminfer {

// Modalities understood by the unified contract. Audio and text run today;
// image and video are capability declarations only.
enum class UnifiedModality { kAudio, kText, kImage, kVideo };

const char* to_string(UnifiedModality modality);
std::optional<UnifiedModality> modality_from_string(const std::string& token);

// Stage ids used inside unified events. They are contract values, not
// Coordinator stage names: deployed stages keep their configured names.
inline constexpr const char* kAsrStageId = "asr";
inline constexpr const char* kVlmStageId = "vlm";
inline constexpr const char* kImageStageId = "image";
inline constexpr const char* kTtsStageId = "tts";

// Child request id suffixes submitted after the VLM barrier. Like ":vlm" they
// never advance the Coordinator id counter.
inline constexpr const char* kImageSuffix = ":image";
inline constexpr const char* kTtsSuffix = ":tts";

// Mixed-output byte contract (frozen). The HTTP boundary rejects anything past
// these limits; the client library mirrors the same values.
inline constexpr size_t kMaxArtifactImageBytes = 48u << 20;
inline constexpr size_t kMaxArtifactAudioBytes = 48u << 20;
inline constexpr size_t kMaxArtifactAggregateBytes = 60u << 20;
// Raw bytes per artifact chunk. Its base64 line stays around 2.7 MiB, below
// both the 3 MiB wire ceiling and the 4 MiB SSE parser limit.
inline constexpr size_t kArtifactChunkTargetBytes = 2u << 20;
inline constexpr size_t kMaxSseFrameWireBytes = 3u << 20;

// One event of a unified request. The sequence number is gap-free per request
// and starts at 0, so a consumer verifies ordering directly.
struct UnifiedEvent {
  enum class Kind {
    kAsrPartial,  // streaming ASR transcript, is_final == false
    kAsrFinal,    // terminal ASR transcript, is_final == true
    kVlmDelta,    // streaming VLM token
    kArtifact,    // one generated image/audio chunk (see artifact fields)
    kLegError,    // an optional output leg failed; the request still completes
    kComplete,    // terminal success
    kError,       // terminal failure
    kCancelled,   // terminal cancellation
  };

  std::string request_id;
  std::string session_id;
  // kAsrStageId / kVlmStageId / kImageStageId / kTtsStageId; empty on
  // kCancelled.
  std::string stage_id;
  Kind kind = Kind::kError;
  int64_t seq = 0;
  bool is_final = false;
  // ASR transcript text or one VLM delta. Empty on terminal error/cancel.
  std::string text;
  // Present on kComplete: the aggregated VLM result value.
  Value result;
  // Present on kError.
  std::string error;

  // kArtifact only. Descriptor map without inline bytes or hashes:
  // artifact_id, media_type, mime_type, index, total, and the format-specific
  // fields (image: format/size/seed, audio: sample_rate/channels). The raw
  // chunk lives in artifact_bytes; the HTTP layer base64-encodes it, and adds
  // sha256/bytes on the final chunk, on its provider thread.
  Value artifact;
  std::shared_ptr<const std::string> artifact_bytes;
  size_t artifact_offset = 0;
  size_t artifact_length = 0;

  // Terminal only, and only for requests that used optional output legs:
  // [{stage, state ("completed"|"failed"|"aborted"), error?}]. Absent on the
  // default text path so the public event stream stays byte-identical.
  Value legs;
};

const char* to_string(UnifiedEvent::Kind kind);

struct UnifiedConfig {
  // Coordinator stage names the two deployed stages registered under.
  std::string asr_stage = "asr";
  std::string vlm_stage = "vlm";
  // Optional output stages. Empty means the stage is not registered and a
  // request asking for that modality is rejected at the HTTP boundary. They
  // never enter the /ready gate: a zero-optional-worker deployment stays a
  // text-only server.
  std::string image_stage;
  std::string tts_stage;
  // Hard backpressure: requests past this many in-flight are rejected at
  // submission instead of being buffered without limit.
  size_t max_in_flight = 32;
  // Opt-in multi-turn history. Disabled by default: session_id stays a grouping
  // label. When enabled, a request must also set inputs.session_history=true;
  // only requests carrying an explicit session_id then share history. The store
  // is process-local, guarded by the engine mutex, and dies with the engine.
  struct SessionHistory {
    bool enabled = false;
    // Completed turns kept per session. Oldest turns are evicted first.
    size_t max_turns = 8;
    // Hard character budget per session across transcript and answer text.
    // Oldest turns leave first until the budget holds.
    size_t max_chars = 16384;
    // Hard cap on distinct sessions retained. The session whose last recorded
    // turn is oldest is evicted first.
    size_t max_sessions = 256;
  } session_history;
};

// Thrown by UnifiedEngine::submit when the admission limit is reached.
class QueueFullError : public std::runtime_error {
 public:
  QueueFullError() : std::runtime_error("unified engine queue is full") {}
};

// Owns no threads. Construction must precede the first Coordinator::run call;
// exchange_sinks_before_run enforces this. Keep other Coordinator calls quiescent
// during construction. The caller owns the pump and must stop and join it before
// destroying the Coordinator. The Coordinator must outlive the engine.
// Quiesce other engine API calls before destruction. Do not destroy the engine
// from one of its callbacks: the destructor waits for active callbacks.
// Destruction does not require stopping the pump; retired sinks keep forwarding
// to the sinks that preceded this engine.
class UnifiedEngine {
 public:
  UnifiedEngine(Coordinator& coordinator, UnifiedConfig config);
  // Destructor retirement drains forwarded callbacks but does not propagate a
  // foreign sink exception. Non-destructor copied sinks drain then rethrow.
  ~UnifiedEngine() noexcept;
  UnifiedEngine(const UnifiedEngine&) = delete;
  UnifiedEngine& operator=(const UnifiedEngine&) = delete;

  // Submits one unified request. Reuses StagePayload:
  //   payload.request.inputs = {"messages": [...], "modalities": [...]}
  // An additional binary "audio" value selects the real ASR worker contract.
  // The ASR leg then receives only that audio with stream=false/temperature=0;
  // the VLM leg keeps the original sampling parameters and receives the image
  // parts (at most one) plus the ASR transcript as text.
  // inputs.session_id is optional. It is a grouping label by default: every
  // event of every request carrying it reports the same session_id, and
  // requests in one session stay independent (own request id, own seq, own
  // terminal, no history). When config_.session_history.enabled is true, a
  // request may additionally set inputs.session_history=true; the VLM prompt of
  // later requests in the same session then carries prior completed turns.
  // Returns the Coordinator-allocated request id. Throws ProtocolError on a
  // bad contract (including video) and QueueFullError on backpressure.
  std::string submit(StagePayload payload);

  // Cancels the request on both stages via the Coordinator abort broadcast.
  // Returns false when the request is unknown or already terminal. source
  // labels the abort metric: "client", "timeout" or "shutdown".
  bool cancel(const std::string& request_id, const char* source = "client");

  // Returns false for an unknown id or timeout, true only for a terminal record.
  bool wait(const std::string& request_id, double timeout_s);
  // Drops the terminal engine record. Mirrors Coordinator::release.
  bool release(const std::string& request_id);
  // Returns only unified state: unified id, public stage id, and unified result.
  std::optional<RequestInfo> info(const std::string& request_id);
  size_t in_flight() const;
  // Whether the optional output stage was configured. The HTTP boundary uses
  // these to reject an unregistered modality with 400 before submit.
  bool image_output_enabled() const { return !config_.image_stage.empty(); }
  bool tts_output_enabled() const { return !config_.tts_stage.empty(); }
  // Whether the engine was constructed with session history enabled.
  bool history_enabled() const { return config_.session_history.enabled; }
  // Whether an optional output stage is registered in this process. The HTTP
  // boundary turns a false into a 400 before any submit.
  bool image_enabled() const { return !config_.image_stage.empty(); }
  bool tts_enabled() const { return !config_.tts_stage.empty(); }

  // Fires for every ASR partial/final and VLM delta, in sequence order.
  std::function<void(const UnifiedEvent&)> on_event;
  // Fires once per request, including failures and cancellations.
  std::function<void(const UnifiedEvent&)> on_terminal;

 private:
  enum class Phase { kAsr, kVlm, kOutputs, kDone };
  struct CallbackGate;
  friend bool unified_retirement_waiting_for_test(const UnifiedEngine& engine);

  // One optional output leg's runtime state.
  struct OutputLeg {
    std::string stage_name;  // Coordinator stage, empty when unused
    std::string child_id;    // request_id + kImageSuffix / kTtsSuffix
    std::string stage_id;    // kImageStageId / kTtsStageId
    bool enabled = false;
    bool submitted = false;
    // completed, failed, or aborted; empty while running.
    std::string state;
    std::string error;
    std::chrono::steady_clock::time_point started;
  };

  struct Record {
    std::string session_id;
    Phase phase = Phase::kAsr;
    // Metrics and tracing. steady_clock measures durations; start_epoch_ms is
    // the wall-clock start the trace record carries. A negative leg value means
    // that leg did not run for this request.
    std::string request_type = "chat";
    std::chrono::steady_clock::time_point started;
    int64_t start_epoch_ms = 0;
    std::chrono::steady_clock::time_point asr_started;
    std::chrono::steady_clock::time_point vlm_started;
    double asr_ms = -1;
    double vlm_ms = -1;
    double image_ms = -1;
    double tts_ms = -1;
    RequestInfo public_info;
    int64_t next_seq = 0;
    bool asr_final_seen = false;
    // Set by cancel() under mu_ before any coordinator call. A completion that
    // wins the race against abort is still required to settle as kCancelled
    // and must not start the VLM leg.
    bool cancel_requested = false;
    // The request opted into multi-turn history. Its turn is only recorded on
    // a successful terminal; cancelled and failed requests leave no history.
    bool history_requested = false;
    std::string transcript;  // partials concatenated; final replaces them
    // The real current-turn user text. On an audio turn transcript carries the
    // ASR result; on a VLM-only turn transcript is empty and this holds the
    // string/text-part input so session history keeps what the user said.
    std::string user_text;
    std::string vlm_text;
    std::string vlm_id;
    Value original_inputs;
    Value::Map vlm_params;
    Value::Map vlm_metadata;
    // Options the HTTP boundary attached for the optional output legs. They
    // are stripped from the VLM prompt and consumed only by the legs.
    Value::Map image_options;
    Value::Map audio_options;
    // Opt-in generated outputs. The legs exist only when the request named
    // the modality; legs_remaining counts submitted legs that have not
    // reached a terminal state yet.
    OutputLeg image_leg;
    std::vector<OutputLeg> tts_legs;
    bool tts_enabled = false;
    std::string tts_voice;
    // Sentence pipeline and per-chunk audio delivery (audio_options.stream).
    bool tts_stream = false;
    std::string sentence_buf;
    size_t tts_bytes = 0;
    bool tts_failed = false;
    std::string tts_error;
    // The VLM leg reached its terminal. An output leg may finish earlier, so
    // the request completes only when this is true and legs_remaining is 0.
    bool vlm_done = false;
    int legs_remaining = 0;
    // VLM result held across the output join; it is the terminal result.
    Value vlm_result;
    // Artifact delivery order ("art-0", "art-1") and the decoded-byte budget
    // of everything already delivered for this request.
    int next_artifact = 0;
    size_t artifact_bytes_total = 0;
  };

  // One completed user/assistant turn kept in process-local session history.
  struct SessionTurn {
    std::string user;
    std::string assistant;
  };

  void handle_stream(const StreamMessage& msg);
  void handle_complete(const RequestInfo& info);
  void drain_dispatch();
  void start_vlm_locked(const std::string& request_id, Record& record);
  // Optional output legs, submitted together at the single VLM barrier.
  void start_outputs_locked(const std::string& request_id, Record& record);
  void start_image_locked(const std::string& request_id, Record& record);
  // Submits one TTS leg. `first` keeps the contract child id <id>:tts for the
  // single-leg (non-streaming) request; streamed sentences use :tts:0, :tts:1.
  void start_tts_locked(const std::string& request_id, Record& record,
                        std::string text, bool first);
  // Cuts every complete sentence out of sentence_buf and submits it.
  void drain_sentences_locked(const std::string& request_id, Record& record);
  // Submits the tail of sentence_buf; called at the VLM terminal.
  void flush_sentences_locked(const std::string& request_id, Record& record);
  // Turns one streamed TTS chunk into a PCM16 artifact event.
  void emit_tts_chunk_locked(const std::string& request_id, Record& record,
                             const Value& chunk);
  // Aborts every submitted TTS leg; used when the VLM leg fails.
  void abort_tts_legs_locked(Record& record);
  // Builds the inputs of one output leg from the VLM answer and the options
  // the HTTP boundary attached to the original request.
  static Value build_image_inputs(const Record& record);
  static Value build_tts_inputs(const Record& record, const std::string& text);
  // Turns a successful output-leg completion into kArtifact events with
  // reserved gap-free seq values.
  void emit_artifact_locked(const std::string& request_id, Record& record,
                            const OutputLeg& leg, const Value& result);
  // Records a failed/aborted output leg as a kLegError event and advances the
  // join; a VLM failure settles the whole request through finish_locked.
  void settle_output_leg_locked(const std::string& request_id, Record& record,
                                OutputLeg& leg, UnifiedEvent::Kind kind,
                                std::string error);
  void on_legs_drained_locked(const std::string& request_id, Record& record);
  // The legs array carried by the terminal event; nil when no output leg ran.
  Value legs_value_locked(const Record& record) const;
  // Returns the current messages with prior completed turns of
  // record.session_id prepended as user/assistant messages.
  Value::Array with_session_history_locked(
      const Record& record, const Value::Array& current_messages);
  // Stores the finished turn and enforces the turn and character bounds.
  void record_session_turn_locked(const Record& record);
  UnifiedEvent finish_locked(const std::string& request_id, Record& record,
                             UnifiedEvent::Kind kind, std::string error,
                             Value result);
  // Records are keyed by the ASR leg id; VLM events carry the derived child id.
  // ponytail: linear fallback scan; max_in_flight bounds the record count.
  std::map<std::string, Record>::iterator find_record_locked(
      const std::string& event_request_id);
  static Value build_vlm_inputs(const std::string& transcript,
                                const Value& original_inputs);

  Coordinator& coordinator_;
  UnifiedConfig config_;
  std::shared_ptr<CallbackGate> callback_gate_;

  mutable std::mutex mu_;
  std::condition_variable cv_;
  // FIFO of active unified ids, bounded by config_.max_in_flight.
  std::deque<std::string> admission_;
  std::map<std::string, Record> records_;
  using DispatchItem = std::variant<UnifiedEvent, StreamMessage, RequestInfo>;
  std::deque<DispatchItem> dispatch_queue_;
  bool dispatching_ = false;  // guarded by mu_; reentrant callbacks only queue
  std::atomic<uint64_t> session_counter_{0};
  // Process-local, in-memory history. No persistence: engine destruction or
  // process restart drops every turn.
  std::map<std::string, std::deque<SessionTurn>> session_history_;
  // Insertion order of sessions with retained turns; front is the oldest.
  // Entries for sessions that emptied through self-eviction are stale and are
  // skipped when the global cap is enforced.
  std::deque<std::string> session_order_;
};

}  // namespace umminfer
