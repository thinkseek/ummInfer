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
// string text and image+text requests use the same public chat route. Optional
// output legs (image, audio/TTS, video) fan out in parallel after the VLM leg
// and share one unified LegState lifecycle.
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
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "umminfer/pipeline/pipeline.hpp"
#include "umminfer/pipeline/session_store.hpp"
#include "umminfer/proto/messages.hpp"

namespace umminfer {

class NarrativeStreamParser;

// Modalities understood by the unified contract. Text, image, audio and video
// share one modality representation and leg lifecycle in UnifiedEngine.
enum class UnifiedModality { kAudio, kText, kImage, kVideo };

const char* to_string(UnifiedModality modality);
std::optional<UnifiedModality> modality_from_string(const std::string& token);

// Unified lifecycle states shared by every request leg (ASR, VLM, image,
// audio/TTS, video).
enum class LegState { kPending, kRunning, kCompleted, kFailed, kAborted };

const char* to_string(LegState state);

struct LegSnapshot {
  UnifiedModality modality = UnifiedModality::kText;
  std::string stage_id;
  LegState state = LegState::kPending;
  std::string error;
  // "slot-N" for an interleaved narrative visual slot; empty otherwise.
  std::string slot_id;
};

// Stage ids used inside unified events. They are contract values, not
// Coordinator stage names: deployed stages keep their configured names.
inline constexpr const char* kAsrStageId = "asr";
inline constexpr const char* kVlmStageId = "vlm";
inline constexpr const char* kImageStageId = "image";
inline constexpr const char* kTtsStageId = "tts";
inline constexpr const char* kVideoStageId = "video";

// Child request id suffixes submitted after the VLM barrier. Like ":vlm" they
// never advance the Coordinator id counter.
inline constexpr const char* kImageSuffix = ":image";
inline constexpr const char* kTtsSuffix = ":tts";
inline constexpr const char* kVideoSuffix = ":video";

// Mixed-output byte contract (frozen). The HTTP boundary rejects anything past
// these limits; the client library mirrors the same values.
inline constexpr size_t kMaxArtifactImageBytes = 48u << 20;
inline constexpr size_t kMaxArtifactAudioBytes = 48u << 20;
inline constexpr size_t kMaxArtifactVideoBytes = 48u << 20;
inline constexpr size_t kMaxArtifactAggregateBytes = 60u << 20;
// Raw bytes per artifact chunk. Its base64 line stays around 2.7 MiB, below
// both the 3 MiB wire ceiling and the 4 MiB SSE parser limit.
inline constexpr size_t kArtifactChunkTargetBytes = 2u << 20;
inline constexpr size_t kMaxSseFrameWireBytes = 3u << 20;
// Hard cap on streamed TTS sentence legs per unified request. Once N-1 legs are
// submitted, remaining text stays in sentence_buf for the single final tail leg.
inline constexpr size_t kMaxStreamedTtsLegs = 32;

// One event of a unified request. The sequence number is gap-free per request
// and starts at 0, so a consumer verifies ordering directly.
struct UnifiedEvent {
  enum class Kind {
    kAsrPartial,  // streaming ASR transcript, is_final == false
    kAsrFinal,    // terminal ASR transcript, is_final == true
    kVlmDelta,    // streaming VLM token
    kVisualSlot,  // one narrative visual marker closed mid-stream
    kArtifact,    // one generated image/audio/video chunk (see artifact fields)
    kLegError,    // an optional output leg failed; the request still completes
    kComplete,    // terminal success
    kError,       // terminal failure
    kCancelled,   // terminal cancellation
  };

  std::string request_id;
  std::string session_id;
  // kAsrStageId / kVlmStageId / kImageStageId / kTtsStageId / kVideoStageId;
  // empty on kCancelled.
  std::string stage_id;
  Kind kind = Kind::kError;
  int64_t seq = 0;
  bool is_final = false;
  // ASR transcript text or one VLM delta. Empty on terminal error/cancel.
  std::string text;
  // kVisualSlot and slot-attributed kArtifact/kLegError. slot_index is -1 when
  // the event does not belong to a narrative slot.
  std::string slot_id;
  int slot_index = -1;
  // kVisualSlot: the directive prompt and its byte offset in the clean prose.
  std::string prompt;
  int64_t text_offset = 0;
  // kVisualSlot continuity metadata (opt-in; empty / -1 when continuity is off).
  std::string continuity_mode;
  int anchor_slot_index = -1;
  // Explicit directives never expose a timing-dependent digest on visual_slot.
  // Settled I2V artifacts and terminal legs carry the verified anchor digest.
  bool anchor_explicit = false;
  std::string anchor_sha256;
  // Present on kComplete: the aggregated VLM result value.
  Value result;
  // Present only for an explicit telemetry opt-in terminal event.
  Value telemetry;
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
  std::string video_stage;
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

  enum class CancelStatus { kCancelled, kNotFoundOrTerminal, kSessionMismatch };

  struct BargeInResult {
    // The id and session belonged to an active or playback-pending realtime
    // response. Repeated calls are safe but record history at most once.
    bool accepted = false;
    bool generation_cancelled = false;
    bool history_recorded = false;
    size_t acknowledged_text_cursor = 0;
    std::string assistant_prefix;
  };

  // Realtime-only cancellation with an explicit client playback cursor. This
  // is the sole cancelled-request history exception: it requires the internal
  // realtime_playback_ack submission contract and never applies to HTTP
  // cancellation or failure. A generation-complete response can still be in
  // playback, so this also accepts a terminal record awaiting its final ack.
  BargeInResult cancel_realtime_barge_in(
      const std::string& request_id, std::string_view expected_session_id,
      size_t acknowledged_text_cursor);
  // Commits a fully played, generation-complete realtime response. Returns
  // false for a stale id, wrong session, partial cursor, or in-flight response.
  bool complete_realtime_playback(const std::string& request_id,
                                  std::string_view expected_session_id,
                                  size_t acknowledged_text_cursor);

  // Cancels the request on both stages via the Coordinator abort broadcast.
  // Returns false when the request is unknown or already terminal. source
  // labels the abort metric: "client", "timeout" or "shutdown".
  bool cancel(const std::string& request_id, const char* source = "client");
  // Same as cancel(), additionally verifying `expected_session_id` when
  // non-empty so multi-tenant callers cannot abort another session's request.
  CancelStatus cancel_for_session(const std::string& request_id,
                                  std::string_view expected_session_id,
                                  const char* source = "client");

  // Returns false for an unknown id or timeout, true only for a terminal record.
  bool wait(const std::string& request_id, double timeout_s);
  // Drops the terminal engine record. Mirrors Coordinator::release.
  bool release(const std::string& request_id);
  // Returns only unified state: unified id, public stage id, and unified result.
  std::optional<RequestInfo> info(const std::string& request_id);
  // Returns the current unified leg lifecycle states (pending, running,
  // completed, failed, aborted) for every leg of an active or terminal request.
  std::vector<LegSnapshot> legs_snapshot(const std::string& request_id) const;
  size_t in_flight() const;
  // Whether the optional output stage was configured. The HTTP boundary uses
  // these to reject an unregistered modality with 400 before submit.
  bool image_output_enabled() const { return !config_.image_stage.empty(); }
  bool tts_output_enabled() const { return !config_.tts_stage.empty(); }
  bool video_output_enabled() const { return !config_.video_stage.empty(); }
  // Whether the engine was constructed with session history enabled.
  bool history_enabled() const { return config_.session_history.enabled; }
  // Whether an optional output stage is registered in this process. The HTTP
  // boundary turns a false into a 400 before any submit.
  bool image_enabled() const { return !config_.image_stage.empty(); }
  bool tts_enabled() const { return !config_.tts_stage.empty(); }
  bool video_enabled() const { return !config_.video_stage.empty(); }
  // Coordinator stage name of the TTS worker, empty when none is registered.
  // The standalone /v1/audio/speech route submits to that same stage.
  const std::string& tts_stage_name() const { return config_.tts_stage; }
  // Coordinator stage name of the image worker, empty when none is registered.
  // The standalone /v1/visual/render route submits to that same stage.
  const std::string& image_stage_name() const { return config_.image_stage; }
  // Coordinator stage name of the video worker, empty when none is registered.
  const std::string& video_stage_name() const { return config_.video_stage; }

  // Fires for every ASR partial/final and VLM delta, in sequence order.
  std::function<void(const UnifiedEvent&)> on_event;
  // Fires once per request, including failures and cancellations.
  std::function<void(const UnifiedEvent&)> on_terminal;

 private:
  enum class Phase { kAsr, kVlm, kOutputs, kDone };
  struct CallbackGate;

  // One optional output leg's runtime state.
  struct OutputLeg {
    UnifiedModality modality = UnifiedModality::kText;
    std::string stage_name;  // Coordinator stage, empty when unused
    std::string child_id;    // request_id + :image / :tts / :video
    std::string stage_id;    // kImageStageId / kTtsStageId / kVideoStageId
    bool enabled = false;
    bool submitted = false;
    // Byte range in Record::vlm_text spoken by this TTS leg. audio.delta
    // acknowledgements use text_begin, a conservative prefix that excludes the
    // current (possibly partially played) sentence; text_end is informational.
    size_t text_begin = 0;
    size_t text_end = 0;
    LegState state = LegState::kPending;
    std::string error;
    std::chrono::steady_clock::time_point started;
    // Narrative slot attribution. slot_index is -1 for the single-shot image
    // and video legs and for every TTS leg.
    std::string slot_id;
    int slot_index = -1;
    // Stage B i2v_chain continuity state. Captured on completed image slots
    // while the request is active and cleared on request completion/cancel.
    std::string rendered_image_data_url;
    // Immutable request-local bytes and their content address. The data URL is
    // retained for the mainline implicit-chain contract; explicit references
    // dispatch these exact bytes after recomputing their SHA-256.
    std::shared_ptr<const std::string> rendered_image_bytes;
    std::string rendered_image_sha256;
    std::string rendered_image_mime_type;
    // Persistent anchor attribution for artifact and terminal metadata.
    int anchor_slot_index = -1;
    std::string anchor_sha256;
    bool anchor_explicit = false;
    // Deferred video slot waiting on an earlier running image slot in
    // i2v_chain mode.
    int deferred_chain_anchor_slot = -1;
    std::optional<StagePayload> deferred_payload;
  };

  struct Record {
    std::string session_id;
    Phase phase = Phase::kAsr;
    bool terminal_dispatched = false;
    bool has_audio = false;
    LegState asr_state = LegState::kPending;
    std::string asr_error;
    LegState vlm_state = LegState::kPending;
    std::string vlm_error;
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
    double video_ms = -1;
    // Fixed-size stage telemetry records are collected only for an explicit
    // request opt-in and projected to one bounded terminal envelope.
    bool telemetry_requested = false;
    std::vector<Value> telemetry_stages;
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
    // Internal realtime playback-ack mode defers even successful history until
    // the client acknowledges playback. Barge-in may then retain exactly the
    // acknowledged prefix once instead of the generated answer.
    bool realtime_playback_ack = false;
    bool realtime_history_recorded = false;
    std::string realtime_history_prefix;
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
    Value::Map video_options;
    // Opt-in generated outputs. The legs exist only when the request named
    // the modality; legs_remaining counts submitted legs that have not
    // reached a terminal state yet.
    OutputLeg image_leg;
    OutputLeg video_leg;
    std::vector<OutputLeg> tts_legs;
    // Interleaved narrative. The image/video legs above describe the enabled
    // output modality, but in narrative mode visuals come from parsed
    // directives; each dispatched directive becomes one slot leg. The parser
    // owns the prose/directive split and is constructed only when enabled.
    bool narrative_enabled = false;
    std::string narrative_visual_mode = "auto";
    int narrative_max_slots = 0;
    bool narrative_allow_text_only = false;
    std::string narrative_continuity_mode = "off";
    std::string narrative_style_anchor;
    std::string narrative_anchor_prompt;
    std::shared_ptr<NarrativeStreamParser> narrative_parser;
    std::vector<OutputLeg> slot_legs;
    int narrative_next_slot = 0;
    // A VLM delta stream was consumed. At completion the parser is only
    // flushed, never re-fed, so directives already dispatched mid-stream are
    // not dispatched a second time.
    bool vlm_stream_seen = false;
    bool tts_enabled = false;
    std::string tts_voice;
    // Sentence pipeline and per-chunk audio delivery (audio_options.stream).
    bool tts_stream = false;
    std::string sentence_buf;
    // Raw vlm_text byte offset consumed by the ordered TTS sentence fan-out.
    size_t tts_text_cursor = 0;
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

    // Single-shot optional output legs (image, video) share one lifecycle;
    // multi-sentence streaming TTS keeps its own vector because its fan-out
    // semantics differ.
    template <typename Fn>
    void for_each_single_output_leg(Fn&& fn) {
      fn(image_leg);
      fn(video_leg);
    }
    template <typename Fn>
    void for_each_single_output_leg(Fn&& fn) const {
      fn(image_leg);
      fn(video_leg);
    }
  };

  void handle_stream(const StreamMessage& msg);
  void handle_complete(const RequestInfo& info);
  void drain_dispatch();
  void start_vlm_locked(const std::string& request_id, Record& record);
  // Optional output legs, submitted together at the single VLM barrier.
  void start_outputs_locked(const std::string& request_id, Record& record);
  // Submits one single-shot optional output leg (image or video) with unified
  // child_id registration, timing, state transitions, and rollback on throw.
  void submit_single_output_leg_locked(const std::string& request_id,
                                       Record& record, OutputLeg& leg,
                                       const char* suffix, Value inputs);
  // Resolves a parsed directive to "image"/"video" for this request, or "" when
  // no visual stage can serve it.
  std::string slot_kind_locked(const Record& record,
                               const std::string& directive_kind) const;
  // Emits kVisualSlot and submits one <id>:slot:<index> child to the image or
  // video stage. Silent once max_slots is reached.
  void dispatch_narrative_slot_locked(const std::string& request_id,
                                      Record& record,
                                      std::string kind,
                                      const std::string& prompt,
                                      size_t prose_offset,
                                      const std::string& source_slot = {},
                                      const std::string& reference_error = {});
  // Binds a previously selected image anchor to a render request. Explicit
  // references use binary bytes; implicit mainline links keep their data URL.
  bool bind_chained_anchor_locked(Record& record, OutputLeg& leg,
                                  Value::Map& inputs, std::string& error) const;
  // Releases any deferred i2v_chain video slots waiting on `anchor_slot_index`.
  // Attaches `anchor_image_data_url` when non-empty, falls back to T2V when
  // empty, or aborts them cleanly if the request was cancelled.
  void release_chained_slots_locked(const std::string& request_id,
                                    Record& record, int anchor_slot_index,
                                    const std::string& anchor_image_data_url);
  // Submits one TTS leg. `first` keeps the contract child id <id>:tts for the
  // single-leg (non-streaming) request; streamed sentences use :tts:0, :tts:1.
  void start_tts_locked(const std::string& request_id, Record& record,
                        std::string text, bool first, size_t text_begin,
                        size_t text_end);
  // Cuts every complete sentence out of sentence_buf and submits it.
  void drain_sentences_locked(const std::string& request_id, Record& record);
  // Submits the tail of sentence_buf; called at the VLM terminal.
  void flush_sentences_locked(const std::string& request_id, Record& record);
  // Turns one streamed TTS chunk into a PCM16 artifact event. Appends any
  // active TTS child ids to `deferred_aborts` if the byte limit is exceeded.
  void emit_tts_chunk_locked(
      const std::string& request_id, Record& record,
      const std::string& child_id, const Value& chunk,
      std::vector<std::pair<std::string, const char*>>& deferred_aborts);
  // Collects active TTS child ids so the caller can abort them outside mu_
  // without self-deadlocking on Coordinator::abort -> on_complete -> mu_.
  static void collect_tts_aborts_locked(
      const Record& record, const char* reason,
      std::vector<std::pair<std::string, const char*>>& out);
  // Collects every active output child id (image, video, tts) for aborting
  // outside mu_.
  static void collect_output_aborts_locked(
      const Record& record, const char* reason,
      std::vector<std::pair<std::string, const char*>>& out);
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
  std::vector<LegSnapshot> legs_snapshot_locked(const Record& record) const;
  void trace_legs_locked(const std::string& request_id, const char* phase,
                         const Record& record) const;
  void record_session_turn_locked(const Record& record);
  bool record_realtime_history_locked(Record& record,
                                      size_t acknowledged_text_cursor,
                                      std::string& assistant_prefix);
  UnifiedEvent finish_locked(const std::string& request_id, Record& record,
                             UnifiedEvent::Kind kind, std::string error,
                             Value result);
  std::map<std::string, Record>::iterator find_record_locked(
      const std::string& event_request_id);

  Coordinator& coordinator_;
  UnifiedConfig config_;
  std::shared_ptr<CallbackGate> callback_gate_;

  mutable std::mutex mu_;
  std::condition_variable cv_;
  // FIFO of active unified ids, bounded by config_.max_in_flight.
  std::deque<std::string> admission_;
  std::map<std::string, Record> records_;
  std::unordered_map<std::string, std::string> child_to_unified_;
  using DispatchItem = std::variant<UnifiedEvent, StreamMessage, RequestInfo>;
  std::deque<DispatchItem> dispatch_queue_;
  bool dispatching_ = false;  // guarded by mu_; reentrant callbacks only queue
  std::atomic<uint64_t> session_counter_{0};
  SessionHistoryStore session_store_;
};

}  // namespace umminfer
