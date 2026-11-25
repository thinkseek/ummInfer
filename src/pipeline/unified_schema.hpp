// SPDX-License-Identifier: Apache-2.0
// Stateless contract validation, prompt synthesis, and artifact slicing for
// UnifiedEngine. Kept separate from UnifiedEngine's lock/callback state
// machine so contract shapes and orchestration lifetimes evolve independently.
#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "umminfer/pipeline/unified.hpp"

namespace umminfer::unified_detail {

struct ValidatedSubmission {
  Value original_inputs;
  Value asr_inputs;
  bool has_audio = false;
  bool asr_partials = false;
  bool history_requested = false;
  // Internal realtime-only opt-in. Playback acknowledgement, rather than
  // server emission, decides when this turn may enter session history.
  bool realtime_playback_ack = false;
  bool explicit_session = false;
  std::string session_id;
  bool image_output = false;
  bool audio_output = false;
  bool video_output = false;
  Value::Map image_options;
  Value::Map audio_options;
  Value::Map video_options;
  // Interleaved narrative. Parsed from the optional top-level "narrative"
  // object; defaults keep every non-narrative request byte-identical.
  bool narrative_enabled = false;
  std::string narrative_visual_mode = "auto";
  int narrative_max_slots = 3;
  std::string user_text;
};

UnifiedModality modality_of_part_type(const std::string& type);
std::string_view chunk_text(const Value& chunk);
bool chunk_is_final(const Value& chunk);
std::optional<std::string> asr_result_text(const Value& result);
std::optional<std::string> vlm_result_text(const Value& result);
std::string user_text_from_inputs(const Value& original_inputs);

ValidatedSubmission validate_submission(const Value& inputs,
                                        const UnifiedConfig& config);

Value build_vlm_inputs(const std::string& transcript,
                       const Value& original_inputs);
// Director system prompt prepended to the VLM messages when narrative is
// enabled. visual_mode is "auto", "image" or "video".
std::string narrative_director_prompt(const std::string& visual_mode,
                                      int max_slots);
// Returns `result` with its assistant text replaced by the clean prose. Used
// for narrative terminals so raw [[...]] markers never reach the wire.
Value replace_result_text(const Value& result, const std::string& text);
// Prepends the director system message to inputs["messages"]. No-op when the
// messages array is absent or malformed.
void prepend_narrative_instructions(Value& vlm_inputs,
                                    const std::string& visual_mode,
                                    int max_slots);
Value build_image_inputs(const Value::Map& image_options,
                         const std::string& vlm_text);
Value build_video_inputs(const Value::Map& video_options,
                         const std::string& vlm_text);
Value build_tts_inputs(const Value::Map& audio_options,
                       const std::string& text, bool tts_stream);

// Validates a stage artifact map against per-leg and aggregate byte caps,
// updates `artifact_bytes_total` and `next_artifact`, and returns the gap-free
// kArtifact events starting at `next_seq` (advancing `next_seq`).
std::vector<UnifiedEvent> slice_artifact_events(
    const std::string& request_id, const std::string& session_id,
    const std::string& stage_id, const Value& result,
    size_t& artifact_bytes_total, int& next_artifact, int64_t& next_seq);

}  // namespace umminfer::unified_detail
