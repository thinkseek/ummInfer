// SPDX-License-Identifier: Apache-2.0
// Wire-shape adapters shared by the three native stage kinds. Everything here
// is engine-free and host-testable.
//
// The shapes follow the existing external-stage contract:
//   ASR  inputs {"audio": <PCM16 Bin>}                        -> {"text", ...}
//   VLM  inputs {"messages": [...], "modalities": [...]}      -> {"content", ...}
//   TTS  inputs {"text", "voice", "format", "sample_rate"}    -> artifact map
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "umminfer/proto/wire.hpp"

namespace umminfer::stage {

long long epoch_millis();

// Wraps the scheduler's abort check and remembers when it first fired, so the
// stage log can state the abort-to-exit latency.
class AbortTracker {
 public:
  explicit AbortTracker(std::function<bool()> check) : check_(std::move(check)) {}
  bool operator()() {
    if (fired_) return true;
    if (!check_ || !check_()) return false;
    fired_ = true;
    detected_epoch_ms_ = epoch_millis();
    return true;
  }
  bool fired() const { return fired_; }
  long long detected_epoch_ms() const { return detected_epoch_ms_; }

 private:
  std::function<bool()> check_;
  bool fired_ = false;
  long long detected_epoch_ms_ = 0;
};

// Stage hands the compute function the payload data for a chained stage, or
// the request inputs for an entry stage. Accept an "inputs" wrapper too.
const Value& unwrap_inputs(const Value& work);

struct AsrWork {
  std::vector<float> samples;  // mono, 16 kHz, [-1, 1]
};

// Accepts raw PCM16 (what the HTTP boundary sends) or a 16 kHz mono PCM16 WAV
// (what a client may hand to the stage directly).
AsrWork parse_asr_work(const Value& work);

struct OmniWork {
  std::string text;
  std::string image_url;  // empty when the turn has no image
};

// The last user turn wins. Content is either a string or a part array; at most
// one image_url part is accepted.
OmniWork parse_omni_work(const Value& work);

// Same message shape for the qwen38 stage, with the fields that engine cannot
// consume rejected instead of ignored: audio/video parts, an audio/video
// modality claim, and more than one image.
OmniWork parse_qwen38_work(const Value& work);

// Chat turn exactly as native/omni_cli renders it. The vision block is two
// adjacent special tokens; the caller expands the image pads between them.
std::string render_omni_chat(const std::string& text, bool has_image);

struct TtsWork {
  std::string text;
  std::string voice;  // lowercased request value, possibly unsupported
  int sample_rate = 24000;
};

TtsWork parse_tts_work(const Value& work);

struct ImageWork {
  std::string prompt;
  std::string negative_prompt = " ";
  std::string output_format = "png";
  uint64_t seed = 0;
  int steps = 0;  // zero selects the stage default
  float guidance = 4.0f;
  int width = 512;
  int height = 512;
};

ImageWork parse_image_work(const Value& work);

// Selects the codec language id from the text script.
bool text_is_mostly_cjk(const std::string& text);

// data: URLs only. Returns the decoded bytes; throws on any other scheme.
std::string decode_data_url(const std::string& url);
std::string base64_decode(const std::string& text);
// data: URL or a local file path.
std::string load_image_bytes(const std::string& url);

Value asr_result(const std::string& text, int generated_tokens,
                 const std::string& finish_reason);
Value omni_result(const std::string& text, int generated_tokens);
// Artifact map the unified engine expects: media_type + Bin under "data".
Value tts_result(std::string wav_bytes, int sample_rate, int channels,
                 const std::string& voice);
Value image_result(std::string png_bytes, int width, int height,
                   uint64_t seed, int steps);

}  // namespace umminfer::stage
