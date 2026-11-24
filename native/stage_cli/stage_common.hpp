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
#include <utility>
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

struct OmniMessage {
  std::string role;
  std::string text;
  bool has_image = false;
};

struct OmniWork {
  std::string text;       // last text-bearing message
  std::string image_url;  // empty when the turn has no image
  // Every message in wire order, for the multi-turn prompt. The omni stage
  // renders all of them so session history is part of the prompt.
  std::vector<OmniMessage> messages;
  // Opt-in prefix-cache key. Empty disables reuse for this request.
  std::string session_id;
};

// Every message is kept; content is either a string or a part array; at most
// one image_url part is accepted, in the last message.
OmniWork parse_omni_work(const Value& work);

// Same message shape for the qwen38 stage, with the fields that engine cannot
// consume rejected instead of ignored: audio/video parts, an audio/video
// modality claim, and more than one image.
OmniWork parse_qwen38_work(const Value& work);

// Resolves per-request `max_new_tokens` from Stage request params, validating
// positivity and clamping to `max_seq` when provided.
int resolve_max_new_tokens(const Value::Map& params, int default_tokens,
                           int max_seq);

// Sampling knobs for one request. The defaults select argmax, so a request
// without sampling fields stays bit-identical to the greedy release.
struct SamplingParams {
  float temperature = 0.0f;
  float top_p = 1.0f;
  uint64_t seed = 0;
  bool has_seed = false;
};

// Reads temperature / top_p / seed from Stage request params. An absent or nil
// field keeps its default. seed is any signed int64, including negative
// values; it maps to uint64 by two's-complement conversion, so non-negative
// seeds keep their exact value and negative seeds stay distinct from every
// non-negative one.
SamplingParams parse_sampling_params(const Value::Map& params);

// Chat turn exactly as native/omni_cli renders it. The vision block is two
// adjacent special tokens; the caller expands the image pads between them.
std::string render_omni_chat(const std::string& text, bool has_image);
// Multi-turn rendering: one <|im_start|>role block per message and a final
// assistant header. With one message it is byte-identical to the call above.
std::string render_omni_chat(const std::vector<OmniMessage>& messages,
                             bool has_image);

// Longest prefix of `text` that ends on a complete UTF-8 sequence. A token
// decode can end mid-character; stream deltas must not, or the JSON layer
// rejects the chunk.
inline size_t utf8_complete_prefix(const std::string& text) {
  size_t index = text.size();
  size_t continue_bytes = 0;
  while (index > 0 && continue_bytes < 4) {
    const unsigned char byte = static_cast<unsigned char>(text[index - 1]);
    if ((byte & 0xC0) == 0x80) {
      ++continue_bytes;
      --index;
      continue;
    }
    size_t need = 1;
    if ((byte & 0xE0) == 0xC0) need = 2;
    else if ((byte & 0xF0) == 0xE0) need = 3;
    else if ((byte & 0xF8) == 0xF0) need = 4;
    return (continue_bytes + 1 == need) ? text.size() : index - 1;
  }
  return continue_bytes < 4 ? text.size() : text.size() - continue_bytes;
}

struct TtsWork {
  std::string text;
  std::string voice;  // lowercased request value, possibly unsupported
  int sample_rate = 24000;
  // Emit PCM16 chunks over the stage stream while the codec decodes, instead
  // of one WAV artifact at the end. The terminal result still carries the WAV.
  bool stream = false;
};

TtsWork parse_tts_work(const Value& work);

enum class ImageBackend { kQwen2512, kQwen21 };

// Selects the native image backend. Empty or "2512" / "qwen-image-2512" keeps
// the default Qwen-Image-2512 executor; "2.1" / "qwen-image-2.1" selects
// Qwen-Image-2.1. Any other value fails loudly.
ImageBackend parse_image_backend(const std::string& value);
const char* image_backend_name(ImageBackend backend);

// Validates a WIDTHxHEIGHT string against the validated serving size (512x512).
// Never rounds or falls back to a different resolution.
std::pair<int, int> parse_supported_image_size(const std::string& size);

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

struct VideoWork {
  std::string prompt;
  std::string negative_prompt;
  std::string output_format = "mp4";
  int width = 480;
  int height = 320;
  int seconds = 1;
  int fps = 16;
  int num_frames = 13;
  int steps = 20;
  float guidance_scale = 5.0f;
  uint64_t seed = 0;
  bool has_seed = false;
};

// Parses UnifiedEngine video leg inputs ({prompt, size?, seconds?, fps?,
// num_frames?, steps?, guidance_scale?, seed?, negative_prompt?,
// output_format?}) with Wan spatial alignment and temporal stride rules.
VideoWork parse_video_work(const Value& work, int default_width = 480,
                           int default_height = 320, int spatial_align = 32,
                           int default_seconds = 1, int max_seconds = 10,
                           int default_fps = 16, int max_frames = 81,
                           int temporal_stride = 4, int default_steps = 20,
                           int max_steps = 100);

// Selects the codec language id from the text script.
bool text_is_mostly_cjk(const std::string& text);

// data: URLs only (capped at 48 MiB decoded). Returns the decoded bytes;
// throws on any other scheme or local file path.
std::string decode_data_url(const std::string& url);
std::string base64_decode(const std::string& text);
std::string load_image_bytes(const std::string& url);

Value asr_result(const std::string& text, int generated_tokens,
                 const std::string& finish_reason);
// Optional phase times are milliseconds the stage measured itself. A negative
// value omits the field, so the server records nothing instead of a zero.
Value omni_result(const std::string& text, int generated_tokens,
                  double prefill_ms = -1, double decode_ms = -1,
                  const std::string& finish_reason = "stop");
// Artifact map the unified engine expects: media_type + Bin under "data".
// When streamed is true, marks "streamed": true so UnifiedEngine skips
// re-emitting the full WAV artifact after PCM16 chunks have been streamed.
Value tts_result(std::string wav_bytes, int sample_rate, int channels,
                 const std::string& voice, double first_frame_ms = -1,
                 double total_ms = -1, bool streamed = false);
Value image_result(std::string png_bytes, int width, int height,
                   uint64_t seed, int steps, double denoise_ms = -1);
Value video_result(std::string mp4_bytes, int width, int height, int fps,
                   int frames, double seconds, uint64_t seed, int steps,
                   const std::string& model, const std::string& revision,
                   const std::string& backend, double denoise_ms = -1,
                   double encode_ms = -1);

}  // namespace umminfer::stage
