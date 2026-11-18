// SPDX-License-Identifier: Apache-2.0
#include "stage_common.hpp"

#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string_view>

#include "umm_asr_native/wav.hpp"
#include "umminfer/util/codec.hpp"

namespace umminfer::stage {
namespace {

[[noreturn]] void bad(const std::string& message) {
  throw ProtocolError(message);
}

std::string lowercase(std::string text) {
  for (char& ch : text) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return text;
}

bool starts_with(std::string_view text, std::string_view prefix) {
  return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

// Joins the text parts of one content array; returns false when it has none.
bool join_text_parts(const Value& content, std::string& text) {
  bool found = false;
  std::string joined;
  for (const Value& part : content.as_array()) {
    if (!part.is_map()) bad("content part must be a map");
    const Value* type = part.find("type");
    if (type == nullptr || !type->is_str()) bad("content part requires a type");
    if (type->as_str() != "text") continue;
    const Value* value = part.find("text");
    if (value == nullptr || !value->is_str() || value->as_str().empty()) {
      bad("text part requires a nonempty text string");
    }
    if (!joined.empty()) joined += "\n";
    joined += value->as_str();
    found = true;
  }
  text = std::move(joined);
  return found;
}

void take_image_part(const Value& part, std::string_view stage_name,
                     OmniWork& work) {
  const Value* image = part.find("image_url");
  if (image == nullptr || !image->is_map()) {
    bad("image_url part requires an image_url map");
  }
  const Value* url = image->find("url");
  if (url == nullptr || !url->is_str() || url->as_str().empty()) {
    bad("image_url part requires a nonempty url");
  }
  if (!work.image_url.empty()) {
    bad(std::string(stage_name) + " stage accepts at most one image");
  }
  work.image_url = url->as_str();
}

OmniWork parse_vlm_work(const Value& work, std::string_view stage_name,
                        bool text_and_image_only) {
  const Value& inputs = unwrap_inputs(work);
  if (text_and_image_only) {
    if (inputs.find("audio") != nullptr) {
      bad(std::string(stage_name) +
          " stage does not accept audio input; audio goes through the ASR "
          "stage and reaches the VLM as text");
    }
    if (const Value* declared = inputs.find("modalities"); declared != nullptr) {
      if (!declared->is_array()) {
        bad(std::string(stage_name) + " modalities must be an array");
      }
      for (const Value& token : declared->as_array()) {
        if (!token.is_str()) {
          bad(std::string(stage_name) + " modality must be a string");
        }
        const std::string& name = token.as_str();
        if (name == "audio" || name == "video") {
          bad(std::string(stage_name) + " stage does not accept modality '" +
              name + "': text and image only; audio goes through the ASR stage");
        }
        if (name != "text" && name != "image") {
          bad("unknown modality for " + std::string(stage_name) + " stage: " +
              name);
        }
      }
    }
  }
  const Value* messages = inputs.find("messages");
  if (messages == nullptr || !messages->is_array() || messages->as_array().empty()) {
    bad(std::string(stage_name) + " inputs require a nonempty messages array");
  }
  OmniWork out;
  if (const Value* session = inputs.find("session_id");
      session != nullptr && session->is_str()) {
    out.session_id = session->as_str();
  }
  bool have_text = false;
  for (const Value& message : messages->as_array()) {
    if (!message.is_map()) bad("each message must be a map");
    OmniMessage turn;
    if (const Value* role = message.find("role");
        role != nullptr && role->is_str() && !role->as_str().empty()) {
      turn.role = role->as_str();
    } else {
      turn.role = "user";
    }
    const Value* content = message.find("content");
    if (content == nullptr) continue;
    if (content->is_str()) {
      turn.text = content->as_str();
      if (!content->as_str().empty()) {
        out.text = content->as_str();
        have_text = true;
      }
      out.messages.push_back(std::move(turn));
      continue;
    }
    if (!content->is_array()) {
      bad("message content must be a text string or a content part array");
    }
    if (text_and_image_only) {
      for (const Value& part : content->as_array()) {
        if (!part.is_map()) bad("content part must be a map");
        const Value* type = part.find("type");
        if (type == nullptr || !type->is_str()) bad("content part requires a type");
        const std::string& kind = type->as_str();
        if (kind == "text" || kind == "image_url") continue;
        bad(std::string(stage_name) + " stage does not support content part type '" +
            kind + "': text and image_url only; audio goes through the ASR stage");
      }
    }
    std::string joined;
    const bool has_text = join_text_parts(*content, joined);
    for (const Value& part : content->as_array()) {
      const Value* type = part.find("type");
      if (type->as_str() == "image_url") {
        take_image_part(part, stage_name, out);
        turn.has_image = true;
      }
    }
    if (has_text) {
      out.text = joined;
      have_text = true;
      turn.text = std::move(joined);
    }
    out.messages.push_back(std::move(turn));
  }
  if (!have_text && out.image_url.empty()) {
    bad(std::string(stage_name) + " inputs require text or an image");
  }
  return out;
}

}  // namespace

long long epoch_millis() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

const Value& unwrap_inputs(const Value& work) {
  if (const Value* inputs = work.find("inputs"); inputs != nullptr) {
    if (!inputs->is_map()) bad("inputs must be a map");
    return *inputs;
  }
  if (!work.is_map()) bad("stage work must be a map");
  return work;
}

AsrWork parse_asr_work(const Value& work) {
  const Value& inputs = unwrap_inputs(work);
  const Value* audio = inputs.find("audio");
  if (audio == nullptr || (!audio->is_bin() && !audio->is_str())) {
    bad("ASR inputs require an audio byte string");
  }
  const std::string& bytes = audio->as_bin();
  if (bytes.empty()) bad("ASR audio must be nonempty");
  if (bytes.size() % 2 != 0) bad("ASR audio must contain whole PCM16 samples");

  AsrWork out;
  if (starts_with(bytes, "RIFF")) {
    // A WAV container already carries the sample rate; the accepted parser
    // rejects anything that is not 16 kHz mono PCM16.
    native::PcmAudio parsed = native::load_wav_pcm16(bytes);
    out.samples = std::move(parsed.samples);
    return out;
  }
  out.samples.resize(bytes.size() / 2);
  constexpr float kScale = 1.0f / 32768.0f;
  for (size_t index = 0; index < out.samples.size(); ++index) {
    const auto low = static_cast<uint16_t>(static_cast<unsigned char>(bytes[index * 2]));
    const auto high =
        static_cast<uint16_t>(static_cast<unsigned char>(bytes[index * 2 + 1]));
    const auto sample = static_cast<int16_t>(low | static_cast<uint16_t>(high << 8));
    out.samples[index] = static_cast<float>(sample) * kScale;
  }
  return out;
}

OmniWork parse_omni_work(const Value& work) {
  return parse_vlm_work(work, "omni", /*text_and_image_only=*/false);
}

OmniWork parse_qwen38_work(const Value& work) {
  return parse_vlm_work(work, "qwen38", /*text_and_image_only=*/true);
}

int resolve_max_new_tokens(const Value::Map& params, int default_tokens,
                           int max_seq) {
  const auto it = params.find("max_new_tokens");
  if (it == params.end() || it->second.is_nil()) return default_tokens;
  if (!it->second.is_int() || it->second.as_int() <= 0) {
    bad("max_new_tokens must be a positive integer");
  }
  return static_cast<int>(std::min<int64_t>(it->second.as_int(), max_seq));
}

std::string render_omni_chat(const std::string& text, bool has_image) {
  std::string rendered = "<|im_start|>user\n";
  if (has_image) rendered += "<|vision_start|><|vision_end|>";
  rendered += text;
  rendered += "<|im_end|>\n<|im_start|>assistant\n";
  return rendered;
}

std::string render_omni_chat(const std::vector<OmniMessage>& messages,
                             bool has_image) {
  bool any_marked = false;
  for (const auto& m : messages) {
    if (m.has_image) any_marked = true;
  }
  std::string rendered;
  for (size_t i = 0; i < messages.size(); ++i) {
    rendered += "<|im_start|>" + messages[i].role + "\n";
    if (has_image &&
        (any_marked ? messages[i].has_image : (i + 1 == messages.size()))) {
      rendered += "<|vision_start|><|vision_end|>";
    }
    rendered += messages[i].text;
    rendered += "<|im_end|>\n";
  }
  rendered += "<|im_start|>assistant\n";
  return rendered;
}

TtsWork parse_tts_work(const Value& work) {
  const Value& inputs = unwrap_inputs(work);
  TtsWork out;
  const Value* text = inputs.find("text");
  if (text == nullptr || !text->is_str() || text->as_str().empty()) {
    bad("TTS inputs require a nonempty text");
  }
  out.text = text->as_str();
  if (const Value* voice = inputs.find("voice"); voice != nullptr) {
    if (!voice->is_str()) bad("TTS voice must be a string");
    out.voice = lowercase(voice->as_str());
  }
  if (const Value* format = inputs.find("format"); format != nullptr) {
    if (!format->is_str() || format->as_str() != "wav") {
      bad("TTS format must be wav");
    }
  }
  if (const Value* rate = inputs.find("sample_rate"); rate != nullptr) {
    if (!rate->is_int() || rate->as_int() != 24000) {
      bad("TTS sample_rate must be 24000");
    }
    out.sample_rate = 24000;
  }
  if (const Value* stream = inputs.find("stream"); stream != nullptr) {
    if (!stream->is_bool()) bad("TTS stream must be a boolean");
    out.stream = stream->as_bool();
  }
  return out;
}

ImageBackend parse_image_backend(const std::string& value) {
  const std::string norm = lowercase(value);
  if (norm.empty() || norm == "2512" || norm == "qwen-image-2512") {
    return ImageBackend::kQwen2512;
  }
  if (norm == "2.1" || norm == "qwen-image-2.1") {
    return ImageBackend::kQwen21;
  }
  bad("unsupported --image-backend '" + value + "': expected 2512 or 2.1");
}

const char* image_backend_name(ImageBackend backend) {
  switch (backend) {
    case ImageBackend::kQwen2512:
      return "2512";
    case ImageBackend::kQwen21:
      return "2.1";
  }
  return "2512";
}

std::pair<int, int> parse_supported_image_size(const std::string& text) {
  const size_t split = text.find('x');
  if (split == std::string::npos || split == 0 || split + 1 == text.size() ||
      text.find('x', split + 1) != std::string::npos) {
    bad("image size must be WIDTHxHEIGHT");
  }
  int width = 0;
  int height = 0;
  try {
    const std::string w_str = text.substr(0, split);
    const std::string h_str = text.substr(split + 1);
    size_t w_used = 0;
    size_t h_used = 0;
    width = std::stoi(w_str, &w_used);
    height = std::stoi(h_str, &h_used);
    if (w_used != w_str.size() || h_used != h_str.size()) {
      bad("image size must be WIDTHxHEIGHT");
    }
  } catch (const ProtocolError&) {
    throw;
  } catch (const std::exception&) {
    bad("image size must be WIDTHxHEIGHT");
  }
  if (width != 512 || height != 512) {
    bad("native image currently supports only 512x512, got " + text);
  }
  return {width, height};
}

ImageWork parse_image_work(const Value& work) {
  const Value& inputs = unwrap_inputs(work);
  ImageWork out;
  const Value* prompt = inputs.find("prompt");
  if (prompt == nullptr || !prompt->is_str() || prompt->as_str().empty()) {
    bad("image inputs require a nonempty prompt");
  }
  out.prompt = prompt->as_str();
  if (const Value* negative = inputs.find("negative_prompt"); negative != nullptr) {
    if (!negative->is_str()) bad("image negative_prompt must be a string");
    out.negative_prompt = negative->as_str();
  }
  if (const Value* format = inputs.find("output_format"); format != nullptr) {
    if (!format->is_str() || format->as_str() != "png") {
      bad("native image output_format must be png");
    }
    out.output_format = "png";
  }
  if (const Value* seed = inputs.find("seed"); seed != nullptr) {
    if (!seed->is_int() || seed->as_int() < 0) {
      bad("image seed must be a non-negative integer");
    }
    out.seed = static_cast<uint64_t>(seed->as_int());
  }
  if (const Value* steps = inputs.find("steps"); steps != nullptr) {
    if (!steps->is_int() || steps->as_int() < 0 || steps->as_int() > 200) {
      bad("image steps must be between 0 and 200");
    }
    out.steps = static_cast<int>(steps->as_int());
  }
  if (const Value* guidance = inputs.find("guidance"); guidance != nullptr) {
    if ((!guidance->is_float() && !guidance->is_int()) ||
        !std::isfinite(guidance->as_double()) || guidance->as_double() <= 0.0) {
      bad("image guidance must be a finite positive number");
    }
    out.guidance = static_cast<float>(guidance->as_double());
  }
  if (const Value* size = inputs.find("size"); size != nullptr) {
    if (!size->is_str()) bad("image size must be a string");
    const auto [width, height] = parse_supported_image_size(size->as_str());
    out.width = width;
    out.height = height;
  }
  return out;
}

bool text_is_mostly_cjk(const std::string& text) {
  size_t cjk = 0;
  size_t latin = 0;
  for (size_t index = 0; index < text.size();) {
    const unsigned char lead = static_cast<unsigned char>(text[index]);
    size_t length = 1;
    uint32_t code = lead;
    if (lead >= 0xF0) length = 4;
    else if (lead >= 0xE0) length = 3;
    else if (lead >= 0xC0) length = 2;
    if (index + length > text.size()) break;
    if (length > 1) {
      code = lead & (0xFFu >> (length + 1));
      for (size_t offset = 1; offset < length; ++offset) {
        code = (code << 6) |
               (static_cast<unsigned char>(text[index + offset]) & 0x3Fu);
      }
    } else if ((lead >= 'a' && lead <= 'z') || (lead >= 'A' && lead <= 'Z')) {
      ++latin;
    }
    if (code >= 0x2E80 && code <= 0x9FFF) ++cjk;
    index += length;
  }
  return cjk > 0 && cjk >= latin;
}

namespace {
constexpr size_t kMaxImageBytes = 48u << 20;
}  // namespace

std::string base64_decode(const std::string& text) {
  return util::base64_decode(text, kMaxImageBytes, /*require_nonempty=*/true);
}

std::string decode_data_url(const std::string& url) {
  if (!starts_with(url, "data:")) bad("image url must be a data: URL");
  const size_t comma = url.find(',');
  if (comma == std::string::npos) bad("data URL has no payload separator");
  const std::string header = url.substr(5, comma - 5);
  if (header.find(";base64") == std::string::npos) {
    bad("data URL must carry a base64 payload");
  }
  return base64_decode(std::string(
      std::string_view(url).substr(comma + 1)));
}

std::string load_image_bytes(const std::string& url) {
  if (!starts_with(url, "data:")) {
    bad("image url must be a base64 data: URL; http(s) and local file paths are rejected");
  }
  return decode_data_url(url);
}

Value asr_result(const std::string& text, int generated_tokens,
                 const std::string& finish_reason) {
  Value::Map out;
  out["text"] = Value(text);
  out["generated_token_count"] = Value(static_cast<int64_t>(generated_tokens));
  out["finish_reason"] = Value(finish_reason);
  return Value(std::move(out));
}

Value omni_result(const std::string& text, int generated_tokens,
                  double prefill_ms, double decode_ms,
                  const std::string& finish_reason) {
  if (finish_reason.empty()) bad("VLM finish_reason must not be empty");
  Value::Map out;
  out["content"] = Value(text);
  out["generated_token_count"] = Value(static_cast<int64_t>(generated_tokens));
  out["finish_reason"] = Value(finish_reason);
  if (prefill_ms >= 0) out["prefill_ms"] = Value(prefill_ms);
  if (decode_ms >= 0) out["decode_ms"] = Value(decode_ms);
  return Value(std::move(out));
}

Value tts_result(std::string wav_bytes, int sample_rate, int channels,
                 const std::string& voice, double first_frame_ms,
                 double total_ms, bool streamed) {
  if (wav_bytes.empty()) bad("TTS produced no audio");
  Value::Map out;
  out["media_type"] = Value("audio");
  out["mime_type"] = Value("audio/wav");
  out["format"] = Value("wav");
  out["sample_rate"] = Value(static_cast<int64_t>(sample_rate));
  out["channels"] = Value(static_cast<int64_t>(channels));
  out["voice"] = Value(voice);
  if (streamed) out["streamed"] = Value(true);
  if (first_frame_ms >= 0) out["first_frame_ms"] = Value(first_frame_ms);
  if (total_ms >= 0) out["total_ms"] = Value(total_ms);
  out["data"] = Value::bin(std::move(wav_bytes));
  return Value(std::move(out));
}

Value image_result(std::string png_bytes, int width, int height,
                   uint64_t seed, int steps, double denoise_ms) {
  if (png_bytes.empty()) bad("image produced no PNG bytes");
  Value::Map out;
  out["media_type"] = Value("image");
  out["mime_type"] = Value("image/png");
  out["format"] = Value("png");
  out["size"] = Value(std::to_string(width) + "x" + std::to_string(height));
  out["width"] = Value(static_cast<int64_t>(width));
  out["height"] = Value(static_cast<int64_t>(height));
  out["seed"] = Value(static_cast<int64_t>(seed));
  out["steps"] = Value(static_cast<int64_t>(steps));
  if (denoise_ms >= 0) out["denoise_ms"] = Value(denoise_ms);
  out["data"] = Value::bin(std::move(png_bytes));
  return Value(std::move(out));
}

}  // namespace umminfer::stage
