// SPDX-License-Identifier: Apache-2.0
// Host checks for the engine-free stage adapters: image backend selection,
// TTS request parsing, CJK language detection, and the TTS artifact descriptor.
#include "stage_common.hpp"

#include <cmath>
#include <cstdio>
#include <string>

namespace {

int g_failures = 0;

void check(bool cond, const char* what) {
  if (!cond) {
    std::printf("FAIL: %s\n", what);
    ++g_failures;
  }
}

template <typename Fn>
bool throws_protocol(Fn fn) {
  try {
    fn();
  } catch (const umminfer::ProtocolError&) {
    return true;
  }
  return false;
}

void test_parse_image_backend() {
  using umminfer::stage::ImageBackend;
  using umminfer::stage::image_backend_name;
  using umminfer::stage::parse_image_backend;
  check(parse_image_backend("") == ImageBackend::kQwen2512, "empty backend -> 2512");
  check(parse_image_backend("2512") == ImageBackend::kQwen2512, "2512 -> 2512");
  check(parse_image_backend("qwen-image-2512") == ImageBackend::kQwen2512,
        "qwen-image-2512 -> 2512");
  check(parse_image_backend("2.1") == ImageBackend::kQwen21, "2.1 -> 2.1");
  check(parse_image_backend("Qwen-Image-2.1") == ImageBackend::kQwen21,
        "backend is case-insensitive");
  check(throws_protocol([] { (void)parse_image_backend("1.5"); }),
        "unknown backend rejected");
  check(throws_protocol([] { (void)parse_image_backend("2512x"); }),
        "suffixed backend rejected");
  check(std::string(image_backend_name(ImageBackend::kQwen2512)) == "2512",
        "2512 name");
  check(std::string(image_backend_name(ImageBackend::kQwen21)) == "2.1",
        "2.1 name");
}

void test_text_is_mostly_cjk() {
  using umminfer::stage::text_is_mostly_cjk;
  check(text_is_mostly_cjk("你好，世界"), "Chinese text is CJK");
  check(!text_is_mostly_cjk("hello world"), "English text is not CJK");
  check(!text_is_mostly_cjk(""), "empty text is not CJK");
  check(!text_is_mostly_cjk("😀😀"), "emoji-only text is not CJK");
  check(text_is_mostly_cjk("你好世界 hi"), "CJK majority wins a mix");
  check(!text_is_mostly_cjk("hello 世"), "latin majority wins a mix");
  check(text_is_mostly_cjk("日本語のテキスト"), "kana/kanji range counted");
}

void test_tts_work() {
  using umminfer::Value;
  using umminfer::stage::parse_tts_work;
  Value::Map inputs;
  inputs["text"] = Value("hello");
  const auto plain = parse_tts_work(Value(inputs));
  check(plain.text == "hello", "tts text parsed");
  check(plain.voice.empty(), "tts voice empty by default");
  check(plain.sample_rate == 24000, "tts default sample rate");
  check(!plain.stream, "tts stream off by default");

  inputs["voice"] = Value("RYAN");
  inputs["format"] = Value("wav");
  inputs["sample_rate"] = Value(static_cast<int64_t>(24000));
  inputs["stream"] = Value(true);
  const auto full = parse_tts_work(Value(inputs));
  check(full.voice == "ryan", "tts voice lowercased");
  check(full.stream, "tts stream flag parsed");

  Value::Map no_text;
  check(throws_protocol([&] { (void)parse_tts_work(Value(no_text)); }),
        "tts rejects missing text");
  Value::Map bad_format(inputs);
  bad_format["format"] = Value("mp3");
  check(throws_protocol([&] { (void)parse_tts_work(Value(bad_format)); }),
        "tts rejects non-wav format");
  Value::Map bad_rate(inputs);
  bad_rate["sample_rate"] = Value(static_cast<int64_t>(16000));
  check(throws_protocol([&] { (void)parse_tts_work(Value(bad_rate)); }),
        "tts rejects non-24000 rate");
  Value::Map bad_stream(inputs);
  bad_stream["stream"] = Value("yes");
  check(throws_protocol([&] { (void)parse_tts_work(Value(bad_stream)); }),
        "tts rejects non-boolean stream");
}

void test_tts_result() {
  using umminfer::stage::tts_result;
  const std::string wav(44, '\x7f');
  const umminfer::Value out =
      tts_result(wav, 24000, 1, "vivian", 120.5, 900.0, true);
  check(out.is_map(), "tts_result is a map");
  check(out.find("media_type")->as_str() == "audio", "tts media_type audio");
  check(out.find("mime_type")->as_str() == "audio/wav", "tts mime audio/wav");
  check(out.find("format")->as_str() == "wav", "tts format wav");
  check(out.find("sample_rate")->as_int() == 24000, "tts sample_rate");
  check(out.find("channels")->as_int() == 1, "tts channels mono");
  check(out.find("voice")->as_str() == "vivian", "tts voice recorded");
  check(out.find("streamed")->as_bool(), "tts streamed flag set");
  check(out.find("first_frame_ms")->as_double() == 120.5, "tts first_frame_ms");
  check(out.find("total_ms")->as_double() == 900.0, "tts total_ms");
  check(out.find("data")->as_bin() == wav, "tts data carries the WAV bytes");

  const umminfer::Value plain = tts_result(wav, 24000, 1, "vivian");
  check(plain.find("streamed") == nullptr, "tts streamed flag omitted");
  check(plain.find("first_frame_ms") == nullptr, "tts first_frame_ms omitted");
  check(throws_protocol([&] { (void)tts_result("", 24000, 1, "vivian"); }),
        "tts_result rejects empty audio");
}

}  // namespace

int main() {
  test_parse_image_backend();
  test_text_is_mostly_cjk();
  test_tts_work();
  test_tts_result();
  if (g_failures) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("all stage common checks passed\n");
  return 0;
}
