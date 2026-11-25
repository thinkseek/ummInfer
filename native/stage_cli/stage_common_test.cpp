// SPDX-License-Identifier: Apache-2.0
// Host checks for the engine-free stage adapters: image backend selection,
// TTS request parsing, CJK language detection, and the TTS artifact descriptor.
#include "stage_common.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
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

  Value::Map with_bin_ref(inputs);
  with_bin_ref["ref_audio"] = Value::bin("RIFF0000WAVE");
  check(parse_tts_work(Value(with_bin_ref)).ref_audio_wav == "RIFF0000WAVE",
        "tts parses binary ref_audio");

  Value::Map with_url_ref(inputs);
  with_url_ref["ref_audio"] = Value("data:audio/wav;base64,UklGRg==");
  check(parse_tts_work(Value(with_url_ref)).ref_audio_wav == "RIFF",
        "tts parses data-URL ref_audio");

  Value::Map bad_ref(inputs);
  bad_ref["ref_audio"] = Value(static_cast<int64_t>(7));
  check(throws_protocol([&] { (void)parse_tts_work(Value(bad_ref)); }),
        "tts rejects non-binary/non-string ref_audio");

  Value::Map img_edit;
  img_edit["prompt"] = Value("a lantern");
  img_edit["image"] = Value("data:image/png;base64,AAAA");
  check(throws_protocol([&] { (void)umminfer::stage::parse_image_work(Value(img_edit)); }),
        "native Qwen-Image parse_image_work rejects inputs.image editing");

  // Native Qwen3.8 video_url parsing & mutual exclusion with image_url.
  Value::Map vpart;
  vpart["type"] = Value("video_url");
  vpart["video_url"] = Value(Value::Map{{"url", Value("data:video/mp4;base64,AAAA")}});
  Value::Map tpart;
  tpart["type"] = Value("text");
  tpart["text"] = Value("what moves?");
  Value::Map vmsg;
  vmsg["role"] = Value("user");
  vmsg["content"] = Value::Array{Value(vpart), Value(tpart)};
  Value::Map vwork;
  vwork["messages"] = Value::Array{Value(vmsg)};
  const auto parsed_vid = umminfer::stage::parse_qwen38_work(Value(vwork));
  check(parsed_vid.video_url == "data:video/mp4;base64,AAAA" &&
            parsed_vid.messages.size() == 1 && parsed_vid.messages[0].has_video,
        "qwen38 parses video_url content part");
  check(!parsed_vid.frame_selection.temporal &&
            parsed_vid.frame_selection.motion_threshold == 0.02 &&
            parsed_vid.frame_selection.scene_threshold == 0.25,
        "omitted frame_selection keeps the historical uniform defaults");

  Value::Map temporal_part = vpart;
  temporal_part["video_url"] = Value(Value::Map{
      {"url", Value("data:video/mp4;base64,AAAA")},
      {"frame_selection", Value(Value::Map{
                              {"policy", Value("temporal")},
                              {"motion_threshold", Value(0.04)},
                              {"scene_threshold", Value(0.3)},
                          })}});
  Value::Map temporal_msg;
  temporal_msg["role"] = Value("user");
  temporal_msg["content"] = Value::Array{Value(temporal_part), Value(tpart)};
  Value::Map temporal_work;
  temporal_work["messages"] = Value::Array{Value(temporal_msg)};
  const auto parsed_temporal = umminfer::stage::parse_qwen38_work(Value(temporal_work));
  check(parsed_temporal.frame_selection.temporal &&
            parsed_temporal.frame_selection.motion_threshold == 0.04 &&
            parsed_temporal.frame_selection.scene_threshold == 0.3,
        "qwen38 parses explicit temporal selection and thresholds");

  Value::Map uniform_part = vpart;
  uniform_part["video_url"] = Value(Value::Map{
      {"url", Value("data:video/mp4;base64,AAAA")},
      {"frame_selection", Value(Value::Map{{"policy", Value("uniform")}})}});
  Value::Map uniform_msg;
  uniform_msg["role"] = Value("user");
  uniform_msg["content"] = Value::Array{Value(uniform_part), Value(tpart)};
  Value::Map uniform_work;
  uniform_work["messages"] = Value::Array{Value(uniform_msg)};
  check(!umminfer::stage::parse_qwen38_work(Value(uniform_work))
             .frame_selection.temporal,
        "explicit uniform and omitted policy have identical native options");

  auto reject_selection = [&](Value::Map selection) {
    Value::Map bad_part = vpart;
    bad_part["video_url"] = Value(Value::Map{
        {"url", Value("data:video/mp4;base64,AAAA")},
        {"frame_selection", Value(std::move(selection))}});
    Value::Map bad_msg;
    bad_msg["role"] = Value("user");
    bad_msg["content"] = Value::Array{Value(bad_part), Value(tpart)};
    Value::Map bad_work;
    bad_work["messages"] = Value::Array{Value(bad_msg)};
    return throws_protocol([&] {
      (void)umminfer::stage::parse_qwen38_work(Value(bad_work));
    });
  };
  check(reject_selection(Value::Map{{"policy", Value("random")}}),
        "unknown frame selection policy rejected");
  check(reject_selection(Value::Map{{"policy", Value("temporal")},
                                    {"motion_threshold", Value(-0.1)}}),
        "negative motion threshold rejected");
  check(reject_selection(Value::Map{{"policy", Value("temporal")},
                                    {"scene_threshold", Value(1.1)}}),
        "scene threshold above one rejected");
  check(reject_selection(Value::Map{{"policy", Value("uniform")},
                                    {"motion_threshold", Value(0.1)}}),
        "uniform silently ignored thresholds are rejected");
  check(reject_selection(Value::Map{{"policy", Value("temporal")},
                                    {"max_frames", Value(static_cast<int64_t>(100))}}),
        "API cannot raise the native frame budget");

  Value::Map ipart;
  ipart["type"] = Value("image_url");
  ipart["image_url"] = Value(Value::Map{{"url", Value("data:image/png;base64,AAAA")}});
  Value::Map both_msg;
  both_msg["role"] = Value("user");
  both_msg["content"] = Value::Array{Value(ipart), Value(vpart)};
  Value::Map both_work;
  both_work["messages"] = Value::Array{Value(both_msg)};
  check(throws_protocol([&] {
          (void)umminfer::stage::parse_qwen38_work(Value(both_work));
        }),
        "qwen38 rejects combining image_url and video_url in one turn");
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

void test_sampling_params() {
  using umminfer::Value;
  using umminfer::stage::parse_sampling_params;

  // Absent fields keep the greedy default, so the request is bit-identical to
  // the release that had no sampling knobs.
  Value::Map empty;
  const auto greedy = parse_sampling_params(empty);
  check(greedy.temperature == 0.0f, "sampling: default temperature is greedy");
  check(greedy.top_p == 1.0f, "sampling: default top_p is 1");
  check(!greedy.has_seed, "sampling: no seed by default");
  check(greedy.seed == 0, "sampling: default seed value is zero");

  Value::Map nil;
  nil["temperature"] = Value();
  nil["top_p"] = Value();
  nil["seed"] = Value();
  check(!parse_sampling_params(nil).has_seed, "sampling: nil seed is absent");

  Value::Map valid;
  valid["temperature"] = Value(0.8);
  valid["top_p"] = Value(0.9);
  valid["seed"] = Value(static_cast<int64_t>(42));
  const auto seeded = parse_sampling_params(valid);
  check(seeded.temperature == 0.8f, "sampling: temperature parsed");
  check(seeded.top_p == 0.9f, "sampling: top_p parsed");
  check(seeded.has_seed, "sampling: seed recorded");
  check(seeded.seed == 42, "sampling: positive seed keeps its value");

  // Boundary values of both ranges are accepted.
  Value::Map edges;
  edges["temperature"] = Value(2.0);
  edges["top_p"] = Value(1.0);
  const auto upper = parse_sampling_params(edges);
  check(upper.temperature == 2.0f, "sampling: temperature 2 accepted");
  check(upper.top_p == 1.0f, "sampling: top_p 1 accepted");
  edges["temperature"] = Value(0.0);
  edges["top_p"] = Value(0.001);
  const auto lower = parse_sampling_params(edges);
  check(lower.temperature == 0.0f, "sampling: temperature 0 accepted");
  check(lower.top_p == 0.001f, "sampling: small top_p accepted");

  // seed is a signed int64. Zero is a seed, not an absent field.
  Value::Map zero_seed;
  zero_seed["seed"] = Value(static_cast<int64_t>(0));
  const auto zero = parse_sampling_params(zero_seed);
  check(zero.has_seed, "sampling: seed 0 is present");
  check(zero.seed == 0, "sampling: seed 0 keeps its value");

  // Negative seeds are accepted and map by two's-complement conversion.
  Value::Map negative;
  negative["seed"] = Value(static_cast<int64_t>(-42));
  const auto negative_seed = parse_sampling_params(negative);
  check(negative_seed.has_seed, "sampling: negative seed recorded");
  check(negative_seed.seed == static_cast<uint64_t>(-42),
        "sampling: -42 maps to two's-complement uint64");
  check(negative_seed.seed == ~uint64_t{41}, "sampling: -42 equals ~41");

  Value::Map lowest;
  lowest["seed"] = Value(std::numeric_limits<int64_t>::min());
  const auto lowest_seed = parse_sampling_params(lowest);
  check(lowest_seed.has_seed, "sampling: INT64_MIN recorded");
  check(lowest_seed.seed == uint64_t{1} << 63,
        "sampling: INT64_MIN maps to 2^63");

  Value::Map highest;
  highest["seed"] = Value(std::numeric_limits<int64_t>::max());
  check(parse_sampling_params(highest).seed ==
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
        "sampling: INT64_MAX keeps its value");

  // The mapping is injective: -42 and 42 must not collide.
  check(negative_seed.seed != seeded.seed,
        "sampling: -42 and 42 map to different uint64 seeds");

  // Invalid values are rejected, not clamped or silently ignored.
  auto rejects = [](const char* key, Value value) {
    Value::Map params;
    params[key] = std::move(value);
    return throws_protocol([&] { (void)parse_sampling_params(params); });
  };
  check(rejects("temperature", Value(-0.1)), "sampling: temperature -0.1 rejected");
  check(rejects("temperature", Value(2.1)), "sampling: temperature 2.1 rejected");
  check(rejects("temperature", Value(std::nan(""))),
        "sampling: NaN temperature rejected");
  check(rejects("temperature", Value(std::string("0.8"))),
        "sampling: string temperature rejected");
  check(rejects("temperature", Value(true)), "sampling: bool temperature rejected");

  check(rejects("top_p", Value(0.0)), "sampling: top_p 0 rejected");
  check(rejects("top_p", Value(1.1)), "sampling: top_p 1.1 rejected");
  check(rejects("top_p", Value(std::nan(""))), "sampling: NaN top_p rejected");
  check(rejects("top_p", Value(std::string("0.9"))),
        "sampling: string top_p rejected");

  check(rejects("seed", Value(1.5)), "sampling: float seed rejected");
  check(rejects("seed", Value(std::string("42"))), "sampling: string seed rejected");
  check(rejects("seed", Value(true)), "sampling: bool seed rejected");

  // An integer temperature is a number, not a type error.
  Value::Map int_temperature;
  int_temperature["temperature"] = Value(static_cast<int64_t>(1));
  check(parse_sampling_params(int_temperature).temperature == 1.0f,
        "sampling: integer temperature accepted as a number");
}

}  // namespace

int main() {
  test_parse_image_backend();
  test_text_is_mostly_cjk();
  test_tts_work();
  test_tts_result();
  test_sampling_params();
  if (g_failures) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("all stage common checks passed\n");
  return 0;
}
