// SPDX-License-Identifier: Apache-2.0
// Host tests for the wire-shape adapters the native stages share. No CUDA and
// no model: these run on the Mac build host.
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "stage_common.hpp"
#include "umminfer/scheduling/simple_scheduler.hpp"

namespace {

int g_failures = 0;

void check(bool condition, const char* what) {
  if (condition) return;
  std::fprintf(stderr, "FAIL: %s\n", what);
  ++g_failures;
}

template <typename Fn>
void check_throws(Fn&& fn, const char* what) {
  try {
    fn();
  } catch (const std::exception&) {
    return;
  }
  std::fprintf(stderr, "FAIL: %s (no exception)\n", what);
  ++g_failures;
}

void append_u16(std::string& out, uint16_t value) {
  out.push_back(static_cast<char>(value & 0xff));
  out.push_back(static_cast<char>((value >> 8) & 0xff));
}

void append_u32(std::string& out, uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8) {
    out.push_back(static_cast<char>((value >> shift) & 0xff));
  }
}

std::string pcm_bytes(const std::vector<int16_t>& samples) {
  std::string out;
  for (int16_t sample : samples) {
    append_u16(out, static_cast<uint16_t>(sample));
  }
  return out;
}

std::string wav_16k(const std::vector<int16_t>& samples, uint32_t rate = 16000) {
  const std::string pcm = pcm_bytes(samples);
  std::string wav = "RIFF";
  append_u32(wav, static_cast<uint32_t>(36 + pcm.size()));
  wav += "WAVE";
  wav += "fmt ";
  append_u32(wav, 16);
  append_u16(wav, 1);  // PCM
  append_u16(wav, 1);  // mono
  append_u32(wav, rate);
  append_u32(wav, rate * 2);
  append_u16(wav, 2);
  append_u16(wav, 16);
  wav += "data";
  append_u32(wav, static_cast<uint32_t>(pcm.size()));
  wav += pcm;
  return wav;
}

umminfer::Value audio_input(const std::string& bytes) {
  umminfer::Value::Map map;
  map["audio"] = umminfer::Value::bin(bytes);
  return umminfer::Value(std::move(map));
}

umminfer::Value text_part(const std::string& text) {
  umminfer::Value::Map part;
  part["type"] = umminfer::Value("text");
  part["text"] = umminfer::Value(text);
  return umminfer::Value(std::move(part));
}

umminfer::Value image_part(const std::string& url) {
  umminfer::Value::Map image;
  image["url"] = umminfer::Value(url);
  umminfer::Value::Map part;
  part["type"] = umminfer::Value("image_url");
  part["image_url"] = umminfer::Value(std::move(image));
  return umminfer::Value(std::move(part));
}

umminfer::Value messages_input(const umminfer::Value& content) {
  umminfer::Value::Map user;
  user["role"] = umminfer::Value("user");
  user["content"] = content;
  umminfer::Value::Array messages;
  messages.push_back(umminfer::Value(std::move(user)));
  umminfer::Value::Map inputs;
  inputs["messages"] = umminfer::Value(std::move(messages));
  return umminfer::Value(std::move(inputs));
}

void test_asr_parsing() {
  using namespace umminfer::stage;
  const std::vector<int16_t> samples = {0, 16384, -32768};
  const AsrWork raw = parse_asr_work(audio_input(pcm_bytes(samples)));
  check(raw.samples.size() == 3, "raw PCM16 sample count");
  check(std::fabs(raw.samples[1] - 0.5f) < 1e-6f, "raw PCM16 scale");
  check(std::fabs(raw.samples[2] + 1.0f) < 1e-6f, "raw PCM16 negative scale");

  const AsrWork from_wav = parse_asr_work(audio_input(wav_16k(samples)));
  check(from_wav.samples == raw.samples, "WAV and raw PCM16 agree");

  // The wrapper Stage passes for a non-entry payload.
  umminfer::Value::Map wrapped;
  wrapped["inputs"] = audio_input(pcm_bytes(samples));
  check(parse_asr_work(umminfer::Value(std::move(wrapped))).samples.size() == 3,
        "inputs wrapper accepted");

  check_throws([] { parse_asr_work(umminfer::Value::bin(std::string())); },
               "non-map work rejected");
  check_throws([] { parse_asr_work(audio_input("")); }, "empty audio rejected");
  check_throws([] { parse_asr_work(audio_input("abc")); }, "odd PCM16 rejected");
  check_throws([&] { parse_asr_work(audio_input(wav_16k(samples, 48000))); },
               "non-16k WAV rejected");
  umminfer::Value::Map no_audio;
  no_audio["text"] = umminfer::Value("x");
  check_throws([&] { parse_asr_work(umminfer::Value(std::move(no_audio))); },
               "missing audio rejected");
}

void test_omni_parsing() {
  using namespace umminfer::stage;
  const OmniWork text_only = parse_omni_work(messages_input(umminfer::Value("hi")));
  check(text_only.text == "hi" && text_only.image_url.empty(),
        "string content parsed");
  check(render_omni_chat("hi", false) ==
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
        "text-only chat render");

  umminfer::Value::Array parts;
  parts.push_back(image_part("data:image/png;base64,aGVsbG8="));
  parts.push_back(text_part("describe"));
  const OmniWork with_image = parse_omni_work(messages_input(umminfer::Value(parts)));
  check(with_image.text == "describe", "text part parsed next to image");
  check(with_image.image_url == "data:image/png;base64,aGVsbG8=", "image url parsed");
  check(render_omni_chat("describe", true) ==
            "<|im_start|>user\n<|vision_start|><|vision_end|>describe"
            "<|im_end|>\n<|im_start|>assistant\n",
        "image chat render");

  umminfer::Value::Array two_images;
  two_images.push_back(image_part("data:image/png;base64,aGVsbG8="));
  two_images.push_back(image_part("data:image/png;base64,aGVsbG8="));
  check_throws([&] {
    parse_omni_work(messages_input(umminfer::Value(two_images)));
  }, "two images rejected");

  umminfer::Value::Array bad_image;
  umminfer::Value::Map part;
  part["type"] = umminfer::Value("image_url");
  bad_image.push_back(umminfer::Value(std::move(part)));
  check_throws([&] {
    parse_omni_work(messages_input(umminfer::Value(bad_image)));
  }, "image part without url rejected");

  umminfer::Value::Map empty_inputs;
  empty_inputs["messages"] = umminfer::Value(umminfer::Value::Array{});
  check_throws([&] { parse_omni_work(umminfer::Value(std::move(empty_inputs))); },
               "empty messages rejected");
}

void test_base64_and_data_url() {
  using namespace umminfer::stage;
  check(base64_decode("aGVsbG8=") == "hello", "base64 with padding");
  check(base64_decode("YQ==") == "a", "base64 one byte");
  check(base64_decode("YWI=") == "ab", "base64 two bytes");
  check_throws([] { base64_decode("abc"); }, "base64 bad length rejected");
  check_throws([] { base64_decode("aGV*sbG8="); }, "base64 bad char rejected");
  check(decode_data_url("data:image/png;base64,aGVsbG8=") == "hello",
        "data URL decode");
  check_throws([] { decode_data_url("http://example.com/a.png"); },
               "http image URL rejected");
  check_throws([] { decode_data_url("data:image/png,hello"); },
               "non-base64 data URL rejected");
}

void test_tts_parsing() {
  using namespace umminfer::stage;
  umminfer::Value::Map inputs;
  inputs["text"] = umminfer::Value("hello");
  inputs["voice"] = umminfer::Value("Vivian");
  inputs["format"] = umminfer::Value("wav");
  inputs["sample_rate"] = umminfer::Value(static_cast<int64_t>(24000));
  const TtsWork parsed = parse_tts_work(umminfer::Value(std::move(inputs)));
  check(parsed.text == "hello", "tts text parsed");
  check(parsed.voice == "vivian", "tts voice lowercased");
  check(parsed.sample_rate == 24000, "tts sample rate");

  auto with = [](const char* key, umminfer::Value value) {
    umminfer::Value::Map map;
    map["text"] = umminfer::Value("x");
    map[key] = std::move(value);
    return umminfer::Value(std::move(map));
  };
  check_throws([&] { parse_tts_work(with("format", umminfer::Value("mp3"))); },
               "non-wav format rejected");
  check_throws([&] { parse_tts_work(with("sample_rate", umminfer::Value(int64_t{16000}))); },
               "non-24k sample rate rejected");
  check_throws([&] { parse_tts_work(with("voice", umminfer::Value(int64_t{1}))); },
               "non-string voice rejected");
  umminfer::Value::Map no_text;
  no_text["voice"] = umminfer::Value("vivian");
  check_throws([&] { parse_tts_work(umminfer::Value(std::move(no_text))); },
               "missing tts text rejected");

  check(text_is_mostly_cjk("你好，世界"), "CJK text detected");
  check(!text_is_mostly_cjk("hello world"), "Latin text detected");
  check(text_is_mostly_cjk("其实我真的有发现，我是一个特别善于观察别人情绪的人。"),
        "Chinese sentence detected");
}

void test_image_parsing() {
  using namespace umminfer::stage;
  umminfer::Value::Map inputs;
  inputs["prompt"] = umminfer::Value("a fox");
  inputs["negative_prompt"] = umminfer::Value("blurred");
  inputs["size"] = umminfer::Value("512x512");
  inputs["seed"] = umminfer::Value(static_cast<int64_t>(42));
  inputs["steps"] = umminfer::Value(static_cast<int64_t>(50));
  inputs["guidance"] = umminfer::Value(4.0);
  inputs["output_format"] = umminfer::Value("png");
  const ImageWork parsed = parse_image_work(umminfer::Value(std::move(inputs)));
  check(parsed.prompt == "a fox" && parsed.negative_prompt == "blurred",
        "image prompts parsed");
  check(parsed.width == 512 && parsed.height == 512, "image size parsed");
  check(parsed.seed == 42 && parsed.steps == 50 && parsed.guidance == 4.0f,
        "image generation options parsed");

  auto with = [](const char* key, umminfer::Value value) {
    umminfer::Value::Map map;
    map["prompt"] = umminfer::Value("x");
    map[key] = std::move(value);
    return umminfer::Value(std::move(map));
  };
  check(parse_image_work(with("size", umminfer::Value("512x512"))).steps == 0,
        "image default steps preserved");
  check_throws([&] { parse_image_work(with("size", umminfer::Value("1024x1024"))); },
               "unvalidated image size rejected");
  check_throws([&] { parse_image_work(with("output_format", umminfer::Value("jpeg"))); },
               "unsupported image format rejected");
  check_throws([&] { parse_image_work(with("steps", umminfer::Value(static_cast<int64_t>(201)))); },
               "excessive image steps rejected");
  check_throws([&] { parse_image_work(with("seed", umminfer::Value(static_cast<int64_t>(-1)))); },
               "negative image seed rejected");
  check_throws([] { parse_image_work(umminfer::Value(umminfer::Value::Map{})); },
               "missing image prompt rejected");
}

void test_result_shapes() {
  using namespace umminfer::stage;
  const umminfer::Value asr = asr_result("text", 7, "stop");
  check(asr.find("text") != nullptr && asr.find("text")->as_str() == "text",
        "asr result text");
  check(asr.find("generated_token_count")->as_int() == 7, "asr token count");
  check(asr.find("finish_reason")->as_str() == "stop", "asr finish reason");

  const umminfer::Value omni = omni_result("answer", 3);
  check(omni.find("content")->as_str() == "answer", "omni result content");

  const umminfer::Value tts = tts_result("RIFFdata", 24000, 1, "vivian");
  check(tts.find("media_type")->as_str() == "audio", "tts media type");
  check(tts.find("mime_type")->as_str() == "audio/wav", "tts mime type");
  check(tts.find("data")->is_bin() && tts.find("data")->as_bin() == "RIFFdata",
        "tts artifact bytes");
  check(tts.find("sample_rate")->as_int() == 24000, "tts artifact sample rate");
  check_throws([] { tts_result("", 24000, 1, "vivian"); }, "empty TTS output rejected");

  const umminfer::Value image = image_result("PNGdata", 512, 512, 42, 50);
  check(image.find("media_type")->as_str() == "image", "image media type");
  check(image.find("mime_type")->as_str() == "image/png", "image mime type");
  check(image.find("size")->as_str() == "512x512", "image descriptor size");
  check(image.find("seed")->as_int() == 42 && image.find("steps")->as_int() == 50,
        "image generation metadata");
  check(image.find("data")->is_bin() && image.find("data")->as_bin() == "PNGdata",
        "image artifact bytes");
  check_throws([] { image_result("", 512, 512, 0, 50); }, "empty image output rejected");
}

// The scheduler must hand each request its own abort check: a running request
// stops at its next boundary, an abort for another id changes nothing, and a
// later request on the same scheduler still succeeds.
void test_scheduler_cancellation() {
  using namespace umminfer;
  std::atomic<int> steps{0};
  std::atomic<bool> running{false};
  std::atomic<bool> aborted_seen{false};
  SimpleScheduler scheduler(ComputeFn{});
  scheduler.set_cancellable_compute([&](const Value& work, const AbortCheck& aborted) {
    if (work.is_str() && work.as_str() == "fast") return Value(std::string("ok"));
    running = true;
    for (int index = 0; index < 20000; ++index) {
      if (aborted()) {
        aborted_seen = true;
        throw RequestAborted("aborted at step " + std::to_string(index));
      }
      steps = index;
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return Value(std::string("finished"));
  });
  std::thread runner([&] { scheduler.start(); });
  scheduler.enqueue(IncomingMessage{"req-1", "new_request", Value(std::string("slow"))});
  const auto wait_until = [](auto predicate) {
    for (int tries = 0; tries < 2000 && !predicate(); ++tries) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return predicate();
  };
  check(wait_until([&] { return running.load(); }), "cancellable compute started");

  scheduler.abort("req-other");
  const int before = steps.load();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  check(steps.load() > before, "abort for an unknown id does not stop the request");

  scheduler.abort("req-1");
  check(wait_until([&] { return aborted_seen.load(); }),
        "aborted request left the loop at a boundary");
  const int settled = steps.load();
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  check(steps.load() == settled, "aborted request stopped advancing");
  check(settled < 20000, "request stopped before its natural end");

  scheduler.enqueue(IncomingMessage{"req-2", "new_request", Value(std::string("fast"))});
  const auto result = scheduler.outbox().get(2.0);
  check(result.has_value() && result->request_id == "req-2" &&
            result->data.as_str() == "ok",
        "a later request completes after the abort");
  const auto none = scheduler.outbox().get(0.2);
  check(!none.has_value(), "the aborted request emitted no result");

  scheduler.stop();
  runner.join();
}

}  // namespace

int main() {
  test_asr_parsing();
  test_omni_parsing();
  test_base64_and_data_url();
  test_tts_parsing();
  test_image_parsing();
  test_result_shapes();
  test_scheduler_cancellation();
  if (g_failures != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("stage host tests passed\n");
  return 0;
}
