// SPDX-License-Identifier: Apache-2.0
// Minimal assert-based host tests. No framework: one binary, nonzero exit on
// failure. Covers the pieces that must work without a GPU.
#include "umm_asr_native/bf16.hpp"
#include "umm_asr_native/feature_frontend.hpp"
#include "umm_asr_native/safetensors.hpp"
#include "umm_asr_native/tokenizer.hpp"
#include "umm_asr_native/wav.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {
int g_failures = 0;
void check(bool cond, const char* what) {
  if (!cond) {
    std::printf("FAIL: %s\n", what);
    ++g_failures;
  }
}

std::string make_wav(uint16_t format, uint16_t channels, uint32_t rate, uint16_t bits,
                     const std::string& pcm) {
  std::string w;
  w += "RIFF";
  uint32_t riff = 36 + static_cast<uint32_t>(pcm.size());
  w.append(reinterpret_cast<const char*>(&riff), 4);
  w += "WAVEfmt ";
  uint32_t fmt_size = 16;
  w.append(reinterpret_cast<const char*>(&fmt_size), 4);
  w.append(reinterpret_cast<const char*>(&format), 2);
  w.append(reinterpret_cast<const char*>(&channels), 2);
  w.append(reinterpret_cast<const char*>(&rate), 4);
  uint32_t byte_rate = rate * channels * bits / 8;
  w.append(reinterpret_cast<const char*>(&byte_rate), 4);
  uint16_t block = channels * bits / 8;
  w.append(reinterpret_cast<const char*>(&block), 2);
  w.append(reinterpret_cast<const char*>(&bits), 2);
  w += "data";
  uint32_t dsize = static_cast<uint32_t>(pcm.size());
  w.append(reinterpret_cast<const char*>(&dsize), 4);
  w += pcm;
  return w;
}

void test_wav_ok() {
  std::string pcm;
  for (int16_t s : {int16_t(0), int16_t(32767), int16_t(-32768), int16_t(-16384)})
    pcm.append(reinterpret_cast<const char*>(&s), 2);
  auto wav = make_wav(1, 1, 16000, 16, pcm);
  auto audio = umminfer::native::load_wav_pcm16(wav);
  check(audio.sample_rate == 16000, "wav rate parsed");
  check(audio.samples.size() == 4, "wav sample count");
  check(audio.samples[0] == 0.0f, "wav zero sample");
  check(std::fabs(audio.samples[1] - 32767.0f / 32768.0f) < 1e-5f, "wav max sample");
  check(std::fabs(audio.samples[2] + 1.0f) < 1e-6f, "wav min sample");
}

void test_wav_errors() {
  const std::string pcm(64, '\x01');
  auto expect_fail = [](const std::string& bytes, const char* name) {
    bool threw = false;
    try {
      umminfer::native::load_wav_pcm16(bytes);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    check(threw, name);
  };
  expect_fail("not a wav file at all", "wav rejects garbage");
  expect_fail(make_wav(1, 2, 16000, 16, pcm), "wav rejects stereo");
  expect_fail(make_wav(1, 1, 48000, 16, pcm), "wav rejects 48k");
  expect_fail(make_wav(1, 1, 16000, 24, pcm), "wav rejects 24-bit");
  expect_fail(make_wav(3, 1, 16000, 16, pcm), "wav rejects float format");
  std::string truncated = make_wav(1, 1, 16000, 16, pcm);
  truncated.resize(truncated.size() - 10);
  expect_fail(truncated, "wav rejects truncation");
}

void test_bf16() {
  using umminfer::native::Bf16;
  check(Bf16(0.0f).to_float() == 0.0f, "bf16 zero");
  check(Bf16(1.0f).to_float() == 1.0f, "bf16 one");
  check(Bf16(-2.0f).to_float() == -2.0f, "bf16 negative two");
  // 0.1 cannot be represented exactly; must round to a nearby value.
  float r = Bf16(0.1f).to_float();
  check(std::fabs(r - 0.1f) < 0.001f, "bf16 0.1 rounds within 1e-3");
}

void test_safetensors_roundtrip() {
  using namespace umminfer::native;
  std::vector<std::pair<std::string, TensorInfo>> tensors;
  TensorInfo a;
  a.dtype = "BF16";
  a.shape = {2, 3};
  a.byte_offset = 0;
  a.byte_size = 12;
  TensorInfo b;
  b.dtype = "F32";
  b.shape = {4};
  b.byte_offset = 12;
  b.byte_size = 16;
  tensors.emplace_back("thinker.lm_head.weight", a);
  tensors.emplace_back("thinker.model.norm.weight", b);
  std::string data(28, '\x00');
  auto file = SafeTensors::write(tensors, data);
  SafeTensors reader(file);
  check(reader.tensors().size() == 2, "st tensor count");
  const auto& info = reader.info("thinker.lm_head.weight");
  check(info.shape.size() == 2 && info.shape[1] == 3, "st shape parsed");
  check(info.dtype == "BF16", "st dtype parsed");
  check(reader.tensor_bytes("thinker.model.norm.weight").size() == 16, "st byte view");
}

void test_safetensors_errors() {
  using namespace umminfer::native;
  auto expect_fail = [](std::string bytes, const char* name) {
    bool threw = false;
    try {
      SafeTensors reader(std::move(bytes));
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    check(threw, name);
  };
  expect_fail("short", "st rejects short file");
  // Header length claims 1 MiB but the file is empty past the length prefix.
  std::string bogus(8, '\x00');
  uint64_t huge = 1u << 20;
  bogus.append(reinterpret_cast<const char*>(&huge), 8);
  bogus.resize(8);
  for (int i = 0; i < 8; ++i) bogus[i] = static_cast<char>((huge >> (8 * i)) & 0xff);
  expect_fail(bogus, "st rejects oversized header");
  // Shape/dtype mismatch: BF16 [3] needs 6 bytes, claims 4.
  TensorInfo bad;
  bad.dtype = "BF16";
  bad.shape = {3};
  bad.byte_offset = 0;
  bad.byte_size = 4;
  expect_fail(SafeTensors::write({{"x", bad}}, std::string(4, '\0')),
              "st rejects shape size mismatch");
  bool missing_threw = false;
  SafeTensors ok(SafeTensors::write(
      [] {
        TensorInfo t;
        t.dtype = "F32";
        t.shape = {1};
        t.byte_size = 4;
        return std::vector<std::pair<std::string, TensorInfo>>{{"present", t}};
      }(),
      std::string(4, '\0')));
  try {
    ok.tensor_bytes("absent");
  } catch (const std::invalid_argument&) {
    missing_threw = true;
  }
  check(missing_threw, "st missing tensor throws");
}

void test_mel_silence() {
  umminfer::native::WhisperMel mel;
  auto one_sec = mel.compute(std::vector<float>(16000, 0.0f));
  check(one_sec.num_mels == 128 && one_sec.num_frames == 3000, "mel output shape 128x3000");
  check(one_sec.valid_frames == 100, "mel valid frames for 1 s");
  check(one_sec.data.size() == 128u * 3000u, "mel element count");
  // Silence: power floored at 1e-10 -> log10 = -10 -> (-10+4)/4 = -1.5.
  check(std::fabs(one_sec.data[0] + 1.5f) < 1e-6f, "mel silence floor");
  check(std::fabs(one_sec.data.back() + 1.5f) < 1e-6f, "mel padded tail floor");
  auto full = mel.compute(std::vector<float>(480000, 0.0f));
  check(full.valid_frames == 3000, "mel valid frames for 30 s");
  auto over = mel.compute(std::vector<float>(500000, 0.0f));
  check(over.valid_frames == 3000, "mel truncates beyond 30 s");
}

void test_tokenizer() {
  using umminfer::native::QwenTokenizer;
  // Minimal vocab: printable ASCII maps 1:1; U+0120 ("Ġ") maps to byte 0x20.
  const char* path = "/tmp/umm_asr_vocab_fixture.json";
  {
    std::ofstream f(path, std::ios::binary);
    f << R"({"A":0,"B":1,"Ġ":2,"<|endoftext|>":151643,"<|im_start|>":151644})";
  }
  QwenTokenizer tok(path);
  auto prompt = QwenTokenizer::build_asr_prompt(3);
  const std::vector<int> expected = {
      151644, 872, 198, 151669, 151676, 151676, 151676, 151670,
      151645, 198, 151644, 77091, 198};
  check(prompt == expected, "asr prompt token sequence");
  bool threw = false;
  try { QwenTokenizer::build_asr_prompt(0); } catch (const std::invalid_argument&) { threw = true; }
  check(threw, "asr prompt rejects zero audio tokens");
  check(tok.is_special(151643) && tok.is_special(151676) && !tok.is_special(2), "special id range");
  check(tok.decode({0, 1}) == "AB", "byte-level decode ascii");
  check(tok.decode({2}) == " ", "byte-level decode maps Ġ to space");
  check(tok.decode({151643, 0}) == "A", "decode skips special tokens");
}

}  // namespace

int main() {
  test_wav_ok();
  test_wav_errors();
  test_bf16();
  test_safetensors_roundtrip();
  test_safetensors_errors();
  test_mel_silence();
  test_tokenizer();
  if (g_failures) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("all host checks passed\n");
  return 0;
}
