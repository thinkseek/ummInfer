// SPDX-License-Identifier: Apache-2.0
// Host-only checks: BPE golden ids, added-token boundaries, WAV writer and
// stats, safetensors round trip. No CUDA. Run with the model data directory
// as argv[1] (or UMM_TTS_TEST_DATA); it only needs vocab.json, merges.txt and
// tokenizer_config.json.
#include "bpe_tokenizer.hpp"
#include "prompt.hpp"
#include "safetensors.hpp"
#include "wav_io.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace umminfer::native;

#define CHECK(cond)                                                      \
  do {                                                                   \
    if (!(cond)) {                                                       \
      std::cerr << "CHECK failed: " #cond " at " << __FILE__ << ":"      \
                << __LINE__ << "\n";                                    \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

static void check_eq(const std::vector<int>& got, const std::vector<int>& want,
                     const std::string& label) {
  if (got != want) {
    std::cerr << "mismatch: " << label << "\n got:";
    for (int x : got) std::cerr << " " << x;
    std::cerr << "\nwant:";
    for (int x : want) std::cerr << " " << x;
    std::cerr << "\n";
    std::exit(1);
  }
}

static void test_tokenizer(const std::string& data_dir) {
  Qwen2BpeTokenizer tok(data_dir);

  // Golden bodies, spec 4 section 4.
  check_eq(tok.encode("其实我真的有发现，我是一个特别善于观察别人情绪的人。"),
           {101045, 110146, 18830, 99879, 3837, 35946, 101909, 100654, 106614,
            104144, 101106, 104405, 100623, 1773},
           "zh body");
  check_eq(tok.encode("Hello, welcome to the native inference engine test."),
           {9707, 11, 10565, 311, 279, 9867, 44378, 4712, 1273, 13},
           "en body");

  // Full wrapped prompts as the CLI builds them.
  check_eq(tok.encode("<|im_start|>assistant\n"
                      "其实我真的有发现，我是一个特别善于观察别人情绪的人。"
                      "<|im_end|>\n<|im_start|>assistant\n"),
           {151644, 77091, 198, 101045, 110146, 18830, 99879, 3837, 35946,
            101909, 100654, 106614, 104144, 101106, 104405, 100623, 1773,
            151645, 198, 151644, 77091, 198},
           "zh wrapped");
  check_eq(tok.encode("<|im_start|>assistant\n"
                      "Hello, welcome to the native inference engine test."
                      "<|im_end|>\n<|im_start|>assistant\n"),
           {151644, 77091, 198, 9707, 11, 10565, 311, 279, 9867, 44378, 4712,
            1273, 13, 151645, 198, 151644, 77091, 198},
           "en wrapped");

  // Added tokens split adjacent letters atomically.
  check_eq(tok.encode("a<|im_start|>b"), {64, 151644, 65}, "added inline");
  check_eq(tok.encode("<tts_pad><tts_text_bos><tts_text_eod>"),
           {151671, 151672, 151673}, "tts specials");
  check_eq(tok.encode("<tts_text_bos_single>"), {151674}, "longest added match");

  // Digit chunks never merge; whitespace pretoken boundaries.
  check_eq(tok.encode("123"), {16, 17, 18}, "digits isolated");
  check_eq(tok.encode("  hi "), {220, 15588, 220}, "spaces");
  check_eq(tok.encode("a,b/c\nd"), {64, 8402, 2899, 198, 67}, "punct and newline");
  check_eq(tok.encode("abc ABC aB"), {13683, 19360, 264, 33}, "case split P");
}

static void test_prompt() {
  // zh: body N=14 -> prefill N+11 = 25 rows.
  std::vector<int> zh_ids = {
      151644, 77091, 198, 101045, 110146, 18830, 99879, 3837, 35946, 101909,
      100654, 106614, 104144, 101106, 104405, 100623, 1773, 151645, 198,
      151644, 77091, 198};
  PromptPlan zh = build_prompt_plan(zh_ids, kLangChinese, kSpkVivian);
  CHECK(zh.body_tokens == 14);
  CHECK(zh.rows.size() == 25);
  CHECK(zh.codec_prefill_ids ==
        std::vector<int>({2154, 2156, 2055, 2157, 3065, 2148, 2149}));
  CHECK(zh.rows[0].size() == 1 && zh.rows[0][0].kind == PromptTerm::kTextProj &&
        zh.rows[0][0].id == 151644);
  CHECK(zh.rows[3].size() == 2 && zh.rows[3][0].id == 2154 &&
        zh.rows[3][1].kind == PromptTerm::kTtsPad);
  CHECK(zh.rows[8][0].id == 2148 && zh.rows[8][1].kind == PromptTerm::kTtsBos);
  CHECK(zh.rows[9][0].id == 101045 && zh.rows[9][1].id == 2148);
  CHECK(zh.rows.back().size() == 2 &&
        zh.rows.back()[0].kind == PromptTerm::kTtsPad &&
        zh.rows.back()[1].id == 2149);

  // en: N=10 -> 21 rows; english/ryan codec ids.
  std::vector<int> en_ids = {151644, 77091, 198, 9707, 11, 10565, 311, 279,
                             9867, 44378, 4712, 1273, 13, 151645, 198, 151644,
                             77091, 198};
  PromptPlan en = build_prompt_plan(en_ids, kLangEnglish, kSpkRyan);
  CHECK(en.body_tokens == 10 && en.rows.size() == 21);
  CHECK(en.codec_prefill_ids ==
        std::vector<int>({2154, 2156, 2050, 2157, 3061, 2148, 2149}));

  bool threw = false;
  try {
    (void)speaker_id_from_name("nope");
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  CHECK(speaker_id_from_name("Vivian") == kSpkVivian);
  CHECK(language_id_from_name("EN") == kLangEnglish);
}

static void test_wav() {
  const std::vector<float> samples = {0.0f, 0.5f, -0.5f, 1.0f, -1.0f, 1.5f,
                                      -1.5f, 0.0005f};
  const std::string wav = write_wav_pcm16_mono(samples, 24000);
  CHECK(wav.size() == 44 + samples.size() * 2);
  CHECK(wav.substr(0, 4) == "RIFF" && wav.substr(8, 4) == "WAVE");
  CHECK(wav.substr(12, 4) == "fmt " && wav.substr(36, 4) == "data");
  const auto u16 = [&](size_t off) {
    return static_cast<uint16_t>(static_cast<uint8_t>(wav[off])) |
           (static_cast<uint16_t>(static_cast<uint8_t>(wav[off + 1])) << 8);
  };
  const auto u32 = [&](size_t off) {
    return static_cast<uint32_t>(u16(off)) |
           (static_cast<uint32_t>(u16(off + 2)) << 16);
  };
  CHECK(u16(20) == 1);                 // PCM
  CHECK(u16(22) == 1);                 // mono
  CHECK(u32(24) == 24000);             // sample rate
  CHECK(u32(28) == 48000);             // byte rate
  CHECK(u16(32) == 2 && u16(34) == 16);
  CHECK(u32(40) == samples.size() * 2);
  CHECK(static_cast<int16_t>(u16(44)) == 0);
  CHECK(static_cast<int16_t>(u16(46)) == static_cast<int16_t>(lrintf(0.5f * 32767.0f)));
  CHECK(static_cast<int16_t>(u16(50)) == 32767);   // +1.0 exact
  CHECK(static_cast<int16_t>(u16(52)) == static_cast<int16_t>(-32767));  // -1.0 exact
  CHECK(static_cast<int16_t>(u16(54)) == 32767);   // +1.5 clamps to max
  CHECK(static_cast<int16_t>(u16(56)) == -32768);  // -1.5 clips to min

  bool threw = false;
  try {
    (void)write_wav_pcm16_mono(samples, 0);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);

  const WavStats st = pcm_stats(samples, 24000);
  CHECK(st.samples == 8 && st.sample_rate == 24000);
  CHECK(st.duration_s == 8.0 / 24000.0);
  CHECK(st.peak == 1.5f);
  CHECK(st.rms > 0.85 && st.rms < 0.95);  // sqrt(7.5/8) ~= 0.968 minus... exact
  CHECK(st.non_silence_ratio == 6.0 / 8.0);  // 0.0 and 0.0005 are silent
}

static void test_safetensors() {
  // Two tensors: one BF16 (2 bytes), one F32 (4 bytes).
  std::vector<std::pair<std::string, TensorInfo>> tensors;
  TensorInfo a;
  a.dtype = "BF16";
  a.shape = {2, 2};
  a.byte_offset = 0;
  a.byte_size = 8;
  TensorInfo b;
  b.dtype = "F32";
  b.shape = {3};
  b.byte_offset = 8;
  b.byte_size = 12;
  tensors.emplace_back("a.bf16", a);
  tensors.emplace_back("b.f32", b);
  const std::string data(20, '\x55');
  const std::string file = SafeTensors::write(tensors, data);
  SafeTensors st(file);
  CHECK(st.tensors().size() == 2);
  CHECK(st.info("a.bf16").dtype == "BF16");
  CHECK(st.info("b.f32").shape == std::vector<size_t>({3}));
  CHECK(st.tensor_bytes("a.bf16").size() == 8);
  bool threw = false;
  try {
    (void)st.info("missing");
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}

int main(int argc, char** argv) {
  const char* env = std::getenv("UMM_TTS_TEST_DATA");
  std::string data_dir = argc > 1 ? argv[1] : (env ? env : "");
  if (!data_dir.empty()) {
    test_tokenizer(data_dir);
  } else {
    std::cout << "SKIP test_tokenizer (no MODEL_DATA_DIR / UMM_TTS_TEST_DATA)\n";
  }
  test_prompt();
  test_wav();
  test_safetensors();
  std::cout << "all tts host tests passed\n";
  return 0;
}
