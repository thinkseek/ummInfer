// SPDX-License-Identifier: Apache-2.0
// Assert-based host tests for the Qwen3-ASR host slice: WAV parsing, Whisper
// log-mel frontend shape, and byte-level tokenizer decode. No CUDA and no
// model weights: the vocab is written to a temp directory at run time.
#include "feature_frontend.hpp"
#include "tokenizer.hpp"
#include "wav.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool cond, const char* what) {
  if (!cond) {
    std::printf("FAIL: %s\n", what);
    ++g_failures;
  }
}

void put_u16(std::string& b, size_t off, uint16_t v) {
  b[off] = static_cast<char>(v & 0xff);
  b[off + 1] = static_cast<char>((v >> 8) & 0xff);
}

void put_u32(std::string& b, size_t off, uint32_t v) {
  b[off] = static_cast<char>(v & 0xff);
  b[off + 1] = static_cast<char>((v >> 8) & 0xff);
  b[off + 2] = static_cast<char>((v >> 16) & 0xff);
  b[off + 3] = static_cast<char>((v >> 24) & 0xff);
}

// Minimal canonical 44-byte-header PCM16 WAV.
std::string make_wav(const std::vector<int16_t>& samples, uint16_t channels = 1,
                     uint16_t bits = 16, uint32_t rate = 16000,
                     uint16_t format = 1) {
  const size_t data_bytes = samples.size() * 2;
  std::string wav(44 + data_bytes, '\0');
  wav.replace(0, 4, "RIFF");
  put_u32(wav, 4, static_cast<uint32_t>(36 + data_bytes));
  wav.replace(8, 4, "WAVE");
  wav.replace(12, 4, "fmt ");
  put_u32(wav, 16, 16);
  put_u16(wav, 20, format);
  put_u16(wav, 22, channels);
  put_u32(wav, 24, rate);
  put_u32(wav, 28, rate * channels * (bits / 8));
  put_u16(wav, 32, channels * (bits / 8));
  put_u16(wav, 34, bits);
  wav.replace(36, 4, "data");
  put_u32(wav, 40, static_cast<uint32_t>(data_bytes));
  for (size_t i = 0; i < samples.size(); ++i)
    put_u16(wav, 44 + i * 2, static_cast<uint16_t>(samples[i]));
  return wav;
}

template <typename Fn>
bool throws_invalid(Fn fn) {
  try {
    fn();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

void test_wav_valid() {
  using namespace umminfer::native;
  const std::vector<int16_t> pcm = {0, 16384, -16384, 32767, -32768};
  const PcmAudio got = load_wav_pcm16(make_wav(pcm));
  check(got.sample_rate == 16000, "wav sample rate 16000");
  check(got.samples.size() == pcm.size(), "wav sample count");
  check(got.samples[0] == 0.0f, "wav first sample zero");
  check(std::fabs(got.samples[1] - 0.5f) < 1e-6f, "wav +0.5 scale");
  check(std::fabs(got.samples[2] + 0.5f) < 1e-6f, "wav -0.5 scale");
  check(std::fabs(got.samples[3] - 32767.0f / 32768.0f) < 1e-6f,
        "wav full-scale positive");
  check(got.samples[4] == -1.0f, "wav full-scale negative");

  // An even-length data chunk is required; a trailing odd chunk is rejected.
  std::string odd = make_wav({1, 2, 3});
  put_u32(odd, 40, 5);
  odd.resize(44 + 5);
  check(throws_invalid([&] { (void)load_wav_pcm16(odd); }),
        "wav rejects odd PCM16 byte count");
}

void test_wav_invalid() {
  using namespace umminfer::native;
  const auto pcm = std::vector<int16_t>{1, 2};

  check(throws_invalid([&] { (void)load_wav_pcm16(""); }),
        "wav rejects empty input");
  check(throws_invalid([&] { (void)load_wav_pcm16("RIFFxxxxNOPE"); }),
        "wav rejects non-WAVE form");

  std::string truncated = make_wav(pcm);
  truncated.resize(truncated.size() - 1);
  check(throws_invalid([&] { (void)load_wav_pcm16(truncated); }),
        "wav rejects truncated container");

  check(throws_invalid([&] { (void)load_wav_pcm16(make_wav(pcm, 2)); }),
        "wav rejects stereo");
  check(throws_invalid([&] { (void)load_wav_pcm16(make_wav(pcm, 1, 8)); }),
        "wav rejects 8-bit PCM");
  check(throws_invalid([&] { (void)load_wav_pcm16(make_wav(pcm, 1, 16, 8000)); }),
        "wav rejects non-16 kHz rate");
  check(throws_invalid([&] { (void)load_wav_pcm16(make_wav(pcm, 1, 16, 16000, 3)); }),
        "wav rejects non-PCM format");

  std::string no_data = make_wav(pcm);
  no_data.replace(36, 4, "junk");
  check(throws_invalid([&] { (void)load_wav_pcm16(no_data); }),
        "wav rejects missing data chunk");

  std::string no_fmt = make_wav(pcm);
  no_fmt.replace(12, 4, "junk");
  check(throws_invalid([&] { (void)load_wav_pcm16(no_fmt); }),
        "wav rejects missing fmt chunk");
}

void test_mel_frontend() {
  using namespace umminfer::native;
  check(WhisperMel::valid_frame_count(0) == 0, "mel frame count zero samples");
  check(WhisperMel::valid_frame_count(16000) == 100,
        "mel frame count 1 s == 100 frames");
  check(WhisperMel::valid_frame_count(16000 * 60) == 3000,
        "mel frame count clamps at 3000");

  const WhisperMel mel;
  // A short 100 Hz tone exercise the full padded 30 s chunk.
  std::vector<float> tone(16000, 0.0f);
  for (size_t i = 0; i < tone.size(); ++i)
    tone[i] = 0.25f * std::sin(2.0f * 3.14159265358979f * 100.0f * i / 16000.0f);
  const MelFeatures feats = mel.compute(tone);
  check(feats.num_mels == 128, "mel num_mels 128");
  check(feats.num_frames == 3000, "mel num_frames 3000");
  check(feats.valid_frames == 100, "mel valid_frames for 1 s");
  check(feats.data.size() == 128u * 3000u, "mel data element count");
  bool all_finite = true;
  bool in_range = true;
  for (float v : feats.data) {
    if (!std::isfinite(v)) all_finite = false;
    if (v < -10.0f || v > 10.0f) in_range = false;
  }
  check(all_finite, "mel values finite");
  check(in_range, "mel normalized values inside [-10,10]");
  // Silence must still produce finite, floored values.
  const MelFeatures quiet = mel.compute(std::vector<float>(16000, 0.0f));
  bool quiet_finite = true;
  for (float v : quiet.data)
    if (!std::isfinite(v)) quiet_finite = false;
  check(quiet_finite, "mel silence is finite");
  // The tone must carry more energy in a low mel band than in the top band.
  double low = 0.0, high = 0.0;
  for (int m = 0; m < 8; ++m)
    for (int t = 0; t < 100; ++t) low += feats.data[m * 3000 + t];
  for (int m = 120; m < 128; ++m)
    for (int t = 0; t < 100; ++t) high += feats.data[m * 3000 + t];
  check(low > high, "100 Hz tone energy sits in the low mel bands");
}

void test_tokenizer_prompt_and_decode() {
  using namespace umminfer::native;
  const auto prompt = QwenTokenizer::build_asr_prompt(3);
  const std::vector<int> want = {
      151644, 872, 198, 151669, 151676, 151676, 151676, 151670,
      151645, 198, 151644, 77091, 198};
  check(prompt == want, "ASR prompt ids for 3 audio tokens");
  check(throws_invalid([] { (void)QwenTokenizer::build_asr_prompt(0); }),
        "ASR prompt rejects zero audio tokens");
  check(QwenTokenizer::is_special(151643), "is_special on added token");
  check(QwenTokenizer::is_special(151700), "is_special above added range");
  check(!QwenTokenizer::is_special(198), "is_special false for newline");
  check(QwenTokenizer::is_eos(151643) && QwenTokenizer::is_eos(151645),
        "is_eos accepts end-of-text and im_end");
  check(!QwenTokenizer::is_eos(198), "is_eos false for newline");
}

// GPT-2 byte-level symbol for one byte, as UTF-8 (same table the tokenizer
// builds internally). Lets the fixture use the real on-disk vocab format.
std::string byte_symbol(int b) {
  auto printable = [](int c) {
    return (c >= 33 && c <= 126) || (c >= 161 && c <= 172) ||
           (c >= 174 && c <= 255);
  };
  uint32_t cp;
  if (printable(b)) {
    cp = static_cast<uint32_t>(b);
  } else {
    int n = 0;
    for (int x = 0; x < b; ++x)
      if (!printable(x)) ++n;
    cp = 256u + static_cast<uint32_t>(n);
  }
  std::string out;
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  } else {
    out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  }
  return out;
}

std::string byte_symbols(const std::string& raw) {
  std::string out;
  for (unsigned char c : raw) out += byte_symbol(c);
  return out;
}

std::string json_escape(const std::string& s) {
  std::string out;
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') {
      out.push_back('\\');
      out.push_back(static_cast<char>(c));
    } else if (c < 0x20) {
      static const char* hex = "0123456789abcdef";
      out += "\\u00";
      out.push_back(hex[c >> 4]);
      out.push_back(hex[c & 0xf]);
    } else {
      out.push_back(static_cast<char>(c));
    }
  }
  return out;
}

void test_tokenizer_decode() {
  namespace fs = std::filesystem;
  using namespace umminfer::native;
  const fs::path dir = fs::temp_directory_path() / "umm_asr_vocab_fixture";
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);
  // Real GPT-2 byte-level symbols for ASCII, a space-marked word, and the
  // UTF-8 bytes of 你好, plus one special token decode must skip.
  const std::string v_hello = byte_symbols("Hello");
  const std::string v_world = byte_symbol(' ') + byte_symbols("world");
  const std::string v_ni = byte_symbols("\xE4\xBD\xA0");
  const std::string v_hao = byte_symbols("\xE5\xA5\xBD");
  {
    std::ofstream f(dir / "vocab.json");
    f << "{\"" << json_escape(v_hello) << "\":0,"
      << "\"" << json_escape(v_world) << "\":1,"
      << "\"" << json_escape(v_ni) << "\":2,"
      << "\"" << json_escape(v_hao) << "\":3,"
      << "\"<|im_start|>\":151644}";
  }

  const QwenTokenizer tok((dir / "vocab.json").string());
  check(tok.decode({0}) == "Hello", "tokenizer decodes ASCII piece");
  check(tok.decode({1}) == " world", "tokenizer restores space marker");
  check(tok.decode({2, 3}) == "\xE4\xBD\xA0\xE5\xA5\xBD",
        "tokenizer re-emits raw UTF-8 bytes");
  // Special ids are skipped, unknown ids are ignored, order is preserved.
  check(tok.decode({151644, 0, 151645}) == "Hello",
        "tokenizer skips special tokens");
  check(tok.decode({0, 9999}) == "Hello", "tokenizer ignores out-of-range ids");
  check(tok.decode({}).empty(), "tokenizer decodes empty id list");
  fs::remove_all(dir, ec);
}

}  // namespace

int main() {
  test_wav_valid();
  test_wav_invalid();
  test_mel_frontend();
  test_tokenizer_prompt_and_decode();
  test_tokenizer_decode();
  if (g_failures) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("all asr host checks passed\n");
  return 0;
}
