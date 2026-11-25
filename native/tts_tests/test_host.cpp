// SPDX-License-Identifier: Apache-2.0
// Host-only checks: BPE golden ids, added-token boundaries, WAV writer and
// stats, safetensors round trip. No CUDA. Run with the model data directory
// as argv[1] (or UMM_TTS_TEST_DATA); it only needs vocab.json, merges.txt and
// tokenizer_config.json.
#include "bpe_tokenizer.hpp"
#include "decode_helpers.hpp"
#include "engine.hpp"
#include "prompt.hpp"
#include "safetensors.hpp"
#include "wav_io.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

// GPT-2 byte -> printable-unicode symbol, as UTF-8 (same table the tokenizer
// builds internally). Kept local so the fixture needs no model weights.
static std::string byte_symbol(int b) {
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
  std::string s;
  if (cp < 0x80) {
    s.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    s.push_back(static_cast<char>(0xc0 | (cp >> 6)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  } else {
    s.push_back(static_cast<char>(0xe0 | (cp >> 12)));
    s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  }
  return s;
}

// JSON string body for a symbol: quote, backslash and control bytes must be
// escaped or the fixture is not valid JSON.
static std::string json_escape(const std::string& s) {
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

// Synthetic vocab/merges/config so tokenizer coverage never depends on the
// model data directory.
static void test_tokenizer_synthetic() {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "umm_tts_tok_fixture";
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);

  // Byte 34 (") and byte 92 (\) are printable symbols and must be JSON-escaped.
  auto json_str = [](const std::string& raw) {
    return "\"" + json_escape(raw) + "\"";
  };
  std::string vocab = "{";
  int next_id = 0;
  for (int b = 0; b < 256; ++b) {
    if (next_id) vocab += ",";
    vocab += json_str(byte_symbol(b)) + ":" + std::to_string(next_id++);
  }
  const std::string merged = byte_symbol('a') + byte_symbol('b');
  vocab += "," + json_str(merged) + ":" + std::to_string(next_id++) + "}";
  {
    std::ofstream f(dir / "vocab.json");
    f << vocab;
  }
  {
    std::ofstream f(dir / "merges.txt");
    f << "#version: 0.2\n" << byte_symbol('a') << " " << byte_symbol('b') << "\n";
  }
  {
    std::ofstream f(dir / "tokenizer_config.json");
    f << R"({"added_tokens_decoder":{
      "151644":{"content":"<|im_start|>","special":true},
      "151645":{"content":"<|im_end|>","special":true},
      "151671":{"content":"<tts_pad>","special":true}}})";
  }

  Qwen2BpeTokenizer tok(dir.string());
  CHECK(tok.encode("ab") == std::vector<int>({256}));  // the one merge
  CHECK(tok.encode("a") == std::vector<int>({97}));
  CHECK(tok.encode("12") == std::vector<int>({49, 50}));  // digits never merge
  CHECK(tok.encode("<|im_start|>ab<|im_end|>") ==
        std::vector<int>({151644, 256, 151645}));  // specials stay atomic
  CHECK(tok.encode("<tts_pad>") == std::vector<int>({151671}));
  CHECK(tok.encode("").empty());
  fs::remove_all(dir, ec);
}

static void test_stream_matches() {
  using umminfer::native::stream_matches;
  const std::vector<float> a = {0.0f, 0.5f, -0.5f, 1.0f};
  size_t first_diff = 999;
  CHECK(stream_matches(a, a, &first_diff) && first_diff == a.size());
  std::vector<float> b = a;
  b[2] = 0.25f;
  CHECK(!stream_matches(a, b, &first_diff) && first_diff == 2);
  std::vector<float> short_a(a.begin(), a.end() - 1);
  CHECK(!stream_matches(short_a, a, &first_diff) && first_diff == short_a.size());
  CHECK(stream_matches({}, {}, &first_diff) && first_diff == 0);
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

  PromptPlan clone = build_clone_prompt_plan(zh_ids, kLangChinese);
  CHECK(clone.rows.size() == zh.rows.size());
  CHECK(clone.codec_prefill_ids ==
        std::vector<int>({2154, 2156, 2055, 2157, -1, 2148, 2149}));
  CHECK(clone.rows[7].size() == 2 &&
        clone.rows[7][0].kind == PromptTerm::kXVector &&
        clone.rows[7][1].kind == PromptTerm::kTtsPad);
  for (size_t r = 0; r < zh.rows.size(); ++r) {
    if (r == 7) continue;
    CHECK(clone.rows[r].size() == zh.rows[r].size());
    for (size_t j = 0; j < zh.rows[r].size(); ++j)
      CHECK(clone.rows[r][j].kind == zh.rows[r][j].kind &&
            clone.rows[r][j].id == zh.rows[r][j].id);
  }

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

  const std::vector<float> decoded_mem = read_wav_pcm16_mono_bytes(wav);
  CHECK(decoded_mem.size() == samples.size());
  CHECK(decoded_mem[0] == 0.0f && decoded_mem[1] > 0.499f && decoded_mem[1] < 0.501f);
  CHECK(decoded_mem[6] == -1.0f);
  threw = false;
  try {
    (void)read_wav_pcm16_mono_bytes(wav, 16000);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);

  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "umm_tts_wav_reader.wav";
  auto write_file = [&](const std::string& bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  };
  write_file(wav);
  const std::vector<float> decoded = read_wav_pcm16_mono(path.string());
  CHECK(decoded.size() == samples.size());
  CHECK(decoded[0] == 0.0f && decoded[1] > 0.499f && decoded[1] < 0.501f);
  CHECK(decoded[2] < -0.499f && decoded[2] > -0.501f && decoded[6] == -1.0f);
  threw = false;
  try {
    (void)read_wav_pcm16_mono(path.string(), 16000);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  write_file(wav.substr(0, wav.size() - 1));
  threw = false;
  try {
    (void)read_wav_pcm16_mono(path.string());
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  write_file(wav + "x");
  threw = false;
  try {
    (void)read_wav_pcm16_mono(path.string());
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  std::error_code ec;
  std::filesystem::remove(path, ec);

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

static void test_decode_helpers() {
  // Unroll schedules for decode GEMV and dual-weight SwiGLU GEMV.
  CHECK(tts_gemv_unroll_for(1) == 4);
  CHECK(tts_gemv_unroll_for(2) == 4);
  CHECK(tts_gemv_unroll_for(3) == 1);
  CHECK(tts_gemv_unroll_for(8) == 1);
  CHECK(tts_swiglu_unroll_for(1) == 2);
  CHECK(tts_swiglu_unroll_for(2) == 2);
  CHECK(tts_swiglu_unroll_for(4) == 1);

  // Bijective 3-way QKV row mapping (2048 + 1024 + 1024 = 4096).
  const int qkv_n[3] = {2048, 1024, 1024};
  CHECK(multi_gemv_locate_row(-1, qkv_n, 3).proj == -1);
  CHECK(multi_gemv_locate_row(4096, qkv_n, 3).proj == -1);
  for (int g = 0; g < 4096; ++g) {
    const MultiGemvRowLoc loc = multi_gemv_locate_row(g, qkv_n, 3);
    if (g < 2048) {
      CHECK(loc.proj == 0 && loc.local_row == g);
    } else if (g < 3072) {
      CHECK(loc.proj == 1 && loc.local_row == g - 2048);
    } else {
      CHECK(loc.proj == 2 && loc.local_row == g - 3072);
    }
  }

  // Exact peak scratch size for codec chunks.
  CHECK(codec_peak_scratch_elems(1) == 1920u * 96u);
  CHECK(codec_peak_scratch_elems(32) == 32u * 1920u * 96u);

  // Bit-for-bit equivalence: stateful causal Conv1d (including chunk_len < pad
  // for dilation=9, pad=54) vs one-shot causal Conv1d.
  auto verify_stateful_conv1d = [](int K, int dilation,
                                   const std::vector<int>& chunks) {
    int total = 0;
    for (int c : chunks) total += c;
    const int pad = (K - 1) * dilation;
    std::vector<float> x(total), w(K);
    for (int t = 0; t < total; ++t) x[t] = std::sin(0.17f * t + 0.3f);
    for (int kk = 0; kk < K; ++kk) w[kk] = 0.25f * (kk + 1) - 0.5f;

    // One-shot reference.
    std::vector<float> ref(total, 0.0f);
    for (int t = 0; t < total; ++t) {
      float acc = 0.0f;
      for (int kk = 0; kk < K; ++kk) {
        const int src = t - pad + kk * dilation;
        if (src >= 0 && src < total) acc += x[src] * w[kk];
      }
      ref[t] = acc;
    }

    // Stateful chunked execution using decode_helpers index rules.
    std::vector<float> hist(pad, 0.0f);
    std::vector<float> got;
    int off = 0;
    for (int L : chunks) {
      for (int t = 0; t < L; ++t) {
        float acc = 0.0f;
        for (int kk = 0; kk < K; ++kk) {
          const CausalSrcLoc loc = causal_conv_src_lookup(t, kk, dilation, pad);
          const float xv = loc.from_hist ? hist[loc.idx] : x[off + loc.idx];
          acc += xv * w[kk];
        }
        got.push_back(acc);
      }
      std::vector<float> next_hist(pad, 0.0f);
      for (int j = 0; j < pad; ++j) {
        const CausalSrcLoc src = causal_conv_hist_update_src(j, pad, L);
        next_hist[j] = src.from_hist ? hist[src.idx] : x[off + src.idx];
      }
      hist.swap(next_hist);
      off += L;
    }
    CHECK(got.size() == ref.size());
    for (size_t i = 0; i < ref.size(); ++i) CHECK(got[i] == ref[i]);
  };

  verify_stateful_conv1d(3, 1, {2, 8, 8, 1, 5, 32, 3});
  verify_stateful_conv1d(7, 1, {2, 3, 1, 8, 16, 7});
  verify_stateful_conv1d(7, 3, {2, 8, 8, 8, 12, 5});   // pad=18 > chunk 2, 8
  verify_stateful_conv1d(7, 9, {2, 8, 8, 8, 8, 8, 8, 32, 4});  // pad=54 > chunk 2, 8, 32

  // Bit-for-bit equivalence: stateful ConvTranspose1d (K = 2*stride) vs one-shot.
  for (int stride : {2, 3, 4, 5, 8}) {
    const int K = (stride == 2) ? 2 : (2 * stride);
    const std::vector<int> chunks = {2, 8, 8, 1, 5};
    int total = 0;
    for (int c : chunks) total += c;
    std::vector<float> x(total), w(K);
    for (int t = 0; t < total; ++t) x[t] = std::cos(0.23f * t - 0.4f);
    for (int kk = 0; kk < K; ++kk) w[kk] = 0.1f * (kk + 1) - 0.3f;

    std::vector<float> ref(total * stride, 0.0f);
    for (int o = 0; o < total * stride; ++o) {
      float acc = 0.0f;
      for (int kk = 0; kk < K; ++kk) {
        const int rem = o - kk;
        if (rem < 0 || rem % stride != 0) continue;
        const int t = rem / stride;
        if (t < total) acc += x[t] * w[kk];
      }
      ref[o] = acc;
    }

    float hist = 0.0f;
    std::vector<float> got;
    int off = 0;
    for (int L : chunks) {
      for (int o = 0; o < L * stride; ++o) {
        const int t_local = o / stride;
        const int p = o - t_local * stride;
        float acc = 0.0f;
        for (int step = 0, kk = p; kk < K; ++step, kk += stride) {
          const CausalSrcLoc loc = transconv_step_src(t_local, step);
          const float xv = loc.from_hist ? hist : x[off + loc.idx];
          acc += xv * w[kk];
        }
        got.push_back(acc);
      }
      hist = x[off + L - 1];
      off += L;
    }
    CHECK(got.size() == ref.size());
    for (size_t i = 0; i < ref.size(); ++i) CHECK(got[i] == ref[i]);
  }

  // Rolling KV cache equivalence across 600 frames (window=72, cap=192):
  // verify non-overlapping compaction and exact global key coverage [max(0,t-71), t].
  {
    constexpr int kWin = 72;
    constexpr int kCap = 192;
    std::vector<int> kv(kCap, -1);
    int cached = 0;
    int global_t = 0;
    std::vector<int> schedule = {2};
    while (static_cast<int>(schedule.size()) < 80) schedule.push_back(8);
    schedule.push_back(32);
    for (int chunk : schedule) {
      const RollingKvPlan plan = rolling_kv_plan(cached, chunk, kWin, kCap);
      if (plan.need_shift) {
        // Non-overlapping compaction guarantee.
        CHECK(plan.shift_src >= plan.keep_len);
        for (int i = 0; i < plan.keep_len; ++i) kv[i] = kv[plan.shift_src + i];
        cached = plan.keep_len;
      }
      CHECK(cached + chunk <= kCap);
      for (int qi = 0; qi < chunk; ++qi) kv[cached + qi] = global_t + qi;
      for (int qi = 0; qi < chunk; ++qi) {
        const int t = global_t + qi;
        const int lo = std::max(0, cached + qi - kWin + 1);
        const int hi = cached + qi + 1;
        const int want_lo = std::max(0, t - kWin + 1);
        const int want_hi = t + 1;
        CHECK(hi - lo == want_hi - want_lo);
        for (int k = 0; k < hi - lo; ++k) CHECK(kv[lo + k] == want_lo + k);
      }
      cached += chunk;
      global_t += chunk;
    }
  }
}

int main(int argc, char** argv) {
  const char* env = std::getenv("UMM_TTS_TEST_DATA");
  std::string data_dir = argc > 1 ? argv[1] : (env ? env : "");
  if (!data_dir.empty()) {
    test_tokenizer(data_dir);
  } else {
    std::cout << "SKIP test_tokenizer golden (no MODEL_DATA_DIR / UMM_TTS_TEST_DATA)\n";
  }
  test_tokenizer_synthetic();
  test_stream_matches();
  test_prompt();
  test_wav();
  test_safetensors();
  test_decode_helpers();
  std::cout << "all tts host tests passed\n";
  return 0;
}
