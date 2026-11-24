// SPDX-License-Identifier: Apache-2.0
// Assert-based host tests for the Qwen3-ASR host slice: WAV parsing, Whisper
// log-mel frontend shape, byte-level tokenizer decode, single-file and sharded
// safetensors loading, and config.json parsing. No CUDA and no model weights:
// every fixture is written to a temp directory at run time.
#include "feature_frontend.hpp"
#include "model_config.hpp"
#include "safetensors.hpp"
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
  const auto custom = QwenTokenizer::build_asr_prompt(2, 7, 8, 9);
  check(custom[3] == 7 && custom[4] == 8 && custom[5] == 8 && custom[6] == 9,
        "ASR prompt uses config-supplied audio token ids");
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

// Qwen3-ASR config.json as published. Keys sit on their own line so the
// negative tests can rewrite exactly one value.
const char* kConfig06b = R"({
  "model_type": "qwen3_asr",
  "thinker_config": {
    "audio_start_token_id": 151669,
    "audio_end_token_id": 151670,
    "audio_token_id": 151676,
    "audio_config": {
      "num_mel_bins": 128,
      "d_model": 896,
      "encoder_attention_heads": 14,
      "encoder_ffn_dim": 3584,
      "encoder_layers": 18,
      "downsample_hidden_size": 480,
      "output_dim": 1024,
      "n_window": 50,
      "n_window_infer": 800
    },
    "text_config": {
      "num_hidden_layers": 28,
      "hidden_size": 1024,
      "num_attention_heads": 16,
      "num_key_value_heads": 8,
      "head_dim": 128,
      "intermediate_size": 3072,
      "vocab_size": 151936,
      "rms_norm_eps": 1e-06,
      "rope_theta": 1000000
    }
  }
})";

// Qwen/Qwen3-ASR-1.7B config.json as published on ModelScope (sha256
// 2e74a751548b8ad7d7526d29365ad8144c345d8b412b1152d25dc6698452712f). Only the
// keys AsrModelConfig reads are kept; every value below is the published one.
const char* kConfig17b = R"({
  "model_type": "qwen3_asr",
  "thinker_config": {
    "audio_start_token_id": 151669,
    "audio_end_token_id": 151670,
    "audio_token_id": 151676,
    "audio_config": {
      "num_mel_bins": 128,
      "d_model": 1024,
      "encoder_attention_heads": 16,
      "encoder_ffn_dim": 4096,
      "encoder_layers": 24,
      "downsample_hidden_size": 480,
      "output_dim": 2048,
      "n_window": 50,
      "n_window_infer": 800
    },
    "text_config": {
      "num_hidden_layers": 28,
      "hidden_size": 2048,
      "num_attention_heads": 16,
      "num_key_value_heads": 8,
      "head_dim": 128,
      "intermediate_size": 6144,
      "vocab_size": 151936,
      "rms_norm_eps": 1e-06,
      "rope_theta": 1000000
    }
  }
})";

// Replaces one literal config snippet; a typo in the fixture itself fails the
// test instead of silently passing an unrelated edit.
std::string with_value(std::string json, const std::string& from, const std::string& to) {
  const size_t pos = json.find(from);
  if (pos == std::string::npos) {
    check(false, "config fixture contains the edited value");
    return json;
  }
  return json.replace(pos, from.size(), to);
}

void test_model_config_06b() {
  using namespace umminfer::native;
  const AsrModelConfig cfg = AsrModelConfig::from_json_text(kConfig06b);
  const AsrModelConfig want = AsrModelConfig::preset_06b();
  auto same = [&](long long got, long long expected, const char* what) {
    check(got == expected, what);
  };
  same(cfg.num_mel_bins, want.num_mel_bins, "0.6B num_mel_bins");
  same(cfg.audio_d, want.audio_d, "0.6B audio d_model");
  same(cfg.audio_heads, want.audio_heads, "0.6B audio heads");
  same(cfg.audio_ffn, want.audio_ffn, "0.6B audio ffn");
  same(cfg.audio_layers, want.audio_layers, "0.6B audio layers");
  same(cfg.audio_down, want.audio_down, "0.6B audio downsample size");
  same(cfg.audio_out, want.audio_out, "0.6B audio output_dim");
  same(cfg.n_window, want.n_window, "0.6B n_window");
  same(cfg.n_window_infer, want.n_window_infer, "0.6B n_window_infer");
  same(cfg.text_layers, want.text_layers, "0.6B text layers");
  same(cfg.text_h, want.text_h, "0.6B text hidden_size");
  same(cfg.text_q_heads, want.text_q_heads, "0.6B text q heads");
  same(cfg.text_kv_heads, want.text_kv_heads, "0.6B text kv heads");
  same(cfg.text_head_dim, want.text_head_dim, "0.6B text head_dim");
  same(cfg.text_inter, want.text_inter, "0.6B text intermediate_size");
  same(cfg.vocab, want.vocab, "0.6B vocab_size");
  same(cfg.audio_start_id, want.audio_start_id, "0.6B audio_start_token_id");
  same(cfg.audio_end_id, want.audio_end_id, "0.6B audio_end_token_id");
  same(cfg.audio_token_id, want.audio_token_id, "0.6B audio_token_id");
  check(cfg.rms_norm_eps == want.rms_norm_eps, "0.6B rms_norm_eps");
  check(cfg.rope_theta == want.rope_theta, "0.6B rope_theta");

  check(cfg.audio_head_dim() == 64, "0.6B audio head_dim 64");
  check(cfg.window_len() == 104, "0.6B attention window 104");
  check(cfg.audio_perm_row() == 7680, "0.6B conv permute row 7680");
  check(cfg.max_audio_tokens() == 390, "0.6B max audio tokens 390");
  check(cfg.au_norm_elements() == 390 * 7680, "0.6B au_norm element math");
  check(cfg.kv_cache_elements() == 28u * 2u * 1024u * 8u * 128u, "0.6B KV cache size");
  check(cfg.estimate_workspace_bytes() > 0, "0.6B workspace estimate positive");

  // from_model_dir reads <dir>/config.json, the same path the engine uses.
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "umm_asr_config_fixture";
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);
  {
    std::ofstream f(dir / "config.json");
    f << kConfig06b;
  }
  const AsrModelConfig from_dir = AsrModelConfig::from_model_dir(dir.string());
  check(from_dir.audio_d == want.audio_d, "from_model_dir reads config.json");
  check(throws_invalid([&] { (void)AsrModelConfig::from_model_dir((dir / "nope").string()); }),
        "from_model_dir rejects a directory without config.json");
  fs::remove_all(dir, ec);
}

void test_model_config_17b() {
  using namespace umminfer::native;
  const AsrModelConfig cfg = AsrModelConfig::from_json_text(kConfig17b);
  check(cfg.num_mel_bins == 128, "1.7B num_mel_bins 128");
  check(cfg.audio_layers == 24, "1.7B audio layers 24");
  check(cfg.audio_d == 1024, "1.7B audio d_model 1024");
  check(cfg.audio_heads == 16, "1.7B audio heads 16");
  check(cfg.audio_ffn == 4096, "1.7B audio ffn 4096");
  check(cfg.audio_down == 480, "1.7B downsample_hidden_size 480");
  check(cfg.audio_out == 2048, "1.7B audio output_dim 2048");
  check(cfg.n_window == 50, "1.7B n_window 50");
  check(cfg.n_window_infer == 800, "1.7B n_window_infer 800");
  check(cfg.text_h == 2048, "1.7B text hidden_size 2048");
  check(cfg.text_inter == 6144, "1.7B text intermediate_size 6144");
  check(cfg.text_layers == 28, "1.7B text layers 28");
  check(cfg.text_q_heads == 16, "1.7B text q heads 16");
  check(cfg.text_kv_heads == 8, "1.7B text kv heads 8");
  check(cfg.text_head_dim == 128, "1.7B text head_dim 128");
  check(cfg.vocab == 151936, "1.7B vocab_size 151936");
  check(cfg.rms_norm_eps == 1e-6f, "1.7B rms_norm_eps 1e-6");
  check(cfg.rope_theta == 1000000.0f, "1.7B rope_theta 1e6");
  check(cfg.audio_start_id == 151669, "1.7B audio_start_token_id 151669");
  check(cfg.audio_end_id == 151670, "1.7B audio_end_token_id 151670");
  check(cfg.audio_token_id == 151676, "1.7B audio_token_id 151676");
  // audio_out == text_h is what lets the projector rows be copied straight
  // into the decoder hidden state; from_json_text() already enforced it.
  check(cfg.audio_out == cfg.text_h, "1.7B projector width equals decoder width");
  check(cfg.audio_head_dim() == 64, "1.7B audio head_dim 64");
  check(cfg.window_len() == 104, "1.7B attention window 104");
  check(cfg.audio_perm_row() == 480 * 16, "1.7B conv permute row 7680");
  check(cfg.au_norm_elements() == 390 * 7680, "1.7B au_norm element math");

  const size_t small = AsrModelConfig::preset_06b().estimate_workspace_bytes();
  const size_t big = cfg.estimate_workspace_bytes();
  check(big > small, "1.7B workspace estimate exceeds 0.6B");
  // The decoder shape is identical across the two sizes, so the KV cache is
  // the one buffer that does not grow.
  check(cfg.kv_cache_elements() == AsrModelConfig::preset_06b().kv_cache_elements(),
        "1.7B KV cache equals 0.6B (same decoder shape)");
  // Overflow guard: every element count and the byte total must stay exact
  // when widened to size_t.
  check(cfg.au_qkv_elements() == 390 * 1024, "1.7B au_qkv element math");
  check(cfg.conv1_elements() == 30 * 480 * 64 * 50, "1.7B conv1 element math");
  check(big / sizeof(float) > cfg.kv_cache_elements(),
        "1.7B workspace estimate not truncated or wrapped");
}

void test_model_config_rejects_invalid() {
  using namespace umminfer::native;
  const std::string base = kConfig06b;
  auto rejects = [&](const std::string& json, const char* what) {
    check(throws_invalid([&] { (void)AsrModelConfig::from_json_text(json); }), what);
  };
  rejects(with_value(base, "\"d_model\": 896", "\"d_model\": 900"),
          "rejects audio width not divisible by heads");
  rejects(with_value(base, "\"output_dim\": 1024", "\"output_dim\": 512"),
          "rejects audio output_dim != text hidden_size");
  rejects(with_value(base, "\"num_mel_bins\": 128", "\"num_mel_bins\": 80"),
          "rejects num_mel_bins != 128");
  rejects(with_value(base, "\"num_attention_heads\": 16", "\"num_attention_heads\": 15"),
          "rejects text q heads not divisible by kv heads");
  rejects(with_value(base, "\"encoder_layers\": 18", "\"encoder_layers\": 0"),
          "rejects non-positive audio layers");
  rejects(with_value(base, "\"n_window\": 50", "\"n_window\": 0"),
          "rejects non-positive n_window");
  rejects(with_value(base, "\"n_window_infer\": 800", "\"n_window_infer\": 810"),
          "rejects n_window_infer not a multiple of 2 * n_window");
  rejects(with_value(base, "\"head_dim\": 128", "\"head_dim\": 127"),
          "rejects odd text head_dim");
  rejects("{\"model_type\": \"qwen3_asr\"}", "rejects config without thinker_config");
  rejects(with_value(base, "\"n_window\": 50", "\"n_window\": \"wide\""),
          "rejects non-numeric field");
  rejects("not json", "rejects malformed JSON");
}

// ---- safetensors ------------------------------------------------------------

struct FakeTensor {
  std::string name;
  std::vector<size_t> shape;
  std::string data;  // BF16 payload, 2 bytes per element
};

std::string make_shard(const std::vector<FakeTensor>& tensors) {
  using namespace umminfer::native;
  std::vector<std::pair<std::string, TensorInfo>> infos;
  std::string data;
  for (const auto& t : tensors) {
    TensorInfo info;
    info.dtype = "BF16";
    info.shape = t.shape;
    info.byte_offset = data.size();
    info.byte_size = t.data.size();
    infos.emplace_back(t.name, info);
    data += t.data;
  }
  return SafeTensors::write(infos, data);
}

std::string bf16_payload(std::initializer_list<int> values) {
  std::string out;
  for (int v : values) {
    out.push_back(static_cast<char>(v & 0xff));
    out.push_back(static_cast<char>((v >> 8) & 0xff));
  }
  return out;
}

namespace fs = std::filesystem;

// Temp model directory that removes itself.
struct ModelDir {
  fs::path path;
  explicit ModelDir(const char* leaf) {
    path = fs::temp_directory_path() / leaf;
    std::error_code ec;
    fs::remove_all(path, ec);
    fs::create_directories(path, ec);
  }
  ~ModelDir() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};

void write_bytes(const fs::path& path, const std::string& bytes) {
  std::ofstream f(path, std::ios::binary);
  f << bytes;
}

void test_safetensors_single_file() {
  using namespace umminfer::native;
  const std::string payload = bf16_payload({0x0400, 0x0401, 0x0402});
  const SafeTensors st(make_shard({{"a.weight", {3}, payload}}));
  check(st.shard_count() == 1, "single file has one shard");
  check(st.info("a.weight").byte_size == payload.size(), "single file tensor size");
  check(st.tensor_bytes("a.weight") == std::string_view(payload), "single file tensor bytes");
  check(throws_invalid([&] { (void)st.info("missing.weight"); }), "missing tensor rejected");
}

void test_safetensors_sharded_dir() {
  using namespace umminfer::native;
  ModelDir dir("umm_asr_sharded_fixture");
  const std::string a = bf16_payload({1, 2});
  const std::string b = bf16_payload({3, 4, 5, 6});
  write_bytes(dir.path / "model-00001-of-00002.safetensors",
              make_shard({{"a.weight", {2}, a}}));
  write_bytes(dir.path / "model-00002-of-00002.safetensors",
              make_shard({{"b.weight", {4}, b}}));
  write_bytes(dir.path / "model.safetensors.index.json", R"({
    "metadata": {"format": "pt"},
    "weight_map": {
      "a.weight": "model-00001-of-00002.safetensors",
      "b.weight": "model-00002-of-00002.safetensors"
    }
  })");

  const SafeTensors st = SafeTensors::from_model_dir(dir.path.string());
  check(st.shard_count() == 2, "index loads both shards");
  check(st.shard_name(0) == "model-00001-of-00002.safetensors", "shard 0 name");
  check(st.info("a.weight").shard == 0, "a.weight resolved to shard 0");
  check(st.info("b.weight").shard == 1, "b.weight resolved to shard 1");
  check(st.tensor_bytes("a.weight") == std::string_view(a), "shard 0 tensor bytes");
  check(st.tensor_bytes("b.weight") == std::string_view(b), "shard 1 tensor bytes");
  check(st.data_region_bytes(0) >= a.size(), "shard 0 region covers its tensors");

  // Both forms present: model.safetensors wins.
  write_bytes(dir.path / "model.safetensors", make_shard({{"c.weight", {2}, a}}));
  const SafeTensors single = SafeTensors::from_model_dir(dir.path.string());
  check(single.shard_count() == 1, "model.safetensors preferred over the index");
  check(single.info("c.weight").shard == 0, "single-file tensor loaded");
}

void test_safetensors_shard_rejections() {
  using namespace umminfer::native;
  ModelDir dir("umm_asr_shard_reject");
  const std::string payload = bf16_payload({1, 2});
  const auto write_index = [&](const std::string& json) {
    write_bytes(dir.path / "model.safetensors.index.json", json);
  };
  const auto load = [&] { return SafeTensors::from_model_dir(dir.path.string()); };
  const std::string shard1 = dir.path.string() + "/model-00001-of-00002.safetensors";
  const std::string shard2 = dir.path.string() + "/model-00002-of-00002.safetensors";

  // Neither checkout form exists.
  check(throws_invalid([&] { (void)load(); }), "directory without checkpoint rejected");

  // The index names a shard that is not on disk.
  write_index(R"({"weight_map": {"a.weight": "model-00009-of-00009.safetensors"}})");
  check(throws_invalid([&] { (void)load(); }), "missing shard file rejected");

  // A shard name must be a plain file name inside the model directory.
  for (const char* bad : {"../model-00001-of-00002.safetensors",
                          "/tmp/model-00001-of-00002.safetensors",
                          "sub/model-00001-of-00002.safetensors", "-rf", ".", ""}) {
    write_index(std::string(R"({"weight_map": {"a.weight": ")") + bad + R"("}})");
    check(throws_invalid([&] { (void)load(); }), "non-plain shard name rejected");
  }

  // Baseline: two shards, two listed tensors, loads clean.
  write_bytes(shard1, make_shard({{"a.weight", {2}, payload}}));
  write_bytes(shard2, make_shard({{"b.weight", {2}, payload}}));
  write_index(R"({"weight_map": {
      "a.weight": "model-00001-of-00002.safetensors",
      "b.weight": "model-00002-of-00002.safetensors"}})");
  check(load().shard_count() == 2, "consistent two-shard index loads");

  // A shard holds a tensor the index does not list.
  write_bytes(shard2, make_shard({{"b.weight", {2}, payload}, {"c.weight", {2}, payload}}));
  check(throws_invalid([&] { (void)load(); }), "unlisted shard tensor rejected");

  // The same tensor appears in two shards.
  write_bytes(shard2, make_shard({{"a.weight", {2}, payload}}));
  check(throws_invalid([&] { (void)load(); }), "tensor duplicated across shards rejected");

  // The index lists a tensor that no shard holds.
  write_index(R"({"weight_map": {
      "a.weight": "model-00001-of-00002.safetensors",
      "ghost.weight": "model-00001-of-00002.safetensors"}})");
  check(throws_invalid([&] { (void)load(); }), "index entry missing from shard rejected");

  // A tensor sits in a shard other than the one the index names.
  write_bytes(shard2, make_shard({{"b.weight", {2}, payload}, {"c.weight", {2}, payload}}));
  write_index(R"({"weight_map": {
      "a.weight": "model-00001-of-00002.safetensors",
      "b.weight": "model-00001-of-00002.safetensors",
      "c.weight": "model-00002-of-00002.safetensors"}})");
  check(throws_invalid([&] { (void)load(); }), "tensor mapped to the wrong shard rejected");
}

}  // namespace

int main() {
  test_wav_valid();
  test_wav_invalid();
  test_mel_frontend();
  test_tokenizer_prompt_and_decode();
  test_tokenizer_decode();
  test_model_config_06b();
  test_model_config_17b();
  test_model_config_rejects_invalid();
  test_safetensors_single_file();
  test_safetensors_sharded_dir();
  test_safetensors_shard_rejections();
  if (g_failures) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("all asr host checks passed\n");
  return 0;
}
