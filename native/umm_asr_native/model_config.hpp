// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace umminfer::native {

// Dimensions of one Qwen3-ASR checkpoint, read from <model_dir>/config.json.
// Every CUDA buffer, weight shape check and layer loop in the engine derives
// from this struct, so a checkpoint of another size only needs its own
// config.json. Header-only: both native/CMakeLists.txt and
// native/stage_cli/CMakeLists.txt pick it up through the engine include.
struct AsrModelConfig {
  // Audio tower (Whisper-style AuT encoder).
  int num_mel_bins = 0;
  int audio_d = 0;
  int audio_heads = 0;
  int audio_ffn = 0;
  int audio_layers = 0;
  int audio_down = 0;  // downsample_hidden_size; conv channels and permute row
  int audio_out = 0;   // projector width, must equal text_h
  int n_window = 0;
  int n_window_infer = 0;
  float audio_norm_eps = 1e-5f;

  // Text decoder (Qwen3 dense).
  int text_layers = 0;
  int text_h = 0;
  int text_q_heads = 0;
  int text_kv_heads = 0;
  int text_head_dim = 0;
  int text_inter = 0;
  int vocab = 0;
  float rms_norm_eps = 0.0f;
  float rope_theta = 0.0f;

  // Special token ids of the ASR prompt.
  int audio_start_id = 0;
  int audio_end_id = 0;
  int audio_token_id = 0;

  // Operational capacity, not read from config.json.
  int max_seq = 1024;

  // The frontend always emits a padded 30 s chunk, 128 bins x 3000 frames,
  // cut into 100-frame pieces. The conv stack halves twice and floors once:
  // 100 -> 64x50 -> 32x25 -> 16x13, so one chunk yields 13 tokens.
  static constexpr int kMelFrames = 3000;
  static constexpr int kChunkFrames = 100;
  static constexpr int kConvTokensPerChunk = 13;
  static constexpr int kConv1H = 64, kConv1W = 50;
  static constexpr int kConv2H = 32, kConv2W = 25;
  static constexpr int kConv3H = 16, kConv3W = 13;

  // 0.6B values, fixed for byte-identical behaviour and used by the tests to
  // cross-check a parsed config.json.
  static AsrModelConfig preset_06b() {
    AsrModelConfig c;
    c.num_mel_bins = 128;
    c.audio_d = 896;
    c.audio_heads = 14;
    c.audio_ffn = 3584;
    c.audio_layers = 18;
    c.audio_down = 480;
    c.audio_out = 1024;
    c.n_window = 50;
    c.n_window_infer = 800;
    c.audio_norm_eps = 1e-5f;
    c.text_layers = 28;
    c.text_h = 1024;
    c.text_q_heads = 16;
    c.text_kv_heads = 8;
    c.text_head_dim = 128;
    c.text_inter = 3072;
    c.vocab = 151936;
    c.rms_norm_eps = 1e-6f;
    c.rope_theta = 1000000.0f;
    c.audio_start_id = 151669;
    c.audio_end_id = 151670;
    c.audio_token_id = 151676;
    c.max_seq = 1024;
    return c;
  }

  static AsrModelConfig from_json_text(const std::string& json_text) {
    nlohmann::json root;
    try {
      root = nlohmann::json::parse(json_text);
    } catch (const nlohmann::json::exception& e) {
      throw std::invalid_argument(std::string("config.json is not valid JSON: ") + e.what());
    }
    auto obj = [](const nlohmann::json& parent, const char* key) -> const nlohmann::json& {
      if (!parent.is_object() || !parent.contains(key))
        throw std::invalid_argument(std::string("config.json missing section: ") + key);
      const nlohmann::json& v = parent.at(key);
      if (!v.is_object())
        throw std::invalid_argument(std::string("config.json section is not an object: ") + key);
      return v;
    };
    auto num = [](const nlohmann::json& j, const char* key) {
      if (!j.contains(key))
        throw std::invalid_argument(std::string("config.json missing field: ") + key);
      const nlohmann::json& v = j.at(key);
      if (!v.is_number())
        throw std::invalid_argument(std::string("config.json field is not a number: ") + key);
      if (v.is_number_integer())
        return static_cast<long long>(v.get<long long>());
      const double d = v.get<double>();
      if (d != static_cast<double>(static_cast<long long>(d)))
        throw std::invalid_argument(std::string("config.json field is not integral: ") + key);
      return static_cast<long long>(d);
    };
    auto as_int = [&num](const nlohmann::json& j, const char* key) {
      const long long v = num(j, key);
      if (v < -2147483648LL || v > 2147483647LL)
        throw std::invalid_argument(std::string("config.json field out of int range: ") + key);
      return static_cast<int>(v);
    };
    auto as_float = [](const nlohmann::json& j, const char* key) {
      if (!j.contains(key))
        throw std::invalid_argument(std::string("config.json missing field: ") + key);
      const nlohmann::json& v = j.at(key);
      if (!v.is_number())
        throw std::invalid_argument(std::string("config.json field is not a number: ") + key);
      return v.get<float>();
    };

    const nlohmann::json& thinker = obj(root, "thinker_config");
    const nlohmann::json& audio = obj(thinker, "audio_config");
    const nlohmann::json& text = obj(thinker, "text_config");

    AsrModelConfig c;
    c.num_mel_bins = as_int(audio, "num_mel_bins");
    c.audio_d = as_int(audio, "d_model");
    c.audio_heads = as_int(audio, "encoder_attention_heads");
    c.audio_ffn = as_int(audio, "encoder_ffn_dim");
    c.audio_layers = as_int(audio, "encoder_layers");
    c.audio_down = as_int(audio, "downsample_hidden_size");
    c.audio_out = as_int(audio, "output_dim");
    c.n_window = as_int(audio, "n_window");
    c.n_window_infer = as_int(audio, "n_window_infer");

    c.text_layers = as_int(text, "num_hidden_layers");
    c.text_h = as_int(text, "hidden_size");
    c.text_q_heads = as_int(text, "num_attention_heads");
    c.text_kv_heads = as_int(text, "num_key_value_heads");
    c.text_head_dim = as_int(text, "head_dim");
    c.text_inter = as_int(text, "intermediate_size");
    c.vocab = as_int(text, "vocab_size");
    c.rms_norm_eps = as_float(text, "rms_norm_eps");
    c.rope_theta = as_float(text, "rope_theta");

    c.audio_start_id = as_int(thinker, "audio_start_token_id");
    c.audio_end_id = as_int(thinker, "audio_end_token_id");
    c.audio_token_id = as_int(thinker, "audio_token_id");

    c.validate();
    return c;
  }

  static AsrModelConfig from_model_dir(const std::string& model_dir) {
    const std::string path = model_dir + "/config.json";
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::invalid_argument("cannot open " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return from_json_text(ss.str());
  }

  // Throws std::invalid_argument for any dimension the engine cannot run.
  void validate() const {
    auto positive = [](int v, const char* what) {
      if (v <= 0)
        throw std::invalid_argument(std::string("config.json ") + what +
                                    " must be positive, got " + std::to_string(v));
    };
    positive(num_mel_bins, "num_mel_bins");
    positive(audio_d, "audio_config.d_model");
    positive(audio_heads, "audio_config.encoder_attention_heads");
    positive(audio_ffn, "audio_config.encoder_ffn_dim");
    positive(audio_layers, "audio_config.encoder_layers");
    positive(audio_down, "audio_config.downsample_hidden_size");
    positive(audio_out, "audio_config.output_dim");
    positive(n_window, "audio_config.n_window");
    positive(n_window_infer, "audio_config.n_window_infer");
    positive(text_layers, "text_config.num_hidden_layers");
    positive(text_h, "text_config.hidden_size");
    positive(text_q_heads, "text_config.num_attention_heads");
    positive(text_kv_heads, "text_config.num_key_value_heads");
    positive(text_head_dim, "text_config.head_dim");
    positive(text_inter, "text_config.intermediate_size");
    positive(vocab, "text_config.vocab_size");
    positive(max_seq, "max_seq");

    if (num_mel_bins != 128)
      throw std::invalid_argument("config.json num_mel_bins must be 128, got " +
                                  std::to_string(num_mel_bins));
    if (audio_d % audio_heads != 0)
      throw std::invalid_argument("config.json audio d_model " + std::to_string(audio_d) +
                                  " is not divisible by encoder_attention_heads " +
                                  std::to_string(audio_heads));
    if (text_q_heads % text_kv_heads != 0)
      throw std::invalid_argument("config.json num_attention_heads " +
                                  std::to_string(text_q_heads) +
                                  " is not divisible by num_key_value_heads " +
                                  std::to_string(text_kv_heads));
    // The audio projector output is copied straight into the decoder hidden
    // state rows, so both widths must agree.
    if (audio_out != text_h)
      throw std::invalid_argument("config.json audio output_dim " +
                                  std::to_string(audio_out) +
                                  " must equal text hidden_size " + std::to_string(text_h));
    if (text_head_dim % 2 != 0)
      throw std::invalid_argument("config.json text head_dim must be even, got " +
                                  std::to_string(text_head_dim));
    if (n_window_infer % (2 * n_window) != 0)
      throw std::invalid_argument("config.json n_window_infer " +
                                  std::to_string(n_window_infer) +
                                  " is not a multiple of 2 * n_window " +
                                  std::to_string(2 * n_window));
    if (!(rms_norm_eps > 0.0f))
      throw std::invalid_argument("config.json rms_norm_eps must be positive");
    if (!(rope_theta > 0.0f))
      throw std::invalid_argument("config.json rope_theta must be positive");
  }

  int audio_head_dim() const { return audio_d / audio_heads; }
  // conv_permute lays each chunk out as [13][audio_down * 16].
  int audio_perm_row() const { return audio_down * kConv3H; }
  int max_chunks() const { return (kMelFrames + kChunkFrames - 1) / kChunkFrames; }
  int max_audio_tokens() const { return max_chunks() * kConvTokensPerChunk; }
  // HF window_aftercnn = 13 * (n_window_infer / (n_window * 2)); 104 for
  // both 0.6B and 1.7B (800 / (50 * 2) = 8).
  int window_len() const { return kConvTokensPerChunk * (n_window_infer / (2 * n_window)); }

  // d_au_norm holds the permuted conv output, the layer-norm output, the
  // attention output and the projector output, so it is sized for the widest.
  int au_norm_row() const { return std::max({audio_perm_row(), audio_d, audio_out}); }

  int mel_elements() const { return num_mel_bins * kMelFrames; }
  int inv_freq_elements() const { return text_head_dim / 2; }
  int logits_elements() const { return vocab; }
  int text_row_elements() const { return max_seq * text_h; }
  int text_q_elements() const { return max_seq * text_q_heads * text_head_dim; }
  int text_kv_elements() const { return max_seq * text_kv_heads * text_head_dim; }
  int text_ffn_elements() const { return max_seq * text_inter; }
  int audio_io_elements() const { return max_seq * std::max(audio_d, audio_out); }
  int au_norm_elements() const {
    return static_cast<int>(static_cast<size_t>(max_audio_tokens()) *
                            static_cast<size_t>(au_norm_row()));
  }
  int au_qkv_elements() const { return max_audio_tokens() * audio_d; }
  int au_ffn_elements() const { return max_audio_tokens() * audio_ffn; }
  int conv_in_elements() const { return max_chunks() * num_mel_bins * kChunkFrames; }
  int conv1_elements() const { return max_chunks() * audio_down * kConv1H * kConv1W; }
  int conv2_elements() const { return max_chunks() * audio_down * kConv2H * kConv2W; }
  int conv3_elements() const { return max_chunks() * audio_down * kConv3H * kConv3W; }
  int pe_elements() const { return kConvTokensPerChunk * audio_d; }
  size_t kv_cache_elements() const {
    return static_cast<size_t>(text_layers) * 2 * static_cast<size_t>(max_seq) *
           static_cast<size_t>(text_kv_heads) * static_cast<size_t>(text_head_dim);
  }

  // Device scratch footprint of one engine instance, excluding weights.
  size_t estimate_workspace_bytes() const {
    size_t bf16 = 0;
    bf16 += static_cast<size_t>(text_row_elements());   // hidden
    bf16 += static_cast<size_t>(text_row_elements());   // norm
    bf16 += static_cast<size_t>(text_row_elements());   // proj
    bf16 += static_cast<size_t>(text_q_elements());     // q
    bf16 += static_cast<size_t>(text_kv_elements());    // k
    bf16 += static_cast<size_t>(text_kv_elements());    // v
    bf16 += static_cast<size_t>(text_ffn_elements());   // gate
    bf16 += static_cast<size_t>(text_ffn_elements());   // up
    bf16 += static_cast<size_t>(audio_io_elements());   // d_audio
    bf16 += static_cast<size_t>(au_norm_elements());    // d_au_norm
    bf16 += 3 * static_cast<size_t>(au_qkv_elements()); // d_au_q/k/v
    bf16 += static_cast<size_t>(au_ffn_elements());     // d_au_ffn
    bf16 += static_cast<size_t>(conv_in_elements());
    bf16 += static_cast<size_t>(conv1_elements());
    bf16 += static_cast<size_t>(conv2_elements());
    bf16 += static_cast<size_t>(conv3_elements());
    bf16 += kv_cache_elements();
    const size_t f32 = static_cast<size_t>(mel_elements()) +
                       static_cast<size_t>(inv_freq_elements()) +
                       static_cast<size_t>(logits_elements()) +
                       static_cast<size_t>(pe_elements());
    const size_t i32 = 2 * static_cast<size_t>(max_seq) + 1;  // ids, positions, argmax
    return bf16 * sizeof(uint16_t) + f32 * sizeof(float) + i32 * sizeof(int);
  }
};

}  // namespace umminfer::native
