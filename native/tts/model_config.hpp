// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace umminfer::native {

// One weight tensor the engine dereferences, and the shape config.json
// predicts for it. The engine compares these against the checkpoint header so a
// config and a weight file from different sizes fail at load, not by writing
// past a buffer.
struct TensorShape {
  std::string name;
  std::vector<size_t> dims;
};

// Dimensions of one Qwen3-TTS CustomVoice checkpoint, read from
// <model_dir>/config.json. Covers both wired sizes, 0.6B and 1.7B: their
// talker_config blocks differ only in hidden_size (1024/2048) and
// intermediate_size (3072/6144), and the code predictor is 1024-wide in both,
// so a 1.7B checkpoint carries a small_to_mtp_projection from the talker width
// into the predictor width. Header-only and free of CUDA headers so the host
// tests build without a GPU.
struct TtsModelConfig {
  // Top-level prompt token ids.
  int im_start_token_id = 0;
  int im_end_token_id = 0;
  int assistant_token_id = 0;
  int tts_pad_token_id = 0;
  int tts_bos_token_id = 0;
  int tts_eos_token_id = 0;

  std::string model_size;  // "0b6" or "1b7"
  std::string model_type;  // must be "custom_voice"

  // Talker (Qwen3 decoder with a codec head).
  int hidden = 0;
  int inter = 0;
  int layers = 0;
  int q_heads = 0;
  int kv_heads = 0;
  int head_dim = 0;
  int text_hidden = 0;
  int text_vocab = 0;
  int codec_vocab = 0;
  float rms_norm_eps = 0.0f;
  float rope_theta = 0.0f;
  bool rope_interleaved = false;

  // Code predictor: five 1024-wide decoder layers over the 15 residual
  // sub-codebooks of one frame.
  int cp_hidden = 0;
  int cp_inter = 0;
  int cp_layers = 0;
  int cp_q_heads = 0;
  int cp_kv_heads = 0;
  int cp_head_dim = 0;
  int cp_vocab = 0;

  // Codec control ids and the name -> id maps the prompt builder wires.
  int code_groups = 0;
  int codec_pad_id = 0;
  int codec_bos_id = 0;
  int codec_eos_id = 0;
  int codec_think_id = 0;
  int codec_think_bos_id = 0;
  int codec_think_eos_id = 0;
  int codec_nothink_id = 0;
  std::map<std::string, int> language_ids;
  std::map<std::string, int> speaker_ids;

  // Widths the GEMM kernels need, derived rather than stored.
  int q_width() const { return q_heads * head_dim; }
  int kv_width() const { return kv_heads * head_dim; }
  int cp_q_width() const { return cp_q_heads * cp_head_dim; }
  int cp_kv_width() const { return cp_kv_heads * cp_head_dim; }
  // True when the talker width differs from the predictor width, which is the
  // case for 1.7B only. The checkpoint then carries the projection weights.
  bool has_mtp_projection() const { return hidden != cp_hidden; }

  int language_id(const std::string& name) const {
    auto it = language_ids.find(name);
    if (it == language_ids.end())
      throw std::invalid_argument("config.json has no language id for: " + name);
    return it->second;
  }
  int speaker_id(const std::string& name) const {
    auto it = speaker_ids.find(name);
    if (it == speaker_ids.end())
      throw std::invalid_argument("config.json has no speaker id for: " + name);
    return it->second;
  }

  // 0.6B CustomVoice values, fixed for byte-identical behaviour and used by
  // the tests to cross-check a parsed config.json.
  static TtsModelConfig preset_0_6b() {
    TtsModelConfig c;
    c.im_start_token_id = 151644;
    c.im_end_token_id = 151645;
    c.assistant_token_id = 77091;
    c.tts_pad_token_id = 151671;
    c.tts_bos_token_id = 151672;
    c.tts_eos_token_id = 151673;
    c.model_size = "0b6";
    c.model_type = "custom_voice";
    c.hidden = 1024;
    c.inter = 3072;
    c.layers = 28;
    c.q_heads = 16;
    c.kv_heads = 8;
    c.head_dim = 128;
    c.text_hidden = 2048;
    c.text_vocab = 151936;
    c.codec_vocab = 3072;
    c.rms_norm_eps = 1e-6f;
    c.rope_theta = 1000000.0f;
    c.rope_interleaved = true;
    c.cp_hidden = 1024;
    c.cp_inter = 3072;
    c.cp_layers = 5;
    c.cp_q_heads = 16;
    c.cp_kv_heads = 8;
    c.cp_head_dim = 128;
    c.cp_vocab = 2048;
    c.code_groups = 16;
    c.codec_pad_id = 2148;
    c.codec_bos_id = 2149;
    c.codec_eos_id = 2150;
    c.codec_think_id = 2154;
    c.codec_think_bos_id = 2156;
    c.codec_think_eos_id = 2157;
    c.codec_nothink_id = 2155;
    c.language_ids = {{"beijing_dialect", 2074}, {"chinese", 2055},
                      {"english", 2050},         {"french", 2061},
                      {"german", 2053},          {"italian", 2070},
                      {"japanese", 2058},        {"korean", 2064},
                      {"portuguese", 2071},      {"russian", 2069},
                      {"sichuan_dialect", 2062}, {"spanish", 2054}};
    c.speaker_ids = {{"aiden", 2861},   {"dylan", 2878}, {"eric", 2875},
                     {"ono_anna", 2873}, {"ryan", 3061},  {"serena", 3066},
                     {"sohee", 2864},   {"uncle_fu", 3010}, {"vivian", 3065}};
    return c;
  }

  // 1.7B CustomVoice values. Same layout as 0.6B except the doubled talker
  // hidden/intermediate width, which forces the small_to_mtp_projection.
  static TtsModelConfig preset_1_7b() {
    TtsModelConfig c = preset_0_6b();
    c.model_size = "1b7";
    c.hidden = 2048;
    c.inter = 6144;
    return c;
  }

  static TtsModelConfig from_json_text(const std::string& json_text) {
    nlohmann::json root;
    try {
      root = nlohmann::json::parse(json_text);
    } catch (const nlohmann::json::exception& e) {
      throw std::invalid_argument(std::string("config.json is not valid JSON: ") +
                                  e.what());
    }
    auto obj = [](const nlohmann::json& parent, const char* key) -> const nlohmann::json& {
      if (!parent.is_object() || !parent.contains(key))
        throw std::invalid_argument(std::string("config.json missing section: ") + key);
      const nlohmann::json& v = parent.at(key);
      if (!v.is_object())
        throw std::invalid_argument(std::string("config.json section is not an object: ") +
                                    key);
      return v;
    };
    auto num = [](const nlohmann::json& j, const char* key) {
      if (!j.is_object() || !j.contains(key))
        throw std::invalid_argument(std::string("config.json missing field: ") + key);
      const nlohmann::json& v = j.at(key);
      if (!v.is_number())
        throw std::invalid_argument(std::string("config.json field is not a number: ") + key);
      if (v.is_number_integer()) return static_cast<long long>(v.get<long long>());
      const double d = v.get<double>();
      if (d != static_cast<double>(static_cast<long long>(d)))
        throw std::invalid_argument(std::string("config.json field is not integral: ") + key);
      return static_cast<long long>(d);
    };
    auto as_int = [&num](const nlohmann::json& j, const char* key) {
      const long long v = num(j, key);
      if (v < -2147483648LL || v > 2147483647LL)
        throw std::invalid_argument(std::string("config.json field out of int range: ") +
                                    key);
      return static_cast<int>(v);
    };
    auto as_float = [](const nlohmann::json& j, const char* key) {
      if (!j.is_object() || !j.contains(key))
        throw std::invalid_argument(std::string("config.json missing field: ") + key);
      const nlohmann::json& v = j.at(key);
      if (!v.is_number())
        throw std::invalid_argument(std::string("config.json field is not a number: ") + key);
      return v.get<float>();
    };
    auto as_str = [](const nlohmann::json& j, const char* key) {
      if (!j.is_object() || !j.contains(key))
        throw std::invalid_argument(std::string("config.json missing field: ") + key);
      const nlohmann::json& v = j.at(key);
      if (!v.is_string())
        throw std::invalid_argument(std::string("config.json field is not a string: ") + key);
      return v.get<std::string>();
    };
    auto id_map = [&num](const nlohmann::json& parent, const char* key) {
      std::map<std::string, int> out;
      if (!parent.contains(key) || !parent.at(key).is_object()) return out;
      for (auto it = parent.at(key).begin(); it != parent.at(key).end(); ++it) {
        const long long v = num(parent.at(key), it.key().c_str());
        out.emplace(it.key(), static_cast<int>(v));
      }
      return out;
    };

    const nlohmann::json& talker = obj(root, "talker_config");
    const nlohmann::json& cp = obj(talker, "code_predictor_config");

    TtsModelConfig c;
    c.im_start_token_id = as_int(root, "im_start_token_id");
    c.im_end_token_id = as_int(root, "im_end_token_id");
    c.assistant_token_id = as_int(root, "assistant_token_id");
    c.tts_pad_token_id = as_int(root, "tts_pad_token_id");
    c.tts_bos_token_id = as_int(root, "tts_bos_token_id");
    c.tts_eos_token_id = as_int(root, "tts_eos_token_id");
    c.model_size = as_str(root, "tts_model_size");
    c.model_type = as_str(root, "tts_model_type");

    c.hidden = as_int(talker, "hidden_size");
    c.inter = as_int(talker, "intermediate_size");
    c.layers = as_int(talker, "num_hidden_layers");
    c.q_heads = as_int(talker, "num_attention_heads");
    c.kv_heads = as_int(talker, "num_key_value_heads");
    c.head_dim = as_int(talker, "head_dim");
    c.text_hidden = as_int(talker, "text_hidden_size");
    c.text_vocab = as_int(talker, "text_vocab_size");
    c.codec_vocab = as_int(talker, "vocab_size");
    c.rms_norm_eps = as_float(talker, "rms_norm_eps");
    c.rope_theta = as_float(talker, "rope_theta");
    c.code_groups = as_int(talker, "num_code_groups");
    c.rope_interleaved = false;
    if (talker.contains("rope_scaling") && talker.at("rope_scaling").is_object())
      c.rope_interleaved = talker.at("rope_scaling").value("interleaved", false);

    c.cp_hidden = as_int(cp, "hidden_size");
    c.cp_inter = as_int(cp, "intermediate_size");
    c.cp_layers = as_int(cp, "num_hidden_layers");
    c.cp_q_heads = as_int(cp, "num_attention_heads");
    c.cp_kv_heads = as_int(cp, "num_key_value_heads");
    c.cp_head_dim = as_int(cp, "head_dim");
    c.cp_vocab = as_int(cp, "vocab_size");

    c.codec_pad_id = as_int(talker, "codec_pad_id");
    c.codec_bos_id = as_int(talker, "codec_bos_id");
    c.codec_eos_id = as_int(talker, "codec_eos_token_id");
    c.codec_think_id = as_int(talker, "codec_think_id");
    c.codec_think_bos_id = as_int(talker, "codec_think_bos_id");
    c.codec_think_eos_id = as_int(talker, "codec_think_eos_id");
    c.codec_nothink_id = as_int(talker, "codec_nothink_id");
    c.language_ids = id_map(talker, "codec_language_id");
    c.speaker_ids = id_map(talker, "spk_id");

    c.validate();
    return c;
  }

  static TtsModelConfig from_model_dir(const std::string& model_dir) {
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
    positive(hidden, "talker hidden_size");
    positive(inter, "talker intermediate_size");
    positive(layers, "talker num_hidden_layers");
    positive(q_heads, "talker num_attention_heads");
    positive(kv_heads, "talker num_key_value_heads");
    positive(head_dim, "talker head_dim");
    positive(text_hidden, "talker text_hidden_size");
    positive(text_vocab, "talker text_vocab_size");
    positive(codec_vocab, "talker vocab_size");
    positive(cp_hidden, "code_predictor hidden_size");
    positive(cp_inter, "code_predictor intermediate_size");
    positive(cp_layers, "code_predictor num_hidden_layers");
    positive(cp_q_heads, "code_predictor num_attention_heads");
    positive(cp_kv_heads, "code_predictor num_key_value_heads");
    positive(cp_head_dim, "code_predictor head_dim");
    positive(cp_vocab, "code_predictor vocab_size");

    if (model_type != "custom_voice")
      throw std::invalid_argument("config.json tts_model_type must be custom_voice, got " +
                                  model_type);
    if (code_groups != 16)
      throw std::invalid_argument("config.json num_code_groups must be 16, got " +
                                  std::to_string(code_groups));
    // The talker and predictor projections are wider than their hidden size
    // (16 * 128 = 2048 against a 1024-wide 0.6B residual stream), so only the
    // head split is constrained, not q_heads * head_dim == hidden.
    if (q_heads % kv_heads != 0)
      throw std::invalid_argument("config.json num_attention_heads is not divisible by "
                                  "num_key_value_heads");
    if (cp_q_heads % cp_kv_heads != 0)
      throw std::invalid_argument("config.json code_predictor num_attention_heads is not "
                                  "divisible by num_key_value_heads");
    // The attention kernels take one head width for the talker and predictor.
    if (cp_head_dim != head_dim)
      throw std::invalid_argument("config.json code_predictor head_dim differs from the talker "
                                  "head_dim");
    if (head_dim % 2 != 0)
      throw std::invalid_argument("config.json head_dim must be even for half-rotation RoPE");
    if (!(rms_norm_eps > 0.0f))
      throw std::invalid_argument("config.json rms_norm_eps must be positive");
    if (!(rope_theta > 0.0f))
      throw std::invalid_argument("config.json rope_theta must be positive");
    // The native kernels implement plain half-rotation RoPE on equal mRoPE
    // axes. A checkpoint that rotates differently would silently mis-position.
    if (!rope_interleaved)
      throw std::invalid_argument("config.json rope_scaling must set interleaved=true");

    // The prompt builder has one wired prefill plan; these ids must match it.
    if (tts_pad_token_id <= 0 || tts_bos_token_id <= 0 || tts_eos_token_id <= 0)
      throw std::invalid_argument("config.json tts special token ids must be positive");
    if (codec_pad_id <= 0 || codec_bos_id <= 0 || codec_eos_id <= 0)
      throw std::invalid_argument("config.json codec control ids must be positive");
    if (codec_eos_id >= codec_vocab)
      throw std::invalid_argument("config.json codec_eos_token_id " +
                                  std::to_string(codec_eos_id) +
                                  " is outside the codec vocabulary");
    for (const char* name : {"chinese", "english"})
      if (language_ids.find(name) == language_ids.end())
        throw std::invalid_argument(std::string("config.json codec_language_id has no ") +
                                    name);
    for (const char* name : {"vivian", "ryan"})
      if (speaker_ids.find(name) == speaker_ids.end())
        throw std::invalid_argument(std::string("config.json spk_id has no ") + name);
    for (const auto& [name, id] : language_ids)
      if (id < 0 || id >= codec_vocab)
        throw std::invalid_argument("config.json language id for " + name +
                                    " is outside the codec vocabulary");
    for (const auto& [name, id] : speaker_ids)
      if (id < 0 || id >= codec_vocab)
        throw std::invalid_argument("config.json speaker id for " + name +
                                    " is outside the codec vocabulary");
  }

  // Every tensor the engine dereferences, with the shape config.json implies.
  // The talker and predictor weights are all BF16 in both wired checkpoints.
  std::vector<TensorShape> tensor_shape_plan() const {
    std::vector<TensorShape> out;
    auto add = [&out](std::string n, std::initializer_list<size_t> d) {
      out.push_back({std::move(n), std::vector<size_t>(d)});
    };
    const size_t H = static_cast<size_t>(hidden), I = static_cast<size_t>(inter);
    const size_t CPH = static_cast<size_t>(cp_hidden),
                 CPI = static_cast<size_t>(cp_inter);
    const size_t QW = static_cast<size_t>(q_width()),
                 KVW = static_cast<size_t>(kv_width());
    const size_t CQW = static_cast<size_t>(cp_q_width()),
                 CKVW = static_cast<size_t>(cp_kv_width());
    const size_t TH = static_cast<size_t>(text_hidden),
                 HD = static_cast<size_t>(head_dim);

    add("talker.model.text_embedding.weight", {static_cast<size_t>(text_vocab), TH});
    add("talker.text_projection.linear_fc1.weight", {TH, TH});
    add("talker.text_projection.linear_fc1.bias", {TH});
    add("talker.text_projection.linear_fc2.weight", {H, TH});
    add("talker.text_projection.linear_fc2.bias", {H});
    add("talker.model.codec_embedding.weight", {static_cast<size_t>(codec_vocab), H});
    add("talker.codec_head.weight", {static_cast<size_t>(codec_vocab), H});
    add("talker.model.norm.weight", {H});

    auto add_layers = [&add](const std::string& base, int count, size_t h,
                             size_t in, size_t qw, size_t kvw, size_t hd) {
      for (int l = 0; l < count; ++l) {
        const std::string p = base + ".layers." + std::to_string(l);
        add(p + ".input_layernorm.weight", {h});
        add(p + ".post_attention_layernorm.weight", {h});
        add(p + ".self_attn.q_proj.weight", {qw, h});
        add(p + ".self_attn.k_proj.weight", {kvw, h});
        add(p + ".self_attn.v_proj.weight", {kvw, h});
        add(p + ".self_attn.o_proj.weight", {h, qw});
        add(p + ".self_attn.q_norm.weight", {hd});
        add(p + ".self_attn.k_norm.weight", {hd});
        add(p + ".mlp.gate_proj.weight", {in, h});
        add(p + ".mlp.up_proj.weight", {in, h});
        add(p + ".mlp.down_proj.weight", {h, in});
      }
    };
    add_layers("talker.model", layers, H, I, QW, KVW, HD);
    add_layers("talker.code_predictor.model", cp_layers, CPH, CPI, CQW, CKVW,
               static_cast<size_t>(cp_head_dim));
    add("talker.code_predictor.model.norm.weight", {CPH});
    if (has_mtp_projection()) {
      add("talker.code_predictor.small_to_mtp_projection.weight", {CPH, H});
      add("talker.code_predictor.small_to_mtp_projection.bias", {CPH});
    }
    for (int j = 0; j < code_groups - 1; ++j) {
      const std::string s = std::to_string(j);
      add("talker.code_predictor.lm_head." + s + ".weight",
          {static_cast<size_t>(cp_vocab), CPH});
      // The predictor embeddings are talker-width in both wired checkpoints;
      // the projection above narrows them to the predictor width.
      add("talker.code_predictor.model.codec_embedding." + s + ".weight",
          {static_cast<size_t>(cp_vocab), H});
    }
    return out;
  }
};

}  // namespace umminfer::native
