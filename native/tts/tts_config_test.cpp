// SPDX-License-Identifier: Apache-2.0
// Host-only checks for model_config.hpp: both wired CustomVoice configs parse
// to the expected dimensions, the 1.7B checkpoint carries the talker-width
// predictor embeddings and the small_to_mtp_projection, and malformed or
// mismatched configs are rejected. No CUDA and no weights, so this runs on the
// development host. With a model directory as argv[1] (or UMM_TTS_TEST_DATA)
// it also parses that real config.json.
#include "model_config.hpp"
#include "speaker_encoder.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace umminfer::native;

#define CHECK(cond)                                                     \
  do {                                                                  \
    if (!(cond)) {                                                      \
      std::cerr << "CHECK failed: " #cond " at " << __FILE__ << ":"     \
                << __LINE__ << "\n";                                    \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

// Verbatim copies of the two published config.json files, so the fixtures stay
// exact without needing the weights on disk.
static const char* kConfig06b = R"JSON({
  "architectures": [
    "Qwen3TTSForConditionalGeneration"
  ],
  "assistant_token_id": 77091,
  "im_end_token_id": 151645,
  "im_start_token_id": 151644,
  "tts_bos_token_id": 151672,
  "tts_eos_token_id": 151673,
  "tts_pad_token_id": 151671,
  "model_type": "qwen3_tts",
  "tokenizer_type": "qwen3_tts_tokenizer_12hz",
  "tts_model_size": "0b6",
  "tts_model_type": "custom_voice",
  "talker_config": {
    "attention_bias": false,
    "attention_dropout": 0,
    "code_predictor_config": {
      "_name_or_path": "",
      "add_cross_attention": false,
      "architectures": null,
      "attention_bias": false,
      "attention_dropout": 0,
      "bad_words_ids": null,
      "begin_suppress_tokens": null,
      "bos_token_id": null,
      "chunk_size_feed_forward": 0,
      "cross_attention_hidden_size": null,
      "decoder_start_token_id": null,
      "diversity_penalty": 0.0,
      "do_sample": false,
      "early_stopping": false,
      "encoder_no_repeat_ngram_size": 0,
      "eos_token_id": null,
      "exponential_decay_length_penalty": null,
      "finetuning_task": null,
      "forced_bos_token_id": null,
      "forced_eos_token_id": null,
      "head_dim": 128,
      "hidden_act": "silu",
      "hidden_size": 1024,
      "id2label": {
        "0": "LABEL_0",
        "1": "LABEL_1"
      },
      "initializer_range": 0.02,
      "intermediate_size": 3072,
      "is_decoder": false,
      "is_encoder_decoder": false,
      "label2id": {
        "LABEL_0": 0,
        "LABEL_1": 1
      },
      "layer_types": [
        "full_attention",
        "full_attention",
        "full_attention",
        "full_attention",
        "full_attention"
      ],
      "length_penalty": 1.0,
      "max_length": 20,
      "max_position_embeddings": 65536,
      "max_window_layers": 28,
      "min_length": 0,
      "model_type": "qwen3_tts_talker_code_predictor",
      "no_repeat_ngram_size": 0,
      "num_attention_heads": 16,
      "num_beam_groups": 1,
      "num_beams": 1,
      "num_code_groups": 16,
      "num_hidden_layers": 5,
      "num_key_value_heads": 8,
      "num_return_sequences": 1,
      "output_attentions": false,
      "output_hidden_states": false,
      "output_scores": false,
      "pad_token_id": null,
      "prefix": null,
      "problem_type": null,
      "pruned_heads": {},
      "remove_invalid_values": false,
      "repetition_penalty": 1.0,
      "return_dict": true,
      "return_dict_in_generate": false,
      "rms_norm_eps": 1e-06,
      "rope_scaling": null,
      "rope_theta": 1000000,
      "sep_token_id": null,
      "sliding_window": null,
      "suppress_tokens": null,
      "task_specific_params": null,
      "temperature": 1.0,
      "tf_legacy_loss": false,
      "tie_encoder_decoder": false,
      "tie_word_embeddings": false,
      "tokenizer_class": null,
      "top_k": 50,
      "top_p": 1.0,
      "dtype": null,
      "torchscript": false,
      "typical_p": 1.0,
      "use_bfloat16": false,
      "use_cache": true,
      "use_sliding_window": false,
      "vocab_size": 2048
    },
    "codec_bos_id": 2149,
    "codec_eos_token_id": 2150,
    "codec_think_id": 2154,
    "codec_language_id": {
        "chinese": 2055,
        "english": 2050,
        "german": 2053,
        "italian": 2070,
        "portuguese": 2071,
        "spanish": 2054,
        "japanese": 2058,
        "korean": 2064,
        "french": 2061,
        "russian": 2069,
        "beijing_dialect": 2074,
        "sichuan_dialect": 2062
    },
    "codec_nothink_id": 2155,
    "codec_pad_id": 2148,
    "codec_think_bos_id": 2156,
    "codec_think_eos_id": 2157,
    "spk_id": {
      "serena": 3066,
      "vivian": 3065,
      "uncle_fu": 3010,
      "ryan": 3061,
      "aiden": 2861,
      "ono_anna": 2873,
      "sohee": 2864,
      "eric": 2875,
      "dylan": 2878
    },
    "spk_is_dialect": {
      "serena": false,
      "vivian": false,
      "uncle_fu": false,
      "ryan": false,
      "aiden": false,
      "ono_anna": false,
      "sohee": false,
      "eric": "sichuan_dialect",
      "dylan": "beijing_dialect"
    },
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 32768,
    "model_type": "qwen3_tts_talker",
    "num_attention_heads": 16,
    "num_code_groups": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "position_id_per_seconds": 13,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
      "interleaved": true,
      "mrope_section": [
        24,
        20,
        20
      ],
      "rope_type": "default",
      "type": "default"
    },
    "rope_theta": 1000000,
    "sliding_window": null,
    "text_hidden_size": 2048,
    "text_vocab_size": 151936,
    "use_cache": true,
    "use_sliding_window": false,
    "vocab_size": 3072
  },
  "transformers_version": "4.57.3"
}
)JSON";
static const char* kConfig17b = R"JSON({
  "architectures": [
    "Qwen3TTSForConditionalGeneration"
  ],
  "assistant_token_id": 77091,
  "im_end_token_id": 151645,
  "im_start_token_id": 151644,
  "tts_bos_token_id": 151672,
  "tts_eos_token_id": 151673,
  "tts_pad_token_id": 151671,
  "model_type": "qwen3_tts",
  "tokenizer_type": "qwen3_tts_tokenizer_12hz",
  "tts_model_size": "1b7",
  "tts_model_type": "custom_voice",
  "talker_config": {
    "attention_bias": false,
    "attention_dropout": 0,
    "code_predictor_config": {
      "_name_or_path": "",
      "add_cross_attention": false,
      "architectures": null,
      "attention_bias": false,
      "attention_dropout": 0,
      "bad_words_ids": null,
      "begin_suppress_tokens": null,
      "bos_token_id": null,
      "chunk_size_feed_forward": 0,
      "cross_attention_hidden_size": null,
      "decoder_start_token_id": null,
      "diversity_penalty": 0.0,
      "do_sample": false,
      "early_stopping": false,
      "encoder_no_repeat_ngram_size": 0,
      "eos_token_id": null,
      "exponential_decay_length_penalty": null,
      "finetuning_task": null,
      "forced_bos_token_id": null,
      "forced_eos_token_id": null,
      "head_dim": 128,
      "hidden_act": "silu",
      "hidden_size": 1024,
      "id2label": {
        "0": "LABEL_0",
        "1": "LABEL_1"
      },
      "initializer_range": 0.02,
      "intermediate_size": 3072,
      "is_decoder": false,
      "is_encoder_decoder": false,
      "label2id": {
        "LABEL_0": 0,
        "LABEL_1": 1
      },
      "layer_types": [
        "full_attention",
        "full_attention",
        "full_attention",
        "full_attention",
        "full_attention"
      ],
      "length_penalty": 1.0,
      "max_length": 20,
      "max_position_embeddings": 65536,
      "max_window_layers": 28,
      "min_length": 0,
      "model_type": "qwen3_tts_talker_code_predictor",
      "no_repeat_ngram_size": 0,
      "num_attention_heads": 16,
      "num_beam_groups": 1,
      "num_beams": 1,
      "num_code_groups": 16,
      "num_hidden_layers": 5,
      "num_key_value_heads": 8,
      "num_return_sequences": 1,
      "output_attentions": false,
      "output_hidden_states": false,
      "output_scores": false,
      "pad_token_id": null,
      "prefix": null,
      "problem_type": null,
      "pruned_heads": {},
      "remove_invalid_values": false,
      "repetition_penalty": 1.0,
      "return_dict": true,
      "return_dict_in_generate": false,
      "rms_norm_eps": 1e-06,
      "rope_scaling": null,
      "rope_theta": 1000000,
      "sep_token_id": null,
      "sliding_window": null,
      "suppress_tokens": null,
      "task_specific_params": null,
      "temperature": 1.0,
      "tf_legacy_loss": false,
      "tie_encoder_decoder": false,
      "tie_word_embeddings": false,
      "tokenizer_class": null,
      "top_k": 50,
      "top_p": 1.0,
      "dtype": null,
      "torchscript": false,
      "typical_p": 1.0,
      "use_bfloat16": false,
      "use_cache": true,
      "use_sliding_window": false,
      "vocab_size": 2048
    },
    "codec_bos_id": 2149,
    "codec_eos_token_id": 2150,
    "codec_think_id": 2154,
    "codec_language_id": {
        "chinese": 2055,
        "english": 2050,
        "german": 2053,
        "italian": 2070,
        "portuguese": 2071,
        "spanish": 2054,
        "japanese": 2058,
        "korean": 2064,
        "french": 2061,
        "russian": 2069,
        "beijing_dialect": 2074,
        "sichuan_dialect": 2062
    },
    "codec_nothink_id": 2155,
    "codec_pad_id": 2148,
    "codec_think_bos_id": 2156,
    "codec_think_eos_id": 2157,
    "spk_id": {
      "serena": 3066,
      "vivian": 3065,
      "uncle_fu": 3010,
      "ryan": 3061,
      "aiden": 2861,
      "ono_anna": 2873,
      "sohee": 2864,
      "eric": 2875,
      "dylan": 2878
    },
    "spk_is_dialect": {
      "serena": false,
      "vivian": false,
      "uncle_fu": false,
      "ryan": false,
      "aiden": false,
      "ono_anna": false,
      "sohee": false,
      "eric": "sichuan_dialect",
      "dylan": "beijing_dialect"
    },
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 6144,
    "max_position_embeddings": 32768,
    "model_type": "qwen3_tts_talker",
    "num_attention_heads": 16,
    "num_code_groups": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "position_id_per_seconds": 13,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
      "interleaved": true,
      "mrope_section": [
        24,
        20,
        20
      ],
      "rope_type": "default",
      "type": "default"
    },
    "rope_theta": 1000000,
    "sliding_window": null,
    "text_hidden_size": 2048,
    "text_vocab_size": 151936,
    "use_cache": true,
    "use_sliding_window": false,
    "vocab_size": 3072
  },
  "transformers_version": "4.57.3"
}
)JSON";

static std::string replaced(std::string text, const std::string& from,
                            const std::string& to) {
  const size_t at = text.find(from);
  if (at == std::string::npos) {
    std::cerr << "fixture does not contain: " << from << "\n";
    std::exit(1);
  }
  return text.replace(at, from.size(), to);
}

static std::string base_config() {
  nlohmann::json j = nlohmann::json::parse(kConfig17b);
  j["tts_model_type"] = "base";
  j["talker_config"]["spk_id"] = nlohmann::json::object();
  j["speaker_encoder_config"] = {{"enc_dim", 2048}, {"sample_rate", 24000}};
  return j.dump();
}

static void expect_reject(const std::string& json, const char* label) {
  try {
    (void)TtsModelConfig::from_json_text(json);
  } catch (const std::invalid_argument&) {
    return;
  }
  std::cerr << "expected rejection: " << label << "\n";
  std::exit(1);
}

static void check_same(const TtsModelConfig& got, const TtsModelConfig& want,
                       const std::string& label) {
  const auto fail = [&](const char* field) {
    std::cerr << "mismatch: " << label << " field " << field << "\n";
    std::exit(1);
  };
  if (got.im_start_token_id != want.im_start_token_id) fail("im_start_token_id");
  if (got.im_end_token_id != want.im_end_token_id) fail("im_end_token_id");
  if (got.assistant_token_id != want.assistant_token_id) fail("assistant_token_id");
  if (got.tts_pad_token_id != want.tts_pad_token_id) fail("tts_pad_token_id");
  if (got.tts_bos_token_id != want.tts_bos_token_id) fail("tts_bos_token_id");
  if (got.tts_eos_token_id != want.tts_eos_token_id) fail("tts_eos_token_id");
  if (got.model_size != want.model_size) fail("tts_model_size");
  if (got.model_type != want.model_type) fail("tts_model_type");
  if (got.speaker_encoder.enc_dim != want.speaker_encoder.enc_dim ||
      got.speaker_encoder.sample_rate != want.speaker_encoder.sample_rate)
    fail("speaker_encoder_config");
  if (got.hidden != want.hidden) fail("hidden");
  if (got.inter != want.inter) fail("inter");
  if (got.layers != want.layers) fail("layers");
  if (got.q_heads != want.q_heads) fail("q_heads");
  if (got.kv_heads != want.kv_heads) fail("kv_heads");
  if (got.head_dim != want.head_dim) fail("head_dim");
  if (got.text_hidden != want.text_hidden) fail("text_hidden");
  if (got.text_vocab != want.text_vocab) fail("text_vocab");
  if (got.codec_vocab != want.codec_vocab) fail("codec_vocab");
  if (got.rms_norm_eps != want.rms_norm_eps) fail("rms_norm_eps");
  if (got.rope_theta != want.rope_theta) fail("rope_theta");
  if (got.rope_interleaved != want.rope_interleaved) fail("rope_interleaved");
  if (got.cp_hidden != want.cp_hidden) fail("cp_hidden");
  if (got.cp_inter != want.cp_inter) fail("cp_inter");
  if (got.cp_layers != want.cp_layers) fail("cp_layers");
  if (got.cp_q_heads != want.cp_q_heads) fail("cp_q_heads");
  if (got.cp_kv_heads != want.cp_kv_heads) fail("cp_kv_heads");
  if (got.cp_head_dim != want.cp_head_dim) fail("cp_head_dim");
  if (got.cp_vocab != want.cp_vocab) fail("cp_vocab");
  if (got.code_groups != want.code_groups) fail("code_groups");
  if (got.codec_pad_id != want.codec_pad_id) fail("codec_pad_id");
  if (got.codec_bos_id != want.codec_bos_id) fail("codec_bos_id");
  if (got.codec_eos_id != want.codec_eos_id) fail("codec_eos_id");
  if (got.codec_think_id != want.codec_think_id) fail("codec_think_id");
  if (got.codec_think_bos_id != want.codec_think_bos_id) fail("codec_think_bos_id");
  if (got.codec_think_eos_id != want.codec_think_eos_id) fail("codec_think_eos_id");
  if (got.codec_nothink_id != want.codec_nothink_id) fail("codec_nothink_id");
  // The presets record only the ids the prompt builder wires; a checkpoint may
  // publish more languages and speakers.
  for (const auto& [name, id] : want.language_ids)
    if (got.language_ids.count(name) == 0 || got.language_ids.at(name) != id)
      fail("language_ids");
  for (const auto& [name, id] : want.speaker_ids)
    if (got.speaker_ids.count(name) == 0 || got.speaker_ids.at(name) != id)
      fail("speaker_ids");
}

static const TensorShape* find_shape(const std::vector<TensorShape>& plan,
                                     const std::string& name) {
  for (const TensorShape& s : plan)
    if (s.name == name) return &s;
  return nullptr;
}

static void test_parsing() {
  const TtsModelConfig c06 = TtsModelConfig::from_json_text(kConfig06b);
  check_same(c06, TtsModelConfig::preset_0_6b(), "0.6B");
  CHECK(!c06.has_mtp_projection());
  CHECK(c06.language_id("chinese") == 2055 && c06.language_id("english") == 2050);
  CHECK(c06.speaker_id("vivian") == 3065 && c06.speaker_id("ryan") == 3061);
  CHECK(c06.codec_think_id == 2154 && c06.codec_think_eos_id == 2157);
  CHECK(c06.q_width() == 2048 && c06.kv_width() == 1024);
  CHECK(c06.cp_q_width() == 2048 && c06.cp_kv_width() == 1024);

  const TtsModelConfig c17 = TtsModelConfig::from_json_text(kConfig17b);
  check_same(c17, TtsModelConfig::preset_1_7b(), "1.7B");
  CHECK(c17.model_size == "1b7");
  CHECK(c17.hidden == 2048 && c17.inter == 6144);
  CHECK(c17.cp_hidden == 1024 && c17.cp_inter == 3072);
  CHECK(c17.has_mtp_projection());
  CHECK(c17.speaker_id("vivian") == 3065 && c17.speaker_id("ryan") == 3061);

  nlohmann::json no_speakers = nlohmann::json::parse(base_config());
  no_speakers["talker_config"].erase("spk_id");
  const TtsModelConfig base = TtsModelConfig::from_json_text(no_speakers.dump());
  CHECK(base.is_base() && base.speaker_ids.empty());
  CHECK(base.speaker_encoder.enc_dim == 2048 &&
        base.speaker_encoder.sample_rate == 24000);
}

static void test_shape_plan() {
  const TtsModelConfig c06 = TtsModelConfig::preset_0_6b();
  const auto p06 = c06.tensor_shape_plan();
  // 8 fixed tensors + 28 talker layers * 11 + 5 predictor layers * 11 + the
  // predictor norm + 15 * (lm_head + embedding); no projection tensor.
  CHECK(p06.size() == 8 + 28 * 11 + 5 * 11 + 1 + 15 * 2);
  CHECK(find_shape(p06, "talker.code_predictor.small_to_mtp_projection.weight") ==
        nullptr);
  const TensorShape* emb06 =
      find_shape(p06, "talker.code_predictor.model.codec_embedding.14.weight");
  CHECK(emb06 != nullptr && emb06->dims == std::vector<size_t>({2048, 1024}));
  const TensorShape* down06 =
      find_shape(p06, "talker.model.layers.27.mlp.down_proj.weight");
  CHECK(down06 != nullptr && down06->dims == std::vector<size_t>({1024, 3072}));

  const TtsModelConfig c17 = TtsModelConfig::preset_1_7b();
  const auto p17 = c17.tensor_shape_plan();
  CHECK(p17.size() == p06.size() + 2);
  const TensorShape* mtp =
      find_shape(p17, "talker.code_predictor.small_to_mtp_projection.weight");
  CHECK(mtp != nullptr && mtp->dims == std::vector<size_t>({1024, 2048}));
  const TensorShape* mtpb =
      find_shape(p17, "talker.code_predictor.small_to_mtp_projection.bias");
  CHECK(mtpb != nullptr && mtpb->dims == std::vector<size_t>({1024}));
  const TensorShape* down17 =
      find_shape(p17, "talker.model.layers.27.mlp.down_proj.weight");
  CHECK(down17 != nullptr && down17->dims == std::vector<size_t>({2048, 6144}));
  const TensorShape* o17 =
      find_shape(p17, "talker.model.layers.0.self_attn.o_proj.weight");
  CHECK(o17 != nullptr && o17->dims == std::vector<size_t>({2048, 2048}));
  const TensorShape* kv17 =
      find_shape(p17, "talker.model.layers.0.self_attn.k_proj.weight");
  CHECK(kv17 != nullptr && kv17->dims == std::vector<size_t>({1024, 2048}));
  // The predictor stays 1024-wide even in the 1.7B checkpoint.
  const TensorShape* cpq17 =
      find_shape(p17, "talker.code_predictor.model.layers.4.self_attn.q_proj.weight");
  CHECK(cpq17 != nullptr && cpq17->dims == std::vector<size_t>({2048, 1024}));
  const TensorShape* cpe17 =
      find_shape(p17, "talker.code_predictor.model.codec_embedding.0.weight");
  CHECK(cpe17 != nullptr && cpe17->dims == std::vector<size_t>({2048, 2048}));
  const TensorShape* lh17 =
      find_shape(p17, "talker.code_predictor.lm_head.14.weight");
  CHECK(lh17 != nullptr && lh17->dims == std::vector<size_t>({2048, 1024}));

  const TtsModelConfig base = TtsModelConfig::from_json_text(base_config());
  const auto pb = base.tensor_shape_plan();
  CHECK(pb.size() == p17.size() + 76);
  const TensorShape* first =
      find_shape(pb, "speaker_encoder.blocks.0.conv.weight");
  CHECK(first != nullptr && first->dims == std::vector<size_t>({512, 128, 5}));
  const TensorShape* res2 =
      find_shape(pb, "speaker_encoder.blocks.3.res2net_block.blocks.6.conv.weight");
  CHECK(res2 != nullptr && res2->dims == std::vector<size_t>({64, 64, 3}));
  const TensorShape* asp = find_shape(pb, "speaker_encoder.asp.tdnn.conv.weight");
  CHECK(asp != nullptr && asp->dims == std::vector<size_t>({128, 4608, 1}));
  const TensorShape* fc = find_shape(pb, "speaker_encoder.fc.weight");
  CHECK(fc != nullptr && fc->dims == std::vector<size_t>({2048, 3072, 1}));
}

static void test_rejects() {
  expect_reject("{not json", "malformed JSON");
  expect_reject(replaced(kConfig06b, "\"talker_config\"", "\"not_talker_config\""),
                "missing talker_config");
  // The talker-level field, at four-space indentation; the predictor block has
  // its own copies at six spaces.
  expect_reject(replaced(kConfig06b, "\n    \"num_code_groups\": 16,",
                         "\n    \"num_code_groups\": 8,"),
                "code group count");
  expect_reject(replaced(kConfig06b, "\n    \"num_key_value_heads\": 8,",
                         "\n    \"num_key_value_heads\": 7,"),
                "kv head split");
  expect_reject(replaced(kConfig06b, "\"tts_model_type\": \"custom_voice\"",
                         "\"tts_model_type\": \"unknown\""),
                "unknown model type");
  nlohmann::json bad_base = nlohmann::json::parse(base_config());
  bad_base.erase("speaker_encoder_config");
  expect_reject(bad_base.dump(), "missing speaker_encoder_config");
  bad_base = nlohmann::json::parse(base_config());
  bad_base["speaker_encoder_config"].erase("enc_dim");
  expect_reject(bad_base.dump(), "missing speaker enc_dim");
  bad_base = nlohmann::json::parse(base_config());
  bad_base["speaker_encoder_config"]["sample_rate"] = 16000;
  expect_reject(bad_base.dump(), "speaker sample rate");
  bad_base = nlohmann::json::parse(base_config());
  bad_base["speaker_encoder_config"]["enc_dim"] = 1024;
  expect_reject(bad_base.dump(), "speaker output width");
  expect_reject(replaced(kConfig06b, "\"head_dim\": 128,", "\"head_dim\": 32,"),
                "predictor head width");
  expect_reject(replaced(kConfig06b, "\"ryan\": 3061,", ""), "missing ryan");
  expect_reject(replaced(kConfig06b, "\"english\": 2050,", ""), "missing english");
  // The kernels only implement interleaved RoPE.
  expect_reject(
      replaced(kConfig06b, "\"interleaved\": true", "\"interleaved\": false"),
      "non-interleaved RoPE");
  expect_reject(replaced(kConfig06b, "\n    \"rms_norm_eps\": 1e-06,",
                         "\n    \"rms_norm_eps\": 0,"),
                "zero rms eps");
}

// A config whose predictor width equals the talker width is legal on its own;
// the loader then expects no projection tensor and talker-width predictor
// weights. This is the shape rule the 0.6B checkpoint follows.
static void test_projection_rule() {
  const std::string same =
      replaced(kConfig17b,
               "\"hidden_size\": 1024,\n      \"id2label\"",
               "\"hidden_size\": 2048,\n      \"id2label\"");
  const TtsModelConfig cfg = TtsModelConfig::from_json_text(same);
  CHECK(cfg.hidden == 2048 && cfg.cp_hidden == 2048);
  CHECK(!cfg.has_mtp_projection());
  const auto plan = cfg.tensor_shape_plan();
  CHECK(find_shape(plan, "talker.code_predictor.small_to_mtp_projection.weight") ==
        nullptr);
  const TensorShape* lh =
      find_shape(plan, "talker.code_predictor.lm_head.0.weight");
  CHECK(lh != nullptr && lh->dims == std::vector<size_t>({2048, 2048}));
}

static void test_speaker_mel() {
  constexpr double pi = 3.14159265358979323846;
  std::vector<float> pcm(24000);
  for (int i = 0; i < 24000; ++i)
    pcm[i] = static_cast<float>(0.5 * std::sin(2.0 * pi * 440.0 * i / 24000.0));
  const SpeakerMel mel = compute_speaker_log_mel(pcm);
  CHECK(mel.frames == 93 && mel.log_mel.size() == 128 * 93);
  struct Golden { int channel, frame; float value; };
  const Golden golden[] = {{0, 0, -1.16011894f}, {5, 10, -7.24690628f},
                           {20, 50, -4.52280998f}, {127, 92, -7.87227249f}};
  for (const Golden& g : golden) {
    const float got = mel.log_mel[static_cast<size_t>(g.channel) * 93 + g.frame];
    if (!(std::fabs(got - g.value) < 2e-4f)) {
      std::cerr << "mel mismatch channel=" << g.channel << " frame=" << g.frame
                << " got=" << got << " want=" << g.value << "\n";
      std::exit(1);
    }
  }
  bool threw = false;
  try {
    (void)compute_speaker_log_mel(std::vector<float>(384));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}

static void test_real_config(const std::string& model_dir) {
  if (model_dir.empty()) {
    std::cout << "SKIP test_real_config (no model dir)\n";
    return;
  }
  const TtsModelConfig cfg = TtsModelConfig::from_model_dir(model_dir);
  if (cfg.is_base()) {
    CHECK(cfg.model_size == "1b7");
    TtsModelConfig expected = TtsModelConfig::preset_1_7b();
    expected.model_type = "base";
    expected.speaker_encoder = {2048, 24000};
    expected.speaker_ids.clear();
    expected.language_ids.erase("beijing_dialect");
    expected.language_ids.erase("sichuan_dialect");
    check_same(cfg, expected, "real Base 1.7B");
  } else if (cfg.model_size == "0b6")
    check_same(cfg, TtsModelConfig::preset_0_6b(), "real 0.6B");
  else if (cfg.model_size == "1b7")
    check_same(cfg, TtsModelConfig::preset_1_7b(), "real 1.7B");
  else {
    std::cerr << "unexpected tts_model_size: " << cfg.model_size << "\n";
    std::exit(1);
  }
  std::cout << "real config parsed: size=" << cfg.model_size
            << " hidden=" << cfg.hidden << " plan=" << cfg.tensor_shape_plan().size()
            << "\n";
}

int main(int argc, char** argv) {
  test_parsing();
  test_shape_plan();
  test_rejects();
  test_projection_rule();
  test_speaker_mel();
  const char* env = std::getenv("UMM_TTS_TEST_DATA");
  test_real_config(argc > 1 ? argv[1] : (env ? env : ""));
  std::cout << "all tts config tests passed\n";
  return 0;
}
