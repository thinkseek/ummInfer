// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <vector>

namespace umminfer::native {

// Token-space constants from config.json (talker_config / top-level).
inline constexpr int kImStart = 151644;
inline constexpr int kImEnd = 151645;
inline constexpr int kAssistantToken = 77091;
inline constexpr int kNewlineToken = 198;
inline constexpr int kTtsPad = 151671;
inline constexpr int kTtsBos = 151672;
inline constexpr int kTtsEos = 151673;
inline constexpr int kCodecPad = 2148;
inline constexpr int kCodecBos = 2149;
inline constexpr int kCodecEos = 2150;
inline constexpr int kCodecThink = 2154;
inline constexpr int kCodecThinkBos = 2156;
inline constexpr int kCodecThinkEos = 2157;
inline constexpr int kLangChinese = 2055;
inline constexpr int kLangEnglish = 2050;
inline constexpr int kSpkVivian = 3065;
inline constexpr int kSpkRyan = 3061;

// One additive term of a prefill row. Text terms go through text_embedding
// [151936,2048] then the text_projection MLP into width 1024; codec terms and
// the precomputed tts_* vectors already have width 1024.
struct PromptTerm {
  enum Kind { kTextProj, kCodec, kTtsPad, kTtsBos, kTtsEos };
  Kind kind;
  int id = 0;  // text or codec id when kind requires it
};

struct PromptPlan {
  std::vector<std::vector<PromptTerm>> rows;  // rows are summed term-wise
  std::vector<int> codec_prefill_ids;         // C stream, for documentation/tests
  int body_tokens = 0;                         // N = text ids between R and S
};

// Builds the non-streaming CustomVoice prefill plan (modeling_qwen3_tts.py
// generate(), batch 1, explicit language and speaker). text_ids is the encoded
// wrapper R || body || S (R and S are 3 and 5 tokens).
PromptPlan build_prompt_plan(const std::vector<int>& text_ids, int language_id,
                             int speaker_id);

// Resolves the CLI speaker/language names; throws on unsupported values.
int speaker_id_from_name(const std::string& name);
int language_id_from_name(const std::string& name);

}  // namespace umminfer::native
