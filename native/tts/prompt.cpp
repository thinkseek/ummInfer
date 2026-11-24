// SPDX-License-Identifier: Apache-2.0
#include "prompt.hpp"

#include <cctype>
#include <stdexcept>

namespace umminfer::native {
namespace {
PromptTerm text(int id) { return {PromptTerm::kTextProj, id}; }
PromptTerm codec(int id) { return {PromptTerm::kCodec, id}; }
PromptTerm fixed(PromptTerm::Kind k) { return {k, 0}; }

PromptPlan build_plan(const std::vector<int>& text_ids, int language_id,
                      int speaker_id, bool clone) {
  // R = first 3 ids, S = last 5 ids, body T in between.
  if (text_ids.size() < 9)
    throw std::invalid_argument("wrapped prompt must contain R + body + S");
  const std::vector<int> role(text_ids.begin(), text_ids.begin() + 3);
  const std::vector<int> body(text_ids.begin() + 3, text_ids.end() - 5);

  // Codec id stream C (explicit language, speaker present), length 7.
  std::vector<int> c = {kCodecThink, kCodecThinkBos, language_id, kCodecThinkEos,
                        clone ? -1 : speaker_id, kCodecPad, kCodecBos};

  PromptPlan plan;
  plan.codec_prefill_ids = c;
  plan.body_tokens = static_cast<int>(body.size());

  // Stage A: role prefix through the text projection.
  for (int id : role) plan.rows.push_back({text(id)});

  // Stage B: first 6 codec entries, five tts_pad then one tts_bos.
  for (int j = 0; j < 6; ++j)
    plan.rows.push_back({clone && j == 4 ? fixed(PromptTerm::kXVector) : codec(c[j]),
                         j == 5 ? fixed(PromptTerm::kTtsBos)
                                : fixed(PromptTerm::kTtsPad)});

  // Stage C: body text plus tts_eos, each offset by codec_pad embedding.
  for (int id : body) plan.rows.push_back({text(id), codec(kCodecPad)});
  plan.rows.push_back({fixed(PromptTerm::kTtsEos), codec(kCodecPad)});

  // Stage D: final codec_bos position with tts_pad.
  plan.rows.push_back({fixed(PromptTerm::kTtsPad), codec(kCodecBos)});
  return plan;
}
}  // namespace

PromptPlan build_prompt_plan(const std::vector<int>& text_ids, int language_id,
                             int speaker_id) {
  return build_plan(text_ids, language_id, speaker_id, false);
}

PromptPlan build_clone_prompt_plan(const std::vector<int>& text_ids, int language_id) {
  return build_plan(text_ids, language_id, 0, true);
}

int speaker_id_from_name(const std::string& name) {
  std::string n;
  n.reserve(name.size());
  for (char ch : name) n.push_back(static_cast<char>(std::tolower(ch)));
  if (n == "vivian") return kSpkVivian;
  if (n == "ryan") return kSpkRyan;
  throw std::invalid_argument("unsupported speaker (only vivian/ryan wired): " + name);
}

int language_id_from_name(const std::string& name) {
  std::string n;
  n.reserve(name.size());
  for (char ch : name) n.push_back(static_cast<char>(std::tolower(ch)));
  if (n == "chinese" || n == "zh") return kLangChinese;
  if (n == "english" || n == "en") return kLangEnglish;
  throw std::invalid_argument("unsupported language (only chinese/english wired): " + name);
}

}  // namespace umminfer::native
