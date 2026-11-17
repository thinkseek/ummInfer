// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "image_preprocess.hpp"

namespace umminfer::qwen38 {

struct GenResult {
  std::vector<int> ids;        // generated token ids, EOS excluded
  int prompt_tokens = 0;
  int image_tokens = 0;
  double vision_ms = 0.0;      // image tower + merger
  double prefill_ms = 0.0;     // embedding -> first-token logits (excludes vision)
  double decode_ms = 0.0;      // steps after the first generated token
  int decode_steps = 0;
};

// Native Qwen3.8-27B (Qwen3_5ForConditionalGeneration) executor.
class Engine {
 public:
  // max_seq bounds prompt + generated tokens; the KV cache is allocated once.
  Engine(const std::string& model_dir, int gpu, int max_seq = 8192);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // prompt_ids must already contain one image_pad id per merged vision token.
  // dump_dir, when non-empty, receives per-layer bf16 tensors for parity tests.
  GenResult generate(const std::vector<int>& prompt_ids,
                     const PreprocessedImage* image, int max_new_tokens,
                     const std::string& dump_dir = "");

  // Opaque state; public only so the translation-unit helpers can name it.
  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

// Prompt helpers shared with the CLI and tests.
std::string render_chat_prompt(const std::string& user_text, bool has_image,
                               bool enable_thinking);
// Expand a single image_pad placeholder to n_tokens copies.
std::vector<int> expand_image_pads(const std::vector<int>& ids, int n_tokens);

}  // namespace umminfer::qwen38
