// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "image_preprocess.hpp"

namespace umminfer::qwen38 {

// Thrown when the abort check fires. The caller decides the terminal state;
// the engine stops at the boundary it reached, so no partial result escapes.
struct GenerationAborted : std::runtime_error {
  explicit GenerationAborted(std::string boundary)
      : std::runtime_error(std::move(boundary)) {}
};

using AbortCheck = std::function<bool()>;
// Receives each generated token id before the next forward pass.
using TokenCallback = std::function<void(int)>;

// Sampling knobs for one request. The defaults are exactly the original
// executor: temperature <= 0 selects argmax, so an unset request stays greedy
// and bit-identical to the previous release.
struct SamplingParams {
  float temperature = 0.0f;
  float top_p = 1.0f;
  uint64_t seed = 0;
  bool has_seed = false;
};

struct GenResult {
  std::vector<int> ids;        // generated token ids, EOS excluded
  bool stopped_by_eos = false;
  // prompt_tokens is the whole prompt; prefill_tokens is what this call
  // actually computed, and reused_tokens came from the prefix snapshot.
  int prompt_tokens = 0;
  int prefill_tokens = 0;
  int reused_tokens = 0;
  int image_tokens = 0;
  double vision_ms = 0.0;      // image tower + merger
  double prefill_ms = 0.0;     // embedding -> first-token logits (excludes vision)
  double decode_ms = 0.0;      // steps after the first generated token
  int decode_steps = 0;
};

// One request for the continuous-batching path.
struct BatchRequest {
  std::vector<int> prompt_ids;
  const PreprocessedImage* image = nullptr;  // must outlive the call
  int max_new_tokens = 0;
  SamplingParams sampling;
  AbortCheck abort;
  TokenCallback on_token;
};

// Native Qwen3.8-27B (Qwen3_5ForConditionalGeneration) executor.
//
// Thread-safety contract: an engine with batch_capacity == 1 holds one CUDA
// stream and one set of buffers, so generate() must run on one thread at a
// time. An engine with batch_capacity > 1 owns a single batch thread and
// generate_batched() may be called from several threads at once; every call
// queues its request and blocks until its own result is ready. generate() and
// generate_batched() are mutually exclusive on one engine.
class Engine {
 public:
  // max_seq bounds prompt + generated tokens; the KV cache is allocated once.
  // batch_capacity is the number of sequences that decode together; 1 keeps
  // the original single-request executor and its memory footprint.
  Engine(const std::string& model_dir, int gpu, int max_seq = 8192,
         int batch_capacity = 1);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // prompt_ids must already contain one image_pad id per merged vision token.
  // dump_dir, when non-empty, receives per-layer bf16 tensors for parity tests.
  // An empty abort check and an empty token callback keep the original run.
  //
  // A pure-text call whose ids extend the previous successful pure-text prompt
  // reuses that call's snapshot: KV rows, the linear-attention recurrent state
  // and the conv history at the cached length. The reused region was written
  // by prefill, so a reused run is bit-identical to a full prefill.
  GenResult generate(const std::vector<int>& prompt_ids,
                     const PreprocessedImage* image, int max_new_tokens,
                     const std::string& dump_dir = "",
                     const AbortCheck& abort = {},
                     const TokenCallback& on_token = {},
                     const SamplingParams& sampling = {});

  // Continuous batching: the request joins the batch thread's queue, decodes
  // next to the other active requests, and leaves the batch on EOS or at its
  // token limit. Callable concurrently from up to batch_capacity threads.
  GenResult generate_batched(const BatchRequest& request);

  int batch_capacity() const;

  // Opaque state; public only so the translation-unit helpers can name it.
  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

// Prompt helpers shared with the CLI, the native stage, and tests.
struct ChatTurn {
  std::string role;
  std::string text;
  bool has_image = false;
};

std::string render_chat_prompt(const std::string& user_text, bool has_image,
                               bool enable_thinking);
std::string render_chat_prompt(const std::vector<ChatTurn>& turns,
                               bool enable_thinking);
// Expand a single image_pad placeholder to n_tokens copies.
std::vector<int> expand_image_pads(const std::vector<int>& ids, int n_tokens);

}  // namespace umminfer::qwen38
