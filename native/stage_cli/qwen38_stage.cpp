// SPDX-License-Identifier: Apache-2.0
// qwen38 kind: Qwen3.8-27B (text + one optional image or native video) behind
// the Stage protocol. Prompt rendering and visual preprocessing follow the
// native Qwen3.8 host-tested helpers.
//
// Deltas: every generated token is detokenized and the new suffix is sent as a
// stream chunk {"text": ...}. The HTTP layer forwards them as SSE deltas and
// still receives the aggregate result at completion.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

// Same subset the accepted image_preprocess.cpp compiles in, so the decoder
// declarations match its implementation.
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_NO_THREAD_LOCALS
#include "qwen38/stb_image.h"

#include "qwen38/engine.hpp"
#include "qwen38/image_preprocess.hpp"
#include "qwen38/prefix_cache.hpp"
#include "qwen38/tokenizer.hpp"
#include "qwen38/video_input.hpp"
#include "stage_common.hpp"
#include "stage_kind.hpp"
#include "umminfer/obs/behavior_evidence.hpp"
#include "pipeline/narrative_parser.hpp"

namespace umminfer::stage {
namespace {

constexpr int kDefaultMaxNewTokens = 256;
constexpr int kMinPixels = 65536;
constexpr int kMaxPixels = 16777216;

// Adapts the shared, host-tested sampling parse to the engine's struct. The
// engine's type is local to the CUDA tree, so the fields are copied explicitly
// rather than depending on layout compatibility.
qwen38::SamplingParams parse_sampling(const Value::Map& params) {
  const SamplingParams parsed = parse_sampling_params(params);
  qwen38::SamplingParams sampling;
  sampling.temperature = parsed.temperature;
  sampling.top_p = parsed.top_p;
  sampling.seed = parsed.seed;
  sampling.has_seed = parsed.has_seed;
  return sampling;
}

qwen38::PreprocessedImage decode_image(const std::string& url) {
  const std::string bytes = load_image_bytes(url);
  int width = 0;
  int height = 0;
  int channels = 0;
  stbi_uc* pixels = stbi_load_from_memory(
      reinterpret_cast<const stbi_uc*>(bytes.data()),
      static_cast<int>(bytes.size()), &width, &height, &channels, 3);
  if (pixels == nullptr) {
    throw ProtocolError(std::string("image decode failed: ") + stbi_failure_reason());
  }
  std::vector<uint8_t> rgb(pixels, pixels + static_cast<size_t>(width) * height * 3);
  stbi_image_free(pixels);
  return qwen38::preprocess_rgb(rgb.data(), height, width, kMinPixels, kMaxPixels);
}

std::string hex_u64(uint64_t value) {
  char buffer[17];
  std::snprintf(buffer, sizeof(buffer), "%016llx",
                static_cast<unsigned long long>(value));
  return buffer;
}

// Non-cryptographic diagnostic digest of the exact position vectors passed to
// the same host helper used by serial and batched engine prefill.
std::string mrope_digest(const qwen38::MRoPEPositions& positions) {
  uint64_t hash = 0xcbf29ce484222325ULL;
  auto absorb = [&hash](int value) {
    hash ^= static_cast<uint32_t>(value);
    hash *= 0x100000001b3ULL;
  };
  for (size_t i = 0; i < positions.t.size(); ++i) {
    absorb(positions.t[i]);
    absorb(positions.h[i]);
    absorb(positions.w[i]);
  }
  return hex_u64(hash);
}

}  // namespace

const char* compiled_kind() { return "qwen38"; }

const char* stage_usage_flags() {
  return "[--max-new-tokens N] [--max-seq N] [--batch-size N] ";
}

bool parse_stage_flag(const std::string& key, const std::string& value,
                      StageOptions& options) {
  if (key == "max-new-tokens") {
    options.max_new_tokens = parse_stage_int(value, "--max-new-tokens");
    return true;
  }
  if (key == "max-seq") {
    options.max_seq = parse_stage_int(value, "--max-seq");
    return true;
  }
  if (key == "batch-size") {
    options.batch_size = parse_stage_int(value, "--batch-size");
    return true;
  }
  return false;
}

CancellableComputeFn make_compute(const StageOptions& options) {
  const int max_new_tokens =
      options.max_new_tokens > 0 ? options.max_new_tokens : kDefaultMaxNewTokens;
  auto tokenizer_source = qwen38::Qwen2BpeTokenizerSource::from_files(
      options.model_dir + "/vocab.json", options.model_dir + "/merges.txt",
      options.model_dir + "/tokenizer_config.json");
  const std::string tokenizer_sha256 = loaded_tokenizer_sha256({
      {"vocab.json", tokenizer_source.vocab_json},
      {"merges.txt", tokenizer_source.merges_txt},
      {"tokenizer_config.json", tokenizer_source.tokenizer_config_json},
  });
  auto tokenizer = std::make_shared<qwen38::Qwen2BpeTokenizer>(
      std::move(tokenizer_source));
  if (tokenizer_sha256.empty() || tokenizer->id_of("<|im_end|>") != 248046 ||
      tokenizer->id_of("<|endoftext|>") != 248044) {
    throw ProtocolError("qwen38 tokenizer identity/EOS contract mismatch");
  }
  // batch_size > 1 enables continuous batching; the engine owns one batch
  // thread and is safe to call from several scheduler workers.
  const int batch_size = options.batch_size > 1 ? options.batch_size : 1;
  const int max_seq = options.max_seq > 0 ? options.max_seq : 8192;
  auto engine = std::make_shared<qwen38::Engine>(
      options.model_dir, options.gpu, max_seq, batch_size);

  const int gpu = options.gpu;
  return [tokenizer, tokenizer_sha256, engine, max_new_tokens, max_seq, gpu](
             const Value& work, const AbortCheck& aborted) {
    const auto started = std::chrono::steady_clock::now();
    // The compute function runs on the Stage scheduler thread, whose CUDA
    // current device defaults to 0. The engine's buffers belong to --gpu.
    if (cudaSetDevice(gpu) != cudaSuccess) {
      throw ProtocolError("cudaSetDevice(" + std::to_string(gpu) + ") failed");
    }
    AbortTracker tracker(aborted);
    const OmniWork input = parse_qwen38_work(work);
    const std::string req_id = bound_request_id();
    const int req_max_new_tokens =
        resolve_max_new_tokens(bound_request_params(), max_new_tokens, max_seq);
    qwen38::SamplingParams sampling = parse_sampling(bound_request_params());
    const Value::Map& request_params = bound_request_params();
    const bool capture_behavior =
        engine->batch_capacity() == 1 &&
        obs::behavior_logprobs_requested(request_params);
    sampling.capture_behavior_logprobs = capture_behavior;
    const auto narrative_param = request_params.find(obs::kBehaviorNarrativeParam);
    const bool narrative_behavior =
        narrative_param != request_params.end() && narrative_param->second.is_bool() &&
        narrative_param->second.as_bool();

    std::unique_ptr<qwen38::PreprocessedImage> image;
    std::unique_ptr<qwen38::DecodedVideo> decoded_video;
    std::vector<double> video_timestamps;
    bool is_video = false;
    int image_tokens = 0;
    double video_input_ms = 0.0;
    if (!input.image_url.empty()) {
      image = std::make_unique<qwen38::PreprocessedImage>(decode_image(input.image_url));
      if (image->grid_t != 1) {
        throw ProtocolError("qwen38 image path expects grid_t == 1");
      }
      image_tokens = image->rows() / 4;
    } else if (!input.video_url.empty()) {
      qwen38::VideoFrameSelection selection;
      selection.policy = input.frame_selection.temporal
                             ? qwen38::VideoFramePolicy::kTemporal
                             : qwen38::VideoFramePolicy::kUniform;
      selection.max_frames = input.frame_selection.max_frames;
      selection.motion_threshold = input.frame_selection.motion_threshold;
      selection.scene_threshold = input.frame_selection.scene_threshold;
      try {
        const auto video_started = std::chrono::steady_clock::now();
        decoded_video = std::make_unique<qwen38::DecodedVideo>(
            qwen38::decode_video_bytes(load_video_bytes(input.video_url), selection));
        video_input_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - video_started).count();
      } catch (const std::exception& error) {
        throw ProtocolError(std::string("video decode failed: ") + error.what());
      }
      video_timestamps = decoded_video->timestamps;
      image = std::make_unique<qwen38::PreprocessedImage>(
          std::move(decoded_video->preprocessed));
      is_video = true;
      image_tokens = (image->grid_h * image->grid_w) / 4;
    }
    const auto prompt_started = std::chrono::steady_clock::now();
    std::vector<qwen38::ChatTurn> turns;
    turns.reserve(input.messages.size() + 1);
    bool any_marked = false;
    for (const auto& m : input.messages) {
      if (is_video ? m.has_video : m.has_image) {
        any_marked = true;
        break;
      }
    }
    for (size_t i = 0; i < input.messages.size(); ++i) {
      qwen38::ChatTurn turn;
      turn.role = input.messages[i].role;
      turn.text = input.messages[i].text;
      const bool attach_here =
          is_video
              ? (input.messages[i].has_video ||
                 (!any_marked && i + 1 == input.messages.size()))
              : (input.messages[i].has_image ||
                 (!any_marked && i + 1 == input.messages.size()));
      if (image != nullptr && attach_here) {
        if (is_video) {
          turn.video_slices = image->grid_t;
          turn.video_timestamps = video_timestamps;
        } else {
          turn.has_image = true;
        }
      }
      turns.push_back(std::move(turn));
    }
    if (turns.empty()) {
      qwen38::ChatTurn turn;
      turn.role = "user";
      turn.text = input.text;
      if (image != nullptr) {
        if (is_video) {
          turn.video_slices = image->grid_t;
          turn.video_timestamps = video_timestamps;
        } else {
          turn.has_image = true;
        }
      }
      turns.push_back(std::move(turn));
    }
    // The prompt splits into the completed turns and the open assistant header.
    // The header is not cacheable: the next turn replaces it with the reply, so
    // a snapshot taken after it never matches.
    const qwen38::RenderedChatPrompt parts =
        qwen38::render_chat_prompt_parts(turns, /*enable_thinking=*/false);
    std::vector<int> ids = tokenizer->encode(parts.full());
    // Snapshot at the turn boundary for both pure-text and multimodal prompts.
    // Visual placeholders live in parts.prefix (never in the open assistant
    // header), so expanding both keeps prefix_ids a strict leading slice of
    // ids with all image/video pads strictly inside [0, cache_prefix_len).
    std::vector<int> prefix_ids = tokenizer->encode(parts.prefix);
    if (image != nullptr) {
      if (is_video) {
        ids = qwen38::expand_video_pads(ids, image_tokens, image->grid_t);
        prefix_ids =
            qwen38::expand_video_pads(prefix_ids, image_tokens, image->grid_t);
      } else {
        ids = qwen38::expand_image_pads(ids, image_tokens);
        prefix_ids = qwen38::expand_image_pads(prefix_ids, image_tokens);
      }
    }
    int cache_prefix_len = -1;
    if (prefix_ids.size() < ids.size() &&
        std::equal(prefix_ids.begin(), prefix_ids.end(), ids.begin())) {
      cache_prefix_len = static_cast<int>(prefix_ids.size());
    }

    if (decoded_video != nullptr) {
      // One bounded, request-keyed record makes the actual selected source
      // frames, temporal-2 packing and serial/batched MRoPE inputs auditable
      // without adding anything to the public response schema.
      const qwen38::MRoPEPositions mrope = qwen38::build_mrope_positions(
          ids, image->grid_t, image->grid_h, image->grid_w, true);
      nlohmann::json evidence = {
          {"request_id", req_id},
          {"policy", input.frame_selection.temporal ? "temporal" : "uniform"},
          {"decoded_frame_count", decoded_video->decoded_frame_count},
          {"selected_frame_count", decoded_video->selected_frame_indices.size()},
          {"selected_candidate_indices", decoded_video->selected_frame_indices},
          {"selected_source_frame_indices",
           decoded_video->selected_source_frame_indices},
          {"source_indices_exact", decoded_video->source_indices_exact},
          {"selected_frame_timestamps", decoded_video->selected_frame_timestamps},
          {"selected_frame_content_fnv1a64",
           decoded_video->selected_frame_content_fnv1a64},
          {"has_nominal_timestamps", decoded_video->has_nominal_timestamps},
          {"temporal2_pair_first_timestamps", video_timestamps},
          {"max_frames", input.frame_selection.max_frames},
          {"max_candidate_frames", qwen38::kVideoCandidateFrameBudget},
          {"motion_threshold", input.frame_selection.motion_threshold},
          {"scene_threshold", input.frame_selection.scene_threshold},
          {"grid_t", image->grid_t},
          {"grid_h", image->grid_h},
          {"grid_w", image->grid_w},
          {"premerge_patch_rows", image->rows()},
          {"merged_tokens_per_slice", image_tokens},
          {"video_token_count", image_tokens * image->grid_t},
          {"prompt_token_count", ids.size()},
          {"media_hash", hex_u64(qwen38::compute_media_hash(image.get()))},
          {"mrope_next", mrope.next},
          {"mrope_fnv1a64", mrope_digest(mrope)},
          {"engine_path", engine->batch_capacity() > 1 ? "batched" : "serial"},
      };
      std::cerr << ("qwen38_video_evidence " + evidence.dump() + "\n");
    }

    // Byte-level BPE decode is additive per token: buffer only incomplete
    // trailing UTF-8 bytes so streaming stays O(1) per token instead of O(T^2).
    std::string pending_utf8;
    // Copy the sink and id bound to this compute call. In batched mode the
    // callback fires on the engine's batch thread, where the scheduler's
    // thread-locals are not bound.
    const StreamFn stream = bound_stream_fn();
    auto emit_token = [&tokenizer, &pending_utf8, stream, req_id](int id) {
      pending_utf8 += tokenizer->decode({id});
      const size_t safe = utf8_complete_prefix(pending_utf8);
      if (safe == 0) return;
      Value::Map chunk;
      chunk["text"] = Value(pending_utf8.substr(0, safe));
      Value value(std::move(chunk));
      if (stream) {
        stream(req_id, std::move(value));
      } else {
        emit_stage_stream(std::move(value));
      }
      pending_utf8.erase(0, safe);
    };

    const double prompt_prep_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - prompt_started).count();
    qwen38::GenResult result;
    try {
      if (engine->batch_capacity() > 1) {
        qwen38::BatchRequest request;
        request.prompt_ids = std::move(ids);
        request.image = image.get();
        request.max_new_tokens = req_max_new_tokens;
        request.sampling = sampling;
        request.abort = [&tracker] { return tracker(); };
        request.on_token = emit_token;
        request.cache_prefix_len = cache_prefix_len;
        result = engine->generate_batched(request);
      } else {
        result = engine->generate(ids, image.get(), req_max_new_tokens, "",
                                  [&tracker] { return tracker(); }, emit_token,
                                  sampling, cache_prefix_len);
      }
    } catch (const qwen38::GenerationAborted& error) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - started);
      std::ostringstream log;
      log << "qwen38 request aborted request_id=" << req_id
          << " boundary=\"" << error.what() << "\""
          << " abort_epoch_ms=" << tracker.detected_epoch_ms()
          << " exit_epoch_ms=" << epoch_millis()
          << " compute_ms=" << elapsed.count() << "\n";
      std::cerr << log.str();
      throw RequestAborted(error.what());
    }
    const std::string text = tokenizer->decode(result.ids);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started);
    const double decode_tok_s = result.decode_ms > 0.0
                                    ? result.decode_steps * 1000.0 / result.decode_ms
                                    : 0.0;
    const char* finish_reason = result.stopped_by_eos ? "stop" : "length";
    std::ostringstream log;
    log << "qwen38 request request_id=" << req_id
        << " prompt_tokens=" << result.prompt_tokens
        << " prefill_tokens=" << result.prefill_tokens
        << " prefix_reused=" << result.reused_tokens
        << " image_tokens=" << result.image_tokens
        << " generated_tokens=" << result.ids.size()
        << " finish_reason=" << finish_reason
        << " chars=" << text.size()
        << " vision_ms=" << std::fixed << std::setprecision(1)
        << result.vision_ms
        << " prefill_ms=" << std::fixed << std::setprecision(1)
        << result.prefill_ms
        << " decode_ms=" << std::fixed << std::setprecision(1)
        << result.decode_ms
        << " decode_tok_per_s=" << std::fixed << std::setprecision(1)
        << decode_tok_s
        << " video_input_ms=" << std::fixed << std::setprecision(3)
        << video_input_ms
        << " prompt_prep_ms=" << prompt_prep_ms
        << " unattributed_ms="
        << (static_cast<double>(elapsed.count()) - video_input_ms - prompt_prep_ms -
            result.vision_ms - result.prefill_ms - result.decode_ms)
        << " elapsed_ms=" << elapsed.count() << "\n";
    std::cerr << log.str();
    Value output = omni_result(text, static_cast<int>(result.ids.size()),
                               result.prefill_ms, result.decode_ms, finish_reason);
    if (capture_behavior && result.stopped_by_eos) {
      std::vector<int64_t> behavior_ids(result.ids.begin(), result.ids.end());
      std::optional<std::vector<int64_t>> narrative_mask;
      if (narrative_behavior) {
        narrative_mask = narrative_behavior_masks(
            result.ids, [&tokenizer](int id) { return tokenizer->decode({id}); });
      }
      const std::vector<int64_t> mask =
          narrative_mask ? *narrative_mask
                         : std::vector<int64_t>(result.ids.size(), 1);
      Value evidence = obs::behavior_logprobs_value(
          behavior_ids, result.behavior_logprobs, mask, tokenizer_sha256);
      // A failed narrative alignment must not fall back to an all-ones mask.
      if ((!narrative_behavior || narrative_mask.has_value()) &&
          !evidence.is_nil()) {
        output[obs::kBehaviorLogprobsResultKey] = std::move(evidence);
      }
    }
    return output;
  };
}

}  // namespace umminfer::stage
