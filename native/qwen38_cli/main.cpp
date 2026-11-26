// SPDX-License-Identifier: Apache-2.0
// umm-qwen38-native: model directory + prompt + optional image -> answer.
// No Python, no PyTorch, no SGLang, no cuBLAS; only the CUDA runtime.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "qwen38/engine.hpp"
#include "qwen38/image_preprocess.hpp"
#include "qwen38/tokenizer.hpp"
#include "qwen38_tests/independent_logsoftmax.hpp"
#include "umminfer/obs/behavior_evidence.hpp"
#include "umminfer/util/codec.hpp"
#include "pipeline/narrative_parser.hpp"
#include "pipeline/unified_schema.hpp"

namespace {

struct Args {
  std::string model_dir;
  std::string prompt;
  std::string image;
  std::string dump_dir;
  int gpu = 0;
  int max_new = 256;
  bool thinking = false;
  bool behavior = false;
  bool independent_behavior = false;
  bool narrative_image = false;
  bool narrative_auto = false;
  float temperature = 0.0f;
  float top_p = 1.0f;
  uint64_t seed = 0;
  bool has_seed = false;
};

void usage() {
  std::cerr << "usage: umm-qwen38-native --model DIR --prompt TEXT [--image PATH] "
               "[--gpu N] [--max-new-tokens M] [--thinking] [--dump-dir DIR] "
               "[--behavior-evidence|--independent-behavior-evidence] "
               "[--narrative-image|--narrative-auto] [--temperature T] "
               "[--top-p P] [--seed N]\n";
}

using umminfer::qwen38::PreprocessedImage;

}  // namespace

int main(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << "\n";
        usage();
        std::exit(2);
      }
      return argv[++i];
    };
    if (flag == "--model")
      args.model_dir = need("--model");
    else if (flag == "--prompt")
      args.prompt = need("--prompt");
    else if (flag == "--image")
      args.image = need("--image");
    else if (flag == "--gpu")
      args.gpu = std::stoi(need("--gpu"));
    else if (flag == "--max-new-tokens")
      args.max_new = std::stoi(need("--max-new-tokens"));
    else if (flag == "--dump-dir")
      args.dump_dir = need("--dump-dir");
    else if (flag == "--thinking")
      args.thinking = true;
    else if (flag == "--behavior-evidence")
      args.behavior = true;
    else if (flag == "--independent-behavior-evidence")
      args.independent_behavior = true;
    else if (flag == "--narrative-image")
      args.narrative_image = true;
    else if (flag == "--narrative-auto")
      args.narrative_auto = true;
    else if (flag == "--temperature")
      args.temperature = std::stof(need("--temperature"));
    else if (flag == "--top-p")
      args.top_p = std::stof(need("--top-p"));
    else if (flag == "--seed") {
      args.seed = static_cast<uint64_t>(std::stoll(need("--seed")));
      args.has_seed = true;
    } else {
      std::cerr << "unknown argument: " << flag << "\n";
      usage();
      return 2;
    }
  }
  if (args.model_dir.empty() || args.prompt.empty() ||
      (args.narrative_image && args.narrative_auto) ||
      (args.behavior && args.independent_behavior)) {
    usage();
    return 2;
  }

  try {
    using umminfer::qwen38::Qwen2BpeTokenizer;
    auto tokenizer_source = umminfer::qwen38::Qwen2BpeTokenizerSource::from_files(
        args.model_dir + "/vocab.json", args.model_dir + "/merges.txt",
        args.model_dir + "/tokenizer_config.json");
    const std::string tokenizer_sha256 = umminfer::obs::loaded_tokenizer_sha256({
        {"vocab.json", tokenizer_source.vocab_json},
        {"merges.txt", tokenizer_source.merges_txt},
        {"tokenizer_config.json", tokenizer_source.tokenizer_config_json},
    });
    Qwen2BpeTokenizer tokenizer(std::move(tokenizer_source));

    std::unique_ptr<PreprocessedImage> image;
    int n_img = 0;
    if (!args.image.empty()) {
      image = std::make_unique<PreprocessedImage>(
          umminfer::qwen38::preprocess_image(args.image));
      if (image->grid_t != 1)
        throw std::runtime_error("only still images are supported");
      n_img = image->rows() / 4;
    }

    std::string rendered;
    if (args.narrative_image || args.narrative_auto) {
      std::vector<umminfer::qwen38::ChatTurn> turns;
      turns.push_back(umminfer::qwen38::ChatTurn{
          "system", umminfer::unified_detail::narrative_director_prompt(
                        args.narrative_auto ? "auto" : "image", 1, "off"),
          false, 0, {}});
      turns.push_back(umminfer::qwen38::ChatTurn{
          "user", args.prompt, image != nullptr, 0, {}});
      rendered = umminfer::qwen38::render_chat_prompt_parts(
                     turns, args.thinking)
                     .full();
    } else {
      rendered = umminfer::qwen38::render_chat_prompt(
          args.prompt, image != nullptr, args.thinking);
    }
    std::vector<int> ids = tokenizer.encode(rendered);
    if (image) ids = umminfer::qwen38::expand_image_pads(ids, n_img);

    const auto t0 = std::chrono::steady_clock::now();
    auto engine = std::make_unique<umminfer::qwen38::Engine>(args.model_dir, args.gpu);
    const auto t1 = std::chrono::steady_clock::now();
    umminfer::qwen38::SamplingParams sampling;
    sampling.temperature = args.temperature;
    sampling.top_p = args.top_p;
    sampling.seed = args.seed;
    sampling.has_seed = args.has_seed;
    sampling.capture_behavior_logprobs = args.behavior;
    std::vector<int64_t> independent_ids;
    std::vector<double> independent_logprobs;
    nlohmann::json independent_rows = nlohmann::json::array();
    if (args.independent_behavior) {
      sampling.pre_sampling_logits_observer =
          [&independent_ids, &independent_logprobs, &independent_rows](
              int id, const float* logits, int vocab_size) {
            umminfer::qwen38_validation::IndependentLogSoftmax scored;
            if (!umminfer::qwen38_validation::independent_presampling_logsoftmax(
                    logits, vocab_size, id, &scored)) {
              throw std::runtime_error(
                  "independent pre-sampling FP32 log-softmax rejected a row");
            }
            const std::string_view raw_row(
                reinterpret_cast<const char*>(logits),
                static_cast<size_t>(vocab_size) * sizeof(float));
            independent_ids.push_back(id);
            independent_logprobs.push_back(scored.logprob);
            independent_rows.push_back({
                {"token_id", id},
                {"vocab_size", vocab_size},
                {"raw_logits_sha256", umminfer::util::sha256_hex(raw_row)},
                {"selected_raw_logit", scored.selected_raw_logit},
                {"max_raw_logit", scored.max_raw_logit},
                {"log_normalizer", scored.log_normalizer},
                {"logprob", scored.logprob},
            });
          };
    }
    auto result = engine->generate(ids, image.get(), args.max_new, args.dump_dir,
                                   {}, {}, sampling);
    const auto t2 = std::chrono::steady_clock::now();

    if (args.behavior || args.independent_behavior) {
      if (!result.stopped_by_eos) {
        throw std::runtime_error("behavior replay requires an EOS-complete generation");
      }
      std::vector<int64_t> behavior_ids(result.ids.begin(), result.ids.end());
      std::optional<std::vector<int64_t>> narrative_mask;
      if (args.narrative_image || args.narrative_auto) {
        narrative_mask = umminfer::narrative_behavior_masks(
            result.ids, [&tokenizer](int id) { return tokenizer.decode({id}); });
        if (!narrative_mask) {
          throw std::runtime_error("narrative token/prose alignment failed");
        }
      }
      const std::vector<int64_t> mask =
          narrative_mask ? *narrative_mask
                         : std::vector<int64_t>(result.ids.size(), 1);
      std::string clean;
      for (size_t i = 0; i < result.ids.size(); ++i) {
        if (mask[i] == 1) clean += tokenizer.decode({result.ids[i]});
      }
      nlohmann::json out = {
          {"prompt_token_ids", ids},
          {"raw_text", tokenizer.decode(result.ids)},
          {"clean_text", clean},
          {"finish_reason", "stop"},
      };
      if (args.independent_behavior) {
        if (independent_ids != behavior_ids ||
            independent_logprobs.size() != behavior_ids.size() ||
            independent_rows.size() != behavior_ids.size()) {
          throw std::runtime_error("independent raw-logit observation is incomplete");
        }
        out["independent_behavior"] = {
            {"format", "umminfer-independent-presampling-logsoftmax-v1"},
            {"tokenizer_sha256", tokenizer_sha256},
            {"token_ids", independent_ids},
            {"logprobs", independent_logprobs},
            {"mask", mask},
            {"rows", std::move(independent_rows)},
            {"input_precision", "FP32_PRE_SAMPLING_LM_HEAD_LOGITS"},
            {"arithmetic", "INDEPENDENT_LONG_DOUBLE_NORMALIZER_FLOAT64_OUTPUT"},
            {"producer_scoring_used", false},
        };
      } else {
        const umminfer::Value evidence = umminfer::obs::behavior_logprobs_value(
            behavior_ids, result.behavior_logprobs, mask, tokenizer_sha256);
        if (evidence.is_nil()) {
          throw std::runtime_error("native behavior evidence is incomplete or invalid");
        }
        out["behavior"] = {
            {"format", umminfer::obs::kBehaviorLogprobsFormat},
            {"tokenizer_sha256", tokenizer_sha256},
            {"token_ids", behavior_ids},
            {"logprobs", result.behavior_logprobs},
            {"mask", mask},
        };
      }
      std::cout << out.dump() << "\n";
    } else {
      std::cout << tokenizer.decode(result.ids) << "\n";
    }

    const double load_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double wall_ms =
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    const double decode_tok_s =
        result.decode_ms > 0.0 ? result.decode_steps * 1000.0 / result.decode_ms : 0.0;
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    std::cerr << "load_ms=" << static_cast<long long>(load_ms) << "\n"
              << "prompt_tokens=" << result.prompt_tokens << "\n"
              << "image_tokens=" << result.image_tokens << "\n"
              << "vision_ms=" << static_cast<long long>(result.vision_ms) << "\n"
              << "prefill_ms=" << static_cast<long long>(result.prefill_ms) << "\n"
              << "decode_steps=" << result.decode_steps << "\n"
              << "decode_ms=" << static_cast<long long>(result.decode_ms) << "\n"
              << "decode_tok_per_s=" << decode_tok_s << "\n"
              << "wall_ms=" << static_cast<long long>(wall_ms) << "\n"
              << "gpu_used_mb=" << (total_b - free_b) / (1024 * 1024) << "\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
