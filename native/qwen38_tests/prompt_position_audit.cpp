// SPDX-License-Identifier: Apache-2.0
// Host-only tokenizer/prompt audit for temporal-quality intervention evidence.
// This tool never constructs an Engine and never initializes CUDA. Python
// independently derives MRoPE coordinates from the emitted expanded token ids.
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "qwen38/engine.hpp"
#include "qwen38/tokenizer.hpp"

namespace {

using json = nlohmann::json;

json read_json(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::invalid_argument("failed to open audit input: " + path);
  return json::parse(input);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7 || std::string(argv[1]) != "--model" ||
      std::string(argv[3]) != "--input" || std::string(argv[5]) != "--output") {
    std::cerr << "usage: qwen38_prompt_position_audit --model DIR --input IN.json "
                 "--output OUT.json\n";
    return 2;
  }
  try {
    const umminfer::qwen38::Qwen2BpeTokenizer tokenizer(
        std::string(argv[2]) + "/vocab.json", std::string(argv[2]) + "/merges.txt",
        std::string(argv[2]) + "/tokenizer_config.json");
    const json input = read_json(argv[4]);
    if (input.at("schema_version") != 1 || !input.at("requests").is_array() ||
        input.at("requests").empty()) {
      throw std::invalid_argument("invalid prompt-position audit request list");
    }
    json output = {{"schema_version", 1}, {"requests", json::array()}};
    for (const auto& request : input.at("requests")) {
      const std::string id = request.at("id").get<std::string>();
      const std::string prompt = request.at("prompt").get<std::string>();
      const std::vector<double> timestamps =
          request.at("timestamps").get<std::vector<double>>();
      const int grid_h = request.at("grid_h").get<int>();
      const int grid_w = request.at("grid_w").get<int>();
      if (id.empty() || prompt.empty() || timestamps.empty() || grid_h <= 0 ||
          grid_w <= 0 || grid_h % 2 != 0 || grid_w % 2 != 0) {
        throw std::invalid_argument("invalid prompt-position audit request: " + id);
      }
      umminfer::qwen38::ChatTurn turn;
      turn.role = "user";
      turn.text = prompt;
      turn.video_slices = static_cast<int>(timestamps.size());
      turn.video_timestamps = timestamps;
      const auto rendered = umminfer::qwen38::render_chat_prompt_parts(
          {turn}, /*enable_thinking=*/false);
      const int tokens_per_slice = grid_h * grid_w / 4;
      const std::vector<int> ids = umminfer::qwen38::expand_video_pads(
          tokenizer.encode(rendered.full()), tokens_per_slice,
          static_cast<int>(timestamps.size()));
      output["requests"].push_back({
          {"id", id}, {"input_prompt", prompt}, {"rendered_prompt", rendered.full()},
          {"expanded_ids", ids},
          {"grid_t", static_cast<int>(timestamps.size())}, {"grid_h", grid_h},
          {"grid_w", grid_w}, {"merged_tokens_per_slice", tokens_per_slice},
          {"timestamps", timestamps},
      });
    }
    std::ofstream output_file(argv[6], std::ios::binary | std::ios::trunc);
    if (!output_file) throw std::runtime_error("failed to create audit output");
    output_file << output.dump(2) << '\n';
    if (!output_file) throw std::runtime_error("failed to write audit output");
  } catch (const std::exception& error) {
    std::cerr << "prompt-position audit failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
