// SPDX-License-Identifier: Apache-2.0
// Host-only replay of observed VLM history messages through the native renderer.
// This tool never constructs an Engine, loads weights, or initializes CUDA.
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "qwen38/engine.hpp"

namespace {
using json = nlohmann::json;
}

int main(int argc, char** argv) {
  if (argc != 5 || std::string(argv[1]) != "--input" ||
      std::string(argv[3]) != "--output") {
    std::cerr << "usage: qwen38_history_prompt_replay --input IN.json --output OUT.json\n";
    return 2;
  }
  try {
    std::ifstream input_file(argv[2], std::ios::binary);
    if (!input_file) throw std::invalid_argument("failed to open history replay input");
    const json input = json::parse(input_file);
    if (input.at("schema") != "umminfer-vlm-history-prompt-replay-v1" ||
        !input.at("requests").is_array() || input.at("requests").empty()) {
      throw std::invalid_argument("invalid history replay request list");
    }
    json output = {{"schema", "umminfer-vlm-history-prompt-replay-v1"},
                   {"requests", json::array()}};
    for (const auto& request : input.at("requests")) {
      const std::string id = request.at("id").get<std::string>();
      if (id.empty() || !request.at("messages").is_array() ||
          request.at("messages").empty()) {
        throw std::invalid_argument("invalid history replay request: " + id);
      }
      std::vector<umminfer::qwen38::ChatTurn> turns;
      for (const auto& message : request.at("messages")) {
        turns.push_back({message.at("role").get<std::string>(),
                         message.at("content").get<std::string>(), false, 0, {}});
      }
      output["requests"].push_back({
          {"id", id},
          {"messages", request.at("messages")},
          {"rendered_prompt", umminfer::qwen38::render_chat_prompt_parts(turns, false).full()},
      });
    }
    std::ofstream output_file(argv[4], std::ios::binary | std::ios::trunc);
    if (!output_file) throw std::runtime_error("failed to create history replay output");
    output_file << output.dump(2) << '\n';
    if (!output_file) throw std::runtime_error("failed to write history replay output");
  } catch (const std::exception& error) {
    std::cerr << "history prompt replay failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
