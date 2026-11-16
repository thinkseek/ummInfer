// SPDX-License-Identifier: Apache-2.0
// Debug helper: print token ids for prompt strings, for parity checks against
// the HuggingFace Qwen2Tokenizer. Not installed.
#include "image/tokenizer.hpp"

#include <cstdio>
#include <fstream>
#include <string>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: tok_dump TOKENIZER_DIR PROMPT [PROMPT...]\n");
    return 2;
  }
  umminfer::image::Qwen2BpeTokenizer tok(argv[1]);
  if (std::string(argv[2]) == "--file") {
    // One prompt per line, read byte-exact (trailing newline stripped once).
    std::ifstream in(argv[3]);
    if (!in) return 2;
    std::string prompt;
    while (std::getline(in, prompt)) {
      auto ids = tok.encode_prompt(prompt);
      std::printf("%zu", ids.size());
      for (int id : ids) std::printf(" %d", id);
      std::printf("\n");
    }
    return 0;
  }
  if (std::string(argv[2]) == "--pretok") {
    for (int i = 3; i < argc; ++i) {
      auto words = tok.debug_pretokens(argv[i]);
      std::printf("%zu", words.size());
      for (const auto& w : words) std::printf(" [%s]", w.c_str());
      std::printf("\n");
    }
    return 0;
  }
  for (int i = 2; i < argc; ++i) {
    auto ids = tok.encode_prompt(argv[i]);
    std::printf("%zu", ids.size());
    for (int id : ids) std::printf(" %d", id);
    std::printf("\n");
  }
  return 0;
}
