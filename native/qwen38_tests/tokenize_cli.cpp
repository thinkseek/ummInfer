// SPDX-License-Identifier: Apache-2.0
// Debug tool: tokenize a UTF-8 file with the native BPE and print ids.
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "qwen38/tokenizer.hpp"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: qwen38_tokenize MODEL_DIR TEXT_FILE\n";
    return 2;
  }
  std::ifstream f(argv[2], std::ios::binary);
  if (!f) {
    std::cerr << "cannot open " << argv[2] << "\n";
    return 2;
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  umminfer::qwen38::Qwen2BpeTokenizer tok(
      std::string(argv[1]) + "/vocab.json", std::string(argv[1]) + "/merges.txt",
      std::string(argv[1]) + "/tokenizer_config.json");
  const std::vector<int> ids = tok.encode(ss.str());
  for (size_t i = 0; i < ids.size(); ++i)
    std::cout << (i ? " " : "") << ids[i];
  std::cout << "\n";
  return 0;
}
