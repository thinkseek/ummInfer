// SPDX-License-Identifier: Apache-2.0
// Test-only fuzz driver: reads one UTF-8 line per stdin line, emits the BPE
// id vector space-separated. Compared against the Python tokenizers 0.22.2
// reference by tok_fuzz.py.
#include "bpe_tokenizer.hpp"

#include <iostream>
#include <string>

using namespace umminfer::native;

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: tok_fuzz MODEL_DIR < lines\n";
    return 2;
  }
  Qwen2BpeTokenizer tok(argv[1]);
  std::string line;
  // NUL-delimited records so embedded CR/LF survive transport.
  while (std::getline(std::cin, line, '\0')) {
    const auto ids = tok.encode(line);
    for (size_t i = 0; i < ids.size(); ++i) {
      if (i) std::cout << ' ';
      std::cout << ids[i];
    }
    std::cout << '\n';
  }
  return 0;
}
