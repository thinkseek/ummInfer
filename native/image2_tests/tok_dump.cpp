// SPDX-License-Identifier: Apache-2.0
// Prints the token ids the native tokenizer produces for one prompt.
#include "image2/tokenizer.hpp"

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: image2_tok_dump PROCESSOR_DIR PROMPT\n");
    return 2;
  }
  umminfer::image2::Qwen2BpeTokenizer tok(argv[1]);
  const auto ids = tok.encode_prompt(argv[2]);
  std::printf("drop %d\nids %zu\n", tok.system_prefix_tokens(), ids.size());
  for (size_t i = 0; i < ids.size(); ++i) std::printf("%d%c", ids[i],
                                                      (i + 1) % 16 == 0 ? '\n' : ' ');
  std::printf("\n");
  return 0;
}
