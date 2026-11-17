// SPDX-License-Identifier: Apache-2.0
// Host test for the streaming sentence splitter. No model, no GPU.
#include <cassert>
#include <cstdio>
#include <string>

#include "umminfer/pipeline/sentence_split.hpp"

using umminfer::sentence_cut;

int main() {
  // A strong CJK terminator ends the sentence, including the terminator.
  assert(sentence_cut("你好。世界", 48) == std::string("你好。").size());
  // An ASCII period ends a sentence only at whitespace or end of text.
  assert(sentence_cut("Hello world. Next", 48) ==
         std::string("Hello world.").size());
  assert(sentence_cut("version 3.14 is out", 48) == 0);
  // No punctuation yet: no cut.
  assert(sentence_cut("still speaking", 48) == 0);
  // The length threshold cuts a long run without punctuation.
  const std::string long_text(60, 'a');
  assert(sentence_cut(long_text, 48) == 48);
  // A weak break in the second half wins over the raw window edge.
  std::string clause = std::string(30, 'a') + ", " + std::string(30, 'b');
  const size_t cut = sentence_cut(clause, 48);
  assert(cut == 32);  // through the space after the comma
  // Closing quotes stay with the sentence.
  assert(sentence_cut("他说：“好。”然后走了。", 48) ==
         std::string("他说：“好。”").size());
  // A cut never splits a UTF-8 sequence.
  const std::string multi = "中文中文中文中文中文中文中文中文中文中文中文中文"
                            "中文中文中文中文中文中文中文中文中文中文中文中文";
  const size_t m = sentence_cut(multi, 48);
  assert(m == 48 * 3);
  // A repeated terminator is absorbed.
  assert(sentence_cut("真的吗？！然后", 48) == std::string("真的吗？！").size());
  // Empty input never cuts.
  assert(sentence_cut("", 48) == 0);
  std::puts("sentence_split: PASS");
  return 0;
}
