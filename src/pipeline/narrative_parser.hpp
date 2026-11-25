// SPDX-License-Identifier: Apache-2.0
// Incremental parser for the interleaved-narrative directive grammar.
//
// The VLM writes prose plus standalone visual markers:
//   [[image: <prompt>]]   [[video: <prompt>]]   [[visual: <prompt>]]
// The marker may arrive split across any number of stream chunks, including
// inside a UTF-8 sequence; the parser holds only the undecidable suffix and
// emits prose as soon as it is safe. Directives are stripped from the prose
// stream so downstream text, history and TTS never see them.
//
// False positives are handled without stalling prose: a "[[..." prefix that
// cannot become one of the three tags is emitted as prose one byte at a time,
// which also keeps overlapping "[[[" cases intact. A directive candidate that
// never closes is bounded by kMaxDirectiveBytes and then released as prose.
#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace umminfer {

struct NarrativeDirective {
  // Lowercased tag as written: "image", "video" or "visual". The caller
  // resolves "visual" against its visual mode; the parser stays pure.
  std::string kind;
  // Trimmed directive body with an optional reference suffix removed. Never
  // empty for an ordinary directive; an empty body without a reference is
  // dropped. The explicit suffix is `| source=slot-N` on a video directive.
  std::string prompt;
  // Same-request image slot named by the explicit reference suffix. Empty when
  // the directive has no valid reference.
  std::string source_slot;
  // Non-empty when a source-like suffix is malformed or duplicated. The engine
  // reports a failed slot instead of silently rendering unconditioned video.
  std::string reference_error;
  // Byte offset of the marker in the clean prose stream emitted so far.
  size_t prose_offset = 0;
};

class NarrativeStreamParser {
 public:
  // A candidate longer than this without a closing "]]" is treated as prose.
  static constexpr size_t kMaxDirectiveBytes = 8192;

  // Consumes one chunk and returns the prose that is safe to emit now.
  // Completed directives are appended to `out` in stream order.
  std::string feed(std::string_view chunk,
                   std::vector<NarrativeDirective>& out);

  // End of input: any held "[" prefix or unclosed candidate becomes prose.
  std::string flush(std::vector<NarrativeDirective>& out);

  // Total clean prose bytes emitted so far.
  size_t prose_bytes() const { return prose_emitted_; }

 private:
  std::string scan(bool eof, std::vector<NarrativeDirective>& out);

  std::string buf_;
  size_t prose_emitted_ = 0;
};

}  // namespace umminfer
