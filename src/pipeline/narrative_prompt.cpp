// SPDX-License-Identifier: Apache-2.0
#include "pipeline/unified_schema.hpp"

namespace umminfer::unified_detail {

std::string narrative_director_prompt(const std::string& visual_mode,
                                      int max_slots,
                                      const std::string& continuity_mode) {
  std::string out;
  if (visual_mode == "auto") {
    out =
        "You are a multimodal assistant that chooses between pure text and "
        "interleaved visuals. Write coherent, well-structured prose in the "
        "user's language. Only emit visual markers when a visual genuinely "
        "clarifies a scene, diagram, or motion; for pure math, code, logic "
        "proofs, short factual questions, or text-only requests, emit zero "
        "markers and respond in clean prose only. When visuals genuinely help, "
        "insert up to " +
        std::to_string(max_slots) +
        " standalone markers on their own lines (use [[image: ...]] for still "
        "scenes or diagrams, and [[video: ...]] for motion or dynamics); a "
        "generated visual replaces each marker, so never describe or mention "
        "the marker itself.\n"
        "[[image: <vivid concrete visual scene description in English>]]\n"
        "[[video: <vivid concrete motion and camera description in English>]]";
  } else {
    out =
        "You are a film director narrating with interleaved visuals. Write "
        "coherent, well-structured prose in the user's language. At up to " +
        std::to_string(max_slots) +
        " natural narrative beats, insert one standalone visual marker on its "
        "own line; a generated visual replaces it, so never describe or mention "
        "the marker itself.";
    if (visual_mode == "image") {
      out +=
          "\n[[image: <vivid concrete visual scene description in English>]]";
    } else {
      out +=
          "\n[[video: <vivid concrete motion and camera description in English>]]";
    }
  }
  if (continuity_mode == "i2v_chain" && visual_mode != "image") {
    out +=
        "\nTo animate a specific earlier generated image, use "
        "[[video: <motion and camera description> | source=slot-N]] where slot "
        "numbers are 0-based (the first visual tag is slot-0, the second is "
        "slot-1, etc.). Use source only for I2V continuity; never invent, "
        "duplicate, or forward-reference a slot. Without this explicit suffix "
        "the engine's ordinary i2v_chain anchor selection applies.";
  }
  out += "\nDo not use any other bracket syntax.";
  return out;
}

}  // namespace umminfer::unified_detail
