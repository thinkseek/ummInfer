#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline contracts for the NS-01 opening evidence tools."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
import ns01_streaming_opening_scorecard as scorecard  # noqa: E402
from ns01_streaming_opening_live import request_body  # noqa: E402


class SpeechProjectionTests(unittest.TestCase):
    def test_markdown_and_literal_operators(self):
        self.assertEqual(
            scorecard.spoken_text_projection(
                "## Heading\n*opening* **bold** 3*4 snake_case `code`"),
            "Headingopeningbold3*4snake_casecode")
        self.assertEqual(scorecard.spoken_text_projection("2 * 3 * 4"), "2*3*4")
        self.assertEqual(scorecard.spoken_text_projection("a_b and _word_"), "a_bandword")

    def test_real_m87_presentation_boundary(self):
        raw = "Earlier *emphasis* and the black hole M87*, we see a presentation span"
        speech = "Earlier emphasis and the black hole M87, we see a presentation span"
        self.assertEqual(scorecard.spoken_text_projection(raw),
                         scorecard.spoken_text_projection(speech))


class FrozenIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prereg = json.loads((ROOT / "benchmarks" / "ns01-streaming-opening" /
                                 "preregistration.json").read_text())
        cls.candidate = json.loads((ROOT / "benchmarks" / "ns01-streaming-opening" /
                                    "candidate.json").read_text())

    def test_exact_requests_and_no_manual_directives(self):
        for case in ("chinese", "english"):
            body = request_body(self.prereg, case)
            self.assertEqual(body["max_tokens"], 768)
            self.assertEqual(body["temperature"], 0)
            self.assertEqual(body["narrative"]["visual_mode"], "auto")
            self.assertEqual(body["narrative"]["continuity_mode"], "i2v_chain")
            self.assertNotIn("[[", body["messages"][0]["content"])
            self.assertNotIn("teacache_thresh", body["video"])

    def test_candidate_is_bounded_and_explicitly_not_a_runtime_requirement(self):
        change = self.candidate["runtime_change"]
        self.assertEqual(change["kind"], "default_auto_director_prompt_only")
        self.assertFalse(change["parser_changes"])
        self.assertFalse(change["orchestration_changes"])
        self.assertFalse(change["native_arithmetic_changes"])
        self.assertTrue(self.candidate["promotion_rule"]
                        ["hard_ttfa_first_image_and_i2v_misses_remain_not_met_even_when_promoted"])


if __name__ == "__main__":
    unittest.main()
