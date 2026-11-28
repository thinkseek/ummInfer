#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deterministic host-only tests for exact observed VLM history prompt replay."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


class HistoryPromptReplayTest(unittest.TestCase):
    binary = ""

    def run_replay(self, requests, expect=0):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.json"
            target = root / "output.json"
            source.write_text(json.dumps({
                "schema": "umminfer-vlm-history-prompt-replay-v1",
                "requests": requests,
            }), encoding="utf-8")
            result = subprocess.run(
                [self.binary, "--input", str(source), "--output", str(target)],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, expect, result.stdout + result.stderr)
            if expect:
                self.assertFalse(target.exists())
                return None
            return json.loads(target.read_text(encoding="utf-8"))

    def test_native_renderer_preserves_complete_history(self):
        messages = [
            {"role": "system", "content": "  route safely  "},
            {"role": "user", "content": "Set up the three-step proof."},
            {"role": "assistant", "content": "First, define p. Second, multiply."},
            {"role": "user", "content": "Expand the second step."},
        ]
        result = self.run_replay([{"id": "case/detail", "messages": messages}])
        self.assertEqual(result["requests"][0]["messages"], messages)
        self.assertEqual(result["requests"][0]["rendered_prompt"],
                         "<|im_start|>system\nroute safely<|im_end|>\n"
                         "<|im_start|>user\nSet up the three-step proof.<|im_end|>\n"
                         "<|im_start|>assistant\nFirst, define p. Second, multiply.<|im_end|>\n"
                         "<|im_start|>user\nExpand the second step.<|im_end|>\n"
                         "<|im_start|>assistant\n<think>\n\n</think>\n\n")

    def test_invalid_history_fails_without_output(self):
        for request in (
            {"id": "", "messages": [{"role": "user", "content": "x"}]},
            {"id": "empty", "messages": []},
            {"id": "non-string", "messages": [{"role": "user", "content": ["x"]}]},
        ):
            self.run_replay([request], expect=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    args, remaining = parser.parse_known_args()
    HistoryPromptReplayTest.binary = args.binary
    unittest.main(argv=[__file__, *remaining])
