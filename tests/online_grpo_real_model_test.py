#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only arithmetic/raw-transport checks for the real-model online-GRPO gate."""

import ast
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import online_grpo_real_model_verify as verify  # noqa: E402


OBJECTIVE = {"clip_eps": 0.2, "kl_coef": 0.04, "log_ratio_clamp": 10.0}


class RealOnlineGrpoValidationTests(unittest.TestCase):
    def test_independent_arithmetic_controls(self):
        result = verify.arithmetic_controls(OBJECTIVE)
        self.assertEqual(set(result), {
            "positive_advantage_clipping", "negative_advantage_min_selection",
            "schulman_exp_minus_one", "log_ratio_clamp"})
        self.assertEqual(result["positive_advantage_clipping"]["clipped_ratio"], "1.2")
        self.assertEqual(result["negative_advantage_min_selection"]["clip_active"], False)

    def test_zero_kl_and_zero_variance_are_finite(self):
        metrics = verify.decimal_metrics(
            [-0.25, -0.5], [-0.25, -0.5], [-0.25, -0.5], [1, 1], 0.0, OBJECTIVE)
        self.assertEqual(metrics["ratio"], verify.D1)
        self.assertEqual(metrics["schulman_kl"], verify.D0)
        self.assertEqual(metrics["objective_loss"], verify.D0)

    def test_objective_tamper_is_detected(self):
        behavior = [-0.25, -0.5, -1.0]
        current = [-0.2, -0.45, -0.9]
        reference = [-0.3, -0.55, -1.1]
        expected = verify.decimal_metrics(behavior, current, reference, [1, 0, 1], 0.5,
                                         OBJECTIVE)
        actual = {key: float(value) if isinstance(value, verify.Decimal) else value
                  for key, value in expected.items()}
        self.assertLessEqual(verify.compare_metrics(actual, expected, "valid"),
                             verify.OBJECTIVE_TOLERANCE)
        actual["objective_loss"] += 0.01
        with self.assertRaises(verify.VerificationError):
            verify.compare_metrics(actual, expected, "tampered")

    def test_raw_sse_behavior_binding_and_tamper(self):
        with tempfile.TemporaryDirectory(prefix="umm-online-grpo-raw-test-") as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            evidence = {
                "format": "umminfer-behavior-logprobs-v1", "tokenizer_sha256": "ab" * 32,
                "token_ids": [11, 12], "logprobs": [-0.25, -0.5], "mask": [0, 1],
            }
            chunk = {"choices": [{"finish_reason": "stop"}], "behavior": evidence}
            raw = (b"data: " + verify.canonical(chunk) + b"\n\n" + b"data: [DONE]\n\n")
            raw_sha = verify.sha256_bytes(raw)
            (artifacts / (raw_sha + ".sse")).write_bytes(raw)
            summary = {**copy.deepcopy(evidence), "status": "COVERED",
                       "evidence_sha256": verify.sha256_bytes(verify.canonical(evidence))}
            record = {
                "terminal_state": "completed",
                "sse_replay": {"raw_ref": raw_sha + ".sse", "raw_sha256": raw_sha,
                               "raw_bytes": len(raw)},
                "rollout_provenance": {"behavior": summary,
                                       "response": {"behavior_evidence": "exposed"}},
            }
            parsed, digest = verify.raw_behavior(record, root)
            self.assertEqual(parsed, evidence)
            self.assertEqual(digest, raw_sha)
            record["rollout_provenance"]["behavior"]["logprobs"][0] -= 0.01
            with self.assertRaises(verify.VerificationError):
                verify.raw_behavior(record, root)

    def test_verifier_has_no_producer_or_ml_imports(self):
        tree = ast.parse((ROOT / "tests/online_grpo_real_model_verify.py").read_text(
            encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        forbidden = {"umminfer_online_rollout", "umminfer_online_grpo_preflight",
                     "umminfer_behavior_token_alignment", "umminfer_online_evidence_audit",
                     "torch", "transformers", "numpy", "mlx"}
        self.assertTrue(forbidden.isdisjoint(imported))

    def test_live_gate_default_is_not_run(self):
        with tempfile.TemporaryDirectory(prefix="umm-online-grpo-not-run-") as tmp:
            out = Path(tmp) / "new-evidence"
            command = [
                sys.executable, "-I", str(ROOT / "tests/online_grpo_real_model_live.py"),
                "--model", "qwen3-native", "--model-dir", str(Path(tmp) / "model"),
                "--model-revision", "UNKNOWN", "--replay-binary", str(Path(tmp) / "replay"),
                "--source-rollout-bundle", str(Path(tmp) / "source"),
                "--source-live-summary", str(Path(tmp) / "summary.json"),
                "--source-commit", "0" * 40, "--out-dir", str(out),
            ]
            env = os.environ.copy()
            env.pop("UMMINFER_LIVE_OPT_IN", None)
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                    text=True, timeout=30, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("NOT_RUN", result.stdout)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
