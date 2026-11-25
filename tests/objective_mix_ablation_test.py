#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit and contract tests for Stage C Ablation #5 (Objective-Mix & Pareto Frontier)."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import umminfer_objective_mix_ablation as ablation  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "training" / "objective_mix_ablation.json"
SCRIPT_PATH = ROOT / "python" / "umminfer_objective_mix_ablation.py"


class ObjectiveMixConfigAndInvarianceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config, self.raw_bytes, self.invariance = ablation.load_and_validate_config(
            CONFIG_PATH
        )

    def test_default_config_verifies_single_variable_invariance(self) -> None:
        self.assertTrue(self.invariance["single_variable_verified"])
        self.assertEqual(self.invariance["variable_name"], "objective_mix")
        self.assertEqual(
            self.invariance["arm_ids"], list(ablation.EXPECTED_ARM_IDS)
        )
        fixed_hashes = set(self.invariance["per_arm_fixed_fields_sha256"].values())
        obj_hashes = set(self.invariance["per_arm_objective_mix_sha256"].values())
        self.assertEqual(len(fixed_hashes), 1)
        self.assertEqual(len(obj_hashes), 4)

    def test_tampering_any_fixed_field_fails_closed(self) -> None:
        mutations = [
            ("learning_rate", lambda c: c["arms"][1]["optimization"].__setitem__("learning_rate", 0.0002)),
            ("seed", lambda c: c["arms"][2]["optimization"].__setitem__("seed", "different-seed")),
            ("max_seq_len", lambda c: c["arms"][3]["optimization"].__setitem__("max_seq_len", 512)),
            ("steps", lambda c: c["arms"][1]["optimization"].__setitem__("steps", 3)),
            ("clip_eps", lambda c: c["arms"][2]["grpo"].__setitem__("clip_eps", 0.15)),
            ("kl_coef", lambda c: c["arms"][3]["grpo"].__setitem__("kl_coef", 0.08)),
            ("sft_weight", lambda c: (
                c["arms"][1].__setitem__("sft_weight", 0.5),
                c["arms"][1]["objective_mix"].__setitem__("sft_weight", 0.5),
            )),
            ("model_placeholder", lambda c: c["arms"][0]["model"].__setitem__("base_model_placeholder", "Other/Model")),
        ]
        for label, mutate in mutations:
            with self.subTest(field=label):
                tampered = copy.deepcopy(self.config)
                mutate(tampered)
                with self.assertRaises(ValueError):
                    ablation.verify_single_variable_invariance(tampered)

    def test_wrong_arm_set_or_duplicate_objective_mix_fails_closed(self) -> None:
        missing_arm = copy.deepcopy(self.config)
        missing_arm["arms"].pop()
        with self.assertRaises(ValueError):
            ablation.verify_single_variable_invariance(missing_arm)

        renamed_arm = copy.deepcopy(self.config)
        renamed_arm["arms"][2]["arm_id"] = "sft_plus_unknown"
        with self.assertRaises(ValueError):
            ablation.verify_single_variable_invariance(renamed_arm)

        duplicate_mix = copy.deepcopy(self.config)
        duplicate_mix["arms"][2]["objective_mix"] = copy.deepcopy(
            duplicate_mix["arms"][1]["objective_mix"]
        )
        with self.assertRaises(ValueError):
            ablation.verify_single_variable_invariance(duplicate_mix)


class CompositeRewardAndAdvantageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config, _, _ = ablation.load_and_validate_config(CONFIG_PATH)

    def test_all_four_arms_produce_distinct_advantage_vectors(self) -> None:
        arm_advantages: dict[str, list[list[float]]] = {}
        arm_rankings: dict[str, list[list[str]]] = {}
        for arm in self.config["arms"]:
            arm_id = arm["arm_id"]
            scored = ablation.score_rollouts_for_arm(self.config, arm)
            self.assertEqual(len(scored), 3)
            adv_matrix = []
            rankings = []
            for group in scored:
                cands = group["candidates"]
                self.assertEqual(len(cands), 3)
                advs = [round(c["advantage"], 6) for c in cands]
                adv_matrix.append(advs)
                if arm_id == "sft_only":
                    self.assertIsNone(group["group_reward_mean"])
                    self.assertEqual(advs, [0.0, 0.0, 0.0])
                    for c in cands:
                        self.assertIsNone(c["arm_reward_score"])
                else:
                    self.assertGreater(group["group_reward_std"], 0.0)
                    self.assertAlmostEqual(sum(c["advantage"] for c in cands), 0.0, places=9)
                ordered = sorted(
                    cands, key=lambda c: (-c["advantage"], c["candidate_id"])
                )
                rankings.append([c["candidate_profile"] for c in ordered])
            arm_advantages[arm_id] = adv_matrix
            arm_rankings[arm_id] = rankings

        # Every pair of arms must produce distinct advantage matrices.
        arm_ids = list(ablation.EXPECTED_ARM_IDS)
        for i, a_id in enumerate(arm_ids):
            for b_id in arm_ids[i + 1 :]:
                self.assertNotEqual(
                    arm_advantages[a_id],
                    arm_advantages[b_id],
                    f"{a_id} and {b_id} produced identical advantage matrices",
                )

        # Verify key behavioral distinctions between the 3 GRPO reward mixes:
        # Group 2 (barge-in): routing ranks slow (b) above text-only (c),
        # while consistency & full_latency_pareto rank clean fast text-only (c) above failed slow (b).
        routing_g2 = arm_advantages["sft_plus_routing"][1]
        consistency_g2 = arm_advantages["sft_plus_consistency"][1]
        pareto_g2 = arm_advantages["sft_plus_full_latency_pareto"][1]
        self.assertGreater(routing_g2[1], routing_g2[2])
        self.assertGreater(consistency_g2[2], consistency_g2[1])
        self.assertGreater(pareto_g2[2], pareto_g2[1])

        # Group 3 (orbit mechanics): routing ties optimal (a) and slow redundant (b);
        # consistency ties slow (b) and fast under-triggered (c);
        # full_latency_pareto strictly prefers fast (c) over slow redundant (b).
        routing_g3 = arm_advantages["sft_plus_routing"][2]
        consistency_g3 = arm_advantages["sft_plus_consistency"][2]
        pareto_g3 = arm_advantages["sft_plus_full_latency_pareto"][2]
        self.assertAlmostEqual(routing_g3[0], routing_g3[1], places=6)
        self.assertAlmostEqual(consistency_g3[1], consistency_g3[2], places=6)
        self.assertGreater(pareto_g3[2], pareto_g3[1])


class ParetoFrontierTests(unittest.TestCase):
    def test_exact_2d_pareto_dominance_and_dominated_by(self) -> None:
        points = [
            {"arm_id": "fast_low_q", "composite_quality_score": 0.60, "expected_p95_terminal_ms": 2000.0},
            {"arm_id": "balanced_knee", "composite_quality_score": 0.85, "expected_p95_terminal_ms": 4000.0},
            {"arm_id": "slow_high_q", "composite_quality_score": 0.92, "expected_p95_terminal_ms": 12000.0},
            {"arm_id": "dominated_mid", "composite_quality_score": 0.75, "expected_p95_terminal_ms": 6000.0},
            {"arm_id": "dominated_worst", "composite_quality_score": 0.50, "expected_p95_terminal_ms": 15000.0},
        ]
        res = ablation.compute_pareto_frontier(points)
        self.assertEqual(
            res["pareto_optimal_arms"],
            ["fast_low_q", "balanced_knee", "slow_high_q"],
        )
        self.assertEqual(
            res["dominated_arms"],
            ["dominated_mid", "dominated_worst"],
        )
        by_id = {entry["arm_id"]: entry for entry in res["arms"]}
        self.assertTrue(by_id["balanced_knee"]["is_pareto_optimal"])
        self.assertEqual(by_id["balanced_knee"]["dominated_by"], [])
        self.assertFalse(by_id["dominated_mid"]["is_pareto_optimal"])
        self.assertEqual(by_id["dominated_mid"]["dominated_by"], ["balanced_knee"])
        self.assertEqual(
            by_id["dominated_worst"]["dominated_by"],
            ["balanced_knee", "dominated_mid", "fast_low_q", "slow_high_q"],
        )

    def test_exact_ties_do_not_dominate_each_other(self) -> None:
        points = [
            {"arm_id": "tie_a", "composite_quality_score": 0.80, "expected_p95_terminal_ms": 5000.0},
            {"arm_id": "tie_b", "composite_quality_score": 0.80, "expected_p95_terminal_ms": 5000.0},
            {"arm_id": "worse", "composite_quality_score": 0.79, "expected_p95_terminal_ms": 5000.0},
        ]
        res = ablation.compute_pareto_frontier(points)
        self.assertEqual(res["pareto_optimal_arms"], ["tie_a", "tie_b"])
        self.assertEqual(res["dominated_arms"], ["worse"])
        by_id = {entry["arm_id"]: entry for entry in res["arms"]}
        self.assertEqual(by_id["worse"]["dominated_by"], ["tie_a", "tie_b"])

    def test_invalid_pareto_inputs_fail_closed(self) -> None:
        bad_inputs = [
            [],
            [{"arm_id": "a", "composite_quality_score": math.nan, "expected_p95_terminal_ms": 100.0}],
            [{"arm_id": "a", "composite_quality_score": 0.5, "expected_p95_terminal_ms": -1.0}],
            [{"arm_id": "a", "composite_quality_score": True, "expected_p95_terminal_ms": 100.0}],
            [
                {"arm_id": "dup", "composite_quality_score": 0.5, "expected_p95_terminal_ms": 100.0},
                {"arm_id": "dup", "composite_quality_score": 0.6, "expected_p95_terminal_ms": 90.0},
            ],
        ]
        for bad in bad_inputs:
            with self.assertRaises(ValueError):
                ablation.compute_pareto_frontier(bad)

    def test_weighted_quantile_responds_monotonically_to_tail_weight(self) -> None:
        vals = [1000.0, 3000.0, 25000.0, 28000.0]
        p95_heavy_tail = ablation.weighted_quantile_linear(vals, [0.25, 0.25, 0.25, 0.25], 0.95)
        p95_light_tail = ablation.weighted_quantile_linear(vals, [0.40, 0.40, 0.12, 0.08], 0.95)
        p95_tiny_tail = ablation.weighted_quantile_linear(vals, [0.48, 0.48, 0.02, 0.02], 0.95)
        self.assertGreater(p95_heavy_tail, p95_light_tail)
        self.assertGreater(p95_light_tail, p95_tiny_tail)


class OfflineContractCliAndHygieneTests(unittest.TestCase):
    def _run_cli(self, args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-I", str(SCRIPT_PATH), *args],
            capture_output=True,
            text=True,
        )

    def test_offline_contract_determinism_sha256sums_and_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out1 = Path(tmp) / "run1"
            out2 = Path(tmp) / "run2"
            out3 = Path(tmp) / "run3_alias"

            p1 = self._run_cli(["--config", str(CONFIG_PATH), "--offline-contract", "--out", str(out1)])
            p2 = self._run_cli(["--config", str(CONFIG_PATH), "--offline-contract", "--out", str(out2)])
            p3 = self._run_cli(["--config", str(CONFIG_PATH), "--replay-only", "--out", str(out3)])
            self.assertEqual(p1.returncode, 0, msg=p1.stderr)
            self.assertEqual(p2.returncode, 0, msg=p2.stderr)
            self.assertEqual(p3.returncode, 0, msg=p3.stderr)

            # Byte-identical determinism across runs and --replay-only alias
            verified_files = ablation.verify_sha256sums(out1)
            self.assertEqual(len(verified_files), 5)  # report.json + 4 arm reports
            for rel in verified_files + ["SHA256SUMS"]:
                b1 = (out1 / rel).read_bytes()
                b2 = (out2 / rel).read_bytes()
                b3 = (out3 / rel).read_bytes()
                self.assertEqual(b1, b2, f"{rel} differed across offline runs")
                self.assertEqual(b1, b3, f"{rel} differed with --replay-only alias")

            report = json.loads((out1 / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["evidence_class"], "CONTRACT_REPLAY")
            self.assertEqual(report["real_training_evidence"], "NOT_RUN")
            self.assertTrue(report["single_variable_verification"]["single_variable_verified"])
            self.assertTrue(report["pairwise_divergence"]["all_pairs_distinct_delta_logp"])
            self.assertIn(
                "sft_plus_full_latency_pareto",
                report["pareto_frontier"]["pareto_optimal_arms"],
            )
            # Latency-aware Pareto arm must achieve strictly lower P95 latency than pure routing
            arms = report["arm_summaries"]
            self.assertLess(
                arms["sft_plus_full_latency_pareto"]["expected_p95_terminal_ms"],
                arms["sft_plus_routing"]["expected_p95_terminal_ms"],
            )

            # Tampering an output file fails SHA256SUMS verification
            (out1 / "report.json").write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                ablation.verify_sha256sums(out1)

    def test_fail_closed_cli_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"

            # Mutually exclusive flags
            both = self._run_cli([
                "--config", str(CONFIG_PATH),
                "--offline-contract", "--live-gpu",
                "--out", str(out_dir),
            ])
            self.assertNotEqual(both.returncode, 0)

            # Missing --model-dir in --live-gpu mode
            no_model = self._run_cli([
                "--config", str(CONFIG_PATH),
                "--live-gpu",
                "--out", str(out_dir),
            ])
            self.assertNotEqual(no_model.returncode, 0)
            self.assertIn("--model-dir", no_model.stderr)

            # Tampered config fails closed via CLI
            bad_cfg_path = Path(tmp) / "tampered.json"
            bad_cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            bad_cfg["arms"][3]["grpo"]["kl_coef"] = 0.99
            bad_cfg_path.write_text(json.dumps(bad_cfg), encoding="utf-8")
            tampered = self._run_cli([
                "--config", str(bad_cfg_path),
                "--offline-contract",
                "--out", str(out_dir),
            ])
            self.assertNotEqual(tampered.returncode, 0)
            self.assertIn("single-variable violation", tampered.stderr)

    def test_pure_stdlib_imports_at_module_load(self) -> None:
        probe = (
            "import sys; "
            f"sys.path.insert(0, {str(PYTHON_DIR)!r}); "
            "import umminfer_objective_mix_ablation; "
            "forbidden = [m for m in ('torch', 'transformers', 'src', 'include', 'native') "
            "if m in sys.modules]; "
            "assert not forbidden, f'forbidden modules loaded: {forbidden}'; "
            "print('OK')"
        )
        proc = subprocess.run(
            [sys.executable, "-I", "-c", probe],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertEqual(proc.stdout.strip(), "OK")


if __name__ == "__main__":
    unittest.main()
