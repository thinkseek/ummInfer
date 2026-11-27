#!/usr/bin/env python3
"""Deterministic contract tests for unified NS scorecards and fresh verification."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ns_scorecard", ROOT / "tools/umminfer_ns_scorecard.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CASES = [f"NS-{index:02d}" for index in range(1, 9)]

NS07_METRICS = {
    "elapsed_s": 1800.0, "client_count": 6, "request_count": 120,
    "completed_count": 100, "cancelled_count": 8, "expected_fault_count": 12,
    "unexpected_error_count": 0, "process_crash_count": 0,
    "cross_session_event_count": 0, "stale_event_count": 0,
    "duplicate_terminal_count": 0, "pcm_mismatch_count": 0,
    "recovery_attempt_count": 12, "recovery_success_count": 12,
    "final_in_flight": 0, "device_growth_mib": 0, "rss_growth_mib": 12,
    "latency_p95_ms": 1000.0,
}
NS08_BEFORE = {
    "visible_modality_control_count": 0, "visible_routing_control_count": 0,
    "visible_sampling_control_count": 0, "visible_frame_control_count": 0,
    "visible_voice_routing_control_count": 2, "contract_checks_failed": 0,
    "browser_checks_failed": 1, "export_import_exact": 1, "reconnect_exact": 1,
    "source_bound_revision": 1, "cancellation_retry_exact": 1,
}
NS08_AFTER = dict(NS08_BEFORE, visible_voice_routing_control_count=0,
                  browser_checks_failed=0)


def arm(case_id, status, metrics):
    return {
        "schema": "fixture-only", "case_id": case_id, "arm": "fixture",
        "engineering_status": status, "production_status": "NOT_RUN",
        "full_gate_status": "NOT_RUN", "metrics": metrics, "hard_gates": {},
        "failures": [], "limitations": ["synthetic harness test; not real model evidence"],
        "source_identity": {"commit": "a" * 40, "hardware": {"gpu": 0},
                            "harness_sha256": "5" * 64, "preregistration_sha256": "6" * 64,
                            "adapter_sha256": "7" * 64},
        "binary_identities": {"browser": "fixture-v1"},
        "model_identities": {"fixture": "same-model"}, "config_identity": {"fixture": "same-config"},
        "raw_evidence": {"events_sha256": "b" * 64},
        "coverage": {"macos_real": "NOT_RUN", "human_walkthrough": "NOT_RUN"},
        "deployment_scope": "isolated-fixture",
    }


def write_arm(root, case_id, status, metrics):
    root.mkdir(parents=True, exist_ok=True)
    report = arm(case_id, status, metrics)
    report["source_identity"]["commit"] = ("b" if root.name == "before" else "a") * 40
    (root / f"{case_id}.json").write_text(json.dumps(report) + "\n", encoding="utf-8")


class ScorecardTests(unittest.TestCase):
    def test_registry_and_missing_arms_cover_all_eight_cases(self):
        registry = MODULE.load(MODULE.REGISTRY)
        self.assertEqual(registry["case_order"], CASES)
        self.assertEqual([case["case_id"] for case in registry["cases"]], CASES)
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "external"
            result = subprocess.run([sys.executable, "-I", "tools/umminfer_ns_scorecard.py",
                                     "--evidence-dir", str(evidence), "--skip-contracts"],
                                    cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            scorecard = json.loads((evidence / "ns-scorecard.json").read_text())
            self.assertEqual(scorecard["case_order"], CASES)
            self.assertTrue(all(case["north_star_hard_gate"] == "NOT_RUN"
                                for case in scorecard["cases"]))
            self.assertEqual(scorecard["candidate_promotion"], "NOT_RUN")

    def test_full_gate_requires_independent_real_coverage(self):
        case = {"case_id": "NS-08"}
        projected = MODULE.clean_projection(arm("NS-08", "PASS", NS08_AFTER))
        projected["status"] = "PASS"
        projected["declared_full_gate_status"] = "PASS"
        projected["coverage"]["independent_verification"] = "PASS"
        self.assertEqual(MODULE.full_gate(case, projected), "NOT_COVERED")
        projected["coverage"].update({"web_real": "PASS", "macos_real": "PASS",
                                      "human_walkthrough": "PASS", "lossless_round_trip": "PASS"})
        self.assertEqual(MODULE.full_gate(case, projected), "PASS")
        projected["coverage"]["macos_real"] = "NOT_RUN"
        self.assertEqual(MODULE.full_gate(case, projected), "NOT_COVERED")

    def test_paired_comparisons_reject_regressions(self):
        before = arm("NS-07", "ISOLATED_PASS", dict(NS07_METRICS))
        before["status"] = "ISOLATED_PASS"
        before["source_identity"]["commit"] = "b" * 40
        after = arm("NS-07", "ISOLATED_PASS", dict(NS07_METRICS))
        after["status"] = "ISOLATED_PASS"
        self.assertEqual(MODULE.compare_ns07(before, after)["status"], "PAIRED_NON_REGRESSION")
        after["source_identity"]["harness_sha256"] = "f" * 64
        self.assertFalse(MODULE.compare_ns07(before, after)["no_regression"])
        after["source_identity"]["harness_sha256"] = "5" * 64
        after["metrics"]["latency_p95_ms"] = 1050.01
        self.assertEqual(MODULE.compare_ns07(before, after)["status"], "NOT_PROMOTED")
        after["metrics"]["latency_p95_ms"] = 1000.0
        after["metrics"]["recovery_success_count"] = 11
        self.assertFalse(MODULE.compare_ns07(before, after)["no_regression"])

        old = arm("NS-08", "FAIL", dict(NS08_BEFORE))
        old["source_identity"]["commit"] = "b" * 40
        new = arm("NS-08", "CONTRACT_PASS", dict(NS08_AFTER))
        self.assertEqual(MODULE.compare_ns08(old, new)["status"],
                         "PAIRED_ENGINEERING_IMPROVEMENT")
        new["source_identity"]["adapter_sha256"] = "f" * 64
        self.assertFalse(MODULE.compare_ns08(old, new)["no_regression"])
        new["source_identity"]["adapter_sha256"] = "7" * 64
        new["metrics"]["export_import_exact"] = 0
        self.assertEqual(MODULE.compare_ns08(old, new)["status"], "NOT_PROMOTED")

    def test_fresh_process_verifier_and_manifest_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            before, after = base / "before", base / "after"
            write_arm(before, "NS-07", "ISOLATED_PASS", NS07_METRICS)
            write_arm(after, "NS-07", "ISOLATED_PASS", NS07_METRICS)
            write_arm(before, "NS-08", "FAIL", NS08_BEFORE)
            write_arm(after, "NS-08", "CONTRACT_PASS", NS08_AFTER)
            evidence = base / "scorecard-evidence"
            produced = subprocess.run([sys.executable, "-I", "tools/umminfer_ns_scorecard.py",
                "--evidence-dir", str(evidence), "--before-arm", str(before),
                "--after-arm", str(after), "--skip-contracts"], cwd=ROOT,
                capture_output=True, text=True, timeout=60)
            self.assertEqual(produced.returncode, 0, produced.stderr)
            scorecard = json.loads((evidence / "ns-scorecard.json").read_text())
            self.assertEqual(scorecard["candidate_promotion"],
                             "PROMOTED_ISOLATED_ENGINEERING_ONLY")
            verification = base / "independent.json"
            command = [sys.executable, "-I", "tests/ns_scorecard_verify.py",
                       "--scorecard", str(evidence / "ns-scorecard.json"),
                       "--manifest", str(evidence / "SHA256SUMS"),
                       "--root", str(evidence), "--out", str(verification)]
            verified = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(verified.returncode, 0, verified.stderr)
            report = json.loads(verification.read_text())
            self.assertEqual(report["status"], "PASS")
            self.assertFalse(report["producer_imported"])
            with (evidence / "ns-scorecard.json").open("a", encoding="utf-8") as target:
                target.write(" \n")
            tampered = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(tampered.returncode, 0)
            self.assertIn("manifest digest mismatch", tampered.stderr)


if __name__ == "__main__":
    unittest.main()
