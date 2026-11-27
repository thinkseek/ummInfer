#!/usr/bin/env python3
"""Offline tests for candidate upgrade/rollback identity verification."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/umminfer_ns_upgrade_check.py"


def values():
    before, after = "b" * 40, "a" * 40
    binaries = {"server": {"sha256": "1" * 64, "bytes": 100}}
    models = {"vlm": {"model_id": "fixture", "tree_sha256": "2" * 64, "file_count": 1}}
    config = {"preregistration_sha256": "3" * 64}
    candidate = {
        "schema": "umminfer-ns07-upgrade-candidate/v1",
        "owner_rollout_authorization": False, "resident_contact": "FORBIDDEN",
        "source": {"before_commit": before, "after_commit": after},
        "binaries": binaries, "models": models, "configuration": config,
        "required_health_invariants": ["zero crashes", "zero stale events", "zero cross-session events",
                                       "one terminal", "lossless PCM", "in_flight=0"],
        "abort_conditions": ["GPU not idle", "port collision", "unexpected exit", "hash mismatch",
                             "model mismatch", "missing evidence"],
        "rollback": {"source_commit": before, "execution": "NOT_RUN"},
    }
    arm = {
        "case_id": "NS-07", "engineering_status": "ISOLATED_PASS", "production_status": "NOT_RUN",
        "source_identity": {"commit": after}, "binary_identities": binaries,
        "model_identities": models, "config_identity": config,
        "metrics": {"elapsed_s": 1800, "client_count": 6, "unexpected_error_count": 0,
                    "process_crash_count": 0, "cross_session_event_count": 0,
                    "stale_event_count": 0, "duplicate_terminal_count": 0,
                    "pcm_mismatch_count": 0, "final_in_flight": 0,
                    "recovery_attempt_count": 4, "recovery_success_count": 4,
                    "device_growth_mib": 0, "rss_growth_mib": 1},
    }
    return candidate, arm


class UpgradeCheckTests(unittest.TestCase):
    def run_fixture(self, mutate=None):
        directory = tempfile.TemporaryDirectory()
        root = Path(directory.name)
        candidate, arm = values()
        if mutate:
            mutate(candidate, arm)
        candidate_path, arm_path, out = root / "candidate.json", root / "after.json", root / "report.json"
        candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
        arm_path.write_text(json.dumps(arm), encoding="utf-8")
        result = subprocess.run([sys.executable, "-I", str(TOOL), "--candidate", str(candidate_path),
                                 "--after-arm", str(arm_path), "--out", str(out)],
                                capture_output=True, text=True, timeout=30)
        return directory, result, out

    def test_static_preflight_passes_without_network_or_rollout(self):
        directory, result, out = self.run_fixture()
        self.addCleanup(directory.cleanup)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(out.read_text())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["resident_upgrade"], "NOT_RUN")
        self.assertEqual(report["resident_rollback"], "NOT_RUN")
        self.assertEqual(report["network_or_process_actions"], 0)

    def test_health_regression_fails_closed(self):
        def mutate(_candidate, arm):
            arm["metrics"]["duplicate_terminal_count"] = 1
        directory, result, out = self.run_fixture(mutate)
        self.addCleanup(directory.cleanup)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate_terminal_count", json.loads(out.read_text())["checks_failed"])

    def test_identity_drift_fails_closed(self):
        def mutate(_candidate, arm):
            arm["model_identities"] = json.loads(json.dumps(arm["model_identities"]))
            arm["model_identities"]["vlm"]["tree_sha256"] = "f" * 64
        directory, result, out = self.run_fixture(mutate)
        self.addCleanup(directory.cleanup)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("model identity match", json.loads(out.read_text())["checks_failed"])


if __name__ == "__main__":
    unittest.main()
