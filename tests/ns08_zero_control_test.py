#!/usr/bin/env python3
"""Deterministic tests for the NS-08 zero-control arm adapter."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ns08_zero_control", ROOT / "tests/ns08_zero_control.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
PARENT = "b4dca5b0e3f4fb18ecd857847cf2205c660cb8a7"


class ZeroControlTests(unittest.TestCase):
    def source_with_page(self, directory, page):
        root = Path(directory)
        (root / "web").mkdir()
        (root / "web" / "index.html").write_text(page, encoding="utf-8")
        return root

    def test_current_restored_baseline_is_measured_not_assumed(self):
        counts, visible = MODULE.static_controls(ROOT)
        self.assertEqual(visible, ["realtimeEchoSuppression", "voiceToggle"])
        self.assertEqual(sum(counts.values()), 2)

    def test_collapsed_candidate_fixture_has_zero_default_controls(self):
        body = "".join(f'<input id="{control}">' for control in MODULE.ALL_CONTROLS)
        page = '<details id="advanced"><summary>Advanced</summary>' + body + "</details>"
        with tempfile.TemporaryDirectory() as directory:
            root = self.source_with_page(directory, page)
            counts, visible = MODULE.static_controls(root)
        self.assertEqual(visible, [])
        self.assertEqual(sum(counts.values()), 0)

    def test_parent_baseline_retains_two_visible_voice_developer_controls(self):
        page = subprocess.check_output(["git", "show", f"{PARENT}:web/index.html"],
                                       cwd=ROOT, text=True)
        with tempfile.TemporaryDirectory() as directory:
            root = self.source_with_page(directory, page)
            counts, visible = MODULE.static_controls(root)
        self.assertEqual(visible, ["realtimeEchoSuppression", "voiceToggle"])
        self.assertEqual(counts["visible_voice_routing_control_count"], 2)
        self.assertEqual(sum(counts.values()), 2)

    def test_void_elements_do_not_hide_following_controls(self):
        page = """<details id="advanced"><summary>Advanced</summary>
                  <input id="seed"><select id="visualMode"></select></details>
                  <input id="importSessionInput" hidden><button id="voiceToggle">Voice</button>"""
        parser = MODULE.DefaultVisibility(False)
        parser.feed(page)
        self.assertFalse(parser.advanced_open)
        self.assertEqual(parser.visible, {"voiceToggle"})

    def test_control_groups_cover_every_pinned_developer_control(self):
        self.assertEqual(len(MODULE.ALL_CONTROLS), len(set(MODULE.ALL_CONTROLS)))
        self.assertIn("seed", MODULE.ALL_CONTROLS)
        self.assertIn("videoFrameBudget", MODULE.ALL_CONTROLS)
        self.assertIn("voiceToggle", MODULE.ALL_CONTROLS)
        self.assertNotIn("realtimeToggle", MODULE.ALL_CONTROLS)

    def test_external_manifest_is_complete_and_self_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw.log").write_text("raw\n", encoding="utf-8")
            (root / "nested").mkdir()
            (root / "nested" / "event.json").write_text("{}\n", encoding="utf-8")
            result = MODULE.manifest(root)
            rows = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
            self.assertEqual(result["file_count"], 2)
            self.assertEqual(len(rows), 2)
            self.assertFalse(any(row.endswith("  SHA256SUMS") for row in rows))
            for row in rows:
                digest, name = row.split("  ", 1)
                self.assertEqual(MODULE.sha_file(root / name), digest)
    def test_fresh_arm_verifier_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "arm"
            root.mkdir()
            browser = {"status": "PASS", "visible_controls": [],
                       "visible_counts": {name: 0 for name in MODULE.CONTROL_GROUPS},
                       "checks": [{"name": "zero", "ok": True}], "checks_failed": 0}
            (root / "browser-result.json").write_text(json.dumps(browser) + "\n")
            metrics = {name: 0 for name in MODULE.CONTROL_GROUPS}
            metrics.update({"contract_checks_failed": 0, "browser_checks_failed": 0,
                            "export_import_exact": 1, "reconnect_exact": 1,
                            "source_bound_revision": 1, "cancellation_retry_exact": 1})
            report = {"schema": "umminfer-ns08-zero-control-arm/v1", "case_id": "NS-08",
                      "arm": "after", "engineering_status": "CONTRACT_PASS",
                      "production_status": "NOT_RUN", "full_gate_status": "NOT_RUN",
                      "metrics": metrics, "source_identity": {"commit": "a" * 40},
                      "coverage": {"macos_real": "NOT_RUN", "human_walkthrough": "NOT_RUN"},
                      "raw_evidence": {"browser_result_sha256":
                                       MODULE.sha_file(root / "browser-result.json")}}
            (root / "NS-08.json").write_text(json.dumps(report) + "\n")
            MODULE.manifest(root)
            out = Path(directory) / "verify.json"
            command = [sys.executable, "-I", str(ROOT / "tests/ns08_zero_control_verify.py"),
                       "--arm-dir", str(root), "--out", str(out)]
            verified = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(verified.returncode, 0, verified.stderr)
            self.assertEqual(json.loads(out.read_text())["status"], "PASS")
            with (root / "browser-result.json").open("a") as target:
                target.write(" \n")
            rejected = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("digest", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
