#!/usr/bin/env python3
"""Deterministic Task #47 fixture/oracle/intervention regressions; no GPU/model."""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import umminfer_long_video_fixture as fixture  # noqa: E402
import umminfer_long_video_live as long_live  # noqa: E402
import umminfer_temporal_quality as quality  # noqa: E402


class OracleAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = fixture.validate_bundle(ROOT / "benchmarks/video-temporal-long", deep=False)
        cls.by_id = {case["id"]: case for case in cls.manifest["cases"]}

    def test_uniform_expected_answer_anchors_are_not_visible(self):
        intended = {"separated-motion": {64, 176, 320}, "scene-static": {224},
                    "near-duplicate": {288}, "bframe-anchors": {0, 448}}
        for cid, anchors in intended.items():
            with self.subTest(case=cid):
                uniform = set(self.by_id[cid]["expected_uniform_8"]["source_frame_indices"])
                temporal = set(self.by_id[cid]["expected_temporal"]["8"]["source_frame_indices"])
                self.assertEqual(max(uniform), 59)
                self.assertFalse(uniform & anchors)
                self.assertTrue(anchors <= temporal)

    def test_colored_binding_tags_and_answer_leakage_are_explicit(self):
        for case in self.manifest["cases"]:
            with self.subTest(case=case["id"]):
                self.assertIn(0, case["expected_temporal"]["8"]["source_frame_indices"])
                frame = fixture.source_frame(case["id"], 0)
                offset = (fixture.WIDTH + 1) * 3
                self.assertEqual(list(frame[offset:offset + 3]), [0, 255, 32])
        leading = self.by_id["bframe-anchors"]
        self.assertTrue(all(term in leading["question"] for term in leading["expect_all"]))

    def test_original_answer_scoring_is_not_relaxed(self):
        self.assertFalse(long_live.answer_hit("green", self.by_id["separated-motion"]))
        self.assertTrue(long_live.answer_hit("red", self.by_id["separated-motion"]))
        self.assertFalse(long_live.answer_hit("绿色", self.by_id["scene-static"]))
        self.assertTrue(long_live.answer_hit("蓝色", self.by_id["scene-static"]))
        self.assertTrue(long_live.answer_hit("マゼンタ", self.by_id["near-duplicate"]))
        self.assertFalse(long_live.answer_hit("blanco", self.by_id["bframe-anchors"]))
        self.assertTrue(long_live.answer_hit("blanco y negro", self.by_id["bframe-anchors"]))

    def test_uniform_nominal_markers_are_not_selected_source_pts(self):
        nominal = quality.uniform_nominal_timestamps(224.5)
        self.assertEqual(quality.pair_first(nominal), [0.0, 56.125, 112.25, 168.375])
        self.assertEqual(quality.native_showinfo_timestamp(140.3125), 140.312)
        self.assertEqual(quality.native_showinfo_timestamp(196.4375), 196.438)
        for case in self.manifest["cases"]:
            exact = case["expected_uniform_8"]["source_timestamps_seconds"]
            self.assertEqual(exact, [1.5, 5.5, 9.5, 13.5, 17.5, 21.5, 25.5, 29.5])
            self.assertNotEqual(quality.pair_first(exact), quality.pair_first(nominal))

    def test_original_bframe_packet_and_presentation_order_are_distinct(self):
        for case in self.manifest["cases"]:
            self.assertTrue(case["decode_order"]["decode_order_pts_nonmonotonic"])
            self.assertTrue(case["decode_order"]["decode_order_dts_monotonic"])
            self.assertEqual(case["expected_temporal"]["8"]["source_timestamps_seconds"],
                             sorted(case["expected_temporal"]["8"]["source_timestamps_seconds"]))


class PositionOracleTests(unittest.TestCase):
    def ids_for_markers(self, markers):
        ids = [10, 11]
        for marker in markers:
            ids.extend([marker, 248052])
            ids.extend([248057] * 35)
            ids.append(248053)
        ids.extend([20, 21, 22])
        return ids

    def test_exact_full_mrope_coordinates_and_timestamp_token_independence(self):
        ids = self.ids_for_markers([100, 200, 300, 400])
        positions = quality.independent_positions(ids, 4, 10, 14)
        self.assertEqual(len(positions["t"]), len(ids))
        self.assertEqual(positions["next"], 45)
        self.assertEqual(quality.position_digest(positions), "e9e144edd11feb0f")
        changed = self.ids_for_markers([101, 201, 301, 401])
        self.assertEqual(quality.independent_positions(changed, 4, 10, 14), positions)
        self.assertEqual(quality.position_digest(positions),
                         quality.position_digest(quality.independent_positions(changed, 4, 10, 14)))
        for run in range(4):
            start = 4 + run * 38
            t = positions["t"][start:start + 35]
            h = positions["h"][start:start + 35]
            w = positions["w"][start:start + 35]
            self.assertEqual(len(set(t)), 1)
            self.assertEqual(h, [t[0] + row for row in range(5) for _ in range(7)])
            self.assertEqual(w, [t[0] + index % 7 for index in range(35)])

    def test_mrope_rejects_grid_or_run_mismatch(self):
        with self.assertRaises(ValueError):
            quality.independent_positions(self.ids_for_markers([1, 2, 3]), 4, 10, 14)
        with self.assertRaises(ValueError):
            quality.independent_positions(self.ids_for_markers([1, 2, 3, 4]), 4, 9, 14)


class ControlCheckTests(unittest.TestCase):
    def records(self, *, valid):
        rows = []
        for arm in quality.ARMS:
            native = {field: 0 for field in quality.short_native_fields()}
            native.update({"media_hash": "same-pixels", "selected_frame_content_fnv1a64": []})
            # Explicit positions make all four factor-isolation comparisons valid.
            native["selected_frame_timestamps"] = (
                [0.0, 1.0] if arm == "temporal_source_pts" else
                [0.0, 2.0] if arm == "temporal_uniform_positions" else
                [0.0, 3.0] if arm == "temporal_ordinal_pts" else
                [0.0, 4.0] if arm == "uniform_nominal_positions" else
                [0.0, 5.0] if arm == "uniform_source_relative_pts" else [0.0, 1.0])
            prompt = "translated" if arm.endswith("alt_language") else "original"
            rows.append({"phase": "warmup", "repetition": 1, "clip": "separated-motion",
                         "arm": arm, "native_evidence_valid": valid, "native": native,
                         "raw_answer": "same" if valid else None,
                         "request": {"messages": [{"content": [{"text": prompt}]}]}})
        return rows

    def test_grouped_controls_and_repetition_regression(self):
        checks = quality.control_checks(self.records(valid=True))
        self.assertEqual(len(checks), 10)
        self.assertTrue(all(check["passed"] for check in checks))

    def test_failed_records_fail_closed_without_grouping_exception(self):
        checks = quality.control_checks(self.records(valid=False))
        self.assertEqual(len(checks), 10)
        self.assertFalse(any(check["passed"] for check in checks))

    def test_nested_sealed_inputs_and_checksums_are_covered(self):
        with tempfile.TemporaryDirectory(prefix="umm-task47-nested-seal-") as tmp:
            root = Path(tmp)
            quality.write_json(root / "report.json", {
                "schema_version": 1, "records": [], "aggregates": [],
                "answer_outcomes": {}, "parity": [], "gate_pass": False,
                "gate_reasons": ["synthetic test only"]})
            quality.write_json(root / "input/r4-sealed/artifact-manifest.json", {"r4": True})
            (root / "input/source").mkdir(parents=True)
            (root / "input/source/SHA256SUMS").write_text("nested fixture checksums\n")
            quality.seal_evidence(root)
            quality.validate_evidence_files(root)
            sums = (root / "SHA256SUMS").read_text()
            artifact = (root / "artifact-manifest.json").read_text()
            self.assertIn("input/r4-sealed/artifact-manifest.json", sums)
            self.assertIn("input/source/SHA256SUMS", sums)
            self.assertIn("input/r4-sealed/artifact-manifest.json", artifact)
            self.assertIn("input/source/SHA256SUMS", artifact)


class LosslessInterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="umm-task47-offline-test-")
        cls.root = Path(cls.temp.name) / "plan"
        cls.plan = quality.prepare_plan(ROOT / "benchmarks/video-temporal-long", cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_fresh_full_pixel_pts_and_seal_verification(self):
        plan, oracle = quality.validate_plan(self.root, deep=True, require_positions=False)
        self.assertIsNone(oracle)
        self.assertEqual(len(plan["cases"]), 4)

    def test_only_positions_change_within_matched_pixel_arms(self):
        for case in self.plan["cases"]:
            with self.subTest(case=case["id"]):
                temporal = [case["arms"][arm] for arm in quality.TEMPORAL_POSITION_ARMS]
                uniform = [case["arms"][arm]
                           for arm in ("uniform_nominal_positions", "uniform_source_relative_pts")]
                self.assertEqual(len({row["selected_rgb_sha256"] for row in temporal}), 1)
                self.assertEqual(len({tuple(row["timestamps"]) for row in temporal}), 3)
                self.assertEqual(len({row["selected_rgb_sha256"] for row in uniform}), 1)
                self.assertEqual(len({tuple(row["timestamps"]) for row in uniform}), 2)
                self.assertEqual(case["arms"]["temporal_source_pts"]["video_sha256"],
                                 case["arms"]["temporal_source_pts_alt_language"]["video_sha256"])
                self.assertNotEqual(case["arms"]["temporal_source_pts"]["prompt"],
                                    case["arms"]["temporal_source_pts_alt_language"]["prompt"])

    def test_lossless_rgb_controls_do_not_claim_bframe_coverage(self):
        for case in self.plan["cases"]:
            packets = json.loads((self.root / "media" / case["id"] /
                                  "temporal_source_pts.packets.json").read_text())
            self.assertEqual(packets["packet_count"], 8)
            self.assertFalse(packets["packet_pts_nonmonotonic"])
            self.assertTrue(packets["packet_dts_strictly_increasing"])

    def test_semantic_audit_preserves_all_r4_negative_counts(self):
        audit = json.loads((self.root / "audit.json").read_text())
        self.assertTrue(audit["negative_outcomes_preserved"])
        self.assertEqual(audit["r4_answer_counts_across_both_modes"]["temporal"], [0, 24])
        for case in audit["cases"]:
            self.assertFalse(case["uniform_expected_answer_supported_by_selected_pixels"])
            self.assertTrue(case["temporal_expected_answer_supported_by_selected_pixels"])
            self.assertGreater(case["chromatic_anchor_tag"]["native_frame_zero_green_dominant_pixels"], 0)

    def test_pts_drift_fails_even_after_resealing(self):
        with tempfile.TemporaryDirectory(prefix="umm-task47-tamper-") as tmp:
            copied = Path(tmp) / "plan"
            shutil.copytree(self.root, copied)
            plan_path = copied / "plan.json"
            changed = json.loads(plan_path.read_bytes())
            changed["cases"][0]["arms"]["temporal_source_pts"]["timestamps"][1] += 1.0
            quality.write_json(plan_path, changed)
            quality.seal_tree(copied)
            with self.assertRaisesRegex(ValueError, "prompt/PTS drift"):
                quality.validate_plan(copied, deep=False, require_positions=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
