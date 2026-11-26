#!/usr/bin/env python3
"""Deterministic CPU tests for the preregistered temporal motion-run experiment."""
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import temporal_motion_run_live as producer  # noqa: E402
import temporal_motion_run_verify as verifier  # noqa: E402
import umminfer_long_video_fixture as frozen  # noqa: E402
import umminfer_temporal_clean_fixture as fixture  # noqa: E402
from temporal_clean_test import FakeHarness  # noqa: E402


class MotionRunOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = producer.clean.load_manifest(
            fixture.DEFAULT_BUNDLE / "manifest.json", deep=False)
        cls.oracles = producer.candidate_oracles(cls.manifest, fixture.DEFAULT_BUNDLE)

    def test_frozen_scores_reproduce_both_independent_selectors(self):
        rng = random.Random(610061)
        for _ in range(100):
            scores = [[0.0, 0.0]] + [[rng.choice((0.0, 0.01, 0.03, 0.1)),
                                      rng.choice((0.0, 0.1, 0.3, 0.8))]
                                     for _ in range(20)]
            for budget in (4, 8, 16):
                self.assertEqual(verifier.select(scores, budget, False),
                                 frozen.select_temporal(scores, budget))
                self.assertEqual(verifier.select(scores, budget, True),
                                 producer.select_motion_run(scores, budget))

    def test_motion_run_changes_only_bounded_event_representation(self):
        self.assertEqual(self.oracles["motion-clean"]["8"]["source_frame_indices"],
                         [0, 24, 64, 128, 184, 192, 328, 448])
        self.assertEqual(self.oracles["near-clean"]["4"]["source_frame_indices"],
                         [0, 152, 288, 448])
        self.assertEqual(self.oracles["scene-clean"]["8"],
                         self.manifest["videos"]["scene-clean"]["expected_temporal"]["8"])
        for video_id, budgets in self.oracles.items():
            for budget, selected in budgets.items():
                self.assertEqual(len(selected["candidate_indices"]), int(budget))
                self.assertEqual(selected["candidate_indices"],
                                 sorted(set(selected["candidate_indices"])))
                self.assertEqual(selected["source_frame_indices"][0], 0, video_id)
                self.assertEqual(selected["source_frame_indices"][-1], 448, video_id)
                self.assertEqual(selected["content_fnv1a64"],
                                 [self._row(video_id, source)["decoded_native_rgb_fnv1a64"]
                                  for source in selected["source_frame_indices"]])

    def _row(self, video_id, source):
        rows = self.manifest["videos"][video_id]["content_frames"]
        if not hasattr(self, "_rows"):
            self._rows = {}
        if video_id not in self._rows:
            self._rows[video_id] = [json.loads(line) for line in
                                    (fixture.DEFAULT_BUNDLE / rows).read_text().splitlines()]
        return self._rows[video_id][source]

    def test_run_boundary_ties_static_fill_and_anchors(self):
        scores = [[0.0, 0.0] for _ in range(16)]
        for index in (2, 3, 4, 9, 10):
            scores[index] = [0.5, 0.1]
        scores[3] = [0.9, 0.1]
        scores[9] = [0.8, 0.2]
        self.assertEqual(producer.threshold_runs(scores, 0.02, scene=False),
                         [[2, 3, 4], [9, 10]])
        selected = producer.select_motion_run(scores, 8)
        self.assertEqual(selected, [0, 2, 3, 4, 6, 9, 11, 15])
        self.assertIn(3, selected)  # strongest member of first run
        self.assertIn(9, selected)  # second run remains eligible
        static = producer.select_motion_run([[0.0, 0.0] for _ in range(16)], 8)
        self.assertEqual(static, [0, 2, 4, 6, 9, 11, 13, 15])

    def test_frozen_scoring_retains_japanese_miss_and_no_new_alias(self):
        case = next(case for case in self.manifest["cases"] if case["id"] == "near-static-ja")
        self.assertFalse(verifier.hit("円", case))
        self.assertTrue(verifier.hit("リング", case))
        self.assertFalse(verifier.hit("triangle", case))


class ControlledAggregationTests(unittest.TestCase):
    def test_synthetic_schedule_parity_and_preregistered_decision(self):
        with tempfile.TemporaryDirectory(prefix="umm-motion-run-test-") as tmp:
            synthetic = FakeHarness(warmup=1, measured=3, budgets=(4, 8, 16)).run(
                Path(tmp) / "synthetic")
            self.assertEqual(synthetic["evidence_class"], producer.clean.SYNTHETIC_CLASS)
            modes = {}
            for mode in producer.MODES:
                modes[mode] = {
                    "records": [row for row in synthetic["records"] if row["mode"] == mode],
                    "provenance": synthetic["provenance"][mode],
                    "steps": [row for row in synthetic["steps"] if row["mode"] == mode],
                    "resources": [row for row in synthetic["resources"] if row["mode"] == mode],
                    "cleanup_checks": [row for row in synthetic["cleanup_checks"]
                                       if row["mode"] == mode],
                    "exclusive": [row for row in synthetic["exclusive_device"]
                                  if row["mode"] == mode],
                    "stale_errors": [], "candidate_oracles": None,
                    "post_stop_cleanup": {"passed": True},
                }
            baseline = producer.make_phase(modes)
            self.assertEqual(len(baseline["records"]), 448)
            self.assertEqual(len(baseline["parity"]), 400)
            self.assertTrue(baseline["gate_pass"], baseline["gate_reasons"])
            negative = producer.comparison(baseline, copy.deepcopy(baseline))
            self.assertTrue(any("did not exceed" in reason for reason in negative["gate_reasons"]))
            self.assertEqual(producer.hit_to_miss_repetitions(baseline, baseline, "temporal", 8), 0)
            swapped_base = copy.deepcopy(baseline)
            swapped_candidate = copy.deepcopy(baseline)
            base_rows = producer.quality_rows(swapped_base, "temporal", 8)
            candidate_rows = producer.quality_rows(swapped_candidate, "temporal", 8)
            motion = [i for i, row in enumerate(base_rows) if row["clip"] == "motion-first-en"
                      and row["mode"] == "serial"]
            base_rows[motion[1]]["expected_answer_hit"] = False
            candidate_rows[motion[0]]["expected_answer_hit"] = False
            self.assertEqual(producer.hit_to_miss_repetitions(
                swapped_base, swapped_candidate, "temporal", 8), 1)
            self.assertEqual(len(negative["default_uniform_cross_phase"]), 256)
            self.assertTrue(all(row["passed"] for row in negative["default_uniform_cross_phase"]))

            positive_base = copy.deepcopy(baseline)
            for row in positive_base["records"]:
                if row["phase"] == "measured" and row["quality_scored"] \
                        and row["arm"] == "temporal" and row["budget"] == 8 \
                        and row["clip"] == "motion-first-en":
                    row["expected_answer_hit"] = False
            positive = producer.comparison(positive_base, copy.deepcopy(baseline))
            self.assertEqual(positive["gate_reasons"], [])
            effect = positive["quality"]["temporal-8"]["paired_case_mode_effect"]
            self.assertEqual((effect["improved_pairs"], effect["worsened_pairs"]), (2, 0))
            self.assertEqual(effect["exact_two_sided_sign_test_p"], 0.5)


class SealTests(unittest.TestCase):
    def test_independent_checksum_set_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory(prefix="umm-motion-seal-") as tmp:
            root = Path(tmp)
            (root / "a.json").write_text("{}")
            (root / "SHA256SUMS").write_text(
                f"{producer.sha256_file(root / 'a.json')}  a.json\n", encoding="ascii")
            self.assertEqual(verifier.verify_sums(root, fixture=True), 1)
            (root / "a.json").write_text("changed")
            with self.assertRaises(ValueError):
                verifier.verify_sums(root, fixture=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
