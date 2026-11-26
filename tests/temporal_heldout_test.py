#!/usr/bin/env python3
"""Deterministic CPU tests for Task66; synthetic records are never model evidence."""
from __future__ import annotations

import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import temporal_heldout_live as producer  # noqa: E402
import temporal_heldout_verify as verifier  # noqa: E402
import umminfer_long_video_live as historical_live  # noqa: E402
import umminfer_temporal_clean_fixture as historical_fixture  # noqa: E402
import umminfer_temporal_heldout_fixture as fixture  # noqa: E402
import video_temporal_live as short  # noqa: E402


class FrozenPopulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = fixture.validate_bundle(fixture.DEFAULT_BUNDLE, deep=False)
        cls.freeze = fixture.validate_freeze(fixture.DEFAULT_BUNDLE, fixture.DEFAULT_FREEZE)

    def test_freeze_population_sources_and_no_natural_video_claim(self):
        self.assertEqual(len(self.manifest["videos"]), 8)
        self.assertEqual(len(self.manifest["cases"]), 10)
        self.assertFalse(self.manifest["natural_video_claim"])
        self.assertEqual(self.manifest["source"]["asset_class"],
                         "NEW_PROCEDURAL_SYNTHETIC_INPUT_NOT_MODEL_OUTPUT")
        self.assertFalse(self.freeze["historical_task61_cases_in_denominator"])
        self.assertFalse(fixture.HISTORICAL_TASK61 & {case["id"] for case in self.manifest["cases"]})
        self.assertEqual(self.manifest["measurement"]["requests_total"], 560)
        self.assertEqual(self.manifest["measurement"]["expected_parity_checks"], 500)

    def test_task61_bytes_scores_and_japanese_miss_are_unchanged(self):
        old = historical_fixture.DEFAULT_BUNDLE
        self.assertEqual(fixture.sha256_file(old / "manifest.json"),
                         "62f475a74732fa3504850ebe4517dc98903ba39202dce457441c52133a6c96c0")
        self.assertEqual(fixture.sha256_file(old / "preregistration.json"),
                         "96608dc793d54f64f55a7c46862d1b1737cbf807a8807d1b22b2fcd0502d2a77")
        manifest = json.loads((old / "manifest.json").read_bytes())
        japanese = next(case for case in manifest["cases"] if case["id"] == "near-static-ja")
        self.assertFalse(historical_live.answer_hit("円", japanese))
        self.assertTrue(historical_live.answer_hit("リング", japanese))
        self.assertFalse(verifier.hit("円", japanese))

    def test_new_multilingual_scoring_is_frozen_without_observed_aliases(self):
        by_id = {case["id"]: case for case in self.manifest["cases"]}
        self.assertTrue(verifier.hit("正方形", by_id["heldout-static-square-ja"]))
        self.assertFalse(verifier.hit("円", by_id["heldout-static-square-ja"]))
        self.assertTrue(verifier.hit("croix", by_id["heldout-bframe-first-fr"]))
        self.assertTrue(verifier.hit("Quadrat", by_id["heldout-order-after-de"]))
        self.assertTrue(verifier.hit("anello", by_id["heldout-single-flash-it"]))
        for case in self.manifest["cases"]:
            folded = case["question"].casefold()
            self.assertTrue(all(term.casefold() not in folded for term in case["expect_all"]))

    def test_anchor_coverage_and_all_temporal_first_last_budgets(self):
        for video_id, video in self.manifest["videos"].items():
            self.assertTrue(video["decode_order"]["decode_order_pts_nonmonotonic"], video_id)
            self.assertTrue(video["decode_order"]["decode_order_dts_monotonic"], video_id)
            for budget in fixture.BUDGETS:
                selected = video["expected_temporal"][str(budget)]
                self.assertEqual(selected["source_frame_indices"][0], 0)
                self.assertEqual(selected["source_frame_indices"][-1], 448)
                self.assertEqual(selected["source_frame_indices"],
                                 sorted(set(selected["source_frame_indices"])))
                self.assertEqual(len(selected["content_fnv1a64"]), budget)
        for case in self.manifest["cases"]:
            for budget in case["answer_anchor"]["required_temporal_budgets"]:
                self.assertTrue(case["anchor_audit"][f"temporal-{budget}"]["answer_anchor_present"],
                                f"{case['id']}/{budget}")

    def test_every_source_pixel_is_neutral_and_declared_static_intervals_repeat(self):
        for video_id, video in self.manifest["videos"].items():
            rows = [json.loads(line) for line in
                    (fixture.DEFAULT_BUNDLE / video["content_frames"]).read_text().splitlines()]
            for index in (0, 1, 63, 64, 223, 448):
                frame = fixture.source_frame(video_id, index)
                for offset in range(0, len(frame), 3):
                    self.assertEqual(frame[offset], frame[offset + 1])
                    self.assertEqual(frame[offset], frame[offset + 2])
                self.assertEqual(fixture.sha256_bytes(frame), rows[index]["source_rgb96x64_sha256"])
            for start, end in video["events"]["exact_static_intervals"]:
                self.assertEqual(len({row["source_rgb96x64_sha256"] for row in rows[start:end + 1]}), 1)

    def test_request_contract_and_schedule_counts(self):
        prepared = producer.prepare(self.manifest)
        self.assertEqual(prepared["_budgets"], [4, 8, 16])
        contract = self.manifest["request_contract"]
        for arm, budget in fixture.ARM_BUDGETS:
            case = self.manifest["cases"][0]
            body = producer.clean.build_body_after_context(
                "data:video/mp4;base64,AA==", case["question"], arm, "qwen3-native",
                contract, budget)
            self.assertEqual(body["seed"], 510066)
            self.assertEqual(body["max_tokens"], 64)
            video = body["messages"][0]["content"][0]["video_url"]
            if arm == "default":
                self.assertNotIn("frame_selection", video)
            else:
                self.assertEqual(video["frame_selection"]["max_frames"], budget)
        for budget in fixture.BUDGETS:
            arms = producer.clean.arms_for_budget(budget)
            schedule = producer.clean.controlled.build_schedule(
                arms, producer.clean.controlled.parse_contract({
                    "schema_version": 1, "warmup_repetitions": 1, "measured_repetitions": 3,
                    "arm_order": "interleaved", "seed": 510066}))
            self.assertEqual(len(schedule) * 10, {4: 80, 8: 120, 16: 80}[budget])


class IndependentSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = fixture.validate_bundle(fixture.DEFAULT_BUNDLE, deep=False)

    def test_random_scores_match_independent_accepted_selector(self):
        rng = random.Random(660066)
        for _ in range(100):
            scores = [[0.0, 0.0]] + [[rng.choice((0.0, 0.01, 0.03, 0.1)),
                                      rng.choice((0.0, 0.1, 0.3, 0.8))] for _ in range(56)]
            for budget in fixture.BUDGETS:
                self.assertEqual(fixture.select_motion_run(scores, budget),
                                 verifier.select_motion_run(scores, budget))

    def test_frozen_scores_match_independent_selector_and_content(self):
        for video_id, video in self.manifest["videos"].items():
            rows = [json.loads(line) for line in
                    (fixture.DEFAULT_BUNDLE / video["content_frames"]).read_text().splitlines()]
            for budget in fixture.BUDGETS:
                positions = verifier.select_motion_run(video["candidate_scores_motion_scene"], budget)
                selected = video["expected_temporal"][str(budget)]
                self.assertEqual(positions, selected["candidate_indices"])
                self.assertEqual(selected["content_fnv1a64"],
                                 [rows[position * 8]["decoded_native_rgb_fnv1a64"]
                                  for position in positions])

    def test_static_fill_and_run_representative_tie_breaks(self):
        scores = [[0.0, 0.0] for _ in range(57)]
        for index in (2, 3, 4, 20, 21, 22, 23, 40, 41):
            scores[index] = [0.2, 0.1]
        scores[3] = [0.9, 0.1]
        scores[21] = [0.8, 0.2]
        selected = verifier.select_motion_run(scores, 8)
        self.assertIn(3, selected)
        self.assertIn(21, selected)
        self.assertIn(40, selected)  # equal-score third run keeps the earliest representative
        self.assertNotIn(41, selected)
        static = verifier.select_motion_run([[0.0, 0.0] for _ in range(57)], 8)
        self.assertEqual(static, [0, 8, 16, 24, 32, 40, 48, 56])


def synthetic_records(manifest):
    records = []
    for case in manifest["cases"]:
        for arm, budget in fixture.ARM_BUDGETS:
            for mode in producer.MODES:
                for repetition in (1, 2, 3):
                    identifier = f"{case['id']}:{mode}:{arm}:{budget}:{repetition}"
                    records.append({
                        "clip": case["id"], "video_id": case["video_id"], "mode": mode,
                        "phase": "measured", "repetition": repetition, "arm": arm,
                        "budget": budget, "quality_scored": True, "evaluation_role": "quality_scored",
                        "status": "COMPLETED", "native_evidence_valid": True,
                        "expected_answer_hit": True, "raw_answer": "synthetic-hit",
                        "response": {"request_id": identifier}, "latency_ms": {"total": 10.0,
                                                                               "native_elapsed": 9.0},
                    })
    return records


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = fixture.validate_bundle(fixture.DEFAULT_BUNDLE, deep=False)

    def test_wilson_and_exact_sign_test_values(self):
        self.assertEqual(producer.sign_test(2, 0), 0.5)
        self.assertEqual(producer.sign_test(0, 0), 1.0)
        self.assertEqual(producer.sign_test(1, 1), 1.0)
        self.assertEqual(producer.wilson(30), verifier.wilson(30))
        self.assertAlmostEqual(producer.wilson(30)[0], 0.3773502424155577)

    def test_quality_population_raw_answers_and_every_miss(self):
        records = synthetic_records(self.manifest)
        analysis = producer.quality_analysis(records, self.manifest)
        self.assertEqual(len(records), 420)
        self.assertTrue(all(value["hits"] == 60 for value in analysis["quality"].values()))
        primary = analysis["comparisons"]["primary_temporal8_vs_uniform8"]
        self.assertEqual(primary["paired_case_mode_count"], 20)
        self.assertEqual(primary["exact_two_sided_sign_test_p"], 1.0)
        self.assertEqual(analysis["every_miss"], [])
        self.assertEqual(producer.per_case_quality(records, self.manifest),
                         verifier.quality_analysis(records, self.manifest)["per_case"])

    def test_hit_to_miss_regression_and_outlier_are_retained(self):
        records = synthetic_records(self.manifest)
        target = next(row for row in records if row["clip"] == "heldout-separated-middle-en"
                      and row["mode"] == "serial" and row["arm"] == "temporal"
                      and row["budget"] == 8 and row["repetition"] == 1)
        target["expected_answer_hit"] = False
        target["raw_answer"] = "synthetic-miss"
        target["latency_ms"] = {"total": 100.0, "native_elapsed": 90.0}
        analysis = producer.quality_analysis(records, self.manifest)
        self.assertEqual(analysis["quality"]["temporal-8"]["hits"], 59)
        self.assertEqual(len(analysis["every_miss"]), 1)
        self.assertEqual(analysis["every_miss"][0]["raw_answer"], "synthetic-miss")
        primary = analysis["comparisons"]["primary_temporal8_vs_uniform8"]
        self.assertEqual((primary["improved_pairs"], primary["worsened_pairs"]), (0, 1))
        self.assertEqual(primary["exact_two_sided_sign_test_p"], 1.0)
        # The same temporal-8 regression is retained in both preregistered
        # primary and temporal-8-versus-default comparisons.
        self.assertEqual(analysis["quality_regression_count"], 2)
        self.assertEqual(analysis["runtime_change_in_response_to_answers"], "FORBIDDEN_NONE_MADE")
        outliers = producer.latency_outliers(records)
        self.assertEqual(len(outliers), 2)
        self.assertTrue(all(row["retained_in_aggregates"] for row in outliers))
        self.assertEqual(producer.latency_outliers(records), verifier.latency_outliers(records))

    def test_checksum_tamper_rejection(self):
        with tempfile.TemporaryDirectory(prefix="umm-heldout-seal-") as tmp:
            root = Path(tmp)
            (root / "a.json").write_text("{}")
            (root / "SHA256SUMS").write_text(
                f"{fixture.sha256_file(root / 'a.json')}  a.json\n", encoding="ascii")
            self.assertEqual(verifier.verify_sums(root, fixture=True), 1)
            (root / "a.json").write_text("changed")
            with self.assertRaises(ValueError):
                verifier.verify_sums(root, fixture=True)


class DeepFixtureProofTests(unittest.TestCase):
    def test_deep_decode_and_byte_reproducible_regeneration(self):
        fixture.validate_bundle(fixture.DEFAULT_BUNDLE, deep=True)
        with tempfile.TemporaryDirectory(prefix="umm-heldout-generate-") as tmp:
            generated = Path(tmp) / "bundle"
            fixture.generate_bundle(
                generated, preregistration_path=fixture.DEFAULT_BUNDLE / "preregistration.json")
            actual = {path.relative_to(fixture.DEFAULT_BUNDLE).as_posix(): path.read_bytes()
                      for path in fixture.DEFAULT_BUNDLE.rglob("*") if path.is_file()}
            rebuilt = {path.relative_to(generated).as_posix(): path.read_bytes()
                       for path in generated.rglob("*") if path.is_file()}
            self.assertEqual(rebuilt, actual)


if __name__ == "__main__":
    unittest.main(verbosity=2)
