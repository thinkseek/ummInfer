#!/usr/bin/env python3
"""Deterministic CPU tests for Task72; synthetic records are never model evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import temporal_natural_live as producer  # noqa: E402
import temporal_natural_verify as verifier  # noqa: E402
import umminfer_temporal_clean_live as clean  # noqa: E402
import umminfer_temporal_heldout_fixture as historical  # noqa: E402
import umminfer_temporal_natural_fixture as fixture  # noqa: E402
import video_temporal_live as short  # noqa: E402


class FrozenNaturalPopulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = fixture.validate_bundle(fixture.DEFAULT_BUNDLE, deep=False)
        cls.freeze = fixture.validate_freeze(fixture.DEFAULT_BUNDLE, fixture.DEFAULT_FREEZE)
        assert cls.freeze["model_outputs_seen_for_this_revision_at_freeze"] is False
        assert cls.freeze["prior_excluded_natural_outputs_seen_at_freeze"] is True

    def test_eight_distinct_neutral_natural_excerpts_and_languages(self):
        self.assertEqual(len(self.manifest["videos"]), 8)
        self.assertEqual(len(self.manifest["cases"]), 8)
        self.assertTrue(self.manifest["natural_video_claim"])
        self.assertEqual(len({case["language"] for case in self.manifest["cases"]}), 8)
        ranges = [tuple(video["source_frame_range"])
                  for video in self.manifest["videos"].values()]
        self.assertEqual(len(set(ranges)), 8)
        for (_, previous_end), (next_start, _) in zip(sorted(ranges), sorted(ranges)[1:]):
            self.assertLess(previous_end, next_start)
        self.assertEqual(self.manifest["source"]["original_git_blob_sha1"],
                         "fa078dc543b273922dfc8c07e74f63306778a3ed")
        self.assertEqual(self.manifest["measurement"]["requests_total"], 320)
        self.assertEqual(self.manifest["measurement"]["expected_parity_checks"], 304)

    def test_no_question_or_request_metadata_answer_leakage(self):
        for case in self.manifest["cases"]:
            folded = case["question"].casefold()
            self.assertTrue(all(term.casefold() not in folded for term in case["expect_all"]))
            self.assertRegex(case["id"], r"natural-0[1-8]\Z")
            body = clean.build_body_after_context(
                "data:video/mp4;base64,AA==", case["question"], "temporal", "qwen3-native",
                self.manifest["request_contract"], 8)
            serialized = json.dumps(body, ensure_ascii=False)
            for term in case["expect_all"]:
                self.assertNotIn(term.casefold(), body["messages"][0]["content"][1]["text"].casefold())
            self.assertNotIn(case["ground_truth"]["answer"], serialized)
            self.assertNotIn("source_frame_range", serialized)

    def test_first_last_anchors_and_all_answer_evidence(self):
        for video_id, video in self.manifest["videos"].items():
            self.assertTrue(video["decode_order"]["decode_order_pts_nonmonotonic"])
            self.assertTrue(video["decode_order"]["decode_order_dts_strictly_increasing"])
            for budget in fixture.BUDGETS:
                selected = video["expected_temporal"][str(budget)]
                self.assertEqual(selected["source_frame_indices"][0], 0)
                self.assertEqual(selected["source_frame_indices"][-1], video["frame_count"] - 1)
                self.assertEqual(len(selected["content_fnv1a64"]), budget)
        for case in self.manifest["cases"]:
            self.assertTrue(all(row["answer_anchor_present"]
                                for row in case["anchor_audit"].values()))

    def test_uniform_eight_retains_repeated_source_candidates_and_nominal_pts(self):
        video = self.manifest["videos"]["natural-01"]
        selected = video["expected_uniform"]["8"]
        self.assertEqual(selected["source_frame_binding"],
                         "AMBIGUOUS_EXACT_REPEATED_SOURCE_RGB_ALL_CANDIDATES_RETAINED")
        self.assertEqual(len(selected["source_frame_candidates"]), 8)
        self.assertEqual(len(selected["source_pts_candidates"]), 8)
        self.assertTrue(any(len(matches) > 1 for matches in selected["source_frame_candidates"]))
        self.assertEqual(len(selected["native_nominal_timestamps"]), 8)

    def test_stream_duration_precedes_container_duration_for_native_nominal_pts(self):
        video = self.manifest["videos"]["natural-03"]
        self.assertEqual(video["duration_seconds"], 0.666667)
        self.assertEqual(video["ffprobe"]["format"]["duration"], "0.667000")
        self.assertEqual(video["expected_uniform"]["8"]["native_nominal_timestamps"],
                         [0.0, 0.083333375, 0.16666675, 0.250000125,
                          0.3333335, 0.416666875, 0.50000025, 0.583333625])

    def test_task66_fixture_and_scores_unchanged(self):
        old = historical.DEFAULT_BUNDLE
        self.assertEqual(fixture.sha256_file(old / "manifest.json"), fixture.TASK66_MANIFEST_SHA256)
        self.assertEqual(fixture.sha256_file(old / "preregistration.json"),
                         fixture.TASK66_PREREGISTRATION_SHA256)
        self.assertEqual(fixture.sha256_file(old / "SHA256SUMS"), fixture.TASK66_SUMS_SHA256)
        control = self.manifest["historical_task66_control"]
        self.assertEqual((control["primary_temporal8_hits"], control["primary_uniform8_hits"],
                          control["primary_sign_test_p"]), (18, 18, 1.0))
        self.assertFalse(control["included_in_natural_denominator"])

    def test_schedule_request_contract_and_counts(self):
        prepared = producer.prepare(self.manifest)
        self.assertEqual(prepared["_budgets"], [4, 8])
        for budget in fixture.BUDGETS:
            arms = clean.arms_for_budget(budget)
            schedule = clean.controlled.build_schedule(arms, clean.controlled.parse_contract({
                "schema_version": 1, "warmup_repetitions": 1, "measured_repetitions": 3,
                "arm_order": "interleaved", "seed": 510072}))
            self.assertEqual(len(schedule) * 8, 96 if budget == 8 else 64)
            for arm in arms:
                body = clean.build_body_after_context(
                    "data:video/mp4;base64,AA==", self.manifest["cases"][0]["question"], arm,
                    "qwen3-native", self.manifest["request_contract"], budget)
                self.assertEqual(body["seed"], 510072)
                self.assertEqual(body["max_tokens"], 64)
                video = body["messages"][0]["content"][0]["video_url"]
                if arm == "default":
                    self.assertNotIn("frame_selection", video)
                else:
                    self.assertEqual(video["frame_selection"]["max_frames"], budget)


class IndependentSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = fixture.validate_bundle(fixture.DEFAULT_BUNDLE, deep=False)

    def test_random_scores_match_independent_accepted_selector(self):
        rng = random.Random(720072)
        for count in (2, 4, 8, 15, 20, 30, 64):
            for _ in range(25):
                scores = [[0.0, 0.0]] + [[rng.choice((0.0, 0.01, 0.03, 0.1)),
                                          rng.choice((0.0, 0.1, 0.3, 0.8))]
                                         for _ in range(count - 1)]
                for budget in fixture.BUDGETS:
                    self.assertEqual(fixture.select_motion_run(scores, budget),
                                     verifier.select_motion_run(scores, budget))

    def test_frozen_scores_and_content_match_independent_selector(self):
        for video_id, video in self.manifest["videos"].items():
            rows = [json.loads(line) for line in
                    (fixture.DEFAULT_BUNDLE / video["content_frames"]).read_text().splitlines()]
            for budget in fixture.BUDGETS:
                positions = verifier.select_motion_run(video["candidate_scores_motion_scene"], budget)
                selected = video["expected_temporal"][str(budget)]
                self.assertEqual(positions, selected["candidate_indices"])
                self.assertEqual(selected["content_fnv1a64"],
                                 [rows[position]["decoded_native_rgb_fnv1a64"] for position in positions])

    def test_static_fill_and_run_representative_tie_breaks(self):
        scores = [[0.0, 0.0] for _ in range(30)]
        for index in (2, 3, 4, 20, 21, 22, 23, 27, 28):
            scores[index] = [0.2, 0.1]
        scores[3], scores[21] = [0.9, 0.1], [0.8, 0.2]
        selected = verifier.select_motion_run(scores, 8)
        self.assertIn(3, selected)
        self.assertIn(21, selected)
        self.assertIn(27, selected)
        self.assertNotIn(28, selected)
        static = verifier.select_motion_run([[0.0, 0.0] for _ in range(30)], 8)
        self.assertEqual(static, [0, 4, 8, 12, 17, 21, 25, 29])


def native_for(case, arm, budget, mode="serial"):
    expected = case["expected_temporal"][str(budget)] if arm == "temporal" else (
        case["_expected_uniform"][str(budget)])
    timestamps = expected["source_timestamps_seconds"]
    if arm in ("default", "uniform") and budget == 8:
        timestamps = expected["native_nominal_timestamps"]
    grid_t = (budget + 1) // 2
    prompt_tokens = 100 + grid_t * 40
    return {
        "request_id": "http-test:vlm", "policy": "temporal" if arm == "temporal" else "uniform",
        "engine_path": "serial" if mode == "serial" else "batched",
        "decoded_frame_count": case["candidate_count"] if arm == "temporal" else budget,
        "selected_frame_count": budget, "selected_candidate_indices": expected["candidate_indices"],
        "selected_source_frame_indices": expected["source_frame_indices"] if arm == "temporal" or budget == 4
            else list(range(8)),
        "source_indices_exact": arm == "temporal" or budget == 4,
        "selected_frame_timestamps": timestamps,
        "selected_frame_content_fnv1a64": expected["content_fnv1a64"],
        "has_nominal_timestamps": arm in ("default", "uniform") and budget == 8,
        "temporal2_pair_first_timestamps": [timestamps[min(2 * i, budget - 1)] for i in range(grid_t)],
        "max_frames": budget, "max_candidate_frames": 64, "motion_threshold": 0.02,
        "scene_threshold": 0.25, "grid_t": grid_t, "grid_h": 10, "grid_w": 16,
        "premerge_patch_rows": grid_t * 10 * 16, "merged_tokens_per_slice": 40,
        "video_token_count": grid_t * 40, "prompt_token_count": prompt_tokens,
        "media_hash": "0123456789abcdef",
        "mrope_next": prompt_tokens - grid_t * 40 + grid_t * 8,
        "mrope_fnv1a64": "0123456789abcdef",
    }


class NativeEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = producer.prepare(fixture.validate_bundle(fixture.DEFAULT_BUNDLE, deep=False))
        cls.case = cls.manifest["cases"][0]

    def execution(self):
        return {"request_id": "http-test:vlm", "prefix_reused": 0, "vision_ms": 1.0,
                "prefill_ms": 2.0, "decode_ms": 3.0, "elapsed_ms": 6.0}

    def test_temporal2_mrope_and_source_pts_accepted(self):
        for arm, budget in fixture.ARM_BUDGETS:
            native = native_for(self.case, arm, budget)
            short.validate_native(native, arm)
            result = producer.long_live.validate_native_selection(
                native, self.execution(), case=self.case, arm=arm, mode="serial",
                short=short, budget=budget)
            self.assertEqual(result["status"], "COVERED")
            self.assertEqual(native["mrope_next"], native["prompt_token_count"]
                             - native["video_token_count"] + native["grid_t"] * 8)

    def test_wrong_rgb_source_pts_and_prefix_reuse_fail_closed(self):
        for field in ("selected_frame_content_fnv1a64", "selected_frame_timestamps",
                      "temporal2_pair_first_timestamps"):
            native = native_for(self.case, "temporal", 8)
            native[field] = list(native[field])
            native[field][0] = "0" * 16 if field == "selected_frame_content_fnv1a64" else 999.0
            with self.assertRaises(ValueError):
                producer.long_live.validate_native_selection(
                    native, self.execution(), case=self.case, arm="temporal", mode="serial",
                    short=short, budget=8)
        execution = self.execution()
        execution["prefix_reused"] = 1
        with self.assertRaises(ValueError):
            producer.long_live.validate_native_selection(
                native_for(self.case, "temporal", 8), execution, case=self.case,
                arm="temporal", mode="serial", short=short, budget=8)


def synthetic_records(manifest):
    records = []
    for case in manifest["cases"]:
        for arm, budget in fixture.ARM_BUDGETS:
            for mode in producer.MODES:
                for repetition in (1, 2, 3):
                    identifier = f"{case['id']}:{mode}:{arm}:{budget}:{repetition}"
                    records.append({"clip": case["id"], "video_id": case["video_id"],
                                    "mode": mode, "phase": "measured", "repetition": repetition,
                                    "arm": arm, "budget": budget, "quality_scored": True,
                                    "evaluation_role": "quality_scored", "status": "COMPLETED",
                                    "native_evidence_valid": True, "expected_answer_hit": True,
                                    "raw_answer": "synthetic-hit", "response": {"request_id": identifier},
                                    "latency_ms": {"total": 10.0, "native_elapsed": 9.0}})
    return records


class AnalysisAndVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = fixture.validate_bundle(fixture.DEFAULT_BUNDLE, deep=False)

    def test_exact_statistics_and_primary_population(self):
        self.assertEqual(producer.sign_test(2, 0), verifier.sign_test(2, 0))
        self.assertEqual(producer.sign_test(2, 0), 0.5)
        self.assertEqual(producer.sign_test(0, 0), 1.0)
        self.assertEqual(producer.wilson(24), verifier.wilson(24))
        records = synthetic_records(self.manifest)
        self.assertEqual(len(records), 240)
        analysis = producer.quality_analysis(records, self.manifest)
        self.assertTrue(all(value["hits"] == 48 for value in analysis["quality"].values()))
        primary = analysis["comparisons"]["primary_temporal8_vs_uniform8"]
        self.assertEqual(primary["paired_case_mode_count"], 16)
        self.assertEqual(primary["exact_two_sided_sign_test_p"], 1.0)
        self.assertEqual(analysis["every_miss"], [])
        self.assertEqual(analysis, verifier.quality_analysis(records, self.manifest))

    def test_miss_regression_and_outlier_retained(self):
        records = synthetic_records(self.manifest)
        target = next(row for row in records if row["clip"] == "natural-03"
                      and row["mode"] == "serial" and row["arm"] == "temporal"
                      and row["budget"] == 8 and row["repetition"] == 1)
        target.update({"expected_answer_hit": False, "raw_answer": "synthetic-miss",
                       "latency_ms": {"total": 100.0, "native_elapsed": 90.0}})
        analysis = producer.quality_analysis(records, self.manifest)
        self.assertEqual(analysis["quality"]["temporal-8"]["hits"], 47)
        self.assertEqual(len(analysis["every_miss"]), 1)
        primary = analysis["comparisons"]["primary_temporal8_vs_uniform8"]
        self.assertEqual((primary["improved_pairs"], primary["worsened_pairs"]), (0, 1))
        self.assertEqual(analysis["quality_regression_count"], 2)
        outliers = producer.latency_outliers(records)
        self.assertEqual(len(outliers), 2)
        self.assertEqual(outliers, verifier.latency_outliers(records))
        self.assertTrue(all(row["retained_in_aggregates"] for row in outliers))

    def test_nested_excluded_attempt_manifest_and_sums_are_covered(self):
        with tempfile.TemporaryDirectory(prefix="umm-natural-nested-seal-") as tmp:
            root = Path(tmp)
            nested = root / "excluded-attempts" / "old"
            nested.mkdir(parents=True)
            (root / "report.json").write_text("{}")
            (nested / "record.json").write_text("{}")
            (nested / "artifact-manifest.json").write_text("{}")
            (nested / "SHA256SUMS").write_text("nested-checksum\n", encoding="ascii")
            producer.seal(root)
            self.assertEqual(verifier.verify_sums(root), 5)
            listed = {row["path"] for row in json.loads(
                (root / "artifact-manifest.json").read_bytes())["files"]}
            self.assertIn("excluded-attempts/old/artifact-manifest.json", listed)
            self.assertIn("excluded-attempts/old/SHA256SUMS", listed)

    def test_checksum_and_verifier_input_tamper_rejection(self):
        with tempfile.TemporaryDirectory(prefix="umm-natural-seal-") as tmp:
            root = Path(tmp)
            (root / "a.json").write_text("{}")
            (root / "SHA256SUMS").write_text(
                f"{fixture.sha256_file(root / 'a.json')}  a.json\n", encoding="ascii")
            self.assertEqual(verifier.verify_sums(root, fixture=True), 1)
            (root / "a.json").write_text("changed")
            with self.assertRaises(ValueError):
                verifier.verify_sums(root, fixture=True)
        manifest = copy.deepcopy(self.manifest)
        manifest["cases"][0]["question"] = "tree"
        with self.assertRaises(ValueError):
            fixture.validate_case_contract(manifest["cases"] and json.loads(
                (fixture.DEFAULT_BUNDLE / "preregistration.json").read_bytes()), manifest)


class DeepFixtureProofTests(unittest.TestCase):
    def test_full_decode_and_byte_reproducible_regeneration_from_pinned_git_blob(self):
        fixture.validate_bundle(fixture.DEFAULT_BUNDLE, deep=True)
        with tempfile.TemporaryDirectory(prefix="umm-natural-generate-") as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            source_bytes = subprocess.run([
                "git", "-C", str(ROOT), "show",
                self.manifest_commit() + ":docs/assets/umminfer-apple-16x9.mp4"],
                check=True, capture_output=True, timeout=60).stdout
            source.write_bytes(source_bytes)
            generated = root / "bundle"
            fixture.generate_bundle(generated, source,
                                    fixture.DEFAULT_BUNDLE / "preregistration.json")
            actual = {path.relative_to(fixture.DEFAULT_BUNDLE).as_posix(): path.read_bytes()
                      for path in fixture.DEFAULT_BUNDLE.rglob("*") if path.is_file()}
            rebuilt = {path.relative_to(generated).as_posix(): path.read_bytes()
                       for path in generated.rglob("*") if path.is_file()}
            self.assertEqual(rebuilt, actual)

    @staticmethod
    def manifest_commit():
        return fixture.DEFAULT_BUNDLE.joinpath("preregistration.json").is_file() and \
            json.loads((fixture.DEFAULT_BUNDLE / "preregistration.json").read_bytes())[
                "source"]["original_commit"]


if __name__ == "__main__":
    unittest.main(verbosity=2)
