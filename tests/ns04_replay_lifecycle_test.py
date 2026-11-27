#!/usr/bin/env python3
"""Deterministic fixtures for the NS-04 replay-inclusive scorecard gate."""

import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ns04_replay_lifecycle_scorecard", ROOT / "tests/ns04_replay_lifecycle_scorecard.py")
scorecard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scorecard)
VERIFY_SPEC = importlib.util.spec_from_file_location(
    "ns04_replay_lifecycle_verify", ROOT / "tests/ns04_replay_lifecycle_verify.py")
verifier = importlib.util.module_from_spec(VERIFY_SPEC)
VERIFY_SPEC.loader.exec_module(verifier)
VALID_CONTROLS = {name: {"status": "PASS"} for name in scorecard.REQUIRED_CONTROLS}


def report(arm, repetition, total, replay, asr=100.0, cancel=3.0, silence=1.0):
    digest = scorecard.hashlib.sha256(f"{arm}-{repetition}".encode()).hexdigest()
    return {
        "schema": "umminfer-ns04-replay-lifecycle-independent/v1",
        "status": "PASS",
        "arm": arm,
        "repetition": repetition,
        "raw_frames_sha256": digest,
        "report_identity": digest,
        "preregistration_sha256": "p" * 64,
        "execution_commit": f"{arm}-execution",
        "accepted_parent_commit": "accepted-parent",
        "binaries": {"umm-infer": f"{arm}-binary"},
        "harness": {"sha256": "same-harness"},
        "verifier_sha256": "same-verifier",
        "model_manifest_sha256": {"asr": "same-asr", "vlm": "same-vlm", "tts": "same-tts"},
        "input_schedule": {name: {"chunks": 1, "pcm_bytes": 3200}
                           for name in ("solar", "saturn", "english")},
        "transcripts": {"solar": "太阳系八大行星", "saturn": "土星环", "english": "英文"},
        "history": {"verified_prompts": 3, "final_history_turns": 2,
                    "accepted_cursors": {"req-1": 10, "req-2": 12},
                    "vlm_prompts_sha256": "f" * 64,
                    "stale_events": 0, "cross_session_leaks": 0,
                    "duplicate_terminal_callbacks": 0},
        "first_barge": {
            "timings_ms": {
                "barge_to_cancel": cancel,
                "cancel_to_digital_silence": silence,
                "barge_to_speech_stopped_replay": replay,
                "speech_stopped_to_asr_final": 50.0,
                "asr_final_to_new_audio": asr,
                "replay_inclusive_barge_to_new_audio": total,
            },
            "old_pcm_sha256": "1" * 64,
            "pending_pcm_sha256": "2" * 64,
            "old_response_text_sha256": "3" * 64,
            "played": {"acknowledged_cursor": 10, "acknowledged_text": "The small ",
                       "played_pcm_sha256": "a" * 64},
        },
        "second_barge": {
            "timings_ms": {
                "barge_to_cancel": cancel,
                "cancel_to_digital_silence": silence,
                "barge_to_speech_stopped_replay": replay,
                "speech_stopped_to_asr_final": 50.0,
                "asr_final_to_new_audio": asr,
                "replay_inclusive_barge_to_new_audio": total,
            },
            "old_pcm_sha256": "1" * 64,
            "pending_pcm_sha256": "2" * 64,
            "old_response_text_sha256": "3" * 64,
            "played": {"acknowledged_cursor": 12, "acknowledged_text": "土星环",
                       "played_pcm_sha256": "b" * 64},
        },
        "final_response": {"text_sha256": "c" * 64, "pcm_sha256": "d" * 64},
        "cleanup": {"final_in_flight": 0, "selected_gpu_idle": True},
    }


class ReplayInclusiveGateTests(unittest.TestCase):
    def test_post_asr_pass_cannot_hide_replay_wait(self):
        reports = [report(arm, i, 1000.0 + i, 800.0 + i)
                   for arm in ("before", "after") for i in range(1, 4)]
        result = scorecard.compute_scorecard(reports, preregistration_sha256="p" * 64)
        self.assertEqual(result["promotion_status"], "NOT_PROMOTED")
        self.assertTrue(result["arms"]["after"]["thresholds"]
                        ["asr_final_to_new_audio_max_strictly_below_800"]["pass"])
        # The fixture's post-ASR component itself passes; only the full path fails.
        self.assertFalse(result["arms"]["after"]["thresholds"]
                         ["replay_inclusive_barge_to_new_audio_max_strictly_below_800"]["pass"])

    def test_single_full_path_outlier_blocks_promotion(self):
        reports = [report("before", i, 2000.0, 1700.0) for i in range(1, 4)]
        reports += [report("after", i, 700.0 if i < 3 else 800.0,
                           400.0 if i < 3 else 500.0) for i in range(1, 4)]
        result = scorecard.compute_scorecard(reports, preregistration_sha256="p" * 64)
        threshold = result["arms"]["after"]["thresholds"][
            "replay_inclusive_barge_to_new_audio_max_strictly_below_800"]
        self.assertEqual(threshold["observed_max_ms"], 800.0)
        self.assertFalse(threshold["pass"])
        self.assertEqual(result["promotion_status"], "NOT_PROMOTED")

    def test_lossless_candidate_can_pass_only_with_exact_pairing(self):
        reports = [report("before", i, 2000.0, 1500.0) for i in range(1, 4)]
        reports += [report("after", i, 600.0, 250.0) for i in range(1, 4)]
        result = scorecard.compute_scorecard(
            reports, preregistration_sha256="p" * 64, controls=VALID_CONTROLS)
        self.assertEqual(result["promotion_status"], "PROMOTED")
        tampered = [dict(row) for row in reports]
        tampered[-1] = dict(tampered[-1])
        tampered[-1]["final_response"] = dict(tampered[-1]["final_response"])
        tampered[-1]["final_response"]["pcm_sha256"] = "e" * 64
        failed = scorecard.compute_scorecard(
            tampered, preregistration_sha256="p" * 64, controls=VALID_CONTROLS)
        self.assertEqual(failed["promotion_status"], "NOT_PROMOTED")
        self.assertIn("exact_pairing", failed["failures"])

    def test_missing_controls_and_mixed_models_block_promotion(self):
        reports = [report("before", i, 2000.0, 1500.0) for i in range(1, 4)]
        reports += [report("after", i, 600.0, 250.0) for i in range(1, 4)]
        missing = scorecard.compute_scorecard(reports, preregistration_sha256="p" * 64)
        self.assertEqual(missing["promotion_status"], "NOT_PROMOTED")
        self.assertTrue(any(name.startswith("control_") for name in missing["failures"]))
        mixed = [dict(row) for row in reports]
        mixed[-1] = dict(mixed[-1])
        mixed[-1]["model_manifest_sha256"] = dict(mixed[-1]["model_manifest_sha256"])
        mixed[-1]["model_manifest_sha256"]["tts"] = "different-model"
        rejected = scorecard.compute_scorecard(
            mixed, preregistration_sha256="p" * 64, controls=VALID_CONTROLS)
        self.assertIn("mixed_models", rejected["failures"])
        self.assertEqual(rejected["promotion_status"], "NOT_PROMOTED")

    def candidate_reports(self):
        reports = [report("before", i, 2000.0, 1500.0) for i in range(1, 4)]
        reports += [report("after", i, 600.0, 250.0) for i in range(1, 4)]
        for row in reports:
            row["candidate_preregistration_sha256"] = "c" * 64
            row["incremental"] = {}
            for label in ("first_barge", "second_barge"):
                eligible = row["arm"] == "after"
                row["incremental"][label] = {
                    "eligible_sha256": "e" * 64 if eligible else None,
                    "barge_to_eligible_ms": 250.0 if eligible else None,
                    "barge_to_vlm_start_ms": 251.0,
                    "barge_to_tts_start_ms": 400.0,
                    "eligible_to_vlm_start_ms": 1.0 if eligible else None,
                    "eligible_to_tts_start_ms": 150.0 if eligible else None,
                    "final_matches_eligible": True if eligible else None,
                    "incorrect_early_audible_response": False,
                    "preview_revision_count": 0,
                    "previews": [],
                }
        return reports

    def test_incremental_candidate_hard_failure_gates(self):
        reports = self.candidate_reports()
        result = scorecard.compute_scorecard(
            reports, preregistration_sha256="p" * 64, controls=VALID_CONTROLS,
            candidate_preregistration_sha256="c" * 64)
        self.assertEqual(result["promotion_status"], "PROMOTED")
        self.assertEqual(len(result["incremental_by_run"]), 6)
        for field, value in (("incorrect_early_audible_response", True),
                             ("preview_revision_count", 1),
                             ("final_matches_eligible", False)):
            tampered = copy.deepcopy(reports)
            tampered[-1]["incremental"]["first_barge"][field] = value
            failed = scorecard.compute_scorecard(
                tampered, preregistration_sha256="p" * 64, controls=VALID_CONTROLS,
                candidate_preregistration_sha256="c" * 64)
            self.assertEqual(failed["promotion_status"], "NOT_PROMOTED", field)
        uncertain = copy.deepcopy(reports)
        uncertain[-1]["incremental"]["first_barge"]["eligible_sha256"] = None
        uncertain[-1]["incremental"]["first_barge"]["final_matches_eligible"] = None
        uncertain[-1]["first_barge"]["timings_ms"]["replay_inclusive_barge_to_new_audio"] = 900.0
        failed = scorecard.compute_scorecard(
            uncertain, preregistration_sha256="p" * 64, controls=VALID_CONTROLS,
            candidate_preregistration_sha256="c" * 64)
        self.assertEqual(failed["promotion_status"], "NOT_PROMOTED")

    def test_independent_eligibility_adversarial_rules(self):
        for text in ("停一下，直接讲土星环。", "停一下,直接讲土星环!", "换成英文讲。"):
            self.assertTrue(verifier.normalized_complete_command(text))
        for text in ("停一下，直接讲土。", "停一下，直接讲土星。", "换成英文讲",
                     "不要换成英文讲。", "换成英文讲，但是继续讲中文。", "换成英文\n讲。",
                     "请换成英文讲。", "换成英文讲�。"):
            self.assertFalse(verifier.normalized_complete_command(text), text)

    def test_quantiles_and_outliers_are_retained(self):
        summary = scorecard.summarize([1.0, 2.0, 3.0, 100.0])
        self.assertEqual(summary["max_ms"], 100.0)
        self.assertEqual(summary["outliers_ms"], [100.0])
        self.assertEqual(summary["samples_ms"], [1.0, 2.0, 3.0, 100.0])


if __name__ == "__main__":
    unittest.main()
