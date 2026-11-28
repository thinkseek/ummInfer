#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns02_fidelity_speedup_scorecard as scorecard


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(scorecard.canonical(value) + b"\n")


def fixture(root, *, fail=False):
    shapes = scorecard.EXPECTED_CASES
    pairs = []
    runs = []
    for repetition in range(5):
        for shape, geometry in shapes.items():
            after = 5001.0 if fail and shape == "tall" and repetition == 0 else 4000.0
            exact = not (fail and shape == "tall" and repetition == 0)
            ssim = 0.894 if fail and shape == "tall" and repetition == 0 else 0.95
            delta = after - 11000.0
            quality_pass = exact and ssim >= 0.9 and after > 0
            pairs.append({
                "shape": shape, "repetition": repetition,
                "baseline_ms": 11000.0, "candidate_ms": after,
                "candidate_minus_baseline_ms": delta,
                "baseline_sha256": "a" * 64, "candidate_sha256": "b" * 64,
                "quality": {"first_frame_rgb_exact": exact,
                            "first_frame_rgb_sha256_before": "c" * 64,
                            "first_frame_rgb_sha256_candidate": "d" * 64,
                            "ssim": {"all": ssim}, "all_frame_psnr_db": 30.0,
                            "temporal_motion_ratio": 1.0,
                            "candidate_first_last_mae": 5.0,
                            "quality_gate_pass": quality_pass},
            })
            for arm, milliseconds, media_hash in (
                    ("baseline", 11000.0, "a" * 64),
                    ("candidate", after, "b" * 64)):
                runs.append({"role": "measured", "shape": shape, "arm": arm,
                             "repetition": repetition, "first_frame_mae": 2.0,
                             "public_complete_artifact_ms": milliseconds,
                             "source_sha256": hashlib.sha256(shape.encode()).hexdigest(),
                             "media_sha256": media_hash,
                             "video_stream": {"codec_name": "h264",
                                              "height": geometry["target"][1],
                                              "width": geometry["target"][0],
                                              "nb_frames": "29", "pix_fmt": "yuv420p",
                                              "r_frame_rate": "16/1"}})
    controls = []
    for arm in ("baseline", "candidate"):
        controls.append({"role": "malformed_dimension", "arm": arm,
                         "http_status": 500, "native_success": False})
        for role in ("post_cancel_recovery", "post_failure_recovery"):
            controls.append({"role": role, "arm": arm, "http_status": 200,
                             "media_sha256": "a" * 64 if arm == "baseline" else "b" * 64})
    cancellations = [{"arm": arm, "disconnect_to_native_terminal_ms": 1.0,
                      "disconnect_to_client_settled_ms": 2.0,
                      "late_native_records_after_terminal": 0, "public_media_bytes": 0,
                      "health": {"status": "ok", "in_flight": 0},
                      "public_abort_ack": "NOT_APPLICABLE"}
                     for arm in ("baseline", "candidate")]
    promoted = not fail
    analysis = {"candidate_id": scorecard.EXPECTED_CANDIDATE,
                "candidate_commit": "b" * 40,
                "candidate_policy": {"protected_step_indices": [0, 1, 18, 19],
                                     "cacheable_step_indices": list(range(2, 18))},
                "paired": pairs, "promotion_decision": "PROMOTED" if promoted else "NOT_PROMOTED"}
    report = {"mode": "ns02", "physical_gpu": 7, "base_commit": "c" * 40,
              "execution_commit": "d" * 40, "candidate_analysis": analysis,
              "runs": runs, "controls": controls, "cancellations": cancellations,
              "sources": {shape: {"source_sha256": hashlib.sha256(shape.encode()).hexdigest()}
                          for shape in shapes},
              "final_health": {"http_status": 200, "body": {"status": "ok", "in_flight": 0}},
              "provenance": {key: {"sha256": "e" * 64, "bytes": 1}
                             for key in ("server_binary", "baseline_wan_binary",
                                         "candidate_wan_binary")},
              "models": {"wan22": {"tree_sha256": "f" * 64}}}
    write_json(root / "report.json", report)
    (root / "SHA256SUMS").write_text("external manifest fixture\n")
    write_json(root / "preregistration.json", {"frozen": True})
    write_json(root / "baseline-reference" / "report.json", {"mode": "profile"})
    (root / "baseline-reference" / "SHA256SUMS").write_text("profile manifest\n")
    verification = {"status": "PASS", "mode": "ns02",
                    "report_sha256": scorecard.sha_file(root / "report.json"),
                    "manifest_sha256": scorecard.sha_file(root / "SHA256SUMS")}
    write_json(root / "verification.json", verification)


class ScorecardTests(unittest.TestCase):
    def test_summary_retains_outlier_and_extrema(self):
        result = scorecard.summary([1.0, 2.0, 3.0, 100.0])
        self.assertEqual(result["max_ms"], 100.0)
        self.assertEqual(result["iqr_outliers_ms"], [100.0])
        self.assertEqual(result["count"], 4)

    def test_promoted_fixture_and_all_failure_classes(self):
        with tempfile.TemporaryDirectory(prefix="umm-ns02-scorecard-") as directory:
            root = Path(directory)
            fixture(root)
            result = scorecard.build_scorecard(root, root / "verification.json")
            self.assertEqual(result["promotion_decision"], "PROMOTED")
            self.assertEqual(result["failures"], [])
            self.assertTrue(all(item["all_after_below_5000_ms"]
                                for item in result["per_shape"].values()))

            fixture(root, fail=True)
            result = scorecard.build_scorecard(root, root / "verification.json")
            self.assertEqual(result["promotion_decision"], "NOT_PROMOTED")
            codes = {item["code"] for item in result["failures"]}
            self.assertIn("EXACT_FIRST_FRAME_RGB", codes)
            self.assertIn("SSIM_BELOW_0.90", codes)
            self.assertIn("MEASURED_ARTIFACT_AT_OR_ABOVE_5000_MS", codes)

    def test_verification_hash_binding_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="umm-ns02-scorecard-") as directory:
            root = Path(directory)
            fixture(root)
            verification = json.loads((root / "verification.json").read_text())
            verification["manifest_sha256"] = "0" * 64
            write_json(root / "verification.json", verification)
            with self.assertRaises(AssertionError):
                scorecard.build_scorecard(root, root / "verification.json")


if __name__ == "__main__":
    unittest.main()
