#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deterministic offline contracts for the I2V geometry/latency evidence tools."""

import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import i2v_geometry_latency_live as live
import i2v_geometry_latency_verify as verify
import ns_interactive_preview_scorecard as scorecard
import ns_interactive_preview_verify as final_verify


class MessagePackTests(unittest.TestCase):
    def test_native_wire_scalars_binary_and_nested_containers(self):
        # {"ok":true,"n":-2} and {"bin":00ff,"text":"图","items":[1,512]}
        first = bytes.fromhex("82a26f6bc3a16efe")
        text = "图".encode("utf-8")
        second = (b"\x83\xa3bin\xc4\x02\x00\xff\xa4text"
                  + bytes([0xa0 | len(text)]) + text
                  + b"\xa5items\x92\x01\xcd\x02\x00")
        self.assertEqual(verify.unpack_msgpack(first), {"ok": True, "n": -2})
        self.assertEqual(verify.unpack_msgpack(second),
                         {"bin": b"\x00\xff", "text": "图", "items": [1, 512]})

    def test_truncated_trailing_and_extension_frames_fail_closed(self):
        for raw in (b"\xc4\x02\x00", b"\x01\x00", b"\xd4\x00\x00"):
            with self.subTest(raw=raw), self.assertRaises((AssertionError, ValueError)):
                verify.unpack_msgpack(raw)


class PreregistrationTests(unittest.TestCase):
    def test_schedule_and_statistics_are_frozen_and_retain_outliers(self):
        self.assertEqual(live.candidate_schedule(), verify.expected_schedule())
        summary = verify.expected_summary([1.0, 2.0, 3.0, 100.0])
        self.assertEqual(summary["iqr_outliers_ms"], [100.0])
        self.assertEqual(summary["all_ms"], [1.0, 2.0, 3.0, 100.0])
        self.assertEqual(summary["count"], 4)

    def test_changed_profile_or_sampler_gate_is_rejected(self):
        document = live.preregistration_document("a" * 64, "test-candidate", "contract")
        verify.verify_preregistration(document, "a" * 64)
        for mutate in (
                lambda item: item.update(profile_report_sha256="b" * 64),
                lambda item: item["settings"].update(steps=19),
                lambda item: item["regression_gates"].update(first_frame_mae_max=60.0)):
            changed = json.loads(json.dumps(document))
            mutate(changed)
            with self.subTest(mutate=mutate), self.assertRaises(AssertionError):
                verify.verify_preregistration(changed, "a" * 64)
    def test_ns02_protocol_rejects_threshold_sampler_or_gate_drift(self):
        path = (Path(__file__).resolve().parent.parent / "benchmarks" /
                "ns-interactive-preview" / "preregistration.json")
        document = json.loads(path.read_text())
        verify.verify_ns02_preregistration(document, document["profile_report_sha256"])
        self.assertEqual(live.candidate_schedule("ns02"), verify.expected_schedule("ns02"))
        self.assertEqual(len(live.candidate_schedule("ns02")), 10)
        for mutate in (
                lambda item: item.update(candidate_commit="0" * 40),
                lambda item: item["settings"].update(steps=19),
                lambda item: item["settings"].update(candidate_effective_teacache=0.2),
                lambda item: item["regression_gates"].update(minimum_ssim_all=0.5),
                lambda item: item.update(repetitions=3)):
            changed = json.loads(json.dumps(document))
            mutate(changed)
            with self.subTest(mutate=mutate), self.assertRaises(AssertionError):
                verify.verify_ns02_preregistration(changed, document["profile_report_sha256"])


class MediaMetricTests(unittest.TestCase):
    def test_ffmpeg_ssim_all_layout_is_parsed_by_producer_and_verifier(self):
        import subprocess
        log = ("[Parsed_ssim_0 @ x] SSIM Y:0.892661 (9.692430) "
               "U:0.963254 (14.347930) V:0.942325 (12.390132) "
               "All:0.912704 (10.590055)\n")
        completed = subprocess.CompletedProcess([], 0, "", log)
        with mock.patch.object(live.subprocess, "run", return_value=completed):
            self.assertEqual(live.ssim_all("ffmpeg", "before.mp4", "after.mp4")["all"],
                             0.912704)
        with mock.patch.object(verify.subprocess, "run", return_value=completed):
            self.assertEqual(verify.ns02_ssim("ffmpeg", "before.mp4", "after.mp4")["all"],
                             0.912704)


class ScorecardContractTests(unittest.TestCase):
    def test_independent_statistics_match_and_thresholds_fail_closed(self):
        values = [1.0, 2.0, 3.0, 100.0]
        self.assertEqual(final_verify.summary(values), live.summarize(values))
        self.assertFalse(scorecard.threshold(values, 50.0)["p95_pass"])
        self.assertTrue(scorecard.threshold([1.0, 2.0, 3.0], 4.0)["all_runs_pass"])
        with self.assertRaises(AssertionError):
            final_verify.close(1.0, 1.1, "deliberate mismatch")

    def test_final_manifest_covers_nested_manifests_and_rejects_extras(self):
        with tempfile.TemporaryDirectory(prefix="umm-ns-final-manifest-") as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "nested" / "SHA256SUMS").write_text("nested evidence\n")
            (root / "scorecard.json").write_text("{}\n")
            rows = []
            for path in (root / "scorecard.json", root / "nested" / "SHA256SUMS"):
                rows.append(f"{final_verify.sha_file(path)}  "
                            f"{path.relative_to(root).as_posix()}\n")
            (root / "SHA256SUMS").write_text("".join(sorted(rows)))
            self.assertEqual(final_verify.manifest(root)[0], 2)
            (root / "nested" / "extra").write_text("not sealed")
            with self.assertRaises(AssertionError):
                final_verify.manifest(root)


class VerifierIdentityTests(unittest.TestCase):
    def test_current_and_explicit_historical_verifier_identities(self):
        recorded = {"bytes": 123, "sha256": "a" * 64}
        current = {"bytes": 456, "sha256": "b" * 64}
        verify.check_verifier_identity(recorded, "a" * 64, current)
        with self.assertRaises(AssertionError):
            verify.check_verifier_identity(recorded, None, current)
        with self.assertRaises(AssertionError):
            verify.check_verifier_identity(recorded, "c" * 64, current)
        verify.check_verifier_identity(current, None, current)


class ManifestTests(unittest.TestCase):
    def test_complete_self_excluded_manifest_detects_extra_and_tamper(self):
        with tempfile.TemporaryDirectory(prefix="umm-i2v-manifest-test-") as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "a.bin").write_bytes(b"a\x00\xff")
            (root / "nested" / "b.json").write_text("{}\n")
            rows = []
            for path in (root / "a.bin", root / "nested" / "b.json"):
                rows.append(f"{verify.sha256_file(path)['sha256']}  "
                            f"{path.relative_to(root).as_posix()}\n")
            (root / "SHA256SUMS").write_text("".join(sorted(rows)))
            self.assertEqual(verify.check_manifest(root)["file_count"], 2)
            (root / "extra").write_bytes(b"not sealed")
            with self.assertRaises(AssertionError):
                verify.check_manifest(root)
            (root / "extra").unlink()
            (root / "a.bin").write_bytes(b"tampered")
            with self.assertRaises(AssertionError):
                verify.check_manifest(root)


class ResourceContractTests(unittest.TestCase):
    def test_post_exit_residual_utilization_does_not_relax_initial_admission(self):
        residual = {"index": 1, "memory_used_mib": 3, "utilization_percent": 4,
                    "compute_processes": []}
        with mock.patch.object(live, "gpu_state", return_value=residual):
            self.assertEqual(live.require_gpu_idle(
                "nvidia-smi", 1, after_owned_stage=True), residual)
            with self.assertRaises(AssertionError):
                live.require_gpu_idle("nvidia-smi", 1)
        occupied = dict(residual, compute_processes=[{"pid": 123, "used_gpu_memory_mib": 3}])
        with mock.patch.object(live, "gpu_state", return_value=occupied):
            with self.assertRaises(AssertionError):
                live.require_gpu_idle("nvidia-smi", 1, after_owned_stage=True)


class GeometryContractTests(unittest.TestCase):
    def test_png_and_exact_source_url_binding(self):
        raw = (b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR"
               + struct.pack(">II", 480, 704))
        self.assertEqual(verify.png_dimensions(raw), [480, 704])
        source = b"source\x00\xff"
        encoded = live.source_data_url(source)
        self.assertEqual(verify.decode_data_url(encoded), source)
        self.assertEqual(live.CASES["wide"]["source"], (704, 480))
        self.assertEqual(live.CASES["wide"]["target"], (704, 480))
        self.assertEqual(live.CASES["tall"]["source"], (480, 704))
        self.assertEqual(live.CASES["tall"]["target"], (480, 704))
        self.assertEqual(live.CASES["square"]["source"], (512, 512))
        self.assertEqual(live.CASES["square"]["target"], (704, 480))


if __name__ == "__main__":
    unittest.main()
