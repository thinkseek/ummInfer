#!/usr/bin/env python3
"""Deterministic CPU tests for the long-video temporal fixture and live contract.

All endpoint responses, native logs, GPU samples, and latency values in
FakeHarness are synthetic contract evidence only. No model inference, GPU,
remote host, deployment, or real SM90 run occurs. Real generation, live
latency, and perceptual quality remain NOT_RUN.
"""

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import umminfer_long_video_fixture as fixture  # noqa: E402
import umminfer_long_video_live as live  # noqa: E402
import video_temporal_live as short  # noqa: E402

BUNDLE = ROOT / "benchmarks" / "video-temporal-long"


def tree_bytes(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in Path(root).rglob("*") if path.is_file()}


class FixtureGenerationTests(unittest.TestCase):
    def test_tracked_bundle_deep_oracle_and_reproducible_generation(self):
        manifest = live.load_manifest(BUNDLE / "manifest.json", deep=True)
        self.assertEqual(len(manifest["cases"]), 4)
        with tempfile.TemporaryDirectory(prefix="umm-long-fixture-") as tmp:
            generated = Path(tmp) / "generated"
            fixture.generate_bundle(generated)
            fixture.validate_bundle(generated, deep=True)
            self.assertEqual(tree_bytes(generated), tree_bytes(BUNDLE))

    def test_exact_source_content_order_and_anchors(self):
        manifest = live.load_manifest(BUNDLE / "manifest.json", deep=False)
        for case in manifest["cases"]:
            with self.subTest(case=case["id"]):
                self.assertTrue(case["decode_order"]["decode_order_pts_nonmonotonic"])
                self.assertTrue(case["decode_order"]["decode_order_dts_monotonic"])
                self.assertEqual(case["ffprobe"]["streams"][0]["codec_name"], "h264")
                self.assertEqual(case["expected_uniform_8"]["source_frame_indices"],
                                 [3, 11, 19, 27, 35, 43, 51, 59])
                for budget in fixture.CPU_BUDGETS:
                    selected = case["expected_temporal"][str(budget)]
                    self.assertEqual(len(selected["candidate_indices"]), budget)
                    self.assertEqual(selected["source_frame_indices"][0], 0)
                    self.assertEqual(selected["source_frame_indices"][-1], 448)
                    self.assertEqual(selected["candidate_indices"],
                                     sorted(selected["candidate_indices"]))
                    content = (BUNDLE / case["content_frames"]).read_text().splitlines()
                    self.assertEqual(len(content), fixture.FRAME_COUNT)
                    rows = [json.loads(line) for line in content]
                    for start, end in case["events"]["exact_static_intervals"]:
                        self.assertEqual(len({row["source_rgb96x64_sha256"]
                                              for row in rows[start:end + 1]}), 1,
                                         f"non-static interval {case['id']}:{start}-{end}")
                    first = rows[0]
                    last = rows[-1]
                    self.assertEqual(first["source_frame_index"], 0)
                    self.assertEqual(last["source_frame_index"], 448)
                    self.assertEqual(first["source_timestamp_seconds"], 0.0)
                    self.assertEqual(last["source_timestamp_seconds"], 224.0)

    def test_fixture_checksum_and_missing_artifact_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix="umm-long-tamper-") as tmp:
            copied = Path(tmp) / "bundle"
            shutil.copytree(BUNDLE, copied)
            video = copied / "fixtures" / "scene-static.mp4"
            video.write_bytes(video.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "checksum"):
                fixture.validate_bundle(copied, deep=False)
            video.unlink()
            with self.assertRaises(ValueError):
                fixture.validate_bundle(copied, deep=False)


class IndependentSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = live.load_manifest(BUNDLE / "manifest.json", deep=False)

    def test_long_expected_indices_are_independently_pinned(self):
        expected = {
            "separated-motion": {
                4: [0, 320, 336, 448],
                8: [0, 64, 176, 184, 320, 328, 336, 448],
                16: [0, 32, 56, 64, 80, 88, 120, 152, 176, 184, 208, 240,
                     320, 328, 336, 448],
            },
            "scene-static": {
                4: [0, 224, 336, 448],
                8: [0, 64, 112, 128, 192, 224, 336, 448],
                16: [0, 32, 56, 88, 112, 120, 152, 176, 208, 224, 240, 272,
                     296, 328, 336, 448],
            },
        }
        for case in self.manifest["cases"][:2]:
            for budget, indices in expected[case["id"]].items():
                self.assertEqual(case["expected_temporal"][str(budget)]["source_frame_indices"],
                                 indices)

    def test_static_duplicates_zero_threshold_and_budget_16(self):
        scores = [(0.0, 0.0)] * 57
        for budget in (4, 8, 16):
            selected = fixture.select_temporal(scores, budget, motion_threshold=0.0)
            self.assertEqual(len(selected), budget)
            self.assertEqual(selected[0], 0)
            self.assertEqual(selected[-1], 56)
            self.assertEqual(selected, fixture.select_temporal(scores, budget))
        with self.assertRaises(ValueError):
            fixture.select_temporal(scores, 0)
        with self.assertRaises(ValueError):
            live.validate_budgets([4, 8, 16])

    def test_multilingual_answer_expectations_do_not_make_quality_claims(self):
        by_id = {case["id"]: case for case in self.manifest["cases"]}
        self.assertTrue(live.answer_hit("红色", by_id["separated-motion"]) is False)
        self.assertTrue(live.answer_hit("Red.", by_id["separated-motion"]))
        self.assertTrue(live.answer_hit("蓝色", by_id["scene-static"]))
        self.assertTrue(live.answer_hit("マゼンタ", by_id["near-duplicate"]))
        self.assertFalse(live.answer_hit("blanco", by_id["bframe-anchors"]))
        self.assertTrue(live.answer_hit("blanco y negro", by_id["bframe-anchors"]))


class FakeHarness:
    """Synthetic-only transport/native/device evidence; never real inference."""

    def __init__(self, *, warmup=0, measured=1, prefix_reused=0,
                 diverge_selection=False, diverge_mrope=False, fail_request=False):
        self.manifest = live.prepare_cases(
            live.load_manifest(BUNDLE / "manifest.json", deep=False), BUNDLE, short)
        self.manifest["measurement"]["warmup_repetitions"] = warmup
        self.manifest["measurement"]["measured_repetitions"] = measured
        self.prefix_reused = prefix_reused
        self.diverge_selection = diverge_selection
        self.diverge_mrope = diverge_mrope
        self.fail_request = fail_request
        self.native = {mode: {} for mode in live.MODES}
        self.execution = {mode: {} for mode in live.MODES}
        self.counters = {mode: 0 for mode in live.MODES}
        self.provenances = {}
        for index, mode in enumerate(live.MODES):
            self.provenances[mode] = {
                "schema_version": 1, "stack_id": f"synthetic-{mode}",
                "stack_run_id": "synthetic-contract-only",
                "source_commit": "0" * 40, "source_state_sha256": "0" * 64,
                "physical_gpu": (0, 1)[index], "batch_size": (1, 4)[index],
                "qwen38_prefix_cache": "disabled", "max_residual_memory_mib": 128,
                "gateway": {"pid": 1000 + index, "sha256": "1" * 64},
                "native_stage": {"pid": 2000 + index, "sha256": "2" * 64},
                "model": {"id": "synthetic-not-a-model", "revision": "NOT_A_MODEL",
                          "files": {"contract-only": "3" * 64}},
            }

    def native_record(self, request_id, case, arm, mode, serial_number):
        expected = case["expected_uniform_8"] if arm != "temporal" \
            else case["expected_temporal"]["8"]
        temporal = arm == "temporal"
        if temporal:
            timestamps = expected["source_timestamps_seconds"]
            pairs = expected["temporal2_pair_first_timestamps"]
            candidates = expected["candidate_indices"]
            sources = expected["source_frame_indices"]
        else:
            duration = case["duration_seconds"]
            timestamps = [duration * index / 8 for index in range(8)]
            pairs = [timestamps[min(2 * index, 7)] for index in range(4)]
            candidates = list(range(8))
            sources = list(range(8))
        identity = "uniform" if arm != "temporal" else "temporal"
        mrope = fixture.sha256_bytes(f"{case['id']}:{identity}".encode())[:16]
        if self.diverge_mrope and arm == "default" and mode == "serial":
            mrope = "f" * 16
        content = list(expected["content_fnv1a64"])
        if self.diverge_selection and arm == "temporal" and mode == "serial":
            content[0] = "0" * 16
        return {
            "request_id": request_id + ":vlm", "policy": identity,
            "engine_path": "serial" if mode == "serial" else "batched",
            "decoded_frame_count": case["candidate_count"] if temporal else 8,
            "selected_frame_count": 8, "selected_candidate_indices": candidates,
            "selected_source_frame_indices": sources,
            "source_indices_exact": temporal,
            "selected_frame_timestamps": timestamps,
            "selected_frame_content_fnv1a64": content,
            "has_nominal_timestamps": not temporal,
            "temporal2_pair_first_timestamps": pairs,
            "max_frames": 8, "max_candidate_frames": 64,
            "motion_threshold": fixture.MOTION_THRESHOLD,
            "scene_threshold": fixture.SCENE_THRESHOLD,
            "grid_t": 4, "grid_h": 10, "grid_w": 14,
            "premerge_patch_rows": 560, "merged_tokens_per_slice": 35,
            "video_token_count": 140, "prompt_token_count": 200,
            "media_hash": fixture.sha256_bytes(
                f"{case['id']}:{identity}:media".encode())[:16],
            "mrope_next": 260, "mrope_fnv1a64": mrope,
        }

    def transport(self, mode):
        def call(body, timeout):
            del timeout
            serial = self.counters[mode]
            self.counters[mode] += 1
            if self.fail_request and mode == "batch4" and serial == 0:
                raise RuntimeError("synthetic injected request failure")
            video = body["messages"][0]["content"][1]["video_url"]
            selection = video.get("frame_selection")
            arm = selection["policy"] if selection else "default"
            digest = body["messages"][0]["content"][1]["video_url"]["url"]
            case = next(case for case in self.manifest["cases"]
                        if case["_data_url"] == digest)
            request_id = f"synthetic-{mode}-{serial}"
            self.native[mode][request_id + ":vlm"] = self.native_record(
                request_id, case, arm, mode, serial)
            self.execution[mode][request_id + ":vlm"] = {
                "request_id": request_id + ":vlm", "prompt_tokens": 200,
                "prefill_tokens": 200, "prefix_reused": self.prefix_reused
                    if serial == 0 else 0,
                "image_tokens": 140, "generated_tokens": 4,
                "elapsed_ms": 50 + serial, "vision_ms": 10.0 + serial % 3,
                "prefill_ms": 20.0 + serial % 5, "decode_ms": 15.0,
                "decode_tok_per_s": 50.0,
            }
            answers = {"separated-motion": "red", "scene-static": "蓝色",
                       "near-duplicate": "magenta", "bframe-anchors": "blanco y negro"}
            answer = answers[case["id"]] if arm == "temporal" else "synthetic baseline"
            response = {"request_id": request_id, "http_status": 200, "answer": answer,
                        "answer_sha256": fixture.sha256_bytes(answer.encode()),
                        "first_event_s": 0.005 + serial * 0.0001,
                        "ttft_s": 0.010 + serial * 0.0001,
                        "total_s": 0.050 + serial * 0.0001,
                        "finish_reason": "stop", "request_bytes": 1000}
            frames = [{"synthetic_contract_only": True, "id": request_id,
                       "choices": [{"delta": {"content": answer}, "finish_reason": "stop"}]}]
            return response, frames
        return call

    def run(self, out, *, exclusive_pass=True):
        Path(out).mkdir(parents=True)
        return live.execute_synthetic(
            self.manifest,
            transports={mode: self.transport(mode) for mode in live.MODES},
            native_logs=self.native, execution_logs=self.execution,
            health_probes={mode: lambda: {"in_flight": 0} for mode in live.MODES},
            memory_samplers={mode: lambda indexes: {index: 100 for index in indexes}
                             for mode in live.MODES},
            provenances=self.provenances,
            exclusive=[{"mode": mode, "check": "exclusive_device",
                        "passed": exclusive_pass, "problems": []} for mode in live.MODES],
            out_dir=out, short=short)


class ControlledContractTests(unittest.TestCase):
    def test_seeded_schedule_aggregates_and_fake_bundle_recompute(self):
        with tempfile.TemporaryDirectory(prefix="umm-long-contract-") as tmp:
            harness = FakeHarness(warmup=1, measured=2)
            report = harness.run(Path(tmp) / "evidence")
            self.assertTrue(report["gate_pass"], report["gate_reasons"])
            self.assertEqual(report["evidence_class"], live.SYNTHETIC_CLASS)
            self.assertEqual(report["claims"], {
                "real_generation": "NOT_RUN", "live_latency": "NOT_RUN",
                "perceptual_quality": "NOT_RUN", "generalized_quality_claim": False})
            self.assertEqual(len(report["records"]), 72)
            self.assertEqual(len(report["aggregates"]), 24)
            self.assertTrue(all(row["declared_repetitions"] == 2
                                and row["native_valid_repetitions"] == 2
                                for row in report["aggregates"]))
            for mode in live.MODES:
                steps = [step for step in report["steps"] if step["mode"] == mode]
                self.assertEqual([step["phase"] for step in steps[:3]], ["warmup"] * 3)
                measured = [step for step in steps if step["phase"] == "measured"]
                for offset in range(0, len(measured), 3):
                    block = measured[offset:offset + 3]
                    self.assertEqual({step["arm"] for step in block}, set(live.ARMS))
                    self.assertEqual(len({step["repetition"] for step in block}), 1)
            self.assertTrue(all(row["passed"] for row in report["parity"]))
            validated = live.validate_bundle(Path(tmp) / "evidence")
            self.assertEqual(validated["aggregates"], report["aggregates"])

    def test_fail_closed_cold_prefix_device_selection_mrope_and_requests(self):
        cases = (
            ("prefix", {"prefix_reused": 9}, True, "native evidence"),
            ("device", {}, False, "exclusive_device"),
            ("selection", {"diverge_selection": True}, True, "native evidence"),
            ("mrope", {"diverge_mrope": True}, True, "parity"),
            ("request", {"fail_request": True}, True, "requests failed"),
        )
        for name, kwargs, exclusive_pass, reason in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory(
                    prefix=f"umm-long-{name}-") as tmp:
                report = FakeHarness(**kwargs).run(Path(tmp) / "evidence",
                                                   exclusive_pass=exclusive_pass)
                self.assertFalse(report["gate_pass"])
                self.assertTrue(any(reason in text for text in report["gate_reasons"]),
                                report["gate_reasons"])
                self.assertEqual(report["evidence_class"], live.SYNTHETIC_CLASS)
                self.assertEqual(report["claims"]["live_latency"], "NOT_RUN")
                self.assertTrue(live.validate_bundle(Path(tmp) / "evidence"))

    def test_incomplete_repetition_gate(self):
        with tempfile.TemporaryDirectory(prefix="umm-long-incomplete-") as tmp:
            harness = FakeHarness(measured=1)
            out = Path(tmp) / "evidence"
            out.mkdir()
            result = live.execute_schedule(
                harness.manifest,
                transports={mode: harness.transport(mode) for mode in live.MODES},
                native_logs=harness.native, execution_logs=harness.execution,
                health_probes={mode: lambda: {"in_flight": 0} for mode in live.MODES},
                memory_samplers={mode: lambda indexes: {index: 100 for index in indexes}
                                 for mode in live.MODES},
                provenances=harness.provenances, out_dir=out, short=short, synthetic=True)
            removed = result["records"].pop()
            (out / "records" / removed["mode"] / removed["phase"] /
             f"repetition-{removed['repetition']}" / removed["arm"] /
             f"{removed['clip']}.json").unlink()
            provenances = dict(harness.provenances, _short=short)
            report = live.finish_report(
                harness.manifest, result, provenances=provenances,
                exclusive=[{"mode": mode, "passed": True, "problems": []}
                           for mode in live.MODES],
                evidence_class=live.SYNTHETIC_CLASS)
            self.assertFalse(report["gate_pass"])
            self.assertTrue(any("incomplete" in reason for reason in report["gate_reasons"]))

    def test_artifact_alteration_missing_file_and_aggregate_tamper_fail(self):
        with tempfile.TemporaryDirectory(prefix="umm-long-artifacts-") as tmp:
            root = Path(tmp) / "evidence"
            FakeHarness().run(root)
            frames = next(root.rglob("*.sse.json"))
            original = frames.read_bytes()
            frames.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "checksum|artifact"):
                live.validate_bundle(root)
            frames.write_bytes(original)
            live.validate_bundle(root)
            victim = next(root.rglob("*.sse.json"))
            victim.unlink()
            with self.assertRaises(ValueError):
                live.validate_bundle(root)

        with tempfile.TemporaryDirectory(prefix="umm-long-aggregate-") as tmp:
            root = Path(tmp) / "evidence"
            FakeHarness().run(root)
            report_path = root / "report.json"
            report = json.loads(report_path.read_text())
            report["aggregates"][0]["latency_ms"]["ttft"]["median_ms"] += 1.0
            report_path.write_text(json.dumps(report))
            # Reseal raw file hashes to prove aggregate recomputation itself is
            # independent, not merely a checksum comparison.
            artifact_path = root / "artifact-manifest.json"
            artifact = json.loads(artifact_path.read_text())
            for row in artifact["files"]:
                path = root / row["path"]
                row["bytes"] = path.stat().st_size
                row["sha256"] = live.sha256_file(path)
            artifact_path.write_text(json.dumps(artifact))
            sums = []
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.name != "SHA256SUMS":
                    sums.append(f"{live.sha256_file(path)}  "
                                f"{path.relative_to(root).as_posix()}\n")
            (root / "SHA256SUMS").write_text("".join(sums))
            with self.assertRaisesRegex(ValueError, "aggregate"):
                live.validate_bundle(root)


class ProvenanceGateTests(unittest.TestCase):
    def provenance(self, tmp):
        root = Path(tmp)
        binary = root / "gateway.bin"
        stage = root / "stage.bin"
        model = root / "model"
        model.mkdir()
        binary.write_bytes(b"synthetic gateway")
        stage.write_bytes(b"synthetic stage")
        (model / "weights.bin").write_bytes(b"synthetic weights")
        state = {"source_commit": "a" * 40, "source_state_sha256": "b" * 64,
                 "source_tree_clean": True}
        record = {"schema_version": 1, "stack_id": "new-isolated-stack",
                  "stack_run_id": "run-39", "source_commit": "a" * 40,
                  "source_state_sha256": "b" * 64, "physical_gpu": 7,
                  "batch_size": 1, "qwen38_prefix_cache": "disabled",
                  "max_residual_memory_mib": 128,
                  "gateway": {"path": str(binary), "sha256": live.sha256_file(binary),
                              "pid": os.getpid()},
                  "native_stage": {"path": str(stage), "sha256": live.sha256_file(stage),
                                   "pid": os.getpid()},
                  "model": {"id": "qwen3-native", "root": str(model),
                            "revision": "synthetic-revision",
                            "files": {"weights.bin": live.sha256_file(model / "weights.bin")}}}
        path = root / "provenance.json"
        path.write_text(json.dumps(record))
        return path, record, state

    def test_stale_binary_model_source_and_disallowed_gpu_fail(self):
        with tempfile.TemporaryDirectory(prefix="umm-long-provenance-") as tmp:
            path, record, state = self.provenance(tmp)
            with mock.patch.object(live, "source_state", return_value=state):
                validated = live.validate_provenance(path, mode="serial", run_id="run-39",
                                                     source_root=ROOT,
                                                     process_digest=lambda pid, path:
                                                     live.sha256_file(path))
                self.assertEqual(validated["physical_gpu"], 7)
                Path(record["gateway"]["path"]).write_bytes(b"stale gateway")
                with self.assertRaisesRegex(ValueError, "stale.*gateway"):
                    live.validate_provenance(path, mode="serial", run_id="run-39",
                                             source_root=ROOT,
                                                     process_digest=lambda pid, path:
                                                     live.sha256_file(path))
                Path(record["gateway"]["path"]).write_bytes(b"synthetic gateway")
                (Path(record["model"]["root"]) / "weights.bin").write_bytes(b"stale")
                with self.assertRaisesRegex(ValueError, "stale model"):
                    live.validate_provenance(path, mode="serial", run_id="run-39",
                                             source_root=ROOT,
                                                     process_digest=lambda pid, path:
                                                     live.sha256_file(path))
            for key, value in (("physical_gpu", 2), ("qwen38_prefix_cache", "enabled"),
                               ("stack_run_id", "old-run")):
                changed = dict(record, **{key: value})
                path.write_text(json.dumps(changed))
                with mock.patch.object(live, "source_state", return_value=state), \
                        self.assertRaises(ValueError):
                    live.validate_provenance(path, mode="serial", run_id="run-39",
                                             source_root=ROOT, rehash=False)

    def test_exclusive_gate_requires_recorded_pid_and_no_foreign_process(self):
        provenance = {"physical_gpu": 0, "native_stage": {"pid": 42},
                      "max_residual_memory_mib": 128}
        record, _, _ = live.exclusive_check(
            provenance, lambda indexes: {index: [{"pid": 42, "used_mib": 100}]
                                         for index in indexes}, lambda indexes: {0: 100})
        self.assertTrue(record["passed"])
        foreign, _, _ = live.exclusive_check(
            provenance, lambda indexes: {index: [{"pid": 42, "used_mib": 100},
                                                 {"pid": 99, "used_mib": 10}]
                                         for index in indexes}, lambda indexes: {0: 100})
        self.assertFalse(foreign["passed"])


class CliBoundaryTests(unittest.TestCase):
    def test_self_check_and_default_not_run(self):
        for args in (["--self-check"], []):
            completed = subprocess.run(
                [sys.executable, "-I", str(ROOT / "tests" /
                                          "video_temporal_long_live.py"), *args],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertIn("REAL_GENERATION: NOT_RUN", completed.stdout)
            self.assertIn("LIVE_LATENCY: NOT_RUN", completed.stdout)
            self.assertIn("PERCEPTUAL_QUALITY: NOT_RUN", completed.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
