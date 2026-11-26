#!/usr/bin/env python3
"""Deterministic CPU/source-pixel tests for temporal-clean-r1; no GPU or model."""

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

import umminfer_long_video_fixture as original_fixture  # noqa: E402
import umminfer_long_video_live as long_live  # noqa: E402
import umminfer_temporal_clean_fixture as fixture  # noqa: E402
import umminfer_temporal_clean_live as clean  # noqa: E402
import video_temporal_live as short  # noqa: E402

BUNDLE = fixture.DEFAULT_BUNDLE
FREEZE = fixture.DEFAULT_FREEZE
ORIGINAL_BUNDLE = ROOT / "benchmarks" / "video-temporal-long"


def tree_bytes(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in Path(root).rglob("*") if path.is_file()}


def reseal_tree(root):
    root = Path(root)
    files = sorted(path for path in root.rglob("*") if path.is_file()
                   and path.name != "SHA256SUMS")
    (root / "SHA256SUMS").write_text("".join(
        f"{fixture.sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
        for path in files), encoding="ascii")


class FrozenFixtureTests(unittest.TestCase):
    def test_deep_oracle_and_byte_reproducible_generation(self):
        manifest = clean.load_manifest(BUNDLE / "manifest.json", deep=True, freeze_path=FREEZE)
        self.assertEqual(len(manifest["videos"]), 4)
        self.assertEqual(len(manifest["cases"]), 8)
        with tempfile.TemporaryDirectory(prefix="umm-clean-generate-") as tmp:
            generated = Path(tmp) / "bundle"
            fixture.generate_bundle(generated, preregistration_path=BUNDLE / "preregistration.json")
            fixture.validate_bundle(generated, deep=True)
            self.assertEqual(tree_bytes(generated), tree_bytes(BUNDLE))

    def test_freeze_binds_preregistration_manifest_and_complete_checksums(self):
        receipt = fixture.validate_freeze(BUNDLE, FREEZE)
        self.assertTrue(receipt["frozen_before_any_model_response"])
        self.assertFalse(receipt["model_response_inspected"])
        self.assertEqual(len(receipt["quality_case_ids"]), 4)
        self.assertEqual(len(receipt["selection_miss_control_ids"]), 4)
        with tempfile.TemporaryDirectory(prefix="umm-clean-freeze-") as tmp:
            copied = Path(tmp) / "bundle"
            shutil.copytree(BUNDLE, copied)
            prereg = json.loads((copied / "preregistration.json").read_bytes())
            prereg["cases"][0]["question"] = "Answer triangle."
            (copied / "preregistration.json").write_text(json.dumps(prereg))
            reseal_tree(copied)
            with self.assertRaises(ValueError):
                fixture.validate_freeze(copied, FREEZE)

    def test_original_r4_fixture_and_scoring_are_unchanged(self):
        self.assertEqual(fixture.sha256_file(ORIGINAL_BUNDLE / "manifest.json"),
                         "642ce61e4983cd47d549b0315e0d6f92f01495f936e109487ea2407592a28980")
        original = original_fixture.validate_bundle(ORIGINAL_BUNDLE, deep=False)
        by_id = {case["id"]: case for case in original["cases"]}
        self.assertFalse(long_live.answer_hit("green", by_id["separated-motion"]))
        self.assertTrue(long_live.answer_hit("red", by_id["separated-motion"]))
        self.assertFalse(long_live.answer_hit("blanco", by_id["bframe-anchors"]))
        self.assertTrue(long_live.answer_hit("blanco y negro", by_id["bframe-anchors"]))


class IndependentSourcePixelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = fixture.validate_bundle(BUNDLE, deep=False)
        cls.prereg = json.loads((BUNDLE / "preregistration.json").read_bytes())

    def test_every_source_pixel_and_selected_decode_is_non_chromatic(self):
        for video_id in self.manifest["videos"]:
            for index in range(fixture.FRAME_COUNT):
                frame = fixture.source_frame(video_id, index)
                for offset in range(0, len(frame), 3):
                    self.assertEqual(frame[offset], frame[offset + 1])
                    self.assertEqual(frame[offset], frame[offset + 2])
        self.assertEqual(self.manifest["marker_neutrality"]["decoded_selected_max_channel_spread"], 0)
        for video in self.manifest["videos"].values():
            for selection in video["selections"].values():
                record = json.loads((BUNDLE / selection["record"]).read_bytes())
                self.assertEqual(len(record["decoded_native_rgb_sha256"]), 8)

    def test_binding_barcode_cannot_change_central_answer_geometry(self):
        for video_id in ("motion-clean", "near-clean", "bframe-clean"):
            left = fixture.source_frame(video_id, 3)
            right = fixture.source_frame(video_id, 11)
            for y in range(8, 54):
                for x in range(24, 72):
                    offset = (y * fixture.WIDTH + x) * 3
                    if video_id == "motion-clean":
                        continue  # the semantic triangle itself moves
                    self.assertEqual(left[offset:offset + 3], right[offset:offset + 3])
            # The machine barcode is restricted to x=2..17,y=55..56, outside
            # the x=24..71,y=8..53 semantic card used above.
            self.assertTrue(fixture.WIDTH >= 24)

    def test_independent_reference_geometry_is_not_a_source_crop(self):
        for marker in fixture.MARKERS:
            pixels = fixture.render_reference(marker)
            self.assertEqual(len(pixels), 1600)
            for y in range(40):
                for x in range(40):
                    expected = 24
                    if marker == "triangle":
                        # Independent barycentric oracle around (20,5),(5,34),(35,34).
                        ax, ay, bx, by, cx, cy = 20, 5, 5, 34, 35, 34
                        px, py = x + 0.5, y + 0.5
                        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
                        aa = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den
                        bb = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den
                        cc = 1.0 - aa - bb
                        if aa >= 0 and bb >= 0 and cc >= 0:
                            expected = 224
                    elif marker == "ring":
                        radius = ((x + 0.5 - 20) ** 2 + (y + 0.5 - 20) ** 2) ** 0.5
                        expected = 224 if 8 <= radius <= 15 else 24
                    elif marker == "cross":
                        if (16 <= x < 24 and 5 <= y < 35) or (5 <= x < 35 and 16 <= y < 24):
                            expected = 224
                    elif marker == "square":
                        expected = 224 if 7 <= x < 33 and 7 <= y < 33 else 24
                    elif marker == "stripes" and 5 <= x < 35 and 5 <= y < 35:
                        expected = 224 if ((x - 5) // 4) % 2 == 0 else 24
                    self.assertEqual(pixels[y * 40 + x], expected,
                                     f"{marker} reference differs at {x},{y}")

    def test_quality_anchors_and_selection_controls_are_independently_enforced(self):
        for case in self.manifest["cases"]:
            fixture.validate_anchor_rules(case, case["anchor_audit"], self.prereg)
            for arm in fixture.ARMS:
                audit = case["anchor_audit"][arm]
                for observation in audit["observations"]:
                    self.assertTrue(fixture.in_intervals(
                        observation["source_frame_index"], case["answer_anchor"]["source_intervals"]))
                    reference = self.manifest["anchor_references"][observation["marker"]]
                    self.assertEqual(observation["reference_anchor_sha256"], reference["rgb_sha256"])
            if case["evaluation_role"] == "quality_scored":
                self.assertTrue(all(case["anchor_audit"][arm]["answer_anchor_visible"]
                                    for arm in fixture.ARMS))
            else:
                self.assertFalse(case["anchor_audit"]["default"]["declared_anchor_source_frames_present"])
                self.assertFalse(case["anchor_audit"]["uniform"]["declared_anchor_source_frames_present"])
                self.assertTrue(case["anchor_audit"]["temporal"]["answer_anchor_visible"])

    def test_prompt_non_leakage_roles_and_question_after_context(self):
        for case in self.manifest["cases"]:
            folded = case["question"].casefold()
            self.assertTrue(all(term.casefold() not in folded for term in case["expect_all"]))
            body = clean.build_body_after_context(
                "data:video/mp4;base64,AA==", case["question"], "default", "qwen3-native",
                self.manifest["request_contract"])
            self.assertEqual([part["type"] for part in body["messages"][0]["content"]],
                             ["video_url", "text"])
            self.assertNotIn("Responde:", case["question"])
        self.assertFalse(clean.answer_hit("green", self.manifest["cases"][0]))
        self.assertTrue(clean.answer_hit("a triangle", self.manifest["cases"][0]))

    def test_static_duplicates_bframe_order_pts_and_first_last_anchors(self):
        for video_id, video in self.manifest["videos"].items():
            self.assertTrue(video["decode_order"]["decode_order_pts_nonmonotonic"])
            self.assertTrue(video["decode_order"]["decode_order_dts_monotonic"])
            self.assertEqual(video["expected_uniform_8"]["source_frame_indices"],
                             [3, 11, 19, 27, 35, 43, 51, 59])
            for budget in fixture.CPU_BUDGETS:
                selected = video["expected_temporal"][str(budget)]
                self.assertEqual(selected["source_frame_indices"][0], 0)
                self.assertEqual(selected["source_frame_indices"][-1], 448)
                self.assertEqual(selected["source_timestamps_seconds"],
                                 sorted(selected["source_timestamps_seconds"]))
            rows = [json.loads(line) for line in
                    (BUNDLE / video["content_frames"]).read_text().splitlines()]
            for start, end in video["events"]["exact_static_intervals"]:
                self.assertEqual(len({row["source_rgb96x64_sha256"] for row in rows[start:end + 1]}), 1,
                                 f"non-static interval {video_id}:{start}-{end}")


class FakeHarness:
    """Synthetic-only transport/native evidence; never real model inference."""

    def __init__(self, *, warmup=0, measured=1, prefix_reused=0,
                 diverge_selection=False, diverge_mrope=False, fail_request=False):
        self.manifest = long_live.prepare_cases(
            clean.load_manifest(BUNDLE / "manifest.json", deep=False, freeze_path=FREEZE),
            BUNDLE, short)
        self.manifest["measurement"]["warmup_repetitions"] = warmup
        self.manifest["measurement"]["measured_repetitions"] = measured
        self.prefix_reused = prefix_reused
        self.diverge_selection = diverge_selection
        self.diverge_mrope = diverge_mrope
        self.fail_request = fail_request
        self.native = {mode: {} for mode in clean.MODES}
        self.execution = {mode: {} for mode in clean.MODES}
        self.counters = {mode: 0 for mode in clean.MODES}
        self.provenances = {}
        for index, mode in enumerate(clean.MODES):
            self.provenances[mode] = {
                "schema_version": 1, "stack_id": f"synthetic-{mode}",
                "stack_run_id": "synthetic-clean-contract-only", "source_commit": "0" * 40,
                "source_state_sha256": "0" * 64, "physical_gpu": index,
                "batch_size": (1, 4)[index], "qwen38_prefix_cache": "disabled",
                "max_residual_memory_mib": 128,
                "gateway": {"pid": 1000 + index, "sha256": "1" * 64},
                "native_stage": {"pid": 2000 + index, "sha256": "2" * 64},
                "model": {"id": "synthetic-not-a-model", "revision": "NOT_A_MODEL",
                          "files": {"contract-only": "3" * 64}},
            }

    def native_record(self, request_id, case, arm, mode):
        expected = case["expected_uniform_8"] if arm != "temporal" else case["expected_temporal"]["8"]
        temporal = arm == "temporal"
        if temporal:
            timestamps = expected["source_timestamps_seconds"]
            candidates = expected["candidate_indices"]
            sources = expected["source_frame_indices"]
        else:
            timestamps = [case["duration_seconds"] * index / 8 for index in range(8)]
            candidates = list(range(8))
            sources = expected["source_frame_indices"]
        pairs = [timestamps[min(2 * index, 7)] for index in range(4)]
        identity = "uniform" if not temporal else "temporal"
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
            "selected_source_frame_indices": sources, "source_indices_exact": temporal,
            "selected_frame_timestamps": timestamps, "selected_frame_content_fnv1a64": content,
            "has_nominal_timestamps": not temporal, "temporal2_pair_first_timestamps": pairs,
            "max_frames": 8, "max_candidate_frames": 64,
            "motion_threshold": 0.02, "scene_threshold": 0.25,
            "grid_t": 4, "grid_h": 10, "grid_w": 14, "premerge_patch_rows": 560,
            "merged_tokens_per_slice": 35, "video_token_count": 140, "prompt_token_count": 200,
            "media_hash": fixture.sha256_bytes(f"{case['id']}:{identity}:media".encode())[:16],
            "mrope_next": 260, "mrope_fnv1a64": mrope,
        }

    def transport(self, mode):
        def call(body, timeout):
            del timeout
            serial = self.counters[mode]
            self.counters[mode] += 1
            if self.fail_request and mode == "batch4" and serial == 0:
                raise RuntimeError("synthetic injected request failure")
            content = body["messages"][0]["content"]
            self.assert_content_order(content)
            selection = content[0]["video_url"].get("frame_selection")
            arm = selection["policy"] if selection else "default"
            data_url = content[0]["video_url"]["url"]
            case = next(case for case in self.manifest["cases"] if case["_data_url"] == data_url
                        and case["question"] == content[1]["text"])
            request_id = f"synthetic-{mode}-{serial}"
            self.native[mode][request_id + ":vlm"] = self.native_record(request_id, case, arm, mode)
            self.execution[mode][request_id + ":vlm"] = {
                "request_id": request_id + ":vlm", "prompt_tokens": 200, "prefill_tokens": 200,
                "prefix_reused": self.prefix_reused if serial == 0 else 0,
                "image_tokens": 140, "generated_tokens": 2, "elapsed_ms": 50 + serial,
                "vision_ms": 10.0, "prefill_ms": 20.0, "decode_ms": 15.0,
                "decode_tok_per_s": 50.0,
            }
            if case["evaluation_role"] == "quality_scored" or arm == "temporal":
                answer = case["expect_all"][0]
            else:
                answer = "unknown synthetic selection miss"
            response = {"request_id": request_id, "http_status": 200, "answer": answer,
                        "answer_sha256": fixture.sha256_bytes(answer.encode()),
                        "first_event_s": 0.005, "ttft_s": 0.010, "total_s": 0.050,
                        "finish_reason": "stop", "request_bytes": 1000}
            frames = [{"synthetic_contract_only": True, "id": request_id,
                       "choices": [{"delta": {"content": answer}, "finish_reason": "stop"}]}]
            return response, frames
        return call

    @staticmethod
    def assert_content_order(content):
        if [part["type"] for part in content] != ["video_url", "text"]:
            raise AssertionError("question did not follow video context")

    def run(self, out, *, exclusive_pass=True):
        return clean.execute_synthetic(
            self.manifest, transports={mode: self.transport(mode) for mode in clean.MODES},
            native_logs=self.native, execution_logs=self.execution,
            health_probes={mode: lambda: {"in_flight": 0} for mode in clean.MODES},
            memory_samplers={mode: lambda indexes: {index: 100 for index in indexes}
                             for mode in clean.MODES},
            provenances=self.provenances,
            exclusive=[{"mode": mode, "check": "exclusive_device", "passed": exclusive_pass,
                        "problems": []} for mode in clean.MODES],
            out_dir=out, short=short, bundle=BUNDLE, freeze_path=FREEZE)


class ControlledContractTests(unittest.TestCase):
    def test_seeded_schedule_quality_accuracy_excludes_controls_and_revalidates(self):
        with tempfile.TemporaryDirectory(prefix="umm-clean-contract-") as tmp:
            root = Path(tmp) / "evidence"
            report = FakeHarness(warmup=1, measured=2).run(root)
            self.assertTrue(report["gate_pass"], report["gate_reasons"])
            self.assertEqual(len(report["records"]), 144)
            self.assertEqual(report["claims"]["fixture_substring_accuracy"], "NOT_RUN")
            for mode in clean.MODES:
                for arm in clean.ARMS:
                    quality = report["answer_outcomes"]["quality_scored"][mode][arm]
                    controls = report["answer_outcomes"]["selection_miss_control"][mode][arm]
                    self.assertEqual(quality["declared"], 8)
                    self.assertEqual(quality["answer_hits"], 8)
                    self.assertTrue(quality["included_in_quality_accuracy"])
                    self.assertFalse(controls["included_in_quality_accuracy"])
                    self.assertEqual(controls["answer_hits"], 8 if arm == "temporal" else 0)
                    self.assertEqual(controls["answer_misses"], 0 if arm == "temporal" else 8)
            self.assertIn("unknown synthetic selection miss",
                          report["answer_outcomes"]["selection_miss_control"]["serial"]["default"]
                          ["cases"]["motion-late-control"]["answers"])
            validated = clean.validate_bundle(root, short, deep=False)
            self.assertEqual(validated["aggregates"], report["aggregates"])

    def test_fail_closed_prefix_device_selection_mrope_and_request(self):
        cases = (
            ("prefix", {"prefix_reused": 9}, True),
            ("device", {}, False),
            ("selection", {"diverge_selection": True}, True),
            ("mrope", {"diverge_mrope": True}, True),
            ("request", {"fail_request": True}, True),
        )
        for name, kwargs, exclusive_pass in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory(prefix=f"umm-clean-{name}-") as tmp:
                report = FakeHarness(**kwargs).run(Path(tmp) / "evidence",
                                                     exclusive_pass=exclusive_pass)
                self.assertFalse(report["gate_pass"])
                self.assertTrue(report["gate_reasons"])
                self.assertEqual(report["claims"]["live_latency"], "NOT_RUN")
                self.assertTrue(clean.validate_bundle(Path(tmp) / "evidence", short, deep=False))

    def test_evidence_tamper_and_aggregate_recompute_fail(self):
        with tempfile.TemporaryDirectory(prefix="umm-clean-tamper-") as tmp:
            root = Path(tmp) / "evidence"
            FakeHarness().run(root)
            frames = next(root.rglob("*.sse.json"))
            original = frames.read_bytes()
            frames.write_bytes(original + b"\n")
            with self.assertRaises(ValueError):
                clean.validate_bundle(root, short, deep=False)
            frames.write_bytes(original)
            clean.validate_bundle(root, short, deep=False)
            report_path = root / "report.json"
            report = json.loads(report_path.read_bytes())
            report["answer_outcomes"]["quality_scored"]["serial"]["default"]["answer_hits"] += 1
            report_path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                clean.validate_bundle(root, short, deep=False)


class BoundaryTests(unittest.TestCase):
    def test_gpu_zero_one_only_and_resident_port_forbidden(self):
        for gpu in (2, 6, 7):
            with self.subTest(gpu=gpu), mock.patch.object(
                    long_live, "validate_provenance", return_value={"physical_gpu": gpu}):
                with self.assertRaises(ValueError):
                    clean.validate_provenance("unused", mode="serial", run_id="r", source_root=ROOT)
        with self.assertRaises(ValueError):
            long_live.loopback_url("http://127.0.0.1:43301")

    def test_stack_identity_excludes_only_pid_and_live_resource_observations(self):
        provenances = {
            "serial": {"native_stage": {"pid": 10, "compute_pid": 20, "compute_used_mib": 100,
                                        "path": "/private/native", "sha256": "a" * 64}},
            "batch4": {"native_stage": {"pid": 30, "compute_pid": 40, "compute_used_mib": 200,
                                        "path": "/private/native", "sha256": "a" * 64}},
        }
        left, right = clean.comparable_stack_identity(provenances, "native_stage")
        self.assertEqual(left, right)
        self.assertEqual(left, {"sha256": "a" * 64})
        provenances["batch4"]["native_stage"]["sha256"] = "b" * 64
        left, right = clean.comparable_stack_identity(provenances, "native_stage")
        self.assertNotEqual(left, right)

    def test_committed_freeze_gate_rejects_uncommitted_or_modified_bytes(self):
        with tempfile.TemporaryDirectory(prefix="umm-clean-git-") as tmp:
            root = Path(tmp)
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "frozen.txt").write_text("frozen")
            (root / "freeze.json").write_text("{}")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=cklxx", "-c",
                            "user.email=q1293822641@gmail.com", "commit", "-q", "-m", "freeze"],
                           check=True)
            self.assertEqual(clean.validate_committed_fixture(root, bundle, root / "freeze.json")
                             ["committed_frozen_files"], 2)
            (bundle / "frozen.txt").write_text("changed")
            with self.assertRaises(ValueError):
                clean.validate_committed_fixture(root, bundle, root / "freeze.json")

    def test_self_check_and_default_not_run(self):
        for args in (["--self-check"], []):
            completed = subprocess.run(
                [sys.executable, "-I", str(ROOT / "tests" / "temporal_clean_live.py"), *args],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertIn("REAL_GENERATION: NOT_RUN", completed.stdout)
            self.assertIn("FIXTURE_SUBSTRING_ACCURACY: NOT_RUN", completed.stdout)
            self.assertIn("PERCEPTUAL_QUALITY: NOT_RUN", completed.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
