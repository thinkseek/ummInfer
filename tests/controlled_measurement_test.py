#!/usr/bin/env python3
"""CPU-only deterministic tests for the Stage C controlled-measurement layer.

Covers schedule reproducibility, single-variable invariance, failed and
precondition-failing arms, incomplete repetitions, latency aggregation, and
contract/hash tamper rejection. Live-mode runs use loopback canned responses
plus injected device/health samplers only: no GPU, remote host, real model,
optimizer, or resident service is touched, and no run here is real controlled
performance evidence.

Usage: python3 -I tests/controlled_measurement_test.py
"""

import base64
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import umminfer_ablation_run as runner  # noqa: E402
import umminfer_controlled_measurement as controlled  # noqa: E402

EXAMPLE = ROOT / "benchmarks" / "ablations" / "narrative-routing-controlled-example.json"
CONTROLLED_SM90 = ROOT / "benchmarks" / "ablations" / "narrative-routing-controlled-sm90.json"
VISUAL_CONTROLLED_SM90 = ROOT / "benchmarks" / "ablations" / "visual-form-controlled-sm90.json"
ROUTING_REPLAY = ROOT / "benchmarks" / "ablations" / "narrative-routing-replay.json"
VISUAL_CONFIG = ROOT / "benchmarks" / "ablations" / "visual-form.json"
VISUAL_REPLAY = ROOT / "benchmarks" / "ablations" / "visual-form-replay.json"
WEIGHTS = "a" * 64
CONFIG_HASH = "0" * 64
BINARY_HASH = "1" * 64


def pinned_model(model_id):
    return {"id": model_id, "revision": "test-local-snapshot",
            "config_file_sha256": CONFIG_HASH, "weights_sha256": WEIGHTS,
            "binary_sha256": BINARY_HASH}


def tree_bytes(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in Path(root).rglob("*") if path.is_file()}


def run_cli(out, config=EXAMPLE, extra=()):
    return subprocess.run(
        [sys.executable, "-I", str(ROOT / "python" / "umminfer_ablation_run.py"),
         "--config", str(config), "--replay-only", "--out", str(out), *extra],
        capture_output=True, text=True, timeout=60)


class ControlledHandler(BaseHTTPRequestHandler):
    """Serves canned responses in the exact declared schedule order.

    Any request that deviates from the expected (arm, prompt) sequence, or any
    extra request after a fail-fast precondition abort, fails the test.
    """

    cases = {}
    renders = {}
    expected = []
    requests = []
    chat_count = 0
    fail_indices = set()
    misordered = []
    telemetry_responses = False

    def log_message(self, *args):
        pass

    def _fail(self, status, message):
        raw = json.dumps({"error": {"message": message}}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
        self.close_connection = True

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length))
        type(self).requests.append((self.path, body))
        if self.path == "/v1/visual/render":
            case = self.renders[runner._sha256(body)]
            raw = base64.b64decode(case["response_base64"])
            content_type = "video/mp4"
        else:
            index = type(self).chat_count
            type(self).chat_count = index + 1
            if index >= len(self.expected):
                self._fail(500, "request beyond declared schedule")
                return
            arm, prompt_id = self.expected[index]
            actual_prompt = "conservation" if "momentum" in json.dumps(
                body["messages"]) else "lantern"
            if actual_prompt != prompt_id:
                type(self).misordered.append((index, arm, prompt_id, actual_prompt))
            if index in self.fail_indices:
                self._fail(500, "injected measured-request failure")
                return
            case = self.cases[(arm, prompt_id)]
            raw = base64.b64decode(case["response_sse_base64"])
            if type(self).telemetry_responses:
                frames = []
                for line in raw.decode("utf-8").splitlines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    unified = chunk.get("unified")
                    if unified is not None and unified.get("kind") == "complete":
                        unified["telemetry"] = request_envelope(
                            chunk["id"], epoch=index + 1, peak=4096 + index)
                    frames.append("data: " + json.dumps(
                        chunk, sort_keys=True, separators=(",", ":")) + "\n\n")
                frames.append("data: [DONE]\n\n")
                raw = "".join(frames).encode("utf-8")
            content_type = "text/event-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)
        self.close_connection = True


def good_contract(**overrides):
    contract = {
        "schema_version": 1, "warmup_repetitions": 1, "measured_repetitions": 2,
        "arm_order": "interleaved", "seed": 7, "cooldown_ms": 0,
        "exclusive_device": {"required": True, "gpu_indexes": [0],
                             "allowed_pids": [4242], "max_other_compute_processes": 0},
        "cleanup": {"require_in_flight_zero": True,
                    "max_residual_memory_mib": 128, "timeout_s": 5.0},
    }
    contract.update(overrides)
    return contract


def good_coverage(arms=("a", "b")):
    return {arm: {"warmups_declared": 1, "warmups_executed": 1, "warmup_failures": 0,
                  "measured_declared": 2, "measured_completed": 2, "measured_failed": 0}
            for arm in arms}


def good_provenance():
    return {"source_commit": "b" * 40, "source_state_sha256": "c" * 64,
            "config_sha256": "d" * 64, "data_sha256": "e" * 64,
            "artifact_manifest_sha256": "f" * 64,
            "visual_backends": {name: pinned_model(name)
                                for name in ("image", "video")}}


def native_stage(request_id="http-test", epoch=1, peak=4096):
    return {
        "schema_version": 1, "coverage": "COVERED",
        "scope": "request-native-cuda-runtime", "request_id": request_id,
        "stage": "vlm", "native_stage": "vlm-native", "backend": "qwen38",
        "device_index": 0, "lifecycle_status": "completed",
        "execution": {"mode": "serial", "batch_capacity": 1},
        "epoch": {"id": epoch, "reset": True,
                  "semantics": "new_allocation_ownership_epoch"},
        "allocated_bytes_peak": peak, "reserved_bytes_peak": peak,
        "allocated_bytes_final": peak // 2, "reserved_bytes_final": peak // 2,
        "sample_coverage": {"allocation_events": 2, "free_events": 1,
                            "terminal_snapshots": 1},
        "attribution": controlled.NATIVE_ALLOCATOR_ATTRIBUTION,
    }


def request_envelope(request_id="http-test", epoch=1, peak=4096):
    return {
        "schema_version": 1, "coverage": "COVERED",
        "scope": "request-native-allocator", "request_id": request_id,
        "lifecycle_status": "completed",
        "sample_coverage": {"stage_count": 1, "covered_stage_count": 1,
                            "truncated": False},
        "stages": [native_stage(request_id, epoch, peak)],
        "whole_device_nvml": "separate_measurement_resources",
    }


class ContractValidationTests(unittest.TestCase):
    def assert_rejected(self, mutate):
        contract = good_contract()
        mutate(contract)
        with self.assertRaises(ValueError):
            controlled.parse_contract(contract)

    def test_valid_contract_normalizes_defaults(self):
        parsed = controlled.parse_contract(good_contract())
        self.assertEqual(parsed["measured_repetitions"], 2)
        self.assertEqual(parsed["exclusive_device"]["max_memory_used_mib"], None)
        self.assertIsNone(controlled.parse_contract(None))
        self.assertNotIn("request_memory", parsed)
        opted_in = good_contract(request_memory={"enabled": True, "required": True})
        self.assertEqual(controlled.parse_contract(opted_in)["request_memory"],
                         {"enabled": True, "required": True})

    def test_tampered_contract_fields_are_rejected(self):
        cases = [
            lambda c: c.update(unknown_key=1),
            lambda c: c.update(schema_version=2),
            lambda c: c.pop("seed"),
            lambda c: c.pop("warmup_repetitions"),
            lambda c: c.pop("measured_repetitions"),
            lambda c: c.pop("arm_order"),
            lambda c: c.update(seed="7"),
            lambda c: c.update(seed=True),
            lambda c: c.update(measured_repetitions=0),
            lambda c: c.update(warmup_repetitions=-1),
            lambda c: c.update(arm_order="sequential"),
            lambda c: c.update(cooldown_ms=-5),
            lambda c: c["exclusive_device"].update(required=True, gpu_indexes=[]),
            lambda c: c["exclusive_device"].update(gpu_indexes=[0, 0]),
            lambda c: c["exclusive_device"].update(gpu_indexes=[-1]),
            lambda c: c["exclusive_device"].update(unknown=1),
            lambda c: c["cleanup"].update(timeout_s=0),
            lambda c: c["cleanup"].update(timeout_s=float("inf")),
            lambda c: c["cleanup"].update(require_in_flight_zero="yes"),
            lambda c: c["cleanup"].update(max_residual_memory_mib=-2),
            lambda c: c.update(request_memory=[]),
            lambda c: c.update(request_memory={"enabled": True, "unknown": 1}),
            lambda c: c.update(request_memory={"enabled": "yes"}),
            lambda c: c.update(request_memory={"enabled": False, "required": True}),
        ]
        for index, mutate in enumerate(cases):
            with self.subTest(case=index):
                self.assert_rejected(mutate)

    def test_example_config_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="umm-controlled-tamper-") as tmp:
            config = json.loads(EXAMPLE.read_text())
            config["dataset"]["path"] = str(
                ROOT / "benchmarks" / "ablations" / config["dataset"]["path"])
            config["replay"]["path"] = str(
                ROOT / "benchmarks" / "ablations" / config["replay"]["path"])
            config["measurement"]["seed"] = "not-an-integer"
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                runner.load_experiment(path)


    def test_sm90_controlled_config_pins_the_real_gate(self):
        experiment = runner.load_experiment(CONTROLLED_SM90)
        contract = controlled.parse_contract(experiment.config["measurement"])
        self.assertEqual(contract["warmup_repetitions"], 1)
        self.assertEqual(contract["measured_repetitions"], 3)
        self.assertEqual(contract["arm_order"], "interleaved")
        self.assertEqual(contract["exclusive_device"]["gpu_indexes"], [0, 1])
        self.assertTrue(all("expected_visual_modalities" in prompt
                            for prompt in experiment.prompts))
        for policy in runner.POLICIES:
            for prompt in experiment.prompts:
                body = runner.build_request(experiment, prompt, policy)
                self.assertNotIn("audio", body["modalities"])
                self.assertNotIn("audio", body)
        for block in [experiment.config["model"],
                      *experiment.config["visual_backends"].values(),
                      *experiment.config["auxiliary_backends"].values()]:
            self.assertTrue(controlled.HEX_256.fullmatch(block["weights_sha256"]))
            self.assertTrue(controlled.HEX_256.fullmatch(block["binary_sha256"]))
            self.assertTrue(block.get("config_file_sha256")
                            or block.get("config_files_sha256"))

    def test_sm90_visual_controlled_config_pins_single_device_gate(self):
        experiment = runner.load_experiment(VISUAL_CONTROLLED_SM90)
        contract = controlled.parse_contract(experiment.config["measurement"])
        self.assertEqual(contract["warmup_repetitions"], 1)
        self.assertEqual(contract["measured_repetitions"], 3)
        self.assertEqual(contract["arm_order"], "interleaved")
        self.assertEqual(contract["exclusive_device"]["gpu_indexes"], [0])
        self.assertEqual(len(contract["exclusive_device"]["allowed_pids"]), 3)
        self.assertTrue(all("expected_visual_modalities" in prompt
                            for prompt in experiment.prompts))
        self.assertEqual(experiment.config["visual_forms"], list(runner.VISUAL_FORMS))
        for form in runner.VISUAL_FORMS:
            for prompt in experiment.prompts:
                body = runner.build_request(experiment, prompt, form)
                self.assertNotIn("audio", body["modalities"])
                self.assertNotIn("audio", body)
        for block in [experiment.config["model"],
                      *experiment.config["visual_backends"].values()]:
            self.assertTrue(controlled.HEX_256.fullmatch(block["weights_sha256"]))
            self.assertTrue(controlled.HEX_256.fullmatch(block["binary_sha256"]))
            self.assertTrue(block.get("config_file_sha256")
                            or block.get("config_files_sha256"))


class ScheduleTests(unittest.TestCase):
    def test_schedule_is_reproducible_and_seed_bound(self):
        arms = list(runner.POLICIES)
        contract = controlled.parse_contract(good_contract(measured_repetitions=3))
        first = controlled.build_schedule(arms, contract)
        second = controlled.build_schedule(arms, contract)
        self.assertEqual(first, second)
        descriptor = controlled.schedule_descriptor(arms, contract, first)
        self.assertEqual(descriptor["schedule_sha256"],
                         controlled.schedule_descriptor(arms, contract, second)[
                             "schedule_sha256"])
        reseeded = controlled.parse_contract(good_contract(
            measured_repetitions=3, seed=8))
        self.assertNotEqual(first, controlled.build_schedule(arms, reseeded))
        self.assertEqual(len(first), len(arms) * (1 + 3))
        self.assertEqual([step["index"] for step in first], list(range(len(first))))

    def test_warmups_precede_measured_and_interleave_covers_arms(self):
        arms = list(runner.POLICIES)
        for order in ("interleaved", "randomized"):
            with self.subTest(order=order):
                contract = controlled.parse_contract(good_contract(arm_order=order))
                steps = controlled.build_schedule(arms, contract)
                warmup = [step for step in steps if step["phase"] == "warmup"]
                measured = [step for step in steps if step["phase"] == "measured"]
                self.assertEqual(steps[:len(warmup)], warmup)
                self.assertEqual({step["arm"] for step in warmup}, set(arms))
                self.assertEqual(len(measured), 2 * len(arms))
                self.assertEqual(sorted((step["arm"], step["repetition"])
                                        for step in measured),
                                 sorted((arm, rep) for arm in arms for rep in (1, 2)))
                if order == "interleaved":
                    for offset in range(0, len(measured), len(arms)):
                        block = measured[offset:offset + len(arms)]
                        self.assertEqual({step["arm"] for step in block}, set(arms))
                        self.assertEqual(len({step["repetition"] for step in block}), 1)

    def test_schedule_hash_detects_step_tampering(self):
        arms = list(runner.POLICIES)
        contract = controlled.parse_contract(good_contract())
        steps = controlled.build_schedule(arms, contract)
        digest = controlled.schedule_descriptor(arms, contract, steps)["schedule_sha256"]
        tampered = copy.deepcopy(steps)
        tampered[0]["arm"], tampered[1]["arm"] = tampered[1]["arm"], tampered[0]["arm"]
        self.assertNotEqual(digest, controlled.schedule_descriptor(
            arms, contract, tampered)["schedule_sha256"])


class AggregationTests(unittest.TestCase):
    def test_median_p95_and_dispersion_with_sample_counts(self):
        summary = controlled.summarize([4.0, 1.0, 3.0, 2.0])
        self.assertEqual(summary["sample_count"], 4)
        self.assertEqual(summary["total_count"], 4)
        self.assertEqual(summary["coverage"], "COVERED")
        self.assertEqual(summary["median_ms"], 2.5)
        self.assertEqual(summary["p95_ms"], 4.0)
        self.assertEqual(summary["iqr_ms"], 2.0)
        self.assertEqual(summary["mad_ms"], 1.0)
        self.assertEqual(summary["min_ms"], 1.0)
        self.assertEqual(summary["max_ms"], 4.0)
        odd = controlled.summarize([1.0, 2.0, 3.0])
        self.assertEqual(odd["median_ms"], 2.0)
        self.assertEqual(odd["iqr_ms"], 2.0)
        self.assertEqual(odd["mad_ms"], 1.0)
        self.assertEqual(odd["p95_ms"], 3.0)

    def test_missing_samples_reduce_coverage_not_totals(self):
        partial = controlled.summarize([1.0, None, 3.0])
        self.assertEqual(partial["coverage"], "PARTIAL")
        self.assertEqual(partial["sample_count"], 2)
        self.assertEqual(partial["total_count"], 3)
        empty = controlled.summarize([None, None])
        self.assertEqual(empty["coverage"], "NOT_COVERED")
        self.assertEqual(empty["sample_count"], 0)
        self.assertIsNone(empty["median_ms"])
        self.assertEqual(empty["total_count"], 2)

    def test_arm_resources_keep_true_polled_peaks_and_sample_counts(self):
        rows = [
            {"arm": "a", "poll_interval_ms": 50.0, "whole_device_gpus": {
                "0": {"baseline_mib": 100, "peak_mib": 140,
                      "final_mib": 100, "sample_count": 3}}},
            {"arm": "a", "poll_interval_ms": 50.0, "whole_device_gpus": {
                "0": {"baseline_mib": 100, "peak_mib": 160,
                      "final_mib": 100, "sample_count": 4}}},
            {"arm": "b", "poll_interval_ms": 50.0, "whole_device_gpus": {
                "0": {"baseline_mib": 100, "peak_mib": 900,
                      "final_mib": 900, "sample_count": 2}}},
        ]
        resources = runner._arm_resources("a", rows, [0])
        self.assertEqual(resources["gpus"]["0"]["peak_mib"], 160)
        self.assertEqual(resources["gpus"]["0"]["sample_count"], 7)
        self.assertEqual(resources["poll_interval_ms"], 50.0)


class EligibilityTests(unittest.TestCase):
    def evaluate(self, *, contract=None, provenance=None, model=None, arms=None,
                 preconditions=None, cleanups=None, visual_equivalence=None,
                 request_telemetry=None):
        return controlled.evaluate_eligibility(
            contract or controlled.parse_contract(good_contract()),
            provenance=provenance or good_provenance(),
            model=model or pinned_model("language"),
            arms=arms or good_coverage(),
            preconditions=preconditions if preconditions is not None else [
                {"check": "exclusive_device", "passed": True, "problems": []}],
            cleanups=cleanups if cleanups is not None else [
                {"passed": True, "schedule_index": 0, "problems": []}],
            visual_form_equivalence=visual_equivalence,
            request_telemetry=request_telemetry)

    def test_fully_covered_run_is_controlled(self):
        verdict = self.evaluate()
        self.assertEqual(verdict["label"], "CONTROLLED_LIVE")
        self.assertEqual(verdict["reasons"], [])

    def test_failed_visual_source_equivalence_refuses_controlled(self):
        for equivalence in ({"coverage": "COVERED", "passed": False},
                            {"coverage": "NOT_COVERED", "passed": True}):
            with self.subTest(equivalence=equivalence):
                verdict = self.evaluate(visual_equivalence=equivalence)
                self.assertEqual(verdict["label"], "UNCONTROLLED_LIVE")
                self.assertTrue(any("source-PNG equivalence" in reason
                                    for reason in verdict["reasons"]))

    def test_each_missing_dimension_refuses_controlled(self):
        cases = {
            "weights": {"model": {"weights_sha256": None}},
            "source": {"provenance": dict(good_provenance(), source_commit=None)},
            "state": {"provenance": dict(good_provenance(), source_state_sha256="zz")},
            "config": {"provenance": dict(good_provenance(), config_sha256=None)},
            "data": {"provenance": dict(good_provenance(), data_sha256=None)},
            "artifact": {"provenance": dict(good_provenance(),
                                            artifact_manifest_sha256=None)},
            "precondition-failed": {"preconditions": [
                {"check": "exclusive_device", "passed": False,
                 "problems": ["foreign pid"]}]},
            "precondition-missing": {"preconditions": []},
            "cleanup-failed": {"cleanups": [
                {"passed": False, "schedule_index": 3, "problems": ["in_flight=2"]}]},
        }
        for name, kwargs in cases.items():
            with self.subTest(case=name):
                verdict = self.evaluate(**kwargs)
                self.assertEqual(verdict["label"], "UNCONTROLLED_LIVE")
                self.assertTrue(verdict["reasons"])

    def test_incomplete_backend_provenance_refuses_controlled(self):
        cases = []
        cases.append(({"weights_sha256": WEIGHTS}, good_provenance()))
        missing_image = good_provenance()
        del missing_image["visual_backends"]["image"]
        cases.append((pinned_model("language"), missing_image))
        no_video_weights = good_provenance()
        no_video_weights["visual_backends"]["video"]["weights_sha256"] = None
        cases.append((pinned_model("language"), no_video_weights))
        no_aux_binary = good_provenance()
        no_aux_binary["auxiliary_backends"] = {
            "asr": dict(pinned_model("asr"), binary_sha256=None)}
        cases.append((pinned_model("language"), no_aux_binary))
        for index, (model, provenance) in enumerate(cases):
            with self.subTest(case=index):
                verdict = self.evaluate(model=model, provenance=provenance)
                self.assertEqual(verdict["label"], "UNCONTROLLED_LIVE")
                self.assertTrue(verdict["reasons"])

    def test_incomplete_repetitions_and_failed_arms_refuse_controlled(self):
        arms = good_coverage()
        arms["a"]["measured_completed"] = 1
        verdict = self.evaluate(arms=arms)
        self.assertEqual(verdict["label"], "UNCONTROLLED_LIVE")
        self.assertTrue(any("incomplete" in reason for reason in verdict["reasons"]))
        arms = good_coverage()
        arms["b"]["measured_failed"] = 1
        self.assertEqual(self.evaluate(arms=arms)["label"], "UNCONTROLLED_LIVE")
        arms = good_coverage()
        arms["a"]["warmups_executed"] = 0
        self.assertEqual(self.evaluate(arms=arms)["label"], "UNCONTROLLED_LIVE")
        arms = good_coverage()
        arms["a"]["warmup_failures"] = 1
        self.assertEqual(self.evaluate(arms=arms)["label"], "UNCONTROLLED_LIVE")

    def test_undeclared_exclusivity_refuses_controlled(self):
        contract = good_contract()
        contract["exclusive_device"]["required"] = False
        verdict = self.evaluate(contract=controlled.parse_contract(contract),
                                preconditions=[])
        self.assertEqual(verdict["label"], "UNCONTROLLED_LIVE")
        self.assertTrue(any("exclusivity" in reason for reason in verdict["reasons"]))
    def test_required_request_memory_must_be_fully_covered(self):
        contract = controlled.parse_contract(good_contract(
            request_memory={"enabled": True, "required": True}))
        summary = controlled.request_scoped_telemetry([request_envelope()])
        self.assertEqual(self.evaluate(contract=contract,
                                       request_telemetry=summary)["label"],
                         "CONTROLLED_LIVE")
        verdict = self.evaluate(contract=contract, request_telemetry={
            "coverage": "PARTIAL"})
        self.assertEqual(verdict["label"], "UNCONTROLLED_LIVE")
        self.assertTrue(any("allocator telemetry" in reason
                            for reason in verdict["reasons"]))


class PreconditionAndCleanupTests(unittest.TestCase):
    def test_foreign_compute_process_fails_exclusivity(self):
        contract = controlled.parse_contract(good_contract())
        record = controlled.check_exclusive_device(
            contract, lambda indexes: {i: [{"pid": 999, "used_mib": 50}]
                                       for i in indexes})
        self.assertFalse(record["passed"])
        self.assertEqual(record["foreign_compute_processes"],
                         [{"gpu_index": 0, "pid": 999, "used_mib": 50}])
        allowed = controlled.check_exclusive_device(
            contract, lambda indexes: {i: [{"pid": 4242, "used_mib": 50}]
                                       for i in indexes})
        self.assertTrue(allowed["passed"])

    def test_memory_ceiling_and_residual_cleanup(self):
        contract = controlled.parse_contract(good_contract())
        contract["exclusive_device"]["max_memory_used_mib"] = 512
        record = controlled.check_exclusive_device(
            contract, lambda indexes: {i: [] for i in indexes},
            baseline_mib={0: 900})
        self.assertFalse(record["passed"])
        cleanup = controlled.evaluate_cleanup(contract, in_flight=0,
                                              residual_mib={0: 200})
        self.assertFalse(cleanup["passed"])
        ok = controlled.evaluate_cleanup(contract, in_flight=0,
                                         residual_mib={0: 128})
        self.assertTrue(ok["passed"])
        stuck = controlled.evaluate_cleanup(contract, in_flight=2,
                                            residual_mib={0: 0})
        self.assertFalse(stuck["passed"])

    def test_request_scoped_telemetry_is_honestly_not_covered(self):
        telemetry = controlled.request_scoped_telemetry()
        self.assertEqual(telemetry["coverage"], "NOT_COVERED")
        self.assertEqual(telemetry["scope"], "whole-device-used-memory")


class RequestTelemetryTests(unittest.TestCase):
    def test_covered_envelope_is_normalized_and_paths_are_not_projected(self):
        envelope = request_envelope()
        envelope["stages"][0]["model_path"] = "/private/model/weights"
        envelope["stages"][0]["internal_note"] = "x" * 10000
        normalized = controlled.request_scoped_telemetry(
            envelope, request_id="http-test", terminal_state="completed")
        self.assertEqual(normalized["coverage"], "COVERED")
        self.assertEqual(normalized["stages"][0]["allocated_bytes_peak"], 4096)
        self.assertNotIn("model_path", normalized["stages"][0])
        self.assertNotIn("internal_note", normalized["stages"][0])
        self.assertEqual(normalized["whole_device_nvml"],
                         "separate_measurement_resources")
        optional_failure = copy.deepcopy(envelope)
        optional_failure["stages"][0]["lifecycle_status"] = "failed"
        normalized_failure = controlled.request_scoped_telemetry(optional_failure)
        self.assertEqual(normalized_failure["lifecycle_status"], "completed")
        self.assertEqual(normalized_failure["stages"][0]["lifecycle_status"],
                         "failed")
        child = copy.deepcopy(envelope)
        child["stages"][0]["request_id"] = "http-test:vlm"
        normalized_child = controlled.request_scoped_telemetry(
            child, request_id="http-test", terminal_state="completed")
        self.assertEqual(normalized_child["stages"][0]["request_id"],
                         "http-test:vlm")

    def test_aggregate_preserves_per_request_peaks_and_counts(self):
        rows = [request_envelope("http-a", 1, 100),
                request_envelope("http-b", 2, 300)]
        summary = controlled.request_scoped_telemetry(rows)
        self.assertEqual(summary["coverage"], "COVERED")
        self.assertEqual(summary["sample_coverage"]["request_count"], 2)
        self.assertEqual(summary["sample_coverage"]["covered_stage_count"], 2)
        self.assertEqual(summary["max_stage_allocated_bytes_peak"], 300)
        self.assertEqual([row["request_id"] for row in summary["requests"]],
                         ["http-a", "http-b"])
        missing = controlled.request_scoped_telemetry(
            None, request_id="http-c", terminal_state="cancelled")
        self.assertEqual(missing["coverage"], "NOT_COVERED")
        self.assertEqual(missing["lifecycle_status"], "cancelled")
        partial = controlled.request_scoped_telemetry(rows + [missing])
        self.assertEqual(partial["coverage"], "PARTIAL")

    def test_unsupported_backend_has_null_bytes_not_zero(self):
        envelope = request_envelope()
        stage = envelope["stages"][0]
        stage.update(coverage="NOT_COVERED", device_index=None,
                     execution={"mode": "batched", "batch_capacity": 4},
                     reason="batched_request_attribution_unavailable",
                     allocated_bytes_peak=None, reserved_bytes_peak=None,
                     allocated_bytes_final=None, reserved_bytes_final=None,
                     sample_coverage={"allocation_events": 0, "free_events": 0,
                                      "terminal_snapshots": 0})
        stage["epoch"] = {"id": 4, "reset": False,
                          "semantics": "not_reset_shared_or_unsupported"}
        envelope["coverage"] = "NOT_COVERED"
        envelope["sample_coverage"]["covered_stage_count"] = 0
        normalized = controlled.request_scoped_telemetry(envelope)
        self.assertEqual(normalized["coverage"], "NOT_COVERED")
        self.assertIsNone(normalized["stages"][0]["allocated_bytes_peak"])

    def test_missing_stage_snapshot_allows_explicit_no_epoch_shape(self):
        envelope = request_envelope()
        envelope["coverage"] = "NOT_COVERED"
        envelope["sample_coverage"]["covered_stage_count"] = 0
        envelope["stages"] = [{
            "schema_version": 1, "coverage": "NOT_COVERED",
            "scope": "request-native-cuda-runtime", "request_id": "http-test",
            "stage": "vlm", "native_stage": "unknown", "backend": "unknown",
            "device_index": None, "lifecycle_status": "completed",
            "execution": {"mode": "unknown", "batch_capacity": None},
            "epoch": {"id": None, "reset": False, "semantics": "no_request_epoch"},
            "allocated_bytes_peak": None, "reserved_bytes_peak": None,
            "allocated_bytes_final": None, "reserved_bytes_final": None,
            "sample_coverage": {"allocation_events": 0, "free_events": 0,
                                "terminal_snapshots": 0},
            "reason": "stage_terminal_snapshot_missing",
            "attribution": controlled.NATIVE_ALLOCATOR_ATTRIBUTION,
        }]
        normalized = controlled.request_scoped_telemetry(envelope)
        self.assertEqual(normalized["coverage"], "NOT_COVERED")
        self.assertIsNone(normalized["stages"][0]["epoch"]["id"])

    def test_malformed_metrics_fail_closed(self):
        mutations = [
            lambda row: row["stages"][0].update(allocated_bytes_peak=-1),
            lambda row: row["stages"][0].update(allocated_bytes_peak=True),
            lambda row: row["stages"][0].update(request_id="http-other"),
            lambda row: row.update(lifecycle_status="running"),
            lambda row: row["stages"][0].update(native_stage="/secret/model"),
            lambda row: row["stages"][0].update(backend="x" * 100),
            lambda row: row["stages"][0].update(allocated_bytes_final=99999),
            lambda row: row["stages"][0]["epoch"].update(reset=False),
            lambda row: row["stages"][0].update(sample_coverage={}),
            lambda row: row.update(whole_device_nvml="merged"),
            lambda row: row.update(request_id="bad/id"),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index):
                row = copy.deepcopy(request_envelope())
                mutate(row)
                with self.assertRaises(ValueError):
                    controlled.request_scoped_telemetry(row)
        with self.assertRaises(ValueError):
            controlled.request_scoped_telemetry(
                request_envelope(), request_id="http-other",
                terminal_state="completed")


class CaseSuccessTests(unittest.TestCase):
    def test_terminal_and_leg_coverage_must_be_valid(self):
        case = {"terminal_state": "completed", "latency": {"terminal_ms": 1.0},
                "terminal_legs": [{"stage": "image", "state": "completed"}],
                "leg_errors": [], "media_validation": [], "animations": []}
        self.assertTrue(runner._case_success(case))
        for mutate in (
                lambda value: value["terminal_legs"][0].update(state="failed"),
                lambda value: value.update(leg_errors=[{"error": "failed"}]),
                lambda value: value["latency"].update(terminal_ms=None),
                lambda value: value.update(terminal_state="failed")):
            changed = copy.deepcopy(case)
            mutate(changed)
            self.assertFalse(runner._case_success(changed))


class LiveHarness:
    def __init__(self, test, config, *, visual=False, fail_indices=(), telemetry=False):
        self.test = test
        self.config = config
        self.visual = visual
        self.telemetry = telemetry
        replay_path = VISUAL_REPLAY if visual else ROUTING_REPLAY
        replay = json.loads(replay_path.read_text())
        key = "visual_form" if visual else "routing_policy"
        ControlledHandler.cases = {(case[key], case["prompt_id"]): case
                                   for case in replay["cases"]}
        ControlledHandler.renders = {}
        if visual:
            for case in replay["cases"]:
                for render in case.get("renders", []):
                    ControlledHandler.renders[render["request_sha256"]] = render
        ControlledHandler.fail_indices = set(fail_indices)
        ControlledHandler.requests = []
        ControlledHandler.misordered = []
        ControlledHandler.chat_count = 0
        ControlledHandler.telemetry_responses = self.telemetry
        self.server = None

    def __enter__(self):
        experiment_config = self.config
        contract = controlled.parse_contract(experiment_config["measurement"])
        variable = experiment_config["variable"]
        arms = experiment_config["routing_policies" if variable == "routing_policy"
                                 else "visual_forms"]
        dataset = json.loads(Path(experiment_config["dataset"]["path"]).read_text())
        steps = controlled.build_schedule(arms, contract)
        ControlledHandler.expected = [
            (step["arm"], prompt["id"])
            for step in steps for prompt in dataset["prompts"]]
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), ControlledHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        return False

    def run(self, out, *, require=False, foreign_pid=False, stuck_health=False):
        memory = lambda indexes: {index: 1000 for index in indexes}  # noqa: E731
        if foreign_pid:
            apps = lambda indexes: {index: [{"pid": 999, "used_mib": 5}]  # noqa: E731
                                    for index in indexes}
        else:
            apps = lambda indexes: {index: [{"pid": 4242, "used_mib": 5}]  # noqa: E731
                                    for index in indexes}
        health = (lambda: {"in_flight": 1}) if stuck_health \
            else (lambda: {"in_flight": 0})
        experiment = runner.load_experiment(self.config_path)
        return runner.run_experiment(
            experiment, out, replay_only=False,
            base_url=f"127.0.0.1:{self.server.server_port}", timeout_s=5,
            resource_gpu_indexes=[0], gpu_sampler=memory,
            compute_apps_sampler=apps, health_probe=health,
            sleep_fn=lambda seconds: None, require_controlled=require)

    def write_config(self, tmp):
        path = Path(tmp) / "config.json"
        path.write_text(json.dumps(self.config, sort_keys=True))
        self.config_path = path
        return path


def example_config(*, weights=WEIGHTS, measured=2, cleanup_timeout=5.0):
    config = json.loads(EXAMPLE.read_text())
    config["dataset"]["path"] = str(
        ROOT / "benchmarks" / "ablations" / config["dataset"]["path"])
    config["replay"]["path"] = str(
        ROOT / "benchmarks" / "ablations" / config["replay"]["path"])
    config["model"].update(config_file_sha256=CONFIG_HASH,
                           weights_sha256=weights, binary_sha256=BINARY_HASH)
    config["visual_backends"] = {name: pinned_model(name)
                                 for name in ("image", "video")}
    config["measurement"]["measured_repetitions"] = measured
    config["measurement"]["exclusive_device"]["allowed_pids"] = [4242]
    config["measurement"]["cleanup"]["timeout_s"] = cleanup_timeout
    return config


class ControlledLiveTests(unittest.TestCase):
    def test_controlled_live_end_to_end_with_single_variable_invariance(self):
        with tempfile.TemporaryDirectory(prefix="umm-controlled-live-") as tmp:
            config = example_config()
            harness = LiveHarness(self, config)
            harness.write_config(tmp)
            with harness:
                report = harness.run(Path(tmp) / "out")
            self.assertEqual(report["run_status"], "COMPLETED")
            self.assertEqual(report["evidence_class"], "CONTROLLED_LIVE")
            self.assertEqual(report["claims"]["latency_evidence"], "CONTROLLED_LIVE")
            measurement = report["measurement"]
            self.assertEqual(measurement["status"], "EXECUTED")
            self.assertEqual(measurement["eligibility"]["reasons"], [])
            self.assertEqual(ControlledHandler.misordered, [])
            self.assertEqual(ControlledHandler.chat_count,
                             len(ControlledHandler.expected))
            arms = list(runner.POLICIES)
            self.assertEqual(len(report["cases"]), 2 * 2 * len(arms))
            for arm in arms:
                coverage = measurement["arms"][arm]
                self.assertEqual(coverage["warmups_executed"], 1)
                self.assertEqual(coverage["warmup_failures"], 0)
                self.assertEqual(coverage["measured_completed"], 2)
                self.assertEqual(coverage["measured_failed"], 0)
                for prompt_id in ("lantern", "conservation"):
                    aggregates = coverage["aggregates"]["per_prompt"][prompt_id]
                    self.assertEqual(aggregates["repetitions"], 2)
                    terminal = aggregates["terminal_ms"]
                    self.assertEqual(terminal["sample_count"], 2)
                    self.assertEqual(terminal["total_count"], 2)
                    self.assertEqual(terminal["coverage"], "COVERED")
                    for key in ("median_ms", "p95_ms", "iqr_ms", "mad_ms"):
                        self.assertIsNotNone(terminal[key])
                per_arm = json.loads((Path(tmp) / "out" / arm / "report.json").read_text())
                self.assertEqual(per_arm["evidence_class"], "CONTROLLED_LIVE")
                self.assertEqual(per_arm["measurement"]["schedule_sha256"],
                                 measurement["schedule"]["schedule_sha256"])
                self.assertEqual(per_arm["resources"]["coverage"], "COVERED")
                self.assertEqual(per_arm["resources"]["scope"],
                                 "whole-device-used-memory")
                matrix = per_arm["single_variable"]["fixed_matrix_sha256"]
                self.assertEqual(matrix,
                                 report["single_variable"]["fixed_matrix_sha256"])
            self.assertEqual(measurement["request_scoped_telemetry"]["coverage"],
                             "NOT_COVERED")
            self.assertTrue(all(row["passed"] for row in measurement["cleanup_checks"]))
            self.assertEqual(len(measurement["cleanup_checks"]),
                             len(measurement["schedule"]["steps"]))
            self.assertEqual(len(measurement["resources"]["steps"]),
                             len(measurement["schedule"]["steps"]))
            for row in measurement["resources"]["steps"]:
                sample = row["whole_device_gpus"]["0"]
                self.assertGreaterEqual(sample["sample_count"], 2)
                self.assertGreaterEqual(sample["peak_mib"], sample["final_mib"])
                self.assertIn("poll_interval_ms", row)
            for warmups in measurement["warmups"]:
                self.assertTrue(all(case["terminal_state"] == "completed"
                                    for case in warmups["cases"]))
            # Every repetition of one (arm, prompt) sends a byte-identical request,
            # and the fixed projection is identical across all arms and repetitions.
            by_pair = {}
            for case in report["cases"]:
                self.assertEqual(case["phase"], "measured")
                by_pair.setdefault((case["routing_policy"], case["prompt_id"]),
                                   []).append(case)
            for (arm, prompt_id), cases in by_pair.items():
                self.assertEqual(len(cases), 2)
                self.assertEqual({case["request_sha256"] for case in cases},
                                 {cases[0]["request_sha256"]})
                self.assertEqual(sorted(case["repetition"] for case in cases), [1, 2])
            fixed_by_prompt = {}
            for path, body in ControlledHandler.requests:
                if path != "/v1/chat/completions":
                    continue
                prompt_key = "momentum" in json.dumps(body["messages"])
                projection = runner.fixed_request(body)
                if prompt_key in fixed_by_prompt:
                    self.assertEqual(fixed_by_prompt[prompt_key], projection)
                else:
                    fixed_by_prompt[prompt_key] = projection
            self.assertEqual(len(fixed_by_prompt), 2)
            # Warmup traffic is exported out-of-band, never mixed into metrics.
            for arm in arms:
                warmup_path = Path(tmp) / "out" / "warmup" / arm / "trajectory.jsonl"
                self.assertTrue(warmup_path.is_file())
                measured = (Path(tmp) / "out" / arm / "trajectory.jsonl").read_text()
                self.assertEqual(len(measured.strip().splitlines()), 4)
                self.assertIn('"repetition":2', measured)
                self.assertNotIn("warmup", measured)

    def test_unpinned_weights_refuse_controlled_label(self):
        with tempfile.TemporaryDirectory(prefix="umm-controlled-weights-") as tmp:
            config = example_config(weights=None, measured=1)
            harness = LiveHarness(self, config)
            harness.write_config(tmp)
            with harness:
                report = harness.run(Path(tmp) / "out")
            self.assertEqual(report["run_status"], "COMPLETED")
            self.assertEqual(report["evidence_class"], "UNCONTROLLED_LIVE")
            reasons = report["measurement"]["eligibility"]["reasons"]
            self.assertTrue(any("weights_sha256" in reason for reason in reasons))

    def test_precondition_failure_downgrades_or_aborts(self):
        with tempfile.TemporaryDirectory(prefix="umm-controlled-precond-") as tmp:
            config = example_config(measured=1)
            harness = LiveHarness(self, config)
            harness.write_config(tmp)
            with harness:
                report = harness.run(Path(tmp) / "out", foreign_pid=True)
                self.assertEqual(report["evidence_class"], "UNCONTROLLED_LIVE")
                reasons = report["measurement"]["eligibility"]["reasons"]
                self.assertTrue(any("precondition failed" in reason
                                    for reason in reasons))
                with self.assertRaises(ValueError):
                    harness.run(Path(tmp) / "out-strict", require=True,
                                foreign_pid=True)
            # The strict abort happened before any request of the second run.
            self.assertEqual(ControlledHandler.chat_count,
                             len(ControlledHandler.expected))
            self.assertFalse((Path(tmp) / "out-strict" / "report.json").exists())

    def test_failed_measured_repetition_refuses_controlled(self):
        with tempfile.TemporaryDirectory(prefix="umm-controlled-fail-") as tmp:
            config = example_config(measured=1)
            harness = LiveHarness(self, config, fail_indices={8})
            harness.write_config(tmp)
            with harness:
                report = harness.run(Path(tmp) / "out")
            self.assertEqual(report["run_status"], "FAILED")
            self.assertEqual(report["evidence_class"], "UNCONTROLLED_LIVE")
            reasons = report["measurement"]["eligibility"]["reasons"]
            self.assertTrue(any("did not complete" in reason for reason in reasons))
            failed = [case for case in report["cases"]
                      if case["terminal_state"] != "completed"]
            self.assertEqual(len(failed), 1)

    def test_stuck_in_flight_cleanup_refuses_controlled(self):
        with tempfile.TemporaryDirectory(prefix="umm-controlled-cleanup-") as tmp:
            config = example_config(measured=1, cleanup_timeout=0.02)
            harness = LiveHarness(self, config)
            harness.write_config(tmp)
            with harness:
                report = harness.run(Path(tmp) / "out", stuck_health=True)
            self.assertEqual(report["run_status"], "COMPLETED")
            self.assertEqual(report["evidence_class"], "UNCONTROLLED_LIVE")
            reasons = report["measurement"]["eligibility"]["reasons"]
            self.assertTrue(any("cleanup" in reason for reason in reasons))
            self.assertTrue(any(not row["passed"]
                                for row in report["measurement"]["cleanup_checks"]))

    def test_visual_form_controlled_live_binds_every_repetition(self):
        with tempfile.TemporaryDirectory(prefix="umm-controlled-visual-") as tmp:
            config = json.loads(VISUAL_CONFIG.read_text())
            config["ablation_id"] = "stage-c-visual-form-controlled-test-v1"
            config["dataset"]["path"] = str(
                ROOT / "benchmarks" / "ablations" / config["dataset"]["path"])
            config["replay"]["path"] = str(VISUAL_REPLAY)
            config["model"].update(config_file_sha256=CONFIG_HASH,
                                   weights_sha256=WEIGHTS, binary_sha256=BINARY_HASH)
            config["visual_backends"] = {name: pinned_model(name)
                                         for name in ("image", "video")}
            config["measurement"] = good_contract(measured_repetitions=2)
            harness = LiveHarness(self, config, visual=True)
            harness.write_config(tmp)
            with harness:
                report = harness.run(Path(tmp) / "out")
            self.assertEqual(report["run_status"], "COMPLETED")
            self.assertEqual(report["evidence_class"], "CONTROLLED_LIVE")
            self.assertEqual(ControlledHandler.misordered, [])
            equivalence = report["first_stage_image_equivalence"]
            self.assertTrue(equivalence["passed"])
            self.assertEqual(len(equivalence["cases"]), 4)
            for row in equivalence["cases"]:
                self.assertIn("repetition", row)
                self.assertEqual(row["paired_image_case_count"], 1)
                self.assertTrue(row["request_sha256_equal"])
                self.assertTrue(row["source_bytes_identical"])
            broken = copy.deepcopy(report["cases"])
            next(case for case in broken
                 if case["visual_form"] == "image_then_i2v_animate"
                 )["artifacts"][0]["sha256"] = "0" * 64
            failed_equivalence = runner._first_stage_image_equivalence(
                broken, "visual_form")
            self.assertFalse(failed_equivalence["passed"])
            verdict = controlled.evaluate_eligibility(
                controlled.parse_contract(config["measurement"]),
                provenance=good_provenance(), model=pinned_model("language"),
                arms=good_coverage(runner.VISUAL_FORMS),
                preconditions=[{"check": "exclusive_device", "passed": True,
                                "problems": []}],
                cleanups=[{"passed": True, "schedule_index": 0, "problems": []}],
                visual_form_equivalence=failed_equivalence)
            self.assertEqual(verdict["label"], "UNCONTROLLED_LIVE")
            for case in report["cases"]:
                if case["visual_form"] == "image_then_i2v_animate":
                    self.assertEqual(len(case["animations"]), 1)
                    self.assertEqual(case["animations"][0]["terminal_state"],
                                     "completed")
    def test_request_native_allocator_telemetry_feeds_controlled_reports(self):
        with tempfile.TemporaryDirectory(prefix="umm-controlled-request-memory-") as tmp:
            config = example_config(measured=1)
            config["measurement"]["request_memory"] = {
                "enabled": True, "required": True}
            harness = LiveHarness(self, config, telemetry=True)
            harness.write_config(tmp)
            with harness:
                report = harness.run(Path(tmp) / "out")
            self.assertEqual(report["evidence_class"], "CONTROLLED_LIVE")
            summary = report["measurement"]["request_scoped_telemetry"]
            self.assertEqual(summary["coverage"], "COVERED")
            self.assertEqual(summary["scope"], "request-native-allocator")
            self.assertEqual(summary["sample_coverage"]["request_count"],
                             len(report["cases"]))
            self.assertGreater(summary["max_stage_allocated_bytes_peak"], 4096)
            self.assertEqual(report["measurement"]["resources"]["scope"],
                             "whole-device-used-memory")
            for case in report["cases"]:
                self.assertEqual(case["metric_coverage"]["request_native_allocator"],
                                 "COVERED")
                self.assertEqual(case["request_scoped_telemetry"]["lifecycle_status"],
                                 case["terminal_state"])
            for path, body in ControlledHandler.requests:
                if path == "/v1/chat/completions":
                    self.assertEqual(body.get("telemetry"),
                                     {"native_allocator": True})

    def test_required_request_memory_refuses_missing_public_envelopes(self):
        with tempfile.TemporaryDirectory(prefix="umm-controlled-request-missing-") as tmp:
            config = example_config(measured=1)
            config["measurement"]["request_memory"] = {
                "enabled": True, "required": True}
            harness = LiveHarness(self, config, telemetry=False)
            harness.write_config(tmp)
            with harness:
                report = harness.run(Path(tmp) / "out")
            self.assertEqual(report["evidence_class"], "UNCONTROLLED_LIVE")
            summary = report["measurement"]["request_scoped_telemetry"]
            self.assertEqual(summary["coverage"], "NOT_COVERED")
            self.assertIsNone(summary["max_stage_allocated_bytes_peak"])
            self.assertTrue(all(case["request_scoped_telemetry"]["coverage"] ==
                                "NOT_COVERED" for case in report["cases"]))
            self.assertTrue(any("allocator telemetry" in reason for reason in
                                report["measurement"]["eligibility"]["reasons"]))


class ReplayContractTests(unittest.TestCase):
    def test_replay_with_measurement_is_deterministic_and_not_live(self):
        with tempfile.TemporaryDirectory(prefix="umm-controlled-replay-") as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            for out in (first, second):
                completed = run_cli(out)
                self.assertEqual(completed.returncode, 0,
                                 completed.stdout + completed.stderr)
            self.assertEqual(tree_bytes(first), tree_bytes(second))
            report = json.loads((first / "report.json").read_text())
            self.assertEqual(report["evidence_class"], "CONTRACT_REPLAY")
            self.assertEqual(report["claims"]["latency_evidence"], "CONTRACT_REPLAY")
            measurement = report["measurement"]
            self.assertEqual(measurement["status"], "NOT_RUN")
            contract = controlled.parse_contract(
                json.loads(EXAMPLE.read_text())["measurement"])
            steps = controlled.build_schedule(list(runner.POLICIES), contract)
            self.assertEqual(measurement["schedule"]["schedule_sha256"],
                             controlled.schedule_descriptor(
                                 list(runner.POLICIES), contract,
                                 steps)["schedule_sha256"])
            self.assertEqual(len(measurement["schedule"]["steps"]), 4 * (1 + 3))
            self.assertEqual(measurement["contract"]["seed"], 20261126)

    def test_require_controlled_rejects_replay_mode(self):
        with tempfile.TemporaryDirectory(prefix="umm-controlled-strict-") as tmp:
            completed = run_cli(Path(tmp) / "out", extra=("--require-controlled",))
            self.assertEqual(completed.returncode, 2)
            self.assertIn("measurement", completed.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
