#!/usr/bin/env python3
"""Deterministic synthetic-only contracts for the NS-07 six-way harness.

No GPU, real model, remote service, or realtime network listener is used.  These
results are harness evidence only and cannot satisfy the live NS-07 gate.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import wave

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


soak = load("umminfer_ns07_soak_contract", ROOT / "tests/ns07_six_way_soak.py")
verify = load("umminfer_ns07_verify_contract", ROOT / "tests/ns07_six_way_soak_verify.py")
scorecard = load("umminfer_ns07_scorecard_contract", ROOT / "tools/umminfer_ns_scorecard.py")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sse(events):
    rows = []
    for index, event in enumerate(events):
        event = dict(event, seq=index)
        rows.append(b"data: " + json.dumps({"id": "http-1", "unified": event},
                                           separators=(",", ":")).encode() + b"\n\n")
    return b"".join(rows) + b"data: [DONE]\n\n"


def ws_frame(payload, direction="rx", opcode=1):
    raw = payload if isinstance(payload, bytes) else json.dumps(payload, separators=(",", ":")).encode()
    first = 0x80 | opcode
    if direction == "tx":
        mask = b"\x01\x02\x03\x04"
        masked = bytes(value ^ mask[index % 4] for index, value in enumerate(raw))
        return bytes((first, 0x80 | len(raw))) + mask + masked
    return bytes((first, len(raw))) + raw


def wav_pcm(pcm):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(pcm)
    return output.getvalue()


class FakeProcess:
    def __init__(self):
        self.process = self
        self.command = ["synthetic", "--gpu", "0"]
        self.stop_calls = 0

    @staticmethod
    def poll():
        return None

    def stop(self, endpoint=None):
        del endpoint
        self.stop_calls += 1

    def record(self):
        return {"name": "synthetic", "pid": 1, "start_monotonic_ns": 1,
                "ready_monotonic_ns": 1, "stop_monotonic_ns": 2,
                "returncode": 0, "stop_error": None, "log": "synthetic.log"}


class ContractTests(unittest.TestCase):
    def test_frozen_registry_and_normalized_scorecard_projection(self):
        document = json.loads((ROOT / "benchmarks/north-star/ns07-six-way-preregistration.json").read_text())
        registry = json.loads((ROOT / "benchmarks/north-star/cases.json").read_text())
        case = next(item for item in registry["cases"] if item["case_id"] == "NS-07")
        self.assertEqual(document["duration_s"], 1800)
        self.assertEqual(document["physical_gpu"], 0)
        self.assertEqual(tuple(document["lanes"]), soak.LANES)
        self.assertEqual(tuple(document["normalized_metrics"]), tuple(case["arm_metrics"]))
        self.assertEqual(tuple(case["arm_metrics"]), verify.METRIC_NAMES)
        with tempfile.TemporaryDirectory(prefix="umm-ns07-scorecard-") as tmp:
            root = Path(tmp)
            run_path = root / "run.json"
            write_json(run_path, {"placeholder": True})
            run = {"arm": "before", "source": {"execution_commit": "0" * 40},
                   "binaries": {"server": {"sha256": "1" * 64}},
                   "models": {"vlm": {"model_id": "synthetic", "tree_sha256": "2" * 64,
                                      "file_count": 1}},
                   "configs": {"vlm": {"config.json": "3" * 64}}, "inputs": {},
                   "_run_path": str(run_path)}
            verification = {"engineering_status": "FAIL", "production_status": "NOT_RUN",
                "metrics": {name: 0 for name in verify.METRIC_NAMES}, "hard_gates": {},
                "failures": ["synthetic"], "coverage": {}, "verification_sha256": "4" * 64}
            normalized = soak.normalized_from_verification(verification, run, 1)
            normalized_path = root / "NS-07.json"
            write_json(normalized_path, normalized)
            projected = scorecard.project_arm(root, case)
            self.assertEqual(projected["status"], "FAIL")
            self.assertEqual(set(projected["metrics"]), set(verify.METRIC_NAMES))
            self.assertEqual(normalized["source_identity"]["commit"], "0" * 40)
            self.assertEqual(normalized["arm"], "before")
            self.assertEqual(normalized["production_status"], "NOT_RUN")

    def test_tts_handoff_requires_complete_post_vlm_leg_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            instance = object.__new__(soak.Soak)
            instance.out = Path(tmp)
            trace = instance.out / "server.trace.jsonl"
            started = {"event": "legs_snapshot", "phase": "outputs_started",
                       "request_id": "http-1", "legs": [
                           {"stage": "vlm", "state": "completed"},
                           {"stage": "tts", "state": "running"},
                           {"stage": "tts", "state": "running"}]}
            trace.write_text(json.dumps(started) + "\n")
            self.assertFalse(instance._trace_tts_complete("http-1"))
            latest = {"event": "legs_snapshot", "phase": "leg_settled",
                      "request_id": "http-1", "legs": [
                          {"stage": "vlm", "state": "completed"},
                          {"stage": "tts", "state": "completed"},
                          {"stage": "tts", "state": "running"}]}
            with trace.open("a") as output:
                output.write(json.dumps(latest) + "\n")
            self.assertFalse(instance._trace_tts_complete("http-1"))
            latest["legs"][2]["state"] = "completed"
            with trace.open("a") as output:
                output.write(json.dumps(latest) + "\n")
            self.assertTrue(instance._trace_tts_complete("http-1"))
            latest["legs"].pop()
            with trace.open("a") as output:
                output.write(json.dumps(latest) + "\n")
            self.assertFalse(instance._trace_tts_complete("http-1"))

    def test_http_realtime_model_and_render_wire_stay_consistent(self):
        source = (ROOT / "tests/ns07_six_way_soak.py").read_text()
        self.assertNotIn("qwen3-native", source)
        self.assertIn('"--model", "qwen3-unified"', source)
        malformed = source.split('base = {"kind": "video"', 1)[1].split("self.request", 1)[0]
        self.assertNotIn("output_format", malformed)
        self.assertIn('bool(event.get("final", False)) == final', source)
        self.assertIn('if unified.get("kind") == "leg_error":\n                    fail_current()', source)
        self.assertIn('def fail_current():', source)
        self.assertIn('signal.signal(signal.SIGTERM, interrupted)', source)
        self.assertIn('except BaseException as error:', source)
        self.assertIn('body = public_base.request_body("qwen3-unified")', source)
        execute = source.split("    def execute(self):", 1)[1]
        self.assertLess(execute.index("self.setup_stack()"), execute.index("self.identities()"))
        self.assertLess(execute.index("self.identities()"), execute.index("self.run_lanes()"))
        self.assertIn('if not snapshot["positive_cursors"]:', source)
        self.assertNotIn('"positive-cursor audio before barge-in"', source)
        self.assertIn('"positive-cursor-or-terminal response decision"', source)
        self.assertIn('"stage_transition": stage_transition', source)

    def test_self_check_is_offline_and_ports_are_safe(self):
        completed = subprocess.run([sys.executable, "-I",
                                    str(ROOT / "tests/ns07_six_way_soak.py"), "--self-check"],
                                   capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("REAL_GENERATION: NOT_RUN", completed.stdout)
        self.assertIn("ISOLATED_PASS: NOT_RUN", completed.stdout)
        ports = soak.free_loopback_ports()
        self.assertEqual(len(set(ports)), 2)
        self.assertNotIn(43301, ports)

    def test_short_or_synthetic_run_never_isolated_pass(self):
        with tempfile.TemporaryDirectory(prefix="umm-ns07-short-") as tmp:
            root = Path(tmp)
            write_json(root / "run.json", {"schema": soak.SCHEMA, "case_id": "NS-07",
                "evidence_class": "SYNTHETIC_CONTRACT_ONLY", "duration_s": 1,
                "window_started_monotonic_ns": 1, "window_ended_monotonic_ns": 2_000_000_001})
            result = verify.Verify(root, allow_synthetic=True).run_verification()
            self.assertNotEqual(result["engineering_status"], "ISOLATED_PASS")
            self.assertFalse(result["hard_gates"]["elapsed_at_least_1800"])
            self.assertLess(result["metrics"]["elapsed_s"], 1800)

    def test_uncapped_jsonl_does_not_roll_or_delete(self):
        with tempfile.TemporaryDirectory(prefix="umm-ns07-jsonl-") as tmp:
            path = Path(tmp) / "all.jsonl"
            writer = soak.UncappedJsonl(path)
            for index in range(1000):
                writer.write({"index": index, "payload": "x" * 1024})
            self.assertGreater(path.stat().st_size, 1024 * 1024)
            rows = verify.read_jsonl(path)
            self.assertEqual(len(rows), 1000)
            self.assertEqual(rows[-1]["index"], 999)
            self.assertFalse(Path(str(path) + ".1").exists())

    def test_central_lease_blocks_switch_until_all_leases_release(self):
        manager = soak.OutputStageManager.__new__(soak.OutputStageManager)
        manager.condition = threading.Condition()
        manager.current_name = "flux"
        manager.current_process = FakeProcess()
        manager.switching = False
        manager.active = 0
        manager.serial = 0
        manager.history = []
        manager.leases = []
        manager.closed = False
        first = manager.acquire("flux", "first")
        second = manager.acquire("flux", "second")
        acquired = []

        def fake_start(stage):
            manager.current_name = stage
            manager.current_process = FakeProcess()

        manager._start = fake_start
        manager._command = lambda stage: (None, None, None)
        thread = threading.Thread(target=lambda: acquired.append(manager.acquire("wan", "wan")),
                                  daemon=True)
        thread.start()
        time.sleep(0.05)
        self.assertFalse(acquired)
        manager.release(first)
        time.sleep(0.05)
        self.assertFalse(acquired)
        manager.release(second)
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(acquired[0]["stage"], "wan")
        manager.release(acquired[0])
        self.assertEqual(manager.active, 0)

    def test_raw_sse_sequence_terminal_and_cross_session_tamper(self):
        with tempfile.TemporaryDirectory(prefix="umm-ns07-sse-") as tmp:
            root = Path(tmp)
            events = [{"kind": "vlm_delta", "text": "ok", "session_id": "s"},
                      {"kind": "complete", "text": "ok", "session_id": "s", "legs": []}]
            raw = sse(events)
            (root / "raw.sse").write_bytes(raw)
            (root / "input.json").write_text(json.dumps({"session_id": "s", "messages": []}))
            record = {"request_key": "text-0", "lane": "text_code", "kind": "text",
                "transport": "http", "path": "/v1/chat/completions", "session_id": "s",
                "request_id": "http-1", "http_status": 200, "http_headers": {"content-type": "text/event-stream"},
                "started_monotonic_ns": 1, "ended_monotonic_ns": 1000001, "latency_ms": 1.0,
                "raw_file": "raw.sse", "raw_sha256": sha(raw), "raw_bytes": len(raw),
                "request_file": "input.json", "request_sha256": sha((root / "input.json").read_bytes()),
                "terminal": "complete", "terminal_count": 1, "done_count": 1,
                "status": "completed", "ok": True, "expected_fault": False,
                "unexpected_error": False, "recovery_attempt": False, "recovery_success": False,
                "artifacts": []}
            (root / "requests.jsonl").write_text(json.dumps(record) + "\n")
            event_log = [{"request_key": "text-0", "lane": "text_code",
                          "observed_monotonic_ns": index + 1, "outer_id": "http-1",
                          "event": verify.project_event(dict(event, seq=index))}
                         for index, event in enumerate(events)]
            (root / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in event_log))
            write_json(root / "run.json", {"schema": soak.SCHEMA, "case_id": "NS-07",
                "evidence_class": "SYNTHETIC_CONTRACT_ONLY", "duration_s": 1,
                "window_started_monotonic_ns": 1, "window_ended_monotonic_ns": 2_000_000_001})
            result = verify.Verify(root, allow_synthetic=True).run_verification()
            self.assertEqual(result["metrics"]["request_count"], 1)
            self.assertEqual(result["metrics"]["completed_count"], 1)
            self.assertEqual(result["metrics"]["duplicate_terminal_count"], 0)
            record["unexpected_error"] = False
            (root / "requests.jsonl").write_text(json.dumps(record) + "\n")
            raw_bad = sse(events + [events[-1]])
            (root / "raw.sse").write_bytes(raw_bad)
            record["raw_sha256"] = sha(raw_bad)
            record["raw_bytes"] = len(raw_bad)
            (root / "requests.jsonl").write_text(json.dumps(record) + "\n")
            bad = verify.Verify(root, allow_synthetic=True).run_verification()
            self.assertGreater(bad["metrics"]["duplicate_terminal_count"], 0)
            self.assertNotEqual(bad["engineering_status"], "ISOLATED_PASS")

    def test_realtime_raw_websocket_decode_and_tamper(self):
        with tempfile.TemporaryDirectory(prefix="umm-ns07-ws-") as tmp:
            path = Path(tmp) / "frames.jsonl"
            rows = [
                {"index": 0, "monotonic_ns": 1, "direction": "rx",
                 "raw_base64": base64.b64encode(ws_frame({"type": "audio.delta", "response_id": "r",
                     "audio": base64.b64encode(b"\x01\x00").decode(), "text_cursor": 0})).decode()},
                {"index": 1, "monotonic_ns": 2, "direction": "tx",
                 "raw_base64": base64.b64encode(ws_frame({"type": "output_audio_buffer.playback_ack",
                     "response_id": "r", "text_cursor": 0}, "tx")).decode()},
                {"index": 2, "monotonic_ns": 3, "direction": "rx",
                 "raw_base64": base64.b64encode(ws_frame({"type": "response.completed",
                     "response_id": "r"})).decode()},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            frames = verify.decode_capture(path)
            self.assertEqual([row["event"]["type"] for row in frames if "event" in row],
                             ["audio.delta", "output_audio_buffer.playback_ack", "response.completed"])
            rows[1]["raw_base64"] = rows[0]["raw_base64"]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaises(ValueError):
                verify.decode_capture(path)

    def test_same_residency_growth_ignores_output_load_oscillation(self):
        with tempfile.TemporaryDirectory(prefix="umm-ns07-resource-") as tmp:
            root = Path(tmp)
            gpu = {"index": 0, "memory_used_mib": 100, "utilization_percent": 0,
                   "compute_processes": []}
            baseline = {"phase": "baseline", "residency": "asr+vlm", "gpu0": gpu,
                        "processes": {name: {"pid": 10, "rss_mib": 10} for name in ("server", "asr", "vlm")}}
            peak = {"phase": "measured", "residency": "asr+vlm+wan",
                    "gpu0": dict(gpu, memory_used_mib=80000), "processes": {}}
            final = {"phase": "final_same_residency", "residency": "asr+vlm",
                     "gpu0": dict(gpu, memory_used_mib=100),
                     "processes": {name: {"pid": 10, "rss_mib": 12} for name in ("server", "asr", "vlm")}}
            (root / "resources.jsonl").write_text("".join(json.dumps(row) + "\n"
                                                           for row in (baseline, peak, final)))
            checker = verify.Verify(root, allow_synthetic=True)
            idle = dict(gpu, memory_used_mib=3)
            checker.run = {"baseline_resource": baseline, "final_same_residency": final,
                           "gpu_initial": idle, "gpu_final": idle,
                           "processes": [{"pid": 10}]}
            device, rss = checker.verify_resources()
            self.assertEqual(device, 0)
            self.assertEqual(rss, 6)
            self.assertFalse(checker.failures)
            # NVML host PIDs need not equal container PIDs. The expected two-base
            # plus one-output topology is valid even when every integer differs.
            baseline["gpu0"] = dict(gpu, compute_processes=[{"pid": 1001}, {"pid": 1002}])
            final["gpu0"] = dict(gpu, memory_used_mib=100,
                                 compute_processes=[{"pid": 1001}, {"pid": 1002}])
            peak["gpu0"] = dict(gpu, memory_used_mib=80000,
                                compute_processes=[{"pid": 1001}, {"pid": 1002}, {"pid": 1003}])
            (root / "resources.jsonl").write_text("".join(json.dumps(row) + "\n"
                                                           for row in (baseline, peak, final)))
            namespaced = verify.Verify(root, allow_synthetic=True)
            namespaced.run = checker.run
            namespaced.verify_resources()
            self.assertFalse(namespaced.failures, namespaced.failures)
            transition = dict(peak, stage_transition=True, residency="asr+vlm",
                              gpu0=dict(gpu, compute_processes=[{"pid": 1001}, {"pid": 1002},
                                                                 {"pid": 1003}, {"pid": 1004}]))
            (root / "resources.jsonl").write_text("".join(json.dumps(row) + "\n"
                for row in (baseline, transition, peak, final)))
            transitioning = verify.Verify(root, allow_synthetic=True)
            transitioning.run = checker.run
            transitioning.verify_resources()
            self.assertFalse(transitioning.failures, transitioning.failures)
            # The same third process outside the leased Wan residency is excess.
            peak["residency"] = "asr+vlm"
            (root / "resources.jsonl").write_text("".join(json.dumps(row) + "\n"
                                                           for row in (baseline, peak, final)))
            foreign = verify.Verify(root, allow_synthetic=True)
            foreign.run = checker.run
            foreign.verify_resources()
            self.assertEqual(foreign.coverage["foreign_gpu_process_count"], 1)
            self.assertTrue(foreign.failures)

    def test_expected_fault_classification(self):
        self.assertTrue(verify.expected_fault_ok(
            {"expected_fault": True, "kind": "random_abort"}, "cancelled", True))
        self.assertTrue(verify.expected_fault_ok(
            {"expected_fault": True, "kind": "malformed_audio"}, "expected_error", True))
        self.assertFalse(verify.expected_fault_ok(
            {"expected_fault": True, "kind": "malformed_audio"}, "completed", False))

    @unittest.skipUnless(importlib.util.find_spec("msgpack"), "msgpack unavailable")
    def test_exact_native_public_pcm_join_and_tamper(self):
        import msgpack
        with tempfile.TemporaryDirectory(prefix="umm-ns07-pcm-") as tmp:
            root = Path(tmp)
            (root / "taps/tts").mkdir(parents=True)
            (root / "taps/tts-output").mkdir(parents=True)
            pcm = b"\x01\x00\x02\x00"
            for parent in ("http-1", "req-1"):
                child = parent + ":tts:0"
                submit = {"type": "submit", "request_id": child,
                          "data": {"request": {"inputs": {"text": "ok", "stream": True}}}}
                (root / "taps/tts" / (parent + ".msgpack")).write_bytes(
                    msgpack.packb(submit, use_bin_type=True))
                stream = {"type": "stream", "request_id": child,
                          "chunk": {"index": 0, "audio": pcm, "sample_rate": 24000, "channels": 1}}
                complete = {"type": "complete", "request_id": child, "success": True,
                            "result": {"streamed": True, "data": wav_pcm(pcm)}}
                (root / "taps/tts-output" / (parent + "-stream.msgpack")).write_bytes(
                    msgpack.packb(stream, use_bin_type=True))
                (root / "taps/tts-output" / (parent + "-complete.msgpack")).write_bytes(
                    msgpack.packb(complete, use_bin_type=True))
            for parent in ("parent-abort-http", "parent-abort-ws"):
                control = {"type": "complete", "request_id": parent, "success": False,
                           "error": "request aborted", "result": None}
                (root / "taps/tts-output" / (parent + ".msgpack")).write_bytes(
                    msgpack.packb(control, use_bin_type=True))
            requests = [
                {"request_key": "http", "transport": "http", "status": "completed"},
                {"request_key": "ws", "transport": "websocket", "status": "completed"},
            ]
            checker = verify.Verify(root, allow_synthetic=True)
            checker.requests = requests
            checker.verified = {
                "http": {"request_id": "http-1", "public_audio": [
                    {"stream_index": 0, "raw": pcm}]},
                "ws": {"request_id": "req-1", "public_audio": []},
            }
            checker.realtime = {"ws": {"audio_rows": [{"event": {
                "audio": base64.b64encode(pcm).decode(), "text_cursor": 0}}]}}
            checker.verify_native_pcm()
            self.assertFalse(checker.failures)
            self.assertEqual(checker.coverage["native_pcm_join_counts"], {"http": 1, "realtime": 1})
            checker.verified["http"]["public_audio"][0]["raw"] = b"\x00\x00\x00\x00"
            checker.verify_native_pcm()
            self.assertGreater(checker.counts["pcm_mismatch_count"], 0)

            # A cancelled native leg can have more retained native chunks than the
            # public prefix and has no successful terminal WAV. The exact public
            # prefix must still pass without weakening completed-response checks.
            for parent in ("cancel-http", "cancel-ws"):
                child = parent + ":tts:0"
                for index in (0, 1):
                    stream = {"type": "stream", "request_id": child,
                              "chunk": {"index": index, "audio": pcm,
                                        "sample_rate": 24000, "channels": 1}}
                    (root / "taps/tts-output" / f"{parent}-stream-{index}.msgpack").write_bytes(
                        msgpack.packb(stream, use_bin_type=True))
                complete = {"type": "complete", "request_id": child, "success": False,
                            "result": {"error": "aborted"}}
                (root / "taps/tts-output" / (parent + "-cancel.msgpack")).write_bytes(
                    msgpack.packb(complete, use_bin_type=True))
            cancelled = verify.Verify(root, allow_synthetic=True)
            cancelled.requests = [
                {"request_key": "http", "transport": "http", "status": "cancelled"},
                {"request_key": "ws", "transport": "websocket", "status": "cancelled"},
            ]
            cancelled.verified = {
                "http": {"request_id": "cancel-http", "public_audio": [
                    {"stream_index": 0, "raw": pcm}]},
                "ws": {"request_id": "cancel-ws", "public_audio": []},
            }
            cancelled.realtime = {"ws": {"pcm": pcm, "audio_rows": [{"event": {
                "audio": base64.b64encode(pcm).decode(), "text_cursor": 0}}]}}
            cancelled.verify_native_pcm()
            self.assertFalse(cancelled.failures, cancelled.failures)
            self.assertEqual(cancelled.coverage["native_pcm_join_counts"],
                             {"http": 1, "realtime": 1})

    def test_manifest_complete_self_excluded_and_tamper_rejected(self):
        with tempfile.TemporaryDirectory(prefix="umm-ns07-manifest-") as tmp:
            root = Path(tmp)
            run = {"schema": soak.SCHEMA, "case_id": "NS-07"}
            write_json(root / "run.json", run)
            normalized = {"case_id": "NS-07", "schema": soak.NORMALIZED_SCHEMA,
                "arm": "after", "engineering_status": "FAIL", "production_status": "NOT_RUN",
                "full_gate_status": "NOT_RUN", "metrics": {name: 0 for name in verify.METRIC_NAMES},
                "hard_gates": {}, "failures": [], "limitations": [], "source_identity": {},
                "binary_identities": {}, "model_identities": {}, "config_identity": {},
                "raw_evidence": {"run_sha256": verify.file_sha(root / "run.json"),
                                 "manifest_file_count": 2, "manifest_self_excluded": True},
                "coverage": {}, "deployment_scope": "SYNTHETIC_ONLY"}
            write_json(root / "NS-07.json", normalized)
            rows = []
            for name in ("NS-07.json", "run.json"):
                rows.append(f"{verify.file_sha(root / name)}  {name}")
            (root / "SHA256SUMS").write_text("\n".join(rows) + "\n")
            checker = verify.Verify(root, allow_synthetic=True)
            checker.coverage = {}
            checker.verify_manifest()
            self.assertEqual(checker.coverage["manifest"]["file_count"], 2)
            write_json(root / "run.json", {"tampered": True})
            with self.assertRaises(ValueError):
                checker.verify_manifest()


if __name__ == "__main__":
    unittest.main(verbosity=2)
