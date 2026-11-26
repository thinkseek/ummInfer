#!/usr/bin/env python3
"""Deterministic split/coalesced, ACK, cancel, reconnect and tamper fixtures."""

import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "python/umminfer_realtime_latency_audit.py"
SPEC = importlib.util.spec_from_file_location("umminfer_realtime_latency_audit_test", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pack(value):
    if value is None:
        return b"\xc0"
    if value is False:
        return b"\xc2"
    if value is True:
        return b"\xc3"
    if isinstance(value, int) and not isinstance(value, bool):
        if 0 <= value <= 0x7f:
            return bytes([value])
        if -32 <= value < 0:
            return bytes([value & 0xff])
        if 0 <= value <= 0xff:
            return b"\xcc" + bytes([value])
        if 0 <= value <= 0xffff:
            return b"\xcd" + struct.pack("!H", value)
        return b"\xd3" + struct.pack("!q", value)
    if isinstance(value, str):
        raw = value.encode("utf-8")
        if len(raw) < 32:
            return bytes([0xa0 | len(raw)]) + raw
        if len(raw) <= 0xff:
            return b"\xd9" + bytes([len(raw)]) + raw
        return b"\xda" + struct.pack("!H", len(raw)) + raw
    if isinstance(value, bytes):
        return b"\xc4" + bytes([len(value)]) + value
    if isinstance(value, list):
        if len(value) < 16:
            return bytes([0x90 | len(value)]) + b"".join(pack(row) for row in value)
        return b"\xdc" + struct.pack("!H", len(value)) + b"".join(pack(row) for row in value)
    if isinstance(value, dict):
        if len(value) < 16:
            return bytes([0x80 | len(value)]) + b"".join(
                pack(str(key)) + pack(row) for key, row in value.items())
        return b"\xde" + struct.pack("!H", len(value)) + b"".join(
            pack(str(key)) + pack(row) for key, row in value.items())
    raise TypeError(type(value))


def prompt_bytes(request_id, messages):
    return pack({"type": "submit", "request_id": request_id,
                 "data": {"request": {"inputs": {"messages": messages}}}})


def wire(event, tx=False, fin=True, opcode=1, payload=None):
    if payload is None:
        payload = json.dumps(event, separators=(",", ":"), sort_keys=True).encode("utf-8")
    first = (0x80 if fin else 0) | opcode
    mask_bit = 0x80 if tx else 0
    if len(payload) < 126:
        header = bytes([first, mask_bit | len(payload)])
    elif len(payload) < 65536:
        header = bytes([first, mask_bit | 126]) + struct.pack("!H", len(payload))
    else:
        header = bytes([first, mask_bit | 127]) + struct.pack("!Q", len(payload))
    if not tx:
        return header + payload
    mask = b"\x13\x07\x2a\x51"
    masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    return header + mask + masked


class Capture:
    def __init__(self):
        self.rows = []
        self.clock = 1_000_000_000

    def raw(self, raw, tx=False, advance=1_000_000):
        self.clock += advance
        row = {"index": len(self.rows), "monotonic_ns": self.clock,
               "direction": "tx" if tx else "rx",
               "raw_base64": base64.b64encode(raw).decode("ascii")}
        self.rows.append(row)
        return row

    def event(self, event, tx=False, advance=1_000_000):
        return self.raw(wire(event, tx=tx), tx=tx, advance=advance)

    def fragmented_event(self, event):
        payload = json.dumps(event, separators=(",", ":"), sort_keys=True).encode("utf-8")
        cut = len(payload) // 2
        self.raw(wire(None, fin=False, opcode=1, payload=payload[:cut]))
        return self.raw(wire(None, fin=True, opcode=0, payload=payload[cut:]))

    def write(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        write_capture(path, self.rows)


def write_capture(path, rows):
    path.write_bytes(b"".join((json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
                              for row in rows))


def append_turn(capture, segment, text):
    capture.event({"type": "input_audio_buffer.append",
                   "audio": base64.b64encode(b"\x01\x00" * 160).decode()}, tx=True)
    capture.event({"type": "input_audio_buffer.speech_started", "segment_id": segment})
    capture.event({"type": "input_audio_buffer.speech_stopped", "segment_id": segment,
                   "audio_ms": 100})
    capture.event({"type": "transcription.segment", "segment_id": segment,
                   "event_index": 0, "is_final": False, "text": text[:1]})
    capture.event({"type": "transcription.completed", "segment_id": segment, "text": text,
                   "generation_metadata": {"finish_reason": "stop", "generated_token_count": 2}})


def add_prompt(prompts, request_id, observed_ns, messages):
    raw = prompt_bytes(request_id, messages)
    prompts.append({"observed_monotonic_ns": observed_ns, "request_id": request_id,
                    "messages": messages, "raw_sha256": digest(raw),
                    "raw_base64": base64.b64encode(raw).decode("ascii")})


def ack(capture, rid, segment, cursor, final=False, accepted=True, advance=1_000_000):
    capture.event({"type": "output_audio_buffer.playback_ack", "response_id": rid,
                   "text_cursor": cursor}, tx=True, advance=advance)
    if accepted:
        capture.event({"type": "response.playback_ack", "response_id": rid, "segment_id": segment,
                       "text_cursor": cursor, "final": final})


def make_fixture(root):
    root = Path(root)
    (root / "frames").mkdir(parents=True)
    plan = audit.expected_plan(0x53, 1, 1)
    (root / "plan.json").write_text(json.dumps(plan, sort_keys=True) + "\n")
    prompts = []
    ack_schedules = {"session-0-connection-0": [], "session-0-connection-1": []}

    def record_ack_schedule(connection, capture, rid, cursor, label):
        row = next(row for row in reversed(capture.rows) if row["direction"] == "tx")
        ack_schedules[connection].append({
            "monotonic_ns": row["monotonic_ns"] - 1000, "response_id": rid,
            "text_cursor": cursor, "delay_ms": 5 if label.startswith("delayed") else 0,
            "label": label,
        })

    first = Capture()
    first.event({"type": "session.update", "session": {"audio_output": {"playback_ack": True}}}, tx=True)
    first.event({"type": "transcription_session.updated", "session": {"audio_output": {"playback_ack": True}}})
    append_turn(first, 0, "hello")
    prompt_ns = first.clock + 500_000
    add_prompt(prompts, "req-1:vlm", prompt_ns, [{"role": "user", "content": "hello"}])
    first.event({"type": "response.text.delta", "response_id": "req-1", "segment_id": 0,
                 "text": "a", "text_cursor": 1})
    # Fragmented WS message exercises split application payloads in the real decoder.
    first.fragmented_event({"type": "response.text.delta", "response_id": "req-1", "segment_id": 0,
                            "text": "bc", "text_cursor": 3})
    pcm = b"\x01\x00\x02\x00" * 20
    first.event({"type": "audio.delta", "response_id": "req-1", "segment_id": 0,
                 "format": "pcm16", "sample_rate": 24000, "channels": 1,
                 "audio": base64.b64encode(pcm).decode(), "text_cursor": 0, "text_end_cursor": 1})
    ack(first, "req-1", 0, 0)
    record_ack_schedule("session-0-connection-0", first, "req-1", 0, "initial-zero")
    first.event({"type": "audio.delta", "response_id": "req-1", "segment_id": 0,
                 "format": "pcm16", "sample_rate": 24000, "channels": 1,
                 "audio": base64.b64encode(pcm).decode(), "text_cursor": 1, "text_end_cursor": 3})
    ack(first, "req-1", 0, 1)
    record_ack_schedule("session-0-connection-0", first, "req-1", 1, "delayed-partial")
    # Duplicate cursor is accepted and echoed; decreasing and unissued probes are ignored.
    ack(first, "req-1", 0, 1)
    record_ack_schedule("session-0-connection-0", first, "req-1", 1, "duplicate-partial")
    ack(first, "req-1", 0, 0, accepted=False)
    record_ack_schedule("session-0-connection-0", first, "req-1", 0, "decreasing-ignored")
    ack(first, "req-1", 0, 2, accepted=False)
    record_ack_schedule("session-0-connection-0", first, "req-1", 2, "unissued-ignored")
    first.event({"type": "audio.done", "response_id": "req-1", "segment_id": 0, "text_cursor": 3})
    first.event({"type": "response.completed", "response_id": "req-1", "segment_id": 0, "text_cursor": 3})
    # Delayed post-completion barge-in retains exactly the acknowledged one-byte prefix.
    first.event({"type": "input_audio_buffer.append",
                 "audio": base64.b64encode(b"\x03\x00" * 160).decode()}, tx=True, advance=5_000_000)
    cancel_schedule_ns = first.rows[-1]["monotonic_ns"] - 1000
    first.event({"type": "response.cancelled", "response_id": "req-1", "segment_id": 0,
                 "reason": "barge_in", "text_cursor": 1, "acknowledged_text": "a",
                 "history_recorded": True})
    first.event({"type": "input_audio_buffer.speech_started", "segment_id": 1})
    first.event({"type": "input_audio_buffer.speech_stopped", "segment_id": 1, "audio_ms": 100})
    first.event({"type": "transcription.completed", "segment_id": 1, "text": "next",
                 "generation_metadata": {"finish_reason": "stop", "generated_token_count": 2}})
    expected_second = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "a"},
                       {"role": "user", "content": "next"}]
    add_prompt(prompts, "req-2:vlm", first.clock + 500_000, expected_second)
    first.event({"type": "response.text.delta", "response_id": "req-2", "segment_id": 1,
                 "text": "xy", "text_cursor": 2})
    first.event({"type": "audio.delta", "response_id": "req-2", "segment_id": 1,
                 "format": "pcm16", "sample_rate": 24000, "channels": 1,
                 "audio": base64.b64encode(pcm).decode(), "text_cursor": 0, "text_end_cursor": 2})
    first.event({"type": "audio.done", "response_id": "req-2", "segment_id": 1, "text_cursor": 2})
    first.event({"type": "response.completed", "response_id": "req-2", "segment_id": 1, "text_cursor": 2})
    ack(first, "req-2", 1, 2, final=True)
    record_ack_schedule("session-0-connection-0", first, "req-2", 2, "delayed-final")
    first.raw(wire(None, tx=True, opcode=8, payload=struct.pack("!H", 1000)), tx=True)
    first.write(root / "frames/session-0-connection-0.jsonl")

    second = Capture()
    second.clock = first.clock + 1_000_000
    second.event({"type": "session.update", "session": {"audio_output": {"playback_ack": True}}}, tx=True)
    second.event({"type": "transcription_session.updated", "session": {"audio_output": {"playback_ack": True}}})
    # Late old-response ACK after reconnect is stale and must not extend history.
    ack(second, "req-1", 0, 0, accepted=False)
    record_ack_schedule("session-0-connection-1", second, "req-1", 0, "late-old-ack-after-reconnect")
    append_turn(second, 0, "fresh")
    add_prompt(prompts, "req-3:vlm", second.clock + 500_000, [{"role": "user", "content": "fresh"}])
    second.event({"type": "response.text.delta", "response_id": "req-3", "segment_id": 0,
                  "text": "z", "text_cursor": 1})
    second.event({"type": "audio.delta", "response_id": "req-3", "segment_id": 0,
                  "format": "pcm16", "sample_rate": 24000, "channels": 1,
                  "audio": base64.b64encode(pcm).decode(), "text_cursor": 0, "text_end_cursor": 1})
    second.event({"type": "audio.done", "response_id": "req-3", "segment_id": 0, "text_cursor": 1})
    second.event({"type": "response.completed", "response_id": "req-3", "segment_id": 0, "text_cursor": 1})
    ack(second, "req-3", 0, 1, final=True)
    record_ack_schedule("session-0-connection-1", second, "req-3", 1, "delayed-final")
    second.raw(wire(None, tx=True, opcode=8, payload=struct.pack("!H", 1000)), tx=True)
    second.write(root / "frames/session-0-connection-1.jsonl")

    (root / "vlm-prompts.jsonl").write_bytes(b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in prompts))
    (root / "resources.jsonl").write_text(
        json.dumps({"time_ms": 0.0, "processes": {"server": {
            "pid": 53, "rss_bytes": 1000, "cpu_percent": None}}}, sort_keys=True) + "\n" +
        json.dumps({"time_ms": 50.0, "processes": {"server": {
            "pid": 53, "rss_bytes": 1400, "cpu_percent": 2.5}}}, sort_keys=True) + "\n")
    run = {
        "schema": "umminfer-realtime-network-live/v1",
        "validation_class": "DETERMINISTIC_CONTRACT_FIXTURE",
        "physical_microphone_speaker": "NOT_COVERED", "perceptual_listening_quality": "NOT_COVERED",
        "connections_per_session": {"0": 2}, "scenario_results": {},
        "final_health": {"status": "ok", "in_flight": 0},
        "cleanup": {"owned_processes_exited": True, "selected_gpus_idle": True},
        "owned_processes": [{"name": "server", "pid": 53}],
        "queue_contract": dict(audit.QUEUE_CONTRACT),
        "resource_bounds": {"server_rss_growth_max_bytes": 1024},
        "resource_usage": {"server": {"samples": 2, "rss_min_bytes": 1000,
                                      "rss_max_bytes": 1400, "rss_growth_bytes": 400}},
        "ack_schedules": ack_schedules,
        "cancel_schedules": {"session-0-connection-0": [{
            "monotonic_ns": cancel_schedule_ns, "response_id": "req-1", "kind": "delayed",
            "ack_cursor": 1}]},
    }
    (root / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    pcm_path = root / "pcm/session-0/round-0-req-1.pcm"
    pcm_path.parent.mkdir(parents=True)
    pcm_path.write_bytes(pcm * 2)
    reseal(root)


def reseal(root):
    root = Path(root)
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{digest(path.read_bytes())}  {path.relative_to(root)}\n")
    (root / "SHA256SUMS").write_text("".join(lines))


def mutate_capture(root, name, mutate):
    path = Path(root) / "frames" / f"{name}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(rows)
    for index, row in enumerate(rows):
        row["index"] = index
    write_capture(path, rows)
    reseal(root)


def replace_event(row, event):
    row["raw_base64"] = base64.b64encode(wire(event, tx=row["direction"] == "tx")).decode("ascii")


class TransportAndStatsTests(unittest.TestCase):
    def test_split_and_coalesced_frames_are_identical(self):
        events = [{"type": "one", "text": "x" * 200}, {"type": "two", "value": 2}]
        raw = [wire(event, tx=True) for event in events]
        whole = audit.decode_transport_chunks(raw, "tx")
        coalesced = audit.decode_transport_chunks([b"".join(raw)], "tx")
        split = audit.decode_transport_chunks([b"".join(raw)[index:index + 1]
                                               for index in range(sum(map(len, raw)))], "tx")
        self.assertEqual([row["payload"] for row in whole], [row["payload"] for row in coalesced])
        self.assertEqual([row["payload"] for row in whole], [row["payload"] for row in split])
        with self.assertRaises(audit.AuditError):
            audit.decode_transport_chunks([raw[0][:-1]], "tx")

    def test_stats_keep_every_outlier(self):
        metric = audit.summarize([{"value": value, "sample": index}
                                  for index, value in enumerate([1, 2, 3, 4, 100])])
        self.assertEqual(metric["summary"]["median"], 3)
        self.assertEqual(metric["summary"]["p95"], 80.8)
        self.assertEqual(metric["summary"]["iqr"], 2)
        self.assertEqual(metric["summary"]["mad"], 1)
        self.assertEqual(len(metric["samples"]), 5)
        self.assertEqual([row["value"] for row in metric["outliers"]], [100])


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="umminfer-realtime-latency-test-")
        self.root = Path(self.temp.name) / "evidence"
        self.root.mkdir()
        make_fixture(self.root)
        self.output = Path(self.temp.name) / "audit.json"

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self):
        return subprocess.run([sys.executable, "-I", str(MODULE_PATH), str(self.root),
                               "--profile", "generic", "--output", str(self.output)],
                              text=True, capture_output=True, check=False)

    def assert_fail_closed(self, expected):
        result = self.run_cli()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn(expected, result.stderr)
        self.assertFalse(self.output.exists(), "failure must not leave an audit report")

    def test_valid_delayed_cancel_reconnect_and_ack_fixture(self):
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(self.output.read_text())
        self.assertEqual(report["counts"]["server_responses"], 3)
        self.assertEqual(report["counts"]["native_vlm_prompts"], 3)
        self.assertEqual(report["counts"]["history_turns"], 3)
        self.assertEqual(report["counts"]["reconnects"], 1)
        self.assertEqual(report["protocol"]["accepted_decreasing_cursors"], 0)
        self.assertEqual(report["protocol"]["ignored_ack_categories"]["decreasing"], 1)
        self.assertEqual(report["protocol"]["ignored_ack_categories"]["unissued"], 1)
        self.assertEqual(report["protocol"]["ignored_ack_categories"]["stale_after_reconnect"], 1)
        self.assertGreater(report["metrics"]["global"]["barge_first_append_to_cancel"]["summary"]["median"], 0)
        self.assertEqual(report["history_turns"][0]["assistant"], "a")
        self.assertEqual(report["responses"][0]["accepted_cursors"], [0, 1, 1])
        second = audit.audit_evidence(self.root, "generic")
        self.assertEqual(audit.render_report(report), audit.render_report(second))

    def test_non_monotonic_clock_fail_closed(self):
        def mutate(rows):
            rows[-1]["monotonic_ns"] = rows[-2]["monotonic_ns"]
        mutate_capture(self.root, "session-0-connection-0", mutate)
        self.assert_fail_closed("non-monotonic capture clock")

    def test_tampered_frame_fail_closed(self):
        def mutate(rows):
            raw = bytearray(base64.b64decode(rows[2]["raw_base64"]))
            raw[-1] ^= 1
            rows[2]["raw_base64"] = base64.b64encode(raw).decode("ascii")
        mutate_capture(self.root, "session-0-connection-0", mutate)
        self.assert_fail_closed("invalid realtime JSON")

    def test_decreasing_accepted_ack_fail_closed(self):
        def mutate(rows):
            accepted = [row for row in rows if b"response.playback_ack" in base64.b64decode(row["raw_base64"])]
            replace_event(accepted[2], {"type": "response.playback_ack", "response_id": "req-1",
                                        "segment_id": 0, "text_cursor": 0, "final": False})
        mutate_capture(self.root, "session-0-connection-0", mutate)
        self.assert_fail_closed("decreasing cursor")

    def test_stale_audio_after_delayed_cancel_fail_closed(self):
        def mutate(rows):
            source = next(row for row in rows if b"audio.delta" in base64.b64decode(row["raw_base64"]))
            stale = dict(source)
            stale["monotonic_ns"] = rows[-1]["monotonic_ns"] + 1_000_000
            rows.append(stale)
        mutate_capture(self.root, "session-0-connection-0", mutate)
        self.assert_fail_closed("stale event after cancellation")

    def test_cross_connection_old_audio_fail_closed(self):
        old_rows = [json.loads(line) for line in
                    (self.root / "frames/session-0-connection-0.jsonl").read_text().splitlines()]
        source = next(row for row in old_rows if b"audio.delta" in base64.b64decode(row["raw_base64"]))
        def mutate(rows):
            stale = dict(source)
            stale["monotonic_ns"] = rows[-1]["monotonic_ns"] + 1_000_000
            rows.append(stale)
        mutate_capture(self.root, "session-0-connection-1", mutate)
        self.assert_fail_closed("cross-session/connection response leak")

    def test_observer_prompt_tamper_fail_closed(self):
        path = self.root / "vlm-prompts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["messages"][0]["content"] = "tampered-history"
        path.write_bytes(b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows))
        reseal(self.root)
        self.assert_fail_closed("observer normalization mismatch")

    def test_manifest_tamper_fail_closed(self):
        path = self.root / "run.json"
        path.write_text(path.read_text().replace('"in_flight": 0', '"in_flight": 1'))
        self.assert_fail_closed("evidence hash mismatch")


if __name__ == "__main__":
    unittest.main(verbosity=2)
