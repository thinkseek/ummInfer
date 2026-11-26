#!/usr/bin/env python3
"""Deterministic scheduler, wire-codec and independent-verifier mutation tests."""

import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "umminfer_realtime_network_test", ROOT / "python/umminfer_realtime_network.py")
net = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(net)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def frame(event, tx=False):
    payload = json.dumps(event, separators=(",", ":")).encode()
    mask_bit = 0x80 if tx else 0
    if len(payload) < 126:
        header = bytes((0x81, mask_bit | len(payload)))
    else:
        header = bytes((0x81, mask_bit | 126)) + struct.pack("!H", len(payload))
    if not tx:
        return header + payload
    mask = b"\x01\x02\x03\x04"
    masked = bytes(value ^ mask[i % 4] for i, value in enumerate(payload))
    return header + mask + masked


class Capture:
    def __init__(self):
        self.rows = []
        self.clock = 1000

    def add(self, event, tx=False):
        raw = frame(event, tx)
        self.clock += 100
        row = {"index": len(self.rows), "monotonic_ns": self.clock,
               "direction": "tx" if tx else "rx",
               "raw_base64": base64.b64encode(raw).decode()}
        self.rows.append(row)
        return row["index"]

    def write(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"".join((json.dumps(row, sort_keys=True) + "\n").encode()
                                  for row in self.rows))


def response(capture, response_id, segment, user, text, cancelled=False):
    capture.add({"type": "transcription.completed", "segment_id": segment, "text": user})
    capture.add({"type": "response.text.delta", "response_id": response_id,
                 "segment_id": segment, "text": text, "text_cursor": len(text.encode())})
    pcm = (response_id.encode() + b"\x00\x01") * 4
    first = pcm[:len(pcm) // 2] if cancelled else pcm
    second = pcm[len(first):]
    capture.add({"type": "audio.delta", "response_id": response_id, "segment_id": segment,
                 "format": "pcm16", "sample_rate": 24000, "channels": 1,
                 "audio": base64.b64encode(first).decode(), "text_cursor": 0,
                 "text_end_cursor": len(text.encode())})
    if cancelled:
        capture.add({"type": "audio.delta", "response_id": response_id, "segment_id": segment,
                     "format": "pcm16", "sample_rate": 24000, "channels": 1,
                     "audio": base64.b64encode(second).decode(), "text_cursor": 3,
                     "text_end_cursor": len(text.encode())})
    capture.add({"type": "audio.done", "response_id": response_id, "segment_id": segment,
                 "text_cursor": len(text.encode())})
    capture.add({"type": "response.completed", "response_id": response_id,
                 "segment_id": segment, "text_cursor": len(text.encode())})
    return pcm


def ack(capture, response_id, segment, cursor, final):
    capture.add({"type": "output_audio_buffer.playback_ack", "response_id": response_id,
                 "text_cursor": cursor}, tx=True)
    return capture.add({"type": "response.playback_ack", "response_id": response_id,
                        "segment_id": segment, "text_cursor": cursor, "final": final})


def cancel(capture, response_id, segment, text="abc"):
    return capture.add({"type": "response.cancelled", "response_id": response_id,
                        "segment_id": segment, "reason": "barge_in", "text_cursor": 3,
                        "acknowledged_text": text, "history_recorded": True})


def prompt(request_id, user):
    return {"observed_monotonic_ns": 1, "request_id": request_id,
            "messages": [{"role": "user", "content": user}],
            "raw_sha256": digest(request_id.encode()),
            "raw_base64": base64.b64encode(request_id.encode()).decode()}


def fixture(root):
    root = Path(root)
    (root / "frames").mkdir(parents=True)
    plan = net.make_plan(0x41C0DE, 6, 2)
    (root / "plan.json").write_text(json.dumps(plan, sort_keys=True) + "\n")
    prompts = []
    echo_info = {}
    for session in range(6):
        capture = Capture()
        response_id = f"req-{session}"
        user = f"user-{session}"
        response(capture, response_id, 0, user, "answer")
        ack(capture, response_id, 0, 6, True)
        prompts.append(prompt(response_id + ":vlm", user))
        if session == 0:
            # Protected echo window contains no cut-in; disabled arm cancels the
            # same response after its exact acknowledged three-byte prefix.
            echo_pcm = response(capture, "echo-req", 1, "echo-user", "abcdef", cancelled=True)
            protected_start = len(capture.rows)
            ack(capture, "echo-req", 1, 3, False)
            protected_end = len(capture.rows)
            disabled_cancel = cancel(capture, "echo-req", 1)
            prompts.append({"observed_monotonic_ns": 1, "request_id": "echo-req:vlm",
                            "messages": [{"role": "user", "content": "user-0"},
                                         {"role": "assistant", "content": "answer"},
                                         {"role": "user", "content": "echo-user"}],
                            "raw_sha256": digest(b"echo-req:vlm"),
                            "raw_base64": base64.b64encode(b"echo-req:vlm").decode()})
            (root / "echo-far-end-24k.pcm").write_bytes(echo_pcm)
            (root / "echo-microphone-16k.pcm").write_bytes(b"identical-mic")
            double_ids = {}
            double_indices = {}
            expected_history = [{"role": "user", "content": "user-0"},
                                {"role": "assistant", "content": "answer"},
                                {"role": "user", "content": "echo-user"},
                                {"role": "assistant", "content": "abc"}]
            for arm, segment in (("true", 2), ("false", 3)):
                response_id = f"double-{arm}"
                response(capture, response_id, segment, f"double-user-{arm}", "abcdef",
                         cancelled=True)
                ack(capture, response_id, segment, 3, False)
                double_indices[arm] = cancel(capture, response_id, segment)
                double_ids[arm] = response_id
                prompts.append({"observed_monotonic_ns": 1, "request_id": response_id + ":vlm",
                                "messages": expected_history + [
                                    {"role": "user", "content": f"double-user-{arm}"}],
                                "raw_sha256": digest(response_id.encode()),
                                "raw_base64": base64.b64encode(response_id.encode()).decode()})
                expected_history = expected_history + [
                    {"role": "user", "content": f"double-user-{arm}"},
                    {"role": "assistant", "content": "abc"}]
            echo_info = {
                "connection": "session-0-connection-0", "response_id": "echo-req",
                "protected_microphone_sha256": digest(b"identical-mic"),
                "disabled_microphone_sha256": digest(b"identical-mic"),
                "protected_far_end_sha256": digest(echo_pcm),
                "disabled_far_end_sha256": digest(echo_pcm),
                "protected_cutins": 0, "disabled_cutins": 1,
                "protected_start_capture_index": protected_start,
                "protected_end_capture_index": protected_end,
                "disabled_cancel_capture_index": disabled_cancel,
                "double_talk_protected_cancelled": True,
                "double_talk_disabled_cancelled": True,
                "double_talk_response_ids": double_ids,
                "double_talk_cancel_capture_indices": double_indices,
            }
        capture.write(root / "frames" / f"session-{session}-connection-0.jsonl")
    (root / "vlm-prompts.jsonl").write_bytes(b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode() for row in prompts))
    run = {
        "schema": "umminfer-realtime-network-live/v1",
        "concurrent_session_peak": 6,
        "connections_per_session": {str(i): 1 for i in range(6)},
        "final_health": {"status": "ok", "in_flight": 0},
        "cleanup": {"owned_processes_exited": True, "selected_gpus_idle": True},
        "physical_microphone_speaker": "NOT_COVERED",
        "perceptual_listening_quality": "NOT_COVERED",
        "late_callback_count": 1,
        "queue_contract": {"outbound_frames": 1024, "partial_events": 4096,
                           "unified_events": 4096, "max_audio_bytes": 64 * 1024 * 1024},
        "resource_bounds": {"server_rss_growth_max_bytes": 1024},
        "resource_usage": {"server": {"rss_growth_bytes": 10}},
        "echo_ab": echo_info,
    }
    (root / "run.json").write_text(json.dumps(run, sort_keys=True) + "\n")


class SchedulerTests(unittest.TestCase):
    def test_seeded_schedule_replays_and_covers_all_actions(self):
        first = net.make_plan(0x1234, 6, 3)
        second = net.make_plan(0x1234, 6, 3)
        self.assertEqual(first, second)
        self.assertNotEqual(first, net.make_plan(0x1235, 6, 3))
        self.assertEqual(len(first["sessions"]), 6)
        self.assertEqual(len(first["rounds"]), 3)
        for round_row in first["rounds"]:
            self.assertEqual(sorted(round_row["start_order"]), list(range(6)))
        scenarios = [name for row in first["sessions"] for name in row["scenarios"]]
        for name in net.SCENARIOS:
            self.assertGreaterEqual(scenarios.count(name), 2)
        for row in first["sessions"]:
            self.assertEqual(len(row["scenarios"]), 3)

    def test_driver_session_runner_keeps_full_round_schedule(self):
        spec = importlib.util.spec_from_file_location(
            "umminfer_realtime_network_live_test", ROOT / "tests/realtime_network_live.py")
        live = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = live
        spec.loader.exec_module(live)
        plan = net.make_plan(0x1234, 6, 3)
        runner = live.SessionRunner(2, plan, None, (0, 0), b"", ROOT)
        self.assertEqual(runner.scenarios, plan["sessions"][2]["scenarios"])
        self.assertEqual(runner.plan["rounds"], plan["rounds"])
        self.assertEqual(len(runner.scenarios), 3)

    def test_response_identity_returns_integer_event_index(self):
        spec = importlib.util.spec_from_file_location(
            "umminfer_realtime_network_live_index_test", ROOT / "tests/realtime_network_live.py")
        live = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = live
        spec.loader.exec_module(live)
        client = live.LiveClient.__new__(live.LiveClient)
        record = {"event_index": 7, "event": {"type": "audio.delta", "response_id": "req-7"}}
        class FakeSocket:
            def wait_for(self, predicate, after, timeout, description):
                self.args = (after, timeout, description)
                assert predicate(record["event"])
                return record
        client.ws = FakeSocket()
        self.assertEqual(client.response_identity(3), ("req-7", 7))
        self.assertIsInstance(client.response_identity(3)[1], int)
        normalized = live.normalize_vlm_messages([
            {"role": "user", "content": [{"type": "text", "text": "exact history"}]}])
        self.assertEqual(normalized, [{"role": "user", "content": "exact history"}])
        with self.assertRaises(ValueError):
            live.normalize_vlm_messages([
                {"role": "user", "content": [{"type": "image_url", "url": "not-text"}]}])

    def test_wire_decoder_handles_masking_and_fragment_boundaries(self):
        event = {"type": "x", "text": "pcm" * 100}
        for tx in (False, True):
            raw = frame(event, tx)
            fin, opcode, payload = net.decode_wire_frame(raw, "tx" if tx else "rx")
            self.assertTrue(fin)
            self.assertEqual(opcode, 1)
            self.assertEqual(json.loads(payload), event)
        with self.assertRaises(net.VerificationError):
            net.decode_wire_frame(frame(event, True), "rx")
        with self.assertRaises(net.VerificationError):
            net.decode_wire_frame(b"\x81\x7e\x00\x01x", "rx")


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="umminfer-network-test-")
        self.root = Path(self.temp.name)
        fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def rewrite_capture(self, session, mutate):
        path = self.root / "frames" / f"session-{session}-connection-0.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        mutate(rows)
        for index, row in enumerate(rows):
            row["index"] = index
        path.write_bytes(b"".join((json.dumps(row, sort_keys=True) + "\n").encode()
                                  for row in rows))

    def assert_rejected(self, mutate, session=0):
        self.rewrite_capture(session, mutate)
        with self.assertRaises(net.VerificationError):
            net.verify_evidence(self.root)

    def test_independent_fixture_passes(self):
        report = net.verify_evidence(self.root)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["sessions"], 6)
        self.assertEqual(report["verified_next_turn_prompts"], 9)
        self.assertEqual(report["cross_session_leaks"], 0)
        self.assertEqual(report["double_terminal_callbacks"], 0)

    def test_duplicate_terminal_rejected(self):
        def mutate(rows):
            original = next(row for row in rows if b"response.completed" in
                            base64.b64decode(row["raw_base64"]))
            duplicate = dict(original)
            duplicate["monotonic_ns"] = rows[-1]["monotonic_ns"] + 100
            rows.append(duplicate)
        self.assert_rejected(mutate)

    def test_decreasing_ack_rejected(self):
        def mutate(rows):
            forged = Capture()
            forged.rows = rows
            forged.clock = rows[-1]["monotonic_ns"]
            forged.add({"type": "response.playback_ack", "response_id": "echo-req",
                        "segment_id": 1, "text_cursor": 0, "final": False})
        self.assert_rejected(mutate)

    def test_cross_session_response_rejected(self):
        def mutate(rows):
            forged = Capture()
            forged.rows = rows
            forged.clock = rows[-1]["monotonic_ns"]
            forged.add({"type": "response.text.delta", "response_id": "echo-req",
                        "segment_id": 1, "text": "leak", "text_cursor": 4})
        self.assert_rejected(mutate, session=1)

    def test_stale_audio_after_cancel_rejected(self):
        def mutate(rows):
            forged = Capture()
            forged.rows = rows
            forged.clock = rows[-1]["monotonic_ns"]
            forged.add({"type": "audio.delta", "response_id": "echo-req", "segment_id": 1,
                        "format": "pcm16", "sample_rate": 24000, "channels": 1,
                        "audio": base64.b64encode(b"\x00\x00").decode(), "text_cursor": 3})
        self.assert_rejected(mutate)

    def test_nonmonotonic_capture_clock_rejected(self):
        def mutate(rows):
            rows[-1]["monotonic_ns"] = rows[-2]["monotonic_ns"]
        self.assert_rejected(mutate)

    def test_altered_next_turn_prompt_rejected(self):
        path = self.root / "vlm-prompts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["messages"][0]["content"] = "foreign-history"
        path.write_bytes(b"".join((json.dumps(row, sort_keys=True) + "\n").encode()
                                  for row in rows))
        with self.assertRaises(net.VerificationError):
            net.verify_evidence(self.root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
