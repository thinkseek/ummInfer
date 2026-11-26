#!/usr/bin/env python3
"""Deterministic raw-SSE capture/replay and tamper-rejection contracts.

Usage: python3 -I tests/sse_replay_test.py
Fixture clocks are logical ticks; one chunked-HTTP smoke records real monotonic
arrival ticks. No wall time, GPU, or serving import is used.
"""

import copy
from contextlib import contextmanager
import hashlib
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
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import narrative_rl_fixtures as fixture  # noqa: E402
import sse_replay_live as live_harness  # noqa: E402
import umminfer_rl_reward as reward_mod  # noqa: E402
import umminfer_rl_trajectory as traj  # noqa: E402
import umminfer_rl_worker as worker  # noqa: E402
import umminfer_sft_dataset as dataset  # noqa: E402
import umminfer_stage_c_pipeline as pipeline  # noqa: E402
from umminfer_sse_replay import (  # noqa: E402
    LogicalTickClock, RawSSECapture, ReplayError, replay_record,
    validate_replay_metadata,
)


def request_body():
    return {
        "model": "narrative-fixture", "stream": True,
        "messages": [{"role": "user", "content": "tell the lantern story"}],
        "narrative": {"enabled": True, "visual_mode": "auto", "max_slots": 2},
        "modalities": ["text", "image", "video"],
        "seed": 42,
    }


class ChunkReader:
    def __init__(self, chunks):
        self._chunks = iter(chunks)

    def read1(self, _size):
        return next(self._chunks, b"")


class CannedHandler(BaseHTTPRequestHandler):
    response_bytes = b""
    chunked = False

    def log_message(self, *_args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        if self.chunked:
            self.send_header("Transfer-Encoding", "chunked")
        else:
            self.send_header("Content-Length", str(len(self.response_bytes)))
        self.send_header("Connection", "close")
        self.end_headers()
        # Several writes exercise real HTTP capture; the exact read boundaries
        # themselves are recorded rather than assumed from these writes.
        cut = max(1, len(self.response_bytes) // 3)
        for start in range(0, len(self.response_bytes), cut):
            part = self.response_bytes[start:start + cut]
            if self.chunked:
                self.wfile.write(f"{len(part):X}\r\n".encode() + part + b"\r\n")
            else:
                self.wfile.write(part)
            self.wfile.flush()
        if self.chunked:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        self.close_connection = True


@contextmanager
def canned_server(raw, *, chunked=False):
    handler = type("Handler", (CannedHandler,),
                   {"response_bytes": raw, "chunked": chunked})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def capture_bytes(raw, root, *, events=None, chunk_size=13):
    recorder = traj.TrajectoryRecorder(Path(root) / "trajectories.jsonl")
    record = pipeline.sse_bytes_to_v2_record(
        raw, request_body=request_body(), recorder=recorder,
        session_id="replay-session", chunk_size=chunk_size,
        replay_clock=LogicalTickClock(start=7, step=3))
    return recorder, record


def replay_copy(record, source_recorder, root):
    recorder = traj.TrajectoryRecorder(Path(root) / "trajectories.jsonl")
    replayed = replay_record(
        record, recorder=recorder, artifact_dir=source_recorder.artifact_dir)
    return recorder, replayed


def strip_source(events):
    varying = {"source_event_index", "source_event_sha256", "source_event_tick"}
    return [{key: value for key, value in event.items() if key not in varying}
            for event in events]


class RawSSEReplayTests(unittest.TestCase):
    def test_live_http_vs_replay_byte_identity_and_explicit_anchor(self):
        raw = fixture.sse_bytes(fixture.linked_events())
        with tempfile.TemporaryDirectory(prefix="umm-sse-live-replay-") as tmp:
            root = Path(tmp)
            recorder = traj.TrajectoryRecorder(root / "live" / "trajectories.jsonl")
            with canned_server(raw) as url:
                live = traj.record_chat_completion(
                    url, request_body(), recorder, "replay-session",
                    capture_replay=True,
                    replay_clock=LogicalTickClock(start=11, step=5))
            replay = live["sse_replay"]
            self.assertEqual(replay["format_version"], 1)
            self.assertEqual(replay["clock"]["kind"], "logical_tick")
            self.assertEqual(
                (recorder.artifact_dir / replay["raw_ref"]).read_bytes(), raw)
            self.assertTrue(raw.endswith(b"data: [DONE]\n\n"))
            self.assertEqual(replay["frames"][-1]["kind"], "done")
            self.assertEqual(replay["event_count"], len(live["events"]))
            self.assertEqual(live["visual_slots"][1]["anchor_slot_index"], 0)
            self.assertTrue(live["visual_slots"][1]["anchor_explicit"])
            self.assertNotIn("source=slot-0", live["visual_slots"][1]["prompt"])
            ticks = [event["source_event_tick"] for event in live["events"]]
            self.assertEqual(ticks, sorted(ticks))
            self.assertTrue(all(len(event["source_event_sha256"]) == 64
                                for event in live["events"]))
            worker.validate_trajectory(recorder.jsonl_path)

            replayed_recorder, replayed = replay_copy(
                live, recorder, root / "replayed")
            self.assertEqual(replayed, live)
            self.assertEqual(replayed_recorder.jsonl_path.read_bytes(),
                             recorder.jsonl_path.read_bytes())
            self.assertEqual(
                (replayed_recorder.artifact_dir / replay["raw_ref"]).read_bytes(), raw)
            worker.validate_trajectory(replayed_recorder.jsonl_path)
            _, summary = pipeline.validate_trajectory_inputs(
                [replayed_recorder.jsonl_path], strict_artifacts=True)
            self.assertEqual(summary["replay_records"], 1)
            self.assertEqual(summary["verified_replay_refs"], 1)

            out = root / "cli" / "trajectories.jsonl"
            proc = subprocess.run(
                [sys.executable, "-I", str(ROOT / "python" / "umminfer_sse_replay.py"),
                 str(recorder.jsonl_path), "--out", str(out)],
                capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("RAW_SSE_REPLAY: PASS", proc.stdout)
            self.assertEqual(out.read_bytes(), recorder.jsonl_path.read_bytes())

    def test_live_harness_real_chunked_response_is_not_preconsumed(self):
        raw = fixture.sse_bytes(fixture.linked_events())
        with tempfile.TemporaryDirectory(prefix="umm-sse-live-chunked-") as tmp:
            recorder = traj.TrajectoryRecorder(Path(tmp) / "trajectories.jsonl")
            with canned_server(raw, chunked=True) as url:
                record, abort = live_harness.capture_public_sse(
                    url, request_body(), recorder, "chunked-session")
            self.assertIsNone(abort)
            self.assertEqual(record["terminal_state"], "completed")
            self.assertEqual(record["sse_replay"]["clock"]["kind"], "monotonic_ns")
            self.assertEqual(
                (recorder.artifact_dir / record["sse_replay"]["raw_ref"]).read_bytes(),
                raw)
            self.assertGreater(len(record["sse_replay"]["chunks"]), 1)

    def test_split_coalesced_crlf_comments_keepalives_and_plain_terminal(self):
        linked = fixture.sse_bytes(fixture.linked_events(delayed=True))
        raw = linked.replace(b"\n\n", b"\r\n\r\n")
        raw = b": keepalive\r\n\r\n" + raw + b""
        # Insert a comment and an empty keepalive between frames without
        # changing any public data payload.
        cut = raw.find(b"data:", 1)
        raw = raw[:cut] + b"\r\n: second keepalive\r\n\r\n" + raw[cut:]
        with tempfile.TemporaryDirectory(prefix="umm-sse-framing-") as tmp:
            root = Path(tmp)
            recorder = traj.TrajectoryRecorder(root / "split" / "trajectories.jsonl")
            split = pipeline.sse_stream_to_v2_record(
                ChunkReader(raw[index:index + 1] for index in range(len(raw))),
                request_body=request_body(), recorder=recorder,
                session_id="replay-session", replay_clock=LogicalTickClock())
            self.assertGreater(len(split["sse_replay"]["chunks"]), 100)
            self.assertEqual(split["sse_replay"]["frames"][0]["kind"], "comment")
            self.assertIn("keepalive", [frame["kind"] for frame in
                                        split["sse_replay"]["frames"]])
            replayed_recorder, replayed = replay_copy(split, recorder, root / "replay")
            self.assertEqual(replayed, split)
            self.assertEqual(replayed_recorder.jsonl_path.read_bytes(),
                             recorder.jsonl_path.read_bytes())

            coalesced_recorder, coalesced = capture_bytes(
                raw, root / "coalesced", chunk_size=len(raw) + 1)
            self.assertEqual(len(coalesced["sse_replay"]["chunks"]), 1)
            self.assertEqual(strip_source(coalesced["events"]), strip_source(split["events"]))
            self.assertEqual(coalesced["artifacts"], split["artifacts"])

            plain_chunks = [
                {"id": "plain-1", "choices": [{"delta": {"content": "plain"},
                                               "finish_reason": None}]},
                {"id": "plain-1", "choices": [{"delta": {},
                                               "finish_reason": "stop"}]},
            ]
            plain_raw = b"".join(
                b"data: " + json.dumps(chunk).encode() + b"\n\n"
                for chunk in plain_chunks) + b"data: [DONE]\n\n"
            plain_recorder, plain = capture_bytes(plain_raw, root / "plain")
            self.assertEqual(plain["events"][-1]["type"], "complete")
            self.assertEqual(plain["events"][-1]["source_event_index"],
                             plain["sse_replay"]["done_frame"])
            _, plain_replayed = replay_copy(plain, plain_recorder, root / "plain-replay")
            self.assertEqual(plain_replayed, plain)

    def test_cancelled_partial_artifacts_replay_identically(self):
        with tempfile.TemporaryDirectory(prefix="umm-sse-cancel-") as tmp:
            recorder, cancelled = capture_bytes(
                fixture.sse_bytes(fixture.cancelled_events()), Path(tmp) / "live")
            self.assertEqual(cancelled["terminal_state"], "cancelled")
            self.assertEqual([item["artifact_id"] for item in cancelled["artifacts"]],
                             ["art-video-1"])
            replayed_recorder, replayed = replay_copy(
                cancelled, recorder, Path(tmp) / "replayed")
            self.assertEqual(replayed, cancelled)
            self.assertEqual(replayed_recorder.jsonl_path.read_bytes(),
                             recorder.jsonl_path.read_bytes())

    def test_malformed_truncated_and_post_done_streams_fail_cleanly(self):
        capture = RawSSECapture(clock=LogicalTickClock())
        capture.feed(b"data: {not-json}\n\n")
        with self.assertRaisesRegex(ReplayError, "before final"):
            capture.finish()

        capture = RawSSECapture(clock=LogicalTickClock())
        capture.feed(b"data: [DONE]\n\n")
        with self.assertRaisesRegex(ReplayError, "follows terminal"):
            capture.feed(b"data: {}\n\n")

        capture = RawSSECapture(clock=LogicalTickClock())
        capture.feed(b"data: [DONE]\r")
        with self.assertRaisesRegex(ReplayError, "truncated SSE frame"):
            capture.finish()

        capture = RawSSECapture(clock=LogicalTickClock())
        with self.assertRaisesRegex(ReplayError, "not valid UTF-8"):
            capture.feed(b"data: \xff\n\n")

        with tempfile.TemporaryDirectory(prefix="umm-sse-malformed-") as tmp:
            for index, payload in enumerate((b"{not-json}", b"null", b"[]")):
                with self.subTest(payload=payload):
                    recorder = traj.TrajectoryRecorder(Path(tmp) / f"bad-{index}.jsonl")
                    with self.assertRaisesRegex(ReplayError, "frame 0"):
                        pipeline.sse_bytes_to_v2_record(
                            b"data: " + payload + b"\n\ndata: [DONE]\n\n",
                            request_body=request_body(), recorder=recorder,
                            session_id="bad")
                    self.assertFalse(recorder.jsonl_path.exists())

    def test_duplicate_out_of_order_events_and_nonmonotonic_ticks_rejected(self):
        with tempfile.TemporaryDirectory(prefix="umm-sse-sequence-") as tmp:
            for name, events in (
                ("duplicate", [fixture.envelope(0, "vlm_delta", "a"),
                               fixture.envelope(0, "vlm_delta", "b"),
                               fixture.envelope(1, "complete", "ab", legs=[])]),
                ("out-of-order", [fixture.envelope(1, "vlm_delta", "a"),
                                  fixture.envelope(0, "vlm_delta", "b"),
                                  fixture.envelope(2, "complete", "ab", legs=[])]),
            ):
                with self.subTest(name=name):
                    recorder = traj.TrajectoryRecorder(Path(tmp) / (name + ".jsonl"))
                    with self.assertRaisesRegex(ReplayError, "server event sequence"):
                        pipeline.sse_bytes_to_v2_record(
                            fixture.sse_bytes(events), request_body=request_body(),
                            recorder=recorder, session_id=name)
            capture = RawSSECapture()
            capture.feed(b": first\n\n", tick=2)
            with self.assertRaisesRegex(ReplayError, "not monotonic"):
                capture.feed(b": second\n\n", tick=1)

    def test_tampered_raw_bytes_hashes_sequence_timing_and_terminal_rejected(self):
        with tempfile.TemporaryDirectory(prefix="umm-sse-tamper-") as tmp:
            recorder, valid = capture_bytes(
                fixture.sse_bytes(fixture.linked_events()), Path(tmp))
            raw_path = recorder.artifact_dir / valid["sse_replay"]["raw_ref"]
            original_raw = raw_path.read_bytes()

            def recompute_events(record):
                record["sse_replay"]["normalized_events_sha256"] = hashlib.sha256(
                    traj.canonical_json(record["events"])).hexdigest()

            mutations = {}
            for name, path in (
                ("raw-sha", ("sse_replay", "raw_sha256")),
                ("chunk-hash", ("sse_replay", "chunks", 0, "sha256")),
                ("frame-hash", ("sse_replay", "frames", 0, "sha256")),
                ("frame-tick", ("sse_replay", "frames", 0, "tick")),
                ("event-source-hash", ("events", 0, "source_event_sha256")),
                ("event-source-tick", ("events", 0, "source_event_tick")),
            ):
                def mutate(record, path=path):
                    target = record
                    for key in path[:-1]:
                        target = target[key]
                    target[path[-1]] = 0 if path[-1].endswith("sha256") \
                        else target[path[-1]] + 1
                    if path[0] == "events":
                        recompute_events(record)
                mutations[name] = mutate

            def out_of_order(record):
                record["events"][0]["server_seq"] = 1
                record["events"][1]["server_seq"] = 0
                recompute_events(record)
            mutations["server-sequence"] = out_of_order

            def terminal(record):
                record["terminal_state"] = "cancelled"
                record["sse_replay"]["terminal_state"] = "cancelled"
            mutations["terminal"] = terminal

            def drop_done(record):
                record["sse_replay"]["frames"].pop()
                record["sse_replay"]["done_frame"] -= 1
            mutations["done"] = drop_done

            def rebind_data_frame(record):
                target = record["sse_replay"]["frames"][1]
                record["events"][0]["source_event_index"] = 1
                record["events"][0]["source_event_sha256"] = target["sha256"]
                record["events"][0]["source_event_tick"] = target["tick"]
                recompute_events(record)
            mutations["semantic-frame-rebinding"] = rebind_data_frame

            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    bad = copy.deepcopy(valid)
                    mutate(bad)
                    recorder.jsonl_path.write_bytes(traj.canonical_json(bad) + b"\n")
                    with self.assertRaises((ReplayError, ValueError)):
                        worker.validate_trajectory(recorder.jsonl_path)
                    ok, category, message, _ = dataset.verify_record_integrity(
                        bad, artifact_dir=recorder.artifact_dir,
                        require_artifact_dir=True)
                    self.assertFalse(ok, name)
                    self.assertTrue(category and message, name)
                    with self.assertRaises((ReplayError, ValueError)):
                        replay_record(bad, recorder=traj.TrajectoryRecorder(
                            Path(tmp) / (name + "-replay.jsonl")),
                            artifact_dir=recorder.artifact_dir)

            recorder.jsonl_path.write_bytes(traj.canonical_json(valid) + b"\n")
            raw_path.write_bytes(b"X" + original_raw[1:])
            with self.assertRaisesRegex((ReplayError, ValueError), "SHA-256"):
                worker.validate_trajectory(recorder.jsonl_path)
            with self.assertRaisesRegex(ReplayError, "SHA-256"):
                reward_mod.apply_reward(recorder.jsonl_path, reward_mod.NullReward())
            with self.assertRaisesRegex(ReplayError, "SHA-256"):
                replay_record(valid, recorder=traj.TrajectoryRecorder(
                    Path(tmp) / "raw-replay.jsonl"),
                    artifact_dir=recorder.artifact_dir)
            raw_path.write_bytes(original_raw)
            worker.validate_trajectory(recorder.jsonl_path)

            bad_tick = copy.deepcopy(valid)
            bad_tick["events"][0]["source_event_tick"] += 1
            recompute_events(bad_tick)
            with self.assertRaisesRegex(ReplayError, "source tick"):
                reward_mod.CompositeUmmReward().score(bad_tick)

    def test_comment_rebinding_missing_raw_and_idless_request_tamper_rejected(self):
        with tempfile.TemporaryDirectory(prefix="umm-sse-binding-") as tmp:
            root = Path(tmp)
            raw = b": keepalive\n\n" + fixture.sse_bytes(fixture.linked_events())
            recorder, valid = capture_bytes(raw, root / "live")
            comment = valid["sse_replay"]["frames"][0]
            bad = copy.deepcopy(valid)
            bad["events"][0]["source_event_index"] = 0
            bad["events"][0]["source_event_sha256"] = comment["sha256"]
            bad["events"][0]["source_event_tick"] = comment["tick"]
            bad["sse_replay"]["normalized_events_sha256"] = hashlib.sha256(
                traj.canonical_json(bad["events"])).hexdigest()
            recorder.jsonl_path.write_bytes(traj.canonical_json(bad) + b"\n")
            with self.assertRaisesRegex(ValueError, "non-data/comment"):
                worker.validate_trajectory(recorder.jsonl_path)

            recorder.jsonl_path.write_bytes(traj.canonical_json(valid) + b"\n")
            raw_path = recorder.artifact_dir / valid["sse_replay"]["raw_ref"]
            raw_path.unlink()
            with self.assertRaisesRegex(ReplayError, "artifact is missing"):
                reward_mod.apply_reward(recorder.jsonl_path, reward_mod.NullReward())
            with self.assertRaisesRegex(ValueError, "artifact is missing"):
                pipeline.validate_trajectory_inputs([recorder.jsonl_path])

            plain_chunks = [{"choices": [{"delta": {"content": "idless"},
                                          "finish_reason": "stop"}]}]
            plain_raw = b"".join(
                b"data: " + json.dumps(chunk).encode() + b"\n\n"
                for chunk in plain_chunks) + b"data: [DONE]\n\n"
            plain_recorder, plain = capture_bytes(plain_raw, root / "idless")
            plain["request_id"] = "rollout-" + "0" * 16
            with self.assertRaisesRegex(ReplayError, "not byte-identical.*request_id"):
                replay_record(plain, recorder=traj.TrajectoryRecorder(
                    root / "idless-replay" / "trajectories.jsonl"),
                    artifact_dir=plain_recorder.artifact_dir)

    def test_old_v2_records_without_replay_remain_valid(self):
        with tempfile.TemporaryDirectory(prefix="umm-sse-compat-") as tmp:
            recorder = traj.TrajectoryRecorder(Path(tmp) / "legacy-v2.jsonl")
            with canned_server(fixture.sse_bytes(fixture.linked_events())) as url:
                record = traj.record_chat_completion(
                    url, request_body(), recorder, "old-v2")
            self.assertNotIn("sse_replay", record)
            self.assertNotIn("source_event_index", record["events"][0])
            worker.validate_trajectory(recorder.jsonl_path)
            ok, category, message, _ = dataset.verify_record_integrity(record)
            self.assertTrue(ok, (category, message))


if __name__ == "__main__":
    unittest.main(verbosity=2)
