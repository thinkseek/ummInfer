#!/usr/bin/env python3
"""CPU-only deterministic tests for multimodal trajectory schema v2.

Usage: python3 -I tests/rl_architecture_test.py
"""

import base64
import copy
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import narrative_rl_fixtures as fixture  # noqa: E402
import umminfer_rl_reward as reward_mod  # noqa: E402
import umminfer_rl_trajectory as traj  # noqa: E402
import umminfer_rl_worker as worker  # noqa: E402


class CannedHandler(BaseHTTPRequestHandler):
    response_bytes = b""
    content_type = "text/event-stream"

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", self.content_type)
        self.send_header("Content-Length", str(len(self.response_bytes)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(self.response_bytes)
        self.close_connection = True


@contextmanager
def canned_server(response_bytes, content_type="text/event-stream"):
    handler = type("Handler", (CannedHandler,), {
        "response_bytes": response_bytes, "content_type": content_type,
    })
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def data_url(mime, raw):
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def narrative_body():
    return {
        "model": "narrative-fixture", "stream": True,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "tell the lantern story"},
            {"type": "input_audio", "input_audio": {
                "format": "wav", "data": base64.b64encode(fixture.WAV).decode()}},
            {"type": "image_url", "image_url": {
                "url": data_url("image/png", fixture.PNG)}},
        ]}],
        "narrative": {"enabled": True, "visual_mode": "auto", "max_slots": 3},
        "modalities": ["text", "image", "audio", "video"],
        "image": {"steps": 4, "image": data_url("image/png", fixture.PNG)},
        "audio": {"voice": "vivian", "stream": True,
                  "ref_audio": base64.b64encode(fixture.WAV).decode()},
        "video": {"num_frames": 13, "steps": 4},
        "seed": 42,
    }


def plain_body():
    return {"model": "plain-fixture", "stream": True,
            "messages": [{"role": "user", "content": "answer exactly"}]}


def record_response(root, events, body=None):
    root = Path(root)
    recorder = traj.TrajectoryRecorder(root / "rollout.jsonl")
    response = fixture.sse_bytes(events)
    with canned_server(response) as base_url:
        record = traj.record_chat_completion(
            base_url, copy.deepcopy(body or narrative_body()), recorder,
            "narrative-session")
    return recorder, record


class NarrativeReward(reward_mod.Reward):
    name = "export-only-narrative"

    def score(self, record):
        # Deliberately reads only the JSON object: no server, model, or CAS IO.
        return (len(record["clean_text"]) + len(record["visual_slots"])
                + len(record["artifacts"])
                + (1 if record["terminal_state"] == "completed" else 0))


class TrajectoryV2Tests(unittest.TestCase):
    def test_two_recordings_are_byte_identical_and_media_is_content_addressed(self):
        with tempfile.TemporaryDirectory(prefix="umm-rl-v2-determinism-") as tmp:
            first_recorder, first = record_response(Path(tmp) / "first",
                                                    fixture.narrative_events())
            second_recorder, second = record_response(Path(tmp) / "second",
                                                      fixture.narrative_events())
            self.assertEqual(first_recorder.jsonl_path.read_bytes(),
                             second_recorder.jsonl_path.read_bytes())
            first_files = sorted(path.name for path in
                                 first_recorder.artifact_dir.iterdir())
            second_files = sorted(path.name for path in
                                  second_recorder.artifact_dir.iterdir())
            self.assertEqual(first_files, second_files)
            for name in first_files:
                self.assertEqual((first_recorder.artifact_dir / name).read_bytes(),
                                 (second_recorder.artifact_dir / name).read_bytes())

            self.assertEqual(first["schema_version"], 2)
            self.assertEqual(first["clean_text"], fixture.CLEAN_TEXT)
            self.assertNotIn("[[", first["clean_text"])
            self.assertEqual(len(first["visual_slots"]), 3)
            self.assertEqual(first["leg_errors"][0]["error"], "render denied")
            self.assertEqual(first["terminal_legs"][2]["state"], "failed")
            self.assertEqual({item["media_type"] for item in first["artifacts"]},
                             {"image", "audio", "video"})
            self.assertEqual(first["artifacts"][0]["chunk_count"], 2)
            for output in first["artifacts"]:
                raw = (first_recorder.artifact_dir / output["ref"]).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), output["sha256"])
                self.assertEqual(len(raw), output["bytes"])
            self.assertNotIn("data", first["events"][2]["artifact"])
            self.assertEqual(first["events"][2]["artifact"]["chunk_bytes"],
                             len(fixture.PNG[:len(fixture.PNG) // 2]))
            serialized = first_recorder.jsonl_path.read_text()
            self.assertNotIn("data:image/", serialized)
            self.assertNotIn("data:audio/", serialized)
            self.assertNotIn('"data":', serialized)
            self.assertIsInstance(first["request_params"]["image"]["image"], dict)
            self.assertIsInstance(first["request_params"]["audio"]["ref_audio"], dict)
            worker.validate_trajectory(first_recorder.jsonl_path)

    def test_bad_final_hash_is_failed_and_never_published(self):
        events = copy.deepcopy(fixture.narrative_events())
        events[7]["artifact"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory(prefix="umm-rl-v2-bad-hash-") as tmp:
            recorder, record = record_response(tmp, events, body=plain_body())
            self.assertEqual(record["terminal_state"], "failed")
            self.assertIn("artifact SHA-256 mismatch", record["error"])
            self.assertEqual(record["artifacts"], [])
            image_path = recorder.artifact_dir / (
                hashlib.sha256(fixture.PNG).hexdigest() + ".png")
            self.assertFalse(image_path.exists())
            worker.validate_trajectory(recorder.jsonl_path)

    def test_completed_turn_rejects_missing_artifact_chunk(self):
        events = copy.deepcopy(fixture.narrative_events())
        del events[7]
        with tempfile.TemporaryDirectory(prefix="umm-rl-v2-missing-chunk-") as tmp:
            _, record = record_response(tmp, events)
            self.assertEqual(record["terminal_state"], "failed")
            self.assertIn("missing artifact chunks", record["error"])

    def test_cancelled_turn_keeps_terminal_and_drops_partial_outputs(self):
        with tempfile.TemporaryDirectory(prefix="umm-rl-v2-cancel-") as tmp:
            recorder, record = record_response(
                tmp, fixture.cancelled_events(), body=plain_body())
            self.assertEqual(record["terminal_state"], "cancelled")
            self.assertIsNone(record["error"])
            self.assertEqual(record["events"][-1]["type"], "cancelled")
            self.assertEqual(record["terminal_legs"][0]["state"], "aborted")
            self.assertEqual([item["artifact_id"] for item in record["artifacts"]],
                             ["art-video-1"])
            self.assertEqual(len(record["visual_slots"]), 3)
            self.assertEqual(len(record["leg_errors"]), 1)
            worker.validate_trajectory(recorder.jsonl_path)
            self.assertTrue((recorder.artifact_dir /
                             record["artifacts"][0]["ref"]).is_file())

    def test_worker_rejects_slot_sequence_hash_and_terminal_tampering(self):
        with tempfile.TemporaryDirectory(prefix="umm-rl-v2-tamper-") as tmp:
            recorder, valid = record_response(tmp, fixture.narrative_events())

            def change_slot(record):
                record["visual_slots"][0]["slot_index"] = 9

            def change_event_seq(record):
                record["events"][1]["seq"] = 99

            def change_server_seq(record):
                record["events"][1]["server_seq"] = 99

            def change_output_hash(record):
                record["output_sha256"] = "0" * 64

            def change_clean_text(record):
                record["clean_text"] += " tampered"

            def change_artifact_hash(record):
                record["artifacts"][0]["sha256"] = "0" * 64

            def change_chunk_hash(record):
                record["events"][2]["artifact"]["chunk_sha256"] = "0" * 64

            def change_terminal(record):
                record["terminal_state"] = "cancelled"

            def change_legs(record):
                record["terminal_legs"][0]["state"] = "failed"

            def embed_media(record):
                record["events"][2]["artifact"]["data"] = "AA=="

            def embed_request_media(record):
                record["request_params"]["image"]["image"] = "data:image/png;base64,AA=="

            mutations = {
                "slot": change_slot, "event sequence": change_event_seq,
                "server sequence": change_server_seq,
                "output hash": change_output_hash,
                "clean text": change_clean_text,
                "artifact hash": change_artifact_hash,
                "chunk hash": change_chunk_hash,
                "terminal": change_terminal, "terminal legs": change_legs,
                "embedded artifact": embed_media,
                "embedded request media": embed_request_media,
            }
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    bad = copy.deepcopy(valid)
                    mutate(bad)
                    recorder.jsonl_path.write_text(
                        json.dumps(bad, sort_keys=True) + "\n")
                    with self.assertRaises(ValueError):
                        worker.validate_trajectory(recorder.jsonl_path)

            recorder.jsonl_path.write_text(
                json.dumps(valid, sort_keys=True) + "\n")
            output_path = recorder.artifact_dir / valid["artifacts"][0]["ref"]
            original = output_path.read_bytes()
            output_path.write_bytes(b"X" * len(original))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                worker.validate_trajectory(recorder.jsonl_path)

    def test_schema_v1_is_accepted_without_inventing_v2_fields(self):
        legacy = {
            "schema_version": 1, "session_id": "legacy", "request_id": "old-1",
            "turn_index": 0, "endpoint": "/v1/chat/completions", "inputs": [],
            "request_params": {"stream": True},
            "events": [{"seq": 0, "type": "output_delta", "text": "legacy"}],
            "terminal_state": "completed", "error": None,
            "output_text": "legacy",
            "output_sha256": hashlib.sha256(b"legacy").hexdigest(),
        }
        with tempfile.TemporaryDirectory(prefix="umm-rl-v1-compat-") as tmp:
            path = Path(tmp) / "legacy.jsonl"
            path.write_text(json.dumps(legacy) + "\n")
            records = worker.validate_trajectory(path)
            self.assertEqual(records, [legacy])
            self.assertNotIn("clean_text", records[0])
            for version in (0, 3, True, None):
                bad = dict(legacy, schema_version=version)
                path.write_text(json.dumps(bad) + "\n")
                with self.assertRaises(ValueError):
                    worker.validate_trajectory(path)

    def test_reward_is_computed_from_export_alone(self):
        with tempfile.TemporaryDirectory(prefix="umm-rl-v2-reward-") as tmp:
            recorder, record = record_response(tmp, fixture.narrative_events())
            expected = (len(record["clean_text"]) + len(record["visual_slots"])
                        + len(record["artifacts"]) + 1)
            reward_mod.apply_reward(recorder.jsonl_path, NarrativeReward())
            scored = worker.validate_trajectory(recorder.jsonl_path)[0]
            self.assertEqual(scored["reward"]["score"], expected)
            self.assertEqual(scored["reward"]["name"], "export-only-narrative")
            reward_mod.apply_reward(recorder.jsonl_path, reward_mod.NullReward())
            self.assertIsNone(worker.validate_trajectory(
                recorder.jsonl_path)[0]["reward"]["score"])

    def test_plain_sse_nonstream_and_realtime_remain_recordable(self):
        plain_chunks = [
            {"id": "plain-1", "choices": [{"delta": {"content": "plain"},
                                           "finish_reason": None}]},
            {"id": "plain-1", "choices": [{"delta": {"content": " text"},
                                           "finish_reason": "stop"}]},
        ]
        response = b"".join(
            b"data: " + json.dumps(chunk).encode() + b"\n\n"
            for chunk in plain_chunks) + b"data: [DONE]\n\n"
        with tempfile.TemporaryDirectory(prefix="umm-rl-v2-plain-") as tmp:
            recorder = traj.TrajectoryRecorder(Path(tmp) / "plain.jsonl")
            with canned_server(response) as base_url:
                record = traj.record_chat_completion(
                    base_url, plain_body(), recorder, "plain")
            self.assertEqual(record["clean_text"], "plain text")
            self.assertEqual(record["events"][-1]["type"], "complete")
            worker.validate_trajectory(recorder.jsonl_path)

            realtime = traj.TrajectoryRecorder(Path(tmp) / "realtime.jsonl")
            completed = traj.record_realtime(realtime, "rt", "rt-1", [
                {"direction": "outbound", "type": "transcription.completed",
                 "text": "transcript", "is_final": True},
            ])
            self.assertEqual(completed["clean_text"], "transcript")
            cancelled = traj.record_realtime(realtime, "rt", "rt-2", [
                {"direction": "inbound", "type": "input_audio_buffer.append"},
            ])
            self.assertEqual(cancelled["terminal_state"], "cancelled")
            worker.validate_trajectory(realtime.jsonl_path)

    def test_nonstream_mixed_artifacts_are_verified(self):
        body = {
            "id": "mixed-1", "choices": [{"message": {
                "role": "assistant", "content": "mixed"}, "finish_reason": "stop"}],
            "artifacts": [{
                "artifact_id": "mixed-image", "media_type": "image",
                "mime_type": "image/png", "format": "png",
                "encoding": "base64", "data": base64.b64encode(fixture.PNG).decode(),
                "bytes": len(fixture.PNG),
                "sha256": hashlib.sha256(fixture.PNG).hexdigest(),
            }], "legs": [],
        }
        with tempfile.TemporaryDirectory(prefix="umm-rl-v2-nonstream-") as tmp:
            recorder = traj.TrajectoryRecorder(Path(tmp) / "mixed.jsonl")
            with canned_server(json.dumps(body).encode(), "application/json") as base_url:
                record = traj.record_chat_completion(
                    base_url, dict(plain_body(), stream=False), recorder, "mixed")
            self.assertEqual(record["artifacts"][0]["bytes"], len(fixture.PNG))
            self.assertNotIn("data", record["events"][0]["artifact"])
            worker.validate_trajectory(recorder.jsonl_path)

    def test_serving_does_not_import_rl_or_optimizer_modules(self):
        for directory in (ROOT / "src", ROOT / "include"):
            for path in directory.rglob("*"):
                if path.is_file():
                    self.assertNotIn("umminfer_rl_", path.read_text(errors="ignore"),
                                     str(path))
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)


class CompositeUmmRewardTests(unittest.TestCase):
    def test_export_only_components_and_byte_determinism(self):
        with tempfile.TemporaryDirectory(prefix="umm-rl-composite-") as tmp:
            recorder, record = record_response(tmp, fixture.narrative_events())
            record["ablation"] = {
                "expected_visual_modalities": ["image", "video"],
                "latency": {"ttft_ms": 500, "terminal_ms": 15000},
            }
            recorder.jsonl_path.write_text(
                json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            reward = reward_mod.CompositeUmmReward()
            exported = json.loads(recorder.jsonl_path.read_text())
            score, components = reward.evaluate(exported)
            self.assertEqual(components["routing_f1"], 1.0)
            self.assertAlmostEqual(components["prose_visual_balance"], 0.8)
            self.assertEqual(components["latency_penalty"], 0.5)
            self.assertEqual(components["barge_in_stability"], 1.0)
            self.assertFalse(components["barge_in_observed"])
            self.assertAlmostEqual(score, 0.84)

            reward_mod.apply_reward(recorder.jsonl_path, reward)
            first_bytes = recorder.jsonl_path.read_bytes()
            scored = worker.validate_trajectory(recorder.jsonl_path)[0]
            self.assertAlmostEqual(scored["reward"]["score"], 0.84)
            self.assertEqual(scored["reward"]["components"], components)
            reward_mod.apply_reward(recorder.jsonl_path, reward)
            self.assertEqual(recorder.jsonl_path.read_bytes(), first_bytes)

            # Scoring needs no CAS, live server, model, or external observation.
            hidden_artifacts = Path(tmp) / "hidden-artifacts"
            recorder.artifact_dir.rename(hidden_artifacts)
            reward_mod.apply_reward(recorder.jsonl_path, reward)
            self.assertEqual(recorder.jsonl_path.read_bytes(), first_bytes)
            hidden_artifacts.rename(recorder.artifact_dir)
            worker.validate_trajectory(recorder.jsonl_path)

    def test_missing_coverage_and_invalid_observations(self):
        with tempfile.TemporaryDirectory(prefix="umm-rl-composite-coverage-") as tmp:
            _, record = record_response(tmp, fixture.narrative_events())
            reward = reward_mod.CompositeUmmReward()
            self.assertIsNone(reward.score(record))
            record["ablation"] = {"expected_visual_modalities": ["image", "video"]}
            self.assertIsNone(reward.score(record))
            record["ablation"]["latency"] = {}
            self.assertIsNone(reward.score(record))
            for value in (-1, True, "1", float("nan"), float("inf")):
                with self.subTest(value=value):
                    bad = copy.deepcopy(record)
                    bad["ablation"]["latency"] = {"terminal_ms": value}
                    with self.assertRaises(ValueError):
                        reward.score(bad)
            with self.assertRaisesRegex(ValueError, "unknown observation"):
                reward.score(dict(record, ablation={
                    "expected_visual_modalities": [],
                    "latency": {"terminal": 1},
                }))
            with self.assertRaisesRegex(ValueError, "schema v2"):
                reward.score(dict(record, schema_version=1))
            with self.assertRaises(ValueError):
                reward_mod.CompositeUmmReward(weights={"unknown": 1})
            with self.assertRaises(ValueError):
                reward_mod.CompositeUmmReward(weights={"routing_f1": -1})
            with self.assertRaises(ValueError):
                reward_mod.CompositeUmmReward(latency_budget_ms={"terminal_ms": 0})

    def test_barge_in_stability_comes_from_realtime_events(self):
        with tempfile.TemporaryDirectory(prefix="umm-rl-composite-barge-") as tmp:
            recorder = traj.TrajectoryRecorder(Path(tmp) / "realtime.jsonl")
            stable = traj.record_realtime(
                recorder, "duplex", "stable", [
                    {"direction": "outbound",
                     "type": "input_audio_buffer.speech_started"},
                    {"direction": "outbound", "type": "response.cancelled"},
                ], terminal_state="cancelled")
            stable["ablation"] = {
                "expected_visual_modalities": [],
                "latency": {"barge_in_ms": 125},
            }
            _, stable_components = reward_mod.CompositeUmmReward().evaluate(stable)
            self.assertTrue(stable_components["barge_in_observed"])
            self.assertEqual(stable_components["barge_in_stability"], 1.0)
            self.assertEqual(stable_components["latency_penalty"], 0.5)

            unstable = traj.record_realtime(
                recorder, "duplex", "unstable", [
                    {"direction": "outbound",
                     "type": "input_audio_buffer.speech_started"},
                    {"direction": "outbound", "type": "transcription.completed",
                     "text": "continued after interruption", "is_final": True},
                ])
            unstable["ablation"] = copy.deepcopy(stable["ablation"])
            _, unstable_components = reward_mod.CompositeUmmReward().evaluate(unstable)
            self.assertTrue(unstable_components["barge_in_observed"])
            self.assertEqual(unstable_components["barge_in_stability"], 0.0)
            worker.validate_trajectory(recorder.jsonl_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
