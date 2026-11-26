#!/usr/bin/env python3
"""Deterministic fake-endpoint tests for the online Stage C GRPO contract.

Usage: python3 -I tests/online_rollout_test.py
No GPU, optimizer, real model, or resident service is used.
"""

import builtins
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import narrative_rl_fixtures as fixture  # noqa: E402
import umminfer_joint_grpo_live as frozen_grpo  # noqa: E402
import umminfer_online_rollout as online  # noqa: E402
import umminfer_rl_trajectory as traj  # noqa: E402
import umminfer_sft_dataset as dataset  # noqa: E402
from umminfer_rl_reward import load_reward  # noqa: E402
from umminfer_sse_replay import LogicalTickClock, replay_record, validate_replay_metadata  # noqa: E402

PLAN_PATH = ROOT / "configs/training/online_grpo_rollout_fixture.json"


def fixture_plan():
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def response_events(group_index, rollout_index):
    """Full, partial, and redundant routes with exact terminal/CAS evidence."""
    first = f"Case {group_index} candidate {rollout_index} opens. "
    second = f"Candidate {rollout_index} closes with coherent prose."
    routes = [("image", "video"), ("image",), ("image", "image")][rollout_index]
    events = [fixture.envelope(0, "vlm_delta", first)]
    slots = []
    artifacts = []
    for slot_index, modality in enumerate(routes):
        offset = len(first) if slot_index == 0 else len(first + second)
        continuity = "i2v_chain" if rollout_index == 0 else None
        explicit = modality == "video" and rollout_index == 0
        slot = fixture.visual_slot(
            len(events), slot_index, modality,
            f"group {group_index} candidate {rollout_index} {modality} {slot_index}",
            offset, continuity_mode=continuity,
            anchor_slot_index=0 if explicit else None,
            anchor_explicit=True if explicit else None)
        events.append(slot)
        slots.append(slot)
        raw = (fixture.PNG if modality == "image" else fixture.MP4) \
            + f"-g{group_index}-r{rollout_index}-s{slot_index}".encode("ascii")
        image_sha = None
        if explicit:
            image_raw = fixture.PNG + f"-g{group_index}-r{rollout_index}-s0".encode("ascii")
            import hashlib
            image_sha = hashlib.sha256(image_raw).hexdigest()
        artifact = fixture.artifact(
            len(events), f"art-g{group_index}-r{rollout_index}-s{slot_index}",
            modality, "image/png" if modality == "image" else "video/mp4",
            raw, modality, slot_id=f"slot-{slot_index}", slot_index=slot_index,
            anchor_slot_index=0 if explicit else None,
            anchor_explicit=True if explicit else None,
            anchor_sha256=image_sha)
        events.append(artifact)
        artifacts.append(artifact)
    events.append(fixture.envelope(len(events), "vlm_delta", second))
    legs = []
    for slot, artifact in zip(slots, artifacts):
        leg = {
            "stage": slot["media_type"], "slot_id": slot["slot_id"],
            "slot_index": slot["slot_index"], "state": "completed",
        }
        if artifact["artifact"].get("anchor_explicit"):
            leg.update({
                "anchor_slot_index": 0, "anchor_explicit": True,
                "anchor_sha256": artifact["artifact"]["anchor_sha256"],
            })
        legs.append(leg)
    events.append(fixture.envelope(
        len(events), "complete", first + second, legs=legs))
    return events


def response_bytes(events, item, request_id, behavior=None, *, inner_drift=False):
    frames = []
    for index, original in enumerate(events):
        unified = copy.deepcopy(original)
        unified["request_id"] = "stale-inner-request" if inner_drift else request_id
        unified["session_id"] = "stale-inner-session" if inner_drift else item["session_id"]
        chunk = {
            "id": request_id, "object": "chat.completion.chunk", "created": 0,
            "model": item["body"]["model"],
            "choices": [{
                "index": 0,
                "delta": {"content": unified.get("text", "")
                          if unified["kind"] == "vlm_delta" else ""},
                "finish_reason": None,
            }],
            "unified": unified,
        }
        if behavior is not None and index == len(events) - 1:
            chunk["behavior"] = copy.deepcopy(behavior)
        frames.append(b"data: " + online.canonical_json(chunk) + b"\n\n")
    frames.append(b"data: [DONE]\n\n")
    return b"".join(frames)


class FakeState:
    def __init__(self, plan=None):
        self.plan = online.prepare_plan(plan or fixture_plan())
        self.schedule, self.schedule_sha = online.build_schedule(self.plan)
        self.by_rollout = {item["rollout_id"]: item for item in self.schedule}
        self.cancelled = set()
        self.timeout_first = set()
        self.retry_first = {}
        self.malformed = set()
        self.malformed_status = {}
        self.inner_drift = set()
        self.header_overrides = {}
        self.dropped_headers = set()
        self.behavior_mode = "not_exposed"
        self.repeat_request_id = False
        self.request_id_overrides = {}
        self.counts = {}
        self.requests = []
        self.raw_by_request_id = {}

    def behavior_block(self):
        if self.behavior_mode == "exposed":
            return {
                "format": online.BEHAVIOR_FORMAT,
                "tokenizer_sha256": "ab" * 32,
                "token_ids": [11, 12, 13],
                "logprobs": [-0.125, -0.25, -0.5],
                "mask": [0, 1, 1],
            }
        if self.behavior_mode == "malformed":
            return {
                "format": online.BEHAVIOR_FORMAT,
                "tokenizer_sha256": "ab" * 32,
                "token_ids": [11, 12, 13],
                "logprobs": [-0.125, -0.25, -0.5],
                "mask": [1, 0],
            }
        return None


class FakeHandler(BaseHTTPRequestHandler):
    state = None

    def log_message(self, *_args):
        pass

    def _send(self, status, payload, content_type, item, attempt, request_id):
        state = self.state
        coverage = "exposed" if state.behavior_mode in ("exposed", "malformed") \
            else "not_exposed"
        attestation = online._expected_attestation(
            item, state.plan, http_status=status,
            behavior_evidence=coverage, attempt=attempt)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        for key, header in online.RESPONSE_HEADERS.items():
            value = state.header_overrides.get((item["rollout_id"], key), attestation[key])
            if key not in state.dropped_headers:
                self.send_header(header, str(value))
        self.end_headers()
        if payload:
            self.wfile.write(payload)
        if content_type == "text/event-stream":
            state.raw_by_request_id[request_id] = payload
        self.close_connection = True

    def do_POST(self):
        state = self.state
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        rollout_id = self.headers.get("X-UmmInfer-Rollout-Id")
        if rollout_id not in state.by_rollout:
            self.send_error(400)
            return
        item = state.by_rollout[rollout_id]
        attempt = int(self.headers.get("X-UmmInfer-Rollout-Attempt"))
        count = state.counts.get((rollout_id, attempt), 0)
        state.counts[(rollout_id, attempt)] = count + 1
        state.requests.append({
            "rollout_id": rollout_id, "attempt": attempt, "body": body,
            "headers": {name.lower(): value for name, value in self.headers.items()},
        })
        if body != item["body"]:
            raise AssertionError("fake endpoint received a request outside the schedule")
        request_id = f"request-{rollout_id}-{attempt}"
        if state.repeat_request_id:
            request_id = "request-repeated"
        request_id = state.request_id_overrides.get(rollout_id, request_id)
        if attempt == 0 and count == 0 and rollout_id in state.timeout_first:
            threading.Event().wait(0.10)
            self.close_connection = True
            return
        if attempt == 0 and count == 0 and rollout_id in state.retry_first:
            status = state.retry_first[rollout_id]
            payload = online.canonical_json({
                "id": request_id, "model": item["body"]["model"],
                "error": {"message": "transient fixture status"},
            })
            self._send(status, payload, "application/json", item, attempt, request_id)
            return
        if rollout_id in state.malformed:
            self._send(state.malformed_status.get(rollout_id, 200),
                       b"data: {not-json}\n\n", "text/event-stream",
                       item, attempt, request_id)
            return
        events = fixture.cancelled_events() if rollout_id in state.cancelled \
            else response_events(item["group_index"], item["rollout_index"])
        raw = response_bytes(
            events, item, request_id, state.behavior_block(),
            inner_drift=rollout_id in state.inner_drift)
        self._send(200, raw, "text/event-stream", item, attempt, request_id)


class fake_endpoint:
    def __init__(self, state):
        self.state = state

    def __enter__(self):
        handler = type("Handler", (FakeHandler,), {"state": self.state})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return f"127.0.0.1:{self.server.server_port}"

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def execute(root, state):
    with fake_endpoint(state) as endpoint:
        return online.execute_plan(
            state.plan, endpoint, Path(root) / "bundle",
            clock_factory=lambda: LogicalTickClock(start=5, step=3))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(
        encoding="utf-8").splitlines() if line]


def accepted_files(root):
    root = Path(root)
    result = {
        Path(name) for name in (
            "plan.json", "trajectories.jsonl", "rollout_manifest.json",
            "rollout_manifest.json.sha256")
    }
    for directory in ("artifacts", "pack"):
        result.update(path.relative_to(root) for path in (root / directory).rglob("*")
                      if path.is_file())
    return result


def resign_grpo_pack(pack_dir):
    pack_dir = Path(pack_dir)
    manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    for name in ("grpo_train", "grpo_eval"):
        path = pack_dir / f"{name}.jsonl"
        payload = path.read_bytes()
        manifest["outputs"][name].update({
            "sha256": online.sha256_bytes(payload), "bytes": len(payload),
            "record_count": len(read_jsonl(path)),
        })
    payload = online.canonical_json(manifest) + b"\n"
    (pack_dir / "manifest.json").write_bytes(payload)
    (pack_dir / "manifest.json.sha256").write_text(
        online.sha256_bytes(payload) + "\n", encoding="ascii")


class OnlineRolloutContractTests(unittest.TestCase):
    def test_complete_groups_rewards_raw_sse_and_fresh_process_replay(self):
        with tempfile.TemporaryDirectory(prefix="umm-online-grpo-") as tmp:
            root = Path(tmp)
            state = FakeState()
            real_import = builtins.__import__

            def no_ml(name, *args, **kwargs):
                if name == "torch" or name == "numpy" or name.startswith("transformers"):
                    raise AssertionError(f"ML import entered online orchestrator: {name}")
                return real_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=no_ml):
                result = execute(root, state)
            bundle = root / "bundle"
            self.assertEqual(result["group_count"], 2)
            self.assertEqual(result["rollout_count"], 6)
            self.assertEqual(result["behavior_logprobs"], "NOT_COVERED")
            records = read_jsonl(bundle / "trajectories.jsonl")
            self.assertEqual([record["turn_index"] for record in records], list(range(6)))
            self.assertEqual(len({record["request_id"] for record in records}), 6)
            for record in records:
                provenance = record["rollout_provenance"]
                self.assertEqual(provenance["mode"], "online")
                self.assertEqual(provenance["behavior"]["status"], "NOT_COVERED")
                self.assertIsNone(provenance["behavior"]["logprobs"])
                self.assertIsNone(provenance["behavior"]["mask"])
                raw = validate_replay_metadata(
                    record, artifact_dir=bundle / "artifacts", require_raw=True)
                self.assertEqual(raw, state.raw_by_request_id[record["request_id"]])
                self.assertTrue(raw.endswith(b"data: [DONE]\n\n"))
                self.assertEqual(record["sse_replay"]["clock"]["kind"], "logical_tick")
            rows = read_jsonl(bundle / "pack/grpo_train.jsonl")
            self.assertEqual(len(rows), 2)
            self.assertEqual(read_jsonl(bundle / "pack/grpo_eval.jsonl"), [])
            for row in rows:
                self.assertEqual(row["rollout_provenance"]["mode"], "online")
                self.assertEqual(row["group_size"], 3)
                self.assertEqual(len({item["rollout_provenance"]["rollout_id"]
                                      for item in row["rollouts"]}), 3)
                scores = [item["score"] for item in row["rollouts"]]
                self.assertTrue(math.isclose(scores[0], 1.0, abs_tol=1e-12))
                self.assertTrue(math.isclose(scores[1], 2.0 / 3.0, abs_tol=1e-12))
                self.assertTrue(math.isclose(scores[2], 1.0 / 6.0, abs_tol=1e-12))
                self.assertEqual([item["rank"] for item in row["rollouts"]], [0, 1, 2])
                self.assertTrue(all(item["rollout_provenance"]["attempt"] == 0
                                    for item in row["rollouts"]))
            self.assertNotIn("torch", sys.modules)
            self.assertNotIn("transformers", sys.modules)

            replayed = root / "replayed"
            proc = subprocess.run([
                sys.executable, "-I", str(ROOT / "python/umminfer_online_rollout.py"),
                "replay", "--input", str(bundle), "--out", str(replayed),
            ], capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("ONLINE_GRPO_REPLAY: PASS", proc.stdout)
            self.assertEqual(accepted_files(bundle), accepted_files(replayed))
            for relative in accepted_files(bundle):
                self.assertEqual((bundle / relative).read_bytes(),
                                 (replayed / relative).read_bytes(), relative)

    def test_behavior_evidence_requires_aligned_endpoint_values(self):
        for mode in ("exposed", "malformed"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(
                    prefix="umm-online-behavior-") as tmp:
                state = FakeState()
                state.behavior_mode = mode
                if mode == "malformed":
                    with self.assertRaisesRegex(ValueError, "aligned"):
                        execute(tmp, state)
                    self.assertFalse((Path(tmp) / "bundle/rollout_manifest.json").exists())
                    continue
                result = execute(tmp, state)
                self.assertEqual(result["behavior_logprobs"], "COVERED")
                records = read_jsonl(Path(tmp) / "bundle/trajectories.jsonl")
                for record in records:
                    evidence = record["rollout_provenance"]["behavior"]
                    self.assertEqual(evidence["token_ids"], [11, 12, 13])
                    self.assertEqual(evidence["mask"], [0, 1, 1])
                    self.assertEqual(evidence["masked_token_count"], 2)
                replayed = online.replay_bundle(
                    Path(tmp) / "bundle", Path(tmp) / "replayed")
                self.assertEqual(replayed, result)

    def test_cancelled_response_replays_exactly_but_is_not_trainable(self):
        with tempfile.TemporaryDirectory(prefix="umm-online-cancel-") as tmp:
            root = Path(tmp)
            plan = fixture_plan()
            plan["transport"].update({
                "max_attempts": 3, "retry_statuses": [503],
                "retry_transport": True, "retry_timeouts": True,
            })
            state = FakeState(plan)
            state.cancelled.add(state.schedule[0]["rollout_id"])
            with self.assertRaisesRegex(ValueError, "cancelled.*rejected"):
                execute(root, state)
            bundle = root / "bundle"
            attempts = read_jsonl(bundle / "attempts.jsonl")
            self.assertEqual(len(attempts), 1)
            cancelled = attempts[0]
            self.assertEqual(cancelled["terminal_state"], "cancelled")
            self.assertEqual(cancelled["events"][-1]["type"], "cancelled")
            self.assertEqual(cancelled["rollout_provenance"]["attempt"], 0)
            raw = validate_replay_metadata(
                cancelled, artifact_dir=bundle / "artifacts", require_raw=True)
            self.assertTrue(raw.endswith(b"data: [DONE]\n\n"))
            replay_recorder = traj.TrajectoryRecorder(root / "cancel-replay/trajectories.jsonl")
            replayed = replay_record(
                cancelled, recorder=replay_recorder,
                artifact_dir=bundle / "artifacts")
            self.assertEqual(replayed, cancelled)
            self.assertFalse((bundle / "trajectories.jsonl").exists())
            self.assertFalse((bundle / "pack").exists())
            self.assertFalse((bundle / "rollout_manifest.json").exists())

    def test_timeout_retry_and_no_retry_policy(self):
        for retry in (False, True):
            with self.subTest(retry=retry), tempfile.TemporaryDirectory(
                    prefix="umm-online-timeout-") as tmp:
                plan = fixture_plan()
                plan["transport"].update({
                    "timeout_s": 0.02, "max_attempts": 2 if retry else 1,
                    "retry_timeouts": retry,
                })
                state = FakeState(plan)
                state.timeout_first.add(state.schedule[0]["rollout_id"])
                if not retry:
                    with self.assertRaisesRegex(ValueError, "without a permitted retry"):
                        execute(tmp, state)
                    self.assertFalse((Path(tmp) / "bundle/pack").exists())
                    continue
                result = execute(tmp, state)
                self.assertEqual(result["rollout_count"], 6)
                first = read_jsonl(Path(tmp) / "bundle/trajectories.jsonl")[0]
                self.assertEqual(first["rollout_provenance"]["attempt"], 1)
                attempts = read_jsonl(Path(tmp) / "bundle/attempts.jsonl")
                self.assertEqual(len(attempts), 7)
                self.assertEqual(attempts[0]["rollout_provenance"]["transport"]["kind"],
                                 "timeout")

    def test_http_status_retry_is_bounded_and_no_retry_is_default(self):
        for retry in (False, True):
            with self.subTest(retry=retry), tempfile.TemporaryDirectory(
                    prefix="umm-online-http-retry-") as tmp:
                plan = fixture_plan()
                plan["transport"].update({
                    "max_attempts": 2 if retry else 1,
                    "retry_statuses": [503] if retry else [],
                })
                state = FakeState(plan)
                state.retry_first[state.schedule[0]["rollout_id"]] = 503
                if not retry:
                    with self.assertRaisesRegex(ValueError, "without a permitted retry"):
                        execute(tmp, state)
                    continue
                execute(tmp, state)
                records = read_jsonl(Path(tmp) / "bundle/trajectories.jsonl")
                self.assertEqual(records[0]["rollout_provenance"]["attempt"], 1)
                self.assertEqual(records[0]["request_id"],
                                 f"request-{state.schedule[0]['rollout_id']}-1")

    def test_drift_missing_stale_cross_group_and_malformed_responses_fail(self):
        cases = (
            ("model", "model_sha256", "aa" * 32),
            ("model revision", "model_revision", "changed-model-revision"),
            ("policy revision", "behavior_policy_revision", "changed-policy-revision"),
            ("model config", "model_config_sha256", "bb" * 32),
            ("sampling", "sampling_sha256", "22" * 32),
            ("request", "request_sha256", "33" * 32),
            ("attempt", "attempt", "1"),
            ("freshness", "generation", "cached"),
        )
        for name, key, value in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory(
                    prefix="umm-online-drift-") as tmp:
                state = FakeState()
                state.header_overrides[(state.schedule[0]["rollout_id"], key)] = value
                with self.assertRaisesRegex(ValueError, "drift|cross-group"):
                    execute(tmp, state)
                self.assertFalse((Path(tmp) / "bundle/rollout_manifest.json").exists())
                self.assertFalse((Path(tmp) / "bundle/pack").exists())
        for mode in ("missing-header", "cross-group", "duplicate-request", "malformed",
                     "inner-identity"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(
                    prefix="umm-online-invalid-response-") as tmp:
                state = FakeState()
                first = state.schedule[0]["rollout_id"]
                if mode == "missing-header":
                    state.dropped_headers.add("behavior_policy_sha256")
                    error = "lacks online rollout attestation"
                elif mode == "cross-group":
                    state.header_overrides[(first, "group_id")] = \
                        state.schedule[3]["group_id"]
                    error = "cross-group"
                elif mode == "duplicate-request":
                    state.repeat_request_id = True
                    error = "duplicate or stale"
                elif mode == "inner-identity":
                    state.inner_drift.add(first)
                    error = "inner request/session"
                else:
                    state.plan["transport"].update({
                        "max_attempts": 2, "retry_statuses": [503],
                        "retry_transport": True, "retry_timeouts": True,
                    })
                    state = FakeState(state.plan)
                    state.malformed.add(first)
                    state.malformed_status[first] = 503
                    error = "without a permitted retry"
                with self.assertRaisesRegex(ValueError, error):
                    execute(tmp, state)
                if mode == "malformed":
                    self.assertEqual(len(state.requests), 1)
                self.assertFalse((Path(tmp) / "bundle/pack").exists())

    def test_malformed_exported_groups_fail_closed(self):
        mutations = (
            ("missing", lambda rows: rows[0]["rollouts"].pop()),
            ("duplicate", lambda rows: rows[0]["rollouts"].__setitem__(
                1, {**copy.deepcopy(rows[0]["rollouts"][0]), "rank": 1})),
            ("reordered", lambda rows: rows[0].update(rollouts=[
                {**rows[0]["rollouts"][1], "rank": 0},
                {**rows[0]["rollouts"][0], "rank": 1},
                rows[0]["rollouts"][2],
            ])),
            ("reward", lambda rows: rows[0]["rollouts"][0].update(
                score=rows[0]["rollouts"][0]["score"] + 0.125)),
            ("reward-name", lambda rows: rows[0].update(
                reward_name="unrelated-reward")),
            ("stale", lambda rows: rows[0]["rollouts"][0]["provenance"].update(
                source_request_id="stale-request")),
            ("cross-group", lambda rows: rows[1]["rollouts"].__setitem__(
                0, {**copy.deepcopy(rows[0]["rollouts"][0]),
                    "rollout_provenance": copy.deepcopy(
                        rows[0]["rollouts"][0]["rollout_provenance"])})),
        )
        for name, mutate in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory(
                    prefix="umm-online-malformed-group-") as tmp:
                root = Path(tmp)
                state = FakeState()
                execute(root / "source", state)
                bundle = root / "copy"
                shutil.copytree(root / "source/bundle", bundle)
                rows = read_jsonl(bundle / "pack/grpo_train.jsonl")
                mutate(rows)
                online._write_jsonl(bundle / "pack/grpo_train.jsonl", rows)
                resign_grpo_pack(bundle / "pack")
                records = read_jsonl(bundle / "trajectories.jsonl")
                schedule, schedule_sha = online.build_schedule(state.plan)
                with self.assertRaises(ValueError):
                    online._annotate_grpo_pack(
                        state.plan, schedule, schedule_sha, records, bundle / "pack",
                        load_reward(state.plan["reward"]["spec"]))

    def test_replay_rejects_non_200_and_stale_session_turn_identity(self):
        for mode in ("http-status", "session-turn"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(
                    prefix="umm-online-replay-identity-") as tmp:
                root = Path(tmp)
                state = FakeState()
                execute(root, state)
                bundle = root / "bundle"
                records = read_jsonl(bundle / "trajectories.jsonl")
                provenance = records[0]["rollout_provenance"]
                if mode == "http-status":
                    provenance["response"] = online._expected_attestation(
                        state.schedule[0], state.plan, http_status=503,
                        behavior_evidence=provenance["response"]["behavior_evidence"],
                        attempt=provenance["attempt"])
                    error = "HTTP 200"
                else:
                    records[0]["session_id"] = "stale-session"
                    records[0]["turn_index"] = 999
                    error = "session/turn/endpoint"
                online._write_jsonl(bundle / "trajectories.jsonl", records)
                with self.assertRaisesRegex(ValueError, error):
                    online.replay_bundle(bundle, root / "replayed")

    def test_replay_rejects_duplicate_request_ids_across_valid_records(self):
        with tempfile.TemporaryDirectory(prefix="umm-online-replay-duplicate-") as tmp:
            root = Path(tmp)
            first_state = FakeState()
            execute(root / "first", first_state)
            first_bundle = root / "first/bundle"
            first_records = read_jsonl(first_bundle / "trajectories.jsonl")
            duplicate_id = first_records[0]["request_id"]

            second_state = FakeState()
            second_state.request_id_overrides = {
                second_state.schedule[0]["rollout_id"]: "different-first-request",
                second_state.schedule[1]["rollout_id"]: duplicate_id,
            }
            execute(root / "second", second_state)
            second_bundle = root / "second/bundle"
            second_records = read_jsonl(second_bundle / "trajectories.jsonl")
            first_records[1] = second_records[1]
            online._write_jsonl(first_bundle / "trajectories.jsonl", first_records)
            shutil.copytree(second_bundle / "artifacts", first_bundle / "artifacts",
                            dirs_exist_ok=True)
            self.assertEqual(first_records[0]["request_id"],
                             first_records[1]["request_id"])
            with self.assertRaisesRegex(ValueError, "duplicate or stale.*accepted replay"):
                online.replay_bundle(first_bundle, root / "replayed")

    def test_frozen_offline_optimizer_rejects_online_groups_before_ml(self):
        with tempfile.TemporaryDirectory(prefix="umm-online-frozen-boundary-") as tmp:
            root = Path(tmp)
            state = FakeState()
            execute(root, state)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({
                "model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"],
            }), encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"contract-fixture-not-model-weights")
            real_import = builtins.__import__

            def no_ml(name, *args, **kwargs):
                if name == "torch" or name.startswith("transformers"):
                    raise AssertionError(f"ML import before online rejection: {name}")
                return real_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=no_ml):
                with self.assertRaisesRegex(ValueError, "frozen-offline.*refuses online"):
                    frozen_grpo.prepare_live_run(
                        dataset_dir=root / "bundle/pack",
                        trajectory_paths=[root / "bundle/trajectories.jsonl"],
                        output_dir=root / "optimizer", config_path=(
                            ROOT / "configs/training/joint_grpo_live_gate.json"),
                        model_path=model, source_commit="0" * 40, max_steps=2)
            stripped = root / "stripped-pack"
            dataset.build_dataset(
                [root / "bundle/trajectories.jsonl"], stripped, mode="preference",
                seed=state.plan["split_seed"], eval_ratio=0.0,
                reward_spec=state.plan["reward"]["spec"], min_group_size=3,
                strict_artifacts=True)
            with mock.patch("builtins.__import__", side_effect=no_ml):
                with self.assertRaisesRegex(ValueError, "frozen-offline.*refuses online"):
                    frozen_grpo.prepare_live_run(
                        dataset_dir=stripped,
                        trajectory_paths=[root / "bundle/trajectories.jsonl"],
                        output_dir=root / "stripped-optimizer", config_path=(
                            ROOT / "configs/training/joint_grpo_live_gate.json"),
                        model_path=model, source_commit="0" * 40, max_steps=2)
            self.assertFalse((root / "optimizer/live_manifest.json").exists())
            self.assertFalse((root / "stripped-optimizer/live_manifest.json").exists())
            self.assertNotIn("torch", sys.modules)
            self.assertNotIn("transformers", sys.modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
