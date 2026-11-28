#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fresh-process deterministic fixtures for ns08_first_turn_history_verify.py."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import msgpack
except ImportError:
    msgpack = None

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "tests/ns08_first_turn_history_verify.py"
SPEC = importlib.util.spec_from_file_location("ns08_first_turn_history_verify", VERIFIER)
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)

SETUP_PROMPT = ("For a small Mars greenhouse, give exactly three short steps: choose a warm site, "
                "collect water with a solar still, then monitor seedlings. Use concise prose only; "
                "do not create an image or video.")
FOLLOWUP_PROMPT = ("Without restating the setup, explain the second step and why it matters on Mars. "
                   "Use concise prose only; no image or video.")
CANCEL_PROMPT = ("Write a long detailed text-only explanation of greenhouse humidity control; "
                 "no image or video.")
POST_PROMPT = ("Return to the original Mars greenhouse setup: what was its second step? "
               "Prose only; no image or video.")
SETUP_ANSWER = "Choose a warm site. Collect water with a solar still. Monitor seedlings."
FOLLOWUP_ANSWER = "A solar still collects water, which matters because Mars has little usable surface water."
CANCEL_ANSWER = "Partial humidity explanation"
POST_ANSWER = "The second step is collecting water with a solar still."
STATELESS_PROMPT = "Mention STATELESS_QUASAR_MARKER_84 and answer with one short sentence."
STATELESS_ANSWER = "STATELESS_QUASAR_MARKER_84 is only a stateless marker."
CROSS_PROMPT = "What was the earlier Mars greenhouse setup?"
CROSS_ANSWER = "No earlier context is available in this session."
PCM = b"\x01\x00\x02\x00"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")


def response_fixture(public_id, session, text, outcome="complete", pcm=False):
    events = [{"seq": 0, "kind": "vlm_delta", "request_id": public_id,
               "session_id": session, "stage": "vlm", "text": text, "is_final": False}]
    media = []
    if pcm:
        artifact = {
            "artifact_id": "audio-setup", "media_type": "audio", "mime_type": "audio/pcm",
            "encoding": "base64", "data": base64.b64encode(PCM).decode(), "index": 0,
            "total": 1, "bytes": len(PCM), "sha256": digest(PCM), "format": "pcm16",
            "sample_rate": 24000, "channels": 1, "stream_index": 0,
        }
        events.append({"seq": 1, "kind": "artifact", "request_id": public_id,
                       "session_id": session, "stage": "tts", "is_final": False,
                       "artifact": artifact})
        media.append({"artifact_id": "audio-setup", "media_type": "audio",
                      "mime_type": "audio/pcm", "bytes": len(PCM), "sha256": digest(PCM),
                      "slot_id": None, "slot_index": None, "stream_index": 0,
                      "sample_rate": 24000, "channels": 1,
                      "anchor_slot_index": None, "anchor_sha256": None})
    events.append({"seq": len(events), "kind": outcome, "request_id": public_id,
                   "session_id": session, "stage": "vlm", "is_final": True,
                   "text": text, "legs": []})
    raw = b"".join(b"data: " + json.dumps({"id": public_id, "unified": event},
                                          separators=(",", ":")).encode() + b"\r\n\r\n"
                   for event in events)
    if outcome == "complete":
        raw += b"data: [DONE]\r\n\r\n"
    projected = json.loads(json.dumps(events))
    for event in projected:
        if "artifact" in event:
            artifact = event["artifact"]
            artifact["data_base64_bytes"] = len(artifact["data"])
            del artifact["data"]
    turn = {
        "index": 1, "role": "turn", "prompt": "", "response": text,
        "visual_slots": [], "outcome": "completed" if outcome == "complete" else "cancelled",
        "terminal": {"state": "completed" if outcome == "complete" else "cancelled",
                     "request_id": public_id, "legs": []},
        "narrative_events": projected, "media_artifacts": media,
        "replay": {"transport": "raw-sse", "raw_sse": base64.b64encode(raw).decode(),
                   "raw_sse_bytes": len(raw), "raw_sse_sha256": digest(raw),
                   "chunk_lengths": [1, len(raw) - 1], "terminal_required": outcome == "complete"},
    }
    return raw, turn


def native_row(public_id, session, messages):
    native_id = public_id + ":vlm"
    packed = msgpack.packb({"type": "submit", "request_id": native_id,
                            "data": {"request": {"inputs": {
                                "session_id": session, "messages": messages}}}},
                           use_bin_type=True)
    return {"observed_monotonic_ns": 1, "request_id": native_id,
            "raw_sha256": digest(packed), "raw_base64": base64.b64encode(packed).decode(),
            "messages": messages}


def seal(root):
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            entries.append(f"{digest(path.read_bytes())}  {path.relative_to(root).as_posix()}\n")
    (root / "SHA256SUMS").write_text("".join(entries), encoding="utf-8")


def make_arm(root, arm):
    root.mkdir()
    (root / "requests").mkdir()
    (root / "responses").mkdir()
    main_session = f"ummweb-{arm}-history-84"
    cross_session = "umminfer-cross-session-isolated-84"
    prompts = {
        "browser-01": SETUP_PROMPT, "browser-02": FOLLOWUP_PROMPT,
        "browser-03": CANCEL_PROMPT, "browser-04": POST_PROMPT,
        "explicit-stateless": STATELESS_PROMPT, "cross-session": CROSS_PROMPT,
    }
    answers = {
        "browser-01": SETUP_ANSWER, "browser-02": FOLLOWUP_ANSWER,
        "browser-03": CANCEL_ANSWER, "browser-04": POST_ANSWER,
        "explicit-stateless": STATELESS_ANSWER, "cross-session": CROSS_ANSWER,
    }
    public = {name: f"public-{arm}-{name}" for name in prompts}
    bodies = {}
    network = []
    raw_responses = {}
    turns = {}
    for name, prompt in prompts.items():
        session = cross_session if name == "cross-session" else main_session
        history = (arm == "after" if name == "browser-01" else
                   name in ("browser-02", "browser-03", "browser-04", "cross-session"))
        body = {"model": "qwen3-native", "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 64, "temperature": 0, "stream": True, "session_id": session}
        if history or name == "explicit-stateless":
            body["session_history"] = history
        bodies[name] = body
        outcome = "cancelled" if name == "browser-03" else "complete"
        raw, turn = response_fixture(public[name], session, answers[name], outcome,
                                     pcm=name == "browser-01")
        turn["prompt"] = prompt
        raw_responses[name] = raw
        turns[name] = turn
        request_raw = json.dumps(body, separators=(",", ":")).encode()
        (root / "requests" / f"{name}.json").write_bytes(request_raw)
        (root / "responses" / f"{name}.sse").write_bytes(raw)
        row = {"name": name, "request_sha256": digest(request_raw),
               "session_id": session, "session_history": history}
        if name in ("explicit-stateless", "cross-session"):
            row.update(status=200, response_sha256=digest(raw))
        else:
            row.update(url="http://127.0.0.1/v1/chat/completions", prompt=prompt,
                       manual_frame_selection=False, seed_present=False)
        network.append(row)
    original = {"schema": "umminfer-web-session-replay-v1",
                "replay_format": "umminfer-raw-sse-replay-v1",
                "exported_at": "2026-11-28T00:00:00.000Z", "session_id": main_session,
                "model": "qwen3-native", "turn_count": 1, "turns": [turns["browser-01"]],
                "replay_artifact_count": 1}
    cancelled = {"schema": "umminfer-web-session-replay-v1",
                 "replay_format": "umminfer-raw-sse-replay-v1",
                 "exported_at": "2026-11-28T00:00:00.000Z", "session_id": main_session,
                 "model": "qwen3-native", "turn_count": 3,
                 "turns": [turns["browser-01"], turns["browser-02"], turns["browser-03"]],
                 "replay_artifact_count": 1}
    (root / "original.replay.json").write_text(json.dumps(original, indent=2) + "\n")
    (root / "reexported.replay.json").write_bytes((root / "original.replay.json").read_bytes())
    (root / "cancelled.replay.json").write_text(json.dumps(cancelled, indent=2) + "\n")

    system = {"role": "system", "content": "route safely"}
    setup_pair = [{"role": "user", "content": SETUP_PROMPT},
                  {"role": "assistant", "content": SETUP_ANSWER}]
    followup_pair = [{"role": "user", "content": FOLLOWUP_PROMPT},
                     {"role": "assistant", "content": FOLLOWUP_ANSWER}]
    histories = {
        "browser-01": [{"role": "user", "content": SETUP_PROMPT}],
        "browser-02": ((setup_pair if arm == "after" else []) +
                       [{"role": "user", "content": FOLLOWUP_PROMPT}]),
        "browser-03": ((setup_pair if arm == "after" else []) + followup_pair +
                       [{"role": "user", "content": CANCEL_PROMPT}]),
        "browser-04": ((setup_pair if arm == "after" else []) + followup_pair +
                       [{"role": "user", "content": POST_PROMPT}]),
        "explicit-stateless": [{"role": "user", "content": STATELESS_PROMPT}],
        "cross-session": [{"role": "user", "content": CROSS_PROMPT}],
    }
    rows = []
    for name in prompts:
        session = cross_session if name == "cross-session" else main_session
        rows.append(native_row(public[name], session, [system] + histories[name]))
    (root / "vlm-prompts.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                                                     for row in rows))
    result = {
        "schema": "umminfer-ns08-first-turn-history-browser-v1", "arm": arm,
        "real_model": True, "physical_microphone": "NOT_RUN", "physical_speaker": "NOT_RUN",
        "perceptual_quality": "NOT_RUN", "checks": [{"name": "fixture", "ok": True}],
        "failures": [], "answers": [
            {"label": "setup", "text": SETUP_ANSWER},
            {"label": "followup", "text": FOLLOWUP_ANSWER},
            {"label": "cancelled", "text": CANCEL_ANSWER},
            {"label": "post-cancel-stateless-check", "text": POST_ANSWER},
        ], "network": network, "screenshots": [], "console": [],
    }
    write_json(root / "result.json", result)
    gpu = {"index": 1, "memory_used_mib": 3, "utilization_percent": 0, "compute_processes": []}
    stack = {
        "schema": "umminfer-storyboard-84-live-stack/v1",
        "execution_commit": "84f82ee879efe384f1eea6a7ee2544d9d3b7b274" if arm == "before"
                            else "a" * 40,
        "vlm_prompt_capture": "vlm-prompts.jsonl", "gpu1_before": gpu, "gpu1_after": gpu,
        "transitions": [], "terminal_requests": sorted(public.values()),
        "processes": [{"name": "server", "pid": 1234, "stop_monotonic_ns": 10,
                       "returncode": -15, "stop_error": None}],
        "binaries": {name: digest(name.encode()) for name in ("server", "vlm", "tts", "wan")},
        "models": {name: {"tree_sha256": digest(name.encode())}
                   for name in ("vlm", "flux2", "tts", "wan22")},
    }
    write_json(root / "stack-report.json", stack)
    seal(root)


@unittest.skipUnless(msgpack is not None, "real msgpack dependency unavailable; fixture tests NOT_RUN")
class Ns08FirstTurnHistoryVerifyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.before = self.root / "before"
        self.after = self.root / "after"
        make_arm(self.before, "before")
        make_arm(self.after, "after")
        self.counter = 0

    def tearDown(self):
        self.temp.cleanup()

    def run_verifier(self, replay_binary=None):
        self.counter += 1
        out = self.root / f"report-{self.counter}.json"
        if replay_binary is None:
            replay_binary = self.root / "not-needed-without-rendered-prompts"
        result = subprocess.run(
            [sys.executable, "-B", "-I", str(VERIFIER),
             "--before", str(self.before), "--after", str(self.after),
             "--replay-binary", str(replay_binary),
             "--out", str(out)], capture_output=True, text=True, timeout=30)
        self.assertTrue(out.is_file(), result.stdout + result.stderr)
        return result, json.loads(out.read_text())

    def fake_native_replay(self):
        path = self.root / "qwen38_history_prompt_replay"
        path.write_text(f"""#!{sys.executable}
import json, sys
source = json.load(open(sys.argv[2], encoding='utf-8'))
for row in source['requests']:
    row['rendered_prompt'] = 'NATIVE:' + json.dumps(row['messages'], sort_keys=True)
json.dump({{'schema': 'umminfer-vlm-history-prompt-replay-v1',
           'requests': source['requests']}}, open(sys.argv[4], 'w', encoding='utf-8'))
""", encoding="utf-8")
        path.chmod(0o755)
        return path

    def add_rendered_prompts(self, root, tamper=False):
        rows = [json.loads(line) for line in (root / "vlm-prompts.jsonl").read_text().splitlines()]
        entries = [{"id": row["request_id"], "messages": row["messages"],
                    "rendered_prompt": "NATIVE:" + json.dumps(row["messages"], sort_keys=True)}
                   for row in rows]
        if tamper:
            entries[0]["rendered_prompt"] += "tampered"
        write_json(root / "rendered-prompts.json", {
            "schema": "umminfer-vlm-history-prompt-replay-v1", "requests": entries})
        seal(root)

    def rewrite_native(self, root, name, messages):
        public_id = f"public-{root.name}-{name}"
        native_id = public_id + ":vlm"
        rows = [json.loads(line) for line in (root / "vlm-prompts.jsonl").read_text().splitlines()]
        for index, row in enumerate(rows):
            if row["request_id"] == native_id:
                session = ("umminfer-cross-session-isolated-84" if name == "cross-session"
                           else f"ummweb-{root.name}-history-84")
                rows[index] = native_row(public_id, session, messages)
        (root / "vlm-prompts.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                                                         for row in rows))
        seal(root)

    def test_paired_real_msgpack_fixture_passes(self):
        result, report = self.run_verifier()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(report["arms"]["before"]["metrics"]["setup_retained"])
        self.assertTrue(report["arms"]["after"]["metrics"]["setup_retained"])
        self.assertEqual(report["arms"]["after"]["metrics"]["original_pcm_bytes"], len(PCM))
        self.assertEqual(report["arms"]["after"]["metrics"]["unique_native_request_ids"], 6)
        self.assertEqual(report["arms"]["after"]["native_rendered_prompts"]["status"], "NOT_RUN")
        self.assertTrue(report["paired"]["first_turn_history_fix_verified"])

    def test_transport_only_cancelled_response_is_explicitly_supported(self):
        for root in (self.before, self.after):
            (root / "responses/browser-03.sse").write_bytes(b"")
            archive_path = root / "cancelled.replay.json"
            archive = json.loads(archive_path.read_text())
            turn = archive["turns"][2]
            turn["response"] = ""
            turn["terminal"] = {"state": "cancelled", "request_id": None, "legs": []}
            for key in ("replay", "narrative_events", "media_artifacts"):
                turn.pop(key, None)
            archive_path.write_text(json.dumps(archive, indent=2) + "\n")
            result_path = root / "result.json"
            result = json.loads(result_path.read_text())
            result["answers"][2]["text"] = ""
            result_path.write_text(json.dumps(result, indent=2) + "\n")
            seal(root)
        result, report = self.run_verifier()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for arm in ("before", "after"):
            self.assertTrue(report["arms"][arm]["metrics"]["transport_only_cancelled"])
            self.assertEqual(report["arms"][arm]["metrics"]["unique_public_request_ids"], 5)
            self.assertEqual(report["arms"][arm]["metrics"]["unique_native_request_ids"], 6)

    def test_archive_terminal_may_omit_optional_public_request_id(self):
        for root in (self.before, self.after):
            document = json.loads((root / "original.replay.json").read_text())
            document["turns"][0]["terminal"]["request_id"] = None
            encoded = json.dumps(document, indent=2) + "\n"
            (root / "original.replay.json").write_text(encoded)
            (root / "reexported.replay.json").write_text(encoded)
            path = root / "cancelled.replay.json"
            document = json.loads(path.read_text())
            document["turns"][1]["terminal"]["request_id"] = None
            path.write_text(json.dumps(document, indent=2) + "\n")
            seal(root)
        result, report = self.run_verifier()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(report["status"], "PASS")

    def test_whole_file_reexport_tamper_fails_after_rehash(self):
        document = json.loads((self.after / "reexported.replay.json").read_text())
        document["exported_at"] = "2026-11-28T00:00:01.000Z"
        (self.after / "reexported.replay.json").write_text(json.dumps(document, indent=2) + "\n")
        seal(self.after)
        result, report = self.run_verifier()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("whole-file re-export", " ".join(report["failures"]))

    def test_after_followup_without_setup_history_fails(self):
        self.rewrite_native(self.after, "browser-02", [
            {"role": "system", "content": "route safely"},
            {"role": "user", "content": FOLLOWUP_PROMPT},
        ])
        result, report = self.run_verifier()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("after/browser-02: exact retained-history mismatch",
                      " ".join(report["failures"]))

    def test_cancelled_prompt_in_later_history_fails(self):
        self.rewrite_native(self.after, "browser-04", [
            {"role": "system", "content": "route safely"},
            {"role": "user", "content": SETUP_PROMPT},
            {"role": "assistant", "content": SETUP_ANSWER},
            {"role": "user", "content": FOLLOWUP_PROMPT},
            {"role": "assistant", "content": FOLLOWUP_ANSWER},
            {"role": "user", "content": CANCEL_PROMPT},
            {"role": "assistant", "content": CANCEL_ANSWER},
            {"role": "user", "content": POST_PROMPT},
        ])
        result, report = self.run_verifier()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("after/browser-04: exact retained-history mismatch",
                      " ".join(report["failures"]))

    def test_cross_session_history_leak_fails(self):
        self.rewrite_native(self.after, "cross-session", [
            {"role": "system", "content": "route safely"},
            {"role": "user", "content": SETUP_PROMPT},
            {"role": "assistant", "content": SETUP_ANSWER},
            {"role": "user", "content": CROSS_PROMPT},
        ])
        result, report = self.run_verifier()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("after/cross-session: exact retained-history mismatch",
                      " ".join(report["failures"]))

    def test_public_session_ownership_tamper_fails(self):
        path = self.after / "responses/browser-01.sse"
        path.write_bytes(path.read_bytes().replace(b"ummweb-after-history-84",
                                                   b"ummweb-foreign-session-84"))
        seal(self.after)
        result, report = self.run_verifier()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cross-session public event", " ".join(report["failures"]))

    def test_msgpack_observer_tamper_fails(self):
        path = self.after / "vlm-prompts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        packed = bytearray(base64.b64decode(rows[0]["raw_base64"]))
        packed[-1] ^= 1
        rows[0]["raw_base64"] = base64.b64encode(packed).decode()
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        seal(self.after)
        result, report = self.run_verifier()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("observer raw digest mismatch", " ".join(report["failures"]))

    def test_unlisted_or_missing_manifest_file_fails(self):
        (self.after / "unlisted.txt").write_text("not covered")
        result, report = self.run_verifier()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("manifest does not cover exactly", " ".join(report["failures"]))

    def test_optional_native_replay_recomputes_every_entry(self):
        self.add_rendered_prompts(self.after)
        binary = self.fake_native_replay()
        result, report = self.run_verifier(binary)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(report["arms"]["before"]["native_rendered_prompts"]["status"], "NOT_RUN")
        self.assertEqual(report["arms"]["after"]["native_rendered_prompts"],
                         {"status": "PASS", "requests": 6})

    def test_optional_native_replay_tamper_fails(self):
        self.add_rendered_prompts(self.after, tamper=True)
        binary = self.fake_native_replay()
        result, report = self.run_verifier(binary)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("native rendered prompt/messages are not byte-exact",
                      " ".join(report["failures"]))

    def test_output_inside_evidence_root_fails(self):
        out = self.after / "verification.json"
        result = subprocess.run(
            [sys.executable, "-B", "-I", str(VERIFIER),
             "--before", str(self.before), "--after", str(self.after),
             "--replay-binary", "/unused", "--out", str(out)],
            capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        report = json.loads(out.read_text())
        self.assertIn("outside evidence root", " ".join(report["failures"]))


if __name__ == "__main__":
    unittest.main()
