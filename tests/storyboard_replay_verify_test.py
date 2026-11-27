#!/usr/bin/env python3
"""Fresh-process deterministic tests for storyboard_replay_verify.py."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

VERIFIER = Path(__file__).with_name("storyboard_replay_verify.py")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def event(seq, kind, **extra):
    return {"seq": seq, "kind": kind, "request_id": "r1", "session_id": "s1",
            "stage": "vlm", "is_final": kind == "complete", "text": "clean", **extra}


def archive(cancelled=False, pcm=b"\x00\x01\x02\x03"):
    artifact = {
        "artifact_id": "a0", "media_type": "audio", "mime_type": "audio/pcm",
        "encoding": "base64", "data": base64.b64encode(pcm).decode(), "index": 0,
        "total": 1, "bytes": len(pcm), "sha256": sha(pcm), "format": "pcm16",
        "sample_rate": 24000, "channels": 1, "stream_index": 0,
    }
    unified = [event(0, "vlm_delta"), event(1, "artifact", stage="tts", artifact=artifact)]
    if not cancelled:
        unified.append(event(2, "complete"))
    raw = b"".join(b"data: " + json.dumps({"unified": x}, separators=(",", ":")).encode() + b"\r\n\r\n"
                   for x in unified)
    if not cancelled:
        raw += b"data: [DONE]\r\n\r\n"
    projected = json.loads(json.dumps(unified))
    projected[1]["artifact"]["data_base64_bytes"] = len(projected[1]["artifact"]["data"])
    del projected[1]["artifact"]["data"]
    outcome = "cancelled" if cancelled else "completed"
    turn = {
        "index": 1, "role": "turn", "prompt": "story", "response": "clean",
        "visual_slots": [], "outcome": outcome,
        "terminal": {"state": outcome, "request_id": "r1", "legs": []},
        "narrative_events": projected,
        "media_artifacts": [{
            "artifact_id": "a0", "media_type": "audio", "mime_type": "audio/pcm",
            "bytes": len(pcm), "sha256": sha(pcm), "slot_id": None, "slot_index": None,
            "stream_index": 0, "sample_rate": 24000, "channels": 1,
            "anchor_slot_index": None, "anchor_sha256": None,
        }],
        "replay": {
            "transport": "raw-sse", "raw_sse": base64.b64encode(raw).decode(),
            "raw_sse_bytes": len(raw), "raw_sse_sha256": sha(raw),
            "chunk_lengths": [1, 2, len(raw) - 3], "terminal_required": not cancelled,
        },
    }
    return {"schema": "umminfer-web-session-replay-v1",
            "replay_format": "umminfer-raw-sse-replay-v1", "exported_at": "fixed",
            "session_id": "s1", "model": "qwen3-native", "turn_count": 1,
            "turns": [turn], "replay_artifact_count": 1}


def replace_unified(document, unified):
    turn = document["turns"][0]
    raw = b"".join(b"data: " + json.dumps({"unified": x}, separators=(",", ":")).encode() + b"\r\n\r\n"
                   for x in unified) + b"data: [DONE]\r\n\r\n"
    projected = json.loads(json.dumps(unified))
    for item in projected:
        if "artifact" in item:
            item["artifact"]["data_base64_bytes"] = len(item["artifact"]["data"])
            del item["artifact"]["data"]
    turn["narrative_events"] = projected
    turn["replay"].update(raw_sse=base64.b64encode(raw).decode(), raw_sse_bytes=len(raw),
                          raw_sse_sha256=sha(raw), chunk_lengths=[len(raw)])


class StoryboardReplayVerifyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def run_verifier(self, document, name, manifest=False):
        archive = self.root / (name + ".json")
        archive.write_text(json.dumps(document, separators=(",", ":")) + "\n")
        out = self.root.parent / (self.root.name + "-" + name + "-report.json")
        command = [sys.executable, "-I", str(VERIFIER), "--archive", str(archive), "--out", str(out)]
        if manifest:
            evidence = self.root / (name + "-evidence")
            evidence.mkdir()
            (evidence / "archive.json").write_bytes(archive.read_bytes())
            (evidence / "browser.log").write_text("real-model=false\n")
            sums = evidence / "SHA256SUMS"
            sums.write_text("".join(f"{sha(p.read_bytes())}  {p.name}\n"
                                    for p in sorted(evidence.iterdir()) if p.name != "SHA256SUMS"))
            command += ["--manifest", str(sums), "--manifest-root", str(evidence)]
            extra = evidence / "extra"
            if name.endswith("manifest-extra"):
                extra.write_text("unlisted")
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        report = json.loads(out.read_text())
        return result, report

    def test_valid_pcm_and_cancelled_partial(self):
        for name, document in (("complete", archive()), ("cancelled", archive(cancelled=True))):
            result, report = self.run_verifier(document, name)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["totals"]["pcm_bytes"], 4)

    def test_event_projection_tamper_rejected(self):
        document = archive()
        document["turns"][0]["narrative_events"][0]["text"] = "tampered"
        result, report = self.run_verifier(document, "event-tamper")
        self.assertEqual(result.returncode, 2)
        self.assertIn("event projection", report["error"])

    def test_artifact_tamper_rejected_despite_raw_rehash(self):
        document = archive()
        replay = document["turns"][0]["replay"]
        raw = base64.b64decode(replay["raw_sse"])
        lines = []
        for line in raw.decode().splitlines():
            if line.startswith("data: ") and '"kind":"artifact"' in line:
                root = json.loads(line[6:])
                root["unified"]["artifact"]["data"] = base64.b64encode(b"\x09\x01\x02\x03").decode()
                line = "data: " + json.dumps(root, separators=(",", ":"))
            lines.append(line)
        raw = ("\r\n".join(lines) + "\r\n").encode()
        replay["raw_sse"] = base64.b64encode(raw).decode()
        replay["raw_sse_bytes"] = len(raw)
        replay["raw_sse_sha256"] = sha(raw)
        replay["chunk_lengths"] = [len(raw)]
        result, report = self.run_verifier(document, "artifact-tamper")
        self.assertEqual(result.returncode, 2)
        self.assertIn("SHA-256 mismatch", report["error"])

    def test_manifest_self_exclusion_and_exact_coverage(self):
        result, report = self.run_verifier(archive(), "manifest-ok", manifest=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(report["manifest"]["files"], 2)
        result, report = self.run_verifier(archive(), "manifest-extra", manifest=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("exactly", report["error"])

    def test_chunk_attribution_tamper_rejected(self):
        document = archive()
        turn = document["turns"][0]
        original = turn["narrative_events"][1]["artifact"]
        first = json.loads(json.dumps(original))
        first.update(data=base64.b64encode(b"\x00\x01").decode(), index=0, total=2,
                     mime_type="audio/evil")
        first.pop("bytes", None); first.pop("sha256", None)
        last = json.loads(json.dumps(original))
        last.update(data=base64.b64encode(b"\x02\x03").decode(), index=1, total=2)
        complete = turn["narrative_events"][-1]
        complete["seq"] = 3
        unified = [turn["narrative_events"][0], event(1, "artifact", stage="tts", artifact=first),
                   event(2, "artifact", stage="tts", artifact=last), complete]
        raw = b"".join(b"data: " + json.dumps({"unified": x}, separators=(",", ":")).encode() + b"\r\n\r\n"
                       for x in unified) + b"data: [DONE]\r\n\r\n"
        for item in unified:
            if "artifact" in item:
                item["artifact"]["data_base64_bytes"] = len(item["artifact"]["data"])
                del item["artifact"]["data"]
        turn["narrative_events"] = unified
        turn["replay"].update(raw_sse=base64.b64encode(raw).decode(), raw_sse_bytes=len(raw),
                              raw_sse_sha256=sha(raw), chunk_lengths=[len(raw)])
        result, report = self.run_verifier(document, "chunk-attribution")
        self.assertEqual(result.returncode, 2)
        self.assertIn("mime_type changed", report["error"])

    def test_terminal_outcome_tamper_rejected(self):
        document = archive()
        document["turns"][0]["terminal"]["state"] = "error"
        result, report = self.run_verifier(document, "terminal-outcome-tamper")
        self.assertEqual(result.returncode, 2)
        self.assertIn("terminal kind/outcome", report["error"])

        document = archive()
        replay = document["turns"][0]["replay"]
        roots = []
        for line in base64.b64decode(replay["raw_sse"]).decode().splitlines():
            if line.startswith("data: ") and '"kind":"complete"' in line:
                root = json.loads(line[6:])
                root["unified"]["kind"] = "cancelled"
                roots.append(root)
            elif line == "data: [DONE]":
                roots.append(None)
            elif line:
                roots.append(json.loads(line[6:]))
        document["turns"][0]["narrative_events"][-1]["kind"] = "cancelled"
        raw = b"".join(b"data: [DONE]\r\n\r\n" if root is None else
                       b"data: " + json.dumps(root, separators=(",", ":")).encode() + b"\r\n\r\n"
                       for root in roots)
        replay.update(raw_sse=base64.b64encode(raw).decode(), raw_sse_bytes=len(raw),
                      raw_sse_sha256=sha(raw), chunk_lengths=[len(raw)])
        result, report = self.run_verifier(document, "raw-terminal-tamper")
        self.assertEqual(result.returncode, 2)
        self.assertIn("terminal kind/outcome", report["error"])

    def test_omitted_request_ids_allowed_but_conflicts_rejected(self):
        document = archive()
        replay = document["turns"][0]["replay"]
        roots = []
        for line in base64.b64decode(replay["raw_sse"]).decode().splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                root = json.loads(line[6:])
                if root["unified"]["kind"] != "vlm_delta":
                    root["unified"].pop("request_id", None)
                roots.append(root)
            else:
                roots.append(line)
        for item in document["turns"][0]["narrative_events"]:
            if item["kind"] != "vlm_delta":
                item.pop("request_id", None)
        raw = b"".join((item + "\r\n\r\n").encode() if isinstance(item, str) else
                       b"data: " + json.dumps(item, separators=(",", ":")).encode() + b"\r\n\r\n"
                       for item in roots)
        replay.update(raw_sse=base64.b64encode(raw).decode(), raw_sse_bytes=len(raw),
                      raw_sse_sha256=sha(raw), chunk_lengths=[len(raw)])
        result, report = self.run_verifier(document, "optional-request-ids")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        conflict = archive()
        conflict["turns"][0]["narrative_events"][1]["request_id"] = "other"
        result, report = self.run_verifier(conflict, "request-id-projection-conflict")
        self.assertEqual(result.returncode, 2)

    def test_transport_only_cancelled_response_allows_omitted_event_index(self):
        document = archive(cancelled=True)
        turn = document["turns"][0]
        raw = b'data: {"choices":[{"delta":{"role":"assistant"}}],"id":"r1"}\n\n'
        turn["replay"].update(raw_sse=base64.b64encode(raw).decode(), raw_sse_bytes=len(raw),
                              raw_sse_sha256=sha(raw), chunk_lengths=[len(raw)])
        del turn["narrative_events"]
        turn["media_artifacts"] = []
        result, report = self.run_verifier(document, "transport-only-cancelled")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_direct_i2v_variant_and_source_binding(self):
        document = archive()
        turn = document["turns"][0]
        png = b"\x89PNG\r\n\x1a\n" + b"original-image"
        mp4 = b"\x00\x00\x00\x18ftyp" + b"direct-i2v"
        image_artifact = {
            "artifact_id": "image-0", "media_type": "image", "mime_type": "image/png",
            "encoding": "base64", "data": base64.b64encode(png).decode(), "index": 0,
            "total": 1, "bytes": len(png), "sha256": sha(png), "slot_id": "slot-0",
            "slot_index": 0, "format": "png",
        }
        original_unified = [json.loads(line[6:])["unified"]
                            for line in base64.b64decode(turn["replay"]["raw_sse"]).decode().splitlines()
                            if line.startswith("data: ") and line != "data: [DONE]"]
        unified = [
            original_unified[0],
            event(1, "visual_slot", stage="image", slot_id="slot-0", slot_index=0,
                  media_type="image", prompt="keyframe"),
            {**original_unified[1], "seq": 2},
            event(3, "artifact", stage="image", artifact=image_artifact),
            event(4, "complete", legs=[{"slot_id": "slot-0", "state": "completed"}]),
        ]
        replace_unified(document, unified)
        turn["media_artifacts"].append({
            "artifact_id": "image-0", "media_type": "image", "mime_type": "image/png",
            "bytes": len(png), "sha256": sha(png), "slot_id": "slot-0", "slot_index": 0,
            "stream_index": None, "sample_rate": None, "channels": None,
            "anchor_slot_index": None, "anchor_sha256": None,
        })
        turn["terminal"]["legs"] = [{"slot_id": "slot-0", "state": "completed"}]
        turn["visual_slots"] = [{
            "id": "slot-0", "slot_index": 0, "kind": "video", "bytes": len(mp4),
            "sha256": sha(mp4), "terminal_state": "completed",
            "variants": [
                {"kind": "image", "bytes": len(png), "sha256": sha(png)},
                {"kind": "video", "bytes": len(mp4), "sha256": sha(mp4)},
            ],
        }]
        turn["direct_renders"] = [{
            "slot_id": "slot-0", "kind": "video", "mime_type": "video/mp4",
            "bytes": len(mp4), "sha256": sha(mp4), "source_sha256": sha(png),
            "prompt": "animate", "request_options": {},
            "response_base64": base64.b64encode(mp4).decode(),
        }]
        result, report = self.run_verifier(document, "direct-i2v-valid")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(report["totals"]["direct_renders"], 1)
        document["turns"][0]["direct_renders"][0]["source_sha256"] = sha(b"wrong")
        result, report = self.run_verifier(document, "direct-i2v-wrong-source")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Animate source", report["error"])

    def test_unresolved_slot_cannot_be_completed(self):
        document = archive()
        document["turns"][0]["visual_slots"] = [{"id": "slot-0", "kind": "image", "terminal_state": "running"}]
        result, report = self.run_verifier(document, "unresolved")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unbound non-slot visual", report["error"])


if __name__ == "__main__":
    unittest.main()
