#!/usr/bin/env python3
"""Deterministic CPU fixtures for first-audio planning, raw verification and A/B."""

import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


first = load("umminfer_realtime_first_audio_test_module", ROOT / "python/umminfer_realtime_first_audio.py")
fixtures = load("umminfer_realtime_first_audio_fixture_helpers", ROOT / "tests/realtime_latency_audit_test.py")
pack = fixtures.pack
wire = fixtures.wire


def sha(data):
    return hashlib.sha256(data).hexdigest()


class EvidenceBuilder:
    def __init__(self, root, tts_queue_ms=200, drop_public_chunk=False):
        self.root = Path(root)
        self.root.mkdir()
        self.tts_queue_ns = tts_queue_ms * 1_000_000
        self.drop_public_chunk = drop_public_chunk
        self.contract = json.loads((ROOT / "benchmarks/realtime-first-audio/preregistration.json").read_text())
        self.fixture_data = {"short": b"short-fixture", "long": b"long-fixture"}
        for name, data in self.fixture_data.items():
            self.contract["fixtures"][name]["sha256"] = sha(data)
            for role in ("asr", "vlm", "tts"):
                self.contract["topology"][role]["model_manifest_sha256"] = sha(role.encode())
        (self.root / "preregistration.json").write_text(json.dumps(self.contract, indent=2, sort_keys=True) + "\n")
        self.plan = first.make_plan(self.contract)
        (self.root / "plan.json").write_text(json.dumps(self.plan, indent=2, sort_keys=True) + "\n")
        (self.root / "fixtures").mkdir()
        for name, data in self.fixture_data.items():
            (self.root / "fixtures" / self.contract["fixtures"][name]["file"]).write_bytes(data)
        self.run = {"schema": "umminfer-realtime-first-audio-live/v1", "source_commit": "fixture",
                    "physical_microphone_speaker": "NOT_COVERED", "wan": "NOT_COVERED",
                    "perceptual_listening_quality": "NOT_COVERED", "binaries": {},
                    "models": {role: {"model_manifest_sha256": sha(role.encode())}
                               for role in ("asr", "vlm", "tts")},
                    "final_health": {"status": "ok", "in_flight": 0},
                    "cleanup": {"owned_processes_exited": True, "selected_gpus_idle": True},
                    "gpu_preflight": {}, "gpu_postflight": {}}

    def build(self):
        for arm in self.plan["arms"]:
            self.build_arm(arm)
        (self.root / "run.json").write_text(json.dumps(self.run, indent=2, sort_keys=True) + "\n")
        return self.root

    def build_arm(self, arm):
        root = self.root / "arms" / arm["name"]
        (root / "frames").mkdir(parents=True)
        (root / "observers").mkdir()
        self.clock = 1_000_000_000
        self.rows = {name: [] for name in ("asr-submit", "vlm-submit", "tts-submit", "completion")}
        self.logs = {"asr": [], "vlm": [], "tts": []}
        request_number = 0
        # Real peer-ready admin traffic shares each submit channel and must not
        # be mistaken for a model request.
        for channel in ("asr-submit", "vlm-submit", "tts-submit"):
            self.clock += 1000
            raw = pack({"type": "admin", "operation": {"op_id": "peer_ready"}})
            self.rows[channel].append({"index": 0, "channel": channel,
                "observed_monotonic_ns": self.clock, "forwarded_monotonic_ns": self.clock + 1000,
                "raw_sha256": sha(raw), "raw_base64": base64.b64encode(raw).decode()})
            self.clock += 1000
            raw = pack({"type": "abort", "request_id": "req-cancelled"})
            self.rows[channel].append({"index": 1, "channel": channel,
                "observed_monotonic_ns": self.clock, "forwarded_monotonic_ns": self.clock + 1000,
                "raw_sha256": sha(raw), "raw_base64": base64.b64encode(raw).decode()})
        pcm = b"\x01\x00" * 32
        response_pcm = b"\x02\x00" * 32
        for connection in arm["connections"]:
            capture = fixtures.Capture()
            capture.clock = self.clock
            history = []
            pending_tail = b""
            for turn_index in range(connection["turns"]):
                request_number += 1
                number = request_number
                rid = f"req-{number}"
                asr_rid = f"realtime-{number}"
                transcript = f"{arm['fixture']} transcript"
                text = "Hello world.\n\n Response"
                first_text = "Hello world."
                segment = turn_index
                committed_pcm = pending_tail + pcm

                def tick(ns=1_000_000):
                    self.clock += ns
                    return self.clock

                def capture_event(event, tx=False, ns=1_000_000):
                    capture.clock = self.clock
                    row = capture.event(event, tx=tx, advance=ns)
                    self.clock = capture.clock
                    return row

                def observe(channel, message, ns=1_000_000):
                    observed = tick(ns)
                    raw = pack(message)
                    self.rows[channel].append({"index": len(self.rows[channel]), "channel": channel,
                        "observed_monotonic_ns": observed, "forwarded_monotonic_ns": tick(1000),
                        "raw_sha256": sha(raw), "raw_base64": base64.b64encode(raw).decode()})
                    return observed

                capture_event({"type": "input_audio_buffer.append",
                               "audio": base64.b64encode(pcm).decode()}, tx=True)
                stop_ns = capture_event({"type": "input_audio_buffer.speech_stopped",
                                         "segment_id": segment,
                                         "audio_ms": len(committed_pcm) // 32})["monotonic_ns"]
                # The VAD-triggering/remaining tail can be captured around RX
                # stop. audio_ms pins the exact native committed prefix.
                capture_event({"type": "input_audio_buffer.append",
                               "audio": base64.b64encode(b"\x00\x00" * 32).decode()}, tx=True)
                pending_tail = b"\x00\x00" * 32
                asr_submit = observe("asr-submit", {"type": "submit", "request_id": asr_rid,
                    "data": {"request_id": asr_rid, "request": {"inputs": {"audio": committed_pcm},
                        "params": {"temperature": 0}, "metadata": {}}}})
                asr_done = observe("completion", {"type": "complete", "request_id": asr_rid,
                    "from_stage": "asr", "success": True,
                    "result": {"text": transcript, "generated_token_count": 2, "finish_reason": "stop"},
                    "error": None}, ns=44_000_000)
                # Late native duplicates are legal after cancellation; public
                # terminal duplicates remain forbidden by the WS verifier.
                observe("completion", {"type": "complete", "request_id": asr_rid,
                    "from_stage": "asr", "success": False, "result": {},
                    "error": "request aborted"}, ns=1000)
                asr_final = capture_event({"type": "transcription.completed", "segment_id": segment,
                                           "text": transcript})["monotonic_ns"]
                messages = history + [{"role": "user", "content": transcript}]
                prompt = observe("vlm-submit", {"type": "submit", "request_id": rid + ":vlm",
                    "data": {"request_id": rid + ":vlm", "request": {"inputs": {"messages": messages},
                        "params": {}, "metadata": {}}}})
                native_text = []
                for delta in ("Hello", " world.", "\n\n Response"):
                    native_text.append(observe("completion", {"type": "stream", "request_id": rid + ":vlm",
                        "from_stage": "vlm", "chunk": {"text": delta}}))
                    capture_event({"type": "response.text.delta", "response_id": rid,
                                   "segment_id": segment, "text": delta,
                                   "text_cursor": len("".join(("Hello", " world.", "\n\n Response")[
                                       :len(native_text)]).encode())})
                first_ready = native_text[1]
                tts_submit = observe("tts-submit", {"type": "submit", "request_id": rid + ":tts:0",
                    "data": {"request_id": rid + ":tts:0", "request": {"inputs": {
                        "text": first_text, "stream": True, "voice": "vivian"},
                        "params": {}, "metadata": {}}}})
                first_pcm = observe("completion", {"type": "stream", "request_id": rid + ":tts:0",
                    "from_stage": "tts", "chunk": {"audio": response_pcm, "index": 0}},
                    ns=self.tts_queue_ns)
                observe("completion", {"type": "complete", "request_id": rid + ":tts:0",
                    "from_stage": "tts", "success": True, "result": {}, "error": None})
                public_pcm = response_pcm[:-2] if self.drop_public_chunk else response_pcm
                capture_event({"type": "audio.delta", "response_id": rid, "segment_id": segment,
                               "format": "pcm16", "sample_rate": 24000, "channels": 1,
                               "audio": base64.b64encode(public_pcm).decode(),
                               "text_cursor": 0, "text_end_cursor": len(first_text.encode())})
                tail_pcm = b"\x03\x00" * 32
                observe("tts-submit", {"type": "submit", "request_id": rid + ":tts:1",
                    "data": {"request_id": rid + ":tts:1", "request": {"inputs": {
                        "text": "Response", "stream": True, "voice": "vivian"},
                        "params": {}, "metadata": {}}}})
                observe("completion", {"type": "stream", "request_id": rid + ":tts:1",
                    "from_stage": "tts", "chunk": {"audio": tail_pcm, "index": 0}})
                observe("completion", {"type": "complete", "request_id": rid + ":tts:1",
                    "from_stage": "tts", "success": True, "result": {}, "error": None})
                capture_event({"type": "audio.delta", "response_id": rid, "segment_id": segment,
                               "format": "pcm16", "sample_rate": 24000, "channels": 1,
                               "audio": base64.b64encode(tail_pcm).decode(),
                               "text_cursor": len(first_text.encode()) + 2,
                               "text_end_cursor": len(text.encode())})
                capture_event({"type": "audio.done", "response_id": rid, "segment_id": segment,
                               "text_cursor": len(text.encode())})
                capture_event({"type": "response.completed", "response_id": rid, "segment_id": segment})
                capture_event({"type": "output_audio_buffer.playback_ack", "response_id": rid,
                               "text_cursor": len(text.encode())}, tx=True)
                capture_event({"type": "response.playback_ack", "response_id": rid, "segment_id": segment,
                               "text_cursor": len(text.encode()), "final": True})
                history.extend([{"role": "user", "content": transcript},
                                {"role": "assistant", "content": text}])
                self.logs["asr"].append(f"asr request request_id={asr_rid} "
                                        f"audio_samples={len(committed_pcm) // 2} "
                                        "generated_tokens=2 chars=4 elapsed_ms=44")
                reused = 15 if arm["prefix_cache"] and turn_index > 0 else 0
                self.logs["vlm"].append(
                    f"qwen38 request request_id={rid}:vlm prompt_tokens=22 prefill_tokens=7 "
                    f"prefix_reused={reused} image_tokens=0 generated_tokens=4 finish_reason=stop "
                    "chars=21 vision_ms=0.0 prefill_ms=20.0 decode_ms=30.0 "
                    "decode_tok_per_s=1.0 elapsed_ms=50")
                self.logs["tts"].append(
                    f"tts stream_check request_id={rid}:tts:0 identical=1 chunks=1 samples=32 "
                    "streamed=32 first_diff=32 first_audio_ms=50")
                self.logs["tts"].append(f"tts request request_id={rid}:tts:0 voice=vivian "
                                        "language=english samples=32 elapsed_ms=60")
                self.logs["tts"].append(
                    f"tts stream_check request_id={rid}:tts:1 identical=1 chunks=1 samples=32 "
                    "streamed=32 first_diff=32 first_audio_ms=50")
                self.logs["tts"].append(f"tts request request_id={rid}:tts:1 voice=vivian "
                                        "language=english samples=32 elapsed_ms=60")
                # Keep the local variables explicitly tied to raw chronology in fixture reviews.
                require_order = stop_ns < asr_submit < asr_done < asr_final < prompt < first_ready < tts_submit < first_pcm
                if not require_order:
                    raise AssertionError("fixture clock order")
            capture.write(root / "frames" / f"{connection['name']}.jsonl")
        for channel, rows in self.rows.items():
            (root / "observers" / f"{channel}.jsonl").write_bytes(b"".join(
                (json.dumps(row, sort_keys=True) + "\n").encode() for row in rows))
        for kind, lines in self.logs.items():
            (root / f"{kind}.log").write_text("".join(line + "\n" for line in lines))

    def seal(self):
        report = first.analyze(self.root)
        (self.root / "report.json").write_text(first.render_report(report))
        files = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                files.append(f"{first.file_sha256(path)}  {path.relative_to(self.root)}\n")
        (self.root / "SHA256SUMS").write_text("".join(files))
        return report


class FirstAudioTests(unittest.TestCase):
    def test_plan_is_seeded_and_complete(self):
        contract = first.load_contract(ROOT / "benchmarks/realtime-first-audio/preregistration.json")
        plan = first.make_plan(contract)
        self.assertEqual(plan, first.make_plan(contract))
        self.assertEqual(sum(a["sessions"] * a["rounds"] for a in plan["arms"]), 81)
        self.assertEqual(plan["arms"][2]["schedule"][0]["start_order"], [5, 4, 3, 0, 2, 1])
        self.assertTrue(all(row["measured"] for row in plan["arms"][0]["schedule"]))
        self.assertFalse(plan["arms"][1]["schedule"][0]["measured"])

    def test_full_raw_fixture_fresh_process_and_ab(self):
        with tempfile.TemporaryDirectory() as temporary:
            baseline_root = EvidenceBuilder(Path(temporary) / "baseline").build()
            candidate_root = EvidenceBuilder(Path(temporary) / "candidate", tts_queue_ms=60).build()
            # Use the builder instances without invoking any driver code.
            baseline_builder = EvidenceBuilder.__new__(EvidenceBuilder)
            baseline_builder.root = baseline_root
            baseline_report = baseline_builder.seal()
            candidate_builder = EvidenceBuilder.__new__(EvidenceBuilder)
            candidate_builder.root = candidate_root
            candidate_report = candidate_builder.seal()
            self.assertEqual(baseline_report["counts"]["measured_responses"], 81)
            self.assertEqual(baseline_report["coverage"]["warm_prefix_reused_max"], 15)
            result = first.compare_reports(baseline_report, candidate_report)
            self.assertEqual(result["status"], "PASS")
            self.assertGreater(result["primary"]["median_gain_fraction"], 0.20)
            verify = subprocess.run([sys.executable, "-I", str(ROOT / "tests/realtime_first_audio_verify.py"),
                                     str(baseline_root)], capture_output=True, text=True)
            self.assertEqual(verify.returncode, 0, verify.stdout + verify.stderr)
            tampered = baseline_root / "arms" / "serial-cold-short" / "observers" / "tts-submit.jsonl"
            rows = tampered.read_text().splitlines()
            row = json.loads(rows[0])
            row["raw_base64"] = base64.b64encode(b"altered").decode()
            rows[0] = json.dumps(row, sort_keys=True)
            tampered.write_text("\n".join(rows) + "\n")
            with self.assertRaises(first.FirstAudioError):
                first.analyze(baseline_root)

    def test_missing_public_pcm_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = EvidenceBuilder(Path(temporary) / "missing-pcm",
                                   drop_public_chunk=True).build()
            with self.assertRaisesRegex(first.FirstAudioError, "native/public PCM mismatch"):
                first.analyze(root)

    def test_ab_rejects_quality_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            left = EvidenceBuilder(Path(temporary) / "left").build()
            right = EvidenceBuilder(Path(temporary) / "right", tts_queue_ms=60).build()
            left_builder = EvidenceBuilder.__new__(EvidenceBuilder)
            left_builder.root = left
            left_report = left_builder.seal()
            right_builder = EvidenceBuilder.__new__(EvidenceBuilder)
            right_builder.root = right
            right_report = right_builder.seal()
            right_report["responses"][0]["pcm_sha256"] = "0" * 64
            with self.assertRaises(first.FirstAudioError):
                first.compare_reports(left_report, right_report)

    @staticmethod
    def playback_timing_fixture():
        records = [
            {"direction": "rx", "monotonic_ns": 1000, "event": {
                "type": "audio.delta", "response_id": "req-old",
                "audio": base64.b64encode(b"abcdef").decode(), "text_end_cursor": 3}},
            {"direction": "rx", "monotonic_ns": 1100, "event": {
                "type": "audio.delta", "response_id": "req-old",
                "audio": base64.b64encode(b"ghijkl").decode(), "text_end_cursor": 6}},
            {"direction": "rx", "monotonic_ns": 1300, "event": {
                "type": "response.playback_ack", "response_id": "req-old", "text_cursor": 3}},
            {"direction": "rx", "monotonic_ns": 1320, "event": {
                "type": "response.playback_ack", "response_id": "req-old", "text_cursor": 3}},
            {"direction": "tx", "monotonic_ns": 1400, "event": {
                "type": "input_audio_buffer.append"}},
            {"direction": "rx", "monotonic_ns": 1500, "event": {
                "type": "response.cancelled", "response_id": "req-old", "text_cursor": 3}},
            {"direction": "rx", "monotonic_ns": 1600, "event": {
                "type": "input_audio_buffer.speech_started",
                "interrupted_response_id": "req-old"}},
        ]
        delayed = {"response_id": "req-old", "text_cursor": 3,
                   "barge_started_monotonic_ns": 1350,
                   "timed_playback_pcm_bytes": 6,
                   "timed_playback_pcm_sha256": sha(b"abcdef"),
                   "accepted_playback_ack_monotonic_ns": 1300,
                   "cancel_received_monotonic_ns": 1500,
                   "virtual_playback_silence_monotonic_ns": 1550,
                   "unplayed_pcm_bytes_discarded": 6,
                   "cancel_to_virtual_playback_silence_ns": 50}
        return delayed, records

    def test_ack_anchored_barge_timing_uses_client_receive_and_signed_notification(self):
        delayed, records = self.playback_timing_fixture()
        result = first.verify_delayed_playback_timing(delayed, records)
        self.assertEqual(result["first_append_to_cancel_including_vad_ns"], 100)
        self.assertEqual(result["speech_started_notification_to_cancel_signed_ns"], -100)
        self.assertEqual(result["accepted_playback_ack_to_cancel_ns"], 200)
        self.assertEqual(result["cancel_to_virtual_playback_silence_ns"], 50)
        self.assertEqual(result["physical_playback"], "NOT_COVERED")

    def test_ack_anchored_barge_timing_rejects_pcm_or_clock_tampering(self):
        delayed, records = self.playback_timing_fixture()
        delayed["timed_playback_pcm_sha256"] = sha(b"tampered")
        with self.assertRaisesRegex(first.FirstAudioError, "PCM prefix"):
            first.verify_delayed_playback_timing(delayed, records)
        delayed, records = self.playback_timing_fixture()
        delayed["virtual_playback_silence_monotonic_ns"] = 1499
        delayed["cancel_to_virtual_playback_silence_ns"] = -1
        with self.assertRaisesRegex(first.FirstAudioError, "silence interval"):
            first.verify_delayed_playback_timing(delayed, records)


if __name__ == "__main__":
    unittest.main()
