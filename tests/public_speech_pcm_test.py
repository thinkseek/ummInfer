#!/usr/bin/env python3
"""Deterministic public/native PCM verifier regression tests."""

import ast
import base64
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import threading
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "public_speech_pcm_verify", ROOT / "tests/public_speech_pcm_verify.py")
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


class PublicSpeechPcmTest(unittest.TestCase):
    def native(self, index, payload, first_frame=0):
        return {"index": index, "audio": payload, "first_frame": first_frame,
                "frames": len(payload) // 3840}

    def public(self, index, payload):
        return {"stream_index": index, "pcm": payload}

    def test_exact_chunk_binding_and_seeded_fragment_join(self):
        rng = random.Random(5630241)
        source = bytes(rng.randrange(256) for _ in range(30720))
        cuts = []
        cursor = 0
        while cursor < len(source):
            width = min(rng.randint(2, 4094), len(source) - cursor)
            width -= width % 2
            if not width:
                width = 2
            cuts.append(source[cursor:cursor + width])
            cursor += width
        self.assertEqual(b"".join(cuts), source)
        native = [self.native(0, cuts[0]), self.native(1, cuts[1], 2)]
        public = [self.public(0, cuts[0]), self.public(1, cuts[1])]
        rows = verify.bind_chunks(public, native, "exact")
        self.assertEqual([row["pcm_sha256"] for row in rows],
                         [verify.sha(cuts[0]), verify.sha(cuts[1])])

    def test_missing_chunk_fails_closed(self):
        native = [self.native(0, b"ab"), self.native(1, b"cd")]
        with self.assertRaisesRegex(AssertionError, "chunk count mismatch"):
            verify.bind_chunks([self.public(0, b"ab")], native, "missing")

    def test_stale_or_reordered_chunk_fails_closed(self):
        native = [self.native(0, b"ab"), self.native(1, b"cd")]
        with self.assertRaisesRegex(AssertionError, "chunk index mismatch"):
            verify.bind_chunks([self.public(1, b"ab"), self.public(0, b"cd")],
                               native, "stale")
        with self.assertRaisesRegex(AssertionError, "PCM mismatch"):
            verify.bind_chunks([self.public(0, b"ab"), self.public(1, b"XX")],
                               native, "tampered")

    def test_cursor_binding_is_utf8_byte_exact(self):
        text = "Opening. 第二句。"
        begin, end = verify.cursor_for_text(text, "第二句", 0, "utf8")
        self.assertEqual(text.encode()[begin:end].decode(), "第二句")
        self.assertEqual(begin, len("Opening. ".encode()))
        with self.assertRaisesRegex(AssertionError, "absent"):
            verify.cursor_for_text(text, "missing", 0, "bad-cursor")

    def test_terminal_and_sequence_reject_stale_or_missing_events(self):
        events = [
            {"unified": {"kind": "vlm_delta", "seq": 0, "text": "A"},
             "choices": [{"delta": {"content": "A"}}]},
            {"unified": {"kind": "complete", "seq": 1}},
        ]
        verify.check_terminal(events, "complete", "complete")
        stale = events + [{"unified": {"kind": "artifact", "seq": 2}}]
        with self.assertRaisesRegex(AssertionError, "stale public event"):
            verify.check_terminal(stale, "complete", "stale")
        with self.assertRaisesRegex(AssertionError, "sequence gap"):
            verify.check_terminal([events[0], {"unified": {"kind": "complete", "seq": 2}}],
                                  "complete", "gap")

    def test_raw_chunked_http_reconstruction(self):
        body = b'data: {"ok":true}\n\ndata: [DONE]\n\n'
        chunks = [body[:1], body[1:9], body[9:17], body[17:]]
        wire = b"".join(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n"
                        for chunk in chunks) + b"0\r\n\r\n"
        raw = (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
               b"Transfer-Encoding: chunked\r\n\r\n" + wire)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "response.raw"
            path.write_bytes(raw)
            _, _, decoded, retained = verify.parse_http_raw(path)
        self.assertEqual(decoded, body)
        self.assertEqual(retained, raw)
        events = verify.parse_sse(decoded, "fragmented")
        self.assertEqual(events, [{"ok": True}])

    def test_native_chunk_indices_must_be_consecutive_even_without_public_indices(self):
        native = [self.native(0, b"ab"), self.native(0, b"cd")]
        with self.assertRaisesRegex(AssertionError, "native chunk index gap/reorder"):
            verify.bind_chunks([self.public(0, b"ab"), self.public(0, b"cd")],
                               native, "duplicate-native", check_index=False)

    def test_cursor_gaps_and_bounds_fail_closed(self):
        with self.assertRaisesRegex(AssertionError, "non-whitespace PCM cursor gap"):
            verify.require_whitespace_gap(b"abc def", 0, 4, "unspoken-prefix")
        verify.require_whitespace_gap(b"abc \n\t", 3, 6, "whitespace")
        with self.assertRaisesRegex(AssertionError, "cursor outside response text"):
            verify.require_whitespace_gap(b"abc", 0, 4, "oversize")

    def test_realtime_ack_requires_issued_monotonic_cursor(self):
        rows = [
            {"event": {"type": "audio.delta", "text_cursor": 0}},
            {"event": {"type": "response.playback_ack", "text_cursor": 0}},
            {"event": {"type": "audio.delta", "text_cursor": 2}},
            {"event": {"type": "response.playback_ack", "text_cursor": 2}},
            {"event": {"type": "response.playback_ack", "text_cursor": 2}},
        ]
        accepted = verify.validate_realtime_acks(rows, "r1")
        self.assertEqual(len(accepted), 3)
        with self.assertRaisesRegex(AssertionError, "unissued"):
            verify.validate_realtime_acks(rows + [
                {"event": {"type": "response.playback_ack", "text_cursor": 3}}], "r1")
        with self.assertRaisesRegex(AssertionError, "decreasing"):
            verify.validate_realtime_acks(rows + [
                {"event": {"type": "audio.delta", "text_cursor": 1}},
                {"event": {"type": "response.playback_ack", "text_cursor": 1}}], "r1")

    def test_cancel_text_is_prefix_but_success_is_exact(self):
        self.assertTrue(verify.native_public_text_matches("abcdef", "abc", True))
        self.assertFalse(verify.native_public_text_matches("abXdef", "abc", True))
        self.assertFalse(verify.native_public_text_matches("abcdef", "abc", False))
        self.assertTrue(verify.native_public_text_matches("abc", "abc", False))

    def test_vlm_id_compatibility_is_unambiguous(self):
        self.assertEqual(verify.vlm_request_id({"http-1": {}}, "http-1"), "http-1")
        self.assertEqual(verify.vlm_request_id({"http-1:vlm": {}}, "http-1"), "http-1:vlm")
        with self.assertRaisesRegex(AssertionError, "missing or ambiguous"):
            verify.vlm_request_id({"http-1": {}, "http-1:vlm": {}}, "http-1")
        with self.assertRaisesRegex(AssertionError, "missing or ambiguous"):
            verify.vlm_request_id({}, "http-1")

    def test_submit_edge_retains_control_without_treating_it_as_work(self):
        self.assertIsNone(verify.submission_or_none(
            {"type": "abort", "request_id": "http-20:vlm"}, "vlm-submit"))
        self.assertIsNone(verify.submission_or_none({"type": "shutdown"}, "tts-submit"))
        submit = {"type": "submit", "request_id": "req-1", "data": {"request": {"inputs": {}}}}
        self.assertIs(verify.submission_or_none(submit, "tts-submit"), submit)
        with self.assertRaisesRegex(AssertionError, "unexpected message"):
            verify.submission_or_none({"type": "stream"}, "tts-submit")

    def test_realtime_clients_use_pausable_socket_and_restore_constructor(self):
        live_spec = importlib.util.spec_from_file_location(
            "public_speech_pcm_live_constructor_test", ROOT / "tests/public_speech_pcm_live.py")
        live = importlib.util.module_from_spec(live_spec)
        live_spec.loader.exec_module(live)
        original_client = live.base.LiveClient
        original_raw = live.base.RawWebSocket
        seen = []
        class Args:
            realtime_sessions = 3
        try:
            live.base.LiveClient = lambda session, generation, port, root: seen.append(
                (session, live.base.RawWebSocket)) or object()
            clients = live.make_realtime_clients(Args(), Path("/unused"), (1, 2))
            self.assertEqual(len(clients), 3)
            self.assertEqual(seen, [(index, live.PausableRawWebSocket) for index in range(3)])
            self.assertIs(live.base.RawWebSocket, original_raw)
        finally:
            live.base.LiveClient = original_client
            live.base.RawWebSocket = original_raw

    def test_audio_backlog_release_is_fail_closed(self):
        live_spec = importlib.util.spec_from_file_location(
            "public_speech_pcm_live_test", ROOT / "tests/public_speech_pcm_live.py")
        live = importlib.util.module_from_spec(live_spec)
        live_spec.loader.exec_module(live)
        live.BACKPRESSURE_RELEASE.clear()
        with live.BACKPRESSURE_LOCK:
            live.BACKPRESSURE.update({"enabled": True, "native": 4096, "public": 1,
                                      "paused": 2, "target": 2, "release": None})
        live.backpressure_maybe_release()
        self.assertFalse(live.BACKPRESSURE_RELEASE.is_set(),
                         "4095 pending audio events must not release readers")
        with live.BACKPRESSURE_LOCK:
            live.BACKPRESSURE["native"] = 4097
        live.backpressure_maybe_release()
        self.assertTrue(live.BACKPRESSURE_RELEASE.is_set())
        release = live.BACKPRESSURE["release"]
        self.assertEqual(release["pending_native_minus_public"], 4096)
        self.assertEqual(release["native_audio_events"], 4097)
        self.assertEqual(release["public_audio_events"], 1)
        self.assertNotIn("native_stream_events", release)
        with live.BACKPRESSURE_LOCK:
            live.BACKPRESSURE["enabled"] = False

    def test_backlog_requires_an_active_pause_but_not_every_configured_reader(self):
        live_spec = importlib.util.spec_from_file_location(
            "public_speech_pcm_live_partial_pause_test", ROOT / "tests/public_speech_pcm_live.py")
        live = importlib.util.module_from_spec(live_spec)
        live_spec.loader.exec_module(live)
        live.BACKPRESSURE_RELEASE.clear()
        with live.BACKPRESSURE_LOCK:
            live.BACKPRESSURE.update({"enabled": True, "native": 4096, "public": 0,
                                      "paused": 0, "target": 29, "release": None})
        live.backpressure_maybe_release()
        self.assertFalse(live.BACKPRESSURE_RELEASE.is_set())
        with live.BACKPRESSURE_LOCK:
            live.BACKPRESSURE["paused"] = 1
            live.BACKPRESSURE["paused_ids"] = {"req-1"}
        live.backpressure_maybe_release()
        self.assertTrue(live.BACKPRESSURE_RELEASE.is_set())
        self.assertEqual(live.BACKPRESSURE["release"]["pending_native_minus_public"], 4096)
        self.assertEqual(live.BACKPRESSURE["release"]["configured_readers"], 29)
        self.assertEqual(live.BACKPRESSURE["release"]["paused_response_ids"], ["req-1"])
        with live.BACKPRESSURE_LOCK:
            live.BACKPRESSURE["enabled"] = False
            live.BACKPRESSURE["paused"] = 0
            live.BACKPRESSURE["paused_ids"] = set()

    def test_delayed_reader_snapshot_uses_long_fail_closed_deadline(self):
        live_spec = importlib.util.spec_from_file_location(
            "public_speech_pcm_live_snapshot_test", ROOT / "tests/public_speech_pcm_live.py")
        live = importlib.util.module_from_spec(live_spec)
        live_spec.loader.exec_module(live)
        terminal = {"event_index": 2, "event": {"type": "response.completed", "response_id": "r1"}}
        records = [
            {"event": {"type": "response.text.delta", "response_id": "r1", "text": "A"}},
            {"event": {"type": "audio.delta", "response_id": "r1",
                       "audio": base64.b64encode(b"\x01\x00").decode(), "text_cursor": 0}},
            terminal,
        ]
        class FakeWs:
            def __init__(self):
                self.timeout = None
            def wait_for(self, predicate, after, timeout, description):
                self.timeout = timeout
                return terminal
            def since(self, after, types=None, response_id=None):
                return records if types is None else [terminal]
        ws = FakeWs()
        client = types.SimpleNamespace(ws=ws, terminal_callbacks=[])
        snapshot = live.realtime_response_snapshot(client, "r1", 0)
        self.assertEqual(ws.timeout, 1800)
        self.assertEqual(snapshot["text"], "A")
        self.assertEqual(snapshot["pcm"], b"\x01\x00")
        self.assertEqual(client.terminal_callbacks, ["response.completed"])

    def test_live_chunk_decoder_reassembles_fragmented_artifact_marker(self):
        live_spec = importlib.util.spec_from_file_location(
            "public_speech_pcm_live_chunk_test", ROOT / "tests/public_speech_pcm_live.py")
        live = importlib.util.module_from_spec(live_spec)
        live_spec.loader.exec_module(live)
        event = b'data: {"unified":{"kind":"artifact","artifact":{"data":"AA=="}}}\n\n'
        wire = f"{len(event):x}\r\n".encode() + event + b"\r\n0\r\n\r\n"
        decoder = live.ChunkedDecoder()
        decoded = bytearray()
        for value in wire:
            decoded.extend(decoder.feed(bytes([value])))
        self.assertTrue(decoder.done)
        self.assertEqual(bytes(decoded), event)
        self.assertEqual(json.loads(event[6:].decode())["unified"]["kind"], "artifact")

    def test_paused_reader_heartbeat_stops_on_release_and_timeout(self):
        live_spec = importlib.util.spec_from_file_location(
            "public_speech_pcm_live_heartbeat_test", ROOT / "tests/public_speech_pcm_live.py")
        live = importlib.util.module_from_spec(live_spec)
        live_spec.loader.exec_module(live)
        ws = live.PausableRawWebSocket.__new__(live.PausableRawWebSocket)
        ws.heartbeat_interval_s = 0.01
        sent = []
        ws.send_frame = lambda opcode, payload: sent.append((opcode, payload))
        live.BACKPRESSURE_RELEASE.clear()
        timer = threading.Timer(0.035, live.BACKPRESSURE_RELEASE.set)
        timer.start()
        ws.wait_for_release_with_heartbeat(1)
        timer.join()
        self.assertTrue(sent and all(opcode == 9 and payload == b"" for opcode, payload in sent))
        stopped_at = len(sent)
        import time
        time.sleep(0.03)
        self.assertEqual(len(sent), stopped_at)
        live.BACKPRESSURE_RELEASE.clear()
        with self.assertRaisesRegex(AssertionError, "timed out"):
            ws.wait_for_release_with_heartbeat(0.02)

    def test_realtime_verifier_requires_exact_backlog_and_all_paused_readers(self):
        for paused, pending in ((28, 4096), (29, 4095)):
            with self.subTest(paused=paused, pending=pending), \
                    tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "realtime").mkdir()
                identities = [{"response_id": f"req-{index}"} for index in range(29)]
                (root / "realtime/responses.json").write_text(json.dumps(identities))
                (root / "realtime/turn-failures.jsonl").write_text("")
                release = {"configured_readers": 29, "paused_readers": paused,
                           "native_audio_events": pending + 29,
                           "public_audio_events": 29,
                           "pending_native_minus_public": pending}
                (root / "realtime/backpressure-release.json").write_text(json.dumps(release))
                with self.assertRaisesRegex(AssertionError, "exact audio-only bounded backlog"):
                    verify.verify_realtime(root, {"realtime": {}}, {}, {}, {}, [])

    def test_realtime_gate_has_exactly_one_measured_response_per_session(self):
        tree = ast.parse((ROOT / "tests/public_speech_pcm_live.py").read_text())
        block = next(node for node in ast.walk(tree)
                     if isinstance(node, ast.FunctionDef) and node.name == "realtime_block")
        measured = [node for node in ast.walk(block)
                    if isinstance(node, ast.Call) and
                    isinstance(node.func, ast.Attribute) and node.func.attr == "map" and
                    isinstance(node.func.value, ast.Name) and node.func.value.id == "pool"]
        self.assertEqual(len(measured), 1,
                         "the 29-session gate must not repeat its paused response round")

    def test_masked_websocket_fragment_reconstruction(self):
        payload = json.dumps({"type": "audio.done", "text_cursor": 4}).encode()
        mask = b"\x01\x02\x03\x04"
        masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
        first = bytes([0x01, 0x80 | len(payload)]) + mask + masked
        second = bytes([0x80, 0x80]) + mask
        rows = [
            {"direction": "tx", "monotonic_ns": 1, "index": 0,
             "raw_base64": base64.b64encode(first).decode()},
            {"direction": "tx", "monotonic_ns": 2, "index": 1,
             "raw_base64": base64.b64encode(second).decode()},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            events = verify.ws_events(path)
        self.assertEqual(events[0]["event"], {"type": "audio.done", "text_cursor": 4})


if __name__ == "__main__":
    unittest.main()
