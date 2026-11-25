#!/usr/bin/env python3
"""Offline contract and unit regression tests for python/umminfer_video_stage.py.

Covers request validation, Wan frame alignment, MP4 header checks, abort
tombstones, mid-generation cancellation, failure propagation, and the
replaceable OpenAI `/v1/videos` upstream adapter without requiring a GPU.
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_PATH = REPO_ROOT / "python" / "umminfer_video_stage.py"


def load_stage_module():
    spec = importlib.util.spec_from_file_location("umminfer_video_stage", STAGE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["umminfer_video_stage"] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    import msgpack  # noqa: F401
    import zmq  # noqa: F401
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False

STAGE = load_stage_module()
FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 64


class FakeVideoBackend:
    """In-process stub backend for offline ZMQ stage protocol contract tests."""

    def __init__(self, delay_s=0.0, fail_error=None):
        self.model_id = "stub-video-model"
        self.revision = "0123456789abcdef0123456789abcdef01234567"
        self.spatial_align = 32
        self.temporal_stride = 4
        self.delay_s = delay_s
        self.fail_error = fail_error
        self.started = threading.Event()
        self.aborted = threading.Event()
        self.calls = []

    def load(self):
        return

    def generate(self, req, abort_flag):
        self.calls.append(req)
        self.started.set()
        if self.delay_s > 0:
            deadline = time.monotonic() + self.delay_s
            while time.monotonic() < deadline:
                if abort_flag.is_set():
                    self.aborted.set()
                    raise STAGE.RequestAborted("aborted during stub step")
                time.sleep(0.01)
        if abort_flag.is_set():
            self.aborted.set()
            raise STAGE.RequestAborted("aborted before stub completion")
        if self.fail_error:
            raise RuntimeError(self.fail_error)
        return STAGE.VideoGenerationResult(
            mp4_bytes=FAKE_MP4,
            width=req.width,
            height=req.height,
            fps=req.fps,
            frames=req.num_frames,
            seconds=round(req.num_frames / float(req.fps), 3),
            model=self.model_id,
            revision=self.revision,
            elapsed_s=0.05,
        )


class VideoStageUnitTests(unittest.TestCase):
    def test_parse_size_and_alignment(self):
        self.assertEqual(STAGE.parse_size(None, 704, 480, align=32), (704, 480))
        self.assertEqual(STAGE.parse_size("512x320", 704, 480, align=32), (512, 320))
        with self.assertRaises(ValueError):
            STAGE.parse_size("500x480", 704, 480, align=32)
        with self.assertRaises(ValueError):
            STAGE.parse_size("not-a-size", 704, 480, align=32)
        with self.assertRaises(ValueError):
            STAGE.parse_size("64x64", 704, 480, align=32)

    def test_align_frame_count(self):
        self.assertEqual(STAGE.align_frame_count(1, 16, 81, 4), 13)
        self.assertEqual(STAGE.align_frame_count(2, 16, 81, 4), 29)
        self.assertEqual(STAGE.align_frame_count(5, 24, 81, 4), 81)
        self.assertEqual((STAGE.align_frame_count(3, 16, 81, 4) - 1) % 4, 0)
        with self.assertRaises(ValueError):
            STAGE.align_frame_count(0, 16, 81, 4)

    def test_mp4_header_and_revision_detection(self):
        self.assertTrue(STAGE.is_valid_mp4_header(FAKE_MP4))
        self.assertFalse(STAGE.is_valid_mp4_header(b"not an mp4"))
        self.assertFalse(STAGE.is_valid_mp4_header(b""))
        self.assertEqual(
            STAGE.detect_model_revision("/nonexistent", "abc123"), "abc123")
        self.assertEqual(
            STAGE.detect_model_revision("/nonexistent"), "UNKNOWN")

    def test_serve_cli_accepts_video_entry_flags(self):
        binary = REPO_ROOT / "zig-out" / "bin" / "umm-infer"
        if not binary.is_file():
            self.skipTest("umm-infer binary not built in zig-out/bin")
        import subprocess
        proc = subprocess.run(
            [str(binary), "serve", "--video-entry", "video"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertNotIn("unknown option", proc.stderr)
        self.assertIn("optional output stages need both", proc.stderr)

    def test_parse_request_validation(self):
        backend = FakeVideoBackend()
        stage = STAGE.VideoStage("video", "ipc:///tmp/unused1",
                                 "ipc:///tmp/unused2", "ipc:///tmp/unused3",
                                 backend)
        msg = {
            "request_id": "req-1:video",
            "data": {
                "request": {
                    "inputs": {
                        "prompt": "a red lantern floating on calm water",
                        "size": "512x320",
                        "seconds": 2,
                        "fps": 16,
                        "steps": 12,
                        "seed": 42,
                        "output_format": "mp4",
                    }
                }
            },
        }
        parsed = stage.parse_request(msg)
        self.assertEqual(parsed.prompt, "a red lantern floating on calm water")
        self.assertEqual((parsed.width, parsed.height), (512, 320))
        self.assertEqual(parsed.num_frames, 29)
        self.assertEqual(parsed.steps, 12)
        self.assertEqual(parsed.seed, 42)

        bad_fmt = json.loads(json.dumps(msg))
        bad_fmt["data"]["request"]["inputs"]["output_format"] = "webm"
        with self.assertRaises(ValueError):
            stage.parse_request(bad_fmt)

    def test_parse_explicit_i2v_image_and_anchor_hash(self):
        stage = STAGE.VideoStage(
            "video", "ipc:///tmp/unused1", "ipc:///tmp/unused2",
            "ipc:///tmp/unused3", FakeVideoBackend())
        source = b"settled-source-image"
        digest = hashlib.sha256(source).hexdigest()
        message = {"data": {"request": {"inputs": {
            "prompt": "animate the source", "size": "512x320",
            "num_frames": 13, "image": source,
            "anchor_explicit": True, "anchor_sha256": digest,
        }}}}
        parsed = stage.parse_request(message)
        self.assertEqual(parsed.init_image_bytes, source)

        message["data"]["request"]["inputs"]["image"] = (
            "data:image/png;base64," + base64.b64encode(source).decode("ascii"))
        self.assertEqual(stage.parse_request(message).init_image_bytes, source)
        message["data"]["request"]["inputs"]["anchor_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "anchor_sha256"):
            stage.parse_request(message)
        del message["data"]["request"]["inputs"]["image"]
        message["data"]["request"]["inputs"]["anchor_sha256"] = digest
        with self.assertRaisesRegex(ValueError, "requires video.image"):
            stage.parse_request(message)


@unittest.skipUnless(HAS_ZMQ, "pyzmq and msgpack required for VideoStage ZMQ tests")
class VideoStageProtocolTests(unittest.TestCase):
    def _run_stage_pair(self, backend):
        tmpdir = Path(tempfile.mkdtemp(prefix="umm-video-test-"))
        recv_ep = f"ipc://{tmpdir}/video"
        done_ep = f"ipc://{tmpdir}/done"
        abort_ep = f"ipc://{tmpdir}/abort"

        ctx = zmq.Context()
        coord_pull = ctx.socket(zmq.PULL)
        coord_pull.RCVTIMEO = 5000
        coord_pull.bind(done_ep)
        abort_pub = ctx.socket(zmq.PUB)
        abort_pub.bind(abort_ep)

        stage = STAGE.VideoStage("video", recv_ep, done_ep, abort_ep, backend)
        stage.start()
        thread = threading.Thread(target=stage.run, daemon=True)
        thread.start()

        ready = STAGE.unpack_message(coord_pull.recv())
        self.assertEqual(ready["type"], "stage_ready")
        self.assertEqual(ready["from_stage"], "video")

        submit_push = ctx.socket(zmq.PUSH)
        submit_push.connect(recv_ep)
        time.sleep(0.05)  # allow SUB subscription propagation
        return tmpdir, ctx, coord_pull, abort_pub, submit_push, stage, thread

    def _cleanup(self, tmpdir, ctx, coord_pull, abort_pub, submit_push, stage, thread):
        submit_push.send(STAGE.pack_message({"type": "shutdown"}))
        thread.join(timeout=5)
        for sock in (coord_pull, abort_pub, submit_push):
            sock.close(linger=0)
        ctx.term()
        for p in tmpdir.glob("*"):
            p.unlink(missing_ok=True)
        tmpdir.rmdir()

    def test_submit_complete_and_error(self):
        backend = FakeVideoBackend()
        res = self._run_stage_pair(backend)
        tmpdir, ctx, coord_pull, abort_pub, submit_push, stage, thread = res
        try:
            submit_push.send(STAGE.pack_message({
                "type": "submit",
                "request_id": "req-1:video",
                "data": {"request": {"inputs": {
                    "prompt": "ocean waves at sunset",
                    "size": "512x320",
                    "seconds": 1,
                    "fps": 16,
                    "output_format": "mp4",
                }}},
            }))
            done = STAGE.unpack_message(coord_pull.recv())
            self.assertEqual(done["type"], "complete")
            self.assertEqual(done["request_id"], "req-1:video")
            self.assertTrue(done["success"])
            artifact = done["result"]
            self.assertEqual(artifact["media_type"], "video")
            self.assertEqual(artifact["mime_type"], "video/mp4")
            self.assertEqual(artifact["format"], "mp4")
            self.assertEqual(artifact["data"], FAKE_MP4)
            self.assertEqual(artifact["model"], "stub-video-model")
            self.assertEqual(artifact["revision"], backend.revision)

            # Invalid input must return complete(success=False) rather than crashing.
            submit_push.send(STAGE.pack_message({
                "type": "submit",
                "request_id": "req-2:video",
                "data": {"request": {"inputs": {"prompt": ""}}},
            }))
            failed = STAGE.unpack_message(coord_pull.recv())
            self.assertEqual(failed["type"], "complete")
            self.assertEqual(failed["request_id"], "req-2:video")
            self.assertFalse(failed["success"])
            self.assertIn("video.prompt", failed["error"])
        finally:
            self._cleanup(tmpdir, ctx, coord_pull, abort_pub, submit_push, stage, thread)

    def test_mid_generation_abort_and_tombstone(self):
        backend = FakeVideoBackend(delay_s=2.0)
        res = self._run_stage_pair(backend)
        tmpdir, ctx, coord_pull, abort_pub, submit_push, stage, thread = res
        try:
            # 1) Abort-before-submit tombstone: must never start generation.
            abort_pub.send(STAGE.pack_message({
                "type": "abort", "request_id": "req-pre:video",
            }))
            time.sleep(0.05)
            submit_push.send(STAGE.pack_message({
                "type": "submit",
                "request_id": "req-pre:video",
                "data": {"request": {"inputs": {"prompt": "pre-aborted"}}},
            }))
            time.sleep(0.1)
            self.assertEqual(len(backend.calls), 0)

            # 2) Mid-generation abort: interrupts worker and emits no complete.
            submit_push.send(STAGE.pack_message({
                "type": "submit",
                "request_id": "req-mid:video",
                "data": {"request": {"inputs": {"prompt": "slow video"}}},
            }))
            self.assertTrue(backend.started.wait(timeout=3.0))
            abort_pub.send(STAGE.pack_message({
                "type": "abort", "request_id": "req-mid:video",
            }))
            self.assertTrue(backend.aborted.wait(timeout=3.0))
            # Wait briefly for drain_outbox to clear inflight.
            deadline = time.monotonic() + 2.0
            while stage.inflight and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(len(stage.inflight), 0)
            coord_pull.RCVTIMEO = 150
            with self.assertRaises(zmq.Again):
                coord_pull.recv()
        finally:
            self._cleanup(tmpdir, ctx, coord_pull, abort_pub, submit_push, stage, thread)


class UpstreamVideoBackendTests(unittest.TestCase):
    def test_upstream_roundtrip_and_abort_delete(self):
        state = {"polls": 0, "deleted": [], "mode": "ok"}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                _ = self.rfile.read(length)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"id": "vid-1", "status": "queued"}).encode())

            def do_GET(self):
                if self.path.endswith("/content"):
                    self.send_response(200)
                    self.send_header("Content-Type", "video/mp4")
                    self.end_headers()
                    self.wfile.write(FAKE_MP4 if state["mode"] == "ok" else b"bad")
                    return
                state["polls"] += 1
                status = "completed" if state["mode"] == "ok" and state["polls"] >= 2 else "in_progress"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"id": "vid-1", "status": status}).encode())

            def do_DELETE(self):
                state["deleted"].append(self.path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            backend = STAGE.UpstreamVideoBackend(
                base_url=f"http://127.0.0.1:{port}",
                model="upstream-video-v1",
                poll_interval_s=0.02,
            )
            backend.load()
            req = STAGE.VideoRequest(
                prompt="clouds over mountains",
                width=512, height=320, seconds=2, fps=16,
                num_frames=29, steps=10, guidance_scale=5.0,
                seed=7, negative_prompt="", output_format="mp4",
            )
            out = backend.generate(req, threading.Event())
            self.assertEqual(out.mp4_bytes, FAKE_MP4)
            self.assertEqual(out.model, "upstream-video-v1")
            self.assertEqual(out.revision, "UNKNOWN")

            # Abort during polling must trigger DELETE /v1/videos/vid-1.
            state["mode"] = "hang"
            flag = threading.Event()
            threading.Timer(0.08, flag.set).start()
            with self.assertRaises(STAGE.RequestAborted):
                backend.generate(req, flag)
            self.assertIn("/v1/videos/vid-1", state["deleted"])
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
