#!/usr/bin/env python3
"""Offline contract and unit regression tests for python/umminfer_flux2_stage.py.

Covers size validation, data-URL decoding, dependency-free PNG encoding, mock
backend determinism, request validation, abort tombstones, and mid-generation
cancellation. The pure-logic tests need nothing outside the standard library;
the ZMQ protocol tests run when pyzmq and msgpack are installed.
"""

import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
import time
import unittest
import zlib


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_PATH = REPO_ROOT / "python" / "umminfer_flux2_stage.py"


def load_stage_module():
    spec = importlib.util.spec_from_file_location("umminfer_flux2_stage", STAGE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["umminfer_flux2_stage"] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    import msgpack  # noqa: F401
    import zmq  # noqa: F401
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False

STAGE = load_stage_module()
TINY_PNG = STAGE.encode_png(16, 16, bytes(16 * 16 * 3))
TINY_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64


class PureLogicTests(unittest.TestCase):
    def test_parse_size_and_alignment(self):
        self.assertEqual(STAGE.parse_size(None, 1024, 1024, align=16), (1024, 1024))
        self.assertEqual(STAGE.parse_size("512x512", 1024, 1024, align=16), (512, 512))
        self.assertEqual(STAGE.parse_size("768X512", 1024, 1024, align=16), (768, 512))
        for bad in ("500x512", "not-a-size", "12x12", "4096x4096", "512"):
            with self.assertRaises(ValueError, msg=bad):
                STAGE.parse_size(bad, 1024, 1024, align=16)

    def test_encode_png_roundtrip(self):
        width, height = 24, 16
        rgb = bytes((i * 7) & 0xFF for i in range(width * height * 3))
        png = STAGE.encode_png(width, height, rgb)
        self.assertTrue(STAGE.is_valid_png(png))
        self.assertEqual(STAGE.png_dimensions(png), (width, height, 8, 2))
        # The IDAT payload must inflate back to PNG scanlines with filter 0.
        offset = 8
        idat = b""
        while offset < len(png):
            length = struct.unpack(">I", png[offset:offset + 4])[0]
            tag = png[offset + 4:offset + 8]
            payload = png[offset + 8:offset + 8 + length]
            if tag == b"IDAT":
                idat += payload
            offset += 12 + length
        raw = zlib.decompress(idat)
        self.assertEqual(len(raw), height * (1 + width * 3))
        self.assertTrue(all(raw[row * (1 + width * 3)] == 0 for row in range(height)))
        with self.assertRaises(ValueError):
            STAGE.encode_png(4, 4, b"\x00" * 10)

    def test_is_valid_png_rejects_garbage(self):
        self.assertFalse(STAGE.is_valid_png(b""))
        self.assertFalse(STAGE.is_valid_png(b"not a png at all, definitely not"))
        self.assertFalse(STAGE.is_valid_png(TINY_JPEG))

    def test_decode_data_url(self):
        png_url = "data:image/png;base64," + base64.b64encode(TINY_PNG).decode()
        self.assertEqual(STAGE.decode_data_url(png_url), TINY_PNG)
        jpeg_url = "data:image/jpeg;base64," + base64.b64encode(TINY_JPEG).decode()
        self.assertEqual(STAGE.decode_data_url(jpeg_url), TINY_JPEG)
        for bad in (
            "",
            "http://example.com/a.png",
            "/tmp/a.png",
            "data:image/png,notbase64",
            "data:image/png;base64,",
            "data:image/png;base64," + base64.b64encode(b"GIF89a").decode(),
            None,
        ):
            with self.assertRaises(ValueError, msg=repr(bad)):
                STAGE.decode_data_url(bad)
        with self.assertRaises(ValueError):
            STAGE.decode_data_url(png_url, max_bytes=4)

    def test_detect_model_revision(self):
        self.assertEqual(STAGE.detect_model_revision("/nonexistent", "abc123"), "abc123")
        self.assertEqual(STAGE.detect_model_revision("/nonexistent"), "UNKNOWN")
        with tempfile.TemporaryDirectory() as tmp:
            snap = Path(tmp) / "snapshots" / ("a" * 40)
            snap.mkdir(parents=True)
            self.assertEqual(STAGE.detect_model_revision(str(snap)), "a" * 40)

    def test_mock_backend_deterministic_and_edit_differs(self):
        backend = STAGE.MockFlux2Backend()
        backend.load()
        flag = threading.Event()
        base = STAGE.Flux2Request(
            prompt="a red lantern", width=32, height=32, steps=4,
            guidance_scale=4.0, seed=42)
        first = backend.generate(base, flag)
        second = backend.generate(base, flag)
        self.assertTrue(STAGE.is_valid_png(first.png_bytes))
        self.assertEqual(first.png_bytes, second.png_bytes)
        self.assertEqual(first.seed, 42)
        edited = STAGE.Flux2Request(
            prompt="a red lantern", width=32, height=32, steps=4,
            guidance_scale=4.0, seed=42, image_bytes=first.png_bytes)
        third = backend.generate(edited, flag)
        self.assertTrue(STAGE.is_valid_png(third.png_bytes))
        self.assertNotEqual(third.png_bytes, first.png_bytes)
        flag.set()
        with self.assertRaises(STAGE.RequestAborted):
            backend.generate(base, flag)


class RequestParsingTests(unittest.TestCase):
    def _stage(self):
        return STAGE.Flux2Stage(
            "image", "ipc:///tmp/unused1", "ipc:///tmp/unused2",
            "ipc:///tmp/unused3", STAGE.MockFlux2Backend())

    def _message(self, **inputs):
        return {
            "request_id": "req-1:image",
            "data": {"request": {"inputs": inputs}},
        }

    def test_parse_request_defaults_and_overrides(self):
        stage = self._stage()
        parsed = stage.parse_request(self._message(prompt="a red lantern"))
        self.assertEqual(parsed.prompt, "a red lantern")
        self.assertEqual((parsed.width, parsed.height), (1024, 1024))
        self.assertEqual(parsed.steps, 4)
        self.assertEqual(parsed.guidance_scale, 4.0)
        self.assertIsNone(parsed.seed)
        self.assertIsNone(parsed.image_bytes)

        parsed = stage.parse_request(self._message(
            prompt=" lantern ", size="512x512", steps=12, seed=7,
            guidance_scale=2.5, output_format="png"))
        self.assertEqual(parsed.prompt, "lantern")
        self.assertEqual((parsed.width, parsed.height), (512, 512))
        self.assertEqual(parsed.steps, 12)
        self.assertEqual(parsed.seed, 7)
        self.assertEqual(parsed.guidance_scale, 2.5)

    def test_parse_request_accepts_edit_image(self):
        stage = self._stage()
        url = "data:image/png;base64," + base64.b64encode(TINY_PNG).decode()
        parsed = stage.parse_request(self._message(prompt="sunset", image=url))
        self.assertEqual(parsed.image_bytes, TINY_PNG)

    def test_parse_request_accepts_content_addressed_binary_edit(self):
        stage = self._stage()
        digest = hashlib.sha256(TINY_PNG).hexdigest()
        parsed = stage.parse_request(self._message(
            prompt="same character, night scene", image=TINY_PNG,
            anchor_explicit=True, anchor_slot_index=0, anchor_sha256=digest))
        self.assertEqual(parsed.image_bytes, TINY_PNG)

    def test_parse_request_rejects_bad_content_addressed_edit(self):
        stage = self._stage()
        digest = hashlib.sha256(TINY_PNG).hexdigest()
        for inputs in (
            {"prompt": "x", "image": TINY_PNG, "anchor_explicit": True,
             "anchor_slot_index": 0, "anchor_sha256": "0" * 64},
            {"prompt": "x", "image": TINY_PNG + b"tampered",
             "anchor_explicit": True, "anchor_slot_index": 0,
             "anchor_sha256": digest},
            {"prompt": "x", "anchor_explicit": True,
             "anchor_slot_index": 0, "anchor_sha256": digest},
            {"prompt": "x", "image": TINY_PNG, "anchor_explicit": "true",
             "anchor_slot_index": 0, "anchor_sha256": digest},
            {"prompt": "x", "image": TINY_PNG, "anchor_explicit": True,
             "anchor_slot_index": -1, "anchor_sha256": digest},
            {"prompt": "x", "image": TINY_PNG, "anchor_sha256": digest},
        ):
            with self.assertRaises(ValueError, msg=json.dumps(inputs, default=str)):
                stage.parse_request(self._message(**inputs))

    def test_parse_request_rejects_invalid_input(self):
        stage = self._stage()
        for inputs in (
            {"prompt": ""},
            {"prompt": "   "},
            {"prompt": "x", "size": "500x512"},
            {"prompt": "x", "steps": 0},
            {"prompt": "x", "steps": 999},
            {"prompt": "x", "steps": True},
            {"prompt": "x", "seed": -1},
            {"prompt": "x", "seed": "42"},
            {"prompt": "x", "guidance_scale": float("nan")},
            {"prompt": "x", "output_format": "webp"},
            {"prompt": "x", "image": "http://example.com/a.png"},
        ):
            with self.assertRaises(ValueError, msg=json.dumps(inputs)):
                stage.parse_request(self._message(**inputs))
        with self.assertRaises(ValueError):
            stage.parse_request({"request_id": "r"})

    def test_complete_artifact_wire_shape(self):
        stage = self._stage()
        res = STAGE.Flux2Result(
            png_bytes=TINY_PNG, width=16, height=16, steps=4, seed=42,
            model="m", revision="UNKNOWN", backend="mock",
            elapsed_s=0.1, denoise_ms=0.0)
        message = stage.complete_artifact("req-1:image", res)
        self.assertEqual(message["type"], "complete")
        self.assertEqual(message["request_id"], "req-1:image")
        self.assertTrue(message["success"])
        artifact = message["result"]
        self.assertEqual(artifact["media_type"], "image")
        self.assertEqual(artifact["mime_type"], "image/png")
        self.assertEqual(artifact["format"], "png")
        self.assertEqual(artifact["size"], "16x16")
        self.assertEqual(artifact["width"], 16)
        self.assertEqual(artifact["height"], 16)
        self.assertEqual(artifact["seed"], 42)
        self.assertEqual(artifact["steps"], 4)
        self.assertEqual(artifact["data"], TINY_PNG)
        self.assertIn("model", artifact)
        self.assertIn("backend", artifact)

    def test_complete_error_shape(self):
        stage = self._stage()
        message = stage.complete_error("req-1:image", "bad input")
        self.assertEqual(message["type"], "complete")
        self.assertFalse(message["success"])
        self.assertEqual(message["error"], "bad input")
        self.assertEqual(message["result"], {})


@unittest.skipUnless(HAS_ZMQ, "pyzmq and msgpack required for Flux2Stage ZMQ tests")
class StageProtocolTests(unittest.TestCase):
    def _run_stage_pair(self, backend):
        tmpdir = Path(tempfile.mkdtemp(prefix="umm-flux2-test-"))
        recv_ep = f"ipc://{tmpdir}/image"
        done_ep = f"ipc://{tmpdir}/done"
        abort_ep = f"ipc://{tmpdir}/abort"

        ctx = zmq.Context()
        coord_pull = ctx.socket(zmq.PULL)
        coord_pull.RCVTIMEO = 5000
        coord_pull.bind(done_ep)
        abort_pub = ctx.socket(zmq.PUB)
        abort_pub.bind(abort_ep)

        stage = STAGE.Flux2Stage("image", recv_ep, done_ep, abort_ep, backend,
                                 default_width=256, default_height=256)
        stage.start()
        thread = threading.Thread(target=stage.run, daemon=True)
        thread.start()

        ready = STAGE.unpack_message(coord_pull.recv())
        self.assertEqual(ready["type"], "stage_ready")
        self.assertEqual(ready["from_stage"], "image")

        submit_push = ctx.socket(zmq.PUSH)
        submit_push.connect(recv_ep)
        time.sleep(0.05)  # allow SUB subscription propagation
        return tmpdir, ctx, coord_pull, abort_pub, submit_push, stage, thread

    def _cleanup(self, tmpdir, ctx, coord_pull, abort_pub, submit_push, thread):
        submit_push.send(STAGE.pack_message({"type": "shutdown"}))
        thread.join(timeout=5)
        for sock in (coord_pull, abort_pub, submit_push):
            sock.close(linger=0)
        ctx.term()
        for path in tmpdir.glob("*"):
            path.unlink(missing_ok=True)
        tmpdir.rmdir()

    def test_submit_complete_and_validation_error(self):
        backend = STAGE.MockFlux2Backend()
        res = self._run_stage_pair(backend)
        tmpdir, ctx, coord_pull, abort_pub, submit_push, stage, thread = res
        try:
            submit_push.send(STAGE.pack_message({
                "type": "submit",
                "request_id": "req-1:image",
                "data": {"request": {"inputs": {
                    "prompt": "a red lantern",
                    "size": "256x256",
                    "seed": 42,
                }}},
            }))
            done = STAGE.unpack_message(coord_pull.recv())
            self.assertEqual(done["type"], "complete")
            self.assertTrue(done["success"])
            self.assertEqual(done["result"]["media_type"], "image")
            self.assertTrue(STAGE.is_valid_png(done["result"]["data"]))

            submit_push.send(STAGE.pack_message({
                "type": "submit",
                "request_id": "req-2:image",
                "data": {"request": {"inputs": {"prompt": ""}}},
            }))
            failed = STAGE.unpack_message(coord_pull.recv())
            self.assertFalse(failed["success"])
            self.assertIn("image.prompt", failed["error"])
        finally:
            self._cleanup(tmpdir, ctx, coord_pull, abort_pub, submit_push, thread)

    def test_abort_tombstone_and_mid_generation(self):
        class SlowBackend(STAGE.MockFlux2Backend):
            def __init__(self):
                super().__init__()
                self.started = threading.Event()
                self.aborted = threading.Event()
                self.calls = []

            def generate(self, req, abort_flag):
                self.calls.append(req)
                self.started.set()
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline:
                    if abort_flag.is_set():
                        self.aborted.set()
                        raise STAGE.RequestAborted("aborted during stub step")
                    time.sleep(0.01)
                return super().generate(req, abort_flag)

        backend = SlowBackend()
        res = self._run_stage_pair(backend)
        tmpdir, ctx, coord_pull, abort_pub, submit_push, stage, thread = res
        try:
            abort_pub.send(STAGE.pack_message({
                "type": "abort", "request_id": "req-pre:image"}))
            time.sleep(0.05)
            submit_push.send(STAGE.pack_message({
                "type": "submit",
                "request_id": "req-pre:image",
                "data": {"request": {"inputs": {"prompt": "pre-aborted"}}},
            }))
            time.sleep(0.15)
            self.assertEqual(len(backend.calls), 0)

            submit_push.send(STAGE.pack_message({
                "type": "submit",
                "request_id": "req-mid:image",
                "data": {"request": {"inputs": {"prompt": "slow image"}}},
            }))
            self.assertTrue(backend.started.wait(timeout=3.0))
            abort_pub.send(STAGE.pack_message({
                "type": "abort", "request_id": "req-mid:image"}))
            self.assertTrue(backend.aborted.wait(timeout=3.0))
            deadline = time.monotonic() + 3.0
            while stage.inflight and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(len(stage.inflight), 0)
            coord_pull.RCVTIMEO = 200
            with self.assertRaises(zmq.Again):
                coord_pull.recv()
        finally:
            self._cleanup(tmpdir, ctx, coord_pull, abort_pub, submit_push, thread)


class ServiceDescriptionTests(unittest.TestCase):
    def test_optional_stages_include_flux2_and_video(self):
        data = json.loads(
            (REPO_ROOT / "service" / "native-service.json").read_text(encoding="utf-8"))
        names = [entry.get("name") for entry in data.get("optional_stages", [])]
        self.assertIn("flux2", names)
        self.assertIn("video", names)
        # The four required native stages stay untouched.
        self.assertEqual(
            [entry["name"] for entry in data["native_stages"]],
            ["asr", "qwen38", "image", "tts"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
