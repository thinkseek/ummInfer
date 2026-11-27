#!/usr/bin/env python3
"""Replaceable FLUX.2-klein image stage for the ummInfer UnifiedEngine.

Speaks the native ZeroMQ + MessagePack stage protocol (`stage_ready`, `submit`,
`abort`, `shutdown`, `complete`) so a local Diffusers FLUX.2-klein checkpoint
plugs into `umm-infer serve --image-entry image --image-entry-endpoint
ipc://...` without changing the C++ control plane.

Supports text-to-image (`inputs.prompt`) and image editing (`inputs.image`,
a base64 data URL or exact MessagePack binary) through the same
`Flux2KleinPipeline`. Explicit narrative references additionally bind the binary
source to `anchor_slot_index` and a verified `anchor_sha256` before generation.
"""

import argparse
import base64
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import queue
import random
import signal
import struct
import sys
import threading
import time
import zlib

try:
    import msgpack
except ImportError:  # pragma: no cover - optional for pure-Python unit tests
    msgpack = None

try:
    import zmq
except ImportError:  # pragma: no cover - optional for pure-Python unit tests
    zmq = None


# An abort that arrives before its submit leaves a tombstone. The tombstone is
# removed when the submit arrives, or by this deadline if no submit follows.
ABORT_TOMBSTONE_TTL_S = 60.0

# Mirror UnifiedEngine kMaxArtifactImageBytes (48 MiB) so oversized encodings
# fail fast inside the stage with a clear error message.
MAX_IMAGE_ARTIFACT_BYTES = 48 * 1024 * 1024

# Mirror the stage_common.cpp image input limit (48 MiB decoded image bytes).
MAX_INPUT_IMAGE_BYTES = 48 * 1024 * 1024

# Periodic stage_ready re-registration. A restarted gateway rebuilds its ready
# set from these frames, so a server restart never requires a stage restart.
READY_HEARTBEAT_S = 1.0

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class RequestAborted(Exception):
    """Raised inside a backend when the stage receives a coordinator abort."""


@dataclass(frozen=True)
class Flux2Request:
    prompt: str
    width: int
    height: int
    steps: int
    guidance_scale: float
    seed: int | None
    image_bytes: bytes | None = None


@dataclass(frozen=True)
class Flux2Result:
    png_bytes: bytes
    width: int
    height: int
    steps: int
    seed: int
    model: str
    revision: str
    backend: str
    elapsed_s: float
    denoise_ms: float


def pack_message(message):
    return msgpack.packb(message, use_bin_type=True)


def unpack_message(data):
    return msgpack.unpackb(data, raw=False, strict_map_key=False)


def parse_size(size_str, default_width, default_height, align=16,
               min_dim=256, max_dim=2048):
    """Parse a WxH string and validate alignment and bounds."""
    if size_str is None:
        width, height = default_width, default_height
    else:
        if not isinstance(size_str, str) or "x" not in size_str.lower():
            raise ValueError("image.size must be formatted as WIDTHxHEIGHT")
        parts = size_str.lower().split("x")
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            raise ValueError("image.size must contain positive integers WIDTHxHEIGHT")
        width, height = int(parts[0]), int(parts[1])
    if width < min_dim or height < min_dim or width > max_dim or height > max_dim:
        raise ValueError(
            f"image.size {width}x{height} is outside [{min_dim}, {max_dim}]")
    if align > 1 and (width % align != 0 or height % align != 0):
        raise ValueError(
            f"image.size {width}x{height} must be divisible by {align}")
    return width, height


def is_valid_png(data):
    """Structural check for a PNG signature and IHDR chunk."""
    return (isinstance(data, (bytes, bytearray)) and len(data) >= 33
            and bytes(data[:8]) == PNG_SIGNATURE and bytes(data[12:16]) == b"IHDR")


def decode_data_url(value, max_bytes=MAX_INPUT_IMAGE_BYTES):
    """Decode a base64 `data:` URL into raw image bytes."""
    if not isinstance(value, str) or not value.startswith("data:"):
        raise ValueError(
            "image must be a base64 data: URL; http(s) and local paths are rejected")
    comma = value.find(",")
    if comma < 0:
        raise ValueError("data URL has no payload separator")
    header = value[5:comma]
    if ";base64" not in header:
        raise ValueError("data URL must carry a base64 payload")
    payload = value[comma + 1:]
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ValueError(f"data URL payload is not valid base64: {exc}") from exc
    if not raw:
        raise ValueError("data URL payload is empty")
    if len(raw) > max_bytes:
        raise ValueError(
            f"decoded image ({len(raw)} bytes) exceeds {max_bytes} byte limit")
    if not (raw.startswith(PNG_SIGNATURE) or raw.startswith(b"\xff\xd8\xff")):
        raise ValueError("decoded image is not PNG or JPEG")
    return raw


def decode_image_input(value, max_bytes=MAX_INPUT_IMAGE_BYTES):
    """Decode a legacy data URL or validate exact internal binary image bytes."""
    if isinstance(value, str):
        return decode_data_url(value, max_bytes=max_bytes)
    if not isinstance(value, (bytes, bytearray)):
        raise ValueError("image must be a base64 data URL or MessagePack binary")
    raw = bytes(value)
    if not raw:
        raise ValueError("binary image payload is empty")
    if len(raw) > max_bytes:
        raise ValueError(
            f"decoded image ({len(raw)} bytes) exceeds {max_bytes} byte limit")
    if not (raw.startswith(PNG_SIGNATURE) or raw.startswith(b"\xff\xd8\xff")):
        raise ValueError("decoded image is not PNG or JPEG")
    return raw


def encode_png(width, height, rgb, level=6):
    """Encode raw RGB bytes as a PNG without any third-party dependency."""
    stride = width * 3
    if len(rgb) != stride * height:
        raise ValueError(
            f"expected {stride * height} RGB bytes, got {len(rgb)}")

    def chunk(tag, payload):
        body = tag + payload
        return (struct.pack(">I", len(payload)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    raw = bytearray()
    for row in range(height):
        raw.append(0)  # filter type 0 (None)
        raw += rgb[row * stride:(row + 1) * stride]
    return (PNG_SIGNATURE
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), level))
            + chunk(b"IEND", b""))


def png_dimensions(data):
    """Read (width, height, bit_depth, color_type) from a PNG IHDR chunk."""
    if not is_valid_png(data):
        raise ValueError("not a PNG")
    width, height, depth, color = struct.unpack(">IIBB", bytes(data[16:26]))
    return width, height, depth, color


def detect_model_revision(model_path, explicit_revision=None):
    """Return explicit revision, git HEAD, HF snapshot SHA, or UNKNOWN."""
    if explicit_revision:
        return explicit_revision
    path = Path(model_path)
    if path.exists():
        if path.parent.name == "snapshots" and len(path.name) == 40:
            try:
                int(path.name, 16)
                return path.name
            except ValueError:
                pass
    return "UNKNOWN"


class MockFlux2Backend:
    """Deterministic offline backend: a valid PNG, no torch, no GPU."""

    model_id = "flux2-klein-4b-mock"
    revision = "UNKNOWN"
    backend_name = "mock"
    align = 16

    def __init__(self):
        self.model_id = "flux2-klein-4b-mock"

    def load(self):
        return

    def generate(self, req: Flux2Request, abort_flag: threading.Event):
        started = time.perf_counter()
        seed = req.seed if req.seed is not None else random.randrange(2**31)
        width, height = req.width, req.height
        # A cheap gradient that depends on the prompt, seed and edit input so
        # determinism and image-to-image difference checks are meaningful.
        digest = zlib.crc32(req.prompt.encode("utf-8")) ^ (seed & 0xFFFFFFFF)
        if req.image_bytes:
            digest ^= zlib.crc32(req.image_bytes)
        rows = []
        for y in range(height):
            if abort_flag.is_set():
                raise RequestAborted("aborted during mock generation")
            base = (y * 255) // max(1, height - 1)
            row = bytearray()
            for x in range(width):
                row += bytes((
                    (base + (digest & 0xFF)) & 0xFF,
                    (base + ((digest >> 8) & 0xFF)) & 0xFF,
                    (base + (x * 255 // max(1, width - 1))) & 0xFF,
                ))
            rows.append(bytes(row))
        png = encode_png(width, height, b"".join(rows))
        return Flux2Result(
            png_bytes=png, width=width, height=height, steps=req.steps,
            seed=seed, model=self.model_id, revision=self.revision,
            backend=self.backend_name,
            elapsed_s=time.perf_counter() - started, denoise_ms=0.0)


class DiffusersFlux2Backend:
    """Real local Diffusers FLUX.2-klein backend (text-to-image and edit)."""

    backend_name = "diffusers"

    def __init__(self, model_path, gpu=0, model_id=None, revision=None,
                 align=16):
        self.model_path = str(model_path)
        self.gpu = int(gpu)
        self.model_id = model_id or Path(self.model_path).name
        self.revision = detect_model_revision(self.model_path, revision)
        self.align = int(align)
        self.pipe = None
        self.device = None
        self._accepts_guidance = True

    def load(self):
        if self.pipe is not None:
            return
        import torch
        import diffusers

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for diffusers flux2 backend")
        if self.gpu < 0 or self.gpu >= torch.cuda.device_count():
            raise RuntimeError(
                f"requested --gpu {self.gpu} out of range "
                f"(device_count={torch.cuda.device_count()})")
        self.device = torch.device(f"cuda:{self.gpu}")
        torch.cuda.set_device(self.device)

        cls = getattr(diffusers, "Flux2KleinPipeline", None)
        if cls is None:
            cls = getattr(diffusers, "Flux2Pipeline", None)
        if cls is not None:
            self.pipe = cls.from_pretrained(
                self.model_path, torch_dtype=torch.bfloat16)
        else:
            self.pipe = diffusers.DiffusionPipeline.from_pretrained(
                self.model_path, torch_dtype=torch.bfloat16)

        try:
            import inspect
            params = inspect.signature(self.pipe.__call__).parameters
            self._accepts_guidance = "guidance_scale" in params
        except (TypeError, ValueError):
            self._accepts_guidance = True

        self.pipe.to(self.device)
        if hasattr(self.pipe, "set_progress_bar_config"):
            self.pipe.set_progress_bar_config(disable=True)

    def _prepare_image(self, raw: bytes):
        from PIL import Image

        image = Image.open(io.BytesIO(raw)).convert("RGB")
        width = max(self.align, (image.width // self.align) * self.align)
        height = max(self.align, (image.height // self.align) * self.align)
        if (width, height) != image.size:
            image = image.resize((width, height), Image.LANCZOS)
        return image, width, height

    def generate(self, req: Flux2Request, abort_flag: threading.Event):
        import torch

        if abort_flag.is_set():
            raise RequestAborted("aborted before diffusion started")

        started = time.perf_counter()
        seed = req.seed if req.seed is not None else random.randrange(2**31)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        state = {"last_step": 0}

        def on_step_end(pipe, step_index, timestep, callback_kwargs):
            state["last_step"] = int(step_index) + 1
            if abort_flag.is_set():
                if hasattr(pipe, "_interrupt"):
                    pipe._interrupt = True
                raise RequestAborted(
                    f"aborted at denoising step {step_index + 1}/{req.steps}")
            return callback_kwargs

        width, height = req.width, req.height
        kwargs = {
            "prompt": req.prompt,
            "num_inference_steps": req.steps,
            "generator": generator,
            "callback_on_step_end": on_step_end,
        }
        if self._accepts_guidance:
            kwargs["guidance_scale"] = float(req.guidance_scale)
        if req.image_bytes is not None:
            image, width, height = self._prepare_image(req.image_bytes)
            kwargs["image"] = image
        else:
            kwargs["width"] = width
            kwargs["height"] = height

        denoise_started = time.perf_counter()
        try:
            with torch.inference_mode():
                output = self.pipe(**kwargs)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        denoise_ms = (time.perf_counter() - denoise_started) * 1000.0
        if abort_flag.is_set():
            raise RequestAborted("aborted after diffusion completed")

        images = getattr(output, "images", None)
        if not images:
            raise RuntimeError("flux2 pipeline returned no images")
        image = images[0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        png = buffer.getvalue()
        actual_w, actual_h = image.size
        if not is_valid_png(png):
            raise RuntimeError("flux2 pipeline produced invalid PNG bytes")
        if len(png) > MAX_IMAGE_ARTIFACT_BYTES:
            raise RuntimeError(
                f"encoded PNG ({len(png)} bytes) exceeds "
                f"{MAX_IMAGE_ARTIFACT_BYTES} byte limit")
        return Flux2Result(
            png_bytes=png, width=int(actual_w), height=int(actual_h),
            steps=req.steps, seed=int(seed), model=self.model_id,
            revision=self.revision, backend=self.backend_name,
            elapsed_s=time.perf_counter() - started, denoise_ms=denoise_ms)


class _Finished:
    __slots__ = ("request_id",)

    def __init__(self, request_id):
        self.request_id = request_id


class Flux2Stage:
    """Persistent single-request image stage speaking ummInfer ZMQ/msgpack."""

    def __init__(self, name, recv_endpoint, coordinator_endpoint, abort_endpoint,
                 backend, default_width=1024, default_height=1024,
                 default_steps=4, max_steps=50, default_guidance=4.0,
                 perf_log=None):
        self.name = name
        self.recv_endpoint = recv_endpoint
        self.coordinator_endpoint = coordinator_endpoint
        self.abort_endpoint = abort_endpoint
        self.backend = backend
        self.default_width = int(default_width)
        self.default_height = int(default_height)
        self.default_steps = int(default_steps)
        self.max_steps = int(max_steps)
        self.default_guidance = float(default_guidance)
        self.perf_log = perf_log
        self.perf_sink = None
        self.perf_lock = threading.Lock()

        self.context = None
        self.recv_socket = None
        self.coordinator_socket = None
        self.abort_socket = None
        self.poller = None
        self.pending = deque()
        self.aborted = {}
        self.outbox = queue.Queue()
        self.abort_flags = {}
        self.inflight = set()
        self.local = threading.local()
        self.running = True

    def write_perf(self, record):
        if self.perf_log is None:
            return
        try:
            with self.perf_lock:
                if self.perf_sink is None:
                    Path(self.perf_log).parent.mkdir(parents=True, exist_ok=True)
                    self.perf_sink = open(self.perf_log, "a", encoding="utf-8")
                self.perf_sink.write(json.dumps(record, sort_keys=True) + "\n")
                self.perf_sink.flush()
        except OSError as exc:
            print(f"image stage perf log write failed: {exc}", file=sys.stderr)

    def start(self):
        started = time.perf_counter()
        self.backend.load()
        self.write_perf({
            "event": "load",
            "stage": self.name,
            "model": getattr(self.backend, "model_id", "UNKNOWN"),
            "revision": getattr(self.backend, "revision", "UNKNOWN"),
            "seconds": round(time.perf_counter() - started, 3),
        })
        if not self.running:
            return
        self.context = zmq.Context()
        self.recv_socket = self.context.socket(zmq.PULL)
        self.recv_socket.RCVHWM = 128
        self.recv_socket.LINGER = 0
        self.recv_socket.bind(self.recv_endpoint)
        self.coordinator_socket = self.context.socket(zmq.PUSH)
        self.coordinator_socket.SNDHWM = 128
        self.coordinator_socket.SNDTIMEO = 5000
        self.coordinator_socket.LINGER = 0
        self.coordinator_socket.connect(self.coordinator_endpoint)
        self.abort_socket = self.context.socket(zmq.SUB)
        self.abort_socket.RCVHWM = 128
        self.abort_socket.LINGER = 0
        self.abort_socket.connect(self.abort_endpoint)
        self.abort_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.poller = zmq.Poller()
        self.poller.register(self.recv_socket, zmq.POLLIN)
        self.poller.register(self.abort_socket, zmq.POLLIN)
        self.announce_ready()
        print(
            f"ready kind=image name={self.name} "
            f"model={getattr(self.backend, 'model_id', 'UNKNOWN')} "
            f"revision={getattr(self.backend, 'revision', 'UNKNOWN')}",
            flush=True,
        )

    def close(self):
        self.running = False
        for flag in list(self.abort_flags.values()):
            flag.set()
        sink = self.perf_sink
        self.perf_sink = None
        if sink is not None:
            sink.close()
        sockets = (self.recv_socket, self.coordinator_socket, self.abort_socket)
        self.recv_socket = self.coordinator_socket = self.abort_socket = None
        context = self.context
        self.context = None
        for sock in sockets:
            if sock is not None:
                sock.close(linger=0)
        if context is not None:
            context.term()

    def emit(self, message):
        if getattr(self.local, "pool_thread", False):
            self.outbox.put(message)
            return
        self.coordinator_socket.send(pack_message(message))

    def announce_ready(self):
        """Re-register with the coordinator without ever blocking the loop.

        A dead or restarting gateway fills the PUSH queue; NOBLOCK drops this
        frame and the next heartbeat retries.
        """
        if self.coordinator_socket is None:
            return
        try:
            self.coordinator_socket.send(
                pack_message({"type": "stage_ready", "from_stage": self.name}),
                zmq.NOBLOCK)
        except zmq.Again:
            pass

    def expire_aborts(self):
        now = time.monotonic()
        for request_id in [k for k, deadline in self.aborted.items()
                           if deadline <= now]:
            del self.aborted[request_id]

    def record_abort(self, request_id):
        self.expire_aborts()
        flag = self.abort_flags.get(request_id)
        if flag is not None:
            flag.set()
        removed = any(m.get("request_id") == request_id for m in self.pending)
        self.pending = deque(m for m in self.pending
                             if m.get("request_id") != request_id)
        self.aborted[request_id] = time.monotonic() + ABORT_TOMBSTONE_TTL_S
        self.write_perf({
            "event": "abort",
            "request_id": request_id,
            "removed_pending": removed,
            "in_flight": flag is not None,
        })

    def consume_abort(self, request_id):
        if getattr(self.local, "pool_thread", False):
            flag = getattr(self.local, "abort_flag", None)
            return flag is not None and flag.is_set()
        self.expire_aborts()
        return self.aborted.pop(request_id, None) is not None

    def parse_request(self, message) -> Flux2Request:
        data = message.get("data")
        if not isinstance(data, dict):
            raise ValueError("image submit missing data map")
        request = data.get("request")
        if not isinstance(request, dict):
            raise ValueError("image submit missing request map")
        inputs = request.get("inputs")
        if not isinstance(inputs, dict):
            raise ValueError("image submit missing inputs map")
        prompt = inputs.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("image.prompt must be a non-empty string")

        output_format = inputs.get("output_format", "png")
        if not isinstance(output_format, str) or output_format != "png":
            raise ValueError("image.output_format must be png")

        align = getattr(self.backend, "align", 16)
        width, height = parse_size(
            inputs.get("size"), self.default_width, self.default_height,
            align=align)

        steps = inputs.get("steps", inputs.get("num_inference_steps",
                                              self.default_steps))
        if isinstance(steps, bool) or not isinstance(steps, int) \
                or not (1 <= steps <= self.max_steps):
            raise ValueError(f"image.steps must be an integer in [1, {self.max_steps}]")

        guidance = inputs.get("guidance_scale", inputs.get("guidance",
                                                          self.default_guidance))
        if isinstance(guidance, bool) or not isinstance(guidance, (int, float)) \
                or not math.isfinite(guidance) or guidance < 0:
            raise ValueError("image.guidance_scale must be a finite non-negative number")

        seed = inputs.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("image.seed must be an integer")
        if seed is not None and seed < 0:
            raise ValueError("image.seed must be a non-negative integer")

        image_bytes = None
        if inputs.get("image") is not None:
            image_bytes = decode_image_input(inputs["image"])

        if "anchor_crop" in inputs:
            raise ValueError("anchor_crop is only valid for native video I2V")
        explicit = inputs.get("anchor_explicit", False)
        if not isinstance(explicit, bool):
            raise ValueError("image.anchor_explicit must be boolean")
        has_anchor_metadata = any(
            key in inputs for key in ("anchor_slot_index", "anchor_sha256"))
        if has_anchor_metadata and not explicit:
            raise ValueError("image anchor metadata requires anchor_explicit=true")
        if explicit:
            if image_bytes is None:
                raise ValueError("explicit image reference requires image bytes")
            anchor_index = inputs.get("anchor_slot_index")
            if isinstance(anchor_index, bool) or not isinstance(anchor_index, int) \
                    or anchor_index < 0:
                raise ValueError("image.anchor_slot_index must be a non-negative integer")
            declared = inputs.get("anchor_sha256")
            actual = hashlib.sha256(image_bytes).hexdigest()
            if not isinstance(declared, str) or declared != actual:
                raise ValueError("image does not match anchor_sha256")

        return Flux2Request(
            prompt=prompt.strip(), width=width, height=height, steps=steps,
            guidance_scale=float(guidance), seed=seed, image_bytes=image_bytes)

    def complete_artifact(self, request_id, res: Flux2Result):
        return {
            "type": "complete",
            "request_id": request_id,
            "from_stage": self.name,
            "success": True,
            "result": {
                "media_type": "image",
                "mime_type": "image/png",
                "format": "png",
                "size": f"{res.width}x{res.height}",
                "width": res.width,
                "height": res.height,
                "seed": res.seed,
                "steps": res.steps,
                "denoise_ms": round(res.denoise_ms, 3),
                "model": res.model,
                "revision": res.revision,
                "backend": res.backend,
                "data": res.png_bytes,
            },
            "error": None,
        }

    def complete_error(self, request_id, error):
        return {
            "type": "complete",
            "request_id": request_id,
            "from_stage": self.name,
            "success": False,
            "result": {},
            "error": str(error),
        }

    def process(self, message, abort_flag: threading.Event):
        request_id = message.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            print("image stage discarded submit without a request id",
                  file=sys.stderr)
            return
        if abort_flag.is_set() or not self.running:
            return
        try:
            req = self.parse_request(message)
            self.write_perf({
                "event": "request_start",
                "request_id": request_id,
                "width": req.width,
                "height": req.height,
                "steps": req.steps,
                "seed": req.seed,
                "edit": req.image_bytes is not None,
            })
            res = self.backend.generate(req, abort_flag)
            if abort_flag.is_set() or not self.running:
                return
            self.write_perf({
                "event": "request",
                "request_id": request_id,
                "bytes": len(res.png_bytes),
                "width": res.width,
                "height": res.height,
                "steps": res.steps,
                "seed": res.seed,
                "denoise_ms": round(res.denoise_ms, 3),
                "elapsed_s": round(res.elapsed_s, 3),
                "model": res.model,
                "revision": res.revision,
                "backend": res.backend,
            })
            print(
                f"image request request_id={request_id} status=ok "
                f"bytes={len(res.png_bytes)} size={res.width}x{res.height} "
                f"steps={res.steps} seed={res.seed} "
                f"elapsed_s={res.elapsed_s:.2f}",
                file=sys.stderr, flush=True,
            )
            self.emit(self.complete_artifact(request_id, res))
        except RequestAborted as exc:
            self.write_perf({
                "event": "aborted",
                "request_id": request_id,
                "reason": str(exc),
            })
            print(
                f"image request request_id={request_id} status=aborted reason={exc}",
                file=sys.stderr, flush=True,
            )
            return
        except Exception as exc:
            if abort_flag.is_set() or not self.running:
                return
            error = str(exc) or type(exc).__name__
            self.write_perf({
                "event": "error",
                "request_id": request_id,
                "error": error,
            })
            print(
                f"image request request_id={request_id} status=error error={error}",
                file=sys.stderr, flush=True,
            )
            self.emit(self.complete_error(request_id, error))

    def poll_events(self, timeout_ms=0):
        if self.poller is None:
            return
        for index in range(64):
            events = dict(self.poller.poll(timeout_ms if index == 0 else 0))
            if not events:
                return
            if self.abort_socket in events:
                try:
                    msg = unpack_message(self.abort_socket.recv())
                    req_id = msg.get("request_id")
                    if msg.get("type") != "abort" or not isinstance(req_id, str):
                        raise ValueError("invalid abort message")
                    self.record_abort(req_id)
                except Exception as exc:
                    print(f"image stage discarded malformed abort: {exc}",
                          file=sys.stderr)
            if self.recv_socket not in events:
                continue
            if len(self.pending) >= 128:
                return
            try:
                msg = unpack_message(self.recv_socket.recv())
                if not isinstance(msg, dict):
                    raise ValueError("control message must be a map")
                mtype = msg.get("type")
                if mtype == "shutdown":
                    self.running = False
                    for flag in list(self.abort_flags.values()):
                        flag.set()
                elif mtype == "abort":
                    req_id = msg.get("request_id")
                    if not isinstance(req_id, str):
                        raise ValueError("invalid abort message")
                    self.record_abort(req_id)
                elif mtype == "submit":
                    self.pending.append(msg)
                else:
                    print(f"image stage ignored message type {mtype!r}",
                          file=sys.stderr)
            except Exception as exc:
                print(f"image stage discarded malformed control message: {exc}",
                      file=sys.stderr)

    def dispatch(self, pool):
        while self.pending and len(self.inflight) < 1:
            message = self.pending.popleft()
            request_id = message.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                continue
            if self.consume_abort(request_id):
                continue
            flag = threading.Event()
            self.abort_flags[request_id] = flag
            self.inflight.add(request_id)
            pool.submit(self._run_worker, message, flag)

    def _run_worker(self, message, flag):
        self.local.pool_thread = True
        self.local.abort_flag = flag
        try:
            self.process(message, flag)
        finally:
            self.local.pool_thread = False
            self.local.abort_flag = None
            self.outbox.put(_Finished(message.get("request_id")))

    def drain_outbox(self):
        while True:
            try:
                item = self.outbox.get_nowait()
            except queue.Empty:
                return
            if isinstance(item, _Finished):
                self.inflight.discard(item.request_id)
                self.abort_flags.pop(item.request_id, None)
                self.aborted.pop(item.request_id, None)
            else:
                self.emit(item)

    def run(self):
        pool = ThreadPoolExecutor(max_workers=1,
                                  thread_name_prefix=f"{self.name}-worker")
        next_ready = time.monotonic() + READY_HEARTBEAT_S
        try:
            while self.running:
                self.poll_events(5 if self.inflight else 100)
                self.dispatch(pool)
                self.drain_outbox()
                # Re-announce readiness so a restarted gateway re-learns this
                # stage without the supervisor killing it.
                now = time.monotonic()
                if now >= next_ready:
                    next_ready = now + READY_HEARTBEAT_S
                    self.announce_ready()
        finally:
            self.running = False
            for flag in list(self.abort_flags.values()):
                flag.set()
            pool.shutdown(wait=True)
            self.drain_outbox()
            self.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="image")
    parser.add_argument("--recv", "--recv-endpoint", dest="recv", required=True)
    parser.add_argument("--coordinator", "--coordinator-endpoint",
                        dest="coordinator", required=True)
    parser.add_argument("--abort", "--abort-endpoint", dest="abort", required=True)
    parser.add_argument("--backend", choices=("diffusers", "mock"), default="diffusers")
    parser.add_argument("--model", default="", help="Local Diffusers model directory")
    parser.add_argument("--model-id", default="",
                        help="Human-readable model identifier for metadata")
    parser.add_argument("--revision", default="",
                        help="Pinned model revision SHA (defaults to auto-detect or UNKNOWN)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--default-size", default="1024x1024")
    parser.add_argument("--default-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--default-guidance", type=float, default=4.0)
    parser.add_argument("--perf-log", default=None)
    args = parser.parse_args(argv)

    if args.backend == "diffusers":
        if not args.model:
            parser.error("--model is required for --backend diffusers")
        backend = DiffusersFlux2Backend(
            model_path=args.model, gpu=args.gpu,
            model_id=args.model_id or None, revision=args.revision or None)
    else:
        backend = MockFlux2Backend()
        if args.model_id:
            backend.model_id = args.model_id

    align = getattr(backend, "align", 16)
    default_width, default_height = parse_size(
        args.default_size, 1024, 1024, align=align)

    stage = Flux2Stage(
        name=args.name,
        recv_endpoint=args.recv,
        coordinator_endpoint=args.coordinator,
        abort_endpoint=args.abort,
        backend=backend,
        default_width=default_width,
        default_height=default_height,
        default_steps=args.default_steps,
        max_steps=args.max_steps,
        default_guidance=args.default_guidance,
        perf_log=args.perf_log,
    )

    def stop(signum, frame):
        stage.running = False
        for flag in list(stage.abort_flags.values()):
            flag.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        stage.start()
        stage.run()
    finally:
        stage.close()


if __name__ == "__main__":
    main()
