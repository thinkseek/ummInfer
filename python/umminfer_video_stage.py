#!/usr/bin/env python3
"""Replaceable real video generation stage for the ummInfer UnifiedEngine.

Speaks the native ZeroMQ + MessagePack stage protocol (`stage_ready`, `submit`,
`abort`, `shutdown`, `complete`) so any local Diffusers video checkpoint (for
example Wan2.2-TI2V-5B-Diffusers or Wan2.1-T2V-1.3B-Diffusers) or an authorized
OpenAI-compatible `/v1/videos` upstream plugs into `umm-infer serve
--video-entry video --video-entry-endpoint ipc://...` without changing the
C++ control plane.
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
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

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

# Mirror UnifiedEngine kMaxArtifactVideoBytes (48 MiB) so oversized encodings
# fail fast inside the stage with a clear error message.
MAX_VIDEO_ARTIFACT_BYTES = 48 * 1024 * 1024

# Periodic stage_ready re-registration. A restarted gateway rebuilds its ready
# set from these frames, so a server restart never requires a stage restart.
READY_HEARTBEAT_S = 1.0


class RequestAborted(Exception):
    """Raised inside a backend when the stage receives a coordinator abort."""


@dataclass(frozen=True)
class VideoRequest:
    prompt: str
    width: int
    height: int
    seconds: int
    fps: int
    num_frames: int
    steps: int
    guidance_scale: float
    seed: int | None
    negative_prompt: str
    output_format: str
    init_image_bytes: bytes | None = None

@dataclass(frozen=True)
class VideoGenerationResult:
    mp4_bytes: bytes
    width: int
    height: int
    fps: int
    frames: int
    seconds: float
    model: str
    revision: str
    elapsed_s: float


def pack_message(message):
    return msgpack.packb(message, use_bin_type=True)


def unpack_message(data):
    return msgpack.unpackb(data, raw=False, strict_map_key=False)


def parse_size(size_str, default_width, default_height, align=32,
               min_dim=128, max_dim=1280):
    """Parse a WxH string and validate alignment and bounds."""
    if size_str is None:
        width, height = default_width, default_height
    else:
        if not isinstance(size_str, str) or "x" not in size_str.lower():
            raise ValueError("video.size must be formatted as WIDTHxHEIGHT")
        parts = size_str.lower().split("x")
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            raise ValueError("video.size must contain positive integers WIDTHxHEIGHT")
        width, height = int(parts[0]), int(parts[1])
    if width < min_dim or height < min_dim or width > max_dim or height > max_dim:
        raise ValueError(
            f"video.size {width}x{height} is outside [{min_dim}, {max_dim}]")
    if align > 1 and (width % align != 0 or height % align != 0):
        raise ValueError(
            f"video.size {width}x{height} must be divisible by {align}")
    return width, height


def align_frame_count(seconds, fps, max_frames, temporal_stride=4):
    """Compute a valid Wan/DiT frame count (k * temporal_stride + 1)."""
    if seconds < 1:
        raise ValueError("video.seconds must be a positive integer")
    if fps < 1 or fps > 60:
        raise ValueError("video.fps must be between 1 and 60")
    raw = int(seconds) * int(fps)
    target = max(temporal_stride + 1, min(raw, max_frames))
    if temporal_stride > 1:
        steps = max(1, (target - 1) // temporal_stride)
        target = steps * temporal_stride + 1
        while target > max_frames and steps > 1:
            steps -= 1
            target = steps * temporal_stride + 1
    return target


def is_valid_mp4_header(data):
    """Quick structural check for an ISO BMFF ftyp box before returning bytes."""
    return isinstance(data, (bytes, bytearray)) and len(data) >= 12 and data[4:8] == b"ftyp"


def detect_model_revision(model_path, explicit_revision=None):
    """Return explicit revision, git HEAD, HF snapshot SHA, or UNKNOWN."""
    if explicit_revision:
        return explicit_revision
    path = Path(model_path)
    if path.exists():
        # HF cache layout: .../snapshots/<40-hex-sha>
        if path.parent.name == "snapshots" and len(path.name) == 40:
            try:
                int(path.name, 16)
                return path.name
            except ValueError:
                pass
        git_dir = path / ".git"
        if git_dir.exists():
            try:
                out = subprocess.check_output(
                    ["git", "-C", str(path), "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL, text=True, timeout=3).strip()
                if out:
                    return out
            except Exception:
                pass
    return "UNKNOWN"


def encode_frames_mp4(frames, fps, abort_flag=None, ffmpeg_bin="ffmpeg"):
    """Encode a sequence of uint8 RGB frames into a faststart H.264 MP4."""
    import numpy as np

    if len(frames) == 0:
        raise ValueError("video pipeline produced zero frames")
    first = np.asarray(frames[0])
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError(f"expected RGB frames (H, W, 3), got shape {first.shape}")
    height, width = int(first.shape[0]), int(first.shape[1])

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        out_path = Path(tmp.name)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-f", "mp4",
        str(out_path),
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            if abort_flag is not None and abort_flag.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=3)
                raise RequestAborted("aborted during MP4 encoding")
            arr = np.asarray(frame)
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            proc.stdin.write(arr.tobytes())
        proc.stdin.close()
        proc.stdin = None
        while proc.poll() is None:
            if abort_flag is not None and abort_flag.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=3)
                raise RequestAborted("aborted while finalizing MP4")
            time.sleep(0.02)
        stderr = proc.stderr.read().decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}): {stderr}")
        mp4_bytes = out_path.read_bytes()
        if not is_valid_mp4_header(mp4_bytes):
            raise RuntimeError("ffmpeg produced invalid MP4 output (missing ftyp box)")
        return mp4_bytes, width, height
    finally:
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except Exception:
                pass
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=3)
        try:
            out_path.unlink(missing_ok=True)
        except OSError:
            pass


class DiffusersVideoBackend:
    """Real local Diffusers video backend (Wan2.2 / Wan2.1 / DiffusionPipeline)."""

    def __init__(self, model_path, gpu=0, model_id=None, revision=None,
                 ffmpeg_bin="ffmpeg"):
        self.model_path = str(model_path)
        self.gpu = int(gpu)
        self.model_id = model_id or Path(self.model_path).name
        self.revision = detect_model_revision(self.model_path, revision)
        self.ffmpeg_bin = ffmpeg_bin
        self.pipe = None
        self.device = None
        self.spatial_align = 32
        self.temporal_stride = 4

    def load(self):
        if self.pipe is not None:
            return
        import torch
        import diffusers

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for diffusers video backend")
        if self.gpu < 0 or self.gpu >= torch.cuda.device_count():
            raise RuntimeError(
                f"requested --gpu {self.gpu} out of range (device_count={torch.cuda.device_count()})")
        self.device = torch.device(f"cuda:{self.gpu}")
        torch.cuda.set_device(self.device)

        model_index = Path(self.model_path) / "model_index.json"
        class_name = None
        if model_index.is_file():
            meta = json.loads(model_index.read_text(encoding="utf-8"))
            class_name = meta.get("_class_name")

        if class_name == "WanPipeline" and hasattr(diffusers, "WanPipeline"):
            vae = diffusers.AutoencoderKLWan.from_pretrained(
                self.model_path, subfolder="vae", torch_dtype=torch.float32)
            self.pipe = diffusers.WanPipeline.from_pretrained(
                self.model_path, vae=vae, torch_dtype=torch.bfloat16)
            scale_spatial = getattr(self.pipe, "vae_scale_factor_spatial", 8)
            patch = 1
            if hasattr(self.pipe, "transformer") and hasattr(self.pipe.transformer, "config"):
                patch_size = getattr(self.pipe.transformer.config, "patch_size", (1, 2, 2))
                if isinstance(patch_size, (list, tuple)) and len(patch_size) >= 2:
                    patch = int(patch_size[-1])
            self.spatial_align = max(16, int(scale_spatial) * patch)
            self.temporal_stride = int(getattr(self.pipe, "vae_scale_factor_temporal", 4))
        else:
            self.pipe = diffusers.DiffusionPipeline.from_pretrained(
                self.model_path, torch_dtype=torch.bfloat16)
            self.spatial_align = 16
            self.temporal_stride = 4

        self.pipe.to(self.device)
        if hasattr(self.pipe, "set_progress_bar_config"):
            self.pipe.set_progress_bar_config(disable=True)

    def generate(self, req: VideoRequest, abort_flag: threading.Event) -> VideoGenerationResult:
        import numpy as np
        import torch

        if abort_flag.is_set():
            raise RequestAborted("aborted before diffusion started")

        started = time.perf_counter()
        generator = None
        if req.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(req.seed))

        def on_step_end(pipe, step_index, timestep, callback_kwargs):
            if abort_flag.is_set():
                pipe._interrupt = True
                raise RequestAborted(
                    f"aborted at denoising step {step_index + 1}/{req.steps}")
            return callback_kwargs

        kwargs = {
            "prompt": req.prompt,
            "height": req.height,
            "width": req.width,
            "num_frames": req.num_frames,
            "num_inference_steps": req.steps,
            "guidance_scale": req.guidance_scale,
            "callback_on_step_end": on_step_end,
            "output_type": "np",
        }
        if req.init_image_bytes is not None:
            from PIL import Image
            with Image.open(io.BytesIO(req.init_image_bytes)) as source_image:
                kwargs["image"] = source_image.convert("RGB")
        if req.negative_prompt:
            kwargs["negative_prompt"] = req.negative_prompt
        if generator is not None:
            kwargs["generator"] = generator

        try:
            with torch.inference_mode():
                output = self.pipe(**kwargs)
            if abort_flag.is_set():
                raise RequestAborted("aborted after diffusion completed")
            frames = output.frames[0]
            if isinstance(frames, np.ndarray) and frames.ndim == 4:
                frame_list = [frames[i] for i in range(frames.shape[0])]
            else:
                frame_list = list(frames)
            mp4_bytes, width, height = encode_frames_mp4(
                frame_list, req.fps, abort_flag=abort_flag,
                ffmpeg_bin=self.ffmpeg_bin)
            if len(mp4_bytes) > MAX_VIDEO_ARTIFACT_BYTES:
                raise RuntimeError(
                    f"encoded MP4 ({len(mp4_bytes)} bytes) exceeds {MAX_VIDEO_ARTIFACT_BYTES} byte limit")
            elapsed_s = time.perf_counter() - started
            actual_frames = len(frame_list)
            return VideoGenerationResult(
                mp4_bytes=mp4_bytes,
                width=width,
                height=height,
                fps=req.fps,
                frames=actual_frames,
                seconds=round(actual_frames / float(req.fps), 3),
                model=self.model_id,
                revision=self.revision,
                elapsed_s=elapsed_s,
            )
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class UpstreamVideoBackend:
    """Replaceable external OpenAI-compatible `/v1/videos` upstream adapter."""

    def __init__(self, base_url, model, api_key="", revision=None,
                 poll_interval_s=1.0, timeout_s=540.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("UMMINFER_UPSTREAM_API_KEY", "")
        self.revision = revision or "UNKNOWN"
        self.poll_interval_s = float(poll_interval_s)
        self.timeout_s = float(timeout_s)
        self.spatial_align = 16
        self.temporal_stride = 1

    def load(self):
        if not self.base_url:
            raise ValueError("upstream backend requires --upstream-base-url")
        if not self.model:
            raise ValueError("upstream backend requires --upstream-model")

    def _request(self, method, path, body=None, timeout=30.0):
        url = f"{self.base_url}{path}"
        headers = {}
        data = None
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()

    def _delete_best_effort(self, job_id):
        try:
            self._request("DELETE", f"/v1/videos/{urllib.parse.quote(job_id)}", timeout=10.0)
        except Exception:
            pass

    def generate(self, req: VideoRequest, abort_flag: threading.Event) -> VideoGenerationResult:
        if req.init_image_bytes is not None:
            raise ValueError("upstream video backend does not support I2V init images")
        if abort_flag.is_set():
            raise RequestAborted("aborted before upstream submission")
        started = time.perf_counter()
        create_body = {
            "model": self.model,
            "prompt": req.prompt,
            "size": f"{req.width}x{req.height}",
            "seconds": req.seconds,
            "fps": req.fps,
        }
        if req.seed is not None:
            create_body["seed"] = req.seed
        _, raw = self._request("POST", "/v1/videos", body=create_body, timeout=60.0)
        job = json.loads(raw.decode("utf-8"))
        job_id = job.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError("upstream /v1/videos returned no job id")

        deadline = time.monotonic() + self.timeout_s
        while True:
            if abort_flag.is_set():
                self._delete_best_effort(job_id)
                raise RequestAborted(f"aborted upstream video job {job_id}")
            if time.monotonic() >= deadline:
                self._delete_best_effort(job_id)
                raise RuntimeError(f"upstream video job {job_id} timed out")
            status = str(job.get("status", "")).lower()
            if status in ("completed", "succeeded"):
                break
            if status in ("failed", "cancelled", "canceled", "expired"):
                err = job.get("error") or status
                raise RuntimeError(f"upstream video job {job_id} {status}: {err}")
            time.sleep(min(self.poll_interval_s, 0.25))
            _, raw = self._request("GET", f"/v1/videos/{urllib.parse.quote(job_id)}", timeout=30.0)
            job = json.loads(raw.decode("utf-8"))

        if abort_flag.is_set():
            self._delete_best_effort(job_id)
            raise RequestAborted(f"aborted before downloading upstream video {job_id}")
        _, mp4_bytes = self._request(
            "GET", f"/v1/videos/{urllib.parse.quote(job_id)}/content", timeout=120.0)
        if not is_valid_mp4_header(mp4_bytes):
            raise RuntimeError("upstream video content is not a valid MP4 (missing ftyp box)")
        if len(mp4_bytes) > MAX_VIDEO_ARTIFACT_BYTES:
            raise RuntimeError(
                f"upstream MP4 ({len(mp4_bytes)} bytes) exceeds {MAX_VIDEO_ARTIFACT_BYTES} byte limit")
        elapsed_s = time.perf_counter() - started
        return VideoGenerationResult(
            mp4_bytes=mp4_bytes,
            width=req.width,
            height=req.height,
            fps=req.fps,
            frames=req.num_frames,
            seconds=float(req.seconds),
            model=self.model,
            revision=self.revision,
            elapsed_s=elapsed_s,
        )


class _Finished:
    __slots__ = ("request_id",)

    def __init__(self, request_id):
        self.request_id = request_id


class VideoStage:
    """Persistent single-request video stage speaking ummInfer ZMQ/msgpack."""

    def __init__(self, name, recv_endpoint, coordinator_endpoint, abort_endpoint,
                 backend, default_width=704, default_height=480,
                 default_seconds=2, max_seconds=6, default_fps=16,
                 max_frames=81, default_steps=20, max_steps=50,
                 default_guidance=5.0, perf_log=None):
        self.name = name
        self.recv_endpoint = recv_endpoint
        self.coordinator_endpoint = coordinator_endpoint
        self.abort_endpoint = abort_endpoint
        self.backend = backend
        self.default_width = int(default_width)
        self.default_height = int(default_height)
        self.default_seconds = int(default_seconds)
        self.max_seconds = int(max_seconds)
        self.default_fps = int(default_fps)
        self.max_frames = int(max_frames)
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
            print(f"video stage perf log write failed: {exc}", file=sys.stderr)

    def start(self):
        started = time.perf_counter()
        self.backend.load()
        self.write_perf({
            "event": "load",
            "stage": self.name,
            "model": getattr(self.backend, "model_id", getattr(self.backend, "model", "UNKNOWN")),
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
            f"ready kind=video name={self.name} "
            f"model={getattr(self.backend, 'model_id', getattr(self.backend, 'model', 'UNKNOWN'))} "
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
        for request_id in [k for k, deadline in self.aborted.items() if deadline <= now]:
            del self.aborted[request_id]

    def record_abort(self, request_id):
        self.expire_aborts()
        flag = self.abort_flags.get(request_id)
        if flag is not None:
            flag.set()
        removed = any(m.get("request_id") == request_id for m in self.pending)
        self.pending = deque(m for m in self.pending if m.get("request_id") != request_id)
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

    def parse_request(self, message) -> VideoRequest:
        data = message.get("data")
        if not isinstance(data, dict):
            raise ValueError("video submit missing data map")
        request = data.get("request")
        if not isinstance(request, dict):
            raise ValueError("video submit missing request map")
        inputs = request.get("inputs")
        if not isinstance(inputs, dict):
            raise ValueError("video submit missing inputs map")
        prompt = inputs.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("video.prompt must be a non-empty string")

        output_format = inputs.get("output_format", "mp4")
        if not isinstance(output_format, str) or output_format != "mp4":
            raise ValueError("video.output_format must be mp4")

        align = getattr(self.backend, "spatial_align", 32)
        width, height = parse_size(
            inputs.get("size"), self.default_width, self.default_height, align=align)

        seconds = inputs.get("seconds", self.default_seconds)
        if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds < 1:
            raise ValueError("video.seconds must be a positive integer")
        if seconds > self.max_seconds:
            raise ValueError(f"video.seconds {seconds} exceeds max {self.max_seconds}")

        fps = inputs.get("fps", self.default_fps)
        if isinstance(fps, bool) or not isinstance(fps, int) or not (1 <= fps <= 60):
            raise ValueError("video.fps must be an integer in [1, 60]")

        temporal_stride = getattr(self.backend, "temporal_stride", 4)
        if "num_frames" in inputs and inputs["num_frames"] is not None:
            num_frames = inputs["num_frames"]
            if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames < 1:
                raise ValueError("video.num_frames must be a positive integer")
            if num_frames > self.max_frames:
                raise ValueError(f"video.num_frames {num_frames} exceeds max {self.max_frames}")
            if temporal_stride > 1 and (num_frames - 1) % temporal_stride != 0:
                raise ValueError(
                    f"video.num_frames {num_frames} must satisfy (n - 1) % {temporal_stride} == 0")
        else:
            num_frames = align_frame_count(
                seconds, fps, self.max_frames, temporal_stride=temporal_stride)

        steps = inputs.get("steps", inputs.get("num_inference_steps", self.default_steps))
        if isinstance(steps, bool) or not isinstance(steps, int) or not (1 <= steps <= self.max_steps):
            raise ValueError(f"video.steps must be an integer in [1, {self.max_steps}]")

        guidance = inputs.get("guidance_scale", self.default_guidance)
        if isinstance(guidance, bool) or not isinstance(guidance, (int, float)) or not math.isfinite(guidance):
            raise ValueError("video.guidance_scale must be a finite number")

        seed = inputs.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("video.seed must be an integer")

        negative_prompt = inputs.get("negative_prompt", "")
        if not isinstance(negative_prompt, str):
            raise ValueError("video.negative_prompt must be a string")

        image_input = inputs.get("image")
        init_image_bytes = None
        if image_input is not None:
            if isinstance(image_input, bytes):
                init_image_bytes = image_input
            elif isinstance(image_input, str):
                if not image_input.startswith("data:image/") or "," not in image_input:
                    raise ValueError("video.image must be binary bytes or an image data URL")
                header, encoded = image_input.split(",", 1)
                if ";base64" not in header:
                    raise ValueError("video.image data URL must use base64")
                try:
                    init_image_bytes = base64.b64decode(encoded, validate=True)
                except (ValueError, base64.binascii.Error) as exc:
                    raise ValueError("video.image has invalid base64") from exc
            else:
                raise ValueError("video.image must be binary bytes or an image data URL")
            if not init_image_bytes or len(init_image_bytes) > 16 * 1024 * 1024:
                raise ValueError("video.image must be non-empty and at most 16 MiB")
        if inputs.get("anchor_explicit", False) and init_image_bytes is None:
            raise ValueError("explicit I2V anchor requires video.image bytes")
        anchor_sha256 = inputs.get("anchor_sha256")
        if anchor_sha256 is not None:
            if init_image_bytes is None or not isinstance(anchor_sha256, str) \
                    or hashlib.sha256(init_image_bytes).hexdigest() != anchor_sha256:
                raise ValueError("video.image does not match anchor_sha256")

        return VideoRequest(
            prompt=prompt.strip(),
            width=width,
            height=height,
            seconds=seconds,
            fps=fps,
            num_frames=num_frames,
            steps=steps,
            guidance_scale=float(guidance),
            seed=seed,
            negative_prompt=negative_prompt,
            output_format=output_format,
            init_image_bytes=init_image_bytes,
        )

    def complete_artifact(self, request_id, res: VideoGenerationResult):
        return {
            "type": "complete",
            "request_id": request_id,
            "from_stage": self.name,
            "success": True,
            "result": {
                "media_type": "video",
                "mime_type": "video/mp4",
                "format": "mp4",
                "data": res.mp4_bytes,
                "size": f"{res.width}x{res.height}",
                "width": res.width,
                "height": res.height,
                "fps": res.fps,
                "frames": res.frames,
                "seconds": res.seconds,
                "model": res.model,
                "revision": res.revision,
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
            print("video stage discarded submit without a request id", file=sys.stderr)
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
                "fps": req.fps,
                "num_frames": req.num_frames,
                "steps": req.steps,
                "seed": req.seed,
            })
            res = self.backend.generate(req, abort_flag)
            if abort_flag.is_set() or not self.running:
                return
            self.write_perf({
                "event": "request",
                "request_id": request_id,
                "bytes": len(res.mp4_bytes),
                "width": res.width,
                "height": res.height,
                "fps": res.fps,
                "frames": res.frames,
                "seconds": res.seconds,
                "elapsed_s": round(res.elapsed_s, 3),
                "model": res.model,
                "revision": res.revision,
            })
            print(
                f"video request request_id={request_id} status=ok "
                f"bytes={len(res.mp4_bytes)} size={res.width}x{res.height} "
                f"frames={res.frames} fps={res.fps} elapsed_s={res.elapsed_s:.2f}",
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
                f"video request request_id={request_id} status=aborted reason={exc}",
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
                f"video request request_id={request_id} status=error error={error}",
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
                    print(f"video stage discarded malformed abort: {exc}", file=sys.stderr)
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
                    print(f"video stage ignored message type {mtype!r}", file=sys.stderr)
            except Exception as exc:
                print(f"video stage discarded malformed control message: {exc}", file=sys.stderr)

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
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{self.name}-worker")
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
    parser.add_argument("--name", default="video")
    parser.add_argument("--recv", required=True)
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--abort", required=True)
    parser.add_argument("--backend", choices=("diffusers", "upstream"), default="diffusers")
    parser.add_argument("--model", default="", help="Local Diffusers model directory (diffusers backend)")
    parser.add_argument("--model-id", default="", help="Human-readable model identifier for metadata")
    parser.add_argument("--revision", default="", help="Pinned model revision SHA (defaults to auto-detect or UNKNOWN)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--upstream-base-url", default="")
    parser.add_argument("--upstream-model", default="")
    parser.add_argument("--upstream-api-key", default="")
    parser.add_argument("--default-width", type=int, default=704)
    parser.add_argument("--default-height", type=int, default=480)
    parser.add_argument("--default-seconds", type=int, default=2)
    parser.add_argument("--max-seconds", type=int, default=6)
    parser.add_argument("--default-fps", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=81)
    parser.add_argument("--default-steps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--default-guidance", type=float, default=5.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--perf-log", default=None)
    args = parser.parse_args(argv)

    if args.backend == "diffusers":
        if not args.model:
            parser.error("--model is required for --backend diffusers")
        backend = DiffusersVideoBackend(
            model_path=args.model,
            gpu=args.gpu,
            model_id=args.model_id or None,
            revision=args.revision or None,
            ffmpeg_bin=args.ffmpeg,
        )
    else:
        if not args.upstream_base_url or not args.upstream_model:
            parser.error("--upstream-base-url and --upstream-model are required for --backend upstream")
        backend = UpstreamVideoBackend(
            base_url=args.upstream_base_url,
            model=args.upstream_model,
            api_key=args.upstream_api_key,
            revision=args.revision or None,
        )

    stage = VideoStage(
        name=args.name,
        recv_endpoint=args.recv,
        coordinator_endpoint=args.coordinator,
        abort_endpoint=args.abort,
        backend=backend,
        default_width=args.default_width,
        default_height=args.default_height,
        default_seconds=args.default_seconds,
        max_seconds=args.max_seconds,
        default_fps=args.default_fps,
        max_frames=args.max_frames,
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
