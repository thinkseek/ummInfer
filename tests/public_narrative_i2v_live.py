#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real public VLM narrative image-slot -> explicit native-I2V gate on GPU 1.

Offline check:
  python3 -I tests/public_narrative_i2v_live.py --self-check

The opt-in live mode starts only task-owned processes on physical GPU 1. A
serial Qwen3.8 VLM stays resident. FLUX.2, native streaming TTS, and native
Wan2.2 run one at a time, in that order. Read-only ZMQ taps forward submits
unchanged while retaining exact TTS text and Wan source-byte binding evidence.
"""
import argparse
import base64
import hashlib
import http.client
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import wave

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from umminfer_rl_trajectory import (  # noqa: E402
    TrajectoryRecorder, input_references, parse_sse_stream,
    request_parameter_references,
)
from umminfer_sse_replay import PerfCounterClock, RawSSECapture  # noqa: E402

SEED = 42
SOURCE_SIZE = "512x512"
VIDEO_SIZE = "704x480"
FRAMES = 29
STEPS = 20
FPS = 16
GUIDANCE = 5.0
FIRST_FRAME_MAX_MAE = 48.0
MOTION_MIN_MAE = 1.0
METRIC_VERSION = "center-cover-64x40-rgb8-frame0-frame28-v1"
REPORT_FORMAT = "umminfer-public-narrative-i2v-v1"
DEFAULT_LIBRARY_PATH = ""
IMAGE_DIRECTIVE = (
    "[[image: a single vivid red paper lantern above a misty blue mountain lake "
    "at dawn, cinematic composition, clear subject, gentle ripples]]")
VIDEO_DIRECTIVE = (
    "[[video: the red lantern sways gently while the camera slowly orbits above "
    "the misty lake | source=slot-0]]")
USER_TEXT = (
    "Create a two-beat visual story. Output one short standalone English prose "
    "sentence, then exactly this standalone marker: " + IMAGE_DIRECTIVE +
    " Then output one short standalone English prose sentence and exactly this "
    "standalone marker: " + VIDEO_DIRECTIVE +
    " Do not output any other marker or repeat these instructions.")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def clock_ns():
    return time.perf_counter_ns()


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(8 << 20)
            if not block:
                break
            size += len(block)
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": size}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def spoken_text_projection(text):
    # The engine deliberately trims each streamed sentence/clause at its span
    # boundaries. Compare linguistic content without inventing spoken whitespace.
    return " ".join(text.split())


def model_identity(root, model_id):
    root = Path(root)
    require(root.is_dir(), f"model directory is unavailable: {root}")
    files = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        files[path.relative_to(root).as_posix()] = sha256_file(path)
    require(any(name.endswith(".safetensors") for name in files),
            f"no model weights under {root}")
    require("config.json" in files or "model_index.json" in files,
            f"model config/manifest missing under {root}")
    return {"model_id": model_id, "root": str(root), "file_count": len(files),
            "files": files, "tree_sha256": sha256_bytes(canonical(files))}


def runtime_identity():
    names = ("torch", "diffusers", "transformers", "huggingface-hub", "tokenizers",
             "safetensors", "accelerate", "numpy", "pillow", "pyzmq", "msgpack")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "ABSENT"
    return versions


def request_body(model):
    return {
        "model": model, "stream": True,
        "messages": [{"role": "user", "content": USER_TEXT}],
        "modalities": ["text", "image", "audio", "video"],
        "image": {"size": SOURCE_SIZE, "steps": 4, "seed": SEED,
                  "output_format": "png"},
        "audio": {"voice": "vivian", "format": "wav", "stream": True},
        "video": {"size": VIDEO_SIZE, "num_frames": FRAMES, "steps": STEPS,
                  "fps": FPS, "guidance_scale": GUIDANCE, "seed": SEED,
                  "output_format": "mp4"},
        "narrative": {"enabled": True, "visual_mode": "auto", "max_slots": 2,
                      "allow_text_only": False, "continuity_mode": "i2v_chain"},
        "max_tokens": 224, "temperature": 0, "seed": SEED,
    }


def gpu1_memory(nvidia_smi):
    result = subprocess.run(
        [nvidia_smi, "-i", "1", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], check=True, capture_output=True,
        text=True, timeout=20)
    fields = [int(value.strip()) for value in result.stdout.strip().split(",")]
    require(len(fields) == 3 and fields[0] == 1,
            f"unexpected physical GPU 1 response: {result.stdout!r}")
    return {"index": fields[0], "memory_used_mib": fields[1],
            "utilization_percent": fields[2]}


def require_gpu1_idle(nvidia_smi):
    observed = gpu1_memory(nvidia_smi)
    require(observed["memory_used_mib"] <= 16,
            f"physical GPU 1 is not idle: {observed}")
    return observed


def http_text(port, path, timeout=30):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        raw = response.read()
        require(response.status == 200, f"GET {path} returned {response.status}: {raw!r}")
        return raw.decode("utf-8")
    finally:
        conn.close()


def coordinator_stage_state(port, stage):
    counters = {status: 0 for status in ("submitted", "completed", "failed", "aborted")}
    in_flight = None
    for line in http_text(port, "/metrics", timeout=10).splitlines():
        if line.startswith("umm_stage_requests_total{") and f'stage="{stage}"' in line:
            for status in counters:
                if f'status="{status}"' in line:
                    counters[status] += int(float(line.rsplit(" ", 1)[1]))
        elif line.startswith("umm_stage_in_flight{") and f'stage="{stage}"' in line:
            in_flight = int(float(line.rsplit(" ", 1)[1]))
    require(in_flight is not None, f"Coordinator has no {stage} in-flight metric")
    counters["in_flight"] = in_flight
    return counters


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def http_json(port, method, path, body=None, timeout=30):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        payload = None if body is None else json.dumps(body)
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        conn.request(method, path, body=payload, headers=headers)
        response = conn.getresponse()
        raw = response.read()
        return response.status, json.loads(raw) if raw else {}
    finally:
        conn.close()


class ManagedProcess:
    def __init__(self, name, command, log_path, env=None):
        self.name = name
        self.command = [str(item) for item in command]
        self.log_path = Path(log_path)
        self.env = env
        self.start_ns = clock_ns()
        self.ready_ns = None
        self.stop_ns = None
        self.returncode = None
        self.stop_error = None
        self.log = self.log_path.open("wb")
        self.process = subprocess.Popen(
            self.command, stdout=self.log, stderr=subprocess.STDOUT, env=self.env,
            start_new_session=True)

    def wait_log(self, needle, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.log_path.exists() and needle in self.log_path.read_text(
                    encoding="utf-8", errors="replace"):
                self.ready_ns = clock_ns()
                return
            if self.process.poll() is not None:
                raise RuntimeError(f"{self.name} exited before readiness; see {self.log_path}")
            time.sleep(0.05)
        raise TimeoutError(f"{self.name} did not log readiness marker {needle!r}")

    def stop(self, shutdown_endpoint=None, timeout=45):
        if self.stop_ns is not None:
            return
        context = None
        push = None
        try:
            if self.process.poll() is None and shutdown_endpoint is not None \
                    and self.ready_ns is not None:
                import msgpack
                import zmq
                context = zmq.Context()
                push = context.socket(zmq.PUSH)
                push.connect(shutdown_endpoint)
                push.send(msgpack.packb({"type": "shutdown"}, use_bin_type=True),
                          zmq.NOBLOCK)
            elif self.process.poll() is None:
                self.process.terminate()
            if self.process.poll() is None:
                self.process.wait(timeout=timeout)
        except Exception as error:
            self.stop_error = str(error)
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        finally:
            if push is not None:
                push.close(0)
            if context is not None:
                context.term()
        self.stop_ns = clock_ns()
        self.returncode = self.process.returncode
        self.log.close()

    def record(self):
        return {"name": self.name, "pid": self.process.pid,
                "argv_sha256": sha256_bytes("\0".join(self.command).encode()),
                "start_monotonic_ns": self.start_ns, "ready_monotonic_ns": self.ready_ns,
                "stop_monotonic_ns": self.stop_ns, "returncode": self.returncode,
                "stop_error": self.stop_error, "log": self.log_path.name}


class TapProxy:
    """One read-only PUSH/PULL forwarder; packed submits are retained by hash."""
    def __init__(self, name, listen, destination, evidence_dir):
        self.name = name
        self.listen = listen
        self.destination = destination
        self.evidence_dir = Path(evidence_dir)
        self.records = []
        self.error = None
        self.stop_requested = False
        self.thread = threading.Thread(target=self._run, name=f"tap-{name}", daemon=True)
        self.thread.start()

    def _run(self):
        import msgpack
        import zmq
        context = pull = push = None
        try:
            context = zmq.Context()
            pull = context.socket(zmq.PULL)
            pull.bind(self.listen)
            push = context.socket(zmq.PUSH)
            push.setsockopt(zmq.LINGER, 0)
            push.connect(self.destination)
            while not self.stop_requested:
                try:
                    packed = pull.recv(zmq.NOBLOCK)
                except zmq.Again:
                    time.sleep(0.005)
                    continue
                tick = clock_ns()
                message = msgpack.unpackb(packed, raw=False)
                request_id = message.get("request_id")
                record = {"tap": self.name, "request_id": request_id,
                          "observed_monotonic_ns": tick,
                          "packed_sha256": sha256_bytes(packed),
                          "packed_bytes": len(packed)}
                if message.get("type") == "shutdown":
                    forwarded = False
                    while not forwarded and not self.stop_requested:
                        try:
                            push.send(packed, zmq.NOBLOCK)
                            forwarded = True
                        except zmq.Again:
                            time.sleep(0.005)
                    if not forwarded:
                        break
                    continue
                if self.name == "tts-output":
                    message_type = message.get("type")
                    record["message_type"] = message_type
                    if message_type == "stream":
                        chunk = message.get("chunk") or {}
                        audio = chunk.get("audio")
                        require(isinstance(audio, bytes) and bool(audio),
                                "native TTS stream lacks PCM bytes")
                        record.update({
                            "stream_index": chunk.get("index"),
                            "audio_sha256": sha256_bytes(audio), "audio_bytes": len(audio),
                            "sample_rate": chunk.get("sample_rate"),
                            "channels": chunk.get("channels"),
                        })
                    elif message_type == "complete":
                        result = message.get("result") or {}
                        wav = result.get("data")
                        require(isinstance(wav, bytes) and bool(wav),
                                "native TTS terminal lacks WAV bytes")
                        record.update({"success": message.get("success"),
                                       "streamed": result.get("streamed"),
                                       "wav_sha256": sha256_bytes(wav), "wav_bytes": len(wav)})
                    packed_name = (f"{self.name}-{len(self.records)}-"
                                   f"{record['packed_sha256']}.msgpack")
                    (self.evidence_dir / packed_name).write_bytes(packed)
                    record["packed_file"] = packed_name
                    forwarded = False
                    while not forwarded and not self.stop_requested:
                        try:
                            push.send(packed, zmq.NOBLOCK)
                            forwarded = True
                        except zmq.Again:
                            time.sleep(0.005)
                    if not forwarded:
                        break
                    record["forwarded_monotonic_ns"] = clock_ns()
                    self.records.append(record)
                    continue
                inputs = (((message.get("data") or {}).get("request") or {})
                          .get("inputs") or {})
                packed_name = f"{self.name}-{len(self.records)}-{record['packed_sha256']}.msgpack"
                (self.evidence_dir / packed_name).write_bytes(packed)
                record["packed_file"] = packed_name
                if self.name == "video":
                    image = inputs.get("image")
                    require(isinstance(image, bytes) and bool(image),
                            "Wan submit lacks binary source image")
                    record.update({
                        "prompt": inputs.get("prompt"),
                        "image_sha256": sha256_bytes(image), "image_bytes": len(image),
                        "anchor_slot_index": inputs.get("anchor_slot_index"),
                        "anchor_explicit": inputs.get("anchor_explicit"),
                        "anchor_sha256": inputs.get("anchor_sha256"),
                        "anchor_crop": inputs.get("anchor_crop"),
                    })
                elif self.name == "tts":
                    text = inputs.get("text")
                    require(isinstance(text, str), "TTS submit lacks text")
                    record.update({"text": text, "text_sha256": sha256_bytes(text.encode()),
                                   "stream": inputs.get("stream")})
                forwarded = False
                while not forwarded and not self.stop_requested:
                    try:
                        push.send(packed, zmq.NOBLOCK)
                        forwarded = True
                    except zmq.Again:
                        time.sleep(0.005)
                if not forwarded:
                    break
                record["forwarded_monotonic_ns"] = clock_ns()
                self.records.append(record)
        except Exception as error:
            self.error = str(error)
        finally:
            if pull is not None:
                pull.close(0)
            if push is not None:
                push.close(0)
            if context is not None:
                context.term()

    def stop(self):
        self.stop_requested = True
        self.thread.join(timeout=10)
        require(not self.thread.is_alive(), f"{self.name} tap did not stop")
        require(self.error is None, f"{self.name} tap failed: {self.error}")


class GpuMonitor:
    def __init__(self, nvidia_smi):
        self.nvidia_smi = nvidia_smi
        self.samples = []
        self.stop_requested = False
        self.error = None
        self.thread = threading.Thread(target=self._run, name="gpu1-monitor", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stop_requested:
                self.samples.append({"observed_monotonic_ns": clock_ns(),
                                     **gpu1_memory(self.nvidia_smi)})
                time.sleep(0.05)
        except Exception as error:
            self.error = str(error)

    def stop(self):
        self.stop_requested = True
        self.thread.join(timeout=10)
        require(not self.thread.is_alive() and self.error is None,
                f"GPU monitor failed: {self.error}")


def decode_ppm(payload):
    require(payload.startswith(b"P6\n"), "expected binary PPM")
    header, raw = payload.split(b"\n255\n", 1)
    width, height = (int(value) for value in header.splitlines()[1].split())
    require(len(raw) == width * height * 3, "PPM RGB byte count mismatch")
    return width, height, raw


def rgb_frame(ffmpeg, media, selector=None, evaluation=False):
    filters = []
    if selector:
        filters.append(selector)
    if evaluation:
        filters.extend(["scale=64:40:force_original_aspect_ratio=increase", "crop=64:40"])
    command = [ffmpeg, "-v", "error", "-i", str(media)]
    if filters:
        command += ["-vf", ",".join(filters)]
    command += ["-frames:v", "1", "-f", "image2pipe", "-vcodec", "ppm", "-"]
    result = subprocess.run(command, check=True, capture_output=True, timeout=60)
    return decode_ppm(result.stdout)


def mean_absolute_rgb(first, second):
    require(len(first) == len(second) and bool(first), "RGB buffers differ or are empty")
    return sum(abs(a - b) for a, b in zip(first, second)) / len(first)


def probe_video(ffprobe, path):
    result = subprocess.run([
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=codec_name,width,height,nb_frames,r_frame_rate,pix_fmt",
        "-of", "json", str(path)], check=True, capture_output=True, text=True, timeout=30)
    stream = json.loads(result.stdout)["streams"][0]
    require(stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
            and (stream["width"], stream["height"]) == (704, 480)
            and int(stream["nb_frames"]) == FRAMES
            and stream["r_frame_rate"] == f"{FPS}/1",
            f"unexpected native Wan video: {stream}")
    return stream


def trace_records(path):
    if not Path(path).exists():
        return []
    records = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def join_tts_pcm(submits, output_records, public_audio, evidence_dir):
    """Join every native leg/chunk to exact public PCM and terminal WAV bytes."""
    import msgpack
    joins = []
    public_cursor = 0
    for submit in submits:
        child_id = submit["request_id"]
        native_chunks = [item for item in output_records
                         if item.get("message_type") == "stream"
                         and item.get("request_id") == child_id]
        terminals = [item for item in output_records
                     if item.get("message_type") == "complete"
                     and item.get("request_id") == child_id]
        require(len(terminals) == 1 and terminals[0].get("success") is True
                and terminals[0].get("streamed") is True,
                f"native TTS leg did not complete streamed: {child_id}")
        require(native_chunks, f"native TTS leg emitted no PCM: {child_id}")
        native_raw = []
        for expected_index, native in enumerate(native_chunks):
            require(native["stream_index"] == expected_index,
                    f"native TTS chunk index gap: {child_id}")
            require(public_cursor < len(public_audio),
                    f"public PCM missing for native leg {child_id}")
            public_item, public_raw = public_audio[public_cursor]
            require(public_item.get("stream_index") == expected_index,
                    f"public PCM cursor mismatch for {child_id}")
            native_packed = (Path(evidence_dir) / native["packed_file"]).read_bytes()
            native_message = msgpack.unpackb(native_packed, raw=False)
            native_audio = native_message["chunk"]["audio"]
            require(public_item["sha256"] == native["audio_sha256"]
                    and public_raw == native_audio,
                    f"public PCM differs from native chunk {child_id}:{expected_index}")
            native_raw.append(native_audio)
            public_cursor += 1
        complete_packed = (Path(evidence_dir) / terminals[0]["packed_file"]).read_bytes()
        complete_message = msgpack.unpackb(complete_packed, raw=False)
        wav_raw = complete_message["result"]["data"]
        require(sha256_bytes(wav_raw) == terminals[0]["wav_sha256"],
                f"native WAV digest mismatch: {child_id}")
        with wave.open(io.BytesIO(wav_raw), "rb") as wav:
            require(wav.getnchannels() == 1 and wav.getsampwidth() == 2
                    and wav.getframerate() == 24000, f"unexpected native WAV: {child_id}")
            wav_pcm = wav.readframes(wav.getnframes())
        native_pcm = b"".join(native_raw)
        require(native_pcm == wav_pcm,
                f"public/native chunk concatenation differs from terminal WAV: {child_id}")
        joins.append({"request_id": child_id, "chunk_count": len(native_chunks),
                      "pcm_bytes": len(native_pcm), "pcm_sha256": sha256_bytes(native_pcm),
                      "terminal_wav_sha256": sha256_bytes(wav_raw),
                      "public_refs": [public_audio[public_cursor - len(native_chunks) + offset][0]["ref"]
                                      for offset in range(len(native_chunks))],
                      "exact_native_public_wav_join": True})
    require(public_cursor == len(public_audio),
            f"unowned public PCM chunks remain: {len(public_audio) - public_cursor}")
    return joins


def live(args):
    require(os.environ.get("UMMINFER_LIVE_OPT_IN") == "1",
            "REAL_GENERATION: NOT_RUN (explicit UMMINFER_LIVE_OPT_IN=1 required)")
    require(args.gpu == 1, "this gate permits only physical GPU 1")
    require(len(args.base_commit) == 40 and len(args.execution_commit) == 40,
            "full base and execution commits are required")
    out = Path(args.out_dir)
    require(not out.exists(), f"evidence directory already exists: {out}")
    out.mkdir(parents=True)
    run = out / "run"
    run.mkdir()
    (out / "capture").mkdir()
    provenance = {
        "base_commit": args.base_commit, "execution_commit": args.execution_commit,
        "server_binary": {"path": str(args.server_binary), **sha256_file(args.server_binary)},
        "vlm_binary": {"path": str(args.vlm_binary), **sha256_file(args.vlm_binary)},
        "tts_binary": {"path": str(args.tts_binary), **sha256_file(args.tts_binary)},
        "wan_binary": {"path": str(args.wan_binary), **sha256_file(args.wan_binary)},
        "flux_stage": {"path": str(args.flux_stage), **sha256_file(args.flux_stage)},
        "live_harness": {"path": str(Path(__file__).resolve()), **sha256_file(__file__)},
        "independent_verifier": {"path": str(args.verifier), **sha256_file(args.verifier)},
        "ffmpeg": sha256_file(shutil.which(args.ffmpeg)),
        "ffprobe": sha256_file(shutil.which(args.ffprobe)),
        "python_runtime": runtime_identity(),
    }
    models = {
        "vlm": model_identity(args.vlm_model, "Qwen/Qwen3.8-27B"),
        "flux2": model_identity(args.flux_model, "black-forest-labs/FLUX.2-klein-4B"),
        "tts": model_identity(args.tts_model, "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        "wan22": model_identity(args.wan_model, "Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
    }
    body = request_body(args.model)
    (out / "request.json").write_bytes(canonical(body) + b"\n")
    gpu = {"before": require_gpu1_idle(args.nvidia_smi)}
    monitor = GpuMonitor(args.nvidia_smi)
    port = args.port or free_port()
    require(port != 43301, "resident service port is forbidden")
    endpoints = {name: f"ipc://{run}/{name}" for name in
                 ("completion", "abort", "entry-in", "vlm-in", "image-in", "tts-tap-in",
                  "tts-in", "tts-output-in", "video-tap-in", "video-in")}
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if args.library_path:
        env["LD_LIBRARY_PATH"] = str(args.library_path) + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["UMM_TRACE"] = "1"
    env["UMM_TRACE_FILE"] = str(out / "server.trace.jsonl")
    server = entry = vlm = flux = tts = wan = None
    video_tap = tts_tap = tts_output_tap = None
    source_ready = threading.Event()
    request_ready = threading.Event()
    capture_done = threading.Event()
    manager_error = []
    manager_record = {"flux_to_tts": False, "tts_to_wan": False}
    stage_record = {}
    recorder = TrajectoryRecorder(out / "capture" / "trajectories.jsonl")
    record = source_final = audio_streams = None
    health_final = None

    def start_tts():
        nonlocal tts
        tts = ManagedProcess("tts", [
            args.tts_binary, "--kind", "tts", "--name", "tts", "--model", args.tts_model,
            "--gpu", "1", "--recv-endpoint", endpoints["tts-in"],
            "--coordinator-endpoint", endpoints["tts-output-in"],
            "--abort-endpoint", endpoints["abort"]], out / "tts.log", env=env)
        tts.wait_log("native stage tts ready kind=tts", args.timeout)
        gpu["tts_ready"] = gpu1_memory(args.nvidia_smi)

    def start_wan():
        nonlocal wan
        wan = ManagedProcess("video", [
            args.wan_binary, "--kind", "video", "--name", "video", "--video-backend", "cuda",
            "--model", args.wan_model, "--gpu", "1", "--size", VIDEO_SIZE,
            "--steps", str(STEPS), "--fps", str(FPS), "--recv-endpoint", endpoints["video-in"],
            "--coordinator-endpoint", endpoints["completion"],
            "--abort-endpoint", endpoints["abort"]], out / "video.log", env=env)
        wan.wait_log("native stage video ready kind=video", args.timeout)
        gpu["wan_ready"] = gpu1_memory(args.nvidia_smi)

    def output_manager():
        try:
            require(request_ready.wait(args.timeout), "no public request id observed")
            while not source_ready.wait(0.05):
                require(not capture_done.is_set(),
                        "public stream ended before a source image completed")
            stage_record["source_public_complete_ns"] = clock_ns()
            flux.stop(endpoints["image-in"])
            require(flux.returncode == 0, f"FLUX shutdown failed: {flux.returncode}")
            gpu["after_flux_stop_vlm_resident"] = gpu1_memory(args.nvidia_smi)
            stage_record["flux_stopped_ns"] = clock_ns()
            start_tts()
            manager_record["flux_to_tts"] = True
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                traces = trace_records(out / "server.trace.jsonl")
                outputs_started = any(row.get("event") == "legs_snapshot"
                                      and row.get("phase") == "outputs_started"
                                      and row.get("request_id") == record_request_id[0]
                                      for row in traces)
                state = coordinator_stage_state(port, "tts")
                settled = state["completed"] + state["failed"] + state["aborted"]
                if outputs_started and state["submitted"] > 0 \
                        and settled == state["submitted"]:
                    require(state["failed"] == 0 and state["aborted"] == 0
                            and state["completed"] == state["submitted"]
                            and state["in_flight"] == 0,
                            f"TTS Coordinator terminals are not cleanly complete: {state}")
                    tts_log = (out / "tts.log").read_text(encoding="utf-8", errors="replace")
                    native_ends = tts_log.count("tts request request_id=")
                    tapped_terminals = sum(
                        item.get("message_type") == "complete"
                        and item.get("success") is True
                        for item in tts_output_tap.records)
                    require(native_ends == state["submitted"]
                            and len(tts_tap.records) == state["submitted"]
                            and tapped_terminals == state["submitted"],
                            "TTS submit/native/tap counts are incomplete before stage switch: "
                            f"{state} native={native_ends} inputs={len(tts_tap.records)} "
                            f"outputs={tapped_terminals}")
                    stage_record["tts_coordinator_complete"] = state
                    break
                if tts.process.poll() is not None:
                    raise RuntimeError("TTS exited before all streaming legs completed")
                time.sleep(0.02)
            else:
                raise TimeoutError("TTS streaming legs did not settle after outputs_started")
            tts.stop(endpoints["tts-in"])
            require(tts.returncode == 0, f"TTS shutdown failed: {tts.returncode}")
            tts_tap.stop()
            tts_output_tap.stop()
            gpu["after_tts_stop_vlm_resident"] = gpu1_memory(args.nvidia_smi)
            stage_record["tts_stopped_ns"] = clock_ns()
            start_wan()
            manager_record["tts_to_wan"] = True
        except Exception as error:
            manager_error.append(str(error))
            if record_request_id[0] is not None:
                try:
                    http_json(port, "POST", "/abort", {
                        "request_id": record_request_id[0],
                        "session_id": "public-narrative-i2v-65"}, timeout=30)
                except Exception:
                    pass
            source_ready.set()

    record_request_id = [None]
    try:
        video_tap = TapProxy("video", endpoints["video-tap-in"], endpoints["video-in"], out)
        tts_tap = TapProxy("tts", endpoints["tts-tap-in"], endpoints["tts-in"], out)
        tts_output_tap = TapProxy(
            "tts-output", endpoints["tts-output-in"], endpoints["completion"], out)
        server = ManagedProcess("server", [
            args.server_binary, "serve", "--host", "127.0.0.1", "--port", str(port),
            "--model", args.model, "--entry", "entry",
            "--entry-endpoint", endpoints["entry-in"], "--vlm-entry", "vlm",
            "--vlm-entry-endpoint", endpoints["vlm-in"],
            "--image-entry", "image", "--image-entry-endpoint", endpoints["image-in"],
            "--tts-entry", "tts", "--tts-entry-endpoint", endpoints["tts-tap-in"],
            "--video-entry", "video", "--video-entry-endpoint", endpoints["video-tap-in"],
            "--completion", endpoints["completion"], "--abort", endpoints["abort"],
            "--max-in-flight", "1"], out / "server.log", env=env)
        entry = ManagedProcess("entry", [
            args.server_binary, "stage", "--name", "entry", "--recv", endpoints["entry-in"],
            "--coordinator", endpoints["completion"], "--abort", endpoints["abort"]],
            out / "entry.log", env=env)
        vlm = ManagedProcess("vlm", [
            args.vlm_binary, "--kind", "qwen38", "--name", "vlm", "--model", args.vlm_model,
            "--gpu", "1", "--batch-size", "1", "--max-new-tokens", "224",
            "--recv-endpoint", endpoints["vlm-in"],
            "--coordinator-endpoint", endpoints["completion"],
            "--abort-endpoint", endpoints["abort"]], out / "vlm.log", env=env)
        vlm.wait_log("native stage vlm ready kind=qwen38", args.timeout)
        gpu["vlm_ready"] = gpu1_memory(args.nvidia_smi)
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            try:
                status, _ = http_json(port, "GET", "/ready", timeout=5)
                if status == 200:
                    entry.ready_ns = clock_ns()
                    break
            except (OSError, http.client.HTTPException):
                pass
            require(server.process.poll() is None, "public gateway exited before readiness")
            time.sleep(0.05)
        else:
            raise TimeoutError("public gateway did not become ready")
        flux = ManagedProcess("image", [
            args.python, "-I", args.flux_stage, "--name", "image", "--backend", "diffusers",
            "--model", args.flux_model, "--model-id", "black-forest-labs/FLUX.2-klein-4B",
            "--revision", "UNKNOWN", "--gpu", "1", "--default-size", SOURCE_SIZE,
            "--default-steps", "4", "--recv", endpoints["image-in"],
            "--coordinator", endpoints["completion"], "--abort", endpoints["abort"]],
            out / "image.log", env=env)
        flux.wait_log("ready kind=image name=image", args.timeout)
        gpu["flux_ready"] = gpu1_memory(args.nvidia_smi)
        manager = threading.Thread(target=output_manager, name="output-manager", daemon=True)
        manager.start()
        try:
            record, source_final, audio_streams = capture_public_with_id(
                port, body, recorder, source_ready, request_ready, record_request_id)
        finally:
            capture_done.set()
        manager.join(timeout=10)
        require(not manager.is_alive(), "output manager did not exit")
        require(not manager_error, f"output stage sequencing failed: {manager_error}")
        require(manager_record == {"flux_to_tts": True, "tts_to_wan": True},
                f"output stages were not serially switched: {manager_record}")
        require(record["terminal_state"] == "completed",
                f"public narrative failed: {record['error']}")
        video_tap.stop()
        health_status, health_final = http_json(port, "GET", "/health", timeout=30)
        require(health_status == 200 and health_final.get("in_flight") == 0,
                f"public gateway retained in-flight work: {health_status} {health_final}")
        gpu["after_complete_before_cleanup"] = gpu1_memory(args.nvidia_smi)
    finally:
        for process, endpoint in ((wan, endpoints["video-in"]),
                                  (tts, endpoints["tts-in"]),
                                  (flux, endpoints["image-in"]),
                                  (vlm, endpoints["vlm-in"]),
                                  (entry, endpoints["entry-in"])):
            if process is not None and process.stop_ns is None:
                process.stop(endpoint)
        if server is not None:
            server.stop(None)
        for tap in (video_tap, tts_tap, tts_output_tap):
            if tap is not None and tap.thread.is_alive():
                tap.stop()
        monitor.stop()
        gpu["after_cleanup"] = require_gpu1_idle(args.nvidia_smi)
        stage_record["processes"] = [process.record() for process in
                                     (server, entry, vlm, flux, tts, wan) if process is not None]
        (out / "gpu1-monitor.json").write_text(
            json.dumps(monitor.samples, indent=2, sort_keys=True) + "\n")
        require(not manager_error, f"live run failed; raw evidence retained: {manager_error}")

    require(record is not None and source_final is not None, "no retained public record")
    artifacts = record["artifacts"]
    image = next(item for item in artifacts if item["media_type"] == "image")
    video = next(item for item in artifacts if item["media_type"] == "video")
    audio = [item for item in artifacts if item["media_type"] == "audio"]
    source_path = recorder.artifact_dir / image["ref"]
    video_path = recorder.artifact_dir / video["ref"]
    require(source_path.read_bytes() and video_path.read_bytes(), "empty media CAS")
    (out / "source.png").write_bytes(source_path.read_bytes())
    (out / "video.mp4").write_bytes(video_path.read_bytes())
    rgb = {}
    for label, path, selector, evaluation in (
            ("source_full", source_path, None, False),
            ("source_eval", source_path, None, True),
            ("first_full", video_path, "select=eq(n\\,0)", False),
            ("last_full", video_path, "select=eq(n\\,28)", False),
            ("first_eval", video_path, "select=eq(n\\,0)", True),
            ("last_eval", video_path, "select=eq(n\\,28)", True)):
        width, height, raw = rgb_frame(args.ffmpeg, path, selector, evaluation)
        name = f"{label}.rgb{width}x{height}"
        (out / name).write_bytes(raw)
        rgb[label] = {"file": name, "width": width, "height": height, **sha256_file(out / name)}
    first_mae = round(mean_absolute_rgb(
        (out / rgb["source_eval"]["file"]).read_bytes(),
        (out / rgb["first_eval"]["file"]).read_bytes()), 6)
    motion_mae = round(mean_absolute_rgb(
        (out / rgb["first_eval"]["file"]).read_bytes(),
        (out / rgb["last_eval"]["file"]).read_bytes()), 6)
    require(first_mae <= FIRST_FRAME_MAX_MAE, f"first-frame MAE gate failed: {first_mae}")
    require(motion_mae >= MOTION_MIN_MAE, f"motion MAE gate failed: {motion_mae}")
    pcm = []
    public_audio = []
    for item in audio:
        raw = (recorder.artifact_dir / item["ref"]).read_bytes()
        public_audio.append((item, raw))
        require(len(raw) % 2 == 0 and raw, "invalid PCM16 artifact")
        samples = [int.from_bytes(raw[index:index + 2], "little", signed=True)
                   for index in range(0, len(raw), 2)]
        rms = (sum(sample * sample for sample in samples) / len(samples)) ** 0.5
        require(rms > 1.0, "TTS artifact is silent")
        pcm.append({"ref": item["ref"], "sha256": item["sha256"], "bytes": len(raw),
                    "sample_rate": item.get("sample_rate", 24000),
                    "channels": item.get("channels", 1), "rms": rms})
    require(pcm, "no streaming speech artifacts")
    tts_joins = join_tts_pcm(tts_tap.records, tts_output_tap.records, public_audio, out)
    require(len(video_tap.records) == 1
            and len(tts_joins) == len(tts_tap.records) and bool(tts_joins),
            "unexpected Wan/TTS submit counts")
    video_bind = video_tap.records[0]
    require(video_bind["image_sha256"] == image["sha256"]
            == video_bind["anchor_sha256"] == video["anchor_sha256"],
            "Wan submit/source/public anchor hashes differ")
    tts_text = " ".join(item["text"] for item in tts_tap.records)
    require(spoken_text_projection(tts_text)
            == spoken_text_projection(record["clean_text"]),
            "TTS text differs from clean narrative prose beyond trimmed span whitespace")
    require("[[" not in tts_text and "source=" not in tts_text,
            "directive/source suffix leaked into TTS submits")
    events = record["events"]
    event_counts = {}
    for event in events:
        event_counts[event["type"]] = event_counts.get(event["type"], 0) + 1
    report = {
        "format": REPORT_FORMAT, "result": "CONTROLLED_LIVE_PASS", "physical_gpu": 1,
        "base_commit": args.base_commit, "execution_commit": args.execution_commit,
        "request_sha256": sha256_bytes(canonical(body)),
        "public_request_id": record["request_id"], "public_event_count": len(events),
        "public_event_counts": event_counts, "public_artifact_count": len(artifacts),
        "public_artifact_chunk_count": event_counts.get("artifact", 0),
        "raw_sse": {key: record["sse_replay"][key] for key in
                    ("raw_ref", "raw_sha256", "raw_bytes", "event_count", "done_frame")},
        "raw_transport_chunks": len(record["sse_replay"]["chunks"]),
        "raw_frames": len(record["sse_replay"]["frames"]),
        "source": image, "video": video, "audio": pcm,
        "clean_text": record["clean_text"], "clean_text_sha256": record["clean_text_sha256"],
        "visual_slots": record["visual_slots"], "terminal_legs": record["terminal_legs"],
        "source_final": source_final, "audio_stream_indices": audio_streams,
        "video_submit": video_bind, "tts_submits": tts_tap.records,
        "tts_output": tts_output_tap.records, "tts_legs": tts_joins,
        "first_frame_mae": first_mae, "motion_mae": motion_mae,
        "first_frame_max_mae": FIRST_FRAME_MAX_MAE, "motion_min_mae": MOTION_MIN_MAE,
        "metric_version": METRIC_VERSION, "decoded_rgb": rgb,
        "video_stream": probe_video(args.ffprobe, video_path),
        "provenance": provenance, "models": models, "gpu1": gpu,
        "stage_lifecycle": stage_record, "manager": manager_record,
        "health_final": health_final,
        "trajectory": "capture/trajectories.jsonl", "gpu_monitor": "gpu1-monitor.json",
        "native_video_log": "video.log", "native_tts_log": "tts.log",
        "server_trace": "server.trace.jsonl",
        "perceptual_quality": "NOT_RUN", "generalization": "NOT_COVERED",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    verify = subprocess.run([
        sys.executable, "-I", str(args.verifier), str(out), "--out",
        str(out / "independent-verification.json"), "--ffmpeg", args.ffmpeg,
        "--ffprobe", args.ffprobe], capture_output=True, text=True, timeout=600)
    require(verify.returncode == 0,
            f"fresh-process verification failed:\n{verify.stdout}\n{verify.stderr}")
    summary = {key: report[key] for key in
               ("format", "result", "public_event_count", "public_artifact_count",
                "public_artifact_chunk_count", "raw_sse", "source", "video",
                "first_frame_mae", "motion_mae", "perceptual_quality", "generalization")}
    summary["independent_verification"] = "PASS"
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    files = {}
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"
                       and "run" not in p.relative_to(out).parts):
        files[path.relative_to(out).as_posix()] = sha256_file(path)
    (out / "SHA256SUMS").write_text("".join(
        f"{item['sha256']}  {name}\n" for name, item in files.items()))
    print(json.dumps(summary, sort_keys=True))
    print("PUBLIC_NARRATIVE_I2V: PASS")
    print("PERCEPTUAL_QUALITY: NOT_RUN")


def capture_public_with_id(port, body, recorder, source_ready, request_ready, id_out):
    # Keep the transport loop identical to capture_public while handing the real
    # public id to the output manager before any terminal join can finish.
    params, option_inputs = request_parameter_references(body, recorder.artifact_dir)
    inputs = input_references(body["messages"], recorder.artifact_dir) + option_inputs
    capture = RawSSECapture(clock=PerfCounterClock(origin_ns=clock_ns()))
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    inspected = 0
    source_final = None
    audio_streams = set()
    request_id = None
    try:
        conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
                     headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
        response = conn.getresponse()
        if response.status != 200:
            error_body = response.read()
            raise AssertionError(f"public chat HTTP {response.status}: {error_body!r}")
        require(response.getheader("Content-Type", "").startswith("text/event-stream"),
                "public response is not SSE")
        while True:
            chunk = response.read1(65536)
            if not chunk:
                break
            capture.feed(chunk)
            payloads = capture.payload_frames
            for frame_index, payload in payloads[inspected:]:
                if payload == "[DONE]":
                    continue
                frame = json.loads(payload)
                request_id = request_id or frame.get("id")
                if request_id is not None:
                    id_out[0] = request_id
                    request_ready.set()
                unified = frame.get("unified") or {}
                if unified.get("kind") == "artifact":
                    desc = unified["artifact"]
                    if desc.get("media_type") == "audio":
                        audio_streams.add(desc.get("stream_index"))
                    if (desc.get("media_type") == "image" and desc.get("slot_id") == "slot-0"
                            and desc.get("index") + 1 == desc.get("total")):
                        source_final = {"frame_index": frame_index,
                                        "observed_monotonic_ns": capture._frames[frame_index]["tick"],
                                        "observed_absolute_monotonic_ns": clock_ns(),
                                        "seq": unified.get("seq"), "sha256": desc.get("sha256")}
                        source_ready.set()
            inspected = len(payloads)
        capture.finish()
        details = {}
        request_id, events, output_text, state, error = parse_sse_stream(
            io.BytesIO(capture.raw_bytes), artifact_dir=recorder.artifact_dir,
            details=details, capture=capture)
        record = recorder.record(
            session_id="public-narrative-i2v-65", request_id=request_id,
            endpoint="/v1/chat/completions", inputs=inputs, params=params,
            events=events, terminal_state=state, error=error, output_text=output_text,
            artifacts=details.get("artifacts", []), capture=capture)
        return record, source_final, sorted(audio_streams)
    finally:
        conn.close()


def exercise_pcm_join():
    import msgpack
    chunks = [b"\x01\x00\x02\x00\x03\x00", b"\xff\xff\x00\x00"]
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"".join(chunks))
    wav_raw = wav_io.getvalue()
    with tempfile.TemporaryDirectory(prefix="umm-public-narrative-pcm-") as tmp:
        root = Path(tmp)
        outputs = []
        for index, raw in enumerate(chunks):
            message = {"type": "stream", "request_id": "http-1:tts:0",
                       "chunk": {"audio": raw, "index": index,
                                 "sample_rate": 24000, "channels": 1}}
            packed = msgpack.packb(message, use_bin_type=True)
            name = f"native-{index}.msgpack"
            (root / name).write_bytes(packed)
            outputs.append({"message_type": "stream", "request_id": "http-1:tts:0",
                            "stream_index": index, "audio_sha256": sha256_bytes(raw),
                            "packed_file": name})
        complete = {"type": "complete", "request_id": "http-1:tts:0", "success": True,
                    "result": {"streamed": True, "data": wav_raw}}
        packed = msgpack.packb(complete, use_bin_type=True)
        (root / "native-complete.msgpack").write_bytes(packed)
        outputs.append({"message_type": "complete", "request_id": "http-1:tts:0",
                        "success": True, "streamed": True,
                        "wav_sha256": sha256_bytes(wav_raw),
                        "packed_file": "native-complete.msgpack"})
        public = [({"stream_index": index, "sha256": sha256_bytes(raw),
                    "ref": f"chunk-{index}.pcm"}, raw)
                  for index, raw in enumerate(chunks)]
        join = join_tts_pcm([{"request_id": "http-1:tts:0"}], outputs, public, root)
        require(len(join) == 1 and join[0]["chunk_count"] == 2
                and join[0]["pcm_sha256"] == sha256_bytes(b"".join(chunks)),
                "deterministic exact PCM join failed")
        tampered = list(public)
        tampered[1] = (tampered[1][0], b"\x00\x00\x00\x00")
        for invalid in (tampered, public + [({"stream_index": 0,
                                              "sha256": "0" * 64,
                                              "ref": "extra.pcm"}, b"\x00\x00")]):
            try:
                join_tts_pcm([{"request_id": "http-1:tts:0"}], outputs, invalid, root)
            except AssertionError:
                pass
            else:
                raise AssertionError("invalid PCM mapping was not rejected")
    print("OFFLINE_NATIVE_PUBLIC_WAV_JOIN: PASS (2 chunks; tamper/extra rejected)")


def self_check():
    body = request_body("qwen3-native")
    require(body["narrative"]["continuity_mode"] == "i2v_chain", "continuity not opted in")
    require(body["audio"]["stream"] is True and "tts" not in body, "invalid speech fixture")
    require(VIDEO_DIRECTIVE.endswith("| source=slot-0]]"), "explicit source missing")
    require(FIRST_FRAME_MAX_MAE == 48.0 and MOTION_MIN_MAE == 1.0,
            "frozen thresholds changed")
    require(DEFAULT_LIBRARY_PATH == "", "portable loader default must be empty")
    require(free_port() != 43301, "unexpected resident port allocation")
    try:
        exercise_pcm_join()
    except ModuleNotFoundError as error:
        require(error.name == "msgpack", f"unexpected self-check dependency failure: {error}")
        print("OFFLINE_NATIVE_PUBLIC_WAV_JOIN: NOT_RUN (msgpack unavailable)")
    print("OFFLINE_PUBLIC_NARRATIVE: PASS (request shape, speech opt-in, frozen metrics)")
    print("REAL_GENERATION: NOT_RUN")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--out-dir", default="/tmp/umminfer-public-narrative-65-evidence")
    parser.add_argument("--server-binary", type=Path)
    parser.add_argument("--vlm-binary", type=Path)
    parser.add_argument("--tts-binary", type=Path)
    parser.add_argument("--wan-binary", type=Path)
    parser.add_argument("--flux-stage", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--vlm-model", type=Path)
    parser.add_argument("--flux-model", type=Path)
    parser.add_argument("--tts-model", type=Path)
    parser.add_argument("--wan-model", type=Path)
    parser.add_argument("--model", default="qwen3-native")
    parser.add_argument("--verifier", type=Path,
                        default=Path(__file__).with_name("public_narrative_i2v_verify.py"))
    parser.add_argument("--base-commit")
    parser.add_argument("--execution-commit")
    parser.add_argument("--port", type=int)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--library-path", default=DEFAULT_LIBRARY_PATH,
                        help="optional loader prefix; default inherits environment/system loader")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    for name in ("server_binary", "vlm_binary", "tts_binary", "wan_binary", "flux_stage",
                 "vlm_model", "flux_model", "tts_model", "wan_model"):
        if getattr(args, name) is None:
            parser.error(f"live mode requires --{name.replace('_', '-')}")
    live(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
