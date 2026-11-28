#!/usr/bin/env python3
"""Opt-in six-lane native-model soak on one task-owned physical GPU 0.

This driver intentionally has no offline/mock live substitute.  The only PASS
verdict is for real, externally retained evidence that passes the independent
fresh-process verifier.  ``--duration-s`` below 1800 is useful only for failure
injection/diagnostics and can never produce ISOLATED_PASS.

Required live inputs are explicit; no model, binary, input, or loader location
has an absolute default.  The caller must set UMMINFER_LIVE_OPT_IN=1.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import hashlib
import http.client
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.parse
import wave

ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "benchmarks/north-star/ns07-six-way-preregistration.json"
VERIFIER = Path(__file__).with_name("ns07_six_way_soak_verify.py")
LANES = ("text_code", "interleaved_visual", "full_duplex_voice",
         "natural_video_input", "abort_retry", "malformed_recovery")
OUTPUT_STAGES = ("flux", "tts", "wan")
TERMINALS = {"complete", "cancelled", "error"}
SCHEMA = "umminfer-ns07-six-way-soak/v1"
NORMALIZED_SCHEMA = "umminfer-ns07-arm/v1"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 << 20), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": Path(path).stat().st_size}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                    ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The process, tap, identity, and clock implementations are reused, not copied.
public_base = load_module("umminfer_ns07_public_base",
                          ROOT / "tests/public_narrative_i2v_live.py")
realtime_base = load_module("umminfer_ns07_realtime_base",
                            ROOT / "tests/realtime_network_live.py")


class UncappedJsonl:
    """Every record is retained; unlike soak_native.RollingJsonl nothing rolls."""
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.count = 0

    def write(self, value):
        raw = (json.dumps(value, sort_keys=True, ensure_ascii=False,
                          allow_nan=False) + "\n").encode()
        with self.lock:
            with self.path.open("ab", buffering=0) as output:
                output.write(raw)
            self.count += 1


def free_loopback_ports():
    sockets = []
    try:
        ports = []
        for _ in range(2):
            sock = socket.socket()
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
            ports.append(sock.getsockname()[1])
        require(len(set(ports)) == 2 and 43301 not in ports,
                f"unsafe ephemeral port allocation: {ports}")
        return ports
    finally:
        for sock in sockets:
            sock.close()


def gpu0_state():
    row = subprocess.check_output([
        "nvidia-smi", "-i", "0", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, timeout=20).strip()
    fields = [item.strip() for item in row.split(",")]
    require(len(fields) == 5 and int(fields[0]) == 0, f"unexpected GPU 0 row: {row!r}")
    compute = subprocess.check_output([
        "nvidia-smi", "-i", "0", "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits"], text=True, timeout=20).strip()
    processes = []
    for line in compute.splitlines():
        pid, memory = [item.strip() for item in line.split(",")]
        processes.append({"pid": int(pid), "used_memory_mib": int(memory)})
    return {"index": 0, "name": fields[1], "memory_used_mib": int(fields[2]),
            "memory_total_mib": int(fields[3]), "utilization_percent": int(fields[4]),
            "compute_processes": processes}


def require_gpu0_idle():
    state = gpu0_state()
    require(state["memory_used_mib"] <= 64 and state["utilization_percent"] == 0
            and not state["compute_processes"], f"physical GPU 0 is not idle: {state}")
    return state


def wait_gpu0_idle(timeout=45):
    deadline = time.monotonic() + timeout
    latest = None
    while time.monotonic() < deadline:
        try:
            return require_gpu0_idle()
        except (AssertionError, subprocess.SubprocessError) as error:
            latest = str(error)
            time.sleep(0.5)
    raise AssertionError(f"GPU 0 did not return to idle: {latest}")


def process_rss_mib(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def http_json(port, path, body=None, timeout=30):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        payload = None if body is None else json.dumps(body).encode()
        connection.request("GET" if body is None else "POST", path, body=payload,
                           headers={"Content-Type": "application/json"} if body is not None else {})
        response = connection.getresponse()
        raw = response.read()
        try:
            parsed = json.loads(raw) if raw else {}
        except ValueError:
            parsed = {"raw_sha256": sha_bytes(raw), "raw_bytes": len(raw)}
        return response.status, parsed
    finally:
        connection.close()


def wait_ready(port, processes, timeout=600):
    deadline = time.monotonic() + timeout
    markers = {"asr": "ready kind=asr", "vlm": "ready kind=qwen38"}
    while time.monotonic() < deadline:
        for process in processes.values():
            require(process.process.poll() is None,
                    f"{process.name} exited during startup: {process.process.returncode}")
        if all(marker in processes[name].log_path.read_text(encoding="utf-8", errors="replace")
               for name, marker in markers.items()):
            try:
                status, body = http_json(port, "/ready", timeout=5)
                if status == 200 and body.get("status") == "ready":
                    return
            except (OSError, http.client.HTTPException):
                pass
        time.sleep(0.1)
    raise TimeoutError("task-owned ASR/VLM gateway did not become ready")


def wait_health_zero(port, timeout=120):
    deadline = time.monotonic() + timeout
    latest = None
    while time.monotonic() < deadline:
        try:
            status, latest = http_json(port, "/health", timeout=5)
            if status == 200 and latest.get("in_flight") == 0:
                return latest
        except (OSError, http.client.HTTPException):
            pass
        time.sleep(0.1)
    raise AssertionError(f"in_flight did not drain: {latest}")


def parse_sse_frame(frame):
    data = []
    for line in frame.split(b"\n"):
        if line.startswith(b"data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(b" ") else value)
    return b"\n".join(data) if data else None


def project_event(event):
    projected = json.loads(canonical(event))
    artifact = projected.get("artifact")
    if isinstance(artifact, dict) and isinstance(artifact.get("data"), str):
        raw = base64.b64decode(artifact["data"], validate=True)
        artifact["data_sha256"] = sha_bytes(raw)
        artifact["data_bytes"] = len(raw)
        del artifact["data"]
    return projected


def percentile(values, fraction=0.95):
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(fraction * len(ordered)) - 1)])


class OutputStageManager:
    """Central lease and process-lifetime arbiter for FLUX/TTS/Wan."""
    def __init__(self, args, evidence, endpoints, env, taps):
        self.args = args
        self.evidence = evidence
        self.endpoints = endpoints
        self.env = env
        self.taps = taps
        self.condition = threading.Condition()
        self.current_name = None
        self.current_process = None
        self.switching = False
        self.active = 0
        self.serial = 0
        self.history = []
        self.drain_history = []
        self.leases = []
        self.waiters = []
        self.closed = False

    def _record_process(self, process, stage, expected_stop):
        record = process.record()
        record.update({"stage": stage, "expected_stop": expected_stop,
                       "command": list(process.command),
                       "recorded_monotonic_ns": time.monotonic_ns()})
        self.history.append(record)

    def _tts_outstanding(self):
        taps = getattr(self, "taps", None)
        if not taps or len(taps) != 2 or taps[0] is None or taps[1] is None:
            return set()
        submitted = {row.get("request_id") for row in taps[0].records
                     if row.get("message_type") == "submit" and row.get("request_id")}
        terminal = {row.get("request_id") for row in taps[1].records
                    if row.get("message_type") == "complete" and row.get("request_id")}
        return submitted - terminal

    def _wait_tts_drained(self):
        """Do not kill a stage between its final PCM chunk and native terminal.

        RawTap appends output records only after the frame is forwarded to the
        Coordinator. A zero lease count therefore cannot prove process drain:
        narrative work may share a stage resident for another lane, and its last
        success terminal can still be in the native outbox.
        """
        taps = getattr(self, "taps", None)
        if not taps or len(taps) != 2 or taps[0] is None or taps[1] is None:
            return
        started_ns = time.monotonic_ns()
        deadline = time.monotonic() + self.args.request_timeout_s
        initial = self._tts_outstanding()
        outstanding = initial
        while outstanding:
            if time.monotonic() >= deadline:
                raise AssertionError("native TTS terminal drain exceeded the existing request timeout: "
                                     + ",".join(sorted(outstanding)))
            time.sleep(0.005)
            outstanding = self._tts_outstanding()
        self.drain_history.append({
            "started_monotonic_ns": started_ns,
            "ended_monotonic_ns": time.monotonic_ns(),
            "initial_outstanding_count": len(initial),
            "submitted_count": len({row.get("request_id") for row in taps[0].records
                                    if row.get("message_type") == "submit" and row.get("request_id")}),
        })

    def _command(self, stage):
        args = self.args
        models = args.model_map
        if stage == "flux":
            return [args.python, "-I", str(args.source / "python/umminfer_flux2_stage.py"),
                    "--name", "image", "--backend", "diffusers", "--model", models["flux"],
                    "--model-id", "black-forest-labs/FLUX.2-klein-4B", "--revision", "UNKNOWN",
                    "--gpu", "0", "--default-size", "512x512", "--default-steps", "4",
                    "--recv", self.endpoints["image-in"],
                    "--coordinator", self.endpoints["completion"],
                    "--abort", self.endpoints["abort"]], "ready kind=image name=image", self.endpoints["image-in"]
        if stage == "tts":
            return [args.tts, "--kind", "tts", "--name", "tts", "--model", models["tts"],
                    "--gpu", "0", "--recv-endpoint", self.endpoints["tts-in"],
                    "--coordinator-endpoint", self.endpoints["tts-output-in"],
                    "--abort-endpoint", self.endpoints["abort"], "--max-frames", "192"], \
                   "native stage tts ready kind=tts", self.endpoints["tts-in"]
        return [args.wan, "--kind", "video", "--name", "video", "--video-backend", "cuda",
                "--model", models["wan"], "--gpu", "0", "--size", "704x480", "--steps", "20",
                "--fps", "16", "--recv-endpoint", self.endpoints["video-in"],
                "--coordinator-endpoint", self.endpoints["completion"],
                "--abort-endpoint", self.endpoints["abort"]], \
               "native stage video ready kind=video", self.endpoints["video-in"]

    def _start(self, stage):
        command, marker, _ = self._command(stage)
        process = public_base.ManagedProcess(stage, command,
            self.evidence / "logs" / f"{stage}-{self.serial}.log", env=self.env)
        self.serial += 1
        try:
            process.wait_log(marker, self.args.startup_timeout_s)
        except Exception:
            process.stop(None)
            self._record_process(process, stage, False)
            raise
        self.current_process = process
        self.current_name = stage

    def acquire(self, stage, owner):
        require(stage in OUTPUT_STAGES and not self.closed, f"invalid/closed output lease: {stage}")
        waiter = {"stage": stage, "owner": owner}
        with self.condition:
            # Strict acquisition order still permits concurrent leases for one
            # resident stage, but a later same-stage request cannot leapfrog an
            # earlier switch. Without this queue the voice lane can reacquire TTS
            # forever and starve narrative Wan until the public request timeout.
            self.waiters.append(waiter)
            try:
                while True:
                    require(not self.closed, "output manager closed while waiting for lease")
                    first = self.waiters and self.waiters[0] is waiter
                    if first and not self.switching and (
                            self.current_name == stage or self.active == 0):
                        break
                    self.condition.wait(0.2)
                if self.current_name == stage and self.current_process.process.poll() is not None:
                    self.current_process.stop(None)
                    self._record_process(self.current_process, stage, False)
                    self.current_name = self.current_process = None
                if self.current_name is not None and self.current_name != stage:
                    # A different model may replace the resident stage only when no
                    # lease remains. Mark the entire stop/start interval as a transition
                    # so resource samples never mix retiring and ready topologies.
                    require(self.active == 0, "output stage switch attempted with active leases")
                    self.switching = True
                    process, previous = self.current_process, self.current_name
                    was_alive = process.process.poll() is None
                    if previous == "tts" and was_alive:
                        self._wait_tts_drained()
                    process.stop(self._command(previous)[2])
                    self._record_process(process, previous, was_alive)
                    self.current_name = self.current_process = None
                if self.current_name is None:
                    self.switching = True
                else:
                    self.active += 1
                    token = {"id": len(self.leases), "stage": stage, "owner": owner,
                             "start_monotonic_ns": time.monotonic_ns(), "end_monotonic_ns": None}
                    self.leases.append(token)
                    self.waiters.remove(waiter)
                    self.condition.notify_all()
                    return token
            except BaseException:
                self.waiters.remove(waiter)
                self.condition.notify_all()
                raise
        try:
            self._start(stage)
        except Exception:
            with self.condition:
                self.waiters.remove(waiter)
                self.switching = False
                self.condition.notify_all()
            raise
        with self.condition:
            self.switching = False
            self.active += 1
            token = {"id": len(self.leases), "stage": stage, "owner": owner,
                     "start_monotonic_ns": time.monotonic_ns(), "end_monotonic_ns": None}
            self.leases.append(token)
            self.waiters.remove(waiter)
            self.condition.notify_all()
            return token

    def release(self, token):
        with self.condition:
            require(token["end_monotonic_ns"] is None and self.active > 0,
                    "duplicate or unowned output-stage lease release")
            token["end_monotonic_ns"] = time.monotonic_ns()
            self.active -= 1
            self.condition.notify_all()

    @contextmanager
    def lease(self, stage, owner):
        token = self.acquire(stage, owner)
        try:
            yield token
        finally:
            self.release(token)

    def residency(self):
        with self.condition:
            return "asr+vlm" + ("+" + self.current_name if self.current_name else "")

    def close(self):
        forced_active_stop = False
        with self.condition:
            self.closed = True
            deadline = time.monotonic() + 120
            while self.active and time.monotonic() < deadline:
                self.condition.wait(0.2)
            # A wedged client must not leak a task-owned process.  Such a forced
            # cleanup is recorded as a crash/lease failure and can never PASS.
            forced_active_stop = self.active != 0
            self.switching = True
            process, stage = self.current_process, self.current_name
            self.current_process = self.current_name = None
            self.condition.notify_all()
        if process is not None:
            was_alive = process.process.poll() is None
            _, _, endpoint = self._command(stage)
            process.stop(endpoint)
            self._record_process(process, stage, was_alive and not forced_active_stop)


class RawTap:
    """Byte-preserving PUSH/PULL tap; cancelled TTS terminals need no WAV bytes.

    Semantic acceptance is intentionally left to the independent MessagePack
    verifier.  This transport retains every frame without imposing the existing
    narrative TapProxy's successful-WAV-only assumption on cancelled work.
    """
    def __init__(self, name, listen, destination, evidence_dir):
        import zmq
        self.name = name
        self.evidence_dir = Path(evidence_dir)
        self.records = []
        self.error = None
        self.stop_requested = False
        self.context = zmq.Context()
        self.pull = self.context.socket(zmq.PULL)
        self.pull.bind(listen)
        self.push = self.context.socket(zmq.PUSH)
        self.push.setsockopt(zmq.LINGER, 0)
        self.push.connect(destination)
        self.thread = threading.Thread(target=self._run, name=f"ns07-tap-{name}", daemon=True)
        self.thread.start()

    def _run(self):
        import msgpack
        import zmq
        try:
            while not self.stop_requested or self.pull.poll(0):
                try:
                    packed = self.pull.recv(zmq.NOBLOCK)
                except zmq.Again:
                    time.sleep(0.005)
                    continue
                message = msgpack.unpackb(packed, raw=False, strict_map_key=False)
                record = {"request_id": message.get("request_id"),
                          "message_type": message.get("type"),
                          "success": message.get("success"),
                          "observed_monotonic_ns": time.monotonic_ns(),
                          "packed_sha256": sha_bytes(packed), "packed_bytes": len(packed)}
                name = f"{self.name}-{len(self.records)}-{record['packed_sha256']}.msgpack"
                (self.evidence_dir / name).write_bytes(packed)
                record["packed_file"] = name
                while True:
                    try:
                        self.push.send(packed, zmq.NOBLOCK)
                        break
                    except zmq.Again:
                        time.sleep(0.005)
                record["forwarded_monotonic_ns"] = time.monotonic_ns()
                self.records.append(record)
        except Exception as error:
            self.error = str(error)

    def stop(self):
        self.stop_requested = True
        self.thread.join(10)
        require(not self.thread.is_alive(), f"{self.name} tap did not stop")
        self.pull.close(0)
        self.push.close(0)
        self.context.term()
        require(self.error is None, f"{self.name} tap failed: {self.error}")


class DelayedRawWebSocket(realtime_base.RawWebSocket):
    """A real socket read pause, not a post-processing timestamp invention."""
    delay_s = 1.0
    delayed_writer = None
    evidence_root = None

    def __init__(self, *args, **kwargs):
        self.did_pause = False
        super().__init__(*args, **kwargs)

    def record(self, direction, raw):
        row = super().record(direction, raw)
        if direction == "rx" and not self.did_pause and b'"audio.delta"' in raw:
            self.did_pause = True
            started = time.monotonic_ns()
            time.sleep(self.delay_s)
            ended = time.monotonic_ns()
            response = re.search(br'"response_id"\s*:\s*"([^"]+)"', raw)
            self.delayed_writer.write({
                "capture_file": str(self.capture_path.relative_to(self.evidence_root)),
                "response_id": response.group(1).decode() if response else None,
                "started_monotonic_ns": started, "ended_monotonic_ns": ended,
                "requested_delay_ns": int(self.delay_s * 1e9),
                "actual_delay_ns": ended - started,
                "first_audio_capture_index": row["index"],
            })
        return row


class Soak:
    def __init__(self, args):
        self.args = args
        self.out = args.evidence_dir.resolve()
        require(self.out != Path(self.out.anchor), "evidence directory cannot be a filesystem root")
        require(self.out != args.source and args.source not in self.out.parents
                and self.out not in args.source.parents,
                "raw evidence must remain outside and disjoint from the source repository")
        if self.out.exists():
            require(self.out.is_dir() and not any(self.out.iterdir()),
                    "evidence directory must be absent or empty")
        self.out.mkdir(parents=True, exist_ok=True)
        for name in ("events", "media", "logs", "requests", "frames", "taps", "run"):
            (self.out / name).mkdir()
        self.requests = UncappedJsonl(self.out / "requests.jsonl")
        self.events = UncappedJsonl(self.out / "events.jsonl")
        self.resources = UncappedJsonl(self.out / "resources.jsonl")
        self.acks = UncappedJsonl(self.out / "playback-acks.jsonl")
        self.delayed = UncappedJsonl(self.out / "delayed-read.jsonl")
        self.counter = 0
        self.counter_lock = threading.Lock()
        self.stop = threading.Event()
        self.stop_resource_sampling = threading.Event()
        self.processes = {}
        self.process_history = []
        self.process_lock = threading.Lock()
        self.stages = None
        self.resource_thread = None
        self.threads = []
        self.voice_pcm = self.read_wav(args.voice_wav)
        self.barge_pcm = self.read_wav(args.barge_in_wav)
        self.video_raw = args.natural_video.read_bytes()
        require(0 < len(self.video_raw) <= 32 << 20, "natural video must be 1..32 MiB")
        mime = {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime"}.get(
            args.natural_video.suffix.lower())
        require(mime is not None, "natural video must be .mp4, .webm, or .mov")
        self.video_url = f"data:{mime};base64," + base64.b64encode(self.video_raw).decode()
        self.tts_tap = self.tts_output_tap = None
        self.driver_error = None
        self.window_started_ns = None
        self.window_ended_ns = None
        self.baseline_resource = None
        self.final_same_resource = None
        self.gpu_initial = None
        self.gpu_final = None
        self.final_health = None
        self.ports = None

    @staticmethod
    def read_wav(path):
        with wave.open(str(path), "rb") as source:
            require((source.getnchannels(), source.getsampwidth(), source.getframerate()) ==
                    (1, 2, 16000), f"{path} must be PCM16 mono 16 kHz")
            raw = source.readframes(source.getnframes())
        require(len(raw) >= 64000, f"{path} needs at least two seconds of PCM")
        return raw

    def key(self, lane):
        with self.counter_lock:
            self.counter += 1
            return f"{lane}-{self.counter:07d}"

    def _new_process(self, name, command, log_name):
        process = public_base.ManagedProcess(name, command, self.out / "logs" / log_name,
                                            env=self.args.env)
        self.processes[name] = process
        return process

    def setup_stack(self):
        args = self.args
        self.gpu_initial = require_gpu0_idle()
        self.ports = free_loopback_ports()
        ipc = self.out / "run/ipc"
        shm = self.out / "run/shm"
        ipc.mkdir()
        shm.mkdir()
        require(all(len(str(path)) < 96 for path in (ipc, shm)), "IPC evidence path is too long")
        names = ("asr", "vlm", "image-in", "tts-tap-in", "tts-in", "tts-output-in",
                 "video-in", "completion", "abort")
        endpoints = {name: f"ipc://{ipc}/{name}" for name in names}
        self.endpoints = endpoints
        (self.out / "taps/tts").mkdir()
        (self.out / "taps/tts-output").mkdir()
        self.tts_tap = RawTap("tts", endpoints["tts-tap-in"],
                              endpoints["tts-in"], self.out / "taps/tts")
        self.tts_output_tap = RawTap("tts-output", endpoints["tts-output-in"],
                                     endpoints["completion"], self.out / "taps/tts-output")
        self._new_process("server", [args.server, "serve", "--host", "127.0.0.1",
            "--port", str(self.ports[0]), "--realtime-port", str(self.ports[1]),
            "--model", "qwen3-unified", "--entry", "asr", "--entry-endpoint", endpoints["asr"],
            "--vlm-entry", "vlm", "--vlm-entry-endpoint", endpoints["vlm"],
            "--image-entry", "image", "--image-entry-endpoint", endpoints["image-in"],
            "--tts-entry", "tts", "--tts-entry-endpoint", endpoints["tts-tap-in"],
            "--video-entry", "video", "--video-entry-endpoint", endpoints["video-in"],
            "--completion", endpoints["completion"], "--abort", endpoints["abort"],
            "--session-history", "--max-in-flight", "8", "--ipc-dir", str(shm)], "server.log")
        self._new_process("asr", [args.asr, "--kind", "asr", "--name", "asr",
            "--model", args.model_map["asr"], "--gpu", "0", "--recv-endpoint", endpoints["asr"],
            "--coordinator-endpoint", endpoints["completion"], "--abort-endpoint", endpoints["abort"]], "asr.log")
        self._new_process("vlm", [args.vlm, "--kind", "qwen38", "--name", "vlm",
            "--model", args.model_map["vlm"], "--gpu", "0", "--max-new-tokens", "256",
            "--batch-size", "1", "--recv-endpoint", endpoints["vlm"],
            "--coordinator-endpoint", endpoints["completion"], "--abort-endpoint", endpoints["abort"]], "vlm.log")
        wait_ready(self.ports[0], self.processes, args.startup_timeout_s)
        self.stages = OutputStageManager(args, self.out, endpoints, args.env,
                                         (self.tts_tap, self.tts_output_tap))
        self.warmup_base()
        self.baseline_resource = self.sample_resource("baseline")
        self.resource_thread = threading.Thread(target=self.resource_loop,
                                                name="gpu0-resource-sampler", daemon=True)
        self.resource_thread.start()

    def warmup_base(self):
        """Populate fixed base-stage workspaces before the zero-growth baseline."""
        self.request("text_code", "text", {"model": "qwen3-unified", "stream": True,
            "max_tokens": 16, "temperature": 0, "seed": 70080, "session_id": "ns07-warmup-text",
            "messages": [{"role": "user", "content": "Reply with one short warmup sentence."}]},
            warmup=True)
        self.request("natural_video_input", "natural_video", {"model": "qwen3-unified",
            "stream": True, "max_tokens": 16, "temperature": 0, "seed": 70080,
            "session_id": "ns07-warmup-video",
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": self.video_url}},
                {"type": "text", "text": "Name one visible warmup detail."}]}]},
            timeout=self.args.request_timeout_s, warmup=True)
        body, content_type = self.multipart_wav(self.args.voice_wav.read_bytes())
        self.request("malformed_recovery", "valid_audio_recovery", body,
            path="/v1/audio/transcriptions", expected="transcription", content_type=content_type,
            warmup=True)
        wait_health_zero(self.ports[0], 120)

    def sample_resource(self, phase):
        state = gpu0_state()
        rows = {}
        stage_process = stage_name = None
        stage_transition = False
        if self.stages:
            with self.stages.condition:
                stage_transition = self.stages.switching
                stage_process = self.stages.current_process
                stage_name = self.stages.current_name
                residency = self.stages.residency()
        else:
            residency = "asr+vlm"
        for name, process in self.processes.items():
            rows[name] = {"pid": process.process.pid,
                          "rss_mib": process_rss_mib(process.process.pid),
                          "returncode": process.process.poll()}
        if stage_process is not None:
            rows[stage_name] = {"pid": stage_process.process.pid,
                "rss_mib": process_rss_mib(stage_process.process.pid),
                "returncode": stage_process.process.poll()}
        row = {"monotonic_ns": time.monotonic_ns(), "phase": phase,
               "stage_transition": stage_transition,
               "residency": residency,
               "gpu0": state, "processes": rows}
        self.resources.write(row)
        return row

    def resource_loop(self):
        while not self.stop_resource_sampling.wait(1.0):
            try:
                self.sample_resource("measured")
            except Exception as error:
                self.resources.write({"monotonic_ns": time.monotonic_ns(),
                                      "phase": "sampler_error", "error": str(error)})

    def _artifacts(self, events, request_key):
        groups = {}
        for item in events:
            unified = item["event"]
            if unified.get("kind") != "artifact":
                continue
            artifact = unified.get("artifact") or {}
            groups.setdefault(artifact.get("artifact_id"), []).append(artifact)
        output = []
        directory = self.out / "media" / request_key
        directory.mkdir(parents=True, exist_ok=True)
        for artifact_id, chunks in groups.items():
            try:
                ordered = sorted(chunks, key=lambda item: item["index"])
                raw = b"".join(base64.b64decode(item["data"], validate=True) for item in ordered)
                final = ordered[-1]
                safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(artifact_id))
                media = directory / (safe + ".bin")
                media.write_bytes(raw)
                output.append({"artifact_id": artifact_id, "media_file": str(media.relative_to(self.out)),
                    "bytes": len(raw), "sha256": sha_bytes(raw),
                    "reported_sha256": final.get("sha256"), "reported_bytes": final.get("bytes"),
                    "media_type": final.get("media_type"), "mime_type": final.get("mime_type"),
                    "slot_id": final.get("slot_id"), "slot_index": final.get("slot_index"),
                    "stream_index": final.get("stream_index"), "sample_rate": final.get("sample_rate"),
                    "channels": final.get("channels"), "chunks": len(ordered)})
            except Exception as error:
                output.append({"artifact_id": artifact_id, "error": str(error), "chunks": len(chunks)})
        return output

    def request(self, lane, kind, body, *, path="/v1/chat/completions", session_id=None,
                expected="complete", expected_fault=False, recovery=False, on_event=None,
                timeout=900, content_type="application/json", warmup=False):
        key = self.key(lane)
        if session_id is None and isinstance(body, dict):
            session_id = body.get("session_id")
        started = time.monotonic_ns()
        suffix = ".sse" if path == "/v1/chat/completions" else ".raw"
        raw_path = self.out / "requests" / (key + suffix)
        request_path = self.out / "requests" / (key + ".input")
        events = []
        headers = {}
        status = 0
        failure = None
        terminal_kinds = []
        done_count = 0
        raw_sha = hashlib.sha256()
        raw_bytes = 0
        connection = None
        try:
            payload = body if isinstance(body, bytes) else json.dumps(body, separators=(",", ":")).encode()
            request_path.write_bytes(payload)
            connection = http.client.HTTPConnection("127.0.0.1", self.ports[0], timeout=timeout)
            connection.request("POST", path, body=payload,
                               headers={"Content-Type": content_type, "Accept": "text/event-stream"})
            response = connection.getresponse()
            status = response.status
            headers = {name.lower(): value for name, value in response.getheaders()}
            with raw_path.open("wb") as raw_output:
                if headers.get("content-type", "").startswith("text/event-stream"):
                    buffer = bytearray()
                    while True:
                        chunk = response.read1(65536)
                        if not chunk:
                            break
                        raw_output.write(chunk)
                        raw_sha.update(chunk)
                        raw_bytes += len(chunk)
                        buffer.extend(chunk)
                        while b"\n\n" in buffer:
                            frame, _, rest = bytes(buffer).partition(b"\n\n")
                            buffer[:] = rest
                            data = parse_sse_frame(frame)
                            if data is None:
                                continue
                            if data == b"[DONE]":
                                done_count += 1
                                continue
                            outer = json.loads(data)
                            unified = outer.get("unified")
                            if not isinstance(unified, dict):
                                continue
                            item = {"request_key": key, "lane": lane,
                                    "observed_monotonic_ns": time.monotonic_ns(),
                                    "outer_id": outer.get("id"), "event": unified}
                            events.append(item)
                            self.events.write({"request_key": key, "lane": lane,
                                "observed_monotonic_ns": item["observed_monotonic_ns"],
                                "outer_id": outer.get("id"), "event": project_event(unified)})
                            if unified.get("kind") in TERMINALS:
                                terminal_kinds.append(unified.get("kind"))
                            if on_event is not None:
                                on_event(outer, unified)
                    if buffer:
                        raise AssertionError("truncated final SSE frame")
                else:
                    while True:
                        chunk = response.read(65536)
                        if not chunk:
                            break
                        raw_output.write(chunk)
                        raw_sha.update(chunk)
                        raw_bytes += len(chunk)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        finally:
            if connection is not None:
                connection.close()
        ended = time.monotonic_ns()
        artifacts = self._artifacts(events, key)
        terminal = terminal_kinds[-1] if terminal_kinds else None
        ok = False
        record_status = "error"
        if expected in ("complete", "cancelled"):
            wanted = expected
            ok = (status == 200 and done_count == 1 and terminal_kinds == [wanted]
                  and not failure)
            record_status = "completed" if terminal == "complete" else "cancelled" if terminal == "cancelled" else "error"
        elif expected == "error":
            ok = status >= 400 and raw_bytes > 0 and not failure
            record_status = "expected_error" if ok else "error"
        elif expected == "transcription":
            try:
                text = json.loads(raw_path.read_text()).get("text", "") if raw_path.exists() else ""
            except ValueError:
                text = ""
            ok = status == 200 and bool(text.strip()) and not failure
            record_status = "completed" if ok else "error"
        elif expected.startswith("binary:"):
            mime = expected.split(":", 1)[1]
            ok = (status == 200 and headers.get("content-type", "").split(";", 1)[0] == mime
                  and raw_bytes > 0 and not failure)
            record_status = "completed" if ok else "error"
            if ok:
                media = self.out / "media" / (key + ".bin")
                shutil.copyfile(raw_path, media)
                artifacts.append({"artifact_id": key, "media_file": str(media.relative_to(self.out)),
                                  "bytes": raw_bytes, "sha256": raw_sha.hexdigest(),
                                  "media_type": "video" if mime == "video/mp4" else "image",
                                  "mime_type": mime, "chunks": 1})
        unexpected = not ok and not (expected_fault and record_status in ("cancelled", "expected_error"))
        actual_request_id = next((item["outer_id"] for item in events if item["outer_id"]), None)
        record = {"schema": "umminfer-ns07-request/v1", "request_key": key, "lane": lane,
            "kind": kind, "transport": "http", "path": path, "session_id": session_id,
            "request_id": actual_request_id, "http_status": status, "http_headers": headers,
            "started_monotonic_ns": started, "ended_monotonic_ns": ended,
            "latency_ms": (ended - started) / 1e6, "raw_file": str(raw_path.relative_to(self.out)),
            "request_file": str(request_path.relative_to(self.out)),
            "request_sha256": sha_file(request_path)["sha256"] if request_path.exists() else None,
            "raw_sha256": raw_sha.hexdigest(), "raw_bytes": raw_bytes, "terminal": terminal,
            "terminal_count": len(terminal_kinds), "done_count": done_count,
            "status": record_status, "ok": ok, "expected_fault": expected_fault, "warmup": warmup,
            "unexpected_error": unexpected, "recovery_attempt": recovery,
            "recovery_success": bool(recovery and ok), "failure": failure,
            "artifacts": artifacts}
        self.requests.write(record)
        return record

    def abort_request(self, request_id, session_id):
        require(request_id, "cannot abort a request without its public id")
        status, reply = http_json(self.ports[0], "/abort", {
            "request_id": request_id, "session_id": session_id}, timeout=10)
        require(status == 200 and reply.get("success") is True,
                f"abort rejected: {status} {reply}")

    def realtime_record(self, lane, client, response_id, after, terminal, started_ns,
                        recovery=False):
        snapshot = client.response_snapshot(response_id, after, wait_terminal=False)
        ended = time.monotonic_ns()
        safe = self.key(lane)
        pcm_path = self.out / "media" / (safe + ".pcm")
        pcm_path.write_bytes(snapshot["pcm"])
        terminal_type = terminal["event"]["type"] if terminal else None
        ok = terminal_type in ("response.completed", "response.cancelled") and bool(snapshot["text"])
        record = {"schema": "umminfer-ns07-request/v1", "request_key": safe, "lane": lane,
            "kind": "realtime_voice", "transport": "websocket", "session_id": client.name,
            "request_id": response_id, "response_id": response_id,
            "capture_file": str(client.ws.capture_path.relative_to(self.out)),
            "started_monotonic_ns": started_ns, "ended_monotonic_ns": ended,
            "latency_ms": (ended - started_ns) / 1e6, "terminal": terminal_type,
            "terminal_count": 1 if terminal_type else 0,
            "status": "completed" if terminal_type == "response.completed" else
                      "cancelled" if terminal_type == "response.cancelled" else "error",
            "ok": ok, "expected_fault": terminal_type == "response.cancelled",
            "unexpected_error": not ok, "recovery_attempt": recovery,
            "recovery_success": bool(recovery and terminal_type == "response.completed"),
            "text_sha256": sha_bytes(snapshot["text"].encode()),
            "pcm_file": str(pcm_path.relative_to(self.out)), "pcm_bytes": len(snapshot["pcm"]),
            "pcm_sha256": sha_bytes(snapshot["pcm"]), "artifacts": []}
        self.requests.write(record)
        return snapshot, record

    def consume_and_ack(self, client, snapshot, cursor, final):
        response_id = snapshot["response_id"]
        audio = snapshot["audio"]
        raw = b"".join(base64.b64decode(item["event"]["audio"], validate=True) for item in audio)
        rates = {int(item["event"]["sample_rate"]) for item in audio}
        channels = {int(item["event"]["channels"]) for item in audio}
        require(rates == {24000} and channels == {1} and raw and len(raw) % 2 == 0,
                "invalid realtime PCM for virtual consumption")
        started = time.monotonic_ns()
        for item in audio:
            pcm = base64.b64decode(item["event"]["audio"], validate=True)
            time.sleep(len(pcm) / (2.0 * 24000))
        ended = time.monotonic_ns()
        marker = client.ws.mark()
        client.ws.send_json({"type": "output_audio_buffer.playback_ack",
                             "response_id": response_id, "text_cursor": cursor})
        accepted = client.ws.wait_for(lambda event: event.get("type") == "response.playback_ack"
            and event.get("response_id") == response_id and event.get("text_cursor") == cursor
            and bool(event.get("final", False)) == final, marker, 15, "accepted playback ACK")
        self.acks.write({"capture_file": str(client.ws.capture_path.relative_to(self.out)),
            "response_id": response_id, "text_cursor": cursor, "final": final,
            "audio_capture_indices": [item["capture_index"] for item in audio],
            "pcm_bytes": len(raw), "pcm_sha256": sha_bytes(raw),
            "sample_rate": 24000, "channels": 1,
            "consumption_started_monotonic_ns": started,
            "consumption_ended_monotonic_ns": ended,
            "sent_monotonic_ns": ended, "accepted_monotonic_ns": accepted["arrival_ns"],
            "minimum_consumption_ns": int(len(raw) / 48000.0 * 1e9)})

    def lane_text_code(self, rng):
        prompts = ("Prove that there are infinitely many primes in four concise lines.",
                   "Write a tiny Python function that reverses a UTF-8 string safely.",
                   "Explain a balanced binary tree in two short sentences.")
        index = 0
        while not self.stop.is_set():
            self.request("text_code", "text", {"model": "qwen3-unified", "stream": True,
                "max_tokens": 64, "temperature": 0, "seed": 70080 + index,
                "session_id": "ns07-text-code",
                "messages": [{"role": "user", "content": prompts[index % len(prompts)]}]})
            index += 1
            self.stop.wait(rng.random() * 0.05)

    def _trace_tts_complete(self, request_id):
        """Wait for the complete post-VLM TTS leg set, not an early subset.

        UMM_TRACE rotates at 64 KiB, much faster than a slow native TTS leg set.
        Remember the expected count across polls and read the one retained
        rotation before the current file so handoff never depends on both the
        outputs_started and final leg_settled rows remaining in one file.
        """
        states = getattr(self, "_tts_trace_states", None)
        if states is None:
            states = self._tts_trace_states = {}
        state = states.setdefault(request_id, {"expected": 0, "latest": []})
        path = self.out / "server.trace.jsonl"
        for candidate in (Path(str(path) + ".1"), path):
            if not candidate.exists():
                continue
            for line in candidate.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("event") != "legs_snapshot" or row.get("request_id") != request_id:
                    continue
                if row.get("phase") == "outputs_started":
                    state["expected"] = sum(
                        leg.get("stage") == "tts" for leg in row.get("legs", []))
                state["latest"] = row.get("legs", [])
        expected = state["expected"]
        latest = state["latest"]
        current = [leg for leg in latest if leg.get("stage") == "tts"]
        vlm = [leg for leg in latest if leg.get("stage") == "vlm"]
        return (expected > 0 and len(current) == expected and
                all(leg.get("state") == "completed" for leg in current) and
                all(leg.get("state") in ("completed", "failed", "aborted") for leg in vlm))

    def lane_interleaved_visual(self, rng):
        index = 0
        while not self.stop.is_set():
            owner = f"interleaved-{index}"
            current = self.stages.acquire("flux", owner)
            current_lock = threading.Lock()
            watcher = None
            watcher_stop = threading.Event()
            state = {"request_id": None, "tts_started": False, "wan_started": False,
                     "abort_sent": False}

            def release_current():
                with current_lock:
                    nonlocal current
                    if current is not None:
                        self.stages.release(current)
                        current = None

            def fail_current():
                release_current()
                if state.get("request_id") and not state["abort_sent"]:
                    state["abort_sent"] = True
                    self.abort_request(state["request_id"], "ns07-interleaved-visual")

            def watch_tts():
                try:
                    deadline = time.monotonic() + self.args.request_timeout_s
                    while not watcher_stop.is_set() and time.monotonic() < deadline:
                        request_id = state["request_id"]
                        if request_id and self._trace_tts_complete(request_id):
                            prefix = request_id + ":tts"
                            submits = {row["request_id"] for row in self.tts_tap.records
                                       if str(row.get("request_id", "")).startswith(prefix)}
                            completed = {row["request_id"] for row in self.tts_output_tap.records
                                         if row.get("message_type") == "complete" and row.get("success") is True}
                            if submits and submits <= completed:
                                with current_lock:
                                    nonlocal current
                                    if current is not None and current["stage"] == "tts":
                                        self.stages.release(current)
                                        current = self.stages.acquire("wan", owner)
                                        state["wan_started"] = True
                                return
                        watcher_stop.wait(0.02)
                except Exception as error:
                    self.events.write({"lane": "interleaved_visual", "kind": "harness_error",
                                       "error": str(error), "monotonic_ns": time.monotonic_ns()})
                    try:
                        fail_current()
                    except Exception as abort_error:
                        self.events.write({"lane": "interleaved_visual", "kind": "abort_error",
                                           "error": str(abort_error), "monotonic_ns": time.monotonic_ns()})
                    watcher_stop.set()

            def on_event(outer, unified):
                state["request_id"] = state["request_id"] or outer.get("id")
                artifact = unified.get("artifact") or {}
                if unified.get("kind") == "leg_error":
                    fail_current()
                    return
                if unified.get("kind") == "artifact" and artifact.get("media_type") == "image" \
                        and artifact.get("index") + 1 == artifact.get("total") and not state["tts_started"]:
                    release_current()
                    with current_lock:
                        nonlocal current
                        current = self.stages.acquire("tts", owner)
                    state["tts_started"] = True
                    watcher = threading.Thread(target=watch_tts, daemon=True)
                    watcher.start()
                    state["watcher"] = watcher
                if unified.get("kind") == "artifact" and artifact.get("media_type") == "video" \
                        and artifact.get("index") + 1 == artifact.get("total"):
                    release_current()
                if unified.get("kind") in TERMINALS:
                    release_current()

            try:
                # Reuse the accepted real public image -> explicit source=slot-0
                # I2V golden request instead of hoping an ambiguous prompt emits video.
                body = public_base.request_body("qwen3-unified")
                body["session_id"] = "ns07-interleaved-visual"
                self.request("interleaved_visual", "narrative", body,
                    on_event=on_event, timeout=self.args.request_timeout_s)
            finally:
                watcher_stop.set()
                release_current()
                watcher = state.get("watcher")
                if watcher is not None:
                    watcher.join(10)
            index += 1
            self.stop.wait(rng.random() * 0.1)

    def lane_full_duplex_voice(self, rng):
        old_ws = realtime_base.RawWebSocket
        DelayedRawWebSocket.delayed_writer = self.delayed
        DelayedRawWebSocket.evidence_root = self.out
        DelayedRawWebSocket.delay_s = 1.0
        realtime_base.RawWebSocket = DelayedRawWebSocket
        client = None
        try:
            client = realtime_base.LiveClient(0, 0, self.ports[1], self.out)
            iteration = 0
            while not self.stop.is_set():
                owner = f"full-duplex-{iteration}"
                with self.stages.lease("tts", owner):
                    turn_started = time.monotonic_ns()
                    turn = client.send_user(self.voice_pcm)
                    response_id, after = client.response_identity(turn["completed"]["event_index"] + 1)
                    # Wait long enough to distinguish a multi-part response from
                    # a valid single-cursor one. First audio alone is too early:
                    # a positive cursor may arrive in a later sentence. Terminal
                    # or audio.done selects the normal final-ACK path instead.
                    client.ws.wait_for(lambda event: event.get("response_id") == response_id
                        and ((event.get("type") == "audio.delta"
                              and int(event.get("text_cursor", 0)) > 0)
                             or event.get("type") in ("audio.done", "response.completed",
                                                      "response.failed")), after, 180,
                        "positive-cursor-or-terminal response decision")
                    snapshot = client.response_snapshot(response_id, after, wait_terminal=False)
                    if not snapshot["positive_cursors"]:
                        # A valid one-sentence response can expose only cursor 0.
                        # Complete/ACK it normally instead of inventing a positive
                        # cursor or forcing an error. Later multi-part turns still
                        # need to prove actual barge-in and recovery for the gate.
                        snapshot = client.response_snapshot(response_id, after, wait_terminal=True)
                        self.consume_and_ack(client, snapshot, len(snapshot["text"].encode()), True)
                        self.realtime_record("full_duplex_voice", client, response_id, after,
                                             snapshot["terminal"], turn_started, recovery=False)
                        iteration += 1
                        self.stop.wait(rng.random() * 0.05)
                        continue
                    cursor = snapshot["positive_cursors"][0]
                    self.consume_and_ack(client, snapshot, cursor, False)
                    next_turn = client.barge(snapshot, self.barge_pcm, cursor, immediate=False)
                    cancelled = next_turn["cancelled"]
                    self.realtime_record("full_duplex_voice", client, response_id, after,
                                         cancelled, turn_started, recovery=False)
                    next_started = time.monotonic_ns()
                    next_id, next_after = client.response_identity(
                        next_turn["completed"]["event_index"] + 1)
                    next_snapshot = client.response_snapshot(next_id, next_after, wait_terminal=True)
                    self.consume_and_ack(client, next_snapshot, len(next_snapshot["text"].encode()), True)
                    terminal = next_snapshot["terminal"]
                    self.realtime_record("full_duplex_voice", client, next_id, next_after,
                                         terminal, next_started, recovery=True)
                iteration += 1
                self.stop.wait(rng.random() * 0.05)
        finally:
            realtime_base.RawWebSocket = old_ws
            if client is not None:
                client.close()

    def lane_natural_video_input(self, rng):
        questions = ("Describe the visible action and camera changes in two concise sentences.",
                     "What changes between the opening and closing scenes? Answer briefly.")
        index = 0
        while not self.stop.is_set():
            self.request("natural_video_input", "natural_video", {"model": "qwen3-unified",
                "stream": True, "max_tokens": 96, "temperature": 0, "seed": 70080,
                "session_id": "ns07-natural-video",
                "messages": [{"role": "user", "content": [
                    {"type": "video_url", "video_url": {"url": self.video_url}},
                    {"type": "text", "text": questions[index % 2]}]}]},
                timeout=self.args.request_timeout_s)
            index += 1
            self.stop.wait(rng.random() * 0.05)

    def lane_abort_retry(self, rng):
        index = 0
        while not self.stop.is_set():
            session = "ns07-abort-retry"
            aborted = {"sent": False}

            def on_event(outer, unified):
                if unified.get("kind") == "vlm_delta" and not aborted["sent"]:
                    aborted["sent"] = True
                    self.stop.wait(rng.random() * 0.03)
                    self.abort_request(outer.get("id") or unified.get("request_id"), session)

            cancelled = self.request("abort_retry", "random_abort", {"model": "qwen3-unified",
                "stream": True, "max_tokens": 160, "temperature": 0, "seed": 70080 + index,
                "session_id": session, "messages": [{"role": "user", "content":
                    "Give a detailed eight-part explanation of the solar system with one paragraph per planet."}]},
                expected="cancelled", expected_fault=True, on_event=on_event)
            retried = self.request("abort_retry", "abort_valid_retry", {"model": "qwen3-unified",
                "stream": True, "max_tokens": 32, "temperature": 0, "seed": 70080 + index,
                "session_id": session, "messages": [{"role": "user", "content":
                    "Reply with exactly: recovery ok"}]}, recovery=True)
            if not cancelled["ok"] or not retried["ok"]:
                self.stop.wait(0.5)
            index += 1

    def multipart_wav(self, raw, filename="input.wav"):
        boundary = "----ns07-six-way"
        body = bytearray()
        for name, value in (("model", "qwen3-unified"),):
            body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").encode()
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
                 "Content-Type: audio/wav\r\n\r\n").encode() + raw
        body += f"\r\n--{boundary}--\r\n".encode()
        return bytes(body), f"multipart/form-data; boundary={boundary}"

    def lane_malformed_recovery(self, rng):
        index = 0
        while not self.stop.is_set():
            if index % 2 == 0:
                body, content_type = self.multipart_wav(b"not a wave file\x00\xff")
                self.request("malformed_recovery", "malformed_audio", body,
                    path="/v1/audio/transcriptions", expected="error", expected_fault=True,
                    content_type=content_type)
                body, content_type = self.multipart_wav(self.args.voice_wav.read_bytes())
                self.request("malformed_recovery", "valid_audio_recovery", body,
                    path="/v1/audio/transcriptions", expected="transcription", recovery=True,
                    content_type=content_type)
            else:
                with self.stages.lease("wan", f"malformed-video-{index}"):
                    base = {"kind": "video", "prompt": "A red cube rotates on a table",
                            "num_frames": 9, "fps": 16, "steps": 1, "seed": 42}
                    self.request("malformed_recovery", "malformed_dimensions",
                        dict(base, size="500x480"), path="/v1/visual/render",
                        expected="error", expected_fault=True)
                    self.request("malformed_recovery", "valid_dimension_recovery",
                        dict(base, size="352x256"), path="/v1/visual/render",
                        expected="binary:video/mp4", recovery=True,
                        timeout=self.args.request_timeout_s)
            index += 1
            self.stop.wait(rng.random() * 0.05)

    def run_lanes(self):
        functions = (self.lane_text_code, self.lane_interleaved_visual,
                     self.lane_full_duplex_voice, self.lane_natural_video_input,
                     self.lane_abort_retry, self.lane_malformed_recovery)
        barrier = threading.Barrier(len(LANES) + 1)
        errors = []

        def wrapped(name, function, seed):
            try:
                barrier.wait()
                function(random.Random(seed))
            except Exception as error:
                errors.append({"lane": name, "error": f"{type(error).__name__}: {error}"})
                self.events.write({"lane": name, "kind": "lane_exception",
                                   "error": str(error), "monotonic_ns": time.monotonic_ns()})
                self.stop.set()

        self.window_started_ns = time.monotonic_ns()
        for index, (name, function) in enumerate(zip(LANES, functions)):
            thread = threading.Thread(target=wrapped, args=(name, function, 70080 + index),
                                      name="ns07-" + name, daemon=True)
            self.threads.append(thread)
            thread.start()
        barrier.wait()  # all six persistent lane threads exist before new work
        deadline = time.monotonic() + self.args.duration_s
        while time.monotonic() < deadline and not self.stop.is_set():
            self.stop.wait(min(0.25, deadline - time.monotonic()))
        self.stop.set()
        self.window_ended_ns = time.monotonic_ns()
        for thread in self.threads:
            thread.join(self.args.request_timeout_s + 120)
        alive = [thread.name for thread in self.threads if thread.is_alive()]
        if alive:
            raise AssertionError(f"client lanes did not drain: {alive}")
        wait_health_zero(self.ports[0], 120)
        self.final_health = wait_health_zero(self.ports[0], 10)
        self.stop_resource_sampling.set()
        self.resource_thread.join(10)
        require(not self.resource_thread.is_alive(), "resource sampler did not stop")
        self.stages.close()
        self.final_same_resource = self.sample_resource("final_same_residency")
        return errors

    def cleanup(self):
        if self.stages is not None:
            try:
                self.stages.close()
            except Exception as error:
                self.driver_error = self.driver_error or str(error)
        self.stop_resource_sampling.set()
        if self.resource_thread is not None:
            self.resource_thread.join(10)
        for name in ("vlm", "asr", "server"):
            process = self.processes.get(name)
            if process is not None and process.stop_ns is None:
                was_alive = process.process.poll() is None
                process.stop(self.endpoints.get(name) if name in ("vlm", "asr") else None)
                process.expected_stop = was_alive
        for process in self.processes.values():
            record = process.record()
            record["command"] = list(process.command)
            record["expected_stop"] = getattr(process, "expected_stop", True)
            self.process_history.append(record)
        for tap in (self.tts_tap, self.tts_output_tap):
            if tap is not None and tap.thread.is_alive():
                try:
                    tap.stop()
                except Exception as error:
                    self.driver_error = self.driver_error or str(error)
        if self.ports is not None:
            try:
                self.gpu_final = wait_gpu0_idle()
            except Exception as error:
                self.driver_error = self.driver_error or f"GPU cleanup: {error}"
        try:
            shutil.rmtree(self.out / "run/ipc")
            shutil.rmtree(self.out / "run/shm")
        except OSError:
            pass

    def identities(self):
        args = self.args
        models = {}
        configs = {}
        for role, root in args.model_map.items():
            identity = public_base.model_identity(root, {
                "asr": "Qwen/Qwen3-ASR-1.7B", "vlm": "Qwen/Qwen3.8-27B",
                "flux": "black-forest-labs/FLUX.2-klein-4B",
                "tts": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                "wan": "Wan-AI/Wan2.2-TI2V-5B-Diffusers"}[role])
            models[role] = identity
            config = {}
            for name in ("config.json", "model_index.json"):
                path = Path(root) / name
                if path.is_file():
                    config[name] = sha_file(path)["sha256"]
            configs[role] = config
        binaries = {}
        for role, path in (("server", args.server), ("vlm", args.vlm), ("asr", args.asr),
                           ("tts", args.tts), ("wan", args.wan)):
            binaries[role] = {**sha_file(path), "executable": str(path)}
        return models, configs, binaries

    def execute(self):
        models, configs, binaries = {}, {}, {}
        errors = []
        try:
            # Do not spend minutes hashing model trees before the idle preflight.
            # Start owned base stages immediately, then hash exact identities while
            # they are resident but before the measured mixed-work window.
            self.setup_stack()
            models, configs, binaries = self.identities()
            errors = self.run_lanes()
        except BaseException as error:
            # Signals and KeyboardInterrupt still take the owned-process cleanup
            # path and produce failed evidence; they can never become a PASS.
            self.driver_error = f"{type(error).__name__}: {error}"
        finally:
            self.cleanup()
        run = {"schema": SCHEMA, "case_id": "NS-07", "arm": self.args.arm,
            "evidence_class": "REAL_DIGITAL_LOOPBACK_NATIVE_MODELS",
            "source_root": str(self.args.source),
            "source": {"execution_commit": self.args.execution_commit,
                       "hardware": {"physical_gpu": 0,
                                    "accelerator": (self.gpu_initial or {}).get("name", "NOT_OBSERVED")},
                       "source_tree_sha256": subprocess.check_output(
                           ["git", "-C", str(self.args.source), "rev-parse", "HEAD^{tree}"],
                           text=True).strip(),
                       "harness_sha256": sha_file(Path(__file__))["sha256"],
                       "preregistration_sha256": sha_file(PREREGISTRATION)["sha256"]},
            "models": models, "configs": configs, "binaries": binaries,
            "inputs": {"voice_wav": sha_file(self.args.voice_wav),
                       "barge_in_wav": sha_file(self.args.barge_in_wav),
                       "natural_video": sha_file(self.args.natural_video)},
            "duration_s": self.args.duration_s, "seed": 70080, "lanes": list(LANES),
            "ports": {"http": self.ports[0], "realtime": self.ports[1]} if self.ports else None,
            "window_started_monotonic_ns": self.window_started_ns,
            "window_ended_monotonic_ns": self.window_ended_ns,
            "gpu_initial": self.gpu_initial, "gpu_final": self.gpu_final,
            "baseline_resource": self.baseline_resource,
            "final_same_residency": self.final_same_resource,
            "final_health": self.final_health, "driver_error": self.driver_error,
            "lane_errors": errors, "processes": self.process_history + (self.stages.history if self.stages else []),
            "leases": self.stages.leases if self.stages else [],
            "output_stage_history": self.stages.history if self.stages else [],
            "output_stage_drain_history": self.stages.drain_history if self.stages else [],
            "taps": {"tts": self.tts_tap.records if self.tts_tap else [],
                     "tts_output": self.tts_output_tap.records if self.tts_output_tap else [],
                     "errors": [tap.error for tap in (self.tts_tap, self.tts_output_tap)
                                if tap is not None and tap.error]},
            "limits": {"production": "NOT_RUN", "physical_playback": "NOT_RUN",
                       "perceptual_quality": "NOT_RUN", "five_gpu_resident_gate": "NOT_RUN"}}
        write_json(self.out / "run.json", run)
        return run


def parse_models(values):
    result = {}
    for value in values or []:
        role, separator, path = value.partition("=")
        require(separator and role in ("asr", "vlm", "flux", "tts", "wan") and path,
                "--models requires ROLE=DIR with roles asr,vlm,flux,tts,wan")
        require(role not in result, f"duplicate model role: {role}")
        result[role] = Path(path).expanduser().resolve()
    require(set(result) == {"asr", "vlm", "flux", "tts", "wan"},
            "--models must supply exactly asr,vlm,flux,tts,wan")
    return result


def self_check():
    document = json.loads(PREREGISTRATION.read_text())
    require(document["duration_s"] == 1800 and document["client_count"] == 6,
            "frozen NS-07 duration/client contract changed")
    require(tuple(document["lanes"]) == LANES, "lane identity/order changed")
    ports = free_loopback_ports()
    require(43301 not in ports and ports[0] != ports[1], "unsafe ports")
    assert percentile([1, 2, 3, 4], 0.95) == 4
    print("NS07_SIX_WAY_OFFLINE_CONTRACT: PASS")
    print("REAL_GENERATION: NOT_RUN")
    print("ISOLATED_PASS: NOT_RUN")
    return 0


def call_verifier(evidence, output, final=False):
    command = [sys.executable, "-I", str(VERIFIER), str(evidence), "--output", str(output)]
    if final:
        command.append("--verify-manifest")
    result = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    if final:
        output.with_suffix(output.suffix + ".log").write_text(
            result.stdout + result.stderr, encoding="utf-8")
    else:
        (evidence / "logs/verifier-initial.log").write_text(
            result.stdout + result.stderr, encoding="utf-8")
    return result


def normalized_from_verification(verification, run, manifest_count):
    status = verification.get("engineering_status", "FAIL")
    return {"case_id": "NS-07", "schema": NORMALIZED_SCHEMA, "arm": run.get("arm", "after"),
        "engineering_status": status, "production_status": "NOT_RUN",
        "full_gate_status": "NOT_RUN", "metrics": verification.get("metrics", {}),
        "hard_gates": verification.get("hard_gates", {}),
        "failures": verification.get("failures", []),
        "limitations": ["five-GPU resident production NOT_RUN", "physical acoustics NOT_RUN",
                        "perceptual quality NOT_RUN", "deployment NOT_RUN"],
        "source_identity": {**run["source"], "commit": run["source"]["execution_commit"]},
        "binary_identities": {name: value["sha256"] for name, value in run["binaries"].items()},
        "model_identities": {name: {"model_id": value["model_id"],
                                    "tree_sha256": value["tree_sha256"],
                                    "file_count": value["file_count"]}
                             for name, value in run["models"].items()},
        "config_identity": run["configs"],
        "raw_evidence": {"run_sha256": sha_file(run["_run_path"])["sha256"],
                         "verification_sha256": verification["verification_sha256"],
                         "input_sha256": {name: value["sha256"] for name, value in run["inputs"].items()},
                         "manifest": "SHA256SUMS", "manifest_file_count": manifest_count,
                         "manifest_self_excluded": True},
        "coverage": verification.get("coverage", {}),
        "deployment_scope": "TASK_OWNED_LOOPBACK_PHYSICAL_GPU0_ONLY"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--models", action="append", metavar="ROLE=DIR")
    parser.add_argument("--server", type=Path)
    parser.add_argument("--vlm", type=Path)
    parser.add_argument("--asr", type=Path)
    parser.add_argument("--tts", type=Path)
    parser.add_argument("--wan", type=Path)
    parser.add_argument("--voice-wav", type=Path)
    parser.add_argument("--barge-in-wav", type=Path)
    parser.add_argument("--natural-video", type=Path)
    parser.add_argument("--python")
    parser.add_argument("--library-path")
    parser.add_argument("--execution-commit")
    parser.add_argument("--arm", choices=("before", "after"))
    parser.add_argument("--duration-s", type=float, default=1800.0)
    parser.add_argument("--startup-timeout-s", type=float, default=600.0)
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    args = parser.parse_args(argv)
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    if args.self_check:
        return self_check()
    required = ("evidence_dir", "source", "server", "vlm", "asr", "tts", "wan",
                "voice_wav", "barge_in_wav", "natural_video", "python", "library_path",
                "execution_commit", "arm")
    missing = [name for name in required if getattr(args, name) is None]
    if missing or args.models is None:
        parser.error("live mode requires all inputs; missing " + ", ".join(missing + ([] if args.models else ["models"])))
    require(os.environ.get("UMMINFER_LIVE_OPT_IN") == "1",
            "REAL_GENERATION: NOT_RUN (explicit UMMINFER_LIVE_OPT_IN=1 required)")
    require(math.isfinite(args.duration_s) and args.duration_s > 0, "duration must be positive")
    require(len(args.execution_commit) == 40, "execution commit must be a full 40-character id")
    args.source = args.source.expanduser().resolve()
    args.model_map = parse_models(args.models)
    for name in ("server", "vlm", "asr", "tts", "wan", "voice_wav", "barge_in_wav", "natural_video"):
        path = getattr(args, name).expanduser().resolve()
        require(path.is_file(), f"--{name.replace('_', '-')} is not a file: {path}")
        setattr(args, name, path)
    require((args.source / "python/umminfer_flux2_stage.py").is_file(), "source lacks FLUX stage")
    actual_commit = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"],
                                            text=True).strip()
    require(actual_commit == args.execution_commit,
            f"source HEAD {actual_commit} differs from --execution-commit")
    args.env = dict(os.environ)
    args.env["CUDA_VISIBLE_DEVICES"] = "0"
    args.env["UMM_QWEN38_PREFIX_CACHE"] = "0"
    if args.library_path:
        args.env["LD_LIBRARY_PATH"] = args.library_path + (
            ":" + args.env["LD_LIBRARY_PATH"] if args.env.get("LD_LIBRARY_PATH") else "")
    args.env["UMM_TRACE"] = "1"
    soak = Soak(args)
    args.env["UMM_TRACE_FILE"] = str(soak.out / "server.trace.jsonl")
    run = soak.execute()
    run["_run_path"] = str(soak.out / "run.json")
    initial_output = soak.out.parent / (soak.out.name + "-verification-initial.json")
    require(not initial_output.exists(), f"verification output already exists: {initial_output}")
    initial = call_verifier(soak.out, initial_output)
    verification = json.loads(initial_output.read_text()) if initial_output.exists() else {
        "engineering_status": "FAIL", "failures": ["initial verifier did not write a report"],
        "metrics": {}, "verification_sha256": None}
    verification["verification_sha256"] = sha_file(initial_output)["sha256"] if initial_output.exists() else None
    # The manifest includes the normalized result, but the normalized result does
    # not embed the manifest hash (which would create a circular identity).
    manifest_count = sum(path.is_file() and path.name != "SHA256SUMS"
                         for path in soak.out.rglob("*")) + 1
    normalized = normalized_from_verification(verification, run, manifest_count)
    write_json(soak.out / "NS-07.json", normalized)
    rows = []
    for path in sorted(soak.out.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            require(not path.is_symlink(), f"symlink evidence is forbidden: {path}")
            rows.append(f"{sha_file(path)['sha256']}  {path.relative_to(soak.out).as_posix()}")
    (soak.out / "SHA256SUMS").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="ascii")
    final_output = soak.out.parent / (soak.out.name + "-verification-final.json")
    require(not final_output.exists(), f"verification output already exists: {final_output}")
    final = call_verifier(soak.out, final_output, final=True)
    passed = final.returncode == 0 and normalized["engineering_status"] == "ISOLATED_PASS"
    print(json.dumps({"case_id": "NS-07", "engineering_status": normalized["engineering_status"],
                      "production_status": "NOT_RUN", "evidence_dir": str(soak.out),
                      "verification": str(final_output),
                      "metrics": normalized["metrics"]}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
