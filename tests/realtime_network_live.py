#!/usr/bin/env python3
"""Real public-WebSocket/network gate for native ASR -> Qwen3.8 -> streaming TTS.

The driver owns only loopback processes and physical GPUs 0/1. It records exact
WebSocket wire frames, deterministic PCM, ACK/cancel schedules and an unmodified
forwarding observer's native VLM prompts. Acceptance is delegated to a fresh
independent verifier process. No physical audio device or listening claim is made.
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


net = load_module("umminfer_realtime_network_driver", ROOT / "python/umminfer_realtime_network.py")
echo = load_module("umminfer_full_duplex_echo_helpers", ROOT / "tests/full_duplex_echo_live.py")

FRAME_BYTES = 320
CHUNK_BYTES = 3200
SILENCE_MS = 400
END_SILENCE = bytes(16000 * 2 * 6 // 10)
CUTIN_TYPES = {
    "input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped",
    "response.cancelled", "transcription.completed",
}
STALE_TYPES = {
    "audio.delta", "audio.done", "response.text.delta", "response.playback_ack",
    "response.completed", "response.failed",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def hash_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def free_ports(count):
    sockets = []
    try:
        for _ in range(count):
            sock = socket.socket()
            sockets.append(sock)
            sock.bind(("127.0.0.1", 0))
        ports = [sock.getsockname()[1] for sock in sockets]
        require(43301 not in ports, "ephemeral port collided with the resident service")
        return ports
    finally:
        for sock in sockets:
            sock.close()


def health(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as reply:
        return json.load(reply)


def wait_health_zero(port, timeout=30):
    deadline = time.monotonic() + timeout
    latest = None
    while time.monotonic() < deadline:
        latest = health(port)
        if latest.get("in_flight") == 0:
            return latest
        time.sleep(0.05)
    raise AssertionError(f"in_flight did not drain: {latest}")


def gpu_state(gpu):
    row = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits",
    ], text=True).strip()
    fields = [value.strip() for value in row.split(",")]
    compute = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader",
    ], text=True).strip()
    return {"index": int(fields[0]), "memory_used_mib": int(fields[1]),
            "utilization_percent": int(fields[2]),
            "compute_processes": [] if not compute else [int(v) for v in compute.splitlines()]}


def preflight():
    rows = {str(gpu): gpu_state(gpu) for gpu in (0, 1)}
    for gpu, row in rows.items():
        require(row["memory_used_mib"] <= 64 and row["utilization_percent"] == 0 and
                not row["compute_processes"], f"assigned GPU {gpu} is not idle: {row}")
    return rows


def wait_postflight(timeout=30):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            return preflight()
        except (AssertionError, subprocess.CalledProcessError) as error:
            last = str(error)
            time.sleep(0.5)
    raise AssertionError(f"assigned GPUs did not return to idle: {last}")


def process_start_ticks(pid):
    with open(f"/proc/{pid}/stat", "rb") as source:
        return int(source.read().split()[21])


class OwnedProc:
    def __init__(self, name, args, log_path):
        self.name = name
        self.args = [str(value) for value in args]
        self.log = echo.CappedLog(log_path)
        self.started_wall_ns = time.time_ns()
        self.started_monotonic_ns = time.monotonic_ns()
        self.proc = subprocess.Popen(
            self.args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=dict(os.environ), start_new_session=True)
        self.start_ticks = process_start_ticks(self.proc.pid)
        self.log.start(self.proc)
        self.exit_code = None

    @property
    def pid(self):
        return self.proc.pid

    def identity(self):
        return {"name": self.name, "pid": self.pid, "process_start_ticks": self.start_ticks,
                "started_wall_ns": self.started_wall_ns,
                "started_monotonic_ns": self.started_monotonic_ns}

    def belongs_to_this_run(self):
        try:
            return process_start_ticks(self.pid) == self.start_ticks
        except (FileNotFoundError, ProcessLookupError):
            return False

    def stop(self):
        if self.proc.poll() is None and self.belongs_to_this_run():
            self.proc.send_signal(signal.SIGTERM)
        try:
            self.exit_code = self.proc.wait(timeout=25)
        except subprocess.TimeoutExpired:
            require(self.belongs_to_this_run(),
                    f"refusing to kill reused pid {self.pid} ({self.name})")
            os.killpg(self.pid, signal.SIGKILL)
            self.exit_code = self.proc.wait(timeout=5)
        self.log.wait()


def stop_stack(procs):
    for proc in reversed(list(procs.values())):
        proc.stop()


def model_manifest(model):
    rows = []
    for path in sorted(model.rglob("*")):
        if path.is_file():
            rows.append({"path": str(path.relative_to(model)), "bytes": path.stat().st_size,
                         "sha256": hash_file(path)})
    encoded = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    return {"revision": "UNKNOWN", "files": rows, "model_manifest_sha256": sha(encoded)}


def hash_models(args, out):
    models = {"asr": args.asr_model, "vlm": args.vlm_model, "tts": args.tts_model}
    with ThreadPoolExecutor(max_workers=3) as pool:
        values = list(pool.map(model_manifest, models.values()))
    result = dict(zip(models, values))
    for role, manifest in result.items():
        (out / "models").mkdir(parents=True, exist_ok=True)
        (out / "models" / f"{role}.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return result


def normalize_vlm_messages(messages):
    normalized = []
    for message_row in messages:
        content = message_row.get("content")
        if isinstance(content, list):
            if (len(content) != 1 or content[0].get("type") != "text" or
                    not isinstance(content[0].get("text"), str)):
                raise ValueError("VLM observer expected one text content part")
            content = content[0]["text"]
        normalized.append({"role": message_row.get("role"), "content": content})
    return normalized


class VlmObserver:
    """Byte-preserving PUSH/PULL forwarding; no model result is synthesized."""
    def __init__(self, public_endpoint, native_endpoint, output_path):
        import zmq
        self.context = zmq.Context()
        self.public = self.context.socket(zmq.PULL)
        self.public.bind(public_endpoint)
        self.native = self.context.socket(zmq.PUSH)
        self.native.connect(native_endpoint)
        self.output_path = output_path
        self.stop_event = threading.Event()
        self.error = None
        self.rows = 0
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        import msgpack
        import zmq
        try:
            with open(self.output_path, "ab", buffering=0) as output:
                while not self.stop_event.is_set():
                    if not self.public.poll(100):
                        continue
                    frames = self.public.recv_multipart()
                    self.native.send_multipart(frames)
                    if len(frames) != 1:
                        continue
                    message = msgpack.unpackb(frames[0], raw=False, strict_map_key=False)
                    if message.get("type") != "submit":
                        continue
                    data = message.get("data", {})
                    request = data.get("request", {})
                    inputs = request.get("inputs", {})
                    messages = inputs.get("messages")
                    if not isinstance(messages, list):
                        continue
                    normalized = normalize_vlm_messages(messages)
                    row = {"observed_monotonic_ns": time.monotonic_ns(),
                           "request_id": message.get("request_id"),
                           "messages": normalized,
                           "raw_sha256": sha(frames[0]),
                           "raw_base64": base64.b64encode(frames[0]).decode()}
                    output.write((json.dumps(row, sort_keys=True) + "\n").encode())
                    self.rows += 1
        except Exception as error:
            self.error = error

    def close(self):
        self.stop_event.set()
        self.thread.join(10)
        self.public.close(0)
        self.native.close(0)
        self.context.term()
        if self.error:
            raise self.error


def start_stack(args, ports, out, logs):
    unique = f"{os.getpid()}-{time.time_ns()}"
    ipc = logs / f"ipc-{unique}"
    shm = logs / f"shm-{unique}"
    ipc.mkdir(parents=True, exist_ok=False)
    shm.mkdir(parents=True, exist_ok=False)
    paths = {name: f"ipc://{ipc}/{name}" for name in
             ("asr", "vlm-native", "vlm-public", "tts", "completion", "abort")}
    observer = VlmObserver(paths["vlm-public"], paths["vlm-native"], out / "vlm-prompts.jsonl")
    procs = {}
    try:
        procs["asr"] = OwnedProc("asr", [
            args.bin_dir / "umm-native-stage-asr", "--kind", "asr", "--name", "asr",
            "--model", args.asr_model, "--gpu", 0,
            "--recv-endpoint", paths["asr"], "--coordinator-endpoint", paths["completion"],
            "--abort-endpoint", paths["abort"]], logs / "asr.log")
        procs["vlm"] = OwnedProc("vlm", [
            args.bin_dir / "umm-native-stage-qwen38", "--kind", "qwen38", "--name", "vlm",
            "--model", args.vlm_model, "--gpu", 1,
            "--recv-endpoint", paths["vlm-native"], "--coordinator-endpoint", paths["completion"],
            "--abort-endpoint", paths["abort"], "--max-new-tokens", args.vlm_max_tokens],
            logs / "vlm.log")
        procs["tts"] = OwnedProc("tts", [
            args.bin_dir / "umm-native-stage-tts", "--kind", "tts", "--name", "tts",
            "--model", args.tts_model, "--gpu", 1,
            "--recv-endpoint", paths["tts"], "--coordinator-endpoint", paths["completion"],
            "--abort-endpoint", paths["abort"], "--max-frames", args.tts_max_frames],
            logs / "tts.log")
        procs["server"] = OwnedProc("server", [
            args.bin_dir / "umm-infer", "serve", "--host", "127.0.0.1",
            "--port", ports[0], "--realtime-port", ports[1],
            "--model", "qwen3-unified", "--entry", "asr",
            "--entry-endpoint", paths["asr"], "--vlm-entry", "vlm",
            "--vlm-entry-endpoint", paths["vlm-public"], "--tts-entry", "tts",
            "--tts-entry-endpoint", paths["tts"], "--completion", paths["completion"],
            "--abort", paths["abort"], "--session-history", "--max-in-flight", 32,
            "--ipc-dir", shm], logs / "server.log")
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            require(observer.error is None, f"VLM observer failed: {observer.error}")
            for name, proc in procs.items():
                require(proc.proc.poll() is None, f"{name} exited during model load")
            ready = all(marker in procs[name].log.text() for name, marker in {
                "asr": "ready kind=asr", "vlm": "ready kind=qwen38", "tts": "ready kind=tts",
            }.items())
            if ready:
                try:
                    with urllib.request.urlopen(
                            f"http://127.0.0.1:{ports[0]}/ready", timeout=1) as reply:
                        if json.load(reply).get("status") == "ready":
                            return procs, observer, (ipc, shm)
                except Exception:
                    pass
            time.sleep(0.25)
        raise AssertionError("task-owned real stages did not become ready")
    except Exception:
        if procs:
            stop_stack(procs)
        observer.close()
        shutil.rmtree(ipc, ignore_errors=True)
        shutil.rmtree(shm, ignore_errors=True)
        raise


class RawWebSocket:
    def __init__(self, port, capture_path):
        self.capture_path = capture_path
        self.capture_path.parent.mkdir(parents=True, exist_ok=True)
        self.capture = open(capture_path, "xb", buffering=0)
        self.capture_lock = threading.Lock()
        self.send_lock = threading.Lock()
        self.condition = threading.Condition()
        self.events = []
        self.closed = False
        self.close_error = None
        self.fragments = None
        self.socket = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.socket.settimeout(0.25)
        key = base64.b64encode(secrets.token_bytes(16)).decode()
        request = (
            "GET /v1/realtime?intent=transcription&model=qwen3-unified HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode()
        self.socket.sendall(request)
        response = bytearray()
        while b"\r\n\r\n" not in response:
            block = self.socket.recv(4096)
            require(block, "WebSocket handshake closed early")
            response.extend(block)
            require(len(response) <= 65536, "oversized WebSocket handshake")
        head, remainder = bytes(response).split(b"\r\n\r\n", 1)
        require(head.startswith(b"HTTP/1.1 101"), f"WebSocket handshake failed: {head!r}")
        expected = base64.b64encode(hashlib.sha1(
            (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest())
        require(b"sec-websocket-accept: " + expected.lower() in head.lower(),
                "WebSocket accept digest mismatch")
        self.remainder = remainder
        self.index = 0
        self.reader = threading.Thread(target=self.read_loop, daemon=True)
        self.reader.start()

    def record(self, direction, raw):
        with self.capture_lock:
            row = {"index": self.index, "monotonic_ns": time.monotonic_ns(),
                   "direction": direction, "raw_base64": base64.b64encode(raw).decode()}
            self.index += 1
            self.capture.write((json.dumps(row, sort_keys=True) + "\n").encode())
        return row

    def read_exact(self, count):
        output = bytearray()
        if self.remainder:
            take = min(count, len(self.remainder))
            output.extend(self.remainder[:take])
            self.remainder = self.remainder[take:]
        while len(output) < count:
            try:
                block = self.socket.recv(count - len(output))
            except socket.timeout:
                if self.closed:
                    raise EOFError("closed")
                continue
            if not block:
                raise EOFError("peer closed")
            output.extend(block)
        return bytes(output)

    def read_loop(self):
        try:
            while not self.closed:
                header = self.read_exact(2)
                second = header[1]
                length = second & 0x7F
                extra = bytearray()
                if length == 126:
                    extra.extend(self.read_exact(2))
                    length = struct.unpack("!H", extra)[0]
                elif length == 127:
                    extra.extend(self.read_exact(8))
                    length = struct.unpack("!Q", extra)[0]
                require(length <= (1 << 22), "oversized incoming WebSocket frame")
                require(not (second & 0x80), "server frame was unexpectedly masked")
                payload = self.read_exact(length)
                raw = header + bytes(extra) + payload
                arrival = self.record("rx", raw)
                fin = bool(header[0] & 0x80)
                opcode = header[0] & 0x0F
                if opcode == 9:
                    self.send_frame(10, payload)
                    continue
                if opcode == 8:
                    self.closed = True
                    with self.condition:
                        self.condition.notify_all()
                    break
                if opcode == 10:
                    continue
                if opcode != 0:
                    self.fragments = bytearray()
                require(self.fragments is not None, "continuation without start")
                self.fragments.extend(payload)
                if not fin:
                    continue
                message = bytes(self.fragments)
                self.fragments = None
                event = json.loads(message)
                with self.condition:
                    self.events.append({"arrival_ns": arrival["monotonic_ns"],
                                        "capture_index": arrival["index"],
                                        "event_index": len(self.events), "event": event})
                    self.condition.notify_all()
        except Exception as error:
            if not self.closed:
                self.close_error = error
            with self.condition:
                self.closed = True
                self.condition.notify_all()

    def send_frame(self, opcode, payload):
        with self.send_lock:
            if self.closed and opcode != 8:
                raise EOFError("attempted send on closed realtime connection")
            mask = secrets.token_bytes(4)
            first = 0x80 | opcode
            length = len(payload)
            if length < 126:
                header = bytes((first, 0x80 | length))
            elif length < 65536:
                header = bytes((first, 0x80 | 126)) + struct.pack("!H", length)
            else:
                header = bytes((first, 0x80 | 127)) + struct.pack("!Q", length)
            masked = bytes(value ^ mask[i % 4] for i, value in enumerate(payload))
            raw = header + mask + masked
            self.socket.sendall(raw)
            self.record("tx", raw)

    def send_json(self, event):
        self.send_frame(1, json.dumps(event, separators=(",", ":")).encode())

    def mark(self):
        with self.condition:
            return len(self.events)

    def wait_for(self, predicate, after, timeout, description):
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                for record in self.events[after:]:
                    if predicate(record["event"]):
                        return record
                if self.close_error:
                    raise AssertionError(f"socket failed while waiting for {description}: {self.close_error}")
                if self.closed or time.monotonic() >= deadline:
                    recent = []
                    for record in self.events[after:][-6:]:
                        event = dict(record["event"])
                        if "audio" in event:
                            event["audio"] = f"<{len(event['audio'])} base64 chars>"
                        recent.append(event)
                    raise AssertionError(f"timeout waiting for {description}; recent={recent}")
                self.condition.wait(min(0.25, max(0.01, deadline - time.monotonic())))

    def since(self, after, types=None, response_id=None):
        with self.condition:
            return [record for record in self.events[after:]
                    if (types is None or record["event"].get("type") in types) and
                    (response_id is None or record["event"].get("response_id") == response_id)]

    def close(self):
        if not self.closed:
            try:
                self.send_frame(8, struct.pack("!H", 1000))
            except (OSError, EOFError):
                pass
            self.closed = True
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.socket.close()
        self.reader.join(5)
        self.capture.close()


class LiveClient:
    def __init__(self, session, generation, port, root):
        self.session = session
        self.generation = generation
        self.name = f"session-{session}-connection-{generation}"
        self.ws = RawWebSocket(port, root / "frames" / f"{self.name}.jsonl")
        self.ack_schedule = []
        self.cancel_schedule = []
        self.terminal_callbacks = []
        self.late_callback_count = 0
        marker = self.ws.mark()
        self.ws.send_json({"type": "session.update", "session": {
            "input_audio_format": "pcm16",
            "turn_detection": {"type": "server_vad", "silence_ms": SILENCE_MS},
            "max_new_tokens": 96,
            "audio_output": {"voice": "vivian", "playback_ack": True},
        }})
        updated = self.ws.wait_for(lambda e: e.get("type") == "transcription_session.updated",
                                   marker, 20, "initial session update")
        require(updated["event"]["session"]["audio_output"].get("playback_ack") is True,
                "playback acknowledgement was not negotiated")

    def append(self, pcm):
        self.ws.send_json({"type": "input_audio_buffer.append",
                           "audio": base64.b64encode(pcm).decode()})

    def stream(self, pcm):
        started = time.monotonic_ns()
        for offset in range(0, len(pcm), CHUNK_BYTES):
            chunk = pcm[offset:offset + CHUNK_BYTES]
            self.append(chunk)
            time.sleep(len(chunk) / 32000.0)
        return started

    def send_user(self, user):
        marker = self.ws.mark()
        started = self.stream(user[:64000] + END_SILENCE)
        completed = self.ws.wait_for(lambda e: e.get("type") == "transcription.completed",
                                     marker, 180, "real ASR final")
        require(completed["event"].get("text", "").strip(), "empty real ASR transcript")
        return {"marker": marker, "completed": completed, "started_ns": started}

    def response_identity(self, after, timeout=180):
        first = self.ws.wait_for(
            lambda e: e.get("response_id") and e.get("type") in
            ("response.text.delta", "audio.delta"), after, timeout, "real VLM/TTS response")
        return first["event"]["response_id"], first["event_index"]

    def response_snapshot(self, response_id, after, wait_terminal=True):
        terminal = None
        if wait_terminal:
            terminal = self.ws.wait_for(
                lambda e: e.get("response_id") == response_id and e.get("type") in
                ("response.completed", "response.failed"), after, 240, "real response terminal")
            require(terminal["event"]["type"] == "response.completed",
                    f"real response failed: {terminal['event']}")
            for record in self.ws.since(after, types={"response.completed", "response.failed",
                                                     "response.cancelled"}, response_id=response_id):
                self.terminal_callbacks.append(record["event"]["type"])
        records = self.ws.since(after, response_id=response_id)
        audio = [record for record in records if record["event"]["type"] == "audio.delta"]
        text = "".join(record["event"].get("text", "") for record in records
                       if record["event"]["type"] == "response.text.delta")
        pcm = b"".join(base64.b64decode(record["event"]["audio"]) for record in audio)
        cursors = sorted({int(record["event"].get("text_cursor", 0)) for record in audio})
        positive = [cursor for cursor in cursors if 0 < cursor < len(text.encode())]
        return {"response_id": response_id, "after": after, "terminal": terminal,
                "records": records, "audio": audio, "text": text, "pcm": pcm,
                "pcm_sha256": sha(pcm), "positive_cursors": positive}

    def ack(self, response_id, cursor, delay_ms, label):
        time.sleep(delay_ms / 1000.0)
        self.ack_schedule.append({"monotonic_ns": time.monotonic_ns(), "response_id": response_id,
                                  "text_cursor": cursor, "delay_ms": delay_ms, "label": label})
        self.ws.send_json({"type": "output_audio_buffer.playback_ack",
                           "response_id": response_id, "text_cursor": cursor})

    def exercise_acks(self, snapshot, delay_ms, final):
        response_id = snapshot["response_id"]
        text_bytes = len(snapshot["text"].encode())
        cursor = snapshot["positive_cursors"][0] if snapshot["positive_cursors"] else 0
        if cursor > 0:
            self.ack(response_id, 0, 0, "decreasing-before-partial-valid-at-zero")
            self.ack(response_id, cursor, delay_ms, "delayed-partial")
            marker = self.ws.mark()
            self.ack(response_id, cursor, 0, "duplicate-partial")
            self.ws.wait_for(lambda e: e.get("type") == "response.playback_ack" and
                             e.get("response_id") == response_id and
                             e.get("text_cursor") == cursor, marker, 10, "duplicate ACK")
            self.ack(response_id, max(0, cursor - 1), 0, "decreasing-ignored")
            self.ack(f"foreign-{self.session}-{self.generation}", cursor, 0, "foreign-ignored")
            self.ack(response_id, text_bytes + 1, 0, "unissued-ignored")
        if final:
            require(text_bytes > 0, "real response had no text")
            marker = self.ws.mark()
            self.ack(response_id, text_bytes, delay_ms, "delayed-final")
            ack = self.ws.wait_for(
                lambda e: e.get("type") == "response.playback_ack" and
                e.get("response_id") == response_id and e.get("final") is True,
                marker, 15, "final playback ACK")
            require(ack["event"]["text_cursor"] == text_bytes, "wrong final cursor")
            self.ack(response_id, text_bytes, 0, "duplicate-final-after-release")
            self.ack(response_id, 0, 0, "decreasing-final-after-release")
        return cursor

    def barge(self, snapshot, user, ack_cursor, immediate):
        response_id = snapshot["response_id"] if snapshot else None
        marker = self.ws.mark()
        started = time.monotonic_ns()
        self.cancel_schedule.append({"monotonic_ns": started, "response_id": response_id,
                                     "kind": "immediate" if immediate else "delayed",
                                     "ack_cursor": ack_cursor})
        stream_thread = threading.Thread(target=self.stream, args=(user[:64000] + END_SILENCE,))
        stream_thread.start()
        cancelled = None
        if response_id is not None:
            cancelled = self.ws.wait_for(
                lambda e: e.get("type") == "response.cancelled" and
                e.get("response_id") == response_id, marker, 5, "server barge-in cancellation")
            require(cancelled["event"]["text_cursor"] == ack_cursor,
                    "cancellation changed acknowledged prefix")
            require(cancelled["event"]["history_recorded"] is True,
                    "barge-in history was not recorded")
            self.terminal_callbacks.append("response.cancelled")
        stream_thread.join()
        stopped = self.ws.wait_for(lambda e: e.get("type") == "input_audio_buffer.speech_stopped",
                                   marker, 30, "post-barge endpoint")
        completed = self.ws.wait_for(
            lambda e: e.get("type") == "transcription.completed" and
            e.get("segment_id") == stopped["event"].get("segment_id"),
            stopped["event_index"], 180, "post-barge real ASR final")
        if cancelled is not None:
            stale = self.ws.since(cancelled["event_index"] + 1, types=STALE_TYPES,
                                  response_id=response_id)
            require(not stale, f"late old-response events survived barge-in: {stale}")
        return {"marker": marker, "completed": completed, "started_ns": started,
                "cancelled": cancelled}

    def set_echo(self, enabled):
        marker = self.ws.mark()
        self.ws.send_json({"type": "session.update", "session": {
            "turn_detection": {"type": "server_vad", "silence_ms": SILENCE_MS,
                               "echo_suppression": enabled}}})
        reply = self.ws.wait_for(lambda e: e.get("type") == "transcription_session.updated",
                                 marker, 10, "echo A/B update")
        require(reply["event"]["session"]["turn_detection"].get("echo_suppression", True) is enabled,
                "echo mode did not change")

    def close(self):
        self.ws.close()


class SessionRunner:
    def __init__(self, session, plan, args, ports, user, out):
        self.session = session
        self.plan = plan
        self.scenarios = plan["sessions"][session]["scenarios"]
        self.args = args
        self.ports = ports
        self.user = user
        self.out = out
        self.client = None
        self.generation = -1
        self.rows = []
        self.retired_clients = []
        self.late_callbacks = 0
        self.lock = threading.Lock()
        self.connected_peak = 0

    def reconnect(self):
        if self.client is not None:
            self.retired_clients.append(self.client)
            self.client.close()
        self.generation += 1
        self.client = LiveClient(self.session, self.generation, self.ports[1], self.out)

    def all_clients(self):
        return self.retired_clients + [self.client]

    def next_response(self, turn):
        return self.client.response_identity(turn["completed"]["event_index"] + 1)

    def _late_callback(self, response_id):
        # A real timer fires after its socket has been torn down. It records the
        # attempt but cannot send into a new connection or resurrect old ACK state.
        self.late_callbacks += 1

    def _save_pcm(self, snapshot, round_index):
        if not snapshot["pcm"]:
            return
        directory = self.out / "pcm" / f"session-{self.session}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"round-{round_index}-{snapshot['response_id']}.pcm").write_bytes(snapshot["pcm"])


def echo_and_double_talk(client, user, out):
    turn = client.send_user(user)
    response_id, after = client.response_identity(turn["completed"]["event_index"] + 1)
    snapshot = client.response_snapshot(response_id, after, wait_terminal=True)
    require(len(snapshot["pcm"]) >= 48000, "real TTS response is too short for echo A/B")
    echo_response_id = response_id
    echo_far_sha256 = snapshot["pcm_sha256"]
    cursor = client.exercise_acks(snapshot, 10, final=False)
    require(cursor > 0, "echo A/B needs a positive acknowledged prefix")
    reference = echo.resample_reference(snapshot["pcm"], 24000, 1)
    mic, dsp = echo.make_echo(reference, echo.ECHO_CASES[1])
    (out / "echo-far-end-24k.pcm").write_bytes(snapshot["pcm"])
    (out / "echo-microphone-16k.pcm").write_bytes(mic)
    protected_marker = client.ws.mark()
    protected_start_capture = client.ws.index
    client.stream(mic)
    time.sleep(0.3)
    protected_end_capture = client.ws.index
    protected = client.ws.since(protected_marker, types=CUTIN_TYPES)
    require(not protected, f"protected echo cut in: {protected}")
    client.set_echo(False)
    disabled_marker = client.ws.mark()
    disabled_thread = threading.Thread(target=client.stream, args=(mic,))
    disabled_thread.start()
    cancelled = client.ws.wait_for(
        lambda e: e.get("type") == "response.cancelled" and e.get("response_id") == response_id,
        disabled_marker, 5, "disabled echo A/B cancellation")
    disabled_thread.join()
    client.terminal_callbacks.append("response.cancelled")
    echo_disabled_cancel_index = cancelled["capture_index"]
    require(cancelled["event"]["text_cursor"] == cursor, "echo A/B changed ACK prefix")
    # The disabled echo turn creates the next real response. Complete it, then
    # use the following active response for simultaneous double-talk.
    stopped = client.ws.wait_for(lambda e: e.get("type") == "input_audio_buffer.speech_stopped",
                                 disabled_marker, 30, "echo-disabled endpoint")
    completed = client.ws.wait_for(
        lambda e: e.get("type") == "transcription.completed" and
        e.get("segment_id") == stopped["event"].get("segment_id"),
        stopped["event_index"], 180, "echo-disabled ASR")
    next_id, next_after = client.response_identity(completed["event_index"] + 1)
    next_snapshot = client.response_snapshot(next_id, next_after, wait_terminal=True)
    client.exercise_acks(next_snapshot, 5, final=True)
    double_results = {}
    double_cancel_indices = {}
    double_response_ids = {}
    for enabled in (True, False):
        turn = client.send_user(user)
        response_id, after = client.response_identity(turn["completed"]["event_index"] + 1)
        snapshot = client.response_snapshot(response_id, after, wait_terminal=True)
        cursor = client.exercise_acks(snapshot, 5, final=False)
        require(cursor > 0, "double-talk needs a positive acknowledged prefix")
        reference = echo.resample_reference(snapshot["pcm"], 24000, 1)
        echo_pcm, _ = echo.make_echo(reference, echo.ECHO_CASES[1])
        mixed = echo.make_double_talk(echo_pcm, user)
        (out / f"double-talk-{enabled}-far.pcm").write_bytes(snapshot["pcm"])
        (out / f"double-talk-{enabled}-near.pcm").write_bytes(mixed)
        client.set_echo(enabled)
        marker = client.ws.mark()
        stream_thread = threading.Thread(target=client.stream, args=(mixed,))
        stream_thread.start()
        cancelled = client.ws.wait_for(
            lambda e: e.get("type") == "response.cancelled" and
            e.get("response_id") == response_id, marker, 5, "double-talk cancellation")
        stream_thread.join()
        client.terminal_callbacks.append("response.cancelled")
        require(cancelled["event"]["text_cursor"] == cursor, "double-talk ACK prefix mismatch")
        double_results[str(enabled).lower()] = True
        double_cancel_indices[str(enabled).lower()] = cancelled["capture_index"]
        double_response_ids[str(enabled).lower()] = response_id
        # Finish ASR and fully ACK the resulting response before the next arm.
        stopped = client.ws.wait_for(lambda e: e.get("type") == "input_audio_buffer.speech_stopped",
                                     marker, 30, "double-talk endpoint")
        completed = client.ws.wait_for(
            lambda e: e.get("type") == "transcription.completed" and
            e.get("segment_id") == stopped["event"].get("segment_id"),
            stopped["event_index"], 180, "double-talk ASR")
        new_id, new_after = client.response_identity(completed["event_index"] + 1)
        new_snapshot = client.response_snapshot(new_id, new_after, wait_terminal=True)
        client.exercise_acks(new_snapshot, 5, final=True)
    return {
        "response_id": echo_response_id,
        "protected_microphone_sha256": sha(mic), "disabled_microphone_sha256": sha(mic),
        "protected_far_end_sha256": echo_far_sha256,
        "disabled_far_end_sha256": echo_far_sha256,
        "protected_cutins": len(protected), "disabled_cutins": 1,
        "protected_start_capture_index": protected_start_capture,
        "protected_end_capture_index": protected_end_capture,
        "disabled_cancel_capture_index": echo_disabled_cancel_index,
        "double_talk_protected_cancelled": double_results["true"],
        "double_talk_disabled_cancelled": double_results["false"],
        "double_talk_response_ids": double_response_ids,
        "double_talk_cancel_capture_indices": double_cancel_indices,
        "echo_delay_ms": dsp["delay_ms"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--asr-model", type=Path, required=True)
    parser.add_argument("--vlm-model", type=Path, required=True)
    parser.add_argument("--tts-model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x41C0DE)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--vlm-max-tokens", type=int, default=96)
    parser.add_argument("--tts-max-frames", type=int, default=192)
    args = parser.parse_args()
    require(args.rounds >= 2, "at least two rounds are required")
    require(args.bin_dir.is_absolute() and args.out.is_absolute() and args.log_dir.is_absolute(),
            "binary/output/log paths must be absolute")
    args.out.mkdir(parents=True, exist_ok=False)
    args.log_dir.mkdir(parents=True, exist_ok=False)
    (args.out / "vlm-prompts.jsonl").write_bytes(b"")
    plan = net.make_plan(args.seed, 6, args.rounds)
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    user = echo.trim_digital_lead_in(echo.pcm_from_wav(args.audio))
    require(len(user) >= 64000, "recorded microphone fixture needs at least two seconds")
    (args.out / "near-end-user.wav").write_bytes(args.audio.read_bytes())
    (args.out / "near-end-active.pcm").write_bytes(user[:64000])
    run = {
        "schema": "umminfer-realtime-network-live/v1",
        "source_commit": args.source_commit,
        "validation_class": "REAL_DIGITAL_LOOPBACK_NETWORK_AND_NATIVE_MODELS",
        "physical_microphone_speaker": "NOT_COVERED",
        "perceptual_listening_quality": "NOT_COVERED",
        "gpus": {"asr": 0, "vlm": 1, "tts": 1},
        "gpu_preflight": preflight(),
        "ports": {},
        "queue_contract": {"outbound_frames": 1024, "partial_events": 4096,
                           "unified_events": 4096, "max_audio_bytes": 64 * 1024 * 1024},
        "resource_bounds": {"server_rss_growth_max_bytes": 256 * 1024 * 1024},
        "binaries": {name: hash_file(args.bin_dir / name) for name in
                     ("umm-infer", "umm-native-stage-asr", "umm-native-stage-qwen38",
                      "umm-native-stage-tts")},
        "harness_sha256": hash_file(Path(__file__)),
        "scheduler_verifier_sha256": hash_file(ROOT / "python/umminfer_realtime_network.py"),
        "near_end_sha256": sha(user[:64000]),
    }
    print("Hashing exact native model files", flush=True)
    run["models"] = hash_models(args, args.out)
    run["gpu_launch_preflight"] = preflight()
    ports = free_ports(2)
    run["ports"] = {"http": ports[0], "websocket": ports[1]}
    procs = {}
    observer = None
    isolated = ()
    sampler = None
    clients = []
    cleanup = {"owned_processes_exited": False, "selected_gpus_idle": False}
    run["cleanup"] = cleanup
    try:
        procs, observer, isolated = start_stack(args, ports, args.out, args.log_dir)
        run["owned_processes"] = [proc.identity() for proc in procs.values()]
        sampler = echo.ResourceSampler(procs, args.out / "resources.jsonl")
        connected = set()
        connected_lock = threading.Lock()
        peak = [0]

        def drive(session):
            runner = SessionRunner(session, plan, args, ports, user, args.out)
            # SessionRunner opens its first connection in run(); the barrier below
            # is entered after construction through this small wrapper protocol.
            return runner

        runners = [drive(session) for session in range(6)]
        # Open all six before any model traffic to make concurrency an observed fact.
        for runner in runners:
            runner.reconnect()
            with connected_lock:
                connected.add(runner.client.name)
                peak[0] = max(peak[0], len(connected))
        run["concurrent_session_peak"] = peak[0]
        round_barriers = [threading.Barrier(6) for _ in range(args.rounds)]

        def execute(runner):
            # run() normally reconnects; preserve the already negotiated connection.
            turn = runner.client.send_user(runner.user)
            response_id, after = runner.next_response(turn)
            runner.generation = 0
            for round_index, scenario in enumerate(runner.scenarios):
                delay = runner.plan["rounds"][round_index]["ack_delay_ms"][str(runner.session)]
                jitter = runner.plan["rounds"][round_index]["start_jitter_ms"][str(runner.session)]
                round_barriers[round_index].wait()
                order = runner.plan["rounds"][round_index]["start_order"]
                rank = order.index(runner.session)
                time.sleep((rank * 3 + jitter) / 1000.0)
                if scenario == "reconnect_generation":
                    runner.client.close()
                    runner.reconnect()
                    turn = runner.client.send_user(runner.user)
                    response_id, after = runner.next_response(turn)
                elif scenario == "reconnect_playback":
                    runner.client.ws.wait_for(
                        lambda e: e.get("type") == "audio.delta" and
                        e.get("response_id") == response_id, after, 240, "playback before reconnect")
                    snapshot = runner.client.response_snapshot(response_id, after, False)
                    require(snapshot["audio"], "reconnect playback had no PCM")
                    runner.client.close()
                    runner.reconnect()
                    require(not runner.client.ws.since(0, response_id=response_id),
                            "stale response replayed after reconnect")
                    runner.client.ack(response_id, 0, 0, "late-old-ack-after-reconnect")
                    turn = runner.client.send_user(runner.user)
                    response_id, after = runner.next_response(turn)
                elif scenario == "teardown_callbacks":
                    runner.client.ws.wait_for(
                        lambda e: e.get("type") == "audio.delta" and
                        e.get("response_id") == response_id, after, 240, "playback before teardown")
                    snapshot = runner.client.response_snapshot(response_id, after, False)
                    require(snapshot["audio"], "teardown had no PCM")
                    timer = threading.Timer(0.25, lambda rid=response_id: runner._late_callback(rid))
                    timer.start()
                    runner.client.close()
                    timer.join()
                    runner.reconnect()
                    turn = runner.client.send_user(runner.user)
                    response_id, after = runner.next_response(turn)
                elif scenario == "immediate_barge":
                    snapshot = runner.client.response_snapshot(response_id, after, False)
                    turn = runner.client.barge(snapshot, runner.user, 0, True)
                    response_id, after = runner.next_response(turn)
                elif scenario == "delayed_barge":
                    snapshot = runner.client.response_snapshot(response_id, after, True)
                    require(snapshot["positive_cursors"], "no non-final positive cursor")
                    cursor = runner.client.exercise_acks(snapshot, delay, False)
                    turn = runner.client.barge(snapshot, runner.user, cursor, False)
                    response_id, after = runner.next_response(turn)
                elif scenario == "final_ack":
                    snapshot = runner.client.response_snapshot(response_id, after, True)
                    runner._save_pcm(snapshot, round_index)
                    runner.client.exercise_acks(snapshot, delay, True)
                    turn = runner.client.send_user(runner.user)
                    response_id, after = runner.next_response(turn)
                runner.rows.append({"round": round_index, "scenario": scenario,
                                    "response_id": response_id, "connection": runner.client.name})
            snapshot = runner.client.response_snapshot(response_id, after, True)
            runner._save_pcm(snapshot, len(runner.scenarios))
            runner.client.exercise_acks(snapshot, 5, True)
            return runner

        # Seeded per-round start order is applied by staggered scheduling while
        # all six worker threads remain live and their sessions overlap.
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {}
            for round_index in range(args.rounds):
                # Execution itself owns all rounds; submit once in the first
                # round's deterministic order and retain every future.
                if round_index == 0:
                    for session in plan["rounds"][0]["start_order"]:
                        futures[session] = pool.submit(execute, runners[session])
            runners = [future.result() for future in futures.values()]
        clients = [runner.client for runner in runners]
        run["late_callback_count"] = sum(runner.late_callbacks for runner in runners)
        run["connections_per_session"] = {str(r.session): r.generation + 1 for r in runners}
        run["scenario_results"] = {str(r.session): r.rows for r in runners}
        run["ack_schedules"] = {client.name: client.ack_schedule
                                for runner in runners for client in runner.all_clients()}
        run["cancel_schedules"] = {client.name: client.cancel_schedule
                                   for runner in runners for client in runner.all_clients()}
        run["terminal_callback_counts"] = {
            client.name: {kind: client.terminal_callbacks.count(kind) for kind in
                          ("response.completed", "response.failed", "response.cancelled")}
            for runner in runners for client in runner.all_clients()}
        # Continue session 0 so the independent verifier sees every A/B wire frame.
        echo_client = runners[0].client
        run["echo_ab"] = echo_and_double_talk(echo_client, user, args.out)
        run["echo_ab"]["connection"] = echo_client.name
        run["resource_usage"] = sampler.finish()
        sampler = None
        require(run["resource_usage"]["server"]["rss_growth_bytes"] <=
                run["resource_bounds"]["server_rss_growth_max_bytes"],
                "server RSS growth exceeded queue gate")
        run["final_health"] = wait_health_zero(ports[0])
        print("Real capture complete; stopping only owned processes", flush=True)
    except Exception as error:
        run["driver_error"] = str(error)
        for name, proc in procs.items():
            run.setdefault("log_tails", {})[name] = proc.log.text()[-4000:]
        raise
    finally:
        for client in clients:
            try:
                client.close()
            except Exception:
                pass
        if sampler is not None:
            run["resource_usage"] = sampler.finish()
        if procs:
            stop_stack(procs)
            cleanup["owned_processes_exited"] = all(proc.proc.poll() is not None
                                                    for proc in procs.values())
        if observer is not None:
            observer.close()
        for name, proc in procs.items():
            shutil.copyfile(proc.log.path, args.out / f"{name}.log")
        for path in isolated:
            shutil.rmtree(path, ignore_errors=True)
        try:
            run["gpu_postflight"] = wait_postflight()
            cleanup["selected_gpus_idle"] = True
        except Exception as error:
            cleanup["error"] = str(error)
        (args.out / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    if "driver_error" in run:
        raise AssertionError(run["driver_error"])
    print("Running fresh independent raw-frame verifier", flush=True)
    subprocess.run([sys.executable, "-I", str(ROOT / "tests/realtime_network_verify.py"),
                    str(args.out), "--write"], check=True)
    verification = json.loads((args.out / "independent-verification.json").read_text())
    files = []
    for path in sorted(args.out.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            files.append(f"{hash_file(path)}  {path.relative_to(args.out)}\n")
    (args.out / "SHA256SUMS").write_text("".join(files))
    print(f"REALTIME_NETWORK PASS sessions=6 rounds={args.rounds} "
          f"responses={verification['responses']} prompts={verification['verified_next_turn_prompts']} "
          f"terminals={verification['terminal_callbacks']} files={len(files)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
