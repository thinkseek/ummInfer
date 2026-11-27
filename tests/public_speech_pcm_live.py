#!/usr/bin/env python3
"""Real public HTTP speech/native-PCM gate on physical GPU 0 only.

The driver owns one loopback gateway and serial native ASR/VLM/TTS stages. HTTP
coverage is text-only VLM -> TTS. The small realtime block exists only for the
server playback-ACK and delayed-reader lossless-queue contracts; it is not the
full ASR/R4 benchmark. Acceptance is delegated to a fresh independent verifier.
No physical audio device or perceptual-quality claim is made.
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import traceback
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SEED = 5630241
OPENING = "It sounds"
EXPECTED_NATIVE_CONTINUATION = "like you're in the middle of a thought or a"
CONTINUATION = (
    "It sounds like you're in the middle of a thought or a conversation. "
    "Did you want to continue that story, or is there something specific "
    "you'd like to discuss?"
)


def load_base():
    spec = importlib.util.spec_from_file_location(
        "umminfer_public_speech_base", ROOT / "tests/realtime_network_live.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base()
require = base.require

BACKPRESSURE_LOCK = threading.Lock()
BACKPRESSURE_RELEASE = threading.Event()
BACKPRESSURE = {"enabled": False, "native": 0, "public": 0, "paused": 0,
                "paused_ids": set(), "target": 0, "release": None}
PAUSE_ROWS = []


def backpressure_maybe_release():
    with BACKPRESSURE_LOCK:
        if not BACKPRESSURE["enabled"] or BACKPRESSURE_RELEASE.is_set():
            return
        pending = BACKPRESSURE["native"] - BACKPRESSURE["public"]
        if BACKPRESSURE["paused"] > 0 and pending >= 4096:
            BACKPRESSURE["release"] = {
                "monotonic_ns": time.monotonic_ns(), "native_audio_events": BACKPRESSURE["native"],
                "public_audio_events": BACKPRESSURE["public"],
                "pending_native_minus_public": pending,
                "paused_readers": BACKPRESSURE["paused"],
                "paused_response_ids": sorted(BACKPRESSURE["paused_ids"]),
                "configured_readers": BACKPRESSURE["target"],
                "bounded_queue_boundary": 4096}
            BACKPRESSURE_RELEASE.set()


class ForwardObserver:
    """Byte-preserving PUSH/PULL forwarding with raw retained messages."""

    def __init__(self, channel, public_endpoint, native_endpoint, output_path):
        import zmq
        self.channel = channel
        self.context = zmq.Context()
        self.public = self.context.socket(zmq.PULL)
        self.public.bind(public_endpoint)
        self.native = self.context.socket(zmq.PUSH)
        self.native.connect(native_endpoint)
        self.output = open(output_path, "xb", buffering=0)
        self.rows = 0
        self.error = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        try:
            while not self.stop_event.is_set() or self.public.poll(0):
                if not self.public.poll(100):
                    continue
                frames = self.public.recv_multipart()
                observed = time.monotonic_ns()
                self.native.send_multipart(frames)
                forwarded = time.monotonic_ns()
                for frame in frames:
                    row = {"index": self.rows, "channel": self.channel,
                           "observed_monotonic_ns": observed,
                           "forwarded_monotonic_ns": forwarded,
                           "raw_sha256": base.sha(frame),
                           "raw_base64": base64.b64encode(frame).decode()}
                    self.rows += 1
                    self.output.write((json.dumps(row, sort_keys=True) + "\n").encode())
                    if self.channel == "completion" and BACKPRESSURE["enabled"]:
                        import msgpack
                        message = msgpack.unpackb(frame, raw=False, strict_map_key=False)
                        chunk = message.get("chunk", {})
                        countable = (
                            message.get("type") == "stream" and
                            message.get("from_stage") == "tts" and
                            isinstance(chunk.get("audio"), bytes) and bool(chunk["audio"]))
                        if countable:
                            with BACKPRESSURE_LOCK:
                                BACKPRESSURE["native"] += 1
                            backpressure_maybe_release()
        except Exception as error:  # retained and checked at stack close
            self.error = error

    def close(self):
        self.stop_event.set()
        self.thread.join(15)
        require(not self.thread.is_alive(), f"observer did not drain: {self.channel}")
        self.output.close()
        self.public.close(0)
        self.native.close(0)
        self.context.term()
        if self.error:
            raise self.error


class ChunkedDecoder:
    def __init__(self):
        self.buffer = bytearray()
        self.output = bytearray()
        self.done = False

    def feed(self, data):
        produced_at = len(self.output)
        self.buffer.extend(data)
        while not self.done:
            marker = self.buffer.find(b"\r\n")
            if marker < 0:
                break
            line = bytes(self.buffer[:marker])
            try:
                size = int(line.split(b";", 1)[0], 16)
            except ValueError as error:
                raise AssertionError(f"invalid HTTP chunk size: {line!r}") from error
            needed = marker + 2 + size + 2
            if len(self.buffer) < needed:
                break
            payload = bytes(self.buffer[marker + 2:marker + 2 + size])
            require(self.buffer[marker + 2 + size:needed] == b"\r\n",
                    "invalid HTTP chunk terminator")
            del self.buffer[:needed]
            if size == 0:
                self.done = True
                require(not self.buffer, "unexpected bytes after final HTTP chunk")
            else:
                self.output.extend(payload)
        return bytes(self.output[produced_at:])


def http_case(port, out, spec):
    """Capture raw HTTP, seeded recv sizes, optional delayed read and abort."""
    case_dir = out / "http" / spec["name"]
    case_dir.mkdir(parents=True, exist_ok=False)
    payload = dict(spec["payload"])
    body = json.dumps(payload, separators=(",", ":")).encode()
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    sock.settimeout(0.25)
    request = (
        f"POST {spec.get('route', '/v1/chat/completions')} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode() + body
    (case_dir / "request.json").write_bytes(body + b"\n")
    raw_path = case_dir / "response.raw"
    reads_path = case_dir / "reads.jsonl"
    rng = random.Random(SEED + spec["numeric_seed"])
    raw = bytearray()
    reads = []
    abort_record = None
    pause_record = None
    started = time.monotonic_ns()
    try:
        sock.sendall(request)
        header = bytearray()
        while b"\r\n\r\n" not in header:
            width = rng.randint(1, 4093)
            try:
                block = sock.recv(width)
            except socket.timeout:
                reads.append({"index": len(reads), "requested_bytes": width,
                              "bytes": 0, "timeout": True,
                              "monotonic_ns": time.monotonic_ns(), "sha256": base.sha(b"")})
                continue
            require(block, "HTTP response closed before headers")
            header.extend(block)
            reads.append({"index": len(reads), "requested_bytes": width,
                          "bytes": len(block), "timeout": False,
                          "monotonic_ns": time.monotonic_ns(), "sha256": base.sha(block)})
        head, remainder = bytes(header).split(b"\r\n\r\n", 1)
        raw.extend(header)
        require(head.startswith(b"HTTP/1.1 200"), f"HTTP speech failed: {head!r}")
        chunked = b"transfer-encoding: chunked" in head.lower()
        decoder = ChunkedDecoder() if chunked else None
        decoded = bytearray()
        sse_buffer = bytearray()
        cancel_sent = False
        pause_done = False
        eof = False

        def consume(data):
            nonlocal cancel_sent, abort_record, pause_record, pause_done
            if not data:
                return
            decoded.extend(data)
            sse_buffer.extend(data)
            while b"\n\n" in sse_buffer:
                frame, _, rest = bytes(sse_buffer).partition(b"\n\n")
                sse_buffer[:] = rest
                if not frame.startswith(b"data: {"):
                    continue
                want_cancel = spec.get("cancel_after_artifact") is not None
                want_pause = spec.get("pause_after_artifact_ms") is not None
                if not want_cancel and not want_pause:
                    continue
                try:
                    event = json.loads(frame[6:])
                except json.JSONDecodeError:
                    continue
                unified = event.get("unified", {})
                if unified.get("kind") != "artifact":
                    continue
                if want_pause and not pause_done:
                    pause_done = True
                    before = time.monotonic_ns()
                    time.sleep(spec["pause_after_artifact_ms"] / 1000.0)
                    pause_record = {"start_monotonic_ns": before,
                                    "end_monotonic_ns": time.monotonic_ns(),
                                    "duration_ms": spec["pause_after_artifact_ms"]}
                if want_cancel and not cancel_sent:
                    cancel_sent = True
                    request_id = event["id"]
                    session_id = unified["session_id"]
                    abort_body = json.dumps({"request_id": request_id,
                                             "session_id": session_id}).encode()
                    abort_req = urllib.request.Request(
                        f"http://127.0.0.1:{port}/abort", data=abort_body,
                        headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(abort_req, timeout=10) as reply:
                        abort_reply = {"status": reply.status, "body": reply.read().decode()}
                    abort_record = {"request_id": request_id, "session_id": session_id,
                                    "monotonic_ns": time.monotonic_ns(), "reply": abort_reply}

        def receive(data):
            consume(decoder.feed(data) if chunked else data)

        receive(remainder)
        while not eof:
            try:
                width = rng.randint(1, 4093)
                block = sock.recv(width)
            except socket.timeout:
                reads.append({"index": len(reads), "requested_bytes": width,
                              "bytes": 0, "timeout": True,
                              "monotonic_ns": time.monotonic_ns(), "sha256": base.sha(b"")})
                continue
            arrival = time.monotonic_ns()
            if not block:
                eof = True
            else:
                raw.extend(block)
                receive(block)
            reads.append({"index": len(reads), "requested_bytes": width,
                          "bytes": len(block), "timeout": False,
                          "monotonic_ns": arrival, "sha256": base.sha(block)})
        if chunked:
            require(decoder.done, "chunked HTTP response lacked terminal chunk")
        (case_dir / "response.raw").write_bytes(raw)
        (case_dir / "response.body").write_bytes(decoded)
        (case_dir / "reads.jsonl").write_bytes(b"".join(
            (json.dumps(row, sort_keys=True) + "\n").encode() for row in reads))
        result = {"raw_sha256": base.sha(raw), "body_sha256": base.sha(decoded),
                  "raw_bytes": len(raw), "decoded_body_bytes": len(decoded),
                  "read_count": len(reads), "abort": abort_record, "pause": pause_record,
                  "started_monotonic_ns": started,
                  "finished_monotonic_ns": time.monotonic_ns(),
                  "elapsed_ns": time.monotonic_ns() - started}
        (case_dir / "capture.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result
    finally:
        sock.close()


def gpu_preflight():
    row = base.gpu_state(0)
    require(row["memory_used_mib"] <= 64 and row["utilization_percent"] == 0 and
            not row["compute_processes"], f"GPU 0 is not idle: {row}")
    return {"0": row}


def gpu_postflight(timeout=45):
    deadline = time.monotonic() + timeout
    latest = None
    while time.monotonic() < deadline:
        try:
            return gpu_preflight()
        except (AssertionError, subprocess.CalledProcessError) as error:
            latest = str(error)
            time.sleep(0.5)
    raise AssertionError(f"GPU 0 did not return to idle: {latest}")


def start_stack(args, out, logs):
    ports = base.free_ports(2)
    ipc = Path(tempfile.mkdtemp(prefix="umm-pcm63-ipc-", dir="/tmp"))
    shm = Path(tempfile.mkdtemp(prefix="umm-pcm63-shm-", dir="/tmp"))
    public = {name: f"ipc://{ipc}/{name}-public"
              for name in ("asr", "vlm", "tts", "completion")}
    native = {name: f"ipc://{ipc}/{name}-native"
              for name in ("asr", "vlm", "tts", "completion")}
    abort = f"ipc://{ipc}/abort"
    (out / "observers").mkdir(parents=True, exist_ok=False)
    observers = {}
    procs = {}
    old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    old_prefix = os.environ.get("UMM_QWEN38_PREFIX_CACHE")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["UMM_QWEN38_PREFIX_CACHE"] = "0"
    try:
        for channel, name in (("asr-submit", "asr"), ("vlm-submit", "vlm"),
                              ("tts-submit", "tts"), ("completion", "completion")):
            observers[channel] = ForwardObserver(
                channel, public[name], native[name], out / "observers" / f"{channel}.jsonl")
        procs["asr"] = base.OwnedProc("asr", [
            args.bin_dir / "umm-native-stage-asr", "--kind", "asr", "--name", "asr",
            "--model", args.asr_model, "--gpu", 0, "--recv-endpoint", native["asr"],
            "--coordinator-endpoint", public["completion"], "--abort-endpoint", abort],
            logs / "asr.log")
        procs["vlm"] = base.OwnedProc("vlm", [
            args.bin_dir / "umm-native-stage-qwen38", "--kind", "qwen38", "--name", "vlm",
            "--model", args.vlm_model, "--gpu", 0, "--recv-endpoint", native["vlm"],
            "--coordinator-endpoint", public["completion"], "--abort-endpoint", abort,
            "--max-new-tokens", 512, "--batch-size", 1], logs / "vlm.log")
        procs["tts"] = base.OwnedProc("tts", [
            args.bin_dir / "umm-native-stage-tts", "--kind", "tts", "--name", "tts",
            "--model", args.tts_model, "--gpu", 0, "--recv-endpoint", native["tts"],
            "--coordinator-endpoint", public["completion"], "--abort-endpoint", abort,
            "--max-frames", 192], logs / "tts.log")
        procs["server"] = base.OwnedProc("server", [
            args.bin_dir / "umm-infer", "serve", "--host", "127.0.0.1",
            "--port", ports[0], "--realtime-port", ports[1], "--model", "qwen3-unified",
            "--entry", "asr", "--entry-endpoint", public["asr"],
            "--vlm-entry", "vlm", "--vlm-entry-endpoint", public["vlm"],
            "--tts-entry", "tts", "--tts-entry-endpoint", public["tts"],
            "--completion", native["completion"], "--abort", abort,
            "--session-history", "--max-in-flight", 64, "--ipc-dir", shm], logs / "server.log")
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            for observer in observers.values():
                require(observer.error is None, f"observer failed: {observer.error}")
            for name, proc in procs.items():
                require(proc.proc.poll() is None, f"{name} exited during model load")
            ready = all(marker in procs[name].log.text() for name, marker in {
                "asr": "ready kind=asr", "vlm": "ready kind=qwen38",
                "tts": "ready kind=tts"}.items())
            if ready:
                try:
                    with urllib.request.urlopen(
                            f"http://127.0.0.1:{ports[0]}/ready", timeout=1) as reply:
                        if json.load(reply).get("status") == "ready":
                            return {"ports": ports, "procs": procs, "observers": observers,
                                    "isolated": (ipc, shm), "native_endpoints": native,
                                    "public_endpoints": public, "abort_endpoint": abort}
                except Exception:
                    pass
            time.sleep(0.25)
        raise AssertionError("task-owned real stages did not become ready")
    except Exception:
        if procs:
            base.stop_stack(procs)
        for observer in observers.values():
            observer.close()
        for path in (ipc, shm):
            shutil.rmtree(path, ignore_errors=True)
        raise
    finally:
        if old_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda
        if old_prefix is None:
            os.environ.pop("UMM_QWEN38_PREFIX_CACHE", None)
        else:
            os.environ["UMM_QWEN38_PREFIX_CACHE"] = old_prefix


def start_tts_process(args, stack, logs, name):
    return base.OwnedProc("tts", [
        args.bin_dir / "umm-native-stage-tts", "--kind", "tts", "--name", "tts",
        "--model", args.tts_model, "--gpu", 0,
        "--recv-endpoint", stack["native_endpoints"]["tts"],
        "--coordinator-endpoint", stack["public_endpoints"]["completion"],
        "--abort-endpoint", stack["abort_endpoint"], "--max-frames", 192], logs / name)


class PausableRawWebSocket(base.RawWebSocket):
    pause_after_first_audio_ms = 0
    heartbeat_interval_s = 30.0

    def __init__(self, *args):
        self.pause_done = False
        super().__init__(*args)

    def wait_for_release_with_heartbeat(self, timeout_s):
        stop = threading.Event()

        def heartbeat():
            while not stop.wait(self.heartbeat_interval_s):
                try:
                    self.send_frame(9, b"")
                except Exception:
                    return

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            require(BACKPRESSURE_RELEASE.wait(timeout_s),
                    "timed out waiting for a proven 4096-event public/native backlog")
        finally:
            stop.set()
            thread.join(2)
            require(not thread.is_alive(), "realtime heartbeat did not stop")

    def record(self, direction, raw):
        row = super().record(direction, raw)
        if BACKPRESSURE["enabled"] and direction == "rx" and b"audio.delta" in raw:
            with BACKPRESSURE_LOCK:
                BACKPRESSURE["public"] += 1
        pause_ms = self.pause_after_first_audio_ms
        if direction == "rx" and pause_ms > 0 and not self.pause_done and b"audio.delta" in raw:
            self.pause_done = True
            if BACKPRESSURE_RELEASE.is_set():
                return row
            start = time.monotonic_ns()
            match = re.search(br'"response_id":"([^"]+)"', raw)
            require(match is not None, "paused audio.delta lacks response_id")
            response_id = match.group(1).decode()
            with BACKPRESSURE_LOCK:
                BACKPRESSURE["paused"] += 1
                BACKPRESSURE["paused_ids"].add(response_id)
            backpressure_maybe_release()
            self.wait_for_release_with_heartbeat(pause_ms / 1000.0)
            end = time.monotonic_ns()
            with BACKPRESSURE_LOCK:
                PAUSE_ROWS.append({"capture": str(self.capture_path),
                                   "response_id": response_id,
                                   "capture_index": row["index"],
                                   "start_monotonic_ns": start,
                                   "end_monotonic_ns": end,
                                   "duration_ms": (end - start) / 1_000_000})
        return row


def realtime_response_snapshot(client, response_id, after, timeout_s=1800):
    """Base snapshot contract with a deadline compatible with delayed readers."""
    terminal = client.ws.wait_for(
        lambda event: event.get("response_id") == response_id and
        event.get("type") in ("response.completed", "response.failed"),
        after, timeout_s, "real response terminal")
    require(terminal["event"]["type"] == "response.completed",
            f"real response failed: {terminal['event']}")
    for record in client.ws.since(
            after, types={"response.completed", "response.failed", "response.cancelled"},
            response_id=response_id):
        client.terminal_callbacks.append(record["event"]["type"])
    records = client.ws.since(after, response_id=response_id)
    audio = [record for record in records if record["event"]["type"] == "audio.delta"]
    text = "".join(record["event"].get("text", "") for record in records
                   if record["event"]["type"] == "response.text.delta")
    pcm = b"".join(base64.b64decode(record["event"]["audio"]) for record in audio)
    cursors = sorted({int(record["event"].get("text_cursor", 0)) for record in audio})
    positive = [cursor for cursor in cursors if 0 < cursor < len(text.encode())]
    return {"response_id": response_id, "after": after, "terminal": terminal,
            "records": records, "audio": audio, "text": text, "pcm": pcm,
            "pcm_sha256": base.sha(pcm), "positive_cursors": positive}


def send_full_realtime_user(client, audio):
    """Stream the full fixture; base send_user intentionally caps R4 at two seconds."""
    marker = client.ws.mark()
    started = client.stream(audio + base.END_SILENCE)
    completed = client.ws.wait_for(
        lambda event: event.get("type") == "transcription.completed",
        marker, 300, "full real ASR instruction")
    require(completed["event"].get("text", "").strip(), "empty real ASR transcript")
    return {"marker": marker, "completed": completed, "started_ns": started}


def make_realtime_clients(args, root, ports):
    previous_raw = base.RawWebSocket
    base.RawWebSocket = PausableRawWebSocket
    try:
        return [base.LiveClient(session, 0, ports[1], root)
                for session in range(args.realtime_sessions)]
    finally:
        base.RawWebSocket = previous_raw


def realtime_block(args, out, logs, ports, audio, stack):
    root = out / "realtime"
    root.mkdir(parents=True, exist_ok=False)
    BACKPRESSURE_RELEASE.clear()
    PAUSE_ROWS.clear()
    with BACKPRESSURE_LOCK:
        BACKPRESSURE.update({"enabled": True, "native": 0, "public": 0, "paused": 0,
                             "paused_ids": set(), "target": args.realtime_sessions,
                             "release": None})
    clients = make_realtime_clients(args, root, ports)
    for client in clients:
        marker = client.ws.mark()
        client.ws.send_json({"type": "session.update",
                             "session": {"max_new_tokens": 512}})
        updated = client.ws.wait_for(
            lambda event: event.get("type") == "transcription_session.updated",
            marker, 20, "realtime generation-limit update")
        require(updated["event"]["session"].get("max_new_tokens") == 512,
                "realtime generation-limit update was not accepted")
    identities = []
    restart_cancel = threading.Event()

    def restart_tts():
        if restart_cancel.wait(args.tts_restart_delay_s):
            return
        process = start_tts_process(args, stack, logs, "tts-restart.log")
        stack["procs"]["tts"] = process
        stack["tts_restart_identity"] = process.identity()

    restart_thread = threading.Thread(target=restart_tts, daemon=True)
    restart_thread.start()
    try:
        for round_index in range(1):
            PausableRawWebSocket.pause_after_first_audio_ms = (
                args.realtime_pause_ms if round_index == 0 else 0)

            failure_lock = threading.Lock()
            failure_path = root / "turn-failures.jsonl"

            def turn(pair):
                session, client = pair
                phase = "start"
                try:
                    # Retain a fresh response id for every measured repeat. Start jitter
                    # is seeded; no slow/failed response is discarded.
                    time.sleep(((session * 7 + round_index * 11) % 17) / 1000.0)
                    phase = "send_full_realtime_user"
                    sent = send_full_realtime_user(client, audio)
                    phase = "response_identity"
                    response_id, after = client.response_identity(
                        sent["completed"]["event_index"] + 1)
                    phase = "response_snapshot"
                    snapshot = realtime_response_snapshot(client, response_id, after)
                    phase = "exercise_acks"
                    cursor = client.exercise_acks(
                        snapshot, 250 if round_index == 0 else 5, final=True)
                    return {"round": round_index, "session": session,
                            "response_id": response_id, "after": after,
                            "transcript": sent["completed"]["event"].get("text"),
                            "capture": str(client.ws.capture_path),
                            "acknowledged_partial_cursor": cursor,
                            "text_sha256": base.sha(snapshot["text"].encode()),
                            "pcm_sha256": base.sha(snapshot["pcm"])}
                except Exception as error:
                    with failure_lock, open(failure_path, "ab", buffering=0) as output:
                        output.write((json.dumps({
                            "session": session, "round": round_index, "phase": phase,
                            "monotonic_ns": time.monotonic_ns(),
                            "error": str(error), "traceback": traceback.format_exc()},
                            sort_keys=True) + "\n").encode())
                    raise

            with ThreadPoolExecutor(max_workers=args.realtime_sessions) as pool:
                identities.extend(pool.map(turn, enumerate(clients)))
    finally:
        restart_cancel.set()
        restart_thread.join(15)
        require(not restart_thread.is_alive(), "delayed TTS restart did not finish")
        for client in clients:
            client.ws.close()
        with BACKPRESSURE_LOCK:
            BACKPRESSURE["enabled"] = False
            release_snapshot = BACKPRESSURE["release"] or {
                "native_audio_events": BACKPRESSURE["native"],
                "public_audio_events": BACKPRESSURE["public"],
                "pending_native_minus_public": BACKPRESSURE["native"] - BACKPRESSURE["public"],
                "paused_readers": BACKPRESSURE["paused"],
                "paused_response_ids": sorted(BACKPRESSURE["paused_ids"]),
                "configured_readers": BACKPRESSURE["target"],
                "bounded_queue_boundary": 4096}
        (root / "backpressure-release.json").write_text(
            json.dumps(release_snapshot, indent=2, sort_keys=True) + "\n")
    failure_path.touch()
    (root / "responses.json").write_text(
        json.dumps(identities, indent=2, sort_keys=True) + "\n")
    (root / "read-pauses.json").write_text(
        json.dumps(PAUSE_ROWS, indent=2, sort_keys=True) + "\n")
    (root / "fixture.sha256").write_text(base.hash_file(args.short_audio) + "\n")
    return {"responses": len(identities), "paused_readers": len(PAUSE_ROWS)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--asr-model", type=Path, required=True)
    parser.add_argument("--vlm-model", type=Path, required=True)
    parser.add_argument("--tts-model", type=Path, required=True)
    parser.add_argument("--short-audio", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--realtime-pause-ms", type=int, default=1800000)
    parser.add_argument("--realtime-sessions", type=int, default=29)
    parser.add_argument("--tts-restart-delay-s", type=float, default=120.0)
    parser.add_argument("--skip-realtime", action="store_true")
    args = parser.parse_args()
    require(1 <= args.realtime_sessions <= 64, "realtime sessions must be within existing server capacity")
    require(args.tts_restart_delay_s >= 0, "TTS restart delay must be nonnegative")
    args.out.mkdir(parents=True, exist_ok=False)
    args.log_dir.mkdir(parents=True, exist_ok=False)
    require(args.short_audio.is_file(), "realtime fixture is missing")
    with open(args.short_audio, "rb") as source:
        import wave
        with wave.open(source, "rb") as wav:
            require((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) ==
                    (1, 2, 16000), "realtime fixture must be PCM16 mono 16 kHz")
            audio = wav.readframes(wav.getnframes())
    manifest = {
        "schema": "umminfer-public-speech-pcm-capture/v1",
        "source_commit": args.source_commit, "seed": SEED,
        "prompts": {"opening": OPENING, "continuation": CONTINUATION},
        "limits": {"full_asr_two_gpu_r4": "NOT_RUN",
                   "physical_playback": "NOT_RUN",
                   "perceptual_quality": "NOT_RUN"},
        "gpu_preflight": gpu_preflight(), "http_cases": [],
    }
    (args.out / "source").mkdir()
    for path in (ROOT / "tests/public_speech_pcm_live.py",
                 ROOT / "tests/public_speech_pcm_verify.py"):
        shutil.copyfile(path, args.out / "source" / path.name)
    manifest["source_sha256"] = {
        path.name: base.hash_file(path) for path in
        (ROOT / "tests/public_speech_pcm_live.py", ROOT / "tests/public_speech_pcm_verify.py")}
    manifest["binaries"] = {
        name: base.hash_file(args.bin_dir / name) for name in
        ("umm-infer", "umm-native-stage-asr", "umm-native-stage-qwen38",
         "umm-native-stage-tts")}
    manifest["models"] = base.hash_models(args, args.out)
    stack = None
    try:
        stack = start_stack(args, args.out, args.log_dir)
        ports = stack["ports"]
        forms = (("sse-pcm", True, True), ("sse-wav", True, False),
                 ("json-wav", False, False))
        numeric_seed = 0
        for prompt_name in ("opening", "continuation"):
            prompt = (f"Reply with exactly this text and no other text: {manifest['prompts'][prompt_name]}")
            for form, outer_stream, audio_stream in forms:
                for repeat in range(3):
                    numeric_seed += 1
                    name = f"{prompt_name}-{form}-r{repeat}"
                    payload = {"model": "qwen3-unified", "stream": outer_stream,
                               "messages": [{"role": "user", "content": prompt}],
                               "modalities": ["text", "audio"],
                               "audio": {"voice": "vivian", "format": "wav",
                                         "sample_rate": 24000, "stream": audio_stream},
                               "session_id": f"public-{name}",
                               "temperature": 0, "top_p": 1, "seed": SEED,
                               "max_tokens": 96}
                    spec = {"name": name, "kind": "http", "prompt": prompt_name,
                            "form": form, "repeat": repeat, "repeat_group": name[:-2],
                            "payload": payload, "numeric_seed": numeric_seed}
                    capture = http_case(ports[0], args.out, spec)
                    spec = {key: value for key, value in spec.items() if key != "payload"}
                    spec["capture"] = capture
                    manifest["http_cases"].append(spec)
                    base.wait_health_zero(ports[0], 60)
        # One delayed HTTP reader and a cancel/retry pair are additional, not
        # replacements for the two fresh repeats above.
        payload = {"model": "qwen3-unified", "stream": True,
                   "messages": [{"role": "user", "content":
                                 f"Reply with exactly this text and no other text: {CONTINUATION}"}],
                   "modalities": ["text", "audio"],
                   "audio": {"voice": "vivian", "format": "wav", "sample_rate": 24000,
                             "stream": True},
                   "temperature": 0, "top_p": 1, "seed": SEED, "max_tokens": 96}
        for name, extra in (("continuation-sse-pcm-delayed",
                             {"pause_after_artifact_ms": 3000}),
                            ("continuation-sse-pcm-cancel",
                             {"cancel_after_artifact": 1}),
                            ("continuation-sse-pcm-retry", {})):
            numeric_seed += 1
            request_payload = dict(payload)
            request_payload["session_id"] = f"public-{name}"
            spec = {"name": name, "kind": "http", "prompt": "continuation",
                    "form": "sse-pcm", "repeat": "delayed" if "delayed" in name else name.rsplit("-", 1)[-1],
                    "repeat_group": "continuation-sse-pcm",
                    "payload": request_payload, "numeric_seed": numeric_seed, **extra}
            capture = http_case(ports[0], args.out, spec)
            spec = {key: value for key, value in spec.items() if key != "payload"}
            spec["capture"] = capture
            manifest["http_cases"].append(spec)
            base.wait_health_zero(ports[0], 60)
        # Exact-text public speech control for the accepted Task #59 continuation.
        # It does not rewrite or hide the VLM's independently retained prose.
        for repeat in range(3):
            numeric_seed += 1
            name = f"direct-continuation-pcm-r{repeat}"
            spec = {"name": name, "kind": "http", "prompt": "native-continuation",
                    "form": "direct-pcm", "repeat": repeat,
                    "repeat_group": "direct-continuation-pcm", "route": "/v1/audio/speech",
                    "payload": {"model": "qwen3-unified", "input": EXPECTED_NATIVE_CONTINUATION,
                                "voice": "vivian", "response_format": "pcm"},
                    "numeric_seed": numeric_seed}
            capture = http_case(ports[0], args.out, spec)
            spec = {key: value for key, value in spec.items() if key != "payload"}
            spec["capture"] = capture
            manifest["http_cases"].append(spec)
            base.wait_health_zero(ports[0], 60)
        numeric_seed += 1
        spec = {"name": "direct-continuation-wav-r0", "kind": "http",
                "prompt": "native-continuation", "form": "direct-wav", "repeat": 0,
                "repeat_group": "direct-continuation-wav", "route": "/v1/audio/speech",
                "payload": {"model": "qwen3-unified", "input": EXPECTED_NATIVE_CONTINUATION,
                            "voice": "vivian", "response_format": "wav"},
                "numeric_seed": numeric_seed}
        capture = http_case(ports[0], args.out, spec)
        spec = {key: value for key, value in spec.items() if key != "payload"}
        spec["capture"] = capture
        manifest["http_cases"].append(spec)
        base.wait_health_zero(ports[0], 60)
        if not args.skip_realtime:
            initial_tts = stack["procs"].pop("tts")
            initial_identity = initial_tts.identity()
            initial_tts.stop()
            require(initial_tts.exit_code in (0, -15),
                    f"initial TTS did not stop cleanly: {initial_tts.exit_code}")
            time.sleep(1.0)  # Let the existing stage-down monitor gate new TTS submits.
            manifest["tts_restart"] = {"delay_s": args.tts_restart_delay_s,
                                       "initial": initial_identity}
            manifest["realtime"] = realtime_block(
                args, args.out, args.log_dir, ports, audio, stack)
            if "tts_restart_identity" in stack:
                manifest["tts_restart"]["restarted"] = stack["tts_restart_identity"]
        else:
            manifest["realtime"] = {"status": "NOT_RUN"}
        manifest["final_health"] = base.wait_health_zero(ports[0], 60)
        manifest["owned_processes"] = [proc.identity() for proc in stack["procs"].values()]
        base.stop_stack(stack["procs"])
        for proc in stack["procs"].values():
            require(proc.exit_code in (0, -15), f"owned process exit: {proc.name}={proc.exit_code}")
        for observer in stack["observers"].values():
            observer.close()
        (args.out / "owned-exit.json").write_text(json.dumps({
            name: proc.exit_code for name, proc in stack["procs"].items()},
            indent=2, sort_keys=True) + "\n")
        for path in stack["isolated"]:
            shutil.rmtree(path, ignore_errors=True)
        stack = None
        manifest["cleanup"] = {"owned_processes_exited": True}
        manifest["gpu_postflight"] = gpu_postflight()
        shutil.copytree(args.log_dir, args.out / "logs")
        manifest["status"] = "CAPTURED"
        (args.out / "capture-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        reports = []
        for run in (1, 2):
            report = args.out / f"verification-r{run}.json"
            subprocess.run([sys.executable, "-I", str(ROOT / "tests/public_speech_pcm_verify.py"),
                            "--root", str(args.out), "--output", str(report)], check=True)
            reports.append(base.hash_file(report))
        require(reports[0] == reports[1], "independent verification was not reproducible")
        files = sorted(path for path in args.out.rglob("*")
                       if path.is_file() and path.name != "SHA256SUMS")
        (args.out / "SHA256SUMS").write_text("".join(
            f"{base.hash_file(path)}  {path.relative_to(args.out)}\n" for path in files))
        evidence_sha = base.hash_file(args.out / "SHA256SUMS")
        print(f"PUBLIC_SPEECH_PCM PASS http={len(manifest['http_cases'])} "
              f"realtime={manifest['realtime'].get('responses', 0)} "
              f"verification_sha256={reports[0]} files={len(files)} "
              f"evidence_sha256={evidence_sha}")
    finally:
        if stack is not None:
            try:
                base.stop_stack(stack["procs"])
                (args.out / "owned-exit.json").write_text(json.dumps({
                    name: proc.exit_code for name, proc in stack["procs"].items()},
                    indent=2, sort_keys=True) + "\n")
            finally:
                for observer in stack["observers"].values():
                    observer.close()
        if stack is not None:
            for path in stack["isolated"]:
                shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    main()
