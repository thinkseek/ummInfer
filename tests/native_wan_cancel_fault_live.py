#!/usr/bin/env python3
"""Real public/native Wan2.2 cancellation and fault-recovery gate.

The CPU echo stages only satisfy the existing public UnifiedEngine text path.
Every generated video comes from the real native C++/CUDA Wan2.2 stage. No
contract-stub or fabricated media is accepted. Run with --self-check for the
stdlib-only deterministic checks; the live gate requires UMMINFER_LIVE_OPT_IN=1
and an idle physical GPU 7.
"""

import argparse
import base64
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

FORMAT = "umminfer-native-wan-cancel-fault-v1"
CONTROL = {
    "size": "352x256", "num_frames": 9, "steps": 4, "fps": 16,
    "guidance_scale": 5.0, "seed": 42, "output_format": "mp4",
    "prompt": "A red lantern sways gently above a misty blue mountain lake",
}
DENOISE = dict(CONTROL, size="704x480", num_frames=29, steps=20)
ARTIFACT = dict(CONTROL, size="1280x704", num_frames=77, steps=1)
TERMINAL_KINDS = {"complete", "error", "cancelled"}


def clock_ns():
    return time.monotonic_ns()


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def parse_sse_frame(raw):
    """Return a list of data payloads from one exact raw SSE frame."""
    payloads = []
    for line in raw.replace(b"\r\n", b"\n").split(b"\n"):
        if line.startswith(b"data:"):
            value = line[5:]
            if value.startswith(b" "):
                value = value[1:]
            payloads.append(value)
    return payloads


def parse_abort_log(line):
    match = re.search(
        r'video request aborted .*?request_id=(\S+).*?boundary="([^"]+)"', line)
    if not match:
        return None
    result = {"request_id": match.group(1), "boundary": match.group(2), "raw": line}
    for field in ("abort_epoch_ms", "exit_epoch_ms", "compute_ms"):
        value = re.search(rf"\b{field}=(\d+)", line)
        if value:
            result[field] = int(value.group(1))
    return result


def terminal_events(events):
    return [event for event in events if event.get("kind") in TERMINAL_KINDS]


def validate_event_order(events):
    """Offline/shared structural checks; media and native joins are separate."""
    terminals = terminal_events(events)
    require(len(terminals) == 1, f"expected one public terminal, got {len(terminals)}")
    terminal_index = events.index(terminals[0])
    require(not events[terminal_index + 1:], "late public event after terminal")
    for index, event in enumerate(events):
        if "seq" in event:
            require(event["seq"] == index,
                    f"unified sequence gap at {index}: {event.get('seq')}")
    return terminals[0]


def http_json(port, method, path, body=None, timeout=30):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        payload = None if body is None else canonical(body)
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        before = clock_ns()
        conn.request(method, path, body=payload, headers=headers)
        response = conn.getresponse()
        raw = response.read()
        after = clock_ns()
        parsed = json.loads(raw) if raw else {}
        return response.status, parsed, before, after
    finally:
        conn.close()


def http_text(port, path, timeout=30):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        raw = response.read()
        require(response.status == 200, f"GET {path}: HTTP {response.status}: {raw!r}")
        return raw.decode("utf-8")
    finally:
        conn.close()


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def gpu7_snapshot(nvidia_smi):
    result = subprocess.run(
        [nvidia_smi, "-i", "7", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"], check=True, capture_output=True,
        text=True, timeout=20)
    fields = [item.strip() for item in result.stdout.strip().split(",")]
    require(len(fields) == 5 and int(fields[0]) == 7,
            f"unexpected physical GPU 7 response: {result.stdout!r}")
    return {"index": 7, "name": fields[1], "memory_used_mib": int(fields[2]),
            "memory_total_mib": int(fields[3]), "utilization_percent": int(fields[4]),
            "observed_monotonic_ns": clock_ns()}


def gpu7_processes(nvidia_smi):
    result = subprocess.run(
        [nvidia_smi, "-i", "7", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"], check=True, capture_output=True,
        text=True, timeout=20)
    records = []
    for line in result.stdout.splitlines():
        if line.strip():
            pid, memory = [item.strip() for item in line.split(",")]
            records.append({"pid": int(pid), "used_memory_mib": int(memory)})
    return records


def require_gpu7_idle(nvidia_smi):
    snapshot = gpu7_snapshot(nvidia_smi)
    processes = gpu7_processes(nvidia_smi)
    require(snapshot["memory_used_mib"] <= 16 and not processes,
            f"physical GPU 7 is not idle: {snapshot} {processes}")
    return {"snapshot": snapshot, "compute_processes": processes}


class GpuMonitor:
    def __init__(self, nvidia_smi):
        self.nvidia_smi = nvidia_smi
        self.samples = []
        self.error = None
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        try:
            while not self.stopping.wait(0.05):
                self.samples.append(gpu7_snapshot(self.nvidia_smi))
        except Exception as error:
            self.error = str(error)

    def stop(self):
        self.stopping.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=10)
        require(not self.thread.is_alive() and self.error is None,
                f"GPU monitor failed: {self.error}")


class ManagedProcess:
    def __init__(self, name, command, log_path, env):
        self.name = name
        self.command = [str(item) for item in command]
        self.log_path = Path(log_path)
        self.start_ns = clock_ns()
        self.ready_ns = None
        self.stop_ns = None
        self.stop_error = None
        self.returncode = None
        self.log = self.log_path.open("wb")
        self.process = subprocess.Popen(
            self.command, stdout=self.log, stderr=subprocess.STDOUT, env=env,
            start_new_session=True)
        self.pid = self.process.pid
        self.proc_start_ticks = proc_start_ticks(self.pid)

    def stop(self, shutdown_endpoint=None, timeout=45):
        if self.stop_ns is not None:
            return
        context = push = None
        try:
            if self.process.poll() is None and shutdown_endpoint and self.ready_ns:
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
        return {"name": self.name, "pid": self.pid,
                "proc_start_ticks": self.proc_start_ticks,
                "argv_sha256": sha256_bytes("\0".join(self.command).encode()),
                "start_monotonic_ns": self.start_ns, "ready_monotonic_ns": self.ready_ns,
                "stop_monotonic_ns": self.stop_ns, "returncode": self.returncode,
                "stop_error": self.stop_error, "log": self.log_path.name}


def proc_start_ticks(pid):
    # /proc stat field 22; the final ')' separates the parenthesized comm field.
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    return int(raw[raw.rfind(")") + 2:].split()[19])


def native_ffmpeg_children(wan_pid):
    children = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="ascii")
            end = raw.rfind(")")
            comm = raw[raw.find("(") + 1:end]
            fields = raw[end + 2:].split()
            if comm == "ffmpeg" and int(fields[1]) == wan_pid:
                children.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
    return children


class TapProxy:
    """Ordered read-only MessagePack forwarding with raw-byte retention."""
    def __init__(self, name, listen, destination, evidence_dir):
        self.name = name
        self.listen = listen
        self.destination = destination
        self.evidence_dir = Path(evidence_dir)
        self.records = []
        self.error = None
        self.condition = threading.Condition()
        self.stopping = False
        self.thread = threading.Thread(target=self._run, daemon=True)
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
            while not self.stopping:
                try:
                    packed = pull.recv(zmq.NOBLOCK)
                except zmq.Again:
                    time.sleep(0.002)
                    continue
                tick = clock_ns()
                message = msgpack.unpackb(packed, raw=False, strict_map_key=False)
                request_id = message.get("request_id")
                message_type = message.get("type")
                digest = sha256_bytes(packed)
                filename = f"{self.name}-{len(self.records):05d}-{digest}.msgpack"
                (self.evidence_dir / filename).write_bytes(packed)
                record = {"tap": self.name, "index": len(self.records),
                          "message_type": message_type, "request_id": request_id,
                          "from_stage": message.get("from_stage"),
                          "observed_monotonic_ns": tick, "packed_file": filename,
                          "packed_sha256": digest, "packed_bytes": len(packed)}
                if message_type == "complete":
                    result = message.get("result") or {}
                    record.update({"success": bool(message.get("success")),
                                   "error": message.get("error")})
                    if isinstance(result.get("data"), bytes):
                        record.update({"media_sha256": sha256_bytes(result["data"]),
                                       "media_bytes": len(result["data"])})
                forwarded = False
                while not forwarded and not self.stopping:
                    try:
                        push.send(packed, zmq.NOBLOCK)
                        forwarded = True
                    except zmq.Again:
                        time.sleep(0.002)
                if not forwarded:
                    break
                record["forwarded_monotonic_ns"] = clock_ns()
                with self.condition:
                    self.records.append(record)
                    self.condition.notify_all()
        except Exception as error:
            self.error = str(error)
            with self.condition:
                self.condition.notify_all()
        finally:
            if pull is not None:
                pull.close(0)
            if push is not None:
                push.close(0)
            if context is not None:
                context.term()

    def snapshot(self):
        with self.condition:
            return [dict(record) for record in self.records]

    def wait(self, predicate, timeout=120):
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                require(self.error is None, f"{self.name} tap failed: {self.error}")
                for record in self.records:
                    if predicate(record):
                        return dict(record)
                remaining = deadline - time.monotonic()
                require(remaining > 0, f"timed out waiting for {self.name} message")
                self.condition.wait(min(remaining, 0.1))

    def stop(self):
        self.stopping = True
        self.thread.join(timeout=10)
        require(not self.thread.is_alive() and self.error is None,
                f"{self.name} tap did not stop cleanly: {self.error}")


class PublicRequest:
    def __init__(self, label, port, body, out_dir):
        self.label = label
        self.port = port
        self.body = body
        self.out_dir = Path(out_dir)
        self.request_id = None
        self.session_id = body["session_id"]
        self.http_status = None
        self.events = []
        self.raw_chunks = []
        self.frames = []
        self.done_count = 0
        self.error = None
        self.submit_ns = None
        self.end_ns = None
        self.condition = threading.Condition()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.artifacts = {}
        self.chunk_dir = self.out_dir / f"{self.label}.chunks"
        self.chunk_dir.mkdir()
        (self.out_dir / f"{self.label}.request.json").write_bytes(canonical(body) + b"\n")

    def start(self):
        self.thread.start()
        return self

    def _notify(self):
        with self.condition:
            self.condition.notify_all()

    def _run(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=300)
        buffer = b""
        try:
            self.submit_ns = clock_ns()
            conn.request("POST", "/v1/chat/completions", body=canonical(self.body),
                         headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            self.http_status = response.status
            self._notify()
            while True:
                chunk = response.read1(65536)
                if not chunk:
                    break
                tick = clock_ns()
                chunk_name = f"{self.label}.chunks/{len(self.raw_chunks):05d}.bin"
                (self.out_dir / chunk_name).write_bytes(chunk)
                self.raw_chunks.append({"file": chunk_name,
                                        "offset": sum(item["bytes"] for item in self.raw_chunks),
                                        "bytes": len(chunk), "sha256": sha256_bytes(chunk),
                                        "observed_monotonic_ns": tick})
                buffer += chunk
                while True:
                    match = re.search(rb"\r?\n\r?\n", buffer)
                    if not match:
                        break
                    raw_frame = buffer[:match.end()]
                    buffer = buffer[match.end():]
                    frame_tick = clock_ns()
                    frame_index = len(self.frames)
                    self.frames.append({"index": frame_index, "bytes": len(raw_frame),
                                        "sha256": sha256_bytes(raw_frame),
                                        "observed_monotonic_ns": frame_tick})
                    for payload in parse_sse_frame(raw_frame):
                        if payload == b"[DONE]":
                            self.done_count += 1
                            continue
                        outer = json.loads(payload)
                        outer_request_id = outer.get("id")
                        if self.request_id is None and isinstance(outer_request_id, str):
                            self.request_id = outer_request_id
                            self._notify()
                        unified = outer.get("unified")
                        if not isinstance(unified, dict):
                            continue
                        request_id = outer.get("id") or unified.get("request_id")
                        if self.request_id is None and isinstance(request_id, str):
                            self.request_id = request_id
                            self._notify()
                        event = dict(unified)
                        observed = {"label": self.label, "event_index": len(self.events),
                                    "frame_index": frame_index,
                                    "observed_monotonic_ns": frame_tick, "event": event}
                        artifact = event.get("artifact")
                        if event.get("kind") == "artifact" and isinstance(artifact, dict):
                            raw = base64.b64decode(artifact["data"], validate=True)
                            key = artifact.get("stage") or event.get("stage") or "unknown"
                            self.artifacts.setdefault(key, []).append((artifact, raw))
                            observed["artifact_data_sha256"] = sha256_bytes(raw)
                            observed["artifact_data_bytes"] = len(raw)
                        self.events.append(event)
                        with (self.out_dir / f"{self.label}.events.jsonl").open(
                                "a", encoding="utf-8") as handle:
                            handle.write(json.dumps(observed, sort_keys=True,
                                                    separators=(",", ":")) + "\n")
                        self._notify()
            require(self.http_status == 200, f"{self.label}: HTTP {self.http_status}")
            require(self.request_id is not None, f"{self.label}: no public request id")
            require(not buffer, f"{self.label}: truncated SSE frame")
        except Exception as error:
            self.error = str(error)
            self._notify()
        finally:
            conn.close()
            self.end_ns = clock_ns()
            self._write_capture_files()

    def _write_capture_files(self):
        record = {"label": self.label, "session_id": self.session_id,
                  "request_id": self.request_id, "http_status": self.http_status,
                  "submit_monotonic_ns": self.submit_ns, "end_monotonic_ns": self.end_ns,
                  "done_count": self.done_count, "error": self.error,
                  "request_sha256": sha256_bytes(canonical(self.body)),
                  "events": len(self.events), "frames": self.frames,
                  "chunks": self.raw_chunks}
        raw_path = self.out_dir / f"{self.label}.sse"
        with raw_path.open("wb") as handle:
            for chunk in self.raw_chunks:
                handle.write((self.out_dir / chunk["file"]).read_bytes())
        record["raw_file"] = raw_path.name
        record["raw_bytes"] = raw_path.stat().st_size
        record["raw_sha256"] = sha256_file(raw_path)
        (self.out_dir / f"{self.label}.capture.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def wait_request_id(self, timeout=30):
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.request_id is None:
                require(self.error is None, f"{self.label}: {self.error}")
                remaining = deadline - time.monotonic()
                require(remaining > 0 and self.thread.is_alive(),
                        f"{self.label}: request id unavailable")
                self.condition.wait(min(remaining, 0.1))
            return self.request_id

    def join(self, timeout=300):
        self.thread.join(timeout=timeout)
        require(not self.thread.is_alive(), f"{self.label}: public stream did not finish")
        require(self.error is None, f"{self.label}: {self.error}")
        return self

    def abort(self):
        request_id = self.wait_request_id()
        send_wall_ns = time.time_ns()
        status, body, send_ns, ack_ns = http_json(
            self.port, "POST", "/abort",
            {"request_id": request_id, "session_id": self.session_id}, timeout=30)
        ack_wall_ns = time.time_ns()
        require(send_ns <= ack_ns and (ack_ns - send_ns) <= 1_000_000_000,
                f"{self.label}: abort acknowledgment exceeded 1 s")
        return {"request_id": request_id, "http_status": status, "body": body,
                "send_monotonic_ns": send_ns, "ack_monotonic_ns": ack_ns,
                "send_wall_time_ns": send_wall_ns, "ack_wall_time_ns": ack_wall_ns,
                "ack_latency_ms": (ack_ns - send_ns) / 1e6}

    def summarize(self, expected_terminal):
        terminal = validate_event_order(self.events)
        require(self.done_count == 1, f"{self.label}: expected one [DONE]")
        require(terminal.get("kind") == expected_terminal,
                f"{self.label}: expected {expected_terminal}, got {terminal}")
        require(all(event.get("session_id") == self.session_id for event in self.events),
                f"{self.label}: stale session event")
        terminal_tick = None
        if self.events:
            lines = (self.out_dir / f"{self.label}.events.jsonl").read_text(
                encoding="utf-8").splitlines()
            terminal_tick = json.loads(lines[-1])["observed_monotonic_ns"]
        media = None
        chunks = self.artifacts.get("video", [])
        if chunks:
            require(len({item[0].get("total") for item in chunks}) == 1,
                    f"{self.label}: inconsistent artifact totals")
            total = int(chunks[0][0]["total"])
            require(sorted(int(item[0]["index"]) for item in chunks) == list(range(total)),
                    f"{self.label}: missing video chunks")
            raw = b"".join(raw for _, raw in sorted(chunks, key=lambda item: int(item[0]["index"])))
            path = self.out_dir / f"{self.label}.mp4"
            path.write_bytes(raw)
            media = {"file": path.name, "bytes": len(raw), "sha256": sha256_bytes(raw),
                     "chunks": total}
        return {"label": self.label, "request_id": self.request_id,
                "native_request_id": f"{self.request_id}:video",
                "session_id": self.session_id,
                "request_sha256": sha256_bytes(canonical(self.body)),
                "http_status": self.http_status, "submit_monotonic_ns": self.submit_ns,
                "end_monotonic_ns": self.end_ns, "terminal_kind": terminal.get("kind"),
                "terminal_monotonic_ns": terminal_tick, "legs": terminal.get("legs"),
                "event_count": len(self.events), "raw_frame_count": len(self.frames),
                "transport_chunk_count": len(self.raw_chunks), "done_count": self.done_count,
                "capture_file": f"{self.label}.capture.json",
                "events_file": f"{self.label}.events.jsonl", "media": media}


def make_body(session_id, video):
    return {"model": "qwen3-native", "stream": True, "max_tokens": 16,
            "session_id": session_id,
            "modalities": ["text", "video"],
            "messages": [{"role": "user", "content": "Say ok."}],
            "video": dict(video)}


def native_video_records(taps, request_id):
    native_id = f"{request_id}:video"
    messages = [record for tap in taps for record in tap.snapshot()
                if record.get("request_id") == native_id and
                (record["tap"] != "native-output" or record.get("from_stage") == "video")]
    return sorted(messages, key=lambda item: item["observed_monotonic_ns"])


def wait_native_abort_log(log_path, request_id, timeout=120):
    deadline = time.monotonic() + timeout
    seen = ""
    while time.monotonic() < deadline:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            parsed = parse_abort_log(line)
            if parsed and parsed["request_id"] == f"{request_id}:video":
                parsed["observed_monotonic_ns"] = clock_ns()
                return parsed
        seen = text
        time.sleep(0.005)
    raise TimeoutError(f"native abort log unavailable for {request_id}; bytes={len(seen)}")


def hash_tree(root):
    root = Path(root)
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append({"path": path.relative_to(root).as_posix(),
                      "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    tree_sha = sha256_bytes(b"".join(
        f"{item['sha256']}  {item['path']}\n".encode() for item in files))
    return {"root_name": root.name, "file_count": len(files), "tree_sha256": tree_sha,
            "files": files}


def source_identities(source_root):
    root = Path(source_root)
    paths = [Path("native/stage_cli") / name for name in
             ("main.cpp", "video_stage.cpp", "stage_common.cpp", "stage_common.hpp")]
    paths += [Path("native/video") / path.name for path in
              (root / "native/video").iterdir() if path.is_file()]
    paths += [Path("src/pipeline/pipeline.cpp"),
              Path("src/scheduling/simple_scheduler.cpp"), Path("src/comm/comm.cpp"),
              Path("tests/native_wan_cancel_fault_live.py"),
              Path("tests/native_wan_cancel_fault_verify.py")]
    return [{"path": path.as_posix(), "bytes": (root / path).stat().st_size,
             "sha256": sha256_file(root / path)} for path in sorted(set(paths))]


def probe_video(ffprobe, path):
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries",
         "stream=codec_name,width,height,nb_frames,r_frame_rate,pix_fmt",
         "-of", "json", str(path)], check=True, capture_output=True, text=True,
        timeout=30)
    stream = json.loads(result.stdout)["streams"][0]
    require(stream == {"codec_name": "h264", "width": 352, "height": 256,
                       "nb_frames": "9", "r_frame_rate": "16/1", "pix_fmt": "yuv420p"},
            f"unexpected valid-control MP4: {stream}")
    return stream


def run_live(args):
    require(os.environ.get("UMMINFER_LIVE_OPT_IN") == "1",
            "live gate requires UMMINFER_LIVE_OPT_IN=1")
    require(re.fullmatch(r"[0-9a-f]{40}", args.base_commit) and
            re.fullmatch(r"[0-9a-f]{40}", args.execution_commit),
            "exact base and execution commits are required")
    out = args.out_dir.resolve()
    require(not out.exists(), f"evidence directory already exists: {out}")
    out.mkdir(parents=True)
    (out / "native-tmp").mkdir()
    args.server_binary = args.server_binary.resolve()
    args.wan_binary = args.wan_binary.resolve()
    args.model = args.model.resolve()
    args.source_root = args.source_root.resolve()
    for path in (args.server_binary, args.wan_binary):
        require(path.is_file() and os.access(path, os.X_OK), f"not executable: {path}")
    args.ffmpeg = shutil.which(args.ffmpeg)
    args.ffprobe = shutil.which(args.ffprobe)
    args.nvidia_smi = shutil.which(args.nvidia_smi)
    require(args.ffmpeg and args.ffprobe and args.nvidia_smi,
            "ffmpeg, ffprobe, and nvidia-smi must resolve")
    require(args.wan_binary.is_file() and args.model.is_dir(), "missing Wan assets")
    initial_gpu = require_gpu7_idle(args.nvidia_smi)
    (out / "gpu-initial.json").write_text(json.dumps(initial_gpu, indent=2) + "\n")
    (out / "bin").mkdir()
    shutil.copyfile(args.server_binary, out / "bin/umm-infer")
    shutil.copyfile(args.wan_binary, out / "bin/umm-native-stage-video")
    (out / "source").mkdir()
    for name in ("native_wan_cancel_fault_live.py", "native_wan_cancel_fault_verify.py"):
        shutil.copyfile(args.source_root / "tests" / name, out / "source" / name)
    identities = {"base_commit": args.base_commit,
                  "execution_commit": args.execution_commit,
                  "server_binary": {"file": "bin/umm-infer",
                                    "bytes": args.server_binary.stat().st_size,
                                    "sha256": sha256_file(args.server_binary)},
                  "wan_binary": {"file": "bin/umm-native-stage-video",
                                 "bytes": args.wan_binary.stat().st_size,
                                 "sha256": sha256_file(args.wan_binary)},
                  "ffmpeg": sha256_file(args.ffmpeg), "ffprobe": sha256_file(args.ffprobe),
                  "model": hash_tree(args.model), "sources": source_identities(args.source_root),
                  "runtime": {"python": sys.version.split()[0]}}
    import msgpack
    import zmq
    identities["runtime"].update({"msgpack": msgpack.__version__, "pyzmq": zmq.__version__})
    (out / "identities.json").write_text(json.dumps(identities, indent=2, sort_keys=True) + "\n")

    port = free_port()
    ipc = out / "ipc"
    ipc.mkdir()
    endpoints = {name: f"ipc://{ipc / name}" for name in
                 ("entry", "vlm", "video-tap", "video", "output-tap", "completion", "abort")}
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["TMPDIR"] = str(out / "native-tmp")
    processes = []
    taps = []
    monitor = GpuMonitor(args.nvidia_smi)
    requests = []
    active = []
    server = wan = None
    fatal = None
    try:
        video_tap = TapProxy("native-input", endpoints["video-tap"], endpoints["video"], out)
        output_tap = TapProxy("native-output", endpoints["output-tap"], endpoints["completion"], out)
        taps = [video_tap, output_tap]
        server = ManagedProcess("server", [
            args.server_binary, "serve", "--host", "127.0.0.1", "--port", str(port),
            "--model", "qwen3-native", "--entry", "entry", "--entry-endpoint", endpoints["entry"],
            "--vlm-entry", "vlm", "--vlm-entry-endpoint", endpoints["vlm"],
            "--video-entry", "video", "--video-entry-endpoint", endpoints["video-tap"],
            "--completion", endpoints["completion"], "--abort", endpoints["abort"],
            "--max-in-flight", "2"], out / "server.log", env)
        processes.append(server)
        entry = ManagedProcess("entry", [args.server_binary, "stage", "--name", "entry",
            "--recv", endpoints["entry"], "--coordinator", endpoints["output-tap"],
            "--abort", endpoints["abort"]], out / "entry.log", env)
        entry.ready_ns = clock_ns()
        processes.append(entry)
        vlm = ManagedProcess("vlm", [args.server_binary, "stage", "--name", "vlm",
            "--recv", endpoints["vlm"], "--coordinator", endpoints["output-tap"],
            "--abort", endpoints["abort"]], out / "vlm.log", env)
        vlm.ready_ns = clock_ns()
        processes.append(vlm)
        wan = ManagedProcess("wan", [args.wan_binary, "--kind", "video", "--name", "video",
            "--model", args.model, "--gpu", "7", "--video-backend", "cuda",
            "--ffmpeg-bin", args.ffmpeg, "--recv-endpoint", endpoints["video"],
            "--coordinator-endpoint", endpoints["output-tap"],
            "--abort-endpoint", endpoints["abort"]], out / "wan.log", env)
        processes.append(wan)
        monitor.start()
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            require(wan.process.poll() is None, "Wan exited before readiness")
            try:
                status, ready, _, _ = http_json(port, "GET", "/ready", timeout=5)
                if status == 200:
                    server.ready_ns = clock_ns()
                    break
            except (OSError, http.client.HTTPException):
                pass
            time.sleep(0.05)
        else:
            raise TimeoutError("public gateway did not become ready")
        output_tap.wait(lambda record: record["message_type"] == "stage_ready" and
                        record["from_stage"] == "video", timeout=args.timeout)
        wan.ready_ns = clock_ns()
        gpu_ready = gpu7_snapshot(args.nvidia_smi)
        gpu_ready_processes = gpu7_processes(args.nvidia_smi)
        require(len(gpu_ready_processes) == 1 and
                gpu_ready_processes[0]["used_memory_mib"] > 0,
                "isolated GPU 7 must have exactly one native Wan NVML owner")
        (out / "gpu-ready.json").write_text(json.dumps(
            {"snapshot": gpu_ready, "compute_processes": gpu_ready_processes,
             "wan_container_pid": wan.pid, "nvml_pid_namespace": "driver_host",
             "ownership_rule": "single NVML compute owner on isolated physical GPU 7"},
            indent=2) + "\n")

        def launch(label, video):
            request = PublicRequest(label, port, make_body(f"wan-67-{label}", video), out).start()
            active.append(request)
            return request

        def finish(request, terminal):
            request.join(args.timeout)
            active.remove(request)
            summary = request.summarize(terminal)
            requests.append(summary)
            return summary

        def native_abort_join(summary, expected_boundary):
            native_id = summary["native_request_id"]
            native = output_tap.wait(lambda record: record["message_type"] == "complete" and
                                     record["request_id"] == native_id and
                                     record.get("from_stage") == "video" and
                                     not record["success"], timeout=30)
            summary["native_abort_terminal"] = native
            log = wait_native_abort_log(out / "wan.log", summary["request_id"], args.timeout)
            summary["native_abort_log"] = log
            require(expected_boundary in log["boundary"],
                    f"{summary['label']}: unexpected cooperative boundary {log}")
            return log

        def recovery(label):
            request = launch(label, CONTROL)
            summary = finish(request, "complete")
            require(summary["media"] is not None, f"{label}: recovery emitted no MP4")
            require(summary["media"]["sha256"] == control_hash,
                    f"{label}: recovery media differs from valid control")
            probe_video(args.ffprobe, out / summary["media"]["file"])
            summary["recovery_exact_control"] = True
            return summary

        # A valid real generation is mandatory before any cancellation/fault injection.
        control = finish(launch("control", CONTROL), "complete")
        require(control["media"] is not None, "valid control emitted no MP4")
        control_hash = control["media"]["sha256"]
        control["ffprobe"] = probe_video(args.ffprobe, out / control["media"]["file"])
        control_native = output_tap.wait(lambda record: record["message_type"] == "complete" and
            record["request_id"] == control["native_request_id"] and record["success"], timeout=30)
        require(control_native.get("media_sha256") == control_hash,
                "public control MP4 differs from raw native terminal media")

        # One serial running request holds native compute. A second request is
        # submitted and acknowledged by the tap but cancelled before dispatch.
        running = launch("denoise-cancel", DENOISE)
        running_id = running.wait_request_id()
        queued = launch("queued-cancel", CONTROL)
        queued_id = queued.wait_request_id()
        running_native_submit = video_tap.wait(
            lambda record: record["message_type"] == "submit" and
            record["request_id"] == f"{running_id}:video", timeout=30)
        queued_native_submit = video_tap.wait(
            lambda record: record["message_type"] == "submit" and
            record["request_id"] == f"{queued_id}:video", timeout=30)
        require(running_native_submit["observed_monotonic_ns"] <=
                queued_native_submit["observed_monotonic_ns"],
                "queued victim was not submitted behind the running native request")
        queued_abort = queued.abort()
        queued_summary = finish(queued, "cancelled")
        queued_summary["abort"] = queued_abort
        require(queued_abort["http_status"] == 200 and
                queued_abort["body"] == {"success": True}, "queued abort was not acknowledged")
        require(queued_summary["media"] is None, "queued cancellation emitted an artifact")
        denoise_target_ns = running_native_submit["observed_monotonic_ns"] + 1_500_000_000
        remaining_ns = denoise_target_ns - clock_ns()
        if remaining_ns > 0:
            time.sleep(remaining_ns / 1e9)
        running_abort = running.abort()
        running_summary = finish(running, "cancelled")
        running_summary["abort"] = running_abort
        require(running_abort["http_status"] == 200, "denoise abort was not acknowledged")
        require(running_summary["media"] is None, "denoise cancellation emitted an artifact")
        native_abort_join(running_summary, "step")
        queued_native = output_tap.wait(lambda record: record["message_type"] == "complete" and
            record["request_id"] == queued_summary["native_request_id"] and
            record.get("from_stage") == "video" and not record["success"], timeout=30)
        queued_summary["native_abort_terminal"] = queued_native
        queued_summary["dispatch_state"] = "native_submit_queued_behind_serial_running_request"
        running_summary["dispatch_state"] = "cooperative_native_cuda_boundary_not_device_preemption"
        status, health, _, health_ns = http_json(port, "GET", "/health")
        require(status == 200 and health.get("in_flight") == 0,
                f"in_flight retained after queued/denoise cancellation: {health}")
        queued_summary["health_zero_monotonic_ns"] = health_ns
        running_summary["health_zero_monotonic_ns"] = health_ns
        recovery("recovery-after-queued-denoise")

        # Cancel only after a task-owned native MP4 exists and has nonzero bytes.
        artifact = launch("artifact-cancel", ARTIFACT)
        artifact_id = artifact.wait_request_id()
        deadline = time.monotonic() + args.timeout
        partial_path = None
        partial_bytes = b""
        while time.monotonic() < deadline:
            candidates = list((out / "native-tmp").glob("umminfer-video-*.mp4"))
            if candidates:
                require(len(candidates) == 1, f"ambiguous task MP4 files: {candidates}")
                candidate = candidates[0]
                current = candidate.read_bytes()
                if len(current) >= 4096:
                    partial_path = candidate
                    partial_bytes = current
                    break
            require(artifact.thread.is_alive(), f"artifact request ended before cancel: {artifact.error}")
            time.sleep(0.002)
        require(partial_path is not None,
                "ARTIFACT_CANCEL_NOT_RUN: no nonzero native partial MP4 window")
        partial_name = "artifact-cancel.partial.mp4"
        (out / partial_name).write_bytes(partial_bytes)
        artifact_abort = artifact.abort()
        artifact_summary = finish(artifact, "cancelled")
        artifact_summary["abort"] = artifact_abort
        artifact_summary["partial_artifact"] = {
            "file": partial_name, "bytes": len(partial_bytes),
            "sha256": sha256_bytes(partial_bytes), "native_temp_basename": partial_path.name,
            "bounded_by_public_limit": len(partial_bytes) <= 48 * 1024 * 1024}
        require(artifact_abort["http_status"] == 200, "artifact abort was not acknowledged")
        require(artifact_summary["media"] is None, "artifact cancellation emitted public media")
        native_abort_join(artifact_summary, "MP4")
        deadline = time.monotonic() + 30
        while partial_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        require(not partial_path.exists(), "cancelled native temporary MP4 was not retired")
        require(not native_ffmpeg_children(wan.pid), "cancelled request retained an ffmpeg child")
        status, health, _, health_ns = http_json(port, "GET", "/health")
        require(status == 200 and health.get("in_flight") == 0, f"artifact in_flight: {health}")
        artifact_summary["health_zero_monotonic_ns"] = health_ns
        recovery("recovery-after-artifact")

        # Late cancellation of an already terminal request changes no result and
        # emits neither another native terminal nor another public event.
        terminal_request = launch("terminal-then-cancel", CONTROL)
        terminal_summary = finish(terminal_request, "complete")
        require(terminal_summary["media"]["sha256"] == control_hash,
                "terminal control differs from initial control")
        before_messages = len(native_video_records(taps, terminal_summary["request_id"]))
        late_abort = terminal_request.abort()
        terminal_summary["abort"] = late_abort
        require(late_abort["http_status"] == 200, "late terminal abort was not acknowledged")
        time.sleep(0.5)
        after_records = native_video_records(taps, terminal_summary["request_id"])
        require(len(after_records) == before_messages and
                not any(record["message_type"] == "abort" for record in after_records),
                "terminal cancellation reached native stage or created stale messages")
        terminal_summary["post_terminal_native_message_count"] = len(after_records)
        terminal_summary["post_terminal_no_new_events"] = True
        recovery("recovery-after-terminal")

        # Native parser rejection is an optional-leg failure: exactly one public
        # complete carries a failed video leg, with no media and no model line.
        invalid_video = dict(CONTROL, size="500x480")
        invalid = launch("invalid-input", invalid_video)
        invalid_summary = finish(invalid, "complete")
        invalid_summary["fault"] = "native Wan 32-pixel alignment rejection"
        video_leg = next(leg for leg in invalid_summary["legs"] if leg["stage"] == "video")
        require(video_leg["state"] == "failed", f"invalid input leg: {video_leg}")
        require(invalid_summary["media"] is None, "invalid input emitted media")
        native_fault = output_tap.wait(lambda record: record["message_type"] == "complete" and
            record["request_id"] == invalid_summary["native_request_id"] and
            not record["success"], timeout=30)
        require("divisible by 32" in (native_fault.get("error") or ""),
                f"unexpected native fault: {native_fault}")
        invalid_summary["native_fault_terminal"] = native_fault
        status, health, _, health_ns = http_json(port, "GET", "/health")
        require(status == 200 and health.get("in_flight") == 0, f"fault in_flight: {health}")
        invalid_summary["health_zero_monotonic_ns"] = health_ns
        recovery("recovery-after-invalid")

        # Every accepted case is now closed. Native aborts have exactly one
        # terminal; no cancelled native request produced success or media.
        native_log_text = (out / "wan.log").read_text(encoding="utf-8", errors="replace")
        for summary in requests:
            records = native_video_records(taps, summary["request_id"])
            terminals = [record for record in records if record["message_type"] == "complete"]
            require(len(terminals) == 1,
                    f"{summary['label']}: expected one native terminal, got {terminals}")
            summary["native_terminal_count"] = len(terminals)
            if summary["label"] in {"queued-cancel", "denoise-cancel", "artifact-cancel"}:
                require(not terminals[0]["success"],
                        f"{summary['label']}: native terminal content invalid: {terminals}")
                require(not any(record.get("media_sha256") for record in terminals),
                        f"{summary['label']}: cancelled native media escaped")
                require(summary["abort"]["send_monotonic_ns"] <=
                        summary["terminal_monotonic_ns"] and
                        summary["terminal_monotonic_ns"] - summary["abort"]["send_monotonic_ns"]
                        <= 1_000_000_000,
                        f"{summary['label']}: public cancellation terminal exceeded 1 s")
            if summary["label"] == "invalid-input":
                require(not terminals[0]["success"],
                        "invalid input did not have one failed native terminal")
        for request_id in (queued_id, invalid_summary["request_id"]):
            native_id = f"{request_id}:video"
            require(not any(parse_abort_log(line) and
                            parse_abort_log(line)["request_id"] == native_id
                            for line in native_log_text.splitlines()),
                    f"{native_id}: unexpected engine abort boundary")
            require(not any("video request backend=" in line and
                            f"request_id={native_id} " in line
                            for line in native_log_text.splitlines()),
                    f"{native_id}: queued/invalid request entered native generation")
        status, health, _, health_ns = http_json(port, "GET", "/health")
        require(status == 200 and health.get("in_flight") == 0, f"final in_flight: {health}")
        (out / "health-final.json").write_text(json.dumps(
            {"status": status, "body": health, "observed_monotonic_ns": health_ns}, indent=2) + "\n")
    except Exception as error:
        fatal = error
        for request in active:
            try:
                if request.request_id:
                    request.abort()
            except Exception:
                pass
    finally:
        for request in active:
            request.thread.join(timeout=5)
        for process, endpoint in ((wan, endpoints["video"]),):
            if process is not None:
                process.stop(endpoint)
        for process in reversed(processes):
            if process is not wan:
                process.stop(None)
        for tap in taps:
            tap.stop()
        monitor.stop()
        (out / "taps.json").write_text(json.dumps(
            {tap.name: tap.snapshot() for tap in taps}, indent=2, sort_keys=True) + "\n")
        (out / "processes.json").write_text(json.dumps(
            [process.record() for process in processes], indent=2, sort_keys=True) + "\n")
        (out / "requests.json").write_text(json.dumps(requests, indent=2, sort_keys=True) + "\n")
        final_gpu = require_gpu7_idle(args.nvidia_smi)
        (out / "gpu-final.json").write_text(json.dumps(final_gpu, indent=2) + "\n")
        (out / "gpu-samples.json").write_text(json.dumps(monitor.samples, indent=2) + "\n")
        if fatal is not None:
            (out / "fatal.txt").write_text(str(fatal) + "\n", encoding="utf-8")
            raise fatal
    summary = {"format": FORMAT, "status": "PASS", "base_commit": args.base_commit,
               "execution_commit": args.execution_commit, "physical_gpu": 7,
               "control_count": 1, "valid_generation_count": 6,
               "cancellation_count": 4, "fault_count": 1, "recovery_count": 4,
               "control_media_sha256": control_hash,
               "request_labels": [item["label"] for item in requests],
               "evidence": {"identities": "identities.json", "requests": "requests.json",
                            "taps": "taps.json", "processes": "processes.json",
                            "gpu_initial": "gpu-initial.json", "gpu_final": "gpu-final.json"}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return out


class OfflineTests(unittest.TestCase):
    def test_sse_data_and_done(self):
        self.assertEqual(parse_sse_frame(b"event: x\r\ndata: {\"a\":1}\r\n\r\n"),
                         [b'{"a":1}'])
        self.assertEqual(parse_sse_frame(b"data: [DONE]\n\n"), [b"[DONE]"])

    def test_transport_only_role_frame_carries_public_id(self):
        payload = parse_sse_frame(
            b'data: {"id":"http-2","choices":[{"delta":{"role":"assistant"}}]}\n\n')[0]
        outer = json.loads(payload)
        self.assertNotIn("unified", outer)
        self.assertEqual(outer["id"], "http-2")

    def test_abort_boundary(self):
        line = ('video request aborted backend=native-cuda-wan request_id=http-4:video '
                'boundary="aborted during MP4 encoding" abort_epoch_ms=1')
        self.assertEqual(parse_abort_log(line)["boundary"], "aborted during MP4 encoding")
        self.assertIsNone(parse_abort_log("video request completed"))

    def test_one_terminal_and_no_stale_events(self):
        events = [{"kind": "vlm_delta", "seq": 0},
                  {"kind": "cancelled", "seq": 1}]
        self.assertEqual(validate_event_order(events)["kind"], "cancelled")
        with self.assertRaises(RuntimeError):
            validate_event_order(events + [{"kind": "artifact", "seq": 2}])
        with self.assertRaises(RuntimeError):
            validate_event_order([{"kind": "complete", "seq": 1}])

    def test_frozen_real_settings(self):
        self.assertEqual(CONTROL["num_frames"], 9)
        self.assertEqual(ARTIFACT["num_frames"], 77)
        self.assertNotEqual(DENOISE, CONTROL)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--server-binary", type=Path)
    parser.add_argument("--wan-binary", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--base-commit")
    parser.add_argument("--execution-commit")
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    args = parser.parse_args(argv)
    if args.self_check:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(OfflineTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    for name in ("server_binary", "wan_binary", "model", "out_dir", "base_commit",
                 "execution_commit"):
        require(getattr(args, name) is not None, f"--{name.replace('_', '-')} is required")
    out = run_live(args)
    print(f"NATIVE_WAN_CANCEL_FAULT: PASS evidence={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
