#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real public default-client NS-03 capture on explicitly idle physical GPU 1.

No request runs without UMMINFER_LIVE_OPT_IN=1. The gateway, native VLM/TTS/Wan,
and FLUX stages are task-owned. FLUX and TTS stay resident with the VLM; Wan is
loaded only after their real output settles, then unloaded before the next case.
Raw SSE, clean text, and content-addressed media stay in the external out-dir.
"""
import argparse
import base64
import hashlib
import http.client
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from ns03_routing import canonical, default_request, load_suite, sha256_bytes, sha256_file  # noqa: E402

BASELINE_SOURCES = {
    "src/pipeline/narrative_prompt.cpp": "0c495a8800bdf2ca351b8862299eae3c3c5adbe45d8adfb2e213650e2892dc7b",
    "src/pipeline/narrative_parser.cpp": "cb864c55f67ca13ae70604a4a3ce1945f68f37b52ac5afa4e59fd3eb355461c8",
    "src/pipeline/narrative_parser.hpp": "c876b79ae314ebd5a876377a01e8818c7210fa97e55b6bfc0ee33ad5e145e9b3",
    "src/pipeline/unified.cpp": "6a45488b040cdd35cf2ff304f12c6a57296d90c1bea155e7166229ab2a60a07f",
    "src/pipeline/unified_schema.cpp": "c62ce75b576f68669ff15c34d0f1ab605fc7dc36bd4034bbdae769e3ce6efe9a",
}
TERMINALS = {"complete", "error", "cancelled"}
REQUEST_TIMEOUT_S = 1200
DIRECTIVE = re.compile(r"\[\[\s*(?:image|video|visual)\s*:", re.IGNORECASE)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def clock_ns():
    return time.perf_counter_ns()


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def gpu_once(nvidia_smi, gpu):
    result = subprocess.run(
        [nvidia_smi, "-i", str(gpu), "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=20)
    fields = [int(value.strip()) for value in result.stdout.strip().split(",")]
    require(len(fields) == 3 and fields[0] == gpu, f"nvidia-smi did not address physical GPU {gpu}")
    return {"index": fields[0], "memory_used_mib": fields[1], "utilization_percent": fields[2]}


def gpu_idle(nvidia_smi, gpu, idle=False):
    observed = gpu_once(nvidia_smi, gpu)
    if not idle:
        return observed
    # NVML utilization can briefly remain nonzero after the last CUDA context
    # exits. Require three consecutive idle samples rather than rejecting a
    # completed cleanup on a single transient sample.
    stable = 1 if observed["memory_used_mib"] <= 16 and observed["utilization_percent"] == 0 else 0
    deadline = time.monotonic() + 30
    while stable < 3 and time.monotonic() < deadline:
        time.sleep(0.5)
        observed = gpu_once(nvidia_smi, gpu)
        stable = stable + 1 if observed["memory_used_mib"] <= 16 and observed["utilization_percent"] == 0 else 0
    require(stable == 3, f"physical GPU {gpu} is not stably idle: {observed}")
    observed["stable_idle_samples"] = 3
    return observed


def model_identity(root, model_id):
    root = Path(root)
    require(root.is_dir(), f"model directory missing for {model_id}")
    files = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        files[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    require(any(name.endswith(".safetensors") for name in files), f"no weights for {model_id}")
    return {"model_id": model_id, "root": str(root), "files": files,
            "tree_sha256": sha256_bytes(canonical(files))}


def runtime_identity():
    result = {}
    for package in ("torch", "diffusers", "transformers", "pyzmq", "msgpack"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "ABSENT"
    return result


def normalize_vlm_messages(messages):
    normalized = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            require(len(content) == 1 and content[0].get("type") == "text" and
                    isinstance(content[0].get("text"), str),
                    "VLM observer expected one text content part")
            content = content[0]["text"]
        require(isinstance(content, str), "VLM observer expected text content")
        normalized.append({"role": message.get("role"), "content": content})
    return normalized


class VlmObserver:
    """Byte-preserving PUSH/PULL proxy with one capture per public request."""

    def __init__(self, public_endpoint, native_endpoint):
        import zmq
        self.context = zmq.Context()
        self.public = self.context.socket(zmq.PULL)
        self.native = self.context.socket(zmq.PUSH)
        self.public.bind(public_endpoint)
        self.native.connect(native_endpoint)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.case_dir = None
        self.rows = 0
        self.native_request_id = None
        self.error = None
        self.thread = threading.Thread(target=self.run, name="ns03-vlm-observer", daemon=True)
        self.thread.start()

    def begin(self, case_dir):
        with self.lock:
            require(self.error is None, f"VLM observer failed: {self.error}")
            require(self.case_dir is None, "VLM observer capture is already active")
            path = Path(case_dir) / "vlm-prompts.jsonl"
            with path.open("xb"):
                pass
            self.case_dir = Path(case_dir)
            self.rows = 0
            self.native_request_id = None

    def run(self):
        import msgpack
        try:
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
                inputs = data.get("request", {}).get("inputs", {})
                messages = inputs.get("messages")
                require(isinstance(messages, list), "VLM submit has no normalized messages")
                row = {"observed_monotonic_ns": time.monotonic_ns(),
                       "request_id": message.get("request_id"),
                       "raw_sha256": sha256_bytes(frames[0]),
                       "raw_base64": base64.b64encode(frames[0]).decode(),
                       "messages": normalize_vlm_messages(messages)}
                with self.lock:
                    require(self.case_dir is not None, "VLM submit arrived outside a public capture")
                    with (self.case_dir / "vlm-prompts.jsonl").open("ab", buffering=0) as output:
                        output.write((json.dumps(row, sort_keys=True) + "\n").encode())
                    self.rows += 1
                    self.native_request_id = row["request_id"]
        except Exception as error:
            self.error = error

    def finish(self, public_request_id):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with self.lock:
                if self.error is not None or self.rows:
                    break
            time.sleep(0.01)
        with self.lock:
            try:
                require(self.error is None, f"VLM observer failed: {self.error}")
                require(self.rows == 1, f"expected one VLM submit, observed {self.rows}")
                require(isinstance(public_request_id, str) and public_request_id,
                        "public request identity is unavailable for VLM binding")
                require(self.native_request_id in (public_request_id, public_request_id + ":vlm"),
                        f"VLM/public request mismatch: {self.native_request_id} != {public_request_id}")
                path = self.case_dir / "vlm-prompts.jsonl"
                return {"path": path.name, "rows": 1, "sha256": sha256_file(path),
                        "native_request_id": self.native_request_id}
            finally:
                self.case_dir = None

    def close(self):
        self.stop_event.set()
        self.thread.join(10)
        require(not self.thread.is_alive(), "VLM observer did not stop")
        self.public.close(0)
        self.native.close(0)
        self.context.term()
        with self.lock:
            require(self.case_dir is None, "VLM observer closed during an active capture")
            if self.error is not None:
                raise self.error


def http_json(port, method, path, body=None, timeout=30):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        payload = None if body is None else json.dumps(body)
        conn.request(method, path, body=payload,
                     headers={"Content-Type": "application/json"} if payload is not None else {})
        response = conn.getresponse()
        raw = response.read()
        return response.status, json.loads(raw) if raw else {}
    finally:
        conn.close()


def http_text(port, path, timeout=30):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        raw = response.read()
        require(response.status == 200, f"GET {path} returned {response.status}")
        return raw.decode("utf-8")
    finally:
        conn.close()


def coordinator_stage_state(port, stage):
    metrics = http_text(port, "/metrics", timeout=10)
    counters = {status: 0 for status in ("submitted", "completed", "failed", "aborted")}
    in_flight = None
    for line in metrics.splitlines():
        if line.startswith("umm_stage_requests_total{") and f'stage="{stage}"' in line:
            for status in counters:
                if f'status="{status}"' in line:
                    counters[status] += int(float(line.rsplit(" ", 1)[1]))
        elif line.startswith("umm_stage_in_flight{") and f'stage="{stage}"' in line:
            in_flight = int(float(line.rsplit(" ", 1)[1]))
    require(in_flight is not None, f"missing {stage} coordinator metric")
    counters["in_flight"] = in_flight
    counters["settled"] = counters["completed"] + counters["failed"] + counters["aborted"]
    return counters


class ManagedProcess:
    def __init__(self, name, command, log_path, env, event_sink=None):
        self.name = name
        self.command = [str(item) for item in command]
        self.log_path = Path(log_path)
        self.event_sink = event_sink
        self.start_ns = clock_ns()
        self.ready_ns = None
        self.stop_ns = None
        self.returncode = None
        self.stop_error = None
        self.log = self.log_path.open("wb")
        self.process = subprocess.Popen(self.command, stdout=self.log, stderr=subprocess.STDOUT,
                                        env=env, start_new_session=True)
        if event_sink is not None:
            event_sink(self, "start")

    def wait_log(self, needle, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.log_path.exists() and needle in self.log_path.read_text(encoding="utf-8", errors="replace"):
                self.ready_ns = clock_ns()
                return
            if self.process.poll() is not None:
                raise RuntimeError(f"{self.name} exited before readiness; see external log {self.log_path.name}")
            time.sleep(0.05)
        raise TimeoutError(f"{self.name} did not log readiness marker {needle!r}")

    def stop(self, shutdown_endpoint=None, timeout=45):
        if self.stop_ns is not None:
            return
        context = push = None
        try:
            if self.process.poll() is None and shutdown_endpoint and self.ready_ns is not None:
                import msgpack
                import zmq
                context = zmq.Context()
                push = context.socket(zmq.PUSH)
                push.setsockopt(zmq.LINGER, 0)
                push.connect(shutdown_endpoint)
                push.send(msgpack.packb({"type": "shutdown"}, use_bin_type=True), zmq.NOBLOCK)
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
        if self.event_sink is not None:
            self.event_sink(self, "stop")

    def record(self):
        return {"name": self.name, "pid": self.process.pid,
                "argv_sha256": sha256_bytes("\0".join(self.command).encode()),
                "start_monotonic_ns": self.start_ns, "ready_monotonic_ns": self.ready_ns,
                "stop_monotonic_ns": self.stop_ns, "returncode": self.returncode,
                "stop_error": self.stop_error, "log": self.log_path.name}


class Stack:
    def __init__(self, args, out):
        self.args = args
        self.out = Path(out)
        self.run = self.out / "run"
        self.run.mkdir()
        self.port = args.port or free_port()
        require(self.port != 43301, "resident service port is forbidden")
        self.endpoints = {name: f"ipc://{self.run}/{name}" for name in
                          ("completion", "abort", "entry-in", "vlm-public", "vlm-in",
                           "image-in", "tts-in", "video-in")}
        self.event_log = self.out / "processes.jsonl"
        self.env = dict(os.environ)
        self.env.pop("CUDA_VISIBLE_DEVICES", None)
        if args.library_path:
            self.env["LD_LIBRARY_PATH"] = args.library_path + (
                ":" + self.env["LD_LIBRARY_PATH"] if self.env.get("LD_LIBRARY_PATH") else "")
        self.server = self.entry = self.vlm = self.flux = self.tts = self.wan = None
        self.wan_used = False
        self.retired = []
        self.observer = VlmObserver(self.endpoints["vlm-public"], self.endpoints["vlm-in"])

    def record_process_event(self, process, event):
        with self.event_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": event, "record": process.record()},
                                    sort_keys=True) + "\n")
            handle.flush()

    def start_base(self):
        a = self.args
        self.server = ManagedProcess("server", [
            a.server_binary, "serve", "--host", "127.0.0.1", "--port", str(self.port),
            "--model", a.model, "--entry", "entry", "--entry-endpoint", self.endpoints["entry-in"],
            "--vlm-entry", "vlm", "--vlm-entry-endpoint", self.endpoints["vlm-public"],
            "--image-entry", "image", "--image-entry-endpoint", self.endpoints["image-in"],
            "--tts-entry", "tts", "--tts-entry-endpoint", self.endpoints["tts-in"],
            "--video-entry", "video", "--video-entry-endpoint", self.endpoints["video-in"],
            "--completion", self.endpoints["completion"], "--abort", self.endpoints["abort"],
            "--session-history", "--max-in-flight", "1"], self.out / "server.log", self.env, self.record_process_event)
        self.entry = ManagedProcess("entry", [
            a.server_binary, "stage", "--name", "entry", "--recv", self.endpoints["entry-in"],
            "--coordinator", self.endpoints["completion"], "--abort", self.endpoints["abort"]],
            self.out / "entry.log", self.env, self.record_process_event)
        self.vlm = ManagedProcess("vlm", [
            a.vlm_binary, "--kind", "qwen38", "--name", "vlm", "--model", a.vlm_model,
            "--gpu", str(a.gpu), "--batch-size", "1", "--max-new-tokens", "256",
            "--recv-endpoint", self.endpoints["vlm-in"], "--coordinator-endpoint", self.endpoints["completion"],
            "--abort-endpoint", self.endpoints["abort"]], self.out / "vlm.log", self.env, self.record_process_event)
        self.vlm.wait_log("native stage vlm ready kind=qwen38", a.timeout)
        self.start_flux_tts()
        deadline = time.monotonic() + a.timeout
        while time.monotonic() < deadline:
            require(self.observer.error is None, f"VLM observer failed: {self.observer.error}")
            try:
                status, _ = http_json(self.port, "GET", "/ready", timeout=5)
                if status == 200:
                    return
            except (OSError, http.client.HTTPException):
                pass
            require(self.server.process.poll() is None, "public gateway exited before readiness")
            time.sleep(0.05)
        raise TimeoutError("public gateway did not become ready")

    def start_flux_tts(self):
        a = self.args
        require(self.flux is None and self.tts is None and self.wan is None, "output stage state is inconsistent")
        self.flux = ManagedProcess("image", [
            a.python, "-I", a.flux_stage, "--name", "image", "--backend", "diffusers",
            "--model", a.flux_model, "--model-id", "black-forest-labs/FLUX.2-klein-4B",
            "--revision", "UNKNOWN", "--gpu", str(a.gpu), "--default-size", "512x512",
            "--default-steps", "4", "--recv", self.endpoints["image-in"],
            "--coordinator", self.endpoints["completion"], "--abort", self.endpoints["abort"]],
            self.out / "image.log", self.env, self.record_process_event)
        self.flux.wait_log("ready kind=image name=image", a.timeout)
        self.tts = ManagedProcess("tts", [
            a.tts_binary, "--kind", "tts", "--name", "tts", "--model", a.tts_model,
            "--gpu", str(a.gpu), "--recv-endpoint", self.endpoints["tts-in"],
            "--coordinator-endpoint", self.endpoints["completion"],
            "--abort-endpoint", self.endpoints["abort"]], self.out / "tts.log", self.env, self.record_process_event)
        self.tts.wait_log("native stage tts ready kind=tts", a.timeout)

    def outputs_settled(self):
        states = {stage: coordinator_stage_state(self.port, stage) for stage in ("vlm", "image", "tts")}
        return all(state["in_flight"] == 0 and state["submitted"] == state["settled"]
                   for state in states.values()), states

    def start_wan_after_outputs(self):
        if self.wan_used:
            return
        deadline = time.monotonic() + self.args.timeout
        while time.monotonic() < deadline:
            settled, _ = self.outputs_settled()
            if settled:
                break
            time.sleep(0.05)
        else:
            raise TimeoutError("VLM/FLUX/TTS did not settle before Wan load")
        self.flux.stop(self.endpoints["image-in"])
        self.tts.stop(self.endpoints["tts-in"])
        self.retired.extend((self.flux.record(), self.tts.record()))
        self.flux = self.tts = None
        a = self.args
        self.wan = ManagedProcess("video", [
            a.wan_binary, "--kind", "video", "--name", "video", "--video-backend", "cuda",
            "--model", a.wan_model, "--gpu", str(a.gpu), "--size", "704x480", "--steps", "25",
            "--fps", "16", "--recv-endpoint", self.endpoints["video-in"],
            "--coordinator-endpoint", self.endpoints["completion"],
            "--abort-endpoint", self.endpoints["abort"]], self.out / "video.log", self.env, self.record_process_event)
        try:
            self.wan.wait_log("native stage video ready kind=video", a.timeout)
            self.wan_used = True
        except Exception:
            # Never leave a partial Wan or absent FLUX/TTS for a later case.
            self.wan.stop(self.endpoints["video-in"])
            self.retired.append(self.wan.record())
            self.wan = None
            self.start_flux_tts()
            raise

    def restore_after_wan(self):
        if not self.wan_used:
            return
        self.wan.stop(self.endpoints["video-in"])
        self.retired.append(self.wan.record())
        self.wan = None
        self.wan_used = False
        self.start_flux_tts()

    def stop_all(self):
        for process, endpoint in ((self.wan, self.endpoints["video-in"]),
                                  (self.flux, self.endpoints["image-in"]),
                                  (self.tts, self.endpoints["tts-in"]),
                                  (self.vlm, self.endpoints["vlm-in"]),
                                  (self.entry, self.endpoints["entry-in"])):
            if process is not None:
                process.stop(endpoint)
        if self.server is not None:
            self.server.stop(None)

    def processes(self):
        current = [p.record() for p in (self.server, self.entry, self.vlm, self.flux, self.tts, self.wan)
                   if p is not None]
        return self.retired + current


def sse_payloads(raw):
    """Parse SSE data fields without trusting transport chunk boundaries."""
    payloads = []
    data_lines = []
    normalized_input = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    for line in normalized_input.split(b"\n"):
        if not line:
            if data_lines:
                payloads.append(b"\n".join(data_lines).decode("utf-8"))
                data_lines = []
            continue
        if line.startswith(b":"):
            continue
        if line.startswith(b"data:"):
            value = line[5:]
            if value.startswith(b" "):
                value = value[1:]
            data_lines.append(value)
    if data_lines:
        payloads.append(b"\n".join(data_lines).decode("utf-8"))
    return payloads


def decode_media(ffmpeg, path, media):
    if media == "image":
        require(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), "image is not PNG")
    elif media == "video":
        require(path.read_bytes()[4:8] == b"ftyp", "video is not ISO-BMFF MP4")
    else:
        return False
    result = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-f", "null", "-"],
                            capture_output=True, timeout=120)
    require(result.returncode == 0,
            f"ffmpeg rejected {media}: {result.stderr.decode(errors='replace')[:500]}")
    return True


def analyze_stream(raw, request, session_id, case_dir, ffmpeg):
    errors = []
    record = {"status": "captured", "errors": errors, "leg_errors": [], "media_errors": [],
              "request_sha256": sha256_bytes(canonical(request)),
              "raw_response_sha256": sha256_bytes(raw), "session_history": request.get("session_history", False)}
    try:
        payloads = sse_payloads(raw)
        require(payloads and payloads[-1] == "[DONE]", "SSE does not end in [DONE]")
        done_count = payloads.count("[DONE]")
        require(done_count == 1, f"expected one [DONE], got {done_count}")
        events = []
        public_ids = set()
        for payload in payloads:
            if payload == "[DONE]":
                continue
            frame = json.loads(payload)
            if frame.get("id"):
                public_ids.add(frame["id"])
            unified = frame.get("unified")
            if isinstance(unified, dict):
                events.append(unified)
        require(events, "no unified public events")
        require(len(public_ids) == 1, f"ambiguous public request ids: {sorted(public_ids)}")
        request_id = next(iter(public_ids))
        require(all(event.get("request_id", request_id) == request_id for event in events),
                "cross-request unified event")
        require(all(event.get("session_id") == session_id for event in events),
                "cross-session unified event")
        seqs = [event.get("seq") for event in events]
        require(seqs == list(range(len(events))), "unified seq is not gap-free from zero")
        deltas = [event.get("text", "") for event in events if event.get("kind") == "vlm_delta"]
        clean = "".join(deltas)
        terminal_events = [event for event in events if event.get("kind") in TERMINALS]
        require(len(terminal_events) == 1 and terminal_events[0] is events[-1],
                "expected exactly one final unified terminal")
        terminal = terminal_events[0]
        require(terminal.get("kind") == "complete", f"terminal is {terminal.get('kind')}: {terminal.get('error')}")
        if terminal.get("text", "") != clean:
            errors.append("terminal text differs from clean deltas")
        if DIRECTIVE.search(clean):
            errors.append("raw narrative directive leaked into clean text")
        if not clean.strip():
            errors.append("empty clean response text")
        slots = [{"slot_id": event.get("slot_id"), "slot_index": event.get("slot_index"),
                  "media_type": event.get("media_type"), "stage": event.get("stage"),
                  "prompt_sha256": sha256_bytes(str(event.get("prompt", "")).encode()),
                  "text_offset": event.get("text_offset")}
                 for event in events if event.get("kind") == "visual_slot"]
        leg_errors = [{"stage": event.get("stage"), "slot_id": event.get("slot_id"),
                       "slot_index": event.get("slot_index"), "error": event.get("error", "")}
                      for event in events if event.get("kind") == "leg_error"]
        record.update({"slots": slots, "terminal_kind": terminal.get("kind"),
                       "terminal_count": len(terminal_events), "done_count": done_count,
                       "clean_text": clean, "clean_text_sha256": sha256_bytes(clean.encode()),
                       "leg_errors": leg_errors})
        for index, slot in enumerate(slots):
            require(slot["media_type"] in ("image", "video") and slot["stage"] == slot["media_type"],
                    "slot modality/stage mismatch")
        groups = {}
        for event in events:
            if event.get("kind") != "artifact":
                continue
            artifact = event.get("artifact") or {}
            artifact_id = artifact.get("artifact_id")
            require(isinstance(artifact_id, str) and artifact_id, "artifact id missing")
            require(artifact.get("encoding") == "base64", "artifact is not base64")
            groups.setdefault(artifact_id, []).append(artifact)
        artifacts = []
        record["artifacts"] = artifacts
        audio_count = audio_bytes = 0
        pcm16_valid = True
        for artifact_id, chunks in groups.items():
            chunks.sort(key=lambda item: item.get("index", -1))
            total = chunks[0].get("total")
            require(isinstance(total, int) and total > 0 and len(chunks) == total,
                    f"incomplete artifact {artifact_id}")
            require([chunk.get("index") for chunk in chunks] == list(range(total)),
                    f"noncontiguous artifact {artifact_id}")
            first = chunks[0]
            for chunk in chunks:
                for key in ("total", "media_type", "mime_type", "slot_id", "slot_index"):
                    require(chunk.get(key) == first.get(key),
                            f"artifact chunk {key} changed for {artifact_id}")
                final_chunk = chunk.get("index") + 1 == total
                require(("bytes" in chunk) == final_chunk and ("sha256" in chunk) == final_chunk,
                        f"artifact digest fields are not final-chunk-only for {artifact_id}")
            raw_artifact = b"".join(base64.b64decode(chunk["data"], validate=True) for chunk in chunks)
            final = chunks[-1]
            digest = sha256_bytes(raw_artifact)
            require(final.get("sha256") == digest and final.get("bytes") == len(raw_artifact),
                    f"artifact digest/bytes mismatch {artifact_id}")
            media = final.get("media_type")
            item = {"artifact_id": artifact_id, "media_type": media,
                    "mime_type": final.get("mime_type"), "sha256": digest, "bytes": len(raw_artifact),
                    "slot_id": final.get("slot_id"), "slot_index": final.get("slot_index"),
                    "decoded": None}
            if media in ("image", "video"):
                matching = [slot for slot in slots if slot["slot_id"] == item["slot_id"]]
                require(len(matching) == 1 and matching[0]["media_type"] == media and
                        matching[0]["slot_index"] == item["slot_index"],
                        f"false/stale visual completion {artifact_id}")
                target = case_dir / "media" / f"{digest}.{ 'png' if media == 'image' else 'mp4'}"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw_artifact)
                try:
                    item["decoded"] = decode_media(ffmpeg, target, media)
                except Exception as error:
                    item["decoded"] = False
                    record["media_errors"].append(
                        {"slot_id": item["slot_id"], "sha256": digest,
                         "error": f"{type(error).__name__}: {error}"})
            elif media == "audio":
                require(final.get("mime_type") == "audio/pcm", "unexpected final WAV or audio MIME")
                require(len(raw_artifact) > 0 and len(raw_artifact) % 2 == 0,
                        "invalid PCM16 chunk length")
                audio_count += 1
                audio_bytes += len(raw_artifact)
                # Raw SSE retains every PCM chunk; the compact record carries
                # only the frozen aggregate, not thousands of duplicate rows.
                continue
            else:
                raise AssertionError(f"unexpected artifact modality {media}")
            artifacts.append(item)
        terminal_legs = terminal.get("legs") or []
        for leg in terminal_legs:
            require(leg.get("state") not in ("pending", "running"), "pending/running terminal leg leak")
        settled_slots = {leg.get("slot_id") for leg in terminal_legs if leg.get("slot_id")}
        require(settled_slots == {slot["slot_id"] for slot in slots}, "terminal omitted a visual slot")
        record.update({
            "request_id": request_id, "session_id": session_id, "terminal_kind": terminal.get("kind"),
            "terminal_count": len(terminal_events), "done_count": done_count, "event_count": len(events),
            "slots": slots, "artifacts": artifacts, "clean_text": clean,
            "clean_text_sha256": sha256_bytes(clean.encode()),
            "audio": {"artifact_count": audio_count, "bytes": audio_bytes, "pcm16_valid": pcm16_valid},
            "terminal_legs": [{"stage": leg.get("stage"), "slot_id": leg.get("slot_id"),
                               "state": leg.get("state")} for leg in terminal_legs],
            "parse_ambiguous": False,
        })
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
        record["parse_ambiguous"] = True
        record.setdefault("terminal_count", 0)
        record.setdefault("done_count", 0)
        record.setdefault("slots", [])
        record.setdefault("artifacts", [])
        record.setdefault("audio", {"artifact_count": 0, "bytes": 0, "pcm16_valid": False})
        record["clean_text"] = locals().get("clean", "")
        record["clean_text_sha256"] = sha256_bytes(record["clean_text"].encode())
        record["terminal_kind"] = locals().get("terminal", {}).get("kind") if isinstance(locals().get("terminal"), dict) else None
    return record


def capture_request(stack, request, session_id, case_dir, timeout=REQUEST_TIMEOUT_S):
    require(timeout == REQUEST_TIMEOUT_S, "frozen overall request deadline must be 1200 seconds")
    case_dir.mkdir(parents=True, exist_ok=True)
    stack.observer.begin(case_dir)
    deadline = time.monotonic() + REQUEST_TIMEOUT_S
    observed_request_id = None
    video_seen = threading.Event()
    manager_error = []

    def manager():
        try:
            require(video_seen.wait(timeout), "manager timed out without video")
            stack.start_wan_after_outputs()
        except Exception as error:
            manager_error.append(str(error))

    thread = None
    raw = bytearray()
    conn = http.client.HTTPConnection("127.0.0.1", stack.port, timeout=timeout)
    try:
        conn.request("POST", "/v1/chat/completions", body=json.dumps(request),
                     headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
        response = conn.getresponse()
        if response.status != 200:
            detail = response.read()
            raw.extend(detail)
            record = {"status": "http_error", "errors": [f"HTTP {response.status}: {detail[:500]!r}"], "leg_errors": [], "media_errors": [],
                      "request_sha256": sha256_bytes(canonical(request)),
                      "raw_response_sha256": sha256_bytes(raw), "slots": [], "artifacts": [],
                      "terminal_count": 0, "done_count": 0, "terminal_kind": None,
                      "audio": {"artifact_count": 0, "bytes": 0, "pcm16_valid": False},
                      "clean_text": "", "clean_text_sha256": sha256_bytes(b""),
                      "session_history": request.get("session_history", False), "parse_ambiguous": True}
        else:
            pending = b""
            while True:
                chunk = response.read1(65536)
                if not chunk:
                    break
                raw.extend(chunk)
                if time.monotonic() > deadline:
                    raise TimeoutError("frozen 1200-second overall request deadline exceeded")
                pending += chunk
                while b"\n\n" in pending.replace(b"\r\n", b"\n").replace(b"\r", b"\n"):
                    normalized = pending.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
                    frame, remainder = normalized.split(b"\n\n", 1)
                    # Re-read normalized remainder; transport chunks are already retained in raw.
                    pending = remainder
                    for line in frame.split(b"\n"):
                        if not line.startswith(b"data:"):
                            continue
                        payload = line[5:].lstrip()
                        if payload == b"[DONE]":
                            continue
                        try:
                            public_frame = json.loads(payload)
                            observed_request_id = observed_request_id or public_frame.get("id")
                            unified = public_frame.get("unified") or {}
                        except Exception:
                            continue
                        if unified.get("kind") == "visual_slot" and unified.get("media_type") == "video" \
                                and thread is None:
                            video_seen.set()
                            thread = threading.Thread(target=manager, name="ns03-wan-manager", daemon=True)
                            thread.start()
            record = analyze_stream(bytes(raw), request, session_id, case_dir, stack.args.ffmpeg)
    except Exception as error:
        raw_digest = sha256_bytes(bytes(raw))
        record = {"status": "transport_error", "errors": [f"{type(error).__name__}: {error}"], "leg_errors": [], "media_errors": [],
                  "request_sha256": sha256_bytes(canonical(request)), "raw_response_sha256": raw_digest,
                  "slots": [], "artifacts": [], "terminal_count": 0, "done_count": 0,
                  "terminal_kind": None, "audio": {"artifact_count": 0, "bytes": 0, "pcm16_valid": False},
                  "clean_text": "", "clean_text_sha256": sha256_bytes(b""),
                  "session_history": request.get("session_history", False), "parse_ambiguous": True}
        try:
            if observed_request_id:
                http_json(stack.port, "POST", "/abort", {
                    "request_id": observed_request_id, "session_id": session_id})
                abort_deadline = time.monotonic() + 30
                while time.monotonic() < abort_deadline:
                    _, health = http_json(stack.port, "GET", "/health", timeout=5)
                    if health.get("in_flight") == 0:
                        break
                    time.sleep(0.05)
                else:
                    raise RuntimeError("aborted request did not reach in_flight=0")
        except Exception as abort_error:
            record["errors"].append(f"abort/cleanup failure: {abort_error}")
    finally:
        conn.close()
        if thread is not None:
            thread.join(timeout=stack.args.timeout)
            if thread.is_alive():
                manager_error.append("Wan manager remained alive after its bounded join")
        if manager_error:
            record["errors"].extend(manager_error)
            record["parse_ambiguous"] = True
    (case_dir / "raw.sse").write_bytes(bytes(raw))
    (case_dir / "request.json").write_bytes(canonical(request) + b"\n")
    (case_dir / "clean.txt").write_text(record.get("clean_text", ""), encoding="utf-8")
    record["vlm_capture"] = stack.observer.finish(record.get("request_id") or observed_request_id)
    stack.restore_after_wan()
    if manager_error or any(error.startswith("abort/cleanup failure:") for error in record["errors"]):
        raise RuntimeError(f"unsafe NS-03 process state after {case_dir.name}: {record['errors']}")
    return record


def arm_initial_history(arm, baseline=None, candidate=None):
    """Bind first-turn history independently from the prompt-routing arm."""
    binding = baseline if arm == "before" else candidate
    if binding is not None and "initial_history" in binding:
        require(isinstance(binding["initial_history"], bool),
                "initial_history binding must be boolean")
        return binding["initial_history"]
    return arm == "after"


def selected_cases(suite, population, case_class):
    cases = [case for case in suite["cases"] if case["population"] == population and
             (case_class == "all" or case["class"] == case_class)]
    require(len(cases) == (20 if case_class == "all" else 5),
            f"incomplete {population}/{case_class} case selection")
    if case_class == "followup":
        require([case["id"] for case in cases] ==
                [case["id"] for case in suite["cases"]
                 if case["population"] == population and case["class"] == "followup"],
                "follow-up-only capture must retain all five cases in suite order")
    return cases


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default=str(ROOT / "benchmarks/ns03-routing/suite.json"))
    p.add_argument("--population", choices=("development", "heldout"), required=True)
    p.add_argument("--case-class", choices=("all", "text", "static", "dynamic", "followup"), default="all")
    p.add_argument("--arm", choices=("before", "after"), required=True)
    p.add_argument("--base-commit", required=True)
    p.add_argument("--execution-commit", required=True)
    p.add_argument("--candidate-manifest")
    p.add_argument("--baseline-manifest")
    p.add_argument("--server-binary", required=True)
    p.add_argument("--vlm-binary", required=True)
    p.add_argument("--tts-binary", required=True)
    p.add_argument("--wan-binary", required=True)
    p.add_argument("--flux-stage", required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--vlm-model", required=True)
    p.add_argument("--flux-model", required=True)
    p.add_argument("--tts-model", required=True)
    p.add_argument("--wan-model", required=True)
    p.add_argument("--model-manifest")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model", default="qwen3-native")
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--nvidia-smi", default="nvidia-smi")
    p.add_argument("--gpu", type=int, default=1, choices=(1, 7),
                   help="explicitly idle physical GPU; default preserves the frozen GPU-1 gate")
    p.add_argument("--library-path")
    p.add_argument("--port", type=int)
    p.add_argument("--timeout", type=int, default=900)
    return p.parse_args(argv)


def live(args):
    require(os.environ.get("UMMINFER_LIVE_OPT_IN") == "1", "REAL_GENERATION: NOT_RUN; explicit opt-in required")
    require(len(args.base_commit) == 40 and len(args.execution_commit) == 40, "full commit ids required")
    require(args.model == "qwen3-native", "frozen public model identity must be qwen3-native")
    require(args.arm == "before" or not args.baseline_manifest,
            "--baseline-manifest is valid only for Before")
    baseline = None
    candidate = None
    if args.baseline_manifest:
        baseline = json.loads(Path(args.baseline_manifest).read_text())
        require(args.execution_commit == args.base_commit,
                "baseline-manifest Before must execute its base commit")
    if args.candidate_manifest:
        candidate = json.loads(Path(args.candidate_manifest).read_text())
    if args.arm == "before" and baseline is None:
        require(args.execution_commit == "91310b36184c86a1c147debb27db8643cc0cb3e4",
                "Before must execute the unmodified official main commit")
    else:
        require(args.arm == "before" or candidate is not None,
                "After requires the frozen candidate manifest")
    out = Path(args.out_dir)
    started_at_utc = datetime.now(timezone.utc).isoformat()
    require(not out.exists(), f"evidence directory already exists: {out}")
    out.mkdir(parents=True)
    (out / "responses").mkdir()
    suite = load_suite(args.suite)
    cases = selected_cases(suite, args.population, args.case_class)
    initial_history = arm_initial_history(args.arm, baseline, candidate)
    source_hashes = {name: sha256_file(ROOT / name) for name in BASELINE_SOURCES}
    for name in (*BASELINE_SOURCES, "tests/ns03_routing.py", "tests/ns03_routing_live.py"):
        destination = out / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    if args.arm == "before":
        if baseline is None:
            require(source_hashes == BASELINE_SOURCES,
                    f"Before source differs from official main: {source_hashes}")
        else:
            require(baseline.get("source_sha256") == source_hashes,
                    "Before source differs from baseline manifest")
    else:
        require(candidate["execution_commit"] == args.execution_commit, "candidate commit mismatch")
        require(candidate["source_sha256"] == source_hashes, "candidate source hash mismatch")
    gpu = {"before": gpu_idle(args.nvidia_smi, args.gpu, idle=True)}
    if args.model_manifest:
        models = json.loads(Path(args.model_manifest).read_text())
    else:
        models = {
            "vlm": model_identity(args.vlm_model, "Qwen/Qwen3.8-27B"),
            "flux2": model_identity(args.flux_model, "black-forest-labs/FLUX.2-klein-4B"),
            "tts": model_identity(args.tts_model, "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
            "wan22": model_identity(args.wan_model, "Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
        }
        (out / "model-manifest.json").write_text(json.dumps(models, indent=2, sort_keys=True) + "\n")
    identities = {
        "base_commit": args.base_commit, "execution_commit": args.execution_commit,
        "population": args.population, "arm": args.arm, "case_class": args.case_class,
        "initial_history": initial_history, "vlm_observer": True, "physical_gpu": args.gpu,
        "started_at_utc": started_at_utc,
        "source_sha256": source_hashes, "suite_sha256": sha256_file(args.suite),
        "binaries": {name: {"path": str(Path(path).resolve()), "sha256": sha256_file(path),
                            "bytes": Path(path).stat().st_size}
                     for name, path in (("server", args.server_binary), ("vlm", args.vlm_binary),
                                        ("tts", args.tts_binary), ("wan", args.wan_binary))},
        "flux_stage": {"path": str(Path(args.flux_stage).resolve()), "sha256": sha256_file(args.flux_stage)},
        "harness_sha256": sha256_file(__file__),
        "models": models, "python_runtime": runtime_identity(),
    }
    if baseline is not None:
        identities["baseline_manifest"] = {
            "path": str(Path(args.baseline_manifest).resolve()),
            "sha256": sha256_file(args.baseline_manifest),
        }
    (out / "run-identity.json").write_text(json.dumps(identities, indent=2, sort_keys=True) + "\n")
    stack = Stack(args, out)
    results = []
    health = None
    try:
        stack.start_base()
        for case in cases:
            session = f"ns03-{args.arm}-{case['id']}"
            setup_record = None
            if case["class"] == "followup":
                setup_request = default_request(case, model=args.model, session_id=session, followup=False,
                                                initial_history=initial_history)
                setup_record = capture_request(stack, setup_request, session,
                                               out / "responses" / case["id"] / "setup", REQUEST_TIMEOUT_S)
                setup_status, setup_health = http_json(stack.port, "GET", "/health", timeout=30)
                require(setup_status == 200 and setup_health.get("in_flight") == 0,
                        f"setup left in-flight work before {case['id']}: {setup_health}")
                setup_record["same_session_as_setup"] = True
            request = default_request(case, model=args.model, session_id=session,
                                      followup=case["class"] == "followup",
                                      initial_history=initial_history)
            record = capture_request(stack, request, session, out / "responses" / case["id"] / "answer",
                                     REQUEST_TIMEOUT_S)
            record["case_id"] = case["id"]
            case_status, case_health = http_json(stack.port, "GET", "/health", timeout=30)
            require(case_status == 200 and case_health.get("in_flight") == 0,
                    f"case {case['id']} left in-flight work: {case_health}")
            if setup_record is not None:
                record["setup"] = setup_record
                record["same_session_as_setup"] = (record.get("session_id") == setup_record.get("session_id") == session)
            results.append(record)
            print(json.dumps({"case": case["id"], "slots": [s["media_type"] for s in record["slots"]],
                              "terminal": record["terminal_kind"], "errors": record["errors"]}), flush=True)
        status, health = http_json(stack.port, "GET", "/health", timeout=30)
        require(status == 200 and health.get("in_flight") == 0, f"final in_flight is not zero: {health}")
    finally:
        try:
            stack.stop_all()
        finally:
            try:
                stack.observer.close()
            finally:
                gpu["after_cleanup"] = gpu_idle(args.nvidia_smi, args.gpu, idle=True)
                # Only task-owned ephemeral IPC sockets are removed after every process exits.
                shutil.rmtree(stack.run)
    report = {"format": "umminfer-ns03-routing-run-v1", "population": args.population,
              "arm": args.arm, "case_class": args.case_class,
              "identity_file": "run-identity.json", "cases": results,
              "processes": stack.processes(), "gpu": gpu, "gpu1": gpu, "health_final": health,
              "started_at_utc": started_at_utc,
              "finished_at_utc": datetime.now(timezone.utc).isoformat(),
              "physical_perceptual_quality": "NOT_RUN"}
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    files = {path.relative_to(out).as_posix(): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
             for path in sorted(out.rglob("*")) if path.is_file() and "run" not in path.relative_to(out).parts}
    (out / "evidence-files.json").write_text(json.dumps(files, indent=2, sort_keys=True) + "\n")
    print(f"NS03_{args.population}_{args.arm}: CAPTURED {len(results)} cases")
    print("PERCEPTUAL_QUALITY: NOT_RUN")


if __name__ == "__main__":
    live(parse_args())
