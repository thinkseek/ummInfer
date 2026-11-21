#!/usr/bin/env python3
"""Long-run stability and fault-recovery test for the native H20B service.

Runs inside the container next to the service, because it also reads the
service's own process state (`/proc` RSS) and NVML through `nvidia-smi`. It
talks to the server over its public HTTP surface only; it never starts, stops
or reconfigures the service except for the deliberate fault injection, which
signals only the child PIDs that the supervisor's own state file lists.

  python3 -I tests/soak_native.py --out /root/soak-out --duration-s 1800

Modes:
  soak    mixed request load for --duration-s, with a memory/queue time series
  faults  kill a stage mid-request, time a request out, fill the queue
  all     soak, then faults (default)

Request mix is rate-gated, not weight-sampled: the image leg needs ~225 s and
the TTS leg ~11 s per request, so offering them at a fixed share would just
pile requests into the admission queue. A type is offered only when its stage
could plausibly take it; otherwise the worker falls back to text/ASR/VLM.

Every written file is capped at 64 KiB. Per-request detail is a rolling JSONL
(`.1` holds the previous file), so a long run is bounded.
"""

import argparse
import base64
import json
import math
import os
import random
import re
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

LOG_CAP = 64 * 1024
TERMINAL = ("complete", "error", "cancelled")
# One request type is offered only this often. The stages are serial, so a
# faster offer rate would queue rather than measure anything.
TYPE_MIN_INTERVAL_S = {
    "text": 0.0,
    "asr": 0.0,
    "vlm_image": 0.0,
    "tts": 12.0,
    "image": 240.0,
}
TYPE_MIX = ("text", "text", "text", "asr", "asr", "vlm_image", "vlm_image",
            "tts", "image")
PROMPTS = (
    "Reply with one short sentence about the sea.",
    "Name three primary colours, one per line.",
    "What is two plus two? Answer with one word.",
    "Write one sentence about a red fox in snow.",
    "List two uses of a copper wire.",
    "Describe a rainy city street in one sentence.",
)
ASR_PROMPT = "What did the speaker say?"


def now():
    return time.time()


def med(values):
    return statistics.median(values) if values else float("nan")


def percentile(values, fraction):
    """Nearest-rank percentile: the smallest sample at or above the fraction."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[min(len(ordered) - 1, rank - 1)]


class RollingJsonl:
    """Append-only JSONL bounded at LOG_CAP, keeping one previous file."""

    def __init__(self, path):
        self.path = path
        self.previous = path + ".1"
        self.lock = threading.Lock()
        self.size = 0
        if os.path.exists(path):
            self.size = os.path.getsize(path)

    def write(self, record):
        line = json.dumps(record, sort_keys=True) + "\n"
        encoded = line.encode()
        if len(encoded) > LOG_CAP:
            return
        with self.lock:
            if self.size + len(encoded) > LOG_CAP:
                os.replace(self.path, self.previous)
                self.size = 0
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line)
            self.size += len(encoded)


class Client:
    def __init__(self, base_url, wav_path, image_path, wav_b64, image_b64):
        self.base = base_url.rstrip("/")
        self.wav = wav_path
        self.image = image_path
        self.wav_b64 = wav_b64
        self.image_b64 = image_b64

    def open(self, path, data=None, timeout=600):
        request = urllib.request.Request(
            self.base + path, data=data,
            headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(request, timeout=timeout)

    def json_request(self, path, body, timeout=600):
        # A None body selects GET; json.dumps(None) would send "null" as a POST
        # body and /health, /ready, /metrics would answer 404.
        payload = None if body is None else json.dumps(body).encode()
        started = now()
        try:
            with self.open(path, payload, timeout) as response:
                return response.status, response.read(), now() - started
        except urllib.error.HTTPError as error:
            return error.code, error.read(), now() - started

    def transcription(self, timeout=600):
        boundary = "----soak-native"
        body = bytearray()
        for name, value in (("model", "qwen3-native"),):
            body += (f"--{boundary}\r\nContent-Disposition: form-data; "
                     f"name=\"{name}\"\r\n\r\n{value}\r\n").encode()
        body += (f"--{boundary}\r\nContent-Disposition: form-data; "
                 "name=\"file\"; filename=\"a.wav\"\r\n"
                 "Content-Type: audio/wav\r\n\r\n").encode()
        body += open(self.wav, "rb").read()
        body += f"\r\n--{boundary}--\r\n".encode()
        request = urllib.request.Request(
            self.base + "/v1/audio/transcriptions", data=bytes(body),
            headers={"Content-Type":
                     "multipart/form-data; boundary=" + boundary})
        started = now()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read(), now() - started
        except urllib.error.HTTPError as error:
            return error.code, error.read(), now() - started

    def stream(self, body, timeout=600, on_event=None):
        """Returns (http_status, kinds, text, artifacts, leg_error, error, secs)."""
        payload = json.dumps(body).encode()
        started = now()
        kinds, text, artifacts = [], "", []
        leg_error = None
        error_message = ""
        try:
            with self.open("/v1/chat/completions", payload, timeout) as response:
                pending = []
                for raw in response:
                    line = raw.decode("utf-8", "replace").rstrip("\r\n")
                    if line.startswith("data:"):
                        pending.append(line[5:].lstrip())
                        continue
                    if line or not pending:
                        continue
                    frame = "\n".join(pending)
                    pending = []
                    if frame == "[DONE]":
                        break
                    parsed = json.loads(frame)
                    unified = parsed.get("unified") or {}
                    if unified:
                        kinds.append(unified.get("kind"))
                        if unified.get("kind") == "artifact":
                            artifacts.append(unified.get("artifact") or {})
                        if unified.get("kind") == "leg_error":
                            leg_error = unified
                        if unified.get("kind") == "error":
                            error_message = str(unified.get("error") or "")
                        if on_event is not None:
                            on_event(unified)
                    delta = (parsed.get("choices") or [{}])[0].get("delta") or {}
                    # OpenAI-style content may be a string or a list of parts.
                    piece = delta.get("content")
                    if isinstance(piece, list):
                        piece = "".join(part.get("text", "") for part in piece
                                        if isinstance(part, dict))
                    if isinstance(piece, str):
                        text += piece
                return response.status, kinds, text, artifacts, leg_error, \
                    error_message, now() - started
        except urllib.error.HTTPError as error:
            return error.code, kinds, text, artifacts, leg_error, \
                error.read().decode("utf-8", "replace")[:400], now() - started


# --------------------------------------------------------------------------
# Process / GPU / metrics sampling


def read_state(state_file):
    try:
        with open(state_file, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def process_starttime(pid):
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            data = handle.read()
        end = data.rfind(b")")
        return "proc:" + data[end + 2:].split()[19].decode()
    except (OSError, IndexError):
        return None


def process_rss_mib(pid):
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def gpu_sample(indices):
    try:
        query = "index,memory.used,temperature.gpu,utilization.gpu"
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        rows = {}
        for line in out.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4:
                continue
            index = int(parts[0])
            if index in indices:
                rows[index] = {"memory_mib": int(parts[1]),
                               "temp_c": int(parts[2]),
                               "util_pct": int(parts[3])}
        return rows
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


METRIC_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9.eE+-]+)$")


def scrape_metrics(client):
    try:
        with client.open("/metrics", None, 10) as response:
            text = response.read().decode("utf-8", "replace")
    except (OSError, urllib.error.URLError):
        return {}
    values = {}
    for line in text.splitlines():
        match = METRIC_LINE.match(line.strip())
        if not match:
            continue
        name, labels, value = match.groups()
        try:
            number = float(value)
        except ValueError:
            continue
        values[(name, labels or "")] = number
    return values


def metric(values, name, **labels):
    for (metric_name, label_text), value in values.items():
        if metric_name != name:
            continue
        if all(f'{key}="{val}"' in label_text for key, val in labels.items()):
            return value
    return None


class Sampler(threading.Thread):
    """One time-series row per interval: RSS, GPU, queue depth, in flight."""

    STAGES = ("server", "asr", "qwen38", "image", "tts")

    def __init__(self, client, args, stop_event, rows, lock):
        super().__init__(daemon=True)
        self.client = client
        self.args = args
        self.stop_event = stop_event
        self.rows = rows
        self.lock = lock
        self.started = now()

    def sample_once(self):
        state = read_state(self.args.state_file) or {}
        children = {child.get("name"): child
                    for child in state.get("children", [])}
        row = {"t_s": round(now() - self.started, 1)}
        for stage in self.STAGES:
            child = children.get(stage) or {}
            pid = child.get("pid")
            row[f"rss_{stage}_mib"] = (
                round(process_rss_mib(pid), 1)
                if pid and child.get("alive") else None)
        gpus = gpu_sample(set(self.args.gpus))
        for index in self.args.gpus:
            entry = gpus.get(index) or {}
            row[f"gpu{index}_mem_mib"] = entry.get("memory_mib")
            row[f"gpu{index}_temp_c"] = entry.get("temp_c")
            row[f"gpu{index}_util_pct"] = entry.get("util_pct")
        values = scrape_metrics(self.client)
        row["admission_depth"] = metric(values, "umm_queue_depth",
                                        queue="unified_admission")
        row["coordinator_pending"] = metric(values, "umm_queue_depth",
                                            queue="coordinator_pending")
        for stage in ("asr", "qwen38", "image", "tts"):
            row[f"inflight_{stage}"] = metric(
                values, "umm_stage_in_flight", stage=stage)
        row["in_flight_health"] = health_in_flight(self.client)
        with self.lock:
            self.rows.append(row)
        return row

    def run(self):
        while not self.stop_event.is_set():
            try:
                self.sample_once()
            except Exception as error:  # a sampler must never kill the soak
                print(f"sampler error: {error}", file=sys.stderr)
            self.stop_event.wait(self.args.sample_interval_s)


def health_in_flight(client):
    try:
        status, body, _ = client.json_request("/health", None, 10)
        del status
        return json.loads(body.decode()).get("in_flight")
    except (OSError, ValueError, urllib.error.URLError):
        return None


# --------------------------------------------------------------------------
# Load generation


class Load:
    def __init__(self, client, args, deadline, records, lock, log):
        self.client = client
        self.args = args
        self.deadline = deadline
        self.records = records
        self.lock = lock
        self.log = log
        # Start gated types at their interval, not at zero: warm-up already
        # ran one image, and the soak measures steady state.
        self.next_allowed = {
            kind: now() + interval
            for kind, interval in TYPE_MIN_INTERVAL_S.items() if interval > 0.0}
        self.cursor = 0
        self.gate_lock = threading.Lock()
        self.stop = threading.Event()

    def choose(self):
        with self.gate_lock:
            timestamp = now()
            for _ in range(len(TYPE_MIX)):
                kind = TYPE_MIX[self.cursor % len(TYPE_MIX)]
                self.cursor += 1
                if kind in self.next_allowed and self.next_allowed[kind] > timestamp:
                    continue
                # A request must not start so late that it overruns the run.
                if kind == "image" and timestamp + 260 > self.deadline:
                    continue
                if kind == "tts" and timestamp + 15 > self.deadline:
                    continue
                if kind in ("tts", "image"):
                    self.next_allowed[kind] = timestamp + TYPE_MIN_INTERVAL_S[kind]
                return kind
            # Every gated type is cooling down; text is always acceptable.
            return "text"

    def run_one(self, kind):
        prompt = random.choice(PROMPTS)
        started = now()
        record = {"type": kind, "start_epoch": round(started, 3)}
        if kind == "text":
            status, body, seconds = self.client.json_request(
                "/v1/chat/completions",
                {"model": "qwen3-native", "max_tokens": 48,
                 "messages": [{"role": "user", "content": prompt}]})
            record.update(http=status, seconds=round(seconds, 3))
            ok = status == 200
            if ok:
                try:
                    answer = json.loads(body.decode())
                    content = answer["choices"][0]["message"]["content"]
                    ok = bool(content.strip())
                    record["text_chars"] = len(content)
                except (ValueError, KeyError, IndexError):
                    ok = False
                    record["error"] = "unparseable body"
        elif kind == "asr":
            status, body, seconds = self.client.transcription()
            record.update(http=status, seconds=round(seconds, 3))
            ok = status == 200
            if ok:
                try:
                    ok = bool(json.loads(body.decode()).get("text", "").strip())
                except ValueError:
                    ok = False
                    record["error"] = "unparseable body"
        elif kind == "vlm_image":
            status, kinds, text, _, leg_error, error, seconds = self.client.stream(
                {"model": "qwen3-native", "stream": True, "max_tokens": 64,
                 "messages": [{"role": "user", "content": [
                     {"type": "image_url", "image_url": {
                         "url": "data:image/png;base64," + self.client.image_b64}},
                     {"type": "text", "text": "Describe the image in one sentence."}]}]})
            record.update(http=status, seconds=round(seconds, 3),
                          terminal=kinds[-1] if kinds else None)
            ok = status == 200 and kinds and kinds[-1] == "complete" and \
                bool(text.strip())
            if error:
                record["error"] = error[:200]
        elif kind == "tts":
            status, kinds, text, artifacts, leg_error, error, seconds = \
                self.client.stream(
                    {"model": "qwen3-native", "stream": True, "max_tokens": 32,
                     "messages": [{"role": "user", "content": prompt}],
                     "modalities": ["text", "audio"],
                     "audio": {"voice": "vivian", "format": "wav",
                               "sample_rate": 24000}})
            record.update(http=status, seconds=round(seconds, 3),
                          terminal=kinds[-1] if kinds else None)
            audio = [item for item in artifacts
                     if item.get("media_type") == "audio"]
            ok = status == 200 and kinds and kinds[-1] == "complete" and bool(audio)
            if leg_error:
                record["leg_error"] = json.dumps(leg_error)[:200]
        else:  # image
            status, kinds, text, artifacts, leg_error, error, seconds = \
                self.client.stream(
                    {"model": "qwen3-native", "stream": True, "max_tokens": 16,
                     "messages": [{"role": "user", "content": prompt}],
                     "modalities": ["text", "image"],
                     "image": {"prompt": "A red fox beside a chair in snow",
                               "size": "512x512", "seed": 42,
                               "output_format": "png"}},
                    timeout=900)
            record.update(http=status, seconds=round(seconds, 3),
                          terminal=kinds[-1] if kinds else None)
            image = [item for item in artifacts
                     if item.get("media_type") == "image"]
            ok = status == 200 and kinds and kinds[-1] == "complete" and \
                bool(image) and leg_error is None
            if leg_error:
                record["leg_error"] = json.dumps(leg_error)[:200]
        record["ok"] = bool(ok)
        record["error_kind"] = (
            "none" if ok else
            "rejected" if record.get("http") == 503 else
            "leg_error" if record.get("leg_error") else
            "http" if record.get("http") not in (200, None) else "stream")
        record["end_epoch"] = round(now(), 3)
        with self.lock:
            self.records.append(record)
        self.log.write(record)

    def worker(self):
        while not self.stop.is_set() and now() < self.deadline:
            kind = self.choose()
            try:
                self.run_one(kind)
            except Exception as error:  # one bad request must not end the soak
                with self.lock:
                    self.records.append({
                        "type": kind, "ok": False, "error_kind": "exception",
                        "error": str(error)[:200], "start_epoch": round(now(), 3),
                        "end_epoch": round(now(), 3)})
                self.log.write({"type": kind, "ok": False,
                                "error_kind": "exception",
                                "error": str(error)[:200]})
                # A service outage fails every attempt instantly. Back off so
                # one outage produces a bounded number of failures instead of
                # thousands that drown the success rate.
                self.stop.wait(0.5)


def warm_up(client, records):
    """One request per type, so the first sample is not a cold-model sample."""
    status, _, seconds = client.json_request(
        "/v1/chat/completions",
        {"model": "qwen3-native", "max_tokens": 16,
         "messages": [{"role": "user", "content": "Say hello."}]})
    print(f"warmup text http={status} {seconds:.2f}s")
    status, _, seconds = client.transcription()
    print(f"warmup asr http={status} {seconds:.2f}s")
    status, kinds, _, artifacts, leg_error, error, seconds = client.stream(
        {"model": "qwen3-native", "stream": True, "max_tokens": 16,
         "messages": [{"role": "user", "content": "Say hello."}],
         "modalities": ["text", "audio"],
         "audio": {"voice": "vivian", "format": "wav", "sample_rate": 24000}})
    print(f"warmup tts http={status} terminal={kinds[-1:] } {seconds:.2f}s")
    del artifacts, leg_error, error
    status, kinds, _, artifacts, leg_error, error, seconds = client.stream(
        {"model": "qwen3-native", "stream": True, "max_tokens": 16,
         "messages": [{"role": "user", "content": "Reply with one word: fox."}],
         "modalities": ["text", "image"],
         "image": {"prompt": "A red fox beside a chair in snow",
                   "size": "512x512", "seed": 42, "output_format": "png"}},
        timeout=900)
    produced = [item for item in artifacts if item.get("media_type") == "image"]
    print(f"warmup image http={status} terminal={kinds[-1:]} "
          f"images={len(produced)} leg_error={'yes' if leg_error else 'no'} "
          f"{seconds:.2f}s")
    if leg_error:
        print("warmup image leg_error: " + json.dumps(leg_error)[:400])
    del error
    records.append({"type": "warmup", "ok": bool(produced) and leg_error is None,
                    "seconds": round(seconds, 3)})


# --------------------------------------------------------------------------
# Analysis


def summarise(records):
    # Every attempt counts in the success rate, including one that raised
    # before it had a duration. Latency uses only attempts that returned.
    measured = [record for record in records
                if record.get("type") not in ("warmup",)]
    total = len(measured)
    ok = sum(1 for record in measured if record.get("ok"))
    latencies = [record["seconds"] * 1000.0 for record in measured
                 if record.get("seconds") is not None]
    by_type = {}
    for record in measured:
        bucket = by_type.setdefault(record["type"],
                                    {"count": 0, "ok": 0, "ms": []})
        bucket["count"] += 1
        bucket["ok"] += 1 if record.get("ok") else 0
        if record.get("seconds") is not None:
            bucket["ms"].append(record["seconds"] * 1000.0)
    per_type = {}
    for kind, bucket in by_type.items():
        per_type[kind] = {
            "count": bucket["count"],
            "ok": bucket["ok"],
            "success_rate": round(bucket["ok"] / bucket["count"], 4),
            "p50_ms": round(percentile(bucket["ms"], 0.50), 1),
            "p95_ms": round(percentile(bucket["ms"], 0.95), 1),
            "p99_ms": round(percentile(bucket["ms"], 0.99), 1),
        }
    errors = {}
    for record in records:
        kind = record.get("error_kind")
        if kind and kind != "none":
            errors[kind] = errors.get(kind, 0) + 1
    return {
        "requests": total,
        "success": ok,
        "success_rate": round(ok / total, 4) if total else None,
        "p50_ms": round(percentile(latencies, 0.50), 1),
        "p95_ms": round(percentile(latencies, 0.95), 1),
        "p99_ms": round(percentile(latencies, 0.99), 1),
        "max_ms": round(max(latencies), 1) if latencies else None,
        "per_type": per_type,
        "errors": errors,
    }


def leak_report(rows, keys, unit, threshold):
    """Compare first, middle and last thirds; report monotonic growth."""
    usable = [row for row in rows
              if all(row.get(key) is not None for key in keys)]
    if len(usable) < 6:
        return {"samples": len(usable), "verdict": "too_few_samples"}
    third = max(1, len(usable) // 3)
    head = usable[:third]
    middle = usable[third:2 * third]
    tail = usable[-third:]
    report = {"samples": len(usable), "points": {}}
    flagged = []
    for key in keys:
        first = med([row[key] for row in head])
        mid = med([row[key] for row in middle])
        last = med([row[key] for row in tail])
        delta = last - first
        monotonic = first <= mid <= last
        grow = monotonic and delta > threshold
        report["points"][key] = {
            "start": round(first, 1), "mid": round(mid, 1),
            "end": round(last, 1), "delta": round(delta, 1),
            "monotonic": monotonic, "growth": grow, "unit": unit}
        if grow:
            flagged.append(key)
    report["flagged"] = flagged
    report["verdict"] = "growth" if flagged else "stable"
    return report


# --------------------------------------------------------------------------
# Fault injection


def service_children(state_file):
    state = read_state(state_file) or {}
    return {child.get("name"): child for child in state.get("children", [])}


def restart_counts(state_file):
    """Supervisor restart counter per child, to report outside interference."""
    return {name: int(child.get("restarts") or 0)
            for name, child in service_children(state_file).items()}


def wait_ready(client, timeout):
    deadline = now() + timeout
    while now() < deadline:
        try:
            status, _, _ = client.json_request("/ready", None, 10)
            if status == 200:
                return now()
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.25)
    return None


def wait_in_flight_zero(client, timeout):
    deadline = now() + timeout
    while now() < deadline:
        if health_in_flight(client) == 0:
            return now()
        time.sleep(0.25)
    return None


def next_text_ok(client, timeout=120):
    started = now()
    deadline = started + timeout
    while now() < deadline:
        try:
            status, body, _ = client.json_request(
                "/v1/chat/completions",
                {"model": "qwen3-native", "max_tokens": 16,
                 "messages": [{"role": "user", "content": "Say ok."}]}, 60)
        except (OSError, urllib.error.URLError):
            status, body = 0, b""
        if status == 200:
            try:
                if json.loads(body.decode())["choices"][0]["message"]["content"]:
                    return now()
            except (ValueError, KeyError, IndexError):
                pass
        time.sleep(1.0)
    return None


def fault_kill_stage(client, args):
    """Kill qwen38 while it decodes, then measure supervisor recovery."""
    result = {"fault": "kill_stage", "stage": "qwen38"}
    child = service_children(args.state_file).get("qwen38") or {}
    pid = child.get("pid")
    starttime = child.get("starttime")
    if not pid or process_starttime(pid) != starttime:
        result["error"] = "stage pid does not match the supervisor record"
        return result
    with open(f"/proc/{pid}/comm", "r", encoding="utf-8") as handle:
        comm = handle.read().strip()
    # The kernel truncates comm to 15 bytes ("umm-native-stag"), so the
    # decision uses argv[0], and comm is recorded as evidence only.
    with open(f"/proc/{pid}/cmdline", "rb") as handle:
        argv0 = handle.read().split(b"\0")[0].decode("utf-8", "replace")
    result["comm"] = comm
    result["argv0"] = argv0
    if not os.path.basename(argv0).startswith("umm-native-stage-"):
        result["error"] = "refusing to signal a process that is not a stage"
        return result

    terminal = {"kind": None, "epoch": None, "error": ""}
    started = now()

    def on_event(event):
        if event.get("kind") in TERMINAL and terminal["kind"] is None:
            terminal["kind"] = event.get("kind")
            terminal["epoch"] = now()
            terminal["error"] = str(event.get("error") or "")[:300]

    def request():
        try:
            client.stream(
                {"model": "qwen3-native", "stream": True, "max_tokens": 96,
                 "messages": [{"role": "user", "content":
                               "List the integers from 1 to 200, comma separated."}]},
                # Must fail from the server's death notification, not from this
                # timeout. 30 s is longer than any real detection latency.
                timeout=30, on_event=on_event)
        except (OSError, urllib.error.URLError) as error:
            terminal["client_error"] = str(error)[:200]

    ready_before = None
    try:
        status, _, _ = client.json_request("/ready", None, 5)
        ready_before = status
    except (OSError, urllib.error.URLError):
        pass
    result["ready_before"] = ready_before

    thread = threading.Thread(target=request, daemon=True)
    thread.start()
    # Wait until the stage is certainly decoding before the kill.
    deadline = now() + 20
    while terminal["kind"] is None and now() < deadline:
        time.sleep(0.2)
        if now() - started > 2.0:
            break
    killed_at = now()
    os.kill(pid, signal.SIGKILL)
    result["killed_at_epoch"] = round(killed_at, 3)

    # Recovery is watched concurrently with the in-flight request, so the
    # numbers are not inflated by the client timeout.
    recovery = {"new_pid": None, "restart_detected_s": None,
                "ready_down_s": None, "ready_back_s": None,
                "first_success_s": None}

    def probe_success(timeout):
        try:
            status, _, _ = client.json_request(
                "/v1/chat/completions",
                {"model": "qwen3-native", "max_tokens": 4,
                 "messages": [{"role": "user", "content": "Say ok."}]}, timeout)
        except (OSError, urllib.error.URLError):
            return False
        return status == 200

    def watch():
        deadline = now() + 300
        while now() < deadline:
            current = service_children(args.state_file).get("qwen38") or {}
            candidate = current.get("pid")
            if (recovery["new_pid"] is None and candidate and candidate != pid
                    and current.get("alive")
                    and process_starttime(candidate) == current.get("starttime")):
                recovery["new_pid"] = candidate
                recovery["restart_detected_s"] = round(now() - killed_at, 3)
            try:
                status, _, _ = client.json_request("/ready", None, 5)
            except (OSError, urllib.error.URLError):
                status = 0
            if status == 503 and recovery["ready_down_s"] is None:
                recovery["ready_down_s"] = round(now() - killed_at, 3)
            if (status == 200 and recovery["ready_down_s"] is not None
                    and recovery["ready_back_s"] is None):
                recovery["ready_back_s"] = round(now() - killed_at, 3)
            if (recovery["first_success_s"] is None
                    and recovery["new_pid"] is not None
                    and probe_success(10)):
                recovery["first_success_s"] = round(now() - killed_at, 3)
            if (recovery["new_pid"] and recovery["ready_down_s"] is not None
                    and recovery["ready_back_s"] is not None
                    and recovery["first_success_s"] is not None):
                break
            time.sleep(0.2)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    thread.join(60)
    result["request_returned"] = not thread.is_alive()
    result["terminal"] = terminal
    if terminal["epoch"]:
        result["terminal_latency_s"] = round(terminal["epoch"] - killed_at, 3)
    else:
        result["terminal_latency_s"] = None
        result["hang_within_s"] = 60
    watcher.join(300)
    result.update(recovery)
    # The HTTP cleanup releases the record right after the terminal; allow it a
    # moment before reading the gauge.
    zero_at = wait_in_flight_zero(client, 10)
    result["in_flight_zero_s"] = (
        round(zero_at - killed_at, 3) if zero_at else None)
    result["in_flight_after"] = health_in_flight(client)
    result["ok"] = (
        ready_before == 200
        and terminal["kind"] == "error"
        and result.get("terminal_latency_s") is not None
        and result["terminal_latency_s"] <= 10.0
        and recovery["ready_down_s"] is not None
        and recovery["ready_back_s"] is not None
        and recovery["first_success_s"] is not None
        and result["in_flight_after"] == 0)
    return result


def fault_timeout(client, args):
    """Abandon a long image request at the client; check the server reclaims it."""
    result = {"fault": "request_timeout", "type": "client read timeout"}
    state = scrape_metrics(client)
    aborts_before = metric(state, "umm_aborts_total", source="client") or 0.0
    terminal = {"kind": None}

    def on_event(event):
        if event.get("kind") in TERMINAL:
            terminal["kind"] = event.get("kind")

    error_text = ""
    started = now()
    try:
        client.stream(
            {"model": "qwen3-native", "stream": True, "max_tokens": 16,
             "messages": [{"role": "user", "content": "Reply with one word."}],
             "modalities": ["text", "image"],
             "image": {"prompt": "A red fox beside a chair in snow",
                       "size": "512x512", "seed": 42, "output_format": "png"}},
            timeout=20, on_event=on_event)
    except (socket.timeout, TimeoutError) as error:
        error_text = f"client read timeout: {error}"
    except (OSError, urllib.error.URLError) as error:
        error_text = f"client error: {error}"
    result["client_error"] = error_text
    result["client_timeout_s"] = 20
    result["terminal_before_timeout"] = terminal["kind"]
    zero_at = wait_in_flight_zero(client, 60)
    result["in_flight_zero_s"] = round(zero_at - started, 3) if zero_at else None
    after = scrape_metrics(client)
    aborts_after = metric(after, "umm_aborts_total", source="client") or 0.0
    result["aborts_client_delta"] = aborts_after - aborts_before
    success_at = next_text_ok(client, 120)
    result["first_success_s"] = (
        round(success_at - started, 3) if success_at else None)
    result["ok"] = (
        result["in_flight_zero_s"] is not None
        and result["aborts_client_delta"] >= 1
        and result["first_success_s"] is not None)
    return result


def fault_queue(client, args, concurrency=48):
    """Offer far more streaming requests than max_in_flight and count rejects."""
    result = {"fault": "queue_full", "offered": concurrency}
    barrier = threading.Barrier(concurrency)
    outcomes = []
    lock = threading.Lock()

    def one(index):
        del index
        barrier.wait()
        started = now()
        status, kinds, _, _, _, error, _ = client.stream(
            {"model": "qwen3-native", "stream": True, "max_tokens": 8,
             "messages": [{"role": "user", "content": "Say hi."}]}, timeout=120)
        with lock:
            outcomes.append({"http": status, "terminal": kinds[-1] if kinds else None,
                             "seconds": round(now() - started, 3),
                             "error": (error or "")[:160]})

    threads = [threading.Thread(target=one, args=(index,), daemon=True)
               for index in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(180)
    burst_end = now()
    statuses = {}
    for outcome in outcomes:
        key = str(outcome["http"])
        statuses[key] = statuses.get(key, 0) + 1
    result["http_status_counts"] = statuses
    rejected = [outcome for outcome in outcomes if outcome["http"] == 503]
    result["rejected"] = len(rejected)
    result["rejection_latency_s"] = (
        round(med([outcome["seconds"] for outcome in rejected]), 3)
        if rejected else None)
    result["rejection_messages"] = sorted(
        {outcome["error"] for outcome in rejected})[:3]
    zero_at = wait_in_flight_zero(client, 120)
    result["in_flight_zero_s"] = (
        round(zero_at - burst_end, 3) if zero_at else None)
    success_at = next_text_ok(client, 120)
    result["first_success_s"] = (
        round(success_at - burst_end, 3) if success_at else None)
    result["in_flight_after"] = health_in_flight(client)
    result["ok"] = (
        bool(rejected)
        and all(outcome["http"] in (200, 503) for outcome in outcomes)
        and result["in_flight_after"] == 0
        and result["first_success_s"] is not None)
    return result


# --------------------------------------------------------------------------


def self_check():
    """Small runnable check for the analysis helpers; no network, no service."""
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0
    records = [
        {"type": "text", "ok": True, "seconds": 1.0},
        {"type": "text", "ok": False, "seconds": 3.0, "error_kind": "http",
         "http": 500},
        {"type": "image", "ok": True, "seconds": 200.0},
        {"type": "warmup", "ok": True, "seconds": 5.0},
    ]
    summary = summarise(records)
    assert summary["requests"] == 3, summary
    assert summary["success"] == 2
    assert summary["per_type"]["text"]["count"] == 2
    assert summary["per_type"]["text"]["success_rate"] == 0.5
    assert summary["errors"] == {"http": 1}
    assert round(summary["p99_ms"], 1) == 200000.0
    flat = [{"t_s": index, "rss_server_mib": 100.0} for index in range(9)]
    assert leak_report(flat, ["rss_server_mib"], "MiB", 32.0)["verdict"] == "stable"
    growing = [{"t_s": index, "rss_server_mib": 100.0 + index * 10}
               for index in range(9)]
    report = leak_report(growing, ["rss_server_mib"], "MiB", 32.0)
    assert report["verdict"] == "growth"
    assert report["flagged"] == ["rss_server_mib"]
    write_samples("/tmp/soak-selfcheck.csv", [
        {"a": index, "b": "x" * 1000} for index in range(200)])
    assert os.path.getsize("/tmp/soak-selfcheck.csv") <= LOG_CAP
    print("self_check: all assertions passed")
    return 0


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_samples(path, rows):
    """Write the time series, downsampled to stay under LOG_CAP."""
    if not rows:
        return
    keys = list(rows[0].keys())

    def line(row):
        return ",".join("" if row.get(key) is None else str(row[key])
                        for key in keys)

    header = ",".join(keys) + "\n"
    body = [line(row) + "\n" for row in rows]
    budget = LOG_CAP - len(header)
    if sum(len(item) for item in body) > budget:
        # Keep the first and last sample; take an even stride in between.
        stride = 2
        while sum(len(item) for item in body[::stride]) > budget:
            stride += 1
        body = body[::stride]
        body.append(line(rows[-1]) + "\n")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(header)
        handle.writelines(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url",
                        default=os.environ.get("UMM_SOAK_URL", "http://127.0.0.1:43301"),
                        help="service base URL (override with --base-url or UMM_SOAK_URL)")
    parser.add_argument("--mode", choices=("soak", "faults", "all"), default="all")
    parser.add_argument("--duration-s", type=float, default=1800.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--sample-interval-s", type=float, default=20.0)
    parser.add_argument("--out", help="output directory")
    parser.add_argument("--state-file",
                        default=os.environ.get("UMM_SOAK_STATE_FILE",
                                               "/opt/umminfer/native-service/run/state.json"))
    parser.add_argument("--wav",
                        default=os.environ.get("UMM_SOAK_WAV",
                                               "/opt/umminfer/inputs/asr_en_pcm16.wav"))
    parser.add_argument("--image",
                        default=os.environ.get("UMM_SOAK_IMAGE",
                                               "/opt/umminfer/inputs/man_ironing.png"))
    parser.add_argument("--gpus", default="2,3,4,5")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--self-check", action="store_true",
                        help="run the offline analysis self-check and exit")
    args = parser.parse_args()
    if args.self_check:
        return self_check()
    if not args.out:
        parser.error("--out is required")
    args.gpus = [int(item) for item in args.gpus.split(",") if item.strip()]

    os.makedirs(args.out, exist_ok=True)
    wav_b64 = base64.b64encode(open(args.wav, "rb").read()).decode()
    image_b64 = base64.b64encode(open(args.image, "rb").read()).decode()
    client = Client(args.base_url, args.wav, args.image, wav_b64, image_b64)
    random.seed(20261117)

    if wait_ready(client, 900) is None:
        print("FAIL /ready did not become 200")
        return 1

    records, rows, lock = [], [], threading.Lock()
    log = RollingJsonl(os.path.join(args.out, "soak-requests.jsonl"))
    summary = {"environment": {
        "base_url": args.base_url,
        "duration_s": args.duration_s,
        "concurrency": args.concurrency,
        "sample_interval_s": args.sample_interval_s,
        "wav": args.wav, "image": args.image, "gpus": args.gpus,
        "start_epoch": round(now(), 3)}}

    if args.mode in ("soak", "all"):
        restarts_before = restart_counts(args.state_file)
        if not args.skip_warmup:
            warm_up(client, records)
        stop = threading.Event()
        sampler = Sampler(client, args, stop, rows, lock)
        sampler.sample_once()          # explicit start point
        sampler.start()
        deadline = now() + args.duration_s
        load = Load(client, args, deadline, records, lock, log)
        workers = [threading.Thread(target=load.worker, daemon=True)
                   for _ in range(args.concurrency)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(args.duration_s + 600)
        load.stop.set()
        stop.set()
        sampler.join(30)
        sampler.sample_once()          # explicit end point
        summary["soak"] = summarise(records)
        summary["rss"] = leak_report(
            rows, [f"rss_{stage}_mib" for stage in Sampler.STAGES],
            "MiB", 32.0)
        summary["gpu_memory"] = leak_report(
            rows, [f"gpu{index}_mem_mib" for index in args.gpus], "MiB", 256.0)
        summary["queue"] = {
            "admission_max": max(
                (row["admission_depth"] for row in rows
                 if row.get("admission_depth") is not None), default=None),
            "coordinator_pending_max": max(
                (row["coordinator_pending"] for row in rows
                 if row.get("coordinator_pending") is not None), default=None),
            "inflight_max": {
                stage: max((row[f"inflight_{stage}"] for row in rows
                            if row.get(f"inflight_{stage}") is not None),
                           default=None)
                for stage in ("asr", "qwen38", "image", "tts")},
        }
        restarts_after = restart_counts(args.state_file)
        summary["service"] = {
            "restarts_before": restarts_before,
            "restarts_after": restarts_after,
            "restart_delta": {
                name: restarts_after.get(name, 0) - restarts_before.get(name, 0)
                for name in restarts_after},
        }
        print(json.dumps(summary["service"], indent=2, sort_keys=True))
        write_samples(os.path.join(args.out, "soak-samples.csv"), rows)
        for line in (json.dumps(summary["soak"], indent=2, sort_keys=True),
                     json.dumps(summary["rss"], indent=2, sort_keys=True),
                     json.dumps(summary["gpu_memory"], indent=2, sort_keys=True),
                     json.dumps(summary["queue"], indent=2, sort_keys=True)):
            print(line)

    if args.mode in ("faults", "all"):
        faults = []
        print("fault: kill stage")
        faults.append(fault_kill_stage(client, args))
        print(json.dumps(faults[-1], indent=2, sort_keys=True))
        wait_ready(client, 300)
        print("fault: request timeout")
        faults.append(fault_timeout(client, args))
        print(json.dumps(faults[-1], indent=2, sort_keys=True))
        wait_in_flight_zero(client, 120)
        print("fault: queue full")
        faults.append(fault_queue(client, args))
        print(json.dumps(faults[-1], indent=2, sort_keys=True))
        summary["faults"] = faults
        write_json(os.path.join(args.out, "fault-report.json"), faults)

    failed_faults = []
    if "faults" in summary:
        failed_faults = [fault.get("fault") for fault in summary["faults"]
                         if not fault.get("ok")]
        summary["faults_ok"] = not failed_faults
    write_json(os.path.join(args.out, "soak-summary.json"), summary)
    print("summary: " + os.path.join(args.out, "soak-summary.json"))
    if failed_faults:
        print("FAILED faults: " + ", ".join(failed_faults))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
