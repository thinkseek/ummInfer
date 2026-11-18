#!/usr/bin/env python3
"""Shared pure helpers for local validation and multimodal benchmark suites."""

import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time

LOG_CAP = 64 * 1024


def percentile(values, p):
    """Nearest-rank percentile: no interpolation, no invented samples."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100.0 * len(ordered)))
    return ordered[rank - 1]


def capped_append(buffer, data, cap=LOG_CAP):
    """Keep only the last `cap` bytes; a log must never grow past the cap."""
    buffer.extend(data)
    overflow = len(buffer) - cap
    if overflow > 0:
        del buffer[:overflow]
    return buffer


def sse_events(chunks):
    """Turn an iterable of decoded SSE lines into parsed data payloads."""
    lines = []
    for line in chunks:
        if not line:
            if not lines:
                continue
            payload = "\n".join(lines)
            lines = []
            yield payload
            continue
        if line.startswith("data:"):
            lines.append(line[5:].lstrip())
    if lines:
        yield "\n".join(lines)


class CappedProcess:
    """A child process whose retained output is bounded by LOG_CAP bytes."""

    def __init__(self, args, name, cwd=None):
        self.name = name
        self.buffer = bytearray()
        self.lock = threading.Lock()
        self.process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd)
        self.reader = threading.Thread(target=self._drain, daemon=True)
        self.reader.start()

    def _drain(self):
        while True:
            try:
                data = os.read(self.process.stdout.fileno(), 65536)
            except OSError:
                return
            if not data:
                return
            with self.lock:
                capped_append(self.buffer, data)

    def tail(self):
        with self.lock:
            return bytes(self.buffer).decode(errors="replace")

    def stop(self, log_path):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.reader.join(5)
        try:
            self.process.stdout.close()
        except OSError:
            pass
        text = self.tail()
        Path(log_path).write_text(text, encoding="utf-8")
        if os.path.getsize(log_path) > LOG_CAP:
            raise RuntimeError(f"{self.name} log exceeded its cap")
        return self.process.returncode


def _rss_window(path, start, end):
    peak = {}
    if not os.path.exists(path):
        return peak
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split(",")
            if len(parts) != 4:
                continue
            try:
                at = float(parts[0])
                rss = int(parts[3])
            except ValueError:
                continue
            if not start <= at <= end:
                continue
            label = parts[1]
            peak[label] = max(peak.get(label, 0), rss)
    return peak


def wait_ready(port, server, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.process.poll() is not None:
            raise RuntimeError(f"server exited: {server.tail()[-2000:]}")
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            connection.request("GET", "/ready")
            response = connection.getresponse()
            response.read()
            connection.close()
            if response.status == 200:
                return
        except OSError:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"server never became ready: {server.tail()[-2000:]}")


def server_port(server, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.process.poll() is not None:
            raise RuntimeError(f"server exited: {server.tail()[-2000:]}")
        match = re.search(r"serving on [^\s:]+:(\d+)", server.tail())
        if match:
            return int(match.group(1))
        time.sleep(0.05)
    raise RuntimeError("server did not report its port")


def sha256_path(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
