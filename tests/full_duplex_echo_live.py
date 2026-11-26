#!/usr/bin/env python3
"""Real-model realtime echo/double-talk gate over deterministic PCM.

Native Qwen3-ASR, Qwen3.8 and Qwen3-TTS run on three explicitly selected idle
physical GPUs. The client drives /v1/realtime, captures actual audio.delta PCM,
and constructs 16 kHz microphone PCM from that response. Each echo-only case is
first injected with the default echo protection, then with protection disabled
on the *same active response*. The two arms therefore receive byte-identical
microphone bytes and the same real far-end reference.

Timed virtual PCM consumption is used only for playback_ack. No speaker,
microphone, acoustic path, listening quality, or physical AEC claim is made.
"""

import argparse
import base64
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request
import wave

LOG_CAP = 262144
FRAME_BYTES = 320                 # 10 ms, PCM16 mono 16 kHz
CHUNK_BYTES = 3200                # 100 ms pacing
SILENCE_MS = 400
ECHO_TAIL_SAMPLES = 8000          # 500 ms from the response's 1 s ring
END_SILENCE = bytes(16000 * 2 * 6 // 10)
CUTIN_TYPES = {
    "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped",
    "response.cancelled",
    "transcription.completed",
}
STALE_TYPES = {
    "audio.delta", "audio.done", "response.text.delta",
    "response.playback_ack", "response.completed", "response.failed",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def hash_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pcm_from_wav(path):
    with wave.open(str(path), "rb") as wav:
        require((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) ==
                (1, 2, 16000), "audio must be PCM16 mono 16 kHz WAV")
        return wav.readframes(wav.getnframes())


def pcm16_bytes(samples):
    return b"".join(struct.pack("<h", max(-32768, min(32767, int(v))))
                    for v in samples)


def pcm16_values(data):
    require(len(data) % 2 == 0, "odd PCM16 byte count")
    return [v[0] for v in struct.iter_unpack("<h", data)]


def rms(data):
    values = pcm16_values(data)
    return math.sqrt(sum(v * v for v in values) / max(1, len(values)))


def trim_digital_lead_in(data):
    """Remove only sub-threshold 10 ms frames before the first real speech."""
    threshold = 32768.0 * (10.0 ** (-55.0 / 20.0))
    for offset in range(0, len(data), FRAME_BYTES):
        frame = data[offset:offset + FRAME_BYTES]
        if len(frame) == FRAME_BYTES and rms(frame) > threshold:
            return data[offset:]
    raise AssertionError("near-end fixture has no frame above the VAD floor")


def last_voiced_ms(data):
    """Last 10 ms frame end above EnergyVad's absolute -55 dBFS floor."""
    threshold = 32768.0 * (10.0 ** (-55.0 / 20.0))
    last = 0
    for offset in range(0, len(data), FRAME_BYTES):
        frame = data[offset:offset + FRAME_BYTES]
        if len(frame) == FRAME_BYTES and rms(frame) > threshold:
            last = offset // FRAME_BYTES + 1
    require(last > 0, "echo fixture contains no voiced frame")
    return float(last * 10)


def resample_reference(data, sample_rate, channels):
    """Byte-for-byte input model of EchoRobustVad::push_reference."""
    require(8000 <= sample_rate <= 96000 and 1 <= channels <= 32,
            "invalid far-end format")
    require(len(data) % (2 * channels) == 0, "unaligned far-end PCM16")
    if sample_rate == 16000 and channels == 1:
        return [float(v) for v in pcm16_values(data)]
    output = []
    previous = 0
    have_previous = False
    phase = 0
    for offset in range(0, len(data), 2 * channels):
        frame = struct.unpack_from("<" + "h" * channels, data, offset)
        mono = int(sum(frame) / channels)
        if sample_rate == 16000:
            output.append(float(mono))
            continue
        if not have_previous:
            previous = mono
            have_previous = True
            phase = 0
            continue
        while phase < 16000:
            fraction = phase / 16000.0
            output.append(previous + (mono - previous) * fraction)
            phase += sample_rate
        phase -= 16000
        previous = mono
    return output


@dataclass(frozen=True)
class EchoCase:
    name: str
    delays_ms: tuple
    gain: float
    clip_ratio: float = 0.0

    def delay_samples(self, frame):
        return int(round(self.delays_ms[(frame // 10) % len(self.delays_ms)] * 16))

    @property
    def max_delay_samples(self):
        return max(int(round(v * 16)) for v in self.delays_ms)


ECHO_CASES = (
    EchoCase("delay_000ms_gain_035", (0,), 0.35),
    EchoCase("delay_073ms_gain_055", (73,), 0.55),
    EchoCase("delay_249ms_gain_140", (249,), 1.40),
    EchoCase("delay_499ms_gain_008", (499,), 0.08),
    EchoCase("delay_117ms_clipped", (117,), 1.0, 0.45),
    EchoCase("delay_jumps", (13, 117, 249, 499, 73), 0.45),
)


def make_echo(reference, case):
    """A 500 ms echo window whose first sample is exactly 500 ms old.

    At the first near-end frame the oldest lag is tail + max(delay), which is
    fixed at 8000 samples. Later frames only get newer. This keeps every fixed
    delay (including 499 ms) inside the production 0..500 ms search window and
    the one-second reference ring, while preserving continuous pre-window echo.
    """
    require(len(reference) >= 16000, "real TTS response needs at least 1 s PCM")
    tail_samples = ECHO_TAIL_SAMPLES - case.max_delay_samples
    require(0 < tail_samples <= ECHO_TAIL_SAMPLES, "invalid echo geometry")
    start = len(reference) - tail_samples
    active_samples = ECHO_TAIL_SAMPLES
    selected = reference[start - case.max_delay_samples:len(reference)]
    peak = max(1.0, max(abs(v) for v in selected))
    clip_limit = max(1, int(peak * case.clip_ratio)) if case.clip_ratio else 0
    echo = []
    clipped = 0
    max_reference_lag = 0
    for i in range(active_samples):
        delay = case.delay_samples(i // 160)
        if i % 160 == 0:
            max_reference_lag = max(max_reference_lag,
                                    tail_samples + delay - i - 160)
        source = start + i - delay
        value = case.gain * reference[source] if 0 <= source < len(reference) else 0.0
        if clip_limit and abs(value) > clip_limit:
            value = math.copysign(clip_limit, value)
            clipped += 1
        echo.append(int(round(value)))
    if case.clip_ratio:
        require(clipped > 0, f"{case.name} did not exercise clipping")
    require(rms(pcm16_bytes(echo)) > 60.0,
            f"{case.name} is too quiet to exercise the energy-VAD A/B baseline")
    require(max_reference_lag <= 8000 and
            start - case.max_delay_samples == len(reference) - 8000,
            f"{case.name} escaped the bounded playback-reference window")
    active = pcm16_bytes(echo)
    return active + END_SILENCE, {
        "delay_ms": list(case.delays_ms),
        "gain": case.gain,
        "clip_ratio": case.clip_ratio,
        "clip_limit_pcm16": clip_limit,
        "clipped_samples": clipped,
        "active_bytes": len(active),
        "active_ms": len(active) / 32.0,
        "last_voiced_ms": last_voiced_ms(active),
        "reference_samples": len(reference),
        "reference_tail_start": start,
        "reference_oldest_sample": start - case.max_delay_samples,
        "max_reference_lag_samples": max_reference_lag,
        "active_rms": rms(active),
    }


def make_double_talk(echo, user):
    """Near-end starts at sample zero, simultaneous with the real-TTS echo."""
    active_bytes = len(echo) - len(END_SILENCE)
    near = user[:active_bytes]
    require(len(near) >= CHUNK_BYTES, "not enough near-end speech")
    left = pcm16_values(echo[:len(near)])
    right = pcm16_values(near)
    mixed = [int(round(a + b)) for a, b in zip(left, right)]
    return pcm16_bytes(mixed) + echo[len(near):]


class Evidence:
    def __init__(self, root):
        self.root = Path(root)

    def bytes(self, relative, data):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def json(self, relative, value):
        return self.bytes(relative, (json.dumps(value, indent=2, sort_keys=True) +
                                    "\n").encode())

    def jsonl(self, relative, rows):
        return self.bytes(relative, b"".join(
            (json.dumps(row, sort_keys=True) + "\n").encode() for row in rows))

    def manifest(self):
        rows = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                rows.append(f"{hash_file(path)}  {path.relative_to(self.root)}\n")
        (self.root / "SHA256SUMS").write_text("".join(rows))
        return len(rows)


class CappedLog:
    def __init__(self, path):
        self.path = Path(path)
        self.data = bytearray()
        self.lock = threading.Lock()
        self.thread = None

    def start(self, proc):
        def drain():
            with open(self.path, "wb") as out:
                while True:
                    chunk = proc.stdout.readline(4096)
                    if not chunk:
                        return
                    with self.lock:
                        if len(self.data) + len(chunk) <= LOG_CAP:
                            self.data.extend(chunk)
                    out.write(chunk)
                    out.flush()
        self.thread = threading.Thread(target=drain, daemon=True)
        self.thread.start()

    def text(self):
        with self.lock:
            return bytes(self.data).decode("utf-8", "replace")

    def wait(self):
        if self.thread:
            self.thread.join(5)


class Proc:
    def __init__(self, name, args, log_path):
        self.name = name
        self.args = [str(arg) for arg in args]
        self.log = CappedLog(log_path)
        self.proc = subprocess.Popen(
            self.args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=dict(os.environ), start_new_session=True)
        self.log.start(self.proc)

    @property
    def pid(self):
        return self.proc.pid


def free_ports(count):
    sockets = []
    try:
        for _ in range(count):
            sock = socket.socket()
            sockets.append(sock)
            sock.bind(("127.0.0.1", 0))
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def health(port):
    with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=2) as reply:
        return json.load(reply)


def wait_in_flight_zero(port, timeout_s=10):
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        last = health(port)
        if last.get("in_flight") == 0:
            return last
        time.sleep(0.05)
    raise AssertionError(f"in_flight did not return to zero: {last}")


def preflight_gpus(gpus):
    require(len(set(gpus)) == 3 and all(g in (0, 1, 7) for g in gpus),
            "exactly distinct physical GPUs 0, 1 and 7 are required")
    evidence = {}
    for gpu in gpus:
        row = subprocess.check_output([
            "nvidia-smi", "-i", str(gpu),
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True).strip()
        fields = [v.strip() for v in row.split(",")]
        require(len(fields) == 3, f"unexpected GPU row: {row}")
        compute = subprocess.check_output([
            "nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
            "--format=csv,noheader"], text=True).strip()
        used, utilization = int(fields[1]), int(fields[2])
        require(used <= 64 and utilization == 0 and not compute,
                f"GPU {gpu} is not idle: memory={used}, util={utilization}, "
                f"compute={compute or 'none'}")
        evidence[str(gpu)] = {"memory_used_mib": used,
                              "utilization_percent": utilization,
                              "compute_processes": 0}
    return evidence


def wait_for_idle_gpus(gpus, timeout_s=15):
    """Allow a final NVML utilization window to drain after process exit."""
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        try:
            return preflight_gpus(gpus)
        except AssertionError as error:
            last = error
            time.sleep(0.5)
    raise AssertionError(f"selected GPUs did not return to idle: {last}")


def start_stack(args, ports):
    unique = f"{os.getpid()}-{time.time_ns()}"
    ipc = args.log_dir / f"ipc-{unique}"
    shm = args.log_dir / f"shm-{unique}"
    ipc.mkdir(parents=True, exist_ok=False)
    shm.mkdir(parents=True, exist_ok=False)
    entry = f"ipc://{ipc}/entry"
    vlm = f"ipc://{ipc}/vlm"
    tts = f"ipc://{ipc}/tts"
    completion = f"ipc://{ipc}/completion"
    abort = f"ipc://{ipc}/abort"
    procs = {}
    try:
        procs["asr"] = Proc("asr", [
            args.bin_dir / "umm-native-stage-asr", "--kind", "asr", "--name", "asr",
            "--model", args.asr_model, "--gpu", args.gpu_asr,
            "--recv-endpoint", entry, "--coordinator-endpoint", completion,
            "--abort-endpoint", abort], args.log_dir / "asr.log")
        procs["vlm"] = Proc("vlm", [
            args.bin_dir / "umm-native-stage-qwen38", "--kind", "qwen38", "--name", "vlm",
            "--model", args.vlm_model, "--gpu", args.gpu_vlm,
            "--recv-endpoint", vlm, "--coordinator-endpoint", completion,
            "--abort-endpoint", abort, "--max-new-tokens", args.vlm_max_tokens],
            args.log_dir / "vlm.log")
        procs["tts"] = Proc("tts", [
            args.bin_dir / "umm-native-stage-tts", "--kind", "tts", "--name", "tts",
            "--model", args.tts_model, "--gpu", args.gpu_tts,
            "--recv-endpoint", tts, "--coordinator-endpoint", completion,
            "--abort-endpoint", abort, "--max-frames", args.tts_max_frames],
            args.log_dir / "tts.log")
        procs["server"] = Proc("server", [
            args.bin_dir / "umm-infer", "serve", "--host", "127.0.0.1",
            "--port", ports[0], "--realtime-port", ports[1],
            "--model", "qwen3-unified", "--entry", "asr",
            "--entry-endpoint", entry, "--vlm-entry", "vlm",
            "--vlm-entry-endpoint", vlm, "--tts-entry", "tts",
            "--tts-entry-endpoint", tts, "--completion", completion,
            "--abort", abort, "--session-history", "--max-in-flight", 8,
            "--ipc-dir", shm], args.log_dir / "server.log")
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
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
                            return procs, (ipc, shm)
                except Exception:
                    pass
            time.sleep(0.3)
        raise AssertionError("real stages or server never became ready")
    except Exception:
        stop_stack(procs)
        shutil.rmtree(ipc, ignore_errors=True)
        shutil.rmtree(shm, ignore_errors=True)
        raise


def stop_stack(procs):
    for proc in reversed(list(procs.values())):
        if proc.proc.poll() is None:
            proc.proc.send_signal(signal.SIGTERM)
    for proc in procs.values():
        try:
            proc.proc.wait(20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.proc.wait(5)
        proc.log.wait()


class ResourceSampler:
    def __init__(self, procs, path):
        self.procs = procs
        self.path = Path(path)
        self.stop_event = threading.Event()
        self.rows = []
        self.previous = {}
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    @staticmethod
    def read(pid):
        with open(f"/proc/{pid}/stat", "rb") as source:
            stat = source.read().split()
        with open(f"/proc/{pid}/status", "rb") as source:
            status = source.read().decode()
        rss_kib = 0
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                rss_kib = int(line.split()[1])
                break
        ticks = os.sysconf("SC_CLK_TCK")
        return (int(stat[13]) + int(stat[14])) / ticks, rss_kib * 1024

    def run(self):
        while not self.stop_event.is_set():
            now = time.monotonic()
            row = {"time_ms": round((now - self.started) * 1000, 3),
                   "processes": {}}
            for name, proc in self.procs.items():
                if proc.proc.poll() is not None:
                    continue
                try:
                    cpu_s, rss = self.read(proc.pid)
                except (FileNotFoundError, ProcessLookupError):
                    continue
                previous = self.previous.get(name)
                cpu_percent = None
                if previous:
                    elapsed = now - previous[0]
                    if elapsed > 0:
                        cpu_percent = max(0.0, (cpu_s - previous[1]) / elapsed * 100.0)
                self.previous[name] = (now, cpu_s)
                row["processes"][name] = {
                    "pid": proc.pid, "rss_bytes": rss,
                    "cpu_percent": None if cpu_percent is None else round(cpu_percent, 3)}
            self.rows.append(row)
            self.stop_event.wait(0.05)

    def finish(self):
        self.stop_event.set()
        self.thread.join(5)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                                     for row in self.rows))
        summary = {}
        for name in self.procs:
            samples = [row["processes"][name] for row in self.rows
                       if name in row["processes"]]
            cpu = sorted(v["cpu_percent"] for v in samples
                         if v["cpu_percent"] is not None)
            rss = [v["rss_bytes"] for v in samples]
            if not samples:
                continue
            summary[name] = {
                "samples": len(samples),
                "cpu_percent_mean": round(sum(cpu) / max(1, len(cpu)), 3),
                "cpu_percent_p95": round(cpu[min(len(cpu) - 1,
                                                   int(len(cpu) * 0.95))], 3) if cpu else None,
                "cpu_percent_max": round(max(cpu), 3) if cpu else None,
                "rss_min_bytes": min(rss), "rss_max_bytes": max(rss),
                "rss_growth_bytes": max(0, max(rss) - rss[0]),
            }
        return summary


@dataclass
class EventRecord:
    index: int
    time_s: float
    raw: str
    event: dict


class Recorder:
    def __init__(self, ws):
        self.ws = ws
        self.records = []
        self.condition = threading.Condition()
        self.closed = False
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.stop_event.is_set():
            try:
                raw = self.ws.recv(timeout=0.2)
                event = json.loads(raw)
            except TimeoutError:
                continue
            except Exception:
                with self.condition:
                    self.closed = True
                    self.condition.notify_all()
                return
            with self.condition:
                self.records.append(EventRecord(
                    len(self.records), time.monotonic(), raw, event))
                self.condition.notify_all()

    def mark(self):
        with self.condition:
            return len(self.records)

    def wait_for(self, predicate, after, timeout_s, description):
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while True:
                for record in self.records[after:]:
                    if predicate(record.event):
                        return record
                if self.closed or time.monotonic() >= deadline:
                    recent = []
                    for record in self.records[after:][-8:]:
                        summary = dict(record.event)
                        if "audio" in summary:
                            summary["audio"] = f"<{len(summary['audio'])} base64 chars>"
                        recent.append(summary)
                    raise AssertionError(f"timed out waiting for {description}; recent={recent}")
                self.condition.wait(min(0.2, max(0.01, deadline - time.monotonic())))

    def since(self, after, types=None, response_id=None):
        with self.condition:
            return [r for r in self.records[after:]
                    if (types is None or r.event.get("type") in types) and
                    (response_id is None or r.event.get("response_id") == response_id)]

    def close(self):
        self.stop_event.set()
        self.thread.join(5)

    def journal(self):
        return [{"index": r.index, "time_ms": round(r.time_s * 1000, 3),
                 "raw": r.raw} for r in self.records]


class Client:
    def __init__(self, ws_port, name, evidence):
        from websockets.sync.client import connect
        self.name = name
        self.evidence = evidence
        self.url = (f"ws://127.0.0.1:{ws_port}/v1/realtime?"
                    "intent=transcription&model=qwen3-unified")
        self.ws = connect(self.url, compression=None, open_timeout=10,
                          close_timeout=10, ping_interval=None, max_size=1 << 22,
                          legacy=True)
        self.rec = Recorder(self.ws)
        self.tx = []
        self.response = None
        marker = self.rec.mark()
        self.ws.send(json.dumps({"type": "session.update", "session": {
            "input_audio_format": "pcm16",
            "turn_detection": {"type": "server_vad", "silence_ms": SILENCE_MS},
            "max_new_tokens": 96,
            "audio_output": {"voice": "vivian", "playback_ack": True}}}))
        self.rec.wait_for(lambda e: e.get("type") == "transcription_session.updated",
                          marker, 20, "initial session update")

    def close(self):
        try:
            self.ws.close()
        finally:
            self.rec.close()

    def set_echo_suppression(self, enabled):
        marker = self.rec.mark()
        self.ws.send(json.dumps({"type": "session.update", "session": {
            "turn_detection": {"type": "server_vad", "silence_ms": SILENCE_MS,
                               "echo_suppression": enabled}}}))
        reply = self.rec.wait_for(
            lambda e: e.get("type") == "transcription_session.updated", marker, 10,
            "active echo_suppression update").event
        turn = reply["session"]["turn_detection"]
        require(turn.get("echo_suppression", True) is enabled,
                "effective session did not apply echo_suppression")

    def append(self, pcm):
        self.ws.send(json.dumps({"type": "input_audio_buffer.append",
                                 "audio": base64.b64encode(pcm).decode()}))

    def stream(self, pcm, label):
        started = time.monotonic()
        for offset in range(0, len(pcm), CHUNK_BYTES):
            chunk = pcm[offset:offset + CHUNK_BYTES]
            self.tx.append({"label": label, "time_ms": round(time.monotonic() * 1000, 3),
                            "offset": offset, "bytes": len(chunk), "sha256": sha256(chunk)})
            self.append(chunk)
            time.sleep(len(chunk) / 32000.0)
        return started

    def send_user(self, user):
        marker = self.rec.mark()
        started = self.stream(user[:64000] + END_SILENCE, "prime-user")
        completed = self.rec.wait_for(
            lambda e: e.get("type") == "transcription.completed", marker, 120,
            "priming transcription")
        require(completed.event.get("text", "").strip(), "real ASR returned empty priming text")
        return completed, started

    def collect_response(self, after):
        first = self.rec.wait_for(
            lambda e: e.get("type") in ("response.text.delta", "audio.delta") and
            e.get("response_id"), after, 120, "real VLM response")
        response_id = first.event["response_id"]
        response_origin_s = (self.rec.records[after - 1].time_s if after > 0
                             else first.time_s)
        terminal = self.rec.wait_for(
            lambda e: e.get("response_id") == response_id and
            e.get("type") in ("response.completed", "response.failed"),
            first.index + 1, 180, "real TTS response completion")
        require(terminal.event["type"] == "response.completed",
                f"real response failed: {terminal.event}")
        records = self.rec.since(after, response_id=response_id)
        audio_events = [r for r in records if r.event.get("type") == "audio.delta"]
        require(audio_events, "real response produced no audio.delta")
        rates = {int(r.event["sample_rate"]) for r in audio_events}
        channels = {int(r.event["channels"]) for r in audio_events}
        require(rates == {24000} and channels == {1}, "unexpected real TTS PCM format")
        pcm = b"".join(base64.b64decode(r.event["audio"]) for r in audio_events)
        text = "".join(r.event.get("text", "") for r in records
                       if r.event.get("type") == "response.text.delta")
        final_cursor = int(terminal.event.get("text_cursor", len(text.encode())))
        require(final_cursor == len(text.encode()), "final cursor is not a UTF-8 byte length")
        candidate = None
        consumed = []
        for record in audio_events:
            consumed.append(record)
            cursor = int(record.event.get("text_cursor", 0))
            if 0 < cursor < final_cursor:
                candidate = (record, cursor)
                break
        require(candidate is not None,
                "response had no non-final positive acknowledged-prefix cursor")
        audio_record, cursor = candidate
        for record in consumed:
            raw = base64.b64decode(record.event["audio"])
            time.sleep(len(raw) / (2.0 * 24000))
        ack_marker = self.rec.mark()
        self.ws.send(json.dumps({"type": "output_audio_buffer.playback_ack",
                                 "response_id": response_id, "text_cursor": cursor}))
        ack = self.rec.wait_for(
            lambda e: e.get("type") == "response.playback_ack" and
            e.get("response_id") == response_id and e.get("text_cursor") == cursor,
            ack_marker, 10, "partial playback acknowledgement").event
        require(ack.get("final") is False, "partial ACK incorrectly finalized history")
        acknowledged = text.encode()[:cursor].decode("utf-8")
        self.response = {
            "response_id": response_id, "pcm": pcm, "sample_rate": 24000,
            "channels": 1, "text": text, "text_cursor": cursor,
            "acknowledged_text": acknowledged, "segment_id": terminal.event.get("segment_id"),
            "audio_chunks": len(audio_events), "audio_bytes": len(pcm),
            "audio_sha256": sha256(pcm),
            "first_audio_after_transcript_ms": round(
                (audio_events[0].time_s - response_origin_s) * 1000, 3),
            "terminal_index": terminal.index,
        }
        return self.response

    def prime(self, user):
        completed, _ = self.send_user(user)
        return self.collect_response(completed.index + 1)

    def check_cancel(self, response, marker, started):
        cancelled = self.rec.wait_for(
            lambda e: e.get("type") == "response.cancelled" and
            e.get("response_id") == response["response_id"], marker, 1.0,
            "response cancellation")
        latency = (cancelled.time_s - started) * 1000.0
        require(latency <= 1000.0, f"cancellation exceeded L1: {latency:.1f} ms")
        require(cancelled.event.get("reason") == "barge_in", "wrong cancellation reason")
        require(cancelled.event.get("history_recorded") is True,
                "acknowledged realtime prefix was not recorded")
        require(cancelled.event.get("text_cursor") == response["text_cursor"],
                "cancellation changed the acknowledged cursor")
        require(cancelled.event.get("acknowledged_text") == response["acknowledged_text"],
                "cancellation did not preserve the exact acknowledged prefix")
        speech = self.rec.wait_for(
            lambda e: e.get("type") == "input_audio_buffer.speech_started", marker, 2,
            "speech start")
        return cancelled, speech, latency

    def finish_endpoint(self, marker, response=None, cancelled=None):
        stopped = self.rec.wait_for(
            lambda e: e.get("type") == "input_audio_buffer.speech_stopped", marker, 20,
            "speech endpoint")
        completed = self.rec.wait_for(
            lambda e: e.get("type") == "transcription.completed" and
            e.get("segment_id") == stopped.event.get("segment_id"), stopped.index + 1,
            120, "endpoint final transcription")
        if response is not None and cancelled is not None:
            stale = self.rec.since(cancelled.index + 1, types=STALE_TYPES,
                                   response_id=response["response_id"])
            require(not stale, f"stale response events survived cancellation: "
                               f"{[r.event.get('type') for r in stale]}")
        return stopped, completed

    def save_events(self, relative):
        self.evidence.jsonl(relative + "/events.jsonl", self.rec.journal())
        self.evidence.jsonl(relative + "/client-transmit.jsonl", self.tx)


def echo_case(client, case, user, evidence):
    response = client.prime(user)
    reference = resample_reference(response["pcm"], 24000, 1)
    mic, dsp = make_echo(reference, case)
    directory = f"echo/{case.name}"
    evidence.bytes(directory + "/far-end-24k.pcm", response["pcm"])
    evidence.bytes(directory + "/microphone-16k.pcm", mic)
    evidence.json(directory + "/response.json",
                  {k: v for k, v in response.items() if k != "pcm"})
    evidence.json(directory + "/dsp.json", dsp)

    protected_marker = client.rec.mark()
    client.stream(mic, "protected-echo")
    time.sleep(0.3)
    protected = client.rec.since(protected_marker, types=CUTIN_TYPES)
    require(not protected, f"protected echo-only cut in: {[r.event for r in protected]}")

    # Same real response, same reference tail, and exactly the same mic bytes.
    client.set_echo_suppression(False)
    baseline_marker = client.rec.mark()
    baseline_started = client.stream(mic, "baseline-echo-disabled")
    cancelled, _, latency = client.check_cancel(response, baseline_marker, baseline_started)
    stopped, completed = client.finish_endpoint(baseline_marker, response, cancelled)
    endpoint_ms = (stopped.time_s -
                   (baseline_started + dsp["last_voiced_ms"] / 1000.0)) * 1000.0
    require(250.0 <= endpoint_ms <= 1500.0,
            f"unexpected endpoint latency: {endpoint_ms:.1f} ms")
    require(completed.event.get("segment_id") == stopped.event.get("segment_id"),
            "endpoint and final transcription segment mismatch")
    return {
        "case": case.name, "protected_echo_only_cutins": 0,
        "baseline_echo_only_cutins": 1,
        "same_microphone_sha256": sha256(mic),
        "same_real_response_id": response["response_id"],
        "baseline_cancel_ms": round(latency, 3),
        "baseline_endpoint_from_last_voiced_ms": round(endpoint_ms, 3),
        "baseline_transcript": completed.event.get("text"),
        "baseline_cancel_segment": cancelled.event.get("segment_id"),
        "protected_transcription_events": 0,
        "acknowledged_prefix_bytes": response["text_cursor"],
        "dsp": dsp,
    }


def near_end_case(ws_port, arm, user, evidence):
    client = Client(ws_port, f"near-end-{arm}", evidence)
    try:
        if arm == "baseline":
            client.set_echo_suppression(False)
        response = client.prime(user)
        mic = user[:64000] + END_SILENCE
        evidence.bytes(f"near-end/{arm}/microphone-16k.pcm", mic)
        marker = client.rec.mark()
        started = client.stream(mic, f"near-end-{arm}")
        cancelled, speech, latency = client.check_cancel(response, marker, started)
        start_ms = (speech.time_s - started) * 1000.0
        require(start_ms <= 250.0,
                f"near-end onset missed the first 100 ms append: {start_ms:.1f} ms")
        stopped, completed = client.finish_endpoint(marker, response, cancelled)
        require(completed.event.get("text", "").strip(), "near-end ASR final was empty")
        client.save_events(f"near-end/{arm}")
        return {"arm": arm, "speech_started_ms": round(start_ms, 3),
                "cancel_ms": round(latency, 3), "microphone_sha256": sha256(mic),
                "endpoint_segment": stopped.event.get("segment_id"),
                "transcript": completed.event.get("text")}
    finally:
        client.close()


def repeated_double_talk_case(client, user, evidence):
    results = []
    stale = None
    response = client.prime(user)
    for cycle in range(2):
        reference = resample_reference(response["pcm"], 24000, 1)
        echo, dsp = make_echo(reference, ECHO_CASES[1])
        mic = make_double_talk(echo, user)
        directory = f"repeated-double-talk/response-{cycle}"
        evidence.bytes(directory + "/far-end-24k.pcm", response["pcm"])
        evidence.bytes(directory + "/microphone-16k.pcm", mic)
        marker = client.rec.mark()
        started = client.stream(mic, f"double-talk-{cycle}")
        cancelled, speech, latency = client.check_cancel(response, marker, started)
        require(speech.time_s - started <= 0.25,
                "simultaneous double-talk did not start on the first append")
        stopped, completed = client.finish_endpoint(marker, response, cancelled)
        results.append({"cycle": cycle, "response_id": response["response_id"],
                        "double_talk_cancel_ms": round(latency, 3),
                        "segment_id": stopped.event.get("segment_id"),
                        "transcript": completed.event.get("text"),
                        "microphone_sha256": sha256(mic),
                        "reference_response_sha256": response["audio_sha256"]})
        if cycle == 0:
            # Start the next response, but inject response 0's echo after its
            # first text delta and before any new audio.delta. A stale ring would
            # suppress this input; the correct reset exposes it to energy VAD.
            next_first = client.rec.wait_for(
                lambda e: e.get("type") in ("response.text.delta", "audio.delta") and
                e.get("response_id"), completed.index + 1, 120,
                "next response before stale-reference probe")
            next_id = next_first.event["response_id"]
            require(next_first.event["type"] == "response.text.delta" and not
                    client.rec.since(next_first.index + 1, types={"audio.delta"},
                                     response_id=next_id),
                    "next response emitted audio before the stale-reference probe")
            evidence.bytes("repeated-double-talk/stale-response/microphone-16k.pcm", echo)
            stale_response = {"response_id": next_id, "text_cursor": 0,
                              "acknowledged_text": ""}
            stale_marker = client.rec.mark()
            stale_started = client.stream(echo, "stale-response-echo")
            stale_cancel, _, stale_latency = client.check_cancel(
                stale_response, stale_marker, stale_started)
            require(not client.rec.since(next_first.index + 1, types={"audio.delta"},
                                         response_id=next_id),
                    "stale-reference probe was contaminated by new far-end audio")
            _, stale_completed = client.finish_endpoint(
                stale_marker, stale_response, stale_cancel)
            stale = {"old_response_id": response["response_id"],
                     "new_response_id": next_id,
                     "new_response_audio_before_probe": 0,
                     "stale_echo_cancel_ms": round(stale_latency, 3),
                     "microphone_sha256": sha256(echo),
                     "endpoint_transcript": stale_completed.event.get("text")}
            response = client.collect_response(stale_completed.index + 1)
    require(results[0]["response_id"] != results[1]["response_id"],
            "repeated barge-in did not create a new response")
    return {"double_talk": results, "stale_response": stale}


def two_session_case(ws_port, user, evidence):
    first = Client(ws_port, "session-a", evidence)
    second = None
    try:
        response = first.prime(user)
        reference = resample_reference(response["pcm"], 24000, 1)
        mic, dsp = make_echo(reference, ECHO_CASES[1])
        evidence.bytes("two-sessions/far-end-24k.pcm", response["pcm"])
        evidence.bytes("two-sessions/microphone-16k.pcm", mic)
        second = Client(ws_port, "session-b", evidence)

        marker_a = first.rec.mark()
        first.stream(mic, "session-a-own-echo")
        require(not first.rec.since(marker_a, types=CUTIN_TYPES),
                "session A did not suppress its own real echo")

        # B has no far-end history. A's echo must remain visible as near-end
        # energy there; no reference or tracking state may cross sessions.
        marker_b = second.rec.mark()
        started_b = second.stream(mic, "session-b-foreign-echo")
        speech_b = second.rec.wait_for(
            lambda e: e.get("type") == "input_audio_buffer.speech_started", marker_b,
            1.0, "cross-session energy fallback")
        stopped_b, completed_b = second.finish_endpoint(marker_b)
        require(not first.rec.since(marker_a, types={"response.cancelled"}),
                "session B input cancelled session A")
        first.save_events("two-sessions/session-a")
        second.save_events("two-sessions/session-b")
        return {"session_a_own_echo_cutins": 0,
                "session_b_foreign_echo_starts": 1,
                "session_b_start_ms": round((speech_b.time_s - started_b) * 1000, 3),
                "session_b_transcript": completed_b.event.get("text"),
                "session_b_endpoint_segment": stopped_b.event.get("segment_id"),
                "same_microphone_sha256": sha256(mic), "dsp": dsp}
    finally:
        if second is not None:
            second.close()
        first.close()


def model_provenance(args):
    output = {}
    for role, model in (("asr", args.asr_model), ("vlm", args.vlm_model),
                        ("tts", args.tts_model)):
        config = model / "config.json"
        entry = {"revision": "UNKNOWN", "config_sha256": hash_file(config),
                 "config_bytes": config.stat().st_size}
        index = model / "model.safetensors.index.json"
        single = model / "model.safetensors"
        if index.exists():
            entry["weight_index_sha256"] = hash_file(index)
            weight_map = json.loads(index.read_text())["weight_map"]
            entry["weight_shards"] = len(set(weight_map.values()))
            entry["weight_tensors"] = len(weight_map)
        elif single.exists():
            entry["weight_file_bytes"] = single.stat().st_size
        output[role] = entry
    return output


def self_check():
    sample_rate = 24000
    source = [int(9000 * math.sin(2 * math.pi * 397 * i / sample_rate) +
                  5000 * math.sin(2 * math.pi * 1153 * i / sample_rate))
              for i in range(sample_rate * 2)]
    reference = resample_reference(pcm16_bytes(source), sample_rate, 1)
    require(abs(len(reference) - 32000) <= 1, "24 kHz resampling length drift")
    hashes = set()
    for case in ECHO_CASES:
        first, metadata = make_echo(reference, case)
        second, _ = make_echo(reference, case)
        require(first == second and sha256(first) not in hashes,
                f"non-deterministic or duplicate echo fixture: {case.name}")
        hashes.add(sha256(first))
        require(len(first) == metadata["active_bytes"] + len(END_SILENCE),
                "echo endpoint fixture length mismatch")
    double = make_double_talk(make_echo(reference, ECHO_CASES[1])[0],
                              pcm16_bytes(source[:64000]))
    require(sha256(double) not in hashes, "double-talk collapsed into echo-only")
    print(f"PASS full-duplex echo self-check cases={len(ECHO_CASES)} "
          "resample=24k_to_16k deterministic=1 clipping=1 double_talk=1")
    return 0


def run(args):
    require(args.gpu_asr == 0 and args.gpu_vlm == 1 and args.gpu_tts == 7,
            "the agreed assignment is ASR=0, Qwen3.8=1, TTS=7")
    args.out.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    evidence = Evidence(args.out)
    user = trim_digital_lead_in(pcm_from_wav(args.audio))
    require(len(user) >= 64000, "near-end fixture needs at least 2 s real speech")
    evidence.bytes("corpus/near-end-user-16k.wav", args.audio.read_bytes())
    evidence.bytes("corpus/near-end-active-16k.pcm", user[:64000])
    result = {
        "schema": "umminfer-full-duplex-echo-live/v1", "status": "FAIL",
        "source_commit": args.source_commit,
        "validation_class": "REAL_MODEL_SERVER_PCM",
        "physical_speaker_microphone": "NOT_RUN", "listening_quality": "NOT_RUN",
        "gpus": {"asr": 0, "qwen38": 1, "tts": 7},
        "fixed_echo_corpus": [case.name for case in ECHO_CASES],
        "near_end_audio_sha256": sha256(user),
        "production_echo_suppression_default": True,
        "ab_method": "same active real TTS response and byte-identical microphone PCM",
        "resource_bounds": {"server_cpu_percent_p95_max": 200.0,
                            "server_rss_max_bytes": 512 * 1024 * 1024,
                            "server_rss_growth_max_bytes": 128 * 1024 * 1024},
        "model_provenance": model_provenance(args),
        "binaries": {name: hash_file(args.bin_dir / name) for name in (
            "umm-infer", "umm-native-stage-asr", "umm-native-stage-qwen38",
            "umm-native-stage-tts")},
        "harness_sha256": hash_file(Path(__file__)),
    }
    ports = free_ports(2)
    require(43301 not in ports, "ephemeral allocation unexpectedly selected resident port")
    result["ports"] = {"http": ports[0], "websocket": ports[1]}
    procs = {}
    isolated = ()
    sampler = None
    try:
        result["gpu_preflight"] = preflight_gpus([0, 1, 7])
        procs, isolated = start_stack(args, ports)
        sampler = ResourceSampler(procs, args.out / "timing/resources.jsonl")
        result["scenarios"] = {"echo": []}
        for case in ECHO_CASES:
            client = Client(ports[1], case.name, evidence)
            try:
                row = echo_case(client, case, user, evidence)
                client.save_events(f"echo/{case.name}")
                result["scenarios"]["echo"].append(row)
                print(f"PASS echo {case.name} protected=0 baseline=1 "
                      f"cancel={row['baseline_cancel_ms']} ms", flush=True)
            finally:
                client.close()
            wait_in_flight_zero(ports[0])

        protected = near_end_case(ports[1], "protected", user, evidence)
        baseline = near_end_case(ports[1], "baseline", user, evidence)
        require(protected["microphone_sha256"] == baseline["microphone_sha256"],
                "near-end A/B did not use identical input")
        require(protected["speech_started_ms"] <= baseline["speech_started_ms"] + 100.0,
                "echo wrapper regressed near-end onset by more than 100 ms")
        result["scenarios"]["near_end"] = {"protected": protected, "baseline": baseline,
                                            "onset_regression": False}
        print(f"PASS near-end protected={protected['speech_started_ms']} ms "
              f"baseline={baseline['speech_started_ms']} ms", flush=True)

        client = Client(ports[1], "repeated-double-talk", evidence)
        try:
            result["scenarios"]["repeated_double_talk"] = repeated_double_talk_case(
                client, user, evidence)
            client.save_events("repeated-double-talk")
        finally:
            client.close()
        wait_in_flight_zero(ports[0])
        print("PASS simultaneous double-talk repeated=2/2 within_L1=1", flush=True)

        result["scenarios"]["two_sessions"] = two_session_case(ports[1], user, evidence)
        wait_in_flight_zero(ports[0])
        print("PASS two-session isolation own_echo=0 foreign_echo=1", flush=True)

        result["resource_usage"] = sampler.finish()
        sampler = None
        server = result["resource_usage"]["server"]
        require(server["cpu_percent_p95"] <= 200.0 and
                server["rss_max_bytes"] <= 512 * 1024 * 1024 and
                server["rss_growth_bytes"] <= 128 * 1024 * 1024,
                f"server CPU/RSS exceeded bounds: {server}")
        result["final_health"] = wait_in_flight_zero(ports[0])
        result["status"] = "PASS"
    except Exception as error:
        result["error"] = str(error)
        for name, proc in procs.items():
            result.setdefault("log_tails", {})[name] = proc.log.text()[-4000:]
        raise
    finally:
        if sampler is not None:
            result["resource_usage"] = sampler.finish()
        if procs:
            stop_stack(procs)
        for name, proc in procs.items():
            evidence.bytes(f"logs/{name}.log", proc.log.path.read_bytes())
        for path in isolated:
            shutil.rmtree(path, ignore_errors=True)
        try:
            result["gpu_postflight"] = wait_for_idle_gpus([0, 1, 7])
            result["selected_gpus_idle"] = True
        except Exception as cleanup_error:
            result["selected_gpus_idle"] = False
            result["cleanup_error"] = str(cleanup_error)
            result["status"] = "FAIL"
        evidence.json("full-duplex-echo-evidence.json", result)
        result["evidence_files_hashed"] = evidence.manifest()
        # Rewrite once so the summary includes the manifest count, then refresh
        # the manifest to bind the final summary bytes as well.
        evidence.json("full-duplex-echo-evidence.json", result)
        evidence.manifest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--bin-dir", type=Path)
    parser.add_argument("--asr-model", type=Path)
    parser.add_argument("--vlm-model", type=Path)
    parser.add_argument("--tts-model", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--gpu-asr", type=int, default=0)
    parser.add_argument("--gpu-vlm", type=int, default=1)
    parser.add_argument("--gpu-tts", type=int, default=7)
    parser.add_argument("--vlm-max-tokens", type=int, default=96)
    parser.add_argument("--tts-max-frames", type=int, default=192)
    args = parser.parse_args()
    if args.self_check:
        return self_check()
    for name in ("bin_dir", "asr_model", "vlm_model", "tts_model", "audio",
                 "out", "log_dir", "source_commit"):
        require(getattr(args, name) is not None, f"--{name.replace('_', '-')} is required")
    result = run(args)
    require(result.get("status") == "PASS" and result.get("selected_gpus_idle") is True,
            f"real gate or GPU cleanup failed: {result.get('error') or result.get('cleanup_error')}")
    print("PASS real-model echo/double-talk gate echo_only_cutins=0 "
          "near_end_regression=0 double_talk_L1=PASS in_flight=0", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
