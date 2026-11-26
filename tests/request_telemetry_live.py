#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Focused live gate for opt-in request-native CUDA allocator telemetry.

Run inside the GPU container against an already running, isolated loopback
ummInfer stack. This client never starts, stops, signals, or reconfigures a
process. The stack owner must select the physical GPU and native batch/backend
profile. Every request is opt-in through UMMINFER_LIVE_OPT_IN=1.

Suites:
  serial        Qwen3.8 and Wan2.2 serial covered paths, protocol parity,
                prefix/VAE-cache repeats, failure/cancel, and uninstrumented ASR
  tts           real Qwen3.8 + CustomVoice TTS serial requests, two prompts,
                warm repeats, off/on JSON/SSE/PCM parity, failure/cancel, and
                independent 50 ms whole-device NVML (isolated stack required)
  batched       Qwen3.8 --batch-size>1 must be NOT_COVERED with null peaks
  contract-stub Wan2.2 contract-stub must be NOT_COVERED with null peaks

Evidence includes raw responses, normalized protocol-parity hashes, independent
50 ms whole-device nvidia-smi samples, summaries, and SHA256SUMS. Request-native
peaks are never inferred from those whole-device samples.
"""

import argparse
import base64
import copy
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import wave


class GateFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise GateFailure(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def normalize(value, request_id=None):
    """Remove transport identity/time, retaining model and artifact bytes."""
    if isinstance(value, dict):
        return {key: normalize(item, request_id) for key, item in value.items()
                if key not in ("created", "session_id", "telemetry",
                               "denoise_ms", "encode_ms", "first_frame_ms",
                               "total_ms", "first_audio_ms", "elapsed_ms")}
    if isinstance(value, list):
        return [normalize(item, request_id) for item in value]
    if isinstance(value, str) and request_id:
        return value.replace(request_id, "<request>")
    return value


def response_telemetry(body):
    if not isinstance(body, dict):
        return None
    if isinstance(body.get("telemetry"), dict):
        return body["telemetry"]
    error = body.get("error")
    if isinstance(error, dict) and isinstance(error.get("telemetry"), dict):
        return error["telemetry"]
    return None


def frame_telemetry(frame):
    direct = response_telemetry(frame)
    if direct is not None:
        return direct
    unified = frame.get("unified") if isinstance(frame, dict) else None
    return response_telemetry(unified) if isinstance(unified, dict) else None


def only_stage(envelope, native_stage):
    require(isinstance(envelope, dict), f"missing telemetry envelope for {native_stage}")
    stages = [row for row in envelope.get("stages", [])
              if row.get("native_stage") == native_stage]
    require(len(stages) == 1,
            f"expected one {native_stage} stage, got {len(stages)}")
    return stages[0]


def require_covered(envelope, native_stage, device, lifecycle="completed",
                    nonzero=False):
    stage = only_stage(envelope, native_stage)
    require(stage.get("coverage") == "COVERED", f"{native_stage} is not COVERED")
    require(stage.get("device_index") == device, f"{native_stage} wrong device")
    require(stage.get("lifecycle_status") == lifecycle,
            f"{native_stage} lifecycle is not {lifecycle}")
    require(stage.get("execution") == {"mode": "serial", "batch_capacity": 1},
            f"{native_stage} is not serial batch-capacity 1")
    epoch = stage.get("epoch", {})
    require(isinstance(epoch.get("id"), int) and epoch["id"] > 0,
            f"{native_stage} has no positive epoch")
    require(epoch.get("reset") is True and
            epoch.get("semantics") == "new_allocation_ownership_epoch",
            f"{native_stage} epoch semantics are wrong")
    allocated = stage.get("allocated_bytes_peak")
    reserved = stage.get("reserved_bytes_peak")
    require(isinstance(allocated, int) and isinstance(reserved, int),
            f"{native_stage} peaks are null")
    require(allocated == reserved, f"{native_stage} direct peaks differ")
    if nonzero:
        require(allocated > 0, f"{native_stage} peak is zero")
    samples = stage.get("sample_coverage", {})
    require(samples.get("terminal_snapshots") == 1,
            f"{native_stage} terminal snapshot count is not one")
    if samples.get("allocation_events") == 0:
        require(allocated == 0 and stage.get("allocated_bytes_final") == 0,
                f"{native_stage} has bytes without an allocation event")
    return stage


def require_not_covered(stage, reason):
    require(stage.get("coverage") == "NOT_COVERED", "path unexpectedly COVERED")
    require(stage.get("reason") == reason,
            f"NOT_COVERED reason {stage.get('reason')!r} != {reason!r}")
    for key in ("allocated_bytes_peak", "reserved_bytes_peak",
                "allocated_bytes_final", "reserved_bytes_final"):
        require(stage.get(key) is None, f"{key} must be null, not zero")
    require(stage.get("sample_coverage") == {
        "allocation_events": 0, "free_events": 0, "terminal_snapshots": 0},
        "NOT_COVERED path carries fabricated samples")


class NvmlMonitor:
    def __init__(self, device, interval_s=0.05):
        self.device = device
        self.interval_s = interval_s
        self.rows = []
        self.error = None
        self.stop_event = threading.Event()
        self.thread = None

    def sample(self):
        output = subprocess.run([
            "nvidia-smi", "-i", str(self.device),
            "--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits"], check=True, capture_output=True,
            text=True, timeout=10).stdout.strip()
        fields = [field.strip() for field in output.split(",")]
        require(len(fields) == 5, "unexpected nvidia-smi sample shape")
        self.rows.append({
            "timestamp": fields[0], "device_index": int(fields[1]),
            "memory_used_mib": int(fields[2]), "memory_total_mib": int(fields[3]),
            "gpu_utilization_percent": int(fields[4]),
            "scope": "whole-device-used-memory",
        })

    def run(self):
        try:
            while not self.stop_event.wait(self.interval_s):
                self.sample()
        except Exception as exc:  # surfaced at stop; never silently drop samples
            self.error = str(exc)
            self.stop_event.set()

    def start(self):
        self.sample()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=15)
        require(self.error is None, f"NVML monitor failed: {self.error}")
        self.sample()
        require(len(self.rows) >= 2, "insufficient independent NVML samples")


class Gate:
    def __init__(self, args):
        self.args = args
        self.base = args.base_url.rstrip("/")
        self.evidence = args.evidence_dir
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.failures = []
        self.epochs = {}
        self.peaks = {}
        self.terminal_files = []
        self.media_hashes = {}
        self.monitor = NvmlMonitor(args.device)

    def save_json(self, name, value):
        path = self.evidence / name
        path.write_bytes(json.dumps(value, indent=2, ensure_ascii=False,
                                    sort_keys=True).encode() + b"\n")
        return path

    def record(self, name, status, detail=None):
        row = {"case": name, "status": status, "detail": detail}
        self.results.append(row)
        print(f"{status} {name}" + (f": {detail}" if detail else ""), flush=True)

    def case(self, name, function):
        try:
            detail = function()
            self.record(name, "PASS", detail)
        except Exception as exc:
            self.failures.append({"case": name, "error": str(exc)})
            self.record(name, "FAIL", str(exc))

    def get_json(self, path):
        with urllib.request.urlopen(self.base + path, timeout=15) as response:
            return response.status, json.loads(response.read())

    def chat(self, name, body):
        raw_request = canonical(body)
        (self.evidence / f"{name}.request.json").write_bytes(raw_request + b"\n")
        request = urllib.request.Request(
            self.base + "/v1/chat/completions", data=raw_request,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.args.timeout_s) as response:
                status, raw = response.status, response.read()
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, exc.read()
        path = self.evidence / f"{name}.response.json"
        path.write_bytes(raw)
        parsed = json.loads(raw)
        return status, parsed, path

    def stream(self, name, body, abort_first_frame=False,
               abort_first_audio=False):
        raw_request = canonical(body)
        (self.evidence / f"{name}.request.json").write_bytes(raw_request + b"\n")
        parts = urllib.parse.urlsplit(self.base)
        connection = http.client.HTTPConnection(parts.hostname, parts.port,
                                                timeout=self.args.timeout_s)
        connection.request("POST", "/v1/chat/completions", body=raw_request,
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        frames, pending, raw_lines = [], [], bytearray()
        abort_result = None
        try:
            for raw in response:
                raw_lines.extend(raw)
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if line.startswith("data:"):
                    pending.append(line[5:].lstrip())
                    continue
                if line or not pending:
                    continue
                data = "\n".join(pending)
                pending = []
                if data == "[DONE]":
                    break
                frame = json.loads(data)
                frames.append(frame)
                if abort_first_audio and abort_result is None:
                    unified = frame.get("unified", {})
                    artifact = unified.get("artifact", {})
                    if (unified.get("kind") == "artifact" and
                            artifact.get("mime_type") == "audio/pcm"):
                        request_id = frame.get("id") or unified.get("request_id")
                        require(isinstance(request_id, str) and request_id,
                                "audio SSE frame has no request id")
                        abort_result = self.abort(request_id)
                if abort_first_frame and abort_result is None:
                    request_id = frame.get("id")
                    require(isinstance(request_id, str) and request_id,
                            "first SSE frame has no request id")
                    abort_result = self.abort(request_id)
        finally:
            connection.close()
        (self.evidence / f"{name}.response.sse").write_bytes(bytes(raw_lines))
        self.save_json(f"{name}.frames.json", frames)
        return response.status, frames, abort_result

    def abort(self, request_id):
        request = urllib.request.Request(
            self.base + "/abort", data=canonical({"request_id": request_id}),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return {"status": response.status, "body": response.read().decode()}
        except urllib.error.HTTPError as exc:
            return {"status": exc.code, "body": exc.read().decode()}

    def note_epochs(self, envelope):
        for stage in envelope.get("stages", []):
            if stage.get("coverage") != "COVERED":
                continue
            backend = stage["native_stage"]
            epoch = stage["epoch"]["id"]
            previous = self.epochs.get(backend, [])
            require(epoch not in previous, f"{backend} reused epoch {epoch}")
            if previous:
                require(epoch > previous[-1], f"{backend} epoch is not monotonic")
            previous.append(epoch)
            self.epochs[backend] = previous
            self.peaks.setdefault(backend, []).append(
                stage["allocated_bytes_peak"])

    def remember_terminal(self, path):
        self.terminal_files.append((path, sha256_bytes(path.read_bytes())))

    def text_body(self, telemetry=False, stream=False, max_tokens=16):
        body = {
            "model": self.args.model, "stream": stream, "temperature": 0,
            "top_p": 1, "seed": 42, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content":
                          "Answer with one short deterministic sentence."}],
        }
        if telemetry:
            body["telemetry"] = {"native_allocator": True}
        return body

    def video_body(self, telemetry=False):
        body = self.text_body(telemetry=telemetry, max_tokens=8)
        body["modalities"] = ["text", "video"]
        body["video"] = {
            "prompt": "A calm blue circle over a still gray background",
            "size": "352x256", "num_frames": 9, "fps": 16, "steps": 2,
            "guidance_scale": 5.0, "seed": 42, "output_format": "mp4",
        }
        return body

    def tts_body(self, telemetry=False, stream=False, prompt_index=0,
                 max_tokens=32):
        prompts = (
            "Reply with exactly this sentence and nothing else: "
            "Allocator telemetry remains optional for every voice request.",
            "Reply with exactly this sentence and nothing else: "
            "A different prompt exercises the same warmed native speech pools.",
        )
        body = self.text_body(telemetry=telemetry, stream=stream,
                              max_tokens=max_tokens)
        body["messages"] = [{"role": "user", "content": prompts[prompt_index]}]
        body["modalities"] = ["text", "audio"]
        body["audio"] = {
            "voice": "vivian", "format": "wav", "stream": stream,
            "sample_rate": 24000,
        }
        return body

    def save_wav_and_pcm(self, name, body):
        artifacts = [item for item in body.get("artifacts", [])
                     if item.get("media_type") == "audio"]
        require(len(artifacts) == 1, f"{name} did not return one audio artifact")
        artifact = artifacts[0]
        require("request_telemetry" not in artifact,
                "internal TTS telemetry leaked into an artifact")
        wav = base64.b64decode(artifact["data"], validate=True)
        require(sha256_bytes(wav) == artifact.get("sha256"),
                f"{name} WAV SHA-256 mismatch")
        with wave.open(io.BytesIO(wav), "rb") as source:
            require(source.getnchannels() == 1 and source.getsampwidth() == 2 and
                    source.getframerate() == 24000,
                    f"{name} is not 24 kHz mono PCM16")
            pcm = source.readframes(source.getnframes())
        require(len(pcm) > 0 and len(pcm) % 2 == 0, f"{name} has invalid PCM")
        (self.evidence / f"{name}.wav").write_bytes(wav)
        (self.evidence / f"{name}.pcm").write_bytes(pcm)
        digest = sha256_bytes(pcm)
        self.media_hashes[name] = {"pcm_sha256": digest, "pcm_bytes": len(pcm),
                                   "wav_sha256": sha256_bytes(wav)}
        return digest

    def save_stream_pcm(self, name, frames):
        chunks = {}
        expected = {}
        combined = bytearray()
        for frame in frames:
            unified = frame.get("unified", {})
            if unified.get("kind") != "artifact":
                continue
            artifact = unified.get("artifact", {})
            if artifact.get("mime_type") != "audio/pcm":
                continue
            require(artifact.get("format") == "pcm16" and
                    artifact.get("sample_rate") == 24000 and
                    artifact.get("channels") == 1,
                    f"{name} streamed non-PCM16/24k/mono audio")
            identity = artifact["artifact_id"]
            data = base64.b64decode(artifact["data"], validate=True)
            chunks.setdefault(identity, bytearray()).extend(data)
            combined.extend(data)
            if "sha256" in artifact:
                expected[identity] = (artifact["sha256"], artifact["bytes"])
        require(chunks and expected.keys() == chunks.keys(),
                f"{name} did not stream complete PCM artifacts")
        for identity, (digest, size) in expected.items():
            require(size == len(chunks[identity]) and
                    digest == sha256_bytes(chunks[identity]),
                    f"{name} streamed PCM digest/size mismatch for {identity}")
        pcm = bytes(combined)
        require(len(pcm) > 0 and len(pcm) % 2 == 0, f"{name} has invalid PCM")
        (self.evidence / f"{name}.pcm").write_bytes(pcm)
        self.media_hashes[name] = {
            "pcm_sha256": sha256_bytes(pcm), "pcm_bytes": len(pcm),
            "artifact_count": len(chunks),
            "artifact_sha256": {identity: sha256_bytes(data)
                                for identity, data in chunks.items()},
        }
        return sha256_bytes(pcm)

    def require_tts_warm_covered(self, envelope, lifecycle="completed"):
        rows = [row for row in envelope.get("stages", [])
                if row.get("native_stage") == "tts"]
        require(rows, "missing native TTS telemetry stage")
        covered = []
        for row in rows:
            single = dict(envelope)
            single["stages"] = [row]
            stage = require_covered(single, "tts", self.args.device,
                                    lifecycle=lifecycle)
            samples = stage["sample_coverage"]
            require(stage["allocated_bytes_peak"] == 0 and
                    stage["reserved_bytes_peak"] == 0 and
                    stage["allocated_bytes_final"] == 0 and
                    stage["reserved_bytes_final"] == 0 and
                    samples["allocation_events"] == 0 and
                    samples["free_events"] == 0,
                    "TTS pre-request pools were incorrectly attributed to a request")
            covered.append(stage)
        return covered

    def tts(self):
        status, health = self.get_json("/health")
        require(status == 200 and health.get("in_flight") == 0,
                f"initial health is not idle: {health}")
        status, ready = self.get_json("/ready")
        require(status == 200 and ready.get("status") == "ready",
                "Qwen/TTS stack is not ready")
        log_offset = 0
        if self.args.tts_log:
            log_offset = Path(self.args.tts_log).stat().st_size

        nonstream = {}
        for name, enabled, prompt_index in (
                ("tts-cold-on", True, 0), ("tts-off", False, 0),
                ("tts-warm-on", True, 0), ("tts-second-prompt-on", True, 1)):
            body = self.tts_body(telemetry=enabled, prompt_index=prompt_index)
            status, response, path = self.chat(name, body)
            require(status == 200, f"{name} HTTP {status}: {response}")
            envelope = response_telemetry(response)
            if enabled:
                require_covered(envelope, "qwen38", self.args.device)
                self.require_tts_warm_covered(envelope)
                self.note_epochs(envelope)
                self.remember_terminal(path)
            else:
                require(envelope is None and "telemetry" not in response,
                        "default TTS JSON emitted telemetry")
            self.save_wav_and_pcm(name, response)
            nonstream[name] = response
        parity = self.compare(nonstream["tts-off"], nonstream["tts-cold-on"],
                              nonstream["tts-off"].get("id"),
                              nonstream["tts-cold-on"].get("id"))
        self.record("tts-json-off-on-protocol-parity", "PASS", parity)
        require(self.media_hashes["tts-off"]["pcm_sha256"] ==
                self.media_hashes["tts-cold-on"]["pcm_sha256"] ==
                self.media_hashes["tts-warm-on"]["pcm_sha256"],
                "TTS off/on/warm PCM bytes differ")
        self.record("tts-off-on-warm-pcm-parity", "PASS",
                    self.media_hashes["tts-off"]["pcm_sha256"])

        streamed = {}
        for name, enabled in (("tts-sse-off", False), ("tts-sse-on", True)):
            status, frames, _ = self.stream(
                name, self.tts_body(telemetry=enabled, stream=True))
            require(status == 200 and frames, f"{name} SSE failed")
            envelopes = [frame_telemetry(frame) for frame in frames
                         if frame_telemetry(frame) is not None]
            if enabled:
                require(len(envelopes) == 1,
                        "TTS SSE did not emit one terminal envelope")
                require_covered(envelopes[0], "qwen38", self.args.device)
                self.require_tts_warm_covered(envelopes[0])
                self.note_epochs(envelopes[0])
            else:
                require(not envelopes and
                        all("telemetry" not in frame for frame in frames),
                        "default TTS SSE emitted telemetry")
            self.save_stream_pcm(name, frames)
            streamed[name] = frames
        first_id = streamed["tts-sse-off"][0].get("id")
        second_id = streamed["tts-sse-on"][0].get("id")
        parity = self.compare(streamed["tts-sse-off"],
                              streamed["tts-sse-on"], first_id, second_id)
        self.record("tts-sse-off-on-protocol-parity", "PASS", parity)
        require(self.media_hashes["tts-sse-off"]["pcm_sha256"] ==
                self.media_hashes["tts-sse-on"]["pcm_sha256"],
                "TTS off/on streamed PCM bytes differ")
        self.record("tts-sse-off-on-pcm-parity", "PASS",
                    self.media_hashes["tts-sse-off"]["pcm_sha256"])

        # A valid reference WAV is a deterministic native-stage failure for a
        # CustomVoice checkpoint, after public validation and VLM completion.
        reference = io.BytesIO()
        with wave.open(reference, "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(24000)
            target.writeframes(b"\x01\x00" * 240)
        failed_body = self.tts_body(telemetry=True)
        failed_body["audio"]["ref_audio"] = base64.b64encode(
            reference.getvalue()).decode()
        status, failed, path = self.chat("tts-failed", failed_body)
        require(status == 200, f"TTS optional-leg failure became HTTP {status}")
        require(any(leg.get("stage") == "tts" and leg.get("state") == "failed"
                    for leg in failed.get("legs", [])),
                f"TTS failure was not reported in legs: {failed}")
        envelope = response_telemetry(failed)
        self.require_tts_warm_covered(envelope, lifecycle="failed")
        self.note_epochs(envelope)
        self.remember_terminal(path)

        # Cancel only after native streaming PCM proves the TTS leg is running.
        cancel_body = self.tts_body(telemetry=True, stream=True, max_tokens=160)
        cancel_body["messages"] = [{"role": "user", "content":
            "Write one continuous sixty-word sentence about a calm mountain "
            "lake at sunrise, without lists or headings."}]
        status, frames, abort = self.stream(
            "tts-cancelled", cancel_body, abort_first_audio=True)
        require(status == 200 and abort and abort.get("status") == 200,
                f"TTS in-flight abort failed: {abort}")
        envelopes = [frame_telemetry(frame) for frame in frames
                     if frame_telemetry(frame) is not None]
        require(len(envelopes) == 1 and
                envelopes[0].get("lifecycle_status") == "cancelled",
                "TTS cancellation did not emit one cancelled envelope")
        tts_rows = [row for row in envelopes[0].get("stages", [])
                    if row.get("stage") == "tts"]
        if tts_rows:
            saw_native_tts = False
            for row in tts_rows:
                if row.get("native_stage") == "tts":
                    single = dict(envelopes[0])
                    single["stages"] = [row]
                    self.require_tts_warm_covered(
                        single, lifecycle=row.get("lifecycle_status"))
                    saw_native_tts = True
                else:
                    require_not_covered(row, "stage_terminal_snapshot_missing")
            if saw_native_tts:
                self.note_epochs(envelopes[0])
        self.save_json("tts-cancelled-boundary.json", {
            "abort": abort, "telemetry": envelopes[0],
            "boundary": "public cancellation after first native audio/pcm artifact",
        })

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            _, health = self.get_json("/health")
            if health.get("in_flight") == 0:
                break
            time.sleep(0.1)
        require(health.get("in_flight") == 0, f"in_flight did not drain: {health}")
        if self.args.tts_log:
            # Public cancellation is intentionally immediate; the native abort
            # marker can be appended just after the terminal SSE envelope.
            native_log = b""
            marker = "request_id=" + envelopes[0]["request_id"] + ":tts"
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                native_log = Path(self.args.tts_log).read_bytes()[log_offset:]
                if (b"tts request aborted" in native_log and
                        b"aborted at frame" in native_log and
                        marker.encode() in native_log):
                    break
                time.sleep(0.05)
            (self.evidence / "tts-native-log-tail.txt").write_bytes(native_log)
            require(b"tts request aborted" in native_log and
                    b"aborted at frame" in native_log and
                    marker.encode() in native_log,
                    "native TTS log did not retain this request's cancellation")
        for path, digest in self.terminal_files:
            require(sha256_bytes(path.read_bytes()) == digest,
                    f"terminal response was rewritten: {path.name}")
        require(len(self.epochs.get("tts", [])) >= 5,
                "insufficient unique TTS request epochs")

    def compare(self, off, on, off_id=None, on_id=None):
        left = canonical(normalize(off, off_id))
        right = canonical(normalize(on, on_id))
        require(left == right, "telemetry off/on normalized protocol bytes differ")
        return sha256_bytes(left)

    def serial(self):
        status, health = self.get_json("/health")
        require(status == 200 and health.get("in_flight") == 0,
                f"initial health is not idle: {health}")
        status, ready = self.get_json("/ready")
        require(status == 200 and ready.get("status") == "ready", "stack not ready")

        qwen = {}
        for name, enabled in (("qwen-on", True), ("qwen-off", False),
                              ("qwen-on-second", True)):
            status, body, path = self.chat(name, self.text_body(telemetry=enabled))
            require(status == 200, f"{name} HTTP {status}: {body}")
            envelope = response_telemetry(body)
            if not enabled:
                require(envelope is None and "telemetry" not in body,
                        "default JSON emitted telemetry")
            else:
                require_covered(envelope, "qwen38", self.args.device)
                self.note_epochs(envelope)
                self.remember_terminal(path)
            qwen[name] = body
        parity = self.compare(qwen["qwen-off"], qwen["qwen-on"],
                              qwen["qwen-off"].get("id"), qwen["qwen-on"].get("id"))
        self.record("qwen-nonstream-off-on-protocol-parity", "PASS", parity)

        sse = {}
        for name, enabled in (("qwen-sse-off", False), ("qwen-sse-on", True)):
            status, frames, _ = self.stream(name, self.text_body(
                telemetry=enabled, stream=True, max_tokens=16))
            require(status == 200 and frames, f"{name} SSE failed")
            envelopes = [frame_telemetry(frame) for frame in frames
                         if frame_telemetry(frame) is not None]
            if enabled:
                require(len(envelopes) == 1, "SSE did not emit one terminal envelope")
                require_covered(envelopes[0], "qwen38", self.args.device)
                self.note_epochs(envelopes[0])
            else:
                require(not envelopes, "default SSE emitted telemetry")
                require(all("telemetry" not in frame for frame in frames),
                        "default SSE top-level telemetry field appeared")
            sse[name] = frames
        # Request ids differ numerically; compare after replacing each actual id.
        first_id = sse["qwen-sse-off"][0].get("id")
        second_id = sse["qwen-sse-on"][0].get("id")
        parity = self.compare(sse["qwen-sse-off"], sse["qwen-sse-on"],
                              first_id, second_id)
        self.record("qwen-sse-off-on-protocol-parity", "PASS", parity)

        answer = qwen["qwen-on"]["choices"][0]["message"]["content"]
        messages = [
            {"role": "user", "content": self.text_body()["messages"][0]["content"]},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": "Reply with the next short sentence."},
        ]
        previous_answer = answer
        for index in range(2):
            body = self.text_body(telemetry=True, max_tokens=16)
            body["messages"] = copy.deepcopy(messages)
            name = f"qwen-prefix-{index + 1}"
            status, response, path = self.chat(name, body)
            require(status == 200, f"{name} HTTP {status}: {response}")
            envelope = response_telemetry(response)
            require_covered(envelope, "qwen38", self.args.device)
            self.note_epochs(envelope)
            self.remember_terminal(path)
            previous_answer = response["choices"][0]["message"]["content"]
            messages.extend([{"role": "assistant", "content": previous_answer},
                             {"role": "user", "content": "Continue briefly."}])
        if self.args.qwen_log:
            text = Path(self.args.qwen_log).read_text(errors="replace")
            reused = [int(value) for value in
                      re.findall(r"prefix_reused=(\d+)", text)]
            require(any(value > 0 for value in reused),
                    f"no nonzero Qwen prefix reuse observed: {reused[-20:]}")
            self.save_json("qwen-prefix-reuse.json", {
                "observed_reused_token_counts": reused,
                "nonzero_observed": True,
            })

        videos = {}
        for name, enabled in (("wan-off", False), ("wan-on", True),
                              ("wan-on-vae-cache", True)):
            status, body, path = self.chat(name, self.video_body(telemetry=enabled))
            require(status == 200, f"{name} HTTP {status}: {body}")
            envelope = response_telemetry(body)
            if enabled:
                require_covered(envelope, "qwen38", self.args.device)
                stage = require_covered(envelope, "video", self.args.device)
                self.note_epochs(envelope)
                self.remember_terminal(path)
            else:
                require(envelope is None, "default video JSON emitted telemetry")
            artifacts = body.get("artifacts", [])
            require(artifacts, f"{name} returned no video artifact")
            videos[name] = body
        parity = self.compare(videos["wan-off"], videos["wan-on"],
                              videos["wan-off"].get("id"), videos["wan-on"].get("id"))
        self.record("wan-off-on-protocol-and-artifact-parity", "PASS", parity)

        bad = self.text_body(telemetry=True)
        bad["messages"] = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64,AAEC"}},
            {"type": "text", "text": "Decode this image"},
        ]}]
        status, body, path = self.chat("qwen-failed", bad)
        require(status >= 400, f"invalid native image unexpectedly succeeded: {status}")
        envelope = response_telemetry(body)
        require(envelope and envelope.get("lifecycle_status") == "failed",
                f"failed lifecycle missing: {body}")
        require_covered(envelope, "qwen38", self.args.device,
                        lifecycle="failed", nonzero=False)
        self.note_epochs(envelope)
        self.remember_terminal(path)

        status, frames, abort = self.stream(
            "qwen-immediate-cancel",
            self.text_body(telemetry=True, stream=True, max_tokens=256),
            abort_first_frame=True)
        require(status == 200 and abort and abort.get("status") == 200,
                f"immediate abort failed: {abort}")
        envelopes = [frame_telemetry(frame) for frame in frames
                     if frame_telemetry(frame) is not None]
        require(len(envelopes) == 1, "cancel did not emit one terminal envelope")
        envelope = envelopes[0]
        require(envelope.get("lifecycle_status") == "cancelled",
                "cancel lifecycle is not cancelled")
        missing = [row for row in envelope.get("stages", [])
                   if row.get("stage") == "vlm"]
        require(len(missing) == 1, "immediate cancel did not retain its VLM boundary")
        stage = missing[0]
        require(stage.get("native_stage") == "unknown",
                "missing terminal fabricated a native stage identity")
        require_not_covered(stage, "stage_terminal_snapshot_missing")
        self.save_json("immediate-cancel-boundary.json", {
            "abort": abort, "telemetry": envelope,
            "boundary": "public cancellation before native terminal snapshot",
        })

        if self.args.audio:
            audio = base64.b64encode(Path(self.args.audio).read_bytes()).decode()
            body = self.text_body(telemetry=True)
            body["messages"] = [{"role": "user", "content": [
                {"type": "input_audio", "input_audio": {
                    "data": audio, "format": "wav"}},
                {"type": "text", "text": "Transcribe briefly"},
            ]}]
            status, response, path = self.chat("asr-uninstrumented", body)
            require(status == 200, f"ASR path failed: {response}")
            envelope = response_telemetry(response)
            require(envelope.get("coverage") == "PARTIAL",
                    "mixed covered/uninstrumented envelope is not PARTIAL")
            require_not_covered(only_stage(envelope, "asr"),
                                "backend_not_instrumented")
            require_covered(envelope, "qwen38", self.args.device)
            self.note_epochs(envelope)
            self.remember_terminal(path)

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            _, health = self.get_json("/health")
            if health.get("in_flight") == 0:
                break
            time.sleep(0.1)
        require(health.get("in_flight") == 0, f"in_flight did not drain: {health}")
        for path, digest in self.terminal_files:
            require(sha256_bytes(path.read_bytes()) == digest,
                    f"terminal response was rewritten: {path.name}")
        for backend in ("qwen38", "video"):
            require(any(peak > 0 for peak in self.peaks.get(backend, [])),
                    f"no truthful nonzero serial allocation peak observed for {backend}")

    def batched(self):
        status, body, _ = self.chat("qwen-batched", self.text_body(telemetry=True))
        require(status == 200, f"batched request failed: {body}")
        envelope = response_telemetry(body)
        stage = only_stage(envelope, "qwen38")
        require_not_covered(stage, "batched_request_attribution_unavailable")
        require(stage.get("execution", {}).get("mode") == "batched",
                "batch execution identity missing")
        require(stage.get("execution", {}).get("batch_capacity", 1) > 1,
                "configured batch capacity not retained")

    def contract_stub(self):
        status, body, _ = self.chat("wan-contract-stub", self.video_body(telemetry=True))
        require(status == 200, f"contract-stub request failed: {body}")
        envelope = response_telemetry(body)
        require_not_covered(only_stage(envelope, "video"),
                            "contract_stub_has_no_cuda_allocator")
        require_covered(envelope, "qwen38", self.args.device)

    def run(self):
        self.monitor.start()
        try:
            if self.args.suite == "serial":
                self.serial()
            elif self.args.suite == "batched":
                self.batched()
            elif self.args.suite == "tts":
                self.tts()
            else:
                self.contract_stub()
        finally:
            self.monitor.stop()
        self.save_json("nvml-samples.json", {
            "scope": "whole-device-used-memory", "poll_interval_ms": 50,
            "samples": self.monitor.rows,
        })
        summary = {
            "suite": self.args.suite, "device_index": self.args.device,
            "status": "PASS" if not self.failures else "FAIL",
            "results": self.results, "failures": self.failures,
            "covered_epochs": self.epochs,
            "allocated_bytes_peaks": self.peaks,
            "media_hashes": self.media_hashes,
            "whole_device_nvml": "independent 50 ms samples; never request-only peaks",
        }
        self.save_json("summary.json", summary)
        rows = []
        for path in sorted(self.evidence.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                rows.append(f"{sha256_bytes(path.read_bytes())}  "
                            f"{path.relative_to(self.evidence).as_posix()}\n")
        (self.evidence / "SHA256SUMS").write_text("".join(rows))
        return 0 if not self.failures else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:43435")
    parser.add_argument("--model", default="qwen3-native")
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--suite", choices=("serial", "batched", "contract-stub", "tts"),
                        default="serial")
    parser.add_argument("--audio", help="optional real PCM16 WAV for uninstrumented ASR")
    parser.add_argument("--qwen-log", help="optional stage log for prefix_reused evidence")
    parser.add_argument("--tts-log", help="fresh native TTS log for cancellation evidence")
    parser.add_argument("--timeout-s", type=float, default=600)
    args = parser.parse_args(argv)
    parsed = urllib.parse.urlsplit(args.base_url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" \
            or parsed.username is not None or parsed.password is not None:
        parser.error("live telemetry gate requires a credential-free loopback HTTP URL")
    if args.device < 0 or args.timeout_s <= 0:
        parser.error("device must be nonnegative and timeout positive")
    if os.environ.get("UMMINFER_LIVE_OPT_IN") != "1":
        print("BLOCKED set UMMINFER_LIVE_OPT_IN=1 for the isolated live gate")
        return 0
    if args.evidence_dir.exists() and any(args.evidence_dir.iterdir()):
        parser.error("evidence directory must be absent or empty")
    return Gate(args).run()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"FAIL request telemetry gate: {exc}", file=sys.stderr)
        sys.exit(2)
