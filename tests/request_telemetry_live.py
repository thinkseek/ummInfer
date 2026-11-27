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
  asr           real Qwen3-ASR + serial Qwen3.8, English/Mandarin PCM16,
                warm repeats, JSON/SSE/transcript parity, failure/cancel, and
                independent 50 ms whole-device NVML (isolated stack required)
  image2        real serial Qwen-Image-2.1 + Qwen3.8, JSON/SSE/PNG parity,
                warm reuse, native failure/cancel, and independent NVML
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
import struct
import zlib
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
    """Remove transport identity/time and opt-in telemetry, retaining media bytes."""
    if isinstance(value, dict):
        return {key: normalize(item, request_id) for key, item in value.items()
                if key not in ("created", "session_id", "telemetry",
                               "request_telemetry", "denoise_ms", "encode_ms",
                               "first_frame_ms",
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
        self.transcripts = {}
        self.input_audio = {}
        self.request_hashes = {}
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
        self.request_hashes[name] = sha256_bytes(raw_request)
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
               abort_first_audio=False, abort_delay_s=0.0,
               abort_image_after_running=False):
        raw_request = canonical(body)
        self.request_hashes[name] = sha256_bytes(raw_request)
        (self.evidence / f"{name}.request.json").write_bytes(raw_request + b"\n")
        parts = urllib.parse.urlsplit(self.base)
        connection = http.client.HTTPConnection(parts.hostname, parts.port,
                                                timeout=self.args.timeout_s)
        connection.request("POST", "/v1/chat/completions", body=raw_request,
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        frames, pending, raw_lines = [], [], bytearray()
        abort_result = None
        trace_abort = {}
        trace_thread = None
        trace_stop = threading.Event()

        def start_trace_abort(request_id):
            def watch():
                try:
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline and not trace_stop.is_set():
                        if self.args.image_trace.exists():
                            for line in self.args.image_trace.read_text(
                                    errors="replace").splitlines():
                                try:
                                    event = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                                legs = event.get("legs", [])
                                if (event.get("request_id") == request_id and
                                        event.get("event") == "legs_snapshot" and
                                        event.get("phase") == "outputs_started" and
                                        any(leg.get("stage") == "image" and
                                            leg.get("state") == "running"
                                            for leg in legs)):
                                    trace_abort["event"] = event
                                    # The trace is emitted immediately after the
                                    # child submit. Let native compute cross its
                                    # first boundary before sending the abort.
                                    time.sleep(0.75)
                                    trace_abort["abort"] = self.abort(request_id)
                                    self.save_json(f"{name}.trace-trigger.json",
                                                   trace_abort)
                                    return
                        trace_stop.wait(0.02)
                    if "abort" not in trace_abort and not trace_stop.is_set():
                        trace_abort["error"] = "image outputs_started trace timeout"
                except Exception as exc:
                    trace_abort["error"] = str(exc)

            trace_thread = threading.Thread(target=watch, daemon=True)
            trace_thread.start()
            return trace_thread

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
                if abort_image_after_running and trace_thread is None:
                    request_id = frame.get("id")
                    require(isinstance(request_id, str) and request_id,
                            "image trace abort frame has no request id")
                    trace_thread = start_trace_abort(request_id)
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
                    if abort_delay_s > 0:
                        time.sleep(abort_delay_s)
                    abort_result = self.abort(request_id)
        finally:
            connection.close()
            trace_stop.set()
            if trace_thread is not None:
                trace_thread.join(timeout=35)
            require(trace_thread is None or not trace_thread.is_alive(),
                    "image trace abort thread did not stop")
            if "error" in trace_abort:
                raise GateFailure(trace_abort["error"])
            if "abort" in trace_abort:
                abort_result = trace_abort["abort"]
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

    def asr_body(self, wav, telemetry=False, stream=False):
        body = self.text_body(telemetry=telemetry, stream=stream, max_tokens=4)
        body["messages"] = [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {
                "data": base64.b64encode(wav).decode(), "format": "wav"}},
            {"type": "text", "text":
             "Transcribe the audio, then give one short acknowledgement."},
        ]}]
        return body

    def retain_pcm16_wav(self, label, path):
        wav = Path(path).read_bytes()
        with wave.open(io.BytesIO(wav), "rb") as source:
            require(source.getnchannels() == 1 and source.getsampwidth() == 2 and
                    source.getframerate() == 16000,
                    f"{label} is not 16 kHz mono PCM16")
            pcm = source.readframes(source.getnframes())
        require(len(pcm) > 0 and len(pcm) % 2 == 0,
                f"{label} has empty or malformed PCM16")
        (self.evidence / f"{label}.input.wav").write_bytes(wav)
        (self.evidence / f"{label}.input.pcm").write_bytes(pcm)
        self.input_audio[label] = {
            "wav_sha256": sha256_bytes(wav), "wav_bytes": len(wav),
            "pcm_sha256": sha256_bytes(pcm), "pcm_bytes": len(pcm),
            "sample_rate": 16000, "channels": 1, "format": "pcm16",
        }
        return wav

    def require_asr_warm_covered(self, envelope, lifecycle="completed"):
        rows = [row for row in envelope.get("stages", [])
                if row.get("native_stage") == "asr"]
        require(len(rows) == 1, "missing unique native ASR telemetry stage")
        single = dict(envelope)
        single["stages"] = rows
        stage = require_covered(single, "asr", self.args.device,
                                lifecycle=lifecycle)
        samples = stage["sample_coverage"]
        require(stage["allocated_bytes_peak"] == 0 and
                stage["reserved_bytes_peak"] == 0 and
                stage["allocated_bytes_final"] == 0 and
                stage["reserved_bytes_final"] == 0 and
                samples["allocation_events"] == 0 and
                samples["free_events"] == 0,
                "ASR pre-request weights/workspaces/graphs entered request ownership")
        return stage

    def retain_asr_transcript(self, name, frames):
        transcripts = []
        for frame in frames:
            unified = frame.get("unified", {})
            if unified.get("kind") == "asr_final":
                transcripts.append(unified.get("text"))
        require(len(transcripts) == 1 and isinstance(transcripts[0], str) and
                transcripts[0], f"{name} did not emit one nonempty ASR transcript")
        text = transcripts[0]
        (self.evidence / f"{name}.transcript.txt").write_bytes(text.encode("utf-8"))
        self.transcripts[name] = {
            "text": text, "utf8_sha256": sha256_bytes(text.encode("utf-8"))}
        return text

    def asr(self):
        status, health = self.get_json("/health")
        require(status == 200 and health.get("in_flight") == 0,
                f"initial health is not idle: {health}")
        status, ready = self.get_json("/ready")
        require(status == 200 and ready.get("status") == "ready",
                "ASR/Qwen stack is not ready")
        english = self.retain_pcm16_wav("asr-english", self.args.audio_en)
        mandarin = self.retain_pcm16_wav("asr-mandarin", self.args.audio_zh)
        log_offset = Path(self.args.asr_log).stat().st_size

        nonstream = {}
        for name, enabled, wav in (
                ("asr-en-on", True, english), ("asr-en-off", False, english),
                ("asr-en-warm-on", True, english),
                ("asr-en-warm-second-on", True, english),
                ("asr-zh-on", True, mandarin)):
            status, response, path = self.chat(
                name, self.asr_body(wav, telemetry=enabled))
            require(status == 200, f"{name} HTTP {status}: {response}")
            envelope = response_telemetry(response)
            if enabled:
                require_covered(envelope, "qwen38", self.args.device)
                self.require_asr_warm_covered(envelope)
                self.note_epochs(envelope)
                self.remember_terminal(path)
            else:
                require(envelope is None and "telemetry" not in response,
                        "default ASR JSON emitted telemetry")
            nonstream[name] = response
        parity = self.compare(nonstream["asr-en-off"], nonstream["asr-en-on"],
                              nonstream["asr-en-off"].get("id"),
                              nonstream["asr-en-on"].get("id"))
        self.record("asr-json-off-on-protocol-parity", "PASS", parity)
        warm_parity = self.compare(
            nonstream["asr-en-warm-on"], nonstream["asr-en-warm-second-on"],
            nonstream["asr-en-warm-on"].get("id"),
            nonstream["asr-en-warm-second-on"].get("id"))
        self.record("asr-warm-repeat-protocol-parity", "PASS", warm_parity)

        streamed = {}
        for name, enabled, wav in (
                ("asr-en-sse-off", False, english),
                ("asr-en-sse-on", True, english),
                ("asr-zh-sse-on", True, mandarin)):
            status, frames, _ = self.stream(
                name, self.asr_body(wav, telemetry=enabled, stream=True))
            require(status == 200 and frames, f"{name} SSE failed")
            envelopes = [frame_telemetry(frame) for frame in frames
                         if frame_telemetry(frame) is not None]
            if enabled:
                require(len(envelopes) == 1,
                        "ASR SSE did not emit one terminal envelope")
                require_covered(envelopes[0], "qwen38", self.args.device)
                self.require_asr_warm_covered(envelopes[0])
                self.note_epochs(envelopes[0])
            else:
                require(not envelopes and
                        all("telemetry" not in frame for frame in frames),
                        "default ASR SSE emitted telemetry")
            self.retain_asr_transcript(name, frames)
            streamed[name] = frames
        first_id = streamed["asr-en-sse-off"][0].get("id")
        second_id = streamed["asr-en-sse-on"][0].get("id")
        parity = self.compare(streamed["asr-en-sse-off"],
                              streamed["asr-en-sse-on"], first_id, second_id)
        self.record("asr-sse-off-on-protocol-parity", "PASS", parity)
        require(self.transcripts["asr-en-sse-off"]["text"] ==
                self.transcripts["asr-en-sse-on"]["text"],
                "ASR off/on English transcript bytes differ")
        self.record("asr-sse-off-on-transcript-parity", "PASS",
                    self.transcripts["asr-en-sse-off"]["utf8_sha256"])

        # The public WAV is valid nonempty PCM16. Its PCM happens to start with
        # RIFF, so the existing native stage auto-detection treats those payload
        # bytes as a malformed inner WAV and raises before STFT/encoder work.
        # This reaches a real native failed terminal without changing defaults.
        invalid_pcm = b"RIFF" + b"\x01\x00" * 1600
        invalid_wav_buffer = io.BytesIO()
        with wave.open(invalid_wav_buffer, "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(16000)
            target.writeframes(invalid_pcm)
        invalid_wav = invalid_wav_buffer.getvalue()
        (self.evidence / "asr-invalid-native.input.wav").write_bytes(invalid_wav)
        (self.evidence / "asr-invalid-native.input.pcm").write_bytes(invalid_pcm)
        status, failed, path = self.chat(
            "asr-invalid-native-failed",
            self.asr_body(invalid_wav, telemetry=True))
        require(status >= 500, f"invalid native audio became HTTP {status}: {failed}")
        envelope = response_telemetry(failed)
        require(envelope and envelope.get("lifecycle_status") == "failed",
                f"ASR failed lifecycle missing: {failed}")
        self.require_asr_warm_covered(envelope, lifecycle="failed")
        self.note_epochs(envelope)
        self.remember_terminal(path)

        # Abort at the first public SSE frame, while the English ASR decode is
        # still running. The native log independently retains its actual abort.
        status, frames, abort = self.stream(
            "asr-cancelled", self.asr_body(english, telemetry=True, stream=True),
            abort_first_frame=True, abort_delay_s=0.05)
        require(status == 200 and abort and abort.get("status") == 200,
                f"ASR in-flight abort failed: {abort}")
        envelopes = [frame_telemetry(frame) for frame in frames
                     if frame_telemetry(frame) is not None]
        require(len(envelopes) == 1 and
                envelopes[0].get("lifecycle_status") == "cancelled",
                "ASR cancellation did not emit one cancelled envelope")
        envelope = envelopes[0]
        asr_rows = [row for row in envelope.get("stages", [])
                    if row.get("stage") == "asr"]
        require(len(asr_rows) == 1, "ASR cancellation lost its ASR boundary")
        if asr_rows[0].get("native_stage") == "asr":
            self.require_asr_warm_covered(envelope, lifecycle="cancelled")
            self.note_epochs(envelope)
        else:
            require_not_covered(asr_rows[0], "stage_terminal_snapshot_missing")
        self.save_json("asr-cancelled-boundary.json", {
            "abort": abort, "telemetry": envelope,
            "boundary": "public cancellation 50 ms after first SSE frame",
        })

        request_id = envelope["request_id"]
        marker = ("asr request aborted request_id=" + request_id).encode()
        native_log = b""
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            native_log = Path(self.args.asr_log).read_bytes()[log_offset:]
            if marker in native_log:
                break
            time.sleep(0.05)
        (self.evidence / "asr-native-log-tail.txt").write_bytes(native_log)
        require(marker in native_log,
                "native ASR log did not retain this request's actual cancellation")

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            _, health = self.get_json("/health")
            if health.get("in_flight") == 0:
                break
            time.sleep(0.1)
        require(health.get("in_flight") == 0, f"in_flight did not drain: {health}")
        self.save_json("asr-final-health.json", health)
        for path, digest in self.terminal_files:
            require(sha256_bytes(path.read_bytes()) == digest,
                    f"terminal response was rewritten: {path.name}")
        require(len(self.epochs.get("asr", [])) >= 7,
                "insufficient unique ASR request epochs")
        require(all(peak == 0 for peak in self.peaks.get("asr", [])),
                "ASR pool-reuse peak was not the measured zero")

    def image2_body(self, telemetry=False, stream=False, steps=2, edit=False):
        body = self.text_body(telemetry=telemetry, stream=stream, max_tokens=1)
        body["messages"] = [{"role": "user", "content": "Reply with exactly K."}]
        body["modalities"] = ["text", "image"]
        body["image"] = {
            "prompt": "A red fox sitting in a snowy forest at dawn",
            "size": "512x512", "steps": steps, "seed": 1234,
            "output_format": "png",
        }
        if edit:
            # Valid public edit input reaches the existing native Qwen-Image
            # rejection boundary; no fault injection or schema change is used.
            body["image"]["image"] = (
                "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"
                "AAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")
        return body

    def save_png(self, name, png):
        require(png[:8] == b"\x89PNG\r\n\x1a\n", f"{name} is not PNG")
        offset = 8
        idat = bytearray()
        width = height = None
        ended = False
        while offset < len(png):
            require(offset + 12 <= len(png), f"{name} truncated PNG chunk")
            size = struct.unpack(">I", png[offset:offset + 4])[0]
            kind = png[offset + 4:offset + 8]
            end = offset + 12 + size
            require(end <= len(png), f"{name} overflowing PNG chunk")
            payload = png[offset + 8:offset + 8 + size]
            expected_crc = struct.unpack(">I", png[offset + 8 + size:end])[0]
            require((zlib.crc32(kind + payload) & 0xffffffff) == expected_crc,
                    f"{name} PNG CRC mismatch")
            if kind == b"IHDR":
                width, height, depth, color, compression, filtering, interlace = \
                    struct.unpack(">IIBBBBB", payload)
                require((width, height, depth, color, compression, filtering,
                         interlace) == (512, 512, 8, 6, 0, 0, 0),
                        f"{name} is not 512x512 non-interlaced RGBA8")
            elif kind == b"IDAT":
                idat.extend(payload)
            elif kind == b"IEND":
                require(size == 0 and end == len(png), f"{name} invalid PNG IEND")
                ended = True
            offset = end
        require(ended and width == 512 and height == 512 and idat,
                f"{name} incomplete PNG")
        decoded = zlib.decompress(bytes(idat))
        row_bytes = 512 * 4
        require(len(decoded) == (row_bytes + 1) * 512,
                f"{name} decoded RGBA length mismatch")
        rows = [decoded[y * (row_bytes + 1):(y + 1) * (row_bytes + 1)]
                for y in range(512)]
        require(all(row[0] == 0 for row in rows),
                f"{name} unexpectedly changed native PNG filter type")
        pixels = b"".join(row[1:] for row in rows)
        require(len(set(pixels)) > 1, f"{name} decoded to a constant image")
        (self.evidence / f"{name}.png").write_bytes(png)
        self.media_hashes[name] = {
            "png_sha256": sha256_bytes(png), "png_bytes": len(png),
            "rgba_sha256": sha256_bytes(pixels), "rgba_bytes": len(pixels),
            "width": width, "height": height, "color_type": "RGBA8",
        }
        return sha256_bytes(png)

    def save_nonstream_png(self, name, body):
        artifacts = [item for item in body.get("artifacts", [])
                     if item.get("media_type") == "image"]
        require(len(artifacts) == 1, f"{name} did not return one image artifact")
        artifact = artifacts[0]
        require("request_telemetry" not in artifact,
                "internal image telemetry leaked into an artifact")
        png = base64.b64decode(artifact["data"], validate=True)
        require(len(png) == artifact.get("bytes") and
                sha256_bytes(png) == artifact.get("sha256"),
                f"{name} image SHA-256/byte count mismatch")
        return self.save_png(name, png)

    def save_stream_png(self, name, frames):
        chunks = {}
        totals = {}
        expected = {}
        for frame in frames:
            unified = frame.get("unified", {})
            if unified.get("kind") != "artifact":
                continue
            artifact = unified.get("artifact", {})
            if artifact.get("mime_type") != "image/png":
                continue
            identity = artifact["artifact_id"]
            index, total = artifact["index"], artifact["total"]
            require(type(index) is int and type(total) is int and
                    0 <= index < total,
                    f"{name} invalid image chunk index/total")
            # Totals are fixed for one artifact and every index arrives once.
            require(totals.setdefault(identity, total) == total,
                    f"{name} image chunk total changed")
            parts = chunks.setdefault(identity, {})
            require(index not in parts, f"{name} duplicate image chunk {index}")
            parts[index] = base64.b64decode(artifact["data"], validate=True)
            if "sha256" in artifact:
                require(index + 1 == total, f"{name} digest arrived before final chunk")
                expected[identity] = (artifact["sha256"], artifact["bytes"])
        require(len(chunks) == 1 and expected.keys() == chunks.keys(),
                f"{name} did not stream one complete PNG")
        identity = next(iter(chunks))
        require(set(chunks[identity]) == set(range(totals[identity])),
                f"{name} PNG chunk indexes are incomplete")
        png = b"".join(chunks[identity][index]
                       for index in range(totals[identity]))
        digest, size = expected[identity]
        require(size == len(png) and digest == sha256_bytes(png),
                f"{name} streamed PNG digest/size mismatch")
        return self.save_png(name, png)

    def require_image2_covered(self, envelope, lifecycle="completed",
                               nonzero=True):
        stage = require_covered(envelope, "image", self.args.device,
                                lifecycle=lifecycle, nonzero=nonzero)
        require(stage.get("backend") == "qwen-image-2.1",
                f"wrong image backend identity: {stage.get('backend')}")
        samples = stage["sample_coverage"]
        if nonzero:
            require(samples["allocation_events"] > 0 and
                    samples["free_events"] == samples["allocation_events"] and
                    stage["allocated_bytes_final"] == 0 and
                    stage["reserved_bytes_final"] == 0,
                    "image2 direct request blocks did not close at native terminal")
        else:
            require(samples["allocation_events"] == 0 and
                    samples["free_events"] == 0 and
                    stage["allocated_bytes_peak"] == 0 and
                    stage["allocated_bytes_final"] == 0,
                    "native input-validation failure invented direct allocations")
        return stage

    def image2(self):
        status, health = self.get_json("/health")
        require(status == 200 and health.get("in_flight") == 0,
                f"initial health is not idle: {health}")
        status, ready = self.get_json("/ready")
        require(status == 200 and ready.get("status") == "ready",
                "Qwen/Image2 stack is not ready")
        image_log_offset = self.args.image_log.stat().st_size

        # Load and retain model weights before the first covered epoch. This
        # makes the pre-request exclusion observable instead of charging lazy
        # model construction to the first measured request.
        nonstream = {}
        for name, enabled in (("image2-warmup-off", False),
                              ("image2-json-off", False),
                              ("image2-json-on", True),
                              ("image2-json-warm-on", True)):
            status, response, path = self.chat(
                name, self.image2_body(telemetry=enabled))
            require(status == 200, f"{name} HTTP {status}: {response}")
            envelope = response_telemetry(response)
            if enabled:
                require_covered(envelope, "qwen38", self.args.device)
                self.require_image2_covered(envelope)
                self.note_epochs(envelope)
                self.remember_terminal(path)
            else:
                require(envelope is None and "telemetry" not in response,
                        f"default image JSON emitted telemetry: {name}")
            self.save_nonstream_png(name, response)
            nonstream[name] = response
        parity = self.compare(
            nonstream["image2-json-off"], nonstream["image2-json-on"],
            nonstream["image2-json-off"].get("id"),
            nonstream["image2-json-on"].get("id"))
        self.record("image2-json-off-on-protocol-parity", "PASS", parity)
        warm_parity = self.compare(
            nonstream["image2-json-on"], nonstream["image2-json-warm-on"],
            nonstream["image2-json-on"].get("id"),
            nonstream["image2-json-warm-on"].get("id"))
        self.record("image2-warm-repeat-protocol-parity", "PASS", warm_parity)

        streamed = {}
        for name, enabled in (("image2-sse-off", False),
                              ("image2-sse-on", True)):
            status, frames, _ = self.stream(
                name, self.image2_body(telemetry=enabled, stream=True))
            require(status == 200 and frames, f"{name} SSE failed")
            envelopes = [frame_telemetry(frame) for frame in frames
                         if frame_telemetry(frame) is not None]
            if enabled:
                require(len(envelopes) == 1,
                        "image2 SSE did not emit one terminal envelope")
                require_covered(envelopes[0], "qwen38", self.args.device)
                self.require_image2_covered(envelopes[0])
                self.note_epochs(envelopes[0])
            else:
                require(not envelopes and
                        all("telemetry" not in frame for frame in frames),
                        "default image SSE emitted telemetry")
            self.save_stream_png(name, frames)
            streamed[name] = frames
        sse_parity = self.compare(
            streamed["image2-sse-off"], streamed["image2-sse-on"],
            streamed["image2-sse-off"][0].get("id"),
            streamed["image2-sse-on"][0].get("id"))
        self.record("image2-sse-off-on-protocol-parity", "PASS", sse_parity)

        status, failed, path = self.chat(
            "image2-native-failed", self.image2_body(telemetry=True, edit=True))
        require(status == 200, f"optional image failure became HTTP {status}")
        require(any(leg.get("stage") == "image" and leg.get("state") == "failed"
                    for leg in failed.get("legs", [])),
                f"native image failure was not reported in legs: {failed}")
        failed_envelope = response_telemetry(failed)
        require_covered(failed_envelope, "qwen38", self.args.device)
        failed_stage = self.require_image2_covered(
            failed_envelope, lifecycle="failed", nonzero=False)
        require(failed_stage["epoch"]["id"] > 0, "failed image epoch missing")
        self.note_epochs(failed_envelope)
        self.remember_terminal(path)

        status, frames, abort = self.stream(
            "image2-cancelled", self.image2_body(
                telemetry=True, stream=True, steps=40),
            abort_image_after_running=True)
        require(status == 200 and abort and abort.get("status") == 200,
                f"image in-flight abort failed: {abort}")
        envelopes = [frame_telemetry(frame) for frame in frames
                     if frame_telemetry(frame) is not None]
        require(len(envelopes) == 1 and
                envelopes[0].get("lifecycle_status") == "cancelled",
                "image cancellation did not emit one cancelled envelope")
        cancel_envelope = envelopes[0]
        image_rows = [row for row in cancel_envelope.get("stages", [])
                      if row.get("stage") == "image"]
        require(len(image_rows) == 1, "image cancellation lost its image boundary")
        if image_rows[0].get("native_stage") == "image":
            self.require_image2_covered(
                cancel_envelope, lifecycle="cancelled", nonzero=True)
            self.note_epochs(cancel_envelope)
        else:
            require_not_covered(image_rows[0], "stage_terminal_snapshot_missing")
        self.save_json("image2-cancelled-boundary.json", {
            "abort": abort, "telemetry": cancel_envelope,
            "boundary": "abort after traced native image child was running",
        })

        request_id = cancel_envelope["request_id"]
        marker = ("image request aborted backend=2.1 request_id=" +
                  request_id + ":image").encode()
        native_log = b""
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            native_log = self.args.image_log.read_bytes()[image_log_offset:]
            if marker in native_log and b"denoise step" in native_log:
                break
            time.sleep(0.05)
        (self.evidence / "image2-native-log-tail.txt").write_bytes(native_log)
        require(marker in native_log and b"denoise step" in native_log,
                "native image log did not prove denoise cancellation")

        status, recovered, path = self.chat(
            "image2-after-cancel-on", self.image2_body(telemetry=True))
        require(status == 200, f"post-cancel image request failed: {recovered}")
        recovered_envelope = response_telemetry(recovered)
        require_covered(recovered_envelope, "qwen38", self.args.device)
        self.require_image2_covered(recovered_envelope)
        self.note_epochs(recovered_envelope)
        self.remember_terminal(path)
        self.save_nonstream_png("image2-after-cancel-on", recovered)
        recovery_parity = self.compare(
            nonstream["image2-json-on"], recovered,
            nonstream["image2-json-on"].get("id"), recovered.get("id"))
        self.record("image2-post-cancel-protocol-parity", "PASS", recovery_parity)

        success_names = (
            "image2-warmup-off", "image2-json-off", "image2-json-on",
            "image2-json-warm-on", "image2-sse-off", "image2-sse-on",
            "image2-after-cancel-on")
        digests = {self.media_hashes[name]["png_sha256"] for name in success_names}
        require(len(digests) == 1,
                f"fixed-seed off/on/warm/SSE/recovery PNG bytes differ: {digests}")
        self.record("image2-all-success-png-byte-parity", "PASS", next(iter(digests)))

        # Native timings prove the first uninstrumented request loaded the model
        # and all measured successful repeats reused those resident components.
        final_native_log = self.args.image_log.read_bytes()[image_log_offset:]
        (self.evidence / "image2-native-log-tail.txt").write_bytes(final_native_log)
        request_ids = {name: response["id"] for name, response in nonstream.items()}
        request_ids.update({name: frames[0]["id"] for name, frames in streamed.items()})
        request_ids["image2-after-cancel-on"] = recovered["id"]
        load_records = {}
        for name in ("image2-warmup-off", "image2-json-off", "image2-json-on",
                     "image2-json-warm-on", "image2-sse-off", "image2-sse-on",
                     "image2-after-cancel-on"):
            child = request_ids[name] + ":image"
            match = re.search(
                rb"image request backend=2\.1 request_id=" + re.escape(child.encode()) +
                rb"[^\n]*\bload_ms=([0-9]+)(?:\.[0-9]+)?", final_native_log)
            require(match is not None, f"missing native load timing for {child}")
            load_ms = int(match.group(1))
            if name == "image2-warmup-off":
                require(load_ms > 0, "warmup did not perform the actual model load")
            else:
                require(load_ms == 0, f"{name} did not reuse pre-request model weights")
            load_records[name] = {"request_id": request_ids[name], "load_ms": load_ms}
        self.save_json("image2-warm-reuse.json", load_records)

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            _, health = self.get_json("/health")
            if health.get("in_flight") == 0:
                break
            time.sleep(0.1)
        require(health.get("in_flight") == 0, f"in_flight did not drain: {health}")
        self.save_json("image2-final-health.json", health)
        for terminal_path, digest in self.terminal_files:
            require(sha256_bytes(terminal_path.read_bytes()) == digest,
                    f"terminal response was rewritten: {terminal_path.name}")
        require(len(self.epochs.get("image", [])) >= 5,
                "insufficient unique covered image2 request epochs")
        require(len(self.request_hashes) == 9,
                f"expected nine public image2 requests, got {len(self.request_hashes)}")
        self.save_json("image2-zero-event-contract.json", {
            "successful_live_epochs": "nonzero direct request allocations required",
            "native_validation_failure": "zero events because parsing failed before allocation",
            "warm_fixture_zero": "truthful only because that fixture epoch performs no direct allocation",
            "unsupported": "NOT_COVERED with null bytes, never zero",
        })

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
            elif self.args.suite == "asr":
                self.asr()
            elif self.args.suite == "batched":
                self.batched()
            elif self.args.suite == "tts":
                self.tts()
            elif self.args.suite == "image2":
                self.image2()
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
            "transcripts": self.transcripts,
            "input_audio": self.input_audio,
            "canonical_request_sha256": self.request_hashes,
            "public_request_count": len(self.request_hashes),
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
    parser.add_argument("--suite", choices=("serial", "asr", "tts", "image2",
                                            "batched", "contract-stub"),
                        default="serial")
    parser.add_argument("--audio", help="optional real PCM16 WAV for uninstrumented ASR")
    parser.add_argument("--audio-en", help="real English 16 kHz mono PCM16 WAV for ASR suite")
    parser.add_argument("--audio-zh", help="real Mandarin 16 kHz mono PCM16 WAV for ASR suite")
    parser.add_argument("--asr-log", help="fresh native ASR log for cancellation evidence")
    parser.add_argument("--qwen-log", help="optional stage log for prefix_reused evidence")
    parser.add_argument("--tts-log", help="fresh native TTS log for cancellation evidence")
    parser.add_argument("--image-log", type=Path,
                        help="fresh native image2 log for cancellation evidence")
    parser.add_argument("--image-trace", type=Path,
                        help="fresh gateway UMM_TRACE_FILE for image-running cancellation")
    parser.add_argument("--timeout-s", type=float, default=600)
    args = parser.parse_args(argv)
    parsed = urllib.parse.urlsplit(args.base_url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" \
            or parsed.username is not None or parsed.password is not None:
        parser.error("live telemetry gate requires a credential-free loopback HTTP URL")
    if args.device < 0 or args.timeout_s <= 0:
        parser.error("device must be nonnegative and timeout positive")
    if args.suite == "asr" and not (args.audio_en and args.audio_zh and args.asr_log):
        parser.error("ASR suite requires --audio-en, --audio-zh, and --asr-log")
    if args.suite == "image2" and not (args.image_log and args.image_trace):
        parser.error("image2 suite requires --image-log and --image-trace")
    if args.suite == "image2" and args.device != 1:
        parser.error("image2 suite requires explicitly isolated physical GPU 1")
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
