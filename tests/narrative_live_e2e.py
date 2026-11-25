#!/usr/bin/env python3
"""Offline contract check and live E2E harness for interleaved narratives.

  python3 -I tests/narrative_live_e2e.py --self-check
  python3 -I tests/narrative_live_e2e.py --url http://127.0.0.1:43301 \
      --model qwen3-native --out-dir /tmp/narrative-live

Only the Python standard library is required. Live video swaps additionally
require ffprobe on PATH. Live mode runs real image/video/TTS inference.
"""

import argparse
import base64
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request


PNG = b"\x89PNG\r\n\x1a\n" + b"mock-image"
MP4 = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"mock-video"
PCM = b"\x01\x00\x02\x00" * 128
MARKER = re.compile(r"\[\[\s*(?:image|video|visual)\s*:", re.IGNORECASE)
TERMINAL_KINDS = {"complete", "error", "cancelled"}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def envelope(seq, kind, text="", finish=None, **fields):
    unified = {
        "seq": seq,
        "kind": kind,
        "request_id": "http-1",
        "session_id": "narrative-self-check",
        "stage": "vlm",
        "is_final": kind in TERMINAL_KINDS,
        "text": text,
        **fields,
    }
    delta = {"content": text} if kind == "vlm_delta" else {}
    return {
        "id": "http-1",
        "object": "chat.completion.chunk",
        "created": 1764000000,
        "model": "narrative-mock",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        "unified": unified,
    }


def artifact_event(seq, artifact_id, media_type, mime_type, raw, stage,
                   slot_id=None, slot_index=None):
    artifact = {
        "artifact_id": artifact_id,
        "media_type": media_type,
        "mime_type": mime_type,
        "encoding": "base64",
        "data": base64.b64encode(raw).decode("ascii"),
        "index": 0,
        "total": 1,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if media_type == "audio":
        artifact.update({
            "format": "pcm16", "sample_rate": 24000, "channels": 1,
            "stream_index": 0,
        })
    else:
        artifact.update({
            "format": "png" if media_type == "image" else "mp4",
            "size": "512x512" if media_type == "image" else "480x320",
            "seed": 42, "slot_id": slot_id, "slot_index": slot_index,
        })
    return envelope(seq, "artifact", stage=stage, is_final=True, artifact=artifact)


def mock_events():
    first = "At dawn, a red lantern"
    second = " rises above the lake."
    return [
        envelope(0, "vlm_delta", first),
        envelope(1, "visual_slot", stage="image", slot_id="slot-0",
                 slot_index=0, media_type="image",
                 prompt="a glowing red lantern over a misty mountain lake",
                 text_offset=len(first)),
        artifact_event(2, "art-audio-0", "audio", "audio/pcm", PCM, "tts"),
        envelope(3, "vlm_delta", second),
        envelope(4, "visual_slot", stage="video", slot_id="slot-1",
                 slot_index=1, media_type="video",
                 prompt="mist drifting across the lake as the lantern floats downstream",
                 text_offset=len(first + second)),
        artifact_event(5, "art-visual-0", "image", "image/png", PNG, "image",
                       "slot-0", 0),
        artifact_event(6, "art-visual-1", "video", "video/mp4", MP4, "video",
                       "slot-1", 1),
        envelope(7, "complete", first + second, finish="stop", legs=[
            {"stage": "image", "slot_id": "slot-0", "state": "completed"},
            {"stage": "video", "slot_id": "slot-1", "state": "completed"},
            {"stage": "tts", "state": "completed"},
        ]),
    ]


class MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def reply(self, status, body, content_type="application/json", kind=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if kind is not None:
            self.send_header("X-UmmInfer-Kind", kind)
        self.end_headers()
        self.wfile.write(body)

    def error(self, message, status=400):
        self.reply(status, json.dumps({
            "error": {"message": message, "type": "invalid_request"}
        }).encode("utf-8"))

    def do_GET(self):
        if self.path == "/health":
            self.reply(200, b'{"status":"ok","in_flight":0}')
        else:
            self.error("not found", 404)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
            if len(raw) > 1024 * 1024:
                self.error("request too large")
                return
            if self.path == "/v1/visual/render":
                self.render(raw)
                return
            if self.path != "/v1/chat/completions":
                self.error("not found", 404)
                return
            request = json.loads(raw)
            self.server.chat_requests.append(request)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            for event in mock_events():
                frame = json.dumps(event, separators=(",", ":")).encode("utf-8")
                self.wfile.write(b"data: " + frame + b"\n\n")
                self.wfile.flush()
                time.sleep(0.002)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.error("invalid JSON")
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def render(self, raw):
        request = json.loads(raw)
        allowed = {"kind", "prompt", "size", "steps", "num_frames", "fps",
                   "seconds", "guidance_scale", "seed", "image"}
        if not isinstance(request, dict) or set(request) - allowed:
            self.error("unknown or invalid fields")
            return
        kind = request.get("kind")
        prompt = request.get("prompt")
        if kind not in ("image", "video"):
            self.error("kind must be image or video")
            return
        if not isinstance(prompt, str) or not prompt.strip():
            self.error("prompt must be a non-empty string")
            return
        for name in ("steps", "num_frames", "fps", "seconds"):
            if name in request and (not is_int(request[name]) or request[name] <= 0):
                self.error(f"{name} must be a positive integer")
                return
        if "size" in request and (not isinstance(request["size"], str)
                                  or not request["size"]):
            self.error("size must be a non-empty string")
            return
        if "guidance_scale" in request and (
                isinstance(request["guidance_scale"], bool)
                or not isinstance(request["guidance_scale"], (int, float))
                or (isinstance(request["guidance_scale"], float)
                    and not math.isfinite(request["guidance_scale"]))
                or request["guidance_scale"] < 0):
            self.error("guidance_scale must be non-negative")
            return
        if "seed" in request and (not is_int(request["seed"])
                                  or not -(1 << 63) <= request["seed"] < (1 << 63)):
            self.error("seed must be a signed 64-bit integer")
            return
        if "image" in request and (not isinstance(request["image"], str)
                                   or not request["image"].startswith("data:image/")):
            self.error("image must be an image data URL")
            return
        if kind == "image" and set(request) & {"num_frames", "fps", "seconds",
                                                "guidance_scale"}:
            self.error("video fields require kind=video")
            return
        if kind == "video" and "image" in request:
            self.error("image requires kind=image")
            return
        self.server.render_requests.append(request)
        if kind == "image":
            self.reply(200, PNG + b"-render", "image/png", "image")
        else:
            self.reply(200, MP4 + b"-render", "video/mp4", "video")


def narrative_request(model):
    return {
        "model": model,
        "stream": True,
        "max_tokens": 256,
        "messages": [{
            "role": "user",
            "content": (
                "Tell a two-beat story about a red lantern crossing a mountain "
                "lake at dawn. Use one still visual for the first beat and one "
                "motion visual for the second beat."
            ),
        }],
        "modalities": ["text", "image", "audio", "video"],
        "image": {"size": "512x512", "output_format": "png", "steps": 4},
        "audio": {"voice": "vivian", "format": "wav", "stream": True},
        "video": {
            "size": "480x320", "output_format": "mp4",
            "num_frames": 13, "steps": 15, "guidance_scale": 5.0,
        },
        "narrative": {"enabled": True, "visual_mode": "auto", "max_slots": 2},
    }


def http_request(url, body=None, timeout=30.0):
    data = body if isinstance(body, bytes) else (
        None if body is None else json.dumps(body).encode("utf-8"))
    request = urllib.request.Request(
        url, data=data, method="GET" if body is None else "POST",
        headers={"Content-Type": "application/json"} if body is not None else {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers, error.read()


def stream_chat(base_url, body, timeout):
    url = base_url.rstrip("/") + "/v1/chat/completions"
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    started = time.perf_counter()
    observations = []
    payloads = []
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:2000]
        raise AssertionError(f"chat HTTP {error.code}: {detail}") from error
    with response:
        require(response.status == 200, f"chat HTTP {response.status}")
        require(response.headers.get_content_type() == "text/event-stream",
                "chat response is not text/event-stream")
        for raw_line in response:
            line = raw_line.decode("utf-8").rstrip("\r\n")
            if not line.startswith("data:"):
                continue
            payload = line[5:].lstrip(" ")
            payloads.append(payload)
            if payload == "[DONE]":
                continue
            frame = json.loads(payload)
            unified = frame.get("unified")
            if isinstance(unified, dict):
                observations.append({
                    "time": time.perf_counter() - started,
                    "event": unified,
                    "frame": frame,
                })
    return {
        "observations": observations,
        "payloads": payloads,
        "total_s": time.perf_counter() - started,
    }


def assemble_artifacts(observations):
    groups = {}
    for observation in observations:
        event = observation["event"]
        if event.get("kind") != "artifact":
            continue
        artifact = event.get("artifact")
        require(isinstance(artifact, dict), "artifact event lacks an artifact object")
        artifact_id = artifact.get("artifact_id")
        require(isinstance(artifact_id, str) and artifact_id,
                "artifact_id must be a non-empty string")
        require(artifact.get("encoding") == "base64", "artifact encoding is not base64")
        index, total = artifact.get("index"), artifact.get("total")
        require(is_int(index) and is_int(total) and total > 0 and 0 <= index < total,
                f"invalid artifact chunk bounds: {artifact_id}")
        group = groups.setdefault(artifact_id, {
            "chunks": {}, "next_index": 0, "first": observation,
            "last": observation, "descriptors": [], "total": total,
            "stage": event.get("stage"),
        })
        require(total == group["total"],
                f"artifact total changed between chunks: {artifact_id}")
        require(event.get("stage") == group["stage"],
                f"artifact stage changed between chunks: {artifact_id}")
        require(index == group["next_index"],
                f"out-of-order artifact chunk: {artifact_id} index={index}")
        group["next_index"] += 1
        group["last"] = observation
        group["descriptors"].append(artifact)
        try:
            group["chunks"][index] = base64.b64decode(artifact["data"], validate=True)
        except Exception as error:
            raise AssertionError(f"invalid base64 in {artifact_id}: {error}") from error
        final = index + 1 == total
        require(("bytes" in artifact) == final and ("sha256" in artifact) == final,
                f"bytes/sha256 must be present only on the final chunk: {artifact_id}")
        if event.get("stage") != "tts":
            require(isinstance(artifact.get("slot_id"), str)
                    and is_int(artifact.get("slot_index")),
                    f"visual artifact lacks slot attribution: {artifact_id}")
    assembled = []
    for artifact_id, group in groups.items():
        descriptors = group["descriptors"]
        total = descriptors[0]["total"]
        require(sorted(group["chunks"]) == list(range(total)),
                f"missing artifact chunks: {artifact_id}")
        raw = b"".join(group["chunks"][index] for index in range(total))
        final = descriptors[-1]
        require(is_int(final.get("bytes")) and final["bytes"] == len(raw),
                f"artifact byte mismatch: {artifact_id}")
        require(isinstance(final.get("sha256"), str)
                and final["sha256"] == hashlib.sha256(raw).hexdigest(),
                f"artifact SHA-256 mismatch: {artifact_id}")
        for key in ("media_type", "mime_type", "slot_id", "slot_index", "total"):
            values = {descriptor.get(key) for descriptor in descriptors}
            require(len(values) == 1, f"artifact {key} changed between chunks: {artifact_id}")
        assembled.append({
            "artifact_id": artifact_id, "bytes": raw, "descriptor": final,
            "first_time": group["first"]["time"], "time": group["last"]["time"],
            "stage": group["last"]["event"].get("stage"),
            "seq": group["last"]["event"]["seq"],
        })
    return assembled


def validate_stream(stream, mock=False):
    observations = stream["observations"]
    require(observations, "no unified events received")
    seqs = [item["event"].get("seq") for item in observations]
    require(seqs == list(range(len(seqs))), f"seq is not gap-free from zero: {seqs}")
    terminals = [item for item in observations
                 if item["event"].get("kind") in TERMINAL_KINDS]
    require(len(terminals) == 1 and terminals[0] is observations[-1],
            "stream must contain exactly one final terminal event")
    require(terminals[0]["event"]["kind"] == "complete",
            f"request did not complete: {terminals[0]['event']}")
    require(stream["payloads"].count("[DONE]") == 1
            and stream["payloads"][-1] == "[DONE]",
            "stream must end with exactly one [DONE]")

    deltas = [item for item in observations if item["event"].get("kind") == "vlm_delta"]
    require(deltas, "no vlm_delta received")
    for item in deltas:
        event = item["event"]
        require(isinstance(event.get("text"), str), "vlm_delta text must be a string")
        content = item["frame"]["choices"][0]["delta"].get("content")
        require(content == event["text"], "OpenAI delta and unified text differ")
    prose = "".join(item["event"]["text"] for item in deltas)
    terminal_text = terminals[0]["event"].get("text", "")
    require(not MARKER.search(prose) and not MARKER.search(terminal_text),
            "clean prose contains a raw visual directive")
    require(terminal_text == prose, "terminal text differs from aggregated vlm_delta prose")

    slots = [item for item in observations if item["event"].get("kind") == "visual_slot"]
    require(slots, "no visual_slot received")
    require(len(slots) <= 2, "narrative exceeded max_slots=2")
    for expected_index, item in enumerate(slots):
        event = item["event"]
        require(event.get("slot_id") == f"slot-{expected_index}"
                and event.get("slot_index") == expected_index,
                f"visual slot attribution is inconsistent: {event}")
        require(event.get("media_type") in ("image", "video")
                and event.get("stage") == event["media_type"],
                f"visual slot stage/media_type mismatch: {event}")
        require(isinstance(event.get("prompt"), str) and event["prompt"].strip(),
                "visual slot prompt must be non-empty")
        require(is_int(event.get("text_offset")) and event["text_offset"] >= 0,
                "visual slot text_offset must be a non-negative integer")
    require(slots[0]["time"] < deltas[-1]["time"],
            "first visual slot was dispatched only after the final vlm_delta")

    artifacts = assemble_artifacts(observations)
    visual = [item for item in artifacts if item["descriptor"]["media_type"] in ("image", "video")]
    audio = [item for item in artifacts if item["descriptor"]["mime_type"] == "audio/pcm"]
    audio.sort(key=lambda item: item["descriptor"].get("stream_index", 0))
    require(len(artifacts) == len(visual) + len(audio)
            and all(item["descriptor"]["media_type"] == "audio" for item in audio),
            "narrative returned an unexpected artifact modality or a final WAV")
    require(audio, "no streaming audio/pcm artifact received")
    for item in audio:
        descriptor = item["descriptor"]
        require(descriptor.get("format") == "pcm16" and len(item["bytes"]) % 2 == 0,
                "streaming audio is not complete PCM16 samples")
        require(is_int(descriptor.get("stream_index")) and descriptor["stream_index"] >= 0,
                "streaming audio lacks a non-negative stream_index")
        require(is_int(descriptor.get("sample_rate", 24000))
                and descriptor.get("sample_rate", 24000) > 0
                and is_int(descriptor.get("channels", 1))
                and descriptor.get("channels", 1) > 0,
                "streaming audio lacks valid sample format metadata")
    first_audio_time = min(item["first_time"] for item in audio)
    require(first_audio_time < terminals[0]["time"],
            "streaming audio arrived only at the terminal")

    by_slot = {}
    for item in visual:
        descriptor = item["descriptor"]
        slot_id = descriptor["slot_id"]
        require(slot_id not in by_slot, f"multiple visual artifacts for {slot_id}")
        matches = [slot for slot in slots if slot["event"]["slot_id"] == slot_id]
        require(len(matches) == 1, f"artifact binds to an unknown slot: {slot_id}")
        slot = matches[0]["event"]
        require(descriptor["slot_index"] == slot["slot_index"]
                and descriptor["media_type"] == slot["media_type"]
                and item["stage"] == slot["stage"],
                f"artifact does not match its visual slot: {slot_id}")
        require(item["time"] >= matches[0]["time"],
                f"artifact predates its visual slot: {slot_id}")
        by_slot[slot_id] = item
    require(len(by_slot) == len(slots), "not every visual slot produced an artifact")
    pcm_chunk_times = [
        item["time"] for item in observations
        if item["event"].get("kind") == "artifact"
        and item["event"].get("artifact", {}).get("mime_type") == "audio/pcm"
    ]
    require(any(slot["time"] <= chunk_time < by_slot[slot["event"]["slot_id"]]["time"]
                for slot in slots for chunk_time in pcm_chunk_times),
            "no audio/pcm chunk arrived while a visual slot was pending")

    legs = terminals[0]["event"].get("legs")
    require(isinstance(legs, list), "complete event lacks legs[]")
    for slot in slots:
        event = slot["event"]
        matching = [leg for leg in legs if leg.get("slot_id") == event["slot_id"]]
        require(len(matching) == 1 and matching[0].get("stage") == event["stage"]
                and matching[0].get("state") == "completed",
                f"terminal leg did not complete for {event['slot_id']}: {legs}")
    require(any(leg.get("stage") == "tts" and leg.get("state") == "completed"
                for leg in legs), f"terminal tts leg did not complete: {legs}")

    if mock:
        kinds = [item["event"]["kind"] for item in observations]
        require(kinds == ["vlm_delta", "visual_slot", "artifact", "vlm_delta",
                          "visual_slot", "artifact", "artifact", "complete"],
                f"mock event order changed: {kinds}")
        require([item["event"]["media_type"] for item in slots] == ["image", "video"],
                "mock must exercise one image slot and one video slot")
        require(slots[0]["time"] < audio[0]["first_time"] < visual[0]["time"],
                "streaming PCM did not overlap pending visual slots")
        require(visual[0]["bytes"].startswith(b"\x89PNG\r\n\x1a\n")
                and visual[1]["bytes"][4:8] == b"ftyp",
                "mock visual artifact signatures are invalid")

    slot_reports = []
    for slot in slots:
        event = slot["event"]
        item = by_slot[event["slot_id"]]
        slot_reports.append({
            "slot_id": event["slot_id"], "slot_index": event["slot_index"],
            "media_type": event["media_type"], "prompt": event["prompt"],
            "slot_arrival_s": round(slot["time"], 6),
            "artifact_arrival_s": round(item["time"], 6),
            "render_latency_s": round(item["time"] - slot["time"], 6),
            "bytes": len(item["bytes"]), "sha256": item["descriptor"]["sha256"],
        })
    report = {
        "event_count": len(observations),
        "total_wall_s": round(stream["total_s"], 6),
        "ttft_s": round(deltas[0]["time"], 6),
        "ttfa_s": round(first_audio_time, 6),
        "first_visual_slot_s": round(slots[0]["time"], 6),
        "vlm_finish_s": round(deltas[-1]["time"], 6),
        "first_slot_minus_vlm_finish_s": round(slots[0]["time"] - deltas[-1]["time"], 6),
        "clean_prose": True,
        "prose": prose,
        "slots": slot_reports,
        "audio": {
            "artifact_count": len(audio),
            "chunk_count": sum(item["descriptor"]["total"] for item in audio),
            "bytes": sum(len(item["bytes"]) for item in audio),
            "sample_rates": sorted({item["descriptor"].get("sample_rate", 24000)
                                    for item in audio}),
            "channels": sorted({item["descriptor"].get("channels", 1) for item in audio}),
        },
        "legs": legs,
    }
    return report, visual, audio


def check_multichunk_and_timing(stream):
    multi = copy.deepcopy(stream)
    pcm_index = next(index for index, item in enumerate(multi["observations"])
                     if item["event"].get("artifact", {}).get("mime_type") == "audio/pcm")
    first = multi["observations"][pcm_index]
    descriptor = first["event"]["artifact"]
    raw = base64.b64decode(descriptor["data"], validate=True)
    require(descriptor["total"] == 1 and len(raw) >= 2,
            "self-check PCM fixture must be a non-empty one-chunk artifact")
    original_sha256 = descriptor["sha256"]
    split = len(raw) // 2
    first_time = first["time"]
    descriptor.update({
        "index": 0, "total": 2,
        "data": base64.b64encode(raw[:split]).decode("ascii"),
    })
    descriptor.pop("bytes")
    descriptor.pop("sha256")
    second = copy.deepcopy(first)
    second["time"] = first_time + 0.0001
    second["event"]["seq"] += 1
    second["event"]["artifact"].update({
        "index": 1, "total": 2,
        "data": base64.b64encode(raw[split:]).decode("ascii"),
        "bytes": len(raw), "sha256": original_sha256,
    })
    multi["observations"].insert(pcm_index + 1, second)
    for item in multi["observations"][pcm_index + 2:]:
        item["event"]["seq"] += 1
    report, _, audio = validate_stream(multi)
    require(report["ttfa_s"] == round(first_time, 6),
            "TTFA used the final PCM chunk instead of the first")
    require(report["audio"]["artifact_count"] == 1
            and report["audio"]["chunk_count"] == 2
            and len(audio) == 1 and audio[0]["bytes"] == raw,
            "multi-chunk PCM was not reassembled and counted correctly")

    bad_total = copy.deepcopy(multi["observations"])
    bad_total[pcm_index + 1]["event"]["artifact"]["total"] = 3
    try:
        assemble_artifacts(bad_total)
    except AssertionError as error:
        require("total changed" in str(error), f"wrong inconsistent-total error: {error}")
    else:
        raise AssertionError("inconsistent artifact total was accepted")

    bad_sha = copy.deepcopy(multi["observations"])
    bad_sha[pcm_index + 1]["event"]["artifact"]["sha256"] = "0" * 64
    try:
        assemble_artifacts(bad_sha)
    except AssertionError as error:
        require("SHA-256" in str(error), f"wrong SHA-256 error: {error}")
    else:
        raise AssertionError("a corrupt artifact SHA-256 was accepted")

    late = copy.deepcopy(stream)
    latest_visual = max(
        item["time"] for item in late["observations"]
        if item["event"].get("artifact", {}).get("media_type") in ("image", "video"))
    for item in late["observations"]:
        if item["event"].get("artifact", {}).get("mime_type") == "audio/pcm":
            item["time"] = latest_visual + 0.5
        elif item["event"].get("kind") in TERMINAL_KINDS:
            item["time"] = latest_visual + 1.0
    try:
        validate_stream(late)
    except AssertionError as error:
        require("visual slot was pending" in str(error),
                f"wrong late-audio error: {error}")
    else:
        raise AssertionError("PCM delivered after all visual slots was accepted")


def visual_render(base_url, body, timeout=900.0):
    return http_request(base_url.rstrip("/") + "/v1/visual/render", body, timeout)


def check_render_response(response, kind, prompt):
    status, headers, raw = response
    require(status == 200, f"visual render HTTP {status}: {raw[:1000]!r}")
    expected_type = "image/png" if kind == "image" else "video/mp4"
    require(headers.get_content_type() == expected_type,
            f"visual render Content-Type is not {expected_type}")
    require(headers.get("X-UmmInfer-Kind") == kind,
            "visual render X-UmmInfer-Kind header is wrong")
    if kind == "image":
        require(raw.startswith(b"\x89PNG\r\n\x1a\n"), "rendered image lacks a PNG header")
        return {"kind": kind, "prompt": prompt, "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest()}
    require(len(raw) >= 12 and raw[4:8] == b"ftyp", "rendered video lacks an ftyp header")
    return {"kind": kind, "prompt": prompt, "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def wait_for_idle(base_url):
    health = None
    for _ in range(100):
        status, _, raw = http_request(base_url.rstrip("/") + "/health", timeout=10.0)
        try:
            health = json.loads(raw)
        except json.JSONDecodeError:
            health = {"raw": raw.decode("utf-8", errors="replace")[:500]}
        if status == 200 and health.get("in_flight") == 0:
            return health
        time.sleep(0.1)
    raise AssertionError(f"/health in_flight did not return to zero: {health}")


def self_check():
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    server.chat_requests = []
    server.render_requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        stream = stream_chat(base_url, narrative_request("narrative-mock"), timeout=10.0)
        report, visual, audio = validate_stream(stream, mock=True)
        check_multichunk_and_timing(stream)
        require(len(server.chat_requests) == 1, "mock did not receive one chat request")
        sent = server.chat_requests[0]
        require(sent["narrative"] == {"enabled": True, "visual_mode": "auto", "max_slots": 2}
                and sent["modalities"] == ["text", "image", "audio", "video"]
                and sent["audio"]["stream"] is True and sent["max_tokens"] == 256,
                "narrative request fields changed")

        slot_prompts = [slot["prompt"] for slot in report["slots"]]
        swapped_video = visual_render(base_url, {
            "kind": "video", "prompt": slot_prompts[0], "size": "480x320",
            "steps": 4, "num_frames": 13, "fps": 16, "seconds": 1,
            "guidance_scale": 5.0, "seed": 42,
        }, timeout=10.0)
        video_report = check_render_response(swapped_video, "video", slot_prompts[0])
        require(swapped_video[2][4:8] == b"ftyp", "slot-0 PNG -> MP4 swap failed")
        swapped_image = visual_render(base_url, {
            "kind": "image", "prompt": slot_prompts[1], "size": "512x512",
            "steps": 4, "seed": 43,
        }, timeout=10.0)
        image_report = check_render_response(swapped_image, "image", slot_prompts[1])
        require(swapped_image[2].startswith(b"\x89PNG\r\n\x1a\n"),
                "slot-1 MP4 -> PNG swap failed")
        require([(item["kind"], item["prompt"]) for item in server.render_requests]
                == [("video", slot_prompts[0]), ("image", slot_prompts[1])],
                "visual swaps did not reuse the selected slot prompts")

        bad_requests = [
            {"kind": "image", "prompt": ""},
            {"kind": "audio", "prompt": "invalid kind"},
            {"kind": "image", "prompt": "unknown field", "unknown": True},
            {"kind": "video", "prompt": "bad frames", "num_frames": 0},
            {"kind": "image", "prompt": "video option", "fps": 16},
            {"kind": "video", "prompt": "image option", "image": "data:image/png;base64,AA=="},
            {"kind": "image", "prompt": "bad seed", "seed": 1 << 63},
            {"kind": "image", "prompt": "bad seed", "seed": -(1 << 63) - 1},
        ]
        for body in bad_requests:
            status, _, _ = visual_render(base_url, body, timeout=10.0)
            require(status == 400, f"invalid render request was not rejected: {body}")
        status, _, _ = http_request(base_url + "/v1/visual/render", b"{", timeout=10.0)
        require(status == 400, "malformed render JSON was not rejected with 400")
        require(wait_for_idle(base_url) == {"status": "ok", "in_flight": 0},
                "mock health contract changed")
        print("SELF-CHECK PASS: seq, terminal, [DONE], clean prose, interleaving, "
              "streaming PCM, slot binding, bytes, SHA-256, PNG/MP4 swaps, HTTP 400")
        print(json.dumps({"metrics": report, "swaps": [video_report, image_report],
                          "verified_artifacts": len(visual) + len(audio)}, indent=2))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


def ffprobe_h264(path):
    output = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name",
        "-of", "json", str(path),
    ], text=True, timeout=30.0)
    probe = json.loads(output)
    require(any(stream.get("codec_type") == "video" and stream.get("codec_name") == "h264"
                for stream in probe.get("streams", [])),
            f"ffprobe found no H.264 video stream: {probe}")
    return probe


def live(args):
    temporary = None
    if args.out_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="narrative-live-")
        work_dir = Path(temporary.name)
    else:
        work_dir = args.out_dir
        work_dir.mkdir(parents=True, exist_ok=True)
    try:
        stream = stream_chat(args.url, narrative_request(args.model), timeout=900.0)
        report, visual, audio = validate_stream(stream)
        if args.out_dir is not None:
            for item in visual:
                descriptor = item["descriptor"]
                extension = "png" if descriptor["media_type"] == "image" else "mp4"
                (work_dir / f"narrative_{descriptor['slot_id']}.{extension}").write_bytes(
                    item["bytes"])
            (work_dir / "narrative_audio.pcm").write_bytes(
                b"".join(item["bytes"] for item in audio))

        selected = report["slots"][0]
        target = "video" if selected["media_type"] == "image" else "image"
        body = {"kind": target, "prompt": selected["prompt"]}
        if target == "video":
            body.update({"size": "480x320", "steps": 4, "num_frames": 13,
                         "fps": 16, "seconds": 1})
        else:
            body.update({"size": "512x512", "steps": 4})
        response = visual_render(args.url, body)
        swap = check_render_response(response, target, selected["prompt"])
        raw = response[2]
        if target == "video":
            swap_path = work_dir / "narrative_swap.mp4"
            swap_path.write_bytes(raw)
            swap["ffprobe"] = ffprobe_h264(swap_path)
            if args.out_dir is None:
                swap_path.unlink()
        else:
            swap_path = work_dir / "narrative_swap.png"
            if args.out_dir is not None:
                swap_path.write_bytes(raw)
        swap["source_slot_id"] = selected["slot_id"]
        swap["source_media_type"] = selected["media_type"]
        report["visual_swap"] = swap
        report["health"] = wait_for_idle(args.url)
        report["out_dir"] = str(args.out_dir) if args.out_dir is not None else None
        if args.out_dir is not None:
            (work_dir / "narrative_live_report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        if temporary is not None:
            temporary.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true",
                        help="run the deterministic offline mock-server contract check")
    parser.add_argument("--url", help="live server root URL, for example http://127.0.0.1:43301")
    parser.add_argument("--model", default="qwen3-native")
    parser.add_argument("--out-dir", type=Path, help="save live artifacts, swap and JSON report")
    args = parser.parse_args()
    if args.self_check:
        if args.url is not None or args.out_dir is not None:
            parser.error("--self-check does not accept --url or --out-dir")
        self_check()
    else:
        if not args.url:
            parser.error("--url is required unless --self-check is used")
        live(args)


if __name__ == "__main__":
    main()
