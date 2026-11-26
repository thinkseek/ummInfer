#!/usr/bin/env python3
"""Offline end-to-end tests for the `image` and `video` CLI subcommands.

Starts a local stdlib HTTP server that records every request and mocks the
OpenAI media routes, then drives the real `umm-infer` binary:

  image generate   POST /v1/images/generations
  image edit       POST /v1/images/edits (multipart)
  video create     POST /v1/videos
  video status     GET  /v1/videos/{id}
  video content    GET  /v1/videos/{id}/content[?variant=N]
  video delete     DELETE /v1/videos/{id}

Every request is checked for method, path, fields and multipart order. Every
output file is checked for exact bytes. An HTTP error must exit non-zero and
leave no output file behind.

Usage: python3 -I tests/client_media_cli_e2e.py [path/to/umm-infer]
"""

import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN = REPO_ROOT / "zig-out" / "bin" / "umm-infer"
PNG_ONE = b"\x89PNG\r\n\x1a\n" + b"first-image" * 8
PNG_TWO = b"\x89PNG\r\n\x1a\n" + b"second-image" * 8
EDITED = b"\x89PNG\r\n\x1a\n" + b"edited-image" * 8
MP4 = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"video-payload" * 8

RECEIVED = []
LOCK = threading.Lock()


def record(handler, body):
    entry = {
        "method": handler.command,
        "path": handler.path,
        "content_type": handler.headers.get("Content-Type", ""),
        "authorization": handler.headers.get("Authorization", ""),
        "body": body,
    }
    with LOCK:
        RECEIVED.append(entry)
    return entry


def parse_multipart(content_type, body):
    boundary = content_type.split("boundary=", 1)[1].split(";")[0].strip().strip('"')
    separator = b"--" + boundary.encode()
    parts = []
    for raw in body.split(separator):
        if raw.lstrip(b"\r\n") in (b"", b"--") or raw.strip(b"\r\n-") == b"":
            continue
        raw = raw.lstrip(b"\r\n")
        head, _, data = raw.partition(b"\r\n\r\n")
        if data.endswith(b"\r\n"):
            data = data[:-2]
        headers = {}
        for line in head.split(b"\r\n"):
            name, _, value = line.partition(b":")
            if name:
                headers[name.decode().lower().strip()] = value.decode().strip()
        disposition = headers.get("content-disposition", "")
        meta = {}
        for item in disposition.split(";")[1:]:
            key, _, value = item.strip().partition("=")
            meta[key] = value.strip().strip('"')
        parts.append({
            "name": meta.get("name"),
            "filename": meta.get("filename"),
            "type": headers.get("content-type"),
            "data": data,
        })
    return parts


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def reply(self, status, body, kind="application/json", extra_headers=None):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        if extra_headers:
            for name, value in extra_headers.items():
                self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def json(self, status, payload):
        self.reply(status, json.dumps(payload).encode())

    def do_POST(self):
        body = self.read_body()
        entry = record(self, body)
        if self.path == "/v1/images/generations":
            request = json.loads(body)
            if request.get("prompt") == "boom":
                self.json(400, {"error": {"message": "prompt rejected"}})
                return
            count = int(request.get("n", 1))
            images = [PNG_ONE, PNG_TWO][:count]
            self.json(200, {"data": [{"b64_json": base64.b64encode(image).decode()}
                                     for image in images]})
            return
        if self.path == "/v1/images/edits":
            entry["parts"] = parse_multipart(entry["content_type"], body)
            fields = {part["name"]: part["data"].decode() for part in entry["parts"]}
            count = int(fields.get("n", 1))
            images = [PNG_ONE, PNG_TWO][:count] if count > 1 else [EDITED]
            self.json(200, {"data": [{"b64_json": base64.b64encode(image).decode()}
                                     for image in images]})
            return
        if self.path == "/v1/videos":
            self.json(200, {"id": "vid-1", "object": "video", "status": "queued"})
            return
        if self.path == "/v1/audio/speech":
            request = json.loads(body)
            if request.get("input") == "boom":
                self.json(400, {"error": {"message": "speech rejected"}})
                return
            if request.get("response_format") == "pcm":
                self.reply(200, b"\x01\x02\x03\x04", "audio/pcm")
            else:
                self.reply(200, b"RIFF\x24\x00\x00\x00WAVEfmt ", "audio/wav")
            return
        if self.path == "/v1/visual/render":
            request = json.loads(body)
            kind = request.get("kind")
            if request.get("prompt") == "boom":
                self.json(400, {"error": {"message": "render rejected"}})
                return
            if kind == "image":
                self.reply(200, PNG_ONE, "image/png", {"X-UmmInfer-Kind": "image"})
                return
            if kind == "video":
                self.reply(200, MP4, "video/mp4", {"X-UmmInfer-Kind": "video"})
                return
            self.json(400, {"error": {"message": "kind must be image or video"}})
            return
        if self.path == "/v1/chat/completions":
            import hashlib
            req = json.loads(body)
            messages = req.get("messages") or []
            last_content = messages[-1].get("content", "") if messages else ""
            if req.get("stream"):
                chunks = [
                    {
                        "id": "chat-tel-sse",
                        "object": "chat.completion.chunk",
                        "model": req.get("model", "m"),
                        "choices": [{"index": 0, "delta": {"content": "hello "},
                                     "finish_reason": None}],
                    },
                    {
                        "id": "chat-tel-sse",
                        "object": "chat.completion.chunk",
                        "model": req.get("model", "m"),
                        "choices": [{"index": 0, "delta": {"content": "world"},
                                     "finish_reason": None}],
                    },
                ]
                if last_content == "telemetry covered sse":
                    chunks.append({
                        "id": "chat-tel-sse",
                        "object": "chat.completion.chunk",
                        "model": req.get("model", "m"),
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "unified": {
                            "seq": 0,
                            "kind": "complete",
                            "request_id": "chat-tel-sse",
                            "telemetry": {
                                "schema_version": 1,
                                "coverage": "COVERED",
                                "scope": "request",
                                "request_id": "chat-tel-sse",
                                "lifecycle_status": "completed",
                                "stages": [{
                                    "stage": "vlm",
                                    "native_stage": "qwen3_vl",
                                    "backend": "cuda_native",
                                    "coverage": "COVERED",
                                    "allocated_bytes_peak": 134217728,
                                    "reserved_bytes_peak": 268435456,
                                }],
                                "truncated": False,
                            },
                        },
                    })
                elif last_content == "telemetry malformed top-level sse":
                    chunks.append({
                        "id": "chat-tel-sse",
                        "object": "chat.completion.chunk",
                        "model": req.get("model", "m"),
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "telemetry": {
                            "schema_version": 1,
                            "coverage": "COVERED",
                            "scope": "request",
                            "request_id": "chat-tel-sse",
                            "lifecycle_status": "completed",
                            "stages": [{
                                "stage": "vlm",
                                "coverage": "COVERED",
                                "allocated_bytes_peak": False,
                                "reason": "malformed_peak_rejected",
                            }],
                            "truncated": False,
                        },
                    })
                else:
                    chunks.append({
                        "id": "chat-tel-sse",
                        "object": "chat.completion.chunk",
                        "model": req.get("model", "m"),
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    })
                sse = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
                sse += "data: [DONE]\n\n"
                self.reply(200, sse.encode(), "text/event-stream")
                return
            artifacts = []
            mods = set(req.get("modalities", []))
            if "image" in mods:
                artifacts.append({
                    "artifact_id": "art-img", "media_type": "image",
                    "mime_type": "image/png", "encoding": "base64",
                    "bytes": len(EDITED),
                    "sha256": hashlib.sha256(EDITED).hexdigest(),
                    "data": base64.b64encode(EDITED).decode(),
                })
            if "audio" in mods:
                wav = b"RIFF\x24\x00\x00\x00WAVEfmt "
                artifacts.append({
                    "artifact_id": "art-aud", "media_type": "audio",
                    "mime_type": "audio/wav", "encoding": "base64",
                    "bytes": len(wav),
                    "sha256": hashlib.sha256(wav).hexdigest(),
                    "data": base64.b64encode(wav).decode(),
                })
            if "video" in mods:
                artifacts.append({
                    "artifact_id": "art-vid", "media_type": "video",
                    "mime_type": "video/mp4", "encoding": "base64",
                    "bytes": len(MP4),
                    "sha256": hashlib.sha256(MP4).hexdigest(),
                    "data": base64.b64encode(MP4).decode(),
                })
            payload = {
                "id": "chat-w9", "object": "chat.completion", "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "done"}}],
                "artifacts": artifacts,
            }
            if last_content == "telemetry not covered":
                payload["telemetry"] = {
                    "schema_version": 1,
                    "coverage": "NOT_COVERED",
                    "scope": "request",
                    "request_id": "chat-w9",
                    "lifecycle_status": "completed",
                    "stages": [{
                        "stage": "vlm",
                        "native_stage": "qwen3_vl",
                        "coverage": "NOT_COVERED",
                        "allocated_bytes_peak": None,
                        "reason": "batched_request_attribution_unavailable",
                    }],
                    "truncated": False,
                }
            self.json(200, payload)
            return
        self.json(404, {"error": "not found"})

    def do_GET(self):
        record(self, b"")
        if self.path == "/v1/videos/vid-1":
            self.json(200, {"id": "vid-1", "object": "video", "status": "completed"})
            return
        if self.path in ("/v1/videos/vid-1/content", "/v1/videos/vid-1/content?variant=1"):
            self.reply(200, MP4, "video/mp4")
            return
        self.json(404, {"error": "not found"})

    def do_DELETE(self):
        record(self, self.read_body())
        if self.path == "/v1/videos/vid-1":
            self.json(200, {"id": "vid-1", "deleted": True})
            return
        self.json(404, {"error": "not found"})


def start_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def take():
    with LOCK:
        entries = list(RECEIVED)
        RECEIVED.clear()
    return entries


def run_cli(binary, args):
    return subprocess.run([str(binary), *args], capture_output=True, text=True,
                          timeout=120)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def test_image_generate(binary, base, workdir):
    out = workdir / "gen.png"
    result = run_cli(binary, [
        "image", "generate", "--url", base, "--model", "img-model",
        "--prompt", "a cat", "--out", str(out), "--n", "2", "--size", "512x512",
        "--field", "seed=7", "--field", 'extra={"a":1}',
    ])
    require(result.returncode == 0, f"generate failed: {result.stderr}")
    [entry] = take()
    require(entry["method"] == "POST", f"generate method: {entry['method']}")
    require(entry["path"] == "/v1/images/generations", f"generate path: {entry['path']}")
    request = json.loads(entry["body"])
    require(request["model"] == "img-model", "generate must send the model")
    require(request["prompt"] == "a cat", "generate must send the prompt")
    require(request["response_format"] == "b64_json", "generate must force b64_json")
    require(request["n"] == 2, f"generate n: {request.get('n')}")
    require(request["size"] == "512x512", f"generate size: {request.get('size')}")
    require(request["seed"] == 7, f"a JSON field must keep its number type: {request.get('seed')}")
    require(request["extra"] == {"a": 1}, "a JSON object field must be decoded")
    require(out.read_bytes() == PNG_ONE, "the first image must land at --out")
    second = workdir / "gen-2.png"
    require(second.exists(), "a second image must use the -2 suffix")
    require(second.read_bytes() == PNG_TWO, "the second image bytes must match")
    lines = result.stdout.strip().splitlines()
    require(lines == [str(out), str(second)], f"generate must print both paths: {lines}")


def test_image_edit(binary, base, workdir):
    first = workdir / "one.png"
    second = workdir / "two.jpg"
    first.write_bytes(b"first-input")
    second.write_bytes(b"second-input")
    out = workdir / "edit.png"
    result = run_cli(binary, [
        "image", "edit", "--url", base, "--model", "img-model",
        "--prompt", "make it warm", "--out", str(out),
        "--image", str(first), "--image", str(second),
        "--field", "strength=0.5", "--size", "256x256", "--n", "2",
    ])
    require(result.returncode == 0, f"edit failed: {result.stderr}")
    [entry] = take()
    require(entry["path"] == "/v1/images/edits", f"edit path: {entry['path']}")
    parts = entry["parts"]
    names = [part["name"] for part in parts]
    require(names.count("image") == 2, f"edit must send two image parts: {names}")
    images = [part for part in parts if part["name"] == "image"]
    require(images[0]["data"] == b"first-input" and images[1]["data"] == b"second-input",
            "image parts must keep command-line order")
    require(images[0]["filename"] == "one.png", f"image filename: {images[0]['filename']}")
    require(images[0]["type"] == "image/png", f"image type: {images[0]['type']}")
    require(images[1]["type"] == "image/jpeg", f"jpeg type: {images[1]['type']}")
    fields = {part["name"]: part["data"].decode() for part in parts if part["name"] != "image"}
    require(fields["prompt"] == "make it warm", "edit prompt must be a field")
    require(fields["model"] == "img-model", "edit model must be a field")
    require(fields["response_format"] == "b64_json", "edit must force b64_json")
    require(fields["strength"] == "0.5", f"edit field: {fields.get('strength')}")
    require(fields["size"] == "256x256", "--size must be forwarded as a field")
    require(fields["n"] == "2", "--n must be forwarded as a field")
    require(out.read_bytes() == PNG_ONE, "the first edited image must land at --out")
    require((workdir / "edit-2.png").read_bytes() == PNG_TWO,
            "the second edited image must use the -2 suffix")


def test_image_http_error(binary, base, workdir):
    out = workdir / "err.png"
    result = run_cli(binary, [
        "image", "generate", "--url", base, "--model", "img-model",
        "--prompt", "boom", "--out", str(out),
    ])
    require(result.returncode != 0, "an HTTP error must exit non-zero")
    require("400" in result.stderr, f"the status must be reported: {result.stderr}")
    require(not out.exists(), "a failed generation must leave no output file")
    take()


def test_video_lifecycle(binary, base, workdir):
    created = run_cli(binary, [
        "video", "create", "--url", base, "--model", "vid-model",
        "--prompt", "a river", "--field", "seconds=4",
    ])
    require(created.returncode == 0, f"create failed: {created.stderr}")
    require(json.loads(created.stdout)["id"] == "vid-1", "create must print the job")
    [entry] = take()
    require(entry["method"] == "POST" and entry["path"] == "/v1/videos",
            f"create request: {entry['method']} {entry['path']}")
    request = json.loads(entry["body"])
    require(request["model"] == "vid-model" and request["prompt"] == "a river",
            f"create body: {request}")
    require(request["seconds"] == 4, f"create field: {request.get('seconds')}")

    status = run_cli(binary, ["video", "status", "--url", base, "--id", "vid-1"])
    require(status.returncode == 0, f"status failed: {status.stderr}")
    require(json.loads(status.stdout)["status"] == "completed", "status must print the job")
    [entry] = take()
    require(entry["method"] == "GET" and entry["path"] == "/v1/videos/vid-1",
            f"status request: {entry['method']} {entry['path']}")

    out = workdir / "clip.mp4"
    content = run_cli(binary, [
        "video", "content", "--url", base, "--id", "vid-1", "--out", str(out),
    ])
    require(content.returncode == 0, f"content failed: {content.stderr}")
    require(out.read_bytes() == MP4, "content bytes must be written exactly")
    require(content.stdout.strip() == str(out), "content must print the output path")
    [entry] = take()
    require(entry["path"] == "/v1/videos/vid-1/content", f"content path: {entry['path']}")

    variant_out = workdir / "clip-1.mp4"
    variant = run_cli(binary, [
        "video", "content", "--url", base, "--id", "vid-1",
        "--out", str(variant_out), "--variant", "1",
    ])
    require(variant.returncode == 0, f"variant content failed: {variant.stderr}")
    [entry] = take()
    require(entry["path"] == "/v1/videos/vid-1/content?variant=1",
            f"--variant must select the zero-based output index: {entry['path']}")

    removed = run_cli(binary, ["video", "delete", "--url", base, "--id", "vid-1"])
    require(removed.returncode == 0, f"delete failed: {removed.stderr}")
    require(json.loads(removed.stdout)["deleted"] is True, "delete must print the result")
    [entry] = take()
    require(entry["method"] == "DELETE" and entry["path"] == "/v1/videos/vid-1",
            f"delete request: {entry['method']} {entry['path']}")


def test_video_rejects_a_bad_id(binary, base, _workdir):
    result = run_cli(binary, ["video", "status", "--url", base, "--id", "bad id"])
    require(result.returncode != 0, "an invalid id must exit non-zero")
    require("invalid video id" in result.stderr, f"reason: {result.stderr}")
    require(not take(), "an invalid id must not reach the server")


def test_speech_subcommand(binary, base, workdir):
    ref = workdir / "ref.wav"
    ref.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
    wav_out = workdir / "out.wav"
    res = run_cli(binary, [
        "speech", "--url", base, "--model", "tts-m", "--text", "hello world",
        "--voice", "clone", "--ref-audio", str(ref), "--out", str(wav_out),
    ])
    require(res.returncode == 0, f"speech wav failed: {res.stderr}")
    [entry] = take()
    require(entry["path"] == "/v1/audio/speech", f"speech path: {entry['path']}")
    req = json.loads(entry["body"])
    require(req["model"] == "tts-m" and req["input"] == "hello world", f"body: {req}")
    require(req["voice"] == "clone" and req["response_format"] == "wav", f"body: {req}")
    require(base64.b64decode(req["ref_audio"]) == ref.read_bytes(), "ref_audio mismatch")
    require(wav_out.read_bytes().startswith(b"RIFF"), "wav bytes not written")

    pcm_out = workdir / "out.pcm"
    res_pcm = run_cli(binary, [
        "speech", "--url", base, "--model", "tts-m", "--text", "hi",
        "--format", "pcm", "--out", str(pcm_out),
    ])
    require(res_pcm.returncode == 0, f"speech pcm failed: {res_pcm.stderr}")
    [entry_pcm] = take()
    require(json.loads(entry_pcm["body"])["response_format"] == "pcm", "pcm format")
    require(pcm_out.read_bytes() == b"\x01\x02\x03\x04", "pcm bytes not written")

    bad_out = workdir / "bad.wav"
    bad = run_cli(binary, [
        "speech", "--url", base, "--model", "tts-m", "--text", "boom",
        "--out", str(bad_out),
    ])
    require(bad.returncode != 0 and not bad_out.exists(), "failed speech must not write file")
    take()


def test_chat_wave9_multimodal_flags(binary, base, workdir):
    img_ref = workdir / "src.png"
    img_ref.write_bytes(PNG_ONE)
    vid_ref = workdir / "frame.png"
    vid_ref.write_bytes(PNG_ONE)
    aud_ref = workdir / "voice.wav"
    aud_ref.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
    img_out = workdir / "chat-edit.png"
    aud_out = workdir / "chat-clone.wav"
    vid_out = workdir / "chat-vid.mp4"

    res = run_cli(binary, [
        "chat", "--url", base, "--model", "qwen3-native", "--prompt", "remix",
        "--temperature", "0.7", "--top-p", "0.9", "--seed", "-42",
        "--image-out", str(img_out), "--image-prompt", "lantern",
        "--image-size", "1024x1024", "--image-steps", "4",
        "--image-ref", str(img_ref), "--image-seed", "7",
        "--audio-out", str(aud_out), "--voice", "clone", "--ref-audio", str(aud_ref),
        "--video-out", str(vid_out), "--video-prompt", "waves",
        "--video-size", "480x320", "--video-fps", "16", "--video-frames", "13",
        "--video-steps", "15", "--video-guidance", "5.0", "--video-seed", "9",
        "--video-image", str(vid_ref),
    ])
    require(res.returncode == 0, f"chat wave9 failed: {res.stderr}")
    [entry] = take()
    req = json.loads(entry["body"])
    require(req["seed"] == -42 and req["top_p"] == 0.9, f"sampling: {req}")
    require(set(req["modalities"]) == {"text", "image", "audio", "video"}, f"mods: {req['modalities']}")
    require(req["image"]["steps"] == 4, f"image.steps: {req['image']}")
    require(req["image"]["image"].startswith("data:image/png;base64,"), "image.image data URL")
    require(base64.b64decode(req["image"]["image"].split(",", 1)[1]) == PNG_ONE, "image.image bytes")
    require(base64.b64decode(req["audio"]["ref_audio"]) == aud_ref.read_bytes(), "audio.ref_audio bytes")
    require(req["video"]["num_frames"] == 13 and req["video"]["steps"] == 15 and
            req["video"]["guidance_scale"] == 5.0 and req["video"]["seed"] == 9,
            f"video opts: {req['video']}")
    require(req["video"]["image"].startswith("data:image/png;base64,"), "video.image data URL")
    require(base64.b64decode(req["video"]["image"].split(",", 1)[1]) == PNG_ONE,
            "video.image bytes")
    require(img_out.read_bytes() == EDITED, "chat --image-out mismatch")
    require(aud_out.read_bytes().startswith(b"RIFF"), "chat --audio-out mismatch")
    require(vid_out.read_bytes() == MP4, "chat --video-out mismatch")


def test_chat_video_input(binary, base, workdir):
    clip = workdir / "input.mp4"
    clip.write_bytes(MP4)
    result = run_cli(binary, [
        "chat", "--url", base, "--model", "qwen3-native",
        "--prompt", "describe the motion", "--video-input", str(clip),
    ])
    require(result.returncode == 0, f"chat --video-input failed: {result.stderr}")
    [entry] = take()
    request = json.loads(entry["body"])
    parts = request["messages"][0]["content"]
    require([part["type"] for part in parts] == ["text", "video_url"],
            f"video input content parts: {parts}")
    url = parts[1]["video_url"]["url"]
    require(url.startswith("data:video/mp4;base64,"), f"video data URL: {url[:80]}")
    require(base64.b64decode(url.split(",", 1)[1]) == MP4,
            "--video-input must preserve the exact file bytes")
    require("frame_selection" not in parts[1]["video_url"],
            "default video input must not acquire a frame_selection field")

    temporal = run_cli(binary, [
        "chat", "--url", base, "--model", "qwen3-native",
        "--prompt", "find the scene change", "--video-input", str(clip),
        "--video-frame-policy", "temporal", "--video-motion-threshold", "0.04",
        "--video-scene-threshold", "0.3",
    ])
    require(temporal.returncode == 0, f"temporal video input failed: {temporal.stderr}")
    [temporal_entry] = take()
    temporal_parts = json.loads(temporal_entry["body"])["messages"][0]["content"]
    selection = temporal_parts[1]["video_url"]["frame_selection"]
    require(selection == {"policy": "temporal", "motion_threshold": 0.04,
                          "scene_threshold": 0.3}, f"temporal selection: {selection}")

    uniform = run_cli(binary, [
        "chat", "--url", base, "--model", "qwen3-native",
        "--prompt", "uniform", "--video-input", str(clip),
        "--video-frame-policy", "uniform",
    ])
    require(uniform.returncode == 0, f"explicit uniform failed: {uniform.stderr}")
    [uniform_entry] = take()
    uniform_parts = json.loads(uniform_entry["body"])["messages"][0]["content"]
    require(uniform_parts[1]["video_url"]["frame_selection"] == {"policy": "uniform"},
            "explicit uniform must be the only selection field")

    invalid_threshold = run_cli(binary, [
        "chat", "--url", base, "--model", "qwen3-native", "--prompt", "bad",
        "--video-input", str(clip), "--video-motion-threshold", "0.2",
    ])
    require(invalid_threshold.returncode == 2 and
            "require --video-frame-policy temporal" in invalid_threshold.stderr,
            f"threshold without temporal opt-in: {invalid_threshold.stderr}")
    require(not take(), "invalid temporal options must be rejected before HTTP")

    rejected = run_cli(binary, [
        "chat", "--url", base, "--model", "qwen3-native", "--prompt", "mixed",
        "--video-input", str(clip), "--image-url", "https://example.com/frame.png",
    ])
    require(rejected.returncode == 2 and "mutually exclusive" in rejected.stderr,
            f"image/video input conflict: {rejected.returncode} {rejected.stderr}")
    require(not take(), "image/video conflict must be rejected before HTTP")

    remote_mix = run_cli(binary, [
        "chat", "--url", base, "--model", "qwen3-native", "--prompt", "mixed",
        "--image-url", "https://example.com/frame.png",
        "--video-url", "https://example.com/clip.mp4",
    ])
    require(remote_mix.returncode != 0 and "cannot combine" in remote_mix.stderr,
            f"library image/video input conflict: {remote_mix.stderr}")
    require(not take(), "library image/video conflict must be rejected before HTTP")

    oversized = workdir / "oversized.mp4"
    oversized.write_bytes(b"V" * (32 * 1024 * 1024 + 1))
    too_large = run_cli(binary, [
        "chat", "--url", base, "--model", "qwen3-native", "--prompt", "large",
        "--video-input", str(oversized),
    ])
    require(too_large.returncode != 0 and "too large" in too_large.stderr,
            f"32 MiB cap: {too_large.stderr}")
    require(not take(), "oversized video must be rejected before HTTP")

    empty = workdir / "empty.mp4"
    empty.write_bytes(b"")
    empty_result = run_cli(binary, [
        "chat", "--url", base, "--model", "qwen3-native", "--prompt", "empty",
        "--video-input", str(empty),
    ])
    require(empty_result.returncode != 0 and "empty" in empty_result.stderr,
            f"empty video: {empty_result.stderr}")
    require(not take(), "empty video must be rejected before HTTP")

    wrong_extension = workdir / "clip.avi"
    wrong_extension.write_bytes(MP4)
    wrong_type = run_cli(binary, [
        "chat", "--url", base, "--model", "qwen3-native", "--prompt", "type",
        "--video-input", str(wrong_extension),
    ])
    require(wrong_type.returncode == 2 and ".mp4, .webm, or .mov" in wrong_type.stderr,
            f"video extension check: {wrong_type.stderr}")
    require(not take(), "unsupported video type must be rejected before HTTP")


def test_chat_native_allocator_telemetry(binary, base, _workdir):
    # (a) Default (without --native-allocator-telemetry): omit telemetry from request and stderr.
    default_res = run_cli(binary, [
        "chat", "--url", base, "--model", "qwen3-native",
        "--prompt", "telemetry default off",
    ])
    require(default_res.returncode == 0, f"default chat failed: {default_res.stderr}")
    [default_entry] = take()
    default_req = json.loads(default_entry["body"])
    require("telemetry" not in default_req,
            f"default chat must omit telemetry key: {default_req}")
    require(default_res.stdout == "done\n", f"default stdout: {default_res.stdout!r}")
    require("[telemetry]" not in default_res.stderr and default_res.stderr == "",
            f"default chat must emit no telemetry stderr: {default_res.stderr!r}")

    # (b) Opt-in on SSE stream with unified.telemetry COVERED stage (128.0 MiB).
    covered_res = run_cli(binary, [
        "chat", "--url", base, "--model", "qwen3-native",
        "--prompt", "telemetry covered sse", "--stream",
        "--native-allocator-telemetry",
    ])
    require(covered_res.returncode == 0, f"covered SSE chat failed: {covered_res.stderr}")
    [covered_entry] = take()
    covered_req = json.loads(covered_entry["body"])
    require(covered_req.get("stream") is True, f"expected stream=true: {covered_req}")
    require(covered_req.get("telemetry") == {"native_allocator": True},
            f"opt-in must send telemetry.native_allocator=true: {covered_req}")
    require(covered_res.stdout == "hello world\n",
            f"SSE stdout must stay clean prose: {covered_res.stdout!r}")
    require("[telemetry] VRAM: 128.0 MiB peak (covered=1/1)" in covered_res.stderr,
            f"expected covered VRAM summary on stderr: {covered_res.stderr!r}")

    # (c) Opt-in on non-stream JSON with NOT_COVERED (allocated_bytes_peak: null).
    nc_res = run_cli(binary, [
        "chat", "--url", base, "--model", "qwen3-native",
        "--prompt", "telemetry not covered",
        "--native-allocator-telemetry",
    ])
    require(nc_res.returncode == 0, f"not-covered chat failed: {nc_res.stderr}")
    [nc_entry] = take()
    nc_req = json.loads(nc_entry["body"])
    require(nc_req.get("telemetry") == {"native_allocator": True},
            f"non-stream opt-in must send telemetry.native_allocator=true: {nc_req}")
    require(nc_res.stdout == "done\n", f"non-stream stdout: {nc_res.stdout!r}")
    require("[telemetry] VRAM: N/C (batched_request_attribution_unavailable)" in nc_res.stderr,
            f"expected honest N/C reason on stderr: {nc_res.stderr!r}")
    require("0 B" not in nc_res.stderr and "0 MiB" not in nc_res.stderr and
            "0.0 MiB" not in nc_res.stderr,
            f"NOT_COVERED must never fabricate zero bytes: {nc_res.stderr!r}")

    # (d) Top-level SSE telemetry with boolean peak must be rejected honestly as N/C.
    malformed_res = run_cli(binary, [
        "chat", "--url", base, "--model", "qwen3-native",
        "--prompt", "telemetry malformed top-level sse", "--stream",
        "--native-allocator-telemetry",
    ])
    require(malformed_res.returncode == 0,
            f"malformed peak SSE chat failed: {malformed_res.stderr}")
    [malformed_entry] = take()
    malformed_req = json.loads(malformed_entry["body"])
    require(malformed_req.get("telemetry") == {"native_allocator": True},
            f"top-level SSE opt-in request: {malformed_req}")
    require(malformed_res.stdout == "hello world\n",
            f"top-level SSE stdout must stay clean prose: {malformed_res.stdout!r}")
    require("[telemetry] VRAM: N/C (malformed_peak_rejected)" in malformed_res.stderr,
            f"boolean peak must render N/C: {malformed_res.stderr!r}")
    require("0 B" not in malformed_res.stderr,
            f"boolean peak must never fabricate 0 B: {malformed_res.stderr!r}")


def test_chat_video_teacache_thresh(binary, base, workdir):
    # (a) Default --video-out without --video-teacache-thresh omits teacache_thresh.
    default_out = workdir / "vid-default.mp4"
    default_res = run_cli(binary, [
        "chat", "--url", base, "--model", "qwen3-native",
        "--prompt", "ocean waves", "--video-out", str(default_out),
    ])
    require(default_res.returncode == 0, f"default video failed: {default_res.stderr}")
    [default_entry] = take()
    default_req = json.loads(default_entry["body"])
    require("teacache_thresh" not in default_req["video"],
            f"default video must omit teacache_thresh: {default_req['video']}")
    require(default_out.read_bytes() == MP4, "default video output bytes mismatch")

    # (b) Valid boundaries 0.0, 0.05, 0.25 forward exact video.teacache_thresh.
    for idx, thresh_str in enumerate(["0.0", "0.05", "0.25"]):
        out_path = workdir / f"vid-fast-{idx}.mp4"
        res = run_cli(binary, [
            "chat", "--url", base, "--model", "qwen3-native",
            "--prompt", "fast ocean waves", "--video-out", str(out_path),
            "--video-teacache-thresh", thresh_str,
        ])
        require(res.returncode == 0,
                f"video teacache {thresh_str} failed: {res.stderr}")
        [entry] = take()
        req = json.loads(entry["body"])
        require(req["video"]["teacache_thresh"] == float(thresh_str),
                f"expected teacache_thresh={thresh_str}: {req['video']}")
        require(out_path.read_bytes() == MP4, "fast video output bytes mismatch")

    # (c) Out-of-range / non-finite / malformed values rejected before HTTP.
    for bad in ["-0.01", "0.26", "nan", "inf", "abc", ""]:
        bad_out = workdir / "vid-bad.mp4"
        res = run_cli(binary, [
            "chat", "--url", base, "--model", "qwen3-native",
            "--prompt", "bad", "--video-out", str(bad_out),
            "--video-teacache-thresh", bad,
        ])
        require(res.returncode == 2,
                f"bad teacache {bad!r} must exit 2: {res.returncode} {res.stderr}")
        require(not bad_out.exists(), "bad teacache must not write output file")
        require(not take(), f"bad teacache {bad!r} must not reach server")

    # (d) --video-teacache-thresh without --video-out or --narrative rejected.
    orphan = run_cli(binary, [
        "chat", "--url", base, "--model", "qwen3-native",
        "--prompt", "orphan", "--video-teacache-thresh", "0.05",
    ])
    require(orphan.returncode == 2 and
            "requires --video-out or --narrative" in orphan.stderr,
            f"orphan teacache: {orphan.returncode} {orphan.stderr}")
    require(not take(), "orphan teacache must not reach server")


def test_chat_narrative_controls(binary, base, _workdir):
    # (a) Default chat omits narrative completely on the wire.
    plain_res = run_cli(binary, [
        "chat", "--url", base, "--model", "qwen3-native", "--prompt", "plain",
    ])
    require(plain_res.returncode == 0, f"plain chat failed: {plain_res.stderr}")
    [plain_entry] = take()
    plain_req = json.loads(plain_entry["body"])
    require("narrative" not in plain_req,
            f"default chat must omit narrative: {plain_req}")

    # (b) --narrative enables streaming, image+video modalities, allow_text_only=true.
    basic_res = run_cli(binary, [
        "chat", "--url", base, "--model", "qwen3-native",
        "--prompt", "tell a story", "--narrative",
    ])
    require(basic_res.returncode == 0, f"basic narrative failed: {basic_res.stderr}")
    [basic_entry] = take()
    basic_req = json.loads(basic_entry["body"])
    require(basic_req.get("stream") is True, f"narrative must stream: {basic_req}")
    require(set(basic_req.get("modalities", [])) == {"text", "image", "video"},
            f"narrative default modalities: {basic_req.get('modalities')}")
    require(basic_req.get("narrative") == {"enabled": True, "allow_text_only": True},
            f"basic narrative wire JSON: {basic_req.get('narrative')}")
    require("video" not in basic_req,
            f"basic narrative without teacache must omit video object: {basic_req}")
    require(basic_res.stdout == "hello world\n",
            f"narrative stdout: {basic_res.stdout!r}")

    # (c) Full narrative flags + --video-teacache-thresh 0.05.
    full_res = run_cli(binary, [
        "chat", "--url", base, "--model", "qwen3-native",
        "--prompt", "storyboard", "--narrative",
        "--narrative-visual-mode", "auto",
        "--narrative-no-allow-text-only",
        "--narrative-continuity", "i2v_chain",
        "--narrative-style-anchor", "  cinematic 35mm film  ",
        "--video-teacache-thresh", "0.05",
    ])
    require(full_res.returncode == 0, f"full narrative failed: {full_res.stderr}")
    [full_entry] = take()
    full_req = json.loads(full_entry["body"])
    require(full_req.get("stream") is True, f"full narrative must stream: {full_req}")
    require(set(full_req.get("modalities", [])) == {"text", "image", "video"},
            f"full narrative modalities: {full_req.get('modalities')}")
    require(full_req.get("narrative") == {
        "enabled": True,
        "visual_mode": "auto",
        "allow_text_only": False,
        "continuity_mode": "i2v_chain",
        "style_anchor": "cinematic 35mm film",
    }, f"full narrative wire JSON: {full_req.get('narrative')}")
    require(full_req.get("video") == {
        "size": "1280x720",
        "output_format": "mp4",
        "teacache_thresh": 0.05,
    }, f"narrative fast video wire JSON: {full_req.get('video')}")

    # (d) visual_mode=image / video narrow modalities + explicit continuity=off.
    img_res = run_cli(binary, [
        "chat", "--url", base, "--model", "qwen3-native",
        "--prompt", "illustrate", "--narrative",
        "--narrative-visual-mode", "image",
        "--narrative-continuity", "off",
        "--narrative-style-anchor", "a" * 256,
    ])
    require(img_res.returncode == 0, f"image narrative failed: {img_res.stderr}")
    [img_entry] = take()
    img_req = json.loads(img_entry["body"])
    require(set(img_req.get("modalities", [])) == {"text", "image"},
            f"image narrative modalities: {img_req.get('modalities')}")
    require(img_req["narrative"]["continuity_mode"] == "off" and
            len(img_req["narrative"]["style_anchor"].encode("utf-8")) == 256,
            f"image narrative fields: {img_req['narrative']}")

    vid_res = run_cli(binary, [
        "chat", "--url", base, "--model", "qwen3-native",
        "--prompt", "animate", "--narrative",
        "--narrative-visual-mode", "video",
        "--video-teacache-thresh", "0.05",
    ])
    require(vid_res.returncode == 0, f"video narrative failed: {vid_res.stderr}")
    [vid_entry] = take()
    vid_req = json.loads(vid_entry["body"])
    require(set(vid_req.get("modalities", [])) == {"text", "video"},
            f"video narrative modalities: {vid_req.get('modalities')}")
    require(vid_req["video"]["teacache_thresh"] == 0.05,
            f"video narrative teacache: {vid_req['video']}")

    # (e) Fail-closed UsageError rejections before HTTP.
    invalid_cases = [
        (["--narrative-visual-mode", "auto"], "require --narrative"),
        (["--narrative-no-allow-text-only"], "require --narrative"),
        (["--narrative-continuity", "i2v_chain"], "require --narrative"),
        (["--narrative-style-anchor", "film"], "require --narrative"),
        (["--narrative", "--narrative-visual-mode", "audio"], "auto, image, or video"),
        (["--narrative", "--narrative-continuity", "bogus"], "off, prompt_anchor, or i2v_chain"),
        (["--narrative", "--narrative-style-anchor", ""], "must not be empty"),
        (["--narrative", "--narrative-style-anchor", "   "], "must not be empty"),
        (["--narrative", "--narrative-style-anchor", "a" * 257], "256 UTF-8 bytes"),
        (["--narrative", "--narrative-style-anchor", "\u4e2d" * 86], "256 UTF-8 bytes"),
        (["--narrative", "--narrative-visual-mode", "image",
          "--video-teacache-thresh", "0.05"], "cannot be used with --narrative-visual-mode image"),
        (["--narrative", "--video-teacache-thresh", "0.3"], "invalid --video-teacache-thresh"),
    ]
    for extra_args, expected_err in invalid_cases:
        res = run_cli(binary, [
            "chat", "--url", base, "--model", "qwen3-native",
            "--prompt", "bad", *extra_args,
        ])
        require(res.returncode == 2 and expected_err in res.stderr,
                f"args={extra_args}: expected exit 2 + {expected_err!r}, got {res.returncode} {res.stderr!r}")
        require(not take(), f"invalid args {extra_args} must not reach server")


def test_visual_render_subcommand(binary, base, workdir):
    # (a) Image render: byte-exact PNG, unset optional fields omitted on wire.
    img_out = workdir / "render-img.png"
    img_res = run_cli(binary, [
        "render", "--url", base, "--kind", "image",
        "--prompt", "a glowing red lantern", "--out", str(img_out),
    ])
    require(img_res.returncode == 0, f"render image failed: {img_res.stderr}")
    [img_entry] = take()
    require(img_entry["method"] == "POST" and img_entry["path"] == "/v1/visual/render",
            f"render image request: {img_entry['method']} {img_entry['path']}")
    img_req = json.loads(img_entry["body"])
    require(img_req == {"kind": "image", "prompt": "a glowing red lantern"},
            f"render image wire JSON must omit unset optionals: {img_req}")
    require(img_out.read_bytes() == PNG_ONE, "render image output bytes mismatch")
    require(img_res.stdout.strip() == str(img_out),
            f"render image stdout: {img_res.stdout!r}")

    # (b) Video I2V render with full flags + teacache_thresh=0.05.
    src_frame = workdir / "render-first-frame.png"
    src_frame.write_bytes(PNG_TWO)
    vid_out = workdir / "render-i2v.mp4"
    vid_res = run_cli(binary, [
        "render", "--url", base, "--kind", "video",
        "--prompt", "the lantern drifts across the lake",
        "--image", str(src_frame),
        "--frames", "29", "--steps", "10", "--guidance", "5.0",
        "--seed", "42", "--teacache-thresh", "0.05",
        "--out", str(vid_out),
    ])
    require(vid_res.returncode == 0, f"render i2v failed: {vid_res.stderr}")
    [vid_entry] = take()
    require(vid_entry["method"] == "POST" and vid_entry["path"] == "/v1/visual/render",
            f"render i2v request: {vid_entry['method']} {vid_entry['path']}")
    vid_req = json.loads(vid_entry["body"])
    expected_image = "data:image/png;base64," + base64.b64encode(PNG_TWO).decode()
    require(vid_req == {
        "kind": "video",
        "prompt": "the lantern drifts across the lake",
        "num_frames": 29,
        "steps": 10,
        "guidance_scale": 5.0,
        "seed": 42,
        "teacache_thresh": 0.05,
        "image": expected_image,
    }, f"render i2v exact wire JSON mismatch: {vid_req}")
    require(vid_out.read_bytes() == MP4, "render i2v output bytes mismatch")
    require(vid_res.stdout.strip() == str(vid_out),
            f"render i2v stdout: {vid_res.stdout!r}")

    # (c) Default video render without --teacache-thresh omits teacache_thresh.
    default_vid_out = workdir / "render-default.mp4"
    default_vid_res = run_cli(binary, [
        "render", "--url", base, "--kind", "video",
        "--prompt", "calm lake at dawn", "--out", str(default_vid_out),
    ])
    require(default_vid_res.returncode == 0,
            f"render default video failed: {default_vid_res.stderr}")
    [default_vid_entry] = take()
    default_vid_req = json.loads(default_vid_entry["body"])
    require(default_vid_req == {"kind": "video", "prompt": "calm lake at dawn"},
            f"default video must omit teacache_thresh and unset optionals: {default_vid_req}")
    require(default_vid_out.read_bytes() == MP4, "render default video bytes mismatch")

    # (d) HTTP error exits non-zero and leaves no output file.
    err_out = workdir / "render-err.png"
    err_res = run_cli(binary, [
        "render", "--url", base, "--kind", "image",
        "--prompt", "boom", "--out", str(err_out),
    ])
    require(err_res.returncode != 0 and "400" in err_res.stderr,
            f"render HTTP error: {err_res.returncode} {err_res.stderr!r}")
    require(not err_out.exists(), "failed render must leave no output file")
    take()

    # (e) Fail-closed exit-2 rejections with zero HTTP requests.
    bad_img = workdir / "bad.bmp"
    bad_img.write_bytes(b"BM\x00\x00")
    unused_out = workdir / "render-never.mp4"
    invalid_cases = [
        (["--url", base, "--kind", "audio", "--prompt", "hi", "--out", str(unused_out)],
         "--kind must be image or video"),
        (["--url", base, "--kind", "image", "--prompt", "hi", "--out", str(unused_out),
          "--teacache-thresh", "0.05"],
         "only valid with --kind video"),
        (["--url", base, "--kind", "video", "--prompt", "hi", "--out", str(unused_out),
          "--teacache-thresh", "0.5"],
         "invalid --teacache-thresh"),
        (["--url", base, "--kind", "video", "--prompt", "hi", "--out", str(unused_out),
          "--teacache-thresh", "-0.01"],
         "invalid --teacache-thresh"),
        (["--url", base, "--kind", "video", "--prompt", "hi", "--out", str(unused_out),
          "--teacache-thresh", "nan"],
         "invalid --teacache-thresh"),
        (["--url", base, "--kind", "video", "--prompt", "hi", "--out", str(unused_out),
          "--teacache-thresh", "inf"],
         "invalid --teacache-thresh"),
        (["--url", base, "--kind", "image", "--prompt", "hi", "--out", str(unused_out),
          "--frames", "29"],
         "only valid with --kind video"),
        (["--url", base, "--kind", "image", "--prompt", "hi", "--out", str(unused_out),
          "--guidance", "5.0"],
         "only valid with --kind video"),
        (["--url", base, "--kind", "image", "--out", str(unused_out)],
         "--prompt is required"),
        (["--url", base, "--kind", "image", "--prompt", "hi"],
         "--out is required"),
        (["--url", base, "--prompt", "hi", "--out", str(unused_out)],
         "--kind is required"),
        (["--url", base, "--kind", "video", "--prompt", "hi", "--out", str(unused_out),
          "--image", str(bad_img)],
         ".png, .jpg, .jpeg, or .webp"),
    ]
    for extra_args, expected_err in invalid_cases:
        res = run_cli(binary, ["render", *extra_args])
        require(res.returncode == 2 and expected_err in res.stderr,
                f"render args={extra_args}: expected exit 2 + {expected_err!r}, "
                f"got {res.returncode} {res.stderr!r}")
        require(not unused_out.exists(), "rejected render must not write output file")
        require(not take(), f"rejected render args {extra_args} must not reach server")


def main():
    binary = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BIN
    if not binary.exists():
        print(f"missing binary: {binary}", file=sys.stderr)
        return 2
    server = start_server()
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    workdir = Path(tempfile.mkdtemp(prefix="umm-cli-media-"))
    tests = [
        test_image_generate,
        test_image_edit,
        test_image_http_error,
        test_video_lifecycle,
        test_video_rejects_a_bad_id,
        test_speech_subcommand,
        test_visual_render_subcommand,
        test_chat_wave9_multimodal_flags,
        test_chat_video_input,
        test_chat_native_allocator_telemetry,
        test_chat_video_teacache_thresh,
        test_chat_narrative_controls,
    ]
    failures = 0
    try:
        for test in tests:
            try:
                test(binary, base, workdir)
            except Exception as error:  # noqa: BLE001 - report every failure
                failures += 1
                print(f"FAIL {test.__name__}: {error}", flush=True)
            else:
                print(f"ok {test.__name__}", flush=True)
    finally:
        server.shutdown()
        server.server_close()
        for path in workdir.glob("*"):
            path.unlink()
        os.rmdir(workdir)
    if failures:
        print(f"{failures} of {len(tests)} tests failed", flush=True)
        return 1
    print(f"{len(tests)} tests passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
