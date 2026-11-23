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

    def reply(self, status, body, kind="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", kind)
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
