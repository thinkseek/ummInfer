#!/usr/bin/env python3
"""Live smoke for the native stage deployment.

Drives the real HTTP surface of umm-infer serve with native ASR/Omni/Image/TTS
stages and writes every command result under --out-dir. Stdlib only.

  python3 -I native_stage_smoke.py --base-url http://172.17.0.2:43301 \
      --wav asr_en_pcm16.wav --image man_ironing.png --out-dir evidence
"""

import argparse
import array
import base64
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib

EXPECTED_ASR = ("Oh yeah, yeah. He wasn't even that big when I started listening "
                "to him. But in his solo music, didn't do overly well, but he did "
                "very well when he started writing for other people.")
# The TTS leg speaks the aggregated VLM answer, so this prompt asks for one
# short sentence; the round-trip check compares against that answer.
TTS_TEXT = ("Reply with exactly this sentence and nothing else: "
            "Hello, welcome to the native inference engine test.")


class Failure(Exception):
    pass


def record(out_dir, name, payload):
    path = os.path.join(out_dir, name)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(payload if payload.endswith("\n") else payload + "\n")


def request(url, body=None, headers=None, timeout=900):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers or {})
    if data is not None:
        req.add_header("Content-Type", "application/json")
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read(), time.time() - started
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), time.time() - started


def request_bytes(url, data, headers, timeout=900):
    req = urllib.request.Request(url, data=data, headers=headers)
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read(), time.time() - started
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), time.time() - started


def get_json(url, timeout=30):
    status, body, _ = request(url, timeout=timeout)
    return status, json.loads(body.decode("utf-8"))


def sse(url, body, timeout=1800):
    """Yields decoded JSON frames from one chat completion stream."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        pending = []
        for raw in response:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if line.startswith("data:"):
                pending.append(line[5:].lstrip())
                continue
            if line or not pending:
                continue
            payload = "\n".join(pending)
            pending = []
            if payload == "[DONE]":
                return
            yield json.loads(payload)


def wav_bytes(path):
    with open(path, "rb") as handle:
        return handle.read()


def audio_part(path):
    return {
        "type": "input_audio",
        "input_audio": {
            "data": base64.b64encode(wav_bytes(path)).decode("ascii"),
            "format": "wav",
        },
    }


def image_part(path):
    mime = "image/png" if path.endswith(".png") else "image/jpeg"
    encoded = base64.b64encode(wav_bytes(path)).decode("ascii")
    return {"type": "image_url",
            "image_url": {"url": "data:%s;base64,%s" % (mime, encoded)}}


def chat_stream(base, content, extra=None, max_tokens=256):
    body = {"model": "qwen3-native", "stream": True,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}]}
    if extra:
        body.update(extra)
    return sse(base + "/v1/chat/completions", body)


def collect(stream, out_dir, name):
    """Runs one stream to its terminal event and writes a readable trace."""
    request_id = None
    text = ""
    artifacts = []
    terminal = None
    lines = []
    for frame in stream:
        request_id = frame.get("id", request_id)
        choice = (frame.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        if delta.get("content"):
            text += delta["content"]
        unified = frame.get("unified")
        if unified:
            # The trace keeps descriptors, never the inline base64 payload.
            summary = dict(unified)
            if isinstance(summary.get("artifact"), dict):
                artifact = dict(summary["artifact"])
                artifact.pop("data", None)
                summary["artifact"] = artifact
            lines.append(json.dumps(summary, sort_keys=True))
            if unified.get("kind") == "artifact":
                artifacts.append(unified["artifact"])
            if unified.get("kind") in ("complete", "error", "cancelled"):
                terminal = unified
    record(out_dir, name + ".events.jsonl", "\n".join(lines))
    return request_id, text, artifacts, terminal


def decode_artifact(chunks, out_path):
    raw = b"".join(base64.b64decode(chunk["data"]) for chunk in chunks)
    with open(out_path, "wb") as handle:
        handle.write(raw)
    return raw


def png_stats(raw):
    """Validate the stage's non-interlaced, filter-0 RGB PNG artifact."""
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise Failure("image artifact is not a PNG")
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    offset = 8
    compressed = bytearray()
    while offset + 12 <= len(raw):
        size = int.from_bytes(raw[offset:offset + 4], "big")
        kind = raw[offset + 4:offset + 8]
        body = raw[offset + 8:offset + 8 + size]
        if kind == b"IDAT":
            compressed += body
        offset += 12 + size
    scanlines = zlib.decompress(bytes(compressed))
    stride = 1 + width * 3
    if len(scanlines) != stride * height:
        raise Failure("PNG scanline size mismatch")
    pixels = bytearray()
    for row in range(height):
        start = row * stride
        if scanlines[start] != 0:
            raise Failure("unexpected PNG row filter")
        pixels += scanlines[start + 1:start + stride]
    colors = len(set(zip(pixels[0::3], pixels[1::3], pixels[2::3])))
    return {"width": width, "height": height, "colors": colors,
            "png_bytes": len(raw)}


def wav_stats(raw):
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise Failure("artifact is not a RIFF WAVE file")
    offset = 12
    fmt = None
    data = None
    while offset + 8 <= len(raw):
        chunk_id = raw[offset:offset + 4]
        size = int.from_bytes(raw[offset + 4:offset + 8], "little")
        body = raw[offset + 8:offset + 8 + size]
        if chunk_id == b"fmt ":
            fmt = {"format": int.from_bytes(body[0:2], "little"),
                   "channels": int.from_bytes(body[2:4], "little"),
                   "rate": int.from_bytes(body[4:8], "little"),
                   "bits": int.from_bytes(body[14:16], "little")}
        elif chunk_id == b"data":
            data = body
        offset += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise Failure("artifact is missing fmt or data")
    samples = array.array("h")
    samples.frombytes(data)
    if sys.byteorder == "big":
        samples.byteswap()
    count = len(samples)
    energy = sum(float(s) * s for s in samples) / max(count, 1)
    non_silent = sum(1 for s in samples if abs(s) > 32) / max(count, 1)
    return {"samples": count, "duration_s": count / max(fmt["rate"], 1),
            "rms": energy ** 0.5 / 32768.0, "non_silence_ratio": non_silent,
            "peak": max((abs(s) for s in samples), default=0) / 32768.0,
            "wav_bytes": len(raw), "fmt": fmt, "pcm": samples}


def resample_16k(samples, in_rate=24000, out_rate=16000):
    """Test-only 24 kHz -> 16 kHz band-limited interpolation.

    The server accepts 16 kHz WAV only, and a plain 3-sample average is a poor
    anti-alias filter: it made the native ASR misread a correct utterance.
    A Hann-windowed sinc with a 7.2 kHz cutoff is enough for that hop.
    """
    if in_rate == out_rate:
        return list(samples)
    ratio = in_rate / out_rate
    half = 24
    cutoff = 0.45 * out_rate / in_rate
    out = []
    for index in range(int(len(samples) * out_rate / in_rate)):
        center = index * ratio
        base = int(center)
        acc = 0.0
        for tap in range(base - half, base + half + 1):
            if tap < 0 or tap >= len(samples):
                continue
            u = center - tap
            x = 2.0 * cutoff * u
            sinc = 1.0 if x == 0 else math.sin(math.pi * x) / (math.pi * x)
            window = 0.5 + 0.5 * math.cos(math.pi * u / half)
            acc += samples[tap] * 2.0 * cutoff * sinc * window
        out.append(max(-1.0, min(1.0, acc)))
    return out


def write_wav_16k(path, samples):
    # `samples` are raw int16 values from wav_stats; the filter works in
    # normalized units.
    normalized = [value / 32768.0 for value in samples]
    pcm = bytearray()
    for value in resample_16k(normalized):
        pcm += int(round(value * 32767.0)).to_bytes(2, "little", signed=True)
    header = b"RIFF" + (36 + len(pcm)).to_bytes(4, "little") + b"WAVEfmt "
    header += (16).to_bytes(4, "little") + (1).to_bytes(2, "little")
    header += (1).to_bytes(2, "little") + (16000).to_bytes(4, "little")
    header += (32000).to_bytes(4, "little") + (2).to_bytes(2, "little")
    header += (16).to_bytes(2, "little") + b"data"
    header += len(pcm).to_bytes(4, "little")
    with open(path, "wb") as handle:
        handle.write(header + bytes(pcm))


def normalize(text):
    return " ".join("".join(ch.lower() if ch.isalnum() or ch.isspace() else " "
                            for ch in text).split())


def count_omni_requests(directory):
    """Counts completed Omni requests. The stage log is truncated at start but
    stays sparse, so NUL padding is stripped before decoding."""
    try:
        with open(os.path.join(directory, "omni.log"), "rb") as handle:
            data = handle.read().replace(b"\x00", b"")
    except OSError:
        return 0
    # Only completed requests count: an aborted request writes a line that
    # starts the same way but reports a boundary instead of token counts.
    return sum(1 for line in data.decode("utf-8", "replace").splitlines()
               if line.startswith("omni request ") and
               "generated_tokens=" in line and "boundary=" not in line)


def transcribe_curl(base, path, out_dir, name):
    """Direct ASR with the exact command, exit code and raw streams."""
    stdout_path = os.path.join(out_dir, name + ".stdout")
    cmd = ["curl", "-sS", "-o", stdout_path, "-w", "%{http_code}",
           "-F", "model=qwen3-native", "-F", "file=@" + path,
           base + "/v1/audio/transcriptions"]
    started = time.time()
    completed = subprocess.run(cmd, capture_output=True, text=True)
    http_status = completed.stdout.strip()
    record(out_dir, name + ".stderr", completed.stderr or "")
    record(out_dir, name + ".meta", "\n".join([
        "cmd: " + " ".join(cmd),
        "process_exit: %d" % completed.returncode,
        "http_status: %s" % http_status,
        "seconds: %.3f" % (time.time() - started),
        "input: " + path,
    ]) + "\n")
    body = open(stdout_path, "r", encoding="utf-8").read()
    if completed.returncode != 0 or http_status != "200":
        raise Failure("transcription curl exit %d http %s: %s" %
                      (completed.returncode, http_status, completed.stderr[:400]))
    return json.loads(body), time.time() - started


def transcribe(base, path):
    boundary = "----umm-native-smoke"
    body = bytearray()
    for name, value in (("model", "qwen3-native"), ("response_format", "json")):
        body += ("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                 % (boundary, name, value)).encode()
    body += ("--%s\r\nContent-Disposition: form-data; name=\"file\"; "
             "filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
             % boundary).encode()
    body += wav_bytes(path) + ("\r\n--%s--\r\n" % boundary).encode()
    status, payload, elapsed = request_bytes(
        base + "/v1/audio/transcriptions", bytes(body),
        {"Content-Type": "multipart/form-data; boundary=" + boundary})
    if status != 200:
        raise Failure("transcription HTTP %d: %s" % (status, payload[:400]))
    result = json.loads(payload.decode("utf-8"))
    return result, elapsed


def wait_ready(base, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            status, body = get_json(base + "/ready", timeout=10)
        except (urllib.error.URLError, OSError):
            time.sleep(1)
            continue
        if status == 200:
            return True
        time.sleep(2)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stage-log-dir", default="/root/umm-native-service/logs")
    parser.add_argument("--ready-timeout", type=float, default=900)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    os.makedirs(args.out_dir, exist_ok=True)
    checks = []

    def expect(condition, message):
        checks.append({"check": message, "ok": bool(condition)})
        if not condition:
            raise Failure(message)

    try:
        if not wait_ready(base, args.ready_timeout):
            raise Failure("/ready did not become 200 within %ss" % args.ready_timeout)
        status, health = get_json(base + "/health")
        expect(status == 200 and health.get("in_flight") == 0,
               "health 200 and in_flight 0 before requests")
        record(args.out_dir, "health.json", json.dumps({"status": status, "body": health}))

        # 1. Direct ASR through the transcription route, with the exact
        # command, process exit code and raw stdout/stderr kept as evidence.
        asr_result, asr_elapsed = transcribe_curl(base, args.wav, args.out_dir,
                                                  "asr_direct")
        record(args.out_dir, "asr_direct.json",
               json.dumps({"result": asr_result, "seconds": round(asr_elapsed, 3)},
                          ensure_ascii=False))
        expect(normalize(asr_result.get("text", "")) == normalize(EXPECTED_ASR),
               "direct ASR transcript matches the known WAV")
        print("ASR direct:", asr_result.get("text"))

        # 2. Text -> VLM.
        status, body, elapsed = request(base + "/v1/chat/completions", {
            "model": "qwen3-native", "max_tokens": 64,
            "messages": [{"role": "user",
                          "content": "Reply with one short sentence about the sea."}],
        })
        answer = json.loads(body.decode("utf-8"))
        record(args.out_dir, "vlm_text.json",
               json.dumps({"status": status, "seconds": round(elapsed, 3),
                           "body": answer}, ensure_ascii=False, indent=1))
        content = answer["choices"][0]["message"]["content"]
        expect(status == 200 and content.strip(), "text-only VLM returns content")
        print("VLM text:", content[:160])

        # 3. Image -> VLM.
        status, body, elapsed = request(base + "/v1/chat/completions", {
            "model": "qwen3-native", "max_tokens": 96,
            "messages": [{"role": "user", "content": [
                image_part(args.image),
                {"type": "text", "text": "Describe what the person is doing in one sentence."},
            ]}],
        })
        image_answer = json.loads(body.decode("utf-8"))
        record(args.out_dir, "vlm_image.json",
               json.dumps({"status": status, "seconds": round(elapsed, 3),
                           "body": image_answer}, ensure_ascii=False, indent=1))
        image_content = image_answer["choices"][0]["message"]["content"]
        expect(status == 200 and len(image_content.strip()) > 10,
               "image VLM returns a description")
        print("VLM image:", image_content[:200])

        # 4. Audio -> ASR -> VLM, streamed.
        request_id, text, _, terminal = collect(
            chat_stream(base, [audio_part(args.wav),
                               {"type": "text", "text": "What did the speaker say?"}]),
            args.out_dir, "audio_to_text")
        record(args.out_dir, "audio_to_text.json", json.dumps({
            "request_id": request_id, "answer": text, "terminal": terminal,
        }, ensure_ascii=False, indent=1))
        expect(terminal is not None and terminal.get("kind") == "complete",
               "audio->text stream ends with complete")
        expect(text.strip(), "audio->text stream returns VLM text")
        print("audio->text:", text[:160])

        # 5. One request carries text, image and audio inputs and asks for all
        # three outputs. This exercises ASR -> VLM -> native Image + TTS.
        mixed_id, mixed_text, mixed_artifacts, mixed_terminal = collect(
            chat_stream(base, [
                audio_part(args.wav),
                image_part(args.image),
                {"type": "text", "text": "Use the audio and image; answer briefly."},
            ], extra={
                "modalities": ["text", "image", "audio"],
                "image": {"prompt": "A red fox beside a chair in snow",
                          "size": "512x512", "seed": 42,
                          "output_format": "png"},
                "audio": {"voice": "vivian", "format": "wav",
                          "sample_rate": 24000},
            }, max_tokens=64), args.out_dir, "mixed_all")
        mixed_images = [item for item in mixed_artifacts
                        if item.get("media_type") == "image"]
        mixed_audio = [item for item in mixed_artifacts
                       if item.get("media_type") == "audio"]
        mixed_png = decode_artifact(
            mixed_images, os.path.join(args.out_dir, "mixed_image.png"))
        mixed_wav = decode_artifact(
            mixed_audio, os.path.join(args.out_dir, "mixed_audio.wav"))
        png_summary = png_stats(mixed_png)
        mixed_wav_summary = wav_stats(mixed_wav)
        mixed_wav_summary = {key: mixed_wav_summary[key] for key in
                             ("samples", "duration_s", "rms", "non_silence_ratio",
                              "peak", "wav_bytes", "fmt")}
        record(args.out_dir, "mixed_all.json", json.dumps({
            "request_id": mixed_id, "answer": mixed_text,
            "terminal": mixed_terminal, "image": png_summary,
            "audio": mixed_wav_summary}, ensure_ascii=False, indent=1))
        expect(mixed_terminal is not None and
               mixed_terminal.get("kind") == "complete" and mixed_text.strip(),
               "mixed text+image+audio request completes with text")
        expect(png_summary["width"] == 512 and png_summary["height"] == 512 and
               png_summary["colors"] > 100,
               "mixed request returns a real non-monochrome native image")
        expect(mixed_wav_summary["fmt"]["rate"] == 24000 and
               mixed_wav_summary["rms"] > 0.01,
               "mixed request returns native non-silent audio")
        print("mixed all:", mixed_text[:120], json.dumps(png_summary))

        # 6. Text -> TTS, streamed artifact.
        # The TTS leg speaks the aggregated VLM answer, not the client text.
        tts_request, spoken, artifacts, terminal = collect(
            chat_stream(base, TTS_TEXT, extra={
                "modalities": ["text", "audio"],
                "audio": {"voice": "vivian", "format": "wav", "sample_rate": 24000},
            }, max_tokens=64), args.out_dir, "text_to_tts")
        raw = decode_artifact(artifacts, os.path.join(args.out_dir, "tts.wav"))
        stats = wav_stats(raw)
        summary = {key: stats[key] for key in
                   ("samples", "duration_s", "rms", "non_silence_ratio", "peak",
                    "wav_bytes", "fmt")}
        record(args.out_dir, "text_to_tts.json", json.dumps({
            "request_id": tts_request, "terminal": terminal,
            "spoken_text": spoken, "artifacts": len(artifacts), "wav": summary,
        }, indent=1))
        expect(terminal is not None and terminal.get("kind") == "complete",
               "text->tts stream ends with complete")
        expect(stats["fmt"]["rate"] == 24000 and stats["fmt"]["channels"] == 1
               and stats["fmt"]["bits"] == 16, "TTS artifact is 24 kHz mono PCM16")
        expect(stats["rms"] > 0.01 and stats["non_silence_ratio"] > 0.1,
               "TTS artifact is not silent")
        print("TTS:", json.dumps(summary))

        # 7. Native ASR round-trip of the generated WAV (24k -> 16k test resample).
        resampled = os.path.join(args.out_dir, "tts_16k.wav")
        write_wav_16k(resampled, stats["pcm"])
        round_trip, round_elapsed = transcribe(base, resampled)
        record(args.out_dir, "tts_roundtrip.json", json.dumps({
            "result": round_trip, "seconds": round(round_elapsed, 3),
            "expected": spoken}, ensure_ascii=False, indent=1))
        expect(normalize(round_trip.get("text", "")) == normalize(spoken),
               "native ASR recognizes the synthesized speech")
        print("round-trip:", round_trip.get("text"))

        # 8. Cooperative cancel at each engine's own boundary.
        def abort_now(request_id):
            sent = int(time.time() * 1000)
            status, payload, _ = request(base + "/abort", {"request_id": request_id})
            return sent, status, json.loads(payload.decode("utf-8"))

        def drain(stream):
            kinds = []
            terminal = None
            for frame in stream:
                unified = frame.get("unified")
                if unified:
                    kinds.append(unified.get("kind"))
                    if unified.get("kind") in ("complete", "error", "cancelled"):
                        terminal = unified
            return kinds, terminal

        # 8a. ASR: abort as soon as the request id is known, so the stage stops
        # inside the audio tower or the prefill.
        stream = chat_stream(base, [audio_part(args.wav),
                                    {"type": "text", "text": "Repeat the transcript."}])
        asr_cancel_id = next(stream)["id"]
        # The ASR stage needs about a second; aborting after ~150 ms lands
        # inside the audio tower or the prefill, not before the submit.
        time.sleep(0.15)
        sent, abort_status, abort_body = abort_now(asr_cancel_id)
        kinds, terminal = drain(stream)
        _, health_asr = get_json(base + "/health")
        record(args.out_dir, "cancel_asr.json", json.dumps({
            "request_id": asr_cancel_id, "abort_epoch_ms": sent,
            "abort_status": abort_status, "abort_body": abort_body,
            "kinds": kinds, "terminal": terminal,
            "in_flight_after": health_asr.get("in_flight")}, indent=1))
        expect(abort_status == 200 and kinds and kinds[-1] == "cancelled",
               "ASR cancel settles as cancelled")
        expect(health_asr.get("in_flight") == 0, "in_flight 0 after ASR cancel")
        print("cancel ASR:", kinds[-1])

        # 8b. Omni: tensor-core decode is much faster than the original slice.
        # Abort at 0.5 s, inside the token loop of a 96-token response.
        stream = chat_stream(base, "Write a long paragraph about the sea.")
        omni_cancel_id = next(stream)["id"]
        time.sleep(0.5)
        sent, abort_status, abort_body = abort_now(omni_cancel_id)
        kinds, terminal = drain(stream)
        _, health_omni = get_json(base + "/health")
        record(args.out_dir, "cancel_omni.json", json.dumps({
            "request_id": omni_cancel_id, "abort_epoch_ms": sent,
            "abort_status": abort_status, "abort_body": abort_body,
            "kinds": kinds, "terminal": terminal,
            "in_flight_after": health_omni.get("in_flight")}, indent=1))
        expect(abort_status == 200 and kinds and kinds[-1] == "cancelled",
               "Omni cancel settles as cancelled")
        expect(health_omni.get("in_flight") == 0, "in_flight 0 after Omni cancel")
        print("cancel Omni:", kinds[-1])

        # 8c. Image: abort after model load and the first denoise steps.
        stream = chat_stream(base, TTS_TEXT, extra={
            "modalities": ["text", "image"],
            "image": {"prompt": "A red fox beside a chair in snow",
                      "size": "512x512", "seed": 42, "output_format": "png"},
        }, max_tokens=64)
        image_cancel_id = next(stream)["id"]
        time.sleep(15.0)
        sent, abort_status, abort_body = abort_now(image_cancel_id)
        kinds, terminal = drain(stream)
        _, health_image = get_json(base + "/health")
        record(args.out_dir, "cancel_image.json", json.dumps({
            "request_id": image_cancel_id, "abort_epoch_ms": sent,
            "abort_status": abort_status, "abort_body": abort_body,
            "kinds": kinds, "terminal": terminal,
            "in_flight_after": health_image.get("in_flight")}, indent=1))
        expect(abort_status == 200 and "artifact" not in kinds and
               terminal is not None and terminal.get("kind") == "cancelled",
               "Image cancel settles as cancelled without an artifact")
        expect(health_image.get("in_flight") == 0,
               "in_flight 0 after Image cancel")
        print("cancel Image:", terminal.get("kind") if terminal else None)

        # 8d. TTS: the leg emits nothing until its artifact, so wait for the
        # VLM leg to finish (the omni stage logs one line per completed
        # request), then abort 5 s into synthesis.
        omni_done = count_omni_requests(args.stage_log_dir)
        stream = chat_stream(base, TTS_TEXT, extra={
            "modalities": ["text", "audio"],
            "audio": {"voice": "ryan", "format": "wav", "sample_rate": 24000},
        }, max_tokens=64)
        tts_cancel_id = next(stream)["id"]
        deadline = time.time() + 240
        while (time.time() < deadline and
               count_omni_requests(args.stage_log_dir) <= omni_done):
            time.sleep(0.5)
        omni_after = count_omni_requests(args.stage_log_dir)
        # The VLM leg is done; synthesis is under way. The talker loop takes
        # about 3 s and the waveform decoder about 30 s, so 10 s lands inside
        # the decoder with many boundaries left.
        time.sleep(10.0)
        sent, abort_status, abort_body = abort_now(tts_cancel_id)
        kinds, terminal = drain(stream)
        _, health_tts = get_json(base + "/health")
        record(args.out_dir, "cancel_tts.json", json.dumps({
            "request_id": tts_cancel_id, "abort_epoch_ms": sent,
            "abort_status": abort_status, "abort_body": abort_body,
            "kinds": kinds, "terminal": terminal,
            "omni_completions_before": omni_done,
            "omni_completions_after": omni_after,
            "in_flight_after": health_tts.get("in_flight")}, indent=1))
        expect(abort_status == 200 and "artifact" not in kinds and
               terminal is not None and terminal.get("kind") == "cancelled",
               "TTS cancel settles as cancelled without an artifact")
        expect(health_tts.get("in_flight") == 0, "in_flight 0 after TTS cancel")
        print("cancel TTS:", terminal.get("kind") if terminal else None)

        # 9. The service still answers after the cancel.
        status, body, _ = request(base + "/v1/chat/completions", {
            "model": "qwen3-native", "max_tokens": 32,
            "messages": [{"role": "user", "content": "Say OK."}],
        })
        after = json.loads(body.decode("utf-8"))
        record(args.out_dir, "after_cancel.json", json.dumps(
            {"status": status, "body": after}, indent=1))
        expect(status == 200 and after["choices"][0]["message"]["content"].strip(),
               "a later request completes")
        print("after cancel:", after["choices"][0]["message"]["content"][:80])
    except Failure as error:
        record(args.out_dir, "checks.json", json.dumps(
            {"checks": checks, "failure": str(error)}, indent=1))
        print("SMOKE FAILED:", error, file=sys.stderr)
        return 1
    except Exception as error:  # noqa: BLE001 - evidence must name any error
        record(args.out_dir, "checks.json", json.dumps(
            {"checks": checks, "failure": "%s: %s" % (type(error).__name__, error)},
            indent=1))
        print("SMOKE ERROR:", type(error).__name__, error, file=sys.stderr)
        return 1
    record(args.out_dir, "checks.json", json.dumps({"checks": checks}, indent=1))
    print("SMOKE PASSED: %d checks" % len(checks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
