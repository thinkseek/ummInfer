#!/usr/bin/env python3
"""Drive one unified umm-infer request and record measured evidence.

Committed copy of the client used for docs/cases/0001-end-to-end-native.md.

Input parts: real text prompt + real 16 kHz WAV. Output: text answer, PNG, WAV.
One request, one request_id, one SSE stream.

Usage:
  python3 -I e2e_client.py run --url http://172.17.0.2:43301 \
      --audio asr_en_pcm16.wav --prompt "..." --out out/run1
  python3 -I e2e_client.py abortrun --url ... --out out/abort
"""

import argparse
import base64
import hashlib
import http.client
import json
import struct
import sys
import threading
import time
import urllib.parse
from pathlib import Path

IMAGE_PROMPT = "A songwriter at a grand piano in a warm studio, soft window light, photographic, 512x512"
TEXT_PROMPT = ("Listen to the audio. In one sentence, state the speaker's main point, "
               "then name one object that would make a good illustration.")
VOICE = "vivian"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_body(audio_bytes, prompt, stream=True, max_tokens=64, temperature=0):
    return {
        "model": "qwen3-native",
        "stream": stream,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "input_audio", "input_audio": {
                "data": base64.b64encode(audio_bytes).decode(), "format": "wav"}},
        ]}],
        "modalities": ["text", "image", "audio"],
        "image": {"prompt": IMAGE_PROMPT, "size": "512x512", "seed": 42,
                  "output_format": "png"},
        "audio": {"voice": VOICE, "format": "wav", "sample_rate": 24000},
    }


def png_info(data):
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return {"error": "not a PNG"}
    width, height = struct.unpack(">II", data[16:24])
    return {"width": width, "height": height, "magic_ok": True}


def wav_info(data):
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return {"error": "not a WAV"}
    offset, info = 12, {}
    while offset + 8 <= len(data):
        cid = data[offset:offset + 4]
        size = struct.unpack("<I", data[offset + 4:offset + 8])[0]
        payload = data[offset + 8:offset + 8 + size]
        if cid == b"fmt ":
            fmt, channels, rate, _, align, bits = struct.unpack("<HHIIHH", payload[:16])
            info.update(fmt=fmt, channels=channels, rate=rate, bits=bits, align=align)
        elif cid == b"data":
            info["data_bytes"] = len(payload)
        offset += 8 + size + (size & 1)
    if info.get("channels"):
        info["samples"] = info["data_bytes"] // (info["channels"] * info["bits"] // 8)
        info["duration_s"] = round(info["samples"] / info["rate"], 3)
    return info


def abort_request(url, request_id):
    parts = urllib.parse.urlsplit(url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=10)
    body = json.dumps({"request_id": request_id})
    conn.request("POST", "/abort", body=body,
                 headers={"Content-Type": "application/json"})
    response = conn.getresponse()
    payload = response.read().decode("utf-8", "replace")
    conn.close()
    return response.status, payload


def do_run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    audio_bytes = Path(args.audio).read_bytes()
    body = build_body(audio_bytes, args.prompt, max_tokens=args.max_tokens)
    redacted = json.loads(json.dumps(body))
    redacted["messages"][0]["content"][1]["input_audio"]["data"] = "<omitted>"
    (out / "request.json").write_text(json.dumps(redacted, indent=2) + "\n")

    parts = urllib.parse.urlsplit(args.url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=args.timeout)
    raw_path = "/v1/chat/completions"
    t_submit = time.time()
    conn.request("POST", raw_path, body=json.dumps(body),
                 headers={"Content-Type": "application/json"})
    response = conn.getresponse()
    status = response.status
    received_at = time.time()

    events, raw_lines = [], []
    pending = []
    artifacts = {}
    request_id = None
    t_abort_sent = None
    abort_status = None
    abort_body = ""
    first_event_at = None

    abort_timer = [None]

    def send_abort():
        nonlocal t_abort_sent, abort_status, abort_body
        t_abort_sent = time.time()
        abort_status, abort_body = abort_request(args.url, request_id)

    def handle(frame_text):
        nonlocal request_id, first_event_at
        if frame_text == "[DONE]":
            return True
        frame = json.loads(frame_text)
        now = time.time()
        if request_id is None:
            request_id = frame.get("id")
            first_event_at = now
            if args.abort_after_first_event:
                abort_timer[0] = threading.Timer(args.abort_delay, send_abort)
                abort_timer[0].daemon = True
                abort_timer[0].start()
        entry = {"t": round(now - t_submit, 3), "id": frame.get("id"),
                 "frame": frame}
        choices = frame.get("choices") or []
        if choices and isinstance(choices[0].get("delta"), dict):
            content = choices[0]["delta"].get("content")
            if content:
                entry["delta"] = content
        if "unified" in frame:
            unified = frame["unified"]
            entry["kind"] = unified.get("kind")
            entry["stage"] = unified.get("stage")
            entry["seq"] = unified.get("seq")
            entry["text"] = unified.get("text", "")
            if "artifact" in unified:
                artifact = unified["artifact"]
                slot = artifacts.setdefault(artifact["artifact_id"],
                                            {"chunks": [], "meta": None})
                slot["chunks"].append((artifact.get("index", 0),
                                       artifact.get("data", "")))
                meta = {k: v for k, v in artifact.items() if k != "data"}
                slot["meta"] = meta
                entry["artifact"] = meta
                frame["unified"]["artifact"]["data"] = \
                    f"<omitted {len(artifact.get('data', ''))} b64 chars>"
        events.append(entry)
        return False

    done = False
    for raw in response:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        raw_lines.append(line)
        if line.startswith("data:"):
            pending.append(line[5:].lstrip())
        elif not line and pending:
            payload = "\n".join(pending)
            pending = []
            if handle(payload):
                done = True
                break
    if not done and pending:
        handle("\n".join(pending))
    t_stream_end = time.time()
    conn.close()
    terminal = next((e for e in events
                     if e.get("kind") in ("complete", "error", "cancelled")), None)
    t_terminal = (t_submit + terminal["t"]) if terminal else None

    # Decode artifacts, verify the transport sha256 against the decoded bytes.
    artifact_results = {}
    for art_id, slot in sorted(artifacts.items()):
        blob = b"".join(base64.b64decode(chunk)
                        for _, chunk in sorted(slot["chunks"], key=lambda c: c[0]))
        digest = sha256_bytes(blob)
        meta = slot["meta"] or {}
        media = meta.get("media_type")
        suffix = {"image": "png", "audio": "wav"}.get(media, "bin")
        path = out / f"{art_id}.{suffix}"
        path.write_bytes(blob)
        info = png_info(blob) if media == "image" else wav_info(blob)
        artifact_results[art_id] = {
            "path": str(path), "media_type": media, "bytes": len(blob),
            "sha256": digest, "meta": meta, "info": info,
            "transport_sha256": meta.get("sha256"),
            "sha256_matches": meta.get("sha256") == digest,
            "chunks": len(slot["chunks"]),
        }

    answer = "".join(e.get("delta") or
                     (e["text"] if e.get("kind") == "vlm_delta" else "")
                     for e in events)
    transcript = "".join(e["text"] for e in events if e.get("kind") == "asr_final")
    (out / "text.txt").write_text(answer)
    (out / "transcript.txt").write_text(transcript)
    (out / "response.sse").write_text("\n".join(raw_lines) + "\n")
    redacted_events = json.loads(json.dumps(events))
    (out / "events.json").write_text(json.dumps(redacted_events, indent=1) + "\n")

    def t_of(kind, which="last"):
        hits = [e["t"] for e in events if e.get("kind") == kind]
        return (hits[-1] if which == "last" else hits[0]) if hits else None

    asr_t = t_of("asr_final")
    first_delta = t_of("vlm_delta", "first")
    last_delta = t_of("vlm_delta")
    image_t = next((e["t"] for e in events if e.get("kind") == "artifact"
                    and e.get("stage") == "image"), None)
    tts_t = next((e["t"] for e in events if e.get("kind") == "artifact"
                  and e.get("stage") == "tts"), None)
    vlm_end = last_delta if last_delta is not None else asr_t
    leg_ms = {
        "asr_stream": round(asr_t * 1000) if asr_t is not None else None,
        "vlm_prefill": round((first_delta - asr_t) * 1000)
        if first_delta is not None and asr_t is not None else None,
        "vlm_decode": round((last_delta - first_delta) * 1000)
        if last_delta is not None and first_delta is not None else None,
        "image": round((image_t - vlm_end) * 1000) if image_t is not None else None,
        "tts": round((tts_t - vlm_end) * 1000) if tts_t is not None else None,
        "total_to_terminal": round((t_terminal - t_submit) * 1000)
        if t_terminal is not None else None,
        "artifact_at": {"image": image_t, "tts": tts_t},
        "asr_final_at": asr_t,
        "stream_end": round((t_stream_end - t_submit) * 1000),
    }
    abort_ms = None
    if t_abort_sent is not None:
        end_ref = t_terminal if t_terminal is not None else t_stream_end
        abort_ms = {
            "send_to_terminal": round((end_ref - t_abort_sent) * 1000),
            "send_to_socket_eof": round((t_stream_end - t_abort_sent) * 1000),
            "http_status": abort_status, "response": abort_body,
        }

    result = {
        "http_status": status,
        "request_id": request_id,
        "received_headers_at_s": round(received_at - t_submit, 3),
        "event_count": len(events),
        "kinds": [e.get("kind") for e in events],
        "terminal": terminal["frame"]["unified"] if terminal else None,
        "answer": answer,
        "transcript": transcript,
        "leg_ms": leg_ms,
        "artifacts": artifact_results,
        "abort": abort_ms,
        "audio_input": {"path": str(Path(args.audio).resolve()),
                        "bytes": len(audio_bytes),
                        "sha256": sha256_bytes(audio_bytes)},
        "image_prompt": IMAGE_PROMPT, "text_prompt": args.prompt,
        "voice": VOICE, "seed": 42, "max_tokens": args.max_tokens,
        "started_epoch_s": round(t_submit, 3),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in
                      ("http_status", "request_id", "kinds", "leg_ms", "abort")},
                     indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "abortrun"])
    parser.add_argument("--url", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--prompt", default=TEXT_PROMPT)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--abort-after-first-event", action="store_true")
    parser.add_argument("--abort-delay", type=float, default=0.3)
    args = parser.parse_args()
    if args.mode == "abortrun":
        args.abort_after_first_event = True
    return do_run(args)


if __name__ == "__main__":
    sys.exit(main())
