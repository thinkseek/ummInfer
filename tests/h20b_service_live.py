#!/usr/bin/env python3
"""Live validation of the resident ummInfer service on H20B.

Talks to an already running service over public HTTP/SSE. It never starts or
stops the service; the supervisor owns those processes. It covers:

  1. /health and /ready;
  2. a real audio -> ASR -> VLM streaming request;
  3. a real joint image + audio request;
  4. a cancel during the VLM stream, with in_flight returning to zero;
  5. a later request that completes.

Raw SSE frames and input hashes are written to --evidence-dir. The run is
opt-in: without UMMINFER_LIVE_OPT_IN=1 it prints BLOCKED and exits 0.
"""

import argparse
import base64
import hashlib
import http.client
import json
import os
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"PASS {name}")
    else:
        print(f"FAIL {name} {detail}")
        FAILURES.append(name)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audio_part(path):
    return {"type": "input_audio",
            "input_audio": {"data": base64.b64encode(Path(path).read_bytes())
                            .decode(), "format": "wav"}}


def image_part(path):
    raw = Path(path).read_bytes()
    suffix = "jpeg" if raw[:2] == b"\xff\xd8" else "png"
    return {"type": "image_url", "image_url": {
        "url": f"data:image/{suffix};base64,"
               + base64.b64encode(raw).decode()}}


def request_body(model, audio, prompt, image=None, max_tokens=128):
    content = []
    if image:
        content.append(image_part(image))
    if prompt:
        content.append({"type": "text", "text": prompt})
    content.append(audio_part(audio))
    return {"model": model, "stream": True, "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}]}


def stream(url, body, on_frame=None, timeout=600):
    parts = urllib.parse.urlsplit(url)
    connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout)
    path = parts.path + ("?" + parts.query if parts.query else "")
    connection.request("POST", path, body=json.dumps(body),
                       headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    frames = []
    pending = []
    try:
        for raw in response:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if line.startswith("data:"):
                pending.append(line[5:].lstrip())
            elif not line and pending:
                data = "\n".join(pending)
                pending = []
                if data == "[DONE]":
                    break
                frame = json.loads(data)
                frames.append(frame)
                if on_frame is not None:
                    on_frame(frame)
    finally:
        connection.close()
    return response.status, frames


def events(frames):
    return [frame["unified"] for frame in frames if "unified" in frame]


def summarise(name, status, frames):
    found = events(frames)
    kinds = [event["kind"] for event in found]
    seqs = [event["seq"] for event in found]
    terminals = [event for event in found
                 if event["kind"] in ("complete", "error", "cancelled")]
    finals = [event for event in found if event["kind"] == "asr_final"]
    deltas = [event for event in found if event["kind"] == "vlm_delta"]
    return {
        "case": name,
        "http_status": status,
        "frames": len(frames),
        "kinds": kinds,
        "seq_gap_free": seqs == list(range(len(seqs))),
        "terminals": [event["kind"] for event in terminals],
        "asr_final_text": finals[0]["text"] if finals else None,
        "vlm_text": "".join(event["text"] for event in deltas),
        "request_id": frames[0].get("id") if frames else None,
    }


def get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status, json.loads(response.read().decode())


def post_abort(base, request_id):
    parts = urllib.parse.urlsplit(base)
    connection = http.client.HTTPConnection(parts.hostname, parts.port, 10)
    try:
        connection.request("POST", "/abort",
                           body=json.dumps({"request_id": request_id}),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.read().decode()
    finally:
        connection.close()


def save(evidence, name, payload):
    path = evidence / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:43300")
    parser.add_argument("--model", default="qwen3-unified")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--audio-second", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt", default="Answer briefly in one sentence.")
    args = parser.parse_args()
    if os.environ.get("UMMINFER_LIVE_OPT_IN") != "1":
        print("BLOCKED set UMMINFER_LIVE_OPT_IN=1 to run the live service check")
        return 0
    evidence = Path(args.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    chat_url = args.base_url.rstrip("/") + "/v1/chat/completions"

    status, health = get_json(args.base_url + "/health")
    check("health 200", status == 200 and health.get("status") == "ok",
          f"{status} {health}")
    status, ready = get_json(args.base_url + "/ready")
    check("ready 200", status == 200 and ready.get("status") == "ready",
          f"{status} {ready}")
    check("health in_flight starts at 0", health.get("in_flight") == 0,
          str(health))

    inputs = {"audio": {"path": args.audio, "sha256": sha256(args.audio)},
              "model": args.model}
    if args.audio_second:
        inputs["audio_second"] = {"path": args.audio_second,
                                  "sha256": sha256(args.audio_second)}
    if args.image:
        inputs["image"] = {"path": args.image, "sha256": sha256(args.image)}

    results = {}
    body = request_body(args.model, args.audio, args.prompt,
                        max_tokens=args.max_tokens)
    status, frames = stream(chat_url, body)
    summary = summarise("asr_vlm", status, frames)
    save(evidence, "asr-vlm-frames.json", frames)
    results["asr_vlm"] = summary
    check("asr_vlm complete", summary["terminals"] == ["complete"],
          str(summary["terminals"]))
    check("asr_vlm gap-free seq", summary["seq_gap_free"])
    check("asr_vlm transcript non-empty", bool(summary["asr_final_text"]))
    check("asr_vlm answer non-empty", bool(summary["vlm_text"].strip()))

    if args.image:
        body = request_body(args.model, args.audio_second or args.audio,
                            "Describe the image, then answer the speech.",
                            image=args.image,
                            max_tokens=args.max_tokens)
        status, frames = stream(chat_url, body)
        summary = summarise("joint_image_audio", status, frames)
        save(evidence, "joint-image-audio-frames.json", frames)
        results["joint_image_audio"] = summary
        check("joint complete", summary["terminals"] == ["complete"],
              str(summary["terminals"]))
        check("joint transcript non-empty", bool(summary["asr_final_text"]))
        check("joint answer non-empty", bool(summary["vlm_text"].strip()))

    state = {}

    def abort_on_first_delta(frame):
        unified = frame.get("unified") or {}
        if (unified.get("kind") == "vlm_delta" and "abort" not in state):
            state["abort"] = post_abort(args.base_url, frame["id"])
            state["abort_id"] = frame["id"]
            state["abort_seq"] = unified.get("seq")
            state["abort_time"] = time.time()

    body = request_body(args.model, args.audio_second or args.audio,
                        "Count slowly from one to one hundred.", image=None,
                        max_tokens=256)
    status, frames = stream(chat_url, body,
                            on_frame=abort_on_first_delta)
    summary = summarise("cancel", status, frames)
    save(evidence, "cancel-frames.json", frames)
    results["cancel"] = summary
    results["cancel"]["abort"] = state.get("abort")
    found = events(frames)
    terminals = [event for event in found
                 if event["kind"] in ("complete", "error", "cancelled")]
    terminal_seq = terminals[0]["seq"] if terminals else None
    after_terminal = [event for event in found
                      if terminal_seq is not None
                      and event.get("seq") > terminal_seq]
    # Deltas already queued when /abort was posted are delivered before the
    # cancelled terminal. The terminal is the boundary that must be final.
    results["cancel"]["deltas_after_abort_seq"] = sum(
        1 for event in found if event.get("kind") == "vlm_delta"
        and state.get("abort_seq") is not None
        and event.get("seq") > state["abort_seq"])
    check("cancel sent an abort", bool(state.get("abort"))
          and state["abort"][0] == 200, str(state.get("abort")))
    check("cancel has exactly one terminal", len(terminals) == 1,
          str([event["kind"] for event in terminals]))
    check("cancel terminal is cancelled",
          bool(terminals) and terminals[0]["kind"] == "cancelled",
          str([event["kind"] for event in terminals]))
    check("cancel has no event after the terminal", not after_terminal,
          f"{len(after_terminal)} later events")
    time.sleep(1)
    status, health_after = get_json(args.base_url + "/health")
    results["health_after_cancel"] = health_after
    check("in_flight returns to 0 after cancel",
          health_after.get("in_flight") == 0, str(health_after))

    body = request_body(args.model, args.audio_second or args.audio,
                        args.prompt, max_tokens=args.max_tokens)
    status, frames = stream(chat_url, body)
    summary = summarise("recovery", status, frames)
    save(evidence, "recovery-frames.json", frames)
    results["recovery"] = summary
    check("recovery complete", summary["terminals"] == ["complete"],
          str(summary["terminals"]))
    check("recovery answer non-empty", bool(summary["vlm_text"].strip()))

    status, health_end = get_json(args.base_url + "/health")
    results["health_end"] = health_end
    check("in_flight ends at 0", health_end.get("in_flight") == 0,
          str(health_end))
    save(evidence, "live-summary.json",
         {"inputs": inputs, "results": results})
    print(f"RESULT {'PASS' if not FAILURES else 'FAIL'} "
          f"failures={len(FAILURES)} evidence={evidence}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
