#!/usr/bin/env python3
"""Closed-loop load driver for the ummInfer unified engine.

Each worker keeps one request in flight. Prints one JSON summary line.
Usage: umm-load.py --kind text|image|tts --concurrency N --requests M
                    [--retry-503 N]
"""
import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

PROMPT_CHAT = "Explain in one short sentence why the sky is blue."
PROMPT_IMAGE = "A red cube on a wooden table, studio light."
PROMPT_TTS = "The quick brown fox jumps over the lazy dog."


def build_payload(kind, max_tokens):
    if kind == "image":
        return {
            "model": "qwen3-native",
            "messages": [{"role": "user", "content": PROMPT_IMAGE}],
            "stream": True,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "modalities": ["text", "image"],
            "image": {"size": "512x512"},
        }
    if kind == "tts":
        return {
            "model": "qwen3-native",
            "messages": [{"role": "user", "content": PROMPT_TTS}],
            "stream": True,
            "temperature": 0.0,
            "max_tokens": 32,
            "modalities": ["text", "audio"],
            "audio": {"voice": "vivian", "format": "wav"},
        }
    return {
        "model": "qwen3-native",
        "messages": [{"role": "user", "content": PROMPT_CHAT}],
        "stream": False,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }


def read_sse(res, started):
    """Drains an SSE response; returns (ms, text, artifact_frames, done)."""
    text_parts = []
    artifacts = 0
    hashes = []
    done = False
    buf = b""
    while True:
        chunk = res.read(65536)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            frame, buf = buf.split(b"\n\n", 1)
            for line in frame.split(b"\n"):
                if not line.startswith(b"data: "):
                    continue
                data = line[6:]
                if data == b"[DONE]":
                    done = True
                    continue
                try:
                    event = json.loads(data)
                except Exception:  # noqa: BLE001
                    continue
                choices = event.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    if delta.get("content"):
                        text_parts.append(delta["content"])
                unified = event.get("unified") or {}
                artifact = unified.get("artifact") or {}
                if artifact.get("data"):
                    artifacts += 1
                if artifact.get("sha256"):
                    hashes.append(artifact["sha256"])
    return ((time.perf_counter() - started) * 1000, "".join(text_parts),
            artifacts, done, hashes)


def attempt_once(url, body, kind, timeout, started):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            if kind == "text":
                payload = res.read()
                elapsed = (time.perf_counter() - started) * 1000
                text = ""
                try:
                    parsed = json.loads(payload)
                    choices = parsed.get("choices") or []
                    if choices:
                        text = choices[0].get("message", {}).get("content", "")
                except Exception:  # noqa: BLE001
                    pass
                return {"status": res.status, "ms": elapsed, "bytes": len(payload),
                        "text": text, "artifacts": 0, "sha256": "", "done": True}
            elapsed, text, artifacts, done, hashes = read_sse(res, started)
            return {"status": res.status, "ms": elapsed, "text": text,
                    "artifacts": artifacts, "sha256": hashes[-1] if hashes else "",
                    "done": done}
    except urllib.error.HTTPError as err:
        payload = err.read()
        return {"status": err.code,
                "error": payload[:200].decode(errors="replace"),
                "retry_after": err.headers.get("Retry-After") or "",
                "ms": (time.perf_counter() - started) * 1000,
                "text": "", "artifacts": 0, "done": False}
    except Exception as err:  # noqa: BLE001 - record transport failures
        return {"status": 0, "error": str(err),
                "ms": (time.perf_counter() - started) * 1000,
                "text": "", "artifacts": 0, "done": False}


def one(url, body, kind, timeout, retry_503, retry_ms):
    started = time.perf_counter()
    rejections = 0
    while True:
        item = attempt_once(url, body, kind, timeout, started)
        item["rejections"] = rejections
        if item["status"] != 503 or rejections >= retry_503:
            return item
        rejections += 1
        if retry_ms > 0:
            # Jittered retry: fixed one-second wakeups synchronize every
            # rejected worker and turn shedding into a thundering herd.
            delay = (retry_ms + random.random() * retry_ms) / 1000.0
        else:
            try:
                delay = float(item.get("retry_after") or 0.05)
            except ValueError:
                delay = 0.05
        time.sleep(max(0.02, min(delay, 2.0)))


def pct(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",
                        default=os.environ.get("UMM_CONCURRENCY_URL",
                                               "http://127.0.0.1:43302"),
                        help="service URL (override with --url or UMM_CONCURRENCY_URL)")
    parser.add_argument("--kind", choices=["text", "image", "tts"], default="text")
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, default=0,
                        help="total requests; 0 means 3x concurrency")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--retry-503", type=int, default=0,
                        help="retries per request when the server answers 503")
    parser.add_argument("--retry-ms", type=int, default=0,
                        help="jittered retry in [ms, 2*ms]; 0 honors Retry-After")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    total = args.requests or args.concurrency * 3
    body = build_payload(args.kind, args.max_tokens)
    results = []
    lock = threading.Lock()
    next_index = [0]

    def worker():
        while True:
            with lock:
                if next_index[0] >= total:
                    return
                next_index[0] += 1
            item = one(args.url, body, args.kind, args.timeout,
                       args.retry_503, args.retry_ms)
            with lock:
                results.append(item)

    started = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - started
    ok = [r for r in results if r["status"] == 200 and r["done"]]
    latencies = [r["ms"] for r in ok]
    errors = {}
    for r in results:
        if r["status"] != 200 or not r["done"]:
            key = str(r["status"]) + ":" + r.get("error", "no_done")[:80]
            errors[key] = errors.get(key, 0) + 1
    summary = {
        "kind": args.kind,
        "concurrency": args.concurrency,
        "requests": total,
        "wall_s": round(wall, 3),
        "throughput_rps": round(len(ok) / wall, 3),
        "ok": len(ok),
        "errors": errors,
        "rejections_503": sum(r.get("rejections", 0) for r in results),
        "p50_ms": round(pct(latencies, 0.50), 1) if latencies else None,
        "p95_ms": round(pct(latencies, 0.95), 1) if latencies else None,
        "p99_ms": round(pct(latencies, 0.99), 1) if latencies else None,
        "mean_ms": round(statistics.fmean(latencies), 1) if latencies else None,
        "min_ms": round(min(latencies), 1) if latencies else None,
        "max_ms": round(max(latencies), 1) if latencies else None,
        "distinct_texts": len({r.get("text", "") for r in ok}),
        "artifact_frames": sorted({r.get("artifacts", 0) for r in ok}),
        "artifact_sha256": sorted({r.get("sha256", "") for r in ok if r.get("sha256")}),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as sink:
            for item in results:
                sink.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    sys.exit(main())
