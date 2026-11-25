#!/usr/bin/env python3
"""Opt-in live SM90 comparison of uniform vs temporal Qwen3.8 video input.

Each --video is paired with one --question (or one question shared by all
videos). Both policies receive identical model, prompt, video bytes, seed, and
stream settings; only video_url.frame_selection differs. Answers and TTFT/total
latency are recorded for human/model-quality review, never synthesized here.

Offline:
  python3 -I tests/video_temporal_live.py --self-check
Live (explicit opt-in, on the designated worker only):
  UMMINFER_LIVE_OPT_IN=1 python3 -I tests/video_temporal_live.py \
    --url "$UMMINFER_URL" --video motion.mp4 --question 'What changes first?' \
    --expect 'scene' --out-dir /tmp/video-temporal-evidence
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

MAX_VIDEO_BYTES = 32 << 20


def encode_video(path: Path) -> tuple[str, str, str]:
    size = path.stat().st_size
    if size <= 0 or size > MAX_VIDEO_BYTES:
        raise ValueError(f"video must contain 1..{MAX_VIDEO_BYTES} bytes: {path}")
    data = path.read_bytes()
    suffix = path.suffix.lower()
    mime = {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime"}.get(suffix)
    if mime is None:
        raise ValueError("video extension must be .mp4, .webm, or .mov")
    url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    return url, hashlib.sha256(data).hexdigest(), mime


def build_body(url: str, question: str, policy: str, model: str,
               motion_threshold: float = 0.02, scene_threshold: float = 0.25) -> dict:
    if policy not in ("uniform", "temporal"):
        raise ValueError("policy must be uniform or temporal")
    if not question:
        raise ValueError("question must be nonempty")
    for threshold in (motion_threshold, scene_threshold):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("thresholds must be in [0,1]")
    video = {"url": url}
    if policy == "temporal":
        video["frame_selection"] = {
            "policy": "temporal",
            "motion_threshold": motion_threshold,
            "scene_threshold": scene_threshold,
        }
    return {
        "model": model,
        "stream": True,
        "temperature": 0.0,
        "seed": 42,
        "max_tokens": 128,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "video_url", "video_url": video},
            ],
        }],
    }


def self_check() -> int:
    url = "data:video/mp4;base64,AAAAIGZ0eXBpc29t"
    plain = build_body(url, "What moves?", "uniform", "qwen3-native")
    temporal = build_body(url, "What moves?", "temporal", "qwen3-native")
    plain_video = plain["messages"][0]["content"][1]["video_url"]
    temporal_video = temporal["messages"][0]["content"][1]["video_url"]
    assert "frame_selection" not in plain_video
    assert temporal_video["url"] == plain_video["url"]
    assert temporal_video["frame_selection"]["policy"] == "temporal"
    comparable_plain = json.loads(json.dumps(plain))
    comparable_temporal = json.loads(json.dumps(temporal))
    del comparable_temporal["messages"][0]["content"][1]["video_url"]["frame_selection"]
    assert comparable_plain == comparable_temporal
    try:
        build_body(url, "bad", "other", "qwen3-native")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown policy was accepted")
    print("SELF-CHECK PASS: uniform/temporal requests differ only by explicit selection")
    print("REAL_MODEL_QUALITY: NOT_RUN")
    print("LIVE_LATENCY: NOT_RUN")
    return 0


def stream_chat(base_url: str, body: dict, timeout_s: float) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    ttft_s = None
    text_parts = []
    terminal = None
    done = False
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                done = True
                break
            frame = json.loads(payload)
            for choice in frame.get("choices", []):
                delta = choice.get("delta", {}).get("content")
                if isinstance(delta, str) and delta:
                    if ttft_s is None:
                        ttft_s = time.monotonic() - started
                    text_parts.append(delta)
                if choice.get("finish_reason") is not None:
                    terminal = choice["finish_reason"]
            unified = frame.get("unified")
            if isinstance(unified, dict) and unified.get("kind") in ("error", "cancelled"):
                terminal = unified["kind"]
    total_s = time.monotonic() - started
    if not done:
        raise RuntimeError("SSE stream ended before [DONE]")
    answer = "".join(text_parts)
    if terminal != "stop" or not answer.strip():
        raise RuntimeError(f"invalid live terminal/answer: terminal={terminal!r} chars={len(answer)}")
    return {"answer": answer, "ttft_s": ttft_s, "total_s": total_s,
            "finish_reason": terminal}


def run_live(args: argparse.Namespace) -> int:
    if not args.url or not args.video or not args.question or not args.out_dir:
        raise ValueError("live mode requires --url, --video, --question, and --out-dir")
    if len(args.question) not in (1, len(args.video)):
        raise ValueError("provide one --question per --video, or one shared question")
    if args.expect and len(args.expect) != len(args.video):
        raise ValueError("provide no --expect or exactly one per --video")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "model": args.model, "cases": []}
    quality_checks = []
    for index, video_arg in enumerate(args.video):
        path = Path(video_arg)
        url, digest, mime = encode_video(path)
        question = args.question[0] if len(args.question) == 1 else args.question[index]
        expected = args.expect[index] if args.expect else None
        case = {"video": path.name, "video_sha256": digest, "mime": mime,
                "question": question, "expected_substring": expected, "policies": {}}
        answers = {}
        for policy in ("uniform", "temporal"):
            body = build_body(url, question, policy, args.model,
                              args.motion_threshold, args.scene_threshold)
            result = stream_chat(args.url, body, args.timeout_s)
            answers[policy] = result["answer"]
            case["policies"][policy] = result
            if expected is not None:
                hit = expected.casefold() in result["answer"].casefold()
                result["expected_substring_hit"] = hit
                quality_checks.append(hit)
        case["answers_identical"] = answers["uniform"] == answers["temporal"]
        report["cases"].append(case)
    if args.expect:
        report["real_model_quality"] = "PASS" if all(quality_checks) else "FAIL"
        report["live_latency"] = "MEASURED"
    else:
        report["real_model_quality"] = "REVIEW_REQUIRED"
        report["live_latency"] = "MEASURED"
    report_path = out_dir / "video_temporal_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"cases": len(report["cases"]),
                      "real_model_quality": report["real_model_quality"],
                      "live_latency": report["live_latency"],
                      "report": str(report_path)}, indent=2))
    return 0 if not args.expect or all(quality_checks) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--url", default=os.environ.get("UMMINFER_URL", ""))
    parser.add_argument("--model", default="qwen3-native")
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument("--question", action="append", default=[])
    parser.add_argument("--expect", action="append", default=[])
    parser.add_argument("--motion-threshold", type=float, default=0.02)
    parser.add_argument("--scene-threshold", type=float, default=0.25)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args(argv)
    if args.self_check:
        return self_check()
    if os.environ.get("UMMINFER_LIVE_OPT_IN") != "1":
        print("REAL_MODEL_QUALITY: NOT_RUN")
        print("LIVE_LATENCY: NOT_RUN")
        print("Set UMMINFER_LIVE_OPT_IN=1 and provide the live arguments to run on SM90.")
        return 0
    try:
        return run_live(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"LIVE FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
