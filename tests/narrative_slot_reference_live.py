#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deferred SM90 text -> explicit image slot -> I2V continuity live gate.

CPU-only contract self-check:
  python3 -I tests/narrative_slot_reference_live.py --self-check

Later real-model gate (requires an explicitly supplied running service):
  UMMINFER_LIVE_OPT_IN=1 python3 -I tests/narrative_slot_reference_live.py \
    --url http://HOST:PORT --model MODEL --out-dir /tmp/umm-slot-reference-live

No service is started or contacted without UMMINFER_LIVE_OPT_IN=1. The offline
check uses byte fixtures only and never claims real generation or quality.
"""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from narrative_rl_fixtures import MP4, PNG, linked_events, sse_bytes  # noqa: E402


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def _mean_absolute_rgb(first, second):
    require(len(first) == len(second) and len(first) % 3 == 0,
            "RGB buffers must have equal nonzero sizes")
    require(bool(first), "RGB buffers must not be empty")
    return sum(abs(a - b) for a, b in zip(first, second)) / len(first)


def _decode_ppm(payload):
    # ffmpeg image2pipe PPM is P6 with a single binary frame here.
    require(payload.startswith(b"P6\n"), "expected binary PPM (P6)")
    header, raw = payload.split(b"\n255\n", 1)
    dimensions = header.splitlines()[1].split()
    require(len(dimensions) == 2, " malformed PPM dimensions")
    width, height = (int(value) for value in dimensions)
    require(len(raw) == width * height * 3, "PPM RGB byte count mismatch")
    return raw


def _rgb_frame(ffmpeg, media, selector=None):
    filters = []
    if selector:
        filters.append(selector)
    filters.extend([
        "scale=64:40:force_original_aspect_ratio=increase",
        "crop=64:40",
    ])
    command = [ffmpeg, "-v", "error", "-i", str(media)]
    if filters:
        command += ["-vf", ",".join(filters)]
    command += ["-frames:v", "1", "-f", "image2pipe", "-vcodec", "ppm", "-"]
    result = subprocess.run(command, check=True, capture_output=True, timeout=60)
    return _decode_ppm(result.stdout)


def measure_continuity(ffmpeg, source_png, video_mp4):
    source = _rgb_frame(ffmpeg, source_png)
    first = _rgb_frame(ffmpeg, video_mp4, "select=eq(n\\,0)")
    last = _rgb_frame(ffmpeg, video_mp4, "select=eq(n\\,28)")
    first_mae = _mean_absolute_rgb(source, first)
    motion_mae = _mean_absolute_rgb(first, last)
    # These are continuity/motion proxies, not a learned or human quality score.
    require(first_mae <= 48.0, f"I2V first frame diverged from source: MAE={first_mae:.3f}")
    require(motion_mae >= 1.0, f"I2V output appears static: MAE={motion_mae:.3f}")
    return {"first_frame_mae": round(first_mae, 6),
            "motion_mae": round(motion_mae, 6),
            "crop": "center_cover", "evaluation": "NUMERIC_PROXY"}


def validate_linked_events(events):
    require([event["seq"] for event in events] == list(range(len(events))),
            "unified event sequence is not gap-free")
    slots = [event for event in events if event["kind"] == "visual_slot"]
    require(len(slots) == 2, "expected one image and one I2V visual slot")
    linked = slots[1]
    require(linked["slot_id"] == "slot-1"
            and linked.get("continuity_mode") == "i2v_chain"
            and linked.get("anchor_slot_index") == 0
            and linked.get("anchor_explicit") is True
            and "anchor_sha256" not in linked,
            f"invalid explicit visual_slot linkage: {linked}")
    require("source=" not in linked["prompt"], "reference suffix leaked into prompt")
    grouped = {}
    for event in events:
        if event["kind"] == "artifact":
            grouped.setdefault(event["artifact"]["artifact_id"], []).append(event)
    require(len(grouped) == 2, "expected one source image and one I2V artifact")
    groups = list(grouped.values())
    image_events, video_events = groups
    require(image_events[0]["artifact"]["media_type"] == "image"
            and video_events[0]["artifact"]["media_type"] == "video",
            "source/I2V modalities are reversed or invalid")
    require(video_events[0]["seq"] > image_events[-1]["seq"],
            "I2V artifact predates the source image's final settled chunk")
    image = image_events[-1]["artifact"]
    video = video_events[-1]["artifact"]
    source_sha = image["sha256"]
    require(video.get("anchor_slot_index") == 0
            and video.get("anchor_explicit") is True
            and video.get("anchor_sha256") == source_sha,
            "I2V artifact does not bind the source SHA-256")
    require(events[-1]["kind"] == "complete", "linked stream did not complete")
    legs = events[-1]["legs"]
    linked_leg = next(leg for leg in legs if leg.get("slot_id") == "slot-1")
    require(linked_leg.get("state") == "completed"
            and linked_leg.get("anchor_sha256") == source_sha,
            "terminal I2V leg lacks source SHA-256")
    clean = "".join(event.get("text", "") for event in events
                    if event["kind"] == "vlm_delta")
    require("[[" not in clean and "source=" not in clean,
            "directive/reference leaked into clean prose")
    return {"source_sha256": source_sha,
            "i2v_sha256": video["sha256"],
            "anchor_slot_index": 0, "anchor_explicit": True,
            "clean_prose": True}


def self_check():
    summary = validate_linked_events(linked_events(delayed=True))
    raw = sse_bytes(linked_events(delayed=True))
    require(b"data: [DONE]\n\n" in raw, "offline SSE fixture lacks [DONE]")
    black, white = bytes([0, 0, 0]), bytes([255, 255, 255])
    require(_mean_absolute_rgb(black, black) == 0.0, "zero MAE mismatch")
    require(_mean_absolute_rgb(black, white) == 255.0, "RGB MAE mismatch")
    require(summary["source_sha256"] == hashlib.sha256(PNG).hexdigest(),
            "offline source digest mismatch")
    require(summary["i2v_sha256"] == hashlib.sha256(MP4).hexdigest(),
            "offline I2V digest mismatch")
    print("OFFLINE_CONTRACT: PASS (immediate/delayed linkage, SHA-256, clean prose)")
    print("REAL_GENERATION: NOT_RUN")
    print("PERCEPTUAL_QUALITY: NOT_RUN")


def _stream(url, body, timeout):
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    events = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            if not raw_line.startswith(b"data: "):
                continue
            payload = raw_line[6:].strip()
            if payload == b"[DONE]":
                break
            frame = json.loads(payload)
            unified = frame.get("unified")
            if unified:
                events.append(unified)
    return events


def _materialize(events, out_dir):
    grouped = {}
    for event in events:
        if event["kind"] != "artifact":
            continue
        desc = event["artifact"]
        varying = {"index", "data", "encoding", "bytes", "sha256"}
        identity = {key: value for key, value in desc.items() if key not in varying}
        group = grouped.setdefault(desc["artifact_id"], {
            "chunks": {}, "identity": identity, "total": desc["total"],
            "final": None,
        })
        require(group["identity"] == identity and group["total"] == desc["total"],
                "artifact metadata/total changed between chunks")
        group["chunks"][desc["index"]] = base64.b64decode(desc["data"], validate=True)
        if desc["index"] + 1 == desc["total"]:
            group["final"] = desc
    raw_by_media = {}
    for group in grouped.values():
        desc = group["final"]
        require(desc is not None, "artifact is missing its final descriptor")
        require(sorted(group["chunks"]) == list(range(group["total"])),
                "artifact chunks are incomplete")
        raw = b"".join(group["chunks"][index] for index in range(group["total"]))
        require(len(raw) == desc["bytes"]
                and hashlib.sha256(raw).hexdigest() == desc["sha256"],
                "artifact SHA-256/bytes mismatch")
        raw_by_media[desc["media_type"]] = raw
    require(set(raw_by_media) == {"image", "video"},
            "live stream did not produce exactly one image and one video")
    image_path = out_dir / "slot-0.png"
    video_path = out_dir / "slot-1.mp4"
    image_path.write_bytes(raw_by_media["image"])
    video_path.write_bytes(raw_by_media["video"])
    return image_path, video_path


def live(args):
    require(os.environ.get("UMMINFER_LIVE_OPT_IN") == "1",
            "LIVE_GENERATION: NOT_RUN (set UMMINFER_LIVE_OPT_IN=1 explicitly)")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    body = {
        "model": args.model,
        "stream": True,
        "messages": [{"role": "user", "content":
            "Create a two-beat visual story. First output one standalone "
            "[[image: a red lantern above a misty mountain lake at dawn]] marker. "
            "Then output one standalone [[video: the lantern sways gently as the "
            "camera slowly orbits | source=slot-0]] marker. Keep prose minimal."}],
        "modalities": ["text", "image", "video"],
        "image": {"size": "512x512", "steps": 4, "seed": 42,
                  "output_format": "png"},
        "video": {"size": "704x480", "num_frames": 29, "steps": 20,
                  "guidance_scale": 5.0, "seed": 42, "output_format": "mp4"},
        "narrative": {"enabled": True, "visual_mode": "auto", "max_slots": 2,
                      "allow_text_only": False, "continuity_mode": "i2v_chain"},
        "seed": 42,
    }
    events = _stream(args.url, body, args.timeout)
    summary = validate_linked_events(events)
    image_path, video_path = _materialize(events, out_dir)
    probe = subprocess.run([
        args.ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,nb_frames",
        "-of", "json", str(video_path),
    ], check=True, capture_output=True, text=True, timeout=30)
    stream = json.loads(probe.stdout)["streams"][0]
    require(stream["codec_name"] == "h264"
            and (stream["width"], stream["height"]) == (704, 480)
            and int(stream["nb_frames"]) == 29,
            f"unexpected real I2V video contract: {stream}")
    summary.update(measure_continuity(args.ffmpeg, image_path, video_path))
    summary.update({"evidence_class": "SM90_LIVE", "video_stream": stream,
                    "request_sha256": hashlib.sha256(
                        json.dumps(body, sort_keys=True).encode()).hexdigest()})
    (out_dir / "report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    print("REAL_GENERATION: PASS")
    print("PERCEPTUAL_QUALITY: NUMERIC_PROXY_ONLY (human quality NOT_RUN)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--url")
    parser.add_argument("--model")
    parser.add_argument("--out-dir", default="/tmp/umm-slot-reference-live")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    if not args.url or not args.model:
        parser.error("live mode requires --url and --model (or use --self-check)")
    live(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
