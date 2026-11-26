#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deterministic CPU-only long-video H.264 fixture generation and oracle.

The generated assets are synthetic input fixtures, never model output. Four
449-frame clips stress separated motion, scene cuts/static runs, near
duplicates, anchors, and H.264 B-frame decode/presentation order. FFmpeg and
libx264 are pinned by binary/version hashes in the manifest; any drift fails
closed instead of silently replacing the accepted source bytes.

An independent Python implementation recomputes native's 16x16 Rec.601 luma
and 16-bin histogram scores from decoded candidate RGB24 bytes. Expected
selection is generated for budgets 4, 8, and 16. The public native live
contract accepts only 8; 4 and 16 are deliberately CPU-oracle-only coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

SCHEMA_VERSION = 1
WIDTH = 96
HEIGHT = 64
NATIVE_WIDTH = 224
NATIVE_HEIGHT = 160
FPS = 2
FRAME_COUNT = 449  # ceil(449 / 64) = 8, so candidate 56 is source frame 448.
CANDIDATE_STRIDE = 8
CANDIDATE_COUNT = 57
CPU_BUDGETS = (4, 8, 16)
NATIVE_LIVE_BUDGET = 8
MOTION_THRESHOLD = 0.02
SCENE_THRESHOLD = 0.25
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
HEX256 = re.compile(r"[0-9a-f]{64}\Z")


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fnv1a64(data: bytes) -> str:
    value = FNV_OFFSET
    for byte in data:
        value ^= byte
        value = (value * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def run(args, *, data=None, timeout=180):
    return subprocess.run(args, input=data, check=True, capture_output=True,
                          timeout=timeout)


def solid(value):
    return bytearray([value, value, value] * (WIDTH * HEIGHT))


def rgb(color):
    return bytearray(list(color) * (WIDTH * HEIGHT))


def rectangle(frame, x0, y0, x1, y1, color):
    for y in range(max(0, y0), min(HEIGHT, y1)):
        for x in range(max(0, x0), min(WIDTH, x1)):
            offset = (y * WIDTH + x) * 3
            frame[offset:offset + 3] = bytes(color)


def anchor_tag(frame, source_index):
    """Unique low-area tags only where historical uniform samples can land.

    They make exact independent source binding unambiguous without turning the
    long repeated/static intervals into high-motion content.
    """
    if source_index <= 63:
        # The fps filter's rounding can choose either neighbouring source frame.
        # Tag every frame in this short opening window, not just multiples of
        # eight; five-pixel markers survive the native bicubic scale. Later
        # exact-static intervals remain completely untagged and byte-repeated.
        x = 1 + (source_index % 12) * 6
        y = 1 + (source_index // 12) * 6
        rectangle(frame, x, y, x + 5, y + 5,
                  ((source_index * 37) % 256, 255 - source_index * 3,
                   32 + (source_index * 5) % 200))


def source_frame(case_id, index):
    if case_id == "separated-motion":
        frame = rgb((18, 28, 46))
        if 64 <= index < 80:
            rectangle(frame, 12 + (index - 64), 20, 36 + (index - 64), 44,
                      (238, 30, 36))
        elif 176 <= index < 184:
            rectangle(frame, 50 - (index - 176) * 2, 18, 74 - (index - 176) * 2,
                      46, (30, 220, 70))
        elif 320 <= index < 336:
            rectangle(frame, 34, 8 + (index - 320), 62, 36 + (index - 320),
                      (245, 220, 30))
    elif case_id == "scene-static":
        if index < 112:
            frame = rgb((72, 72, 72))
        elif index < 224:
            frame = rgb((198, 36, 42))
        elif index < 336:
            frame = rgb((28, 62, 202))
        else:
            frame = rgb((30, 156, 76))
        # A small invariant shape gives the model a stable spatial reference.
        rectangle(frame, 8, 48, 20, 60, (235, 235, 235))
    elif case_id == "near-duplicate":
        frame = rgb((202, 202, 198))
        # One-pixel changes stay below the motion threshold. Exact repeats fill
        # 128..255 and 352..447, exercising bounded fallback and zero-threshold
        # duplicate handling without inventing motion peaks.
        shift = min(index, 127) % 19
        rectangle(frame, 18 + shift, 20, 38 + shift, 40, (184, 184, 180))
        if 128 <= index < 256 or 352 <= index < 448:
            frame = rgb((202, 202, 198))
            rectangle(frame, 25, 20, 45, 40, (184, 184, 180))
        if 288 <= index < 296:
            rectangle(frame, 32, 4, 80, 60, (206, 30, 196))
    elif case_id == "bframe-anchors":
        frame = rgb((242, 242, 242)) if index == 0 else rgb((20, 20, 24))
        if index == 0:
            rectangle(frame, 8, 8, 40, 56, (12, 12, 12))
        elif index == FRAME_COUNT - 1:
            frame = rgb((12, 12, 16))
            rectangle(frame, 56, 8, 88, 56, (245, 245, 245))
        else:
            level = 20 + (index * 31) % 90
            rectangle(frame, 10 + index % 37, 14, 30 + index % 37, 34,
                      (level, level, min(255, level + 20)))
            if index % 3 == 0:
                rectangle(frame, 64, 40, 84, 54, (level // 2, level // 2, level // 2))
    else:
        raise ValueError(f"unknown synthetic case {case_id}")
    anchor_tag(frame, index)
    return bytes(frame)


CASES = (
    {
        "id": "separated-motion", "language": "en",
        "question": "Which separated motion event appears first: red, green, or yellow? Answer with one color.",
        "expect_all": ("red",),
        "events": {"motion_peaks": [64, 176, 320], "scene_cuts": [],
                   "exact_static_intervals": [[80, 175], [336, 447]],
                   "uniform_visible_source_frames": [3, 11, 19, 27, 35, 43, 51, 59]},
    },
    {
        "id": "scene-static", "language": "zh",
        "question": "视频中第二次突然切换后的主色是什么？只回答颜色。",
        "expect_all": ("蓝",),
        "events": {"motion_peaks": [], "scene_cuts": [112, 224, 336],
                   "exact_static_intervals": [[64, 111], [120, 223], [232, 335]],
                   "uniform_visible_source_frames": [3, 11, 19, 27, 35, 43, 51, 59]},
    },
    {
        "id": "near-duplicate", "language": "ja",
        "question": "長い静止区間の後に現れる短い帯の色は何ですか？色だけで答えてください。",
        "expect_all": ("magenta", "マゼンタ", "紫"),
        "expect_any_group": True,
        "events": {"motion_peaks": [288], "scene_cuts": [288],
                   "exact_static_intervals": [[128, 255], [352, 447]],
                   "uniform_visible_source_frames": [3, 11, 19, 27, 35, 43, 51, 59]},
    },
    {
        "id": "bframe-anchors", "language": "es",
        "question": "¿Cuál es el color del primer cuadro y cuál el del último? Responde: blanco y negro.",
        "expect_all": ("blanco", "negro"),
        "events": {"motion_peaks": [], "scene_cuts": [8],
                   "exact_static_intervals": [],
                   "required_source_anchors": [0, 448],
                   "uniform_visible_source_frames": [3, 11, 19, 27, 35, 43, 51, 59]},
    },
)


def encoder_identity(ffmpeg):
    binary = Path(shutil.which(ffmpeg) or ffmpeg).resolve()
    version = run([str(binary), "-hide_banner", "-version"]).stdout
    first = version.decode("utf-8", errors="replace").splitlines()[0]
    if "ffmpeg version" not in first:
        raise ValueError("unable to pin FFmpeg version")
    return {"binary_name": "ffmpeg", "binary_sha256": sha256_file(binary),
            "version_sha256": sha256_bytes(version), "version": first,
            "codec": "libx264", "h264_profile": "recorded_per_case_ffprobe",
            "settings": ["-preset", "veryslow", "-crf", "1", "-threads", "1",
                         "-g", "32", "-bf", "2", "-pix_fmt", "yuv420p",
                         "-video_track_timescale", "2", "+bitexact"]}


def encode_case(ffmpeg, case_id, output):
    raw = b"".join(source_frame(case_id, index) for index in range(FRAME_COUNT))
    args = [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "pipe:0",
            "-an", "-map_metadata", "-1", "-c:v", "libx264", "-preset",
            "veryslow", "-crf", "1", "-threads", "1", "-g", "32", "-bf", "2",
            "-pix_fmt", "yuv420p", "-video_track_timescale", "2", "-fflags",
            "+bitexact", "-flags:v", "+bitexact", "-movflags", "+faststart",
            "-frames:v", str(FRAME_COUNT), str(output)]
    run(args, data=raw)


def decode_rgb(ffmpeg, path, vf, max_frames, *, vsync0=True):
    args = [ffmpeg, "-v", "info", "-i", str(path), "-an", "-sn", "-dn",
            "-vf", vf + ",showinfo"]
    if vsync0:
        args += ["-vsync", "0"]
    args += ["-frames:v", str(max_frames), "-f", "rawvideo", "-pix_fmt", "rgb24",
             "pipe:1"]
    result = run(args)
    frame_bytes = NATIVE_WIDTH * NATIVE_HEIGHT * 3
    if len(result.stdout) % frame_bytes:
        raise ValueError(f"partial decoded RGB frame in {path.name}")
    frames = [result.stdout[offset:offset + frame_bytes]
              for offset in range(0, len(result.stdout), frame_bytes)]
    timestamps = []
    for raw_line in result.stderr.decode("utf-8", errors="replace").splitlines():
        match = re.search(r"pts_time:([-+0-9.eE]+)", raw_line)
        if match and " n:" in raw_line:
            value = float(match.group(1))
            if math.isfinite(value):
                timestamps.append(value)
    if len(timestamps) != len(frames):
        raise ValueError(f"showinfo/RGB count mismatch for {path.name}: "
                         f"{len(timestamps)} != {len(frames)}")
    return frames, timestamps


def probe(ffprobe, path):
    result = run([ffprobe, "-v", "error", "-select_streams", "v:0",
                  "-show_entries", "stream=codec_name,profile,width,height,nb_frames,"
                  "avg_frame_rate,has_b_frames,pix_fmt:format=duration,size",
                  "-of", "json", str(path)])
    return json.loads(result.stdout)


def packet_order(ffprobe, path):
    result = run([ffprobe, "-v", "error", "-select_streams", "v:0",
                  "-show_packets", "-show_entries",
                  "packet=pts,pts_time,dts,dts_time,duration,duration_time,flags",
                  "-of", "json", str(path)])
    packets = json.loads(result.stdout)["packets"]
    if len(packets) != FRAME_COUNT:
        raise ValueError(f"packet count mismatch for {path.name}")
    pts = [int(packet["pts"]) for packet in packets]
    dts = [int(packet["dts"]) for packet in packets]
    nonmonotonic = any(later < earlier for earlier, later in zip(pts, pts[1:]))
    return packets, {"packet_count": len(packets),
                     "decode_order_pts_nonmonotonic": nonmonotonic,
                     "decode_order_dts_monotonic": all(
                         later >= earlier for earlier, later in zip(dts, dts[1:])),
                     "first_16_pts": pts[:16], "first_16_dts": dts[:16]}


def analyze_frames(frames):
    analyzed = []
    for frame in frames:
        if len(frame) != NATIVE_WIDTH * NATIVE_HEIGHT * 3:
            raise ValueError("oracle RGB buffer size mismatch")
        luma = []
        histogram = [0] * 16
        for ay in range(16):
            y0, y1 = ay * NATIVE_HEIGHT // 16, (ay + 1) * NATIVE_HEIGHT // 16
            for ax in range(16):
                x0, x1 = ax * NATIVE_WIDTH // 16, (ax + 1) * NATIVE_WIDTH // 16
                values = []
                for y in range(y0, y1):
                    for x in range(x0, x1):
                        offset = (y * NATIVE_WIDTH + x) * 3
                        values.append((77 * frame[offset] + 150 * frame[offset + 1]
                                       + 29 * frame[offset + 2]) >> 8)
                mean = sum(values) // len(values)
                luma.append(mean)
                histogram[mean * 16 // 256] += 1
        analyzed.append((luma, histogram))
    scores = [(0.0, 0.0)]
    for index in range(1, len(frames)):
        previous, current = analyzed[index - 1], analyzed[index]
        motion_sum = sum(abs(a - b) for a, b in zip(previous[0], current[0]))
        scene_sum = sum(abs(a - b) for a, b in zip(previous[1], current[1]))
        scores.append((motion_sum / (255.0 * 256.0), scene_sum / (2.0 * 256.0)))
    return scores


def select_temporal(scores, budget, *, motion_threshold=MOTION_THRESHOLD,
                    scene_threshold=SCENE_THRESHOLD):
    """Independent reference for the documented native priority/fallback rules."""
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("selection budget must be a positive integer")
    count = len(scores)
    target = min(count, budget)
    if count <= target:
        return list(range(count))
    selected = set()
    ordered = []

    def select(index):
        if len(ordered) < target and index not in selected:
            selected.add(index)
            ordered.append(index)

    select(0)
    select(count - 1)
    scenes = []
    index = 1
    while index < count:
        if not scores[index][1] > scene_threshold:
            index += 1
            continue
        end = index + 1
        best = index
        while end < count and scores[end][1] > scene_threshold:
            if scores[end][1] > scores[best][1] or (
                    scores[end][1] == scores[best][1]
                    and scores[end][0] > scores[best][0]):
                best = end
            end += 1
        scenes.append(best)
        index = end
    scenes.sort(key=lambda i: (-scores[i][1], -scores[i][0], i))
    for index in scenes:
        select(index)
    motion = [index for index in range(1, count)
              if scores[index][0] > motion_threshold]
    motion.sort(key=lambda i: (-scores[i][0], -scores[i][1], i))
    for index in motion:
        select(index)
    for slot in range(target):
        if len(ordered) >= target:
            break
        select((slot * (count - 1) + (target - 1) // 2) // (target - 1)
               if target > 1 else 0)
    for index in range(count):
        if len(ordered) >= target:
            break
        select(index)
    if len(ordered) != target:
        raise AssertionError("independent selector failed to fill its valid budget")
    return sorted(ordered)


def selection_record(selected, source_indices, timestamps, frames):
    return {"candidate_indices": selected,
            "source_frame_indices": [source_indices[index] for index in selected],
            "source_timestamps_seconds": [timestamps[index] for index in selected],
            "content_fnv1a64": [fnv1a64(frames[index]) for index in selected],
            "content_sha256": [sha256_bytes(frames[index]) for index in selected],
            "temporal2_pair_first_timestamps": [
                timestamps[selected[min(2 * slice_index, len(selected) - 1)]]
                for slice_index in range((len(selected) + 1) // 2)]}


def unique_source_binding(frames, source_frames, timestamps):
    mapping = {}
    for index, frame in enumerate(source_frames):
        mapping.setdefault(sha256_bytes(frame), []).append(index)
    source_indices = []
    for frame in frames:
        matches = mapping.get(sha256_bytes(frame), [])
        if len(matches) != 1:
            raise ValueError("uniform fixture frame has ambiguous/missing exact source "
                             f"binding: {len(matches)} matches")
        source_indices.append(matches[0])
    return selection_record(list(range(len(frames))), source_indices,
                            [timestamps[index] for index in source_indices], frames)


def build_case_oracle(ffmpeg, ffprobe, path):
    metadata = probe(ffprobe, path)
    stream = metadata["streams"][0]
    if stream["codec_name"] != "h264" or int(stream["nb_frames"]) != FRAME_COUNT:
        raise ValueError(f"fixture is not the pinned {FRAME_COUNT}-frame H.264 source")
    if stream["avg_frame_rate"] != f"{FPS}/1":
        raise ValueError("fixture source FPS drift")
    duration = float(metadata["format"]["duration"])
    source_frames, source_pts = decode_rgb(
        ffmpeg, path, f"scale={NATIVE_WIDTH}:{NATIVE_HEIGHT}:flags=bicubic",
        FRAME_COUNT)
    if len(source_frames) != FRAME_COUNT:
        raise ValueError("short source decode")
    candidates, candidate_pts = decode_rgb(
        ffmpeg, path, f"select='not(mod(n,{CANDIDATE_STRIDE}))',"
        f"scale={NATIVE_WIDTH}:{NATIVE_HEIGHT}:flags=bicubic", 64)
    source_indices = [index * CANDIDATE_STRIDE for index in range(len(candidates))]
    if len(candidates) != CANDIDATE_COUNT or source_indices[-1] != FRAME_COUNT - 1:
        raise ValueError("candidate stride no longer covers first/last source anchors")
    for position, source_index in enumerate(source_indices):
        if sha256_bytes(candidates[position]) != sha256_bytes(source_frames[source_index]):
            raise ValueError("candidate decode diverged from independent full decode")
        if abs(candidate_pts[position] - source_pts[source_index]) > 1e-9:
            raise ValueError("candidate PTS diverged from independent full decode")
    scores = analyze_frames(candidates)
    expected = {}
    for budget in CPU_BUDGETS:
        expected[str(budget)] = selection_record(
            select_temporal(scores, budget), source_indices, candidate_pts, candidates)
    sample_fps = max(0.25, NATIVE_LIVE_BUDGET / max(0.5, duration))
    uniform, _ = decode_rgb(
        ffmpeg, path, f"fps={sample_fps:.4f},"
        f"scale={NATIVE_WIDTH}:{NATIVE_HEIGHT}:flags=bicubic", NATIVE_LIVE_BUDGET)
    expected_uniform = unique_source_binding(uniform, source_frames, source_pts)
    packets, order = packet_order(ffprobe, path)
    content = []
    for index, (frame, timestamp) in enumerate(zip(source_frames, source_pts)):
        content.append({"source_frame_index": index,
                        "source_timestamp_seconds": timestamp,
                        "source_rgb96x64_sha256": sha256_bytes(source_frame(
                            path.stem, index)),
                        "decoded_native_rgb_sha256": sha256_bytes(frame),
                        "decoded_native_rgb_fnv1a64": fnv1a64(frame)})
    return {"ffprobe": metadata, "duration_seconds": duration,
            "candidate_stride": CANDIDATE_STRIDE,
            "candidate_count": len(candidates),
            "candidate_scores_motion_scene": [list(score) for score in scores],
            "expected_temporal": expected, "expected_uniform_8": expected_uniform,
            "source_content": content, "decode_order": order,
            "packets": packets}


def write_json(path, value):
    path.write_bytes(json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8")
                     + b"\n")


def generate_bundle(out_dir, ffmpeg="ffmpeg", ffprobe="ffprobe"):
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty fixture directory: {out}")
    (out / "fixtures").mkdir(parents=True, exist_ok=True)
    (out / "content").mkdir(exist_ok=True)
    encoder = encoder_identity(ffmpeg)
    cases = []
    checksums = {}
    for declaration in CASES:
        case_id = declaration["id"]
        video = out / "fixtures" / f"{case_id}.mp4"
        encode_case(ffmpeg, case_id, video)
        oracle = build_case_oracle(ffmpeg, ffprobe, video)
        if not oracle["decode_order"]["decode_order_pts_nonmonotonic"]:
            raise ValueError(f"pinned B-frame encoder produced no non-monotonic PTS: {case_id}")
        if oracle["expected_temporal"]["8"]["source_frame_indices"][:1] != [0] \
                or oracle["expected_temporal"]["8"]["source_frame_indices"][-1:] != [448]:
            raise ValueError(f"first/last source anchors missing for {case_id}")
        content_path = out / "content" / f"{case_id}.frames.jsonl"
        content_path.write_bytes(b"".join(canonical(row) + b"\n"
                                          for row in oracle["source_content"]))
        packet_path = out / "content" / f"{case_id}.packets.json"
        write_json(packet_path, oracle["packets"])
        case = {key: value for key, value in declaration.items()}
        case.update({
            "video": f"fixtures/{case_id}.mp4", "video_sha256": sha256_file(video),
            "video_bytes": video.stat().st_size,
            "content_frames": f"content/{case_id}.frames.jsonl",
            "content_frames_sha256": sha256_file(content_path),
            "decode_order_packets": f"content/{case_id}.packets.json",
            "decode_order_packets_sha256": sha256_file(packet_path),
            "ffprobe": oracle["ffprobe"], "duration_seconds": oracle["duration_seconds"],
            "candidate_stride": oracle["candidate_stride"],
            "candidate_count": oracle["candidate_count"],
            "candidate_scores_motion_scene": oracle["candidate_scores_motion_scene"],
            "expected_temporal": oracle["expected_temporal"],
            "expected_uniform_8": oracle["expected_uniform_8"],
            "decode_order": oracle["decode_order"],
        })
        cases.append(case)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "fixture_class": "SYNTHETIC_INPUT_FIXTURE_NOT_MODEL_OUTPUT",
        "source": {"frames": FRAME_COUNT, "fps": FPS, "width": WIDTH, "height": HEIGHT,
                   "duration_seconds": FRAME_COUNT / FPS},
        "native_decode": {"width": NATIVE_WIDTH, "height": NATIVE_HEIGHT,
                          "candidate_frames": 64, "selected_frames": 8,
                          "max_premerge_patches": 3584},
        "selection": {"motion_threshold": MOTION_THRESHOLD,
                      "scene_threshold": SCENE_THRESHOLD,
                      "analysis": "16x16 Rec.601 integer luma; 16-bin normalized L1 histogram",
                      "cpu_budgets": list(CPU_BUDGETS), "native_live_budgets": [8],
                      "unsupported_live_budgets": [4, 16]},
        "encoder": encoder,
        "default_uniform_semantics": "historical fps/scale decode; 0.25 fps floor; "
                                     "8 decoded frames over the opening 32 seconds",
        "temporal_semantics": "stride-8 source candidates 0..448; source PTS passthrough; "
                              "first/last anchors; decode-order output",
        "bframe_order_scope": "ffprobe packet decode order has non-monotonic PTS versus DTS; "
                              "decoded presentation order is not falsely labelled non-monotonic",
        "measurement": {"schema_version": 1, "warmup_repetitions": 1,
                        "measured_repetitions": 3, "arm_order": "interleaved",
                        "seed": 390039, "cooldown_ms": 0,
                        "arms": ["default", "uniform", "temporal"],
                        "modes": {"serial": {"concurrency": 1, "batch_size": 1},
                                  "batch4": {"concurrency": 4, "batch_size": 4}}},
        "claims": {"real_generation": "NOT_RUN", "live_latency": "NOT_RUN",
                   "perceptual_quality": "NOT_RUN", "generalized_quality_claim": False},
        "cases": cases,
    }
    write_json(out / "manifest.json", manifest)
    paths = sorted(path for path in out.rglob("*")
                   if path.is_file() and path.name != "SHA256SUMS")
    checksums = [f"{sha256_file(path)}  {path.relative_to(out).as_posix()}"
                 for path in paths]
    (out / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="ascii")
    return manifest


def validate_bundle(root, *, deep=True, ffmpeg="ffmpeg", ffprobe="ffprobe"):
    """Recompute hashes and, optionally, the complete independent decode oracle."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported long-fixture schema")
    lines = (root / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    declared = {}
    for line in lines:
        digest, relative = line.split("  ", 1)
        if not HEX256.fullmatch(digest) or relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError("malformed fixture SHA256SUMS entry")
        if relative in declared:
            raise ValueError("duplicate fixture checksum entry")
        declared[relative] = digest
    actual_paths = sorted(path.relative_to(root).as_posix() for path in root.rglob("*")
                          if path.is_file() and path.name != "SHA256SUMS")
    if set(declared) != set(actual_paths):
        raise ValueError("fixture artifact set differs from SHA256SUMS")
    for relative, digest in declared.items():
        if sha256_file(root / relative) != digest:
            raise ValueError(f"altered fixture checksum: {relative}")
    if manifest["encoder"] != encoder_identity(ffmpeg):
        raise ValueError("pinned FFmpeg/libx264 encoder drift")
    for case in manifest["cases"]:
        if case["video_sha256"] != sha256_file(root / case["video"]):
            raise ValueError(f"stale source video: {case['id']}")
        if deep:
            oracle = build_case_oracle(ffmpeg, ffprobe, root / case["video"])
            for key in ("expected_temporal", "expected_uniform_8", "candidate_scores_motion_scene",
                        "decode_order", "ffprobe"):
                if oracle[key] != case[key]:
                    raise ValueError(f"independent oracle mismatch: {case['id']}:{key}")
            content = [json.loads(line) for line in
                       (root / case["content_frames"]).read_text().splitlines()]
            if content != oracle["source_content"]:
                raise ValueError(f"independent source content mismatch: {case['id']}")
            packets = json.loads((root / case["decode_order_packets"]).read_bytes())
            if packets != oracle["packets"]:
                raise ValueError(f"decode-order packet evidence mismatch: {case['id']}")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="", help="new empty output directory")
    parser.add_argument("--check", default="", help="validate an existing fixture bundle")
    parser.add_argument("--shallow", action="store_true", help="checksum/encoder checks only")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args(argv)
    if bool(args.out) == bool(args.check):
        parser.error("choose exactly one of --out or --check")
    try:
        if args.out:
            manifest = generate_bundle(args.out, args.ffmpeg, args.ffprobe)
            validate_bundle(args.out, deep=not args.shallow,
                            ffmpeg=args.ffmpeg, ffprobe=args.ffprobe)
        else:
            manifest = validate_bundle(args.check, deep=not args.shallow,
                                       ffmpeg=args.ffmpeg, ffprobe=args.ffprobe)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"LONG FIXTURE FAIL: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"cases": len(manifest["cases"]), "frames_per_case": FRAME_COUNT,
                      "duration_seconds_per_case": FRAME_COUNT / FPS,
                      "cpu_budgets": list(CPU_BUDGETS), "native_live_budgets": [8],
                      "real_generation": "NOT_RUN", "live_latency": "NOT_RUN",
                      "perceptual_quality": "NOT_RUN", "result": "PASS"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
