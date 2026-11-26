#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate and independently validate the frozen temporal-clean-r1 fixture.

All generated videos and anchor cards are synthetic inputs, never model output.
The module is stdlib-first and uses the pinned FFmpeg/libx264 identity from the
accepted long-video fixture without changing that fixture or its scoring.
"""

from __future__ import annotations

import argparse
import binascii
import copy
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import sys
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import umminfer_long_video_fixture as base  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE = ROOT / "benchmarks" / "video-temporal-clean"
DEFAULT_FREEZE = ROOT / "benchmarks" / "video-temporal-clean.freeze.json"
SCHEMA_VERSION = 1
REVISION = "temporal-clean-r1"
WIDTH = base.WIDTH
HEIGHT = base.HEIGHT
NATIVE_WIDTH = base.NATIVE_WIDTH
NATIVE_HEIGHT = base.NATIVE_HEIGHT
FRAME_COUNT = base.FRAME_COUNT
FPS = base.FPS
CPU_BUDGETS = base.CPU_BUDGETS
ARMS = ("default", "uniform", "temporal")
MARKERS = ("triangle", "ring", "cross", "square", "stripes")
HEX256 = re.compile(r"[0-9a-f]{64}\Z")

VIDEO_DECLARATIONS = (
    {
        "id": "motion-clean",
        "events": {
            "separated_motion": [[8, 31], [176, 191], [320, 335]],
            "scene_cuts": [],
            "exact_static_intervals": [[80, 175], [336, 447]],
        },
    },
    {
        "id": "scene-clean",
        "events": {
            "separated_motion": [],
            "scene_cuts": [24, 112, 224, 336],
            "exact_static_intervals": [[64, 111], [128, 223], [240, 335], [352, 447]],
        },
    },
    {
        "id": "near-clean",
        "events": {
            "separated_motion": [288],
            "scene_cuts": [288],
            "exact_static_intervals": [[128, 255], [352, 447]],
            "near_duplicate_intervals": [[64, 127], [296, 351]],
        },
    },
    {
        "id": "bframe-clean",
        "events": {
            "separated_motion": [],
            "scene_cuts": [64],
            "exact_static_intervals": [],
            "required_source_anchors": [0, 448],
        },
    },
)


def canonical(value):
    return base.canonical(value)


def sha256_bytes(data):
    return base.sha256_bytes(data)


def sha256_file(path):
    return base.sha256_file(Path(path))


def write_json(path, value):
    path.write_bytes(json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8") + b"\n")


def blank(value=24):
    return bytearray([value] * (WIDTH * HEIGHT * 3))


def set_gray(frame, x, y, value):
    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
        offset = (y * WIDTH + x) * 3
        frame[offset:offset + 3] = bytes((value, value, value))


def rectangle(frame, x0, y0, x1, y1, value):
    for y in range(max(0, y0), min(HEIGHT, y1)):
        for x in range(max(0, x0), min(WIDTH, x1)):
            set_gray(frame, x, y, value)


def draw_marker(frame, marker, cx=48, cy=32, value=224):
    """Source-space semantic geometry; channels are always identical."""
    if marker == "triangle":
        vertices = ((cx, cy - 15), (cx - 15, cy + 14), (cx + 15, cy + 14))
        for y in range(cy - 15, cy + 15):
            for x in range(cx - 15, cx + 16):
                if point_in_triangle(x + 0.5, y + 0.5, vertices):
                    set_gray(frame, x, y, value)
    elif marker == "ring":
        for y in range(cy - 16, cy + 17):
            for x in range(cx - 16, cx + 17):
                radius = math.hypot(x + 0.5 - cx, y + 0.5 - cy)
                if 8.0 <= radius <= 15.0:
                    set_gray(frame, x, y, value)
    elif marker == "cross":
        rectangle(frame, cx - 4, cy - 15, cx + 4, cy + 15, value)
        rectangle(frame, cx - 15, cy - 4, cx + 15, cy + 4, value)
    elif marker == "square":
        rectangle(frame, cx - 13, cy - 13, cx + 13, cy + 13, value)
    elif marker == "stripes":
        for x in range(cx - 15, cx + 15):
            shade = value if ((x - (cx - 15)) // 4) % 2 == 0 else 24
            rectangle(frame, x, cy - 15, x + 1, cy + 15, shade)
    else:
        raise ValueError(f"unknown non-chromatic marker: {marker}")


def point_in_triangle(x, y, vertices):
    (ax, ay), (bx, by), (cx, cy) = vertices
    den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if den == 0:
        raise ValueError("degenerate triangle")
    a = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / den
    b = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / den
    c = 1.0 - a - b
    return a >= 0 and b >= 0 and c >= 0


def binding_tag(frame, source_index):
    """Machine-only luminance barcode, disjoint from the central answer ROI."""
    if source_index > 63:
        return
    # Twelve data bits plus a fixed 4-bit guard; never encodes semantic markers.
    code = source_index & 0xFFF
    for bit in range(16):
        x = 1 + bit % 8
        y = 55 + bit // 8
        on = (bit >= 12 and ((0b1010 >> (bit - 12)) & 1)) or (
            bit < 12 and ((code >> bit) & 1))
        rectangle(frame, x * 2, y, x * 2 + 2, y + 2, 192 if on else 48)


def central_card(frame):
    rectangle(frame, 24, 8, 72, 56, 24)


def source_frame(video_id, index):
    if not 0 <= index < FRAME_COUNT:
        raise ValueError("source frame index out of range")
    frame = blank(24)
    if video_id == "motion-clean":
        if 8 <= index < 32:
            draw_marker(frame, "triangle", cx=32 + (index - 8) * 4 // 3)
        elif 176 <= index < 192:
            draw_marker(frame, "square", cy=16 + (index - 176) * 2)
        elif 320 <= index < 336:
            draw_marker(frame, "ring", cx=64 - (index - 320) * 2)
    elif video_id == "scene-clean":
        scene = 0 if index < 24 else 1 if index < 112 else 2 if index < 224 else 3 if index < 336 else 4
        if scene:
            rectangle(frame, 0, 0, WIDTH, HEIGHT, 208 if scene % 2 else 40)
            central_card(frame)
            draw_marker(frame, ("triangle", "square", "ring", "stripes")[scene - 1])
    elif video_id == "near-clean":
        rectangle(frame, 0, 0, WIDTH, HEIGHT, 202)
        central_card(frame)
        if 288 <= index < 296:
            draw_marker(frame, "cross")
        else:
            draw_marker(frame, "ring")
        if not (128 <= index < 256 or 352 <= index < 448):
            shift = min(index, 127) % 19
            rectangle(frame, 4 + shift, 42, 12 + shift, 50, 184)
        else:
            rectangle(frame, 12, 42, 20, 50, 184)
    elif video_id == "bframe-clean":
        if index < 64:
            draw_marker(frame, "cross")
        elif index == FRAME_COUNT - 1:
            draw_marker(frame, "ring")
        else:
            level = 32 + (index * 37) % 160
            rectangle(frame, 4 + index % 31, 4, 12 + index % 31, 60, level)
            rectangle(frame, 72 - index % 23, 12, 84 - index % 23, 52, 224 - level // 2)
    else:
        raise ValueError(f"unknown clean video: {video_id}")
    binding_tag(frame, index)
    result = bytes(frame)
    if any(result[offset] != result[offset + 1] or result[offset] != result[offset + 2]
           for offset in range(0, len(result), 3)):
        raise AssertionError("chromatic source pixel escaped non-chromatic fixture contract")
    return result


def render_reference(marker):
    """Independent 40x40 anchor-card renderer, not a source/decoded crop."""
    if marker not in MARKERS:
        raise ValueError(f"unknown reference marker: {marker}")
    pixels = [24] * (40 * 40)

    def put(x, y, value=224):
        if 0 <= x < 40 and 0 <= y < 40:
            pixels[y * 40 + x] = value

    if marker == "triangle":
        vertices = ((20, 5), (5, 34), (35, 34))
        for y in range(40):
            for x in range(40):
                if point_in_triangle(x + 0.5, y + 0.5, vertices):
                    put(x, y)
    elif marker == "ring":
        for y in range(40):
            for x in range(40):
                radius = math.hypot(x + 0.5 - 20, y + 0.5 - 20)
                if 8 <= radius <= 15:
                    put(x, y)
    elif marker == "cross":
        for y in range(40):
            for x in range(40):
                if (16 <= x < 24 and 5 <= y < 35) or (5 <= x < 35 and 16 <= y < 24):
                    put(x, y)
    elif marker == "square":
        for y in range(7, 33):
            for x in range(7, 33):
                put(x, y)
    elif marker == "stripes":
        for y in range(5, 35):
            for x in range(5, 35):
                put(x, y, 224 if ((x - 5) // 4) % 2 == 0 else 24)
    return bytes(pixels)


def png_chunk(kind, payload):
    return (struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF))


def encode_gray_png(width, height, pixels):
    if len(pixels) != width * height:
        raise ValueError("gray PNG buffer size mismatch")
    raw = b"".join(b"\x00" + pixels[y * width:(y + 1) * width] for y in range(height))
    return (b"\x89PNG\r\n\x1a\n"
            + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
            + png_chunk(b"IDAT", zlib.compress(raw, 9)) + png_chunk(b"IEND", b""))


def rgb_to_gray(data):
    if len(data) % 3:
        raise ValueError("RGB buffer is not aligned")
    gray = bytearray(len(data) // 3)
    spread = 0
    for index in range(len(gray)):
        r, g, b = data[index * 3:index * 3 + 3]
        spread = max(spread, abs(r - g), abs(r - b), abs(g - b))
        gray[index] = (int(r) + int(g) + int(b)) // 3
    return bytes(gray), spread


def native_to_reference_luma(frame):
    """Area-average native x=64..159,y=30..129 into the independent 40x40 frame."""
    if len(frame) != NATIVE_WIDTH * NATIVE_HEIGHT * 3:
        raise ValueError("native RGB frame size mismatch")
    out = []
    for oy in range(40):
        y0, y1 = 30 + oy * 100 // 40, 30 + (oy + 1) * 100 // 40
        for ox in range(40):
            x0, x1 = 64 + ox * 96 // 40, 64 + (ox + 1) * 96 // 40
            total = 0
            count = 0
            for y in range(y0, y1):
                for x in range(x0, x1):
                    offset = (y * NATIVE_WIDTH + x) * 3
                    total += (frame[offset] + frame[offset + 1] + frame[offset + 2]) // 3
                    count += 1
            out.append(round(total / count))
    return out


def normalized_cross_correlation(left, right):
    if len(left) != len(right) or not left:
        raise ValueError("NCC shape mismatch")
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    left_energy = sum((a - mean_left) ** 2 for a in left)
    right_energy = sum((b - mean_right) ** 2 for b in right)
    if left_energy <= 0 or right_energy <= 0:
        return 0.0
    return numerator / math.sqrt(left_energy * right_energy)


def assess_anchor(frame, marker, preregistration):
    reference = list(render_reference(marker))
    observed = native_to_reference_luma(frame)
    ncc = normalized_cross_correlation(observed, reference)
    contrast = max(observed) - min(observed)
    rule = preregistration["anchor_visibility"]
    return {
        "marker": marker,
        "normalized_cross_correlation": round(ncc, 9),
        "luma_contrast": contrast,
        "visible": bool(ncc >= rule["minimum_normalized_cross_correlation"]
                        and contrast >= rule["minimum_luma_contrast"]),
    }


def in_intervals(index, intervals):
    return any(start <= index <= end for start, end in intervals)


def selection_frames(arm, expected_uniform, expected_temporal, uniform, candidates):
    if arm in ("default", "uniform"):
        return expected_uniform["source_frame_indices"], uniform
    return expected_temporal["source_frame_indices"], [candidates[index] for index in
                                                        expected_temporal["candidate_indices"]]


def build_anchor_audit(prereg_case, expected_uniform, expected_temporal, uniform, candidates,
                       preregistration, references):
    anchor = prereg_case["answer_anchor"]
    audit = {}
    for arm in ARMS:
        expected = expected_uniform if arm != "temporal" else expected_temporal
        source_indices, frames = selection_frames(arm, expected_uniform, expected_temporal,
                                                  uniform, candidates)
        observations = []
        for source_index, frame in zip(source_indices, frames):
            if not in_intervals(source_index, anchor["source_intervals"]):
                continue
            row = assess_anchor(frame, anchor["marker"], preregistration)
            row.update({
                "source_frame_index": source_index,
                "decoded_native_rgb_sha256": sha256_bytes(frame),
                "reference_anchor_sha256": references[anchor["marker"]]["rgb_sha256"],
            })
            observations.append(row)
        audit[arm] = {
            "selected_source_frame_indices": source_indices,
            "declared_anchor_source_frames_present": bool(observations),
            "answer_anchor_visible": any(row["visible"] for row in observations),
            "observations": observations,
            "expected_content_fnv1a64": expected["content_fnv1a64"],
        }
    return audit


def validate_anchor_rules(prereg_case, audit, preregistration):
    role = prereg_case["evaluation_role"]
    for arm in ARMS:
        row = audit[arm]
        if role == "quality_scored":
            if not row["answer_anchor_visible"]:
                raise ValueError(f"quality anchor is not visible: {prereg_case['id']}/{arm}")
        elif role == "selection_miss_control":
            if arm == "temporal":
                if not row["answer_anchor_visible"]:
                    raise ValueError(f"temporal control anchor missing: {prereg_case['id']}")
            elif row["declared_anchor_source_frames_present"] or row["answer_anchor_visible"]:
                raise ValueError(f"selection-miss control contains its late anchor: "
                                 f"{prereg_case['id']}/{arm}")
        else:
            raise ValueError("unknown evaluation role")
    minimum = preregistration["anchor_visibility"]["minimum_normalized_cross_correlation"]
    for row in audit.values():
        for observation in row["observations"]:
            if observation["visible"] and (
                    observation["normalized_cross_correlation"] < minimum
                    or observation["luma_contrast"] <
                    preregistration["anchor_visibility"]["minimum_luma_contrast"]):
                raise ValueError("anchor visibility arithmetic is inconsistent")


def validate_preregistration(prereg):
    if prereg.get("schema_version") != SCHEMA_VERSION or prereg.get("fixture_revision") != REVISION:
        raise ValueError("unsupported clean preregistration")
    if len(prereg["cases"]) != 8 or len({case["id"] for case in prereg["cases"]}) != 8:
        raise ValueError("clean preregistration requires eight unique questions")
    if [case["evaluation_role"] for case in prereg["cases"]].count("quality_scored") != 4:
        raise ValueError("clean preregistration requires four quality-scored cases")
    for case in prereg["cases"]:
        if not case["question"].strip() or not case["expect_all"]:
            raise ValueError("question and expected substrings must be nonempty")
        folded = case["question"].casefold()
        if any(term.casefold() in folded for term in case["expect_all"]):
            raise ValueError(f"expected answer leaks into prompt: {case['id']}")
        if case["answer_anchor"]["marker"] not in MARKERS:
            raise ValueError("answer anchor is not a registered non-chromatic marker")
    if prereg["request_contract"]["content_order"] != ["video_url", "text_question"]:
        raise ValueError("questions must follow video context")
    if prereg["measurement"]["measured_repetitions"] < 3:
        raise ValueError("at least three measured repetitions are required")
    if prereg["measurement"]["allowed_physical_gpus"] != [0, 1]:
        raise ValueError("clean live gate permits only physical GPUs 0 and 1")
    return prereg


def encode_video(ffmpeg, video_id, output):
    raw = b"".join(source_frame(video_id, index) for index in range(FRAME_COUNT))
    args = [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "pipe:0",
            "-an", "-map_metadata", "-1", "-c:v", "libx264", "-preset", "veryslow",
            "-crf", "1", "-threads", "1", "-g", "32", "-bf", "2", "-pix_fmt", "yuv420p",
            "-video_track_timescale", "2", "-fflags", "+bitexact", "-flags:v", "+bitexact",
            "-movflags", "+faststart", "-frames:v", str(FRAME_COUNT), str(output)]
    base.run(args, data=raw)


def build_video_oracle(ffmpeg, ffprobe, path):
    metadata = base.probe(ffprobe, path)
    stream = metadata["streams"][0]
    if stream["codec_name"] != "h264" or int(stream["nb_frames"]) != FRAME_COUNT:
        raise ValueError("clean fixture is not the pinned 449-frame H.264 source")
    if stream["avg_frame_rate"] != f"{FPS}/1":
        raise ValueError("clean fixture source FPS drift")
    duration = float(metadata["format"]["duration"])
    source_frames, source_pts = base.decode_rgb(
        ffmpeg, path, f"scale={NATIVE_WIDTH}:{NATIVE_HEIGHT}:flags=bicubic", FRAME_COUNT)
    candidates, candidate_pts = base.decode_rgb(
        ffmpeg, path, "select='not(mod(n,8))',"
        f"scale={NATIVE_WIDTH}:{NATIVE_HEIGHT}:flags=bicubic", 64)
    source_indices = [index * 8 for index in range(len(candidates))]
    if len(source_frames) != FRAME_COUNT or len(candidates) != 57:
        raise ValueError("clean source/candidate decode is incomplete")
    for position, source_index in enumerate(source_indices):
        if sha256_bytes(candidates[position]) != sha256_bytes(source_frames[source_index]):
            raise ValueError("candidate RGB diverged from independent full decode")
        if abs(candidate_pts[position] - source_pts[source_index]) > 1e-9:
            raise ValueError("candidate PTS diverged from independent full decode")
    scores = base.analyze_frames(candidates)
    expected_temporal = {str(budget): base.selection_record(
        base.select_temporal(scores, budget), source_indices, candidate_pts, candidates)
        for budget in CPU_BUDGETS}
    sample_fps = max(0.25, 8 / max(0.5, duration))
    uniform, _ = base.decode_rgb(
        ffmpeg, path, f"fps={sample_fps:.4f},"
        f"scale={NATIVE_WIDTH}:{NATIVE_HEIGHT}:flags=bicubic", 8)
    expected_uniform = base.unique_source_binding(uniform, source_frames, source_pts)
    packets, order = base.packet_order(ffprobe, path)
    content = [{"source_frame_index": index, "source_timestamp_seconds": timestamp,
                "source_rgb96x64_sha256": sha256_bytes(source_frame(path.stem, index)),
                "decoded_native_rgb_sha256": sha256_bytes(frame),
                "decoded_native_rgb_fnv1a64": base.fnv1a64(frame)}
               for index, (frame, timestamp) in enumerate(zip(source_frames, source_pts))]
    return {
        "ffprobe": metadata, "duration_seconds": duration,
        "candidate_scores_motion_scene": [list(score) for score in scores],
        "expected_temporal": expected_temporal, "expected_uniform_8": expected_uniform,
        "source_content": content, "decode_order": order, "packets": packets,
        "source_frames": source_frames, "candidates": candidates, "uniform": uniform,
    }


def selection_png(frames):
    if len(frames) != 8:
        raise ValueError("selection contact sheet requires eight frames")
    sheet = bytearray(NATIVE_WIDTH * 4 * NATIVE_HEIGHT * 2)
    max_spread = 0
    for index, frame in enumerate(frames):
        gray, spread = rgb_to_gray(frame)
        max_spread = max(max_spread, spread)
        x_offset = (index % 4) * NATIVE_WIDTH
        y_offset = (index // 4) * NATIVE_HEIGHT
        for y in range(NATIVE_HEIGHT):
            start = (y_offset + y) * NATIVE_WIDTH * 4 + x_offset
            sheet[start:start + NATIVE_WIDTH] = gray[y * NATIVE_WIDTH:(y + 1) * NATIVE_WIDTH]
    return encode_gray_png(NATIVE_WIDTH * 4, NATIVE_HEIGHT * 2, bytes(sheet)), max_spread


def generate_bundle(out_dir, *, preregistration_path=None, ffmpeg="ffmpeg", ffprobe="ffprobe"):
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty clean fixture directory: {out}")
    prereg_path = Path(preregistration_path or DEFAULT_BUNDLE / "preregistration.json")
    prereg_bytes = prereg_path.read_bytes()
    prereg = validate_preregistration(json.loads(prereg_bytes))
    (out / "fixtures").mkdir(parents=True, exist_ok=True)
    (out / "content").mkdir(exist_ok=True)
    (out / "anchors").mkdir(exist_ok=True)
    (out / "selections").mkdir(exist_ok=True)
    (out / "preregistration.json").write_bytes(prereg_bytes)
    references = {}
    for marker in MARKERS:
        raw = render_reference(marker)
        png = encode_gray_png(40, 40, raw)
        rgb_path = out / "anchors" / f"{marker}.rgb40x40"
        png_path = out / "anchors" / f"{marker}.png"
        rgb_path.write_bytes(raw)
        png_path.write_bytes(png)
        references[marker] = {
            "rgb": rgb_path.relative_to(out).as_posix(), "png": png_path.relative_to(out).as_posix(),
            "rgb_sha256": sha256_bytes(raw), "png_sha256": sha256_bytes(png),
            "renderer": "independent_40x40_geometric_luminance_renderer",
        }
    encoder = base.encoder_identity(ffmpeg)
    videos = {}
    decoded_max_spread = 0
    for declaration in VIDEO_DECLARATIONS:
        video_id = declaration["id"]
        video = out / "fixtures" / f"{video_id}.mp4"
        encode_video(ffmpeg, video_id, video)
        oracle = build_video_oracle(ffmpeg, ffprobe, video)
        if not oracle["decode_order"]["decode_order_pts_nonmonotonic"]:
            raise ValueError(f"pinned B-frame encoder produced monotonic packet PTS: {video_id}")
        temporal8 = oracle["expected_temporal"]["8"]
        if temporal8["source_frame_indices"][0] != 0 or temporal8["source_frame_indices"][-1] != 448:
            raise ValueError(f"first/last temporal anchors missing: {video_id}")
        content_path = out / "content" / f"{video_id}.frames.jsonl"
        content_path.write_bytes(b"".join(canonical(row) + b"\n" for row in oracle["source_content"]))
        packet_path = out / "content" / f"{video_id}.packets.json"
        write_json(packet_path, oracle["packets"])
        selections = {}
        for arm in ARMS:
            expected = oracle["expected_uniform_8"] if arm != "temporal" else temporal8
            source_indices, frames = selection_frames(
                arm, oracle["expected_uniform_8"], temporal8, oracle["uniform"], oracle["candidates"])
            png, spread = selection_png(frames)
            decoded_max_spread = max(decoded_max_spread, spread)
            png_path = out / "selections" / video_id / f"{arm}.png"
            json_path = out / "selections" / video_id / f"{arm}.json"
            png_path.parent.mkdir(parents=True, exist_ok=True)
            png_path.write_bytes(png)
            write_json(json_path, {
                "video_id": video_id, "arm": arm, "source_frame_indices": source_indices,
                "source_timestamps_seconds": expected["source_timestamps_seconds"],
                "decoded_native_rgb_sha256": [sha256_bytes(frame) for frame in frames],
                "content_fnv1a64": expected["content_fnv1a64"],
                "contact_sheet_layout": "4 columns x 2 rows; presentation order left-to-right, top-to-bottom",
            })
            selections[arm] = {
                "png": png_path.relative_to(out).as_posix(), "png_sha256": sha256_bytes(png),
                "record": json_path.relative_to(out).as_posix(),
                "record_sha256": sha256_file(json_path),
            }
        videos[video_id] = {
            **copy.deepcopy(declaration), "video": f"fixtures/{video_id}.mp4",
            "video_sha256": sha256_file(video), "video_bytes": video.stat().st_size,
            "content_frames": content_path.relative_to(out).as_posix(),
            "content_frames_sha256": sha256_file(content_path),
            "decode_order_packets": packet_path.relative_to(out).as_posix(),
            "decode_order_packets_sha256": sha256_file(packet_path),
            **{key: oracle[key] for key in ("ffprobe", "duration_seconds", "candidate_scores_motion_scene",
                                            "expected_temporal", "expected_uniform_8", "decode_order")},
            "candidate_stride": 8, "candidate_count": 57, "selections": selections,
        }
    cases = []
    for prereg_case in prereg["cases"]:
        video = videos[prereg_case["video_id"]]
        # Decode results are regenerated from the sealed video for each question's independent audit.
        oracle = build_video_oracle(ffmpeg, ffprobe, out / video["video"])
        audit = build_anchor_audit(prereg_case, oracle["expected_uniform_8"],
                                   oracle["expected_temporal"]["8"], oracle["uniform"],
                                   oracle["candidates"], prereg, references)
        validate_anchor_rules(prereg_case, audit, prereg)
        cases.append({**copy.deepcopy(prereg_case), "video": video["video"],
                      "video_sha256": video["video_sha256"], "duration_seconds": video["duration_seconds"],
                      "candidate_count": 57, "expected_uniform_8": video["expected_uniform_8"],
                      "expected_temporal": video["expected_temporal"], "anchor_audit": audit})
    measurement = copy.deepcopy(prereg["measurement"])
    measurement["arms"] = list(ARMS)
    manifest = {
        "schema_version": SCHEMA_VERSION, "fixture_revision": REVISION,
        "fixture_class": "SYNTHETIC_INPUT_FIXTURE_NOT_MODEL_OUTPUT",
        "preregistration_sha256": sha256_bytes(prereg_bytes),
        "source": copy.deepcopy(prereg["source"]),
        "native_decode": {"width": NATIVE_WIDTH, "height": NATIVE_HEIGHT,
                          "candidate_frames": 64, "selected_frames": 8,
                          "max_premerge_patches": 3584},
        "selection": copy.deepcopy(prereg["selection"]),
        "encoder": encoder,
        "marker_neutrality": {
            "source_all_frames_rgb_channels_equal": True,
            "decoded_selected_max_channel_spread": decoded_max_spread,
            "semantic_dimensions": ["shape", "binary pattern", "luminance"],
            "binding_marker_disjoint_from_answer_roi": True,
        },
        "anchor_visibility": copy.deepcopy(prereg["anchor_visibility"]),
        "anchor_references": references, "videos": videos, "cases": cases,
        "request_contract": copy.deepcopy(prereg["request_contract"]),
        "scoring": copy.deepcopy(prereg["scoring"]), "measurement": measurement,
        "claims": copy.deepcopy(prereg["claims_before_live"]),
    }
    if decoded_max_spread > 8:
        raise ValueError(f"decoded grayscale neutrality drift exceeds 8: {decoded_max_spread}")
    write_json(out / "manifest.json", manifest)
    paths = sorted(path for path in out.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(
        f"{sha256_file(path)}  {path.relative_to(out).as_posix()}\n" for path in paths), encoding="ascii")
    return manifest


def validate_bundle(root, *, deep=True, ffmpeg="ffmpeg", ffprobe="ffprobe"):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_bytes())
    prereg_bytes = (root / "preregistration.json").read_bytes()
    prereg = validate_preregistration(json.loads(prereg_bytes))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("fixture_revision") != REVISION:
        raise ValueError("unsupported clean fixture manifest")
    if manifest["preregistration_sha256"] != sha256_bytes(prereg_bytes):
        raise ValueError("manifest is bound to a different preregistration")
    declared = {}
    for line in (root / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        if not HEX256.fullmatch(digest) or relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError("malformed clean fixture checksum")
        if relative in declared:
            raise ValueError("duplicate clean fixture checksum")
        declared[relative] = digest
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name != "SHA256SUMS"}
    if set(declared) != actual:
        raise ValueError("clean fixture artifact set differs from SHA256SUMS")
    for relative, digest in declared.items():
        if sha256_file(root / relative) != digest:
            raise ValueError(f"altered clean fixture checksum: {relative}")
    if manifest["encoder"] != base.encoder_identity(ffmpeg):
        raise ValueError("pinned clean-fixture FFmpeg/libx264 encoder drift")
    for marker, reference in manifest["anchor_references"].items():
        raw = render_reference(marker)
        if raw != (root / reference["rgb"]).read_bytes() or sha256_bytes(raw) != reference["rgb_sha256"]:
            raise ValueError(f"independent anchor renderer mismatch: {marker}")
        png = encode_gray_png(40, 40, raw)
        if png != (root / reference["png"]).read_bytes() or sha256_bytes(png) != reference["png_sha256"]:
            raise ValueError(f"independent anchor PNG mismatch: {marker}")
    prereg_by_id = {case["id"]: case for case in prereg["cases"]}
    if [case["id"] for case in manifest["cases"]] != [case["id"] for case in prereg["cases"]]:
        raise ValueError("clean case order differs from preregistration")
    for case in manifest["cases"]:
        expected = prereg_by_id[case["id"]]
        for key in ("video_id", "language", "evaluation_role", "question", "expect_all",
                    "expect_any_group", "answer_anchor"):
            if case[key] != expected[key]:
                raise ValueError(f"frozen prompt/answer/role drift: {case['id']}:{key}")
        validate_anchor_rules(expected, case["anchor_audit"], prereg)
    for video_id, video in manifest["videos"].items():
        if sha256_file(root / video["video"]) != video["video_sha256"]:
            raise ValueError(f"stale clean source video: {video_id}")
        if not deep:
            continue
        oracle = build_video_oracle(ffmpeg, ffprobe, root / video["video"])
        for key in ("ffprobe", "duration_seconds", "candidate_scores_motion_scene",
                    "expected_temporal", "expected_uniform_8", "decode_order"):
            if oracle[key] != video[key]:
                raise ValueError(f"independent clean oracle mismatch: {video_id}:{key}")
        content = [json.loads(line) for line in (root / video["content_frames"]).read_text().splitlines()]
        if content != oracle["source_content"]:
            raise ValueError(f"independent clean source content mismatch: {video_id}")
        if json.loads((root / video["decode_order_packets"]).read_bytes()) != oracle["packets"]:
            raise ValueError(f"clean packet evidence mismatch: {video_id}")
        for arm in ARMS:
            selection = video["selections"][arm]
            expected = video["expected_uniform_8"] if arm != "temporal" else video["expected_temporal"]["8"]
            _, frames = selection_frames(arm, video["expected_uniform_8"], video["expected_temporal"]["8"],
                                         oracle["uniform"], oracle["candidates"])
            png, _ = selection_png(frames)
            if png != (root / selection["png"]).read_bytes():
                raise ValueError(f"selection contact sheet drift: {video_id}/{arm}")
            record = json.loads((root / selection["record"]).read_bytes())
            if record["decoded_native_rgb_sha256"] != [sha256_bytes(frame) for frame in frames] \
                    or record["source_frame_indices"] != expected["source_frame_indices"]:
                raise ValueError(f"selection source binding drift: {video_id}/{arm}")
        for case in manifest["cases"]:
            if case["video_id"] != video_id:
                continue
            audit = build_anchor_audit(prereg_by_id[case["id"]], oracle["expected_uniform_8"],
                                       oracle["expected_temporal"]["8"], oracle["uniform"],
                                       oracle["candidates"], prereg, manifest["anchor_references"])
            if audit != case["anchor_audit"]:
                raise ValueError(f"independent anchor visibility mismatch: {case['id']}")
    if deep:
        for video_id in manifest["videos"]:
            for index in range(FRAME_COUNT):
                source_frame(video_id, index)  # asserts exact RGB-channel neutrality
    return manifest


def write_freeze(bundle, freeze_path=DEFAULT_FREEZE):
    bundle = Path(bundle)
    manifest = validate_bundle(bundle, deep=False)
    receipt = {
        "schema_version": 1, "fixture_revision": REVISION,
        "frozen_on": "2026-11-26", "frozen_before_any_model_response": True,
        "model_response_inspected": False,
        "preregistration_sha256": sha256_file(bundle / "preregistration.json"),
        "manifest_sha256": sha256_file(bundle / "manifest.json"),
        "sha256sums_sha256": sha256_file(bundle / "SHA256SUMS"),
        "quality_case_ids": [case["id"] for case in manifest["cases"]
                             if case["evaluation_role"] == "quality_scored"],
        "selection_miss_control_ids": [case["id"] for case in manifest["cases"]
                                       if case["evaluation_role"] == "selection_miss_control"],
        "post_freeze_prompt_answer_or_scoring_changes": "forbidden",
    }
    write_json(Path(freeze_path), receipt)
    return receipt


def validate_freeze(bundle, freeze_path=DEFAULT_FREEZE):
    bundle = Path(bundle)
    receipt = json.loads(Path(freeze_path).read_bytes())
    if receipt.get("schema_version") != 1 or receipt.get("fixture_revision") != REVISION:
        raise ValueError("unsupported clean fixture freeze receipt")
    if receipt.get("frozen_before_any_model_response") is not True \
            or receipt.get("model_response_inspected") is not False:
        raise ValueError("clean fixture was not frozen before model inspection")
    for key, path in (("preregistration_sha256", "preregistration.json"),
                      ("manifest_sha256", "manifest.json"), ("sha256sums_sha256", "SHA256SUMS")):
        if not HEX256.fullmatch(receipt.get(key, "")) or receipt[key] != sha256_file(bundle / path):
            raise ValueError(f"clean fixture freeze mismatch: {key}")
    manifest = validate_bundle(bundle, deep=False)
    if receipt["quality_case_ids"] != [case["id"] for case in manifest["cases"]
                                      if case["evaluation_role"] == "quality_scored"] \
            or receipt["selection_miss_control_ids"] != [case["id"] for case in manifest["cases"]
            if case["evaluation_role"] == "selection_miss_control"]:
        raise ValueError("freeze case-role binding drift")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="")
    parser.add_argument("--check", default="")
    parser.add_argument("--freeze", default="")
    parser.add_argument("--check-freeze", default="")
    parser.add_argument("--preregistration", default="")
    parser.add_argument("--shallow", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args(argv)
    if bool(args.out) == bool(args.check):
        parser.error("choose exactly one of --out or --check")
    try:
        if args.out:
            manifest = generate_bundle(args.out, preregistration_path=args.preregistration or None,
                                       ffmpeg=args.ffmpeg, ffprobe=args.ffprobe)
            validate_bundle(args.out, deep=not args.shallow, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe)
            if args.freeze:
                write_freeze(args.out, args.freeze)
                validate_freeze(args.out, args.freeze)
        else:
            manifest = validate_bundle(args.check, deep=not args.shallow,
                                       ffmpeg=args.ffmpeg, ffprobe=args.ffprobe)
            if args.check_freeze:
                validate_freeze(args.check, args.check_freeze)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"TEMPORAL CLEAN FIXTURE FAIL: {error}", file=sys.stderr)
        return 1
    print(json.dumps({
        "result": "PASS", "fixture_revision": REVISION, "videos": len(manifest["videos"]),
        "quality_scored_cases": sum(case["evaluation_role"] == "quality_scored"
                                    for case in manifest["cases"]),
        "selection_miss_controls": sum(case["evaluation_role"] == "selection_miss_control"
                                       for case in manifest["cases"]),
        "real_generation": "NOT_RUN", "live_latency": "NOT_RUN",
        "fixture_substring_accuracy": "NOT_RUN", "perceptual_quality": "NOT_RUN",
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
