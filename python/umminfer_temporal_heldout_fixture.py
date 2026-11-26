#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate and validate the frozen temporal-heldout-r1 synthetic fixture.

No natural-video or model-output claim is made. The accepted motion-run selector
is evaluated unchanged; the four Task61 clean cases are neither copied into nor
counted by this fixture.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import umminfer_long_video_fixture as base  # noqa: E402
import umminfer_temporal_clean_fixture as clean  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE = ROOT / "benchmarks" / "video-temporal-heldout"
DEFAULT_FREEZE = ROOT / "benchmarks" / "video-temporal-heldout.freeze.json"
SCHEMA_VERSION = 1
REVISION = "temporal-heldout-r1"
WIDTH, HEIGHT = base.WIDTH, base.HEIGHT
NATIVE_WIDTH, NATIVE_HEIGHT = base.NATIVE_WIDTH, base.NATIVE_HEIGHT
FRAME_COUNT, FPS = base.FRAME_COUNT, base.FPS
BUDGETS = (4, 8, 16)
ARMS = ("default", "uniform", "temporal")
ARM_BUDGETS = (("default", 8),) + tuple(
    (arm, budget) for arm in ("uniform", "temporal") for budget in BUDGETS)
HEX256 = re.compile(r"[0-9a-f]{64}\Z")
HISTORICAL_TASK61 = {"motion-first-en", "scene-cut-zh", "near-static-ja", "bframe-first-es"}

VIDEO_DECLARATIONS = (
    {
        "id": "separated-events",
        "events": {
            "separated_motion": [[8, 23], [168, 183], [336, 351]],
            "scene_cuts": [],
            "exact_static_intervals": [[64, 159], [240, 327]],
        },
        "minimum_motion_runs": 3,
    },
    {
        "id": "contiguous-run",
        "events": {
            "contiguous_motion": [[32, 55], [56, 79], [80, 103]],
            "scene_cuts": [],
            "exact_static_intervals": [[112, 447]],
        },
        "minimum_longest_motion_run": 7,
    },
    {
        "id": "scene-boundaries",
        "events": {
            "separated_motion": [],
            "scene_cuts": [48, 144, 256, 368],
            "exact_static_intervals": [[64, 143], [160, 255], [272, 367], [384, 447]],
        },
    },
    {
        "id": "static-square",
        "events": {
            "separated_motion": [280],
            "scene_cuts": [280],
            "exact_static_intervals": [[64, 255], [304, 448]],
        },
    },
    {
        "id": "bframe-anchors-heldout",
        "events": {
            "separated_motion": [],
            "scene_cuts": [64],
            "exact_static_intervals": [],
            "required_source_anchors": [0, 448],
        },
    },
    {
        "id": "order-after",
        "events": {
            "separated_motion": [[40, 55], [200, 215]],
            "scene_cuts": [],
            "exact_static_intervals": [[64, 191], [224, 447]],
        },
        "minimum_motion_runs": 2,
    },
    {
        "id": "five-events",
        "events": {
            "separated_motion": [[24, 39], [104, 119], [184, 199], [264, 279], [384, 399]],
            "scene_cuts": [],
            "exact_static_intervals": [[64, 95], [128, 175], [208, 255], [288, 375], [408, 447]],
        },
        "minimum_motion_runs": 5,
    },
    {
        "id": "single-flash",
        "events": {
            "separated_motion": [312],
            "scene_cuts": [312],
            "exact_static_intervals": [[64, 287], [328, 447]],
        },
        "minimum_motion_runs": 1,
    },
)


def canonical(value):
    return base.canonical(value)


def sha256_bytes(data):
    return base.sha256_bytes(data)


def sha256_file(path):
    return base.sha256_file(Path(path))


def write_json(path, value):
    path = Path(path)
    path.write_bytes(json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8") + b"\n")


def in_intervals(index, intervals):
    return any(start <= index <= end for start, end in intervals)


def render_source(video_id, index):
    if not 0 <= index < FRAME_COUNT:
        raise ValueError("source frame index out of range")
    frame = clean.blank(24)
    label = "none"
    if video_id == "separated-events":
        if 8 <= index < 24:
            label = "square"
            clean.draw_marker(frame, label, cx=28 + (index - 8) * 2, cy=32)
        elif 168 <= index < 184:
            label = "triangle"
            clean.draw_marker(frame, label, cx=68 - (index - 168) * 3, cy=32)
        elif 336 <= index < 352:
            label = "ring"
            clean.draw_marker(frame, label, cx=32 + (index - 336) * 2, cy=32)
    elif video_id == "contiguous-run":
        if 32 <= index < 56:
            label = "cross"
            clean.draw_marker(frame, label, cx=26 + (index - 32) * 2, cy=32)
        elif 56 <= index < 80:
            label = "triangle"
            clean.draw_marker(frame, label, cx=66 - (index - 56) * 2, cy=32)
        elif 80 <= index < 104:
            label = "square"
            clean.draw_marker(frame, label, cx=20 + (index - 80) * 3, cy=32)
    elif video_id == "scene-boundaries":
        scene = 0 if index < 48 else 1 if index < 144 else 2 if index < 256 else 3 if index < 368 else 4
        if scene:
            clean.rectangle(frame, 0, 0, WIDTH, HEIGHT, 208 if scene % 2 else 40)
            clean.central_card(frame)
            label = ("triangle", "square", "cross", "stripes")[scene - 1]
            clean.draw_marker(frame, label)
    elif video_id == "static-square":
        clean.rectangle(frame, 0, 0, WIDTH, HEIGHT, 40 if 280 <= index < 304 else 202)
        clean.central_card(frame)
        label = "cross" if 280 <= index < 304 else "square"
        clean.draw_marker(frame, label)
    elif video_id == "bframe-anchors-heldout":
        if index >= 64:
            clean.rectangle(frame, 0, 0, WIDTH, HEIGHT, 208)
        if index < 64:
            label = "cross"
            clean.draw_marker(frame, label)
        elif index == FRAME_COUNT - 1:
            label = "ring"
            clean.draw_marker(frame, label)
        else:
            level = 40 + (index * 53) % 144
            clean.rectangle(frame, 3 + index % 37, 6, 10 + index % 37, 58, level)
            clean.rectangle(frame, 86 - index % 29, 10, 93 - index % 29, 54, 232 - level // 2)
    elif video_id == "order-after":
        if 40 <= index < 56:
            label = "triangle"
            clean.draw_marker(frame, label, cx=28 + (index - 40) * 2, cy=32)
        elif 200 <= index < 216:
            label = "square"
            clean.draw_marker(frame, label, cx=64 - (index - 200) * 2, cy=32)
    elif video_id == "five-events":
        events = ((24, 39), (104, 119), (184, 199), (264, 279), (384, 399))
        for event_index, (start, end) in enumerate(events):
            if start <= index < end:
                label = "cross"
                offset = index - start
                cx = 24 + offset * 3 if event_index % 2 == 0 else 72 - offset * 3
                clean.draw_marker(frame, label, cx=cx, cy=20 + event_index * 3)
                break
    elif video_id == "single-flash":
        clean.rectangle(frame, 0, 0, WIDTH, HEIGHT, 40 if 312 <= index < 320 else 202)
        clean.central_card(frame)
        label = "ring" if 312 <= index < 320 else "cross"
        clean.draw_marker(frame, label)
    else:
        raise ValueError(f"unknown held-out video: {video_id}")
    clean.binding_tag(frame, index)
    result = bytes(frame)
    if any(result[offset] != result[offset + 1] or result[offset] != result[offset + 2]
           for offset in range(0, len(result), 3)):
        raise AssertionError("chromatic source pixel escaped the synthetic fixture contract")
    return result, label


def source_frame(video_id, index):
    return render_source(video_id, index)[0]


def semantic_label(video_id, index):
    return render_source(video_id, index)[1]


def threshold_runs(scores, threshold, component):
    runs = []
    index = 1
    while index < len(scores):
        if not scores[index][component] > threshold:
            index += 1
            continue
        end = index
        while end < len(scores) and scores[end][component] > threshold:
            end += 1
        runs.append(list(range(index, end)))
        index = end
    return runs


def select_motion_run(scores, budget, *, motion_threshold=0.02, scene_threshold=0.25):
    """Producer oracle for the already accepted native motion-run algorithm."""
    if budget not in BUDGETS or not scores:
        raise ValueError("motion-run oracle requires candidates and a 4/8/16 budget")
    target = min(len(scores), budget)
    used, ordered = set(), []

    def select(index):
        if len(ordered) < target and index not in used:
            used.add(index)
            ordered.append(index)

    select(0)
    select(len(scores) - 1)
    scenes = [min(run, key=lambda i: (-scores[i][1], -scores[i][0], i))
              for run in threshold_runs(scores, scene_threshold, 1)]
    scenes.sort(key=lambda i: (-scores[i][1], -scores[i][0], i))
    for index in scenes:
        select(index)
    motions = [min(run, key=lambda i: (-scores[i][0], -scores[i][1], i))
               for run in threshold_runs(scores, motion_threshold, 0)]
    motions.sort(key=lambda i: (-scores[i][0], -scores[i][1], i))
    for index in motions:
        select(index)
    for slot in range(target):
        if len(ordered) >= target:
            break
        select((slot * (len(scores) - 1) + (target - 1) // 2) // (target - 1))
    for index in range(len(scores)):
        if len(ordered) >= target:
            break
        select(index)
    if len(ordered) != target:
        raise AssertionError("accepted-selector oracle failed to fill its budget")
    return sorted(ordered)


def encode_video(ffmpeg, video_id, output):
    raw = b"".join(source_frame(video_id, index) for index in range(FRAME_COUNT))
    base.run([ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
              "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "pipe:0",
              "-an", "-map_metadata", "-1", "-c:v", "libx264", "-preset", "veryslow",
              "-crf", "1", "-threads", "1", "-g", "32", "-bf", "2", "-pix_fmt", "yuv420p",
              "-video_track_timescale", "2", "-fflags", "+bitexact", "-flags:v", "+bitexact",
              "-movflags", "+faststart", "-frames:v", str(FRAME_COUNT), str(output)], data=raw)


def explicit_uniform(ffmpeg, path, source_frames, source_pts, budget):
    indices = [(slot * (FRAME_COUNT - 1) + (budget - 1) // 2) // (budget - 1)
               for slot in range(budget)]
    expression = "+".join(f"eq(n,{index})" for index in indices)
    frames, timestamps = base.decode_rgb(
        ffmpeg, path, f"select='{expression}',scale={NATIVE_WIDTH}:{NATIVE_HEIGHT}:flags=bicubic",
        budget)
    if len(frames) != budget or len(timestamps) != budget:
        raise ValueError("explicit uniform oracle returned a short decode")
    for position, source_index in enumerate(indices):
        if sha256_bytes(frames[position]) != sha256_bytes(source_frames[source_index]) \
                or timestamps[position] != source_pts[source_index]:
            raise ValueError("explicit uniform RGB/PTS differs from full-source decode")
    return base.selection_record(list(range(budget)), indices, timestamps, frames)


def build_video_oracle(ffmpeg, ffprobe, path):
    metadata = base.probe(ffprobe, path)
    stream = metadata["streams"][0]
    if stream["codec_name"] != "h264" or int(stream["nb_frames"]) != FRAME_COUNT \
            or stream["avg_frame_rate"] != f"{FPS}/1" or int(stream["has_b_frames"]) < 1:
        raise ValueError("held-out source is not the pinned 449-frame B-frame H.264 input")
    duration = float(metadata["format"]["duration"])
    source_frames, source_pts = base.decode_rgb(
        ffmpeg, path, f"scale={NATIVE_WIDTH}:{NATIVE_HEIGHT}:flags=bicubic", FRAME_COUNT)
    candidates, candidate_pts = base.decode_rgb(
        ffmpeg, path, f"select='not(mod(n,8))',scale={NATIVE_WIDTH}:{NATIVE_HEIGHT}:flags=bicubic", 64)
    source_indices = list(range(0, FRAME_COUNT, 8))
    if len(source_frames) != FRAME_COUNT or len(candidates) != 57:
        raise ValueError("held-out source/candidate decode is incomplete")
    for position, source_index in enumerate(source_indices):
        if sha256_bytes(candidates[position]) != sha256_bytes(source_frames[source_index]) \
                or candidate_pts[position] != source_pts[source_index]:
            raise ValueError("candidate RGB/PTS differs from independent full decode")
    scores = base.analyze_frames(candidates)
    temporal = {str(budget): base.selection_record(
        select_motion_run(scores, budget), source_indices, candidate_pts, candidates)
        for budget in BUDGETS}
    uniform = {}
    for budget in (4, 16):
        uniform[str(budget)] = explicit_uniform(
            ffmpeg, path, source_frames, source_pts, budget)
    sample_fps = max(0.25, 8 / max(0.5, duration))
    historical, _ = base.decode_rgb(
        ffmpeg, path, f"fps={sample_fps:.4f},scale={NATIVE_WIDTH}:{NATIVE_HEIGHT}:flags=bicubic", 8)
    uniform["8"] = base.unique_source_binding(historical, source_frames, source_pts)
    packets, order = base.packet_order(ffprobe, path)
    content = [{
        "source_frame_index": index,
        "source_timestamp_seconds": source_pts[index],
        "semantic_label": semantic_label(path.stem, index),
        "source_rgb96x64_sha256": sha256_bytes(source_frame(path.stem, index)),
        "decoded_native_rgb_sha256": sha256_bytes(source_frames[index]),
        "decoded_native_rgb_fnv1a64": base.fnv1a64(source_frames[index]),
    } for index in range(FRAME_COUNT)]
    return {
        "ffprobe": metadata, "duration_seconds": duration,
        "candidate_scores_motion_scene": [list(score) for score in scores],
        "expected_temporal": temporal, "expected_uniform": uniform,
        "source_content": content, "decode_order": order, "packets": packets,
        "source_frames": source_frames, "candidates": candidates,
        "uniform_frames": {key: None for key in uniform},
        "historical_uniform_frames": historical,
    }


def selection_frames(oracle, arm, budget):
    if arm == "temporal":
        expected = oracle["expected_temporal"][str(budget)]
        return [oracle["candidates"][index] for index in expected["candidate_indices"]]
    expected = oracle["expected_uniform"][str(budget)]
    if budget == 8:
        return oracle["historical_uniform_frames"]
    return [oracle["source_frames"][index] for index in expected["source_frame_indices"]]


def selection_png(frames):
    if len(frames) not in BUDGETS:
        raise ValueError("selection contact sheet has an invalid budget")
    columns, rows = 4, (len(frames) + 3) // 4
    sheet = bytearray(NATIVE_WIDTH * columns * NATIVE_HEIGHT * rows)
    max_spread = 0
    for index, frame in enumerate(frames):
        gray, spread = clean.rgb_to_gray(frame)
        max_spread = max(max_spread, spread)
        x_offset, y_offset = (index % 4) * NATIVE_WIDTH, (index // 4) * NATIVE_HEIGHT
        for y in range(NATIVE_HEIGHT):
            start = (y_offset + y) * NATIVE_WIDTH * columns + x_offset
            sheet[start:start + NATIVE_WIDTH] = gray[y * NATIVE_WIDTH:(y + 1) * NATIVE_WIDTH]
    return clean.encode_gray_png(NATIVE_WIDTH * columns, NATIVE_HEIGHT * rows, bytes(sheet)), max_spread


def anchor_observation(anchor, expected):
    selected = expected["source_frame_indices"]
    if anchor["kind"] == "marker":
        matching = [index for index in selected if in_intervals(index, anchor["source_intervals"])]
        present = bool(matching)
        required_count = 1
    elif anchor["kind"] == "interval_count":
        matching = [[start, end] for start, end in anchor["source_intervals"]
                    if any(start <= index <= end for index in selected)]
        present = len(matching) == len(anchor["source_intervals"])
        required_count = len(anchor["source_intervals"])
    elif anchor["kind"] == "exact_frames":
        matching = [index for index in anchor["source_frames"] if index in selected]
        present = len(matching) == len(anchor["source_frames"])
        required_count = len(anchor["source_frames"])
    else:
        raise ValueError("unknown held-out answer-anchor kind")
    return {
        "kind": anchor["kind"], "selected_source_frame_indices": selected,
        "matching_source_evidence": matching, "required_evidence_count": required_count,
        "observed_evidence_count": len(matching), "answer_anchor_present": present,
        "expected_content_fnv1a64": expected["content_fnv1a64"],
    }


def build_anchor_audit(prereg_case, video):
    result = {}
    for arm, budget in ARM_BUDGETS:
        expected = video["expected_temporal"][str(budget)] if arm == "temporal" \
            else video["expected_uniform"][str(budget)]
        result[f"{arm}-{budget}"] = anchor_observation(prereg_case["answer_anchor"], expected)
    for budget in prereg_case["answer_anchor"]["required_temporal_budgets"]:
        if not result[f"temporal-{budget}"]["answer_anchor_present"]:
            raise ValueError(f"required temporal answer evidence is absent: {prereg_case['id']}/{budget}")
    return result


def validate_preregistration(prereg):
    if prereg.get("schema_version") != SCHEMA_VERSION or prereg.get("fixture_revision") != REVISION:
        raise ValueError("unsupported held-out preregistration")
    cases = prereg.get("cases", [])
    if len(cases) != 10 or len({case["id"] for case in cases}) != 10:
        raise ValueError("held-out population requires ten unique quality cases")
    if any(case["evaluation_role"] != "quality_scored" for case in cases):
        raise ValueError("all ten new cases must remain in the quality denominator")
    if HISTORICAL_TASK61 & {case["id"] for case in cases}:
        raise ValueError("Task61 historical controls entered the held-out population")
    video_ids = {declaration["id"] for declaration in VIDEO_DECLARATIONS}
    if prereg["source"]["video_count"] != len(video_ids) \
            or prereg["source"]["natural_video_claim"] is not False:
        raise ValueError("held-out source identity/natural-video boundary drift")
    for case in cases:
        if case["video_id"] not in video_ids or not case["question"].strip() \
                or not case["expect_all"]:
            raise ValueError("held-out case identity/question/answer is incomplete")
        folded = case["question"].casefold()
        if any(term.casefold() in folded for term in case["expect_all"]):
            raise ValueError(f"expected answer leaks into prompt: {case['id']}")
        if not case["answer_anchor"]["required_temporal_budgets"] or any(
                budget not in BUDGETS for budget in
                case["answer_anchor"]["required_temporal_budgets"]):
            raise ValueError("required temporal budgets are invalid")
    request = prereg["request_contract"]
    if request["content_order"] != ["video_url", "text_question"] \
            or request["seed"] != 510066 or request["max_tokens"] != 64:
        raise ValueError("held-out request contract drift")
    measurement = prereg["measurement"]
    if measurement["allowed_physical_gpus"] != [1] \
            or measurement["requests_total"] != 560 \
            or measurement["measured_answers_per_arm_budget"] != 60 \
            or measurement["expected_parity_checks"] != 500:
        raise ValueError("held-out GPU/population contract drift")
    return prereg


def validate_event_contract(declaration, oracle):
    video_id = declaration["id"]
    scores = oracle["candidate_scores_motion_scene"]
    runs = threshold_runs(scores, 0.02, 0)
    minimum = declaration.get("minimum_motion_runs")
    if minimum is not None and len(runs) < minimum:
        raise ValueError(f"motion-run coverage failed: {video_id}: {len(runs)} < {minimum}")
    longest = declaration.get("minimum_longest_motion_run")
    if longest is not None and max(map(len, runs), default=0) < longest:
        raise ValueError(f"contiguous-motion coverage failed: {video_id}")
    for source_index in declaration["events"].get("scene_cuts", []):
        if source_index % 8 == 0 and not scores[source_index // 8][1] > 0.25:
            raise ValueError(f"scene-boundary threshold coverage failed: {video_id}/{source_index}")
    for budget in BUDGETS:
        selected = oracle["expected_temporal"][str(budget)]["source_frame_indices"]
        if selected[0] != 0 or selected[-1] != 448 or selected != sorted(set(selected)):
            raise ValueError(f"temporal anchors/order failed: {video_id}/{budget}")
    if not oracle["decode_order"]["decode_order_pts_nonmonotonic"] \
            or not oracle["decode_order"]["decode_order_dts_monotonic"]:
        raise ValueError(f"B-frame packet/presentation-order evidence failed: {video_id}")
    for start, end in declaration["events"].get("exact_static_intervals", []):
        rows = oracle["source_content"][start:end + 1]
        if len({row["source_rgb96x64_sha256"] for row in rows}) != 1:
            raise ValueError(f"declared exact-static interval varies: {video_id}/{start}-{end}")


def generate_bundle(out_dir, *, preregistration_path=None, ffmpeg="ffmpeg", ffprobe="ffprobe"):
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty held-out fixture directory: {out}")
    prereg_path = Path(preregistration_path or DEFAULT_BUNDLE / "preregistration.json")
    prereg_bytes = prereg_path.read_bytes()
    prereg = validate_preregistration(json.loads(prereg_bytes))
    for name in ("fixtures", "content", "selections"):
        (out / name).mkdir(parents=True, exist_ok=True)
    (out / "preregistration.json").write_bytes(prereg_bytes)
    encoder = base.encoder_identity(ffmpeg)
    videos = {}
    decoded_max_spread = 0
    for declaration in VIDEO_DECLARATIONS:
        video_id = declaration["id"]
        video_path = out / "fixtures" / f"{video_id}.mp4"
        encode_video(ffmpeg, video_id, video_path)
        oracle = build_video_oracle(ffmpeg, ffprobe, video_path)
        validate_event_contract(declaration, oracle)
        content_path = out / "content" / f"{video_id}.frames.jsonl"
        content_path.write_bytes(b"".join(canonical(row) + b"\n" for row in oracle["source_content"]))
        packet_path = out / "content" / f"{video_id}.packets.json"
        write_json(packet_path, oracle["packets"])
        selections = {}
        for arm, budget in ARM_BUDGETS:
            expected = oracle["expected_temporal"][str(budget)] if arm == "temporal" \
                else oracle["expected_uniform"][str(budget)]
            png, spread = selection_png(selection_frames(oracle, arm, budget))
            decoded_max_spread = max(decoded_max_spread, spread)
            key = f"{arm}-{budget}"
            png_path = out / "selections" / video_id / f"{key}.png"
            json_path = out / "selections" / video_id / f"{key}.json"
            png_path.parent.mkdir(parents=True, exist_ok=True)
            png_path.write_bytes(png)
            write_json(json_path, {
                "video_id": video_id, "arm": arm, "budget": budget,
                "source_frame_indices": expected["source_frame_indices"],
                "source_timestamps_seconds": expected["source_timestamps_seconds"],
                "decoded_native_rgb_sha256": expected["content_sha256"],
                "content_fnv1a64": expected["content_fnv1a64"],
                "contact_sheet_layout": "4 columns; presentation order left-to-right, top-to-bottom",
            })
            selections[key] = {
                "png": png_path.relative_to(out).as_posix(), "png_sha256": sha256_bytes(png),
                "record": json_path.relative_to(out).as_posix(),
                "record_sha256": sha256_file(json_path),
            }
        videos[video_id] = {
            **copy.deepcopy(declaration), "video": f"fixtures/{video_id}.mp4",
            "video_sha256": sha256_file(video_path), "video_bytes": video_path.stat().st_size,
            "content_frames": content_path.relative_to(out).as_posix(),
            "content_frames_sha256": sha256_file(content_path),
            "decode_order_packets": packet_path.relative_to(out).as_posix(),
            "decode_order_packets_sha256": sha256_file(packet_path),
            **{key: oracle[key] for key in ("ffprobe", "duration_seconds", "candidate_scores_motion_scene",
                                            "expected_temporal", "expected_uniform", "decode_order")},
            "candidate_stride": 8, "candidate_count": 57, "selections": selections,
        }
    cases = []
    for prereg_case in prereg["cases"]:
        video = videos[prereg_case["video_id"]]
        cases.append({
            **copy.deepcopy(prereg_case), "video": video["video"],
            "video_sha256": video["video_sha256"], "duration_seconds": video["duration_seconds"],
            "candidate_count": 57, "expected_uniform": copy.deepcopy(video["expected_uniform"]),
            "expected_uniform_8": copy.deepcopy(video["expected_uniform"]["8"]),
            "expected_temporal": copy.deepcopy(video["expected_temporal"]),
            "anchor_audit": build_anchor_audit(prereg_case, video),
        })
    measurement = copy.deepcopy(prereg["measurement"])
    measurement["arms"] = list(ARMS)
    manifest = {
        "schema_version": SCHEMA_VERSION, "fixture_revision": REVISION,
        "fixture_class": "NEW_PROCEDURAL_SYNTHETIC_INPUT_NOT_MODEL_OUTPUT",
        "natural_video_claim": False,
        "preregistration_sha256": sha256_bytes(prereg_bytes),
        "source": copy.deepcopy(prereg["source"]),
        "native_decode": {"width": NATIVE_WIDTH, "height": NATIVE_HEIGHT,
                          "candidate_frames": 64, "default_selected_frames": 8,
                          "max_premerge_patches": 3584},
        "selection": copy.deepcopy(prereg["selection"]), "encoder": encoder,
        "marker_neutrality": {"source_all_frames_rgb_channels_equal": True,
                              "decoded_selected_max_channel_spread": decoded_max_spread,
                              "binding_marker_disjoint_from_answer_roi": True},
        "videos": videos, "cases": cases,
        "request_contract": copy.deepcopy(prereg["request_contract"]),
        "scoring": copy.deepcopy(prereg["scoring"]), "measurement": measurement,
        "analysis_plan": copy.deepcopy(prereg["analysis_plan"]),
        "claims": copy.deepcopy(prereg["claims_before_live"]),
    }
    if decoded_max_spread > 8:
        raise ValueError(f"decoded grayscale neutrality drift exceeds 8: {decoded_max_spread}")
    write_json(out / "manifest.json", manifest)
    paths = sorted(path for path in out.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(
        f"{sha256_file(path)}  {path.relative_to(out).as_posix()}\n" for path in paths), encoding="ascii")
    return manifest


def read_sums(root):
    declared = {}
    for line in (Path(root) / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        if not HEX256.fullmatch(digest) or relative.startswith("/") \
                or ".." in Path(relative).parts or relative in declared:
            raise ValueError("malformed or duplicate held-out fixture checksum")
        declared[relative] = digest
    actual = {path.relative_to(root).as_posix() for path in Path(root).rglob("*")
              if path.is_file() and path.name != "SHA256SUMS"}
    if set(declared) != actual:
        raise ValueError("held-out fixture artifact set differs from SHA256SUMS")
    for relative, digest in declared.items():
        if sha256_file(Path(root) / relative) != digest:
            raise ValueError(f"altered held-out fixture checksum: {relative}")
    return declared


def validate_bundle(root, *, deep=True, ffmpeg="ffmpeg", ffprobe="ffprobe"):
    root = Path(root)
    read_sums(root)
    manifest = json.loads((root / "manifest.json").read_bytes())
    prereg_bytes = (root / "preregistration.json").read_bytes()
    prereg = validate_preregistration(json.loads(prereg_bytes))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("fixture_revision") != REVISION \
            or manifest["preregistration_sha256"] != sha256_bytes(prereg_bytes):
        raise ValueError("held-out manifest/preregistration identity mismatch")
    prereg_by_id = {case["id"]: case for case in prereg["cases"]}
    if [case["id"] for case in manifest["cases"]] != list(prereg_by_id):
        raise ValueError("held-out case order differs from preregistration")
    for case in manifest["cases"]:
        expected = prereg_by_id[case["id"]]
        for key in ("video_id", "language", "evaluation_role", "question", "expect_all",
                    "expect_any_group", "answer_anchor"):
            if case[key] != expected[key]:
                raise ValueError(f"frozen held-out prompt/answer drift: {case['id']}:{key}")
        if case["anchor_audit"] != build_anchor_audit(expected, manifest["videos"][case["video_id"]]):
            raise ValueError(f"held-out answer-anchor audit drift: {case['id']}")
    for video_id, video in manifest["videos"].items():
        if sha256_file(root / video["video"]) != video["video_sha256"]:
            raise ValueError(f"stale held-out source video: {video_id}")
        declaration = next(item for item in VIDEO_DECLARATIONS if item["id"] == video_id)
        if not deep:
            continue
        oracle = build_video_oracle(ffmpeg, ffprobe, root / video["video"])
        validate_event_contract(declaration, oracle)
        for key in ("ffprobe", "duration_seconds", "candidate_scores_motion_scene",
                    "expected_temporal", "expected_uniform", "decode_order"):
            if oracle[key] != video[key]:
                raise ValueError(f"independent held-out oracle mismatch: {video_id}:{key}")
        content = [json.loads(line) for line in (root / video["content_frames"]).read_text().splitlines()]
        if content != oracle["source_content"] \
                or json.loads((root / video["decode_order_packets"]).read_bytes()) != oracle["packets"]:
            raise ValueError(f"independent held-out source/packet mismatch: {video_id}")
        for arm, budget in ARM_BUDGETS:
            key = f"{arm}-{budget}"
            selection = video["selections"][key]
            expected = video["expected_temporal"][str(budget)] if arm == "temporal" \
                else video["expected_uniform"][str(budget)]
            png, _ = selection_png(selection_frames(oracle, arm, budget))
            if png != (root / selection["png"]).read_bytes():
                raise ValueError(f"held-out selection contact sheet drift: {video_id}/{key}")
            record = json.loads((root / selection["record"]).read_bytes())
            if record["decoded_native_rgb_sha256"] != expected["content_sha256"] \
                    or record["source_frame_indices"] != expected["source_frame_indices"] \
                    or record["source_timestamps_seconds"] != expected["source_timestamps_seconds"]:
                raise ValueError(f"held-out selection source binding drift: {video_id}/{key}")
    if deep and manifest["encoder"] != base.encoder_identity(ffmpeg):
        raise ValueError("pinned held-out FFmpeg/libx264 encoder drift")
    return manifest


def write_freeze(bundle, freeze_path=DEFAULT_FREEZE):
    bundle = Path(bundle)
    manifest = validate_bundle(bundle, deep=False)
    receipt = {
        "schema_version": 1, "fixture_revision": REVISION,
        "frozen_on": "2026-11-27", "frozen_before_any_model_response": True,
        "model_response_inspected": False,
        "preregistration_sha256": sha256_file(bundle / "preregistration.json"),
        "manifest_sha256": sha256_file(bundle / "manifest.json"),
        "sha256sums_sha256": sha256_file(bundle / "SHA256SUMS"),
        "quality_case_ids": [case["id"] for case in manifest["cases"]],
        "historical_task61_cases_in_denominator": [],
        "post_freeze_case_prompt_source_answer_scoring_population_or_runtime_changes": "forbidden",
    }
    write_json(Path(freeze_path), receipt)
    return receipt


def validate_freeze(bundle, freeze_path=DEFAULT_FREEZE):
    bundle = Path(bundle)
    receipt = json.loads(Path(freeze_path).read_bytes())
    if receipt.get("schema_version") != 1 or receipt.get("fixture_revision") != REVISION \
            or receipt.get("frozen_before_any_model_response") is not True \
            or receipt.get("model_response_inspected") is not False:
        raise ValueError("held-out fixture was not frozen before model inspection")
    for key, path in (("preregistration_sha256", "preregistration.json"),
                      ("manifest_sha256", "manifest.json"),
                      ("sha256sums_sha256", "SHA256SUMS")):
        if not HEX256.fullmatch(receipt.get(key, "")) or receipt[key] != sha256_file(bundle / path):
            raise ValueError(f"held-out fixture freeze mismatch: {key}")
    manifest = validate_bundle(bundle, deep=False)
    if receipt["quality_case_ids"] != [case["id"] for case in manifest["cases"]] \
            or receipt["historical_task61_cases_in_denominator"]:
        raise ValueError("held-out freeze population binding drift")
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
        print(f"TEMPORAL HELD-OUT FIXTURE FAIL: {error}", file=sys.stderr)
        return 1
    print(json.dumps({
        "result": "PASS", "fixture_revision": REVISION, "videos": len(manifest["videos"]),
        "quality_scored_cases": len(manifest["cases"]),
        "natural_video_quality": "NOT_COVERED", "real_generation": "NOT_RUN",
        "live_latency": "NOT_RUN", "heldout_fixture_substring_accuracy": "NOT_RUN",
        "perceptual_quality": "NOT_RUN",
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
