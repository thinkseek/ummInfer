#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task #47 offline audit and controlled serial temporal-quality interventions.

The preparation path independently decodes the tracked source frames, creates
lossless RGB H.264 containers with identical pixels and explicitly varied PTS,
and seals all inputs. The live path uses the unchanged public native temporal
policy on one serial stack. Expected-substring observations never gate or imply
perceptual/generalized quality.
"""

from __future__ import annotations

import binascii
import base64
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import zlib

import umminfer_controlled_measurement as controlled
import umminfer_long_video_fixture as fixture
import umminfer_long_video_live as long_live

SCHEMA_VERSION = 1
BUDGET = 8
WIDTH = 224
HEIGHT = 160
FRAME_BYTES = WIDTH * HEIGHT * 3
SEED = 390039
WARMUP_REPETITIONS = 1
MEASURED_REPETITIONS = 3
ARMS = (
    "temporal_source_pts",
    "temporal_uniform_positions",
    "temporal_ordinal_pts",
    "uniform_nominal_positions",
    "uniform_source_relative_pts",
    "temporal_source_pts_alt_language",
)
TEMPORAL_POSITION_ARMS = ARMS[:3]
ALT_PROMPTS = {
    "separated-motion": "哪个分离运动事件最先出现：红色、绿色还是黄色？只回答一种颜色。",
    "scene-static": "What is the dominant color after the second sudden cut in the video? "
                    "Answer with only the color.",
    "near-duplicate": "What color is the short band that appears after the long static "
                      "interval? Answer with only the color.",
    # Preserve the original instruction, including its answer leakage. This is a
    # language-only control, not a debiased prompt or a new quality oracle.
    "bframe-anchors": "What color is the first frame and what color is the last frame? "
                      "Answer: white and black.",
}
R4_NATIVE_FIELDS = (
    "grid_t", "grid_h", "grid_w", "premerge_patch_rows",
    "merged_tokens_per_slice", "video_token_count", "prompt_token_count",
    "media_hash", "mrope_next", "mrope_fnv1a64",
)
DIGITS = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, ensure_ascii=False, indent=2).encode() + b"\n")


def sha256_file(path):
    return fixture.sha256_file(Path(path))


def seal_tree(root):
    root = Path(root)
    files = sorted(path for path in root.rglob("*")
                   if path.is_file() and path.name != "SHA256SUMS")
    (root / "SHA256SUMS").write_text("".join(
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files),
        encoding="ascii")


def validate_seal(root):
    root = Path(root)
    declared = {}
    for line in (root / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        if not long_live.HEX64.fullmatch(digest) or relative.startswith("/") \
                or ".." in Path(relative).parts or relative in declared:
            raise ValueError("malformed or duplicate input checksum")
        declared[relative] = digest
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name != "SHA256SUMS"}
    if set(declared) != actual:
        raise ValueError("missing, extra, or renamed sealed input artifacts")
    for relative, digest in declared.items():
        if sha256_file(root / relative) != digest:
            raise ValueError(f"altered sealed input: {relative}")


def fnv_frames(data):
    if len(data) != BUDGET * FRAME_BYTES:
        raise ValueError(f"selected RGB byte count is not exactly eight {WIDTH}x{HEIGHT} frames")
    return [fixture.fnv1a64(data[offset:offset + FRAME_BYTES])
            for offset in range(0, len(data), FRAME_BYTES)]


def split_frames(data):
    fnv_frames(data)
    return [data[offset:offset + FRAME_BYTES]
            for offset in range(0, len(data), FRAME_BYTES)]


def uniform_nominal_timestamps(duration):
    return [duration * index / BUDGET for index in range(BUDGET)]


def pair_first(timestamps):
    return [timestamps[min(2 * index, len(timestamps) - 1)]
            for index in range((len(timestamps) + 1) // 2)]


def native_showinfo_timestamp(value):
    """Exact float parsed from FFmpeg showinfo's six-significant-digit text."""
    return float(format(value, ".6g"))


def selection_for_arm(case, arm):
    kind = "uniform" if arm.startswith("uniform_") else "temporal"
    return kind, case["expected_uniform_8"] if kind == "uniform" \
        else case["expected_temporal"][str(BUDGET)]


def timestamps_for_arm(case, arm, duration):
    if arm == "temporal_ordinal_pts":
        return [float(index) for index in range(BUDGET)]
    kind, selection = selection_for_arm(case, arm)
    if arm in ("temporal_source_pts", "temporal_source_pts_alt_language"):
        return list(selection["source_timestamps_seconds"])
    if arm == "uniform_source_relative_pts":
        source = selection["source_timestamps_seconds"]
        return [value - source[0] for value in source]
    if arm in ("temporal_uniform_positions", "uniform_nominal_positions"):
        return uniform_nominal_timestamps(duration)
    raise ValueError(f"unknown temporal-quality arm: {arm}")


def prompt_for_arm(case, arm):
    return ALT_PROMPTS[case["id"]] if arm == "temporal_source_pts_alt_language" \
        else case["question"]


def run_process(args, *, timeout=180, capture=True):
    return subprocess.run([str(value) for value in args], check=True,
                          capture_output=capture, timeout=timeout)


def decode_selected(source, indices):
    expression = "+".join(f"eq(n\\,{index})" for index in indices)
    result = run_process([
        "ffmpeg", "-v", "error", "-i", Path(source), "-vf",
        "select=" + expression + f",scale={WIDTH}:{HEIGHT}:flags=bicubic",
        "-vsync", "0", "-frames:v", BUDGET, "-f", "rawvideo", "-pix_fmt", "rgb24",
        "pipe:1",
    ])
    fnv_frames(result.stdout)
    return result.stdout


def setpts_expression(timestamps):
    ticks = [round(value * 16) for index, value in enumerate(timestamps)]
    if any(abs(ticks[index] / 16 - value) > 1e-12
           for index, value in enumerate(timestamps)):
        raise ValueError("intervention PTS is not exactly representable at 1/16 second")
    expression = str(ticks[-1])
    for index in range(BUDGET - 2, -1, -1):
        expression = f"if(eq(N\\,{index})\\,{ticks[index]}\\,{expression})"
    return "setpts=" + expression


def container_timescale(arm):
    return 16


def encode_intervention_video(path):
    path = Path(path)
    data = path.read_bytes()
    if not 0 < len(data) <= 32 * 1024 * 1024:
        raise ValueError("intervention video violates the native 32 MiB input bound")
    mime = {".mp4": "video/mp4", ".mkv": "video/x-matroska"}.get(path.suffix)
    if mime is None:
        raise ValueError("intervention harness accepts only audited MP4/Matroska containers")
    url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    return url, fixture.sha256_bytes(data), mime


def encode_lossless(raw_path, timestamps, output):
    # High444 lossless RGB is deliberately B-frame-free. Original B-frame
    # handling is independently pinned by the sealed source packet/decode evidence.
    args = [
        "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-framerate", "16", "-i", Path(raw_path), "-an",
        "-vf", setpts_expression(timestamps), "-c:v", "libx264rgb", "-preset", "veryslow",
        "-crf", "0", "-bf", "0", "-pix_fmt", "rgb24",
    ]
    if Path(output).suffix == ".mkv":
        # Matroska preserves a nonzero initial PTS without an MP4 edit list. Its
        # pinned millisecond timebase exactly represents the uniform source PTS.
        args += ["-enc_time_base", "1/16", "-avoid_negative_ts", "disabled",
                 "-vsync", "0", "-f", "matroska", Path(output)]
    else:
        args += ["-video_track_timescale", "16", "-enc_time_base", "1/16",
                 "-use_editlist", "0", "-avoid_negative_ts", "disabled",
                 "-vsync", "0", "-f", "mp4", Path(output)]
    run_process(args)


def decode_with_pts(path):
    result = run_process([
        "ffmpeg", "-v", "info", "-i", Path(path), "-vf", "showinfo", "-vsync", "0",
        "-frames:v", BUDGET, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ])
    timestamps = []
    for line in result.stderr.decode("utf-8", errors="replace").splitlines():
        if " n:" in line and "pts_time:" in line:
            timestamps.append(float(line.split("pts_time:", 1)[1].split()[0]))
    if len(timestamps) != BUDGET:
        raise ValueError(f"derived video did not decode exactly eight PTS rows: {path}")
    fnv_frames(result.stdout)
    return result.stdout, timestamps


def packet_evidence(path):
    value = json.loads(run_process([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
        "-show_entries", "packet=pts,dts,flags", "-of", "json", Path(path),
    ]).stdout)
    packets = value["packets"]
    pts = [int(packet["pts"]) for packet in packets]
    dts = [int(packet["dts"]) for packet in packets]
    return {
        "packet_count": len(packets),
        "decode_order_pts": pts,
        "decode_order_dts": dts,
        "packet_pts_nonmonotonic": any(later < earlier for earlier, later in zip(pts, pts[1:])),
        "packet_dts_strictly_increasing": all(later > earlier for earlier, later in zip(dts, dts[1:])),
        "packets": packets,
    }


def png_chunk(kind, payload):
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(
        ">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)


def write_png(path, width, height, pixels):
    if len(pixels) != width * height * 3:
        raise ValueError("invalid PNG RGB buffer")
    scanlines = b"".join(b"\x00" + pixels[y * width * 3:(y + 1) * width * 3]
                         for y in range(height))
    data = (b"\x89PNG\r\n\x1a\n" +
            png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
            png_chunk(b"IDAT", zlib.compress(scanlines, 9)) + png_chunk(b"IEND", b""))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)


def contact_sheet(path, data, source_indices):
    frames = split_frames(data)
    label_h = 20
    sheet_w, sheet_h = WIDTH * 4, (HEIGHT + label_h) * 2
    canvas = bytearray(sheet_w * sheet_h * 3)
    for position, (frame, source_index) in enumerate(zip(frames, source_indices)):
        ox, oy = (position % 4) * WIDTH, (position // 4) * (HEIGHT + label_h)
        for y in range(HEIGHT):
            start = ((oy + label_h + y) * sheet_w + ox) * 3
            canvas[start:start + WIDTH * 3] = frame[y * WIDTH * 3:(y + 1) * WIDTH * 3]
        text = str(source_index)
        cursor = ox + 4
        for char in text:
            for row, pattern in enumerate(DIGITS[char]):
                for col, bit in enumerate(pattern):
                    if bit != "1":
                        continue
                    for dy in range(2):
                        for dx in range(2):
                            x, y = cursor + col * 2 + dx, oy + 5 + row * 2 + dy
                            offset = (y * sheet_w + x) * 3
                            canvas[offset:offset + 3] = b"\xff\xff\xff"
            cursor += 8
    write_png(path, sheet_w, sheet_h, bytes(canvas))


def r4_baselines(report):
    required = {(case["id"], arm) for case in fixture.CASES for arm in ("uniform", "temporal")}
    found = {}
    for record in report["records"]:
        key = (record["clip"], record["arm"])
        if record["mode"] == "serial" and record["phase"] == "measured" \
                and record["repetition"] == 1 and key in required:
            found[key] = {
                "native": {field: record["native"][field] for field in R4_NATIVE_FIELDS},
                "raw_answer": record["raw_answer"],
                "expected_answer_hit": record["expected_answer_hit"],
                "source_frame_indices": record["source_selection"]["exact_source_frame_indices"],
                "source_timestamps_seconds": record["source_selection"]["exact_source_timestamps_seconds"],
            }
    if set(found) != required:
        raise ValueError("sealed R4 baseline does not cover all serial temporal/uniform cases")
    return found


def semantic_audit(case, temporal_rgb):
    cid = case["id"]
    uniform = set(case["expected_uniform_8"]["source_frame_indices"])
    temporal = set(case["expected_temporal"]["8"]["source_frame_indices"])
    intended = {
        "separated-motion": {64, 176, 320},
        "scene-static": {224},
        "near-duplicate": {288},
        "bframe-anchors": {0, 448},
    }[cid]
    source_zero = fixture.source_frame(cid, 0)
    tag = bytes(source_zero[(1 * fixture.WIDTH + 1) * 3:(1 * fixture.WIDTH + 1) * 3 + 3])
    first_native = temporal_rgb[:FRAME_BYTES]
    green_pixels = sum(
        1 for offset in range(0, FRAME_BYTES, 3)
        if first_native[offset + 1] > 150 and first_native[offset + 1] >
        first_native[offset] * 1.5 and first_native[offset + 1] > first_native[offset + 2] * 1.25)
    return {
        "case": cid,
        "question": case["question"],
        "expected_terms": list(case["expect_all"]),
        "uniform_source_frames": sorted(uniform),
        "temporal_source_frames": sorted(temporal),
        "intended_answer_anchor_frames": sorted(intended),
        "uniform_intended_anchor_frames": sorted(uniform & intended),
        "temporal_intended_anchor_frames": sorted(temporal & intended),
        "uniform_expected_answer_supported_by_selected_pixels": bool(uniform & intended),
        "temporal_expected_answer_supported_by_selected_pixels": temporal >= intended,
        "temporal2_pairs": [list(case["expected_temporal"]["8"]["source_frame_indices"][i:i + 2])
                            for i in range(0, BUDGET, 2)],
        "temporal2_pair_first_timestamps": case["expected_temporal"]["8"][
            "temporal2_pair_first_timestamps"],
        "chromatic_anchor_tag": {
            "source_frame_zero_temporally_selected": 0 in temporal,
            "source_pixel_1_1_rgb": list(tag),
            "source_tag_nominal_pixels": 25,
            "native_frame_zero_green_dominant_pixels": green_pixels,
            "limitation": "unmentioned colored source-binding tag can confound color questions",
        },
        "prompt_contains_all_expected_terms": all(
            term.casefold() in case["question"].casefold() for term in case["expect_all"]),
        "oracle_limitations": [
            "expected-substring matching is not perceptual or generalized quality",
            "uniform coverage does not contain the intended long-video answer anchors",
        ] + (["both frames contain both black and white regions; the prompt repeats both expected terms"]
             if cid == "bframe-anchors" else []),
    }


def prepare_plan(bundle, out_dir, *, r4_evidence=None):
    bundle = Path(bundle).resolve()
    out = Path(out_dir).resolve()
    if out.exists():
        raise ValueError("refusing to overwrite an existing intervention plan")
    manifest = fixture.validate_bundle(bundle, deep=False)
    baselines = None
    if r4_evidence is not None:
        r4_report = long_live.validate_bundle(Path(r4_evidence).resolve())
        if not r4_report["gate_pass"]:
            raise ValueError("R4 technical evidence is not a passing sealed baseline")
        baselines = r4_baselines(r4_report)
    out.mkdir(parents=True)
    shutil.copytree(bundle, out / "source")
    if r4_evidence is not None:
        shutil.copytree(Path(r4_evidence).resolve(), out / "r4-sealed")
    plan = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "Task47_R4_root_cause_pixel_position_prompt_controls",
        "width": WIDTH, "height": HEIGHT, "budget": BUDGET,
        "seed": SEED, "warmup_repetitions": WARMUP_REPETITIONS,
        "measured_repetitions": MEASURED_REPETITIONS,
        "arm_order": "seeded_interleaved", "arms": list(ARMS),
        "source_manifest_sha256": sha256_file(bundle / "manifest.json"),
        "r4_evidence_sha256sums": sha256_file(Path(r4_evidence) / "SHA256SUMS")
            if r4_evidence is not None else None,
        "answer_scoring": "unchanged_original_expected_substrings",
        "claims": {"real_generation": "NOT_RUN", "live_latency": "NOT_RUN",
                   "perceptual_quality": "NOT_RUN", "generalized_quality_claim": False},
        "cases": [],
    }
    position_requests = []
    audits = []
    for case in manifest["cases"]:
        cid = case["id"]
        case_out = {"id": cid, "video_sha256": case["video_sha256"],
                    "original_question": case["question"], "expect_all": case["expect_all"],
                    "expect_any_group": case.get("expect_any_group", False), "arms": {}}
        decoded = {}
        for kind in ("temporal", "uniform"):
            selection = case["expected_temporal"]["8"] if kind == "temporal" \
                else case["expected_uniform_8"]
            raw = decode_selected(out / "source" / case["video"],
                                  selection["source_frame_indices"])
            if fnv_frames(raw) != selection["content_fnv1a64"]:
                raise ValueError(f"independent selected RGB mismatch: {cid}/{kind}")
            raw_path = out / "media" / cid / f"{kind}.rgb"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(raw)
            contact_sheet(out / "audit" / f"{cid}-{kind}.png", raw,
                          selection["source_frame_indices"])
            decoded[kind] = raw
        audits.append(semantic_audit(case, decoded["temporal"]))
        for arm in ARMS:
            kind, selection = selection_for_arm(case, arm)
            timestamps = timestamps_for_arm(case, arm, case["duration_seconds"])
            native_timestamps = [native_showinfo_timestamp(value) for value in timestamps]
            prompt = prompt_for_arm(case, arm)
            if arm == "temporal_source_pts_alt_language":
                video_relative = f"media/{cid}/temporal_source_pts.mp4"
            else:
                video_relative = f"media/{cid}/{arm}.mp4"
                video_path = out / video_relative
                encode_lossless(out / "media" / cid / f"{kind}.rgb", timestamps, video_path)
                decoded_actual, actual_pts = decode_with_pts(video_path)
                if decoded_actual != decoded[kind] or actual_pts != native_timestamps:
                    raise ValueError(
                        f"lossless pixel/PTS round trip failed: {cid}/{arm}: "
                        f"rgb_equal={decoded_actual == decoded[kind]} actual_pts={actual_pts} "
                        f"expected_native_pts={native_timestamps} actual_fnv={fnv_frames(decoded_actual)} "
                        f"expected_fnv={selection['content_fnv1a64']}")
                packets = packet_evidence(video_path)
                expected_ticks = [round(value * container_timescale(arm)) for value in timestamps]
                if packets["packet_count"] != BUDGET or not packets["packet_dts_strictly_increasing"] \
                        or packets["decode_order_pts"] != expected_ticks:
                    raise ValueError(f"derived packet count/order/rational-PTS mismatch: {cid}/{arm}")
                write_json(out / "media" / cid / f"{arm}.packets.json", packets)
            baseline = None
            if baselines is not None and arm in ("temporal_source_pts", "uniform_nominal_positions"):
                baseline = baselines[(cid, "temporal" if kind == "temporal" else "uniform")]
            expected_media = baselines[(cid, kind)]["native"]["media_hash"] \
                if baselines is not None else None
            case_out["arms"][arm] = {
                "pixel_source": kind,
                "original_source_frame_indices": selection["source_frame_indices"],
                "original_source_timestamps_seconds": selection["source_timestamps_seconds"],
                "selected_frame_content_fnv1a64": selection["content_fnv1a64"],
                "selected_rgb_sha256": fixture.sha256_bytes(decoded[kind]),
                "timestamps": timestamps,
                "native_showinfo_timestamps": native_timestamps,
                "container_pts_ticks_per_second": container_timescale(arm),
                "container_pts_ticks": [round(value * container_timescale(arm)) for value in timestamps],
                "temporal2_pair_first_timestamps": pair_first(native_timestamps),
                "prompt": prompt,
                "video": video_relative,
                "video_sha256": sha256_file(out / video_relative),
                "expected_media_hash": expected_media,
                "r4_baseline": baseline,
            }
            position_requests.append({
                "id": f"{cid}/{arm}", "prompt": prompt,
                "timestamps": pair_first(native_timestamps), "grid_h": 10, "grid_w": 14,
            })
        plan["cases"].append(case_out)
    # Preserve the failed absolute-nonzero-start construction as an explicit
    # negative probe. The native-style decode command normalizes it to zero even
    # though Matroska packet PTS retains the absolute values. No R4 source has a
    # nonzero start, and no live arm claims absolute-start coverage.
    first_case = manifest["cases"][0]
    absolute_pts = first_case["expected_uniform_8"]["source_timestamps_seconds"]
    probe_path = out / "audit/nonzero-start.mkv"
    encode_lossless(out / "media" / first_case["id"] / "temporal.rgb", absolute_pts, probe_path)
    probe_rgb, probe_native_pts = decode_with_pts(probe_path)
    relative_pts = [value - absolute_pts[0] for value in absolute_pts]
    if probe_native_pts != relative_pts:
        raise ValueError(f"nonzero-start normalization probe drifted: {probe_native_pts}")
    write_json(out / "audit/nonzero-start.json", {
        "container": "Matroska lossless RGB H.264",
        "container_absolute_pts_seconds": absolute_pts,
        "container_packet_evidence": packet_evidence(probe_path),
        "native_style_decode_pts_seconds": probe_native_pts,
        "expected_zero_normalized_relative_pts_seconds": relative_pts,
        "selected_rgb_sha256": fixture.sha256_bytes(probe_rgb),
        "r4_impact": "NONE_ALL_R4_SOURCES_START_AT_ZERO",
        "live_absolute_nonzero_start_coverage": "NOT_RUN",
    })
    write_json(out / "plan.json", plan)
    write_json(out / "position-input.json", {"schema_version": 1, "requests": position_requests})
    write_json(out / "audit.json", {
        "schema_version": 1, "cases": audits,
        "r4_answer_counts_across_both_modes": {"default": [6, 24], "uniform": [6, 24],
                                                "temporal": [0, 24]},
        "negative_outcomes_preserved": True,
        "native_defect_conclusion": "NOT_DECIDED_BY_OFFLINE_PIXEL_AUDIT",
    })
    seal_tree(out)
    return plan


def independent_positions(ids, grid_t, grid_h, grid_w):
    if grid_h % 2 or grid_w % 2:
        raise ValueError("mRoPE audit grid must be mergeable")
    gh, gw = grid_h // 2, grid_w // 2
    slice_tokens = gh * gw
    runs = []
    index = 0
    while index < len(ids):
        if ids[index] not in (248056, 248057):
            index += 1
            continue
        begin = index
        while index < len(ids) and ids[index] in (248056, 248057):
            index += 1
        runs.append((begin, index - begin))
    if len(runs) != grid_t or any(length != slice_tokens for _, length in runs):
        raise ValueError("expanded token ids do not match the declared video grid")
    positions = {"t": [0] * len(ids), "h": [0] * len(ids), "w": [0] * len(ids)}
    counter = 0
    run_index = 0
    index = 0
    while index < len(ids):
        if run_index < len(runs) and index == runs[run_index][0]:
            for offset in range(slice_tokens):
                positions["t"][index + offset] = counter
                positions["h"][index + offset] = counter + offset // gw
                positions["w"][index + offset] = counter + offset % gw
            counter += max(gh, gw)
            index += slice_tokens
            run_index += 1
        else:
            positions["t"][index] = positions["h"][index] = positions["w"][index] = counter
            counter += 1
            index += 1
    positions["next"] = counter
    return positions


def position_digest(positions):
    value = 0xCBF29CE484222325
    for t, h, w in zip(positions["t"], positions["h"], positions["w"]):
        for coordinate in (t, h, w):
            value ^= coordinate & 0xFFFFFFFF
            value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def plan_case(plan, case_id):
    return next(case for case in plan["cases"] if case["id"] == case_id)


def finalize_positions(plan_dir, host_output):
    root = Path(plan_dir).resolve()
    validate_seal(root)
    copied = root / "position-ids.json"
    shutil.copyfile(Path(host_output).resolve(), copied)
    host = json.loads(copied.read_bytes())
    requests = {request["id"]: request for request in host.get("requests", [])}
    expected = {request["id"]: request for request in
                json.loads((root / "position-input.json").read_bytes())["requests"]}
    if set(requests) != set(expected) or len(requests) != len(host["requests"]):
        raise ValueError("host prompt-position audit request set mismatch")
    entries = {}
    plan = json.loads((root / "plan.json").read_bytes())
    for request_id, request in expected.items():
        actual = requests[request_id]
        if actual.get("input_prompt") != request["prompt"]:
            raise ValueError(f"host audit drifted {request_id}/prompt")
        for field in ("timestamps", "grid_h", "grid_w"):
            if actual[field] != request[field]:
                raise ValueError(f"host audit drifted {request_id}/{field}")
        ids = actual["expanded_ids"]
        positions = independent_positions(ids, actual["grid_t"], actual["grid_h"], actual["grid_w"])
        digest = position_digest(positions)
        cid, arm = request_id.split("/", 1)
        arm_record = plan_case(plan, cid)["arms"][arm]
        baseline = arm_record.get("r4_baseline")
        if baseline is not None:
            native = baseline["native"]
            if len(ids) != native["prompt_token_count"] or digest != native["mrope_fnv1a64"] \
                    or positions["next"] != native["mrope_next"]:
                raise ValueError(f"independent mRoPE reconstruction diverges from R4: {request_id}")
        entries[request_id] = {
            "id": request_id, "expanded_id_count": len(ids),
            "expanded_ids_sha256": fixture.sha256_bytes(json.dumps(ids, separators=(",", ":")).encode()),
            "rendered_prompt_sha256": fixture.sha256_bytes(actual["rendered_prompt"].encode()),
            "grid_t": actual["grid_t"], "grid_h": actual["grid_h"], "grid_w": actual["grid_w"],
            "timestamps": actual["timestamps"], "mrope_fnv1a64": digest,
            "mrope_next": positions["next"], "positions": positions,
        }
    write_json(root / "position-oracle.json", {"schema_version": 1, "entries": entries,
               "method": "independent_python_coordinates_from_host_emitted_expanded_ids"})
    seal_tree(root)
    return entries


def validate_plan(plan_dir, *, deep=True, require_positions=True):
    root = Path(plan_dir).resolve()
    validate_seal(root)
    plan = json.loads((root / "plan.json").read_bytes())
    if plan["schema_version"] != SCHEMA_VERSION or tuple(plan["arms"]) != ARMS:
        raise ValueError("unsupported intervention plan")
    source_manifest = fixture.validate_bundle(root / "source", deep=deep)
    if fixture.sha256_file(root / "source/manifest.json") != plan["source_manifest_sha256"]:
        raise ValueError("intervention source manifest drift")
    if require_positions and any(case["arms"][arm]["expected_media_hash"] is None
                                 for case in plan["cases"] for arm in ARMS):
        raise ValueError("live plan lacks frozen R4 media identities")
    oracle = None
    if require_positions:
        oracle = json.loads((root / "position-oracle.json").read_bytes())["entries"]
        host = json.loads((root / "position-ids.json").read_bytes())
        host_by_id = {request["id"]: request for request in host["requests"]}
        if set(oracle) != set(host_by_id) or len(oracle) != len(plan["cases"]) * len(ARMS):
            raise ValueError("position oracle coverage mismatch")
        for request_id, entry in oracle.items():
            ids = host_by_id[request_id]["expanded_ids"]
            positions = independent_positions(ids, entry["grid_t"], entry["grid_h"], entry["grid_w"])
            if positions != entry["positions"] or position_digest(positions) != entry["mrope_fnv1a64"]:
                raise ValueError(f"independent position recomputation failed: {request_id}")
    for case in plan["cases"]:
        source_case = next(item for item in source_manifest["cases"] if item["id"] == case["id"])
        for arm, record in case["arms"].items():
            if record["prompt"] != prompt_for_arm(source_case, arm) \
                    or record["timestamps"] != timestamps_for_arm(
                        source_case, arm, source_case["duration_seconds"]) \
                    or record["native_showinfo_timestamps"] != [
                        native_showinfo_timestamp(value) for value in record["timestamps"]] \
                    or record["container_pts_ticks_per_second"] != container_timescale(arm) \
                    or record["container_pts_ticks"] != [
                        round(value * container_timescale(arm)) for value in record["timestamps"]]:
                raise ValueError(f"intervention prompt/PTS drift: {case['id']}/{arm}")
            if deep:
                raw, timestamps = decode_with_pts(root / record["video"])
                packets = packet_evidence(root / record["video"])
                if fixture.sha256_bytes(raw) != record["selected_rgb_sha256"] \
                        or fnv_frames(raw) != record["selected_frame_content_fnv1a64"] \
                        or timestamps != record["native_showinfo_timestamps"] \
                        or packets["decode_order_pts"] != record["container_pts_ticks"]:
                    raise ValueError(f"fresh derived pixel/PTS verification failed: {case['id']}/{arm}")
            if require_positions:
                entry = oracle[f"{case['id']}/{arm}"]
                if entry["timestamps"] != record["temporal2_pair_first_timestamps"]:
                    raise ValueError(f"position oracle timestamp mismatch: {case['id']}/{arm}")
                baseline = record.get("r4_baseline")
                if baseline is not None:
                    native = baseline["native"]
                    if entry["expanded_id_count"] != native["prompt_token_count"] \
                            or entry["mrope_fnv1a64"] != native["mrope_fnv1a64"] \
                            or entry["mrope_next"] != native["mrope_next"]:
                        raise ValueError(f"fresh R4 position baseline mismatch: {case['id']}/{arm}")
    return plan, oracle


def native_validate(record, arm_record, oracle_entry, short):
    native = record["native"]
    execution = record["native_execution"]
    short.validate_native(native, "temporal")
    if native["engine_path"] != "serial" or native["decoded_frame_count"] != BUDGET \
            or native["selected_frame_count"] != BUDGET \
            or native["selected_candidate_indices"] != list(range(BUDGET)) \
            or native["selected_source_frame_indices"] != list(range(BUDGET)) \
            or not native["source_indices_exact"] or native["has_nominal_timestamps"]:
        raise ValueError("derived native temporal selection/order/PTS contract diverged")
    if native["selected_frame_content_fnv1a64"] != arm_record["selected_frame_content_fnv1a64"] \
            or native["selected_frame_timestamps"] != arm_record["native_showinfo_timestamps"] \
            or native["temporal2_pair_first_timestamps"] != arm_record["temporal2_pair_first_timestamps"]:
        raise ValueError("native selected pixels or PTS diverged from independent expectation")
    if native["media_hash"] != arm_record["expected_media_hash"]:
        raise ValueError("native preprocessed media hash diverged from frozen R4 pixel identity")
    if native["prompt_token_count"] != oracle_entry["expanded_id_count"] \
            or native["mrope_fnv1a64"] != oracle_entry["mrope_fnv1a64"] \
            or native["mrope_next"] != oracle_entry["mrope_next"]:
        raise ValueError("native prompt/mRoPE diverged from independent position oracle")
    if execution["prefix_reused"] != 0 or execution["prompt_tokens"] != native["prompt_token_count"] \
            or execution["prefill_tokens"] != native["prompt_token_count"]:
        raise ValueError("cold-prefix/full-prefill proof failed")


def latency_record(response, execution):
    return {"first_event": response["first_event_s"] * 1000.0,
            "ttft": response["ttft_s"] * 1000.0, "total": response["total_s"] * 1000.0,
            "vision": execution["vision_ms"], "prefill": execution["prefill_ms"],
            "decode": execution["decode_ms"], "native_elapsed": execution["elapsed_ms"]}


def answer_outcomes(records):
    result = {}
    for record in records:
        if record["phase"] != "measured":
            continue
        arm = result.setdefault(record["arm"], {"declared": 0, "completed": 0,
                                                "answer_hits": 0, "answer_misses": 0,
                                                "cases": {}})
        arm["declared"] += 1
        arm["completed"] += record["status"] == "COMPLETED"
        hit = record["expected_answer_hit"]
        arm["answer_hits"] += hit is True
        arm["answer_misses"] += hit is False
        case = arm["cases"].setdefault(record["clip"], {"answers": [], "hits": []})
        case["answers"].append(record["raw_answer"])
        case["hits"].append(hit)
    return result


def control_checks(records):
    checks = []
    grouped = {}
    for record in records:
        grouped.setdefault((record["phase"], record["repetition"], record["clip"]), {})[record["arm"]] = record
    controls = (
        ("temporal_source_pts", "temporal_uniform_positions", "temporal_pixels_position_only"),
        ("temporal_source_pts", "temporal_ordinal_pts", "temporal_pixels_pts_normalization"),
        ("uniform_nominal_positions", "uniform_source_relative_pts", "uniform_pixels_position_only"),
        ("temporal_source_pts", "temporal_source_pts_alt_language", "prompt_language_only"),
    )
    for key, arms in sorted(grouped.items()):
        for left_name, right_name, scope in controls:
            left, right = arms[left_name], arms[right_name]
            valid = left["native_evidence_valid"] and right["native_evidence_valid"]
            same_pixels = valid and left["native"]["media_hash"] == right["native"]["media_hash"] \
                and left["native"]["selected_frame_content_fnv1a64"] == \
                right["native"]["selected_frame_content_fnv1a64"]
            left_prompt = left["request"]["messages"][0]["content"][0]["text"]
            right_prompt = right["request"]["messages"][0]["content"][0]["text"]
            same_positions = valid and left["native"]["selected_frame_timestamps"] == \
                right["native"]["selected_frame_timestamps"]
            if scope == "prompt_language_only":
                passed = same_pixels and same_positions and left_prompt != right_prompt
            else:
                passed = same_pixels and not same_positions and left_prompt == right_prompt
            checks.append({"scope": scope, "phase": key[0], "repetition": key[1],
                           "clip": key[2], "left": left_name, "right": right_name,
                           "same_pixels": same_pixels, "same_positions": same_positions,
                           "same_prompt": left_prompt == right_prompt,
                           "left_answer": left["raw_answer"], "right_answer": right["raw_answer"],
                           "answer_byte_identical": left["raw_answer"] == right["raw_answer"],
                           "passed": bool(passed)})
    repeated = {}
    for record in records:
        repeated.setdefault((record["clip"], record["arm"]), []).append(record)
    for (clip, arm), rows in sorted(repeated.items()):
        valid = all(row["native_evidence_valid"] for row in rows)
        requests = len({long_live.canonical(row["request"]) for row in rows}) == 1
        answers = len({row["raw_answer"] for row in rows}) == 1
        native = valid and all(short_native_equal(rows[0]["native"], row["native"])
                               for row in rows[1:])
        checks.append({"scope": "within_serial_repetitions", "clip": clip, "arm": arm,
                       "passed": bool(valid and requests and answers and native),
                       "request_byte_identical": requests, "answer_byte_identical": answers,
                       "native_identical": native})
    return checks


def short_native_equal(left, right):
    fields = set(short_native_fields())
    return all(left.get(field) == right.get(field) for field in fields)


def short_native_fields():
    return long_live.fixture_fields if hasattr(long_live, "fixture_fields") else (
        "policy", "decoded_frame_count", "selected_frame_count", "selected_candidate_indices",
        "selected_source_frame_indices", "source_indices_exact", "selected_frame_timestamps",
        "selected_frame_content_fnv1a64", "has_nominal_timestamps", "temporal2_pair_first_timestamps",
        "max_frames", "max_candidate_frames", "motion_threshold", "scene_threshold", "grid_t",
        "grid_h", "grid_w", "premerge_patch_rows", "merged_tokens_per_slice", "video_token_count",
        "prompt_token_count", "media_hash", "mrope_next", "mrope_fnv1a64", "engine_path")


def execute_live(args, short):
    if os.environ.get("UMMINFER_LIVE_OPT_IN") != "1":
        raise ValueError("controlled live intervention requires UMMINFER_LIVE_OPT_IN=1")
    plan_root = Path(args.plan).resolve()
    plan, oracle = validate_plan(plan_root, deep=True, require_positions=True)
    out = Path(args.out_dir).resolve()
    if out.exists():
        raise ValueError("refusing to overwrite an existing evidence directory")
    url = long_live.loopback_url(args.url)
    if url.rsplit(":", 1)[-1] == "43301":
        raise ValueError("resident service is forbidden")
    provenance = long_live.validate_provenance(
        args.provenance, mode="serial", run_id=args.run_id, source_root=args.source_root)
    if provenance["physical_gpu"] != 0 or provenance["batch_size"] != 1:
        raise ValueError("Task47 live interventions require explicitly idle physical GPU 0 serial only")
    long_live.health_probe(url)
    exclusive = []
    pre, _, _ = long_live.exclusive_check(
        provenance, controlled.query_compute_apps, long_live.query_memory)
    pre["phase"] = "pre_schedule"
    exclusive.append(pre)
    if not pre["passed"]:
        raise ValueError("GPU 0 exclusive-device preflight failed before any model request")
    out.mkdir(parents=True)
    records, steps, cleanup_checks, resources = [], [], [], []
    contract = controlled.parse_contract({
        "schema_version": 1, "warmup_repetitions": WARMUP_REPETITIONS,
        "measured_repetitions": MEASURED_REPETITIONS, "arm_order": "interleaved", "seed": SEED,
        "exclusive_device": {"required": True, "gpu_indexes": [0],
                             "allowed_pids": [provenance["native_stage"].get(
                                 "compute_pid", provenance["native_stage"]["pid"])],
                             "max_other_compute_processes": 0},
        "cleanup": {"require_in_flight_zero": True,
                    "max_residual_memory_mib": provenance["max_residual_memory_mib"]},
    })
    schedule = controlled.build_schedule(list(ARMS), contract)
    model = provenance["model"].get("id", "qwen3-native")
    for step in schedule:
        monitor = long_live.ResourceMonitor(0, long_live.query_memory).start()
        step_record = {"mode": "serial", **step, "budget": BUDGET}
        for case in plan["cases"]:
            arm_record = case["arms"][step["arm"]]
            data_url, video_sha256, _ = encode_intervention_video(plan_root / arm_record["video"])
            if video_sha256 != arm_record["video_sha256"]:
                raise ValueError("derived video bytes changed during live schedule")
            body = short.build_body(data_url, arm_record["prompt"], "temporal", model,
                                    fixture.MOTION_THRESHOLD, fixture.SCENE_THRESHOLD, SEED, 128)
            response, frames, error = None, [], None
            try:
                response, frames = short.stream_chat(url, body, 240.0)
            except Exception as exc:  # retained; no miss is removed or rewritten
                error = f"{type(exc).__name__}: {exc}"[:4096]
            relative = Path("records") / step["phase"] / f"repetition-{step['repetition']}" / \
                step["arm"] / f"{case['id']}.sse.json"
            (out / relative).parent.mkdir(parents=True, exist_ok=True)
            write_json(out / relative, frames)
            source_case = next(item for item in json.loads(
                (plan_root / "source/manifest.json").read_bytes())["cases"]
                               if item["id"] == case["id"])
            record = {
                "mode": "serial", "phase": step["phase"], "repetition": step["repetition"],
                "arm": step["arm"], "budget": BUDGET, "clip": case["id"],
                "schedule_index": len(steps), "status": "COMPLETED" if response else "FAILED",
                "error": error, "response": response,
                "raw_answer": response.get("answer") if response else None,
                "expected_answer_hit": long_live.answer_hit(response["answer"], source_case)
                    if response else None,
                "request": short.request_summary(body, video_sha256),
                "response_frames": relative.as_posix(), "native_evidence_valid": False,
                "original_source_frame_indices": arm_record["original_source_frame_indices"],
                "independent_position_oracle": f"{case['id']}/{step['arm']}",
            }
            records.append(record)
        resource = monitor.stop()
        for record in records:
            if record["schedule_index"] == len(steps):
                record["resources"] = dict(resource, attribution="shared_by_four_clips_in_schedule_step")
        cleanup = controlled.evaluate_cleanup(
            contract, in_flight=long_live.health_probe(url)["in_flight"],
            residual_mib={0: resource["final_mib"]})
        cleanup["schedule_index"] = len(steps)
        cleanup_checks.append(cleanup)
        resources.append(resource)
        steps.append({**step_record, "resources": resource, "cleanup": cleanup})
    native_records = short.parse_native_log(Path(args.native_log).resolve())
    execution_records = short.parse_native_execution_log(Path(args.native_log).resolve())
    baseline_checks = []
    for record in records:
        if record["status"] != "COMPLETED":
            continue
        case = plan_case(plan, record["clip"])
        arm_record = case["arms"][record["arm"]]
        try:
            request_id = record["response"]["request_id"]
            record["native"] = short.native_for_request(native_records, request_id)
            record["native_execution"] = short.execution_for_request(execution_records, request_id)
            native_validate(record, arm_record, oracle[record["independent_position_oracle"]], short)
            record["latency_ms"] = latency_record(record["response"], record["native_execution"])
            record["native_evidence_valid"] = True
            baseline = arm_record.get("r4_baseline")
            if baseline is not None:
                native_equal = all(record["native"][field] == baseline["native"][field]
                                   for field in R4_NATIVE_FIELDS)
                answer_equal = record["raw_answer"] == baseline["raw_answer"]
                passed = native_equal and answer_equal
                baseline_checks.append({"clip": record["clip"], "arm": record["arm"],
                                        "phase": record["phase"], "repetition": record["repetition"],
                                        "native_equal": native_equal, "answer_equal": answer_equal,
                                        "before_answer": baseline["raw_answer"],
                                        "after_answer": record["raw_answer"], "passed": passed})
        except (OSError, ValueError, RuntimeError) as exc:
            record["native_validation_error"] = str(exc)[:4096]
    post, _, _ = long_live.exclusive_check(
        provenance, controlled.query_compute_apps, long_live.query_memory)
    post["phase"] = "post_schedule"
    exclusive.append(post)
    stale_errors = []
    try:
        long_live.validate_provenance(args.provenance, mode="serial", run_id=args.run_id,
                                      source_root=args.source_root)
        validate_plan(plan_root, deep=True, require_positions=True)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        stale_errors.append(f"frozen_stack_and_data: {exc}")
    parity = control_checks(records)
    reasons = list(stale_errors)
    if any(not row["passed"] for row in exclusive) or any(not row["passed"] for row in cleanup_checks):
        reasons.append("exclusive_device_or_cleanup_failed")
    if any(row["status"] != "COMPLETED" or not row["native_evidence_valid"] for row in records):
        reasons.append("request_or_native_evidence_failed")
    wanted = {(case["id"], arm, phase, repetition) for case in plan["cases"] for arm in ARMS
              for phase, count in (("warmup", WARMUP_REPETITIONS),
                                   ("measured", MEASURED_REPETITIONS))
              for repetition in range(1, count + 1)}
    actual = {(row["clip"], row["arm"], row["phase"], row["repetition"]) for row in records}
    if actual != wanted:
        reasons.append("complete_repetitions_failed")
    if any(not row["passed"] for row in parity):
        reasons.append("controlled_parity_or_repetition_failed")
    if any(not row["passed"] for row in baseline_checks):
        reasons.append("r4_baseline_identity_or_answer_parity_failed")
    gate_pass = not reasons
    report = {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": controlled.CONTROLLED_LIVE if gate_pass else controlled.UNCONTROLLED_LIVE,
        "fixture_sha256": plan["source_manifest_sha256"], "arms": list(ARMS), "modes": ["serial"],
        "records": records, "steps": steps, "resources": resources,
        "cleanup_checks": cleanup_checks, "exclusive_device": exclusive,
        "provenance": {"serial": long_live.public_provenance(provenance)},
        "parity": parity, "r4_baseline_parity": baseline_checks,
        "aggregates": long_live.aggregate_results(records), "answer_outcomes": answer_outcomes(records),
        "gate_rules": ["cold_prefix_zero", "exclusive_gpu0_serial", "frozen_stack_and_data",
                       "exact_selected_rgb_pts_order", "independent_full_mrope_positions",
                       "complete_repetitions", "control_factor_isolation", "artifact_checksums"],
        "gate_reasons": sorted(set(reasons)), "gate_pass": gate_pass,
        "claims": {"real_generation": "MEASURED_LIMITED_FIXTURE",
                   "live_latency": "CONTROLLED_LIVE" if gate_pass else "UNCONTROLLED_LIVE",
                   "perceptual_quality": "NOT_RUN", "generalized_quality_claim": False},
        "answer_scope": "unchanged original expected substrings and raw answers only; "
                        "alternate-language controls are not given new scoring aliases; "
                        "no perceptual/generalized quality claim",
        "oracle_audit": json.loads((plan_root / "audit.json").read_bytes()),
        "r4_evidence_sha256sums": plan["r4_evidence_sha256sums"],
    }
    shutil.copytree(plan_root, out / "input")
    write_evidence_bundle(out, report)
    print(json.dumps({"records": len(records), "gate_pass": gate_pass,
                      "gate_reasons": report["gate_reasons"],
                      "answer_outcomes": report["answer_outcomes"],
                      "evidence_class": report["evidence_class"]},
                     ensure_ascii=False, indent=2))
    return 0 if gate_pass else 1


def seal_evidence(out_dir):
    """Seal nested inputs correctly; only root bookkeeping files are special."""
    out = Path(out_dir)
    files = sorted(path for path in out.rglob("*") if path.is_file())
    artifact_rows = []
    for path in files:
        relative = path.relative_to(out).as_posix()
        if relative in ("artifact-manifest.json", "SHA256SUMS"):
            continue
        artifact_rows.append({"path": relative, "bytes": path.stat().st_size,
                              "sha256": sha256_file(path)})
    write_json(out / "artifact-manifest.json",
               {"schema_version": 1, "hash": "sha256", "files": artifact_rows})
    sums = sorted(path for path in out.rglob("*") if path.is_file()
                  and path.relative_to(out).as_posix() != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(
        f"{sha256_file(path)}  {path.relative_to(out).as_posix()}\n" for path in sums),
        encoding="ascii")


def validate_evidence_files(out_dir):
    out = Path(out_dir).resolve()
    declared = {}
    for line in (out / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        if not long_live.HEX64.fullmatch(digest) or relative.startswith("/") \
                or ".." in Path(relative).parts or relative in declared:
            raise ValueError("malformed or duplicate evidence checksum")
        declared[relative] = digest
    files = {path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file()}
    expected_sums = files - {"SHA256SUMS"}
    if set(declared) != expected_sums:
        raise ValueError("missing, extra, renamed, or nested evidence checksum mismatch")
    for relative, digest in declared.items():
        if sha256_file(out / relative) != digest:
            raise ValueError(f"altered evidence checksum: {relative}")
    artifact = json.loads((out / "artifact-manifest.json").read_bytes())
    listed = {row["path"]: row for row in artifact.get("files", [])}
    expected_artifacts = files - {"artifact-manifest.json", "SHA256SUMS"}
    if set(listed) != expected_artifacts:
        raise ValueError("artifact manifest does not cover the complete nested evidence set")
    for relative, row in listed.items():
        path = out / relative
        if row["bytes"] != path.stat().st_size or row["sha256"] != sha256_file(path):
            raise ValueError(f"artifact manifest mismatch: {relative}")
    report = json.loads((out / "report.json").read_bytes())
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported evidence report schema")
    if long_live.aggregate_results(report["records"]) != report["aggregates"] \
            or report.get("answer_outcomes") != answer_outcomes(report["records"]) \
            or report.get("parity") != control_checks(report["records"]):
        raise ValueError("independent aggregate/answer/control recomputation failed")
    if not report["gate_pass"] and not report["gate_reasons"]:
        raise ValueError("failed gate has no fail-closed reason")
    return report


def write_evidence_bundle(out_dir, report):
    write_json(Path(out_dir) / "report.json", report)
    seal_evidence(out_dir)
    validate_bundle(out_dir, deep=False)


def reseal_evidence(out_dir):
    # Does not alter report.json, records, answers, or any input bytes.
    seal_evidence(out_dir)
    return validate_bundle(out_dir, deep=True)


def validate_bundle(out_dir, *, deep=True):
    out = Path(out_dir).resolve()
    report = validate_evidence_files(out)
    plan, oracle = validate_plan(out / "input", deep=deep, require_positions=True)
    if report.get("answer_outcomes") != answer_outcomes(report["records"]):
        raise ValueError("independent per-arm answer recomputation failed")
    if report.get("parity") != control_checks(report["records"]):
        raise ValueError("independent control/repetition recomputation failed")
    source_manifest = json.loads((out / "input/source/manifest.json").read_bytes())
    for record in report["records"]:
        if record["status"] == "COMPLETED" and record["native_evidence_valid"]:
            case = plan_case(plan, record["clip"])
            native_validate(record, case["arms"][record["arm"]],
                            oracle[record["independent_position_oracle"]],
                            __import__("video_temporal_live"))
            source_case = next(case for case in source_manifest["cases"] if case["id"] == record["clip"])
            if record["expected_answer_hit"] != long_live.answer_hit(record["raw_answer"], source_case):
                raise ValueError("original expected-substring result changed during verification")
    return report
