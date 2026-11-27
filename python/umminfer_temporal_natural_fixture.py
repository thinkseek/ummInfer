#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Freeze and validate eight excerpts of existing repository natural footage.

The only source is the historical, repository-tracked real-footage compilation
identified in preregistration.json. No procedural renderer or model output is
used. Generation never performs inference. Validation independently decodes all
frozen excerpt frames, source PTS, B-frame packet order, and selection oracles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE = ROOT / "benchmarks" / "video-temporal-natural"
DEFAULT_FREEZE = ROOT / "benchmarks" / "video-temporal-natural.freeze.json"
REVISION = "temporal-natural-r2"
WIDTH, HEIGHT, FPS = 256, 160, 30
FRAME_BYTES = WIDTH * HEIGHT * 3
BUDGETS = (4, 8)
ARM_BUDGETS = (("default", 8), ("uniform", 4), ("uniform", 8),
               ("temporal", 4), ("temporal", 8))
MOTION_THRESHOLD, SCENE_THRESHOLD = 0.02, 0.25
HEX256 = re.compile(r"[0-9a-f]{64}\Z")
TASK66_MANIFEST_SHA256 = "25c51b6c97cdf9bd6b95991da8a424e8a366ae3105a59099a181d22b65eb2db8"
TASK66_PREREGISTRATION_SHA256 = "74eb3eac15233ec5fe508168dfa6b974259544b5b29ef9c28f74ca3539d24bbe"
TASK66_SUMS_SHA256 = "9a0d0389ac38f48e91dd251dacd6d5e1fd398e01fd7ca52a6705b3585468c000"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, ensure_ascii=False, indent=2).encode() + b"\n")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fnv1a64(data):
    value = 14695981039346656037
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def run(args, *, timeout=180):
    return subprocess.run([str(arg) for arg in args], check=True,
                          capture_output=True, timeout=timeout)


def probe(path, ffprobe="ffprobe"):
    result = run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                  "stream=codec_name,profile,width,height,duration,nb_frames,avg_frame_rate,"
                  "has_b_frames,pix_fmt:format=duration,size", "-of", "json", path])
    return json.loads(result.stdout)


def decode(path, vf=None, *, frames=None, fps=False, ffmpeg="ffmpeg"):
    """Decode presentation-order RGB and finite source PTS from showinfo."""
    if vf is None:
        vf = f"scale={WIDTH}:{HEIGHT}:flags=bicubic"
    args = [ffmpeg, "-v", "info", "-i", Path(path), "-an", "-sn", "-dn",
            "-vf", vf + ",showinfo"]
    if not fps:
        args += ["-vsync", "0"]
    if frames is not None:
        args += ["-frames:v", str(frames)]
    args += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    result = run(args)
    if not result.stdout or len(result.stdout) % FRAME_BYTES:
        raise ValueError(f"partial RGB decode: {path}")
    decoded = [result.stdout[offset:offset + FRAME_BYTES]
               for offset in range(0, len(result.stdout), FRAME_BYTES)]
    timestamps = []
    for line in result.stderr.decode("utf-8", errors="replace").splitlines():
        if " n:" not in line or "pts_time:" not in line:
            continue
        match = re.match(r"[-+0-9.eE]+", line.split("pts_time:", 1)[1].lstrip())
        if match and math.isfinite(float(match.group(0))):
            timestamps.append(float(match.group(0)))
    if len(timestamps) != len(decoded):
        raise ValueError(f"showinfo/RGB count mismatch: {path}")
    return decoded, timestamps


def packet_evidence(path, ffprobe="ffprobe"):
    value = json.loads(run([
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_packets",
        "-show_entries", "packet=pts,dts,flags", "-of", "json", path]).stdout)
    packets = value["packets"]
    pts = [int(packet["pts"]) for packet in packets]
    dts = [int(packet["dts"]) for packet in packets]
    return {
        "packet_count": len(packets), "decode_order_pts": pts, "decode_order_dts": dts,
        "decode_order_pts_nonmonotonic": any(b < a for a, b in zip(pts, pts[1:])),
        "decode_order_dts_strictly_increasing": all(b > a for a, b in zip(dts, dts[1:])),
        "packets": packets,
    }


def analyze(frames):
    analyzed = []
    for frame in frames:
        luma, histogram = [], [0] * 16
        for ay in range(16):
            y0, y1 = ay * HEIGHT // 16, (ay + 1) * HEIGHT // 16
            for ax in range(16):
                x0, x1 = ax * WIDTH // 16, (ax + 1) * WIDTH // 16
                total = count = 0
                for y in range(y0, y1):
                    for x in range(x0, x1):
                        offset = (y * WIDTH + x) * 3
                        total += (77 * frame[offset] + 150 * frame[offset + 1]
                                  + 29 * frame[offset + 2]) >> 8
                        count += 1
                mean = total // count
                luma.append(mean)
                histogram[mean * 16 // 256] += 1
        analyzed.append((luma, histogram))
    scores = [[0.0, 0.0]]
    for index in range(1, len(frames)):
        previous, current = analyzed[index - 1], analyzed[index]
        scores.append([
            sum(abs(a - b) for a, b in zip(previous[0], current[0])) / (255.0 * 256.0),
            sum(abs(a - b) for a, b in zip(previous[1], current[1])) / (2.0 * 256.0),
        ])
    return scores


def score_runs(scores, threshold, component):
    result = []
    index = 1
    while index < len(scores):
        if not scores[index][component] > threshold:
            index += 1
            continue
        end = index
        while end < len(scores) and scores[end][component] > threshold:
            end += 1
        result.append(list(range(index, end)))
        index = end
    return result


def select_motion_run(scores, budget):
    if budget not in BUDGETS or not scores:
        raise ValueError("invalid natural-fixture selection budget")
    target = min(len(scores), budget)
    if len(scores) <= target:
        return list(range(len(scores)))
    used, ordered = set(), []

    def take(index):
        if len(ordered) < target and index not in used:
            used.add(index)
            ordered.append(index)

    take(0)
    take(len(scores) - 1)
    scenes = [min(run, key=lambda i: (-scores[i][1], -scores[i][0], i))
              for run in score_runs(scores, SCENE_THRESHOLD, 1)]
    scenes.sort(key=lambda i: (-scores[i][1], -scores[i][0], i))
    for index in scenes:
        take(index)
    motion = [min(run, key=lambda i: (-scores[i][0], -scores[i][1], i))
              for run in score_runs(scores, MOTION_THRESHOLD, 0)]
    motion.sort(key=lambda i: (-scores[i][0], -scores[i][1], i))
    for index in motion:
        take(index)
    for slot in range(target):
        if len(ordered) >= target:
            break
        take((slot * (len(scores) - 1) + (target - 1) // 2) // (target - 1))
    for index in range(len(scores)):
        if len(ordered) >= target:
            break
        take(index)
    if len(ordered) != target:
        raise ValueError("accepted selector failed to fill natural-fixture budget")
    return sorted(ordered)


def selection_record(candidates, sources, timestamps, frames):
    return {
        "candidate_indices": candidates, "source_frame_indices": sources,
        "source_timestamps_seconds": timestamps,
        "content_fnv1a64": [fnv1a64(frame) for frame in frames],
        "content_sha256": [sha256_bytes(frame) for frame in frames],
        "temporal2_pair_first_timestamps": [
            timestamps[min(2 * index, len(timestamps) - 1)]
            for index in range((len(timestamps) + 1) // 2)],
    }


def derive_oracles(video_path, *, ffmpeg="ffmpeg", ffprobe="ffprobe"):
    frames, pts = decode(video_path, ffmpeg=ffmpeg)
    count = len(frames)
    metadata = probe(video_path, ffprobe)
    stream = metadata["streams"][0]
    if int(stream["nb_frames"]) != count or stream["codec_name"] != "h264" \
            or int(stream["width"]) != WIDTH or int(stream["height"]) != HEIGHT:
        raise ValueError(f"natural excerpt identity/dimensions changed: {video_path}")
    if pts != sorted(pts) or len(set(pts)) != count:
        raise ValueError(f"presentation PTS are not strictly increasing: {video_path}")
    packets = packet_evidence(video_path, ffprobe)
    if packets["packet_count"] != count or not packets["decode_order_pts_nonmonotonic"] \
            or not packets["decode_order_dts_strictly_increasing"]:
        raise ValueError(f"B-frame decode/presentation evidence failed: {video_path}")
    scores = analyze(frames)
    native_duration = float(stream.get("duration") or metadata["format"]["duration"])
    temporal = {}
    for budget in BUDGETS:
        candidates = select_motion_run(scores, budget)
        temporal[str(budget)] = selection_record(
            candidates, candidates, [pts[index] for index in candidates],
            [frames[index] for index in candidates])
    uniform = {}
    for budget in BUDGETS:
        if budget == 8:
            duration = native_duration
            selected, timestamps = decode(
                video_path, f"fps={max(0.25, budget / duration):.4f},"
                f"scale={WIDTH}:{HEIGHT}:flags=bicubic", frames=budget, fps=True,
                ffmpeg=ffmpeg)
            by_hash = {}
            for index, frame in enumerate(frames):
                by_hash.setdefault(sha256_bytes(frame), []).append(index)
            sources = []
            source_candidates = []
            for frame in selected:
                matches = by_hash.get(sha256_bytes(frame), [])
                if not matches:
                    raise ValueError(f"unbound natural uniform-eight source pixels: {video_path}")
                # Historical uniform-eight exposes no source index. Natural
                # sources can contain exact repeated frames, so retain every
                # matching source/PTS candidate instead of inventing uniqueness.
                source_candidates.append(matches)
                sources.append(matches[0])
            uniform[str(budget)] = selection_record(
                list(range(budget)), sources, [pts[index] for index in sources], selected)
            uniform[str(budget)]["source_frame_candidates"] = source_candidates
            uniform[str(budget)]["source_pts_candidates"] = [
                [pts[index] for index in matches] for matches in source_candidates]
            uniform[str(budget)]["source_frame_binding"] = (
                "EXACT" if all(len(matches) == 1 for matches in source_candidates)
                else "AMBIGUOUS_EXACT_REPEATED_SOURCE_RGB_ALL_CANDIDATES_RETAINED")
            uniform[str(budget)]["output_filter_timestamps_seconds"] = timestamps
            # Native still exposes nominal timestamps; source PTS candidates
            # above are independently rebound from the complete source decode.
            uniform[str(budget)]["native_nominal_timestamps"] = [
                duration * index / budget for index in range(budget)]
        else:
            sources = [(slot * (count - 1) + (budget - 1) // 2) // (budget - 1)
                       for slot in range(budget)]
            expression = "+".join(f"eq(n,{index})" for index in sources)
            selected, timestamps = decode(
                video_path, f"select='{expression}',scale={WIDTH}:{HEIGHT}:flags=bicubic",
                frames=budget, ffmpeg=ffmpeg)
            if timestamps != [pts[index] for index in sources]:
                raise ValueError(f"explicit uniform source PTS mismatch: {video_path}")
            uniform[str(budget)] = selection_record(
                list(range(budget)), sources, timestamps, selected)
    content = [{"source_frame_index": index, "source_timestamp_seconds": pts[index],
                "decoded_native_rgb_sha256": sha256_bytes(frame),
                "decoded_native_rgb_fnv1a64": fnv1a64(frame)}
               for index, frame in enumerate(frames)]
    return {"frames": frames, "pts": pts, "scores": scores, "native_duration_seconds": native_duration,
            "temporal": temporal,
            "uniform": uniform, "content": content, "packets": packets,
            "probe": metadata}


def encode_excerpt(source, spec, output, ffmpeg):
    start, end = spec["source_frame_range"]
    count = end - start + 1
    args = [ffmpeg, "-y", "-v", "error", "-i", source, "-map", "0:v:0",
            "-an", "-sn", "-dn", "-vf",
            f"select='between(n,{start},{end})',scale={WIDTH}:{HEIGHT}:flags=bicubic,"
            f"setpts=N/({FPS}*TB)", "-vsync", "0", "-map_metadata", "-1",
            "-c:v", "libx264", "-preset", "veryslow", "-crf", "12", "-threads", "1",
            "-g", str(FPS), "-bf", "2", "-pix_fmt", "yuv420p",
            "-video_track_timescale", str(FPS), "-fflags", "+bitexact",
            "-flags:v", "+bitexact", "-movflags", "+faststart", "-frames:v", str(count),
            output]
    run(args)


def review_sheet(video_path, output, ffmpeg):
    # Neutral visual review artifact: no labels, answers, or metadata are overlaid.
    run([ffmpeg, "-y", "-v", "error", "-i", video_path, "-an", "-vf",
         "select='not(mod(n,5))',tile=4x1", "-vsync", "0", "-frames:v", "1",
         "-update", "1", output])


def seal_fixture(bundle):
    bundle = Path(bundle)
    (bundle / "SHA256SUMS").write_text("".join(
        f"{sha256_file(path)}  {path.relative_to(bundle).as_posix()}\n"
        for path in sorted(bundle.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"),
        encoding="ascii")


def validate_sums(bundle):
    bundle = Path(bundle)
    declared = {}
    for line in (bundle / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        if not HEX256.fullmatch(digest) or relative.startswith("/") \
                or ".." in Path(relative).parts or relative in declared:
            raise ValueError("malformed or duplicate natural fixture checksum")
        declared[relative] = digest
    actual = {path.relative_to(bundle).as_posix() for path in bundle.rglob("*")
              if path.is_file() and path.name != "SHA256SUMS"}
    if set(declared) != actual:
        raise ValueError("natural fixture checksum file-set mismatch")
    for relative, digest in declared.items():
        if sha256_file(bundle / relative) != digest:
            raise ValueError(f"natural fixture checksum mismatch: {relative}")
    return len(declared)


def validate_case_contract(prereg, manifest):
    cases = prereg["cases"]
    if len(cases) != 8 or len({case["id"] for case in cases}) != 8 \
            or len({case["language"] for case in cases}) != 8:
        raise ValueError("natural gate requires eight distinct cases and languages")
    ranges = [case["source_frame_range"] for case in cases]
    if len({tuple(value) for value in ranges}) != 8:
        raise ValueError("natural excerpts are not distinct")
    ordered = sorted((value[0], value[1]) for value in ranges)
    if any(start > end or start < 0 for start, end in ordered) \
            or any(ordered[index][1] >= ordered[index + 1][0]
                   for index in range(len(ordered) - 1)):
        raise ValueError("natural excerpt source ranges overlap or are invalid")
    for case in cases:
        folded = case["question"].casefold()
        if not case["expect_all"] or any(term.casefold() in folded for term in case["expect_all"]):
            raise ValueError(f"question answer leakage: {case['id']}")
        if case["id"] != case["video_id"] or not re.fullmatch(r"natural-0[1-8]", case["id"]):
            raise ValueError("case/video identifiers must remain neutral")
    if [case["id"] for case in manifest["cases"]] != [case["id"] for case in cases]:
        raise ValueError("manifest case order differs from preregistration")
    for case, recorded in zip(cases, manifest["cases"]):
        for field in ("id", "video_id", "language", "question", "expect_all", "expect_any_group"):
            if recorded[field] != case[field]:
                raise ValueError(f"manifest scoring/prompt drift: {case['id']}/{field}")


def generate_bundle(output, source, preregistration, *, ffmpeg="ffmpeg", ffprobe="ffprobe"):
    output, source = Path(output), Path(source)
    if output.exists() and any(output.iterdir()):
        raise ValueError("natural fixture output must be a new empty directory")
    output.mkdir(parents=True, exist_ok=True)
    prereg = json.loads(Path(preregistration).read_bytes())
    if prereg.get("fixture_revision") != REVISION or prereg["claims_before_live"][
            "natural_video_fixture_accuracy"] != "NOT_RUN":
        raise ValueError("unsupported or already-used natural preregistration")
    validate_case_contract(prereg, {"cases": prereg["cases"]})
    source_info = prereg["source"]
    if sha256_file(source) != source_info["original_sha256"]:
        raise ValueError("natural source SHA-256 differs from preregistration")
    (output / "preregistration.json").write_bytes(Path(preregistration).read_bytes())
    videos, cases = {}, []
    for spec in prereg["cases"]:
        cid = spec["id"]
        relative = f"fixtures/{cid}.mp4"
        (output / relative).parent.mkdir(parents=True, exist_ok=True)
        encode_excerpt(source, spec, output / relative, ffmpeg)
        oracle = derive_oracles(output / relative, ffmpeg=ffmpeg, ffprobe=ffprobe)
        content_relative = f"content/{cid}.frames.jsonl"
        packets_relative = f"content/{cid}.packets.json"
        (output / content_relative).parent.mkdir(parents=True, exist_ok=True)
        (output / content_relative).write_text("".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in oracle["content"]),
            encoding="utf-8")
        write_json(output / packets_relative, oracle["packets"])
        review_relative = f"review/{cid}.png"
        (output / review_relative).parent.mkdir(parents=True, exist_ok=True)
        review_sheet(output / relative, output / review_relative, ffmpeg)
        count = len(oracle["frames"])
        video = {
            "video": relative, "video_sha256": sha256_file(output / relative),
            "video_bytes": (output / relative).stat().st_size,
            "source_frame_range": spec["source_frame_range"], "frame_count": count,
            "candidate_count": count, "candidate_stride": 1,
            "duration_seconds": oracle["native_duration_seconds"],
            "native_decode": {"width": WIDTH, "height": HEIGHT, "fps": FPS},
            "ffprobe": oracle["probe"], "content_frames": content_relative,
            "packets": packets_relative, "review_sheet": review_relative,
            "decode_order": {key: oracle["packets"][key] for key in (
                "decode_order_pts_nonmonotonic", "decode_order_dts_strictly_increasing")},
            "candidate_scores_motion_scene": oracle["scores"],
            "expected_temporal": oracle["temporal"], "expected_uniform": oracle["uniform"],
        }
        videos[cid] = video
        case = {key: spec[key] for key in (
            "id", "video_id", "language", "question", "expect_all", "expect_any_group",
            "evaluation_role", "ground_truth")}
        case.update({"video": relative, "video_sha256": video["video_sha256"],
                     "candidate_count": count, "expected_temporal": oracle["temporal"],
                     "expected_uniform_8": oracle["uniform"]["8"], "anchor_audit": {}})
        for arm, budget in ARM_BUDGETS:
            expected = oracle["uniform"][str(budget)] if arm in ("default", "uniform") \
                else oracle["temporal"][str(budget)]
            required = set(range(count)) if spec["ground_truth"].get(
                "evidence_visible_throughout_excerpt") else set(spec["ground_truth"]["source_frames"])
            present = sorted(required & set(expected["source_frame_indices"]))
            case["anchor_audit"][f"{arm}-{budget}"] = {
                "answer_anchor_present": bool(present),
                "evidence_source_frames_present": present,
            }
        if not all(row["answer_anchor_present"] for row in case["anchor_audit"].values()):
            raise ValueError(f"preregistered natural answer evidence absent: {cid}")
        cases.append(case)
    manifest = {
        "schema_version": 1, "fixture_revision": REVISION,
        "natural_video_claim": True,
        "natural_video_scope": "EIGHT_NONOVERLAPPING_EXCERPTS_OF_ONE_EXISTING_REAL_FOOTAGE_COMPILATION",
        "source": source_info, "preregistration_sha256": sha256_file(output / "preregistration.json"),
        "selection": {"selector": "qwen38_temporal_motion_run_v1", "candidate_stride": 1,
                      "max_candidate_frames": 64, "motion_threshold": MOTION_THRESHOLD,
                      "scene_threshold": SCENE_THRESHOLD, "budgets": list(BUDGETS)},
        "request_contract": prereg["request_contract"], "measurement": prereg["measurement"],
        "analysis_plan": prereg["analysis_plan"], "videos": videos, "cases": cases,
        "claims": prereg["claims_before_live"],
        "historical_task66_control": {
            "manifest_sha256": TASK66_MANIFEST_SHA256,
            "preregistration_sha256": TASK66_PREREGISTRATION_SHA256,
            "sha256sums_sha256": TASK66_SUMS_SHA256,
            "primary_temporal8_hits": 18, "primary_uniform8_hits": 18,
            "primary_sign_test_p": 1.0, "included_in_natural_denominator": False,
            "scores_or_fixture_bytes_changed": False,
        },
    }
    write_json(output / "manifest.json", manifest)
    seal_fixture(output)
    return validate_bundle(output, deep=True, ffmpeg=ffmpeg, ffprobe=ffprobe)


def validate_bundle(bundle, *, deep=True, ffmpeg="ffmpeg", ffprobe="ffprobe"):
    bundle = Path(bundle)
    if validate_sums(bundle) < 1:
        raise ValueError("empty natural fixture")
    prereg = json.loads((bundle / "preregistration.json").read_bytes())
    manifest = json.loads((bundle / "manifest.json").read_bytes())
    if manifest.get("fixture_revision") != REVISION or manifest.get("natural_video_claim") is not True \
            or len(manifest.get("videos", {})) != 8 or len(manifest.get("cases", [])) != 8:
        raise ValueError("natural fixture revision/population mismatch")
    if manifest["preregistration_sha256"] != sha256_file(bundle / "preregistration.json"):
        raise ValueError("manifest is bound to another natural preregistration")
    validate_case_contract(prereg, manifest)
    if manifest["historical_task66_control"] != {
            "manifest_sha256": TASK66_MANIFEST_SHA256,
            "preregistration_sha256": TASK66_PREREGISTRATION_SHA256,
            "sha256sums_sha256": TASK66_SUMS_SHA256,
            "primary_temporal8_hits": 18, "primary_uniform8_hits": 18,
            "primary_sign_test_p": 1.0, "included_in_natural_denominator": False,
            "scores_or_fixture_bytes_changed": False}:
        raise ValueError("Task66 regression-control declaration changed")
    for cid, video in manifest["videos"].items():
        path = bundle / video["video"]
        if sha256_file(path) != video["video_sha256"]:
            raise ValueError(f"natural video hash mismatch: {cid}")
        if not deep:
            continue
        oracle = derive_oracles(path, ffmpeg=ffmpeg, ffprobe=ffprobe)
        rows = [json.loads(line) for line in
                (bundle / video["content_frames"]).read_text().splitlines()]
        if rows != oracle["content"] \
                or json.loads((bundle / video["packets"]).read_bytes()) != oracle["packets"] \
                or video["candidate_scores_motion_scene"] != oracle["scores"] \
                or video["expected_temporal"] != oracle["temporal"] \
                or video["expected_uniform"] != oracle["uniform"] \
                or video["frame_count"] != len(oracle["frames"]):
            raise ValueError(f"independent full natural decode/oracle mismatch: {cid}")
        for budget in BUDGETS:
            selected = video["expected_temporal"][str(budget)]["source_frame_indices"]
            if selected[0] != 0 or selected[-1] != len(rows) - 1:
                raise ValueError(f"temporal first/last anchor missing: {cid}/{budget}")
    return manifest


def validate_freeze(bundle, receipt):
    bundle, receipt = Path(bundle), Path(receipt)
    value = json.loads(receipt.read_bytes())
    expected = {
        "schema_version": 1, "fixture_revision": REVISION,
        "frozen_before_any_model_output_for_this_revision": True,
        "model_outputs_seen_for_this_revision_at_freeze": False,
        "prior_excluded_natural_outputs_seen_at_freeze": True,
        "prior_excluded_attempt": "format-duration-r1",
        "preregistration_sha256": sha256_file(bundle / "preregistration.json"),
        "manifest_sha256": sha256_file(bundle / "manifest.json"),
        "sha256sums_sha256": sha256_file(bundle / "SHA256SUMS"),
        "quality_case_ids": [f"natural-0{index}" for index in range(1, 9)],
        "task66_control_scores_or_fixture_changed": False,
    }
    if value != expected:
        raise ValueError("natural freeze receipt mismatch")
    validate_sums(bundle)
    return value


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--source", required=True)
    generate.add_argument("--preregistration", required=True)
    generate.add_argument("--output", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    validate.add_argument("--shallow", action="store_true")
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    freeze.add_argument("--receipt", default=str(DEFAULT_FREEZE))
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "generate":
            manifest = generate_bundle(args.output, args.source, args.preregistration)
            print(json.dumps({"result": "PASS", "videos": len(manifest["videos"]),
                              "cases": len(manifest["cases"]), "deep_decode": True}, indent=2))
        elif args.command == "validate":
            manifest = validate_bundle(args.bundle, deep=not args.shallow)
            print(json.dumps({"result": "PASS", "videos": len(manifest["videos"]),
                              "cases": len(manifest["cases"]),
                              "deep_decode": not args.shallow}, indent=2))
        else:
            manifest = validate_bundle(args.bundle, deep=True)
            value = {
                "schema_version": 1, "fixture_revision": REVISION,
                "frozen_before_any_model_output_for_this_revision": True,
                "model_outputs_seen_for_this_revision_at_freeze": False,
                "prior_excluded_natural_outputs_seen_at_freeze": True,
                "prior_excluded_attempt": "format-duration-r1",
                "preregistration_sha256": sha256_file(Path(args.bundle) / "preregistration.json"),
                "manifest_sha256": sha256_file(Path(args.bundle) / "manifest.json"),
                "sha256sums_sha256": sha256_file(Path(args.bundle) / "SHA256SUMS"),
                "quality_case_ids": [case["id"] for case in manifest["cases"]],
                "task66_control_scores_or_fixture_changed": False,
            }
            write_json(args.receipt, value)
            print(json.dumps(value, indent=2))
        return 0
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"NATURAL FIXTURE FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
