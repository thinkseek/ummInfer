#!/usr/bin/env python3
"""Fresh-process, stdlib-only independent verifier for Task72 natural evidence.

No producer, generator, transport, fixture, or scoring module is imported. Every
frozen excerpt is fully decoded again; RGB, source PTS candidates, B-frame order,
accepted motion-run selection, temporal-2/mRoPE invariants, requests, raw SSE
answers, exact statistics, misses, outliers, controls, and checksums are rebuilt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

HEX256 = re.compile(r"[0-9a-f]{64}\Z")
MODES = ("serial", "batch4")
BUDGETS = (4, 8)
ARM_BUDGETS = (("default", 8), ("uniform", 4), ("uniform", 8),
               ("temporal", 4), ("temporal", 8))
WIDTH, HEIGHT = 256, 160
FRAME_BYTES = WIDTH * HEIGHT * 3
ANALYSIS_SOURCES = {
    "python/umminfer_temporal_natural_fixture.py", "tests/temporal_natural_live.py",
    "tests/temporal_natural_test.py", "tests/temporal_natural_verify.py",
}
NATIVE_PARITY_FIELDS = (
    "policy", "decoded_frame_count", "selected_frame_count", "selected_candidate_indices",
    "selected_source_frame_indices", "source_indices_exact", "selected_frame_timestamps",
    "selected_frame_content_fnv1a64", "has_nominal_timestamps", "temporal2_pair_first_timestamps",
    "max_frames", "max_candidate_frames", "motion_threshold", "scene_threshold", "grid_t",
    "grid_h", "grid_w", "premerge_patch_rows", "merged_tokens_per_slice", "video_token_count",
    "prompt_token_count", "media_hash", "mrope_next", "mrope_fnv1a64",
)
TASK66_HASHES = {
    "manifest.json": "25c51b6c97cdf9bd6b95991da8a424e8a366ae3105a59099a181d22b65eb2db8",
    "preregistration.json": "74eb3eac15233ec5fe508168dfa6b974259544b5b29ef9c28f74ca3539d24bbe",
    "SHA256SUMS": "9a0d0389ac38f48e91dd251dacd6d5e1fd398e01fd7ca52a6705b3585468c000",
}


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


def verify_sums(root, *, fixture=False, legacy_basename_exclude=False):
    root = Path(root)
    sums = {}
    for line in (root / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        if not HEX256.fullmatch(digest) or relative.startswith("/") \
                or ".." in Path(relative).parts or relative in sums:
            raise ValueError("malformed or duplicate checksum entry")
        sums[relative] = digest
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and (path.name != "SHA256SUMS" if legacy_basename_exclude
                                     else path.relative_to(root).as_posix() != "SHA256SUMS")}
    if set(sums) != actual:
        raise ValueError("checksum file-set mismatch")
    for relative, digest in sums.items():
        if sha256_file(root / relative) != digest:
            raise ValueError(f"checksum mismatch: {relative}")
    if not fixture:
        artifact = json.loads((root / "artifact-manifest.json").read_bytes())
        listed = {row["path"]: row for row in artifact["files"]}
        if set(listed) != actual - {"artifact-manifest.json"}:
            raise ValueError("artifact manifest does not cover complete evidence")
        for relative, row in listed.items():
            path = root / relative
            if row["bytes"] != path.stat().st_size or row["sha256"] != sha256_file(path):
                raise ValueError(f"artifact manifest mismatch: {relative}")
    return len(sums)


def decode(video, *, uniform8=False, select=None):
    if select is not None:
        vf = f"select='{select}',scale={WIDTH}:{HEIGHT}:flags=bicubic"
    elif uniform8:
        probe = json.loads(subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=duration:format=duration", "-of", "json", str(video)],
            check=True, capture_output=True, timeout=60).stdout)
        duration = float(probe["streams"][0].get("duration") or probe["format"]["duration"])
        vf = f"fps={max(0.25, 8 / duration):.4f},scale={WIDTH}:{HEIGHT}:flags=bicubic"
    else:
        vf = f"scale={WIDTH}:{HEIGHT}:flags=bicubic"
    args = ["ffmpeg", "-v", "info", "-i", str(video), "-an", "-sn", "-dn",
            "-vf", vf + ",showinfo"]
    if not uniform8:
        args += ["-vsync", "0"]
    args += ["-frames:v", "8" if uniform8 else str(64), "-f", "rawvideo",
             "-pix_fmt", "rgb24", "pipe:1"]
    result = subprocess.run(args, check=True, capture_output=True, timeout=180)
    if not result.stdout or len(result.stdout) % FRAME_BYTES:
        raise ValueError(f"partial independent RGB decode: {video}")
    frames = [result.stdout[offset:offset + FRAME_BYTES]
              for offset in range(0, len(result.stdout), FRAME_BYTES)]
    timestamps = []
    for line in result.stderr.decode(errors="replace").splitlines():
        if " n:" not in line or "pts_time:" not in line:
            continue
        match = re.match(r"[-+0-9.eE]+", line.split("pts_time:", 1)[1].lstrip())
        if match and math.isfinite(float(match.group(0))):
            timestamps.append(float(match.group(0)))
    if len(timestamps) != len(frames):
        raise ValueError(f"independent showinfo PTS count mismatch: {video}")
    return frames, timestamps


def packets(video):
    value = json.loads(subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
        "-show_entries", "packet=pts,dts,flags", "-of", "json", str(video)],
        check=True, capture_output=True, timeout=60).stdout)
    rows = value["packets"]
    pts, dts = [int(row["pts"]) for row in rows], [int(row["dts"]) for row in rows]
    return {"packet_count": len(rows), "decode_order_pts": pts, "decode_order_dts": dts,
            "decode_order_pts_nonmonotonic": any(b < a for a, b in zip(pts, pts[1:])),
            "decode_order_dts_strictly_increasing": all(b > a for a, b in zip(dts, dts[1:])),
            "packets": rows}


def analyze(candidates):
    analyzed = []
    for frame in candidates:
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
    for index in range(1, len(candidates)):
        previous, current = analyzed[index - 1], analyzed[index]
        scores.append([
            sum(abs(a - b) for a, b in zip(previous[0], current[0])) / (255.0 * 256.0),
            sum(abs(a - b) for a, b in zip(previous[1], current[1])) / (2.0 * 256.0)])
    return scores


def runs(scores, threshold, component):
    result, index = [], 1
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
        raise ValueError("invalid independent natural selection budget")
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
              for run in runs(scores, 0.25, 1)]
    scenes.sort(key=lambda i: (-scores[i][1], -scores[i][0], i))
    for index in scenes:
        take(index)
    motion = [min(run, key=lambda i: (-scores[i][0], -scores[i][1], i))
              for run in runs(scores, 0.02, 0)]
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
        raise ValueError("independent selector failed to fill budget")
    return sorted(ordered)


def selection(candidates, sources, timestamps, frames):
    return {"candidate_indices": candidates, "source_frame_indices": sources,
            "source_timestamps_seconds": timestamps,
            "content_fnv1a64": [fnv1a64(frame) for frame in frames],
            "content_sha256": [sha256_bytes(frame) for frame in frames],
            "temporal2_pair_first_timestamps": [timestamps[min(2 * i, len(timestamps) - 1)]
                                                for i in range((len(timestamps) + 1) // 2)]}


def build_oracles(root, manifest):
    result = {}
    for cid, video in manifest["videos"].items():
        path = root / video["video"]
        if sha256_file(path) != video["video_sha256"]:
            raise ValueError(f"source video hash mismatch: {cid}")
        frames, pts = decode(path)
        count = video["frame_count"]
        if len(frames) != count or len(pts) != count or pts != sorted(pts) or len(set(pts)) != count:
            raise ValueError(f"source frame/presentation PTS mismatch: {cid}")
        rows = [json.loads(line) for line in (root / video["content_frames"]).read_text().splitlines()]
        for index, frame in enumerate(frames):
            if rows[index] != {"source_frame_index": index, "source_timestamp_seconds": pts[index],
                               "decoded_native_rgb_sha256": sha256_bytes(frame),
                               "decoded_native_rgb_fnv1a64": fnv1a64(frame)}:
                raise ValueError(f"independent full RGB/source-PTS divergence: {cid}/{index}")
        packet_rows = packets(path)
        stored_packets = json.loads((root / video["packets"]).read_bytes())
        def semantic_packets(value):
            return {"packet_count": value["packet_count"],
                    "decode_order_pts": value["decode_order_pts"],
                    "decode_order_dts": value["decode_order_dts"],
                    "decode_order_pts_nonmonotonic": value["decode_order_pts_nonmonotonic"],
                    "decode_order_dts_strictly_increasing": value[
                        "decode_order_dts_strictly_increasing"],
                    "packets": [{"pts": row["pts"], "dts": row["dts"],
                                 "flags": row["flags"].rstrip("_")} for row in value["packets"]]}
        if semantic_packets(packet_rows) != semantic_packets(stored_packets) \
                or not packet_rows["decode_order_pts_nonmonotonic"] \
                or not packet_rows["decode_order_dts_strictly_increasing"]:
            raise ValueError(f"independent B-frame/decode-order mismatch: {cid}")
        scores = analyze(frames)
        if scores != video["candidate_scores_motion_scene"]:
            raise ValueError(f"independent natural candidate scores diverge: {cid}")
        temporal = {}
        for budget in BUDGETS:
            selected = select_motion_run(scores, budget)
            if selected[0] != 0 or selected[-1] != count - 1:
                raise ValueError(f"temporal first/last anchor missing: {cid}/{budget}")
            temporal[str(budget)] = selection(
                selected, selected, [pts[i] for i in selected], [frames[i] for i in selected])
        if temporal != video["expected_temporal"]:
            raise ValueError(f"frozen natural temporal oracle mismatch: {cid}")
        uniform = {}
        for budget in BUDGETS:
            if budget == 4:
                sources = [(slot * (count - 1) + 1) // 3 for slot in range(4)]
                expression = "+".join(f"eq(n,{index})" for index in sources)
                selected, timestamps = decode(path, select=expression)
                if timestamps != [pts[index] for index in sources]:
                    raise ValueError(f"uniform-four source PTS mismatch: {cid}")
                uniform[str(budget)] = selection(list(range(4)), sources, timestamps, selected)
            else:
                selected, output_pts = decode(path, uniform8=True)
                if len(selected) != 8:
                    raise ValueError(f"historical uniform-eight count mismatch: {cid}")
                by_hash = {}
                for index, frame in enumerate(frames):
                    by_hash.setdefault(sha256_bytes(frame), []).append(index)
                candidates = []
                for frame in selected:
                    matches = by_hash.get(sha256_bytes(frame), [])
                    if not matches:
                        raise ValueError(f"uniform-eight RGB has no full-source match: {cid}")
                    candidates.append(matches)
                sources = [matches[0] for matches in candidates]
                uniform["8"] = selection(list(range(8)), sources,
                                         [pts[index] for index in sources], selected)
                uniform["8"]["source_frame_candidates"] = candidates
                uniform["8"]["source_pts_candidates"] = [[pts[i] for i in row] for row in candidates]
                uniform["8"]["source_frame_binding"] = (
                    "EXACT" if all(len(row) == 1 for row in candidates)
                    else "AMBIGUOUS_EXACT_REPEATED_SOURCE_RGB_ALL_CANDIDATES_RETAINED")
                uniform["8"]["output_filter_timestamps_seconds"] = output_pts
                duration = float(video["duration_seconds"])
                uniform["8"]["native_nominal_timestamps"] = [duration * i / 8 for i in range(8)]
        if uniform != video["expected_uniform"]:
            raise ValueError(f"independent natural uniform oracle mismatch: {cid}")
        result[cid] = {"frames": frames, "pts": pts, "scores": scores,
                       "temporal": temporal, "uniform": uniform}
    return result


def hit(answer, case):
    folded = answer.casefold()
    terms = case["expect_all"]
    if case["expect_any_group"]:
        return any(term.casefold() in folded for term in terms)
    return all(term.casefold() in folded for term in terms)


def native_equal(left, right):
    return all(left.get(field) == right.get(field) for field in NATIVE_PARITY_FIELDS)


def validate_native(record, case, oracle):
    native, execution = record["native"], record["native_execution"]
    budget, arm = record["budget"], record["arm"]
    request_id = record["response"]["request_id"]
    if native["request_id"] not in (request_id, request_id + ":vlm") \
            or execution["request_id"] != native["request_id"] \
            or native["engine_path"] != ("serial" if record["mode"] == "serial" else "batched") \
            or execution["prefix_reused"] != 0:
        raise ValueError("public/native request, engine, or cold-prefix identity mismatch")
    if native["policy"] != ("temporal" if arm == "temporal" else "uniform") \
            or native["max_frames"] != budget or native["selected_frame_count"] != budget:
        raise ValueError("native natural policy/budget mismatch")
    if arm == "temporal":
        positions = select_motion_run(oracle["scores"], budget)
        sources = positions
        expected = oracle["temporal"][str(budget)]
        if native["selected_candidate_indices"] != positions \
                or native["selected_source_frame_indices"] != sources \
                or native["selected_frame_timestamps"] != expected["source_timestamps_seconds"] \
                or not native["source_indices_exact"] or native["has_nominal_timestamps"] \
                or native["decoded_frame_count"] != len(oracle["frames"]):
            raise ValueError("independent temporal RGB/source/PTS/order mismatch")
        timestamps = expected["source_timestamps_seconds"]
    elif budget == 4:
        expected = oracle["uniform"]["4"]
        sources = expected["source_frame_indices"]
        timestamps = expected["source_timestamps_seconds"]
        if native["selected_candidate_indices"] != list(range(4)) \
                or native["selected_source_frame_indices"] != sources \
                or native["selected_frame_timestamps"] != timestamps \
                or not native["source_indices_exact"] or native["has_nominal_timestamps"] \
                or native["decoded_frame_count"] != 4:
            raise ValueError("explicit uniform-four source/PTS mismatch")
    else:
        expected = oracle["uniform"]["8"]
        sources = expected["source_frame_indices"]
        timestamps = expected["native_nominal_timestamps"]
        if native["selected_candidate_indices"] != list(range(8)) \
                or native["selected_source_frame_indices"] != list(range(8)) \
                or native["selected_frame_timestamps"] != timestamps \
                or native["source_indices_exact"] or not native["has_nominal_timestamps"] \
                or native["decoded_frame_count"] != 8:
            raise ValueError("historical uniform/default nominal contract changed")
    if native["selected_frame_content_fnv1a64"] != expected["content_fnv1a64"]:
        raise ValueError("native selected RGB differs from independent full decode")
    pairs = [timestamps[min(2 * i, budget - 1)] for i in range((budget + 1) // 2)]
    if native["temporal2_pair_first_timestamps"] != pairs \
            or native["grid_t"] != (budget + 1) // 2 \
            or native["premerge_patch_rows"] != native["grid_t"] * native["grid_h"] * native["grid_w"] \
            or native["video_token_count"] != native["grid_t"] * native["merged_tokens_per_slice"]:
        raise ValueError("temporal-2/grid/token identity mismatch")
    expected_next = native["prompt_token_count"] - native["video_token_count"] + native[
        "grid_t"] * max(native["grid_h"] // 2, native["grid_w"] // 2)
    if native["mrope_next"] != expected_next or not re.fullmatch(
            r"[0-9a-f]{16}", native["mrope_fnv1a64"]) \
            or native["max_candidate_frames"] != 64 or native["motion_threshold"] != 0.02 \
            or native["scene_threshold"] != 0.25:
        raise ValueError("native mRoPE/threshold diagnostic drift")
    selected = record["source_selection"]
    if selected["exact_source_frame_indices"] != sources \
            or selected["exact_source_timestamps_seconds"] != expected["source_timestamps_seconds"] \
            or selected["temporal2_pair_first_timestamps"] != expected[
                "temporal2_pair_first_timestamps"]:
        raise ValueError("recorded independent source-selection summary mismatch")
    for field in ("vision_ms", "prefill_ms", "decode_ms", "elapsed_ms"):
        name = "native_elapsed" if field == "elapsed_ms" else field.removesuffix("_ms")
        if record["latency_ms"][name] != execution[field]:
            raise ValueError("native latency projection mismatch")


def validate_record(record, case, oracle, mode_dir):
    if record["status"] != "COMPLETED" or not record["native_evidence_valid"] \
            or record["response"]["http_status"] != 200 \
            or record["response"]["finish_reason"] != "stop":
        raise ValueError("failed or invalid request in accepted natural population")
    if record["raw_answer"] != record["response"]["answer"] \
            or sha256_bytes(record["raw_answer"].encode()) != record["response"]["answer_sha256"] \
            or record["expected_answer_hit"] != hit(record["raw_answer"], case):
        raise ValueError("raw answer/hash/frozen-score mismatch")
    content = record["request"]["messages"][0]["content"]
    if [part["type"] for part in content] != ["video_url", "text"] \
            or content[0]["video_url"]["url"]["data_url_sha256"] != case["video_sha256"] \
            or content[1]["text"] != case["question"]:
        raise ValueError("request source/question order binding mismatch")
    selection_value = content[0]["video_url"].get("frame_selection")
    expected_selection = None if record["arm"] == "default" else (
        {"policy": "uniform", "max_frames": record["budget"]} if record["arm"] == "uniform" else
        {"policy": "temporal", "motion_threshold": 0.02, "scene_threshold": 0.25,
         "max_frames": record["budget"]})
    if selection_value != expected_selection:
        raise ValueError("recorded natural frame-selection request mismatch")
    for field, expected in (("temperature", 0.0), ("top_p", 1.0), ("seed", 510072),
                            ("max_tokens", 64), ("stream", True)):
        if record["request"][field] != expected:
            raise ValueError(f"request sampling drift: {field}")
    canonical = json.loads(json.dumps(record["request"]))
    digest = canonical.pop("canonical_request_sha256")
    if sha256_bytes(json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False).encode("utf-8")) != digest:
        raise ValueError("canonical natural request digest mismatch")
    validate_native(record, case, oracle)
    relative = Path("records") / record["mode"] / f"budget-{record['budget']}" \
        / record["phase"] / f"repetition-{record['repetition']}" / record["arm"] \
        / f"{record['clip']}.json"
    if json.loads((mode_dir / relative).read_bytes()) != record:
        raise ValueError("raw per-request record differs from report")
    frames = json.loads((mode_dir / relative.with_suffix(".sse.json")).read_bytes())
    if not isinstance(frames, list) or not frames:
        raise ValueError("raw natural SSE frames missing")
    answer = "".join(choice.get("delta", {}).get("content", "")
                     for frame in frames for choice in frame.get("choices", [])
                     if isinstance(choice.get("delta", {}).get("content"), str))
    if answer != record["raw_answer"]:
        raise ValueError("SSE deltas do not reconstruct retained raw answer")


def parity(records):
    checks = []
    by_default = {}
    for row in records:
        key = (row["mode"], row["phase"], row["repetition"], row["clip"], row["budget"])
        by_default.setdefault(key, {})[row["arm"]] = row
    for key, values in sorted(by_default.items()):
        if "default" not in values:
            continue
        a, b = values["default"], values["uniform"]
        def without(row):
            value = json.loads(json.dumps(row["request"]))
            value["messages"][0]["content"][0]["video_url"].pop("frame_selection", None)
            value.pop("canonical_request_sha256", None)
            return value
        request = without(a) == without(b)
        answer = a["raw_answer"] == b["raw_answer"]
        native = native_equal(a["native"], b["native"])
        checks.append({"scope": "default_uniform", "mode": key[0], "phase": key[1],
                       "repetition": key[2], "clip": key[3], "budget": key[4],
                       "passed": request and answer and native, "request_projection_equal": request,
                       "answer_byte_identical": answer,
                       "native_selection_temporal2_mrope_identical": native})
    cross = {}
    for row in records:
        key = (row["phase"], row["repetition"], row["clip"], row["arm"], row["budget"])
        cross.setdefault(key, {})[row["mode"]] = row
    for key, values in sorted(cross.items()):
        a, b = values["serial"], values["batch4"]
        request, answer = a["request"] == b["request"], a["raw_answer"] == b["raw_answer"]
        native = native_equal(a["native"], b["native"])
        checks.append({"scope": "serial_batch4", "phase": key[0], "repetition": key[1],
                       "clip": key[2], "arm": key[3], "budget": key[4],
                       "request_byte_identical": request, "answer_byte_identical": answer,
                       "native_selection_temporal2_mrope_identical": native,
                       "passed": request and answer and native})
    repeated = {}
    for row in records:
        key = (row["mode"], row["clip"], row["arm"], row["budget"])
        repeated.setdefault(key, []).append(row)
    for key, rows in sorted(repeated.items()):
        request = len({json.dumps(row["request"], sort_keys=True, ensure_ascii=False)
                       for row in rows}) == 1
        answer = len({row["raw_answer"] for row in rows}) == 1
        native = all(native_equal(rows[0]["native"], row["native"]) for row in rows[1:])
        checks.append({"scope": "within_mode_repetitions", "mode": key[0], "clip": key[1],
                       "arm": key[2], "budget": key[3], "request_byte_identical": request,
                       "answer_byte_identical": answer,
                       "native_selection_temporal2_mrope_identical": native,
                       "passed": request and answer and native})
    if len(checks) != 304 or any(not row["passed"] for row in checks):
        raise ValueError("independent 304-check natural parity gate failed")
    return checks


def wilson(hits, count=48):
    z = 1.959963984540054
    rate = hits / count
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def sign_test(improved, worsened):
    count = improved + worsened
    if not count:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(count, k)
               for k in range(min(improved, worsened) + 1)) / (2 ** count))


def measured(records, arm=None, budget=None):
    return [row for row in records if row["phase"] == "measured"
            and (arm is None or row["arm"] == arm) and (budget is None or row["budget"] == budget)]


def paired_effect(records, baseline, candidate):
    def grouped(arm, budget):
        result = {}
        for row in measured(records, arm, budget):
            result.setdefault((row["clip"], row["mode"]), []).append(row)
        if len(result) != 16 or any(len(rows) != 3 for rows in result.values()):
            raise ValueError("independent natural paired population mismatch")
        return result
    left, right = grouped(*baseline), grouped(*candidate)
    pairs, transitions = [], []
    for key in sorted(left):
        lrows = sorted(left[key], key=lambda row: row["repetition"])
        rrows = sorted(right[key], key=lambda row: row["repetition"])
        lsum = sum(row["expected_answer_hit"] is True for row in lrows)
        rsum = sum(row["expected_answer_hit"] is True for row in rrows)
        outcome = "improved" if rsum > lsum else "worsened" if rsum < lsum else "unchanged"
        pairs.append({"case_id": key[0], "mode": key[1], "baseline_hits": lsum,
                      "candidate_hits": rsum, "outcome": outcome,
                      "baseline_answers": [row["raw_answer"] for row in lrows],
                      "candidate_answers": [row["raw_answer"] for row in rrows]})
        for lrow, rrow in zip(lrows, rrows):
            transition = ("hit_to_hit" if lrow["expected_answer_hit"] and rrow["expected_answer_hit"]
                          else "hit_to_miss" if lrow["expected_answer_hit"]
                          else "miss_to_hit" if rrow["expected_answer_hit"] else "miss_to_miss")
            transitions.append({"case_id": key[0], "mode": key[1], "repetition": lrow["repetition"],
                                "transition": transition, "baseline_answer": lrow["raw_answer"],
                                "candidate_answer": rrow["raw_answer"],
                                "baseline_request_id": lrow["response"]["request_id"],
                                "candidate_request_id": rrow["response"]["request_id"]})
    improved = sum(row["outcome"] == "improved" for row in pairs)
    worsened = sum(row["outcome"] == "worsened" for row in pairs)
    counts = {name: sum(row["transition"] == name for row in transitions)
              for name in ("hit_to_hit", "hit_to_miss", "miss_to_hit", "miss_to_miss")}
    return {"baseline": f"{baseline[0]}-{baseline[1]}", "candidate": f"{candidate[0]}-{candidate[1]}",
            "paired_case_mode_count": 16, "improved_pairs": improved, "worsened_pairs": worsened,
            "unchanged_pairs": 16 - improved - worsened,
            "exact_two_sided_sign_test_p": sign_test(improved, worsened),
            "case_mode_pairs": pairs, "repetition_transition_counts": counts,
            "repetition_transitions": transitions}


def quality_analysis(records, manifest):
    quality = {}
    for arm, budget in ARM_BUDGETS:
        rows = measured(records, arm, budget)
        if len(rows) != 48:
            raise ValueError("independent natural quality population mismatch")
        hits = sum(row["expected_answer_hit"] is True for row in rows)
        quality[f"{arm}-{budget}"] = {"hits": hits, "total": 48, "misses": 48 - hits,
                                     "accuracy": hits / 48, "wilson_95": wilson(hits)}
    comparisons = {}
    for name, baseline, candidate in (
            ("primary_temporal8_vs_uniform8", ("uniform", 8), ("temporal", 8)),
            ("diagnostic_temporal4_vs_uniform4", ("uniform", 4), ("temporal", 4)),
            ("temporal8_vs_default8", ("default", 8), ("temporal", 8))):
        comparisons[name] = paired_effect(records, baseline, candidate)
    per_case = {}
    for case in sorted(manifest["cases"], key=lambda item: item["id"]):
        arms = {}
        for arm, budget in ARM_BUDGETS:
            rows = sorted((row for row in measured(records, arm, budget)
                           if row["clip"] == case["id"]), key=lambda row: (row["mode"], row["repetition"]))
            if len(rows) != 6:
                raise ValueError("independent per-case natural population mismatch")
            arms[f"{arm}-{budget}"] = {
                "hits": sum(row["expected_answer_hit"] is True for row in rows), "total": 6,
                "answer_anchor_present": case["anchor_audit"][f"{arm}-{budget}"]["answer_anchor_present"],
                "answers": [{"mode": row["mode"], "repetition": row["repetition"],
                             "request_id": row["response"]["request_id"],
                             "raw_answer": row["raw_answer"], "hit": row["expected_answer_hit"]}
                            for row in rows]}
        per_case[case["id"]] = {"arms": arms}
    misses = []
    for row in sorted(measured(records), key=lambda item: (
            item["clip"], item["mode"], item["arm"], item["budget"], item["repetition"])):
        if row["expected_answer_hit"] is False:
            case = next(case for case in manifest["cases"] if case["id"] == row["clip"])
            misses.append({"case_id": row["clip"], "mode": row["mode"], "arm": row["arm"],
                           "budget": row["budget"], "repetition": row["repetition"],
                           "request_id": row["response"]["request_id"], "raw_answer": row["raw_answer"],
                           "answer_anchor_present": case["anchor_audit"][
                               f"{row['arm']}-{row['budget']}"]["answer_anchor_present"]})
    regressions = []
    for name, comparison in comparisons.items():
        regressions.extend({"comparison": name, **row} for row in comparison["repetition_transitions"]
                           if row["transition"] == "hit_to_miss")
    return {"quality": quality, "comparisons": comparisons, "per_case": per_case,
            "every_miss": misses, "quality_regressions_retained": regressions,
            "quality_regression_count": len(regressions),
            "runtime_change_in_response_to_answers": "FORBIDDEN_NONE_MADE"}


def latency_summary(records):
    result = {}
    for mode in MODES:
        for arm, budget in ARM_BUDGETS:
            rows = [row for row in measured(records, arm, budget) if row["mode"] == mode]
            if len(rows) != 24:
                raise ValueError("independent natural latency population mismatch")
            metrics = {}
            for metric in ("first_event", "ttft", "total", "vision", "prefill", "decode",
                           "native_elapsed"):
                values = sorted(row["latency_ms"][metric] for row in rows)
                metrics[metric] = {"count": 24, "median_ms": statistics.median(values),
                                   "p95_ms": values[math.ceil(0.95 * len(values)) - 1],
                                   "min_ms": values[0], "max_ms": values[-1]}
            result[f"{mode}/{arm}-{budget}"] = metrics
    return result


def latency_outliers(records):
    result = []
    for mode in MODES:
        for arm, budget in ARM_BUDGETS:
            rows = [row for row in measured(records, arm, budget) if row["mode"] == mode]
            for metric in ("total", "native_elapsed"):
                values = sorted(row["latency_ms"][metric] for row in rows)
                q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
                fence = q3 + 1.5 * (q3 - q1)
                for row in rows:
                    value = row["latency_ms"][metric]
                    if value > fence:
                        result.append({"case_id": row["clip"], "mode": mode, "arm": arm,
                                       "budget": budget, "repetition": row["repetition"],
                                       "request_id": row["response"]["request_id"], "metric": metric,
                                       "value_ms": value, "q1_ms": q1, "q3_ms": q3,
                                       "upper_fence_ms": fence, "retained_in_aggregates": True})
    return result


def verify(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if root in output.parents:
        raise ValueError("verification output must remain outside sealed evidence")
    file_count = verify_sums(root)
    report = json.loads((root / "report.json").read_bytes())
    if report.get("schema_version") != 1 \
            or report.get("experiment") != "qwen38_temporal_natural_quality_v1" \
            or report.get("fixture_revision") != "temporal-natural-r2":
        raise ValueError("unsupported natural evidence")
    fixture_root = root / "input" / "fixture"
    frozen_count = verify_sums(fixture_root, fixture=True)
    manifest = json.loads((fixture_root / "manifest.json").read_bytes())
    freeze = json.loads((root / "input" / "freeze.json").read_bytes())
    if len(manifest["cases"]) != 8 or len(manifest["videos"]) != 8 \
            or manifest["natural_video_claim"] is not True \
            or manifest["source"]["original_git_blob_sha1"] != "fa078dc543b273922dfc8c07e74f63306778a3ed":
        raise ValueError("natural source/population boundary mismatch")
    if freeze["manifest_sha256"] != sha256_file(fixture_root / "manifest.json") \
            or freeze["preregistration_sha256"] != sha256_file(fixture_root / "preregistration.json") \
            or freeze["sha256sums_sha256"] != sha256_file(fixture_root / "SHA256SUMS") \
            or freeze != report["freeze_receipt"] \
            or not freeze["frozen_before_any_model_output_for_this_revision"] \
            or freeze["model_outputs_seen_for_this_revision_at_freeze"] \
            or not freeze["prior_excluded_natural_outputs_seen_at_freeze"] \
            or freeze["prior_excluded_attempt"] != "format-duration-r1":
        raise ValueError("frozen natural receipt mismatch")
    for case in manifest["cases"]:
        folded = case["question"].casefold()
        if any(term.casefold() in folded for term in case["expect_all"]):
            raise ValueError("prompt contains a frozen answer term")
    control = root / "input" / "task66-regression-control"
    if verify_sums(control, fixture=True) != 138:
        raise ValueError("Task66 regression-control file count changed")
    for name, digest in TASK66_HASHES.items():
        if sha256_file(control / name) != digest:
            raise ValueError("Task66 fixture hash changed")
    if report["historical_controls"] != {
            "task66_fixture_sha256sums_sha256": TASK66_HASHES["SHA256SUMS"],
            "primary_temporal8_hits": 18, "primary_uniform8_hits": 18,
            "primary_sign_test_p": 1.0, "included_in_natural_denominator": False,
            "scores_or_fixture_changed": False}:
        raise ValueError("Task66 regression-control scores changed")
    analysis_hashes = report.get("analysis_source_sha256", {})
    if set(analysis_hashes) != ANALYSIS_SOURCES:
        raise ValueError("sealed natural analysis source set incomplete")
    for name, digest in analysis_hashes.items():
        if not HEX256.fullmatch(digest) or sha256_file(root / "input" / "analysis" / name) != digest:
            raise ValueError("sealed natural analysis source hash mismatch")
    if sha256_file(Path(__file__).resolve()) != analysis_hashes["tests/temporal_natural_verify.py"]:
        raise ValueError("running independent verifier differs from sealed verifier")
    oracles = build_oracles(fixture_root, manifest)
    cases = {case["id"]: case for case in manifest["cases"]}
    records = report["records"]
    if len(records) != 320 or not report["gate_pass"] or report["gate_reasons"]:
        raise ValueError("producer natural technical gate did not pass 320 requests")
    expected_gpu = os.environ.get("UMM_NATURAL_PHYSICAL_GPU", "1")
    if expected_gpu not in ("0", "1"):
        raise ValueError("UMM_NATURAL_PHYSICAL_GPU must be 0 or 1")
    for mode in MODES:
        mode_report = json.loads((root / mode / "mode-result.json").read_bytes())
        if mode_report["provenance"]["physical_gpu"] != int(expected_gpu) \
                or mode_report["provenance"]["batch_size"] != (1 if mode == "serial" else 4) \
                or mode_report["provenance"]["qwen38_prefix_cache"] != "disabled" \
                or mode_report["stale_errors"] \
                or any(not row["passed"] for row in mode_report["exclusive"]):
            raise ValueError("natural mode provenance/GPU/staleness gate failed")
        if not report["post_stop_cleanup"].get(mode) \
                or not report["post_stop_cleanup"][mode].get("passed"):
            raise ValueError("natural post-stop cleanup proof failed")
    if any(not row["passed"] for row in report["cleanup_checks"] + report["exclusive"]):
        raise ValueError("natural cleanup/in_flight/residual-memory gate failed")
    for row in records:
        validate_record(row, cases[row["clip"]], oracles[row["video_id"]], root / row["mode"])
    independent_parity = parity(records)
    if independent_parity != report["parity"]:
        raise ValueError("reported natural parity differs from independent reconstruction")
    analysis = quality_analysis(records, manifest)
    if analysis != report["analysis"]:
        raise ValueError("reported quality/paired/raw-answer/miss analysis differs")
    if latency_summary(records) != report["latency_summary"]:
        raise ValueError("independent natural latency summary mismatch")
    outliers = latency_outliers(records)
    if outliers != report["latency_outliers"]:
        raise ValueError("natural latency outlier reconstruction mismatch")
    for excluded in report["excluded_attempts"]:
        excluded_root = root / "excluded-attempts" / excluded["name"]
        if not excluded_root.is_dir() or excluded.get("included_in_accepted_results") is not False:
            raise ValueError("excluded natural attempt is missing or admitted")
        verify_sums(excluded_root, legacy_basename_exclude=True)
    if report["claims"]["natural_video_fixture_accuracy"] != "MEASURED_PREREGISTERED_NATURAL_FIXTURE" \
            or report["claims"]["perceptual_quality"] != "NOT_RUN" \
            or report["claims"]["generalized_quality_claim"] is not False \
            or any(row.get("included_in_accepted_results") for row in report["excluded_attempts"]):
        raise ValueError("natural evidence escaped its scope/exclusion boundary")
    result = {
        "result": "PASS", "independent_implementation": "stdlib_only_no_producer_scoring_import",
        "verifier_source_sha256": sha256_file(Path(__file__).resolve()),
        "evidence_files_hashed": file_count, "frozen_fixture_files_hashed": frozen_count,
        "task66_control_files_hashed": 138,
        "full_source_frames_independently_decoded": sum(
            len(value["frames"]) for value in oracles.values()),
        "requests_independently_validated": 320, "parity_checks": 304,
        "quality": analysis["quality"],
        "primary_comparison": {key: analysis["comparisons"]["primary_temporal8_vs_uniform8"][key]
                               for key in ("improved_pairs", "worsened_pairs",
                                           "exact_two_sided_sign_test_p")},
        "misses_independently_retained": len(analysis["every_miss"]),
        "latency_outliers_independently_retained": len(outliers),
        "quality_regressions_retained": analysis["quality_regression_count"],
        "sha256sums_sha256": sha256_file(root / "SHA256SUMS"),
        "report_sha256": sha256_file(root / "report.json"), "claims": report["claims"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(json.dumps(result, ensure_ascii=False, indent=2).encode() + b"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        return verify(args.evidence, args.output)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"INDEPENDENT NATURAL VERIFICATION FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
