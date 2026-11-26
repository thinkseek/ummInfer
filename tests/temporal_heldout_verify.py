#!/usr/bin/env python3
"""Fresh-process stdlib-only independent verifier for Task66 held-out evidence.

No producer, fixture generator, transport, or scoring module is imported. Every
source is decoded again; selected RGB, source indices/PTS/order, temporal-2,
answers, aggregates, parity, misses, outliers, decisions and checksums are rebuilt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

HEX256 = re.compile(r"[0-9a-f]{64}\Z")
MODES = ("serial", "batch4")
BUDGETS = (4, 8, 16)
ARM_BUDGETS = (("default", 8),) + tuple(
    (arm, budget) for arm in ("uniform", "temporal") for budget in BUDGETS)
ANALYSIS_SOURCES = {
    "python/umminfer_temporal_heldout_fixture.py", "tests/temporal_heldout_live.py",
    "tests/temporal_heldout_test.py", "tests/temporal_heldout_verify.py",
}
NATIVE_PARITY_FIELDS = (
    "policy", "decoded_frame_count", "selected_frame_count", "selected_candidate_indices",
    "selected_source_frame_indices", "source_indices_exact", "selected_frame_timestamps",
    "selected_frame_content_fnv1a64", "has_nominal_timestamps", "temporal2_pair_first_timestamps",
    "max_frames", "max_candidate_frames", "motion_threshold", "scene_threshold", "grid_t",
    "grid_h", "grid_w", "premerge_patch_rows", "merged_tokens_per_slice", "video_token_count",
    "prompt_token_count", "media_hash", "mrope_next", "mrope_fnv1a64",
)
HISTORICAL_TASK61 = {"motion-first-en", "scene-cut-zh", "near-static-ja", "bframe-first-es"}


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


def verify_sums(root, *, fixture=False):
    root = Path(root)
    sums = {}
    for line in (root / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        if not HEX256.fullmatch(digest) or relative.startswith("/") \
                or ".." in Path(relative).parts or relative in sums:
            raise ValueError("malformed or duplicate checksum entry")
        sums[relative] = digest
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name != "SHA256SUMS"}
    if set(sums) != actual:
        raise ValueError("checksum file-set mismatch")
    for relative, digest in sums.items():
        if sha256_file(root / relative) != digest:
            raise ValueError(f"checksum mismatch: {relative}")
    if not fixture:
        artifact = json.loads((root / "artifact-manifest.json").read_bytes())
        listed = {row["path"]: row for row in artifact["files"]}
        if set(listed) != actual - {"artifact-manifest.json"}:
            raise ValueError("artifact manifest does not cover the complete evidence set")
        for relative, row in listed.items():
            path = root / relative
            if row["bytes"] != path.stat().st_size or row["sha256"] != sha256_file(path):
                raise ValueError(f"artifact manifest mismatch: {relative}")
    return len(sums)


def decode(video, *, historical_uniform=False):
    if historical_uniform:
        args = ["ffmpeg", "-v", "error", "-i", str(video), "-an", "-sn", "-dn",
                "-vf", "fps=0.2500,scale=224:160:flags=bicubic", "-frames:v", "8",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    else:
        args = ["ffmpeg", "-v", "info", "-i", str(video), "-an", "-sn", "-dn",
                "-vf", "scale=224:160:flags=bicubic,showinfo", "-vsync", "0",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    result = subprocess.run(args, check=True, capture_output=True, timeout=180)
    size = 224 * 160 * 3
    if not result.stdout or len(result.stdout) % size:
        raise ValueError(f"partial RGB decode: {video}")
    frames = [result.stdout[offset:offset + size] for offset in range(0, len(result.stdout), size)]
    timestamps = []
    if not historical_uniform:
        for line in result.stderr.decode(errors="replace").splitlines():
            if " n:" not in line or "pts_time:" not in line:
                continue
            match = re.match(r"[-+0-9.eE]+", line.split("pts_time:", 1)[1].lstrip())
            if match:
                timestamps.append(float(match.group(0)))
        if len(timestamps) != len(frames):
            raise ValueError(f"showinfo PTS count mismatch: {video}")
    return frames, timestamps


def analyze(candidates):
    analyzed = []
    for frame in candidates:
        luma, histogram = [], [0] * 16
        for ay in range(16):
            y0, y1 = ay * 160 // 16, (ay + 1) * 160 // 16
            for ax in range(16):
                x0, x1 = ax * 224 // 16, (ax + 1) * 224 // 16
                total = count = 0
                for y in range(y0, y1):
                    for x in range(x0, x1):
                        offset = (y * 224 + x) * 3
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
            sum(abs(a - b) for a, b in zip(previous[1], current[1])) / (2.0 * 256.0),
        ])
    return scores


def runs(scores, threshold, component):
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
        raise ValueError("invalid independent selection budget")
    target = min(len(scores), budget)
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
        raise ValueError("independent selector failed to fill its budget")
    return sorted(ordered)


def build_oracles(fixture_root, manifest):
    result = {}
    for video_id, video in manifest["videos"].items():
        path = fixture_root / video["video"]
        if sha256_file(path) != video["video_sha256"]:
            raise ValueError(f"source video hash mismatch: {video_id}")
        frames, pts = decode(path)
        if len(frames) != 449 or len(pts) != 449:
            raise ValueError(f"source frame count mismatch: {video_id}")
        rows = [json.loads(line) for line in
                (fixture_root / video["content_frames"]).read_text().splitlines()]
        if len(rows) != 449:
            raise ValueError("source-content population mismatch")
        for index, frame in enumerate(frames):
            if rows[index]["source_frame_index"] != index \
                    or sha256_bytes(frame) != rows[index]["decoded_native_rgb_sha256"] \
                    or fnv1a64(frame) != rows[index]["decoded_native_rgb_fnv1a64"] \
                    or pts[index] != rows[index]["source_timestamp_seconds"]:
                raise ValueError(f"independent full decode diverges: {video_id}/{index}")
        candidates = [frames[index] for index in range(0, 449, 8)]
        scores = analyze(candidates)
        if scores != video["candidate_scores_motion_scene"]:
            raise ValueError(f"independent candidate scores diverge: {video_id}")
        for budget in BUDGETS:
            positions = select_motion_run(scores, budget)
            expected = video["expected_temporal"][str(budget)]
            sources = [position * 8 for position in positions]
            timestamps = [pts[source] for source in sources]
            fingerprints = [fnv1a64(frames[source]) for source in sources]
            hashes = [sha256_bytes(frames[source]) for source in sources]
            pairs = [timestamps[min(2 * index, budget - 1)] for index in range((budget + 1) // 2)]
            if expected != {"candidate_indices": positions, "source_frame_indices": sources,
                            "source_timestamps_seconds": timestamps, "content_fnv1a64": fingerprints,
                            "content_sha256": hashes, "temporal2_pair_first_timestamps": pairs}:
                raise ValueError(f"frozen temporal oracle mismatch: {video_id}/{budget}")
        historical, _ = decode(path, historical_uniform=True)
        if len(historical) != 8:
            raise ValueError("historical uniform decode count mismatch")
        by_hash = {}
        for index, frame in enumerate(frames):
            by_hash.setdefault(sha256_bytes(frame), []).append(index)
        uniform8_sources = []
        for frame in historical:
            matches = by_hash.get(sha256_bytes(frame), [])
            if len(matches) != 1:
                raise ValueError(f"ambiguous historical source binding: {video_id}")
            uniform8_sources.append(matches[0])
        if uniform8_sources != [3, 11, 19, 27, 35, 43, 51, 59]:
            raise ValueError("historical source positions changed")
        result[video_id] = {"frames": frames, "pts": pts, "scores": scores,
                            "uniform8_sources": uniform8_sources, "uniform8_frames": historical}
    return result


def hit(answer, case):
    terms = case.get("expect_all", [])
    if not terms:
        return None
    folded = answer.casefold()
    if case.get("expect_any_group"):
        return any(term.casefold() in folded for term in terms)
    return all(term.casefold() in folded for term in terms)


def native_equal(left, right):
    return all(left.get(field) == right.get(field) for field in NATIVE_PARITY_FIELDS)


def expected_record(record, case, oracle):
    native, execution = record["native"], record["native_execution"]
    budget, arm = record["budget"], record["arm"]
    request_id = record["response"]["request_id"]
    if native["request_id"] not in (request_id, request_id + ":vlm") \
            or execution["request_id"] != native["request_id"]:
        raise ValueError("public/native request identity mismatch")
    if native["policy"] != ("temporal" if arm == "temporal" else "uniform") \
            or native["engine_path"] != ("serial" if record["mode"] == "serial" else "batched"):
        raise ValueError("native policy/engine mismatch")
    if execution["prefix_reused"] != 0 or native["max_frames"] != budget \
            or native["selected_frame_count"] != budget:
        raise ValueError("cold-prefix or selected budget mismatch")
    if arm == "temporal":
        positions = select_motion_run(oracle["scores"], budget)
        sources = [position * 8 for position in positions]
        frames = [oracle["frames"][source] for source in sources]
        timestamps = [oracle["pts"][source] for source in sources]
        if native["selected_candidate_indices"] != positions \
                or native["selected_source_frame_indices"] != sources \
                or native["selected_frame_timestamps"] != timestamps \
                or not native["source_indices_exact"] or native["has_nominal_timestamps"] \
                or native["decoded_frame_count"] != 57:
            raise ValueError("independent temporal source/PTS/order mismatch")
    elif budget == 8:
        sources = oracle["uniform8_sources"]
        frames = oracle["uniform8_frames"]
        timestamps = [case["duration_seconds"] * index / 8 for index in range(8)]
        if native["selected_candidate_indices"] != list(range(8)) \
                or native["selected_source_frame_indices"] != list(range(8)) \
                or native["selected_frame_timestamps"] != timestamps \
                or native["source_indices_exact"] or not native["has_nominal_timestamps"] \
                or native["decoded_frame_count"] != 8:
            raise ValueError("historical uniform/default nominal contract changed")
    else:
        sources = [(slot * 448 + (budget - 1) // 2) // (budget - 1) for slot in range(budget)]
        frames = [oracle["frames"][source] for source in sources]
        timestamps = [oracle["pts"][source] for source in sources]
        if native["selected_candidate_indices"] != list(range(budget)) \
                or native["selected_source_frame_indices"] != sources \
                or native["selected_frame_timestamps"] != timestamps \
                or not native["source_indices_exact"] or native["has_nominal_timestamps"] \
                or native["decoded_frame_count"] != budget:
            raise ValueError("explicit uniform source/PTS/order mismatch")
    if native["selected_frame_content_fnv1a64"] != [fnv1a64(frame) for frame in frames]:
        raise ValueError("native selected RGB differs from independent full decode")
    pairs = [timestamps[min(2 * index, budget - 1)] for index in range((budget + 1) // 2)]
    if native["temporal2_pair_first_timestamps"] != pairs \
            or native["grid_t"] != (budget + 1) // 2 \
            or native["premerge_patch_rows"] != native["grid_t"] * native["grid_h"] * native["grid_w"] \
            or native["video_token_count"] != native["grid_t"] * native["merged_tokens_per_slice"]:
        raise ValueError("temporal-2/grid/token identity mismatch")
    if native["max_candidate_frames"] != 64 or native["motion_threshold"] != 0.02 \
            or native["scene_threshold"] != 0.25 \
            or not re.fullmatch(r"[0-9a-f]{16}", native["mrope_fnv1a64"]):
        raise ValueError("native thresholds or mRoPE diagnostic drift")
    selection = record["source_selection"]
    source_pairs = [oracle["pts"][sources[min(2 * index, budget - 1)]]
                    for index in range((budget + 1) // 2)]
    if selection["exact_source_frame_indices"] != sources \
            or selection["exact_source_timestamps_seconds"] != [oracle["pts"][source] for source in sources] \
            or selection["temporal2_pair_first_timestamps"] != source_pairs:
        raise ValueError("recorded independent source-selection summary mismatch")
    for field in ("vision_ms", "prefill_ms", "decode_ms", "elapsed_ms"):
        latency_name = "native_elapsed" if field == "elapsed_ms" else field.removesuffix("_ms")
        if record["latency_ms"][latency_name] != execution[field]:
            raise ValueError("native latency projection mismatch")


def validate_record(record, case, oracle, mode_dir):
    if record["status"] != "COMPLETED" or not record["native_evidence_valid"] \
            or record["response"]["http_status"] != 200 \
            or record["response"]["finish_reason"] != "stop":
        raise ValueError("failed or invalid request in the technical population")
    if record["raw_answer"] != record["response"]["answer"] \
            or sha256_bytes(record["raw_answer"].encode()) != record["response"]["answer_sha256"] \
            or record["expected_answer_hit"] != hit(record["raw_answer"], case):
        raise ValueError("raw answer/hash/frozen-score mismatch")
    content = record["request"]["messages"][0]["content"]
    if [part["type"] for part in content] != ["video_url", "text"] \
            or content[0]["video_url"]["url"]["data_url_sha256"] != case["video_sha256"] \
            or content[1]["text"] != case["question"]:
        raise ValueError("request source/question order binding mismatch")
    selection = content[0]["video_url"].get("frame_selection")
    expected_selection = None if record["arm"] == "default" else (
        {"policy": "uniform", "max_frames": record["budget"]} if record["arm"] == "uniform" else
        {"policy": "temporal", "motion_threshold": 0.02, "scene_threshold": 0.25,
         "max_frames": record["budget"]})
    if selection != expected_selection:
        raise ValueError("recorded frame-selection request mismatch")
    for field, expected in (("temperature", 0.0), ("top_p", 1.0), ("seed", 510066),
                            ("max_tokens", 64), ("stream", True)):
        if record["request"][field] != expected:
            raise ValueError(f"request sampling drift: {field}")
    canonical = json.loads(json.dumps(record["request"]))
    digest = canonical.pop("canonical_request_sha256")
    if sha256_bytes(json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False).encode("utf-8")) != digest:
        raise ValueError("canonical request digest mismatch")
    expected_record(record, case, oracle)
    relative = Path("records") / record["mode"] / f"budget-{record['budget']}" \
        / record["phase"] / f"repetition-{record['repetition']}" / record["arm"] \
        / f"{record['clip']}.json"
    if json.loads((mode_dir / relative).read_bytes()) != record:
        raise ValueError("raw per-request record differs from report")
    frames = json.loads((mode_dir / relative.with_suffix(".sse.json")).read_bytes())
    if not isinstance(frames, list) or not frames:
        raise ValueError("raw SSE frames missing")
    answer = "".join(choice.get("delta", {}).get("content", "")
                     for frame in frames for choice in frame.get("choices", [])
                     if isinstance(choice.get("delta", {}).get("content"), str))
    if answer != record["raw_answer"]:
        raise ValueError("SSE deltas do not reconstruct the retained raw answer")


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
    if len(checks) != 500 or any(not row["passed"] for row in checks):
        raise ValueError("independent 500-check parity gate failed")
    return checks


def wilson(hits, count=60):
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
        if len(result) != 20 or any(len(rows) != 3 for rows in result.values()):
            raise ValueError("independent paired population mismatch")
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
            "paired_case_mode_count": 20, "improved_pairs": improved, "worsened_pairs": worsened,
            "unchanged_pairs": 20 - improved - worsened,
            "exact_two_sided_sign_test_p": sign_test(improved, worsened),
            "case_mode_pairs": pairs, "repetition_transition_counts": counts,
            "repetition_transitions": transitions}


def anchor_present(manifest, case_id, arm, budget):
    case = next(case for case in manifest["cases"] if case["id"] == case_id)
    return case["anchor_audit"][f"{arm}-{budget}"]["answer_anchor_present"]


def quality_analysis(records, manifest):
    quality = {}
    for arm, budget in ARM_BUDGETS:
        rows = measured(records, arm, budget)
        if len(rows) != 60:
            raise ValueError("quality population is not 60 answers")
        hits = sum(row["expected_answer_hit"] is True for row in rows)
        quality[f"{arm}-{budget}"] = {"hits": hits, "total": 60, "misses": 60 - hits,
                                     "accuracy": hits / 60, "wilson_95": wilson(hits)}
    comparisons = {}
    for name, baseline, candidate in (
            ("primary_temporal8_vs_uniform8", ("uniform", 8), ("temporal", 8)),
            ("temporal4_vs_uniform4", ("uniform", 4), ("temporal", 4)),
            ("temporal16_vs_uniform16", ("uniform", 16), ("temporal", 16)),
            ("temporal8_vs_default8", ("default", 8), ("temporal", 8))):
        comparisons[name] = paired_effect(records, baseline, candidate)
    per_case = {}
    for case in sorted(manifest["cases"], key=lambda item: item["id"]):
        arms = {}
        for arm, budget in ARM_BUDGETS:
            rows = sorted((row for row in measured(records, arm, budget)
                           if row["clip"] == case["id"]),
                          key=lambda row: (row["mode"], row["repetition"]))
            if len(rows) != 6:
                raise ValueError("per-case population mismatch")
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
            misses.append({"case_id": row["clip"], "mode": row["mode"], "arm": row["arm"],
                           "budget": row["budget"], "repetition": row["repetition"],
                           "request_id": row["response"]["request_id"], "raw_answer": row["raw_answer"],
                           "answer_anchor_present": anchor_present(
                               manifest, row["clip"], row["arm"], row["budget"])})
    regressions = []
    for name, comparison in comparisons.items():
        regressions.extend({"comparison": name, **row} for row in comparison["repetition_transitions"]
                           if row["transition"] == "hit_to_miss")
    return {"quality": quality, "comparisons": comparisons, "per_case": per_case,
            "every_miss": misses, "quality_regressions_retained": regressions,
            "quality_regression_count": len(regressions),
            "runtime_change_in_response_to_answers": "FORBIDDEN_NONE_MADE"}


def latency_outliers(records):
    result = []
    for mode in MODES:
        for arm, budget in ARM_BUDGETS:
            rows = [row for row in measured(records, arm, budget) if row["mode"] == mode]
            if len(rows) != 30:
                raise ValueError("outlier population mismatch")
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


def summarize(values):
    present = sorted(value for value in values if value is not None)
    total = len(values)
    coverage = "NOT_COVERED" if not present else "COVERED" if len(present) == total else "PARTIAL"
    result = {"coverage": coverage, "sample_count": len(present), "total_count": total,
              "median_ms": None, "p95_ms": None, "iqr_ms": None, "mad_ms": None,
              "min_ms": None, "max_ms": None}
    if not present:
        return result
    median = statistics.median(present)
    half = len(present) // 2
    lower, upper = present[:half], present[-half:] if len(present) % 2 == 0 else present[half + 1:]
    q1, q3 = (statistics.median(lower) if lower else median), (statistics.median(upper) if upper else median)
    result.update({"median_ms": median,
                   "p95_ms": present[max(0, math.ceil(0.95 * len(present)) - 1)],
                   "iqr_ms": q3 - q1,
                   "mad_ms": statistics.median([abs(value - median) for value in present]),
                   "min_ms": present[0], "max_ms": present[-1]})
    return result


def aggregates(records):
    groups = {}
    for record in measured(records):
        key = (record["mode"], record["clip"], record["arm"], record["budget"])
        groups.setdefault(key, []).append(record)
    result = []
    for key in sorted(groups):
        rows = groups[key]
        latency = {name: summarize([row.get("latency_ms", {}).get(name)
                                    if row["native_evidence_valid"] else None for row in rows])
                   for name in ("first_event", "ttft", "total", "vision", "prefill", "decode",
                                "native_elapsed")}
        hit_values = [row["expected_answer_hit"] for row in rows]
        covered = [value for value in hit_values if value is not None]
        result.append({
            "mode": key[0], "clip": key[1], "arm": key[2], "budget": key[3],
            "evaluation_role": rows[0]["evaluation_role"], "quality_scored": rows[0]["quality_scored"],
            "declared_repetitions": len(rows),
            "completed_repetitions": sum(row["status"] == "COMPLETED" for row in rows),
            "native_valid_repetitions": sum(row["native_evidence_valid"] for row in rows),
            "failed_repetitions": sum(row["status"] != "COMPLETED" for row in rows),
            "latency_ms": latency,
            "whole_device_resource": {
                "scope": "whole-device-used-memory-shared-by-cases-in-step",
                "sample_count": sum(row.get("resources", {}).get("sample_count", 0) for row in rows),
                "baseline_mib": [row.get("resources", {}).get("baseline_mib") for row in rows],
                "peak_mib": max((row.get("resources", {}).get("peak_mib", 0) for row in rows), default=0),
                "final_mib": [row.get("resources", {}).get("final_mib") for row in rows],
                "request_native_allocator": "NOT_USED"},
            "raw_substring_observations": {"covered": len(covered),
                "hits": sum(value is True for value in covered),
                "misses": sum(value is False for value in covered), "values": hit_values,
                "included_in_quality_accuracy": rows[0]["quality_scored"]},
            "raw_answer_sha256": [sha256_bytes((row["raw_answer"] or "").encode())
                                  if row["raw_answer"] is not None else None for row in rows]})
    return result


def answer_outcomes(records):
    grouped = {}
    for row in measured(records):
        key = (row["evaluation_role"], row["mode"], row["arm"], row["budget"])
        grouped.setdefault(key, []).append(row)
    result = {}
    for (role, mode, arm, budget), rows in sorted(grouped.items()):
        completed = [row for row in rows if row["status"] == "COMPLETED"]
        covered = [row for row in completed if row["expected_answer_hit"] is not None]
        cases = {}
        for row in sorted(completed, key=lambda item: (item["clip"], item["repetition"])):
            entry = cases.setdefault(row["clip"], {"answers": [], "hits": [],
                                                  "quality_scored": row["quality_scored"]})
            entry["answers"].append(row["raw_answer"])
            entry["hits"].append(row["expected_answer_hit"])
        block = {"declared": len(rows), "completed": len(completed),
                 "answer_hits": sum(row["expected_answer_hit"] is True for row in covered),
                 "answer_misses": sum(row["expected_answer_hit"] is False for row in covered),
                 "cases": cases, "raw_answers_retained": True,
                 "included_in_quality_accuracy": role == "quality_scored"}
        rate = block["answer_hits"] / len(covered) if covered else None
        if role == "quality_scored":
            block["preregistered_substring_accuracy"] = rate
        else:
            block["diagnostic_substring_rate_not_quality"] = rate
        block["budget"] = budget
        result.setdefault(role, {}).setdefault(mode, {}).setdefault(arm, {})[str(budget)] = block
    return result


def verify(root, output):
    root = Path(root).resolve()
    output = Path(output).resolve()
    if root in output.parents:
        raise ValueError("verification output must remain outside the sealed evidence tree")
    file_count = verify_sums(root)
    report = json.loads((root / "report.json").read_bytes())
    if report.get("schema_version") != 1 \
            or report.get("experiment") != "qwen38_temporal_heldout_quality_v1" \
            or report.get("fixture_revision") != "temporal-heldout-r1":
        raise ValueError("unsupported held-out evidence")
    fixture_root = root / "input" / "fixture"
    frozen_count = verify_sums(fixture_root, fixture=True)
    manifest = json.loads((fixture_root / "manifest.json").read_bytes())
    freeze = json.loads((root / "input" / "freeze.json").read_bytes())
    if len(manifest["cases"]) != 10 or len(manifest["videos"]) != 8 \
            or HISTORICAL_TASK61 & {case["id"] for case in manifest["cases"]} \
            or manifest["source"]["natural_video_claim"] is not False:
        raise ValueError("held-out source/population boundary mismatch")
    if freeze["manifest_sha256"] != sha256_file(fixture_root / "manifest.json") \
            or freeze["preregistration_sha256"] != sha256_file(fixture_root / "preregistration.json") \
            or freeze["sha256sums_sha256"] != sha256_file(fixture_root / "SHA256SUMS") \
            or freeze != report["freeze_receipt"] \
            or freeze["historical_task61_cases_in_denominator"]:
        raise ValueError("frozen fixture receipt mismatch")
    if sha256_file(fixture_root / "manifest.json") != report["manifest_sha256"] \
            or sha256_file(fixture_root / "preregistration.json") != report["preregistration_sha256"]:
        raise ValueError("report is bound to a different frozen input")
    analysis_hashes = report.get("analysis_source_sha256", {})
    if set(analysis_hashes) != ANALYSIS_SOURCES:
        raise ValueError("sealed analysis source set is incomplete")
    for name, digest in analysis_hashes.items():
        if not HEX256.fullmatch(digest) or sha256_file(root / "input" / "analysis" / name) != digest:
            raise ValueError("sealed analysis source hash mismatch")
    if sha256_file(Path(__file__).resolve()) != analysis_hashes["tests/temporal_heldout_verify.py"]:
        raise ValueError("running independent verifier differs from the sealed verifier")
    oracles = build_oracles(fixture_root, manifest)
    cases = {case["id"]: case for case in manifest["cases"]}
    records = report["records"]
    if len(records) != 560 or not report["gate_pass"] or report["gate_reasons"]:
        raise ValueError("producer technical gate did not pass 560 requests")
    for mode in MODES:
        mode_report = json.loads((root / mode / "mode-result.json").read_bytes())
        if mode_report["provenance"]["physical_gpu"] != 1 \
                or mode_report["provenance"]["batch_size"] != (1 if mode == "serial" else 4) \
                or mode_report["provenance"]["qwen38_prefix_cache"] != "disabled" \
                or mode_report["stale_errors"] \
                or any(not row["passed"] for row in mode_report["exclusive"]):
            raise ValueError("mode provenance/GPU/staleness gate failed")
        if not report["post_stop_cleanup"].get(mode) \
                or not report["post_stop_cleanup"][mode].get("passed"):
            raise ValueError("post-stop cleanup proof failed")
    if any(not row["passed"] for row in report["cleanup_checks"] + report["exclusive"]):
        raise ValueError("cleanup/in_flight/residual-memory gate failed")
    for row in records:
        validate_record(row, cases[row["clip"]], oracles[row["video_id"]], root / row["mode"])
    independent_parity = parity(records)
    if independent_parity != report["parity"]:
        raise ValueError("reported parity differs from independent reconstruction")
    analysis = quality_analysis(records, manifest)
    if analysis != report["analysis"]:
        raise ValueError("reported quality/paired/raw-answer/miss analysis differs from independent reconstruction")
    outliers = latency_outliers(records)
    if outliers != report["latency_outliers"]:
        raise ValueError("latency outlier reconstruction mismatch")
    if aggregates(records) != report["aggregates"] \
            or answer_outcomes(records) != report["answer_outcomes"]:
        raise ValueError("independent aggregate/outcome recomputation failed")
    if report["claims"]["natural_video_quality"] != "NOT_COVERED" \
            or report["claims"]["perceptual_quality"] != "NOT_RUN" \
            or report["claims"]["generalized_quality_claim"] is not False \
            or report["historical_controls"]["included_in_heldout_denominator"] is not False \
            or report["historical_controls"]["scores_changed"] is not False \
            or report["historical_controls"]["japanese_miss_and_aliases_changed"] is not False:
        raise ValueError("evidence escaped its historical/natural/perceptual claim boundary")
    result = {
        "result": "PASS", "independent_implementation": "stdlib_only_no_producer_scoring_import",
        "verifier_source_sha256": sha256_file(Path(__file__).resolve()),
        "evidence_files_hashed": file_count, "frozen_fixture_files_hashed": frozen_count,
        "full_source_frames_independently_decoded": 8 * 449,
        "requests_independently_validated": 560, "parity_checks": 500,
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
        print(f"INDEPENDENT HELD-OUT VERIFICATION FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
