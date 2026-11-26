#!/usr/bin/env python3
"""Fresh-process, stdlib-only verifier for temporal-motion-run evidence.

No producer, fixture-generator, transport, or scoring module is imported. The
verifier independently decodes source RGB/PTS, recomputes luma/scene scores and
both selectors, applies the frozen substring rules, and rebuilds all parity and
promotion decisions. Verification output must be written outside the sealed tree.
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

HEX256 = re.compile(r"[0-9a-f]{64}")
MODES = ("serial", "batch4")
PHASES = ("baseline", "candidate")
BUDGETS = (4, 8, 16)
ARMS = (("default", 8),) + tuple((arm, budget) for arm in ("uniform", "temporal")
                                 for budget in BUDGETS)
NATIVE_PARITY_FIELDS = (
    "policy", "decoded_frame_count", "selected_frame_count",
    "selected_candidate_indices", "selected_source_frame_indices",
    "source_indices_exact", "selected_frame_timestamps",
    "selected_frame_content_fnv1a64", "has_nominal_timestamps",
    "temporal2_pair_first_timestamps", "max_frames", "max_candidate_frames",
    "motion_threshold", "scene_threshold", "grid_t", "grid_h", "grid_w",
    "premerge_patch_rows", "merged_tokens_per_slice", "video_token_count",
    "prompt_token_count", "media_hash", "mrope_next", "mrope_fnv1a64",
)


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
        if not HEX256.fullmatch(digest) or relative.startswith("/") or ".." in Path(relative).parts \
                or relative in sums:
            raise ValueError("malformed or duplicate checksum entry")
        sums[relative] = digest
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name != "SHA256SUMS"}
    if set(sums) != actual:
        missing = sorted(set(sums) - actual)[:3]
        extra = sorted(actual - set(sums))[:3]
        raise ValueError(f"checksum file-set mismatch: missing={missing}, extra={extra}")
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
            text = line.split("pts_time:", 1)[1].lstrip()
            match = re.match(r"[-+0-9.eE]+", text)
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
                values = []
                for y in range(y0, y1):
                    for x in range(x0, x1):
                        offset = (y * 224 + x) * 3
                        values.append((77 * frame[offset] + 150 * frame[offset + 1]
                                       + 29 * frame[offset + 2]) >> 8)
                mean = sum(values) // len(values)
                luma.append(mean)
                histogram[mean * 16 // 256] += 1
        analyzed.append((luma, histogram))
    scores = [[0.0, 0.0]]
    for index in range(1, len(candidates)):
        previous, current = analyzed[index - 1], analyzed[index]
        motion = sum(abs(a - b) for a, b in zip(previous[0], current[0])) / (255.0 * 256.0)
        scene = sum(abs(a - b) for a, b in zip(previous[1], current[1])) / (2.0 * 256.0)
        scores.append([motion, scene])
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


def select(scores, budget, candidate):
    target = min(len(scores), budget)
    used, ordered = set(), []

    def take(index):
        if len(ordered) < target and index not in used:
            used.add(index)
            ordered.append(index)

    take(0)
    take(len(scores) - 1)
    scene_runs = runs(scores, 0.25, 1)
    scene_reps = [min(run, key=lambda i: (-scores[i][1], -scores[i][0], i))
                  for run in scene_runs]
    scene_reps.sort(key=lambda i: (-scores[i][1], -scores[i][0], i))
    for index in scene_reps:
        take(index)
    if candidate:
        motion_runs = runs(scores, 0.02, 0)
        motion = [min(run, key=lambda i: (-scores[i][0], -scores[i][1], i))
                  for run in motion_runs]
    else:
        motion = [index for index in range(1, len(scores)) if scores[index][0] > 0.02]
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
        raise ValueError("independent selector did not fill budget")
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
        for index, frame in enumerate(frames):
            if sha256_bytes(frame) != rows[index]["decoded_native_rgb_sha256"] \
                    or fnv1a64(frame) != rows[index]["decoded_native_rgb_fnv1a64"] \
                    or pts[index] != rows[index]["source_timestamp_seconds"]:
                raise ValueError(f"independent full decode diverges: {video_id}/{index}")
        candidate_indices = list(range(0, 449, 8))
        scores = analyze([frames[index] for index in candidate_indices])
        if scores != video["candidate_scores_motion_scene"]:
            raise ValueError(f"independent candidate scores diverge: {video_id}")
        historical, _ = decode(path, historical_uniform=True)
        if len(historical) != 8:
            raise ValueError(f"historical uniform decode count mismatch: {video_id}")
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
            raise ValueError(f"historical source positions changed: {video_id}")
        result[video_id] = {"frames": frames, "pts": pts, "scores": scores,
                            "uniform8_sources": uniform8_sources,
                            "uniform8_frames": historical}
    return result


def hit(answer, case):
    folded = answer.casefold()
    terms = case.get("expect_all", [])
    if not terms:
        return None
    if case.get("expect_any_group"):
        return any(term.casefold() in folded for term in terms)
    return all(term.casefold() in folded for term in terms)


def expected_record(record, case, phase, oracle):
    native = record["native"]
    execution = record["native_execution"]
    budget, arm = record["budget"], record["arm"]
    if native["request_id"] != record["response"]["request_id"] \
            or execution["request_id"] != native["request_id"]:
        raise ValueError("public/native request identity mismatch")
    if native["policy"] != ("temporal" if arm == "temporal" else "uniform") \
            or native["engine_path"] != ("serial" if record["mode"] == "serial" else "batched"):
        raise ValueError("native policy/engine mismatch")
    if execution["prefix_reused"] != 0 or native["max_frames"] != budget \
            or native["selected_frame_count"] != budget:
        raise ValueError("cold-prefix or selected budget mismatch")
    if arm == "temporal":
        positions = select(oracle["scores"], budget, phase == "candidate")
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
    fingerprints = [fnv1a64(frame) for frame in frames]
    if native["selected_frame_content_fnv1a64"] != fingerprints:
        raise ValueError("native selected RGB differs from independent full decode")
    expected_pairs = [timestamps[min(2 * index, budget - 1)] for index in range((budget + 1) // 2)]
    if native["temporal2_pair_first_timestamps"] != expected_pairs \
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


def validate_record(record, case, phase, oracle, mode_dir):
    if record["status"] != "COMPLETED" or not record["native_evidence_valid"] \
            or record["response"]["http_status"] != 200:
        raise ValueError("failed or invalid request retained in a passing technical population")
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
    for field, expected in (("temperature", 0.0), ("top_p", 1.0), ("seed", 510051),
                            ("max_tokens", 64)):
        if record["request"][field] != expected:
            raise ValueError(f"request sampling drift: {field}")
    expected_record(record, case, phase, oracle)
    relative = Path("records") / record["mode"] / f"budget-{record['budget']}" \
        / record["phase"] / f"repetition-{record['repetition']}" / record["arm"] \
        / f"{record['clip']}.json"
    if json.loads((mode_dir / relative).read_bytes()) != record:
        raise ValueError("raw per-request record differs from report")
    frames = json.loads((mode_dir / relative.with_suffix(".sse.json")).read_bytes())
    if not isinstance(frames, list) or not frames:
        raise ValueError("raw SSE frames missing")


def native_equal(left, right):
    return all(left.get(field) == right.get(field) for field in NATIVE_PARITY_FIELDS)


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
        # Canonical request comparison excludes only frame_selection and its old digest.
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
                       "request_projection_equal": request, "answer_byte_identical": answer,
                       "native_selection_temporal2_mrope_identical": native,
                       "passed": request and answer and native})
    cross = {}
    for row in records:
        key = (row["phase"], row["repetition"], row["clip"], row["arm"], row["budget"])
        cross.setdefault(key, {})[row["mode"]] = row
    for key, values in sorted(cross.items()):
        a, b = values["serial"], values["batch4"]
        request = a["request"] == b["request"]
        answer = a["raw_answer"] == b["raw_answer"]
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
    if len(checks) != 400 or any(not row["passed"] for row in checks):
        raise ValueError("independent 400-check parity gate failed")
    return checks


def wilson(hits, count=24):
    z = 1.959963984540054
    rate = hits / count
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def qrows(phase, arm, budget):
    rows = [row for row in phase["records"] if row["phase"] == "measured" and row["quality_scored"]
            and row["arm"] == arm and row["budget"] == budget]
    if len(rows) != 24:
        raise ValueError("quality population incomplete")
    return rows


def paired(phase_a, phase_b, arm, budget):
    def grouped(phase):
        result = {}
        for row in qrows(phase, arm, budget):
            result.setdefault((row["clip"], row["mode"]), []).append(row["expected_answer_hit"])
        return {key: sum(values) for key, values in result.items()}
    a, b = grouped(phase_a), grouped(phase_b)
    improved = sum(b[key] > a[key] for key in a)
    worsened = sum(b[key] < a[key] for key in a)
    n = improved + worsened
    p = 1.0 if not n else min(1.0, 2 * sum(math.comb(n, k)
            for k in range(min(improved, worsened) + 1)) / (2 ** n))
    return improved, worsened, p


def median(phase, mode, budget, metric):
    values = [row["latency_ms"][metric] for row in phase["records"]
              if row["phase"] == "measured" and row["arm"] == "temporal"
              and row["mode"] == mode and row["budget"] == budget]
    if len(values) != 24:
        raise ValueError("latency population incomplete")
    return statistics.median(values)


def verify_decision(report):
    baseline, candidate = report["phases"]["baseline"], report["phases"]["candidate"]
    quality = {}
    reasons = []
    for arm, budget in ARMS:
        b = sum(row["expected_answer_hit"] is True for row in qrows(baseline, arm, budget))
        c = sum(row["expected_answer_hit"] is True for row in qrows(candidate, arm, budget))
        improved, worsened, p = paired(baseline, candidate, arm, budget)
        def transitions(phase):
            return {(row["clip"], row["mode"], row["repetition"]): row["expected_answer_hit"]
                    for row in qrows(phase, arm, budget)}
        transition_a, transition_b = transitions(baseline), transitions(candidate)
        hit_to_miss = sum(transition_a[key] is True and transition_b[key] is False
                          for key in transition_a)
        key = f"{arm}-{budget}"
        quality[key] = {"baseline_hits": b, "candidate_hits": c, "total": 24,
                        "baseline_wilson_95": wilson(b), "candidate_wilson_95": wilson(c),
                        "improved_pairs": improved, "worsened_pairs": worsened, "sign_test_p": p,
                        "hit_to_miss_repetitions": hit_to_miss}
        recorded = report["comparison"]["quality"][key]
        if recorded["baseline"]["hits"] != b or recorded["candidate"]["hits"] != c \
                or recorded["baseline"]["wilson_95"] != wilson(b) \
                or recorded["candidate"]["wilson_95"] != wilson(c) \
                or recorded["paired_case_mode_effect"]["improved_pairs"] != improved \
                or recorded["paired_case_mode_effect"]["worsened_pairs"] != worsened \
                or recorded["paired_case_mode_effect"]["exact_two_sided_sign_test_p"] != p \
                or recorded["hit_to_miss_repetitions"] != hit_to_miss:
            raise ValueError(f"reported quality/uncertainty mismatch: {key}")
    if not quality["temporal-8"]["candidate_hits"] > quality["temporal-8"]["baseline_hits"]:
        reasons.append("primary_improvement_failed")
    if quality["temporal-8"]["candidate_hits"] < max(quality["default-8"]["baseline_hits"],
                                                     quality["uniform-8"]["baseline_hits"]):
        reasons.append("primary_uniform_coverage_target_failed")
    for budget in BUDGETS:
        block = quality[f"temporal-{budget}"]
        if block["candidate_hits"] < block["baseline_hits"] or block["worsened_pairs"] \
                or block["hit_to_miss_repetitions"]:
            reasons.append(f"temporal_{budget}_quality_regression")
    def unchanged(phase):
        return {(row["mode"], row["phase"], row["repetition"], row["clip"], row["arm"], row["budget"]): row
                for row in phase["records"] if row["arm"] in ("default", "uniform")}
    a, b = unchanged(baseline), unchanged(candidate)
    if len(a) != 256 or set(a) != set(b):
        raise ValueError("cross-phase default/uniform population mismatch")
    for key in a:
        if a[key]["request"] != b[key]["request"] or a[key]["raw_answer"] != b[key]["raw_answer"] \
                or not native_equal(a[key]["native"], b[key]["native"]):
            reasons.append("default_or_uniform_changed")
            break
    for mode in MODES:
        for budget in BUDGETS:
            for metric in ("total", "native_elapsed"):
                left, right = median(baseline, mode, budget, metric), median(candidate, mode, budget, metric)
                recorded_latency = [row for row in report["comparison"]["latency"]
                                    if row["mode"] == mode and row["budget"] == budget
                                    and row["metric"] == metric]
                if len(recorded_latency) != 1:
                    raise ValueError("reported latency population mismatch")
                row = recorded_latency[0]
                if row["baseline_median_ms"] != left or row["candidate_median_ms"] != right \
                        or row["ratio"] != right / left or row["passed"] != (right <= left * 1.05):
                    raise ValueError("reported latency median/ratio mismatch")
                if right > left * 1.05:
                    reasons.append(f"latency_regression:{mode}:{budget}:{metric}")
    accepted = not reasons and all(report["phases"][phase]["gate_pass"] for phase in PHASES) \
        and report["runtime_diff"]["passed"] \
        and baseline["provenances"]["serial"]["gateway"]["sha256"] == \
            candidate["provenances"]["serial"]["gateway"]["sha256"] \
        and baseline["provenances"]["serial"]["model"] == candidate["provenances"]["serial"]["model"]
    if accepted != report["candidate_accepted"]:
        raise ValueError(f"independent promotion decision mismatch: recomputed={accepted}, reasons={reasons}")
    return quality, sorted(set(reasons)), accepted


def verify(root, output):
    root = Path(root).resolve()
    output = Path(output).resolve()
    if root in output.parents:
        raise ValueError("verification output must remain outside the sealed evidence tree")
    file_count = verify_sums(root)
    report = json.loads((root / "report.json").read_bytes())
    if report.get("schema_version") != 1 or report.get("experiment") != "qwen38_temporal_motion_run_v1":
        raise ValueError("unsupported experiment evidence")
    fixture_root = root / "input" / "fixture"
    frozen_count = verify_sums(fixture_root, fixture=True)
    manifest = json.loads((fixture_root / "manifest.json").read_bytes())
    freeze = json.loads((root / "input" / "freeze.json").read_bytes())
    if freeze["manifest_sha256"] != sha256_file(fixture_root / "manifest.json") \
            or freeze["preregistration_sha256"] != sha256_file(fixture_root / "preregistration.json") \
            or freeze["sha256sums_sha256"] != sha256_file(fixture_root / "SHA256SUMS") \
            or freeze != report["freeze_receipt"]:
        raise ValueError("frozen fixture receipt mismatch")
    task = json.loads((root / "input" / "task-preregistration.json").read_bytes())
    if sha256_file(root / "input" / "task-preregistration.json") != report["task_preregistration_sha256"] \
            or task["schedule"]["physical_gpu"] != 1 or task["decision"]["perceptual_quality"] != "NOT_RUN":
        raise ValueError("task preregistration mismatch")
    analysis_hashes = report.get("analysis_source_sha256", {})
    if set(analysis_hashes) != {"temporal_motion_run_live.py", "temporal_motion_run_test.py",
                                "temporal_motion_run_verify.py"}:
        raise ValueError("sealed analysis source set is incomplete")
    for name, digest in analysis_hashes.items():
        if not HEX256.fullmatch(digest) or sha256_file(root / "input" / "analysis" / name) != digest:
            raise ValueError("sealed analysis source hash mismatch")
    if sha256_file(Path(__file__).resolve()) != analysis_hashes["temporal_motion_run_verify.py"]:
        raise ValueError("running independent verifier differs from the sealed verifier")
    oracles = build_oracles(fixture_root, manifest)
    cases = {case["id"]: case for case in manifest["cases"]}
    independent_parity = {}
    for phase_name in PHASES:
        phase = report["phases"][phase_name]
        if len(phase["records"]) != 448 or phase["gate_reasons"] or not phase["gate_pass"]:
            raise ValueError(f"producer technical gate did not pass: {phase_name}")
        for mode in MODES:
            mode_report = json.loads((root / phase_name / mode / "mode-result.json").read_bytes())
            if mode_report["provenance"]["physical_gpu"] != 1 \
                    or mode_report["provenance"]["batch_size"] != (1 if mode == "serial" else 4) \
                    or mode_report["provenance"]["qwen38_prefix_cache"] != "disabled" \
                    or mode_report["stale_errors"]:
                raise ValueError("mode provenance/GPU/staleness gate failed")
            if any(not row["passed"] for row in mode_report["exclusive"]):
                raise ValueError("whole-device exclusivity gate failed")
        if any(not row["passed"] for row in phase["cleanup_checks"] + phase["exclusive"]):
            raise ValueError("cleanup/in_flight/residual-memory gate failed")
        if any(not phase["post_stop_cleanup"].get(mode)
               or not phase["post_stop_cleanup"][mode].get("passed") for mode in MODES):
            raise ValueError("post-stop process/GPU cleanup proof failed")
        for row in phase["records"]:
            validate_record(row, cases[row["clip"]], phase_name,
                            oracles[row["video_id"]], root / phase_name / row["mode"])
        independent_parity[phase_name] = parity(phase["records"])
        if independent_parity[phase_name] != phase["parity"]:
            raise ValueError(f"reported parity differs from independent reconstruction: {phase_name}")
    quality, reasons, accepted = verify_decision(report)
    for phase_name in PHASES:
        phase = report["phases"][phase_name]
        if parity(phase["records"]) != phase["parity"]:
            raise ValueError("producer parity reconstruction mismatch")
    if report["claims"]["perceptual_quality"] != "NOT_RUN" \
            or report["claims"]["generalized_quality_claim"] is not False:
        raise ValueError("evidence escaped its perceptual/generalization boundary")
    result = {
        "result": "PASS", "independent_implementation": "stdlib_only_no_producer_scoring_import",
        "verifier_source_sha256": sha256_file(Path(__file__).resolve()),
        "evidence_files_hashed": file_count, "frozen_fixture_files_hashed": frozen_count,
        "full_source_frames_independently_decoded": 4 * 449,
        "requests_independently_validated": sum(len(report["phases"][p]["records"]) for p in PHASES),
        "parity_checks_per_phase": {phase: len(independent_parity[phase]) for phase in PHASES},
        "quality": quality, "decision_reasons": reasons, "candidate_accepted": accepted,
        "sha256sums_sha256": sha256_file(root / "SHA256SUMS"),
        "report_sha256": sha256_file(root / "report.json"),
        "claims": report["claims"],
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
        print(f"INDEPENDENT VERIFICATION FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
