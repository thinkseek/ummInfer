#!/usr/bin/env python3
"""Serial-GPU-1 controlled experiment for the preregistered motion-run selector.

This producer reuses the frozen clean schedule/transport but does not implement
the independent evidence verifier. Use tests/temporal_motion_run_verify.py in a
fresh process for acceptance; never treat this file's scoring as independent.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import umminfer_long_video_live as long_live  # noqa: E402
import umminfer_temporal_clean_fixture as fixture  # noqa: E402
import umminfer_temporal_clean_live as clean  # noqa: E402
import video_temporal_live as short  # noqa: E402

TASK_PREREG = ROOT / "docs/evidence/video-temporal-motion-run-preregistration.json"
MODES = ("serial", "batch4")
PHASES = ("baseline", "candidate")
BUDGETS = (4, 8, 16)
ALLOWED_RUNTIME_DIFF = {
    "native/qwen38/video_sampling.cpp",
    "native/qwen38_tests/test_video_input.cpp",
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, ensure_ascii=False, indent=2).encode() + b"\n")


def sha256_file(path):
    return fixture.sha256_file(Path(path))


def load_task_prereg(path=TASK_PREREG):
    value = json.loads(Path(path).read_bytes())
    if value.get("schema_version") != 1 or value.get("experiment") != "qwen38_temporal_motion_run_v1":
        raise ValueError("unsupported task preregistration")
    frozen = value["frozen_fixture"]
    if frozen["manifest_sha256"] != "62f475a74732fa3504850ebe4517dc98903ba39202dce457441c52133a6c96c0" \
            or frozen["preregistration_sha256"] != "96608dc793d54f64f55a7c46862d1b1737cbf807a8807d1b22b2fcd0502d2a77":
        raise ValueError("task preregistration is not bound to temporal-clean-r1")
    if value["schedule"]["physical_gpu"] != 1 or value["schedule"]["requests_per_phase"] != 448:
        raise ValueError("task schedule drift")
    return value


def representative(run, scores, *, scene_primary):
    if scene_primary:
        return min(run, key=lambda i: (-scores[i][1], -scores[i][0], i))
    return min(run, key=lambda i: (-scores[i][0], -scores[i][1], i))


def threshold_runs(scores, threshold, *, scene):
    """Contiguous strict-threshold runs; index zero is an anchor, never an event."""
    runs = []
    index = 1
    score_index = 1 if scene else 0
    while index < len(scores):
        if not scores[index][score_index] > threshold:
            index += 1
            continue
        end = index
        while end < len(scores) and scores[end][score_index] > threshold:
            end += 1
        runs.append(list(range(index, end)))
        index = end
    return runs


def select_motion_run(scores, budget, *, motion_threshold=0.02, scene_threshold=0.25):
    if budget not in BUDGETS or not scores:
        raise ValueError("motion-run oracle requires a 4/8/16 budget and candidates")
    count = len(scores)
    target = min(count, budget)
    selected, ordered = set(), []

    def select(index):
        if len(ordered) < target and index not in selected:
            selected.add(index)
            ordered.append(index)

    select(0)
    select(count - 1)
    scenes = [representative(run, scores, scene_primary=True)
              for run in threshold_runs(scores, scene_threshold, scene=True)]
    scenes.sort(key=lambda i: (-scores[i][1], -scores[i][0], i))
    for index in scenes:
        select(index)
    motions = [representative(run, scores, scene_primary=False)
               for run in threshold_runs(scores, motion_threshold, scene=False)]
    motions.sort(key=lambda i: (-scores[i][0], -scores[i][1], i))
    for index in motions:
        select(index)
    for slot in range(target):
        if len(ordered) >= target:
            break
        select((slot * (count - 1) + (target - 1) // 2) // (target - 1))
    for index in range(count):
        if len(ordered) >= target:
            break
        select(index)
    if len(ordered) != target:
        raise AssertionError("motion-run oracle failed to fill its bounded budget")
    return sorted(ordered)


def candidate_oracles(manifest, bundle):
    result = {}
    for video_id, video in manifest["videos"].items():
        rows = [json.loads(line) for line in
                (Path(bundle) / video["content_frames"]).read_text().splitlines()]
        if len(rows) != fixture.FRAME_COUNT:
            raise ValueError(f"incomplete frozen source-content records: {video_id}")
        scores = video["candidate_scores_motion_scene"]
        if len(scores) != video["candidate_count"] or video["candidate_stride"] != 8:
            raise ValueError(f"frozen candidate topology drift: {video_id}")
        result[video_id] = {}
        for budget in BUDGETS:
            # Confirm the unchanged baseline before deriving the sole intervention.
            baseline = clean.base.select_temporal(scores, budget)
            if baseline != video["expected_temporal"][str(budget)]["candidate_indices"]:
                raise ValueError(f"frozen baseline oracle mismatch: {video_id}/{budget}")
            selected = select_motion_run(scores, budget)
            source_indices = [index * 8 for index in selected]
            timestamps = [rows[index]["source_timestamp_seconds"] for index in source_indices]
            result[video_id][str(budget)] = {
                "candidate_indices": selected,
                "source_frame_indices": source_indices,
                "source_timestamps_seconds": timestamps,
                "content_fnv1a64": [rows[index]["decoded_native_rgb_fnv1a64"]
                                    for index in source_indices],
                "content_sha256": [rows[index]["decoded_native_rgb_sha256"]
                                   for index in source_indices],
                "temporal2_pair_first_timestamps": [
                    timestamps[min(2 * slice_index, len(timestamps) - 1)]
                    for slice_index in range((len(timestamps) + 1) // 2)],
            }
    return result


def attach_candidate_oracles(prepared, raw, bundle):
    oracles = candidate_oracles(raw, bundle)
    for case in prepared["cases"]:
        case["expected_temporal"] = copy.deepcopy(oracles[case["video_id"]])
    return oracles


def public_provenance(value):
    return long_live.public_provenance(value)


def run_mode(args):
    if os.environ.get("UMMINFER_LIVE_OPT_IN") != "1":
        raise ValueError("live execution requires UMMINFER_LIVE_OPT_IN=1")
    if args.phase not in PHASES or args.mode not in MODES:
        raise ValueError("invalid phase/mode")
    task = load_task_prereg(args.task_preregistration)
    manifest_path = Path(args.manifest).resolve()
    freeze_path = Path(args.freeze).resolve()
    raw = clean.load_manifest(manifest_path, deep=True, freeze_path=freeze_path)
    if sha256_file(manifest_path) != task["frozen_fixture"]["manifest_sha256"]:
        raise ValueError("clean manifest drift")
    committed = clean.validate_committed_fixture(args.source_root, manifest_path.parent, freeze_path)
    prepared = clean.prepare_budget_cases(raw, manifest_path.parent, list(BUDGETS), short)
    oracles = attach_candidate_oracles(prepared, raw, manifest_path.parent) \
        if args.phase == "candidate" else None
    out = Path(args.out_dir).resolve()
    if out.exists():
        raise ValueError(f"refusing to overwrite mode evidence: {out}")
    out.mkdir(parents=True)
    provenance_path = Path(args.provenance).resolve()
    provenance = long_live.validate_provenance(
        provenance_path, mode=args.mode, run_id=args.run_id, source_root=args.source_root)
    if provenance["physical_gpu"] != 1:
        raise ValueError("Task #61 permits only physical GPU 1")
    url = long_live.loopback_url(args.url)
    long_live.health_probe(url)
    pre, _, _ = long_live.exclusive_check(
        provenance, clean.controlled.query_compute_apps, long_live.query_memory)
    pre.update({"mode": args.mode, "phase": "pre_schedule"})
    if not pre["passed"]:
        raise ValueError(f"GPU 1 is not exclusive before traffic: {pre['problems']}")
    shutil.copy2(provenance_path, out / "provenance.json")
    write_json(out / "pre-request.json", {
        "task_preregistration_sha256": sha256_file(args.task_preregistration),
        "committed_frozen_fixture": committed, "model_requests_started": False,
        "phase": args.phase, "mode": args.mode,
    })
    original_modes = clean.MODES
    stale_errors = []
    try:
        clean.MODES = (args.mode,)
        result = clean.execute_schedule(
            prepared, transports={args.mode: lambda body, timeout: short.stream_chat(url, body, timeout)},
            native_logs=lambda: {args.mode: short.parse_native_log(Path(args.native_log).resolve())},
            execution_logs=lambda: {
                args.mode: short.parse_native_execution_log(Path(args.native_log).resolve())},
            health_probes={args.mode: lambda: long_live.health_probe(url)},
            memory_samplers={args.mode: long_live.query_memory},
            provenances={args.mode: provenance}, out_dir=out, short=short)
    finally:
        clean.MODES = original_modes
    post, _, _ = long_live.exclusive_check(
        provenance, clean.controlled.query_compute_apps, long_live.query_memory)
    post.update({"mode": args.mode, "phase": "post_schedule"})
    try:
        long_live.validate_provenance(
            provenance_path, mode=args.mode, run_id=args.run_id, source_root=args.source_root)
        clean.load_manifest(manifest_path, deep=True, freeze_path=freeze_path)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        stale_errors.append(str(error))
    shutil.copy2(Path(args.native_log).resolve(), out / "native-stage.log")
    mode_report = {
        "schema_version": 1, "phase": args.phase, "mode": args.mode,
        "run_id": args.run_id, "task_preregistration_sha256": sha256_file(args.task_preregistration),
        "fixture_sha256": sha256_file(manifest_path),
        "provenance": public_provenance(provenance), "exclusive": [pre, post],
        "stale_errors": stale_errors, "candidate_oracles": oracles,
        **result,
    }
    write_json(out / "mode-result.json", mode_report)
    print(json.dumps({"phase": args.phase, "mode": args.mode, "records": len(result["records"]),
                      "completed": sum(row["status"] == "COMPLETED" for row in result["records"]),
                      "native_valid": sum(row["native_evidence_valid"] for row in result["records"]),
                      "stale_errors": stale_errors}, indent=2))
    return 0


def load_mode(root, phase, mode):
    path = Path(root) / phase / mode / "mode-result.json"
    value = json.loads(path.read_bytes())
    if value.get("schema_version") != 1 or value.get("phase") != phase or value.get("mode") != mode:
        raise ValueError(f"invalid mode result: {path}")
    if len(value["records"]) != 224:
        raise ValueError(f"incomplete mode schedule: {phase}/{mode}")
    cleanup_path = path.parent / "cleanup.json"
    value["post_stop_cleanup"] = json.loads(cleanup_path.read_bytes()) if cleanup_path.exists() else None
    for record in value["records"]:
        relative = Path("records") / mode
        relative /= f"budget-{record['budget']}"
        relative = relative / record["phase"] / f"repetition-{record['repetition']}" \
            / record["arm"] / f"{record['clip']}.json"
        disk = json.loads((path.parent / relative).read_bytes())
        if disk != record:
            raise ValueError(f"raw record differs from mode result: {phase}/{mode}/{record['clip']}")
        frames = json.loads((path.parent / relative.with_suffix(".sse.json")).read_bytes())
        if not isinstance(frames, list) or not frames:
            raise ValueError("missing raw SSE frame evidence")
    return value


def phase_gate(phase):
    reasons = list(phase["stale_errors"])
    records = phase["records"]
    if len(records) != 448 or any(row["status"] != "COMPLETED" for row in records):
        reasons.append("complete_repetitions: phase is not 448/448 completed")
    if any(not row["native_evidence_valid"] for row in records):
        reasons.append("native evidence/RGB/PTS/temporal-2/cold-prefix validation failed")
    if len(phase["parity"]) != 400 or any(not row["passed"] for row in phase["parity"]):
        reasons.append("default/uniform, serial/batch or repetition parity failed")
    if any(not row["passed"] for row in phase["cleanup_checks"] + phase["exclusive"]):
        reasons.append("device exclusivity, in_flight or residual-memory cleanup failed")
    if any(not phase["post_stop_cleanup"].get(mode) or
           not phase["post_stop_cleanup"][mode].get("passed") for mode in MODES):
        reasons.append("post-stop process/GPU cleanup failed or is missing")
    for mode in MODES:
        if sum(row["mode"] == mode for row in records) != 224:
            reasons.append(f"incomplete {mode} request count")
    left = phase["provenances"]["serial"]
    right = phase["provenances"]["batch4"]
    for field in ("source_commit", "source_state_sha256", "model", "qwen38_prefix_cache"):
        if public_provenance(left[field]) != public_provenance(right[field]):
            reasons.append(f"serial/batch {field} identity differs")
    for field in ("gateway", "native_stage"):
        volatile = {"pid", "compute_pid", "compute_used_mib"}
        lval = {k: v for k, v in public_provenance(left[field]).items() if k not in volatile}
        rval = {k: v for k, v in public_provenance(right[field]).items() if k not in volatile}
        if lval != rval:
            reasons.append(f"serial/batch {field} immutable identity differs")
    return sorted(set(reasons))


def make_phase(mode_results):
    records = [record for mode in MODES for record in mode_results[mode]["records"]]
    provenances = {mode: mode_results[mode]["provenance"] for mode in MODES}
    phase = {
        "records": records,
        "provenances": provenances,
        "steps": [step for mode in MODES for step in mode_results[mode]["steps"]],
        "resources": [row for mode in MODES for row in mode_results[mode]["resources"]],
        "cleanup_checks": [row for mode in MODES for row in mode_results[mode]["cleanup_checks"]],
        "exclusive": [row for mode in MODES for row in mode_results[mode]["exclusive"]],
        "post_stop_cleanup": {mode: mode_results[mode].get("post_stop_cleanup") for mode in MODES},
        "stale_errors": [error for mode in MODES for error in mode_results[mode]["stale_errors"]],
        "candidate_oracles": {mode: mode_results[mode]["candidate_oracles"] for mode in MODES}
            if any(mode_results[mode]["candidate_oracles"] is not None for mode in MODES) else None,
    }
    phase["aggregates"] = clean.aggregate_results(records)
    phase["answer_outcomes"] = clean.answer_outcomes(records)
    phase["parity"] = long_live.parity_checks(records, clean.parity_adapter(short))
    phase["gate_reasons"] = phase_gate(phase)
    phase["gate_pass"] = not phase["gate_reasons"]
    return phase


def quality_rows(phase, arm, budget):
    return [row for row in phase["records"] if row["phase"] == "measured"
            and row["quality_scored"] and row["arm"] == arm and row["budget"] == budget]


def quality_count(phase, arm, budget):
    rows = quality_rows(phase, arm, budget)
    if len(rows) != 24 or any(row["expected_answer_hit"] is None for row in rows):
        raise ValueError(f"quality population incomplete: {arm}/{budget}")
    return sum(row["expected_answer_hit"] is True for row in rows)


def wilson(hits, count=24):
    if count == 0:
        return None
    z = 1.959963984540054
    rate = hits / count
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def sign_test(improved, worsened):
    discordant = improved + worsened
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(0, min(improved, worsened) + 1)) \
        / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def paired_effect(baseline, candidate, arm, budget):
    def by_pair(phase):
        result = {}
        for row in quality_rows(phase, arm, budget):
            key = (row["clip"], row["mode"])
            result.setdefault(key, []).append(row["expected_answer_hit"])
        if any(len(values) != 3 for values in result.values()):
            raise ValueError("paired semantic population has missing repetitions")
        return result
    left, right = by_pair(baseline), by_pair(candidate)
    improved = worsened = 0
    pairs = []
    for key in sorted(left):
        # A pair improves/worsens only when its three paired hit sums differ.
        lsum, rsum = sum(left[key]), sum(right[key])
        outcome = "unchanged"
        if rsum > lsum:
            improved += 1
            outcome = "improved"
        elif rsum < lsum:
            worsened += 1
            outcome = "worsened"
        pairs.append({"clip": key[0], "mode": key[1], "baseline_hits": lsum,
                      "candidate_hits": rsum, "outcome": outcome})
    return {"improved_pairs": improved, "worsened_pairs": worsened,
            "unchanged_pairs": len(pairs) - improved - worsened,
            "exact_two_sided_sign_test_p": sign_test(improved, worsened), "pairs": pairs}


def hit_to_miss_repetitions(baseline, candidate, arm, budget):
    def indexed(phase):
        return {(row["clip"], row["mode"], row["repetition"]): row["expected_answer_hit"]
                for row in quality_rows(phase, arm, budget)}
    left, right = indexed(baseline), indexed(candidate)
    if set(left) != set(right):
        raise ValueError("paired repetition population drift")
    return sum(left[key] is True and right[key] is False for key in left)


def cross_phase_unchanged(baseline, candidate):
    def index(phase):
        return {(row["mode"], row["phase"], row["repetition"], row["clip"],
                 row["arm"], row["budget"]): row for row in phase["records"]
                if row["arm"] in ("default", "uniform")}
    left, right = index(baseline), index(candidate)
    if set(left) != set(right) or len(left) != 256:
        raise ValueError("default/uniform cross-phase population drift")
    checks = []
    for key in sorted(left):
        a, b = left[key], right[key]
        request = a["request"] == b["request"]
        answer = a["raw_answer"] == b["raw_answer"]
        native = short.native_parity(a["native"], b["native"])
        checks.append({"mode": key[0], "phase": key[1], "repetition": key[2],
                       "clip": key[3], "arm": key[4], "budget": key[5],
                       "request_byte_identical": request, "answer_byte_identical": answer,
                       "native_selection_temporal2_mrope_identical": native,
                       "passed": request and answer and native})
    return checks


def median_latency(phase, mode, budget, metric):
    values = [row["latency_ms"][metric] for row in phase["records"]
              if row["phase"] == "measured" and row["arm"] == "temporal"
              and row["mode"] == mode and row["budget"] == budget]
    if len(values) != 24:
        raise ValueError("latency population incomplete")
    return statistics.median(values)


def comparison(baseline, candidate):
    arms = [("default", 8)] + [(arm, budget) for arm in ("uniform", "temporal")
                              for budget in BUDGETS]
    quality = {}
    for arm, budget in arms:
        key = f"{arm}-{budget}"
        b = quality_count(baseline, arm, budget)
        c = quality_count(candidate, arm, budget)
        quality[key] = {
            "baseline": {"hits": b, "total": 24, "misses": 24 - b,
                         "accuracy": b / 24, "wilson_95": wilson(b)},
            "candidate": {"hits": c, "total": 24, "misses": 24 - c,
                          "accuracy": c / 24, "wilson_95": wilson(c)},
            "paired_case_mode_effect": paired_effect(baseline, candidate, arm, budget),
            "hit_to_miss_repetitions": hit_to_miss_repetitions(baseline, candidate, arm, budget),
        }
    unchanged = cross_phase_unchanged(baseline, candidate)
    latency = []
    for mode in MODES:
        for budget in BUDGETS:
            for metric in ("total", "native_elapsed"):
                b = median_latency(baseline, mode, budget, metric)
                c = median_latency(candidate, mode, budget, metric)
                latency.append({"mode": mode, "budget": budget, "metric": metric,
                                "baseline_median_ms": b, "candidate_median_ms": c,
                                "limit_ratio": 1.05, "ratio": c / b,
                                "passed": c <= b * 1.05})
    reasons = []
    primary = quality["temporal-8"]
    if not primary["candidate"]["hits"] > primary["baseline"]["hits"]:
        reasons.append("primary: candidate temporal-8 did not exceed baseline temporal-8")
    if primary["candidate"]["hits"] < max(quality["default-8"]["baseline"]["hits"],
                                          quality["uniform-8"]["baseline"]["hits"]):
        reasons.append("primary: candidate temporal-8 remains below default/uniform-8")
    for budget in BUDGETS:
        if quality[f"temporal-{budget}"]["candidate"]["hits"] < \
                quality[f"temporal-{budget}"]["baseline"]["hits"]:
            reasons.append(f"quality no-regression: temporal-{budget} hits decreased")
        effect = quality[f"temporal-{budget}"]["paired_case_mode_effect"]
        if effect["worsened_pairs"] or quality[f"temporal-{budget}"]["hit_to_miss_repetitions"]:
            reasons.append(f"quality no-regression: temporal-{budget} has hit-to-miss repetitions")
    if any(not row["passed"] for row in unchanged):
        reasons.append("default/uniform cross-phase byte identity failed")
    if any(not row["passed"] for row in latency):
        reasons.append("temporal median latency exceeded the preregistered 5 percent limit")
    return {"quality": quality, "default_uniform_cross_phase": unchanged,
            "latency": latency, "gate_reasons": sorted(set(reasons))}


def runtime_diff(source_root, baseline_commit, candidate_commit):
    names = subprocess.run(["git", "-C", str(source_root), "diff", "--name-only",
                            baseline_commit, candidate_commit], check=True,
                           capture_output=True, text=True, timeout=30).stdout.splitlines()
    diff = subprocess.run(["git", "-C", str(source_root), "diff", baseline_commit,
                           candidate_commit, "--", *sorted(ALLOWED_RUNTIME_DIFF)],
                          check=True, capture_output=True, timeout=30).stdout
    return {"baseline_commit": baseline_commit, "candidate_commit": candidate_commit,
            "changed_files": names, "allowed_files": sorted(ALLOWED_RUNTIME_DIFF),
            "passed": bool(names) and set(names) <= ALLOWED_RUNTIME_DIFF,
            "diff_sha256": hashlib.sha256(diff).hexdigest()}


def seal(root):
    root = Path(root)
    report_path = root / "report.json"
    if not report_path.is_file():
        raise ValueError("cannot seal without report.json")
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in ("artifact-manifest.json", "SHA256SUMS"):
            entries.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                            "sha256": sha256_file(path)})
    write_json(root / "artifact-manifest.json", {"schema_version": 1, "hash": "sha256", "files": entries})
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n")
    (root / "SHA256SUMS").write_text("".join(lines), encoding="ascii")


def finalize(args):
    root = Path(args.out_dir).resolve()
    if root.exists():
        raise ValueError("refusing to overwrite an experiment bundle")
    task = load_task_prereg(args.task_preregistration)
    phases = {}
    for phase_name in PHASES:
        modes = {mode: load_mode(args.mode_root, phase_name, mode) for mode in MODES}
        for mode in MODES:
            if modes[mode]["task_preregistration_sha256"] != sha256_file(args.task_preregistration):
                raise ValueError("mode used a different task preregistration")
        phases[phase_name] = make_phase(modes)
    root.mkdir(parents=True)
    clean.copy_frozen_input(Path(args.manifest).resolve().parent, Path(args.freeze).resolve(), root)
    shutil.copy2(args.task_preregistration, root / "input" / "task-preregistration.json")
    analysis_names = ("temporal_motion_run_live.py", "temporal_motion_run_test.py",
                      "temporal_motion_run_verify.py")
    analysis_hashes = {}
    for name in analysis_names:
        source = ROOT / "tests" / name
        target = root / "input" / "analysis" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        analysis_hashes[name] = sha256_file(target)
    # Move (not copy) completed raw mode trees into their final sealed locations.
    for phase_name in PHASES:
        for mode in MODES:
            source = Path(args.mode_root) / phase_name / mode
            target = root / phase_name / mode
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target)
    compare = comparison(phases["baseline"], phases["candidate"])
    diff = runtime_diff(args.source_root, args.baseline_commit, args.candidate_commit)
    reasons = list(compare["gate_reasons"])
    for phase_name in PHASES:
        reasons.extend(f"{phase_name}: {reason}" for reason in phases[phase_name]["gate_reasons"])
    if not diff["passed"]:
        reasons.append("candidate commit changes files outside the declared native/test intervention")
    if phases["baseline"]["provenances"]["serial"]["gateway"]["sha256"] != \
            phases["candidate"]["provenances"]["serial"]["gateway"]["sha256"]:
        reasons.append("gateway binary changed between baseline and candidate")
    if phases["baseline"]["provenances"]["serial"]["model"] != \
            phases["candidate"]["provenances"]["serial"]["model"]:
        reasons.append("model identity changed between baseline and candidate")
    accepted = not reasons
    report = {
        "schema_version": 1, "experiment": task["experiment"],
        "evidence_class": "CONTROLLED_LIVE" if all(phases[p]["gate_pass"] for p in PHASES)
            else "UNCONTROLLED_LIVE",
        "task_preregistration_sha256": sha256_file(args.task_preregistration),
        "analysis_source_sha256": analysis_hashes,
        "freeze_receipt": fixture.validate_freeze(root / "input" / "fixture", root / "input" / "freeze.json"),
        "phases": phases, "comparison": compare, "runtime_diff": diff,
        "candidate_accepted": accepted, "decision_reasons": sorted(set(reasons)),
        "claims": {"real_generation": "MEASURED_LIMITED_FIXTURE",
                   "live_latency": "CONTROLLED_LIVE" if all(phases[p]["gate_pass"] for p in PHASES)
                       else "UNCONTROLLED_LIVE",
                   "fixture_substring_accuracy": "MEASURED_LIMITED_FIXTURE",
                   "perceptual_quality": "NOT_RUN", "generalized_quality_claim": False},
    }
    write_json(root / "report.json", report)
    seal(root)
    print(json.dumps({"candidate_accepted": accepted, "decision_reasons": report["decision_reasons"],
                      "quality": {key: {"baseline": value["baseline"]["hits"],
                                        "candidate": value["candidate"]["hits"]}
                                  for key, value in compare["quality"].items()},
                      "records": {phase: len(value["records"]) for phase, value in phases.items()},
                      "sha256sums_sha256": sha256_file(root / "SHA256SUMS")}, indent=2))
    return 0 if accepted else 2


def self_check():
    task = load_task_prereg()
    raw = clean.load_manifest(ROOT / "benchmarks/video-temporal-clean/manifest.json", deep=False)
    oracles = candidate_oracles(raw, ROOT / "benchmarks/video-temporal-clean")
    assert oracles["motion-clean"]["8"]["source_frame_indices"] == [0, 24, 64, 128, 184, 192, 328, 448]
    assert oracles["scene-clean"]["8"] == raw["videos"]["scene-clean"]["expected_temporal"]["8"]
    for video in oracles.values():
        for budget in BUDGETS:
            selected = video[str(budget)]
            assert len(selected["candidate_indices"]) == budget
            assert selected["source_frame_indices"][0] == 0
            assert selected["source_frame_indices"][-1] == 448
            assert selected["candidate_indices"] == sorted(set(selected["candidate_indices"]))
    assert task["decision"]["perceptual_quality"] == "NOT_RUN"
    print("SELF-CHECK PASS: frozen baseline, motion-run boundaries, anchors, budgets and claim boundary")
    print("REAL_GENERATION: NOT_RUN")
    print("PERCEPTUAL_QUALITY: NOT_RUN")
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-mode")
    for name in ("phase", "mode", "url", "native-log", "provenance", "source-root", "run-id", "out-dir"):
        run.add_argument(f"--{name}", required=True)
    run.add_argument("--manifest", default=str(fixture.DEFAULT_BUNDLE / "manifest.json"))
    run.add_argument("--freeze", default=str(fixture.DEFAULT_FREEZE))
    run.add_argument("--task-preregistration", default=str(TASK_PREREG))
    final = sub.add_parser("finalize")
    for name in ("mode-root", "out-dir", "source-root", "baseline-commit", "candidate-commit"):
        final.add_argument(f"--{name}", required=True)
    final.add_argument("--manifest", default=str(fixture.DEFAULT_BUNDLE / "manifest.json"))
    final.add_argument("--freeze", default=str(fixture.DEFAULT_FREEZE))
    final.add_argument("--task-preregistration", default=str(TASK_PREREG))
    sub.add_parser("self-check")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "self-check":
        return self_check()
    if args.command == "run-mode":
        return run_mode(args)
    return finalize(args)


if __name__ == "__main__":
    sys.exit(main())
