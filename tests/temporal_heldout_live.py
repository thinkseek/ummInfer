#!/usr/bin/env python3
"""Run/finalize the frozen held-out long-video quality evaluation on GPU 1.

Live execution needs UMMINFER_LIVE_OPT_IN=1 and fresh serial/batch-four stacks
run sequentially on physical GPU 1. This evaluation never changes the selector,
thresholds, public runtime, historical Task61 controls, prompts, or scoring.
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
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import umminfer_long_video_live as long_live  # noqa: E402
import umminfer_temporal_heldout_fixture as fixture  # noqa: E402
import umminfer_temporal_clean_live as clean  # noqa: E402
import video_temporal_live as short  # noqa: E402

MODES = ("serial", "batch4")
BUDGETS = (4, 8, 16)
ARM_BUDGETS = fixture.ARM_BUDGETS
DEFAULT_MANIFEST = fixture.DEFAULT_BUNDLE / "manifest.json"
DEFAULT_FREEZE = fixture.DEFAULT_FREEZE
ANALYSIS_SOURCES = (
    "python/umminfer_temporal_heldout_fixture.py",
    "tests/temporal_heldout_live.py",
    "tests/temporal_heldout_test.py",
    "tests/temporal_heldout_verify.py",
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, ensure_ascii=False, indent=2).encode() + b"\n")


def sha256_file(path):
    return fixture.sha256_file(Path(path))


def sha256_bytes(data):
    return fixture.sha256_bytes(data)


def load_manifest(path=DEFAULT_MANIFEST, freeze_path=DEFAULT_FREEZE, *, deep=True):
    path = Path(path).resolve()
    fixture.validate_freeze(path.parent, freeze_path)
    manifest = fixture.validate_bundle(path.parent, deep=deep)
    if len(manifest["cases"]) != 10 or len(manifest["videos"]) != 8 \
            or manifest["measurement"]["allowed_physical_gpus"] != [1]:
        raise ValueError("held-out population/GPU identity drift")
    return manifest


def prepare(manifest, bundle=DEFAULT_MANIFEST.parent):
    prepared = copy.deepcopy(manifest)
    prepared["_budgets"] = list(BUDGETS)
    for case in prepared["cases"]:
        case["_expected_uniform"] = copy.deepcopy(case["expected_uniform"])
    return long_live.prepare_cases(prepared, bundle, short)


def wilson(hits, count=60):
    if not count:
        return None
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
    tail = sum(math.comb(count, k) for k in range(min(improved, worsened) + 1)) / (2 ** count)
    return min(1.0, 2.0 * tail)


def measured(records, arm=None, budget=None):
    return [row for row in records if row["phase"] == "measured"
            and (arm is None or row["arm"] == arm)
            and (budget is None or row["budget"] == budget)]


def quality_count(records, arm, budget):
    rows = measured(records, arm, budget)
    if len(rows) != 60 or any(row["expected_answer_hit"] is None for row in rows):
        raise ValueError(f"incomplete 60-answer quality population: {arm}/{budget}")
    return sum(row["expected_answer_hit"] is True for row in rows)


def paired_effect(records, baseline, candidate):
    def grouped(arm, budget):
        result = {}
        for row in measured(records, arm, budget):
            result.setdefault((row["clip"], row["mode"]), []).append(row)
        if len(result) != 20 or any(len(rows) != 3 for rows in result.values()):
            raise ValueError("paired case/mode population is not 20 complete three-repeat units")
        return result
    left = grouped(*baseline)
    right = grouped(*candidate)
    pairs = []
    repetition_transitions = []
    for key in sorted(left):
        lrows = sorted(left[key], key=lambda row: row["repetition"])
        rrows = sorted(right[key], key=lambda row: row["repetition"])
        lsum = sum(row["expected_answer_hit"] is True for row in lrows)
        rsum = sum(row["expected_answer_hit"] is True for row in rrows)
        outcome = "improved" if rsum > lsum else "worsened" if rsum < lsum else "unchanged"
        pairs.append({
            "case_id": key[0], "mode": key[1], "baseline_hits": lsum,
            "candidate_hits": rsum, "outcome": outcome,
            "baseline_answers": [row["raw_answer"] for row in lrows],
            "candidate_answers": [row["raw_answer"] for row in rrows],
        })
        for lrow, rrow in zip(lrows, rrows):
            if lrow["repetition"] != rrow["repetition"]:
                raise ValueError("paired repetition order drift")
            transition = ("hit_to_hit" if lrow["expected_answer_hit"] and rrow["expected_answer_hit"]
                          else "hit_to_miss" if lrow["expected_answer_hit"]
                          else "miss_to_hit" if rrow["expected_answer_hit"] else "miss_to_miss")
            repetition_transitions.append({
                "case_id": key[0], "mode": key[1], "repetition": lrow["repetition"],
                "transition": transition, "baseline_answer": lrow["raw_answer"],
                "candidate_answer": rrow["raw_answer"],
                "baseline_request_id": lrow["response"]["request_id"],
                "candidate_request_id": rrow["response"]["request_id"],
            })
    improved = sum(row["outcome"] == "improved" for row in pairs)
    worsened = sum(row["outcome"] == "worsened" for row in pairs)
    counts = {name: sum(row["transition"] == name for row in repetition_transitions)
              for name in ("hit_to_hit", "hit_to_miss", "miss_to_hit", "miss_to_miss")}
    return {
        "baseline": f"{baseline[0]}-{baseline[1]}",
        "candidate": f"{candidate[0]}-{candidate[1]}",
        "paired_case_mode_count": len(pairs), "improved_pairs": improved,
        "worsened_pairs": worsened, "unchanged_pairs": len(pairs) - improved - worsened,
        "exact_two_sided_sign_test_p": sign_test(improved, worsened),
        "case_mode_pairs": pairs, "repetition_transition_counts": counts,
        "repetition_transitions": repetition_transitions,
    }


def per_case_quality(records, manifest):
    audits = {case["id"]: case["anchor_audit"] for case in manifest["cases"]}
    result = {}
    for case_id in sorted(audits):
        arms = {}
        for arm, budget in ARM_BUDGETS:
            rows = sorted(measured(records, arm, budget), key=lambda row: (row["mode"], row["repetition"]))
            rows = [row for row in rows if row["clip"] == case_id]
            if len(rows) != 6:
                raise ValueError(f"per-case measured population drift: {case_id}/{arm}/{budget}")
            arms[f"{arm}-{budget}"] = {
                "hits": sum(row["expected_answer_hit"] is True for row in rows), "total": 6,
                "answer_anchor_present": audits[case_id][f"{arm}-{budget}"]["answer_anchor_present"],
                "answers": [{"mode": row["mode"], "repetition": row["repetition"],
                             "request_id": row["response"]["request_id"],
                             "raw_answer": row["raw_answer"], "hit": row["expected_answer_hit"]}
                            for row in rows],
            }
        result[case_id] = {"arms": arms}
    return result


def every_miss(records, manifest):
    audits = {case["id"]: case["anchor_audit"] for case in manifest["cases"]}
    misses = []
    for row in sorted(measured(records), key=lambda item: (
            item["clip"], item["mode"], item["arm"], item["budget"], item["repetition"])):
        if row["expected_answer_hit"] is not False:
            continue
        misses.append({
            "case_id": row["clip"], "mode": row["mode"], "arm": row["arm"],
            "budget": row["budget"], "repetition": row["repetition"],
            "request_id": row["response"]["request_id"], "raw_answer": row["raw_answer"],
            "answer_anchor_present": audits[row["clip"]][f"{row['arm']}-{row['budget']}"][
                "answer_anchor_present"],
        })
    return misses


def latency_outliers(records):
    outliers = []
    for mode in MODES:
        for arm, budget in ARM_BUDGETS:
            rows = [row for row in measured(records, arm, budget) if row["mode"] == mode]
            if len(rows) != 30:
                raise ValueError("latency outlier population is incomplete")
            for metric in ("total", "native_elapsed"):
                values = sorted(row["latency_ms"][metric] for row in rows)
                # Frozen before traffic: inclusive linear-interpolation quartiles.
                q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
                fence = q3 + 1.5 * (q3 - q1)
                for row in rows:
                    value = row["latency_ms"][metric]
                    if value > fence:
                        outliers.append({
                            "case_id": row["clip"], "mode": mode, "arm": arm,
                            "budget": budget, "repetition": row["repetition"],
                            "request_id": row["response"]["request_id"], "metric": metric,
                            "value_ms": value, "q1_ms": q1, "q3_ms": q3,
                            "upper_fence_ms": fence, "retained_in_aggregates": True,
                        })
    return outliers


def quality_analysis(records, manifest):
    quality = {}
    for arm, budget in ARM_BUDGETS:
        hits = quality_count(records, arm, budget)
        quality[f"{arm}-{budget}"] = {
            "hits": hits, "total": 60, "misses": 60 - hits, "accuracy": hits / 60,
            "wilson_95": wilson(hits),
        }
    comparisons = {}
    for name, baseline, candidate in (
            ("primary_temporal8_vs_uniform8", ("uniform", 8), ("temporal", 8)),
            ("temporal4_vs_uniform4", ("uniform", 4), ("temporal", 4)),
            ("temporal16_vs_uniform16", ("uniform", 16), ("temporal", 16)),
            ("temporal8_vs_default8", ("default", 8), ("temporal", 8))):
        comparisons[name] = paired_effect(records, baseline, candidate)
    regressions = []
    for name, comparison in comparisons.items():
        for row in comparison["repetition_transitions"]:
            if row["transition"] == "hit_to_miss":
                regressions.append({"comparison": name, **row})
    return {
        "quality": quality, "comparisons": comparisons,
        "per_case": per_case_quality(records, manifest), "every_miss": every_miss(records, manifest),
        "quality_regressions_retained": regressions,
        "quality_regression_count": len(regressions),
        "runtime_change_in_response_to_answers": "FORBIDDEN_NONE_MADE",
    }


def public_provenance(value):
    return long_live.public_provenance(value)


def validate_committed_fixture(source_root, bundle, freeze_path):
    return clean.validate_committed_fixture(source_root, bundle, freeze_path)


def run_mode(args):
    if os.environ.get("UMMINFER_LIVE_OPT_IN") != "1":
        raise ValueError("live execution requires UMMINFER_LIVE_OPT_IN=1")
    if args.mode not in MODES:
        raise ValueError("invalid execution mode")
    manifest_path = Path(args.manifest).resolve()
    freeze_path = Path(args.freeze).resolve()
    # The committed-byte gate is authoritative before traffic. Full independent
    # decode is performed by temporal_heldout_verify.py in a fresh process; the
    # producer need not assume the fixture-generation FFmpeg major version.
    raw = load_manifest(manifest_path, freeze_path, deep=False)
    committed = validate_committed_fixture(args.source_root, manifest_path.parent, freeze_path)
    prepared = prepare(raw, manifest_path.parent)
    out = Path(args.out_dir).resolve()
    if out.exists():
        raise ValueError("refusing to overwrite held-out mode evidence")
    out.mkdir(parents=True)
    provenance_path = Path(args.provenance).resolve()
    provenance = long_live.validate_provenance(
        provenance_path, mode=args.mode, run_id=args.run_id, source_root=args.source_root)
    if provenance["physical_gpu"] != 1:
        raise ValueError("Task66 permits only physical GPU 1")
    url = long_live.loopback_url(args.url)
    long_live.health_probe(url)
    pre, _, _ = long_live.exclusive_check(
        provenance, clean.controlled.query_compute_apps, long_live.query_memory)
    pre.update({"mode": args.mode, "phase": "pre_schedule"})
    if not pre["passed"]:
        raise ValueError(f"GPU 1 is not exclusive before traffic: {pre['problems']}")
    shutil.copy2(provenance_path, out / "provenance.json")
    write_json(out / "pre-request.json", {
        "fixture_revision": fixture.REVISION,
        "preregistration_sha256": sha256_file(manifest_path.parent / "preregistration.json"),
        "manifest_sha256": sha256_file(manifest_path),
        "committed_frozen_fixture": committed, "model_requests_started": False,
        "pre_request_unix_ns": time.time_ns(), "mode": args.mode,
        "historical_task61_cases_in_denominator": [],
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
        load_manifest(manifest_path, freeze_path, deep=False)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        stale_errors.append(str(error))
    shutil.copy2(Path(args.native_log).resolve(), out / "native-stage.log")
    report = {
        "schema_version": 1, "fixture_revision": fixture.REVISION, "mode": args.mode,
        "run_id": args.run_id, "manifest_sha256": sha256_file(manifest_path),
        "preregistration_sha256": sha256_file(manifest_path.parent / "preregistration.json"),
        "provenance": public_provenance(provenance), "exclusive": [pre, post],
        "stale_errors": stale_errors, **result,
    }
    write_json(out / "mode-result.json", report)
    valid = sum(row["native_evidence_valid"] for row in result["records"])
    print(json.dumps({"mode": args.mode, "records": len(result["records"]),
                      "completed": sum(row["status"] == "COMPLETED" for row in result["records"]),
                      "native_valid": valid, "stale_errors": stale_errors}, indent=2))
    return 0 if len(result["records"]) == 280 and valid == 280 and not stale_errors else 2


def load_mode(root, mode):
    path = Path(root) / mode / "mode-result.json"
    value = json.loads(path.read_bytes())
    if value.get("schema_version") != 1 or value.get("fixture_revision") != fixture.REVISION \
            or value.get("mode") != mode or len(value["records"]) != 280:
        raise ValueError(f"incomplete held-out mode result: {mode}")
    cleanup = path.parent / "cleanup.json"
    value["post_stop_cleanup"] = json.loads(cleanup.read_bytes()) if cleanup.exists() else None
    for record in value["records"]:
        relative = Path("records") / mode / f"budget-{record['budget']}" \
            / record["phase"] / f"repetition-{record['repetition']}" / record["arm"] \
            / f"{record['clip']}.json"
        if json.loads((path.parent / relative).read_bytes()) != record:
            raise ValueError(f"raw held-out record differs from mode result: {record['clip']}")
        frames = json.loads((path.parent / relative.with_suffix(".sse.json")).read_bytes())
        if not isinstance(frames, list) or not frames:
            raise ValueError("missing raw held-out SSE evidence")
    return value


def technical_gate(records, parity, modes):
    reasons = []
    if len(records) != 560 or any(row["status"] != "COMPLETED" for row in records):
        reasons.append("560/560 completed requests required")
    if any(not row["native_evidence_valid"] for row in records):
        reasons.append("native RGB/source/PTS/temporal-2/mRoPE/cold-prefix evidence failed")
    if len(parity) != 500 or any(not row["passed"] for row in parity):
        reasons.append("500/500 default-uniform, serial-batch and repetition parity checks required")
    for mode in MODES:
        result = modes[mode]
        if result["stale_errors"] or any(not row["passed"] for row in result["exclusive"]) \
                or any(not row["passed"] for row in result["cleanup_checks"]):
            reasons.append(f"{mode}: exclusivity/staleness/cleanup failed")
        if not result["post_stop_cleanup"] or not result["post_stop_cleanup"].get("passed"):
            reasons.append(f"{mode}: post-stop cleanup is missing or failed")
        provenance = result["provenance"]
        if provenance["physical_gpu"] != 1 \
                or provenance["batch_size"] != (1 if mode == "serial" else 4) \
                or provenance["qwen38_prefix_cache"] != "disabled":
            reasons.append(f"{mode}: GPU/batch/prefix provenance failed")
    for field in ("source_commit", "source_state_sha256", "model", "qwen38_prefix_cache"):
        if public_provenance(modes["serial"]["provenance"][field]) != public_provenance(
                modes["batch4"]["provenance"][field]):
            reasons.append(f"serial/batch immutable identity differs: {field}")
    for field in ("gateway", "native_stage"):
        volatile = {"pid", "compute_pid", "compute_used_mib"}
        left = {k: v for k, v in public_provenance(modes["serial"]["provenance"][field]).items()
                if k not in volatile}
        right = {k: v for k, v in public_provenance(modes["batch4"]["provenance"][field]).items()
                 if k not in volatile}
        if left != right:
            reasons.append(f"serial/batch binary identity differs: {field}")
    return sorted(set(reasons))


def seal(root):
    root = Path(root)
    if not (root / "report.json").is_file():
        raise ValueError("cannot seal without report.json")
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in ("artifact-manifest.json", "SHA256SUMS"):
            entries.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                            "sha256": sha256_file(path)})
    write_json(root / "artifact-manifest.json", {"schema_version": 1, "hash": "sha256", "files": entries})
    (root / "SHA256SUMS").write_text("".join(
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
        for path in sorted(root.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"),
        encoding="ascii")


def finalize(args):
    root = Path(args.out_dir).resolve()
    if root.exists():
        raise ValueError("refusing to overwrite held-out evidence")
    manifest = load_manifest(args.manifest, args.freeze, deep=True)
    modes = {mode: load_mode(args.mode_root, mode) for mode in MODES}
    records = [record for mode in MODES for record in modes[mode]["records"]]
    parity = long_live.parity_checks(records, clean.parity_adapter(short))
    reasons = technical_gate(records, parity, modes)
    root.mkdir(parents=True)
    (root / "input").mkdir()
    shutil.copytree(Path(args.manifest).resolve().parent, root / "input" / "fixture")
    shutil.copy2(args.freeze, root / "input" / "freeze.json")
    if args.prefreeze_log:
        shutil.copy2(args.prefreeze_log, root / "input" / "prefreeze-failures.log")
    analysis_hashes = {}
    for relative in ANALYSIS_SOURCES:
        source = ROOT / relative
        target = root / "input" / "analysis" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        analysis_hashes[relative] = sha256_file(source)
    for mode in MODES:
        shutil.copytree(Path(args.mode_root) / mode, root / mode)
    analysis = quality_analysis(records, manifest)
    gate_pass = not reasons
    report = {
        "schema_version": 1, "experiment": "qwen38_temporal_heldout_quality_v1",
        "fixture_revision": fixture.REVISION,
        "evidence_class": "CONTROLLED_LIVE" if gate_pass else "UNCONTROLLED_LIVE",
        "manifest_sha256": sha256_file(args.manifest),
        "preregistration_sha256": sha256_file(Path(args.manifest).parent / "preregistration.json"),
        "freeze_receipt": fixture.validate_freeze(root / "input" / "fixture", root / "input" / "freeze.json"),
        "analysis_source_sha256": analysis_hashes,
        "records": records,
        "provenances": {mode: modes[mode]["provenance"] for mode in MODES},
        "steps": [step for mode in MODES for step in modes[mode]["steps"]],
        "resources": [row for mode in MODES for row in modes[mode]["resources"]],
        "cleanup_checks": [row for mode in MODES for row in modes[mode]["cleanup_checks"]],
        "exclusive": [row for mode in MODES for row in modes[mode]["exclusive"]],
        "post_stop_cleanup": {mode: modes[mode]["post_stop_cleanup"] for mode in MODES},
        "stale_errors": [error for mode in MODES for error in modes[mode]["stale_errors"]],
        "aggregates": clean.aggregate_results(records),
        "answer_outcomes": clean.answer_outcomes(records),
        "parity": parity, "analysis": analysis, "latency_outliers": latency_outliers(records),
        "gate_rules": list(manifest["analysis_plan"]["no_regression_technical_gates"]),
        "gate_reasons": reasons, "gate_pass": gate_pass,
        "evaluation_complete": gate_pass,
        "claims": {
            "real_generation": "MEASURED_PREREGISTERED_SYNTHETIC_HELDOUT",
            "live_latency": "CONTROLLED_LIVE" if gate_pass else "UNCONTROLLED_LIVE",
            "heldout_fixture_substring_accuracy": "MEASURED_PREREGISTERED_SYNTHETIC_HELDOUT",
            "natural_video_quality": "NOT_COVERED", "perceptual_quality": "NOT_RUN",
            "generalized_quality_claim": False,
        },
        "historical_controls": {
            "task61_case_ids": sorted(fixture.HISTORICAL_TASK61),
            "included_in_heldout_denominator": False, "scores_changed": False,
            "japanese_miss_and_aliases_changed": False,
        },
        "runtime_decision": "EVALUATION_ONLY_FINAL_RUNTIME_UNCHANGED_EVEN_IF_QUALITY_REGRESSIONS_EXIST",
    }
    write_json(root / "report.json", report)
    seal(root)
    print(json.dumps({
        "gate_pass": gate_pass, "gate_reasons": reasons, "records": len(records),
        "parity": len(parity), "quality": {key: value["hits"] for key, value in
                                          analysis["quality"].items()},
        "primary": {key: analysis["comparisons"]["primary_temporal8_vs_uniform8"][key]
                    for key in ("improved_pairs", "worsened_pairs", "exact_two_sided_sign_test_p")},
        "misses": len(analysis["every_miss"]), "latency_outliers": len(report["latency_outliers"]),
        "quality_regressions": analysis["quality_regression_count"],
        "sha256sums_sha256": sha256_file(root / "SHA256SUMS"),
    }, indent=2))
    return 0 if gate_pass else 2


def self_check():
    manifest = load_manifest(deep=False)
    assert len(manifest["cases"]) == 10
    assert not fixture.HISTORICAL_TASK61 & {case["id"] for case in manifest["cases"]}
    assert wilson(30) == wilson(30, 60)
    assert sign_test(2, 0) == 0.5
    assert sign_test(0, 0) == 1.0
    print("SELF-CHECK PASS: frozen ten-case population, GPU1 schedule, exact statistics, historical controls excluded")
    print("REAL_GENERATION: NOT_RUN")
    print("NATURAL_VIDEO_QUALITY: NOT_COVERED")
    print("PERCEPTUAL_QUALITY: NOT_RUN")
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-mode")
    for name in ("mode", "url", "native-log", "provenance", "source-root", "run-id", "out-dir"):
        run.add_argument(f"--{name}", required=True)
    run.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    run.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    final = sub.add_parser("finalize")
    for name in ("mode-root", "out-dir"):
        final.add_argument(f"--{name}", required=True)
    final.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    final.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    final.add_argument("--prefreeze-log", default="")
    sub.add_parser("self-check")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "self-check":
            return self_check()
        if args.command == "run-mode":
            return run_mode(args)
        return finalize(args)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"TEMPORAL HELD-OUT LIVE FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
