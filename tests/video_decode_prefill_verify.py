#!/usr/bin/env python3
"""Independent stdlib-only raw verifier for the Task75 natural throughput gate.

No producer, transport, fixture-generator, or scoring module is imported. The
output must be outside the evidence root; the root manifest self-excludes only
its own manifest/checksum files and covers every producer input and raw result.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

ARMS = ("default-8", "uniform-16", "temporal-16")
SEED = 510075
SOURCE_SHA = "81178f8af6c859d908cd6fa4a5348ed9bbe271c071a94a196ff2488c63f0db2e"
QUESTION = "Describe the visible natural scenes and their order in one concise English sentence."
PARITY = ("policy", "decoded_frame_count", "selected_frame_count",
    "selected_candidate_indices", "selected_source_frame_indices", "source_indices_exact",
    "selected_frame_timestamps", "selected_frame_content_fnv1a64", "has_nominal_timestamps",
    "temporal2_pair_first_timestamps", "max_frames", "max_candidate_frames",
    "motion_threshold", "scene_threshold", "grid_t", "grid_h", "grid_w",
    "premerge_patch_rows", "merged_tokens_per_slice", "video_token_count",
    "prompt_token_count", "media_hash", "mrope_next", "mrope_fnv1a64")
METRICS = ("total", "ttft", "native_elapsed", "video_input", "prompt_prep", "vision",
           "prefill", "decode", "unattributed")
RUNS = (("serial", "A", "baseline", "serial-A-baseline"),
        ("serial", "B", "candidate", "serial-B-candidate"),
        ("serial", "C", "candidate", "serial-C-candidate"),
        ("serial", "D", "baseline", "serial-D-baseline"),
        ("batch4", "A", "baseline", "batch4-A-baseline"),
        ("batch4", "B", "candidate", "batch4-B-candidate"),
        ("batch4", "C", "candidate", "batch4-C-candidate"),
        ("batch4", "D", "baseline", "batch4-D-baseline"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def dump(value):
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def make_schedule():
    rng = random.Random(SEED)
    warm = [{"phase": "warmup", "arm": arm, "repetition": 0, "padding": False}
            for arm in ARMS]
    warm.append({"phase": "warmup", "arm": "default-8", "repetition": 1, "padding": True})
    rng.shuffle(warm)
    measured = [{"phase": "measured", "arm": arm, "repetition": rep, "padding": False}
                for rep in range(8) for arm in ARMS]
    rng.shuffle(measured)
    result = warm + measured
    for index, row in enumerate(result):
        row["schedule_index"] = index
    return result


def request(source_bytes, arm):
    video = {"url": "data:video/mp4;base64," + base64.b64encode(source_bytes).decode("ascii")}
    if arm == "uniform-16":
        video["frame_selection"] = {"policy": "uniform", "max_frames": 16}
    elif arm == "temporal-16":
        video["frame_selection"] = {"policy": "temporal", "motion_threshold": 0.02,
                                    "scene_threshold": 0.25, "max_frames": 16}
    body = {"model": "qwen3-native", "stream": True, "temperature": 0.0, "top_p": 1.0,
            "seed": SEED, "max_tokens": 32, "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": video},
                {"type": "text", "text": QUESTION}]}]}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def p95(ordered):
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def distribution(values):
    ordered = sorted(values)
    med = statistics.median(ordered)
    quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
    mean = statistics.fmean(ordered)
    return {"count": len(ordered), "median_ms": med, "p95_ms": p95(ordered),
            "iqr_ms": quartiles[2] - quartiles[0],
            "mad_ms": statistics.median(abs(value - med) for value in ordered),
            "mean_ms": mean, "cv": statistics.pstdev(ordered) / mean if mean else 0.0,
            "min_ms": ordered[0], "max_ms": ordered[-1]}


def exact_sign(positive, negative):
    count = positive + negative
    if count == 0:
        return 1.0
    smaller = min(positive, negative)
    return min(1.0, 2.0 * sum(math.comb(count, k) for k in range(smaller + 1)) / 2 ** count)


def paired(deltas):
    med = statistics.median(deltas)
    rng = random.Random(SEED)
    sampled = sorted(statistics.median(rng.choice(deltas) for _ in deltas)
                     for _ in range(20000))
    positive = sum(delta > 0 for delta in deltas)
    negative = sum(delta < 0 for delta in deltas)
    return {"count": len(deltas), "median_delta_ms": med,
            "mean_delta_ms": statistics.fmean(deltas),
            "mad_delta_ms": statistics.median(abs(delta - med) for delta in deltas),
            "negative_pairs": negative, "positive_pairs": positive,
            "zero_pairs": len(deltas) - positive - negative,
            "exact_two_sided_sign_test_p": exact_sign(positive, negative),
            "bootstrap_95_median_delta_ms": [sampled[math.floor(0.025 * (len(sampled) - 1))],
                sampled[math.ceil(0.975 * (len(sampled) - 1))]], "deltas_ms": deltas}


def verify_runs(root):
    source_bytes = (root / "source/natural-full-330.mp4").read_bytes()
    require(hashlib.sha256(source_bytes).hexdigest() == SOURCE_SHA, "source bytes")
    expected_schedule = make_schedule()
    reports = []
    rows = []
    for mode, block, implementation, name in RUNS:
        run_dir = root / "live" / name
        report = load(run_dir / "mode-result.json")
        require((report["mode"], report["block"], report["implementation"]) ==
                (mode, block, implementation), "run topology")
        require(report["physical_gpu"] == 1 and report["prefix_cache"] == "disabled",
                "GPU/prefix contract")
        require(report["post_stop_cleanup"]["passed"] and
                not report["post_stop_cleanup"]["post_stop"]["compute_apps"] and
                report["post_stop_cleanup"]["post_stop"]["memory_used_mib"] <= 16,
                "owned process cleanup")
        require(report["source_sha256"] == SOURCE_SHA, "source identity")
        require(report["source_commit"] == ("03b4ab65ea861491374c00954b06269f9baadc4c"
                if implementation == "baseline" else
                "1ff89b764962d270613d42344b9879eab122c006"), "execution commit")
        require(report["binary"]["gateway_sha256"] == digest(root / "binaries/gateway"),
                "gateway binary")
        require(report["binary"]["native_stage_sha256"] == digest(
            root / "binaries" / f"native-{implementation}"), "native binary")
        require(report["model"]["config_sha256"] ==
                load(root / "model-manifest.json")["files"]["config.json"], "model config")
        records = report["records"]
        require(len(records) == 28, "28 requests per stack")
        for row, expected in zip(records, expected_schedule):
            for field in ("phase", "arm", "repetition", "padding", "schedule_index"):
                require(row[field] == expected[field], "seeded schedule")
            relative = Path("records") / f"schedule-{row['schedule_index']:03d}.json"
            require(load(run_dir / relative) == row, "raw request record")
            frames = load(run_dir / relative.with_suffix(".sse.json"))
            require(isinstance(frames, list) and frames, "raw SSE frames")
            response = row["response"]
            joined = "".join(choice.get("delta", {}).get("content", "")
                for frame in frames for choice in frame.get("choices", [])
                if isinstance(choice.get("delta", {}).get("content"), str))
            require(joined == response["answer"] and response["answer_sha256"] ==
                    hashlib.sha256(joined.encode()).hexdigest(), "SSE answer join")
            require(response["finish_reason"] in ("stop", "length"), "terminal")
            require(row["canonical_request_sha256"] == request(source_bytes, row["arm"]),
                    "canonical request")
            native = row["native_evidence"]
            execution = row["native_execution"]
            budget = 8 if row["arm"] == "default-8" else 16
            require(native["selected_frame_count"] == budget and native["max_frames"] == budget,
                    "native budget")
            require(native["engine_path"] == ("serial" if mode == "serial" else "batched"),
                    "engine path")
            require(execution["prefix_reused"] == 0 and execution["unattributed_ms"] >= -0.001,
                    "cold prefix/timing")
            calculated = {"total": response["total_s"] * 1000, "ttft": response["ttft_s"] * 1000,
                "native_elapsed": execution["elapsed_ms"], "video_input": execution["video_input_ms"],
                "prompt_prep": execution["prompt_prep_ms"], "vision": execution["vision_ms"],
                "prefill": execution["prefill_ms"], "decode": execution["decode_ms"],
                "unattributed": execution["unattributed_ms"]}
            require(row["latency_ms"] == calculated, "latency projection")
            rows.append({"implementation": implementation, "mode": mode, "block": block,
                         "run_id": report["run_id"], **row})
        reports.append(report)
    return reports, rows


def independent_analysis(reports, rows):
    pairs = []
    for mode in ("serial", "batch4"):
        for bblock, cblock in (("A", "B"), ("D", "C")):
            baseline = next(run for run in reports if run["mode"] == mode and run["block"] == bblock)
            candidate = next(run for run in reports if run["mode"] == mode and run["block"] == cblock)
            for brow in baseline["records"]:
                crow = next(row for row in candidate["records"]
                            if row["schedule_index"] == brow["schedule_index"])
                answer = brow["response"]["answer"] == crow["response"]["answer"]
                native = all(brow["native_evidence"][field] == crow["native_evidence"][field]
                             for field in PARITY)
                pairs.append({"mode": mode, "baseline_block": bblock, "candidate_block": cblock,
                    "schedule_index": brow["schedule_index"], "phase": brow["phase"],
                    "arm": brow["arm"], "repetition": brow["repetition"], "answer_exact": answer,
                    "native_semantics_exact": native,
                    "total_delta_ms": crow["latency_ms"]["total"] - brow["latency_ms"]["total"]})
    require(len(pairs) == 112 and all(row["answer_exact"] and row["native_semantics_exact"]
                                      for row in pairs), "112 exact pairs")
    distributions = {}
    comparisons = {}
    for mode in ("serial", "batch4"):
        for arm in ARMS:
            for implementation in ("baseline", "candidate"):
                selected = [row for row in rows if row["phase"] == "measured"
                    and row["mode"] == mode and row["implementation"] == implementation
                    and row["arm"] == arm]
                require(len(selected) == 16, "16 measured per cell")
                distributions[f"{mode}/{implementation}/{arm}"] = {
                    metric: distribution([row["latency_ms"][metric] for row in selected])
                    for metric in METRICS}
            comparisons[f"{mode}/{arm}"] = paired([row["total_delta_ms"] for row in pairs
                if row["mode"] == mode and row["arm"] == arm and row["phase"] == "measured"])
    outlier_rows = []
    for mode in ("serial", "batch4"):
        for implementation in ("baseline", "candidate"):
            for arm in ARMS:
                selected = [row for row in rows if row["phase"] == "measured"
                    and row["mode"] == mode and row["implementation"] == implementation
                    and row["arm"] == arm]
                for metric in ("total", "native_elapsed"):
                    values = [row["latency_ms"][metric] for row in selected]
                    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
                    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
                    for row in selected:
                        value = row["latency_ms"][metric]
                        if value < low or value > high:
                            outlier_rows.append({"run_id": row["run_id"],
                                "schedule_index": row["schedule_index"],
                                "request_id": row["response"]["request_id"], "mode": mode,
                                "implementation": implementation, "arm": arm, "metric": metric,
                                "value_ms": value, "lower_fence_ms": low, "upper_fence_ms": high,
                                "retained": True})
    return pairs, distributions, comparisons, outlier_rows


def verify_host_profiles(root):
    checked = 0
    diagnostic_variations = []
    for source in ("natural-full-330", "natural-01", "natural-03", "natural-06"):
        for implementation in ("baseline", "candidate"):
            profile = load(root / "profile" / implementation / f"{source}.json")
            require(profile["seed"] == SEED and profile["measured_repetitions"] == 9,
                    "host profile protocol")
            require(len(profile["records"]) == 63, "63 host measurements")
            for arm in {row["arm"] for row in profile["records"]}:
                selected = [row for row in profile["records"] if row["arm"] == arm]
                # This flag also records missing PTS on *unselected* candidates.
                # Require every selected/model-affecting field to remain exact and
                # retain any diagnostic-only flag/digest variation explicitly.
                material = {json.dumps({key: value for key, value in row["semantics"].items()
                    if key not in ("has_nominal_timestamps", "semantic_fnv1a64")}, sort_keys=True)
                    for row in selected}
                require(len(material) == 1, "host model-affecting semantic parity")
                first = selected[0]["semantics"]
                for row in selected[1:]:
                    changed = [key for key in first if first[key] != row["semantics"][key]]
                    if changed:
                        require(set(changed) <= {"has_nominal_timestamps", "semantic_fnv1a64"},
                                "unexpected host semantic variation")
                        diagnostic_variations.append({"source": source,
                            "implementation": implementation, "arm": arm,
                            "repetition": row["repetition"], "fields": changed,
                            "selected_rgb_pts_indices_unchanged": True})
        for semantic in sorted((root / "profile/baseline/semantics" / source).glob("*.bin")):
            other = root / "profile/candidate/semantics" / source / semantic.name
            require(other.is_file() and semantic.read_bytes() == other.read_bytes(),
                    "exact host semantic binary")
            checked += 1
        require(checked >= 7, "host semantic population")
    require(checked == 28, "28 exact host semantic binaries")
    return checked, diagnostic_variations


def verify_natural(root):
    baseline = load(root / "natural/task72-baseline-report.json")
    candidate = load(root / "natural/candidate/report.json")
    receipt = load(root / "natural/candidate-verification-local.json")
    require(receipt["result"] == "PASS" and candidate["gate_pass"], "natural independent gate")
    key = lambda row: (row["mode"], row["phase"], row["arm"], row["budget"],
                       row["clip"], row["repetition"])
    baseline_rows = {key(row): row for row in baseline["records"]}
    require(len(baseline_rows) == 320 and len(candidate["records"]) == 320,
            "natural population")
    for row in candidate["records"]:
        other = baseline_rows[key(row)]
        require(row["raw_answer"] == other["raw_answer"], "natural baseline answer")
        require(all(row["native"][field] == other["native"][field] for field in PARITY),
                "natural baseline native semantics")
    return {"answers_exact": 320, "native_semantics_exact": 320,
            "quality": candidate["analysis"]["quality"],
            "misses": len(candidate["analysis"]["every_miss"]),
            "latency_outliers": len(candidate["latency_outliers"]),
            "report_sha256": digest(root / "natural/candidate/report.json")}


def verify_manifest(root):
    manifest_path = root / "artifact-manifest.json"
    sums_path = root / "SHA256SUMS"
    manifest = load(manifest_path)
    entries = manifest["files"]
    actual = sorted(path for path in root.rglob("*") if path.is_file())
    expected = sorted(path for path in actual if path not in (manifest_path, sums_path))
    require([root / entry["path"] for entry in entries] == expected,
            "manifest completeness/self-exclusion")
    for entry in entries:
        path = root / entry["path"]
        require(path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"],
                f"manifest hash: {entry['path']}")
    sums = {}
    for line in sums_path.read_text(encoding="ascii").splitlines():
        sha, relative = line.split("  ", 1)
        sums[relative] = sha
    expected_sums = {path.relative_to(root).as_posix(): digest(path)
                     for path in actual if path != sums_path}
    require(sums == expected_sums, "SHA256SUMS completeness")
    return len(entries)


def verify(root, output):
    root = Path(root).resolve()
    output = Path(output).resolve()
    require(root != output and root not in output.parents, "verification output must be external")
    prereg = load(root / "preregistration.json")
    require(prereg["live_workload"]["requests_total"] == 224, "preregistered population")
    reports, rows = verify_runs(root)
    pairs, distributions, comparisons, outlier_rows = independent_analysis(reports, rows)
    produced = load(root / "throughput-report.json")
    require(produced["paired_requests"] == pairs, "producer paired rows")
    require(produced["distributions"] == distributions, "producer distributions")
    require(produced["paired_comparisons"] == comparisons, "producer paired statistics")
    require(produced["outliers"] == outlier_rows, "producer outliers")
    base = distributions["serial/baseline/temporal-16"]["total"]
    candidate = distributions["serial/candidate/temporal-16"]["total"]
    primary = comparisons["serial/temporal-16"]
    delta = primary["median_delta_ms"]
    relative = delta / base["median_ms"]
    primary_pass = (delta <= -25 and relative <= -0.05
        and primary["bootstrap_95_median_delta_ms"][1] < 0
        and primary["exact_two_sided_sign_test_p"] <= 0.05
        and base["cv"] <= 0.10 and candidate["cv"] <= 0.10)
    require(produced["primary"]["passed"] == primary_pass, "primary acceptance")
    host_checked, host_diagnostic_variations = verify_host_profiles(root)
    natural = verify_natural(root)
    file_count = verify_manifest(root)
    result = {"schema_version": 1, "result": "PASS",
        "independent": "stdlib_only_no_producer_transport_or_scoring_import",
        "requests": len(rows), "exact_paired_requests": len(pairs),
        "host_semantic_binaries_exact": host_checked,
        "host_diagnostic_only_variations": host_diagnostic_variations, "natural": natural,
        "primary": {"baseline_median_ms": base["median_ms"],
                    "candidate_median_ms": candidate["median_ms"],
                    "paired_delta_ms": delta, "relative_delta": relative,
                    "sign_test_p": primary["exact_two_sided_sign_test_p"],
                    "bootstrap_95": primary["bootstrap_95_median_delta_ms"],
                    "five_percent_acceptance_passed": primary_pass},
        "video_input_median_ms": {
            "baseline": distributions["serial/baseline/temporal-16"]["video_input"]["median_ms"],
            "candidate": distributions["serial/candidate/temporal-16"]["video_input"]["median_ms"]},
        "outliers_retained": len(outlier_rows), "manifest_files_verified": file_count,
        "manifest_sha256": digest(root / "artifact-manifest.json"),
        "sha256sums_sha256": digest(root / "SHA256SUMS"),
        "throughput_report_sha256": digest(root / "throughput-report.json"),
        "claims": {"output_and_native_semantics_parity": True,
                   "five_percent_public_total_improvement": primary_pass,
                   "generalized_quality": "NOT_COVERED", "perceptual_quality": "NOT_RUN",
                   "isolated_h2d_transfer": "NOT_COVERED"}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(dump(result), encoding="utf-8")
    print(dump(result), end="")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        return verify(args.evidence_root, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"INDEPENDENT VERIFY FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
