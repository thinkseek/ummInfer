#!/usr/bin/env python3
"""Independent stdlib-only verifier for the bounded NS-05 Before/After bundle."""
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

SEED = 510075
CASES = ("phone-action", "driver-scene-change")
METRICS = ("decode_prefill", "first_text", "first_voice", "public_total")
RUNS = (("baseline", "A", "baseline-A"), ("candidate", "B", "candidate-B"),
        ("candidate", "C", "candidate-C"), ("baseline", "D", "baseline-D"))
PARITY = ("policy", "decoded_frame_count", "selected_frame_count",
    "selected_candidate_indices", "selected_source_frame_indices", "source_indices_exact",
    "selected_frame_timestamps", "selected_frame_content_fnv1a64", "has_nominal_timestamps",
    "temporal2_pair_first_timestamps", "max_frames", "max_candidate_frames",
    "motion_threshold", "scene_threshold", "grid_t", "grid_h", "grid_w",
    "premerge_patch_rows", "merged_tokens_per_slice", "video_token_count",
    "prompt_token_count", "media_hash", "mrope_next", "mrope_fnv1a64")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def schedule():
    rng = random.Random(SEED)
    warm = [{"phase": "warmup", "case_id": case, "repetition": 0} for case in CASES]
    rng.shuffle(warm)
    measured = [{"phase": "measured", "case_id": case, "repetition": rep}
                for rep in range(8) for case in CASES]
    rng.shuffle(measured)
    result = warm + measured
    for index, row in enumerate(result):
        row["schedule_index"] = index
    return result


def canonical(source, case):
    url = "data:video/mp4;base64," + base64.b64encode(source).decode("ascii")
    body = {"model": "qwen3-native", "stream": True, "temperature": 0.0, "top_p": 1.0,
            "seed": SEED, "max_tokens": 64, "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": url}},
                {"type": "text", "text": case["question"]}]}],
            "modalities": ["text", "audio"],
            "audio": {"voice": "vivian", "format": "wav", "stream": True}}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def answer_hit(answer, case):
    folded = answer.casefold()
    return all(term.casefold() in folded for term in case["expect_all"]) and (
        not case["expect_any"] or any(term.casefold() in folded for term in case["expect_any"]))


def summary(values):
    ordered = sorted(values)
    med = statistics.median(ordered)
    q = statistics.quantiles(ordered, n=4, method="inclusive")
    mean = statistics.fmean(ordered)
    return {"count": len(ordered), "median_ms": med,
            "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1], "iqr_ms": q[2] - q[0],
            "mad_ms": statistics.median(abs(value - med) for value in ordered),
            "mean_ms": mean, "cv": statistics.pstdev(ordered) / mean if mean else 0.0,
            "min_ms": ordered[0], "max_ms": ordered[-1]}


def paired(deltas):
    med = statistics.median(deltas)
    rng = random.Random(SEED)
    boots = sorted(statistics.median(rng.choice(deltas) for _ in deltas) for _ in range(20000))
    positive, negative = sum(x > 0 for x in deltas), sum(x < 0 for x in deltas)
    n = positive + negative
    p = 1.0 if not n else min(1.0, 2.0 * sum(math.comb(n, k)
        for k in range(min(positive, negative) + 1)) / 2 ** n)
    return {"count": len(deltas), "median_delta_ms": med,
            "mean_delta_ms": statistics.fmean(deltas),
            "mad_delta_ms": statistics.median(abs(x - med) for x in deltas),
            "negative_pairs": negative, "positive_pairs": positive,
            "zero_pairs": len(deltas) - positive - negative, "exact_two_sided_sign_test_p": p,
            "bootstrap_95_median_delta_ms": [boots[math.floor(0.025 * (len(boots) - 1))],
                boots[math.ceil(0.975 * (len(boots) - 1))]], "deltas_ms": deltas}


def verify_pcm(frames):
    groups = {}
    for frame in frames:
        unified = frame.get("unified", {})
        if unified.get("kind") != "artifact":
            continue
        artifact = unified.get("artifact", {})
        if artifact.get("mime_type") != "audio/pcm":
            continue
        groups.setdefault(artifact["artifact_id"], []).append(artifact)
    raw_parts = []
    for artifact_id in sorted(groups):
        chunks = groups[artifact_id]
        require([x["index"] for x in chunks] == list(range(len(chunks))), "PCM order")
        raw = b"".join(base64.b64decode(x["data"], validate=True) for x in chunks)
        require(chunks[-1]["bytes"] == len(raw) and chunks[-1]["sha256"] ==
                hashlib.sha256(raw).hexdigest(), "PCM final hash")
        raw_parts.append(raw)
    joined = b"".join(raw_parts)
    return len(joined), hashlib.sha256(joined).hexdigest(), len(groups)


def verify_runs(root, prereg):
    source = (root / "source/ns05-driver-30-150s.mp4").read_bytes()
    require(hashlib.sha256(source).hexdigest() == prereg["source"]["derivative_sha256"], "media")
    require(sha(root / "source/LICENSE") == prereg["source"]["license_sha256"], "license")
    require(hashlib.sha256((root / "source/upstream-driver-action-recognition.mp4").read_bytes()).hexdigest()
            == prereg["source"]["upstream_sha256"], "upstream media")
    expected_schedule = schedule()
    expected_requests = {case["id"]: canonical(source, case) for case in prereg["cases"]}
    reports, rows = [], []
    for implementation, block, name in RUNS:
        run = load(root / "live" / name / "mode-result.json")
        require((run["implementation"], run["block"]) == (implementation, block), "ABBA")
        require(run["source_commit"] == ("df563436b3255fec7aed955c4ff18fe06ef0c706"
            if implementation == "baseline" else "c2b383561236151e347516a701093f2f194e51c3"), "commit")
        require(run["post_stop_cleanup"]["passed"] and
                not run["post_stop_cleanup"]["post_stop"]["compute_apps"] and
                run["post_stop_cleanup"]["post_stop"]["memory_used_mib"] <= 16, "cleanup")
        require(run["binary"]["gateway_sha256"] == sha(root / "binaries/gateway"), "gateway")
        require(run["binary"]["vlm_sha256"] == sha(root / "binaries" / f"vlm-{implementation}"), "VLM")
        require(run["binary"]["asr_sha256"] == sha(root / "binaries/asr"), "ASR")
        require(run["binary"]["tts_sha256"] == sha(root / "binaries/tts"), "TTS")
        require(len(run["records"]) == 18, "18 records")
        for row, expected in zip(run["records"], expected_schedule):
            for key in ("phase", "case_id", "repetition", "schedule_index"):
                require(row[key] == expected[key], "schedule")
            relative = Path("records") / f"schedule-{row['schedule_index']:03d}.json"
            require(load(root / "live" / name / relative) == row, "raw record")
            frames = load(root / "live" / name / relative.with_suffix(".frames.json"))
            response = row["response"]
            answer = "".join(choice.get("delta", {}).get("content", "")
                for frame in frames for choice in frame.get("choices", [])
                if isinstance(choice.get("delta", {}).get("content"), str))
            require(answer == response["answer"] and response["answer_sha256"] ==
                    hashlib.sha256(answer.encode()).hexdigest(), "answer join")
            require(row["canonical_request_sha256"] == expected_requests[row["case_id"]], "request")
            require(verify_pcm(frames) == (response["pcm_bytes"], response["pcm_sha256"],
                    response["pcm_artifact_count"]), "PCM reconstruction")
            case = next(case for case in prereg["cases"] if case["id"] == row["case_id"])
            require(row["expected_answer_hit"] == answer_hit(answer, case), "quality scoring")
            native, execution = row["native_evidence"], row["native_execution"]
            require(native["max_frames"] == 8 and native["selected_frame_count"] == 8
                and not native["source_indices_exact"] and native["has_nominal_timestamps"]
                and execution["prefix_reused"] == 0, "default/cold native")
            calculated = {"decode_prefill": execution["video_input_ms"] + execution["vision_ms"]
                + execution["prefill_ms"], "first_text": response["first_text_s"] * 1000,
                "first_voice": response["first_voice_s"] * 1000,
                "public_total": response["total_s"] * 1000}
            require(row["latency_ms"] == calculated, "latency projection")
            rows.append({"implementation": implementation, "block": block,
                         "run_id": run["run_id"], **row})
        reports.append(run)
    return reports, rows


def verify_analysis(reports, rows):
    pairs = []
    for left, right in (("A", "B"), ("D", "C")):
        baseline = next(run for run in reports if run["block"] == left)
        candidate = next(run for run in reports if run["block"] == right)
        for brow in baseline["records"]:
            crow = next(row for row in candidate["records"] if row["schedule_index"] == brow["schedule_index"])
            pair = {"baseline_block": left, "candidate_block": right,
                "schedule_index": brow["schedule_index"], "case_id": brow["case_id"],
                "phase": brow["phase"], "repetition": brow["repetition"],
                "answer_exact": brow["response"]["answer"] == crow["response"]["answer"],
                "native_exact": all(brow["native_evidence"][field] == crow["native_evidence"][field]
                                    for field in PARITY),
                "pcm_exact": brow["response"]["pcm_sha256"] == crow["response"]["pcm_sha256"]
                    and brow["response"]["pcm_bytes"] == crow["response"]["pcm_bytes"]}
            pair.update({metric + "_delta_ms": crow["latency_ms"][metric] - brow["latency_ms"][metric]
                         for metric in METRICS})
            pairs.append(pair)
    require(len(pairs) == 36 and all(row["answer_exact"] and row["native_exact"] and row["pcm_exact"]
                                     for row in pairs), "36 exact pairs")
    distributions, comparisons, quality = {}, {}, {}
    for implementation in ("baseline", "candidate"):
        for case in CASES:
            selected = [row for row in rows if row["phase"] == "measured"
                        and row["implementation"] == implementation and row["case_id"] == case]
            key = f"{implementation}/{case}"
            distributions[key] = {metric: summary([row["latency_ms"][metric] for row in selected])
                                  for metric in METRICS}
            hits = sum(row["expected_answer_hit"] for row in selected)
            quality[key] = {"hits": hits, "total": 16, "misses": 16 - hits,
                "missed_answers": [row["response"]["answer"] for row in selected
                                   if not row["expected_answer_hit"]]}
    for case in CASES:
        for metric in METRICS:
            comparisons[f"{case}/{metric}"] = paired([row[metric + "_delta_ms"]
                for row in pairs if row["case_id"] == case and row["phase"] == "measured"])
    return pairs, distributions, comparisons, quality


def verify_natural(root):
    baseline = load(root / "natural/task72-baseline-report.json")
    candidate = load(root / "natural/candidate/report.json")
    receipt = load(root / "natural/verification-local.json")
    require(receipt["result"] == "PASS" and candidate["gate_pass"], "natural gate")
    key = lambda row: (row["mode"], row["phase"], row["arm"], row["budget"],
                       row["clip"], row["repetition"])
    before = {key(row): row for row in baseline["records"]}
    require(len(before) == 320 and len(candidate["records"]) == 320, "natural population")
    for row in candidate["records"]:
        other = before[key(row)]
        require(row["raw_answer"] == other["raw_answer"] and
                all(row["native"][field] == other["native"][field] for field in PARITY),
                "Task72 baseline/candidate mismatch")
    return {"answers_exact": 320, "native_exact": 320,
            "report_sha256": sha(root / "natural/candidate/report.json"),
            "verification_sha256": sha(root / "natural/verification-local.json")}


def verify_manifest(root):
    manifest_path, sums_path = root / "artifact-manifest.json", root / "SHA256SUMS"
    manifest = load(manifest_path)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    expected = sorted(path for path in files if path not in (manifest_path, sums_path))
    require([root / row["path"] for row in manifest["files"]] == expected, "manifest complete")
    for row in manifest["files"]:
        path = root / row["path"]
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "manifest hash")
    sums = dict(line.split("  ", 1)[::-1] for line in sums_path.read_text().splitlines())
    require(sums == {path.relative_to(root).as_posix(): sha(path)
                     for path in files if path != sums_path}, "SHA256SUMS")
    return len(manifest["files"])


def verify(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(root not in output.parents, "output must be external")
    prereg = load(root / "preregistration.json")
    reports, rows = verify_runs(root, prereg)
    pairs, distributions, comparisons, quality = verify_analysis(reports, rows)
    produced = load(root / "report.json")
    require(produced["pairs"] == pairs and produced["distributions"] == distributions
            and produced["paired_comparisons"] == comparisons and produced["quality"] == quality,
            "producer analysis mismatch")
    threshold_failures = []
    for key, metrics in distributions.items():
        for metric in ("first_text", "first_voice"):
            if metrics[metric]["p95_ms"] >= 3000:
                threshold_failures.append({"cell": key, "metric": metric})
    require(not threshold_failures, "NS05 3s threshold")
    primary = [comparisons[f"{case}/first_voice"] for case in CASES]
    promotion = all(row["median_delta_ms"] < 0 and row["bootstrap_95_median_delta_ms"][1] < 0
        and row["exact_two_sided_sign_test_p"] <= 0.05 for row in primary) \
        and not produced["regressions_over_1_percent"]
    require(promotion == produced["candidate_promotion"] and not promotion
            and produced["candidate_disposition"] == "NOT_PROMOTED", "promotion decision")
    natural = verify_natural(root)
    files = verify_manifest(root)
    result = {"schema_version": 1, "result": "PASS", "requests": len(rows),
        "exact_pairs": len(pairs), "quality": quality, "outliers_retained": len(produced["outliers"]),
        "first_text_p95_max_ms": max(x["first_text"]["p95_ms"] for x in distributions.values()),
        "first_voice_p95_max_ms": max(x["first_voice"]["p95_ms"] for x in distributions.values()),
        "candidate_promotion": False, "natural": natural, "manifest_files_verified": files,
        "manifest_sha256": sha(root / "artifact-manifest.json"),
        "sha256sums_sha256": sha(root / "SHA256SUMS"), "report_sha256": sha(root / "report.json"),
        "physical_playback": "NOT_RUN", "perceptual_quality": "NOT_RUN",
        "barge_in_and_cross_modal_generation": "NOT_RUN"}
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        return verify(args.evidence_root, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"NS05 INDEPENDENT VERIFY FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
