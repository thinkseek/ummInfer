#!/usr/bin/env python3
"""Independent fresh-process verifier for the Task #88 NS-05 paired bundle."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import statistics

SEED = 510075
CASES = ("phone-action", "driver-scene-change")
METRICS = ("decode_prefill", "first_text", "first_voice", "public_total")
RUNS = (("baseline", "A", "e084d8404b1d96bee5cd6f83d8299c2498a3af65"),
        ("candidate", "B", "feb8e97"), ("candidate", "C", "feb8e97"),
        ("baseline", "D", "e084d8404b1d96bee5cd6f83d8299c2498a3af65"))
PARITY = ("policy", "decoded_frame_count", "selected_frame_count",
    "selected_candidate_indices", "selected_source_frame_indices", "source_indices_exact",
    "selected_frame_timestamps", "selected_frame_content_fnv1a64", "has_nominal_timestamps",
    "temporal2_pair_first_timestamps", "max_frames", "max_candidate_frames",
    "motion_threshold", "scene_threshold", "grid_t", "grid_h", "grid_w",
    "premerge_patch_rows", "merged_tokens_per_slice", "video_token_count",
    "prompt_token_count", "media_hash", "mrope_next", "mrope_fnv1a64")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def schedule():
    rng = random.Random(SEED)
    rows = [{"phase": "warmup", "case_id": case, "repetition": 0} for case in CASES]
    rng.shuffle(rows)
    measured = [{"phase": "measured", "case_id": case, "repetition": rep}
                for rep in range(8) for case in CASES]
    rng.shuffle(measured)
    rows += measured
    for index, row in enumerate(rows):
        row["schedule_index"] = index
    return rows


def hit(answer, case):
    folded = answer.casefold()
    return all(term.casefold() in folded for term in case["expect_all"]) and (
        not case["expect_any"] or any(term.casefold() in folded for term in case["expect_any"]))


def summary(values):
    ordered = sorted(values)
    median = statistics.median(ordered)
    quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
    mean = statistics.fmean(ordered)
    return {"count": len(ordered), "median_ms": median,
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "iqr_ms": quartiles[2] - quartiles[0],
        "mad_ms": statistics.median(abs(value - median) for value in ordered),
        "mean_ms": mean, "cv": statistics.pstdev(ordered) / mean if mean else 0.0,
        "min_ms": ordered[0], "max_ms": ordered[-1]}


def pcm(frames):
    groups = {}
    for frame in frames:
        unified = frame.get("unified", {})
        if unified.get("kind") != "artifact":
            continue
        artifact = unified.get("artifact", {})
        if artifact.get("mime_type") == "audio/pcm":
            groups.setdefault(artifact["artifact_id"], []).append(artifact)
    joined = []
    for artifact_id in sorted(groups):
        rows = groups[artifact_id]
        require([row["index"] for row in rows] == list(range(len(rows))), "PCM order")
        raw = b"".join(base64.b64decode(row["data"], validate=True) for row in rows)
        require(rows[-1]["bytes"] == len(raw) and rows[-1]["sha256"] ==
                hashlib.sha256(raw).hexdigest(), "PCM hash")
        joined.append(raw)
    raw = b"".join(joined)
    return len(raw), hashlib.sha256(raw).hexdigest(), len(groups)


def verify(root):
    root = Path(root).resolve()
    prereg = load(root / "preregistration.json")
    require(prereg["experiment"] == "umminfer_ns05_duration_budget_voice_cocreation_v1", "protocol")
    require(sha(root / "source/ns05-driver-30-150s.mp4") == prereg["source"]["derivative_sha256"], "source")
    require(sha(root / "source/LICENSE") == prereg["source"]["license_sha256"], "license")
    expected_schedule = schedule()
    reports, rows = [], []
    for implementation, block, commit in RUNS:
        run_dir = root / "live" / f"{implementation}-{block}"
        report = load(run_dir / "mode-result.json")
        require(report["implementation"] == implementation and report["block"] == block, "ABBA")
        require(report["source_commit"].startswith(commit), f"execution commit {implementation}")
        require(report["physical_gpu"] == 0 and report["post_stop_cleanup"]["passed"], "GPU/cleanup")
        require(not report["post_stop_cleanup"]["post_stop"]["compute_apps"] and
                report["post_stop_cleanup"]["post_stop"]["memory_used_mib"] <= 16, "final GPU 0")
        require(report["binary"]["gateway_sha256"] == sha(root / "binaries/gateway"), "gateway hash")
        require(report["binary"]["vlm_sha256"] == sha(root / "binaries" / f"vlm-{implementation}"), "VLM hash")
        require(report["binary"]["asr_sha256"] == sha(root / "binaries/asr"), "ASR hash")
        require(report["binary"]["tts_sha256"] == sha(root / "binaries/tts"), "TTS hash")
        require(len(report["records"]) == 18, "18 requests")
        for row, expected in zip(report["records"], expected_schedule):
            require(all(row[key] == expected[key] for key in
                ("phase", "case_id", "repetition", "schedule_index")), "schedule")
            relative = Path("records") / f"schedule-{row['schedule_index']:03d}.json"
            require(load(run_dir / relative) == row, "raw record")
            frames = load(run_dir / relative.with_suffix(".frames.json"))
            response = row["response"]
            answer = "".join(choice.get("delta", {}).get("content", "")
                for frame in frames for choice in frame.get("choices", [])
                if isinstance(choice.get("delta", {}).get("content"), str))
            require(answer == response["answer"] and response["answer_sha256"] ==
                    hashlib.sha256(answer.encode()).hexdigest(), "raw answer")
            require(pcm(frames) == (response["pcm_bytes"], response["pcm_sha256"],
                    response["pcm_artifact_count"]), "raw PCM")
            case = next(case for case in prereg["frozen_comprehension_cases"]
                        if case["id"] == row["case_id"])
            require(row["expected_answer_hit"] == hit(answer, case), "frozen score")
            native, execution = row["native_evidence"], row["native_execution"]
            require(execution["prefix_reused"] == 0, "cold prefix")
            if implementation == "baseline":
                require(native["max_frames"] == 8 and native["selected_frame_count"] == 8
                    and not native["source_indices_exact"] and native["has_nominal_timestamps"], "Before eight")
            else:
                require(native["max_frames"] == 16 and native["selected_frame_count"] == 16
                    and native["source_indices_exact"] and not native["has_nominal_timestamps"]
                    and native["requested_max_frames"] == 8 and native["max_frames_explicit"] is False
                    and native["automatic_frame_budget_applied"] is True
                    and native["source_duration_seconds"] == 120.0
                    and native["selected_source_frame_indices"][0] == 0
                    and native["selected_source_frame_indices"][-1] == 3599, "After sixteen/PTS")
                require(native["grid_t"] == 8 and native["temporal2_pair_first_timestamps"] ==
                    [native["selected_frame_timestamps"][min(2 * index, 15)] for index in range(8)],
                    "temporal-2")
            latency = {"decode_prefill": execution["video_input_ms"] + execution["vision_ms"]
                + execution["prefill_ms"], "first_text": response["first_text_s"] * 1000,
                "first_voice": response["first_voice_s"] * 1000,
                "public_total": response["total_s"] * 1000}
            require(row["latency_ms"] == latency, "latency projection")
            rows.append({"implementation": implementation, "block": block,
                         "run_id": report["run_id"], **row})
        reports.append(report)
    distributions, quality, thresholds = {}, {}, []
    for implementation in ("baseline", "candidate"):
        for case in CASES:
            selected = [row for row in rows if row["implementation"] == implementation
                and row["case_id"] == case and row["phase"] == "measured"]
            require(len(selected) == 16, "measured population")
            key = f"{implementation}/{case}"
            distributions[key] = {metric: summary([row["latency_ms"][metric] for row in selected])
                                  for metric in METRICS}
            hits = sum(row["expected_answer_hit"] for row in selected)
            quality[key] = {"hits": hits, "total": 16, "misses": 16 - hits}
            for metric in ("first_text", "first_voice"):
                if distributions[key][metric]["p95_ms"] >= 3000:
                    thresholds.append({"cell": key, "metric": metric,
                        "p95_ms": distributions[key][metric]["p95_ms"]})
    for left, right in (("A", "D"), ("B", "C")):
        first = next(report for report in reports if report["block"] == left)
        second = next(report for report in reports if report["block"] == right)
        for arow in first["records"]:
            brow = next(row for row in second["records"]
                        if row["schedule_index"] == arow["schedule_index"])
            require(arow["response"]["answer"] == brow["response"]["answer"] and
                arow["response"]["pcm_sha256"] == brow["response"]["pcm_sha256"] and
                arow["response"]["pcm_bytes"] == brow["response"]["pcm_bytes"] and
                all(arow["native_evidence"][field] == brow["native_evidence"][field]
                    for field in PARITY), "same-arm fresh-stack repeat")
    outcomes = {case: {"baseline_hits": quality[f"baseline/{case}"]["hits"],
        "candidate_hits": quality[f"candidate/{case}"]["hits"],
        "delta_hits": quality[f"candidate/{case}"]["hits"] - quality[f"baseline/{case}"]["hits"]}
        for case in CASES}
    quality_pass = (any(row["delta_hits"] > 0 for row in outcomes.values()) and
                    all(row["delta_hits"] >= 0 for row in outcomes.values()))
    promoted = quality_pass and not thresholds
    analysis = load(root / "analysis.json")
    require(analysis["quality_pass"] == quality_pass and analysis["candidate_promotion"] == promoted,
            "producer promotion")
    require(analysis["distributions"] == distributions, "producer distributions")
    require(analysis["threshold_failures"] == thresholds, "producer thresholds")
    for case in CASES:
        require(analysis["quality_outcomes"][case]["baseline_hits"] == outcomes[case]["baseline_hits"]
            and analysis["quality_outcomes"][case]["candidate_hits"] == outcomes[case]["candidate_hits"],
            "producer quality")
    return {"status": "PASS", "requests": len(rows), "quality": quality,
        "quality_outcomes": outcomes, "quality_pass": quality_pass,
        "first_response_threshold_failures": thresholds,
        "candidate_promotion_independent": promoted,
        "distributions": distributions,
        "same_arm_repeat_identity": "36/36 exact answer/native/PCM pairs",
        "source_pts_temporal2_cold_prefix": "PASS"}


def verify_manifest(root):
    root = Path(root).resolve()
    manifest = root / "SHA256SUMS"
    expected = {}
    for line in manifest.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        path = PurePosixPath(relative)
        require(not path.is_absolute() and ".." not in path.parts and relative != "SHA256SUMS", "unsafe manifest")
        expected[relative] = digest
    actual = {path.relative_to(root).as_posix(): path for path in root.rglob("*")
              if path.is_file() and path.name != "SHA256SUMS"}
    require(set(actual) == set(expected), f"manifest file set differs: missing={set(expected)-set(actual)} extra={set(actual)-set(expected)}")
    for relative, path in actual.items():
        require(sha(path) == expected[relative], f"manifest digest: {relative}")
    return {"status": "PASS", "files": len(actual),
            "bytes": sum(path.stat().st_size for path in actual.values()),
            "sha256sums_sha256": sha(manifest)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    try:
        result = verify_manifest(args.root) if args.manifest_only else verify(args.root)
        Path(args.output).write_bytes(json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
        print(json.dumps({"status": "PASS", "output": args.output}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"NS05 VERIFY FAIL: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
