#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Emit a compact NS-02 Before/After scorecard from sealed external evidence.

The caller selects every input and output path. Raw media and logs stay external;
this stdlib-only tool copies no absolute paths into its JSON output.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

FORMAT = "umminfer-ns02-fidelity-speedup-scorecard-v1"
EXPECTED_CANDIDATE = "automatic-i2v-boundary2-teacache-0.25"
EXPECTED_CASES = {
    "wide": {"source": [704, 480], "target": [704, 480]},
    "tall": {"source": [480, 704], "target": [480, 704]},
    "square": {"source": [512, 512], "target": [704, 480]},
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    require(ordered and 0.0 <= fraction <= 1.0, "invalid quantile input")
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summary(values):
    require(values, "empty latency population")
    median = statistics.median(values)
    q1 = quantile(values, 0.25)
    q3 = quantile(values, 0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return {
        "count": len(values), "all_ms": values,
        "median_ms": median, "p95_ms": quantile(values, 0.95),
        "min_ms": min(values), "max_ms": max(values),
        "iqr_ms": iqr,
        "mad_ms": statistics.median(abs(value - median) for value in values),
        "iqr_outliers_ms": [value for value in values if value < low or value > high],
        "outlier_policy": "retained_in_all_statistics",
    }


def build_scorecard(evidence, verification_path):
    evidence = Path(evidence)
    report_path = evidence / "report.json"
    manifest_path = evidence / "SHA256SUMS"
    verification_path = Path(verification_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    report_sha = sha_file(report_path)
    manifest_sha = sha_file(manifest_path)
    verification_sha = sha_file(verification_path)
    require(report["mode"] == "ns02" and report["physical_gpu"] == 7,
            "wrong NS-02 execution scope")
    analysis = report["candidate_analysis"]
    require(analysis["candidate_id"] == EXPECTED_CANDIDATE
            and analysis["candidate_policy"]["protected_step_indices"] == [0, 1, 18, 19]
            and analysis["candidate_policy"]["cacheable_step_indices"] == list(range(2, 18)),
            "candidate or frozen boundary2 schedule mismatch")
    require(verification["status"] == "PASS"
            and verification["mode"] == "ns02"
            and verification["report_sha256"] == report_sha
            and verification["manifest_sha256"] == manifest_sha,
            "fresh-process evidence verification is absent or does not bind this report")
    require(report["final_health"] == {"http_status": 200, "body": {
        "status": "ok", "in_flight": 0}}, "final in_flight is not zero")

    measured = [run for run in report["runs"] if run["role"] == "measured"]
    require(len(measured) == 30, "NS-02 requires exactly 30 measured arm/shape runs")
    pairs = analysis["paired"]
    require(len(pairs) == 15, "NS-02 requires exactly 15 retained pairs")
    failures = []
    shapes = {}
    all_after = []
    all_before = []
    all_deltas = []
    for shape, geometry in EXPECTED_CASES.items():
        shape_pairs = [item for item in pairs if item["shape"] == shape]
        require(len(shape_pairs) == 5, f"{shape} does not have five pairs")
        before = [item["baseline_ms"] for item in shape_pairs]
        after = [item["candidate_ms"] for item in shape_pairs]
        deltas = [item["candidate_minus_baseline_ms"] for item in shape_pairs]
        all_before.extend(before)
        all_after.extend(after)
        all_deltas.extend(deltas)
        shape_runs = [run for run in measured if run["shape"] == shape]
        source_hashes = {run["source_sha256"] for run in shape_runs}
        require(source_hashes == {report["sources"][shape]["source_sha256"]},
                f"{shape} source hash binding mismatch")
        expected_stream = {"codec_name": "h264", "height": geometry["target"][1],
                           "nb_frames": "29", "pix_fmt": "yuv420p",
                           "r_frame_rate": "16/1", "width": geometry["target"][0]}
        require(all(run["video_stream"] == expected_stream for run in shape_runs),
                f"{shape} decoded MP4 semantics mismatch")
        for item in shape_pairs:
            quality = item["quality"]
            if not quality["first_frame_rgb_exact"]:
                failures.append({"code": "EXACT_FIRST_FRAME_RGB", "shape": shape,
                                 "repetition": item["repetition"]})
            if quality["ssim"]["all"] < 0.90:
                failures.append({"code": "SSIM_BELOW_0.90", "shape": shape,
                                 "repetition": item["repetition"],
                                 "actual": quality["ssim"]["all"]})
            if quality["all_frame_psnr_db"] < 24.0:
                failures.append({"code": "PSNR_BELOW_24_DB", "shape": shape,
                                 "repetition": item["repetition"]})
            if not 0.5 <= quality["temporal_motion_ratio"] <= 1.5:
                failures.append({"code": "TEMPORAL_MOTION_RATIO", "shape": shape,
                                 "repetition": item["repetition"]})
            candidate_run = next(run for run in shape_runs
                                 if run["arm"] == "candidate"
                                 and run["repetition"] == item["repetition"])
            if candidate_run["first_frame_mae"] > 48.0:
                failures.append({"code": "SOURCE_FIRST_FRAME_MAE", "shape": shape,
                                 "repetition": item["repetition"]})
            if quality["candidate_first_last_mae"] < 1.0:
                failures.append({"code": "NONZERO_MOTION", "shape": shape,
                                 "repetition": item["repetition"]})
        latency_pass = all(value < 5000.0 for value in after)
        if not latency_pass:
            failures.append({"code": "MEASURED_ARTIFACT_AT_OR_ABOVE_5000_MS",
                             "shape": shape,
                             "observations_ms": [v for v in after if v >= 5000.0]})
        shapes[shape] = {
            "source_sha256": next(iter(source_hashes)),
            "source_size": geometry["source"], "target_size": geometry["target"],
            "decoded_mp4": expected_stream,
            "before": summary(before), "after": summary(after),
            "paired_delta": summary(deltas),
            "all_after_below_5000_ms": latency_pass,
            "pairs": [{
                "repetition": item["repetition"],
                "before_ms": item["baseline_ms"], "after_ms": item["candidate_ms"],
                "delta_ms": item["candidate_minus_baseline_ms"],
                "before_mp4_sha256": item["baseline_sha256"],
                "after_mp4_sha256": item["candidate_sha256"],
                "first_frame_rgb_exact": item["quality"]["first_frame_rgb_exact"],
                "first_frame_rgb_sha256_before":
                    item["quality"]["first_frame_rgb_sha256_before"],
                "first_frame_rgb_sha256_after":
                    item["quality"]["first_frame_rgb_sha256_candidate"],
                "ssim": item["quality"]["ssim"]["all"],
                "all_frame_psnr_db": item["quality"]["all_frame_psnr_db"],
                "temporal_motion_ratio": item["quality"]["temporal_motion_ratio"],
                "quality_gate_pass": item["quality"]["quality_gate_pass"],
            } for item in shape_pairs],
        }

    cancellations = {}
    cancellation_pass = True
    for arm in ("baseline", "candidate"):
        records = [item for item in report["cancellations"] if item["arm"] == arm]
        require(len(records) == 1, f"{arm} cancellation population is not one")
        item = records[0]
        passed = (item["disconnect_to_native_terminal_ms"] < 500.0
                  and item["disconnect_to_client_settled_ms"] < 500.0
                  and item["late_native_records_after_terminal"] == 0
                  and item["public_media_bytes"] == 0
                  and item["health"]["in_flight"] == 0)
        cancellation_pass &= passed
        cancellations[arm] = {
            "native_terminal_ms": item["disconnect_to_native_terminal_ms"],
            "client_settled_ms": item["disconnect_to_client_settled_ms"],
            "late_native_records": item["late_native_records_after_terminal"],
            "late_public_media_bytes": item["public_media_bytes"],
            "final_in_flight": item["health"]["in_flight"],
            "public_abort_ack": item["public_abort_ack"], "pass": passed,
        }
    if not cancellation_pass:
        failures.append({"code": "CANCELLATION_LIFECYCLE"})

    controls = report["controls"]
    malformed = [item for item in controls if item["role"] == "malformed_dimension"]
    retries = [item for item in controls if item["role"] in
               ("post_cancel_recovery", "post_failure_recovery")]
    require(len(malformed) == 2 and all(item["http_status"] == 500 for item in malformed),
            "malformed-input failure controls are incomplete")
    require(len(retries) == 4 and all(item["http_status"] == 200 for item in retries),
            "exact-hash retry/recovery controls are incomplete")
    speed_pass = (statistics.median(all_deltas) <= -2000.0
                  and all(delta < 0.0 for delta in all_deltas))
    quality_pass = all(item["quality"]["quality_gate_pass"] for item in pairs)
    latency_pass = all(value < 5000.0 for value in all_after)
    if not speed_pass:
        failures.append({"code": "PAIRED_SPEEDUP"})
    decision = ("PROMOTED" if all((speed_pass, quality_pass, latency_pass,
                                   cancellation_pass)) else "NOT_PROMOTED")
    require(decision == analysis["promotion_decision"],
            "producer and compact scorecard promotion decisions differ")

    baseline_report = evidence / "baseline-reference" / "report.json"
    baseline_manifest = evidence / "baseline-reference" / "SHA256SUMS"
    return {
        "format": FORMAT,
        "status": "CONTROLLED_LIVE",
        "official_base_commit": report["base_commit"],
        "execution_commit": report["execution_commit"],
        "candidate_commit": analysis["candidate_commit"],
        "candidate_id": analysis["candidate_id"],
        "candidate_policy": analysis["candidate_policy"],
        "physical_gpu": 7,
        "promotion_decision": decision,
        "gates": {"paired_speedup": speed_pass, "fidelity": quality_pass,
                  "every_measured_geometry_below_5000_ms": latency_pass,
                  "cancellation": cancellation_pass,
                  "malformed_input_failure": True,
                  "immediate_exact_hash_retry_and_recovery": True,
                  "zero_late_records_media_and_final_in_flight_zero": cancellation_pass,
                  "native_terminal_dedup": True},
        "pooled": {"before": summary(all_before), "after": summary(all_after),
                   "paired_delta": summary(all_deltas)},
        "per_shape": shapes,
        "cancellation": cancellations,
        "controls": {
            "malformed": [{"arm": item["arm"], "http_status": item["http_status"],
                           "native_success": item["native_success"]} for item in malformed],
            "retry_recovery": [{"arm": item["arm"], "role": item["role"],
                                "http_status": item["http_status"],
                                "mp4_sha256": item["media_sha256"]} for item in retries],
        },
        "failures": failures,
        "human_perceptual_review": "NOT_RUN",
        "deployment": "NOT_RUN",
        "resident_service": "NOT_CONTACTED",
        "raw_hashes": {
            "report_sha256": report_sha, "manifest_sha256": manifest_sha,
            "independent_verification_sha256": verification_sha,
            "preregistration_sha256": sha_file(evidence / "preregistration.json"),
            "baseline_profile_report_sha256": sha_file(baseline_report),
            "baseline_profile_manifest_sha256": sha_file(baseline_manifest),
        },
        "identities": {
            "server_binary": report["provenance"]["server_binary"],
            "baseline_wan_binary": report["provenance"]["baseline_wan_binary"],
            "candidate_wan_binary": report["provenance"]["candidate_wan_binary"],
            "wan_model_tree_sha256": report["models"]["wan22"]["tree_sha256"],
            "model_revision": "UNKNOWN",
        },
        "evidence_policy": "caller-selected external raw bundle with complete recursive self-excluded SHA256SUMS",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.out.exists(), f"scorecard already exists: {args.out}")
    require(args.out.parent.is_dir(), "scorecard parent directory must be caller-created")
    scorecard = build_scorecard(args.evidence, args.verification)
    args.out.write_bytes(canonical(scorecard) + b"\n")
    print(json.dumps({"promotion_decision": scorecard["promotion_decision"],
                      "failure_count": len(scorecard["failures"]),
                      "scorecard_sha256": sha_file(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
