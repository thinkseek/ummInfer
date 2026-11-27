#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent fresh-process verification of the sealed NS-01/NS-02 bundle.

The verifier does not import the scorecard producer or live drivers. It rejects
extra/missing/tampered files, rehashes raw report/media evidence, reconstructs
all latency populations and thresholds, and confirms every NOT_RUN boundary.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))


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
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] * (1.0 - (position - lower)) + ordered[upper] * (position - lower)


def summary(values):
    median = statistics.median(values)
    q1, q3 = quantile(values, 0.25), quantile(values, 0.75)
    iqr = q3 - q1
    return {"count": len(values), "all_ms": values, "median_ms": median,
            "p95_ms": quantile(values, 0.95), "iqr_ms": iqr,
            "mad_ms": statistics.median(abs(value - median) for value in values),
            "iqr_outliers_ms": [value for value in values
                                if value < q1 - 1.5 * iqr or value > q3 + 1.5 * iqr],
            "outlier_policy": "retained_in_all_statistics"}


def close(actual, expected, label):
    require(math.isfinite(actual) and math.isfinite(expected)
            and abs(actual - expected) <= 1e-9 * max(1.0, abs(expected)),
            f"numeric mismatch for {label}: {actual} != {expected}")


def check_summary(actual, values, label):
    expected = summary(values)
    require(actual == expected, f"summary mismatch: {label}")


def manifest(root):
    expected = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) and relative != "SHA256SUMS"
                and not relative.startswith("/") and ".." not in Path(relative).parts,
                f"invalid manifest row: {line}")
        require(relative not in expected, "duplicate manifest row")
        expected[relative] = digest
    actual = {path.relative_to(root).as_posix(): path for path in root.rglob("*")
              if path.is_file() and path != root / "SHA256SUMS"}
    require(set(actual) == set(expected), "final manifest is incomplete or has extra files")
    for relative, digest in expected.items():
        require(sha_file(actual[relative]) == digest, f"manifest digest mismatch: {relative}")
    return len(expected), sha_file(root / "SHA256SUMS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    root = args.root.resolve()
    require(not args.out.exists() and root not in args.out.resolve().parents,
            "verification output must be new and outside the sealed bundle")
    file_count, manifest_sha = manifest(root)
    scorecard_path = root / "scorecard.json"
    score = json.loads(scorecard_path.read_text())
    require(score["format"] == "umminfer-ns01-ns02-interactive-preview-scorecard-v1"
            and score["official_base_commit"] == "91310b36184c86a1c147debb27db8643cc0cb3e4"
            and score["candidate_commit"] == "3f98fe1645d4d82877372056ba8ad2544aa399c5"
            and score["preregistration_sha256"]
            == sha_file(root / "ns02" / "preregistration.json")
            and score["physical_gpu"] == 7, "scorecard identity mismatch")
    ns01 = score["ns01"]
    require(ns01["human_perceptual_review"] == "NOT_RUN"
            and ns01["exact_chinese_golden_case"] == "FAIL_RETAINED_SEPARATELY"
            and len(ns01["runs"]) == 6 and len(ns01["paired"]) == 3,
            "NS-01 population or boundary mismatch")
    require([pair["pair"] for pair in ns01["paired"]] == [0, 1, 2],
            "NS-01 pair IDs must be unique and complete")
    require([(run["label"], run["arm"], run["pair"]) for run in ns01["runs"]]
            == [(f"before-{i}", "before", i) for i in range(3)]
            + [(f"after-{i}", "after", i) for i in range(3)],
            "NS-01 run identities, arms, or pair coverage changed")
    metric_fields = {"ttft": "ttft_ns", "ttfa": "ttfa_ns",
                     "first_image": "first_image_ns", "video": "video_public_complete_ns"}
    limits = {"ttft": 800.0, "ttfa": 800.0, "first_image": 2000.0, "video": 5000.0}
    for run in ns01["runs"]:
        bundle = root / run["label"]
        report_path = bundle / "report.json"
        report = json.loads(report_path.read_text())
        require(sha_file(report_path) == run["report_sha256"]
                and sha_file(bundle / "SHA256SUMS") == run["manifest_sha256"]
                and sha_file(bundle / "independent-verification.json")
                == run["independent_verification_sha256"], f"NS-01 identity mismatch: {run['label']}")
        require(report["scenario"] == "ns01-black-hole" and report["physical_gpu"] == 7
                and report["source"]["sha256"] == run["source_sha256"]
                and report["video"]["sha256"] == run["video_sha256"]
                and report["clean_text_sha256"] == run["clean_text_sha256"]
                and report["raw_sse"]["raw_sha256"] == run["raw_sse_sha256"]
                and report["health_final"]["in_flight"] == 0, f"NS-01 raw binding mismatch: {run['label']}")
        require(sha_file(bundle / "source.png") == run["source_sha256"]
                and sha_file(bundle / "video.mp4") == run["video_sha256"],
                f"NS-01 media digest mismatch: {run['label']}")
        native = (bundle / "video.log").read_text(errors="replace")
        threshold = re.search(r"teacache_thresh=([^\s]+)", native)
        require(threshold and float(threshold.group(1)) == run["effective_teacache"],
                f"NS-01 effective policy mismatch: {run['label']}")
        native_lines = [line for line in native.splitlines()
                        if "video request backend=native-cuda-wan" in line]
        require(len(native_lines) == 1, "NS-01 native phase line count mismatch")
        fields = dict(re.findall(r"([a-z0-9_]+)=([^\s]+)", native_lines[0]))
        hits, total = (int(value) for value in fields["teacache_hits"].split("/"))
        phase = run["native_phase"]
        require(phase["teacache_hits"] == hits and phase["teacache_total_calls"] == total
                and phase["teacache_thresh"] == float(fields["teacache_thresh"]),
                "NS-01 native cache counters differ")
        for key in ("umt5_ms", "dit_ms", "vae_encode_ms", "vae_decode_ms",
                    "denoise_ms", "encode_ms", "elapsed_ms"):
            close(phase[key], float(fields[key]), f"{run['label']}.{key}")
        for metric, field in metric_fields.items():
            close(run["timings_ms"][metric], report["interactive_timings"][field],
                  f"{run['label']}.{metric}")
    for metric, field in metric_fields.items():
        for arm in ("before", "after"):
            values = [run["timings_ms"][metric] for run in ns01["runs"] if run["arm"] == arm]
            check_summary(ns01["metrics"][metric][arm], values, f"ns01.{metric}.{arm}")
            if arm == "after":
                threshold = ns01["metrics"][metric]["after_threshold"]
                require(threshold == {"limit_ms": limits[metric],
                    "median_pass": statistics.median(values) < limits[metric],
                    "p95_pass": quantile(values, 0.95) < limits[metric],
                    "all_runs_pass": all(value < limits[metric] for value in values)},
                    f"NS-01 threshold mismatch: {metric}")
        deltas = [a["timings_ms"][metric] - b["timings_ms"][metric]
                  for b, a in zip(ns01["runs"][:3], ns01["runs"][3:])]
        check_summary(ns01["metrics"][metric]["paired_delta_ms"], deltas, f"ns01.{metric}.delta")
    for key in ("dit_ms", "denoise_ms", "vae_decode_ms", "elapsed_ms"):
        for arm in ("before", "after"):
            values = [run["native_phase"][key] for run in ns01["runs"] if run["arm"] == arm]
            check_summary(ns01["native_phase_metrics"][key][arm], values,
                          f"ns01.native.{key}.{arm}")
    from types import SimpleNamespace
    from i2v_geometry_latency_verify import (
        ns02_quality as independent_quality, verify_bundle as verify_ns02_bundle)
    for pair in ns01["paired"]:
        b = ns01["runs"][pair["pair"]]
        a = ns01["runs"][pair["pair"] + 3]
        b_report = json.loads((root / b["label"] / "report.json").read_text())
        a_report = json.loads((root / a["label"] / "report.json").read_text())
        def quality_run(report):
            return {"shape": "square", "media_file": "video.mp4",
                    "decoded_rgb": report["decoded_rgb"],
                    "first_frame_mae": report["first_frame_mae"],
                    "motion_mae": report["motion_mae"]}
        quality = independent_quality(root / a["label"], args, quality_run(b_report),
                                      quality_run(a_report), baseline_root=root / b["label"])
        quality["quality_gate_pass"] = bool(quality["quality_gate_pass"]
                                                and pair["source_bytes_exact"]
                                                and pair["clean_text_exact"])
        require(quality == pair["quality"], f"NS-01 independent quality mismatch: {pair['pair']}")
        require(pair["source_bytes_exact"] == (b["source_sha256"] == a["source_sha256"])
                and pair["clean_text_exact"] == (b["clean_text_sha256"] == a["clean_text_sha256"])
                and pair["before_video_sha256"] == b["video_sha256"]
                and pair["after_video_sha256"] == a["video_sha256"],
                "NS-01 paired identity mismatch")
        close(pair["video_delta_ms"], a["timings_ms"]["video"] - b["timings_ms"]["video"],
              f"ns01.pair{pair['pair']}.video_delta")
    ns01_gate = all(pair["quality"]["quality_gate_pass"] and pair["video_delta_ms"] < 0
                    for pair in ns01["paired"])
    ns01_gate = ns01_gate and ns01["metrics"]["video"]["after_threshold"]["median_pass"] \
        and ns01["metrics"]["video"]["after_threshold"]["p95_pass"]
    for metric, values in ns01["metrics"].items():
        expected_no_regression = metric == "video" or (
            values["after"]["median_ms"] <= 1.05 * values["before"]["median_ms"])
        require(values["non_video_no_regression_5_percent"] == expected_no_regression,
                f"NS-01 no-regression flag mismatch: {metric}")
        ns01_gate = ns01_gate and expected_no_regression
    require(ns01["paired_video_gate_pass"] == ns01_gate, "NS-01 paired gate mismatch")
    ns02 = json.loads((root / "ns02" / "report.json").read_text())
    require(sha_file(root / "ns02" / "report.json") == score["ns02"]["report_sha256"]
            and sha_file(root / "ns02" / "SHA256SUMS") == score["ns02"]["manifest_sha256"]
            and sha_file(root / "ns02-independent-verification.json")
            == score["ns02"]["independent_verification_sha256"]
            and ns02["candidate_analysis"] == score["ns02"]["analysis"]
            and ns02["mode"] == "ns02" and ns02["physical_gpu"] == 7,
            "NS-02 raw report binding mismatch")
    geometry_pass = True
    for shape in ("wide", "tall", "square"):
        values = [run["public_complete_artifact_ms"] for run in ns02["runs"]
                  if run["role"] == "measured" and run["arm"] == "candidate"
                  and run["shape"] == shape]
        expected = {"limit_ms": 5000.0,
                    "median_pass": statistics.median(values) < 5000.0,
                    "p95_pass": quantile(values, 0.95) < 5000.0,
                    "all_runs_pass": all(value < 5000.0 for value in values)}
        require(score["ns02"]["per_geometry_after_thresholds"][shape] == expected,
                f"NS-02 per-geometry threshold mismatch: {shape}")
        geometry_pass = geometry_pass and expected["median_pass"] and expected["p95_pass"] \
            and expected["all_runs_pass"]
    require(score["ns02"]["per_geometry_hard_threshold_pass"] == geometry_pass,
            "NS-02 aggregate geometry threshold mismatch")
    for arm in ("baseline", "candidate"):
        item = next(item for item in ns02["candidate_analysis"]["cancellations"]
                    if item["arm"] == arm)
        check_summary(score["ns02"]["cancellation_latency"][arm]["native_terminal"],
                      [item["disconnect_to_native_terminal_ms"]], f"score.ns02.{arm}.native")
        check_summary(score["ns02"]["cancellation_latency"][arm]["client_settled"],
                      [item["disconnect_to_client_settled_ms"]], f"score.ns02.{arm}.client")
    identities = score["identities"]
    for key in ("server_binary", "baseline_wan_binary", "candidate_wan_binary",
                "live_harness", "independent_verifier", "ffmpeg", "ffprobe"):
        require(identities[key] == ns02["provenance"][key], f"identity mismatch: {key}")
    first_ns01 = json.loads((root / "before-0" / "report.json").read_text())
    for key in ("vlm_binary", "tts_binary"):
        expected = {field: first_ns01["provenance"][key][field] for field in ("sha256", "bytes")}
        require(identities[key] == expected, f"identity mismatch: {key}")
    require(identities["model_revision"] == "UNKNOWN", "model revision identity mismatch")
    wan_index = ns02["models"]["wan22"]["files"]["model_index.json"]
    wan_index_sha = wan_index["sha256"] if isinstance(wan_index, dict) else wan_index
    require(identities["models"]["wan22"]["config_sha256"] == wan_index_sha,
            "compact Wan config identity does not match raw full-file evidence")
    for name, identity in first_ns01["models"].items():
        compact = identities["models"][name]
        require(compact["tree_sha256"] == identity["tree_sha256"]
                and compact["file_count"] == identity["file_count"]
                and compact["model_id"] == identity["model_id"], f"model identity mismatch: {name}")
    ns02_args = SimpleNamespace(
        ffmpeg=args.ffmpeg, ffprobe=args.ffprobe, skip_execution_binary_recheck=True,
        recorded_verifier_sha256=None, server_binary=None, baseline_wan_binary=None,
        candidate_wan_binary=None, flux_stage=None, wan_model=None, flux_model=None)
    ns02_audit = verify_ns02_bundle(root / "ns02", ns02_args,
                                    nested=True, allow_nested_ns02=True)
    require(ns02_audit["candidate"]["promotion_decision"]
            == ns02["candidate_analysis"]["promotion_decision"],
            "independent NS-02 promotion recomputation disagrees")
    for pair in ns02["candidate_analysis"]["paired"]:
        for arm, key in (("baseline", "baseline_sha256"), ("candidate", "candidate_sha256")):
            run = next(item for item in ns02["runs"] if item["run_id"] == pair[f"{arm}_run_id"])
            media = root / "ns02" / run["media_file"]
            require(sha_file(media) == run["media_sha256"] == pair[key],
                    f"NS-02 media mismatch: {run['run_id']}")
    for failure in score["failures"]:
        bundle = root / "failures" / failure["label"]
        if failure["label"] == "ns02-r1":
            files = {path.relative_to(bundle).as_posix(): sha_file(path)
                     for path in sorted(bundle.rglob("*")) if path.is_file()}
            require(sha_file(bundle / "ns02-r1.log") == failure["log_sha256"]
                    and len(files) == failure["file_count"]
                    and hashlib.sha256(canonical(files)).hexdigest() == failure["tree_sha256"]
                    and "FFmpeg SSIM output is missing"
                    in (bundle / "ns02-r1.log").read_text(),
                    "NS-02 R1 raw tooling-failure tree mismatch")
            continue
        require(sha_file(bundle / "capture" / "trajectories.jsonl")
                == failure["trajectory_sha256"]
                and sha_file(bundle / "request.json") == failure["request_sha256"],
                f"failure trajectory/request mismatch: {failure['label']}")
        for name, digest in failure["native_log_sha256"].items():
            require(sha_file(bundle / name) == digest, f"failure log mismatch: {failure['label']}/{name}")
    expected_decision = "PROMOTED" if ns01_gate and geometry_pass and \
        ns02["candidate_analysis"]["promotion_decision"] == "PROMOTED" else "NOT_PROMOTED"
    require("human_or_perceptual_judging" in score["not_run"]
            and "resident_service" in score["not_run"]
            and score["promotion_decision"] == expected_decision,
            "NOT_RUN or independently recomputed promotion boundary mismatch")
    result = {"status": "PASS", "scorecard_sha256": sha_file(scorecard_path),
              "manifest_sha256": manifest_sha, "manifest_files": file_count,
              "promotion_decision": score["promotion_decision"],
              "ns01_pairs": 3, "ns02_pairs": len(ns02["candidate_analysis"]["paired"]),
              "raw_media_rehash": "PASS", "statistics_and_thresholds": "PASS",
              "ns02_full_decision_recomputed": "PASS",
              "ns01_unique_pair_coverage": "PASS",
              "human_perceptual_review": "NOT_RUN"}
    args.out.write_bytes(canonical(result) + b"\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
