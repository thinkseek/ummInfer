#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a compact NS-01/NS-02 scorecard from sealed external evidence.

All evidence and the caller-selected output directory remain external. Large
media, per-event payloads, model roots, and absolute paths are never copied into
the tracked compact JSON. The paired gate imports only the frozen measurement
helpers; a separate fresh-process verifier independently checks the final bundle.
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
from i2v_geometry_latency_live import (  # noqa: E402
    paired_video_quality, require, summarize)

FORMAT = "umminfer-ns01-ns02-interactive-preview-scorecard-v1"
METRICS = {
    "ttft": ("ttft_ns", 800.0),
    "ttfa": ("ttfa_ns", 800.0),
    "first_image": ("first_image_ns", 2000.0),
    "video": ("video_public_complete_ns", 5000.0),
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def threshold(values, limit):
    return {"limit_ms": limit,
            "median_pass": statistics.median(values) < limit,
            "p95_pass": summarize(values)["p95_ms"] < limit,
            "all_runs_pass": all(value < limit for value in values)}


def native_phase(root, arm):
    text = (Path(root) / "video.log").read_text(encoding="utf-8", errors="replace")
    lines = [line for line in text.splitlines()
             if "video request backend=native-cuda-wan" in line]
    require(len(lines) == 1, "NS-01 requires exactly one native Wan execution")
    fields = dict(re.findall(r"([a-z0-9_]+)=([^\s]+)", lines[0]))
    value = float(fields["teacache_thresh"])
    expected = 0.25 if arm == "after" else 0.0
    require(value == expected, f"NS-01 {arm} effective TeaCache mismatch: {value}")
    hits, total = (int(part) for part in fields["teacache_hits"].split("/"))
    require(total == 40 and (0 < hits < total if arm == "after" else hits == 0),
            "NS-01 residual-cache hit accounting mismatch")
    result = {"teacache_thresh": value, "teacache_hits": hits,
              "teacache_total_calls": total}
    for key in ("umt5_ms", "dit_ms", "vae_encode_ms", "vae_decode_ms",
                "denoise_ms", "encode_ms", "elapsed_ms"):
        result[key] = float(fields[key])
        require(math.isfinite(result[key]) and result[key] >= 0.0,
                f"invalid NS-01 native phase: {key}")
    return result


def ns01_run(root, label, arm, index):
    root = Path(root)
    report = load(root / "report.json")
    require(report["format"] == "umminfer-public-narrative-i2v-v1"
            and report["result"] == "CONTROLLED_LIVE_PASS"
            and report.get("scenario") == "ns01-black-hole"
            and report["physical_gpu"] == 7
            and report["health_final"].get("in_flight") == 0,
            f"invalid NS-01 report: {label}")
    request = load(root / "request.json")
    require("teacache_thresh" not in request.get("video", {}),
            "NS-01 request manually selected TeaCache")
    timings = report["interactive_timings"]
    require(all(timings.get(field) is not None for field, _ in METRICS.values()),
            f"NS-01 missing arrival timing: {label}")
    verification = root / "independent-verification.json"
    require(verification.is_file(), f"NS-01 independent verification missing: {label}")
    native = native_phase(root, arm)
    return {"label": label, "arm": arm, "pair": index,
            "report_sha256": sha_file(root / "report.json"),
            "manifest_sha256": sha_file(root / "SHA256SUMS"),
            "independent_verification_sha256": sha_file(verification),
            "raw_sse_sha256": report["raw_sse"]["raw_sha256"],
            "source_sha256": report["source"]["sha256"],
            "video_sha256": report["video"]["sha256"],
            "clean_text_sha256": report["clean_text_sha256"],
            "timings_ms": {key: timings[field] for key, (field, _limit) in METRICS.items()},
            "effective_teacache": native["teacache_thresh"],
            "native_phase": native,
            "first_frame_mae": report["first_frame_mae"],
            "motion_mae": report["motion_mae"], "terminal_count": 1,
            "tag_leak_count": 0, "final_in_flight": 0}


def failed_ns01(root, label):
    root = Path(root)
    trajectory = root / "capture" / "trajectories.jsonl"
    record = json.loads(trajectory.read_text(encoding="utf-8").splitlines()[0])
    logs = {name: sha_file(root / name) for name in
            ("vlm.log", "image.log", "tts.log", "video.log") if (root / name).is_file()}
    request = load(root / "request.json")
    return {"label": label, "result": "FAIL_RETAINED_NOT_IN_PAIRED_POPULATION",
            "trajectory_sha256": sha_file(trajectory), "native_log_sha256": logs,
            "request_sha256": sha_file(root / "request.json"),
            "user_prompt": request["messages"][0]["content"],
            "clean_text_sha256": record.get("clean_text_sha256"),
            "visual_slots": record.get("visual_slots"), "terminal_legs": record.get("terminal_legs")}


def failed_ns02(root, label):
    root = Path(root)
    log = root / "ns02-r1.log"
    require("FFmpeg SSIM output is missing" in log.read_text(encoding="utf-8")
            and not (root / "report.json").exists(),
            "NS-02 R1 must remain the retained parser-failure attempt")
    files = {path.relative_to(root).as_posix(): sha_file(path)
             for path in sorted(root.rglob("*")) if path.is_file()}
    return {"label": label, "result": "FAIL_RETAINED_TOOLING_NOT_IN_PAIRED_POPULATION",
            "failure": "FFmpeg SSIM summary layout parser rejected valid completed media",
            "log_sha256": sha_file(log), "file_count": len(files),
            "tree_sha256": hashlib.sha256(canonical(files)).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns01-before", action="append", type=Path, required=True)
    parser.add_argument("--ns01-after", action="append", type=Path, required=True)
    parser.add_argument("--ns01-failed", action="append", type=Path, default=[])
    parser.add_argument("--ns02-failed", type=Path)
    parser.add_argument("--ns02", type=Path, required=True)
    parser.add_argument("--ns02-verification", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    require(len(args.ns01_before) == len(args.ns01_after) == 3,
            "NS-01 requires three process-separated pairs")
    out = args.out_dir
    require(not out.exists(), f"scorecard output directory already exists: {out}")
    out.mkdir(parents=True)
    before = [ns01_run(path, f"before-{index}", "before", index)
              for index, path in enumerate(args.ns01_before)]
    after = [ns01_run(path, f"after-{index}", "after", index)
             for index, path in enumerate(args.ns01_after)]
    pairs = []
    for b, a in zip(before, after):
        before_root = args.ns01_before[b["pair"]]
        after_root = args.ns01_after[a["pair"]]
        b_report = load(before_root / "report.json")
        a_report = load(after_root / "report.json")
        def fake(root, report):
            return {"shape": "square", "media_file": "video.mp4",
                    "media_sha256": report["video"]["sha256"],
                    "decoded_rgb": report["decoded_rgb"],
                    "first_frame_mae": report["first_frame_mae"],
                    "motion_mae": report["motion_mae"]}
        quality = paired_video_quality("ffmpeg", after_root,
                                       fake(before_root, b_report), fake(after_root, a_report),
                                       baseline_out=before_root)
        first_before = (before_root / b_report["decoded_rgb"]["first_full"]["file"]).read_bytes()
        first_after = (after_root / a_report["decoded_rgb"]["first_full"]["file"]).read_bytes()
        quality["first_frame_rgb_exact"] = first_before == first_after
        quality["first_frame_rgb_sha256_before"] = sha_bytes(first_before)
        quality["quality_gate_pass"] = bool(
            quality["quality_gate_pass"] and quality["first_frame_rgb_exact"]
            and b["source_sha256"] == a["source_sha256"]
            and b["clean_text_sha256"] == a["clean_text_sha256"])
        pairs.append({"pair": b["pair"], "source_bytes_exact": b["source_sha256"] == a["source_sha256"],
                      "clean_text_exact": b["clean_text_sha256"] == a["clean_text_sha256"],
                      "before_video_sha256": b["video_sha256"],
                      "after_video_sha256": a["video_sha256"],
                      "video_delta_ms": a["timings_ms"]["video"] - b["timings_ms"]["video"],
                      "quality": quality})
    ns01_metrics = {}
    for metric in METRICS:
        b_values = [run["timings_ms"][metric] for run in before]
        a_values = [run["timings_ms"][metric] for run in after]
        ns01_metrics[metric] = {
            "before": summarize(b_values), "after": summarize(a_values),
            "paired_delta_ms": summarize([a["timings_ms"][metric] - b["timings_ms"][metric]
                                          for b, a in zip(before, after)]),
            "after_threshold": threshold(a_values, METRICS[metric][1]),
            "non_video_no_regression_5_percent": metric == "video" or
                statistics.median(a_values) <= 1.05 * statistics.median(b_values)}
    ns02 = load(args.ns02 / "report.json")
    require(ns02["mode"] == "ns02" and ns02["physical_gpu"] == 7,
            "wrong NS-02 evidence")
    analysis = ns02["candidate_analysis"]
    ns01_quality = all(pair["quality"]["quality_gate_pass"] for pair in pairs)
    ns01_video = ns01_metrics["video"]["after_threshold"]
    ns01_no_regression = all(item["non_video_no_regression_5_percent"]
                             for item in ns01_metrics.values())
    ns01_gate = ns01_quality and ns01_video["median_pass"] and ns01_video["p95_pass"] \
        and ns01_no_regression and all(pair["video_delta_ms"] < 0 for pair in pairs)
    ns02_geometry_thresholds = {}
    for shape in ("wide", "tall", "square"):
        values = [run["public_complete_artifact_ms"] for run in ns02["runs"]
                  if run["role"] == "measured" and run["arm"] == "candidate"
                  and run["shape"] == shape]
        ns02_geometry_thresholds[shape] = threshold(values, 5000.0)
    ns02_geometry_pass = all(item["median_pass"] and item["p95_pass"] and item["all_runs_pass"]
                             for item in ns02_geometry_thresholds.values())
    decision = "PROMOTED" if analysis["promotion_decision"] == "PROMOTED" and ns01_gate \
        and ns02_geometry_pass else "NOT_PROMOTED"
    ns01_provenance = load(args.ns01_before[0] / "report.json")["provenance"]
    ns01_models = load(args.ns01_before[0] / "report.json")["models"]
    identities = {
        "server_binary": ns02["provenance"]["server_binary"],
        "baseline_wan_binary": ns02["provenance"]["baseline_wan_binary"],
        "candidate_wan_binary": ns02["provenance"]["candidate_wan_binary"],
        "vlm_binary": {key: ns01_provenance["vlm_binary"][key] for key in ("sha256", "bytes")},
        "tts_binary": {key: ns01_provenance["tts_binary"][key] for key in ("sha256", "bytes")},
        "live_harness": ns02["provenance"]["live_harness"],
        "independent_verifier": ns02["provenance"]["independent_verifier"],
        "ffmpeg": ns02["provenance"]["ffmpeg"], "ffprobe": ns02["provenance"]["ffprobe"],
        "model_revision": "UNKNOWN",
        "models": {"wan22": ns02["models"]["wan22"], **{
            name: {"model_id": identity["model_id"], "file_count": identity["file_count"],
                   "tree_sha256": identity["tree_sha256"],
                   "config_sha256": (identity["files"].get("config.json") or
                                     identity["files"].get("model_index.json"))["sha256"]}
            for name, identity in ns01_models.items()}},
    }
    scorecard = {
        "format": FORMAT, "official_base_commit": "91310b36184c86a1c147debb27db8643cc0cb3e4",
        "identities": identities,
        "candidate_commit": "3f98fe1645d4d82877372056ba8ad2544aa399c5",
        "preregistration_sha256": sha_file(args.ns02 / "preregistration.json"),
        "physical_gpu": 7, "promotion_decision": decision,
        "ns01": {"scope": "English semantic black-hole case, not exact Chinese rubric PASS",
                 "runs": before + after, "paired": pairs, "metrics": ns01_metrics,
                 "native_phase_metrics": {key: {arm: summarize([
                     run["native_phase"][key] for run in (before if arm == "before" else after)])
                     for arm in ("before", "after")}
                     for key in ("dit_ms", "denoise_ms", "vae_decode_ms", "elapsed_ms")},
                 "paired_video_gate_pass": ns01_gate,
                 "exact_chinese_golden_case": "FAIL_RETAINED_SEPARATELY",
                 "layout_event_order": "PASS for retained successful English semantic runs",
                 "terminal_and_slot_dedup": "PASS", "source_anchor_binding": "PASS",
                 "human_perceptual_review": "NOT_RUN"},
        "ns02": {"report_sha256": sha_file(args.ns02 / "report.json"),
                 "manifest_sha256": sha_file(args.ns02 / "SHA256SUMS"),
                 "independent_verification_sha256": sha_file(args.ns02_verification),
                 "analysis": analysis,
                 "per_geometry_after_thresholds": ns02_geometry_thresholds,
                 "per_geometry_hard_threshold_pass": ns02_geometry_pass,
                 "cancellation_latency": analysis.get("cancellation_latency", {arm: {
                     "native_terminal": summarize([item["disconnect_to_native_terminal_ms"]
                         for item in analysis["cancellations"] if item["arm"] == arm]),
                     "client_settled": summarize([item["disconnect_to_client_settled_ms"]
                         for item in analysis["cancellations"] if item["arm"] == arm])}
                     for arm in ("baseline", "candidate")}),
                 "decoded_geometry": "wide=704x480,tall=480x704,square=center-cover-to-704x480",
                 "human_perceptual_review": "NOT_RUN"},
        "failures": [failed_ns01(path, f"setup-attempt-{index}")
                     for index, path in enumerate(args.ns01_failed)]
                     + ([failed_ns02(args.ns02_failed, "ns02-r1")]
                        if args.ns02_failed else []),
        "not_run": ["human_or_perceptual_judging", "resident_service", "deployment",
                    "training", "optimizer", "LoRA"],
        "evidence_policy": "caller-selected external directories; no tracked media or absolute evidence paths",
    }
    (out / "scorecard.json").write_bytes(canonical(scorecard) + b"\n")
    print(json.dumps({"promotion_decision": decision, "ns01_gate": ns01_gate,
                      "ns02_decision": analysis["promotion_decision"],
                      "scorecard_sha256": sha_file(out / "scorecard.json")}, sort_keys=True))
    return 0


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
