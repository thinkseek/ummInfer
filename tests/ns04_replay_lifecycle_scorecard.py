#!/usr/bin/env python3
"""Aggregate independently verified NS-04 replay/lifecycle runs without deleting outliers."""

import argparse
import hashlib
import json
import math
from pathlib import Path

METRICS = (
    "barge_to_cancel",
    "cancel_to_digital_silence",
    "barge_to_speech_stopped_replay",
    "speech_stopped_to_asr_final",
    "asr_final_to_new_audio",
    "replay_inclusive_barge_to_new_audio",
)


def quantile(values, p):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty sample")
    position = (len(ordered) - 1) * p
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values):
    if not values:
        raise ValueError("empty timing sample")
    ordered = sorted(float(value) for value in values)
    mean = math.fsum(ordered) / len(ordered)
    variance = math.fsum((value - mean) ** 2 for value in ordered) / len(ordered)
    median = quantile(ordered, 0.5)
    q1 = quantile(ordered, 0.25)
    q3 = quantile(ordered, 0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    deviations = sorted(abs(value - median) for value in ordered)
    return {
        "n": len(ordered),
        "samples_ms": ordered,
        "min_ms": ordered[0],
        "median_ms": median,
        "p95_ms": quantile(ordered, 0.95),
        "max_ms": ordered[-1],
        "mean_ms": mean,
        "population_cv": math.sqrt(variance) / mean if mean else 0.0,
        "iqr_ms": iqr,
        "mad_ms": quantile(deviations, 0.5),
        "tukey_lower_fence_ms": lower,
        "tukey_upper_fence_ms": upper,
        "outliers_ms": [value for value in ordered if value < lower or value > upper],
    }


def threshold(summary, limit):
    return {
        "rule": f"max < {limit} ms",
        "observed_median_ms": summary["median_ms"],
        "observed_p95_ms": summary["p95_ms"],
        "observed_max_ms": summary["max_ms"],
        "outliers_ms": summary["outliers_ms"],
        "pass": summary["max_ms"] < limit,
    }


def exact_pair(before, after):
    fields = [("transcripts", before.get("transcripts") == after.get("transcripts"))]
    for label in ("first_barge", "second_barge"):
        b = before[label]["played"]
        a = after[label]["played"]
        for key in ("acknowledged_cursor", "acknowledged_text", "played_pcm_sha256"):
            fields.append((f"{label}.played.{key}", b.get(key) == a.get(key)))
        for key in ("old_pcm_sha256", "pending_pcm_sha256", "old_response_text_sha256"):
            fields.append((f"{label}.{key}", before[label].get(key) == after[label].get(key)))
    for key in ("text_sha256", "pcm_sha256"):
        fields.append((f"final_response.{key}",
                       before["final_response"].get(key) == after["final_response"].get(key)))
    for key in ("accepted_cursors", "vlm_prompts_sha256"):
        fields.append((f"history.{key}", before["history"].get(key) == after["history"].get(key)))
    return {"pass": all(ok for _, ok in fields),
            "checks": [{"field": name, "pass": ok} for name, ok in fields]}


REQUIRED_CONTROLS = (
    "umm_test_local", "native_tts_cuda", "public_pcm_29_29", "network_reconnect",
    "task81_ns01_43_43", "task81_ns04_cursor10",
)


def input_shape(report):
    return {name: {"chunks": row["chunks"], "pcm_bytes": row["pcm_bytes"]}
            for name, row in report["input_schedule"].items()}


def compact_incremental(row, label):
    item = row.get("incremental", {}).get(label, {})
    return {
        "repetition": row.get("repetition"),
        "eligible_sha256": item.get("eligible_sha256"),
        "barge_to_eligible_ms": item.get("barge_to_eligible_ms"),
        "barge_to_vlm_start_ms": item.get("barge_to_vlm_start_ms"),
        "barge_to_tts_start_ms": item.get("barge_to_tts_start_ms"),
        "eligible_to_vlm_start_ms": item.get("eligible_to_vlm_start_ms"),
        "eligible_to_tts_start_ms": item.get("eligible_to_tts_start_ms"),
        "final_matches_eligible": item.get("final_matches_eligible"),
        "incorrect_early_audible_response": item.get("incorrect_early_audible_response"),
        "preview_revision_count": item.get("preview_revision_count"),
        "previews": [{
            "text_sha256": preview.get("text_sha256"),
            "stable_count": preview.get("stable_count"),
            "barge_to_preview_ms": preview.get("barge_to_preview_ms"),
            "eligible": preview.get("eligible"),
            "action": preview.get("action"),
        } for preview in item.get("previews", [])],
    }


def compact_lifecycle(row):
    result = {
        "arm": row.get("arm"), "repetition": row.get("repetition"),
        "transcript_sha256": {name: hashlib.sha256(text.encode()).hexdigest()
                              for name, text in row.get("transcripts", {}).items()},
        "history": row.get("history"),
        "final_response": row.get("final_response"),
        "cleanup": row.get("cleanup"),
    }
    for label in ("first_barge", "second_barge"):
        item = row.get(label, {})
        result[label] = {
            "old_response_id": item.get("old_response_id"),
            "new_response_id": item.get("new_response_id"),
            "played": item.get("played"),
            "old_pcm_sha256": item.get("old_pcm_sha256"),
            "pending_pcm_sha256": item.get("pending_pcm_sha256"),
            "old_response_text_sha256": item.get("old_response_text_sha256"),
            "stale_events": item.get("stale_events"),
            "timings_ms": item.get("timings_ms"),
        }
    return result


def compute_scorecard(reports, *, preregistration_sha256, controls=None,
                      candidate_preregistration_sha256=None):
    require_three = {arm: sorted((row for row in reports if row.get("arm") == arm),
                                key=lambda row: row.get("repetition", 0))
                     for arm in ("before", "after")}
    failures = []
    controls = controls if isinstance(controls, dict) else {}
    for name in REQUIRED_CONTROLS:
        if not isinstance(controls.get(name), dict) or controls[name].get("status") != "PASS":
            failures.append(f"control_{name}_not_passed")
    if any(row.get("preregistration_sha256") != preregistration_sha256 for row in reports):
        failures.append("mixed_preregistration")
    if candidate_preregistration_sha256 is not None:
        if any(row.get("candidate_preregistration_sha256") !=
               candidate_preregistration_sha256 for row in reports):
            failures.append("mixed_candidate_preregistration")
        for row in require_three["after"]:
            for label in ("first_barge", "second_barge"):
                item = row.get("incremental", {}).get(label, {})
                if item.get("incorrect_early_audible_response") is not False:
                    failures.append("incorrect_early_audible_response")
                if item.get("preview_revision_count") != 0:
                    failures.append("frozen_preview_revision")
                if (item.get("eligible_sha256") is not None and
                        item.get("final_matches_eligible") is not True):
                    failures.append("eligible_final_identity_mismatch")
    if len({json.dumps(row.get("harness"), sort_keys=True) for row in reports}) > 1:
        failures.append("mixed_harness")
    if len({json.dumps(row.get("model_manifest_sha256"), sort_keys=True) for row in reports}) > 1:
        failures.append("mixed_models")
    if len({row.get("verifier_sha256") for row in reports}) > 1:
        failures.append("mixed_verifier")
    if len({json.dumps(input_shape(row), sort_keys=True) for row in reports}) > 1:
        failures.append("mixed_input_shape")
    if len({row.get("accepted_parent_commit") for row in reports}) > 1:
        failures.append("mixed_accepted_parent")
    arms = {}
    for arm, rows in require_three.items():
        if [row.get("repetition") for row in rows] != [1, 2, 3]:
            failures.append(f"{arm}_requires_repetitions_1_2_3")
        if any(row.get("status") != "PASS" for row in rows):
            failures.append(f"{arm}_raw_verification_failed")
        if len({row.get("execution_commit") for row in rows}) > 1:
            failures.append(f"{arm}_mixed_execution_commit")
        if len({json.dumps(row.get("binaries"), sort_keys=True) for row in rows}) > 1:
            failures.append(f"{arm}_mixed_binaries")
        metrics = {}
        for metric in METRICS:
            samples = [row["first_barge"]["timings_ms"][metric] for row in rows]
            metrics[metric] = summarize(samples)
        second_cancel = summarize([row["second_barge"]["timings_ms"]["barge_to_cancel"]
                                   for row in rows])
        second_full = summarize([
            row["second_barge"]["timings_ms"]["replay_inclusive_barge_to_new_audio"]
            for row in rows])
        incremental_metrics = {}
        if candidate_preregistration_sha256 is not None:
            for label in ("first_barge", "second_barge"):
                incremental_metrics[label] = {}
                for metric in ("barge_to_eligible_ms", "barge_to_vlm_start_ms",
                               "barge_to_tts_start_ms", "eligible_to_vlm_start_ms",
                               "eligible_to_tts_start_ms"):
                    samples = [row.get("incremental", {}).get(label, {}).get(metric)
                               for row in rows]
                    samples = [value for value in samples if value is not None]
                    incremental_metrics[label][metric] = summarize(samples) if samples else None
        thresholds = {
            "barge_to_cancel_max_strictly_below_300": threshold(metrics["barge_to_cancel"], 300),
            "cancel_to_digital_silence_max_strictly_below_300": threshold(
                metrics["cancel_to_digital_silence"], 300),
            "asr_final_to_new_audio_max_strictly_below_800": threshold(
                metrics["asr_final_to_new_audio"], 800),
            "replay_inclusive_barge_to_new_audio_max_strictly_below_800": threshold(
                metrics["replay_inclusive_barge_to_new_audio"], 800),
            "second_barge_to_cancel_max_strictly_below_300": threshold(second_cancel, 300),
        }
        for name, result in thresholds.items():
            if arm == "after" and not result["pass"]:
                failures.append(name)
        arms[arm] = {
            "metrics": metrics,
            "second_barge_to_cancel": second_cancel,
            "second_replay_inclusive_barge_to_new_audio_diagnostic": second_full,
            "incremental_metrics": incremental_metrics,
            "thresholds": thresholds,
            "runs": [{
                "repetition": row["repetition"],
                "raw_frames_sha256": row["raw_frames_sha256"],
                "independent_report_sha256": row.get("report_file_sha256", row.get("report_identity")),
            } for row in rows],
        }
    pairs = [exact_pair(before, after)
             for before, after in zip(require_three["before"], require_three["after"])]
    exact_ok = len(pairs) == 3 and all(pair["pass"] for pair in pairs)
    if not exact_ok:
        failures.append("exact_pairing")
    if "after" in arms and "before" in arms:
        try:
            improved = (arms["after"]["metrics"]["replay_inclusive_barge_to_new_audio"]["median_ms"] <
                        arms["before"]["metrics"]["replay_inclusive_barge_to_new_audio"]["median_ms"])
            if not improved:
                failures.append("full_path_median_not_improved")
        except KeyError:
            failures.append("missing_full_path_metric")
    if any(not row.get("binaries") for row in reports):
        failures.append("missing_binary_identity")
    elif require_three["before"] and require_three["after"]:
        before_binary = require_three["before"][0]["binaries"]
        after_binary = require_three["after"][0]["binaries"]
        if before_binary == after_binary:
            failures.append("same_source_diagnostic_A/A")
    result = {
        "schema": "umminfer-ns04-replay-lifecycle-scorecard/v1",
        "preregistration_sha256": preregistration_sha256,
        "candidate_preregistration_sha256": candidate_preregistration_sha256,
        "promotion_status": "NOT_PROMOTED" if failures else "PROMOTED",
        "failures": sorted(set(failures)),
        "arms": arms,
        "exact_pairing": {"pass": exact_ok, "pairs": pairs},
        "incremental_by_run": [{
            "arm": row.get("arm"), "repetition": row.get("repetition"),
            "first_barge": compact_incremental(row, "first_barge"),
            "second_barge": compact_incremental(row, "second_barge"),
        } for row in sorted(reports, key=lambda item: (item.get("arm"), item.get("repetition")))]
            if candidate_preregistration_sha256 is not None else [],
        "lifecycle_by_run": [compact_lifecycle(row) for row in sorted(
            reports, key=lambda item: (item.get("arm"), item.get("repetition")))]
            if candidate_preregistration_sha256 is not None else [],
        "source_identity": {
            "accepted_parent_commit": require_three["before"][0].get("accepted_parent_commit")
                if require_three["before"] else None,
            "execution_commit_by_arm": {arm: rows[0].get("execution_commit")
                                        for arm, rows in require_three.items() if rows},
            "harness": require_three["before"][0].get("harness")
                if require_three["before"] else None,
            "verifier_sha256": require_three["before"][0].get("verifier_sha256")
                if require_three["before"] else None,
            "binaries_by_arm": {arm: rows[0].get("binaries")
                                for arm, rows in require_three.items() if rows},
            "model_manifest_sha256": require_three["before"][0].get("model_manifest_sha256")
                if require_three["before"] else None,
            "candidate_runtime_changed": bool(require_three["before"] and require_three["after"] and
                require_three["before"][0].get("binaries") != require_three["after"][0].get("binaries")),
        },
        "evidence_policy": "Raw PCM, wire events, logs and media stay in the caller-selected external bundle; tracked scorecard contains compact hashes/results only.",
        "controls": controls or {},
        "boundaries": {
            "physical_microphone": "NOT_RUN",
            "physical_speaker": "NOT_RUN",
            "acoustic_echo": "NOT_RUN",
            "perceptual_quality": "NOT_RUN",
            "training": "NOT_RUN",
            "optimizer": "NOT_RUN",
            "resident_service": "NOT_RUN",
            "deployment": "NOT_RUN",
            "audit_replay": "post-run independent raw decode only; never public audio",
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs=3, action="append", required=True,
                        metavar=("ARM", "REPETITION", "REPORT"))
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--candidate-preregistration", type=Path)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("scorecard output must be new")
    reports = []
    for arm, repetition, path in args.run:
        report_path = Path(path)
        raw = report_path.read_bytes()
        row = json.loads(raw)
        if row.get("arm") != arm or row.get("repetition") != int(repetition):
            raise SystemExit(f"arm/repetition mismatch in {report_path}")
        row["report_file_sha256"] = hashlib.sha256(raw).hexdigest()
        reports.append(row)
    prereg_raw = args.preregistration.read_bytes()
    candidate_raw = (args.candidate_preregistration.read_bytes()
                     if args.candidate_preregistration else None)
    controls = json.loads(args.controls.read_text()) if args.controls else None
    result = compute_scorecard(
        reports, preregistration_sha256=hashlib.sha256(prereg_raw).hexdigest(), controls=controls,
        candidate_preregistration_sha256=(hashlib.sha256(candidate_raw).hexdigest()
                                         if candidate_raw else None))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"{result['promotion_status']} NS-04 replay-inclusive scorecard: "
          f"failures={result['failures']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
