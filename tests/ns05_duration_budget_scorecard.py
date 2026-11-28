#!/usr/bin/env python3
"""Build the compact tracked Task #88 NS-05 Before/After scorecard."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_answers(quality):
    result = {}
    for key, cell in quality.items():
        unique = {}
        for answer in cell["answers"]:
            identity = (answer["text"], answer["sha256"], answer["pcm_sha256"],
                        answer["pcm_bytes"], answer["hit"])
            unique[identity] = unique.get(identity, 0) + 1
        result[key] = {"hits": cell["hits"], "total": cell["total"],
            "misses": cell["misses"], "unique_raw_answers": [
                {"text": identity[0], "answer_sha256": identity[1],
                 "pcm_sha256": identity[2], "pcm_bytes": identity[3],
                 "expected_hit": identity[4], "count": count}
                for identity, count in unique.items()]}
    return result


def co_create(root, verification):
    result = {}
    for arm in ("before", "after"):
        raw = Path(root) / arm
        barge = load(raw / "barge.json")
        retry = load(raw / "cyberpunk-retry.json")
        phone = load(raw / "phone.json")
        initial = load(raw / "cyberpunk-initial.json")
        scene = load(raw / "scene.json")
        result[arm] = {
            "status": "NOT_MET",
            "failure": "MODEL_EMITTED_VIDEO_SLOT0_SOURCE_SLOT0_SELF_REFERENCE",
            "phone_first_public_voice_ms": (phone["first_audio_monotonic_ns"] -
                phone["started_monotonic_ns"]) / 1e6,
            "retry_first_public_voice_ms": (retry["first_audio_monotonic_ns"] -
                retry["started_monotonic_ns"]) / 1e6,
            "full_barge_to_retry_first_audio_ms": (retry["first_audio_monotonic_ns"] -
                barge["barge_start_monotonic_ns"]) / 1e6,
            "scene_terminal": scene["terminal"]["kind"],
            "initial_cocreation_terminal": initial["terminal"]["kind"],
            "retry_terminal": retry["terminal"]["kind"],
            "retry_asr_transcripts": retry["asr_transcripts"],
            "retry_text_sha256": retry["text_sha256"],
            "image_artifacts": 0, "video_artifacts": 0, "native_wan_submits": 0,
            "native_pcm_joins": verification[arm]["native_pcm_joins"],
            "cancel_retry_exact": verification[arm]["cancel_retry_exact"],
            "self_reference_fail_closed": verification[arm]["self_reference_fail_closed"],
            "stale_or_cross_session_events": verification[arm]["stale_or_cross_session_events"],
            "final_in_flight": verification[arm]["final_in_flight"],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-bundle", required=True)
    parser.add_argument("--cocreation-bundle", required=True)
    parser.add_argument("--controls", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paired = Path(args.paired_bundle)
    cocreation = Path(args.cocreation_bundle)
    analysis = load(paired / "analysis.json")
    verification = load(paired / "independent-verification.json")
    negative = load(cocreation / "independent-verification.json")
    controls = load(args.controls)
    runs = {name: load(paired / "live" / name / "mode-result.json")
            for name in ("baseline-A", "candidate-B")}
    thresholds = []
    for cell, metrics in analysis["distributions"].items():
        for metric in ("first_text", "first_voice"):
            p95 = metrics[metric]["p95_ms"]
            thresholds.append({"cell": cell, "metric": metric, "p95_ms": p95,
                               "threshold_ms": 3000, "passed": p95 < 3000})
    scorecard = {
        "schema": "umminfer-ns05-duration-budget-voice-cocreation-v1",
        "experiment": analysis["experiment"],
        "decision": "NOT_PROMOTED",
        "final_default_runtime_candidate": "REMOVED",
        "candidate_execution_commit": "feb8e9721d35ee9b44eb4d768e6fb2898a6d5988",
        "candidate_removal_commit": "038842b443cf5fdb3014d7a29a8478456ea84a82",
        "baseline_execution_commit": "e084d8404b1d96bee5cd6f83d8299c2498a3af65",
        "physical_gpu": 0,
        "source": load(paired / "preregistration.json")["source"],
        "identities": {name: {"source_commit": run["source_commit"],
            "source_tree_sha256": run["source_tree_sha256"], "binary": run["binary"],
            "model_config": run["model_config"]} for name, run in runs.items()},
        "paired": {
            "requests": analysis["requests"],
            "same_arm_repeat_identity": verification["same_arm_repeat_identity"],
            "quality": unique_answers(analysis["quality"]),
            "quality_outcomes": analysis["quality_outcomes"],
            "distributions": analysis["distributions"],
            "first_response_thresholds": thresholds,
            "threshold_failures": analysis["threshold_failures"],
            "regressions_over_1_percent_diagnostic": analysis["regressions_over_1_percent_diagnostic"],
            "quality_pass": analysis["quality_pass"],
            "candidate_promotion": analysis["candidate_promotion"],
            "analysis_sha256": sha(paired / "analysis.json"),
            "independent_verification_sha256": sha(paired / "independent-verification.json"),
            "manifest_sha256": sha(paired / "SHA256SUMS"),
        },
        "voice_cocreation": {
            "status": "NOT_MET_IN_BOTH_ARMS",
            "arms": co_create(cocreation, negative),
            "independent_verification_sha256": sha(cocreation / "independent-verification.json"),
            "manifest_sha256": sha(cocreation / "SHA256SUMS"),
        },
        "controls": controls,
        "human_perceptual_physical_acoustic_full_production": "NOT_RUN",
        "raw_media_and_per_event_blobs": "EXTERNAL_ONLY",
    }
    if scorecard["paired"]["candidate_promotion"] or not scorecard["paired"]["threshold_failures"]:
        raise ValueError("scorecard cannot hide an unexpected promotion/threshold result")
    Path(args.output).write_bytes(json.dumps(scorecard, indent=2, sort_keys=True,
        ensure_ascii=False, allow_nan=False).encode() + b"\n")
    print(json.dumps({"decision": scorecard["decision"],
        "voice_cocreation": scorecard["voice_cocreation"]["status"],
        "output": args.output}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
