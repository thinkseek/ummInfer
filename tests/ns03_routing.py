#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Frozen NS-03 suite validation, public-request construction, and scoring.

This producer has no serving or ML imports. Live capture and independent raw
verification are separate entry points.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys

FORMAT = "umminfer-ns03-routing-scorecard-v1"
FOLLOWUP_FORMAT = "umminfer-ns03-followup-history-scorecard-v1"
CLASSES = ("text", "static", "dynamic", "followup")
POPULATIONS = ("development", "heldout")
DIRECTIVE = re.compile(r"\[\[\s*(?:image|video|visual)\s*:", re.IGNORECASE)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_suite(path):
    suite = load_json(path)
    if suite.get("format") != "umminfer-ns03-routing-suite-v1":
        raise ValueError("unexpected NS-03 suite format")
    ids = [case.get("id") for case in suite.get("cases", [])]
    if len(ids) != 40 or len(set(ids)) != 40:
        raise ValueError("NS-03 requires exactly 40 unique cases")
    for population in POPULATIONS:
        for kind in CLASSES:
            cases = [case for case in suite["cases"]
                     if case["population"] == population and case["class"] == kind]
            if len(cases) != 5:
                raise ValueError(f"expected five {population}/{kind} cases")
    for case in suite["cases"]:
        expected = suite["classes"][case["class"]]
        if case["class"] == "followup":
            setup = case.get("setup") or {}
            if not setup.get("prompt") or setup.get("expected") != {"images": 0, "videos": 0}:
                raise ValueError(f"invalid real-context setup: {case['id']}")
            groups = case.get("context_any")
            if not groups or any(not isinstance(group, list) or not group for group in groups):
                raise ValueError(f"follow-up lacks frozen context checks: {case['id']}")
        if not case.get("prompt") or not case.get("language"):
            raise ValueError(f"incomplete case: {case['id']}")
        if set(expected) != {"images", "videos"}:
            raise ValueError(f"invalid golden cardinality: {case['id']}")
    return suite


def default_request(case, *, model="qwen3-native", session_id, followup=False,
                    initial_history=True):
    """Default public request; legacy capture may omit first-turn history."""
    prompt = case["setup"]["prompt"] if case["class"] == "followup" and not followup else case["prompt"]
    if followup and case["class"] != "followup":
        raise ValueError("follow-up requests require a frozen follow-up case")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0,
        "stream": True,
        "session_id": session_id,
        "image": {"size": "512x512", "output_format": "png"},
        "video": {"size": "704x480", "output_format": "mp4", "seconds": 2,
                  "num_frames": 29, "steps": 25, "guidance_scale": 5.0},
        "narrative": {"enabled": True, "visual_mode": "auto", "max_slots": 3,
                      "allow_text_only": True},
        "audio": {"voice": "vivian", "format": "wav", "stream": True},
        "modalities": ["text", "image", "video", "audio"],
    }
    if initial_history or followup:
        body["session_history"] = True
    return body


def evaluate_record(case, record, setup_record=None):
    expected = {"text": {"images": 0, "videos": 0},
                "static": {"images": 1, "videos": 0},
                "dynamic": {"images": 0, "videos": 1},
                "followup": {"images": 0, "videos": 0}}[case["class"]]
    slots = record.get("slots") or []
    images = sum(slot.get("media_type") == "image" for slot in slots)
    videos = sum(slot.get("media_type") == "video" for slot in slots)
    actual_total = images + videos
    expected_total = expected["images"] + expected["videos"]
    errors = list(record.get("errors") or [])
    terminal_complete = record.get("terminal_kind") == "complete" and record.get("terminal_count") == 1
    exact_cardinality = images == expected["images"] and videos == expected["videos"]
    routing_correct = bool(exact_cardinality and terminal_complete and not errors)
    visual_artifacts = [a for a in record.get("artifacts", [])
                        if a.get("media_type") in ("image", "video")]
    valid_visuals = [a for a in visual_artifacts if a.get("decoded") is True]
    artifact_exact = (len(valid_visuals) == expected_total and
                      len(visual_artifacts) == expected_total and
                      len({a.get("slot_id") for a in valid_visuals}) == expected_total)
    audio = record.get("audio") or {}
    voice = bool(audio.get("artifact_count", 0) > 0 and audio.get("bytes", 0) > 0 and
                 audio.get("pcm16_valid") is True)
    end_to_end = bool(routing_correct and artifact_exact and voice)
    redundant = bool((expected_total == 0 and actual_total > 0) or
                     (expected_total > 0 and actual_total > expected_total))
    redundant_slots = (actual_total if expected_total == 0
                       else max(0, actual_total - expected_total))
    artifact_images = sum(a.get("media_type") == "image" for a in valid_visuals)
    artifact_videos = sum(a.get("media_type") == "video" for a in valid_visuals)
    artifact_total = artifact_images + artifact_videos
    redundant_artifact = bool((expected_total == 0 and artifact_total > 0) or
                              (expected_total > 0 and artifact_total > expected_total))
    missing = max(0, expected_total - actual_total)
    wrong_modality = actual_total == expected_total and not exact_cardinality
    ambiguous = record.get("status") != "captured" or record.get("parse_ambiguous") is True
    result = {
        "case_id": case["id"], "population": case["population"], "class": case["class"],
        "language": case["language"], "expected": expected,
        "decision": {"images": images, "videos": videos, "slots": copy.deepcopy(slots)},
        "artifacts": copy.deepcopy(record.get("artifacts") or []),
        "raw_response_sha256": record.get("raw_response_sha256"),
        "clean_text_sha256": record.get("clean_text_sha256"),
        "request_sha256": record.get("request_sha256"),
        "terminal": {"kind": record.get("terminal_kind"),
                     "count": record.get("terminal_count", 0),
                     "done_count": record.get("done_count", 0)},
        "media_counts": {"visual_artifacts": len(visual_artifacts),
                         "valid_visual_artifacts": len(valid_visuals),
                         "audio_artifacts": audio.get("artifact_count", 0),
                         "audio_bytes": audio.get("bytes", 0)},
        "routing_correct": routing_correct,
        "end_to_end_correct": end_to_end,
        "voice_present": voice,
        "redundant_visual_case": redundant,
        "redundant_image_case": images > expected["images"],
        "redundant_video_case": videos > expected["videos"],
        "redundant_visual_slots": redundant_slots,
        "redundant_artifact_case": redundant_artifact,
        "missing_visual_slots": missing,
        "wrong_modality": wrong_modality,
        "ambiguous": ambiguous,
        "errors": errors,
        "public_leg_errors": copy.deepcopy(record.get("leg_errors") or []),
        "media_errors": copy.deepcopy(record.get("media_errors") or []),
        "all_output_legs_completed": not (record.get("leg_errors") or []),
        "clean_text_present": bool(record.get("clean_text", "").strip()),
    }
    if case["class"] == "followup":
        text = record.get("clean_text", "")
        folded = text.casefold()
        groups = []
        for group in case["context_any"]:
            hits = [token for token in group if token.casefold() in folded]
            groups.append({"accepted": group, "matched": hits, "correct": bool(hits)})
        setup_ok = bool(setup_record and setup_record.get("terminal_kind") == "complete" and
                        not setup_record.get("errors"))
        history = record.get("session_history") is True and record.get("same_session_as_setup") is True
        result["followup"] = {
            "setup_complete": setup_ok, "real_history_requested": history,
            "setup_raw_response_sha256": setup_record.get("raw_response_sha256") if setup_record else None,
            "setup_clean_text_sha256": setup_record.get("clean_text_sha256") if setup_record else None,
            "setup_request_sha256": setup_record.get("request_sha256") if setup_record else None,
            "context_groups": groups,
            "context_correct": all(group["correct"] for group in groups),
        }
        result["followup_correct"] = bool(end_to_end and setup_ok and history and
                                           result["followup"]["context_correct"])
        # Full response text is external evidence only; the scorecard stores hashes.
    result.pop("clean_text", None)
    return result


def wilson(successes, total, z=1.959963984540054):
    if not total:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - radius, center + radius]


def aggregate(results, population, kind=None):
    selected = [r for r in results if (population == "all" or r["population"] == population)
                and (kind is None or r["class"] == kind)]
    total = len(selected)
    correct = sum(r["routing_correct"] for r in selected)
    e2e = sum(r["end_to_end_correct"] for r in selected)
    redundant = sum(r["redundant_visual_case"] for r in selected)
    redundant_artifacts = sum(r["redundant_artifact_case"] for r in selected)
    expected_visuals = sum(r["expected"]["images"] + r["expected"]["videos"] for r in selected)
    followups = [r for r in selected if r["class"] == "followup"]
    return {
        "cases": total, "routing_correct": correct,
        "routing_accuracy": correct / total if total else None,
        "routing_accuracy_wilson_95": wilson(correct, total),
        "end_to_end_correct": e2e,
        "end_to_end_accuracy": e2e / total if total else None,
        "redundant_visual_cases": redundant,
        "redundant_visual_rate": redundant / total if total else None,
        "redundant_image_cases": sum(r["redundant_image_case"] for r in selected),
        "redundant_image_rate": sum(r["redundant_image_case"] for r in selected) / total if total else None,
        "redundant_video_cases": sum(r["redundant_video_case"] for r in selected),
        "redundant_video_rate": sum(r["redundant_video_case"] for r in selected) / total if total else None,
        "redundant_visual_slots": sum(r["redundant_visual_slots"] for r in selected),
        "redundant_visual_slot_rate": (sum(r["redundant_visual_slots"] for r in selected) /
                                       expected_visuals if expected_visuals else None),
        "redundant_artifact_cases": redundant_artifacts,
        "redundant_artifact_rate": redundant_artifacts / total if total else None,
        "missing_visual_slots": sum(r["missing_visual_slots"] for r in selected),
        "wrong_modality_cases": sum(r["wrong_modality"] for r in selected),
        "ambiguous_cases": sum(r["ambiguous"] for r in selected),
        "voice_missing_cases": sum(not r["voice_present"] for r in selected),
        "public_leg_error_cases": sum(bool(r["public_leg_errors"]) for r in selected),
        "public_leg_error_count": sum(len(r["public_leg_errors"]) for r in selected),
        "media_error_cases": sum(bool(r["media_errors"]) for r in selected),
        "terminal_count": sum(r["terminal"]["count"] for r in selected),
        "image_count": sum(r["decision"]["images"] for r in selected),
        "video_count": sum(r["decision"]["videos"] for r in selected),
        "valid_image_artifacts": sum(sum(a.get("media_type") == "image" and a.get("decoded") is True
                                         for a in r["artifacts"]) for r in selected),
        "valid_video_artifacts": sum(sum(a.get("media_type") == "video" and a.get("decoded") is True
                                         for a in r["artifacts"]) for r in selected),
        "audio_artifact_count": sum(r["media_counts"]["audio_artifacts"] for r in selected),
        "followup_cases": len(followups),
        "followup_correct": sum(r.get("followup_correct") is True for r in followups),
        "followup_accuracy": (sum(r.get("followup_correct") is True for r in followups) /
                              len(followups) if followups else None),
    }


def exact_mcnemar(before, after):
    by_before = {r["case_id"]: r["routing_correct"] for r in before}
    by_after = {r["case_id"]: r["routing_correct"] for r in after}
    ids = sorted(by_before)
    if set(by_after) != set(by_before):
        raise ValueError("paired arms contain different case ids")
    improved = sum(not by_before[i] and by_after[i] for i in ids)
    worsened = sum(by_before[i] and not by_after[i] for i in ids)
    discordant = improved + worsened
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(0, min(improved, worsened) + 1)) / (2 ** discordant)
        p_value = min(1.0, 2 * tail)
    return {"cases": len(ids), "improved": improved, "worsened": worsened,
            "discordant": discordant, "two_sided_exact_p": p_value,
            "accuracy_delta_percentage_points": 100 * (sum(by_after.values()) - sum(by_before.values())) / len(ids)}


def build_aggregates(results):
    out = {}
    for population in (*POPULATIONS, "all"):
        out[population] = {"overall": aggregate(results, population),
                           "classes": {kind: aggregate(results, population, kind) for kind in CLASSES}}
    return out


def make_scorecard(suite, preregistration, runs, identities, candidate):
    run_results = {}
    raw_records = {}
    for key, report in runs.items():
        population, arm = key.split("/", 1)
        expected_ids = [c["id"] for c in suite["cases"] if c["population"] == population]
        by_id = {r["case_id"]: r for r in report["cases"]}
        if set(by_id) != set(expected_ids):
            raise ValueError(f"{key} does not contain exactly its frozen population")
        results = []
        for case in suite["cases"]:
            if case["population"] != population:
                continue
            record = by_id[case["id"]]
            raw_records[f"{key}/{case['id']}"] = record
            results.append(evaluate_record(case, record, record.get("setup")))
        run_results[key] = results
    all_before = run_results["development/before"] + run_results["heldout/before"]
    all_after = run_results["development/after"] + run_results["heldout/after"]
    scorecard = {
        "format": FORMAT,
        "suite_sha256": sha256_bytes(canonical(suite)),
        "preregistration_sha256": sha256_bytes(canonical(preregistration)),
        "identities": identities, "candidate": candidate,
        "arms": {key: {"aggregate": build_aggregates(results), "cases": results}
                 for key, results in run_results.items()},
        "paired": {
            "development": exact_mcnemar(run_results["development/before"], run_results["development/after"]),
            "heldout": exact_mcnemar(run_results["heldout/before"], run_results["heldout/after"]),
            "all": exact_mcnemar(all_before, all_after),
        },
        "physical_perceptual_quality": "NOT_RUN",
    }
    after_all = scorecard["arms"]["development/after"]["aggregate"]["all"]["overall"]
    # Recompute pooled After rather than averaging the two populations.
    after_all = aggregate(all_after, "all")
    before_all = aggregate(all_before, "all")
    after_heldout = aggregate(run_results["heldout/after"], "heldout")
    before_heldout = aggregate(run_results["heldout/before"], "heldout")
    owner = bool(after_all["routing_accuracy"] >= 0.80 and after_all["redundant_visual_rate"] < 0.05 and
                 after_heldout["routing_accuracy"] >= 0.80 and after_heldout["redundant_visual_rate"] < 0.05)
    gain = scorecard["paired"]["all"]["accuracy_delta_percentage_points"] > 0
    no_heldout_regression = after_heldout["routing_accuracy"] >= before_heldout["routing_accuracy"]
    scorecard["thresholds"] = {
        "owner": {"routing_accuracy_min": 0.80, "redundant_visual_rate_max_exclusive": 0.05,
                  "after_all_met": after_all["routing_accuracy"] >= 0.80 and after_all["redundant_visual_rate"] < 0.05,
                  "after_heldout_met": after_heldout["routing_accuracy"] >= 0.80 and after_heldout["redundant_visual_rate"] < 0.05,
                  "positive_paired_gain": gain, "heldout_non_regression": no_heldout_regression,
                  "all_met": owner and gain and no_heldout_regression},
        "rubric_hard_ideal": {"routing_accuracy_min": 0.90,
                              "after_all_met": after_all["routing_accuracy"] >= 0.90,
                              "after_heldout_met": after_heldout["routing_accuracy"] >= 0.90},
        "before_all": {"routing_accuracy": before_all["routing_accuracy"],
                       "redundant_visual_rate": before_all["redundant_visual_rate"]},
        "after_all": {"routing_accuracy": after_all["routing_accuracy"],
                      "redundant_visual_rate": after_all["redundant_visual_rate"]},
        "before_heldout": {"routing_accuracy": before_heldout["routing_accuracy"],
                           "redundant_visual_rate": before_heldout["redundant_visual_rate"]},
        "after_heldout": {"routing_accuracy": after_heldout["routing_accuracy"],
                          "redundant_visual_rate": after_heldout["redundant_visual_rate"]},
    }
    scorecard["promotion"] = "PROMOTION_ELIGIBLE" if scorecard["thresholds"]["owner"]["all_met"] else "NOT_PROMOTED"
    misses = []
    for arm, results in run_results.items():
        for result in results:
            if (not result["end_to_end_correct"] or result["errors"] or
                    result["public_leg_errors"] or result["media_errors"] or
                    result["redundant_visual_case"] or
                    (result["class"] == "followup" and not result["followup_correct"])):
                misses.append({"arm": arm, "case_id": result["case_id"],
                               "routing_correct": result["routing_correct"],
                               "end_to_end_correct": result["end_to_end_correct"],
                               "redundant_visual_case": result["redundant_visual_case"],
                               "missing_visual_slots": result["missing_visual_slots"],
                               "wrong_modality": result["wrong_modality"],
                               "errors": result["errors"],
                               "public_leg_errors": result["public_leg_errors"],
                               "media_errors": result["media_errors"],
                               "followup_correct": result.get("followup_correct")})
    scorecard["misses"] = misses
    return scorecard


def make_followup_scorecard(suite, preregistration, runs, identities, candidate):
    """Focused real-history evidence; full routing promotion remains separate."""
    if candidate.get("first_turn_history") is not True:
        raise ValueError("focused follow-up After requires candidate first_turn_history=true")
    expected_keys = {"development/before", "development/after", "heldout/before", "heldout/after"}
    if set(runs) != expected_keys:
        raise ValueError("follow-up scorecard requires exactly four paired population/arm reports")
    run_results = {}
    for key in sorted(expected_keys):
        report = runs[key]
        population = key.split("/", 1)[0]
        if report.get("case_class") != "followup":
            raise ValueError(f"{key} is not a followup-only report")
        expected_ids = [c["id"] for c in suite["cases"]
                        if c["population"] == population and c["class"] == "followup"]
        by_id = {r["case_id"]: r for r in report["cases"]}
        if len(by_id) != 5 or set(by_id) != set(expected_ids):
            raise ValueError(f"{key} does not contain exactly its five frozen follow-up cases")
        run_results[key] = [evaluate_record(case, by_id[case["id"]], by_id[case["id"]].get("setup"))
                            for case in suite["cases"]
                            if case["population"] == population and case["class"] == "followup"]
    all_before = run_results["development/before"] + run_results["heldout/before"]
    all_after = run_results["development/after"] + run_results["heldout/after"]

    def history(results):
        correct = sum(result["followup_correct"] is True for result in results)
        return {"cases": len(results), "correct": correct, "accuracy": correct / len(results)}

    before_history = history(all_before)
    after_history = history(all_after)
    gain = after_history["accuracy"] - before_history["accuracy"]
    misses = []
    for arm, results in run_results.items():
        for result in results:
            if (not result["end_to_end_correct"] or result["errors"] or
                    result["public_leg_errors"] or result["media_errors"] or
                    result["redundant_visual_case"] or not result["followup_correct"]):
                misses.append({"arm": arm, "case_id": result["case_id"],
                               "routing_correct": result["routing_correct"],
                               "end_to_end_correct": result["end_to_end_correct"],
                               "redundant_visual_case": result["redundant_visual_case"],
                               "missing_visual_slots": result["missing_visual_slots"],
                               "wrong_modality": result["wrong_modality"],
                               "errors": result["errors"],
                               "public_leg_errors": result["public_leg_errors"],
                               "media_errors": result["media_errors"],
                               "followup_correct": result["followup_correct"]})
    return {
        "format": FOLLOWUP_FORMAT,
        "suite_sha256": sha256_bytes(canonical(suite)),
        "preregistration_sha256": sha256_bytes(canonical(preregistration)),
        "identities": identities, "candidate": candidate,
        "arms": {key: {"aggregate": build_aggregates(results), "cases": results}
                 for key, results in run_results.items()},
        "paired": {
            "development": exact_mcnemar(run_results["development/before"], run_results["development/after"]),
            "heldout": exact_mcnemar(run_results["heldout/before"], run_results["heldout/after"]),
            "all": exact_mcnemar(all_before, all_after),
        },
        "history_gate": {
            "before": before_history, "after": after_history, "target": 0.8,
            "before_to_after_gain": gain,
            "target_met": after_history["accuracy"] >= 0.8,
            "verifiable_gain": gain > 0,
        },
        "promotion": "COMPARATOR_REQUIRED",
        "misses": misses,
        "physical_perceptual_quality": "NOT_RUN",
    }


def freeze(paths):
    suite_path, prereg_path = paths
    suite = load_suite(suite_path)
    prereg = load_json(prereg_path)
    expected = "COMPUTED_BY_FREEZE_COMMAND_BEFORE_FIRST_OUTPUT"
    if prereg.get("suite_sha256") != expected:
        raise ValueError("preregistration is already frozen; refusing to alter it")
    prereg["suite_sha256"] = sha256_bytes(canonical(suite))
    prereg["freeze_rule"] = "Frozen by tests/ns03_routing.py freeze before the first model output; suite and scoring changes are forbidden."
    Path(prereg_path).write_text(json.dumps(prereg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(prereg["suite_sha256"])


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("suite")
    freeze_parser.add_argument("preregistration")
    score = sub.add_parser("scorecard")
    score.add_argument("--suite", required=True)
    score.add_argument("--preregistration", required=True)
    score.add_argument("--run", action="append", nargs=2, metavar=("KEY", "REPORT"), required=True)
    score.add_argument("--identities", required=True)
    score.add_argument("--candidate", required=True)
    score.add_argument("--out", required=True)
    followup_score = sub.add_parser("followup-scorecard")
    followup_score.add_argument("--suite", required=True)
    followup_score.add_argument("--preregistration", required=True)
    followup_score.add_argument("--run", action="append", nargs=2, metavar=("KEY", "REPORT"), required=True)
    followup_score.add_argument("--identities", required=True)
    followup_score.add_argument("--candidate", required=True)
    followup_score.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze((args.suite, args.preregistration))
        return
    suite = load_suite(args.suite)
    preregistration = load_json(args.preregistration)
    if preregistration["suite_sha256"] != sha256_bytes(canonical(suite)):
        raise ValueError("suite differs from its frozen preregistration")
    runs = {key: load_json(path) for key, path in args.run}
    if set(runs) != {"development/before", "development/after", "heldout/before", "heldout/after"}:
        raise ValueError("scorecard requires exactly four paired population/arm reports")
    if args.command == "scorecard":
        result = make_scorecard(suite, preregistration, runs, load_json(args.identities), load_json(args.candidate))
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"promotion": result["promotion"], "thresholds": result["thresholds"],
                          "paired": result["paired"]}, sort_keys=True))
    else:
        result = make_followup_scorecard(
            suite, preregistration, runs, load_json(args.identities), load_json(args.candidate))
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"promotion": result["promotion"], "history_gate": result["history_gate"],
                          "paired": result["paired"]}, sort_keys=True))


if __name__ == "__main__":
    main()
