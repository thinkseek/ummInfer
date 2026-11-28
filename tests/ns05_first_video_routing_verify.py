#!/usr/bin/env python3
"""Independently compare frozen NS-03 cases for the NS-05 director correction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
import ns03_routing_verify as raw  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def verify_arm(root, suite, ffmpeg):
    root = Path(root).resolve()
    report = read(root / "report.json")
    identity = read(root / "run-identity.json")
    require(identity["physical_gpu"] == 7 and identity["baseline_profile"] == "task88",
            "Task 88/GPU 7 identity")
    require(report["health_final"]["in_flight"] == 0, "final in_flight")
    require(all(report["gpu1"][point]["index"] == 7
                and report["gpu1"][point]["memory_used_mib"] <= 16
                and report["gpu1"][point]["utilization_percent"] == 0
                for point in ("before", "after_cleanup")), "GPU cleanup")
    processes = report["processes"]
    require(processes and all(row.get("returncode") in (0, -15)
                              and row.get("stop_monotonic_ns") is not None
                              and row.get("stop_error") is None
                              for row in processes), "owned processes did not exit cleanly")
    for name, binary in identity["binaries"].items():
        path = Path(binary["path"])
        require(raw.sha_file(path) == binary["sha256"]
                and path.stat().st_size == binary["bytes"], f"binary identity: {name}")
    require(raw.sha_file(identity["flux_stage"]["path"])
            == identity["flux_stage"]["sha256"], "FLUX stage identity")
    reports = {row["case_id"]: row for row in report["cases"]}
    results = []
    for original in suite["cases"]:
        if original["population"] != identity["population"]:
            continue
        case = dict(original)
        case["_expected"] = [suite["classes"][case["class"]]["images"],
                             suite["classes"][case["class"]]["videos"]]
        answer = raw.raw_case(case, root, "answer", ffmpeg)
        raw.compare_public_record(answer, reports[case["id"]], case["id"])
        setup = None
        if case["class"] == "followup":
            setup = raw.raw_case(case, root, "setup", ffmpeg)
            raw.compare_public_record(setup, reports[case["id"]]["setup"], case["id"] + "/setup")
        result = raw.evaluate(case, answer, setup)
        result["_case"], result["_record"], result["_setup"] = case, answer, setup
        results.append(result)
    require(len(results) == 20, "frozen population cardinality")
    return {
        "execution_commit": identity["execution_commit"],
        "source_sha256": identity["source_sha256"],
        "binary_sha256": {name: row["sha256"] for name, row in identity["binaries"].items()},
        "models": identity["models"],
        "python_runtime": identity["python_runtime"],
        "results": results,
    }


def compact(results):
    aggregate = raw.aggregate(results, "all")
    aggregate["empty_prose_cases"] = sum(not row["_record"]["clean_text"].strip() for row in results)
    aggregate["cases_detail"] = [{
        "id": row["_case"]["id"], "class": row["_case"]["class"],
        "routing_correct": row["routing_correct"],
        "redundant_visual": row["redundant_visual_case"],
        "empty_prose": not row["_record"]["clean_text"].strip(),
        "voice_present": row["voice_present"],
        "public_leg_errors": len(row["_record"]["leg_errors"]),
        "clean_text_sha256": row["_record"]["clean_text_sha256"],
        "raw_response_sha256": row["_record"]["raw_response_sha256"],
        "image_slots": sum(slot["media_type"] == "image" for slot in row["_record"]["slots"]),
        "video_slots": sum(slot["media_type"] == "video" for slot in row["_record"]["slots"]),
        "followup_correct": row.get("followup_correct"),
    } for row in results]
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--development-before", required=True)
    parser.add_argument("--development-after", required=True)
    parser.add_argument("--heldout-before", required=True)
    parser.add_argument("--heldout-after", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    try:
        suite = read(args.suite)
        arms = {
            "development/before": verify_arm(args.development_before, suite, args.ffmpeg),
            "development/after": verify_arm(args.development_after, suite, args.ffmpeg),
            "heldout/before": verify_arm(args.heldout_before, suite, args.ffmpeg),
            "heldout/after": verify_arm(args.heldout_after, suite, args.ffmpeg),
        }
        require(len(args.base_commit) == len(args.execution_commit) == 40, "full commit ids")
        for key, arm in arms.items():
            expected = args.base_commit if key.endswith("/before") else args.execution_commit
            require(arm["execution_commit"] == expected, f"execution identity: {key}")
        native_reference = None
        model_cache = {}
        for key, arm in arms.items():
            for model_name, model in arm["models"].items():
                model_root = Path(model["root"])
                cache_key = str(model_root)
                if cache_key not in model_cache:
                    files = {}
                    for path in sorted(p for p in model_root.rglob("*") if p.is_file()):
                        relative = path.relative_to(model_root).as_posix()
                        files[relative] = {"sha256": raw.sha_file(path), "bytes": path.stat().st_size}
                    model_cache[cache_key] = files
                require(model_cache[cache_key] == model["files"]
                        and model["tree_sha256"] == raw.sha_bytes(raw.canonical(model["files"])),
                        f"model tree identity: {key}/{model_name}")
            signature = {"models": arm["models"], "python_runtime": arm["python_runtime"],
                         "native": {name: digest for name, digest in arm["binary_sha256"].items()
                                    if name != "server"}}
            if native_reference is None:
                native_reference = signature
            else:
                require(signature == native_reference, f"native/model identity drift: {key}")
        summaries = {key: compact(arm["results"]) for key, arm in arms.items()}
        comparisons = {}
        non_regression = True
        for population in ("development", "heldout"):
            before, after = summaries[f"{population}/before"], summaries[f"{population}/after"]
            checks = {
                "routing_accuracy_not_reduced": after["routing_accuracy"] >= before["routing_accuracy"],
                "redundant_visual_cases_not_increased": after["redundant_visual_cases"] <= before["redundant_visual_cases"],
                "empty_prose_cases_not_increased": after["empty_prose_cases"] <= before["empty_prose_cases"],
                "voice_missing_cases_not_increased": after["voice_missing_cases"] <= before["voice_missing_cases"],
                "public_leg_errors_not_increased": after["public_leg_error_count"] <= before["public_leg_error_count"],
                "followup_correct_not_reduced": after["followup_correct"] >= before["followup_correct"],
            }
            comparisons[population] = checks
            non_regression = non_regression and all(checks.values())
        before_all = arms["development/before"]["results"] + arms["heldout/before"]["results"]
        after_all = arms["development/after"]["results"] + arms["heldout/after"]["results"]
        overall = {"before": compact(before_all), "after": compact(after_all)}
        overall_checks = {
            "routing_accuracy_not_reduced": overall["after"]["routing_accuracy"] >= overall["before"]["routing_accuracy"],
            "redundant_visual_cases_not_increased": overall["after"]["redundant_visual_cases"] <= overall["before"]["redundant_visual_cases"],
            "empty_prose_cases_not_increased": overall["after"]["empty_prose_cases"] <= overall["before"]["empty_prose_cases"],
            "voice_missing_cases_not_increased": overall["after"]["voice_missing_cases"] <= overall["before"]["voice_missing_cases"],
            "public_leg_errors_not_increased": overall["after"]["public_leg_error_count"] <= overall["before"]["public_leg_error_count"],
            "followup_correct_not_reduced": overall["after"]["followup_correct"] >= overall["before"]["followup_correct"],
        }
        non_regression = non_regression and all(overall_checks.values())
        result = {"schema_version": 1, "experiment": "umminfer_ns05_first_video_ns03_non_regression_v1",
                  "status": "PASS" if non_regression else "FAIL", "physical_gpu": 7,
                  "base_commit": args.base_commit, "execution_commit": args.execution_commit,
                  "suite_canonical_sha256": raw.sha_bytes(raw.canonical(suite)),
                  "arm_identities": {key: {
                      "execution_commit": arm["execution_commit"], "source_sha256": arm["source_sha256"],
                      "binary_sha256": arm["binary_sha256"],
                      "model_tree_sha256": {name: model["tree_sha256"] for name, model in arm["models"].items()}}
                      for key, arm in arms.items()},
                  "arms": summaries, "overall": overall, "population_checks": comparisons,
                  "overall_checks": overall_checks, "non_regression": non_regression,
                  "perceptual_quality": "NOT_RUN"}
        Path(args.output).write_bytes(json.dumps(result, indent=2, sort_keys=True,
            ensure_ascii=False, allow_nan=False).encode() + b"\n")
        print(json.dumps({"status": result["status"], "overall_checks": overall_checks,
                          "before": {key: overall["before"][key] for key in
                                     ("routing_correct", "redundant_visual_cases", "empty_prose_cases", "voice_missing_cases")},
                          "after": {key: overall["after"][key] for key in
                                    ("routing_correct", "redundant_visual_cases", "empty_prose_cases", "voice_missing_cases")}}, indent=2))
        return 0 if non_regression else 1
    except (OSError, ValueError, KeyError, AssertionError) as error:
        print(f"NS03 PAIR VERIFY FAIL: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
