#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Score an After-only frozen NS-03 40-case non-regression control."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from ns03_routing import (  # noqa: E402
    build_aggregates, canonical, evaluate_record, load_suite, sha256_file,
)

BASELINE = {
    "development": {"routing_correct": 15, "redundant_visual_cases_max": 1,
                    "followup_correct": 2,
                    "classes": {"text": 5, "static": 2, "dynamic": 3, "followup": 5}},
    "heldout": {"routing_correct": 17, "redundant_visual_cases_max": 0,
                "followup_correct": 2,
                "classes": {"text": 5, "static": 4, "dynamic": 3, "followup": 5}},
    "all": {"routing_correct": 32, "redundant_visual_cases_max": 1,
            "followup_correct": 4, "voice_missing_cases_max": 2,
            "public_leg_error_count_max": 160,
            "classes": {"text": 10, "static": 6, "dynamic": 6, "followup": 10}},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    suite = load_suite(args.suite)
    candidate = json.loads(args.candidate.read_text())
    reports = {"development": json.loads((args.development / "report.json").read_text()),
               "heldout": json.loads((args.heldout / "report.json").read_text())}
    results = []
    identities = {}
    for population, report in reports.items():
        require_ids = {case["id"] for case in suite["cases"] if case["population"] == population}
        records = {record["case_id"]: record for record in report["cases"]}
        if set(records) != require_ids:
            raise AssertionError(f"{population} case set differs from frozen suite")
        run_dir = args.development if population == "development" else args.heldout
        identity = json.loads((run_dir / "run-identity.json").read_text())
        if identity["execution_commit"] != candidate["execution_commit"] \
                or identity["physical_gpu"] != 7 or identity["arm"] != "after" \
                or identity["source_sha256"] != candidate["source_sha256"]:
            raise AssertionError(f"{population} identity does not match the frozen GPU-7 candidate")
        identities[population] = hashlib.sha256(
            (run_dir / "run-identity.json").read_bytes()).hexdigest()
        if report.get("health_final", {}).get("in_flight") != 0:
            raise AssertionError(f"{population} final in_flight is nonzero")
        for case in suite["cases"]:
            if case["population"] == population:
                record = records[case["id"]]
                results.append(evaluate_record(case, record, record.get("setup")))
    aggregates = build_aggregates(results)
    failures = []
    for population, baseline in BASELINE.items():
        overall = aggregates[population]["overall"]
        if overall["routing_correct"] < baseline["routing_correct"]:
            failures.append(f"{population}:routing_correct_regressed")
        if overall["redundant_visual_cases"] > baseline["redundant_visual_cases_max"]:
            failures.append(f"{population}:redundant_visual_regressed")
        if overall["followup_correct"] < baseline["followup_correct"]:
            failures.append(f"{population}:followup_context_regressed")
        for kind, minimum in baseline["classes"].items():
            if aggregates[population]["classes"][kind]["routing_correct"] < minimum:
                failures.append(f"{population}:{kind}:routing_regressed")
    overall = aggregates["all"]["overall"]
    if overall["voice_missing_cases"] > BASELINE["all"]["voice_missing_cases_max"]:
        failures.append("all:voice_missing_regressed")
    if overall["public_leg_error_count"] > BASELINE["all"]["public_leg_error_count_max"]:
        failures.append("all:public_leg_errors_regressed")
    if overall["media_error_cases"] != 0:
        failures.append("all:media_decode_failures")
    misses = [{"case_id": result["case_id"], "population": result["population"],
               "class": result["class"], "routing_correct": result["routing_correct"],
               "end_to_end_correct": result["end_to_end_correct"],
               "redundant_visual_case": result["redundant_visual_case"],
               "decision": result["decision"], "errors": result["errors"],
               "public_leg_errors": result["public_leg_errors"],
               "followup_correct": result.get("followup_correct")}
              for result in results if not result["routing_correct"]
              or not result["end_to_end_correct"] or result["redundant_visual_case"]
              or result.get("followup_correct") is False]
    scorecard = {
        "format": "umminfer-ns03-routing-after-control-v1",
        "suite_sha256": hashlib.sha256(canonical(suite)).hexdigest(),
        "candidate_execution_commit": candidate["execution_commit"],
        "run_identity_sha256": identities,
        "raw_report_sha256": {population: sha256_file(
            (args.development if population == "development" else args.heldout) / "report.json")
            for population in reports},
        "case_results": results, "aggregates": aggregates, "accepted_baseline": BASELINE,
        "failures": failures, "pass": not failures, "misses": misses,
        "routing_accuracy": overall["routing_accuracy"],
        "redundant_visual_rate": overall["redundant_visual_rate"],
        "followup_accuracy": overall["followup_accuracy"],
        "rubric_hard_ideal_90_percent": overall["routing_accuracy"] >= 0.9,
        "physical_perceptual_quality": "NOT_RUN",
    }
    if args.out.exists():
        raise AssertionError(f"output exists: {args.out}")
    args.out.write_text(json.dumps(scorecard, indent=2, sort_keys=True,
                                   ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"pass": not failures, "routing_accuracy": overall["routing_accuracy"],
                      "redundant_visual_rate": overall["redundant_visual_rate"],
                      "followup_accuracy": overall["followup_accuracy"],
                      "failures": failures}, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
