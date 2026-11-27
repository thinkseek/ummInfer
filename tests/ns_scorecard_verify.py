#!/usr/bin/env python3
"""Independent fresh-process verifier for an external unified NS scorecard."""

import argparse
import hashlib
import json
import math
from pathlib import Path

CASES = [f"NS-{index:02d}" for index in range(1, 9)]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def numbers_close(left, right):
    if isinstance(left, int) and isinstance(right, int):
        return left == right
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-9)


def verify_arm(scorecard, root, case_id, side):
    case = next(item for item in scorecard["cases"] if item["case_id"] == case_id)
    arm = case[side]
    if arm.get("status") == "NOT_RUN":
        require(arm.get("result_sha256") is None, f"{side} {case_id} NOT_RUN has a result hash")
        return
    relative = Path("arms") / side / (case_id + ".json")
    path = root / relative
    require(path.is_file(), f"missing sealed {relative}")
    require(sha(path) == arm.get("result_sha256"), f"{relative} digest drift")
    source = arm.get("source_identity", {})
    require(len(source.get("commit", "")) == 40, f"{relative} lacks an exact source commit")
    for name in ("registry_sha256", "harness_sha256"):
        value = source.get(name)
        if value is not None:
            require(isinstance(value, str) and len(value) == 64 and
                    all(c in "0123456789abcdef" for c in value),
                    f"{relative} malformed {name}")
    for failure in arm.get("failures", []):
        require(isinstance(failure, str) and failure, f"{relative} has an empty failure")


def verify_full_gate(case):
    arm = case.get("after", {})
    gate = case.get("north_star_hard_gate")
    require(gate in {"PASS", "FAIL", "NOT_RUN", "NOT_COVERED"}, "invalid hard-gate state")
    if arm.get("status") == "NOT_RUN":
        require(gate == "NOT_RUN", f"{case['case_id']} missing arm did not remain NOT_RUN")
        return
    if gate != "PASS":
        return
    require(arm.get("status") == "PASS" and arm.get("declared_full_gate_status") == "PASS",
            f"{case['case_id']} full PASS lacks a passing arm declaration")
    coverage = arm.get("coverage", {})
    require(coverage.get("independent_verification") == "PASS",
            f"{case['case_id']} full PASS lacks independent verification")
    if case["case_id"] == "NS-07":
        require(arm.get("production_status") == "PASS" and
                arm.get("deployment_scope") == "resident_5_gpu" and
                coverage.get("resident_5_gpu") == "PASS" and
                coverage.get("owner_authorization") == "PASS",
                "NS-07 PASS lacks authorized resident five-GPU evidence")
    elif case["case_id"] == "NS-08":
        require(all(coverage.get(name) == "PASS" for name in
                    ("web_real", "macos_real", "human_walkthrough", "lossless_round_trip")),
                "NS-08 PASS lacks real Web/macOS/human/lossless evidence")
    else:
        require(coverage.get("all_required") == "PASS",
                f"{case['case_id']} PASS lacks complete required coverage")


def verify_comparison(case_id, paired):
    if paired is None or paired.get("status") == "NOT_RUN":
        return
    require(isinstance(paired.get("no_regression"), bool), f"{case_id} lacks no_regression")
    deltas = paired.get("after_minus_before", {})
    require(isinstance(deltas, dict) and deltas, f"{case_id} lacks paired deltas")
    require(all(isinstance(value, (int, float)) and not isinstance(value, bool) and
                math.isfinite(value) for value in deltas.values()),
            f"{case_id} has a non-finite delta")
    if case_id == "NS-08":
        before = paired.get("before_visible_developer_controls")
        after = paired.get("after_visible_developer_controls")
        require(isinstance(before, (int, float)) and isinstance(after, (int, float)),
                "NS-08 lacks visible-control totals")
        if paired["status"] == "PAIRED_ENGINEERING_IMPROVEMENT":
            require(after == 0 and after < before and paired["no_regression"],
                    "NS-08 improvement claim is not supported")
    if case_id == "NS-07" and paired["status"] == "PAIRED_NON_REGRESSION":
        require(paired["no_regression"] and not paired["failures"],
                "NS-07 non-regression claim retains failures")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path,
                        help="verification report, normally outside the sealed root")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    scorecard_path = args.scorecard.resolve()
    manifest_path = args.manifest.resolve()
    require(scorecard_path == root / "ns-scorecard.json", "scorecard is outside manifest root")
    require(manifest_path == root / "SHA256SUMS", "manifest is outside manifest root")
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    require(scorecard.get("schema") == "umminfer-ns-scorecard/v1", "unknown scorecard schema")
    require(scorecard.get("case_order") == CASES, "case order is incomplete or unstable")
    require([case.get("case_id") for case in scorecard.get("cases", [])] == CASES,
            "scorecard silently omitted or reordered a North-Star case")
    by_id = {case["case_id"]: case for case in scorecard["cases"]}
    require(scorecard.get("resident_production_ns07") == by_id["NS-07"]["north_star_hard_gate"],
            "resident production status disagrees with the NS-07 hard gate")
    require(scorecard.get("macos_and_human_ns08") == by_id["NS-08"]["north_star_hard_gate"],
            "macOS/human status disagrees with the NS-08 hard gate")
    expected_update = (scorecard.get("candidate_promotion") == "PROMOTED_NORTH_STAR" and
                       all(case["north_star_hard_gate"] == "PASS" for case in scorecard["cases"]))
    require(scorecard.get("scoreboard_update_allowed") is expected_update,
            "scoreboard-update permission is inconsistent with complete hard-gate evidence")
    for case in scorecard["cases"]:
        verify_full_gate(case)
    for case_id in ("NS-07", "NS-08"):
        verify_arm(scorecard, root, case_id, "before")
        verify_arm(scorecard, root, case_id, "after")
        case = by_id[case_id]
        verify_comparison(case_id, case.get("paired"))
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    listed = {}
    for line in lines:
        digest, separator, name = line.partition("  ")
        require(separator and len(digest) == 64 and
                all(c in "0123456789abcdef" for c in digest), "malformed manifest row")
        require(name and name != "SHA256SUMS" and not Path(name).is_absolute() and
                ".." not in Path(name).parts, f"unsafe manifest path: {name}")
        require(name not in listed, f"duplicate manifest path: {name}")
        listed[name] = digest
    actual = {path.relative_to(root).as_posix(): path for path in root.rglob("*")
              if path.is_file() and path != manifest_path}
    require(set(listed) == set(actual),
            f"manifest coverage mismatch: missing={sorted(set(actual)-set(listed))}, extra={sorted(set(listed)-set(actual))}")
    for name, path in actual.items():
        require(sha(path) == listed[name], f"manifest digest mismatch: {name}")
    report = {
        "schema": "umminfer-ns-scorecard-independent-verification/v1",
        "status": "PASS",
        "case_count": len(CASES),
        "case_statuses": {case["case_id"]: case["north_star_hard_gate"]
                          for case in scorecard["cases"]},
        "candidate_promotion": scorecard.get("candidate_promotion"),
        "scorecard_sha256": sha(scorecard_path),
        "manifest_sha256": sha(manifest_path),
        "manifest_file_count": len(listed),
        "verifier_sha256": sha(Path(__file__).resolve()),
        "producer_imported": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
