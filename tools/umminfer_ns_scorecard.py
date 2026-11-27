#!/usr/bin/env python3
"""Run the frozen NS-01..NS-08 contract drivers and build a compact scorecard.

Live evidence is supplied as caller-owned Before/After arm directories. An arm
contains NS-07.json and/or NS-08.json produced by the existing golden-case
adapter. Raw streams, media, logs, and model files never enter the tracked
scorecard. Every case is present even when its live gate is NOT_RUN.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "benchmarks" / "north-star" / "cases.json"
SCHEMA = "umminfer-ns-scorecard/v1"
TERMINAL_STATES = {"PASS", "FAIL", "BLOCKED", "NOT_RUN", "NOT_COVERED"}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def load(path):
    with Path(path).open("r", encoding="utf-8") as source:
        return json.load(source)


def current_source_identity():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        cwd=ROOT, text=True).strip())
    return {"commit": commit, "tracked_runtime_dirty": dirty,
            "registry_sha256": sha_file(REGISTRY),
            "harness_sha256": sha_file(Path(__file__).resolve())}


def require_external_directory(path):
    resolved = Path(path).expanduser().resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError("evidence directory must not be a filesystem root")
    if resolved == ROOT or ROOT in resolved.parents:
        raise ValueError("raw evidence directory must be outside the repository")
    if resolved.exists() and any(resolved.iterdir()):
        raise ValueError("evidence directory must be absent or empty")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def run_contract_driver(case, driver, args, destination):
    path = ROOT / driver
    record = {"driver": driver, "args": args,
              "driver_sha256": None, "status": "NOT_COVERED",
              "exit_code": None, "log": None}
    if not path.is_file():
        record["failure"] = "driver is not present at the frozen repository path"
        return record
    record["driver_sha256"] = sha_file(path)
    output = destination / (driver.replace("/", "_") + ".log")
    command = [sys.executable, "-I", driver, *args]
    try:
        with output.open("wb") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=900, check=False)
        record["exit_code"] = result.returncode
        record["status"] = "PASS" if result.returncode == 0 else "FAIL"
    except subprocess.TimeoutExpired:
        record["status"] = "FAIL"
        record["failure"] = "frozen 900-second contract-driver timeout"
    except OSError as error:
        record["status"] = "FAIL"
        record["failure"] = "driver launch failed: " + type(error).__name__
    if output.exists():
        record["log"] = {"file": str(output.relative_to(destination.parents[1])),
                         "bytes": output.stat().st_size, "sha256": sha_file(output)}
    return record


def clean_projection(value, key=""):
    """Copy only compact evidence; reject paths, media, and non-finite values."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not (-1e300 < value < 1e300):
            raise ValueError("non-finite arm metric")
        return value
    if isinstance(value, str):
        lowered = key.lower()
        if value.startswith("/") or lowered in {"path", "root", "dir", "directory", "url"}:
            raise ValueError("absolute paths and endpoint URLs do not belong in the compact scorecard")
        if len(value) > 4096:
            raise ValueError("oversized arm string")
        return value
    if isinstance(value, list):
        return [clean_projection(item, key) for item in value]
    if isinstance(value, dict):
        blocked = {"raw_sse", "raw_bytes", "base64", "audio", "video_bytes", "image_bytes"}
        return {str(k): clean_projection(v, str(k)) for k, v in value.items()
                if str(k).lower() not in blocked}
    raise ValueError("unsupported arm evidence type")


def project_arm(arm_root, case):
    case_id = case["case_id"]
    path = Path(arm_root) / (case_id + ".json")
    if not path.is_file():
        return {"status": "NOT_RUN", "reason": "no normalized arm result was supplied",
                "result_sha256": None}
    raw = load(path)
    if raw.get("case_id") != case_id:
        raise ValueError(f"{path.name} has the wrong case_id")
    allowed = {"case_id", "schema", "arm", "engineering_status", "production_status",
               "full_gate_status", "metrics", "hard_gates", "failures", "limitations",
               "source_identity", "binary_identities", "model_identities", "config_identity",
               "raw_evidence", "coverage", "deployment_scope"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"{path.name} contains unapproved scorecard fields: {unknown}")
    status = raw.get("engineering_status", "NOT_RUN")
    if status not in TERMINAL_STATES | {"ISOLATED_PASS", "CONTRACT_PASS", "NOT_PROMOTED"}:
        raise ValueError(f"invalid engineering status: {status}")
    metrics = raw.get("metrics", {})
    projected_metrics = {}
    for name in case["arm_metrics"]:
        if name not in metrics:
            raise ValueError(f"{path.name} omits required metric {name}")
        value = metrics[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{path.name} metric {name} is not numeric")
        projected_metrics[name] = clean_projection(value, name)
    projected = {
        "status": status,
        "declared_full_gate_status": raw.get("full_gate_status", "NOT_RUN"),
        "production_status": raw.get("production_status", "NOT_RUN"),
        "deployment_scope": raw.get("deployment_scope", "UNSPECIFIED"),
        "source_identity": clean_projection(raw.get("source_identity", {})),
        "binary_identities": clean_projection(raw.get("binary_identities", {})),
        "model_identities": clean_projection(raw.get("model_identities", {})),
        "config_identity": clean_projection(raw.get("config_identity", {})),
        "metrics": projected_metrics,
        "hard_gates": clean_projection(raw.get("hard_gates", {})),
        "failures": clean_projection(raw.get("failures", [])),
        "limitations": clean_projection(raw.get("limitations", [])),
        "coverage": clean_projection(raw.get("coverage", {})),
        "raw_evidence": clean_projection(raw.get("raw_evidence", {})),
        "result_sha256": sha_file(path),
    }
    arm_manifest = Path(arm_root) / "SHA256SUMS"
    if arm_manifest.is_file():
        rows = [line for line in arm_manifest.read_text(encoding="utf-8").splitlines() if line]
        projected["raw_evidence"]["arm_manifest"] = {
            "sha256": sha_file(arm_manifest), "file_count": len(rows), "self_excluded": True}
    return projected


def number(metrics, name):
    value = metrics.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"missing numeric comparison metric: {name}")
    return value


def compare_ns07(before, after):
    b, a = before["metrics"], after["metrics"]
    failures = []
    if b["elapsed_s"] < 1800 or b["client_count"] != 6:
        failures.append("Before did not execute the full six-client 1800-second schedule")
    if before.get("model_identities") != after.get("model_identities"):
        failures.append("Before/After model identities differ")
    if before.get("config_identity") != after.get("config_identity"):
        failures.append("Before/After frozen configurations differ")
    if before.get("source_identity", {}).get("commit") == after.get("source_identity", {}).get("commit"):
        failures.append("Before/After source commits are not distinct")
    for name in ("hardware", "harness_sha256", "preregistration_sha256"):
        if before.get("source_identity", {}).get(name) != after.get("source_identity", {}).get(name):
            failures.append(f"Before/After {name} differs")
    if after["status"] != "ISOLATED_PASS":
        failures.append("After isolated engineering gate is not ISOLATED_PASS")
    if a["elapsed_s"] < 1800 or a["client_count"] != 6:
        failures.append("After did not execute the full six-client 1800-second schedule")
    for name in ("unexpected_error_count", "process_crash_count", "cross_session_event_count",
                 "stale_event_count", "duplicate_terminal_count", "pcm_mismatch_count"):
        if a[name] != 0:
            failures.append(f"After {name} is nonzero")
    if a["recovery_attempt_count"] == 0 or a["recovery_success_count"] != a["recovery_attempt_count"]:
        failures.append("After did not recover from every injected fault")
    if a["final_in_flight"] != 0:
        failures.append("After did not drain in_flight to zero")
    if a["device_growth_mib"] > 0 or a["rss_growth_mib"] > 64:
        failures.append("After exceeded the frozen same-topology resource-growth bound")
    if a["latency_p95_ms"] > number(b, "latency_p95_ms") * 1.05:
        failures.append("After p95 latency regressed by more than 5 percent")
    deltas = {name: number(a, name) - number(b, name) for name in a}
    return {"status": "PAIRED_NON_REGRESSION" if not failures else "NOT_PROMOTED",
            "no_regression": not failures, "failures": failures, "after_minus_before": deltas}


def compare_ns08(before, after):
    b, a = before["metrics"], after["metrics"]
    failures = []
    if before.get("source_identity", {}).get("commit") == after.get("source_identity", {}).get("commit"):
        failures.append("Before/After source commits are not distinct")
    if before.get("binary_identities") != after.get("binary_identities"):
        failures.append("Before/After Chromium/browser binary identities differ")
    if before.get("model_identities") != after.get("model_identities"):
        failures.append("Before/After model identities differ")
    if before.get("config_identity") != after.get("config_identity"):
        failures.append("Before/After frozen configurations differ")
    if before.get("source_identity", {}).get("adapter_sha256") != after.get("source_identity", {}).get("adapter_sha256"):
        failures.append("Before/After NS-08 adapter hashes differ")
    visible_names = [name for name in a if name.startswith("visible_")]
    before_visible = sum(number(b, name) for name in visible_names)
    after_visible = sum(number(a, name) for name in visible_names)
    if after_visible >= before_visible:
        failures.append("After does not reduce default-visible developer controls")
    if after_visible != 0:
        failures.append("After still exposes default developer controls")
    for name in ("contract_checks_failed", "browser_checks_failed"):
        if a[name] != 0:
            failures.append(f"After {name} is nonzero")
    for name in ("export_import_exact", "reconnect_exact", "source_bound_revision",
                 "cancellation_retry_exact"):
        if a[name] != 1:
            failures.append(f"After did not preserve {name}")
    deltas = {name: number(a, name) - number(b, name) for name in a}
    return {"status": "PAIRED_ENGINEERING_IMPROVEMENT" if not failures else "NOT_PROMOTED",
            "no_regression": not failures, "failures": failures,
            "before_visible_developer_controls": before_visible,
            "after_visible_developer_controls": after_visible,
            "after_minus_before": deltas}


def full_gate(case, arm):
    """Require explicit independent coverage; an adapter declaration alone is insufficient."""
    if arm is None or arm.get("status") == "NOT_RUN":
        return "NOT_RUN"
    declared = arm.get("declared_full_gate_status", "NOT_RUN")
    coverage = arm.get("coverage", {})
    if declared == "FAIL" or arm.get("status") == "FAIL":
        return "FAIL"
    if declared != "PASS" or arm.get("status") != "PASS":
        return "NOT_RUN" if declared in {"NOT_RUN", "NOT_COVERED"} else "NOT_COVERED"
    if coverage.get("independent_verification") != "PASS":
        return "NOT_COVERED"
    case_id = case["case_id"]
    if case_id == "NS-07":
        required = (arm.get("production_status") == "PASS" and
                    arm.get("deployment_scope") == "resident_5_gpu" and
                    coverage.get("resident_5_gpu") == "PASS" and
                    coverage.get("owner_authorization") == "PASS")
    elif case_id == "NS-08":
        required = all(coverage.get(name) == "PASS" for name in
                       ("web_real", "macos_real", "human_walkthrough", "lossless_round_trip"))
    else:
        required = coverage.get("all_required") == "PASS"
    return "PASS" if required else "NOT_COVERED"


def write_manifest(root):
    manifest = root / "SHA256SUMS"
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == manifest:
            continue
        rows.append(f"{sha_file(path)}  {path.relative_to(root).as_posix()}")
    manifest.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return {"file_count": len(rows), "sha256": sha_file(manifest),
            "self_excluded": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path,
                        help="new/empty caller-selected external evidence directory")
    parser.add_argument("--before-arm", type=Path,
                        help="external directory containing normalized Before NS-07/NS-08 JSON")
    parser.add_argument("--after-arm", type=Path,
                        help="external directory containing normalized After NS-07/NS-08 JSON")
    parser.add_argument("--require-clean", action="store_true",
                        help="fail if tracked repository files are modified")
    parser.add_argument("--skip-contracts", action="store_true",
                        help="aggregate arms only; each missing contract result is explicitly NOT_RUN")
    args = parser.parse_args(argv)
    registry = load(REGISTRY)
    source = current_source_identity()
    if args.require_clean and source["tracked_runtime_dirty"]:
        raise SystemExit("tracked source is dirty; commit the candidate before final verification")
    evidence = require_external_directory(args.evidence_dir)
    contract_root = evidence / "contracts"
    cases = []
    for case in registry["cases"]:
        case_id = case["case_id"]
        contracts = []
        if not args.skip_contracts:
            destination = contract_root / case_id
            destination.mkdir(parents=True, exist_ok=True)
            overrides = case.get("contract_args_by_driver", {})
            for driver in case["contract_drivers"]:
                contracts.append(run_contract_driver(case, driver,
                    overrides.get(driver, case["contract_args"]), destination))
        else:
            contracts = [{"driver": driver, "status": "NOT_RUN"}
                         for driver in case["contract_drivers"]]
        before = project_arm(args.before_arm, case) if args.before_arm else None
        after = project_arm(args.after_arm, case) if args.after_arm else None
        paired = None
        if case.get("paired_target"):
            if before and after and before.get("status") != "NOT_RUN" and after.get("status") != "NOT_RUN":
                paired = compare_ns07(before, after) if case_id == "NS-07" else compare_ns08(before, after)
            else:
                paired = {"status": "NOT_RUN", "no_regression": None,
                          "failures": ["paired Before and After arm results were not both supplied"]}
        case_result = {
            "case_id": case_id,
            "title": case["title"],
            "contract_scope": case["contract_scope"],
            "contract_status": ("NOT_RUN" if args.skip_contracts else
                                "PASS" if contracts and all(x["status"] == "PASS" for x in contracts)
                                else "NOT_COVERED" if all(x["status"] == "NOT_COVERED" for x in contracts)
                                else "FAIL"),
            "contract_drivers": contracts,
            "hard_thresholds": case["hard_thresholds"],
            "live_requirements": case["live_requirements"],
            "north_star_hard_gate": full_gate(case, after),
            "before": before or {"status": "NOT_RUN"},
            "after": after or {"status": "NOT_RUN"},
            "paired": paired,
        }
        cases.append(case_result)
    by_id = {case["case_id"]: case for case in cases}
    ns07 = by_id["NS-07"]["paired"]
    ns08 = by_id["NS-08"]["paired"]
    if ns07 and ns08 and ns07["status"] == "PAIRED_NON_REGRESSION" and ns08["status"] == "PAIRED_ENGINEERING_IMPROVEMENT":
        promotion = "PROMOTED_ISOLATED_ENGINEERING_ONLY"
    elif args.before_arm and args.after_arm:
        promotion = "NOT_PROMOTED"
    else:
        promotion = "NOT_RUN"
    all_hard_gates_pass = all(case["north_star_hard_gate"] == "PASS" for case in cases)
    if promotion == "PROMOTED_ISOLATED_ENGINEERING_ONLY" and all_hard_gates_pass:
        promotion = "PROMOTED_NORTH_STAR"
    scorecard = {
        "schema": SCHEMA,
        "registry_sha256": source["registry_sha256"],
        "source_identity": source,
        "execution": {"python_version": platform.python_version(),
                      "contract_timeout_s": 900, "contracts_skipped": args.skip_contracts},
        "case_order": registry["case_order"],
        "cases": cases,
        "target_outcomes": {"NS-07": ns07, "NS-08": ns08},
        "candidate_promotion": promotion,
        "scoreboard_update_allowed": promotion == "PROMOTED_NORTH_STAR",
        "resident_production_ns07": by_id["NS-07"]["north_star_hard_gate"],
        "macos_and_human_ns08": by_id["NS-08"]["north_star_hard_gate"],
        "claim_boundary": "Offline contracts and isolated engineering evidence never substitute for production, macOS, physical-acoustic, perceptual-quality, or human hard gates.",
    }
    for label, arm_root in (("before", args.before_arm), ("after", args.after_arm)):
        if not arm_root:
            continue
        for case_id in ("NS-07", "NS-08"):
            source_report = Path(arm_root) / (case_id + ".json")
            if source_report.is_file():
                target = evidence / "arms" / label / (case_id + ".json")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_report, target)
    out = evidence / "ns-scorecard.json"
    dump(out, scorecard)
    manifest = write_manifest(evidence)
    print(json.dumps({"scorecard": str(out), "case_count": len(cases),
                      "candidate_promotion": promotion, "manifest": manifest}, sort_keys=True))
    return 0 if all(case["contract_status"] in {"PASS", "NOT_RUN"} for case in cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
