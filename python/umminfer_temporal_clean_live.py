#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Controlled serial/batch-four execution for the frozen temporal-clean-r1 gate.

Synthetic tests are always SYNTHETIC_FAKE_ENDPOINT_CONTRACT_ONLY. Real execution
requires an explicit opt-in, a committed hash-frozen fixture, two new isolated
loopback stacks on physical GPUs 0 and 1, and complete native provenance.
"""

from __future__ import annotations

import concurrent.futures
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import types

sys.path.insert(0, str(Path(__file__).resolve().parent))
import umminfer_controlled_measurement as controlled  # noqa: E402
import umminfer_long_video_live as long_live  # noqa: E402
import umminfer_temporal_clean_fixture as fixture  # noqa: E402

SCHEMA_VERSION = 1
ARMS = fixture.ARMS
MODES = ("serial", "batch4")
BUDGET = 8
EXPECTED_GPUS = {"serial": 0, "batch4": 1}
SYNTHETIC_CLASS = long_live.SYNTHETIC_CLASS
TECHNICAL_GATES = long_live.TECHNICAL_GATES + (
    "committed_preregistration_and_freeze", "quality_anchor_coverage",
    "selection_controls_excluded_from_quality_accuracy",
)


def sha256_bytes(data):
    return fixture.sha256_bytes(data)


def sha256_file(path):
    return fixture.sha256_file(Path(path))


def canonical(value):
    return fixture.canonical(value)


def load_manifest(path, *, deep=True, freeze_path=fixture.DEFAULT_FREEZE):
    manifest_path = Path(path).resolve()
    fixture.validate_freeze(manifest_path.parent, freeze_path)
    manifest = fixture.validate_bundle(manifest_path.parent, deep=deep)
    if manifest["selection"]["native_live_budget"] != BUDGET:
        raise ValueError("clean fixture live budget drift")
    if len(manifest["cases"]) != 8 or len(manifest["videos"]) != 4:
        raise ValueError("clean fixture requires four videos and eight questions")
    return manifest


def build_body_after_context(data_url, question, policy, model, request_contract):
    if policy not in ARMS:
        raise ValueError("unknown frame-selection arm")
    video = {"url": data_url}
    if policy == "uniform":
        video["frame_selection"] = {"policy": "uniform"}
    elif policy == "temporal":
        video["frame_selection"] = {
            "policy": "temporal", "motion_threshold": 0.02, "scene_threshold": 0.25,
        }
    return {
        "model": model, "stream": True,
        "temperature": request_contract["temperature"], "top_p": request_contract["top_p"],
        "seed": request_contract["seed"], "max_tokens": request_contract["max_tokens"],
        "messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": video},
            {"type": "text", "text": question},
        ]}],
    }


def request_summary_after_context(body, video_sha256):
    result = copy.deepcopy(body)
    content = result["messages"][0]["content"]
    if [part["type"] for part in content] != ["video_url", "text"]:
        raise ValueError("clean request does not place its question after video context")
    content[0]["video_url"]["url"] = {"data_url_sha256": video_sha256}
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    result["canonical_request_sha256"] = sha256_bytes(encoded)
    return result


def without_selection(body):
    result = copy.deepcopy(body)
    result["messages"][0]["content"][0]["video_url"].pop("frame_selection", None)
    result.pop("canonical_request_sha256", None)
    return result


def answer_hit(answer, case):
    return long_live.answer_hit(answer, case)


def parity_adapter(short):
    """Reuse native parity while honoring question-after-context request order."""
    return types.SimpleNamespace(without_selection=without_selection,
                                 native_parity=short.native_parity)


def execute_schedule(manifest, *, transports, native_logs, execution_logs,
                     health_probes, memory_samplers, provenances, out_dir,
                     short, synthetic=False):
    measurement = manifest["measurement"]
    steps, records, cleanup_checks, resource_steps = [], [], [], []
    lock = threading.Lock()
    for mode in MODES:
        config = measurement["modes"][mode]
        contract = controlled.parse_contract({
            "schema_version": 1,
            "warmup_repetitions": measurement["warmup_repetitions"],
            "measured_repetitions": measurement["measured_repetitions"],
            "arm_order": measurement["arm_order"], "seed": measurement["seed"],
            "exclusive_device": {"required": True,
                                 "gpu_indexes": [provenances[mode]["physical_gpu"]],
                                 "allowed_pids": [provenances[mode]["native_stage"]["pid"]]},
            "cleanup": {"require_in_flight_zero": True,
                        "max_residual_memory_mib": provenances[mode]["max_residual_memory_mib"]},
        })
        mode_steps = controlled.build_schedule(ARMS, contract)
        for schedule_index, step in enumerate(mode_steps):
            monitor = long_live.ResourceMonitor(
                provenances[mode]["physical_gpu"], memory_samplers[mode],
                synthetic=synthetic).start()
            step_record = {"mode": mode, **step, "budget": BUDGET,
                           "case_count": len(manifest["cases"]),
                           "schedule_index": schedule_index}

            def execute(case):
                body = build_body_after_context(
                    case["_data_url"], case["question"], step["arm"],
                    provenances[mode]["model"].get("id", "qwen3-native"),
                    manifest["request_contract"])
                response, frames, error = None, [], None
                try:
                    response, frames = transports[mode](body, 240.0)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"[:4096]
                relative = Path("records") / mode / step["phase"] / (
                    f"repetition-{step['repetition']}") / step["arm"] / f"{case['id']}.json"
                frame_relative = relative.with_suffix(".sse.json")
                quality_scored = case["evaluation_role"] == "quality_scored"
                base = {
                    "mode": mode, "phase": step["phase"], "repetition": step["repetition"],
                    "arm": step["arm"], "budget": BUDGET, "clip": case["id"],
                    "video_id": case["video_id"], "evaluation_role": case["evaluation_role"],
                    "quality_scored": quality_scored, "schedule_index": schedule_index,
                    "status": "COMPLETED" if response is not None else "FAILED",
                    "error": error, "response": response,
                    "raw_answer": response.get("answer") if response else None,
                    "expected_answer_hit": answer_hit(response["answer"], case) if response else None,
                    "answer_outcome_scope": "QUALITY_SCORED" if quality_scored else
                        "SELECTION_MISS_CONTROL_NOT_QUALITY_SCORED",
                    "request": request_summary_after_context(body, case["video_sha256"]),
                    "response_frames": frame_relative.as_posix(), "native_evidence_valid": False,
                }
                (out_dir / frame_relative).parent.mkdir(parents=True, exist_ok=True)
                (out_dir / frame_relative).write_bytes(
                    json.dumps(frames, ensure_ascii=False, indent=2).encode() + b"\n")
                with lock:
                    records.append(base)
                return base

            if config["concurrency"] == 1:
                for case in manifest["cases"]:
                    execute(case)
            else:
                with concurrent.futures.ThreadPoolExecutor(
                        max_workers=config["concurrency"]) as pool:
                    list(pool.map(execute, manifest["cases"]))
            resources = monitor.stop()
            for row in records:
                if row["mode"] == mode and row["schedule_index"] == schedule_index:
                    row["resources"] = dict(
                        resources, attribution="shared_by_cases_in_schedule_step",
                        case_count=len(manifest["cases"]))
            cleanup = controlled.evaluate_cleanup(
                contract, in_flight=health_probes[mode]()["in_flight"],
                residual_mib={provenances[mode]["physical_gpu"]: resources["final_mib"]})
            cleanup.update({"mode": mode, "schedule_index": schedule_index})
            cleanup_checks.append(cleanup)
            resource_steps.append({"mode": mode, **step, "resources": resources,
                                   "cleanup": cleanup})
            steps.append(step_record)

    if callable(native_logs):
        native_logs = native_logs()
    if callable(execution_logs):
        execution_logs = execution_logs()
    for record in records:
        if record["status"] != "COMPLETED":
            continue
        case = next(case for case in manifest["cases"] if case["id"] == record["clip"])
        try:
            request_id = record["response"]["request_id"]
            native = short.native_for_request(native_logs[record["mode"]], request_id)
            execution = short.execution_for_request(execution_logs[record["mode"]], request_id)
            record["source_selection"] = long_live.validate_native_selection(
                native, execution, case=case, arm=record["arm"], mode=record["mode"], short=short)
            record["native"] = native
            record["native_execution"] = execution
            record["native_evidence_valid"] = True
            record["latency_ms"] = {
                "first_event": record["response"]["first_event_s"] * 1000.0,
                "ttft": record["response"]["ttft_s"] * 1000.0,
                "total": record["response"]["total_s"] * 1000.0,
                "vision": execution["vision_ms"], "prefill": execution["prefill_ms"],
                "decode": execution["decode_ms"], "native_elapsed": execution["elapsed_ms"],
            }
        except (ValueError, TypeError, KeyError) as error:
            record["native_evidence_error"] = str(error)
        artifact = out_dir / "records" / record["mode"] / record["phase"] / (
            f"repetition-{record['repetition']}") / record["arm"] / f"{record['clip']}.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(json.dumps(record, ensure_ascii=False, indent=2).encode() + b"\n")
    return {"records": records, "steps": steps, "resources": resource_steps,
            "cleanup_checks": cleanup_checks}


def aggregate_results(records):
    groups = {}
    for record in records:
        if record["phase"] == "measured":
            key = (record["mode"], record["clip"], record["arm"], record["budget"])
            groups.setdefault(key, []).append(record)
    aggregates = []
    for key in sorted(groups):
        rows = groups[key]
        latency = {name: controlled.summarize([
            row.get("latency_ms", {}).get(name) if row["native_evidence_valid"] else None
            for row in rows]) for name in ("first_event", "ttft", "total", "vision",
                                           "prefill", "decode", "native_elapsed")}
        hit_values = [row["expected_answer_hit"] for row in rows]
        covered = [value for value in hit_values if value is not None]
        quality = rows[0]["quality_scored"]
        aggregates.append({
            "mode": key[0], "clip": key[1], "arm": key[2], "budget": key[3],
            "evaluation_role": rows[0]["evaluation_role"], "quality_scored": quality,
            "declared_repetitions": len(rows),
            "completed_repetitions": sum(row["status"] == "COMPLETED" for row in rows),
            "native_valid_repetitions": sum(row["native_evidence_valid"] for row in rows),
            "failed_repetitions": sum(row["status"] != "COMPLETED" for row in rows),
            "latency_ms": latency,
            "whole_device_resource": {
                "scope": "whole-device-used-memory-shared-by-cases-in-step",
                "sample_count": sum(row.get("resources", {}).get("sample_count", 0) for row in rows),
                "baseline_mib": [row.get("resources", {}).get("baseline_mib") for row in rows],
                "peak_mib": max((row.get("resources", {}).get("peak_mib", 0) for row in rows), default=0),
                "final_mib": [row.get("resources", {}).get("final_mib") for row in rows],
                "request_native_allocator": "NOT_USED",
            },
            "raw_substring_observations": {
                "covered": len(covered), "hits": sum(value is True for value in covered),
                "misses": sum(value is False for value in covered), "values": hit_values,
                "included_in_quality_accuracy": quality,
            },
            "raw_answer_sha256": [sha256_bytes((row["raw_answer"] or "").encode())
                                  if row["raw_answer"] is not None else None for row in rows],
        })
    return aggregates


def answer_outcomes(records):
    grouped = {}
    for row in records:
        if row["phase"] != "measured":
            continue
        key = (row["evaluation_role"], row["mode"], row["arm"])
        grouped.setdefault(key, []).append(row)
    result = {}
    for (role, mode, arm), rows in sorted(grouped.items()):
        completed = [row for row in rows if row["status"] == "COMPLETED"]
        covered = [row for row in completed if row["expected_answer_hit"] is not None]
        cases = {}
        for row in sorted(completed, key=lambda item: (item["clip"], item["repetition"])):
            entry = cases.setdefault(row["clip"], {"answers": [], "hits": [], "quality_scored":
                                                  row["quality_scored"]})
            entry["answers"].append(row["raw_answer"])
            entry["hits"].append(row["expected_answer_hit"])
        block = {
            "declared": len(rows), "completed": len(completed),
            "answer_hits": sum(row["expected_answer_hit"] is True for row in covered),
            "answer_misses": sum(row["expected_answer_hit"] is False for row in covered),
            "cases": cases, "raw_answers_retained": True,
            "included_in_quality_accuracy": role == "quality_scored",
        }
        rate = block["answer_hits"] / len(covered) if covered else None
        if role == "quality_scored":
            block["preregistered_substring_accuracy"] = rate
        else:
            block["diagnostic_substring_rate_not_quality"] = rate
        result.setdefault(role, {}).setdefault(mode, {})[arm] = block
    return result


def anchor_coverage(manifest):
    return {case["id"]: {
        "evaluation_role": case["evaluation_role"],
        "arms": {arm: {"answer_anchor_visible": case["anchor_audit"][arm]["answer_anchor_visible"],
                       "declared_anchor_source_frames_present": case["anchor_audit"][arm][
                           "declared_anchor_source_frames_present"]} for arm in ARMS},
    } for case in manifest["cases"]}


def gate_reasons(result, provenances, exclusive, manifest, short, stale_errors=()):
    provenance_view = dict(provenances)
    provenance_view.update({"_measurement": manifest["measurement"],
                            "_cases": manifest["cases"], "_short": parity_adapter(short)})
    reasons = long_live.gate_reasons(result, provenance_view, exclusive, stale_errors)
    for case in manifest["cases"]:
        try:
            fixture.validate_anchor_rules(case, case["anchor_audit"], manifest)
        except ValueError as error:
            reasons.append(f"quality_anchor_coverage: {error}")
    if any(row["status"] == "COMPLETED" and row["expected_answer_hit"] is None
           for row in result["records"]):
        reasons.append("raw answer scoring is incomplete")
    return sorted(set(reasons))


def finish_report(manifest, result, *, provenances, exclusive, evidence_class,
                  short, freeze_receipt, stale_errors=()):
    reasons = gate_reasons(result, provenances, exclusive, manifest, short, stale_errors)
    gate_pass = not reasons
    synthetic = evidence_class == SYNTHETIC_CLASS
    claims = {
        "real_generation": "NOT_RUN", "live_latency": "NOT_RUN",
        "fixture_substring_accuracy": "NOT_RUN", "perceptual_quality": "NOT_RUN",
        "generalized_quality_claim": False,
    } if synthetic else {
        "real_generation": "MEASURED_LIMITED_FIXTURE",
        "live_latency": "CONTROLLED_LIVE" if gate_pass else "UNCONTROLLED_LIVE",
        "fixture_substring_accuracy": "MEASURED_LIMITED_FIXTURE" if gate_pass else "UNCONTROLLED_LIVE",
        "perceptual_quality": "NOT_RUN", "generalized_quality_claim": False,
    }
    return {
        "schema_version": SCHEMA_VERSION, "fixture_revision": fixture.REVISION,
        "evidence_class": evidence_class if synthetic else (
            controlled.CONTROLLED_LIVE if gate_pass else controlled.UNCONTROLLED_LIVE),
        "fixture_sha256": manifest["_manifest_sha256"],
        "preregistration_sha256": manifest["preregistration_sha256"],
        "freeze_receipt": freeze_receipt,
        "budget_support": {"cpu_oracle": list(fixture.CPU_BUDGETS),
                           "native_live_executed": [BUDGET],
                           "native_live_unsupported_rejected": [4, 16]},
        "arms": list(ARMS), "modes": list(MODES), "records": result["records"],
        "measurement": copy.deepcopy(manifest["measurement"]),
        "steps": result["steps"], "resources": result["resources"],
        "cleanup_checks": result["cleanup_checks"], "exclusive_device": exclusive,
        "provenance": {mode: long_live.public_provenance(provenances[mode]) for mode in MODES},
        "parity": long_live.parity_checks(result["records"], parity_adapter(short)),
        "aggregates": aggregate_results(result["records"]),
        "answer_outcomes": answer_outcomes(result["records"]),
        "anchor_coverage": anchor_coverage(manifest),
        "gate_rules": list(TECHNICAL_GATES), "gate_reasons": reasons, "gate_pass": gate_pass,
        "claims": claims,
        "answer_scope": "pre-registered casefold substring hits; only quality_scored cases enter "
                        "quality accuracy; selection-miss controls retain raw hits/misses separately; "
                        "no observed-answer aliases and no generalized perceptual-quality claim",
        "latency_scope": "reported independently from answer outcomes; batch4 includes concurrency-4 "
                         "queueing and is not a serial speedup ratio",
    }


def copy_frozen_input(bundle, freeze_path, out_dir):
    target = out_dir / "input"
    target.mkdir(parents=True)
    shutil.copytree(bundle, target / "fixture")
    shutil.copy2(freeze_path, target / "freeze.json")
    fixture.validate_freeze(target / "fixture", target / "freeze.json")


def seal_bundle(out_dir, report, short):
    out_dir = Path(out_dir)
    (out_dir / "report.json").write_bytes(
        json.dumps(report, ensure_ascii=False, indent=2).encode() + b"\n")
    files = sorted(path for path in out_dir.rglob("*") if path.is_file()
                   and path.name not in ("artifact-manifest.json", "SHA256SUMS"))
    entries = [{"path": path.relative_to(out_dir).as_posix(), "bytes": path.stat().st_size,
                "sha256": sha256_file(path)} for path in files]
    (out_dir / "artifact-manifest.json").write_bytes(json.dumps(
        {"schema_version": 1, "hash": "sha256", "files": entries}, indent=2).encode() + b"\n")
    files = sorted(path for path in out_dir.rglob("*") if path.is_file()
                   and path.name != "SHA256SUMS")
    (out_dir / "SHA256SUMS").write_text("".join(
        f"{sha256_file(path)}  {path.relative_to(out_dir).as_posix()}\n" for path in files),
        encoding="ascii")
    validate_bundle(out_dir, short, deep=False)


def validate_bundle(out_dir, short, *, deep=True):
    out_dir = Path(out_dir)
    sums = {}
    for line in (out_dir / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        if not fixture.HEX256.fullmatch(digest) or relative.startswith("/") \
                or ".." in Path(relative).parts or relative in sums:
            raise ValueError("malformed or duplicate clean evidence checksum")
        sums[relative] = digest
    actual = {path.relative_to(out_dir).as_posix() for path in out_dir.rglob("*")
              if path.is_file() and path.name != "SHA256SUMS"}
    if set(sums) != actual:
        raise ValueError("missing, extra, or renamed clean evidence artifacts")
    for relative, digest in sums.items():
        if sha256_file(out_dir / relative) != digest:
            raise ValueError(f"altered clean evidence checksum: {relative}")
    artifact = json.loads((out_dir / "artifact-manifest.json").read_bytes())
    listed = {row["path"]: row for row in artifact.get("files", [])}
    if set(listed) != actual - {"artifact-manifest.json"}:
        raise ValueError("clean artifact manifest does not cover the complete evidence set")
    for relative, row in listed.items():
        path = out_dir / relative
        if row["bytes"] != path.stat().st_size or row["sha256"] != sha256_file(path):
            raise ValueError(f"clean artifact manifest mismatch: {relative}")
    receipt = fixture.validate_freeze(out_dir / "input" / "fixture", out_dir / "input" / "freeze.json")
    manifest = fixture.validate_bundle(out_dir / "input" / "fixture", deep=deep)
    report = json.loads((out_dir / "report.json").read_bytes())
    if report.get("schema_version") != SCHEMA_VERSION or report.get("fixture_revision") != fixture.REVISION:
        raise ValueError("unsupported clean evidence report")
    if report["fixture_sha256"] != sha256_file(out_dir / "input" / "fixture" / "manifest.json") \
            or report["preregistration_sha256"] != manifest["preregistration_sha256"] \
            or report["freeze_receipt"] != receipt:
        raise ValueError("clean evidence is bound to a different frozen fixture")
    if aggregate_results(report["records"]) != report["aggregates"] \
            or answer_outcomes(report["records"]) != report["answer_outcomes"]:
        raise ValueError("independent clean aggregate/outcome recomputation failed")
    if long_live.parity_checks(report["records"], parity_adapter(short)) != report["parity"]:
        raise ValueError("independent clean parity recomputation failed")
    by_id = {case["id"]: case for case in manifest["cases"]}
    for row in report["records"]:
        case = by_id[row["clip"]]
        if row["evaluation_role"] != case["evaluation_role"] \
                or row["quality_scored"] != (case["evaluation_role"] == "quality_scored"):
            raise ValueError("record quality role differs from preregistration")
        if row["status"] == "COMPLETED":
            if row["raw_answer"] != row["response"]["answer"] \
                    or sha256_bytes(row["raw_answer"].encode()) != row["response"]["answer_sha256"]:
                raise ValueError("raw answer bytes/hash are inconsistent")
            if answer_hit(row["raw_answer"], case) != row["expected_answer_hit"]:
                raise ValueError("recorded answer hit differs from frozen scoring")
            content = row["request"]["messages"][0]["content"]
            if [part["type"] for part in content] != ["video_url", "text"] \
                    or content[1]["text"] != case["question"]:
                raise ValueError("recorded prompt/order differs from preregistration")
    gate_manifest = dict(manifest)
    gate_manifest["measurement"] = report["measurement"]
    reasons = gate_reasons(
        {"records": report["records"], "cleanup_checks": report["cleanup_checks"]},
        report["provenance"], report["exclusive_device"], gate_manifest, short,
        report.get("stale_errors", ()))
    if reasons != report["gate_reasons"] or (not reasons) != report["gate_pass"]:
        raise ValueError("clean gate outcome recomputation failed: "
                         f"recorded={report['gate_reasons']} recomputed={reasons}")
    if report["evidence_class"] == SYNTHETIC_CLASS:
        required = {"real_generation": "NOT_RUN", "live_latency": "NOT_RUN",
                    "fixture_substring_accuracy": "NOT_RUN", "perceptual_quality": "NOT_RUN",
                    "generalized_quality_claim": False}
        if report["claims"] != required:
            raise ValueError("synthetic clean evidence escaped its NOT_RUN boundary")
    if not report["gate_pass"] and not report["gate_reasons"]:
        raise ValueError("failed clean gate has no reason")
    return report


def validate_committed_fixture(source_root, bundle, freeze_path):
    """Prove every frozen byte is in HEAD before the first model request."""
    root = Path(source_root).resolve()
    bundle = Path(bundle).resolve()
    freeze_path = Path(freeze_path).resolve()
    files = [path for path in bundle.rglob("*") if path.is_file()] + [freeze_path]
    for path in files:
        relative = path.relative_to(root).as_posix()
        listed = subprocess.run(["git", "-C", str(root), "ls-tree", "HEAD", "--", relative],
                                check=True, capture_output=True, timeout=30).stdout.decode().strip()
        if not listed:
            raise ValueError(f"frozen fixture is not committed: {relative}")
        blob = listed.split()[2]
        actual = subprocess.run(["git", "-C", str(root), "hash-object", str(path)],
                                check=True, capture_output=True, timeout=30).stdout.decode().strip()
        if blob != actual:
            raise ValueError(f"committed frozen fixture byte mismatch: {relative}")
    return {"committed_frozen_files": len(files), "source_root_bound": True}


def validate_provenance(path, *, mode, run_id, source_root):
    provenance = long_live.validate_provenance(path, mode=mode, run_id=run_id,
                                               source_root=source_root)
    if provenance["physical_gpu"] != EXPECTED_GPUS[mode]:
        raise ValueError(f"{mode}: clean gate requires physical GPU {EXPECTED_GPUS[mode]}; "
                         "GPU 7 and resident GPUs 2-6 are forbidden")
    return provenance


def comparable_stack_identity(provenances, field):
    """Compare immutable stack identity, excluding live PID/resource observations."""
    left = long_live.public_provenance(provenances["serial"][field])
    right = long_live.public_provenance(provenances["batch4"][field])
    if field in ("gateway", "native_stage"):
        volatile = {"pid", "compute_pid", "compute_used_mib"}
        left = {key: value for key, value in left.items() if key not in volatile}
        right = {key: value for key, value in right.items() if key not in volatile}
    return left, right


def execute_live(args, short):
    long_live.validate_budgets(args.budgets)
    manifest_path = Path(args.manifest).resolve()
    freeze_path = Path(args.freeze).resolve()
    manifest = load_manifest(manifest_path, deep=True, freeze_path=freeze_path)
    committed = validate_committed_fixture(args.source_root, manifest_path.parent, freeze_path)
    manifest = long_live.prepare_cases(manifest, manifest_path.parent, short)
    if Path(args.out_dir).exists():
        raise ValueError("refusing to overwrite an existing clean evidence directory")
    provenance_args = {"serial": args.serial_provenance, "batch4": args.batch_provenance}
    urls = {"serial": long_live.loopback_url(args.serial_url),
            "batch4": long_live.loopback_url(args.batch_url)}
    logs = {"serial": args.serial_native_log, "batch4": args.batch_native_log}
    provenances, exclusive = {}, []
    for mode in MODES:
        provenance = validate_provenance(provenance_args[mode], mode=mode, run_id=args.run_id,
                                         source_root=args.source_root)
        provenances[mode] = provenance
        long_live.health_probe(urls[mode])
        record, _, _ = long_live.exclusive_check(provenance, controlled.query_compute_apps,
                                                 long_live.query_memory)
        record["mode"] = mode
        exclusive.append(record)
    identity_fields = ("source_commit", "source_state_sha256", "gateway", "native_stage",
                       "model", "qwen38_prefix_cache")
    for field in identity_fields:
        left, right = comparable_stack_identity(provenances, field)
        if left != right:
            raise ValueError(f"frozen serial/batch4 {field} identity differs")
    if any(not row["passed"] for row in exclusive):
        raise ValueError("exclusive-device preflight failed before any model request")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True)
    copy_frozen_input(manifest_path.parent, freeze_path, out_dir)
    receipt = fixture.validate_freeze(manifest_path.parent, freeze_path)
    fixture.write_json(out_dir / "input" / "pre-request-freeze-validation.json", {
        **committed, "freeze_receipt": receipt, "source_commit": provenances["serial"]["source_commit"],
        "pre_request_unix_ns": time.time_ns(), "pre_request_monotonic_ns": time.monotonic_ns(),
        "model_requests_started": False,
    })

    def transport(mode):
        return lambda body, timeout: short.stream_chat(urls[mode], body, timeout)

    result = execute_schedule(
        manifest, transports={mode: transport(mode) for mode in MODES},
        native_logs=lambda: {mode: short.parse_native_log(Path(logs[mode]).resolve()) for mode in MODES},
        execution_logs=lambda: {mode: short.parse_native_execution_log(Path(logs[mode]).resolve())
                                for mode in MODES},
        health_probes={mode: (lambda mode=mode: long_live.health_probe(urls[mode])) for mode in MODES},
        memory_samplers={mode: long_live.query_memory for mode in MODES},
        provenances=provenances, out_dir=out_dir, short=short)
    for mode in MODES:
        record, _, _ = long_live.exclusive_check(provenances[mode], controlled.query_compute_apps,
                                                 long_live.query_memory)
        record.update({"mode": mode, "phase": "post_schedule"})
        exclusive.append(record)
    stale_errors = []
    for mode in MODES:
        try:
            validate_provenance(provenance_args[mode], mode=mode, run_id=args.run_id,
                                source_root=args.source_root)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            stale_errors.append(f"frozen_stack_and_data: {mode}: {error}")
    try:
        load_manifest(manifest_path, deep=True, freeze_path=freeze_path)
        fixture.validate_bundle(out_dir / "input" / "fixture", deep=True)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        stale_errors.append(f"frozen_stack_and_data: clean fixture changed during run: {error}")
    report = finish_report(manifest, result, provenances=provenances, exclusive=exclusive,
                           evidence_class=controlled.CONTROLLED_LIVE, short=short,
                           freeze_receipt=receipt, stale_errors=stale_errors)
    report["stale_errors"] = list(stale_errors)
    seal_bundle(out_dir, report, short)
    print(json.dumps({
        "records": len(result["records"]), "aggregates": len(report["aggregates"]),
        "gate_pass": report["gate_pass"], "gate_reasons": report["gate_reasons"],
        "evidence_class": report["evidence_class"], "claims": report["claims"],
        "quality_answer_outcomes": report["answer_outcomes"].get("quality_scored", {}),
        "selection_controls": "RETAINED_SEPARATELY_NOT_QUALITY_SCORED",
        "report": str(out_dir / "report.json"),
    }, indent=2, ensure_ascii=False))
    return 0 if report["gate_pass"] else 1


def execute_synthetic(manifest, *, transports, native_logs, execution_logs,
                      health_probes, memory_samplers, provenances, exclusive,
                      out_dir, short, bundle, freeze_path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True)
    copy_frozen_input(bundle, freeze_path, out_dir)
    receipt = fixture.validate_freeze(bundle, freeze_path)
    result = execute_schedule(
        manifest, transports=transports, native_logs=native_logs,
        execution_logs=execution_logs, health_probes=health_probes,
        memory_samplers=memory_samplers, provenances=provenances,
        out_dir=out_dir, short=short, synthetic=True)
    report = finish_report(manifest, result, provenances=provenances, exclusive=exclusive,
                           evidence_class=SYNTHETIC_CLASS, short=short, freeze_receipt=receipt)
    seal_bundle(out_dir, report, short)
    return report
