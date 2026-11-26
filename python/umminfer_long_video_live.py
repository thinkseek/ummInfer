#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Controlled-live orchestration for the long-video temporal stress fixture.

This module has no serving or ML imports. Live execution is opt-in through the
CLI in tests/video_temporal_long_live.py. Tests inject explicitly synthetic
endpoint/native/device evidence; such bundles are always labelled
SYNTHETIC_FAKE_ENDPOINT_CONTRACT_ONLY and can never claim real generation,
live latency, perceptual quality, or generalized quality.
"""

from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.parse
import urllib.request

import umminfer_controlled_measurement as controlled
import umminfer_long_video_fixture as fixture

SCHEMA_VERSION = 1
ARMS = ("default", "uniform", "temporal")
MODES = ("serial", "batch4")
BUDGET = 8
ALLOWED_GPUS = (0, 1, 7)
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SYNTHETIC_CLASS = "SYNTHETIC_FAKE_ENDPOINT_CONTRACT_ONLY"
TECHNICAL_GATES = (
    "cold_prefix_zero", "exclusive_device", "frozen_stack_and_data",
    "selection_temporal2_mrope_parity", "complete_repetitions",
    "default_uniform_compatibility", "artifact_checksums",
)


def sha256_bytes(data):
    return fixture.sha256_bytes(data)


def sha256_file(path):
    return fixture.sha256_file(Path(path))


def canonical(value):
    return fixture.canonical(value)


def load_manifest(path, *, deep=True):
    manifest = fixture.validate_bundle(Path(path).parent, deep=deep)
    if [case["id"] for case in manifest["cases"]] != [case["id"] for case in fixture.CASES]:
        raise ValueError("long fixture case order drift")
    if manifest["selection"]["native_live_budgets"] != [BUDGET]:
        raise ValueError("long fixture native live budget drift")
    return manifest


def validate_budgets(budgets):
    if budgets != [BUDGET]:
        raise ValueError("public native live supports only --budgets 8; "
                         "4 and 16 are independent CPU-oracle budgets and fail closed live")


def source_state(source_root):
    root = Path(source_root).resolve()

    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, timeout=30).stdout

    commit = git("rev-parse", "HEAD").decode().strip()
    tree_listing = git("ls-tree", "-r", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=no").decode().strip()
    return {"source_commit": commit,
            "source_state_sha256": sha256_bytes(tree_listing),
            "source_tree_clean": not dirty}


def _file_digest(record, label):
    if not isinstance(record, dict) or not isinstance(record.get("path"), str) \
            or not HEX64.fullmatch(str(record.get("sha256") or "")):
        raise ValueError(f"{label} path/hash provenance is incomplete")
    path = Path(record["path"]).resolve()
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise ValueError(f"stale or missing {label} binary")
    return path


def validate_provenance(path, *, mode, run_id, source_root, rehash=True,
                        process_digest=None):
    raw = Path(path).read_bytes()
    provenance = json.loads(raw)
    if provenance.get("schema_version") != 1:
        raise ValueError(f"{mode}: unsupported provenance schema")
    for key in ("stack_id", "stack_run_id", "source_commit", "source_state_sha256",
                "physical_gpu", "batch_size", "qwen38_prefix_cache", "gateway",
                "native_stage", "model", "max_residual_memory_mib"):
        if key not in provenance:
            raise ValueError(f"{mode}: provenance missing {key}")
    if not provenance["stack_id"] or provenance["stack_run_id"] != run_id:
        raise ValueError(f"{mode}: provenance is not newly recorded for this run id")
    if provenance["physical_gpu"] not in ALLOWED_GPUS:
        raise ValueError(f"{mode}: GPU is not explicitly allowed (0, 1, or 7)")
    expected_batch = 1 if mode == "serial" else 4
    if provenance["batch_size"] != expected_batch:
        raise ValueError(f"{mode}: batch-size topology drift")
    if provenance["qwen38_prefix_cache"] != "disabled":
        raise ValueError(f"{mode}: cold-prefix proof requires a disabled native prefix cache")
    if not HEX40.fullmatch(provenance["source_commit"]) \
            or not HEX64.fullmatch(provenance["source_state_sha256"]):
        raise ValueError(f"{mode}: malformed source identity")
    current = source_state(source_root)
    if not current["source_tree_clean"] \
            or current["source_commit"] != provenance["source_commit"] \
            or current["source_state_sha256"] != provenance["source_state_sha256"]:
        raise ValueError(f"{mode}: source tree is dirty or stale")
    residual = provenance["max_residual_memory_mib"]
    if isinstance(residual, bool) or not isinstance(residual, int) or residual < 0:
        raise ValueError(f"{mode}: invalid residual-memory ceiling")
    for name in ("gateway", "native_stage"):
        pid = provenance[name].get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError(f"{mode}: {name} PID is missing")
        compute_pid = provenance[name].get("compute_pid")
        if compute_pid is not None and (
                isinstance(compute_pid, bool) or not isinstance(compute_pid, int)
                or compute_pid <= 0):
            raise ValueError(f"{mode}: {name} compute PID is malformed")
        if rehash:
            binary_path = _file_digest(provenance[name], f"{mode} {name}")
            digest = process_digest(pid, binary_path) if process_digest else \
                sha256_file(Path(f"/proc/{pid}/exe"))
            if digest != provenance[name]["sha256"]:
                raise ValueError(f"{mode}: running {name} process differs from its pinned binary")
            try:
                os.kill(pid, 0)
            except OSError as error:
                raise ValueError(f"{mode}: {name} is not a live newly recorded stack process") from error
    model = provenance["model"]
    if not isinstance(model.get("revision"), str) or not model["revision"] \
            or not isinstance(model.get("files"), dict) or not model["files"]:
        raise ValueError(f"{mode}: model revision/file hashes are not pinned")
    model_root = Path(model.get("root", "")).resolve()
    if rehash:
        actual_files = {path.relative_to(model_root).as_posix()
                        for path in model_root.rglob("*") if path.is_file()}
        if actual_files != set(model["files"]):
            raise ValueError(f"{mode}: model file manifest is incomplete or stale")
    for relative, digest in model["files"].items():
        if not HEX64.fullmatch(str(digest)) or Path(relative).is_absolute() \
                or ".." in Path(relative).parts:
            raise ValueError(f"{mode}: malformed model file hash")
        if rehash and sha256_file(model_root / relative) != digest:
            raise ValueError(f"{mode}: stale model bytes for {relative}")
    provenance["_sha256"] = sha256_bytes(raw)
    return provenance


def loopback_url(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1") \
            or parsed.port is None or parsed.port == 43301:
        raise ValueError("live gate requires an isolated loopback URL, never the resident service")
    return url.rstrip("/")


def health_probe(url, timeout=10):
    with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as response:
        body = json.load(response)
    if response.status != 200 or not isinstance(body.get("in_flight"), int):
        raise ValueError("health response lacks a valid in_flight count")
    return body


def query_memory(indexes, timeout_s=10):
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits", "-i", ",".join(str(i) for i in indexes)],
        check=True, capture_output=True, text=True, timeout=timeout_s).stdout
    observed = {}
    for line in result.splitlines():
        if line.strip():
            index, used = line.split(",", 1)
            observed[int(index.strip())] = int(used.strip())
    if set(observed) != set(indexes):
        raise ValueError("nvidia-smi memory sampler returned a different device set")
    return observed


class ResourceMonitor:
    """Whole-device observations only; never request-native allocator telemetry."""

    def __init__(self, gpu, sampler, poll_interval_s=0.05, synthetic=False):
        self.gpu = gpu
        self.sampler = sampler
        self.poll_interval_s = poll_interval_s
        self.synthetic = synthetic
        self.samples = []
        self.error = None
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        value = self.sampler([self.gpu])[self.gpu]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("invalid whole-device memory sample")
        self.samples.append(value)

    def start(self):
        self._sample()
        if not self.synthetic:
            def poll():
                while not self._stop.is_set():
                    try:
                        self._sample()
                    except Exception as error:  # retained and failed closed at stop
                        self.error = str(error)
                        return
                    self._stop.wait(self.poll_interval_s)
            self._thread = threading.Thread(target=poll, daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._sample()
        if self.error:
            raise ValueError(f"whole-device resource sampling failed: {self.error}")
        if len(self.samples) < 2:
            raise ValueError("resource monitor has incomplete samples")
        return {"scope": "whole-device-used-memory", "gpu_index": self.gpu,
                "poll_interval_ms": self.poll_interval_s * 1000.0,
                "sample_count": len(self.samples), "baseline_mib": self.samples[0],
                "peak_mib": max(self.samples), "final_mib": self.samples[-1],
                "samples_mib": self.samples}


def exclusive_check(provenance, compute_apps_sampler, memory_sampler):
    gpu = provenance["physical_gpu"]
    native_pid = provenance["native_stage"]["pid"]
    compute_pid = provenance["native_stage"].get("compute_pid", native_pid)
    if isinstance(compute_pid, bool) or not isinstance(compute_pid, int) or compute_pid <= 0:
        raise ValueError("native-stage compute PID is malformed")
    contract = controlled.parse_contract({
        "schema_version": 1, "warmup_repetitions": 0, "measured_repetitions": 1,
        "arm_order": "interleaved", "seed": 0,
        "exclusive_device": {"required": True, "gpu_indexes": [gpu],
                             "allowed_pids": [compute_pid],
                             "max_other_compute_processes": 0},
        "cleanup": {"require_in_flight_zero": True,
                    "max_residual_memory_mib": provenance["max_residual_memory_mib"]},
    })
    baseline = memory_sampler([gpu])
    record = controlled.check_exclusive_device(contract, compute_apps_sampler, baseline)
    observed = compute_apps_sampler([gpu])
    if compute_pid not in [app["pid"] for app in observed[gpu]]:
        record["passed"] = False
        record["problems"].append("recorded native-stage compute PID is absent from the device")
    record["native_stage_pid"] = native_pid
    record["native_stage_compute_pid"] = compute_pid
    return record, contract, baseline


def answer_hit(answer, case):
    terms = case.get("expect_all", [])
    if not terms:
        return None
    folded = answer.casefold()
    if case.get("expect_any_group"):
        return any(term.casefold() in folded for term in terms)
    return all(term.casefold() in folded for term in terms)


def validate_native_selection(native, execution, *, case, arm, mode, short):
    short.validate_native(native, arm)
    if native["selected_frame_count"] != BUDGET or native["max_frames"] != BUDGET:
        raise ValueError("native selected-frame budget is not exactly eight")
    expected_engine = "serial" if mode == "serial" else "batched"
    if native["engine_path"] != expected_engine:
        raise ValueError("native engine path differs from the requested mode")
    if execution["prefix_reused"] != 0:
        raise ValueError(f"cold-prefix proof failed: prefix_reused={execution['prefix_reused']}")
    expected = case["expected_uniform_8"] if arm in ("default", "uniform") \
        else case["expected_temporal"][str(BUDGET)]
    if native["selected_frame_content_fnv1a64"] != expected["content_fnv1a64"]:
        raise ValueError("native selected RGB sequence diverges from independent source content")
    if arm == "temporal" and native["temporal2_pair_first_timestamps"] != \
            expected["temporal2_pair_first_timestamps"]:
        raise ValueError("temporal-2 pair-first timestamps diverge")
    if arm == "temporal":
        for native_key, expected_key in (
                ("selected_candidate_indices", "candidate_indices"),
                ("selected_source_frame_indices", "source_frame_indices"),
                ("selected_frame_timestamps", "source_timestamps_seconds")):
            if native[native_key] != expected[expected_key]:
                raise ValueError(f"temporal source selection mismatch: {native_key}")
        if not native["source_indices_exact"] or native["has_nominal_timestamps"]:
            raise ValueError("temporal source indices/PTS are not fully exact")
        if native["decoded_frame_count"] != case["candidate_count"]:
            raise ValueError("temporal candidate decode is incomplete")
    else:
        if native["source_indices_exact"] or not native["has_nominal_timestamps"]:
            raise ValueError("uniform nominal diagnostics changed contract")
        if native["decoded_frame_count"] != BUDGET:
            raise ValueError("uniform historical decode count changed")
    return {"status": "COVERED", "source_binding": "native_scaled_rgb_fnv1a64_"
            "matched_to_independent_full_decode",
            "exact_source_frame_indices": expected["source_frame_indices"],
            "exact_source_timestamps_seconds": expected["source_timestamps_seconds"],
            "temporal2_pair_first_timestamps": expected["temporal2_pair_first_timestamps"],
            "fingerprint_scope": "non-cryptographic parity fingerprint; bundle uses SHA-256"}


def execute_schedule(manifest, *, transports, native_logs, execution_logs,
                     health_probes, memory_samplers, provenances, out_dir,
                     short, synthetic=False):
    measurement = manifest["measurement"]
    steps = []
    records = []
    cleanup_checks = []
    resource_steps = []
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
        for step in mode_steps:
            monitor = ResourceMonitor(provenances[mode]["physical_gpu"],
                                      memory_samplers[mode], synthetic=synthetic).start()
            step_record = {"mode": mode, **step, "budget": BUDGET}

            def execute(case):
                body = short.build_body(
                    case["_data_url"], case["question"], step["arm"],
                    provenances[mode]["model"].get("id", "qwen3-native"),
                    fixture.MOTION_THRESHOLD, fixture.SCENE_THRESHOLD,
                    measurement["seed"], 128)
                response = None
                frames = []
                error = None
                try:
                    response, frames = transports[mode](body, 240.0)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"[:4096]
                relative = Path("records") / mode / step["phase"] / (
                    f"repetition-{step['repetition']}") / step["arm"] / f"{case['id']}.json"
                frame_relative = relative.with_suffix(".sse.json")
                base = {
                    "mode": mode, "phase": step["phase"],
                    "repetition": step["repetition"], "arm": step["arm"],
                    "budget": BUDGET, "clip": case["id"],
                    "schedule_index": len(steps) + len(mode_steps) * 0,
                    "status": "COMPLETED" if response is not None else "FAILED",
                    "error": error, "response": response, "raw_answer":
                        response.get("answer") if response else None,
                    "expected_answer_hit": answer_hit(response["answer"], case)
                        if response else None,
                    "request": short.request_summary(body, case["video_sha256"]),
                    "response_frames": frame_relative.as_posix(),
                    "native_evidence_valid": False,
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
                if row["mode"] == mode and row["phase"] == step["phase"] \
                        and row["repetition"] == step["repetition"] \
                        and row["arm"] == step["arm"]:
                    row["resources"] = dict(resources,
                                            attribution="shared_by_four_clips_in_schedule_step")
            cleanup = controlled.evaluate_cleanup(
                contract, in_flight=health_probes[mode]()["in_flight"],
                residual_mib={provenances[mode]["physical_gpu"]: resources["final_mib"]})
            cleanup["mode"] = mode
            cleanup["schedule_index"] = len(steps)
            cleanup_checks.append(cleanup)
            resource_steps.append({"mode": mode, **step, "resources": resources,
                                   "cleanup": cleanup})
            steps.append(step_record)

    # Join and validate every response, including warmups, against native logs.
    # Live passes lazy providers so no pre-existing request-keyed log can be
    # mistaken for evidence produced by this schedule.
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
            record["source_selection"] = validate_native_selection(
                native, execution, case=case, arm=record["arm"], mode=record["mode"],
                short=short)
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
        if record["phase"] != "measured":
            continue
        key = (record["mode"], record["clip"], record["arm"], record["budget"])
        groups.setdefault(key, []).append(record)
    aggregates = []
    for key in sorted(groups):
        rows = groups[key]
        latency = {}
        for name in ("first_event", "ttft", "total", "vision", "prefill", "decode",
                     "native_elapsed"):
            latency[name] = controlled.summarize([
                row.get("latency_ms", {}).get(name)
                if row["native_evidence_valid"] else None for row in rows])
        hit_values = [row["expected_answer_hit"] for row in rows]
        covered = [value for value in hit_values if value is not None]
        aggregates.append({
            "mode": key[0], "clip": key[1], "arm": key[2], "budget": key[3],
            "declared_repetitions": len(rows),
            "completed_repetitions": sum(row["status"] == "COMPLETED" for row in rows),
            "native_valid_repetitions": sum(row["native_evidence_valid"] for row in rows),
            "failed_repetitions": sum(row["status"] != "COMPLETED" for row in rows),
            "latency_ms": latency,
            "whole_device_resource": {
                "scope": "whole-device-used-memory-shared-by-four-clips",
                "sample_count": sum(row.get("resources", {}).get("sample_count", 0)
                                    for row in rows),
                "baseline_mib": [row.get("resources", {}).get("baseline_mib")
                                 for row in rows],
                "peak_mib": max((row.get("resources", {}).get("peak_mib", 0)
                                 for row in rows), default=0),
                "final_mib": [row.get("resources", {}).get("final_mib") for row in rows],
                "request_native_allocator": "NOT_USED",
            },
            "expected_answer_coverage": {
                "covered": len(covered), "hits": sum(value is True for value in covered),
                "misses": sum(value is False for value in covered), "values": hit_values,
                "quality_claim": "explicit_expected_substrings_only"},
            "raw_answer_sha256": [sha256_bytes((row["raw_answer"] or "").encode())
                                  if row["raw_answer"] is not None else None for row in rows],
        })
    return aggregates


def parity_checks(records, short):
    checks = []
    by_key = {}
    for record in records:
        key = (record["mode"], record["phase"], record["repetition"], record["clip"])
        by_key.setdefault(key, {})[record["arm"]] = record
    for key, arms in sorted(by_key.items()):
        default = arms.get("default", {})
        uniform = arms.get("uniform", {})
        valid = default.get("native_evidence_valid") and uniform.get("native_evidence_valid")
        requests = bool(default and uniform) and \
            short.without_selection(default["request"]) == \
            short.without_selection(uniform["request"])
        answers = bool(default and uniform) and \
            default.get("raw_answer") == uniform.get("raw_answer")
        native = bool(valid) and short.native_parity(default["native"], uniform["native"])
        checks.append({"scope": "default_uniform", "mode": key[0], "phase": key[1],
                       "repetition": key[2], "clip": key[3], "passed": bool(
                           requests and answers and native),
                       "request_projection_equal": requests, "answer_byte_identical": answers,
                       "native_selection_temporal2_mrope_identical": native})
    cross = {}
    for record in records:
        key = (record["phase"], record["repetition"], record["clip"], record["arm"])
        cross.setdefault(key, {})[record["mode"]] = record
    for key, modes in sorted(cross.items()):
        serial = modes.get("serial", {})
        batch = modes.get("batch4", {})
        valid = serial.get("native_evidence_valid") and batch.get("native_evidence_valid")
        native = bool(valid) and short.native_parity(serial["native"], batch["native"])
        answers = bool(serial and batch) and serial.get("raw_answer") == batch.get("raw_answer")
        requests = bool(serial and batch) and serial.get("request") == batch.get("request")
        checks.append({"scope": "serial_batch4", "phase": key[0], "repetition": key[1],
                       "clip": key[2], "arm": key[3], "passed": bool(
                           native and answers and requests),
                       "request_byte_identical": requests,
                       "answer_byte_identical": answers,
                       "native_selection_temporal2_mrope_identical": native})
    repeated = {}
    for record in records:
        key = (record["mode"], record["clip"], record["arm"], record["budget"])
        repeated.setdefault(key, []).append(record)
    for key, rows in sorted(repeated.items()):
        valid = all(row.get("native_evidence_valid") for row in rows)
        native = valid and all(short.native_parity(rows[0]["native"], row["native"])
                               for row in rows[1:])
        requests = len({canonical(row.get("request")) for row in rows}) == 1
        answers = len({row.get("raw_answer") for row in rows}) == 1
        checks.append({"scope": "within_mode_repetitions", "mode": key[0], "clip": key[1],
                       "arm": key[2], "budget": key[3],
                       "passed": bool(native and requests and answers),
                       "request_byte_identical": requests, "answer_byte_identical": answers,
                       "native_selection_temporal2_mrope_identical": native})
    return checks


def gate_reasons(result, provenances, exclusive, stale_errors=()):
    reasons = list(stale_errors)
    records = result["records"]
    covered_modes = {row.get("mode") for row in exclusive if row.get("passed")}
    if covered_modes != set(MODES) or any(not row.get("passed") for row in exclusive):
        reasons.append("exclusive_device: missing or failed whole-device exclusivity proof")
    if any(not row.get("passed") for row in result["cleanup_checks"]):
        reasons.append("exclusive_device: cleanup/in_flight/residual-memory check failed")
    if any(row["status"] == "COMPLETED" and not row["native_evidence_valid"]
           for row in records):
        reasons.append("selection_temporal2_mrope_parity: missing/invalid native evidence or cold prefix")
    for mode in MODES:
        for phase in ("warmup", "measured"):
            expected = (provenances["_measurement"]["warmup_repetitions"]
                        if phase == "warmup" else
                        provenances["_measurement"]["measured_repetitions"])
            actual = {(row["clip"], row["arm"], row["repetition"])
                      for row in records if row["mode"] == mode and row["phase"] == phase}
            wanted = {(case["id"], arm, repetition)
                      for case in provenances["_cases"] for arm in ARMS
                      for repetition in range(1, expected + 1)}
            if actual != wanted:
                reasons.append(f"complete_repetitions: {mode}/{phase} schedule is incomplete")
    if any(row["status"] != "COMPLETED" for row in records):
        reasons.append("complete_repetitions: one or more requests failed")
    parity = parity_checks(records, provenances["_short"])
    if any(not row["passed"] for row in parity):
        reasons.append("selection_temporal2_mrope_parity/default_uniform: parity failed")
    return sorted(set(reasons))


def write_bundle(out_dir, report):
    out_dir = Path(out_dir)
    report_path = out_dir / "report.json"
    report_path.write_bytes(json.dumps(report, ensure_ascii=False, indent=2).encode() + b"\n")
    files = sorted(path for path in out_dir.rglob("*") if path.is_file()
                   and path.name not in ("artifact-manifest.json", "SHA256SUMS"))
    entries = [{"path": path.relative_to(out_dir).as_posix(), "bytes": path.stat().st_size,
                "sha256": sha256_file(path)} for path in files]
    manifest = {"schema_version": 1, "hash": "sha256", "files": entries}
    (out_dir / "artifact-manifest.json").write_bytes(
        json.dumps(manifest, indent=2).encode() + b"\n")
    sums = sorted(path for path in out_dir.rglob("*") if path.is_file()
                  and path.name != "SHA256SUMS")
    (out_dir / "SHA256SUMS").write_text("".join(
        f"{sha256_file(path)}  {path.relative_to(out_dir).as_posix()}\n" for path in sums),
        encoding="ascii")
    validate_bundle(out_dir)


def validate_bundle(out_dir):
    """Independent artifact-set, checksum, aggregate, and parity recomputation."""
    out_dir = Path(out_dir)
    sums = {}
    for line in (out_dir / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        if not HEX64.fullmatch(digest) or relative.startswith("/") \
                or ".." in Path(relative).parts or relative in sums:
            raise ValueError("malformed or duplicate evidence checksum")
        sums[relative] = digest
    actual = {path.relative_to(out_dir).as_posix() for path in out_dir.rglob("*")
              if path.is_file() and path.name != "SHA256SUMS"}
    if set(sums) != actual:
        raise ValueError("missing, extra, or renamed evidence artifacts")
    for relative, digest in sums.items():
        if sha256_file(out_dir / relative) != digest:
            raise ValueError(f"altered evidence checksum: {relative}")
    artifact = json.loads((out_dir / "artifact-manifest.json").read_bytes())
    listed = {row["path"]: row for row in artifact.get("files", [])}
    expected = actual - {"artifact-manifest.json"}
    if set(listed) != expected:
        raise ValueError("artifact manifest does not cover the complete evidence set")
    for relative, row in listed.items():
        path = out_dir / relative
        if row["bytes"] != path.stat().st_size or row["sha256"] != sha256_file(path):
            raise ValueError(f"artifact manifest mismatch: {relative}")
    report = json.loads((out_dir / "report.json").read_bytes())
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported evidence report schema")
    if aggregate_results(report["records"]) != report["aggregates"]:
        raise ValueError("independent aggregate recomputation failed")
    if report["evidence_class"] == SYNTHETIC_CLASS:
        required = {"real_generation": "NOT_RUN", "live_latency": "NOT_RUN",
                    "perceptual_quality": "NOT_RUN", "generalized_quality_claim": False}
        if report["claims"] != required:
            raise ValueError("synthetic fake-endpoint evidence escaped its NOT_RUN boundary")
    if not report["gate_pass"] and not report["gate_reasons"]:
        raise ValueError("failed gate has no fail-closed reason")
    return report


def prepare_cases(manifest, root, short):
    cases = copy.deepcopy(manifest["cases"])
    for case in cases:
        data_url, digest, _ = short.encode_video(root / case["video"])
        if digest != case["video_sha256"]:
            raise ValueError(f"stale fixture bytes: {case['id']}")
        case["_data_url"] = data_url
    manifest = copy.deepcopy(manifest)
    manifest["cases"] = cases
    manifest["_manifest_sha256"] = sha256_file(root / "manifest.json")
    return manifest


def execute_live(args, short):
    validate_budgets(args.budgets)
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path, deep=True)
    manifest = prepare_cases(manifest, manifest_path.parent, short)
    if Path(args.out_dir).exists():
        raise ValueError("refusing to overwrite an existing evidence directory")
    provenance_args = {"serial": args.serial_provenance, "batch4": args.batch_provenance}
    urls = {"serial": loopback_url(args.serial_url), "batch4": loopback_url(args.batch_url)}
    logs = {"serial": args.serial_native_log, "batch4": args.batch_native_log}
    provenances = {}
    exclusive = []
    for mode in MODES:
        provenance = validate_provenance(
            provenance_args[mode], mode=mode, run_id=args.run_id,
            source_root=args.source_root)
        provenances[mode] = provenance
        health_probe(urls[mode])
        exclusive_record, _, _ = exclusive_check(
            provenance, controlled.query_compute_apps, query_memory)
        exclusive_record["mode"] = mode
        exclusive.append(exclusive_record)
    identity_fields = ("source_commit", "source_state_sha256", "gateway", "native_stage",
                       "model", "qwen38_prefix_cache")
    for field in identity_fields:
        left = public_provenance(provenances["serial"][field])
        right = public_provenance(provenances["batch4"][field])
        if field in ("gateway", "native_stage"):
            left = {key: value for key, value in left.items()
                    if key not in ("pid", "compute_pid")}
            right = {key: value for key, value in right.items()
                     if key not in ("pid", "compute_pid")}
        if left != right:
            raise ValueError(f"frozen serial/batch4 {field} identity differs")
    if any(not row["passed"] for row in exclusive):
        raise ValueError("exclusive-device preflight failed before any model request")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True)

    def transport(mode):
        return lambda body, timeout: short.stream_chat(urls[mode], body, timeout)

    result = execute_schedule(
        manifest, transports={mode: transport(mode) for mode in MODES},
        native_logs=lambda: {mode: short.parse_native_log(Path(logs[mode]).resolve())
                             for mode in MODES},
        execution_logs=lambda: {
            mode: short.parse_native_execution_log(Path(logs[mode]).resolve())
            for mode in MODES},
        health_probes={mode: (lambda mode=mode: health_probe(urls[mode])) for mode in MODES},
        memory_samplers={mode: query_memory for mode in MODES},
        provenances=provenances, out_dir=out_dir, short=short)
    for mode in MODES:
        post_record, _, _ = exclusive_check(
            provenances[mode], controlled.query_compute_apps, query_memory)
        post_record["mode"] = mode
        post_record["phase"] = "post_schedule"
        exclusive.append(post_record)
    stale_errors = []
    for mode in MODES:
        try:
            validate_provenance(provenance_args[mode], mode=mode, run_id=args.run_id,
                                source_root=args.source_root)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            stale_errors.append(f"frozen_stack_and_data: {mode}: {error}")
    try:
        load_manifest(manifest_path, deep=True)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        stale_errors.append(f"frozen_stack_and_data: fixture changed during run: {error}")
    provenances["_short"] = short
    manifest["_manifest_path"] = manifest_path
    report = finish_report(manifest, result, provenances=provenances,
                           exclusive=exclusive, evidence_class=controlled.CONTROLLED_LIVE)
    write_bundle(out_dir, report)
    print(json.dumps({"records": len(result["records"]),
                      "aggregates": len(report["aggregates"]),
                      "gate_pass": report["gate_pass"],
                      "gate_reasons": report["gate_reasons"],
                      "evidence_class": report["evidence_class"],
                      "perceptual_quality": "NOT_RUN",
                      "generalized_quality_claim": False,
                      "report": str(out_dir / "report.json")}, indent=2, ensure_ascii=False))
    return 0 if report["gate_pass"] else 1


def public_provenance(value):
    """Keep hash/pid/topology evidence, never absolute host paths."""
    if isinstance(value, dict):
        return {key: public_provenance(item) for key, item in value.items()
                if not key.startswith("_") and key not in ("path", "root")}
    if isinstance(value, list):
        return [public_provenance(item) for item in value]
    return value


def finish_report(manifest, result, *, provenances, exclusive, evidence_class,
                  stale_errors=()):
    provenance_view = dict(provenances)
    provenance_view["_measurement"] = manifest["measurement"]
    provenance_view["_cases"] = manifest["cases"]
    provenance_view["_short"] = provenance_view["_short"]
    reasons = gate_reasons(result, provenance_view, exclusive, stale_errors)
    synthetic = evidence_class == SYNTHETIC_CLASS
    gate_pass = not reasons
    claims = {"real_generation": "NOT_RUN", "live_latency": "NOT_RUN",
              "perceptual_quality": "NOT_RUN", "generalized_quality_claim": False} if synthetic else {
        "real_generation": "MEASURED_LIMITED_FIXTURE",
        "live_latency": "CONTROLLED_LIVE" if gate_pass else "UNCONTROLLED_LIVE",
        "perceptual_quality": "NOT_RUN", "generalized_quality_claim": False}
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": evidence_class if synthetic else (
            controlled.CONTROLLED_LIVE if gate_pass else controlled.UNCONTROLLED_LIVE),
        "fixture_sha256": manifest["_manifest_sha256"],
        "budget_support": {"cpu_oracle": list(fixture.CPU_BUDGETS),
                           "native_live_executed": [BUDGET],
                           "native_live_unsupported_rejected": [4, 16]},
        "arms": list(ARMS), "modes": list(MODES), "records": result["records"],
        "steps": result["steps"], "resources": result["resources"],
        "cleanup_checks": result["cleanup_checks"], "exclusive_device": exclusive,
        "provenance": {mode: public_provenance(provenances[mode]) for mode in MODES},
        "parity": parity_checks(result["records"], provenance_view["_short"]),
        "aggregates": aggregate_results(result["records"]),
        "gate_rules": list(TECHNICAL_GATES), "gate_reasons": reasons,
        "gate_pass": gate_pass, "claims": claims,
        "answer_scope": "raw answers and explicit expected substrings only; misses do not "
                        "change the transport gate and never imply perceptual quality",
    }


def execute_synthetic(manifest, *, transports, native_logs, execution_logs,
                      health_probes, memory_samplers, provenances, exclusive,
                      out_dir, short):
    result = execute_schedule(
        manifest, transports=transports, native_logs=native_logs,
        execution_logs=execution_logs, health_probes=health_probes,
        memory_samplers=memory_samplers, provenances=provenances,
        out_dir=Path(out_dir), short=short, synthetic=True)
    provenances = dict(provenances)
    provenances["_short"] = short
    report = finish_report(manifest, result, provenances=provenances,
                           exclusive=exclusive, evidence_class=SYNTHETIC_CLASS)
    write_bundle(out_dir, report)
    return report
