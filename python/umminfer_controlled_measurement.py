#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Controlled-measurement contract for Stage C ablation runners.

Pure-stdlib, CPU-only experiment layer. It declares warmup/measured repetition
counts, builds deterministic seeded arm schedules, evaluates exclusive-device
preconditions and cooldown/cleanup checks, aggregates median/p95/dispersion
with explicit sample counts, and decides whether a live run may be labelled
CONTROLLED_LIVE. The label is refused whenever repetitions, exclusivity,
hashes, or completion coverage are missing.

Nothing here touches serving code, model weights, or a GPU by itself; every
device or health observation arrives through an injected sampler so CPU tests
stay deterministic. Real controlled performance remains NOT_RUN until a
separately scheduled exclusive live gate executes the contract.
"""

import hashlib
import json
import math
import re
import statistics
import subprocess

MEASUREMENT_SCHEMA_VERSION = 1
CONTRACT_REPLAY = "CONTRACT_REPLAY"
CONTROLLED_LIVE = "CONTROLLED_LIVE"
UNCONTROLLED_LIVE = "UNCONTROLLED_LIVE"
ARM_ORDERS = ("interleaved", "randomized")
HEX_256 = re.compile(r"[0-9a-f]{64}\Z")
HEX_160 = re.compile(r"[0-9a-f]{40}\Z")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _sha256(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _integer(value, label, minimum):
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"measurement.{label} must be an integer >= {minimum}")
    return value


def _integer_list(value, label):
    if not isinstance(value, list) \
            or any(not isinstance(item, int) or isinstance(item, bool) or item < 0
                   for item in value):
        raise ValueError(f"measurement.{label} must be a list of non-negative integers")
    if len(set(value)) != len(value):
        raise ValueError(f"measurement.{label} must not contain duplicates")
    return list(value)


def _known_keys(block, allowed, label):
    if not isinstance(block, dict):
        raise ValueError(f"measurement.{label} must be an object")
    unknown = sorted(set(block) - set(allowed))
    if unknown:
        raise ValueError(f"measurement.{label} has unknown keys: {unknown}")


def parse_contract(block):
    """Validate the optional config `measurement` block; fail closed on typos."""
    if block is None:
        return None
    _known_keys(block, ("schema_version", "warmup_repetitions",
                        "measured_repetitions", "arm_order", "seed",
                        "cooldown_ms", "exclusive_device", "cleanup"),
                "contract")
    if block.get("schema_version") != MEASUREMENT_SCHEMA_VERSION:
        raise ValueError("measurement.schema_version must be 1")
    for key in ("warmup_repetitions", "measured_repetitions", "arm_order", "seed"):
        if key not in block:
            raise ValueError(f"measurement.{key} must be declared explicitly")
    contract = {
        "schema_version": MEASUREMENT_SCHEMA_VERSION,
        "warmup_repetitions": _integer(block["warmup_repetitions"],
                                       "warmup_repetitions", 0),
        "measured_repetitions": _integer(block["measured_repetitions"],
                                         "measured_repetitions", 1),
        "arm_order": block["arm_order"],
        "seed": _integer(block["seed"], "seed", 0),
        "cooldown_ms": _integer(block.get("cooldown_ms", 0), "cooldown_ms", 0),
    }
    if contract["arm_order"] not in ARM_ORDERS:
        raise ValueError(f"measurement.arm_order must be one of {ARM_ORDERS}")

    exclusive = block.get("exclusive_device", {})
    _known_keys(exclusive, ("required", "gpu_indexes", "allowed_pids",
                            "max_other_compute_processes", "max_memory_used_mib"),
                "exclusive_device")
    required = exclusive.get("required", False)
    if not isinstance(required, bool):
        raise ValueError("measurement.exclusive_device.required must be boolean")
    max_other = _integer(exclusive.get("max_other_compute_processes", 0),
                         "exclusive_device.max_other_compute_processes", 0)
    ceiling = exclusive.get("max_memory_used_mib")
    if ceiling is not None:
        ceiling = _integer(ceiling, "exclusive_device.max_memory_used_mib", 0)
    contract["exclusive_device"] = {
        "required": required,
        "gpu_indexes": _integer_list(exclusive.get("gpu_indexes", []),
                                     "exclusive_device.gpu_indexes"),
        "allowed_pids": _integer_list(exclusive.get("allowed_pids", []),
                                      "exclusive_device.allowed_pids"),
        "max_other_compute_processes": max_other,
        "max_memory_used_mib": ceiling,
    }
    if required and not contract["exclusive_device"]["gpu_indexes"]:
        raise ValueError("exclusive_device.required=true requires declared gpu_indexes")

    cleanup = block.get("cleanup", {})
    _known_keys(cleanup, ("require_in_flight_zero", "max_residual_memory_mib",
                          "timeout_s"), "cleanup")
    in_flight = cleanup.get("require_in_flight_zero", True)
    if not isinstance(in_flight, bool):
        raise ValueError("measurement.cleanup.require_in_flight_zero must be boolean")
    residual = cleanup.get("max_residual_memory_mib")
    if residual is not None:
        residual = _integer(residual, "cleanup.max_residual_memory_mib", 0)
    timeout_s = cleanup.get("timeout_s", 30.0)
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) \
            or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("measurement.cleanup.timeout_s must be a finite positive number")
    contract["cleanup"] = {
        "require_in_flight_zero": in_flight,
        "max_residual_memory_mib": residual,
        "timeout_s": float(timeout_s),
    }
    return contract


def _order_key(seed, phase, repetition, arm):
    # SHA-256 ordered shuffle: identical on every host and Python version,
    # unlike implementation-defined PRNG shuffle sequences.
    return hashlib.sha256(_canonical([seed, phase, repetition, arm])).hexdigest()


def build_schedule(variants, contract):
    """Deterministic step list; one step executes every prompt of one arm once.

    Warmup steps always precede measured steps. `interleaved` rotates through
    every arm once per repetition (seeded order); `randomized` hash-shuffles
    the full measured arm x repetition matrix with the recorded seed.
    """
    if not variants:
        raise ValueError("schedule requires at least one arm")
    seed = contract["seed"]
    steps = []

    def emit(phase, repetition, arm):
        steps.append({"index": len(steps), "phase": phase, "arm": arm,
                      "repetition": repetition})

    for repetition in range(1, contract["warmup_repetitions"] + 1):
        for arm in sorted(variants, key=lambda a: _order_key(
                seed, "warmup", repetition, a)):
            emit("warmup", repetition, arm)
    pairs = [(arm, repetition)
             for repetition in range(1, contract["measured_repetitions"] + 1)
             for arm in variants]
    if contract["arm_order"] == "interleaved":
        ordered = []
        for repetition in range(1, contract["measured_repetitions"] + 1):
            for arm in sorted(variants, key=lambda a: _order_key(
                    seed, "measured", repetition, a)):
                ordered.append((arm, repetition))
    else:
        ordered = sorted(pairs, key=lambda pair: _order_key(
            seed, "measured", pair[1], pair[0]))
    for arm, repetition in ordered:
        emit("measured", repetition, arm)
    return steps


def schedule_descriptor(variants, contract, steps):
    """Report-ready schedule with a tamper-evident hash of contract + steps."""
    body = {
        "arm_order": contract["arm_order"],
        "seed": contract["seed"],
        "warmup_repetitions": contract["warmup_repetitions"],
        "measured_repetitions": contract["measured_repetitions"],
        "arms": list(variants),
        "steps": steps,
    }
    return dict(body, schedule_sha256=_sha256(body))


def summarize(values):
    """Median, nearest-rank p95, IQR and MAD with explicit sample counts.

    None entries stay unmeasured: they reduce coverage instead of being
    silently dropped from the declared total.
    """
    present = sorted(value for value in values if value is not None)
    total = len(values)
    coverage = "NOT_COVERED"
    if present and len(present) == total:
        coverage = "COVERED"
    elif present:
        coverage = "PARTIAL"
    result = {"coverage": coverage, "sample_count": len(present),
              "total_count": total, "median_ms": None, "p95_ms": None,
              "iqr_ms": None, "mad_ms": None, "min_ms": None, "max_ms": None}
    if not present:
        return result
    median = statistics.median(present)
    half = len(present) // 2
    lower = present[:half]
    upper = present[-half:] if len(present) % 2 == 0 else present[half + 1:]
    q1 = statistics.median(lower) if lower else median
    q3 = statistics.median(upper) if upper else median
    result.update({
        "median_ms": median,
        "p95_ms": present[max(0, math.ceil(0.95 * len(present)) - 1)],
        "iqr_ms": q3 - q1,
        "mad_ms": statistics.median([abs(value - median) for value in present]),
        "min_ms": present[0],
        "max_ms": present[-1],
    })
    return result


def query_compute_apps(indexes, timeout_s=10.0):
    """Whole-device compute-process observation via nvidia-smi (live only)."""
    observed = {}
    for index in indexes:
        try:
            output = subprocess.run([
                "nvidia-smi", "-i", str(index),
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True,
                timeout=timeout_s).stdout
        except (OSError, subprocess.TimeoutExpired,
                subprocess.CalledProcessError) as exc:
            raise ValueError(f"nvidia-smi compute-apps query failed: {exc}") from exc
        apps = []
        for line in output.splitlines():
            if not line.strip():
                continue
            pid, used = line.split(",", 1)
            apps.append({"pid": int(pid.strip()), "used_mib": int(used.strip())})
        observed[index] = apps
    return observed


def check_exclusive_device(contract, sampler, baseline_mib=None):
    """Exclusive-device precondition; foreign compute processes fail the check."""
    exclusive = contract["exclusive_device"]
    indexes = exclusive["gpu_indexes"]
    observed = sampler(indexes)
    if set(observed) != set(indexes):
        raise ValueError("compute-apps sampler returned a different device set")
    allowed = set(exclusive["allowed_pids"])
    foreign = []
    for index in indexes:
        for app in observed[index]:
            if app["pid"] not in allowed:
                foreign.append({"gpu_index": index, "pid": app["pid"],
                                "used_mib": app["used_mib"]})
    problems = []
    if len(foreign) > exclusive["max_other_compute_processes"]:
        problems.append(f"{len(foreign)} foreign compute processes exceed "
                        f"allowed {exclusive['max_other_compute_processes']}")
    ceiling = exclusive["max_memory_used_mib"]
    if ceiling is not None:
        if baseline_mib is None:
            raise ValueError("memory ceiling precondition requires a baseline sample")
        for index in indexes:
            used = baseline_mib.get(index)
            if used is None or used > ceiling:
                problems.append(f"GPU {index} baseline memory {used} MiB exceeds "
                                f"ceiling {ceiling} MiB")
    return {
        "check": "exclusive_device", "source": "nvidia-smi-compute-apps",
        "scope": "whole-device",
        "gpu_indexes": list(indexes),
        "allowed_pids": sorted(allowed),
        "max_other_compute_processes": exclusive["max_other_compute_processes"],
        "max_memory_used_mib": ceiling,
        "baseline_mib": ({str(k): v for k, v in baseline_mib.items()}
                         if baseline_mib is not None else None),
        "foreign_compute_processes": foreign,
        "passed": not problems,
        "problems": problems,
    }


def evaluate_cleanup(contract, *, in_flight, residual_mib=None):
    """Post-step cleanup verdict: in-flight drain and VRAM return to baseline."""
    cleanup = contract["cleanup"]
    problems = []
    if cleanup["require_in_flight_zero"] and in_flight != 0:
        problems.append(f"in_flight={in_flight} did not drain to zero")
    tolerance = cleanup["max_residual_memory_mib"]
    if tolerance is not None:
        if residual_mib is None:
            problems.append("residual memory tolerance declared without a sampler")
        else:
            for index, residual in sorted(residual_mib.items()):
                if residual > tolerance:
                    problems.append(f"GPU {index} residual {residual} MiB exceeds "
                                    f"tolerance {tolerance} MiB")
    return {"check": "cooldown_cleanup", "passed": not problems,
            "in_flight": in_flight,
            "residual_memory_mib": residual_mib,
            "max_residual_memory_mib": tolerance,
            "problems": problems}


def evaluate_eligibility(contract, *, provenance, model, arms, preconditions,
                         cleanups):
    """CONTROLLED_LIVE only when every contract dimension is fully covered.

    `arms` maps arm -> {warmups_declared, warmups_executed, warmup_failures,
    measured_declared, measured_completed, measured_failed}. Any missing
    repetition, exclusivity verification, hash, or completion coverage adds an
    explicit refusal reason and forces UNCONTROLLED_LIVE.
    """
    reasons = []
    if not HEX_160.fullmatch(str(provenance.get("source_commit") or "")):
        reasons.append("source commit hash is missing")
    if not HEX_256.fullmatch(str(provenance.get("source_state_sha256") or "")):
        reasons.append("source state hash is missing")
    for key in ("config_sha256", "data_sha256", "artifact_manifest_sha256"):
        if not HEX_256.fullmatch(str(provenance.get(key) or "")):
            reasons.append(f"provenance hash {key} is missing")
    def require_complete_model(label, value):
        if not isinstance(value, dict):
            reasons.append(f"{label} provenance is missing")
            return
        if not HEX_256.fullmatch(str(value.get("weights_sha256") or "")):
            reasons.append(f"{label} weights_sha256 is NOT_COVERED; controlled "
                           "measurement requires a pinned available-weights hash")
        config_files = value.get("config_files_sha256")
        config_covered = HEX_256.fullmatch(str(value.get("config_file_sha256") or "")) \
            or (isinstance(config_files, dict) and bool(config_files)
                and all(HEX_256.fullmatch(str(digest) or "")
                        for digest in config_files.values()))
        if not config_covered:
            reasons.append(f"{label} model config-file hash is NOT_COVERED")
        if not HEX_256.fullmatch(str(value.get("binary_sha256") or "")):
            reasons.append(f"{label} backend executable binary_sha256 is NOT_COVERED")

    require_complete_model("model", model)
    backends = provenance.get("visual_backends")
    if not isinstance(backends, dict):
        backends = {}
    for name in ("image", "video"):
        if name not in backends:
            reasons.append(f"visual_backends.{name} provenance is missing")
        else:
            require_complete_model(f"visual_backends.{name}", backends[name])
    auxiliary = provenance.get("auxiliary_backends", {})
    if not isinstance(auxiliary, dict):
        reasons.append("auxiliary_backends provenance must be an object")
    else:
        for name, backend in sorted(auxiliary.items()):
            require_complete_model(f"auxiliary_backends.{name}", backend)
    if not contract["exclusive_device"]["required"]:
        reasons.append("exclusive_device.required=false; exclusivity is not declared")
    verified = [row for row in preconditions if row.get("check") == "exclusive_device"]
    if contract["exclusive_device"]["required"] and not verified:
        reasons.append("exclusive-device precondition was never executed")
    for row in verified:
        if not row.get("passed"):
            reasons.append("exclusive-device precondition failed: "
                           + "; ".join(row.get("problems", [])))
    for arm, coverage in sorted(arms.items()):
        if coverage["warmups_executed"] != coverage["warmups_declared"]:
            reasons.append(f"arm {arm}: executed warmups "
                           f"{coverage['warmups_executed']}/"
                           f"{coverage['warmups_declared']} are incomplete")
        if coverage["warmup_failures"]:
            reasons.append(f"arm {arm}: {coverage['warmup_failures']} warmup "
                           "requests failed")
        if coverage["measured_completed"] != coverage["measured_declared"]:
            reasons.append(f"arm {arm}: measured repetitions "
                           f"{coverage['measured_completed']}/"
                           f"{coverage['measured_declared']} are incomplete")
        if coverage["measured_failed"]:
            reasons.append(f"arm {arm}: {coverage['measured_failed']} measured "
                           "requests did not complete")
    for row in cleanups:
        if not row.get("passed"):
            reasons.append(f"cleanup after schedule step {row.get('schedule_index')} "
                           "failed: " + "; ".join(row.get("problems", [])))
    return {
        "label": CONTROLLED_LIVE if not reasons else UNCONTROLLED_LIVE,
        "reasons": reasons,
        "rules": ["declared warmups executed", "declared measured repetitions completed",
                  "exclusive-device precondition passed",
                  "cooldown/cleanup checks passed",
                  "source/config/data/artifact hashes pinned",
                  "language/visual/auxiliary model config, available-weights, and backend binary hashes pinned",
                  "no failed measured requests"],
    }


def request_scoped_telemetry():
    """No existing public interface exposes request-scoped native allocator or

    VRAM telemetry (/health carries only in_flight; /metrics and nvidia-smi are
    whole-device). Until such an interface exists, request-scoped coverage is
    honestly NOT_COVERED and whole-device sampling is retained and labelled.
    """
    return {
        "coverage": "NOT_COVERED",
        "scope": "whole-device-used-memory",
        "reason": "no existing public interface exposes request-scoped native "
                  "allocator/VRAM telemetry; whole-device sampling is retained",
    }
