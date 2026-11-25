#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage C single-variable narrative-routing ablation runner.

The only request overlay is routing policy. Prompt, seed, sampling, audio and
visual option bytes stay fixed. Schema-v2 recording/validation and CAS are
reused from the RL exporter; latency and labels remain out-of-band analysis.

  python3 -I python/umminfer_ablation_run.py \
    --config benchmarks/ablations/narrative-routing.json \
    --replay-only --out /tmp/umminfer-narrative-routing-ablation

Live mode additionally requires --base-url host:port (or http://host:port).
Replay is deterministic contract evidence, never real model/performance evidence.
"""

import argparse
import base64
import copy
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import umminfer_ablation_metrics as routing_metrics  # noqa: E402
import umminfer_rl_reward as reward_mod  # noqa: E402
import umminfer_rl_trajectory as traj  # noqa: E402
import umminfer_rl_worker as worker  # noqa: E402

ABLATION_SCHEMA_VERSION = 1
REPLAY_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
POLICIES = ("text_only", "forced_image", "forced_video", "auto_tag")
VISUAL_MODALITIES = ("image", "video")
HEX_256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass
class Experiment:
    config_path: Path
    config_bytes: bytes
    config: dict
    dataset_bytes: bytes
    dataset: dict
    replay_bytes: bytes | None
    replay: dict | None

    @property
    def prompts(self):
        return self.dataset["prompts"]

    @property
    def config_sha256(self):
        return hashlib.sha256(self.config_bytes).hexdigest()

    @property
    def data_sha256(self):
        return hashlib.sha256(self.dataset_bytes).hexdigest()

    @property
    def replay_sha256(self):
        if self.replay_bytes is None:
            return None
        return hashlib.sha256(self.replay_bytes).hexdigest()


def _canonical(value):
    return traj.canonical_json(value)


def _sha256(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b"\n")


def _read_json(path):
    raw = Path(path).read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return raw, value


def _check_hash(raw, expected, label):
    if not isinstance(expected, str) or not HEX_256.fullmatch(expected):
        raise ValueError(f"{label} SHA-256 must be 64 lowercase hex characters")
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def _resolve(base, relative):
    path = Path(relative)
    return path if path.is_absolute() else Path(base) / path


def _load_versioned(base, block, label, expected_version):
    if not isinstance(block, dict) or not isinstance(block.get("path"), str):
        raise ValueError(f"{label} requires path and sha256")
    path = _resolve(base, block["path"])
    raw, value = _read_json(path)
    _check_hash(raw, block.get("sha256"), label)
    if value.get("schema_version") != expected_version:
        raise ValueError(f"{label} has unsupported schema_version")
    return raw, value


def _validate_prompt(prompt):
    if not isinstance(prompt, dict) or not isinstance(prompt.get("id"), str) \
            or not prompt["id"] or not re.fullmatch(r"[a-z0-9_-]+", prompt["id"]):
        raise ValueError("prompt id must be a non-empty lowercase slug")
    if not isinstance(prompt.get("messages"), list) or not prompt["messages"]:
        raise ValueError(f"prompt {prompt['id']} lacks messages")
    if "expected_visual_modalities" in prompt:
        labels = prompt["expected_visual_modalities"]
        if not isinstance(labels, list) or any(
                label not in VISUAL_MODALITIES for label in labels):
            raise ValueError("expected_visual_modalities accepts only image/video")


def load_experiment(config_path):
    config_path = Path(config_path).resolve()
    config_bytes, config = _read_json(config_path)
    if config.get("ablation_schema_version") != ABLATION_SCHEMA_VERSION \
            or config.get("trajectory_schema_version") != 2 \
            or config.get("variable") != "routing_policy":
        raise ValueError("config requires ablation v1, trajectory v2, routing_policy")
    if config.get("routing_policies") != list(POLICIES):
        raise ValueError("routing_policies must contain exactly the four honest baselines")
    if not isinstance(config.get("ablation_id"), str) or not config["ablation_id"]:
        raise ValueError("ablation_id must be a non-empty string")

    dataset_bytes, dataset = _load_versioned(
        config_path.parent, config.get("dataset"), "dataset", 1)
    prompts = dataset.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("dataset prompts must be a non-empty list")
    ids = []
    for prompt in prompts:
        _validate_prompt(prompt)
        ids.append(prompt["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("prompt ids must be unique")

    fixed = config.get("fixed")
    if not isinstance(fixed, dict) or not isinstance(fixed.get("sampling"), dict) \
            or not isinstance(fixed.get("audio"), dict) \
            or not isinstance(fixed.get("visual"), dict):
        raise ValueError("fixed requires sampling, audio and visual objects")
    visual = fixed["visual"]
    if not isinstance(visual.get("max_slots"), int) \
            or isinstance(visual.get("max_slots"), bool) \
            or not 1 <= visual["max_slots"] <= 6:
        raise ValueError("fixed.visual.max_slots must be in [1,6]")
    for modality in VISUAL_MODALITIES:
        if not isinstance(visual.get(modality), dict):
            raise ValueError(f"fixed.visual.{modality} must be an object")
    forbidden = {"model", "messages", "stream", "narrative", "modalities",
                 "audio", "image", "video"}
    overlap = forbidden & set(fixed["sampling"])
    if overlap:
        raise ValueError(f"sampling contains non-sampling fields: {sorted(overlap)}")
    model = config.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("id"), str) \
            or not model["id"] or not isinstance(model.get("revision"), str) \
            or not model["revision"]:
        raise ValueError("model requires non-empty id and revision")
    weights_hash = model.get("weights_sha256")
    if weights_hash is not None and (
            not isinstance(weights_hash, str) or not HEX_256.fullmatch(weights_hash)):
        raise ValueError("model.weights_sha256 must be null or lowercase SHA-256")

    replay_bytes = replay = None
    if "replay" in config:
        replay_bytes, replay = _load_versioned(
            config_path.parent, config["replay"], "replay", REPLAY_SCHEMA_VERSION)
        if not isinstance(replay.get("cases"), list):
            raise ValueError("replay cases must be a list")
    return Experiment(config_path, config_bytes, config, dataset_bytes, dataset,
                      replay_bytes, replay)


def routing_overlay(policy, max_slots):
    """The only fields allowed to vary between otherwise identical requests."""
    if policy not in POLICIES:
        raise ValueError(f"unknown routing policy: {policy}")
    narrative = {"enabled": policy != "text_only", "visual_mode": "auto",
                 "max_slots": max_slots}
    modalities = ["text", "audio"]
    if policy == "forced_image":
        narrative["visual_mode"] = "image"
        modalities.append("image")
    elif policy == "forced_video":
        narrative["visual_mode"] = "video"
        modalities.append("video")
    elif policy == "auto_tag":
        modalities.extend(VISUAL_MODALITIES)
    return {"modalities": modalities, "narrative": narrative}


def build_request(experiment, prompt, policy):
    fixed = experiment.config["fixed"]
    body = {
        "model": experiment.config["model"]["id"],
        "stream": True,
        "messages": copy.deepcopy(prompt["messages"]),
        "audio": copy.deepcopy(fixed["audio"]),
        "image": copy.deepcopy(fixed["visual"]["image"]),
        "video": copy.deepcopy(fixed["visual"]["video"]),
    }
    body.update(copy.deepcopy(fixed["sampling"]))
    body.update(routing_overlay(policy, fixed["visual"]["max_slots"]))
    return body


def fixed_request(body):
    return {key: copy.deepcopy(value) for key, value in body.items()
            if key not in ("modalities", "narrative")}


def fixed_request_sha256(body):
    return _sha256(fixed_request(body))


def fixed_matrix(experiment):
    rows = []
    for prompt in experiment.prompts:
        body = build_request(experiment, prompt, "auto_tag")
        rows.append({"prompt_id": prompt["id"],
                     "messages_sha256": _sha256(prompt["messages"]),
                     "fixed_request_sha256": fixed_request_sha256(body)})
    return {"variable": "routing_policy",
            "overlay_fields": ["modalities", "narrative"],
            "fixed_matrix_sha256": _sha256(rows), "prompts": rows}


def _model_provenance(experiment):
    model = experiment.config["model"]
    identity = {"id": model["id"], "revision": model["revision"]}
    return {
        "id": model["id"], "revision": model["revision"],
        "config_sha256": _sha256(identity),
        "weights_sha256": model.get("weights_sha256"),
        "weights_coverage": "COVERED" if model.get("weights_sha256") else "NOT_COVERED",
    }


def _source_provenance():
    """Bind HEAD plus every tracked diff and untracked source byte."""
    def git_bytes(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                              capture_output=True).stdout

    commit = git_bytes("rev-parse", "HEAD").decode("ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("git returned an invalid source commit")
    diff = git_bytes("diff", "--binary", "HEAD")
    untracked_names = git_bytes(
        "ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
    untracked = []
    for encoded in sorted(name for name in untracked_names if name):
        name = encoded.decode("utf-8", "surrogateescape")
        path = ROOT / name
        if not path.is_file():
            raise ValueError(f"untracked source is not a regular file: {name}")
        raw = path.read_bytes()
        untracked.append({"path": name, "bytes": len(raw),
                          "sha256": hashlib.sha256(raw).hexdigest(),
                          "content_base64": base64.b64encode(raw).decode("ascii")})
    state = {"commit": commit, "tracked_diff_base64":
             base64.b64encode(diff).decode("ascii"), "untracked": untracked}
    return {"commit": commit, "dirty": bool(diff or untracked),
            "state_sha256": _sha256(state)}


def _prepare_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"output directory must be absent or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_base_url(value):
    if "://" not in value:
        value = "http://" + value
    parsed = urlsplit(value)
    if parsed.scheme != "http" or not parsed.netloc \
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment \
            or parsed.username is not None or parsed.password is not None:
        raise ValueError("base URL must be plain http://host:port without credentials/path")
    return parsed.netloc


def _elapsed_us(start_ns):
    return max(0, (time.monotonic_ns() - start_ns) // 1000)


class LatencyTracker:
    """Out-of-band monotonic observations; nothing is added to trajectory v2."""

    def __init__(self):
        self.start_ns = time.monotonic_ns()
        self.raw = {"ttft_us": None, "ttfa_us": None, "terminal_us": None,
                    "slots": []}
        self.slots = {}

    def observe(self, event):
        elapsed = _elapsed_us(self.start_ns)
        kind = event.get("type")
        if kind in ("vlm_delta", "output_delta") and event.get("text") \
                and self.raw["ttft_us"] is None:
            self.raw["ttft_us"] = elapsed
        if kind == "artifact":
            descriptor = event.get("artifact", {})
            if descriptor.get("media_type") == "audio" \
                    and self.raw["ttfa_us"] is None:
                self.raw["ttfa_us"] = elapsed
            slot_id = descriptor.get("slot_id")
            if slot_id in self.slots:
                slot = self.slots[slot_id]
                if slot["first_artifact_us"] is None:
                    slot["first_artifact_us"] = elapsed
                if descriptor.get("index", 0) + 1 == descriptor.get("total", 1):
                    slot["settlement_us"] = elapsed
        elif kind == "visual_slot":
            slot_id = event["slot_id"]
            slot = {"slot_id": slot_id, "dispatch_us": elapsed,
                    "first_artifact_us": None, "settlement_us": None}
            self.slots[slot_id] = slot
            self.raw["slots"].append(slot)
        elif kind == "leg_error" and event.get("slot_id") in self.slots:
            self.slots[event["slot_id"]]["settlement_us"] = elapsed
        elif kind in ("complete", "error", "cancelled"):
            self.raw["terminal_us"] = elapsed

    def finish(self):
        if self.raw["terminal_us"] is None:
            self.raw["terminal_us"] = _elapsed_us(self.start_ns)
        return self.raw


def _optional_us(value, label):
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be null or a non-negative integer")
    return value


def _ms(value):
    return None if value is None else value / 1000.0


def normalize_latency(raw, record, source):
    if not isinstance(raw, dict):
        raise ValueError("latency observation must be an object")
    expected_slots = {slot["slot_id"]: slot for slot in record["visual_slots"]}
    raw_slots = raw.get("slots")
    if not isinstance(raw_slots, list):
        raise ValueError("latency slots must be a list")
    by_id = {}
    for slot in raw_slots:
        if not isinstance(slot, dict) or not isinstance(slot.get("slot_id"), str) \
                or slot["slot_id"] in by_id:
            raise ValueError("latency slot ids are missing or duplicated")
        by_id[slot["slot_id"]] = slot
    if set(by_id) != set(expected_slots):
        raise ValueError("latency slots do not match trajectory visual_slots")
    slots = []
    for slot_id, event in expected_slots.items():
        observed = by_id[slot_id]
        dispatch = _optional_us(observed.get("dispatch_us"), "slot dispatch_us")
        first = _optional_us(observed.get("first_artifact_us"), "slot first_artifact_us")
        settled = _optional_us(observed.get("settlement_us"), "slot settlement_us")
        if dispatch is None:
            raise ValueError("every visual slot requires dispatch_us")
        slot_to_artifact = None if first is None else first - dispatch
        if slot_to_artifact is not None and slot_to_artifact < 0:
            raise ValueError("slot artifact predates dispatch")
        slots.append({
            "slot_id": slot_id, "slot_index": event["slot_index"],
            "media_type": event["media_type"], "dispatch_ms": _ms(dispatch),
            "first_artifact_ms": _ms(first),
            "slot_to_artifact_ms": _ms(slot_to_artifact),
            "settlement_ms": _ms(settled),
        })
    return {
        "source": source,
        "ttft_ms": _ms(_optional_us(raw.get("ttft_us"), "ttft_us")),
        "ttfa_ms": _ms(_optional_us(raw.get("ttfa_us"), "ttfa_us")),
        "terminal_ms": _ms(_optional_us(raw.get("terminal_us"), "terminal_us")),
        "slots": slots,
    }


def _replay_map(experiment):
    if experiment.replay is None:
        raise ValueError("replay-only mode requires config.replay")
    expected = {(policy, prompt["id"]) for policy in POLICIES
                for prompt in experiment.prompts}
    result = {}
    for case in experiment.replay["cases"]:
        if not isinstance(case, dict):
            raise ValueError("replay case must be an object")
        key = (case.get("routing_policy"), case.get("prompt_id"))
        if key not in expected or key in result:
            raise ValueError(f"unexpected or duplicate replay case: {key}")
        result[key] = case
    if set(result) != expected:
        raise ValueError(f"replay is missing cases: {sorted(expected - set(result))}")
    return result


def _record_replay(case, body, recorder, session_id):
    response_hash = case.get("response_sha256")
    encoded = case.get("response_sse_base64")
    if not isinstance(encoded, str):
        raise ValueError("replay response_sse_base64 must be a string")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid replay response base64: {exc}") from exc
    _check_hash(raw, response_hash, "replay response")
    if case.get("request_sha256") != _sha256(body):
        raise ValueError("replay request_sha256 does not match the fixed config/prompt")
    if case.get("fixed_request_sha256") != fixed_request_sha256(body):
        raise ValueError("replay fixed_request_sha256 mismatch")
    params, option_inputs = traj.request_parameter_references(
        body, recorder.artifact_dir)
    inputs = traj.input_references(body["messages"], recorder.artifact_dir) \
        + option_inputs
    details = {}
    request_id, events, output_text, state, error = traj.parse_sse_stream(
        io.BytesIO(raw), artifact_dir=recorder.artifact_dir, details=details)
    record = recorder.record(
        session_id=session_id, request_id=request_id or "replay-request",
        endpoint="/v1/chat/completions", inputs=inputs, params=params,
        events=events, terminal_state=state, error=error,
        output_text=output_text, artifacts=details.get("artifacts", []))
    latency = normalize_latency(case.get("latency_us"), record,
                                "recorded-contract-observation")
    return record, latency


def _session_id(policy, prompt_id):
    return f"narrative-routing:{policy}:{prompt_id}"


def _finalize_trajectory(path, experiment, policy):
    records = []
    prompts = experiment.prompts
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line:
            records.append(json.loads(line))
    if len(records) != len(prompts):
        raise ValueError("trajectory record count differs from prompt count")
    for record, prompt in zip(records, prompts):
        metadata = {"schema_version": 1,
                    "ablation_id": experiment.config["ablation_id"],
                    "routing_policy": policy, "prompt_id": prompt["id"]}
        if "expected_visual_modalities" in prompt:
            metadata["expected_visual_modalities"] = copy.deepcopy(
                prompt["expected_visual_modalities"])
        record["ablation"] = metadata
    Path(path).write_bytes(b"".join(_canonical(record) + b"\n" for record in records))
    reward_mod.apply_reward(path, routing_metrics.NarrativeRoutingReward())
    records = [json.loads(line) for line in Path(path).read_text(
        encoding="utf-8").splitlines() if line]
    Path(path).write_bytes(b"".join(_canonical(record) + b"\n" for record in records))
    return worker.validate_trajectory(path)


def _latency_coverage(value):
    return "COVERED" if value is not None else "NOT_COVERED"


def _case_report(experiment, policy, prompt, body, record, latency):
    observation = routing_metrics.routing_observation(record)
    slots = latency["slots"]
    settled = [slot["slot_to_artifact_ms"] for slot in slots
               if slot["slot_to_artifact_ms"] is not None]
    if not slots and record["terminal_state"] == "completed":
        slot_coverage = "NOT_APPLICABLE"
    elif len(settled) == len(slots) and slots:
        slot_coverage = "COVERED"
    elif settled:
        slot_coverage = "PARTIAL"
    else:
        slot_coverage = "NOT_COVERED"
    return {
        "routing_policy": policy, "prompt_id": prompt["id"],
        "request_sha256": _sha256(body),
        "fixed_request_sha256": fixed_request_sha256(body),
        "terminal_state": record["terminal_state"], "error": record["error"],
        "terminal_legs": copy.deepcopy(record["terminal_legs"]),
        "leg_errors": copy.deepcopy(record["leg_errors"]),
        "latency": latency,
        "routing": observation,
        "reward": copy.deepcopy(record.get("reward")),
        "output_sha256": record["output_sha256"],
        "clean_text_sha256": record["clean_text_sha256"],
        "artifacts": copy.deepcopy(record["artifacts"]),
        "metric_coverage": {
            "ttft": _latency_coverage(latency["ttft_ms"]),
            "ttfa": _latency_coverage(latency["ttfa_ms"]),
            "terminal_latency": _latency_coverage(latency["terminal_ms"]),
            "slot_latency": slot_coverage,
            "routing_f1": observation["coverage"],
            "redundant_visual_slot_rate": observation["coverage"],
        },
    }


def _summary(values):
    present = [value for value in values if value is not None]
    coverage = "NOT_COVERED"
    if len(present) == len(values) and values:
        coverage = "COVERED"
    elif present:
        coverage = "PARTIAL"
    result = {"coverage": coverage, "sample_count": len(present),
              "total_count": len(values), "median_ms": None,
              "p95_ms": None, "min_ms": None, "max_ms": None}
    if present:
        ordered = sorted(present)
        result.update({
            "median_ms": statistics.median(ordered),
            "p95_ms": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
            "min_ms": ordered[0], "max_ms": ordered[-1],
        })
    return result


def _claims(mode):
    return {
        "learned_routing": "NOT_RUN",
        "optimizer": "NOT_RUN",
        "quality_improvement": "NOT_COVERED",
        "real_model_quality": "NOT_RUN" if mode == "replay" else "NOT_MEASURED",
        "latency_evidence": "CONTRACT_OBSERVATIONS_ONLY" if mode == "replay"
            else "UNCONTROLLED_LIVE_MEASUREMENT",
    }


def _artifact_entries(case_reports):
    entries = []
    for case in case_reports:
        for artifact in case["artifacts"]:
            entries.append({"routing_policy": case["routing_policy"],
                            "prompt_id": case["prompt_id"],
                            "artifact": artifact})
    return entries


def _provenance(experiment, source, artifact_entries):
    return {
        "source_commit": source["commit"],
        "source_tree_dirty": source["dirty"],
        "source_state_sha256": source["state_sha256"],
        "config_sha256": experiment.config_sha256,
        "data_sha256": experiment.data_sha256,
        "replay_data_sha256": experiment.replay_sha256,
        "artifact_manifest_sha256": _sha256(artifact_entries),
        "model": _model_provenance(experiment),
        "trajectory_schema_version": 2,
    }


def _metric_view(cases, records):
    aggregate = routing_metrics.aggregate_routing(records)
    slot_values = [slot["slot_to_artifact_ms"] for case in cases
                   for slot in case["latency"]["slots"]]
    ttft = _summary([case["latency"]["ttft_ms"] for case in cases])
    ttfa = _summary([case["latency"]["ttfa_ms"] for case in cases])
    terminal = _summary([case["latency"]["terminal_ms"] for case in cases])
    slots = _summary(slot_values)
    if not slot_values and all(not case["latency"]["slots"]
                               and case["terminal_state"] == "completed"
                               for case in cases):
        slots["coverage"] = "NOT_APPLICABLE"
    metrics = {
        "routing_f1": aggregate["f1"],
        "redundant_visual_slot_rate": aggregate["redundant_slot_rate"],
        "routing_counts": aggregate,
        "ttft_ms": ttft, "ttfa_ms": ttfa, "terminal_ms": terminal,
        "slot_to_artifact_ms": slots,
    }
    coverage = {
        "routing_f1": aggregate["coverage"],
        "redundant_visual_slot_rate": aggregate["coverage"],
        "routing_reward": aggregate["coverage"],
        "ttft": ttft["coverage"], "ttfa": ttfa["coverage"],
        "terminal_latency": terminal["coverage"],
        "slot_latency": slots["coverage"], "quality": "NOT_COVERED",
    }
    return metrics, coverage


def _policy_report(experiment, source, mode, policy, cases, records,
                   trajectory_path, output_root):
    entries = _artifact_entries(cases)
    metrics, metric_coverage = _metric_view(cases, records)
    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "ablation_id": experiment.config["ablation_id"],
        "mode": mode,
        "evidence_class": "contract-replay" if mode == "replay" else "live-uncontrolled",
        "routing_policy": policy,
        "single_variable": fixed_matrix(experiment),
        "cases": cases,
        "metrics": metrics,
        "metric_coverage": metric_coverage,
        "terminal_state_counts": {
            state: sum(case["terminal_state"] == state for case in cases)
            for state in ("completed", "failed", "cancelled")},
        "trajectory_file": str(Path(trajectory_path).relative_to(output_root).as_posix()),
        "trajectory_sha256": hashlib.sha256(Path(trajectory_path).read_bytes()).hexdigest(),
        "claims": _claims(mode),
    }
    report.update(_provenance(experiment, source, entries))
    return report


def run_experiment(experiment, output_root, *, replay_only, base_url=None,
                   timeout_s=120.0):
    output_root = _prepare_output(output_root)
    source = _source_provenance()
    mode = "replay" if replay_only else "live"
    replay_cases = _replay_map(experiment) if replay_only else None
    if not replay_only and base_url is None:
        raise ValueError("live mode requires --base-url")
    normalized_url = _normalize_base_url(base_url) if base_url else None
    policy_reports = []
    all_cases = []
    all_records = []

    for policy in POLICIES:
        policy_dir = output_root / policy
        trajectory_path = policy_dir / "trajectory.jsonl"
        recorder = traj.TrajectoryRecorder(trajectory_path)
        pending = []
        for prompt in experiment.prompts:
            body = build_request(experiment, prompt, policy)
            session_id = _session_id(policy, prompt["id"])
            if replay_only:
                case = replay_cases[(policy, prompt["id"])]
                record, latency = _record_replay(
                    case, body, recorder, session_id)
            else:
                tracker = LatencyTracker()
                record = traj.record_chat_completion(
                    normalized_url, body, recorder, session_id,
                    timeout=timeout_s, event_observer=tracker.observe)
                latency = normalize_latency(
                    tracker.finish(), record, "monotonic-live")
            pending.append((prompt, body, record, latency))
        records = _finalize_trajectory(trajectory_path, experiment, policy)
        cases = [_case_report(experiment, policy, prompt, body, record, latency)
                 for (prompt, body, _, latency), record in zip(pending, records)]
        report = _policy_report(experiment, source, mode, policy, cases,
                                records, trajectory_path, output_root)
        _write_json(policy_dir / "report.json", report)
        policy_reports.append(report)
        all_cases.extend(cases)
        all_records.extend(records)

    all_entries = _artifact_entries(all_cases)
    summaries = []
    for report in policy_reports:
        summaries.append({
            "routing_policy": report["routing_policy"],
            "metrics": report["metrics"],
            "metric_coverage": report["metric_coverage"],
            "terminal_state_counts": report["terminal_state_counts"],
            "trajectory_file": report["trajectory_file"],
            "trajectory_sha256": report["trajectory_sha256"],
            "artifact_manifest_sha256": report["artifact_manifest_sha256"],
        })
    overall_metrics, overall_coverage = _metric_view(all_cases, all_records)
    combined = {
        "ablation_id": experiment.config["ablation_id"],
        "mode": mode,
        "evidence_class": "contract-replay" if mode == "replay" else "live-uncontrolled",
        "run_status": "COMPLETED" if all(
            case["terminal_state"] == "completed" for case in all_cases) else "FAILED",
        "routing_policies": list(POLICIES),
        "single_variable": fixed_matrix(experiment),
        "policy_summaries": summaries,
        "cases": all_cases,
        "metrics": overall_metrics,
        "metric_coverage": overall_coverage,
        "claims": _claims(mode),
    }
    combined.update(_provenance(experiment, source, all_entries))
    _write_json(output_root / "report.json", combined)
    _write_checksums(output_root)
    return combined


def _write_checksums(output_root):
    rows = []
    for path in sorted(Path(output_root).rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        relative = path.relative_to(output_root).as_posix()
        rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}\n")
    (Path(output_root) / "SHA256SUMS").write_text("".join(rows), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--replay-only", action="store_true")
    parser.add_argument("--base-url")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("--timeout-s must be a finite positive number")
    try:
        experiment = load_experiment(args.config)
        report = run_experiment(
            experiment, args.out, replay_only=args.replay_only,
            base_url=args.base_url, timeout_s=args.timeout_s)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f"ablation error: {exc}\n")
    print(json.dumps({"status": report["run_status"],
                      "report": str(Path(args.out) / "report.json"),
                      "policies": list(POLICIES)}, sort_keys=True))
    return 0 if report["run_status"] == "COMPLETED" else 1


if __name__ == "__main__":
    sys.exit(main())
