#!/usr/bin/env python3
"""Fail-closed three-round orchestration for the real multimodal benchmark.

The live driver invokes tests/multimodal_benchmark.py in a fresh process for one
warmup and exactly three measured attempts. It never starts a server, touches a
GPU, retries a request, or combines case subsets. A frozen effective run config
pins source, binaries, model/config/available-weight bytes, manifest and resolved
input identities, request parameters, endpoint/GPU declarations, and tolerances.

Round verification deliberately imports the independent coverage auditor, not
the benchmark runner's scoring or gate functions. Raw SSE, responses, terminals,
metrics, aggregates, timing coverage, and per-case deltas are checked again.

--replay-self-check creates real-shaped synthetic SSE evidence on CPU. Every
result is labelled NOT_REAL_EVIDENCE and cannot become a real baseline or a real
STABILITY_PASS claim. The real three-round SM90 execution is separate and remains
NOT_RUN until an operator supplies the pinned service, weights, data, and baseline.
"""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "benchmarks"))
import multimodal_coverage_audit as audit  # noqa: E402
import multimodal_data as data  # noqa: E402

CONFIG_SCHEMA = "umm-multimodal-stability-run/v1"
RESULT_SCHEMA = "umm-multimodal-stability-run-result/v1"
SCHEDULE_SCHEMA = "umm-multimodal-stability-schedule/v1"
BASELINE_SCHEMA = "umm-multimodal-combined-coverage-audit/v1"
REQUIRED_ROUNDS = 3
WARMUP_ROUNDS = 1
ALLOWED_GPUS = frozenset({0, 1, 7})
RESIDENT_PORT = 43301
TASKS = ("asr_en", "asr_zh", "gsm8k", "joint", "vqa")
HEX_256 = re.compile(r"[0-9a-f]{64}\Z")
HEX_160 = re.compile(r"[0-9a-f]{40}\Z")
RUN_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
CHECKSUM_LINE = re.compile(r"([0-9a-f]{64}) ([ *])(.+)")
QUALITY_DIRECTION = {
    "asr_en_wer": "lower", "asr_en_cer": "lower",
    "asr_zh_cer": "lower", "asr_zh_wer": "lower",
    "vqa_accuracy": "higher", "gsm8k_accuracy": "higher",
    "joint_asr_wer": "lower", "joint_keyword_hit": "higher",
}
DEFAULT_QUALITY_TOLERANCE = {
    "asr_en_wer": 0.02, "asr_en_cer": 0.02,
    "asr_zh_cer": 0.02, "asr_zh_wer": 0.02,
    "vqa_accuracy": 0.05, "gsm8k_accuracy": 0.05,
    "joint_asr_wer": 0.05, "joint_keyword_hit": 0.05,
}


class StabilityError(RuntimeError):
    def __init__(self, code, message, **details):
        super().__init__(message)
        self.reason = {"code": code, "message": message, **details}


def require(condition, code, message, **details):
    if not condition:
        raise StabilityError(code, message, **details)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path):
    return audit.sha256_file(Path(path))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                            for row in rows), encoding="utf-8")


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise StabilityError("MISSING_FILE", f"missing JSON file: {path}")
    except (OSError, json.JSONDecodeError) as error:
        raise StabilityError("INVALID_JSON", f"cannot read {path}: {error}")


def known_keys(block, allowed, label):
    require(isinstance(block, dict), "INVALID_CONFIG", f"{label} must be an object")
    unknown = sorted(set(block) - set(allowed))
    require(not unknown, "INVALID_CONFIG", f"{label} has unknown keys: {unknown}")
    missing = sorted(set(allowed) - set(block))
    require(not missing, "INVALID_CONFIG", f"{label} is missing keys: {missing}")


def finite_number(value, label, minimum=0.0):
    require(type(value) in (int, float) and math.isfinite(value)
            and value >= minimum, "INVALID_CONFIG",
            f"{label} must be finite and >= {minimum}")
    return float(value)


def positive_int(value, label):
    require(type(value) is int and value > 0, "INVALID_CONFIG",
            f"{label} must be a positive integer")
    return value


def hash256(value, label):
    require(isinstance(value, str) and HEX_256.fullmatch(value),
            "INVALID_CONFIG", f"{label} must be a lowercase SHA-256")
    return value


def hash160(value, label, allow_head=False):
    require(isinstance(value, str) and
            (HEX_160.fullmatch(value) or (allow_head and value == "HEAD")),
            "INVALID_CONFIG", f"{label} must be a full commit or permitted HEAD")
    return value


def absolute_path(value, label):
    require(isinstance(value, str) and value and Path(value).is_absolute(),
            "INVALID_CONFIG", f"{label} must be an absolute path")
    return str(Path(value))


def safe_relative(value, label):
    require(isinstance(value, str) and value, "INVALID_CONFIG",
            f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and "." not in path.parts,
            "INVALID_CONFIG", f"{label} is not a safe relative path: {value!r}")
    return value


def git_value(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args],
                            capture_output=True, text=True, timeout=30)
    require(result.returncode == 0, "SOURCE_CHANGED",
            f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def derive_data_identity(manifest):
    """Derive the resolved input hash contract without reading external bytes."""
    identities = {}
    for case in manifest["cases"]:
        case_id = case["id"]
        item = case["input"]
        kind = item["type"]
        if kind in ("librispeech_utterance", "fleurs_utterance"):
            identity = item["wav_sha256"]
        elif kind == "dataset_row":
            prompt = manifest["prompt_templates"]["gsm8k"].format(
                question=case["question"])
            identity = sha256_bytes(prompt.encode("utf-8"))
        elif kind == "image_url":
            prompt = manifest["prompt_templates"]["vqa"].format(
                question=case["question"])
            identity = {"image": item["sha256"],
                        "prompt": sha256_bytes(prompt.encode("utf-8"))}
        elif kind == "speechcoco_row":
            prompt = manifest["prompt_templates"]["joint"]
            identity = {"audio": item["audio_sha256"],
                        "image": item["image_sha256"],
                        "sent_audio": item["audio_sha256"],
                        "prompt": sha256_bytes(prompt.encode("utf-8"))}
        else:
            raise StabilityError("INVALID_MANIFEST",
                                 f"{case_id}: unsupported input type {kind}")
        identities[case_id] = identity
    return {"case_input_sha256": identities,
            "case_input_sha256_digest": sha256_bytes(canonical(identities))}


def normalize_config(raw, config_path, replay=False):
    config_path = Path(config_path).resolve()
    known_keys(raw, ("schema_version", "run_id", "evidence_dir", "source",
                     "baseline", "manifest", "cache_dir", "binaries", "models",
                     "request", "topology", "schedule", "tolerances"), "config")
    require(raw["schema_version"] == CONFIG_SCHEMA, "INVALID_CONFIG",
            f"schema_version must be {CONFIG_SCHEMA}")
    require(isinstance(raw["run_id"], str) and RUN_ID.fullmatch(raw["run_id"]),
            "INVALID_CONFIG", "run_id must be a bounded lowercase identifier")
    config = copy.deepcopy(raw)
    config["evidence_dir"] = absolute_path(raw["evidence_dir"], "evidence_dir")
    config["cache_dir"] = absolute_path(raw["cache_dir"], "cache_dir")

    known_keys(raw["source"], ("root", "commit", "require_clean",
                               "harness_sha256", "auditor_sha256"), "source")
    source = config["source"]
    root = Path(source["root"])
    if not root.is_absolute():
        root = config_path.parent / root
    source["root"] = str(root.resolve())
    require(Path(source["root"]) == ROOT, "INVALID_CONFIG",
            "source.root must be the checkout containing this orchestrator")
    source["commit"] = hash160(source["commit"], "source.commit",
                               allow_head=not replay)
    require(type(source["require_clean"]) is bool and (replay or source["require_clean"]),
            "INVALID_CONFIG", "live source.require_clean must be true")
    hash256(source["harness_sha256"], "source.harness_sha256")
    hash256(source["auditor_sha256"], "source.auditor_sha256")

    known_keys(raw["baseline"], ("path", "sha256"), "baseline")
    config["baseline"]["path"] = absolute_path(raw["baseline"]["path"],
                                                "baseline.path")
    hash256(raw["baseline"]["sha256"], "baseline.sha256")
    known_keys(raw["manifest"], ("path", "sha256"), "manifest")
    manifest_relative = safe_relative(raw["manifest"]["path"], "manifest.path")
    config["manifest"]["path"] = str(Path(source["root"]) / manifest_relative)
    hash256(raw["manifest"]["sha256"], "manifest.sha256")

    known_keys(raw["binaries"], ("gateway", "asr", "vlm"), "binaries")
    for role in ("gateway", "asr", "vlm"):
        known_keys(raw["binaries"][role], ("path", "sha256", "source_commit"),
                   f"binaries.{role}")
        item = config["binaries"][role]
        item["path"] = absolute_path(item["path"], f"binaries.{role}.path")
        hash256(item["sha256"], f"binaries.{role}.sha256")
        hash160(item["source_commit"], f"binaries.{role}.source_commit")

    known_keys(raw["models"], ("asr", "vlm"), "models")
    for role in ("asr", "vlm"):
        known_keys(raw["models"][role], ("id", "root", "config_files",
                                         "weights_manifest"), f"models.{role}")
        model = config["models"][role]
        require(isinstance(model["id"], str) and model["id"], "INVALID_CONFIG",
                f"models.{role}.id must be non-empty")
        model["root"] = absolute_path(model["root"], f"models.{role}.root")
        require(isinstance(model["config_files"], dict) and model["config_files"],
                "INVALID_CONFIG", f"models.{role}.config_files must be non-empty")
        for name, digest in model["config_files"].items():
            safe_relative(name, f"models.{role}.config_files key")
            hash256(digest, f"models.{role}.config_files.{name}")
        known_keys(model["weights_manifest"], ("path", "sha256"),
                   f"models.{role}.weights_manifest")
        model["weights_manifest"]["path"] = absolute_path(
            model["weights_manifest"]["path"],
            f"models.{role}.weights_manifest.path")
        hash256(model["weights_manifest"]["sha256"],
                f"models.{role}.weights_manifest.sha256")

    known_keys(raw["request"], ("model", "temperature", "top_p", "seed",
                                "max_tokens", "asr_vlm_max_tokens",
                                "asr_max_new_tokens", "stream",
                                "asr_route", "timeout_s", "case_latency_cap_ms",
                                "retry"), "request")
    request = config["request"]
    require(isinstance(request["model"], str) and request["model"],
            "INVALID_CONFIG", "request.model must be non-empty")
    require(finite_number(request["temperature"], "request.temperature") == 0.0,
            "INVALID_CONFIG", "the stability gate requires temperature=0")
    require(request["top_p"] is None and request["seed"] is None,
            "INVALID_CONFIG", "top_p and seed must be null (omitted by this runner)")
    positive_int(request["max_tokens"], "request.max_tokens")
    positive_int(request["asr_vlm_max_tokens"], "request.asr_vlm_max_tokens")
    positive_int(request["asr_max_new_tokens"], "request.asr_max_new_tokens")
    require(request["stream"] is True and request["retry"] is False,
            "INVALID_CONFIG", "stream=true and retry=false are required")
    require(request["asr_route"] == "chat", "INVALID_CONFIG",
            "asr_route=chat is required so every ASR case has raw SSE and a terminal")
    finite_number(request["timeout_s"], "request.timeout_s", 0.001)
    positive_int(request["case_latency_cap_ms"], "request.case_latency_cap_ms")

    known_keys(raw["topology"], ("base_url", "ipc_namespace", "gpu_by_stage",
                                 "allowed_idle_gpus", "resident_port_forbidden"),
               "topology")
    topology = config["topology"]
    parsed = urlsplit(topology["base_url"])
    require(parsed.scheme in ("http", "https") and
            parsed.hostname in ("127.0.0.1", "localhost", "::1") and
            parsed.port is not None and not parsed.query and not parsed.fragment
            and parsed.path in ("", "/") and parsed.username is None
            and parsed.password is None,
            "UNSAFE_TOPOLOGY", "base_url must be an isolated loopback endpoint")
    require(parsed.port != RESIDENT_PORT and
            topology["resident_port_forbidden"] == RESIDENT_PORT,
            "UNSAFE_TOPOLOGY", "the resident service endpoint is forbidden")
    topology["base_url"] = topology["base_url"].rstrip("/")
    topology["ipc_namespace"] = absolute_path(topology["ipc_namespace"],
                                               "topology.ipc_namespace")
    require(topology["ipc_namespace"] not in ("/", "/tmp"),
            "UNSAFE_TOPOLOGY", "ipc_namespace must be an isolated child path")
    known_keys(topology["gpu_by_stage"], ("asr", "vlm"), "topology.gpu_by_stage")
    gpus = topology["gpu_by_stage"]
    require(all(type(gpus[role]) is int and gpus[role] in ALLOWED_GPUS
                for role in ("asr", "vlm")) and gpus["asr"] != gpus["vlm"],
            "UNSAFE_TOPOLOGY", "ASR/VLM must use distinct declared idle GPUs 0, 1, or 7")
    allowed = topology["allowed_idle_gpus"]
    require(isinstance(allowed, list) and allowed and
            all(type(index) is int and index in ALLOWED_GPUS for index in allowed)
            and len(set(allowed)) == len(allowed)
            and set(gpus.values()).issubset(allowed),
            "UNSAFE_TOPOLOGY", "allowed_idle_gpus may contain only unique 0, 1, 7")

    known_keys(raw["schedule"], ("warmup_rounds", "measured_rounds", "case_order",
                                 "runner_timeout_s"), "schedule")
    schedule = config["schedule"]
    require(schedule["warmup_rounds"] == WARMUP_ROUNDS and
            schedule["measured_rounds"] == REQUIRED_ROUNDS and
            schedule["case_order"] == "manifest",
            "INVALID_SCHEDULE", "exactly one warmup and three manifest-order rounds required")
    positive_int(schedule["runner_timeout_s"], "schedule.runner_timeout_s")

    known_keys(raw["tolerances"], ("quality", "median_relative", "p95_relative",
                                   "latency_tasks"), "tolerances")
    tolerances = config["tolerances"]
    known_keys(tolerances["quality"], tuple(QUALITY_DIRECTION), "tolerances.quality")
    for key in QUALITY_DIRECTION:
        finite_number(tolerances["quality"][key], f"tolerances.quality.{key}")
    finite_number(tolerances["median_relative"], "tolerances.median_relative")
    finite_number(tolerances["p95_relative"], "tolerances.p95_relative")
    require(isinstance(tolerances["latency_tasks"], list) and
            set(tolerances["latency_tasks"]) == set(TASKS) and
            len(tolerances["latency_tasks"]) == len(TASKS),
            "INVALID_CONFIG", "median and p95 gates must cover every task")

    evidence = Path(config["evidence_dir"])
    require(evidence != Path(source["root"]) and
            not evidence.is_relative_to(Path(source["root"])),
            "INVALID_CONFIG", "evidence_dir must be outside the clean source checkout")
    config["declared_config_path"] = str(config_path)
    config["declared_config_sha256"] = sha256_file(config_path)
    if source["commit"] == "HEAD":
        source["commit"] = git_value(source["root"], "rev-parse", "--verify", "HEAD")
    source["tree"] = git_value(source["root"], "rev-parse", "HEAD^{tree}")
    return config


def load_manifest(config):
    path = Path(config["manifest"]["path"])
    require(path.is_file(), "MANIFEST_CHANGED", f"manifest missing: {path}")
    actual = sha256_file(path)
    require(actual == config["manifest"]["sha256"], "MANIFEST_CHANGED",
            "manifest SHA-256 differs from the run config", expected=config["manifest"]["sha256"],
            actual=actual)
    try:
        manifest = data.load_manifest(str(path))
    except (data.DataError, OSError, json.JSONDecodeError) as error:
        raise StabilityError("INVALID_MANIFEST", str(error))
    require(manifest.get("total_cases") == len(manifest.get("cases", [])) == 70,
            "INVALID_MANIFEST", "stability requires the complete unique 70-case manifest")
    ids = [case["id"] for case in manifest["cases"]]
    require(len(ids) == len(set(ids)), "INVALID_MANIFEST", "duplicate manifest case id")
    config["data"] = derive_data_identity(manifest)
    return manifest


def metric_values(case, metric):
    """Flatten every declared primary and auxiliary quality value."""
    if not isinstance(metric, dict):
        return {}
    task = case["task"]
    value = metric.get("value")
    detail = metric.get("detail") or {}
    if task == "asr" and case.get("language") == "en":
        return {"asr_en_wer": value, "asr_en_cer": detail.get("cer")}
    if task == "asr" and case.get("language") == "zh":
        return {"asr_zh_cer": value, "asr_zh_wer": detail.get("wer")}
    if task == "gsm8k":
        return {"gsm8k_accuracy": value}
    if task == "vqa":
        return {"vqa_accuracy": value}
    if task == "joint" and isinstance(value, dict):
        return {"joint_asr_wer": value.get("asr_wer"),
                "joint_keyword_hit": value.get("keyword_hit")}
    return {}


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def load_baseline(config, manifest, replay=False):
    path = Path(config["baseline"]["path"])
    require(path.is_file(), "BASELINE_MISSING", f"accepted baseline missing: {path}")
    actual_sha = sha256_file(path)
    require(actual_sha == config["baseline"]["sha256"], "BASELINE_SHA256_MISMATCH",
            "accepted baseline bytes differ from the pinned SHA-256",
            expected=config["baseline"]["sha256"], actual=actual_sha)
    baseline = load_json(path)
    require(isinstance(baseline, dict), "BASELINE_UNSAFE_SCHEMA",
            "baseline JSON root must be an object")
    require(baseline.get("schema_version") == BASELINE_SCHEMA,
            "BASELINE_UNSAFE_SCHEMA", "baseline is not the combined coverage schema")
    require(baseline.get("status") == "PASS" and
            baseline.get("coverage_complete") is True,
            "BASELINE_INCOMPLETE", "baseline coverage audit did not pass completely")
    if replay:
        require(baseline.get("real_model_evidence") is False and
                baseline.get("evidence_class") == "NOT_REAL_EVIDENCE",
                "BASELINE_UNSAFE_REPLAY", "replay baseline must be synthetic and labelled")
    else:
        require(baseline.get("real_model_evidence") is True and
                baseline.get("evidence_class") in (None, "real"),
                "BASELINE_NOT_REAL", "contract/replay evidence cannot be a real baseline")
        require(baseline.get("real_cases") == 70 and
                baseline.get("unique_real_cases") == 70 and
                baseline.get("http_200") == 70,
                "BASELINE_NOT_REAL", "real baseline must document 70 unique HTTP-200 cases")
    require(baseline.get("declared_cases") == 70 and
            baseline.get("completed_terminals") == 70,
            "BASELINE_INCOMPLETE", "baseline must declare 70 cases and completed terminals")
    require(baseline.get("stability_verdict") == "BASELINE_REQUIRED" and
            baseline.get("stability_pass_claim") is False,
            "BASELINE_UNSAFE_CLAIM", "coverage-only baseline must not claim stability")
    require(baseline.get("manifest_version") == manifest["manifest_version"] and
            baseline.get("manifest_sha256") == config["manifest"]["sha256"],
            "BASELINE_MANIFEST_MISMATCH", "baseline and run manifest identities differ")
    rows = baseline.get("case_results")
    require(isinstance(rows, list), "BASELINE_INCOMPLETE", "baseline case_results missing")
    by_case = {}
    for row in rows:
        require(isinstance(row, dict), "BASELINE_INCOMPLETE",
                "baseline case_results entries must be objects")
        case_id = row.get("case_id")
        require(isinstance(case_id, str) and case_id, "BASELINE_INCOMPLETE",
                "baseline case id must be a non-empty string")
        require(case_id not in by_case, "BASELINE_INCOMPLETE",
                f"duplicate baseline case: {case_id}")
        by_case[case_id] = row
    expected_ids = {case["id"] for case in manifest["cases"]}
    require(set(by_case) == expected_ids and len(rows) == 70,
            "BASELINE_INCOMPLETE", "baseline must contain exactly the 70 declared cases",
            missing=sorted(expected_ids - set(by_case)),
            unexpected=sorted(set(by_case) - expected_ids))
    require(baseline.get("missing_metric_case_ids") == [],
            "BASELINE_METRICS_MISSING", "baseline explicitly contains missing metrics")
    input_details = (baseline.get("input_verification") or {}).get("cases_detail")
    require(isinstance(input_details, dict), "BASELINE_INCOMPLETE",
            "baseline input verification is missing")
    for case in manifest["cases"]:
        case_id = case["id"]
        row = by_case[case_id]
        require(row.get("task") == case["task"] and
                row.get("language") == case.get("language") and
                row.get("completed_terminal") is True,
                "BASELINE_INCOMPLETE", f"{case_id}: incomplete or changed baseline identity")
        values = metric_values(case, row.get("metric"))
        require(set(values) == set(metric_values(case, {"value": {}, "detail": {}}))
                or values, "BASELINE_INCOMPLETE", f"{case_id}: unknown metric shape")
        require(values and all(numeric(value) for value in values.values()),
                "BASELINE_METRICS_MISSING", f"{case_id}: null/non-numeric baseline metric",
                metrics=values)
        latency = row.get("latency")
        require(isinstance(latency, dict) and
                all(numeric(latency.get(key)) for key in
                    ("total_ms", "first_delta_ms", "terminal_ms")) and
                ("audio" not in case["modalities"] or numeric(latency.get("asr_final_ms"))),
                "BASELINE_INCOMPLETE", f"{case_id}: missing baseline timing evidence")
        detail = input_details.get(case_id) or {}
        require(detail.get("input_sha256") ==
                config["data"]["case_input_sha256"][case_id],
                "BASELINE_DATA_MISMATCH", f"{case_id}: baseline input hash differs")
    normalized_rows = []
    for row in rows:
        normalized = copy.deepcopy(row)
        normalized["terminal_kind"] = "complete"
        normalized["failures"] = []
        normalized_rows.append(normalized)
    recomputed = audit.recompute_aggregate(normalized_rows)
    try:
        baseline_aggregates = baseline.get("aggregates")
        require(isinstance(baseline_aggregates, dict) and
                isinstance(baseline_aggregates.get("tasks"), dict),
                "BASELINE_INCOMPLETE", "baseline aggregates/tasks must be objects")
        baseline_tasks = baseline_aggregates["tasks"]
        require(set(baseline_tasks) == set(TASKS), "BASELINE_INCOMPLETE",
                "baseline task coverage differs from the five declared tasks")
        for task, expected in baseline_tasks.items():
            for key, expected_value in expected.items():
                if key in ("tokens_per_s_p50", "tokens_per_s_p95"):
                    continue  # token-rate details are not copied into case_results
                audit.assert_same(recomputed["tasks"][task].get(key), expected_value,
                                  f"baseline.aggregates.tasks.{task}.{key}")
    except audit.AuditError as error:
        raise StabilityError("BASELINE_AGGREGATE_MISMATCH", str(error))
    return {"path": str(path), "sha256": actual_sha, "document": baseline,
            "by_case": by_case, "aggregates": baseline["aggregates"]}


def verify_segmenter():
    try:
        import jieba
    except ImportError as error:
        raise StabilityError("SEGMENTER_REQUIRED",
                             "pinned jieba 0.42.1 is required") from error
    require(jieba.__version__ == "0.42.1", "SEGMENTER_CHANGED",
            f"jieba 0.42.1 required, got {jieba.__version__}")


def verify_source(config):
    source = config["source"]
    current = git_value(source["root"], "rev-parse", "--verify", "HEAD")
    require(current == source["commit"], "SOURCE_CHANGED",
            "source HEAD differs from the frozen run commit",
            expected=source["commit"], actual=current)
    if source["require_clean"]:
        dirty = git_value(source["root"], "status", "--porcelain",
                          "--untracked-files=all")
        require(not dirty, "SOURCE_CHANGED", "tracked source checkout is dirty", diff=dirty)
    for relative, expected, code in (
            ("tests/multimodal_benchmark.py", source["harness_sha256"], "HARNESS_CHANGED"),
            ("tools/multimodal_coverage_audit.py", source["auditor_sha256"],
             "AUDITOR_CHANGED")):
        actual = sha256_file(Path(source["root"]) / relative)
        require(actual == expected, code, f"{relative} bytes changed",
                expected=expected, actual=actual)
    orchestrator = sha256_file(Path(__file__).resolve())
    return {"commit": current, "tree": source["tree"],
            "harness_sha256": source["harness_sha256"],
            "auditor_sha256": source["auditor_sha256"],
            "orchestrator_sha256": orchestrator, "clean": source["require_clean"]}


def parse_weight_manifest(path):
    entries = []
    seen = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = CHECKSUM_LINE.fullmatch(line)
        require(match is not None, "MODEL_WEIGHT_MANIFEST_INVALID",
                f"invalid weight checksum line in {path}")
        expected, _, name = match.groups()
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts
                and name not in seen and relative.suffix == ".safetensors",
                "MODEL_WEIGHT_MANIFEST_UNSAFE", f"unsafe/duplicate weight entry: {name!r}")
        seen.add(name)
        entries.append((name, expected))
    require(entries, "MODEL_WEIGHT_MANIFEST_INVALID", f"empty weight manifest: {path}")
    return entries


def verify_artifacts(config, replay=False):
    actual_config = sha256_file(config["declared_config_path"])
    require(actual_config == config["declared_config_sha256"], "RUN_CONFIG_CHANGED",
            "declared config changed after the run was frozen",
            expected=config["declared_config_sha256"], actual=actual_config)
    actual_manifest = sha256_file(config["manifest"]["path"])
    require(actual_manifest == config["manifest"]["sha256"], "MANIFEST_CHANGED",
            "manifest changed after the run was frozen",
            expected=config["manifest"]["sha256"], actual=actual_manifest)
    provenance = {"source": verify_source(config) if not replay else {
        "commit": config["source"]["commit"], "tree": config["source"]["tree"],
        "synthetic": True}}
    provenance["binaries"] = {}
    for role, item in config["binaries"].items():
        actual = sha256_file(item["path"])
        require(actual == item["sha256"], "BINARY_CHANGED",
                f"{role} binary bytes changed", expected=item["sha256"], actual=actual)
        provenance["binaries"][role] = {**item, "actual_sha256": actual}
    provenance["models"] = {}
    for role, model in config["models"].items():
        model_record = {"id": model["id"], "root": model["root"], "config_files": {},
                        "weights": {}}
        for relative, expected in model["config_files"].items():
            path = Path(model["root"]) / relative
            actual = sha256_file(path)
            require(actual == expected, "MODEL_CONFIG_CHANGED",
                    f"{role} config changed: {relative}", expected=expected, actual=actual)
            model_record["config_files"][relative] = actual
        manifest_path = Path(model["weights_manifest"]["path"])
        actual_manifest = sha256_file(manifest_path)
        require(actual_manifest == model["weights_manifest"]["sha256"],
                "MODEL_WEIGHT_MANIFEST_CHANGED", f"{role} weight manifest changed",
                expected=model["weights_manifest"]["sha256"], actual=actual_manifest)
        model_record["weights"]["manifest_path"] = str(manifest_path)
        model_record["weights"]["manifest_sha256"] = actual_manifest
        entries = parse_weight_manifest(manifest_path)
        available = sorted(path.relative_to(model["root"]).as_posix()
                           for path in Path(model["root"]).rglob("*.safetensors")
                           if path.is_file())
        require(len(available) == len(set(available)) and
                set(available) == {name for name, _ in entries},
                "MODEL_WEIGHT_SET_CHANGED",
                f"{role} available Safetensors set differs from its pinned manifest",
                available=available, declared=[name for name, _ in entries])
        model_record["weights"]["files"] = {}
        for relative, expected in entries:
            target = Path(model["root"]) / relative
            actual = sha256_file(target)
            require(actual == expected, "MODEL_WEIGHT_CHANGED",
                    f"{role} available weight changed: {relative}",
                    expected=expected, actual=actual)
            model_record["weights"]["files"][relative] = actual
        provenance["models"][role] = model_record
    return provenance


def check_data_identities(config, actual):
    expected = config["data"]["case_input_sha256"]
    require(actual == expected, "DATA_CHANGED", "resolved input identities changed",
            actual_digest=sha256_bytes(canonical(actual)),
            expected_digest=config["data"]["case_input_sha256_digest"])
    return {"case_input_sha256": actual,
            "case_input_sha256_digest": sha256_bytes(canonical(actual)),
            "verified_cases": len(actual)}


def verify_resolved_data(config, manifest):
    try:
        resolved = data.resolve_cases(manifest, list(manifest["cases"]),
                                      Path(config["cache_dir"]))
    except (data.DataError, OSError) as error:
        raise StabilityError("DATA_CHANGED", f"cannot resolve pinned data: {error}")
    actual = {}
    for case in manifest["cases"]:
        case_id = case["id"]
        item = resolved[case_id]
        require(item.get("source") != "mock", "DATA_NOT_REAL",
                f"{case_id}: mock input in a real run")
        actual[case_id] = item.get("input_sha256")
    return check_data_identities(config, actual)


def build_schedule(config, manifest):
    ids = [case["id"] for case in manifest["cases"]]
    order_sha = sha256_bytes(canonical(ids))
    steps = []
    for phase, number in [("warmup", 1)] + [
            ("measured", number) for number in range(1, REQUIRED_ROUNDS + 1)]:
        steps.append({"index": len(steps), "phase": phase, "round": number,
                      "directory": "warmup" if phase == "warmup" else f"round-{number}",
                      "case_order": "manifest", "case_order_sha256": order_sha,
                      "case_ids": ids})
    schedule = {"schema_version": SCHEDULE_SCHEMA, "run_id": config["run_id"],
                "warmup_rounds": 1, "measured_rounds": 3, "steps": steps}
    schedule["schedule_sha256"] = sha256_bytes(canonical(schedule))
    return schedule


def quality_comparison(name, baseline, current, tolerance):
    result = {"metric": name, "baseline": baseline, "current": current,
              "tolerance": tolerance, "direction": QUALITY_DIRECTION[name],
              "coverage": "COVERED", "ok": False}
    if not numeric(baseline) or not numeric(current):
        result.update({"coverage": "MISSING", "delta": None, "regression": None,
                       "reason": "MISSING_METRIC"})
        return result
    delta = float(current) - float(baseline)
    regression = delta if QUALITY_DIRECTION[name] == "lower" else -delta
    result.update({"delta": delta, "regression": regression,
                   "ok": regression <= tolerance + 1e-12})
    if not result["ok"]:
        result["reason"] = "QUALITY_REGRESSION"
    return result


def latency_comparison(baseline, current, tolerance, statistic, task):
    result = {"task": task, "statistic": statistic, "baseline_ms": baseline,
              "current_ms": current, "relative_tolerance": tolerance,
              "coverage": "COVERED", "blocking": True, "ok": False}
    if not numeric(baseline) or not numeric(current) or baseline <= 0 or current < 0:
        result.update({"coverage": "MISSING", "relative_delta": None,
                       "reason": "MISSING_LATENCY"})
        return result
    delta = (float(current) - float(baseline)) / float(baseline)
    result.update({"relative_delta": delta, "ok": abs(delta) <= tolerance + 1e-12})
    if not result["ok"]:
        result["reason"] = "LATENCY_REGRESSION"
    return result


def compare_round(rows, verification, baseline, config, manifest):
    current_by_case = {row["case_id"]: row for row in rows
                       if row.get("case_id") in verification["raw_valid_case_ids"]}
    cases = []
    for case in manifest["cases"]:
        case_id = case["id"]
        base_values = metric_values(case, baseline["by_case"][case_id]["metric"])
        row = current_by_case.get(case_id)
        current_values = metric_values(case, row.get("metric")) if row else {}
        comparisons = []
        for name, base_value in base_values.items():
            comparisons.append(quality_comparison(
                name, base_value, current_values.get(name),
                config["tolerances"]["quality"][name]))
        case_covered = bool(row and comparisons and all(
            numeric(item["current"]) for item in comparisons))
        cases.append({"case_id": case_id, "task": case["task"],
                      "coverage": "COVERED" if case_covered else "MISSING",
                      "quality": comparisons,
                      "response_sha256": (verification["response_sha256"].get(case_id)
                                          if row else None),
                      "latency": (row.get("latency") if row else None),
                      "baseline_latency": baseline["by_case"][case_id].get("latency")})
    aggregate_quality = []
    for name in QUALITY_DIRECTION:
        task = name.rsplit("_", 1)[0] if name.startswith("asr_") else None
        if name.startswith("asr_en_"):
            task, field = "asr_en", name[len("asr_en_"):]
        elif name.startswith("asr_zh_"):
            task, field = "asr_zh", name[len("asr_zh_"):]
        else:
            field = name
            task = {"vqa_accuracy": "vqa", "gsm8k_accuracy": "gsm8k",
                    "joint_asr_wer": "joint", "joint_keyword_hit": "joint"}[name]
        base_value = baseline["aggregates"]["tasks"][task].get(field)
        current_value = verification["aggregates"]["tasks"].get(task, {}).get(field)
        comparison = quality_comparison(name, base_value, current_value,
                                        config["tolerances"]["quality"][name])
        task_complete = task in verification["complete_metric_tasks"]
        comparison["task_coverage"] = "COVERED" if task_complete else "MISSING"
        comparison["ok"] = comparison["ok"] and task_complete
        if not task_complete:
            comparison["reason"] = "INCOMPLETE_METRIC_COVERAGE"
        aggregate_quality.append(comparison)
    latency = []
    for task in TASKS:
        baseline_task = baseline["aggregates"]["tasks"].get(task, {})
        current_task = verification["aggregates"]["tasks"].get(task, {})
        for statistic, field, tolerance_key in (
                ("median", "p50_ms", "median_relative"),
                ("p95", "p95_ms", "p95_relative")):
            comparison = latency_comparison(
                baseline_task.get(field), current_task.get(field),
                config["tolerances"][tolerance_key], statistic, task)
            complete = task in verification["complete_latency_tasks"]
            comparison["task_coverage"] = "COVERED" if complete else "MISSING"
            comparison["ok"] = comparison["ok"] and complete
            if not complete:
                comparison["reason"] = "INCOMPLETE_LATENCY_COVERAGE"
            latency.append(comparison)
    quality_ok = all(item["ok"] for case in cases for item in case["quality"]) and \
        all(item["ok"] for item in aggregate_quality)
    median_ok = all(item["ok"] for item in latency if item["statistic"] == "median")
    p95_ok = all(item["ok"] for item in latency if item["statistic"] == "p95")
    return {"schema_version": "umm-multimodal-stability-comparison/v1",
            "case_count": len(cases), "cases": cases,
            "aggregate_quality": aggregate_quality, "latency": latency,
            "quality_ok": quality_ok, "median_ok": median_ok, "p95_ok": p95_ok,
            "missing_case_ids": verification["missing_case_ids"],
            "missing_metric_case_ids": verification["missing_metric_case_ids"],
            "zero_valued_metrics": verification["zero_valued_metrics"],
            "coverage_complete": verification["complete"],
            "passed": quality_ok and median_ok and p95_ok and verification["complete"]}


def verify_round(round_dir, baseline, config, manifest, round_number, replay=False):
    round_dir = Path(round_dir)
    verification = {"schema_version": "umm-multimodal-stability-round-audit/v1",
                    "round": round_number, "complete": False,
                    "expected_cases": 70, "case_count": 0,
                    "raw_errors": [], "raw_valid_case_ids": [],
                    "response_sha256": {}, "missing_metric_case_ids": [],
                    "zero_valued_metrics": [], "missing_case_ids": [],
                    "unexpected_case_ids": []}
    try:
        report = load_json(round_dir / "report.json")
        if not isinstance(report, dict):
            raise StabilityError("PARTIAL_ROUND", "report.json root must be an object")
        rows = [json.loads(line) for line in
                (round_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
        if not all(isinstance(row, dict) and isinstance(row.get("case_id"), str)
                   and row.get("case_id") for row in rows):
            raise StabilityError("PARTIAL_ROUND",
                                 "cases.jsonl rows need non-empty string case_id values")
    except (StabilityError, OSError, json.JSONDecodeError) as error:
        verification["raw_errors"].append({"code": "PARTIAL_ROUND", "message": str(error)})
        verification["missing_case_ids"] = [case["id"] for case in manifest["cases"]]
        verification["aggregates"] = {"tasks": {}}
        verification["complete_metric_tasks"] = []
        verification["complete_latency_tasks"] = []
        verification["hard_failures"] = ["PARTIAL_ROUND"]
        return [], verification, compare_round([], verification, baseline, config, manifest)

    evidence_ok = (report.get("evidence_class") == ("contract-test" if replay else "real")
                   and report.get("real_model_evidence") is (not replay))
    run = report.get("run") if isinstance(report.get("run"), dict) else {}
    report_config = report.get("config") if isinstance(report.get("config"), dict) else {}
    ids = [row.get("case_id") for row in rows]
    expected_ids = [case["id"] for case in manifest["cases"]]
    verification.update({
        "case_count": len(rows), "unique_case_count": len(set(ids)),
        "duplicate_case_ids": sorted({case_id for case_id in ids if ids.count(case_id) > 1}),
        "missing_case_ids": sorted(set(expected_ids) - set(ids)),
        "unexpected_case_ids": sorted(set(ids) - set(expected_ids)),
        "manifest_order_match": ids == expected_ids,
        "report_cases_match": report.get("cases") == rows,
        "manifest_version_match": report.get("manifest_version") == manifest["manifest_version"],
        "evidence_class_match": evidence_ok,
        "endpoint_match": run.get("base_url") == config["topology"]["base_url"],
        "request_config_match":
            run.get("model") == config["request"]["model"] and
            run.get("asr_route") == config["request"]["asr_route"] and
            report_config.get("max_tokens") == config["request"]["max_tokens"] and
            report_config.get("asr_vlm_max_tokens") ==
                config["request"]["asr_vlm_max_tokens"] and
            report_config.get("asr_max_new_tokens") ==
                config["request"]["asr_max_new_tokens"] and
            report_config.get("temperature") == config["request"]["temperature"] and
            report_config.get("top_p") is None and report_config.get("seed") is None and
            report_config.get("retry") is False and
            report_config.get("timeout_s") == config["request"]["timeout_s"] and
            report_config.get("case_latency_cap_ms") ==
                config["request"]["case_latency_cap_ms"] and
            report_config.get("p95_tolerance") == config["tolerances"]["p95_relative"],
    })
    cases = {case["id"]: case for case in manifest["cases"]}
    verification["request_rows_match"] = all(
        isinstance(row.get("request"), dict) and
        row["request"].get("model") == config["request"]["model"] and
        row["request"].get("stream") is True and
        row["request"].get("temperature") == config["request"]["temperature"] and
        row["request"].get("top_p") is None and row["request"].get("seed") is None and
        row["request"].get("retry") is False and
        row["request"].get("max_tokens") ==
        (config["request"]["asr_vlm_max_tokens"] if row.get("task") == "asr"
         else config["request"]["max_tokens"])
        for row in rows)
    for row in rows:
        case_id = row.get("case_id")
        case = cases.get(case_id)
        if case is None:
            continue
        try:
            independent = audit.recompute_metric(case, row)
            audit.assert_same(row.get("metric"), independent, f"{case_id}.metric")
            raw = audit.rebuild_sse(round_dir, row)
            require(row.get("input_sha256") == config["data"]["case_input_sha256"][case_id],
                    "DATA_CHANGED", f"{case_id}: row input identity changed")
            verification["raw_valid_case_ids"].append(case_id)
            verification["response_sha256"][case_id] = {
                "answer": raw["answer_sha256"], "transcript": raw["transcript_sha256"]}
            for name, value in metric_values(case, independent).items():
                if value is None:
                    verification["missing_metric_case_ids"].append(
                        {"case_id": case_id, "metric": name})
                elif numeric(value) and value == 0:
                    verification["zero_valued_metrics"].append(
                        {"case_id": case_id, "metric": name, "value": 0})
        except (audit.AuditError, StabilityError, KeyError, TypeError, ValueError,
                OSError, AttributeError) as error:
            verification["raw_errors"].append({
                "case_id": case_id, "code": getattr(error, "reason", {}).get("code",
                                                                             "RAW_SSE_CONTRACT"),
                "message": str(error)})
    aggregates = None
    try:
        aggregates = audit.recompute_aggregate(rows)
        audit.assert_same(report.get("aggregates"), aggregates, "round.aggregates")
        verification["aggregates_match"] = True
    except (audit.AuditError, KeyError, TypeError, ValueError, AttributeError) as error:
        verification["aggregates_match"] = False
        verification["raw_errors"].append({"code": "AGGREGATE_MISMATCH", "message": str(error)})
    if aggregates is None:
        aggregates = {"tasks": {}, "completed_terminals": 0}
    verification["aggregates"] = aggregates
    complete_metric_tasks = []
    for task in TASKS:
        task_cases = [case for case in manifest["cases"]
                      if (("asr_" + case.get("language", "")) if case["task"] == "asr"
                          else case["task"]) == task]
        valid = [case for case in task_cases
                 if case["id"] in verification["raw_valid_case_ids"]]
        if len(valid) == len(task_cases) and all(
                all(numeric(value) for value in metric_values(
                    case, next(row for row in rows if row["case_id"] == case["id"])["metric"]).values())
                for case in valid):
            complete_metric_tasks.append(task)
    verification["complete_metric_tasks"] = complete_metric_tasks
    verification["complete_latency_tasks"] = [
        task for task in TASKS
        if task in complete_metric_tasks and all(
            numeric(row["latency"].get(field))
            for row in rows
            if ((row.get("task") == "asr" and "asr_" + str(row.get("language")) == task)
                or row.get("task") == task)
            for field in ("total_ms", "first_delta_ms", "terminal_ms"))]
    hard_failures = []
    for row in rows:
        row_failures = row.get("failures", [])
        if not isinstance(row_failures, list):
            hard_failures.append({"case_id": row.get("case_id"),
                                  "code": "INVALID_FAILURE_LIST"})
            continue
        for failure in row_failures:
            if not isinstance(failure, dict):
                hard_failures.append({"case_id": row.get("case_id"),
                                      "code": "INVALID_FAILURE"})
            elif failure.get("category") in ("infrastructure", "public_contract", "performance"):
                hard_failures.append({"case_id": row.get("case_id"), **failure})
    for key in ("report_cases_match", "manifest_version_match", "evidence_class_match",
                "endpoint_match", "request_config_match", "request_rows_match",
                "manifest_order_match", "aggregates_match"):
        if not verification.get(key):
            hard_failures.append({"code": "ROUND_CONTRACT_MISMATCH", "check": key})
    verification["hard_failures"] = hard_failures
    verification["complete"] = bool(
        len(rows) == len(set(ids)) == 70 and not verification["missing_case_ids"]
        and not verification["unexpected_case_ids"] and not verification["raw_errors"]
        and not hard_failures and len(complete_metric_tasks) == len(TASKS)
        and len(verification["complete_latency_tasks"]) == len(TASKS))
    comparison = compare_round(rows, verification, baseline, config, manifest)
    return rows, verification, comparison


def full_sha256sums(root):
    root = Path(root)
    files = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise StabilityError("UNSAFE_EVIDENCE", f"symlink in evidence: {path}")
        if path.is_file() and path != root / "SHA256SUMS":
            relative = path.relative_to(root).as_posix()
            safe_relative(relative, "evidence path")
            files.append((relative, path))
    lines = [f"{sha256_file(path)}  {relative}\n" for relative, path in sorted(files)]
    (root / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")
    return len(lines)


def round_reasons(round_number, verification, comparison):
    reasons = []
    if not verification["complete"]:
        reasons.append({"code": "PARTIAL_ROUND" if verification["case_count"] != 70
                        else "ROUND_CONTRACT_FAILED", "round": round_number,
                        "missing_case_ids": verification.get("missing_case_ids", []),
                        "raw_errors": verification.get("raw_errors", []),
                        "hard_failures": verification.get("hard_failures", [])})
    for case in comparison["cases"]:
        for item in case["quality"]:
            if not item["ok"]:
                reasons.append({"code": item.get("reason", "QUALITY_REGRESSION"),
                                "round": round_number, "case_id": case["case_id"],
                                "metric": item["metric"], "baseline": item["baseline"],
                                "current": item["current"], "delta": item["delta"],
                                "tolerance": item["tolerance"]})
    for item in comparison["aggregate_quality"]:
        if not item["ok"]:
            reasons.append({"code": item.get("reason", "QUALITY_REGRESSION"),
                            "round": round_number, "aggregate": item["metric"],
                            "baseline": item["baseline"], "current": item["current"]})
    for item in comparison["latency"]:
        if not item["ok"]:
            reasons.append({"code": item.get("reason", "LATENCY_REGRESSION"),
                            "round": round_number, "task": item["task"],
                            "statistic": item["statistic"],
                            "baseline_ms": item["baseline_ms"],
                            "current_ms": item["current_ms"],
                            "relative_delta": item["relative_delta"]})
    return reasons


def finalize_evidence(config, manifest, baseline, replay=False, run_completed=True,
                      fatal_reasons=None, executed=True):
    evidence = Path(config["evidence_dir"])
    evidence.mkdir(parents=True, exist_ok=True)
    write_json(evidence / "run-config.json", config)
    config_sha = sha256_file(evidence / "run-config.json")
    schedule = build_schedule(config, manifest)
    write_json(evidence / "schedule.json", schedule)
    baseline_name = "synthetic-baseline.json" if replay else "accepted-baseline.json"
    shutil.copyfile(baseline["path"], evidence / baseline_name)
    if replay:
        (evidence / "NOT_REAL_EVIDENCE").write_text(
            "Synthetic replay only. No real model, data, GPU, service, or baseline claim.\n",
            encoding="utf-8")
    def write_provenance(directory, number):
        try:
            provenance = verify_artifacts(config, replay=replay)
            provenance["topology"] = copy.deepcopy(config["topology"])
            provenance["data"] = copy.deepcopy(config["data"])
            provenance["run_config_sha256"] = config_sha
            write_json(directory / "provenance.json", provenance)
            return []
        except (StabilityError, OSError) as error:
            reason = error.reason if isinstance(error, StabilityError) else {
                "code": "PROVENANCE_ERROR", "message": str(error)}
            reasons = [{**reason, "round": number}]
            write_json(directory / "provenance-error.json", reasons)
            return reasons

    rounds = []
    reasons = list(fatal_reasons or [])
    warmup_dir = evidence / "warmup"
    warmup = {"present": warmup_dir.is_dir(), "complete": False, "passed": False}
    if warmup["present"]:
        provenance_reasons = write_provenance(warmup_dir, 0)
        _, warmup_verification, warmup_comparison = verify_round(
            warmup_dir, baseline, config, manifest, 0, replay=replay)
        write_json(warmup_dir / "verification.json", warmup_verification)
        write_json(warmup_dir / "comparison.json", warmup_comparison)
        reasons.extend(provenance_reasons)
        warmup.update({"complete": warmup_verification["complete"],
                       "passed": warmup_verification["complete"] and not provenance_reasons,
                       "case_count": warmup_verification.get("case_count", 0),
                       "completed_terminals": warmup_verification["aggregates"].get(
                           "completed_terminals", 0),
                       "verification_sha256": sha256_file(warmup_dir / "verification.json")})
        if run_completed and not warmup["passed"]:
            reasons.append({"code": "WARMUP_FAILED", "round": 0,
                            "raw_errors": warmup_verification.get("raw_errors", []),
                            "hard_failures": warmup_verification.get("hard_failures", [])})
        full_sha256sums(warmup_dir)
    elif run_completed:
        reasons.append({"code": "WARMUP_MISSING", "round": 0,
                        "message": "the declared complete warmup is absent"})
    for number in range(1, REQUIRED_ROUNDS + 1):
        round_dir = evidence / f"round-{number}"
        if not round_dir.is_dir():
            rounds.append({"round": number, "present": False, "complete": False,
                           "passed": False, "run_config_sha256": config_sha})
            continue
        provenance_reasons = write_provenance(round_dir, number)
        rows, verification, comparison = verify_round(
            round_dir, baseline, config, manifest, number, replay=replay)
        write_json(round_dir / "verification.json", verification)
        write_json(round_dir / "comparison.json", comparison)
        these_reasons = provenance_reasons + round_reasons(number, verification, comparison)
        reasons.extend(these_reasons)
        passed = verification["complete"] and comparison["passed"] and not provenance_reasons
        rounds.append({"round": number, "present": True,
                       "complete": verification["complete"], "passed": passed,
                       "run_config_sha256": config_sha,
                       "case_count": verification.get("case_count", 0),
                       "completed_terminals": verification["aggregates"].get(
                           "completed_terminals", 0),
                       "quality_ok": comparison["quality_ok"],
                       "median_ok": comparison["median_ok"], "p95_ok": comparison["p95_ok"],
                       "verification_sha256": sha256_file(round_dir / "verification.json"),
                       "comparison_sha256": sha256_file(round_dir / "comparison.json")})
        full_sha256sums(round_dir)
    consecutive = 0
    seen_present = False
    for item in reversed(rounds):
        if not item["present"]:
            if seen_present:
                break
            continue
        seen_present = True
        if item["passed"]:
            consecutive += 1
        else:
            break
    all_three = warmup["passed"] and len(rounds) == REQUIRED_ROUNDS and \
        all(item["passed"] for item in rounds)
    if all_three and not reasons and run_completed:
        mechanical = "STABILITY_PASS"
    elif reasons or any(item["present"] and not item["passed"] for item in rounds):
        mechanical = "FAIL"
    else:
        mechanical = "STABILITY_PENDING"
        reasons.append({"code": "MEASURED_ROUNDS_PENDING",
                        "required": REQUIRED_ROUNDS, "complete_matching": consecutive})
    verdict = "NOT_REAL_EVIDENCE" if replay else mechanical
    summary = {
        "schema_version": RESULT_SCHEMA, "run_id": config["run_id"],
        "evidence_class": "NOT_REAL_EVIDENCE" if replay else "real",
        "real_model_evidence": not replay and executed,
        "verdict": verdict, "stability_verdict": verdict,
        "replay_stability_verdict": mechanical if replay else None,
        "real_stability_pass_claim": mechanical == "STABILITY_PASS" and not replay and executed,
        "real_execution": "EXECUTED" if executed and not replay else "NOT_RUN",
        "baseline": {"path": baseline_name, "sha256": baseline["sha256"],
                     "schema_version": BASELINE_SCHEMA,
                     "coverage_only": True, "measured_rounds_merged": False},
        "run_config_sha256": config_sha,
        "schedule_sha256": schedule["schedule_sha256"],
        "manifest_sha256": config["manifest"]["sha256"],
        "data_case_input_sha256_digest": config["data"]["case_input_sha256_digest"],
        "warmup_rounds": 1, "warmup": warmup, "rounds_requested": 3,
        "rounds_run": sum(item["present"] for item in rounds),
        "consecutive_complete_matching_rounds": consecutive,
        "missing_vs_zero": "Missing cases/metrics/timings fail coverage; numeric zero remains a measurement.",
        "latency_gate": "Per-task median and nearest-rank p95 are both blocking at the declared relative tolerance.",
        "rounds": rounds, "reasons": reasons,
    }
    write_json(evidence / "stability.json", summary)
    write_json(evidence / "summary.json", summary)
    full_sha256sums(evidence)
    return summary


def move_runner_evidence(scratch, final):
    final.mkdir(parents=True, exist_ok=True)
    inner = scratch / "round-1"
    if inner.is_dir():
        for child in list(inner.iterdir()):
            shutil.move(str(child), str(final / child.name))
    for name in ("summary.json", "stability.json", "run.log"):
        source = scratch / name
        if source.is_file():
            shutil.move(str(source), str(final / f"runner-{name}"))
    for name in ("stdout.txt", "stderr.txt", "execution.json"):
        source = scratch / name
        if source.is_file():
            shutil.move(str(source), str(final / f"runner-{name}"))
    shutil.rmtree(scratch, ignore_errors=True)


def execute_runner(config, manifest, phase, number):
    evidence = Path(config["evidence_dir"])
    label = "warmup" if phase == "warmup" else f"round-{number}"
    final = evidence / label
    scratch = evidence / ".runner" / label
    scratch.mkdir(parents=True, exist_ok=False)
    request = config["request"]
    argv = [sys.executable, "-I", str(Path(config["source"]["root"]) /
                                      "tests/multimodal_benchmark.py"),
            "--base-url", config["topology"]["base_url"],
            "--model", request["model"], "--manifest", config["manifest"]["path"],
            "--cache-dir", config["cache_dir"], "--evidence-dir", str(scratch),
            "--rounds", "1", "--evidence-class", "real", "--asr-route", "chat",
            "--max-tokens", str(request["max_tokens"]),
            "--asr-vlm-max-tokens", str(request["asr_vlm_max_tokens"]),
            "--asr-max-new-tokens", str(request["asr_max_new_tokens"]),
            "--timeout", str(request["timeout_s"]),
            "--case-latency-cap-ms", str(request["case_latency_cap_ms"]),
            "--p95-tolerance", str(config["tolerances"]["p95_relative"])]
    execution = {"phase": phase, "round": number, "argv_without_credentials": argv,
                 "timed_out": False}
    try:
        with (scratch / "stdout.txt").open("w", encoding="utf-8") as stdout, \
                (scratch / "stderr.txt").open("w", encoding="utf-8") as stderr:
            try:
                result = subprocess.run(argv, stdout=stdout, stderr=stderr,
                                        timeout=config["schedule"]["runner_timeout_s"],
                                        check=False)
                execution["returncode"] = result.returncode
            except subprocess.TimeoutExpired:
                execution["timed_out"] = True
                execution["returncode"] = None
    finally:
        write_json(scratch / "execution.json", execution)
        move_runner_evidence(scratch, final)
    report_path = final / "report.json"
    summary_path = final / "runner-summary.json"
    if report_path.is_file() and summary_path.is_file():
        summary = load_json(summary_path)
        require(summary.get("stability_verdict") == "BASELINE_REQUIRED",
                "RUNNER_VERDICT_CHANGED", f"{label}: unexpected runner baseline verdict")
        # Exit 1 is the existing runner's BASELINE_REQUIRED result and is not
        # itself an execution failure. Other exits remain visible to the audit.
        require(execution.get("returncode") in (0, 1),
                "RUNNER_EXIT", f"{label}: runner exited {execution.get('returncode')}")
    else:
        require(False, "PARTIAL_ROUND", f"{label}: runner did not write a complete report",
                execution=execution)
    return final


def prepare_evidence_dir(config):
    evidence = Path(config["evidence_dir"])
    require(not evidence.exists() or (evidence.is_dir() and not any(evidence.iterdir())),
            "EVIDENCE_EXISTS", f"evidence directory is not empty: {evidence}")
    evidence.mkdir(parents=True, exist_ok=True)


def run_live(config_path, preflight_only=False):
    raw = load_json(config_path)
    # This read is intentionally sufficient to place even early rejection output.
    evidence_value = raw.get("evidence_dir") if isinstance(raw, dict) else None
    evidence = Path(evidence_value) if isinstance(evidence_value, str) and \
        Path(evidence_value).is_absolute() else None
    try:
        config = normalize_config(raw, config_path, replay=False)
        evidence = Path(config["evidence_dir"])
        prepare_evidence_dir(config)
        verify_segmenter()
        manifest = load_manifest(config)
        baseline = load_baseline(config, manifest, replay=False)
        write_json(evidence / "run-config.json", config)
        write_json(evidence / "schedule.json", build_schedule(config, manifest))
        shutil.copyfile(baseline["path"], evidence / "accepted-baseline.json")
        verify_artifacts(config)
        verify_resolved_data(config, manifest)
        if preflight_only:
            return finalize_evidence(config, manifest, baseline, run_completed=False,
                                     executed=False)
        fatal = []
        try:
            execute_runner(config, manifest, "warmup", 1)
            verify_artifacts(config)
            verify_resolved_data(config, manifest)
            _, warmup_verification, _ = verify_round(
                evidence / "warmup", baseline, config, manifest, 0, replay=False)
            require(warmup_verification["complete"], "WARMUP_FAILED",
                    "warmup failed independent full-manifest/raw-SSE verification",
                    raw_errors=warmup_verification.get("raw_errors", []),
                    hard_failures=warmup_verification.get("hard_failures", []))
            for number in range(1, REQUIRED_ROUNDS + 1):
                execute_runner(config, manifest, "measured", number)
                # Recheck after every attempt; never carry a changed stack into
                # the next round or bless it during finalization.
                verify_artifacts(config)
                verify_resolved_data(config, manifest)
        except (StabilityError, OSError) as error:
            fatal.append(error.reason if isinstance(error, StabilityError) else {
                "code": "ORCHESTRATION_ERROR", "message": str(error)})
        return finalize_evidence(config, manifest, baseline, fatal_reasons=fatal)
    except (StabilityError, OSError) as error:
        reason = error.reason if isinstance(error, StabilityError) else {
            "code": "ORCHESTRATION_ERROR", "message": str(error)}
        verdict = "BASELINE_REQUIRED" if reason["code"].startswith("BASELINE_") else "FAIL"
        summary = {"schema_version": RESULT_SCHEMA, "verdict": verdict,
                   "stability_verdict": verdict, "real_model_evidence": False,
                   "real_stability_pass_claim": False, "real_execution": "NOT_RUN",
                   "reasons": [reason], "rounds_requested": 3, "rounds_run": 0}
        if evidence is not None and not evidence.is_relative_to(ROOT):
            try:
                evidence.mkdir(parents=True, exist_ok=True)
                write_json(evidence / "stability.json", summary)
                write_json(evidence / "summary.json", summary)
                full_sha256sums(evidence)
            except OSError:
                pass
        return summary


def synthetic_response(case):
    task = case["task"]
    if task == "asr":
        return "Synthetic acknowledgement", case["reference"]["transcript"]
    if task == "gsm8k":
        answer = case["reference"]["answer"]
        if case["id"] == "gsm8k-test-0008":
            answer = 75
        elif case["id"] == "gsm8k-test-0012":
            answer = 12
        return f"Synthetic reasoning with a terminal marker.\n#### {answer}", ""
    if task == "vqa":
        from collections import Counter
        return Counter(case["reference"]["answers"]).most_common(1)[0][0], ""
    if task == "joint":
        words = list(case["reference"]["expected_keywords"])
        if case["id"] == "joint-speechcoco-val-0005":
            words.remove("room")
        return "Synthetic answer: " + " ".join(words), case["reference"]["caption"]
    raise AssertionError(task)


def synthetic_latency(case, index, scale=1.0, p95_spike=False):
    task = case["task"]
    if task == "asr":
        asr_final = 80.0 + index
        first = 350.0 + index
        total = 400.0 + index
    elif task == "gsm8k":
        asr_final = None
        first = 300.0 + index
        total = 1500.0 + 10 * index
    elif task == "vqa":
        asr_final = None
        first = 600.0 + index
        total = 650.0 + index
    else:
        asr_final = 50.0 + index
        first = 600.0 + index
        total = 1300.0 + 10 * index
    # p95 at these task sizes is the maximum. Spike only a stable per-task case.
    spike_case = {"asr": "asr-en-ls-1580-141084-0000" if case.get("language") == "en"
                  else "asr-zh-fleurs-16106752411261640702",
                  "gsm8k": "gsm8k-test-0015", "vqa": "vqa2-val-262197000",
                  "joint": "joint-speechcoco-val-0005"}[task]
    if p95_spike and case["id"] == spike_case:
        total *= 1.5
    total *= scale
    first *= scale
    if asr_final is not None:
        asr_final *= scale
    terminal = total - 0.1
    return {"total_ms": total, "first_delta_ms": first,
            "asr_final_ms": asr_final, "stream_ms": max(0.0, total - first),
            "terminal_ms": terminal}


def write_synthetic_round(round_dir, config, manifest, baseline_document=None,
                          scale=1.0, p95_spike=False, wrong_case=None,
                          failed_case=None, omit_case=None):
    """Write deterministic real-shaped synthetic evidence; never real evidence."""
    round_dir = Path(round_dir)
    rows = []
    failures = []
    for index, case in enumerate(manifest["cases"]):
        case_id = case["id"]
        if case_id == omit_case:
            continue
        answer, transcript = synthetic_response(case)
        if case_id == wrong_case:
            if case["task"] == "gsm8k":
                answer = f"Synthetic wrong response.\n#### {case['reference']['answer'] + 1}"
            elif case["task"] == "vqa":
                answer = "synthetic-no-reference-match"
            elif case["task"] == "asr":
                transcript = "synthetic wrong transcript"
                answer = transcript
            else:
                answer = "synthetic answer without declared keywords"
                transcript = "synthetic wrong caption"
        latency = synthetic_latency(case, index, scale, p95_spike)
        # Chunk boundaries preserve the exact answer bytes after normalization.
        answer = " ".join(answer.split())
        failed = case_id == failed_case
        events = []
        if failed:
            answer, transcript = "", ""
            events.append({"unified": {"kind": "error", "stage": "vlm"},
                           "error": {"code": 500, "message": "synthetic infrastructure failure"},
                           "choices": []})
            latency["first_delta_ms"] = None
            latency["asr_final_ms"] = None
            latency["terminal_ms"] = latency["total_ms"]
        else:
            if transcript and "audio" in case["modalities"]:
                events.append({"unified": {"kind": "asr_final", "text": transcript},
                               "choices": []})
            words = answer.split()
            chunks = [" ".join(words[pos:pos + 4]) + (" " if pos + 4 < len(words) else "")
                      for pos in range(0, len(words), 4)] or [""]
            for chunk in chunks:
                events.append({"choices": [{"delta": {"content": chunk},
                                            "finish_reason": None}]})
            events.append({"choices": [{"delta": {}, "finish_reason": "stop"}],
                           "unified": {"kind": "complete", "stage": "vlm"},
                           "usage": {"completion_tokens": len(words)}})
        payloads = [json.dumps(event, ensure_ascii=False) for event in events] + ["[DONE]"]
        write_jsonl(round_dir / "events" / f"{case_id}.jsonl",
                    [{"index": position, "data": payload}
                     for position, payload in enumerate(payloads)])
        (round_dir / "answers").mkdir(parents=True, exist_ok=True)
        (round_dir / "transcripts").mkdir(parents=True, exist_ok=True)
        if answer:
            (round_dir / "answers" / f"{case_id}.txt").write_text(answer, encoding="utf-8")
        if transcript:
            (round_dir / "transcripts" / f"{case_id}.txt").write_text(
                transcript, encoding="utf-8")
        row = {
            "case_id": case_id, "task": case["task"], "language": case.get("language"),
            "modalities": case["modalities"], "metric_name": case["metric"],
            "endpoint": "/v1/chat/completions", "http_status": 500 if failed else 200,
            "answer": answer, "transcript": transcript,
            "usage": None if failed else {"completion_tokens": len(answer.split())},
            "finish_reason": None if failed else "stop", "events": len(payloads),
            "done": True, "terminal_count": 1, "terminal_kind": "error" if failed else "complete",
            "terminal_stage": "vlm", "terminal_legs": None,
            "input_sha256": config["data"]["case_input_sha256"][case_id],
            "request": {"model": config["request"]["model"], "stream": True,
                        "temperature": 0, "top_p": None, "seed": None,
                        "max_tokens": (config["request"]["asr_vlm_max_tokens"]
                                       if case["task"] == "asr" else
                                       config["request"]["max_tokens"]),
                        "retry": False},
            "latency": latency, "output_tokens": len(answer.split()),
            "tokens_source": "usage", "tokens_per_s": None,
            "human_review_required": bool(case.get("review", {}).get("human_review_required")
                                          or case["task"] == "asr"),
        }
        row["metric"] = {"name": case["metric"], "value": None, "detail": {}} if failed \
            else audit.recompute_metric(case, row)
        if failed:
            failure = {"case_id": case_id, "category": "infrastructure",
                       "code": "stream_error_event", "severity": "P1",
                       "message": "synthetic infrastructure failure"}
            failures.append(failure)
        elif case["task"] == "gsm8k" and row["metric"]["value"] == 0.0:
            failures.append({"case_id": case_id, "category": "quality", "code": "gsm8k_wrong",
                             "severity": "P2", "message": "synthetic quality miss"})
        row["failures"] = [failure for failure in failures if failure["case_id"] == case_id]
        rows.append(row)
        if row["human_review_required"]:
            write_json(round_dir / "human_review" / f"{case_id}.json", {
                "case_id": case_id, "answer": answer, "transcript": transcript,
                "scored": row["metric"], "evidence_class": "NOT_REAL_EVIDENCE"})
    aggregates = audit.recompute_aggregate(rows)
    write_jsonl(round_dir / "cases.jsonl", rows)
    write_json(round_dir / "failures.json", {"counts": aggregates["failure_counts"],
                                              "failures": failures})
    write_json(round_dir / "report.json", {
        "schema_version": "umm-multimodal-benchmark-report/v1",
        "manifest_version": manifest["manifest_version"], "evidence_class": "contract-test",
        "real_model_evidence": False,
        "run": {"started_at": "1970-01-01T00:00:00+0000",
                "base_url": config["topology"]["base_url"],
                "model": config["request"]["model"], "asr_route": "chat",
                "round": 1, "rounds_requested": 1, "subset": False,
                "selected_cases": 70, "manifest_cases": 70, "excluded_case_ids": []},
        "config": {"max_tokens": config["request"]["max_tokens"],
                   "timeout_s": config["request"]["timeout_s"],
                   "case_latency_cap_ms": config["request"]["case_latency_cap_ms"],
                   "p95_tolerance": config["tolerances"]["p95_relative"],
                   "asr_vlm_max_tokens": config["request"]["asr_vlm_max_tokens"],
                   "asr_max_new_tokens": config["request"]["asr_max_new_tokens"],
                   "temperature": 0, "top_p": None, "seed": None, "retry": False},
        "cases": rows, "aggregates": aggregates, "failures": failures,
        "human_review_required": [row["case_id"] for row in rows
                                  if row["human_review_required"]],
        "evidence_class_notice": "NOT_REAL_EVIDENCE"})
    return rows


def make_synthetic_baseline(path, config, manifest, rows):
    case_results = []
    for row in rows:
        case_results.append({"case_id": row["case_id"], "source": "synthetic-replay",
                             "task": row["task"], "language": row["language"],
                             "metric": row["metric"], "latency": row["latency"],
                             "failure_codes": [item["code"] for item in row["failures"]],
                             "completed_terminal": True,
                             "answer_sha256": sha256_bytes(row["answer"].encode()),
                             "transcript_sha256": sha256_bytes(row["transcript"].encode())})
    details = {case_id: {"input_sha256": identity}
               for case_id, identity in config["data"]["case_input_sha256"].items()}
    document = {
        "schema_version": BASELINE_SCHEMA, "status": "PASS",
        "real_model_evidence": False, "evidence_class": "NOT_REAL_EVIDENCE",
        "manifest_version": manifest["manifest_version"],
        "manifest_sha256": config["manifest"]["sha256"],
        "declared_cases": 70, "real_cases": 0, "unique_real_cases": 0,
        "coverage_complete": True, "completed_terminals": 70,
        "missing_metric_case_ids": [], "stability_verdict": "BASELINE_REQUIRED",
        "stability_pass_claim": False,
        "stability_limit": "Synthetic replay reference; never accepted as real evidence.",
        "sources": [{"label": "synthetic-replay", "cases": 70,
                     "stability_verdict": "BASELINE_REQUIRED"}],
        "aggregates": audit.recompute_aggregate(rows), "case_results": case_results,
        "input_verification": {"cases": 70, "all_manifest_hashes_match": False,
                               "all_media_contracts_pass": False,
                               "cases_detail": details},
    }
    write_json(path, document)
    return document


def make_replay_workspace(root):
    """Create small fake artifacts solely to exercise hash/schedule/gate logic."""
    root = Path(root).resolve()
    require(not root.exists() or not any(root.iterdir()), "EVIDENCE_EXISTS",
            f"replay workspace is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    fixture = root / "synthetic-fixtures"
    binaries = {}
    for role in ("gateway", "asr", "vlm"):
        path = fixture / "bin" / role
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"NOT_REAL_EVIDENCE synthetic {role} binary\n".encode())
        binaries[role] = {"path": str(path), "sha256": sha256_file(path),
                          "source_commit": "0" * 40}
    models = {}
    for role in ("asr", "vlm"):
        model_root = fixture / "models" / role
        model_root.mkdir(parents=True, exist_ok=True)
        (model_root / "config.json").write_text(
            json.dumps({"synthetic": True, "role": role}) + "\n", encoding="utf-8")
        weight = model_root / "model.safetensors"
        weight.write_bytes(f"NOT_REAL_EVIDENCE {role} weights\n".encode())
        weight_manifest = fixture / f"{role}-weights-SHA256SUMS"
        weight_manifest.write_text(f"{sha256_file(weight)}  model.safetensors\n",
                                   encoding="utf-8")
        models[role] = {"id": f"synthetic-{role}", "root": str(model_root),
                        "config_files": {"config.json": sha256_file(model_root / "config.json")},
                        "weights_manifest": {"path": str(weight_manifest),
                                             "sha256": sha256_file(weight_manifest)}}
    raw = {
        "schema_version": CONFIG_SCHEMA, "run_id": "synthetic-replay-not-real-evidence",
        "evidence_dir": str(root / "evidence"),
        "source": {"root": str(ROOT), "commit": "0" * 40, "require_clean": False,
                   "harness_sha256": sha256_file(ROOT / "tests/multimodal_benchmark.py"),
                   "auditor_sha256": sha256_file(ROOT / "tools/multimodal_coverage_audit.py")},
        "baseline": {"path": str(fixture / "synthetic-baseline.json"), "sha256": "0" * 64},
        "manifest": {"path": "benchmarks/multimodal_manifest.json",
                     "sha256": sha256_file(ROOT / "benchmarks/multimodal_manifest.json")},
        "cache_dir": str(fixture / "unused-cache"), "binaries": binaries, "models": models,
        "request": {"model": "synthetic-unified", "temperature": 0.0, "top_p": None,
                    "seed": None, "max_tokens": 256, "asr_vlm_max_tokens": 1,
                    "asr_max_new_tokens": 256,
                    "stream": True, "asr_route": "chat", "timeout_s": 180.0,
                    "case_latency_cap_ms": 120000, "retry": False},
        "topology": {"base_url": "http://127.0.0.1:43999",
                     "ipc_namespace": str(fixture / "ipc"),
                     "gpu_by_stage": {"asr": 1, "vlm": 7},
                     "allowed_idle_gpus": [0, 1, 7], "resident_port_forbidden": 43301},
        "schedule": {"warmup_rounds": 1, "measured_rounds": 3,
                     "case_order": "manifest", "runner_timeout_s": 21600},
        "tolerances": {"quality": dict(DEFAULT_QUALITY_TOLERANCE),
                       "median_relative": 0.10, "p95_relative": 0.10,
                       "latency_tasks": list(TASKS)},
    }
    config_path = root / "replay-config.json"
    write_json(config_path, raw)
    config = normalize_config(raw, config_path, replay=True)
    manifest = load_manifest(config)
    reference_dir = fixture / "reference-round"
    rows = write_synthetic_round(reference_dir, config, manifest)
    make_synthetic_baseline(Path(config["baseline"]["path"]), config, manifest, rows)
    raw["baseline"]["sha256"] = sha256_file(config["baseline"]["path"])
    write_json(config_path, raw)
    config = normalize_config(raw, config_path, replay=True)
    manifest = load_manifest(config)
    baseline = load_baseline(config, manifest, replay=True)
    return config, manifest, baseline


def run_replay_self_check(output):
    verify_segmenter()
    config, manifest, baseline = make_replay_workspace(output)
    evidence = Path(config["evidence_dir"])
    evidence.mkdir(parents=True, exist_ok=True)
    write_synthetic_round(evidence / "warmup", config, manifest)
    for number in range(1, REQUIRED_ROUNDS + 1):
        write_synthetic_round(evidence / f"round-{number}", config, manifest)
    return finalize_evidence(config, manifest, baseline, replay=True)


def print_result(result):
    print(json.dumps({"verdict": result["verdict"],
                      "stability_verdict": result["stability_verdict"],
                      "replay_stability_verdict": result.get("replay_stability_verdict"),
                      "real_stability_pass_claim": result.get("real_stability_pass_claim", False),
                      "rounds_run": result.get("rounds_run", 0),
                      "consecutive_complete_matching_rounds":
                          result.get("consecutive_complete_matching_rounds", 0),
                      "reasons": result.get("reasons", []),
                      "evidence_dir": result.get("evidence_dir")}, indent=2, ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="declared stability run JSON")
    parser.add_argument("--preflight-only", action="store_true",
                        help="validate and emit STABILITY_PENDING without requests")
    parser.add_argument("--replay-self-check", action="store_true",
                        help="CPU-only synthetic replay; always NOT_REAL_EVIDENCE")
    parser.add_argument("--output", help="new empty directory for replay self-check")
    args = parser.parse_args(argv)
    if args.replay_self_check:
        if not args.output or args.config:
            parser.error("--replay-self-check requires --output and forbids --config")
        try:
            result = run_replay_self_check(args.output)
            result["evidence_dir"] = str(Path(args.output).resolve() / "evidence")
            print_result(result)
            return 0 if result.get("replay_stability_verdict") == "STABILITY_PASS" else 1
        except (StabilityError, audit.AuditError, data.DataError, OSError) as error:
            reason = error.reason if isinstance(error, StabilityError) else {
                "code": "REPLAY_ERROR", "message": str(error)}
            print_result({"verdict": "NOT_REAL_EVIDENCE",
                          "stability_verdict": "NOT_REAL_EVIDENCE",
                          "replay_stability_verdict": "FAIL", "reasons": [reason]})
            return 1
    if not args.config or args.output:
        parser.error("live mode requires --config and does not accept --output")
    try:
        result = run_live(args.config, preflight_only=args.preflight_only)
        result["evidence_dir"] = load_json(args.config).get("evidence_dir")
    except (StabilityError, OSError) as error:
        reason = error.reason if isinstance(error, StabilityError) else {
            "code": "ORCHESTRATION_ERROR", "message": str(error)}
        result = {"verdict": "FAIL", "stability_verdict": "FAIL",
                  "real_stability_pass_claim": False, "reasons": [reason]}
    print_result(result)
    return 0 if result["verdict"] == "STABILITY_PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
