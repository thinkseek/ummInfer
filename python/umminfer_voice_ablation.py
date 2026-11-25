#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage C Ablation #3: Voice Collaboration Strategy (full_response_tts vs sentence_streaming_tts).

Single-variable ablation over the public POST /v1/chat/completions audio contract:
  - full_response_tts:      audio.stream = false (synthesize full response WAV after VLM finish)
  - sentence_streaming_tts: audio.stream = true  (stream sentence-level PCM16 audio chunks)

Model, messages, sampling parameters, modalities (["text", "audio"]), voice, format,
and sample_rate stay strictly fixed across arms. `fixed_request_sha256` hashes every
request field except `audio.stream` and must be byte-identical across arms.

Pure Python 3 standard library only — zero imports from C++ serving code or ML frameworks.

Usage:
  python3 -I python/umminfer_voice_ablation.py \\
    --config benchmarks/ablations/voice-strategy.json \\
    --replay-only --out /tmp/umm-voice-ablation

  python3 -I python/umminfer_voice_ablation.py \\
    --config benchmarks/ablations/voice-strategy.json \\
    --base-url http://127.0.0.1:43301 --out /tmp/umm-voice-ablation-live

  python3 -I python/umminfer_voice_ablation.py \\
    --config benchmarks/ablations/voice-strategy-controlled-sm90.json \\
    --base-url http://127.0.0.1:43301 \\
    --controlled-live --provenance-json /path/to/provenance.json \\
    --out /tmp/umm-voice-ablation-controlled

Evidence honesty:
  - --replay-only produces CONTRACT_REPLAY (REAL_MODEL_EVIDENCE: NOT_RUN).
    Replay fixtures verify protocol, hashing, and metric calculation only; they
    are NEVER real latency or hardware performance evidence.
  - --base-url (without --controlled-live) produces UNCONTROLLED_LIVE observations
    against the target server.
  - --base-url --controlled-live (with a controlled_measurement config block and
    verified --provenance-json when require_exclusive_gpu=true) produces
    CONTROLLED_LIVE / COMPLETED only when every measured case succeeds with 0 failures.
"""

from __future__ import annotations

import argparse
import base64
import copy
from dataclasses import dataclass
import hashlib
import io
import ipaddress
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import wave


ABLATION_SCHEMA_VERSION = 1
DATASET_SCHEMA_VERSION = 1
REPLAY_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
CONTROLLED_MEASUREMENT_SCHEMA_VERSION = 1

VARIABLE_NAME = "voice_strategy"
VOICE_STRATEGIES = ("full_response_tts", "sentence_streaming_tts")
ALLOWED_SCHEDULE_MODES = ("interleaved_alternating",)
REQUIRED_PROVENANCE_BINARIES = ("gateway", "vlm", "tts")
REQUIRED_PROVENANCE_MODEL_MANIFESTS = ("vlm", "tts")
DEFAULT_STALL_THRESHOLD_MS = 500.0
TERMINAL_KINDS = {"complete", "error", "cancelled"}
HEX_256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_SHA_RE = re.compile(r"[0-9a-f]{7,40}\Z")
PROMPT_ID_RE = re.compile(r"[a-z0-9_-]+\Z")


@dataclass(frozen=True)
class VoiceExperiment:
    config_path: Path
    config_bytes: bytes
    config: dict
    dataset_bytes: bytes
    dataset: dict
    replay_bytes: bytes | None
    replay: dict | None

    @property
    def prompts(self) -> list[dict]:
        return self.dataset["prompts"]

    @property
    def config_sha256(self) -> str:
        return hashlib.sha256(self.config_bytes).hexdigest()

    @property
    def dataset_sha256(self) -> str:
        return hashlib.sha256(self.dataset_bytes).hexdigest()

    @property
    def replay_sha256(self) -> str | None:
        if self.replay_bytes is None:
            return None
        return hashlib.sha256(self.replay_bytes).hexdigest()

    @property
    def controlled_measurement(self) -> dict | None:
        cm = self.config.get("controlled_measurement")
        return copy.deepcopy(cm) if isinstance(cm, dict) else None


def canonical_json(value: object) -> bytes:
    """Deterministic UTF-8 canonical JSON encoding (sorted keys, compact separators)."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_json(value: object) -> str:
    return sha256_bytes(canonical_json(value))


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_sha256(raw: bytes, expected: object, label: str) -> str:
    if not isinstance(expected, str) or not HEX_256_RE.fullmatch(expected):
        raise ValueError(f"{label} sha256 must be 64 lowercase hex characters")
    actual = sha256_bytes(raw)
    if actual != expected:
        raise ValueError(f"{label} sha256 mismatch: expected {expected}, got {actual}")
    return actual


def _read_json_object(path: Path) -> tuple[bytes, dict]:
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")
    return raw, data


def _resolve_relative(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else base_dir / path


def validate_base_url(url: str, *, allow_remote_http: bool = False) -> str:
    """Normalize and validate an HTTP/HTTPS server base URL."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("base-url must be a non-empty string")
    parsed = urllib.parse.urlsplit(url.strip())
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"unsupported URL scheme {scheme!r}; expected http or https")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("base-url must not contain embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("base-url must not contain query parameters or fragments")
    host = parsed.hostname
    if not host:
        raise ValueError("base-url must include a hostname")

    is_loopback = host.lower() == "localhost"
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            is_loopback = False

    if scheme == "http" and not is_loopback and not allow_remote_http:
        raise ValueError(
            "refusing non-loopback plain HTTP URL; use https:// or pass --allow-remote-http"
        )
    return f"{scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"


def validate_controlled_measurement(block: object) -> dict:
    """Validate optional controlled_measurement schema v1 block (fail closed)."""
    if not isinstance(block, dict):
        raise ValueError("config.controlled_measurement must be an object")
    schema_version = block.get("schema_version")
    if not _is_int(schema_version) or schema_version != CONTROLLED_MEASUREMENT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported controlled_measurement.schema_version: {schema_version!r} "
            f"(expected {CONTROLLED_MEASUREMENT_SCHEMA_VERSION})"
        )
    warmup_runs = block.get("warmup_runs")
    if not _is_int(warmup_runs) or warmup_runs < 0:
        raise ValueError("controlled_measurement.warmup_runs must be an integer >= 0")
    repetitions = block.get("repetitions")
    if not _is_int(repetitions) or repetitions < 1:
        raise ValueError("controlled_measurement.repetitions must be an integer >= 1")
    schedule_mode = block.get("schedule_mode")
    if not isinstance(schedule_mode, str) or schedule_mode not in ALLOWED_SCHEDULE_MODES:
        raise ValueError(
            f"unsupported controlled_measurement.schedule_mode: {schedule_mode!r}; "
            f"expected one of {list(ALLOWED_SCHEDULE_MODES)}"
        )
    require_exclusive_gpu = block.get("require_exclusive_gpu")
    if not isinstance(require_exclusive_gpu, bool):
        raise ValueError("controlled_measurement.require_exclusive_gpu must be a boolean")
    max_failed_cases = block.get("max_failed_cases")
    if not _is_int(max_failed_cases) or max_failed_cases != 0:
        raise ValueError("controlled_measurement.max_failed_cases must be 0")

    return {
        "schema_version": schema_version,
        "warmup_runs": warmup_runs,
        "repetitions": repetitions,
        "schedule_mode": schedule_mode,
        "require_exclusive_gpu": require_exclusive_gpu,
        "max_failed_cases": max_failed_cases,
    }


def build_arm_schedule(
    repetitions: int,
    schedule_mode: str = "interleaved_alternating",
) -> list[list[str]]:
    """Return deterministic per-repetition arm orderings to counter thermal/cache drift."""
    if not _is_int(repetitions) or repetitions < 0:
        raise ValueError("repetitions must be an integer >= 0")
    if schedule_mode not in ALLOWED_SCHEDULE_MODES:
        raise ValueError(
            f"unsupported schedule_mode {schedule_mode!r}; expected one of {list(ALLOWED_SCHEDULE_MODES)}"
        )
    schedule: list[list[str]] = []
    forward = list(VOICE_STRATEGIES)
    reverse = list(reversed(VOICE_STRATEGIES))
    for rep in range(repetitions):
        schedule.append(list(forward if rep % 2 == 0 else reverse))
    return schedule


def load_and_validate_provenance(
    provenance_path: Path | str,
    *,
    require_exclusive_gpu: bool,
    strict: bool = True,
    expected_vlm_weights_sha256: str | None = None,
) -> tuple[str, dict]:
    """Load and fail-closed validate an external controlled-measurement provenance JSON."""
    resolved = Path(provenance_path).resolve()
    raw, data = _read_json_object(resolved)
    digest = sha256_bytes(raw)
    if not strict:
        return digest, data

    commit_sha = data.get("commit_sha")
    if not isinstance(commit_sha, str) or not COMMIT_SHA_RE.fullmatch(commit_sha.strip()):
        raise ValueError("provenance.commit_sha must be 7-40 lowercase hex characters")

    host_profile = data.get("host_profile")
    if not isinstance(host_profile, dict) or not host_profile:
        raise ValueError("provenance.host_profile must be a non-empty object")
    for k, v in host_profile.items():
        if not isinstance(k, str) or not k.strip():
            raise ValueError("provenance.host_profile keys must be non-empty strings")
        if v is None or (isinstance(v, str) and not v.strip()):
            raise ValueError(f"provenance.host_profile.{k} must not be null or empty")

    binary_sha256 = data.get("binary_sha256")
    if not isinstance(binary_sha256, dict) or not binary_sha256:
        raise ValueError("provenance.binary_sha256 must be a non-empty object")
    for req_bin in REQUIRED_PROVENANCE_BINARIES:
        if req_bin not in binary_sha256:
            raise ValueError(f"provenance.binary_sha256 missing required binary: {req_bin!r}")
    for bin_name, bin_hash in binary_sha256.items():
        if not isinstance(bin_name, str) or not bin_name.strip():
            raise ValueError("provenance.binary_sha256 keys must be non-empty strings")
        if not isinstance(bin_hash, str) or not HEX_256_RE.fullmatch(bin_hash):
            raise ValueError(f"provenance.binary_sha256[{bin_name!r}] must be 64 lowercase hex characters")

    model_manifests = data.get("model_manifests")
    if not isinstance(model_manifests, dict) or not model_manifests:
        raise ValueError("provenance.model_manifests must be a non-empty object")
    for req_model in REQUIRED_PROVENANCE_MODEL_MANIFESTS:
        if req_model not in model_manifests:
            raise ValueError(f"provenance.model_manifests missing required entry: {req_model!r}")
    for model_key, manifest in model_manifests.items():
        if not isinstance(model_key, str) or not model_key.strip():
            raise ValueError("provenance.model_manifests keys must be non-empty strings")
        if not isinstance(manifest, dict):
            raise ValueError(f"provenance.model_manifests[{model_key!r}] must be an object")
        w_sha = manifest.get("weights_sha256")
        if not isinstance(w_sha, str) or not HEX_256_RE.fullmatch(w_sha):
            raise ValueError(
                f"provenance.model_manifests[{model_key!r}].weights_sha256 must be 64 lowercase hex characters"
            )
        count = manifest.get("file_count")
        if count is None:
            count = manifest.get("shard_count")
        if count is None:
            count = manifest.get("weight_file_count")
        if not _is_int(count) or count < 1:
            raise ValueError(
                f"provenance.model_manifests[{model_key!r}] must include positive integer file_count/shard_count"
            )
    if expected_vlm_weights_sha256 is not None:
        actual_vlm_sha = model_manifests["vlm"]["weights_sha256"]
        if actual_vlm_sha != expected_vlm_weights_sha256:
            raise ValueError(
                f"provenance.model_manifests.vlm.weights_sha256 mismatch: "
                f"config expects {expected_vlm_weights_sha256}, provenance has {actual_vlm_sha}"
            )

    gpus = data.get("gpus")
    if not isinstance(gpus, list) or not gpus:
        raise ValueError("provenance.gpus must be a non-empty list")
    seen_gpu_indices: set[int] = set()
    for idx, gpu in enumerate(gpus):
        if not isinstance(gpu, dict):
            raise ValueError(f"provenance.gpus[{idx}] must be an object")
        gpu_index = gpu.get("index")
        if not _is_int(gpu_index) or gpu_index < 0:
            raise ValueError(f"provenance.gpus[{idx}].index must be a non-negative integer")
        if gpu_index in seen_gpu_indices:
            raise ValueError(f"duplicate GPU index in provenance.gpus: {gpu_index}")
        seen_gpu_indices.add(gpu_index)
        gpu_name = gpu.get("name")
        if not isinstance(gpu_name, str) or not gpu_name.strip():
            raise ValueError(f"provenance.gpus[{idx}].name must be a non-empty string")

        if "exclusive" in gpu:
            if not isinstance(gpu["exclusive"], bool):
                raise ValueError(f"provenance.gpus[{idx}].exclusive must be a boolean")
            if require_exclusive_gpu and gpu["exclusive"] is not True:
                raise ValueError(f"provenance.gpus[{idx}] (GPU {gpu_index}) is not exclusive (exclusive=false)")

        for check_key in ("pre_check", "post_check"):
            check = gpu.get(check_key)
            if not isinstance(check, dict):
                raise ValueError(f"provenance.gpus[{idx}].{check_key} must be an object")
            excl = check.get("exclusive")
            if not isinstance(excl, bool):
                raise ValueError(f"provenance.gpus[{idx}].{check_key}.exclusive must be a boolean")
            if require_exclusive_gpu and excl is not True:
                raise ValueError(
                    f"provenance.gpus[{idx}] (GPU {gpu_index}) failed {check_key} exclusive isolation check"
                )
            if "other_compute_processes" in check:
                other_procs = check["other_compute_processes"]
                if not _is_int(other_procs) or other_procs < 0:
                    raise ValueError(
                        f"provenance.gpus[{idx}].{check_key}.other_compute_processes must be an integer >= 0"
                    )
                if require_exclusive_gpu and other_procs != 0:
                    raise ValueError(
                        f"provenance.gpus[{idx}] (GPU {gpu_index}) reported {other_procs} "
                        f"other compute processes during {check_key}"
                    )

    return digest, data


def load_experiment(config_path: Path | str) -> VoiceExperiment:
    """Load and cryptographically verify config, dataset, and optional replay bundle."""
    resolved = Path(config_path).resolve()
    config_bytes, config = _read_json_object(resolved)

    if config.get("ablation_schema_version") != ABLATION_SCHEMA_VERSION:
        raise ValueError(f"unsupported ablation_schema_version: {config.get('ablation_schema_version')}")
    if not isinstance(config.get("ablation_id"), str) or not config["ablation_id"].strip():
        raise ValueError("ablation_id must be a non-empty string")
    if config.get("variable") != VARIABLE_NAME:
        raise ValueError(f"config variable must be {VARIABLE_NAME!r}, got {config.get('variable')!r}")
    if config.get("voice_strategies") != list(VOICE_STRATEGIES):
        raise ValueError(
            f"voice_strategies must be exactly {list(VOICE_STRATEGIES)}, got {config.get('voice_strategies')!r}"
        )

    stall_threshold_ms = config.get("stall_threshold_ms", DEFAULT_STALL_THRESHOLD_MS)
    if not _is_number(stall_threshold_ms) or float(stall_threshold_ms) <= 0.0:
        raise ValueError("stall_threshold_ms must be a finite positive number")

    model = config.get("model")
    if (
        not isinstance(model, dict)
        or not isinstance(model.get("id"), str)
        or not model["id"].strip()
        or not isinstance(model.get("revision"), str)
        or not model["revision"].strip()
    ):
        raise ValueError("model must specify non-empty string id and revision")
    weights_sha = model.get("weights_sha256")
    if weights_sha is not None and (not isinstance(weights_sha, str) or not HEX_256_RE.fullmatch(weights_sha)):
        raise ValueError("model.weights_sha256 must be null or 64 lowercase hex chars")

    fixed = config.get("fixed")
    if not isinstance(fixed, dict):
        raise ValueError("config.fixed must be an object")
    modalities = fixed.get("modalities")
    if modalities != ["text", "audio"]:
        raise ValueError("fixed.modalities must be ['text', 'audio']")
    audio = fixed.get("audio")
    if not isinstance(audio, dict):
        raise ValueError("fixed.audio must be an object")
    if "stream" in audio:
        raise ValueError("fixed.audio must not pin 'stream'; audio.stream is the single ablated variable")
    if not isinstance(audio.get("voice"), str) or not audio["voice"].strip():
        raise ValueError("fixed.audio.voice must be a non-empty string")
    if audio.get("format") != "wav":
        raise ValueError("fixed.audio.format must be 'wav' per public /v1/chat/completions contract")
    sample_rate = audio.get("sample_rate", 24000)
    if not _is_int(sample_rate) or sample_rate != 24000:
        raise ValueError("fixed.audio.sample_rate must be 24000")

    sampling = fixed.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("fixed.sampling must be an object")
    forbidden_sampling = {"model", "messages", "stream", "modalities", "audio", "image", "video", "narrative"}
    overlap = forbidden_sampling & set(sampling)
    if overlap:
        raise ValueError(f"fixed.sampling contains non-sampling keys: {sorted(overlap)}")

    dataset_ref = config.get("dataset")
    if not isinstance(dataset_ref, dict) or not isinstance(dataset_ref.get("path"), str):
        raise ValueError("config.dataset must specify path and sha256")
    dataset_path = _resolve_relative(resolved.parent, dataset_ref["path"])
    dataset_bytes, dataset = _read_json_object(dataset_path)
    _check_sha256(dataset_bytes, dataset_ref.get("sha256"), "dataset")
    if dataset.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError(f"unsupported dataset schema_version: {dataset.get('schema_version')}")
    prompts = dataset.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("dataset.prompts must be a non-empty list")
    seen_ids: set[str] = set()
    for prompt in prompts:
        if not isinstance(prompt, dict):
            raise ValueError("each dataset prompt must be an object")
        pid = prompt.get("id")
        if not isinstance(pid, str) or not PROMPT_ID_RE.fullmatch(pid):
            raise ValueError(f"invalid prompt id: {pid!r}")
        if pid in seen_ids:
            raise ValueError(f"duplicate prompt id: {pid!r}")
        seen_ids.add(pid)
        messages = prompt.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"prompt {pid!r} lacks non-empty messages")

    replay_bytes: bytes | None = None
    replay: dict | None = None
    if "replay" in config and config["replay"] is not None:
        replay_ref = config["replay"]
        if not isinstance(replay_ref, dict) or not isinstance(replay_ref.get("path"), str):
            raise ValueError("config.replay must specify path and sha256")
        replay_path = _resolve_relative(resolved.parent, replay_ref["path"])
        replay_bytes, replay = _read_json_object(replay_path)
        _check_sha256(replay_bytes, replay_ref.get("sha256"), "replay")
        if replay.get("schema_version") != REPLAY_SCHEMA_VERSION:
            raise ValueError(f"unsupported replay schema_version: {replay.get('schema_version')}")
        if not isinstance(replay.get("cases"), list):
            raise ValueError("replay.cases must be a list")

    if "controlled_measurement" in config and config["controlled_measurement"] is not None:
        validate_controlled_measurement(config["controlled_measurement"])

    return VoiceExperiment(
        config_path=resolved,
        config_bytes=config_bytes,
        config=config,
        dataset_bytes=dataset_bytes,
        dataset=dataset,
        replay_bytes=replay_bytes,
        replay=replay,
    )


def build_request(experiment: VoiceExperiment, prompt: dict, strategy: str) -> dict:
    """Build a public POST /v1/chat/completions request toggling ONLY audio.stream."""
    if strategy not in VOICE_STRATEGIES:
        raise ValueError(f"unsupported voice strategy: {strategy!r}")
    fixed = experiment.config["fixed"]
    audio_opts = copy.deepcopy(fixed["audio"])
    audio_opts["stream"] = strategy == "sentence_streaming_tts"

    body: dict = {
        "model": experiment.config["model"]["id"],
        "stream": True,
        "messages": copy.deepcopy(prompt["messages"]),
        "modalities": list(fixed["modalities"]),
        "audio": audio_opts,
    }
    body.update(copy.deepcopy(fixed["sampling"]))
    return body


def fixed_request_projection(request_body: dict) -> dict:
    """Return request payload with ONLY the ablated variable (audio.stream) removed."""
    projected = copy.deepcopy(request_body)
    audio = projected.get("audio")
    if not isinstance(audio, dict) or not isinstance(audio.get("stream"), bool):
        raise ValueError("request must contain boolean audio.stream for single-variable projection")
    del audio["stream"]
    return projected


def fixed_request_sha256(request_body: dict) -> str:
    """Hash every field of the request EXCEPT audio.stream."""
    return sha256_json(fixed_request_projection(request_body))


def verify_single_variable_matrix(experiment: VoiceExperiment) -> dict:
    """Build and verify requests across both arms; fail closed if anything besides audio.stream varies."""
    by_arm: dict[str, list[dict]] = {}
    fixed_hashes_by_prompt: dict[str, dict[str, str]] = {}
    full_hashes_by_prompt: dict[str, dict[str, str]] = {}

    for strategy in VOICE_STRATEGIES:
        arm_items = []
        for prompt in experiment.prompts:
            pid = prompt["id"]
            req = build_request(experiment, prompt, strategy)
            f_hash = fixed_request_sha256(req)
            r_hash = sha256_json(req)
            fixed_hashes_by_prompt.setdefault(pid, {})[strategy] = f_hash
            full_hashes_by_prompt.setdefault(pid, {})[strategy] = r_hash
            arm_items.append({
                "prompt_id": pid,
                "request": req,
                "fixed_request_sha256": f_hash,
                "request_sha256": r_hash,
                "audio_stream": req["audio"]["stream"],
            })
        by_arm[strategy] = arm_items

    prompt_fixed_hashes: dict[str, str] = {}
    for prompt in experiment.prompts:
        pid = prompt["id"]
        f_set = set(fixed_hashes_by_prompt[pid].values())
        if len(f_set) != 1:
            raise ValueError(
                f"single-variable violation on prompt {pid!r}: fixed_request_sha256 disagreed across arms: "
                f"{fixed_hashes_by_prompt[pid]}"
            )
        r_set = set(full_hashes_by_prompt[pid].values())
        if len(r_set) != len(VOICE_STRATEGIES):
            raise ValueError(
                f"single-variable violation on prompt {pid!r}: full request_sha256 did not differ across arms"
            )
        prompt_fixed_hashes[pid] = next(iter(f_set))

    arm_matrix_hashes = {
        strategy: sha256_json([item["fixed_request_sha256"] for item in items])
        for strategy, items in by_arm.items()
    }
    if len(set(arm_matrix_hashes.values())) != 1:
        raise ValueError(f"fixed_matrix_sha256 disagreed across arms: {arm_matrix_hashes}")

    return {
        "by_arm": by_arm,
        "fixed_request_sha256_by_prompt": prompt_fixed_hashes,
        "fixed_matrix_sha256": next(iter(arm_matrix_hashes.values())),
        "single_variable_verified": True,
    }


def _extract_delta_text(frame: dict, unified: dict | None) -> str:
    if isinstance(unified, dict) and unified.get("kind") == "vlm_delta":
        text = unified.get("text")
        if isinstance(text, str):
            return text
    choices = frame.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        delta = choices[0].get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            return delta["content"]
    return ""


def parse_sse_bytes(raw_sse: bytes) -> tuple[list[dict], list[str]]:
    """Parse SSE text/event-stream bytes into ordered JSON frames and raw data payloads."""
    text = raw_sse.decode("utf-8")
    frames: list[dict] = []
    payloads: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r\n")
        if not line.startswith("data:"):
            continue
        payload = line[5:].lstrip(" ")
        payloads.append(payload)
        if payload == "[DONE]":
            continue
        frame = json.loads(payload)
        if not isinstance(frame, dict):
            raise ValueError("SSE frame payload must be a JSON object")
        if "error" in frame and isinstance(frame["error"], dict):
            raise ValueError(f"SSE stream returned error frame: {frame['error']}")
        frames.append(frame)
    return frames, payloads


def percentile(values: list[float], pct: float) -> float | None:
    """Compute linear-interpolated percentile (0..100) deterministically."""
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return round(ordered[0], 6)
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return round(ordered[lo], 6)
    frac = rank - lo
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * frac, 6)


def compute_iqr(values: list[float]) -> float | None:
    """Compute interquartile range (Q3 - Q1) using linear-interpolated percentiles."""
    if not values:
        return None
    q1 = percentile(values, 25.0)
    q3 = percentile(values, 75.0)
    assert q1 is not None and q3 is not None
    return round(q3 - q1, 6)


def compute_mad(values: list[float]) -> float | None:
    """Compute median absolute deviation: median(|x_i - median(x)|)."""
    if not values:
        return None
    ordered = [float(v) for v in values]
    med = float(statistics.median(ordered))
    deviations = [abs(v - med) for v in ordered]
    return round(float(statistics.median(deviations)), 6)


def summarize_metric(
    values: list[float | int | None],
    *,
    not_applicable_when_empty: bool = False,
    include_dispersion: bool = False,
) -> dict:
    """Aggregate median, p95, min, max, optional iqr/mad, and explicit coverage over metric samples."""
    total = len(values)
    present = [float(v) for v in values if v is not None]
    sample_count = len(present)
    if total == 0:
        coverage = "NOT_COVERED"
    elif sample_count == 0:
        coverage = "NOT_APPLICABLE" if not_applicable_when_empty else "NOT_COVERED"
    elif sample_count == total:
        coverage = "COVERED"
    else:
        coverage = "PARTIAL"

    if not present:
        result: dict = {
            "coverage": coverage,
            "sample_count": 0,
            "total_count": total,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }
        if include_dispersion:
            result["iqr"] = None
            result["mad"] = None
        return result
    result = {
        "coverage": coverage,
        "sample_count": sample_count,
        "total_count": total,
        "median": round(float(statistics.median(present)), 6),
        "p95": percentile(present, 95.0),
        "min": round(min(present), 6),
        "max": round(max(present), 6),
    }
    if include_dispersion:
        result["iqr"] = compute_iqr(present)
        result["mad"] = compute_mad(present)
    return result


def evaluate_sse_case(
    *,
    raw_sse: bytes,
    strategy: str,
    prompt_id: str,
    request_sha256: str,
    fixed_req_sha256: str,
    stall_threshold_ms: float,
    recorded_latency_us: dict | None = None,
    live_frame_times_ms: list[float] | None = None,
) -> dict:
    """Parse, cryptographically verify audio artifacts, and compute voice ablation metrics."""
    frames, payloads = parse_sse_bytes(raw_sse)
    if not frames:
        raise ValueError(f"{strategy}/{prompt_id}: SSE stream contained no JSON frames")
    if payloads.count("[DONE]") != 1 or payloads[-1] != "[DONE]":
        raise ValueError(f"{strategy}/{prompt_id}: SSE stream must terminate with exactly one [DONE]")
    if live_frame_times_ms is not None and len(live_frame_times_ms) != len(frames):
        raise ValueError("live_frame_times_ms length must match parsed SSE frame count")

    unified_events: list[tuple[int, dict, dict]] = []
    for idx, frame in enumerate(frames):
        unified = frame.get("unified")
        if isinstance(unified, dict):
            unified_events.append((idx, unified, frame))

    if not unified_events:
        raise ValueError(f"{strategy}/{prompt_id}: SSE stream contained no unified events")

    seqs = [ev.get("seq") for _, ev, _ in unified_events]
    if seqs != list(range(len(seqs))):
        raise ValueError(f"{strategy}/{prompt_id}: unified SSE seq not strictly monotonic from 0: {seqs}")

    terminals = [(idx, ev) for idx, ev, _ in unified_events if ev.get("kind") in TERMINAL_KINDS]
    if len(terminals) != 1 or terminals[0][0] != unified_events[-1][0]:
        raise ValueError(f"{strategy}/{prompt_id}: stream must end with exactly one terminal event")
    terminal_idx, terminal_ev = terminals[0]
    terminal_state = terminal_ev.get("kind")
    if terminal_state != "complete":
        raise ValueError(f"{strategy}/{prompt_id}: stream terminated with kind={terminal_state!r}")

    # Collect text deltas and verify audio artifacts.
    first_text_frame_idx: int | None = None
    prose_parts: list[str] = []
    audio_groups: dict[str, dict] = {}
    audio_chunk_frame_indices: list[int] = []
    all_decoded_audio_chunks: list[bytes] = []

    for idx, frame in enumerate(frames):
        unified = frame.get("unified") if isinstance(frame.get("unified"), dict) else None
        delta_text = _extract_delta_text(frame, unified)
        if delta_text:
            prose_parts.append(delta_text)
            if first_text_frame_idx is None:
                first_text_frame_idx = idx

        if not unified or unified.get("kind") != "artifact":
            continue
        artifact = unified.get("artifact")
        if not isinstance(artifact, dict):
            raise ValueError(f"{strategy}/{prompt_id}: artifact event missing artifact object")
        if artifact.get("media_type") != "audio" and unified.get("stage") != "tts":
            raise ValueError(
                f"{strategy}/{prompt_id}: unexpected non-audio artifact in voice ablation: "
                f"stage={unified.get('stage')!r} media_type={artifact.get('media_type')!r}"
            )
        artifact_id = artifact.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ValueError(f"{strategy}/{prompt_id}: audio artifact lacks artifact_id")
        if artifact.get("encoding") != "base64":
            raise ValueError(f"{strategy}/{prompt_id}: audio artifact {artifact_id} encoding is not base64")
        chunk_idx = artifact.get("index")
        chunk_total = artifact.get("total")
        if not (_is_int(chunk_idx) and _is_int(chunk_total) and chunk_total > 0 and 0 <= chunk_idx < chunk_total):
            raise ValueError(f"{strategy}/{prompt_id}: invalid chunk bounds on {artifact_id}")

        try:
            raw_chunk = base64.b64decode(artifact["data"], validate=True)
        except Exception as exc:
            raise ValueError(f"{strategy}/{prompt_id}: invalid base64 in {artifact_id}: {exc}") from exc

        group = audio_groups.setdefault(
            artifact_id,
            {
                "chunks": {},
                "next_index": 0,
                "total": chunk_total,
                "descriptors": [],
                "frame_indices": [],
            },
        )
        if chunk_total != group["total"]:
            raise ValueError(f"{strategy}/{prompt_id}: total changed between chunks for {artifact_id}")
        if chunk_idx != group["next_index"]:
            raise ValueError(f"{strategy}/{prompt_id}: out-of-order chunk {chunk_idx} for {artifact_id}")
        group["next_index"] += 1
        group["chunks"][chunk_idx] = raw_chunk
        group["descriptors"].append(artifact)
        group["frame_indices"].append(idx)
        audio_chunk_frame_indices.append(idx)
        all_decoded_audio_chunks.append(raw_chunk)

        is_final_chunk = chunk_idx + 1 == chunk_total
        if ("bytes" in artifact) != is_final_chunk or ("sha256" in artifact) != is_final_chunk:
            raise ValueError(f"{strategy}/{prompt_id}: bytes/sha256 must appear only on final chunk of {artifact_id}")

    if first_text_frame_idx is None:
        raise ValueError(f"{strategy}/{prompt_id}: stream contained no non-empty text deltas")
    if not audio_chunk_frame_indices:
        raise ValueError(f"{strategy}/{prompt_id}: stream contained no audio chunks")

    prose = "".join(prose_parts)
    if terminal_ev.get("text") != prose:
        raise ValueError(f"{strategy}/{prompt_id}: terminal complete text differs from aggregated deltas")

    verified_artifacts: list[dict] = []
    audio_bytes_total = 0
    for artifact_id, group in audio_groups.items():
        total = group["total"]
        if sorted(group["chunks"]) != list(range(total)):
            raise ValueError(f"{strategy}/{prompt_id}: missing chunks in {artifact_id}")
        raw_audio = b"".join(group["chunks"][i] for i in range(total))
        final_desc = group["descriptors"][-1]
        if not _is_int(final_desc.get("bytes")) or final_desc["bytes"] != len(raw_audio):
            raise ValueError(
                f"{strategy}/{prompt_id}: audio artifact {artifact_id} byte count mismatch "
                f"(expected {final_desc.get('bytes')}, actual {len(raw_audio)})"
            )
        actual_sha = sha256_bytes(raw_audio)
        if final_desc.get("sha256") != actual_sha:
            raise ValueError(
                f"{strategy}/{prompt_id}: audio artifact {artifact_id} sha256 mismatch "
                f"(expected {final_desc.get('sha256')}, actual {actual_sha})"
            )
        mime_type = final_desc.get("mime_type")
        fmt = final_desc.get("format")
        if mime_type == "audio/wav" or fmt == "wav":
            if len(raw_audio) < 44 or raw_audio[:4] != b"RIFF" or raw_audio[8:12] != b"WAVE":
                raise ValueError(f"{strategy}/{prompt_id}: WAV artifact {artifact_id} lacks RIFF/WAVE header")
            with wave.open(io.BytesIO(raw_audio), "rb") as wf:
                if wf.getnframes() <= 0 or wf.getsampwidth() != 2:
                    raise ValueError(f"{strategy}/{prompt_id}: invalid PCM16 WAV payload in {artifact_id}")
        elif mime_type == "audio/pcm" or fmt == "pcm16":
            if len(raw_audio) == 0 or len(raw_audio) % 2 != 0:
                raise ValueError(f"{strategy}/{prompt_id}: PCM16 artifact {artifact_id} has invalid length")
        else:
            raise ValueError(f"{strategy}/{prompt_id}: unsupported audio mime/format: {mime_type!r}/{fmt!r}")

        audio_bytes_total += len(raw_audio)
        verified_artifacts.append({
            "artifact_id": artifact_id,
            "mime_type": mime_type,
            "format": fmt,
            "bytes": len(raw_audio),
            "sha256": actual_sha,
            "chunks": total,
            "stream_index": final_desc.get("stream_index"),
        })

    # Verify legs include completed tts leg.
    legs = terminal_ev.get("legs")
    if not isinstance(legs, list) or not any(
        isinstance(leg, dict) and leg.get("stage") == "tts" and leg.get("state") == "completed"
        for leg in legs
    ):
        raise ValueError(f"{strategy}/{prompt_id}: terminal complete event lacks completed tts leg")

    # Resolve timings from either recorded replay observations or live wall-clock capture.
    if recorded_latency_us is not None:
        if not isinstance(recorded_latency_us, dict):
            raise ValueError(f"{strategy}/{prompt_id}: recorded latency_us must be an object")
        for key in ("ttft_us", "ttfa_us", "terminal_us"):
            if not _is_int(recorded_latency_us.get(key)) or recorded_latency_us[key] < 0:
                raise ValueError(f"{strategy}/{prompt_id}: latency_us.{key} must be a non-negative integer")
        chunk_us = recorded_latency_us.get("audio_chunk_us")
        if not isinstance(chunk_us, list) or len(chunk_us) != len(audio_chunk_frame_indices):
            raise ValueError(
                f"{strategy}/{prompt_id}: latency_us.audio_chunk_us count ({len(chunk_us) if isinstance(chunk_us, list) else None}) "
                f"must match audio chunk count ({len(audio_chunk_frame_indices)})"
            )
        if any(not _is_int(t) or t < 0 for t in chunk_us):
            raise ValueError(f"{strategy}/{prompt_id}: latency_us.audio_chunk_us entries must be non-negative ints")
        if chunk_us != sorted(chunk_us):
            raise ValueError(f"{strategy}/{prompt_id}: latency_us.audio_chunk_us must be non-decreasing")
        if chunk_us[0] != recorded_latency_us["ttfa_us"]:
            raise ValueError(f"{strategy}/{prompt_id}: latency_us.ttfa_us must equal audio_chunk_us[0]")
        if recorded_latency_us["terminal_us"] < chunk_us[-1] or recorded_latency_us["terminal_us"] < recorded_latency_us["ttft_us"]:
            raise ValueError(f"{strategy}/{prompt_id}: latency_us.terminal_us predates ttft/audio chunks")

        ttft_ms = round(recorded_latency_us["ttft_us"] / 1000.0, 6)
        audio_chunk_times_ms = [round(t / 1000.0, 6) for t in chunk_us]
        ttfa_ms = audio_chunk_times_ms[0]
        terminal_ms = round(recorded_latency_us["terminal_us"] / 1000.0, 6)
        timing_source = "recorded_contract_replay"
    elif live_frame_times_ms is not None:
        ttft_ms = round(live_frame_times_ms[first_text_frame_idx], 6)
        audio_chunk_times_ms = [round(live_frame_times_ms[i], 6) for i in audio_chunk_frame_indices]
        ttfa_ms = audio_chunk_times_ms[0]
        terminal_ms = round(live_frame_times_ms[terminal_idx], 6)
        timing_source = "live_monotonic_clock"
    else:
        raise ValueError("either recorded_latency_us or live_frame_times_ms must be provided")

    ttfa_after_ttft_ms = round(ttfa_ms - ttft_ms, 6)
    inter_chunk_gaps_ms = [
        round(audio_chunk_times_ms[i] - audio_chunk_times_ms[i - 1], 6)
        for i in range(1, len(audio_chunk_times_ms))
    ]
    inter_chunk_gap_p95_ms = percentile(inter_chunk_gaps_ms, 95.0) if inter_chunk_gaps_ms else None
    stall_count = sum(1 for gap in inter_chunk_gaps_ms if gap > stall_threshold_ms)
    stall_rate = round(stall_count / len(inter_chunk_gaps_ms), 6) if inter_chunk_gaps_ms else None

    return {
        "prompt_id": prompt_id,
        "voice_strategy": strategy,
        "audio_stream": strategy == "sentence_streaming_tts",
        "request_sha256": request_sha256,
        "fixed_request_sha256": fixed_req_sha256,
        "response_sha256": sha256_bytes(raw_sse),
        "terminal_state": terminal_state,
        "timing_source": timing_source,
        "prose_chars": len(prose),
        "ttft_ms": ttft_ms,
        "ttfa_ms": ttfa_ms,
        "ttfa_after_ttft_ms": ttfa_after_ttft_ms,
        "terminal_ms": terminal_ms,
        "audio_chunk_count": len(audio_chunk_times_ms),
        "audio_bytes_total": audio_bytes_total,
        "audio_stream_sha256": sha256_bytes(b"".join(all_decoded_audio_chunks)),
        "audio_chunk_times_ms": audio_chunk_times_ms,
        "inter_chunk_gaps_ms": inter_chunk_gaps_ms,
        "inter_chunk_gap_p95_ms": inter_chunk_gap_p95_ms,
        "stall_threshold_ms": stall_threshold_ms,
        "stall_count": stall_count,
        "stall_rate": stall_rate,
        "audio_artifacts": verified_artifacts,
    }


def stream_live_chat(
    base_url: str,
    body: dict,
    *,
    timeout_s: float = 300.0,
) -> tuple[bytes, list[float]]:
    """Send a live POST /v1/chat/completions SSE request and capture per-frame monotonic timestamps."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=canonical_json(body),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )
    started = time.perf_counter()
    raw_chunks: list[bytes] = []
    frame_times_ms: list[float] = []
    try:
        resp = urllib.request.urlopen(req, timeout=timeout_s)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"POST {url} returned HTTP {exc.code}: {detail}") from exc

    with resp:
        if resp.status != 200:
            raise RuntimeError(f"POST {url} returned unexpected HTTP {resp.status}")
        ctype = resp.headers.get_content_type()
        if ctype != "text/event-stream":
            raise RuntimeError(f"POST {url} returned Content-Type {ctype!r}, expected 'text/event-stream'")
        for raw_line in resp:
            raw_chunks.append(raw_line)
            line = raw_line.decode("utf-8").rstrip("\r\n")
            if not line.startswith("data:"):
                continue
            payload = line[5:].lstrip(" ")
            if payload == "[DONE]":
                continue
            frame_times_ms.append((time.perf_counter() - started) * 1000.0)
    return b"".join(raw_chunks), frame_times_ms


def aggregate_arm(strategy: str, cases: list[dict], stall_threshold_ms: float) -> dict:
    """Compute per-arm summary statistics across all evaluated cases."""
    pooled_gaps = [gap for case in cases for gap in case["inter_chunk_gaps_ms"]]
    total_stalls = sum(case["stall_count"] for case in cases)
    total_gaps = len(pooled_gaps)
    return {
        "voice_strategy": strategy,
        "audio_stream": strategy == "sentence_streaming_tts",
        "completed_case_count": sum(1 for c in cases if c["terminal_state"] == "complete"),
        "total_case_count": len(cases),
        "stall_threshold_ms": stall_threshold_ms,
        "total_audio_chunks": sum(c["audio_chunk_count"] for c in cases),
        "total_audio_bytes": sum(c["audio_bytes_total"] for c in cases),
        "total_inter_chunk_gaps": total_gaps,
        "total_stall_count": total_stalls,
        "overall_stall_rate": round(total_stalls / total_gaps, 6) if total_gaps > 0 else None,
        "pooled_inter_chunk_gap_p95_ms": percentile(pooled_gaps, 95.0) if pooled_gaps else None,
        "metrics": {
            "ttft_ms": summarize_metric([c["ttft_ms"] for c in cases], include_dispersion=True),
            "ttfa_ms": summarize_metric([c["ttfa_ms"] for c in cases], include_dispersion=True),
            "ttfa_after_ttft_ms": summarize_metric(
                [c["ttfa_after_ttft_ms"] for c in cases],
                include_dispersion=True,
            ),
            "terminal_ms": summarize_metric([c["terminal_ms"] for c in cases], include_dispersion=True),
            "audio_chunk_count": summarize_metric([c["audio_chunk_count"] for c in cases]),
            "audio_bytes_total": summarize_metric([c["audio_bytes_total"] for c in cases]),
            "inter_chunk_gap_p95_ms": summarize_metric(
                [c["inter_chunk_gap_p95_ms"] for c in cases],
                not_applicable_when_empty=True,
                include_dispersion=True,
            ),
            "stall_count": summarize_metric([c["stall_count"] for c in cases]),
        },
    }


def build_comparison(
    arm_summaries: dict[str, dict],
    *,
    evidence_class: str,
) -> dict:
    """Compute cross-arm comparison (full_response_tts vs sentence_streaming_tts)."""
    full_arm = arm_summaries["full_response_tts"]["metrics"]
    stream_arm = arm_summaries["sentence_streaming_tts"]["metrics"]

    full_ttfa = full_arm["ttfa_ms"]["median"]
    stream_ttfa = stream_arm["ttfa_ms"]["median"]
    ttfa_speedup_x = (
        round(full_ttfa / stream_ttfa, 6)
        if full_ttfa is not None and stream_ttfa is not None and stream_ttfa > 0
        else None
    )
    ttfa_delta_ms = (
        round(stream_ttfa - full_ttfa, 6)
        if full_ttfa is not None and stream_ttfa is not None
        else None
    )

    full_after = full_arm["ttfa_after_ttft_ms"]["median"]
    stream_after = stream_arm["ttfa_after_ttft_ms"]["median"]
    ttfa_after_ttft_speedup_x = (
        round(full_after / stream_after, 6)
        if full_after is not None and stream_after is not None and stream_after > 0
        else None
    )

    full_term = full_arm["terminal_ms"]["median"]
    stream_term = stream_arm["terminal_ms"]["median"]
    terminal_ratio = (
        round(stream_term / full_term, 6)
        if full_term is not None and stream_term is not None and full_term > 0
        else None
    )

    if evidence_class == "CONTRACT_REPLAY":
        warning = (
            "CONTRACT_REPLAY (REAL_MODEL_EVIDENCE: NOT_RUN): metrics are derived from canned "
            "contract replay fixtures and MUST NOT be cited as real model or hardware speedup."
        )
    elif evidence_class == "CONTROLLED_LIVE":
        warning = (
            "CONTROLLED_LIVE: interleaved multi-repetition live measurement with verified "
            "provenance and zero failed cases; perceptual audio quality remains NOT_COVERED."
        )
    else:
        warning = (
            "UNCONTROLLED_LIVE: ad-hoc live server observations without an isolated controlled "
            "measurement manifest; perceptual audio quality remains NOT_COVERED."
        )

    return {
        "baseline_arm": "full_response_tts",
        "variant_arm": "sentence_streaming_tts",
        "ttfa_speedup_x": ttfa_speedup_x,
        "ttfa_delta_ms": ttfa_delta_ms,
        "ttfa_after_ttft_speedup_x": ttfa_after_ttft_speedup_x,
        "terminal_ratio": terminal_ratio,
        "evidence_warning": warning,
    }


def write_sha256sums(out_dir: Path, relative_files: list[str]) -> Path:
    """Write deterministic SHA256SUMS sorted by relative POSIX path."""
    lines = []
    for rel in sorted(set(relative_files)):
        digest = sha256_bytes((out_dir / rel).read_bytes())
        lines.append(f"{digest}  {rel}")
    sums_path = out_dir / "SHA256SUMS"
    sums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sums_path


def verify_sha256sums(out_dir: Path) -> int:
    """Verify every line of SHA256SUMS inside out_dir."""
    sums_path = out_dir / "SHA256SUMS"
    lines = [line for line in sums_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError("SHA256SUMS is empty")
    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise ValueError(f"malformed SHA256SUMS line: {line!r}")
        expected, rel = parts
        if not HEX_256_RE.fullmatch(expected):
            raise ValueError(f"invalid SHA-256 hex digest in SHA256SUMS: {expected!r}")
        target = out_dir / rel
        if not target.is_file():
            raise ValueError(f"SHA256SUMS references missing file: {rel}")
        actual = sha256_bytes(target.read_bytes())
        if actual != expected:
            raise ValueError(f"SHA256SUMS mismatch for {rel}: expected {expected}, got {actual}")
    return len(lines)


def run_ablation(
    experiment: VoiceExperiment,
    *,
    out_dir: Path,
    replay_only: bool,
    base_url: str | None = None,
    controlled_live: bool = False,
    provenance_json: Path | str | None = None,
    stall_threshold_ms: float | None = None,
    timeout_s: float = 300.0,
) -> dict:
    """Execute the voice collaboration ablation in replay, uncontrolled live, or controlled live mode."""
    if replay_only and controlled_live:
        raise ValueError("--controlled-live is mutually exclusive with --replay-only")
    if replay_only and provenance_json is not None:
        raise ValueError("--provenance-json cannot be used with --replay-only")
    if controlled_live and base_url is None:
        raise ValueError("--controlled-live requires --base-url")
    if replay_only == (base_url is not None):
        raise ValueError("specify exactly one of replay_only=True or base_url")

    threshold_ms = float(
        stall_threshold_ms
        if stall_threshold_ms is not None
        else experiment.config.get("stall_threshold_ms", DEFAULT_STALL_THRESHOLD_MS)
    )
    if not math.isfinite(threshold_ms) or threshold_ms <= 0.0:
        raise ValueError("stall_threshold_ms must be a finite positive number")

    matrix = verify_single_variable_matrix(experiment)

    cm_config: dict | None = None
    prov_sha256: str | None = None
    prov_data: dict | None = None
    provenance_verified = False

    if controlled_live:
        raw_cm = experiment.controlled_measurement
        if raw_cm is None:
            raise ValueError("--controlled-live requires a config with a controlled_measurement block")
        cm_config = validate_controlled_measurement(raw_cm)
        if cm_config["require_exclusive_gpu"] and provenance_json is None:
            raise ValueError(
                "--controlled-live with require_exclusive_gpu=true requires --provenance-json"
            )
        if provenance_json is not None:
            prov_sha256, prov_data = load_and_validate_provenance(
                provenance_json,
                require_exclusive_gpu=cm_config["require_exclusive_gpu"],
                strict=True,
                expected_vlm_weights_sha256=experiment.config["model"].get("weights_sha256"),
            )
            provenance_verified = True
    elif provenance_json is not None:
        prov_sha256, prov_data = load_and_validate_provenance(
            provenance_json,
            require_exclusive_gpu=False,
            strict=False,
        )

    replay_index: dict[tuple[str, str], dict] = {}
    if replay_only:
        if experiment.replay is None:
            raise ValueError("replay_only requested, but config lacks a replay bundle")
        for case in experiment.replay["cases"]:
            key = (case.get("voice_strategy"), case.get("prompt_id"))
            if key in replay_index:
                raise ValueError(f"duplicate replay case for {key}")
            replay_index[key] = case
        evidence_class = "CONTRACT_REPLAY"
        real_model_evidence = "NOT_RUN"
        claims_allowed = [
            "protocol_contract_verified",
            "single_variable_invariance_verified",
            "deterministic_metric_computation_verified",
        ]
        claims_forbidden = [
            "real_model_latency_improvement",
            "hardware_throughput_or_stall_rate_benchmark",
            "perceptual_audio_quality",
        ]
    elif controlled_live:
        evidence_class = "CONTROLLED_LIVE"
        real_model_evidence = "CONTROLLED_LIVE"
        claims_allowed = [
            "controlled_live_latency_and_stall_comparison",
            "single_variable_invariance_verified",
            "interleaved_multi_repetition_measurement",
        ]
        claims_forbidden = [
            "perceptual_audio_quality",
            "full_duplex_barge_in_latency",
        ]
    else:
        evidence_class = "UNCONTROLLED_LIVE"
        real_model_evidence = "UNCONTROLLED_LIVE"
        claims_allowed = ["uncontrolled_live_observation"]
        claims_forbidden = [
            "controlled_reproducible_benchmark_without_isolation_manifest",
            "perceptual_audio_quality",
        ]

    arm_cases: dict[str, list[dict]] = {}
    arm_summaries: dict[str, dict] = {}
    warmup_summary: dict | None = None
    repetitions_report: list[dict] | None = None
    schedule_execution: list[dict] | None = None

    if controlled_live:
        assert base_url is not None and cm_config is not None
        schedule_execution = []

        # 1. Warmup phase: execute full live requests and exclude from measured aggregates.
        warmup_runs = cm_config["warmup_runs"]
        warmup_schedule = build_arm_schedule(warmup_runs, cm_config["schedule_mode"])
        warmup_by_arm: dict[str, list[dict]] = {s: [] for s in VOICE_STRATEGIES}
        warmup_requests: list[dict] = []

        for w_idx, w_arm_order in enumerate(warmup_schedule):
            for order_idx, strategy in enumerate(w_arm_order):
                for item in matrix["by_arm"][strategy]:
                    pid = item["prompt_id"]
                    raw_sse, frame_times_ms = stream_live_chat(base_url, item["request"], timeout_s=timeout_s)
                    evaluated = evaluate_sse_case(
                        raw_sse=raw_sse,
                        strategy=strategy,
                        prompt_id=pid,
                        request_sha256=item["request_sha256"],
                        fixed_req_sha256=item["fixed_request_sha256"],
                        stall_threshold_ms=threshold_ms,
                        live_frame_times_ms=frame_times_ms,
                    )
                    warmup_by_arm[strategy].append(evaluated)
                    warmup_requests.append({
                        "warmup_run": w_idx,
                        "arm_order_index": order_idx,
                        "voice_strategy": strategy,
                        "prompt_id": pid,
                        "terminal_state": evaluated["terminal_state"],
                        "ttft_ms": evaluated["ttft_ms"],
                        "ttfa_ms": evaluated["ttfa_ms"],
                        "terminal_ms": evaluated["terminal_ms"],
                        "request_sha256": evaluated["request_sha256"],
                        "response_sha256": evaluated["response_sha256"],
                    })
                    schedule_execution.append({
                        "phase": "warmup",
                        "warmup_run": w_idx,
                        "arm_order_index": order_idx,
                        "voice_strategy": strategy,
                        "prompt_id": pid,
                        "status": "completed",
                    })

        warmup_summary = {
            "warmup_runs": warmup_runs,
            "completed_request_count": len(warmup_requests),
            "excluded_from_measured_aggregates": True,
            "by_arm": {
                strategy: {
                    "runs": warmup_runs,
                    "completed_count": len(warmup_by_arm[strategy]),
                    "ttft_ms_median": summarize_metric([c["ttft_ms"] for c in warmup_by_arm[strategy]])["median"],
                    "ttfa_ms_median": summarize_metric([c["ttfa_ms"] for c in warmup_by_arm[strategy]])["median"],
                    "terminal_ms_median": summarize_metric(
                        [c["terminal_ms"] for c in warmup_by_arm[strategy]]
                    )["median"],
                }
                for strategy in VOICE_STRATEGIES
            },
            "requests": warmup_requests,
        }

        # 2. Measured interleaved repetitions (fail closed on any error; max_failed_cases == 0).
        measured_schedule = build_arm_schedule(cm_config["repetitions"], cm_config["schedule_mode"])
        arm_cases = {s: [] for s in VOICE_STRATEGIES}
        repetitions_report = []

        for rep_idx, arm_order in enumerate(measured_schedule):
            rep_cases: dict[str, list[dict]] = {s: [] for s in VOICE_STRATEGIES}
            for order_idx, strategy in enumerate(arm_order):
                for item in matrix["by_arm"][strategy]:
                    pid = item["prompt_id"]
                    raw_sse, frame_times_ms = stream_live_chat(base_url, item["request"], timeout_s=timeout_s)
                    evaluated = evaluate_sse_case(
                        raw_sse=raw_sse,
                        strategy=strategy,
                        prompt_id=pid,
                        request_sha256=item["request_sha256"],
                        fixed_req_sha256=item["fixed_request_sha256"],
                        stall_threshold_ms=threshold_ms,
                        live_frame_times_ms=frame_times_ms,
                    )
                    evaluated["repetition"] = rep_idx
                    evaluated["arm_order_index"] = order_idx
                    rep_cases[strategy].append(evaluated)
                    arm_cases[strategy].append(evaluated)
                    schedule_execution.append({
                        "phase": "measured",
                        "repetition": rep_idx,
                        "arm_order_index": order_idx,
                        "voice_strategy": strategy,
                        "prompt_id": pid,
                        "status": "completed",
                    })
            rep_arm_metrics = {
                strategy: aggregate_arm(strategy, rep_cases[strategy], threshold_ms)
                for strategy in VOICE_STRATEGIES
            }
            repetitions_report.append({
                "repetition": rep_idx,
                "arm_order": list(arm_order),
                "arm_metrics": rep_arm_metrics,
            })

        expected_per_arm = cm_config["repetitions"] * len(experiment.prompts)
        for strategy in VOICE_STRATEGIES:
            summary = aggregate_arm(strategy, arm_cases[strategy], threshold_ms)
            if (
                summary["completed_case_count"] != expected_per_arm
                or summary["total_case_count"] != expected_per_arm
            ):
                raise RuntimeError(
                    f"controlled measurement failed completeness check for {strategy}: "
                    f"completed={summary['completed_case_count']}, expected={expected_per_arm}"
                )
            arm_summaries[strategy] = summary
    else:
        for strategy in VOICE_STRATEGIES:
            cases_out: list[dict] = []
            for item in matrix["by_arm"][strategy]:
                pid = item["prompt_id"]
                req = item["request"]
                req_sha = item["request_sha256"]
                fixed_sha = item["fixed_request_sha256"]

                if replay_only:
                    key = (strategy, pid)
                    if key not in replay_index:
                        raise ValueError(f"replay bundle missing case for {strategy}/{pid}")
                    rcase = replay_index[key]
                    if rcase.get("fixed_request_sha256") != fixed_sha:
                        raise ValueError(
                            f"replay case {strategy}/{pid} fixed_request_sha256 mismatch: "
                            f"expected {fixed_sha}, got {rcase.get('fixed_request_sha256')}"
                        )
                    if rcase.get("request_sha256") != req_sha:
                        raise ValueError(
                            f"replay case {strategy}/{pid} request_sha256 mismatch: "
                            f"expected {req_sha}, got {rcase.get('request_sha256')}"
                        )
                    raw_sse = base64.b64decode(rcase["response_sse_base64"], validate=True)
                    _check_sha256(raw_sse, rcase.get("response_sha256"), f"replay response {strategy}/{pid}")
                    evaluated = evaluate_sse_case(
                        raw_sse=raw_sse,
                        strategy=strategy,
                        prompt_id=pid,
                        request_sha256=req_sha,
                        fixed_req_sha256=fixed_sha,
                        stall_threshold_ms=threshold_ms,
                        recorded_latency_us=rcase.get("latency_us"),
                    )
                else:
                    assert base_url is not None
                    raw_sse, frame_times_ms = stream_live_chat(base_url, req, timeout_s=timeout_s)
                    evaluated = evaluate_sse_case(
                        raw_sse=raw_sse,
                        strategy=strategy,
                        prompt_id=pid,
                        request_sha256=req_sha,
                        fixed_req_sha256=fixed_sha,
                        stall_threshold_ms=threshold_ms,
                        live_frame_times_ms=frame_times_ms,
                    )
                cases_out.append(evaluated)
            arm_cases[strategy] = cases_out
            arm_summaries[strategy] = aggregate_arm(strategy, cases_out, threshold_ms)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written_files: list[str] = []

    for strategy in VOICE_STRATEGIES:
        arm_report: dict = {
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "ablation_id": experiment.config["ablation_id"],
            "variable": VARIABLE_NAME,
            "voice_strategy": strategy,
            "evidence_class": evidence_class,
            "real_model_evidence": real_model_evidence,
            "fixed_matrix_sha256": matrix["fixed_matrix_sha256"],
            "summary": arm_summaries[strategy],
            "cases": arm_cases[strategy],
        }
        if controlled_live:
            arm_report["controlled_measurement_status"] = "COMPLETED"
        rel_path = f"arms/{strategy}/report.json"
        target = out_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(canonical_json(arm_report) + b"\n")
        written_files.append(rel_path)

    provenance_section: dict = {
        "config_sha256": experiment.config_sha256,
        "dataset_id": experiment.dataset.get("dataset_id"),
        "dataset_sha256": experiment.dataset_sha256,
        "replay_sha256": experiment.replay_sha256 if replay_only else None,
        "model": copy.deepcopy(experiment.config["model"]),
        "mode": "replay" if replay_only else ("controlled_live" if controlled_live else "live"),
    }
    if prov_data is not None:
        provenance_section["provenance_json_sha256"] = prov_sha256
        provenance_section["external_provenance"] = prov_data

    top_report: dict = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "ablation_id": experiment.config["ablation_id"],
        "variable": VARIABLE_NAME,
        "voice_strategies": list(VOICE_STRATEGIES),
        "evidence_class": evidence_class,
        "real_model_evidence": real_model_evidence,
        "claims_allowed": claims_allowed,
        "claims_forbidden": claims_forbidden,
        "provenance": provenance_section,
        "single_variable": {
            "ablated_field": "audio.stream",
            "arms": {
                "full_response_tts": {"audio.stream": False},
                "sentence_streaming_tts": {"audio.stream": True},
            },
            "single_variable_verified": matrix["single_variable_verified"],
            "fixed_matrix_sha256": matrix["fixed_matrix_sha256"],
            "fixed_request_sha256_by_prompt": matrix["fixed_request_sha256_by_prompt"],
        },
        "stall_threshold_ms": threshold_ms,
        "arms": arm_summaries,
        "comparison": build_comparison(arm_summaries, evidence_class=evidence_class),
        "arm_reports": {
            strategy: f"arms/{strategy}/report.json" for strategy in VOICE_STRATEGIES
        },
    }
    if controlled_live:
        assert cm_config is not None and warmup_summary is not None
        assert repetitions_report is not None and schedule_execution is not None
        top_report["controlled_measurement_status"] = "COMPLETED"
        top_report["controlled_measurement"] = {
            **cm_config,
            "failed_case_count": 0,
            "provenance_verified": provenance_verified,
        }
        top_report["warmup_summary"] = warmup_summary
        top_report["repetitions"] = repetitions_report
        top_report["schedule_execution"] = schedule_execution

    top_path = out_dir / "report.json"
    top_path.write_bytes(canonical_json(top_report) + b"\n")
    written_files.append("report.json")

    write_sha256sums(out_dir, written_files)
    verify_sha256sums(out_dir)
    return top_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage C Ablation #3: Voice Collaboration Strategy (full_response_tts vs sentence_streaming_tts)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to voice-strategy ablation config JSON",
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--replay-only",
        action="store_true",
        help="Run deterministic offline replay (produces CONTRACT_REPLAY; REAL_MODEL_EVIDENCE: NOT_RUN)",
    )
    mode_group.add_argument(
        "--base-url",
        default=None,
        help="Live ummInfer server base URL (e.g. http://127.0.0.1:43301; produces UNCONTROLLED_LIVE unless --controlled-live)",
    )
    parser.add_argument(
        "--controlled-live",
        action="store_true",
        help=(
            "Execute controlled multi-repetition interleaved live measurement (requires --base-url, "
            "a config with controlled_measurement, and --provenance-json when require_exclusive_gpu=true)"
        ),
    )
    parser.add_argument(
        "--provenance-json",
        type=Path,
        default=None,
        help=(
            "Path to provenance manifest JSON (optional for ad-hoc --base-url; required for "
            "--controlled-live when require_exclusive_gpu=true)"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for report.json, arms/*/report.json, and SHA256SUMS",
    )
    parser.add_argument(
        "--stall-threshold-ms",
        type=float,
        default=None,
        help="Inter-audio-chunk gap threshold in ms counted as a stall (default: config value or 500.0)",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=300.0,
        help="Per-request HTTP/SSE timeout in seconds for --base-url live runs (default: 300.0)",
    )
    parser.add_argument(
        "--allow-remote-http",
        action="store_true",
        help="Explicitly permit plain HTTP against a non-loopback host in --base-url mode",
    )
    args = parser.parse_args(argv)

    try:
        if args.controlled_live and args.replay_only:
            raise ValueError("--controlled-live is mutually exclusive with --replay-only")
        if args.controlled_live and args.base_url is None:
            raise ValueError("--controlled-live requires --base-url")
        if args.replay_only and args.provenance_json is not None:
            raise ValueError("--provenance-json cannot be used with --replay-only")

        experiment = load_experiment(args.config)
        normalized_url = (
            validate_base_url(args.base_url, allow_remote_http=args.allow_remote_http)
            if args.base_url is not None
            else None
        )
        report = run_ablation(
            experiment,
            out_dir=args.out,
            replay_only=args.replay_only,
            base_url=normalized_url,
            controlled_live=args.controlled_live,
            provenance_json=args.provenance_json,
            stall_threshold_ms=args.stall_threshold_ms,
            timeout_s=args.timeout_s,
        )
    except Exception as exc:
        print(f"VOICE ABLATION ERROR: {exc}", file=sys.stderr)
        return 1

    cmp = report["comparison"]
    status_part = (
        f" | STATUS: {report['controlled_measurement_status']}"
        if "controlled_measurement_status" in report
        else ""
    )
    print(
        f"VOICE ABLATION OK [{report['evidence_class']} | REAL_MODEL_EVIDENCE: {report['real_model_evidence']}"
        f"{status_part}]: "
        f"fixed_matrix_sha256={report['single_variable']['fixed_matrix_sha256'][:16]}... "
        f"ttfa_speedup_x={cmp['ttfa_speedup_x']} out={args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
