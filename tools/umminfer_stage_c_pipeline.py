#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage C Live-to-Training E2E Pipeline Driver for ummInfer.

Connects exported ``umminfer_rl_trajectory`` schema-v1/v2 JSONL records (from
offline exports or optional ``--capture-live`` SSE capture) directly through:

1. Strict trajectory & content-addressed media (CAS) validation
   (``umminfer_rl_trajectory`` + ``umminfer_sft_dataset.verify_record_integrity``).
2. Deterministic SFT (``sft/``), Preference DPO+GRPO (``preference/``), and
   combined Joint (``joint/``) dataset pack construction with pluggable
   export-only reward scoring (default ``umminfer_rl_reward:CompositeUmmReward``)
   and mandatory zero prompt-leakage auditing (``leakage_count == 0``).
3. Decoupled Stage C training / ablation-matrix execution via
   ``umminfer_joint_trainer`` in ``dry_run_micro_step`` (default), ``plan``, or
   ``gpu_launch_script`` mode, binding the freshly built dataset pack manifests.
4. Content-addressed ``stage_c_e2e_report.json`` +
   ``stage_c_e2e_report.json.sha256`` with explicit ``REAL_TRAINING: "NOT_RUN"``.

Architecture boundary:
- Pure Python 3 stdlib only (zero torch/numpy/transformers).
- 100% decoupled from C++ serving code under ``src/`` and ``include/``.
- Never fabricates GPU training runs or claims model weights were updated.
"""

from __future__ import annotations

import argparse
import copy
import io
import ipaddress
import json
import math
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import umminfer_joint_trainer as trainer  # noqa: E402
import umminfer_rl_trajectory as traj  # noqa: E402
import umminfer_sft_dataset as ds  # noqa: E402

PIPELINE_SCHEMA_VERSION = 1
DEFAULT_REWARD_SPEC = "umminfer_rl_reward:CompositeUmmReward"
DEFAULT_ABLATION_MATRIX = REPO_ROOT / "configs" / "training" / "ablation_matrix.json"
DEFAULT_DRY_RUN_STEPS = 2
DEFAULT_EXECUTION_MODE = "dry_run_micro_step"
DEFAULT_LIVE_MODEL = "qwen3-native"
DEFAULT_LIVE_TIMEOUT_S = 300.0
_ALLOWED_LATENCY_KEYS = frozenset(("ttft_ms", "ttfa_ms", "terminal_ms", "barge_in_ms"))

# Deterministic live-capture prompt suite: covers pure-text (allow_text_only),
# visual-scene image, visual-scene video/auto, and mixed narrative beats, with
# two contrasting rollouts per prompt so preference pairs carry real reward margin.
LIVE_CAPTURE_PROMPT_SUITE = (
    {
        "prompt_id": "stage_c_live_pure_text",
        "prompt": (
            "Prove concisely in four numbered lines that the square root of 2 "
            "is irrational using integer parity. Pure symbolic math prose only; "
            "do not insert any visual illustrations."
        ),
        "expected_visual_modalities": [],
        "rollouts": (
            {
                "rollout_label": "auto_text_only",
                "temperature": 0.0,
                "seed": 42,
                "max_tokens": 256,
                "modalities": ["text", "image", "video"],
                "narrative": {
                    "enabled": True,
                    "visual_mode": "auto",
                    "max_slots": 2,
                    "allow_text_only": True,
                },
            },
            {
                "rollout_label": "forced_image_contrast",
                "temperature": 0.7,
                "seed": 43,
                "max_tokens": 256,
                "modalities": ["text", "image"],
                "narrative": {
                    "enabled": True,
                    "visual_mode": "image",
                    "max_slots": 1,
                    "allow_text_only": False,
                },
            },
        ),
    },
    {
        "prompt_id": "stage_c_live_visual_image",
        "prompt": (
            "Describe a sunlit botanical conservatory in two vivid sentences "
            "and illustrate the glass dome with a single still frame."
        ),
        "expected_visual_modalities": ["image"],
        "rollouts": (
            {
                "rollout_label": "image_primary",
                "temperature": 0.2,
                "seed": 42,
                "max_tokens": 256,
                "modalities": ["text", "image"],
                "narrative": {
                    "enabled": True,
                    "visual_mode": "image",
                    "max_slots": 1,
                    "allow_text_only": False,
                },
            },
            {
                "rollout_label": "auto_contrast",
                "temperature": 0.8,
                "seed": 43,
                "max_tokens": 256,
                "modalities": ["text", "image", "video"],
                "narrative": {
                    "enabled": True,
                    "visual_mode": "auto",
                    "max_slots": 2,
                    "allow_text_only": True,
                },
            },
        ),
    },
    {
        "prompt_id": "stage_c_live_visual_video_auto",
        "prompt": (
            "Narrate a short ocean wave cresting and breaking along a rocky "
            "shoreline in motion, showing the continuous wave movement."
        ),
        "expected_visual_modalities": ["video"],
        "rollouts": (
            {
                "rollout_label": "video_primary",
                "temperature": 0.2,
                "seed": 42,
                "max_tokens": 256,
                "modalities": ["text", "image", "video"],
                "narrative": {
                    "enabled": True,
                    "visual_mode": "video",
                    "max_slots": 1,
                    "allow_text_only": False,
                },
            },
            {
                "rollout_label": "image_contrast",
                "temperature": 0.7,
                "seed": 43,
                "max_tokens": 256,
                "modalities": ["text", "image"],
                "narrative": {
                    "enabled": True,
                    "visual_mode": "image",
                    "max_slots": 1,
                    "allow_text_only": False,
                },
            },
        ),
    },
    {
        "prompt_id": "stage_c_live_mixed_narrative",
        "prompt": (
            "Narrate a two-beat documentary vignette of a mountain observatory "
            "at dusk transitioning into a starry night time-lapse. Include one "
            "still frame for the dome at sunset and one motion clip for the "
            "rotating night sky."
        ),
        "expected_visual_modalities": ["image", "video"],
        "rollouts": (
            {
                "rollout_label": "auto_mixed_primary",
                "temperature": 0.2,
                "seed": 42,
                "max_tokens": 320,
                "modalities": ["text", "image", "video"],
                "narrative": {
                    "enabled": True,
                    "visual_mode": "auto",
                    "max_slots": 2,
                    "allow_text_only": False,
                },
            },
            {
                "rollout_label": "image_only_contrast",
                "temperature": 0.7,
                "seed": 43,
                "max_tokens": 320,
                "modalities": ["text", "image"],
                "narrative": {
                    "enabled": True,
                    "visual_mode": "image",
                    "max_slots": 1,
                    "allow_text_only": False,
                },
            },
        ),
    },
)

_DEFAULT_IMAGE_OPTS = {
    "size": "512x512",
    "output_format": "png",
    "steps": 4,
    "seed": 42,
}
_DEFAULT_VIDEO_OPTS = {
    "size": "480x320",
    "output_format": "mp4",
    "num_frames": 13,
    "steps": 15,
    "guidance_scale": 5.0,
    "seed": 42,
}
_DEFAULT_AUDIO_OPTS = {
    "voice": "Chelsie",
    "format": "wav",
    "stream": True,
}


def _is_finite(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def _prepare_out_dir(out_dir: Path, *, overwrite: bool) -> Path:
    out_path = Path(out_dir)
    if out_path.exists():
        if not out_path.is_dir():
            raise ValueError(f"output path exists and is not a directory: {out_path}")
        if any(out_path.iterdir()) and not overwrite:
            raise ValueError(
                "output directory must be absent or empty (or pass --overwrite)"
            )
    else:
        out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def validate_server_url(url: str, *, allow_remote_http: bool = False) -> str:
    """Validate and normalize a serving base URL with strict transport hygiene."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Server URL must be a non-empty string")
    parsed = urllib.parse.urlsplit(url.strip())
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(
            f"Unsupported URL scheme {scheme!r}; expected http or https"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Server URL must not embed credentials")
    host = parsed.hostname
    if not host:
        raise ValueError("Server URL must include a hostname")
    if parsed.query or parsed.fragment:
        raise ValueError("Server URL must not include query parameters or fragments")

    is_loopback = host.lower() == "localhost"
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            is_loopback = False

    if scheme == "http" and not is_loopback and not allow_remote_http:
        raise ValueError(
            "Refusing non-loopback plain HTTP URL; use https:// or pass "
            "--allow-remote-http explicitly"
        )

    return f"{scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"


def build_auth_headers(api_key_env: str | None) -> dict[str, str]:
    if not api_key_env:
        return {}
    token = os.environ.get(api_key_env, "").strip()
    if not token:
        raise ValueError(f"Environment variable {api_key_env!r} is unset or empty")
    return {"Authorization": f"Bearer {token}"}


def resolve_trajectory_paths(raw_paths) -> list[Path]:
    """Expand trajectory file and directory arguments into sorted JSONL paths."""
    if not raw_paths:
        raise ValueError("at least one trajectory JSONL file or directory is required")
    collected: list[Path] = []
    seen: set[Path] = set()
    for item in raw_paths:
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(f"trajectory input not found: {path}")
        if path.is_file():
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                collected.append(resolved)
        elif path.is_dir():
            matches = sorted(
                p.resolve() for p in path.rglob("*.jsonl") if p.is_file()
            )
            if not matches:
                raise ValueError(
                    f"trajectory directory contains no .jsonl files: {path}"
                )
            for match in matches:
                if match not in seen:
                    seen.add(match)
                    collected.append(match)
        else:
            raise ValueError(f"unsupported trajectory input path: {path}")
    collected.sort(key=lambda p: (p.name, str(p)))
    return collected


def validate_trajectory_inputs(
    input_paths,
    *,
    artifact_dir=None,
    strict_artifacts: bool = False,
) -> tuple[list[Path], dict]:
    """Strictly validate every trajectory record via umminfer_rl_trajectory rules.

    Fails closed on malformed JSON, unsupported schema_version (must be 1 or 2),
    digest/event/slot/artifact-chain corruption, or missing/tampered CAS files.
    """
    resolved_files = resolve_trajectory_paths(input_paths)
    explicit_artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
    if explicit_artifact_dir is not None and not explicit_artifact_dir.is_dir():
        raise FileNotFoundError(
            f"artifact directory not found: {explicit_artifact_dir}"
        )

    schema_counts = {"1": 0, "2": 0}
    terminal_counts = {"completed": 0, "cancelled": 0, "failed": 0}
    file_entries = []
    total_records = 0
    verified_input_media = 0
    verified_output_media = 0
    any_disk_verified = False

    for path in resolved_files:
        raw_bytes = path.read_bytes()
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{path.name}: trajectory file is not valid UTF-8: {exc}"
            ) from exc

        effective_artifact_dir = (
            explicit_artifact_dir
            if explicit_artifact_dir is not None
            else path.parent / "artifacts"
        )
        require_disk = bool(strict_artifacts or explicit_artifact_dir is not None)

        file_record_count = 0
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path.name}:{line_no}: invalid JSON: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"{path.name}:{line_no}: trajectory record must be a JSON object"
                )
            version = record.get("schema_version")
            if isinstance(version, bool) or not isinstance(version, int) \
                    or version not in traj.SUPPORTED_TRAJECTORY_SCHEMA_VERSIONS:
                raise ValueError(
                    f"{path.name}:{line_no}: unsupported schema_version {version!r}; "
                    f"expected one of {sorted(traj.SUPPORTED_TRAJECTORY_SCHEMA_VERSIONS)}"
                )

            ok, category, message, stats = ds.verify_record_integrity(
                record,
                artifact_dir=effective_artifact_dir,
                require_artifact_dir=require_disk,
            )
            if not ok:
                raise ValueError(
                    f"{path.name}:{line_no}: trajectory integrity failed "
                    f"({category}): {message}"
                )

            file_record_count += 1
            total_records += 1
            schema_counts[str(version)] += 1
            state = record["terminal_state"]
            terminal_counts[state] = terminal_counts.get(state, 0) + 1
            if stats["checked_disk"]:
                any_disk_verified = True
            verified_input_media += stats["checked_input_media_refs"]
            verified_output_media += stats["checked_output_media_refs"]

        if file_record_count == 0:
            raise ValueError(f"{path.name}: trajectory file contains zero records")

        file_entries.append({
            "name": path.name,
            "sha256": traj.sha256_bytes(raw_bytes),
            "bytes": len(raw_bytes),
            "record_count": file_record_count,
        })

    file_entries.sort(key=lambda item: (item["sha256"], item["name"]))
    summary = {
        "file_count": len(file_entries),
        "total_records": total_records,
        "completed_records": terminal_counts["completed"],
        "cancelled_records": terminal_counts["cancelled"],
        "failed_records": terminal_counts["failed"],
        "schema_versions": schema_counts,
        "artifact_verification_mode": (
            "verified_on_disk" if any_disk_verified else "no_artifact_dir"
        ),
        "verified_input_media_refs": verified_input_media,
        "verified_output_media_refs": verified_output_media,
        "files": file_entries,
    }
    return resolved_files, summary


def _sanitize_ablation_metadata(metadata: dict | None) -> dict | None:
    if metadata is None:
        return None
    clean = copy.deepcopy(metadata)
    if "latency" in clean and clean["latency"] is not None:
        latency = {}
        for key, value in clean["latency"].items():
            if key not in _ALLOWED_LATENCY_KEYS or value is None:
                continue
            if not _is_finite(value) or float(value) < 0.0:
                raise ValueError(f"invalid latency observation {key}={value!r}")
            latency[key] = round(float(value), 3)
        clean["latency"] = latency
    return clean


def _rewrite_last_jsonl_record(jsonl_path: Path, record: dict) -> None:
    raw_lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    if not raw_lines:
        raise RuntimeError("expected recorded JSONL line before annotation")
    raw_lines[-1] = traj.canonical_json(record).decode("utf-8")
    payload = ("\n".join(raw_lines) + "\n").encode("utf-8")
    _write_bytes_atomic(jsonl_path, payload)


def sse_stream_to_v2_record(
    response,
    *,
    request_body: dict,
    recorder: traj.TrajectoryRecorder,
    session_id: str,
    ablation_metadata: dict | None = None,
    event_observer=None,
) -> dict:
    """Convert an HTTP/SSE response stream into a verified schema-v2 record.

    Stores all decoded input and output media in ``recorder.artifact_dir`` using
    content-addressed ``<sha256>.<ext>`` files and verifies integrity immediately.
    """
    recorder.artifact_dir.mkdir(parents=True, exist_ok=True)
    params, option_inputs = traj.request_parameter_references(
        request_body, recorder.artifact_dir
    )
    inputs = traj.input_references(
        request_body["messages"], recorder.artifact_dir
    ) + option_inputs

    details: dict = {}
    request_id, events, output_text, terminal_state, error = traj.parse_sse_stream(
        response,
        artifact_dir=recorder.artifact_dir,
        details=details,
        event_observer=event_observer,
    )
    artifacts = details.get("artifacts", [])
    record = recorder.record(
        session_id=session_id,
        request_id=request_id or traj._synthetic_request_id(events),
        endpoint="/v1/chat/completions",
        inputs=inputs,
        params=params,
        events=events,
        terminal_state=terminal_state,
        error=error,
        output_text=output_text,
        artifacts=artifacts,
    )

    sanitized_ablation = _sanitize_ablation_metadata(ablation_metadata)
    if sanitized_ablation is not None:
        record = copy.deepcopy(record)
        record["ablation"] = sanitized_ablation
        _rewrite_last_jsonl_record(recorder.jsonl_path, record)

    ok, category, message, _stats = ds.verify_record_integrity(
        record,
        artifact_dir=recorder.artifact_dir,
        require_artifact_dir=True,
    )
    if not ok:
        raise ValueError(
            f"converted schema-v2 record failed integrity verification "
            f"({category}): {message}"
        )
    return record


def sse_bytes_to_v2_record(
    sse_payload: bytes,
    *,
    request_body: dict,
    recorder: traj.TrajectoryRecorder,
    session_id: str,
    ablation_metadata: dict | None = None,
) -> dict:
    """Convert raw SSE frame bytes into a verified schema-v2 trajectory record."""
    if not isinstance(sse_payload, (bytes, bytearray)):
        raise TypeError("sse_payload must be bytes")
    return sse_stream_to_v2_record(
        io.BytesIO(bytes(sse_payload)),
        request_body=request_body,
        recorder=recorder,
        session_id=session_id,
        ablation_metadata=ablation_metadata,
    )


def _build_live_request_body(
    case: dict,
    rollout: dict,
    *,
    model: str,
    enable_audio: bool = False,
) -> dict:
    modalities = list(rollout["modalities"])
    if enable_audio and "audio" not in modalities:
        modalities.append("audio")

    body = {
        "model": model,
        "stream": True,
        "max_tokens": int(rollout["max_tokens"]),
        "temperature": float(rollout["temperature"]),
        "seed": int(rollout["seed"]),
        "messages": [{"role": "user", "content": case["prompt"]}],
        "modalities": modalities,
        "narrative": copy.deepcopy(rollout["narrative"]),
    }
    if "image" in modalities:
        body["image"] = copy.deepcopy(_DEFAULT_IMAGE_OPTS)
    if "video" in modalities:
        body["video"] = copy.deepcopy(_DEFAULT_VIDEO_OPTS)
    if "audio" in modalities:
        body["audio"] = copy.deepcopy(_DEFAULT_AUDIO_OPTS)
    return body


def capture_live_trajectories(
    base_url: str,
    *,
    out_dir: Path,
    model: str = DEFAULT_LIVE_MODEL,
    prompt_suite=LIVE_CAPTURE_PROMPT_SUITE,
    timeout: float = DEFAULT_LIVE_TIMEOUT_S,
    extra_headers: dict[str, str] | None = None,
    allow_remote_http: bool = False,
    enable_audio: bool = False,
) -> tuple[Path, Path, dict]:
    """Capture live SSE rollouts into verified schema-v2 JSONL + local CAS dir."""
    normalized_url = validate_server_url(
        base_url, allow_remote_http=allow_remote_http
    )
    capture_dir = Path(out_dir) / "live_capture"
    capture_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = capture_dir / "trajectories.jsonl"
    artifact_dir = capture_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if jsonl_path.exists():
        jsonl_path.unlink()

    recorder = traj.TrajectoryRecorder(jsonl_path, artifact_dir=artifact_dir)
    endpoint_url = normalized_url.rstrip("/") + "/v1/chat/completions"

    case_summaries = []
    completed = 0
    failed = 0

    for case in prompt_suite:
        prompt_id = case["prompt_id"]
        expected_modalities = list(case["expected_visual_modalities"])
        for rollout_idx, rollout in enumerate(case["rollouts"]):
            body = _build_live_request_body(
                case, rollout, model=model, enable_audio=enable_audio
            )
            session_id = f"{prompt_id}:r{rollout_idx}:{rollout['rollout_label']}"
            headers = {
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            }
            if extra_headers:
                headers.update(extra_headers)

            req = urllib.request.Request(
                endpoint_url,
                data=json.dumps(body).encode("utf-8"),
                method="POST",
                headers=headers,
            )

            started = time.perf_counter()
            timings: dict[str, float | None] = {
                "ttft_ms": None,
                "ttfa_ms": None,
            }

            def _observer(event):
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                etype = event.get("type")
                if timings["ttft_ms"] is None and etype in ("vlm_delta", "output_delta"):
                    timings["ttft_ms"] = elapsed_ms
                if timings["ttfa_ms"] is None and etype == "artifact":
                    art = event.get("artifact") or {}
                    if art.get("media_type") == "audio":
                        timings["ttfa_ms"] = elapsed_ms

            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"chat HTTP {resp.status}")
                    ctype = resp.headers.get_content_type()
                    if ctype != "text/event-stream":
                        raise RuntimeError(
                            f"expected text/event-stream, got {ctype!r}"
                        )
                    record = sse_stream_to_v2_record(
                        resp,
                        request_body=body,
                        recorder=recorder,
                        session_id=session_id,
                        ablation_metadata=None,
                        event_observer=_observer,
                    )
            except (urllib.error.URLError, OSError, RuntimeError, ValueError) as exc:
                terminal_ms = (time.perf_counter() - started) * 1000.0
                message = f"live capture error: {exc}"
                params, option_inputs = traj.request_parameter_references(
                    body, recorder.artifact_dir
                )
                inputs = traj.input_references(
                    body["messages"], recorder.artifact_dir
                ) + option_inputs
                events = [{
                    "seq": 0,
                    "type": "error",
                    "server_seq": None,
                    "stage": None,
                    "is_final": True,
                    "text": message,
                    "error": message,
                }]
                record = recorder.record(
                    session_id=session_id,
                    request_id=traj._synthetic_request_id(events),
                    endpoint="/v1/chat/completions",
                    inputs=inputs,
                    params=params,
                    events=events,
                    terminal_state=traj.TERMINAL_FAILED,
                    error=message,
                    output_text="",
                    artifacts=[],
                )
            else:
                terminal_ms = (time.perf_counter() - started) * 1000.0

            latency_obs = {"terminal_ms": round(max(0.0, terminal_ms), 3)}
            if timings["ttft_ms"] is not None:
                latency_obs["ttft_ms"] = round(max(0.0, timings["ttft_ms"]), 3)
            if timings["ttfa_ms"] is not None:
                latency_obs["ttfa_ms"] = round(max(0.0, timings["ttfa_ms"]), 3)

            record = copy.deepcopy(record)
            record["ablation"] = {
                "prompt_id": prompt_id,
                "rollout_index": rollout_idx,
                "rollout_label": rollout["rollout_label"],
                "expected_visual_modalities": expected_modalities,
                "latency": latency_obs,
            }
            _rewrite_last_jsonl_record(jsonl_path, record)

            if record["terminal_state"] == traj.TERMINAL_COMPLETED:
                completed += 1
            else:
                failed += 1

            case_summaries.append({
                "prompt_id": prompt_id,
                "rollout_index": rollout_idx,
                "rollout_label": rollout["rollout_label"],
                "terminal_state": record["terminal_state"],
                "visual_slot_count": len(record.get("visual_slots") or []),
                "artifact_count": len(record.get("artifacts") or []),
                "latency_ms": latency_obs,
            })

    summary = {
        "model": model,
        "prompt_count": len(prompt_suite),
        "captured_records": len(case_summaries),
        "completed_records": completed,
        "failed_records": failed,
        "trajectory_file": "live_capture/trajectories.jsonl",
        "trajectory_sha256": ds.sha256_file(jsonl_path),
        "artifact_dir": "live_capture/artifacts",
        "cases": case_summaries,
    }
    return jsonl_path, artifact_dir, summary


def _load_jsonl_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _verify_cross_pack_integrity(sft_dir: Path, pref_dir: Path, joint_dir: Path,
                                 sft_manifest: dict, pref_manifest: dict,
                                 joint_manifest: dict) -> dict:
    """Verify byte-level identity between joint/ and sft/+preference/ plus zero leakage."""
    for key in ("sft_train", "sft_eval"):
        if joint_manifest["outputs"][key]["sha256"] != sft_manifest["outputs"][key]["sha256"]:
            raise RuntimeError(
                f"joint/{key} SHA-256 diverges from sft/{key}"
            )
    for key in ("dpo_train", "dpo_eval", "grpo_train", "grpo_eval"):
        if joint_manifest["outputs"][key]["sha256"] != pref_manifest["outputs"][key]["sha256"]:
            raise RuntimeError(
                f"joint/{key} SHA-256 diverges from preference/{key}"
            )

    train_rows = (
        _load_jsonl_rows(sft_dir / "sft_train.jsonl")
        + _load_jsonl_rows(pref_dir / "dpo_train.jsonl")
        + _load_jsonl_rows(pref_dir / "grpo_train.jsonl")
    )
    eval_rows = (
        _load_jsonl_rows(sft_dir / "sft_eval.jsonl")
        + _load_jsonl_rows(pref_dir / "dpo_eval.jsonl")
        + _load_jsonl_rows(pref_dir / "grpo_eval.jsonl")
    )
    cross_audit = ds.audit_split_leakage(train_rows, eval_rows)
    if cross_audit["leakage_count"] != 0:
        raise RuntimeError(
            f"cross-pack train/eval prompt leakage detected: "
            f"{cross_audit['leakage_count']}"
        )
    return cross_audit


def _extract_span_coverage(statistics: dict) -> dict:
    """Summarize span-mask coverage from a trainer run_manifest statistics block."""
    total_tokens = 0
    active_tokens = 0
    masked_tokens = 0
    span_counts = {name: 0 for name in trainer.SPAN_TYPES}
    span_token_counts = {name: 0 for name in trainer.SPAN_TYPES}
    weighted_token_mass = {name: 0.0 for name in trainer.SPAN_TYPES}
    by_split = {}

    def _add_block(split_label: str, block: dict) -> None:
        nonlocal total_tokens, active_tokens, masked_tokens
        t_tok = int(block.get("total_tokens", 0))
        a_tok = int(block.get("active_tokens", 0))
        m_tok = int(block.get("masked_tokens", 0))
        by_split[split_label] = {
            "total_tokens": t_tok,
            "active_tokens": a_tok,
            "masked_tokens": m_tok,
            "active_token_ratio": float(block.get("active_token_ratio", 0.0)),
            "span_counts": dict(block.get("span_counts") or {}),
            "span_token_counts": dict(block.get("span_token_counts") or {}),
            "weighted_token_mass": dict(block.get("weighted_token_mass") or {}),
        }
        if split_label.endswith("_train") or "_train:" in split_label:
            total_tokens += t_tok
            active_tokens += a_tok
            masked_tokens += m_tok
            for stype in trainer.SPAN_TYPES:
                span_counts[stype] += int(
                    (block.get("span_counts") or {}).get(stype, 0)
                )
                span_token_counts[stype] += int(
                    (block.get("span_token_counts") or {}).get(stype, 0)
                )
                weighted_token_mass[stype] += float(
                    (block.get("weighted_token_mass") or {}).get(stype, 0.0)
                )

    for split_key in sorted(statistics):
        entry = statistics[split_key]
        if not isinstance(entry, dict):
            continue
        if "chosen_spans" in entry and "rejected_spans" in entry:
            _add_block(f"{split_key}:chosen", entry["chosen_spans"])
            _add_block(f"{split_key}:rejected", entry["rejected_spans"])
        elif "total_tokens" in entry:
            _add_block(split_key, entry)

    return {
        "total_tokens": total_tokens,
        "active_tokens": active_tokens,
        "masked_tokens": masked_tokens,
        "active_token_ratio": (
            round(active_tokens / total_tokens, 6) if total_tokens > 0 else 0.0
        ),
        "span_counts": span_counts,
        "span_token_counts": span_token_counts,
        "weighted_token_mass": {
            k: round(v, 6) for k, v in weighted_token_mass.items()
        },
        "by_split": by_split,
    }


def _summarize_micro_steps(micro_step_verification: dict | None) -> tuple[int, bool, dict]:
    if not micro_step_verification:
        return 0, True, {}

    total_steps = 0
    all_finite = True
    stage_summaries = {}
    stages = micro_step_verification.get("stages") or {}
    for stage_name in sorted(stages):
        stage = stages[stage_name]
        steps_executed = int(stage.get("steps_executed", 0))
        total_steps += steps_executed
        initial_loss = stage.get("initial_loss")
        final_loss = stage.get("final_loss")
        history = stage.get("history") or []

        stage_finite = True
        if steps_executed > 0:
            if not (_is_finite(initial_loss) and _is_finite(final_loss)):
                stage_finite = False
            for item in history:
                if not _is_finite(item.get("loss")):
                    stage_finite = False
        if not stage_finite:
            all_finite = False
            raise RuntimeError(
                f"non-finite micro-step loss detected in stage {stage_name}"
            )

        stage_summaries[stage_name] = {
            "status": stage.get("status"),
            "steps_executed": steps_executed,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_finite": stage_finite,
        }
    return total_steps, all_finite, stage_summaries


def _resolve_trainer_plan(
    *,
    config_path: str | Path | None,
    ablation_matrix_path: str | Path | None,
    ablation_override: str | None,
    training_mode_override: str | None,
) -> tuple[list[dict], str | None]:
    """Resolve list of trainer runs from --config and/or --ablation-matrix."""
    if ablation_matrix_path is not None:
        matrix_path = Path(ablation_matrix_path)
        if not matrix_path.is_file():
            raise FileNotFoundError(f"ablation matrix not found: {matrix_path}")
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        if not isinstance(matrix, dict) \
                or matrix.get("config_type") != "ablation_matrix" \
                or not isinstance(matrix.get("arms"), dict) \
                or not matrix["arms"]:
            raise ValueError(f"invalid ablation matrix JSON: {matrix_path}")
        arms = matrix["arms"]
        if ablation_override is not None:
            if ablation_override not in arms:
                raise ValueError(
                    f"ablation profile {ablation_override!r} not in matrix {matrix_path.name}"
                )
            selected_keys = [ablation_override]
        else:
            selected_keys = sorted(arms.keys())
        plan = []
        for arm_id in selected_keys:
            arm = arms[arm_id]
            profile = arm.get("ablation_profile", arm_id)
            if profile not in trainer.VALID_ABLATION_PROFILES:
                raise ValueError(
                    f"matrix arm {arm_id!r} references unknown ablation profile {profile!r}"
                )
            run_mode = training_mode_override or arm.get("recommended_mode", "joint")
            plan.append({
                "run_id": arm_id,
                "ablation_profile": profile,
                "training_mode": run_mode,
                "config_path": str(config_path) if config_path is not None else None,
            })
        return plan, matrix.get("matrix_name")

    # Single config run
    cfg = trainer.load_training_config(config_path)
    profile = ablation_override or cfg.get("ablation_profile", "joint_full")
    if training_mode_override:
        run_mode = training_mode_override
    elif cfg.get("config_type") == "joint_sft":
        run_mode = "sft"
    else:
        run_mode = "joint"
    return [{
        "run_id": profile,
        "ablation_profile": profile,
        "training_mode": run_mode,
        "config_path": str(config_path) if config_path is not None else None,
    }], None


def run_stage_c_pipeline(
    *,
    output_dir: str | Path,
    trajectory_paths=None,
    capture_live_url: str | None = None,
    artifact_dir: str | Path | None = None,
    strict_artifacts: bool = False,
    seed: str = ds.DEFAULT_SEED,
    eval_ratio: float = ds.DEFAULT_EVAL_RATIO,
    reward_spec: str = DEFAULT_REWARD_SPEC,
    min_margin: float = ds.DEFAULT_MIN_MARGIN,
    min_group_size: int = ds.DEFAULT_MIN_GROUP_SIZE,
    config_path: str | Path | None = None,
    ablation_matrix_path: str | Path | None = None,
    ablation_override: str | None = None,
    training_mode_override: str | None = None,
    execution_mode: str = DEFAULT_EXECUTION_MODE,
    dry_run_steps: int = DEFAULT_DRY_RUN_STEPS,
    live_model: str = DEFAULT_LIVE_MODEL,
    live_timeout_s: float = DEFAULT_LIVE_TIMEOUT_S,
    live_headers: dict[str, str] | None = None,
    allow_remote_http: bool = False,
    live_enable_audio: bool = False,
    live_prompt_suite=LIVE_CAPTURE_PROMPT_SUITE,
    overwrite: bool = False,
) -> dict:
    """Run the complete Stage C serving-export -> dataset -> trainer E2E pipeline."""
    if execution_mode not in trainer.VALID_EXECUTION_MODES:
        raise ValueError(
            f"unsupported execution_mode {execution_mode!r}; "
            f"expected one of {trainer.VALID_EXECUTION_MODES}"
        )
    if execution_mode == "dry_run_micro_step":
        if isinstance(dry_run_steps, bool) or not isinstance(dry_run_steps, int) \
                or dry_run_steps <= 0:
            raise ValueError(
                "dry_run_steps must be a positive integer for dry_run_micro_step mode"
            )
    else:
        dry_run_steps = 0

    has_offline = bool(trajectory_paths)
    has_live = capture_live_url is not None
    if has_offline == has_live:
        raise ValueError(
            "specify exactly one of trajectory_paths (--from-trajectories) or "
            "capture_live_url (--capture-live)"
        )

    # Default to the full ablation matrix when neither --config nor --ablation-matrix is set.
    if config_path is None and ablation_matrix_path is None:
        ablation_matrix_path = DEFAULT_ABLATION_MATRIX

    out_path = _prepare_out_dir(Path(output_dir), overwrite=overwrite)

    # 0. Optional live SSE capture -> verified schema-v2 JSONL + local CAS dir.
    live_summary = None
    effective_artifact_dir = artifact_dir
    if has_live:
        source_mode = "live_capture"
        captured_jsonl, captured_cas, live_summary = capture_live_trajectories(
            capture_live_url,
            out_dir=out_path,
            model=live_model,
            prompt_suite=live_prompt_suite,
            timeout=live_timeout_s,
            extra_headers=live_headers,
            allow_remote_http=allow_remote_http,
            enable_audio=live_enable_audio,
        )
        effective_inputs = [captured_jsonl]
        effective_artifact_dir = captured_cas
        strict_artifacts = True
    else:
        source_mode = "offline_trajectories"
        effective_inputs = list(trajectory_paths)

    # 1. Strict trajectory & CAS validation (fail-closed on any corrupt record).
    resolved_files, input_summary = validate_trajectory_inputs(
        effective_inputs,
        artifact_dir=effective_artifact_dir,
        strict_artifacts=strict_artifacts,
    )

    # 2. Build SFT, Preference (DPO+GRPO), and combined Joint packs + leakage audits.
    sft_dir = out_path / "sft"
    pref_dir = out_path / "preference"
    joint_dir = out_path / "joint"

    sft_manifest = ds.build_dataset(
        resolved_files,
        sft_dir,
        mode="sft",
        artifact_dir=effective_artifact_dir,
        seed=seed,
        eval_ratio=eval_ratio,
        strict_artifacts=strict_artifacts,
        overwrite=overwrite,
    )
    sft_audit = ds.audit_dataset_pack(sft_dir)
    if sft_audit["split_audit"]["leakage_count"] != 0:
        raise RuntimeError("SFT pack failed zero prompt-leakage audit")

    pref_manifest = ds.build_dataset(
        resolved_files,
        pref_dir,
        mode="preference",
        artifact_dir=effective_artifact_dir,
        seed=seed,
        eval_ratio=eval_ratio,
        reward_spec=reward_spec,
        min_margin=min_margin,
        min_group_size=min_group_size,
        strict_artifacts=strict_artifacts,
        overwrite=overwrite,
    )
    pref_audit = ds.audit_dataset_pack(pref_dir)
    if pref_audit["split_audit"]["leakage_count"] != 0:
        raise RuntimeError("Preference pack failed zero prompt-leakage audit")

    joint_manifest = ds.build_dataset(
        resolved_files,
        joint_dir,
        mode="both",
        artifact_dir=effective_artifact_dir,
        seed=seed,
        eval_ratio=eval_ratio,
        reward_spec=reward_spec,
        min_margin=min_margin,
        min_group_size=min_group_size,
        strict_artifacts=strict_artifacts,
        overwrite=overwrite,
    )
    joint_audit = ds.audit_dataset_pack(joint_dir)
    if joint_audit["split_audit"]["leakage_count"] != 0:
        raise RuntimeError("Joint pack failed zero prompt-leakage audit")

    cross_audit = _verify_cross_pack_integrity(
        sft_dir, pref_dir, joint_dir, sft_manifest, pref_manifest, joint_manifest
    )

    # 3. Invoke umminfer_joint_trainer across single config or full ablation matrix.
    trainer_plan, matrix_name = _resolve_trainer_plan(
        config_path=config_path,
        ablation_matrix_path=ablation_matrix_path,
        ablation_override=ablation_override,
        training_mode_override=training_mode_override,
    )
    trainer_root = out_path / "trainer_runs"
    trainer_root.mkdir(parents=True, exist_ok=True)

    trainer_runs = {}
    for spec in trainer_plan:
        run_id = spec["run_id"]
        run_mode = spec["training_mode"]
        if run_mode == "sft":
            bound_dataset_dir = sft_dir
        elif run_mode in ("dpo", "grpo"):
            bound_dataset_dir = pref_dir
        else:
            bound_dataset_dir = joint_dir

        run_out_dir = trainer_root / run_id
        run_manifest = trainer.prepare_training_run(
            mode=run_mode,
            ablation=spec["ablation_profile"],
            dataset_dir=bound_dataset_dir,
            output_dir=run_out_dir,
            config_path=spec["config_path"],
            dry_run_steps=dry_run_steps if execution_mode == "dry_run_micro_step" else 0,
            seed=seed,
            gpu_launch_script=(execution_mode == "gpu_launch_script"),
            overwrite=overwrite,
        )

        run_manifest_path = run_out_dir / "run_manifest.json"
        run_manifest_sha = ds.sha256_file(run_manifest_path)
        sidecar_sha = (
            run_out_dir / "run_manifest.json.sha256"
        ).read_text(encoding="utf-8").strip()
        if run_manifest_sha != sidecar_sha:
            raise RuntimeError(
                f"trainer run_manifest.json.sha256 mismatch for {run_id}"
            )

        step_count, all_losses_finite, stage_metrics = _summarize_micro_steps(
            run_manifest.get("micro_step_verification")
        )
        span_coverage = _extract_span_coverage(run_manifest.get("statistics") or {})

        trainer_runs[run_id] = {
            "ablation_profile": spec["ablation_profile"],
            "training_mode": run_manifest["training_mode"],
            "execution_mode": run_manifest["mode"],
            "active_stages": run_manifest["active_stages"],
            "REAL_TRAINING": run_manifest["REAL_TRAINING"],
            "step_count": step_count,
            "all_losses_finite": all_losses_finite,
            "stage_metrics": stage_metrics,
            "span_mask_coverage": span_coverage,
            "bound_dataset_manifest_sha256": run_manifest["dataset"]["manifest_sha256"],
            "bound_sft_manifest_sha256": sft_audit["manifest_sha256"],
            "bound_preference_manifest_sha256": pref_audit["manifest_sha256"],
            "manifest_path": f"trainer_runs/{run_id}/run_manifest.json",
            "manifest_sha256": run_manifest_sha,
            "launch_script_sha256": run_manifest["artifacts"]["launch_script"]["sha256"],
        }

    # 4. Assemble and atomically write content-addressed stage_c_e2e_report.json.
    sft_train_count = sft_manifest["outputs"]["sft_train"]["record_count"]
    sft_eval_count = sft_manifest["outputs"]["sft_eval"]["record_count"]
    dpo_train_count = pref_manifest["outputs"]["dpo_train"]["record_count"]
    dpo_eval_count = pref_manifest["outputs"]["dpo_eval"]["record_count"]
    grpo_train_count = pref_manifest["outputs"]["grpo_train"]["record_count"]
    grpo_eval_count = pref_manifest["outputs"]["grpo_eval"]["record_count"]

    report = {
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "status": "COMPLETED",
        "source_mode": source_mode,
        "REAL_TRAINING": "NOT_RUN",
        "claims": {
            "REAL_TRAINING": "NOT_RUN",
            "optimizer": (
                "CPU_REFERENCE_MICRO_STEP_ONLY"
                if execution_mode == "dry_run_micro_step"
                else "NOT_RUN"
            ),
            "model_weights_updated": False,
            "real_gpu_training_executed": False,
            "decoupled_from_cpp_serving": True,
        },
        "seed": str(seed),
        "eval_ratio": float(eval_ratio),
        "execution_mode": execution_mode,
        "dry_run_steps": int(dry_run_steps),
        "ablation_matrix_name": matrix_name,
        "input_trajectories": input_summary,
        "sft_pack": {
            "manifest_path": "sft/manifest.json",
            "manifest_sha256": sft_audit["manifest_sha256"],
            "record_count": sft_train_count + sft_eval_count,
            "split_counts": {
                "train": sft_train_count,
                "eval": sft_eval_count,
            },
            "leakage_count": sft_audit["split_audit"]["leakage_count"],
            "audit_passed": sft_audit["status"] == "PASSED",
            "outputs": sft_manifest["outputs"],
        },
        "preference_pack": {
            "manifest_path": "preference/manifest.json",
            "manifest_sha256": pref_audit["manifest_sha256"],
            "dpo_pair_count": dpo_train_count + dpo_eval_count,
            "grpo_group_count": grpo_train_count + grpo_eval_count,
            "split_counts": {
                "dpo_train": dpo_train_count,
                "dpo_eval": dpo_eval_count,
                "grpo_train": grpo_train_count,
                "grpo_eval": grpo_eval_count,
            },
            "reward_spec": pref_manifest["reward"]["spec"],
            "reward_name": pref_manifest["reward"]["name"],
            "unscored_records": pref_manifest["counts"]["rejected"]["unscored"],
            "leakage_count": pref_audit["split_audit"]["leakage_count"],
            "audit_passed": pref_audit["status"] == "PASSED",
            "outputs": pref_manifest["outputs"],
        },
        "joint_pack": {
            "manifest_path": "joint/manifest.json",
            "manifest_sha256": joint_audit["manifest_sha256"],
            "leakage_count": joint_audit["split_audit"]["leakage_count"],
            "audit_passed": joint_audit["status"] == "PASSED",
            "cross_pack_sha256_verified": True,
        },
        "leakage_audit": {
            "sft_leakage_count": sft_audit["split_audit"]["leakage_count"],
            "preference_leakage_count": pref_audit["split_audit"]["leakage_count"],
            "joint_leakage_count": joint_audit["split_audit"]["leakage_count"],
            "cross_pack_leakage_count": cross_audit["leakage_count"],
            "total_leakage_count": (
                sft_audit["split_audit"]["leakage_count"]
                + pref_audit["split_audit"]["leakage_count"]
                + joint_audit["split_audit"]["leakage_count"]
                + cross_audit["leakage_count"]
            ),
            "passed": True,
        },
        "trainer_runs": trainer_runs,
        "live_capture": live_summary,
    }

    report_bytes = traj.canonical_json(report) + b"\n"
    report_sha = traj.sha256_bytes(report_bytes)
    _write_bytes_atomic(out_path / "stage_c_e2e_report.json", report_bytes)
    _write_bytes_atomic(
        out_path / "stage_c_e2e_report.json.sha256",
        (report_sha + "\n").encode("utf-8"),
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage C Live-to-Training E2E Pipeline Driver: trajectory validation "
            "-> SFT/DPO/GRPO packs -> leakage audit -> joint trainer / ablation matrix"
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--from-trajectories",
        action="extend",
        nargs="+",
        default=None,
        help="Offline trajectory JSONL file(s) and/or directory path(s)",
    )
    source.add_argument(
        "--capture-live",
        default=None,
        help="Opt-in live serving base URL (e.g. http://127.0.0.1:43301) to capture SSE rollouts",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for dataset packs, trainer runs, and stage_c_e2e_report.json",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Optional explicit CAS artifact directory for offline trajectories",
    )
    parser.add_argument(
        "--strict-artifacts",
        action="store_true",
        help="Require CAS artifact directory on disk for all referenced media",
    )
    parser.add_argument(
        "--seed",
        default=ds.DEFAULT_SEED,
        help="Deterministic seed for split assignment, batch scheduling, and micro-steps",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=ds.DEFAULT_EVAL_RATIO,
        help="Fraction of prompts routed to eval split [0.0, 1.0]",
    )
    parser.add_argument(
        "--reward",
        default=DEFAULT_REWARD_SPEC,
        help="Pluggable module:attr reward spec for DPO/GRPO packing",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=ds.DEFAULT_MIN_MARGIN,
        help="Minimum chosen-rejected reward margin for DPO pairs",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=ds.DEFAULT_MIN_GROUP_SIZE,
        help="Minimum scored rollouts per prompt for GRPO/DPO grouping",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional training JSON config (single run, or base config for --ablation-matrix)",
    )
    parser.add_argument(
        "--ablation-matrix",
        default=None,
        help="Ablation matrix JSON path (defaults to configs/training/ablation_matrix.json if --config is omitted)",
    )
    parser.add_argument(
        "--ablation",
        choices=trainer.VALID_ABLATION_PROFILES,
        default=None,
        help="Optional single ablation profile filter/override",
    )
    parser.add_argument(
        "--training-mode",
        choices=trainer.VALID_TRAINING_MODES,
        default=None,
        help="Optional training objective mode override (sft, dpo, grpo, joint)",
    )
    parser.add_argument(
        "--execution-mode",
        choices=trainer.VALID_EXECUTION_MODES,
        default=DEFAULT_EXECUTION_MODE,
        help="Trainer execution mode: dry_run_micro_step (default), plan, or gpu_launch_script",
    )
    parser.add_argument(
        "--dry-run-steps",
        type=int,
        default=DEFAULT_DRY_RUN_STEPS,
        help="Number of CPU numerical micro-steps per stage in dry_run_micro_step mode",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LIVE_MODEL,
        help="Model name sent to /v1/chat/completions in --capture-live mode",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_LIVE_TIMEOUT_S,
        help="Per-request HTTP timeout in seconds for --capture-live mode",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Optional env var name holding Bearer token for --capture-live mode",
    )
    parser.add_argument(
        "--allow-remote-http",
        action="store_true",
        help="Allow non-loopback plain http:// URLs in --capture-live mode",
    )
    parser.add_argument(
        "--live-audio",
        action="store_true",
        help="Include audio modality in --capture-live requests",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty --out directory",
    )
    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        headers = build_auth_headers(args.api_key_env)
        report = run_stage_c_pipeline(
            output_dir=args.out,
            trajectory_paths=args.from_trajectories,
            capture_live_url=args.capture_live,
            artifact_dir=args.artifact_dir,
            strict_artifacts=args.strict_artifacts,
            seed=args.seed,
            eval_ratio=args.eval_ratio,
            reward_spec=args.reward,
            min_margin=args.min_margin,
            min_group_size=args.min_group_size,
            config_path=args.config,
            ablation_matrix_path=args.ablation_matrix,
            ablation_override=args.ablation,
            training_mode_override=args.training_mode,
            execution_mode=args.execution_mode,
            dry_run_steps=args.dry_run_steps,
            live_model=args.model,
            live_timeout_s=args.timeout,
            live_headers=headers,
            allow_remote_http=args.allow_remote_http,
            live_enable_audio=args.live_audio,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"stage c pipeline error: {exc}\n")
    sys.stdout.write(traj.canonical_json(report).decode("utf-8") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
