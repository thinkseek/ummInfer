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
import builtins
import copy
import importlib.util
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
from umminfer_sse_replay import LogicalTickClock  # noqa: E402

PIPELINE_SCHEMA_VERSION = 1
DEFAULT_REWARD_SPEC = "umminfer_rl_reward:CompositeUmmReward"
DEFAULT_ABLATION_MATRIX = REPO_ROOT / "configs" / "training" / "ablation_matrix.json"
DEFAULT_DRY_RUN_STEPS = 2
DEFAULT_EXECUTION_MODE = "dry_run_micro_step"
DEFAULT_LIVE_MODEL = "qwen3-native"
DEFAULT_LIVE_TIMEOUT_S = 300.0
_ALLOWED_LATENCY_KEYS = frozenset(("ttft_ms", "ttfa_ms", "terminal_ms", "barge_in_ms"))
_BEHAVIOR_FORMAT = "umminfer-behavior-logprobs-v1"
_BEHAVIOR_KEYS = frozenset(("format", "tokenizer_sha256", "token_ids", "logprobs", "mask"))
_HEX_CHARS = frozenset("0123456789abcdef")

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
    verified_replay_refs = 0
    replay_records = 0
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
            verified_replay_refs += stats.get("checked_replay_refs", 0)
            replay_records += 1 if record.get("sse_replay") is not None else 0

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
        "replay_records": replay_records,
        "verified_replay_refs": verified_replay_refs,
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
    capture_replay: bool = True,
    replay_clock=None,
    chunk_size: int = 65536,
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
    capture = None
    if capture_replay:
        capture = traj.capture_sse_response(
            response, clock=replay_clock, read_size=chunk_size)
        response = io.BytesIO(capture.raw_bytes)
    request_id, events, output_text, terminal_state, error = traj.parse_sse_stream(
        response,
        artifact_dir=recorder.artifact_dir,
        details=details,
        event_observer=event_observer,
        capture=capture,
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
        capture=capture,
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
    chunk_size: int = 65536,
    replay_clock=None,
) -> dict:
    """Convert bytes with explicit logical ticks and replayable raw capture."""
    if not isinstance(sse_payload, (bytes, bytearray)):
        raise TypeError("sse_payload must be bytes")
    return sse_stream_to_v2_record(
        io.BytesIO(bytes(sse_payload)),
        request_body=request_body,
        recorder=recorder,
        session_id=session_id,
        ablation_metadata=ablation_metadata,
        replay_clock=replay_clock or LogicalTickClock(),
        chunk_size=chunk_size,
    )


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_behavior_evidence_block(evidence) -> dict:
    """Strictly validate a raw umminfer-behavior-logprobs-v1 SSE block."""
    if not isinstance(evidence, dict) or set(evidence) != _BEHAVIOR_KEYS:
        raise ValueError(
            "exposed SSE behavior block schema differs from umminfer-behavior-logprobs-v1"
        )
    if evidence.get("format") != _BEHAVIOR_FORMAT:
        raise ValueError(
            f"unsupported SSE behavior format {evidence.get('format')!r}"
        )
    digest = evidence.get("tokenizer_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or (set(digest) - _HEX_CHARS):
        raise ValueError("SSE behavior tokenizer_sha256 must be 64 lowercase hex chars")
    token_ids = evidence.get("token_ids")
    logprobs = evidence.get("logprobs")
    mask = evidence.get("mask")
    if not all(isinstance(arr, list) and arr for arr in (token_ids, logprobs, mask)):
        raise ValueError("SSE behavior token_ids/logprobs/mask must be non-empty lists")
    if not (len(token_ids) == len(logprobs) == len(mask)):
        raise ValueError("SSE behavior token_ids/logprobs/mask lengths differ")
    if not all(_is_int(v) and v >= 0 for v in token_ids):
        raise ValueError("SSE behavior token_ids must be non-negative integers")
    if not all(_is_finite(v) and float(v) <= 0.0 for v in logprobs):
        raise ValueError("SSE behavior logprobs must be finite and non-positive")
    if not all(_is_int(v) and v in (0, 1) for v in mask) or not any(mask):
        raise ValueError(
            "SSE behavior mask must be binary integers with at least one supervised token"
        )
    return copy.deepcopy(evidence)


def _extract_behavior_from_raw_sse(raw_bytes: bytes, header_status: str) -> dict | None:
    """Extract raw final-chunk behavior evidence iff header is exposed; never fabricate."""
    chunks = []
    for raw_frame in raw_bytes.replace(b"\r\n", b"\n").split(b"\n\n"):
        data_lines = []
        for line in raw_frame.split(b"\n"):
            if line.startswith(b"data:"):
                payload = line[5:]
                if payload.startswith(b" "):
                    payload = payload[1:]
                data_lines.append(payload)
        if not data_lines:
            continue
        joined = b"\n".join(data_lines).strip()
        if not joined or joined == b"[DONE]":
            continue
        parsed = json.loads(joined.decode("utf-8"))
        if isinstance(parsed, dict):
            chunks.append(parsed)

    blocks = [chunk["behavior"] for chunk in chunks if "behavior" in chunk]
    if header_status == "exposed":
        if not chunks or len(blocks) != 1 or "behavior" not in chunks[-1]:
            raise ValueError(
                "X-UmmInfer-Behavior-Evidence=exposed requires exactly one behavior "
                "block in the final SSE data chunk"
            )
        return _validate_behavior_evidence_block(blocks[0])
    if header_status == "not_exposed":
        if blocks:
            raise ValueError(
                "X-UmmInfer-Behavior-Evidence=not_exposed response unexpectedly "
                "contained a behavior block"
            )
        return None
    if header_status == "absent":
        if blocks:
            raise ValueError(
                "SSE response contained a behavior block without "
                "X-UmmInfer-Behavior-Evidence header"
            )
        return None
    raise ValueError(f"unsupported behavior evidence header state: {header_status!r}")


def _hash_bundle_tree(bundle_dir: Path) -> tuple[str, str]:
    """Compute path-free canonical SHA-256 digests of alignment.json and all bundle files."""
    root = Path(bundle_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"behavior alignment bundle directory not found: {root}")
    manifest_path = root / "alignment.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"alignment.json not found in bundle directory: {root}")
    manifest_sha = ds.sha256_file(manifest_path)

    entries = []
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError(
                f"behavior alignment bundle contains symlink: {path.relative_to(root)}"
            )
        if not path.is_file():
            continue
        entries.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": ds.sha256_file(path),
        })
    if not entries:
        raise ValueError(f"behavior alignment bundle contains no files: {root}")
    bundle_sha = traj.sha256_bytes(traj.canonical_json(entries))
    return bundle_sha, manifest_sha


def _run_behavior_token_alignment(
    bundle_dir: str | Path,
    *,
    tokenizer_dir: str | Path | None = None,
    reencode: bool = False,
    tokenizer=None,
) -> dict:
    """Run read-only umminfer_behavior_token_alignment.validate_bundle() verification."""
    import umminfer_behavior_token_alignment as align  # noqa: PLC0415

    bundle_path = Path(bundle_dir)
    if tokenizer_dir is not None:
        tok_path = Path(tokenizer_dir)
        if not tok_path.is_dir():
            raise FileNotFoundError(
                f"alignment tokenizer directory not found: {tok_path}"
            )
    else:
        tok_path = None
        if reencode or tokenizer is not None:
            raise ValueError(
                "--alignment-reencode requires --alignment-tokenizer-dir"
            )

    bundle_sha, manifest_sha = _hash_bundle_tree(bundle_path)
    try:
        alignment_report = align.validate_bundle(
            bundle_path,
            tok_path,
            tokenizer=tokenizer,
            reencode=bool(reencode),
        )
    except (
        align.AlignmentError,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
    ) as exc:
        raise ValueError(f"behavior token alignment verification failed: {exc}") from exc

    if alignment_report.get("status") != "PASS":
        raise ValueError(
            f"behavior token alignment returned non-PASS status: "
            f"{alignment_report.get('status')!r}"
        )

    accepted = alignment_report.get("accepted_cases") or []
    excluded = alignment_report.get("excluded_cases") or []
    claims = dict(alignment_report.get("claims") or {})
    claims["new_inference"] = "NOT_RUN"
    claims["optimizer_step"] = "NOT_RUN"

    return {
        "status": alignment_report["status"],
        "schema": alignment_report["schema"],
        "evidence_class": alignment_report["evidence_class"],
        "validation_level": alignment_report["validation_level"],
        "bundle_sha256": bundle_sha,
        "alignment_manifest_sha256": manifest_sha,
        "tokenizer_files": alignment_report["tokenizer_files"],
        "tokenizer_reencode": alignment_report["tokenizer_reencode"],
        "raw_sse_fresh_replay": alignment_report["raw_sse_fresh_replay"],
        "accepted_case_count": len(accepted),
        "excluded_case_count": len(excluded),
        "sampled_token_count": sum(int(c["sampled_token_count"]) for c in accepted),
        "masked_token_count": sum(int(c["masked_token_count"]) for c in accepted),
        "report_sha256": traj.sha256_bytes(traj.canonical_json(alignment_report)),
        "accepted_cases": accepted,
        "excluded_cases": excluded,
        "claims": claims,
    }


def _run_dpo_export_preflight(
    pref_dir: Path,
    pref_manifest: dict,
    pref_audit: dict,
    *,
    tokenizer_dir: str | Path | None = None,
    tokenizer=None,
) -> dict:
    """Validate preference/dpo_{train,eval}.jsonl via umminfer_joint_dpo_live CPU preflight.

    Strictly forbids torch import and fails closed on missing, partial, mixed,
    or tampered token_export metadata (never silently falling back to legacy).
    """
    if "torch" in sys.modules:
        raise RuntimeError("torch must not be imported before DPO export preflight")
    if tokenizer is not None and tokenizer_dir is None:
        raise ValueError(
            "dpo_preflight_tokenizer requires dpo_preflight_tokenizer_dir"
        )
    tok_dir_path = Path(tokenizer_dir) if tokenizer_dir is not None else None
    if tok_dir_path is not None and not tok_dir_path.is_dir():
        raise FileNotFoundError(
            f"DPO preflight tokenizer directory not found: {tok_dir_path}"
        )

    real_import = builtins.__import__
    real_find_spec = importlib.util.find_spec

    def _forbid_torch_spec(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            return None
        return real_find_spec(name, *args, **kwargs)

    def _forbid_torch_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch import is forbidden in Stage C DPO export preflight")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = _forbid_torch_import
    importlib.util.find_spec = _forbid_torch_spec
    try:
        import umminfer_joint_dpo_live as dpo_live  # noqa: PLC0415
        import umminfer_token_export as token_export  # noqa: PLC0415

        dataset = trainer.verify_and_load_dataset_pack(pref_dir)
        train_rows = dataset["packs"].get("dpo_train") or []
        eval_rows = dataset["packs"].get("dpo_eval") or []

        train_records = dpo_live._structural_export_pairs(train_rows)
        eval_records = dpo_live._structural_export_pairs(eval_rows) if eval_rows else {}

        unified_tokenizer_identity = None
        format_versions: set[int] = set()
        pair_summaries = []

        for split_name, rows, record_map in (
            ("train", train_rows, train_records),
            ("eval", eval_rows, eval_records),
        ):
            for row in rows:
                pair_id = row["pair_id"]
                sides = record_map[pair_id]
                side_meta = {}
                for side in ("chosen", "rejected"):
                    record = sides[side]
                    block = record["token_export"]
                    identity = block["tokenizer"]
                    if unified_tokenizer_identity is None:
                        unified_tokenizer_identity = copy.deepcopy(identity)
                    elif traj.canonical_json(identity) != traj.canonical_json(
                        unified_tokenizer_identity
                    ):
                        raise ValueError(
                            "mixed DPO token-export tokenizer identities across splits"
                        )
                    if tok_dir_path is not None:
                        token_export.validate_record_token_export(
                            record,
                            tokenizer=tokenizer,
                            tokenizer_dir=tok_dir_path,
                            expected_tokenizer_id=unified_tokenizer_identity["id"],
                            expected_tokenizer_revision=unified_tokenizer_identity["revision"],
                            expected_tokenization_version=unified_tokenizer_identity[
                                "tokenization_version"
                            ],
                            expected_sft_target=row[side]["assistant_target"],
                        )
                    format_versions.add(int(block["format_version"]))
                    target_stream = block["streams"]["sft_target"]
                    side_meta[side] = {
                        "format": block["format"],
                        "format_version": int(block["format_version"]),
                        "export_sha256": block["export_sha256"],
                        "source_record_sha256": block["source_record_sha256"],
                        "sft_target_token_count": int(target_stream["token_count"]),
                        "sft_target_token_ids_sha256": target_stream["token_ids_sha256"],
                        "sft_mask_weights_sha256": target_stream["sft_mask"][
                            "token_weights_sha256"
                        ],
                    }
                pair_summaries.append({
                    "pair_id": pair_id,
                    "split": split_name,
                    "prompt_sha256": row["prompt_sha256"],
                    "pair_identity_sha256": traj.sha256_text(
                        f"{pair_id}:{row['prompt_sha256']}:"
                        f"{side_meta['chosen']['export_sha256']}:"
                        f"{side_meta['rejected']['export_sha256']}"
                    ),
                    "chosen": side_meta["chosen"],
                    "rejected": side_meta["rejected"],
                })

        if "torch" in sys.modules:
            raise RuntimeError("torch was imported during Stage C DPO export preflight")
    finally:
        builtins.__import__ = real_import
        importlib.util.find_spec = real_find_spec

    summary = {
        "schema": "umminfer-stage-c-dpo-export-preflight-v1",
        "status": "PASS",
        "token_source": "token_export",
        "EXPORT_DPO_OPTIMIZER": "NOT_RUN",
        "REAL_TRAINING": "NOT_RUN",
        "torch_imported": False,
        "validation_level": "STRONG" if tokenizer is not None else "STRUCTURAL",
        "tokenizer_files_verified": tok_dir_path is not None,
        "tokenizer_reencoded": tokenizer is not None,
        "preference_manifest_sha256": pref_audit["manifest_sha256"],
        "dpo_train_sha256": pref_manifest["outputs"]["dpo_train"]["sha256"],
        "dpo_eval_sha256": pref_manifest["outputs"]["dpo_eval"]["sha256"],
        "dpo_train_pair_count": len(train_rows),
        "dpo_eval_pair_count": len(eval_rows),
        "validated_pair_count": len(train_rows) + len(eval_rows),
        "format_versions": sorted(format_versions),
        "tokenizer": unified_tokenizer_identity,
        "pairs": pair_summaries,
        "claims": {
            "EXPORT_DPO_OPTIMIZER": "NOT_RUN",
            "real_dpo_optimizer": "NOT_RUN",
            "model_quality": "NOT_RUN",
            "sampled_online_behavior_ids": "NOT_CLAIMED",
            "legacy_fallback_used": False,
        },
    }
    summary["preflight_sha256"] = traj.sha256_bytes(traj.canonical_json(summary))
    return summary


def _build_live_request_body(
    case: dict,
    rollout: dict,
    *,
    model: str,
    enable_audio: bool = False,
    capture_behavior_logprobs: bool = False,
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
    if capture_behavior_logprobs:
        body["behavior"] = {"logprobs": True}
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
    capture_behavior_logprobs: bool = False,
    behavior_summary_out: dict | None = None,
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
    behavior_cases = []
    completed = 0
    failed = 0

    for case in prompt_suite:
        prompt_id = case["prompt_id"]
        expected_modalities = list(case["expected_visual_modalities"])
        for rollout_idx, rollout in enumerate(case["rollouts"]):
            body = _build_live_request_body(
                case,
                rollout,
                model=model,
                enable_audio=enable_audio,
                capture_behavior_logprobs=capture_behavior_logprobs,
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

            started_ns = time.perf_counter_ns()
            started = started_ns / 1_000_000_000.0
            replay_clock = traj.PerfCounterClock(origin_ns=started_ns)
            timings: dict[str, float | None] = {
                "ttft_ms": None,
                "ttfa_ms": None,
            }
            raw_behavior_header: str | None = None
            got_http_sse = False

            def _observer(event):
                # Raw capture records arrival before deferred normalization;
                # use that observation instead of the later parse time.
                if "source_event_tick" in event:
                    elapsed_ms = event["source_event_tick"] / 1_000_000.0
                else:
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
                    if capture_behavior_logprobs:
                        raw_behavior_header = resp.headers.get(
                            "X-UmmInfer-Behavior-Evidence"
                        )
                    record = sse_stream_to_v2_record(
                        resp,
                        request_body=body,
                        recorder=recorder,
                        session_id=session_id,
                        ablation_metadata=None,
                        event_observer=_observer,
                        replay_clock=replay_clock,
                    )
                    got_http_sse = True
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

            if capture_behavior_logprobs:
                if raw_behavior_header is None or not raw_behavior_header.strip():
                    header_status = "absent"
                else:
                    normalized_hdr = raw_behavior_header.strip().lower()
                    if normalized_hdr not in ("exposed", "not_exposed"):
                        raise ValueError(
                            f"invalid X-UmmInfer-Behavior-Evidence header "
                            f"{raw_behavior_header!r}; expected 'exposed' or 'not_exposed'"
                        )
                    header_status = normalized_hdr

                behavior_block = None
                if got_http_sse:
                    replay_meta = record.get("sse_replay")
                    if not isinstance(replay_meta, dict) or not replay_meta.get("raw_ref"):
                        raise ValueError(
                            "missing raw SSE replay capture for behavior logprobs audit"
                        )
                    raw_sse_bytes = (artifact_dir / replay_meta["raw_ref"]).read_bytes()
                    behavior_block = _extract_behavior_from_raw_sse(
                        raw_sse_bytes, header_status
                    )

                behavior_cases.append({
                    "prompt_id": prompt_id,
                    "rollout_index": rollout_idx,
                    "rollout_label": rollout["rollout_label"],
                    "terminal_state": record["terminal_state"],
                    "behavior_evidence_header": header_status,
                    "behavior": behavior_block,
                    "behavior_sha256": (
                        traj.sha256_bytes(traj.canonical_json(behavior_block))
                        if behavior_block is not None
                        else None
                    ),
                })

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

    if capture_behavior_logprobs and behavior_summary_out is not None:
        behavior_summary_out.clear()
        behavior_summary_out.update({
            "requested": True,
            "exposed_count": sum(
                1 for c in behavior_cases if c["behavior_evidence_header"] == "exposed"
            ),
            "not_exposed_count": sum(
                1
                for c in behavior_cases
                if c["behavior_evidence_header"] == "not_exposed"
            ),
            "absent_count": sum(
                1 for c in behavior_cases if c["behavior_evidence_header"] == "absent"
            ),
            "total_requests": len(behavior_cases),
            "cases": behavior_cases,
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
    token_export_file: str | Path | None = None,
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
    capture_behavior_logprobs: bool = False,
    verify_behavior_alignment: str | Path | None = None,
    alignment_tokenizer_dir: str | Path | None = None,
    alignment_reencode: bool = False,
    alignment_tokenizer=None,
    dpo_export_preflight: bool = False,
    dpo_preflight_tokenizer_dir: str | Path | None = None,
    dpo_preflight_tokenizer=None,
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
    has_token_export_file = token_export_file is not None
    has_live = capture_live_url is not None
    source_count = int(has_offline) + int(has_token_export_file) + int(has_live)

    if capture_behavior_logprobs and not has_live:
        raise ValueError("--capture-behavior-logprobs requires --capture-live")
    if (
        alignment_tokenizer_dir is not None
        or alignment_reencode
        or alignment_tokenizer is not None
    ) and verify_behavior_alignment is None:
        raise ValueError(
            "--alignment-tokenizer-dir and --alignment-reencode require "
            "--verify-behavior-alignment"
        )
    if alignment_reencode and alignment_tokenizer_dir is None:
        raise ValueError("--alignment-reencode requires --alignment-tokenizer-dir")
    if (
        dpo_preflight_tokenizer_dir is not None
        or dpo_preflight_tokenizer is not None
    ) and not dpo_export_preflight:
        raise ValueError(
            "--dpo-preflight-tokenizer-dir requires --dpo-export-preflight"
        )

    if source_count > 1:
        raise ValueError(
            "specify at most one of trajectory_paths (--from-trajectories), "
            "token_export_file (--token-export-file), or capture_live_url (--capture-live)"
        )
    if source_count == 0:
        if verify_behavior_alignment is None:
            raise ValueError(
                "specify exactly one of trajectory_paths (--from-trajectories) or "
                "capture_live_url (--capture-live)"
            )
        if dpo_export_preflight:
            raise ValueError(
                "--dpo-export-preflight requires trajectory inputs "
                "(--from-trajectories or --token-export-file)"
            )
        alignment_summary = _run_behavior_token_alignment(
            verify_behavior_alignment,
            tokenizer_dir=alignment_tokenizer_dir,
            reencode=alignment_reencode,
            tokenizer=alignment_tokenizer,
        )
        out_path = _prepare_out_dir(Path(output_dir), overwrite=overwrite)
        report = {
            "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
            "status": "COMPLETED",
            "source_mode": "behavior_alignment_only",
            "REAL_TRAINING": "NOT_RUN",
            "claims": {
                "REAL_TRAINING": "NOT_RUN",
                "optimizer": "NOT_RUN",
                "model_weights_updated": False,
                "real_gpu_training_executed": False,
                "decoupled_from_cpp_serving": True,
            },
            "behavior_token_alignment": alignment_summary,
        }
        report_bytes = traj.canonical_json(report) + b"\n"
        report_sha = traj.sha256_bytes(report_bytes)
        _write_bytes_atomic(out_path / "stage_c_e2e_report.json", report_bytes)
        _write_bytes_atomic(
            out_path / "stage_c_e2e_report.json.sha256",
            (report_sha + "\n").encode("utf-8"),
        )
        return report

    # Optional read-only behavior-token alignment verification (fail-closed before writes).
    alignment_summary = None
    if verify_behavior_alignment is not None:
        alignment_summary = _run_behavior_token_alignment(
            verify_behavior_alignment,
            tokenizer_dir=alignment_tokenizer_dir,
            reencode=alignment_reencode,
            tokenizer=alignment_tokenizer,
        )

    # Default to the full ablation matrix when neither --config nor --ablation-matrix is set.
    if config_path is None and ablation_matrix_path is None:
        ablation_matrix_path = DEFAULT_ABLATION_MATRIX

    out_path = _prepare_out_dir(Path(output_dir), overwrite=overwrite)

    # 0. Optional live SSE capture -> verified schema-v2 JSONL + local CAS dir.
    live_summary = None
    behavior_capture_summary: dict | None = {} if capture_behavior_logprobs else None
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
            capture_behavior_logprobs=capture_behavior_logprobs,
            behavior_summary_out=behavior_capture_summary,
        )
        effective_inputs = [captured_jsonl]
        effective_artifact_dir = captured_cas
        strict_artifacts = True
    elif has_token_export_file:
        source_mode = "offline_trajectories"
        effective_inputs = [token_export_file]
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

    # Optional exported-token DPO CPU preflight (fail-closed; never falls back to legacy).
    dpo_preflight_summary = None
    if dpo_export_preflight:
        dpo_preflight_summary = _run_dpo_export_preflight(
            pref_dir,
            pref_manifest,
            pref_audit,
            tokenizer_dir=dpo_preflight_tokenizer_dir,
            tokenizer=dpo_preflight_tokenizer,
        )

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
    if behavior_capture_summary is not None:
        report["behavior_logprobs_capture"] = behavior_capture_summary
    if alignment_summary is not None:
        report["behavior_token_alignment"] = alignment_summary
    if dpo_preflight_summary is not None:
        report["dpo_export_preflight"] = dpo_preflight_summary

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
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument(
        "--from-trajectories",
        action="extend",
        nargs="+",
        default=None,
        help="Offline trajectory JSONL file(s) and/or directory path(s)",
    )
    source.add_argument(
        "--token-export-file",
        default=None,
        help="Offline trajectory JSONL carrying additive token_export blocks",
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
        "--capture-behavior-logprobs",
        action="store_true",
        help="Opt-in (--capture-live only): send behavior.logprobs=true and record raw exposed SSE behavior blocks",
    )
    parser.add_argument(
        "--verify-behavior-alignment",
        default=None,
        help="Opt-in: validate a pinned behavior-token alignment bundle directory read-only",
    )
    parser.add_argument(
        "--alignment-tokenizer-dir",
        default=None,
        help="Optional local tokenizer directory for --verify-behavior-alignment",
    )
    parser.add_argument(
        "--alignment-reencode",
        action="store_true",
        help="Opt-in (--verify-behavior-alignment + --alignment-tokenizer-dir): strongly re-encode token_export",
    )
    parser.add_argument(
        "--dpo-export-preflight",
        action="store_true",
        help="Opt-in: run CPU-only exported-token DPO preflight over preference/dpo_{train,eval}.jsonl (no torch)",
    )
    parser.add_argument(
        "--dpo-preflight-tokenizer-dir",
        default=None,
        help="Optional local tokenizer directory to verify available_files during --dpo-export-preflight",
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
            token_export_file=args.token_export_file,
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
            capture_behavior_logprobs=args.capture_behavior_logprobs,
            verify_behavior_alignment=args.verify_behavior_alignment,
            alignment_tokenizer_dir=args.alignment_tokenizer_dir,
            alignment_reencode=args.alignment_reencode,
            dpo_export_preflight=args.dpo_export_preflight,
            dpo_preflight_tokenizer_dir=args.dpo_preflight_tokenizer_dir,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"stage c pipeline error: {exc}\n")
    sys.stdout.write(traj.canonical_json(report).decode("utf-8") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
