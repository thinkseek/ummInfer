#!/usr/bin/env python3
"""North-Star Native Multimodal Showcase & Verification Suite for ummInfer.

Runs all five native multimodal showcase journeys against either an in-process
deterministic loopback mock server (--self-check) or a live ummInfer server
(--url):

  1. Interleaved Narrative + Global Streaming Voice
     POST /v1/chat/completions (stream=true, narrative.enabled=true,
     modalities=[text,image,video,audio], audio.stream=true)
  2. In-Place Image-to-Video Animation
     POST /v1/visual/render (kind=video, image=data:image/png;base64,...)
  3. 3-Turn OpenAI Multi-Turn Native Video Understanding (Prefix KV-Cache)
     POST /v1/chat/completions (video_url anchored on Turn 0, 3 turns)
  4. Autonomous Modality Routing Contrast (visual_mode=auto)
     Pure-text reasoning prompt vs rich visual scene prompt under identical
     narrative settings.
  5. Native Cross-Slot Storyboard I2V Chain (continuity_mode=i2v_chain)
     POST /v1/chat/completions (stream=true, narrative.continuity_mode=i2v_chain,
     style_anchor, video.teacache_thresh=0.05, modalities=[text,audio,image,video])

Usage:
  python3 -I tools/umminfer_showcase.py --self-check
  python3 -I tools/umminfer_showcase.py --url http://127.0.0.1:43301 \\
      --model qwen3-native --out-dir /tmp/umminfer-showcase --journey all

Pure Python 3 standard library only (optional ffprobe on PATH for live MP4
metadata inspection).
"""

from __future__ import annotations

import argparse
import base64
import binascii
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
import zlib


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "benchmarks" / "showcase" / "prompts.json"

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
MARKER_RE = re.compile(r"\[\[\s*(?:image|video|visual)\s*:", re.IGNORECASE)
TERMINAL_KINDS = {"complete", "error", "cancelled"}
ALL_JOURNEYS = (1, 2, 3, 4, 5)
VIDEO_FRAME_POLICIES = ("uniform", "temporal")
NATIVE_ALLOCATOR_SCOPE = "request-native-allocator"
TELEMETRY_STAGE_COVERAGES = ("COVERED", "NOT_COVERED")
TELEMETRY_ENVELOPE_COVERAGES = ("COVERED", "PARTIAL", "NOT_COVERED")
BEHAVIOR_LOGPROBS_FORMAT = "umminfer-behavior-logprobs-v1"
BEHAVIOR_EVIDENCE_HEADER = "X-UmmInfer-Behavior-Evidence"
BEHAVIOR_EVIDENCE_STATUSES = ("exposed", "not_exposed", "absent")


def require(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def build_video_frame_selection(
    policy: str | None,
    *,
    motion_threshold: float | None = None,
    scene_threshold: float | None = None,
) -> dict | None:
    """Validate CLI video frame selection flags (parity with src/client/cli.cpp)."""
    has_motion = motion_threshold is not None
    has_scene = scene_threshold is not None
    if policy is None:
        if has_motion or has_scene:
            raise ValueError("video thresholds require --video-frame-policy temporal")
        return None
    if not isinstance(policy, str) or policy not in VIDEO_FRAME_POLICIES:
        raise ValueError("--video-frame-policy must be uniform or temporal")
    if (has_motion or has_scene) and policy != "temporal":
        raise ValueError("video thresholds require --video-frame-policy temporal")

    selection: dict = {"policy": policy}
    for flag_name, key, raw_val in (
        ("--video-motion-threshold", "motion_threshold", motion_threshold),
        ("--video-scene-threshold", "scene_threshold", scene_threshold),
    ):
        if raw_val is None:
            continue
        if isinstance(raw_val, bool) or not isinstance(raw_val, (int, float)):
            raise ValueError(f"invalid {flag_name}: must be a finite number in [0.0, 1.0]")
        val = float(raw_val)
        if not math.isfinite(val) or val < 0.0 or val > 1.0:
            raise ValueError(
                f"invalid {flag_name}: must be a finite number in [0.0, 1.0], got {raw_val!r}"
            )
        selection[key] = val
    return selection


def validate_video_frame_selection(selection: object) -> dict:
    """Validate a wire video_url.frame_selection object (parity with server schema)."""
    if not isinstance(selection, dict):
        raise ValueError("video_url.frame_selection must be an object")
    policy = selection.get("policy")
    if not isinstance(policy, str) or policy not in VIDEO_FRAME_POLICIES:
        raise ValueError("video_url.frame_selection.policy must be uniform or temporal")
    for key in selection:
        if key not in ("policy", "motion_threshold", "scene_threshold"):
            raise ValueError(f"unsupported video_url.frame_selection field: {key}")
    if policy == "uniform" and len(selection) != 1:
        raise ValueError("uniform video frame_selection does not accept temporal thresholds")
    normalized: dict = {"policy": policy}
    for key in ("motion_threshold", "scene_threshold"):
        if key not in selection:
            continue
        raw_val = selection[key]
        if isinstance(raw_val, bool) or not isinstance(raw_val, (int, float)):
            raise ValueError(f"video_url.frame_selection.{key} must be a number")
        val = float(raw_val)
        if not math.isfinite(val) or val < 0.0 or val > 1.0:
            raise ValueError(f"video_url.frame_selection.{key} must be in [0,1]")
        normalized[key] = val
    return normalized


def format_video_frame_selection(selection: dict) -> str:
    """Format a validated video_frame_selection dict for Markdown summaries."""
    parts = [f"policy={selection['policy']}"]
    if "motion_threshold" in selection:
        parts.append(f"motion_threshold={selection['motion_threshold']:g}")
    if "scene_threshold" in selection:
        parts.append(f"scene_threshold={selection['scene_threshold']:g}")
    return ", ".join(parts)


def format_telemetry_bytes(num_bytes: int) -> str:
    """Format a non-negative integer byte count in B / KiB / MiB / GiB."""
    require(is_int(num_bytes) and num_bytes >= 0, f"invalid telemetry byte count: {num_bytes!r}")
    kib = 1024.0
    mib = 1024.0 * 1024.0
    gib = 1024.0 * 1024.0 * 1024.0
    value = float(num_bytes)
    if value >= gib:
        return f"{value / gib:.1f} GiB"
    if value >= mib:
        return f"{value / mib:.1f} MiB"
    if value >= kib:
        return f"{value / kib:.1f} KiB"
    return f"{num_bytes} B"


def validate_native_allocator_telemetry(
    envelope: object,
    *,
    expected_request_id: str | None = None,
) -> dict:
    """Strictly validate one request-native allocator telemetry envelope (fail-closed)."""
    require(
        isinstance(envelope, dict),
        "native allocator telemetry requested, but response omitted or returned non-object telemetry",
    )
    schema_version = envelope.get("schema_version")
    require(
        is_int(schema_version) and schema_version == 1,
        f"telemetry schema_version must be integer 1, got {schema_version!r}",
    )
    scope = envelope.get("scope")
    require(
        scope == NATIVE_ALLOCATOR_SCOPE,
        f"telemetry scope must be {NATIVE_ALLOCATOR_SCOPE!r}, got {scope!r}",
    )
    envelope_coverage = envelope.get("coverage")
    require(
        envelope_coverage in TELEMETRY_ENVELOPE_COVERAGES,
        f"telemetry coverage invalid: {envelope_coverage!r}",
    )

    env_request_id: str | None = None
    if "request_id" in envelope:
        raw_req_id = envelope.get("request_id")
        require(
            isinstance(raw_req_id, str) and bool(raw_req_id.strip()),
            f"telemetry request_id must be a non-empty string, got {raw_req_id!r}",
        )
        env_request_id = raw_req_id
        if expected_request_id is not None:
            require(
                env_request_id == expected_request_id,
                f"telemetry request_id mismatch: expected={expected_request_id!r} actual={env_request_id!r}",
            )

    if "whole_device_nvml" in envelope:
        whole_device_nvml = envelope.get("whole_device_nvml")
        require(
            isinstance(whole_device_nvml, str) and bool(whole_device_nvml.strip()),
            f"telemetry whole_device_nvml must be a separation marker string, got {whole_device_nvml!r}",
        )
    else:
        whole_device_nvml = "separate_measurement_resources"

    stages = envelope.get("stages")
    require(
        isinstance(stages, list) and len(stages) > 0,
        "telemetry stages must be a non-empty list",
    )

    stage_summaries: list[dict] = []
    reasons: list[str] = []
    seen_reasons: set[str] = set()
    covered_alloc_peaks: list[int] = []
    covered_res_peaks: list[int] = []

    def _add_reason(value: object, context: str) -> str:
        require(
            isinstance(value, str) and bool(value.strip()),
            f"{context} reason must be a non-empty string, got {value!r}",
        )
        if value not in seen_reasons:
            seen_reasons.add(value)
            reasons.append(value)
        return value

    for idx, entry in enumerate(stages):
        require(isinstance(entry, dict), f"telemetry stages[{idx}] must be an object")
        stage_name = entry.get("stage")
        require(
            isinstance(stage_name, str) and bool(stage_name.strip()),
            f"telemetry stages[{idx}].stage must be a non-empty string, got {stage_name!r}",
        )
        stage_cov = entry.get("coverage")
        require(
            stage_cov in TELEMETRY_STAGE_COVERAGES,
            f"telemetry stages[{idx}].coverage invalid: {stage_cov!r}",
        )

        if "request_id" in entry and entry["request_id"] is not None:
            stage_req_id = entry["request_id"]
            require(
                isinstance(stage_req_id, str) and bool(stage_req_id.strip()),
                f"telemetry stages[{idx}].request_id must be a non-empty string",
            )
            if env_request_id is not None:
                require(
                    stage_req_id == env_request_id
                    or stage_req_id.startswith(env_request_id + ":"),
                    f"telemetry stages[{idx}].request_id mismatch: {stage_req_id!r} vs {env_request_id!r}",
                )

        alloc_peak = entry.get("allocated_bytes_peak")
        res_peak = entry.get("reserved_bytes_peak")
        alloc_final = entry.get("allocated_bytes_final")
        res_final = entry.get("reserved_bytes_final")

        stage_reason: str | None = None
        if "reason" in entry and entry["reason"] is not None:
            stage_reason = _add_reason(entry["reason"], f"telemetry stages[{idx}]")

        if stage_cov == "NOT_COVERED":
            require(
                alloc_peak is None and res_peak is None,
                f"NOT_COVERED telemetry stage {stage_name!r} must have null peaks, "
                f"got allocated_bytes_peak={alloc_peak!r} reserved_bytes_peak={res_peak!r}",
            )
            require(
                alloc_final is None and res_final is None,
                f"NOT_COVERED telemetry stage {stage_name!r} must have null final bytes",
            )
        else:
            require(
                is_int(alloc_peak) and alloc_peak >= 0,
                f"COVERED telemetry stage {stage_name!r} allocated_bytes_peak must be a "
                f"non-negative integer (not bool), got {alloc_peak!r}",
            )
            require(
                is_int(res_peak) and res_peak >= 0,
                f"COVERED telemetry stage {stage_name!r} reserved_bytes_peak must be a "
                f"non-negative integer (not bool), got {res_peak!r}",
            )
            if alloc_final is not None:
                require(
                    is_int(alloc_final) and alloc_final >= 0,
                    f"COVERED telemetry stage {stage_name!r} allocated_bytes_final invalid: {alloc_final!r}",
                )
            if res_final is not None:
                require(
                    is_int(res_final) and res_final >= 0,
                    f"COVERED telemetry stage {stage_name!r} reserved_bytes_final invalid: {res_final!r}",
                )
            covered_alloc_peaks.append(alloc_peak)
            covered_res_peaks.append(res_peak)

        summary: dict = {
            "stage": stage_name,
            "coverage": stage_cov,
            "allocated_bytes_peak": alloc_peak,
            "reserved_bytes_peak": res_peak,
        }
        if isinstance(entry.get("native_stage"), str) and entry["native_stage"].strip():
            summary["native_stage"] = entry["native_stage"]
        if isinstance(entry.get("backend"), str) and entry["backend"].strip():
            summary["backend"] = entry["backend"]
        if stage_reason is not None:
            summary["reason"] = stage_reason
        stage_summaries.append(summary)

    if "reason" in envelope and envelope["reason"] is not None:
        _add_reason(envelope["reason"], "telemetry envelope")

    covered_stages = len(covered_alloc_peaks)
    total_stages = len(stage_summaries)
    expected_coverage = (
        "NOT_COVERED"
        if covered_stages == 0
        else ("COVERED" if covered_stages == total_stages else "PARTIAL")
    )
    require(
        envelope_coverage == expected_coverage,
        f"telemetry envelope coverage {envelope_coverage!r} inconsistent with stages "
        f"({covered_stages}/{total_stages} covered -> expected {expected_coverage!r})",
    )

    if "sample_coverage" in envelope and envelope["sample_coverage"] is not None:
        sc = envelope["sample_coverage"]
        require(isinstance(sc, dict), "telemetry sample_coverage must be an object")
        if "stage_count" in sc:
            require(
                is_int(sc["stage_count"]) and sc["stage_count"] == total_stages,
                f"telemetry sample_coverage.stage_count mismatch: {sc['stage_count']!r} != {total_stages}",
            )
        if "covered_stage_count" in sc:
            require(
                is_int(sc["covered_stage_count"]) and sc["covered_stage_count"] == covered_stages,
                f"telemetry sample_coverage.covered_stage_count mismatch: "
                f"{sc['covered_stage_count']!r} != {covered_stages}",
            )

    result = {
        "schema_version": 1,
        "scope": NATIVE_ALLOCATOR_SCOPE,
        "coverage": expected_coverage,
        "covered_stages": covered_stages,
        "total_stages": total_stages,
        "max_allocated_bytes_peak": max(covered_alloc_peaks) if covered_alloc_peaks else None,
        "max_reserved_bytes_peak": max(covered_res_peaks) if covered_res_peaks else None,
        "reasons": reasons,
        "stages": stage_summaries,
        "whole_device_nvml": whole_device_nvml,
    }
    if env_request_id is not None:
        result["request_id"] = env_request_id
    return result


def aggregate_native_allocator_telemetry(summaries: list[dict]) -> dict:
    """Aggregate validated request-native allocator telemetry summaries without mixing NVML."""
    require(bool(summaries), "cannot aggregate empty native allocator telemetry summaries")
    covered_stages = 0
    total_stages = 0
    alloc_peaks: list[int] = []
    res_peaks: list[int] = []
    reasons: list[str] = []
    seen_reasons: set[str] = set()
    all_stages: list[dict] = []
    nvml_markers: set[str] = set()

    for item in summaries:
        covered_stages += item["covered_stages"]
        total_stages += item["total_stages"]
        if item["max_allocated_bytes_peak"] is not None:
            alloc_peaks.append(item["max_allocated_bytes_peak"])
        if item["max_reserved_bytes_peak"] is not None:
            res_peaks.append(item["max_reserved_bytes_peak"])
        for reason in item.get("reasons", []):
            if reason not in seen_reasons:
                seen_reasons.add(reason)
                reasons.append(reason)
        all_stages.extend(item.get("stages", []))
        marker = item.get("whole_device_nvml")
        if isinstance(marker, str) and marker:
            nvml_markers.add(marker)

    coverage = (
        "NOT_COVERED"
        if total_stages == 0 or covered_stages == 0
        else ("COVERED" if covered_stages == total_stages else "PARTIAL")
    )
    return {
        "schema_version": 1,
        "scope": NATIVE_ALLOCATOR_SCOPE,
        "coverage": coverage,
        "requests": len(summaries),
        "covered_stages": covered_stages,
        "total_stages": total_stages,
        "max_allocated_bytes_peak": max(alloc_peaks) if alloc_peaks else None,
        "max_reserved_bytes_peak": max(res_peaks) if res_peaks else None,
        "reasons": reasons,
        "stages": all_stages,
        "whole_device_nvml": (
            next(iter(nvml_markers))
            if len(nvml_markers) == 1
            else "separate_measurement_resources"
        ),
    }


def format_native_allocator_summary(summary: dict) -> str:
    """Format a validated telemetry summary honestly (COVERED peak or N/C; never 0 B for null)."""
    covered = summary["covered_stages"]
    total = summary["total_stages"]
    peak = summary["max_allocated_bytes_peak"]
    reasons = summary.get("reasons") or []
    if covered > 0 and peak is not None:
        base = f"{format_telemetry_bytes(peak)} peak (covered={covered}/{total})"
        if reasons:
            return f"{base} [N/C: {', '.join(reasons)}]"
        return base
    if reasons:
        return f"N/C ({', '.join(reasons)})"
    return "N/C"


def validate_behavior_logprobs_evidence(
    header_value: object,
    raw_behavior_payload: object,
) -> dict:
    """Strictly validate X-UmmInfer-Behavior-Evidence header and behavior payload (fail-closed)."""
    if header_value is None:
        status = "absent"
    else:
        require(
            isinstance(header_value, str),
            f"{BEHAVIOR_EVIDENCE_HEADER} header must be a string or None, got {header_value!r}",
        )
        normalized = header_value.strip().lower()
        if not normalized or normalized == "absent":
            status = "absent"
        elif normalized in ("exposed", "not_exposed"):
            status = normalized
        else:
            raise AssertionError(
                f"invalid {BEHAVIOR_EVIDENCE_HEADER} header value: {header_value!r}; "
                f"expected exposed, not_exposed, or absent"
            )

    if status == "absent":
        require(
            raw_behavior_payload is None,
            f"stray behavior payload received when {BEHAVIOR_EVIDENCE_HEADER} header is absent",
        )
        return {
            "status": "absent",
            "format": None,
            "tokenizer_sha256": None,
            "token_count": None,
            "supervised_count": None,
        }

    if status == "not_exposed":
        require(
            raw_behavior_payload is None,
            f"behavior payload must not be present when {BEHAVIOR_EVIDENCE_HEADER} is not_exposed",
        )
        return {
            "status": "not_exposed",
            "format": None,
            "tokenizer_sha256": None,
            "token_count": None,
            "supervised_count": None,
        }

    require(
        isinstance(raw_behavior_payload, dict),
        f"{BEHAVIOR_EVIDENCE_HEADER} is exposed, but response omitted or returned non-object behavior payload",
    )
    fmt = raw_behavior_payload.get("format")
    require(
        fmt == BEHAVIOR_LOGPROBS_FORMAT,
        f"behavior.format must be {BEHAVIOR_LOGPROBS_FORMAT!r}, got {fmt!r}",
    )
    tokenizer_sha256 = raw_behavior_payload.get("tokenizer_sha256")
    require(
        isinstance(tokenizer_sha256, str) and bool(tokenizer_sha256.strip()),
        f"behavior.tokenizer_sha256 must be a non-empty string, got {tokenizer_sha256!r}",
    )
    token_ids = raw_behavior_payload.get("token_ids")
    require(
        isinstance(token_ids, list) and len(token_ids) > 0,
        "behavior.token_ids must be a non-empty list of integers",
    )
    for idx, tid in enumerate(token_ids):
        require(
            is_int(tid) and tid >= 0,
            f"behavior.token_ids[{idx}] must be a non-negative integer (not bool), got {tid!r}",
        )
    logprobs = raw_behavior_payload.get("logprobs")
    require(
        isinstance(logprobs, list) and len(logprobs) == len(token_ids),
        f"behavior.logprobs length mismatch: expected {len(token_ids)}, got "
        f"{len(logprobs) if isinstance(logprobs, list) else type(logprobs).__name__}",
    )
    for idx, lp in enumerate(logprobs):
        require(
            not isinstance(lp, bool)
            and isinstance(lp, (int, float))
            and math.isfinite(float(lp)),
            f"behavior.logprobs[{idx}] must be a finite number (not bool), got {lp!r}",
        )
    mask = raw_behavior_payload.get("mask")
    require(
        isinstance(mask, list) and len(mask) == len(token_ids),
        f"behavior.mask length mismatch: expected {len(token_ids)}, got "
        f"{len(mask) if isinstance(mask, list) else type(mask).__name__}",
    )
    supervised_count = 0
    for idx, mval in enumerate(mask):
        require(
            is_int(mval) and mval in (0, 1),
            f"behavior.mask[{idx}] must be integer 0 or 1 (not bool), got {mval!r}",
        )
        supervised_count += mval

    return {
        "status": "exposed",
        "format": BEHAVIOR_LOGPROBS_FORMAT,
        "tokenizer_sha256": tokenizer_sha256,
        "token_count": len(token_ids),
        "supervised_count": supervised_count,
    }


def aggregate_behavior_logprobs(summaries: list[dict]) -> dict:
    """Aggregate validated behavior logprobs summaries without fabricating counts."""
    require(bool(summaries), "cannot aggregate empty behavior logprobs summaries")
    requests = 0
    exposed_requests = 0
    not_exposed_requests = 0
    absent_requests = 0
    token_total = 0
    supervised_total = 0
    tokenizers: set[str] = set()

    for item in summaries:
        item_reqs = item.get("requests", 1)
        requests += item_reqs
        if "exposed_requests" in item:
            exposed_requests += item["exposed_requests"]
            not_exposed_requests += item["not_exposed_requests"]
            absent_requests += item["absent_requests"]
        else:
            st = item["status"]
            if st == "exposed":
                exposed_requests += 1
            elif st == "not_exposed":
                not_exposed_requests += 1
            elif st == "absent":
                absent_requests += 1
        if item.get("token_count") is not None:
            token_total += item["token_count"]
        if item.get("supervised_count") is not None:
            supervised_total += item["supervised_count"]
        tok_sha = item.get("tokenizer_sha256")
        if isinstance(tok_sha, str) and tok_sha.strip():
            tokenizers.add(tok_sha)

    if exposed_requests == requests and requests > 0:
        status = "exposed"
    elif exposed_requests > 0:
        status = "partial"
    elif not_exposed_requests > 0:
        status = "not_exposed"
    else:
        status = "absent"

    return {
        "status": status,
        "requests": requests,
        "exposed_requests": exposed_requests,
        "not_exposed_requests": not_exposed_requests,
        "absent_requests": absent_requests,
        "format": BEHAVIOR_LOGPROBS_FORMAT if exposed_requests > 0 else None,
        "tokenizer_sha256": next(iter(tokenizers)) if len(tokenizers) == 1 else None,
        "token_count": token_total if exposed_requests > 0 else None,
        "supervised_count": supervised_total if exposed_requests > 0 else None,
    }


def format_behavior_logprobs_summary(summary: dict) -> str:
    """Format a validated behavior logprobs summary honestly (zero fabrication)."""
    status = summary["status"]
    if status == "exposed":
        tok = summary["token_count"]
        sup = summary["supervised_count"]
        require(
            is_int(tok) and is_int(sup),
            f"exposed behavior summary missing integer counts: {summary!r}",
        )
        return f"Behavior: {tok} tok ({sup} sup)"
    if status == "partial":
        tok = summary["token_count"]
        sup = summary["supervised_count"]
        exp = summary.get("exposed_requests", 0)
        reqs = summary.get("requests", 0)
        require(
            is_int(tok) and is_int(sup),
            f"partial behavior summary missing integer counts: {summary!r}",
        )
        return f"Behavior: {tok} tok ({sup} sup) [exposed={exp}/{reqs}]"
    if status == "not_exposed":
        return "Behavior: not_exposed"
    if status == "absent":
        return "Behavior: absent"
    if status == "not_applicable":
        return "Behavior: N/A"
    raise AssertionError(f"unknown behavior logprobs summary status: {status!r}")


def _extract_stream_behavior_payload(stream: dict) -> object:
    """Extract optional top-level behavior block from the final SSE frame (fail-closed if non-final)."""
    frames = stream.get("frames")
    if frames is None:
        frames = [
            obs["frame"]
            for obs in stream.get("observations", [])
            if isinstance(obs, dict) and isinstance(obs.get("frame"), dict)
        ]
    if not frames:
        return None
    for idx, frame in enumerate(frames[:-1]):
        if isinstance(frame, dict) and "behavior" in frame and frame["behavior"] is not None:
            raise AssertionError(
                f"behavior payload appeared on non-final SSE frame index {idx}"
            )
    final_frame = frames[-1]
    if isinstance(final_frame, dict) and "behavior" in final_frame:
        return final_frame["behavior"]
    return None


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def make_valid_png(
    width: int = 8,
    height: int = 8,
    rgb: tuple[int, int, int] = (232, 122, 48),
) -> bytes:
    """Create a deterministic, valid RGB PNG using stdlib zlib/struct."""
    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = b"\x00" + bytes(rgb) * width
    idat = zlib.compress(row * height, level=9)
    return PNG_MAGIC + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def make_mock_mp4(tag: bytes = b"showcase-video") -> bytes:
    """Create a deterministic minimal ISO BMFF MP4 container header stub."""
    ftyp = struct.pack(">I", 24) + b"ftypmp42" + b"\x00\x00\x00\x00mp42isom"
    free_payload = tag
    free_box = struct.pack(">I", 8 + len(free_payload)) + b"free" + free_payload
    return ftyp + free_box


def make_mock_pcm16(
    sample_rate: int = 24000,
    duration_s: float = 0.20,
    freq_hz: float = 440.0,
) -> bytes:
    """Create deterministic mono 16-bit signed PCM audio."""
    count = int(sample_rate * duration_s)
    samples = [
        int(6000 * math.sin(2.0 * math.pi * freq_hz * i / sample_rate))
        for i in range(count)
    ]
    return struct.pack(f"<{count}h", *samples)


MOCK_PNG_J1 = make_valid_png(8, 8, (235, 138, 42))
MOCK_MP4_J1 = make_mock_mp4(b"j1-narrative-orbit")
MOCK_MP4_J2 = make_mock_mp4(b"j2-i2v-animate-704x480-29f")
MOCK_PNG_J4 = make_valid_png(8, 8, (36, 178, 196))
MOCK_MP4_J4 = make_mock_mp4(b"j4-bioluminescent-current")
MOCK_PNG_J5 = make_valid_png(8, 8, (242, 176, 86))
MOCK_MP4_J5 = make_mock_mp4(b"j5-storyboard-i2v-chain-704x480-29f")
MOCK_PCM_J1 = make_mock_pcm16(sample_rate=24000, duration_s=0.20, freq_hz=440.0)
MOCK_PCM_J5 = make_mock_pcm16(sample_rate=24000, duration_s=0.20, freq_hz=523.25)


def validate_server_url(url: str, allow_remote_http: bool = False) -> str:
    """Validate and normalize server base URL with strict transport hygiene.

    Rejects embedded credentials and refuses non-loopback plain HTTP URLs
    unless allow_remote_http is explicitly True.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Server URL must be a non-empty string")
    parsed = urllib.parse.urlsplit(url.strip())
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme {scheme!r}; expected http or https")
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

    netloc = parsed.netloc
    path = parsed.path.rstrip("/")
    return f"{scheme}://{netloc}{path}"


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(data, dict) and data.get("schema_version") == 1,
            f"Unsupported manifest schema in {path}")
    journeys = data.get("journeys")
    require(isinstance(journeys, dict), "Manifest lacks journeys object")
    for key in (
        "journey_1_interleaved_narrative",
        "journey_2_i2v_animation",
        "journey_3_multiturn_video_qa",
        "journey_4_modality_routing_contrast",
        "journey_5",
    ):
        require(isinstance(journeys.get(key), dict), f"Manifest missing {key}")
    return data


def build_auth_headers(api_key_env: str | None) -> dict[str, str]:
    if not api_key_env:
        return {}
    token = os.environ.get(api_key_env, "").strip()
    if not token:
        raise ValueError(f"Environment variable {api_key_env!r} is unset or empty")
    return {"Authorization": f"Bearer {token}"}


def http_request(
    url: str,
    body: dict | bytes | None = None,
    timeout: float = 30.0,
    extra_headers: dict[str, str] | None = None,
) -> tuple[int, object, bytes]:
    data = (
        body
        if isinstance(body, bytes)
        else (None if body is None else json.dumps(body).encode("utf-8"))
    )
    headers: dict[str, str] = {}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(
        url,
        data=data,
        method="GET" if body is None else "POST",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.headers, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read()


def stream_chat(
    base_url: str,
    body: dict,
    timeout: float = 600.0,
    extra_headers: dict[str, str] | None = None,
) -> dict:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers=headers,
    )
    started = time.perf_counter()
    observations: list[dict] = []
    frames: list[dict] = []
    payloads: list[str] = []
    behavior_header: str | None = None
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise AssertionError(f"chat HTTP {exc.code}: {detail}") from exc

    with resp:
        require(resp.status == 200, f"chat HTTP {resp.status}")
        require(
            resp.headers.get_content_type() == "text/event-stream",
            f"chat response is not text/event-stream: {resp.headers.get_content_type()}",
        )
        behavior_header = resp.headers.get(BEHAVIOR_EVIDENCE_HEADER)
        for raw_line in resp:
            line = raw_line.decode("utf-8").rstrip("\r\n")
            if not line.startswith("data:"):
                continue
            payload = line[5:].lstrip(" ")
            payloads.append(payload)
            if payload == "[DONE]":
                continue
            frame = json.loads(payload)
            if "error" in frame and isinstance(frame["error"], dict):
                raise AssertionError(f"stream error frame: {frame['error']}")
            frames.append(frame)
            unified = frame.get("unified")
            if isinstance(unified, dict):
                observations.append({
                    "time": time.perf_counter() - started,
                    "event": unified,
                    "frame": frame,
                })

    return {
        "observations": observations,
        "frames": frames,
        "payloads": payloads,
        "behavior_evidence_header": behavior_header,
        "total_s": time.perf_counter() - started,
    }


def assemble_artifacts(observations: list[dict]) -> list[dict]:
    """Reassemble streamed base64 artifacts and strictly verify bytes & SHA-256."""
    groups: dict[str, dict] = {}
    for obs in observations:
        event = obs["event"]
        if event.get("kind") != "artifact":
            continue
        artifact = event.get("artifact")
        require(isinstance(artifact, dict), "artifact event lacks an artifact object")
        artifact_id = artifact.get("artifact_id")
        require(
            isinstance(artifact_id, str) and bool(artifact_id),
            "artifact_id must be a non-empty string",
        )
        require(artifact.get("encoding") == "base64", "artifact encoding is not base64")
        index = artifact.get("index")
        total = artifact.get("total")
        require(
            is_int(index) and is_int(total) and total > 0 and 0 <= index < total,
            f"invalid artifact chunk bounds: {artifact_id}",
        )
        group = groups.setdefault(
            artifact_id,
            {
                "chunks": {},
                "next_index": 0,
                "first": obs,
                "last": obs,
                "descriptors": [],
                "total": total,
                "stage": event.get("stage"),
            },
        )
        require(total == group["total"], f"artifact total changed between chunks: {artifact_id}")
        require(event.get("stage") == group["stage"],
                f"artifact stage changed between chunks: {artifact_id}")
        require(index == group["next_index"],
                f"out-of-order artifact chunk: {artifact_id} index={index}")
        group["next_index"] += 1
        group["last"] = obs
        group["descriptors"].append(artifact)
        try:
            group["chunks"][index] = base64.b64decode(artifact["data"], validate=True)
        except Exception as exc:
            raise AssertionError(f"invalid base64 in {artifact_id}: {exc}") from exc

        is_final_chunk = index + 1 == total
        require(
            ("bytes" in artifact) == is_final_chunk
            and ("sha256" in artifact) == is_final_chunk,
            f"bytes/sha256 must appear only on the final chunk: {artifact_id}",
        )
        if event.get("stage") != "tts":
            require(
                isinstance(artifact.get("slot_id"), str)
                and is_int(artifact.get("slot_index")),
                f"visual artifact lacks slot attribution: {artifact_id}",
            )

    assembled: list[dict] = []
    for artifact_id, group in groups.items():
        descriptors = group["descriptors"]
        total = descriptors[0]["total"]
        require(
            sorted(group["chunks"]) == list(range(total)),
            f"missing artifact chunks: {artifact_id}",
        )
        raw = b"".join(group["chunks"][i] for i in range(total))
        final = descriptors[-1]
        require(
            is_int(final.get("bytes")) and final["bytes"] == len(raw),
            f"artifact byte count mismatch: {artifact_id} expected={final.get('bytes')} actual={len(raw)}",
        )
        actual_sha = sha256_bytes(raw)
        require(
            isinstance(final.get("sha256"), str) and final["sha256"] == actual_sha,
            f"artifact SHA-256 mismatch: {artifact_id} expected={final.get('sha256')} actual={actual_sha}",
        )
        for key in ("media_type", "mime_type", "slot_id", "slot_index", "total"):
            values = {desc.get(key) for desc in descriptors}
            require(len(values) == 1, f"artifact {key} changed between chunks: {artifact_id}")
        assembled.append({
            "artifact_id": artifact_id,
            "bytes": raw,
            "descriptor": final,
            "first_time": group["first"]["time"],
            "time": group["last"]["time"],
            "stage": group["last"]["event"].get("stage"),
            "seq": group["last"]["event"]["seq"],
        })
    return assembled


def validate_narrative_stream(
    stream: dict,
    *,
    max_slots: int = 2,
    min_slots: int = 1,
    require_audio: bool = False,
    require_mid_stream_slot: bool = False,
    require_native_allocator_telemetry: bool = False,
    require_behavior_logprobs: bool = False,
) -> tuple[dict, list[dict], list[dict]]:
    """Validate SSE narrative stream invariants, seq monotonicity, clean prose, and SHA-256."""
    observations = stream["observations"]
    require(bool(observations), "no unified events received")

    seqs = [item["event"].get("seq") for item in observations]
    require(seqs == list(range(len(seqs))), f"SSE seq is not strictly monotonic from 0: {seqs}")

    terminals = [
        item for item in observations if item["event"].get("kind") in TERMINAL_KINDS
    ]
    require(
        len(terminals) == 1 and terminals[0] is observations[-1],
        "stream must contain exactly one final terminal event",
    )
    require(
        terminals[0]["event"]["kind"] == "complete",
        f"stream did not complete cleanly: {terminals[0]['event']}",
    )
    require(
        stream["payloads"].count("[DONE]") == 1 and stream["payloads"][-1] == "[DONE]",
        "stream must terminate with exactly one [DONE] marker",
    )

    deltas = [item for item in observations if item["event"].get("kind") == "vlm_delta"]
    require(bool(deltas), "no vlm_delta events received")
    for item in deltas:
        event = item["event"]
        require(isinstance(event.get("text"), str), "vlm_delta text must be a string")
        content = item["frame"]["choices"][0]["delta"].get("content")
        require(content == event["text"], "OpenAI delta content differs from unified text")

    prose = "".join(item["event"]["text"] for item in deltas)
    terminal_text = terminals[0]["event"].get("text", "")
    require(
        not MARKER_RE.search(prose) and not MARKER_RE.search(terminal_text),
        "clean prose contains leaked [[image:...]]/[[video:...]] directive marker",
    )
    require(terminal_text == prose, "terminal text differs from aggregated vlm_delta prose")

    slots = [item for item in observations if item["event"].get("kind") == "visual_slot"]
    require(len(slots) >= min_slots, f"expected at least {min_slots} visual_slot(s), got {len(slots)}")
    require(len(slots) <= max_slots, f"narrative exceeded max_slots={max_slots}: got {len(slots)}")

    for expected_index, item in enumerate(slots):
        event = item["event"]
        require(
            event.get("slot_id") == f"slot-{expected_index}"
            and event.get("slot_index") == expected_index,
            f"visual_slot attribution inconsistent: {event}",
        )
        require(
            event.get("media_type") in ("image", "video")
            and event.get("stage") == event["media_type"],
            f"visual_slot stage/media_type mismatch: {event}",
        )
        require(
            isinstance(event.get("prompt"), str) and bool(event["prompt"].strip()),
            "visual_slot prompt must be non-empty",
        )
        require(
            is_int(event.get("text_offset")) and event["text_offset"] >= 0,
            "visual_slot text_offset must be a non-negative integer",
        )
        if "continuity_mode" in event and event["continuity_mode"] is not None:
            require(
                isinstance(event["continuity_mode"], str)
                and bool(event["continuity_mode"].strip()),
                f"visual_slot continuity_mode must be a non-empty string: {event}",
            )
        if "anchor_slot_index" in event and event["anchor_slot_index"] is not None:
            require(
                is_int(event["anchor_slot_index"]) and event["anchor_slot_index"] >= 0,
                f"visual_slot anchor_slot_index must be a non-negative integer: {event}",
            )
        explicit = event.get("anchor_explicit", False)
        require(isinstance(explicit, bool), f"anchor_explicit must be boolean: {event}")
        if explicit:
            require(
                event.get("continuity_mode") == "i2v_chain"
                and event.get("media_type") == "video"
                and is_int(event.get("anchor_slot_index"))
                and 0 <= event["anchor_slot_index"] < expected_index,
                f"explicit anchor must select an earlier image in i2v_chain: {event}",
            )

    vlm_finish_s = deltas[-1]["time"]
    mid_stream_slots = sum(1 for item in slots if item["time"] < vlm_finish_s)
    if require_mid_stream_slot and slots:
        require(
            slots[0]["time"] < vlm_finish_s,
            "first visual_slot was dispatched only after VLM finish",
        )

    artifacts = assemble_artifacts(observations)
    visual = [
        item for item in artifacts if item["descriptor"].get("media_type") in ("image", "video")
    ]
    audio = [
        item for item in artifacts if item["descriptor"].get("mime_type") == "audio/pcm"
    ]
    audio.sort(key=lambda item: item["descriptor"].get("stream_index", 0))
    require(
        len(artifacts) == len(visual) + len(audio)
        and all(item["descriptor"].get("media_type") == "audio" for item in audio),
        "stream contained unexpected artifact modality",
    )

    if require_audio:
        require(bool(audio), "no streaming audio/pcm artifact received")
    for item in audio:
        desc = item["descriptor"]
        require(
            desc.get("format") == "pcm16" and len(item["bytes"]) % 2 == 0,
            "streaming audio artifact is not valid PCM16",
        )
        require(
            is_int(desc.get("stream_index")) and desc["stream_index"] >= 0,
            "streaming audio lacks non-negative stream_index",
        )
        require(
            is_int(desc.get("sample_rate", 24000))
            and desc.get("sample_rate", 24000) > 0
            and is_int(desc.get("channels", 1))
            and desc.get("channels", 1) > 0,
            "streaming audio lacks valid sample rate/channels",
        )

    first_audio_time = min(item["first_time"] for item in audio) if audio else None
    if require_audio and first_audio_time is not None:
        require(
            first_audio_time < terminals[0]["time"],
            "streaming audio arrived only at terminal event",
        )

    by_slot: dict[str, dict] = {}
    for item in visual:
        desc = item["descriptor"]
        slot_id = desc["slot_id"]
        require(slot_id not in by_slot, f"duplicate visual artifact for {slot_id}")
        matches = [slot for slot in slots if slot["event"]["slot_id"] == slot_id]
        require(len(matches) == 1, f"artifact binds to unknown slot: {slot_id}")
        slot_ev = matches[0]["event"]
        require(
            desc["slot_index"] == slot_ev["slot_index"]
            and desc["media_type"] == slot_ev["media_type"]
            and item["stage"] == slot_ev["stage"],
            f"artifact metadata disagrees with visual_slot {slot_id}",
        )
        require(
            item["time"] >= matches[0]["time"],
            f"artifact predates its visual_slot: {slot_id}",
        )
        if desc["media_type"] == "image":
            require(
                item["bytes"].startswith(PNG_MAGIC),
                f"image artifact {slot_id} lacks PNG magic header",
            )
        else:
            require(
                len(item["bytes"]) >= 12 and item["bytes"][4:8] == b"ftyp",
                f"video artifact {slot_id} lacks MP4 ftyp header",
            )
        by_slot[slot_id] = item
    require(len(by_slot) == len(slots), "not every visual_slot produced an artifact")

    for slot in slots:
        slot_ev = slot["event"]
        desc = by_slot[slot_ev["slot_id"]]["descriptor"]
        anchor = slot_ev.get("anchor_slot_index")
        explicit = slot_ev.get("anchor_explicit", False)
        if anchor is None:
            require("anchor_slot_index" not in desc and "anchor_sha256" not in desc,
                    f"unanchored artifact carries anchor metadata: {slot_ev}")
            continue
        if explicit or "anchor_slot_index" in desc or "anchor_sha256" in desc:
            require(desc.get("anchor_slot_index") == anchor
                    and desc.get("anchor_explicit", False) == explicit,
                    f"visual/artifact anchor mismatch: {slot_ev}")
        digest = desc.get("anchor_sha256")
        if explicit or digest is not None:
            source_id = f"slot-{anchor}"
            require(source_id in by_slot
                    and by_slot[source_id]["descriptor"]["media_type"] == "image",
                    f"resolved anchor is not an earlier image: {slot_ev}")
            require(isinstance(digest, str) and len(digest) == 64
                    and digest == by_slot[source_id]["descriptor"]["sha256"],
                    f"anchor SHA-256 does not match source CAS artifact: {slot_ev}")

    legs = terminals[0]["event"].get("legs")
    require(isinstance(legs, list), "complete event lacks legs[]")
    for slot in slots:
        slot_ev = slot["event"]
        matching = [leg for leg in legs if leg.get("slot_id") == slot_ev["slot_id"]]
        require(
            len(matching) == 1
            and matching[0].get("stage") == slot_ev["stage"]
            and matching[0].get("state") == "completed",
            f"terminal leg incomplete for {slot_ev['slot_id']}: {legs}",
        )
        if "anchor_slot_index" in slot_ev:
            desc = by_slot[slot_ev["slot_id"]]["descriptor"]
            leg = matching[0]
            if slot_ev.get("anchor_explicit", False) \
                    or "anchor_slot_index" in leg or "anchor_sha256" in leg:
                require(leg.get("anchor_slot_index") == slot_ev["anchor_slot_index"]
                        and leg.get("anchor_explicit", False)
                            == slot_ev.get("anchor_explicit", False)
                        and leg.get("anchor_sha256") == desc.get("anchor_sha256"),
                        f"terminal anchor metadata mismatch for {slot_ev['slot_id']}")
    if require_audio:
        require(
            any(leg.get("stage") == "tts" and leg.get("state") == "completed" for leg in legs),
            f"terminal tts leg did not complete: {legs}",
        )

    slot_reports = []
    fallback_slots = 0
    for slot in slots:
        slot_ev = slot["event"]
        item = by_slot[slot_ev["slot_id"]]
        is_fallback = (
            slot_ev.get("text_offset") == 0 and slot["time"] >= vlm_finish_s
        )
        if is_fallback:
            fallback_slots += 1
        slot_reports.append({
            "slot_id": slot_ev["slot_id"],
            "slot_index": slot_ev["slot_index"],
            "media_type": slot_ev["media_type"],
            "prompt": slot_ev["prompt"],
            "text_offset": slot_ev["text_offset"],
            "continuity_mode": slot_ev.get("continuity_mode"),
            "anchor_slot_index": slot_ev.get("anchor_slot_index"),
            "slot_arrival_s": round(slot["time"], 6),
            "artifact_arrival_s": round(item["time"], 6),
            "render_latency_s": round(item["time"] - slot["time"], 6),
            "dispatched_before_vlm_finish": slot["time"] < vlm_finish_s,
            "bytes": len(item["bytes"]),
            "sha256": item["descriptor"]["sha256"],
        })
        if slot_ev.get("anchor_explicit", False):
            slot_reports[-1]["anchor_explicit"] = True
        if "anchor_sha256" in item["descriptor"]:
            slot_reports[-1]["anchor_sha256"] = item["descriptor"]["anchor_sha256"]

    metrics = {
        "event_count": len(observations),
        "total_wall_s": round(stream["total_s"], 6),
        "ttft_s": round(deltas[0]["time"], 6),
        "ttfa_s": round(first_audio_time, 6) if first_audio_time is not None else None,
        "vlm_finish_s": round(vlm_finish_s, 6),
        "first_visual_slot_s": round(slots[0]["time"], 6) if slots else None,
        "mid_stream_slot_before_vlm_finish": bool(slots and slots[0]["time"] < vlm_finish_s),
        "visual_slots_emitted": len(slots),
        "mid_stream_slots": mid_stream_slots,
        "fallback_slots_detected": fallback_slots,
        "clean_prose": True,
        "prose_chars": len(prose),
        "prose": prose,
        "slots": slot_reports,
        "legs": legs,
    }
    if require_native_allocator_telemetry:
        terminal_event = terminals[0]["event"]
        stream_request_id = terminal_event.get("request_id")
        if not isinstance(stream_request_id, str) or not stream_request_id.strip():
            frame_id = terminals[0]["frame"].get("id")
            stream_request_id = (
                frame_id if isinstance(frame_id, str) and frame_id.strip() else None
            )
        metrics["native_allocator_telemetry"] = validate_native_allocator_telemetry(
            terminal_event.get("telemetry"),
            expected_request_id=stream_request_id,
        )
    if require_behavior_logprobs:
        metrics["behavior_logprobs"] = validate_behavior_logprobs_evidence(
            stream.get("behavior_evidence_header"),
            _extract_stream_behavior_payload(stream),
        )
    return metrics, visual, audio


def write_and_verify_wav(
    pcm_bytes: bytes,
    out_path: Path,
    sample_rate: int = 24000,
    channels: int = 1,
) -> dict:
    """Wrap raw PCM16 bytes into a RIFF/WAVE file and strictly verify it."""
    require(len(pcm_bytes) > 0 and len(pcm_bytes) % (2 * channels) == 0,
            "PCM16 payload must be non-empty and frame-aligned")
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)

    wav_raw = out_path.read_bytes()
    require(wav_raw[:4] == b"RIFF" and wav_raw[8:12] == b"WAVE",
            f"Generated WAV lacks RIFF/WAVE header: {out_path}")
    with wave.open(str(out_path), "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        nframes = wf.getnframes()
        require(nch == channels and sw == 2 and sr == sample_rate and nframes > 0,
                f"WAV header verification failed for {out_path}")
    duration_s = round(nframes / float(sample_rate), 6)
    return {
        "filename": out_path.name,
        "wav_bytes": len(wav_raw),
        "wav_sha256": sha256_bytes(wav_raw),
        "pcm_bytes": len(pcm_bytes),
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": nframes,
        "duration_s": duration_s,
    }


def probe_mp4_optional(path: Path, *, allow_probe: bool = True) -> dict:
    """Inspect MP4 metadata with ffprobe when installed and applicable."""
    if not allow_probe:
        return {"probed": False, "reason": "offline_self_check_stub"}
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {"probed": False, "reason": "ffprobe_not_installed"}
    try:
        output = subprocess.check_output(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration,size:stream=codec_type,codec_name,width,height,nb_frames,r_frame_rate",
                "-of",
                "json",
                str(path),
            ],
            text=True,
            timeout=30.0,
        )
        parsed = json.loads(output)
        streams = parsed.get("streams", [])
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        require(bool(video_streams), f"ffprobe found no video stream in {path}")
        primary = video_streams[0]
        return {
            "probed": True,
            "codec_name": primary.get("codec_name"),
            "width": primary.get("width"),
            "height": primary.get("height"),
            "nb_frames": primary.get("nb_frames"),
            "r_frame_rate": primary.get("r_frame_rate"),
            "duration": parsed.get("format", {}).get("duration"),
        }
    except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise AssertionError(f"ffprobe failed on {path}: {exc}") from exc


def check_health(base_url: str, extra_headers: dict[str, str] | None = None) -> dict:
    status, _, raw = http_request(
        base_url.rstrip("/") + "/health", timeout=15.0, extra_headers=extra_headers
    )
    require(status == 200, f"/health returned HTTP {status}")
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"/health returned non-JSON: {raw[:200]!r}") from exc
    require(isinstance(body, dict), "/health response must be a JSON object")
    return body


def run_journey_1(
    base_url: str,
    model: str,
    spec: dict,
    out_dir: Path,
    *,
    timeout: float,
    extra_headers: dict[str, str] | None = None,
    native_allocator_telemetry: bool = False,
    behavior_logprobs: bool = False,
) -> tuple[dict, list[dict], dict]:
    """Journey 1: Interleaved Narrative + Global Streaming Voice."""
    body = {
        "model": model,
        "stream": True,
        "max_tokens": spec["max_tokens"],
        "temperature": spec["temperature"],
        "seed": spec["seed"],
        "messages": [{"role": "user", "content": spec["prompt"]}],
        "modalities": list(spec["modalities"]),
        "narrative": copy.deepcopy(spec["narrative"]),
        "image": copy.deepcopy(spec["image"]),
        "video": copy.deepcopy(spec["video"]),
        "audio": copy.deepcopy(spec["audio"]),
    }
    if native_allocator_telemetry:
        body["telemetry"] = {"native_allocator": True}
    if behavior_logprobs:
        body["behavior"] = {"logprobs": True}
    stream = stream_chat(base_url, body, timeout=timeout, extra_headers=extra_headers)
    metrics, visual, audio = validate_narrative_stream(
        stream,
        max_slots=spec["narrative"]["max_slots"],
        min_slots=1,
        require_audio=True,
        require_mid_stream_slot=True,
        require_native_allocator_telemetry=native_allocator_telemetry,
        require_behavior_logprobs=behavior_logprobs,
    )

    saved_artifacts: list[dict] = []
    first_image_info: dict | None = None
    first_video_info: dict | None = None

    slot_by_id = {slot["slot_id"]: slot for slot in metrics["slots"]}
    for item in visual:
        desc = item["descriptor"]
        slot_id = desc["slot_id"]
        ext = "png" if desc["media_type"] == "image" else "mp4"
        filename = f"journey1_{slot_id}.{ext}"
        path = out_dir / filename
        path.write_bytes(item["bytes"])
        slot_by_id[slot_id]["saved_file"] = filename
        record = {
            "filename": filename,
            "journey": 1,
            "media_type": desc["media_type"],
            "bytes": len(item["bytes"]),
            "sha256": desc["sha256"],
        }
        saved_artifacts.append(record)
        if desc["media_type"] == "image" and first_image_info is None:
            first_image_info = {
                "path": path,
                "bytes": item["bytes"],
                "sha256": desc["sha256"],
                "prompt": slot_by_id[slot_id]["prompt"],
                "slot_id": slot_id,
            }
        if desc["media_type"] == "video" and first_video_info is None:
            first_video_info = {
                "path": path,
                "bytes": item["bytes"],
                "sha256": desc["sha256"],
            }

    sample_rates = sorted({item["descriptor"].get("sample_rate", 24000) for item in audio})
    channels_list = sorted({item["descriptor"].get("channels", 1) for item in audio})
    require(len(sample_rates) == 1 and len(channels_list) == 1,
            f"inconsistent streaming PCM format: sr={sample_rates} ch={channels_list}")
    pcm_all = b"".join(item["bytes"] for item in audio)
    wav_path = out_dir / "journey1_audio.wav"
    wav_info = write_and_verify_wav(
        pcm_all, wav_path, sample_rate=sample_rates[0], channels=channels_list[0]
    )
    saved_artifacts.append({
        "filename": wav_info["filename"],
        "journey": 1,
        "media_type": "audio",
        "bytes": wav_info["wav_bytes"],
        "sha256": wav_info["wav_sha256"],
    })

    metrics["audio"] = {
        "artifact_count": len(audio),
        "chunk_count": sum(item["descriptor"]["total"] for item in audio),
        **wav_info,
    }
    metrics["passed"] = True
    handoff = {
        "first_image": first_image_info,
        "first_video": first_video_info,
    }
    return metrics, saved_artifacts, handoff


def run_journey_2(
    base_url: str,
    spec: dict,
    out_dir: Path,
    *,
    source_image_bytes: bytes,
    source_prompt: str | None,
    timeout: float,
    allow_ffprobe: bool,
    extra_headers: dict[str, str] | None = None,
) -> tuple[dict, list[dict], dict]:
    """Journey 2: In-Place Image-to-Video Animation via POST /v1/visual/render."""
    require(
        source_image_bytes.startswith(PNG_MAGIC),
        "Journey 2 source image must be valid PNG bytes",
    )
    source_sha = sha256_bytes(source_image_bytes)
    image_data_url = (
        "data:image/png;base64," + base64.b64encode(source_image_bytes).decode("ascii")
    )
    prompt = (source_prompt or spec["fallback_prompt"]).strip()
    body = {
        "kind": spec.get("kind", "video"),
        "prompt": prompt,
        "image": image_data_url,
        "size": spec["size"],
        "num_frames": spec["num_frames"],
        "fps": spec["fps"],
        "steps": spec["steps"],
        "guidance_scale": spec["guidance_scale"],
        "seed": spec["seed"],
    }
    started = time.perf_counter()
    status, headers, raw = http_request(
        base_url.rstrip("/") + "/v1/visual/render",
        body,
        timeout=timeout,
        extra_headers=extra_headers,
    )
    render_latency_s = time.perf_counter() - started
    require(status == 200, f"/v1/visual/render HTTP {status}: {raw[:500]!r}")
    require(
        headers.get_content_type() == "video/mp4",
        f"/v1/visual/render Content-Type mismatch: {headers.get_content_type()}",
    )
    require(
        headers.get("X-UmmInfer-Kind") == "video",
        f"/v1/visual/render X-UmmInfer-Kind mismatch: {headers.get('X-UmmInfer-Kind')}",
    )
    require(
        len(raw) >= 12 and raw[4:8] == b"ftyp",
        "Journey 2 rendered MP4 lacks ISO BMFF ftyp box signature",
    )

    filename = "journey2_i2v.mp4"
    mp4_path = out_dir / filename
    mp4_path.write_bytes(raw)
    mp4_sha = sha256_bytes(raw)
    probe = probe_mp4_optional(mp4_path, allow_probe=allow_ffprobe)

    metrics = {
        "passed": True,
        "kind": "video",
        "size": spec["size"],
        "num_frames": spec["num_frames"],
        "fps": spec["fps"],
        "steps": spec["steps"],
        "seed": spec["seed"],
        "prompt": prompt,
        "source_image_sha256": source_sha,
        "render_latency_s": round(render_latency_s, 6),
        "bytes": len(raw),
        "sha256": mp4_sha,
        "ftyp_verified": True,
        "ffprobe": probe,
        "saved_file": filename,
    }
    saved_artifacts = [{
        "filename": filename,
        "journey": 2,
        "media_type": "video",
        "bytes": len(raw),
        "sha256": mp4_sha,
    }]
    handoff = {"video_path": mp4_path, "video_bytes": raw, "video_sha256": mp4_sha}
    return metrics, saved_artifacts, handoff


def run_journey_3(
    base_url: str,
    model: str,
    spec: dict,
    *,
    video_bytes: bytes,
    timeout: float,
    extra_headers: dict[str, str] | None = None,
    native_allocator_telemetry: bool = False,
    behavior_logprobs: bool = False,
    video_frame_selection: dict | None = None,
) -> dict:
    """Journey 3: 3-Turn OpenAI Multi-Turn Native Video Understanding (Prefix KV-Cache)."""
    require(
        len(video_bytes) >= 12 and video_bytes[4:8] == b"ftyp",
        "Journey 3 input video must be a valid non-empty MP4 with ftyp header",
    )
    turns_prompts = spec["turns"]
    require(len(turns_prompts) == 3, "Journey 3 requires exactly 3 turns")

    validated_selection = (
        validate_video_frame_selection(video_frame_selection)
        if video_frame_selection is not None
        else None
    )

    video_sha = sha256_bytes(video_bytes)
    video_data_url = (
        "data:video/mp4;base64," + base64.b64encode(video_bytes).decode("ascii")
    )

    messages: list[dict] = []
    turn_results: list[dict] = []
    turn_telemetries: list[dict] = []
    turn_behaviors: list[dict] = []
    for idx, question in enumerate(turns_prompts, start=1):
        if idx == 1:
            video_url_obj: dict = {"url": video_data_url}
            if validated_selection is not None:
                video_url_obj["frame_selection"] = copy.deepcopy(validated_selection)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": video_url_obj},
                    {"type": "text", "text": question},
                ],
            })
        else:
            messages.append({"role": "user", "content": question})

        body = {
            "model": model,
            "stream": True if behavior_logprobs else False,
            "temperature": spec["temperature"],
            "max_tokens": spec["max_tokens"],
            "seed": spec["seed"],
            "messages": copy.deepcopy(messages),
        }
        if native_allocator_telemetry:
            body["telemetry"] = {"native_allocator": True}
        if behavior_logprobs:
            body["behavior"] = {"logprobs": True}
            stream = stream_chat(
                base_url,
                body,
                timeout=timeout,
                extra_headers=extra_headers,
            )
            elapsed_s = stream["total_s"]
            require(
                stream["payloads"].count("[DONE]") == 1 and stream["payloads"][-1] == "[DONE]",
                f"Journey 3 turn {idx} stream must terminate with exactly one [DONE] marker",
            )
            frames = stream.get("frames") or []
            require(bool(frames), f"Journey 3 turn {idx} stream returned no JSON frames")
            answer_parts: list[str] = []
            resp_id: str | None = None
            telemetry_raw: object = None
            for frame in frames:
                if resp_id is None and isinstance(frame.get("id"), str) and frame["id"].strip():
                    resp_id = frame["id"]
                choices = frame.get("choices")
                if isinstance(choices, list) and choices:
                    delta = choices[0].get("delta")
                    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                        answer_parts.append(delta["content"])
                if isinstance(frame.get("telemetry"), dict):
                    telemetry_raw = frame["telemetry"]
                unified = frame.get("unified")
                if isinstance(unified, dict) and isinstance(unified.get("telemetry"), dict):
                    telemetry_raw = unified["telemetry"]
            answer = "".join(answer_parts)
            require(
                isinstance(answer, str) and bool(answer.strip()),
                f"Journey 3 turn {idx} returned empty answer",
            )
            messages.append({"role": "assistant", "content": answer})
            turn_record = {
                "turn": idx,
                "question": question,
                "answer": answer,
                "latency_s": round(elapsed_s, 6),
                "message_count_sent": len(body["messages"]),
            }
            turn_beh = validate_behavior_logprobs_evidence(
                stream.get("behavior_evidence_header"),
                _extract_stream_behavior_payload(stream),
            )
            turn_record["behavior_logprobs"] = turn_beh
            turn_behaviors.append(turn_beh)
            if native_allocator_telemetry:
                turn_tel = validate_native_allocator_telemetry(
                    telemetry_raw,
                    expected_request_id=resp_id,
                )
                turn_record["native_allocator_telemetry"] = turn_tel
                turn_telemetries.append(turn_tel)
            turn_results.append(turn_record)
            continue

        started = time.perf_counter()
        status, _, raw = http_request(
            base_url.rstrip("/") + "/v1/chat/completions",
            body,
            timeout=timeout,
            extra_headers=extra_headers,
        )
        elapsed_s = time.perf_counter() - started
        require(status == 200, f"Journey 3 turn {idx} HTTP {status}: {raw[:500]!r}")
        payload = json.loads(raw)
        choices = payload.get("choices")
        require(
            isinstance(choices, list) and len(choices) >= 1,
            f"Journey 3 turn {idx} missing choices",
        )
        answer = choices[0].get("message", {}).get("content")
        require(
            isinstance(answer, str) and bool(answer.strip()),
            f"Journey 3 turn {idx} returned empty answer",
        )
        messages.append({"role": "assistant", "content": answer})
        turn_record = {
            "turn": idx,
            "question": question,
            "answer": answer,
            "latency_s": round(elapsed_s, 6),
            "message_count_sent": len(body["messages"]),
        }
        if native_allocator_telemetry:
            resp_id = payload.get("id")
            expected_id = resp_id if isinstance(resp_id, str) and resp_id.strip() else None
            turn_tel = validate_native_allocator_telemetry(
                payload.get("telemetry"),
                expected_request_id=expected_id,
            )
            turn_record["native_allocator_telemetry"] = turn_tel
            turn_telemetries.append(turn_tel)
        turn_results.append(turn_record)

    t1 = turn_results[0]["latency_s"]
    t2 = turn_results[1]["latency_s"]
    t3 = turn_results[2]["latency_s"]
    t2_speedup = round(t1 / t2, 4) if t2 > 0 else None
    t3_speedup = round(t1 / t3, 4) if t3 > 0 else None

    result = {
        "passed": True,
        "temperature": spec["temperature"],
        "max_tokens": spec["max_tokens"],
        "video_bytes": len(video_bytes),
        "video_sha256": video_sha,
        "t1_s": t1,
        "t2_s": t2,
        "t3_s": t3,
        "turn2_speedup_vs_turn1": t2_speedup,
        "turn3_speedup_vs_turn1": t3_speedup,
        "turns": turn_results,
    }
    if validated_selection is not None:
        result["video_frame_selection"] = copy.deepcopy(validated_selection)
    if native_allocator_telemetry:
        result["native_allocator_telemetry"] = aggregate_native_allocator_telemetry(
            turn_telemetries
        )
    if behavior_logprobs:
        result["behavior_logprobs"] = aggregate_behavior_logprobs(turn_behaviors)
    return result


def run_journey_4(
    base_url: str,
    model: str,
    spec: dict,
    out_dir: Path,
    *,
    timeout: float,
    strict_zero_text_slots: bool = False,
    extra_headers: dict[str, str] | None = None,
    native_allocator_telemetry: bool = False,
    behavior_logprobs: bool = False,
) -> tuple[dict, list[dict]]:
    """Journey 4: Autonomous Modality Routing Contrast under visual_mode=auto."""
    shared = spec["shared_options"]
    saved_artifacts: list[dict] = []

    def _run_case(case_spec: dict, min_slots: int) -> dict:
        body = {
            "model": model,
            "stream": shared["stream"],
            "max_tokens": shared["max_tokens"],
            "temperature": shared["temperature"],
            "seed": shared["seed"],
            "messages": [{"role": "user", "content": case_spec["prompt"]}],
            "modalities": list(shared["modalities"]),
            "narrative": copy.deepcopy(shared["narrative"]),
            "image": copy.deepcopy(shared["image"]),
            "video": copy.deepcopy(shared["video"]),
        }
        if native_allocator_telemetry:
            body["telemetry"] = {"native_allocator": True}
        if behavior_logprobs:
            body["behavior"] = {"logprobs": True}
        stream = stream_chat(base_url, body, timeout=timeout, extra_headers=extra_headers)
        metrics, visual, _ = validate_narrative_stream(
            stream,
            max_slots=shared["narrative"]["max_slots"],
            min_slots=min_slots,
            require_audio=False,
            require_mid_stream_slot=False,
            require_native_allocator_telemetry=native_allocator_telemetry,
            require_behavior_logprobs=behavior_logprobs,
        )
        slot_by_id = {slot["slot_id"]: slot for slot in metrics["slots"]}
        for item in visual:
            desc = item["descriptor"]
            slot_id = desc["slot_id"]
            ext = "png" if desc["media_type"] == "image" else "mp4"
            filename = f"journey4_{case_spec['id']}_{slot_id}.{ext}"
            (out_dir / filename).write_bytes(item["bytes"])
            slot_by_id[slot_id]["saved_file"] = filename
            saved_artifacts.append({
                "filename": filename,
                "journey": 4,
                "media_type": desc["media_type"],
                "bytes": len(item["bytes"]),
                "sha256": desc["sha256"],
            })
        case_out = {
            "case_id": case_spec["id"],
            "prompt": case_spec["prompt"],
            "visual_slots_emitted": metrics["visual_slots_emitted"],
            "slot_media_types": [s["media_type"] for s in metrics["slots"]],
            "mid_stream_slots": metrics["mid_stream_slots"],
            "fallback_slots_detected": metrics["fallback_slots_detected"],
            "ttft_s": metrics["ttft_s"],
            "total_wall_s": metrics["total_wall_s"],
            "clean_prose": metrics["clean_prose"],
            "prose_chars": metrics["prose_chars"],
            "slots": metrics["slots"],
        }
        if native_allocator_telemetry:
            case_out["native_allocator_telemetry"] = metrics["native_allocator_telemetry"]
        if behavior_logprobs:
            case_out["behavior_logprobs"] = metrics["behavior_logprobs"]
        return case_out

    pure_text = _run_case(spec["pure_text_case"], min_slots=0)
    visual_scene = _run_case(
        spec["visual_scene_case"],
        min_slots=spec["visual_scene_case"].get("expected_min_visual_slots", 1),
    )

    if strict_zero_text_slots:
        require(
            pure_text["visual_slots_emitted"] == 0,
            f"Pure-text prompt emitted {pure_text['visual_slots_emitted']} redundant visual slot(s)",
        )
    require(
        visual_scene["visual_slots_emitted"] >= 1,
        "Visual scene prompt emitted zero visual slots",
    )

    metrics = {
        "passed": True,
        "visual_mode": shared["narrative"]["visual_mode"],
        "max_slots": shared["narrative"]["max_slots"],
        "pure_text": pure_text,
        "visual_scene": visual_scene,
        "pure_text_visual_slots": pure_text["visual_slots_emitted"],
        "visual_scene_visual_slots": visual_scene["visual_slots_emitted"],
        "zero_redundant_visuals_on_pure_text": pure_text["visual_slots_emitted"] == 0,
        "visual_routing_active_on_scene": visual_scene["visual_slots_emitted"] >= 1,
        "contrast_delta_slots": (
            visual_scene["visual_slots_emitted"] - pure_text["visual_slots_emitted"]
        ),
    }
    if native_allocator_telemetry:
        metrics["native_allocator_telemetry"] = aggregate_native_allocator_telemetry([
            pure_text["native_allocator_telemetry"],
            visual_scene["native_allocator_telemetry"],
        ])
    if behavior_logprobs:
        metrics["behavior_logprobs"] = aggregate_behavior_logprobs([
            pure_text["behavior_logprobs"],
            visual_scene["behavior_logprobs"],
        ])
    return metrics, saved_artifacts


def validate_journey_5_stream(
    stream: dict,
    *,
    max_slots: int = 2,
    require_native_allocator_telemetry: bool = False,
    require_behavior_logprobs: bool = False,
) -> tuple[dict, list[dict], list[dict]]:
    """Validate Journey 5 (storyboard_i2v_chain) SSE stream and cross-slot I2V continuity."""
    metrics, visual, audio = validate_narrative_stream(
        stream,
        max_slots=max_slots,
        min_slots=2,
        require_audio=True,
        require_mid_stream_slot=True,
        require_native_allocator_telemetry=require_native_allocator_telemetry,
        require_behavior_logprobs=require_behavior_logprobs,
    )
    slots = metrics["slots"]
    require(len(slots) >= 2, f"Journey 5 requires at least 2 visual slots, got {len(slots)}")

    slot0 = slots[0]
    slot1 = slots[1]
    require(
        slot0["slot_index"] == 0 and slot0["media_type"] == "image",
        f"Journey 5 Slot 0 must be an image keyframe, got {slot0}",
    )
    require(
        slot1["slot_index"] == 1 and slot1["media_type"] == "video",
        f"Journey 5 Slot 1 must be a video continuation, got {slot1}",
    )
    require(
        slot1.get("continuity_mode") == "i2v_chain",
        f"Journey 5 Slot 1 continuity_mode must be 'i2v_chain', got {slot1.get('continuity_mode')!r}",
    )
    require(
        slot1.get("anchor_slot_index") == 0,
        f"Journey 5 Slot 1 anchor_slot_index must be 0, got {slot1.get('anchor_slot_index')!r}",
    )

    visual_by_slot_index = {item["descriptor"]["slot_index"]: item for item in visual}
    img_item = visual_by_slot_index.get(0)
    vid_item = visual_by_slot_index.get(1)
    require(
        img_item is not None and img_item["bytes"].startswith(PNG_MAGIC),
        "Journey 5 Slot 0 missing verified PNG artifact",
    )
    require(
        vid_item is not None and len(vid_item["bytes"]) >= 12 and vid_item["bytes"][4:8] == b"ftyp",
        "Journey 5 Slot 1 missing verified MP4 artifact",
    )

    metrics["continuity_mode"] = "i2v_chain"
    metrics["slot_0_continuity_mode"] = slot0.get("continuity_mode")
    metrics["slot_1_continuity_mode"] = slot1["continuity_mode"]
    metrics["slot_1_anchor_slot_index"] = slot1["anchor_slot_index"]
    metrics["slot_0_png_sha256"] = img_item["descriptor"]["sha256"]
    metrics["slot_1_mp4_sha256"] = vid_item["descriptor"]["sha256"]
    metrics["i2v_chain_verified"] = True
    return metrics, visual, audio


def run_journey_5(
    base_url: str,
    model: str,
    spec: dict,
    out_dir: Path,
    *,
    timeout: float,
    allow_ffprobe: bool,
    extra_headers: dict[str, str] | None = None,
    native_allocator_telemetry: bool = False,
    behavior_logprobs: bool = False,
) -> tuple[dict, list[dict]]:
    """Journey 5: Native Cross-Slot Storyboard I2V Chain (continuity_mode=i2v_chain)."""
    body = {
        "model": model,
        "stream": True,
        "max_tokens": spec["max_tokens"],
        "temperature": spec["temperature"],
        "seed": spec["seed"],
        "messages": [{"role": "user", "content": spec["prompt"]}],
        "modalities": list(spec["modalities"]),
        "narrative": copy.deepcopy(spec["narrative"]),
        "image": copy.deepcopy(spec["image"]),
        "video": copy.deepcopy(spec["video"]),
        "audio": copy.deepcopy(spec["audio"]),
    }
    if native_allocator_telemetry:
        body["telemetry"] = {"native_allocator": True}
    if behavior_logprobs:
        body["behavior"] = {"logprobs": True}
    stream = stream_chat(base_url, body, timeout=timeout, extra_headers=extra_headers)
    metrics, visual, audio = validate_journey_5_stream(
        stream,
        max_slots=spec["narrative"]["max_slots"],
        require_native_allocator_telemetry=native_allocator_telemetry,
        require_behavior_logprobs=behavior_logprobs,
    )

    saved_artifacts: list[dict] = []
    slot_by_id = {slot["slot_id"]: slot for slot in metrics["slots"]}
    mp4_probe: dict | None = None
    for item in visual:
        desc = item["descriptor"]
        slot_id = desc["slot_id"]
        ext = "png" if desc["media_type"] == "image" else "mp4"
        filename = f"journey5_{slot_id}.{ext}"
        path = out_dir / filename
        path.write_bytes(item["bytes"])
        slot_by_id[slot_id]["saved_file"] = filename
        if desc["media_type"] == "video":
            mp4_probe = probe_mp4_optional(path, allow_probe=allow_ffprobe)
        saved_artifacts.append({
            "filename": filename,
            "journey": 5,
            "media_type": desc["media_type"],
            "bytes": len(item["bytes"]),
            "sha256": desc["sha256"],
        })

    sample_rates = sorted({item["descriptor"].get("sample_rate", 24000) for item in audio})
    channels_list = sorted({item["descriptor"].get("channels", 1) for item in audio})
    require(
        len(sample_rates) == 1 and len(channels_list) == 1,
        f"inconsistent Journey 5 streaming PCM format: sr={sample_rates} ch={channels_list}",
    )
    pcm_all = b"".join(item["bytes"] for item in audio)
    wav_path = out_dir / "journey5_audio.wav"
    wav_info = write_and_verify_wav(
        pcm_all, wav_path, sample_rate=sample_rates[0], channels=channels_list[0]
    )
    saved_artifacts.append({
        "filename": wav_info["filename"],
        "journey": 5,
        "media_type": "audio",
        "bytes": wav_info["wav_bytes"],
        "sha256": wav_info["wav_sha256"],
    })

    metrics["journey_id"] = spec.get("id", "storyboard_i2v_chain")
    metrics["style_anchor"] = spec["narrative"].get("style_anchor")
    metrics["video_teacache_thresh"] = spec["video"].get("teacache_thresh")
    metrics["ffprobe"] = mp4_probe
    metrics["audio"] = {
        "artifact_count": len(audio),
        "chunk_count": sum(item["descriptor"]["total"] for item in audio),
        **wav_info,
    }
    metrics["passed"] = True
    return metrics, saved_artifacts


def render_markdown_report(report: dict) -> str:
    """Render a deterministic GitHub-flavored Markdown showcase report."""
    ev = report["evidence_class"]
    rm = report["real_model"]
    lines = [
        "# ummInfer North-Star Native Multimodal Showcase Report",
        "",
        f"- **Suite ID**: `{report['suite_id']}`",
        f"- **Evidence Class**: `{ev}` (`REAL_MODEL: {rm}`)",
        f"- **Model**: `{report['model']}`",
        f"- **Server URL**: `{report['server_url']}`",
        f"- **Journeys Executed**: `{report['journeys_requested']}`",
        f"- **Overall Status**: `{'PASS' if report['summary']['all_passed'] else 'FAIL'}`",
        "",
        "## Executive Summary",
        "",
        "| Journey | Status | Key Metrics |",
        "| --- | --- | --- |",
    ]

    journeys = report["journeys"]
    if "journey_1" in journeys:
        j1 = journeys["journey_1"]
        lines.append(
            f"| 1. Interleaved Narrative + Streaming Voice | PASS | "
            f"TTFT={j1['ttft_s']:.3f}s, TTFA={j1['ttfa_s']:.3f}s, "
            f"slots={j1['visual_slots_emitted']}, "
            f"mid_stream_slot={j1['mid_stream_slot_before_vlm_finish']}, "
            f"clean_prose={j1['clean_prose']} |"
        )
    if "journey_2" in journeys:
        j2 = journeys["journey_2"]
        lines.append(
            f"| 2. In-Place I2V Animation (`/v1/visual/render`) | PASS | "
            f"latency={j2['render_latency_s']:.3f}s, {j2['size']} ({j2['num_frames']}f), "
            f"bytes={j2['bytes']}, ftyp={j2['ftyp_verified']} |"
        )
    if "journey_3" in journeys:
        j3 = journeys["journey_3"]
        fs_suffix = (
            f", frame_selection={format_video_frame_selection(j3['video_frame_selection'])}"
            if "video_frame_selection" in j3
            else ""
        )
        lines.append(
            f"| 3. 3-Turn Multi-Turn Native Video QA | PASS | "
            f"t1={j3['t1_s']:.3f}s, t2={j3['t2_s']:.3f}s ({j3['turn2_speedup_vs_turn1']}x), "
            f"t3={j3['t3_s']:.3f}s ({j3['turn3_speedup_vs_turn1']}x){fs_suffix} |"
        )
    if "journey_4" in journeys:
        j4 = journeys["journey_4"]
        lines.append(
            f"| 4. Autonomous Modality Routing (`visual_mode=auto`) | PASS | "
            f"pure_text_slots={j4['pure_text_visual_slots']}, "
            f"visual_scene_slots={j4['visual_scene_visual_slots']}, "
            f"zero_redundant={j4['zero_redundant_visuals_on_pure_text']} |"
        )
    if "journey_5" in journeys:
        j5 = journeys["journey_5"]
        lines.append(
            f"| 5. Native Storyboard I2V Chain (`storyboard_i2v_chain`) | PASS | "
            f"TTFT={j5['ttft_s']:.3f}s, TTFA={j5['ttfa_s']:.3f}s, "
            f"continuity={j5['slot_1_continuity_mode']}, "
            f"anchor_slot={j5['slot_1_anchor_slot_index']}, "
            f"slots={j5['visual_slots_emitted']} |"
        )

    if "journey_1" in journeys:
        j1 = journeys["journey_1"]
        j1_lines = [
            "",
            "## Journey 1: Interleaved Narrative + Global Streaming Voice",
            "",
            f"- **TTFT**: `{j1['ttft_s']:.6f}s`",
            f"- **TTFA**: `{j1['ttfa_s']:.6f}s`",
            f"- **First Visual Slot Arrival**: `{j1['first_visual_slot_s']:.6f}s` "
            f"(VLM finish: `{j1['vlm_finish_s']:.6f}s`, dispatched mid-stream: `{j1['mid_stream_slot_before_vlm_finish']}`)",
            f"- **Clean Prose**: `{j1['clean_prose']}` ({j1['prose_chars']} chars, zero leaked markers)",
            f"- **Audio WAV**: `{j1['audio']['filename']}` ({j1['audio']['wav_bytes']} bytes, "
            f"{j1['audio']['sample_rate']} Hz, {j1['audio']['channels']}ch, {j1['audio']['duration_s']}s, "
            f"SHA-256 `{j1['audio']['wav_sha256']}`)",
        ]
        if "native_allocator_telemetry" in j1:
            j1_lines.append(
                f"- **Native Allocator Telemetry**: `{format_native_allocator_summary(j1['native_allocator_telemetry'])}`"
            )
        if "behavior_logprobs" in j1:
            j1_lines.append(
                f"- **Behavior Logprobs**: `{format_behavior_logprobs_summary(j1['behavior_logprobs'])}`"
            )
        j1_lines.extend([
            "",
            "| Slot | Media | Slot Arrival (s) | Artifact Arrival (s) | Render Latency (s) | Bytes | SHA-256 |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ])
        lines.extend(j1_lines)
        for slot in j1["slots"]:
            lines.append(
                f"| `{slot['slot_id']}` | `{slot['media_type']}` | "
                f"{slot['slot_arrival_s']:.4f} | {slot['artifact_arrival_s']:.4f} | "
                f"{slot['render_latency_s']:.4f} | {slot['bytes']} | `{slot['sha256']}` |"
            )

    if "journey_2" in journeys:
        j2 = journeys["journey_2"]
        lines.extend([
            "",
            "## Journey 2: In-Place Image-to-Video Animation",
            "",
            f"- **Output File**: `{j2['saved_file']}`",
            f"- **Parameters**: `{j2['size']}`, `{j2['num_frames']}` frames @ `{j2['fps']}` fps, `{j2['steps']}` steps, seed `{j2['seed']}`",
            f"- **Render Latency**: `{j2['render_latency_s']:.6f}s`",
            f"- **MP4 Size / SHA-256**: `{j2['bytes']}` bytes / `{j2['sha256']}`",
            f"- **Source PNG SHA-256**: `{j2['source_image_sha256']}`",
            f"- **MP4 `ftyp` Verified**: `{j2['ftyp_verified']}`",
            f"- **ffprobe**: `{json.dumps(j2['ffprobe'], sort_keys=True)}`",
        ])

    if "journey_3" in journeys:
        j3 = journeys["journey_3"]
        j3_lines = [
            "",
            "## Journey 3: 3-Turn Multi-Turn Native Video Understanding",
            "",
            f"- **Input Video**: `{j3['video_bytes']}` bytes (SHA-256 `{j3['video_sha256']}`)",
            f"- **Turn Latencies**: `t1={j3['t1_s']:.6f}s`, `t2={j3['t2_s']:.6f}s`, `t3={j3['t3_s']:.6f}s`",
            f"- **Prefix KV-Cache Speedups**: Turn 2 = `{j3['turn2_speedup_vs_turn1']}x`, Turn 3 = `{j3['turn3_speedup_vs_turn1']}x`",
        ]
        if "video_frame_selection" in j3:
            j3_lines.append(
                f"- **Video Frame Selection**: `{format_video_frame_selection(j3['video_frame_selection'])}`"
            )
        if "native_allocator_telemetry" in j3:
            j3_lines.append(
                f"- **Native Allocator Telemetry**: `{format_native_allocator_summary(j3['native_allocator_telemetry'])}`"
            )
        if "behavior_logprobs" in j3:
            j3_lines.append(
                f"- **Behavior Logprobs**: `{format_behavior_logprobs_summary(j3['behavior_logprobs'])}`"
            )
        j3_lines.extend([
            "",
            "| Turn | Messages Sent | Latency (s) | Question | Answer |",
            "| ---: | ---: | ---: | --- | --- |",
        ])
        lines.extend(j3_lines)
        for t in j3["turns"]:
            q = t["question"].replace("|", "\\|").replace("\n", " ")
            a = t["answer"].replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {t['turn']} | {t['message_count_sent']} | {t['latency_s']:.4f} | {q} | {a} |"
            )

    if "journey_4" in journeys:
        j4 = journeys["journey_4"]
        lines.extend([
            "",
            "## Journey 4: Autonomous Modality Routing Contrast",
            "",
            "| Case | Visual Slots Emitted | Slot Media | Mid-Stream Slots | Clean Prose | TTFT (s) | Wall (s) |",
            "| --- | ---: | --- | ---: | --- | ---: | ---: |",
            f"| Pure-Text Reasoning (`{j4['pure_text']['case_id']}`) | "
            f"{j4['pure_text']['visual_slots_emitted']} | "
            f"`{j4['pure_text']['slot_media_types']}` | "
            f"{j4['pure_text']['mid_stream_slots']} | "
            f"{j4['pure_text']['clean_prose']} | "
            f"{j4['pure_text']['ttft_s']:.4f} | {j4['pure_text']['total_wall_s']:.4f} |",
            f"| Rich Visual Scene (`{j4['visual_scene']['case_id']}`) | "
            f"{j4['visual_scene']['visual_slots_emitted']} | "
            f"`{j4['visual_scene']['slot_media_types']}` | "
            f"{j4['visual_scene']['mid_stream_slots']} | "
            f"{j4['visual_scene']['clean_prose']} | "
            f"{j4['visual_scene']['ttft_s']:.4f} | {j4['visual_scene']['total_wall_s']:.4f} |",
            "",
            f"- **Zero Redundant Visuals on Pure-Text**: `{j4['zero_redundant_visuals_on_pure_text']}`",
            f"- **Visual Routing Active on Scene**: `{j4['visual_routing_active_on_scene']}`",
            f"- **Contrast Delta Slots**: `{j4['contrast_delta_slots']}`",
        ])
        if "native_allocator_telemetry" in j4:
            lines.append(
                f"- **Native Allocator Telemetry**: `{format_native_allocator_summary(j4['native_allocator_telemetry'])}`"
            )
        if "behavior_logprobs" in j4:
            lines.append(
                f"- **Behavior Logprobs**: `{format_behavior_logprobs_summary(j4['behavior_logprobs'])}`"
            )

    if "journey_5" in journeys:
        j5 = journeys["journey_5"]
        j5_lines = [
            "",
            "## Journey 5: Native Cross-Slot Storyboard I2V Chain",
            "",
            f"- **Journey ID**: `{j5['journey_id']}`",
            f"- **Continuity Mode**: `{j5['slot_1_continuity_mode']}` (Slot 1 `anchor_slot_index={j5['slot_1_anchor_slot_index']}`)",
            f"- **Style Anchor**: `{j5['style_anchor']}`",
            f"- **TTFT**: `{j5['ttft_s']:.6f}s`",
            f"- **TTFA**: `{j5['ttfa_s']:.6f}s`",
            f"- **Audio WAV**: `{j5['audio']['filename']}` ({j5['audio']['wav_bytes']} bytes, "
            f"{j5['audio']['sample_rate']} Hz, {j5['audio']['channels']}ch, {j5['audio']['duration_s']}s, "
            f"SHA-256 `{j5['audio']['wav_sha256']}`)",
        ]
        if "native_allocator_telemetry" in j5:
            j5_lines.append(
                f"- **Native Allocator Telemetry**: `{format_native_allocator_summary(j5['native_allocator_telemetry'])}`"
            )
        if "behavior_logprobs" in j5:
            j5_lines.append(
                f"- **Behavior Logprobs**: `{format_behavior_logprobs_summary(j5['behavior_logprobs'])}`"
            )
        j5_lines.extend([
            "",
            "| Slot | Media | Continuity | Anchor Slot | Bytes | SHA-256 |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ])
        lines.extend(j5_lines)
        for slot in j5["slots"]:
            anchor_val = slot["anchor_slot_index"] if slot["anchor_slot_index"] is not None else "-"
            cont_val = slot["continuity_mode"] or "-"
            lines.append(
                f"| `{slot['slot_id']}` | `{slot['media_type']}` | `{cont_val}` | "
                f"{anchor_val} | {slot['bytes']} | `{slot['sha256']}` |"
            )

    telemetry_rows: list[tuple[str, dict]] = []
    if "journey_1" in journeys and "native_allocator_telemetry" in journeys["journey_1"]:
        telemetry_rows.append(("1. Interleaved Narrative", journeys["journey_1"]["native_allocator_telemetry"]))
    if "journey_3" in journeys and "native_allocator_telemetry" in journeys["journey_3"]:
        telemetry_rows.append(("3. Multi-Turn Video QA (3 turns)", journeys["journey_3"]["native_allocator_telemetry"]))
    if "journey_4" in journeys and "native_allocator_telemetry" in journeys["journey_4"]:
        j4 = journeys["journey_4"]
        if "native_allocator_telemetry" in j4.get("pure_text", {}):
            telemetry_rows.append((
                f"4a. Routing Contrast Pure-Text (`{j4['pure_text']['case_id']}`)",
                j4["pure_text"]["native_allocator_telemetry"],
            ))
        if "native_allocator_telemetry" in j4.get("visual_scene", {}):
            telemetry_rows.append((
                f"4b. Routing Contrast Visual Scene (`{j4['visual_scene']['case_id']}`)",
                j4["visual_scene"]["native_allocator_telemetry"],
            ))
    if "journey_5" in journeys and "native_allocator_telemetry" in journeys["journey_5"]:
        telemetry_rows.append(("5. Storyboard I2V Chain", journeys["journey_5"]["native_allocator_telemetry"]))

    if telemetry_rows or "native_allocator_telemetry" in report:
        top_tel = report.get("native_allocator_telemetry")
        nvml_marker = (
            top_tel.get("whole_device_nvml", "separate_measurement_resources")
            if isinstance(top_tel, dict)
            else "separate_measurement_resources"
        )
        lines.extend([
            "",
            "## Native Allocator Telemetry",
            "",
            f"- **Scope**: `{NATIVE_ALLOCATOR_SCOPE}` (`whole_device_nvml`: `{nvml_marker}`)",
        ])
        if isinstance(top_tel, dict) and top_tel.get("total_stages", 0) > 0:
            lines.append(
                f"- **Overall**: `{format_native_allocator_summary(top_tel)}`"
            )
        if telemetry_rows:
            lines.extend([
                "",
                "| Journey / Case | Coverage | Stages | Peak Allocated | Peak Reserved | Reasons |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ])
            for label, tel in telemetry_rows:
                alloc_peak = tel["max_allocated_bytes_peak"]
                res_peak = tel["max_reserved_bytes_peak"]
                alloc_cell = (
                    f"{format_telemetry_bytes(alloc_peak)} ({alloc_peak} B)"
                    if alloc_peak is not None
                    else "N/C"
                )
                res_cell = (
                    f"{format_telemetry_bytes(res_peak)} ({res_peak} B)"
                    if res_peak is not None
                    else "N/C"
                )
                reasons_cell = ", ".join(tel.get("reasons") or []) or "-"
                lines.append(
                    f"| {label} | `{tel['coverage']}` | "
                    f"{tel['covered_stages']}/{tel['total_stages']} | "
                    f"{alloc_cell} | {res_cell} | {reasons_cell} |"
                )

    behavior_rows: list[tuple[str, dict | None]] = []
    if "journey_1" in journeys and "behavior_logprobs" in journeys["journey_1"]:
        behavior_rows.append(("1. Interleaved Narrative", journeys["journey_1"]["behavior_logprobs"]))
    if "journey_2" in journeys and "behavior_logprobs" in report:
        behavior_rows.append(("2. In-Place I2V Animation (`/v1/visual/render`)", None))
    if "journey_3" in journeys and "behavior_logprobs" in journeys["journey_3"]:
        behavior_rows.append(("3. Multi-Turn Video QA (3 turns)", journeys["journey_3"]["behavior_logprobs"]))
    if "journey_4" in journeys and "behavior_logprobs" in journeys["journey_4"]:
        j4 = journeys["journey_4"]
        if "behavior_logprobs" in j4.get("pure_text", {}):
            behavior_rows.append((
                f"4a. Routing Contrast Pure-Text (`{j4['pure_text']['case_id']}`)",
                j4["pure_text"]["behavior_logprobs"],
            ))
        if "behavior_logprobs" in j4.get("visual_scene", {}):
            behavior_rows.append((
                f"4b. Routing Contrast Visual Scene (`{j4['visual_scene']['case_id']}`)",
                j4["visual_scene"]["behavior_logprobs"],
            ))
    if "journey_5" in journeys and "behavior_logprobs" in journeys["journey_5"]:
        behavior_rows.append(("5. Storyboard I2V Chain", journeys["journey_5"]["behavior_logprobs"]))

    if behavior_rows or "behavior_logprobs" in report:
        top_beh = report.get("behavior_logprobs")
        lines.extend([
            "",
            "## Behavior Logprobs Evidence",
            "",
            f"- **Header**: `{BEHAVIOR_EVIDENCE_HEADER}` (`format`: `{BEHAVIOR_LOGPROBS_FORMAT}`)",
        ])
        if isinstance(top_beh, dict):
            lines.append(
                f"- **Overall**: `{format_behavior_logprobs_summary(top_beh)}`"
            )
        if behavior_rows:
            lines.extend([
                "",
                "| Journey / Case | Status | Summary | Tokenizer SHA-256 |",
                "| --- | --- | --- | --- |",
            ])
            for label, beh in behavior_rows:
                if beh is None:
                    lines.append(
                        f"| {label} | `N/A` | N/A (non-chat endpoint) | - |"
                    )
                    continue
                tok_sha = beh.get("tokenizer_sha256")
                tok_cell = f"`{tok_sha}`" if isinstance(tok_sha, str) and tok_sha else "-"
                lines.append(
                    f"| {label} | `{beh['status']}` | "
                    f"`{format_behavior_logprobs_summary(beh)}` | {tok_cell} |"
                )

    lines.extend([
        "",
        "## Verified Artifacts",
        "",
        "| File | Journey | Media Type | Bytes | SHA-256 |",
        "| --- | ---: | --- | ---: | --- |",
    ])
    for art in report["artifacts"]:
        lines.append(
            f"| `{art['filename']}` | {art['journey']} | `{art['media_type']}` | "
            f"{art['bytes']} | `{art['sha256']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def build_sha256sums(out_dir: Path, filenames: list[str]) -> str:
    """Build deterministic sha256sum-compatible lines sorted by filename."""
    lines = []
    for name in sorted(set(filenames)):
        digest = sha256_bytes((out_dir / name).read_bytes())
        lines.append(f"{digest}  {name}")
    return "\n".join(lines) + "\n"


def write_reports(out_dir: Path, report: dict) -> dict[str, Path]:
    report["artifacts"] = sorted(report["artifacts"], key=lambda item: item["filename"])
    json_path = out_dir / "showcase_report.json"
    md_path = out_dir / "showcase_report.md"
    sums_path = out_dir / "SHA256SUMS"

    json_text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    md_text = render_markdown_report(report)
    json_path.write_text(json_text, encoding="utf-8")
    md_path.write_text(md_text, encoding="utf-8")

    tracked_files = [art["filename"] for art in report["artifacts"]] + [
        json_path.name,
        md_path.name,
    ]
    sums_path.write_text(build_sha256sums(out_dir, tracked_files), encoding="utf-8")
    return {"json": json_path, "markdown": md_path, "sha256sums": sums_path}


# ---------------------------------------------------------------------------
# Offline deterministic loopback fake server (--self-check)
# ---------------------------------------------------------------------------


def _mock_envelope(seq: int, kind: str, text: str = "", finish: str | None = None, **fields) -> dict:
    unified = {
        "seq": seq,
        "kind": kind,
        "request_id": "showcase-mock-1",
        "session_id": "showcase-self-check",
        "stage": "vlm",
        "is_final": kind in TERMINAL_KINDS,
        "text": text,
        **fields,
    }
    delta = {"content": text} if kind == "vlm_delta" else {}
    return {
        "id": "showcase-mock-1",
        "object": "chat.completion.chunk",
        "created": 1764000000,
        "model": "qwen3-native",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        "unified": unified,
    }


def _mock_artifact_frames(
    start_seq: int,
    artifact_id: str,
    media_type: str,
    mime_type: str,
    raw: bytes,
    stage: str,
    *,
    slot_id: str | None = None,
    slot_index: int | None = None,
    stream_index: int | None = None,
    chunks: int = 1,
) -> list[dict]:
    frames = []
    chunk_size = max(1, (len(raw) + chunks - 1) // chunks)
    slices = [raw[i : i + chunk_size] for i in range(0, len(raw), chunk_size)]
    total = len(slices)
    for idx, piece in enumerate(slices):
        art: dict = {
            "artifact_id": artifact_id,
            "media_type": media_type,
            "mime_type": mime_type,
            "encoding": "base64",
            "data": base64.b64encode(piece).decode("ascii"),
            "index": idx,
            "total": total,
        }
        if idx + 1 == total:
            art["bytes"] = len(raw)
            art["sha256"] = sha256_bytes(raw)
        if media_type == "audio":
            art.update({
                "format": "pcm16",
                "sample_rate": 24000,
                "channels": 1,
                "stream_index": stream_index if stream_index is not None else 0,
            })
        else:
            art.update({
                "format": "png" if media_type == "image" else "mp4",
                "size": "512x512" if media_type == "image" else "480x320",
                "seed": 42,
                "slot_id": slot_id,
                "slot_index": slot_index,
            })
        frames.append(
            _mock_envelope(start_seq + idx, "artifact", stage=stage, is_final=False, artifact=art)
        )
    return frames


def _mock_telemetry_stage(
    stage: str,
    *,
    request_id: str,
    coverage: str = "COVERED",
    peak_bytes: int | None = None,
    epoch_id: int = 1,
    reason: str | None = None,
) -> dict:
    native_map = {
        "vlm": ("qwen38", "qwen38"),
        "image": ("wan22", "wan22"),
        "video": ("wan22", "wan22"),
        "tts": ("cosyvoice", "cosyvoice"),
    }
    native_stage, backend = native_map.get(stage, (stage, stage))
    if coverage == "NOT_COVERED":
        return {
            "schema_version": 1,
            "coverage": "NOT_COVERED",
            "scope": "request-native-cuda-runtime",
            "request_id": request_id,
            "stage": stage,
            "native_stage": native_stage,
            "backend": backend,
            "device_index": None,
            "lifecycle_status": "completed",
            "execution": {"mode": "unknown", "batch_capacity": None},
            "epoch": {"id": None, "reset": False, "semantics": "no_request_epoch"},
            "allocated_bytes_peak": None,
            "reserved_bytes_peak": None,
            "allocated_bytes_final": None,
            "reserved_bytes_final": None,
            "sample_coverage": {
                "allocation_events": 0,
                "free_events": 0,
                "terminal_snapshots": 0,
            },
            "reason": reason or "backend_not_instrumented",
        }
    peak = 134217728 if peak_bytes is None else peak_bytes
    return {
        "schema_version": 1,
        "coverage": "COVERED",
        "scope": "request-native-cuda-runtime",
        "request_id": request_id,
        "stage": stage,
        "native_stage": native_stage,
        "backend": backend,
        "device_index": 0,
        "lifecycle_status": "completed",
        "execution": {"mode": "serial", "batch_capacity": 1},
        "epoch": {
            "id": epoch_id,
            "reset": True,
            "semantics": "new_allocation_ownership_epoch",
        },
        "allocated_bytes_peak": peak,
        "reserved_bytes_peak": peak,
        "allocated_bytes_final": 0,
        "reserved_bytes_final": 0,
        "sample_coverage": {
            "allocation_events": 4,
            "free_events": 4,
            "terminal_snapshots": 1,
        },
    }


def _mock_telemetry_envelope(
    request_id: str,
    stages: list[dict],
) -> dict:
    covered = sum(1 for item in stages if item.get("coverage") == "COVERED")
    coverage = (
        "NOT_COVERED"
        if not stages or covered == 0
        else ("COVERED" if covered == len(stages) else "PARTIAL")
    )
    return {
        "schema_version": 1,
        "coverage": coverage,
        "scope": NATIVE_ALLOCATOR_SCOPE,
        "request_id": request_id,
        "lifecycle_status": "completed",
        "sample_coverage": {
            "stage_count": len(stages),
            "covered_stage_count": covered,
            "truncated": False,
        },
        "stages": stages,
        "whole_device_nvml": "separate_measurement_resources",
    }


MOCK_TOKENIZER_SHA256 = (
    "1dd1199e35f6cfb9f547458e02c9fb3a522e9c346ed22ba3d461aeed14310016"
)


def _mock_behavior_payload(
    *,
    token_ids: list[int] | None = None,
    logprobs: list[float] | None = None,
    mask: list[int] | None = None,
    tokenizer_sha256: str = MOCK_TOKENIZER_SHA256,
) -> dict:
    tids = (
        [101, 202, 303, 404, 505, 606, 707, 808]
        if token_ids is None
        else list(token_ids)
    )
    lps = (
        [-0.12, -0.05, -0.34, -0.08, -0.21, -0.03, -0.15, -0.02]
        if logprobs is None
        else list(logprobs)
    )
    msk = [1] * len(tids) if mask is None else list(mask)
    return {
        "format": BEHAVIOR_LOGPROBS_FORMAT,
        "tokenizer_sha256": tokenizer_sha256,
        "token_ids": tids,
        "logprobs": lps,
        "mask": msk,
    }


class _MockShowcaseHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:
        pass

    def _reply(
        self,
        status: int,
        body: bytes,
        content_type: str = "application/json",
        kind: str | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if kind is not None:
            self.send_header("X-UmmInfer-Kind", kind)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: int = 400) -> None:
        self._reply(
            status,
            json.dumps({"error": {"message": message, "type": "invalid_request"}}).encode("utf-8"),
        )

    def do_GET(self) -> None:
        if self.path == "/health":
            self._reply(200, b'{"status":"ok","in_flight":0,"mode":"offline-self-check"}')
        else:
            self._error("not found", 404)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
            if self.path == "/v1/visual/render":
                self._handle_visual_render(raw)
                return
            if self.path == "/v1/chat/completions":
                self._handle_chat(raw)
                return
            self._error("not found", 404)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._error("invalid JSON")
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _handle_visual_render(self, raw: bytes) -> None:
        req = json.loads(raw)
        allowed = {
            "kind", "prompt", "size", "steps", "num_frames", "fps",
            "seconds", "guidance_scale", "seed", "image",
        }
        if not isinstance(req, dict) or set(req) - allowed:
            self._error("unknown or invalid render fields")
            return
        if req.get("kind") != "video":
            self._error("mock showcase render expects kind=video")
            return
        if not isinstance(req.get("prompt"), str) or not req["prompt"].strip():
            self._error("prompt must be non-empty")
            return
        image = req.get("image")
        if (
            not isinstance(image, str)
            or not image.startswith("data:image/")
            or ";base64," not in image
        ):
            self._error("image must be a base64 image data URL")
            return
        try:
            decoded = base64.b64decode(image.split(";base64,", 1)[1], validate=True)
        except (binascii.Error, ValueError):
            self._error("invalid image base64")
            return
        if not decoded.startswith(PNG_MAGIC):
            self._error("image payload must be PNG")
            return
        self.server.render_requests.append(req)
        time.sleep(0.015)
        self._reply(200, MOCK_MP4_J2, "video/mp4", "video")

    def _parse_telemetry_request(self, req: dict) -> bool | None:
        if "telemetry" not in req or req["telemetry"] is None:
            return False
        tel = req["telemetry"]
        if not isinstance(tel, dict):
            self._error("telemetry must be an object")
            return None
        extra_keys = set(tel) - {"native_allocator"}
        if extra_keys:
            self._error(f"unsupported telemetry field: {sorted(extra_keys)[0]}")
            return None
        val = tel.get("native_allocator")
        if not isinstance(val, bool):
            self._error("telemetry.native_allocator must be a boolean")
            return None
        return val

    def _build_mock_telemetry(self, req: dict, request_id: str, stage_specs: list[tuple[str, int]]) -> dict | None:
        factory = getattr(self.server, "telemetry_factory", None)
        if factory is not None:
            return factory(req, request_id, stage_specs)
        stages = [
            _mock_telemetry_stage(
                stage_name,
                request_id=request_id,
                coverage="COVERED",
                peak_bytes=peak_bytes,
                epoch_id=idx,
            )
            for idx, (stage_name, peak_bytes) in enumerate(stage_specs, start=1)
        ]
        return _mock_telemetry_envelope(request_id, stages)

    def _parse_behavior_request(self, req: dict) -> bool | None:
        if "behavior" not in req or req["behavior"] is None:
            return False
        beh = req["behavior"]
        if not isinstance(beh, dict):
            self._error("behavior must be an object")
            return None
        extra_keys = set(beh) - {"logprobs"}
        if extra_keys:
            self._error(f"unsupported behavior field: {sorted(extra_keys)[0]}")
            return None
        val = beh.get("logprobs")
        if not isinstance(val, bool):
            self._error("behavior.logprobs must be a boolean")
            return None
        if val and not req.get("stream"):
            self._error("behavior.logprobs requires stream=true")
            return None
        return val

    def _build_mock_behavior(
        self, req: dict, *, default_exposed: bool
    ) -> tuple[str | None, object]:
        factory = getattr(self.server, "behavior_factory", None)
        if factory is not None:
            return factory(req, default_exposed)
        if default_exposed:
            return ("exposed", _mock_behavior_payload())
        return ("not_exposed", None)

    def _handle_chat(self, raw: bytes) -> None:
        req = json.loads(raw)
        self.server.chat_requests.append(req)
        messages = req.get("messages")
        if not isinstance(messages, list) or not messages:
            self._error("messages must be non-empty")
            return
        telemetry_requested = self._parse_telemetry_request(req)
        if telemetry_requested is None:
            return
        behavior_requested = self._parse_behavior_request(req)
        if behavior_requested is None:
            return
        has_narrative = isinstance(req.get("narrative"), dict) and bool(
            req["narrative"].get("enabled")
        )
        if req.get("stream") and has_narrative:
            self._handle_narrative_stream(
                req,
                telemetry_requested=telemetry_requested,
                behavior_requested=behavior_requested,
            )
        else:
            self._handle_multiturn_video(
                req,
                telemetry_requested=telemetry_requested,
                behavior_requested=behavior_requested,
            )

    def _handle_narrative_stream(
        self,
        req: dict,
        *,
        telemetry_requested: bool = False,
        behavior_requested: bool = False,
    ) -> None:
        narrative = req.get("narrative") or {}
        if not narrative.get("enabled"):
            self._error("expected narrative.enabled=true for streaming showcase request")
            return
        modalities = set(req.get("modalities") or [])
        want_audio = "audio" in modalities
        if want_audio:
            audio_cfg = req.get("audio") or {}
            if audio_cfg.get("format") != "wav" or not audio_cfg.get("voice"):
                self._error("audio requires non-empty voice and format=wav")
                return

        user_prompt = str(req["messages"][-1].get("content", ""))
        is_pure_text = "square root of 2" in user_prompt.lower() or "pure symbolic math" in user_prompt.lower()

        events: list[dict] = []
        seq = 0
        if is_pure_text:
            part1 = "1. Assume sqrt(2) = a/b in lowest terms with integers a, b.\n2. Then a^2 = 2b^2, so a^2 and therefore a is even (a = 2k).\n"
            part2 = "3. Substituting gives 4k^2 = 2b^2, so b^2 = 2k^2 and b is also even.\n4. Both a and b share factor 2, contradicting lowest terms; hence sqrt(2) is irrational."
            events.append(_mock_envelope(seq, "vlm_delta", part1))
            seq += 1
            events.append(_mock_envelope(seq, "vlm_delta", part2))
            seq += 1
            complete_extra: dict = {}
            if telemetry_requested:
                tel = self._build_mock_telemetry(
                    req,
                    "showcase-mock-1",
                    [("vlm", 134217728)],
                )
                if tel is not None:
                    complete_extra["telemetry"] = tel
            events.append(
                _mock_envelope(seq, "complete", part1 + part2, finish="stop", legs=[], **complete_extra)
            )
        else:
            is_i2v_chain = narrative.get("continuity_mode") == "i2v_chain"
            slot0_extra: dict = {}
            slot1_extra: dict = {}
            pcm_bytes = MOCK_PCM_J1
            if is_i2v_chain:
                part1 = "Above the misty coastal cliffs, a lone wooden lighthouse glows with warm golden amber light."
                part2 = " Moments later, the lantern beam sweeps steadily across rolling twilight waves in a smooth cinematic pan."
                png_bytes = MOCK_PNG_J5
                mp4_bytes = MOCK_MP4_J5
                pcm_bytes = MOCK_PCM_J5
                img_prompt = "Lone wooden lighthouse glowing above misty coastal cliffs, cinematic warm golden glow, 35mm film"
                vid_prompt = "Amber lighthouse beam sweeping steadily across rolling misty twilight coastal waves"
                slot0_extra = {"continuity_mode": "i2v_chain"}
                slot1_extra = {"continuity_mode": "i2v_chain", "anchor_slot_index": 0}
            elif want_audio:
                part1 = "Around a supermassive black hole, superheated plasma forms a luminous accretion disk that glows brighter than an entire galaxy."
                part2 = " Intense spacetime curvature bends light from the far side of the disk over and under the shadow, producing a double-arched Einstein ring."
                png_bytes = MOCK_PNG_J1
                mp4_bytes = MOCK_MP4_J1
                img_prompt = "Glowing golden relativistic black hole accretion disk schematic with crisp photon ring"
                vid_prompt = "Slow orbital camera sweep showing light bending across a black hole event horizon"
            else:
                part1 = "At noon, shafts of turquoise sunlight illuminate a dense coral reef teeming with drifting anthias."
                part2 = " After dusk, bioluminescent dinoflagellates ignite along the rolling tide in ribbons of electric blue."
                png_bytes = MOCK_PNG_J4
                mp4_bytes = MOCK_MP4_J4
                img_prompt = "Sunlit shallow coral reef with clear turquoise shafts of light and vivid coral formations"
                vid_prompt = "Slow underwater tracking shot of bioluminescent blue ocean currents pulsing at night"

            events.append(_mock_envelope(seq, "vlm_delta", part1))
            seq += 1
            events.append(
                _mock_envelope(
                    seq,
                    "visual_slot",
                    stage="image",
                    slot_id="slot-0",
                    slot_index=0,
                    media_type="image",
                    prompt=img_prompt,
                    text_offset=len(part1),
                    **slot0_extra,
                )
            )
            seq += 1
            if want_audio:
                half = len(pcm_bytes) // 2
                first_pcm = _mock_artifact_frames(
                    seq, "art-audio-0", "audio", "audio/pcm", pcm_bytes[:half],
                    "tts", stream_index=0, chunks=2,
                )
                events.extend(first_pcm)
                seq += len(first_pcm)
            events.append(_mock_envelope(seq, "vlm_delta", part2))
            seq += 1
            events.append(
                _mock_envelope(
                    seq,
                    "visual_slot",
                    stage="video",
                    slot_id="slot-1",
                    slot_index=1,
                    media_type="video",
                    prompt=vid_prompt,
                    text_offset=len(part1 + part2),
                    **slot1_extra,
                )
            )
            seq += 1
            if want_audio:
                half = len(pcm_bytes) // 2
                second_pcm = _mock_artifact_frames(
                    seq, "art-audio-1", "audio", "audio/pcm", pcm_bytes[half:],
                    "tts", stream_index=1, chunks=1,
                )
                events.extend(second_pcm)
                seq += len(second_pcm)
            img_frames = _mock_artifact_frames(
                seq, "art-visual-0", "image", "image/png", png_bytes,
                "image", slot_id="slot-0", slot_index=0, chunks=1,
            )
            events.extend(img_frames)
            seq += len(img_frames)
            vid_frames = _mock_artifact_frames(
                seq, "art-visual-1", "video", "video/mp4", mp4_bytes,
                "video", slot_id="slot-1", slot_index=1, chunks=1,
            )
            events.extend(vid_frames)
            seq += len(vid_frames)
            legs = [
                {"stage": "image", "slot_id": "slot-0", "slot_index": 0, "state": "completed"},
                {"stage": "video", "slot_id": "slot-1", "slot_index": 1, "state": "completed"},
            ]
            if want_audio:
                legs.append({"stage": "tts", "state": "completed"})
            complete_extra = {}
            if telemetry_requested:
                stage_specs = [
                    ("vlm", 268435456),
                    ("image", 201326592),
                    ("video", 335544320 if is_i2v_chain else 301989888),
                ]
                if want_audio:
                    stage_specs.append(("tts", 67108864))
                tel = self._build_mock_telemetry(req, "showcase-mock-1", stage_specs)
                if tel is not None:
                    complete_extra["telemetry"] = tel
            events.append(
                _mock_envelope(seq, "complete", part1 + part2, finish="stop", legs=legs, **complete_extra)
            )

        behavior_header: str | None = None
        if behavior_requested:
            behavior_header, behavior_payload = self._build_mock_behavior(
                req, default_exposed=is_pure_text
            )
            if behavior_payload is not None:
                events[-1]["behavior"] = behavior_payload

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        if behavior_header is not None:
            self.send_header(BEHAVIOR_EVIDENCE_HEADER, behavior_header)
        self.end_headers()
        for ev in events:
            frame = json.dumps(ev, separators=(",", ":")).encode("utf-8")
            self.wfile.write(b"data: " + frame + b"\n\n")
            self.wfile.flush()
            time.sleep(0.002)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True

    def _handle_multiturn_video(
        self,
        req: dict,
        *,
        telemetry_requested: bool = False,
        behavior_requested: bool = False,
    ) -> None:
        messages = req["messages"]
        video_parts = 0
        for idx, msg in enumerate(messages):
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "video_url":
                        if idx != 0 or msg.get("role") != "user":
                            self._error("video_url must remain anchored on Turn 0 user message")
                            return
                        video_url = part.get("video_url")
                        if not isinstance(video_url, dict):
                            self._error("video_url part requires a video_url object")
                            return
                        if "frame_selection" in video_url:
                            try:
                                validate_video_frame_selection(video_url["frame_selection"])
                            except ValueError as exc:
                                self._error(str(exc), 400)
                                return
                        url = video_url.get("url", "")
                        if not isinstance(url, str) or not url.startswith("data:video/") or ";base64," not in url:
                            self._error("video_url must be data:video/...;base64,...")
                            return
                        try:
                            decoded = base64.b64decode(url.split(";base64,", 1)[1], validate=True)
                        except (binascii.Error, ValueError):
                            self._error("invalid video_url base64")
                            return
                        if len(decoded) < 12 or decoded[4:8] != b"ftyp":
                            self._error("video_url payload lacks ftyp header")
                            return
                        video_parts += 1
        if video_parts != 1:
            self._error(f"expected exactly 1 video_url part, saw {video_parts}")
            return

        turn_num = (len(messages) + 1) // 2
        if turn_num == 1:
            time.sleep(0.045)  # Simulates cold video ViT + prefix prefill
            answer = (
                "The clip centers on a glowing golden accretion disk encircling a dark central shadow. "
                "Warm amber highlights contrast sharply against the deep black cosmic background."
            )
        elif turn_num == 2:
            time.sleep(0.012)  # Simulates cross-turn prefix KV-cache hit
            answer = (
                "The camera pushes smoothly inward while luminous plasma arcs orbit steadily around the core."
            )
        else:
            time.sleep(0.012)  # Simulates cross-turn prefix KV-cache hit
            answer = (
                "The mood is dramatic and serene, anchored by the golden accretion ring and the dark event horizon."
            )

        if req.get("stream"):
            req_id = f"showcase-video-t{turn_num}"
            model_name = req.get("model", "qwen3-native")
            chunks: list[dict] = [
                {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": 1764000000,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                },
                {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": 1764000000,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": answer}, "finish_reason": None}],
                },
            ]
            final_chunk: dict = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": 1764000000,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            if telemetry_requested:
                tel = self._build_mock_telemetry(
                    req,
                    req_id,
                    [("vlm", 234881024 if turn_num == 1 else 167772160)],
                )
                if tel is not None:
                    final_chunk["telemetry"] = tel
            behavior_header: str | None = None
            if behavior_requested:
                behavior_header, behavior_payload = self._build_mock_behavior(
                    req, default_exposed=False
                )
                if behavior_payload is not None:
                    final_chunk["behavior"] = behavior_payload
            chunks.append(final_chunk)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            if behavior_header is not None:
                self.send_header(BEHAVIOR_EVIDENCE_HEADER, behavior_header)
            self.end_headers()
            for chunk in chunks:
                frame = json.dumps(chunk, separators=(",", ":")).encode("utf-8")
                self.wfile.write(b"data: " + frame + b"\n\n")
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
            return

        body = {
            "id": f"showcase-video-t{turn_num}",
            "object": "chat.completion",
            "created": 1764000000,
            "model": req.get("model", "qwen3-native"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }],
        }
        if telemetry_requested:
            tel = self._build_mock_telemetry(
                req,
                body["id"],
                [("vlm", 234881024 if turn_num == 1 else 167772160)],
            )
            if tel is not None:
                body["telemetry"] = tel
        self._reply(200, json.dumps(body).encode("utf-8"))


def parse_journeys(spec: str) -> list[int]:
    cleaned = spec.strip().lower()
    if cleaned == "all":
        return list(ALL_JOURNEYS)
    result = []
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if token.lower() == "all":
            return list(ALL_JOURNEYS)
        num = int(token)
        if num not in ALL_JOURNEYS:
            valid = ",".join(str(x) for x in ALL_JOURNEYS)
            raise ValueError(f"Unsupported journey number: {num}; valid values are {valid} or all")
        if num not in result:
            result.append(num)
    if not result:
        raise ValueError("At least one journey must be selected")
    return result


def execute_showcase(
    *,
    base_url: str,
    model: str,
    out_dir: Path,
    manifest: dict,
    journeys: list[int],
    evidence_class: str,
    real_model: str,
    timeout: float,
    allow_ffprobe: bool,
    strict_zero_text_slots: bool,
    image_path: Path | None = None,
    video_path: Path | None = None,
    extra_headers: dict[str, str] | None = None,
    sanitized_server_url: str | None = None,
    native_allocator_telemetry: bool = False,
    behavior_logprobs: bool = False,
    video_frame_selection: dict | None = None,
    fallback_image_bytes: bytes | None = None,
    fallback_video_bytes: bytes | None = None,
) -> tuple[dict, dict[str, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    health = check_health(base_url, extra_headers=extra_headers)
    journey_specs = manifest["journeys"]

    results: dict[str, dict] = {}
    all_artifacts: list[dict] = []
    j1_handoff: dict = {}
    j2_handoff: dict = {}

    if 1 in journeys:
        j1_metrics, j1_arts, j1_handoff = run_journey_1(
            base_url,
            model,
            journey_specs["journey_1_interleaved_narrative"],
            out_dir,
            timeout=timeout,
            extra_headers=extra_headers,
            native_allocator_telemetry=native_allocator_telemetry,
            behavior_logprobs=behavior_logprobs,
        )
        results["journey_1"] = j1_metrics
        all_artifacts.extend(j1_arts)

    if 2 in journeys:
        if j1_handoff.get("first_image"):
            src_img = j1_handoff["first_image"]["bytes"]
            src_prompt = j1_handoff["first_image"]["prompt"]
        elif image_path is not None:
            src_img = image_path.read_bytes()
            src_prompt = None
        elif fallback_image_bytes is not None:
            src_img = fallback_image_bytes
            src_prompt = None
        else:
            raise ValueError(
                "Journey 2 requires a PNG from Journey 1 or an explicit --image-path"
            )
        j2_metrics, j2_arts, j2_handoff = run_journey_2(
            base_url,
            journey_specs["journey_2_i2v_animation"],
            out_dir,
            source_image_bytes=src_img,
            source_prompt=src_prompt,
            timeout=timeout,
            allow_ffprobe=allow_ffprobe,
            extra_headers=extra_headers,
        )
        results["journey_2"] = j2_metrics
        all_artifacts.extend(j2_arts)

    if 3 in journeys:
        if video_path is not None:
            vid_bytes = video_path.read_bytes()
        elif j2_handoff.get("video_bytes"):
            vid_bytes = j2_handoff["video_bytes"]
        elif j1_handoff.get("first_video"):
            vid_bytes = j1_handoff["first_video"]["bytes"]
        elif fallback_video_bytes is not None:
            vid_bytes = fallback_video_bytes
        else:
            raise ValueError(
                "Journey 3 requires an MP4 from Journey 2/1 or an explicit --video-path"
            )
        j3_metrics = run_journey_3(
            base_url,
            model,
            journey_specs["journey_3_multiturn_video_qa"],
            video_bytes=vid_bytes,
            timeout=timeout,
            extra_headers=extra_headers,
            native_allocator_telemetry=native_allocator_telemetry,
            behavior_logprobs=behavior_logprobs,
            video_frame_selection=video_frame_selection,
        )
        results["journey_3"] = j3_metrics

    if 4 in journeys:
        j4_metrics, j4_arts = run_journey_4(
            base_url,
            model,
            journey_specs["journey_4_modality_routing_contrast"],
            out_dir,
            timeout=timeout,
            strict_zero_text_slots=strict_zero_text_slots,
            extra_headers=extra_headers,
            native_allocator_telemetry=native_allocator_telemetry,
            behavior_logprobs=behavior_logprobs,
        )
        results["journey_4"] = j4_metrics
        all_artifacts.extend(j4_arts)

    if 5 in journeys:
        j5_metrics, j5_arts = run_journey_5(
            base_url,
            model,
            journey_specs["journey_5"],
            out_dir,
            timeout=timeout,
            allow_ffprobe=allow_ffprobe,
            extra_headers=extra_headers,
            native_allocator_telemetry=native_allocator_telemetry,
            behavior_logprobs=behavior_logprobs,
        )
        results["journey_5"] = j5_metrics
        all_artifacts.extend(j5_arts)

    final_health = check_health(base_url, extra_headers=extra_headers)
    passed_count = sum(1 for item in results.values() if item.get("passed"))
    report = {
        "schema_version": 1,
        "suite_id": manifest["suite_id"],
        "evidence_class": evidence_class,
        "real_model": real_model,
        "model": model,
        "server_url": sanitized_server_url or base_url,
        "journeys_requested": journeys,
        "health_before": health,
        "health_after": final_health,
        "journeys": results,
        "artifacts": all_artifacts,
        "summary": {
            "all_passed": passed_count == len(journeys),
            "journeys_passed": passed_count,
            "journeys_total": len(journeys),
            "verified_artifacts": len(all_artifacts),
        },
    }
    if native_allocator_telemetry:
        chat_summaries = [
            results[key]["native_allocator_telemetry"]
            for key in ("journey_1", "journey_3", "journey_4", "journey_5")
            if key in results and "native_allocator_telemetry" in results[key]
        ]
        if chat_summaries:
            agg = aggregate_native_allocator_telemetry(chat_summaries)
            agg["enabled"] = True
            report["native_allocator_telemetry"] = agg
        else:
            report["native_allocator_telemetry"] = {
                "enabled": True,
                "schema_version": 1,
                "scope": NATIVE_ALLOCATOR_SCOPE,
                "coverage": "NOT_COVERED",
                "requests": 0,
                "covered_stages": 0,
                "total_stages": 0,
                "max_allocated_bytes_peak": None,
                "max_reserved_bytes_peak": None,
                "reasons": [],
                "stages": [],
                "whole_device_nvml": "separate_measurement_resources",
            }
    if behavior_logprobs:
        chat_behavior_summaries = [
            results[key]["behavior_logprobs"]
            for key in ("journey_1", "journey_3", "journey_4", "journey_5")
            if key in results and "behavior_logprobs" in results[key]
        ]
        if chat_behavior_summaries:
            agg_beh = aggregate_behavior_logprobs(chat_behavior_summaries)
            agg_beh["enabled"] = True
            report["behavior_logprobs"] = agg_beh
        else:
            report["behavior_logprobs"] = {
                "enabled": True,
                "status": "not_applicable",
                "requests": 0,
                "exposed_requests": 0,
                "not_exposed_requests": 0,
                "absent_requests": 0,
                "format": None,
                "tokenizer_sha256": None,
                "token_count": None,
                "supervised_count": None,
            }
    report_paths = write_reports(out_dir, report)
    return report, report_paths


def verify_sha256sums_file(out_dir: Path, sums_path: Path) -> int:
    lines = [
        line for line in sums_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    require(bool(lines), "SHA256SUMS is empty")
    for line in lines:
        parts = line.split("  ", 1)
        require(len(parts) == 2, f"Malformed SHA256SUMS line: {line!r}")
        expected, name = parts
        require(len(expected) == 64 and all(c in "0123456789abcdef" for c in expected),
                f"Invalid SHA-256 hex digest in SHA256SUMS: {expected!r}")
        target = out_dir / name
        require(target.is_file(), f"SHA256SUMS references missing file: {name}")
        actual = sha256_bytes(target.read_bytes())
        require(actual == expected, f"SHA256SUMS mismatch for {name}: {actual} != {expected}")
    return len(lines)


def run_self_check(
    manifest_path: Path,
    out_dir: Path | None = None,
    journeys: list[int] | None = None,
    *,
    native_allocator_telemetry: bool = False,
    telemetry_factory=None,
    behavior_logprobs: bool = False,
    behavior_factory=None,
    video_frame_selection: dict | None = None,
    image_path: Path | None = None,
    video_path: Path | None = None,
) -> dict:
    validated_selection = (
        validate_video_frame_selection(video_frame_selection)
        if video_frame_selection is not None
        else None
    )
    manifest = load_manifest(manifest_path)
    selected_journeys = list(ALL_JOURNEYS) if journeys is None else list(journeys)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockShowcaseHandler)
    server.chat_requests = []
    server.render_requests = []
    server.telemetry_factory = telemetry_factory
    server.behavior_factory = behavior_factory
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    tmp_ctx = None
    if out_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="umminfer-showcase-selfcheck-")
        work_dir = Path(tmp_ctx.name)
    else:
        work_dir = out_dir
        work_dir.mkdir(parents=True, exist_ok=True)

    try:
        report, paths = execute_showcase(
            base_url=base_url,
            model=manifest.get("default_model", "qwen3-native"),
            out_dir=work_dir,
            manifest=manifest,
            journeys=selected_journeys,
            evidence_class="OFFLINE_SELF_CHECK",
            real_model="NOT_RUN",
            timeout=30.0,
            allow_ffprobe=False,
            strict_zero_text_slots=True,
            image_path=image_path,
            video_path=video_path,
            sanitized_server_url="http://127.0.0.1:0 (offline-self-check)",
            native_allocator_telemetry=native_allocator_telemetry,
            behavior_logprobs=behavior_logprobs,
            video_frame_selection=validated_selection,
            fallback_image_bytes=MOCK_PNG_J1,
            fallback_video_bytes=MOCK_MP4_J2,
        )
        verified_count = verify_sha256sums_file(work_dir, paths["sha256sums"])
        require(report["evidence_class"] == "OFFLINE_SELF_CHECK", "wrong self-check evidence_class")
        require(report["real_model"] == "NOT_RUN", "wrong self-check real_model stamp")
        if 1 in selected_journeys:
            require(report["journeys"]["journey_1"]["mid_stream_slot_before_vlm_finish"],
                    "Journey 1 did not dispatch visual slot mid-stream")
        if 2 in selected_journeys:
            require(report["journeys"]["journey_2"]["ftyp_verified"],
                    "Journey 2 MP4 ftyp verification failed")
        if 3 in selected_journeys:
            require(report["journeys"]["journey_3"]["turn2_speedup_vs_turn1"] > 1.5,
                    "Journey 3 Turn 2 speedup below expected mock threshold")
            require(report["journeys"]["journey_3"]["turn3_speedup_vs_turn1"] > 1.5,
                    "Journey 3 Turn 3 speedup below expected mock threshold")
        if 4 in selected_journeys:
            require(report["journeys"]["journey_4"]["pure_text_visual_slots"] == 0,
                    "Journey 4 pure-text case emitted visual slots")
            require(report["journeys"]["journey_4"]["visual_scene_visual_slots"] == 2,
                    "Journey 4 visual-scene case did not emit expected 2 slots")
        if 5 in selected_journeys:
            j5 = report["journeys"]["journey_5"]
            require(j5["i2v_chain_verified"], "Journey 5 i2v_chain verification failed")
            require(j5["slot_1_continuity_mode"] == "i2v_chain",
                    "Journey 5 Slot 1 continuity_mode mismatch")
            require(j5["slot_1_anchor_slot_index"] == 0,
                    "Journey 5 Slot 1 anchor_slot_index mismatch")

        expected_renders = 1 if 2 in selected_journeys else 0
        expected_chats = (
            (1 if 1 in selected_journeys else 0)
            + (3 if 3 in selected_journeys else 0)
            + (2 if 4 in selected_journeys else 0)
            + (1 if 5 in selected_journeys else 0)
        )
        require(len(server.render_requests) == expected_renders,
                f"expected {expected_renders} /v1/visual/render request(s), got {len(server.render_requests)}")
        require(len(server.chat_requests) == expected_chats,
                f"expected {expected_chats} /v1/chat/completions request(s), got {len(server.chat_requests)}")
        for req in server.render_requests:
            require(
                "telemetry" not in req,
                f"/v1/visual/render must never include telemetry: {req}",
            )
            require(
                "behavior" not in req,
                f"/v1/visual/render must never include behavior: {req}",
            )
        for req in server.chat_requests:
            if native_allocator_telemetry:
                require(
                    req.get("telemetry") == {"native_allocator": True},
                    f"/v1/chat/completions missing opt-in telemetry: {req}",
                )
            else:
                require(
                    "telemetry" not in req,
                    f"/v1/chat/completions must omit telemetry by default: {req}",
                )
            if behavior_logprobs:
                require(
                    req.get("behavior") == {"logprobs": True},
                    f"/v1/chat/completions missing opt-in behavior: {req}",
                )
                require(
                    req.get("stream") is True,
                    f"/v1/chat/completions with behavior.logprobs must set stream=True: {req}",
                )
            else:
                require(
                    "behavior" not in req,
                    f"/v1/chat/completions must omit behavior by default: {req}",
                )
            msg0_content = req["messages"][0].get("content")
            if isinstance(msg0_content, list):
                for part in msg0_content:
                    if isinstance(part, dict) and part.get("type") == "video_url":
                        vurl = part["video_url"]
                        if validated_selection is not None:
                            require(
                                vurl.get("frame_selection") == validated_selection,
                                f"Journey 3 video_url.frame_selection mismatch: "
                                f"{vurl.get('frame_selection')!r} != {validated_selection!r}",
                            )
                        else:
                            require(
                                "frame_selection" not in vurl,
                                f"Journey 3 default video_url must omit frame_selection: {vurl}",
                            )

        print(
            "SELF-CHECK PASS [OFFLINE_SELF_CHECK | REAL_MODEL: NOT_RUN]: "
            f"{report['summary']['journeys_passed']}/{report['summary']['journeys_total']} journeys passed, "
            f"{report['summary']['verified_artifacts']} media artifacts + "
            f"{verified_count} SHA256SUMS entries verified."
        )
        return report
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ummInfer North-Star Native Multimodal Showcase & Verification Suite"
    )
    parser.add_argument(
        "--self-check",
        "--mock",
        dest="self_check",
        action="store_true",
        help="Run deterministic offline loopback server verification across selected journeys (default: all 5)",
    )
    parser.add_argument(
        "--url",
        help="Live ummInfer server URL (e.g. http://127.0.0.1:43301)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model ID to send in /v1/chat/completions (defaults to manifest default_model)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write generated artifacts, reports, and SHA256SUMS",
    )
    parser.add_argument(
        "--journeys",
        "--journey",
        dest="journeys",
        default="all",
        help="Comma-separated journey numbers (1..5) or 'all' (default: all)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to showcase prompts manifest JSON",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="Optional PNG path for Journey 2 when Journey 1 is not run",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help="Optional MP4 path for Journey 3",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="Per-request HTTP/SSE timeout in seconds (default: 900)",
    )
    parser.add_argument(
        "--allow-remote-http",
        action="store_true",
        help="Explicitly allow plain HTTP against a non-loopback host",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Optional environment variable name holding a Bearer token (never logged)",
    )
    parser.add_argument(
        "--strict-zero-text-slots",
        action="store_true",
        help="Fail live Journey 4 if the pure-text prompt emits any visual slot",
    )
    parser.add_argument(
        "--native-allocator-telemetry",
        action="store_true",
        help=(
            "Opt-in: send telemetry.native_allocator=true on /v1/chat/completions "
            "requests and strictly validate/report request-native allocator telemetry"
        ),
    )
    parser.add_argument(
        "--behavior-logprobs",
        action="store_true",
        help=(
            "Opt-in: send behavior.logprobs=true on streaming /v1/chat/completions "
            "requests and strictly validate/report X-UmmInfer-Behavior-Evidence and "
            "umminfer-behavior-logprobs-v1 payloads"
        ),
    )
    parser.add_argument(
        "--video-frame-policy",
        choices=VIDEO_FRAME_POLICIES,
        default=None,
        help=(
            "Opt-in Journey 3 video_url.frame_selection.policy (uniform or temporal); "
            "omitted by default to keep video_url byte-compatible"
        ),
    )
    parser.add_argument(
        "--video-motion-threshold",
        type=float,
        default=None,
        help=(
            "Optional finite float in [0.0, 1.0] for Journey 3 temporal motion threshold "
            "(requires --video-frame-policy temporal)"
        ),
    )
    parser.add_argument(
        "--video-scene-threshold",
        type=float,
        default=None,
        help=(
            "Optional finite float in [0.0, 1.0] for Journey 3 temporal scene threshold "
            "(requires --video-frame-policy temporal)"
        ),
    )
    args = parser.parse_args(argv)

    try:
        video_frame_selection = build_video_frame_selection(
            args.video_frame_policy,
            motion_threshold=args.video_motion_threshold,
            scene_threshold=args.video_scene_threshold,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.self_check:
        if args.url:
            parser.error("--self-check/--mock and --url are mutually exclusive")
        journeys = parse_journeys(args.journeys)
        run_self_check(
            args.manifest,
            out_dir=args.out_dir,
            journeys=journeys,
            native_allocator_telemetry=args.native_allocator_telemetry,
            behavior_logprobs=args.behavior_logprobs,
            video_frame_selection=video_frame_selection,
            image_path=args.image_path,
            video_path=args.video_path,
        )
        return 0

    if not args.url:
        parser.error("Specify either --self-check/--mock or --url <server-url>")
    if args.out_dir is None:
        parser.error("--out-dir is required when running against a live --url")

    base_url = validate_server_url(args.url, allow_remote_http=args.allow_remote_http)
    extra_headers = build_auth_headers(args.api_key_env)
    manifest = load_manifest(args.manifest)
    model = args.model or manifest.get("default_model", "qwen3-native")
    journeys = parse_journeys(args.journeys)

    report, paths = execute_showcase(
        base_url=base_url,
        model=model,
        out_dir=args.out_dir,
        manifest=manifest,
        journeys=journeys,
        evidence_class="LIVE_SERVER",
        real_model="LIVE_SERVER",
        timeout=args.timeout,
        allow_ffprobe=True,
        strict_zero_text_slots=args.strict_zero_text_slots,
        image_path=args.image_path,
        video_path=args.video_path,
        extra_headers=extra_headers,
        native_allocator_telemetry=args.native_allocator_telemetry,
        behavior_logprobs=args.behavior_logprobs,
        video_frame_selection=video_frame_selection,
    )
    verified = verify_sha256sums_file(args.out_dir, paths["sha256sums"])
    print(
        f"SHOWCASE PASS [LIVE_SERVER]: {report['summary']['journeys_passed']}/"
        f"{report['summary']['journeys_total']} journeys passed, "
        f"{verified} SHA256SUMS entries written to {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
