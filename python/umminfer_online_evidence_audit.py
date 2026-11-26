# SPDX-License-Identifier: Apache-2.0
"""Independent, stdlib-only audit of retained online training evidence.

This module deliberately does not import the online rollout producer, the
behavior/token alignment validator, the online-GRPO preflight adapter, dataset
scoring, or objective implementations.  It re-parses public JSON/JSONL and raw
SSE bytes and independently derives identities, groups, masks, and arithmetic.

Malformed evidence raises :class:`AuditInputError` before a report is written.
A well-formed bundle that disagrees with separately accepted evidence instead
produces a precise failed comparison report.  No network, GPU, model, optimizer,
or ML framework is used.
"""

import argparse
import ast
import base64
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import unicodedata


AUDIT_SCHEMA = "umminfer-online-evidence-audit-v1"
PLAN_SCHEMA = "umminfer-online-grpo-plan-v1"
PROVENANCE_CONTRACT = "umminfer-online-rollout-provenance-v1"
BEHAVIOR_FORMAT = "umminfer-behavior-logprobs-v1"
ALIGNMENT_FIXTURE_SCHEMA = "umminfer-behavior-token-alignment-fixture-v1"
PREFLIGHT_CONFIG_SCHEMA = "umminfer-online-grpo-preflight-config-v1"
PREFLIGHT_RUN_SCHEMA = "umminfer-online-grpo-run-identity-v1"
PREFLIGHT_SCHEDULE_SCHEMA = "umminfer-online-grpo-preflight-schedule-v1"
PREFLIGHT_OBJECTIVE_SCHEMA = "umminfer-online-grpo-objective-fixtures-v1"
PREFLIGHT_CHECKPOINT_SCHEMA = "umminfer-online-grpo-preflight-checkpoint-v1"
PREFLIGHT_MANIFEST_SCHEMA = "umminfer-online-grpo-preflight-manifest-v1"

FORBIDDEN_AUDIT_IMPORTS = frozenset({
    "umminfer_online_rollout",
    "umminfer_behavior_token_alignment",
    "umminfer_online_grpo_preflight",
    "umminfer_joint_grpo_live",
    "umminfer_joint_trainer",
    "torch", "transformers", "numpy", "mlx",
})
ML_MODULES = frozenset({"torch", "transformers", "numpy", "mlx"})
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


class AuditInputError(ValueError):
    """Malformed or unverifiable input; no audit report may be emitted."""

    def __init__(self, code, path, detail):
        self.code = str(code)
        self.path = str(path)
        self.detail = str(detail)
        super().__init__(f"{self.code} at {self.path}: {self.detail}")


class EvidenceDisagreement(ValueError):
    """A structurally valid audit disagrees with separately accepted claims."""

    def __init__(self, mismatches):
        self.mismatches = list(mismatches)
        super().__init__("accepted evidence disagrees with independent findings")


def require(condition, code, path, detail="constraint failed"):
    if not condition:
        raise AuditInputError(code, path, detail)


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def finite(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(float(value)))


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def canonical_bytes(value):
    """Canonical public JSON: sorted keys, compact UTF-8, finite numbers."""
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise AuditInputError("NON_CANONICAL_VALUE", "$", str(exc)) from exc
    return text.encode("utf-8")


def canonical_hash(value):
    return sha256_bytes(canonical_bytes(value))


def _strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AuditInputError("DUPLICATE_JSON_MEMBER", "$", key)
        result[key] = value
    return result


def _invalid_constant(value):
    raise AuditInputError("NON_FINITE_JSON", "$", value)


def strict_json_bytes(raw, path):
    require(isinstance(raw, bytes), "NOT_BYTES", path)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AuditInputError("INVALID_UTF8", path, str(exc)) from exc
    try:
        value = json.loads(text, object_pairs_hook=_strict_pairs,
                           parse_constant=_invalid_constant)
    except AuditInputError as exc:
        raise AuditInputError(exc.code, path, exc.detail) from exc
    except (json.JSONDecodeError, RecursionError) as exc:
        raise AuditInputError("MALFORMED_JSON", path, str(exc)) from exc
    _require_finite_tree(value, path)
    return value


def _require_finite_tree(value, path):
    if isinstance(value, float):
        require(math.isfinite(value), "NON_FINITE_JSON", path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _require_finite_tree(item, f"{path}.{key}")


def safe_relative(value, path):
    require(isinstance(value, str) and value and "\\" not in value,
            "INVALID_PATH", path)
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts
            and "." not in pure.parts and pure.as_posix() == value,
            "INVALID_PATH", path, value)
    return pure


def read_file(root, relative, path=None, max_bytes=64 * 1024 * 1024):
    pure = safe_relative(relative, path or str(relative))
    target = Path(root).joinpath(*pure.parts)
    require(target.is_file() and not target.is_symlink(),
            "MISSING_OR_SYMBOLIC_FILE", path or str(relative))
    size = target.stat().st_size
    require(size <= max_bytes, "FILE_TOO_LARGE", path or str(relative), str(size))
    return target.read_bytes()


def read_json(root, relative, *, canonical=False):
    raw = read_file(root, relative)
    value = strict_json_bytes(raw, str(relative))
    if canonical:
        require(raw == canonical_bytes(value) + b"\n", "NON_CANONICAL_DOCUMENT",
                str(relative), "expected compact sorted JSON plus one newline")
    return value


def read_jsonl(root, relative, *, canonical=True, allow_empty=True):
    raw = read_file(root, relative)
    require(not raw.endswith(b"\r"), "INVALID_JSONL_FRAMING", str(relative))
    if not raw:
        require(allow_empty, "MISSING_RECORDS", str(relative))
        return []
    require(raw.endswith(b"\n"), "TRUNCATED_JSONL", str(relative))
    lines = raw[:-1].split(b"\n")
    records = []
    for index, line in enumerate(lines):
        require(bool(line), "EMPTY_JSONL_RECORD", f"{relative}:{index + 1}")
        value = strict_json_bytes(line, f"{relative}:{index + 1}")
        require(isinstance(value, dict), "NON_OBJECT_JSONL_RECORD",
                f"{relative}:{index + 1}")
        if canonical:
            require(line == canonical_bytes(value), "NON_CANONICAL_JSONL_RECORD",
                    f"{relative}:{index + 1}")
        records.append(value)
    return records


def file_identity(root, relative):
    raw = read_file(root, relative)
    return {"path": str(relative), "sha256": sha256_bytes(raw), "bytes": len(raw)}


def verify_sidecar(root, relative):
    raw = read_file(root, relative)
    sidecar = read_file(root, relative + ".sha256", max_bytes=4096)
    require(sidecar == sha256_bytes(raw).encode("ascii") + b"\n",
            "STALE_SHA256_SIDECAR", relative + ".sha256")


def require_sha256(value, path):
    require(isinstance(value, str) and HEX64.fullmatch(value) is not None,
            "INVALID_SHA256", path)
    return value


def same(actual, expected, code, path):
    if actual == expected:
        return
    left = repr(actual)
    right = repr(expected)
    if len(left) > 512:
        left = left[:509] + "..."
    if len(right) > 512:
        right = right[:509] + "..."
    raise AuditInputError(code, path,
                          f"independent={left} evidence={right}")


def parse_sse(raw, path):
    """Parse exact SSE frames and return source payloads with byte spans."""
    require(isinstance(raw, bytes) and raw, "EMPTY_RAW_SSE", path)
    frames = []
    cursor = 0
    separator = re.compile(br"\r?\n\r?\n")
    while cursor < len(raw):
        match = separator.search(raw, cursor)
        require(match is not None, "TRUNCATED_SSE_FRAME", path,
                f"offset={cursor}")
        end = match.end()
        frame_raw = raw[cursor:end]
        try:
            text = frame_raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AuditInputError("INVALID_SSE_UTF8", path, str(exc)) from exc
        data_lines = []
        for line in text.splitlines():
            if line.startswith(":") or not line:
                continue
            field, separator_found, value = line.partition(":")
            if field == "data" and separator_found:
                if value.startswith(" "):
                    value = value[1:]
                data_lines.append(value)
        payload = "\n".join(data_lines)
        kind = "keepalive" if not data_lines else (
            "done" if payload == "[DONE]" else "event")
        value = None
        if kind == "event":
            value = strict_json_bytes(payload.encode("utf-8"),
                                      f"{path}.frame[{len(frames)}].data")
            require(isinstance(value, dict), "NON_OBJECT_SSE_DATA",
                    f"{path}.frame[{len(frames)}]")
        frames.append({
            "index": len(frames), "offset": cursor, "bytes": end - cursor,
            "sha256": sha256_bytes(frame_raw), "data_sha256": sha256_bytes(
                payload.encode("utf-8")), "kind": kind, "data": payload,
            "value": value, "raw": frame_raw,
        })
        cursor = end
    require(frames and frames[-1]["kind"] == "done", "MISSING_DONE_FRAME", path)
    same(sum(frame["kind"] == "done" for frame in frames), 1,
         "DUPLICATE_DONE_FRAME", path)
    return frames


def verify_raw_replay(record, artifact_root, path):
    """Independently verify CAS bytes, chunks, frames, ticks, and event links."""
    metadata = record.get("sse_replay")
    require(isinstance(metadata, dict), "MISSING_RAW_REPLAY", path)
    same(metadata.get("format"), "umminfer-raw-sse-replay-v1",
         "UNSUPPORTED_RAW_REPLAY", path + ".sse_replay.format")
    same(metadata.get("format_version"), 1, "UNSUPPORTED_RAW_REPLAY",
         path + ".sse_replay.format_version")
    clock = metadata.get("clock")
    require(isinstance(clock, dict), "MISSING_REPLAY_CLOCK", path + ".sse_replay.clock")
    clock_kind = clock.get("kind")
    expected_clock = {
        "monotonic_ns": {"source": "python.time.perf_counter_ns", "unit": "ns",
                         "observation": "observed_arrival"},
        "logical_tick": {"source": "caller_supplied_logical_ticks", "unit": "tick",
                         "observation": "deterministic_fixture_not_measured_time"},
    }
    require(clock_kind in expected_clock, "INVALID_REPLAY_CLOCK",
            path + ".sse_replay.clock.kind")
    for key, expected in expected_clock[clock_kind].items():
        same(clock.get(key), expected, "REPLAY_CLOCK_MISMATCH",
             f"{path}.sse_replay.clock.{key}")
    same(clock.get("origin"), "capture_start", "REPLAY_CLOCK_ORIGIN",
         path + ".sse_replay.clock.origin")
    raw_ref = metadata.get("raw_ref")
    require_sha256(metadata.get("raw_sha256"), path + ".sse_replay.raw_sha256")
    same(raw_ref, metadata["raw_sha256"] + ".sse", "RAW_CAS_NAME_MISMATCH",
         path + ".sse_replay.raw_ref")
    raw = read_file(artifact_root, raw_ref)
    same(sha256_bytes(raw), metadata["raw_sha256"], "RAW_CAS_HASH_MISMATCH",
         path + ".sse_replay.raw_ref")
    same(len(raw), metadata.get("raw_bytes"), "RAW_LENGTH_MISMATCH",
         path + ".sse_replay.raw_bytes")
    frames = parse_sse(raw, path + ".sse_replay.raw_ref")
    frame_meta = metadata.get("frames")
    require(isinstance(frame_meta, list) and len(frame_meta) == len(frames),
            "SSE_FRAME_COUNT_MISMATCH", path + ".sse_replay.frames")
    previous_tick = None
    for index, (actual, declared) in enumerate(zip(frames, frame_meta)):
        require(isinstance(declared, dict), "INVALID_SSE_FRAME_METADATA",
                f"{path}.frames[{index}]")
        for key in ("index", "offset", "bytes", "sha256", "kind"):
            same(declared.get(key), actual[key], "SSE_FRAME_METADATA_MISMATCH",
                 f"{path}.frames[{index}].{key}")
        if actual["kind"] in ("event", "done"):
            same(declared.get("data_sha256"), actual["data_sha256"],
                 "SSE_FRAME_DATA_HASH", f"{path}.frames[{index}].data_sha256")
        else:
            require("data_sha256" not in declared, "UNEXPECTED_KEEPALIVE_DATA_HASH",
                    f"{path}.frames[{index}]")
        tick = declared.get("tick")
        require(integer(tick) and tick >= 0, "INVALID_SSE_TICK",
                f"{path}.frames[{index}].tick")
        if previous_tick is not None:
            require(tick >= previous_tick, "REORDERED_SSE_TICK",
                    f"{path}.frames[{index}].tick")
        previous_tick = tick
    chunks = metadata.get("chunks")
    require(isinstance(chunks, list) and chunks, "MISSING_TRANSPORT_CHUNKS",
            path + ".sse_replay.chunks")
    cursor = 0
    previous_tick = None
    for index, chunk in enumerate(chunks):
        require(isinstance(chunk, dict), "INVALID_TRANSPORT_CHUNK",
                f"{path}.chunks[{index}]")
        same(chunk.get("index"), index, "REORDERED_TRANSPORT_CHUNK",
             f"{path}.chunks[{index}].index")
        same(chunk.get("offset"), cursor, "TRANSPORT_CHUNK_GAP",
             f"{path}.chunks[{index}].offset")
        size = chunk.get("bytes")
        require(integer(size) and size > 0 and cursor + size <= len(raw),
                "INVALID_TRANSPORT_CHUNK_SIZE", f"{path}.chunks[{index}].bytes")
        payload = raw[cursor:cursor + size]
        same(chunk.get("sha256"), sha256_bytes(payload), "TRANSPORT_CHUNK_HASH",
             f"{path}.chunks[{index}].sha256")
        tick = chunk.get("tick")
        require(integer(tick) and tick >= 0, "INVALID_CHUNK_TICK",
                f"{path}.chunks[{index}].tick")
        if previous_tick is not None:
            require(tick >= previous_tick, "REORDERED_CHUNK_TICK",
                    f"{path}.chunks[{index}].tick")
        previous_tick = tick
        cursor += size
    same(cursor, len(raw), "TRAILING_UNCHUNKED_RAW_BYTES", path)
    same(metadata.get("done_frame"), len(frames) - 1, "DONE_FRAME_MISMATCH",
         path + ".sse_replay.done_frame")

    events = record.get("events")
    require(isinstance(events, list), "MISSING_NORMALIZED_EVENTS", path + ".events")
    same(metadata.get("normalized_events_sha256"), canonical_hash(events),
         "NORMALIZED_EVENT_HASH_MISMATCH", path + ".events")
    previous_source_index = -1
    final_artifact_hashes = {}
    for frame in frames:
        unified = frame["value"].get("unified") if frame["value"] else None
        if isinstance(unified, dict) and isinstance(unified.get("artifact"), dict):
            artifact = unified["artifact"]
            if artifact.get("sha256") is not None:
                final_artifact_hashes[artifact.get("artifact_id")] = artifact["sha256"]
    for index, event in enumerate(events):
        require(isinstance(event, dict), "INVALID_NORMALIZED_EVENT",
                f"{path}.events[{index}]")
        same(event.get("seq"), index, "REORDERED_NORMALIZED_EVENT",
             f"{path}.events[{index}].seq")
        source_index = event.get("source_event_index")
        require(integer(source_index) and source_index >= previous_source_index,
                "REORDERED_SOURCE_EVENT", f"{path}.events[{index}]")
        previous_source_index = source_index
        require(integer(source_index) and 0 <= source_index < len(frames),
                "MISSING_SOURCE_EVENT", f"{path}.events[{index}]")
        source = frames[source_index]
        same(event.get("source_event_sha256"), source["sha256"],
             "SOURCE_EVENT_HASH_MISMATCH", f"{path}.events[{index}]")
        same(event.get("source_event_tick"), frame_meta[source_index]["tick"],
             "SOURCE_EVENT_TICK_MISMATCH", f"{path}.events[{index}]")
        unified = source["value"].get("unified") if source["value"] else None
        require(isinstance(unified, dict), "SOURCE_EVENT_NOT_UNIFIED",
                f"{path}.events[{index}]")
        expected = copy.deepcopy(unified)
        kind = expected.pop("kind", None)
        server_seq = expected.pop("seq", None)
        expected["type"] = kind
        expected["server_seq"] = server_seq
        expected["seq"] = event.get("seq")
        expected.setdefault("stage", None)
        expected.setdefault("is_final", False)
        expected.setdefault("text", "")
        expected["source_event_index"] = source_index
        expected["source_event_sha256"] = source["sha256"]
        expected["source_event_tick"] = frame_meta[source_index]["tick"]
        if kind == "artifact" and isinstance(expected.get("artifact"), dict):
            artifact = expected["artifact"]
            try:
                payload = base64.b64decode(artifact.pop("data"), validate=True)
            except (ValueError, TypeError) as exc:
                raise AuditInputError("INVALID_ARTIFACT_BASE64",
                                      f"{path}.events[{index}]", str(exc)) from exc
            artifact.pop("encoding", None)
            artifact["chunk_bytes"] = len(payload)
            artifact["chunk_sha256"] = sha256_bytes(payload)
            suffix = {"png": ".png", "jpg": ".jpg", "jpeg": ".jpeg",
                      "mp4": ".mp4", "wav": ".wav", "pcm": ".pcm"}.get(
                          artifact.get("format"), ".bin")
            artifact["ref"] = final_artifact_hashes.get(
                artifact.get("artifact_id"), sha256_bytes(payload)) + suffix
        same(event, expected, "NORMALIZED_EVENT_RECONSTRUCTION",
             f"{path}.events[{index}]")
    same(metadata.get("event_count"), len(events), "EVENT_COUNT_MISMATCH",
         path + ".sse_replay.event_count")
    same(metadata.get("terminal_state"), record.get("terminal_state"),
         "REPLAY_TERMINAL_MISMATCH", path + ".terminal_state")
    terminals = [event for event in events
                 if event.get("type") in ("complete", "error", "cancelled")]
    same(len(terminals), 1, "TERMINAL_COUNT", path + ".events")
    same(events[-1], terminals[0], "TERMINAL_NOT_LAST", path + ".events")
    same(record.get("terminal_event_seq"), terminals[0]["seq"],
         "TERMINAL_SEQUENCE", path + ".terminal_event_seq")
    same(metadata.get("terminal_event_seq"), terminals[0]["seq"],
         "REPLAY_TERMINAL_SEQUENCE", path + ".sse_replay.terminal_event_seq")
    same(record.get("terminal_legs"), terminals[0].get("legs", []),
         "TERMINAL_LEGS_MISMATCH", path + ".terminal_legs")
    visual_slots = []
    for event in events:
        if event.get("type") == "visual_slot":
            slot = {key: copy.deepcopy(value) for key, value in event.items()
                    if key not in {"type", "is_final", "text", "session_id",
                                   "request_id", "source_event_index",
                                   "source_event_sha256", "source_event_tick"}}
            visual_slots.append(slot)
    same(record.get("visual_slots"), visual_slots, "VISUAL_SLOT_RECONSTRUCTION",
         path + ".visual_slots")
    same(record.get("leg_errors"), [event for event in events
                                    if event.get("type") == "leg_error"],
         "LEG_ERROR_RECONSTRUCTION", path + ".leg_errors")
    return {"raw_sha256": metadata["raw_sha256"], "raw_bytes": len(raw),
            "frames": frames, "frame_count": len(frames), "chunk_count": len(chunks)}


def behavior_from_raw(replay, path):
    """Require exactly one final behavior block and return its strict values."""
    candidates = []
    for frame in replay["frames"]:
        value = frame["value"]
        if value is not None and "behavior" in value:
            candidates.append((frame, value["behavior"]))
    same(len(candidates), 1, "BEHAVIOR_BLOCK_COUNT", path)
    frame, behavior = candidates[0]
    require(frame["index"] == replay["frame_count"] - 2,
            "BEHAVIOR_NOT_IN_FINAL_FRAME", path)
    require(isinstance(behavior, dict) and set(behavior) == {
        "format", "tokenizer_sha256", "token_ids", "logprobs", "mask",
    }, "BEHAVIOR_FIELDS", path)
    same(behavior.get("format"), BEHAVIOR_FORMAT, "BEHAVIOR_FORMAT", path)
    require_sha256(behavior.get("tokenizer_sha256"), path + ".tokenizer_sha256")
    ids = behavior.get("token_ids")
    scores = behavior.get("logprobs")
    masks = behavior.get("mask")
    require(isinstance(ids, list) and ids and isinstance(scores, list)
            and isinstance(masks, list) and len(ids) == len(scores) == len(masks),
            "BEHAVIOR_ARRAY_ALIGNMENT", path)
    for index, (token, score, mask) in enumerate(zip(ids, scores, masks)):
        require(integer(token) and token >= 0, "INVALID_SAMPLED_ID",
                f"{path}.token_ids[{index}]")
        require(finite(score) and float(score) <= 0.0, "INVALID_BEHAVIOR_LOGPROB",
                f"{path}.logprobs[{index}]")
        require(integer(mask) and mask in (0, 1), "INVALID_BEHAVIOR_MASK",
                f"{path}.mask[{index}]")
    require(any(masks), "NO_SUPERVISED_BEHAVIOR_TOKEN", path + ".mask")
    return behavior


def _printable(value, path):
    require(isinstance(value, str) and 0 < len(value) <= 256
            and all(ord(char) >= 32 and ord(char) != 127 for char in value),
            "INVALID_TEXT_IDENTITY", path)
    return value


def prepare_plan(value):
    """Validate the closed public plan schema without producer imports."""
    keys = {"schema", "batch_id", "behavior_policy", "sampling",
            "request_template", "reward", "split_seed", "eval_ratio",
            "transport", "groups"}
    require(isinstance(value, dict) and set(value) == keys
            and value.get("schema") == PLAN_SCHEMA,
            "INVALID_PLAN_SCHEMA", "plan")
    plan = copy.deepcopy(value)
    _printable(plan["batch_id"], "plan.batch_id")
    policy_keys = {"policy_id", "revision", "model_id", "model_revision",
                   "model_sha256", "model_config_sha256"}
    policy = plan["behavior_policy"]
    require(isinstance(policy, dict) and set(policy) == policy_keys,
            "INVALID_BEHAVIOR_POLICY", "plan.behavior_policy")
    for key in ("policy_id", "revision", "model_id", "model_revision"):
        _printable(policy[key], "plan.behavior_policy." + key)
    for key in ("model_sha256", "model_config_sha256"):
        require_sha256(policy[key], "plan.behavior_policy." + key)
    sampling = plan["sampling"]
    require(isinstance(sampling, dict)
            and set(sampling) == {"max_tokens", "temperature", "top_p"},
            "INVALID_SAMPLING", "plan.sampling")
    require(integer(sampling["max_tokens"]) and sampling["max_tokens"] > 0,
            "INVALID_MAX_TOKENS", "plan.sampling.max_tokens")
    require(finite(sampling["temperature"])
            and 0.0 <= float(sampling["temperature"]) <= 2.0,
            "INVALID_TEMPERATURE", "plan.sampling.temperature")
    require(finite(sampling["top_p"]) and 0.0 < float(sampling["top_p"]) <= 1.0,
            "INVALID_TOP_P", "plan.sampling.top_p")
    template = plan["request_template"]
    require(isinstance(template, dict), "INVALID_REQUEST_TEMPLATE",
            "plan.request_template")
    require(not ({"model", "messages", "stream", "session_id", "seed",
                  "max_tokens", "temperature", "top_p"} & set(template)),
            "REQUEST_TEMPLATE_OVERRIDES_BINDING", "plan.request_template")
    require(isinstance(plan["reward"], dict)
            and set(plan["reward"]) == {"spec"}, "INVALID_REWARD", "plan.reward")
    _printable(plan["reward"]["spec"], "plan.reward.spec")
    require(":" in plan["reward"]["spec"], "INVALID_REWARD_SPEC", "plan.reward.spec")
    _printable(plan["split_seed"], "plan.split_seed")
    require(finite(plan["eval_ratio"]) and 0.0 <= float(plan["eval_ratio"]) <= 1.0,
            "INVALID_EVAL_RATIO", "plan.eval_ratio")
    transport = plan["transport"]
    require(isinstance(transport, dict) and set(transport) == {
        "timeout_s", "max_attempts", "retry_statuses", "retry_transport",
        "retry_timeouts"}, "INVALID_TRANSPORT", "plan.transport")
    require(finite(transport["timeout_s"]) and transport["timeout_s"] > 0,
            "INVALID_TIMEOUT", "plan.transport.timeout_s")
    require(integer(transport["max_attempts"])
            and 1 <= transport["max_attempts"] <= 8,
            "INVALID_ATTEMPTS", "plan.transport.max_attempts")
    statuses = transport["retry_statuses"]
    require(isinstance(statuses, list)
            and all(integer(item) and 400 <= item <= 599 for item in statuses)
            and len(statuses) == len(set(statuses)), "INVALID_RETRY_STATUSES",
            "plan.transport.retry_statuses")
    require(isinstance(transport["retry_transport"], bool)
            and isinstance(transport["retry_timeouts"], bool),
            "INVALID_RETRY_SWITCH", "plan.transport")
    groups = plan["groups"]
    require(isinstance(groups, list) and groups, "MISSING_PLAN_GROUPS", "plan.groups")
    prompt_ids = set()
    group_size = None
    for index, group in enumerate(groups):
        path = f"plan.groups[{index}]"
        require(isinstance(group, dict) and set(group) == {
            "prompt_id", "messages", "expected_visual_modalities", "seeds"},
            "INVALID_PLAN_GROUP", path)
        _printable(group["prompt_id"], path + ".prompt_id")
        require(group["prompt_id"] not in prompt_ids, "DUPLICATE_PROMPT_ID",
                path + ".prompt_id")
        prompt_ids.add(group["prompt_id"])
        require(isinstance(group["messages"], list) and group["messages"],
                "MISSING_MESSAGES", path + ".messages")
        for message_index, message in enumerate(group["messages"]):
            require(isinstance(message, dict) and "role" in message
                    and "content" in message and isinstance(message["role"], str)
                    and bool(message["role"]), "INVALID_MESSAGE",
                    f"{path}.messages[{message_index}]")
        labels = group["expected_visual_modalities"]
        require(isinstance(labels, list)
                and all(item in ("image", "video") for item in labels)
                and len(labels) == len(set(labels)), "INVALID_VISUAL_LABELS",
                path + ".expected_visual_modalities")
        seeds = group["seeds"]
        require(isinstance(seeds, list) and len(seeds) >= 2
                and all(integer(seed) and INT64_MIN <= seed <= INT64_MAX
                        for seed in seeds)
                and len(seeds) == len(set(seeds)), "INVALID_GROUP_SEEDS",
                path + ".seeds")
        if group_size is None:
            group_size = len(seeds)
        same(len(seeds), group_size, "NON_UNIFORM_GROUP_SIZE", path + ".seeds")
    canonical_bytes(plan)
    return plan


def build_schedule(plan):
    """Derive complete plan-order request and schedule identities."""
    behavior_hash = canonical_hash(plan["behavior_policy"])
    sampling_hash = canonical_hash(plan["sampling"])
    schedule = []
    schedule_identity = []
    for group_index, group in enumerate(plan["groups"]):
        group_identity = {
            "batch_id": plan["batch_id"], "group_index": group_index,
            "prompt_id": group["prompt_id"], "messages": group["messages"],
            "seeds": group["seeds"], "request_template": plan["request_template"],
            "behavior_policy_sha256": behavior_hash,
        }
        group_id = canonical_hash(group_identity)[:32]
        rollouts = []
        for rollout_index, seed in enumerate(group["seeds"]):
            session_id = f"online-{group_id}-{rollout_index}"
            body = copy.deepcopy(plan["request_template"])
            body.update({
                "model": plan["behavior_policy"]["model_id"],
                "messages": copy.deepcopy(group["messages"]), "stream": True,
                "session_id": session_id, "max_tokens": plan["sampling"]["max_tokens"],
                "temperature": plan["sampling"]["temperature"],
                "top_p": plan["sampling"]["top_p"], "seed": seed,
            })
            request_hash = canonical_hash(body)
            rollout_id = canonical_hash({
                "group_id": group_id, "rollout_index": rollout_index, "seed": seed,
                "request_sha256": request_hash,
                "behavior_policy_sha256": behavior_hash,
            })[:32]
            item = {
                "schedule_index": len(schedule), "group_index": group_index,
                "rollout_index": rollout_index, "group_id": group_id,
                "rollout_id": rollout_id, "seed": seed, "session_id": session_id,
                "prompt_id": group["prompt_id"], "body": body,
                "request_sha256": request_hash,
                "expected_visual_modalities": group["expected_visual_modalities"],
            }
            schedule.append(item)
            rollouts.append({"rollout_id": rollout_id,
                             "rollout_index": rollout_index, "seed": seed,
                             "request_sha256": request_hash})
        schedule_identity.append({
            "group_id": group_id, "group_index": group_index,
            "prompt_id": group["prompt_id"], "rollouts": rollouts,
        })
    return schedule, canonical_hash(schedule_identity)


def expected_attestation(item, plan, coverage):
    response = {
        "http_status": 200, "contract": "1", "generation": "fresh",
        "batch_id": plan["batch_id"], "group_id": item["group_id"],
        "rollout_id": item["rollout_id"],
        "rollout_index": str(item["rollout_index"]),
        "attempt": "0", "request_sha256": item["request_sha256"],
        "behavior_policy_sha256": canonical_hash(plan["behavior_policy"]),
        "behavior_policy_revision": plan["behavior_policy"]["revision"],
        "model_id": plan["behavior_policy"]["model_id"],
        "model_revision": plan["behavior_policy"]["model_revision"],
        "model_sha256": plan["behavior_policy"]["model_sha256"],
        "model_config_sha256": plan["behavior_policy"]["model_config_sha256"],
        "sampling_sha256": canonical_hash(plan["sampling"]),
        "behavior_evidence": coverage,
    }
    response["attestation_sha256"] = canonical_hash(response)
    return response


def verify_cas_descriptor(root, descriptor, path):
    require(isinstance(descriptor, dict), "INVALID_CAS_DESCRIPTOR", path)
    ref = descriptor.get("ref")
    require(isinstance(ref, str) and PurePosixPath(ref).name == ref,
            "NON_BASENAME_CAS_REF", path + ".ref")
    digest = descriptor.get("sha256")
    require_sha256(digest, path + ".sha256")
    require(ref.startswith(digest + "."), "CAS_NAME_HASH_MISMATCH", path + ".ref")
    raw = read_file(root, ref)
    same(sha256_bytes(raw), digest, "CAS_HASH_MISMATCH", path + ".ref")
    if descriptor.get("bytes") is not None:
        same(descriptor["bytes"], len(raw), "CAS_LENGTH_MISMATCH", path + ".bytes")
    return raw


def verify_record_artifacts(record, replay, artifact_root, path):
    """Rebuild base64 SSE artifacts and verify normalized CAS descriptors."""
    by_id = {}
    for event_index, event in enumerate(record["events"]):
        if event.get("type") != "artifact":
            continue
        source_index = event["source_event_index"]
        unified = replay["frames"][source_index]["value"]["unified"]
        raw_artifact = unified.get("artifact")
        require(isinstance(raw_artifact, dict), "MISSING_RAW_ARTIFACT",
                f"{path}.events[{event_index}].artifact")
        artifact_id = raw_artifact.get("artifact_id")
        require(isinstance(artifact_id, str) and artifact_id,
                "INVALID_ARTIFACT_ID", f"{path}.events[{event_index}]")
        try:
            payload = base64.b64decode(raw_artifact.get("data", ""), validate=True)
        except (ValueError, TypeError) as exc:
            raise AuditInputError("INVALID_ARTIFACT_BASE64",
                                  f"{path}.events[{event_index}]", str(exc)) from exc
        normalized_artifact = event.get("artifact")
        require(isinstance(normalized_artifact, dict), "MISSING_NORMALIZED_ARTIFACT",
                f"{path}.events[{event_index}].artifact")
        same(normalized_artifact.get("chunk_bytes"), len(payload),
             "ARTIFACT_CHUNK_LENGTH", f"{path}.events[{event_index}].chunk_bytes")
        same(normalized_artifact.get("chunk_sha256"), sha256_bytes(payload),
             "ARTIFACT_CHUNK_HASH", f"{path}.events[{event_index}].chunk_sha256")
        if raw_artifact.get("chunk_bytes") is not None:
            same(raw_artifact["chunk_bytes"], len(payload), "RAW_ARTIFACT_CHUNK_LENGTH",
                 f"{path}.events[{event_index}].chunk_bytes")
        if raw_artifact.get("chunk_sha256") is not None:
            same(raw_artifact["chunk_sha256"], sha256_bytes(payload),
                 "RAW_ARTIFACT_CHUNK_HASH", f"{path}.events[{event_index}].chunk_sha256")
        chunk_index = raw_artifact.get("index")
        require(integer(chunk_index) and chunk_index >= 0,
                "INVALID_ARTIFACT_INDEX", f"{path}.events[{event_index}]")
        group = by_id.setdefault(artifact_id, [])
        require(all(chunk_index != old[0] for old in group),
                "DUPLICATE_ARTIFACT_CHUNK", f"{path}.{artifact_id}")
        group.append((chunk_index, payload, raw_artifact))
    outputs = {}
    for artifact_id, chunks in by_id.items():
        chunks.sort(key=lambda item: item[0])
        first = chunks[0][2]
        total = first.get("total")
        require(integer(total) and total == len(chunks) and total > 0,
                "INCOMPLETE_ARTIFACT", f"{path}.{artifact_id}")
        same([index for index, _payload, _raw in chunks], list(range(total)),
             "ARTIFACT_CHUNK_ORDER", f"{path}.{artifact_id}")
        payload = b"".join(item[1] for item in chunks)
        digest = sha256_bytes(payload)
        final = chunks[-1][2]
        same(final.get("sha256"), digest, "ARTIFACT_FINAL_HASH",
             f"{path}.{artifact_id}")
        same(final.get("bytes"), len(payload), "ARTIFACT_FINAL_LENGTH",
             f"{path}.{artifact_id}")
        for _index, _chunk, raw in chunks[1:]:
            for key in ("artifact_id", "media_type", "mime_type", "format", "total"):
                same(raw.get(key), first.get(key), "ARTIFACT_METADATA_DRIFT",
                     f"{path}.{artifact_id}.{key}")
        chain = sha256_bytes(b"".join(bytes.fromhex(sha256_bytes(item[1]))
                                      for item in chunks))
        outputs[artifact_id] = (first, payload, chain)
    descriptors = record.get("artifacts")
    require(isinstance(descriptors, list), "MISSING_ARTIFACT_LIST", path + ".artifacts")
    same(len(descriptors), len(outputs), "ARTIFACT_DESCRIPTOR_COUNT",
         path + ".artifacts")
    seen = set()
    for index, descriptor in enumerate(descriptors):
        artifact_id = descriptor.get("artifact_id")
        require(artifact_id in outputs and artifact_id not in seen,
                "ARTIFACT_DESCRIPTOR_MISMATCH", f"{path}.artifacts[{index}]")
        seen.add(artifact_id)
        _first, payload, chain = outputs[artifact_id]
        final = by_id[artifact_id][-1][2]
        raw = verify_cas_descriptor(artifact_root, descriptor,
                                    f"{path}.artifacts[{index}]")
        same(raw, payload, "ARTIFACT_SSE_CAS_MISMATCH",
             f"{path}.artifacts[{index}]")
        same(descriptor.get("chunk_count"), len(by_id[artifact_id]),
             "ARTIFACT_DESCRIPTOR_CHUNKS", f"{path}.artifacts[{index}]")
        same(descriptor.get("chunks_sha256"), chain, "ARTIFACT_CHUNK_CHAIN",
             f"{path}.artifacts[{index}]")
        for key in ("media_type", "mime_type", "format", "bytes", "sha256"):
            same(descriptor.get(key), final.get(key), "ARTIFACT_DESCRIPTOR_FIELD",
                 f"{path}.artifacts[{index}].{key}")
    for input_index, descriptor in enumerate(record.get("inputs", [])):
        if descriptor.get("ref") is not None:
            verify_cas_descriptor(artifact_root, descriptor,
                                  f"{path}.inputs[{input_index}]")


def narrative_reward(record):
    expected = record["ablation"]["expected_visual_modalities"]
    predicted = [slot["media_type"] for slot in record.get("visual_slots", [])]
    tp = len(set(predicted) & set(expected))
    fp = len(set(predicted) - set(expected))
    fn = len(set(expected) - set(predicted))
    denominator = 2 * tp + fp + fn
    f1 = 1.0 if denominator == 0 else 2 * tp / denominator
    matched = sum(min(predicted.count(kind), expected.count(kind))
                  for kind in ("image", "video"))
    redundant = len(predicted) - matched
    redundancy = redundant / len(predicted) if predicted else 0.0
    return {"schema_version": 1,
            "name": "narrative-routing-f1-minus-redundancy-v1",
            "score": float(f1 - redundancy)}


def verify_online_record(record, item, plan, schedule_hash, artifact_root, path):
    require(isinstance(record, dict) and record.get("schema_version") == 2,
            "OFFLINE_OR_LEGACY_RECORD", path)
    same(record.get("session_id"), item["session_id"], "STALE_SESSION", path)
    same(record.get("turn_index"), item["schedule_index"], "STALE_TURN", path)
    same(record.get("endpoint"), "/v1/chat/completions", "ENDPOINT_MISMATCH", path)
    same(record.get("terminal_state"), "completed", "CANCELLED_OR_FAILED_RECORD", path)
    same(record.get("error"), None, "FAILED_RECORD", path + ".error")
    ablation = record.get("ablation")
    require(isinstance(ablation, dict), "MISSING_ABLATION_BINDING", path)
    same(ablation.get("prompt_id"), item["prompt_id"], "CROSS_GROUP_PROMPT",
         path + ".ablation.prompt_id")
    same(ablation.get("expected_visual_modalities"),
         item["expected_visual_modalities"], "REWARD_LABEL_SUBSTITUTION",
         path + ".ablation.expected_visual_modalities")
    replay = verify_raw_replay(record, artifact_root, path)
    for frame in replay["frames"]:
        value = frame["value"]
        if value is None:
            continue
        same(value.get("id"), record.get("request_id"), "MIXED_RAW_REQUEST_ID",
             f"{path}.frame[{frame['index']}].id")
        same(value.get("model"), plan["behavior_policy"]["model_id"],
             "MODEL_DRIFT", f"{path}.frame[{frame['index']}].model")
        unified = value.get("unified")
        if isinstance(unified, dict):
            if "request_id" in unified:
                same(unified["request_id"], record["request_id"],
                     "MIXED_INNER_REQUEST_ID", f"{path}.frame[{frame['index']}]")
            if "session_id" in unified:
                same(unified["session_id"], item["session_id"],
                     "MIXED_INNER_SESSION_ID", f"{path}.frame[{frame['index']}]")
    final = replay["frames"][-2]["value"]
    require(isinstance(final, dict), "MISSING_FINAL_EVENT", path)
    choices = final.get("choices")
    require(isinstance(choices, list) and len(choices) == 1
            and choices[0].get("finish_reason") == "stop",
            "TRUNCATED_OR_UNFINISHED_RECORD", path + ".finish_reason")
    output = "".join(event.get("text", "") for event in record["events"]
                     if event.get("type") in ("asr_final", "vlm_delta"))
    clean = "".join(event.get("text", "") for event in record["events"]
                    if event.get("type") == "vlm_delta")
    same(record.get("output_text"), output, "OUTPUT_TEXT_MISMATCH", path)
    same(record.get("clean_text"), clean, "CLEAN_TEXT_MISMATCH", path)
    same(record.get("output_sha256"), sha256_bytes(output.encode("utf-8")),
         "OUTPUT_HASH_MISMATCH", path)
    same(record.get("clean_text_sha256"), sha256_bytes(clean.encode("utf-8")),
         "CLEAN_HASH_MISMATCH", path)
    verify_record_artifacts(record, replay, artifact_root, path)
    provenance = record.get("rollout_provenance")
    require(isinstance(provenance, dict), "OFFLINE_ONLINE_PROVENANCE_MIX",
            path + ".rollout_provenance")
    expected_provenance = {
        "contract": PROVENANCE_CONTRACT, "mode": "online",
        "batch_id": plan["batch_id"], "plan_sha256": canonical_hash(plan),
        "schedule_sha256": schedule_hash, "group_id": item["group_id"],
        "group_index": item["group_index"], "rollout_id": item["rollout_id"],
        "rollout_index": item["rollout_index"], "attempt": 0, "seed": item["seed"],
        "request_sha256": item["request_sha256"],
        "behavior_policy": plan["behavior_policy"],
        "behavior_policy_sha256": canonical_hash(plan["behavior_policy"]),
        "sampling_sha256": canonical_hash(plan["sampling"]),
    }
    for key, expected in expected_provenance.items():
        same(provenance.get(key), expected, "STALE_OR_CROSS_GROUP_PROVENANCE",
             f"{path}.rollout_provenance.{key}")
    same(set(provenance), set(expected_provenance) | {"response", "behavior"},
         "PROVENANCE_FIELDS", path + ".rollout_provenance")
    response = provenance.get("response")
    same(response, expected_attestation(item, plan, "exposed"),
         "ATTESTATION_MISMATCH", path + ".rollout_provenance.response")
    raw_behavior = behavior_from_raw(replay, path + ".behavior")
    summary = {
        "status": "COVERED", **raw_behavior,
        "token_count": len(raw_behavior["token_ids"]),
        "masked_token_count": sum(raw_behavior["mask"]),
        "evidence_sha256": canonical_hash(raw_behavior),
    }
    same(provenance.get("behavior"), summary, "BEHAVIOR_SUMMARY_MISMATCH",
         path + ".rollout_provenance.behavior")
    same(record.get("reward"), narrative_reward(record), "REWARD_MISMATCH",
         path + ".reward")
    return {"record": record, "item": item, "replay": replay,
            "behavior": raw_behavior, "behavior_summary": summary,
            "reward": summary and narrative_reward(record)}


def _candidate_tie(candidate):
    source = candidate["provenance"]
    turn = source.get("source_turn_index")
    return (candidate["prompt_sha256"], candidate["target_sha256"],
            str(source.get("source_session_id") or ""),
            str(source.get("source_request_id") or ""),
            turn if integer(turn) else -1)


def candidate_from_record(audited):
    record = audited["record"]
    prompt_parts = []
    for item in record["inputs"]:
        if item.get("kind") == "text":
            prompt_parts.append({"kind": "text", "text": item["text"],
                                 "sha256": item["sha256"]})
        else:
            prompt_parts.append(copy.deepcopy(item))
    prompt_sha = canonical_hash(prompt_parts)
    prompt_text = "\n".join(item["text"] for item in prompt_parts
                            if item.get("kind") == "text")
    clean = record["clean_text"]
    clean_bytes = clean.encode("utf-8")
    slots = sorted(record.get("visual_slots", []),
                   key=lambda slot: (slot.get("text_offset", 0),
                                     slot.get("slot_index", 0)))
    artifacts_by_slot = {}
    for artifact in record.get("artifacts", []):
        if artifact.get("slot_id") is not None:
            artifacts_by_slot[artifact["slot_id"]] = artifact
    output_slots = []
    for slot in slots:
        kind = slot["media_type"]
        require(kind in ("image", "video"), "INVALID_VISUAL_ROUTE", "visual_slot")
        prompt = " ".join(str(slot.get("prompt", "")).split())
        marker = f"[[{kind}: {prompt}]]"
        if kind == "video" and slot.get("anchor_explicit") is True:
            anchor = slot.get("anchor_slot_index")
            require(integer(anchor) and 0 <= anchor < slot["slot_index"],
                    "INVALID_EXPLICIT_ANCHOR", "visual_slot.anchor_slot_index")
            marker = f"[[video: {prompt} | source=slot-{anchor}]]"
        offset = slot.get("text_offset", 0)
        require(integer(offset), "INVALID_VISUAL_OFFSET", "visual_slot.text_offset")
        enriched = {key: copy.deepcopy(value) for key, value in slot.items()
                    if key not in {"seq", "server_seq", "stage"}}
        enriched["marker"] = marker
        artifact = artifacts_by_slot.get(slot.get("slot_id"))
        enriched["leg_state"] = "completed" if artifact is not None else "missing"
        if artifact is not None:
            enriched.update({"artifact_id": artifact["artifact_id"],
                             "artifact_ref": artifact["ref"],
                             "artifact_sha256": artifact["sha256"]})
            if artifact.get("anchor_sha256") is not None:
                enriched["anchor_sha256"] = artifact["anchor_sha256"]
        output_slots.append(enriched)
    # Insert in stable (offset, slot index) order. Each marker shifts only later
    # source offsets; snap continuation bytes exactly as the packer does.
    output_bytes = clean_bytes
    inserted = 0
    for slot in output_slots:
        offset = max(0, min(len(clean_bytes), slot.get("text_offset", 0)))
        while offset < len(clean_bytes) and clean_bytes[offset] & 0xC0 == 0x80:
            offset += 1
        at = offset + inserted
        marker_bytes = slot["marker"].encode("utf-8")
        output_bytes = output_bytes[:at] + marker_bytes + output_bytes[at:]
        inserted += len(marker_bytes)
    target = output_bytes.decode("utf-8")
    media_ref_keys = {
        "artifact_id", "bytes", "chunk_count", "chunks_sha256", "media_type",
        "mime_type", "ref", "sha256", "slot_id", "slot_index",
        "anchor_explicit", "anchor_slot_index", "anchor_sha256",
    }
    media_refs = [{key: copy.deepcopy(value) for key, value in artifact.items()
                   if key in media_ref_keys}
                  for artifact in record.get("artifacts", [])]
    for ref in media_refs:
        ref["verified_on_disk"] = True
    input_media = []
    for item in prompt_parts:
        if item.get("ref") is not None:
            item = copy.deepcopy(item)
            item["verified_on_disk"] = True
            input_media.append(item)
    provenance = {
        "source_schema_version": record["schema_version"],
        "source_session_id": record["session_id"],
        "source_request_id": record["request_id"],
        "source_turn_index": record["turn_index"],
        "source_endpoint": record["endpoint"],
        "source_prompt_id": record["ablation"]["prompt_id"],
    }
    return {
        "prompt_sha256": prompt_sha, "prompt_text": prompt_text,
        "prompt_parts": prompt_parts, "input_media_refs": input_media,
        "assistant_target": target, "target_sha256": sha256_bytes(
            target.encode("utf-8")), "clean_text": clean,
        "visual_slots": output_slots, "modality_route": [
            slot["media_type"] for slot in output_slots],
        "output_media_refs": media_refs, "provenance": provenance,
        "score": audited["reward"]["score"], "audited": audited,
    }


def _split_for(plan, prompt_sha):
    ratio = float(plan["eval_ratio"])
    if ratio <= 0.0:
        return "train"
    if ratio >= 1.0:
        return "eval"
    digest = sha256_bytes((str(plan["split_seed"]) + prompt_sha).encode("utf-8"))
    return "eval" if int(digest[:16], 16) / float(1 << 64) < ratio else "train"


def build_groups(plan, schedule_hash, records):
    """Reconstruct GRPO/DPO rows from verified records in deterministic order."""
    by_group = {index: [] for index in range(len(plan["groups"]))}
    for audited in records:
        by_group[audited["item"]["group_index"]].append(candidate_from_record(audited))
    groups = []
    pairs = []
    skipped_low_margin = 0
    for group_index, candidates in by_group.items():
        same(len(candidates), len(plan["groups"][group_index]["seeds"]),
             "MISSING_GROUP_MEMBER", f"group[{group_index}]")
        candidates.sort(key=lambda item: (-item["score"], _candidate_tie(item)))
        first = candidates[0]
        reward_name = "narrative-routing-f1-minus-redundancy-v1"
        ordered_targets = [item["target_sha256"] for item in candidates]
        group_id = sha256_bytes((first["prompt_sha256"] + ":" + reward_name + ":"
                                 + ",".join(ordered_targets)).encode("utf-8"))[:32]
        scores = [item["score"] for item in candidates]
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / len(scores)
        std = math.sqrt(variance)
        split = _split_for(plan, first["prompt_sha256"])
        item0 = candidates[0]["audited"]["item"]
        group = {
            "record_type": "grpo_group", "schema_version": 1,
            "group_id": group_id, "prompt_sha256": first["prompt_sha256"],
            "prompt_text": first["prompt_text"], "prompt_parts": first["prompt_parts"],
            "input_media_refs": first["input_media_refs"], "reward_name": reward_name,
            "group_size": len(candidates), "group_mean": mean, "group_std": std,
            "has_reward_variance": std > 0.0, "split": split,
            "rollout_provenance": {
                "contract": PROVENANCE_CONTRACT, "mode": "online",
                "batch_id": plan["batch_id"], "plan_sha256": canonical_hash(plan),
                "schedule_sha256": schedule_hash,
                "schedule_group_id": item0["group_id"], "group_index": group_index,
                "behavior_policy": plan["behavior_policy"],
                "behavior_policy_sha256": canonical_hash(plan["behavior_policy"]),
                "sampling_sha256": canonical_hash(plan["sampling"]),
            },
        }
        rollouts = []
        for rank, candidate in enumerate(candidates):
            score = candidate["score"]
            member = {key: copy.deepcopy(candidate[key]) for key in (
                "assistant_target", "clean_text", "modality_route",
                "output_media_refs", "provenance", "score", "target_sha256",
                "visual_slots")}
            member["rank"] = rank
            member["advantage"] = (0.0 if std == 0.0 else (score - mean) / std)
            member["rollout_provenance"] = copy.deepcopy(
                candidate["audited"]["record"]["rollout_provenance"])
            rollouts.append(member)
        group["rollouts"] = rollouts
        groups.append((group_index, group))
        unique = []
        targets = set()
        for candidate in candidates:
            if candidate["target_sha256"] not in targets:
                unique.append(candidate)
                targets.add(candidate["target_sha256"])
        for chosen_index, chosen in enumerate(unique):
            for rejected in unique[chosen_index + 1:]:
                margin = chosen["score"] - rejected["score"]
                if not margin + 1e-12 >= 1e-6:
                    skipped_low_margin += 1
                    continue
                def preference_side(candidate):
                    keys = ("assistant_target", "clean_text", "modality_route",
                            "output_media_refs", "provenance", "score",
                            "target_sha256", "visual_slots")
                    result = {key: copy.deepcopy(candidate[key]) for key in keys}
                    source_record = candidate["audited"]["record"]
                    export = source_record.get("token_export")
                    if export is not None:
                        result["token_export"] = copy.deepcopy(export)
                        original = copy.deepcopy(source_record)
                        original.pop("token_export", None)
                        result["token_export_source"] = original
                    return result
                pair_id = sha256_bytes((
                    first["prompt_sha256"] + ":" + reward_name + ":"
                    + chosen["target_sha256"] + ":" + rejected["target_sha256"]
                ).encode("utf-8"))[:32]
                pairs.append((first["prompt_sha256"], {
                    "record_type": "dpo_pair", "schema_version": 1,
                    "pair_id": pair_id, "prompt_sha256": first["prompt_sha256"],
                    "prompt_text": first["prompt_text"],
                    "prompt_parts": first["prompt_parts"],
                    "input_media_refs": first["input_media_refs"],
                    "reward_name": reward_name, "chosen": preference_side(chosen),
                    "rejected": preference_side(rejected), "margin": margin,
                    "split": split,
                }))
    groups.sort(key=lambda pair: (pair[1]["prompt_sha256"], pair[1]["group_id"]))
    pairs.sort(key=lambda pair: (pair[0], -pair[1]["margin"], pair[1]["pair_id"]))
    return groups, pairs, skipped_low_margin


def _jsonl_bytes(rows):
    return b"".join(canonical_bytes(row) + b"\n" for row in rows)


def _output_identity(relative, raw, count):
    return {"path": relative, "sha256": sha256_bytes(raw), "bytes": len(raw),
            "record_count": count}


def audit_online_bundle(root, *, require_exports=None):
    """Independently rebuild all accepted online rollout and pack payloads."""
    root = Path(root)
    plan_raw = read_file(root, "plan.json")
    plan = prepare_plan(strict_json_bytes(plan_raw, "plan.json"))
    require(plan_raw == canonical_bytes(plan) + b"\n", "NON_CANONICAL_PLAN",
            "plan.json")
    schedule, schedule_hash = build_schedule(plan)
    same(plan["reward"]["spec"], "umminfer_ablation_metrics:NarrativeRoutingReward",
         "UNSUPPORTED_INDEPENDENT_REWARD", "plan.reward.spec")
    source_records = read_jsonl(root, "trajectories.jsonl", allow_empty=False)
    same(len(source_records), len(schedule), "ROLLOUT_COUNT_MISMATCH",
         "trajectories.jsonl")
    records = []
    request_ids = set()
    rollout_ids = set()
    for index, (record, item) in enumerate(zip(source_records, schedule)):
        path = f"trajectories[{index}]"
        provenance = record.get("rollout_provenance")
        require(isinstance(provenance, dict)
                and provenance.get("contract") == PROVENANCE_CONTRACT
                and provenance.get("mode") == "online",
                "OFFLINE_ONLINE_PROVENANCE_MIX", f"trajectories[{index}]")
        same(provenance.get("rollout_id"), item["rollout_id"],
             "STALE_REORDERED_OR_MISSING_ROLLOUT", path)
        require(record.get("request_id") not in request_ids,
                "DUPLICATE_REQUEST_ID", path)
        request_ids.add(record.get("request_id"))
        require(item["rollout_id"] not in rollout_ids, "DUPLICATE_ROLLOUT_ID", path)
        rollout_ids.add(item["rollout_id"])
        records.append(verify_online_record(
            record, item, plan, schedule_hash, root / "artifacts", path))
    groups, pairs, skipped = build_groups(plan, schedule_hash, records)
    split_groups = {"train": [], "eval": []}
    for _index, group in groups:
        split_groups[group["split"]].append(group)
    split_pairs = {"train": [], "eval": []}
    for _prompt, pair in pairs:
        split_pairs[pair["split"]].append(pair)
    expected_payloads = {"plan.json": plan_raw}
    for split in ("train", "eval"):
        expected_payloads[f"pack/grpo_{split}.jsonl"] = _jsonl_bytes(split_groups[split])
        expected_payloads[f"pack/dpo_{split}.jsonl"] = _jsonl_bytes(split_pairs[split])
    for split in ("train", "eval"):
        for kind in ("grpo", "dpo"):
            relative = f"pack/{kind}_{split}.jsonl"
            actual = read_file(root, relative)
            same(actual, expected_payloads[relative], "RECONSTRUCTED_PACK_MISMATCH",
                 relative)
    trajectory_raw = read_file(root, "trajectories.jsonl")
    coverage = "COVERED"
    train_prompts = {group["prompt_sha256"] for group in split_groups["train"]}
    eval_prompts = {group["prompt_sha256"] for group in split_groups["eval"]}
    dpo_train_prompts = {pair["prompt_sha256"] for pair in split_pairs["train"]}
    dpo_eval_prompts = {pair["prompt_sha256"] for pair in split_pairs["eval"]}
    outputs = {}
    for kind in ("dpo", "grpo"):
        for split in ("train", "eval"):
            relative = f"{kind}_{split}.jsonl"
            raw = expected_payloads["pack/" + relative]
            outputs[f"{kind}_{split}"] = _output_identity(
                relative, raw, len(split_groups[split] if kind == "grpo"
                                   else split_pairs[split]))
    pack_manifest = {
        "dataset_schema_version": 1, "status": "COMPLETED", "REAL_TRAINING": "NOT_RUN",
        "mode": "preference", "seed": plan["split_seed"],
        "eval_ratio": plan["eval_ratio"], "min_group_size": len(schedule) // len(groups),
        "min_margin": 1e-6, "inputs": [{
            "name": "trajectories.jsonl", "sha256": sha256_bytes(trajectory_raw),
            "bytes": len(trajectory_raw), "line_count": len(records)}],
        "outputs": outputs, "counts": {
            "total_records_read": len(records),
            "accepted_verified_completed": len(records),
            "rejected": {"cancelled": 0, "failed": 0, "unverified": 0,
                         "unscored": 0, "duplicate": 0},
            "rejection_details": {}, "skipped_small_groups": 0,
            "skipped_low_margin_pairs": skipped,
        },
        "artifact_verification": {
            "mode": "verified_on_disk",
            "checked_input_media_refs": sum(
                1 for audited in records for item in audited["record"]["inputs"]
                if item.get("ref") is not None),
            "checked_output_media_refs": sum(
                len(audited["record"]["artifacts"]) for audited in records),
        },
        "reward": {"spec": plan["reward"]["spec"],
                   "name": "narrative-routing-f1-minus-redundancy-v1"},
        "split_audit": {
            "train_prompt_count": len(train_prompts),
            "eval_prompt_count": len(eval_prompts),
            "overlap_prompt_sha256s": sorted(train_prompts & eval_prompts),
            "leakage_count": len(train_prompts & eval_prompts), "passed": True,
            "packs": {
                "dpo": {"train_prompt_count": len(dpo_train_prompts),
                        "eval_prompt_count": len(dpo_eval_prompts),
                        "overlap_prompt_sha256s": sorted(dpo_train_prompts & dpo_eval_prompts),
                        "leakage_count": len(dpo_train_prompts & dpo_eval_prompts),
                        "passed": True},
                "grpo": {"train_prompt_count": len(train_prompts),
                         "eval_prompt_count": len(eval_prompts),
                         "overlap_prompt_sha256s": sorted(train_prompts & eval_prompts),
                         "leakage_count": len(train_prompts & eval_prompts),
                         "passed": True},
            },
        },
        "provenance_mode": "online",
        "rollout_provenance": {
            "contract": PROVENANCE_CONTRACT, "plan_sha256": canonical_hash(plan),
            "schedule_sha256": schedule_hash,
            "behavior_policy_sha256": canonical_hash(plan["behavior_policy"]),
            "sampling_sha256": canonical_hash(plan["sampling"]),
            "behavior_logprobs": coverage,
        },
        "claims": {
            "REAL_TRAINING": "NOT_RUN", "optimizer": "NOT_RUN",
            "model_weights_updated": False, "model_quality": "NOT_RUN",
            "model_identity_verification": "ENDPOINT_ATTESTATION_ONLY",
            "online_rollout_export": "VALIDATED", "behavior_logprobs": coverage,
        },
    }
    actual_pack = read_json(root, "pack/manifest.json", canonical=True)
    same(actual_pack, pack_manifest, "PACK_MANIFEST_MISMATCH", "pack/manifest.json")
    verify_sidecar(root, "pack/manifest.json")
    pack_raw = read_file(root, "pack/manifest.json")
    rollout_manifest = {
        "schema": "umminfer-online-grpo-export-v1", "status": "PASS",
        "provenance_mode": "online", "plan_sha256": canonical_hash(plan),
        "schedule_sha256": schedule_hash, "behavior_policy": plan["behavior_policy"],
        "behavior_policy_sha256": canonical_hash(plan["behavior_policy"]),
        "sampling_sha256": canonical_hash(plan["sampling"]),
        "reward": {"spec": plan["reward"]["spec"],
                   "name": "narrative-routing-f1-minus-redundancy-v1"},
        "group_count": len(groups), "rollout_count": len(records),
        "group_size": len(schedule) // len(groups), "behavior_logprobs": coverage,
        "trajectory": {"path": "trajectories.jsonl",
                       "sha256": sha256_bytes(trajectory_raw),
                       "bytes": len(trajectory_raw), "record_count": len(records)},
        "pack_manifest": {"path": "pack/manifest.json",
                          "sha256": sha256_bytes(pack_raw), "bytes": len(pack_raw)},
        "claims": {"REAL_TRAINING": "NOT_RUN", "optimizer": "NOT_RUN",
                   "model_quality": "NOT_RUN", "generalized_improvement": "NOT_RUN",
                   "model_identity_verification": "ENDPOINT_ATTESTATION_ONLY",
                   "behavior_logprobs": coverage},
    }
    actual_rollout = read_json(root, "rollout_manifest.json", canonical=True)
    same(actual_rollout, rollout_manifest, "ROLLOUT_MANIFEST_MISMATCH",
         "rollout_manifest.json")
    verify_sidecar(root, "rollout_manifest.json")
    expected_payloads.update({
        "trajectories.jsonl": trajectory_raw,
        "rollout_manifest.json": read_file(root, "rollout_manifest.json"),
        "rollout_manifest.json.sha256": read_file(root, "rollout_manifest.json.sha256"),
        "pack/manifest.json": pack_raw,
        "pack/manifest.json.sha256": read_file(root, "pack/manifest.json.sha256"),
    })
    for audited in records:
        record = audited["record"]
        expected_payloads["artifacts/" + record["sse_replay"]["raw_ref"]] = read_file(
            root, "artifacts/" + record["sse_replay"]["raw_ref"])
        for descriptor in record["inputs"] + record["artifacts"]:
            if descriptor.get("ref") is not None:
                relative = "artifacts/" + descriptor["ref"]
                expected_payloads[relative] = read_file(root, relative)
    actual_files = sorted(path.relative_to(root).as_posix()
                          for path in root.rglob("*") if path.is_file()
                          and path.name != "attempts.jsonl")
    same(actual_files, sorted(expected_payloads), "ACCEPTED_FILE_SET_MISMATCH",
         "source_bundle")
    source_files = [file_identity(root, relative) for relative in sorted(expected_payloads)]
    export_presence = ["token_export" in audited["record"] for audited in records]
    if require_exports is not None:
        same(all(export_presence), require_exports, "TOKEN_EXPORT_COVERAGE",
             "trajectories")
    return {
        "root": root, "plan": plan, "plan_raw": plan_raw,
        "plan_sha256": canonical_hash(plan), "schedule": schedule,
        "schedule_sha256": schedule_hash, "records": records,
        "groups": groups, "pairs": pairs, "source_files": source_files,
        "source_file_set_sha256": canonical_hash(source_files),
        "rollout_manifest_sha256": sha256_bytes(read_file(root, "rollout_manifest.json")),
        "pack_manifest_sha256": sha256_bytes(pack_raw),
        "group_count": len(groups), "rollout_count": len(records),
        "behavior_logprobs": coverage, "claims": rollout_manifest["claims"],
    }


VISUAL_BEAT_RE = re.compile(r"(\[\[(image|video):[ \t]*)(.*?)(\]\])", re.DOTALL)
SPOKEN_TAG_RE = re.compile(r"<spoken(?:\s+[^>]*)?>.*?</spoken>", re.DOTALL)
SFT_SPAN_WEIGHTS = {
    "narrative_prose": 1.0, "visual_route_tag": 1.5,
    "visual_prompt_body": 1.0, "spoken_style": 1.0,
    "image_form_multiplier": 1.0, "video_form_multiplier": 1.0,
}
EXPORT_POLICY_V1 = {
    "add_special_tokens": False, "padding": "none", "truncation": "none",
    "unknown_tokens": "reject", "offset_unit": "utf8_byte",
    "offset_interval": "half_open", "offset_continuity": "exact_no_gaps_or_overlaps",
    "normalization": "source_bytes_must_equal_token_decode",
    "decode": "per_token_skip_special_false_cleanup_false",
    "sft_mask": "joint_full_codepoint_overlap_weighted_v1",
}
EXPORT_POLICY_V2 = {
    "add_special_tokens": False, "padding": "none", "truncation": "none",
    "unknown_tokens": "reject", "offset_unit": "utf8_byte_and_original_codepoint",
    "offset_interval": "half_open",
    "offset_continuity": "normalized_exact_no_gaps_source_units_may_overlap",
    "normalization": "nfc_with_exact_source_units_v2",
    "decode": "bytelevel_token_bytes_and_full_sequence_no_cleanup_v2",
    "tokenizer_compatibility": "bytelevel_bpe_with_exact_nfc_only",
    "sft_mask": "joint_full_raw_offset_codepoint_overlap_weighted_v1",
}


def _char_byte_map(text):
    result = [0]
    for char in text:
        result.append(result[-1] + len(char.encode("utf-8")))
    return result


def _clean_segments(record):
    events = record["events"]
    parts = [event for event in events
             if event.get("type") in ("vlm_delta", "output_delta")]
    if not parts:
        terminal = events[-1]
        if terminal.get("type") == "complete" and not any(
                event.get("type") == "asr_final" for event in events):
            parts = [terminal]
    result = []
    cursor = 0
    for event in parts:
        text = event.get("text", "")
        require(isinstance(text, str), "INVALID_CLEAN_SOURCE_EVENT", "events")
        raw = text.encode("utf-8")
        if raw:
            result.append({"text": text, "start": cursor, "end": cursor + len(raw),
                           "event_index": event["seq"]})
            cursor += len(raw)
    same("".join(item["text"] for item in result), record["clean_text"],
         "CLEAN_SOURCE_RECONSTRUCTION", "events")
    return result


def _slice_segments(segments, start, end):
    result = []
    for segment in segments:
        left = max(start, segment["start"])
        right = min(end, segment["end"])
        if left >= right:
            continue
        raw = segment["text"].encode("utf-8")
        result.append({"text": raw[left - segment["start"]:right - segment["start"]].decode(
            "utf-8"), "event_index": segment["event_index"]})
    same(sum(len(item["text"].encode("utf-8")) for item in result), end - start,
         "CLEAN_SOURCE_GAP", "segments")
    return result


def reconstruct_target(record):
    clean_segments = _clean_segments(record)
    clean = record["clean_text"]
    raw = clean.encode("utf-8")

    def snap(value):
        require(integer(value) and value >= 0, "INVALID_VISUAL_OFFSET", "text_offset")
        value = min(value, len(raw))
        while value < len(raw) and raw[value] & 0xC0 == 0x80:
            value += 1
        return value

    slots = sorted(record.get("visual_slots") or [],
                   key=lambda slot: (snap(slot.get("text_offset")), slot.get("slot_index")))
    legs = {leg.get("slot_id"): leg for leg in record.get("terminal_legs") or []
            if isinstance(leg, dict)}
    artifacts = {}
    for artifact in record.get("artifacts") or []:
        if artifact.get("slot_id") is not None:
            artifacts.setdefault(artifact["slot_id"], artifact)
    pieces = []
    cursor = 0
    for slot in slots:
        offset = snap(slot.get("text_offset"))
        require(offset >= cursor, "OVERLAPPING_VISUAL_SLOTS", "visual_slots")
        pieces.extend(_slice_segments(clean_segments, cursor, offset))
        cursor = offset
        prompt = " ".join(slot.get("prompt", "").split())
        leg = legs.get(slot.get("slot_id"), {})
        marker = ""
        if prompt or leg.get("state") == "completed":
            require(slot.get("media_type") in ("image", "video"),
                    "INVALID_VISUAL_MEDIA", "visual_slot")
            require(bool(prompt), "EMPTY_COMPLETED_VISUAL_PROMPT", "visual_slot")
            marker = f"[[{slot['media_type']}: {prompt}]]"
        bound = artifacts.get(slot.get("slot_id"))
        if bound is not None and bound.get("anchor_explicit", False):
            anchor = bound.get("anchor_slot_index")
            require(slot.get("media_type") == "video"
                    and anchor == slot.get("anchor_slot_index") and integer(anchor)
                    and 0 <= anchor < slot.get("slot_index")
                    and isinstance(bound.get("anchor_sha256"), str)
                    and HEX64.fullmatch(bound["anchor_sha256"]) is not None,
                    "INVALID_EXPLICIT_ANCHOR", "visual_slot")
            require(leg.get("anchor_slot_index", anchor) == anchor
                    and leg.get("anchor_explicit", True) is True,
                    "ANCHOR_LEG_MISMATCH", "visual_slot")
            source_slot = next((item for item in slots
                                if item.get("slot_index") == anchor), None)
            require(source_slot is not None and source_slot.get("media_type") == "image",
                    "ANCHOR_SOURCE_NOT_EARLIER_IMAGE", "visual_slot")
            source_artifact = artifacts.get(source_slot.get("slot_id"))
            require(source_artifact is not None
                    and source_artifact.get("sha256") == bound["anchor_sha256"],
                    "ANCHOR_CAS_MISMATCH", "visual_slot")
            marker = marker[:-2] + f" | source=slot-{anchor}]]"
        if marker:
            event_index = slot.get("seq")
            require(integer(event_index) and 0 <= event_index < len(record["events"])
                    and record["events"][event_index].get("type") == "visual_slot"
                    and record["events"][event_index].get("slot_id") == slot.get("slot_id"),
                    "VISUAL_MARKER_SOURCE_MISMATCH", "visual_slot")
            pieces.append({"text": marker, "event_index": event_index})
    pieces.extend(_slice_segments(clean_segments, cursor, len(raw)))
    segments = []
    cursor = 0
    for piece in pieces:
        size = len(piece["text"].encode("utf-8"))
        segments.append({"text": piece["text"], "start": cursor,
                         "end": cursor + size, "event_index": piece["event_index"]})
        cursor += size
    return "".join(piece["text"] for piece in pieces), segments


def _new_span(spans, span_type, modality, text, char_start, char_end, char_to_byte):
    spans.append({
        "span_type": span_type, "modality": modality,
        "start": char_to_byte[char_start], "end": char_to_byte[char_end],
        "char_start": char_start, "char_end": char_end,
        "text_sha256": sha256_bytes(text[char_start:char_end].encode("utf-8")),
    })


def _emit_visual_range(spans, text, char_to_byte, start, end, default_type, modality):
    local = text[start:end]
    cursor = 0
    for match in VISUAL_BEAT_RE.finditer(local):
        if match.start() > cursor:
            _new_span(spans, default_type, modality, text,
                      start + cursor, start + match.start(), char_to_byte)
        media = match.group(2)
        _new_span(spans, "visual_route_tag", media, text,
                  start + match.start(1), start + match.end(1), char_to_byte)
        if match.group(3):
            _new_span(spans, "visual_prompt_body", media, text,
                      start + match.start(3), start + match.end(3), char_to_byte)
        _new_span(spans, "visual_route_tag", media, text,
                  start + match.start(4), start + match.end(4), char_to_byte)
        cursor = match.end()
    if cursor < len(local):
        _new_span(spans, default_type, modality, text,
                  start + cursor, start + len(local), char_to_byte)


def parse_spans(text):
    char_to_byte = _char_byte_map(text)
    spans = []
    cursor = 0
    for match in SPOKEN_TAG_RE.finditer(text):
        if match.start() > cursor:
            _emit_visual_range(spans, text, char_to_byte, cursor, match.start(),
                               "narrative_prose", "text")
        _emit_visual_range(spans, text, char_to_byte, match.start(), match.end(),
                           "spoken_style", "audio")
        cursor = match.end()
    if cursor < len(text):
        _emit_visual_range(spans, text, char_to_byte, cursor, len(text),
                           "narrative_prose", "text")
    require(not spans or (spans[0]["start"] == 0
                          and spans[-1]["end"] == len(text.encode("utf-8"))),
            "SFT_SPAN_COVERAGE", "spans")
    for previous, current in zip(spans, spans[1:]):
        same(previous["end"], current["start"], "SFT_SPAN_GAP", "spans")
    return spans


def _source_indices(start, end, segments, events):
    event_indices = []
    for segment in segments:
        if min(end, segment["end"]) <= max(start, segment["start"]):
            continue
        index = segment["event_index"]
        if index not in event_indices:
            event_indices.append(index)
    frames = []
    for index in event_indices:
        require(integer(index) and 0 <= index < len(events)
                and events[index].get("seq") == index,
                "TOKEN_SOURCE_EVENT", "events")
        frame = events[index].get("source_event_index")
        if frame is not None:
            require(integer(frame) and frame >= 0, "TOKEN_SOURCE_FRAME", "events")
            frames.append(frame)
    return event_indices, frames


def _effective_weight(span):
    weight = float(SFT_SPAN_WEIGHTS[span["span_type"]])
    if span["modality"] == "image":
        weight *= float(SFT_SPAN_WEIGHTS["image_form_multiplier"])
    elif span["modality"] == "video":
        weight *= float(SFT_SPAN_WEIGHTS["video_form_multiplier"])
    return weight


def _nfc_units(text):
    normalized_full = unicodedata.normalize("NFC", text)
    ranges = []
    start = 0
    for index, char in enumerate(text):
        if index > start and unicodedata.combining(char) == 0:
            ranges.append((start, index))
            start = index
    if text:
        ranges.append((start, len(text)))
    char_to_byte = _char_byte_map(text)
    units = []
    normalized_parts = []
    cursor = 0
    for index, (char_start, char_end) in enumerate(ranges):
        source = text[char_start:char_end]
        normalized = unicodedata.normalize("NFC", source)
        source_raw = source.encode("utf-8")
        normalized_raw = normalized.encode("utf-8")
        units.append({
            "index": index, "source_start": char_to_byte[char_start],
            "source_end": char_to_byte[char_end], "char_start": char_start,
            "char_end": char_end, "normalized_start": cursor,
            "normalized_end": cursor + len(normalized_raw),
            "source_sha256": sha256_bytes(source_raw),
            "normalized_sha256": sha256_bytes(normalized_raw),
        })
        normalized_parts.append(normalized)
        cursor += len(normalized_raw)
    same("".join(normalized_parts), normalized_full, "CROSS_CLUSTER_NFC", "text")
    return units, normalized_full


def _bytelevel_inverse():
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    codepoints = list(values)
    missing = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            codepoints.append(256 + missing)
            missing += 1
    return {chr(codepoint): value for value, codepoint in zip(values, codepoints)}


BYTELEVEL_INVERSE = _bytelevel_inverse()


def verify_tokenizer_dir(tokenizer_dir, identity, version):
    root = Path(tokenizer_dir)
    require(root.is_dir(), "MISSING_TOKENIZER_DIR", str(root))
    files = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        require(not path.is_symlink(), "TOKENIZER_SYMLINK", str(path))
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            files.append(file_identity(root, relative))
    require(files, "EMPTY_TOKENIZER_DIR", str(root))
    same(identity.get("available_files"), files, "TOKENIZER_FILE_DRIFT",
         "token_export.tokenizer.available_files")
    raw_merges = read_file(root, "merges.txt")
    raw_config = read_file(root, "tokenizer_config.json")
    raw_vocab = read_file(root, "vocab.json")
    rows = []
    for name, raw in (("merges.txt", raw_merges),
                      ("tokenizer_config.json", raw_config), ("vocab.json", raw_vocab)):
        rows.append(name + "\n" + str(len(raw)) + "\n" + sha256_bytes(raw) + "\n")
    loaded = sha256_bytes(("umminfer-qwen38-loaded-tokenizer-v1\n"
                           + "".join(rows)).encode("ascii"))
    generation = strict_json_bytes(read_file(root, "generation_config.json"),
                                   "generation_config.json")
    eos = generation.get("eos_token_id")
    if integer(eos):
        terminal = [eos]
    else:
        terminal = eos
    require(isinstance(terminal, list) and terminal
            and all(integer(value) and value >= 0 for value in terminal)
            and len(terminal) == len(set(terminal)), "INVALID_TERMINAL_IDS",
            "generation_config.eos_token_id")
    terminal = sorted(terminal)
    tokenizer_config = strict_json_bytes(raw_config, "tokenizer_config.json")
    vocab = strict_json_bytes(raw_vocab, "vocab.json")
    eos_token = tokenizer_config.get("eos_token")
    if isinstance(eos_token, dict):
        eos_token = eos_token.get("content")
    added = tokenizer_config.get("added_tokens_decoder", {})
    bound = False
    if isinstance(eos_token, str):
        if isinstance(vocab, dict) and vocab.get(eos_token) in terminal:
            bound = True
        if isinstance(added, dict) and any(
                str(value.get("id")) in {str(item) for item in terminal}
                and value.get("content") == eos_token for value in added.values()):
            bound = True
    require(bound, "TERMINAL_TOKEN_IDENTITY", "tokenizer_config.eos_token")
    if version == 2:
        tokenizer_json = strict_json_bytes(read_file(root, "tokenizer.json"),
                                           "tokenizer.json")
        require(isinstance(tokenizer_json.get("normalizer"), dict)
                and tokenizer_json["normalizer"].get("type") == "NFC"
                and isinstance(tokenizer_json.get("decoder"), dict)
                and tokenizer_json["decoder"].get("type") == "ByteLevel"
                and isinstance(tokenizer_json.get("model"), dict)
                and tokenizer_json["model"].get("type") == "BPE",
                "TOKENIZER_MODEL_DRIFT", "tokenizer.json")

        def bytelevel_ok(value):
            if isinstance(value, dict):
                if value.get("type") == "ByteLevel":
                    return (value.get("add_prefix_space") is False
                            and value.get("trim_offsets") is False
                            and value.get("use_regex") is False)
                return any(bytelevel_ok(item) for item in value.values())
            if isinstance(value, list):
                return any(bytelevel_ok(item) for item in value)
            return False
        require(bytelevel_ok(tokenizer_json.get("pre_tokenizer")),
                "TOKENIZER_PRETOKENIZER_DRIFT", "tokenizer.json")
    return files, loaded, terminal, vocab


def _validate_stream(kind, text, ids, stream, segments, events, identity,
                     version, vocab=None):
    require(isinstance(stream, dict), "MISSING_TOKEN_STREAM", kind)
    raw = text.encode("utf-8")
    normalized_text = unicodedata.normalize("NFC", text) if version == 2 else text
    normalized_raw = normalized_text.encode("utf-8")
    char_to_byte = _char_byte_map(text)
    byte_to_char = {value: index for index, value in enumerate(char_to_byte)}
    spans = parse_spans(text)
    require(isinstance(ids, list) and isinstance(stream.get("tokens"), list)
            and len(ids) == len(stream["tokens"]), "TOKEN_COUNT_MISMATCH", kind)
    require(stream.get("token_ids") == ids, "TOKEN_ID_LIST_MISMATCH", kind)
    tokens = []
    annotations = []
    weights = []
    units = None
    if version == 2:
        units, expected_normalized = _nfc_units(text)
        same(normalized_text, expected_normalized, "NFC_MISMATCH", kind)
    normalized_cursor = 0
    source_cursor = 0
    previous_start = previous_end = 0
    for index, (token_id, entry) in enumerate(zip(ids, stream["tokens"])):
        require(integer(token_id) and 0 <= token_id < identity["token_id_space_size"],
                "TOKEN_ID_OUT_OF_RANGE", f"{kind}[{index}]")
        unknown = identity["unknown_token_id"]
        require(unknown is None or token_id != unknown, "UNKNOWN_SAMPLED_TOKEN",
                f"{kind}[{index}]")
        require(isinstance(entry, dict) and entry.get("index") == index
                and entry.get("token_id") == token_id, "TOKEN_INDEX_MISMATCH",
                f"{kind}[{index}]")
        if version == 1:
            start, end = entry.get("start"), entry.get("end")
            require(integer(start) and integer(end) and start == normalized_cursor
                    and end > start and end <= len(raw) and start in byte_to_char
                    and end in byte_to_char, "V1_TOKEN_OFFSETS", f"{kind}[{index}]")
            piece = raw[start:end]
            char_start, char_end = byte_to_char[start], byte_to_char[end]
            offset_char_start, offset_char_end = char_start, char_end
            expected = {
                "index": index, "token_id": token_id, "start": start, "end": end,
                "char_start": char_start, "char_end": char_end,
                "piece_sha256": sha256_bytes(piece),
            }
            normalized_cursor = end
        else:
            piece_hex = entry.get("piece_hex")
            require(isinstance(piece_hex, str) and len(piece_hex) % 2 == 0
                    and re.fullmatch(r"[0-9a-f]*", piece_hex) is not None,
                    "V2_PIECE_HEX", f"{kind}[{index}]")
            piece = bytes.fromhex(piece_hex)
            require(piece and normalized_raw[normalized_cursor:normalized_cursor + len(piece)]
                    == piece, "V2_NORMALIZED_PIECE", f"{kind}[{index}]")
            normalized_end = normalized_cursor + len(piece)
            overlapping_units = [unit for unit in units
                                 if min(normalized_end, unit["normalized_end"])
                                 > max(normalized_cursor, unit["normalized_start"])]
            require(overlapping_units, "V2_SOURCE_UNIT_MISSING", f"{kind}[{index}]")
            start = overlapping_units[0]["source_start"]
            end = overlapping_units[-1]["source_end"]
            require(start >= previous_start and start <= max(previous_end, source_cursor)
                    and end > start and end <= len(raw), "V2_SOURCE_RANGE",
                    f"{kind}[{index}]")
            char_start = byte_to_char[start]
            char_end = byte_to_char[end]
            offset_char_start = entry.get("offset_char_start")
            offset_char_end = entry.get("offset_char_end")
            require(integer(offset_char_start) and integer(offset_char_end)
                    and 0 <= offset_char_start < offset_char_end <= len(text)
                    and start <= char_to_byte[offset_char_start]
                    and char_to_byte[offset_char_end] <= end,
                    "V2_RAW_OFFSETS", f"{kind}[{index}]")
            expected = {
                "index": index, "token_id": token_id, "start": start, "end": end,
                "char_start": char_start, "char_end": char_end,
                "offset_char_start": offset_char_start, "offset_char_end": offset_char_end,
                "offset_start": char_to_byte[offset_char_start],
                "offset_end": char_to_byte[offset_char_end],
                "normalized_start": normalized_cursor, "normalized_end": normalized_end,
                "piece_hex": piece_hex, "piece_sha256": sha256_bytes(piece),
                "source_sha256": sha256_bytes(raw[start:end]),
            }
            if vocab is not None and isinstance(vocab, dict):
                token_text = next((key for key, value in vocab.items() if value == token_id), None)
                if token_text is not None:
                    try:
                        vocab_piece = bytes(BYTELEVEL_INVERSE[char] for char in token_text)
                    except KeyError as exc:
                        raise AuditInputError("V2_BYTELEVEL_VOCAB", f"{kind}[{index}]",
                                              str(exc)) from exc
                    same(vocab_piece, piece, "V2_TOKEN_PIECE_MISMATCH", f"{kind}[{index}]")
            normalized_cursor = normalized_end
            source_cursor = max(source_cursor, end)
            previous_start, previous_end = start, end
        overlapping = []
        for span_index, span in enumerate(spans):
            char_overlap = max(0, min(offset_char_end, span["char_end"])
                               - max(offset_char_start, span["char_start"]))
            if char_overlap:
                overlapping.append((span_index, char_overlap))
        same(sum(item[1] for item in overlapping), offset_char_end - offset_char_start,
             "TOKEN_SPAN_COVERAGE", f"{kind}[{index}]")
        weight = sum(_effective_weight(spans[span_index]) * overlap
                     for span_index, overlap in overlapping) / (
                         offset_char_end - offset_char_start)
        require(math.isfinite(weight) and weight > 0.0, "INVALID_SFT_WEIGHT",
                f"{kind}[{index}]")
        source_events, source_frames = _source_indices(start, end, segments, events)
        expected["source_event_indices"] = source_events
        expected["source_frame_indices"] = source_frames
        same(entry, expected, "TOKEN_METADATA_MISMATCH", f"{kind}[{index}]")
        tokens.append(entry)
        span_types = sorted({spans[span_index]["span_type"]
                             for span_index, _ in overlapping})
        closing = any(spans[span_index]["span_type"] == "visual_route_tag"
                      and raw[spans[span_index]["start"]:spans[span_index]["end"]] == b"]]"
                      for span_index, _ in overlapping)
        annotations.append({"span_types": span_types, "weight": weight,
                            "includes_terminal_delimiter": closing})
        weights.append(weight)
    same(normalized_cursor, len(normalized_raw), "TOKEN_STREAM_COVERAGE", kind)
    if version == 2:
        same(source_cursor, len(raw), "V2_SOURCE_COVERAGE", kind)
    for span_index, span in enumerate(spans):
        overlapping = [index for index, token in enumerate(tokens)
                       if min(token["end"], span["end"]) > max(token["start"], span["start"])]
        require(overlapping, "SPAN_WITHOUT_TOKENS", f"{kind}.spans[{span_index}]")
        span["token_start"] = overlapping[0]
        span["token_end"] = overlapping[-1] + 1
    same(stream.get("source"), kind, "STREAM_SOURCE", kind)
    same(stream.get("text_bytes"), len(raw), "STREAM_TEXT_BYTES", kind)
    same(stream.get("text_sha256"), sha256_bytes(raw), "STREAM_TEXT_HASH", kind)
    same(stream.get("decoded_sha256"), sha256_bytes(normalized_raw),
         "STREAM_DECODED_HASH", kind)
    same(stream.get("token_count"), len(tokens), "STREAM_TOKEN_COUNT", kind)
    same(stream.get("token_ids_sha256"), canonical_hash(ids), "STREAM_IDS_HASH", kind)
    same(stream.get("tokens_sha256"), canonical_hash(tokens), "STREAM_TOKENS_HASH", kind)
    same(stream.get("spans"), spans, "STREAM_SPANS", kind)
    same(stream.get("spans_sha256"), canonical_hash(spans), "STREAM_SPANS_HASH", kind)
    if version == 2:
        same(stream.get("normalized_bytes"), len(normalized_raw),
             "STREAM_NORMALIZED_BYTES", kind)
        same(stream.get("normalized_sha256"), sha256_bytes(normalized_raw),
             "STREAM_NORMALIZED_HASH", kind)
        same(stream.get("normalization_units"), units, "STREAM_NORMALIZATION_UNITS", kind)
        same(stream.get("normalization_units_sha256"), canonical_hash(units),
             "STREAM_UNITS_HASH", kind)
    mask = stream.get("sft_mask")
    require(isinstance(mask, dict), "MISSING_SFT_MASK", kind)
    same(mask.get("ablation_profile"), "joint_full", "SFT_MASK_PROFILE", kind)
    same(mask.get("span_weights"), SFT_SPAN_WEIGHTS, "SFT_SPAN_WEIGHTS", kind)
    same(mask.get("token_weights"), weights, "SFT_TOKEN_WEIGHTS", kind)
    same(mask.get("token_weights_sha256"), canonical_hash(weights),
         "SFT_TOKEN_WEIGHTS_HASH", kind)
    same(mask.get("annotations_sha256"), canonical_hash(annotations),
         "SFT_ANNOTATIONS_HASH", kind)
    same(mask.get("active_tokens"), sum(weight > 0.0 for weight in weights),
         "SFT_ACTIVE_COUNT", kind)
    same(mask.get("terminal_delimiter_tokens"),
         sum(item["includes_terminal_delimiter"] for item in annotations),
         "SFT_DELIMITER_COUNT", kind)
    if version == 2:
        same(mask.get("offset_semantics"), "fast_tokenizer_original_codepoint_offsets",
             "V2_OFFSET_SEMANTICS", kind)
    return {"stream": stream, "tokens": tokens, "spans": spans,
            "normalized_raw": normalized_raw}


def audit_token_export_alignment(record, behavior, replay, tokenizer_dir,
                                 tokenizer_manifest=None):
    """Independently bind sampled evidence to v1/v2 raw targets and masks."""
    export = record.get("token_export")
    if export is None:
        return {"status": "NOT_COVERED", "reason": "NO_TOKEN_EXPORT_PRESENT",
                "sampled_ids_replaced": False}
    require(isinstance(export, dict) and record.get("schema_version") == 2,
            "TOKEN_EXPORT_SCHEMA", "token_export")
    version = export.get("format_version")
    require(version in (1, 2), "TOKEN_EXPORT_VERSION", "token_export")
    expected_format = ("umminfer-trajectory-token-export-v1" if version == 1
                       else "umminfer-trajectory-token-export-v2")
    same(export.get("format"), expected_format, "TOKEN_EXPORT_FORMAT", "token_export")
    identity = export.get("tokenizer")
    require(isinstance(identity, dict), "TOKENIZER_IDENTITY", "token_export")
    same(identity.get("available_files_sha256"),
         canonical_hash(identity.get("available_files")),
         "AVAILABLE_FILES_HASH", "token_export")
    vocab = None
    if tokenizer_dir is not None:
        _files, loaded, terminal, vocab = verify_tokenizer_dir(
            tokenizer_dir, identity, version)
    else:
        require(isinstance(tokenizer_manifest, dict), "TOKENIZER_FILES_NOT_RUN",
                "token_export")
        same(identity.get("available_files"), tokenizer_manifest.get("files"),
             "PINNED_TOKENIZER_FILES", "token_export")
        loaded = tokenizer_manifest.get("behavior_sha256")
        terminal = tokenizer_manifest.get("terminal_token_ids")
        require_sha256(loaded, "token_export.behavior_tokenizer")
        require(isinstance(terminal, list) and terminal
                and all(integer(value) and value >= 0 for value in terminal)
                and terminal == sorted(set(terminal)), "PINNED_TERMINAL_IDS",
                "token_export")
    same(behavior.get("tokenizer_sha256"), loaded, "SAMPLED_TOKENIZER_DRIFT",
         "behavior.tokenizer_sha256")
    if tokenizer_manifest is not None and tokenizer_manifest.get("identity") is not None:
        expected_identity = tokenizer_manifest["identity"]
        for key in ("id", "revision", "tokenization_version"):
            same(identity.get(key), expected_identity.get(key),
                 "TOKENIZER_IDENTITY_DRIFT", "token_export." + key)
    source = {key: copy.deepcopy(value) for key, value in record.items()
              if key != "token_export"}
    source_sha = canonical_hash(source)
    same(export.get("source_record_sha256"), source_sha, "EXPORT_SOURCE_HASH",
         "token_export")
    policy = EXPORT_POLICY_V1 if version == 1 else EXPORT_POLICY_V2
    same(export.get("policy"), policy, "TOKEN_EXPORT_POLICY", "token_export")
    same(export.get("config_sha256"), canonical_hash({
        "tokenizer": identity, "policy": policy}), "TOKEN_EXPORT_CONFIG_HASH",
        "token_export")
    unsigned = {key: copy.deepcopy(value) for key, value in export.items()
                if key != "export_sha256"}
    export_sha = canonical_hash(unsigned)
    same(export.get("export_sha256"), export_sha, "TOKEN_EXPORT_HASH", "token_export")
    same(set(export.get("streams", {})), {"clean_text", "sft_target"},
         "TOKEN_EXPORT_STREAMS", "token_export")
    clean = record["clean_text"]
    target, target_segments = reconstruct_target(record)
    clean_segments = _clean_segments(record)
    target_result = _validate_stream(
        "sft_target", target, behavior["token_ids"], export["streams"]["sft_target"],
        target_segments, record["events"], identity, version, vocab)
    clean_result = _validate_stream(
        "clean_text", clean, export["streams"]["clean_text"]["token_ids"],
        export["streams"]["clean_text"], clean_segments, record["events"], identity,
        version, vocab)
    terminal_set = set(terminal)
    require(not (set(behavior["token_ids"]) & terminal_set), "TERMINAL_SAMPLED_TOKEN",
            "behavior.token_ids")
    masks = behavior["mask"]
    visual_slots = record.get("visual_slots") or []
    raw = target.encode("utf-8")
    masked_pieces = []
    for index, (token, mask) in enumerate(zip(target_result["tokens"], masks)):
        types = set()
        for span in target_result["spans"]:
            if min(token["end"], span["end"]) > max(token["start"], span["start"]):
                types.add(span["span_type"])
        require(types, "TOKEN_WITHOUT_BEHAVIOR_SPAN", f"behavior[{index}]")
        visual = bool(types & {"visual_route_tag", "visual_prompt_body"})
        prose = bool(types & {"narrative_prose", "spoken_style"})
        require(not (visual and prose), "PROSE_DIRECTIVE_STRADDLE",
                f"behavior[{index}]")
        expected_mask = 0 if visual else 1
        same(mask, expected_mask, "BEHAVIOR_MASK_MISMATCH", f"behavior[{index}]")
        visual_event = any(record["events"][event].get("type") == "visual_slot"
                           for event in token["source_event_indices"])
        require(not (mask and visual_event), "SUPERVISED_VISUAL_TOKEN",
                f"behavior[{index}]")
        if visual_slots and not mask:
            require(visual_event, "ZERO_MASK_WITHOUT_VISUAL_SOURCE",
                    f"behavior[{index}]")
        if mask:
            masked_pieces.append(bytes.fromhex(token["piece_hex"]) if version == 2
                                 else raw[token["start"]:token["end"]])
    if visual_slots:
        require(any(mask == 0 for mask in masks) and any(mask == 1 for mask in masks),
                "VISUAL_MASK_COVERAGE", "behavior.mask")
    else:
        require(all(masks), "TEXT_MASK_NOT_ALL_SUPERVISED", "behavior.mask")
    expected_clean = unicodedata.normalize("NFC", clean).encode("utf-8") \
        if version == 2 else clean.encode("utf-8")
    same(b"".join(masked_pieces), expected_clean, "MASKED_CLEAN_BYTES",
         "behavior.mask")
    clean_ids = clean_result["stream"]["token_ids"]
    masked_ids = [token for token, mask in zip(behavior["token_ids"], masks) if mask]
    exact = masked_ids == clean_ids
    return {
        "status": "PASS", "behavior_sha256": loaded,
        "sampled_token_count": len(behavior["token_ids"]),
        "masked_sampled_token_count": len(masked_ids), "masked_token_count": len(masked_ids),
        "sampled_posthoc_target_ids_equal": True,
        "clean_posthoc_token_count": len(clean_ids), "clean_reencoded_ids_equal": exact,
        "clean_id_relation": "EXACT" if exact else "CONTEXT_BOUNDARY_DIFFERENCE_BYTES_EXACT",
        "source_record_sha256": source_sha, "export_sha256": export_sha,
        "target_sha256": sha256_bytes(target.encode("utf-8")),
        "terminal_token_ids": terminal, "tokenizer_reencode": "NOT_RUN",
        "logprob_numerical_recomputation": "NOT_RUN",
        "offline_sft_mask": "joint_full_validated_separately_not_behavior_mask",
        "validation_level": "STRUCTURAL",
    }


def _task42_attestation(manifest, case, request, response, summary):
    behavior = {
        "policy_id": manifest["behavior_policy"]["policy_id"],
        "revision": manifest["behavior_policy"]["revision"],
        "model_id": request["model"],
        "model_revision": summary["provenance"]["revision"],
        "model_sha256": summary["provenance"]["model_sha256"],
        "model_config_sha256": summary["provenance"]["config_sha256"],
    }
    sampling = {"max_tokens": request["max_tokens"],
                "temperature": request["temperature"], "top_p": request["top_p"]}
    value = {
        "http_status": 200, "contract": "1", "generation": "fresh",
        "batch_id": manifest["batch_id"], "group_id": "behavior-group-" + case["name"],
        "rollout_id": "behavior-rollout-" + case["name"], "rollout_index": "0",
        "attempt": "0", "request_sha256": canonical_hash(request),
        "behavior_policy_sha256": canonical_hash(behavior),
        "behavior_policy_revision": behavior["revision"], "model_id": behavior["model_id"],
        "model_revision": behavior["model_revision"],
        "model_sha256": behavior["model_sha256"],
        "model_config_sha256": behavior["model_config_sha256"],
        "sampling_sha256": canonical_hash(sampling),
        "behavior_evidence": response["headers"].get("x-umminfer-behavior-evidence"),
    }
    attestation = canonical_hash(value)
    header_names = {
        "contract": "x-umminfer-online-rollout-contract",
        "generation": "x-umminfer-rollout-generation",
        "batch_id": "x-umminfer-rollout-batch-id",
        "group_id": "x-umminfer-rollout-group-id",
        "rollout_id": "x-umminfer-rollout-id",
        "rollout_index": "x-umminfer-rollout-index",
        "attempt": "x-umminfer-rollout-attempt",
        "request_sha256": "x-umminfer-request-sha256",
        "behavior_policy_sha256": "x-umminfer-behavior-policy-sha256",
        "behavior_policy_revision": "x-umminfer-behavior-policy-revision",
        "model_id": "x-umminfer-behavior-model-id",
        "model_revision": "x-umminfer-behavior-model-revision",
        "model_sha256": "x-umminfer-behavior-model-sha256",
        "model_config_sha256": "x-umminfer-behavior-model-config-sha256",
        "sampling_sha256": "x-umminfer-sampling-sha256",
        "behavior_evidence": "x-umminfer-behavior-evidence",
    }
    for key, header in header_names.items():
        same(response["headers"].get(header), str(value[key]),
             "RESPONSE_ATTESTATION_HEADER", case["name"] + "." + header)
    return attestation


def _raw_behavior_blocks(replay):
    return [frame["value"]["behavior"] for frame in replay["frames"]
            if frame["value"] is not None and "behavior" in frame["value"]]


def audit_alignment_bundle(root, tokenizer_dir=None):
    """Audit retained real Task #42 captures without tokenizer/ML imports."""
    root = Path(root)
    manifest = read_json(root, "alignment.json", canonical=True)
    same(manifest.get("schema"), ALIGNMENT_FIXTURE_SCHEMA, "ALIGNMENT_SCHEMA",
         "alignment.json")
    same(manifest.get("evidence_class"), "RETAINED_REAL_TASK42",
         "ALIGNMENT_EVIDENCE_CLASS", "alignment.json")
    source_info = manifest["source"]
    source_path = source_info["trajectory"]
    aligned_path = source_info["aligned_trajectory"]
    for key, relative in (("trajectory_sha256", source_path),
                          ("aligned_trajectory_sha256", aligned_path),
                          ("summary_sha256", source_info["summary"]),
                          ("checksums_sha256", source_info["checksums"])):
        same(sha256_bytes(read_file(root, relative)), source_info[key],
             "RETAINED_FILE_HASH", relative)
    source_records = read_jsonl(root, source_path, allow_empty=False)
    aligned_records = read_jsonl(root, aligned_path, allow_empty=False)
    same(len(source_records), 5, "TASK42_SOURCE_COUNT", source_path)
    same(len(aligned_records), len(source_records), "ALIGNED_RECORD_COUNT", aligned_path)
    identities = set()
    for index, (source, aligned) in enumerate(zip(source_records, aligned_records)):
        additive = {key: copy.deepcopy(value) for key, value in aligned.items()
                    if key != "token_export"}
        same(additive, source, "NON_ADDITIVE_TOKEN_EXPORT", f"records[{index}]")
        identity = (source.get("session_id"), source.get("request_id"))
        require(all(isinstance(value, str) and value for value in identity)
                and identity not in identities, "DUPLICATE_RECORD_IDENTITY",
                f"records[{index}]")
        identities.add(identity)
    summary = read_json(root, source_info["summary"], canonical=True)
    same(summary.get("status"), "PASS", "RETAINED_SUMMARY_STATUS", "summary.json")
    same(summary.get("provenance", {}).get("source_commit"), manifest.get("source_commit"),
         "SOURCE_COMMIT_DRIFT", "summary.provenance.source_commit")
    summary_cases = summary.get("cases")
    require(isinstance(summary_cases, list), "MISSING_SUMMARY_CASES", "summary.cases")
    by_name = {}
    for case in summary_cases:
        require(case.get("case") not in by_name, "DUPLICATE_SUMMARY_CASE",
                "summary.cases")
        by_name[case["case"]] = case
    same(set(by_name), {"text", "visual", "truncated", "cancel", "default"},
         "TASK42_CASE_SET", "summary.cases")
    retained_checksums = read_file(root, source_info["checksums"]).decode("ascii").splitlines()
    checksum_verified = []
    checksum_not_retained = []
    for line in retained_checksums:
        parts = line.split("  ", 1)
        require(len(parts) == 2 and HEX64.fullmatch(parts[0]) is not None,
                "INVALID_SOURCE_CHECKSUM_LINE", "SHA256SUMS.source")
        digest, relative = parts
        retained_path = source_path if relative == "live/trajectories.jsonl" else relative
        target = root / retained_path
        if target.is_file():
            require(not target.is_symlink(), "CHECKSUM_SYMLINK", retained_path)
            same(sha256_bytes(target.read_bytes()), digest, "SOURCE_CHECKSUM_MISMATCH",
                 retained_path)
            checksum_verified.append(relative)
        else:
            checksum_not_retained.append(relative)
    accepted = []
    for case in manifest["cases"]:
        name = case["name"]
        index = case["record_index"]
        source = source_records[index]
        aligned = aligned_records[index]
        summary_case = by_name[name]
        request = read_json(root, f"requests/{name}.json", canonical=True)
        response = read_json(root, f"live/responses/{name}.json", canonical=True)
        for key, relative in (("request_file_sha256", f"requests/{name}.json"),
                              ("response_file_sha256", f"live/responses/{name}.json"),
                              ("replay_file_sha256", f"logs/{name}-replay.stdout")):
            same(sha256_bytes(read_file(root, relative)), case[key],
                 "CASE_FILE_HASH", name + "." + relative)
        same(request.get("stream"), True, "REQUEST_NOT_STREAMING", name)
        same(request.get("session_id"), source["session_id"], "REQUEST_SESSION_DRIFT", name)
        same(response.get("status"), 200, "RESPONSE_HTTP_STATUS", name)
        replay = verify_raw_replay(source, root / source_info["artifact_dir"],
                                   f"{name}.source")
        raw_ref = source["sse_replay"]["raw_ref"]
        same(source["sse_replay"]["raw_sha256"], case["raw_sha256"],
             "CASE_RAW_HASH", name)
        same(summary_case.get("raw_sha256"), case["raw_sha256"],
             "SUMMARY_RAW_HASH", name)
        same(summary_case.get("raw_bytes"), replay["raw_bytes"],
             "SUMMARY_RAW_BYTES", name)
        require(integer(summary_case.get("framing", {}).get("chunks"))
                and summary_case["framing"]["chunks"] > 0
                and summary_case["framing"].get("semantic_identity") is True,
                "SUMMARY_FRAMING_IDENTITY", name)
        headers = response["headers"]
        require(headers.get("content-type") == "text/event-stream",
                "RESPONSE_CONTENT_TYPE", name)
        if "content-length" in headers:
            same(headers["content-length"], str(replay["raw_bytes"]),
                 "RESPONSE_CONTENT_LENGTH", name)
            require("transfer-encoding" not in headers, "AMBIGUOUS_RESPONSE_FRAMING", name)
        else:
            same(headers.get("transfer-encoding"), "chunked",
                 "RESPONSE_CHUNKED_FRAMING", name)
        attestation = _task42_attestation(manifest, case, request, response, summary)
        same(attestation, case["attestation_sha256"], "CASE_ATTESTATION_HASH", name)
        same(attestation, summary_case.get("attestation_sha256"),
             "SUMMARY_ATTESTATION_HASH", name)
        for frame in replay["frames"]:
            value = frame["value"]
            if value is None:
                continue
            same(value.get("id"), source["request_id"], "RAW_REQUEST_ID", name)
            same(value.get("model"), request["model"], "RAW_MODEL_ID", name)
            unified = value.get("unified")
            if isinstance(unified, dict):
                same(unified.get("session_id"), source["session_id"],
                     "RAW_SESSION_ID", name)
                if "request_id" in unified:
                    same(unified["request_id"], source["request_id"],
                         "RAW_INNER_REQUEST_ID", name)
        request_params = {key: copy.deepcopy(value) for key, value in request.items()
                          if key != "messages"}
        same(source["request_params"], request_params, "REQUEST_PARAMS_MISMATCH", name)
        expected_inputs = []
        for message in request["messages"]:
            content = message["content"]
            if isinstance(content, str):
                text = content
            else:
                require(isinstance(content, list)
                        and all(part.get("type") == "text" for part in content),
                        "UNVERIFIED_REQUEST_MEDIA", name)
                text = "".join(part["text"] for part in content)
            expected_inputs.append({"kind": "text", "ref": None, "media_type": None,
                                    "sha256": sha256_bytes(text.encode("utf-8")),
                                    "text": text})
        same(source["inputs"], expected_inputs, "REQUEST_INPUT_MISMATCH", name)
        same(source["terminal_state"], "completed", "ACCEPTED_TERMINAL_STATE", name)
        final = replay["frames"][-2]["value"]
        same(final["choices"][0].get("finish_reason"), "stop",
             "ACCEPTED_FINISH_REASON", name)
        raw_behavior = behavior_from_raw(replay, name)
        behavior_summary = {
            "status": "COVERED", **raw_behavior,
            "token_count": len(raw_behavior["token_ids"]),
            "masked_token_count": sum(raw_behavior["mask"]),
            "evidence_sha256": canonical_hash(raw_behavior),
        }
        same(summary_case.get("behavior"), behavior_summary,
             "SUMMARY_BEHAVIOR_MISMATCH", name)
        same(summary_case.get("coverage"), "exposed", "SUMMARY_COVERAGE", name)
        same(behavior_summary["evidence_sha256"], case["evidence_sha256"],
             "CASE_BEHAVIOR_HASH", name)
        replay_log = read_json(root, f"logs/{name}-replay.stdout", canonical=True)
        same(replay_log.get("behavior"), raw_behavior, "REPLAY_BEHAVIOR", name)
        same(replay_log.get("finish_reason"), "stop", "REPLAY_FINISH_REASON", name)
        same(replay_log.get("clean_text"), source["clean_text"], "REPLAY_CLEAN_TEXT", name)
        target, _segments = reconstruct_target(aligned)
        same(replay_log.get("raw_text"), target, "REPLAY_RAW_TARGET", name)
        prompt_ids = replay_log.get("prompt_token_ids")
        require(isinstance(prompt_ids, list) and prompt_ids
                and all(integer(value) and value >= 0 for value in prompt_ids),
                "REPLAY_PROMPT_IDS", name)
        forward = summary_case.get("fresh_forward")
        if forward is not None:
            same(forward.get("returncode"), 0, "RETAINED_FORWARD_STATUS", name)
            same(forward.get("raw_text_sha256"), sha256_bytes(target.encode("utf-8")),
                 "RETAINED_FORWARD_TARGET", name)
            same(forward.get("clean_text_sha256"),
                 sha256_bytes(source["clean_text"].encode("utf-8")),
                 "RETAINED_FORWARD_CLEAN", name)
            same(forward.get("prompt_sha256"), canonical_hash(prompt_ids),
                 "RETAINED_FORWARD_PROMPT", name)
            require(finite(forward.get("max_abs_logprob_diff"))
                    and 0.0 <= forward["max_abs_logprob_diff"] <= 1e-6,
                    "RETAINED_FORWARD_TOLERANCE", name)
        report = audit_token_export_alignment(
            aligned, raw_behavior, replay, tokenizer_dir, manifest["tokenizer"])
        report["name"] = name
        report["raw_sha256"] = case["raw_sha256"]
        report["attestation_sha256"] = attestation
        report["evidence_sha256"] = case["evidence_sha256"]
        accepted.append(report)
    excluded = []
    for case in manifest["excluded_cases"]:
        name = case["name"]
        source = source_records[case["record_index"]]
        require("token_export" not in aligned_records[case["record_index"]],
                "EXCLUDED_CASE_HAS_EXPORT", name)
        replay = verify_raw_replay(source, root / source_info["artifact_dir"],
                                   name + ".source")
        same(replay["raw_sha256"], case["raw_sha256"], "EXCLUDED_RAW_HASH", name)
        response = read_json(root, f"live/responses/{name}.json", canonical=True)
        same(response["headers"].get("x-umminfer-behavior-evidence"), "not_exposed",
             "EXCLUDED_COVERAGE", name)
        same(_raw_behavior_blocks(replay), [], "EXCLUDED_BEHAVIOR_BLOCK", name)
        summary_case = by_name[name]
        same(summary_case.get("behavior"), {
            "status": "NOT_COVERED", "token_ids": None, "logprobs": None, "mask": None},
            "EXCLUDED_SUMMARY_BEHAVIOR", name)
        final = replay["frames"][-2]["value"]
        finish = final["choices"][0].get("finish_reason")
        same(finish, case.get("finish_reason"), "EXCLUDED_FINISH_REASON", name)
        same(source["terminal_state"], case["terminal_state"],
             "EXCLUDED_TERMINAL_STATE", name)
        if name == "truncated":
            same(finish, "length", "TRUNCATION_NOT_REJECTED", name)
        if name == "cancel":
            same(source["events"][-1].get("type"), "cancelled",
                 "CANCEL_NOT_RETAINED", name)
        excluded.append({"name": name, "status": "PASS", "coverage": "NOT_COVERED",
                         "finish_reason": finish, "terminal_state": source["terminal_state"],
                         "raw_sha256": case["raw_sha256"]})
    same([case["name"] for case in accepted], ["text", "visual"],
         "ACCEPTED_CASE_ORDER", "alignment.cases")
    same(sum(case["sampled_token_count"] for case in accepted), 61,
         "TASK42_SAMPLED_TOTAL", "alignment.cases")
    return {
        "schema": AUDIT_SCHEMA, "mode": "retained_task42_alignment", "status": "PASS",
        "evidence_class": "RETAINED_REAL_TASK42", "validation_level": "STRUCTURAL",
        "accepted_cases": accepted, "excluded_cases": excluded,
        "sampled_token_count": 61, "supervised_token_count": sum(
            case["masked_token_count"] for case in accepted),
        "tokenizer_files": "PASS" if tokenizer_dir is not None else "NOT_RUN",
        "tokenizer_reencode": "NOT_RUN", "raw_sse_source_binding": "PASS",
        "original_checksums": {"declared": len(retained_checksums),
                               "retained_verified": checksum_verified,
                               "not_retained": checksum_not_retained},
        "source_files": {
            "trajectory_sha256": sha256_bytes(read_file(root, source_path)),
            "aligned_trajectory_sha256": sha256_bytes(read_file(root, aligned_path)),
            "alignment_sha256": sha256_bytes(read_file(root, "alignment.json")),
            "summary_sha256": sha256_bytes(read_file(root, source_info["summary"])),
            "checksums_sha256": sha256_bytes(read_file(root, source_info["checksums"])),
        },
        "claims": {"new_model_inference": "NOT_RUN",
                   "new_logprob_recomputation": "NOT_RUN", "online_optimizer": "NOT_RUN",
                   "gpu": "NOT_RUN", "serving": "NOT_RUN", "model_quality": "NOT_RUN",
                   "retained_forward_evidence": "RETAINED_TASK42_PASS_NOT_RERUN"},
    }


PREFLIGHT_SOURCE_FILES = (
    "python/umminfer_online_grpo_preflight.py",
    "python/umminfer_online_rollout.py",
    "python/umminfer_behavior_token_alignment.py",
    "python/umminfer_token_export.py",
    "python/umminfer_sse_replay.py",
    "python/umminfer_sft_dataset.py",
    "python/umminfer_rl_reward.py",
    "python/umminfer_rl_trajectory.py",
    "python/umminfer_ablation_metrics.py",
)
PREFLIGHT_BASE_DOCUMENTS = (
    "config.json", "run_identity.json", "schedule.json", "objective_fixtures.json",
)
PREFLIGHT_FILES = frozenset(PREFLIGHT_BASE_DOCUMENTS + (
    "manifest.json", "manifest.json.sha256", "SHA256SUMS",
))


def preflight_claims():
    return {
        "status": "PREFLIGHT_ONLY", "optimizer": "NOT_RUN", "REAL_TRAINING": "NOT_RUN",
        "current_policy_forward": "NOT_RUN", "reference_policy_forward": "NOT_RUN",
        "new_logprob_numerical_recomputation": "NOT_RUN", "inference": "NOT_RUN",
        "serving": "NOT_RUN", "gpu": "NOT_RUN", "model_quality": "NOT_RUN",
        "generalized_improvement": "NOT_RUN", "model_weights_updated": False,
        "torch_imported": False, "online_rollout_generation": "NOT_RUN_BY_PREFLIGHT",
        "source_rollout_evidence": "ENDPOINT_ATTESTATION_ONLY",
    }


def validate_preflight_config(config, plan, behavior_tokenizer):
    keys = {"schema", "current_policy", "reference_policy", "objective",
            "prompt_token_ids", "groups_per_step"}
    require(isinstance(config, dict) and set(config) == keys
            and config.get("schema") == PREFLIGHT_CONFIG_SCHEMA,
            "INVALID_PREFLIGHT_CONFIG", "config")
    config = copy.deepcopy(config)
    policy_keys = {"policy_id", "revision", "model_id", "model_revision",
                   "model_sha256", "model_config_sha256", "tokenizer_sha256"}
    for label in ("current_policy", "reference_policy"):
        policy = config[label]
        require(isinstance(policy, dict) and set(policy) == policy_keys,
                "INVALID_POLICY_IDENTITY", "config." + label)
        for key in ("policy_id", "revision", "model_id", "model_revision"):
            _printable(policy[key], f"config.{label}.{key}")
        for key in ("model_sha256", "model_config_sha256", "tokenizer_sha256"):
            require_sha256(policy[key], f"config.{label}.{key}")
        same(policy["model_id"], plan["behavior_policy"]["model_id"],
             "POLICY_MODEL_DRIFT", "config." + label)
        same(policy["model_config_sha256"],
             plan["behavior_policy"]["model_config_sha256"],
             "POLICY_MODEL_CONFIG_DRIFT", "config." + label)
        same(policy["tokenizer_sha256"], behavior_tokenizer,
             "POLICY_TOKENIZER_DRIFT", "config." + label)
    objective = config["objective"]
    require(isinstance(objective, dict) and set(objective) == {
        "clip_eps", "kl_coef", "log_ratio_clamp", "current_logprob_offsets",
        "reference_logprob_offsets"}, "INVALID_OBJECTIVE_CONFIG", "config.objective")
    require(finite(objective["clip_eps"]) and 0.0 < objective["clip_eps"] < 1.0,
            "INVALID_CLIP_EPS", "config.objective.clip_eps")
    require(finite(objective["kl_coef"]) and objective["kl_coef"] >= 0.0,
            "INVALID_KL_COEF", "config.objective.kl_coef")
    require(finite(objective["log_ratio_clamp"]) and objective["log_ratio_clamp"] > 0.0,
            "INVALID_LOG_RATIO_CLAMP", "config.objective.log_ratio_clamp")
    for key in ("current_logprob_offsets", "reference_logprob_offsets"):
        require(isinstance(objective[key], list) and objective[key]
                and all(finite(value) for value in objective[key]),
                "INVALID_LOGPROB_OFFSETS", "config.objective." + key)
    prompts = config["prompt_token_ids"]
    require(isinstance(prompts, dict) and prompts, "MISSING_PROMPT_IDS",
            "config.prompt_token_ids")
    for prompt_sha, ids in prompts.items():
        require_sha256(prompt_sha, "config.prompt_token_ids key")
        require(isinstance(ids, list) and 0 < len(ids) <= 8192
                and all(integer(token) and token >= 0 for token in ids),
                "INVALID_PROMPT_IDS", f"config.prompt_token_ids.{prompt_sha}")
    require(integer(config["groups_per_step"]) and config["groups_per_step"] > 0,
            "INVALID_GROUPS_PER_STEP", "config.groups_per_step")
    canonical_bytes(config)
    return config


def _mean(values):
    require(values, "EMPTY_SUPERVISED_MEAN", "objective")
    return math.fsum(values) / len(values)


def objective_metrics(policy_mean, behavior_mean, reference_mean, advantage, objective):
    clamp = float(objective["log_ratio_clamp"])
    raw_current = policy_mean - behavior_mean
    current = max(-clamp, min(clamp, raw_current))
    ratio = math.exp(current)
    clipped_ratio = max(1.0 - float(objective["clip_eps"]),
                        min(1.0 + float(objective["clip_eps"]), ratio))
    unclipped = ratio * advantage
    clipped = clipped_ratio * advantage
    raw_reference = reference_mean - policy_mean
    reference = max(-clamp, min(clamp, raw_reference))
    kl = math.exp(reference) - reference - 1.0
    selected = min(unclipped, clipped)
    return {
        "policy_mean_logp": policy_mean, "behavior_mean_logp": behavior_mean,
        "reference_mean_logp": reference_mean,
        "raw_current_minus_behavior_logp": raw_current,
        "current_minus_behavior_logp": current, "ratio": ratio,
        "clipped_ratio": clipped_ratio, "advantage": advantage,
        "unclipped_surrogate": unclipped, "clipped_surrogate": clipped,
        "selected_surrogate": selected, "clip_active": unclipped > clipped,
        "raw_reference_minus_current_logp": raw_reference,
        "reference_minus_current_logp": reference, "schulman_kl": kl,
        "objective_loss": -selected + float(objective["kl_coef"]) * kl,
    }


def _pack_member(audited, group_row, rollout, config, tokenizer_dir):
    record = audited["record"]
    behavior = audited["behavior"]
    prompt_sha = group_row["prompt_sha256"]
    require(prompt_sha in config["prompt_token_ids"], "UNBOUND_PROMPT_IDS",
            "config.prompt_token_ids", prompt_sha)
    prompt = config["prompt_token_ids"][prompt_sha]
    sampled = behavior["token_ids"]
    masks = behavior["mask"]
    packed = {
        "prompt_token_ids": prompt, "sampled_response_token_ids": sampled,
        "input_ids": prompt + sampled,
        "labels": [-100] * len(prompt) + [
            token if mask else -100 for token, mask in zip(sampled, masks)],
        "response_mask": [0] * len(prompt) + masks,
        "causal_shift": "model_logits[t-1]_score_labels[t]",
        "prompt_label": -100, "unsupervised_response_label": -100,
    }
    packed["input_ids_sha256"] = canonical_hash(packed["input_ids"])
    packed["labels_sha256"] = canonical_hash(packed["labels"])
    packed["response_mask_sha256"] = canonical_hash(packed["response_mask"])
    packed["packed_ids_labels_masks_sha256"] = canonical_hash({
        "input_ids": packed["input_ids"], "labels": packed["labels"],
        "response_mask": packed["response_mask"]})
    objective = config["objective"]
    behavior_scores = [float(value) for value in behavior["logprobs"]]
    current = []
    reference = []
    for index, value in enumerate(behavior_scores):
        current_value = value + float(objective["current_logprob_offsets"][
            index % len(objective["current_logprob_offsets"])])
        reference_value = value + float(objective["reference_logprob_offsets"][
            index % len(objective["reference_logprob_offsets"])])
        require(finite(current_value) and finite(reference_value)
                and current_value <= 0.0 and reference_value <= 0.0,
                "INVALID_SYNTHETIC_LOGPROB", "objective")
        current.append(current_value)
        reference.append(reference_value)
    supervised = [index for index, mask in enumerate(masks) if mask == 1]
    metrics = objective_metrics(
        _mean([current[index] for index in supervised]),
        _mean([behavior_scores[index] for index in supervised]),
        _mean([reference[index] for index in supervised]),
        float(rollout["advantage"]), objective)
    packed_sha = packed["packed_ids_labels_masks_sha256"]
    current_input = canonical_hash({
        "policy": config["current_policy"], "packed": packed_sha,
        "logprob_source": "SYNTHETIC_CPU_FIXTURE_NOT_MODEL_OUTPUT"})
    reference_input = canonical_hash({
        "policy": config["reference_policy"], "packed": packed_sha,
        "logprob_source": "SYNTHETIC_CPU_FIXTURE_NOT_MODEL_OUTPUT"})
    alignment = audit_token_export_alignment(
        record, behavior, audited["replay"], tokenizer_dir)
    member = {
        "rollout_id": audited["item"]["rollout_id"], "rank": rollout["rank"],
        "source_record_sha256": canonical_hash(record),
        "behavior_evidence_sha256": audited["behavior_summary"]["evidence_sha256"],
        "packing": packed,
        "behavior_policy": {
            "identity": audited["record"]["rollout_provenance"]["behavior_policy"],
            "tokenizer_sha256": behavior["tokenizer_sha256"],
            "sampled_logprobs": behavior_scores,
            "source": "FINAL_RAW_SSE_SAMPLED_BEHAVIOR"},
        "current_policy": {
            "identity": config["current_policy"], "input_identity_sha256": current_input,
            "logprobs": current, "forward": "NOT_RUN",
            "logprob_source": "SYNTHETIC_CPU_FIXTURE_NOT_MODEL_OUTPUT"},
        "reference_policy": {
            "identity": config["reference_policy"], "input_identity_sha256": reference_input,
            "logprobs": reference, "forward": "NOT_RUN",
            "logprob_source": "SYNTHETIC_CPU_FIXTURE_NOT_MODEL_OUTPUT"},
        "supervised_response_indices": supervised, "metrics": metrics,
        "raw_sse_sha256": audited["replay"]["raw_sha256"],
        "independent_token_alignment": alignment,
    }
    identity = {
        "rollout_id": audited["item"]["rollout_id"],
        "rollout_index": audited["item"]["rollout_index"], "rank": rollout["rank"],
        "seed": audited["item"]["seed"],
        "request_sha256": audited["item"]["request_sha256"],
        "source_record_sha256": member["source_record_sha256"],
        "raw_sse_sha256": member["raw_sse_sha256"],
        "behavior_evidence_sha256": member["behavior_evidence_sha256"],
        "reward": audited["reward"]["score"], "advantage": rollout["advantage"],
        "packed_ids_labels_masks_sha256": packed_sha,
        "current_policy_input_sha256": current_input,
        "reference_policy_input_sha256": reference_input,
        "token_export_alignment": alignment,
    }
    return member, identity


def _correct_checkpoint(run_id, steps, groups, packed_order, policies, cursor):
    completed = [group for step in steps[:cursor] for group in step["group_ids"]]
    remaining = [group for step in steps[cursor:] for group in step["group_ids"]]
    value = {
        "schema": PREFLIGHT_CHECKPOINT_SCHEMA, "mode": "online_preflight_only",
        "run_id": run_id, "next_step": cursor, "schedule_position": len(completed),
        "completed_group_ids": completed, "remaining_group_ids": remaining,
        # Dataset group_id is the identifier stored in both cursor lists. The
        # integrated adapter mistakenly compares schedule_group_id instead.
        "completed_group_exports_sha256": canonical_hash([
            group["group_export_sha256"] for group in groups
            if group["group_id"] in completed]),
        "remaining_group_exports_sha256": canonical_hash([
            group["group_export_sha256"] for group in groups
            if group["group_id"] in remaining]),
        "packed_input_order_sha256": packed_order,
        "behavior_policy_sha256": policies["behavior"],
        "current_policy_sha256": policies["current"],
        "reference_policy_sha256": policies["reference"],
        "model_weights_updated": False, "optimizer_state": "NOT_RUN",
        "optimizer": "NOT_RUN", "current_policy_forward": "NOT_RUN",
        "reference_policy_forward": "NOT_RUN",
    }
    value["checkpoint_id"] = canonical_hash(value)
    return value


def _checkpoint_disagreements(expected, actual):
    """Return the known semantic checkpoint mismatch without hiding others."""
    mismatches = []

    def walk(independent, accepted, path):
        if isinstance(independent, dict) and isinstance(accepted, dict):
            if set(independent) != set(accepted):
                raise AuditInputError("CHECKPOINT_FIELDS", path,
                                      f"{set(independent) ^ set(accepted)}")
            for key in independent:
                walk(independent[key], accepted[key], path + "." + key)
            return
        if isinstance(independent, list) and isinstance(accepted, list):
            if len(independent) != len(accepted):
                raise AuditInputError("CHECKPOINT_LIST_LENGTH", path,
                                      f"{len(independent)} != {len(accepted)}")
            for index, (left, right) in enumerate(zip(independent, accepted)):
                walk(left, right, f"{path}[{index}]")
            return
        if independent != accepted:
            allowed = (
                path.endswith("completed_group_exports_sha256")
                or path.endswith("remaining_group_exports_sha256")
                or path.endswith("checkpoint_id")
                or path.endswith("checkpoint_before")
                or path.endswith("checkpoint_after"))
            if not allowed:
                raise AuditInputError("PREFLIGHT_DOCUMENT_MISMATCH", path,
                                      f"{independent!r} != {accepted!r}")
            mismatches.append({"path": path, "independent": independent,
                               "accepted": accepted,
                               "class": "CHECKPOINT_GROUP_EXPORT_BINDING"})

    walk(expected, actual, "schedule.json")
    return mismatches


def audit_preflight_bundle(source_root, preflight_root, repo_root, tokenizer_dir=None):
    """Rebuild preflight arithmetic and identities; expose checkpoint defects."""
    source = audit_online_bundle(source_root)
    root = Path(preflight_root)
    actual_files = sorted(path.relative_to(root).as_posix()
                          for path in root.rglob("*") if path.is_file())
    same(actual_files, sorted(PREFLIGHT_FILES), "PREFLIGHT_FILE_SET", "preflight")
    for name in PREFLIGHT_BASE_DOCUMENTS + ("manifest.json",):
        read_json(root, name, canonical=True)
    verify_sidecar(root, "manifest.json")
    checksum_lines = read_file(root, "SHA256SUMS").decode("ascii").splitlines()
    expected_checksums = []
    for name in sorted(PREFLIGHT_FILES - {"SHA256SUMS"}):
        expected_checksums.append(sha256_bytes(read_file(root, name)) + "  " + name)
    same(checksum_lines, expected_checksums, "STALE_OR_RESIGNED_CHECKSUMS", "SHA256SUMS")
    config = validate_preflight_config(
        read_json(root, "config.json", canonical=True), source["plan"],
        source["records"][0]["behavior"]["tokenizer_sha256"])
    repo_root = Path(repo_root)
    source_fingerprints = [file_identity(repo_root, relative)
                           for relative in PREFLIGHT_SOURCE_FILES]
    by_rollout = {audited["item"]["rollout_id"]: audited for audited in source["records"]}
    groups = sorted(source["groups"], key=lambda pair: pair[0])
    group_documents = []
    objective_groups = []
    packed_order = []
    for group_index, group in groups:
        objective_members = []
        identity_members = []
        for rollout in group["rollouts"]:
            rollout_id = rollout["rollout_provenance"]["rollout_id"]
            require(rollout_id in by_rollout, "CROSS_GROUP_ROLLOUT", group["group_id"])
            member, identity = _pack_member(
                by_rollout[rollout_id], group, rollout, config, tokenizer_dir)
            objective_members.append(member)
            identity_members.append(identity)
            packed_order.append(member["packing"]["packed_ids_labels_masks_sha256"])
        group_export = canonical_hash(group)
        identity_group = {
            "schedule_group_id": group["rollout_provenance"]["schedule_group_id"],
            "group_id": group["group_id"], "group_index": group_index,
            "split": group["split"], "group_export_sha256": group_export,
            "prompt_sha256": group["prompt_sha256"],
            "prompt_token_ids_sha256": canonical_hash(
                config["prompt_token_ids"][group["prompt_sha256"]]),
            "group_mean": group["group_mean"], "group_std": group["group_std"],
            "has_reward_variance": group["has_reward_variance"],
            "zero_variance_policy": "ZERO_ADVANTAGE", "members": identity_members,
        }
        group_documents.append(identity_group)
        objective_groups.append({
            "schedule_group_id": group["rollout_provenance"]["schedule_group_id"],
            "group_id": group["group_id"], "split": group["split"],
            "group_export_sha256": group_export,
            "group_mean": group["group_mean"], "group_std": group["group_std"],
            "has_reward_variance": group["has_reward_variance"],
            "advantage_semantics": "POPULATION_STD_ZERO_VARIANCE_TO_ZERO",
            "members": objective_members,
            "objective_loss": math.fsum(member["metrics"]["objective_loss"]
                                        for member in objective_members) / len(objective_members),
            "normalization": "ROLLOUT_MEAN",
        })
    packed_order_sha = canonical_hash(packed_order)
    plan = source["plan"]
    behavior_tokenizer = source["records"][0]["behavior"]["tokenizer_sha256"]
    policies = {
        "behavior": canonical_hash(plan["behavior_policy"]),
        "current": canonical_hash(config["current_policy"]),
        "reference": canonical_hash(config["reference_policy"]),
    }
    has_exports = ["token_export" in audited["record"] for audited in source["records"]]
    require(all(has_exports) or not any(has_exports), "PARTIAL_TOKEN_EXPORTS",
            "trajectories")
    identity = {
        "mode": "online", "contract": PROVENANCE_CONTRACT,
        "batch_id": plan["batch_id"], "plan_sha256": source["plan_sha256"],
        "plan_bytes_sha256": sha256_bytes(source["plan_raw"]),
        "rollout_schedule_sha256": source["schedule_sha256"],
        "rollout_manifest_sha256": source["rollout_manifest_sha256"],
        "pack_manifest_sha256": source["pack_manifest_sha256"],
        "source_files": source["source_files"], "source_fingerprints": source_fingerprints,
        "config_sha256": canonical_hash(config),
        "behavior_policy": plan["behavior_policy"],
        "behavior_policy_sha256": policies["behavior"],
        "behavior_tokenizer_sha256": behavior_tokenizer,
        "sampling_sha256": canonical_hash(plan["sampling"]),
        "current_policy": config["current_policy"],
        "current_policy_sha256": policies["current"],
        "reference_policy": config["reference_policy"],
        "reference_policy_sha256": policies["reference"],
        "groups": group_documents, "packed_input_order_sha256": packed_order_sha,
        "objective": {
            **config["objective"],
            "ratio": "EXP_CLAMPED_CURRENT_MEAN_MINUS_SAMPLED_BEHAVIOR_MEAN",
            "surrogate": "MIN_RATIO_AND_CLIPPED_RATIO_TIMES_GROUP_ADVANTAGE",
            "kl": "SCHULMAN_EXP_REFERENCE_MINUS_CURRENT_MINUS_DIFF_MINUS_ONE",
            "logprob_reduction": "MASKED_RESPONSE_MEAN_THEN_ROLLOUT_MEAN",
            "advantage": "EXPORT_RECOMPUTED_POPULATION_STD_ZERO_VARIANCE_TO_ZERO",
            "current_reference_values": "SYNTHETIC_CPU_FIXTURES_NOT_MODEL_OUTPUT",
        },
        "token_export_alignment": "PASS_INDEPENDENT_STRUCTURAL" if all(has_exports)
            else "NOT_COVERED_NO_EXPORTS",
        "sampled_ids_replaced_by_posthoc_ids": False,
    }
    run_id = canonical_hash(identity)
    steps = []
    size = config["groups_per_step"]
    for start in range(0, len(group_documents), size):
        selected = group_documents[start:start + size]
        steps.append({
            "step_index": len(steps),
            "group_ids": [group["group_id"] for group in selected],
            "schedule_group_ids": [group["schedule_group_id"] for group in selected],
            "group_export_sha256": [group["group_export_sha256"] for group in selected],
            "member_export_sha256": [member["source_record_sha256"]
                                     for group in selected for member in group["members"]],
            "packed_input_sha256": [member["packed_ids_labels_masks_sha256"]
                                    for group in selected for member in group["members"]],
        })
    checkpoints = [_correct_checkpoint(run_id, steps, group_documents, packed_order_sha,
                                       policies, cursor)
                   for cursor in range(len(steps) + 1)]
    for index, step in enumerate(steps):
        step["checkpoint_before"] = checkpoints[index]["checkpoint_id"]
        step["checkpoint_after"] = checkpoints[index + 1]["checkpoint_id"]
    claims = preflight_claims()
    documents = {
        "config.json": config,
        "run_identity.json": {"schema": PREFLIGHT_RUN_SCHEMA, "status": "PASS",
                              "run_id": run_id, "identity_sha256": run_id,
                              "identity": identity, "claims": claims},
        "schedule.json": {"schema": PREFLIGHT_SCHEDULE_SCHEMA, "run_id": run_id,
                          "rollout_schedule_sha256": source["schedule_sha256"],
                          "groups_per_step": size, "steps": steps,
                          "checkpoints": checkpoints,
                          "cross_resume": "REQUIRE_EXACT_RUN_CHECKPOINT_GROUP_MEMBER_PACKED_AND_POLICY_IDS",
                          "claims": claims},
        "objective_fixtures.json": {
            "schema": PREFLIGHT_OBJECTIVE_SCHEMA, "run_id": run_id,
            "evidence_class": "DETERMINISTIC_SYNTHETIC_CPU_OBJECTIVE_FIXTURES",
            "groups": objective_groups, "claims": claims},
    }
    # Non-checkpoint documents are mandatory semantic identities. Re-signed
    # changes here are rejected with no report, not normalized or accepted.
    for name in ("config.json", "run_identity.json", "objective_fixtures.json"):
        actual = read_json(root, name, canonical=True)
        same(actual, documents[name], "PREFLIGHT_DOCUMENT_MISMATCH", name)
    actual_schedule = read_json(root, "schedule.json", canonical=True)
    disagreements = _checkpoint_disagreements(documents["schedule.json"], actual_schedule)
    payloads = {name: canonical_bytes(value) + b"\n" for name, value in documents.items()}
    manifest = {
        "schema": PREFLIGHT_MANIFEST_SCHEMA, "status": "PASS", "run_id": run_id,
        "source_bundle": {"schema": "umminfer-online-grpo-export-v1",
                          "files": source["source_files"]},
        "source_fingerprints": source_fingerprints,
        "files": {name: {"sha256": sha256_bytes(payload), "bytes": len(payload)}
                  for name, payload in payloads.items()},
        "group_count": len(group_documents), "rollout_count": len(source["records"]),
        "step_count": len(steps), "checkpoint_count": len(checkpoints), "claims": claims,
    }
    # A checkpoint disagreement intentionally changes only schedule.json and its
    # derived manifest/checksum fields. Everything else must remain exact.
    actual_manifest = read_json(root, "manifest.json", canonical=True)
    expected_manifest = copy.deepcopy(manifest)
    if disagreements:
        expected_manifest["files"]["schedule.json"] = actual_manifest["files"]["schedule.json"]
    same(actual_manifest, expected_manifest, "PREFLIGHT_MANIFEST_MISMATCH", "manifest.json")
    payloads["manifest.json"] = canonical_bytes(manifest) + b"\n"
    payloads["manifest.json.sha256"] = (
        sha256_bytes(payloads["manifest.json"]) + "\n").encode("ascii")
    checksum_text = "".join(
        f"{sha256_bytes(payloads[name])}  {name}\n" for name in sorted(payloads))
    positive = objective_metrics(math.log(2.0), 0.0, math.log(2.0), 1.0,
                                 config["objective"])
    negative = objective_metrics(math.log(2.0), 0.0, math.log(2.0) - 1.0, -1.0,
                                 config["objective"])
    return {
        "schema": AUDIT_SCHEMA, "mode": "online_grpo_preflight",
        "status": "FAIL" if disagreements else "PASS",
        "plan_sha256": source["plan_sha256"],
        "rollout_schedule_sha256": source["schedule_sha256"],
        "source_accepted_file_set_sha256": source["source_file_set_sha256"],
        "config_sha256": canonical_hash(config), "run_id": run_id,
        "group_count": len(group_documents), "rollout_count": len(source["records"]),
        "source_rollout_reconstruction": "PASS_BYTE_IDENTICAL",
        "run_identity_reconstruction": "PASS_BYTE_IDENTICAL",
        "objective_reconstruction": "PASS_BYTE_IDENTICAL",
        "schedule_reconstruction": "FAIL" if disagreements else "PASS_BYTE_IDENTICAL",
        "checkpoint_disagreements": disagreements,
        "groups": [{"group_id": group["group_id"],
                    "schedule_group_id": group["schedule_group_id"],
                    "group_mean": group["group_mean"], "group_std": group["group_std"],
                    "has_reward_variance": group["has_reward_variance"],
                    "objective_loss": objective["objective_loss"],
                    "member_count": len(group["members"])}
                   for group, objective in zip(group_documents, objective_groups)],
        "preflight_files": {name: {"sha256": sha256_bytes(payload), "bytes": len(payload)}
                            for name, payload in payloads.items()},
        "manifest_sha256": sha256_bytes(payloads["manifest.json"]),
        "sha256sums_sha256": sha256_bytes(checksum_text.encode("ascii")),
        "source_files": {item["path"]: item["sha256"] for item in source["source_files"]
                         if item["path"] in {
                             "plan.json", "trajectories.jsonl", "rollout_manifest.json",
                             "pack/manifest.json", "pack/grpo_train.jsonl",
                             "pack/grpo_eval.jsonl"}},
        "objective_oracles": {
            "positive_advantage_clipped": positive,
            "negative_advantage_unclipped_and_reference_kl": negative,
        },
        "source_fingerprints": source_fingerprints,
        "claims": claims,
        "boundaries": {"tokenizer_reencode": "NOT_RUN",
                       "new_logprob_numerical_recomputation": "NOT_RUN",
                       "optimizer": "NOT_RUN", "gpu": "NOT_RUN", "serving": "NOT_RUN"},
    }


def compare_accepted(independent, accepted, paths):
    """Return precise mismatches without changing either side."""
    mismatches = []

    def walk(path):
        try:
            actual = independent
            expected = accepted
            for part in path:
                actual = actual[part]
                expected = expected[part]
            if actual != expected:
                mismatches.append({
                    "path": "$" + "".join(f"[{part!r}]" for part in path),
                    "independent": actual, "accepted": expected,
                })
        except (KeyError, IndexError, TypeError) as exc:
            mismatches.append({"path": repr(path), "error": str(exc)})

    for path in paths:
        walk(tuple(path))
    return mismatches


def task42_evidence_mismatches(report, accepted):
    paths = []
    case_keys = (
        "attestation_sha256", "clean_id_relation", "clean_posthoc_token_count",
        "clean_reencoded_ids_equal", "evidence_sha256", "export_sha256",
        "masked_sampled_token_count", "masked_token_count", "name",
        "offline_sft_mask", "raw_sha256", "sampled_posthoc_target_ids_equal",
        "sampled_token_count", "source_record_sha256", "status",
    )
    for index in range(2):
        for key in case_keys:
            paths.append(["accepted_cases", index, key])
    for index in range(3):
        for key in ("coverage", "finish_reason", "name", "raw_sha256",
                    "status", "terminal_state"):
            paths.append(["excluded_cases", index, key])
    return compare_accepted(report, accepted, paths)


def preflight_evidence_mismatches(report, accepted):
    paths = [["plan_sha256"], ["rollout_schedule_sha256"], ["config_sha256"],
             ["run_id"], ["manifest_sha256"], ["sha256sums_sha256"],
             ["source_accepted_file_set_sha256"]]
    for name in ("config.json", "run_identity.json", "schedule.json",
                 "objective_fixtures.json"):
        paths.append(["preflight_files", name])
    for index in range(3):
        for key in ("group_id", "schedule_group_id", "group_mean", "group_std",
                    "has_reward_variance", "member_count", "objective_loss"):
            paths.append(["groups", index, key])
    for name in ("positive_advantage_clipped",
                 "negative_advantage_unclipped_and_reference_kl"):
        paths.append(["objective_oracles", name])
    mismatches = compare_accepted(report, accepted, paths)
    for name in PREFLIGHT_SOURCE_FILES:
        accepted_row = next((row for row in accepted.get("source_fingerprints", [])
                             if row.get("path") == name), None)
        independent_row = next(row for row in report["source_fingerprints"]
                               if row["path"] == name)
        if accepted_row is None:
            mismatches.append({"path": "source_fingerprints." + name,
                               "independent": independent_row, "accepted": None})
            continue
        for item in compare_accepted(independent_row, accepted_row,
                                     [["path"], ["bytes"], ["sha256"]]):
            item["path"] = "source_fingerprints." + name + "." + item["path"]
            mismatches.append(item)
    return mismatches


def _imports_for(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module.split(".")[0])
    return result


def audit_repository_isolation(root):
    """Check import direction and this auditor's independence from source."""
    root = Path(root)
    trainer_names = (FORBIDDEN_AUDIT_IMPORTS - ML_MODULES) | {
        "umminfer_sft_dataset", "umminfer_joint_sft_live",
        "umminfer_joint_dpo_live", "umminfer_rl_train", "umminfer_rl_worker",
    }
    violations = []
    scanned = 0
    for parent in ("src", "include", "native"):
        for path in sorted((root / parent).rglob("*")):
            if not path.is_file() or path.suffix not in {
                    ".cpp", ".hpp", ".cu", ".h", ".py"}:
                continue
            scanned += 1
            text = path.read_text(encoding="utf-8", errors="strict")
            for name in sorted(trainer_names):
                if name in text:
                    violations.append({"path": path.relative_to(root).as_posix(),
                                       "module": name})
    own_imports = _imports_for(Path(__file__).resolve())
    forbidden = sorted(own_imports & FORBIDDEN_AUDIT_IMPORTS)
    if forbidden:
        violations.append({"path": Path(__file__).name,
                           "modules": forbidden})
    require(not violations, "SERVING_TRAINER_ISOLATION", "repository", violations)
    require(not (ML_MODULES & set(sys.modules)), "ML_IMPORT_BEFORE_AUDIT",
            "sys.modules", sorted(ML_MODULES & set(sys.modules)))
    return {"status": "PASS", "serving_files_scanned": scanned,
            "auditor_imports": sorted(own_imports),
            "forbidden_imports": sorted(FORBIDDEN_AUDIT_IMPORTS),
            "ml_modules_loaded": []}


def write_report(path, report):
    """Write exactly once, refusing overwrite; never called for invalid input."""
    payload = canonical_bytes(report) + b"\n"
    if path is None:
        sys.stdout.buffer.write(payload)
        return
    target = Path(path)
    require(not target.exists(), "REPORT_ALREADY_EXISTS", str(target))
    require(target.parent.is_dir(), "MISSING_REPORT_PARENT", str(target.parent))
    with target.open("xb") as stream:
        stream.write(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=AUDIT_SCHEMA)
    commands = parser.add_subparsers(dest="command", required=True)
    task42 = commands.add_parser("task42", help="audit retained Task #42 alignment evidence")
    task42.add_argument("fixture")
    task42.add_argument("--tokenizer-dir")
    task42.add_argument("--accepted-evidence")
    task42.add_argument("--report")
    preflight = commands.add_parser("preflight", help="audit an online GRPO preflight bundle")
    preflight.add_argument("--online-pack", required=True)
    preflight.add_argument("--preflight-bundle", required=True)
    preflight.add_argument("--tokenizer-dir")
    preflight.add_argument("--repo-root", default=".")
    preflight.add_argument("--accepted-evidence")
    preflight.add_argument("--report")
    isolate = commands.add_parser("isolate", help="audit serving/trainer import isolation")
    isolate.add_argument("--repo-root", default=".")
    isolate.add_argument("--report")
    args = parser.parse_args(argv)
    try:
        if args.command == "task42":
            report = audit_alignment_bundle(args.fixture, args.tokenizer_dir)
            if args.accepted_evidence:
                accepted = strict_json_bytes(read_file(
                    Path(args.accepted_evidence).parent,
                    Path(args.accepted_evidence).name), args.accepted_evidence)
                mismatches = task42_evidence_mismatches(report, accepted)
                if mismatches:
                    report["accepted_evidence_mismatches"] = mismatches
                    report["status"] = "FAIL"
        elif args.command == "preflight":
            report = audit_preflight_bundle(
                args.online_pack, args.preflight_bundle, args.repo_root,
                args.tokenizer_dir)
            if args.accepted_evidence:
                accepted = strict_json_bytes(read_file(
                    Path(args.accepted_evidence).parent,
                    Path(args.accepted_evidence).name), args.accepted_evidence)
                mismatches = preflight_evidence_mismatches(report, accepted)
                if mismatches:
                    report["accepted_evidence_mismatches"] = mismatches
                    report["status"] = "FAIL"
        else:
            report = {"schema": AUDIT_SCHEMA, "mode": "repository_isolation",
                      "status": "PASS",
                      "isolation": audit_repository_isolation(args.repo_root)}
        require(not (ML_MODULES & set(sys.modules)), "ML_IMPORT_BEFORE_REPORT",
                "sys.modules", sorted(ML_MODULES & set(sys.modules)))
        write_report(args.report, report)
        return 0 if report["status"] == "PASS" else 1
    except AuditInputError as exc:
        error = {"schema": AUDIT_SCHEMA, "status": "REJECTED",
                 "code": exc.code, "path": exc.path, "detail": exc.detail}
        sys.stderr.buffer.write(canonical_bytes(error) + b"\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
