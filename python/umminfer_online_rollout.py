# SPDX-License-Identifier: Apache-2.0
"""Trainer-side online-rollout contract for deterministic Stage C GRPO groups.

The orchestrator is an HTTP client, not an inference engine. It schedules fresh
requests, requires the endpoint to attest exact request/group/policy identity,
and exports only completed raw-SSE-backed schema-v2 records. Rewards and GRPO
packing reuse the existing export-only interfaces. No serving, optimizer, ML, or
model code is imported here.

The current public ummInfer response does not expose behavior log-probabilities
or masks. They remain NOT_COVERED unless a compatible endpoint supplies one
complete, structurally valid ``behavior`` evidence block in its final SSE data
frame. Missing values are represented by null, never by fabricated zeros.
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

import umminfer_sft_dataset as dataset  # noqa: E402
from umminfer_rl_reward import REWARD_SCHEMA_VERSION, load_reward  # noqa: E402
from umminfer_rl_trajectory import (  # noqa: E402
    TrajectoryRecorder, canonical_json, input_references, record_chat_completion,
    request_parameter_references, sha256_bytes,
)
from umminfer_sse_replay import (  # noqa: E402
    RawSSECapture, replay_record, validate_replay_metadata,
)

PLAN_SCHEMA = "umminfer-online-grpo-plan-v1"
EXPORT_SCHEMA = "umminfer-online-grpo-export-v1"
PROVENANCE_CONTRACT = "umminfer-online-rollout-provenance-v1"
BEHAVIOR_FORMAT = "umminfer-behavior-logprobs-v1"
CONTRACT_VERSION = "1"
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_HEX_64 = frozenset("0123456789abcdef")

# Every response, including a retryable HTTP error, must carry this binding.
# Unknown headers are ignored by the existing public server, so these request
# headers are additive; a server that does not echo them is not compatible with
# this online contract and fails closed.
RESPONSE_HEADERS = {
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
BEHAVIOR_KEYS = frozenset(
    ("format", "tokenizer_sha256", "token_ids", "logprobs", "mask"))


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(float(value)))


def _text(value, label, *, maximum=256):
    if not isinstance(value, str) or not value or len(value) > maximum \
            or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{label} must be a non-empty printable string")
    return value


def _sha256(value, label):
    if not isinstance(value, str) or len(value) != 64 \
            or set(value) - _HEX_64:
        raise ValueError(f"{label} must be one lowercase SHA-256 digest")
    return value


def _canonical_line(value):
    return canonical_json(value) + b"\n"


def _write_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _write_jsonl(path, records):
    _write_bytes(path, b"".join(_canonical_line(record) for record in records))


def _write_document(path, value):
    payload = _canonical_line(value)
    _write_bytes(path, payload)
    digest = sha256_bytes(payload)
    _write_bytes(Path(str(path) + ".sha256"), (digest + "\n").encode("ascii"))
    return digest


def _new_directory(path):
    path = Path(path)
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise ValueError("output directory must be absent or empty")
    else:
        path.mkdir(parents=True)
    return path


def prepare_plan(value):
    """Validate one complete plan before opening a socket or loading a reward."""
    if not isinstance(value, dict) or value.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"online rollout plan must use {PLAN_SCHEMA}")
    required = {
        "schema", "batch_id", "behavior_policy", "sampling", "request_template",
        "reward", "split_seed", "eval_ratio", "transport", "groups",
    }
    if set(value) != required:
        raise ValueError("online rollout plan has missing or unknown top-level fields")
    plan = copy.deepcopy(value)
    _text(plan["batch_id"], "batch_id")
    canonical_json(plan)  # Reject non-JSON values and NaN/Infinity up front.

    behavior = plan["behavior_policy"]
    behavior_keys = {
        "policy_id", "revision", "model_id", "model_revision",
        "model_sha256", "model_config_sha256",
    }
    if not isinstance(behavior, dict) or set(behavior) != behavior_keys:
        raise ValueError("behavior_policy has missing or unknown fields")
    for key in ("policy_id", "revision", "model_id", "model_revision"):
        _text(behavior[key], f"behavior_policy.{key}")
    _sha256(behavior["model_sha256"], "behavior_policy.model_sha256")
    _sha256(behavior["model_config_sha256"], "behavior_policy.model_config_sha256")

    sampling = plan["sampling"]
    if not isinstance(sampling, dict) \
            or set(sampling) != {"max_tokens", "temperature", "top_p"}:
        raise ValueError("sampling must contain exactly max_tokens/temperature/top_p")
    if not _integer(sampling["max_tokens"]) or sampling["max_tokens"] <= 0:
        raise ValueError("sampling.max_tokens must be a positive integer")
    if not _finite(sampling["temperature"]) \
            or not 0.0 <= float(sampling["temperature"]) <= 2.0 \
            or not _finite(sampling["top_p"]) \
            or not 0.0 < float(sampling["top_p"]) <= 1.0:
        raise ValueError("sampling temperature/top_p is outside the public contract")

    template = plan["request_template"]
    reserved = {"model", "messages", "stream", "session_id", "seed", *sampling}
    if not isinstance(template, dict) or set(template) & reserved:
        raise ValueError("request_template cannot replace scheduled request fields")
    reward = plan["reward"]
    if not isinstance(reward, dict) or set(reward) != {"spec"}:
        raise ValueError("reward must contain exactly one export-only spec")
    _text(reward["spec"], "reward.spec")
    if ":" not in reward["spec"]:
        raise ValueError("reward.spec must use module:attribute form")
    _text(plan["split_seed"], "split_seed")
    if not _finite(plan["eval_ratio"]) or not 0.0 <= float(plan["eval_ratio"]) <= 1.0:
        raise ValueError("eval_ratio must be in [0, 1]")

    transport = plan["transport"]
    transport_keys = {
        "timeout_s", "max_attempts", "retry_statuses",
        "retry_transport", "retry_timeouts",
    }
    if not isinstance(transport, dict) or set(transport) != transport_keys:
        raise ValueError("transport has missing or unknown fields")
    if not _finite(transport["timeout_s"]) or float(transport["timeout_s"]) <= 0:
        raise ValueError("transport.timeout_s must be finite and positive")
    if not _integer(transport["max_attempts"]) \
            or not 1 <= transport["max_attempts"] <= 8:
        raise ValueError("transport.max_attempts must be in [1, 8]")
    statuses = transport["retry_statuses"]
    if not isinstance(statuses, list) \
            or any(not _integer(status) or not 400 <= status <= 599
                   for status in statuses) or len(statuses) != len(set(statuses)):
        raise ValueError("transport.retry_statuses must be unique HTTP error integers")
    if not isinstance(transport["retry_transport"], bool) \
            or not isinstance(transport["retry_timeouts"], bool):
        raise ValueError("transport retry switches must be boolean")

    groups = plan["groups"]
    if not isinstance(groups, list) or not groups:
        raise ValueError("plan requires at least one prompt group")
    prompt_ids = set()
    group_size = None
    for group_index, group in enumerate(groups):
        group_keys = {"prompt_id", "messages", "expected_visual_modalities", "seeds"}
        if not isinstance(group, dict) or set(group) != group_keys:
            raise ValueError(f"group {group_index} has missing or unknown fields")
        _text(group["prompt_id"], f"group {group_index} prompt_id")
        if group["prompt_id"] in prompt_ids:
            raise ValueError("prompt_id values must be unique within a plan")
        prompt_ids.add(group["prompt_id"])
        messages = group["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"group {group_index} messages must be non-empty")
        for message in messages:
            if not isinstance(message, dict) \
                    or not isinstance(message.get("role"), str) \
                    or not message["role"] or "content" not in message:
                raise ValueError(f"group {group_index} contains an invalid message")
        labels = group["expected_visual_modalities"]
        if not isinstance(labels, list) \
                or any(label not in ("image", "video") for label in labels):
            raise ValueError("expected_visual_modalities accepts only image/video")
        seeds = group["seeds"]
        if not isinstance(seeds, list) or len(seeds) < 2 \
                or any(not _integer(seed) or not _INT64_MIN <= seed <= _INT64_MAX
                       for seed in seeds) or len(seeds) != len(set(seeds)):
            raise ValueError("each group needs unique signed-int64 seeds and size >= 2")
        if group_size is None:
            group_size = len(seeds)
        elif len(seeds) != group_size:
            raise ValueError("all online GRPO groups must have the same group size")
    return plan


def behavior_policy_sha256(plan):
    return sha256_bytes(canonical_json(plan["behavior_policy"]))


def sampling_sha256(plan):
    return sha256_bytes(canonical_json(plan["sampling"]))


def plan_sha256(plan):
    return sha256_bytes(canonical_json(plan))


def build_schedule(plan):
    """Return stable group/candidate order and exact public request bodies."""
    plan = prepare_plan(plan)
    behavior_sha = behavior_policy_sha256(plan)
    sampling_sha = sampling_sha256(plan)
    schedule = []
    schedule_identity = []
    for group_index, group in enumerate(plan["groups"]):
        group_identity = {
            "batch_id": plan["batch_id"], "group_index": group_index,
            "prompt_id": group["prompt_id"], "messages": group["messages"],
            "seeds": group["seeds"], "request_template": plan["request_template"],
            "behavior_policy_sha256": behavior_sha,
        }
        group_id = sha256_bytes(canonical_json(group_identity))[:32]
        rollout_identity = []
        for rollout_index, seed in enumerate(group["seeds"]):
            session_id = f"online-{group_id}-{rollout_index}"
            body = copy.deepcopy(plan["request_template"])
            body.update({
                "model": plan["behavior_policy"]["model_id"],
                "messages": copy.deepcopy(group["messages"]),
                "stream": True, "session_id": session_id,
                **copy.deepcopy(plan["sampling"]), "seed": seed,
            })
            request_sha = sha256_bytes(canonical_json(body))
            rollout_id = sha256_bytes(canonical_json({
                "group_id": group_id, "rollout_index": rollout_index,
                "seed": seed, "request_sha256": request_sha,
                "behavior_policy_sha256": behavior_sha,
            }))[:32]
            item = {
                "schedule_index": len(schedule), "group_index": group_index,
                "rollout_index": rollout_index, "group_id": group_id,
                "rollout_id": rollout_id, "prompt_id": group["prompt_id"],
                "seed": seed, "session_id": session_id, "body": body,
                "request_sha256": request_sha,
                "expected_visual_modalities": list(
                    group["expected_visual_modalities"]),
            }
            schedule.append(item)
            rollout_identity.append({
                "rollout_id": rollout_id, "rollout_index": rollout_index,
                "seed": seed, "request_sha256": request_sha,
            })
        schedule_identity.append({
            "group_id": group_id, "group_index": group_index,
            "prompt_id": group["prompt_id"], "rollouts": rollout_identity,
        })
    return schedule, sha256_bytes(canonical_json(schedule_identity))


def _provenance(item, plan, schedule_sha, attempt):
    return {
        "contract": PROVENANCE_CONTRACT, "mode": "online",
        "batch_id": plan["batch_id"], "plan_sha256": plan_sha256(plan),
        "schedule_sha256": schedule_sha,
        "group_id": item["group_id"], "group_index": item["group_index"],
        "rollout_id": item["rollout_id"],
        "rollout_index": item["rollout_index"], "attempt": attempt,
        "seed": item["seed"], "request_sha256": item["request_sha256"],
        "behavior_policy": copy.deepcopy(plan["behavior_policy"]),
        "behavior_policy_sha256": behavior_policy_sha256(plan),
        "sampling_sha256": sampling_sha256(plan),
    }


def _expected_attestation(item, plan, *, http_status, behavior_evidence, attempt):
    behavior = plan["behavior_policy"]
    result = {
        "http_status": http_status, "contract": CONTRACT_VERSION,
        "generation": "fresh",
        "batch_id": plan["batch_id"], "group_id": item["group_id"],
        "rollout_id": item["rollout_id"],
        "rollout_index": str(item["rollout_index"]), "attempt": str(attempt),
        "request_sha256": item["request_sha256"],
        "behavior_policy_sha256": behavior_policy_sha256(plan),
        "behavior_policy_revision": behavior["revision"],
        "model_id": behavior["model_id"], "model_revision": behavior["model_revision"],
        "model_sha256": behavior["model_sha256"],
        "model_config_sha256": behavior["model_config_sha256"],
        "sampling_sha256": sampling_sha256(plan),
        "behavior_evidence": behavior_evidence,
    }
    result["attestation_sha256"] = sha256_bytes(canonical_json(result))
    return result


def _request_headers(item, plan, attempt):
    expected = _expected_attestation(
        item, plan, http_status=200, behavior_evidence="not_exposed", attempt=attempt)
    return {RESPONSE_HEADERS[key]: str(expected[key])
            for key in RESPONSE_HEADERS if key != "behavior_evidence"}


def _attestation_from_response(observed, item, plan, attempt):
    if not isinstance(observed, dict) or not _integer(observed.get("status")):
        raise ValueError("endpoint response observation is invalid")
    headers = observed.get("headers")
    if not isinstance(headers, dict):
        raise ValueError("endpoint response headers are missing")
    actual = {key: headers.get(name) for key, name in RESPONSE_HEADERS.items()}
    missing = sorted(key for key, value in actual.items() if value is None)
    if missing:
        raise ValueError("endpoint lacks online rollout attestation: " + ", ".join(missing))
    if actual["behavior_evidence"] not in ("not_exposed", "exposed"):
        raise ValueError("endpoint behavior-evidence coverage is invalid")
    expected = _expected_attestation(
        item, plan, http_status=observed["status"],
        behavior_evidence=actual["behavior_evidence"], attempt=attempt)
    for key in RESPONSE_HEADERS:
        if actual[key] != str(expected[key]):
            raise ValueError(f"endpoint response drift or cross-group binding: {key}")
    return expected


def _raw_chunks(record, artifact_dir):
    raw = validate_replay_metadata(record, artifact_dir=artifact_dir, require_raw=True)
    capture = RawSSECapture.from_metadata(record["sse_replay"], raw)
    chunks = []
    for frame_index, payload in capture.payload_frames:
        if payload == "[DONE]":
            continue
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError(f"raw SSE data frame {frame_index} is not an object")
        chunks.append(value)
    return chunks


def _behavior_evidence(record, coverage, artifact_dir, chunks=None):
    chunks = _raw_chunks(record, artifact_dir) if chunks is None else chunks
    blocks = [chunk["behavior"] for chunk in chunks if "behavior" in chunk]
    if coverage == "not_exposed":
        if blocks:
            raise ValueError("endpoint supplied behavior values while declaring not_exposed")
        return {
            "status": "NOT_COVERED", "token_ids": None,
            "logprobs": None, "mask": None,
        }
    if len(blocks) != 1 or "behavior" not in chunks[-1]:
        raise ValueError("exposed behavior evidence requires one final SSE block")
    evidence = blocks[0]
    if not isinstance(evidence, dict) or set(evidence) != BEHAVIOR_KEYS \
            or evidence.get("format") != BEHAVIOR_FORMAT:
        raise ValueError("behavior evidence has missing or unknown fields")
    _sha256(evidence["tokenizer_sha256"], "behavior.tokenizer_sha256")
    token_ids, logprobs, mask = (
        evidence["token_ids"], evidence["logprobs"], evidence["mask"])
    if not all(isinstance(values, list) and values for values in
               (token_ids, logprobs, mask)) \
            or not len(token_ids) == len(logprobs) == len(mask):
        raise ValueError("behavior token/logprob/mask arrays must be non-empty and aligned")
    if any(not _integer(token) or token < 0 for token in token_ids) \
            or any(not _finite(value) or value > 0.0 for value in logprobs) \
            or any(not _integer(value) or value not in (0, 1) for value in mask) \
            or not any(mask):
        raise ValueError("behavior evidence contains invalid ids/logprobs/mask")
    return {
        "status": "COVERED", "format": BEHAVIOR_FORMAT,
        "tokenizer_sha256": evidence["tokenizer_sha256"],
        "token_ids": list(token_ids), "logprobs": list(logprobs), "mask": list(mask),
        "token_count": len(token_ids), "masked_token_count": sum(mask),
        "evidence_sha256": sha256_bytes(canonical_json(evidence)),
    }


def _without_cas_filenames(value):
    """Compare request references without creating missing CAS files."""
    if isinstance(value, list):
        return [_without_cas_filenames(item) for item in value]
    if not isinstance(value, dict):
        return value
    digest = value.get("sha256")
    content_addressed = isinstance(digest, str) and len(digest) == 64 \
        and not set(digest) - _HEX_64
    return {
        key: _without_cas_filenames(item) for key, item in value.items()
        if not (key == "ref" and content_addressed)
    }


def validate_online_record(record, item, plan, schedule_sha, artifact_dir, *,
                           require_completed=True):
    """Rebind one exported record to its exact request, response, and raw bytes."""
    if not isinstance(record, dict) or record.get("schema_version") != 2:
        raise ValueError("online GRPO accepts only schema-v2 records")
    provenance = record.get("rollout_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("online record lacks rollout_provenance")
    attempt = provenance.get("attempt")
    if not _integer(attempt) or not 0 <= attempt < plan["transport"]["max_attempts"]:
        raise ValueError("online record has an invalid attempt index")
    expected = _provenance(item, plan, schedule_sha, attempt)
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"online record provenance mismatch: {key}")
    if require_completed and record.get("terminal_state") != "completed":
        raise ValueError("cancelled/failed online responses cannot enter GRPO groups")
    if record.get("session_id") != item["session_id"] \
            or record.get("turn_index") != item["schedule_index"] \
            or record.get("endpoint") != "/v1/chat/completions":
        raise ValueError("online record session/turn/endpoint differs from the schedule")
    # Build expected references without a CAS destination: validation is
    # read-only, and replay_record independently verifies every actual ref.
    params, option_inputs = request_parameter_references(item["body"], None)
    inputs = input_references(item["body"]["messages"], None) + option_inputs
    if _without_cas_filenames(record.get("request_params")) \
            != _without_cas_filenames(params) \
            or _without_cas_filenames(record.get("inputs")) \
            != _without_cas_filenames(inputs):
        raise ValueError("online record request/messages differ from the deterministic plan")
    response = provenance.get("response")
    if not isinstance(response, dict) or response.get("http_status") != 200:
        raise ValueError("raw-SSE online records require an attested HTTP 200 response")
    expected_response = _expected_attestation(
        item, plan, http_status=response.get("http_status"),
        behavior_evidence=response.get("behavior_evidence"),
        attempt=provenance["attempt"])
    if response != expected_response:
        raise ValueError("online record response attestation was changed or is incomplete")
    chunks = _raw_chunks(record, artifact_dir)
    if not chunks:
        raise ValueError("online response has no raw SSE data frames")
    request_id = record.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("online response lacks a server request id")
    for chunk in chunks:
        if chunk.get("id") != request_id \
                or chunk.get("model") != plan["behavior_policy"]["model_id"]:
            raise ValueError("raw SSE contains stale or mixed request/model frames")
        unified = chunk.get("unified")
        if isinstance(unified, dict) and (
                unified.get("request_id") != request_id
                or unified.get("session_id") != item["session_id"]):
            raise ValueError("raw SSE inner request/session identity is stale or mixed")
    behavior = _behavior_evidence(
        record, response["behavior_evidence"], artifact_dir, chunks=chunks)
    if provenance.get("behavior") != behavior:
        raise ValueError("behavior evidence differs from its exact raw SSE block")
    return record


def _retryable(record, transport, response, plan):
    policy = plan["transport"]
    if transport is not None and transport["kind"] == "protocol":
        return False
    if response is not None and response["http_status"] in policy["retry_statuses"]:
        return True
    if transport is None:
        return False
    if transport["kind"] == "timeout":
        return policy["retry_timeouts"]
    if transport["kind"] == "transport":
        return policy["retry_transport"]
    return False


def execute_plan(plan, endpoint, output_dir, *, clock_factory=None):
    """Execute a plan through public HTTP and export only complete online groups."""
    plan = prepare_plan(plan)
    schedule, schedule_sha = build_schedule(plan)
    if not isinstance(endpoint, str) or "://" in endpoint or "@" in endpoint \
            or "/" in endpoint or any(char.isspace() for char in endpoint):
        raise ValueError("endpoint must be a host:port without scheme or credentials")
    host, separator, port = endpoint.rpartition(":")
    if not separator or not host or not port.isdigit() \
            or not 1 <= int(port) <= 65535:
        raise ValueError("endpoint must be a host:port without scheme or credentials")
    output = _new_directory(output_dir)
    plan_bytes = _canonical_line(plan)
    _write_bytes(output / "plan.json", plan_bytes)
    attempt_recorder = TrajectoryRecorder(output / "attempts.jsonl")
    attempts = []
    selected = {}
    request_ids = set()

    for item in schedule:
        transport_policy = plan["transport"]
        for attempt in range(transport_policy["max_attempts"]):
            observed_response = []
            observed_transport = []
            provenance = _provenance(item, plan, schedule_sha, attempt)
            record = record_chat_completion(
                endpoint, copy.deepcopy(item["body"]), attempt_recorder,
                item["session_id"], headers=_request_headers(item, plan, attempt),
                timeout=float(transport_policy["timeout_s"]), capture_replay=True,
                replay_clock=clock_factory() if clock_factory is not None else None,
                response_observer=observed_response.append,
                transport_observer=observed_transport.append,
                turn_index=item["schedule_index"],
                extra_fields={
                    "ablation": {
                        "prompt_id": item["prompt_id"],
                        "expected_visual_modalities": item[
                            "expected_visual_modalities"],
                    },
                    "rollout_provenance": provenance,
                })
            attempts.append(record)
            provenance = record["rollout_provenance"]
            response = None
            if observed_response:
                response = _attestation_from_response(
                    observed_response[0], item, plan, attempt)
                provenance["response"] = response
            transport = observed_transport[0] if observed_transport else None
            if transport is not None:
                provenance["transport"] = transport
            _write_jsonl(attempt_recorder.jsonl_path, attempts)

            if record["terminal_state"] == "completed":
                if response is None or response["http_status"] != 200 \
                        or record.get("sse_replay") is None:
                    raise ValueError("completed online rollout requires raw HTTP 200 SSE")
                provenance["behavior"] = _behavior_evidence(
                    record, response["behavior_evidence"], attempt_recorder.artifact_dir)
                validate_online_record(
                    record, item, plan, schedule_sha, attempt_recorder.artifact_dir)
                if record["request_id"] in request_ids:
                    raise ValueError("duplicate or stale server request id across rollouts")
                request_ids.add(record["request_id"])
                selected[item["rollout_id"]] = record
                _write_jsonl(attempt_recorder.jsonl_path, attempts)
                break
            if record["terminal_state"] == "cancelled":
                if response is None or record.get("sse_replay") is None:
                    raise ValueError("cancelled online rollout lacks raw terminal evidence")
                provenance["behavior"] = _behavior_evidence(
                    record, response["behavior_evidence"], attempt_recorder.artifact_dir)
                validate_online_record(
                    record, item, plan, schedule_sha, attempt_recorder.artifact_dir,
                    require_completed=False)
                _write_jsonl(attempt_recorder.jsonl_path, attempts)
                raise ValueError("cancelled online response rejected; no GRPO group exported")
            if attempt + 1 >= transport_policy["max_attempts"] \
                    or not _retryable(record, transport, response, plan):
                raise ValueError(
                    f"online rollout {item['rollout_id']} failed without a permitted retry: "
                    f"{record.get('error')}")
        else:
            raise ValueError("online rollout exhausted attempts without a response")

    recorder = TrajectoryRecorder(output / "trajectories.jsonl")
    replayed = []
    for item in schedule:
        record = selected[item["rollout_id"]]
        replayed.append(replay_record(
            record, recorder=recorder, artifact_dir=attempt_recorder.artifact_dir))
    return _export_replayed(plan, schedule, schedule_sha, replayed, output)


def _score_records(plan, records):
    reward = load_reward(plan["reward"]["spec"])
    for record in records:
        score, components = reward.evaluate(copy.deepcopy(record))
        if not _finite(score):
            raise ValueError("every online GRPO rollout requires a finite export-only reward")
        block = {
            "schema_version": REWARD_SCHEMA_VERSION, "name": reward.name,
            "score": float(score),
        }
        if components is not None:
            block["components"] = components
        record["reward"] = block
    return reward


def _annotate_grpo_pack(plan, schedule, schedule_sha, records, pack_dir, reward):
    by_identity = {}
    by_rollout = {}
    for item, record in zip(schedule, records):
        identity = (record["session_id"], record["request_id"], record["turn_index"])
        if identity in by_identity or item["rollout_id"] in by_rollout:
            raise ValueError("duplicate source identity in accepted online rollouts")
        candidate = dataset._extract_candidate(record, verified_on_disk=True)
        by_identity[identity] = (item, record, candidate)
        by_rollout[item["rollout_id"]] = (item, record, candidate)

    loaded = dataset.audit_dataset_pack(pack_dir)
    manifest_path = Path(pack_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["counts"]["accepted_verified_completed"] != len(records) \
            or any(manifest["counts"]["rejected"].values()):
        raise ValueError("online source records were rejected or dropped before packing")
    grouped = {}
    for split in ("train", "eval"):
        name = f"grpo_{split}"
        rows = [json.loads(line) for line in
                (Path(pack_dir) / f"{name}.jsonl").read_text(
                    encoding="utf-8").splitlines() if line]
        if len(rows) != manifest["outputs"][name]["record_count"]:
            raise ValueError("GRPO pack row count drift")
        for row in rows:
            matched = []
            for rollout in row.get("rollouts", []):
                source = rollout.get("provenance") or {}
                identity = (source.get("source_session_id"),
                            source.get("source_request_id"),
                            source.get("source_turn_index"))
                if identity not in by_identity:
                    raise ValueError("GRPO pack contains an unknown or stale source rollout")
                matched.append(by_identity[identity])
            group_ids = {entry[0]["group_id"] for entry in matched}
            if len(group_ids) != 1:
                raise ValueError("GRPO pack crosses scheduled prompt groups")
            group_id = next(iter(group_ids))
            if group_id in grouped:
                raise ValueError("scheduled group appears in more than one GRPO row")
            expected_items = [item for item in schedule if item["group_id"] == group_id]
            expected_ids = [item["rollout_id"] for item in expected_items]
            actual_ids = [entry[0]["rollout_id"] for entry in matched]
            if row.get("reward_name") != reward.name:
                raise ValueError("GRPO group reward name differs from the export reward")
            if row.get("group_size") != len(expected_items) \
                    or len(actual_ids) != len(set(actual_ids)) \
                    or set(actual_ids) != set(expected_ids):
                raise ValueError("GRPO group has missing, duplicate, or extra rollouts")
            expected_order = sorted(
                matched, key=lambda entry: (
                    -entry[1]["reward"]["score"], dataset._candidate_tie_key(entry[2])))
            if [entry[0]["rollout_id"] for entry in expected_order] != actual_ids:
                raise ValueError("GRPO group differs from deterministic reward/source order")
            scores = [entry[1]["reward"]["score"] for entry in matched]
            advantages, mean, std = dataset.compute_grpo_advantages(scores)
            if row.get("group_mean") != mean or row.get("group_std") != std \
                    or row.get("has_reward_variance") is not (std > 0.0):
                raise ValueError("GRPO group reward statistics mismatch")
            lead_candidate = expected_order[0][2]
            if row.get("prompt_sha256") != lead_candidate["prompt_sha256"] \
                    or row.get("prompt_text") != lead_candidate["prompt_text"] \
                    or row.get("prompt_parts") != lead_candidate["prompt_parts"] \
                    or row.get("input_media_refs") != lead_candidate["input_media_refs"]:
                raise ValueError("GRPO group prompt differs from its online source")
            expected_group_id = dataset.sha256_text(
                row["prompt_sha256"] + ":" + reward.name + ":"
                + ",".join(entry[2]["target_sha256"] for entry in expected_order))[:32]
            if row.get("group_id") != expected_group_id:
                raise ValueError("GRPO group_id does not bind ordered online membership")
            for rank, (rollout, entry, advantage) in enumerate(
                    zip(row["rollouts"], matched, advantages)):
                item, record, candidate = entry
                recomputed = reward.score(copy.deepcopy(record))
                if rollout.get("rank") != rank \
                        or rollout.get("score") != record["reward"]["score"] \
                        or recomputed != rollout["score"] \
                        or rollout.get("advantage") != advantage:
                    raise ValueError("GRPO rollout rank/reward/advantage mismatch")
                for field in ("target_sha256", "assistant_target", "clean_text",
                              "visual_slots", "modality_route", "output_media_refs",
                              "provenance"):
                    if rollout.get(field) != candidate[field]:
                        raise ValueError(f"GRPO rollout differs from source field {field}")
                rollout["rollout_provenance"] = copy.deepcopy(
                    record["rollout_provenance"])
            row["rollout_provenance"] = {
                "contract": PROVENANCE_CONTRACT, "mode": "online",
                "batch_id": plan["batch_id"], "plan_sha256": plan_sha256(plan),
                "schedule_sha256": schedule_sha, "schedule_group_id": group_id,
                "group_index": matched[0][0]["group_index"],
                "behavior_policy": copy.deepcopy(plan["behavior_policy"]),
                "behavior_policy_sha256": behavior_policy_sha256(plan),
                "sampling_sha256": sampling_sha256(plan),
            }
            grouped[group_id] = row
        payload = b"".join(_canonical_line(row) for row in rows)
        _write_bytes(Path(pack_dir) / f"{name}.jsonl", payload)
        manifest["outputs"][name] = {
            "path": f"{name}.jsonl", "sha256": sha256_bytes(payload),
            "bytes": len(payload), "record_count": len(rows),
        }
    if set(grouped) != {item["group_id"] for item in schedule}:
        raise ValueError("not every scheduled online prompt produced exactly one GRPO group")
    coverage = {record["rollout_provenance"]["behavior"]["status"]
                for record in records}
    if len(coverage) != 1:
        raise ValueError("behavior-evidence coverage drifted within one online batch")
    manifest["provenance_mode"] = "online"
    manifest["rollout_provenance"] = {
        "contract": PROVENANCE_CONTRACT, "plan_sha256": plan_sha256(plan),
        "schedule_sha256": schedule_sha,
        "behavior_policy_sha256": behavior_policy_sha256(plan),
        "sampling_sha256": sampling_sha256(plan),
        "behavior_logprobs": next(iter(coverage)),
    }
    manifest["claims"].update({
        "online_rollout_export": "VALIDATED",
        "behavior_logprobs": next(iter(coverage)),
        "model_quality": "NOT_RUN",
        "model_identity_verification": "ENDPOINT_ATTESTATION_ONLY",
    })
    dataset._write_manifest_atomic(pack_dir, manifest)
    audit = dataset.audit_dataset_pack(pack_dir)
    return manifest, audit["manifest_sha256"], next(iter(coverage))


def _export_replayed(plan, schedule, schedule_sha, records, output):
    if len(records) != len(schedule):
        raise ValueError("accepted online trajectory count differs from schedule")
    recorder_path = Path(output) / "trajectories.jsonl"
    reward = _score_records(plan, records)
    _write_jsonl(recorder_path, records)
    manifest = dataset.build_dataset(
        [recorder_path], Path(output) / "pack", mode="preference",
        seed=plan["split_seed"], eval_ratio=float(plan["eval_ratio"]),
        reward_spec=plan["reward"]["spec"],
        min_group_size=len(plan["groups"][0]["seeds"]), strict_artifacts=True)
    if manifest["outputs"]["grpo_train"]["record_count"] \
            + manifest["outputs"]["grpo_eval"]["record_count"] != len(plan["groups"]):
        raise ValueError("dataset builder dropped or split an online GRPO group")
    pack_manifest, pack_sha, coverage = _annotate_grpo_pack(
        plan, schedule, schedule_sha, records, Path(output) / "pack", reward)
    result = {
        "schema": EXPORT_SCHEMA, "status": "PASS", "provenance_mode": "online",
        "plan_sha256": plan_sha256(plan), "schedule_sha256": schedule_sha,
        "behavior_policy": copy.deepcopy(plan["behavior_policy"]),
        "behavior_policy_sha256": behavior_policy_sha256(plan),
        "sampling_sha256": sampling_sha256(plan),
        "reward": {"spec": plan["reward"]["spec"], "name": reward.name},
        "group_count": len(plan["groups"]), "rollout_count": len(records),
        "group_size": len(plan["groups"][0]["seeds"]),
        "behavior_logprobs": coverage,
        "trajectory": {
            "path": "trajectories.jsonl",
            "sha256": sha256_bytes(recorder_path.read_bytes()),
            "bytes": recorder_path.stat().st_size, "record_count": len(records),
        },
        "pack_manifest": {
            "path": "pack/manifest.json", "sha256": pack_sha,
            "bytes": (Path(output) / "pack/manifest.json").stat().st_size,
        },
        "claims": {
            "optimizer": "NOT_RUN", "REAL_TRAINING": "NOT_RUN",
            "model_quality": "NOT_RUN", "generalized_improvement": "NOT_RUN",
            "behavior_logprobs": coverage,
            "model_identity_verification": "ENDPOINT_ATTESTATION_ONLY",
        },
    }
    if pack_manifest["provenance_mode"] != "online":
        raise ValueError("online pack lost its provenance mode")
    _write_document(Path(output) / "rollout_manifest.json", result)
    return result


def _accepted_relative_files(root):
    files = []
    for name in ("plan.json", "trajectories.jsonl", "rollout_manifest.json",
                 "rollout_manifest.json.sha256"):
        files.append(Path(name))
    for directory in ("artifacts", "pack"):
        base = Path(root) / directory
        files.extend(path.relative_to(root) for path in base.rglob("*") if path.is_file())
    return sorted(files, key=lambda path: path.as_posix())


def replay_bundle(source_dir, output_dir):
    """Fresh-process replay: regenerate accepted records and packs byte-for-byte."""
    source = Path(source_dir)
    source_manifest_path = source / "rollout_manifest.json"
    sidecar = Path(str(source_manifest_path) + ".sha256")
    if not source_manifest_path.is_file() or not sidecar.is_file() \
            or sidecar.read_text(encoding="ascii").strip() \
            != dataset.sha256_file(source_manifest_path):
        raise ValueError("online rollout manifest or SHA-256 sidecar is invalid")
    plan_bytes = (source / "plan.json").read_bytes()
    plan = prepare_plan(json.loads(plan_bytes.decode("utf-8")))
    if plan_bytes != _canonical_line(plan):
        raise ValueError("accepted plan.json is not canonical UTF-8 JSON")
    schedule, schedule_sha = build_schedule(plan)
    records = [json.loads(line) for line in
               (source / "trajectories.jsonl").read_text(
                   encoding="utf-8").splitlines() if line]
    if len(records) != len(schedule):
        raise ValueError("accepted trajectory count differs from the online schedule")
    by_rollout = {}
    for record in records:
        provenance = record.get("rollout_provenance") or {}
        rollout_id = provenance.get("rollout_id")
        if rollout_id in by_rollout:
            raise ValueError("duplicate rollout id in accepted online trajectories")
        by_rollout[rollout_id] = record
    if set(by_rollout) != {item["rollout_id"] for item in schedule}:
        raise ValueError("accepted online trajectories contain missing or stale rollouts")

    output = _new_directory(output_dir)
    _write_bytes(output / "plan.json", plan_bytes)
    recorder = TrajectoryRecorder(output / "trajectories.jsonl")
    replayed = []
    request_ids = set()
    for item in schedule:
        record = by_rollout[item["rollout_id"]]
        validate_online_record(
            record, item, plan, schedule_sha, source / "artifacts")
        if record["request_id"] in request_ids:
            raise ValueError("duplicate or stale server request id in accepted replay")
        request_ids.add(record["request_id"])
        replayed.append(replay_record(
            record, recorder=recorder, artifact_dir=source / "artifacts"))
    result = _export_replayed(plan, schedule, schedule_sha, replayed, output)
    expected_files = _accepted_relative_files(source)
    actual_files = _accepted_relative_files(output)
    if actual_files != expected_files:
        raise ValueError("fresh-process replay file set differs from accepted online bundle")
    for relative in expected_files:
        if (source / relative).read_bytes() != (output / relative).read_bytes():
            raise ValueError(f"fresh-process online replay differs: {relative.as_posix()}")
    return result


def _parser():
    parser = argparse.ArgumentParser(
        description="Generate/validate online GRPO groups without serving or ML imports")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="execute a plan against an attesting endpoint")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--endpoint", required=True, help="host:port; no URL credentials")
    run.add_argument("--out", type=Path, required=True)
    replay = commands.add_parser("replay", help="byte-replay an accepted online bundle")
    replay.add_argument("--input", type=Path, required=True)
    replay.add_argument("--out", type=Path, required=True)
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            plan = prepare_plan(json.loads(args.plan.read_text(encoding="utf-8")))
            result = execute_plan(plan, args.endpoint, args.out)
            print(f"ONLINE_GRPO_EXPORT: PASS groups={result['group_count']} "
                  f"rollouts={result['rollout_count']} "
                  f"behavior_logprobs={result['behavior_logprobs']}")
        else:
            result = replay_bundle(args.input, args.out)
            print(f"ONLINE_GRPO_REPLAY: PASS groups={result['group_count']} "
                  f"rollouts={result['rollout_count']} byte_identical=true")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"online rollout error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
