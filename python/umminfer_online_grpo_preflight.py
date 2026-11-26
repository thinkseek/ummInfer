# SPDX-License-Identifier: Apache-2.0
"""Deterministic CPU-only online-GRPO preflight and run-identity adapter.

This module consumes an already accepted ``umminfer_online_rollout`` bundle.  It
never contacts an endpoint and never imports a model/optimizer framework.  Fresh
raw-SSE replay and export/pack reconstruction run before any output is emitted.
Behavior ids/logprobs/masks are the sampled values from the final raw SSE frame;
post-hoc token_export ids are used only by the independent alignment validator,
never as replacements for sampled behavior values.

The objective document contains explicit *synthetic arithmetic fixtures*, not
current/reference model outputs.  Current and frozen-reference forwards, real
optimizer state, GPU execution, serving, and model quality remain NOT_RUN.
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

import umminfer_behavior_token_alignment as alignment  # noqa: E402
import umminfer_online_rollout as online  # noqa: E402
import umminfer_sft_dataset as dataset  # noqa: E402

CONFIG_SCHEMA = "umminfer-online-grpo-preflight-config-v1"
RUN_SCHEMA = "umminfer-online-grpo-run-identity-v1"
SCHEDULE_SCHEMA = "umminfer-online-grpo-preflight-schedule-v1"
OBJECTIVE_SCHEMA = "umminfer-online-grpo-objective-fixtures-v1"
CHECKPOINT_SCHEMA = "umminfer-online-grpo-preflight-checkpoint-v1"
MANIFEST_SCHEMA = "umminfer-online-grpo-preflight-manifest-v1"
RESULT_SCHEMA = "umminfer-online-grpo-preflight-result-v1"
VERIFY_SCHEMA = "umminfer-online-grpo-preflight-verification-v1"
POLICY_KEYS = {
    "policy_id", "revision", "model_id", "model_revision", "model_sha256",
    "model_config_sha256", "tokenizer_sha256",
}
SOURCE_FILES = (
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
BASE_DOCUMENTS = ("config.json", "run_identity.json", "schedule.json",
                  "objective_fixtures.json")
_ML_MODULES = frozenset(("torch", "transformers", "numpy", "mlx"))


def _canonical(value):
    return online.canonical_json(value)


def _hash_bytes(payload):
    return online.sha256_bytes(payload)


def _hash(value):
    return _hash_bytes(_canonical(value))


def _document_bytes(value):
    return _canonical(value) + b"\n"


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(float(value)))


def _reject_ml_imports():
    loaded = sorted(name for name in _ML_MODULES if name in sys.modules)
    if loaded:
        raise ValueError("CPU-only online preflight refuses loaded ML modules: "
                         + ",".join(loaded))


def _policy(value, label):
    if not isinstance(value, dict) or set(value) != POLICY_KEYS:
        raise ValueError(f"{label} has missing or unknown identity fields")
    result = copy.deepcopy(value)
    for key in ("policy_id", "revision", "model_id", "model_revision"):
        online._text(result[key], f"{label}.{key}")
    for key in ("model_sha256", "model_config_sha256", "tokenizer_sha256"):
        online._sha256(result[key], f"{label}.{key}")
    return result


def _policy_sha256(policy):
    return _hash(policy)


def _validate_config(value):
    required = {
        "schema", "current_policy", "reference_policy", "objective",
        "prompt_token_ids", "groups_per_step",
    }
    if not isinstance(value, dict) or set(value) != required \
            or value.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"online preflight config must use {CONFIG_SCHEMA}")
    config = copy.deepcopy(value)
    config["current_policy"] = _policy(config["current_policy"], "current_policy")
    config["reference_policy"] = _policy(config["reference_policy"], "reference_policy")
    objective = config["objective"]
    objective_keys = {
        "clip_eps", "kl_coef", "log_ratio_clamp",
        "current_logprob_offsets", "reference_logprob_offsets",
    }
    if not isinstance(objective, dict) or set(objective) != objective_keys:
        raise ValueError("objective has missing or unknown fields")
    if not _finite(objective["clip_eps"]) \
            or not 0.0 < float(objective["clip_eps"]) < 1.0:
        raise ValueError("objective.clip_eps must be in (0, 1)")
    if not _finite(objective["kl_coef"]) or float(objective["kl_coef"]) < 0.0:
        raise ValueError("objective.kl_coef must be finite and non-negative")
    if not _finite(objective["log_ratio_clamp"]) \
            or float(objective["log_ratio_clamp"]) <= 0.0:
        raise ValueError("objective.log_ratio_clamp must be finite and positive")
    for key in ("current_logprob_offsets", "reference_logprob_offsets"):
        offsets = objective[key]
        if not isinstance(offsets, list) or not offsets \
                or any(not _finite(item) for item in offsets):
            raise ValueError(f"objective.{key} must be a non-empty finite numeric list")
    prompts = config["prompt_token_ids"]
    if not isinstance(prompts, dict) or not prompts:
        raise ValueError("prompt_token_ids must bind every prompt SHA-256")
    for prompt_sha, ids in prompts.items():
        online._sha256(prompt_sha, "prompt_token_ids key")
        if not isinstance(ids, list) or not ids or len(ids) > 8192 \
                or any(not _integer(token) or token < 0 for token in ids):
            raise ValueError(f"prompt_token_ids[{prompt_sha}] is not a bounded id list")
    if not _integer(config["groups_per_step"]) or config["groups_per_step"] <= 0:
        raise ValueError("groups_per_step must be a positive integer")
    _canonical(config)  # Reject non-JSON values and non-finite floats recursively.
    return config


def load_config(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError("online preflight config does not exist")
    return _validate_config(json.loads(path.read_text(encoding="utf-8")))


def _source_fingerprints():
    files = []
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise ValueError(f"missing online preflight source dependency: {relative}")
        files.append({
            "path": relative, "sha256": dataset.sha256_file(path),
            "bytes": path.stat().st_size,
        })
    return files


def _read_jsonl(path):
    raw = Path(path).read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    if not records and raw:
        raise ValueError(f"JSONL file has no complete records: {Path(path).name}")
    return records


def _json_document(root, relative):
    path = Path(root) / relative
    value = json.loads(path.read_text(encoding="utf-8"))
    if path.read_bytes() != _document_bytes(value):
        raise ValueError(f"accepted document is not canonical JSON: {relative}")
    return value


def _source_file_identity(source):
    result = []
    for relative in online._accepted_relative_files(source):
        path = source / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError("accepted online source contains a missing or symbolic file")
        result.append({
            "path": relative.as_posix(), "sha256": dataset.sha256_file(path),
            "bytes": path.stat().st_size,
        })
    return result


def _extract_groups(replayed, plan, schedule, schedule_sha):
    rows = []
    for split in ("train", "eval"):
        for row in _read_jsonl(replayed / "pack" / f"grpo_{split}.jsonl"):
            if not isinstance(row, dict) or row.get("record_type") != "grpo_group" \
                    or row.get("schema_version") != 1:
                raise ValueError("online preflight accepts only schema-v1 GRPO groups")
            provenance = row.get("rollout_provenance")
            if not isinstance(provenance, dict) \
                    or provenance.get("contract") != online.PROVENANCE_CONTRACT \
                    or provenance.get("mode") != "online" \
                    or provenance.get("plan_sha256") != online.plan_sha256(plan) \
                    or provenance.get("schedule_sha256") != schedule_sha:
                raise ValueError("legacy/frozen-offline or stale GRPO group is forbidden")
            rows.append((split, row))
    by_group = {}
    for split, row in rows:
        group_id = row["rollout_provenance"].get("schedule_group_id")
        if not isinstance(group_id, str) or group_id in by_group:
            raise ValueError("duplicate or invalid online schedule group")
        by_group[group_id] = (split, row)
    scheduled_group_ids = []
    for item in schedule:
        if item["group_id"] not in scheduled_group_ids:
            scheduled_group_ids.append(item["group_id"])
    if set(by_group) != set(scheduled_group_ids) or len(rows) != len(plan["groups"]):
        raise ValueError("online pack has missing, stale, or extra groups")
    return [(group_id, *by_group[group_id]) for group_id in scheduled_group_ids]


def _independent_group_audit(row, split, items, records):
    """Rebind ordered pack membership to source records without trusting the row."""
    group_id = items[0]["group_id"]
    if any(item["group_id"] != group_id for item in items) or split not in ("train", "eval"):
        raise ValueError("cross-group or invalid-split online pack")
    rollouts = row.get("rollouts")
    if not isinstance(rollouts, list) or len(rollouts) != len(items):
        raise ValueError("online group is incomplete")
    by_rollout = {item["rollout_id"]: (item, records[item["rollout_id"]])
                  for item in items}
    matched = []
    for rank, rollout in enumerate(rollouts):
        if not isinstance(rollout, dict) or rollout.get("rank") != rank:
            raise ValueError("online group rank/order is not contiguous")
        provenance = rollout.get("rollout_provenance")
        if not isinstance(provenance, dict) \
                or provenance.get("contract") != online.PROVENANCE_CONTRACT \
                or provenance.get("mode") != "online":
            raise ValueError("legacy/frozen-offline rollout in online group")
        rollout_id = provenance.get("rollout_id")
        if rollout_id not in by_rollout:
            raise ValueError("stale, duplicate, or cross-group online member")
        item, record = by_rollout[rollout_id]
        if provenance != record.get("rollout_provenance") \
                or provenance.get("group_id") != group_id \
                or provenance.get("rollout_index") != item["rollout_index"] \
                or provenance.get("seed") != item["seed"] \
                or provenance.get("request_sha256") != item["request_sha256"]:
            raise ValueError("online member provenance differs from schedule/source")
        candidate = dataset._extract_candidate(record, verified_on_disk=True)
        for field in ("target_sha256", "assistant_target", "clean_text",
                      "visual_slots", "modality_route", "output_media_refs"):
            if rollout.get(field) != candidate[field]:
                raise ValueError(f"online member differs from source field {field}")
        matched.append((item, record, candidate))

    expected_order = sorted(
        matched, key=lambda entry: (
            -entry[1]["reward"]["score"], dataset._candidate_tie_key(entry[2])))
    if [entry[0]["rollout_id"] for entry in expected_order] \
            != [entry[0]["rollout_id"] for entry in matched]:
        raise ValueError("online group is reordered relative to reward/source order")
    scores = [entry[1]["reward"]["score"] for entry in matched]
    if any(not _finite(score) for score in scores):
        raise ValueError("online group reward must be finite")
    advantages, mean, std = dataset.compute_grpo_advantages(scores)
    if row.get("group_size") != len(items) or row.get("group_mean") != mean \
            or row.get("group_std") != std \
            or row.get("has_reward_variance") is not (std > 0.0):
        raise ValueError("online group size/reward statistics differ")
    for rollout, entry, score, advantage in zip(rollouts, matched, scores, advantages):
        if rollout.get("score") != score or rollout.get("advantage") != advantage:
            raise ValueError("online member score/advantage differs from exact group statistics")
    lead = matched[0][2]
    if row.get("prompt_sha256") != lead["prompt_sha256"] \
            or row.get("prompt_text") != lead["prompt_text"] \
            or row.get("prompt_parts") != lead["prompt_parts"] \
            or row.get("input_media_refs") != lead["input_media_refs"]:
        raise ValueError("online group prompt differs from its source exports")
    expected_group_id = dataset.sha256_text(
        row["prompt_sha256"] + ":" + row["reward_name"] + ":"
        + ",".join(entry[2]["target_sha256"] for entry in matched))[:32]
    if row.get("group_id") != expected_group_id:
        raise ValueError("online group id does not bind exact ordered targets")
    return matched, advantages, mean, std


def _mean(values):
    if not values or any(not _finite(value) for value in values):
        raise ValueError("log-probability fixture requires finite non-empty values")
    return math.fsum(float(value) for value in values) / len(values)


def _objective_metrics(policy_mean, behavior_mean, reference_mean, advantage, objective):
    """Online clipped surrogate with behavior ratio and separate reference KL."""
    clamp = float(objective["log_ratio_clamp"])
    clip_eps = float(objective["clip_eps"])
    raw_log_ratio = policy_mean - behavior_mean
    log_ratio = max(-clamp, min(clamp, raw_log_ratio))
    ratio = math.exp(log_ratio)
    clipped_ratio = max(1.0 - clip_eps, min(1.0 + clip_eps, ratio))
    surrogate_1 = ratio * advantage
    surrogate_2 = clipped_ratio * advantage
    selected = min(surrogate_1, surrogate_2)
    raw_kl_diff = reference_mean - policy_mean
    kl_diff = max(-clamp, min(clamp, raw_kl_diff))
    kl = math.exp(kl_diff) - kl_diff - 1.0
    loss = -selected + float(objective["kl_coef"]) * kl
    values = (raw_log_ratio, log_ratio, ratio, clipped_ratio, surrogate_1,
              surrogate_2, selected, raw_kl_diff, kl_diff, kl, loss)
    if not all(math.isfinite(value) for value in values) or ratio <= 0.0:
        raise ValueError("non-finite online GRPO ratio/surrogate/KL fixture")
    return {
        "policy_mean_logp": policy_mean, "behavior_mean_logp": behavior_mean,
        "reference_mean_logp": reference_mean,
        "raw_current_minus_behavior_logp": raw_log_ratio,
        "current_minus_behavior_logp": log_ratio, "ratio": ratio,
        "clipped_ratio": clipped_ratio, "advantage": advantage,
        "unclipped_surrogate": surrogate_1, "clipped_surrogate": surrogate_2,
        "selected_surrogate": selected, "clip_active": surrogate_1 > surrogate_2,
        "raw_reference_minus_current_logp": raw_kl_diff,
        "reference_minus_current_logp": kl_diff, "schulman_kl": kl,
        "objective_loss": loss,
    }


def _pack_member(evidence, prompt_ids):
    sampled_ids = evidence["token_ids"]
    mask = evidence["mask"]
    if len(sampled_ids) != len(mask) or len(sampled_ids) != len(evidence["logprobs"]):
        raise ValueError("sampled behavior arrays are not aligned")
    input_ids = list(prompt_ids) + list(sampled_ids)
    labels = [-100] * len(prompt_ids) + [
        token if supervised else -100 for token, supervised in zip(sampled_ids, mask)]
    response_mask = [0] * len(prompt_ids) + list(mask)
    packing = {
        "prompt_token_ids": list(prompt_ids), "sampled_response_token_ids": list(sampled_ids),
        "input_ids": input_ids, "labels": labels, "response_mask": response_mask,
        "causal_shift": "model_logits[t-1]_score_labels[t]",
        "prompt_label": -100, "unsupervised_response_label": -100,
    }
    packing["input_ids_sha256"] = _hash(input_ids)
    packing["labels_sha256"] = _hash(labels)
    packing["response_mask_sha256"] = _hash(response_mask)
    packing["packed_ids_labels_masks_sha256"] = _hash({
        "input_ids": input_ids, "labels": labels, "response_mask": response_mask})
    if not any(mask):
        raise ValueError("sampled response has no supervised tokens")
    return packing


def _objective_fixture(row, rollout, record, evidence, packing, advantage, config):
    objective = config["objective"]
    behavior = [float(value) for value in evidence["logprobs"]]
    current, reference = [], []
    for index, value in enumerate(behavior):
        current_value = value + float(
            objective["current_logprob_offsets"][index
                % len(objective["current_logprob_offsets"])])
        reference_value = value + float(
            objective["reference_logprob_offsets"][index
                % len(objective["reference_logprob_offsets"])])
        if not _finite(current_value) or not _finite(reference_value) \
                or current_value > 0.0 or reference_value > 0.0:
            raise ValueError("synthetic policy/reference fixture is not a finite log-probability")
        current.append(current_value)
        reference.append(reference_value)
    supervised = [index for index, value in enumerate(evidence["mask"]) if value == 1]
    behavior_mean = _mean([behavior[index] for index in supervised])
    policy_mean = _mean([current[index] for index in supervised])
    reference_mean = _mean([reference[index] for index in supervised])
    metrics = _objective_metrics(
        policy_mean, behavior_mean, reference_mean, float(advantage), objective)
    provenance = record["rollout_provenance"]
    packed_sha = packing["packed_ids_labels_masks_sha256"]
    current_input_sha = _hash({
        "policy": config["current_policy"], "packed": packed_sha,
        "logprob_source": "SYNTHETIC_CPU_FIXTURE_NOT_MODEL_OUTPUT",
    })
    reference_input_sha = _hash({
        "policy": config["reference_policy"], "packed": packed_sha,
        "logprob_source": "SYNTHETIC_CPU_FIXTURE_NOT_MODEL_OUTPUT",
    })
    return {
        "rollout_id": provenance["rollout_id"], "rank": rollout["rank"],
        "source_record_sha256": _hash(record),
        "behavior_evidence_sha256": provenance["behavior"]["evidence_sha256"],
        "packing": packing,
        "behavior_policy": {
            "identity": copy.deepcopy(provenance["behavior_policy"]),
            "tokenizer_sha256": evidence["tokenizer_sha256"],
            "sampled_logprobs": behavior, "source": "FINAL_RAW_SSE_SAMPLED_BEHAVIOR",
        },
        "current_policy": {
            "identity": copy.deepcopy(config["current_policy"]),
            "input_identity_sha256": current_input_sha,
            "logprobs": current, "forward": "NOT_RUN",
            "logprob_source": "SYNTHETIC_CPU_FIXTURE_NOT_MODEL_OUTPUT",
        },
        "reference_policy": {
            "identity": copy.deepcopy(config["reference_policy"]),
            "input_identity_sha256": reference_input_sha,
            "logprobs": reference, "forward": "NOT_RUN",
            "logprob_source": "SYNTHETIC_CPU_FIXTURE_NOT_MODEL_OUTPUT",
        },
        "supervised_response_indices": supervised, "metrics": metrics,
    }


def _checkpoint(run_id, steps, groups, packed_order_sha, policy_shas, cursor):
    if not _integer(cursor) or not 0 <= cursor <= len(steps):
        raise ValueError("invalid preflight checkpoint cursor")
    completed_groups = [group for step in steps[:cursor] for group in step["group_ids"]]
    remaining_groups = [group for step in steps[cursor:] for group in step["group_ids"]]
    value = {
        "schema": CHECKPOINT_SCHEMA, "mode": "online_preflight_only", "run_id": run_id,
        "next_step": cursor, "schedule_position": len(completed_groups),
        "completed_group_ids": completed_groups, "remaining_group_ids": remaining_groups,
        "completed_group_exports_sha256": _hash([
            group["group_export_sha256"] for group in groups
            if group["schedule_group_id"] in completed_groups]),
        "remaining_group_exports_sha256": _hash([
            group["group_export_sha256"] for group in groups
            if group["schedule_group_id"] in remaining_groups]),
        "packed_input_order_sha256": packed_order_sha,
        "behavior_policy_sha256": policy_shas["behavior"],
        "current_policy_sha256": policy_shas["current"],
        "reference_policy_sha256": policy_shas["reference"],
        "model_weights_updated": False, "optimizer_state": "NOT_RUN",
        "optimizer": "NOT_RUN", "current_policy_forward": "NOT_RUN",
        "reference_policy_forward": "NOT_RUN",
    }
    value["checkpoint_id"] = _hash(value)
    return value


def _build_documents(source_dir, config, tokenizer_dir=None):
    source = Path(source_dir)
    if not source.is_dir():
        raise ValueError("accepted online rollout bundle does not exist")
    config = _validate_config(config)
    with tempfile.TemporaryDirectory(prefix="umm-online-grpo-preflight-replay-") as tmp:
        replayed = Path(tmp) / "accepted"
        result = online.replay_bundle(source, replayed)
        if result.get("provenance_mode") != "online" \
                or result.get("behavior_logprobs") != "COVERED":
            raise ValueError("online preflight requires complete COVERED sampled behavior evidence")
        plan_bytes = (replayed / "plan.json").read_bytes()
        plan = online.prepare_plan(json.loads(plan_bytes.decode("utf-8")))
        schedule, schedule_sha = online.build_schedule(plan)
        records_list = _read_jsonl(replayed / "trajectories.jsonl")
        records = {}
        for item in schedule:
            # Re-run the independent per-record gate on replayed output as well as
            # replay_bundle's input gate; no endpoint or optimizer is invoked.
            candidates = [record for record in records_list
                          if (record.get("rollout_provenance") or {}).get("rollout_id")
                          == item["rollout_id"]]
            if len(candidates) != 1:
                raise ValueError("accepted replay has duplicate/missing scheduled rollout")
            record = candidates[0]
            online.validate_online_record(
                record, item, plan, schedule_sha, replayed / "artifacts")
            records[item["rollout_id"]] = record
        if len(records) != len(records_list):
            raise ValueError("accepted replay contains unscheduled records")
        grouped_rows = _extract_groups(replayed, plan, schedule, schedule_sha)

        has_exports = ["token_export" in record for record in records_list]
        if any(has_exports) and not all(has_exports):
            raise ValueError("online token_export supervision must be all-or-nothing")
        if all(has_exports) and tokenizer_dir is None:
            raise ValueError("token_export supervision requires --tokenizer-dir")
        if tokenizer_dir is not None and not Path(tokenizer_dir).is_dir():
            raise ValueError("tokenizer directory does not exist")

        behavior_tokenizers = set()
        group_documents = []
        objective_groups = []
        for schedule_group_id, split, row in grouped_rows:
            items = [item for item in schedule if item["group_id"] == schedule_group_id]
            matched, advantages, mean, std = _independent_group_audit(
                row, split, items, records)
            prompt_sha = row["prompt_sha256"]
            if prompt_sha not in config["prompt_token_ids"]:
                raise ValueError(f"missing exact prompt-token binding for {prompt_sha}")
            prompt_ids = config["prompt_token_ids"][prompt_sha]
            group_members = []
            fixture_members = []
            for rollout, (item, record, _candidate), advantage in zip(
                    row["rollouts"], matched, advantages):
                provenance = record["rollout_provenance"]
                behavior_summary = provenance.get("behavior") or {}
                if behavior_summary.get("status") != "COVERED" \
                        or behavior_summary.get("token_ids") is None \
                        or behavior_summary.get("logprobs") is None \
                        or behavior_summary.get("mask") is None:
                    raise ValueError("online member has missing/NOT_COVERED behavior values")
                raw = online.validate_replay_metadata(
                    record, artifact_dir=replayed / "artifacts", require_raw=True)
                chunks = online._raw_chunks(record, replayed / "artifacts")
                finish = chunks[-1].get("choices", [{}])[0].get("finish_reason")
                if record.get("terminal_state") != "completed" or finish != "stop":
                    raise ValueError("cancelled/truncated/non-stop online response is forbidden")
                evidence = chunks[-1].get("behavior")
                checked = online._behavior_evidence(
                    record, "exposed", replayed / "artifacts", chunks=chunks)
                if checked != behavior_summary or not isinstance(evidence, dict):
                    raise ValueError("raw sampled evidence differs from member provenance")
                behavior_tokenizers.add(evidence["tokenizer_sha256"])
                packing = _pack_member(evidence, prompt_ids)
                fixture = _objective_fixture(
                    row, rollout, record, evidence, packing, advantage, config)
                fixture["raw_sse_sha256"] = _hash_bytes(raw)
                alignment_report = None
                if all(has_exports):
                    alignment_report = alignment.validate_online_record_alignment(
                        record, evidence, tokenizer_dir)
                    if alignment_report["sampled_posthoc_target_ids_equal"] is not True:
                        raise ValueError("independent sampled/token alignment failed")
                group_members.append({
                    "rollout_id": item["rollout_id"], "rollout_index": item["rollout_index"],
                    "rank": rollout["rank"], "seed": item["seed"],
                    "request_sha256": item["request_sha256"],
                    "source_record_sha256": _hash(record),
                    "raw_sse_sha256": _hash_bytes(raw),
                    "behavior_evidence_sha256": behavior_summary["evidence_sha256"],
                    "reward": rollout["score"], "advantage": advantage,
                    "packed_ids_labels_masks_sha256":
                        packing["packed_ids_labels_masks_sha256"],
                    "current_policy_input_sha256":
                        fixture["current_policy"]["input_identity_sha256"],
                    "reference_policy_input_sha256":
                        fixture["reference_policy"]["input_identity_sha256"],
                    "token_export_alignment": alignment_report or {
                        "status": "NOT_COVERED", "reason": "NO_TOKEN_EXPORT_PRESENT",
                        "sampled_ids_replaced": False,
                    },
                })
                fixture["independent_token_alignment"] = alignment_report or {
                    "status": "NOT_COVERED", "reason": "NO_TOKEN_EXPORT_PRESENT",
                    "sampled_ids_replaced": False,
                }
                fixture_members.append(fixture)
            group_export_sha = _hash(row)
            group_documents.append({
                "schedule_group_id": schedule_group_id, "group_id": row["group_id"],
                "group_index": row["rollout_provenance"]["group_index"], "split": split,
                "group_export_sha256": group_export_sha,
                "prompt_sha256": prompt_sha,
                "prompt_token_ids_sha256": _hash(prompt_ids),
                "group_mean": mean, "group_std": std,
                "has_reward_variance": std > 0.0,
                "zero_variance_policy": "ZERO_ADVANTAGE",
                "members": group_members,
            })
            objective_groups.append({
                "schedule_group_id": schedule_group_id, "group_id": row["group_id"],
                "split": split, "group_export_sha256": group_export_sha,
                "group_mean": mean, "group_std": std,
                "has_reward_variance": std > 0.0,
                "advantage_semantics": "POPULATION_STD_ZERO_VARIANCE_TO_ZERO",
                "members": fixture_members,
                "objective_loss": math.fsum(
                    member["metrics"]["objective_loss"] for member in fixture_members)
                    / len(fixture_members),
                "normalization": "ROLLOUT_MEAN",
            })
        if len(behavior_tokenizers) != 1:
            raise ValueError("sampled tokenizer identity drifted within online run")
        behavior_tokenizer = next(iter(behavior_tokenizers))
        behavior = plan["behavior_policy"]
        for label, policy in (("current", config["current_policy"]),
                              ("reference", config["reference_policy"])):
            if policy["model_id"] != behavior["model_id"] \
                    or policy["model_config_sha256"] != behavior["model_config_sha256"]:
                raise ValueError(f"{label} policy model/config differs from behavior architecture")
            if policy["tokenizer_sha256"] != behavior_tokenizer:
                raise ValueError(f"{label} policy tokenizer differs from sampled evidence")

        source_files = _source_file_identity(replayed)
        source_fingerprints = _source_fingerprints()
        packed_order_sha = _hash([
            member["packed_ids_labels_masks_sha256"]
            for group in group_documents for member in group["members"]])
        policy_shas = {
            "behavior": online.behavior_policy_sha256(plan),
            "current": _policy_sha256(config["current_policy"]),
            "reference": _policy_sha256(config["reference_policy"]),
        }
        identity = {
            "mode": "online", "contract": online.PROVENANCE_CONTRACT,
            "batch_id": plan["batch_id"], "plan_sha256": online.plan_sha256(plan),
            "plan_bytes_sha256": _hash_bytes(plan_bytes),
            "rollout_schedule_sha256": schedule_sha,
            "rollout_manifest_sha256": _hash_bytes(
                (replayed / "rollout_manifest.json").read_bytes()),
            "pack_manifest_sha256": result["pack_manifest"]["sha256"],
            "source_files": source_files, "source_fingerprints": source_fingerprints,
            "config_sha256": _hash(config),
            "behavior_policy": copy.deepcopy(behavior),
            "behavior_policy_sha256": policy_shas["behavior"],
            "behavior_tokenizer_sha256": behavior_tokenizer,
            "sampling_sha256": online.sampling_sha256(plan),
            "current_policy": config["current_policy"],
            "current_policy_sha256": policy_shas["current"],
            "reference_policy": config["reference_policy"],
            "reference_policy_sha256": policy_shas["reference"],
            "groups": group_documents, "packed_input_order_sha256": packed_order_sha,
            "objective": {
                **copy.deepcopy(config["objective"]),
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
        run_id = _hash(identity)
        groups_per_step = config["groups_per_step"]
        steps = []
        for start in range(0, len(group_documents), groups_per_step):
            selected = group_documents[start:start + groups_per_step]
            steps.append({
                "step_index": len(steps),
                "group_ids": [group["group_id"] for group in selected],
                "schedule_group_ids": [group["schedule_group_id"] for group in selected],
                "group_export_sha256": [group["group_export_sha256"] for group in selected],
                "member_export_sha256": [
                    member["source_record_sha256"] for group in selected
                    for member in group["members"]],
                "packed_input_sha256": [
                    member["packed_ids_labels_masks_sha256"] for group in selected
                    for member in group["members"]],
            })
        checkpoints = [_checkpoint(run_id, steps, group_documents, packed_order_sha,
                                   policy_shas, cursor)
                       for cursor in range(len(steps) + 1)]
        for index, step in enumerate(steps):
            step["checkpoint_before"] = checkpoints[index]["checkpoint_id"]
            step["checkpoint_after"] = checkpoints[index + 1]["checkpoint_id"]

        claims = {
            "status": "PREFLIGHT_ONLY", "optimizer": "NOT_RUN", "REAL_TRAINING": "NOT_RUN",
            "current_policy_forward": "NOT_RUN", "reference_policy_forward": "NOT_RUN",
            "new_logprob_numerical_recomputation": "NOT_RUN", "inference": "NOT_RUN",
            "serving": "NOT_RUN", "gpu": "NOT_RUN", "model_quality": "NOT_RUN",
            "generalized_improvement": "NOT_RUN", "model_weights_updated": False,
            "torch_imported": False, "online_rollout_generation": "NOT_RUN_BY_PREFLIGHT",
            "source_rollout_evidence": result["claims"]["model_identity_verification"],
        }
        documents = {
            "config.json": config,
            "run_identity.json": {
                "schema": RUN_SCHEMA, "status": "PASS", "run_id": run_id,
                "identity_sha256": run_id, "identity": identity, "claims": claims,
            },
            "schedule.json": {
                "schema": SCHEDULE_SCHEMA, "run_id": run_id,
                "rollout_schedule_sha256": schedule_sha,
                "groups_per_step": groups_per_step, "steps": steps,
                "checkpoints": checkpoints,
                "cross_resume": "REQUIRE_EXACT_RUN_CHECKPOINT_GROUP_MEMBER_PACKED_AND_POLICY_IDS",
                "claims": claims,
            },
            "objective_fixtures.json": {
                "schema": OBJECTIVE_SCHEMA, "run_id": run_id,
                "evidence_class": "DETERMINISTIC_SYNTHETIC_CPU_OBJECTIVE_FIXTURES",
                "groups": objective_groups, "claims": claims,
            },
        }
        payloads = {name: _document_bytes(value) for name, value in documents.items()}
        manifest = {
            "schema": MANIFEST_SCHEMA, "status": "PASS", "run_id": run_id,
            "source_bundle": {"schema": online.EXPORT_SCHEMA, "files": source_files},
            "source_fingerprints": source_fingerprints,
            "files": {name: {"sha256": _hash_bytes(payload), "bytes": len(payload)}
                      for name, payload in payloads.items()},
            "group_count": len(group_documents), "rollout_count": len(records),
            "step_count": len(steps), "checkpoint_count": len(checkpoints),
            "claims": claims,
        }
        payloads["manifest.json"] = _document_bytes(manifest)
        payloads["manifest.json.sha256"] = (
            _hash_bytes(payloads["manifest.json"]) + "\n").encode("ascii")
        checksums = "".join(
            f"{_hash_bytes(payloads[name])}  {name}\n" for name in sorted(payloads))
        payloads["SHA256SUMS"] = checksums.encode("ascii")
        return payloads


def _write_bundle(output_dir, payloads):
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory must be absent or empty")
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=parent))
    try:
        for name, payload in payloads.items():
            path = temporary / name
            path.write_bytes(payload)
        if output.exists():
            output.rmdir()
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _read_verified_bundle(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError("preflight bundle does not exist")
    paths = sorted((path for path in root.rglob("*") if path.is_file()),
                   key=lambda path: path.relative_to(root).as_posix())
    if any(path.is_symlink() for path in paths) or not paths:
        raise ValueError("preflight bundle contains symbolic files or is empty")
    actual = {path.relative_to(root).as_posix(): path.read_bytes() for path in paths}
    checksum_text = actual.get("SHA256SUMS", b"").decode("ascii")
    expected = {}
    for line in checksum_text.splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise ValueError("malformed preflight SHA256SUMS")
        online._sha256(parts[0], "preflight checksum")
        name = parts[1]
        if not name or Path(name).name != name or name in expected:
            raise ValueError("unsafe or duplicate preflight checksum path")
        expected[name] = parts[0]
    if set(expected) != set(actual) - {"SHA256SUMS"}:
        raise ValueError("preflight bundle file set differs from SHA256SUMS")
    for name, digest in expected.items():
        if _hash_bytes(actual[name]) != digest:
            raise ValueError(f"preflight bundle tamper: {name}")
    manifest = json.loads(actual["manifest.json"].decode("utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA \
            or actual["manifest.json.sha256"] != (
                _hash_bytes(actual["manifest.json"]) + "\n").encode("ascii"):
        raise ValueError("preflight manifest or sidecar is invalid")
    return actual


def prepare_preflight(source_dir, output_dir, config, *, tokenizer_dir=None):
    """Validate and write one deterministic bundle; no final output on rejection."""
    payloads = _build_documents(source_dir, config, tokenizer_dir)
    _write_bundle(output_dir, payloads)
    manifest = json.loads(payloads["manifest.json"])
    return {
        "schema": RESULT_SCHEMA, "status": "PASS", "run_id": manifest["run_id"],
        "group_count": manifest["group_count"], "rollout_count": manifest["rollout_count"],
        "step_count": manifest["step_count"], "torch_imported": False,
        "claims": manifest["claims"],
    }


def verify_preflight(bundle_dir, source_dir, output_dir, *, tokenizer_dir=None):
    """Independently reconstruct and byte-compare a bundle in this fresh process."""
    actual = _read_verified_bundle(bundle_dir)
    config = _validate_config(json.loads(actual["config.json"].decode("utf-8")))
    expected = _build_documents(source_dir, config, tokenizer_dir)
    if set(actual) != set(expected):
        raise ValueError("fresh-process reconstruction file set differs")
    for name in sorted(expected):
        if actual[name] != expected[name]:
            raise ValueError(f"fresh-process reconstruction differs: {name}")
    _write_bundle(output_dir, expected)
    manifest = json.loads(expected["manifest.json"])
    return {
        "schema": VERIFY_SCHEMA, "status": "PASS", "run_id": manifest["run_id"],
        "fresh_process_reconstruction": "BYTE_IDENTICAL", "tamper_rejection": "HASH_THEN_REBUILD",
        "file_count": len(expected), "torch_imported": False, "claims": manifest["claims"],
    }


def _parser():
    parser = argparse.ArgumentParser(
        description="CPU-only validated online-GRPO preflight; no inference or optimizer")
    parser.add_argument("--online-pack", type=Path, required=True,
                        help="accepted umminfer_online_rollout bundle")
    parser.add_argument("--config", type=Path,
                        help="online current/reference policy and objective config")
    parser.add_argument("--tokenizer-dir", type=Path,
                        help="exact local tokenizer files when token_export is present")
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--verify-bundle", type=Path,
                      help="existing preflight bundle to reconstruct in a fresh process")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        _reject_ml_imports()
        if args.preflight_only:
            if args.config is None:
                raise ValueError("--preflight-only requires --config")
            result = prepare_preflight(
                args.online_pack, args.out, load_config(args.config),
                tokenizer_dir=args.tokenizer_dir)
        else:
            if args.config is not None:
                raise ValueError("--verify-bundle reads its bound config; omit --config")
            result = verify_preflight(
                args.verify_bundle, args.online_pack, args.out,
                tokenizer_dir=args.tokenizer_dir)
        _reject_ml_imports()
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as error:
        print(f"online GRPO preflight rejected: {error}", file=sys.stderr)
        return 2
    print(_canonical(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
