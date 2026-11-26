# SPDX-License-Identifier: Apache-2.0
"""Bounded real GRPO optimizer adapter for the existing Stage C trainer.

All model-independent validation runs before Torch or Transformers is imported.
The adapter binds every scheduled rollout to its exact schema-v2 export, rereads
CAS bytes, recomputes export-only rewards and group statistics, and reuses the
existing planner, scheduler, response auditor, HF masks, and tensor fingerprints.
The executed objective matches the Stage C sequence-level clipped-GRPO reference:
one frozen initial base supplies old-policy and reference-policy log-probabilities.

This is an architecture/optimizer gate over deterministic pre-generated offline
rollouts, not online generation, model quality, or production training. It never
starts or contacts a serving process.
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

import umminfer_ablation_metrics as routing_metrics  # noqa: E402
import umminfer_joint_dpo_live as dpo_live  # noqa: E402
import umminfer_joint_sft_live as sft_live  # noqa: E402
import umminfer_joint_trainer as stage_c  # noqa: E402
import umminfer_sft_dataset as dataset_builder  # noqa: E402

CHECKPOINT_FORMAT = "umminfer-joint-grpo-live-checkpoint-v1"
LIVE_MANIFEST_SCHEMA = "umminfer-joint-grpo-live-result-v1"
SOURCE_FILES = (
    "python/umminfer_joint_grpo_live.py",
    "python/umminfer_joint_dpo_live.py",
    "python/umminfer_joint_sft_live.py",
    "python/umminfer_joint_trainer.py",
    "python/umminfer_sft_dataset.py",
    "python/umminfer_rl_trajectory.py",
    "python/umminfer_rl_reward.py",
    "python/umminfer_ablation_metrics.py",
)
_HEX_40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(float(value)))


def _source_fingerprints(source_commit):
    if not isinstance(source_commit, str) or not _HEX_40.fullmatch(source_commit):
        raise ValueError("--source-commit must be one lowercase 40-character commit")
    files = {}
    for rel in SOURCE_FILES:
        path = ROOT / rel
        if not path.is_file():
            raise ValueError(f"missing live-gate source file: {rel}")
        files[rel] = {"sha256": stage_c.sha256_file(path), "bytes": path.stat().st_size}
    return {
        "commit": source_commit,
        "files": files,
        "source_tree_scope": "listed Stage C source, export, and reward files only",
    }


def _provenance_key(provenance):
    if not isinstance(provenance, dict):
        raise ValueError("rollout provenance must be an object")
    session = provenance.get("source_session_id")
    request = provenance.get("source_request_id")
    turn = provenance.get("source_turn_index")
    if not isinstance(session, str) or not session \
            or not isinstance(request, str) or not request \
            or not _is_int(turn) or turn < 0:
        raise ValueError("rollout provenance lacks a valid source identity")
    return session, request, turn


def _load_export_evidence(trajectory_paths, manifest, reward_config):
    """Bind exact source bytes to the pack and independently recompute rewards."""
    if not isinstance(trajectory_paths, (list, tuple)) or not trajectory_paths:
        raise ValueError("live GRPO requires at least one --trajectory-path")
    if not isinstance(reward_config, dict) or reward_config.get("export_only") is not True:
        raise ValueError("bounded gate requires an explicit export-only reward")
    reward_spec = reward_config.get("spec")
    if reward_spec != "umminfer_ablation_metrics:NarrativeRoutingReward":
        raise ValueError("bounded gate supports only NarrativeRoutingReward")
    if manifest.get("provenance_mode") == "online" \
            or "rollout_provenance" in manifest:
        raise ValueError("frozen-offline GRPO gate refuses online export provenance")
    if manifest.get("mode") != "preference" \
            or manifest.get("reward") != {
                "spec": reward_spec, "name": routing_metrics.REWARD_NAME}:
        raise ValueError("dataset manifest reward/mode differs from the live GRPO config")
    if manifest.get("artifact_verification", {}).get("mode") != "verified_on_disk":
        raise ValueError("live GRPO requires on-disk-verified source artifacts")
    if manifest.get("min_group_size") is None:
        raise ValueError("dataset manifest lacks its GRPO minimum group size")

    reward = dataset_builder.load_reward(reward_spec)
    exports = {}
    input_entries = []
    for supplied in trajectory_paths:
        path = Path(supplied)
        if not path.is_file():
            raise ValueError(f"source trajectory does not exist: {path}")
        raw_bytes = path.read_bytes()
        lines = [line for line in raw_bytes.decode("utf-8").splitlines() if line.strip()]
        if not lines:
            raise ValueError("source trajectory must contain at least one record")
        input_entries.append({
            "name": path.name,
            "sha256": stage_c.sha256_bytes(raw_bytes),
            "bytes": len(raw_bytes),
            "line_count": len(lines),
        })
        artifact_dir = path.parent / "artifacts"
        if not artifact_dir.is_dir():
            raise ValueError("source trajectory is missing its CAS artifact directory")
        for line in lines:
            record = json.loads(line)
            if "rollout_provenance" in record:
                raise ValueError(
                    "frozen-offline GRPO gate refuses online source trajectories, "
                    "including rebuilt packs without online row annotations")
            ok, category, message, stats = dataset_builder.verify_record_integrity(
                record, artifact_dir=artifact_dir, require_artifact_dir=True)
            if not ok:
                raise ValueError(f"source export failed {category}: {message}")
            if record.get("schema_version") != 2 \
                    or record.get("terminal_state") != "completed" \
                    or not stats["checked_disk"]:
                raise ValueError("live GRPO accepts only completed disk-verified schema-v2 exports")
            candidate = dataset_builder._extract_candidate(record, verified_on_disk=True)
            key = _provenance_key(candidate["provenance"])
            if key in exports:
                raise ValueError("duplicate source rollout identity across exports")
            score = reward.score(copy.deepcopy(record))
            if not _finite(score):
                raise ValueError("export-derived reward must be finite and non-bool")
            components = routing_metrics.routing_observation(record)
            if components["coverage"] != "COVERED":
                raise ValueError("export-derived reward lacks routing labels")
            expected_score = components["f1"] - components["redundant_slot_rate"]
            if not math.isclose(float(score), expected_score, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("reward differs from its export-derived F1/redundancy components")
            exports[key] = {
                "candidate": candidate,
                "score": float(score),
                "reward_components": components,
                "integrity": {
                    "source_record_sha256": stage_c.sha256_bytes(
                        stage_c.canonical_json(record)),
                    "output_sha256": record["output_sha256"],
                    "clean_text_sha256": record["clean_text_sha256"],
                    "terminal_state": record["terminal_state"],
                    "terminal_event_seq": record["terminal_event_seq"],
                    "terminal_legs": record["terminal_legs"],
                    "cas_verified_on_disk": True,
                    "checked_input_media_refs": stats["checked_input_media_refs"],
                    "checked_output_media_refs": stats["checked_output_media_refs"],
                },
            }

    input_entries.sort(key=lambda item: (item["sha256"], item["name"]))
    expected_inputs = sorted(
        manifest.get("inputs") or [], key=lambda item: (item["sha256"], item["name"]))
    if input_entries != expected_inputs:
        raise ValueError("supplied source exports differ from dataset-manifest input hashes")
    counts = manifest.get("counts") or {}
    if counts.get("total_records_read") != len(exports) \
            or counts.get("accepted_verified_completed") != len(exports) \
            or any(value != 0 for value in (counts.get("rejected") or {}).values()):
        raise ValueError("bounded gate requires a complete export set with zero rejected records")
    return {
        "by_key": exports,
        "inputs": input_entries,
        "reward_spec": reward_spec,
        "reward_name": reward.name,
        "source_record_order_sha256": stage_c.sha256_bytes(stage_c.canonical_json([
            list(key) for key in sorted(exports)
        ])),
    }


def _rollout_id(prompt_sha256, key, target_sha256):
    session, request, turn = key
    return stage_c.sha256_text(
        f"{prompt_sha256}:{session}:{request}:{turn}:{target_sha256}")[:32]


def _audit_group(row, export_evidence, *, expected_group_size):
    """Validate one complete ordered group against its verified source exports."""
    if not isinstance(row, dict) or row.get("record_type") != "grpo_group":
        raise ValueError("live gate accepts only grpo_group records")
    if row.get("schema_version") != 1:
        raise ValueError("unsupported Stage C GRPO record schema")
    if "rollout_provenance" in row or any(
            isinstance(rollout, dict) and "rollout_provenance" in rollout
            for rollout in row.get("rollouts") or []):
        raise ValueError(
            "frozen-offline GRPO gate refuses online-provenance groups; "
            "use a separately validated online optimizer gate")
    group_id = row.get("group_id")
    prompt_sha = row.get("prompt_sha256")
    if not isinstance(group_id, str) or not group_id \
            or not isinstance(prompt_sha, str) or not _HEX_64.fullmatch(prompt_sha):
        raise ValueError("GRPO group/prompt identity is invalid")
    if row.get("group_size") != expected_group_size:
        raise ValueError("GRPO group_size differs from the bounded live configuration")
    rollouts = row.get("rollouts")
    if not isinstance(rollouts, list) or len(rollouts) != expected_group_size:
        raise ValueError("GRPO group is missing rollouts")
    prompt_parts = row.get("prompt_parts")
    if not isinstance(prompt_parts, list) or not prompt_parts \
            or stage_c.sha256_bytes(stage_c.canonical_json(prompt_parts)) != prompt_sha:
        raise ValueError("GRPO prompt_parts do not recompute prompt_sha256")
    if row.get("reward_name") != export_evidence["reward_name"]:
        raise ValueError("GRPO group reward name differs from the export reward")

    matched = []
    source_keys = set()
    target_hashes = set()
    audits = []
    for rank, rollout in enumerate(rollouts):
        if not isinstance(rollout, dict) or rollout.get("rank") != rank:
            raise ValueError("GRPO rollout ranks must be contiguous and ordered")
        response_audit = dpo_live._audit_response(rollout, side=f"rollout-{rank}")
        key = _provenance_key(rollout.get("provenance"))
        if key in source_keys:
            raise ValueError("GRPO group contains a duplicate rollout identity")
        source_keys.add(key)
        evidence = export_evidence["by_key"].get(key)
        if evidence is None:
            raise ValueError("GRPO rollout is absent from the verified source exports")
        candidate = evidence["candidate"]
        for field in ("target_sha256", "assistant_target", "clean_text", "visual_slots",
                      "modality_route", "output_media_refs", "provenance"):
            if rollout.get(field) != candidate[field]:
                raise ValueError(f"GRPO rollout {field} differs from its source export")
        if candidate["prompt_sha256"] != prompt_sha:
            raise ValueError("GRPO group crosses different source prompts")
        if rollout["target_sha256"] in target_hashes:
            raise ValueError("bounded GRPO gate requires distinct rollout targets")
        target_hashes.add(rollout["target_sha256"])
        if not _finite(rollout.get("advantage")) \
                or not math.isclose(float(rollout["score"]), evidence["score"],
                                    rel_tol=0.0, abs_tol=0.0):
            raise ValueError("GRPO rollout score is non-finite or differs from the export reward")
        rollout_id = _rollout_id(prompt_sha, key, rollout["target_sha256"])
        audits.append({
            **response_audit,
            "rollout_id": rollout_id,
            "rank": rank,
            "source_key": list(key),
            "reward": {
                "name": export_evidence["reward_name"],
                "spec": export_evidence["reward_spec"],
                "score": evidence["score"],
                "components": evidence["reward_components"],
                "derivation": "schema_v2_export_only",
            },
            "export_integrity": evidence["integrity"],
        })
        matched.append((candidate, evidence["score"], key))

    expected_order = sorted(
        matched, key=lambda item: (-item[1], dataset_builder._candidate_tie_key(item[0])))
    if [item[2] for item in expected_order] != [item[2] for item in matched]:
        raise ValueError("GRPO rollout order differs from deterministic score/provenance order")
    expected_group_id = stage_c.sha256_text(
        prompt_sha + ":" + export_evidence["reward_name"] + ":"
        + ",".join(item[0]["target_sha256"] for item in expected_order))[:32]
    if group_id != expected_group_id:
        raise ValueError("GRPO group_id does not bind ordered group membership")

    lead = matched[0][0]
    for field in ("prompt_text", "prompt_parts", "input_media_refs"):
        if row.get(field) != lead[field]:
            raise ValueError(f"GRPO group {field} differs from its source prompt")
    scores = [item[1] for item in matched]
    advantages, mean, std = dataset_builder.compute_grpo_advantages(scores)
    if not _finite(row.get("group_mean")) or not _finite(row.get("group_std")) \
            or not math.isclose(float(row["group_mean"]), mean, rel_tol=0.0, abs_tol=1e-12) \
            or not math.isclose(float(row["group_std"]), std, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("GRPO group baseline/std differs from export-derived rewards")
    if row.get("has_reward_variance") is not (std > 0.0) or std <= 0.0:
        raise ValueError("bounded GRPO gate requires a non-zero-variance group")
    for rollout, expected in zip(rollouts, advantages):
        if not math.isclose(float(rollout["advantage"]), expected,
                            rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("GRPO rollout advantage differs from normalized export rewards")
    rollout_ids = [audit["rollout_id"] for audit in audits]
    return {
        "group_id": group_id,
        "prompt_sha256": prompt_sha,
        "group_size": expected_group_size,
        "group_mean": mean,
        "group_std": std,
        "has_reward_variance": True,
        "rollout_order": rollout_ids,
        "membership_order_sha256": stage_c.sha256_bytes(stage_c.canonical_json(rollout_ids)),
        "source_keys": [list(item[2]) for item in matched],
        "rollouts": audits,
    }


def prepare_live_run(*, dataset_dir, trajectory_paths, output_dir, config_path,
                     model_path, source_commit, max_steps, checkpoint_every=1,
                     stop_after=None, resume_from=None, model_revision=None):
    """Complete all source, group, reward, CAS, config, and model checks before ML."""
    if not _is_int(max_steps) or max_steps <= 0:
        raise ValueError("--max-steps must be a positive integer")
    if stop_after is None:
        stop_after = max_steps
    if not _is_int(stop_after) or not 0 < stop_after <= max_steps:
        raise ValueError("--stop-after must be in [1, --max-steps]")
    if not _is_int(checkpoint_every) or checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be a positive integer")
    if resume_from is not None:
        resume_path = Path(resume_from)
        sidecar = Path(str(resume_path) + ".sha256")
        if not resume_path.is_file() or not sidecar.is_file() \
                or sidecar.read_text(encoding="ascii").strip() != stage_c.sha256_file(resume_path):
            raise ValueError("resume checkpoint or SHA-256 sidecar is missing or invalid")

    config_file = Path(config_path)
    if not config_file.is_file():
        raise ValueError("live training config does not exist")
    config_sha256 = stage_c.sha256_file(config_file)
    output = Path(output_dir)
    plan = stage_c.prepare_training_run(
        mode="grpo", ablation="joint_full", dataset_dir=dataset_dir,
        output_dir=output / "plan", config_path=config_file, gpu_launch_script=True)
    if plan["REAL_TRAINING"] != "NOT_RUN" or plan["active_stages"] != ["grpo"]:
        raise ValueError("Stage C planner did not produce a GRPO-only NOT_RUN plan")
    hyper = plan["hyperparameters"]
    model_cfg, peft_cfg, opt, grpo = (
        hyper["model"], hyper["peft"], hyper["optimization"], hyper["grpo"])
    expected_model = {
        "dtype": "bfloat16", "attn_implementation": "eager",
        "gradient_checkpointing": False, "trust_remote_code": False,
    }
    if any(model_cfg.get(key) != value for key, value in expected_model.items()):
        raise ValueError("live gate requires bf16 eager local Qwen2 without remote code")
    if peft_cfg.get("method") != "none":
        raise ValueError("bounded gate supports full-parameter GRPO only")
    if opt.get("lr_scheduler") != "constant" or opt.get("warmup_ratio") != 0.0:
        raise ValueError("bounded gate requires constant LR with zero warmup")
    for key in ("per_device_train_batch_size", "gradient_accumulation_steps", "num_train_epochs"):
        if opt.get(key) != 1:
            raise ValueError(f"bounded gate requires {key}=1")
    if not _finite(opt.get("learning_rate")) or float(opt["learning_rate"]) <= 0 \
            or not _finite(opt.get("weight_decay")) or float(opt["weight_decay"]) != 0.0 \
            or not _finite(opt.get("max_grad_norm")) or float(opt["max_grad_norm"]) <= 0 \
            or not _is_int(opt.get("max_seq_len")) or opt["max_seq_len"] <= 1:
        raise ValueError("invalid bounded optimizer configuration")
    if not _finite(grpo.get("clip_eps")) or not 0.0 < float(grpo["clip_eps"]) < 1.0 \
            or not _finite(grpo.get("kl_coef")) or float(grpo["kl_coef"]) < 0.0 \
            or not _finite(grpo.get("advantage_eps")) or float(grpo["advantage_eps"]) != 0.0 \
            or grpo.get("zero_std_policy") != "zero_advantage" \
            or not _is_int(grpo.get("group_size")) or grpo["group_size"] < 2 \
            or grpo.get("old_policy") != "frozen_initial_base_model" \
            or grpo.get("reference_policy") != "frozen_initial_base_model" \
            or grpo.get("ratio_computation") != "span_weighted_target_mean_sequence_logp" \
            or grpo.get("surrogate_normalization") != "rollout_mean" \
            or not _finite(grpo.get("log_ratio_clamp")) or float(grpo["log_ratio_clamp"]) <= 0:
        raise ValueError("invalid bounded GRPO objective configuration")
    if grpo.get("max_length") != opt["max_seq_len"] \
            or not _is_int(grpo.get("max_prompt_length")) \
            or not 0 < grpo["max_prompt_length"] < grpo["max_length"]:
        raise ValueError("GRPO prompt/sequence length limits are inconsistent")

    dataset = stage_c.verify_and_load_dataset_pack(dataset_dir)
    manifest = dataset["manifest"]
    if manifest.get("min_group_size") != grpo["group_size"]:
        raise ValueError("dataset builder minimum group size differs from live group_size")
    export_evidence = _load_export_evidence(
        trajectory_paths, manifest, hyper.get("reward"))
    all_rows = (dataset["packs"].get("grpo_train") or []) \
        + (dataset["packs"].get("grpo_eval") or [])
    all_audits = [_audit_group(row, export_evidence,
                               expected_group_size=grpo["group_size"]) for row in all_rows]
    consumed = [tuple(key) for audit in all_audits for key in audit["source_keys"]]
    if len(consumed) != len(set(consumed)) or set(consumed) != set(export_evidence["by_key"]):
        raise ValueError("GRPO packs have duplicate, missing, or ungrouped source rollouts")

    rows = dataset["packs"].get("grpo_train") or []
    by_id = {row["group_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("GRPO training group ids are non-unique")
    schedule = plan["batch_schedules"]["grpo"]
    if max_steps > schedule["total_batches"]:
        raise ValueError("--max-steps exceeds the deterministic Stage C schedule")
    scheduled_ids = [record_id for batch in schedule["batches"][:max_steps]
                     for record_id in batch["record_ids"]]
    if len(scheduled_ids) != len(set(scheduled_ids)) \
            or any(group_id not in by_id for group_id in scheduled_ids):
        raise ValueError("scheduled group ids are absent or non-unique")
    scheduled_rows = [by_id[group_id] for group_id in scheduled_ids]
    audit_by_id = {audit["group_id"]: audit for audit in all_audits}
    scheduled_audits = [audit_by_id[group_id] for group_id in scheduled_ids]
    for audit in scheduled_audits:
        for rollout in audit["rollouts"]:
            if not rollout["visual_modalities"] or not rollout["terminal_legs_verified"] \
                    or not rollout["verified_cas_refs"]:
                raise ValueError("every scheduled rollout needs structured visual/CAS/terminal targets")

    model = sft_live._fingerprint_model(model_path)
    if model_revision is not None:
        if not isinstance(model_revision, str) or not model_revision:
            raise ValueError("model revision override must be non-empty")
        model["revision"] = model_revision
        model["upstream_revision_verification"] = "NOT_RUN"
    source = _source_fingerprints(source_commit)
    identity = {
        "source": source,
        "config_file_sha256": config_sha256,
        "resolved_config_sha256": plan["config_sha256"],
        "dataset_manifest_sha256": plan["dataset"]["manifest_sha256"],
        "source_export_inputs": export_evidence["inputs"],
        "source_record_order_sha256": export_evidence["source_record_order_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "model": model,
        "optimization": opt,
        "grpo": grpo,
        "reward": hyper["reward"],
        "span_loss_weights": plan["ablation"]["span_weights"],
        "max_steps": max_steps,
        "objective": "full-parameter-sequence-clipped-grpo-with-schulman-kl-v1",
    }
    return {
        "output_dir": output,
        "dataset_dir": Path(dataset_dir),
        "trajectory_paths": [Path(path) for path in trajectory_paths],
        "config_path": config_file,
        "model_path": Path(model_path),
        "plan": plan,
        "rows": scheduled_rows,
        "group_audits": scheduled_audits,
        "all_group_audits": all_audits,
        "export_inputs": export_evidence["inputs"],
        "source": source,
        "model": model,
        "config_file_sha256": config_sha256,
        "identity": identity,
        "identity_sha256": stage_c.sha256_bytes(stage_c.canonical_json(identity)),
        "max_steps": max_steps,
        "stop_after": stop_after,
        "checkpoint_every": checkpoint_every,
        "resume_from": Path(resume_from) if resume_from is not None else None,
    }


def _pack_groups(tokenizer, prepared):
    # The accepted DPO response packer is algorithm-neutral: prompt masking,
    # route/body/terminal annotations, and tensor hashes are shared verbatim.
    packing_view = {"identity": {
        "dpo": {
            "max_length": prepared["identity"]["grpo"]["max_length"],
            "max_prompt_length": prepared["identity"]["grpo"]["max_prompt_length"],
        },
        "span_loss_weights": prepared["identity"]["span_loss_weights"],
    }}
    packed = []
    for row, audit in zip(prepared["rows"], prepared["group_audits"]):
        rollouts = []
        for rank, (response, response_audit) in enumerate(zip(row["rollouts"], audit["rollouts"])):
            item = dpo_live._pack_response(
                tokenizer, row, response, response_audit,
                side=f"rollout-{rank}", prepared=packing_view)
            if rollouts and item["prompt_token_ids"] != rollouts[0]["prompt_token_ids"]:
                raise ValueError("group rollout prompt tokenization differs")
            item["rollout_id"] = response_audit["rollout_id"]
            item["advantage"] = float(response["advantage"])
            rollouts.append(item)
        packed.append({"group_id": row["group_id"], "rollouts": rollouts})
    return packed


def _objective_metrics(policy_mean, behavior_mean, advantage, *, clip_eps, kl_coef, clamp):
    """Pure evidence form of the Stage C clipped surrogate plus Schulman KL."""
    raw_log_ratio = policy_mean - behavior_mean
    log_ratio = max(-clamp, min(clamp, raw_log_ratio))
    ratio = math.exp(log_ratio)
    clipped_ratio = max(1.0 - clip_eps, min(1.0 + clip_eps, ratio))
    surrogate_1 = ratio * advantage
    surrogate_2 = clipped_ratio * advantage
    surrogate = min(surrogate_1, surrogate_2)
    raw_kl_diff = behavior_mean - policy_mean
    kl_diff = max(-clamp, min(clamp, raw_kl_diff))
    kl = math.exp(kl_diff) - kl_diff - 1.0
    objective = -surrogate + kl_coef * kl
    values = (raw_log_ratio, log_ratio, ratio, clipped_ratio, surrogate_1,
              surrogate_2, surrogate, raw_kl_diff, kl_diff, kl, objective)
    if not all(math.isfinite(value) for value in values) or ratio <= 0.0:
        raise RuntimeError("non-finite GRPO ratio/KL/objective evidence")
    return {
        "policy_mean_logp": policy_mean,
        "old_mean_logp": behavior_mean,
        "reference_mean_logp": behavior_mean,
        "raw_log_ratio": raw_log_ratio,
        "log_ratio": log_ratio,
        "ratio": ratio,
        "clipped_ratio": clipped_ratio,
        "advantage": advantage,
        "unclipped_surrogate": surrogate_1,
        "clipped_surrogate": surrogate_2,
        "selected_surrogate": surrogate,
        "clip_active": surrogate_1 > surrogate_2,
        "raw_reference_minus_policy_logp": raw_kl_diff,
        "reference_minus_policy_logp": kl_diff,
        "kl": kl,
        "objective_loss": objective,
    }


def _tensor_objective(torch, policy_mean, behavior_mean, advantage, *, clip_eps, kl_coef, clamp):
    log_ratio = (policy_mean - behavior_mean).clamp(min=-clamp, max=clamp)
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(min=1.0 - clip_eps, max=1.0 + clip_eps)
    surrogate = torch.minimum(ratio * advantage, clipped_ratio * advantage)
    kl_diff = (behavior_mean - policy_mean).clamp(min=-clamp, max=clamp)
    kl = torch.exp(kl_diff) - kl_diff - 1.0
    return -surrogate + kl_coef * kl


def execute_live_run(prepared):
    """Execute measured full-parameter clipped-GRPO AdamW steps on one SM90 GPU."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit() or "," in visible:
        raise ValueError("live gate requires CUDA_VISIBLE_DEVICES=<one numeric idle GPU>")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    print("GRPO_ML_IMPORT_BEGIN", file=sys.stderr, flush=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("live gate requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if tuple(torch.cuda.get_device_capability(device)) != (9, 0):
        raise RuntimeError("bounded live gate requires an SM90 CUDA device")
    torch.use_deterministic_algorithms(True)
    seed_text = str(prepared["identity"]["optimization"]["seed"])
    seed = int(stage_c.sha256_text(seed_text)[:16], 16) % (2 ** 31)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    resume_payload = None
    next_step = 0
    resume_sha256 = None
    if prepared["resume_from"] is not None:
        resume_sha256 = stage_c.sha256_file(prepared["resume_from"])
        resume_payload = torch.load(
            prepared["resume_from"], map_location="cpu", weights_only=True)
        if not isinstance(resume_payload, dict) \
                or resume_payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("unsupported resume checkpoint format")
        if resume_payload.get("run_identity_sha256") != prepared["identity_sha256"]:
            raise ValueError("resume checkpoint source/config/data/model/order identity mismatch")
        next_step = resume_payload.get("next_step")
        if not _is_int(next_step) or not 0 < next_step < prepared["stop_after"] \
                or resume_payload.get("schedule_position") != next_step:
            raise ValueError("resume schedule position leaves no valid bounded step")

    tokenizer = AutoTokenizer.from_pretrained(
        prepared["model_path"], local_files_only=True, use_fast=True, trust_remote_code=False)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("live gate requires a fast tokenizer before model loading")
    packed = _pack_groups(tokenizer, prepared)
    policy = AutoModelForCausalLM.from_pretrained(
        prepared["model_path"], local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="eager")
    behavior = AutoModelForCausalLM.from_pretrained(
        prepared["model_path"], local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="eager")
    policy.config.use_cache = False
    behavior.config.use_cache = False
    policy.to(device)
    behavior.to(device)
    policy.train()
    behavior.eval()
    for param in behavior.parameters():
        param.requires_grad_(False)
    behavior_sha = sft_live._model_fingerprint(torch, behavior)
    if resume_payload is not None \
            and resume_payload.get("behavior_parameter_sha256") != behavior_sha:
        raise ValueError("resume checkpoint behavior-policy fingerprint mismatch")

    behavior_values = [
        [dpo_live._measured_logprob(torch, behavior, item, device)
         for item in group["rollouts"]]
        for group in packed
    ]
    if resume_payload is not None \
            and resume_payload.get("behavior_logprob_cache") != behavior_values:
        raise ValueError("resume checkpoint old/reference log-probability cache mismatch")
    if any(param.grad is not None for param in behavior.parameters()):
        raise RuntimeError("frozen GRPO behavior/reference unexpectedly received gradients")

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=float(prepared["identity"]["optimization"]["learning_rate"]),
        weight_decay=0.0, foreach=False)
    initial_parameter_sha = sft_live._model_fingerprint(torch, policy)
    if next_step == 0 and initial_parameter_sha != behavior_sha:
        raise RuntimeError("initial GRPO policy and behavior/reference weights differ")
    if resume_payload is not None:
        policy.load_state_dict(resume_payload["model_state"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        torch.set_rng_state(resume_payload["cpu_rng_state"])
        torch.cuda.set_rng_state_all(resume_payload["cuda_rng_state"])
        del resume_payload
        initial_parameter_sha = sft_live._model_fingerprint(torch, policy)
    if not all(torch.isfinite(param).all().item() for param in policy.parameters()):
        raise RuntimeError("policy parameters are non-finite before optimizer execution")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    history, checkpoints = [], []
    output = prepared["output_dir"]
    grpo = prepared["identity"]["grpo"]
    clip_eps, kl_coef, clamp = (
        float(grpo["clip_eps"]), float(grpo["kl_coef"]), float(grpo["log_ratio_clamp"]))

    def save_checkpoint(completed_steps):
        checkpoint_dir = output / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"step-{completed_steps:06d}.pt"
        if path.exists():
            raise ValueError(f"refusing to overwrite checkpoint: {path.name}")
        payload = {
            "format": CHECKPOINT_FORMAT,
            "run_identity_sha256": prepared["identity_sha256"],
            "next_step": completed_steps,
            "schedule_position": completed_steps,
            "model_state": policy.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "behavior_parameter_sha256": behavior_sha,
            "behavior_logprob_cache": behavior_values,
        }
        tmp = path.with_name(path.name + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        digest = stage_c.sha256_file(path)
        sft_live._atomic_bytes(Path(str(path) + ".sha256"), (digest + "\n").encode("ascii"))
        record = {"next_step": completed_steps, "path": path.name,
                  "sha256": digest, "bytes": path.stat().st_size}
        checkpoints.append(record)
        return record

    for step_index in range(next_step, prepared["stop_after"]):
        group = packed[step_index]
        torch.cuda.reset_peak_memory_stats(device)
        step_started = time.perf_counter()
        before = {name: param.detach().cpu().clone()
                  for name, param in policy.named_parameters()}
        before_sha = sft_live._model_fingerprint(torch, policy)
        optimizer.zero_grad(set_to_none=True)
        rollout_losses = []
        pre_rollouts = []
        for index, item in enumerate(group["rollouts"]):
            policy_mean, weighted_sum, normalizer = dpo_live._sequence_logprob(
                torch, policy, item, device)
            old_mean = behavior_values[step_index][index]["mean_logp"]
            rollout_losses.append(_tensor_objective(
                torch, policy_mean, old_mean, item["advantage"],
                clip_eps=clip_eps, kl_coef=kl_coef, clamp=clamp))
            pre_rollouts.append({
                "rollout_id": item["rollout_id"],
                "weighted_logp_sum": float(weighted_sum.item()),
                "active_loss_weight": float(normalizer.item()),
                **_objective_metrics(
                    float(policy_mean.item()), old_mean, item["advantage"],
                    clip_eps=clip_eps, kl_coef=kl_coef, clamp=clamp),
            })
        loss = torch.stack(rollout_losses).mean()
        if not torch.isfinite(loss).item():
            raise RuntimeError("non-finite measured GRPO loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float(prepared["identity"]["optimization"]["max_grad_norm"]),
            error_if_nonfinite=True)
        grad_tensors = [param.grad for param in policy.parameters() if param.grad is not None]
        if not grad_tensors or not any(torch.count_nonzero(grad).item() > 0
                                      for grad in grad_tensors):
            raise RuntimeError("real GRPO step produced no non-zero gradient")
        optimizer.step()
        torch.cuda.synchronize(device)

        post_rollouts = []
        for index, item in enumerate(group["rollouts"]):
            measured = dpo_live._measured_logprob(torch, policy, item, device)
            old_mean = behavior_values[step_index][index]["mean_logp"]
            post_rollouts.append({
                "rollout_id": item["rollout_id"],
                "weighted_logp_sum": measured["weighted_logp_sum"],
                "active_loss_weight": measured["active_loss_weight"],
                **_objective_metrics(
                    measured["mean_logp"], old_mean, item["advantage"],
                    clip_eps=clip_eps, kl_coef=kl_coef, clamp=clamp),
            })
        post_loss = sum(item["objective_loss"] for item in post_rollouts) / len(post_rollouts)
        step_elapsed = time.perf_counter() - step_started
        changed_tensors = changed_parameters = 0
        max_delta = delta_l2_squared = 0.0
        finite_after = True
        for name, param in policy.named_parameters():
            after = param.detach().cpu()
            finite_after = finite_after and bool(torch.isfinite(after).all().item())
            difference = after.float() - before[name].float()
            count = int(torch.count_nonzero(difference).item())
            if count:
                changed_tensors += 1
                changed_parameters += count
                max_delta = max(max_delta, float(difference.abs().max().item()))
                delta_l2_squared += float(difference.square().sum().item())
        if not finite_after or not changed_parameters or not math.isfinite(max_delta):
            raise RuntimeError("optimizer did not produce changed finite parameters")
        after_sha = sft_live._model_fingerprint(torch, policy)
        if sft_live._model_fingerprint(torch, behavior) != behavior_sha:
            raise RuntimeError("GRPO behavior/reference weights changed during execution")
        optimizer_sha = sft_live._optimizer_fingerprint(torch, policy, optimizer)
        rollout_ids = [item["rollout_id"] for item in group["rollouts"]]
        entry = {
            "step": step_index + 1,
            "schedule_step_index": step_index,
            "record_ids": [group["group_id"]],
            "rollout_ids": rollout_ids,
            "record_order_sha256": stage_c.sha256_bytes(
                stage_c.canonical_json([group["group_id"]])),
            "membership_order_sha256": stage_c.sha256_bytes(
                stage_c.canonical_json(rollout_ids)),
            "loss": float(loss.item()),
            "post_update_loss": post_loss,
            "post_update_loss_delta": post_loss - float(loss.item()),
            "clip_eps": clip_eps,
            "kl_coef": kl_coef,
            "log_ratio_clamp": clamp,
            "old_policy": "frozen_initial_base_model",
            "reference_policy": "frozen_initial_base_model",
            "behavior_parameter_sha256": behavior_sha,
            "policy_ratio_kl_evidence": {
                "pre_update": pre_rollouts,
                "post_update": post_rollouts,
                "mean_pre_ratio": sum(item["ratio"] for item in pre_rollouts) / len(pre_rollouts),
                "mean_post_ratio": sum(item["ratio"] for item in post_rollouts) / len(post_rollouts),
                "pre_clip_fraction": sum(item["clip_active"] for item in pre_rollouts) / len(pre_rollouts),
                "post_clip_fraction": sum(item["clip_active"] for item in post_rollouts) / len(post_rollouts),
                "mean_pre_kl": sum(item["kl"] for item in pre_rollouts) / len(pre_rollouts),
                "mean_post_kl": sum(item["kl"] for item in post_rollouts) / len(post_rollouts),
            },
            "grad_norm": float(grad_norm),
            "nonzero_gradient_tensors": sum(
                torch.count_nonzero(grad).item() > 0 for grad in grad_tensors),
            "gradient_tensors": len(grad_tensors),
            "optimizer_steps": 1,
            "optimizer": "torch.optim.AdamW",
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
            "changed_parameter_tensors": changed_tensors,
            "changed_parameters": changed_parameters,
            "max_abs_parameter_delta": max_delta,
            "parameter_delta_l2": math.sqrt(delta_l2_squared),
            "parameters_finite_after": finite_after,
            "parameter_sha256_before": before_sha,
            "parameter_sha256_after": after_sha,
            "optimizer_state_sha256": optimizer_sha,
            "step_wall_ms": round(step_elapsed * 1000.0, 3),
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "group_audit": {
                **prepared["group_audits"][step_index],
                "rollouts": [item["audit"] for item in group["rollouts"]],
            },
        }
        history.append(entry)
        if (step_index + 1) % prepared["checkpoint_every"] == 0 \
                or step_index + 1 == prepared["stop_after"]:
            save_checkpoint(step_index + 1)

    torch.cuda.synchronize(device)
    if sft_live._model_fingerprint(torch, behavior) != behavior_sha:
        raise RuntimeError("final GRPO behavior/reference fingerprint differs from initial base")
    result = {
        "schema": LIVE_MANIFEST_SCHEMA,
        "status": "PASS",
        "REAL_TRAINING": "EXECUTED",
        "optimizer_status": "EXECUTED",
        "algorithm": "joint-interleaved-sequence-clipped-grpo-schulman-kl-adamw-v1",
        "claims": {
            "model_quality": "NOT_RUN",
            "generalized_improvement": "NOT_RUN",
            "online_rollout_generation": "NOT_RUN",
            "production_model_training": "NOT_RUN",
            "upstream_revision_verification": "NOT_RUN",
            "vision_or_media_forward": "NOT_RUN",
            "serving_process_started": False,
        },
        "source": prepared["source"],
        "config_file_sha256": prepared["config_file_sha256"],
        "resolved_config_sha256": prepared["plan"]["config_sha256"],
        "dataset": prepared["plan"]["dataset"],
        "source_exports": {
            "inputs": prepared["export_inputs"],
            "reward": prepared["identity"]["reward"],
            "trajectory_revalidation": "PASS",
            "cas_revalidation": "PASS",
        },
        "model": prepared["model"],
        "seed": {"text": seed_text, "derived_uint31": seed},
        "batch_schedule": prepared["plan"]["batch_schedules"]["grpo"],
        "group_order_sha256": stage_c.sha256_bytes(stage_c.canonical_json([
            row["group_id"] for row in prepared["rows"]
        ])),
        "executed_order_sha256": stage_c.sha256_bytes(stage_c.canonical_json([
            row["group_id"] for row in prepared["rows"][next_step:prepared["stop_after"]]
        ])),
        "run_identity_sha256": prepared["identity_sha256"],
        "resume": {"used": next_step > 0, "checkpoint_sha256": resume_sha256,
                   "initial_step": next_step},
        "stop_after": prepared["stop_after"],
        "optimizer_step_count": len(history),
        "steps": history,
        "initial_parameter_sha256": initial_parameter_sha,
        "behavior_parameter_sha256": behavior_sha,
        "final_parameter_sha256": history[-1]["parameter_sha256_after"],
        "final_optimizer_state_sha256": history[-1]["optimizer_state_sha256"],
        "checkpoints": checkpoints,
        "timing": {"total_wall_ms": round((time.perf_counter() - started) * 1000.0, 3)},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_visible_devices": int(visible),
            "visible_device_count": torch.cuda.device_count(),
            "device": "cuda:0",
            "device_name": torch.cuda.get_device_name(device),
            "device_capability": list(torch.cuda.get_device_capability(device)),
            "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        },
    }
    sft_live._atomic_json(output / "live_manifest.json", result)
    return result


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Execute the bounded real Stage C joint-GRPO optimizer gate")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--trajectory-path", action="append", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume-from", default=None)
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        prepared = prepare_live_run(
            dataset_dir=args.dataset_dir, trajectory_paths=args.trajectory_path,
            output_dir=args.output_dir, config_path=args.config,
            model_path=args.model_path, source_commit=args.source_commit,
            max_steps=args.max_steps, checkpoint_every=args.checkpoint_every,
            stop_after=args.stop_after, resume_from=args.resume_from,
            model_revision=args.model_revision)
        result = execute_live_run(prepared)
    except (OSError, ValueError, RuntimeError, ImportError, json.JSONDecodeError) as exc:
        parser.exit(2, f"joint GRPO live gate error: {exc}\n")
    sys.stdout.write(stage_c.canonical_json(result).decode("utf-8") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
