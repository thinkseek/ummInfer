# SPDX-License-Identifier: Apache-2.0
"""Bounded real DPO optimizer adapter for the existing Stage C trainer.

All model-independent validation delegates to the schema-v2 dataset pack,
Stage C planner, span masker, and deterministic scheduler.  Torch and
Transformers are imported only by :func:`execute_live_run` after pair, CAS
metadata, configuration, model-file, and checkpoint-sidecar validation.

The gate executes sigmoid DPO on a local Qwen2 causal LM with a frozen,
independent copy of the initial base model as its reference.  It is an
architecture/optimizer acceptance gate, not a model-quality or production
training result.  It never starts or contacts a serving process.
"""

import argparse
import builtins
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time

# Direct script execution uses -I, so add only this repository's known path.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

import umminfer_joint_sft_live as sft_live  # noqa: E402
import umminfer_joint_trainer as stage_c  # noqa: E402
import umminfer_sft_dataset as sft_dataset  # noqa: E402
import umminfer_token_export as token_export  # noqa: E402

CHECKPOINT_FORMAT = "umminfer-joint-dpo-live-checkpoint-v1"
LIVE_MANIFEST_SCHEMA = "umminfer-joint-dpo-live-result-v1"
TOKEN_EXPORT_BUNDLE_SCHEMA = "umminfer-joint-dpo-token-export-preflight-v1"
SOURCE_FILES = (
    "python/umminfer_joint_dpo_live.py",
    "python/umminfer_joint_sft_live.py",
    "python/umminfer_joint_trainer.py",
    "python/umminfer_rl_trajectory.py",
)
TOKEN_EXPORT_SOURCE_FILES = (
    "python/umminfer_sft_dataset.py",
    "python/umminfer_token_export.py",
)
_HEX_40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(float(value)))


def _source_fingerprints(source_commit, *, include_token_export=False):
    if not isinstance(source_commit, str) or not _HEX_40.fullmatch(source_commit):
        raise ValueError("--source-commit must be one lowercase 40-character commit")
    files = {}
    rels = list(SOURCE_FILES)
    if include_token_export:
        rels.extend(TOKEN_EXPORT_SOURCE_FILES)
    for rel in rels:
        path = ROOT / rel
        if not path.is_file():
            raise ValueError(f"missing live-gate source file: {rel}")
        files[rel] = {
            "sha256": stage_c.sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return {
        "commit": source_commit,
        "files": files,
        "source_tree_scope": "listed Stage C source files only",
    }


def _audit_response(response, *, side):
    """Validate one schema-v2-derived chosen or rejected response."""
    prefix = f"{side} response"
    if not isinstance(response, dict):
        raise ValueError(f"{prefix} must be a JSON object")
    target = response.get("assistant_target")
    clean_text = response.get("clean_text")
    if not isinstance(target, str) or not target:
        raise ValueError(f"{prefix} assistant_target must be non-empty")
    if not isinstance(clean_text, str) or not clean_text:
        raise ValueError(f"{prefix} clean_text must be non-empty")
    target_sha = response.get("target_sha256")
    if not isinstance(target_sha, str) or not _HEX_64.fullmatch(target_sha) \
            or target_sha != stage_c.sha256_text(target):
        raise ValueError(f"{prefix} target_sha256 mismatch")
    if not _finite(response.get("score")):
        raise ValueError(f"{prefix} score must be finite")
    provenance = response.get("provenance")
    if not isinstance(provenance, dict) \
            or provenance.get("source_schema_version") != 2:
        raise ValueError(f"{prefix} requires schema-v2 trajectory provenance")
    if not provenance.get("source_session_id") \
            or not provenance.get("source_request_id"):
        raise ValueError(f"{prefix} lacks source session/request identity")

    slots = response.get("visual_slots")
    if not isinstance(slots, list):
        raise ValueError(f"{prefix} visual_slots must be a list")
    output_refs = response.get("output_media_refs")
    if not isinstance(output_refs, list):
        raise ValueError(f"{prefix} output_media_refs must be a list")
    verified_by_slot = {}
    for ref in output_refs:
        if not isinstance(ref, dict) or ref.get("verified_on_disk") is not True:
            raise ValueError(f"{prefix} requires on-disk-verified CAS references")
        digest = ref.get("sha256")
        if not isinstance(digest, str) or not _HEX_64.fullmatch(digest):
            raise ValueError(f"{prefix} has an invalid CAS digest")
        if not isinstance(ref.get("ref"), str) or Path(ref["ref"]).name != ref["ref"]:
            raise ValueError(f"{prefix} CAS reference must be a relative basename")
        if ref.get("slot_id") is not None:
            verified_by_slot[ref["slot_id"]] = ref

    slot_ids = set()
    for slot in slots:
        if not isinstance(slot, dict) \
                or slot.get("media_type") not in ("image", "video"):
            raise ValueError(f"{prefix} visual slot has an invalid modality")
        slot_id = slot.get("slot_id")
        if not isinstance(slot_id, str) or not slot_id or slot_id in slot_ids:
            raise ValueError(f"{prefix} visual slot ids must be non-empty and unique")
        slot_ids.add(slot_id)
        if slot.get("leg_state") != "completed":
            raise ValueError(f"{prefix} visual slot lacks a completed terminal leg")
        marker = slot.get("marker")
        if not isinstance(marker, str) or marker not in target:
            raise ValueError(f"{prefix} visual marker is absent from assistant_target")
        digest = slot.get("artifact_sha256")
        if not isinstance(digest, str) or not _HEX_64.fullmatch(digest):
            raise ValueError(f"{prefix} visual slot lacks an artifact SHA-256")
        bound = verified_by_slot.get(slot_id)
        if bound is None or bound.get("sha256") != digest \
                or bound.get("media_type") != slot["media_type"]:
            raise ValueError(f"{prefix} slot/CAS/terminal binding is incomplete")
        if slot.get("artifact_ref") != bound.get("ref"):
            raise ValueError(f"{prefix} slot artifact reference differs from verified CAS")

    expected_route = [slot["media_type"] for slot in slots]
    if response.get("modality_route") != expected_route:
        raise ValueError(f"{prefix} modality_route differs from visual slots")
    mask = stage_c.build_span_loss_mask(
        target, ablation_profile="joint_full",
        span_weights=stage_c.DEFAULT_SPAN_WEIGHTS)
    if not mask["active_tokens"] or mask["masked_tokens"]:
        raise ValueError(f"{prefix} must have only active joint_full loss spans")
    if slots and not (mask["span_counts"]["visual_route_tag"]
                      and mask["span_counts"]["visual_prompt_body"]):
        raise ValueError(f"{prefix} lacks route-tag or prompt-body supervision")
    return {
        "target_sha256": target_sha,
        "clean_text_sha256": stage_c.sha256_text(clean_text),
        "score": float(response["score"]),
        "visual_modalities": sorted({slot["media_type"] for slot in slots}),
        "terminal_legs_verified": len(slots),
        "verified_cas_refs": len(output_refs),
        "span_counts": mask["span_counts"],
        "stdlib_mask_tokens": mask["total_tokens"],
        "source_session_id": provenance["source_session_id"],
        "source_request_id": provenance["source_request_id"],
        "source_prompt_id": provenance.get("source_prompt_id"),
    }


def _audit_pair(row):
    """Fail closed on malformed, crossed, or prompt-inconsistent DPO pairs."""
    if not isinstance(row, dict) or row.get("record_type") != "dpo_pair":
        raise ValueError("live gate accepts only dpo_pair records")
    if row.get("schema_version") != 1:
        raise ValueError("unsupported Stage C DPO record schema")
    pair_id = row.get("pair_id")
    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError("DPO pair_id must be non-empty")
    prompt = row.get("prompt_text")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("DPO prompt_text must be a non-empty string")
    prompt_sha = row.get("prompt_sha256")
    if not isinstance(prompt_sha, str) or not _HEX_64.fullmatch(prompt_sha):
        raise ValueError("DPO prompt_sha256 is invalid")
    prompt_parts = row.get("prompt_parts")
    if not isinstance(prompt_parts, list) or not prompt_parts \
            or stage_c.sha256_bytes(stage_c.canonical_json(prompt_parts)) != prompt_sha:
        raise ValueError("DPO prompt_parts do not recompute prompt_sha256")
    reward_name = row.get("reward_name")
    if not isinstance(reward_name, str) or not reward_name:
        raise ValueError("DPO reward_name must be non-empty")

    chosen = _audit_response(row.get("chosen"), side="chosen")
    rejected = _audit_response(row.get("rejected"), side="rejected")
    if chosen["target_sha256"] == rejected["target_sha256"]:
        raise ValueError("DPO chosen and rejected targets must be distinct")
    if chosen["source_session_id"] == rejected["source_session_id"] \
            or chosen["source_request_id"] == rejected["source_request_id"]:
        raise ValueError("DPO chosen and rejected provenance must be distinct")
    chosen_prompt_id = chosen["source_prompt_id"]
    rejected_prompt_id = rejected["source_prompt_id"]
    if chosen_prompt_id != rejected_prompt_id:
        raise ValueError("DPO chosen and rejected provenance cross different prompts")
    if not chosen["score"] > rejected["score"]:
        raise ValueError("crossed DPO pair: chosen score must exceed rejected score")
    margin = row.get("margin")
    if not _finite(margin) or float(margin) <= 0.0 \
            or not math.isclose(float(margin), chosen["score"] - rejected["score"],
                                rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("DPO margin differs from chosen/rejected scores")
    expected_pair_id = stage_c.sha256_text(
        f"{prompt_sha}:{reward_name}:{chosen['target_sha256']}:"
        f"{rejected['target_sha256']}")[:32]
    if pair_id != expected_pair_id:
        raise ValueError("DPO pair_id mismatch: crossed prompt or chosen/rejected targets")
    return {
        "pair_id": pair_id,
        "prompt_sha256": prompt_sha,
        "margin": float(margin),
        "chosen": chosen,
        "rejected": rejected,
    }


def _dpo_export_record(row, side):
    """Rebuild one response's source record and bind it to the whole pair."""
    response = row[side]
    has_block = "token_export" in response
    has_source = "token_export_source" in response
    if not has_block and not has_source:
        return None
    if not has_block or not has_source:
        raise ValueError(f"DPO {side} token_export and source must appear together")
    block = response["token_export"]
    source = response["token_export_source"]
    if not isinstance(block, dict) or not isinstance(source, dict) \
            or "token_export" in source:
        raise ValueError(f"DPO {side} token export/source binding is malformed")
    record = copy.deepcopy(source)
    record["token_export"] = copy.deepcopy(block)
    try:
        token_export.validate_record_token_export(
            record, expected_sft_target=response.get("assistant_target"))
    except token_export.TokenExportError as exc:
        raise ValueError(f"DPO {side} token-export source is invalid: {exc}") from exc
    candidate = sft_dataset._extract_candidate(record, verified_on_disk=True)
    for key in (
            "target_sha256", "assistant_target", "clean_text", "visual_slots",
            "modality_route", "output_media_refs", "provenance"):
        if stage_c.canonical_json(response.get(key)) != stage_c.canonical_json(
                candidate[key]):
            raise ValueError(f"DPO {side} differs from token-export source field: {key}")
    for key in ("prompt_sha256", "prompt_text", "prompt_parts", "input_media_refs"):
        if stage_c.canonical_json(row.get(key)) != stage_c.canonical_json(candidate[key]):
            raise ValueError(f"DPO {side} prompt differs from token-export source: {key}")
    return record


def _structural_export_pairs(rows):
    """Validate all pairs and both sides before tokenizer/ML imports or output."""
    if not rows:
        raise ValueError("token_export mode requires non-empty dpo_train rows")
    records = {}
    tokenizer_identity = None
    for row in rows:
        audit = _audit_pair(row)
        pair_id = audit["pair_id"]
        if pair_id in records:
            raise ValueError(f"duplicate exported DPO pair id: {pair_id}")
        sides = {}
        for side in ("chosen", "rejected"):
            record = _dpo_export_record(row, side)
            if record is None:
                raise ValueError(
                    f"token_export was requested but DPO pair {pair_id!r} {side} "
                    "has no export; refusing per-side legacy fallback")
            identity = record["token_export"].get("tokenizer")
            if tokenizer_identity is None:
                tokenizer_identity = copy.deepcopy(identity)
            elif stage_c.canonical_json(identity) != stage_c.canonical_json(
                    tokenizer_identity):
                raise ValueError("mixed DPO token-export tokenizer identities")
            sides[side] = record
        if sides["chosen"]["token_export"]["source_record_sha256"] == \
                sides["rejected"]["token_export"]["source_record_sha256"]:
            raise ValueError("chosen and rejected exports must be distinct source records")
        records[pair_id] = sides
    return records


def prepare_live_run(*, dataset_dir, output_dir, config_path, model_path,
                     source_commit, max_steps, checkpoint_every=1,
                     stop_after=None, resume_from=None, model_revision=None,
                     token_source="legacy"):
    """Validate every model-independent input before any ML framework import."""
    if token_source not in ("legacy", "token_export"):
        raise ValueError("--token-source must be legacy or token_export")
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
        if not resume_path.is_file() or not sidecar.is_file():
            raise ValueError("resume checkpoint and SHA-256 sidecar must exist")
        if sidecar.read_text(encoding="ascii").strip() != stage_c.sha256_file(resume_path):
            raise ValueError("resume checkpoint SHA-256 mismatch")

    config_file = Path(config_path)
    if not config_file.is_file():
        raise ValueError("live training config does not exist")
    config_sha256 = stage_c.sha256_file(config_file)
    output = Path(output_dir)
    export_preflight = None
    model = None
    if token_source == "token_export":
        # The structural pair gate deliberately precedes planner/output creation,
        # tokenizer loading, model loading, and every ML-framework import.
        early_dataset = stage_c.verify_and_load_dataset_pack(dataset_dir)
        early_rows = early_dataset["packs"].get("dpo_train") or []
        _structural_export_pairs(early_rows)
        raw_config = json.loads(config_file.read_text(encoding="utf-8"))
        raw_dpo = raw_config.get("dpo", {})
        if not _is_int(raw_dpo.get("max_length")) or raw_dpo["max_length"] <= 1:
            raise ValueError("export preflight requires a valid dpo.max_length")
        model = sft_live._fingerprint_model(model_path)
        if model_revision is not None:
            if not isinstance(model_revision, str) or not model_revision:
                raise ValueError("model revision override must be non-empty")
            model["revision"] = model_revision
            model["upstream_revision_verification"] = "NOT_RUN"
        export_preflight = _run_export_preflight(
            dataset_dir=dataset_dir, model_path=model_path,
            config_path=config_file, tokenizer_id=model["model_id"],
            tokenizer_revision=model["revision"])
    plan = stage_c.prepare_training_run(
        mode="dpo", ablation="joint_full", dataset_dir=dataset_dir,
        output_dir=output / "plan", config_path=config_file,
        gpu_launch_script=True)
    if plan["REAL_TRAINING"] != "NOT_RUN" or plan["active_stages"] != ["dpo"]:
        raise ValueError("Stage C planner did not produce a DPO-only NOT_RUN plan")

    hyper = plan["hyperparameters"]
    model_cfg = hyper["model"]
    peft_cfg = hyper["peft"]
    opt = hyper["optimization"]
    dpo = hyper["dpo"]
    expected_model = {
        "dtype": "bfloat16",
        "attn_implementation": "eager",
        "gradient_checkpointing": False,
        "trust_remote_code": False,
    }
    if any(model_cfg.get(key) != value for key, value in expected_model.items()):
        raise ValueError("live gate requires bf16 eager local Qwen2 without remote code")
    if peft_cfg.get("method") != "none":
        raise ValueError("bounded gate supports full-parameter DPO only")
    if opt.get("lr_scheduler") != "constant" or opt.get("warmup_ratio") != 0.0:
        raise ValueError("bounded gate requires constant LR with zero warmup")
    for key in ("per_device_train_batch_size", "gradient_accumulation_steps",
                "num_train_epochs"):
        if opt.get(key) != 1:
            raise ValueError(f"bounded gate requires {key}=1")
    if not _finite(opt.get("learning_rate")) or float(opt["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not _finite(opt.get("weight_decay")) or float(opt["weight_decay"]) != 0.0:
        raise ValueError("weight_decay must be zero for measured gradient updates")
    if not _finite(opt.get("max_grad_norm")) or float(opt["max_grad_norm"]) <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    if not _is_int(opt.get("max_seq_len")) or opt["max_seq_len"] <= 1:
        raise ValueError("max_seq_len must be an integer greater than one")
    if not _finite(dpo.get("beta")) or float(dpo["beta"]) <= 0.0:
        raise ValueError("DPO beta must be finite and positive")
    if dpo.get("loss_type") != "sigmoid" \
            or not _finite(dpo.get("label_smoothing")) \
            or float(dpo["label_smoothing"]) != 0.0:
        raise ValueError("bounded gate requires sigmoid DPO with zero label smoothing")
    if dpo.get("reference_policy") != "frozen_initial_base_model" \
            or dpo.get("reference_logprob_normalization") != "span_weighted_target_mean":
        raise ValueError("bounded gate requires the frozen base reference and weighted-mean logps")
    if dpo.get("max_length") != opt["max_seq_len"] \
            or not _is_int(dpo.get("max_prompt_length")) \
            or not 0 < dpo["max_prompt_length"] < dpo["max_length"]:
        raise ValueError("DPO prompt/sequence length limits are inconsistent")

    dataset = stage_c.verify_and_load_dataset_pack(dataset_dir)
    rows = dataset["packs"].get("dpo_train") or []
    all_audits = [_audit_pair(row) for row in rows]
    schedule = plan["batch_schedules"]["dpo"]
    if max_steps > schedule["total_batches"]:
        raise ValueError("--max-steps exceeds the deterministic Stage C schedule")
    scheduled_ids = [
        record_id
        for batch in schedule["batches"][:max_steps]
        for record_id in batch["record_ids"]
    ]
    by_id = {row["pair_id"]: row for row in rows}
    if len(by_id) != len(rows) or any(item not in by_id for item in scheduled_ids):
        raise ValueError("scheduled pair ids are absent or non-unique")
    scheduled_rows = [by_id[item] for item in scheduled_ids]
    export_packed = None
    if export_preflight is not None:
        if stage_c.canonical_json(plan["ablation"]["span_weights"]) \
                != stage_c.canonical_json(stage_c.DEFAULT_SPAN_WEIGHTS):
            raise ValueError("token_export v1/v2 is bound to the joint_full default weights")
        packed_by_id = {item["pair_id"]: item
                        for item in export_preflight["packed"]}
        if len(packed_by_id) != len(rows) or set(packed_by_id) != set(by_id):
            raise ValueError("token-export preflight pairs differ from the frozen DPO pack")
        export_packed = [packed_by_id[item] for item in scheduled_ids]
    scheduled_audits = [_audit_pair(row) for row in scheduled_rows]
    for audit in scheduled_audits:
        for side in ("chosen", "rejected"):
            if not audit[side]["visual_modalities"] \
                    or not audit[side]["terminal_legs_verified"] \
                    or not audit[side]["verified_cas_refs"]:
                raise ValueError(
                    "scheduled gate requires structured visual/CAS/terminal targets on both sides")

    if model is None:
        model = sft_live._fingerprint_model(model_path)
    if model_revision is not None and token_source != "token_export":
        if not isinstance(model_revision, str) or not model_revision:
            raise ValueError("model revision override must be non-empty")
        model["revision"] = model_revision
        model["upstream_revision_verification"] = "NOT_RUN"
    source = _source_fingerprints(
        source_commit, include_token_export=token_source == "token_export")
    identity = {
        "source": source,
        "config_file_sha256": config_sha256,
        "resolved_config_sha256": plan["config_sha256"],
        "dataset_manifest_sha256": plan["dataset"]["manifest_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "model": model,
        "optimization": opt,
        "dpo": dpo,
        "span_loss_weights": plan["ablation"]["span_weights"],
        "max_steps": max_steps,
        "objective": "full-parameter-span-weighted-sigmoid-dpo-v1",
    }
    reference_policy_inputs = None
    if export_preflight is not None:
        reference_policy_inputs = {
            "policy": dpo["reference_policy"],
            "normalization": dpo["reference_logprob_normalization"],
            "model_weights_sha256": model["available_weights_sha256"],
            "pairs": [
                {
                    "pair_id": pair["pair_id"],
                    "prompt_sha256": pair["prompt_sha256"],
                    "chosen": {
                        key: pair["chosen"][key]
                        for key in ("input_ids_sha256", "labels_sha256",
                                    "loss_weights_sha256")
                    },
                    "rejected": {
                        key: pair["rejected"][key]
                        for key in ("input_ids_sha256", "labels_sha256",
                                    "loss_weights_sha256")
                    },
                }
                for pair in export_preflight["pairs"]
            ],
        }
        identity["token_source"] = {
            "mode": "token_export",
            "schema": export_preflight["schema"],
            "tokenizer": export_preflight["tokenizer"],
            "tokenization_version": export_preflight["tokenization_version"],
            "pairs": export_preflight["pairs"],
            "claims": export_preflight["claims"],
        }
        identity["reference_policy_inputs"] = reference_policy_inputs
    return {
        "output_dir": output,
        "dataset_dir": Path(dataset_dir),
        "config_path": config_file,
        "model_path": Path(model_path),
        "plan": plan,
        "rows": scheduled_rows,
        "token_source": token_source,
        "export_packed": export_packed,
        "token_export_preflight": export_preflight,
        "reference_policy_inputs": reference_policy_inputs,
        "pair_audits": scheduled_audits,
        "all_pair_audits": all_audits,
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


def _pack_response(tokenizer, row, response, audit, *, side, prepared):
    prompt = row["prompt_text"]
    target = response["assistant_target"]
    full_text = prompt + "\n" + target
    target_start = len(prompt) + 1
    encoded = tokenizer(
        full_text, add_special_tokens=False, return_offsets_mapping=True,
        truncation=False)
    ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    dpo = prepared["identity"]["dpo"]
    if not ids or len(ids) > dpo["max_length"]:
        raise ValueError(f"{side} packed target is empty or exceeds max_length")
    mask_info = stage_c.build_span_loss_mask(
        target, ablation_profile="joint_full",
        span_weights=prepared["identity"]["span_loss_weights"])
    labels, weights, annotations = sft_live._token_loss_mask(
        offsets, target_start=target_start, target=target,
        token_ids=ids, mask_info=mask_info)
    route_tokens = sum(
        "visual_route_tag" in item["span_types"] for item in annotations)
    prompt_body_tokens = sum(
        "visual_prompt_body" in item["span_types"] for item in annotations)
    closing_tokens = sum(
        item.get("includes_terminal_delimiter", False) for item in annotations)
    prompt_masked = sum(value == 0 for value in weights)
    supervised = sum(value > 0 for value in weights)
    if prompt_masked > dpo["max_prompt_length"]:
        raise ValueError(f"{side} prompt exceeds max_prompt_length")
    if not route_tokens or not prompt_body_tokens or not closing_tokens:
        raise ValueError(f"{side} HF mask misses visual route, prompt, or terminal delimiter")
    target_audit = {
        **audit,
        "side": side,
        "sequence_tokens": len(ids),
        "supervised_target_tokens": supervised,
        "prompt_masked_tokens": prompt_masked,
        "visual_route_tag_tokens": route_tokens,
        "visual_prompt_body_tokens": prompt_body_tokens,
        "terminal_delimiter_tokens": closing_tokens,
        "input_ids_sha256": stage_c.sha256_bytes(stage_c.canonical_json(ids)),
        "labels_sha256": stage_c.sha256_bytes(stage_c.canonical_json(labels)),
        "loss_weights_sha256": stage_c.sha256_bytes(stage_c.canonical_json(weights)),
        "active_loss_weight": sum(weights),
    }
    return {
        "input_ids": ids,
        "labels": labels,
        "weights": weights,
        "offsets": offsets,
        "annotations": annotations,
        "prompt_token_ids": ids[:prompt_masked],
        "audit": target_audit,
    }


def _pack_export_pairs(rows, tokenizer, *, tokenizer_dir, tokenizer_id,
                       tokenizer_revision, tokenization_version,
                       span_loss_weights, dpo_config):
    """Strongly validate both exports per pair and consume their target ids.

    Only the shared unexported prompt prefix is encoded for the training pack.
    Full legacy encodings are equality references only; no legacy target value
    is substituted for an exported id, offset, label, weight, or annotation.
    """
    records = _structural_export_pairs(rows)
    packed = []
    summaries = []
    identity = None
    reference_prepared = {
        "identity": {"dpo": dpo_config, "span_loss_weights": span_loss_weights},
    }
    for row in rows:
        pair_audit = _audit_pair(row)
        pair_id = row["pair_id"]
        prompt = row["prompt_text"]
        prefix = tokenizer(
            prompt + "\n", add_special_tokens=False, return_offsets_mapping=True,
            truncation=False)
        prefix_ids = list(prefix["input_ids"])
        prefix_offsets = list(prefix["offset_mapping"])
        if not prefix_ids:
            raise ValueError(f"DPO pair {pair_id} has an empty prompt prefix")
        sides = {}
        side_summaries = {}
        for side in ("chosen", "rejected"):
            response = row[side]
            record = records[pair_id][side]
            block = record["token_export"]
            if identity is None:
                identity = copy.deepcopy(block["tokenizer"])
            elif stage_c.canonical_json(identity) != stage_c.canonical_json(
                    block["tokenizer"]):
                raise ValueError("mixed DPO token-export tokenizer identities")
            token_export.validate_record_token_export(
                record, tokenizer=tokenizer, tokenizer_dir=tokenizer_dir,
                expected_tokenizer_id=tokenizer_id,
                expected_tokenizer_revision=tokenizer_revision,
                expected_tokenization_version=tokenization_version,
                expected_sft_target=response["assistant_target"])
            legacy = _pack_response(
                tokenizer, row, response, pair_audit[side], side=side,
                prepared=reference_prepared)
            target = response["assistant_target"]
            stream = block["streams"]["sft_target"]
            target_ids = list(stream["token_ids"])
            if not target_ids or len(prefix_ids) + len(target_ids) != len(
                    legacy["input_ids"]):
                raise ValueError(f"DPO {side} export/prompt token counts differ from legacy")
            if stage_c.canonical_json(prefix_ids) != stage_c.canonical_json(
                    legacy["input_ids"][:len(prefix_ids)]):
                raise ValueError(f"DPO {side} prompt-prefix ids differ from legacy")
            if stage_c.canonical_json(prefix_offsets) != stage_c.canonical_json(
                    legacy["offsets"][:len(prefix_ids)]):
                raise ValueError(f"DPO {side} prompt-prefix offsets differ from legacy")
            target_offsets = sft_live._export_raw_offsets(
                stream, target_start=len(prompt) + 1)
            if stage_c.canonical_json(target_offsets) != stage_c.canonical_json(
                    legacy["offsets"][len(prefix_ids):]):
                raise ValueError(f"DPO {side} raw offsets differ from legacy")
            mask_info = stage_c.build_span_loss_mask(
                target, ablation_profile="joint_full",
                span_weights=span_loss_weights)
            target_labels, target_weights, target_annotations = \
                sft_live._token_loss_mask(
                    target_offsets, target_start=len(prompt) + 1, target=target,
                    token_ids=target_ids, mask_info=mask_info)
            if target_labels != target_ids \
                    or target_weights != stream["sft_mask"]["token_weights"]:
                raise ValueError(f"DPO {side} exported labels/weights differ from its mask")
            annotations_sha = stage_c.sha256_bytes(
                stage_c.canonical_json(target_annotations))
            if annotations_sha != stream["sft_mask"]["annotations_sha256"]:
                raise ValueError(f"DPO {side} annotations do not recompute from raw offsets")
            ids = prefix_ids + target_ids
            labels = [-100] * len(prefix_ids) + target_labels
            weights = [0.0] * len(prefix_ids) + target_weights
            annotations = [{
                "span_types": [], "weight": 0.0,
                "includes_terminal_delimiter": False,
            } for _ in prefix_ids] + target_annotations
            for key, actual, expected in (
                    ("input_ids", ids, legacy["input_ids"]),
                    ("labels", labels, legacy["labels"]),
                    ("weights", weights, legacy["weights"]),
                    ("annotations", annotations, legacy["annotations"])):
                if stage_c.canonical_json(actual) != stage_c.canonical_json(expected):
                    raise ValueError(f"DPO export-driven {side} {key} differ from legacy")
            item = copy.deepcopy(legacy)
            item.update({
                "input_ids": ids, "labels": labels, "weights": weights,
                "annotations": annotations, "prompt_token_ids": prefix_ids,
            })
            raw_offsets_sha = stage_c.sha256_bytes(
                stage_c.canonical_json(target_offsets))
            export_summary = {
                "format": block["format"],
                "format_version": block["format_version"],
                "export_sha256": block["export_sha256"],
                "source_record_sha256": block["source_record_sha256"],
                "input_ids_sha256": item["audit"]["input_ids_sha256"],
                "labels_sha256": item["audit"]["labels_sha256"],
                "loss_weights_sha256": item["audit"]["loss_weights_sha256"],
                "annotations_sha256": annotations_sha,
                "raw_offsets_sha256": raw_offsets_sha,
                "prompt_tokens": len(prefix_ids),
                "target_tokens": len(target_ids),
                "tokenizer_reencoded": True,
                "legacy_equality": "PASS",
            }
            item["token_export"] = export_summary
            item["raw_target_offsets"] = target_offsets
            sides[side] = item
            side_summaries[side] = export_summary
        if sides["chosen"]["prompt_token_ids"] != sides["rejected"]["prompt_token_ids"]:
            raise ValueError("chosen/rejected exported prompt tokenization differs")
        packed.append({"pair_id": pair_id, **sides})
        prompt_prefix_sha = stage_c.sha256_bytes(stage_c.canonical_json({
            "input_ids": prefix_ids, "offsets": prefix_offsets,
        }))
        summaries.append({
            "pair_id": pair_id,
            "pair_identity_sha256": stage_c.sha256_text(
                f"{pair_id}:{row['prompt_sha256']}:"
                f"{side_summaries['chosen']['export_sha256']}:"
                f"{side_summaries['rejected']['export_sha256']}"),
            "prompt_sha256": row["prompt_sha256"],
            "prompt_prefix_sha256": prompt_prefix_sha,
            "chosen": side_summaries["chosen"],
            "rejected": side_summaries["rejected"],
        })
    return {
        "schema": TOKEN_EXPORT_BUNDLE_SCHEMA,
        "status": "PASS",
        "tokenizer": identity,
        "tokenization_version": tokenization_version,
        "pairs": summaries,
        "packed": packed,
        "claims": {
            "real_dpo_optimizer": "NOT_RUN",
            "model_quality": "NOT_RUN",
            "sampled_online_behavior_ids": "NOT_CLAIMED",
            "target_retokenization_for_training": False,
            "legacy_equality_reference_only": True,
        },
    }


def _run_export_preflight(*, dataset_dir, model_path, config_path,
                          tokenizer_id, tokenizer_revision):
    """Run real-tokenizer pair validation in a fresh Torch-isolated process."""
    with tempfile.TemporaryDirectory(prefix="umm-dpo-export-preflight-") as tmp:
        result_path = Path(tmp) / "preflight.json"
        command = [
            sys.executable, "-I", str(ROOT / "python/umminfer_joint_dpo_live.py"),
            "--token-export-preflight", "--dataset-dir", str(dataset_dir),
            "--model-path", str(model_path), "--config", str(config_path),
            "--tokenizer-id", tokenizer_id,
            "--tokenizer-revision", tokenizer_revision,
            "--output", str(result_path),
        ]
        proc = subprocess.run(command, capture_output=True, text=True, timeout=1800)
        if proc.returncode or not result_path.is_file():
            detail = proc.stderr.strip() or proc.stdout.strip() or "no diagnostic"
            raise ValueError(f"DPO token-export real-tokenizer preflight failed: {detail}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "PASS" or not result.get("packed"):
            raise ValueError("DPO token-export preflight returned no validated pairs")
        return result


def _sequence_logprob(torch, model, item, device):
    """Return weighted-mean target logp, weighted sum, and target mass."""
    input_ids = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
    labels = torch.tensor([item["labels"]], dtype=torch.long, device=device)
    weights = torch.tensor([item["weights"]], dtype=torch.float32, device=device)
    attention = torch.ones_like(input_ids)
    logits = model(input_ids=input_ids, attention_mask=attention).logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    shift_weights = weights[:, 1:]
    safe_labels = shift_labels.clamp_min(0)
    token_logps = torch.nn.functional.log_softmax(
        shift_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    normalizer = shift_weights.sum()
    if not torch.isfinite(normalizer).item() or normalizer.item() <= 0:
        raise RuntimeError("DPO target mask has no positive shifted target mass")
    weighted_sum = (token_logps * shift_weights).sum()
    mean_logp = weighted_sum / normalizer
    if not torch.isfinite(weighted_sum).item() or not torch.isfinite(mean_logp).item():
        raise RuntimeError("non-finite DPO policy/reference sequence log-probability")
    return mean_logp, weighted_sum, normalizer


def _measured_logprob(torch, model, item, device):
    with torch.no_grad():
        mean_logp, weighted_sum, normalizer = _sequence_logprob(
            torch, model, item, device)
    return {
        "mean_logp": float(mean_logp.item()),
        "weighted_logp_sum": float(weighted_sum.item()),
        "active_loss_weight": float(normalizer.item()),
    }


def _stdlib_sequence_logprob(item, token_logps):
    """Exact CPU reference for the adapter's causal shifted reference mask."""
    ids = item["input_ids"]
    labels = item["labels"]
    weights = item["weights"]
    if not (len(ids) == len(labels) == len(weights) == len(token_logps)) \
            or len(ids) < 2:
        raise ValueError("DPO CPU logp fixture arrays must have equal lengths >= 2")
    weighted_sum = 0.0
    normalizer = 0.0
    for index in range(1, len(ids)):
        label = labels[index]
        weight = weights[index]
        logp = token_logps[index]
        if not _is_int(label) or not _finite(weight) or not _finite(logp) \
                or float(weight) < 0.0 or float(logp) > 0.0:
            raise ValueError("invalid DPO CPU label, weight, or log-probability")
        if label == -100:
            if float(weight) != 0.0:
                raise ValueError("masked DPO reference token has non-zero weight")
            continue
        if not _is_int(ids[index]) or label != ids[index] or float(weight) <= 0.0:
            raise ValueError("supervised DPO reference token/label/mask is inconsistent")
        weighted_sum += float(logp) * float(weight)
        normalizer += float(weight)
    if normalizer <= 0.0:
        raise ValueError("DPO CPU reference mask has no positive shifted mass")
    return {
        "mean_logp": weighted_sum / normalizer,
        "weighted_logp_sum": weighted_sum,
        "active_loss_weight": normalizer,
    }


def _stdlib_dpo_fixture(pair, *, policy_logps, reference_logps, beta):
    """Compute an exact sigmoid-DPO fixture without importing an ML runtime."""
    if not _finite(beta) or float(beta) <= 0.0:
        raise ValueError("DPO fixture beta must be finite and positive")
    values = {}
    for family, logps in (("policy", policy_logps), ("reference", reference_logps)):
        if set(logps) != {"chosen", "rejected"}:
            raise ValueError("DPO fixture requires independent chosen/rejected logps")
        values[family] = {
            side: _stdlib_sequence_logprob(pair[side], logps[side])
            for side in ("chosen", "rejected")
        }
    policy_logratio = (values["policy"]["chosen"]["mean_logp"]
                       - values["policy"]["rejected"]["mean_logp"])
    reference_logratio = (values["reference"]["chosen"]["mean_logp"]
                          - values["reference"]["rejected"]["mean_logp"])
    scaled_margin = float(beta) * (policy_logratio - reference_logratio)
    return {
        **values,
        "policy_logratio": policy_logratio,
        "reference_logratio": reference_logratio,
        "scaled_margin": scaled_margin,
        "loss": stage_c._log_sigmoid_neg(scaled_margin),
        "real_optimizer": "NOT_RUN",
    }


def _export_bundle(prepared):
    """Create the deterministic CPU artifact consumed by independent verification."""
    preflight = prepared.get("token_export_preflight")
    if prepared.get("token_source") != "token_export" or preflight is None:
        raise ValueError("DPO preflight bundle requires --token-source token_export")
    bundle = copy.deepcopy(preflight)
    reference_sha = stage_c.sha256_bytes(stage_c.canonical_json(
        prepared["reference_policy_inputs"]))
    bundle.update({
        "run_identity": prepared["identity"],
        "run_identity_sha256": prepared["identity_sha256"],
        "reference_policy_inputs": prepared["reference_policy_inputs"],
        "reference_policy_inputs_sha256": reference_sha,
        "schedule": {
            "schedule_sha256": prepared["plan"]["batch_schedules"]["dpo"]
                ["schedule_sha256"],
            "max_steps": prepared["max_steps"],
            "stop_after": prepared["stop_after"],
            "checkpoint_every": prepared["checkpoint_every"],
            "scheduled_pair_ids": [row["pair_id"] for row in prepared["rows"]],
            "pair_order_sha256": stage_c.sha256_bytes(stage_c.canonical_json([
                row["pair_id"] for row in prepared["rows"]
            ])),
        },
    })
    return bundle


def _write_export_bundle(prepared):
    path = prepared["output_dir"] / "token_export_preflight.json"
    sft_live._atomic_json(path, _export_bundle(prepared))
    return path


def _validate_resume_payload(prepared, payload):
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported resume checkpoint format")
    if payload.get("run_identity_sha256") != prepared["identity_sha256"]:
        raise ValueError("resume checkpoint source/config/data/model/order identity mismatch")
    next_step = payload.get("next_step")
    if not _is_int(next_step) or not 0 < next_step < prepared["stop_after"]:
        raise ValueError("resume next_step must leave at least one bounded step")
    if payload.get("schedule_position") != next_step:
        raise ValueError("resume schedule position mismatch")
    if prepared["token_source"] == "token_export":
        expected = stage_c.sha256_bytes(stage_c.canonical_json(
            prepared["reference_policy_inputs"]))
        if payload.get("reference_policy_inputs_sha256") != expected:
            raise ValueError("resume checkpoint reference-policy input identity mismatch")
    return next_step


def execute_live_run(prepared):
    """Execute measured real sigmoid-DPO AdamW steps on one visible GPU."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit() or "," in visible:
        raise ValueError("live gate requires CUDA_VISIBLE_DEVICES=<one numeric idle GPU>")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    print("DPO_ML_IMPORT_BEGIN", file=sys.stderr, flush=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("live gate requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
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
        next_step = _validate_resume_payload(prepared, resume_payload)

    output = prepared["output_dir"]
    export_preflight_sha = None
    if prepared["token_source"] == "token_export":
        packed = prepared["export_packed"]
        if packed is None or len(packed) != len(prepared["rows"]):
            raise ValueError("validated export-driven packed DPO pairs are unavailable")
        export_preflight_sha = stage_c.sha256_file(_write_export_bundle(prepared))
    else:
        # Default legacy path: retain the accepted full prompt+target re-encoding.
        tokenizer = AutoTokenizer.from_pretrained(
            prepared["model_path"], local_files_only=True, use_fast=True,
            trust_remote_code=False)
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("live gate requires a fast tokenizer before model loading")
        packed = []
        for row, pair_audit in zip(prepared["rows"], prepared["pair_audits"]):
            chosen = _pack_response(
                tokenizer, row, row["chosen"], pair_audit["chosen"],
                side="chosen", prepared=prepared)
            rejected = _pack_response(
                tokenizer, row, row["rejected"], pair_audit["rejected"],
                side="rejected", prepared=prepared)
            if chosen["prompt_token_ids"] != rejected["prompt_token_ids"]:
                raise ValueError("chosen/rejected prompt tokenization differs")
            packed.append({
                "pair_id": row["pair_id"],
                "chosen": chosen,
                "rejected": rejected,
            })

    policy = AutoModelForCausalLM.from_pretrained(
        prepared["model_path"], local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="eager")
    reference = AutoModelForCausalLM.from_pretrained(
        prepared["model_path"], local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="eager")
    policy.config.use_cache = False
    reference.config.use_cache = False
    policy.to(device)
    reference.to(device)
    policy.train()
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)
    reference_sha = sft_live._model_fingerprint(torch, reference)
    if resume_payload is not None \
            and resume_payload.get("reference_parameter_sha256") != reference_sha:
        raise ValueError("resume checkpoint reference-policy fingerprint mismatch")
    reference_values = []
    for item in packed:
        reference_values.append({
            side: _measured_logprob(torch, reference, item[side], device)
            for side in ("chosen", "rejected")
        })
    if any(param.grad is not None for param in reference.parameters()):
        raise RuntimeError("frozen DPO reference unexpectedly received gradients")

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(prepared["identity"]["optimization"]["learning_rate"]),
        weight_decay=0.0, foreach=False)
    initial_parameter_sha = sft_live._model_fingerprint(torch, policy)
    if next_step == 0 and initial_parameter_sha != reference_sha:
        raise RuntimeError("initial DPO policy and reference weights differ")
    if resume_payload is not None:
        policy.load_state_dict(resume_payload["model_state"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        torch.set_rng_state(resume_payload["cpu_rng_state"])
        torch.cuda.set_rng_state_all(resume_payload["cuda_rng_state"])
        del resume_payload
        initial_parameter_sha = sft_live._model_fingerprint(torch, policy)

    if not all(torch.isfinite(p).all().item() for p in policy.parameters()):
        raise RuntimeError("policy parameters are non-finite before optimizer execution")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    history = []
    checkpoints = []
    output = prepared["output_dir"]
    beta = float(prepared["identity"]["dpo"]["beta"])

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
            "reference_parameter_sha256": reference_sha,
        }
        if prepared["token_source"] == "token_export":
            payload["reference_policy_inputs_sha256"] = stage_c.sha256_bytes(
                stage_c.canonical_json(prepared["reference_policy_inputs"]))
        tmp = path.with_name(path.name + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        digest = stage_c.sha256_file(path)
        sft_live._atomic_bytes(
            Path(str(path) + ".sha256"), (digest + "\n").encode("ascii"))
        record = {"next_step": completed_steps, "path": path.name,
                  "sha256": digest, "bytes": path.stat().st_size}
        checkpoints.append(record)
        return record

    for step_index in range(next_step, prepared["stop_after"]):
        item = packed[step_index]
        torch.cuda.reset_peak_memory_stats(device)
        step_started = time.perf_counter()
        before = {
            name: param.detach().cpu().clone()
            for name, param in policy.named_parameters()
        }
        before_sha = sft_live._model_fingerprint(torch, policy)
        optimizer.zero_grad(set_to_none=True)
        policy_values = {}
        weighted_sums = {}
        normalizers = {}
        for side in ("chosen", "rejected"):
            mean_logp, weighted_sum, normalizer = _sequence_logprob(
                torch, policy, item[side], device)
            policy_values[side] = mean_logp
            weighted_sums[side] = weighted_sum
            normalizers[side] = normalizer
        ref_chosen = reference_values[step_index]["chosen"]["mean_logp"]
        ref_rejected = reference_values[step_index]["rejected"]["mean_logp"]
        policy_logratio = policy_values["chosen"] - policy_values["rejected"]
        reference_logratio = ref_chosen - ref_rejected
        implicit_margin = policy_logratio - reference_logratio
        scaled_margin = beta * implicit_margin
        loss = torch.nn.functional.softplus(-scaled_margin)
        if not torch.isfinite(loss).item():
            raise RuntimeError("non-finite measured DPO loss")
        pre_metrics = {
            side: {
                "mean_logp": float(policy_values[side].item()),
                "weighted_logp_sum": float(weighted_sums[side].item()),
                "active_loss_weight": float(normalizers[side].item()),
            }
            for side in ("chosen", "rejected")
        }
        pre_scaled_margin = float(scaled_margin.item())
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            float(prepared["identity"]["optimization"]["max_grad_norm"]),
            error_if_nonfinite=True)
        grad_tensors = [p.grad for p in policy.parameters() if p.grad is not None]
        if not grad_tensors or not any(torch.count_nonzero(g).item() > 0
                                      for g in grad_tensors):
            raise RuntimeError("real DPO step produced no non-zero gradient")
        optimizer.step()
        torch.cuda.synchronize(device)

        post_metrics = {
            side: _measured_logprob(torch, policy, item[side], device)
            for side in ("chosen", "rejected")
        }
        post_implicit_margin = (
            (post_metrics["chosen"]["mean_logp"] - ref_chosen)
            - (post_metrics["rejected"]["mean_logp"] - ref_rejected))
        post_scaled_margin = beta * post_implicit_margin
        post_loss = stage_c._log_sigmoid_neg(post_scaled_margin)
        step_elapsed = time.perf_counter() - step_started

        changed_tensors = 0
        changed_parameters = 0
        max_delta = 0.0
        delta_l2_squared = 0.0
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
        if sft_live._model_fingerprint(torch, reference) != reference_sha:
            raise RuntimeError("DPO reference weights changed during optimizer execution")
        optimizer_sha = sft_live._optimizer_fingerprint(torch, policy, optimizer)
        entry = {
            "step": step_index + 1,
            "schedule_step_index": step_index,
            "record_ids": [item["pair_id"]],
            "record_order_sha256": stage_c.sha256_bytes(
                stage_c.canonical_json([item["pair_id"]])),
            "loss": float(loss.item()),
            "beta": beta,
            "reference_policy": "frozen_initial_base_model",
            "reference_parameter_sha256": reference_sha,
            "policy_reference_logprob_evidence": {
                "pre_update": {
                    "policy": pre_metrics,
                    "reference": reference_values[step_index],
                    "policy_logratio": float(policy_logratio.item()),
                    "reference_logratio": float(reference_logratio),
                    "implicit_reward_margin": float(implicit_margin.item()),
                    "scaled_margin": pre_scaled_margin,
                },
                "post_update": {
                    "policy": post_metrics,
                    "reference": reference_values[step_index],
                    "implicit_reward_margin": post_implicit_margin,
                    "scaled_margin": post_scaled_margin,
                    "loss": post_loss,
                    "scaled_margin_delta": post_scaled_margin - pre_scaled_margin,
                },
            },
            "grad_norm": float(grad_norm),
            "nonzero_gradient_tensors": sum(
                torch.count_nonzero(g).item() > 0 for g in grad_tensors),
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
            "pair_audit": {
                "pair_id": item["pair_id"],
                "chosen": item["chosen"]["audit"],
                "rejected": item["rejected"]["audit"],
            },
        }
        history.append(entry)
        if (step_index + 1) % prepared["checkpoint_every"] == 0 \
                or step_index + 1 == prepared["stop_after"]:
            save_checkpoint(step_index + 1)

    torch.cuda.synchronize(device)
    if sft_live._model_fingerprint(torch, reference) != reference_sha:
        raise RuntimeError("final DPO reference fingerprint differs from initial base")
    result = {
        "schema": LIVE_MANIFEST_SCHEMA,
        "status": "PASS",
        "REAL_TRAINING": "EXECUTED",
        "optimizer_status": "EXECUTED",
        "algorithm": "joint-interleaved-span-weighted-sigmoid-dpo-adamw-v1",
        "claims": {
            "model_quality": "NOT_RUN",
            "generalized_improvement": "NOT_RUN",
            "GRPO": "NOT_RUN",
            "production_model_training": "NOT_RUN",
            "upstream_revision_verification": "NOT_RUN",
            "vision_or_media_forward": "NOT_RUN",
            "serving_process_started": False,
        },
        "source": prepared["source"],
        "config_file_sha256": prepared["config_file_sha256"],
        "resolved_config_sha256": prepared["plan"]["config_sha256"],
        "dataset": prepared["plan"]["dataset"],
        "model": prepared["model"],
        "seed": {"text": seed_text, "derived_uint31": seed},
        "batch_schedule": prepared["plan"]["batch_schedules"]["dpo"],
        "pair_order_sha256": stage_c.sha256_bytes(stage_c.canonical_json([
            row["pair_id"] for row in prepared["rows"]
        ])),
        "executed_order_sha256": stage_c.sha256_bytes(stage_c.canonical_json([
            row["pair_id"] for row in prepared["rows"][next_step:prepared["stop_after"]]
        ])),
        "run_identity_sha256": prepared["identity_sha256"],
        "resume": {
            "used": next_step > 0,
            "checkpoint_sha256": resume_sha256,
            "initial_step": next_step,
        },
        "stop_after": prepared["stop_after"],
        "optimizer_step_count": len(history),
        "steps": history,
        "initial_parameter_sha256": initial_parameter_sha,
        "reference_parameter_sha256": reference_sha,
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
    if export_preflight_sha is not None:
        result["claims"]["sampled_online_behavior_ids"] = "NOT_CLAIMED"
        result["token_source"] = {
            "mode": "token_export",
            "preflight_schema": prepared["token_export_preflight"]["schema"],
            "preflight_sha256": export_preflight_sha,
            "tokenizer": prepared["token_export_preflight"]["tokenizer"],
            "tokenization_version": prepared["token_export_preflight"]
                ["tokenization_version"],
            "pairs": prepared["token_export_preflight"]["pairs"],
            "reference_policy_inputs_sha256": stage_c.sha256_bytes(
                stage_c.canonical_json(prepared["reference_policy_inputs"])),
            "target_retokenization_for_training": False,
            "legacy_equality_reference_only": True,
        }
    sft_live._atomic_json(output / "live_manifest.json", result)
    return result


def _raw_export_options(config_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    span_weights = config.get("span_loss_weights")
    dpo = config.get("dpo")
    if not isinstance(span_weights, dict) or not isinstance(dpo, dict):
        raise ValueError("DPO export preflight requires span_loss_weights and dpo config")
    return span_weights, dpo


def _torch_isolated_tokenizer(model_path):
    """Load Transformers with both Torch discovery and actual imports forbidden."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    real_import = builtins.__import__
    real_find_spec = importlib.util.find_spec

    def torch_unavailable(name, *find_args, **find_kwargs):
        if name == "torch" or name.startswith("torch."):
            return None
        return real_find_spec(name, *find_args, **find_kwargs)

    def no_torch(name, *import_args, **import_kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch import is forbidden in DPO token-export preflight")
        return real_import(name, *import_args, **import_kwargs)

    builtins.__import__ = no_torch
    importlib.util.find_spec = torch_unavailable
    try:
        import transformers
        import tokenizers
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, use_fast=True,
            trust_remote_code=False)
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("DPO token-export preflight requires a fast tokenizer")
        version = (f"transformers-{transformers.__version__}_"
                   f"tokenizers-{tokenizers.__version__}")
        return tokenizer, version, real_import, real_find_spec
    except Exception:
        builtins.__import__ = real_import
        importlib.util.find_spec = real_find_spec
        raise


def _token_export_preflight_main(argv):
    """Internal fresh-process real-tokenizer pair preflight; never imports Torch."""
    parser = argparse.ArgumentParser(description="Validate and pack DPO token exports")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    real_import = builtins.__import__
    real_find_spec = importlib.util.find_spec
    try:
        tokenizer, version, real_import, real_find_spec = \
            _torch_isolated_tokenizer(args.model_path)
        dataset = stage_c.verify_and_load_dataset_pack(args.dataset_dir)
        rows = dataset["packs"].get("dpo_train") or []
        span_weights, dpo = _raw_export_options(args.config)
        result = _pack_export_pairs(
            rows, tokenizer, tokenizer_dir=args.model_path,
            tokenizer_id=args.tokenizer_id,
            tokenizer_revision=args.tokenizer_revision,
            tokenization_version=version, span_loss_weights=span_weights,
            dpo_config=dpo)
        if "torch" in sys.modules:
            raise RuntimeError("torch was imported during DPO token-export preflight")
        sft_live._atomic_json(args.output, result)
    finally:
        builtins.__import__ = real_import
        importlib.util.find_spec = real_find_spec
    sys.stdout.write(stage_c.canonical_json({
        "status": "PASS", "pairs": len(result["pairs"]),
        "tokenization_version": version, "torch_imported": False,
        "real_dpo_optimizer": "NOT_RUN",
    }).decode("utf-8") + "\n")
    return 0


def _token_export_verify_main(argv):
    """Independently rebuild and byte-compare a complete preflight bundle."""
    parser = argparse.ArgumentParser(description="Verify a DPO token-export bundle")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    bundle_path = Path(args.bundle)
    sidecar = Path(str(bundle_path) + ".sha256")
    if not bundle_path.is_file() or not sidecar.is_file() \
            or sidecar.read_text(encoding="ascii").strip() != stage_c.sha256_file(bundle_path):
        raise ValueError("DPO token-export bundle or SHA-256 sidecar is invalid")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    identity = bundle.get("run_identity")
    schedule = bundle.get("schedule")
    if not isinstance(identity, dict) or not isinstance(schedule, dict):
        raise ValueError("DPO bundle lacks run identity or schedule")
    revision = identity["model"]["revision"]
    with tempfile.TemporaryDirectory(prefix="umm-dpo-export-verify-") as tmp:
        prepared = prepare_live_run(
            dataset_dir=args.dataset_dir, output_dir=Path(tmp) / "rebuilt",
            config_path=args.config, model_path=args.model_path,
            source_commit=identity["source"]["commit"],
            max_steps=identity["max_steps"], stop_after=schedule["stop_after"],
            checkpoint_every=schedule["checkpoint_every"],
            model_revision=None if revision == "UNKNOWN" else revision,
            token_source="token_export")
        expected = _export_bundle(prepared)
    if stage_c.canonical_json(bundle) != stage_c.canonical_json(expected):
        raise ValueError("independent DPO token-export bundle reconstruction differs")
    if "torch" in sys.modules:
        raise RuntimeError("torch was imported during independent DPO bundle verification")
    result = {
        "schema": "umminfer-joint-dpo-token-export-verification-v1",
        "status": "PASS",
        "bundle_sha256": stage_c.sha256_file(bundle_path),
        "run_identity_sha256": prepared["identity_sha256"],
        "pair_count": len(bundle["pairs"]),
        "scheduled_pair_ids": schedule["scheduled_pair_ids"],
        "tokenizer_reencoded": True,
        "fresh_process_reconstruction": "BYTE_IDENTICAL",
        "torch_imported": False,
        "real_dpo_optimizer": "NOT_RUN",
        "model_quality": "NOT_RUN",
        "sampled_online_behavior_ids": "NOT_CLAIMED",
    }
    sft_live._atomic_json(args.output, result)
    sys.stdout.write(stage_c.canonical_json(result).decode("utf-8") + "\n")
    return 0


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Validate exported DPO data or execute the bounded optimizer gate")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--token-source", choices=("legacy", "token_export"),
                        default="legacy",
                        help="token_export requires both exports on every DPO pair")
    parser.add_argument("--preflight-only", action="store_true",
                        help="write and verify CPU data identity; never import Torch or run a model")
    return parser


def main(argv=None):
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    if actual_argv and actual_argv[0] == "--token-export-preflight":
        return _token_export_preflight_main(actual_argv[1:])
    if actual_argv and actual_argv[0] == "--token-export-verify":
        return _token_export_verify_main(actual_argv[1:])
    parser = _build_parser()
    args = parser.parse_args(actual_argv)
    try:
        if args.preflight_only and (args.token_source != "token_export"
                                    or args.resume_from is not None):
            raise ValueError("--preflight-only requires token_export and no resume checkpoint")
        prepared = prepare_live_run(
            dataset_dir=args.dataset_dir, output_dir=args.output_dir,
            config_path=args.config, model_path=args.model_path,
            source_commit=args.source_commit, max_steps=args.max_steps,
            checkpoint_every=args.checkpoint_every, stop_after=args.stop_after,
            resume_from=args.resume_from, model_revision=args.model_revision,
            token_source=args.token_source)
        if args.preflight_only:
            bundle_path = _write_export_bundle(prepared)
            result = {
                "schema": "umminfer-joint-dpo-cpu-preflight-result-v1",
                "status": "PASS",
                "REAL_TRAINING": "NOT_RUN",
                "optimizer": "NOT_RUN",
                "model_inference": "NOT_RUN",
                "model_quality": "NOT_RUN",
                "sampled_online_behavior_ids": "NOT_CLAIMED",
                "preflight_file": bundle_path.name,
                "preflight_sha256": stage_c.sha256_file(bundle_path),
                "run_identity_sha256": prepared["identity_sha256"],
                "pair_count": len(prepared["token_export_preflight"]["pairs"]),
                "scheduled_pair_ids": [row["pair_id"] for row in prepared["rows"]],
                "torch_imported": False,
            }
        else:
            result = execute_live_run(prepared)
    except (OSError, ValueError, RuntimeError, ImportError,
            json.JSONDecodeError) as exc:
        parser.exit(2, f"joint DPO live gate error: {exc}\n")
    sys.stdout.write(stage_c.canonical_json(result).decode("utf-8") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
