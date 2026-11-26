# SPDX-License-Identifier: Apache-2.0
"""Bounded real optimizer acceptance adapter for the existing Stage C SFT planner.

The stdlib-only preflight delegates dataset verification, ablation resolution,
span weights, and batch ordering to ``umminfer_joint_trainer``.  ML framework
imports happen only in :func:`execute_live_run`, after all model-independent
inputs have passed validation.  The executed objective is full-parameter causal
SFT on the literal prompt plus reconstructed interleaved assistant target.

This is an architecture/optimizer gate, not a model-quality experiment.  It does
not run DPO, GRPO, a vision encoder, media generation, or a serving process.
"""

import argparse
import builtins
import copy
import hashlib
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

# Direct script execution uses -I, so add only this repository's known paths.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

import umminfer_joint_trainer as stage_c  # noqa: E402
import umminfer_sft_dataset as sft_dataset  # noqa: E402
import umminfer_token_export as token_export  # noqa: E402

CHECKPOINT_FORMAT = "umminfer-joint-sft-live-checkpoint-v1"
LIVE_MANIFEST_SCHEMA = "umminfer-joint-sft-live-result-v1"
SOURCE_FILES = (
    "python/umminfer_joint_sft_live.py",
    "python/umminfer_joint_trainer.py",
    "python/umminfer_rl_trajectory.py",
)
_HEX_40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(float(value)))


def _atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def _atomic_json(path, value):
    payload = stage_c.canonical_json(value) + b"\n"
    _atomic_bytes(path, payload)
    _atomic_bytes(Path(str(path) + ".sha256"),
                  (stage_c.sha256_bytes(payload) + "\n").encode("ascii"))


def _fingerprint_model(model_path):
    root = Path(model_path)
    if not root.is_dir():
        raise ValueError("live gate requires an existing local model directory")
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ValueError("model directory is missing config.json")
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    if model_config.get("model_type") != "qwen2":
        raise ValueError("bounded live gate supports only a local Qwen2 causal LM")
    architectures = model_config.get("architectures")
    if architectures != ["Qwen2ForCausalLM"]:
        raise ValueError("model config must declare exactly Qwen2ForCausalLM")

    files = {}
    weight_files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".cache" in path.relative_to(root).parts:
            continue
        rel = path.relative_to(root).as_posix()
        digest = stage_c.sha256_file(path)
        entry = {"sha256": digest, "bytes": path.stat().st_size}
        files[rel] = entry
        if path.suffix == ".safetensors":
            weight_files[rel] = entry
    if not weight_files:
        raise ValueError("model directory contains no Safetensors weights")
    return {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "UNKNOWN",
        "upstream_revision_verification": "NOT_RUN",
        "model_type": model_config["model_type"],
        "config_sha256": files["config.json"]["sha256"],
        "files": files,
        "available_weight_files": weight_files,
        "available_weights_sha256": stage_c.sha256_bytes(
            stage_c.canonical_json(weight_files)),
    }


def _audit_sft_record(row):
    """Validate target, structured slots, and completed terminal-leg bindings."""
    if not isinstance(row, dict) or row.get("record_type") != "sft_interleaved":
        raise ValueError("live gate accepts only sft_interleaved records")
    if row.get("schema_version") != 1:
        raise ValueError("unsupported Stage C SFT record schema")
    provenance = row.get("provenance")
    if not isinstance(provenance, dict) \
            or provenance.get("source_schema_version") != 2:
        raise ValueError("live gate requires schema-v2 trajectory provenance")
    prompt = row.get("prompt_text")
    target = row.get("assistant_target")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("SFT prompt_text must be a non-empty string")
    if not isinstance(target, str) or not target:
        raise ValueError("SFT assistant_target must be a non-empty string")
    if row.get("target_sha256") != stage_c.sha256_text(target):
        raise ValueError("assistant_target SHA-256 mismatch")
    clean_text = row.get("clean_text")
    if not isinstance(clean_text, str) \
            or row.get("clean_text_sha256") != stage_c.sha256_text(clean_text):
        raise ValueError("clean_text or clean_text_sha256 is invalid")
    if row.get("messages") != [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target}]:
        raise ValueError("SFT messages differ from prompt/assistant targets")

    slots = row.get("visual_slots")
    if not isinstance(slots, list):
        raise ValueError("visual_slots must be a list")
    terminal_count = 0
    for slot in slots:
        if not isinstance(slot, dict) \
                or slot.get("media_type") not in ("image", "video"):
            raise ValueError("visual slot has an invalid modality")
        if slot.get("leg_state") != "completed":
            raise ValueError("every trained visual slot must have a completed terminal leg")
        marker = slot.get("marker")
        if not isinstance(marker, str) or marker not in target:
            raise ValueError("visual slot marker is absent from assistant_target")
        if not _HEX_64.fullmatch(str(slot.get("artifact_sha256", ""))):
            raise ValueError("completed visual slot lacks an artifact SHA-256")
        terminal_count += 1
    if bool(slots) == bool(row.get("is_text_only")):
        raise ValueError("is_text_only disagrees with structured visual slots")
    for ref in row.get("output_media_refs") or []:
        if ref.get("verified_on_disk") is not True:
            raise ValueError("live gate requires on-disk-verified output media refs")

    mask = stage_c.build_span_loss_mask(
        target, ablation_profile="joint_full",
        span_weights=stage_c.DEFAULT_SPAN_WEIGHTS)
    if not mask["active_tokens"] or mask["masked_tokens"]:
        raise ValueError("joint_full target must have only active, non-empty loss spans")
    if slots and not (mask["span_counts"]["visual_route_tag"]
                      and mask["span_counts"]["visual_prompt_body"]):
        raise ValueError("structured slots lack route-tag or prompt-body supervision")
    return {
        "example_id": row.get("example_id"),
        "target_sha256": row["target_sha256"],
        "text_only": not slots,
        "visual_modalities": sorted({s["media_type"] for s in slots}),
        "terminal_legs_verified": terminal_count,
        "span_counts": mask["span_counts"],
        "stdlib_mask_tokens": mask["total_tokens"],
    }


def _sft_export_record(row):
    """Rebuild and structurally verify one additive SFT token-export binding."""
    has_block = "token_export" in row
    has_source = "token_export_source" in row
    if not has_block and not has_source:
        return None
    if not has_block or not has_source:
        raise ValueError("SFT token_export and token_export_source must appear together")
    block = row["token_export"]
    source = row["token_export_source"]
    if not isinstance(block, dict) or not isinstance(source, dict) \
            or "token_export" in source:
        raise ValueError("SFT token export/source binding is malformed")
    record = copy.deepcopy(source)
    record["token_export"] = copy.deepcopy(block)
    try:
        token_export.validate_record_token_export(
            record, expected_sft_target=row.get("assistant_target"))
    except token_export.TokenExportError as exc:
        raise ValueError(f"SFT token-export source is invalid: {exc}") from exc
    candidate = sft_dataset._extract_candidate(record, verified_on_disk=True)
    for key in (
            "prompt_sha256", "prompt_text", "prompt_parts", "input_media_refs",
            "target_sha256", "assistant_target", "clean_text", "clean_text_sha256",
            "visual_slots", "modality_route", "is_text_only", "output_media_refs",
            "provenance"):
        if stage_c.canonical_json(row.get(key)) != stage_c.canonical_json(candidate[key]):
            raise ValueError(f"SFT row differs from token-export source field: {key}")
    return record


def _structural_export_rows(rows):
    """Validate every requested row before tokenizer/ML imports or run outputs."""
    records = {}
    for row in rows:
        record = _sft_export_record(row)
        if record is None:
            raise ValueError(
                f"token_export was requested but SFT row {row.get('example_id')!r} "
                "has no export; refusing legacy fallback")
        example_id = row.get("example_id")
        if not isinstance(example_id, str) or not example_id \
                or example_id in records:
            raise ValueError("exported SFT example ids must be non-empty and unique")
        records[example_id] = record
    return records


def _token_loss_mask(offsets, *, target_start, target, token_ids, mask_info):
    """Map HF offset tokens to prompt-masked, span-weighted causal SFT labels."""
    if len(offsets) != len(token_ids):
        raise ValueError("tokenizer offsets and token ids differ in length")
    labels = []
    weights = []
    annotations = []
    spans = mask_info["spans"]
    for token_id, offset in zip(token_ids, offsets):
        if (not isinstance(offset, (list, tuple)) or len(offset) != 2
                or not all(_is_int(v) for v in offset)):
            raise ValueError("live gate requires a fast tokenizer with integer offsets")
        start, end = offset
        if start < 0 or end <= start:
            raise ValueError("tokenizer emitted an empty or invalid offset span")
        if start < target_start < end:
            raise ValueError("tokenizer crosses the prompt/target loss boundary")
        if end <= target_start:
            labels.append(-100)
            weights.append(0.0)
            annotations.append({
                "span_types": [], "weight": 0.0,
                "includes_terminal_delimiter": False,
            })
            continue

        local_start = start - target_start
        local_end = end - target_start
        if local_start < 0 or local_end > len(target):
            raise ValueError("tokenizer offset lies outside assistant_target")
        overlap_weight = 0.0
        overlap_chars = 0
        types = set()
        closing = False
        for span in spans:
            overlap = max(0, min(local_end, span["end"])
                          - max(local_start, span["start"]))
            if not overlap:
                continue
            overlap_chars += overlap
            overlap_weight += overlap * float(span["weight"])
            types.add(span["span_type"])
            if span["span_type"] == "visual_route_tag" and span["text"] == "]]":
                closing = True
        if overlap_chars != local_end - local_start or overlap_chars <= 0:
            raise ValueError("loss spans do not exactly cover a target token")
        weight = overlap_weight / overlap_chars
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("joint_full target token has no positive loss weight")
        labels.append(token_id)
        weights.append(weight)
        annotations.append({
            "span_types": sorted(types),
            "weight": weight,
            "includes_terminal_delimiter": closing,
        })
    if not any(label >= 0 for label in labels):
        raise ValueError("target has no supervised causal-LM tokens")
    return labels, weights, annotations


def _legacy_pack_row(row, tokenizer, *, span_loss_weights, max_seq_len):
    """Preserve the accepted full-text re-encoding path operation-for-operation."""
    prompt = row["prompt_text"]
    target = row["assistant_target"]
    full_text = prompt + "\n" + target
    target_start = len(prompt) + 1
    encoded = tokenizer(
        full_text, add_special_tokens=False, return_offsets_mapping=True,
        truncation=False)
    ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    if not ids or len(ids) > max_seq_len:
        raise ValueError("packed target is empty or exceeds max_seq_len")
    mask_info = stage_c.build_span_loss_mask(
        target, ablation_profile="joint_full", span_weights=span_loss_weights)
    labels, weights, annotations = _token_loss_mask(
        offsets, target_start=target_start, target=target,
        token_ids=ids, mask_info=mask_info)
    route_tokens = sum(
        "visual_route_tag" in item["span_types"] for item in annotations)
    prompt_body_tokens = sum(
        "visual_prompt_body" in item["span_types"] for item in annotations)
    closing_tokens = sum(
        item.get("includes_terminal_delimiter", False) for item in annotations)
    audit = _audit_sft_record(row)
    if audit["visual_modalities"] and not (
            route_tokens and prompt_body_tokens and closing_tokens):
        raise ValueError("HF token mask misses visual route, prompt, or terminal delimiter")
    return {
        "row": row,
        "input_ids": ids,
        "offsets": offsets,
        "labels": labels,
        "weights": weights,
        "annotations": annotations,
        "audit": {
            **audit,
            "sequence_tokens": len(ids),
            "supervised_target_tokens": sum(v > 0 for v in weights),
            "prompt_masked_tokens": sum(v == 0 for v in weights),
            "visual_route_tag_tokens": route_tokens,
            "visual_prompt_body_tokens": prompt_body_tokens,
            "terminal_delimiter_tokens": closing_tokens,
            "input_ids_sha256": stage_c.sha256_bytes(stage_c.canonical_json(ids)),
            "labels_sha256": stage_c.sha256_bytes(stage_c.canonical_json(labels)),
            "loss_weights_sha256": stage_c.sha256_bytes(
                stage_c.canonical_json(weights)),
        },
    }


def _export_raw_offsets(stream, *, target_start):
    offsets = []
    for token in stream["tokens"]:
        if "offset_char_start" in token or "offset_char_end" in token:
            start = token.get("offset_char_start")
            end = token.get("offset_char_end")
        else:
            start = token.get("char_start")
            end = token.get("char_end")
        offsets.append((start + target_start, end + target_start))
    return offsets


def _pack_export_rows(rows, tokenizer, *, tokenizer_dir, tokenizer_id,
                      tokenizer_revision, tokenization_version,
                      span_loss_weights, max_seq_len):
    """Strongly validate exports and pack their ids without target re-tokenizing.

    The legacy full-text encoding is computed only as an equality reference. The
    returned training tensors use exported target ids, raw offsets, labels,
    weights, and annotations; only the unexported prompt prefix is tokenized.
    """
    records = _structural_export_rows(rows)
    packed = []
    summaries = []
    identity = None
    for row in rows:
        record = records[row["example_id"]]
        block = record["token_export"]
        if identity is None:
            identity = copy.deepcopy(block["tokenizer"])
        elif stage_c.canonical_json(identity) != stage_c.canonical_json(block["tokenizer"]):
            raise ValueError("mixed token-export tokenizer identities are not deterministic")
        token_export.validate_record_token_export(
            record, tokenizer=tokenizer, tokenizer_dir=tokenizer_dir,
            expected_tokenizer_id=tokenizer_id,
            expected_tokenizer_revision=tokenizer_revision,
            expected_tokenization_version=tokenization_version,
            expected_sft_target=row["assistant_target"])
        legacy = _legacy_pack_row(
            row, tokenizer, span_loss_weights=span_loss_weights,
            max_seq_len=max_seq_len)
        prompt = row["prompt_text"]
        target = row["assistant_target"]
        prefix = tokenizer(
            prompt + "\n", add_special_tokens=False, return_offsets_mapping=True,
            truncation=False)
        prefix_ids = list(prefix["input_ids"])
        prefix_offsets = list(prefix["offset_mapping"])
        stream = block["streams"]["sft_target"]
        target_ids = list(stream["token_ids"])
        if not prefix_ids or not target_ids \
                or len(prefix_ids) + len(target_ids) != len(legacy["input_ids"]):
            raise ValueError("exported/prompt token counts do not match legacy encoding")
        if stage_c.canonical_json(prefix_ids) != stage_c.canonical_json(
                legacy["input_ids"][:len(prefix_ids)]):
            raise ValueError("prompt-prefix encoding changed at the export boundary")
        legacy_offsets = legacy["offsets"]
        if stage_c.canonical_json(prefix_offsets) != stage_c.canonical_json(
                legacy_offsets[:len(prefix_ids)]):
            raise ValueError("prompt-prefix offsets differ from legacy encoding")
        target_offsets = _export_raw_offsets(stream, target_start=len(prompt) + 1)
        if stage_c.canonical_json(target_offsets) != stage_c.canonical_json(
                legacy_offsets[len(prefix_ids):]):
            raise ValueError("exported raw offsets differ from legacy target offsets")
        mask_info = stage_c.build_span_loss_mask(
            target, ablation_profile="joint_full", span_weights=span_loss_weights)
        target_labels, target_weights, target_annotations = _token_loss_mask(
            target_offsets, target_start=len(prompt) + 1, target=target,
            token_ids=target_ids, mask_info=mask_info)
        if target_labels != target_ids \
                or target_weights != stream["sft_mask"]["token_weights"]:
            raise ValueError("exported labels or weights differ from the SFT mask")
        annotations_sha = stage_c.sha256_bytes(stage_c.canonical_json(target_annotations))
        if annotations_sha != stream["sft_mask"]["annotations_sha256"]:
            raise ValueError("exported annotations do not recompute from raw offsets")
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
                raise ValueError(f"export-driven {key} differ from legacy joint_full packing")
        item = copy.deepcopy(legacy)
        item["input_ids"] = ids
        item["labels"] = labels
        item["weights"] = weights
        item["annotations"] = annotations
        summary = {
            "example_id": row["example_id"],
            "format": block["format"],
            "format_version": block["format_version"],
            "export_sha256": block["export_sha256"],
            "source_record_sha256": block["source_record_sha256"],
            "input_ids_sha256": item["audit"]["input_ids_sha256"],
            "labels_sha256": item["audit"]["labels_sha256"],
            "loss_weights_sha256": item["audit"]["loss_weights_sha256"],
            "annotations_sha256": annotations_sha,
            "raw_offsets_sha256": stage_c.sha256_bytes(
                stage_c.canonical_json(target_offsets)),
            "prompt_tokens": len(prefix_ids),
            "target_tokens": len(target_ids),
            "tokenizer_reencoded": True,
            "legacy_equality": "PASS",
        }
        item["token_export"] = summary
        item["raw_target_offsets"] = target_offsets
        packed.append(item)
        summaries.append(summary)
    return {
        "schema": "umminfer-joint-sft-token-export-preflight-v1",
        "status": "PASS",
        "tokenizer": identity,
        "tokenization_version": tokenization_version,
        "records": summaries,
        "packed": packed,
        "claims": {
            "sampled_online_behavior_ids": "NOT_CLAIMED",
            "target_retokenization_for_training": False,
            "legacy_equality_reference_only": True,
        },
    }


def _run_export_preflight(*, dataset_dir, model_path, tokenizer_id,
                          tokenizer_revision, max_seq_len):
    """Run real-tokenizer validation in a fresh process with Torch import banned."""
    with tempfile.TemporaryDirectory(prefix="umm-sft-export-preflight-") as tmp:
        result_path = Path(tmp) / "preflight.json"
        command = [
            sys.executable, "-I", str(ROOT / "python/umminfer_joint_sft_live.py"),
            "--token-export-preflight", "--dataset-dir", str(dataset_dir),
            "--model-path", str(model_path), "--tokenizer-id", tokenizer_id,
            "--tokenizer-revision", tokenizer_revision,
            "--max-seq-len", str(max_seq_len), "--output", str(result_path),
        ]
        proc = subprocess.run(command, capture_output=True, text=True, timeout=1800)
        if proc.returncode or not result_path.is_file():
            detail = proc.stderr.strip() or proc.stdout.strip() or "no diagnostic"
            raise ValueError(f"token-export real-tokenizer preflight failed: {detail}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "PASS" or not result.get("packed"):
            raise ValueError("token-export preflight returned no validated packed rows")
        return result


def _source_fingerprints(source_commit, *, include_token_export=False):
    if not isinstance(source_commit, str) or not _HEX_40.fullmatch(source_commit):
        raise ValueError("--source-commit must be one lowercase 40-character commit")
    files = {}
    rels = list(SOURCE_FILES)
    if include_token_export:
        rels.extend((
            "python/umminfer_sft_dataset.py",
            "python/umminfer_token_export.py",
        ))
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


def prepare_live_run(*, dataset_dir, output_dir, config_path, model_path,
                     source_commit, max_steps, checkpoint_every=1,
                     stop_after=None, resume_from=None, model_revision=None,
                     token_source="legacy"):
    """Run every model-independent validation before importing torch/transformers."""
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
        # This read-only structural gate deliberately precedes planner/output
        # creation, model loading, and every ML-framework import.
        early_dataset = stage_c.verify_and_load_dataset_pack(dataset_dir)
        early_rows = early_dataset["packs"].get("sft_train") or []
        if not early_rows:
            raise ValueError("token_export mode requires non-empty sft_train rows")
        _structural_export_rows(early_rows)
        raw_config = json.loads(config_file.read_text(encoding="utf-8"))
        raw_max_seq_len = raw_config.get("optimization", {}).get("max_seq_len")
        if not _is_int(raw_max_seq_len) or raw_max_seq_len <= 1:
            raise ValueError("export preflight requires a valid optimization.max_seq_len")
        model = _fingerprint_model(model_path)
        if model_revision is not None:
            if not isinstance(model_revision, str) or not model_revision:
                raise ValueError("model revision override must be non-empty")
            model["revision"] = model_revision
            model["upstream_revision_verification"] = "NOT_RUN"
        export_preflight = _run_export_preflight(
            dataset_dir=dataset_dir, model_path=model_path,
            tokenizer_id=model["model_id"], tokenizer_revision=model["revision"],
            max_seq_len=raw_max_seq_len)
    plan = stage_c.prepare_training_run(
        mode="sft", ablation="joint_full", dataset_dir=dataset_dir,
        output_dir=output / "plan", config_path=config_file,
        gpu_launch_script=True)
    if plan["REAL_TRAINING"] != "NOT_RUN" or plan["active_stages"] != ["sft"]:
        raise ValueError("Stage C planner did not produce an SFT-only NOT_RUN plan")

    hyper = plan["hyperparameters"]
    model_cfg = hyper["model"]
    peft_cfg = hyper["peft"]
    opt = hyper["optimization"]
    expected = {
        "dtype": "bfloat16",
        "attn_implementation": "eager",
        "gradient_checkpointing": False,
        "trust_remote_code": False,
    }
    if any(model_cfg.get(key) != value for key, value in expected.items()):
        raise ValueError("live gate requires bf16 eager local Qwen2 without remote code")
    if peft_cfg.get("method") != "none":
        raise ValueError("bounded gate supports full-parameter SFT only")
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
    if export_preflight is not None:
        if stage_c.canonical_json(plan["ablation"]["span_weights"]) \
                != stage_c.canonical_json(stage_c.DEFAULT_SPAN_WEIGHTS):
            raise ValueError("token_export v1/v2 is bound to the joint_full default weights")

    dataset = stage_c.verify_and_load_dataset_pack(dataset_dir)
    rows = dataset["packs"].get("sft_train") or []
    audits = [_audit_sft_record(row) for row in rows]
    schedule = plan["batch_schedules"]["sft"]
    if max_steps > schedule["total_batches"]:
        raise ValueError("--max-steps exceeds the deterministic Stage C schedule")
    scheduled_ids = [
        record_id
        for batch in schedule["batches"][:max_steps]
        for record_id in batch["record_ids"]
    ]
    by_id = {row["example_id"]: row for row in rows}
    if len(by_id) != len(rows) or any(item not in by_id for item in scheduled_ids):
        raise ValueError("scheduled example ids are absent or non-unique")
    scheduled_rows = [by_id[item] for item in scheduled_ids]
    export_packed = None
    if export_preflight is not None:
        packed_by_id = {item["row"]["example_id"]: item
                        for item in export_preflight["packed"]}
        if len(packed_by_id) != len(rows) or set(packed_by_id) != set(by_id):
            raise ValueError("token-export preflight rows differ from the frozen SFT train pack")
        export_packed = [packed_by_id[item] for item in scheduled_ids]
    scheduled_audits = [_audit_sft_record(row) for row in scheduled_rows]
    if not any(a["text_only"] for a in scheduled_audits) \
            or not any(a["visual_modalities"] for a in scheduled_audits):
        raise ValueError("scheduled gate steps must cover text-only and structured visual targets")
    if not any(a["terminal_legs_verified"] for a in scheduled_audits):
        raise ValueError("scheduled gate steps must verify completed visual terminal legs")

    if model is None:
        model = _fingerprint_model(model_path)
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
        "span_loss_weights": plan["ablation"]["span_weights"],
        "max_steps": max_steps,
        "objective": "full-parameter-span-weighted-causal-sft-v1",
    }
    if export_preflight is not None:
        identity["token_source"] = {
            "mode": "token_export",
            "schema": export_preflight["schema"],
            "tokenizer": export_preflight["tokenizer"],
            "tokenization_version": export_preflight["tokenization_version"],
            "records": export_preflight["records"],
            "claims": export_preflight["claims"],
        }
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
        "row_audits": scheduled_audits,
        "all_row_audits": audits,
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


def _tensor_digest(torch, tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(stage_c.canonical_json(list(value.shape)))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _model_fingerprint(torch, model):
    digest = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_digest(torch, param).encode("ascii"))
    return digest.hexdigest()


def _gradient_fingerprint(torch, model):
    digest = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        if param.grad is None:
            digest.update(b"NO_GRADIENT")
        else:
            digest.update(_tensor_digest(torch, param.grad).encode("ascii"))
    return digest.hexdigest()


def _optimizer_fingerprint(torch, model, optimizer):
    digest = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        state = optimizer.state.get(param, {})
        for key in sorted(state):
            value = state[key]
            digest.update(key.encode("utf-8"))
            if torch.is_tensor(value):
                digest.update(_tensor_digest(torch, value).encode("ascii"))
            else:
                digest.update(stage_c.canonical_json(value))
    for group in optimizer.param_groups:
        digest.update(stage_c.canonical_json({
            key: value for key, value in group.items() if key != "params"
        }))
    return digest.hexdigest()


def execute_live_run(prepared):
    """Tokenize, fail closed, then execute measured real AdamW steps on one GPU."""
    # Required by deterministic CUDA BLAS before torch initializes its handles.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import torch
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("live gate requires CUDA; no optimizer was executed")
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
        if not isinstance(resume_payload, dict) \
                or resume_payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("unsupported resume checkpoint format")
        if resume_payload.get("run_identity_sha256") != prepared["identity_sha256"]:
            raise ValueError("resume checkpoint source/config/data/model/order identity mismatch")
        next_step = resume_payload.get("next_step")
        if not _is_int(next_step) or not 0 < next_step < prepared["stop_after"]:
            raise ValueError("resume next_step must leave at least one bounded step")
        if resume_payload.get("schedule_position") != next_step:
            raise ValueError("resume schedule position mismatch")

    output = prepared["output_dir"]
    export_preflight_sha = None
    if prepared["token_source"] == "token_export":
        packed = prepared["export_packed"]
        if packed is None or len(packed) != len(prepared["rows"]):
            raise ValueError("validated export-driven packed rows are unavailable")
        preflight_path = output / "token_export_preflight.json"
        _atomic_json(preflight_path, prepared["token_export_preflight"])
        export_preflight_sha = stage_c.sha256_file(preflight_path)
    else:
        # Default legacy path: retain the accepted full prompt+target re-encoding.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            prepared["model_path"], local_files_only=True, use_fast=True,
            trust_remote_code=False)
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("live gate requires a fast tokenizer before model loading")
        packed = [
            _legacy_pack_row(
                row, tokenizer,
                span_loss_weights=prepared["identity"]["span_loss_weights"],
                max_seq_len=prepared["identity"]["optimization"]["max_seq_len"])
            for row in prepared["rows"]
        ]

    model = AutoModelForCausalLM.from_pretrained(
        prepared["model_path"], local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="eager")
    model.config.use_cache = False
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(prepared["identity"]["optimization"]["learning_rate"]),
        weight_decay=0.0, foreach=False)
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        torch.set_rng_state(resume_payload["cpu_rng_state"])
        torch.cuda.set_rng_state_all(resume_payload["cuda_rng_state"])
        del resume_payload

    if not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise RuntimeError("model parameters are non-finite before optimizer execution")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    history = []
    checkpoints = []

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
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
        }
        tmp = path.with_name(path.name + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        digest = stage_c.sha256_file(path)
        _atomic_bytes(Path(str(path) + ".sha256"), (digest + "\n").encode("ascii"))
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
            for name, param in model.named_parameters()
        }
        before_sha = _model_fingerprint(torch, model)
        optimizer.zero_grad(set_to_none=True)
        input_ids = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
        labels = torch.tensor([item["labels"]], dtype=torch.long, device=device)
        weights = torch.tensor(
            [item["weights"]], dtype=torch.float32, device=device)
        attention = torch.ones_like(input_ids)
        logits = model(input_ids=input_ids, attention_mask=attention).logits
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        shift_weights = weights[:, 1:]
        safe_labels = shift_labels.clamp_min(0)
        token_losses = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            safe_labels.reshape(-1), reduction="none").reshape(shift_labels.shape)
        normalizer = shift_weights.sum()
        if not torch.isfinite(normalizer).item() or normalizer.item() <= 0:
            raise RuntimeError("causal loss mask has no positive shifted target mass")
        loss = (token_losses * shift_weights).sum() / normalizer
        if not torch.isfinite(loss).item():
            raise RuntimeError("non-finite measured SFT loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(prepared["identity"]["optimization"]["max_grad_norm"]),
            error_if_nonfinite=True)
        grad_tensors = [p.grad for p in model.parameters() if p.grad is not None]
        if not grad_tensors or not any(torch.count_nonzero(g).item() > 0
                                      for g in grad_tensors):
            raise RuntimeError("real SFT step produced no non-zero gradient")
        gradient_sha = _gradient_fingerprint(torch, model)
        optimizer.step()
        torch.cuda.synchronize(device)
        step_elapsed = time.perf_counter() - step_started

        changed_tensors = 0
        changed_parameters = 0
        max_delta = 0.0
        delta_l2_squared = 0.0
        finite_after = True
        update_digest = hashlib.sha256()
        for name, param in sorted(model.named_parameters()):
            after = param.detach().cpu()
            finite_after = finite_after and bool(torch.isfinite(after).all().item())
            difference = after.float() - before[name].float()
            update_digest.update(name.encode("utf-8"))
            update_digest.update(_tensor_digest(torch, difference).encode("ascii"))
            count = int(torch.count_nonzero(difference).item())
            if count:
                changed_tensors += 1
                changed_parameters += count
                max_delta = max(max_delta, float(difference.abs().max().item()))
                delta_l2_squared += float(difference.square().sum().item())
        if not finite_after or not changed_parameters or not math.isfinite(max_delta):
            raise RuntimeError("optimizer did not produce changed finite parameters")
        after_sha = _model_fingerprint(torch, model)
        optimizer_sha = _optimizer_fingerprint(torch, model, optimizer)
        entry = {
            "step": step_index + 1,
            "schedule_step_index": step_index,
            "record_ids": [item["row"]["example_id"]],
            "record_order_sha256": stage_c.sha256_bytes(
                stage_c.canonical_json([item["row"]["example_id"]])),
            "loss": float(loss.item()),
            "grad_norm": float(grad_norm),
            "gradient_sha256": gradient_sha,
            "parameter_update_sha256": update_digest.hexdigest(),
            "nonzero_gradient_tensors": sum(
                torch.count_nonzero(g).item() > 0 for g in grad_tensors),
            "gradient_tensors": len(grad_tensors),
            "active_loss_weight": float(normalizer.item()),
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
            "target_audit": item["audit"],
        }
        if item.get("token_export") is not None:
            entry["token_export"] = item["token_export"]
        history.append(entry)
        if (step_index + 1) % prepared["checkpoint_every"] == 0 \
                or step_index + 1 == prepared["stop_after"]:
            save_checkpoint(step_index + 1)

    torch.cuda.synchronize(device)
    result = {
        "schema": LIVE_MANIFEST_SCHEMA,
        "status": "PASS",
        "REAL_TRAINING": "EXECUTED",
        "optimizer_status": "EXECUTED",
        "algorithm": "joint-interleaved-span-weighted-causal-sft-adamw-v1",
        "claims": {
            "model_quality": "NOT_RUN",
            "generalized_improvement": "NOT_RUN",
            "DPO": "NOT_RUN",
            "GRPO": "NOT_RUN",
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
        "batch_schedule": prepared["plan"]["batch_schedules"]["sft"],
        "executed_order_sha256": stage_c.sha256_bytes(stage_c.canonical_json([
            row["example_id"] for row in prepared["rows"][next_step:prepared["stop_after"]]
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
        "final_parameter_sha256": history[-1]["parameter_sha256_after"],
        "final_optimizer_state_sha256": history[-1]["optimizer_state_sha256"],
        "checkpoints": checkpoints,
        "timing": {"total_wall_ms": round((time.perf_counter() - started) * 1000.0, 3)},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "torch_cuda": torch.version.cuda,
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
            "records": prepared["token_export_preflight"]["records"],
            "target_retokenization_for_training": False,
            "legacy_equality_reference_only": True,
        }
    _atomic_json(output / "live_manifest.json", result)
    return result


def _token_export_preflight_main(argv):
    """Internal fresh-process entry point; never imports Torch."""
    parser = argparse.ArgumentParser(description="Validate and pack SFT token exports")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--max-seq-len", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
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
            raise ImportError("torch import is forbidden in token-export preflight")
        return real_import(name, *import_args, **import_kwargs)

    builtins.__import__ = no_torch
    importlib.util.find_spec = torch_unavailable
    try:
        import transformers
        import tokenizers
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, local_files_only=True, use_fast=True,
            trust_remote_code=False)
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("token-export preflight requires a fast tokenizer")
        dataset = stage_c.verify_and_load_dataset_pack(args.dataset_dir)
        rows = dataset["packs"].get("sft_train") or []
        version = (f"transformers-{transformers.__version__}_"
                   f"tokenizers-{tokenizers.__version__}")
        result = _pack_export_rows(
            rows, tokenizer, tokenizer_dir=args.model_path,
            tokenizer_id=args.tokenizer_id,
            tokenizer_revision=args.tokenizer_revision,
            tokenization_version=version,
            span_loss_weights=stage_c.DEFAULT_SPAN_WEIGHTS,
            max_seq_len=args.max_seq_len)
        if "torch" in sys.modules:
            raise RuntimeError("torch was imported during token-export preflight")
        _atomic_json(args.output, result)
    finally:
        builtins.__import__ = real_import
        importlib.util.find_spec = real_find_spec
    sys.stdout.write(stage_c.canonical_json({
        "status": "PASS", "records": len(result["records"]),
        "tokenization_version": version,
        "torch_imported": False,
    }).decode("utf-8") + "\n")
    return 0


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Execute the bounded real Stage C joint-SFT optimizer gate")
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
                        help="legacy preserves full-text re-encoding; token_export requires every train row")
    return parser


def main(argv=None):
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    if actual_argv and actual_argv[0] == "--token-export-preflight":
        return _token_export_preflight_main(actual_argv[1:])
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        prepared = prepare_live_run(
            dataset_dir=args.dataset_dir, output_dir=args.output_dir,
            config_path=args.config, model_path=args.model_path,
            source_commit=args.source_commit, max_steps=args.max_steps,
            checkpoint_every=args.checkpoint_every, stop_after=args.stop_after,
            resume_from=args.resume_from,
            model_revision=args.model_revision,
            token_source=args.token_source)
        result = execute_live_run(prepared)
    except (OSError, ValueError, RuntimeError, ImportError,
            json.JSONDecodeError) as exc:
        parser.exit(2, f"joint SFT live gate error: {exc}\n")
    sys.stdout.write(stage_c.canonical_json(result).decode("utf-8") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
