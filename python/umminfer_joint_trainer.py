# SPDX-License-Identifier: Apache-2.0
"""Stage C Decoupled Joint SFT + Preference (DPO / GRPO) Training & Ablation Driver.

Architecture boundary (docs/vision.md & docs/joint-training.md):
- 100% decoupled from C++ serving code under ``src/`` and ``include/``.
- Pure Python 3 stdlib core with zero required ``torch``, ``numpy``, or
  ``transformers`` dependencies for dataset verification, span-aware multimodal
  loss masking, ablation profile resolution, deterministic batch scheduling,
  run-manifest generation, GPU launch-script emission, and deterministic CPU
  numerical micro-step verification.
- Never fabricates real GPU training runs or claims model weights were updated
  when executing in ``plan``, ``gpu_launch_script``, or ``dry_run_micro_step``
  modes. Every emitted ``run_manifest.json`` records explicit execution mode
  and honest ``REAL_TRAINING`` status.
"""

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys

TRAINER_SCHEMA_VERSION = 1

VALID_TRAINING_MODES = ("sft", "dpo", "grpo", "joint")
VALID_EXECUTION_MODES = ("plan", "dry_run_micro_step", "gpu_launch_script")
SPAN_TYPES = (
    "narrative_prose",
    "visual_route_tag",
    "visual_prompt_body",
    "spoken_style",
)

DEFAULT_SEED = "umminfer-stage-c-train-v1"

DEFAULT_SPAN_WEIGHTS = {
    "narrative_prose": 1.0,
    "visual_route_tag": 1.5,
    "visual_prompt_body": 1.0,
    "spoken_style": 1.0,
    "image_form_multiplier": 1.0,
    "video_form_multiplier": 1.0,
}

DEFAULT_CONFIG = {
    "config_schema_version": 1,
    "config_type": "joint_default",
    "ablation_profile": "joint_full",
    "model": {
        "base_model_placeholder": "Qwen/Qwen2.5-VL-7B-Instruct",
        "reference_policy_placeholder": "${SFT_CHECKPOINT_DIR:-./checkpoints/sft_latest}",
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "gradient_checkpointing": True,
        "trust_remote_code": False,
    },
    "peft": {
        "method": "lora",
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    },
    "span_loss_weights": dict(DEFAULT_SPAN_WEIGHTS),
    "dpo": {
        "beta": 0.1,
        "loss_type": "sigmoid",
        "label_smoothing": 0.0,
        "max_prompt_length": 2048,
        "max_length": 4096,
    },
    "grpo": {
        "clip_eps": 0.2,
        "kl_coef": 0.04,
        "advantage_eps": 1e-8,
        "group_size": 4,
        "zero_std_policy": "zero_advantage",
    },
    "reward": {
        "spec": "umminfer_ablation_metrics:NarrativeRoutingReward",
        "export_only": True,
    },
    "optimization": {
        "learning_rate": 2e-5,
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.05,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 1,
        "max_seq_len": 4096,
        "seed": DEFAULT_SEED,
    },
    "micro_step_reference": {
        "sft_learning_rate": 0.15,
        "dpo_learning_rate": 0.25,
        "grpo_learning_rate": 0.15,
        "grad_clip": 5.0,
        "vocab_buckets": 32,
        "feature_dim": 32,
    },
}

ABLATION_PROFILES = {
    "joint_full": {
        "description": "Baseline joint training with all multimodal span losses and enabled stages active.",
        "changed_variable": "none (baseline)",
        "span_weight_overrides": {},
        "stage_overrides": {
            "enable_sft": True,
            "enable_dpo": True,
            "enable_grpo": True,
        },
    },
    "ablate_visual_routing": {
        "description": "Zeroes loss on [[image:/[[video: route decision headers and closing ]] delimiters.",
        "changed_variable": "span_loss_weights.visual_route_tag = 0.0",
        "span_weight_overrides": {
            "visual_route_tag": 0.0,
        },
        "stage_overrides": {
            "enable_sft": True,
            "enable_dpo": True,
            "enable_grpo": True,
        },
    },
    "ablate_visual_prompt": {
        "description": "Zeroes loss on visual prompt body spans inside [[image: ...]] / [[video: ...]] markers.",
        "changed_variable": "span_loss_weights.visual_prompt_body = 0.0",
        "span_weight_overrides": {
            "visual_prompt_body": 0.0,
        },
        "stage_overrides": {
            "enable_sft": True,
            "enable_dpo": True,
            "enable_grpo": True,
        },
    },
    "ablate_video_form": {
        "description": "Zeroes/masks [[video: ...]] beats while preserving [[image: ...]] beats to ablate image-vs-video routing.",
        "changed_variable": "span_loss_weights.video_form_multiplier = 0.0",
        "span_weight_overrides": {
            "video_form_multiplier": 0.0,
        },
        "stage_overrides": {
            "enable_sft": True,
            "enable_dpo": True,
            "enable_grpo": True,
        },
    },
    "ablate_spoken_style": {
        "description": "Masks <spoken> voice-leg style spans (weight 0.0) while preserving visual and prose supervision.",
        "changed_variable": "span_loss_weights.spoken_style = 0.0",
        "span_weight_overrides": {
            "spoken_style": 0.0,
        },
        "stage_overrides": {
            "enable_sft": True,
            "enable_dpo": True,
            "enable_grpo": True,
        },
    },
    "sft_only": {
        "description": "Runs interleaved SFT stage only with preference (DPO/GRPO) stages disabled.",
        "changed_variable": "preference_stages = disabled",
        "span_weight_overrides": {},
        "stage_overrides": {
            "enable_sft": True,
            "enable_dpo": False,
            "enable_grpo": False,
        },
    },
    "sft_plus_dpo": {
        "description": "Runs interleaved SFT followed by pairwise DPO (GRPO disabled).",
        "changed_variable": "preference_stage = dpo_only",
        "span_weight_overrides": {},
        "stage_overrides": {
            "enable_sft": True,
            "enable_dpo": True,
            "enable_grpo": False,
        },
    },
    "sft_plus_grpo": {
        "description": "Runs interleaved SFT followed by group-relative GRPO (DPO disabled).",
        "changed_variable": "preference_stage = grpo_only",
        "span_weight_overrides": {},
        "stage_overrides": {
            "enable_sft": True,
            "enable_dpo": False,
            "enable_grpo": True,
        },
    },
}

VALID_ABLATION_PROFILES = tuple(ABLATION_PROFILES.keys())

_HEX_256 = re.compile(r"[0-9a-f]{64}\Z")
_VISUAL_BEAT_RE = re.compile(r"(\[\[(image|video):[ \t]*)(.*?)(\]\])", re.DOTALL)
_SPOKEN_TAG_RE = re.compile(r"<spoken(?:\s+[^>]*)?>.*?</spoken>", re.DOTALL)
_TOKEN_RE = re.compile(r"\[\[|\]\]|</?spoken(?:\s+[^>]*)?>|[A-Za-z0-9_]+|[\u4e00-\u9fff]|[^\s]")


def canonical_json(obj):
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_text(text):
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _tokenize_span(text):
    """Deterministic stdlib subword-like tokenizer for span-level accounting."""
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text)
    if not tokens and text:
        # Whitespace-only spans preserve a single deterministic whitespace token
        # only if non-empty, keeping span accounting non-degenerate.
        return [text]
    return tokens


def _emit_visual_and_plain_spans(text, base_offset, default_span_type,
                                 default_modality):
    """Split ``text`` into visual beat spans and ``default_span_type`` spans."""
    spans = []
    cursor = 0
    for match in _VISUAL_BEAT_RE.finditer(text):
        if match.start() > cursor:
            seg = text[cursor:match.start()]
            spans.append({
                "span_type": default_span_type,
                "modality": default_modality,
                "start": base_offset + cursor,
                "end": base_offset + match.start(),
                "text": seg,
            })
        open_tag = match.group(1)
        modality = match.group(2)
        body = match.group(3)
        close_tag = match.group(4)

        open_start = base_offset + match.start(1)
        open_end = base_offset + match.end(1)
        spans.append({
            "span_type": "visual_route_tag",
            "modality": modality,
            "start": open_start,
            "end": open_end,
            "text": open_tag,
        })
        if body:
            spans.append({
                "span_type": "visual_prompt_body",
                "modality": modality,
                "start": base_offset + match.start(3),
                "end": base_offset + match.end(3),
                "text": body,
            })
        spans.append({
            "span_type": "visual_route_tag",
            "modality": modality,
            "start": base_offset + match.start(4),
            "end": base_offset + match.end(4),
            "text": close_tag,
        })
        cursor = match.end()

    if cursor < len(text):
        seg = text[cursor:]
        spans.append({
            "span_type": default_span_type,
            "modality": default_modality,
            "start": base_offset + cursor,
            "end": base_offset + len(text),
            "text": seg,
        })
    return spans


def parse_interleaved_spans(text):
    """Losslessly partition an interleaved assistant target into typed spans.

    Supported span types:
    - ``narrative_prose``: plain text outside visual beats and ``<spoken>`` tags.
    - ``visual_route_tag``: ``[[image: `` / ``[[video: `` header and closing ``]]``.
    - ``visual_prompt_body``: prompt body inside ``[[image: ...]]`` / ``[[video: ...]]``.
    - ``spoken_style``: ``<spoken>...</spoken>`` voice-aligned segments.

    Invariant: ``"".join(span["text"] for span in spans) == text`` for every
    string ``text``, with contiguous character offsets ``[0, len(text))``.
    """
    if not isinstance(text, str):
        raise TypeError("assistant target text must be a string")
    if not text:
        return []

    spans = []
    cursor = 0
    for spoken_match in _SPOKEN_TAG_RE.finditer(text):
        if spoken_match.start() > cursor:
            spans.extend(_emit_visual_and_plain_spans(
                text[cursor:spoken_match.start()],
                cursor,
                "narrative_prose",
                "text",
            ))
        spans.extend(_emit_visual_and_plain_spans(
            spoken_match.group(0),
            spoken_match.start(),
            "spoken_style",
            "audio",
        ))
        cursor = spoken_match.end()

    if cursor < len(text):
        spans.extend(_emit_visual_and_plain_spans(
            text[cursor:],
            cursor,
            "narrative_prose",
            "text",
        ))

    # Defensive lossless round-trip assertion.
    if "".join(span["text"] for span in spans) != text:
        raise RuntimeError("span parser violated lossless partition invariant")
    return spans


def resolve_ablation_profile(ablation_profile="joint_full",
                             base_span_weights=None):
    """Resolve effective span weights and stage toggles for an ablation profile."""
    if ablation_profile not in ABLATION_PROFILES:
        raise ValueError(
            f"unknown ablation profile {ablation_profile!r}; "
            f"valid profiles: {', '.join(VALID_ABLATION_PROFILES)}")
    profile = ABLATION_PROFILES[ablation_profile]
    weights = dict(DEFAULT_SPAN_WEIGHTS)
    if base_span_weights:
        for key, value in base_span_weights.items():
            if key not in weights:
                raise ValueError(f"unknown span weight key: {key!r}")
            if not _is_finite_number(value) or float(value) < 0.0:
                raise ValueError(
                    f"span weight {key!r} must be a finite number >= 0.0")
            weights[key] = float(value)
    # Ablation overrides always take precedence over base weights so each
    # single-variable ablation arm enforces its exact mask.
    for key, value in profile["span_weight_overrides"].items():
        weights[key] = float(value)

    return {
        "name": ablation_profile,
        "description": profile["description"],
        "changed_variable": profile["changed_variable"],
        "span_weights": weights,
        "stage_overrides": dict(profile["stage_overrides"]),
    }


def _effective_span_weight(span, effective_weights):
    base = float(effective_weights[span["span_type"]])
    modality = span["modality"]
    if modality == "image":
        return base * float(effective_weights["image_form_multiplier"])
    if modality == "video":
        return base * float(effective_weights["video_form_multiplier"])
    return base


def build_span_loss_mask(text, *, ablation_profile="joint_full",
                         span_weights=None):
    """Parse ``text`` and assign deterministic per-span and per-token loss weights."""
    resolved = resolve_ablation_profile(
        ablation_profile=ablation_profile,
        base_span_weights=span_weights,
    )
    eff_weights = resolved["span_weights"]
    raw_spans = parse_interleaved_spans(text)

    annotated_spans = []
    flat_tokens = []
    flat_weights = []
    span_counts = {name: 0 for name in SPAN_TYPES}
    span_token_counts = {name: 0 for name in SPAN_TYPES}
    weighted_token_mass = {name: 0.0 for name in SPAN_TYPES}
    modality_span_counts = {"text": 0, "image": 0, "video": 0, "audio": 0}

    for span in raw_spans:
        weight = _effective_span_weight(span, eff_weights)
        tokens = _tokenize_span(span["text"])
        token_count = len(tokens)
        entry = {
            "span_type": span["span_type"],
            "modality": span["modality"],
            "start": span["start"],
            "end": span["end"],
            "text": span["text"],
            "weight": weight,
            "active": weight > 0.0,
            "tokens": tokens,
            "token_count": token_count,
        }
        annotated_spans.append(entry)
        flat_tokens.extend(tokens)
        flat_weights.extend([weight] * token_count)
        span_counts[span["span_type"]] += 1
        span_token_counts[span["span_type"]] += token_count
        weighted_token_mass[span["span_type"]] += weight * token_count
        modality_span_counts[span["modality"]] += 1

    active_tokens = sum(1 for w in flat_weights if w > 0.0)
    masked_tokens = len(flat_weights) - active_tokens
    return {
        "ablation_profile": ablation_profile,
        "effective_weights": eff_weights,
        "spans": annotated_spans,
        "tokens": flat_tokens,
        "token_weights": flat_weights,
        "total_chars": len(text),
        "total_tokens": len(flat_tokens),
        "active_tokens": active_tokens,
        "masked_tokens": masked_tokens,
        "span_counts": span_counts,
        "span_token_counts": span_token_counts,
        "weighted_token_mass": {
            k: round(v, 9) for k, v in weighted_token_mass.items()
        },
        "modality_span_counts": modality_span_counts,
    }


def verify_and_load_dataset_pack(dataset_dir):
    """Fail-closed verifier and loader for a Stage C dataset pack directory.

    Verifies:
    1. ``manifest.json`` and ``manifest.json.sha256`` exist and match.
    2. Every output JSONL file in ``manifest["outputs"]`` exists and matches
       exact byte length, SHA-256 digest, record count, and expected split.
    3. Zero prompt leakage across train and eval splits (``leakage_count == 0``),
       recomputed independently from loaded JSONL rows.
    """
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {root}")
    manifest_path = root / "manifest.json"
    sidecar_path = root / "manifest.json.sha256"
    if not manifest_path.is_file():
        raise ValueError("dataset pack missing manifest.json")
    if not sidecar_path.is_file():
        raise ValueError("dataset pack missing manifest.json.sha256")

    actual_manifest_sha = sha256_file(manifest_path)
    expected_manifest_sha = sidecar_path.read_text(encoding="utf-8").strip()
    if not _HEX_256.fullmatch(expected_manifest_sha):
        raise ValueError("dataset manifest.json.sha256 is malformed")
    if actual_manifest_sha != expected_manifest_sha:
        raise ValueError(
            "dataset manifest.json.sha256 mismatch (tampered or stale manifest)")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("dataset manifest.json must be a JSON object")

    manifest_split_audit = manifest.get("split_audit")
    if not isinstance(manifest_split_audit, dict) \
            or manifest_split_audit.get("leakage_count") != 0 \
            or not manifest_split_audit.get("passed"):
        raise ValueError(
            "dataset manifest reports prompt leakage or missing split_audit")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("dataset manifest has no output packs")

    packs = {}
    verified_packs = {}
    train_prompts = set()
    eval_prompts = set()

    for logical_name in sorted(outputs):
        meta = outputs[logical_name]
        if not isinstance(meta, dict):
            raise ValueError(f"invalid output metadata for {logical_name}")
        rel_path = meta.get("path")
        if not isinstance(rel_path, str) or not rel_path \
                or Path(rel_path).name != rel_path:
            raise ValueError(f"invalid relative pack path for {logical_name}")
        file_path = root / rel_path
        if not file_path.is_file():
            raise ValueError(f"missing referenced pack file: {rel_path}")
        raw = file_path.read_bytes()
        actual_sha = sha256_bytes(raw)
        if len(raw) != meta.get("bytes") or actual_sha != meta.get("sha256"):
            raise ValueError(
                f"pack SHA-256 or byte length mismatch for {rel_path}")
        rows = [
            json.loads(line)
            for line in raw.decode("utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != meta.get("record_count"):
            raise ValueError(f"record count mismatch for {rel_path}")

        expected_split = "eval" if logical_name.endswith("_eval") else "train"
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"{rel_path} row {idx} is not a JSON object")
            if row.get("split") != expected_split:
                raise ValueError(
                    f"{rel_path} row {idx} split={row.get('split')!r} "
                    f"does not match {expected_split!r}")
            prompt_sha = row.get("prompt_sha256")
            if not isinstance(prompt_sha, str) or not _HEX_256.fullmatch(prompt_sha):
                raise ValueError(
                    f"{rel_path} row {idx} missing valid prompt_sha256")
            if expected_split == "eval":
                eval_prompts.add(prompt_sha)
            else:
                train_prompts.add(prompt_sha)

        packs[logical_name] = rows
        verified_packs[logical_name] = {
            "path": rel_path,
            "sha256": actual_sha,
            "bytes": len(raw),
            "record_count": len(rows),
        }

    overlap = sorted(train_prompts & eval_prompts)
    if overlap:
        raise ValueError(
            f"prompt leakage detected across train/eval splits: "
            f"{len(overlap)} overlapping prompts")

    return {
        "manifest": manifest,
        "manifest_sha256": actual_manifest_sha,
        "verified_packs": verified_packs,
        "packs": packs,
        "leakage_audit": {
            "train_prompt_count": len(train_prompts),
            "eval_prompt_count": len(eval_prompts),
            "overlap_prompt_sha256s": overlap,
            "leakage_count": 0,
            "passed": True,
        },
    }


def _resolve_active_stages(training_mode, ablation_resolved, available_packs):
    if training_mode not in VALID_TRAINING_MODES:
        raise ValueError(
            f"unsupported training mode {training_mode!r}; "
            f"expected one of {VALID_TRAINING_MODES}")
    stage_flags = ablation_resolved["stage_overrides"]

    if training_mode == "sft":
        if not stage_flags.get("enable_sft", True):
            raise ValueError(
                f"ablation {ablation_resolved['name']!r} disables sft stage")
        active = ["sft"]
    elif training_mode == "dpo":
        if not stage_flags.get("enable_dpo", True):
            raise ValueError(
                f"ablation {ablation_resolved['name']!r} disables dpo stage")
        active = ["dpo"]
    elif training_mode == "grpo":
        if not stage_flags.get("enable_grpo", True):
            raise ValueError(
                f"ablation {ablation_resolved['name']!r} disables grpo stage")
        active = ["grpo"]
    else:
        # training_mode == "joint"
        active = []
        if stage_flags.get("enable_sft", True):
            active.append("sft")
        if stage_flags.get("enable_dpo", True):
            active.append("dpo")
        if stage_flags.get("enable_grpo", True):
            active.append("grpo")
        if not active:
            raise ValueError("joint mode resolved to zero active stages")

    for stage in active:
        train_key = f"{stage}_train"
        if train_key not in available_packs:
            raise ValueError(
                f"dataset pack does not contain required {train_key}.jsonl "
                f"for mode={training_mode!r}, ablation={ablation_resolved['name']!r}")
    return active


def _deterministic_shuffle_key(seed, stage, epoch, item_id):
    return sha256_text(f"{seed}:{stage}:epoch{epoch}:{item_id}")


def _record_identifier(stage, record, fallback_index):
    if stage == "sft":
        return str(record.get("example_id") or record.get("target_sha256") or fallback_index)
    if stage == "dpo":
        return str(record.get("pair_id") or fallback_index)
    if stage == "grpo":
        return str(record.get("group_id") or fallback_index)
    return str(fallback_index)


def build_batch_schedule(records, *, stage, seed, batch_size, num_epochs):
    """Build a SHA-256 deterministic shuffled mini-batch schedule."""
    if not _is_int(batch_size) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not _is_int(num_epochs) or num_epochs <= 0:
        raise ValueError("num_epochs must be a positive integer")

    batches = []
    for epoch in range(num_epochs):
        indexed = [
            (_record_identifier(stage, row, idx), idx)
            for idx, row in enumerate(records)
        ]
        indexed.sort(
            key=lambda pair: (
                _deterministic_shuffle_key(seed, stage, epoch, pair[0]),
                pair[0],
                pair[1],
            )
        )
        for offset in range(0, len(indexed), batch_size):
            chunk = indexed[offset:offset + batch_size]
            batches.append({
                "epoch": epoch,
                "step_index": len(batches),
                "record_ids": [item[0] for item in chunk],
                "record_indices": [item[1] for item in chunk],
                "batch_size": len(chunk),
            })

    schedule_digest = sha256_bytes(canonical_json(batches))
    return {
        "stage": stage,
        "seed": str(seed),
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "total_batches": len(batches),
        "schedule_sha256": schedule_digest,
        "batches": batches,
    }


def _accumulate_mask_stats(agg, mask_info):
    agg["total_chars"] += mask_info["total_chars"]
    agg["total_tokens"] += mask_info["total_tokens"]
    agg["active_tokens"] += mask_info["active_tokens"]
    agg["masked_tokens"] += mask_info["masked_tokens"]
    for stype in SPAN_TYPES:
        agg["span_counts"][stype] += mask_info["span_counts"][stype]
        agg["span_token_counts"][stype] += mask_info["span_token_counts"][stype]
        agg["weighted_token_mass"][stype] += mask_info["weighted_token_mass"][stype]
    for mod, count in mask_info["modality_span_counts"].items():
        agg["modality_span_counts"][mod] += count


def _empty_span_agg():
    return {
        "total_chars": 0,
        "total_tokens": 0,
        "active_tokens": 0,
        "masked_tokens": 0,
        "span_counts": {name: 0 for name in SPAN_TYPES},
        "span_token_counts": {name: 0 for name in SPAN_TYPES},
        "weighted_token_mass": {name: 0.0 for name in SPAN_TYPES},
        "modality_span_counts": {"text": 0, "image": 0, "video": 0, "audio": 0},
    }


def _finalize_span_agg(agg):
    total_tok = agg["total_tokens"]
    agg["active_token_ratio"] = (
        round(agg["active_tokens"] / total_tok, 6) if total_tok > 0 else 0.0
    )
    agg["weighted_token_mass"] = {
        k: round(v, 6) for k, v in agg["weighted_token_mass"].items()
    }
    return agg


def compute_dataset_statistics(packs, *, active_stages, ablation_profile,
                               span_weights):
    """Compute exact deterministic sample, span, and token statistics."""
    stats = {}

    if "sft" in active_stages:
        for split_key in ("sft_train", "sft_eval"):
            rows = packs.get(split_key)
            if rows is None:
                continue
            agg = _empty_span_agg()
            text_only = 0
            image_slots = 0
            video_slots = 0
            spoken_samples = 0
            for row in rows:
                target = row.get("assistant_target", "")
                mask_info = build_span_loss_mask(
                    target,
                    ablation_profile=ablation_profile,
                    span_weights=span_weights,
                )
                _accumulate_mask_stats(agg, mask_info)
                slots = row.get("visual_slots") or []
                if not slots and mask_info["modality_span_counts"]["image"] == 0 \
                        and mask_info["modality_span_counts"]["video"] == 0:
                    text_only += 1
                image_slots += sum(
                    1 for s in slots if s.get("media_type") == "image")
                video_slots += sum(
                    1 for s in slots if s.get("media_type") == "video")
                if mask_info["span_counts"]["spoken_style"] > 0:
                    spoken_samples += 1
            stats[split_key] = {
                "sample_count": len(rows),
                "text_only_samples": text_only,
                "spoken_samples": spoken_samples,
                "visual_slot_counts": {
                    "image": image_slots,
                    "video": video_slots,
                },
                **_finalize_span_agg(agg),
            }

    if "dpo" in active_stages:
        for split_key in ("dpo_train", "dpo_eval"):
            rows = packs.get(split_key)
            if rows is None:
                continue
            chosen_agg = _empty_span_agg()
            rejected_agg = _empty_span_agg()
            margins = []
            for row in rows:
                margins.append(float(row["margin"]))
                _accumulate_mask_stats(
                    chosen_agg,
                    build_span_loss_mask(
                        row["chosen"]["assistant_target"],
                        ablation_profile=ablation_profile,
                        span_weights=span_weights,
                    ),
                )
                _accumulate_mask_stats(
                    rejected_agg,
                    build_span_loss_mask(
                        row["rejected"]["assistant_target"],
                        ablation_profile=ablation_profile,
                        span_weights=span_weights,
                    ),
                )
            stats[split_key] = {
                "pair_count": len(rows),
                "margin_summary": {
                    "mean": round(sum(margins) / len(margins), 6) if margins else 0.0,
                    "min": round(min(margins), 6) if margins else 0.0,
                    "max": round(max(margins), 6) if margins else 0.0,
                },
                "chosen_spans": _finalize_span_agg(chosen_agg),
                "rejected_spans": _finalize_span_agg(rejected_agg),
            }

    if "grpo" in active_stages:
        for split_key in ("grpo_train", "grpo_eval"):
            rows = packs.get(split_key)
            if rows is None:
                continue
            agg = _empty_span_agg()
            total_rollouts = 0
            zero_var_groups = 0
            nonzero_var_groups = 0
            single_member_groups = 0
            for group in rows:
                rollouts = group.get("rollouts") or []
                total_rollouts += len(rollouts)
                if len(rollouts) <= 1:
                    single_member_groups += 1
                if float(group.get("group_std", 0.0)) == 0.0 or len(rollouts) <= 1:
                    zero_var_groups += 1
                else:
                    nonzero_var_groups += 1
                for rollout in rollouts:
                    _accumulate_mask_stats(
                        agg,
                        build_span_loss_mask(
                            rollout.get("assistant_target", ""),
                            ablation_profile=ablation_profile,
                            span_weights=span_weights,
                        ),
                    )
            stats[split_key] = {
                "group_count": len(rows),
                "total_rollouts": total_rollouts,
                "single_member_groups": single_member_groups,
                "zero_variance_groups": zero_var_groups,
                "nonzero_variance_groups": nonzero_var_groups,
                **_finalize_span_agg(agg),
            }

    return stats


# ---------------------------------------------------------------------------
# Pure-Python Deterministic Numerical Reference Micro-Step Engine (Stdlib only)
# ---------------------------------------------------------------------------

def _hash_u32(key):
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big")


def _log_sigmoid_neg(x):
    """Numerically stable ``-log(sigmoid(x)) = log(1 + exp(-x))``."""
    if x >= 0.0:
        return math.log1p(math.exp(-x))
    return -x + math.log1p(math.exp(x))


def _sigmoid(x):
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class _LinearSoftmaxReferenceModel:
    """Tiny pure-Python linear/softmax scorer for CPU micro-step verification.

    Uses deterministic SHA-256 feature hashing and exact analytical gradients.
    Never claims to train real neural network weights.
    """

    def __init__(self, *, seed, vocab_buckets=32, feature_dim=32):
        self.vocab_buckets = int(vocab_buckets)
        self.feature_dim = int(feature_dim)
        self.weights = []
        self.bias = []
        for v in range(self.vocab_buckets):
            row = []
            for d in range(self.feature_dim):
                h = _hash_u32(f"{seed}:W:{v}:{d}")
                val = ((h / float(0xFFFFFFFF)) * 2.0 - 1.0) * 0.12
                row.append(val)
            self.weights.append(row)
            hb = _hash_u32(f"{seed}:b:{v}")
            self.bias.append(((hb / float(0xFFFFFFFF)) * 2.0 - 1.0) * 0.05)

    def clone(self):
        copy_model = object.__new__(_LinearSoftmaxReferenceModel)
        copy_model.vocab_buckets = self.vocab_buckets
        copy_model.feature_dim = self.feature_dim
        copy_model.weights = [list(row) for row in self.weights]
        copy_model.bias = list(self.bias)
        return copy_model

    def zero_grads(self):
        grad_w = [[0.0] * self.feature_dim for _ in range(self.vocab_buckets)]
        grad_b = [0.0] * self.vocab_buckets
        return grad_w, grad_b

    def token_bucket(self, token):
        return _hash_u32(f"tok:{token}") % self.vocab_buckets

    def context_features(self, prompt_sha, prev_token, span_type, position):
        """Return sparse list of ``(feature_idx, value)`` with unit L2 norm."""
        keys = (
            f"p:{prompt_sha[:16]}:{position % 8}",
            f"prev:{prev_token}",
            f"span:{span_type}:{position % 4}",
            f"pair:{prompt_sha[:8]}:{prev_token}:{span_type}",
        )
        scale = 0.5  # 1 / sqrt(4)
        feats = {}
        for k in keys:
            h = _hash_u32(k)
            idx = h % self.feature_dim
            sign = 1.0 if ((h >> 16) & 1) == 0 else -1.0
            feats[idx] = feats.get(idx, 0.0) + sign * scale
        return list(feats.items())

    def _forward_token(self, feats):
        V = self.vocab_buckets
        logits = [0.0] * V
        for v in range(V):
            row = self.weights[v]
            acc = self.bias[v]
            for idx, val in feats:
                acc += row[idx] * val
            logits[v] = acc
        max_z = max(logits)
        exps = [math.exp(z - max_z) for z in logits]
        sum_exp = sum(exps)
        lse = max_z + math.log(sum_exp)
        probs = [e / sum_exp for e in exps]
        return logits, lse, probs

    def sequence_weighted_logprob(self, prompt_sha, mask_info,
                                  *, grad_scale=None, grad_w=None, grad_b=None):
        """Compute span-weighted mean log-prob and optionally accumulate exact grads.

        If ``grad_scale`` is provided, accumulates ``grad_scale * d(mean_logp)/d(theta)``
        into ``(grad_w, grad_b)``.
        Returns ``(mean_logp, weighted_logp_sum, weight_sum, active_count)``.
        """
        tokens = mask_info["tokens"]
        token_weights = mask_info["token_weights"]
        # Map each token index back to its span_type for context features.
        token_span_types = []
        for span in mask_info["spans"]:
            token_span_types.extend([span["span_type"]] * span["token_count"])

        weight_sum = sum(w for w in token_weights if w > 0.0)
        if weight_sum <= 0.0 or not tokens:
            return 0.0, 0.0, 0.0, 0

        weighted_lp_sum = 0.0
        active_count = 0
        prev_tok = "<BOS>"
        for pos, (tok, w, stype) in enumerate(
                zip(tokens, token_weights, token_span_types)):
            if w > 0.0:
                y = self.token_bucket(tok)
                feats = self.context_features(prompt_sha, prev_tok, stype, pos)
                logits, lse, probs = self._forward_token(feats)
                lp = logits[y] - lse
                weighted_lp_sum += w * lp
                active_count += 1
                if grad_scale is not None and grad_w is not None and grad_b is not None:
                    coeff = grad_scale * (w / weight_sum)
                    for v in range(self.vocab_buckets):
                        dz = coeff * ((1.0 if v == y else 0.0) - probs[v])
                        grad_b[v] += dz
                        gw_row = grad_w[v]
                        for idx, val in feats:
                            gw_row[idx] += dz * val
            prev_tok = tok

        return weighted_lp_sum / weight_sum, weighted_lp_sum, weight_sum, active_count

    def apply_sgd(self, grad_w, grad_b, *, lr, grad_clip):
        sq_norm = 0.0
        for v in range(self.vocab_buckets):
            sq_norm += grad_b[v] * grad_b[v]
            for g in grad_w[v]:
                sq_norm += g * g
        grad_norm = math.sqrt(sq_norm)
        scale = 1.0
        if grad_clip > 0.0 and grad_norm > grad_clip:
            scale = grad_clip / grad_norm

        delta_sq = 0.0
        for v in range(self.vocab_buckets):
            db = -lr * scale * grad_b[v]
            self.bias[v] += db
            delta_sq += db * db
            row = self.weights[v]
            gw_row = grad_w[v]
            for d in range(self.feature_dim):
                dw = -lr * scale * gw_row[d]
                row[d] += dw
                delta_sq += dw * dw
        return grad_norm, math.sqrt(delta_sq)


def _compute_safe_group_advantages(group):
    """Use group advantages from the dataset pack, handling zero-std/singleton safely."""
    rollouts = group.get("rollouts") or []
    if len(rollouts) <= 1:
        return [0.0 for _ in rollouts], True

    scores = [float(r.get("score", 0.0)) for r in rollouts]
    mean = sum(scores) / len(scores)
    var = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = math.sqrt(var)
    if std == 0.0:
        return [0.0 for _ in rollouts], True

    advantages = []
    for rollout, score in zip(rollouts, scores):
        raw_adv = rollout.get("advantage")
        if raw_adv is not None and _is_finite_number(raw_adv):
            advantages.append(float(raw_adv))
        else:
            advantages.append((score - mean) / std)
    return advantages, False


def _run_sft_micro_steps(model, records, schedule, *, steps, ablation_profile,
                         span_weights, lr, grad_clip):
    batches = schedule["batches"]
    if not batches or not records:
        return {
            "status": "SKIPPED_EMPTY",
            "steps_executed": 0,
            "history": [],
        }

    history = []
    for step in range(steps):
        batch = batches[step % len(batches)]
        batch_rows = [records[idx] for idx in batch["record_indices"]]
        grad_w, grad_b = model.zero_grads()
        sample_losses = []
        total_weight_sum = 0.0
        total_active_tokens = 0
        inv_n = 1.0 / len(batch_rows)
        for row in batch_rows:
            mask_info = build_span_loss_mask(
                row["assistant_target"],
                ablation_profile=ablation_profile,
                span_weights=span_weights,
            )
            # Loss = -mean_logp -> grad_scale = -inv_n
            mean_lp, _w_lp_sum, w_sum, active_tok = model.sequence_weighted_logprob(
                row["prompt_sha256"],
                mask_info,
                grad_scale=-inv_n,
                grad_w=grad_w,
                grad_b=grad_b,
            )
            sample_loss = -mean_lp if w_sum > 0.0 else 0.0
            sample_losses.append(sample_loss)
            total_weight_sum += w_sum
            total_active_tokens += active_tok

        loss = sum(sample_losses) / len(sample_losses)
        grad_norm, weight_delta = model.apply_sgd(
            grad_w, grad_b, lr=lr, grad_clip=grad_clip)
        if not (math.isfinite(loss) and math.isfinite(grad_norm)
                and math.isfinite(weight_delta)):
            raise RuntimeError(f"non-finite SFT micro-step metric at step {step}")
        history.append({
            "step": step,
            "batch_step_index": batch["step_index"],
            "batch_size": len(batch_rows),
            "loss": round(loss, 8),
            "effective_weight_sum": round(total_weight_sum, 6),
            "active_tokens": total_active_tokens,
            "grad_norm": round(grad_norm, 8),
            "weight_delta_l2": round(weight_delta, 8),
        })

    return {
        "status": "COMPLETED_CPU_REFERENCE",
        "steps_executed": len(history),
        "initial_loss": history[0]["loss"],
        "final_loss": history[-1]["loss"],
        "loss_delta": round(history[-1]["loss"] - history[0]["loss"], 8),
        "history": history,
    }


def _run_dpo_micro_steps(model, ref_model, records, schedule, *, steps,
                         ablation_profile, span_weights, beta, lr, grad_clip):
    batches = schedule["batches"]
    if not batches or not records:
        return {
            "status": "SKIPPED_EMPTY",
            "steps_executed": 0,
            "history": [],
        }

    history = []
    for step in range(steps):
        batch = batches[step % len(batches)]
        batch_rows = [records[idx] for idx in batch["record_indices"]]
        grad_w, grad_b = model.zero_grads()
        pair_losses = []
        margins = []
        correct_prefs = 0
        inv_n = 1.0 / len(batch_rows)

        for row in batch_rows:
            prompt_sha = row["prompt_sha256"]
            chosen_mask = build_span_loss_mask(
                row["chosen"]["assistant_target"],
                ablation_profile=ablation_profile,
                span_weights=span_weights,
            )
            rejected_mask = build_span_loss_mask(
                row["rejected"]["assistant_target"],
                ablation_profile=ablation_profile,
                span_weights=span_weights,
            )
            r_w, _, _, _ = model.sequence_weighted_logprob(prompt_sha, chosen_mask)
            r_l, _, _, _ = model.sequence_weighted_logprob(prompt_sha, rejected_mask)
            ref_w, _, _, _ = ref_model.sequence_weighted_logprob(prompt_sha, chosen_mask)
            ref_l, _, _, _ = ref_model.sequence_weighted_logprob(prompt_sha, rejected_mask)

            implicit_margin = (r_w - ref_w) - (r_l - ref_l)
            scaled_margin = beta * implicit_margin
            pair_loss = _log_sigmoid_neg(scaled_margin)
            pair_losses.append(pair_loss)
            margins.append(implicit_margin)
            if scaled_margin > 0.0:
                correct_prefs += 1

            # Exact gradient: d(-log(sigmoid(z)))/dz = -sigmoid(-z)
            d_loss_d_z = -_sigmoid(-scaled_margin) * inv_n
            model.sequence_weighted_logprob(
                prompt_sha, chosen_mask,
                grad_scale=d_loss_d_z * beta,
                grad_w=grad_w, grad_b=grad_b,
            )
            model.sequence_weighted_logprob(
                prompt_sha, rejected_mask,
                grad_scale=-d_loss_d_z * beta,
                grad_w=grad_w, grad_b=grad_b,
            )

        loss = sum(pair_losses) / len(pair_losses)
        mean_margin = sum(margins) / len(margins)
        pref_acc = correct_prefs / len(batch_rows)
        grad_norm, weight_delta = model.apply_sgd(
            grad_w, grad_b, lr=lr, grad_clip=grad_clip)
        if not (math.isfinite(loss) and math.isfinite(mean_margin)
                and math.isfinite(grad_norm)):
            raise RuntimeError(f"non-finite DPO micro-step metric at step {step}")
        history.append({
            "step": step,
            "batch_step_index": batch["step_index"],
            "batch_size": len(batch_rows),
            "loss": round(loss, 8),
            "mean_implicit_reward_margin": round(mean_margin, 8),
            "preference_accuracy": round(pref_acc, 6),
            "grad_norm": round(grad_norm, 8),
            "weight_delta_l2": round(weight_delta, 8),
        })

    # Re-evaluate final batch post-update so final_loss reflects the last update.
    last_batch = batches[(steps - 1) % len(batches)]
    post_losses = []
    post_margins = []
    for idx in last_batch["record_indices"]:
        row = records[idx]
        prompt_sha = row["prompt_sha256"]
        chosen_mask = build_span_loss_mask(
            row["chosen"]["assistant_target"],
            ablation_profile=ablation_profile,
            span_weights=span_weights,
        )
        rejected_mask = build_span_loss_mask(
            row["rejected"]["assistant_target"],
            ablation_profile=ablation_profile,
            span_weights=span_weights,
        )
        r_w, _, _, _ = model.sequence_weighted_logprob(prompt_sha, chosen_mask)
        r_l, _, _, _ = model.sequence_weighted_logprob(prompt_sha, rejected_mask)
        ref_w, _, _, _ = ref_model.sequence_weighted_logprob(prompt_sha, chosen_mask)
        ref_l, _, _, _ = ref_model.sequence_weighted_logprob(prompt_sha, rejected_mask)
        margin = (r_w - ref_w) - (r_l - ref_l)
        post_margins.append(margin)
        post_losses.append(_log_sigmoid_neg(beta * margin))
    post_loss = round(sum(post_losses) / len(post_losses), 8)
    post_margin = round(sum(post_margins) / len(post_margins), 8)

    return {
        "status": "COMPLETED_CPU_REFERENCE",
        "steps_executed": len(history),
        "beta": float(beta),
        "initial_loss": history[0]["loss"],
        "final_loss": post_loss,
        "loss_delta": round(post_loss - history[0]["loss"], 8),
        "final_implicit_reward_margin": post_margin,
        "history": history,
    }


def _run_grpo_micro_steps(model, ref_model, records, schedule, *, steps,
                          ablation_profile, span_weights, clip_eps, kl_coef,
                          lr, grad_clip):
    batches = schedule["batches"]
    if not batches or not records:
        return {
            "status": "SKIPPED_EMPTY",
            "steps_executed": 0,
            "history": [],
        }

    old_model = model.clone()
    history = []
    total_zero_var_groups = 0

    for step in range(steps):
        batch = batches[step % len(batches)]
        batch_groups = [records[idx] for idx in batch["record_indices"]]
        grad_w, grad_b = model.zero_grads()

        rollout_losses = []
        kl_values = []
        abs_advantages = []
        clipped_count = 0
        zero_var_in_batch = 0

        # Count total rollouts in batch for exact mean normalization.
        total_rollouts = sum(len(g.get("rollouts") or []) for g in batch_groups)
        inv_total = 1.0 / max(total_rollouts, 1)

        for group in batch_groups:
            prompt_sha = group["prompt_sha256"]
            rollouts = group.get("rollouts") or []
            advantages, is_zero_var = _compute_safe_group_advantages(group)
            if is_zero_var:
                zero_var_in_batch += 1
                total_zero_var_groups += 1

            for rollout, adv in zip(rollouts, advantages):
                mask_info = build_span_loss_mask(
                    rollout.get("assistant_target", ""),
                    ablation_profile=ablation_profile,
                    span_weights=span_weights,
                )
                lp_cur, _, _, _ = model.sequence_weighted_logprob(
                    prompt_sha, mask_info)
                lp_old, _, _, _ = old_model.sequence_weighted_logprob(
                    prompt_sha, mask_info)
                lp_ref, _, _, _ = ref_model.sequence_weighted_logprob(
                    prompt_sha, mask_info)

                log_ratio = max(-20.0, min(20.0, lp_cur - lp_old))
                ratio = math.exp(log_ratio)
                clipped_ratio = max(1.0 - clip_eps, min(1.0 + clip_eps, ratio))
                surr1 = ratio * adv
                surr2 = clipped_ratio * adv
                if surr1 <= surr2:
                    surr = surr1
                    d_surr_d_lp = -adv * ratio
                else:
                    surr = surr2
                    d_surr_d_lp = 0.0
                    clipped_count += 1

                # Non-negative Schulman KL approximation: exp(log_ref - log_cur) - (log_ref - log_cur) - 1
                kl_diff = max(-20.0, min(20.0, lp_ref - lp_cur))
                kl_val = math.exp(kl_diff) - kl_diff - 1.0
                d_kl_d_lp = 1.0 - math.exp(kl_diff)

                obj_loss = -surr + kl_coef * kl_val
                rollout_losses.append(obj_loss)
                kl_values.append(kl_val)
                abs_advantages.append(abs(adv))

                d_loss_d_lp = (d_surr_d_lp + kl_coef * d_kl_d_lp) * inv_total
                if d_loss_d_lp != 0.0:
                    model.sequence_weighted_logprob(
                        prompt_sha, mask_info,
                        grad_scale=d_loss_d_lp,
                        grad_w=grad_w, grad_b=grad_b,
                    )

        loss = sum(rollout_losses) / max(len(rollout_losses), 1)
        mean_kl = sum(kl_values) / max(len(kl_values), 1)
        mean_abs_adv = sum(abs_advantages) / max(len(abs_advantages), 1)
        clip_frac = clipped_count / max(len(rollout_losses), 1)
        grad_norm, weight_delta = model.apply_sgd(
            grad_w, grad_b, lr=lr, grad_clip=grad_clip)
        if not (math.isfinite(loss) and math.isfinite(mean_kl)
                and math.isfinite(grad_norm) and math.isfinite(weight_delta)):
            raise RuntimeError(f"non-finite GRPO micro-step metric at step {step}")

        history.append({
            "step": step,
            "batch_step_index": batch["step_index"],
            "group_count": len(batch_groups),
            "rollout_count": len(rollout_losses),
            "zero_variance_groups_in_batch": zero_var_in_batch,
            "loss": round(loss, 8),
            "mean_kl": round(mean_kl, 8),
            "mean_abs_advantage": round(mean_abs_adv, 8),
            "clip_fraction": round(clip_frac, 6),
            "grad_norm": round(grad_norm, 8),
            "weight_delta_l2": round(weight_delta, 8),
        })

    return {
        "status": "COMPLETED_CPU_REFERENCE",
        "steps_executed": len(history),
        "clip_eps": float(clip_eps),
        "kl_coef": float(kl_coef),
        "zero_variance_groups_handled": total_zero_var_groups,
        "initial_loss": history[0]["loss"],
        "final_loss": history[-1]["loss"],
        "final_mean_kl": history[-1]["mean_kl"],
        "history": history,
    }


def _render_gpu_launch_script(*, training_mode, ablation_resolved,
                              dataset_manifest_sha256, config, active_stages):
    """Render a deterministic, portable HF/TRL/PEFT GPU launch script."""
    model_cfg = config["model"]
    peft_cfg = config["peft"]
    opt_cfg = config["optimization"]
    dpo_cfg = config["dpo"]
    grpo_cfg = config["grpo"]
    span_weights_json = json.dumps(
        ablation_resolved["span_weights"], sort_keys=True)
    target_modules_csv = ",".join(peft_cfg.get("target_modules", []))
    stages_csv = ",".join(active_stages)

    lines = [
        "#!/usr/bin/env bash",
        "# SPDX-License-Identifier: Apache-2.0",
        "# Auto-generated by umminfer_joint_trainer.py (Stage C decoupled GPU launcher).",
        "# Targets standard Hugging Face Accelerate + TRL + PEFT on multi-GPU nodes.",
        "# Contains zero hardcoded internal hostnames, IPs, or serving imports.",
        "set -euo pipefail",
        "",
        f'TRAINING_MODE="{training_mode}"',
        f'ABLATION_PROFILE="{ablation_resolved["name"]}"',
        f'ACTIVE_STAGES="{stages_csv}"',
        f'EXPECTED_DATASET_MANIFEST_SHA256="{dataset_manifest_sha256}"',
        "",
        f'BASE_MODEL="${{BASE_MODEL:-{model_cfg["base_model_placeholder"]}}}"',
        'DATASET_DIR="${DATASET_DIR:-./dataset_pack}"',
        'OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/stage_c_run}"',
        'NUM_GPUS="${NUM_GPUS:-1}"',
        'ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"',
        'PYTHON_BIN="${PYTHON_BIN:-python3}"',
        "",
        f'LORA_R="${{LORA_R:-{peft_cfg["r"]}}}"',
        f'LORA_ALPHA="${{LORA_ALPHA:-{peft_cfg["lora_alpha"]}}}"',
        f'LORA_DROPOUT="${{LORA_DROPOUT:-{peft_cfg["lora_dropout"]}}}"',
        f'LORA_TARGETS="${{LORA_TARGETS:-{target_modules_csv}}}"',
        f'LEARNING_RATE="${{LEARNING_RATE:-{opt_cfg["learning_rate"]}}}"',
        f'LR_SCHEDULER="${{LR_SCHEDULER:-{opt_cfg["lr_scheduler"]}}}"',
        f'WARMUP_RATIO="${{WARMUP_RATIO:-{opt_cfg["warmup_ratio"]}}}"',
        f'BATCH_SIZE="${{BATCH_SIZE:-{opt_cfg["per_device_train_batch_size"]}}}"',
        f'GRAD_ACCUM="${{GRAD_ACCUM:-{opt_cfg["gradient_accumulation_steps"]}}}"',
        f'NUM_EPOCHS="${{NUM_EPOCHS:-{opt_cfg["num_train_epochs"]}}}"',
        f'MAX_SEQ_LEN="${{MAX_SEQ_LEN:-{opt_cfg["max_seq_len"]}}}"',
        f'DPO_BETA="${{DPO_BETA:-{dpo_cfg["beta"]}}}"',
        f'GRPO_CLIP_EPS="${{GRPO_CLIP_EPS:-{grpo_cfg["clip_eps"]}}}"',
        f'GRPO_KL_COEF="${{GRPO_KL_COEF:-{grpo_cfg["kl_coef"]}}}"',
        f'TRAIN_SEED="${{TRAIN_SEED:-{opt_cfg["seed"]}}}"',
        f"SPAN_WEIGHTS_JSON='{span_weights_json}'",
        "",
        'echo "[stage-c] verifying dataset pack manifest SHA-256..."',
        '"${PYTHON_BIN}" - "${DATASET_DIR}" "${EXPECTED_DATASET_MANIFEST_SHA256}" <<\'PYEOF\'',
        "import hashlib, pathlib, sys",
        "root = pathlib.Path(sys.argv[1])",
        "expected = sys.argv[2]",
        "manifest = root / 'manifest.json'",
        "if not manifest.is_file():",
        "    raise SystemExit(f'missing dataset manifest: {manifest}')",
        "actual = hashlib.sha256(manifest.read_bytes()).hexdigest()",
        "if actual != expected:",
        "    raise SystemExit(f'dataset manifest sha256 mismatch: {actual} != {expected}')",
        "print(f'[stage-c] verified dataset manifest sha256={actual}')",
        "PYEOF",
        "",
        'mkdir -p "${OUTPUT_DIR}"',
        'export UMMINFER_STAGE_C_ABLATION="${ABLATION_PROFILE}"',
        'export UMMINFER_STAGE_C_SPAN_WEIGHTS="${SPAN_WEIGHTS_JSON}"',
        'export UMMINFER_STAGE_C_SEED="${TRAIN_SEED}"',
        "",
    ]

    if "sft" in active_stages:
        lines.extend([
            'echo "[stage-c] launching SFT stage (${ABLATION_PROFILE})..."',
            '"${ACCELERATE_BIN}" launch --num_processes "${NUM_GPUS}" -m trl.scripts.sft \\',
            '  --model_name_or_path "${BASE_MODEL}" \\',
            '  --dataset_name "${DATASET_DIR}/sft_train.jsonl" \\',
            '  --output_dir "${OUTPUT_DIR}/sft" \\',
            '  --learning_rate "${LEARNING_RATE}" \\',
            '  --lr_scheduler_type "${LR_SCHEDULER}" \\',
            '  --warmup_ratio "${WARMUP_RATIO}" \\',
            '  --per_device_train_batch_size "${BATCH_SIZE}" \\',
            '  --gradient_accumulation_steps "${GRAD_ACCUM}" \\',
            '  --num_train_epochs "${NUM_EPOCHS}" \\',
            '  --max_length "${MAX_SEQ_LEN}" \\',
            '  --use_peft \\',
            '  --lora_r "${LORA_R}" \\',
            '  --lora_alpha "${LORA_ALPHA}" \\',
            '  --lora_dropout "${LORA_DROPOUT}" \\',
            '  --lora_target_modules "${LORA_TARGETS}" \\',
            '  --bf16',
            'SFT_OUT="${OUTPUT_DIR}/sft"',
            "",
        ])
    else:
        lines.append('SFT_OUT="${SFT_CHECKPOINT_DIR:-${BASE_MODEL}}"')
        lines.append("")

    if "dpo" in active_stages:
        lines.extend([
            'echo "[stage-c] launching DPO stage (beta=${DPO_BETA}, ${ABLATION_PROFILE})..."',
            '"${ACCELERATE_BIN}" launch --num_processes "${NUM_GPUS}" -m trl.scripts.dpo \\',
            '  --model_name_or_path "${SFT_OUT}" \\',
            '  --dataset_name "${DATASET_DIR}/dpo_train.jsonl" \\',
            '  --output_dir "${OUTPUT_DIR}/dpo" \\',
            '  --beta "${DPO_BETA}" \\',
            '  --learning_rate "${LEARNING_RATE}" \\',
            '  --lr_scheduler_type "${LR_SCHEDULER}" \\',
            '  --warmup_ratio "${WARMUP_RATIO}" \\',
            '  --per_device_train_batch_size "${BATCH_SIZE}" \\',
            '  --gradient_accumulation_steps "${GRAD_ACCUM}" \\',
            '  --num_train_epochs "${NUM_EPOCHS}" \\',
            '  --use_peft \\',
            '  --lora_r "${LORA_R}" \\',
            '  --lora_alpha "${LORA_ALPHA}" \\',
            '  --lora_dropout "${LORA_DROPOUT}" \\',
            '  --bf16',
            'PREF_BASE="${OUTPUT_DIR}/dpo"',
            "",
        ])
    else:
        lines.append('PREF_BASE="${SFT_OUT}"')
        lines.append("")

    if "grpo" in active_stages:
        lines.extend([
            'echo "[stage-c] launching GRPO stage (clip_eps=${GRPO_CLIP_EPS}, kl_coef=${GRPO_KL_COEF}, ${ABLATION_PROFILE})..."',
            '"${ACCELERATE_BIN}" launch --num_processes "${NUM_GPUS}" -m trl.scripts.grpo \\',
            '  --model_name_or_path "${PREF_BASE}" \\',
            '  --dataset_name "${DATASET_DIR}/grpo_train.jsonl" \\',
            '  --output_dir "${OUTPUT_DIR}/grpo" \\',
            '  --epsilon "${GRPO_CLIP_EPS}" \\',
            '  --beta "${GRPO_KL_COEF}" \\',
            '  --learning_rate "${LEARNING_RATE}" \\',
            '  --lr_scheduler_type "${LR_SCHEDULER}" \\',
            '  --warmup_ratio "${WARMUP_RATIO}" \\',
            '  --per_device_train_batch_size "${BATCH_SIZE}" \\',
            '  --gradient_accumulation_steps "${GRAD_ACCUM}" \\',
            '  --num_train_epochs "${NUM_EPOCHS}" \\',
            '  --use_peft \\',
            '  --lora_r "${LORA_R}" \\',
            '  --lora_alpha "${LORA_ALPHA}" \\',
            '  --lora_dropout "${LORA_DROPOUT}" \\',
            '  --bf16',
            "",
        ])

    lines.append('echo "[stage-c] GPU launch sequence completed -> ${OUTPUT_DIR}"')
    return "\n".join(lines) + "\n"


def _prepare_output_dir(output_dir, *, overwrite):
    out = Path(output_dir)
    if out.exists():
        if not out.is_dir():
            raise ValueError("output path exists and is not a directory")
        if any(out.iterdir()) and not overwrite:
            raise ValueError(
                "output directory must be absent or empty (or pass --overwrite)")
    else:
        out.mkdir(parents=True, exist_ok=True)
    return out


def _write_bytes_atomic(path, payload, *, executable=False):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    if executable:
        mode = tmp.stat().st_mode
        tmp.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    os.replace(tmp, path)


def load_training_config(config_path=None):
    """Load and merge a JSON training config onto ``DEFAULT_CONFIG``."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_path is None:
        return config
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"training config not found: {path}")
    user_cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(user_cfg, dict):
        raise ValueError("training config must be a JSON object")
    return _deep_merge(config, user_cfg)


def prepare_training_run(*, mode="joint", ablation="joint_full",
                         dataset_dir, output_dir, config_path=None,
                         dry_run_steps=0, seed=None,
                         gpu_launch_script=False, overwrite=False):
    """Verify dataset pack, plan Stage C training/ablation, and write artifacts.

    Always writes:
    - ``run_manifest.json``
    - ``run_manifest.json.sha256``
    - ``launch_train.sh`` (executable, portable HF/TRL/PEFT GPU launcher)

    When ``dry_run_steps > 0``, executes deterministic pure-Python CPU reference
    micro-steps and records ``REAL_TRAINING: "NOT_RUN (CPU_REFERENCE_MICRO_STEP)"``.
    Never executes real GPU training or modifies model weights.
    """
    if mode not in VALID_TRAINING_MODES:
        raise ValueError(
            f"unsupported mode {mode!r}; expected one of {VALID_TRAINING_MODES}")
    if not _is_int(dry_run_steps) or dry_run_steps < 0:
        raise ValueError("dry_run_steps must be a non-negative integer")
    if dry_run_steps > 0 and gpu_launch_script:
        raise ValueError(
            "cannot combine --dry-run-steps > 0 with --gpu-launch-script")

    # 1. Fail-closed dataset manifest + SHA-256 + leakage verification first.
    dataset = verify_and_load_dataset_pack(dataset_dir)

    # 2. Load config and resolve ablation profile.
    config = load_training_config(config_path)
    effective_seed = str(
        seed if seed is not None else config["optimization"].get("seed", DEFAULT_SEED)
    )
    config["optimization"]["seed"] = effective_seed
    ablation_resolved = resolve_ablation_profile(
        ablation_profile=ablation,
        base_span_weights=config.get("span_loss_weights"),
    )
    config["ablation_profile"] = ablation_resolved["name"]
    config["span_loss_weights"] = dict(ablation_resolved["span_weights"])

    active_stages = _resolve_active_stages(
        mode, ablation_resolved, dataset["packs"])

    # 3. Compute exact span/token statistics and deterministic batch schedules.
    dataset_stats = compute_dataset_statistics(
        dataset["packs"],
        active_stages=active_stages,
        ablation_profile=ablation_resolved["name"],
        span_weights=ablation_resolved["span_weights"],
    )

    batch_size = int(config["optimization"]["per_device_train_batch_size"])
    num_epochs = int(config["optimization"]["num_train_epochs"])
    schedules = {}
    for stage in active_stages:
        train_rows = dataset["packs"][f"{stage}_train"]
        schedules[stage] = build_batch_schedule(
            train_rows,
            stage=stage,
            seed=effective_seed,
            batch_size=batch_size,
            num_epochs=num_epochs,
        )

    # 4. Determine honest execution mode & optionally run CPU micro-steps.
    micro_step_results = None
    if dry_run_steps > 0:
        execution_mode = "dry_run_micro_step"
        real_training_status = "NOT_RUN (CPU_REFERENCE_MICRO_STEP)"
        optimizer_claim = "CPU_REFERENCE_MICRO_STEP_ONLY"

        ref_cfg = config["micro_step_reference"]
        model = _LinearSoftmaxReferenceModel(
            seed=effective_seed,
            vocab_buckets=ref_cfg["vocab_buckets"],
            feature_dim=ref_cfg["feature_dim"],
        )
        micro_step_results = {
            "engine": "pure_python_linear_softmax_reference_v1",
            "steps_requested": dry_run_steps,
            "stages": {},
        }
        if "sft" in active_stages:
            micro_step_results["stages"]["sft"] = _run_sft_micro_steps(
                model,
                dataset["packs"]["sft_train"],
                schedules["sft"],
                steps=dry_run_steps,
                ablation_profile=ablation_resolved["name"],
                span_weights=ablation_resolved["span_weights"],
                lr=float(ref_cfg["sft_learning_rate"]),
                grad_clip=float(ref_cfg["grad_clip"]),
            )

        # Freeze post-SFT weights as reference policy for DPO / GRPO.
        ref_policy = model.clone()

        if "dpo" in active_stages:
            dpo_model = ref_policy.clone()
            micro_step_results["stages"]["dpo"] = _run_dpo_micro_steps(
                dpo_model,
                ref_policy,
                dataset["packs"]["dpo_train"],
                schedules["dpo"],
                steps=dry_run_steps,
                ablation_profile=ablation_resolved["name"],
                span_weights=ablation_resolved["span_weights"],
                beta=float(config["dpo"]["beta"]),
                lr=float(ref_cfg["dpo_learning_rate"]),
                grad_clip=float(ref_cfg["grad_clip"]),
            )
            post_dpo_policy = dpo_model
        else:
            post_dpo_policy = ref_policy

        if "grpo" in active_stages:
            grpo_model = post_dpo_policy.clone()
            micro_step_results["stages"]["grpo"] = _run_grpo_micro_steps(
                grpo_model,
                ref_policy,
                dataset["packs"]["grpo_train"],
                schedules["grpo"],
                steps=dry_run_steps,
                ablation_profile=ablation_resolved["name"],
                span_weights=ablation_resolved["span_weights"],
                clip_eps=float(config["grpo"]["clip_eps"]),
                kl_coef=float(config["grpo"]["kl_coef"]),
                lr=float(ref_cfg["grpo_learning_rate"]),
                grad_clip=float(ref_cfg["grad_clip"]),
            )
    elif gpu_launch_script:
        execution_mode = "gpu_launch_script"
        real_training_status = "NOT_RUN"
        optimizer_claim = "NOT_RUN"
    else:
        execution_mode = "plan"
        real_training_status = "NOT_RUN"
        optimizer_claim = "NOT_RUN"

    # 5. Render deterministic portable GPU launch script.
    launch_script_text = _render_gpu_launch_script(
        training_mode=mode,
        ablation_resolved=ablation_resolved,
        dataset_manifest_sha256=dataset["manifest_sha256"],
        config=config,
        active_stages=active_stages,
    )
    launch_script_bytes = launch_script_text.encode("utf-8")
    launch_script_sha = sha256_bytes(launch_script_bytes)

    config_sha = sha256_bytes(canonical_json(config))

    # 6. Assemble deterministic content-addressed run_manifest.json.
    manifest = {
        "trainer_schema_version": TRAINER_SCHEMA_VERSION,
        "status": "COMPLETED",
        "mode": execution_mode,
        "training_mode": mode,
        "active_stages": active_stages,
        "REAL_TRAINING": real_training_status,
        "claims": {
            "REAL_TRAINING": real_training_status,
            "optimizer": optimizer_claim,
            "model_weights_updated": False,
            "real_gpu_training_executed": False,
            "decoupled_from_cpp_serving": True,
        },
        "seed": effective_seed,
        "ablation": ablation_resolved,
        "config_sha256": config_sha,
        "hyperparameters": {
            "model": config["model"],
            "peft": config["peft"],
            "optimization": config["optimization"],
            "dpo": config["dpo"],
            "grpo": config["grpo"],
            "reward": config["reward"],
        },
        "dataset": {
            "manifest_sha256": dataset["manifest_sha256"],
            "source_mode": dataset["manifest"].get("mode"),
            "source_seed": dataset["manifest"].get("seed"),
            "verified_packs": dataset["verified_packs"],
            "leakage_audit": dataset["leakage_audit"],
        },
        "statistics": dataset_stats,
        "batch_schedules": {
            stage: {
                "stage": sched["stage"],
                "seed": sched["seed"],
                "batch_size": sched["batch_size"],
                "num_epochs": sched["num_epochs"],
                "total_batches": sched["total_batches"],
                "schedule_sha256": sched["schedule_sha256"],
                "batches": [
                    {
                        "epoch": b["epoch"],
                        "step_index": b["step_index"],
                        "batch_size": b["batch_size"],
                        "record_ids": b["record_ids"],
                    }
                    for b in sched["batches"]
                ],
            }
            for stage, sched in schedules.items()
        },
        "micro_step_verification": micro_step_results,
        "artifacts": {
            "launch_script": {
                "path": "launch_train.sh",
                "sha256": launch_script_sha,
                "bytes": len(launch_script_bytes),
            },
        },
    }

    # 7. Write atomic output bundle.
    out_path = _prepare_output_dir(output_dir, overwrite=overwrite)
    _write_bytes_atomic(
        out_path / "launch_train.sh", launch_script_bytes, executable=True)

    manifest_bytes = canonical_json(manifest) + b"\n"
    manifest_sha = sha256_bytes(manifest_bytes)
    _write_bytes_atomic(out_path / "run_manifest.json", manifest_bytes)
    _write_bytes_atomic(
        out_path / "run_manifest.json.sha256",
        (manifest_sha + "\n").encode("utf-8"),
    )
    return manifest


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Stage C Decoupled Joint SFT + Preference (DPO/GRPO) Training & "
            "Ablation Driver (pure Python stdlib core)"
        )
    )
    parser.add_argument(
        "--mode",
        choices=VALID_TRAINING_MODES,
        default="joint",
        help="Training objective mode: sft, dpo, grpo, or joint",
    )
    parser.add_argument(
        "--ablation",
        choices=VALID_ABLATION_PROFILES,
        default="joint_full",
        help="Single-variable ablation profile",
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Verified Stage C dataset pack directory (with manifest.json + .sha256)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON training configuration file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for run_manifest.json, .sha256, and launch_train.sh",
    )
    parser.add_argument(
        "--dry-run-steps",
        type=int,
        default=0,
        help="Run N deterministic pure-Python CPU numerical verification micro-steps",
    )
    parser.add_argument(
        "--seed",
        default=None,
        help="Deterministic seed override for batch schedule and micro-step init",
    )
    parser.add_argument(
        "--gpu-launch-script",
        action="store_true",
        help="Mark execution mode explicitly as gpu_launch_script (no CPU micro-steps)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty --output-dir",
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = prepare_training_run(
            mode=args.mode,
            ablation=args.ablation,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            config_path=args.config,
            dry_run_steps=args.dry_run_steps,
            seed=args.seed,
            gpu_launch_script=args.gpu_launch_script,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"joint trainer error: {exc}\n")
    sys.stdout.write(canonical_json(manifest).decode("utf-8") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
