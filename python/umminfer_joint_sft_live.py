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
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time

# Direct script execution uses -I, so add only this repository's known paths.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

import umminfer_joint_trainer as stage_c  # noqa: E402

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


def _source_fingerprints(source_commit):
    if not isinstance(source_commit, str) or not _HEX_40.fullmatch(source_commit):
        raise ValueError("--source-commit must be one lowercase 40-character commit")
    files = {}
    for rel in SOURCE_FILES:
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
                     stop_after=None, resume_from=None, model_revision=None):
    """Run every model-independent validation before importing torch/transformers."""
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
    scheduled_audits = [_audit_sft_record(row) for row in scheduled_rows]
    if not any(a["text_only"] for a in scheduled_audits) \
            or not any(a["visual_modalities"] for a in scheduled_audits):
        raise ValueError("scheduled gate steps must cover text-only and structured visual targets")
    if not any(a["terminal_legs_verified"] for a in scheduled_audits):
        raise ValueError("scheduled gate steps must verify completed visual terminal legs")

    model = _fingerprint_model(model_path)
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
        "schedule_sha256": schedule["schedule_sha256"],
        "model": model,
        "optimization": opt,
        "span_loss_weights": plan["ablation"]["span_weights"],
        "max_steps": max_steps,
        "objective": "full-parameter-span-weighted-causal-sft-v1",
    }
    return {
        "output_dir": output,
        "dataset_dir": Path(dataset_dir),
        "config_path": config_file,
        "model_path": Path(model_path),
        "plan": plan,
        "rows": scheduled_rows,
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
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    tokenizer = AutoTokenizer.from_pretrained(
        prepared["model_path"], local_files_only=True, use_fast=True,
        trust_remote_code=False)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("live gate requires a fast tokenizer before model loading")

    packed = []
    max_seq_len = prepared["identity"]["optimization"]["max_seq_len"]
    for row, audit in zip(prepared["rows"], prepared["row_audits"]):
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
            target, ablation_profile="joint_full",
            span_weights=prepared["identity"]["span_loss_weights"])
        labels, weights, annotations = _token_loss_mask(
            offsets, target_start=target_start, target=target,
            token_ids=ids, mask_info=mask_info)
        route_tokens = sum(
            "visual_route_tag" in item["span_types"] for item in annotations)
        prompt_body_tokens = sum(
            "visual_prompt_body" in item["span_types"] for item in annotations)
        closing_tokens = sum(
            item.get("includes_terminal_delimiter", False) for item in annotations)
        if audit["visual_modalities"] and not (
                route_tokens and prompt_body_tokens and closing_tokens):
            raise ValueError("HF token mask misses visual route, prompt, or terminal delimiter")
        packed.append({
            "row": row,
            "input_ids": ids,
            "labels": labels,
            "weights": weights,
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
        })

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
    output = prepared["output_dir"]

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
        optimizer.step()
        torch.cuda.synchronize(device)
        step_elapsed = time.perf_counter() - step_started

        changed_tensors = 0
        changed_parameters = 0
        max_delta = 0.0
        delta_l2_squared = 0.0
        finite_after = True
        for name, param in model.named_parameters():
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
    _atomic_json(output / "live_manifest.json", result)
    return result


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
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        prepared = prepare_live_run(
            dataset_dir=args.dataset_dir, output_dir=args.output_dir,
            config_path=args.config, model_path=args.model_path,
            source_commit=args.source_commit, max_steps=args.max_steps,
            checkpoint_every=args.checkpoint_every, stop_after=args.stop_after,
            resume_from=args.resume_from,
            model_revision=args.model_revision)
        result = execute_live_run(prepared)
    except (OSError, ValueError, RuntimeError, ImportError,
            json.JSONDecodeError) as exc:
        parser.exit(2, f"joint SFT live gate error: {exc}\n")
    sys.stdout.write(stage_c.canonical_json(result).decode("utf-8") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
