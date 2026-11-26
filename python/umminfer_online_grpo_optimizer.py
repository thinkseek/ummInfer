# SPDX-License-Identifier: Apache-2.0
"""Bounded online-GRPO LoRA optimizer for the exact local Qwen3.8 model.

The stdlib preparation path verifies a fresh online rollout and its preflight
bundle before creating output or importing Torch.  The live path is deliberately
limited to two whole-group AdamW updates, one fixed rank-1 attention projection,
a frozen BF16 base, and adapter-disabled reference forwards.  It is an optimizer
mechanics gate, not a quality, generalized-improvement, or deployment result.
"""

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import umminfer_behavior_token_alignment as alignment  # noqa: E402
import umminfer_online_grpo_preflight as preflight  # noqa: E402
import umminfer_online_rollout as online  # noqa: E402
import umminfer_rl_trajectory as trajectory  # noqa: E402

OPTIMIZER_SCHEMA = "umminfer-online-grpo-lora-optimizer-v1"
CHECKPOINT_SCHEMA = "umminfer-online-grpo-lora-checkpoint-v1"
UPDATE_SCHEMA = "umminfer-online-grpo-lora-update-v1"
REFERENCE_SCHEMA = "umminfer-online-grpo-lora-reference-v1"
LORA_TARGET = "model.language_model.layers.3.self_attn.q_proj"
LORA_RANK = 1
LORA_ALPHA = 1.0
TOTAL_STEPS = 2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return trajectory.canonical_json(value)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_json(path, value):
    write_bytes(path, canonical(value) + b"\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_identity(path):
    path = Path(path)
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _model_files(model_dir):
    require(model_dir.is_dir(), "exact model directory is absent")
    files = []
    for path in sorted(model_dir.iterdir(), key=lambda item: item.name):
        if path.is_symlink():
            raise ValueError(f"model identity rejects symlink: {path.name}")
        if path.is_file():
            files.append({"path": path.name, **file_identity(path)})
    weights = [item for item in files if item["path"].endswith(".safetensors")]
    require(len(weights) == 18, "the bounded gate requires the exact 18 Qwen weight shards")
    require((model_dir / "config.json").is_file(), "model config.json is absent")
    weight_map = {item["path"]: {"bytes": item["bytes"], "sha256": item["sha256"]}
                  for item in weights}
    return files, weight_map


def _source_identity(source_commit):
    require(isinstance(source_commit, str) and len(source_commit) == 40
            and set(source_commit) <= set("0123456789abcdef"),
            "source commit must be one lowercase 40-character Git identity")
    files = {}
    for relative in (
        "python/umminfer_online_grpo_optimizer.py",
        "python/umminfer_online_grpo_preflight.py",
        "python/umminfer_online_rollout.py",
        "python/umminfer_online_evidence_audit.py",
        "python/umminfer_rl_reward.py", "python/umminfer_ablation_metrics.py",
    ):
        files[relative] = file_identity(ROOT / relative)
    return {"commit": source_commit, "files": files,
            "scope": "online optimizer, rollout, preflight, and independent audit sources"}


def _objective_config(value):
    require(isinstance(value, dict), "preflight objective is missing")
    result = {key: float(value[key]) for key in ("clip_eps", "kl_coef", "log_ratio_clamp")}
    require(0.0 < result["clip_eps"] < 1.0 and result["kl_coef"] >= 0.0
            and result["log_ratio_clamp"] > 0.0
            and all(math.isfinite(item) for item in result.values()),
            "invalid bounded objective coefficients")
    return result


def prepare_run(args):
    """Validate all input/evidence/model/run identities without importing ML."""
    require(os.environ.get("UMMINFER_LIVE_OPT_IN") == "1",
            "set UMMINFER_LIVE_OPT_IN=1 for the real optimizer gate")
    require(args.stop_after in (1, TOTAL_STEPS), "stop-after must be 1 or 2")
    require(args.out.is_absolute(), "optimizer output must be an absolute new path")
    require(not args.out.exists(), "optimizer output already exists")
    require(args.online_pack.is_dir() and args.preflight_bundle.is_dir(),
            "online pack or preflight bundle is absent")
    require(args.learning_rate > 0.0 and math.isfinite(args.learning_rate),
            "learning rate must be finite and positive")
    require(args.max_grad_norm > 0.0 and math.isfinite(args.max_grad_norm),
            "max gradient norm must be finite and positive")
    require(args.seed_text and len(args.seed_text) <= 256, "invalid optimizer seed text")

    # This reconstructs every accepted file in a fresh temporary bundle and
    # compares bytes before an optimizer output directory or ML module exists.
    with tempfile.TemporaryDirectory(prefix="umm-online-optimizer-preflight-") as tmp:
        verification = preflight.verify_preflight(
            args.preflight_bundle, args.online_pack, Path(tmp) / "verified",
            tokenizer_dir=args.model_dir)
        require(verification.get("status") == "PASS"
                and verification.get("fresh_process_reconstruction") == "BYTE_IDENTICAL",
                "fresh online preflight reconstruction did not pass")
    verified_bytes = preflight._read_verified_bundle(args.preflight_bundle)
    documents = {name: json.loads(verified_bytes[name].decode("utf-8")) for name in (
        "manifest.json", "run_identity.json", "schedule.json", "objective_fixtures.json",
        "config.json")}
    manifest = documents["manifest.json"]
    require(manifest.get("status") == "PASS" and manifest.get("step_count") == TOTAL_STEPS
            and manifest.get("group_count") == TOTAL_STEPS,
            "the bounded optimizer requires exactly two preflight groups")
    identity_doc = documents["run_identity.json"]
    schedule = documents["schedule.json"]
    fixtures = documents["objective_fixtures.json"]
    config = documents["config.json"]
    identity = identity_doc["identity"]
    require(identity_doc["run_id"] == manifest["run_id"] == schedule["run_id"]
            == fixtures["run_id"], "preflight run identity drift")
    require(schedule.get("groups_per_step") == 1
            and len(schedule.get("steps", [])) == TOTAL_STEPS,
            "the bounded optimizer requires one whole group per step")
    require(identity.get("mode") == "online"
            and identity.get("sampled_ids_replaced_by_posthoc_ids") is False
            and identity.get("token_export_alignment") == "PASS_INDEPENDENT_STRUCTURAL",
            "online sampled behavior and strong token-export alignment are required")
    require(config.get("groups_per_step") == 1, "preflight group schedule drift")
    model_config = read_json(args.model_dir / "config.json")
    require(model_config.get("model_type") == "qwen3_5"
            and model_config.get("architectures") == ["Qwen3_5ForConditionalGeneration"],
            "local model is not the required Qwen3.8 conditional-generation architecture")
    vocab_size = model_config.get("text_config", {}).get("vocab_size")
    max_sequence = model_config.get("text_config", {}).get("max_position_embeddings")
    require(isinstance(vocab_size, int) and vocab_size > 0
            and isinstance(max_sequence, int) and max_sequence >= 8192,
            "Qwen text vocabulary/sequence limits are invalid")

    files, weight_map = _model_files(args.model_dir)
    model_config_sha = sha256_file(args.model_dir / "config.json")
    model_sha = sha256_bytes(canonical(weight_map))
    tokenizer_sha = alignment._loaded_tokenizer_digest(args.model_dir)
    behavior = identity["behavior_policy"]
    require(behavior["model_id"] == args.model_id
            and behavior["model_revision"] == args.model_revision
            and behavior["model_sha256"] == model_sha
            and behavior["model_config_sha256"] == model_config_sha
            and identity["behavior_tokenizer_sha256"] == tokenizer_sha,
            "exact behavior/model/config/tokenizer identity differs from local files")
    for label in ("current_policy", "reference_policy"):
        policy = identity[label]
        require(policy["model_id"] == args.model_id
                and policy["model_revision"] == args.model_revision
                and policy["model_sha256"] == model_sha
                and policy["model_config_sha256"] == model_config_sha
                and policy["tokenizer_sha256"] == tokenizer_sha,
                f"{label} is not bound to the exact local model and tokenizer")

    run_groups = identity["groups"]
    fixture_groups = fixtures["groups"]
    require(len(run_groups) == len(fixture_groups) == TOTAL_STEPS,
            "incomplete optimizer groups")
    groups = []
    for step_index, (run_group, fixture_group, scheduled) in enumerate(zip(
            run_groups, fixture_groups, schedule["steps"])):
        require(scheduled["step_index"] == step_index
                and scheduled["group_ids"] == [run_group["group_id"]]
                and run_group["group_id"] == fixture_group["group_id"]
                and run_group["group_export_sha256"] == fixture_group["group_export_sha256"],
                "group/data-order identity differs from preflight schedule")
        require(run_group["split"] == "train" and fixture_group["split"] == "train",
                "eval groups are forbidden in the bounded optimizer")
        require(run_group["has_reward_variance"] is True
                and fixture_group["has_reward_variance"] is True
                and run_group["group_std"] > 0.0,
                "zero-reward-variance groups are forbidden as optimizer evidence")
        members = []
        run_members = {item["rollout_id"]: item for item in run_group["members"]}
        for fixture_member in fixture_group["members"]:
            rollout_id = fixture_member["rollout_id"]
            require(rollout_id in run_members, "unknown or missing group member")
            run_member = run_members[rollout_id]
            packing = fixture_member["packing"]
            behavior_values = fixture_member["behavior_policy"]["sampled_logprobs"]
            evidence = fixture_member["behavior_policy"]
            require(evidence.get("source") == "FINAL_RAW_SSE_SAMPLED_BEHAVIOR"
                    and len(behavior_values) == len(packing["sampled_response_token_ids"])
                    and len(behavior_values) == len(packing["response_mask"])
                    - len(packing["prompt_token_ids"]),
                    "sampled behavior/packing lengths differ")
            require(run_member["token_export_alignment"]["status"] == "PASS"
                    and run_member["token_export_alignment"]["sampled_posthoc_target_ids_equal"] is True
                    and run_member["token_export_alignment"]["terminal_token_ids"],
                    "strong sampled/token-export alignment is missing")
            response_ids = packing["sampled_response_token_ids"]
            input_ids = packing["input_ids"]
            labels = packing["labels"]
            response_mask = packing["response_mask"]
            require(0 < len(response_ids) < 8192
                    and len(input_ids) == len(labels) == len(response_mask)
                    and len(input_ids) == len(packing["prompt_token_ids"]) + len(response_ids)
                    and len(input_ids) <= 8192
                    and all(isinstance(token, int) and not isinstance(token, bool)
                            and 0 <= token < vocab_size for token in input_ids)
                    and all(bit in (0, 1) for bit in response_mask)
                    and any(response_mask) and response_mask[0] == 0,
                    "invalid sequence/vocabulary/mask bounds")
            require(all(math.isfinite(float(value)) and float(value) <= 0.0
                        for value in behavior_values), "invalid sampled behavior logprob")
            advantage = float(run_member["advantage"])
            require(math.isfinite(advantage) and advantage != 0.0,
                    "every bounded optimizer member requires a finite nonzero advantage")
            require(fixture_member["metrics"]["advantage"] == advantage,
                    "member advantage differs between run and objective fixture")
            require(fixture_member["current_policy"]["forward"] == "NOT_RUN"
                    and fixture_member["reference_policy"]["forward"] == "NOT_RUN"
                    and fixture_member["current_policy"]["logprob_source"]
                    == "SYNTHETIC_CPU_FIXTURE_NOT_MODEL_OUTPUT",
                    "synthetic preflight policy values cannot become optimizer evidence")
            members.append({
                "rollout_id": rollout_id,
                "rank": fixture_member["rank"],
                "packing": packing,
                "behavior_logprobs": [float(value) for value in behavior_values],
                "advantage": advantage,
                "reward": run_member["reward"],
                "raw_sse_sha256": run_member["raw_sse_sha256"],
                "source_record_sha256": run_member["source_record_sha256"],
                "packed_ids_labels_masks_sha256":
                    run_member["packed_ids_labels_masks_sha256"],
            })
        require(len(members) == len(run_members) >= 2
                and len({item["rollout_id"] for item in members}) == len(members),
                "invalid group membership")
        # Require actual reward variance, never an advantage-sum approximation.
        groups.append({
            "group_id": run_group["group_id"],
            "schedule_group_id": run_group["schedule_group_id"],
            "group_mean": run_group["group_mean"], "group_std": run_group["group_std"],
            "group_export_sha256": run_group["group_export_sha256"],
            "preflight_checkpoint_before": scheduled["checkpoint_before"],
            "preflight_checkpoint_after": scheduled["checkpoint_after"],
            "members": members,
        })

    resume_from = Path(args.resume_from) if args.resume_from else None
    resume_update = None
    if resume_from is not None:
        require(resume_from.is_file(), "resume checkpoint is absent")
        sidecar = Path(str(resume_from) + ".sha256")
        require(sidecar.is_file() and sidecar.read_text(encoding="ascii").strip()
                == sha256_file(resume_from), "resume checkpoint sidecar is invalid")
        require(args.stop_after == TOTAL_STEPS, "a resume must execute the second update")
        resume_update_path = resume_from.parent.parent / "update-000001.json"
        resume_update = read_json(resume_update_path)
        require(resume_update.get("schema") == UPDATE_SCHEMA
                and resume_update.get("step") == 1,
                "resume requires its retained first update")

    source = _source_identity(args.source_commit)
    optimization = {
        "algorithm": "online-clipped-sequence-grpo-schulman-kl-adamw-lora-v1",
        "learning_rate": float(args.learning_rate),
        "weight_decay": 0.0, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8,
        "max_grad_norm": float(args.max_grad_norm), "foreach": False,
        "seed_text": args.seed_text, "total_steps": TOTAL_STEPS,
        "groups_per_step": 1, "rollout_loss_reduction": "MEAN_WITH_INDIVIDUAL_BACKWARD",
        "lora": {"target": LORA_TARGET, "rank": LORA_RANK, "alpha": LORA_ALPHA,
                 "base": "FROZEN_BF16", "adapter": "FP32", "dropout": 0.0},
        "gradient_checkpointing": True, "attention": "eager", "use_cache": False,
        "logits_to_keep": "SAMPLED_RESPONSE_LENGTH_PLUS_ONE",
        "reference": "SAME_FROZEN_BASE_WITH_ALL_LORA_ADAPTERS_DISABLED",
        "objective": _objective_config(identity["objective"]),
    }
    optimizer_identity = {
        "schema": OPTIMIZER_SCHEMA, "source": source,
        "preflight_run_id": manifest["run_id"],
        "preflight_manifest_sha256": sha256_file(args.preflight_bundle / "manifest.json"),
        "online_rollout_manifest_sha256": sha256_file(
            args.online_pack / "rollout_manifest.json"),
        "model": {"model_id": args.model_id, "model_revision": args.model_revision,
                  "model_sha256": model_sha, "model_config_sha256": model_config_sha,
                  "tokenizer_sha256": tokenizer_sha, "files": files,
                  "available_weights": weight_map},
        "behavior_policy": copy.deepcopy(behavior),
        "current_policy": copy.deepcopy(identity["current_policy"]),
        "reference_policy": copy.deepcopy(identity["reference_policy"]),
        "groups": [{"group_id": group["group_id"],
                    "schedule_group_id": group["schedule_group_id"],
                    "group_export_sha256": group["group_export_sha256"],
                    "preflight_checkpoint_before": group["preflight_checkpoint_before"],
                    "preflight_checkpoint_after": group["preflight_checkpoint_after"],
                    "rollout_ids": [member["rollout_id"] for member in group["members"]],
                    "packed_ids_labels_masks_sha256": [
                        member["packed_ids_labels_masks_sha256"]
                        for member in group["members"]]} for group in groups],
        "packed_input_order_sha256": identity["packed_input_order_sha256"],
        "optimization": optimization,
    }
    identity_sha = sha256_bytes(canonical(optimizer_identity))
    return {"identity": optimizer_identity, "identity_sha256": identity_sha,
            "groups": groups, "output": args.out, "model_dir": args.model_dir,
            "stop_after": args.stop_after, "resume_from": resume_from,
            "resume_update": resume_update,
            "preflight_bundle": args.preflight_bundle, "online_pack": args.online_pack}


def _float32_bytes(values):
    flat = []
    for value in values:
        flat.extend(value)
    return struct.pack("<" + "f" * len(flat), *flat)


def _tensor_layout(torch, named_tensors):
    layout = []
    offset = 0
    for name, tensor in named_tensors:
        count = int(tensor.numel())
        layout.append({"name": name, "shape": list(tensor.shape), "offset": offset,
                       "count": count, "dtype": "float32", "endianness": "little"})
        offset += count
    return layout, offset


def _write_tensor_set(torch, directory, stem, named_tensors):
    layout, count = _tensor_layout(torch, named_tensors)
    values = [tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().view(-1).tolist()
              for _name, tensor in named_tensors]
    payload = _float32_bytes(values)
    require(len(payload) == count * 4, "float32 tensor serialization length mismatch")
    path = directory / (stem + ".f32")
    write_bytes(path, payload)
    return {"path": path.name, "bytes": len(payload), "sha256": sha256_bytes(payload),
            "scalar_count": count, "dtype": "float32", "endianness": "little",
            "layout": layout}


def _state_identity(torch, value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu").contiguous()
        return {"type": "tensor", "dtype": str(tensor.dtype), "shape": list(tensor.shape),
                "sha256": sha256_bytes(_float32_bytes([
                    tensor.to(dtype=torch.float32).view(-1).tolist()]))}
    if isinstance(value, dict):
        return {str(key): _state_identity(torch, value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_state_identity(torch, item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        require(math.isfinite(value), "non-finite optimizer state")
        return value
    raise ValueError(f"unsupported optimizer state value: {type(value).__name__}")


def execute_live(prepared):
    """Run only the prepared exact-model LoRA updates on physical GPU 7."""
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "7",
            "live optimizer requires CUDA_VISIBLE_DEVICES=7 and no other device")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print("ONLINE_GRPO_ML_IMPORT_BEGIN", file=sys.stderr, flush=True)
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "live optimizer requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    require(tuple(torch.cuda.get_device_capability(device)) == (9, 0),
            "live optimizer requires an SM90 device")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    seed_text = prepared["identity"]["optimization"]["seed_text"]
    seed = int(sha256_bytes(seed_text.encode("utf-8"))[:16], 16) % (2 ** 31)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    runtime_identity = {
        "python": sys.version.split()[0], "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda, "transformers": transformers.__version__,
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "python_random": "NOT_USED", "numpy_random": "NOT_IMPORTED",
        "amp_scaler": "DISABLED", "scheduler": "NONE", "partial_accumulation": "FORBIDDEN",
    }

    output = prepared["output"]
    output.mkdir(parents=True)
    (output / "tensors").mkdir()
    (output / "checkpoints").mkdir()
    resume_payload = None
    next_step = 0
    if prepared["resume_from"] is not None:
        resume_payload = torch.load(
            prepared["resume_from"], map_location="cpu", weights_only=True)
        require(isinstance(resume_payload, dict)
                and resume_payload.get("schema") == CHECKPOINT_SCHEMA
                and resume_payload.get("run_identity_sha256") == prepared["identity_sha256"]
                and resume_payload.get("runtime_identity") == runtime_identity,
                "resume checkpoint schema, run identity, or runtime identity mismatch")
        next_step = resume_payload.get("next_step")
        require(next_step == 1 and resume_payload.get("schedule_position") == 1
                and resume_payload.get("total_steps") == TOTAL_STEPS
                and prepared["stop_after"] == TOTAL_STEPS,
                "resume cursor must select exactly the second update")
        prior_update = prepared["resume_update"]
        prior_update_sha = sha256_bytes(canonical(prior_update) + b"\n")
        require(prior_update["run_identity_sha256"] == prepared["identity_sha256"]
                and resume_payload.get("update_sha256") == prior_update_sha
                and resume_payload.get("optimizer_state_sha256")
                == prior_update["optimizer_state_sha256"]
                and resume_payload.get("completed_group_ids")
                == [prepared["groups"][0]["group_id"]]
                and resume_payload.get("remaining_group_ids")
                == [prepared["groups"][1]["group_id"]]
                and resume_payload.get("packed_input_order_sha256")
                == prepared["identity"]["packed_input_order_sha256"]
                and resume_payload.get("preflight_checkpoint_before")
                == prepared["groups"][0]["preflight_checkpoint_before"]
                and resume_payload.get("preflight_checkpoint_after")
                == prepared["groups"][0]["preflight_checkpoint_after"],
                "resume checkpoint/update/group/preflight chain mismatch")

    tokenizer = AutoTokenizer.from_pretrained(
        prepared["model_dir"], local_files_only=True, use_fast=True,
        trust_remote_code=False)
    require(getattr(tokenizer, "is_fast", False), "exact Qwen fast tokenizer is required")
    policy = AutoModelForImageTextToText.from_pretrained(
        prepared["model_dir"], local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="eager")
    require(type(policy).__name__ == "Qwen3_5ForConditionalGeneration",
            "the optimizer did not load the exact Qwen3.8 architecture")
    policy.config.use_cache = False
    policy.config.text_config.use_cache = False
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    class LoRALinear(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            generator = torch.Generator(device="cpu").manual_seed(seed)
            self.lora_A = torch.nn.Linear(base.in_features, LORA_RANK, bias=False)
            self.lora_B = torch.nn.Linear(LORA_RANK, base.out_features, bias=False)
            torch.nn.init.kaiming_uniform_(
                self.lora_A.weight, a=math.sqrt(5), generator=generator)
            torch.nn.init.zeros_(self.lora_B.weight)
            self.lora_A.weight.requires_grad_(True)
            self.lora_B.weight.requires_grad_(True)
            self.adapter_enabled = True

        def forward(self, inputs):
            base_output = self.base(inputs)
            if not self.adapter_enabled:
                return base_output
            update = self.lora_B(self.lora_A(inputs.to(dtype=torch.float32)))
            return base_output + update.to(dtype=base_output.dtype) * LORA_ALPHA

    parent = policy.get_submodule(LORA_TARGET.rsplit(".", 1)[0])
    target_name = LORA_TARGET.rsplit(".", 1)[1]
    base_projection = getattr(parent, target_name)
    require(isinstance(base_projection, torch.nn.Linear),
            "fixed LoRA target is not a linear projection")
    setattr(parent, target_name, LoRALinear(base_projection))
    adapters = [module for module in policy.modules() if isinstance(module, LoRALinear)]
    require(len(adapters) == 1, "bounded gate installed an unexpected adapter count")
    policy.to(device)
    policy.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    policy.train()
    trainable = sorted(((name, parameter) for name, parameter in policy.named_parameters()
                        if parameter.requires_grad), key=lambda item: item[0])
    require(len(trainable) == 2
            and sum(parameter.numel() for _name, parameter in trainable) == 17408,
            "bounded trainable surface is not the fixed 17,408-parameter rank-1 LoRA")
    require(all(parameter.dtype == torch.float32 for _name, parameter in trainable),
            "LoRA optimizer tensors must remain FP32")
    frozen_versions = {name: parameter._version for name, parameter in policy.named_parameters()
                       if not parameter.requires_grad}

    @contextlib.contextmanager
    def reference_mode():
        previous = [module.adapter_enabled for module in adapters]
        for module in adapters:
            module.adapter_enabled = False
        try:
            with torch.no_grad():
                yield
        finally:
            for module, enabled in zip(adapters, previous):
                module.adapter_enabled = enabled

    def selected_logprobs(member):
        packing = member["packing"]
        response = packing["sampled_response_token_ids"]
        input_ids = torch.tensor([packing["input_ids"]], dtype=torch.long, device=device)
        with contextlib.nullcontext():
            result = policy(input_ids=input_ids, use_cache=False,
                            logits_to_keep=len(response) + 1)
        logits = result.logits[0, :-1]
        require(tuple(logits.shape) == (len(response), policy.config.text_config.vocab_size),
                "causal response logit shape mismatch")
        scores = torch.log_softmax(logits.to(dtype=torch.float32), dim=-1)
        targets = torch.tensor(response, dtype=torch.long, device=device)
        selected = scores.gather(1, targets.unsqueeze(1)).squeeze(1).to(dtype=torch.float64)
        require(int(selected.numel()) == len(response)
                and bool(torch.isfinite(selected).all().item())
                and bool((selected <= 0.0).all().item()), "invalid selected model logprobs")
        return selected

    def mean_behavior(member):
        mask = member["packing"]["response_mask"]
        mask = mask[-len(member["behavior_logprobs"]):]
        values = [value for value, bit in zip(member["behavior_logprobs"], mask) if bit == 1]
        require(values, "member has no supervised behavior tokens")
        return math.fsum(values) / len(values)

    reference_groups = []
    with reference_mode():
        for group in prepared["groups"]:
            reference_members = []
            for member in group["members"]:
                values = selected_logprobs(member).to(device="cpu").tolist()
                mask = member["packing"]["response_mask"][-len(values):]
                supervised = [value for value, bit in zip(values, mask) if bit == 1]
                reference_members.append({
                    "rollout_id": member["rollout_id"], "logprobs": values,
                    "mean_logp": math.fsum(supervised) / len(supervised),
                    "packed_ids_labels_masks_sha256":
                        member["packed_ids_labels_masks_sha256"],
                })
            reference_groups.append({"group_id": group["group_id"],
                                     "members": reference_members})
    reference_document = {"schema": REFERENCE_SCHEMA,
                          "run_identity_sha256": prepared["identity_sha256"],
                          "groups": reference_groups,
                          "producer": "FROZEN_BASE_WITH_ADAPTER_DISABLED",
                          "base_model_sha256": prepared["identity"]["model"]["model_sha256"]}
    reference_sha = sha256_bytes(canonical(reference_document))
    if resume_payload is not None:
        require(resume_payload.get("reference_document") == reference_document
                and resume_payload.get("reference_document_sha256") == reference_sha,
                "resumed frozen-reference forward/cache differs byte-semantically")

    optimizer = torch.optim.AdamW(
        [parameter for _name, parameter in trainable],
        lr=prepared["identity"]["optimization"]["learning_rate"],
        betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, foreach=False)
    if resume_payload is not None:
        prior_update = prepared["resume_update"]
        expected_names = list(resume_payload["model_state"])
        require(expected_names == [name for name, _parameter in trainable],
                "resume trainable tensor order mismatch")
        with torch.no_grad():
            for name, parameter in trainable:
                require(tuple(parameter.shape) == tuple(resume_payload["model_state"][name].shape)
                        and parameter.dtype == resume_payload["model_state"][name].dtype,
                        f"resume shape/dtype mismatch: {name}")
                parameter.copy_(resume_payload["model_state"][name])
        restored_after = [[parameter.detach().to(device="cpu").view(-1).tolist()]
                          for _name, parameter in trainable]
        require(sha256_bytes(_float32_bytes(restored_after))
                == prior_update["parameters"]["after"]["sha256"],
                "resume model state differs from retained first-update after bytes")
        saved_optimizer = resume_payload["optimizer_state"]
        require(set(saved_optimizer) == {"state", "param_groups"}
                and saved_optimizer["param_groups"] == optimizer.state_dict()["param_groups"],
                "resume AdamW param-group/hyperparameter mismatch")
        require(set(saved_optimizer["state"]) == {0, 1}, "resume AdamW tensor states are incomplete")
        for index, (_name, parameter) in enumerate(trainable):
            state = saved_optimizer["state"][index]
            require(set(state) == {"step", "exp_avg", "exp_avg_sq"}
                    and state["step"].dtype == torch.float32
                    and int(state["step"].item()) == 1
                    and state["exp_avg"].shape == parameter.shape
                    and state["exp_avg_sq"].shape == parameter.shape
                    and state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
                    and bool(torch.isfinite(state["exp_avg"]).all().item())
                    and bool(torch.isfinite(state["exp_avg_sq"]).all().item()),
                    "resume AdamW state shape/dtype/step/finite mismatch")
        for key in ("exp_avg", "exp_avg_sq"):
            values = [[saved_optimizer["state"][index][key].detach().to(
                device="cpu").view(-1).tolist()] for index in range(len(trainable))]
            require(sha256_bytes(_float32_bytes(values))
                    == prior_update["moments_after"][key]["sha256"],
                    f"resume AdamW {key} differs from retained first-update bytes")
        optimizer.load_state_dict(saved_optimizer)
        require(sha256_bytes(canonical(_state_identity(torch, optimizer.state_dict())))
                == prior_update["optimizer_state_sha256"],
                "resume AdamW state digest mismatch")
        torch.set_rng_state(resume_payload["cpu_rng_state"])
        torch.cuda.set_rng_state_all(resume_payload["cuda_rng_state"])
        del resume_payload

    def save_checkpoint(completed_steps, update_sha, optimizer_sha):
        path = output / "checkpoints" / f"step-{completed_steps:06d}.pt"
        require(not path.exists(), "refusing to overwrite a checkpoint")
        payload = {
            "schema": CHECKPOINT_SCHEMA, "run_identity_sha256": prepared["identity_sha256"],
            "runtime_identity": runtime_identity,
            "next_step": completed_steps, "schedule_position": completed_steps,
            "total_steps": TOTAL_STEPS, "model_state": {
                name: parameter.detach().to(device="cpu") for name, parameter in trainable},
            "optimizer_state": optimizer.state_dict(),
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "reference_document": reference_document,
            "reference_document_sha256": reference_sha,
            "completed_group_ids": [group["group_id"]
                                    for group in prepared["groups"][:completed_steps]],
            "remaining_group_ids": [group["group_id"]
                                    for group in prepared["groups"][completed_steps:]],
            "packed_input_order_sha256": prepared["identity"]["packed_input_order_sha256"],
            "preflight_checkpoint_before":
                prepared["groups"][completed_steps - 1]["preflight_checkpoint_before"],
            "preflight_checkpoint_after":
                prepared["groups"][completed_steps - 1]["preflight_checkpoint_after"],
            "update_sha256": update_sha, "optimizer_state_sha256": optimizer_sha,
        }
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        digest = sha256_file(path)
        write_bytes(Path(str(path) + ".sha256"), (digest + "\n").encode("ascii"))
        return {"path": str(path.relative_to(output)), "next_step": completed_steps,
                **file_identity(path)}

    def tensor_objective(policy_mean, behavior_mean, reference_mean, advantage):
        coefficients = prepared["identity"]["optimization"]["objective"]
        clamp = coefficients["log_ratio_clamp"]
        behavior_scalar = torch.tensor(behavior_mean, dtype=torch.float64, device=device)
        reference_scalar = torch.tensor(reference_mean, dtype=torch.float64, device=device)
        raw_ratio = policy_mean - behavior_scalar
        log_ratio = raw_ratio.clamp(min=-clamp, max=clamp)
        ratio = torch.exp(log_ratio)
        clipped_ratio = ratio.clamp(
            min=1.0 - coefficients["clip_eps"], max=1.0 + coefficients["clip_eps"])
        unclipped = ratio * advantage
        clipped = clipped_ratio * advantage
        selected = torch.minimum(unclipped, clipped)
        raw_kl = reference_scalar - policy_mean
        kl_difference = raw_kl.clamp(min=-clamp, max=clamp)
        kl = torch.exp(kl_difference) - kl_difference - 1.0
        loss = -selected + coefficients["kl_coef"] * kl
        metrics = {
            "policy_mean_logp": float(policy_mean.item()),
            "behavior_mean_logp": behavior_mean, "reference_mean_logp": reference_mean,
            "raw_current_minus_behavior_logp": float(raw_ratio.item()),
            "current_minus_behavior_logp": float(log_ratio.item()),
            "ratio": float(ratio.item()), "clipped_ratio": float(clipped_ratio.item()),
            "advantage": advantage, "unclipped_surrogate": float(unclipped.item()),
            "clipped_surrogate": float(clipped.item()),
            "selected_surrogate": float(selected.item()),
            "clip_active": bool((unclipped > clipped).item()),
            "raw_reference_minus_current_logp": float(raw_kl.item()),
            "reference_minus_current_logp": float(kl_difference.item()),
            "schulman_kl": float(kl.item()), "objective_loss": float(loss.item()),
        }
        require(all(math.isfinite(value) for value in metrics.values()
                    if isinstance(value, float)) and metrics["ratio"] > 0.0,
                "non-finite online objective")
        return loss, metrics

    started = time.perf_counter()
    updates = []
    checkpoints = []
    for step_index in range(next_step, prepared["stop_after"]):
        group = prepared["groups"][step_index]
        optimizer.zero_grad(set_to_none=True)
        before_named = [(name, parameter.detach().to(device="cpu").clone())
                        for name, parameter in trainable]
        before_document = _write_tensor_set(
            torch, output / "tensors", f"step-{step_index + 1:06d}-before", before_named)
        member_results = []
        references = {item["rollout_id"]: item
                      for item in reference_groups[step_index]["members"]}
        for member in group["members"]:
            selected = selected_logprobs(member)
            if step_index == 0:
                require(selected.detach().to(device="cpu").tolist()
                        == references[member["rollout_id"]]["logprobs"],
                        "zero-initialized LoRA changed the initial policy forward")
            response_mask = member["packing"]["response_mask"][-len(member["behavior_logprobs"]):]
            supervised_indices = torch.tensor(
                [index for index, bit in enumerate(response_mask) if bit == 1],
                dtype=torch.long, device=device)
            policy_mean = selected.index_select(0, supervised_indices).mean()
            behavior_mean = mean_behavior(member)
            reference_mean = references[member["rollout_id"]]["mean_logp"]
            loss, metrics = tensor_objective(
                policy_mean, behavior_mean, reference_mean, member["advantage"])
            (loss / len(group["members"])).backward()
            member_results.append({
                "rollout_id": member["rollout_id"], "rank": member["rank"],
                "reward": member["reward"], "advantage": member["advantage"],
                "raw_sse_sha256": member["raw_sse_sha256"],
                "source_record_sha256": member["source_record_sha256"],
                "packed_ids_labels_masks_sha256": member["packed_ids_labels_masks_sha256"],
                "current_logprobs": selected.detach().to(device="cpu").tolist(),
                "sampled_behavior_logprobs": member["behavior_logprobs"],
                "reference_logprobs": references[member["rollout_id"]]["logprobs"],
                "response_mask": response_mask, "metrics": metrics,
            })
            del selected, policy_mean, loss
        gradient_named = []
        for name, parameter in trainable:
            require(parameter.grad is not None
                    and bool(torch.isfinite(parameter.grad).all().item()),
                    f"missing or non-finite real gradient: {name}")
            gradient_named.append((name, parameter.grad.detach().to(device="cpu").clone()))
        gradient_document = _write_tensor_set(
            torch, output / "tensors", f"step-{step_index + 1:06d}-gradient-preclip",
            gradient_named)
        nonzero_gradient_scalars = sum(
            int(torch.count_nonzero(parameter.grad).item()) for _name, parameter in trainable)
        require(nonzero_gradient_scalars > 0, "real online GRPO gradient is identically zero")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for _name, parameter in trainable],
            prepared["identity"]["optimization"]["max_grad_norm"], error_if_nonfinite=True)
        require(math.isfinite(float(grad_norm.item())) and float(grad_norm.item()) > 0.0,
                "real online GRPO gradient norm is zero or non-finite")
        postclip_named = [(name, parameter.grad.detach().to(device="cpu").clone())
                          for name, parameter in trainable]
        postclip_document = _write_tensor_set(
            torch, output / "tensors", f"step-{step_index + 1:06d}-gradient-postclip",
            postclip_named)
        optimizer.step()
        torch.cuda.synchronize(device)
        optimizer_state = optimizer.state_dict()
        exp_avg_named = [(name, optimizer_state["state"][index]["exp_avg"])
                         for index, (name, _parameter) in enumerate(trainable)]
        exp_avg_sq_named = [(name, optimizer_state["state"][index]["exp_avg_sq"])
                            for index, (name, _parameter) in enumerate(trainable)]
        exp_avg_document = _write_tensor_set(
            torch, output / "tensors", f"step-{step_index + 1:06d}-exp-avg", exp_avg_named)
        exp_avg_sq_document = _write_tensor_set(
            torch, output / "tensors", f"step-{step_index + 1:06d}-exp-avg-sq",
            exp_avg_sq_named)
        after_named = [(name, parameter.detach().to(device="cpu").clone())
                       for name, parameter in trainable]
        delta_named = [(name, after - before)
                       for (name, after), (_before_name, before) in zip(after_named, before_named)]
        after_document = _write_tensor_set(
            torch, output / "tensors", f"step-{step_index + 1:06d}-after", after_named)
        delta_document = _write_tensor_set(
            torch, output / "tensors", f"step-{step_index + 1:06d}-delta", delta_named)
        delta_values = [value for _name, tensor in delta_named
                        for value in tensor.view(-1).tolist()]
        changed = sum(value != 0.0 for value in delta_values)
        require(changed > 0 and all(math.isfinite(value) for value in delta_values),
                "optimizer produced no finite nonzero parameter delta")
        require(all(parameter.grad is None for name, parameter in policy.named_parameters()
                    if not parameter.requires_grad),
                "a frozen base parameter received a gradient")
        require(frozen_versions == {name: parameter._version
                                    for name, parameter in policy.named_parameters()
                                    if not parameter.requires_grad},
                "a frozen base parameter was modified in place")
        optimizer_sha = sha256_bytes(canonical(_state_identity(torch, optimizer_state)))
        group_loss = math.fsum(item["metrics"]["objective_loss"]
                               for item in member_results) / len(member_results)
        update = {
            "schema": UPDATE_SCHEMA, "run_identity_sha256": prepared["identity_sha256"],
            "step": step_index + 1, "schedule_step_index": step_index,
            "group_id": group["group_id"], "schedule_group_id": group["schedule_group_id"],
            "group_export_sha256": group["group_export_sha256"],
            "group_mean": group["group_mean"], "group_std": group["group_std"],
            "has_reward_variance": True,
            "rollout_ids": [member["rollout_id"] for member in group["members"]],
            "members": member_results, "objective_loss": group_loss,
            "normalization": "MASKED_RESPONSE_MEAN_THEN_ROLLOUT_MEAN",
            "gradient": {**gradient_document, "grad_norm": float(grad_norm.item()),
                         "nonzero_scalars": nonzero_gradient_scalars,
                         "postclip": postclip_document},
            "adamw": {key: prepared["identity"]["optimization"][key]
                      for key in ("learning_rate", "weight_decay", "beta1", "beta2",
                                  "epsilon", "max_grad_norm", "foreach")},
            "moments_after": {"step": step_index + 1, "exp_avg": exp_avg_document,
                              "exp_avg_sq": exp_avg_sq_document},
            "parameters": {
                "trainable_count": 17408, "trainable_tensors": 2,
                "before": before_document, "after": after_document, "delta": delta_document,
                "changed_scalars": changed,
                "max_abs_delta": max(abs(value) for value in delta_values),
                "delta_l2": math.sqrt(math.fsum(value * value for value in delta_values)),
            },
            "reference_document_sha256": reference_sha,
            "optimizer_state_sha256": optimizer_sha,
            "base_weights": "FROZEN_NO_GRAD_NO_OPTIMIZER_NO_VERSION_CHANGE",
            "objective": prepared["identity"]["optimization"]["objective"],
            "lora": prepared["identity"]["optimization"]["lora"],
        }
        update_path = output / f"update-{step_index + 1:06d}.json"
        write_json(update_path, update)
        update_sha = sha256_file(update_path)
        checkpoint = save_checkpoint(step_index + 1, update_sha, optimizer_sha)
        checkpoints.append(checkpoint)
        updates.append({"path": update_path.name, "sha256": update_sha,
                        "bytes": update_path.stat().st_size})

    torch.cuda.synchronize(device)
    environment = {
        "python": sys.version.split()[0], "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda, "transformers": transformers.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(device), "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "visible_device_count": torch.cuda.device_count(),
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    result = {
        "schema": OPTIMIZER_SCHEMA, "status": "PASS",
        "run_identity_sha256": prepared["identity_sha256"],
        "run_identity": prepared["identity"],
        "seed": {"text": seed_text, "derived_uint31": seed},
        "initial_step": next_step, "executed_step_count": prepared["stop_after"] - next_step,
        "resume": {"used": next_step != 0,
                   "checkpoint_sha256": sha256_file(prepared["resume_from"])
                   if prepared["resume_from"] is not None else None},
        "reference_document_sha256": reference_sha,
        "updates": updates, "checkpoints": checkpoints, "environment": environment,
        "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "claims": {"optimizer": "EXECUTED", "REAL_TRAINING": "EXECUTED_BOUNDED_LORA",
                   "trainable_parameters": 17408, "base_weights_updated": False,
                   "model_quality": "NOT_RUN", "generalized_improvement": "NOT_RUN",
                   "production_training": "NOT_RUN", "serving_process_started": False,
                   "online_rollouts": "FRESH_ATTESTED_SOURCE_IN_SEPARATE_PROCESS",
                   "synthetic_preflight_values_used_as_model_output": False},
    }
    write_json(output / "reference.json", reference_document)
    write_json(output / "result.json", result)
    checksums = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            checksums.append(f"{sha256_file(path)}  {path.relative_to(output)}")
    write_bytes(output / "SHA256SUMS", ("\n".join(checksums) + "\n").encode("ascii"))
    print(f"ONLINE_GRPO_OPTIMIZER: PASS run={prepared['identity_sha256']} "
          f"steps={result['executed_step_count']} trainable=17408")
    return result


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-pack", type=Path, required=True)
    parser.add_argument("--preflight-bundle", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="qwen3-native")
    parser.add_argument("--model-revision", default="UNKNOWN")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stop-after", type=int, default=TOTAL_STEPS)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed-text", default="umminfer-online-grpo-qwen38-lora-gate-v1")
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if os.environ.get("UMMINFER_LIVE_OPT_IN") != "1":
        print("REAL ONLINE GRPO OPTIMIZER: NOT_RUN (set UMMINFER_LIVE_OPT_IN=1)")
        return 0
    try:
        prepared = prepare_run(args)
        execute_live(prepared)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(2, f"online optimizer error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
