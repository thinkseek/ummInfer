#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stdlib-only independent verification of bounded online-GRPO optimizer bytes.

No rollout, alignment, preflight, optimizer, reward, Torch, Transformers, NumPy,
or other producer/scoring implementation is imported.  The established separate
independent auditor is invoked as a subprocess for raw-SSE/CAS/export/preflight
identity; this file independently checks model files, update arithmetic, raw
float32 tensors, checkpoint bytes, and deterministic/resume identity.
"""

import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

D0 = Decimal(0)
D1 = Decimal(1)
OBJECTIVE_TOLERANCE = Decimal("1e-8")
NORM_REL_TOLERANCE = 1e-5


class VerificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(
            VerificationError(f"non-finite JSON number: {value}")))


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


def file_identity(path):
    path = Path(path)
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def read_f32(document, run_dir):
    require(isinstance(document, dict) and document.get("endianness") == "little"
            and document.get("scalar_count") >= 0
            and isinstance(document.get("layout"), list), "invalid tensor document")
    path = run_dir / "tensors" / document["path"]
    payload = path.read_bytes()
    require(len(payload) == document["bytes"] == document["scalar_count"] * 4
            and sha256_bytes(payload) == document["sha256"],
            f"raw tensor hash/length mismatch: {document['path']}")
    values = list(struct.unpack("<" + "f" * document["scalar_count"], payload))
    require(all(math.isfinite(value) for value in values),
            f"non-finite raw tensor: {document['path']}")
    expected_offset = 0
    names = []
    for item in document["layout"]:
        require(item["dtype"] == "float32" and item["endianness"] == "little"
                and item["offset"] == expected_offset and item["count"] >= 0
                and item["count"] == math.prod(item["shape"]),
                "invalid tensor layout")
        expected_offset += item["count"]
        names.append(item["name"])
    require(expected_offset == len(values) and names == sorted(names)
            and len(names) == len(set(names)), "noncanonical tensor order/layout")
    return payload, values, document["layout"]


def decimal_mean(values):
    require(bool(values), "empty supervised logprob mean")
    return sum((Decimal(str(value)) for value in values), D0) / Decimal(len(values))


def decimal_metrics(current, behavior, reference, mask, advantage, objective):
    require(len(current) == len(behavior) == len(reference) == len(mask)
            and len(current) > 0 and all(bit in (0, 1) for bit in mask),
            "misaligned objective arrays")
    indices = [index for index, bit in enumerate(mask) if bit == 1]
    require(bool(indices), "no supervised objective token")
    policy_mean = decimal_mean([current[index] for index in indices])
    behavior_mean = decimal_mean([behavior[index] for index in indices])
    reference_mean = decimal_mean([reference[index] for index in indices])
    clamp = Decimal(str(objective["log_ratio_clamp"]))
    epsilon = Decimal(str(objective["clip_eps"]))
    kl_coefficient = Decimal(str(objective["kl_coef"]))
    raw_ratio = policy_mean - behavior_mean
    log_ratio = max(-clamp, min(clamp, raw_ratio))
    ratio = log_ratio.exp()
    clipped_ratio = max(D1 - epsilon, min(D1 + epsilon, ratio))
    advantage_decimal = Decimal(str(advantage))
    unclipped = ratio * advantage_decimal
    clipped = clipped_ratio * advantage_decimal
    selected = min(unclipped, clipped)
    raw_kl = reference_mean - policy_mean
    kl_difference = max(-clamp, min(clamp, raw_kl))
    kl = kl_difference.exp() - kl_difference - D1
    loss = -selected + kl_coefficient * kl
    return {
        "policy_mean_logp": policy_mean, "behavior_mean_logp": behavior_mean,
        "reference_mean_logp": reference_mean,
        "raw_current_minus_behavior_logp": raw_ratio,
        "current_minus_behavior_logp": log_ratio, "ratio": ratio,
        "clipped_ratio": clipped_ratio, "advantage": advantage_decimal,
        "unclipped_surrogate": unclipped, "clipped_surrogate": clipped,
        "selected_surrogate": selected, "clip_active": unclipped > clipped,
        "raw_reference_minus_current_logp": raw_kl,
        "reference_minus_current_logp": kl_difference,
        "schulman_kl": kl, "objective_loss": loss,
    }


def compare_metrics(actual, expected):
    maximum = D0
    for key, expected_value in expected.items():
        require(key in actual, f"missing objective metric: {key}")
        if key == "clip_active":
            require(actual[key] is expected_value, "clip-active mismatch")
            continue
        actual_value = Decimal(str(actual[key]))
        difference = abs(actual_value - expected_value)
        maximum = max(maximum, difference)
        require(difference <= OBJECTIVE_TOLERANCE,
                f"objective metric {key} differs by {difference}")
    return maximum


def verify_model(summary, results, model_dir, replay_binary, repo_root):
    provenance = summary["provenance"]
    require(provenance["model_sha256"] and provenance["config_sha256"]
            and provenance["tokenizer_sha256"], "model identity is incomplete")
    weight_map = provenance["available_weights"]
    require(len(weight_map) == 18, "not all 18 exact weight shards are retained")
    for relative, identity in weight_map.items():
        require(Path(relative).name == relative and not Path(relative).is_absolute(),
                "non-relative model weight path")
        require(file_identity(model_dir / relative) == identity,
                f"model weight identity mismatch: {relative}")
    require(sha256_file(model_dir / "config.json") == provenance["config_sha256"]
            and sha256_file(replay_binary) == provenance["binaries"]["native_replay"]["sha256"],
            "model config or independent replay binary identity mismatch")
    recomputed_weight_set = sha256_bytes(canonical(weight_map))
    require(recomputed_weight_set == provenance["model_sha256"], "weight-set digest mismatch")
    source_commit = summary["source_commit"]
    require(len(source_commit) == 40 and set(source_commit) <= set("0123456789abcdef"),
            "invalid source commit")
    required_sources = {
        "python/umminfer_online_grpo_optimizer.py",
        "python/umminfer_online_grpo_preflight.py",
        "python/umminfer_online_rollout.py", "python/umminfer_online_evidence_audit.py",
        "python/umminfer_rl_reward.py", "python/umminfer_ablation_metrics.py",
        "tests/online_grpo_optimizer_live.py", "tests/online_grpo_optimizer_verify.py",
        "tests/online_rollout_live.py", "native/qwen38_cli/main.cpp",
        "native/qwen38/engine.cu", "src/serve/online_attestation.cpp",
        "src/serve/openai_chat.cpp",
    }
    require(set(provenance["source_files"]) == required_sources,
            "optimizer source inventory is incomplete or extended")
    for relative, identity in provenance["source_files"].items():
        require(file_identity(repo_root / relative) == identity,
                f"retained source identity mismatch: {relative}")
    for result in results.values():
        model = result["run_identity"]["model"]
        require(sha256_bytes(canonical(result["run_identity"]))
                == result["run_identity_sha256"], "optimizer run identity digest mismatch")
        bound_files = result["run_identity"]["source"]["files"]
        require(set(bound_files) == {
            "python/umminfer_online_grpo_optimizer.py",
            "python/umminfer_online_grpo_preflight.py",
            "python/umminfer_online_rollout.py", "python/umminfer_online_evidence_audit.py",
            "python/umminfer_rl_reward.py", "python/umminfer_ablation_metrics.py"},
            "run-identity source inventory mismatch")
        require(bound_files == {name: provenance["source_files"][name]
                                for name in bound_files},
                "run-identity source bytes differ from verified sources")
        require(model["model_sha256"] == recomputed_weight_set
                and model["model_config_sha256"] == provenance["config_sha256"]
                and model["tokenizer_sha256"] == provenance["tokenizer_sha256"]
                and result["run_identity"]["source"]["commit"] == source_commit,
                "optimizer model/source identity mismatch")
    return {"weight_files_rehashed": 18, "weight_set_sha256": recomputed_weight_set,
            "config_sha256": provenance["config_sha256"],
            "tokenizer_sha256": provenance["tokenizer_sha256"],
            "replay_binary_sha256": provenance["binaries"]["native_replay"]["sha256"]}


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def close_arrays(actual, expected, label, *, atol=2e-7, rtol=2e-6):
    require(len(actual) == len(expected), f"{label} length mismatch")
    maximum = max((abs(left - right) for left, right in zip(actual, expected)), default=0.0)
    scale = max((abs(value) for value in expected), default=0.0)
    require(maximum <= max(atol, rtol * scale),
            f"{label} numerical mismatch: max_abs={maximum}")
    return maximum


def verify_update(run_dir, step, run_id, reference_document, preflight_groups,
                  preflight_objective=None, optimization=None):
    update_path = run_dir / f"update-{step:06d}.json"
    update = read_json(update_path)
    require(update["schema"] == "umminfer-online-grpo-lora-update-v1"
            and update["run_identity_sha256"] == run_id
            and update["step"] == step and update["schedule_step_index"] == step - 1
            and update["has_reward_variance"] is True and update["group_std"] > 0.0
            and update["parameters"]["trainable_count"] == 17408
            and update["parameters"]["trainable_tensors"] == 2
            and update["base_weights"] == "FROZEN_NO_GRAD_NO_OPTIMIZER_NO_VERSION_CHANGE",
            "invalid bounded update identity/surface")
    group = preflight_groups[step - 1]
    require(update["group_id"] == group["group_id"]
            and update["rollout_ids"] == [member["rollout_id"] for member in group["members"]]
            and [member["rollout_id"] for member in update["members"]]
            == update["rollout_ids"]
            and update["group_export_sha256"] == group["group_export_sha256"],
            "update whole-group/member/data-order mismatch")
    if preflight_objective is not None:
        require(update["objective"] == {key: float(preflight_objective[key])
                                        for key in ("clip_eps", "kl_coef", "log_ratio_clamp")},
                "update objective differs from audited preflight")
    if optimization is not None:
        require(update["adamw"] == {key: optimization[key] for key in (
            "learning_rate", "weight_decay", "beta1", "beta2", "epsilon",
            "max_grad_norm", "foreach")}, "update AdamW configuration differs from run identity")
    reference_members = {member["rollout_id"]: member
                         for item in reference_document["groups"] for member in item["members"]}
    run_members = {member["rollout_id"]: member for member in group["members"]}
    max_objective_delta = D0
    for member in update["members"]:
        rollout_id = member["rollout_id"]
        expected_member = run_members[rollout_id]
        current = member["current_logprobs"]
        behavior = member["sampled_behavior_logprobs"]
        reference = member["reference_logprobs"]
        mask = member["response_mask"]
        require(reference == reference_members[rollout_id]["logprobs"]
                and member["packed_ids_labels_masks_sha256"]
                == expected_member["packed_ids_labels_masks_sha256"]
                and member["source_record_sha256"] == expected_member["source_record_sha256"]
                and member["raw_sse_sha256"] == expected_member["raw_sse_sha256"]
                and member["advantage"] == expected_member["advantage"]
                and member["reward"] == expected_member["reward"],
                "member source/reference/packing identity mismatch")
        with localcontext() as context:
            context.prec = 50
            expected = decimal_metrics(current, behavior, reference, mask,
                                       member["advantage"], update["objective"])
            max_objective_delta = max(
                max_objective_delta, compare_metrics(member["metrics"], expected))
    with localcontext() as context:
        context.prec = 50
        expected_group_loss = decimal_mean([
            member["metrics"]["objective_loss"] for member in update["members"]])
        require(abs(Decimal(str(update["objective_loss"])) - expected_group_loss)
                <= OBJECTIVE_TOLERANCE, "rollout-mean group loss mismatch")

    documents = {
        "before": update["parameters"]["before"],
        "gradient": update["gradient"],
        "postclip": update["gradient"]["postclip"],
        "after": update["parameters"]["after"],
        "delta": update["parameters"]["delta"],
        "exp_avg": update["moments_after"]["exp_avg"],
        "exp_avg_sq": update["moments_after"]["exp_avg_sq"],
    }
    decoded = {name: read_f32(document, run_dir) for name, document in documents.items()}
    layouts = [decoded[name][2] for name in documents]
    require(all(layout == layouts[0] for layout in layouts), "tensor/AdamW layout drift")
    require([item["name"] for item in layouts[0]] == [
        "model.language_model.layers.3.self_attn.q_proj.lora_A.weight",
        "model.language_model.layers.3.self_attn.q_proj.lora_B.weight"],
        "trainable names are not the fixed Qwen projection LoRA")
    before = decoded["before"][1]
    gradient = decoded["gradient"][1]
    postclip = decoded["postclip"][1]
    after = decoded["after"][1]
    delta = decoded["delta"][1]
    exp_avg = decoded["exp_avg"][1]
    exp_avg_sq = decoded["exp_avg_sq"][1]
    require(len(before) == len(gradient) == len(postclip) == len(after) == len(delta)
            == len(exp_avg) == len(exp_avg_sq) == 17408,
            "raw trainable/AdamW surface is not exactly 17,408 FP32 scalars")
    nonzero_gradient = sum(value != 0.0 for value in gradient)
    require(nonzero_gradient > 0
            and nonzero_gradient == update["gradient"]["nonzero_scalars"],
            "finite nonzero gradient count mismatch")
    independent_grad_norm = math.sqrt(math.fsum(value * value for value in gradient))
    require(independent_grad_norm > 0.0 and math.isfinite(independent_grad_norm)
            and abs(independent_grad_norm - update["gradient"]["grad_norm"])
            <= max(1e-6, NORM_REL_TOLERANCE * independent_grad_norm),
            "gradient norm mismatch")
    coefficients = update["adamw"]
    max_grad_norm = coefficients["max_grad_norm"]
    retained_torch_norm = update["gradient"]["grad_norm"]
    clip_factor = min(1.0, max_grad_norm / (retained_torch_norm + 1e-6))
    expected_postclip = [f32(value * f32(clip_factor)) for value in gradient]
    clipping_error = close_arrays(
        postclip, expected_postclip, "postclip gradient", atol=2e-7, rtol=2e-6)
    if retained_torch_norm <= max_grad_norm:
        require(decoded["postclip"][0] == decoded["gradient"][0],
                "unclipped gradient bytes changed")
    if step == 1:
        previous_avg = [0.0] * 17408
        previous_avg_sq = [0.0] * 17408
    else:
        require(step == 2, "bounded optimizer has only two steps")
        first = read_json(run_dir / "update-000001.json")
        require(decoded["before"][0] == read_f32(
            first["parameters"]["after"], run_dir)[0],
            "step-2 before parameters differ from step-1 after bytes")
        previous_avg = read_f32(first["moments_after"]["exp_avg"], run_dir)[1]
        previous_avg_sq = read_f32(first["moments_after"]["exp_avg_sq"], run_dir)[1]
    require(update["moments_after"]["step"] == step
            and coefficients["weight_decay"] == 0.0
            and coefficients["foreach"] is False,
            "unsupported AdamW state/configuration")
    beta1 = coefficients["beta1"]
    beta2 = coefficients["beta2"]
    expected_avg = []
    expected_avg_sq = []
    for old_m, old_v, grad in zip(previous_avg, previous_avg_sq, postclip):
        # Torch's single-tensor AdamW uses exp_avg.lerp_(grad, 1-beta1) and
        # exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2).
        expected_avg.append(f32(old_m + f32(f32(1.0 - beta1) * f32(grad - old_m))))
        expected_avg_sq.append(f32(f32(old_v * beta2)
                                   + f32(f32(1.0 - beta2) * f32(grad * grad))))
    avg_error = close_arrays(exp_avg, expected_avg, "AdamW first moment")
    avg_sq_error = close_arrays(exp_avg_sq, expected_avg_sq, "AdamW second moment")
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step
    step_size = coefficients["learning_rate"] / bias_correction1
    bias_correction2_sqrt = f32(math.sqrt(bias_correction2))
    expected_after = []
    for parameter, avg, avg_sq in zip(before, exp_avg, exp_avg_sq):
        denom = f32(f32(f32(math.sqrt(avg_sq)) / bias_correction2_sqrt)
                    + f32(coefficients["epsilon"]))
        expected_after.append(f32(parameter + f32(-step_size * f32(avg / denom))))
    adamw_error = close_arrays(after, expected_after, "AdamW parameter update")
    expected_delta = struct.pack(
        "<" + "f" * len(before), *(after_value - before_value
                                   for before_value, after_value in zip(before, after)))
    require(expected_delta == decoded["delta"][0], "raw parameter delta bytes mismatch")
    changed = sum(value != 0.0 for value in delta)
    max_delta = max(abs(value) for value in delta)
    delta_l2 = math.sqrt(math.fsum(value * value for value in delta))
    require(changed > 0 and changed == update["parameters"]["changed_scalars"]
            and max_delta == update["parameters"]["max_abs_delta"]
            and abs(delta_l2 - update["parameters"]["delta_l2"])
            <= max(1e-12, 1e-12 * delta_l2),
            "changed parameter/max-delta/L2 mismatch")
    return {"step": step, "update_sha256": sha256_file(update_path),
            "gradient_sha256": documents["gradient"]["sha256"],
            "postclip_gradient_sha256": documents["postclip"]["sha256"],
            "before_sha256": documents["before"]["sha256"],
            "after_sha256": documents["after"]["sha256"],
            "delta_sha256": documents["delta"]["sha256"],
            "exp_avg_sha256": documents["exp_avg"]["sha256"],
            "exp_avg_sq_sha256": documents["exp_avg_sq"]["sha256"],
            "gradient_norm_independent": independent_grad_norm,
            "nonzero_gradient_scalars": nonzero_gradient,
            "changed_parameter_scalars": changed, "max_abs_parameter_delta": max_delta,
            "parameter_delta_l2_independent": delta_l2,
            "adamw_recomputation_max_abs_errors": {
                "postclip": clipping_error, "exp_avg": avg_error,
                "exp_avg_sq": avg_sq_error, "after": adamw_error},
            "objective_max_abs_delta": float(max_objective_delta)}


def verify_native_replays(root):
    config = read_json(root / "preflight/config.json")
    identity = read_json(root / "preflight/run_identity.json")["identity"]
    fixtures = read_json(root / "preflight/objective_fixtures.json")["groups"]
    fixture_members = {member["rollout_id"]: (group, member)
                       for group in fixtures for member in group["members"]}
    max_delta = D0
    count = 0
    for group in identity["groups"]:
        prompt_ids = config["prompt_token_ids"][group["prompt_sha256"]]
        require(sha256_bytes(canonical(prompt_ids)) == group["prompt_token_ids_sha256"],
                "native prompt-token set hash mismatch")
        for member in group["members"]:
            rollout_id = member["rollout_id"]
            document = read_json(root / "forwards/behavior" / (rollout_id + ".json"))
            evidence = document["independent_behavior"]
            require(evidence["producer_scoring_used"] is False
                    and document["prompt_token_ids"] == prompt_ids,
                "native-rendered prompt ids or scoring independence mismatch")
            fixture_group, fixture = fixture_members[rollout_id]
            require(fixture_group["group_id"] == group["group_id"]
                    and evidence["token_ids"] == fixture["packing"]["sampled_response_token_ids"]
                    and evidence["mask"] == fixture["packing"]["response_mask"][
                        len(prompt_ids):],
                    "independent native sampled ids/masks differ from packing")
            sampled = fixture["behavior_policy"]["sampled_logprobs"]
            require(len(evidence["rows"]) == len(sampled) == len(evidence["logprobs"]),
                    "independent native raw rows are incomplete")
            for row, independent, expected in zip(
                    evidence["rows"], evidence["logprobs"], sampled):
                with localcontext() as context:
                    context.prec = 50
                    recomputed = (Decimal(str(row["selected_raw_logit"]))
                                  - Decimal(str(row["max_raw_logit"]))
                                  - Decimal(str(row["log_normalizer"])))
                    require(abs(recomputed - Decimal(str(row["logprob"])))
                            <= Decimal("1e-9"), "native raw-row log-softmax mismatch")
                require(row["logprob"] == independent, "native row/logprob array mismatch")
                delta = abs(Decimal(str(independent)) - Decimal(str(expected)))
                max_delta = max(max_delta, delta)
                require(delta <= Decimal("1e-9"), "native/sampled behavior logprob mismatch")
            count += 1
    require(count == 6, "not all six fresh native replays were verified")
    return {"cold_processes": count, "sampled_tokens": sum(
        len(member["behavior_policy"]["sampled_logprobs"])
        for group in fixtures for member in group["members"]),
            "max_abs_sampled_logprob_delta": float(max_delta), "tolerance": 1e-9}


def verify(root, model_dir, replay_binary, repo_root):
    root = Path(root)
    summary = read_json(root / "summary.json")
    require(summary["status"] == "PASS" and summary["plan"]["groups"] == 2
            and summary["plan"]["seeds_per_group"] == 3
            and summary["claims"]["optimizer"] == "EXECUTED_BOUNDED_LORA"
            and summary["claims"]["model_quality"] == "NOT_RUN"
            and summary["claims"]["generalized_improvement"] == "NOT_RUN",
            "gate status/claims are not the bounded PASS")
    with tempfile.TemporaryDirectory(prefix="umm-online-optimizer-independent-audit-") as tmp:
        audit_path = Path(tmp) / "audit.json"
        command = [
            sys.executable, "-I", str(repo_root / "python/umminfer_online_evidence_audit.py"),
            "preflight", "--online-pack", str(root / "accepted"),
            "--preflight-bundle", str(root / "preflight"),
            "--tokenizer-dir", str(model_dir), "--repo-root", str(repo_root),
            "--report", str(audit_path),
        ]
        result = subprocess.run(command, cwd=repo_root, capture_output=True,
                                text=True, timeout=1800, check=False)
        require(result.returncode == 0 and audit_path.is_file(),
                "independent raw-SSE/CAS/export/preflight auditor failed")
        audit = read_json(audit_path)
        require(audit["status"] == "PASS" and audit.get("checkpoint_disagreements") == [],
                "independent audit has disagreements")
    run_dirs = {name: root / "optimizer" / name
                for name in ("uninterrupted", "split-first", "resumed")}
    results = {name: read_json(path / "result.json") for name, path in run_dirs.items()}
    model_report = verify_model(summary, results, model_dir, replay_binary, repo_root)
    run_ids = {result["run_identity_sha256"] for result in results.values()}
    require(len(run_ids) == 1, "optimizer run identities drifted")
    run_id = next(iter(run_ids))
    references = {name: read_json(path / "reference.json") for name, path in run_dirs.items()}
    require(references["uninterrupted"] == references["split-first"]
            == references["resumed"]
            and references["uninterrupted"]["run_identity_sha256"] == run_id,
            "frozen reference cache differs")
    preflight_run_document = read_json(root / "preflight/run_identity.json")
    preflight_identity = preflight_run_document["identity"]
    require(preflight_identity["token_export_alignment"] == "PASS_INDEPENDENT_STRUCTURAL"
            and all(group["split"] == "train" and group["has_reward_variance"]
                    for group in preflight_identity["groups"]),
            "preflight groups are not strong train-only nonzero variance")
    native_report = verify_native_replays(root)
    preflight_manifest_sha = sha256_file(root / "preflight/manifest.json")
    rollout_manifest_sha = sha256_file(root / "accepted/rollout_manifest.json")
    for result in results.values():
        identity = result["run_identity"]
        require(identity["preflight_run_id"] == preflight_run_document["run_id"]
                and identity["preflight_manifest_sha256"] == preflight_manifest_sha
                and identity["online_rollout_manifest_sha256"] == rollout_manifest_sha,
                "optimizer/preflight/rollout manifest binding mismatch")
    require(results["uninterrupted"]["initial_step"] == 0
            and results["uninterrupted"]["executed_step_count"] == 2
            and results["split-first"]["initial_step"] == 0
            and results["split-first"]["executed_step_count"] == 1
            and results["resumed"]["initial_step"] == 1
            and results["resumed"]["executed_step_count"] == 1,
            "uninterrupted/split/resume cursors are invalid")
    expected_optimization = results["uninterrupted"]["run_identity"]["optimization"]
    require({key: expected_optimization[key] for key in (
        "learning_rate", "weight_decay", "beta1", "beta2", "epsilon",
        "max_grad_norm", "foreach")} == {
            "learning_rate": 0.001, "weight_decay": 0.0, "beta1": 0.9,
            "beta2": 0.999, "epsilon": 1e-8, "max_grad_norm": 1.0,
            "foreach": False}, "bounded AdamW contract differs from the frozen gate")
    checks = [verify_update(run_dirs["uninterrupted"], step, run_id,
                            references["uninterrupted"], preflight_identity["groups"],
                            preflight_identity["objective"],
                            results["uninterrupted"]["run_identity"]["optimization"])
              for step in (1, 2)]
    comparisons = []
    for step, other in ((1, "split-first"), (2, "resumed")):
        left = run_dirs["uninterrupted"]
        right = run_dirs[other]
        for relative in (f"update-{step:06d}.json", "reference.json",
                         f"checkpoints/step-{step:06d}.pt",
                         f"checkpoints/step-{step:06d}.pt.sha256"):
            require((left / relative).read_bytes() == (right / relative).read_bytes(),
                    f"cross-run byte mismatch: step={step} {relative}")
            comparisons.append({"step": step, "other": other, "path": relative,
                                "sha256": sha256_file(left / relative), "byte_identical": True})
        for path in sorted((left / "tensors").glob(f"step-{step:06d}-*.f32")):
            require(path.read_bytes() == (right / "tensors" / path.name).read_bytes(),
                    f"cross-run raw tensor mismatch: {path.name}")
    resumed_checkpoint = run_dirs["split-first"] / "checkpoints/step-000001.pt"
    require(results["resumed"]["resume"]["checkpoint_sha256"]
            == sha256_file(resumed_checkpoint), "resumed source checkpoint identity mismatch")
    for name, run_dir in run_dirs.items():
        for retained in results[name]["updates"]:
            update_path = run_dir / retained["path"]
            require(file_identity(update_path) == {"bytes": retained["bytes"],
                                                   "sha256": retained["sha256"]},
                    f"optimizer {name} retained update identity mismatch")
        for retained in results[name]["checkpoints"]:
            checkpoint_path = run_dir / retained["path"]
            require(file_identity(checkpoint_path) == {"bytes": retained["bytes"],
                                                       "sha256": retained["sha256"]},
                    f"optimizer {name} retained checkpoint identity mismatch")
            sidecar = Path(str(checkpoint_path) + ".sha256")
            require(sidecar.read_text(encoding="ascii").strip() == retained["sha256"],
                    "checkpoint sidecar mismatch")
        checked = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=run_dir,
                                 capture_output=True, text=True, timeout=120, check=False)
        require(checked.returncode == 0, f"optimizer {name} SHA256SUMS failed")
    max_delta = max(Decimal(str(check["objective_max_abs_delta"])) for check in checks)
    fixtures = read_json(root / "preflight/objective_fixtures.json")["groups"]
    sampled_tokens = sum(len(member["behavior_policy"]["sampled_logprobs"])
                         for group in fixtures for member in group["members"])
    supervised_tokens = sum(sum(member["packing"]["response_mask"][
        -len(member["behavior_policy"]["sampled_logprobs"]):])
        for group in fixtures for member in group["members"])
    directive_zero_tokens = sampled_tokens - supervised_tokens
    derived_groups = [{
        "dataset_group_id": group["group_id"],
        "rewards": [member["reward"] for member in group["members"]],
        "group_std": group["group_std"],
    } for group in preflight_identity["groups"]]
    require(summary["token_counts"] == {"sampled": sampled_tokens,
                                        "supervised": supervised_tokens,
                                        "directive_zero": directive_zero_tokens}
            and [{"dataset_group_id": group["dataset_group_id"], "rewards": group["rewards"],
                  "group_std": group["group_std"]} for group in summary["groups"]]
            == derived_groups, "summary reward/token counts differ from audited evidence")
    return {
        "schema": "umminfer-online-grpo-optimizer-independent-verification-v1",
        "status": "PASS", "run_identity_sha256": run_id,
        "source_commit": summary["source_commit"],
        "groups": 2, "rollouts": 6,
        "sampled_tokens": sampled_tokens,
        "supervised_tokens": supervised_tokens,
        "directive_zero_tokens": directive_zero_tokens,
        "group_reward_variance": derived_groups,
        "model": model_report, "independent_native_replays": native_report,
        "objective_max_abs_delta": float(max_delta),
        "objective_tolerance": float(OBJECTIVE_TOLERANCE),
        "objective_arithmetic": "DECIMAL_50_DIGIT_MASKED_MEANS_CLIPPED_RATIO_SEPARATE_REFERENCE_KL",
        "gradient_and_delta_checks": checks,
        "deterministic_resume": {
            "step_1_uninterrupted_vs_split_first": "BYTE_IDENTICAL",
            "step_2_uninterrupted_vs_fresh_resume": "BYTE_IDENTICAL",
            "comparisons": comparisons,
        },
        "independent_raw_preflight_audit": {
            "status": "PASS", "checkpoint_disagreements": 0,
            "report_sha256": sha256_bytes(canonical(audit)),
        },
        "imports": "STDLIB_ONLY_NO_PRODUCER_SCORING_OR_ML",
        "claims": {"model_quality": "NOT_RUN", "generalized_improvement": "NOT_RUN",
                   "base_weights_updated": False, "trainable_parameters": 17408},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--replay-binary", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        require(not args.report.exists(), "refusing to overwrite verification report")
        report = verify(args.root, args.model_dir, args.replay_binary, args.repo_root)
        payload = canonical(report) + b"\n"
        temporary = args.report.with_name(args.report.name + ".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, args.report)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.exit(2, f"independent optimizer verification error: {error}\n")
    print("INDEPENDENT ONLINE GRPO OPTIMIZER: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
