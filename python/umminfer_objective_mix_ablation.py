#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage C Ablation #5: Post-Training Objective-Mix & Quality-vs-Latency Pareto Frontier.

Realizes Row 5 of docs/vision.md section 3.2 as a standalone, single-variable
4-arm post-training objective-mix ablation:
  1. sft_only                     (sft_weight=1.0, grpo_weight=0.0, reward_weights=null)
  2. sft_plus_consistency         (sft_weight=1.0, grpo_weight=1.0, balance=0.6, barge_in=0.4)
  3. sft_plus_routing             (sft_weight=1.0, grpo_weight=1.0, routing_f1=1.0)
  4. sft_plus_full_latency_pareto (sft_weight=1.0, grpo_weight=1.0, routing=0.4, balance=0.3,
                                   barge_in=0.3, latency_penalty=0.2)

Pure Python 3 stdlib + ``umminfer_rl_reward`` at module load time.
``torch`` and ``transformers`` are imported ONLY lazily inside ``execute_live_gpu_ablation``
when ``--live-gpu`` is explicitly selected. Zero imports from C++ serving code.

Usage:
  # Deterministic CPU-only contract replay (no torch/transformers required):
  python3 -I python/umminfer_objective_mix_ablation.py \
    --config configs/training/objective_mix_ablation.json \
    --offline-contract \
    --out /tmp/umm-obj-mix-contract

  # Real SM90 GPU 4-arm execution:
  python3 -I python/umminfer_objective_mix_ablation.py \
    --config configs/training/objective_mix_ablation.json \
    --live-gpu \
    --model-dir /path/to/local/Qwen2.5-0.5B-Instruct \
    --gpu 0 \
    --out /tmp/umm-obj-mix-live
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

from umminfer_rl_reward import CompositeUmmReward  # noqa: E402

CONFIG_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
CONFIG_TYPE = "objective_mix_ablation"
VARIABLE_NAME = "objective_mix"

EXPECTED_ARM_IDS = (
    "sft_only",
    "sft_plus_consistency",
    "sft_plus_routing",
    "sft_plus_full_latency_pareto",
)
EXPECTED_REWARD_KEYS = frozenset(
    ("routing_f1", "prose_visual_balance", "barge_in_stability", "latency_penalty")
)
EXPECTED_CANDIDATE_PROFILES = (
    "optimal_balanced_fast",
    "overtriggered_redundant_slow",
    "undertriggered_text_fast",
)
IDLE_GPU_MAX_USED_MIB = 100


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def canonical_json(value: object) -> bytes:
    """Deterministic UTF-8 canonical JSON encoding (sorted keys, compact separators)."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def pretty_json(value: object) -> bytes:
    """Deterministic pretty-printed JSON with trailing newline."""
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def write_sha256sums(out_dir: Path, rel_paths: list[str]) -> Path:
    """Write a sorted SHA256SUMS file covering all listed relative paths."""
    lines = []
    for rel in sorted(set(rel_paths)):
        target = out_dir / rel
        if not target.is_file():
            raise ValueError(f"cannot hash missing output artifact: {rel}")
        lines.append(f"{sha256_file(target)}  {rel}")
    manifest_path = out_dir / "SHA256SUMS"
    _atomic_write_bytes(manifest_path, ("\n".join(lines) + "\n").encode("ascii"))
    return manifest_path


def verify_sha256sums(out_dir: Path) -> list[str]:
    """Verify every entry in SHA256SUMS against on-disk files; fail closed."""
    manifest_path = out_dir / "SHA256SUMS"
    if not manifest_path.is_file():
        raise ValueError("missing SHA256SUMS")
    lines = [
        line
        for line in manifest_path.read_text(encoding="ascii").splitlines()
        if line.strip()
    ]
    if not lines:
        raise ValueError("SHA256SUMS is empty")
    verified: list[str] = []
    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise ValueError(f"malformed SHA256SUMS line: {line}")
        expected_hex, rel = parts
        target = out_dir / rel
        if not target.is_file():
            raise ValueError(f"missing hashed file in SHA256SUMS: {rel}")
        actual_hex = sha256_file(target)
        if actual_hex != expected_hex:
            raise ValueError(f"SHA-256 mismatch for {rel}")
        verified.append(rel)
    if verified != sorted(verified):
        raise ValueError("SHA256SUMS entries are not sorted")
    return verified


def _validate_reward_weights(weights: object, *, allow_null: bool) -> dict[str, float] | None:
    if weights is None:
        if allow_null:
            return None
        raise ValueError("reward_weights cannot be null for a GRPO-enabled arm")
    if not isinstance(weights, dict):
        raise ValueError("reward_weights must be a JSON object")
    if set(weights.keys()) != EXPECTED_REWARD_KEYS:
        raise ValueError(
            f"reward_weights keys must match {sorted(EXPECTED_REWARD_KEYS)}"
        )
    out: dict[str, float] = {}
    for key in sorted(EXPECTED_REWARD_KEYS):
        val = weights[key]
        if not _is_finite_number(val) or float(val) < 0.0:
            raise ValueError(f"reward weight {key} must be finite and non-negative")
        out[key] = float(val)
    # Validate via CompositeUmmReward constructor (ensures positive quality weight)
    CompositeUmmReward(weights=out)
    return out


def _validate_arm(arm: object, expected_arm_id: str) -> dict:
    if not isinstance(arm, dict):
        raise ValueError("each arm entry must be a JSON object")
    arm_id = arm.get("arm_id")
    if arm_id != expected_arm_id:
        raise ValueError(
            f"arm_id mismatch: expected {expected_arm_id!r}, got {arm_id!r}"
        )
    model = arm.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"arm {arm_id} missing model block")
    if (
        model.get("dtype") != "bfloat16"
        or model.get("attn_implementation") != "eager"
        or model.get("gradient_checkpointing") is not False
        or model.get("trust_remote_code") is not False
        or not isinstance(model.get("base_model_placeholder"), str)
        or not model["base_model_placeholder"]
    ):
        raise ValueError(f"arm {arm_id} has invalid model configuration")

    opt = arm.get("optimization")
    if not isinstance(opt, dict):
        raise ValueError(f"arm {arm_id} missing optimization block")
    if (
        not isinstance(opt.get("seed"), str)
        or not opt["seed"]
        or not _is_finite_number(opt.get("learning_rate"))
        or float(opt["learning_rate"]) <= 0.0
        or not _is_finite_number(opt.get("weight_decay"))
        or float(opt["weight_decay"]) < 0.0
        or not _is_finite_number(opt.get("max_grad_norm"))
        or float(opt["max_grad_norm"]) <= 0.0
        or not _is_int(opt.get("max_seq_len"))
        or opt["max_seq_len"] < 32
        or not _is_int(opt.get("steps"))
        or opt["steps"] < 1
    ):
        raise ValueError(f"arm {arm_id} has invalid optimization configuration")

    grpo = arm.get("grpo")
    if not isinstance(grpo, dict):
        raise ValueError(f"arm {arm_id} missing grpo block")
    if (
        not _is_finite_number(grpo.get("clip_eps"))
        or not (0.0 < float(grpo["clip_eps"]) < 1.0)
        or not _is_finite_number(grpo.get("kl_coef"))
        or float(grpo["kl_coef"]) < 0.0
        or not _is_finite_number(grpo.get("log_ratio_clamp"))
        or float(grpo["log_ratio_clamp"]) <= 0.0
        or not _is_finite_number(grpo.get("advantage_eps"))
        or float(grpo["advantage_eps"]) < 0.0
        or grpo.get("zero_std_policy") != "zero_advantage"
    ):
        raise ValueError(f"arm {arm_id} has invalid grpo configuration")

    sft_weight = arm.get("sft_weight")
    if not _is_finite_number(sft_weight) or float(sft_weight) <= 0.0:
        raise ValueError(f"arm {arm_id} sft_weight must be finite and positive")

    obj = arm.get("objective_mix")
    if not isinstance(obj, dict):
        raise ValueError(f"arm {arm_id} missing objective_mix block")
    obj_sft = obj.get("sft_weight")
    obj_grpo = obj.get("grpo_weight")
    if not _is_finite_number(obj_sft) or not math.isclose(
        float(obj_sft), float(sft_weight), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"arm {arm_id} objective_mix.sft_weight differs from arm.sft_weight")
    if not _is_finite_number(obj_grpo) or float(obj_grpo) < 0.0:
        raise ValueError(f"arm {arm_id} objective_mix.grpo_weight must be finite and >= 0")

    if arm_id == "sft_only":
        if float(obj_grpo) != 0.0 or obj.get("reward_weights") is not None:
            raise ValueError("sft_only arm must have grpo_weight=0.0 and reward_weights=null")
    else:
        if float(obj_grpo) <= 0.0:
            raise ValueError(f"arm {arm_id} must have grpo_weight > 0")
        _validate_reward_weights(obj.get("reward_weights"), allow_null=False)

    return arm


def _validate_rollout_groups(groups: object, latency_budget_ms: dict[str, float]) -> list[dict]:
    if not isinstance(groups, list) or len(groups) < 3:
        raise ValueError("rollout_groups must contain at least 3 prompt groups")
    seen_group_ids: set[str] = set()
    seen_candidate_ids: set[str] = set()
    ref_reward = CompositeUmmReward(latency_budget_ms=latency_budget_ms)
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("each rollout group must be a JSON object")
        gid = group.get("group_id")
        prompt_text = group.get("prompt_text")
        candidates = group.get("candidates")
        if not isinstance(gid, str) or not gid or gid in seen_group_ids:
            raise ValueError(f"invalid or duplicate group_id: {gid!r}")
        seen_group_ids.add(gid)
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ValueError(f"group {gid} lacks non-empty prompt_text")
        if not isinstance(candidates, list) or len(candidates) < 3:
            raise ValueError(f"group {gid} must contain at least 3 candidate rollouts")
        profiles: set[str] = set()
        targets: set[str] = set()
        for cand in candidates:
            if not isinstance(cand, dict):
                raise ValueError(f"group {gid} contains a non-object candidate")
            cid = cand.get("candidate_id")
            if not isinstance(cid, str) or not cid or cid in seen_candidate_ids:
                raise ValueError(f"invalid or duplicate candidate_id: {cid!r}")
            seen_candidate_ids.add(cid)
            profile = cand.get("candidate_profile")
            if profile not in EXPECTED_CANDIDATE_PROFILES:
                raise ValueError(f"candidate {cid} has unsupported profile {profile!r}")
            profiles.add(profile)
            target = cand.get("assistant_target")
            if not isinstance(target, str) or not target.strip():
                raise ValueError(f"candidate {cid} lacks non-empty assistant_target")
            if target in targets:
                raise ValueError(f"group {gid} has duplicate assistant_target")
            targets.add(target)
            traj = cand.get("trajectory")
            if not isinstance(traj, dict):
                raise ValueError(f"candidate {cid} lacks schema-v2 trajectory object")
            score, components = ref_reward.evaluate(traj)
            if score is None or components is None or not math.isfinite(score):
                raise ValueError(
                    f"candidate {cid} trajectory is not scorable by CompositeUmmReward"
                )
            latency_ms = components.get("latency_ms") or {}
            if not _is_finite_number(latency_ms.get("terminal_ms")) or float(
                latency_ms["terminal_ms"]
            ) < 0.0:
                raise ValueError(f"candidate {cid} lacks valid ablation.latency.terminal_ms")
        if profiles != set(EXPECTED_CANDIDATE_PROFILES):
            raise ValueError(
                f"group {gid} must cover all 3 candidate profiles: "
                f"{list(EXPECTED_CANDIDATE_PROFILES)}"
            )
    return groups


def verify_single_variable_invariance(config: dict) -> dict:
    """Verify all fixed fields excluding objective_mix are identical across all 4 arms.

    Fails closed (raises ValueError) on any fixed-field drift or duplicate objective_mix.
    """
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    if config.get("config_schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported config_schema_version")
    if config.get("config_type") != CONFIG_TYPE:
        raise ValueError(f"config_type must be {CONFIG_TYPE!r}")
    if config.get("variable_name") != VARIABLE_NAME:
        raise ValueError(f"variable_name must be {VARIABLE_NAME!r}")

    evaluation = config.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("config missing evaluation block")
    quality_weights = _validate_reward_weights(
        evaluation.get("quality_reference_weights"), allow_null=False
    )
    if quality_weights["latency_penalty"] != 0.0:
        raise ValueError("quality_reference_weights must set latency_penalty=0.0")
    full_weights = _validate_reward_weights(
        evaluation.get("full_reward_reference_weights"), allow_null=False
    )
    latency_budget = evaluation.get("latency_budget_ms")
    if not isinstance(latency_budget, dict):
        raise ValueError("evaluation.latency_budget_ms must be a JSON object")
    CompositeUmmReward(weights=full_weights, latency_budget_ms=latency_budget)
    proj_scale = evaluation.get("offline_projection_step_scale")
    if not _is_finite_number(proj_scale) or float(proj_scale) <= 0.0:
        raise ValueError("evaluation.offline_projection_step_scale must be finite and > 0")

    groups = _validate_rollout_groups(config.get("rollout_groups"), latency_budget)
    rollout_groups_sha256 = sha256_bytes(canonical_json(groups))

    arms = config.get("arms")
    if not isinstance(arms, list) or len(arms) != len(EXPECTED_ARM_IDS):
        raise ValueError(f"arms must contain exactly {len(EXPECTED_ARM_IDS)} ordered arms")

    fixed_hashes: dict[str, str] = {}
    objective_hashes: dict[str, str] = {}
    for arm, expected_id in zip(arms, EXPECTED_ARM_IDS):
        validated = _validate_arm(arm, expected_id)
        # Every field except arm_id, description, and objective_mix must stay strictly identical.
        fixed_payload = {
            "model": validated["model"],
            "optimization": validated["optimization"],
            "grpo": validated["grpo"],
            "sft_weight": float(validated["sft_weight"]),
            "evaluation": evaluation,
            "rollout_groups_sha256": rollout_groups_sha256,
        }
        fixed_hashes[expected_id] = sha256_bytes(canonical_json(fixed_payload))
        objective_hashes[expected_id] = sha256_bytes(
            canonical_json(validated["objective_mix"])
        )

    unique_fixed = set(fixed_hashes.values())
    if len(unique_fixed) != 1:
        raise ValueError(
            "single-variable violation: fixed fields differ across arms: "
            + json.dumps(fixed_hashes, sort_keys=True)
        )
    if len(set(objective_hashes.values())) != len(EXPECTED_ARM_IDS):
        raise ValueError(
            "single-variable violation: objective_mix must be unique across all 4 arms"
        )

    return {
        "single_variable_verified": True,
        "variable_name": VARIABLE_NAME,
        "fixed_fields_sha256": next(iter(unique_fixed)),
        "rollout_groups_sha256": rollout_groups_sha256,
        "per_arm_fixed_fields_sha256": fixed_hashes,
        "per_arm_objective_mix_sha256": objective_hashes,
        "arm_ids": list(EXPECTED_ARM_IDS),
    }


def load_and_validate_config(config_path: Path) -> tuple[dict, bytes, dict]:
    if not config_path.is_file():
        raise ValueError(f"config file does not exist: {config_path}")
    raw = config_path.read_bytes()
    config = json.loads(raw.decode("utf-8"))
    invariance = verify_single_variable_invariance(config)
    return config, raw, invariance


def compute_grpo_advantages(
    scores: list[float], *, advantage_eps: float = 0.0
) -> tuple[list[float], float, float]:
    """Compute zero-mean unit-std group advantages (zero vector when std == 0)."""
    if not scores:
        raise ValueError("scores cannot be empty")
    if any(not _is_finite_number(s) for s in scores):
        raise ValueError("all scores must be finite numbers")
    vals = [float(s) for s in scores]
    mean = sum(vals) / len(vals)
    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(variance)
    if std <= 0.0:
        return [0.0 for _ in vals], mean, 0.0
    denom = std + float(advantage_eps)
    return [(v - mean) / denom for v in vals], mean, std


def score_rollouts_for_arm(config: dict, arm: dict) -> list[dict]:
    """Evaluate all rollout groups with CompositeUmmReward for one arm.

    Always computes reference quality (latency_penalty=0) and full composite reward
    alongside the arm's own objective reward & normalized GRPO group advantages.
    """
    evaluation = config["evaluation"]
    latency_budget = evaluation["latency_budget_ms"]
    quality_reward = CompositeUmmReward(
        weights=evaluation["quality_reference_weights"],
        latency_budget_ms=latency_budget,
    )
    full_reward = CompositeUmmReward(
        weights=evaluation["full_reward_reference_weights"],
        latency_budget_ms=latency_budget,
    )
    arm_weights = arm["objective_mix"]["reward_weights"]
    arm_reward = (
        CompositeUmmReward(weights=arm_weights, latency_budget_ms=latency_budget)
        if arm_weights is not None
        else None
    )
    adv_eps = float(arm["grpo"]["advantage_eps"])

    scored_groups: list[dict] = []
    for group in config["rollout_groups"]:
        cand_entries: list[dict] = []
        arm_scores: list[float] = []
        for cand in group["candidates"]:
            traj = cand["trajectory"]
            q_score, q_comp = quality_reward.evaluate(traj)
            f_score, _ = full_reward.evaluate(traj)
            if q_score is None or q_comp is None or f_score is None:
                raise ValueError(f"failed to score candidate {cand['candidate_id']}")
            if arm_reward is not None:
                a_score, a_comp = arm_reward.evaluate(traj)
                if a_score is None or a_comp is None:
                    raise ValueError(
                        f"failed to score candidate {cand['candidate_id']} for arm {arm['arm_id']}"
                    )
                arm_scores.append(float(a_score))
            else:
                a_score = None
            terminal_ms = float(q_comp["latency_ms"]["terminal_ms"])
            cand_entries.append(
                {
                    "candidate_id": cand["candidate_id"],
                    "candidate_profile": cand["candidate_profile"],
                    "full_composite_quality": float(q_score),
                    "full_composite_reward_with_latency": float(f_score),
                    "arm_reward_score": float(a_score) if a_score is not None else None,
                    "terminal_ms": terminal_ms,
                    "components": {
                        "routing_f1": float(q_comp["routing_f1"]),
                        "prose_visual_balance": float(q_comp["prose_visual_balance"]),
                        "barge_in_stability": float(q_comp["barge_in_stability"]),
                        "latency_penalty": float(q_comp["latency_penalty"]),
                        "latency_ms": dict(q_comp["latency_ms"]),
                    },
                }
            )
        if arm_reward is not None:
            advantages, mean, std = compute_grpo_advantages(
                arm_scores, advantage_eps=adv_eps
            )
        else:
            advantages = [0.0 for _ in cand_entries]
            mean = None
            std = None
        for entry, adv in zip(cand_entries, advantages):
            entry["advantage"] = float(adv)
        scored_groups.append(
            {
                "group_id": group["group_id"],
                "prompt_text": group["prompt_text"],
                "group_reward_mean": mean,
                "group_reward_std": std,
                "candidates": cand_entries,
            }
        )
    return scored_groups


def softmax(values: list[float]) -> list[float]:
    if not values:
        raise ValueError("softmax requires a non-empty list")
    if any(not _is_finite_number(v) for v in values):
        raise ValueError("softmax inputs must be finite")
    peak = max(float(v) for v in values)
    exps = [math.exp(float(v) - peak) for v in values]
    total = sum(exps)
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("softmax normalization failed")
    return [e / total for e in exps]


def weighted_quantile_linear(
    values: list[float], weights: list[float], q: float
) -> float:
    """Continuous piecewise-linear weighted quantile on [0, 1].

    Given sorted pairs (v_k, w_k) with cumulative weights F_k = sum_{j<=k} w_j
    (F_0 = 0, F_n = 1):
      - If q <= F_1: returns v_1.
      - Else for unique k >= 2 with F_{k-1} < q <= F_k:
        returns v_{k-1} + ((q - F_{k-1}) / (F_k - F_{k-1})) * (v_k - v_{k-1}).

    Strictly responsive to policy weight shifts in the upper tail (including q=0.95).
    """
    if not values or len(values) != len(weights):
        raise ValueError("values and weights must be non-empty and equal length")
    if not _is_finite_number(q) or not (0.0 <= float(q) <= 1.0):
        raise ValueError("quantile q must be in [0, 1]")
    pairs = []
    total_w = 0.0
    for val, w in zip(values, weights):
        if not _is_finite_number(val) or not _is_finite_number(w) or float(w) < 0.0:
            raise ValueError("values must be finite and weights must be finite and >= 0")
        total_w += float(w)
        pairs.append((float(val), float(w)))
    if total_w <= 0.0:
        raise ValueError("sum of weights must be strictly positive")

    # Collapse identical values so zero-width steps stay clean
    pairs.sort(key=lambda item: item[0])
    merged: list[tuple[float, float]] = []
    for val, w in pairs:
        if w == 0.0:
            continue
        nw = w / total_w
        if merged and merged[-1][0] == val:
            merged[-1] = (val, merged[-1][1] + nw)
        else:
            merged.append((val, nw))
    if not merged:
        raise ValueError("no positive-weight observations")
    if len(merged) == 1:
        return merged[0][0]

    target = float(q)
    prev_cdf = 0.0
    for idx, (val, w) in enumerate(merged):
        cdf = min(1.0, prev_cdf + w)
        if idx == 0:
            if target <= cdf:
                return val
        else:
            if target <= cdf or idx == len(merged) - 1:
                span = cdf - prev_cdf
                frac = 0.0 if span <= 0.0 else max(0.0, min(1.0, (target - prev_cdf) / span))
                prev_val = merged[idx - 1][0]
                return prev_val + frac * (val - prev_val)
        prev_cdf = cdf
    return merged[-1][0]


def evaluate_policy_metrics(
    scored_groups: list[dict],
    baseline_logps: list[list[float]],
    post_logps: list[list[float]],
) -> dict:
    """Compute policy-weighted quality, P95 terminal latency, and profile shifts."""
    if len(scored_groups) != len(baseline_logps) or len(scored_groups) != len(post_logps):
        raise ValueError("logprob group count mismatch")

    group_reports: list[dict] = []
    pooled_latencies: list[float] = []
    pooled_weights_post: list[float] = []
    pooled_weights_base: list[float] = []
    group_qualities_post: list[float] = []
    group_qualities_base: list[float] = []
    group_full_reward_post: list[float] = []
    profile_delta_logps: dict[str, list[float]] = {p: [] for p in EXPECTED_CANDIDATE_PROFILES}
    profile_delta_probs: dict[str, list[float]] = {p: [] for p in EXPECTED_CANDIDATE_PROFILES}
    flat_delta_logp: list[float] = []

    num_groups = len(scored_groups)
    for group, base_l, post_l in zip(scored_groups, baseline_logps, post_logps):
        cands = group["candidates"]
        if len(cands) != len(base_l) or len(cands) != len(post_l):
            raise ValueError(f"logprob candidate count mismatch in group {group['group_id']}")
        base_probs = softmax(base_l)
        post_probs = softmax(post_l)
        q_base = sum(p * c["full_composite_quality"] for p, c in zip(base_probs, cands))
        q_post = sum(p * c["full_composite_quality"] for p, c in zip(post_probs, cands))
        r_post = sum(
            p * c["full_composite_reward_with_latency"] for p, c in zip(post_probs, cands)
        )
        group_qualities_base.append(q_base)
        group_qualities_post.append(q_post)
        group_full_reward_post.append(r_post)

        cand_reports = []
        for cand, bl, pl, bp, pp in zip(cands, base_l, post_l, base_probs, post_probs):
            dl = float(pl) - float(bl)
            dp = float(pp) - float(bp)
            flat_delta_logp.append(dl)
            profile_delta_logps[cand["candidate_profile"]].append(dl)
            profile_delta_probs[cand["candidate_profile"]].append(dp)
            pooled_latencies.append(float(cand["terminal_ms"]))
            pooled_weights_base.append(bp / num_groups)
            pooled_weights_post.append(pp / num_groups)
            cand_reports.append(
                {
                    "candidate_id": cand["candidate_id"],
                    "candidate_profile": cand["candidate_profile"],
                    "baseline_mean_logp": float(bl),
                    "post_mean_logp": float(pl),
                    "delta_logp": dl,
                    "baseline_p_policy": float(bp),
                    "post_p_policy": float(pp),
                    "delta_p_policy": dp,
                    "advantage": float(cand["advantage"]),
                    "arm_reward_score": cand["arm_reward_score"],
                    "full_composite_quality": float(cand["full_composite_quality"]),
                    "full_composite_reward_with_latency": float(
                        cand["full_composite_reward_with_latency"]
                    ),
                    "terminal_ms": float(cand["terminal_ms"]),
                    "components": cand["components"],
                }
            )
        group_reports.append(
            {
                "group_id": group["group_id"],
                "group_reward_mean": group["group_reward_mean"],
                "group_reward_std": group["group_reward_std"],
                "baseline_expected_quality": q_base,
                "post_expected_quality": q_post,
                "delta_expected_quality": q_post - q_base,
                "candidates": cand_reports,
            }
        )

    base_quality = sum(group_qualities_base) / num_groups
    post_quality = sum(group_qualities_post) / num_groups
    post_full_reward = sum(group_full_reward_post) / num_groups
    base_expected_ms = sum(
        w * lat for w, lat in zip(pooled_weights_base, pooled_latencies)
    )
    post_expected_ms = sum(
        w * lat for w, lat in zip(pooled_weights_post, pooled_latencies)
    )
    base_p95_ms = weighted_quantile_linear(pooled_latencies, pooled_weights_base, 0.95)
    post_p95_ms = weighted_quantile_linear(pooled_latencies, pooled_weights_post, 0.95)
    post_p50_ms = weighted_quantile_linear(pooled_latencies, pooled_weights_post, 0.50)

    profile_summary = {}
    for profile in EXPECTED_CANDIDATE_PROFILES:
        dls = profile_delta_logps[profile]
        dps = profile_delta_probs[profile]
        profile_summary[profile] = {
            "mean_delta_logp": sum(dls) / len(dls),
            "mean_delta_p_policy": sum(dps) / len(dps),
        }

    return {
        "baseline_composite_quality_score": base_quality,
        "composite_quality_score": post_quality,
        "delta_composite_quality_score": post_quality - base_quality,
        "full_composite_reward_with_latency": post_full_reward,
        "baseline_expected_terminal_ms": base_expected_ms,
        "expected_terminal_ms": post_expected_ms,
        "delta_expected_terminal_ms": post_expected_ms - base_expected_ms,
        "baseline_expected_p95_terminal_ms": base_p95_ms,
        "expected_p95_terminal_ms": post_p95_ms,
        "delta_expected_p95_terminal_ms": post_p95_ms - base_p95_ms,
        "expected_p50_terminal_ms": post_p50_ms,
        "profile_preference_shifts": profile_summary,
        "flat_delta_logp": flat_delta_logp,
        "groups": group_reports,
    }


def compute_pareto_frontier(points: list[dict]) -> dict:
    """Compute exact 2D non-dominated Pareto frontier.

    Objectives:
      - maximize ``composite_quality_score`` (higher is better)
      - minimize ``expected_p95_terminal_ms`` (lower is better)

    Point B dominates Point A iff:
      B.quality >= A.quality and B.latency <= A.latency
      and (B.quality > A.quality or B.latency < A.latency).
    """
    if not isinstance(points, list) or not points:
        raise ValueError("compute_pareto_frontier requires a non-empty list of arms")
    validated: list[dict] = []
    seen_ids: set[str] = set()
    for item in points:
        if not isinstance(item, dict):
            raise ValueError("each Pareto point must be a dict")
        arm_id = item.get("arm_id")
        if not isinstance(arm_id, str) or not arm_id or arm_id in seen_ids:
            raise ValueError(f"invalid or duplicate arm_id in Pareto input: {arm_id!r}")
        seen_ids.add(arm_id)
        quality = item.get("composite_quality_score")
        latency = item.get("expected_p95_terminal_ms")
        if not _is_finite_number(quality):
            raise ValueError(f"arm {arm_id} composite_quality_score must be finite")
        if not _is_finite_number(latency) or float(latency) < 0.0:
            raise ValueError(
                f"arm {arm_id} expected_p95_terminal_ms must be finite and >= 0"
            )
        validated.append(
            {
                "arm_id": arm_id,
                "composite_quality_score": float(quality),
                "expected_p95_terminal_ms": float(latency),
            }
        )

    results: list[dict] = []
    for a in validated:
        dominated_by: list[str] = []
        for b in validated:
            if a["arm_id"] == b["arm_id"]:
                continue
            no_worse = (
                b["composite_quality_score"] >= a["composite_quality_score"]
                and b["expected_p95_terminal_ms"] <= a["expected_p95_terminal_ms"]
            )
            strictly_better = (
                b["composite_quality_score"] > a["composite_quality_score"]
                or b["expected_p95_terminal_ms"] < a["expected_p95_terminal_ms"]
            )
            if no_worse and strictly_better:
                dominated_by.append(b["arm_id"])
        dominated_by.sort()
        results.append(
            {
                "arm_id": a["arm_id"],
                "composite_quality_score": a["composite_quality_score"],
                "expected_p95_terminal_ms": a["expected_p95_terminal_ms"],
                "is_pareto_optimal": len(dominated_by) == 0,
                "dominated_by": dominated_by,
            }
        )

    frontier = sorted(
        [r for r in results if r["is_pareto_optimal"]],
        key=lambda r: (
            r["expected_p95_terminal_ms"],
            -r["composite_quality_score"],
            r["arm_id"],
        ),
    )
    dominated = sorted(
        [r["arm_id"] for r in results if not r["is_pareto_optimal"]]
    )
    return {
        "objectives": {
            "composite_quality_score": "maximize",
            "expected_p95_terminal_ms": "minimize",
        },
        "pareto_optimal_arms": [r["arm_id"] for r in frontier],
        "dominated_arms": dominated,
        "arms": results,
    }


def _compute_pairwise_divergence(arm_reports: dict[str, dict]) -> dict:
    """Compute pairwise L2 divergence across arms' policy delta_logp vectors."""
    arm_ids = list(EXPECTED_ARM_IDS)
    pairs: dict[str, dict] = {}
    min_delta_logp_l2 = float("inf")
    for i, a_id in enumerate(arm_ids):
        vec_a = arm_reports[a_id]["policy_evaluation"]["flat_delta_logp"]
        for b_id in arm_ids[i + 1 :]:
            vec_b = arm_reports[b_id]["policy_evaluation"]["flat_delta_logp"]
            l2 = math.sqrt(sum((x - y) ** 2 for x, y in zip(vec_a, vec_b)))
            min_delta_logp_l2 = min(min_delta_logp_l2, l2)
            entry = {
                "delta_logp_l2": l2,
                "quality_diff": (
                    arm_reports[a_id]["policy_evaluation"]["composite_quality_score"]
                    - arm_reports[b_id]["policy_evaluation"]["composite_quality_score"]
                ),
                "p95_terminal_ms_diff": (
                    arm_reports[a_id]["policy_evaluation"]["expected_p95_terminal_ms"]
                    - arm_reports[b_id]["policy_evaluation"]["expected_p95_terminal_ms"]
                ),
            }
            if (
                "final_parameter_sha256" in arm_reports[a_id]
                and "final_parameter_sha256" in arm_reports[b_id]
            ):
                entry["distinct_final_parameters"] = (
                    arm_reports[a_id]["final_parameter_sha256"]
                    != arm_reports[b_id]["final_parameter_sha256"]
                )
            pairs[f"{a_id}__vs__{b_id}"] = entry
    return {
        "min_pairwise_delta_logp_l2": min_delta_logp_l2,
        "all_pairs_distinct_delta_logp": min_delta_logp_l2 > 1e-9,
        "pairs": pairs,
    }


def _deterministic_offline_baseline_logps(config: dict) -> list[list[float]]:
    """Deterministic CPU-only baseline mean log-probs derived solely from candidate text.

    Identical across all 4 arms before objective-mix projection updates.
    """
    baselines: list[list[float]] = []
    for group in config["rollout_groups"]:
        g_vals: list[float] = []
        for cand in group["candidates"]:
            target = cand["assistant_target"]
            slots = cand["trajectory"]["visual_slots"]
            # Short, deterministic structural negative log-likelihood proxy in [-2.6, -1.6]
            length_term = min(0.6, len(target) / 600.0)
            slot_term = 0.08 * len(slots)
            g_vals.append(round(-(1.65 + length_term + slot_term), 6))
        baselines.append(g_vals)
    return baselines


def _offline_grpo_step_grad(
    policy_mean: float,
    old_mean: float,
    advantage: float,
    *,
    clip_eps: float,
    kl_coef: float,
    clamp: float,
) -> tuple[float, float, float]:
    """Analytical loss and d(loss)/d(policy_mean) for clipped GRPO + Schulman KL."""
    raw_lr = policy_mean - old_mean
    log_ratio = max(-clamp, min(clamp, raw_lr))
    ratio = math.exp(log_ratio)
    clipped_ratio = max(1.0 - clip_eps, min(1.0 + clip_eps, ratio))
    surr1 = ratio * advantage
    surr2 = clipped_ratio * advantage
    surrogate = min(surr1, surr2)
    clip_active = surr1 > surr2

    kl_diff = max(-clamp, min(clamp, old_mean - policy_mean))
    kl = math.exp(kl_diff) - kl_diff - 1.0
    loss = -surrogate + kl_coef * kl

    d_surr = 0.0 if clip_active else ratio * advantage
    d_kl = 1.0 - math.exp(kl_diff)
    d_loss = -d_surr + kl_coef * d_kl
    return loss, d_loss, kl


def run_offline_contract(config_path: Path, out_dir: Path) -> dict:
    """Execute deterministic CPU-only contract replay + Pareto frontier + SHA256SUMS."""
    config, config_bytes, invariance = load_and_validate_config(config_path)
    config_sha256 = sha256_bytes(config_bytes)
    baseline_logps = _deterministic_offline_baseline_logps(config)
    step_scale = float(config["evaluation"]["offline_projection_step_scale"])

    arm_reports: dict[str, dict] = {}
    pareto_inputs: list[dict] = []
    written_rel_paths: list[str] = []

    for arm in config["arms"]:
        arm_id = arm["arm_id"]
        scored_groups = score_rollouts_for_arm(config, arm)
        sft_w = float(arm["objective_mix"]["sft_weight"])
        grpo_w = float(arm["objective_mix"]["grpo_weight"])
        clip_eps = float(arm["grpo"]["clip_eps"])
        kl_coef = float(arm["grpo"]["kl_coef"])
        clamp = float(arm["grpo"]["log_ratio_clamp"])
        steps = int(arm["optimization"]["steps"])

        current_logps = [list(g) for g in baseline_logps]
        step_history: list[dict] = []
        for step_idx in range(1, steps + 1):
            sft_losses = []
            grpo_losses = []
            kls = []
            next_logps = []
            for g_idx, group in enumerate(scored_groups):
                g_next = []
                for c_idx, cand in enumerate(group["candidates"]):
                    p_l = current_logps[g_idx][c_idx]
                    o_l = baseline_logps[g_idx][c_idx]
                    sft_loss = -p_l
                    d_sft = -1.0
                    sft_losses.append(sft_loss)
                    if grpo_w > 0.0:
                        g_loss, d_grpo, kl = _offline_grpo_step_grad(
                            p_l,
                            o_l,
                            float(cand["advantage"]),
                            clip_eps=clip_eps,
                            kl_coef=kl_coef,
                            clamp=clamp,
                        )
                    else:
                        g_loss, d_grpo, kl = 0.0, 0.0, 0.0
                    grpo_losses.append(g_loss)
                    kls.append(kl)
                    # Analytical gradient descent step on log-prob coordinate
                    d_total = sft_w * d_sft + grpo_w * d_grpo
                    # Keep projected log-prob strictly negative (valid log-probability)
                    updated = min(-0.05, p_l - step_scale * d_total)
                    g_next.append(round(updated, 8))
                next_logps.append(g_next)
            current_logps = next_logps
            mean_sft = sum(sft_losses) / len(sft_losses)
            mean_grpo = sum(grpo_losses) / len(grpo_losses)
            step_history.append(
                {
                    "step": step_idx,
                    "total_loss": sft_w * mean_sft + grpo_w * mean_grpo,
                    "sft_loss": mean_sft,
                    "grpo_loss": mean_grpo,
                    "mean_kl": sum(kls) / len(kls),
                    "execution_mode": "analytical_score_projection",
                }
            )

        policy_eval = evaluate_policy_metrics(scored_groups, baseline_logps, current_logps)
        arm_report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "evidence_class": "CONTRACT_REPLAY",
            "real_training_evidence": "NOT_RUN",
            "arm_id": arm_id,
            "description": arm.get("description", ""),
            "objective_mix": arm["objective_mix"],
            "fixed_fields_sha256": invariance["per_arm_fixed_fields_sha256"][arm_id],
            "objective_mix_sha256": invariance["per_arm_objective_mix_sha256"][arm_id],
            "steps": step_history,
            "policy_evaluation": policy_eval,
        }
        arm_reports[arm_id] = arm_report
        pareto_inputs.append(
            {
                "arm_id": arm_id,
                "composite_quality_score": policy_eval["composite_quality_score"],
                "expected_p95_terminal_ms": policy_eval["expected_p95_terminal_ms"],
            }
        )
        rel_arm = f"arms/{arm_id}/report.json"
        _atomic_write_bytes(out_dir / rel_arm, pretty_json(arm_report))
        written_rel_paths.append(rel_arm)

    pareto = compute_pareto_frontier(pareto_inputs)
    divergence = _compute_pairwise_divergence(arm_reports)

    summary_arms = {}
    for arm_id in EXPECTED_ARM_IDS:
        ev = arm_reports[arm_id]["policy_evaluation"]
        p_entry = next(r for r in pareto["arms"] if r["arm_id"] == arm_id)
        summary_arms[arm_id] = {
            "objective_mix": arm_reports[arm_id]["objective_mix"],
            "composite_quality_score": ev["composite_quality_score"],
            "delta_composite_quality_score": ev["delta_composite_quality_score"],
            "expected_terminal_ms": ev["expected_terminal_ms"],
            "expected_p95_terminal_ms": ev["expected_p95_terminal_ms"],
            "delta_expected_p95_terminal_ms": ev["delta_expected_p95_terminal_ms"],
            "is_pareto_optimal": p_entry["is_pareto_optimal"],
            "dominated_by": p_entry["dominated_by"],
            "profile_preference_shifts": ev["profile_preference_shifts"],
        }

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "ablation_row": config.get("ablation_row"),
        "evidence_class": "CONTRACT_REPLAY",
        "real_training_evidence": "NOT_RUN",
        "claims": {
            "contract_and_pareto_math_verified": "EXECUTED",
            "real_gpu_optimizer_step": "NOT_RUN",
            "unseen_human_rated_generalization": "NOT_COVERED",
            "online_rollout_policy_quality": "NOT_COVERED",
            "production_model_training": "NOT_COVERED",
        },
        "config_sha256": config_sha256,
        "single_variable_verification": invariance,
        "arm_summaries": summary_arms,
        "pareto_frontier": pareto,
        "pairwise_divergence": divergence,
    }
    _atomic_write_bytes(out_dir / "report.json", pretty_json(report))
    written_rel_paths.append("report.json")
    write_sha256sums(out_dir, written_rel_paths)
    return report


# ---------------------------------------------------------------------------
# Live SM90 GPU Multi-Arm Execution (torch / transformers imported lazily)
# ---------------------------------------------------------------------------


def fingerprint_model_safetensors(model_dir: Path) -> dict:
    """Compute SHA-256 manifest for local model config and all .safetensors files."""
    if not model_dir.is_dir():
        raise ValueError(f"model directory does not exist: {model_dir}")
    config_file = model_dir / "config.json"
    if not config_file.is_file():
        raise ValueError("model directory is missing config.json")
    model_cfg = json.loads(config_file.read_text(encoding="utf-8"))
    weight_files: dict[str, dict] = {}
    for path in sorted(model_dir.rglob("*.safetensors")):
        if not path.is_file():
            continue
        rel = path.relative_to(model_dir).as_posix()
        weight_files[rel] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    if not weight_files:
        raise ValueError("model directory contains no .safetensors weight files")
    return {
        "model_type": model_cfg.get("model_type"),
        "architectures": model_cfg.get("architectures"),
        "config_sha256": sha256_file(config_file),
        "file_count": len(weight_files),
        "weight_files": weight_files,
        "weights_sha256": sha256_bytes(canonical_json(weight_files)),
    }


def query_nvidia_smi_gpu(gpu_index: int) -> dict:
    """Query nvidia-smi for one physical GPU's memory and active compute processes."""
    if not _is_int(gpu_index) or gpu_index < 0:
        raise ValueError(f"invalid --gpu index: {gpu_index!r}")
    try:
        gpu_proc = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        apps_proc = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"nvidia-smi query failed for GPU {gpu_index}: {exc}") from exc

    gpu_lines = [line.strip() for line in gpu_proc.stdout.splitlines() if line.strip()]
    if len(gpu_lines) != 1:
        raise RuntimeError(f"unexpected nvidia-smi output for GPU {gpu_index}: {gpu_lines}")
    parts = [p.strip() for p in gpu_lines[0].split(",")]
    if len(parts) < 6:
        raise RuntimeError(f"malformed nvidia-smi GPU line: {gpu_lines[0]}")

    processes = []
    for line in apps_proc.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        p_parts = [p.strip() for p in text.split(",")]
        pid = int(p_parts[0])
        proc_name = p_parts[1] if len(p_parts) > 1 else ""
        used_mib = float(p_parts[2]) if len(p_parts) > 2 else 0.0
        processes.append({"pid": pid, "process_name": proc_name, "used_memory_mib": used_mib})

    return {
        "index": int(parts[0]),
        "name": parts[1],
        "memory_used_mib": float(parts[2]),
        "memory_total_mib": float(parts[3]),
        "utilization_gpu_pct": float(parts[4]),
        "driver_version": parts[5],
        "compute_processes": processes,
        "compute_process_count": len(processes),
    }


def _verify_idle_gpu_precheck(gpu_index: int) -> dict:
    info = query_nvidia_smi_gpu(gpu_index)
    if info["memory_used_mib"] > IDLE_GPU_MAX_USED_MIB:
        raise RuntimeError(
            f"GPU {gpu_index} is not idle: memory_used_mib={info['memory_used_mib']} "
            f"> {IDLE_GPU_MAX_USED_MIB}"
        )
    if info["compute_process_count"] != 0:
        raise RuntimeError(
            f"GPU {gpu_index} has {info['compute_process_count']} active compute processes"
        )
    info["exclusive_idle_verified"] = True
    return info


def _model_parameter_sha256(torch, model) -> str:
    """Hash every parameter's exact raw bytes in sorted name order."""
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        cpu_bytes = param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        hasher.update(cpu_bytes)
    return hasher.hexdigest()


def _tokenize_groups(tokenizer, config: dict) -> list[list[dict]]:
    max_seq_len = int(config["arms"][0]["optimization"]["max_seq_len"])
    tokenized: list[list[dict]] = []
    for group in config["rollout_groups"]:
        prompt_prefix = f"User: {group['prompt_text']}\nAssistant:\n"
        prompt_ids = tokenizer(prompt_prefix, add_special_tokens=False)["input_ids"]
        g_items: list[dict] = []
        for cand in group["candidates"]:
            target_ids = tokenizer(cand["assistant_target"], add_special_tokens=False)[
                "input_ids"
            ]
            if not target_ids:
                raise ValueError(f"empty tokenized target for {cand['candidate_id']}")
            input_ids = prompt_ids + target_ids
            labels = ([-100] * len(prompt_ids)) + target_ids
            if len(input_ids) > max_seq_len:
                input_ids = input_ids[:max_seq_len]
                labels = labels[:max_seq_len]
            supervised = sum(1 for tok in labels[1:] if tok != -100)
            if supervised < 4:
                raise ValueError(
                    f"candidate {cand['candidate_id']} has too few supervised target tokens "
                    f"({supervised}) within max_seq_len={max_seq_len}"
                )
            g_items.append(
                {
                    "candidate_id": cand["candidate_id"],
                    "input_ids": input_ids,
                    "labels": labels,
                    "prompt_tokens": len(prompt_ids),
                    "target_tokens": supervised,
                    "total_tokens": len(input_ids),
                }
            )
        tokenized.append(g_items)
    return tokenized


def _sequence_mean_logprob(torch, model, item: dict, device):
    input_ids = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
    labels = torch.tensor([item["labels"]], dtype=torch.long, device=device)
    logits = model(input_ids=input_ids).logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    active = shift_labels != -100
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    safe_labels = shift_labels.clamp(min=0)
    token_logp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    mean_logp = token_logp[active].mean()
    return mean_logp


def _measure_all_logps(torch, model, tokenized_groups: list[list[dict]], device) -> list[list[float]]:
    model.eval()
    measured: list[list[float]] = []
    with torch.no_grad():
        for group in tokenized_groups:
            g_vals: list[float] = []
            for item in group:
                val = float(_sequence_mean_logprob(torch, model, item, device).item())
                if not math.isfinite(val):
                    raise RuntimeError(
                        f"non-finite sequence logp for {item['candidate_id']}"
                    )
                g_vals.append(val)
            measured.append(g_vals)
    return measured


def _load_causal_lm_bf16(torch, AutoModelForCausalLM, model_dir: Path):
    kwargs = {
        "local_files_only": True,
        "trust_remote_code": False,
        "attn_implementation": "eager",
    }
    sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    if "dtype" in sig.parameters:
        kwargs["dtype"] = torch.bfloat16
    else:
        kwargs["torch_dtype"] = torch.bfloat16
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    except TypeError:
        kwargs.pop("dtype", None)
        kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    model.config.use_cache = False
    return model


def execute_live_gpu_ablation(
    config_path: Path,
    model_dir: Path,
    gpu_index: int,
    out_dir: Path,
    *,
    allow_non_sm90: bool = False,
) -> dict:
    """Run all 4 objective-mix arms from bit-identical initial BF16 weights on one SM90 GPU."""
    config, config_bytes, invariance = load_and_validate_config(config_path)
    config_sha256 = sha256_bytes(config_bytes)
    model_manifest = fingerprint_model_safetensors(model_dir)
    gpu_precheck = _verify_idle_gpu_precheck(gpu_index)

    # Pin visibility to the verified physical GPU before importing torch.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("live GPU ablation requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    capability = tuple(torch.cuda.get_device_capability(device))
    if not allow_non_sm90 and capability != (9, 0):
        raise RuntimeError(
            f"live GPU ablation requires an SM90 (9, 0) GPU, got {capability}"
        )
    torch.use_deterministic_algorithms(True)

    seed_text = str(config["arms"][0]["optimization"]["seed"])
    seed_int = int(sha256_text(seed_text)[:16], 16) % (2**31)

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, use_fast=True, trust_remote_code=False
    )
    tokenized_groups = _tokenize_groups(tokenizer, config)
    total_candidates = sum(len(g) for g in tokenized_groups)

    started_wall = time.perf_counter()
    model = _load_causal_lm_bf16(torch, AutoModelForCausalLM, model_dir)
    initial_cpu_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    initial_param_sha = _model_parameter_sha256(torch, model)
    model.to(device)
    baseline_logps = _measure_all_logps(torch, model, tokenized_groups, device)

    arm_reports: dict[str, dict] = {}
    pareto_inputs: list[dict] = []
    written_rel_paths: list[str] = []

    for arm in config["arms"]:
        arm_id = arm["arm_id"]
        arm_started = time.perf_counter()
        scored_groups = score_rollouts_for_arm(config, arm)

        # Reset model weights to exact initial CPU snapshot & identical RNG seed.
        model.load_state_dict(initial_cpu_state, strict=True)
        model.to(device)
        arm_init_sha = _model_parameter_sha256(torch, model)
        if arm_init_sha != initial_param_sha:
            raise RuntimeError(
                f"arm {arm_id} initial parameter SHA mismatch vs shared initial base"
            )
        torch.manual_seed(seed_int)
        torch.cuda.manual_seed_all(seed_int)

        opt_cfg = arm["optimization"]
        grpo_cfg = arm["grpo"]
        sft_w = float(arm["objective_mix"]["sft_weight"])
        grpo_w = float(arm["objective_mix"]["grpo_weight"])
        clip_eps = float(grpo_cfg["clip_eps"])
        kl_coef = float(grpo_cfg["kl_coef"])
        clamp = float(grpo_cfg["log_ratio_clamp"])
        steps = int(opt_cfg["steps"])

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(opt_cfg["learning_rate"]),
            weight_decay=float(opt_cfg["weight_decay"]),
            foreach=False,
        )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        step_history: list[dict] = []

        for step_idx in range(1, steps + 1):
            step_started = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)

            sft_loss_sum = 0.0
            grpo_loss_sum = 0.0
            kl_sum = 0.0
            total_loss_sum = 0.0

            for g_idx, (t_group, s_group) in enumerate(
                zip(tokenized_groups, scored_groups)
            ):
                for c_idx, (item, cand) in enumerate(zip(t_group, s_group["candidates"])):
                    policy_mean = _sequence_mean_logprob(torch, model, item, device)
                    sft_loss = -policy_mean
                    if grpo_w > 0.0:
                        old_mean = baseline_logps[g_idx][c_idx]
                        adv = float(cand["advantage"])
                        log_ratio = (policy_mean - old_mean).clamp(min=-clamp, max=clamp)
                        ratio = torch.exp(log_ratio)
                        clipped_ratio = ratio.clamp(
                            min=1.0 - clip_eps, max=1.0 + clip_eps
                        )
                        surrogate = torch.minimum(ratio * adv, clipped_ratio * adv)
                        kl_diff = (old_mean - policy_mean).clamp(min=-clamp, max=clamp)
                        kl_term = torch.exp(kl_diff) - kl_diff - 1.0
                        grpo_loss = -surrogate + kl_coef * kl_term
                        kl_val = float(kl_term.item())
                        grpo_val = float(grpo_loss.item())
                    else:
                        grpo_loss = torch.zeros((), dtype=torch.float32, device=device)
                        kl_val = 0.0
                        grpo_val = 0.0

                    cand_loss = sft_w * sft_loss + grpo_w * grpo_loss
                    loss_val = float(cand_loss.item())
                    if not math.isfinite(loss_val):
                        raise RuntimeError(
                            f"non-finite loss in arm {arm_id} step {step_idx}"
                        )
                    (cand_loss / float(total_candidates)).backward()
                    sft_loss_sum += float(sft_loss.item())
                    grpo_loss_sum += grpo_val
                    kl_sum += kl_val
                    total_loss_sum += loss_val

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(opt_cfg["max_grad_norm"]),
                error_if_nonfinite=True,
            )
            grad_norm_val = float(grad_norm)
            if not math.isfinite(grad_norm_val):
                raise RuntimeError(f"non-finite grad norm in arm {arm_id} step {step_idx}")
            grad_tensors = [
                p.grad for p in model.parameters() if p.grad is not None
            ]
            nonzero_grads = sum(
                1 for g in grad_tensors if torch.count_nonzero(g).item() > 0
            )
            if nonzero_grads <= 0:
                raise RuntimeError(
                    f"zero nonzero gradient tensors in arm {arm_id} step {step_idx}"
                )

            optimizer.step()
            torch.cuda.synchronize(device)
            step_history.append(
                {
                    "step": step_idx,
                    "total_loss": total_loss_sum / total_candidates,
                    "sft_loss": sft_loss_sum / total_candidates,
                    "grpo_loss": grpo_loss_sum / total_candidates,
                    "mean_kl": kl_sum / total_candidates,
                    "grad_norm": grad_norm_val,
                    "nonzero_gradient_tensors": nonzero_grads,
                    "gradient_tensors": len(grad_tensors),
                    "step_wall_ms": round((time.perf_counter() - step_started) * 1000.0, 3),
                    "peak_memory_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(device)
                    ),
                }
            )

        # Verify finite parameters and positive L2 delta vs identical initial weights.
        changed_tensors = 0
        changed_params = 0
        max_abs_delta = 0.0
        delta_l2_sq = 0.0
        for name, param in model.named_parameters():
            after = param.detach().cpu()
            if not bool(torch.isfinite(after).all().item()):
                raise RuntimeError(f"arm {arm_id} produced non-finite parameter {name}")
            diff = after.float() - initial_cpu_state[name].float()
            nz = int(torch.count_nonzero(diff).item())
            if nz > 0:
                changed_tensors += 1
                changed_params += nz
                max_abs_delta = max(max_abs_delta, float(diff.abs().max().item()))
                delta_l2_sq += float(diff.square().sum().item())
        param_delta_l2 = math.sqrt(delta_l2_sq)
        if changed_params <= 0 or not math.isfinite(param_delta_l2) or param_delta_l2 <= 0.0:
            raise RuntimeError(f"arm {arm_id} produced zero parameter delta")

        final_param_sha = _model_parameter_sha256(torch, model)
        post_logps = _measure_all_logps(torch, model, tokenized_groups, device)
        policy_eval = evaluate_policy_metrics(scored_groups, baseline_logps, post_logps)

        arm_report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "evidence_class": "LIVE_GPU_OBJECTIVE_MIX",
            "real_training_evidence": "REAL_SM90_OPTIMIZER_GATE",
            "arm_id": arm_id,
            "description": arm.get("description", ""),
            "objective_mix": arm["objective_mix"],
            "fixed_fields_sha256": invariance["per_arm_fixed_fields_sha256"][arm_id],
            "objective_mix_sha256": invariance["per_arm_objective_mix_sha256"][arm_id],
            "initial_parameter_sha256": arm_init_sha,
            "final_parameter_sha256": final_param_sha,
            "changed_parameter_tensors": changed_tensors,
            "changed_parameters": changed_params,
            "parameter_delta_l2": param_delta_l2,
            "max_abs_parameter_delta": max_abs_delta,
            "arm_wall_ms": round((time.perf_counter() - arm_started) * 1000.0, 3),
            "steps": step_history,
            "policy_evaluation": policy_eval,
        }
        arm_reports[arm_id] = arm_report
        pareto_inputs.append(
            {
                "arm_id": arm_id,
                "composite_quality_score": policy_eval["composite_quality_score"],
                "expected_p95_terminal_ms": policy_eval["expected_p95_terminal_ms"],
            }
        )
        rel_arm = f"arms/{arm_id}/report.json"
        _atomic_write_bytes(out_dir / rel_arm, pretty_json(arm_report))
        written_rel_paths.append(rel_arm)

        # Clean optimizer, loop GPU tensor refs, and GPU cache between arms.
        del optimizer, grad_tensors, grad_norm, param, cand_loss, sft_loss, grpo_loss, policy_mean
        log_ratio = ratio = clipped_ratio = surrogate = kl_diff = kl_term = None
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

    # Release model, cuBLAS deterministic workspace, and verify GPU cleanup.
    del model
    del initial_cpu_state
    gc.collect()
    if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
        torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    post_alloc_bytes = int(torch.cuda.memory_allocated(device))
    post_reserved_bytes = int(torch.cuda.memory_reserved(device))
    if post_alloc_bytes != 0:
        raise RuntimeError(
            f"GPU memory leak after cleanup: memory_allocated={post_alloc_bytes} bytes"
        )
    gpu_postcheck = query_nvidia_smi_gpu(gpu_index)
    # Pre-check proved 0 processes before launch; inside a container PID namespace
    # nvidia-smi may report the host PID, so at most 1 process (this CUDA context) is permitted.
    other_proc_count = max(0, len(gpu_postcheck["compute_processes"]) - 1)
    if other_proc_count > 0:
        raise RuntimeError(
            f"foreign compute processes detected on GPU {gpu_index} after run: "
            f"{gpu_postcheck['compute_processes']}"
        )
    gpu_postcheck["torch_memory_allocated_bytes"] = post_alloc_bytes
    gpu_postcheck["torch_memory_reserved_bytes"] = post_reserved_bytes
    gpu_postcheck["other_compute_process_count"] = other_proc_count
    gpu_postcheck["cleanup_verified"] = True

    pareto = compute_pareto_frontier(pareto_inputs)
    divergence = _compute_pairwise_divergence(arm_reports)
    unique_final_shas = {
        arm_reports[a]["final_parameter_sha256"] for a in EXPECTED_ARM_IDS
    }
    divergence["all_arms_distinct_final_parameters"] = (
        len(unique_final_shas) == len(EXPECTED_ARM_IDS)
    )

    summary_arms = {}
    for arm_id in EXPECTED_ARM_IDS:
        ar = arm_reports[arm_id]
        ev = ar["policy_evaluation"]
        p_entry = next(r for r in pareto["arms"] if r["arm_id"] == arm_id)
        summary_arms[arm_id] = {
            "objective_mix": ar["objective_mix"],
            "initial_parameter_sha256": ar["initial_parameter_sha256"],
            "final_parameter_sha256": ar["final_parameter_sha256"],
            "parameter_delta_l2": ar["parameter_delta_l2"],
            "composite_quality_score": ev["composite_quality_score"],
            "delta_composite_quality_score": ev["delta_composite_quality_score"],
            "expected_terminal_ms": ev["expected_terminal_ms"],
            "expected_p95_terminal_ms": ev["expected_p95_terminal_ms"],
            "delta_expected_p95_terminal_ms": ev["delta_expected_p95_terminal_ms"],
            "is_pareto_optimal": p_entry["is_pareto_optimal"],
            "dominated_by": p_entry["dominated_by"],
            "profile_preference_shifts": ev["profile_preference_shifts"],
        }

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "ablation_row": config.get("ablation_row"),
        "evidence_class": "LIVE_GPU_OBJECTIVE_MIX",
        "real_training_evidence": "REAL_SM90_OPTIMIZER_GATE",
        "claims": {
            "real_sm90_optimizer_step": "EXECUTED",
            "multi_arm_objective_mix_divergence": "EXECUTED",
            "offline_rollout_pareto_frontier": "EXECUTED",
            "unseen_human_rated_generalization": "NOT_COVERED",
            "online_rollout_policy_quality": "NOT_COVERED",
            "production_model_training": "NOT_COVERED",
        },
        "config_sha256": config_sha256,
        "single_variable_verification": invariance,
        "model_manifest": model_manifest,
        "shared_initial_parameter_sha256": initial_param_sha,
        "seed": {"text": seed_text, "derived_uint31": seed_int},
        "gpu_precheck": gpu_precheck,
        "gpu_postcheck": gpu_postcheck,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "torch_cuda": torch.version.cuda,
            "requested_gpu_index": gpu_index,
            "device_name": torch.cuda.get_device_name(device),
            "device_capability": list(capability),
            "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        },
        "timing": {
            "total_wall_ms": round((time.perf_counter() - started_wall) * 1000.0, 3)
        },
        "arm_summaries": summary_arms,
        "pareto_frontier": pareto,
        "pairwise_divergence": divergence,
    }
    _atomic_write_bytes(out_dir / "report.json", pretty_json(report))
    written_rel_paths.append("report.json")
    write_sha256sums(out_dir, written_rel_paths)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage C Ablation #5: 4-arm Post-Training Objective-Mix & "
            "Quality-vs-Latency Pareto Frontier"
        )
    )
    parser.add_argument("--config", required=True, help="Path to objective_mix_ablation.json")
    parser.add_argument("--out", required=True, help="Output directory for reports & SHA256SUMS")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--offline-contract",
        "--replay-only",
        dest="offline_contract",
        action="store_true",
        help="Run deterministic CPU-only contract replay without importing torch/transformers",
    )
    mode.add_argument(
        "--live-gpu",
        dest="live_gpu",
        action="store_true",
        help="Run all 4 objective-mix arms on an exclusive idle SM90 GPU",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Local HuggingFace model directory (required for --live-gpu)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Physical GPU index for --live-gpu (default: 0)",
    )
    parser.add_argument(
        "--allow-non-sm90",
        action="store_true",
        help="Allow non-SM90 CUDA capability in --live-gpu mode",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config_path = Path(args.config)
        out_dir = Path(args.out)
        if args.offline_contract:
            if args.model_dir is not None:
                raise ValueError("--model-dir cannot be combined with --offline-contract")
            report = run_offline_contract(config_path, out_dir)
        else:
            if not args.model_dir:
                raise ValueError("--model-dir is required when --live-gpu is set")
            report = execute_live_gpu_ablation(
                config_path=config_path,
                model_dir=Path(args.model_dir),
                gpu_index=args.gpu,
                out_dir=out_dir,
                allow_non_sm90=args.allow_non_sm90,
            )
    except (OSError, ValueError, RuntimeError, ImportError, json.JSONDecodeError) as exc:
        parser.exit(2, f"objective-mix ablation error: {exc}\n")

    summary = {
        "evidence_class": report["evidence_class"],
        "real_training_evidence": report["real_training_evidence"],
        "single_variable_verified": report["single_variable_verification"][
            "single_variable_verified"
        ],
        "pareto_optimal_arms": report["pareto_frontier"]["pareto_optimal_arms"],
        "dominated_arms": report["pareto_frontier"]["dominated_arms"],
        "out_dir": str(Path(args.out)),
    }
    sys.stdout.write(json.dumps(summary, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
