#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stdlib-only independent numerical verifier for real online-GRPO evidence.

No rollout producer, token-alignment producer, preflight, objective producer, or
ML-framework module is imported. Raw SSE, packing, model files, cold-process
identity, raw-logit rows, objectives, and arithmetic controls are reconstructed
from evidence. A failing verification writes no report.
"""

import argparse
from decimal import Decimal, getcontext
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

getcontext().prec = 50
D0 = Decimal(0)
D1 = Decimal(1)
TOKEN_TOLERANCE = Decimal("1e-9")
OBJECTIVE_TOLERANCE = Decimal("1e-12")


class VerificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"duplicate JSON member: {key}")
        result[key] = value
    return result


def load_json(path):
    def constant(value):
        raise VerificationError(f"non-finite JSON number: {value}")
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       object_pairs_hook=no_duplicates, parse_constant=constant)
    return value


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    raw = Path(path).read_bytes()
    return [json.loads(line, object_pairs_hook=no_duplicates,
                       parse_constant=lambda value: (_ for _ in ()).throw(
                           VerificationError(f"non-finite JSON number: {value}")))
            for line in raw.decode("utf-8").splitlines() if line]


def dec(value):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(float(value)), f"invalid finite numeric value: {value!r}")
    return Decimal(repr(float(value)))


def decimal_mean(values):
    require(values, "empty masked mean")
    return sum((dec(value) for value in values), D0) / Decimal(len(values))


def clamp(value, bound):
    return max(-bound, min(bound, value))


def decimal_metrics(behavior, current, reference, mask, advantage, objective):
    indices = [index for index, bit in enumerate(mask) if bit == 1]
    require(indices and len(behavior) == len(current) == len(reference) == len(mask),
            "objective arrays/mask are empty or misaligned")
    behavior_mean = decimal_mean([behavior[index] for index in indices])
    current_mean = decimal_mean([current[index] for index in indices])
    reference_mean = decimal_mean([reference[index] for index in indices])
    bound = dec(objective["log_ratio_clamp"])
    epsilon = dec(objective["clip_eps"])
    raw_ratio = current_mean - behavior_mean
    difference = clamp(raw_ratio, bound)
    ratio = difference.exp()
    clipped = max(D1 - epsilon, min(D1 + epsilon, ratio))
    adv = dec(advantage)
    first = ratio * adv
    second = clipped * adv
    selected = min(first, second)
    raw_kl = reference_mean - current_mean
    kl_difference = clamp(raw_kl, bound)
    kl = kl_difference.exp() - kl_difference - D1
    loss = -selected + dec(objective["kl_coef"]) * kl
    return {
        "supervised_response_indices": indices,
        "behavior_mean_logp": behavior_mean, "policy_mean_logp": current_mean,
        "reference_mean_logp": reference_mean,
        "raw_current_minus_behavior_logp": raw_ratio,
        "current_minus_behavior_logp": difference, "ratio": ratio,
        "clipped_ratio": clipped, "advantage": adv,
        "unclipped_surrogate": first, "clipped_surrogate": second,
        "selected_surrogate": selected, "clip_active": first > second,
        "raw_reference_minus_current_logp": raw_kl,
        "reference_minus_current_logp": kl_difference, "schulman_kl": kl,
        "objective_loss": loss,
    }


def arithmetic_controls(objective):
    cases = {
        "positive_advantage_clipping": ([0.0], [math.log(2.0)], [math.log(2.0)], [1], 0.5),
        "negative_advantage_min_selection": ([0.0], [math.log(2.0)], [0.0], [1], -0.5),
        "schulman_exp_minus_one": ([0.0], [0.0], [-1.0], [1], 0.0),
        "log_ratio_clamp": ([0.0], [20.0], [0.0], [1], 0.5),
    }
    results = {}
    for name, values in cases.items():
        metrics = decimal_metrics(*values, objective)
        if name == "positive_advantage_clipping":
            require(abs(metrics["ratio"] - Decimal(2)) < Decimal("1e-12") and
                    metrics["clipped_ratio"] == Decimal("1.2") and
                    abs(metrics["selected_surrogate"] - Decimal("0.6")) <
                    Decimal("1e-12") and
                    metrics["schulman_kl"] == 0 and metrics["clip_active"],
                    "positive-advantage clipping oracle failed")
        elif name == "negative_advantage_min_selection":
            require(abs(metrics["selected_surrogate"] - Decimal(-1)) <
                    Decimal("1e-12") and not metrics["clip_active"],
                    "negative-advantage min oracle failed")
        elif name == "schulman_exp_minus_one":
            require(abs(metrics["schulman_kl"] - Decimal(-1).exp()) < Decimal("1e-45"),
                    "Schulman exp(-1) oracle failed")
        else:
            require(metrics["current_minus_behavior_logp"] == Decimal(10) and
                    metrics["clipped_ratio"] == Decimal("1.2"),
                    "log-ratio clamp oracle failed")
        results[name] = {key: str(value) if isinstance(value, Decimal) else value
                         for key, value in metrics.items()}
    return results


def raw_behavior(record, accepted):
    metadata = record["sse_replay"]
    require(metadata["raw_ref"] == metadata["raw_sha256"] + ".sse",
            "raw SSE reference is not content-addressed")
    raw = (accepted / "artifacts" / metadata["raw_ref"]).read_bytes()
    require(len(raw) == metadata["raw_bytes"] and sha256_bytes(raw) == metadata["raw_sha256"],
            "raw SSE bytes/hash differ")
    frames = re.split(br"\r?\n\r?\n", raw)
    done = 0
    behavior_blocks = []
    final_chunks = []
    for frame in frames:
        if not frame:
            continue
        data = b"\n".join(line[5:].lstrip(b" ") for line in frame.splitlines()
                          if line.startswith(b"data:"))
        if not data:
            continue
        if data == b"[DONE]":
            done += 1
            continue
        chunk = json.loads(data.decode("utf-8"), object_pairs_hook=no_duplicates)
        final_chunks.append(chunk)
        if "behavior" in chunk:
            behavior_blocks.append(chunk["behavior"])
    require(done == 1 and final_chunks and len(behavior_blocks) == 1,
            "raw SSE does not have one terminal chunk, one behavior block, and one DONE")
    require(final_chunks[-1]["choices"][0]["finish_reason"] == "stop" and
            record["terminal_state"] == "completed",
            "accepted raw behavior is cancelled, truncated, or incomplete")
    evidence = behavior_blocks[0]
    summary = record["rollout_provenance"]["behavior"]
    for key in ("format", "tokenizer_sha256", "token_ids", "logprobs", "mask"):
        require(evidence[key] == summary[key], f"raw/provenance behavior differs: {key}")
    require(summary["evidence_sha256"] == sha256_bytes(canonical(evidence)) and
            summary["status"] == "COVERED" and
            record["rollout_provenance"]["response"]["behavior_evidence"] == "exposed",
            "raw behavior evidence hash/coverage differs")
    return evidence, metadata["raw_sha256"]


def compare_metrics(actual, expected, label):
    maximum = D0
    for key, value in expected.items():
        require(key in actual, f"{label} missing metric {key}")
        if isinstance(value, Decimal):
            difference = abs(dec(actual[key]) - value)
            maximum = max(maximum, difference)
            require(difference <= OBJECTIVE_TOLERANCE,
                    f"{label}.{key} objective delta {difference} exceeds tolerance")
        else:
            require(actual[key] == value, f"{label}.{key} differs")
    return maximum


def verify(root, model_dir, replay_binary):
    source_identity = load_json(root / "source_identity.json")
    source_summary_native = source_identity["source_native_process"]
    require(source_summary_native["pid"] > 0 and
            source_summary_native["start_time_ticks"] > 0,
            "source cold-process identity is incomplete")
    real = load_json(root / "real_objectives.json")
    require(real["schema"] == "umminfer-online-grpo-real-model-objectives-v1" and
            real["status"] == "PASS", "real objective evidence is absent or failed")
    require(real["claims"]["optimizer"] == "NOT_RUN" and
            real["claims"]["optimizer_resume_update"] == "NOT_RUN" and
            real["claims"]["model_weights_updated"] is False and
            real["claims"]["model_quality"] == "NOT_RUN",
            "real evidence contains an optimizer or quality claim")
    config = load_json(root / "preflight/config.json")
    for policy_name in ("current", "reference"):
        policy = config[policy_name + "_policy"]
        require(policy == real["policies"][policy_name] and
                policy["model_id"] == real["model"]["model_id"] and
                policy["model_revision"] == real["model"]["model_revision"] and
                policy["model_sha256"] == real["model"]["model_sha256"] and
                policy["model_config_sha256"] == real["model"]["model_config_sha256"] and
                policy["tokenizer_sha256"] == real["model"]["tokenizer_sha256"],
                f"{policy_name} policy would compare different model weights/config/tokenizer")
    run_identity = load_json(root / "preflight/run_identity.json")
    require(run_identity["run_id"] == real["preflight_run_id"] and
            run_identity["identity"]["config_sha256"] == sha256_bytes(canonical(config)),
            "real evidence is not bound to the exact preflight config/run")
    audit = load_json(root / "independent-evidence-audit.json")
    require(audit["status"] == "PASS" and
            audit["source_rollout_reconstruction"] == "PASS_BYTE_IDENTICAL" and
            audit["objective_reconstruction"] == "PASS_BYTE_IDENTICAL" and
            audit["schedule_reconstruction"] == "PASS_BYTE_IDENTICAL" and
            audit["run_identity_reconstruction"] == "PASS_BYTE_IDENTICAL" and
            audit["checkpoint_disagreements"] == [],
            "independent structural/preflight audit is not clean")

    accepted = root / "accepted"
    plan = load_json(accepted / "plan.json")
    records_list = read_jsonl(accepted / "trajectories.jsonl")
    records = {record["rollout_provenance"]["rollout_id"]: record
               for record in records_list}
    require(len(records) == len(records_list) == 6, "accepted rollout identities are incomplete")
    pack_rows = read_jsonl(accepted / "pack/grpo_train.jsonl") + read_jsonl(
        accepted / "pack/grpo_eval.jsonl")
    require(len(pack_rows) == 2, "accepted pack does not contain exactly two groups")
    packed_members = {}
    for row in pack_rows:
        require(len(row["rollouts"]) == 3 and row["group_size"] == 3,
                "dataset group is not complete one-to-one three-member membership")
        for rollout in row["rollouts"]:
            rollout_id = rollout["rollout_provenance"]["rollout_id"]
            require(rollout_id in records and rollout_id not in packed_members,
                    "missing, duplicate, or cross-group dataset member")
            packed_members[rollout_id] = (row, rollout)
    require(set(packed_members) == set(records), "dataset membership is not one-to-one")

    # Independently rehash the complete exact weight set and model config.
    actual_weights = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        require(path.is_file() and not path.is_symlink(), "weight is missing or symbolic")
        actual_weights[path.name] = {"bytes": path.stat().st_size,
                                     "sha256": sha256_file(path)}
    require(actual_weights == source_identity["model"]["available_weights"] and
            sha256_bytes(canonical(actual_weights)) == real["model"]["model_sha256"] ==
            plan["behavior_policy"]["model_sha256"],
            "independent weight-set rehash differs from behavior/current/reference")
    config_raw = (model_dir / "config.json").read_bytes()
    require(sha256_bytes(config_raw) == real["model"]["model_config_sha256"] ==
            plan["behavior_policy"]["model_config_sha256"],
            "independent model-config rehash differs")
    tokenizer_files = records_list[0]["token_export"]["tokenizer"]["available_files"]
    for item in tokenizer_files:
        path = model_dir / item["path"]
        require(path.is_file() and path.stat().st_size == item["bytes"] and
                sha256_file(path) == item["sha256"],
                f"independent tokenizer rehash differs: {item['path']}")
    for record in records_list[1:]:
        require(record["token_export"]["tokenizer"]["available_files"] == tokenizer_files,
                "complete available tokenizer file identity drifted")
    require(sha256_file(replay_binary) == source_identity[
        "independent_replay_binary"]["sha256"], "independent replay binary hash differs")

    # EOS declarations come from exact local config files, never sampled evidence.
    generation = load_json(model_dir / "generation_config.json")
    eos = generation.get("eos_token_id", load_json(model_dir / "config.json").get("eos_token_id"))
    terminal = set(eos if isinstance(eos, list) else [eos])
    terminal.discard(None)
    require(terminal, "model declares no EOS identity")

    process_doc = load_json(root / "forward_processes.json")
    require(process_doc["serial"] is True and len(process_doc["processes"]) == 18,
            "cold forward schedule is not 18 serial processes")
    process_by_key = {}
    for process in process_doc["processes"]:
        key = (process["policy"], process["rollout_id"])
        require(key not in process_by_key and process["returncode"] == 0 and
                process["pid"] > 0 and process["start_time_ticks"] > 0 and
                Path(process["executable"]).resolve() == replay_binary.resolve() and
                process["gpu7_before"]["memory_used_mib"] <= 16 and
                process["gpu7_after"]["memory_used_mib"] <= 16 and
                not process["gpu7_before"]["compute_processes"] and
                not process["gpu7_after"]["compute_processes"],
                f"cold process/GPU identity is invalid for {key}")
        process_by_key[key] = process
    require(len({process["pid"] for process in process_doc["processes"]}) == 18 and
            len({(process["pid"], process["start_time_ticks"])
                 for process in process_doc["processes"]}) == 18,
            "forward processes are not distinct cold loads")

    preflight_fixtures = load_json(root / "preflight/objective_fixtures.json")
    fixture_members = {member["rollout_id"]: member for group in preflight_fixtures["groups"]
                       for member in group["members"]}
    require(set(fixture_members) == set(records), "preflight member set differs from source")
    maximum_token = D0
    maximum_objective = D0
    sampled_count = supervised_count = 0
    seen_real_members = set()
    per_group_losses = []
    for group in real["groups"]:
        row = next((candidate for candidate in pack_rows
                    if candidate["group_id"] == group["dataset_group_id"]), None)
        require(row is not None and
                group["schedule_group_id"] == row["rollout_provenance"]["schedule_group_id"] and
                group["prompt_sha256"] == row["prompt_sha256"] and
                group["group_export_sha256"] == sha256_bytes(canonical(row)),
                "real group identity/prompt/export differs from accepted pack")
        scores = [float(member["score"]) for member in row["rollouts"]]
        mean = math.fsum(scores) / len(scores)
        variance = math.fsum((score - mean) ** 2 for score in scores) / len(scores)
        std = math.sqrt(variance)
        require(abs(group["group_mean"] - mean) <= 1e-12 and
                abs(group["group_std"] - std) <= 1e-12 and
                group["has_reward_variance"] is (std > 0.0),
                "real group reward statistics differ")
        expected_losses = []
        for member in group["members"]:
            rollout_id = member["rollout_id"]
            require(rollout_id in records and rollout_id not in seen_real_members,
                    "real objective has missing/duplicate/cross-group rollout")
            seen_real_members.add(rollout_id)
            pack_row, packed_rollout = packed_members[rollout_id]
            require(pack_row["group_id"] == group["dataset_group_id"] and
                    member["rank"] == packed_rollout["rank"] and
                    member["reward"] == packed_rollout["score"] and
                    member["advantage"] == packed_rollout["advantage"],
                    "real member rank/reward/advantage/group differs")
            record = records[rollout_id]
            behavior, raw_sha = raw_behavior(record, accepted)
            require(member["raw_sse_sha256"] == raw_sha and
                    member["source_record_sha256"] == sha256_bytes(canonical(record)) and
                    member["behavior_evidence_sha256"] == record[
                        "rollout_provenance"]["behavior"]["evidence_sha256"] and
                    member["token_export_sha256"] == record["token_export"]["export_sha256"] and
                    member["token_export_source_sha256"] == record["token_export"][
                        "source_record_sha256"],
                    "real member raw/source/export binding differs")
            require(not terminal.intersection(behavior["token_ids"]),
                    "sampled supervision contains a terminal/EOS token")
            sampled_count += len(behavior["token_ids"])
            supervised_count += sum(behavior["mask"])

            documents = {}
            evidences = {}
            for policy in ("behavior", "current", "reference"):
                path = root / "forwards" / policy / (rollout_id + ".json")
                document = load_json(path)
                documents[policy] = document
                evidence = document["independent_behavior"]
                evidences[policy] = evidence
                require((policy, rollout_id) in process_by_key and
                        process_by_key[(policy, rollout_id)]["document_sha256"] ==
                        sha256_file(path), "forward document/process hash binding differs")
                require(evidence["format"] ==
                        "umminfer-independent-presampling-logsoftmax-v1" and
                        evidence["producer_scoring_used"] is False and
                        evidence["input_precision"] == "FP32_PRE_SAMPLING_LM_HEAD_LOGITS" and
                        evidence["tokenizer_sha256"] == real["model"]["tokenizer_sha256"] and
                        evidence["token_ids"] == behavior["token_ids"] and
                        evidence["mask"] == behavior["mask"] and
                        document["clean_text"] == record["clean_text"],
                        f"independent {policy} ids/masks/tokenizer differ for {rollout_id}")
                rows = evidence["rows"]
                require(len(rows) == len(evidence["logprobs"]) == len(behavior["token_ids"]),
                        "raw per-token evidence length differs")
                for index, (token_id, logprob, raw_row) in enumerate(zip(
                        behavior["token_ids"], evidence["logprobs"], rows)):
                    require(raw_row["token_id"] == token_id and
                            raw_row["vocab_size"] > max(behavior["token_ids"]) and
                            len(raw_row["raw_logits_sha256"]) == 64,
                            "raw row token/vocab/hash binding differs")
                    reconstructed = (dec(raw_row["selected_raw_logit"]) -
                                     dec(raw_row["max_raw_logit"]) -
                                     dec(raw_row["log_normalizer"]))
                    require(abs(reconstructed - dec(raw_row["logprob"])) <=
                            OBJECTIVE_TOLERANCE and
                            abs(dec(logprob) - dec(raw_row["logprob"])) <=
                            OBJECTIVE_TOLERANCE and dec(logprob) <= 0,
                            "raw row components/logprob are inconsistent")
                    require(real["groups"] and member["independent_raw_rows"][policy][index]
                            == raw_row, "real per-token raw row differs from native document")

            prompt_ids = documents["behavior"]["prompt_token_ids"]
            require(all(documents[policy]["prompt_token_ids"] == prompt_ids
                        for policy in ("current", "reference")),
                    "behavior/current/reference exact prompt ids differ")
            input_ids = prompt_ids + behavior["token_ids"]
            labels = [-100] * len(prompt_ids) + [token if bit else -100 for token, bit in
                                                zip(behavior["token_ids"], behavior["mask"])]
            response_mask = [0] * len(prompt_ids) + behavior["mask"]
            packed = fixture_members[rollout_id]["packing"]
            require(packed["prompt_token_ids"] == prompt_ids and
                    packed["input_ids"] == input_ids and packed["labels"] == labels and
                    packed["response_mask"] == response_mask and
                    member["prompt_token_ids"] == prompt_ids and
                    member["input_ids"] == input_ids and member["labels"] == labels and
                    member["response_mask"] == response_mask and
                    member["packed_ids_labels_masks_sha256"] == packed[
                        "packed_ids_labels_masks_sha256"],
                    "independent causal packing differs from real/preflight packing")
            require(member["token_ids"] == behavior["token_ids"] and
                    member["mask"] == behavior["mask"] and
                    member["sampled_behavior_logprobs"] == behavior["logprobs"],
                    "real raw sampled ids/masks/logprobs differ")

            independent = {policy: evidences[policy]["logprobs"]
                           for policy in ("behavior", "current", "reference")}
            for policy in independent:
                require(member["independent_logprobs"][policy] == independent[policy] and
                        member["native_forward_document_sha256"][policy] ==
                        sha256_bytes(canonical(documents[policy])),
                        "real native forward hash/logprob array differs")
            deltas = {
                "behavior_vs_sampled": [abs(dec(a) - dec(b)) for a, b in
                                        zip(independent["behavior"], behavior["logprobs"])],
                "current_vs_behavior": [abs(dec(a) - dec(b)) for a, b in
                                        zip(independent["current"], independent["behavior"])],
                "reference_vs_behavior": [abs(dec(a) - dec(b)) for a, b in
                                          zip(independent["reference"], independent["behavior"])],
            }
            for name, values in deltas.items():
                require(len(values) == len(member["per_token_abs_deltas"][name]),
                        "per-token delta array length differs")
                for actual, expected in zip(member["per_token_abs_deltas"][name], values):
                    require(abs(dec(actual) - expected) <= OBJECTIVE_TOLERANCE,
                            "raw per-token delta evidence differs")
                maximum_token = max(maximum_token, *values)
                require(max(values) <= TOKEN_TOLERANCE,
                        f"{name} exceeds per-token tolerance")
            for policy in ("current", "reference"):
                require(evidences[policy]["rows"] == evidences["behavior"]["rows"],
                        f"{policy}/behavior exact raw FP32 rows or components differ")

            expected_metrics = decimal_metrics(
                behavior["logprobs"], independent["current"], independent["reference"],
                behavior["mask"], member["advantage"], config["objective"])
            maximum_objective = max(maximum_objective, compare_metrics(
                member["metrics"], expected_metrics, rollout_id))
            expected_losses.append(expected_metrics["objective_loss"])
        expected_group_loss = sum(expected_losses, D0) / Decimal(len(expected_losses))
        group_delta = abs(dec(group["objective_loss"]) - expected_group_loss)
        require(group_delta <= OBJECTIVE_TOLERANCE, "real group objective mean differs")
        maximum_objective = max(maximum_objective, group_delta)
        per_group_losses.append({"dataset_group_id": group["dataset_group_id"],
                                 "objective_loss": group["objective_loss"],
                                 "independent_decimal_loss": str(expected_group_loss)})
    require(seen_real_members == set(records) and sampled_count == real[
        "token_counts"]["sampled"] and supervised_count == real[
        "token_counts"]["supervised"], "real per-token/member totals differ")

    controls = arithmetic_controls(config["objective"])
    return {
        "schema": "umminfer-online-grpo-real-model-independent-verification-v1",
        "status": "PASS", "groups": 2, "rollouts": 6, "sampled_tokens": sampled_count,
        "supervised_tokens": supervised_count,
        "directive_zero_tokens": sampled_count - supervised_count,
        "cold_native_processes": 18, "model_weight_files_rehashed": len(actual_weights),
        "tokenizer_files_rehashed": len(tokenizer_files),
        "max_abs_per_token_delta": float(maximum_token),
        "max_abs_objective_delta": float(maximum_objective),
        "per_token_tolerance": float(TOKEN_TOLERANCE),
        "objective_tolerance": float(OBJECTIVE_TOLERANCE),
        "raw_sse_reparsed": 6, "complete_one_to_one_group_membership": True,
        "eos_excluded": True, "causal_packing_exact": True,
        "raw_fp32_rows_exact_across_policies": True,
        "arithmetic_precision": "DECIMAL_50_DIGIT_INDEPENDENT_EXP_AND_MEANS",
        "arithmetic_controls": controls, "group_losses": per_group_losses,
        "optimizer": "NOT_RUN", "optimizer_resume_update": "NOT_RUN",
        "model_quality": "NOT_RUN", "generalized_improvement": "NOT_RUN",
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--replay-binary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        require(args.root.is_absolute() and args.root.is_dir(),
                "evidence root must be an absolute directory")
        require(args.model_dir.is_absolute() and args.model_dir.is_dir() and
                args.replay_binary.is_absolute() and args.replay_binary.is_file(),
                "model and replay binary paths must be absolute")
        require(args.report.is_absolute() and not args.report.exists(),
                "verification report must be a new absolute path")
        result = verify(args.root, args.model_dir, args.replay_binary)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(canonical(result) + b"\n")
        print("ONLINE_GRPO_REAL_MODEL_INDEPENDENT_VERIFY: PASS "
              f"groups=2 rollouts=6 sampled={result['sampled_tokens']} "
              f"max_token_delta={result['max_abs_per_token_delta']} "
              f"max_objective_delta={result['max_abs_objective_delta']}")
        return 0
    except (VerificationError, OSError, KeyError, IndexError, TypeError, ValueError,
            json.JSONDecodeError) as error:
        print(f"independent verification rejected: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
