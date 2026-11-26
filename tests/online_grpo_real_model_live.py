#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real Qwen3.8 online-GRPO preflight and sampled/logprob alignment gate.

Reuses an accepted Task #49 exposed rollout bundle only after independently
rehashing its exact model, tokenizer, configuration, and producer binaries.
Adds token_export v2 supervision, real prompt packing, and separate serial cold
native behavior/current/reference forwards on physical GPU 7. No optimizer or
model-quality evaluation is executed.
"""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import behavior_logprobs_live as behavior_tools  # noqa: E402
import online_rollout_live as rollout_live  # noqa: E402
import sse_replay_live as replay_tools  # noqa: E402
import umminfer_behavior_token_alignment as alignment  # noqa: E402
import umminfer_online_rollout as online  # noqa: E402

TOKEN_TOLERANCE = 1e-9
OBJECTIVE_TOLERANCE = 1e-12
POLICIES = ("behavior", "current", "reference")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def canonical(value):
    return online.canonical_json(value)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    return replay_tools.sha256_file(path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(
        encoding="utf-8").splitlines() if line]


def run_command(command, log_path, *, timeout=1800):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            timeout=timeout, check=False)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).write_text(result.stdout + result.stderr, encoding="utf-8")
    require(result.returncode == 0,
            f"command failed ({result.returncode}): {result.stdout}{result.stderr}")
    return result


def accepted_rows(bundle):
    rows = read_jsonl(Path(bundle) / "pack/grpo_train.jsonl") + read_jsonl(
        Path(bundle) / "pack/grpo_eval.jsonl")
    rows.sort(key=lambda row: row["rollout_provenance"]["group_index"])
    return rows


def validate_retained_source(args, out_dir):
    summary = read_json(args.source_live_summary)
    require(summary.get("status") == "PASS" and
            summary.get("plan") == {"groups": 2, "seeds_per_group": 3},
            "retained live source is not the accepted 2x3 gate")
    require(summary["claims"]["optimizer"] == "NOT_RUN" and
            summary["claims"]["model_quality"] == "NOT_RUN",
            "retained source contains an optimizer or quality claim")
    identity = summary["native_identity"]
    require(identity["schema"] == "umminfer-native-stage-identity-v2" and
            identity["state"] == "loaded" and identity["stage_config"]["gpu"] == 7 and
            identity["stage_config"]["batch_size"] == 1,
            "retained source was not a loaded serial GPU 7 native stage")
    require(Path(identity["model_root"]).resolve() == args.model_dir.resolve(),
            "retained rollout model directory differs from the requested exact model")
    process = identity["process"]
    require(isinstance(process.get("pid"), int) and process["pid"] > 0 and
            isinstance(process.get("start_time_ticks"), int) and
            process["start_time_ticks"] > 0,
            "retained native cold-process identity is incomplete")

    provenance, config_bytes = replay_tools.model_provenance(args.model_dir)
    provenance["config_sha256"] = sha256_bytes(config_bytes)
    provenance["model_sha256"] = sha256_bytes(canonical(provenance["available_weights"]))
    provenance["tokenizer_sha256"] = behavior_tools.tokenizer_digest(args.model_dir)
    for key in ("model_sha256", "config_sha256", "tokenizer_sha256"):
        require(provenance[key] == summary["provenance"][key],
                f"current exact model {key} differs from retained rollouts")
    require(identity["model_sha256"] == provenance["model_sha256"] and
            identity["config"]["sha256"] == provenance["config_sha256"] and
            set(identity["weight_files"]) == set(provenance["available_weights"]),
            "loaded native identity differs from independently rehashed model files")
    for relative, measured in provenance["available_weights"].items():
        require(identity["weight_files"][relative] == measured,
                f"loaded weight identity differs: {relative}")

    binary_paths = {
        "qwen_stage": Path(identity["process"]["executable_path"]),
        "native_replay": Path(summary["fresh_forward"][0]["command"][0]),
        "gateway": Path(next(item["command"][0] for item in summary["stack_commands"]
                             if item["name"] == "gateway")),
    }
    binaries = {}
    for name, path in binary_paths.items():
        measured = sha256_file(path)
        require(measured == summary["provenance"]["binaries"][name],
                f"retained {name} producer binary hash differs")
        binaries[name] = measured
    replay_binary = sha256_file(args.replay_binary)
    require(replay_binary["sha256"] != "",
            "independent replay binary could not be hashed")

    source = args.source_rollout_bundle
    plan_bytes = (source / "plan.json").read_bytes()
    plan = online.prepare_plan(json.loads(plan_bytes.decode("utf-8")))
    require(plan_bytes == online._canonical_line(plan), "retained plan is not canonical")
    require(plan["behavior_policy"]["model_sha256"] == provenance["model_sha256"] and
            plan["behavior_policy"]["model_config_sha256"] == provenance["config_sha256"],
            "retained plan model identity differs from exact local model")
    schedule, schedule_sha = online.build_schedule(plan)
    records = read_jsonl(source / "trajectories.jsonl")
    require(len(records) == len(schedule) == 6, "retained source is not six rollouts")
    by_rollout = {}
    for item, record in zip(schedule, records):
        require(record["rollout_provenance"]["rollout_id"] == item["rollout_id"],
                "retained source order differs from schedule")
        online.validate_online_record(record, item, plan, schedule_sha, source / "artifacts")
        behavior = record["rollout_provenance"]["behavior"]
        require(record["rollout_provenance"]["response"]["behavior_evidence"] == "exposed"
                and behavior["status"] == "COVERED" and
                record["rollout_provenance"]["response"]["generation"] == "fresh",
                "retained source is not fresh exposed behavior evidence")
        by_rollout[item["rollout_id"]] = record

    original_forwards = {}
    for item in schedule:
        rollout_id = item["rollout_id"]
        path = args.source_live_summary.parent / "logs" / (rollout_id + "-forward.stdout")
        document = json.loads(path.read_text(encoding="utf-8").strip().splitlines()[-1])
        expected = by_rollout[rollout_id]["rollout_provenance"]["behavior"]
        require(document["behavior"]["token_ids"] == expected["token_ids"] and
                document["behavior"]["mask"] == expected["mask"] and
                document["clean_text"] == by_rollout[rollout_id]["clean_text"],
                "retained fresh-forward ids/masks differ from raw accepted evidence")
        original_forwards[rollout_id] = document

    retained = {
        "source_summary_sha256": sha256_file(args.source_live_summary)["sha256"],
        "source_trajectory_sha256": sha256_file(source / "trajectories.jsonl")["sha256"],
        "source_pack_manifest_sha256": sha256_file(source / "pack/manifest.json")["sha256"],
        "source_plan_sha256": online.plan_sha256(plan),
        "source_schedule_sha256": schedule_sha,
        "model": provenance,
        "producer_binaries": binaries,
        "independent_replay_binary": replay_binary,
        "source_native_process": process,
        "model_files_rehashed": len(provenance["available_weights"]) + 1,
        "tokenizer_files_rehashed": len(provenance["config_files"]),
        "source_rollouts_revalidated": 6,
    }
    write_json(out_dir / "source_identity.json", retained)
    return summary, plan, schedule, schedule_sha, records, original_forwards, retained


def attach_exports_and_preflight(args, out_dir, plan, schedule, schedule_sha,
                                 source_records, original_forwards, retained):
    exported_path = out_dir / "config/exported-trajectories.jsonl"
    tokenizer_identity = {
        "tokenizer_id": "qwen38-native-loaded-tokenizer",
        "tokenizer_revision": args.model_revision,
        "tokenization_version": "qwen38-fast-tokenizer-export-v2",
    }
    export_command = [
        sys.executable, "-I", str(ROOT / "python/umminfer_token_export.py"),
        str(args.source_rollout_bundle / "trajectories.jsonl"),
        "--out", str(exported_path), "--tokenizer-dir", str(args.model_dir),
        "--tokenizer-id", tokenizer_identity["tokenizer_id"],
        "--tokenizer-revision", tokenizer_identity["tokenizer_revision"],
        "--tokenization-version", tokenizer_identity["tokenization_version"],
        "--token-export-version", "2",
    ]
    exported_path.parent.mkdir(parents=True, exist_ok=True)
    export_result = run_command(export_command, out_dir / "logs/token-export.log")
    exported = read_jsonl(exported_path)
    require(len(exported) == len(source_records) == 6 and
            all(record.get("token_export", {}).get("format_version") == 2
                for record in exported),
            "real token_export v2 attachment is incomplete")

    accepted = out_dir / "accepted"
    accepted.mkdir()
    shutil.copyfile(args.source_rollout_bundle / "plan.json", accepted / "plan.json")
    shutil.copytree(args.source_rollout_bundle / "artifacts", accepted / "artifacts")
    # Rebuild all pack/manifest bytes from the raw-validated additive records.
    online._export_replayed(plan, schedule, schedule_sha, exported, accepted)
    require((accepted / "trajectories.jsonl").read_bytes() == exported_path.read_bytes(),
            "additive token exports changed during accepted pack reconstruction")

    # The fresh export above already re-encodes and strongly validates every
    # record with the exact fast tokenizer; do not spend a second full pass.
    for item, record in zip(schedule, exported):
        evidence = record["rollout_provenance"]["behavior"]
        report = alignment.validate_online_record_alignment(record, evidence, args.model_dir)
        require(report["status"] == "PASS" and
                report["sampled_posthoc_target_ids_equal"] is True,
                f"real sampled/token alignment failed for {item['rollout_id']}")

    replayed = out_dir / "replayed"
    replay_command = [
        sys.executable, "-I", str(ROOT / "python/umminfer_online_rollout.py"),
        "replay", "--input", str(accepted), "--out", str(replayed),
    ]
    run_command(replay_command, out_dir / "logs/fresh-replay.log")
    for relative in online._accepted_relative_files(accepted):
        require((accepted / relative).read_bytes() == (replayed / relative).read_bytes(),
                f"fresh additive replay differs: {relative}")

    rows = accepted_rows(accepted)
    prompt_ids_by_hash = {}
    prompt_members = {}
    for item in schedule:
        row = next(row for row in rows if row["rollout_provenance"]["schedule_group_id"]
                   == item["group_id"])
        prompt_sha = row["prompt_sha256"]
        ids = original_forwards[item["rollout_id"]]["prompt_token_ids"]
        require(ids and all(isinstance(token, int) and not isinstance(token, bool) and
                            token >= 0 for token in ids),
                "retained native prompt ids are invalid")
        if prompt_sha in prompt_ids_by_hash:
            require(prompt_ids_by_hash[prompt_sha] == ids,
                    "exact native prompt ids drifted within a scheduled group")
        else:
            prompt_ids_by_hash[prompt_sha] = ids
        prompt_members.setdefault(prompt_sha, []).append(item["rollout_id"])
    require(len(prompt_ids_by_hash) == 2, "real preflight does not have two exact prompts")

    behavior_policy = plan["behavior_policy"]
    def policy(policy_id, revision):
        return {
            "policy_id": policy_id, "revision": revision,
            "model_id": behavior_policy["model_id"],
            "model_revision": args.model_revision,
            "model_sha256": retained["model"]["model_sha256"],
            "model_config_sha256": retained["model"]["config_sha256"],
            "tokenizer_sha256": retained["model"]["tokenizer_sha256"],
        }
    config = {
        "schema": "umminfer-online-grpo-preflight-config-v1",
        "current_policy": policy("qwen38-current-real-model", args.source_commit),
        "reference_policy": policy("qwen38-frozen-reference-real-model", args.source_commit),
        "objective": {
            "clip_eps": 0.2, "kl_coef": 0.04, "log_ratio_clamp": 10.0,
            "current_logprob_offsets": [0.0], "reference_logprob_offsets": [0.0],
        },
        "prompt_token_ids": prompt_ids_by_hash,
        "groups_per_step": 1,
    }
    config_path = out_dir / "config/online-grpo-real-config.json"
    write_json(config_path, config)
    preflight = out_dir / "preflight"
    preflight_command = [
        sys.executable, "-I", str(ROOT / "python/umminfer_online_grpo_preflight.py"),
        "--preflight-only", "--online-pack", str(accepted),
        "--config", str(config_path), "--tokenizer-dir", str(args.model_dir),
        "--out", str(preflight),
    ]
    run_command(preflight_command, out_dir / "logs/online-grpo-preflight.log")
    verified = out_dir / "preflight-reconstructed"
    verify_command = [
        sys.executable, "-I", str(ROOT / "python/umminfer_online_grpo_preflight.py"),
        "--verify-bundle", str(preflight), "--online-pack", str(accepted),
        "--tokenizer-dir", str(args.model_dir), "--out", str(verified),
    ]
    run_command(verify_command, out_dir / "logs/online-grpo-preflight-verify.log")
    require(sorted(path.relative_to(preflight) for path in preflight.rglob("*")
                   if path.is_file()) ==
            sorted(path.relative_to(verified) for path in verified.rglob("*")
                   if path.is_file()), "preflight reconstruction file set differs")
    for path in preflight.rglob("*"):
        if path.is_file():
            relative = path.relative_to(preflight)
            require(path.read_bytes() == (verified / relative).read_bytes(),
                    f"preflight reconstruction differs: {relative}")

    audit_command = [
        sys.executable, "-I", str(ROOT / "python/umminfer_online_evidence_audit.py"),
        "preflight", "--online-pack", str(accepted), "--preflight-bundle", str(preflight),
        "--tokenizer-dir", str(args.model_dir), "--repo-root", str(ROOT),
        "--report", str(out_dir / "independent-evidence-audit.json"),
    ]
    run_command(audit_command, out_dir / "logs/independent-evidence-audit.log")
    audit = read_json(out_dir / "independent-evidence-audit.json")
    require(audit.get("status") == "PASS" and
            audit.get("checkpoint_disagreements") == [],
            "Task #52 independent evidence audit did not pass without disagreements")
    return {
        "accepted": accepted, "records": exported, "rows": rows, "config": config,
        "preflight": preflight, "tokenizer_identity": tokenizer_identity,
        "export_command": export_command, "export_stdout": export_result.stdout.strip(),
        "accepted_files": len(online._accepted_relative_files(accepted)),
        "preflight_run_id": read_json(preflight / "run_identity.json")["run_id"],
        "independent_audit_sha256": sha256_file(
            out_dir / "independent-evidence-audit.json")["sha256"],
        "strong_tokenizer_reencode": "PASS_FRESH_EXPORT",
    }


def process_identity(process, binary):
    identity = {"pid": process.pid, "executable_path": str(binary)}
    proc = Path(f"/proc/{process.pid}")
    try:
        identity["executable"] = os.readlink(proc / "exe")
        fields = (proc / "stat").read_text(encoding="utf-8").rsplit(") ", 1)[1].split()
        identity["start_time_ticks"] = int(fields[19])
    except (OSError, IndexError, ValueError):
        pass
    return identity


def reuse_validated_preflight(args, out_dir):
    """Reuse R3 artifacts only after exact identity and retained replay checks."""
    source = args.reuse_preflight_from
    require(source.is_absolute() and source.is_dir(),
            "--reuse-preflight-from must be an absolute evidence directory")
    require(read_json(source / "source_identity.json") ==
            read_json(out_dir / "source_identity.json"),
            "retained preflight model/rollout/binary identity differs")
    require(read_json(source / "summary.json")["status"] == "FAIL" and
            "independent=2 evidence=0" in read_json(source / "summary.json").get("error", ""),
            "retained preflight is not the sealed R3 retry-audit failure")
    (out_dir / "config").mkdir(exist_ok=True)
    for name in ("accepted", "replayed", "preflight", "preflight-reconstructed"):
        require((source / name).is_dir(), f"retained preflight directory is missing: {name}")
        shutil.copytree(source / name, out_dir / name)
    for name in ("exported-trajectories.jsonl", "online-grpo-real-config.json"):
        shutil.copyfile(source / "config" / name, out_dir / "config" / name)

    accepted = out_dir / "accepted"
    for relative in online._accepted_relative_files(accepted):
        require((accepted / relative).read_bytes() ==
                (out_dir / "replayed" / relative).read_bytes(),
                f"retained accepted replay differs: {relative}")
    manifest_sidecar = (accepted / "rollout_manifest.json.sha256").read_text(
        encoding="ascii").strip()
    require(manifest_sidecar == sha256_file(accepted / "rollout_manifest.json")["sha256"],
            "retained accepted manifest sidecar differs")
    preflight = out_dir / "preflight"
    checked = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=preflight,
                             capture_output=True, text=True, check=False)
    (out_dir / "logs/reused-preflight-sha256.log").write_text(
        checked.stdout + checked.stderr, encoding="utf-8")
    require(checked.returncode == 0, "retained preflight SHA256SUMS failed")
    reconstructed = out_dir / "preflight-reconstructed"
    preflight_files = sorted(path.relative_to(preflight) for path in preflight.rglob("*")
                             if path.is_file())
    reconstructed_files = sorted(path.relative_to(reconstructed)
                                 for path in reconstructed.rglob("*") if path.is_file())
    require(preflight_files == reconstructed_files,
            "retained preflight reconstruction file set differs")
    for relative in preflight_files:
        require((preflight / relative).read_bytes() ==
                (reconstructed / relative).read_bytes(),
                f"retained preflight reconstruction differs: {relative}")

    audit_command = [
        sys.executable, "-I", str(ROOT / "python/umminfer_online_evidence_audit.py"),
        "preflight", "--online-pack", str(accepted), "--preflight-bundle", str(preflight),
        "--tokenizer-dir", str(args.model_dir), "--repo-root", str(ROOT),
        "--report", str(out_dir / "independent-evidence-audit.json"),
    ]
    run_command(audit_command, out_dir / "logs/independent-evidence-audit.log")
    audit = read_json(out_dir / "independent-evidence-audit.json")
    require(audit.get("status") == "PASS" and
            audit.get("checkpoint_disagreements") == [],
            "corrected independent retry/preflight audit did not pass")
    return {
        "accepted": accepted, "records": read_jsonl(accepted / "trajectories.jsonl"),
        "rows": accepted_rows(accepted),
        "config": read_json(out_dir / "config/online-grpo-real-config.json"),
        "preflight": preflight, "tokenizer_identity": {
            "tokenizer_id": "qwen38-native-loaded-tokenizer",
            "tokenizer_revision": args.model_revision,
            "tokenization_version": "qwen38-fast-tokenizer-export-v2",
        },
        "export_command": ["REUSED_EXACT_R3_STRONG_EXPORT_AND_PREFLIGHT"],
        "export_stdout": "REUSED_AFTER_EXACT_IDENTITY_REPLAY_AND_CHECKSUM_CHECKS",
        "accepted_files": len(online._accepted_relative_files(accepted)),
        "preflight_run_id": read_json(preflight / "run_identity.json")["run_id"],
        "independent_audit_sha256": sha256_file(
            out_dir / "independent-evidence-audit.json")["sha256"],
        "strong_tokenizer_reencode": "PASS_RETAINED_R3_FRESH_EXPORT",
        "reused_source_summary_sha256": sha256_file(source / "summary.json")["sha256"],
    }


def independent_forward(args, item, record, policy, out_dir):
    body = item["body"]
    command = [
        str(args.replay_binary), "--model", str(args.model_dir),
        "--prompt", body["messages"][0]["content"], "--gpu", "7",
        "--max-new-tokens", str(body["max_tokens"]),
        "--temperature", str(body["temperature"]), "--top-p", str(body["top_p"]),
        "--seed", str(body["seed"]), "--independent-behavior-evidence",
        "--narrative-auto",
    ]
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["UMM_QWEN38_PREFIX_CACHE"] = "0"
    destination = out_dir / "forwards" / policy
    destination.mkdir(parents=True, exist_ok=True)
    stdout_path = destination / (item["rollout_id"] + ".stdout")
    stderr_path = destination / (item["rollout_id"] + ".stderr")
    before = rollout_live.gpu7_snapshot()
    rollout_live.require_gpu7_idle(before)
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=env,
                                   stdout=stdout, stderr=stderr)
        identity = process_identity(process, args.replay_binary)
        try:
            returncode = process.wait(timeout=args.timeout)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            raise
    require(returncode == 0,
            f"independent {policy} forward failed for {item['rollout_id']}: "
            + stderr_path.read_text(encoding="utf-8", errors="replace"))
    after = rollout_live.gpu7_snapshot()
    rollout_live.require_gpu7_idle(after)
    require(identity.get("start_time_ticks", 0) > 0 and
            Path(identity.get("executable", "")).resolve() == args.replay_binary.resolve(),
            "fresh native process identity could not be bound to its executable")
    document = json.loads(stdout_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    evidence = document.get("independent_behavior")
    require(isinstance(evidence, dict) and
            evidence.get("format") == "umminfer-independent-presampling-logsoftmax-v1" and
            evidence.get("producer_scoring_used") is False,
            "forward did not use independent raw-logit scoring")
    expected = record["rollout_provenance"]["behavior"]
    require(evidence["token_ids"] == expected["token_ids"] and
            evidence["mask"] == expected["mask"] and
            evidence["tokenizer_sha256"] == expected["tokenizer_sha256"] and
            document["clean_text"] == record["clean_text"],
            f"independent {policy} sampled ids/masks/tokenizer/clean text differ")
    require(document["prompt_token_ids"] and
            len(evidence["rows"]) == len(evidence["logprobs"]) == len(evidence["token_ids"]),
            "independent raw-logit rows are incomplete")
    for row in evidence["rows"]:
        values = [row[key] for key in ("selected_raw_logit", "max_raw_logit",
                                       "log_normalizer", "logprob")]
        require(all(math.isfinite(float(value)) for value in values) and
                float(row["logprob"]) <= 0.0 and
                len(row["raw_logits_sha256"]) == 64,
                "independent raw-logit row contains invalid values")
    write_json(destination / (item["rollout_id"] + ".json"), document)
    identity.update({
        "policy": policy, "rollout_id": item["rollout_id"], "returncode": returncode,
        "command": command, "gpu7_before": before, "gpu7_after": after,
        "stdout_sha256": sha256_file(stdout_path)["sha256"],
        "stderr_sha256": sha256_file(stderr_path)["sha256"],
        "document_sha256": sha256_file(destination / (item["rollout_id"] + ".json"))["sha256"],
    })
    return document, identity


def objective_metrics(behavior, current, reference, mask, advantage, objective):
    supervised = [index for index, bit in enumerate(mask) if bit == 1]
    require(supervised, "sampled response has no supervised tokens")
    behavior_mean = math.fsum(behavior[index] for index in supervised) / len(supervised)
    current_mean = math.fsum(current[index] for index in supervised) / len(supervised)
    reference_mean = math.fsum(reference[index] for index in supervised) / len(supervised)
    clamp = float(objective["log_ratio_clamp"])
    raw_log_ratio = current_mean - behavior_mean
    log_ratio = max(-clamp, min(clamp, raw_log_ratio))
    ratio = math.exp(log_ratio)
    epsilon = float(objective["clip_eps"])
    clipped = max(1.0 - epsilon, min(1.0 + epsilon, ratio))
    first = ratio * advantage
    second = clipped * advantage
    selected = min(first, second)
    raw_kl = reference_mean - current_mean
    kl_difference = max(-clamp, min(clamp, raw_kl))
    kl = math.exp(kl_difference) - kl_difference - 1.0
    loss = -selected + float(objective["kl_coef"]) * kl
    require(all(math.isfinite(value) for value in (
        behavior_mean, current_mean, reference_mean, ratio, clipped, first, second,
        selected, kl, loss)) and ratio > 0.0, "real objective is non-finite")
    return {
        "supervised_response_indices": supervised,
        "behavior_mean_logp": behavior_mean, "policy_mean_logp": current_mean,
        "reference_mean_logp": reference_mean,
        "raw_current_minus_behavior_logp": raw_log_ratio,
        "current_minus_behavior_logp": log_ratio, "ratio": ratio,
        "clipped_ratio": clipped, "advantage": advantage,
        "unclipped_surrogate": first, "clipped_surrogate": second,
        "selected_surrogate": selected, "clip_active": first > second,
        "raw_reference_minus_current_logp": raw_kl,
        "reference_minus_current_logp": kl_difference, "schulman_kl": kl,
        "objective_loss": loss,
    }


def build_real_objectives(out_dir, source_commit, retained, prepared, forwards):
    preflight_dir = prepared["preflight"]
    fixture = read_json(preflight_dir / "objective_fixtures.json")
    fixture_members = {member["rollout_id"]: member for group in fixture["groups"]
                       for member in group["members"]}
    records = {record["rollout_provenance"]["rollout_id"]: record
               for record in prepared["records"]}
    config = prepared["config"]
    groups = []
    maxima = {"behavior_vs_sampled": 0.0, "current_vs_behavior": 0.0,
              "reference_vs_behavior": 0.0}
    for row in prepared["rows"]:
        schedule_group = row["rollout_provenance"]["schedule_group_id"]
        fixture_group = next(group for group in fixture["groups"]
                             if group["schedule_group_id"] == schedule_group)
        members = []
        for rollout in row["rollouts"]:
            provenance = rollout["rollout_provenance"]
            rollout_id = provenance["rollout_id"]
            record = records[rollout_id]
            expected = record["rollout_provenance"]["behavior"]
            packed = copy.deepcopy(fixture_members[rollout_id]["packing"])
            docs = {policy: forwards[policy][rollout_id] for policy in POLICIES}
            evidence = {policy: docs[policy]["independent_behavior"] for policy in POLICIES}
            prompt_ids = docs["behavior"]["prompt_token_ids"]
            require(all(docs[policy]["prompt_token_ids"] == prompt_ids for policy in POLICIES),
                    "current/reference prompt packing differs")
            require(packed["prompt_token_ids"] == prompt_ids and
                    packed["sampled_response_token_ids"] == expected["token_ids"] and
                    packed["input_ids"] == prompt_ids + expected["token_ids"] and
                    packed["labels"] == [-100] * len(prompt_ids) + [
                        token if bit else -100 for token, bit in
                        zip(expected["token_ids"], expected["mask"])] and
                    packed["response_mask"] == [0] * len(prompt_ids) + expected["mask"],
                    "real current/behavior/reference packing differs from preflight")
            sampled = [float(value) for value in expected["logprobs"]]
            independent = {policy: [float(value) for value in evidence[policy]["logprobs"]]
                           for policy in POLICIES}
            deltas = {
                "behavior_vs_sampled": [abs(a - b) for a, b in
                                        zip(independent["behavior"], sampled)],
                "current_vs_behavior": [abs(a - b) for a, b in
                                        zip(independent["current"], independent["behavior"])],
                "reference_vs_behavior": [abs(a - b) for a, b in
                                          zip(independent["reference"], independent["behavior"])],
            }
            for name, values in deltas.items():
                require(values and max(values) <= TOKEN_TOLERANCE,
                        f"{name} exceeds {TOKEN_TOLERANCE} for {rollout_id}")
                maxima[name] = max(maxima[name], max(values))
            raw_hashes = [[row["raw_logits_sha256"] for row in evidence[policy]["rows"]]
                          for policy in POLICIES]
            require(raw_hashes[0] == raw_hashes[1] == raw_hashes[2],
                    "cold policies did not observe identical exact FP32 raw-logit rows")
            metrics = objective_metrics(sampled, independent["current"],
                                        independent["reference"], expected["mask"],
                                        float(rollout["advantage"]), config["objective"])
            member = {
                "rollout_id": rollout_id, "rank": rollout["rank"],
                "request_id": record["request_id"], "session_id": record["session_id"],
                "seed": provenance["seed"], "reward": rollout["score"],
                "advantage": rollout["advantage"],
                "source_record_sha256": sha256_bytes(canonical(record)),
                "raw_sse_sha256": record["sse_replay"]["raw_sha256"],
                "behavior_evidence_sha256": expected["evidence_sha256"],
                "token_export_sha256": record["token_export"]["export_sha256"],
                "token_export_source_sha256": record["token_export"]["source_record_sha256"],
                "token_ids": expected["token_ids"], "mask": expected["mask"],
                "prompt_token_ids": prompt_ids, "input_ids": packed["input_ids"],
                "labels": packed["labels"], "response_mask": packed["response_mask"],
                "packed_ids_labels_masks_sha256": packed["packed_ids_labels_masks_sha256"],
                "sampled_behavior_logprobs": sampled,
                "independent_logprobs": independent,
                "independent_raw_rows": {policy: evidence[policy]["rows"]
                                         for policy in POLICIES},
                "per_token_abs_deltas": deltas,
                "metrics": metrics,
                "native_forward_document_sha256": {
                    policy: sha256_bytes(canonical(docs[policy])) for policy in POLICIES},
            }
            members.append(member)
        groups.append({
            "schedule_group_id": schedule_group, "dataset_group_id": row["group_id"],
            "group_export_sha256": sha256_bytes(canonical(row)),
            "prompt_sha256": row["prompt_sha256"], "group_mean": row["group_mean"],
            "group_std": row["group_std"], "has_reward_variance": row["has_reward_variance"],
            "members": members,
            "objective_loss": math.fsum(member["metrics"]["objective_loss"]
                                        for member in members) / len(members),
            "normalization": "ROLLOUT_MEAN",
            "preflight_group_export_sha256": fixture_group["group_export_sha256"],
        })
    require(len(groups) == 2 and sum(len(group["members"]) for group in groups) == 6,
            "real objective groups are incomplete")
    return {
        "schema": "umminfer-online-grpo-real-model-objectives-v1",
        "status": "PASS", "source_commit": source_commit,
        "preflight_run_id": prepared["preflight_run_id"],
        "model": {
            "model_id": config["current_policy"]["model_id"],
            "model_revision": config["current_policy"]["model_revision"],
            "model_sha256": retained["model"]["model_sha256"],
            "model_config_sha256": retained["model"]["config_sha256"],
            "tokenizer_sha256": retained["model"]["tokenizer_sha256"],
        },
        "policies": {
            "behavior": read_json(prepared["accepted"] / "plan.json")["behavior_policy"],
            "current": config["current_policy"], "reference": config["reference_policy"],
        },
        "objective": config["objective"], "groups": groups,
        "token_counts": {
            "sampled": sum(len(member["token_ids"]) for group in groups
                           for member in group["members"]),
            "supervised": sum(sum(member["mask"]) for group in groups
                              for member in group["members"]),
            "directive_zero": sum(len(member["mask"]) - sum(member["mask"])
                                  for group in groups for member in group["members"]),
        },
        "max_abs_token_deltas": maxima,
        "tolerances": {"per_token_abs": TOKEN_TOLERANCE,
                       "objective_abs": OBJECTIVE_TOLERANCE},
        "raw_evidence_semantics": {
            "logits": "FP32 pre-sampling LM-head rows; one SHA-256 per emitted non-EOS token",
            "independent_native": "long-double max/sum/log; no producer scoring function",
            "objective": "masked response means, clipped surrogate, separate-reference Schulman KL",
            "token_export": "v2 exact raw-target ids/source spans/masks; no clean-id substitution",
        },
        "claims": {
            "optimizer": "NOT_RUN", "optimizer_resume_update": "NOT_RUN",
            "REAL_TRAINING": "NOT_RUN", "model_weights_updated": False,
            "model_quality": "NOT_RUN", "generalized_improvement": "NOT_RUN",
            "synthetic_preflight_objectives": "KEPT_SEPARATE_NOT_MODEL_OUTPUT",
        },
    }


def run_independent_verifier(args, out_dir, report_name="independent-verification.json"):
    report = out_dir / report_name
    command = [
        sys.executable, "-I", str(ROOT / "tests/online_grpo_real_model_verify.py"),
        str(out_dir), "--model-dir", str(args.model_dir),
        "--replay-binary", str(args.replay_binary), "--report", str(report),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            timeout=1800, check=False)
    (out_dir / "logs/independent-verification.log").write_text(
        result.stdout + result.stderr, encoding="utf-8")
    require(result.returncode == 0 and report.is_file(),
            f"independent real-model verification failed: {result.stdout}{result.stderr}")
    return read_json(report)


def run_tamper_controls(args, out_dir):
    cases = ["objective-value", "current-logprob", "directive-mask", "group-membership"]
    results = []
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="umm-online-grpo-real-tamper-") as tmp:
            copied = Path(tmp) / "evidence"
            shutil.copytree(out_dir, copied, ignore=shutil.ignore_patterns(
                "SHA256SUMS", "summary.json", "tamper-controls.json"))
            objectives_path = copied / "real_objectives.json"
            objectives = read_json(objectives_path)
            first_id = objectives["groups"][0]["members"][0]["rollout_id"]
            if case == "objective-value":
                objectives["groups"][0]["objective_loss"] += 0.125
            elif case == "current-logprob":
                path = copied / "forwards/current" / (first_id + ".json")
                document = read_json(path)
                document["independent_behavior"]["logprobs"][0] += 0.01
                write_json(path, document)
            elif case == "directive-mask":
                visual = next(member for group in objectives["groups"]
                              for member in group["members"] if 0 in member["mask"])
                index = visual["mask"].index(0)
                path = copied / "forwards/behavior" / (visual["rollout_id"] + ".json")
                document = read_json(path)
                document["independent_behavior"]["mask"][index] = 1
                write_json(path, document)
            elif case == "group-membership":
                first = objectives["groups"][0]["members"][0]
                second = objectives["groups"][1]["members"][0]
                first["rollout_id"], second["rollout_id"] = second["rollout_id"], first["rollout_id"]
            write_json(objectives_path, objectives)
            report = copied / "tampered-verification.json"
            command = [
                sys.executable, "-I", str(ROOT / "tests/online_grpo_real_model_verify.py"),
                str(copied), "--model-dir", str(args.model_dir),
                "--replay-binary", str(args.replay_binary), "--report", str(report),
            ]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                    timeout=1800, check=False)
            require(result.returncode != 0 and not report.exists(),
                    f"tamper case {case} was accepted or emitted a report")
            results.append({"case": case, "returncode": result.returncode,
                            "rejected": True, "report_emitted": False,
                            "diagnostic_sha256": sha256_bytes(
                                (result.stdout + result.stderr).encode("utf-8"))})
    write_json(out_dir / "tamper-controls.json", {
        "schema": "umminfer-online-grpo-real-model-tamper-v1",
        "status": "PASS", "cases": results,
        "all_rejected_fresh_process_without_report": True,
    })
    return results


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--replay-binary", type=Path, required=True)
    parser.add_argument("--source-rollout-bundle", type=Path, required=True)
    parser.add_argument("--source-live-summary", type=Path, required=True)
    parser.add_argument("--reuse-preflight-from", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args(argv)
    if os.environ.get("UMMINFER_LIVE_OPT_IN") != "1":
        print("REAL_ONLINE_GRPO_ALIGNMENT_GATE: NOT_RUN (set UMMINFER_LIVE_OPT_IN=1)")
        return 0
    for path in (args.model_dir, args.replay_binary, args.source_rollout_bundle,
                 args.source_live_summary):
        require(path.is_absolute() and path.exists(), f"required absolute path is absent: {path}")
    require(args.out_dir.is_absolute() and
            (not args.out_dir.exists() or
             (args.out_dir.is_dir() and not any(args.out_dir.iterdir()))),
            "--out-dir must be an absolute absent or empty directory")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "logs").mkdir(exist_ok=True)
    summary = {
        "schema": "umminfer-online-grpo-real-model-live-v1", "status": "RUNNING",
        "source_commit": args.source_commit, "model_id": args.model,
        "model_revision": args.model_revision,
        "plan": {"groups": 2, "rollouts_per_group": 3, "cold_native_passes": 18},
        "claims": {"optimizer": "NOT_RUN", "optimizer_resume_update": "NOT_RUN",
                   "REAL_TRAINING": "NOT_RUN", "model_quality": "NOT_RUN",
                   "generalized_improvement": "NOT_RUN", "model_weights_updated": False,
                   "resident_service": "untouched", "gpu": 7},
    }
    try:
        summary["gpu7_before"] = rollout_live.gpu7_snapshot()
        rollout_live.require_gpu7_idle(summary["gpu7_before"])
        (source_summary, plan, schedule, schedule_sha, source_records,
         original_forwards, retained) = validate_retained_source(args, args.out_dir)
        summary["retained_source"] = {key: value for key, value in retained.items()
                                      if key not in ("model", "producer_binaries")}
        summary["model_identity"] = {
            key: retained["model"][key] for key in
            ("model_sha256", "config_sha256", "tokenizer_sha256", "model_type",
             "model_architectures", "revision")}
        summary["producer_binary_sha256"] = {
            key: value["sha256"] for key, value in retained["producer_binaries"].items()}
        summary["independent_replay_binary_sha256"] = retained[
            "independent_replay_binary"]["sha256"]
        if args.reuse_preflight_from is not None:
            prepared = reuse_validated_preflight(args, args.out_dir)
        else:
            prepared = attach_exports_and_preflight(
                args, args.out_dir, plan, schedule, schedule_sha, source_records,
                original_forwards, retained)
        summary["token_export"] = {
            "format_version": 2, "records": 6,
            "strong_tokenizer_reencode": prepared["strong_tokenizer_reencode"],
            "accepted_fresh_replay_files": prepared["accepted_files"],
            "preflight_run_id": prepared["preflight_run_id"],
            "independent_audit_sha256": prepared["independent_audit_sha256"],
        }
        if "reused_source_summary_sha256" in prepared:
            summary["token_export"]["reused_source_summary_sha256"] = prepared[
                "reused_source_summary_sha256"]
        forwards = {policy: {} for policy in POLICIES}
        processes = []
        # One model load at a time. No current/reference term is reused or dropped.
        for item in schedule:
            record = prepared["records"][[entry["rollout_id"] for entry in schedule].index(
                item["rollout_id"])]
            for policy in POLICIES:
                document, identity = independent_forward(
                    args, item, record, policy, args.out_dir)
                forwards[policy][item["rollout_id"]] = document
                processes.append(identity)
        require(len({identity["pid"] for identity in processes}) == 18 and
                len({(identity["pid"], identity["start_time_ticks"])
                     for identity in processes}) == 18,
                "all 18 native forwards must be distinct cold processes")
        write_json(args.out_dir / "forward_processes.json", {
            "schema": "umminfer-online-grpo-cold-forward-schedule-v1",
            "schedule": "FOR_EACH_SCHEDULED_ROLLOUT_BEHAVIOR_THEN_CURRENT_THEN_REFERENCE",
            "serial": True, "processes": processes,
        })
        objectives = build_real_objectives(
            args.out_dir, args.source_commit, retained, prepared, forwards)
        write_json(args.out_dir / "real_objectives.json", objectives)
        summary["objectives"] = {
            "groups": len(objectives["groups"]), "rollouts": 6,
            **objectives["token_counts"],
            "max_abs_token_deltas": objectives["max_abs_token_deltas"],
            "group_losses": [group["objective_loss"] for group in objectives["groups"]],
            "real_objectives_sha256": sha256_file(
                args.out_dir / "real_objectives.json")["sha256"],
        }
        independent = run_independent_verifier(args, args.out_dir)
        summary["independent_verification"] = independent
        summary["tamper_controls"] = run_tamper_controls(args, args.out_dir)
        summary["gpu7_final"] = rollout_live.gpu7_snapshot()
        rollout_live.require_gpu7_idle(summary["gpu7_final"])
        summary["status"] = "PASS"
    except Exception as error:
        summary["status"] = "FAIL"
        summary["error"] = str(error)
        raise
    finally:
        write_json(args.out_dir / "summary.json", summary)
        files = []
        for path in sorted(args.out_dir.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                files.append(f"{sha256_file(path)['sha256']}  {path.relative_to(args.out_dir)}")
        (args.out_dir / "SHA256SUMS").write_text("\n".join(files) + "\n", encoding="ascii")
    checked = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=args.out_dir,
                             capture_output=True, text=True, check=False)
    require(checked.returncode == 0,
            f"final evidence checksums failed: {checked.stdout}{checked.stderr}")
    print("REAL_ONLINE_GRPO_ALIGNMENT_GATE: PASS groups=2 rollouts=6 "
          f"sampled={summary['objectives']['sampled']} cold_passes=18 "
          f"files={len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
