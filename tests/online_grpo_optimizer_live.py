#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fresh Qwen3.8 online rollouts, bounded LoRA updates, and exact resume.

Generation uses the public attesting server in an isolated serial GPU-7 stack.
Serving exits before independent native replay and before three separate trainer
processes: uninterrupted, split-after-one, and fresh-process resume.  No quality
or generalized-improvement evaluation is performed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import behavior_logprobs_live as behavior_tools  # noqa: E402
import online_grpo_real_model_live as real_model  # noqa: E402
import online_rollout_live as rollout_live  # noqa: E402
import sse_replay_live as replay_tools  # noqa: E402
import umminfer_online_rollout as online  # noqa: E402

TOTAL_ROLLOUTS = 6


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(
        encoding="utf-8").splitlines() if line]


def write_json(path, value):
    real_model.write_json(path, value)


def sha256_file(path):
    return replay_tools.sha256_file(path)["sha256"]


def fresh_plan(identity, source_commit, batch_id):
    """Two text-only groups with independently verifiable integer rewards."""
    prompts = (
        ("online-arithmetic-6287x5399",
         "Compute 6287 × 5399. Return only the base-10 integer, with no explanation, "
         "whitespace prefix, punctuation, code block, or other text."),
        ("online-arithmetic-8642x7351",
         "Compute 8642 × 7351. Return only the base-10 integer, with no explanation, "
         "whitespace prefix, punctuation, code block, or other text."),
    )
    plan = {
        "schema": online.PLAN_SCHEMA,
        "batch_id": batch_id,
        "behavior_policy": {
            "policy_id": "qwen38-native-online-optimizer-rollout-v1",
            "revision": source_commit,
            "model_id": identity["model_id"],
            "model_revision": identity["model_revision"],
            "model_sha256": identity["model_sha256"],
            "model_config_sha256": identity["config"]["sha256"],
        },
        "sampling": {"max_tokens": 32, "temperature": 1.4, "top_p": 0.98},
        "request_template": {"behavior": {"logprobs": True}},
        "reward": {"spec": "umminfer_rl_reward:VerifiableArithmeticReward"},
        "split_seed": batch_id + "-train-only-split-v1",
        "eval_ratio": 0.0,
        "transport": {"timeout_s": 600.0, "max_attempts": 2,
                      "retry_statuses": [503], "retry_transport": False,
                      "retry_timeouts": False},
        "groups": [
            {"prompt_id": prompts[0][0],
             "messages": [{"role": "user", "content": prompts[0][1]}],
             "expected_visual_modalities": [], "seeds": [7101, 7102, 7103]},
            {"prompt_id": prompts[1][0],
             "messages": [{"role": "user", "content": prompts[1][1]}],
             "expected_visual_modalities": [], "seeds": [-7201, -7202, -7203]},
        ],
    }
    return online.prepare_plan(plan)


def model_provenance(args):
    provenance, config_bytes = replay_tools.model_provenance(args.model_dir)
    provenance["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    provenance["model_sha256"] = hashlib.sha256(
        real_model.canonical(provenance["available_weights"])).hexdigest()
    provenance["tokenizer_sha256"] = behavior_tools.tokenizer_digest(args.model_dir)
    provenance["source_files"] = {
        name: replay_tools.sha256_file(ROOT / name) for name in (
            "python/umminfer_online_grpo_optimizer.py",
            "python/umminfer_online_grpo_preflight.py",
            "python/umminfer_online_rollout.py",
            "python/umminfer_online_evidence_audit.py",
            "python/umminfer_rl_reward.py", "python/umminfer_ablation_metrics.py",
            "tests/online_grpo_optimizer_live.py",
            "tests/online_grpo_optimizer_verify.py",
            "tests/online_rollout_live.py",
            "native/qwen38_cli/main.cpp", "native/qwen38/engine.cu",
            "src/serve/online_attestation.cpp", "src/serve/openai_chat.cpp",
        )
    }
    provenance["binaries"] = {
        "gateway": replay_tools.sha256_file(args.server_binary),
        "qwen_stage": replay_tools.sha256_file(args.stage_binary),
        "native_replay": replay_tools.sha256_file(args.replay_binary),
    }
    return provenance


def generate_and_prepare(args, out_dir, summary):
    summary["gpu7_before"] = rollout_live.gpu7_snapshot()
    rollout_live.require_gpu7_idle(summary["gpu7_before"])
    provenance = model_provenance(args)
    summary["provenance"] = provenance
    config_dir = out_dir / "config"
    config_dir.mkdir(parents=True)
    base_url, processes, launches, plan = rollout_live.start_stack(
        args, out_dir, config_dir / "plan.json", config_dir / "server.json",
        config_dir / "native-identity.json", plan_factory=fresh_plan)
    summary["stack_commands"] = launches
    summary["native_identity"] = read_json(config_dir / "native-identity.json")
    summary["server_config"] = read_json(config_dir / "server.json")
    try:
        command = [sys.executable, "-I", str(ROOT / "python/umminfer_online_rollout.py"),
                   "run", "--plan", str(config_dir / "plan.json"),
                   "--endpoint", base_url, "--out", str(out_dir / "raw-accepted")]
        log_path = out_dir / "logs/online-client.log"
        with log_path.open("wb") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT)
            try:
                returncode = process.wait(timeout=args.timeout)
            except BaseException:
                process.terminate()
                process.wait(timeout=15)
                raise
        summary["online_client"] = {"command": command, "returncode": returncode,
                                    "log_sha256": sha256_file(log_path)}
        require(returncode == 0, "fresh public online rollout client failed")
        attempts = read_jsonl(out_dir / "raw-accepted/attempts.jsonl")
        summary["attempts"] = [{
            "rollout_id": record["rollout_provenance"]["rollout_id"],
            "attempt": record["rollout_provenance"]["attempt"],
            "request_id": record["request_id"],
            "terminal_state": record["terminal_state"],
            "http_status": (record["rollout_provenance"].get("response") or {}).get(
                "http_status"),
            "raw_sse_sha256": (record.get("sse_replay") or {}).get("raw_sha256"),
        } for record in attempts]
        require(len(attempts) >= TOTAL_ROLLOUTS, "scheduled attempts were not retained")
        require(all(item["http_status"] == 200 and item["terminal_state"] == "completed"
                    for item in summary["attempts"][-TOTAL_ROLLOUTS:]),
                "accepted raw attempts are incomplete")
        summary["health_before_shutdown"] = rollout_live.health(base_url)
        require(summary["health_before_shutdown"].get("in_flight") == 0,
                "isolated public server still has in-flight work")
    finally:
        behavior_tools.stop_processes(processes)
    summary["serving_cleanup"] = "gateway, Qwen stage, and CPU fixtures exited before replay/training"
    summary["gpu7_after_serving"] = rollout_live.gpu7_snapshot()
    rollout_live.require_gpu7_idle(summary["gpu7_after_serving"])

    schedule, schedule_sha = online.build_schedule(plan)
    raw_records = read_jsonl(out_dir / "raw-accepted/trajectories.jsonl")
    require(len(raw_records) == len(schedule) == TOTAL_ROLLOUTS,
            "fresh raw schedule is not two complete groups of three")
    for item, record in zip(schedule, raw_records):
        online.validate_online_record(
            record, item, plan, schedule_sha, out_dir / "raw-accepted/artifacts")
        require(record["rollout_provenance"]["behavior"]["status"] == "COVERED",
                "fresh accepted rollout lacks complete sampled behavior")
    rows = real_model.accepted_rows(out_dir / "raw-accepted")
    require(len(rows) == 2 and all(row["has_reward_variance"] for row in rows)
            and all(row["rollout_provenance"]["group_index"] >= 0 for row in rows),
            "fresh groups have zero reward variance; optimizer is forbidden")

    # Independent native replay supplies the exact native-rendered prompt ids and
    # recomputes sampled ids/masks from raw FP32 rows without producer scoring.
    args.source_rollout_bundle = out_dir / "raw-accepted"
    forwards = {}
    forward_identities = []
    for item, record in zip(schedule, raw_records):
        document, identity = real_model.independent_forward(
            args, item, record, "behavior", out_dir)
        independent = document["independent_behavior"]["logprobs"]
        sampled = record["rollout_provenance"]["behavior"]["logprobs"]
        differences = [abs(float(left) - float(right))
                       for left, right in zip(independent, sampled)]
        require(differences and len(independent) == len(sampled)
                and max(differences) <= 1e-9,
                "independent native/sampled behavior logprobs differ")
        identity["max_abs_sampled_logprob_delta"] = max(differences)
        forwards[item["rollout_id"]] = document
        forward_identities.append(identity)
    summary["independent_native_forwards"] = forward_identities
    prepared = real_model.attach_exports_and_preflight(
        args, out_dir, plan, schedule, schedule_sha, raw_records, forwards,
        {"model": provenance})
    rows = real_model.accepted_rows(prepared["accepted"])
    require(len(rows) == 2 and all(row["split"] == "train" and row["has_reward_variance"]
                                  for row in rows),
            "strong exported optimizer groups are not train-only nonzero variance")
    records = read_jsonl(prepared["accepted"] / "trajectories.jsonl")
    require(len(records) == TOTAL_ROLLOUTS
            and all(record["token_export"]["format_version"] == 2 for record in records),
            "fresh strong token exports are incomplete")
    summary["groups"] = [{
        "schedule_group_id": row["rollout_provenance"]["schedule_group_id"],
        "dataset_group_id": row["group_id"], "split": row["split"],
        "rewards": [rollout["score"] for rollout in row["rollouts"]],
        "advantages": [rollout["advantage"] for rollout in row["rollouts"]],
        "group_mean": row["group_mean"], "group_std": row["group_std"],
        "has_reward_variance": row["has_reward_variance"],
        "rollout_ids": [rollout["rollout_provenance"]["rollout_id"]
                        for rollout in row["rollouts"]],
    } for row in rows]
    summary["token_counts"] = {
        "sampled": sum(len(record["rollout_provenance"]["behavior"]["token_ids"])
                       for record in records),
        "supervised": sum(sum(record["rollout_provenance"]["behavior"]["mask"])
                          for record in records),
        "directive_zero": sum(len(record["rollout_provenance"]["behavior"]["mask"])
                              - sum(record["rollout_provenance"]["behavior"]["mask"])
                              for record in records),
    }
    summary["preparation"] = {
        "preflight_run_id": prepared["preflight_run_id"],
        "independent_audit_sha256": prepared["independent_audit_sha256"],
        "strong_tokenizer_reencode": prepared["strong_tokenizer_reencode"],
        "accepted_files": prepared["accepted_files"],
        "accepted_trajectory_sha256": sha256_file(prepared["accepted"] / "trajectories.jsonl"),
        "accepted_pack_manifest_sha256": read_json(
            prepared["accepted"] / "rollout_manifest.json")["pack_manifest"]["sha256"],
        "preflight_manifest_sha256": sha256_file(prepared["preflight"] / "manifest.json"),
        "raw_sse_cas_records": TOTAL_ROLLOUTS,
    }
    summary["status"] = "PREPARED"
    summary["claims"].update({"optimizer": "NOT_RUN", "optimizer_resume_update": "NOT_RUN"})
    return prepared


def run_trainer(args, out_dir, name, *, stop_after=2, resume_from=None,
                expected_failure=None):
    output = out_dir / "optimizer" / name
    log_path = out_dir / "logs" / ("optimizer-" + name + ".log")
    command = [
        str(args.trainer_python), "-I", str(ROOT / "python/umminfer_online_grpo_optimizer.py"),
        "--online-pack", str(out_dir / "accepted"),
        "--preflight-bundle", str(out_dir / "preflight"),
        "--model-dir", str(args.model_dir), "--model-id", args.model,
        "--model-revision", args.model_revision, "--source-commit", args.source_commit,
        "--out", str(output), "--stop-after", str(stop_after),
    ]
    if resume_from is not None:
        command.extend(["--resume-from", str(resume_from)])
    env = os.environ.copy()
    env.update({"UMMINFER_LIVE_OPT_IN": "1", "CUDA_VISIBLE_DEVICES": "7",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"})
    before = rollout_live.gpu7_snapshot()
    rollout_live.require_gpu7_idle(before)
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT)
        identity = real_model.process_identity(process, args.trainer_python)
        try:
            returncode = process.wait(timeout=args.trainer_timeout)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            raise
    after = rollout_live.gpu7_snapshot()
    rollout_live.require_gpu7_idle(after)
    diagnostic = log_path.read_text(encoding="utf-8", errors="replace")
    identity.update({"name": name, "command": command, "returncode": returncode,
                     "result_sha256": None,
                     "log_sha256": sha256_file(log_path),
                     "gpu7_before": before, "gpu7_after": after})
    if expected_failure is not None:
        require(returncode == 2 and expected_failure in diagnostic
                and not (output / "update-000002.json").exists(),
                f"tampered resume was not rejected before update 2: {diagnostic}")
        return output, None, identity
    require(returncode == 0, f"optimizer process {name} failed; see retained log")
    result = read_json(output / "result.json")
    require(result["status"] == "PASS", f"optimizer process {name} did not pass")
    identity["result_sha256"] = sha256_file(output / "result.json")
    return output, result, identity


def run_tamper_controls(args, out_dir):
    """Consistently rehash modified primary metadata; rejection must be semantic."""
    cases = ("objective-value", "gradient", "parameter-delta", "resumed-checkpoint")
    results = []
    import tempfile
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="umm-online-optimizer-real-tamper-") as tmp:
            copied = Path(tmp) / "evidence"
            shutil.copytree(out_dir, copied)
            update_path = copied / "optimizer/uninterrupted/update-000001.json"
            update = read_json(update_path)
            result_path = copied / "optimizer/uninterrupted/result.json"
            result = read_json(result_path)
            if case == "objective-value":
                update["members"][0]["current_logprobs"][0] += 0.01
            elif case in ("gradient", "parameter-delta"):
                document = update["gradient"] if case == "gradient" \
                    else update["parameters"]["delta"]
                tensor_path = copied / "optimizer/uninterrupted/tensors" / document["path"]
                payload = bytearray(tensor_path.read_bytes())
                # Change the last FP32 value from zero to 1.0 while preserving length.
                payload[-4:] = bytes.fromhex("0000803f")
                tensor_path.write_bytes(payload)
                document["sha256"] = hashlib.sha256(payload).hexdigest()
                write_json(update_path, update)
                retained = result["updates"][0]
                retained["sha256"] = sha256_file(update_path)
                retained["bytes"] = update_path.stat().st_size
                write_json(result_path, result)
            elif case == "resumed-checkpoint":
                checkpoint = copied / "optimizer/resumed/checkpoints/step-000002.pt"
                payload = bytearray(checkpoint.read_bytes())
                payload[len(payload) // 2] ^= 1
                checkpoint.write_bytes(payload)
                digest = hashlib.sha256(payload).hexdigest()
                Path(str(checkpoint) + ".sha256").write_text(digest + "\n", encoding="ascii")
                result_path = copied / "optimizer/resumed/result.json"
                result = read_json(result_path)
                result["checkpoints"][0]["sha256"] = digest
                result["checkpoints"][0]["bytes"] = checkpoint.stat().st_size
                write_json(result_path, result)
            if case == "objective-value":
                write_json(update_path, update)
                retained = result["updates"][0]
                retained["sha256"] = sha256_file(update_path)
                retained["bytes"] = update_path.stat().st_size
                write_json(result_path, result)
            # Keep every affected checksum internally consistent; only the
            # intended semantic/content comparison may reject the tamper.
            write_checksums(copied / "optimizer" /
                            ("resumed" if case == "resumed-checkpoint" else "uninterrupted"))
            write_checksums(copied)
            report = copied / "tampered-verification.json"
            command = [
                sys.executable, "-I", str(ROOT / "tests/online_grpo_optimizer_verify.py"),
                str(copied), "--model-dir", str(args.model_dir),
                "--replay-binary", str(args.replay_binary), "--repo-root", str(ROOT),
                "--report", str(report),
            ]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True,
                                       text=True, timeout=3600, check=False)
            log = out_dir / "tamper-logs" / (case + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
            expected_diagnostic = {
                "objective-value": "objective metric",
                "gradient": "postclip gradient",
                "parameter-delta": "raw parameter delta bytes mismatch",
                "resumed-checkpoint": "cross-run byte mismatch",
            }[case]
            diagnostic = completed.stdout + completed.stderr
            diagnostic_matched = expected_diagnostic in diagnostic
            if case == "gradient":
                diagnostic_matched = diagnostic_matched or "AdamW first moment" in diagnostic
            require(completed.returncode == 2 and not report.exists()
                    and diagnostic_matched,
                    f"real tamper control {case} missed its semantic rejection: {diagnostic}")
            results.append({"case": case, "returncode": completed.returncode,
                            "rejected": True, "report_emitted": False,
                            "diagnostic_sha256": hashlib.sha256(
                                (completed.stdout + completed.stderr).encode("utf-8")).hexdigest(),
                            "log_sha256": sha256_file(log)})
    # A structurally valid, consistently rehashed checkpoint-controlled AdamW
    # setting must be rejected by a fresh trainer before a second update.
    tamper_input = out_dir / "tamper-input/split-first"
    shutil.copytree(out_dir / "optimizer/split-first", tamper_input)
    checkpoint = tamper_input / "checkpoints/step-000001.pt"
    rewrite = subprocess.run([
        str(args.trainer_python), "-I", "-c",
        "import sys,torch; p=sys.argv[1]; d=torch.load(p,map_location='cpu',weights_only=True); "
        "d['optimizer_state']['param_groups'][0]['lr'] += 0.125; torch.save(d,p)",
        str(checkpoint),
    ], cwd=ROOT, capture_output=True, text=True, timeout=120, check=False)
    require(rewrite.returncode == 0,
            f"could not construct checkpoint resume control: {rewrite.stdout}{rewrite.stderr}")
    Path(str(checkpoint) + ".sha256").write_text(
        sha256_file(checkpoint) + "\n", encoding="ascii")
    _output, _result, identity = run_trainer(
        args, out_dir, "rejected-resume-tamper", stop_after=2, resume_from=checkpoint,
        expected_failure="param-group/hyperparameter mismatch")
    results.append({"case": "fresh-resume-adamw-learning-rate", "returncode": 2,
                    "rejected_before_update_2": True, "report_emitted": False,
                    "process": identity})
    write_json(out_dir / "tamper-controls.json", {
        "schema": "umminfer-online-grpo-optimizer-tamper-v1", "status": "PASS",
        "cases": results, "all_consistently_rehashed_primary_records_rejected": True,
    })
    return results


def execute_optimizer_and_verify(args, out_dir, summary):
    runs = {}
    identities = []
    for name, stop_after, resume in (
        ("uninterrupted", 2, None), ("split-first", 1, None),
        ("resumed", 2, out_dir / "optimizer/split-first/checkpoints/step-000001.pt"),
    ):
        output, result, identity = run_trainer(
            args, out_dir, name, stop_after=stop_after, resume_from=resume)
        runs[name] = output
        identities.append(identity)
        summary["status"] = "OPTIMIZER_" + name.upper() + "_PASS"
        write_json(out_dir / "summary.json", summary)
    summary["optimizer_processes"] = identities
    comparisons = []
    for step in (1, 2):
        left = runs["uninterrupted"]
        right = runs["split-first"] if step == 1 else runs["resumed"]
        for relative in (f"update-{step:06d}.json", "reference.json",
                         f"checkpoints/step-{step:06d}.pt",
                         f"checkpoints/step-{step:06d}.pt.sha256"):
            require((left / relative).read_bytes() == (right / relative).read_bytes(),
                    f"deterministic/resume byte mismatch at step {step}: {relative}")
            comparisons.append({"step": step, "path": relative, "byte_identical": True,
                                "sha256": sha256_file(left / relative)})
        left_tensors = {path.name: path for path in (left / "tensors").glob(
            f"step-{step:06d}-*.f32")}
        require(left_tensors, "missing raw optimizer tensors")
        for name, path in sorted(left_tensors.items()):
            require(path.read_bytes() == (right / "tensors" / name).read_bytes(),
                    f"raw tensor mismatch at step {step}: {name}")
    summary["deterministic_resume"] = {
        "uninterrupted_vs_split_first_step_1": "BYTE_IDENTICAL",
        "uninterrupted_vs_resumed_step_2": "BYTE_IDENTICAL",
        "raw_tensor_files_byte_identical": True, "comparisons": comparisons,
    }
    summary["status"] = "PASS"
    summary["claims"].update({
        "optimizer": "EXECUTED_BOUNDED_LORA", "REAL_TRAINING": "EXECUTED_BOUNDED_LORA",
        "optimizer_resume_update": "BYTE_IDENTICAL", "model_weights_updated": "LORA_ONLY",
        "model_quality": "NOT_RUN", "generalized_improvement": "NOT_RUN",
    })
    write_json(out_dir / "summary.json", summary)
    verify_command = [
        sys.executable, "-I", str(ROOT / "tests/online_grpo_optimizer_verify.py"),
        str(out_dir), "--model-dir", str(args.model_dir),
        "--replay-binary", str(args.replay_binary), "--repo-root", str(ROOT),
        "--report", str(out_dir / "independent-optimizer-verification.json"),
    ]
    real_model.run_command(verify_command, out_dir / "logs/independent-optimizer-verify.log",
                           timeout=3600)
    verification = read_json(out_dir / "independent-optimizer-verification.json")
    require(verification["status"] == "PASS", "independent optimizer verification failed")
    summary["independent_verification"] = {
        "sha256": sha256_file(out_dir / "independent-optimizer-verification.json"),
        "status": verification["status"],
        "objective_max_abs_delta": verification["objective_max_abs_delta"],
        "gradient_and_delta_checks": verification["gradient_and_delta_checks"],
    }
    summary["tamper_controls"] = run_tamper_controls(args, out_dir)
    write_json(out_dir / "summary.json", summary)
    final_report = out_dir / "independent-optimizer-verification-final.json"
    final_command = list(verify_command)
    final_command[-1] = str(final_report)
    real_model.run_command(final_command, out_dir / "logs/independent-optimizer-final.log",
                           timeout=3600)
    require(final_report.read_bytes()
            == (out_dir / "independent-optimizer-verification.json").read_bytes(),
            "post-tamper fresh-process verification changed the independent report")
    return summary


def write_checksums(out_dir):
    files = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            files.append(f"{sha256_file(path)}  {path.relative_to(out_dir)}")
    (out_dir / "SHA256SUMS").write_text("\n".join(files) + "\n", encoding="ascii")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3-native")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-revision", default="UNKNOWN")
    parser.add_argument("--server-binary", type=Path, required=True)
    parser.add_argument("--stage-binary", type=Path, required=True)
    parser.add_argument("--replay-binary", type=Path, required=True)
    parser.add_argument("--trainer-python", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--trainer-timeout", type=float, default=7200.0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--continue-optimizer", action="store_true")
    args = parser.parse_args(argv)
    if os.environ.get("UMMINFER_LIVE_OPT_IN") != "1":
        print("REAL ONLINE GRPO OPTIMIZER GATE: NOT_RUN (set UMMINFER_LIVE_OPT_IN=1)")
        return 0
    require(args.out_dir.is_absolute(), "--out-dir must be absolute")
    summary = {
        "schema": "umminfer-online-grpo-optimizer-gate-v1", "status": "RUNNING",
        "source_commit": args.source_commit, "batch_id": args.batch_id,
        "plan": {"groups": 2, "seeds_per_group": 3, "temperature": 1.4, "top_p": 0.98,
                 "text_only": True, "reward": "verifiable-integer-arithmetic-v1"},
        "claims": {"optimizer": "NOT_RUN", "REAL_TRAINING": "NOT_RUN",
                   "model_quality": "NOT_RUN", "generalized_improvement": "NOT_RUN",
                   "gpu": 7, "resident_service": "untouched",
                   "visual_artifacts": "NOT_USED_TEXT_ONLY"},
    }
    owns_out_dir = False
    try:
        if args.continue_optimizer:
            require(not args.prepare_only,
                    "--continue-optimizer cannot be combined with --prepare-only")
            require(args.out_dir.is_dir(), "continued evidence directory is absent")
            previous = read_json(args.out_dir / "summary.json")
            require(previous["status"] == "PREPARED"
                    and previous["source_commit"] == args.source_commit
                    and previous["batch_id"] == args.batch_id
                    and not (args.out_dir / "optimizer").exists(),
                    "continuation requires an unoptimized PREPARED evidence directory")
            summary = previous
            owns_out_dir = True
            summary["status"] = "OPTIMIZER_CONTINUING"
        else:
            require(not args.out_dir.exists(), "--out-dir must be absent")
            args.out_dir.mkdir(parents=True)
            owns_out_dir = True
            (args.out_dir / "logs").mkdir()
            for path in (args.model_dir, args.server_binary, args.stage_binary,
                         args.replay_binary, args.trainer_python):
                require(path.is_absolute() and path.exists(), f"required path absent: {path}")
            generate_and_prepare(args, args.out_dir, summary)
        write_json(args.out_dir / "summary.json", summary)
        if args.prepare_only:
            print("REAL ONLINE GRPO OPTIMIZER GATE: PREPARED optimizer=NOT_RUN")
            return 0
        execute_optimizer_and_verify(args, args.out_dir, summary)
        summary["gpu7_final"] = rollout_live.gpu7_snapshot()
        rollout_live.require_gpu7_idle(summary["gpu7_final"])
    except Exception as error:
        summary["status"] = "FAIL"
        summary["error"] = str(error)
        raise
    finally:
        if owns_out_dir:
            write_json(args.out_dir / "summary.json", summary)
            write_checksums(args.out_dir)
    print(f"REAL ONLINE GRPO OPTIMIZER GATE: {summary['status']} groups=2 rollouts=6")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
