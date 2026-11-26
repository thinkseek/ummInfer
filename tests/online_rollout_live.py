#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isolated real online-rollout generation gate on physical GPU 7 only.

Two frozen groups of three fresh scheduled rollouts run through the actual
trainer-side client and public native server. Negative coverage cases never
enter accepted groups. Fresh native processes independently recompute sampled
ids, full-row logprobs, and clean-target masks. No optimizer or quality claim.
"""

import argparse
import copy
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import behavior_logprobs_live as behavior_tools  # noqa: E402
import sse_replay_live as replay_tools  # noqa: E402
import umminfer_online_rollout as online  # noqa: E402
import umminfer_rl_trajectory as traj  # noqa: E402
from umminfer_sse_replay import RawSSECapture, validate_replay_metadata  # noqa: E402

ABS_TOLERANCE = 1e-6


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def canonical(value):
    return traj.canonical_json(value)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return replay_tools.sha256_file(path)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(
        encoding="utf-8").splitlines() if line]


def frozen_plan(identity, source_commit, batch_id):
    """The frozen 2x3 plan; only the unique batch namespace changes per run."""
    visual_prompt = (
        "Write one short sentence about a red lantern, then one standalone line "
        "containing exactly [[image: a red lantern glowing beside misty water]], "
        "then one short closing sentence.")
    text_prompt = "用一句中文解释红灯笼，并自然包含 🌅。不要输出图像或视频标记。"
    plan = {
        "schema": online.PLAN_SCHEMA,
        "batch_id": batch_id,
        "behavior_policy": {
            "policy_id": "qwen38-native-online-rollout-v1",
            "revision": source_commit,
            "model_id": identity["model_id"],
            "model_revision": identity["model_revision"],
            "model_sha256": identity["model_sha256"],
            "model_config_sha256": identity["config"]["sha256"],
        },
        "sampling": {"max_tokens": 96, "temperature": 0.0, "top_p": 1.0},
        "request_template": {
            "behavior": {"logprobs": True},
            "modalities": ["text", "image"],
            "image": {"size": "256x256", "steps": 1, "seed": 42,
                      "output_format": "png"},
            "narrative": {"enabled": True, "visual_mode": "auto", "max_slots": 1,
                          "allow_text_only": True},
        },
        "reward": {"spec": "umminfer_ablation_metrics:NarrativeRoutingReward"},
        "split_seed": "online-rollout-live-49-split-v1",
        "eval_ratio": 0.0,
        "transport": {
            "timeout_s": 600.0, "max_attempts": 5, "retry_statuses": [503],
            "retry_transport": False, "retry_timeouts": False,
        },
        "groups": [
            {"prompt_id": "asr-free-text", "messages": [
                {"role": "user", "content": text_prompt}],
             "expected_visual_modalities": [], "seeds": [101, 102, 103]},
            {"prompt_id": "interleaved-image", "messages": [
                {"role": "user", "content": visual_prompt}],
             "expected_visual_modalities": ["image"], "seeds": [-201, -202, -203]},
        ],
    }
    return online.prepare_plan(plan)


def gpu7_snapshot():
    result = subprocess.run([
        "nvidia-smi", "--id=7", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, timeout=20, check=True)
    line = result.stdout.strip()
    fields = [int(value.strip()) for value in line.split(",")]
    processes = subprocess.run([
        "nvidia-smi", "--id=7", "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, timeout=20, check=True)
    return {"index": fields[0], "memory_used_mib": fields[1],
            "memory_total_mib": fields[2], "utilization_percent": fields[3],
            "compute_processes": processes.stdout.strip()}


def require_gpu7_idle(snapshot):
    require(snapshot["index"] == 7 and snapshot["memory_used_mib"] <= 16,
            f"physical GPU 7 is not explicitly idle: {snapshot}")
    require(not snapshot["compute_processes"],
            f"physical GPU 7 has a compute process: {snapshot}")


def health(base_url):
    conn = http.client.HTTPConnection(base_url, timeout=5)
    try:
        conn.request("GET", "/health")
        response = conn.getresponse()
        payload = response.read()
        require(response.status == 200, f"/health returned {response.status}: {payload!r}")
        return json.loads(payload)
    finally:
        conn.close()


def start_stack(args, out_dir, plan_path, server_config_path, identity_path):
    port = behavior_tools.free_loopback_port()
    base_url = f"127.0.0.1:{port}"
    ipc = out_dir / "run/ipc"
    ipc.mkdir(parents=True)
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    launches = []

    def launch(name, command, *, cpu=False):
        env = os.environ.copy()
        env["UMM_QWEN38_PREFIX_CACHE"] = "0"
        if cpu:
            env["CUDA_VISIBLE_DEVICES"] = ""
        else:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        log_path = logs / (name + ".log")
        log = log_path.open("wb")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT)
        processes.append(process)
        launches.append({"name": name, "pid": process.pid, "command": command,
                         "log": str(log_path.relative_to(out_dir))})
        return process

    try:
        qwen = launch("qwen38", [
            str(args.stage_binary), "--kind", "qwen38", "--name", "vlm",
            "--model", str(args.model_dir), "--gpu", "7", "--batch-size", "1",
            "--max-new-tokens", "512", "--model-id", args.model,
            "--model-revision", args.model_revision,
            "--identity-out", str(identity_path),
            "--recv-endpoint", f"ipc://{ipc}/vlm",
            "--coordinator-endpoint", f"ipc://{ipc}/completion",
            "--abort-endpoint", f"ipc://{ipc}/abort",
        ])
        deadline = time.monotonic() + args.timeout
        while not identity_path.is_file():
            require(qwen.poll() is None, "Qwen stage exited before writing loaded identity")
            require(time.monotonic() < deadline, "native identity emission timed out")
            time.sleep(0.2)
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        plan = frozen_plan(identity, args.source_commit, args.batch_id)
        (out_dir / "config").mkdir(exist_ok=True)
        plan_path.write_bytes(online._canonical_line(plan))
        require(plan_path.read_bytes() == online._canonical_line(plan),
                "trusted plan bytes are not canonical UTF-8 JSON plus newline")
        identity_sha = sha256_file(identity_path)["sha256"]
        server_config = {
            "schema": "umminfer-online-rollout-server-v2",
            "plan_path": str(plan_path),
            "plan_sha256": sha256_file(plan_path)["sha256"],
            "native_identity_path": str(identity_path),
            "native_identity_sha256": identity_sha,
            "native_stage_name": "vlm",
            "max_identity_age_ms": max(1, int(args.timeout * 1000)),
        }
        write_json(server_config_path, server_config)
        launch("gateway", [
            str(args.server_binary), "serve", "--host", "127.0.0.1",
            "--port", str(port), "--model", args.model,
            "--entry", "legacy", "--entry-endpoint", f"ipc://{ipc}/legacy",
            "--vlm-entry", "vlm", "--vlm-entry-endpoint", f"ipc://{ipc}/vlm",
            "--image-entry", "image", "--image-entry-endpoint", f"ipc://{ipc}/image",
            "--completion", f"ipc://{ipc}/completion",
            "--abort", f"ipc://{ipc}/abort", "--max-in-flight", "1",
            "--online-rollout-config", str(server_config_path),
        ])
        for name in ("legacy", "image"):
            launch(name + "-fixture", [
                sys.executable, "-I", str(ROOT / "python/umminfer_flux2_stage.py"),
                "--name", name, "--backend", "mock",
                "--model-id", "contract-fixture-image-not-model-output",
                "--recv-endpoint", f"ipc://{ipc}/{name}",
                "--coordinator-endpoint", f"ipc://{ipc}/completion",
                "--abort-endpoint", f"ipc://{ipc}/abort",
            ], cpu=True)
        behavior_tools.wait_http(base_url, "/ready", 200, processes,
                                 timeout=args.timeout)
        return base_url, processes, launches, plan
    except Exception:
        behavior_tools.stop_processes(processes)
        raise


def capture_unmarked(base_url, case, body, recorder, response_dir, *,
                     cancel_gateway_log=None):
    observed_response = []
    observed_transport = []
    holder = {}

    def run():
        try:
            holder["record"] = traj.record_chat_completion(
                base_url, copy.deepcopy(body), recorder, body["session_id"],
                timeout=600.0, capture_replay=True,
                response_observer=observed_response.append,
                transport_observer=observed_transport.append,
                turn_index=None)
        except BaseException as error:
            holder["error"] = error

    if cancel_gateway_log is None:
        run()
    else:
        initial_size = cancel_gateway_log.stat().st_size
        thread = threading.Thread(target=run)
        thread.start()
        request_id = None
        deadline = time.monotonic() + 120
        while request_id is None and thread.is_alive() and time.monotonic() < deadline:
            with cancel_gateway_log.open("rb") as log:
                log.seek(initial_size)
                text = log.read().decode("utf-8", errors="replace")
            marker = "behavior stream begin request_id="
            if marker in text:
                request_id = text.rsplit(marker, 1)[1].splitlines()[0].strip()
                break
            time.sleep(0.02)
        require(request_id, "could not bind the in-flight behavior cancellation")
        status, abort_body = traj.abort_request(base_url, request_id)
        require(status == 200 and abort_body.get("success") is True,
                f"public abort failed: {status} {abort_body}")
        thread.join(timeout=600)
        require(not thread.is_alive(), "cancelled behavior request did not settle")
        holder["abort"] = {"request_id": request_id, "status": status,
                           "response": abort_body}
    if "error" in holder:
        raise holder["error"]
    record = holder["record"]
    response = observed_response[0] if observed_response else None
    if response is not None:
        write_json(response_dir / (case + ".json"), response)
    return {"case": case, "record": record, "response": response,
            "transport": observed_transport, "abort": holder.get("abort")}


def run_accepted_with_real_retry(args, base_url, plan, out_dir, blocker):
    accepted = out_dir / "accepted"
    command = [
        sys.executable, "-I", str(ROOT / "python/umminfer_online_rollout.py"),
        "run", "--plan", str(out_dir / "config/plan.json"),
        "--endpoint", base_url, "--out", str(accepted),
    ]
    log = (out_dir / "logs/online-client.log").open("wb")
    process = subprocess.Popen(command, cwd=ROOT, stdout=log,
                               stderr=subprocess.STDOUT)
    attempts_path = accepted / "attempts.jsonl"
    overload_seen = False
    deadline = time.monotonic() + args.timeout
    try:
        while process.poll() is None and time.monotonic() < deadline:
            if attempts_path.is_file():
                try:
                    attempts = read_jsonl(attempts_path)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    attempts = []
                for attempt in attempts:
                    response = (attempt.get("rollout_provenance") or {}).get("response")
                    if response and response.get("http_status") == 503:
                        overload_seen = True
                if overload_seen and blocker["request_id"] is not None \
                        and not blocker["abort_sent"]:
                    status, abort_body = traj.abort_request(
                        base_url, blocker["request_id"])
                    require(status == 200, f"blocker abort returned {status}: {abort_body}")
                    blocker["abort_sent"] = True
                    blocker["abort"] = {"status": status, "response": abort_body}
            if blocker["abort_sent"] and process.poll() is not None:
                break
            time.sleep(0.001)
        require(process.poll() is not None, "actual online client timed out")
        blocker["thread"].join(timeout=120)
        require(not blocker["thread"].is_alive(), "retry blocker did not settle")
        log.close()
        stdout = (out_dir / "logs/online-client.log").read_text(
            encoding="utf-8", errors="replace")
        require(process.returncode == 0,
                f"actual online client failed ({process.returncode}): {stdout}")
        require(overload_seen and blocker["abort_sent"],
                "bounded retry did not observe and recover from a real 503")
        result = json.loads((accepted / "rollout_manifest.json").read_text(
            encoding="utf-8"))
        attempts = read_jsonl(attempts_path)
        statuses = [(record.get("rollout_provenance") or {}).get(
            "response", {}).get("http_status") for record in attempts]
        require(503 in statuses and statuses[-1] == 200,
                f"failed attempts were not preserved ahead of success: {statuses}")
        return {"command": command, "returncode": process.returncode,
                "stdout": stdout.strip(), "manifest": result,
                "attempt_http_statuses": statuses,
                "attempt_count": len(attempts)}
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        if not log.closed:
            log.close()


def check_accepted(plan, accepted_dir):
    schedule, schedule_sha = online.build_schedule(plan)
    records = read_jsonl(accepted_dir / "trajectories.jsonl")
    require(len(records) == len(schedule) == 6, "accepted schedule is not 2x3")
    by_rollout = {record["rollout_provenance"]["rollout_id"]: record
                  for record in records}
    comparisons = []
    for item in schedule:
        record = by_rollout[item["rollout_id"]]
        online.validate_online_record(
            record, item, plan, schedule_sha, accepted_dir / "artifacts")
        behavior = record["rollout_provenance"]["behavior"]
        require(record["rollout_provenance"]["response"]["behavior_evidence"] ==
                "exposed" and behavior["status"] == "COVERED",
                f"accepted {item['rollout_id']} was not exposed")
        require(record["terminal_state"] == "completed", "incomplete accepted rollout")
        visual_events = [event for event in record["events"]
                         if event.get("type") == "visual_slot"]
        artifacts = record.get("artifacts") or []
        if item["prompt_id"] == "asr-free-text":
            require(not visual_events and not artifacts,
                    "ASR-free text group gained a visual response")
            require(all(bit == 1 for bit in behavior["mask"]),
                    "plain-text behavior mask excluded tokens")
        else:
            require(len(visual_events) == 1 and len(artifacts) == 1,
                    "interleaved group lacks exactly one supported image response")
            require(any(bit == 0 for bit in behavior["mask"]),
                    "visual narrative directive was not excluded from its mask")
        comparisons.append({"rollout_id": item["rollout_id"], "item": item,
                            "record": record})
    return comparisons


def run_negative_cases(base_url, comparisons, out_dir, gateway_log):
    recorder = traj.TrajectoryRecorder(out_dir / "negative/trajectories.jsonl")
    response_dir = out_dir / "negative/responses"
    first_text = comparisons[0]["item"]["body"]
    first_visual = comparisons[3]["item"]["body"]
    cases = []
    default = copy.deepcopy(first_text)
    default["session_id"] = "negative-default-" + first_text["session_id"]
    del default["behavior"]
    cases.append(("default", default, None))
    truncated = copy.deepcopy(first_text)
    truncated["session_id"] = "negative-truncated-" + first_text["session_id"]
    truncated["max_tokens"] = 1
    cases.append(("truncated", truncated, None))
    cancelled = copy.deepcopy(first_text)
    cancelled["session_id"] = "negative-cancelled-" + first_text["session_id"]
    cancelled["messages"] = [{"role": "user", "content":
        "Count upward from 1, one integer per line, without stopping."}]
    cancelled["max_tokens"] = 512
    cases.append(("cancelled", cancelled, gateway_log))
    unsupported = copy.deepcopy(first_visual)
    unsupported["session_id"] = "negative-unsupported-" + first_visual["session_id"]
    unsupported["modalities"] = ["text", "video"]
    unsupported["narrative"]["visual_mode"] = "video"
    cases.append(("unsupported-video", unsupported, None))

    results = []
    for case, body, cancel_log in cases:
        captured = capture_unmarked(
            base_url, case, body, recorder, response_dir, cancel_gateway_log=cancel_log)
        record = captured["record"]
        response = captured["response"]
        headers = {key.lower(): value for key, value in
                   (response or {}).get("headers", {}).items()}
        if case == "default":
            require(record["clean_text"] == comparisons[0]["record"]["clean_text"],
                    "default protocol changed output under the same model and sampling")
            require("x-umminfer-behavior-evidence" not in headers,
                    "default-off request gained a behavior coverage header")
            require(record["terminal_state"] == "completed", "default request failed")
        elif case == "unsupported-video":
            require(response and response["status"] == 400,
                    "unsupported video response was not rejected")
            require(headers.get("x-umminfer-behavior-evidence") == "not_exposed",
                    "unsupported response did not fail closed")
            require(record["terminal_state"] != "completed",
                    "unsupported response became acceptable")
        else:
            require(headers.get("x-umminfer-behavior-evidence") == "not_exposed",
                    f"{case} did not declare not_exposed")
            chunks = []
            raw = validate_replay_metadata(
                record, artifact_dir=out_dir / "negative/artifacts", require_raw=True)
            capture = RawSSECapture.from_metadata(record["sse_replay"], raw)
            for _, payload in capture.payload_frames:
                if payload != "[DONE]":
                    chunks.append(json.loads(payload))
            require(all("behavior" not in chunk for chunk in chunks),
                    f"{case} emitted a behavior block")
            if case == "truncated":
                require(record["terminal_state"] == "completed" and
                        chunks[-1]["choices"][0]["finish_reason"] == "length",
                        "truncated negative case did not reach length")
            evidence = online._behavior_evidence(
                record, "not_exposed", out_dir / "negative/artifacts", chunks=chunks)
            require(evidence["status"] == "NOT_COVERED", f"{case} became covered")
        if case == "cancelled":
            require(record["terminal_state"] == "cancelled" and captured["abort"],
                    "cancelled negative case did not settle as cancelled")
        results.append({"case": case, "terminal_state": record["terminal_state"],
                        "http_status": response["status"] if response else None,
                        "coverage": "not_exposed", "accepted_into_group": False,
                        "raw_sha256": (record.get("sse_replay") or {}).get("raw_sha256"),
                        "abort": captured["abort"]})
    return results


def fresh_forward(args, entry, out_dir):
    item, record = entry["item"], entry["record"]
    body = item["body"]
    prompt = body["messages"][0]["content"]
    case = item["rollout_id"]
    command = [
        str(args.replay_binary), "--model", str(args.model_dir), "--prompt", prompt,
        "--gpu", "7", "--max-new-tokens", str(body["max_tokens"]),
        "--temperature", str(body["temperature"]), "--top-p", str(body["top_p"]),
        "--seed", str(body["seed"]), "--behavior-evidence", "--narrative-auto",
    ]
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["UMM_QWEN38_PREFIX_CACHE"] = "0"
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                            text=True, timeout=args.timeout, check=False)
    (out_dir / "logs" / (case + "-forward.stdout")).write_text(
        result.stdout, encoding="utf-8")
    (out_dir / "logs" / (case + "-forward.stderr")).write_text(
        result.stderr, encoding="utf-8")
    require(result.returncode == 0,
            f"fresh native forward failed for {case}: {result.stdout}\n{result.stderr}")
    replayed = json.loads(result.stdout.strip().splitlines()[-1])
    expected = record["rollout_provenance"]["behavior"]
    actual = replayed["behavior"]
    require(actual["token_ids"] == expected["token_ids"],
            f"sampled ids diverged for {case}")
    require(actual["mask"] == expected["mask"], f"mask diverged for {case}")
    require(actual["tokenizer_sha256"] == expected["tokenizer_sha256"] ==
            behavior_tools.tokenizer_digest(args.model_dir),
            f"loaded tokenizer identity diverged for {case}")
    differences = [abs(float(a) - float(b)) for a, b in
                   zip(actual["logprobs"], expected["logprobs"])]
    require(differences and len(differences) == len(expected["logprobs"]),
            f"logprob lengths diverged for {case}")
    require(all(math.isfinite(value) and value <= ABS_TOLERANCE
                for value in differences),
            f"logprob mismatch exceeded tolerance for {case}: {max(differences)}")
    require(replayed["clean_text"] == record["clean_text"],
            f"masked sampled bytes differ from public clean target for {case}")
    if item["prompt_id"] == "interleaved-image":
        require(any(bit == 0 for bit in actual["mask"]),
                "visual fresh forward lacks directive exclusion")
    return {"rollout_id": case, "command": command, "returncode": result.returncode,
            "token_count": len(actual["token_ids"]),
            "masked_token_count": sum(actual["mask"]),
            "max_abs_logprob_diff": max(differences), "abs_tolerance": ABS_TOLERANCE,
            "sampled_ids_exact": True, "mask_exact": True,
            "clean_text_exact": True,
            "evidence_sha256": sha256_bytes(canonical(actual))}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--server-binary", type=Path, required=True)
    parser.add_argument("--stage-binary", type=Path, required=True)
    parser.add_argument("--replay-binary", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args(argv)
    if os.environ.get("UMMINFER_LIVE_OPT_IN") != "1":
        print("REAL_ONLINE_ROLLOUT_GATE: NOT_RUN (set UMMINFER_LIVE_OPT_IN=1)")
        return 0
    require(args.out_dir.is_absolute(), "--out-dir must be absolute")
    require(not args.out_dir.exists() or
            (args.out_dir.is_dir() and not any(args.out_dir.iterdir())),
            "--out-dir must be absent or empty")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in (args.model_dir, args.server_binary, args.stage_binary,
                 args.replay_binary):
        require(path.is_absolute() and path.exists(), f"required path is absent: {path}")

    processes = []
    summary = {
        "status": "RUNNING", "source_commit": args.source_commit,
        "batch_id": args.batch_id, "plan": {"groups": 2, "seeds_per_group": 3},
        "claims": {"optimizer": "NOT_RUN", "REAL_TRAINING": "NOT_RUN",
                   "model_quality": "NOT_RUN", "generalized_improvement": "NOT_RUN",
                   "visual_output": "CPU contract fixture; not model generation",
                   "gpu": 7, "resident_service": "untouched"},
    }
    blocker = {"request_id": None, "abort_sent": False, "thread": None}
    try:
        summary["gpu7_before"] = gpu7_snapshot()
        require_gpu7_idle(summary["gpu7_before"])
        provenance, config_bytes = replay_tools.model_provenance(args.model_dir)
        provenance["config_sha256"] = sha256_bytes(config_bytes)
        provenance["model_sha256"] = sha256_bytes(canonical(
            provenance["available_weights"]))
        provenance["tokenizer_sha256"] = behavior_tools.tokenizer_digest(args.model_dir)
        provenance["source_files"] = {
            name: sha256_file(ROOT / name) for name in (
                "python/umminfer_online_rollout.py", "src/serve/online_attestation.cpp",
                "src/serve/openai_chat.cpp", "src/pipeline/pipeline.cpp",
                "src/proto/messages.cpp", "native/stage_cli/main.cpp",
                "native/stage_cli/native_identity.cpp",
                "native/stage_cli/qwen38_stage.cpp", "native/qwen38/engine.cu",
                "native/qwen38/sampling.hpp", "native/qwen38_cli/main.cpp",
                "tests/online_rollout_live.py",
            )
        }
        provenance["binaries"] = {
            "gateway": sha256_file(args.server_binary),
            "qwen_stage": sha256_file(args.stage_binary),
            "native_replay": sha256_file(args.replay_binary),
        }
        summary["provenance"] = provenance
        config_dir = args.out_dir / "config"
        config_dir.mkdir()
        base_url, processes, launches, plan = start_stack(
            args, args.out_dir, config_dir / "plan.json",
            config_dir / "server.json", config_dir / "native-identity.json")
        summary["stack_commands"] = launches
        summary["native_identity"] = json.loads(
            (config_dir / "native-identity.json").read_text(encoding="utf-8"))
        summary["server_config"] = json.loads(
            (config_dir / "server.json").read_text(encoding="utf-8"))
        schedule, _ = online.build_schedule(plan)

        # A real single-slot blocker creates retryable HTTP 503 responses. The
        # actual trainer client, not a wrapper, chooses and records fresh attempts.
        blocker_recorder = traj.TrajectoryRecorder(
            args.out_dir / "retry-blocker/trajectories.jsonl")
        blocker_body = copy.deepcopy(schedule[0]["body"])
        blocker_body["session_id"] = "retry-blocker-" + args.batch_id
        blocker_body["messages"] = [{"role": "user", "content":
            "Count upward from 1, one integer per line, without stopping."}]
        blocker_body["max_tokens"] = 512
        del blocker_body["behavior"]
        blocker_observed_ids = []

        def on_blocker_id(request_id):
            blocker_observed_ids.append(request_id)

        def run_blocker():
            try:
                blocker["record"] = traj.record_chat_completion(
                    base_url, blocker_body, blocker_recorder, blocker_body["session_id"],
                    timeout=600.0, capture_replay=True, on_request_id=on_blocker_id)
            except BaseException as error:
                blocker["error"] = error

        # This is the first chat request on a fresh isolated gateway. Its public
        # id is therefore http-1; the completed record must independently agree.
        # /health proves actual admission before the scheduled client starts.
        blocker["request_id"] = "http-1"
        blocker["thread"] = threading.Thread(target=run_blocker)
        blocker["thread"].start()
        deadline = time.monotonic() + 120
        while health(base_url).get("in_flight") != 1:
            require(blocker["thread"].is_alive(),
                    f"retry blocker ended before admission: {blocker.get('error')}")
            require(time.monotonic() < deadline, "retry blocker admission timed out")
            time.sleep(0.01)
        summary["accepted_run"] = run_accepted_with_real_retry(
            args, base_url, plan, args.out_dir, blocker)
        require("error" not in blocker, f"retry blocker failed: {blocker.get('error')}")
        require(blocker_observed_ids == ["http-1"] and
                blocker["record"]["request_id"] == "http-1",
                "retry blocker public request identity differs from the aborted id")
        require(blocker["record"]["terminal_state"] == "cancelled",
                "retry blocker was not cancelled after the observed 503")
        summary["retry_blocker"] = {
            "terminal_state": blocker["record"]["terminal_state"],
            "raw_sha256": blocker["record"]["sse_replay"]["raw_sha256"],
            "abort": blocker["abort"],
        }
        comparisons = check_accepted(plan, args.out_dir / "accepted")
        summary["accepted_cases"] = [{
            "rollout_id": entry["item"]["rollout_id"],
            "request_id": entry["record"]["request_id"],
            "session_id": entry["record"]["session_id"],
            "group_id": entry["item"]["group_id"],
            "prompt_id": entry["item"]["prompt_id"],
            "seed": entry["item"]["seed"],
            "attempt": entry["record"]["rollout_provenance"]["attempt"],
            "behavior_sha256": entry["record"]["rollout_provenance"]["behavior"][
                "evidence_sha256"],
            "raw_sha256": entry["record"]["sse_replay"]["raw_sha256"],
            "token_count": entry["record"]["rollout_provenance"]["behavior"]["token_count"],
            "masked_token_count": entry["record"]["rollout_provenance"]["behavior"][
                "masked_token_count"],
        } for entry in comparisons]
        summary["negative_cases"] = run_negative_cases(
            base_url, comparisons, args.out_dir, args.out_dir / "logs/gateway.log")
        summary["health_before_shutdown"] = health(base_url)
        require(summary["health_before_shutdown"].get("in_flight") == 0,
                f"in_flight did not return to zero: {summary['health_before_shutdown']}")
        behavior_tools.stop_processes(processes)
        processes = []
        summary["serving_cleanup"] = "gateway, Qwen stage, and both CPU fixtures exited"
        summary["gpu7_after_serving"] = gpu7_snapshot()
        require_gpu7_idle(summary["gpu7_after_serving"])

        summary["fresh_forward"] = [fresh_forward(args, entry, args.out_dir)
                                    for entry in comparisons]
        replay_command = [
            sys.executable, "-I", str(ROOT / "python/umminfer_online_rollout.py"),
            "replay", "--input", str(args.out_dir / "accepted"),
            "--out", str(args.out_dir / "replayed"),
        ]
        replay = subprocess.run(replay_command, cwd=ROOT, capture_output=True,
                                text=True, timeout=300, check=False)
        (args.out_dir / "logs/fresh-bundle-replay.log").write_text(
            replay.stdout + replay.stderr, encoding="utf-8")
        require(replay.returncode == 0,
                f"fresh-process bundle replay failed: {replay.stdout}{replay.stderr}")
        for relative in online._accepted_relative_files(args.out_dir / "accepted"):
            require((args.out_dir / "accepted" / relative).read_bytes() ==
                    (args.out_dir / "replayed" / relative).read_bytes(),
                    f"fresh replay differs: {relative}")
        summary["fresh_bundle_replay"] = {
            "command": replay_command, "returncode": replay.returncode,
            "byte_identical": True, "stdout": replay.stdout.strip(),
            "files": len(online._accepted_relative_files(args.out_dir / "accepted")),
        }
        summary["gpu7_final"] = gpu7_snapshot()
        require_gpu7_idle(summary["gpu7_final"])
        summary["status"] = "PASS"
    except Exception as error:
        summary["status"] = "FAIL"
        summary["error"] = str(error)
        raise
    finally:
        if processes:
            behavior_tools.stop_processes(processes)
            summary["serving_cleanup"] = "task-owned processes stopped after failure"
        if blocker["thread"] is not None and blocker["thread"].is_alive() \
                and blocker["request_id"] and not blocker["abort_sent"]:
            try:
                traj.abort_request(base_url, blocker["request_id"])
            except Exception:
                pass
            blocker["thread"].join(timeout=30)
        write_json(args.out_dir / "summary.json", summary)
        files = []
        for path in sorted(args.out_dir.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                files.append(f"{sha256_file(path)['sha256']}  {path.relative_to(args.out_dir)}")
        (args.out_dir / "SHA256SUMS").write_text("\n".join(files) + "\n",
                                                 encoding="ascii")
    print(f"REAL_ONLINE_ROLLOUT_GATE: {summary['status']} groups=2 rollouts=6")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
