#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isolated real Qwen3.8 behavior-evidence gate (physical GPU 7 only).

The harness starts and stops only its own loopback gateway, one serial native
Qwen stage, and two CPU-only contract fixtures. It then recomputes accepted
cases in fresh native processes after the serving GPU has been released. No
optimizer, model-quality, or generalized-performance claim is made.
"""

import argparse
import copy
import hashlib
import http.client
import io
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import sse_replay_live as replay_tools  # noqa: E402
import umminfer_online_rollout as online  # noqa: E402
import umminfer_rl_trajectory as traj  # noqa: E402
import umminfer_rl_worker as worker  # noqa: E402
import umminfer_stage_c_pipeline as pipeline  # noqa: E402
from umminfer_sse_replay import (  # noqa: E402
    LogicalTickClock, PerfCounterClock, RawSSECapture,
)

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


def tokenizer_digest(model_dir):
    rows = []
    for name in ("merges.txt", "tokenizer_config.json", "vocab.json"):
        raw = (Path(model_dir) / name).read_bytes()
        rows.append(f"{name}\n{len(raw)}\n{sha256_bytes(raw)}\n")
    payload = ("umminfer-qwen38-loaded-tokenizer-v1\n" + "".join(rows)).encode()
    return sha256_bytes(payload)


def request_cases(model):
    behavior = {"logprobs": True}
    text = {
        "model": model, "stream": True, "session_id": "behavior-text-42",
        "messages": [{"role": "user", "content":
                      "用一句中文解释红灯笼，并自然包含 🌅。"}],
        "max_tokens": 96, "temperature": 0.7, "top_p": 0.9, "seed": 42,
        "behavior": behavior,
    }
    visual = {
        "model": model, "stream": True, "session_id": "behavior-visual--42",
        "messages": [{"role": "user", "content":
                      "Write one short sentence about a red lantern, then one "
                      "standalone line containing exactly [[image: a red lantern "
                      "glowing beside misty water]], then one short closing sentence."}],
        "modalities": ["text", "image"],
        "image": {"size": "256x256", "steps": 1, "seed": 42,
                  "output_format": "png"},
        "narrative": {"enabled": True, "visual_mode": "image", "max_slots": 1,
                      "allow_text_only": False},
        "max_tokens": 96, "temperature": 0.0, "top_p": 1.0, "seed": -42,
        "behavior": behavior,
    }
    truncated = copy.deepcopy(text)
    truncated.update({
        "session_id": "behavior-truncated-7", "seed": 7, "max_tokens": 1,
        "messages": [{"role": "user", "content":
                      "Count from one to one hundred, separated by spaces."}],
    })
    cancel = copy.deepcopy(text)
    cancel.update({
        "session_id": "behavior-cancel-9", "seed": 9, "max_tokens": 512,
        "messages": [{"role": "user", "content":
                      "Count upward from 1, one integer per line, without stopping."}],
    })
    default = copy.deepcopy(text)
    default["session_id"] = "behavior-default-42"
    del default["behavior"]
    return (("text", text), ("visual", visual), ("truncated", truncated),
            ("cancel", cancel), ("default", default))


def attestation_for(body, case, provenance):
    plan = {
        "batch_id": "behavior-live-42",
        "behavior_policy": {
            "policy_id": "qwen38-native-behavior-v1", "revision": "live-source-v1",
            "model_id": body["model"], "model_revision": provenance["revision"],
            "model_sha256": provenance["model_sha256"],
            "model_config_sha256": provenance["config_sha256"],
        },
        "sampling": {key: body[key] for key in ("max_tokens", "temperature", "top_p")},
    }
    item = {
        "group_id": "behavior-group-" + case,
        "rollout_id": "behavior-rollout-" + case,
        "rollout_index": 0,
        "request_sha256": sha256_bytes(canonical(body)),
        "session_id": body["session_id"],
    }
    return plan, item, online._request_headers(item, plan, 0)


def capture_case(base_url, case, body, recorder, plan, item, headers, *,
                 cancel_request_id=None):
    params, option_inputs = traj.request_parameter_references(
        body, recorder.artifact_dir)
    inputs = traj.input_references(body["messages"], recorder.artifact_dir) \
        + option_inputs
    started_ns = time.perf_counter_ns()
    capture = RawSSECapture(clock=PerfCounterClock(origin_ns=started_ns))
    conn = http.client.HTTPConnection(base_url, timeout=300.0)
    abort_holder = {}
    try:
        request_headers = {"Content-Type": "application/json",
                           "Accept": "text/event-stream", **headers}
        conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
                     headers=request_headers)
        if cancel_request_id is not None:
            def abort_later():
                time.sleep(0.5)
                status, payload = traj.abort_request(base_url, cancel_request_id)
                abort_holder.update({"status": status, "response": payload})
            threading.Thread(target=abort_later, daemon=True).start()
        response = conn.getresponse()
        response_headers = {key.lower(): value for key, value in response.getheaders()}
        response_dir = recorder.jsonl_path.parent / "responses"
        response_dir.mkdir(parents=True, exist_ok=True)
        (response_dir / (case + ".json")).write_bytes(
            canonical({"status": response.status, "headers": response_headers}) + b"\n")
        if response.status != 200:
            error_payload = response.read()
            require(False, f"{case} returned HTTP {response.status}: {error_payload!r}")
        attestation = online._attestation_from_response(
            {"status": response.status, "headers": response_headers},
            item, plan, 0)
        coverage = response_headers["x-umminfer-behavior-evidence"]
        raw = response.read()
        if raw:
            capture.feed(raw)
        raw_dir = recorder.jsonl_path.parent / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / (case + ".sse")).write_bytes(capture.raw_bytes)
        capture.finish()
        details = {}
        request_id, events, output_text, state, error = traj.parse_sse_stream(
            io.BytesIO(capture.raw_bytes), artifact_dir=recorder.artifact_dir,
            details=details, capture=capture)
        record = recorder.record(
            session_id=body["session_id"], request_id=request_id,
            endpoint="/v1/chat/completions", inputs=inputs, params=params,
            events=events, terminal_state=state, error=error,
            output_text=output_text, artifacts=details.get("artifacts", []),
            capture=capture)
    finally:
        conn.close()
    chunks = []
    for _, payload in capture.payload_frames:
        if payload != "[DONE]":
            chunks.append(json.loads(payload))
    behavior = online._behavior_evidence(
        record, coverage, recorder.artifact_dir, chunks=chunks)
    return {
        "case": case, "record": record, "behavior": behavior,
        "coverage": coverage, "attestation": attestation,
        "abort": abort_holder, "chunks": chunks,
    }


def free_loopback_port():
    while True:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        if port != 43301:
            return port


def wait_http(base_url, path, expected, processes, timeout=300.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for process in processes:
            require(process.poll() is None,
                    f"isolated process {process.pid} exited during startup")
        try:
            conn = http.client.HTTPConnection(base_url, timeout=2.0)
            conn.request("GET", path)
            response = conn.getresponse()
            response.read()
            conn.close()
            if response.status == expected:
                return
        except OSError:
            pass
        time.sleep(0.2)
    raise AssertionError(f"{base_url}{path} did not reach HTTP {expected}")


def stop_processes(processes):
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
    for process in reversed(processes):
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def start_stack(args, out_dir):
    port = free_loopback_port()
    base_url = f"127.0.0.1:{port}"
    ipc = out_dir / "run/ipc"
    ipc.mkdir(parents=True)
    logs = out_dir / "logs"
    logs.mkdir(parents=True)
    processes = []
    launches = []

    def launch(name, command, *, cuda_visible=None):
        env = os.environ.copy()
        if cuda_visible is None:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible
        env["UMM_QWEN38_PREFIX_CACHE"] = "0"
        log = (logs / (name + ".log")).open("wb")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT)
        processes.append(process)
        launches.append({"pid": process.pid, "command": command, "log": str(log.name)})
        return launches[-1]

    try:
        launch("gateway", [
            str(args.server_binary), "serve", "--host", "127.0.0.1",
            "--port", str(port), "--model", args.model,
            "--entry", "legacy", "--entry-endpoint", f"ipc://{ipc}/legacy",
            "--vlm-entry", "vlm", "--vlm-entry-endpoint", f"ipc://{ipc}/vlm",
            "--image-entry", "image", "--image-entry-endpoint", f"ipc://{ipc}/image",
            "--completion", f"ipc://{ipc}/completion",
            "--abort", f"ipc://{ipc}/abort",
        ])
        wait_http(base_url, "/health", 200, processes, timeout=30)
        launch("qwen38", [
            str(args.stage_binary), "--kind", "qwen38", "--name", "vlm",
            "--model", str(args.model_dir), "--gpu", "7", "--batch-size", "1",
            "--max-new-tokens", "512", "--recv-endpoint", f"ipc://{ipc}/vlm",
            "--coordinator-endpoint", f"ipc://{ipc}/completion",
            "--abort-endpoint", f"ipc://{ipc}/abort",
        ])
        for name in ("legacy", "image"):
            launch(name, [
                sys.executable, "-I", str(ROOT / "python/umminfer_flux2_stage.py"),
                "--name", name, "--backend", "mock",
                "--model-id", "contract-fixture-image-not-model-output",
                "--recv-endpoint", f"ipc://{ipc}/{name}",
                "--coordinator-endpoint", f"ipc://{ipc}/completion",
                "--abort-endpoint", f"ipc://{ipc}/abort",
            ], cuda_visible="")
        wait_http(base_url, "/ready", 200, processes, timeout=args.timeout)
        return base_url, processes, launches
    except Exception:
        stop_processes(processes)
        raise


def fresh_native_replay(args, case, body, captured, out_dir):
    command = [
        str(args.replay_binary), "--model", str(args.model_dir),
        "--prompt", body["messages"][0]["content"],
        "--gpu", "7", "--max-new-tokens", str(body["max_tokens"]),
        "--temperature", str(body["temperature"]), "--top-p", str(body["top_p"]),
        "--seed", str(body["seed"]), "--behavior-evidence",
    ]
    if case == "visual":
        command.append("--narrative-image")
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["UMM_QWEN38_PREFIX_CACHE"] = "0"
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                            text=True, timeout=args.timeout, check=False)
    (out_dir / "logs" / (case + "-replay.stdout")).write_text(
        result.stdout, encoding="utf-8")
    (out_dir / "logs" / (case + "-replay.stderr")).write_text(
        result.stderr, encoding="utf-8")
    require(result.returncode == 0,
            f"{case} fresh replay failed: {result.stdout}\n{result.stderr}")
    replayed = json.loads(result.stdout.strip().splitlines()[-1])
    expected = captured["behavior"]
    actual = replayed["behavior"]
    require(set(actual) == {"format", "tokenizer_sha256", "token_ids", "logprobs", "mask"},
            f"{case} replay behavior schema differs")
    require(actual["token_ids"] == expected["token_ids"],
            f"{case} sampled ids diverged in fresh forward recomputation")
    require(actual["mask"] == expected["mask"],
            f"{case} clean-target masks diverged")
    require(actual["tokenizer_sha256"] == expected["tokenizer_sha256"] ==
            tokenizer_digest(args.model_dir),
            f"{case} exact loaded tokenizer identity differs")
    differences = [abs(float(a) - float(b)) for a, b in
                   zip(actual["logprobs"], expected["logprobs"])]
    require(len(differences) == len(expected["logprobs"]) and differences,
            f"{case} replay logprob length differs")
    require(all(math.isfinite(value) and value <= ABS_TOLERANCE
                for value in differences),
            f"{case} logprob mismatch exceeds {ABS_TOLERANCE}: {max(differences)}")
    require(replayed["clean_text"] == captured["record"]["clean_text"],
            f"{case} masked token bytes do not equal public clean target")
    require(replayed["prompt_token_ids"], f"{case} exact prompt prefix is empty")
    if case == "visual":
        require(any(bit == 0 for bit in actual["mask"]),
                "visual narrative has no masked directive tokens")
        require(replayed["raw_text"] != replayed["clean_text"],
                "visual narrative did not sample a removable directive")
        require(any(event.get("type") == "visual_slot"
                    for event in captured["record"]["events"]),
                "visual narrative has no public interleaved visual slot")
    else:
        require(replayed["raw_text"] == replayed["clean_text"],
                "text-only raw and clean sampled text differ")
    return {
        "command": command, "returncode": result.returncode,
        "token_count": len(actual["token_ids"]),
        "masked_token_count": sum(actual["mask"]),
        "max_abs_logprob_diff": max(differences),
        "abs_tolerance": ABS_TOLERANCE,
        "prompt_sha256": sha256_bytes(canonical(replayed["prompt_token_ids"])),
        "raw_text_sha256": sha256_bytes(replayed["raw_text"].encode("utf-8")),
        "clean_text_sha256": sha256_bytes(replayed["clean_text"].encode("utf-8")),
    }


def fresh_replay_records(source_jsonl, replay_jsonl, expected_records):
    replay_jsonl.parent.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-I", str(ROOT / "python/umminfer_sse_replay.py"),
               str(source_jsonl), "--out", str(replay_jsonl)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    require(result.returncode == 0 and "RAW_SSE_REPLAY: PASS" in result.stdout,
            f"fresh raw replay failed: {result.stdout}\n{result.stderr}")
    require(replay_jsonl.read_bytes() == source_jsonl.read_bytes(),
            "fresh replay JSONL is not byte-identical")
    source_tree = replay_tools.file_tree(source_jsonl.parent / "artifacts")
    replay_tree = replay_tools.file_tree(replay_jsonl.parent / "artifacts")
    require(source_tree == replay_tree, "fresh replay CAS tree differs")
    worker.validate_trajectory(str(replay_jsonl))
    _, audit = pipeline.validate_trajectory_inputs([replay_jsonl], strict_artifacts=True)
    require(audit["replay_records"] == expected_records and
            audit["verified_replay_refs"] == expected_records,
            f"not all raw sidecars verified: {audit}")
    return {
        "jsonl_sha256": sha256_file(replay_jsonl)["sha256"],
        "byte_identical": True, "verified_cas_files": len(source_tree),
        "input_audit": audit,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--server-binary", type=Path, required=True)
    parser.add_argument("--stage-binary", type=Path, required=True)
    parser.add_argument("--replay-binary", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args(argv)
    if os.environ.get("UMMINFER_LIVE_OPT_IN") != "1":
        print("REAL_BEHAVIOR_LOGPROBS_GATE: NOT_RUN (set UMMINFER_LIVE_OPT_IN=1)")
        return 0
    require(args.gpu == 7, "this gate permits only explicitly idle physical GPU 7")
    require(args.out_dir.is_absolute(), "--out-dir must be absolute")
    if args.out_dir.exists():
        require(args.out_dir.is_dir() and not any(args.out_dir.iterdir()),
                "--out-dir must be absent or empty")
    else:
        args.out_dir.mkdir(parents=True)
    (args.out_dir / "requests").mkdir()
    provenance, config_bytes = replay_tools.model_provenance(args.model_dir)
    provenance["config_sha256"] = sha256_bytes(config_bytes)
    provenance["model_sha256"] = sha256_bytes(canonical(
        provenance["available_weights"]))
    provenance["tokenizer_sha256"] = tokenizer_digest(args.model_dir)
    provenance["source_commit"] = args.source_commit
    provenance["source_files"] = {
        name: sha256_file(ROOT / name) for name in (
            "src/serve/openai_chat.cpp", "src/pipeline/unified.cpp",
            "src/pipeline/narrative_parser.cpp", "src/obs/behavior_evidence.cpp",
            "native/qwen38/engine.cu", "native/qwen38/sampling.hpp",
            "native/qwen38/tokenizer.cpp", "native/stage_cli/qwen38_stage.cpp",
            "native/qwen38_cli/main.cpp", "tests/behavior_logprobs_live.py",
        )
    }
    provenance["binaries"] = {
        "gateway": sha256_file(args.server_binary),
        "qwen_stage": sha256_file(args.stage_binary),
        "native_replay": sha256_file(args.replay_binary),
    }

    processes = []
    captures = []
    summary = {
        "status": "RUNNING", "provenance": provenance, "cases": [],
        "claims": {
            "optimizer": "NOT_RUN", "model_quality": "NOT_COVERED",
            "visual_output": "CPU contract fixture; not model generation",
            "gpu": 7, "resident_service": "untouched",
        },
    }
    try:
        base_url, processes, launches = start_stack(args, args.out_dir)
        summary["loopback_port"] = base_url.rsplit(":", 1)[1]
        summary["stack_commands"] = launches
        recorder = traj.TrajectoryRecorder(args.out_dir / "live/trajectories.jsonl")
        for index, (case, body) in enumerate(request_cases(args.model), start=1):
            (args.out_dir / "requests" / (case + ".json")).write_bytes(
                canonical(body) + b"\n")
            plan, item, headers = attestation_for(body, case, provenance)
            captured = capture_case(
                base_url, case, body, recorder, plan, item, headers,
                cancel_request_id=f"http-{index}" if case == "cancel" else None)
            record = captured["record"]
            if case in ("text", "visual"):
                require(captured["coverage"] == "exposed" and
                        captured["behavior"]["status"] == "COVERED",
                        f"{case} did not expose complete evidence")
                require(record["terminal_state"] == "completed",
                        f"{case} did not complete")
            elif case == "default":
                require(record["clean_text"] == captures[0]["record"]["clean_text"],
                        "default output changed under the same seed/prompt")
                require(all("behavior" not in chunk for chunk in captured["chunks"]),
                        "default response gained behavior fields")
            else:
                require(captured["coverage"] == "not_exposed" and
                        captured["behavior"]["status"] == "NOT_COVERED",
                        f"{case} did not fail closed")
                require(all("behavior" not in chunk for chunk in captured["chunks"]),
                        f"{case} emitted a misleading behavior block")
            if case == "cancel":
                require(record["terminal_state"] == "cancelled" and
                        captured["abort"].get("status") == 200,
                        "real cancellation did not reach a cancelled terminal")
            if case == "truncated":
                require(record["terminal_state"] == "completed" and
                        captured["chunks"][-1]["choices"][0]["finish_reason"] == "length",
                        "truncation fixture did not end at the token limit")
            captures.append(captured)
            summary["cases"].append({
                "case": case, "coverage": captured["coverage"],
                "behavior": captured["behavior"],
                "raw_sha256": record["sse_replay"]["raw_sha256"],
                "raw_bytes": record["sse_replay"]["raw_bytes"],
                "terminal_state": record["terminal_state"],
                "attestation_sha256": captured["attestation"]["attestation_sha256"],
            })
        worker.validate_trajectory(
            str(args.out_dir / "live/trajectories.jsonl"),
            artifact_dir=args.out_dir / "live/artifacts")
        stop_processes(processes)
        processes = []
        summary["serving_cleanup"] = "gateway, Qwen stage, and both fixtures exited"

        for captured in captures[:2]:
            case = captured["case"]
            body = dict(request_cases(args.model))[case]
            replay = fresh_native_replay(args, case, body, captured, args.out_dir)
            summary["cases"][0 if case == "text" else 1]["fresh_forward"] = replay

        source_jsonl = args.out_dir / "live/trajectories.jsonl"
        replay_jsonl = args.out_dir / "replay/trajectories.jsonl"
        summary["fresh_sse_replay"] = fresh_replay_records(
            source_jsonl, replay_jsonl, len(captures))
        for captured in captures:
            case = captured["case"]
            variant = replay_tools.framing_variant(
                captured["record"], (args.out_dir / "live/artifacts" /
                Path(captured["record"]["sse_replay"]["raw_ref"]).name).read_bytes(),
                args.out_dir / "live/artifacts",
                args.out_dir / "framing" / case / "trajectories.jsonl",
                (1, 2, 3, 5, 8, 13))
            summary["cases"][["text", "visual", "truncated", "cancel", "default"].index(case)]["framing"] = variant
        health = http.client.HTTPConnection(base_url, timeout=2)
        try:
            health.request("GET", "/health")
            response = health.getresponse()
            response.read()
            raise AssertionError("isolated gateway still answers after cleanup")
        except OSError:
            pass
        finally:
            health.close()
        summary["status"] = "PASS"
    except Exception as error:
        summary["status"] = "FAIL"
        summary["error"] = str(error)
        raise
    finally:
        if processes:
            stop_processes(processes)
            summary["serving_cleanup"] = "processes stopped after failure"
        (args.out_dir / "summary.json").write_bytes(canonical(summary) + b"\n")
        files = []
        for path in sorted(args.out_dir.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                files.append(f"{sha256_file(path)['sha256']}  {path.relative_to(args.out_dir)}")
        (args.out_dir / "SHA256SUMS").write_text("\n".join(files) + "\n",
                                                 encoding="ascii")
    print(f"REAL_BEHAVIOR_LOGPROBS_GATE: {summary['status']} cases={len(captures)}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
