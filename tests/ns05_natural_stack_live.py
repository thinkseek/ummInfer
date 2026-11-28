#!/usr/bin/env python3
"""Launch one GPU-0 native stack for the unchanged Task #72 run-mode gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
import umminfer_long_video_live as long_live  # noqa: E402


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def gpu():
    memory = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout.strip()
    apps = subprocess.run(["nvidia-smi", "-i", "0", "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout
    rows = []
    for line in apps.splitlines():
        if line.strip():
            pid, used = [int(value.strip()) for value in line.split(",")]
            rows.append({"pid": pid, "used_mib": used})
    return {"memory_used_mib": int(memory), "compute_apps": rows}


def stop(processes):
    records = []
    for name in ("gateway", "native"):
        process = processes[name]
        if process.poll() is None:
            process.terminate()
        try:
            code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait(timeout=10)
        records.append({"name": name, "pid": process.pid, "returncode": code})
    return records


def run(args):
    if os.environ.get("UMMINFER_LIVE_OPT_IN") != "1":
        raise ValueError("live opt-in required")
    if args.mode not in ("serial", "batch4"):
        raise ValueError("invalid mode")
    work = Path(args.work_dir).resolve()
    out = Path(args.out_dir).resolve()
    if work.exists() or out.exists():
        raise ValueError("refusing to overwrite stack/mode evidence")
    work.mkdir(parents=True)
    pre = gpu()
    if pre["compute_apps"] or pre["memory_used_mib"] > 16:
        raise ValueError(f"GPU 0 is not idle: {pre}")
    ipc = work / "ipc"
    ipc.mkdir()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["UMM_QWEN38_PREFIX_CACHE"] = "0"
    env["UMM_NATURAL_PHYSICAL_GPU"] = "0"
    processes = {}
    logs = {}
    error = None
    child_code = 1
    try:
        native_log = work / "native-stage.log"
        logs["native"] = native_log.open("wb")
        processes["native"] = subprocess.Popen([
            str(args.native_stage), "--kind", "qwen38", "--name", "qwen",
            "--model", str(args.model), "--gpu", "0", "--batch-size",
            "1" if args.mode == "serial" else "4", "--recv-endpoint", f"ipc://{ipc}/entry",
            "--coordinator-endpoint", f"ipc://{ipc}/completion",
            "--abort-endpoint", f"ipc://{ipc}/abort"], cwd=args.source_root,
            env=env, stdout=logs["native"], stderr=subprocess.STDOUT, start_new_session=True)
        gateway_log = work / "gateway.log"
        logs["gateway"] = gateway_log.open("wb")
        processes["gateway"] = subprocess.Popen([
            str(args.gateway), "serve", "--host", "127.0.0.1", "--port", str(args.port),
            "--model", "qwen3-native", "--entry", "qwen", "--entry-endpoint", f"ipc://{ipc}/entry",
            "--completion", f"ipc://{ipc}/completion", "--abort", f"ipc://{ipc}/abort",
            "--max-in-flight", "4"], cwd=args.source_root, env=env,
            stdout=logs["gateway"], stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            for name, process in processes.items():
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited before ready")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/ready", timeout=1) as reply:
                    if reply.status == 200:
                        break
            except OSError:
                time.sleep(0.1)
        else:
            raise TimeoutError("stack readiness")
        ready = gpu()
        native_apps = [row for row in ready["compute_apps"] if row["used_mib"] > 1000]
        if len(native_apps) != 1:
            raise ValueError(f"expected one model compute process: {ready}")
        model_files = {path.relative_to(args.model).as_posix(): sha(path)
            for path in sorted(args.model.rglob("*")) if path.is_file()}
        source = long_live.source_state(args.source_root)
        provenance = {"schema_version": 1, "stack_id": args.run_id,
            "stack_run_id": args.run_id, **source, "physical_gpu": 0,
            "batch_size": 1 if args.mode == "serial" else 4,
            "qwen38_prefix_cache": "disabled", "max_residual_memory_mib": 80000,
            "gateway": {"path": str(args.gateway), "pid": processes["gateway"].pid,
                        "sha256": sha(args.gateway)},
            "native_stage": {"path": str(args.native_stage), "pid": processes["native"].pid,
                "compute_pid": native_apps[0]["pid"], "compute_used_mib": native_apps[0]["used_mib"],
                "sha256": sha(args.native_stage)},
            "model": {"id": "qwen3-native", "root": str(args.model),
                      "revision": "UNKNOWN_LOCAL_NO_GIT", "files": model_files}}
        provenance_path = work / "provenance.json"
        write_json(provenance_path, provenance)
        child = [sys.executable, "-I", str(ROOT / "tests/temporal_natural_live.py"), "run-mode",
            "--mode", args.mode, "--url", f"http://127.0.0.1:{args.port}",
            "--native-log", str(native_log), "--provenance", str(provenance_path),
            "--source-root", str(args.source_root), "--run-id", args.run_id,
            "--out-dir", str(out), "--manifest", str(args.manifest), "--freeze", str(args.freeze)]
        with (work / "run-mode.log").open("wb") as log:
            child_code = subprocess.run(child, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=1800).returncode
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
    finally:
        for handle in logs.values():
            handle.flush()
        records = stop(processes) if processes else []
        deadline = time.monotonic() + 15
        post = gpu()
        while (post["compute_apps"] or post["memory_used_mib"] > 16) and time.monotonic() < deadline:
            time.sleep(0.1)
            post = gpu()
        passed = not post["compute_apps"] and post["memory_used_mib"] <= 16
        if out.exists():
            write_json(out / "cleanup.json", {"owned_processes": records,
                "post_stop": post, "passed": passed})
        write_json(work / "stack-cleanup.json", {"owned_processes": records,
            "post_stop": post, "passed": passed, "run_mode_returncode": child_code,
            "error": error})
    if error:
        raise RuntimeError(error)
    return child_code if child_code == 0 and passed else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mode", "work-dir", "out-dir", "source-root", "gateway", "native-stage",
                 "model", "manifest", "freeze", "run-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    args.model = Path(args.model).resolve()
    args.gateway = Path(args.gateway).resolve()
    args.native_stage = Path(args.native_stage).resolve()
    args.manifest = Path(args.manifest).resolve()
    args.freeze = Path(args.freeze).resolve()
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"NATURAL STACK FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
