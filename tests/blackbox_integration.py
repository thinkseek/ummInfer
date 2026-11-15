#!/usr/bin/env python3
"""Fast local black-box integration gate.

Runs the local suites that need no H20B, no CUDA and no RL training, one after
another, and reuses them as they are. This file adds no assertions: every suite
keeps its own checks and its own exit code, which is reported as REAL_EXIT.

Each suite runs in its own session. Its combined stdout and stderr is capped at
LOG_CAP bytes (tail kept), and the whole process group is killed on every exit,
so no child outlives the gate. The gate exits non-zero if any suite fails.

This is a fast gate, not hardware evidence. Real H20B, CUDA and RL training
results still need the dedicated runs in docs/validation.md.
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


LOG_CAP = 65536


def drain(stream, state):
    """Read the pipe and keep only the last LOG_CAP bytes."""
    tail = bytearray()
    while True:
        chunk = stream.read(65536)
        if not chunk:
            break
        state["total"] += len(chunk)
        if len(chunk) >= LOG_CAP:
            tail[:] = chunk[-LOG_CAP:]
        else:
            overflow = len(tail) + len(chunk) - LOG_CAP
            if overflow > 0:
                del tail[:overflow]
            tail.extend(chunk)
        state["peak"] = max(state["peak"], len(tail))
    state["tail"] = bytes(tail)


def _sigterm(signum, frame):
    raise KeyboardInterrupt


def run_suite(name, command, cwd, env, timeout):
    print(f"=== {name}: {' '.join(str(part) for part in command)}", flush=True)
    process = subprocess.Popen(command, cwd=cwd, env=env,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               start_new_session=True)
    state = {"total": 0, "peak": 0, "tail": b""}
    reader = threading.Thread(target=drain, args=(process.stdout, state),
                              daemon=True)
    reader.start()
    try:
        process.wait(timeout=timeout)
        code = process.returncode
    except subprocess.TimeoutExpired:
        code = 124
    except KeyboardInterrupt:
        code = 130
    finally:
        # A suite is required to leave no descendants, including after its own
        # main process exits or the gate is interrupted.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    reader.join(timeout=10)
    if reader.is_alive():
        # Do not close a pipe from another thread while read() may be blocked:
        # close() itself can wait for that read. The daemon reader cannot keep
        # the gate alive, and an open pipe after group cleanup is a failure.
        if code == 0:
            code = 125
    total, tail = state["total"], state["tail"]
    if total > len(tail):
        print(f"[blackbox] {total - len(tail)} earlier bytes dropped",
              flush=True)
    sys.stdout.write(tail.decode(errors="replace"))
    sys.stdout.write("" if tail.endswith(b"\n") else "\n")
    print(f"[blackbox] {name} log_total={total} retained={len(tail)} "
          f"peak={state['peak']}", flush=True)
    if code == 124:
        print(f"[blackbox] {name} exceeded {timeout}s and its group was killed",
              flush=True)
    if code == 125:
        print(f"[blackbox] {name} left its output pipe open", flush=True)
    print(f"REAL_EXIT {name} {code}", flush=True)
    return code, total, state["peak"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--deps-prefix", type=Path,
                        default=Path(os.environ.get("UMMINFER_DEPS_PREFIX",
                                                    "/opt/homebrew")))
    parser.add_argument("--curl-prefix", type=Path,
                        default=Path(os.environ.get(
                            "UMMINFER_CURL_PREFIX", "/opt/homebrew/opt/curl")))
    parser.add_argument("--timeout", type=int, default=1800,
                        help="per-suite timeout in seconds")
    parser.add_argument("--self-check", action="store_true",
                        help="prove the streaming log cap with a 4 MiB producer")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _sigterm)

    project = args.project.resolve()
    binary = (args.binary or project / "zig-out/bin/umm-infer").resolve()
    deps = str(args.deps_prefix.resolve())
    curl = str(args.curl_prefix.resolve())

    env = os.environ.copy()
    env["UMMINFER_DEPS_PREFIX"] = deps
    env["UMMINFER_CURL_PREFIX"] = curl

    if args.self_check:
        producer = [sys.executable, "-I", "-c",
                    "import sys\n"
                    "block = b'x' * 65536\n"
                    "for _ in range(64):\n"
                    "    sys.stdout.buffer.write(block)\n"]
        code, total, peak = run_suite("log-cap-check", producer, project, env, 120)
        ok = code == 0 and total >= 4 * 1024 * 1024 and peak <= LOG_CAP
        print(f"SELF_CHECK log_cap {'PASS' if ok else 'FAIL'} exit={code} "
              f"total={total} peak={peak} cap={LOG_CAP}")

        marker = f"blackbox-timeout-probe-{os.getpid()}"
        code, total, peak = run_suite(
            "timeout-check",
            [sys.executable, "-I", "-c",
             "import sys, time\n" f"# {marker}\n"
             "print('started', flush=True)\ntime.sleep(60)\n"],
            project, env, 2)
        left = subprocess.run(["pgrep", "-f", marker],
                              stdout=subprocess.DEVNULL)
        timeout_ok = code == 124 and total > 0 and peak <= LOG_CAP and left.returncode == 1
        print(f"SELF_CHECK timeout {'PASS' if timeout_ok else 'FAIL'} "
              f"exit={code} total={total} residue_pgrep_exit={left.returncode}")

        marker = f"blackbox-orphan-probe-{os.getpid()}"
        started = time.monotonic()
        code, total, peak = run_suite(
            "orphan-check",
            [sys.executable, "-I", "-c",
             "import subprocess, sys\n"
             "subprocess.Popen([sys.executable, '-I', '-c', "
             f"'import time\\n# {marker}\\ntime.sleep(60)'])\n"
             "print('parent exited; child still holds stdout', flush=True)\n"],
            project, env, 120)
        elapsed = time.monotonic() - started
        left = subprocess.run(["pgrep", "-f", marker],
                              stdout=subprocess.DEVNULL)
        orphan_ok = (code == 0 and elapsed < 10 and peak <= LOG_CAP
                     and left.returncode == 1)
        print(f"SELF_CHECK orphan {'PASS' if orphan_ok else 'FAIL'} "
              f"exit={code} elapsed={elapsed:.3f} "
              f"residue_pgrep_exit={left.returncode}")
        return 0 if ok and timeout_ok and orphan_ok else 1

    zig_flags = [f"-Ddeps-prefix={deps}", f"-Dcurl-prefix={curl}"]
    python = [sys.executable, "-I"]

    # The build only produces the binary the black-box suites need. Its exit
    # code is reported, but a later suite still runs so one log shows both.
    suites = [
        ("build", ["zig", "build", "-Doptimize=ReleaseFast"] + zig_flags),
        ("zig-test", ["zig", "build", "test"] + zig_flags),
        ("unified-test", ["zig", "build", "unified-test"] + zig_flags),
        ("realtime-contract", python + [
            "tests/realtime_contract_e2e.py",
            "--server", str(binary),
            "--stage", "python/umminfer_tiny_asr_stage.py"]),
        ("realtime-client-contract", python + [
            "tests/realtime_client_contract_e2e.py", str(binary),
            "--project", str(project),
            "--deps-prefix", deps,
            "--curl-prefix", curl]),
        ("qwen-asr-stage", python + ["tests/qwen_asr_stage_test.py"]),
        ("qwen-omni-worker", python + ["tests/qwen_omni_worker_test.py"]),
        ("mlx-stage", python + ["tests/mlx_stage_test.py"]),
        ("rl-phase1", python + ["tests/rl_architecture_test.py"]),
        ("rl-phase2", python + ["tests/rl_phase2_test.py"]),
        ("benchmark", python + ["tests/unified_benchmark_test.py"]),
        ("provenance", python + ["tests/model_provenance_test.py"]),
        ("mac-unified", python + ["tests/mac_unified_validation_test.py"]),
    ]

    failures = []
    for name, command in suites:
        code, _, _ = run_suite(name, command, project, env, args.timeout)
        if code == 130:
            return 130
        if code != 0:
            failures.append((name, code))

    print(f"BLACKBOX_SUMMARY suites={len(suites)} failed={len(failures)}")
    for name, code in failures:
        print(f"BLACKBOX_FAIL {name} exit={code}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
