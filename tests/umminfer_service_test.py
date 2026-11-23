#!/usr/bin/env python3
"""Offline lifecycle tests for python/umminfer_service.py.

The supervisor is driven end to end: config validation, start/status/logs/stop
against a dummy server binary and a dummy stage binary, health-failure restart
with a backoff counter, PID start-time ownership, the rolling log cap, and the
zero-orphan guarantee on exit.

Everything is stdlib, uses ephemeral temp directories and an ephemeral port, and
needs no GPU, no model and no network beyond loopback.

Usage: python3 -I tests/umminfer_service_test.py
"""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE = REPO_ROOT / "python" / "umminfer_service.py"
LOG_CAP = 4096
START_TIMEOUT_S = 20
STOP_TIMEOUT_S = 20


def load_module():
    spec = importlib.util.spec_from_file_location("umminfer_service", SERVICE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["umminfer_service"] = module
    spec.loader.exec_module(module)
    return module


SERVICE_MODULE = load_module()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


DUMMY_SERVER = '''#!/usr/bin/env python3
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

args = sys.argv[1:]
def value(name, fallback):
    try:
        return args[args.index(name) + 1]
    except (ValueError, IndexError):
        return fallback

host = value("--host", "127.0.0.1")
port = int(value("--port", "0"))
mode = os.environ.get("DUMMY_HEALTH", "ok")
print("dummy server argv: " + " ".join(args), flush=True)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def reply(self, status, body, kind="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if mode == "fail" and self.path.startswith("/health"):
            self.reply(500, b'{"error":"unhealthy"}')
            return
        if self.path.startswith("/health"):
            self.reply(200, json.dumps({"in_flight": 0}).encode())
            return
        if self.path.startswith("/ready"):
            self.reply(200, b'{"ready":true}')
            return
        self.reply(404, b'{"error":"not found"}')


def chatter():
    line = b"x" * 200 + b"\\n"
    while True:
        try:
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
        except Exception:
            return
        time.sleep(0.005)


if os.environ.get("DUMMY_CHATTER") == "1":
    threading.Thread(target=chatter, daemon=True).start()

ThreadingHTTPServer((host, port), Handler).serve_forever()
'''

DUMMY_STAGE = '''#!/usr/bin/env python3
import signal
import sys
import time

signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
print("stage " + " ".join(sys.argv[1:]), flush=True)
while True:
    time.sleep(0.1)
'''


def write_script(path, body):
    path.write_text(body)
    path.chmod(0o755)
    return path


def config_for(root, port, **overrides):
    config = {
        "name": "umminfer-test",
        "host": "127.0.0.1",
        "port": port,
        "model": "demo-model",
        "binary": str(root / "dummy-server.py"),
        "stage_binary_dir": str(root / "stages"),
        "run_dir": str(root / "run"),
        "log_dir": str(root / "log"),
        "ipc_dir": str(root / "ipc"),
        "native_stages": [{"name": "omni", "model": "demo-model", "gpu": 0}],
        "health_grace_s": 0.2,
        "health_interval_s": 0.2,
        "ready_grace_s": 0.2,
        "ready_failures": 5,
        "stop_grace_s": 2,
        "backoff_s": [0.1],
        "stable_s": 30,
        "log_cap_bytes": LOG_CAP,
        "env": {"PYTHONUNBUFFERED": "1"},
    }
    config.update(overrides)
    return config


def write_config(root, config, name="service.json"):
    path = root / name
    path.write_text(json.dumps(config, indent=2))
    return path


def service_command(config_path, *args):
    return [sys.executable, "-I", str(SERVICE), "--config", str(config_path), *args]


def run_service(config_path, *args, timeout=30):
    return subprocess.run(service_command(config_path, *args), capture_output=True,
                          text=True, timeout=timeout)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def wait_for(predicate, timeout_s, message):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError(message)


def state_payload(root):
    return read_json(root / "run" / "state.json")


def child_record(root, name):
    state = state_payload(root) or {}
    for child in state.get("children", []):
        if child.get("name") == name:
            return child
    return None


def alive(pid):
    return SERVICE_MODULE.pid_alive(pid)


def make_root():
    root = Path(tempfile.mkdtemp(prefix="umminfer-service-test-"))
    (root / "stages").mkdir()
    (root / "run").mkdir()
    (root / "log").mkdir()
    write_script(root / "dummy-server.py", DUMMY_SERVER)
    write_script(root / "stages" / "umm-native-stage-omni", DUMMY_STAGE)
    return root


def stop(root, config_path):
    result = run_service(config_path, "stop")
    require(result.returncode == 0, f"stop failed: {result.stderr}")
    return json.loads(result.stdout)


def cleanup(root):
    pidfile = root / "run" / "supervisor.pid"
    record = read_json(pidfile) or {}
    pid = record.get("pid")
    if pid and alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------- config

def test_config_rejects_wildcard_hosts():
    root = make_root()
    try:
        for host in ("0.0.0.0", "::", "[::]"):
            config_path = write_config(
                root, config_for(root, free_port(), host=host), f"bad-{host.strip(':[]')}.json")
            result = run_service(config_path, "status")
            require(result.returncode != 0, f"{host} must be refused")
            require("refusing to bind" in (result.stderr + result.stdout),
                    f"{host} must be refused with a reason: {result.stderr}")
    finally:
        cleanup(root)


def test_config_accepts_loopback():
    root = make_root()
    try:
        config_path = write_config(root, config_for(root, free_port()))
        result = run_service(config_path, "status")
        # status exits 1 when no supervisor runs, but the config must load and
        # produce a JSON payload rather than a config error.
        payload = json.loads(result.stdout)
        require(payload["running"] is False, "a fresh run dir has no supervisor")
        require(payload["state"] == {}, "a fresh run dir has no state")
    finally:
        cleanup(root)


def test_config_rejects_unknown_vlm_stage():
    root = make_root()
    try:
        config_path = write_config(root, config_for(root, free_port(), vlm_stage="asr"))
        result = run_service(config_path, "status")
        require(result.returncode != 0 and "vlm_stage" in (result.stderr + result.stdout),
                "vlm_stage must name a started stage")
    finally:
        cleanup(root)


def test_config_rejects_duplicate_stage_names():
    root = make_root()
    try:
        stages = [{"name": "omni", "model": "m", "gpu": 0},
                  {"name": "omni", "model": "m", "gpu": 0}]
        config_path = write_config(
            root, config_for(root, free_port(), native_stages=stages))
        result = run_service(config_path, "status")
        require(result.returncode != 0 and "duplicate" in (result.stderr + result.stdout),
                "duplicate stage names must be refused")
    finally:
        cleanup(root)


# ---------------------------------------------------------------- lifecycle

def test_full_lifecycle():
    root = make_root()
    config_path = write_config(root, config_for(root, free_port()))
    try:
        result = run_service(config_path, "start")
        require(result.returncode == 0, f"start failed: {result.stderr}")
        started = json.loads(result.stdout)
        require(started["started"] is True, "start must report started")
        supervisor_pid = started["supervisor"]["pid"]
        require(alive(supervisor_pid), "supervisor must be alive after start")

        def healthy():
            record = child_record(root, "server")
            state = state_payload(root) or {}
            health = state.get("health", {})
            return (record and record["alive"] and record["pid"]
                    and health.get("ok") is True and record["pid"])

        server_pid = wait_for(healthy, START_TIMEOUT_S,
                              "the dummy server never became healthy")

        status = run_service(config_path, "status")
        require(status.returncode == 0, "status must exit 0 while running")
        payload = json.loads(status.stdout)
        require(payload["running"] is True, "status must report running")
        require(payload["supervisor"]["pid"] == supervisor_pid,
                "status must report the supervisor pid")
        names = {child["name"] for child in payload["state"]["children"]}
        require(names == {"server", "omni"}, f"unexpected children: {names}")

        logs = run_service(config_path, "logs", "--name", "server")
        require(logs.returncode == 0, "logs must exit 0 for an existing log")
        require("dummy-server.py" in logs.stdout or "serve" in logs.stdout,
                f"server log must record its argv: {logs.stdout[:200]!r}")
        missing = run_service(config_path, "logs", "--name", "absent")
        require(missing.returncode == 1, "logs must exit 1 for a missing log")

        stage_pid = wait_for(lambda: (child_record(root, "omni") or {}).get("pid"),
                             START_TIMEOUT_S, "the stage child never started")
        require(alive(stage_pid), "the stage child must be alive")

        stopped = stop(root, config_path)
        require(stopped["supervisor_was_running"] is True,
                "stop must report the running supervisor")
        require(stopped["stopped"], "stop must list what it signalled")

        require(not alive(supervisor_pid), "the supervisor must exit on stop")
        for pid in (server_pid, stage_pid):
            require(not alive(pid), f"child {pid} must be gone after stop")
        require(not (root / "run" / "supervisor.pid").exists(),
                "stop must remove the pid file")

        final = state_payload(root) or {}
        require(final.get("stopped") is True, "the final state must be stopped")
        final_pids = [child["pid"] for child in final.get("children", [])]
        require(all(pid is None for pid in final_pids),
                f"no owned child may remain in the final state: {final_pids}")

        status_after = run_service(config_path, "status")
        require(status_after.returncode == 1, "status must exit 1 after stop")
    finally:
        cleanup(root)


def test_health_failure_restarts_server_with_backoff_counter():
    root = make_root()
    port = free_port()
    config = config_for(root, port, health_failures=1,
                        env={"DUMMY_HEALTH": "fail", "PYTHONUNBUFFERED": "1"})
    config_path = write_config(root, config)
    try:
        result = run_service(config_path, "start")
        require(result.returncode == 0, f"start failed: {result.stderr}")

        supervisor_log = root / "log" / "supervisor.log"

        def restarted():
            text = supervisor_log.read_text(errors="replace") if supervisor_log.exists() else ""
            started = text.count("server started pid=")
            failed = "health check failed (1)" in text
            threshold = "health threshold reached, restarting server" in text
            return text if (started >= 2 and failed and threshold) else None

        text = wait_for(restarted, START_TIMEOUT_S,
                        "health failure never restarted the server")
        # A 500 raises urllib.error.HTTPError, which check_health records as an
        # unreachable status rather than the numeric code.
        require("status=None" in text or "status=500" in text,
                "the failed health check must be logged with its status")
        state = state_payload(root) or {}
        require(state["health"]["ok"] is False, "health must report not ok while failing")
        require(state["health"]["checked_at"] is not None,
                "the health check time must be recorded")

        stop(root, config_path)
    finally:
        cleanup(root)


def test_server_restart_preserves_running_workers():
    root = make_root()
    port = free_port()
    config = config_for(root, port, ready_path="/ready", ready_grace_s=0.2,
                        health_grace_s=0.1)
    config_path = write_config(root, config)
    try:
        result = run_service(config_path, "start")
        require(result.returncode == 0, f"start failed: {result.stderr}")

        def healthy():
            record = child_record(root, "server")
            stage = child_record(root, "omni")
            state = state_payload(root) or {}
            if not (record and record["alive"] and state.get("health", {}).get("ok") is True):
                return None
            if not (stage and stage["alive"] and stage["pid"]):
                return None
            return record["pid"], stage["pid"]

        server_pid, stage_pid = wait_for(healthy, START_TIMEOUT_S,
                                         "the service never became healthy")

        # Crash only the gateway. The stage process must survive the restart.
        os.kill(server_pid, signal.SIGKILL)

        def recovered():
            record = child_record(root, "server")
            stage = child_record(root, "omni")
            state = state_payload(root) or {}
            if not (record and record["alive"] and record["pid"] != server_pid):
                return None
            if not (stage and stage["alive"] and stage["pid"] == stage_pid):
                return None
            if state.get("health", {}).get("ok") is not True:
                return None
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/ready", timeout=2) as response:
                    ready = response.status
            except (urllib.error.URLError, OSError):
                return None
            return record["pid"] if ready == 200 else None

        wait_for(recovered, START_TIMEOUT_S,
                 "the server did not restart while preserving the stage")

        stage_after = child_record(root, "omni") or {}
        require(stage_after.get("pid") == stage_pid,
                "the stage PID changed across the server restart")
        require(stage_after.get("restarts") == 0,
                f"the stage must not be restarted: {stage_after.get('restarts')}")

        supervisor_log = (root / "log" / "supervisor.log").read_text(errors="replace")
        require("stages preserved because server" in supervisor_log,
                "the supervisor did not record worker preservation")

        stop(root, config_path)
    finally:
        cleanup(root)


def test_foreign_pid_is_not_signalled():
    root = make_root()
    victim = subprocess.Popen(["sleep", "30"])
    try:
        require(alive(victim.pid), "the victim must start")
        child = SERVICE_MODULE.Child("server", ["true"])
        child.pid = victim.pid
        # A start time that cannot match any live process.
        child.starttime = "proc:0"
        require(child.alive() is False, "a mismatched start time must not be alive")

        supervisor = SERVICE_MODULE.Supervisor(config_for(root, free_port()))
        supervisor.terminate(child, grace=0.2)
        require(victim.poll() is None,
                "terminate must not signal a PID whose start time changed")

        # The real start time is accepted by the same check.
        child.pid = victim.pid
        child.starttime = SERVICE_MODULE.process_starttime(victim.pid)
        require(child.alive() is True, "a matching start time must be alive")
    finally:
        victim.terminate()
        victim.wait(timeout=5)
        cleanup(root)


def test_rolling_log_cap():
    root = make_root()
    config = config_for(root, free_port(),
                        env={"DUMMY_CHATTER": "1", "PYTHONUNBUFFERED": "1"})
    config_path = write_config(root, config)
    try:
        require(run_service(config_path, "start").returncode == 0, "start failed")
        server_log = root / "log" / "server.log"
        wait_for(lambda: server_log.exists() and server_log.stat().st_size > 0,
                 START_TIMEOUT_S, "the dummy server never logged")
        # 200 bytes every 5 ms overflows the 4 KiB cap in well under a second.
        time.sleep(1.0)
        require(server_log.stat().st_size <= LOG_CAP,
                f"server.log must stay within the cap: {server_log.stat().st_size}")

        tail = run_service(config_path, "logs", "--name", "server", "--bytes", "1048576")
        require(tail.returncode == 0, "logs failed")
        require(len(tail.stdout.encode()) <= LOG_CAP,
                "logs must clamp --bytes to the configured cap")
        require(len(tail.stdout) > 0, "the capped tail must not be empty")

        supervisor_log = root / "log" / "supervisor.log"
        require(supervisor_log.stat().st_size <= LOG_CAP,
                "supervisor.log must stay within the cap")
        stop(root, config_path)
    finally:
        cleanup(root)


def main():
    tests = [
        test_config_rejects_wildcard_hosts,
        test_config_accepts_loopback,
        test_config_rejects_unknown_vlm_stage,
        test_config_rejects_duplicate_stage_names,
        test_full_lifecycle,
        test_health_failure_restarts_server_with_backoff_counter,
        test_server_restart_preserves_running_workers,
        test_foreign_pid_is_not_signalled,
        test_rolling_log_cap,
    ]
    failures = 0
    for test in tests:
        started = time.monotonic()
        try:
            test()
        except Exception as error:  # noqa: BLE001 - one report per failure
            failures += 1
            print(f"FAIL {test.__name__}: {error}", flush=True)
        else:
            print(f"ok {test.__name__} ({time.monotonic() - started:.1f}s)", flush=True)
    if failures:
        print(f"{failures} of {len(tests)} tests failed", flush=True)
        return 1
    print(f"{len(tests)} tests passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
