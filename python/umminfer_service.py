#!/usr/bin/env python3
"""Resident supervisor for ummInfer servers and their model workers.

The default configuration owns three children: the C++ server, a Python ASR
worker and a Python Omni worker. A configuration with `native_stages` owns the
C++ server plus each listed native stage and needs no Python worker or SGLang
URL. It records each child's PID and kernel start time, restarts a dead or
unhealthy child with bounded backoff, keeps at most 64 KiB of each log, and
refuses to signal a PID whose start time no longer matches the recorded one.

It uses only the Python standard library and detaches from the controlling
terminal, so it survives the SSH session that started it.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

LOG_CAP = 64 * 1024
POLL_S = 0.2
FORBIDDEN_HOSTS = {"0.0.0.0", "::", "[::]"}


def process_starttime(pid):
    """Return a stable start-time identity for a PID, or None.

    Linux reads field 22 of /proc/<pid>/stat. Other systems use the process
    start timestamp from ps. Both values change if the PID is reused.
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            data = handle.read()
    except OSError:
        data = None
    if data is not None:
        end = data.rfind(b")")
        if end < 0:
            return None
        fields = data[end + 2:].split()
        if len(fields) < 20:
            return None
        return f"proc:{fields[19].decode()}"
    try:
        result = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)],
                                capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return f"ps:{value}" if value else None


def pid_alive(pid, starttime=None):
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    now = process_starttime(pid)
    if now is None:
        return False
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            data = handle.read()
        if data[data.rfind(b")") + 2:].split()[0] == b"Z":
            return False
    except OSError:
        pass
    return starttime is None or now == starttime


def write_json(path, payload):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def tail_bytes(path, cap=LOG_CAP):
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - cap))
            return handle.read()
    except OSError:
        return b""


class CappedLog:
    """Write a child's merged output, keeping only the last LOG_CAP bytes."""

    def __init__(self, path):
        self.path = Path(path)
        self.bytes = bytearray()
        self.lock = threading.Lock()
        self.thread = None

    def attach(self, process):
        self.thread = threading.Thread(
            target=self._drain, args=(process,), daemon=True)
        self.thread.start()

    def _drain(self, process):
        fd = process.stdout.fileno()
        while True:
            try:
                data = os.read(fd, 4096)
            except OSError:
                break
            if not data:
                break
            with self.lock:
                self.bytes.extend(data)
                if len(self.bytes) > LOG_CAP:
                    del self.bytes[:len(self.bytes) - LOG_CAP]
                self._flush()

    def _flush(self):
        tmp = Path(str(self.path) + ".tmp")
        tmp.write_bytes(bytes(self.bytes))
        os.replace(tmp, self.path)

    def text(self, limit=LOG_CAP):
        with self.lock:
            return bytes(self.bytes[-limit:]).decode("utf-8", "replace")


class Child:
    def __init__(self, name, argv, cwd=None, depends_on=None):
        self.name = name
        self.argv = argv
        self.cwd = cwd
        self.depends_on = depends_on
        self.process = None
        self.pid = None
        self.starttime = None
        self.started_at = None
        self.started_mono = 0.0
        self.failures = 0
        self.restarts = 0
        self.next_start = 0.0
        self.log = None

    def alive(self):
        return pid_alive(self.pid, self.starttime)


class Supervisor:
    def __init__(self, config):
        self.config = config
        self.run_dir = Path(config["run_dir"])
        self.log_dir = Path(config["log_dir"])
        self.state_path = self.run_dir / "state.json"
        self.pid_path = self.run_dir / "supervisor.pid"
        self.stop_requested = False
        self.health_ok = None
        self.health_failures = 0
        self.health_checked_mono = 0.0
        self.health_checked_at = None
        self.in_flight = None
        self.children = self._build_children()
        self.by_name = {child.name: child for child in self.children}

    def _build_children(self):
        cfg = self.config
        ipc = cfg["ipc_dir"]
        if "native_stages" in cfg:
            children = [Child("server", [
                cfg["binary"], "serve",
                "--host", cfg["host"], "--port", str(cfg["port"]),
                "--model", cfg["model"],
                "--entry", "asr", "--entry-endpoint", f"ipc://{ipc}/asr",
                "--vlm-entry", "omni", "--vlm-entry-endpoint", f"ipc://{ipc}/omni",
                "--image-entry", "image", "--image-entry-endpoint", f"ipc://{ipc}/image",
                "--tts-entry", "tts", "--tts-entry-endpoint", f"ipc://{ipc}/tts",
                "--completion", f"ipc://{ipc}/done",
                "--abort", f"ipc://{ipc}/abort",
            ] + list(cfg.get("server_args", [])))]
            binary_dir = cfg["stage_binary_dir"]
            for stage in cfg["native_stages"]:
                name = stage["name"]
                kind = stage.get("kind", name)
                argv = [
                    str(Path(binary_dir) / f"umm-native-stage-{kind}"),
                    "--kind", kind, "--name", name,
                    "--model", stage["model"], "--gpu", str(stage["gpu"]),
                    "--recv-endpoint", f"ipc://{ipc}/{name}",
                    "--coordinator-endpoint", f"ipc://{ipc}/done",
                    "--abort-endpoint", f"ipc://{ipc}/abort",
                ] + list(stage.get("args", []))
                children.append(Child(name, argv, depends_on="server"))
            return children

        python = cfg["python"]
        worker_dir = cfg["python_dir"]
        asr = cfg["asr"]
        omni = cfg["omni"]
        server = Child("server", [
            cfg["binary"], "serve",
            "--host", cfg["host"], "--port", str(cfg["port"]),
            "--model", cfg["model"],
            "--entry", "asr", "--entry-endpoint", f"ipc://{ipc}/asr",
            "--vlm-entry", "vlm", "--vlm-entry-endpoint", f"ipc://{ipc}/vlm",
            "--completion", f"ipc://{ipc}/done",
            "--abort", f"ipc://{ipc}/abort",
        ] + list(cfg.get("server_args", [])))
        asr_worker = Child("asr", [
            python, "-I", str(Path(worker_dir) / "umminfer_qwen_asr_stage.py"),
            "serve",
            "--name", "asr", "--recv", f"ipc://{ipc}/asr",
            "--coordinator", f"ipc://{ipc}/done",
            "--abort", f"ipc://{ipc}/abort",
            "--model", asr["model"], "--backend-url", asr["url"],
            "--concurrency", str(asr.get("concurrency", 1)),
        ], cwd=worker_dir, depends_on="server")
        omni_worker = Child("omni", [
            python, "-I", str(Path(worker_dir) / "umminfer_qwen_omni_worker.py"),
            "--name", "vlm", "--recv", f"ipc://{ipc}/vlm",
            "--coordinator", f"ipc://{ipc}/done",
            "--abort", f"ipc://{ipc}/abort",
            "--upstream-base-url", omni["url"],
            "--upstream-model", omni["model"],
            "--concurrency", str(omni.get("concurrency", 1)),
        ], cwd=worker_dir, depends_on="server")
        return [server, asr_worker, omni_worker]

    def log(self, message):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} supervisor {message}\n"
        path = self.log_dir / "supervisor.log"
        data = tail_bytes(path) + line.encode()
        if len(data) > LOG_CAP:
            data = data[-LOG_CAP:]
        tmp = Path(str(path) + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def spawn(self, child):
        child.log = CappedLog(self.log_dir / f"{child.name}.log")
        env = dict(os.environ)
        env.update(self.config.get("env", {}))
        try:
            child.process = subprocess.Popen(
                child.argv, cwd=child.cwd, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                start_new_session=True, env=env)
        except OSError as error:
            child.process = None
            child.pid = None
            child.starttime = None
            child.failures += 1
            child.next_start = time.monotonic() + self.backoff(child)
            self.log(f"{child.name} spawn failed: {error}")
            return False
        child.pid = child.process.pid
        child.starttime = process_starttime(child.pid)
        child.started_at = time.time()
        child.started_mono = time.monotonic()
        child.log.attach(child.process)
        self.log(f"{child.name} started pid={child.pid} "
                 f"starttime={child.starttime} argv={child.argv}")
        return True

    def backoff(self, child):
        schedule = self.config.get("backoff_s", [1, 2, 4, 8, 15, 30])
        index = min(max(child.failures - 1, 0), len(schedule) - 1)
        return schedule[index]

    def terminate(self, child, grace=None):
        """Stop one owned child. Signals only a matching PID/start-time pair."""
        if not child.alive():
            return
        grace = self.config.get("stop_grace_s", 10) if grace is None else grace
        pid = child.pid
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except PermissionError as error:
            self.log(f"{child.name} SIGTERM denied: {error}")
            return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if child.process is not None:
                if child.process.poll() is not None:
                    break
            if not pid_alive(pid, child.starttime):
                break
            time.sleep(0.1)
        if pid_alive(pid, child.starttime):
            try:
                os.kill(pid, signal.SIGKILL)
                self.log(f"{child.name} SIGKILL pid={pid}")
            except ProcessLookupError:
                pass
        if child.process is not None:
            try:
                child.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        child.process = None
        child.pid = None
        child.starttime = None

    def restart_workers(self, reason):
        for name in ("asr", "omni"):
            child = self.by_name[name]
            self.terminate(child)
            child.failures = 0
            child.next_start = 0.0
        self.log(f"workers restarted because server {reason}")

    def check_health(self):
        url = (f"http://{self.config['host']}:{self.config['port']}"
               f"{self.config.get('health_path', '/health')}")
        status = None
        in_flight = None
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                status = response.status
                body = json.loads(response.read().decode("utf-8", "replace"))
                in_flight = body.get("in_flight")
        except (urllib.error.URLError, OSError, ValueError) as error:
            status = None
            self.last_health_error = str(error)
        self.health_checked_mono = time.monotonic()
        self.health_checked_at = time.time()
        if status == 200:
            self.health_ok = True
            self.health_failures = 0
            self.in_flight = in_flight
            return
        self.health_ok = False
        self.health_failures += 1
        self.log(f"health check failed ({self.health_failures}) "
                 f"status={status} url={url}")
        if self.health_failures >= self.config.get("health_failures", 3):
            server = self.by_name["server"]
            self.health_failures = 0
            self.health_ok = False
            self.log("health threshold reached, restarting server")
            self.terminate(server)
            server.failures += 1
            server.next_start = time.monotonic() + self.backoff(server)
            self.restart_workers("restarted after failed health check")

    def state(self, stopped=False):
        now = time.time()
        children = []
        for child in self.children:
            alive = child.alive()
            children.append({
                "name": child.name,
                "pid": child.pid,
                "starttime": child.starttime,
                "alive": alive,
                "restarts": child.restarts,
                "uptime_s": round(now - child.started_at, 1)
                if alive and child.started_at else None,
                "log": str(self.log_dir / f"{child.name}.log"),
                "log_bytes": (self.log_dir / f"{child.name}.log").stat().st_size
                if (self.log_dir / f"{child.name}.log").exists() else 0,
            })
        return {
            "service": self.config.get("name", "umminfer"),
            "stopped": stopped,
            "supervisor": {
                "pid": os.getpid(),
                "starttime": process_starttime(os.getpid()),
            },
            "address": f"{self.config['host']}:{self.config['port']}",
            "ipc_dir": self.config["ipc_dir"],
            "health": {
                "ok": self.health_ok,
                "checked_at": self.health_checked_at,
                "age_s": round(time.monotonic() - self.health_checked_mono, 1)
                if self.health_checked_at else None,
                "in_flight": self.in_flight,
            },
            "children": children,
        }

    def save_state(self, stopped=False):
        write_json(self.state_path, self.state(stopped))

    def run(self):
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.config["ipc_dir"], exist_ok=True)
        self.pid_path.write_text(json.dumps({
            "pid": os.getpid(),
            "starttime": process_starttime(os.getpid()),
        }) + "\n")
        self.log(f"supervisor pid={os.getpid()} config={json.dumps(self.config, sort_keys=True)}")

        def request_stop(signum, frame):
            self.stop_requested = True

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)

        server = self.by_name["server"]
        try:
            while not self.stop_requested:
                now = time.monotonic()
                for child in self.children:
                    if child.process is not None and child.process.poll() is not None:
                        code = child.process.returncode
                        child.process = None
                        child.pid = None
                        child.starttime = None
                        child.failures += 1
                        child.restarts += 1
                        child.next_start = now + self.backoff(child)
                        self.log(f"{child.name} exited code={code} "
                                 f"failures={child.failures} "
                                 f"restart_in={child.next_start - now:.1f}s")
                        if child.name == "server":
                            self.health_ok = None
                            self.restart_workers("exited")
                    if child.process is None and now >= child.next_start:
                        if child.depends_on == "server" and not (
                                server.alive() and self.health_ok is True):
                            continue
                        self.spawn(child)
                    if (child.process is not None and child.started_mono
                            and now - child.started_mono
                            >= self.config.get("stable_s", 30)):
                        child.failures = 0
                interval = self.config.get("health_interval_s", 2)
                grace = self.config.get("health_grace_s", 10)
                starting = (server.started_at is None
                            or time.time() - server.started_at < grace)
                if time.monotonic() - self.health_checked_mono >= interval:
                    if server.alive() and not starting:
                        self.check_health()
                self.save_state()
                time.sleep(POLL_S)
        finally:
            for child in self.children:
                self.terminate(child)
            self.save_state(stopped=True)
            self.log("supervisor stopped, no owned child remains")
            try:
                self.pid_path.unlink()
            except OSError:
                pass


def load_config(path):
    config = read_json(path)
    if config is None:
        raise SystemExit(f"cannot read config: {path}")
    host = config.get("host", "")
    if host in FORBIDDEN_HOSTS:
        raise SystemExit(
            f"refusing to bind {host}: use the container private address "
            "or loopback")
    for key in ("run_dir", "log_dir", "ipc_dir", "binary", "model"):
        if key not in config:
            raise SystemExit(f"config is missing {key}")
    if "native_stages" in config:
        if "stage_binary_dir" not in config:
            raise SystemExit("config is missing stage_binary_dir")
        if not isinstance(config["native_stages"], list) or not config["native_stages"]:
            raise SystemExit("native_stages must be a nonempty array")
        for stage in config["native_stages"]:
            for key in ("name", "model", "gpu"):
                if key not in stage:
                    raise SystemExit(f"native stage is missing {key}")
        return config
    for key in ("python", "python_dir", "asr", "omni"):
        if key not in config:
            raise SystemExit(f"config is missing {key}")
    return config


def read_state(config):
    return read_json(Path(config["run_dir"]) / "state.json")


def supervisor_identity(config):
    pidfile = Path(config["run_dir"]) / "supervisor.pid"
    record = read_json(pidfile)
    if not record:
        return None
    if pid_alive(record.get("pid"), record.get("starttime")):
        return record
    return None


def cmd_start(config):
    if supervisor_identity(config):
        print(json.dumps({"error": "already running",
                          "pidfile": str(Path(config["run_dir"])
                                          / "supervisor.pid")}))
        return 1
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    pid_path = run_dir / "supervisor.pid"
    try:
        pid_path.unlink()
    except OSError:
        pass
    pid = os.fork()
    if pid > 0:
        os.waitpid(pid, 0)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            record = supervisor_identity(config)
            if record:
                print(json.dumps({"started": True, "supervisor": record}))
                return 0
            time.sleep(0.1)
        print(json.dumps({"error": "supervisor did not write a pid file"}))
        return 1
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)
    supervisor = Supervisor(config)
    try:
        supervisor.run()
    finally:
        os._exit(0)


def cmd_stop(config):
    state = read_state(config) or {}
    identity = supervisor_identity(config)
    stopped = []
    if identity:
        try:
            os.kill(identity["pid"], signal.SIGTERM)
        except ProcessLookupError:
            identity = None
    if identity:
        grace = config.get("stop_grace_s", 10)
        deadline = time.monotonic() + grace + 10
        while time.monotonic() < deadline:
            if not pid_alive(identity["pid"], identity["starttime"]):
                break
            time.sleep(0.1)
        if pid_alive(identity["pid"], identity["starttime"]):
            os.kill(identity["pid"], signal.SIGKILL)
            stopped.append({"pid": identity["pid"], "signal": "SIGKILL"})
        else:
            stopped.append({"pid": identity["pid"], "signal": "SIGTERM"})
    # A supervisor that died without cleanup leaves children behind. Signal
    # only PIDs whose recorded start time still matches.
    for child in state.get("children", []):
        pid = child.get("pid")
        starttime = child.get("starttime")
        if pid_alive(pid, starttime):
            os.kill(pid, signal.SIGTERM)
            grace = config.get("stop_grace_s", 10)
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline and pid_alive(pid, starttime):
                time.sleep(0.1)
            if pid_alive(pid, starttime):
                os.kill(pid, signal.SIGKILL)
            stopped.append({"name": child.get("name"), "pid": pid})
    pid_path = Path(config["run_dir"]) / "supervisor.pid"
    try:
        pid_path.unlink()
    except OSError:
        pass
    print(json.dumps({"stopped": stopped, "supervisor_was_running": bool(identity)}))
    return 0


def cmd_status(config):
    identity = supervisor_identity(config)
    state = read_state(config) or {}
    payload = {
        "running": bool(identity),
        "supervisor": identity,
        "state": state,
        "pidfile": str(Path(config["run_dir"]) / "supervisor.pid"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if identity else 1


def cmd_logs(config, name, size):
    size = min(max(size, 1), LOG_CAP)
    path = Path(config["log_dir"]) / f"{name}.log"
    sys.stdout.write(tail_bytes(path, size).decode("utf-8", "replace"))
    return 0 if path.exists() else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("start")
    commands.add_parser("stop")
    commands.add_parser("status")
    logs = commands.add_parser("logs")
    logs.add_argument("--name", default="supervisor")
    logs.add_argument("--bytes", type=int, default=LOG_CAP)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "start":
        return cmd_start(config)
    if args.command == "stop":
        return cmd_stop(config)
    if args.command == "status":
        return cmd_status(config)
    return cmd_logs(config, args.name, args.bytes)


if __name__ == "__main__":
    sys.exit(main())
