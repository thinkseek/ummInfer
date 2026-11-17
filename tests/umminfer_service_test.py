#!/usr/bin/env python3
"""Local contract test for python/umminfer_service.py.

Runs the real supervisor CLI against a fake HTTP server and four fake native
stage binaries that sleep. It covers start/status/stop, child and server crash
restart, health-failure restart, the 64 KiB log cap, and the rule that stop
never signals a PID whose start time does not match the recorded one.
"""

import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
SERVICE = ROOT / "python" / "umminfer_service.py"
sys.path.insert(0, str(ROOT / "python"))
import umminfer_service  # noqa: E402

CAP = 64 * 1024
FAILURES = []

FAKE_SERVER = '''\
#!/usr/bin/env python3
import http.server, os, sys, time
host = "127.0.0.1"
port = 0
for index, value in enumerate(sys.argv):
    if value == "--host": host = sys.argv[index + 1]
    if value == "--port": port = int(sys.argv[index + 1])
if os.environ.get("FAKE_BIG_LOG"):
    sys.stdout.write("server-start " + "x" * 300000 + "\\n")
    sys.stdout.flush()
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        flag = os.environ.get("FAKE_HEALTH_FLAG")
        down = bool(flag) and os.path.exists(flag)
        body = b'{"status":"ok","in_flight":0}'
        self.send_response(500 if down else 200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass
http.server.HTTPServer((host, port), Handler).serve_forever()
'''

FAKE_STAGE = '''\
#!/usr/bin/env python3
import os, sys, time
if os.environ.get("FAKE_BIG_LOG"):
    sys.stdout.write("stage-start " + "y" * 300000 + "\\n")
    sys.stdout.flush()
while True:
    time.sleep(0.2)
'''


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def check(name, condition, detail=""):
    if condition:
        print(f"PASS {name}")
    else:
        print(f"FAIL {name} {detail}")
        FAILURES.append(name)


def run(config, *args, env=None, timeout=30):
    merged = dict(os.environ)
    if env:
        merged.update(env)
    return subprocess.run(
        [sys.executable, "-I", str(SERVICE), "--config", str(config), *args],
        capture_output=True, text=True, timeout=timeout, env=merged)


def status(config):
    result = run(config, "status")
    return json.loads(result.stdout)


def children(state):
    return {child["name"]: child for child in state.get("children", [])}


def wait_for(predicate, timeout=20, interval=0.2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return None


def health(config):
    try:
        with urllib.request.urlopen(
                f"http://{config['host']}:{config['port']}/health",
                timeout=2) as response:
            return json.loads(response.read().decode())
    except Exception:
        return None


def main():
    work = Path(tempfile.mkdtemp(prefix="umminfer-service-test-"))
    flag = work / "health-down"
    port = free_port()
    fake_server = work / "fake_server.py"
    fake_server.write_text("#!/usr/bin/env python3\n" + FAKE_SERVER)
    os.chmod(fake_server, 0o755)
    bin_dir = work / "bin"
    bin_dir.mkdir()
    for kind in ("asr", "omni", "image", "tts"):
        stage = bin_dir / f"umm-native-stage-{kind}"
        stage.write_text("#!/usr/bin/env python3\n" + FAKE_STAGE)
        os.chmod(stage, 0o755)
    config = {
        "name": "test-service",
        "run_dir": str(work / "run"),
        "log_dir": str(work / "logs"),
        "ipc_dir": str(work / "ipc"),
        "host": "127.0.0.1",
        "port": port,
        "binary": str(fake_server),
        "model": "qwen3-native",
        "stage_binary_dir": str(bin_dir),
        "native_stages": [
            {"name": "asr", "model": "/models/asr", "gpu": 2},
            {"name": "omni", "model": "/models/omni", "gpu": 3,
             "args": ["--max-new-tokens", "96"]},
            {"name": "image", "model": "/models/image", "gpu": 4},
            {"name": "tts", "model": "/models/tts", "gpu": 5},
        ],
        "health_interval_s": 0.2,
        "health_failures": 3,
        "stop_grace_s": 2,
        "backoff_s": [0.2],
        "stable_s": 5,
        "health_path": "/health",
    }
    config_path = work / "config.json"
    config_path.write_text(json.dumps(config, indent=2))

    # The native configuration needs no Python worker or SGLang settings.
    native_supervisor = umminfer_service.Supervisor(config)
    native_children = {child.name: child for child in native_supervisor.children}
    check("native supervisor builds four stages and server",
          set(native_children) == {"server", "asr", "omni", "image", "tts"})
    check("native server exposes all output stages",
          "--image-entry" in native_children["server"].argv and
          "--tts-entry" in native_children["server"].argv)
    check("native stages use binaries not Python workers",
          native_children["image"].argv[0].endswith("umm-native-stage-image") and
          "--max-new-tokens" in native_children["omni"].argv and
          all("python" not in " ".join(child.argv).lower()
              for child in native_children.values()))
    check("native config loads without Python or SGLang keys",
          umminfer_service.load_config(config_path)["native_stages"][2]["gpu"] == 4)

    env = {"FAKE_HEALTH_FLAG": str(flag), "FAKE_BIG_LOG": "1"}
    decoy = None
    started = False
    try:
        result = run(config_path, "start", env=env)
        check("start command exits 0", result.returncode == 0,
              result.stdout + result.stderr)
        started = result.returncode == 0

        state = wait_for(lambda: status(config_path)["state"]
                         if status(config_path)["running"] else None)
        live = wait_for(lambda: all(child["alive"] for child in
                                    children(status(config_path)["state"]).values())
                        and len(children(status(config_path)["state"])) == 5)
        check("five children alive", bool(live))
        check("health ok", wait_for(lambda: (health(config) or {}).get("status") == "ok"))

        first = children(status(config_path)["state"])
        check("server pid recorded with start time",
              first["server"]["pid"] and first["server"]["starttime"])
        old_asr = first["asr"]["pid"]
        if not old_asr:
            raise SystemExit("FAIL no ASR pid recorded")
        os.kill(old_asr, signal.SIGKILL)
        restarted = wait_for(lambda: (
            lambda current: current["asr"]["alive"] and current["asr"]["pid"] != old_asr
            and current["asr"]["restarts"] >= 1)(children(status(config_path)["state"])))
        check("killed stage restarts", bool(restarted))
        check("killed stage pid is gone", not umminfer_service.pid_alive(old_asr))

        first = children(status(config_path)["state"])
        old_server, old_omni = first["server"]["pid"], first["omni"]["pid"]
        os.kill(old_server, signal.SIGKILL)
        cascaded = wait_for(lambda: (
            lambda current: current["server"]["alive"]
            and current["server"]["pid"] != old_server
            and current["omni"]["alive"]
            and current["omni"]["pid"] != old_omni)(children(status(config_path)["state"])))
        check("server crash restarts server and stages", bool(cascaded))

        # Every stage must come back after a server restart, not only asr/omni.
        after = wait_for(lambda: (
            lambda current: all(current[name]["alive"] for name in
                                ("asr", "omni", "image", "tts"))
            and current["image"]["pid"] != first["image"]["pid"]
            and current["tts"]["pid"] != first["tts"]["pid"])(
                children(status(config_path)["state"])))
        check("server crash restarts image and tts stages", bool(after))

        first = children(status(config_path)["state"])
        old_server = first["server"]["pid"]
        flag.write_text("down")
        recovered = wait_for(lambda: (
            lambda current: current["server"]["alive"]
            and current["server"]["pid"] != old_server)(children(status(config_path)["state"])),
            timeout=30)
        check("failed health check restarts server", bool(recovered))
        flag.unlink()
        check("health returns after restart",
              wait_for(lambda: (health(config) or {}).get("status") == "ok"))

        server_log = work / "logs" / "server.log"
        worker_log = work / "logs" / "asr.log"
        check("server log capped at 64 KiB", server_log.stat().st_size <= CAP,
              str(server_log.stat().st_size))
        check("stage log capped at 64 KiB", worker_log.stat().st_size <= CAP,
              str(worker_log.stat().st_size))
        check("log keeps the newest tail",
              server_log.read_bytes().endswith(b"\n")
              or len(server_log.read_bytes()) == CAP)
        logs = run(config_path, "logs", "--name", "server", "--bytes", "999999")
        check("logs command caps output", len(logs.stdout.encode()) <= CAP,
              str(len(logs.stdout.encode())))

        pidfile = work / "run" / "supervisor.pid"
        saved_pidfile = pidfile.read_text()
        decoy = subprocess.Popen([sys.executable, "-c",
                                  "import time; time.sleep(60)"])
        real_start = umminfer_service.process_starttime(decoy.pid)
        tampered = f"{real_start}X"
        pidfile.write_text(json.dumps(
            {"pid": decoy.pid, "starttime": tampered}))
        (work / "run" / "state.json").write_text(json.dumps(
            {"children": [{"name": "decoy", "pid": decoy.pid,
                           "starttime": tampered}]}))
        result = run(config_path, "stop")
        check("stop with changed start time signals nothing",
              decoy.poll() is None and result.returncode == 0,
              result.stdout + result.stderr)
        decoy.kill()
        decoy.wait()
        decoy = None
        pidfile.write_text(saved_pidfile)
        check("live supervisor still reported after tampered stop",
              status(config_path)["running"] is True)

        before = children(status(config_path)["state"])
        owned = [child["pid"] for child in before.values() if child["pid"]]
        result = run(config_path, "stop")
        time.sleep(0.5)
        check("stop exits 0", result.returncode == 0, result.stdout + result.stderr)
        check("stop leaves no owned process",
              all(not umminfer_service.pid_alive(pid) for pid in owned),
              str([pid for pid in owned if umminfer_service.pid_alive(pid)]))
        check("pid file removed",
              not (work / "run" / "supervisor.pid").exists())
        started = False

        result = run(config_path, "start", env=env)
        started = result.returncode == 0
        check("start creates pid file",
              bool(json.loads((work / "run" / "supervisor.pid").read_text())["starttime"]))
        second = run(config_path, "start", env=env)
        check("second start refuses", second.returncode != 0)

        bad = dict(config)
        bad["host"] = "0.0.0.0"
        bad_path = work / "bad.json"
        bad_path.write_text(json.dumps(bad))
        result = run(bad_path, "start", env=env)
        check("0.0.0.0 is refused", result.returncode != 0
              and "refusing to bind" in (result.stdout + result.stderr))
    finally:
        if decoy is not None:
            decoy.kill()
            decoy.wait()
        if started:
            run(config_path, "stop")
        if FAILURES and os.environ.get("UMMINFER_TEST_KEEP") != "1":
            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f"EVIDENCE {work}")
    print(f"RESULT {'PASS' if not FAILURES else 'FAIL'} failures={len(FAILURES)}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
