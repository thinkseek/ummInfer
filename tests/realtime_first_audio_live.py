#!/usr/bin/env python3
"""Controlled SM90 ASR-final to first-audio gate with native-boundary observers.

Only physical GPUs 0 (ASR) and 1 (Qwen3.8 + TTS) are used. All server/stage
processes are owned by this driver. Physical playback, WAN and listening quality
are not covered. Acceptance is delegated to a fresh stdlib-only raw verifier.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("umminfer_realtime_first_audio_base", ROOT / "tests/realtime_network_live.py")
first = load_module("umminfer_realtime_first_audio_driver_contract",
                    ROOT / "python/umminfer_realtime_first_audio.py")
require = base.require
sha = base.sha


class ForwardObserver:
    """Byte-preserving PUSH/PULL forwarding with original monotonic timestamps."""
    def __init__(self, channel, public_endpoint, native_endpoint, output_path):
        import zmq
        self.channel = channel
        self.context = zmq.Context()
        self.public = self.context.socket(zmq.PULL)
        self.public.bind(public_endpoint)
        self.native = self.context.socket(zmq.PUSH)
        self.native.connect(native_endpoint)
        self.output = open(output_path, "xb", buffering=0)
        self.rows = 0
        self.error = None
        self.stop_event = base.threading.Event()
        self.thread = base.threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        try:
            while not self.stop_event.is_set() or self.public.poll(0):
                if not self.public.poll(100):
                    continue
                frames = self.public.recv_multipart()
                observed = time.monotonic_ns()
                self.native.send_multipart(frames)
                forwarded = time.monotonic_ns()
                for frame in frames:
                    row = {"index": self.rows, "channel": self.channel,
                           "observed_monotonic_ns": observed,
                           "forwarded_monotonic_ns": forwarded,
                           "raw_sha256": sha(frame),
                           "raw_base64": base.base64.b64encode(frame).decode()}
                    self.rows += 1
                    self.output.write((json.dumps(row, sort_keys=True) + "\n").encode())
        except Exception as error:
            self.error = error

    def close(self):
        self.stop_event.set()
        self.thread.join(10)
        require(not self.thread.is_alive(), f"observer did not drain: {self.channel}")
        self.output.close()
        self.public.close(0)
        self.native.close(0)
        self.context.term()
        if self.error:
            raise self.error


class PausableRawWebSocket(base.RawWebSocket):
    """Pause consumption only after recording the first raw audio frame."""
    pause_after_first_audio_ms = 0

    def __init__(self, *args):
        self.pause_done = False
        super().__init__(*args)

    def record(self, direction, raw):
        row = super().record(direction, raw)
        pause_ms = self.pause_after_first_audio_ms
        if (direction == "rx" and pause_ms > 0 and not self.pause_done and
                b"audio.delta" in raw and b'"type"' in raw):
            self.pause_done = True
            time.sleep(pause_ms / 1000.0)
        return row


def start_stack(args, arm, arm_root, log_root):
    ports = base.free_ports(2)
    # Unix-domain ZMQ paths have a 107-byte limit. Long arm names belong in the
    # evidence/log hierarchy, not in the socket path.
    ipc = Path(tempfile.mkdtemp(prefix="umm-fa-ipc-", dir="/tmp"))
    shm = Path(tempfile.mkdtemp(prefix="umm-fa-shm-", dir="/tmp"))
    public = {name: f"ipc://{ipc}/{name}-public" for name in ("asr", "vlm", "tts", "completion")}
    native = {name: f"ipc://{ipc}/{name}-native" for name in ("asr", "vlm", "tts", "completion")}
    abort = f"ipc://{ipc}/abort"
    (arm_root / "observers").mkdir(parents=True, exist_ok=False)
    observers = {}
    procs = {}
    try:
        for channel, name in (("asr-submit", "asr"), ("vlm-submit", "vlm"),
                              ("tts-submit", "tts"), ("completion", "completion")):
            observers[channel] = ForwardObserver(channel, public[name], native[name],
                                                  arm_root / "observers" / f"{channel}.jsonl")
        procs["asr"] = base.OwnedProc("asr", [
            args.bin_dir / "umm-native-stage-asr", "--kind", "asr", "--name", "asr",
            "--model", args.asr_model, "--gpu", 0, "--recv-endpoint", native["asr"],
            "--coordinator-endpoint", public["completion"], "--abort-endpoint", abort],
            log_root / "asr.log")
        old_prefix = os.environ.get("UMM_QWEN38_PREFIX_CACHE")
        os.environ["UMM_QWEN38_PREFIX_CACHE"] = "1" if arm["prefix_cache"] else "0"
        try:
            procs["vlm"] = base.OwnedProc("vlm", [
                args.bin_dir / "umm-native-stage-qwen38", "--kind", "qwen38", "--name", "vlm",
                "--model", args.vlm_model, "--gpu", 1, "--recv-endpoint", native["vlm"],
                "--coordinator-endpoint", public["completion"], "--abort-endpoint", abort,
                "--max-new-tokens", args.vlm_max_tokens], log_root / "vlm.log")
        finally:
            if old_prefix is None:
                os.environ.pop("UMM_QWEN38_PREFIX_CACHE", None)
            else:
                os.environ["UMM_QWEN38_PREFIX_CACHE"] = old_prefix
        procs["tts"] = base.OwnedProc("tts", [
            args.bin_dir / "umm-native-stage-tts", "--kind", "tts", "--name", "tts",
            "--model", args.tts_model, "--gpu", 1, "--recv-endpoint", native["tts"],
            "--coordinator-endpoint", public["completion"], "--abort-endpoint", abort,
            "--max-frames", args.tts_max_frames], log_root / "tts.log")
        procs["server"] = base.OwnedProc("server", [
            args.bin_dir / "umm-infer", "serve", "--host", "127.0.0.1",
            "--port", ports[0], "--realtime-port", ports[1], "--model", "qwen3-unified",
            "--entry", "asr", "--entry-endpoint", public["asr"],
            "--vlm-entry", "vlm", "--vlm-entry-endpoint", public["vlm"],
            "--tts-entry", "tts", "--tts-entry-endpoint", public["tts"],
            "--completion", native["completion"], "--abort", abort,
            "--session-history", "--max-in-flight", 32, "--ipc-dir", shm], log_root / "server.log")
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            for observer in observers.values():
                require(observer.error is None, f"observer failed: {observer.error}")
            for name, proc in procs.items():
                require(proc.proc.poll() is None, f"{name} exited during model load")
            ready = all(marker in procs[name].log.text() for name, marker in {
                "asr": "ready kind=asr", "vlm": "ready kind=qwen38", "tts": "ready kind=tts"}.items())
            if ready:
                try:
                    import urllib.request
                    with urllib.request.urlopen(f"http://127.0.0.1:{ports[0]}/ready", timeout=1) as reply:
                        if json.load(reply).get("status") == "ready":
                            return {"ports": ports, "procs": procs, "observers": observers,
                                    "isolated": (ipc, shm)}
                except Exception:
                    pass
            time.sleep(0.25)
        raise AssertionError("task-owned real stages did not become ready")
    except Exception:
        if procs:
            base.stop_stack(procs)
        for observer in observers.values():
            observer.close()
        for path in (ipc, shm):
            shutil.rmtree(path, ignore_errors=True)
        raise


def run_turn(client, user, arm, schedule, session, arm_root):
    rank = schedule["start_order"].index(session)
    jitter = schedule["start_jitter_ms"][str(session)]
    time.sleep((rank * 3 + jitter) / 1000.0)
    turn = client.send_user(user)
    response_id, after = client.response_identity(turn["completed"]["event_index"] + 1)
    snapshot = client.response_snapshot(response_id, after, wait_terminal=True)
    require(snapshot["audio"] and snapshot["text"], "measured real response is empty")
    directory = arm_root / "pcm" / f"round-{schedule['round']}" / f"session-{session}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{response_id}.pcm").write_bytes(snapshot["pcm"])
    client.exercise_acks(snapshot, arm["final_ack_delay_ms"], final=True)
    return {"response_id": response_id, "connection": client.name, "round": schedule["round"],
            "session": session, "measured": schedule["measured"],
            "text_sha256": sha(snapshot["text"].encode()), "pcm_sha256": snapshot["pcm_sha256"]}


def run_regressions(ports, user, arm_root):
    """Delayed barge-in plus the retained echo/double-talk public PCM gate."""
    delayed = base.LiveClient(90, 0, ports[1], arm_root)
    try:
        turn = delayed.send_user(user)
        rid, after = delayed.response_identity(turn["completed"]["event_index"] + 1)
        snapshot = delayed.response_snapshot(rid, after, wait_terminal=True)
        cursor = delayed.exercise_acks(snapshot, 5, final=False)
        require(cursor > 0, "delayed barge-in needs a positive acknowledged cursor")
        next_turn = delayed.barge(snapshot, user, cursor, immediate=False)
        cancelled = next_turn["cancelled"]
        stale = delayed.ws.since(cancelled["event_index"] + 1, types=base.STALE_TYPES,
                                 response_id=rid)
        require(not stale, "delayed barge-in left stale output")
        next_id, next_after = delayed.response_identity(next_turn["completed"]["event_index"] + 1)
        next_snapshot = delayed.response_snapshot(next_id, next_after, wait_terminal=True)
        delayed.exercise_acks(next_snapshot, 5, final=True)
        delayed_record = {"response_id": rid, "cancelled_response_id": cancelled["event"]["response_id"],
                          "text_cursor": cancelled["event"]["text_cursor"],
                          "acknowledged_text_sha256": sha(cancelled["event"]["acknowledged_text"].encode()),
                          "history_recorded": cancelled["event"]["history_recorded"],
                          "stale_events": len(stale)}
    finally:
        delayed.close()
    echo_client = base.LiveClient(91, 0, ports[1], arm_root)
    try:
        echo = base.echo_and_double_talk(echo_client, user, arm_root)
        echo["connection"] = echo_client.name
    finally:
        echo_client.close()
    return {"delayed_barge": delayed_record, "echo_ab": echo}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--asr-model", type=Path, required=True)
    parser.add_argument("--vlm-model", type=Path, required=True)
    parser.add_argument("--tts-model", type=Path, required=True)
    parser.add_argument("--short-audio", type=Path, required=True)
    parser.add_argument("--long-audio", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--vlm-max-tokens", type=int, default=96)
    parser.add_argument("--tts-max-frames", type=int, default=192)
    args = parser.parse_args()
    paths = (args.contract, args.bin_dir, args.asr_model, args.vlm_model, args.tts_model,
             args.short_audio, args.long_audio, args.out, args.log_dir)
    require(all(path.is_absolute() for path in paths), "all paths must be absolute")
    args.out.mkdir(parents=True, exist_ok=False)
    args.log_dir.mkdir(parents=True, exist_ok=False)
    contract = first.load_contract(args.contract)
    plan = first.make_plan(contract)
    require(args.vlm_max_tokens == contract["topology"]["vlm"]["max_new_tokens"] and
            args.tts_max_frames == contract["topology"]["tts"]["max_frames"],
            "generation limits differ from preregistration")
    shutil.copyfile(args.contract, args.out / "preregistration.json")
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    (args.out / "fixtures").mkdir()
    fixture_pcm = {}
    for name, spec in contract["fixtures"].items():
        source = args.short_audio if name == "short" else args.long_audio
        require(base.hash_file(source) == spec["sha256"], f"audio fixture hash mismatch: {name}")
        shutil.copyfile(source, args.out / "fixtures" / spec["file"])
        pcm = base.echo.trim_digital_lead_in(base.echo.pcm_from_wav(source))
        require(len(pcm) >= 64000, f"audio fixture is too short: {name}")
        fixture_pcm[name] = pcm[:64000]

    run = {"schema": "umminfer-realtime-first-audio-live/v1", "source_commit": args.source_commit,
           "validation_class": "REAL_DIGITAL_LOOPBACK_NETWORK_AND_NATIVE_MODELS",
           "physical_microphone_speaker": "NOT_COVERED", "wan": "NOT_COVERED",
           "perceptual_listening_quality": "NOT_COVERED",
           "gpus": {"asr": 0, "vlm": 1, "tts": 1},
           "gpu_preflight": base.preflight(),
           "binaries": {name: base.hash_file(args.bin_dir / name) for name in
                        ("umm-infer", "umm-native-stage-asr", "umm-native-stage-qwen38",
                         "umm-native-stage-tts")},
           "harness_sha256": base.hash_file(Path(__file__)),
           "analyzer_sha256": base.hash_file(ROOT / "python/umminfer_realtime_first_audio.py"),
           "contract_file_sha256": base.hash_file(args.contract),
           "arm_runs": {}, "cleanup": {"owned_processes_exited": False, "selected_gpus_idle": False}}
    print("Hashing exact native model files", flush=True)
    run["models"] = base.hash_models(args, args.out)
    all_cleanup = []
    last_health = None
    try:
        for arm_index, arm in enumerate(plan["arms"]):
            print(f"ARM {arm['name']} sessions={arm['sessions']} rounds={arm['rounds']} "
                  f"prefix_cache={arm['prefix_cache']}", flush=True)
            arm_root = args.out / "arms" / arm["name"]
            log_root = args.log_dir / arm["name"]
            arm_root.mkdir(parents=True, exist_ok=False)
            log_root.mkdir(parents=True, exist_ok=False)
            info = {"gpu_launch_preflight": base.preflight(), "turns": []}
            run["arm_runs"][arm["name"]] = info
            stack = None
            sampler = None
            clients = []
            try:
                stack = start_stack(args, arm, arm_root, log_root)
                procs = stack["procs"]
                info["ports"] = {"http": stack["ports"][0], "websocket": stack["ports"][1]}
                info["owned_processes"] = [proc.identity() for proc in procs.values()]
                sampler = base.echo.ResourceSampler(procs, arm_root / "resources.jsonl")
                previous_raw = base.RawWebSocket
                PausableRawWebSocket.pause_after_first_audio_ms = arm["read_pause_after_first_audio_ms"]
                base.RawWebSocket = PausableRawWebSocket
                try:
                    persistent = {}
                    if arm["prefix_cache"]:
                        for session in range(arm["sessions"]):
                            client = base.LiveClient(session, 0, stack["ports"][1], arm_root)
                            persistent[session] = client
                            clients.append(client)
                    for schedule in arm["schedule"]:
                        round_clients = {}
                        if arm["prefix_cache"]:
                            round_clients = persistent
                        else:
                            for session in range(arm["sessions"]):
                                client = base.LiveClient(session, schedule["round"], stack["ports"][1], arm_root)
                                round_clients[session] = client
                                clients.append(client)
                        user = fixture_pcm[arm["fixture"]]
                        with ThreadPoolExecutor(max_workers=arm["sessions"]) as pool:
                            futures = [pool.submit(run_turn, round_clients[session], user, arm,
                                                   schedule, session, arm_root)
                                       for session in schedule["start_order"]]
                            rows = [future.result() for future in futures]
                        info["turns"].extend(rows)
                        if not arm["prefix_cache"]:
                            for client in round_clients.values():
                                client.close()
                    if arm_index + 1 == len(plan["arms"]):
                        run["regression"] = run_regressions(stack["ports"], fixture_pcm["short"], arm_root)
                    last_health = base.wait_health_zero(stack["ports"][0])
                    info["final_health"] = last_health
                    if sampler is not None:
                        info["resource_usage"] = sampler.finish()
                        sampler = None
                finally:
                    base.RawWebSocket = previous_raw
            except Exception as error:
                info["driver_error"] = str(error)
                if stack is not None:
                    info["log_tails"] = {name: proc.log.text()[-4000:]
                                         for name, proc in stack["procs"].items()}
                raise
            finally:
                for client in clients:
                    try:
                        client.close()
                    except Exception:
                        pass
                if sampler is not None:
                    info["resource_usage"] = sampler.finish()
                if stack is not None:
                    base.stop_stack(stack["procs"])
                    for observer in stack["observers"].values():
                        observer.close()
                    for name, proc in stack["procs"].items():
                        shutil.copyfile(proc.log.path, arm_root / f"{name}.log")
                    for path in stack["isolated"]:
                        shutil.rmtree(path, ignore_errors=True)
                    info["owned_processes_exited"] = all(
                        proc.proc.poll() is not None for proc in stack["procs"].values())
                try:
                    info["gpu_postflight"] = base.wait_postflight()
                    info["selected_gpus_idle"] = True
                except Exception as error:
                    info["cleanup_error"] = str(error)
                    info["selected_gpus_idle"] = False
                all_cleanup.append(info)
        run["final_health"] = last_health
        run["cleanup"] = {"owned_processes_exited": all(
                              row.get("owned_processes_exited") is True for row in all_cleanup),
                          "selected_gpus_idle": all(row.get("selected_gpus_idle") is True
                                                    for row in all_cleanup)}
        run["gpu_postflight"] = base.wait_postflight()
    except Exception as error:
        run["driver_error"] = str(error)
        raise
    finally:
        if all_cleanup:
            run["cleanup"] = {"owned_processes_exited": all(
                                  row.get("owned_processes_exited") is True for row in all_cleanup),
                              "selected_gpus_idle": all(row.get("selected_gpus_idle") is True
                                                        for row in all_cleanup)}
            if "driver_error" in run and run["cleanup"]["selected_gpus_idle"]:
                try:
                    run["gpu_postflight"] = base.wait_postflight()
                except Exception as error:
                    run["cleanup"]["selected_gpus_idle"] = False
                    run["cleanup"]["postflight_error"] = str(error)
        (args.out / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    if "driver_error" in run:
        raise AssertionError(run["driver_error"])
    print("Running fresh independent native/WS verifier", flush=True)
    import subprocess
    import sys
    subprocess.run([sys.executable, "-I", str(ROOT / "tests/realtime_first_audio_verify.py"),
                    str(args.out), "--write"], check=True)
    files = []
    for path in sorted(args.out.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            files.append(f"{base.hash_file(path)}  {path.relative_to(args.out)}\n")
    (args.out / "SHA256SUMS").write_text("".join(files))
    subprocess.run([sys.executable, "-I", str(ROOT / "tests/realtime_first_audio_verify.py"),
                    str(args.out)], check=True)
    report = json.loads((args.out / "report.json").read_text())
    print(f"REALTIME_FIRST_AUDIO PASS arms={len(plan['arms'])} "
          f"measured={report['counts']['measured_responses']} files={len(files)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
