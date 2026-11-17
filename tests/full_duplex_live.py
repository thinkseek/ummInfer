#!/usr/bin/env python3
"""Full-duplex L1 live evidence: real native ASR, real WebSocket, real audio.

Records:
  - chunk-by-chunk submission: first provisional transcript latency,
  - server VAD: silence does not trigger, a real pause commits the turn,
  - barge-in: a new utterance aborts the computing turn, with abort-to-exit ms.

Every timestamp is taken on the container clock, the same clock the server and
the stage log with.
"""

import argparse
import base64
import json
import os
from pathlib import Path
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import wave

from websockets.sync.client import connect

LOG_CAP = 65536


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def free_port_pair(host):
    while True:
        with socket.socket() as a, socket.socket() as b:
            a.bind((host, 0))
            pa = a.getsockname()[1]
            b.bind((host, 0))
            pb = b.getsockname()[1]
        if pa != pb:
            return pa, pb


def spawn(name, args, logs):
    proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    logs[name] = bytearray()

    def drain():
        while True:
            chunk = proc.stdout.readline(4096)
            if not chunk:
                return
            remaining = LOG_CAP - len(logs[name])
            if remaining > 0:
                logs[name].extend(chunk[:remaining])

    threading.Thread(target=drain, daemon=True).start()
    return proc


def wait_http(host, port, proc, timeout_s=300):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        require(proc.poll() is None, f"server exited {proc.returncode}")
        try:
            with socket.create_connection((host, port), timeout=0.5):
                pass
            return
        except OSError:
            time.sleep(0.2)
    raise AssertionError("server did not listen")


def pcm_from_wav(path):
    with wave.open(str(path), "rb") as wav:
        require((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) ==
                (1, 2, 16000), f"{path} is not PCM16 mono 16 kHz")
        return wav.readframes(wav.getnframes())


class Recorder:
    def __init__(self, ws):
        self.ws = ws
        self.events = queue.Queue()
        self.closed = False
        self.stop = threading.Event()
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while not self.stop.is_set():
            try:
                message = self.ws.recv(timeout=0.2)
            except TimeoutError:
                continue
            except Exception:
                self.closed = True
                return
            self.events.put((time.time(), json.loads(message)))

    def wait_for(self, predicate, timeout_s):
        spent = []
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                item = self.events.get(timeout=0.1)
            except queue.Empty:
                if self.closed:
                    break
                continue
            spent.append(item)
            if predicate(item[1]):
                return item, spent
        raise AssertionError(f"event not seen; got {[e for _, e in spent]}")


def append(ws, pcm):
    ws.send(json.dumps({"type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm).decode()}))


def stream(ws, pcm, chunk=3200, pace_s=0.1):
    for offset in range(0, len(pcm), chunk):
        append(ws, pcm[offset:offset + chunk])
        time.sleep(pace_s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--host", default="172.17.0.2")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    logs = {}
    processes = {}
    evidence = {}
    try:
        ipc = Path("/tmp/umm-duplex-ipc")
        ipc.mkdir(parents=True, exist_ok=True)
        for stale in ipc.iterdir():
            stale.unlink()
        port, ws_port = free_port_pair(args.host)
        entry = f"ipc://{ipc}/entry"
        completion = f"ipc://{ipc}/completion"
        abort = f"ipc://{ipc}/abort"
        processes["server"] = spawn("server", [
            str(args.bin_dir / "umm-infer"), "serve",
            "--host", args.host, "--port", str(port),
            "--realtime-port", str(ws_port), "--model", "qwen3-asr",
            "--entry", "asr", "--entry-endpoint", entry,
            "--completion", completion, "--abort", abort], logs)
        processes["stage"] = spawn("stage", [
            str(args.bin_dir / "umm-native-stage-asr"), "--kind", "asr",
            "--name", "asr", "--model", str(args.model), "--gpu", str(args.gpu),
            "--recv-endpoint", entry, "--coordinator-endpoint", completion,
            "--abort-endpoint", abort], logs)
        wait_http(args.host, port, processes["server"])
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            require(processes["stage"].poll() is None, "stage exited during load")
            if b"ready kind=asr" in bytes(logs["stage"]):
                break
            time.sleep(0.2)
        else:
            raise AssertionError("stage did not report ready")
        while time.monotonic() < deadline:
            if b'"status":"ready"' in subprocess.run(
                    ["curl", "-s", f"http://{args.host}:{port}/ready"],
                    capture_output=True).stdout:
                break
            time.sleep(0.2)
        evidence["startup"] = {"port": port, "ws_port": ws_port,
                               "gpu": args.gpu}
        print(f"PASS startup http={port} ws={ws_port}", flush=True)

        pcm = pcm_from_wav(args.audio)
        speech = pcm[:32000 * 2]  # first 2 s of real speech
        silence = b"\x00\x00" * 1600  # 100 ms
        url = f"ws://{args.host}:{ws_port}/v1/realtime?intent=transcription&model=qwen3-asr"

        with connect(url, compression=None, open_timeout=10, close_timeout=10,
                     ping_interval=None, max_size=1 << 20) as ws:
            rec = Recorder(ws)
            ws.send(json.dumps({"type": "session.update", "session": {
                "input_audio_format": "pcm16",
                "turn_detection": {"type": "server_vad", "silence_ms": 400},
                "max_new_tokens": 64}}))
            reply, _ = rec.wait_for(
                lambda e: e.get("type") == "transcription_session.updated", 10)
            evidence["session"] = reply[1]["session"]
            print(f"PASS session {reply[1]['session']}", flush=True)

            # Turn 1: leading silence must not start a turn.
            for _ in range(5):
                append(ws, silence)
                time.sleep(0.1)
            time.sleep(0.5)
            quiet = []
            while True:
                try:
                    quiet.append(rec.events.get_nowait())
                except queue.Empty:
                    break
            require(not any(e.get("type") == "input_audio_buffer.speech_started"
                            for _, e in quiet),
                    f"silence started a turn: {[e for _, e in quiet]}")
            evidence["vad_silence_no_trigger"] = {"events": len(quiet)}
            print("PASS vad_silence_no_trigger", flush=True)

            # Turn 1: real speech streamed in 100 ms chunks.
            t_speech = time.time()
            stream(ws, speech, pace_s=0.1)
            first, _ = rec.wait_for(
                lambda e: e.get("type") == "transcription.segment", 30)
            first_latency = (first[0] - t_speech) * 1000
            evidence["first_partial_ms"] = round(first_latency, 1)
            evidence["first_partial_text"] = first[1].get("text")
            print(f"PASS first_partial_ms={first_latency:.1f} "
                  f"text={first[1].get('text')!r}", flush=True)

            # End of turn: 400 ms of silence commits without transcription.done.
            t_pause = time.time()
            stream(ws, silence * 6, pace_s=0.1)
            done, _ = rec.wait_for(
                lambda e: e.get("type") == "transcription.completed", 60)
            evidence["turn1"] = {
                "commit_ms_from_pause": round((done[0] - t_pause) * 1000, 1),
                "text": done[1].get("text"),
                "segment_id": done[1].get("segment_id")}
            print(f"PASS vad_turn1_commit ms={(done[0]-t_pause)*1000:.1f} "
                  f"text={done[1].get('text')!r}", flush=True)

            # Turn 2 then barge-in: 8 s of speech make the turn-2 final take
            # long enough that turn 3 speech arrives while it computes.
            time.sleep(0.2)
            long_speech = pcm[:16000 * 8 * 2]
            stream(ws, long_speech, pace_s=0.1)
            # 400 ms of silence is the endpoint; the final is submitted on the
            # last of these chunks. The next bytes are already new speech.
            stream(ws, silence * 4, pace_s=0.1)
            t_barge = time.time()
            stream(ws, speech[:16000 * 2], pace_s=0.05)
            stream(ws, silence * 6, pace_s=0.1)
            started, _ = rec.wait_for(
                lambda e: e.get("type") == "input_audio_buffer.speech_started"
                and e.get("segment_id") == 2, 30)
            evidence["barge_in"] = {
                "speech_started_ms_from_new_input":
                    round((started[0] - t_barge) * 1000, 1)}

            # Nothing from turn 2 may complete after the barge-in. Turn 3 is
            # segment 2 and commits after its own pause.
            observed = []
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                try:
                    item = rec.events.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item[0] < t_barge:
                    continue
                observed.append(item)
                if (item[1].get("type") == "transcription.completed" and
                        item[1].get("segment_id") == 2):
                    break
            stale = [e for _, e in observed
                     if e.get("type") == "transcription.completed"
                     and e.get("segment_id") != 2]
            stale_segments = [e for _, e in observed
                              if e.get("type") == "transcription.segment"
                              and e.get("segment_id") == 1]
            evidence["barge_in"]["old_turn_segments_after_barge"] = stale_segments
            require(not stale_segments,
                    f"the aborted turn kept emitting: {stale_segments}")
            evidence["barge_in"]["events_after"] = [
                {"ms": round((t - t_barge) * 1000, 1), "type": e.get("type"),
                 "segment_id": e.get("segment_id"), "text": e.get("text")}
                for t, e in observed]
            evidence["barge_in"]["old_turn_completed_after_barge"] = stale
            print(f"PASS barge_in speech_started_ms="
                  f"{evidence['barge_in']['speech_started_ms_from_new_input']} "
                  f"stale={stale}", flush=True)
            rec.stop.set()

        time.sleep(0.5)
        server_log = bytes(logs["server"]).decode("utf-8", "replace")
        stage_log = bytes(logs["stage"]).decode("utf-8", "replace")
        evidence["server_lines"] = [l for l in server_log.splitlines()
                                    if "realtime" in l][-40:]
        evidence["stage_lines"] = [l for l in stage_log.splitlines()
                                   if "asr request" in l][-40:]
        interrupts = [l for l in evidence["server_lines"] if "interrupt" in l]
        aborts = [l for l in evidence["stage_lines"] if "aborted" in l]
        evidence["interrupt_lines"] = interrupts
        evidence["abort_lines"] = aborts
        barge = [l for l in interrupts if "reason=barge_in" in l]
        require(barge, f"no barge-in interrupt was issued: {interrupts}")
        if barge:
            epoch = int(re.search(r"epoch_ms=(\d+)", barge[-1]).group(1))
            exits = [int(re.search(r"exit_epoch_ms=(\d+)", l).group(1))
                     for l in aborts if "exit_epoch_ms=" in l]
            following = [value for value in exits if value >= epoch]
            evidence["barge_in"]["abort_epoch_ms"] = epoch
            evidence["barge_in"]["t_barge_ms"] = int(t_barge * 1000)
            evidence["barge_in"]["abort_delay_ms"] = epoch - int(t_barge * 1000)
            evidence["barge_in"]["new_input_epoch_ms"] = int(t_barge * 1000)
            evidence["barge_in"]["interrupt_after_new_input_ms"] = (
                epoch - int(t_barge * 1000))
            evidence["barge_in"]["stage_exit_epoch_ms"] = (
                min(following) if following else None)
            evidence["barge_in"]["abort_to_exit_ms"] = (
                min(following) - epoch if following else None)
        print(f"PASS evidence written interrupt_lines={len(interrupts)} "
              f"abort_lines={len(aborts)}", flush=True)
        (args.out / "live-duplex-evidence.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        (args.out / "server.log").write_bytes(bytes(logs["server"]))
        (args.out / "stage.log").write_bytes(bytes(logs["stage"]))
    finally:
        for proc in (processes.get("stage"), processes.get("server")):
            if proc is not None and proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for proc in processes.values():
            try:
                proc.wait(20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(5)


if __name__ == "__main__":
    main()
