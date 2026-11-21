#!/usr/bin/env python3
"""Full-duplex L2 live evidence: streaming TTS, sentence pipeline, prefix cache.

Every number is measured on the container clock with real models, a real
WebSocket, and the real stages. Three modes:

  realtime   N spoken turns over one WebSocket. Records the latency from the
             end of the user's speech to the first audio.delta, plus the
             endpoint and text-delta milestones.
  pipeline   two HTTP requests with the same prompt: audio.stream=true (one
             sentence leg per sentence) and audio.stream=false (one leg after
             the VLM answer). Records the first-audio time relative to the
             last VLM delta.
  prefix     a two-turn session with session_history enabled, once with the
             omni prefix cache on and once with it off. Compares the answers
             byte for byte and the stage-reported prefill_ms.

Exit code 0 only when every mode it ran passed its own checks.
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
import urllib.request

from websockets.sync.client import connect

LOG_CAP = 262144


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def percentile(values, pct):
    ordered = sorted(values)
    if not ordered:
        raise AssertionError("no samples")
    k = max(0, min(len(ordered) - 1, int(round(pct / 100.0 * (len(ordered) - 1)))))
    return ordered[k]


class CappedLog:
    def __init__(self, path):
        self.path = Path(path)
        self.data = bytearray()
        self.lock = threading.Lock()
        self.thread = None

    def start(self, proc):
        def drain():
            with open(self.path, "wb") as out:
                while True:
                    chunk = proc.stdout.readline(4096)
                    if not chunk:
                        return
                    with self.lock:
                        if len(self.data) + len(chunk) <= LOG_CAP:
                            self.data.extend(chunk)
                    out.write(chunk)
                    out.flush()

        self.thread = threading.Thread(target=drain, daemon=True)
        self.thread.start()

    def text(self):
        with self.lock:
            return bytes(self.data).decode("utf-8", "replace")

    def wait(self):
        if self.thread:
            self.thread.join(5)


class Proc:
    def __init__(self, name, args, log_path, env=None):
        self.name = name
        self.log = CappedLog(log_path)
        self.proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, env=env)
        self.log.start(self.proc)


def free_ports(count):
    ports = []
    for _ in range(count):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            ports.append(sock.getsockname()[1])
    return ports


def wait_port(port, proc, timeout_s=300, name="process"):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        require(proc.poll() is None, f"{name} exited")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise AssertionError(f"{name} did not listen")


def wait_ready(port, timeout_s=300):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/ready",
                                        timeout=1) as reply:
                body = json.load(reply)
                if body.get("status") == "ready":
                    return
        except Exception:
            pass
        time.sleep(0.2)
    raise AssertionError("server never became ready")


def pcm_from_wav(path):
    import wave
    with wave.open(str(path), "rb") as wav:
        require((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) ==
                (1, 2, 16000), f"{path} is not PCM16 mono 16 kHz")
        return wav.readframes(wav.getnframes())


def start_stack(args, tag, cache_on, ports, logs_dir):
    """Starts three stages and one server. Returns (procs, base_url, ws_port)."""
    http_port, ws_port = ports
    ipc = Path(logs_dir) / f"ipc-{tag}"
    ipc.mkdir(parents=True, exist_ok=True)
    for stale in ipc.iterdir():
        stale.unlink()
    entry = f"ipc://{ipc}/entry"
    vlm = f"ipc://{ipc}/vlm"
    tts = f"ipc://{ipc}/tts"
    completion = f"ipc://{ipc}/completion"
    abort = f"ipc://{ipc}/abort"
    shm = os.path.join(logs_dir, f"shm-{tag}")
    Path(shm).mkdir(parents=True, exist_ok=True)

    procs = {}
    procs["asr"] = Proc("asr", [
        str(Path(args.bin_dir) / "umm-native-stage-asr"), "--kind", "asr",
        "--name", "asr", "--model", str(args.asr_model),
        "--gpu", str(args.gpu_asr), "--recv-endpoint", entry,
        "--coordinator-endpoint", completion, "--abort-endpoint", abort],
        Path(logs_dir) / f"asr-{tag}.log")
    omni_env = dict(os.environ)
    omni_env["UMM_OMNI_PREFIX_CACHE"] = "1" if cache_on else "0"
    procs["vlm"] = Proc("vlm", [
        str(Path(args.bin_dir) / "umm-native-stage-omni"), "--kind", "omni",
        "--name", "vlm", "--model", str(args.omni_model),
        "--gpu", str(args.gpu_vlm), "--recv-endpoint", vlm,
        "--coordinator-endpoint", completion, "--abort-endpoint", abort,
        "--max-new-tokens", str(args.vlm_max_tokens)],
        Path(logs_dir) / f"vlm-{tag}.log", env=omni_env)
    procs["tts"] = Proc("tts", [
        str(Path(args.bin_dir) / "umm-native-stage-tts"), "--kind", "tts",
        "--name", "tts", "--model", str(args.tts_model),
        "--gpu", str(args.gpu_tts), "--recv-endpoint", tts,
        "--coordinator-endpoint", completion, "--abort-endpoint", abort,
        "--max-frames", str(args.tts_max_frames)],
        Path(logs_dir) / f"tts-{tag}.log")
    server = Proc("server", [
        str(Path(args.bin_dir) / "umm-infer"), "serve",
        "--host", "127.0.0.1", "--port", str(http_port),
        "--realtime-port", str(ws_port), "--model", "qwen3-unified",
        "--entry", "asr", "--entry-endpoint", entry,
        "--vlm-entry", "vlm", "--vlm-entry-endpoint", vlm,
        "--tts-entry", "tts", "--tts-entry-endpoint", tts,
        "--completion", completion, "--abort", abort, "--session-history",
        "--ipc-dir", shm], Path(logs_dir) / f"server-{tag}.log")
    procs["server"] = server
    wait_port(http_port, server.proc, 600, "server")
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        for name in ("asr", "vlm", "tts"):
            require(procs[name].proc.poll() is None, f"{name} exited during load")
        markers = {"asr": "ready kind=asr", "vlm": "ready kind=omni",
                   "tts": "ready kind=tts"}
        if all(marker in procs[name].log.text()
               for name, marker in markers.items()):
            break
        time.sleep(0.3)
    wait_ready(http_port, 300)
    return procs, http_port, ws_port


def stop_stack(procs):
    for proc in reversed(list(procs.values())):
        if proc.proc.poll() is None:
            proc.proc.send_signal(signal.SIGTERM)
    for proc in procs.values():
        try:
            proc.proc.wait(20)
        except subprocess.TimeoutExpired:
            proc.proc.kill()
            proc.proc.wait(5)
        proc.log.wait()


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
        raise AssertionError(f"event not seen; got {[e for _, e in spent][-8:]}")


def append(ws, pcm):
    ws.send(json.dumps({"type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm).decode()}))


def stream(ws, pcm, chunk=3200, pace_s=0.1):
    for offset in range(0, len(pcm), chunk):
        append(ws, pcm[offset:offset + chunk])
        time.sleep(pace_s)


def run_realtime(args, http_port, ws_port, pcm, evidence):
    silence = b"\x00\x00" * 1600  # 100 ms
    speech = pcm[:32000 * 2]      # 2 s of real speech
    url = (f"ws://127.0.0.1:{ws_port}/v1/realtime?"
           f"intent=transcription&model=qwen3-unified")
    latencies = []
    turns = []
    with connect(url, compression=None, open_timeout=10, close_timeout=10,
                 ping_interval=None, max_size=1 << 22) as ws:
        rec = Recorder(ws)
        ws.send(json.dumps({"type": "session.update", "session": {
            "input_audio_format": "pcm16",
            "turn_detection": {"type": "server_vad", "silence_ms": 400},
            "max_new_tokens": 128,
            "audio_output": {"voice": "vivian"}}}))
        reply, _ = rec.wait_for(
            lambda e: e.get("type") == "transcription_session.updated", 20)
        evidence["session"] = reply[1]["session"]
        print(f"PASS session {json.dumps(reply[1]['session'])}", flush=True)
        for turn in range(args.turns):
            if turn > 0:
                time.sleep(0.2)
            stream(ws, speech, pace_s=0.1)
            t_end = time.time()
            # The endpoint fires after 400 ms of silence; send 600 ms.
            for _ in range(6):
                append(ws, silence)
                time.sleep(0.1)
            first_audio, spent = rec.wait_for(
                lambda e: e.get("type") == "audio.delta", 120)
            first_text = next((item for item in spent
                               if item[1].get("type") == "response.text.delta"),
                              None)
            done, spent2 = rec.wait_for(
                lambda e: e.get("type") == "response.completed", 180)
            latency = (first_audio[0] - t_end) * 1000
            latencies.append(latency)
            record = {
                "turn": turn,
                "end_to_first_audio_ms": round(latency, 1),
                "end_to_first_text_ms": round((first_text[0] - t_end) * 1000, 1)
                if first_text else None,
                "first_audio_base64_chars": len(first_audio[1].get("audio", "")),
                "turn_total_ms": round((done[0] - t_end) * 1000, 1),
                "segment_id": first_audio[1].get("segment_id"),
            }
            turns.append(record)
            print(f"PASS turn={turn} end_to_first_audio_ms="
                  f"{record['end_to_first_audio_ms']}", flush=True)
        rec.stop.set()
    evidence["turns"] = turns
    evidence["first_audio_ms"] = {
        "samples": len(latencies),
        "p50": round(percentile(latencies, 50), 1),
        "p95": round(percentile(latencies, 95), 1),
        "min": round(min(latencies), 1),
        "max": round(max(latencies), 1),
    }
    return evidence


def sse_request(port, body, timeout_s=300):
    """Streams one chat completion and returns event records with times."""
    data = json.dumps(body).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    events = []
    with urllib.request.urlopen(request, timeout=timeout_s) as reply:
        for raw in reply:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            events.append((time.time(), json.loads(payload)))
    return events


def unified_kind(event):
    return event.get("unified", {}).get("kind")


def run_pipeline(args, http_port, evidence):
    prompt = args.pipeline_prompt
    results = {}
    for label, stream_audio in (("sentence_stream", True), ("single_leg", False)):
        body = {
            "model": "qwen3-unified",
            "stream": True,
            "messages": [{"role": "user", "content": prompt}],
            "modalities": ["text", "audio"],
            "audio": {"voice": "vivian", "format": "wav",
                      "stream": stream_audio},
            "max_tokens": 160,
        }
        t0 = time.time()
        events = sse_request(http_port, body)
        first_audio = next((t for t, e in events
                            if unified_kind(e) == "artifact"), None)
        last_text = next((t for t, e in reversed(events)
                          if unified_kind(e) == "vlm_delta"), None)
        first_text = next((t for t, e in events
                           if unified_kind(e) == "vlm_delta"), None)
        complete = next((t for t, e in events
                         if unified_kind(e) == "complete"), None)
        require(first_audio is not None, f"{label}: no audio artifact")
        require(complete is not None, f"{label}: no terminal")
        chunks = sum(1 for _, e in events if unified_kind(e) == "artifact")
        results[label] = {
            "submit_to_first_audio_ms": round((first_audio - t0) * 1000, 1),
            "first_text_after_submit_ms":
                round((first_text - t0) * 1000, 1) if first_text else None,
            "last_text_to_first_audio_ms":
                round((first_audio - last_text) * 1000, 1) if last_text else None,
            "first_audio_before_last_text": bool(last_text and first_audio < last_text),
            "audio_chunks": chunks,
            "total_ms": round((complete - t0) * 1000, 1),
        }
        print(f"PASS pipeline {label} {json.dumps(results[label])}", flush=True)
    advance = (results["single_leg"]["last_text_to_first_audio_ms"] -
               results["sentence_stream"]["last_text_to_first_audio_ms"])
    results["first_audio_advance_ms"] = round(advance, 1)
    evidence["pipeline"] = results
    print(f"PASS pipeline advance_ms={results['first_audio_advance_ms']}",
          flush=True)
    return evidence


def http_json(port, path, body):
    data = json.dumps(body).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=300) as reply:
        return json.load(reply)


def run_prefix(args, evidence):
    """Two-turn session, cache on and off, compared byte for byte."""
    runs = {}
    for cache_on in (True, False):
        tag = "cache_on" if cache_on else "cache_off"
        ports = free_ports(2)
        procs, http_port, _ = start_stack(args, tag, cache_on, ports,
                                          args.log_dir)
        try:
            answers = []
            for turn in range(2):
                body = {
                    "model": "qwen3-unified",
                    "stream": False,
                    "messages": [{"role": "user", "content": args.prefix_prompts[turn]}],
                    "session_id": "l2-prefix-session",
                    "session_history": True,
                    "max_tokens": 96,
                }
                reply = http_json(http_port, "/v1/chat/completions", body)
                answers.append(reply["choices"][0]["message"]["content"])
            time.sleep(0.5)
            log = procs["vlm"].log.text()
            lines = [l for l in log.splitlines() if "omni request" in l]
            runs[tag] = {
                "answers": answers,
                "vlm_lines": lines,
                "prefill_ms": [float(re.search(r"prefill_ms=([0-9.]+)", l).group(1))
                               for l in lines],
                "prefix_reused": [int(re.search(r"prefix_reused=(\d+)", l).group(1))
                                  for l in lines],
            }
        finally:
            stop_stack(procs)
        print(f"PASS prefix {tag} prefill_ms={runs[tag]['prefill_ms']} "
              f"reused={runs[tag]['prefix_reused']}", flush=True)
    same = runs["cache_on"]["answers"] == runs["cache_off"]["answers"]
    evidence["prefix"] = runs
    evidence["prefix"]["answers_identical"] = same
    reused = runs["cache_on"]["prefix_reused"][1]
    require(reused > 0, "the second turn did not reuse a cached prefix")
    require(same, "cached and uncached answers differ")
    off = runs["cache_off"]["prefill_ms"][1]
    on = runs["cache_on"]["prefill_ms"][1]
    evidence["prefix"]["second_turn_prefill_ms"] = {"cache_off": off, "cache_on": on}
    evidence["prefix"]["prefill_reduction_pct"] = round((1 - on / off) * 100, 1)
    print(f"PASS prefix identical answers, prefill {off}->{on} ms", flush=True)
    return evidence


def dump_evidence(args, evidence):
    (args.out / "l2-evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--asr-model", type=Path, required=True)
    parser.add_argument("--omni-model", type=Path, required=True)
    parser.add_argument("--tts-model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--gpu-asr", type=int, default=2)
    parser.add_argument("--gpu-vlm", type=int, default=4)
    parser.add_argument("--gpu-tts", type=int, default=5)
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--vlm-max-tokens", type=int, default=96)
    # A pathological one-shot TTS leg can ramble; cap every leg so the
    # baseline run terminates. 512 frames is 41 s of audio.
    parser.add_argument("--tts-max-frames", type=int, default=512)
    parser.add_argument("--mode", default="all",
                        choices=["all", "realtime", "pipeline", "prefix"])
    parser.add_argument("--pipeline-prompt",
                        default="请用一句话介绍杭州，然后再说一句关于西湖的话。")
    parser.add_argument("--prefix-prompts", nargs=2, default=[
        "请阅读下面这段文字并记住其中的数字。杭州是一座历史悠久的城市，"
        "西湖位于城市西侧，湖面面积约六平方公里，周边有苏堤、白堤和许多园林。"
        "每年春天，游客沿湖步行，从断桥走到雷峰塔大约需要两个小时。"
        "请记住数字四十二。",
        "根据上面这段文字，请只回答我让你记住的那个数字。"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    evidence = {"bin_dir": str(args.bin_dir), "gpus": {
        "asr": args.gpu_asr, "vlm": args.gpu_vlm, "tts": args.gpu_tts}}
    try:
        if args.mode in ("all", "realtime"):
            ports = free_ports(2)
            procs, http_port, ws_port = start_stack(
                args, "realtime", True, ports, args.log_dir)
            try:
                pcm = pcm_from_wav(args.audio)
                run_realtime(args, http_port, ws_port, pcm, evidence)
                dump_evidence(args, evidence)
            finally:
                stop_stack(procs)
        if args.mode in ("all", "pipeline"):
            ports = free_ports(2)
            procs, http_port, _ = start_stack(
                args, "pipeline", True, ports, args.log_dir)
            try:
                run_pipeline(args, http_port, evidence)
                dump_evidence(args, evidence)
            finally:
                stop_stack(procs)
        if args.mode in ("all", "prefix"):
            run_prefix(args, evidence)
            dump_evidence(args, evidence)
        (args.out / "l2-evidence.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        print(f"PASS evidence written {args.out / 'l2-evidence.json'}")
    finally:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
