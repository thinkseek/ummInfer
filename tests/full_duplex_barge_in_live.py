#!/usr/bin/env python3
"""Real-model Stage B barge-in playback-history validation.

Three real stages (ASR, VLM, TTS) run on independently selected idle GPUs. The
client uses timed virtual PCM playback: an acknowledgement is sent only after
the corresponding PCM duration has been consumed, never merely because an
audio.delta arrived. No audio device is required.

The harness checks a partial-TTS barge-in, exact acknowledgement metadata,
absence of stale output, a following VLM turn, a second cancellation, and
in_flight=0. Exact role/content prompt assertions live in the deterministic C++
contract suite; this live run validates the same wire contract with real models.
"""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import wave

from websockets.sync.client import connect

LOG_CAP = 262144
STALE_TYPES = {
    "audio.delta", "audio.done", "response.text.delta",
    "response.playback_ack", "response.completed", "response.failed",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return {"sha256": digest.hexdigest(), "bytes": size}


def epoch_ms():
    return time.monotonic() * 1000.0


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
    def __init__(self, name, args, log_path):
        self.name = name
        self.args = [str(arg) for arg in args]
        self.log = CappedLog(log_path)
        self.proc = subprocess.Popen(
            self.args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=dict(os.environ))
        self.log.start(self.proc)


def free_ports(count):
    ports = []
    sockets = []
    try:
        for _ in range(count):
            sock = socket.socket()
            sockets.append(sock)
            sock.bind(("127.0.0.1", 0))
            ports.append(sock.getsockname()[1])
    finally:
        for sock in sockets:
            sock.close()
    return ports


def wait_port(port, proc, timeout_s=600):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        require(proc.poll() is None, "server exited before listening")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise AssertionError("server did not listen")


def wait_ready(port, procs, timeout_s=600):
    markers = {"asr": "ready kind=asr", "tts": "ready kind=tts"}
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for name, proc in procs.items():
            require(proc.proc.poll() is None, f"{name} exited during model load")
        vlm_ready = ("ready kind=qwen38" in procs["vlm"].log.text() or
                     "ready kind=omni" in procs["vlm"].log.text())
        if (vlm_ready and all(marker in procs[name].log.text()
                              for name, marker in markers.items())):
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/ready", timeout=1) as reply:
                    if json.load(reply).get("status") == "ready":
                        return
            except Exception:
                pass
        time.sleep(0.3)
    raise AssertionError("real stages or server never became ready")


def health(port):
    with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=2) as reply:
        return json.load(reply)


def wait_in_flight_zero(port, timeout_s=5):
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        last = health(port)
        if last.get("in_flight") == 0:
            return last
        time.sleep(0.05)
    raise AssertionError(f"in_flight did not return to zero: {last}")


def preflight_gpus(gpus):
    require(all(gpu in (0, 1, 7) for gpu in gpus),
            "only explicitly assigned idle GPUs 0, 1 or 7 are allowed")
    evidence = {}
    # The established default uses three distinct devices. A single-device test
    # may intentionally share serial stages on one authorized idle H20.
    for gpu in sorted(set(gpus)):
        rows = subprocess.check_output([
            "nvidia-smi", "-i", str(gpu),
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True).strip()
        fields = [field.strip() for field in rows.split(",")]
        require(len(fields) == 3, f"unexpected nvidia-smi row for GPU {gpu}: {rows}")
        used_mib, utilization = int(fields[1]), int(fields[2])
        compute = subprocess.check_output([
            "nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
            "--format=csv,noheader"], text=True).strip()
        require(used_mib <= 64 and utilization == 0 and not compute,
                f"GPU {gpu} is not idle: memory={used_mib} MiB, "
                f"utilization={utilization}%, compute_pids={compute or 'none'}")
        evidence[str(gpu)] = {
            "memory_used_mib": used_mib,
            "utilization_percent": utilization,
            "compute_processes": 0,
        }
    return evidence


def postflight_gpus(gpus, timeout_s=30.0):
    deadline = time.monotonic() + timeout_s
    latest = None
    while time.monotonic() < deadline:
        try:
            return preflight_gpus(gpus)
        except (AssertionError, subprocess.CalledProcessError) as error:
            latest = str(error)
            time.sleep(0.5)
    raise AssertionError(f"assigned GPUs did not return to idle: {latest}")


def pcm_from_wav(path):
    with wave.open(str(path), "rb") as wav:
        require((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) ==
                (1, 2, 16000), "audio must be PCM16 mono 16 kHz WAV")
        return wav.readframes(wav.getnframes())


def start_stack(args, http_port, ws_port):
    ipc = args.log_dir / f"ipc-{os.getpid()}"
    shm = args.log_dir / f"shm-{os.getpid()}"
    ipc.mkdir(parents=True, exist_ok=False)
    shm.mkdir(parents=True, exist_ok=False)
    entry = f"ipc://{ipc}/entry"
    vlm = f"ipc://{ipc}/vlm"
    tts = f"ipc://{ipc}/tts"
    completion = f"ipc://{ipc}/completion"
    abort = f"ipc://{ipc}/abort"
    procs = {}
    procs["asr"] = Proc("asr", [
        args.bin_dir / "umm-native-stage-asr", "--kind", "asr", "--name", "asr",
        "--model", args.asr_model, "--gpu", args.gpu_asr,
        "--recv-endpoint", entry, "--coordinator-endpoint", completion,
        "--abort-endpoint", abort], args.log_dir / "asr.log")
    vlm_binary = "umm-native-stage-omni" if args.vlm_kind == "omni" else \
        "umm-native-stage-qwen38"
    procs["vlm"] = Proc("vlm", [
        args.bin_dir / vlm_binary, "--kind", args.vlm_kind, "--name", "vlm",
        "--model", args.vlm_model, "--gpu", args.gpu_vlm,
        "--recv-endpoint", vlm, "--coordinator-endpoint", completion,
        "--abort-endpoint", abort, "--max-new-tokens", args.vlm_max_tokens],
        args.log_dir / "vlm.log")
    procs["tts"] = Proc("tts", [
        args.bin_dir / "umm-native-stage-tts", "--kind", "tts", "--name", "tts",
        "--model", args.tts_model, "--gpu", args.gpu_tts,
        "--recv-endpoint", tts, "--coordinator-endpoint", completion,
        "--abort-endpoint", abort, "--max-frames", args.tts_max_frames],
        args.log_dir / "tts.log")
    procs["server"] = Proc("server", [
        args.bin_dir / "umm-infer", "serve", "--host", "127.0.0.1",
        "--port", http_port, "--realtime-port", ws_port,
        "--model", "qwen3-unified", "--entry", "asr",
        "--entry-endpoint", entry, "--vlm-entry", "vlm",
        "--vlm-entry-endpoint", vlm, "--tts-entry", "tts",
        "--tts-entry-endpoint", tts, "--completion", completion,
        "--abort", abort, "--session-history", "--ipc-dir", shm],
        args.log_dir / "server.log")
    wait_port(http_port, procs["server"].proc)
    wait_ready(http_port, procs)
    return procs


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


ACTIVE_RECORDER = None


class Recorder:
    def __init__(self, ws, raw_path):
        self.ws = ws
        self.events = queue.Queue()
        self.observed = []
        self.closed = False
        self.stop = threading.Event()
        self.cancel_observed = threading.Event()
        self.raw = Path(raw_path).open("xb")
        self.raw_lock = threading.Lock()
        self.raw_records = 0
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def record_raw(self, direction, raw):
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        record = {"direction": direction,
                  "observed_wall_ns": time.time_ns(),
                  "observed_monotonic_ns": time.monotonic_ns(),
                  "raw_sha256": hashlib.sha256(raw).hexdigest(),
                  "raw_message": raw.decode("utf-8")}
        with self.raw_lock:
            self.raw.write((json.dumps(record, sort_keys=True) + "\n").encode())
            self.raw.flush()
            self.raw_records += 1

    def marker(self, kind, response_id):
        self.record_raw("marker", json.dumps({"type": kind, "response_id": response_id},
                                             sort_keys=True))

    def run(self):
        while not self.stop.is_set():
            try:
                message = self.ws.recv(timeout=0.2)
            except TimeoutError:
                continue
            except Exception:
                self.closed = True
                return
            raw = message.encode("utf-8") if isinstance(message, str) else bytes(message)
            self.record_raw("rx", raw)
            item = (time.time(), json.loads(message))
            self.observed.append(item)
            self.events.put(item)
            if item[1].get("type") == "response.cancelled":
                self.cancel_observed.set()

    def close(self):
        self.stop.set()
        self.thread.join(5)
        require(not self.thread.is_alive(), "raw realtime recorder did not stop")
        self.raw.close()

    def wait_for(self, predicate, timeout_s, description):
        deadline = time.monotonic() + timeout_s
        spent = []
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
        recent = []
        for _, event in spent[-10:]:
            summary = dict(event)
            if "audio" in summary:
                summary["audio"] = f"<{len(summary['audio'])} base64 chars>"
            recent.append(summary)
        raise AssertionError(f"timed out waiting for {description}; recent={recent}")


def record_outgoing(raw):
    if ACTIVE_RECORDER is not None:
        ACTIVE_RECORDER.record_raw("tx", raw)


def append(ws, pcm):
    message = json.dumps({"type": "input_audio_buffer.append",
                          "audio": base64.b64encode(pcm).decode()})
    ws.send(message)
    record_outgoing(message)


def send_speech(ws, speech, start=0, pace=True, stop_event=None):
    chunk = 3200  # 100 ms, PCM16 mono 16 kHz
    offset = start
    while offset < len(speech):
        if stop_event is not None and stop_event.is_set():
            break
        append(ws, speech[offset:offset + chunk])
        offset += chunk
        if pace:
            time.sleep(0.1)
    return offset


def send_silence(ws, chunks=6):
    silence = b"\x00\x00" * 1600
    for _ in range(chunks):
        append(ws, silence)
        time.sleep(0.1)


def response_events_since(rec, timestamp, response_id):
    return [event for at, event in rec.observed
            if at > timestamp and event.get("response_id") == response_id and
            event.get("type") in STALE_TYPES]


def consume_and_ack(ws, audio_event):
    """Consume PCM at its real-time duration, then acknowledge its text cursor."""
    pcm = base64.b64decode(audio_event["audio"])
    channels = int(audio_event["channels"])
    sample_rate = int(audio_event["sample_rate"])
    require(channels > 0 and sample_rate > 0 and len(pcm) % (2 * channels) == 0,
            "invalid PCM16 audio.delta")
    duration_s = len(pcm) / (2.0 * channels * sample_rate)
    time.sleep(duration_s)
    ack = {"type": "output_audio_buffer.playback_ack",
           "response_id": audio_event["response_id"],
           "text_cursor": int(audio_event["text_cursor"])}
    message = json.dumps(ack)
    ws.send(message)
    record_outgoing(message)
    return duration_s


def run_live(args, http_port, ws_port, speech, evidence):
    global ACTIVE_RECORDER
    url = (f"ws://127.0.0.1:{ws_port}/v1/realtime?"
           "intent=transcription&model=qwen3-unified")
    with connect(url, compression=None, open_timeout=10, close_timeout=10,
                 ping_interval=None, max_size=1 << 22) as ws:
        rec = Recorder(ws, args.out / "raw-events.jsonl")
        ACTIVE_RECORDER = rec
        session_update = json.dumps({"type": "session.update", "session": {
            "input_audio_format": "pcm16",
            "turn_detection": {"type": "server_vad", "silence_ms": 400},
            "max_new_tokens": args.vlm_max_tokens,
            "audio_output": {"voice": "vivian", "playback_ack": True}}})
        ws.send(session_update)
        record_outgoing(session_update)
        updated, _ = rec.wait_for(
            lambda e: e.get("type") == "transcription_session.updated", 20,
            "session update")
        require(updated[1]["session"]["audio_output"].get("playback_ack") is True,
                "playback acknowledgement was not negotiated")

        speech = speech[:int(32000 * args.speech_seconds)]
        require(len(speech) >= 3200, "audio contains no usable speech prefix")
        send_speech(ws, speech)
        send_silence(ws)
        first_transcript, _ = rec.wait_for(
            lambda e: e.get("type") == "transcription.completed", 120,
            "first transcription")

        first_id = None
        first_text = ""
        playback_seconds = 0.0
        played_pcm = bytearray()
        played_audio_events = 0
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            item, _ = rec.wait_for(
                lambda e: e.get("type") in ("audio.delta", "response.text.delta",
                                            "response.failed"),
                120, "first partial TTS audio")
            event = item[1]
            if event["type"] == "response.failed":
                raise AssertionError(f"first response failed: {event}")
            if first_id is None:
                first_id = event["response_id"]
            require(event.get("response_id") == first_id,
                    "response id changed before barge-in")
            if event["type"] == "response.text.delta":
                first_text += event["text"]
                continue
            played_pcm.extend(base64.b64decode(event["audio"], validate=True))
            played_audio_events += 1
            playback_seconds += consume_and_ack(ws, event)
            if int(event.get("text_cursor", 0)) > 0:
                acknowledged_cursor = int(event["text_cursor"])
                break
        else:
            raise AssertionError("no multi-sentence TTS acknowledgement cursor arrived")
        require(acknowledged_cursor <= len(first_text.encode("utf-8")),
                "audio cursor exceeds emitted response text")

        rec.cancel_observed.clear()
        rec.marker("client_barge_start", first_id)
        barge_started = time.time()
        next_offset = send_speech(ws, speech, stop_event=rec.cancel_observed)
        cancelled, _ = rec.wait_for(
            lambda e: e.get("type") == "response.cancelled" and
            e.get("response_id") == first_id, 1.0, "first response.cancelled")
        cancel_received_monotonic_ns = time.monotonic_ns()
        old_audio = [event for at, event in rec.observed
                     if at <= cancelled[0] and event.get("response_id") == first_id
                     and event.get("type") == "audio.delta"]
        require(played_audio_events <= len(old_audio),
                "played PCM count exceeds raw old-response audio")
        pending_playback = old_audio[played_audio_events:]
        pending_playback_bytes = sum(
            len(base64.b64decode(event["audio"], validate=True))
            for event in pending_playback)
        require(pending_playback and pending_playback_bytes > 0,
                "long-narration barge-in had no emitted-but-unplayed PCM to silence")
        pending_playback.clear()  # the digital playback queue is now silent
        rec.marker("digital_silence", first_id)
        cancel_to_digital_silence_ms = (
            time.monotonic_ns() - cancel_received_monotonic_ns) / 1e6
        interrupt_ms = (cancelled[0] - barge_started) * 1000.0
        require(interrupt_ms <= 1000.0,
                f"old output did not stop within one second: {interrupt_ms:.1f} ms")
        cancelled_event = cancelled[1]
        acknowledged_text = cancelled_event.get("acknowledged_text", "")
        require(cancelled_event.get("history_recorded") is True,
                "eligible realtime barge-in did not record history")
        require(cancelled_event.get("text_cursor") == acknowledged_cursor,
                "cancellation cursor differs from the last playback ack")
        require(len(acknowledged_text.encode("utf-8")) == acknowledged_cursor,
                "acknowledged_text byte length differs from its cursor")
        require(first_text.startswith(acknowledged_text) and acknowledged_text,
                "acknowledged prefix is not a nonempty prefix of emitted text")
        require(len(first_text.encode("utf-8")) > acknowledged_cursor,
                "test did not retain a proper prefix of the generated answer")
        time.sleep(0.2)  # allow any illegal late stage event to reach the socket
        stale = response_events_since(rec, cancelled[0], first_id)
        require(not stale, f"stale output arrived after response.cancelled: {stale}")

        next_offset = send_speech(ws, speech, start=next_offset)
        send_silence(ws)
        second_transcript, _ = rec.wait_for(
            lambda e: e.get("type") == "transcription.completed", 120,
            "post-barge-in transcription")
        second_response, _ = rec.wait_for(
            lambda e: e.get("type") == "audio.delta" and
            e.get("response_id") != first_id, 120, "post-barge-in first audio")
        second_id = second_response[1]["response_id"]
        post_barge_ttfa_ms = (second_response[0] - second_transcript[0]) * 1000.0
        barge_to_new_audio_ms = (second_response[0] - barge_started) * 1000.0

        # A second new utterance proves repeated cancellation does not duplicate
        # history or wedge either the ASR or unified response path.
        rec.cancel_observed.clear()
        rec.marker("client_barge_start", second_id)
        second_barge = time.time()
        send_speech(ws, speech, stop_event=rec.cancel_observed)
        second_cancel, _ = rec.wait_for(
            lambda e: e.get("type") == "response.cancelled" and
            e.get("response_id") == second_id, 1.0, "second response.cancelled")
        second_interrupt_ms = (second_cancel[0] - second_barge) * 1000.0
        require(second_interrupt_ms <= 1000.0,
                f"second cancellation exceeded one second: {second_interrupt_ms:.1f}")
        final_health = wait_in_flight_zero(http_port)
        stale += response_events_since(rec, second_cancel[0], second_id)
        require(not stale, f"stale output survived repeated cancellation: {stale}")
        rec.close()
        ACTIVE_RECORDER = None
        evidence.update({
            "raw_events": file_sha256(args.out / "raw-events.jsonl"),
            "raw_event_count": len(rec.observed),
            "raw_record_count": rec.raw_records,
            "first_response_id": first_id,
            "first_transcript": first_transcript[1].get("text"),
            "post_barge_in_transcript": second_transcript[1].get("text"),
            "first_text": first_text,
            "acknowledged_text": acknowledged_text,
            "acknowledged_cursor": acknowledged_cursor,
            "virtual_playback_seconds": round(playback_seconds, 3),
            "played_pcm_sha256": hashlib.sha256(played_pcm).hexdigest(),
            "played_pcm_bytes": len(played_pcm),
            "played_audio_events": played_audio_events,
            "cancel_pending_audio_events": len(old_audio) - played_audio_events,
            "cancel_pending_pcm_bytes": pending_playback_bytes,
            "cancel_to_digital_silence_ms": cancel_to_digital_silence_ms,
            "cancel_received_wall_ns": int(cancelled[0] * 1e9),
            "first_interrupt_ms": round(interrupt_ms, 1),
            "second_response_id": second_id,
            "post_barge_ttfa_from_asr_final_ms": post_barge_ttfa_ms,
            "barge_to_new_audio_ms": barge_to_new_audio_ms,
            "second_interrupt_ms": round(second_interrupt_ms, 1),
            "stale_output_events": stale,
            "final_health": final_health,
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--asr-model", type=Path, required=True)
    parser.add_argument("--vlm-model", type=Path, required=True)
    parser.add_argument("--tts-model", type=Path, required=True)
    parser.add_argument("--vlm-kind", choices=["qwen38", "omni"], default="qwen38")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--gpu-asr", type=int, default=0)
    parser.add_argument("--gpu-vlm", type=int, default=1)
    parser.add_argument("--gpu-tts", type=int, default=7)
    parser.add_argument("--vlm-max-tokens", type=int, default=192)
    parser.add_argument("--tts-max-frames", type=int, default=512)
    parser.add_argument("--speech-seconds", type=float, default=2.0)
    args = parser.parse_args()
    require(0.2 <= args.speech_seconds <= 30.0,
            "speech-seconds must be within 0.2..30")
    args.out.mkdir(parents=True, exist_ok=False)
    args.log_dir.mkdir(parents=True, exist_ok=False)
    gpus = [args.gpu_asr, args.gpu_vlm, args.gpu_tts]
    evidence = {
        "schema": "umminfer-full-duplex-barge-in/v1",
        "status": "FAIL",
        "gpus": {"asr": args.gpu_asr, "vlm": args.gpu_vlm, "tts": args.gpu_tts},
        "vlm_kind": args.vlm_kind,
        "vlm_max_tokens": args.vlm_max_tokens,
        "tts_max_frames": args.tts_max_frames,
        "speech_seconds": args.speech_seconds,
        "input_audio": file_sha256(args.audio),
        "harness": file_sha256(Path(__file__)),
        "binaries": {name: file_sha256(args.bin_dir / name) for name in
                     ("umm-infer", "umm-native-stage-asr", "umm-native-stage-qwen38",
                      "umm-native-stage-tts")},
        "playback": "timed virtual PCM consumption; no audio device",
    }
    procs = {}
    try:
        evidence["gpu_preflight"] = preflight_gpus(gpus)
        ports = free_ports(2)
        evidence["ports"] = {"http": ports[0], "websocket": ports[1]}
        procs = start_stack(args, *ports)
        run_live(args, ports[0], ports[1], pcm_from_wav(args.audio), evidence)
        evidence["status"] = "PASS"
        print("PASS full-duplex barge-in playback history "
              f"first={evidence['first_interrupt_ms']} ms "
              f"second={evidence['second_interrupt_ms']} ms in_flight=0", flush=True)
    except Exception as error:
        evidence["error"] = str(error)
        for name, proc in procs.items():
            evidence.setdefault("log_tails", {})[name] = proc.log.text()[-4000:]
        raise
    finally:
        if procs:
            stop_stack(procs)
            evidence["gpu_postflight"] = postflight_gpus(gpus)
        (args.out / "full-duplex-barge-in-evidence.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
