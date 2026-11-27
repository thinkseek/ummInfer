#!/usr/bin/env python3
"""Frozen real-model NS-04 two-interruption replay/lifecycle journey.

The input WAVs are complete frozen recordings sent at 100 ms PCM pacing. The
primary clock starts before the first interruption append and ends at the first
new public audio.delta; no endpoint, ASR, replay or history component is excluded.
Virtual playback acknowledges only PCM consumed at its sample clock. No physical
microphone, speaker, acoustic or perceptual claim is made.
"""

import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import urllib.request
import wave

ROOT = Path(__file__).resolve().parents[1]
CHUNK_BYTES = 3200
END_SILENCE = bytes(19200)
STALE_TYPES = {
    "audio.delta", "audio.done", "response.text.delta", "response.playback_ack",
    "response.completed", "response.failed",
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


live = load_module("umminfer_ns04_network_live_helpers", ROOT / "tests/realtime_network_live.py")
contract = load_module("umminfer_ns04_raw_contract", ROOT / "python/umminfer_realtime_network.py")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_identity(path):
    data = Path(path).read_bytes()
    return {"sha256": sha(data), "bytes": len(data)}


def read_pcm16(path):
    with wave.open(str(path), "rb") as source:
        require((source.getnchannels(), source.getsampwidth(), source.getframerate()) ==
                (1, 2, 16000), f"{path} is not PCM16 mono 16 kHz")
        pcm = source.readframes(source.getnframes())
    return pcm


def health_zero(port, timeout=30):
    deadline = time.monotonic() + timeout
    latest = None
    while time.monotonic() < deadline:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as reply:
            latest = json.load(reply)
        if latest.get("in_flight") == 0:
            return latest
        time.sleep(0.05)
    raise AssertionError(f"final in_flight did not drain: {latest}")


class MarkerLog:
    def __init__(self, path):
        self.output = Path(path).open("xb", buffering=0)
        self.lock = threading.Lock()
        self.rows = []

    def mark(self, name, **fields):
        row = {"monotonic_ns": time.monotonic_ns(), "name": name, **fields}
        with self.lock:
            self.output.write((json.dumps(row, sort_keys=True) + "\n").encode())
            self.rows.append(row)
        return row

    def close(self):
        self.output.close()


def send_paced(ws, pcm):
    for offset in range(0, len(pcm), CHUNK_BYTES):
        chunk = pcm[offset:offset + CHUNK_BYTES]
        ws.send_json({"type": "input_audio_buffer.append",
                      "audio": base64.b64encode(chunk).decode()})
        time.sleep(len(chunk) / 32000.0)


def response_text(ws, response_id, after, through=None):
    records = ws.since(after, types={"response.text.delta"}, response_id=response_id)
    return "".join(record["event"].get("text", "") for record in records
                   if through is None or record["event_index"] <= through)


def response_audio(ws, response_id, after, through=None):
    return [record for record in ws.since(after, types={"audio.delta"}, response_id=response_id)
            if through is None or record["event_index"] <= through]


def play_for(ws, response_id, after, target_seconds, markers, label):
    """Consume each raw PCM chunk once at its real sample clock, then ACK it."""
    cursor = after
    played_seconds = 0.0
    played_pcm = bytearray()
    played_capture_indices = []
    ack_cursor = 0
    while played_seconds < target_seconds:
        record = ws.wait_for(
            lambda event: event.get("type") == "audio.delta" and
            event.get("response_id") == response_id,
            cursor, 240, f"{label} virtual playback audio")
        cursor = record["event_index"] + 1
        event = record["event"]
        pcm = base64.b64decode(event.get("audio", ""), validate=True)
        require(event.get("format") == "pcm16" and event.get("sample_rate") == 24000 and
                event.get("channels") == 1 and pcm and len(pcm) % 2 == 0,
                f"invalid {label} PCM metadata")
        text_before = response_text(ws, response_id, after, record["event_index"])
        text_cursor = event.get("text_cursor")
        require(isinstance(text_cursor, int) and 0 <= text_cursor <= len(text_before.encode()),
                f"{label} audio cursor is outside emitted UTF-8 text")
        time.sleep(len(pcm) / 48000.0)
        played_seconds += len(pcm) / 48000.0
        played_pcm.extend(pcm)
        played_capture_indices.append(record["capture_index"])
        ack_cursor = text_cursor
        ws.send_json({"type": "output_audio_buffer.playback_ack",
                      "response_id": response_id, "text_cursor": ack_cursor})
    # Cursor zero is valid when playback is still inside the first native leg.
    # Do not extend the frozen interruption window just to find a later cursor.
    require(ack_cursor >= 0, f"{label} produced a negative text cursor")
    result = {
        "response_id": response_id,
        "played_pcm_sha256": sha(played_pcm),
        "played_pcm_bytes": len(played_pcm),
        "played_audio_events": len(played_capture_indices),
        "played_capture_indices": played_capture_indices,
        "virtual_played_seconds": played_seconds,
        "acknowledged_cursor": ack_cursor,
        "acknowledged_text": response_text(ws, response_id, after).encode()[:ack_cursor].decode(),
    }
    markers.mark(f"{label}_playback_target_reached", **result)
    return result


def wait_response_identity(ws, after, excluded, segment):
    record = ws.wait_for(
        lambda event: event.get("type") == "audio.delta" and
        event.get("response_id") not in excluded and
        event.get("segment_id") == segment,
        after, 240, "new response first public audio")
    return record


def barge(ws, pcm, old_response_id, old_after, played, markers, label):
    marker = ws.mark()
    start = markers.mark(f"{label}_barge_start", response_id=old_response_id)
    stream_error = []

    def send_all():
        try:
            send_paced(ws, pcm)
            send_paced(ws, END_SILENCE)
        except Exception as error:  # retained in the main thread
            stream_error.append(error)

    sender = threading.Thread(target=send_all, daemon=True)
    sender.start()
    cancelled = ws.wait_for(
        lambda event: event.get("type") == "response.cancelled" and
        event.get("response_id") == old_response_id,
        marker, 5, f"{label} response.cancelled")
    cancel_receipt_ns = time.monotonic_ns()
    event = cancelled["event"]
    require(event.get("reason") == "barge_in" and event.get("history_recorded") is True,
            f"{label} did not record eligible barge-in history")
    ack_cursor = played["acknowledged_cursor"]
    require(event.get("text_cursor") == ack_cursor,
            f"{label} cancellation cursor differs from last playback ACK")
    acknowledged = event.get("acknowledged_text")
    require(isinstance(acknowledged, str) and len(acknowledged.encode()) == ack_cursor,
            f"{label} acknowledged prefix has the wrong UTF-8 length")
    emitted_text = response_text(ws, old_response_id, old_after, cancelled["event_index"])
    require(emitted_text.encode()[:ack_cursor].decode() == acknowledged and
            acknowledged == played["acknowledged_text"],
            f"{label} cancellation is not the exact played text prefix")

    old_audio = response_audio(ws, old_response_id, old_after, cancelled["event_index"])
    played_indices = set(played["played_capture_indices"])
    pending = [record for record in old_audio if record["capture_index"] not in played_indices]
    pending_pcm = b"".join(base64.b64decode(record["event"]["audio"], validate=True)
                           for record in pending)
    all_old_pcm = b"".join(base64.b64decode(record["event"]["audio"], validate=True)
                           for record in old_audio)
    require(all_old_pcm[:played["played_pcm_bytes"]] and
            sha(all_old_pcm[:played["played_pcm_bytes"]]) == played["played_pcm_sha256"],
            f"{label} played PCM is not an exact emitted prefix")
    pending.clear()  # the digital playback queue is silent; raw evidence retains its bytes
    silence = markers.mark(f"{label}_digital_silence", response_id=old_response_id,
                           pending_audio_events=len(pending_pcm) and len(old_audio) - len(played_indices),
                           pending_pcm_bytes=len(pending_pcm),
                           pending_pcm_sha256=sha(pending_pcm))
    require(pending_pcm, f"{label} had no emitted-but-unplayed PCM to silence")

    sender.join(60)
    require(not sender.is_alive() and not stream_error,
            f"{label} complete paced replay failed: {stream_error}")
    started = ws.wait_for(
        lambda item: item.get("type") == "input_audio_buffer.speech_started" and
        item.get("interrupted_response_id") == old_response_id,
        marker, 10, f"{label} speech started")
    segment = started["event"].get("segment_id")
    stopped = ws.wait_for(
        lambda item: item.get("type") == "input_audio_buffer.speech_stopped" and
        item.get("segment_id") == segment,
        started["event_index"], 30, f"{label} speech stopped")
    completed = ws.wait_for(
        lambda item: item.get("type") == "transcription.completed" and
        item.get("segment_id") == segment,
        stopped["event_index"], 180, f"{label} real ASR final")
    require(completed["event"].get("text", "").strip(), f"{label} empty ASR transcript")
    first_audio = wait_response_identity(ws, started["event_index"], {old_response_id}, segment)
    new_response_id = first_audio["event"]["response_id"]
    stale = ws.since(cancelled["event_index"] + 1, types=STALE_TYPES,
                     response_id=old_response_id)
    require(not stale, f"{label} stale old-response events survived cancellation")
    return {
        "label": label,
        "marker": marker,
        "old_after": old_after,
        "old_response_id": old_response_id,
        "new_response_id": new_response_id,
        "new_segment": segment,
        "barge_start_ns": start["monotonic_ns"],
        "cancelled": cancelled,
        "cancel_receipt_ns": cancel_receipt_ns,
        "digital_silence_ns": silence["monotonic_ns"],
        "speech_started": started,
        "speech_stopped": stopped,
        "asr_final": completed,
        "first_audio": first_audio,
        "played": played,
        "old_audio_events": len(old_audio),
        "old_pcm_sha256": sha(all_old_pcm),
        "old_pcm_bytes": len(all_old_pcm),
        "pending_audio_events": len(old_audio) - len(played_indices),
        "pending_pcm_bytes": len(pending_pcm),
        "pending_pcm_sha256": sha(pending_pcm),
        "stale_events": stale,
        "timings_ms": {
            "barge_to_cancel": (cancelled["arrival_ns"] - start["monotonic_ns"]) / 1e6,
            "cancel_to_digital_silence": (silence["monotonic_ns"] - cancel_receipt_ns) / 1e6,
            "barge_to_speech_stopped_replay": (stopped["arrival_ns"] - start["monotonic_ns"]) / 1e6,
            "speech_stopped_to_asr_final": (completed["arrival_ns"] - stopped["arrival_ns"]) / 1e6,
            "asr_final_to_new_audio": (first_audio["arrival_ns"] - completed["arrival_ns"]) / 1e6,
            "replay_inclusive_barge_to_new_audio": (first_audio["arrival_ns"] - start["monotonic_ns"]) / 1e6,
        },
    }


def compact_run(run):
    """Remove binary audio/frame payloads from the small driver summary."""
    return run


def execute(args, prereg, audio):
    ports = live.free_ports(2)
    markers = MarkerLog(args.out / "markers.jsonl")
    ws = None
    procs = {}
    observer = None
    isolated = []
    run = {
        "schema": "umminfer-ns04-replay-lifecycle-live/v1",
        "status": "FAIL",
        "arm": args.arm,
        "repetition": args.repetition,
        "accepted_parent_commit": prereg["accepted_parent_commit"],
        "execution_commit": args.execution_commit,
        "preregistration": file_identity(args.preregistration),
        "candidate_preregistration": file_identity(args.candidate_preregistration),
        "harness": file_identity(Path(__file__)),
        "binaries": {name: file_identity(args.bin_dir / name) for name in
                     ("umm-infer", "umm-native-stage-asr", "umm-native-stage-qwen38",
                      "umm-native-stage-tts")},
        "audio": {name: {**file_identity(args.audio_dir / row["audio_file"]),
                         "pcm_sha256": sha(audio[name]["pcm"]),
                         "pcm_bytes": len(audio[name]["pcm"])}
                  for name, row in audio.items()},
        "gpu_preflight": live.preflight(),
        "physical_microphone_speaker": "NOT_RUN",
        "acoustic_echo": "NOT_RUN",
        "perceptual_listening_quality": "NOT_RUN",
    }
    try:
        run["models"] = live.hash_models(args, args.out)
        procs, observer, isolated = live.start_stack(args, ports, args.out, args.log_dir)
        ws = live.RawWebSocket(ports[1], args.out / "frames.jsonl")
        ws.send_json({"type": "session.update", "session": {
            "input_audio_format": "pcm16",
            "turn_detection": {"type": "server_vad", "silence_ms": 400},
            "max_new_tokens": 768,
            "audio_output": {"voice": "vivian", "playback_ack": True},
        }})
        updated = ws.wait_for(lambda event: event.get("type") == "transcription_session.updated",
                              0, 20, "session update")
        require(updated["event"]["session"]["audio_output"].get("playback_ack") is True,
                "playback ACK was not negotiated")

        solar_marker = ws.mark()
        send_paced(ws, audio["solar"]["pcm"])
        send_paced(ws, END_SILENCE)
        solar_final = ws.wait_for(lambda event: event.get("type") == "transcription.completed",
                                  solar_marker, 180, "solar real ASR final")
        for fragment in prereg["journey"]["turns"][0]["required_transcript_fragments"]:
            require(fragment in solar_final["event"].get("text", ""),
                    f"solar ASR missed frozen fragment {fragment}")
        solar_identity = ws.wait_for(
            lambda event: event.get("response_id") and event.get("type") in
            ("response.text.delta", "audio.delta"), solar_final["event_index"], 240,
            "solar response identity")
        solar_id = solar_identity["event"]["response_id"]
        solar_played = play_for(ws, solar_id, solar_marker, 3.0, markers, "solar")
        require(solar_played["acknowledged_cursor"] > 0,
                "solar barge-in did not exercise a positive playback cursor")
        first = barge(ws, audio["saturn"]["pcm"], solar_id, solar_marker, solar_played,
                      markers, "first")
        for fragment in prereg["journey"]["turns"][1]["required_transcript_fragments"]:
            require(fragment in first["asr_final"]["event"].get("text", ""),
                    f"Saturn ASR missed frozen fragment {fragment}")

        saturn_played = play_for(ws, first["new_response_id"], first["marker"], 2.0,
                                 markers, "saturn")
        second = barge(ws, audio["english"]["pcm"], first["new_response_id"], first["marker"],
                       saturn_played, markers, "second")
        for fragment in prereg["journey"]["turns"][2]["required_transcript_fragments"]:
            require(fragment in second["asr_final"]["event"].get("text", ""),
                    f"English-switch ASR missed frozen fragment {fragment}")

        final_id = second["new_response_id"]
        terminal = ws.wait_for(
            lambda event: event.get("response_id") == final_id and event.get("type") in
            ("response.completed", "response.failed"),
            second["marker"], 300, "final English response terminal")
        require(terminal["event"]["type"] == "response.completed",
                f"final English response failed: {terminal['event']}")
        final_text = response_text(ws, final_id, second["marker"], terminal["event_index"])
        final_pcm = b"".join(base64.b64decode(record["event"]["audio"], validate=True)
                             for record in response_audio(ws, final_id, second["marker"],
                                                         terminal["event_index"]))
        ascii_letters = sum(chr(byte).isalpha() for byte in final_text.encode() if byte < 128)
        cjk = sum(0x4E00 <= ord(char) <= 0x9FFF for char in final_text)
        require(ascii_letters > 20 and ascii_letters > 2 * max(1, cjk),
                f"final response is not English-dominant: {final_text[:200]!r}")
        require("saturn" in final_text.lower() or "ring" in final_text.lower(),
                "final English response lost the Saturn/ring topic")
        require(final_pcm, "final English response has no public PCM")

        terminals = {}
        for response_id, after in ((solar_id, solar_marker), (first["new_response_id"], first["marker"]),
                                   (final_id, second["marker"])):
            rows = ws.since(after, types={"response.completed", "response.failed", "response.cancelled"},
                            response_id=response_id)
            kinds = [row["event"]["type"] for row in rows]
            require(len(kinds) == len(set(kinds)), f"duplicate terminal callback for {response_id}: {kinds}")
            terminals[response_id] = kinds
        require(terminals[solar_id] == ["response.cancelled"] and
                terminals[first["new_response_id"]] == ["response.cancelled"] and
                terminals[final_id] == ["response.completed"],
                f"unexpected terminal lifecycle: {terminals}")

        run.update({
            "transcripts": {
                "solar": solar_final["event"]["text"],
                "saturn": first["asr_final"]["event"]["text"],
                "english": second["asr_final"]["event"]["text"],
            },
            "first_barge": compact_run(first),
            "second_barge": compact_run(second),
            "final_response": {
                "response_id": final_id,
                "text": final_text,
                "text_sha256": sha(final_text.encode()),
                "text_utf8_bytes": len(final_text.encode()),
                "pcm_sha256": sha(final_pcm),
                "pcm_bytes": len(final_pcm),
                "ascii_letters": ascii_letters,
                "cjk_codepoints": cjk,
                "terminal": terminal,
            },
            "terminal_callbacks": terminals,
            "observer_rows": observer.rows,
        })
        run["status"] = "PASS_CORRECTNESS"
    except Exception as error:
        run["driver_error"] = str(error)
        for name, proc in procs.items():
            run.setdefault("log_tails", {})[name] = proc.log.text()[-4000:]
        raise
    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        if procs:
            try:
                run["final_health"] = health_zero(ports[0])
            except Exception as error:
                run["final_health_error"] = str(error)
            live.stop_stack(procs)
        if observer is not None:
            observer.close()
        for proc in procs.values():
            shutil.copyfile(proc.log.path, args.out / f"{proc.name}.log")
        for path in isolated:
            shutil.rmtree(path, ignore_errors=True)
        markers.close()
        run["cleanup"] = {
            "owned_processes_exited": bool(procs) and all(proc.proc.poll() is not None
                                                           for proc in procs.values()),
            "observer_error": None if observer is None or observer.error is None else str(observer.error),
        }
        try:
            run["gpu_postflight"] = live.wait_postflight()
            run["cleanup"]["selected_gpu_idle"] = True
        except Exception as error:
            run["cleanup"]["selected_gpu_idle"] = False
            run["cleanup"]["gpu_error"] = str(error)
        (args.out / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--asr-model", type=Path, required=True)
    parser.add_argument("--vlm-model", type=Path, required=True)
    parser.add_argument("--tts-model", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--candidate-preregistration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=["before", "after", "profile"], required=True)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--vlm-max-tokens", type=int, default=768)
    parser.add_argument("--tts-max-frames", type=int, default=8192)
    args = parser.parse_args()
    require(args.gpu == 7, "this task permits only explicitly idle physical GPU 7")
    require(args.repetition >= 1 and args.vlm_max_tokens == 768 and args.tts_max_frames == 8192,
            "frozen journey parameters changed")
    require(not args.out.exists() and not args.log_dir.exists(),
            "caller-selected evidence directories must be new")
    args.out.mkdir(parents=True)
    args.log_dir.mkdir(parents=True)
    prereg = json.loads(args.preregistration.read_text())
    require(prereg["format"] == "umminfer-ns04-replay-lifecycle-preregistration-v1",
            "wrong preregistration schema")
    candidate_prereg = json.loads(args.candidate_preregistration.read_text())
    require(candidate_prereg["format"] ==
            "umminfer-ns04-incremental-early-response-preregistration-v1",
            "wrong incremental candidate preregistration schema")
    args.capture_tts_submits = True
    audio = {}
    for row in prereg["journey"]["turns"]:
        path = args.audio_dir / row["audio_file"]
        identity = file_identity(path)
        require(identity["sha256"] == row["audio_sha256"] and identity["bytes"] == row["audio_bytes"],
                f"frozen audio identity mismatch: {row['name']}")
        pcm = read_pcm16(path)
        require(len(pcm) == row["pcm_bytes"], f"frozen PCM length mismatch: {row['name']}")
        audio[row["name"]] = {"pcm": pcm, **row}
    live.ASR_GPU = live.VLM_GPU = live.TTS_GPU = args.gpu
    require(os.environ.get("UMMINFER_LIVE_OPT_IN") == "1",
            "set UMMINFER_LIVE_OPT_IN=1 for the isolated real-model gate")
    run = execute(args, prereg, audio)
    first = run["first_barge"]["timings_ms"]
    print("PASS_CORRECTNESS NS-04 frozen journey "
          f"first_cancel={first['barge_to_cancel']:.3f} ms "
          f"asr_final_to_audio={first['asr_final_to_new_audio']:.3f} ms "
          f"replay_inclusive={first['replay_inclusive_barge_to_new_audio']:.3f} ms",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
