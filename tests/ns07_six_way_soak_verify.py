#!/usr/bin/env python3
"""Independent fresh-process verifier for the six-way NS-07 soak.

No producer, scheduler, model, or scorecard module is imported.  Synthetic
bundles are useful for contract tests only and can never earn ISOLATED_PASS.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import sys
import wave

LANES = {"text_code", "interleaved_visual", "full_duplex_voice",
         "natural_video_input", "abort_retry", "malformed_recovery"}
OUTPUT_STAGES = {"flux", "tts", "wan"}
TERMINALS = {"complete", "cancelled", "error"}
# Final playback ACKs legitimately follow response.completed after digital drain.
# Their cursor/timing is checked independently in verify_playback, not as media.
WS_STALE = {"audio.delta", "audio.done", "response.text.delta",
            "response.completed", "response.failed", "response.cancelled"}
METRIC_NAMES = ("elapsed_s", "client_count", "request_count", "completed_count",
    "cancelled_count", "expected_fault_count", "unexpected_error_count",
    "process_crash_count", "cross_session_event_count", "stale_event_count",
    "duplicate_terminal_count", "pcm_mismatch_count", "recovery_attempt_count",
    "recovery_success_count", "final_in_flight", "device_growth_mib",
    "rss_growth_mib", "latency_p95_ms")
ALLOWED_NORMALIZED = {"case_id", "schema", "arm", "engineering_status",
    "production_status", "full_gate_status", "metrics", "hard_gates", "failures",
    "limitations", "source_identity", "binary_identities", "model_identities",
    "config_identity", "raw_evidence", "coverage", "deployment_scope"}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    rows = []
    with Path(path).open("rb") as source:
        for number, line in enumerate(source, 1):
            if not line.endswith(b"\n"):
                raise ValueError(f"unterminated JSONL: {path}:{number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL: {path}:{number}")
            rows.append(row)
    return rows


def percentile(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)])


def project_event(event):
    value = json.loads(canonical(event))
    artifact = value.get("artifact")
    if isinstance(artifact, dict) and isinstance(artifact.get("data"), str):
        raw = base64.b64decode(artifact["data"], validate=True)
        artifact["data_sha256"] = sha(raw)
        artifact["data_bytes"] = len(raw)
        del artifact["data"]
    return value


def parse_sse(raw):
    events = []
    done = 0
    cursor = 0
    while cursor < len(raw):
        end = raw.find(b"\n\n", cursor)
        if end < 0:
            raise ValueError("truncated SSE frame")
        frame = raw[cursor:end]
        cursor = end + 2
        data = []
        for line in frame.split(b"\n"):
            if line.startswith(b"data:"):
                value = line[5:]
                data.append(value[1:] if value.startswith(b" ") else value)
        if not data:
            continue
        payload = b"\n".join(data)
        if payload == b"[DONE]":
            done += 1
            if cursor != len(raw):
                raise ValueError("bytes follow final [DONE]")
            continue
        outer = json.loads(payload)
        if not isinstance(outer, dict):
            raise ValueError("SSE outer frame is not an object")
        unified = outer.get("unified")
        if isinstance(unified, dict):
            events.append((outer, unified))
    return events, done


def decode_ws(raw, direction):
    if len(raw) < 2:
        raise ValueError("short WebSocket frame")
    first, second = raw[0], raw[1]
    if first & 0x70:
        raise ValueError("unexpected WebSocket RSV")
    fin = bool(first & 0x80)
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    if masked != (direction == "tx"):
        raise ValueError("incorrect WebSocket mask direction")
    length = second & 0x7F
    offset = 2
    if length == 126:
        if len(raw) < 4:
            raise ValueError("short 16-bit WebSocket length")
        length = struct.unpack_from("!H", raw, 2)[0]
        offset = 4
        if length < 126:
            raise ValueError("noncanonical WebSocket length")
    elif length == 127:
        if len(raw) < 10:
            raise ValueError("short 64-bit WebSocket length")
        length = struct.unpack_from("!Q", raw, 2)[0]
        offset = 10
        if length < 65536 or length >= 1 << 63:
            raise ValueError("invalid WebSocket length")
    mask = None
    if masked:
        if len(raw) < offset + 4:
            raise ValueError("short WebSocket mask")
        mask = raw[offset:offset + 4]
        offset += 4
    if len(raw) != offset + length:
        raise ValueError("WebSocket frame length mismatch")
    payload = raw[offset:]
    if mask:
        payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    if opcode >= 8 and (not fin or length > 125):
        raise ValueError("invalid WebSocket control frame")
    return fin, opcode, payload


def decode_capture(path):
    frames = []
    fragments = {}
    previous_ns = -1
    for index, row in enumerate(read_jsonl(path)):
        if set(row) != {"index", "monotonic_ns", "direction", "raw_base64"}:
            raise ValueError("unexpected realtime capture schema")
        if row["index"] != index or row["monotonic_ns"] <= previous_ns:
            raise ValueError("realtime capture index/clock is not monotonic")
        previous_ns = row["monotonic_ns"]
        direction = row["direction"]
        if direction not in ("rx", "tx"):
            raise ValueError("unknown WebSocket direction")
        raw = base64.b64decode(row["raw_base64"], validate=True)
        fin, opcode, payload = decode_ws(raw, direction)
        record = {"index": index, "monotonic_ns": row["monotonic_ns"],
                  "direction": direction, "raw_sha256": sha(raw), "opcode": opcode}
        frames.append(record)
        if opcode in (8, 9, 10):
            continue
        if opcode not in (0, 1):
            raise ValueError("binary realtime message is unsupported")
        if opcode != 0:
            if direction in fragments:
                raise ValueError("nested WebSocket fragments")
            fragments[direction] = bytearray()
        if direction not in fragments:
            raise ValueError("continuation without start")
        fragments[direction].extend(payload)
        if len(fragments[direction]) > 1 << 22:
            raise ValueError("realtime message exceeds 4 MiB")
        if fin:
            message = bytes(fragments.pop(direction))
            event = json.loads(message.decode("utf-8"))
            if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                raise ValueError("realtime event lacks a string type")
            frames.append({"index": index, "monotonic_ns": row["monotonic_ns"],
                           "direction": direction, "event": event,
                           "payload_sha256": sha(message)})
    if fragments:
        raise ValueError("unterminated WebSocket fragment")
    return frames


def expected_fault_ok(record, status, ok):
    if not record.get("expected_fault"):
        return False
    if record["kind"] in ("random_abort", "realtime_voice"):
        return ok and status == "cancelled"
    if record["kind"] in ("malformed_audio", "malformed_dimensions"):
        return ok and status == "expected_error"
    return False


class Verify:
    def __init__(self, root, allow_synthetic=False):
        self.root = Path(root).resolve()
        self.allow_synthetic = allow_synthetic
        self.failures = []
        self.counts = {name: 0 for name in (
            "cross_session_event_count", "stale_event_count", "duplicate_terminal_count",
            "pcm_mismatch_count", "process_crash_count")}
        self.coverage = {}
        self.realtime = {}
        self.captures = {}
        self.verified = {}
        self.native_parents = {}
        self.native_pcm_exact = {"http": 0, "realtime": 0}

    def fail(self, message):
        self.failures.append(str(message))

    def artifacts(self, record, raw_events):
        groups = {}
        for outer, event in raw_events:
            del outer
            if event.get("kind") != "artifact":
                continue
            artifact = event.get("artifact")
            if not isinstance(artifact, dict) or not isinstance(artifact.get("artifact_id"), str):
                raise ValueError("artifact lacks an id")
            groups.setdefault(artifact["artifact_id"], []).append(artifact)
        actual = []
        public_audio = []
        for artifact_id, chunks in groups.items():
            ordered = sorted(chunks, key=lambda item: item.get("index", -1))
            total = ordered[0].get("total")
            if [item.get("index") for item in ordered] != list(range(total)) \
                    or any(item.get("total") != total for item in ordered):
                raise ValueError(f"artifact chunk gap/total mismatch: {artifact_id}")
            raw = b"".join(base64.b64decode(item["data"], validate=True) for item in ordered)
            final = ordered[-1]
            if final.get("sha256") != sha(raw) or final.get("bytes") != len(raw):
                raise ValueError(f"artifact digest/bytes mismatch: {artifact_id}")
            item = {"artifact_id": artifact_id, "bytes": len(raw), "sha256": sha(raw),
                    "reported_sha256": final.get("sha256"), "reported_bytes": final.get("bytes"),
                    "media_type": final.get("media_type"), "mime_type": final.get("mime_type"),
                    "slot_id": final.get("slot_id"), "slot_index": final.get("slot_index"),
                    "stream_index": final.get("stream_index"),
                    "sample_rate": final.get("sample_rate"), "channels": final.get("channels"),
                    "chunks": len(ordered), "raw": raw}
            actual.append(item)
            if final.get("mime_type") == "audio/pcm":
                if final.get("format") != "pcm16" or final.get("sample_rate") != 24000 \
                        or final.get("channels") != 1 or not raw or len(raw) % 2:
                    raise ValueError(f"invalid public PCM: {artifact_id}")
                public_audio.append(item)
        expected = record.get("artifacts", [])
        if len(expected) != len(actual):
            raise ValueError("producer artifact count differs from raw stream")
        for wanted, observed in zip(expected, actual):
            for key in ("artifact_id", "bytes", "sha256", "media_type", "mime_type",
                        "slot_id", "slot_index", "stream_index", "sample_rate", "channels", "chunks"):
                if wanted.get(key) != observed[key]:
                    raise ValueError(f"producer artifact {key} mismatch: {observed['artifact_id']}")
            media = self.root / wanted.get("media_file", "")
            if not media.is_file() or file_sha(media) != observed["sha256"]:
                raise ValueError(f"retained media differs from raw artifact: {observed['artifact_id']}")
        return public_audio

    def verify_http(self, record, event_rows):
        raw_path = self.root / record["raw_file"]
        request_path = self.root / record["request_file"]
        if not raw_path.is_file() or not request_path.is_file():
            raise ValueError("request or raw response bytes are missing")
        raw = raw_path.read_bytes()
        request_raw = request_path.read_bytes()
        if sha(raw) != record["raw_sha256"] or len(raw) != record["raw_bytes"]:
            raise ValueError("raw response identity mismatch")
        if sha(request_raw) != record["request_sha256"]:
            raise ValueError("request identity mismatch")
        if record["started_monotonic_ns"] > record["ended_monotonic_ns"]:
            raise ValueError("nonmonotonic request clock")
        if abs((record["ended_monotonic_ns"] - record["started_monotonic_ns"]) / 1e6
               - record["latency_ms"]) > 0.001:
            raise ValueError("request latency differs from raw clocks")
        actual_status = "error"
        ok = False
        public_audio = []
        if record["path"] == "/v1/chat/completions":
            body = json.loads(request_raw)
            if body.get("session_id") != record.get("session_id"):
                raise ValueError("request/session identity mismatch")
            if record["kind"] == "natural_video":
                content = body["messages"][0]["content"]
                url = content[0]["video_url"]["url"]
                video = base64.b64decode(url.split(",", 1)[1], validate=True)
                if sha(video) != self.run["inputs"]["natural_video"]["sha256"]:
                    raise ValueError("natural video request differs from frozen input")
            raw_events, done = parse_sse(raw)
            seq = [event.get("seq") for _, event in raw_events]
            if seq != list(range(len(seq))):
                self.fail(f"{record['request_key']}: unified sequence is not gap-free")
            terminals = [index for index, (_, event) in enumerate(raw_events)
                         if event.get("kind") in TERMINALS]
            self.counts["duplicate_terminal_count"] += max(0, len(terminals) - 1)
            if terminals:
                self.counts["stale_event_count"] += len(raw_events) - terminals[0] - 1
            outer_ids = {outer.get("id") for outer, _ in raw_events if outer.get("id") is not None}
            if len(outer_ids) > 1 or (outer_ids and record.get("request_id") not in outer_ids):
                self.fail(f"{record['request_key']}: outer request attribution mismatch")
            for _, event in raw_events:
                if record.get("session_id") and event.get("session_id") not in (None, record["session_id"]):
                    self.counts["cross_session_event_count"] += 1
                if record.get("request_id") and event.get("request_id") not in (None, record["request_id"]):
                    self.fail(f"{record['request_key']}: unified request attribution mismatch")
            projected = [{"request_key": record["request_key"], "lane": record["lane"],
                          "observed_monotonic_ns": row["observed_monotonic_ns"],
                          "outer_id": outer.get("id"), "event": project_event(event)}
                         for row, (outer, event) in zip(event_rows, raw_events)]
            if len(event_rows) != len(raw_events) or any(
                    {k: v for k, v in row.items() if k != "observed_monotonic_ns"} !=
                    {k: v for k, v in other.items() if k != "observed_monotonic_ns"}
                    for row, other in zip(event_rows, projected)):
                raise ValueError("projected event JSONL differs from raw SSE")
            public_audio = self.artifacts(record, raw_events)
            terminal = raw_events[terminals[0]][1]["kind"] if terminals else None
            if terminal == "complete":
                actual_status = "completed"
            elif terminal == "cancelled":
                actual_status = "cancelled"
            wanted = "complete" if record["status"] == "completed" else \
                     "cancelled" if record["status"] == "cancelled" else None
            ok = (record["http_status"] == 200 and done == 1 and terminal == wanted
                  and len(terminals) == 1 and terminals[0] == len(raw_events) - 1
                  and seq == list(range(len(seq))))
        elif record["status"] == "expected_error":
            ok = record["http_status"] >= 400 and bool(raw)
            actual_status = "expected_error" if ok else "error"
        elif record["kind"] == "valid_audio_recovery":
            parsed = json.loads(raw)
            ok = record["http_status"] == 200 and bool(parsed.get("text", "").strip())
            actual_status = "completed" if ok else "error"
        elif record["kind"] == "valid_dimension_recovery":
            ok = (record["http_status"] == 200 and raw
                  and record["http_headers"].get("content-type", "").split(";", 1)[0] == "video/mp4")
            actual_status = "completed" if ok else "error"
            expected = record.get("artifacts", [])
            if len(expected) != 1 or file_sha(self.root / expected[0]["media_file"]) != sha(raw):
                raise ValueError("direct-render retained media differs from raw response")
        else:
            ok = False
        if record.get("unexpected_error"):
            self.fail(f"{record['request_key']}: producer retained an unexpected error")
        if bool(record.get("ok")) != ok:
            self.fail(f"{record['request_key']}: producer ok flag differs from raw evidence")
        if record.get("terminal_count") is not None and record["transport"] == "http" \
                and record["path"] == "/v1/chat/completions":
            # The independent raw terminal count, not the producer scalar, is authoritative.
            pass
        return {"ok": ok, "status": actual_status, "public_audio": public_audio,
                "request_id": record.get("request_id")}

    def verify_realtime(self, record):
        capture = record["capture_file"]
        if capture not in self.captures:
            self.captures[capture] = decode_capture(self.root / capture)
        frames = self.captures[capture]
        response_id = record["response_id"]
        rx = [row for row in frames if row["direction"] == "rx"
              and row.get("event", {}).get("response_id") == response_id]
        terminals = [row for row in rx if row["event"]["type"] in
                     ("response.completed", "response.cancelled", "response.failed")]
        self.counts["duplicate_terminal_count"] += max(0, len(terminals) - 1)
        if terminals:
            terminal_index = terminals[0]["index"]
            self.counts["stale_event_count"] += sum(
                row["index"] > terminal_index and row["event"]["type"] in WS_STALE for row in rx)
        terminal_type = terminals[0]["event"]["type"] if len(terminals) == 1 else None
        status = "completed" if terminal_type == "response.completed" else \
                 "cancelled" if terminal_type == "response.cancelled" else "error"
        text = "".join(row["event"].get("text", "") for row in rx
                       if row["event"]["type"] == "response.text.delta")
        audio_rows = [row for row in rx if row["event"]["type"] == "audio.delta"]
        pcm = b"".join(base64.b64decode(row["event"]["audio"], validate=True) for row in audio_rows)
        pcm_path = self.root / record["pcm_file"]
        if not pcm_path.is_file() or pcm_path.read_bytes() != pcm:
            raise ValueError("retained realtime PCM differs from raw WebSocket")
        if sha(text.encode()) != record["text_sha256"] or sha(pcm) != record["pcm_sha256"] \
                or len(pcm) != record["pcm_bytes"]:
            raise ValueError("realtime text/PCM summary differs from raw capture")
        ok = (len(terminals) == 1 and status == record["status"] and bool(text) and bool(pcm)
              and record["request_id"] == response_id)
        if record.get("unexpected_error") or bool(record.get("ok")) != ok:
            self.fail(f"{record['request_key']}: realtime producer outcome mismatch")
        self.realtime[record["request_key"]] = {
            "response_id": response_id, "capture_file": capture, "frames": frames,
            "audio_rows": audio_rows, "text": text, "pcm": pcm, "ok": ok, "status": status}
        return {"ok": ok, "status": status, "public_audio": [],
                "request_id": response_id}

    def verify_processes(self):
        processes = self.run.get("processes", [])
        leases = self.run.get("leases", [])
        ids = set()
        for lease in leases:
            if lease.get("id") in ids or lease.get("stage") not in OUTPUT_STAGES:
                self.fail("duplicate/unknown output lease")
            ids.add(lease.get("id"))
            if lease.get("end_monotonic_ns") is None \
                    or lease["end_monotonic_ns"] < lease["start_monotonic_ns"]:
                self.fail("unterminated/nonmonotonic output lease")
        for index, left in enumerate(leases):
            for right in leases[index + 1:]:
                if left["stage"] == right["stage"]:
                    continue
                if left["end_monotonic_ns"] is None or right["end_monotonic_ns"] is None:
                    continue
                if left["start_monotonic_ns"] < right["end_monotonic_ns"] \
                        and right["start_monotonic_ns"] < left["end_monotonic_ns"]:
                    self.fail("different output-stage leases overlap")
        history = self.run.get("output_stage_history", [])
        for index, left in enumerate(history):
            if left.get("stage") not in OUTPUT_STAGES:
                self.fail("unknown output-stage process")
            for right in history[index + 1:]:
                if left.get("stage") == right.get("stage"):
                    continue
                if left.get("stop_monotonic_ns") is None or right.get("stop_monotonic_ns") is None:
                    continue
                if left["start_monotonic_ns"] < right["stop_monotonic_ns"] \
                        and right["start_monotonic_ns"] < left["stop_monotonic_ns"]:
                    self.fail("different output-stage process lifetimes overlap")
        for lease in leases:
            candidates = [process for process in history
                          if process.get("stage") == lease["stage"]
                          and process["start_monotonic_ns"] <= lease["start_monotonic_ns"]
                          and (process.get("stop_monotonic_ns") or 0) >= lease["start_monotonic_ns"]]
            if len(candidates) != 1 or lease["end_monotonic_ns"] is None \
                    or candidates[0].get("stop_monotonic_ns") < lease["end_monotonic_ns"]:
                self.fail("output stage stopped during leased work or lacks process ownership")
        for process in processes:
            command = process.get("command", [])
            for index, arg in enumerate(command[:-1]):
                if arg == "--gpu" and str(command[index + 1]) != "0":
                    self.fail("a task-owned stage selected a non-GPU0 device")
            code = process.get("returncode")
            if code is None or process.get("stop_monotonic_ns") is None:
                self.fail(f"owned process did not exit: {process.get('name')}")
            elif not process.get("expected_stop") or code not in (0, -15):
                self.counts["process_crash_count"] += 1
        self.coverage["output_stage_overlap_count"] = sum(
            "process lifetimes overlap" in item for item in self.failures)
        self.coverage["lease_violation_count"] = sum(
            "leases overlap" in item or "leased work" in item for item in self.failures)
        self.coverage["owned_processes_exited"] = bool(processes) and all(
            process.get("returncode") is not None and process.get("stop_monotonic_ns") is not None
            for process in processes)
        server = next((process for process in processes if process.get("name") == "server"), {})
        command = server.get("command", [])
        try:
            admission = int(command[command.index("--max-in-flight") + 1])
        except (ValueError, IndexError):
            admission = 0
        self.coverage["server_max_in_flight"] = admission

    def verify_native_pcm(self):
        submit_root = self.root / "taps/tts"
        output_root = self.root / "taps/tts-output"
        if not submit_root.is_dir() or not output_root.is_dir():
            self.fail("native TTS tap directories are missing")
            return
        try:
            import msgpack
        except ModuleNotFoundError:
            self.fail("msgpack is required to independently decode native TTS evidence")
            self.counts["pcm_mismatch_count"] += 1
            return
        submits = {}
        streams = {}
        finals = {}
        for path in sorted(submit_root.glob("*.msgpack")):
            message = msgpack.unpackb(path.read_bytes(), raw=False, strict_map_key=False)
            if message.get("type") != "submit":
                continue
            request_id = message.get("request_id")
            if request_id in submits:
                self.fail("duplicate native TTS submit")
            submits[request_id] = message["data"]["request"].get("inputs", {})
        for path in sorted(output_root.glob("*.msgpack")):
            packed = path.read_bytes()
            message = msgpack.unpackb(packed, raw=False, strict_map_key=False)
            request_id = message.get("request_id")
            if message.get("type") == "stream":
                chunk = message.get("chunk") or {}
                audio = chunk.get("audio")
                if not isinstance(audio, bytes) or not audio:
                    self.fail("native TTS stream lacks audio")
                    continue
                streams.setdefault(request_id, []).append((chunk.get("index"), audio))
            elif message.get("type") == "complete":
                finals.setdefault(request_id, []).append(message)
        parents = {}
        for request_id in set(streams) | set(finals):
            match = re.fullmatch(r"(.+):tts(?::(\d+))?", request_id)
            if not match:
                # Stage abort broadcasts can produce a parent-only, no-audio
                # acknowledgement. It is a control record, not a TTS model leg.
                if not streams.get(request_id) and all(
                        not final.get("success") and final.get("error") == "request aborted"
                        and not final.get("result") for final in finals.get(request_id, [])):
                    continue
                self.fail("native TTS request lacks public parent attribution")
                continue
            parent, leg = match.group(1), int(match.group(2) or 0)
            chunks = sorted(streams.get(request_id, []), key=lambda item: item[0])
            if [index for index, _ in chunks] != list(range(len(chunks))):
                self.fail("native TTS chunk index gap/reorder")
                self.counts["pcm_mismatch_count"] += 1
                continue
            if len(finals.get(request_id, [])) != 1:
                self.fail("native TTS terminal is missing or duplicated")
                self.counts["pcm_mismatch_count"] += 1
                continue
            final = finals[request_id][0]
            native = b"".join(raw for _, raw in chunks)
            if not final.get("success"):
                # Cancelled native work can have a valid public PCM prefix but no
                # successful terminal WAV. Retain its exact chunks for prefix checks.
                if chunks:
                    parents.setdefault(parent, []).append({"leg": leg, "request_id": request_id,
                                                           "chunks": chunks, "pcm": native,
                                                           "successful_terminal": False})
                continue
            result = final.get("result") or {}
            try:
                with wave.open(io.BytesIO(result.get("data", b"")), "rb") as wav:
                    if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 24000):
                        raise ValueError("invalid native WAV")
                    wav_pcm = wav.readframes(wav.getnframes())
            except Exception as error:
                self.fail(f"invalid native TTS WAV: {error}")
                self.counts["pcm_mismatch_count"] += 1
                continue
            if not chunks or native != wav_pcm:
                self.fail("native TTS chunks differ from terminal WAV")
                self.counts["pcm_mismatch_count"] += 1
                continue
            parents.setdefault(parent, []).append({"leg": leg, "request_id": request_id,
                                                   "chunks": chunks, "pcm": native})
        for values in parents.values():
            values.sort(key=lambda item: item["leg"])
            if [item["leg"] for item in values] != list(range(len(values))):
                self.fail("native TTS parent has a leg gap")
                self.counts["pcm_mismatch_count"] += 1
        self.native_parents = parents

        def mismatch(message):
            self.fail(message)
            self.counts["pcm_mismatch_count"] += 1

        for key, observed in self.verified.items():
            record = next(item for item in self.requests if item["request_key"] == key)
            parent = observed["request_id"]
            children = parents.get(parent, [])
            if record["transport"] == "http":
                public = observed.get("public_audio", [])
                if not public and not children:
                    continue
                cursor = 0
                if record["status"] == "cancelled":
                    native_chunks = [(index, raw) for child in children
                                     for index, raw in child["chunks"]]
                    if len(public) > len(native_chunks):
                        mismatch("cancelled public PCM exceeds retained native chunks")
                    for item, (index, native) in zip(public, native_chunks):
                        if item["stream_index"] != index or item["raw"] != native:
                            mismatch("cancelled public PCM is not the exact native prefix")
                    if public:
                        self.native_pcm_exact["http"] += 1
                    continue
                for child in children:
                    count = len(child["chunks"])
                    selected = public[cursor:cursor + count]
                    if len(selected) != count:
                        mismatch("public PCM is missing a native TTS leg/chunk")
                        break
                    for index, (_, native) in enumerate(child["chunks"]):
                        item = selected[index]
                        if item["stream_index"] != index or item["raw"] != native:
                            mismatch("public/native HTTP PCM chunk mismatch")
                    cursor += count
                    self.native_pcm_exact["http"] += 1
                if cursor != len(public):
                    mismatch("unowned public HTTP PCM remains")
            else:
                detail = self.realtime[key]
                groups = {}
                for row in detail["audio_rows"]:
                    event = row["event"]
                    raw = base64.b64decode(event["audio"], validate=True)
                    groups.setdefault(event.get("text_cursor"), []).append(raw)
                ordered = [groups[cursor] for cursor in sorted(groups)]
                if record["status"] == "cancelled":
                    native_prefix = b"".join(raw for child in children for _, raw in child["chunks"])
                    if not detail["pcm"] or len(detail["pcm"]) > len(native_prefix) or \
                            native_prefix[:len(detail["pcm"])] != detail["pcm"]:
                        mismatch("cancelled realtime PCM is not the exact native byte prefix")
                    else:
                        self.native_pcm_exact["realtime"] += 1
                    continue
                if len(ordered) > len(children):
                    mismatch("realtime public PCM has no native leg")
                    continue
                for index, public_rows in enumerate(ordered):
                    public_pcm = b"".join(public_rows)
                    native_pcm = b"".join(raw for _, raw in children[index]["chunks"])
                    if public_pcm != native_pcm:
                        mismatch("public/native realtime PCM leg mismatch")
                    else:
                        self.native_pcm_exact["realtime"] += 1
                if record["status"] == "completed" and len(ordered) != len(children):
                    mismatch("completed realtime response omitted a native TTS leg")
        self.coverage["native_public_pcm_exact"] = self.native_pcm_exact["http"] > 0 \
            and self.native_pcm_exact["realtime"] > 0
        self.coverage["native_pcm_join_counts"] = dict(self.native_pcm_exact)

    def verify_playback(self):
        acks_path = self.root / "playback-acks.jsonl"
        acks = read_jsonl(acks_path) if acks_path.is_file() else []
        delayed_path = self.root / "delayed-read.jsonl"
        delayed = read_jsonl(delayed_path) if delayed_path.is_file() else []
        partial = final = 0
        by_response = {}
        for ack in acks:
            response_id = ack["response_id"]
            capture = ack["capture_file"]
            if capture not in self.captures:
                self.fail("playback ACK references an unavailable raw capture")
                continue
            frames = self.captures[capture]
            if ack["consumption_ended_monotonic_ns"] < ack["consumption_started_monotonic_ns"] \
                    or ack["consumption_ended_monotonic_ns"] - ack["consumption_started_monotonic_ns"] \
                    < ack["minimum_consumption_ns"]:
                self.fail("playback ACK did not follow full digital PCM duration")
            tx = [row for row in frames if row["direction"] == "tx"
                  and row.get("event", {}).get("type") == "output_audio_buffer.playback_ack"
                  and row["event"].get("response_id") == response_id
                  and row["event"].get("text_cursor") == ack["text_cursor"]
                  and row["monotonic_ns"] >= ack["consumption_ended_monotonic_ns"]]
            if not tx:
                self.fail("no raw playback ACK TX followed virtual consumption")
                continue
            accepted = [row for row in frames if row["direction"] == "rx"
                        and row.get("event", {}).get("type") == "response.playback_ack"
                        and row["event"].get("response_id") == response_id
                        and row["event"].get("text_cursor") == ack["text_cursor"]
                        and row["monotonic_ns"] >= tx[0]["monotonic_ns"]]
            if not accepted or ack["accepted_monotonic_ns"] != accepted[0]["monotonic_ns"] \
                    or bool(accepted[0]["event"].get("final", False)) != bool(ack["final"]):
                self.fail("server playback ACK acceptance differs from raw capture")
            audio = [row for row in frames if row["direction"] == "rx"
                     and row.get("event", {}).get("type") == "audio.delta"
                     and row["event"].get("response_id") == response_id]
            by_index = {row["index"]: row for row in audio}
            issued = {row["event"].get("text_cursor") for row in frames
                      if row["direction"] == "rx" and row.get("event", {}).get("response_id") == response_id
                      and row["event"]["type"] in ("audio.delta", "audio.done")}
            if ack["text_cursor"] not in issued:
                self.fail("playback ACK cursor was never issued for this response")
            try:
                selected = [by_index[index] for index in ack["audio_capture_indices"]]
                pcm = b"".join(base64.b64decode(row["event"]["audio"], validate=True) for row in selected)
                if sha(pcm) != ack["pcm_sha256"] or len(pcm) != ack["pcm_bytes"]:
                    raise ValueError("PCM identity mismatch")
            except (KeyError, ValueError, binascii.Error) as error:
                self.fail(f"playback ACK audio binding failed: {error}")
            previous = by_response.get(response_id)
            if previous is not None and ack["text_cursor"] < previous:
                self.fail("accepted playback cursors decreased")
            by_response[response_id] = ack["text_cursor"]
            partial += not ack["final"]
            final += bool(ack["final"])
        delayed_ok = False
        for row in delayed:
            if row.get("actual_delay_ns", 0) >= row.get("requested_delay_ns", 0) >= 1_000_000_000 \
                    and row.get("capture_file") in self.captures:
                frames = self.captures[row["capture_file"]]
                if any(frame["index"] == row.get("first_audio_capture_index")
                       and frame.get("event", {}).get("type") == "audio.delta" for frame in frames):
                    delayed_ok = True
        self.coverage["playback_ack_after_virtual_consumption"] = partial > 0 and final > 0 \
            and not any("playback" in item or "PCM duration" in item for item in self.failures)
        self.coverage["delayed_read_observed"] = delayed_ok

    def verify_resources(self):
        baseline = self.run.get("baseline_resource")
        final = self.run.get("final_same_residency")
        if not baseline or not final:
            self.fail("same-residency baseline/final resource samples are missing")
            return 0.0, 0.0
        resource_path = self.root / "resources.jsonl"
        rows = read_jsonl(resource_path)
        # NVML may report host-namespace PIDs while ManagedProcess records the
        # container namespace. Never compare those integer domains directly.
        # Bind the topology instead: ASR+VLM=2 and one leased output stage adds
        # exactly 1. Any excess process sample fails; a crashed/replaced owned
        # process is independently caught by the lifecycle checks below.
        excess_samples = 0
        for row in rows:
            if row.get("stage_transition"):
                # Retiring/starting stages remain in raw evidence and in the
                # process-lifetime overlap checks; only stable topologies are
                # eligible for the excess-process gate.
                continue
            residency = row.get("residency")
            if residency not in ("asr+vlm", "asr+vlm+flux", "asr+vlm+tts", "asr+vlm+wan"):
                self.fail("resource sample has an unknown output-stage residency")
                continue
            expected = 2 + (0 if residency == "asr+vlm" else 1)
            observed = row.get("gpu0", {}).get("compute_processes", [])
            if len(observed) > expected:
                excess_samples += 1
        self.coverage["foreign_gpu_process_count"] = excess_samples
        self.coverage["gpu_pid_namespaces_compared_directly"] = False
        if excess_samples:
            self.fail("excess GPU 0 compute process appeared in retained resource samples")
        if baseline not in rows or final not in rows:
            self.fail("embedded resource summaries differ from raw resource JSONL")
        if baseline.get("residency") != "asr+vlm" or final.get("residency") != baseline.get("residency"):
            self.fail("resource samples do not use identical ASR+VLM residency")
        if baseline["gpu0"]["index"] != 0 or final["gpu0"]["index"] != 0:
            self.fail("resource sampler observed a non-GPU0 device")
        device = float(final["gpu0"]["memory_used_mib"] - baseline["gpu0"]["memory_used_mib"])
        rss = 0.0
        for name in ("server", "asr", "vlm"):
            before_process = baseline.get("processes", {}).get(name, {})
            after_process = final.get("processes", {}).get(name, {})
            if before_process.get("pid") != after_process.get("pid"):
                self.fail(f"same-process identity changed: {name}")
            before = before_process.get("rss_mib")
            after = after_process.get("rss_mib")
            if before is None or after is None:
                self.fail(f"same-process RSS baseline/final is missing: {name}")
            else:
                rss += after - before
        initial = self.run.get("gpu_initial")
        cleanup = self.run.get("gpu_final")
        if not initial or initial.get("index") != 0 or initial.get("memory_used_mib", 65) > 64 \
                or initial.get("utilization_percent") != 0 or initial.get("compute_processes"):
            self.fail("GPU 0 initial exclusivity proof failed")
        if not cleanup or cleanup.get("index") != 0 or cleanup.get("memory_used_mib", 65) > 64 \
                or cleanup.get("utilization_percent") != 0 or cleanup.get("compute_processes"):
            self.fail("GPU 0 final cleanup proof failed")
        self.coverage["gpu0_idle_after_cleanup"] = bool(cleanup) and not cleanup.get("compute_processes") \
            and cleanup.get("memory_used_mib", 65) <= 64 and cleanup.get("utilization_percent") == 0
        return device, rss

    def verify_identities(self):
        if self.run.get("evidence_class") == "SYNTHETIC_CONTRACT_ONLY":
            if not self.allow_synthetic:
                self.fail("synthetic evidence is not real model evidence")
            self.coverage["provenance_rehash"] = "SYNTHETIC_NOT_REAL"
            return
        for role, identity in self.run.get("binaries", {}).items():
            path = Path(identity.get("executable", ""))
            if not path.is_file() or file_sha(path) != identity.get("sha256") \
                    or path.stat().st_size != identity.get("bytes"):
                self.fail(f"binary identity cannot be rehashed exactly: {role}")
        for role, identity in self.run.get("models", {}).items():
            root = Path(identity.get("root", ""))
            actual = {}
            if root.is_dir():
                for path in sorted(item for item in root.rglob("*") if item.is_file()):
                    actual[path.relative_to(root).as_posix()] = {
                        "sha256": file_sha(path), "bytes": path.stat().st_size}
            if actual != identity.get("files"):
                self.fail(f"model tree identity differs: {role}")
            elif sha(canonical(actual)) != identity.get("tree_sha256"):
                self.fail(f"model tree digest mismatch: {role}")
            for name, expected in self.run.get("configs", {}).get(role, {}).items():
                path = root / name
                if not path.is_file() or file_sha(path) != expected:
                    self.fail(f"model config identity differs: {role}/{name}")
        harness = Path(__file__).with_name("ns07_six_way_soak.py")
        prereg = Path(__file__).parents[1] / "benchmarks/north-star/ns07-six-way-preregistration.json"
        if file_sha(harness) != self.run["source"].get("harness_sha256"):
            self.fail("executed driver differs from retained harness identity")
        if file_sha(prereg) != self.run["source"].get("preregistration_sha256"):
            self.fail("preregistration identity differs")
        source = Path(self.run.get("source_root", ""))
        try:
            commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
            tree = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD^{tree}"], text=True).strip()
            if commit != self.run["source"].get("execution_commit") \
                    or tree != self.run["source"].get("source_tree_sha256"):
                self.fail("source commit/tree differs from retained execution identity")
        except (OSError, subprocess.SubprocessError):
            self.fail("source identity cannot be revalidated")
        self.coverage["provenance_rehash"] = "EXACT"

    def verify_manifest(self):
        manifest = self.root / "SHA256SUMS"
        entries = {}
        for line in manifest.read_text(encoding="ascii").splitlines():
            if not line:
                continue
            expected, separator, name = line[:64], line[64:66], line[66:]
            if not HEX64.fullmatch(expected) or separator not in ("  ", " *"):
                raise ValueError("malformed manifest line")
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or name == "SHA256SUMS" or name in entries:
                raise ValueError("unsafe/duplicate/self-included manifest entry")
            entries[name] = expected
        actual = set()
        for path in self.root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"symlink evidence is forbidden: {path}")
            if path.is_file() and path.name != "SHA256SUMS":
                actual.add(path.relative_to(self.root).as_posix())
        if set(entries) != actual:
            raise ValueError("manifest does not cover exactly the evidence file set")
        for name, expected in entries.items():
            if file_sha(self.root / name) != expected:
                raise ValueError(f"manifest digest mismatch: {name}")
        normalized_path = self.root / "NS-07.json"
        normalized = read_json(normalized_path)
        unknown = set(normalized) - ALLOWED_NORMALIZED
        if unknown:
            raise ValueError(f"normalized result has unapproved fields: {sorted(unknown)}")
        if normalized.get("case_id") != "NS-07" or normalized.get("production_status") != "NOT_RUN":
            raise ValueError("normalized case/production boundary drift")
        metrics = normalized.get("metrics", {})
        for name in METRIC_NAMES:
            if not isinstance(metrics.get(name), (int, float)) or isinstance(metrics.get(name), bool) \
                    or not math.isfinite(metrics[name]):
                raise ValueError(f"normalized metric is missing/non-numeric: {name}")
        if normalized.get("raw_evidence", {}).get("run_sha256") != file_sha(self.root / "run.json"):
            raise ValueError("normalized run identity differs")
        if normalized["raw_evidence"].get("manifest_file_count") != len(entries):
            raise ValueError("normalized manifest file count differs")
        self.coverage["manifest"] = {"file_count": len(entries),
                                    "sha256": file_sha(manifest), "self_excluded": True}

    def evaluate(self):
        requests_path = self.root / "requests.jsonl"
        self.requests = read_jsonl(requests_path) if requests_path.is_file() else []
        events_path = self.root / "events.jsonl"
        event_rows = read_jsonl(events_path) if events_path.is_file() else []
        events_by_request = {}
        for row in event_rows:
            events_by_request.setdefault(row.get("request_key"), []).append(row)
        keys = set()
        response_owners = {}
        for record in self.requests:
            key = record.get("request_key")
            if key in keys or record.get("lane") not in LANES:
                self.fail("duplicate request key or unknown lane")
                continue
            keys.add(key)
            try:
                if record.get("transport") == "http":
                    observed = self.verify_http(record, events_by_request.get(key, []))
                elif record.get("transport") == "websocket":
                    observed = self.verify_realtime(record)
                    owner = response_owners.setdefault(record["response_id"], record["capture_file"])
                    if owner != record["capture_file"]:
                        self.counts["cross_session_event_count"] += 1
                else:
                    raise ValueError("unknown request transport")
            except Exception as error:
                self.fail(f"{key}: {type(error).__name__}: {error}")
                observed = {"ok": False, "status": "error", "request_id": record.get("request_id"),
                            "public_audio": []}
            self.verified[key] = observed
        measured = [record for record in self.requests if not record.get("warmup")]
        active = {record["lane"] for record in measured}
        request_count = len(measured)
        completed = sum(not record.get("warmup") and
                        self.verified.get(record["request_key"], {}).get("status") == "completed" and
                        self.verified.get(record["request_key"], {}).get("ok") for record in self.requests)
        cancelled = sum(not record.get("warmup") and
                        self.verified.get(record["request_key"], {}).get("status") == "cancelled"
                        for record in self.requests)
        expected_faults = 0
        unexpected = 0
        recoveries = recovery_success = 0
        for record in self.requests:
            observed = self.verified.get(record["request_key"], {"ok": False, "status": "error"})
            if expected_fault_ok(record, observed["status"], observed["ok"]):
                expected_faults += 1
            elif not observed["ok"] or record.get("unexpected_error"):
                unexpected += 1
            if record.get("recovery_attempt"):
                recoveries += 1
                if observed["ok"] and observed["status"] == "completed":
                    recovery_success += 1
        self.verify_processes()
        self.verify_native_pcm()
        self.verify_playback()
        device, rss = self.verify_resources()
        self.verify_identities()
        started = self.run.get("window_started_monotonic_ns")
        ended = self.run.get("window_ended_monotonic_ns")
        elapsed = (ended - started) / 1e9 if isinstance(started, int) and isinstance(ended, int) else 0.0
        latencies = [float(record["latency_ms"]) for record in measured
                     if isinstance(record.get("latency_ms"), (int, float))
                     and not isinstance(record.get("latency_ms"), bool)
                     and math.isfinite(record["latency_ms"]) and record["latency_ms"] >= 0]
        if len(latencies) != len(measured):
            self.fail("request latency coverage is incomplete/non-finite")
        final_health = self.run.get("final_health") or {}
        metrics = {"elapsed_s": elapsed, "client_count": len(active),
            "request_count": request_count, "completed_count": completed,
            "cancelled_count": cancelled, "expected_fault_count": expected_faults,
            "unexpected_error_count": unexpected, **self.counts,
            "recovery_attempt_count": recoveries, "recovery_success_count": recovery_success,
            "final_in_flight": final_health.get("in_flight") if isinstance(final_health.get("in_flight"), (int, float)) else -1,
            "device_growth_mib": device, "rss_growth_mib": rss,
            "latency_p95_ms": percentile(latencies)}
        kinds = {(record["lane"], record["kind"]) for record in self.requests}
        ok_key = lambda lane, kind: any(not record.get("warmup") and
            record["lane"] == lane and record["kind"] == kind
            and self.verified.get(record["request_key"], {}).get("ok") for record in self.requests)
        self.coverage.update({
            "all_six_lanes_active": active == LANES,
            "text_code": ok_key("text_code", "text"),
            "interleaved_visual": any(record["lane"] == "interleaved_visual"
                and record["kind"] == "narrative" and self.verified[record["request_key"]]["ok"]
                and {a.get("media_type") for a in record.get("artifacts", [])} >= {"image", "video", "audio"}
                for record in self.requests),
            "natural_video_input": ok_key("natural_video_input", "natural_video"),
            "abort_retry": ok_key("abort_retry", "random_abort") and ok_key("abort_retry", "abort_valid_retry"),
            "malformed_audio": ok_key("malformed_recovery", "malformed_audio")
                and ok_key("malformed_recovery", "valid_audio_recovery"),
            "malformed_dimensions": ok_key("malformed_recovery", "malformed_dimensions")
                and ok_key("malformed_recovery", "valid_dimension_recovery"),
            "full_duplex_barge_in": ok_key("full_duplex_voice", "realtime_voice")
                and any(record["lane"] == "full_duplex_voice" and record.get("recovery_attempt")
                        and self.verified[record["request_key"]]["ok"] for record in self.requests),
            "physical_playback": "NOT_RUN", "perceptual_quality": "NOT_RUN",
            "production": "NOT_RUN"})
        hard = {"elapsed_at_least_1800": elapsed >= 1800 and self.run.get("duration_s") >= 1800,
            "client_count_exact": len(active) == 6, "all_six_lanes_active": active == LANES,
            "requests_present": request_count > 0 and completed > 0,
            "unexpected_error_count_zero": unexpected == 0,
            "process_crash_count_zero": self.counts["process_crash_count"] == 0,
            "cross_session_event_count_zero": self.counts["cross_session_event_count"] == 0,
            "stale_event_count_zero": self.counts["stale_event_count"] == 0,
            "duplicate_terminal_count_zero": self.counts["duplicate_terminal_count"] == 0,
            "pcm_mismatch_count_zero": self.counts["pcm_mismatch_count"] == 0,
            "recovery_complete": recoveries > 0 and recovery_success == recoveries,
            "final_in_flight_zero": metrics["final_in_flight"] == 0,
            "device_growth_bounded": device <= 0, "rss_growth_bounded": rss <= 64,
            "output_stage_overlap_zero": self.coverage.get("output_stage_overlap_count") == 0,
            "lease_violation_zero": self.coverage.get("lease_violation_count") == 0,
            "owned_processes_exited": self.coverage.get("owned_processes_exited") is True,
            "server_admission_parallel": self.coverage.get("server_max_in_flight", 0) > 1,
            "gpu0_idle_after_cleanup": self.coverage.get("gpu0_idle_after_cleanup") is True,
            "gpu0_exclusive_samples": self.coverage.get("foreign_gpu_process_count") == 0,
            "delayed_read_observed": self.coverage.get("delayed_read_observed") is True,
            "playback_ack_after_virtual_consumption": self.coverage.get("playback_ack_after_virtual_consumption") is True,
            "native_public_pcm_exact": self.coverage.get("native_public_pcm_exact") is True,
            "all_fault_families_covered": all(self.coverage.get(name) for name in
                ("abort_retry", "malformed_audio", "malformed_dimensions", "full_duplex_barge_in")),
            "all_modalities_covered": all(self.coverage.get(name) for name in
                ("text_code", "interleaved_visual", "natural_video_input")),
            "driver_completed": self.run.get("driver_error") is None and not self.run.get("lane_errors"),
            "ports_isolated": bool(self.run.get("ports")) and self.run["ports"]["http"] != 43301
                and self.run["ports"]["realtime"] != 43301
                and self.run["ports"]["http"] != self.run["ports"]["realtime"],
            "provenance_exact": self.coverage.get("provenance_rehash") in ("EXACT", "SYNTHETIC_NOT_REAL")}
        for name, passed in hard.items():
            if not passed:
                self.fail(f"hard gate {name} failed")
        synthetic = self.run.get("evidence_class") == "SYNTHETIC_CONTRACT_ONLY"
        status = "CONTRACT_PASS" if synthetic and not self.failures else \
                 "ISOLATED_PASS" if not synthetic and not self.failures and all(hard.values()) else "FAIL"
        return {"schema": "umminfer-ns07-six-way-verification/v1", "status": status,
                "engineering_status": status, "production_status": "NOT_RUN",
                "evidence_class": self.run.get("evidence_class"), "metrics": metrics,
                "hard_gates": hard, "failures": self.failures, "coverage": self.coverage,
                "counts": {"requests": request_count, "events": len(event_rows),
                           "warmup_requests": len(self.requests) - len(measured),
                           "processes": len(self.run.get("processes", [])),
                           "realtime_responses": len(self.realtime)},
                "claim_boundary": "Synthetic fixtures are harness-only. Real isolated PASS never implies production, five-GPU, physical-acoustic, perceptual, macOS, or human acceptance."}

    def run_verification(self, verify_manifest=False):
        self.run = read_json(self.root / "run.json")
        if self.run.get("schema") != "umminfer-ns07-six-way-soak/v1" \
                or self.run.get("case_id") != "NS-07":
            raise ValueError("unknown NS-07 run schema")
        result = self.evaluate()
        if verify_manifest:
            self.verify_manifest()
            result["coverage"] = self.coverage
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-manifest", action="store_true")
    parser.add_argument("--allow-synthetic", action="store_true",
                        help="contract-test only; synthetic evidence can earn only CONTRACT_PASS")
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error("verification output already exists")
    if args.output == args.evidence.resolve() or args.evidence.resolve() in args.output.parents:
        parser.error("verification output must remain outside the evidence directory")
    try:
        result = Verify(args.evidence, allow_synthetic=args.allow_synthetic).run_verification(
            verify_manifest=args.verify_manifest)
    except Exception as error:
        result = {"schema": "umminfer-ns07-six-way-verification/v1", "status": "FAIL",
                  "engineering_status": "FAIL", "production_status": "NOT_RUN",
                  "metrics": {}, "hard_gates": {},
                  "failures": [f"{type(error).__name__}: {error}"], "coverage": {}}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                      ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["engineering_status"],
                      "metrics": result.get("metrics", {}),
                      "failure_count": len(result.get("failures", []))}, sort_keys=True))
    return 0 if result["engineering_status"] in ("ISOLATED_PASS", "CONTRACT_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
