#!/usr/bin/env python3
"""CPU-only independent timing audit for retained Realtime WebSocket evidence.

The auditor is stdlib-only and does not import the live driver or its verifier.
It decodes WebSocket and MessagePack bytes, checks all input hashes before
analysis, derives protocol/history/timing facts, and writes a report only after
every gate passes. Timings describe the recorded digital loopback run, not a
physical playback device, WAN, or a new model/server execution.
"""

import argparse
import base64
from collections import defaultdict, deque
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

SCENARIOS = (
    "final_ack", "immediate_barge", "delayed_barge",
    "reconnect_playback", "reconnect_generation", "teardown_callbacks",
)
TERMINALS = {"response.completed", "response.failed", "response.cancelled"}
STALE = {
    "audio.delta", "audio.done", "response.text.delta", "response.playback_ack",
    "response.completed", "response.failed",
}
CUTINS = {
    "input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped",
    "response.cancelled", "transcription.completed",
}
QUEUE_CONTRACT = {
    "outbound_frames": 1024, "partial_events": 4096,
    "unified_events": 4096, "max_audio_bytes": 64 * 1024 * 1024,
}
MAX_JSONL_BYTES = 64 * 1024 * 1024
MAX_FRAME_BYTES = 16 * 1024 * 1024


class AuditError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(str(message))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path):
    try:
        return json.loads(path.read_text("utf-8"), object_pairs_hook=_object_no_duplicates)
    except (OSError, UnicodeError, ValueError) as error:
        raise AuditError(f"invalid JSON {path.name}: {error}") from error


def json_lines(path):
    require(path.stat().st_size <= MAX_JSONL_BYTES, f"JSONL too large: {path.name}")
    with open(path, "rb") as source:
        for line_number, line in enumerate(source, 1):
            require(line.endswith(b"\n"), f"unterminated JSONL: {path.name}:{line_number}")
            try:
                yield json.loads(line, object_pairs_hook=_object_no_duplicates)
            except (ValueError, UnicodeError) as error:
                raise AuditError(f"invalid JSONL {path.name}:{line_number}: {error}") from error


def _safe_relative(name):
    path = Path(name)
    require(not path.is_absolute() and name and ".." not in path.parts and "." not in path.parts,
            f"unsafe manifest path: {name}")
    return path


def verify_manifest(root):
    manifest = root / "SHA256SUMS"
    expected = {}
    for line_number, line in enumerate(manifest.read_text("utf-8").splitlines(), 1):
        require(len(line) >= 67 and line[64:66] == "  ", f"bad SHA256SUMS line {line_number}")
        digest, name = line[:64], line[66:]
        require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                f"bad digest on SHA256SUMS line {line_number}")
        relative = _safe_relative(name)
        require(name not in expected, f"duplicate manifest path: {name}")
        expected[name] = digest
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"missing/symlinked evidence file: {name}")
        require(file_sha256(path) == digest, f"evidence hash mismatch: {name}")
    actual = {str(path.relative_to(root)) for path in root.rglob("*")
              if path.is_file() and path.name != "SHA256SUMS"}
    require(actual == set(expected), f"manifest file set mismatch: extra={sorted(actual-set(expected))} "
                                    f"missing={sorted(set(expected)-actual)}")
    require("SHA256SUMS" not in expected, "manifest must not include itself")
    return {
        "algorithm": "sha256", "file_count": len(expected),
        "manifest_sha256": file_sha256(manifest),
        "files": {name: {"sha256": digest, "bytes": (root / name).stat().st_size}
                  for name, digest in sorted(expected.items())},
    }


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & ((1 << 64) - 1)

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
        return value ^ (value >> 31)


def expected_plan(seed, session_count, round_count):
    rng = SplitMix64(seed)
    sessions = [{"session": session, "label": f"isolated-{session}",
                 "scenarios": [SCENARIOS[(session + r) % len(SCENARIOS)]
                               for r in range(round_count)]}
                for session in range(session_count)]
    rounds = []
    for round_index in range(round_count):
        order = list(range(session_count))
        for index in range(len(order) - 1, 0, -1):
            other = rng.next() % (index + 1)
            order[index], order[other] = order[other], order[index]
        rounds.append({
            "round": round_index, "start_order": order,
            "start_jitter_ms": {str(s): int(rng.next() % 17) for s in order},
            "ack_delay_ms": {str(s): int(5 + rng.next() % 55) for s in order},
            "foreign_ack_response": {str(s): f"foreign-{seed:x}-{round_index}-{s}" for s in order},
        })
    return {
        "schema": "umminfer-realtime-network-plan/v1", "seed": seed,
        "sessions": sessions, "rounds": rounds, "required_scenarios": list(SCENARIOS),
        "echo_ab": {"arms": ["protected", "disabled"], "same_active_response": True,
                    "identical_microphone_pcm": True, "identical_far_end_pcm": True,
                    "double_talk": True},
    }


def _frame_view(raw, direction):
    require(isinstance(raw, bytes) and len(raw) >= 2, "truncated WebSocket frame")
    first, second = raw[0], raw[1]
    require(first & 0x70 == 0, "WebSocket RSV bits are nonzero")
    fin, opcode = bool(first & 0x80), first & 0x0F
    require(opcode in (0, 1, 2, 8, 9, 10), f"reserved WebSocket opcode: {opcode}")
    require(bool(second & 0x80) == (direction == "tx"), "incorrect WebSocket mask direction")
    length = second & 0x7F
    offset = 2
    if length == 126:
        require(len(raw) >= 4, "truncated 16-bit WebSocket length")
        length = struct.unpack_from("!H", raw, offset)[0]
        offset += 2
        require(length >= 126, "non-canonical 16-bit WebSocket length")
    elif length == 127:
        require(len(raw) >= 10, "truncated 64-bit WebSocket length")
        length = struct.unpack_from("!Q", raw, offset)[0]
        offset += 8
        require(65536 <= length < (1 << 63), "invalid 64-bit WebSocket length")
    require(length <= MAX_FRAME_BYTES, "WebSocket frame exceeds audit bound")
    mask = None
    if second & 0x80:
        require(len(raw) >= offset + 4, "truncated WebSocket mask")
        mask = raw[offset:offset + 4]
        offset += 4
    require(len(raw) == offset + length, "WebSocket frame length mismatch")
    payload = raw[offset:]
    if mask is not None:
        payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    if opcode >= 8:
        require(fin and length <= 125, "fragmented/oversized WebSocket control frame")
    return fin, opcode, payload


def decode_transport_chunks(chunks, direction):
    """Decode arbitrarily split/coalesced bytes; reject trailing partial frames."""
    require(direction in ("rx", "tx"), "unknown WebSocket direction")
    buffer = bytearray()
    frames = []
    for chunk in chunks:
        require(isinstance(chunk, bytes), "transport chunk is not bytes")
        buffer.extend(chunk)
        require(len(buffer) <= MAX_FRAME_BYTES * 4, "unbounded partial WebSocket input")
        while buffer:
            if len(buffer) < 2:
                break
            length = buffer[1] & 0x7F
            header = 2 + (4 if buffer[1] & 0x80 else 0)
            if length == 126:
                header += 2
                if len(buffer) < 4:
                    break
                length = struct.unpack_from("!H", buffer, 2)[0]
            elif length == 127:
                header += 8
                if len(buffer) < 10:
                    break
                length = struct.unpack_from("!Q", buffer, 2)[0]
            require(length <= MAX_FRAME_BYTES, "WebSocket frame exceeds audit bound")
            if len(buffer) < header + length:
                break
            raw = bytes(buffer[:header + length])
            del buffer[:header + length]
            fin, opcode, payload = _frame_view(raw, direction)
            frames.append({"raw": raw, "fin": fin, "opcode": opcode, "payload": payload,
                           "raw_sha256": sha256(raw)})
    require(not buffer, "trailing partial WebSocket frame")
    return frames


def decode_capture(path):
    records = []
    fragments = {}
    previous_ns = -1
    for index, row in enumerate(json_lines(path)):
        require(set(row) == {"index", "monotonic_ns", "direction", "raw_base64"},
                f"unexpected capture schema: {path.name}:{index}")
        require(type(row["index"]) is int and row["index"] == index,
                f"non-contiguous capture index: {path.name}:{index}")
        monotonic_ns = row["monotonic_ns"]
        require(type(monotonic_ns) is int and monotonic_ns > previous_ns,
                f"non-monotonic capture clock: {path.name}:{index}")
        previous_ns = monotonic_ns
        direction = row["direction"]
        try:
            raw = base64.b64decode(row["raw_base64"], validate=True)
        except (ValueError, TypeError) as error:
            raise AuditError(f"invalid capture base64: {path.name}:{index}") from error
        frames = decode_transport_chunks([raw], direction)
        require(len(frames) == 1, f"capture row does not hold exactly one frame: {path.name}:{index}")
        frame = frames[0]
        record = {"capture_index": index, "monotonic_ns": monotonic_ns,
                  "direction": direction, "opcode": frame["opcode"],
                  "raw_sha256": frame["raw_sha256"]}
        opcode = frame["opcode"]
        if opcode not in (8, 9, 10):
            require(opcode in (0, 1), "binaryRealtime message is outside this audit")
            if opcode != 0:
                require(direction not in fragments, "nested fragmented WebSocket message")
                fragments[direction] = bytearray()
            require(direction in fragments, "continuation without initial frame")
            fragments[direction].extend(frame["payload"])
            require(len(fragments[direction]) <= MAX_FRAME_BYTES, "fragmented message too large")
            if frame["fin"]:
                payload = bytes(fragments.pop(direction))
                try:
                    event = json.loads(payload.decode("utf-8"), object_pairs_hook=_object_no_duplicates)
                except (ValueError, UnicodeError) as error:
                    raise AuditError(f"invalid realtime JSON at {path.name}:{index}: {error}") from error
                require(isinstance(event, dict) and isinstance(event.get("type"), str) and event["type"],
                        f"realtime event lacks a string type: {path.name}:{index}")
                record["event"] = event
                record["payload_sha256"] = sha256(payload)
        records.append(record)
    require(not fragments, f"unterminated fragmented message: {path.name}")
    require(records, f"empty capture: {path.name}")
    return records


class MessagePackReader:
    """Small strict decoder for the retained native submit envelope."""
    def __init__(self, data):
        self.data = data
        self.offset = 0

    def take(self, count):
        require(count >= 0 and self.offset + count <= len(self.data), "truncated MessagePack value")
        value = self.data[self.offset:self.offset + count]
        self.offset += count
        return value

    def number(self, count, fmt):
        return struct.unpack(fmt, self.take(count))[0]

    def value(self, depth=0):
        require(depth <= 64, "MessagePack nesting limit exceeded")
        marker = self.number(1, "!B")
        if marker <= 0x7F:
            return marker
        if marker >= 0xE0:
            return marker - 256
        if 0x80 <= marker <= 0x8F:
            return self.map(marker & 0x0F, depth)
        if 0x90 <= marker <= 0x9F:
            return self.array(marker & 0x0F, depth)
        if 0xA0 <= marker <= 0xBF:
            return self.text(marker & 0x1F)
        if marker == 0xC0:
            return None
        if marker == 0xC2:
            return False
        if marker == 0xC3:
            return True
        if marker in (0xC4, 0xC5, 0xC6):
            sizes = {0xC4: (1, "!B"), 0xC5: (2, "!H"), 0xC6: (4, "!I")}
            size, fmt = sizes[marker]
            return bytes(self.take(self.number(size, fmt)))
        if marker == 0xCA:
            return self.number(4, "!f")
        if marker == 0xCB:
            return self.number(8, "!d")
        integer_formats = {
            0xCC: (1, "!B"), 0xCD: (2, "!H"), 0xCE: (4, "!I"), 0xCF: (8, "!Q"),
            0xD0: (1, "!b"), 0xD1: (2, "!h"), 0xD2: (4, "!i"), 0xD3: (8, "!q"),
        }
        if marker in integer_formats:
            size, fmt = integer_formats[marker]
            return self.number(size, fmt)
        if marker in (0xD9, 0xDA, 0xDB):
            sizes = {0xD9: (1, "!B"), 0xDA: (2, "!H"), 0xDB: (4, "!I")}
            size, fmt = sizes[marker]
            return self.text(self.number(size, fmt))
        if marker in (0xDC, 0xDD):
            return self.array(self.number(2 if marker == 0xDC else 4,
                                            "!H" if marker == 0xDC else "!I"), depth)
        if marker in (0xDE, 0xDF):
            return self.map(self.number(2 if marker == 0xDE else 4,
                                        "!H" if marker == 0xDE else "!I"), depth)
        raise AuditError(f"unsupported MessagePack marker: 0x{marker:02x}")

    def text(self, length):
        try:
            return self.take(length).decode("utf-8")
        except UnicodeError as error:
            raise AuditError("invalid MessagePack UTF-8") from error

    def array(self, length, depth):
        require(length <= 1_000_000, "MessagePack array too large")
        return [self.value(depth + 1) for _ in range(length)]

    def map(self, length, depth):
        require(length <= 1_000_000, "MessagePack map too large")
        result = {}
        for _ in range(length):
            key = self.value(depth + 1)
            require(isinstance(key, str), "non-string MessagePack map key")
            require(key not in result, f"duplicate MessagePack key: {key}")
            result[key] = self.value(depth + 1)
        return result


def unpack_message(data):
    reader = MessagePackReader(data)
    value = reader.value()
    require(reader.offset == len(data), "trailing MessagePack bytes")
    return value


def normalize_messages(messages):
    require(isinstance(messages, list), "VLM messages are not an array")
    normalized = []
    for row in messages:
        require(isinstance(row, dict) and isinstance(row.get("role"), str), "bad VLM message role")
        content = row.get("content")
        if isinstance(content, list):
            require(len(content) == 1 and isinstance(content[0], dict) and
                    content[0].get("type") == "text" and isinstance(content[0].get("text"), str),
                    "VLM observer input is not exactly one text part")
            content = content[0]["text"]
        require(isinstance(content, str), "VLM message content is not text")
        normalized.append({"role": row["role"], "content": content})
    return normalized


def load_prompts(root):
    prompts = {}
    previous_ns = -1
    for index, row in enumerate(json_lines(root / "vlm-prompts.jsonl")):
        require(set(row) == {"observed_monotonic_ns", "request_id", "messages",
                             "raw_sha256", "raw_base64"}, f"bad prompt schema at row {index}")
        observed = row["observed_monotonic_ns"]
        require(type(observed) is int and observed > previous_ns, "non-monotonic VLM observer clock")
        previous_ns = observed
        request_id = row["request_id"]
        require(isinstance(request_id, str) and request_id and request_id not in prompts,
                "duplicate/empty VLM request id")
        raw = base64.b64decode(row["raw_base64"], validate=True)
        require(sha256(raw) == row["raw_sha256"], f"VLM raw hash mismatch: {request_id}")
        message = unpack_message(raw)
        require(isinstance(message, dict) and message.get("type") == "submit" and
                message.get("request_id") == request_id, f"VLM submit identity mismatch: {request_id}")
        try:
            messages = message["data"]["request"]["inputs"]["messages"]
        except (KeyError, TypeError) as error:
            raise AuditError(f"VLM submit has no messages: {request_id}") from error
        normalized = normalize_messages(messages)
        require(row["messages"] == normalized, f"observer normalization mismatch: {request_id}")
        response_id = request_id[:-4] if request_id.endswith(":vlm") else request_id
        require(response_id and response_id not in {p["response_id"] for p in prompts.values()},
                f"ambiguous VLM response identity: {response_id}")
        prompts[request_id] = {"request_id": request_id, "response_id": response_id,
                               "observed_monotonic_ns": observed, "messages": normalized,
                               "raw_sha256": row["raw_sha256"],
                               "messages_sha256": sha256(json.dumps(
                                   normalized, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode("utf-8"))}
    return prompts


def _integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f"invalid integer {label}")
    return value


def _duration_samples(name, start, end, context):
    require(type(start) is int and type(end) is int and end >= start,
            f"negative/non-integer timing {name}: {context}")
    return end - start


def _number(value):
    return int(value) if value.denominator == 1 else float(value)


def _quantile(sorted_values, numerator, denominator=1):
    require(sorted_values, "quantile of empty sample")
    position = Fraction(numerator * (len(sorted_values) - 1), denominator)
    lower = position.numerator // position.denominator
    upper = math.ceil(position)
    fraction = position - lower
    return Fraction(sorted_values[lower]) * (1 - fraction) + Fraction(sorted_values[upper]) * fraction


def summarize(samples, unit="ns"):
    values = sorted(sample["value"] for sample in samples)
    count = len(values)
    require(count > 0, "empty metric summary")
    q1 = _quantile(values, 1, 4)
    median = _quantile(values, 1, 2)
    q3 = _quantile(values, 3, 4)
    p95 = _quantile(values, 95, 100)
    deviations = sorted(abs(Fraction(value) - median) for value in values)
    mad = _quantile(deviations, 1, 2)
    iqr = q3 - q1
    low, high = q1 - Fraction(3, 2) * iqr, q3 + Fraction(3, 2) * iqr
    enriched = []
    for sample in samples:
        row = dict(sample)
        row["outlier"] = bool(Fraction(sample["value"]) < low or Fraction(sample["value"]) > high)
        if unit == "ns":
            row["duration_ms"] = sample["value"] / 1_000_000
        enriched.append(row)
    stats = {"count": count, "min": _number(Fraction(values[0])), "q1": _number(q1),
             "median": _number(median), "p95": _number(p95), "q3": _number(q3),
             "max": _number(Fraction(values[-1])), "iqr": _number(iqr), "mad": _number(mad)}
    result = {"unit": unit, "summary": stats, "samples": enriched,
              "outliers": [sample for sample in enriched if sample["outlier"]]}
    if unit == "ns":
        result["summary_ms"] = {"count": count}
        result["summary_ms"].update({key: value / 1_000_000 for key, value in stats.items()
                                     if key != "count"})
    return result


class Metrics:
    def __init__(self):
        self.values = {"global": defaultdict(list), "by_session": defaultdict(lambda: defaultdict(list)),
                       "by_scenario": defaultdict(lambda: defaultdict(list))}

    def add(self, name, value, context, unit="ns"):
        sample = dict(context)
        sample["value"] = value
        if unit == "ns":
            require(type(value) is int and value >= 0, f"bad duration for {name}")
            sample["start_monotonic_ns"] = context["start_monotonic_ns"]
            sample["end_monotonic_ns"] = context["end_monotonic_ns"]
            require(sample["end_monotonic_ns"] - sample["start_monotonic_ns"] == value,
                f"clock/duration mismatch for {name}")
        self.values["global"][name].append(sample)
        self.values["by_session"][str(context["session"])][name].append(sample)
        self.values["by_scenario"][context["scenario"]][name].append(sample)

    def duration(self, name, start, end, context):
        value = _duration_samples(name, start, end, context)
        timing = dict(context)
        timing.update({"start_monotonic_ns": start, "end_monotonic_ns": end})
        self.add(name, value, timing)

    def render(self):
        global_metrics = {name: summarize(samples, self.unit(name, samples))
                          for name, samples in sorted(self.values["global"].items())}

        def grouped(metric):
            # Global metrics retain every exact sample. Groups retain their full
            # outlier sets but not a second copy of non-outlier samples; each
            # group sample is identifiable by session/scenario/response/clocks.
            return {"unit": metric["unit"], "summary": metric["summary"],
                    **({"summary_ms": metric["summary_ms"]} if "summary_ms" in metric else {}),
                    "outliers": metric["outliers"]}

        by_session = {session: {name: grouped(summarize(samples, self.unit(name, samples)))
                                for name, samples in sorted(metrics.items())}
                      for session, metrics in sorted(self.values["by_session"].items())}
        by_scenario = {scenario: {name: grouped(summarize(samples, self.unit(name, samples)))
                                  for name, samples in sorted(metrics.items())}
                       for scenario, metrics in sorted(self.values["by_scenario"].items())}
        outliers = [{"metric": name, **sample} for name, metric in global_metrics.items()
                    for sample in metric["outliers"]]
        return {"definitions": {
                    "percentile": "linear interpolation at (n-1)*p; exact integer-ns inputs",
                    "iqr": "q3-q1", "mad": "median absolute deviation from the median",
                    "outliers": "outside [q1-1.5*iqr,q3+1.5*iqr]; retained in every summary",
                    "group_samples": "filter global samples by session/scenario; grouped outliers are complete",
                    "clock": "original monotonic_ns values; durations computed in integer ns",
                    "physical_playback_device_latency": "NOT_COVERED",
                }, "global": global_metrics, "by_session": by_session,
                "by_scenario": by_scenario, "global_outliers": outliers}

    @staticmethod
    def unit(name, samples):
        if name.endswith("_bytes") or name.endswith("_percent") or name.endswith("_time_ms"):
            return name.rsplit("_", 1)[-1] if name.endswith("_percent") else (
                "bytes" if name.endswith("_bytes") else "ms")
        if name.endswith("_cursor_bytes"):
            return "bytes"
        return "ns"


def _event_context(response):
    return {"session": response["session"], "scenario": response["scenario"],
            "connection": response["connection"], "response_id": response["response_id"]}


def _base_response(connection, response_id, segment, first_ns):
    return {"response_id": response_id, "connection": connection["name"],
            "session": connection["session"], "segment_id": segment,
            "first_event_monotonic_ns": first_ns, "text_deltas": [], "audio": [],
            "audio_chunk_sha256": [], "audio_done": [], "terminals": [],
            "accepted_cursors": [], "scenario": "unassigned", "round": None}


def assign_scenarios(connections, responses, run, profile):
    if profile != "task41-r3":
        for response in responses.values():
            response["scenario"] = "fixture"
        return
    by_session = defaultdict(list)
    for response in responses.values():
        by_session[response["session"]].append(response)
    for session, rows in by_session.items():
        rows.sort(key=lambda row: (row["first_event_monotonic_ns"], row["response_id"]))
        scenario_rows = sorted(run["scenario_results"][str(session)], key=lambda row: row["round"])
        require(rows and len(scenario_rows) == 3, f"missing scenario chain for session {session}")
        current = rows[0]
        for scenario_row in scenario_rows:
            current["scenario"] = scenario_row["scenario"]
            current["round"] = scenario_row["round"]
            result = responses.get(scenario_row["response_id"])
            require(result is not None and result["session"] == session and
                    result["connection"] == scenario_row["connection"] and
                    result["first_event_monotonic_ns"] >= current["first_event_monotonic_ns"],
                    f"scenario result chain mismatch: {scenario_row}")
            current = result
        current["scenario"] = "post_round_final"
    echo = run["echo_ab"]
    special = {
        echo["response_id"]: "echo_ab",
        echo["double_talk_response_ids"]["true"]: "double_talk_protected",
        echo["double_talk_response_ids"]["false"]: "double_talk_disabled",
    }
    for response_id, scenario in special.items():
        require(response_id in responses, f"missing special response {response_id}")
        responses[response_id]["scenario"] = scenario
    for response_id, scenario in ((echo["response_id"], "echo_disabled_followup"),
                                  (echo["double_talk_response_ids"]["true"],
                                   "double_talk_protected_followup"),
                                  (echo["double_talk_response_ids"]["false"],
                                   "double_talk_disabled_followup")):
        origin = responses[response_id]
        candidates = [row for row in by_session[origin["session"]]
                      if row["first_event_monotonic_ns"] > origin["first_event_monotonic_ns"]]
        require(candidates, f"missing follow-up for {response_id}")
        candidates[0]["scenario"] = scenario
    require(all(row["scenario"] != "unassigned" for row in responses.values()), "unassigned response scenario")


def analyze(root, profile):
    manifest = verify_manifest(root)
    run = load_json(root / "run.json")
    plan = load_json(root / "plan.json")
    require(run.get("schema") == "umminfer-realtime-network-live/v1", "unknown run schema")
    require("driver_error" not in run, "retained driver failed")
    require(run.get("physical_microphone_speaker") == "NOT_COVERED" and
            run.get("perceptual_listening_quality") == "NOT_COVERED", "physical scope changed")
    require(run.get("final_health", {}).get("in_flight") == 0, "final in_flight is nonzero")
    require(run.get("cleanup", {}).get("owned_processes_exited") is True and
            run["cleanup"].get("selected_gpus_idle") is True, "recorded cleanup failed")
    require(run.get("queue_contract") == QUEUE_CONTRACT, "production queue contract changed")
    session_count = len(plan.get("sessions", []))
    round_count = len(plan.get("rounds", []))
    require(plan == expected_plan(plan.get("seed"), session_count, round_count),
            "plan does not match deterministic seed")
    if profile == "task41-r3":
        require(session_count == 6 and round_count == 3, "Task 41 requires six sessions/three rounds")
        flat_scenarios = [name for row in plan["sessions"] for name in row["scenarios"]]
        require(all(flat_scenarios.count(name) == 3 for name in SCENARIOS), "scenario coverage changed")
    prompts = load_prompts(root)
    connections = []
    connection_by_name = {}
    for plan_session in plan["sessions"]:
        session = plan_session["session"]
        generations = run["connections_per_session"][str(session)]
        require(type(generations) is int and generations >= 1, "bad connection generation count")
        for generation in range(generations):
            name = f"session-{session}-connection-{generation}"
            connection = {"name": name, "session": session, "generation": generation,
                          "records": decode_capture(root / "frames" / f"{name}.jsonl")}
            connection["events"] = [row for row in connection["records"] if "event" in row]
            connections.append(connection)
            connection_by_name[name] = connection
    expected_capture_names = {connection["name"] for connection in connections}
    actual_capture_names = {path.stem for path in (root / "frames").glob("*.jsonl")}
    require(actual_capture_names == expected_capture_names, "capture file set differs from run topology")

    responses = {}
    transcripts = {}
    all_identity_kinds = defaultdict(set)
    all_connection_identities = set()
    for connection in connections:
        connection_transcripts = {}
        transcripts[connection["name"]] = connection_transcripts
        for record in connection["events"]:
            event = record["event"]
            kind = event["type"]
            direction = record["direction"]
            response_id = event.get("response_id")
            if response_id is not None:
                require(isinstance(response_id, str) and response_id, "empty response identity")
                all_identity_kinds[response_id].add(f"{direction}:{kind}")
                all_connection_identities.add((connection["name"], response_id))
            if kind == "transcription.completed" and direction == "rx":
                segment = _integer(event.get("segment_id"), "transcription segment")
                require(segment not in connection_transcripts, "duplicate transcription segment")
                require(isinstance(event.get("text"), str) and event["text"].strip(),
                        "empty ASR final")
                connection_transcripts[segment] = {"text": event["text"], "record": record,
                                                   "response_id": None}
            if direction != "rx" or response_id is None:
                continue
            segment = event.get("segment_id")
            if response_id not in responses:
                require(segment is not None, f"response lacks initial segment: {response_id}")
                responses[response_id] = _base_response(connection, response_id, segment, record["monotonic_ns"])
            response = responses[response_id]
            require(response["connection"] == connection["name"],
                f"cross-session/connection response leak: {response_id}")
            if segment is not None:
                require(response["segment_id"] == segment, f"response segment changed: {response_id}")
            if kind == "response.text.delta":
                text = event.get("text")
                require(isinstance(text, str) and text, "empty/non-text VLM delta")
                response["text_deltas"].append({"text": text, "record": record})
                response["text"] = "".join(row["text"] for row in response["text_deltas"])
                require(event.get("text_cursor") == len(response["text"].encode("utf-8")),
                        f"text cursor is not cumulative UTF-8 length: {response_id}")
            elif kind == "audio.delta":
                try:
                    pcm = base64.b64decode(event.get("audio", ""), validate=True)
                except (ValueError, TypeError) as error:
                    raise AuditError(f"invalid PCM base64: {response_id}") from error
                require(pcm and len(pcm) % 2 == 0 and event.get("format") == "pcm16" and
                        event.get("sample_rate") == 24000 and event.get("channels") == 1,
                        f"invalid PCM/metadata: {response_id}")
                cursor = _integer(event.get("text_cursor"), "audio text cursor")
                end_cursor = _integer(event.get("text_end_cursor"), "audio end cursor")
                require(cursor <= end_cursor, f"decreasing audio cursor range: {response_id}")
                response["audio"].append({"pcm": pcm, "sha256": sha256(pcm), "cursor": cursor,
                                          "end_cursor": end_cursor, "record": record})
                response["audio_chunk_sha256"].append(sha256(pcm))
            elif kind == "audio.done":
                response["audio_done"].append({"cursor": _integer(
                    event.get("text_cursor"), "audio.done cursor"), "record": record})
            elif kind in TERMINALS:
                response["terminals"].append({"kind": kind, "record": record, "event": event})
        require(connection_transcripts, f"no ASR final on {connection['name']}")

    assign_scenarios(connections, responses, run, profile)
    for response in responses.values():
        response["text"] = "".join(row["text"] for row in response["text_deltas"])
        response["pcm"] = b"".join(row["pcm"] for row in response["audio"])
        response["pcm_sha256"] = sha256(response["pcm"])
        segment = response["segment_id"]
        require(segment in transcripts[response["connection"]],
                f"response has no initiating ASR final: {response['response_id']}")
        transcripts[response["connection"]][segment]["response_id"] = response["response_id"]
        kinds = [row["kind"] for row in response["terminals"]]
        require(all(kinds.count(kind) == 1 for kind in set(kinds)),
                f"duplicate terminal callback kind: {response['response_id']}")
        require("response.failed" not in kinds or len(kinds) == 1,
                f"failed response has another terminal: {response['response_id']}")
        require(len(response["audio_done"]) <= 1, f"duplicate audio.done: {response['response_id']}")
        cancel_indices = [row["record"]["capture_index"] for row in response["terminals"]
                          if row["kind"] == "response.cancelled"]
        if cancel_indices:
            cancel_index = cancel_indices[0]
            stale = [row for row in connection_by_name[response["connection"]]["events"]
                     if row["capture_index"] > cancel_index and row["direction"] == "rx" and
                     row["event"].get("response_id") == response["response_id"] and
                     row["event"]["type"] in STALE]
            require(not stale, f"stale event after cancellation: {response['response_id']}")
        completed = [row for row in response["terminals"] if row["kind"] == "response.completed"]
        if completed and response["audio_done"]:
            require(response["audio_done"][0]["record"]["capture_index"] <
                    completed[0]["record"]["capture_index"],
                    f"audio.done does not precede completion: {response['response_id']}")

    prompt_by_response = {}
    for prompt in prompts.values():
        response_id = prompt["response_id"]
        require(response_id in responses, f"prompt has no server response: {response_id}")
        require(response_id not in prompt_by_response, f"duplicate prompt for {response_id}")
        prompt_by_response[response_id] = prompt
    require(set(prompt_by_response) == set(responses), "server-response/native-prompt set mismatch")
    for response_id, prompt in prompt_by_response.items():
        response = responses[response_id]
        transcript = transcripts[response["connection"]][response["segment_id"]]["record"]
        # The forwarding observer and the client's ASR-frame reader are separate
        # threads. A forwarded VLM submit can legitimately be logged just before
        # the client records the already-emitted ASR final. Preserve that signed
        # offset instead of inventing a cross-thread ordering or shifting clocks.
        require(prompt["observed_monotonic_ns"] <= response["first_event_monotonic_ns"],
                f"VLM prompt observed after first response event: {response_id}")
        prompt["asr_final_capture_to_prompt_observed_ns"] = (
            prompt["observed_monotonic_ns"] - transcript["monotonic_ns"])
        response["prompt"] = prompt

    foreign_identities = sorted(set(all_identity_kinds) - set(responses))
    server_home_identities = {(response["connection"], rid) for rid, response in responses.items()}
    ack_only_extra_identities = all_connection_identities - server_home_identities
    for response_id in foreign_identities:
        require(all_identity_kinds[response_id] and
                all_identity_kinds[response_id] <= {"tx:output_audio_buffer.playback_ack"},
                f"foreign identity carried non-ACK events: {response_id}")

    metrics = Metrics()
    history = defaultdict(list)
    history_turns = []
    ack_pending = defaultdict(deque)
    ack_matches = []
    ack_ignored = []
    accepted_state = defaultdict(int)
    accepted_seen = defaultdict(set)
    issued_final = {rid: {row["cursor"] for row in response["audio"]} |
                         {row["cursor"] for row in response["audio_done"]}
                    for rid, response in responses.items()}
    ack_request_count = 0
    ack_schedule_wire = []

    for connection in connections:
        name = connection["name"]
        local_accepted = defaultdict(int)
        local_seen = defaultdict(set)
        local_issued = defaultdict(set)
        previous_ack_ns = {}
        previous_ack_cursor = {}
        for record in connection["events"]:
            event = record["event"]
            kind = event["type"]
            direction = record["direction"]
            rid = event.get("response_id")
            ns = record["monotonic_ns"]
            if direction == "rx" and kind == "audio.delta":
                local_issued[rid].add(event["text_cursor"])
            elif direction == "rx" and kind == "audio.done":
                local_issued[rid].add(event["text_cursor"])
            elif direction == "tx" and kind == "output_audio_buffer.playback_ack":
                cursor = _integer(event.get("text_cursor"), "TX playback cursor")
                ack_pending[(name, rid, cursor)].append({"record": record, "matched": False})
                ack_request_count += 1
            elif direction == "rx" and kind == "response.playback_ack":
                require(rid in responses and responses[rid]["connection"] == name,
                        f"ACK crossed connection: {rid}")
                cursor = _integer(event.get("text_cursor"), "RX playback cursor")
                require(cursor in local_issued[rid], f"server accepted unissued cursor: {rid}:{cursor}")
                require(cursor >= local_accepted[rid], f"server accepted decreasing cursor: {rid}")
                queue = ack_pending[(name, rid, cursor)]
                pending = next((row for row in queue if not row["matched"] and
                                row["record"]["monotonic_ns"] <= ns), None)
                require(pending is not None, f"unsolicited playback ACK: {rid}:{cursor}")
                pending["matched"] = True
                tx_ns = pending["record"]["monotonic_ns"]
                response = responses[rid]
                context = _event_context(response)
                context["text_cursor"] = cursor
                metrics.duration("playback_ack_round_trip", tx_ns, ns, context)
                if rid in previous_ack_ns:
                    metrics.duration("accepted_playback_ack_interarrival", previous_ack_ns[rid], ns, context)
                    cursor_delta = cursor - previous_ack_cursor[rid]
                    require(cursor_delta >= 0, "decreasing ACK progression")
                    metrics.add("accepted_playback_ack_cursor_delta_bytes", cursor_delta,
                                {**context, "start_monotonic_ns": previous_ack_ns[rid],
                                 "end_monotonic_ns": ns}, unit="bytes")
                previous_ack_ns[rid], previous_ack_cursor[rid] = ns, cursor
                local_accepted[rid] = cursor
                local_seen[rid].add(cursor)
                accepted_state[(name, rid)] = cursor
                accepted_seen[(name, rid)].add(cursor)
                response["accepted_cursors"].append(cursor)
                ack_matches.append({"connection": name, "response_id": rid, "text_cursor": cursor,
                                    "tx_monotonic_ns": tx_ns, "rx_monotonic_ns": ns,
                                    "final": event.get("final") is True})
                if event.get("final") is True:
                    require(cursor == len(response["text"].encode("utf-8")),
                            f"final ACK is not full UTF-8 text: {rid}")
                    turn = {"connection": name, "session": response["session"],
                            "scenario": response["scenario"], "response_id": rid,
                            "segment_id": response["segment_id"], "reason": "final_ack",
                            "user": transcripts[name][response["segment_id"]]["text"],
                            "assistant": response["text"], "text_cursor": cursor,
                            "monotonic_ns": ns}
                    history[name].append(turn)
                    history_turns.append(turn)
            elif direction == "rx" and kind == "response.cancelled":
                require(rid in responses, f"cancel for unknown response: {rid}")
                response = responses[rid]
                cursor = _integer(event.get("text_cursor"), "cancel cursor")
                acknowledged = event.get("acknowledged_text")
                require(isinstance(acknowledged, str) and event.get("reason") == "barge_in" and
                        event.get("history_recorded") is True, f"invalid barge-in cancellation: {rid}")
                require(cursor == local_accepted[rid], f"cancel cursor differs from ACK prefix: {rid}")
                text_bytes = response["text"].encode("utf-8")
                require(len(acknowledged.encode("utf-8")) == cursor and
                        text_bytes[:cursor].decode("utf-8") == acknowledged,
                        f"cancel prefix is not exact generated-text UTF-8: {rid}")
                turn = {"connection": name, "session": response["session"],
                        "scenario": response["scenario"], "response_id": rid,
                        "segment_id": response["segment_id"], "reason": "barge_in",
                        "user": transcripts[name][response["segment_id"]]["text"],
                        "assistant": acknowledged, "text_cursor": cursor, "monotonic_ns": ns}
                history[name].append(turn)
                history_turns.append(turn)
        for (pending_name, rid, cursor), queue in list(ack_pending.items()):
            if pending_name != name:
                continue
            for pending in queue:
                if pending["matched"]:
                    continue
                record = pending["record"]
                if rid not in responses:
                    category = "foreign_ack_only"
                elif responses[rid]["connection"] != name:
                    category = "stale_after_reconnect"
                elif cursor not in issued_final[rid]:
                    category = "unissued"
                elif cursor < local_accepted[rid]:
                    category = "decreasing"
                elif cursor in local_seen[rid]:
                    category = "duplicate_after_release"
                else:
                    raise AuditError(f"valid ACK was not acknowledged: {name}:{rid}:{cursor}")
                ack_ignored.append({"connection": name, "response_id": rid, "text_cursor": cursor,
                                    "category": category, "tx_monotonic_ns": record["monotonic_ns"]})

    # Independently derive every expected prompt from history at the observer timestamp.
    for response_id, response in responses.items():
        prompt = response["prompt"]
        prior_turns = [turn for turn in history[response["connection"]]
                       if turn["monotonic_ns"] <= prompt["observed_monotonic_ns"]]
        expected = []
        for turn in prior_turns:
            expected.extend([{"role": "user", "content": turn["user"]},
                             {"role": "assistant", "content": turn["assistant"]}])
        expected.append({"role": "user", "content": transcripts[
            response["connection"]][response["segment_id"]]["text"]})
        require(prompt["messages"] == expected,
                f"independently derived next-turn prompt mismatch: {response_id}")

    # ASR, VLM, TTS, TTFA and terminal timing samples.
    no_audio = []
    for connection in connections:
        name = connection["name"]
        for segment, transcript in sorted(transcripts[name].items()):
            rid = transcript["response_id"]
            require(rid in responses, f"transcription has no response: {name}:{segment}")
            response = responses[rid]
            final_ns = transcript["record"]["monotonic_ns"]
            prior = [row for row in connection["events"] if
                     row["capture_index"] < transcript["record"]["capture_index"]]
            stopped = next((row for row in reversed(prior) if row["direction"] == "rx" and
                            row["event"]["type"] == "input_audio_buffer.speech_stopped" and
                            row["event"].get("segment_id") == segment), None)
            started = next((row for row in reversed(prior) if row["direction"] == "rx" and
                            row["event"]["type"] == "input_audio_buffer.speech_started" and
                            row["event"].get("segment_id") == segment), None)
            require(stopped is not None and started is not None, f"missing ASR speech boundaries: {rid}")
            appends = [row for row in prior if row["direction"] == "tx" and
                       row["event"]["type"] == "input_audio_buffer.append"]
            require(appends, f"no input audio before ASR final: {rid}")
            context = _event_context(response)
            context["segment_id"] = segment
            metrics.duration("asr_speech_stop_to_final", stopped["monotonic_ns"], final_ns, context)
            metrics.duration("asr_last_append_to_final", appends[-1]["monotonic_ns"], final_ns, context)
            metrics.duration("asr_speech_start_to_final", started["monotonic_ns"], final_ns, context)
            partial = next((row for row in prior if row["direction"] == "rx" and
                            row["event"]["type"] == "transcription.segment" and
                            row["event"].get("segment_id") == segment), None)
            if partial is not None:
                metrics.duration("asr_speech_start_to_first_partial", started["monotonic_ns"],
                                 partial["monotonic_ns"], context)

    for response in responses.values():
        rid = response["response_id"]
        context = _event_context(response)
        prompt_ns = response["prompt"]["observed_monotonic_ns"]
        asr_ns = transcripts[response["connection"]][response["segment_id"]]["record"]["monotonic_ns"]
        text_rows = response["text_deltas"]
        if text_rows:
            metrics.duration("vlm_prompt_to_first_text", prompt_ns,
                             text_rows[0]["record"]["monotonic_ns"], context)
            metrics.duration("vlm_first_to_last_text", text_rows[0]["record"]["monotonic_ns"],
                             text_rows[-1]["record"]["monotonic_ns"], context)
        audio = response["audio"]
        if audio:
            first_audio_ns = audio[0]["record"]["monotonic_ns"]
            metrics.duration("ttfa_asr_final_to_first_audio", asr_ns, first_audio_ns, context)
            metrics.duration("ttfa_vlm_prompt_to_first_audio", prompt_ns, first_audio_ns, context)
            metrics.duration("tts_first_to_last_audio", first_audio_ns,
                             audio[-1]["record"]["monotonic_ns"], context)
            for previous, current in zip(audio, audio[1:]):
                metrics.duration("tts_audio_delta_interarrival", previous["record"]["monotonic_ns"],
                                 current["record"]["monotonic_ns"], context)
            for row in audio:
                require(row["end_cursor"] <= len(response["text"].encode("utf-8")),
                        f"audio end cursor exceeds final text: {rid}")
        else:
            terminal_kinds = {row["kind"] for row in response["terminals"]}
            reason = "cancelled_before_audio" if "response.cancelled" in terminal_kinds else \
                     "connection_closed_before_audio"
            no_audio.append({"response_id": rid, "session": response["session"],
                             "scenario": response["scenario"], "reason": reason})
        if response["audio_done"]:
            done_ns = response["audio_done"][0]["record"]["monotonic_ns"]
            require(audio, f"audio.done without audio.delta: {rid}")
            metrics.duration("tts_first_audio_to_done", audio[0]["record"]["monotonic_ns"], done_ns, context)
            if text_rows:
                metrics.duration("tts_last_text_to_done", text_rows[-1]["record"]["monotonic_ns"],
                                 done_ns, context)
        for terminal in response["terminals"]:
            metric = {"response.completed": "response_prompt_to_completed",
                      "response.cancelled": "response_prompt_to_cancelled",
                      "response.failed": "response_prompt_to_failed"}[terminal["kind"]]
            metrics.duration(metric, prompt_ns, terminal["record"]["monotonic_ns"], context)

    # Barge-in onset, final old audio, and cancellation callback ordering.
    cancel_start_by_response = {}
    for connection_name, schedules in run.get("cancel_schedules", {}).items():
        for schedule in schedules:
            if schedule.get("response_id") is not None:
                cancel_start_by_response[schedule["response_id"]] = (connection_name, schedule["monotonic_ns"])
    barge_rows = []
    for response in responses.values():
        cancelled = next((row for row in response["terminals"] if row["kind"] == "response.cancelled"), None)
        if cancelled is None:
            continue
        rid = response["response_id"]
        connection = connection_by_name[response["connection"]]
        cancel_record = cancelled["record"]
        cancel_ns = cancel_record["monotonic_ns"]
        old_transcript_index = transcripts[response["connection"]][response["segment_id"]]["record"]["capture_index"]
        new_speech = next((row for row in connection["events"] if
                           row["capture_index"] > old_transcript_index and row["direction"] == "rx" and
                           row["event"]["type"] == "input_audio_buffer.speech_started" and
                           row["event"].get("segment_id") == response["segment_id"] + 1), None)
        require(new_speech is not None, f"barge-in has no new speech onset: {rid}")
        config_indices = [row["capture_index"] for row in connection["events"] if
                          old_transcript_index < row["capture_index"] < new_speech["capture_index"] and
                          row["direction"] == "rx" and row["event"]["type"] == "transcription_session.updated"]
        lower = max([old_transcript_index] + config_indices)
        if rid in cancel_start_by_response:
            schedule_connection, schedule_ns = cancel_start_by_response[rid]
            require(schedule_connection == connection["name"], "cancel schedule crossed connection")
            # Trailing silence from the old turn can arrive after its ASR final.
            # The recorded barge schedule disambiguates that old tail from the
            # first append of the new utterance; the wire frame must still exist.
            first_append = next((row for row in connection["events"] if
                                 row["monotonic_ns"] >= schedule_ns and
                                 row["capture_index"] <= new_speech["capture_index"] and
                                 row["direction"] == "tx" and
                                 row["event"]["type"] == "input_audio_buffer.append"), None)
        else:
            first_append = next((row for row in connection["events"] if
                                 lower < row["capture_index"] <= new_speech["capture_index"] and
                                 row["direction"] == "tx" and
                                 row["event"]["type"] == "input_audio_buffer.append"), None)
        require(first_append is not None, f"barge-in has no initiating append: {rid}")
        context = _event_context(response)
        metrics.duration("barge_first_append_to_speech_started", first_append["monotonic_ns"],
                         new_speech["monotonic_ns"], context)
        metrics.duration("barge_first_append_to_cancel", first_append["monotonic_ns"], cancel_ns, context)
        if new_speech["monotonic_ns"] <= cancel_ns:
            metrics.duration("barge_speech_started_to_cancel", new_speech["monotonic_ns"], cancel_ns, context)
        else:
            metrics.duration("barge_cancel_to_speech_started", cancel_ns,
                             new_speech["monotonic_ns"], context)
        last_audio = response["audio"][-1] if response["audio"] else None
        row = {"response_id": rid, "session": response["session"], "scenario": response["scenario"],
               "first_append_monotonic_ns": first_append["monotonic_ns"],
               "speech_started_monotonic_ns": new_speech["monotonic_ns"],
               "cancel_monotonic_ns": cancel_ns,
               "first_append_to_cancel_ns": cancel_ns - first_append["monotonic_ns"],
               "post_cancel_stale_audio": 0}
        if last_audio is not None:
            audio_ns = last_audio["record"]["monotonic_ns"]
            require(audio_ns <= cancel_ns, f"old audio follows cancellation: {rid}")
            if audio_ns >= first_append["monotonic_ns"]:
                metrics.duration("barge_first_append_to_final_old_audio", first_append["monotonic_ns"],
                                 audio_ns, context)
            else:
                metrics.duration("barge_final_old_audio_to_first_append", audio_ns,
                                 first_append["monotonic_ns"], context)
            metrics.duration("barge_final_old_audio_to_cancel", audio_ns, cancel_ns, context)
            row.update({"final_old_audio_monotonic_ns": audio_ns,
                        "final_old_audio_to_cancel_ns": cancel_ns - audio_ns})
        else:
            row["final_old_audio"] = "NOT_OBSERVED_BEFORE_CANCEL"
        barge_rows.append(row)

    # Reconnect timing is socket generation timing, not physical playback drain.
    reconnects = []
    for session in sorted({row["session"] for row in connections}):
        generations = sorted((row for row in connections if row["session"] == session),
                             key=lambda row: row["generation"])
        for old, new in zip(generations, generations[1:]):
            old_responses = [row for row in responses.values() if row["connection"] == old["name"]]
            old_responses.sort(key=lambda row: row["first_event_monotonic_ns"])
            require(old_responses, f"no old response before reconnect: {old['name']}")
            active = old_responses[-1]
            close = next((row for row in reversed(old["records"]) if row["direction"] == "tx" and
                          row["opcode"] == 8), None)
            updated = next((row for row in new["events"] if row["direction"] == "rx" and
                            row["event"]["type"] == "transcription_session.updated"), None)
            first_append = next((row for row in new["events"] if row["direction"] == "tx" and
                                 row["event"]["type"] == "input_audio_buffer.append"), None)
            require(close is not None and updated is not None and first_append is not None,
                    f"incomplete reconnect boundaries: {old['name']}->{new['name']}")
            require(old["records"][-1]["monotonic_ns"] < new["records"][0]["monotonic_ns"],
                    f"overlapping/non-monotonic reconnect clocks: {old['name']}")
            close_ns = close["monotonic_ns"]
            context = _event_context(active)
            metrics.duration("reconnect_close_to_session_updated", close_ns,
                             updated["monotonic_ns"], context)
            metrics.duration("reconnect_close_to_first_new_append", close_ns,
                             first_append["monotonic_ns"], context)
            new_responses = sorted((row for row in responses.values() if row["connection"] == new["name"]),
                                   key=lambda row: row["first_event_monotonic_ns"])
            require(new_responses, f"no response after reconnect: {new['name']}")
            new_response = new_responses[0]
            new_asr_ns = transcripts[new["name"]][new_response["segment_id"]]["record"]["monotonic_ns"]
            metrics.duration("reconnect_close_to_first_new_asr_final", close_ns, new_asr_ns, context)
            row = {"session": session, "scenario": active["scenario"],
                   "old_connection": old["name"], "new_connection": new["name"],
                   "active_old_response_id": active["response_id"],
                   "close_monotonic_ns": close_ns,
                   "session_updated_monotonic_ns": updated["monotonic_ns"],
                   "close_to_session_updated_ns": updated["monotonic_ns"] - close_ns,
                   "close_to_first_new_asr_final_ns": new_asr_ns - close_ns,
                   "stale_old_response_events_on_new_connection": 0,
                   "physical_playback_drain": "NOT_COVERED"}
            if active["audio"]:
                metrics.duration("reconnect_final_old_audio_to_close",
                                 active["audio"][-1]["record"]["monotonic_ns"], close_ns, context)
                row["final_old_audio_to_close_ns"] = close_ns - active["audio"][-1]["record"]["monotonic_ns"]
            if new_response["audio"]:
                metrics.duration("reconnect_close_to_first_new_audio", close_ns,
                                 new_response["audio"][0]["record"]["monotonic_ns"], context)
                row["close_to_first_new_audio_ns"] = (
                    new_response["audio"][0]["record"]["monotonic_ns"] - close_ns)
            reconnects.append(row)

    # Read and bind the recorded ACK/cancel schedules without using their assertions.
    schedule_count = 0
    for connection_name, schedules in run.get("ack_schedules", {}).items():
        require(connection_name in connection_by_name, "ACK schedule names unknown connection")
        records = connection_by_name[connection_name]["events"]
        for schedule in schedules:
            require(set(schedule) >= {"monotonic_ns", "response_id", "text_cursor", "delay_ms", "label"},
                    "bad ACK schedule")
            candidates = [row for row in records if row["direction"] == "tx" and
                          row["event"]["type"] == "output_audio_buffer.playback_ack" and
                          row["event"].get("response_id") == schedule["response_id"] and
                          row["event"].get("text_cursor") == schedule["text_cursor"] and
                          row["monotonic_ns"] >= schedule["monotonic_ns"]]
            require(candidates, f"ACK schedule has no raw TX frame: {schedule}")
            wire = min(candidates, key=lambda row: row["monotonic_ns"])
            rid = schedule["response_id"]
            if rid in responses:
                response = responses[rid]
                if response["round"] is not None and schedule["label"] in ("delayed-partial", "delayed-final"):
                    expected_delay = plan["rounds"][response["round"]]["ack_delay_ms"][str(response["session"])]
                    require(schedule["delay_ms"] == expected_delay, "ACK delay differs from seeded plan")
                metrics.duration("recorded_ack_schedule_to_wire", schedule["monotonic_ns"],
                                 wire["monotonic_ns"], _event_context(response))
            schedule_count += 1
    cancel_schedule_count = 0
    for connection_name, schedules in run.get("cancel_schedules", {}).items():
        require(connection_name in connection_by_name, "cancel schedule names unknown connection")
        for schedule in schedules:
            rid = schedule.get("response_id")
            if rid is None:
                continue
            require(rid in responses, "cancel schedule names unknown response")
            response = responses[rid]
            cancelled = next(row for row in response["terminals"] if row["kind"] == "response.cancelled")
            require(cancelled["record"]["monotonic_ns"] >= schedule["monotonic_ns"],
                    "cancel precedes recorded schedule")
            cancel_schedule_count += 1
    if profile == "task41-r3":
        require(schedule_count == 147 and cancel_schedule_count == 6,
                "recorded schedule coverage changed")

    # Echo/double-talk anchors locate scenarios only; all outcomes are decoded again above.
    special_report = {"physical_audio": "NOT_COVERED"}
    if profile == "task41-r3":
        echo = run["echo_ab"]
        connection = connection_by_name[echo["connection"]]
        protected = [row for row in connection["events"] if row["direction"] == "rx" and
                     echo["protected_start_capture_index"] <= row["capture_index"] <
                     echo["protected_end_capture_index"] and row["event"]["type"] in CUTINS]
        require(not protected, "protected echo window contains a cut-in")
        disabled = [row for row in connection["events"] if
                    row["capture_index"] == echo["disabled_cancel_capture_index"] and
                    row["direction"] == "rx" and row["event"]["type"] == "response.cancelled" and
                    row["event"].get("response_id") == echo["response_id"]]
        require(len(disabled) == 1, "disabled echo cancellation not found")
        echo_pcm = responses[echo["response_id"]]["pcm"]
        require(sha256(echo_pcm) == echo["protected_far_end_sha256"] ==
                file_sha256(root / "echo-far-end-24k.pcm"), "echo far-end PCM mismatch")
        require(file_sha256(root / "echo-microphone-16k.pcm") == echo["protected_microphone_sha256"] ==
                echo["disabled_microphone_sha256"], "echo A/B microphone mismatch")
        double = {}
        for arm in ("true", "false"):
            rid = echo["double_talk_response_ids"][arm]
            index = echo["double_talk_cancel_capture_indices"][arm]
            require(any(row["capture_index"] == index and row["direction"] == "rx" and
                        row["event"]["type"] == "response.cancelled" and
                        row["event"].get("response_id") == rid for row in connection["events"]),
                    f"double-talk {arm} cancellation missing")
            far_path = root / f"double-talk-{arm.capitalize()}-far.pcm"
            near_path = root / f"double-talk-{arm.capitalize()}-near.pcm"
            require(file_sha256(far_path) == responses[rid]["pcm_sha256"],
                    f"double-talk {arm} far PCM mismatch")
            barge = next(row for row in barge_rows if row["response_id"] == rid)
            double[arm] = {"response_id": rid, "cancel_capture_index": index,
                           "first_append_to_cancel_ns": barge["first_append_to_cancel_ns"],
                           "near_sha256": file_sha256(near_path), "far_sha256": file_sha256(far_path),
                           "acknowledged_prefix": next(
                               turn["assistant"] for turn in history_turns if turn["response_id"] == rid)}
        special_report = {"protected_echo_cutins": 0, "disabled_echo_cancellations": 1,
                          "double_talk": double, "physical_audio": "NOT_COVERED"}

    # Re-hash retained PCM artifacts against independently concatenated wire PCM.
    pcm_artifacts = []
    for path in sorted((root / "pcm").glob("session-*/*.pcm")):
        match = path.stem.rsplit("-", 2)
        require(len(match) == 3, f"unexpected PCM artifact name: {path.name}")
        rid = f"{match[-2]}-{match[-1]}"
        require(rid in responses and file_sha256(path) == responses[rid]["pcm_sha256"],
                f"retained PCM artifact mismatch: {path.name}")
        pcm_artifacts.append({"path": str(path.relative_to(root)), "response_id": rid,
                              "sha256": file_sha256(path), "bytes": path.stat().st_size})

    # Process samples and queue/RSS bounds. Queue depth/saturation was not sampled.
    resource_rows = list(json_lines(root / "resources.jsonl"))
    require(resource_rows, "empty resource samples")
    previous_time = -math.inf
    per_process = defaultdict(list)
    owned_pids = {row["name"]: row["pid"] for row in run.get("owned_processes", [])}
    for index, row in enumerate(resource_rows):
        require(set(row) == {"time_ms", "processes"} and isinstance(row["time_ms"], (int, float)) and
                math.isfinite(row["time_ms"]) and row["time_ms"] > previous_time,
                f"non-monotonic resource clock at row {index}")
        previous_time = row["time_ms"]
        for name, sample in row["processes"].items():
            require(set(sample) == {"pid", "rss_bytes", "cpu_percent"} and
                    _integer(sample["rss_bytes"], "RSS") >= 0 and
                    (sample["cpu_percent"] is None or
                     (isinstance(sample["cpu_percent"], (int, float)) and
                      math.isfinite(sample["cpu_percent"]) and sample["cpu_percent"] >= 0)),
                    f"bad resource sample {name}:{index}")
            if name in owned_pids:
                require(sample["pid"] == owned_pids[name], f"resource pid changed: {name}")
            per_process[name].append({"time_ms": row["time_ms"], **sample})
    resource_report = {}
    for name, samples in sorted(per_process.items()):
        rss_values = [sample["rss_bytes"] for sample in samples]
        rss_samples = [{"session": "server", "scenario": "resource_samples", "sample_index": index,
                        "resource_time_ms": sample["time_ms"], "value": sample["rss_bytes"]}
                       for index, sample in enumerate(samples)]
        rss_summary = summarize(rss_samples, unit="bytes")
        rss_summary = {"unit": rss_summary["unit"], "summary": rss_summary["summary"],
                       "outliers": rss_summary["outliers"]}
        cpu = [sample["cpu_percent"] for sample in samples if sample["cpu_percent"] is not None]
        growth = max(rss_values) - rss_values[0]
        resource_report[name] = {"samples": len(samples), "rss": rss_summary,
                                 "rss_min_bytes": min(rss_values), "rss_max_bytes": max(rss_values),
                                 "rss_growth_from_first_to_max_bytes": growth,
                                 "cpu_percent_min": min(cpu) if cpu else None,
                                 "cpu_percent_max": max(cpu) if cpu else None,
                                 "cpu_percent_mean": sum(cpu) / len(cpu) if cpu else None}
        if name == "server":
            bound = run["resource_bounds"]["server_rss_growth_max_bytes"]
            require(growth <= bound, "server RSS growth exceeds recorded bound")
            recorded = run["resource_usage"]["server"]
            require(recorded["rss_growth_bytes"] == growth and
                    recorded["rss_min_bytes"] == min(rss_values) and
                    recorded["rss_max_bytes"] == max(rss_values) and
                    recorded["samples"] == len(samples), "server RSS summary differs from raw samples")
            resource_report[name]["rss_growth_bound_bytes"] = bound
            resource_report[name]["rss_growth_within_bound"] = True
    require("server" in resource_report, "no server resource samples")

    counts = {
        "sessions": session_count, "rounds": round_count, "connections": len(connections),
        "response_identities_both_directions": len(all_identity_kinds),
        "connection_scoped_response_identities": len(all_connection_identities),
        "server_responses": len(responses), "foreign_ack_only_identities": len(foreign_identities),
        "ack_only_non_server_connection_identities": len(ack_only_extra_identities),
        "foreign_ack_only_connection_identities": sum(
            rid not in responses for _, rid in all_connection_identities),
        "stale_server_response_connection_identities": sum(
            rid in responses for _, rid in ack_only_extra_identities),
        "native_vlm_prompts": len(prompts), "asr_finals": sum(len(v) for v in transcripts.values()),
        "terminal_callbacks": sum(len(row["terminals"]) for row in responses.values()),
        "completed_callbacks": sum(row["kind"] == "response.completed"
                                   for response in responses.values() for row in response["terminals"]),
        "cancelled_callbacks": sum(row["kind"] == "response.cancelled"
                                   for response in responses.values() for row in response["terminals"]),
        "history_turns": len(history_turns), "audio_delta_frames": sum(
            len(response["audio"]) for response in responses.values()),
        "playback_ack_requests": ack_request_count, "accepted_playback_acks": len(ack_matches),
        "ignored_playback_acks": len(ack_ignored), "reconnects": len(reconnects),
        "barge_in_cancellations": len(barge_rows), "recorded_ack_schedule_entries": schedule_count,
        "recorded_cancel_schedule_entries": cancel_schedule_count,
    }
    protocol = {
        "stale_events_after_cancel": 0, "cross_session_or_connection_response_leaks": 0,
        "accepted_decreasing_cursors": 0, "accepted_unissued_cursors": 0,
        "duplicate_terminal_callback_kinds": 0, "final_in_flight": 0,
        "unsolicited_playback_acks": 0, "non_monotonic_capture_clocks": 0,
        "completed_then_cancelled_responses": sum(
            {row["kind"] for row in response["terminals"]} == {"response.completed", "response.cancelled"}
            for response in responses.values()),
        "cancelled_only_responses": sum(
            [row["kind"] for row in response["terminals"]] == ["response.cancelled"]
            for response in responses.values()),
        "no_terminal_due_to_connection_close": sum(not response["terminals"] for response in responses.values()),
        "ignored_ack_categories": {category: sum(row["category"] == category for row in ack_ignored)
                                   for category in sorted({row["category"] for row in ack_ignored})},
    }
    if profile == "task41-r3":
        require(counts["response_identities_both_directions"] == 38 and
                counts["connection_scoped_response_identities"] == 41 and
                counts["server_responses"] == 30 and counts["foreign_ack_only_identities"] == 8 and
                counts["ack_only_non_server_connection_identities"] == 11 and
                counts["foreign_ack_only_connection_identities"] == 8 and
                counts["stale_server_response_connection_identities"] == 3 and
                counts["native_vlm_prompts"] == 30 and counts["terminal_callbacks"] == 27 and
                counts["connections"] == 15 and counts["history_turns"] == 21 and
                counts["audio_delta_frames"] == 1024 and counts["barge_in_cancellations"] == 9 and
                counts["reconnects"] == 9, f"Task 41 count gate failed: {counts}")
        retained = load_json(root / "independent-verification.json")
        require(retained["responses"] == counts["connection_scoped_response_identities"] and
                retained["verified_next_turn_prompts"] == counts["native_vlm_prompts"] and
                retained["terminal_callbacks"] == counts["terminal_callbacks"] and
                retained["connections"] == counts["connections"] and
                retained["final_history_turns"] == counts["history_turns"],
                "retained independent count reconciliation failed")
        for name, result in retained["connection_results"].items():
            for rid, hashes in result["audio_chunk_sha256"].items():
                require(rid in responses and responses[rid]["connection"] == name and
                        responses[rid]["audio_chunk_sha256"] == hashes,
                        f"retained PCM hash reconciliation failed: {name}:{rid}")
            for rid, cursor in result["accepted_cursors"].items():
                require(accepted_state[(name, rid)] == cursor,
                        f"retained ACK cursor reconciliation failed: {name}:{rid}")
    retained_reconciliation = {
        "retained_41_response_label": "41 connection-scoped response-ID entries in both directions",
        "globally_unique_response_ids": len(all_identity_kinds),
        "actual_server_responses": len(responses),
        "ack_only_non_server_connection_entries_in_retained_count": len(ack_only_extra_identities),
        "foreign_client_ACK_only_connection_entries": sum(
            rid not in responses for _, rid in all_connection_identities),
        "stale_server_ACK_only_connection_entries": sum(
            rid in responses for _, rid in ack_only_extra_identities),
        "globally_unique_foreign_ACK_only_ids": len(foreign_identities),
        "foreign_ids": foreign_identities,
        "consequence": "The historical '41 real responses' wording is not supported; "
                       "30 server responses match 30 native VLM prompts.",
    }

    response_report = []
    for response in sorted(responses.values(), key=lambda row: (row["session"], row["first_event_monotonic_ns"])):
        response_report.append({
            "response_id": response["response_id"], "session": response["session"],
            "scenario": response["scenario"], "round": response["round"],
            "connection": response["connection"], "segment_id": response["segment_id"],
            "prompt_observed_monotonic_ns": response["prompt"]["observed_monotonic_ns"],
            "asr_final_capture_to_prompt_observed_ns":
                response["prompt"]["asr_final_capture_to_prompt_observed_ns"],
            "prompt_raw_sha256": response["prompt"]["raw_sha256"],
            "prompt_messages_sha256": response["prompt"]["messages_sha256"],
            "first_event_monotonic_ns": response["first_event_monotonic_ns"],
            "text_utf8_bytes": len(response["text"].encode("utf-8")),
            "text_delta_count": len(response["text_deltas"]),
            "audio_delta_count": len(response["audio"]), "pcm_bytes": len(response["pcm"]),
            "pcm_sha256": response["pcm_sha256"], "audio_chunk_sha256": response["audio_chunk_sha256"],
            "terminal_callback_order": [row["kind"] for row in response["terminals"]],
            "accepted_cursors": response["accepted_cursors"],
        })
    connection_report = [{
        "name": connection["name"], "session": connection["session"],
        "generation": connection["generation"], "frame_count": len(connection["records"]),
        "event_count": len(connection["events"]),
        "first_monotonic_ns": connection["records"][0]["monotonic_ns"],
        "last_monotonic_ns": connection["records"][-1]["monotonic_ns"],
        "strictly_monotonic": True,
    } for connection in connections]
    coverage = {
        "responses_with_first_audio": sum(bool(row["audio"]) for row in responses.values()),
        "responses_without_audio": no_audio,
        "responses_with_audio_done": sum(bool(row["audio_done"]) for row in responses.values()),
        "timing_clock_domain": "single recorded host monotonic clock for WS capture and VLM observer",
        "cross_recorder_ordering": "VLM forwarding and client ASR capture are separate threads; "
                                   "their signed offsets are preserved without clock correction",
        "observed_timing_class": "RETAINED_REAL_DIGITAL_LOOPBACK",
        "new_gpu_serving_model_or_physical_audio_execution": "NOT_RUN",
        "physical_playback_device_latency_and_jitter": "NOT_COVERED",
        "wan_and_real_server_queue_saturation": "NOT_COVERED",
    }
    return {
        "schema": "umminfer-realtime-latency-audit/v1", "status": "PASS", "profile": profile,
        "auditor_sha256": file_sha256(Path(__file__).resolve()), "evidence": manifest,
        "scope": {"cpu_only_reanalysis": True, "new_model_or_server_execution": False,
                  "source_run_validation_class": run.get("validation_class"),
                  "physical_microphone_speaker": "NOT_COVERED",
                  "perceptual_listening_quality": "NOT_COVERED"},
        "counts": counts, "protocol": protocol,
        "retained_verifier_reconciliation": retained_reconciliation,
        "coverage": coverage, "connections": connection_report, "responses": response_report,
        "history_turns": sorted(history_turns, key=lambda row: (row["connection"], row["monotonic_ns"])),
        "acknowledgements": {"accepted": ack_matches, "ignored": ack_ignored},
        "barge_in": barge_rows, "reconnects": reconnects, "echo_and_double_talk": special_report,
        "pcm_artifacts": pcm_artifacts, "resources": resource_report,
        "queue_contract": {**QUEUE_CONTRACT, "observed_queue_depth_samples": 0,
                           "saturation": "NOT_COVERED_BY_RETAINED_REAL_RUN"},
        "metrics": metrics.render(),
    }


def audit_evidence(root, profile="task41-r3"):
    require(profile in ("task41-r3", "generic"), f"unknown audit profile: {profile}")
    root = Path(root)
    require(root.is_dir(), f"evidence directory does not exist: {root}")
    return analyze(root, profile)


def render_report(report):
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_report(report, output):
    """Atomic no-output-on-error behavior; never leave a partial audit report."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(render_report(report), encoding="utf-8")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("task41-r3", "generic"), default="task41-r3")
    args = parser.parse_args(argv)
    try:
        report = audit_evidence(args.evidence, args.profile)
        write_report(report, args.output)
    except (AuditError, OSError, ValueError, TypeError, KeyError) as error:
        print(f"REALTIME_LATENCY_AUDIT FAIL error={error}", file=sys.stderr)
        return 2
    print("REALTIME_LATENCY_AUDIT PASS "
          f"connections={report['counts']['connections']} "
          f"response_ids={report['counts']['response_identities_both_directions']} "
          f"server_responses={report['counts']['server_responses']} "
          f"prompts={report['counts']['native_vlm_prompts']} "
          f"terminals={report['counts']['terminal_callbacks']} "
          f"first_audio={report['coverage']['responses_with_first_audio']} "
          f"outliers={len(report['metrics']['global_outliers'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
