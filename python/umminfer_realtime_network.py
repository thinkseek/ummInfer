#!/usr/bin/env python3
"""Deterministic planning and independent raw-frame verification for realtime gates.

This module is intentionally standard-library only. The verifier does not import
or call the live driver: it decodes the retained WebSocket wire bytes, derives
history from terminal/ACK events, and compares that model with the prompts seen
by the unmodified forwarding observer at the native VLM boundary.
"""

import base64
import hashlib
import json
from pathlib import Path
import struct

SCENARIOS = (
    "final_ack",
    "immediate_barge",
    "delayed_barge",
    "reconnect_playback",
    "reconnect_generation",
    "teardown_callbacks",
)
TERMINALS = {"response.completed", "response.failed", "response.cancelled"}
STALE = {
    "audio.delta", "audio.done", "response.text.delta",
    "response.playback_ack", "response.completed", "response.failed",
}
CUTINS = {
    "input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped",
    "response.cancelled", "transcription.completed",
}


class VerificationError(AssertionError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & ((1 << 64) - 1)

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
        return z ^ (z >> 31)


def make_plan(seed=0x41C0DE, sessions=6, rounds=3):
    require(sessions >= 6, "the live gate requires at least six sessions")
    require(rounds >= 2, "the live gate requires repeated rounds")
    rng = SplitMix64(seed)
    rows = []
    for session in range(sessions):
        rows.append({
            "session": session,
            "label": f"isolated-{session}",
            "scenarios": [SCENARIOS[(session + r) % len(SCENARIOS)]
                          for r in range(rounds)],
        })
    schedule = []
    for round_index in range(rounds):
        order = list(range(sessions))
        for i in range(len(order) - 1, 0, -1):
            j = rng.next() % (i + 1)
            order[i], order[j] = order[j], order[i]
        schedule.append({
            "round": round_index,
            "start_order": order,
            "start_jitter_ms": {str(s): int(rng.next() % 17) for s in order},
            "ack_delay_ms": {str(s): int(5 + rng.next() % 55) for s in order},
            "foreign_ack_response": {str(s): f"foreign-{seed:x}-{round_index}-{s}"
                                     for s in order},
        })
    return {
        "schema": "umminfer-realtime-network-plan/v1",
        "seed": seed,
        "sessions": rows,
        "rounds": schedule,
        "required_scenarios": list(SCENARIOS),
        "echo_ab": {
            "arms": ["protected", "disabled"],
            "same_active_response": True,
            "identical_microphone_pcm": True,
            "identical_far_end_pcm": True,
            "double_talk": True,
        },
    }


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _json_lines(path):
    with open(path, "rb") as source:
        for line_number, line in enumerate(source, 1):
            require(line.endswith(b"\n"), f"unterminated JSONL: {path}:{line_number}")
            try:
                yield json.loads(line)
            except (ValueError, UnicodeDecodeError) as error:
                raise VerificationError(f"invalid JSONL {path}:{line_number}: {error}") from error


def decode_wire_frame(raw, direction):
    """Return (fin, opcode, payload); independently validate the wire envelope."""
    require(len(raw) >= 2, "truncated WebSocket frame")
    first, second = raw[0], raw[1]
    require(first & 0x70 == 0, "unexpected WebSocket RSV bits")
    fin = bool(first & 0x80)
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    require(masked == (direction == "tx"), "incorrect WebSocket masking direction")
    length = second & 0x7F
    offset = 2
    if length == 126:
        require(len(raw) >= offset + 2, "truncated 16-bit WebSocket length")
        length = struct.unpack_from("!H", raw, offset)[0]
        offset += 2
        require(length >= 126, "non-canonical WebSocket length")
    elif length == 127:
        require(len(raw) >= offset + 8, "truncated 64-bit WebSocket length")
        length = struct.unpack_from("!Q", raw, offset)[0]
        offset += 8
        require(length >= 65536 and length < (1 << 63), "invalid WebSocket length")
    mask = None
    if masked:
        require(len(raw) >= offset + 4, "truncated WebSocket mask")
        mask = raw[offset:offset + 4]
        offset += 4
    require(len(raw) == offset + length, "WebSocket frame length mismatch")
    payload = raw[offset:]
    if mask is not None:
        payload = bytes(value ^ mask[i % 4] for i, value in enumerate(payload))
    if opcode >= 8:
        require(fin and length <= 125, "invalid fragmented/oversized control frame")
    return fin, opcode, payload


def decode_capture(path):
    records = []
    fragments = {}
    previous_ns = -1
    for index, row in enumerate(_json_lines(path)):
        require(set(row) == {"index", "monotonic_ns", "direction", "raw_base64"},
                f"unexpected capture schema in {path}:{index}")
        require(row["index"] == index, "capture index is not contiguous")
        require(isinstance(row["monotonic_ns"], int) and
                row["monotonic_ns"] > previous_ns, "capture clock is not strictly monotonic")
        previous_ns = row["monotonic_ns"]
        direction = row["direction"]
        require(direction in ("rx", "tx"), "unknown capture direction")
        raw = base64.b64decode(row["raw_base64"], validate=True)
        fin, opcode, payload = decode_wire_frame(raw, direction)
        records.append({"index": index, "monotonic_ns": row["monotonic_ns"],
                        "direction": direction, "raw_sha256": sha256(raw),
                        "opcode": opcode})
        if opcode in (8, 9, 10):
            continue
        require(opcode in (0, 1), "unexpected binary realtime WebSocket message")
        if opcode != 0:
            require(direction not in fragments, "nested WebSocket fragmented message")
            fragments[direction] = [opcode, bytearray()]
        require(direction in fragments, "continuation without initial frame")
        fragments[direction][1].extend(payload)
        require(len(fragments[direction][1]) <= (1 << 20), "realtime message exceeds 1 MiB")
        if fin:
            _, message = fragments.pop(direction)
            try:
                event = json.loads(message.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as error:
                raise VerificationError(f"invalid realtime JSON message: {error}") from error
            require(isinstance(event, dict) and isinstance(event.get("type"), str),
                    "realtime event has no string type")
            records.append({"index": index, "monotonic_ns": row["monotonic_ns"],
                            "direction": direction, "event": event,
                            "payload_sha256": sha256(message)})
    require(not fragments, "unterminated fragmented WebSocket message")
    return records


def _text(events, response_id):
    return "".join(event.get("event", {}).get("text", "") for event in events
                   if event["direction"] == "rx" and
                   event.get("event", {}).get("response_id") == response_id and
                   event["event"]["type"] == "response.text.delta")


def _history_model(records, prompt_map, connection):
    """Derive exact next prompts from wire events, not from the driver journal."""
    history = []
    transcripts = {}
    responses = {}
    expected_prompts = {}
    accepted_cursor = {}
    issued_cursor = {}
    terminal_count = {}
    cancelled_at = {}
    pcm_hashes = {}
    for record in records:
        if "event" not in record:
            continue
        direction = record["direction"]
        event = record["event"]
        kind = event["type"]
        response_id = event.get("response_id")
        if kind == "transcription.completed" and direction == "rx":
            segment = event.get("segment_id")
            require(isinstance(segment, int), "transcription has no segment id")
            require(segment not in transcripts, "duplicate transcription segment")
            require(isinstance(event.get("text"), str) and event["text"].strip(),
                    "empty real ASR transcription")
            transcripts[segment] = event["text"]
        if response_id is not None:
            require(isinstance(response_id, str) and response_id,
                    "empty response identity")
            state = responses.setdefault(response_id, {
                "connection": connection, "segment": event.get("segment_id"),
                "terminal": [], "cancel_index": None,
            })
            require(state["connection"] == connection, "response id crossed connections")
            if state["segment"] is None:
                state["segment"] = event.get("segment_id")
            if state["cancel_index"] is not None and direction == "rx" and kind in STALE:
                raise VerificationError(f"stale {kind} after cancellation of {response_id}")
        if direction == "rx" and kind == "audio.delta":
            require(response_id is not None, "audio delta has no response id")
            pcm = base64.b64decode(event.get("audio", ""), validate=True)
            require(pcm and len(pcm) % 2 == 0, "invalid/empty realtime PCM")
            require(event.get("format") == "pcm16" and event.get("sample_rate") == 24000 and
                    event.get("channels") == 1, "unexpected real TTS PCM metadata")
            pcm_hashes.setdefault(response_id, []).append(sha256(pcm))
            cursor = event.get("text_cursor")
            require(isinstance(cursor, int) and cursor >= 0, "audio cursor missing")
            issued_cursor.setdefault(response_id, set()).add(cursor)
        elif direction == "rx" and kind == "audio.done":
            cursor = event.get("text_cursor")
            require(isinstance(cursor, int) and cursor >= 0, "final audio cursor missing")
            issued_cursor.setdefault(response_id, set()).add(cursor)
        elif direction == "rx" and kind == "response.playback_ack":
            cursor = event.get("text_cursor")
            require(isinstance(cursor, int) and cursor in issued_cursor.get(response_id, set()),
                    "server accepted an unissued playback cursor")
            previous = accepted_cursor.get(response_id, 0)
            require(cursor >= previous, "server accepted a decreasing playback cursor")
            accepted_cursor[response_id] = cursor
            if event.get("final") is True:
                segment = responses[response_id]["segment"]
                require(segment in transcripts, "final ACK has no initiating user transcript")
                text = _text(records, response_id)
                require(cursor == len(text.encode("utf-8")), "final ACK is not full UTF-8 length")
                history.append({"role": "user", "content": transcripts[segment]})
                history.append({"role": "assistant", "content": text})
        elif direction == "rx" and kind in TERMINALS:
            terminal_count.setdefault(response_id, []).append(kind)
            if kind == "response.cancelled":
                responses[response_id]["cancel_index"] = record["index"]
                cursor = event.get("text_cursor")
                acknowledged = event.get("acknowledged_text")
                require(isinstance(cursor, int) and isinstance(acknowledged, str),
                        "barge-in cancellation lacks acknowledged prefix")
                require(event.get("reason") == "barge_in" and
                        event.get("history_recorded") is True,
                        "eligible barge-in did not record history")
                require(cursor == accepted_cursor.get(response_id, 0),
                        "cancellation cursor differs from last accepted ACK")
                require(len(acknowledged.encode("utf-8")) == cursor,
                        "acknowledged prefix is not exact UTF-8 cursor bytes")
                segment = responses[response_id]["segment"]
                require(segment in transcripts, "cancelled response has no user transcript")
                text = _text(records, response_id)
                require(text.encode("utf-8")[:cursor].decode("utf-8") == acknowledged,
                        "acknowledged prefix is not a prefix of generated text")
                history.append({"role": "user", "content": transcripts[segment]})
                history.append({"role": "assistant", "content": acknowledged})
        if direction == "rx" and kind in ("response.text.delta", "audio.delta"):
            if response_id not in expected_prompts:
                segment = responses[response_id]["segment"]
                require(segment in transcripts, "response has no initiating transcript")
                expected_prompts[response_id] = history + [
                    {"role": "user", "content": transcripts[segment]}]
    for response_id, terminals in terminal_count.items():
        for kind in terminals:
            require(terminals.count(kind) == 1,
                    f"response {response_id} has duplicate {kind} user callbacks")
        require("response.failed" not in terminals or len(terminals) == 1,
                f"failed response {response_id} also has another terminal callback")
    for response_id, expected in expected_prompts.items():
        actual = prompt_map.get(response_id + ":vlm") or prompt_map.get(response_id)
        require(actual is not None, f"no native VLM prompt observed for {response_id}")
        require(actual["messages"] == expected,
                f"next-turn prompt mismatch for {response_id}: "
                f"expected={expected!r} actual={actual['messages']!r}")
    return {
        "responses": len(responses),
        "terminal_callbacks": sum(len(v) for v in terminal_count.values()),
        "final_history_turns": len(history) // 2,
        "verified_prompts": len(expected_prompts),
        "audio_chunk_sha256": pcm_hashes,
        "accepted_cursors": accepted_cursor,
    }


def verify_evidence(root):
    root = Path(root)
    run = json.loads((root / "run.json").read_text())
    require(run["schema"] == "umminfer-realtime-network-live/v1", "unknown live schema")
    plan = json.loads((root / "plan.json").read_text())
    require(plan == make_plan(plan["seed"], len(plan["sessions"]), len(plan["rounds"])),
            "stored schedule is not the deterministic seeded plan")
    require(len(plan["sessions"]) >= 6 and len(plan["rounds"]) >= 2,
            "concurrency/repetition gate not met")
    require(run["concurrent_session_peak"] >= 6, "six sessions were not concurrently connected")
    require(run["final_health"].get("in_flight") == 0, "final in_flight is nonzero")
    require(run["cleanup"]["owned_processes_exited"] is True and
            run["cleanup"]["selected_gpus_idle"] is True, "process/GPU cleanup failed")
    require(run["physical_microphone_speaker"] == "NOT_COVERED" and
            run["perceptual_listening_quality"] == "NOT_COVERED", "physical scope was overstated")
    prompt_map = {}
    for row in _json_lines(root / "vlm-prompts.jsonl"):
        request_id = row["request_id"]
        require(request_id not in prompt_map, f"duplicate VLM request id {request_id}")
        require(row["raw_sha256"] == sha256(base64.b64decode(row["raw_base64"], validate=True)),
                "VLM observer raw digest mismatch")
        prompt_map[request_id] = row
    connection_results = {}
    ownership = {}
    for session in plan["sessions"]:
        for generation in range(run["connections_per_session"][str(session["session"])]):
            name = f"session-{session['session']}-connection-{generation}"
            records = decode_capture(root / "frames" / f"{name}.jsonl")
            result = _history_model(records, prompt_map, name)
            connection_results[name] = result
            for record in records:
                event = record.get("event")
                if event and record["direction"] == "rx" and event.get("response_id"):
                    response_id = event["response_id"]
                    require(ownership.setdefault(response_id, name) == name,
                            f"cross-session response leakage: {response_id}")
    echo = run["echo_ab"]
    require(echo["protected_microphone_sha256"] == echo["disabled_microphone_sha256"] and
            echo["protected_far_end_sha256"] == echo["disabled_far_end_sha256"],
            "echo A/B did not use byte-identical near/far PCM")
    require(echo["protected_cutins"] == 0 and echo["disabled_cutins"] >= 1,
            "echo suppression A/B did not exercise both outcomes")
    require(echo["double_talk_protected_cancelled"] is True and
            echo["double_talk_disabled_cancelled"] is True,
            "double-talk did not cancel in both echo modes")
    echo_records = decode_capture(root / "frames" / f"{echo['connection']}.jsonl")
    protected_cutins = [row for row in echo_records
                        if row["direction"] == "rx" and
                        echo["protected_start_capture_index"] <= row["index"] <
                        echo["protected_end_capture_index"] and row.get("event") and
                        row["event"]["type"] in CUTINS]
    require(not protected_cutins, "independent raw-frame decode found protected echo cut-in")
    disabled = [row for row in echo_records
                if row["index"] == echo["disabled_cancel_capture_index"] and
                row.get("event", {}).get("type") == "response.cancelled" and
                row["event"].get("response_id") == echo["response_id"]]
    require(len(disabled) == 1, "independent raw-frame decode missed disabled echo cancellation")
    echo_far = b"".join(base64.b64decode(row["event"]["audio"], validate=True)
                        for row in echo_records if row["direction"] == "rx" and
                        row.get("event", {}).get("type") == "audio.delta" and
                        row["event"].get("response_id") == echo["response_id"])
    require(sha256(echo_far) == echo["protected_far_end_sha256"] ==
            sha256((root / "echo-far-end-24k.pcm").read_bytes()),
            "independent far-end PCM hash mismatch")
    require(sha256((root / "echo-microphone-16k.pcm").read_bytes()) ==
            echo["protected_microphone_sha256"], "independent microphone PCM hash mismatch")
    for arm, response_id in echo["double_talk_response_ids"].items():
        index = echo["double_talk_cancel_capture_indices"][arm]
        require(any(row["index"] == index and row.get("event", {}).get("type") ==
                    "response.cancelled" and row["event"].get("response_id") == response_id
                    for row in echo_records), f"double-talk {arm} cancellation absent from raw frames")
    scenarios = [name for row in plan["sessions"] for name in row["scenarios"]]
    for required in SCENARIOS:
        require(scenarios.count(required) >= 2, f"scenario was not repeated: {required}")
    require(run["late_callback_count"] >= 1, "teardown did not exercise active callbacks")
    require(run["queue_contract"]["outbound_frames"] == 1024 and
            run["queue_contract"]["partial_events"] == 4096 and
            run["queue_contract"]["unified_events"] == 4096 and
            run["queue_contract"]["max_audio_bytes"] == 64 * 1024 * 1024,
            "production bounded-queue contract changed")
    require(run["resource_usage"]["server"]["rss_growth_bytes"] <=
            run["resource_bounds"]["server_rss_growth_max_bytes"],
            "server memory growth exceeded the bounded-queue gate")
    return {
        "schema": "umminfer-realtime-network-independent/v1",
        "status": "PASS",
        "assertion_source": "fresh decode of raw WebSocket wire frames and VLM observer bytes",
        "sessions": len(plan["sessions"]),
        "rounds": len(plan["rounds"]),
        "connections": len(connection_results),
        "responses": sum(v["responses"] for v in connection_results.values()),
        "terminal_callbacks": sum(v["terminal_callbacks"] for v in connection_results.values()),
        "verified_next_turn_prompts": sum(v["verified_prompts"] for v in connection_results.values()),
        "final_history_turns": sum(v["final_history_turns"] for v in connection_results.values()),
        "cross_session_leaks": 0,
        "stale_events": 0,
        "double_terminal_callbacks": 0,
        "decreasing_accepted_cursors": 0,
        "final_in_flight": 0,
        "physical_microphone_speaker": "NOT_COVERED",
        "perceptual_listening_quality": "NOT_COVERED",
        "connection_results": connection_results,
    }
