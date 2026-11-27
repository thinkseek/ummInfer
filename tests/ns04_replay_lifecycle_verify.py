#!/usr/bin/env python3
"""Independent raw-wire verifier for the frozen NS-04 replay/lifecycle journey.

This process does not import the live journey producer. It decodes the actual
WebSocket bytes and native MessagePack submits, reconstructs the lossless history
model and played PCM prefixes, and recomputes every timing from raw captures plus
the client monotonic marker journal. Latency failure is reported, not hidden.
"""

import argparse
import base64
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import wave

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "umminfer_ns04_independent_wire_contract", ROOT / "python/umminfer_realtime_network.py")
wire = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wire)

CHUNK_BYTES = 3200
END_SILENCE = bytes(19200)
STALE = wire.STALE


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pcm16(path):
    with wave.open(str(path), "rb") as source:
        require((source.getnchannels(), source.getsampwidth(), source.getframerate()) ==
                (1, 2, 16000), f"invalid frozen WAV: {path.name}")
        return source.readframes(source.getnframes())


def chunks(pcm):
    return [pcm[offset:offset + CHUNK_BYTES] for offset in range(0, len(pcm), CHUNK_BYTES)]


def events(records, direction=None, kind=None, response_id=None):
    return [row for row in records if "event" in row and
            (direction is None or row["direction"] == direction) and
            (kind is None or row["event"].get("type") == kind) and
            (response_id is None or row["event"].get("response_id") == response_id)]


def one(rows, description):
    require(len(rows) == 1, f"{description}: expected one, got {len(rows)}")
    return rows[0]


def text_before(records, response_id, capture_index):
    return "".join(row["event"].get("text", "") for row in
                   events(records, "rx", "response.text.delta", response_id)
                   if row["capture_index"] <= capture_index)


def audio_before(records, response_id, capture_index):
    return [row for row in events(records, "rx", "audio.delta", response_id)
            if row["capture_index"] <= capture_index]


def normalized_messages(messages):
    result = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            require(len(content) == 1 and content[0].get("type") == "text" and
                    isinstance(content[0].get("text"), str), "invalid native text content")
            content = content[0]["text"]
        result.append({"role": message.get("role"), "content": content})
    return result


def load_prompts(path):
    import msgpack
    prompt_map = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        raw = base64.b64decode(row["raw_base64"], validate=True)
        require(row["raw_sha256"] == sha(raw), "native observer raw digest mismatch")
        message = msgpack.unpackb(raw, raw=False, strict_map_key=False)
        require(message.get("type") == "submit", "observer captured non-submit")
        request_id = message.get("request_id")
        require(request_id == row["request_id"] and request_id not in prompt_map,
                "native observer request id mismatch/duplicate")
        messages = message.get("data", {}).get("request", {}).get("inputs", {}).get("messages")
        require(isinstance(messages, list), "native VLM submit has no messages")
        require(normalized_messages(messages) == row["messages"],
                "native observer projection differs from MessagePack bytes")
        prompt_map[request_id] = row
    return prompt_map


def load_submit_times(path):
    """Independently decode native TTS submits; no producer import or projection trust."""
    import msgpack
    rows = []
    identities = set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        raw = base64.b64decode(row["raw_base64"], validate=True)
        require(row["raw_sha256"] == sha(raw), "native submit raw digest mismatch")
        message = msgpack.unpackb(raw, raw=False, strict_map_key=False)
        require(message.get("type") == "submit", "native observer captured non-submit")
        request_id = message.get("request_id")
        require(isinstance(request_id, str) and request_id == row["request_id"] and
                request_id not in identities, "native submit id mismatch/duplicate")
        identities.add(request_id)
        rows.append({"request_id": request_id,
                     "observed_monotonic_ns": row["observed_monotonic_ns"]})
    require(rows == sorted(rows, key=lambda row: row["observed_monotonic_ns"]),
            "native submit observer clock/order regressed")
    return rows


def normalized_complete_command(text):
    """Independent implementation of preregistered stable_complete_short_command_v1."""
    text = text.strip(" \t\r\n")
    if not 4 <= len(text) <= 96 or "\n" in text or "\r" in text or "�" in text:
        return False
    if not text.endswith(("。", "！", "？", "!", "?")):
        return False
    body = text[:-1]
    if body == "换成英文讲":
        return True
    if not body.startswith("停一下"):
        return False
    rest = body[len("停一下"):]
    if rest.startswith(("，", ",")):
        rest = rest[1:]
    return rest == "直接讲土星环"


def parse_incremental_logs(text):
    previews = {}
    eligible = {}
    starts = {}
    preview_pattern = re.compile(
        r"realtime incremental_preview segment=(\d+) stable=(\d+) action=(\d+) "
        r"steady_ns=(\d+) text_sha256=([0-9a-f]{64})")
    eligible_pattern = re.compile(
        r"realtime incremental_eligible segment=(\d+) steady_ns=(\d+) "
        r"text_sha256=([0-9a-f]{64})")
    start_pattern = re.compile(
        r"realtime response_start segment=(\d+) request=(\S+) provisional=([01]) "
        r"steady_ns=(\d+)")
    for line in text.splitlines():
        if match := preview_pattern.search(line):
            segment, stable, action, ns, digest = match.groups()
            previews.setdefault(int(segment), []).append({
                "stable": int(stable), "action": int(action),
                "steady_ns": int(ns), "text_sha256": digest})
        if match := eligible_pattern.search(line):
            segment, ns, digest = match.groups()
            require(int(segment) not in eligible, "duplicate eligibility log")
            eligible[int(segment)] = {"steady_ns": int(ns), "text_sha256": digest}
        if match := start_pattern.search(line):
            segment, request_id, provisional, ns = match.groups()
            require(int(segment) not in starts, "duplicate segment response_start")
            starts[int(segment)] = {"request_id": request_id,
                                    "provisional": provisional == "1",
                                    "steady_ns": int(ns)}
    return previews, eligible, starts


def verify_incremental(journal, records, root, prompt_map, tts_submits, log_state,
                       candidate_present):
    """Recompute preview stability/eligibility and bind early work to exact final intent."""
    segment = journal["new_segment"]
    start_ns = journal["barge_start_ns"]
    final = one([row for row in events(records, "rx", "transcription.completed")
                 if row["capture_index"] == journal["asr_final_capture_index"]],
                "incremental ASR final")
    raw_previews = [row for row in events(records, "rx", "transcription.segment")
                    if row["event"].get("segment_id") == segment and
                    row["event"].get("is_final") is False and
                    row["capture_index"] > journal["speech_started_capture_index"] and
                    row["capture_index"] < final["capture_index"]]
    log_previews, eligible_logs, response_starts = log_state
    logs = log_previews.get(segment, [])
    if candidate_present:
        require(len(logs) == len(raw_previews),
                f"segment {segment} completed-preview/raw identity count mismatch")
    else:
        require(not logs, "Before arm unexpectedly contains incremental candidate logs")
    previous = None
    stable = 0
    active = None
    eligible = None
    preview_report = []
    action_names = {0: "wait", 1: "start", 2: "hold", 4: "revise"}
    for index, row in enumerate(raw_previews):
        text = row["event"].get("text", "").strip(" \t\r\n")
        digest = sha(text.encode())
        expected_action = 0
        if active is not None:
            if text == active:
                expected_action = 2
            else:
                active = None
                previous = None
                stable = 0
                expected_action = 4
        else:
            if text == previous:
                stable += 1
            else:
                previous = text
                stable = 1
            if stable >= 3 and normalized_complete_command(text):
                active = text
                eligible = text
                expected_action = 1
        if candidate_present:
            log = logs[index]
            require(log["text_sha256"] == digest and log["stable"] == stable and
                    log["action"] == expected_action,
                    f"segment {segment} preview {index} violates frozen stability rule")
            require(abs(log["steady_ns"] - row["monotonic_ns"]) <= 100_000_000,
                    "completed-preview log is not adjacent to its raw public preview")
        preview_report.append({
            "index": index, "text": text, "text_sha256": digest,
            "barge_to_preview_ms": (row["monotonic_ns"] - start_ns) / 1e6,
            "stable_count": stable, "action": action_names[expected_action],
            "eligible": expected_action == 1,
        })
    new_id = journal["new_response_id"]
    final_text = final["event"].get("text", "").strip(" \t\r\n")
    eligible_ns = None
    vlm_ns = prompt_map[new_id]["observed_monotonic_ns"]
    tts_matches = [row for row in tts_submits
                   if row["request_id"].startswith(new_id + ":tts")]
    require(tts_matches, f"{new_id} has no observed native TTS submit")
    tts_ns = tts_matches[0]["observed_monotonic_ns"]
    require(vlm_ns <= tts_ns, "TTS started before its owning VLM submit")
    incorrect_early = False
    if candidate_present and eligible is not None:
        marker = eligible_logs.get(segment)
        require(marker is not None and marker["text_sha256"] == sha(eligible.encode()),
                "eligible preview lacks exact server identity")
        eligible_ns = marker["steady_ns"]
        response_start = response_starts.get(segment)
        require(response_start is not None and response_start["request_id"] == new_id and
                response_start["provisional"] is True and
                response_start["steady_ns"] >= eligible_ns,
                "eligible command did not retain one provisional response identity")
        require(final_text == eligible,
                "HARD FAILURE: later ASR final revised eligible/audible intent")
        messages = prompt_map[new_id]["messages"]
        require(messages[-1].get("role") == "user" and
                messages[-1].get("content") == eligible,
                "provisional VLM input differs from exact eligible preview")
        cancelled = [row for row in events(records, "rx", "response.cancelled", new_id)
                     if row["capture_index"] < final["capture_index"]]
        require(not cancelled, "eligible frozen response was revised/cancelled before commit")
        first_audio_ns = journal["first_audio_ns"]
        incorrect_early = (first_audio_ns < final["monotonic_ns"] and
                           (eligible_ns is None or first_audio_ns < eligible_ns or
                            final_text != eligible))
        require(not incorrect_early, "incorrect/speculative early audible response")
    elif candidate_present:
        require(segment not in eligible_logs, "eligible log exists without a valid eligible preview")
    return {
        "preview_count": len(preview_report),
        "previews": preview_report,
        "eligible_text": eligible,
        "eligible_sha256": sha(eligible.encode()) if eligible is not None else None,
        "barge_to_eligible_ms": (eligible_ns - start_ns) / 1e6 if eligible_ns else None,
        "barge_to_vlm_start_ms": (vlm_ns - start_ns) / 1e6,
        "barge_to_tts_start_ms": (tts_ns - start_ns) / 1e6,
        "eligible_to_vlm_start_ms": (vlm_ns - eligible_ns) / 1e6 if eligible_ns else None,
        "eligible_to_tts_start_ms": (tts_ns - eligible_ns) / 1e6 if eligible_ns else None,
        "final_matches_eligible": final_text == eligible if eligible is not None else None,
        "incorrect_early_audible_response": incorrect_early,
        "preview_revision_count": sum(row["action"] == "revise" for row in preview_report),
    }


def verify_input_schedule(records, prereg, audio_dir):
    expected = []
    pcm_by_name = {}
    for turn in prereg["journey"]["turns"]:
        name = turn["name"]
        path = audio_dir / turn["audio_file"]
        raw = path.read_bytes()
        require(sha(raw) == turn["audio_sha256"] and len(raw) == turn["audio_bytes"],
                f"frozen input identity changed: {name}")
        pcm = pcm16(path)
        require(len(pcm) == turn["pcm_bytes"], f"frozen PCM length changed: {name}")
        pcm_by_name[name] = pcm
        expected.extend((name, False, chunk) for chunk in chunks(pcm))
        expected.extend((name, True, chunk) for chunk in chunks(END_SILENCE))
    actual = events(records, "tx", "input_audio_buffer.append")
    require(len(actual) == len(expected),
            f"raw input append count {len(actual)} != frozen {len(expected)}")
    schedule = {}
    previous_by_turn = {}
    for index, (row, (name, silence, expected_pcm)) in enumerate(zip(actual, expected)):
        pcm = base64.b64decode(row["event"].get("audio", ""), validate=True)
        require(pcm == expected_pcm, f"raw input append {index} differs from frozen PCM")
        require(not silence or pcm == b"\x00" * len(pcm), "silence chunk is nonzero")
        ns = row["monotonic_ns"]
        state = schedule.setdefault(name, {"chunks": 0, "pcm_bytes": 0,
                                           "first_append_ns": ns, "last_append_ns": ns,
                                           "pacing_deltas_ms": []})
        state["chunks"] += 1
        state["pcm_bytes"] += len(pcm)
        state["last_append_ns"] = ns
        if name in previous_by_turn:
            previous_ns, previous_bytes = previous_by_turn[name]
            delta = (ns - previous_ns) / 1e6
            expected_ms = previous_bytes / 32.0
            require(0.9 * expected_ms <= delta <= expected_ms + 150.0,
                    f"frozen chunk-duration pacing changed for {name}: "
                    f"previous_bytes={previous_bytes} delta={delta} ms")
            state["pacing_deltas_ms"].append(delta)
        previous_by_turn[name] = (ns, len(pcm))
    return schedule, pcm_by_name


def verify_barge(label, records, markers, run, schedule):
    journal = run[label]
    marker_label = label.removesuffix("_barge")
    old_id = journal["old_response_id"]
    start = one([row for row in markers if row["name"] == f"{marker_label}_barge_start"], label)
    silence = one([row for row in markers if row["name"] == f"{marker_label}_digital_silence"], label)
    first_turn = "saturn" if label == "first_barge" else "english"
    first_append_ns = schedule[first_turn]["first_append_ns"]
    require(0 <= first_append_ns - start["monotonic_ns"] <= 5_000_000,
            f"{label} timing anchor is not adjacent to first interruption append")
    cancelled = one([row for row in events(records, "rx", "response.cancelled", old_id)
                     if row["monotonic_ns"] >= start["monotonic_ns"]], f"{label} cancel")
    cancel_event = cancelled["event"]
    require(cancel_event.get("reason") == "barge_in" and
            cancel_event.get("history_recorded") is True, f"{label} history not recorded")
    cancel_ns = cancelled["monotonic_ns"]
    receipt_ns = journal["cancel_receipt_ns"]
    require(cancel_ns <= receipt_ns <= silence["monotonic_ns"],
            f"{label} client cancel receipt is outside raw cancel/silence bounds")
    stale = [row for row in events(records, "rx", response_id=old_id)
             if row["capture_index"] > cancelled["capture_index"] and
             row["event"]["type"] in STALE]
    require(not stale, f"{label} raw stale events survived cancellation")

    tx_acks = [row for row in events(records, "tx", "output_audio_buffer.playback_ack", old_id)
               if row["capture_index"] < cancelled["capture_index"]]
    require(tx_acks, f"{label} has no playback ACK")
    accepted = events(records, "rx", "response.playback_ack", old_id)
    previous = 0
    issued = {0}
    for row in events(records, "rx", response_id=old_id):
        if row["capture_index"] > cancelled["capture_index"]:
            continue
        if row["event"]["type"] in ("audio.delta", "audio.done"):
            issued.add(row["event"].get("text_cursor"))
        elif row["event"]["type"] == "response.playback_ack":
            cursor = row["event"].get("text_cursor")
            require(cursor in issued and cursor >= previous, f"{label} invalid accepted cursor")
            previous = cursor
    previous_tx = 0
    for ack in tx_acks:
        cursor = ack["event"].get("text_cursor")
        require(isinstance(cursor, int) and cursor >= previous_tx,
                f"{label} decreasing client ACK")
        previous_tx = cursor
    require(previous_tx == cancel_event.get("text_cursor"),
            f"{label} cancellation cursor differs from last raw client ACK")

    played_journal = journal["played"]
    played_indices = played_journal["played_capture_indices"]
    require(len(played_indices) == len(set(played_indices)) and
            played_indices == sorted(played_indices), f"{label} played indices invalid")
    by_capture = {row["capture_index"]: row for row in
                  audio_before(records, old_id, cancelled["capture_index"])}
    require(all(index in by_capture for index in played_indices),
            f"{label} played PCM includes unissued event")
    played_rows = [by_capture[index] for index in played_indices]
    played_pcm = b"".join(base64.b64decode(row["event"]["audio"], validate=True)
                          for row in played_rows)
    require(sha(played_pcm) == played_journal["played_pcm_sha256"] and
            len(played_pcm) == played_journal["played_pcm_bytes"],
            f"{label} played PCM hash/length mismatch")
    all_old = audio_before(records, old_id, cancelled["capture_index"])
    all_pcm = b"".join(base64.b64decode(row["event"]["audio"], validate=True) for row in all_old)
    require(all_pcm.startswith(played_pcm) and len(played_pcm) < len(all_pcm),
            f"{label} played PCM is not a proper raw prefix")
    pending_rows = [row for row in all_old if row["capture_index"] not in set(played_indices)]
    pending_pcm = b"".join(base64.b64decode(row["event"]["audio"], validate=True)
                           for row in pending_rows)
    require(pending_pcm and len(pending_pcm) == journal["pending_pcm_bytes"] and
            sha(pending_pcm) == journal["pending_pcm_sha256"],
            f"{label} pending PCM was not cleared exactly")
    acknowledged = cancel_event.get("acknowledged_text")
    cursor = cancel_event.get("text_cursor")
    raw_text = text_before(records, old_id, cancelled["capture_index"])
    require(isinstance(acknowledged, str) and len(acknowledged.encode()) == cursor and
            raw_text.encode()[:cursor].decode() == acknowledged and
            cursor == played_journal["acknowledged_cursor"],
            f"{label} acknowledged history is not exact played UTF-8 prefix")

    started = one([row for row in events(records, "rx", "input_audio_buffer.speech_started")
                   if row["event"].get("interrupted_response_id") == old_id and
                   row["monotonic_ns"] >= start["monotonic_ns"]], f"{label} speech start")
    segment = started["event"].get("segment_id")
    stopped = one([row for row in events(records, "rx", "input_audio_buffer.speech_stopped")
                   if row["event"].get("segment_id") == segment and
                   row["capture_index"] > started["capture_index"]], f"{label} speech stop")
    completed = one([row for row in events(records, "rx", "transcription.completed")
                     if row["event"].get("segment_id") == segment and
                     row["capture_index"] > stopped["capture_index"]], f"{label} ASR final")
    new_id = journal["new_response_id"]
    require(new_id != old_id, f"{label} response id was reused")
    new_audio = [row for row in events(records, "rx", "audio.delta", new_id)
                 if row["event"].get("segment_id") == segment and
                 row["capture_index"] > started["capture_index"]]
    require(new_audio, f"{label} has no new public audio")
    first_audio = min(new_audio, key=lambda row: row["capture_index"])
    timings = {
        "barge_to_cancel": (cancel_ns - start["monotonic_ns"]) / 1e6,
        "cancel_to_digital_silence": (silence["monotonic_ns"] - receipt_ns) / 1e6,
        "barge_to_speech_stopped_replay": (stopped["monotonic_ns"] - start["monotonic_ns"]) / 1e6,
        "speech_stopped_to_asr_final": (completed["monotonic_ns"] - stopped["monotonic_ns"]) / 1e6,
        "asr_final_to_new_audio": (first_audio["monotonic_ns"] - completed["monotonic_ns"]) / 1e6,
        "replay_inclusive_barge_to_new_audio": (first_audio["monotonic_ns"] - start["monotonic_ns"]) / 1e6,
    }
    for key, value in timings.items():
        require(math.isfinite(value) and
                (value >= 0 or key == "asr_final_to_new_audio"),
                f"{label} invalid timing {key}")
        require(abs(value - journal["timings_ms"][key]) < 0.001,
                f"{label} producer/independent timing mismatch for {key}")
    return {
        "old_response_id": old_id,
        "new_response_id": new_id,
        "new_segment": segment,
        "barge_start_ns": start["monotonic_ns"],
        "speech_started_capture_index": started["capture_index"],
        "asr_final_ns": completed["monotonic_ns"],
        "first_audio_ns": first_audio["monotonic_ns"],
        "transcript": completed["event"].get("text"),
        "timings_ms": timings,
        "cancel_raw_arrival_to_digital_silence_ms": (silence["monotonic_ns"] - cancel_ns) / 1e6,
        "played": {
            "acknowledged_cursor": cursor,
            "acknowledged_text": acknowledged,
            "played_pcm_sha256": sha(played_pcm),
            "played_pcm_bytes": len(played_pcm),
            "played_audio_events": len(played_rows),
        },
        "old_pcm_sha256": sha(all_pcm),
        "old_response_text_sha256": sha(raw_text.encode()),
        "old_response_text_utf8_bytes": len(raw_text.encode()),
        "pending_pcm_sha256": sha(pending_pcm),
        "pending_pcm_bytes": len(pending_pcm),
        "pending_audio_events": len(pending_rows),
        "stale_events": len(stale),
        "asr_final_capture_index": completed["capture_index"],
        "first_audio_capture_index": first_audio["capture_index"],
    }


def verify(root, prereg_path, audio_dir, candidate_prereg_path=None):
    root = Path(root)
    run_raw = (root / "run.json").read_bytes()
    run = json.loads(run_raw)
    require(run["schema"] == "umminfer-ns04-replay-lifecycle-live/v1" and
            run["status"] == "PASS_CORRECTNESS", "live correctness capture did not pass")
    prereg_raw = Path(prereg_path).read_bytes()
    prereg = json.loads(prereg_raw)
    require(run["preregistration"]["sha256"] == sha(prereg_raw),
            "capture/preregistration digest mismatch")
    candidate_raw = None
    if candidate_prereg_path is not None:
        candidate_raw = Path(candidate_prereg_path).read_bytes()
        candidate = json.loads(candidate_raw)
        require(candidate["format"] ==
                "umminfer-ns04-incremental-early-response-preregistration-v1" and
                run["candidate_preregistration"]["sha256"] == sha(candidate_raw),
                "capture/candidate preregistration digest mismatch")
    frame_raw = (root / "frames.jsonl").read_bytes()
    records = wire.decode_capture(root / "frames.jsonl")
    for row in records:
        row["capture_index"] = row["index"]
    marker_rows = [json.loads(line) for line in (root / "markers.jsonl").read_text().splitlines()]
    require(all(row["monotonic_ns"] < marker_rows[index + 1]["monotonic_ns"]
                for index, row in enumerate(marker_rows[:-1])), "marker clock not monotonic")
    schedule, _ = verify_input_schedule(records, prereg, Path(audio_dir))
    first = verify_barge("first_barge", records, marker_rows, run, schedule)
    second = verify_barge("second_barge", records, marker_rows, run, schedule)

    solar_id = first["old_response_id"]
    solar_start = events(records, "rx", "transcription.completed")[0]
    transcripts = {
        "solar": solar_start["event"].get("text"),
        "saturn": first["transcript"],
        "english": second["transcript"],
    }
    for turn in prereg["journey"]["turns"]:
        for fragment in turn["required_transcript_fragments"]:
            require(fragment in transcripts[turn["name"]], f"ASR fragment missing: {fragment}")

    prompt_map = load_prompts(root / "vlm-prompts.jsonl")
    history = wire._history_model(records, prompt_map, "frozen-journey")
    require(history["verified_prompts"] == 3 and history["final_history_turns"] == 2,
            f"unexpected exact history coverage: {history}")
    prompt_digest = sha(json.dumps({key: row["messages"] for key, row in prompt_map.items()},
                                   ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode())
    tts_submits = load_submit_times(root / "tts-submits.jsonl")
    log_state = parse_incremental_logs((root / "server.log").read_text())
    incremental = {
        "first_barge": verify_incremental(
            first, records, root, prompt_map, tts_submits, log_state,
            candidate_raw is not None and run["arm"] == "after"),
        "second_barge": verify_incremental(
            second, records, root, prompt_map, tts_submits, log_state,
            candidate_raw is not None and run["arm"] == "after"),
    }

    final_id = second["new_response_id"]
    expected_response_ids = {first["old_response_id"], second["old_response_id"], final_id}
    observed_response_ids = {row["event"].get("response_id") for row in
                             events(records, "rx") if row["event"].get("response_id") is not None}
    unexpected_response_ids = observed_response_ids - expected_response_ids
    require(not unexpected_response_ids,
            f"unexpected/cross-session response ids: {sorted(unexpected_response_ids)}")
    terminal_kinds = {}
    duplicate_terminals = 0
    for response_id in expected_response_ids:
        kinds = [row["event"]["type"] for row in events(records, "rx", response_id=response_id)
                 if row["event"]["type"] in wire.TERMINALS]
        terminal_kinds[response_id] = kinds
        duplicate_terminals += sum(max(0, kinds.count(kind) - 1) for kind in set(kinds))
    require(terminal_kinds[first["old_response_id"]] == ["response.cancelled"] and
            terminal_kinds[second["old_response_id"]] == ["response.cancelled"],
            f"interrupted raw terminal lifecycle mismatch: {terminal_kinds}")
    final_terminal = one(events(records, "rx", "response.completed", final_id),
                         "final English completed")
    require(terminal_kinds[final_id] == ["response.completed"] and duplicate_terminals == 0,
            f"final/duplicate raw terminal lifecycle mismatch: {terminal_kinds}")
    late_final = [row for row in events(records, "rx", response_id=final_id)
                  if row["capture_index"] > final_terminal["capture_index"] and
                  row["event"]["type"] in STALE]
    require(not late_final, f"late final-response text/PCM/terminal events: {late_final}")
    final_text = text_before(records, final_id, final_terminal["capture_index"])
    final_pcm = b"".join(base64.b64decode(row["event"]["audio"], validate=True)
                         for row in audio_before(records, final_id, final_terminal["capture_index"]))
    ascii_letters = sum(chr(byte).isalpha() for byte in final_text.encode() if byte < 128)
    cjk = sum(0x4E00 <= ord(char) <= 0x9FFF for char in final_text)
    require(final_pcm and ascii_letters > 20 and ascii_letters > 2 * max(1, cjk) and
            ("saturn" in final_text.lower() or "ring" in final_text.lower()),
            "final English topical PCM/text gate failed")
    require(run["final_health"]["in_flight"] == 0 and
            run["cleanup"]["owned_processes_exited"] is True and
            run["cleanup"]["selected_gpu_idle"] is True,
            "process/GPU/in_flight cleanup failed")

    thresholds = {
        "barge_to_cancel_strictly_below_300": first["timings_ms"]["barge_to_cancel"] < 300,
        "cancel_to_digital_silence_strictly_below_300":
            first["timings_ms"]["cancel_to_digital_silence"] < 300,
        "asr_final_to_new_audio_strictly_below_800":
            first["timings_ms"]["asr_final_to_new_audio"] < 800,
        "replay_inclusive_barge_to_new_audio_strictly_below_800":
            first["timings_ms"]["replay_inclusive_barge_to_new_audio"] < 800,
        "second_barge_to_cancel_strictly_below_300":
            second["timings_ms"]["barge_to_cancel"] < 300,
    }
    return {
        "schema": "umminfer-ns04-replay-lifecycle-independent/v1",
        "status": "PASS",
        "assertion_source": "fresh raw WebSocket and MessagePack decode without live producer import",
        "verifier_sha256": sha(Path(__file__).read_bytes()),
        "arm": run["arm"],
        "repetition": run["repetition"],
        "execution_commit": run["execution_commit"],
        "accepted_parent_commit": run["accepted_parent_commit"],
        "preregistration_sha256": sha(prereg_raw),
        "candidate_preregistration_sha256": sha(candidate_raw) if candidate_raw else None,
        "run_json_sha256": sha(run_raw),
        "raw_frames_sha256": sha(frame_raw),
        "report_identity": sha(run_raw),
        "binaries": run["binaries"],
        "harness": run["harness"],
        "model_manifest_sha256": {name: manifest["model_manifest_sha256"]
                                  for name, manifest in run["models"].items()},
        "transcripts": transcripts,
        "input_schedule": schedule,
        "first_barge": first,
        "second_barge": second,
        "incremental": incremental,
        "history": {
            "verified_prompts": history["verified_prompts"],
            "final_history_turns": history["final_history_turns"],
            "accepted_cursors": history["accepted_cursors"],
            "vlm_prompts_sha256": prompt_digest,
            "terminal_callbacks": terminal_kinds,
            "stale_events": first["stale_events"] + second["stale_events"],
            "cross_session_leaks": len(unexpected_response_ids),
            "duplicate_terminal_callbacks": duplicate_terminals,
            "late_final_events": len(late_final),
        },
        "final_response": {
            "text": final_text,
            "text_sha256": sha(final_text.encode()),
            "text_utf8_bytes": len(final_text.encode()),
            "pcm_sha256": sha(final_pcm),
            "pcm_bytes": len(final_pcm),
            "ascii_letters": ascii_letters,
            "cjk_codepoints": cjk,
        },
        "thresholds": thresholds,
        "cleanup": {"final_in_flight": 0, "selected_gpu_idle": True,
                    "owned_processes_exited": True},
        "physical_microphone_speaker": "NOT_RUN",
        "acoustic_echo": "NOT_RUN",
        "perceptual_listening_quality": "NOT_RUN",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--candidate-preregistration", type=Path)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("independent report output must be new")
    report = verify(args.root, args.preregistration, args.audio_dir,
                    args.candidate_preregistration)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("PASS independent NS-04 raw verification; "
          f"replay_inclusive={report['first_barge']['timings_ms']['replay_inclusive_barge_to_new_audio']:.3f} ms; "
          f"thresholds={report['thresholds']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
