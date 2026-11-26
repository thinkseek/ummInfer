#!/usr/bin/env python3
"""Independent planning, raw-frame timing and A/B checks for realtime first audio.

The analyzer deliberately does not import the live driver. It uses the accepted
Task 53 stdlib WebSocket/MessagePack decoder, then derives every timestamp from
raw public frames and byte-preserving native-boundary observer records.
"""

from collections import defaultdict
from decimal import Decimal
from fractions import Fraction
import base64
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "umminfer_realtime_first_audio_decoder", ROOT / "python/umminfer_realtime_latency_audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

TERMINALS = {"response.completed", "response.failed", "response.cancelled"}
STALE = {"audio.delta", "audio.done", "response.text.delta", "response.playback_ack",
         "response.completed", "response.failed"}
CUTINS = {"input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped",
          "response.cancelled", "transcription.completed"}


class FirstAudioError(AssertionError):
    pass


def require(condition, message):
    if not condition:
        raise FirstAudioError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_sha256(path):
    return audit.file_sha256(Path(path))


def load_json(path):
    return audit.load_json(Path(path))


def load_contract(path):
    contract = load_json(path)
    require(contract.get("schema") == "umminfer-realtime-first-audio-preregistration/v1",
            "unknown first-audio preregistration schema")
    require(contract.get("seed") == 5630241, "seed changed after preregistration")
    require(len(contract.get("arms", [])) == 7, "the seven preregistered arms are required")
    names = [arm.get("name") for arm in contract["arms"]]
    require(len(set(names)) == 7, "duplicate arm")
    for arm in contract["arms"]:
        require(arm.get("rounds") == 3 and arm.get("sessions") in (1, 6),
                "arm repetition/session count changed")
        require(arm.get("fixture") in contract["fixtures"], "unknown audio fixture")
        require(arm.get("warmup_rounds") in (0, 1), "invalid warmup count")
        require(bool(arm.get("prefix_cache")) == (arm["warmup_rounds"] == 1),
                "cold/warm cache and warmup contract diverged")
    require(contract["topology"]["asr"]["physical_gpu"] == 0 and
            contract["topology"]["vlm"]["physical_gpu"] == 1 and
            contract["topology"]["tts"]["physical_gpu"] == 1, "GPU assignment changed")
    return contract


def make_plan(contract):
    """Return deterministic interleaving and logical connection/turn assignments."""
    rng = audit.SplitMix64(contract["seed"])
    arms = []
    for spec in contract["arms"]:
        arm = dict(spec)
        arm["schedule"] = []
        total_rounds = arm["warmup_rounds"] + arm["rounds"]
        for round_index in range(total_rounds):
            order = list(range(arm["sessions"]))
            for i in range(len(order) - 1, 0, -1):
                j = rng.next() % (i + 1)
                order[i], order[j] = order[j], order[i]
            arm["schedule"].append({
                "round": round_index,
                "measured": round_index >= arm["warmup_rounds"],
                "start_order": order,
                "start_jitter_ms": {str(s): int(rng.next() % 17) for s in order},
            })
        connections = []
        for session in range(arm["sessions"]):
            if arm["prefix_cache"]:
                connections.append({"name": f"session-{session}-connection-0",
                                    "session": session, "turns": total_rounds,
                                    "warmup_turns": arm["warmup_rounds"]})
            else:
                for round_index in range(total_rounds):
                    connections.append({"name": f"session-{session}-connection-{round_index}",
                                        "session": session, "turns": 1, "warmup_turns": 0,
                                        "round": round_index})
        arm["connections"] = connections
        arms.append(arm)
    return {"schema": "umminfer-realtime-first-audio-plan/v1", "seed": contract["seed"],
            "contract_sha256": sha256((json.dumps(contract, indent=2, sort_keys=True) + "\n").encode()),
            "arms": arms}


def normalize_messages(messages):
    return audit.normalize_messages(messages)


def _observer_json_lines(path):
    # The retained R3 auditor intentionally caps whole files. Native completion
    # evidence can exceed that cap because it includes both streamed PCM and the
    # terminal WAV, so stream bounded lines instead of dropping raw records.
    require(path.stat().st_size <= 4 * 1024 * 1024 * 1024,
            f"observer file exceeds 4 GiB: {path.name}")
    with open(path, "rb") as source:
        for line_number, line in enumerate(source, 1):
            require(len(line) <= 16 * 1024 * 1024 and line.endswith(b"\n"),
                    f"invalid/oversized observer line: {path.name}:{line_number}")
            try:
                yield json.loads(line, object_pairs_hook=audit._object_no_duplicates)
            except (ValueError, UnicodeError) as error:
                raise FirstAudioError(f"invalid observer JSON {path.name}:{line_number}: {error}") from error


def _observer_rows(root):
    rows = []
    for channel in ("asr-submit", "vlm-submit", "tts-submit", "completion"):
        path = root / "observers" / f"{channel}.jsonl"
        require(path.is_file(), f"missing observer: {channel}")
        previous = -1
        for index, row in enumerate(_observer_json_lines(path)):
            require(set(row) == {"index", "channel", "observed_monotonic_ns",
                                 "forwarded_monotonic_ns", "raw_sha256", "raw_base64"},
                    f"observer schema changed: {channel}:{index}")
            require(row["index"] == index and row["channel"] == channel,
                    f"observer identity/index mismatch: {channel}:{index}")
            observed = row["observed_monotonic_ns"]
            forwarded = row["forwarded_monotonic_ns"]
            require(type(observed) is int and type(forwarded) is int and
                    observed > previous and forwarded >= observed,
                    f"observer clock is not ordered: {channel}:{index}")
            previous = observed
            raw = base64.b64decode(row["raw_base64"], validate=True)
            require(sha256(raw) == row["raw_sha256"], f"observer raw hash mismatch: {channel}:{index}")
            message = audit.unpack_message(raw)
            require(isinstance(message, dict), f"observer did not carry a map: {channel}")
            rows.append({"channel": channel, "observed": observed, "forwarded": forwarded,
                         "raw_sha256": row["raw_sha256"], "message": message})
    return rows


def _submit_parts(row):
    message = row["message"]
    require(message.get("type") == "submit", "submit observer saw another message type")
    request_id = message.get("request_id")
    require(isinstance(request_id, str) and request_id and message["data"].get("request_id") == request_id,
            "submit request identity mismatch")
    request = message["data"].get("request", {})
    inputs = request.get("inputs", {})
    require(isinstance(inputs, dict), "submit inputs are not a map")
    return request_id, request, inputs


def _parse_native_logs(root):
    parsed = {"vlm": {}, "asr": {}, "tts": {}, "tts_complete": {}}
    patterns = {
        "vlm": re.compile(r"qwen38 request request_id=(\S+).*?prompt_tokens=(\d+).*?"
                          r"prefill_tokens=(\d+).*?prefix_reused=(\d+).*?generated_tokens=(\d+).*?"
                          r"prefill_ms=([0-9.]+).*?decode_ms=([0-9.]+).*?elapsed_ms=(\d+)"),
        "asr": re.compile(r"asr request request_id=(\S+).*?audio_samples=(\d+).*?elapsed_ms=(\d+)"),
        "tts": re.compile(r"tts stream_check request_id=(\S+).*?identical=(\d+).*?"
                          r"chunks=(\d+).*?first_audio_ms=([0-9.]+)"),
        "tts_complete": re.compile(r"tts request request_id=(\S+).*?elapsed_ms=(\d+)"),
    }
    for kind, pattern in patterns.items():
        path = root / f"{kind.split('_')[0]}.log"
        require(path.is_file(), f"missing native log: {path.name}")
        for line in path.read_text(errors="strict").splitlines():
            match = pattern.search(line)
            if not match:
                continue
            rid = match.group(1)
            require(rid not in parsed[kind], f"duplicate native log identity: {kind}:{rid}")
            if kind == "vlm":
                value = {"prompt_tokens": int(match.group(2)), "prefill_tokens": int(match.group(3)),
                         "prefix_reused": int(match.group(4)), "generated_tokens": int(match.group(5)),
                         "prefill_ns": int(Decimal(match.group(6)) * 1_000_000),
                         "decode_ns": int(Decimal(match.group(7)) * 1_000_000),
                         "elapsed_ns": int(Decimal(match.group(8)) * 1_000_000)}
            elif kind == "asr":
                value = {"audio_samples": int(match.group(2)),
                         "elapsed_ns": int(Decimal(match.group(3)) * 1_000_000)}
            elif kind == "tts":
                value = {"identical": int(match.group(2)), "chunks": int(match.group(3)),
                         "first_audio_ns": int(Decimal(match.group(4)) * 1_000_000)}
                require(value["identical"] == 1, f"TTS stream/terminal mismatch: {rid}")
            else:
                value = {"elapsed_ns": int(Decimal(match.group(2)) * 1_000_000)}
            parsed[kind][rid] = value
    return parsed


def _vln_response_id(request_id):
    return request_id[:-4] if request_id.endswith(":vlm") else request_id


def _tts_parent(request_id):
    match = re.match(r"^(req-\d+):tts(?::(\d+))?$", request_id)
    return (match.group(1), int(match.group(2) or 0)) if match else (None, None)


def _summary(samples):
    result = audit.summarize(samples)
    values = [float(sample["value"]) for sample in samples]
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    cv = math.sqrt(variance) / mean if mean > 0.0 else 0.0
    result["summary"]["cv"] = cv
    if "summary_ms" in result:
        result["summary_ms"]["cv"] = cv
    result["definitions"] = {"cv": "population standard deviation / arithmetic mean"}
    return result


def _add_metric(metrics, name, value, response, start=None, end=None, unit="ns"):
    require(type(value) is int, f"non-integer metric {name}")
    if unit == "ns" and not name.endswith("signed"):
        require(value >= 0, f"negative metric {name}: {response['response_id']}")
    row = {"response_id": response["response_id"], "arm": response["arm"],
           "session": response["session"], "round": response["round"], "value": value}
    if start is not None:
        row["start_monotonic_ns"] = start
        row["end_monotonic_ns"] = end
    metrics[name].append(row)


def _empty_response(connection, response_id, segment, first_ns):
    return {"response_id": response_id, "connection": connection, "segment": segment,
            "first_event_ns": first_ns, "text_deltas": [], "audio": [], "audio_done": [],
            "terminals": [], "accepted_cursors": [], "accepted_seen": set()}


def analyze(root, verify_sealed=False):
    """Independently decode one complete seven-arm run and return its report."""
    root = Path(root)
    contract = load_contract(root / "preregistration.json")
    plan = load_json(root / "plan.json")
    require(plan == make_plan(contract), "stored plan differs from seeded preregistration")
    run = load_json(root / "run.json")
    require(run.get("schema") == "umminfer-realtime-first-audio-live/v1", "unknown run schema")
    require("driver_error" not in run, f"driver failed: {run.get('driver_error')}")
    require(run.get("physical_microphone_speaker") == "NOT_COVERED" and
            run.get("perceptual_listening_quality") == "NOT_COVERED" and
            run.get("wan") == "NOT_COVERED", "physical/WAN scope changed")
    require(run.get("final_health", {}).get("in_flight") == 0, "final in_flight is nonzero")
    require(run.get("cleanup", {}).get("owned_processes_exited") is True and
            run["cleanup"].get("selected_gpus_idle") is True, "cleanup failed")
    if verify_sealed:
        audit.verify_manifest(root)

    for fixture_name, fixture in contract["fixtures"].items():
        path = root / "fixtures" / fixture["file"]
        require(path.is_file() and file_sha256(path) == fixture["sha256"],
                f"fixture hash mismatch: {fixture_name}")
    for name, expected in run.get("binaries", {}).items():
        require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
                f"invalid binary digest: {name}")
    for role, spec in contract["topology"].items():
        if role not in ("asr", "vlm", "tts"):
            continue
        actual = run["models"][role]["model_manifest_sha256"]
        require(actual == spec["model_manifest_sha256"], f"model identity changed: {role}")

    responses = {}
    connections = {}
    metrics = defaultdict(list)
    history_turns = []
    ack_requests = 0
    accepted_acks = 0
    ignored_acks = 0
    special_cutins = 0

    for arm in plan["arms"]:
        arm_root = root / "arms" / arm["name"]
        require(arm_root.is_dir(), f"missing arm directory: {arm['name']}")
        observer_rows = _observer_rows(arm_root)
        native = _parse_native_logs(arm_root)
        submits = {"asr": {}, "vlm": {}, "tts": {}}
        completion_streams = defaultdict(list)
        completion_final = defaultdict(list)
        for row in observer_rows:
            channel = row["channel"]
            message = row["message"]
            if channel == "completion":
                if message.get("type") == "stream":
                    rid = message.get("request_id")
                    stage = message.get("from_stage")
                    require(isinstance(rid, str) and isinstance(stage, str), "bad native stream identity")
                    completion_streams[rid].append({"ns": row["observed"], "stage": stage,
                                                    "chunk": message.get("chunk", {}),
                                                    "raw_sha256": row["raw_sha256"]})
                elif message.get("type") == "complete":
                    rid = message.get("request_id")
                    require(isinstance(rid, str) and rid, "missing native completion id")
                    # Abort/late compute can legitimately produce more than one
                    # native terminal. Coordinator ignores late duplicates; keep
                    # all raw records and bind the successful ASR result below.
                    completion_final[rid].append({"ns": row["observed"], "message": message,
                                                  "raw_sha256": row["raw_sha256"]})
                elif message.get("type") != "stage_ready":
                    raise FirstAudioError(f"unknown completion observer type: {message.get('type')}")
                continue
            if message.get("type") != "submit":
                require(message.get("type") in ("admin", "shutdown", "abort"),
                        f"unknown submit-channel message: {message.get('type')}")
                continue
            request_id, request, inputs = _submit_parts(row)
            role = channel.split("-")[0]
            require(request_id not in submits[role], f"duplicate {role} submit: {request_id}")
            submits[role][request_id] = {"ns": row["observed"], "request": request,
                                         "inputs": inputs, "raw_sha256": row["raw_sha256"]}

        actual_frame_names = {path.stem for path in (arm_root / "frames").glob("*.jsonl")}
        planned = {row["name"]: row for row in arm["connections"]}
        require(planned.keys() <= actual_frame_names, "a planned connection capture is missing")
        frame_names = sorted(actual_frame_names)
        for connection_name in frame_names:
            records = audit.decode_capture(arm_root / "frames" / f"{connection_name}.jsonl")
            events = [row for row in records if "event" in row]
            planned_connection = planned.get(connection_name)
            measured_connection = planned_connection is not None
            connection = {"name": connection_name, "arm": arm["name"],
                          "session": planned_connection["session"] if planned_connection else 90,
                          "measured_connection": measured_connection}
            connections[connection_name + "@" + arm["name"]] = {"events": len(events)}
            transcripts = {}
            stops = {}
            turn_audio = bytearray()
            response_order = []
            local_accepted = defaultdict(int)
            local_issued = defaultdict(set)
            local_seen = set()
            history = []
            cancel_index = {}

            for record in events:
                event = record["event"]
                kind = event["type"]
                direction = record["direction"]
                ns = record["monotonic_ns"]
                if direction == "tx" and kind == "input_audio_buffer.append":
                    pcm = base64.b64decode(event.get("audio", ""), validate=True)
                    turn_audio.extend(pcm)
                if direction == "rx" and kind == "input_audio_buffer.speech_stopped":
                    segment = event.get("segment_id")
                    audio_ms = event.get("audio_ms")
                    require(type(audio_ms) is int and audio_ms >= 0,
                            "speech_stopped lacks integer audio_ms")
                    # The frame that triggers VAD can be captured after RX stop.
                    # audio_ms is the server's exact committed PCM byte boundary;
                    # retain the TX sequence through ASR final and truncate to it.
                    stops[segment] = {"ns": ns, "audio_bytes": audio_ms * 32}
                if direction == "rx" and kind == "transcription.completed":
                    segment = event.get("segment_id")
                    require(type(segment) is int and segment not in transcripts,
                            "duplicate ASR final segment")
                    text = event.get("text")
                    require(isinstance(text, str) and text.strip(), "empty ASR final")
                    require(segment in stops and stops[segment]["audio_bytes"] <= len(turn_audio),
                            "ASR final lacks its complete committed PCM")
                    committed = bytes(turn_audio[:stops[segment]["audio_bytes"]])
                    turn_audio = bytearray(turn_audio[stops[segment]["audio_bytes"]:])
                    transcripts[segment] = {"text": text, "ns": ns, "index": record["capture_index"],
                                            "audio_sha256": sha256(committed)}
                rid = event.get("response_id")
                if rid is not None:
                    require(isinstance(rid, str) and rid, "empty response id")
                if direction == "tx" and kind == "output_audio_buffer.playback_ack":
                    ack_requests += 1
                    continue
                if direction != "rx" or rid is None:
                    continue
                segment = event.get("segment_id")
                response_key = f"{arm['name']}:{rid}"
                if response_key not in responses:
                    require(segment is not None, "response lacks segment")
                    responses[response_key] = _empty_response(connection_name, rid, segment, ns)
                    responses[response_key]["arm"] = arm["name"]
                    responses[response_key]["arm_root"] = str(arm_root.relative_to(root))
                    responses[response_key]["session"] = connection["session"]
                    response_order.append(response_key)
                response = responses[response_key]
                require(response["connection"] == connection_name and response["arm"] == arm["name"],
                        f"cross-arm/session response leak: {rid}")
                if segment is not None:
                    require(response["segment"] == segment, "response segment changed")
                if rid in cancel_index:
                    require(kind not in STALE or record["capture_index"] < cancel_index[rid],
                            f"stale old-response event after cancellation: {rid}")
                if kind == "response.text.delta":
                    text = event.get("text")
                    require(isinstance(text, str) and text, "empty VLM text delta")
                    response["text_deltas"].append({"ns": ns, "text": text,
                                                    "cursor": event.get("text_cursor")})
                    joined = "".join(row["text"] for row in response["text_deltas"]).encode()
                    require(event.get("text_cursor") == len(joined), "public text cursor mismatch")
                elif kind == "audio.delta":
                    pcm = base64.b64decode(event.get("audio", ""), validate=True)
                    require(pcm and len(pcm) % 2 == 0 and event.get("format") == "pcm16" and
                            event.get("sample_rate") == 24000 and event.get("channels") == 1,
                            "invalid public PCM")
                    cursor = event.get("text_cursor")
                    end_cursor = event.get("text_end_cursor")
                    require(type(cursor) is int and type(end_cursor) is int and
                            0 <= cursor <= end_cursor, "invalid audio cursors")
                    local_issued[rid].add(cursor)
                    response["audio"].append({"ns": ns, "pcm": pcm, "cursor": cursor,
                                              "end_cursor": end_cursor})
                elif kind == "audio.done":
                    cursor = event.get("text_cursor")
                    require(type(cursor) is int and cursor >= 0, "invalid audio.done cursor")
                    local_issued[rid].add(cursor)
                    response["audio_done"].append({"ns": ns, "cursor": cursor})
                elif kind == "response.playback_ack":
                    cursor = event.get("text_cursor")
                    require(type(cursor) is int and cursor in local_issued[rid] and
                            cursor >= local_accepted[rid], "accepted unissued/decreasing ACK")
                    require((rid, cursor) not in local_seen or cursor == local_accepted[rid],
                            "impossible ACK state")
                    local_seen.add((rid, cursor))
                    local_accepted[rid] = cursor
                    response["accepted_cursors"].append(cursor)
                    accepted_acks += 1
                    if event.get("final") is True:
                        full_text = "".join(row["text"] for row in response["text_deltas"])
                        require(cursor == len(full_text.encode()), "final ACK is not full text")
                        history.extend([{"role": "user", "content": transcripts[response["segment"]]["text"]},
                                        {"role": "assistant", "content": full_text}])
                        history_turns.append({"response_id": rid, "response_key": response_key,
                                              "reason": "final_ack",
                                              "text_cursor": cursor, "monotonic_ns": ns})
                elif kind in TERMINALS:
                    response["terminals"].append({"kind": kind, "ns": ns,
                                                  "index": record["capture_index"], "event": event})
                    if kind == "response.cancelled":
                        cancel_index[rid] = record["capture_index"]
                        require(event.get("reason") == "barge_in" and
                                event.get("history_recorded") is True, "invalid barge-in cancellation")
                        cursor = event.get("text_cursor")
                        acknowledged = event.get("acknowledged_text")
                        full_text = "".join(row["text"] for row in response["text_deltas"])
                        require(cursor == local_accepted[rid] and isinstance(acknowledged, str) and
                                full_text.encode()[:cursor].decode() == acknowledged,
                                "acknowledged-prefix history mismatch")
                        history.extend([{"role": "user", "content": transcripts[response["segment"]]["text"]},
                                        {"role": "assistant", "content": acknowledged}])
                        history_turns.append({"response_id": rid, "response_key": response_key,
                                              "reason": "barge_in",
                                              "text_cursor": cursor, "monotonic_ns": ns})

            # Each planned connection has exactly its preregistered number of ASR turns.
            if measured_connection:
                require(len(transcripts) == planned_connection["turns"],
                        f"turn count mismatch on {connection_name}")
            for segment, transcript in sorted(transcripts.items()):
                rid = next((candidate for candidate in response_order
                            if responses[candidate]["segment"] == segment), None)
                require(rid is not None, f"ASR final has no audio response: {connection_name}:{segment}")
                response = responses[rid]
                response["transcript"] = transcript["text"]
                response["asr_final_ns"] = transcript["ns"]
                response["stop_ns"] = stops[segment]["ns"]
                response["asr_audio_sha256"] = transcript["audio_sha256"]
                response["history_at_final"] = list(history)
                turn_index = sorted(transcripts).index(segment)
                if measured_connection:
                    if arm["prefix_cache"]:
                        response["round"] = turn_index
                        response["measured"] = turn_index >= planned_connection["warmup_turns"]
                    else:
                        response["round"] = planned_connection["round"]
                        response["measured"] = True
                else:
                    response["round"] = None
                    response["measured"] = False

            # Bind final ASR native submit/completion by exact PCM, time and transcript.
            for rid in response_order:
                response = responses[rid]
                if "asr_final_ns" not in response or not response["measured"]:
                    continue
                candidates = []
                for request_id, submit in submits["asr"].items():
                    if request_id.startswith("realtime-preview-"):
                        continue
                    audio = submit["inputs"].get("audio")
                    if not isinstance(audio, bytes) or sha256(audio) != response["asr_audio_sha256"]:
                        continue
                    for done in completion_final.get(request_id, []):
                        if done["message"].get("from_stage") != "asr":
                            continue
                        result = done["message"].get("result", {})
                        if done["message"].get("success") is not True or result.get("text") != response["transcript"]:
                            continue
                        if submit["ns"] <= done["ns"]:
                            candidates.append((submit["ns"], request_id, submit, done))
                require(candidates, f"no exact native ASR binding for {rid}")
                _, request_id, submit, done = min(candidates, key=lambda row: abs(
                    row[3]["ns"] - response["asr_final_ns"]))
                response["asr_request_id"] = request_id
                response["asr_submit_ns"] = submit["ns"]
                response["asr_native_complete_ns"] = done["ns"]

            # Validate each native VLM prompt against raw-wire history at observation time.
            for rid in response_order:
                response = responses[rid]
                raw_rid = response["response_id"]
                prompt_candidates = [(key, value) for key, value in submits["vlm"].items()
                                     if _vln_response_id(key) == raw_rid]
                require(len(prompt_candidates) == 1, f"native VLM prompt count mismatch: {raw_rid}")
                prompt_key, prompt = prompt_candidates[0]
                messages = normalize_messages(prompt["inputs"].get("messages"))
                response["prompt_request_id"] = prompt_key
                response["prompt_ns"] = prompt["ns"]
                response["prompt_messages"] = messages
                response["prompt_raw_sha256"] = prompt["raw_sha256"]
                response["prompt_messages_sha256"] = sha256(json.dumps(
                    messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
                # History events are reconstructed independently below from terminal order;
                # the current transcript must always be the final user message.
                require(messages and messages[-1] == {"role": "user", "content": response["transcript"]},
                        f"current ASR prompt mismatch: {rid}")
                response["history_prefix_expected"] = messages[:-1]

        # Native timing joins and exact stage decomposition for this arm.
        for response in responses.values():
            if response.get("arm") != arm["name"] or "prompt_ns" not in response:
                continue
            rid = response["response_id"]
            vlm_streams = completion_streams.get(response["prompt_request_id"], [])
            vlm_streams = [row for row in vlm_streams if row["stage"] == "vlm" and
                           isinstance(row["chunk"].get("text"), str) and row["chunk"]["text"]]
            response["native_text"] = vlm_streams
            tts_submit_rows = []
            for key, submit in submits["tts"].items():
                parent, leg = _tts_parent(key)
                if parent == rid:
                    tts_submit_rows.append((leg, key, submit))
            tts_submit_rows.sort()
            response["tts_submits"] = [{"leg": leg, "request_id": key, "ns": submit["ns"],
                                        "text": submit["inputs"].get("text"),
                                        "stream": submit["inputs"].get("stream"),
                                        "raw_sha256": submit["raw_sha256"]}
                                       for leg, key, submit in tts_submit_rows]
            if response["measured"]:
                require(response["tts_submits"] and response["tts_submits"][0]["leg"] == 0 and
                        response["tts_submits"][0]["stream"] is True,
                        f"first streaming TTS submit missing: {rid}")
                first_tts = response["tts_submits"][0]
                first_text = first_tts["text"]
                require(isinstance(first_text, str) and first_text, "empty first TTS text")
                cumulative = ""
                ready = None
                for chunk in vlm_streams:
                    cumulative += chunk["chunk"]["text"]
                    if cumulative.lstrip(" \t\r\n").startswith(first_text):
                        ready = chunk["ns"]
                        break
                require(ready is not None, f"first TTS text was not observed in native VLM text: {rid}")
                response["first_tts_text"] = first_tts["text"]
                response["first_sentence_ready_native_ns"] = ready
                first_pcm = [row for row in completion_streams.get(first_tts["request_id"], [])
                             if row["stage"] == "tts" and
                             isinstance(row["chunk"].get("audio"), bytes) and row["chunk"]["audio"]]
                require(first_pcm, f"first native PCM missing: {rid}")
                response["first_tts_native_pcm_ns"] = first_pcm[0]["ns"]
                require(response["audio"], f"measured response has no public PCM: {rid}")
                response["first_public_audio_ns"] = response["audio"][0]["ns"]
                response["first_public_audio_cursor"] = response["audio"][0]["cursor"]
                response["first_public_audio_end_cursor"] = response["audio"][0]["end_cursor"]
                response["first_public_audio_end_cursor"] = response["audio"][0]["end_cursor"]
                response["text"] = "".join(row["text"] for row in response["text_deltas"])
                response["text_sha256"] = sha256(response["text"].encode())
                response["pcm"] = b"".join(row["pcm"] for row in response["audio"])
                response["pcm_sha256"] = sha256(response["pcm"])
                public_groups = defaultdict(bytearray)
                for public_row in response["audio"]:
                    public_groups[(public_row["cursor"], public_row["end_cursor"])].extend(
                        public_row["pcm"])
                require(len(public_groups) == len(response["tts_submits"]),
                        f"public PCM is missing an entire TTS leg: {rid}")
                expected_cursor = 0
                response_text_bytes = response["text"].encode()
                for tts_leg in response["tts_submits"]:
                    matching = sorted(((bounds, pcm) for bounds, pcm in public_groups.items()
                                       if bounds[0] >= expected_cursor),
                                      key=lambda row: row[0])
                    require(matching,
                            f"public PCM cursor does not match the response's TTS leg: {arm['name']}:{rid}")
                    (bounds, public_pcm) = matching[0]
                    # Whitespace-only sentence-buffer cuts intentionally have no
                    # TTS leg. Permit exactly those gaps, never dropped prose.
                    gap = response_text_bytes[expected_cursor:bounds[0]]
                    require(all(byte in b" \t\r\n" for byte in gap),
                            f"non-whitespace PCM cursor gap: {arm['name']}:{rid}")
                    expected_cursor = bounds[0]
                    end_cursor = bounds[1]
                    require(end_cursor > expected_cursor and end_cursor <= len(response_text_bytes),
                            f"invalid public PCM end cursor: {arm['name']}:{rid}")
                    # Verify the actual response/cursor identity, not merely the
                    # sorted order of equally sized PCM groups.
                    spoken_text = response_text_bytes[expected_cursor:end_cursor].decode().strip(
                        " \t\r\n")
                    require(spoken_text == tts_leg["text"],
                            f"public PCM text cursor content mismatch: {arm['name']}:{rid}:{tts_leg['request_id']}")
                    native_pcm = b"".join(
                        row["chunk"]["audio"] for row in completion_streams.get(tts_leg["request_id"], [])
                        if row["stage"] == "tts" and
                        isinstance(row["chunk"].get("audio"), bytes))
                    require(native_pcm and native_pcm == bytes(public_pcm),
                            f"native/public PCM mismatch: {arm['name']}:{rid}:{tts_leg['request_id']}")
                    expected_cursor = end_cursor
                require(all(byte in b" \t\r\n" for byte in response_text_bytes[expected_cursor:]),
                        f"public PCM cursors omit non-whitespace response text: {arm['name']}:{rid}")
                response["native_public_pcm_exact"] = True
                response["transcript_sha256"] = sha256(response["transcript"].encode())
                response["pair_key"] = f"{arm['name']}:{response['session']}:{response['round']}"
                native_vlm = native["vlm"].get(response["prompt_request_id"],
                                               native["vlm"].get(rid))
                require(native_vlm is not None, f"missing native VLM log: {rid}")
                response["native_vlm"] = native_vlm
                native_asr = native["asr"].get(response["asr_request_id"])
                require(native_asr is not None, f"missing native ASR log: {rid}")
                response["native_asr"] = native_asr
                native_tts = native["tts"].get(first_tts["request_id"])
                require(native_tts is not None, f"missing first TTS native log: {rid}")
                response["native_first_tts"] = native_tts
                require(response["terminals"] and response["terminals"][0]["kind"] == "response.completed",
                        f"measured response did not complete: {rid}")
                require(response["audio_done"] and response["audio_done"][0]["cursor"] ==
                        len(response["text"].encode()), "audio.done is not full text")
                require(response["accepted_cursors"] and response["accepted_cursors"][-1] ==
                        len(response["text"].encode()), "missing final playback ACK")

                def add(name, value, start=None, end=None, unit="ns"):
                    _add_metric(metrics, name, value, response, start, end, unit)
                add("asr_submit_to_native_complete", response["asr_native_complete_ns"] -
                    response["asr_submit_ns"], response["asr_submit_ns"], response["asr_native_complete_ns"])
                add("asr_native_final_compute", native_asr["elapsed_ns"])
                add("asr_native_complete_to_public_final", response["asr_final_ns"] -
                    response["asr_native_complete_ns"], response["asr_native_complete_ns"], response["asr_final_ns"])
                add("asr_final_to_vlm_submit_signed", response["prompt_ns"] - response["asr_final_ns"],
                    response["asr_final_ns"], response["prompt_ns"])
                add("vlm_submit_to_first_native_text", vlm_streams[0]["ns"] - response["prompt_ns"],
                    response["prompt_ns"], vlm_streams[0]["ns"])
                add("vlm_native_first_to_last_text", vlm_streams[-1]["ns"] - vlm_streams[0]["ns"],
                    vlm_streams[0]["ns"], vlm_streams[-1]["ns"])
                add("vlm_prefill", native_vlm["prefill_ns"])
                add("vlm_decode", native_vlm["decode_ns"])
                add("vlm_elapsed", native_vlm["elapsed_ns"])
                add("vlm_prefix_reused_tokens", native_vlm["prefix_reused"], unit="tokens")
                add("first_native_text_to_first_sentence_ready", ready - vlm_streams[0]["ns"],
                    vlm_streams[0]["ns"], ready)
                add("first_sentence_ready_to_tts_submit", first_tts["ns"] - ready,
                    ready, first_tts["ns"])
                add("tts_submit_to_first_native_pcm", response["first_tts_native_pcm_ns"] - first_tts["ns"],
                    first_tts["ns"], response["first_tts_native_pcm_ns"])
                add("tts_native_compute_to_first_pcm", native_tts["first_audio_ns"])
                add("tts_first_native_pcm_to_public_audio", response["first_public_audio_ns"] -
                    response["first_tts_native_pcm_ns"], response["first_tts_native_pcm_ns"],
                    response["first_public_audio_ns"])
                add("ttfa_asr_final_to_first_audio", response["first_public_audio_ns"] -
                    response["asr_final_ns"], response["asr_final_ns"], response["first_public_audio_ns"])
                add("ttfa_vlm_submit_to_first_audio", response["first_public_audio_ns"] - response["prompt_ns"],
                    response["prompt_ns"], response["first_public_audio_ns"])
                add("first_tts_segment_utf8_bytes", len(first_text.encode()), unit="bytes")
                if arm["prefix_cache"]:
                    require(response["round"] > 0, "warm measured turn unexpectedly has round zero")
                else:
                    require(native_vlm["prefix_reused"] == 0, f"cold prefix was reused: {rid}")

    # Cross-arm identities are unique because each run uses one monotonically advancing server id space per arm.
    # IDs may repeat across arms; qualify internal maps for terminal checks, while pair_key remains stable for A/B.
    terminal_counts = defaultdict(int)
    for response in responses.values():
        kinds = [row["kind"] for row in response["terminals"]]
        require(all(kinds.count(kind) == 1 for kind in set(kinds)), "duplicate terminal kind")
        require("response.failed" not in kinds, f"response failed: {response['response_id']}")
        if response["audio_done"]:
            completed = [row for row in response["terminals"] if row["kind"] == "response.completed"]
            require(completed and response["audio_done"][0]["ns"] < completed[0]["ns"],
                    "audio.done does not precede completion")
        for kind in kinds:
            terminal_counts[kind] += 1

    # Reconstruct full history chronologically per connection and compare every prompt prefix.
    for response in responses.values():
        if "prompt_ns" not in response:
            continue
        prior = []
        for turn in sorted((row for row in history_turns if
                            responses[row["response_key"]]["connection"] == response["connection"] and
                            responses[row["response_key"]]["arm"] == response["arm"] and
                            row["monotonic_ns"] <= response["prompt_ns"]),
                           key=lambda row: row["monotonic_ns"]):
            source = responses[turn["response_key"]]
            prior.append({"role": "user", "content": source["transcript"]})
            if turn["reason"] == "final_ack":
                text = "".join(row["text"] for row in source["text_deltas"])
            else:
                terminal = next(row for row in source["terminals"] if row["kind"] == "response.cancelled")
                text = terminal["event"]["acknowledged_text"]
            prior.append({"role": "assistant", "content": text})
        require(response["history_prefix_expected"] == prior,
                f"independent history prompt mismatch: {response['response_id']}")

    # Driver-supplied regression indices are only locators; raw bytes decide the outcome.
    regression = run.get("regression", {})
    if regression:
        echo = regression.get("echo_ab", {})
        last_arm = plan["arms"][-1]["name"]
        last_root = root / "arms" / last_arm
        echo_records = audit.decode_capture(last_root / "frames" / f"{echo['connection']}.jsonl")
        protected = [row for row in echo_records if row["direction"] == "rx" and
                     echo["protected_start_capture_index"] <= row["capture_index"] <
                     echo["protected_end_capture_index"] and row.get("event") and
                     row["event"]["type"] in CUTINS]
        require(not protected, "protected echo produced a raw cut-in")
        require(any(row["capture_index"] == echo["disabled_cancel_capture_index"] and
                    row.get("event", {}).get("type") == "response.cancelled" and
                    row["event"].get("response_id") == echo["response_id"] for row in echo_records),
                "disabled echo raw cancellation is missing")
        for mode, rid in echo["double_talk_response_ids"].items():
            index = echo["double_talk_cancel_capture_indices"][mode]
            require(any(row["capture_index"] == index and row.get("event", {}).get("type") ==
                        "response.cancelled" and row["event"].get("response_id") == rid
                        for row in echo_records), f"double-talk raw cancellation missing: {mode}")
        special_cutins = len(protected)
        require(regression.get("delayed_barge", {}).get("history_recorded") is True and
                regression["delayed_barge"].get("stale_events") == 0,
                "delayed barge-in driver record is incomplete")

    rendered_metrics = {name: _summary(samples) for name, samples in sorted(metrics.items())}
    by_arm = {}
    for arm in plan["arms"]:
        by_arm[arm["name"]] = {}
        for name, samples in sorted(metrics.items()):
            selected = [row for row in samples if row["arm"] == arm["name"]]
            if selected:
                by_arm[arm["name"]][name] = _summary(selected)
    response_rows = []
    for response in responses.values():
        if not response.get("measured"):
            continue
        response_rows.append({
            "response_id": response["response_id"], "pair_key": response["pair_key"],
            "arm": response["arm"], "session": response["session"], "round": response["round"],
            "connection": response["connection"], "segment": response["segment"],
            "transcript": response["transcript"], "transcript_sha256": response["transcript_sha256"],
            "text_utf8_bytes": len(response["text"].encode()), "text_sha256": response["text_sha256"],
            "prompt_messages_sha256": response["prompt_messages_sha256"],
            "prompt_raw_sha256": response["prompt_raw_sha256"],
            "pcm_bytes": len(response["pcm"]), "pcm_sha256": response["pcm_sha256"],
            "native_public_pcm_exact": response["native_public_pcm_exact"],
            "audio_delta_count": len(response["audio"]),
            "first_tts_text": response["first_tts_text"],
            "first_tts_segment_utf8_bytes": len(response["first_tts_text"].encode()),
            "first_public_audio_cursor": response["first_public_audio_cursor"],
            "first_public_audio_end_cursor": response["first_public_audio_end_cursor"],
            "prefix_reused": response["native_vlm"]["prefix_reused"],
            "prompt_tokens": response["native_vlm"]["prompt_tokens"],
            "generated_tokens": response["native_vlm"]["generated_tokens"],
            "timestamps": {key: response[key] for key in (
                "stop_ns", "asr_submit_ns", "asr_native_complete_ns", "asr_final_ns", "prompt_ns",
                "first_sentence_ready_native_ns", "first_tts_native_pcm_ns", "first_public_audio_ns")},
            "native_text_first_ns": response["native_text"][0]["ns"],
            "native_text_last_ns": response["native_text"][-1]["ns"],
            "tts_submit_ns": response["tts_submits"][0]["ns"],
        })
    response_rows.sort(key=lambda row: (row["arm"], row["round"], row["session"]))
    warm_reuse = [row["prefix_reused"] for row in response_rows if row["arm"].endswith("warm-short")]
    require(warm_reuse and max(warm_reuse) > 0, "warm arm never exercised prefix reuse")
    primary = rendered_metrics["ttfa_asr_final_to_first_audio"]["summary_ms"]
    cv = primary["cv"]
    return {
        "schema": "umminfer-realtime-first-audio-evidence/v1", "status": "PASS",
        "source_commit": run["source_commit"], "contract_sha256": plan["contract_sha256"],
        "analyzer_sha256": file_sha256(Path(__file__).resolve()),
        "binaries": run["binaries"], "models": run["models"],
        "counts": {"arms": len(plan["arms"]), "measured_responses": len(response_rows),
                   "all_server_responses": len(responses), "asr_finals": len(response_rows),
                   "history_turns": len(history_turns), "ack_requests": ack_requests,
                   "accepted_playback_acks": accepted_acks,
                   "ignored_playback_acks": ack_requests - accepted_acks,
                   "protected_echo_cutins": special_cutins,
                   "terminal_callbacks": dict(sorted(terminal_counts.items()))},
        "coverage": {"observed_timing_class": "REAL_DIGITAL_LOOPBACK_WITH_NATIVE_BOUNDARY_OBSERVERS",
                     "physical_microphone_speaker": "NOT_COVERED", "wan": "NOT_COVERED",
                     "perceptual_listening_quality": "NOT_COVERED",
                     "network_backpressure": "two-second consumer read pause after first recorded audio frame",
                     "warm_prefix_reused_max": max(warm_reuse)},
        "responses": response_rows, "metrics": {"global": rendered_metrics, "by_arm": by_arm,
            "definitions": {"quantile": "linear interpolation at (n-1)*p",
                            "outliers": "outside [q1-1.5*iqr,q3+1.5*iqr], retained in all samples",
                            "coefficient_of_variation": cv,
                            "native_log_resolution": "native decimal-ms durations converted exactly to integer ns; observer and WS clocks remain original integer ns"}},
        "history_turns": history_turns,
        "regression": {"delayed_barge": regression.get("delayed_barge", {}),
                       "echo_and_double_talk": "PASS_FROM_RAW_FRAMES" if regression else "NOT_RUN"},
        "cleanup": run["cleanup"], "final_health": run["final_health"],
        "gpu_preflight": run.get("gpu_preflight"), "gpu_postflight": run.get("gpu_postflight"),
    }


def render_report(report):
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def compare_reports(baseline, candidate):
    """Fail closed on missing pairs, quality drift or preregistered guardrails."""
    require(baseline.get("schema") == candidate.get("schema") ==
            "umminfer-realtime-first-audio-evidence/v1", "unknown A/B report schema")
    require(baseline["contract_sha256"] == candidate["contract_sha256"], "A/B contract changed")
    require(baseline["counts"]["measured_responses"] == candidate["counts"]["measured_responses"],
            "A/B sample count changed")
    base = {row["pair_key"]: row for row in baseline["responses"]}
    cand = {row["pair_key"]: row for row in candidate["responses"]}
    require(len(base) == len(baseline["responses"]) and len(cand) == len(candidate["responses"]),
            "duplicate A/B pair key")
    require(base.keys() == cand.keys(), "A/B response pairs differ")
    mismatches = []
    for key in sorted(base):
        left, right = base[key], cand[key]
        for field in ("arm", "session", "round", "transcript", "transcript_sha256", "text_utf8_bytes",
                      "text_sha256", "prompt_messages_sha256", "pcm_bytes", "pcm_sha256",
                      "first_tts_text", "first_public_audio_cursor", "first_public_audio_end_cursor"):
            if left[field] != right[field]:
                mismatches.append({"pair_key": key, "field": field,
                                   "baseline": left[field], "candidate": right[field]})
    require(not mismatches, f"A/B quality/identity mismatch: {mismatches[:8]}")

    def metric(report, arm, name):
        return report["metrics"]["by_arm"][arm][name]["summary_ms"]
    primary_base = metric(baseline, "concurrent-cold-short", "ttfa_asr_final_to_first_audio")
    primary_cand = metric(candidate, "concurrent-cold-short", "ttfa_asr_final_to_first_audio")
    serial_base = metric(baseline, "serial-cold-short", "ttfa_asr_final_to_first_audio")
    serial_cand = metric(candidate, "serial-cold-short", "ttfa_asr_final_to_first_audio")
    median_gain = 1.0 - primary_cand["median"] / primary_base["median"]
    p95_change = primary_cand["p95"] / primary_base["p95"] - 1.0
    serial_change = serial_cand["median"] / serial_base["median"] - 1.0
    primary_cv = {"baseline": primary_base["cv"], "candidate": primary_cand["cv"]}
    gates = {"primary_median_gain_at_least_20_percent": median_gain >= 0.20,
             "primary_p95_regression_at_most_5_percent": p95_change <= 0.05,
             "serial_median_regression_at_most_5_percent": serial_change <= 0.05,
             "serial_baseline_cv_at_most_0_10": serial_base["cv"] <= 0.10,
             "serial_candidate_cv_at_most_0_10": serial_cand["cv"] <= 0.10,
             "exact_quality_equivalence": True}
    return {"schema": "umminfer-realtime-first-audio-ab/v1",
            "status": "PASS" if all(gates.values()) else "FAIL",
            "baseline_source_commit": baseline["source_commit"],
            "candidate_source_commit": candidate["source_commit"],
            "pairs": len(base), "quality_mismatches": mismatches,
            "primary": {"baseline_ms": primary_base, "candidate_ms": primary_cand,
                        "median_gain_fraction": median_gain, "p95_change_fraction": p95_change,
                        "coefficient_of_variation": primary_cv},
            "serial_guardrail": {"baseline_ms": serial_base, "candidate_ms": serial_cand,
                                 "median_change_fraction": serial_change},
            "gates": gates,
            "coverage": {"physical_playback": "NOT_COVERED", "wan": "NOT_COVERED",
                         "perceptual_listening": "NOT_COVERED"}}
