#!/usr/bin/env python3
"""Independent raw public/native PCM verifier; does not import the live driver."""

import argparse
import base64
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import random
import struct
import wave

EXPECTED_CONTINUATION = "like you're in the middle of a thought or a"
EXPECTED_CONTINUATION_SHA256 = "3eaa2791bba77428558a314086e2264e3279123b03d89a4998f24a3cce11773d"
TERMINALS = {"complete", "error", "cancelled"}
WS_STALE = {"audio.delta", "audio.done", "response.text.delta",
            "response.completed", "response.failed"}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write_pcm_pair(root, relative, public_pcm, native_pcm):
    public_path = root / "reconstructed" / f"{relative}.public.pcm"
    native_path = root / "reconstructed" / f"{relative}.native.pcm"
    public_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.write_bytes(public_pcm)
    native_path.write_bytes(native_pcm)
    return {"public_file": str(public_path.relative_to(root)),
            "native_file": str(native_path.relative_to(root))}


def bind_chunks(public_chunks, native_chunks, label, check_index=True):
    """Fail closed on a missing, duplicated, stale, or reordered native chunk."""
    require(len(public_chunks) == len(native_chunks),
            f"public/native chunk count mismatch: {label}: "
            f"{len(public_chunks)} != {len(native_chunks)}")
    require([chunk.get("index") for chunk in native_chunks] ==
            list(range(len(native_chunks))),
            f"native chunk index gap/reorder: {label}")
    rows = []
    for index, (public, native) in enumerate(zip(public_chunks, native_chunks)):
        if check_index:
            require(public["stream_index"] == native.get("index"),
                    f"public/native chunk index mismatch: {label}:{index}")
        public_pcm = public["pcm"]
        native_pcm = native["audio"]
        require(public_pcm == native_pcm,
                f"public/native PCM mismatch: {label}:{index}")
        require(public_pcm and len(public_pcm) % 2 == 0,
                f"empty or odd PCM chunk: {label}:{index}")
        rows.append({"ordinal": index, "stream_index": native.get("index"),
                     "first_frame": native.get("first_frame"),
                     "frames": native.get("frames"), "bytes": len(public_pcm),
                     "pcm_sha256": sha(public_pcm)})
    return rows


def cursor_for_text(text, submitted, minimum, label):
    text_bytes = text.encode()
    spoken = submitted.strip(" \t\r\n").encode()
    require(spoken, f"empty native TTS text: {label}")
    cursor = text_bytes.find(spoken, minimum)
    require(cursor >= minimum, f"native TTS text is absent from public response: {label}")
    return cursor, cursor + len(spoken)


def require_whitespace_gap(text_bytes, begin, end, label):
    require(0 <= begin <= end <= len(text_bytes), f"cursor outside response text: {label}")
    require(all(byte in b" \t\r\n" for byte in text_bytes[begin:end]),
            f"non-whitespace PCM cursor gap: {label}")


def native_public_text_matches(native_text, public_text, cancelled):
    return native_text.startswith(public_text) if cancelled else native_text == public_text


def check_terminal(events, expected, label):
    unified = [event["unified"] for event in events if "unified" in event]
    terminals = [event for event in unified if event.get("kind") in TERMINALS]
    require(len(terminals) == 1 and terminals[0]["kind"] == expected,
            f"terminal state mismatch: {label}: {terminals}")
    require(unified and unified[-1] is terminals[0],
            f"stale public event after terminal: {label}")
    seq = [event["seq"] for event in unified]
    require(seq == list(range(len(seq))), f"public event sequence gap: {label}")
    return terminals[0]


def parse_http_raw(path):
    raw = Path(path).read_bytes()
    head, marker, body = raw.partition(b"\r\n\r\n")
    require(marker and head.startswith(b"HTTP/1.1 200"), f"invalid raw HTTP response: {path}")
    headers = {}
    for line in head.split(b"\r\n")[1:]:
        key, _, value = line.partition(b":")
        headers[key.strip().lower()] = value.strip().lower()
    if headers.get(b"transfer-encoding") == b"chunked":
        source = bytearray(body)
        decoded = bytearray()
        while True:
            marker_at = source.find(b"\r\n")
            require(marker_at >= 0, "truncated HTTP chunk header")
            size = int(bytes(source[:marker_at]).split(b";", 1)[0], 16)
            needed = marker_at + 2 + size + 2
            require(len(source) >= needed, "truncated HTTP chunk")
            require(source[marker_at + 2 + size:needed] == b"\r\n",
                    "invalid raw HTTP chunk ending")
            payload = source[marker_at + 2:marker_at + 2 + size]
            del source[:needed]
            if size == 0:
                require(not source, "raw HTTP bytes after terminal chunk")
                break
            decoded.extend(payload)
        body = bytes(decoded)
    else:
        if b"content-length" in headers:
            require(len(body) == int(headers[b"content-length"]),
                    "HTTP content-length mismatch")
    return head, headers, body, raw


def parse_sse(body, label):
    events = []
    done = 0
    for frame in body.split(b"\n\n"):
        if not frame:
            continue
        require(frame.startswith(b"data: "), f"invalid SSE frame: {label}")
        data = frame[6:]
        if data == b"[DONE]":
            done += 1
            continue
        events.append(json.loads(data))
    require(done == 1, f"SSE [DONE] count mismatch: {label}: {done}")
    return events


def submission_or_none(message, channel):
    if message.get("type") in ("abort", "shutdown"):
        return None
    require(message.get("type") == "submit", f"unexpected message on {channel}")
    return message


def native_records(root):
    import msgpack

    submits = {}
    completion_streams = defaultdict(list)
    completion_final = defaultdict(list)
    all_completion = []
    for channel in ("asr-submit", "vlm-submit", "tts-submit"):
        for row in read_jsonl(root / "observers" / f"{channel}.jsonl"):
            raw = base64.b64decode(row["raw_base64"], validate=True)
            require(sha(raw) == row["raw_sha256"], f"observer raw hash mismatch: {channel}")
            message = msgpack.unpackb(raw, raw=False, strict_map_key=False)
            if submission_or_none(message, channel) is None:
                continue
            request_id = message["request_id"]
            require(request_id not in submits, f"duplicate native submit: {request_id}")
            submits[request_id] = {"ns": row["observed_monotonic_ns"],
                                   "raw_sha256": row["raw_sha256"],
                                   "channel": channel,
                                   "inputs": message["data"]["request"].get("inputs", {})}
    for row in read_jsonl(root / "observers" / "completion.jsonl"):
        raw = base64.b64decode(row["raw_base64"], validate=True)
        require(sha(raw) == row["raw_sha256"], "completion observer raw hash mismatch")
        message = msgpack.unpackb(raw, raw=False, strict_map_key=False)
        request_id = message.get("request_id")
        record = {"ns": row["observed_monotonic_ns"], "message": message,
                  "raw_sha256": row["raw_sha256"]}
        all_completion.append(record)
        if not request_id:
            continue
        if message.get("type") == "stream":
            completion_streams[request_id].append(record)
        elif message.get("type") == "complete":
            completion_final[request_id].append(record)
    return submits, completion_streams, completion_final, all_completion


def vlm_request_id(submits, parent):
    candidates = [request_id for request_id in (parent + ":vlm", parent)
                  if request_id in submits]
    require(len(candidates) == 1, f"native VLM request id is missing or ambiguous: {parent}")
    return candidates[0]


def tts_legs(submits, parent):
    rows = []
    for request_id, submit in submits.items():
        suffix = request_id[len(parent):] if request_id.startswith(parent) else ""
        if suffix == ":tts":
            leg = 0
            streaming = False
        elif suffix.startswith(":tts:") and suffix[5:].isdigit():
            leg = int(suffix[5:])
            streaming = True
        else:
            continue
        rows.append({"leg": leg, "request_id": request_id, "streaming": streaming,
                     **submit})
    rows.sort(key=lambda row: (row["leg"], row["ns"], row["request_id"]))
    return rows


def native_tts_chunks(streams, request_id):
    rows = []
    for record in streams.get(request_id, []):
        message = record["message"]
        if message.get("from_stage") != "tts":
            continue
        chunk = message.get("chunk", {})
        if isinstance(chunk.get("audio"), bytes) and chunk["audio"]:
            rows.append(chunk)
    return rows


def wav_pcm(wav_bytes, label):
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        require((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) ==
                (1, 2, 24000), f"invalid native/public WAV contract: {label}")
        return wav.readframes(wav.getnframes())


def successful_terminal(finals, request_id, label):
    rows = finals.get(request_id, [])
    require(len(rows) == 1, f"native terminal count mismatch: {label}: {len(rows)}")
    message = rows[0]["message"]
    require(message.get("from_stage") == "tts" and message.get("success") is True,
            f"native TTS did not complete: {label}")
    return message.get("result", {})


def verify_direct_case(root, case, submits, finals, body, raw):
    name = case["name"]
    capture = read_json(root / "http" / name / "capture.json")
    candidates = []
    for request_id, submit in submits.items():
        if (submit["channel"] == "tts-submit" and ":" not in request_id and
                submit["inputs"].get("text") == EXPECTED_CONTINUATION and
                capture["started_monotonic_ns"] <= submit["ns"] <=
                capture["finished_monotonic_ns"]):
            candidates.append((request_id, submit))
    require(len(candidates) == 1, f"direct public/native TTS binding mismatch: {name}")
    request_id, submit = candidates[0]
    result = successful_terminal(finals, request_id, name)
    native_wav = result["data"]
    native_pcm = wav_pcm(native_wav, name)
    if case["form"] == "direct-wav":
        require(body == native_wav, f"direct public/native WAV mismatch: {name}")
        public_pcm = native_pcm
    else:
        require(case["form"] == "direct-pcm" and body == native_pcm,
                f"direct public/native PCM mismatch: {name}")
        public_pcm = body
    files = write_pcm_pair(root, f"http/{name}/leg-0", public_pcm, native_pcm)
    leg = {"leg": 0, "native_request_id": request_id,
           "text": EXPECTED_CONTINUATION, "text_cursor": 0,
           "text_end_cursor": len(EXPECTED_CONTINUATION.encode()),
           "pcm_bytes": len(public_pcm), "public_pcm_sha256": sha(public_pcm),
           "native_pcm_sha256": sha(native_pcm), "wav_bytes": len(native_wav),
           "wav_sha256": sha(native_wav), "chunks": [], **files}
    return {"name": name, "public_response_id": request_id, "form": case["form"],
            "terminal": "complete", "text": EXPECTED_CONTINUATION,
            "text_utf8_bytes": len(EXPECTED_CONTINUATION.encode()),
            "text_sha256": sha(EXPECTED_CONTINUATION.encode()),
            "raw_http_bytes": len(raw), "raw_http_sha256": sha(raw),
            "public_artifact_count": 1, "native_vlm_request_id": None,
            "legs": [leg], "public_pcm_bytes": len(public_pcm),
            "public_pcm_sha256": sha(public_pcm),
            "client_playback_ack": {"response_id": request_id,
                                    "text_cursor": len(EXPECTED_CONTINUATION.encode()),
                                    "pcm_bytes": len(public_pcm), "pcm_sha256": sha(public_pcm),
                                    "terminal": "complete", "class": "digital-consumption-only"}}


def verify_http_case(root, case, submits, streams, finals):
    name = case["name"]
    case_root = root / "http" / name
    capture = read_json(case_root / "capture.json")
    request = read_json(case_root / "request.json")
    if case["form"].startswith("direct-"):
        require(case.get("route") == "/v1/audio/speech" and
                request.get("model") == "qwen3-unified" and
                request.get("input") == EXPECTED_CONTINUATION and
                request.get("voice") == "vivian" and
                request.get("response_format") == case["form"].split("-", 1)[1],
                f"direct public speech request mismatch: {name}")
    else:
        require(request.get("model") == "qwen3-unified" and
                request.get("modalities") == ["text", "audio"] and
                request.get("temperature") == 0 and request.get("seed") == 5630241,
                f"public request scope/sampling mismatch: {name}")
        messages = request.get("messages")
        require(isinstance(messages, list) and len(messages) == 1 and
                messages[0].get("role") == "user" and
                isinstance(messages[0].get("content"), str) and
                not any(key in request for key in ("image", "video", "input_audio")),
                f"public request is not text-only VLM input: {name}")
    reads = read_jsonl(case_root / "reads.jsonl")
    require(len(reads) == capture["read_count"] and
            [row["index"] for row in reads] == list(range(len(reads))),
            f"raw read record gap: {name}")
    rng = random.Random(5630241 + case["numeric_seed"])
    require([row["requested_bytes"] for row in reads] ==
            [rng.randint(1, 4093) for _ in reads],
            f"seeded HTTP fragmentation schedule mismatch: {name}")
    if name.endswith("-delayed"):
        require(capture.get("pause", {}).get("duration_ms") == 3000,
                f"delayed HTTP read was not retained: {name}")
    _, _, body, raw = parse_http_raw(case_root / "response.raw")
    require(sha(raw) == capture["raw_sha256"] and sha(body) == capture["body_sha256"],
            f"HTTP capture digest mismatch: {name}")
    if case["form"].startswith("direct-"):
        return verify_direct_case(root, case, submits, finals, body, raw)
    form = case["form"]
    if form == "json-wav":
        response = json.loads(body)
        events = []
        public_id = response["id"]
        text = response["choices"][0]["message"]["content"]
        terminal = {"kind": "complete", "legs": response.get("legs", [])}
        require(response["choices"][0]["finish_reason"] == "stop",
                f"nonstream finish state mismatch: {name}")
    else:
        events = parse_sse(body, name)
        expected = "cancelled" if name.endswith("-cancel") else "complete"
        terminal = check_terminal(events, expected, name)
        public_id = events[0]["id"]
        text_events = [event for event in events
                       if event.get("unified", {}).get("kind") == "vlm_delta"]
        text = "".join(event["unified"]["text"] for event in text_events)
        for event in text_events:
            require(event["choices"][0]["delta"].get("content") == event["unified"]["text"],
                    f"SSE/public text projection mismatch: {name}")
    require(public_id.startswith("http-"), f"unexpected public HTTP id: {name}:{public_id}")
    vlm_id = vlm_request_id(submits, public_id)
    native_text = "".join(
        record["message"].get("chunk", {}).get("text", "")
        for record in streams.get(vlm_id, [])
        if record["message"].get("from_stage") == "vlm" and
        isinstance(record["message"].get("chunk", {}).get("text"), str))
    cancelled = name.endswith("-cancel")
    require(native_public_text_matches(native_text, text, cancelled),
            f"native/public VLM text mismatch: {name}")
    legs = tts_legs(submits, public_id)
    require(legs, f"no native TTS legs: {name}")
    if cancelled:
        require(case["capture"].get("abort", {}).get("reply", {}).get("status") == 200,
                f"cancel request was not accepted: {name}")
    else:
        require(terminal.get("kind") == "complete", f"unsuccessful public terminal: {name}")
        require(all(leg.get("state") == "completed" for leg in terminal.get("legs", [])
                    if leg.get("stage") == "tts"), f"terminal TTS leg state mismatch: {name}")
    public_chunks = []
    public_artifacts = []
    if form == "json-wav":
        public_artifacts = response.get("artifacts", [])
    else:
        for event in events:
            unified = event.get("unified", {})
            if unified.get("kind") == "artifact":
                public_artifacts.append(unified["artifact"])
    # Slice multi-part WAV artifacts independently. PCM chunks are total=1.
    grouped = defaultdict(list)
    for artifact in public_artifacts:
        pcm = base64.b64decode(artifact["data"], validate=True)
        grouped[artifact["artifact_id"]].append((artifact, pcm))
        if artifact.get("mime_type") == "audio/pcm":
            require(artifact.get("index") == 0 and artifact.get("total") == 1,
                    f"unexpected sliced PCM artifact: {name}")
            require((artifact.get("format"), artifact.get("sample_rate"),
                     artifact.get("channels")) == ("pcm16", 24000, 1),
                    f"invalid public PCM descriptor: {name}")
            public_chunks.append({"stream_index": artifact.get("stream_index"), "pcm": pcm})
    reconstructed_wavs = []
    for artifact_id, parts in grouped.items():
        parts.sort(key=lambda row: row[0].get("index", 0))
        if form == "json-wav":
            require(len(parts) == 1, f"nonstream artifact was unexpectedly sliced: {name}")
        else:
            require([row[0].get("index") for row in parts] == list(range(len(parts))),
                    f"artifact slice gap: {name}:{artifact_id}")
            require(all(row[0].get("total") == len(parts) for row in parts),
                    f"artifact slice total mismatch: {name}:{artifact_id}")
        data = b"".join(row[1] for row in parts)
        descriptor = parts[-1][0]
        require(descriptor.get("sha256") == sha(data) and descriptor.get("bytes") == len(data),
                f"public artifact digest mismatch: {name}:{artifact_id}")
        if descriptor.get("mime_type") == "audio/wav":
            reconstructed_wavs.append(data)
    report = {"name": name, "public_response_id": public_id, "form": form,
              "terminal": terminal["kind"], "text": text, "text_utf8_bytes": len(text.encode()),
              "text_sha256": sha(text.encode()), "raw_http_bytes": len(raw),
              "raw_http_sha256": sha(raw), "public_artifact_count": len(public_artifacts),
              "native_vlm_request_id": vlm_id, "legs": [],
              "client_playback_ack": None}
    if form == "sse-pcm":
        expected_chunks = []
        for leg in legs:
            chunks = native_tts_chunks(streams, leg["request_id"])
            for chunk in chunks:
                expected_chunks.append((leg, chunk))
        if cancelled:
            require(0 < len(public_chunks) <= len(expected_chunks),
                    f"cancel did not retain a valid partial public prefix: {name}")
            for index, public in enumerate(public_chunks):
                leg, native = expected_chunks[index]
                require(public["stream_index"] == native.get("index") and
                        public["pcm"] == native["audio"],
                        f"cancel public PCM prefix mismatch: {name}:{index}")
            report["cancel_prefix_chunks"] = len(public_chunks)
            report["cancel_prefix_pcm_bytes"] = sum(len(row["pcm"]) for row in public_chunks)
            report["cancel_prefix_pcm_sha256"] = sha(b"".join(row["pcm"] for row in public_chunks))
            return report
        require(len(public_chunks) == len(expected_chunks),
                f"missing native/public PCM chunk: {name}")
        cursor = 0
        chunk_cursor = 0
        for leg in legs:
            native_chunks = native_tts_chunks(streams, leg["request_id"])
            terminal_result = successful_terminal(finals, leg["request_id"], name)
            native_terminal_pcm = wav_pcm(terminal_result["data"], name)
            native_pcm = b"".join(chunk["audio"] for chunk in native_chunks)
            require(native_pcm == native_terminal_pcm,
                    f"native stream/terminal PCM mismatch: {name}:{leg['request_id']}")
            begin, end = cursor_for_text(text, leg["inputs"]["text"], cursor,
                                         f"{name}:{leg['request_id']}")
            require_whitespace_gap(text.encode(), cursor, begin,
                                   f"{name}:{leg['request_id']}")
            count = len(native_chunks)
            bound = bind_chunks(public_chunks[chunk_cursor:chunk_cursor + count],
                                native_chunks, f"{name}:{leg['request_id']}")
            chunk_cursor += count
            public_pcm = b"".join(row["pcm"] for row in public_chunks[
                chunk_cursor - count:chunk_cursor])
            files = write_pcm_pair(root, f"http/{name}/leg-{leg['leg']}", public_pcm, native_pcm)
            report["legs"].append({"leg": leg["leg"], "native_request_id": leg["request_id"],
                                   "text": leg["inputs"]["text"], "text_cursor": begin,
                                   "text_end_cursor": end, "pcm_bytes": len(public_pcm),
                                   "public_pcm_sha256": sha(public_pcm),
                                   "native_pcm_sha256": sha(native_pcm), "chunks": bound, **files})
            cursor = end
        require(chunk_cursor == len(public_chunks), f"stale public PCM chunk: {name}")
        require_whitespace_gap(text.encode(), cursor, len(text.encode()), name)
    else:
        require(not cancelled, f"cancel fixture must use streaming PCM: {name}")
        require(len(legs) == 1 and len(reconstructed_wavs) == 1,
                f"one-shot public WAV count mismatch: {name}")
        leg = legs[0]
        result = successful_terminal(finals, leg["request_id"], name)
        native_wav = result["data"]
        public_wav = reconstructed_wavs[0]
        require(public_wav == native_wav, f"public/native WAV mismatch: {name}")
        require(leg["inputs"]["text"] == text, f"one-shot TTS text mismatch: {name}")
        public_pcm = wav_pcm(public_wav, name)
        native_pcm = wav_pcm(native_wav, name)
        files = write_pcm_pair(root, f"http/{name}/leg-0", public_pcm, native_pcm)
        report["legs"] = [{"leg": 0, "native_request_id": leg["request_id"],
                           "text": text, "text_cursor": 0,
                           "text_end_cursor": len(text.encode()),
                           "pcm_bytes": len(public_pcm), "public_pcm_sha256": sha(public_pcm),
                           "native_pcm_sha256": sha(native_pcm),
                           "wav_bytes": len(public_wav), "wav_sha256": sha(public_wav),
                           "chunks": [], **files}]
    report["public_pcm_bytes"] = sum(leg["pcm_bytes"] for leg in report["legs"])
    report["public_pcm_sha256"] = sha(b"".join(
        (root / leg["public_file"]).read_bytes() for leg in report["legs"]))
    # This is a harness consumption acknowledgement, not a claim of physical
    # speaker playback. Realtime's server-side playback ACK is verified below.
    report["client_playback_ack"] = {
        "response_id": public_id, "text_cursor": len(text.encode()),
        "pcm_bytes": report["public_pcm_bytes"], "pcm_sha256": report["public_pcm_sha256"],
        "terminal": terminal["kind"], "class": "digital-consumption-only"}
    return report


def parse_ws_frame(raw, direction):
    require(len(raw) >= 2, "short WebSocket frame")
    first, second = raw[0], raw[1]
    offset = 2
    length = second & 0x7F
    if length == 126:
        length = struct.unpack_from("!H", raw, offset)[0]
        offset += 2
    elif length == 127:
        length = struct.unpack_from("!Q", raw, offset)[0]
        offset += 8
    masked = bool(second & 0x80)
    require(masked == (direction == "tx"), "unexpected WebSocket mask direction")
    mask = None
    if masked:
        mask = raw[offset:offset + 4]
        offset += 4
    payload = raw[offset:]
    require(len(payload) == length, "WebSocket frame length mismatch")
    if mask:
        payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    return bool(first & 0x80), first & 0x0F, payload


def ws_events(path):
    fragments = None
    output = []
    for row in read_jsonl(path):
        raw = base64.b64decode(row["raw_base64"], validate=True)
        fin, opcode, payload = parse_ws_frame(raw, row["direction"])
        if opcode in (8, 9, 10):
            continue
        if opcode != 0:
            fragments = bytearray()
        require(fragments is not None, "WebSocket continuation without start")
        fragments.extend(payload)
        if not fin:
            continue
        message = bytes(fragments)
        fragments = None
        output.append({"direction": row["direction"], "ns": row["monotonic_ns"],
                       "capture_index": row["index"], "event": json.loads(message)})
    return output


def validate_realtime_acks(rx, response_id):
    issued = set()
    accepted = 0
    accepted_rows = []
    for row in rx:
        event = row["event"]
        kind = event["type"]
        if kind == "audio.delta":
            cursor = event.get("text_cursor")
            require(type(cursor) is int and cursor >= 0,
                    f"invalid issued realtime cursor: {response_id}")
            issued.add(cursor)
        elif kind == "audio.done":
            cursor = event.get("text_cursor")
            require(type(cursor) is int and cursor >= 0,
                    f"invalid audio.done cursor: {response_id}")
            issued.add(cursor)
        elif kind == "response.playback_ack":
            cursor = event.get("text_cursor")
            require(type(cursor) is int and cursor in issued,
                f"accepted unissued realtime ACK: {response_id}:{cursor}")
            require(cursor >= accepted,
                    f"accepted decreasing realtime ACK: {response_id}:{cursor}")
            accepted = cursor
            accepted_rows.append(row)
    return accepted_rows


def verify_realtime(root, manifest, submits, streams, finals, all_completion):
    realtime = manifest.get("realtime", {})
    if realtime.get("status") == "NOT_RUN":
        return {"status": "NOT_RUN", "responses": []}
    identities = read_json(root / "realtime/responses.json")
    require(len(identities) == 29, "realtime response count mismatch")
    require(not read_jsonl(root / "realtime/turn-failures.jsonl"),
            "realtime turn failure records are present")
    release = read_json(root / "realtime/backpressure-release.json")
    require(release.get("configured_readers") == 29 and
            0 < release.get("paused_readers", 0) <= 29 and
            release.get("pending_native_minus_public", 0) >= 4096 and
            release.get("native_audio_events", 0) -
            release.get("public_audio_events", 0) ==
            release["pending_native_minus_public"],
            f"audio-only bounded backlog was not proven: {release}")
    release_ids = release.get("paused_response_ids")
    require(isinstance(release_ids, list) and
            len(release_ids) == len(set(release_ids)) == release["paused_readers"],
            "release paused response ids are missing or duplicated")
    pauses = read_json(root / "realtime/read-pauses.json")
    pause_by_capture = {Path(row["capture"]).name: row for row in pauses}
    reports = []
    paused_native_events = 0
    paused_public_events = 0
    paused_response_ids = set()
    known_response_ids = {identity["response_id"] for identity in identities}
    require(set(release_ids) <= known_response_ids,
            "release paused response id is not a measured realtime response")
    for identity in identities:
        response_id = identity["response_id"]
        transcript = identity.get("transcript", "").lower()
        require("sixteen" in transcript and "lantern" in transcript,
                f"realtime instruction ASR transcript mismatch: {response_id}: {transcript}")
        capture_name = Path(identity["capture"]).name
        records = ws_events(root / "realtime/frames" / capture_name)
        for row in records:
            if row["event"].get("type") == "response.playback_ack":
                require(row["event"].get("response_id") in known_response_ids,
                        f"accepted foreign realtime ACK: {row['event']}")
        rx = [row for row in records if row["direction"] == "rx" and
              row["event"].get("response_id") == response_id]
        text_rows = [row for row in rx if row["event"]["type"] == "response.text.delta"]
        audio_rows = [row for row in rx if row["event"]["type"] == "audio.delta"]
        done_rows = [row for row in rx if row["event"]["type"] == "audio.done"]
        terminal_rows = [row for row in rx if row["event"]["type"] in
                         ("response.completed", "response.failed", "response.cancelled")]
        require(len(terminal_rows) == 1 and
                terminal_rows[0]["event"]["type"] == "response.completed",
                f"realtime terminal mismatch: {response_id}")
        terminal_index = terminal_rows[0]["capture_index"]
        require(not [row for row in rx if row["capture_index"] > terminal_index and
                     row["event"]["type"] in WS_STALE],
                f"stale realtime event after terminal: {response_id}")
        text = ""
        for row in text_rows:
            text += row["event"]["text"]
            require(row["event"].get("text_cursor") == len(text.encode()),
                    f"realtime public text cursor mismatch: {response_id}")
        require(sha(text.encode()) == identity["text_sha256"],
                f"realtime text capture mismatch: {response_id}")
        require(audio_rows and len(done_rows) == 1 and
                done_rows[0]["event"].get("text_cursor") == len(text.encode()),
                f"realtime audio.done mismatch: {response_id}")
        ack_rows = validate_realtime_acks(rx, response_id)
        final_acks = [row for row in ack_rows if row["event"].get("final") is True]
        require(len(final_acks) == 1 and
                final_acks[0]["event"].get("text_cursor") == len(text.encode()),
                f"missing final server playback ACK: {response_id}")
        legs = tts_legs(submits, response_id)
        require(legs and legs[0]["leg"] == 0, f"realtime native TTS legs missing: {response_id}")
        public_groups = defaultdict(list)
        for row in audio_rows:
            event = row["event"]
            pcm = base64.b64decode(event["audio"], validate=True)
            require(pcm and len(pcm) % 2 == 0 and
                    (event.get("format"), event.get("sample_rate"), event.get("channels")) ==
                    ("pcm16", 24000, 1), f"invalid realtime PCM: {response_id}")
            public_groups[(event["text_cursor"], event["text_end_cursor"])].append(
                {"pcm": pcm, "ns": row["ns"], "capture_index": row["capture_index"]})
        require(len(public_groups) == len(legs),
                f"realtime public PCM is missing a whole leg: {response_id}")
        cursor = 0
        leg_reports = []
        for leg in legs:
            native_chunks = native_tts_chunks(streams, leg["request_id"])
            result = successful_terminal(finals, leg["request_id"], response_id)
            native_pcm = b"".join(chunk["audio"] for chunk in native_chunks)
            require(native_pcm == wav_pcm(result["data"], response_id),
                    f"realtime native stream/terminal mismatch: {leg['request_id']}")
            matches = sorted((bounds for bounds in public_groups if bounds[0] >= cursor),
                             key=lambda bounds: bounds[0])
            require(matches, f"realtime cursor has no public PCM: {leg['request_id']}")
            bounds = matches[0]
            require(0 <= bounds[0] < bounds[1] <= len(text.encode()),
                    f"realtime cursor outside response text: {leg['request_id']}")
            require_whitespace_gap(text.encode(), cursor, bounds[0], response_id)
            spoken = text.encode()[bounds[0]:bounds[1]].decode().strip(" \t\r\n")
            require(spoken == leg["inputs"]["text"],
                    f"realtime native/public cursor text mismatch: {leg['request_id']}")
            public_rows = public_groups[bounds]
            bound = bind_chunks(public_rows, native_chunks, leg["request_id"], check_index=False)
            public_pcm = b"".join(row["pcm"] for row in public_rows)
            relative = f"realtime/round-{identity['round']}-session-{identity['session']}/leg-{leg['leg']}"
            files = write_pcm_pair(root, relative, public_pcm, native_pcm)
            leg_reports.append({"leg": leg["leg"], "native_request_id": leg["request_id"],
                                "text": leg["inputs"]["text"], "text_cursor": bounds[0],
                                "text_end_cursor": bounds[1], "pcm_bytes": len(public_pcm),
                                "public_pcm_sha256": sha(public_pcm),
                                "native_pcm_sha256": sha(native_pcm), "chunks": bound, **files})
            cursor = bounds[1]
        require_whitespace_gap(text.encode(), cursor, len(text.encode()), response_id)
        public_pcm = b"".join((root / leg["public_file"]).read_bytes() for leg in leg_reports)
        reports.append({**identity, "text_utf8_bytes": len(text.encode()),
                        "text_sha256": sha(text.encode()), "terminal": "response.completed",
                        "final_playback_ack": final_acks[0]["event"],
                        "public_pcm_bytes": len(public_pcm), "public_pcm_sha256": sha(public_pcm),
                        "legs": leg_reports})
        if identity["round"] == 0:
            pause = pause_by_capture.get(capture_name)
            if pause is None:
                require(response_id not in release_ids,
                        f"release-time paused reader lacks pause evidence: {response_id}")
            else:
                require(response_id in release_ids and pause.get("response_id") == response_id,
                        f"pause response identity mismatch: {response_id}")
                paused_response_ids.add(response_id)
                start, end = pause["start_monotonic_ns"], pause["end_monotonic_ns"]
                require(end > start,
                        f"delayed-read window was not measured: {response_id}")
                for record in all_completion:
                    message = record["message"]
                    if (start <= record["ns"] <= end and message.get("type") == "stream" and
                            message.get("request_id") in
                            {leg["request_id"] for leg in legs} and
                            message.get("from_stage") == "tts" and
                            isinstance(message.get("chunk", {}).get("audio"), bytes) and
                            message["chunk"]["audio"]):
                        paused_native_events += 1
                paused_public_events += sum(
                    1 for row in rx if start < row["ns"] <= end and
                    row["event"]["type"] == "audio.delta")
    require(paused_response_ids == set(release_ids),
            "paused reader set does not match the proven audio-backlog release")
    require(paused_public_events == 0,
            f"public audio reads occurred inside supposedly paused windows: {paused_public_events}")
    return {"status": "PASS", "responses": reports,
            "delayed_read": {"paused_readers": len(paused_response_ids),
                             "native_audio_events_during_pause": paused_native_events,
                             "public_audio_events_during_pause": paused_public_events,
                             "release": release,
                             "bounded_lossless_queue_boundary": 4096,
                             "all_pcm_and_cursors_drained_exact": True}}


def verify_repeats(http_reports):
    groups = defaultdict(list)
    for report in http_reports:
        if report["terminal"] == "cancelled":
            continue
        group = report["name"].rsplit("-r", 1)[0]
        if report["name"].endswith("-delayed") or report["name"].endswith("-retry"):
            group = "continuation-sse-pcm"
        groups[group].append(report)
    results = {}
    for group, rows in groups.items():
        baseline = rows[0]
        for row in rows[1:]:
            require(row["text"] == baseline["text"], f"repeat text mismatch: {group}")
            require([leg["public_pcm_sha256"] for leg in row["legs"]] ==
                    [leg["public_pcm_sha256"] for leg in baseline["legs"]],
                    f"repeat PCM mismatch: {group}:{row['name']}")
            require([len(leg["chunks"]) for leg in row["legs"]] ==
                    [len(leg["chunks"]) for leg in baseline["legs"]],
                    f"repeat chunk-count mismatch: {group}:{row['name']}")
        results[group] = {"responses": len(rows), "text_sha256": baseline["text_sha256"],
                          "leg_pcm_sha256": [leg["public_pcm_sha256"] for leg in baseline["legs"]],
                          "exact": True}
    for prompt in ("opening", "continuation"):
        left = next(row for row in http_reports if row["name"] == f"{prompt}-sse-wav-r0")
        right = next(row for row in http_reports if row["name"] == f"{prompt}-json-wav-r0")
        require(left["text"] == right["text"] and
                left["legs"][0]["wav_sha256"] == right["legs"][0]["wav_sha256"],
                f"HTTP SSE/nonstream JSON WAV mismatch: {prompt}")
    opening_stream = next(row for row in http_reports if row["name"] == "opening-sse-pcm-r0")
    opening_wav = next(row for row in http_reports if row["name"] == "opening-sse-wav-r0")
    require(opening_stream["public_pcm_sha256"] == opening_wav["public_pcm_sha256"],
            "single-leg opening stream/nonstream PCM mismatch")
    direct_pcm = next(row for row in http_reports if row["name"] == "direct-continuation-pcm-r0")
    direct_wav = next(row for row in http_reports if row["name"] == "direct-continuation-wav-r0")
    require(direct_pcm["public_pcm_sha256"] == direct_wav["public_pcm_sha256"],
            "direct public WAV payload does not equal direct PCM")
    accepted = [leg for row in http_reports for leg in row["legs"]
                if leg["text"] == EXPECTED_CONTINUATION]
    require(accepted and all(leg["pcm_bytes"] == 737280 and
                             leg["public_pcm_sha256"] == EXPECTED_CONTINUATION_SHA256 and
                             leg["native_pcm_sha256"] == EXPECTED_CONTINUATION_SHA256
                             for leg in accepted),
            "accepted repetition-penalty continuation PCM was not reproduced")
    return {"groups": results, "accepted_continuation_legs": len(accepted),
            "accepted_continuation_pcm_bytes": 737280,
            "accepted_continuation_pcm_sha256": EXPECTED_CONTINUATION_SHA256}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = read_json(root / "capture-manifest.json")
    require(manifest["schema"] == "umminfer-public-speech-pcm-capture/v1" and
            manifest["status"] == "CAPTURED", "capture is incomplete")
    require(manifest["final_health"].get("in_flight") == 0 and
            manifest["gpu_postflight"]["0"]["memory_used_mib"] <= 64 and
            not manifest["gpu_postflight"]["0"]["compute_processes"],
            "cleanup gate failed")
    submits, streams, finals, all_completion = native_records(root)
    http_reports = [verify_http_case(root, case, submits, streams, finals)
                    for case in manifest["http_cases"]]
    require(len(http_reports) == 25, f"public HTTP response count mismatch: {len(http_reports)}")
    repeats = verify_repeats(http_reports)
    realtime = verify_realtime(root, manifest, submits, streams, finals, all_completion)
    report = {
        "schema": "umminfer-public-speech-pcm-verification/v1", "status": "PASS",
        "source_commit": manifest["source_commit"], "seed": manifest["seed"],
        "binaries": manifest["binaries"],
        "models": {name: value["model_manifest_sha256"] for name, value in manifest["models"].items()},
        "source_sha256": manifest["source_sha256"],
        "counts": {
            "public_http_responses": len(http_reports),
            "successful_public_http_responses": sum(row["terminal"] == "complete" for row in http_reports),
            "cancelled_public_http_responses": sum(row["terminal"] == "cancelled" for row in http_reports),
            "public_http_pcm_legs": sum(len(row["legs"]) for row in http_reports),
            "public_http_pcm_chunks": sum(len(leg["chunks"]) for row in http_reports for leg in row["legs"]),
            "public_http_pcm_bytes": sum(row.get("public_pcm_bytes", row.get("cancel_prefix_pcm_bytes", 0))
                                         for row in http_reports),
            "realtime_responses": len(realtime["responses"]),
            "realtime_pcm_legs": sum(len(row["legs"]) for row in realtime["responses"]),
            "realtime_pcm_chunks": sum(len(leg["chunks"]) for row in realtime["responses"] for leg in row["legs"]),
            "realtime_pcm_bytes": sum(row["public_pcm_bytes"] for row in realtime["responses"]),
        },
        "http": http_reports, "repeats": repeats, "realtime": realtime,
        "cancel_retry": {"cancel_terminal": "cancelled", "retry_matches_fresh_repeat": True,
                         "cancel_prefix_chunks": next(row["cancel_prefix_chunks"] for row in http_reports
                                                      if row["terminal"] == "cancelled")},
        "cleanup": {"final_in_flight": manifest["final_health"]["in_flight"],
                    "owned_processes_exited": True, "gpu_postflight": manifest["gpu_postflight"]},
        "limits": manifest["limits"],
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "counts": report["counts"],
                      "delayed_read": realtime.get("delayed_read", "NOT_RUN"),
                      "verification_sha256": sha(args.output.read_bytes())}, sort_keys=True))


if __name__ == "__main__":
    main()
