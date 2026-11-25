# SPDX-License-Identifier: Apache-2.0
"""Deterministic, out-of-process multimodal rollout recording.

The recorder is a normal client of the public HTTP/SSE endpoint. Serving code
is not modified and no inference or optimizer engine is duplicated. Schema v2
keeps clean narrative text, visual-slot decisions, leg failures, terminal legs,
and content-addressed media outputs. Media bytes never live in the JSONL.
There are no wall-clock fields or random ids: identical inputs and canned
response bytes produce identical record and artifact bytes.
"""

import base64
import copy
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
from urllib.parse import unquote_to_bytes

TRAJECTORY_SCHEMA_VERSION = 2
SUPPORTED_TRAJECTORY_SCHEMA_VERSIONS = frozenset((1, 2))

TERMINAL_COMPLETED = "completed"
TERMINAL_FAILED = "failed"
TERMINAL_CANCELLED = "cancelled"
TERMINAL_STATES = frozenset(
    (TERMINAL_COMPLETED, TERMINAL_FAILED, TERMINAL_CANCELLED))
TERMINAL_EVENT_FOR_STATE = {
    TERMINAL_COMPLETED: "complete",
    TERMINAL_FAILED: "error",
    TERMINAL_CANCELLED: "cancelled",
}
TRUNCATED_MESSAGE = "stream ended before [DONE]"
_HEX_256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_SUFFIX = re.compile(r"\.[a-z0-9]{1,8}\Z")


class ArtifactError(ValueError):
    """An artifact stream could not be reassembled and verified."""


def canonical_json(value):
    """Stable UTF-8 JSON; NaN and Infinity are rejected."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_text(text):
    return sha256_bytes(text.encode("utf-8"))


def iter_sse_payloads(response):
    """Yield each SSE ``data:`` payload, including literal ``[DONE]``."""
    data_lines = []
    while True:
        line = response.readline()
        if line == b"" or line == "":
            if data_lines:
                yield "\n".join(data_lines)
            return
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        line = line.rstrip("\r\n")
        if line == "":
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines = []
                yield payload
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _sha256(value):
    return isinstance(value, str) and _HEX_256.fullmatch(value) is not None


def _artifact_identity(descriptor):
    varying = {"index", "total", "data", "encoding", "bytes", "sha256",
               "chunk_bytes", "chunk_sha256", "chunks_sha256", "ref"}
    return {key: value for key, value in descriptor.items()
            if key not in varying}


def _ingest_artifact(descriptor, groups):
    """Decode one chunk, retain bytes outside events, and sanitize metadata."""
    if not isinstance(descriptor, dict):
        raise ArtifactError("artifact descriptor must be an object")
    artifact_id = descriptor.get("artifact_id")
    index, total = descriptor.get("index"), descriptor.get("total")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise ArtifactError("artifact_id must be a non-empty string")
    if not _integer(index) or not _integer(total) or total <= 0 \
            or not 0 <= index < total:
        raise ArtifactError(f"invalid chunk index/total for {artifact_id}")
    if descriptor.get("encoding") != "base64" \
            or not isinstance(descriptor.get("data"), str):
        raise ArtifactError(f"{artifact_id} chunk requires base64 data")
    try:
        raw = base64.b64decode(descriptor["data"], validate=True)
    except (ValueError, TypeError) as exc:
        raise ArtifactError(f"invalid base64 for {artifact_id}: {exc}") from exc

    final = index + 1 == total
    if final:
        if not _integer(descriptor.get("bytes")) \
                or not _sha256(descriptor.get("sha256")):
            raise ArtifactError(f"{artifact_id} final chunk lacks bytes/SHA-256")
    elif "bytes" in descriptor or "sha256" in descriptor:
        raise ArtifactError(
            f"{artifact_id} non-final chunk carries final bytes/SHA-256")

    group = groups.setdefault(artifact_id, {
        "total": total, "chunks": {}, "descriptors": {},
        "identity": _artifact_identity(descriptor),
    })
    if group["total"] != total:
        raise ArtifactError(f"artifact total changed for {artifact_id}")
    if group["identity"] != _artifact_identity(descriptor):
        raise ArtifactError(f"artifact metadata changed for {artifact_id}")
    if index in group["chunks"]:
        raise ArtifactError(f"duplicate chunk {index} for {artifact_id}")

    clean = {key: value for key, value in descriptor.items()
             if key not in ("data", "encoding")}
    clean["chunk_bytes"] = len(raw)
    clean["chunk_sha256"] = sha256_bytes(raw)
    group["chunks"][index] = raw
    group["descriptors"][index] = clean
    return clean


def _artifact_suffix(descriptor):
    by_mime = {
        "image/png": ".png", "image/jpeg": ".jpg",
        "audio/wav": ".wav", "audio/pcm": ".pcm", "video/mp4": ".mp4",
    }
    suffix = by_mime.get(descriptor.get("mime_type"))
    if suffix is None:
        fmt = descriptor.get("format", "bin")
        suffix = "." + fmt.lower() if isinstance(fmt, str) else ".bin"
    return suffix if _SAFE_SUFFIX.fullmatch(suffix) else ".bin"


def _finalize_artifacts(groups, artifact_dir, terminal_state):
    """Verify complete groups, then publish only their reassembled bytes."""
    outputs = []
    for artifact_id, group in groups.items():
        total = group["total"]
        complete = sorted(group["chunks"]) == list(range(total))
        if not complete:
            if terminal_state == TERMINAL_COMPLETED:
                raise ArtifactError(f"missing artifact chunks for {artifact_id}")
            # Cancellation can legitimately interrupt a multi-chunk artifact.
            # Its metadata stays in events, but partial bytes are never stored.
            continue
        raw = b"".join(group["chunks"][index] for index in range(total))
        final = group["descriptors"][total - 1]
        if final["bytes"] != len(raw):
            raise ArtifactError(f"artifact byte mismatch for {artifact_id}")
        digest = sha256_bytes(raw)
        if final["sha256"] != digest:
            raise ArtifactError(f"artifact SHA-256 mismatch for {artifact_id}")
        ref = _store_artifact(
            artifact_dir, digest, _artifact_suffix(final), raw)
        chunk_hashes = [group["descriptors"][index]["chunk_sha256"]
                        for index in range(total)]
        chunks_sha256 = sha256_bytes(
            b"".join(bytes.fromhex(value) for value in chunk_hashes))
        output = {
            key: value for key, value in final.items()
            if key not in ("index", "total", "chunk_bytes", "chunk_sha256")
        }
        output.update({
            "ref": ref, "sha256": digest, "bytes": len(raw),
            "chunk_count": total, "chunks_sha256": chunks_sha256,
        })
        for descriptor in group["descriptors"].values():
            descriptor["ref"] = ref
        outputs.append(output)
    return outputs


def _normalize_unified(unified, seq, artifact_groups):
    event = {"seq": seq, "type": unified["kind"],
             "server_seq": unified.get("seq")}
    for key, value in unified.items():
        if key in ("kind", "seq"):
            continue
        if key == "artifact":
            value = _ingest_artifact(value, artifact_groups)
        event[key] = value
    event.setdefault("stage", None)
    event.setdefault("is_final", False)
    event.setdefault("text", "")
    return event


def parse_sse_stream(response, *, on_request_id=None, artifact_dir=None,
                     details=None, event_observer=None):
    """Parse SSE into the legacy five-tuple plus optional v2 ``details``.

    ``details`` receives ``artifacts`` and ``clean_text`` when supplied. The
    tuple shape remains compatible with phase-1 callers. Artifact base64 is
    consumed here and never returned in an event. ``event_observer`` receives
    normalized events out-of-band; observations never enter the export.
    """
    events = []
    output_parts = []
    clean_parts = []
    artifact_groups = {}
    terminal_state = TERMINAL_FAILED
    error = TRUNCATED_MESSAGE
    request_id = None
    unified_terminal = None

    def publish(event):
        events.append(event)
        if event_observer is not None:
            event_observer(event)

    def add(event_type, **fields):
        event = {"seq": len(events), "type": event_type}
        event.update(fields)
        publish(event)

    for payload in iter_sse_payloads(response):
        if payload == "[DONE]":
            if unified_terminal is not None:
                terminal_state = unified_terminal
                if unified_terminal == TERMINAL_CANCELLED:
                    error = None
            elif not any(event["type"] == "error" for event in events):
                terminal_state = TERMINAL_COMPLETED
                error = None
            break
        chunk = json.loads(payload)
        if chunk.get("id"):
            if request_id is None and on_request_id is not None:
                on_request_id(chunk["id"])
            request_id = chunk["id"]
        if "error" in chunk:
            unified_terminal = None
            terminal_state = TERMINAL_FAILED
            error = chunk["error"].get("message", "stream error")
            add("error", text=error, server_seq=None, stage=None,
                is_final=True)
            continue
        unified = chunk.get("unified")
        if unified is not None:
            kind = unified["kind"]
            text = unified.get("text", "")
            normalized = _normalize_unified(
                unified, len(events), artifact_groups)
            publish(normalized)
            if kind == "cancelled":
                unified_terminal = TERMINAL_CANCELLED
            elif kind == "error":
                unified_terminal = TERMINAL_FAILED
                error = text or unified.get("error") or error
            elif kind == "complete":
                unified_terminal = TERMINAL_COMPLETED
                error = None
            if kind in ("asr_final", "vlm_delta"):
                output_parts.append(text)
            if kind == "vlm_delta":
                clean_parts.append(text)
            continue
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                output_parts.append(content)
                clean_parts.append(content)
                add("output_delta", text=content, server_seq=None,
                    stage=None, is_final=False)
            if choice.get("finish_reason"):
                add("finish", finish_reason=choice["finish_reason"],
                    server_seq=None, stage=None, is_final=False)
    if unified_terminal is not None and terminal_state == TERMINAL_FAILED:
        terminal_state = unified_terminal
        if unified_terminal in (TERMINAL_COMPLETED, TERMINAL_CANCELLED):
            error = None
    output_text = "".join(output_parts)
    artifacts = _finalize_artifacts(
        artifact_groups, artifact_dir, terminal_state)
    if details is not None:
        details.update({"artifacts": artifacts,
                        "clean_text": "".join(clean_parts)})
    return request_id, events, output_text, terminal_state, error


def parse_chat_response(body, *, artifact_dir=None, details=None):
    """Parse non-streaming chat JSON, including optional mixed artifacts."""
    if "error" in body:
        message = body["error"].get("message", "request error")
        events = [{"seq": 0, "type": "error", "server_seq": None,
                   "stage": None, "is_final": True, "text": message,
                   "error": message}]
        return events, "", TERMINAL_FAILED, message
    if not body.get("choices"):
        message = "response carried neither choices nor an error"
        events = [{"seq": 0, "type": "error", "server_seq": None,
                   "stage": None, "is_final": True, "text": message,
                   "error": message}]
        return events, "", TERMINAL_FAILED, message
    try:
        choice = body["choices"][0]
        text = choice["message"]["content"]
    except (KeyError, TypeError, IndexError):
        message = "response choices[0] is missing message.content"
        events = [{"seq": 0, "type": "error", "server_seq": None,
                   "stage": None, "is_final": True, "text": message,
                   "error": message}]
        return events, "", TERMINAL_FAILED, message

    events = []
    groups = {}
    for raw in body.get("artifacts", []):
        descriptor = dict(raw)
        descriptor.setdefault("index", 0)
        descriptor.setdefault("total", 1)
        unified = {"kind": "artifact", "seq": None,
                   "stage": descriptor.get("media_type"),
                   "is_final": True, "text": "", "artifact": descriptor}
        events.append(_normalize_unified(unified, len(events), groups))
    if choice.get("finish_reason"):
        events.append({"seq": len(events), "type": "finish",
                       "server_seq": None, "stage": None, "is_final": False,
                       "finish_reason": choice["finish_reason"]})
    events.append({"seq": len(events), "type": "complete",
                   "server_seq": None, "stage": None, "is_final": True,
                   "text": text, "legs": body.get("legs", [])})
    artifacts = _finalize_artifacts(
        groups, artifact_dir, TERMINAL_COMPLETED)
    if details is not None:
        details.update({"artifacts": artifacts, "clean_text": text})
    return events, text, TERMINAL_COMPLETED, None


def _text_ref(text):
    return {"kind": "text", "ref": None, "media_type": None,
            "sha256": sha256_text(text), "text": text}


def _store_artifact(artifact_dir, digest, suffix, raw):
    if artifact_dir is None:
        return None
    if not _sha256(digest) or digest != sha256_bytes(raw):
        raise ArtifactError("refusing content-addressed storage with a bad hash")
    if not _SAFE_SUFFIX.fullmatch(suffix):
        suffix = ".bin"
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / (digest + suffix)
    if path.exists():
        if sha256_bytes(path.read_bytes()) != digest:
            raise ArtifactError(f"existing artifact failed hash verification: {path.name}")
        return path.name
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(raw)
    if sha256_bytes(tmp.read_bytes()) != digest:
        raise ArtifactError(f"new artifact failed hash verification: {path.name}")
    os.replace(tmp, path)
    return path.name


def _decode_data_url(url):
    header, encoded = url.split(",", 1)
    media_type = header[5:].split(";", 1)[0]
    if "base64" in header:
        raw = base64.b64decode(encoded, validate=True)
    else:
        raw = unquote_to_bytes(encoded)
    return media_type, raw


def _binary_ref(kind, media_type, raw, artifact_dir, *, source=None):
    digest = sha256_bytes(raw)
    suffix = _artifact_suffix({"mime_type": media_type})
    ref = {
        "kind": kind,
        "ref": _store_artifact(artifact_dir, digest, suffix, raw),
        "media_type": media_type,
        "sha256": digest,
        "bytes": len(raw),
    }
    if source is not None:
        ref["source"] = source
    return ref


def input_references(messages, artifact_dir):
    """Reference message parts in request order; never inline binary media."""
    references = []
    for message in messages:
        content = message["content"]
        parts = [content] if isinstance(content, str) else content
        for part in parts:
            if isinstance(part, str):
                references.append(_text_ref(part))
                continue
            part_type = part["type"]
            if part_type == "text":
                references.append(_text_ref(part["text"]))
            elif part_type == "input_audio":
                audio = part["input_audio"]
                raw = base64.b64decode(audio["data"], validate=True)
                media_type = "audio/" + audio["format"]
                references.append(_binary_ref(
                    "audio", media_type, raw, artifact_dir))
            elif part_type == "image_url":
                url = part["image_url"]["url"]
                if url.startswith("data:"):
                    media_type, raw = _decode_data_url(url)
                    references.append(_binary_ref(
                        "image", media_type, raw, artifact_dir))
                else:
                    references.append({"kind": "image", "ref": url,
                                       "media_type": None, "sha256": None})
            else:
                raise ValueError(f"unsupported content part type: {part_type}")
    return references


def request_parameter_references(body, artifact_dir):
    """Replace inline image/audio request options with CAS references."""
    params = copy.deepcopy({key: value for key, value in body.items()
                            if key != "messages"})
    extra_inputs = []
    for section, field, kind in (("image", "image", "image"),
                                 ("video", "image", "image"),
                                 ("audio", "ref_audio", "audio")):
        options = params.get(section)
        if not isinstance(options, dict) or not isinstance(
                options.get(field), str):
            continue
        encoded = options[field]
        if kind == "image":
            if not encoded.startswith("data:"):
                continue
            media_type, raw = _decode_data_url(encoded)
        elif encoded.startswith("data:"):
            media_type, raw = _decode_data_url(encoded)
        else:
            media_type = "audio/wav"
            raw = base64.b64decode(encoded, validate=True)
        source = f"request_params.{section}.{field}"
        ref = _binary_ref(kind, media_type, raw, artifact_dir, source=source)
        options[field] = dict(ref)
        extra_inputs.append(ref)
    return params, extra_inputs


def _fields(event, names):
    return {name: event[name] for name in names if name in event}


def _slot_records(events):
    names = ("seq", "server_seq", "stage", "slot_id", "slot_index",
             "media_type", "prompt", "text_offset")
    return [_fields(event, names) for event in events
            if event.get("type") == "visual_slot"]


def _leg_error_records(events):
    names = ("seq", "server_seq", "stage", "error", "slot_id", "slot_index")
    return [_fields(event, names) for event in events
            if event.get("type") == "leg_error"]


def _terminal_parts(events, terminal_state, error, output_text):
    terminals = [index for index, event in enumerate(events)
                 if event.get("type") in ("complete", "error", "cancelled")]
    if len(terminals) > 1:
        raise ValueError("trajectory contains multiple terminal events")
    if not terminals:
        kind = TERMINAL_EVENT_FOR_STATE[terminal_state]
        event = {"seq": len(events), "type": kind, "server_seq": None,
                 "stage": None, "is_final": True,
                 "text": output_text if kind == "complete" else (error or "")}
        if kind == "error":
            event["error"] = error or "request failed"
        events.append(event)
        terminals = [len(events) - 1]
    terminal_index = terminals[0]
    if terminal_index != len(events) - 1:
        raise ValueError("terminal event is not the final event")
    expected = TERMINAL_EVENT_FOR_STATE[terminal_state]
    if events[terminal_index].get("type") != expected:
        raise ValueError("terminal_state does not match terminal event")
    return terminal_index, events[terminal_index].get("legs", [])


def _derived_clean_text(events, output_text):
    parts = [event.get("text", "") for event in events
             if event.get("type") in ("vlm_delta", "output_delta")]
    if parts:
        return "".join(parts)
    if events and events[-1].get("type") == "complete" \
            and not any(event.get("type") == "asr_final" for event in events):
        return events[-1].get("text", "")
    return "" if any(event.get("type") == "vlm_delta" for event in events) \
        else (output_text if not any(
            event.get("type") in ("asr_final", "asr_partial")
            for event in events) else "")


class TrajectoryRecorder:
    """Append schema-v2 records to one JSONL file (one writer only)."""

    def __init__(self, jsonl_path, artifact_dir=None):
        self.jsonl_path = Path(jsonl_path)
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None \
            else self.jsonl_path.parent / "artifacts"
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._turns = {}

    def next_turn(self, session_id):
        index = self._turns.get(session_id, 0)
        self._turns[session_id] = index + 1
        return index

    def record(self, *, session_id, request_id, endpoint, inputs, params,
               events, terminal_state, error=None, output_text="",
               artifacts=None):
        if terminal_state not in TERMINAL_STATES:
            raise ValueError(f"invalid terminal state: {terminal_state}")
        events = list(events)
        terminal_seq, terminal_legs = _terminal_parts(
            events, terminal_state, error, output_text)
        clean_text = _derived_clean_text(events, output_text)
        record = {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "session_id": session_id,
            "request_id": request_id,
            "turn_index": self.next_turn(session_id),
            "endpoint": endpoint,
            "inputs": inputs,
            "request_params": params,
            "events": events,
            "terminal_state": terminal_state,
            "terminal_event_seq": terminal_seq,
            "terminal_legs": copy.deepcopy(terminal_legs),
            "error": error,
            "output_text": output_text,
            "output_sha256": sha256_text(output_text),
            "clean_text": clean_text,
            "clean_text_sha256": sha256_text(clean_text),
            "visual_slots": _slot_records(events),
            "leg_errors": _leg_error_records(events),
            "artifacts": list(artifacts or []),
        }
        with self.jsonl_path.open("ab") as handle:
            handle.write(canonical_json(record) + b"\n")
        return record


def _http_connection(base_url, timeout):
    return http.client.HTTPConnection(base_url, timeout=timeout)


def abort_request(base_url, request_id, *, headers=None, timeout=30.0):
    """Call public POST /abort; 200 is acceptance, not proof compute stopped."""
    headers = dict(headers or {})
    headers["Content-Type"] = "application/json"
    conn = _http_connection(base_url, timeout)
    try:
        conn.request("POST", "/abort",
                     body=json.dumps({"request_id": request_id}),
                     headers=headers)
        response = conn.getresponse()
        return response.status, json.loads(response.read().decode("utf-8"))
    finally:
        conn.close()


def record_chat_completion(base_url, body, recorder, session_id, *,
                           headers=None, timeout=30.0, on_request_id=None,
                           event_observer=None):
    """Drive and record one public chat request, including transport failure."""
    headers = dict(headers or {})
    headers["Content-Type"] = "application/json"
    conn = _http_connection(base_url, timeout)
    request_id = None
    inputs = []
    params = {}
    artifacts = []
    try:
        try:
            params, option_inputs = request_parameter_references(
                body, recorder.artifact_dir)
            inputs = input_references(
                body["messages"], recorder.artifact_dir) + option_inputs
            conn.request("POST", "/v1/chat/completions",
                         body=json.dumps(body), headers=headers)
            response = conn.getresponse()
            is_sse = response.getheader("Content-Type", "").startswith(
                "text/event-stream")
            details = {}
            if body.get("stream") and is_sse:
                request_id, events, output_text, state, error = parse_sse_stream(
                    response, on_request_id=on_request_id,
                    artifact_dir=recorder.artifact_dir, details=details,
                    event_observer=event_observer)
            else:
                payload = json.loads(response.read().decode("utf-8"))
                request_id = payload.get("id")
                events, output_text, state, error = parse_chat_response(
                    payload, artifact_dir=recorder.artifact_dir, details=details)
            artifacts = details.get("artifacts", [])
        except ArtifactError as exc:
            message = f"artifact error: {exc}"
            events = [{"seq": 0, "type": "error", "server_seq": None,
                       "stage": None, "is_final": True, "text": message,
                       "error": message}]
            output_text, state, error = "", TERMINAL_FAILED, message
            artifacts = []
        except (OSError, http.client.HTTPException, json.JSONDecodeError,
                UnicodeDecodeError, TypeError, KeyError, ValueError) as exc:
            message = f"transport error: {exc}"
            events = [{"seq": 0, "type": "error", "server_seq": None,
                       "stage": None, "is_final": True, "text": message,
                       "error": message}]
            output_text, state, error = "", TERMINAL_FAILED, message
            artifacts = []
    finally:
        conn.close()

    return recorder.record(
        session_id=session_id,
        request_id=request_id or _synthetic_request_id(events),
        endpoint="/v1/chat/completions", inputs=inputs, params=params,
        events=events, terminal_state=state, error=error,
        output_text=output_text, artifacts=artifacts)


def record_realtime(recorder, session_id, request_id, observed_events, *,
                    inputs=None, terminal_state=None, error=None):
    """Record observed realtime events without adding a WS dependency."""
    events = []
    output_parts = []
    derived_state = TERMINAL_CANCELLED
    derived_error = None
    for index, observed in enumerate(observed_events):
        event_type = observed["type"]
        event = {"seq": index, "type": event_type, "server_seq": None,
                 "direction": observed["direction"]}
        if "text" in observed:
            event["text"] = observed["text"]
        if observed.get("is_final") is not None:
            event["is_final"] = bool(observed["is_final"])
        events.append(event)
        if observed["direction"] == "outbound":
            if event_type == "transcription.completed":
                derived_state = TERMINAL_COMPLETED
                output_parts.append(observed.get("text", ""))
            elif event_type == "error":
                derived_state = TERMINAL_FAILED
                derived_error = observed.get("error", {}).get("message")
    state = terminal_state or derived_state
    output_text = "".join(output_parts)
    return recorder.record(
        session_id=session_id, request_id=request_id, endpoint="/v1/realtime",
        inputs=list(inputs or []), params={}, events=events,
        terminal_state=state, error=error or derived_error,
        output_text=output_text)


def _synthetic_request_id(events):
    return "rollout-" + sha256_bytes(canonical_json(events))[:16]
