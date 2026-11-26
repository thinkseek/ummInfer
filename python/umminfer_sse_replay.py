# SPDX-License-Identifier: Apache-2.0
"""Raw public-SSE capture, integrity validation, and deterministic replay support.

The trajectory JSONL stores only descriptors and hashes. Exact response bytes
(including base64 media frames) live in a content-addressed ``.sse`` sidecar.
No serving module is imported. Live timestamps come from an explicitly declared
monotonic clock; deterministic tests use logical ticks, never fake wall times.
"""

import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

# Match the other offline Python entry points under isolated ``python -I``.
sys.path.insert(0, str(Path(__file__).resolve().parent))

REPLAY_FORMAT_VERSION = 1
REPLAY_FORMAT = "umminfer-raw-sse-replay-v1"
RAW_SUFFIX = ".sse"
_TERMINAL_TYPES = frozenset(("complete", "error", "cancelled"))
_FRAME_KINDS = frozenset(("event", "done", "comment", "keepalive", "control"))
_CLOCK_KINDS = frozenset(("monotonic_ns", "logical_tick"))
_HEX = frozenset("0123456789abcdef")


class ReplayError(ValueError):
    """A raw stream, replay binding, or replayed record failed validation."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_sha(value):
    return isinstance(value, str) and len(value) == 64 \
        and bool(value) and not set(value) - _HEX


def _require(condition, message):
    if not condition:
        raise ReplayError(message)


def _tick(value, label):
    _require(_integer(value) and value >= 0, f"{label} must be a non-negative integer")


class PerfCounterClock:
    """Observed arrival ticks relative to capture start, in monotonic ns."""

    def __init__(self, *, origin_ns=None):
        self._origin = time.perf_counter_ns() if origin_ns is None else origin_ns
        _tick(self._origin, "monotonic clock origin")
        self._last = 0

    @property
    def contract(self):
        return {
            "kind": "monotonic_ns",
            "source": "python.time.perf_counter_ns",
            "unit": "ns",
            "origin": "capture_start",
            "observation": "observed_arrival",
        }

    def tick(self):
        value = time.perf_counter_ns() - self._origin
        if value < self._last:  # perf_counter_ns is monotonic; fail closed if not.
            raise ReplayError("monotonic clock moved backwards")
        self._last = value
        return value


class LogicalTickClock:
    """Caller/test-supplied logical ticks; explicitly not a time measurement."""

    def __init__(self, ticks=None, *, start=0, step=1):
        _tick(start, "logical clock start")
        _tick(step, "logical clock step")
        self._ticks = None if ticks is None else iter(ticks)
        self._next = start
        self._step = step
        self._last = None

    @property
    def contract(self):
        return {
            "kind": "logical_tick",
            "source": "caller_supplied_logical_ticks",
            "unit": "tick",
            "origin": "capture_start",
            "observation": "deterministic_fixture_not_measured_time",
        }

    def tick(self):
        value = next(self._ticks) if self._ticks is not None else self._next
        _tick(value, "logical clock tick")
        if self._last is not None and value < self._last:
            raise ReplayError("logical clock moved backwards")
        self._last = value
        if self._ticks is None:
            self._next = value + self._step
        return value


def _frame_fields(raw, index):
    """Return ``(kind, payload-or-None)`` for one blank-line-terminated frame."""
    lines = []
    cursor = 0
    while cursor < len(raw):
        end = cursor
        while end < len(raw) and raw[end] not in (10, 13):
            end += 1
        lines.append(raw[cursor:end])
        if end < len(raw) and raw[end] == 13 and end + 1 < len(raw) \
                and raw[end + 1] == 10:
            cursor = end + 2
        else:
            cursor = min(end + 1, len(raw))
    data = []
    has_field = False
    for line in lines:
        if not line or line.startswith(b":"):
            continue
        has_field = True
        name, separator, value = line.partition(b":")
        if name != b"data":
            continue
        if value.startswith(b" "):
            value = value[1:]
        data.append(value)
        if not separator:
            data[-1] = b""
    if not data:
        if not has_field:
            kind = "keepalive" if not any(lines) else "comment"
        else:
            kind = "control"
        return kind, None
    try:
        payload = b"\n".join(data).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReplayError(f"SSE frame {index} data is not valid UTF-8: {exc}") from exc
    return ("done" if payload == "[DONE]" else "event"), payload


class RawSSECapture:
    """Incrementally capture exact chunks and blank-line-delimited SSE frames."""

    def __init__(self, *, clock=None, metadata=None, raw=None):
        self.clock = clock
        self._metadata = copy.deepcopy(metadata) if metadata is not None else None
        self._raw = bytes(raw) if raw is not None else None
        self._chunks = []
        self._frames = []
        self._payload_frames = []
        self._pending = bytearray()
        self._pending_offset = 0
        self._raw_size = 0
        self._last_tick = None
        self._done_index = None
        self._finished = False

    @classmethod
    def from_metadata(cls, metadata, raw):
        capture = cls(metadata=metadata, raw=raw)
        for frame in metadata["frames"]:
            if frame["kind"] in ("event", "done"):
                _, payload = _frame_fields(
                    raw[frame["offset"]:frame["offset"] + frame["bytes"]],
                    frame["index"])
                capture._payload_frames.append((frame["index"], payload))
        capture._done_index = metadata["done_frame"]
        capture._finished = True
        return capture

    @property
    def payload_frames(self):
        return list(self._payload_frames)

    @property
    def raw_bytes(self):
        if self._raw is not None:
            return self._raw
        return b"".join(chunk["raw"] for chunk in self._chunks)

    def feed(self, data, *, tick=None):
        _require(not self._finished, "cannot feed a finished SSE capture")
        _require(isinstance(data, (bytes, bytearray)), "SSE chunk must be bytes")
        _require(bool(data), "empty reads belong at finish(), not feed()")
        if tick is None:
            _require(self.clock is not None, "a tick or capture clock is required")
            tick = self.clock.tick()
        _tick(tick, "chunk tick")
        if self._last_tick is not None:
            _require(tick >= self._last_tick, "SSE chunk ticks are not monotonic")
        self._last_tick = tick
        raw = bytes(data)
        offset = self._raw_size
        self._chunks.append({
            "index": len(self._chunks), "offset": offset, "bytes": len(raw),
            "sha256": _sha(raw), "tick": tick, "raw": raw,
        })
        self._raw_size += len(raw)
        self._pending.extend(raw)
        self._drain(tick)

    def _drain(self, tick):
        cursor = 0
        line_start = 0
        while cursor < len(self._pending):
            char = self._pending[cursor]
            if char not in (10, 13):
                cursor += 1
                continue
            if char == 13 and cursor + 1 == len(self._pending):
                return  # A CR/LF delimiter may be split across transport chunks.
            terminator = 2 if char == 13 and self._pending[cursor + 1] == 10 else 1
            if cursor == line_start:
                end = cursor + terminator
                raw = bytes(self._pending[:end])
                self._append_frame(raw, tick)
                del self._pending[:end]
                self._pending_offset += end
                cursor = 0
                line_start = 0
                continue
            cursor += terminator
            line_start = cursor

    def _append_frame(self, raw, tick):
        index = len(self._frames)
        _require(self._done_index is None,
                 f"SSE frame {index} follows terminal [DONE]")
        kind, payload = _frame_fields(raw, index)
        descriptor = {
            "index": index, "offset": self._pending_offset, "bytes": len(raw),
            "sha256": _sha(raw), "tick": tick, "kind": kind,
        }
        if payload is not None:
            descriptor["data_sha256"] = _sha(payload.encode("utf-8"))
            self._payload_frames.append((index, payload))
        self._frames.append(descriptor)
        if kind == "done":
            self._done_index = index

    def finish(self):
        if self._finished:
            return
        _require(not self._pending,
                 f"truncated SSE frame at byte {self._pending_offset}")
        _require(self._done_index is not None, "stream ended before final [DONE]")
        _require(self._done_index == len(self._frames) - 1,
                 "final [DONE] is not the last SSE frame")
        _require(bool(self._frames), "empty SSE stream")
        self._raw = self.raw_bytes
        self._finished = True

    def public_metadata(self):
        _require(self._finished, "SSE capture is not finished")
        return {
            "format": REPLAY_FORMAT,
            "format_version": REPLAY_FORMAT_VERSION,
            "clock": copy.deepcopy(self._metadata["clock"] if self._metadata
                                   else self.clock.contract),
            "raw_ref": None,
            "raw_sha256": _sha(self.raw_bytes),
            "raw_bytes": len(self.raw_bytes),
            "chunks": [{key: value for key, value in chunk.items() if key != "raw"}
                       for chunk in self._chunks] if self._metadata is None
                      else copy.deepcopy(self._metadata["chunks"]),
            "frames": copy.deepcopy(self._frames if self._metadata is None
                                    else self._metadata["frames"]),
            "done_frame": self._done_index,
        }


def _validate_clock(clock):
    _require(isinstance(clock, dict), "sse_replay.clock must be an object")
    _require(clock.get("kind") in _CLOCK_KINDS, "unsupported replay clock kind")
    expected = {
        "monotonic_ns": ("python.time.perf_counter_ns", "ns", "observed_arrival"),
        "logical_tick": ("caller_supplied_logical_ticks", "tick",
                         "deterministic_fixture_not_measured_time"),
    }[clock["kind"]]
    _require(clock.get("source") == expected[0], "replay clock source mismatch")
    _require(clock.get("unit") == expected[1], "replay clock unit mismatch")
    _require(clock.get("origin") == "capture_start", "replay clock origin mismatch")
    _require(clock.get("observation") == expected[2],
             "replay clock observation/provenance mismatch")


def _descriptor_sequence(items, label, raw_size):
    previous_tick = None
    previous_end = 0
    for index, item in enumerate(items):
        _require(isinstance(item, dict), f"{label} {index} must be an object")
        _require(item.get("index") == index, f"{label} index gap at {index}")
        _require(item.get("offset") == previous_end,
                 f"{label} offset gap at {index}")
        _require(_integer(item.get("bytes")) and item["bytes"] > 0,
                 f"{label} byte count is invalid at {index}")
        _require(_valid_sha(item.get("sha256")), f"{label} hash is invalid at {index}")
        _tick(item.get("tick"), f"{label} tick at {index}")
        if previous_tick is not None:
            _require(item["tick"] >= previous_tick,
                     f"{label} ticks are not monotonic at {index}")
        previous_tick = item["tick"]
        previous_end += item["bytes"]
    _require(previous_end == raw_size, f"{label} do not cover the raw SSE bytes")


def _validate_raw_normalization(record, raw):
    """Re-normalize without writes and compare every source-bound event."""
    from umminfer_rl_trajectory import _terminal_parts, parse_sse_stream

    metadata = record["sse_replay"]
    capture = RawSSECapture.from_metadata(metadata, raw)
    try:
        _request_id, events, output_text, state, error = parse_sse_stream(
            io.BytesIO(raw), artifact_dir=None, details={}, capture=capture)
        _terminal_parts(events, state, error, output_text)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ReplayError(f"raw SSE normalization failed: {exc}") from exc
    done = metadata["frames"][metadata["done_frame"]]
    for event in events:
        if "source_event_index" not in event:
            event["source_event_index"] = done["index"]
            event["source_event_sha256"] = done["sha256"]
            event["source_event_tick"] = done["tick"]

    def projection(items):
        result = copy.deepcopy(items)
        for event in result:
            descriptor = event.get("artifact")
            if isinstance(descriptor, dict):
                descriptor.pop("ref", None)  # None in write-free normalization.
        return result

    _require(projection(events) == projection(record["events"]),
             "raw SSE re-normalization differs from source-bound trajectory events")


def validate_replay_metadata(record, *, artifact_dir=None, require_raw=False):
    """Validate replay structure/binding and optionally re-hash the raw sidecar.

    Returns the exact raw bytes when a sidecar was read, otherwise ``None``.
    With no ``sse_replay`` block this is a no-op for old v1/v2 compatibility.
    """
    metadata = record.get("sse_replay")
    if metadata is None:
        _require(not require_raw, "record has no raw SSE replay sidecar")
        return None
    _require(isinstance(metadata, dict), "sse_replay must be an object")
    _require(metadata.get("format") == REPLAY_FORMAT
             and metadata.get("format_version") == REPLAY_FORMAT_VERSION,
             "unsupported raw SSE replay format/version")
    _validate_clock(metadata.get("clock"))
    raw_size = metadata.get("raw_bytes")
    _require(_integer(raw_size) and raw_size > 0, "invalid raw SSE byte count")
    _require(_valid_sha(metadata.get("raw_sha256")), "invalid raw SSE SHA-256")
    ref = metadata.get("raw_ref")
    _require(isinstance(ref, str) and Path(ref).name == ref
             and ref == metadata["raw_sha256"] + RAW_SUFFIX,
             "raw SSE ref is not content-addressed")
    chunks = metadata.get("chunks")
    frames = metadata.get("frames")
    _require(isinstance(chunks, list) and bool(chunks), "missing raw SSE chunks")
    _require(isinstance(frames, list) and bool(frames), "missing raw SSE frames")
    _descriptor_sequence(chunks, "SSE chunk", raw_size)
    _descriptor_sequence(frames, "SSE frame", raw_size)
    for frame in frames:
        _require(frame.get("kind") in _FRAME_KINDS,
                 f"invalid SSE frame kind at {frame['index']}")
        if frame["kind"] in ("event", "done"):
            _require(_valid_sha(frame.get("data_sha256")),
                     f"missing data hash at SSE frame {frame['index']}")
        else:
            _require("data_sha256" not in frame,
                     f"non-data SSE frame {frame['index']} has a data hash")
    done = metadata.get("done_frame")
    _require(done == len(frames) - 1 and frames[-1]["kind"] == "done",
             "final [DONE] frame is missing or not last")

    events = record.get("events")
    _require(isinstance(events, list), "replay record lacks normalized events")
    _require(metadata.get("event_count") == len(events),
             "replay event count differs from normalized events")
    _require(metadata.get("normalized_events_sha256") == _sha(_canonical(events)),
             "normalized event/source hash chain mismatch")
    _require(metadata.get("terminal_state") == record.get("terminal_state"),
             "replay terminal state differs from trajectory")
    _require(metadata.get("terminal_event_seq") == record.get("terminal_event_seq"),
             "replay terminal sequence differs from trajectory")
    previous_frame = -1
    previous_tick = 0
    for index, event in enumerate(events):
        _require(isinstance(event, dict) and event.get("seq") == index,
                 f"normalized event sequence gap at {index}")
        source = event.get("source_event_index")
        _require(_integer(source) and 0 <= source < len(frames),
                 f"event {index} has an invalid source frame")
        _require(source >= previous_frame,
                 f"event {index} source frames are out of order")
        frame = frames[source]
        _require(frame["kind"] in ("event", "done"),
                 f"event {index} binds a non-data/comment frame")
        _require(frame["kind"] == "event" or index == metadata["terminal_event_seq"],
                 f"event {index} improperly binds [DONE]")
        _require(event.get("source_event_sha256") == frame["sha256"],
                 f"event {index} source frame hash mismatch")
        _require(event.get("source_event_tick") == frame["tick"],
                 f"event {index} source tick mismatch")
        _require(event["source_event_tick"] >= previous_tick,
                 f"event {index} timing is not monotonic")
        previous_frame = source
        previous_tick = event["source_event_tick"]
    server_seqs = [event.get("server_seq") for event in events
                   if event.get("server_seq") is not None]
    _require(all(_integer(seq) for seq in server_seqs)
             and server_seqs == list(range(len(server_seqs))),
             "replay server event sequence is duplicated, out-of-order, or has a gap")
    terminals = [event for event in events if event.get("type") in _TERMINAL_TYPES]
    _require(len(terminals) == 1 and events and events[-1] is terminals[0],
             "replay requires exactly one final terminal event")

    path = Path(artifact_dir) / ref if artifact_dir is not None else None
    if path is None or not path.is_file():
        _require(not require_raw, f"raw SSE replay artifact is missing: {ref}")
        return None
    raw = path.read_bytes()
    _require(len(raw) == raw_size, "raw SSE replay byte count mismatch")
    _require(_sha(raw) == metadata["raw_sha256"], "raw SSE replay SHA-256 mismatch")
    for label, descriptors in (("chunk", chunks), ("frame", frames)):
        for item in descriptors:
            part = raw[item["offset"]:item["offset"] + item["bytes"]]
            _require(_sha(part) == item["sha256"],
                     f"raw SSE {label} {item['index']} SHA-256 mismatch")
    for frame in frames:
        raw_frame = raw[frame["offset"]:frame["offset"] + frame["bytes"]]
        kind, payload = _frame_fields(raw_frame, frame["index"])
        _require(kind == frame["kind"], f"SSE frame {frame['index']} kind mismatch")
        if payload is not None:
            _require(_sha(payload.encode("utf-8")) == frame["data_sha256"],
                     f"SSE frame {frame['index']} data hash mismatch")
    _validate_raw_normalization(record, raw)
    return raw


def replay_record(record, *, recorder, artifact_dir=None):
    """Replay one v2 record into ``recorder`` and require canonical identity."""
    from umminfer_rl_trajectory import (  # Avoid module cycles.
        _synthetic_request_id, parse_sse_stream,
    )

    root = Path(artifact_dir) if artifact_dir is not None else recorder.artifact_dir
    raw = validate_replay_metadata(record, artifact_dir=root, require_raw=True)
    metadata = record["sse_replay"]
    capture = RawSSECapture.from_metadata(metadata, raw)
    destination = recorder.artifact_dir
    destination.mkdir(parents=True, exist_ok=True)
    for item in list(record.get("inputs", [])) + list(record.get("artifacts", [])):
        if not isinstance(item, dict) or item.get("sha256") is None \
                or item.get("kind") == "text":
            continue
        ref = item.get("ref")
        _require(isinstance(ref, str) and Path(ref).name == ref
                 and ref.startswith(item["sha256"] + "."),
                 "invalid CAS reference during replay")
        source_path = root / ref
        _require(source_path.is_file(), f"replay input/output CAS is missing: {ref}")
        payload = source_path.read_bytes()
        _require(_sha(payload) == item["sha256"],
                 f"replay input/output CAS hash mismatch: {ref}")
        if item.get("bytes") is not None:
            _require(len(payload) == item["bytes"],
                     f"replay input/output CAS byte count mismatch: {ref}")
        target_path = destination / ref
        if target_path.exists():
            _require(target_path.read_bytes() == payload,
                     f"destination CAS conflict: {ref}")
        else:
            temporary = target_path.with_name(target_path.name + ".tmp")
            temporary.write_bytes(payload)
            os.replace(temporary, target_path)
    details = {}
    try:
        request_id, events, output_text, state, error = parse_sse_stream(
            io.BytesIO(raw), artifact_dir=destination, details=details, capture=capture)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ReplayError(f"cannot normalize raw SSE replay: {exc}") from exc
    base_keys = {
        "schema_version", "session_id", "request_id", "turn_index", "endpoint",
        "inputs", "request_params", "events", "terminal_state",
        "terminal_event_seq", "terminal_legs", "error", "output_text",
        "output_sha256", "clean_text", "clean_text_sha256", "visual_slots",
        "leg_errors", "artifacts", "sse_replay",
    }
    extras = {key: copy.deepcopy(value) for key, value in record.items()
              if key not in base_keys}
    replayed = recorder.record(
        session_id=record["session_id"],
        request_id=request_id or _synthetic_request_id(events),
        turn_index=record["turn_index"],
        endpoint=record["endpoint"], inputs=copy.deepcopy(record["inputs"]),
        params=copy.deepcopy(record["request_params"]), events=events,
        terminal_state=state, error=error, output_text=output_text,
        artifacts=details.get("artifacts", []), capture=capture,
        extra_fields=extras)
    if _canonical(replayed) != _canonical(record):
        differing = sorted(key for key in set(record) | set(replayed)
                           if record.get(key) != replayed.get(key))
        raise ReplayError(
            "replayed trajectory is not byte-identical; differing fields: "
            + ", ".join(differing))
    return replayed


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify and byte-replay raw-SSE trajectory-v2 JSONL")
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--out", type=Path, required=True,
                        help="new output JSONL; CAS/replay files go beside it")
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        source = args.trajectory.resolve()
        target = args.out.resolve()
        if source == target:
            raise ReplayError("--out must differ from the input trajectory")
        if target.exists():
            raise ReplayError(f"output already exists: {target}")
        records = [json.loads(line) for line in
                   source.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records:
            raise ReplayError("trajectory contains no records")
        root = args.artifact_dir or source.parent / "artifacts"
        from umminfer_rl_trajectory import TrajectoryRecorder
        recorder = TrajectoryRecorder(target, artifact_dir=target.parent / "artifacts")
        for record in records:
            replay_record(record, recorder=recorder, artifact_dir=root)
        expected = b"".join(_canonical(record) + b"\n" for record in records)
        if target.read_bytes() != expected:
            raise ReplayError("output JSONL differs from canonical input JSONL")
    except (OSError, json.JSONDecodeError, ReplayError, KeyError, ValueError) as exc:
        print(f"replay error: {exc}", file=sys.stderr)
        return 2
    print(f"RAW_SSE_REPLAY: PASS records={len(records)} output={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
