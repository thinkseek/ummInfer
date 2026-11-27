#!/usr/bin/env python3
"""Independent, stdlib-only verifier for ummInfer Web Replay JSON archives.

Does not import the Web producer, Showcase runner, or SSE replay implementation.
"""
import argparse
import base64
import binascii
import hashlib
import io
import json
from pathlib import Path
import wave

SCHEMA = "umminfer-web-session-replay-v1"
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_RAW_BYTES = 96 * 1024 * 1024
TERMINALS = {"complete", "cancelled", "error"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def payloads(raw):
    text = raw.decode("utf-8", errors="strict")
    result = []
    for line in text.split("\n"):
        if line.endswith("\r"):
            line = line[:-1]
        if not line.startswith("data:"):
            continue
        payload = line[5:]
        if payload.startswith(" "):
            payload = payload[1:]
        result.append(payload)
    return result


def projected_event(unified):
    event = json.loads(canonical(unified))
    artifact = event.get("artifact")
    if isinstance(artifact, dict) and isinstance(artifact.get("data"), str):
        artifact["data_base64_bytes"] = len(artifact["data"])
        del artifact["data"]
    return event


def verify_artifact(artifact_id, chunks):
    ordered = sorted(chunks, key=lambda item: item["index"])
    require(len(ordered) == ordered[0]["total"], f"{artifact_id}: incomplete chunks")
    require([x["index"] for x in ordered] == list(range(len(ordered))), f"{artifact_id}: chunk gap")
    for chunk in ordered:
        require(chunk["total"] == len(ordered), f"{artifact_id}: total changed")
        require(chunk["encoding"] == "base64", f"{artifact_id}: encoding changed")
    try:
        raw = b"".join(base64.b64decode(x["data"], validate=True) for x in ordered)
    except (binascii.Error, KeyError) as exc:
        raise ValueError(f"{artifact_id}: invalid base64") from exc
    final = ordered[-1]
    for field in ("mime_type", "media_type", "slot_id", "slot_index", "stream_index",
                  "format", "sample_rate", "channels"):
        require(all(chunk.get(field) == final.get(field) for chunk in ordered),
                f"{artifact_id}: {field} changed between chunks")
    actual = digest(raw)
    require(final.get("sha256", "").lower() == actual, f"{artifact_id}: SHA-256 mismatch")
    require(final.get("bytes") == len(raw), f"{artifact_id}: byte count mismatch")
    mime = final.get("mime_type")
    if mime == "audio/pcm":
        channels = final.get("channels")
        require(final.get("format") == "pcm16", f"{artifact_id}: not PCM16")
        require(isinstance(channels, int) and not isinstance(channels, bool) and 1 <= channels <= 8,
                f"{artifact_id}: invalid channel count")
        require(isinstance(final.get("sample_rate"), int) and final["sample_rate"] > 0,
                f"{artifact_id}: invalid sample rate")
        require(len(raw) % (2 * channels) == 0 and raw, f"{artifact_id}: invalid PCM frame alignment")
    elif mime == "audio/wav":
        with wave.open(io.BytesIO(raw), "rb") as wav:
            require(wav.getsampwidth() == 2 and wav.getnchannels() >= 1 and wav.getframerate() > 0,
                    f"{artifact_id}: invalid WAV contract")
            require(len(wav.readframes(wav.getnframes())) > 0, f"{artifact_id}: empty WAV")
    elif mime == "image/png":
        require(raw.startswith(b"\x89PNG\r\n\x1a\n"), f"{artifact_id}: invalid PNG")
    elif mime == "video/mp4":
        require(len(raw) >= 12 and raw[4:8] == b"ftyp", f"{artifact_id}: invalid MP4")
    return {
        "artifact_id": artifact_id, "media_type": final.get("media_type"),
        "mime_type": mime, "bytes": len(raw), "sha256": actual,
        "slot_id": final.get("slot_id"),
        "slot_index": final.get("slot_index") if type(final.get("slot_index")) is int else None,
        "stream_index": final.get("stream_index") if type(final.get("stream_index")) is int else None,
        "sample_rate": final.get("sample_rate") if type(final.get("sample_rate")) is int else None,
        "channels": final.get("channels") if type(final.get("channels")) is int else None,
        "anchor_slot_index": final.get("anchor_slot_index") if type(final.get("anchor_slot_index")) is int else None,
        "anchor_sha256": final.get("anchor_sha256") if isinstance(final.get("anchor_sha256"), str) else None,
    }, raw


def verify_terminal_and_slots(turn, events, media, request_id, index):
    terminal_event = next((x for x in events if x.get("kind") in TERMINALS), None)
    expected_state = ({"complete": "completed", "cancelled": "cancelled", "error": "error"}[
        terminal_event["kind"]] if terminal_event else
        (turn.get("outcome") if turn.get("outcome") in ("cancelled", "error") else None))
    terminal = turn.get("terminal")
    require(isinstance(terminal, dict) and terminal.get("state") == expected_state,
            f"turn {index}: terminal kind/outcome mismatch")
    if request_id is not None:
        require(terminal.get("request_id") == request_id, f"turn {index}: terminal request mismatch")
    require(terminal.get("legs", []) == (terminal_event or {}).get("legs", []),
            f"turn {index}: terminal legs mismatch")
    outcome = turn.get("outcome")
    if terminal_event and terminal_event["kind"] == "complete":
        require(outcome in ("completed", "incomplete"), f"turn {index}: raw complete/outcome mismatch")
    if terminal_event and terminal_event["kind"] == "cancelled":
        require(outcome == "cancelled", f"turn {index}: raw cancelled/outcome mismatch")
    raw_slots = {}
    for event in events:
        if event.get("kind") == "visual_slot":
            slot_id = event.get("slot_id")
            require(isinstance(slot_id, str) and slot_id not in raw_slots,
                    f"turn {index}: duplicate/missing raw slot id")
            raw_slots[slot_id] = event
    exported = turn.get("visual_slots", [])
    by_id = {}
    for slot in exported:
        require(slot.get("id") not in by_id, f"turn {index}: duplicate exported slot")
        by_id[slot.get("id")] = slot
    require(set(raw_slots).issubset(by_id), f"turn {index}: omitted raw visual slot")
    media_by_slot = {x["slot_id"]: x for x in media if x["slot_id"]}
    media_by_id = {x["artifact_id"]: x for x in media}
    legs = {x.get("slot_id"): x for x in (terminal_event or {}).get("legs", [])
            if isinstance(x, dict) and x.get("slot_id")}

    def position(slot):
        value = slot.get("slot_index")
        if type(value) is int:
            return value
        identifier = slot.get("id", "")
        return int(identifier[5:]) if identifier.startswith("slot-") and identifier[5:].isdigit() else None

    for slot_id, slot in by_id.items():
        raw = raw_slots.get(slot_id)
        if raw is None:
            artifact = media_by_id.get(slot_id)
            require(artifact is not None and artifact["sha256"] == slot.get("sha256") and
                    artifact["bytes"] == slot.get("bytes"),
                    f"turn {index}: unbound non-slot visual")
            continue
        variants = slot.get("variants", [])
        raw_variant = next((x for x in variants if x.get("kind") == raw.get("media_type")),
                           slot if slot.get("kind") == raw.get("media_type") else None)
        direct_variant = slot.get("kind") == raw.get("media_type") or any(
            x.get("slot_id") == slot_id and x.get("kind") == slot.get("kind")
            for x in turn.get("direct_renders", []))
        require(raw_variant is not None and direct_variant,
                f"turn {index}: slot kind/variant mismatch")
        if slot.get("slot_index") is not None:
            require(slot["slot_index"] == raw.get("slot_index"), f"turn {index}: slot index mismatch")
        for field in ("continuity_mode", "anchor_slot_index", "anchor_explicit"):
            left, right = slot.get(field), raw.get(field)
            if field == "continuity_mode":
                left, right = left or None, right or None
            require(left == right and type(left) is type(right),
                    f"turn {index}: slot {field} mismatch")
        if slot_id in legs:
            require(slot.get("terminal_state") == legs[slot_id].get("state"),
                    f"turn {index}: terminal slot state mismatch")
        if slot.get("terminal_state") == "completed":
            artifact = media_by_slot.get(slot_id)
            require(artifact is not None and artifact["sha256"] == raw_variant.get("sha256") and
                    artifact["bytes"] == raw_variant.get("bytes"),
                    f"turn {index}: completed slot/artifact mismatch")
        if outcome == "cancelled":
            require(slot.get("terminal_state") not in ("pending", "running"),
                    f"turn {index}: unresolved slot after cancellation")
        anchor = slot.get("anchor_slot_index")
        if anchor is not None:
            source = next((x for x in exported if position(x) == anchor), None)
            source_sha = next((x.get("sha256") for x in source.get("variants", [])
                               if x.get("kind") == "image"), None) if source else None
            if source and source.get("kind") == "image" and not source_sha:
                source_sha = source.get("sha256")
            require(source is not None and
                    source.get("terminal_state") == "completed" and
                    position(source) is not None and position(source) < position(slot) and
                    isinstance(source_sha, str) and len(source_sha) == 64,
                    f"turn {index}: invalid visual anchor source")
            if slot.get("anchor_explicit") is True:
                require(slot.get("anchor_sha256") == source_sha,
                        f"turn {index}: explicit anchor digest mismatch")


def verify_turn(turn, session_id, index):
    replay = turn.get("replay")
    if replay is None:
        require(not turn.get("narrative_events") and not turn.get("media_artifacts") and
                not turn.get("direct_renders") and
                (turn.get("role") == "user" or not turn.get("response")),
                f"turn {index}: replay is missing from a response turn")
        return {"events": 0, "artifacts": 0, "pcm_bytes": 0, "wav_bytes": 0,
                "outcome": turn.get("outcome"), "raw_sse_sha256": None}
    require(isinstance(replay, dict), f"turn {index}: replay is invalid")
    require(replay.get("transport") == "raw-sse", f"turn {index}: transport mismatch")
    encoded = replay.get("raw_sse")
    require(isinstance(encoded, str), f"turn {index}: raw_sse missing")
    raw = base64.b64decode(encoded, validate=True)
    require(base64.b64encode(raw).decode("ascii") == encoded, f"turn {index}: noncanonical base64")
    require(len(raw) <= MAX_RAW_BYTES and len(raw) == replay.get("raw_sse_bytes"),
            f"turn {index}: raw size mismatch")
    require(digest(raw) == replay.get("raw_sse_sha256"), f"turn {index}: raw SHA-256 mismatch")
    lengths = replay.get("chunk_lengths")
    require(isinstance(lengths, list) and lengths and
            all(type(x) is int and x >= 0 for x in lengths) and sum(lengths) == len(raw),
            f"turn {index}: invalid chunk lengths")
    require(b"".join(raw[n:n + size] for n, size in zip(_offsets(lengths), lengths)) == raw,
            f"turn {index}: chunk reconstruction mismatch")
    events = []
    chunks = {}
    completed = set()
    request_id = None
    terminal = 0
    done = 0
    seq = 0
    for payload in payloads(raw):
        if payload == "[DONE]":
            done += 1
            continue
        root = json.loads(payload)
        unified = root.get("unified")
        if unified is None:
            continue
        require(terminal == 0, f"turn {index}: event after terminal")
        require(unified.get("session_id") == session_id, f"turn {index}: cross-session event")
        require(unified.get("seq") == seq, f"turn {index}: sequence mismatch")
        seq += 1
        observed_request_id = unified.get("request_id")
        if observed_request_id is not None:
            if request_id is None:
                request_id = observed_request_id
            require(observed_request_id == request_id, f"turn {index}: cross-request event")
        if unified.get("kind") in TERMINALS:
            terminal += 1
        events.append(projected_event(unified))
        artifact = unified.get("artifact")
        if unified.get("kind") != "artifact" or artifact is None:
            continue
        artifact_id = artifact.get("artifact_id")
        require(isinstance(artifact_id, str) and artifact_id, f"turn {index}: missing artifact id")
        require(artifact_id not in completed, f"turn {index}: duplicate completed artifact")
        group = chunks.setdefault(artifact_id, [])
        require(all(x["index"] != artifact.get("index") for x in group),
                f"turn {index}: duplicate artifact chunk")
        group.append(artifact)
        if artifact.get("index") + 1 == artifact.get("total"):
            completed.add(artifact_id)
    require(events == turn.get("narrative_events", []), f"turn {index}: event projection mismatch")
    require(done <= 1 and terminal <= 1, f"turn {index}: duplicate terminal/DONE")
    if replay.get("terminal_required") is True:
        require(done == 1 and terminal == 1, f"turn {index}: terminal/DONE missing")
    actual_media = []
    for artifact_id, group in chunks.items():
        record, _raw_bytes = verify_artifact(artifact_id, group)
        actual_media.append(record)
    expected_media = {x.get("artifact_id"): x for x in turn.get("media_artifacts", [])}
    require(len(expected_media) == len(turn.get("media_artifacts", [])) and
            len(actual_media) == len(expected_media) and
            all(canonical(x) == canonical(expected_media.get(x["artifact_id"])) for x in actual_media),
            f"turn {index}: media index mismatch")
    direct_renders = turn.get("direct_renders", [])
    require(isinstance(direct_renders, list), f"turn {index}: direct_renders must be an array")
    slots_by_id = {x.get("id"): x for x in turn.get("visual_slots", [])}
    for record in direct_renders:
        require(isinstance(record, dict) and record.get("kind") in ("image", "video"),
                f"turn {index}: invalid direct render")
        slot = slots_by_id.get(record.get("slot_id"))
        require(slot is not None, f"turn {index}: direct render slot missing")
        encoded = record.get("response_base64")
        require(isinstance(encoded, str), f"turn {index}: direct render bytes missing")
        raw_render = base64.b64decode(encoded, validate=True)
        require(base64.b64encode(raw_render).decode() == encoded and
                len(raw_render) == record.get("bytes") and len(raw_render) <= 48 * 1024 * 1024 and
                digest(raw_render) == record.get("sha256"),
                f"turn {index}: direct render byte/hash mismatch")
        variant = next((x for x in slot.get("variants", []) if x.get("kind") == record["kind"]), None)
        require(variant is not None and variant.get("bytes") == len(raw_render) and
                variant.get("sha256") == record.get("sha256"),
                f"turn {index}: direct render variant missing")
        raw_slot = next((x for x in events if x.get("kind") == "visual_slot" and
                         x.get("slot_id") == record["slot_id"]), None)
        if record["kind"] == "video" and raw_slot and raw_slot.get("media_type") == "image":
            image_sha = next((x.get("sha256") for x in slot.get("variants", [])
                              if x.get("kind") == "image"), None)
            require(record.get("source_sha256") == image_sha and image_sha is not None,
                    f"turn {index}: Animate source mismatch")
    outcome = turn.get("outcome")
    if outcome == "cancelled":
        require(replay.get("terminal_required") is False,
                f"turn {index}: invalid cancellation replay policy")
    verify_terminal_and_slots(turn, events, actual_media, request_id, index)
    return {"events": len(events), "artifacts": len(actual_media),
            "pcm_bytes": sum(x["bytes"] for x in actual_media if x["mime_type"] == "audio/pcm"),
            "wav_bytes": sum(x["bytes"] for x in actual_media if x["mime_type"] == "audio/wav"),
            "outcome": outcome, "raw_sse_sha256": digest(raw),
            "direct_renders": len(direct_renders)}


def _offsets(lengths):
    offset = 0
    for length in lengths:
        yield offset
        offset += length


def verify_manifest(path, root):
    path = Path(path).resolve()
    root = Path(root).resolve()
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        expected, separator, name = line[:64], line[64:66], line[66:]
        require(separator in ("  ", " *") and len(expected) == 64 and
                all(c in "0123456789abcdef" for c in expected), "malformed manifest line")
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts and name != path.name,
                "unsafe or self-included manifest entry")
        require(name not in entries, "duplicate manifest entry")
        target = (root / relative).resolve()
        require(target.is_relative_to(root) and target.is_file() and not target.is_symlink(),
                f"manifest target missing or unsafe: {name}")
        require(digest(target.read_bytes()) == expected, f"manifest digest mismatch: {name}")
        entries[name] = expected
    actual = {str(p.relative_to(root)) for p in root.rglob("*")
              if p.is_file() and p.resolve() != path}
    require(set(entries) == actual, "manifest does not cover exactly the external evidence files")
    return {"files": len(entries), "manifest_sha256": digest(path.read_bytes())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-root", type=Path)
    args = parser.parse_args()
    require(args.archive.is_file() and args.archive.stat().st_size <= MAX_FILE_BYTES,
            "archive missing or too large")
    require(not args.out.exists(), f"refusing to overwrite report: {args.out}")
    report = {"schema": "umminfer-storyboard-replay-verification/v1", "status": "PASS",
              "archive_sha256": digest(args.archive.read_bytes()), "turns": []}
    try:
        root = json.loads(args.archive.read_text(encoding="utf-8"))
        require(isinstance(root, dict) and root.get("schema") == SCHEMA, "unsupported replay schema")
        require(root.get("replay_format") == "umminfer-raw-sse-replay-v1", "replay format mismatch")
        require(isinstance(root.get("turns"), list), "turns missing")
        session_id = root.get("session_id")
        require(isinstance(session_id, str) and session_id, "session id missing")
        slots = []
        for index, turn in enumerate(root["turns"]):
            report["turns"].append(verify_turn(turn, session_id, index))
            reference = turn.get("visual_reference")
            if reference:
                candidates = [slot for slot in slots if slot.get("id") == reference.get("slot_id") and
                              (not reference.get("sha256") or slot.get("sha256") == reference["sha256"] or
                               any(x.get("sha256") == reference["sha256"] for x in slot.get("variants", [])))]
                require(candidates and reference.get("source_attached") is True and
                        reference.get("edit_source_attached") is True,
                        f"turn {index}: unresolved or unbound visual revision reference")
            slots.extend(turn.get("visual_slots", []))
        if args.manifest:
            require(args.manifest_root is not None, "--manifest-root is required with --manifest")
            report["manifest"] = verify_manifest(args.manifest, args.manifest_root)
        report["totals"] = {
            "turns": len(report["turns"]),
            "events": sum(x["events"] for x in report["turns"]),
            "artifacts": sum(x["artifacts"] for x in report["turns"]),
            "pcm_bytes": sum(x["pcm_bytes"] for x in report["turns"]),
            "wav_bytes": sum(x["wav_bytes"] for x in report["turns"]),
            "direct_renders": sum(x.get("direct_renders", 0) for x in report["turns"]),
        }
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = str(exc)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "out": str(args.out),
                      "error": report.get("error"), "totals": report.get("totals")}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
