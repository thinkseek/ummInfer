#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent paired Task-84 NS-08 first-turn-history evidence verifier.

Only the standard library and msgpack are used.  This module deliberately does
not import the browser/stack drivers, serving code, or a producer scorecard.
"""
import argparse
import base64
import binascii
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

try:
    import msgpack
except ImportError:  # Reported as a verification failure, never silently skipped.
    msgpack = None

SCHEMA = "umminfer-ns08-first-turn-history-verification/v1"
REPLAY_SCHEMA = "umminfer-web-session-replay-v1"
REPLAY_FORMAT = "umminfer-raw-sse-replay-v1"
NATIVE_REPLAY_SCHEMA = "umminfer-vlm-history-prompt-replay-v1"
BROWSER_NAMES = ["browser-01", "browser-02", "browser-03", "browser-04"]
DIRECT_NAMES = ["explicit-stateless", "cross-session"]
REQUEST_NAMES = BROWSER_NAMES + DIRECT_NAMES
SETUP, FOLLOWUP, CANCELLED, POST_CANCEL = BROWSER_NAMES
TERMINALS = {"complete", "cancelled", "error"}
STATELESS_MARKER = "STATELESS_QUASAR_MARKER_84"


class VerificationError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON {path}: {exc}") from exc


def read_jsonl(path):
    rows = []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line:
                rows.append(json.loads(line))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSONL {path}: {exc}") from exc
    return rows


def is_int(value):
    return type(value) is int


def verify_manifest(root):
    root = root.resolve()
    manifest = root / "SHA256SUMS"
    require(manifest.is_file() and not manifest.is_symlink(),
            f"{root}: missing SHA256SUMS")
    entries = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        require(len(line) >= 67 and line[64:66] in ("  ", " *"),
                f"{root}: malformed manifest line")
        expected, name = line[:64], line[66:]
        require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
                f"{root}: malformed manifest digest")
        relative = Path(name)
        require(name and name != "SHA256SUMS" and not relative.is_absolute() and
                ".." not in relative.parts and name not in entries,
                f"{root}: unsafe, self-included, or duplicate manifest entry {name!r}")
        entries[name] = expected
    actual = set()
    for path in root.rglob("*"):
        require(not path.is_symlink(), f"{root}: symlink in evidence: {path}")
        if path.is_file() and path != manifest:
            actual.add(path.relative_to(root).as_posix())
    require(set(entries) == actual,
            f"{root}: manifest does not cover exactly the external evidence files; "
            f"missing={sorted(actual - set(entries))}, stale={sorted(set(entries) - actual)}")
    for name, expected in entries.items():
        target = root / name
        require(target.is_file() and sha(target.read_bytes()) == expected,
                f"{root}: manifest digest mismatch: {name}")
    return {"files": len(entries), "sha256": sha(manifest.read_bytes())}


def normalize_content(content, context):
    if isinstance(content, str):
        return content
    require(isinstance(content, list) and len(content) == 1 and
            isinstance(content[0], dict) and content[0].get("type") == "text" and
            isinstance(content[0].get("text"), str),
            f"{context}: expected one text content part")
    return content[0]["text"]


def normalize_messages(messages, context):
    require(isinstance(messages, list) and messages, f"{context}: messages missing")
    normalized = []
    for index, message in enumerate(messages):
        require(isinstance(message, dict) and isinstance(message.get("role"), str),
                f"{context}: invalid message {index}")
        normalized.append({"role": message["role"],
                           "content": normalize_content(message.get("content"),
                                                        f"{context}[{index}]")})
    return normalized


def without_leading_system(messages):
    index = 0
    while index < len(messages) and messages[index]["role"] == "system":
        index += 1
    return messages[index:]


def public_current_user(request, name):
    messages = request.get("messages")
    require(isinstance(messages, list) and len(messages) == 1 and
            isinstance(messages[0], dict) and messages[0].get("role") == "user",
            f"{name}: public request must contain exactly its current user message")
    return {"role": "user",
            "content": normalize_content(messages[0].get("content"), name)}


def decode_base64(encoded, context):
    require(isinstance(encoded, str), f"{context}: base64 string missing")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise VerificationError(f"{context}: invalid base64") from exc
    require(base64.b64encode(raw).decode("ascii") == encoded,
            f"{context}: noncanonical base64")
    return raw


def parse_sse(raw, name, session_id, expected_outcome):
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise VerificationError(f"{name}: non-UTF-8 public response") from exc
    public_ids = set()
    events = []
    done = 0
    for line in text.split("\n"):
        if line.endswith("\r"):
            line = line[:-1]
        if not line.startswith("data:"):
            require(not line.strip(), f"{name}: unexpected non-SSE line")
            continue
        payload = line[5:]
        if payload.startswith(" "):
            payload = payload[1:]
        if payload == "[DONE]":
            done += 1
            continue
        try:
            root = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise VerificationError(f"{name}: invalid SSE JSON") from exc
        require(isinstance(root, dict) and isinstance(root.get("id"), str) and root["id"],
                f"{name}: public request id missing")
        public_ids.add(root["id"])
        unified = root.get("unified")
        if unified is None:
            continue
        require(isinstance(unified, dict), f"{name}: invalid unified event")
        require(unified.get("session_id") == session_id,
                f"{name}: cross-session public event")
        inner_id = unified.get("request_id")
        require(inner_id is None or inner_id == root["id"],
                f"{name}: public request ownership mismatch")
        require(unified.get("seq") == len(events), f"{name}: public sequence gap")
        require(not events or events[-1].get("kind") not in TERMINALS,
                f"{name}: event after terminal")
        events.append(unified)
    if not raw and expected_outcome == "cancelled":
        return {"public_request_id": None, "events": [], "terminal_kind": None,
                "text": "", "done": 0, "transport_only_cancelled": True}
    require(len(public_ids) == 1, f"{name}: missing or ambiguous public request id")
    terminals = [event for event in events if event.get("kind") in TERMINALS]
    terminal_kind = terminals[0]["kind"] if terminals else None
    delta_text = "".join(event.get("text", "") for event in events
                         if event.get("kind") == "vlm_delta")
    if expected_outcome == "completed":
        require(done == 1 and len(terminals) == 1 and terminal_kind == "complete",
                f"{name}: completed response lacks exactly one complete/[DONE]")
        complete_text = terminals[0].get("text")
        require(isinstance(complete_text, str) and delta_text == complete_text,
                f"{name}: terminal prose differs from raw deltas")
        response_text = complete_text
    else:
        require(expected_outcome == "cancelled" and done <= 1 and len(terminals) <= 1 and
                terminal_kind in (None, "cancelled"),
                f"{name}: invalid cancelled raw response")
        response_text = delta_text
    return {"public_request_id": next(iter(public_ids)), "events": events,
            "terminal_kind": terminal_kind, "text": response_text, "done": done}


def projected_event(event):
    projected = json.loads(canonical(event))
    artifact = projected.get("artifact")
    if isinstance(artifact, dict) and isinstance(artifact.get("data"), str):
        artifact["data_base64_bytes"] = len(artifact["data"])
        del artifact["data"]
    return projected


def reassemble_artifacts(events, name):
    groups = {}
    completed = set()
    order = []
    for event in events:
        if event.get("kind") != "artifact":
            continue
        artifact = event.get("artifact")
        require(isinstance(artifact, dict) and isinstance(artifact.get("artifact_id"), str) and
                artifact["artifact_id"], f"{name}: artifact id missing")
        artifact_id = artifact["artifact_id"]
        require(artifact_id not in completed, f"{name}: chunk after completed artifact")
        group = groups.setdefault(artifact_id, [])
        if not group:
            order.append(artifact_id)
        index, total = artifact.get("index"), artifact.get("total")
        require(is_int(index) and is_int(total) and total >= 1 and
                index == len(group) and 0 <= index < total,
                f"{name}: duplicate, reordered, or gapped artifact chunk")
        group.append(artifact)
        if index + 1 == total:
            completed.add(artifact_id)
    records = {}
    for artifact_id in order:
        chunks = groups[artifact_id]
        require(len(chunks) == chunks[-1].get("total"),
                f"{name}: incomplete artifact {artifact_id}")
        parts = [decode_base64(chunk.get("data"), f"{name}:{artifact_id}") for chunk in chunks]
        raw = b"".join(parts)
        final = chunks[-1]
        for field in ("mime_type", "media_type", "slot_id", "slot_index", "stream_index",
                      "format", "sample_rate", "channels", "anchor_slot_index", "anchor_sha256"):
            require(all(chunk.get(field) == final.get(field) for chunk in chunks),
                    f"{name}: {artifact_id} changed {field}")
        require(final.get("bytes") == len(raw) and final.get("sha256", "").lower() == sha(raw),
                f"{name}: artifact byte/hash mismatch {artifact_id}")
        mime = final.get("mime_type")
        if mime == "audio/pcm":
            channels = final.get("channels")
            require(final.get("format") == "pcm16" and is_int(channels) and 1 <= channels <= 8 and
                    is_int(final.get("sample_rate")) and final["sample_rate"] > 0 and raw and
                    len(raw) % (2 * channels) == 0,
                    f"{name}: invalid PCM contract {artifact_id}")
        records[artifact_id] = {
            "artifact_id": artifact_id, "media_type": final.get("media_type"),
            "mime_type": mime, "bytes": len(raw), "sha256": sha(raw),
            "slot_id": final.get("slot_id"),
            "slot_index": final.get("slot_index") if is_int(final.get("slot_index")) else None,
            "stream_index": final.get("stream_index") if is_int(final.get("stream_index")) else None,
            "sample_rate": final.get("sample_rate") if is_int(final.get("sample_rate")) else None,
            "channels": final.get("channels") if is_int(final.get("channels")) else None,
            "anchor_slot_index": final.get("anchor_slot_index")
                if is_int(final.get("anchor_slot_index")) else None,
            "anchor_sha256": final.get("anchor_sha256")
                if isinstance(final.get("anchor_sha256"), str) else None,
        }
    return [records[artifact_id] for artifact_id in order]


def verify_archive_turn(turn, response, name, expected_outcome):
    require(isinstance(turn, dict) and turn.get("outcome") == expected_outcome,
            f"{name}: archive outcome mismatch")
    terminal = turn.get("terminal")
    if response.get("transport_only_cancelled"):
        require(expected_outcome == "cancelled" and response["raw"] == b"" and
                isinstance(terminal, dict) and terminal.get("state") == "cancelled" and
                terminal.get("request_id") is None and terminal.get("legs") == [] and
                turn.get("response") == "" and "replay" not in turn and
                turn.get("narrative_events", []) == [] and turn.get("media_artifacts", []) == [],
                f"{name}: invalid transport-only cancellation notice")
        return {"media": {}, "pcm_bytes": 0, "pcm_artifacts": 0}
    require(isinstance(terminal, dict) and terminal.get("state") == expected_outcome and
            terminal.get("request_id") in (None, response["public_request_id"]),
            f"{name}: archive terminal/request notice mismatch")
    require(turn.get("response") == response["text"],
            f"{name}: archive response differs from raw response")
    replay = turn.get("replay")
    require(isinstance(replay, dict) and replay.get("transport") == "raw-sse",
            f"{name}: raw replay missing")
    raw = decode_base64(replay.get("raw_sse"), f"{name}:raw_sse")
    require(raw == response["raw"] and replay.get("raw_sse_bytes") == len(raw) and
            replay.get("raw_sse_sha256") == sha(raw),
            f"{name}: archive raw SSE is not the captured public response")
    lengths = replay.get("chunk_lengths")
    require(isinstance(lengths, list) and lengths and
            all(is_int(value) and value >= 0 for value in lengths) and sum(lengths) == len(raw),
            f"{name}: invalid transport chunk lengths")
    expected_required = expected_outcome == "completed"
    require(replay.get("terminal_required") is expected_required,
            f"{name}: terminal policy mismatch")
    expected_events = [projected_event(event) for event in response["events"]]
    require(turn.get("narrative_events", []) == expected_events,
            f"{name}: narrative event projection mismatch")
    media = reassemble_artifacts(response["events"], name)
    expected_media = turn.get("media_artifacts", [])
    require(isinstance(expected_media, list) and
            len({item.get("artifact_id") for item in expected_media}) == len(expected_media) and
            {item["artifact_id"]: canonical(item) for item in media} ==
            {item["artifact_id"]: canonical(item) for item in expected_media},
            f"{name}: content-addressed media index mismatch")
    terminal_event = next((event for event in response["events"]
                           if event.get("kind") in TERMINALS), None)
    if terminal_event is not None:
        require(terminal.get("legs", []) == terminal_event.get("legs", []),
                f"{name}: terminal legs mismatch")
    return {"media": media,
            "pcm_bytes": sum(item["bytes"] for item in media if item["mime_type"] == "audio/pcm"),
            "pcm_artifacts": sum(item["mime_type"] == "audio/pcm" for item in media)}


def decode_native_prompts(root):
    require(msgpack is not None, "msgpack is required to decode VlmObserver evidence")
    rows = read_jsonl(root / "vlm-prompts.jsonl")
    prompts = {}
    for index, row in enumerate(rows):
        context = f"vlm-prompts[{index}]"
        require(isinstance(row, dict) and isinstance(row.get("request_id"), str) and
                row["request_id"] and row["request_id"] not in prompts,
                f"{context}: missing or duplicate native request id")
        raw = decode_base64(row.get("raw_base64"), context)
        require(row.get("raw_sha256") == sha(raw), f"{context}: observer raw digest mismatch")
        try:
            message = msgpack.unpackb(raw, raw=False, strict_map_key=False)
        except Exception as exc:
            raise VerificationError(f"{context}: invalid msgpack submit") from exc
        require(isinstance(message, dict) and message.get("type") == "submit" and
                message.get("request_id") == row["request_id"],
                f"{context}: native submit/request id mismatch")
        try:
            inputs = message["data"]["request"]["inputs"]
            session_id = inputs["session_id"]
            messages = normalize_messages(inputs["messages"], context)
        except (KeyError, TypeError) as exc:
            raise VerificationError(f"{context}: native request inputs missing") from exc
        require(isinstance(session_id, str) and session_id,
                f"{context}: native session id missing")
        require(row.get("messages") == messages,
                f"{context}: producer normalization differs from decoded msgpack")
        prompts[row["request_id"]] = {"request_id": row["request_id"],
                                      "session_id": session_id, "messages": messages,
                                      "observed_monotonic_ns": row.get("observed_monotonic_ns")}
    return prompts


def verify_stack(root, arm):
    stack_path = root / "stack-report.json"
    stack = read_json(stack_path)
    require(stack.get("schema") == "umminfer-storyboard-84-live-stack/v1" and
            re.fullmatch(r"[0-9a-f]{40}", stack.get("execution_commit", "")) is not None,
            f"{arm}: invalid stack schema/source identity")
    for phase in ("gpu1_before", "gpu1_after"):
        gpu = stack.get(phase, {})
        require(gpu.get("index") == 1 and gpu.get("memory_used_mib") <= 16 and
                gpu.get("utilization_percent") == 0 and gpu.get("compute_processes") == [],
                f"{arm}: GPU 1 was not idle at {phase}")
    processes = stack.get("processes")
    require(isinstance(processes, list) and processes, f"{arm}: owned process records missing")
    require(len({process.get("pid") for process in processes}) == len(processes),
            f"{arm}: duplicate owned process pid")
    for process in processes:
        require(is_int(process.get("pid")) and process["pid"] > 0 and
                process.get("stop_monotonic_ns") is not None and
                process.get("returncode") in (0, -15) and process.get("stop_error") is None,
                f"{arm}: owned process did not stop cleanly: {process}")
    require(isinstance(stack.get("binaries"), dict) and stack["binaries"] and
            isinstance(stack.get("models"), dict) and stack["models"],
            f"{arm}: binary/model identities missing")
    return stack


def verify_rendered_prompts(root, prompts, replay_binary):
    path = root / "rendered-prompts.json"
    if not path.exists():
        return {"status": "NOT_RUN", "requests": 0}
    require(replay_binary.is_file() and os.access(replay_binary, os.X_OK),
            "rendered-prompts.json exists but --replay-binary is not executable")
    document = read_json(path)
    require(document.get("schema") == NATIVE_REPLAY_SCHEMA and
            isinstance(document.get("requests"), list) and document["requests"],
            "invalid rendered-prompts.json")
    entries = document["requests"]
    ids = [entry.get("id") for entry in entries]
    require(len(ids) == len(set(ids)) and set(ids) == set(prompts),
            "rendered prompts must cover every native VLM prompt exactly once")
    for entry in entries:
        native_id = entry["id"]
        expected_messages = prompts[native_id]["messages"]
        require(entry.get("messages") == expected_messages,
                f"rendered prompt messages differ from msgpack: {native_id}")
        with tempfile.TemporaryDirectory(prefix="ns08-native-replay-") as directory:
            work = Path(directory)
            source = work / "input.json"
            output = work / "output.json"
            source.write_text(json.dumps({"schema": NATIVE_REPLAY_SCHEMA,
                                          "requests": [{"id": native_id,
                                                        "messages": expected_messages}]}),
                              encoding="utf-8")
            result = subprocess.run([str(replay_binary), "--input", str(source),
                                     "--output", str(output)], capture_output=True,
                                    text=True, timeout=30)
            require(result.returncode == 0 and output.is_file(),
                    f"native renderer failed for {native_id}: {result.stdout}{result.stderr}")
            recomputed = read_json(output)
        require(recomputed.get("schema") == NATIVE_REPLAY_SCHEMA and
                recomputed.get("requests") == [entry],
                f"native rendered prompt/messages are not byte-exact for {native_id}")
    return {"status": "PASS", "requests": len(entries)}


def verify_arm(root, arm, replay_binary):
    root = root.resolve()
    require(root.is_dir(), f"{arm}: evidence root is missing")
    manifest = verify_manifest(root)
    result = read_json(root / "result.json")
    require(result.get("schema") == "umminfer-ns08-first-turn-history-browser-v1" and
            result.get("arm") == arm and result.get("real_model") is True,
            f"{arm}: browser result identity mismatch")
    require(result.get("failures") == [] and
            all(check.get("ok") is True for check in result.get("checks", [])),
            f"{arm}: external browser harness retained failures")
    require(result.get("physical_microphone") == "NOT_RUN" and
            result.get("physical_speaker") == "NOT_RUN" and
            result.get("perceptual_quality") == "NOT_RUN",
            f"{arm}: physical/perceptual scope was overstated")
    stack = verify_stack(root, arm)

    request_files = {path.stem for path in (root / "requests").glob("*.json")}
    response_files = {path.stem for path in (root / "responses").glob("*.sse")}
    require(request_files == set(REQUEST_NAMES) and response_files == set(REQUEST_NAMES),
            f"{arm}: expected exactly four browser and two direct request/response captures")
    network = result.get("network")
    require(isinstance(network, list) and len(network) == len(REQUEST_NAMES) and
            {row.get("name") for row in network} == set(REQUEST_NAMES),
            f"{arm}: browser network index is incomplete or ambiguous")
    network_by_name = {row["name"]: row for row in network}

    requests, responses = {}, {}
    public_ids = set()
    for name in REQUEST_NAMES:
        request_raw = (root / "requests" / f"{name}.json").read_bytes()
        response_raw = (root / "responses" / f"{name}.sse").read_bytes()
        request = json.loads(request_raw)
        session_id = request.get("session_id")
        require(isinstance(session_id, str) and session_id,
                f"{arm}/{name}: public session id missing")
        current = public_current_user(request, name)
        requests[name] = {"body": request, "current": current, "raw": request_raw}
        expected_outcome = "cancelled" if name == CANCELLED else "completed"
        response = parse_sse(response_raw, f"{arm}/{name}", session_id, expected_outcome)
        response["raw"] = response_raw
        responses[name] = response
        if response["public_request_id"] is not None:
            require(response["public_request_id"] not in public_ids,
                    f"{arm}: duplicate public request id {response['public_request_id']}")
            public_ids.add(response["public_request_id"])
        network_row = network_by_name[name]
        require(network_row.get("request_sha256") == sha(request_raw) and
                network_row.get("session_id") == session_id and
                network_row.get("session_history") is (request.get("session_history") is True),
                f"{arm}/{name}: network/request identity mismatch")
        if name in DIRECT_NAMES:
            require(network_row.get("response_sha256") == sha(response_raw),
                    f"{arm}/{name}: direct response digest mismatch")
        else:
            require("seed" not in request and "frame_selection" not in canonical(request),
                    f"{arm}/{name}: manual seed/frame control was used")

    prompts = decode_native_prompts(root)
    require(len(prompts) == len(REQUEST_NAMES),
            f"{arm}: expected six unique native VLM submits, got {len(prompts)}")
    native_by_name = {}
    used_native = set()
    for name in REQUEST_NAMES:
        public_id = responses[name]["public_request_id"]
        if public_id is None:
            require(name == CANCELLED and responses[name].get("transport_only_cancelled") is True,
                    f"{arm}/{name}: missing public id is only valid for transport-only cancellation")
            candidates = [native_id for native_id, prompt in prompts.items()
                          if native_id not in used_native and
                          prompt["session_id"] == requests[name]["body"]["session_id"] and
                          without_leading_system(prompt["messages"]) and
                          without_leading_system(prompt["messages"])[-1] == requests[name]["current"]]
        else:
            candidates = [native_id for native_id in (public_id, public_id + ":vlm")
                          if native_id in prompts]
        require(len(candidates) == 1, f"{arm}/{name}: native/public id binding is not unique")
        native_id = candidates[0]
        require(native_id not in used_native, f"{arm}: native request id reused")
        used_native.add(native_id)
        prompt = prompts[native_id]
        require(prompt["session_id"] == requests[name]["body"]["session_id"],
                f"{arm}/{name}: native/public session mismatch")
        history = without_leading_system(prompt["messages"])
        require(history and history[-1] == requests[name]["current"],
                f"{arm}/{name}: current user message is not the final native message")
        native_by_name[name] = prompt

    original_raw = (root / "original.replay.json").read_bytes()
    reexported_raw = (root / "reexported.replay.json").read_bytes()
    require(original_raw == reexported_raw,
            f"{arm}: immediate whole-file re-export is not byte exact")
    original = json.loads(original_raw)
    require(original.get("schema") == REPLAY_SCHEMA and
            original.get("replay_format") == REPLAY_FORMAT and
            isinstance(original.get("turns"), list) and len(original["turns"]) == 1,
            f"{arm}: original archive must contain exactly the completed setup turn")
    main_session = original.get("session_id")
    require(isinstance(main_session, str) and main_session and
            all(requests[name]["body"]["session_id"] == main_session for name in BROWSER_NAMES) and
            requests["explicit-stateless"]["body"]["session_id"] == main_session,
            f"{arm}: browser/stateless requests do not own the original session")
    require(requests["cross-session"]["body"]["session_id"] != main_session,
            f"{arm}: cross-session request reused the main session")
    setup_archive = verify_archive_turn(original["turns"][0], responses[SETUP],
                                        f"{arm}/original-setup", "completed")
    require(setup_archive["pcm_artifacts"] > 0 and setup_archive["pcm_bytes"] > 0,
            f"{arm}: original archive has no raw PCM")
    setup_assistant = original["turns"][0]["response"]
    setup_pair = [requests[SETUP]["current"],
                  {"role": "assistant", "content": setup_assistant}]
    followup_pair = [requests[FOLLOWUP]["current"],
                     {"role": "assistant", "content": responses[FOLLOWUP]["text"]}]

    expected_flags = {SETUP: arm == "after", FOLLOWUP: True, CANCELLED: True,
                      POST_CANCEL: True, "explicit-stateless": False, "cross-session": True}
    for name, expected in expected_flags.items():
        require((requests[name]["body"].get("session_history") is True) is expected,
                f"{arm}/{name}: public session_history flag mismatch")
    expected_history = {
        SETUP: [requests[SETUP]["current"]],
        FOLLOWUP: (setup_pair + [requests[FOLLOWUP]["current"]] if arm == "after"
                   else [requests[FOLLOWUP]["current"]]),
        CANCELLED: ((setup_pair if arm == "after" else []) + followup_pair +
                    [requests[CANCELLED]["current"]]),
        POST_CANCEL: ((setup_pair if arm == "after" else []) + followup_pair +
                      [requests[POST_CANCEL]["current"]]),
        "explicit-stateless": [requests["explicit-stateless"]["current"]],
        "cross-session": [requests["cross-session"]["current"]],
    }
    for name, expected in expected_history.items():
        actual = without_leading_system(native_by_name[name]["messages"])
        require(actual == expected,
                f"{arm}/{name}: exact retained-history mismatch; expected={expected!r} actual={actual!r}")

    cancelled_raw = (root / "cancelled.replay.json").read_bytes()
    cancelled = json.loads(cancelled_raw)
    require(cancelled.get("schema") == REPLAY_SCHEMA and
            cancelled.get("replay_format") == REPLAY_FORMAT and
            cancelled.get("session_id") == main_session and
            isinstance(cancelled.get("turns"), list) and len(cancelled["turns"]) == 3,
            f"{arm}: cancelled archive must contain setup, follow-up, and cancelled turns")
    require(cancelled["turns"][0].get("response") == setup_assistant,
            f"{arm}: cancelled archive changed setup history")
    verify_archive_turn(cancelled["turns"][1], responses[FOLLOWUP],
                        f"{arm}/cancelled-archive-followup", "completed")
    cancelled_archive = verify_archive_turn(cancelled["turns"][2], responses[CANCELLED],
                                            f"{arm}/cancelled-turn", "cancelled")
    answers = result.get("answers")
    require(isinstance(answers, list), f"{arm}: browser answers missing")
    answer_map = {}
    for answer in answers:
        require(answer.get("label") not in answer_map, f"{arm}: duplicate browser answer label")
        answer_map[answer.get("label")] = answer.get("text")
    require(set(answer_map) == {"setup", "followup", "cancelled", "post-cancel-stateless-check"},
            f"{arm}: browser answer labels mismatch")
    require(answer_map["setup"] == setup_assistant and
            answer_map["followup"] == responses[FOLLOWUP]["text"] ==
            cancelled["turns"][1].get("response") and
            answer_map["cancelled"] == responses[CANCELLED]["text"] and
            answer_map["post-cancel-stateless-check"] == responses[POST_CANCEL]["text"],
            f"{arm}: reported browser answer is not verbatim raw/archive prose")
    require(responses[FOLLOWUP]["text"].strip(), f"{arm}: empty follow-up response")

    cancelled_prompt = requests[CANCELLED]["current"]["content"]
    cancelled_response = responses[CANCELLED]["text"]
    post_contents = [message["content"] for message in
                     without_leading_system(native_by_name[POST_CANCEL]["messages"])]
    require(all(cancelled_prompt not in content for content in post_contents) and
            all(not cancelled_response or cancelled_response not in content
                for content in post_contents),
            f"{arm}: cancelled prompt/response entered post-cancel history")
    marker_prompt = requests["explicit-stateless"]["current"]["content"]
    require(STATELESS_MARKER in marker_prompt,
            f"{arm}: explicit-stateless unique marker missing")
    for name in ("cross-session", CANCELLED, POST_CANCEL):
        later_text = canonical(without_leading_system(native_by_name[name]["messages"])) + \
            responses[name]["text"]
        require(STATELESS_MARKER not in later_text,
                f"{arm}: explicit-stateless marker leaked into {name}")
    cross_prompt = requests["cross-session"]["current"]["content"]
    cross_response = responses["cross-session"]["text"]
    for name in (CANCELLED, POST_CANCEL):
        retained = canonical(without_leading_system(native_by_name[name]["messages"]))
        require(cross_prompt not in retained and
                (not cross_response or cross_response not in retained),
                f"{arm}: cross-session history leaked into {name}")

    rendered = verify_rendered_prompts(root, prompts, replay_binary)
    followup_answer = responses[FOLLOWUP]["text"]
    return {
        "status": "PASS", "arm": arm, "manifest": manifest,
        "execution_commit": stack["execution_commit"],
        "stack_report_sha256": sha((root / "stack-report.json").read_bytes()),
        "binaries": stack["binaries"], "models": stack["models"],
        "metrics": {
            "public_requests": len(REQUEST_NAMES), "browser_requests": len(BROWSER_NAMES),
            "direct_requests": len(DIRECT_NAMES),
            "unique_public_request_ids": len(public_ids), "unique_native_request_ids": len(prompts),
            "setup_retained": arm == "after", "followup_history_exact": True,
            "followup_completion_retained": True, "cancelled_prompt_response_excluded": True,
            "explicit_stateless_read_write_isolated": True, "cross_session_isolated": True,
            "whole_original_reexport_byte_exact": True,
            "followup_answer_reported_verbatim": True,
            "original_pcm_artifacts": setup_archive["pcm_artifacts"],
            "original_pcm_bytes": setup_archive["pcm_bytes"],
            "cancelled_pcm_bytes": cancelled_archive["pcm_bytes"],
            "transport_only_cancelled": responses[CANCELLED].get("transport_only_cancelled", False),
            "history_message_counts": {name: len(without_leading_system(prompt["messages"]))
                                       for name, prompt in native_by_name.items()},
            "followup_context_terms": {"solar": "solar" in followup_answer.casefold(),
                                       "water": "water" in followup_answer.casefold()},
        },
        "original_sha256": sha(original_raw), "reexported_sha256": sha(reexported_raw),
        "cancelled_sha256": sha(cancelled_raw),
        "native_rendered_prompts": rendered,
        "physical_microphone": "NOT_RUN", "physical_speaker": "NOT_RUN",
        "perceptual_quality": "NOT_RUN", "macos": "NOT_RUN",
    }


def outside_roots(out, roots):
    resolved = out.resolve()
    for root in roots:
        root = root.resolve()
        require(resolved != root and root not in resolved.parents,
                f"--out must be outside evidence root {root}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--replay-binary", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    report = {"schema": SCHEMA, "status": "FAIL", "arms": {}, "failures": []}
    try:
        require(args.before.resolve() != args.after.resolve(),
                "Before and After evidence roots must differ")
        outside_roots(args.out, (args.before, args.after))
        require(not args.out.exists(), f"refusing to overwrite report: {args.out}")
        for arm, root in (("before", args.before), ("after", args.after)):
            try:
                report["arms"][arm] = verify_arm(root, arm, args.replay_binary)
            except Exception as exc:
                report["arms"][arm] = {"status": "FAIL", "arm": arm}
                report["failures"].append(f"{arm}: {exc}")
        before, after = report["arms"]["before"], report["arms"]["after"]
        if before["status"] == after["status"] == "PASS":
            try:
                require(before["models"] == after["models"],
                        "paired model identities differ")
                for binary in ("vlm", "tts", "wan"):
                    require(before["binaries"].get(binary) == after["binaries"].get(binary),
                            f"paired native binary identity differs: {binary}")
                require(before["metrics"]["setup_retained"] is False and
                        after["metrics"]["setup_retained"] is True,
                        "paired first-turn history improvement was not demonstrated")
            except Exception as exc:
                report["failures"].append(f"paired: {exc}")
        report["paired"] = {
            "first_turn_history_fix_verified": not report["failures"],
            "same_model_identities": not report["failures"],
            "same_native_vlm_tts_wan_identities": not report["failures"],
        }
        report["status"] = "PASS" if not report["failures"] else "FAIL"
    except Exception as exc:
        report["failures"].append(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({"status": report["status"], "out": str(args.out),
                      "failures": report["failures"]}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
