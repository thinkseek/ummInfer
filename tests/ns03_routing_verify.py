#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent fresh-process NS-03 raw-SSE/media/manifest verification.

No producer module is imported. --write-manifest first records every regular file
except the manifest itself; a later process verifies that complete file set.
"""
import argparse
import base64
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

MANIFEST_FORMAT = "umminfer-ns03-routing-external-manifest-v1"
FULL_SCORECARD_FORMAT = "umminfer-ns03-routing-scorecard-v1"
FOLLOWUP_SCORECARD_FORMAT = "umminfer-ns03-followup-history-scorecard-v1"
CLASSES = ("text", "static", "dynamic", "followup")
DIRECTIVE = re.compile(r"\[\[\s*(?:image|video|visual)\s*:", re.IGNORECASE)
OFFICIAL_BEFORE = "91310b36184c86a1c147debb27db8643cc0cb3e4"
BASELINE_SOURCES = {
    "src/pipeline/narrative_prompt.cpp": "0c495a8800bdf2ca351b8862299eae3c3c5adbe45d8adfb2e213650e2892dc7b",
    "src/pipeline/narrative_parser.cpp": "cb864c55f67ca13ae70604a4a3ce1945f68f37b52ac5afa4e59fd3eb355461c8",
    "src/pipeline/narrative_parser.hpp": "c876b79ae314ebd5a876377a01e8818c7210fa97e55b6bfc0ee33ad5e145e9b3",
    "src/pipeline/unified.cpp": "6a45488b040cdd35cf2ff304f12c6a57296d90c1bea155e7166229ab2a60a07f",
    "src/pipeline/unified_schema.cpp": "c62ce75b576f68669ff15c34d0f1ab605fc7dc36bd4034bbdae769e3ce6efe9a",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def expected_request(case, session_id, setup, initial_history=False):
    prompt = case["setup"]["prompt"] if setup else case["prompt"]
    body = {
        "model": "qwen3-native",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256, "temperature": 0, "stream": True,
        "session_id": session_id,
        "image": {"size": "512x512", "output_format": "png"},
        "video": {"size": "704x480", "output_format": "mp4", "seconds": 2,
                  "num_frames": 29, "steps": 25, "guidance_scale": 5.0},
        "narrative": {"enabled": True, "visual_mode": "auto", "max_slots": 3,
                      "allow_text_only": True},
        "audio": {"voice": "vivian", "format": "wav", "stream": True},
        "modalities": ["text", "image", "video", "audio"],
    }
    if initial_history or (case["class"] == "followup" and not setup):
        body["session_history"] = True
    return body


def normalize_vlm_messages(messages):
    normalized = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            require(len(content) == 1 and content[0].get("type") == "text" and
                    isinstance(content[0].get("text"), str),
                    "VLM capture expected one text content part")
            content = content[0]["text"]
        require(isinstance(content, str), "VLM capture expected text content")
        normalized.append({"role": message.get("role"), "content": content})
    return normalized


def expected_vlm_messages(case, answer_name, setup_record, initial_history):
    if answer_name == "setup":
        return [{"role": "user", "content": case["setup"]["prompt"]}]
    messages = []
    if case["class"] == "followup" and initial_history:
        require(setup_record is not None, "follow-up capture lacks its real setup")
        messages.extend([
            {"role": "user", "content": case["setup"]["prompt"]},
            {"role": "assistant", "content": setup_record["clean_text"]},
        ])
    messages.append({"role": "user", "content": case["prompt"]})
    return messages


def verify_vlm_capture(case_dir, reported, public_request_id, expected_messages,
                       native_request_bindings):
    import msgpack
    path = Path(case_dir) / "vlm-prompts.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    require(len(rows) == 1, f"{path}: expected exactly one VLM submit")
    row = rows[0]
    raw = base64.b64decode(row["raw_base64"], validate=True)
    require(sha_bytes(raw) == row["raw_sha256"], f"{path}: raw msgpack hash mismatch")
    message = msgpack.unpackb(raw, raw=False, strict_map_key=False)
    require(message.get("type") == "submit" and message.get("request_id") == row["request_id"],
            f"{path}: invalid submit identity")
    inputs = message.get("data", {}).get("request", {}).get("inputs", {})
    messages = inputs.get("messages")
    require(isinstance(messages, list) and normalize_vlm_messages(messages) == row["messages"],
            f"{path}: normalized messages differ from raw msgpack")
    native_id = row["request_id"]
    require(native_id in (public_request_id, public_request_id + ":vlm"),
            f"{path}: native/public request mismatch")
    require(native_id not in native_request_bindings, f"{path}: native request id was reused")
    native_request_bindings[native_id] = public_request_id
    while row["messages"] and row["messages"][0].get("role") == "system":
        row["messages"].pop(0)
    require(row["messages"] == expected_messages,
            f"{path}: post-prepend VLM history mismatch: {row['messages']} != {expected_messages}")
    metadata = {"path": path.name, "rows": 1, "sha256": sha_file(path),
                "native_request_id": native_id}
    require(reported.get("vlm_capture") == metadata,
            f"{path}: compact producer capture metadata mismatch")
    return metadata


def write_manifest(root):
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    require(not manifest_path.exists(), "refusing to overwrite an existing frozen manifest")
    for name in ("suite.json", "preregistration.json", "candidate.json", "identities.json", "scorecard.json"):
        require((root / name).is_file(), f"missing required evidence file {name}")
    runs = {}
    for key in ("development/before", "development/after", "heldout/before", "heldout/after"):
        relative = "runs/" + key.replace("/", "-")
        require((root / relative / "report.json").is_file(), f"missing run {key}")
        runs[key] = relative
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise AssertionError(f"manifest root contains a symlink: {path.relative_to(root)}")
        if path.is_file() and path != manifest_path:
            relative = path.relative_to(root).as_posix()
            files[relative] = {"sha256": sha_file(path), "bytes": path.stat().st_size}
    manifest = {"format": MANIFEST_FORMAT, "suite": "suite.json",
                "preregistration": "preregistration.json", "candidate": "candidate.json",
                "identities": "identities.json", "scorecard": "scorecard.json",
                "runs": runs, "self_excluded": "manifest.json", "files": files}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"files": len(files), "manifest_sha256": sha_file(manifest_path)}))


def payloads(raw):
    output = []
    data = []
    normalized_input = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    for line in normalized_input.split(b"\n"):
        if not line:
            if data:
                output.append(b"\n".join(data).decode("utf-8"))
                data = []
        elif line.startswith(b"data:"):
            item = line[5:]
            data.append(item[1:] if item.startswith(b" ") else item)
    if data:
        output.append(b"\n".join(data).decode("utf-8"))
    return output


def decode(ffmpeg, path):
    result = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-f", "null", "-"],
                            capture_output=True, timeout=120)
    require(result.returncode == 0, f"independent ffmpeg decode failed for {path.name}")


def require_no_owned_processes(identity):
    owned_dirs = set()
    for name, binary in identity["binaries"].items():
        path = Path(binary["path"]).resolve()
        owned_dirs.add(path.parents[2] if name == "server" else path.parents[1])
    owned_dirs.add(Path(identity["flux_stage"]["path"]).resolve().parents[1])
    patterns = ["umm-infer", "umm-native-stage-qwen38", "umm-native-stage-tts",
                "umm-native-stage-video", "umminfer_flux2_stage.py"]
    for pattern in patterns:
        result = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=20)
        require(result.returncode in (0, 1), f"pgrep failed for {pattern}")
        for pid in result.stdout.split():
            try:
                cwd = Path(os.readlink(f"/proc/{pid}/cwd")).resolve()
                executable = Path(os.readlink(f"/proc/{pid}/exe")).name
            except FileNotFoundError:
                continue
            if pattern == "umminfer_flux2_stage.py":
                if "python" not in executable:
                    continue
            elif executable != pattern:
                continue
            require(cwd not in owned_dirs,
                    f"task-owned process {pid} remains in {cwd}")


def raw_case(case, run_dir, answer_name, ffmpeg, *, initial_history=False,
             require_capture=False, setup_record=None, reported_record=None,
             native_request_bindings=None):
    case_dir = run_dir / "responses" / case["id"] / answer_name
    raw = (case_dir / "raw.sse").read_bytes()
    request = read_json(case_dir / "request.json")
    require(request == expected_request(case, request.get("session_id"), answer_name == "setup",
                                        initial_history=initial_history),
            f"{case['id']}/{answer_name}: request differs from frozen prompt/client contract")
    errors = []
    media_errors = []
    frames = payloads(raw)
    require(frames and frames[-1] == "[DONE]" and frames.count("[DONE]") == 1,
            f"{case['id']}/{answer_name}: invalid DONE framing")
    events = []
    public_ids = set()
    for item in frames:
        if item == "[DONE]":
            continue
        frame = json.loads(item)
        if frame.get("id"):
            public_ids.add(frame["id"])
        if isinstance(frame.get("unified"), dict):
            events.append(frame["unified"])
    require(events and len(public_ids) == 1, f"{case['id']}/{answer_name}: missing public identity")
    request_id = next(iter(public_ids))
    require(all(e.get("request_id", request_id) == request_id and
                e.get("session_id") == request["session_id"] for e in events),
            f"{case['id']}/{answer_name}: cross-session/request event")
    require([e.get("seq") for e in events] == list(range(len(events))),
            f"{case['id']}/{answer_name}: noncontiguous event sequence")
    terminals = [e for e in events if e.get("kind") in ("complete", "error", "cancelled")]
    require(len(terminals) == 1 and terminals[0] is events[-1] and terminals[0].get("kind") == "complete",
            f"{case['id']}/{answer_name}: invalid terminal")
    text = "".join(e.get("text", "") for e in events if e.get("kind") == "vlm_delta")
    if terminals[0].get("text") != text:
        errors.append("terminal text differs from clean deltas")
    if DIRECTIVE.search(text):
        errors.append("raw narrative directive leaked into clean text")
    if not text.strip():
        errors.append("empty clean response text")
    slots = [{"slot_id": e.get("slot_id"), "slot_index": e.get("slot_index"),
              "media_type": e.get("media_type"), "stage": e.get("stage"),
              "prompt_sha256": sha_bytes(str(e.get("prompt", "")).encode()),
              "text_offset": e.get("text_offset")}
             for e in events if e.get("kind") == "visual_slot"]
    for index, slot in enumerate(slots):
        require(slot["slot_id"] == f"slot-{index}" and slot["slot_index"] == index and
                slot["media_type"] in ("image", "video") and slot["stage"] == slot["media_type"],
                f"{case['id']}/{answer_name}: invalid slot")
    groups = {}
    for event in events:
        if event.get("kind") == "artifact":
            descriptor = event.get("artifact") or {}
            require(descriptor.get("encoding") == "base64", "artifact encoding is not base64")
            groups.setdefault(descriptor.get("artifact_id"), []).append(descriptor)
    artifacts = []
    audio_count = audio_bytes = 0
    for artifact_id, chunks in groups.items():
        require(isinstance(artifact_id, str) and artifact_id, "artifact id missing")
        chunks.sort(key=lambda item: item.get("index", -1))
        total = chunks[0].get("total")
        require(isinstance(total, int) and total > 0 and len(chunks) == total and
                [c.get("index") for c in chunks] == list(range(total)), "incomplete artifact")
        first = chunks[0]
        for chunk in chunks:
            for field in ("total", "media_type", "mime_type", "slot_id", "slot_index"):
                require(chunk.get(field) == first.get(field), f"artifact chunk {field} changed")
            final_chunk = chunk.get("index") + 1 == total
            require(("bytes" in chunk) == final_chunk and ("sha256" in chunk) == final_chunk,
                    "artifact digest fields are not final-chunk-only")
        raw_artifact = b"".join(base64.b64decode(c["data"], validate=True) for c in chunks)
        final = chunks[-1]
        digest = sha_bytes(raw_artifact)
        require(final.get("sha256") == digest and final.get("bytes") == len(raw_artifact),
                "artifact hash mismatch")
        media = final.get("media_type")
        item = {"artifact_id": artifact_id, "media_type": media, "mime_type": final.get("mime_type"),
                "sha256": digest, "bytes": len(raw_artifact), "slot_id": final.get("slot_id"),
                "slot_index": final.get("slot_index"), "decoded": None}
        if media in ("image", "video"):
            matches = [s for s in slots if s["slot_id"] == item["slot_id"]]
            require(len(matches) == 1 and matches[0]["media_type"] == media and
                    matches[0]["slot_index"] == item["slot_index"], "stale/false visual artifact")
            suffix = "png" if media == "image" else "mp4"
            media_path = case_dir / "media" / f"{digest}.{suffix}"
            require(media_path.read_bytes() == raw_artifact, "CAS media differs from raw SSE")
            try:
                decode(ffmpeg, media_path)
                item["decoded"] = True
            except Exception as error:
                item["decoded"] = False
                media_errors.append({"slot_id": item["slot_id"], "sha256": digest,
                                     "error": f"{type(error).__name__}: {error}"})
        else:
            require(media == "audio" and final.get("mime_type") == "audio/pcm" and
                    len(raw_artifact) > 0 and len(raw_artifact) % 2 == 0, "invalid PCM16 artifact")
            audio_count += 1
            audio_bytes += len(raw_artifact)
            continue
        artifacts.append(item)
    for leg in terminals[0].get("legs") or []:
        require(leg.get("state") not in ("pending", "running"), "pending/running terminal leg")
        if leg.get("slot_id"):
            require(any(s["slot_id"] == leg["slot_id"] for s in slots), "unknown terminal slot")
    require({leg.get("slot_id") for leg in terminals[0].get("legs") or [] if leg.get("slot_id")} ==
            {slot["slot_id"] for slot in slots}, "missing terminal slot")
    leg_errors = [{"stage": e.get("stage"), "slot_id": e.get("slot_id"),
                   "slot_index": e.get("slot_index"), "error": e.get("error", "")}
                  for e in events if e.get("kind") == "leg_error"]
    result = {"status": "captured", "case_id": case["id"], "errors": errors,
              "leg_errors": leg_errors, "media_errors": media_errors,
              "request_sha256": sha_bytes(canonical(request)), "raw_response_sha256": sha_bytes(raw),
              "clean_text_sha256": sha_bytes(text.encode()), "clean_text": text,
              "slots": slots, "artifacts": artifacts, "terminal_kind": "complete",
              "terminal_count": 1, "done_count": 1, "request_id": request_id,
              "audio": {"artifact_count": audio_count, "bytes": audio_bytes, "pcm16_valid": True},
              "session_history": request.get("session_history", False), "session_id": request["session_id"],
              "parse_ambiguous": False}
    capture_path = case_dir / "vlm-prompts.jsonl"
    if require_capture or capture_path.exists():
        require(reported_record is not None, f"{case['id']}/{answer_name}: missing producer capture record")
        if native_request_bindings is None:
            native_request_bindings = {}
        expected_messages = expected_vlm_messages(case, answer_name, setup_record, initial_history)
        result["vlm_capture"] = verify_vlm_capture(
            case_dir, reported_record, request_id, expected_messages, native_request_bindings)
    return result


def compare_public_record(independent, reported, label):
    keys = ("errors", "leg_errors", "media_errors", "request_sha256", "raw_response_sha256", "clean_text_sha256",
            "slots", "artifacts", "terminal_kind", "terminal_count", "done_count", "audio",
            "request_id", "vlm_capture", "session_history", "session_id", "parse_ambiguous")
    for key in keys:
        require(independent.get(key) == reported.get(key), f"{label}: producer/raw mismatch in {key}")


def evaluate(case, record, setup=None):
    expected = {"text": (0, 0), "static": (1, 0), "dynamic": (0, 1), "followup": (0, 0)}[case["class"]]
    images = sum(s["media_type"] == "image" for s in record["slots"])
    videos = sum(s["media_type"] == "video" for s in record["slots"])
    exact = (images, videos) == expected
    route = exact and record["terminal_kind"] == "complete" and record["terminal_count"] == 1 and not record["errors"]
    visuals = [a for a in record["artifacts"] if a["media_type"] in ("image", "video")]
    valid = [a for a in visuals if a["decoded"] is True]
    total_expected = sum(expected)
    e2e = route and len(valid) == total_expected and len(visuals) == total_expected and \
        len({a["slot_id"] for a in valid}) == total_expected and record["audio"]["artifact_count"] > 0 and \
        record["audio"]["bytes"] > 0 and record["audio"]["pcm16_valid"]
    total = images + videos
    redundant = (total_expected == 0 and total > 0) or (total_expected > 0 and total > total_expected)
    result = {"case_id": case["id"], "routing_correct": route, "end_to_end_correct": e2e,
              "redundant_visual_case": redundant,
              "redundant_image_case": images > expected[0], "redundant_video_case": videos > expected[1],
              "redundant_visual_slots": total if total_expected == 0 else max(0, total - total_expected),
              "missing_visual_slots": max(0, total_expected - total),
              "wrong_modality": total == total_expected and not exact,
              "ambiguous": record["status"] != "captured" or record["parse_ambiguous"],
              "voice_present": record["audio"]["artifact_count"] > 0 and record["audio"]["bytes"] > 0 and record["audio"]["pcm16_valid"]}
    if case["class"] == "followup":
        folded = record["clean_text"].casefold()
        context = all(any(token.casefold() in folded for token in group) for group in case["context_any"])
        result["followup_correct"] = e2e and setup is not None and setup["terminal_kind"] == "complete" and \
            not setup["errors"] and record["session_history"] and \
            record["session_id"] == setup["session_id"] and context
    return result


def wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - radius, center + radius]


def aggregate(results, population, kind=None):
    selected = [r for r in results if (population == "all" or r["_case"]["population"] == population)
                and (kind is None or r["_case"]["class"] == kind)]
    n = len(selected)
    correct = sum(r["routing_correct"] for r in selected)
    e2e = sum(r["end_to_end_correct"] for r in selected)
    redundant = sum(r["redundant_visual_case"] for r in selected)
    expected_visuals = sum(sum(r["_case"]["_expected"]) for r in selected)
    followups = [r for r in selected if r["_case"]["class"] == "followup"]
    redundant_artifacts = 0
    for result in selected:
        record = result["_record"]
        valid = [a for a in record["artifacts"] if a["media_type"] in ("image", "video") and a["decoded"]]
        total = len(valid)
        expected = sum(result["_case"]["_expected"])
        redundant_artifacts += (expected == 0 and total > 0) or (expected > 0 and total > expected)
    return {"cases": n, "routing_correct": correct, "routing_accuracy": correct / n if n else None,
            "routing_accuracy_wilson_95": wilson(correct, n), "end_to_end_correct": e2e,
            "end_to_end_accuracy": e2e / n if n else None, "redundant_visual_cases": redundant,
            "redundant_visual_rate": redundant / n if n else None,
            "redundant_image_cases": sum(r["redundant_image_case"] for r in selected),
            "redundant_image_rate": sum(r["redundant_image_case"] for r in selected) / n if n else None,
            "redundant_video_cases": sum(r["redundant_video_case"] for r in selected),
            "redundant_video_rate": sum(r["redundant_video_case"] for r in selected) / n if n else None,
            "redundant_visual_slots": sum(r["redundant_visual_slots"] for r in selected),
            "redundant_visual_slot_rate": sum(r["redundant_visual_slots"] for r in selected) / expected_visuals if expected_visuals else None,
            "redundant_artifact_cases": redundant_artifacts, "redundant_artifact_rate": redundant_artifacts / n if n else None,
            "missing_visual_slots": sum(r["missing_visual_slots"] for r in selected),
            "wrong_modality_cases": sum(r["wrong_modality"] for r in selected),
            "ambiguous_cases": sum(r["ambiguous"] for r in selected),
            "voice_missing_cases": sum(not r["voice_present"] for r in selected),
            "public_leg_error_cases": sum(bool(r["_record"]["leg_errors"]) for r in selected),
            "public_leg_error_count": sum(len(r["_record"]["leg_errors"]) for r in selected),
            "media_error_cases": sum(bool(r["_record"]["media_errors"]) for r in selected),
            "terminal_count": sum(r["_record"]["terminal_count"] for r in selected),
            "image_count": sum(sum(s["media_type"] == "image" for s in r["_record"]["slots"]) for r in selected),
            "video_count": sum(sum(s["media_type"] == "video" for s in r["_record"]["slots"]) for r in selected),
            "valid_image_artifacts": sum(sum(a["media_type"] == "image" and a["decoded"] for a in r["_record"]["artifacts"]) for r in selected),
            "valid_video_artifacts": sum(sum(a["media_type"] == "video" and a["decoded"] for a in r["_record"]["artifacts"]) for r in selected),
            "audio_artifact_count": sum(r["_record"]["audio"]["artifact_count"] for r in selected),
            "followup_cases": len(followups), "followup_correct": sum(r.get("followup_correct") is True for r in followups),
            "followup_accuracy": sum(r.get("followup_correct") is True for r in followups) / len(followups) if followups else None}


def followup_history_gate(before, after):
    def history(results):
        correct = sum(result.get("followup_correct") is True for result in results)
        return {"cases": len(results), "correct": correct, "accuracy": correct / len(results)}

    before_history = history(before)
    after_history = history(after)
    gain = after_history["accuracy"] - before_history["accuracy"]
    return {"before": before_history, "after": after_history, "target": 0.8,
            "before_to_after_gain": gain, "target_met": after_history["accuracy"] >= 0.8,
            "verifiable_gain": gain > 0}


def verify(root, ffmpeg):
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    require(manifest["format"] == MANIFEST_FORMAT and manifest["self_excluded"] == "manifest.json",
            "invalid external manifest format/self-exclusion")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(actual == set(manifest["files"]) | {"manifest.json"}, "manifest is not complete")
    for relative, identity in manifest["files"].items():
        path = root / relative
        require(sha_file(path) == identity["sha256"] and path.stat().st_size == identity["bytes"],
                f"manifest hash/size mismatch: {relative}")
    suite = read_json(root / manifest["suite"])
    prereq = read_json(root / manifest["preregistration"])
    scorecard = read_json(root / manifest["scorecard"])
    identities = read_json(root / manifest["identities"])
    candidate = read_json(root / manifest["candidate"])
    scorecard_format = scorecard.get("format")
    require(scorecard_format in (FULL_SCORECARD_FORMAT, FOLLOWUP_SCORECARD_FORMAT),
            "unknown NS-03 scorecard format")
    focused_followup = scorecard_format == FOLLOWUP_SCORECARD_FORMAT
    if focused_followup:
        require(candidate.get("first_turn_history") is True,
                "focused follow-up scorecard requires candidate first_turn_history=true")
    require(candidate["suite_canonical_sha256"] == sha_bytes(canonical(suite)) and
            candidate["frozen_before_heldout"] is True and
            candidate.get("evaluation_frozen_at_utc"), "candidate freeze evidence is incomplete")
    evaluation_frozen = datetime.fromisoformat(candidate["evaluation_frozen_at_utc"])
    reused_before = candidate.get("comparator_before_reused", False)
    require(isinstance(reused_before, bool), "comparator_before_reused must be boolean")
    integrated_merge = candidate.get("official_main_merge")
    require(integrated_merge is None or isinstance(integrated_merge, dict),
            "official_main_merge must be an object when present")
    if integrated_merge is not None:
        require(integrated_merge.get("merge_commit") == candidate.get("execution_commit") and
                integrated_merge.get("accepted_parent") == candidate.get("base_commit") and
                isinstance(integrated_merge.get("official_main_commit"), str),
                "official main merge identity mismatch")
    require(not focused_followup or not reused_before,
            "focused follow-up bundles cannot reuse comparator Before")
    accepted_before = None
    if reused_before:
        binding = candidate.get("comparator_before") or {}
        accepted_path = Path(binding["scorecard_path"])
        if not accepted_path.is_absolute():
            accepted_path = root / accepted_path
        require(accepted_path.is_file() and
                sha_file(accepted_path) == binding["scorecard_sha256"],
                "accepted Before scorecard path/hash mismatch")
        accepted_before = read_json(accepted_path)
        require(accepted_before.get("suite_sha256") == scorecard["suite_sha256"] and
                accepted_before.get("preregistration_sha256") == scorecard["preregistration_sha256"],
                "accepted Before scorecard is not bound to this suite/preregistration")
        require(isinstance(candidate.get("after_runtime"), dict),
                "reused Before requires exact candidate after_runtime")
        for before_key in ("development/before", "heldout/before"):
            require(scorecard["arms"][before_key]["cases"] ==
                    accepted_before["arms"][before_key]["cases"] and
                    scorecard["arms"][before_key]["aggregate"] ==
                    accepted_before["arms"][before_key]["aggregate"],
                    f"{before_key}: accepted Before cases/aggregates differ")
    require(scorecard["suite_sha256"] == prereq["suite_sha256"] == sha_bytes(canonical(suite)),
            "suite changed after freeze")
    require(scorecard["preregistration_sha256"] == sha_bytes(canonical(prereq)),
            "preregistration changed after freeze")
    require(scorecard["candidate"] == read_json(root / manifest["candidate"]), "candidate mismatch")
    require(scorecard["identities"] == identities, "final identities mismatch")
    model_cache = {}
    all_results = {}
    run_times = {}
    runtime_by_arm = {}
    nonnative_reference = None
    reference_flux = None
    server_by_arm = {}
    setup_streams_verified = 0
    for key, relative in manifest["runs"].items():
        run_dir = root / relative
        report = read_json(run_dir / "report.json")
        identity = read_json(run_dir / "run-identity.json")
        expected_execution = OFFICIAL_BEFORE if key.endswith("/before") else candidate["execution_commit"]
        expected_sources = BASELINE_SOURCES if key.endswith("/before") else candidate["source_sha256"]
        expected_base = OFFICIAL_BEFORE if key.endswith("/before") else candidate["base_commit"]
        baseline_binding = identity.get("baseline_manifest")
        if focused_followup and key.endswith("/before"):
            require(baseline_binding is not None,
                    f"{key}: focused Before requires a Task83 baseline manifest")
        if baseline_binding is not None:
            require(key.endswith("/before"), f"{key}: baseline manifest is valid only for Before")
            baseline_path = Path(baseline_binding["path"])
            require(baseline_path.is_file() and sha_file(baseline_path) == baseline_binding["sha256"],
                    f"{key}: baseline manifest path/hash mismatch")
            baseline = read_json(baseline_path)
            expected_base = baseline["base_commit"]
            expected_execution = expected_base
            expected_sources = baseline["source_sha256"]
        require(identity["base_commit"] == expected_base and
                identity["execution_commit"] == expected_execution and
                identity["source_sha256"] == expected_sources and identity["physical_gpu"] == 1,
                f"{key}: execution/source does not match its frozen arm")
        started = datetime.fromisoformat(identity["started_at_utc"])
        finished = datetime.fromisoformat(report["finished_at_utc"])
        require(started <= finished, f"{key}: invalid capture timestamps")
        run_times[key] = (started, finished)
        require(identities["runs"][key]["run_identity_sha256"] == sha_file(run_dir / "run-identity.json"),
                f"{key}: final identity binding mismatch")
        for name, digest in identity["source_sha256"].items():
            require(sha_file(run_dir / "source" / name) == digest, f"{key}: source hash mismatch {name}")
        for name, binary in identity["binaries"].items():
            require(sha_file(binary["path"]) == binary["sha256"] and Path(binary["path"]).stat().st_size == binary["bytes"],
                    f"{key}: binary identity mismatch {name}")
        require(sha_file(identity["flux_stage"]["path"]) == identity["flux_stage"]["sha256"],
                f"{key}: FLUX stage mismatch")
        for model_name, model in identity["models"].items():
            root_path = model["root"]
            if root_path not in model_cache:
                files = {}
                for path in sorted(p for p in Path(root_path).rglob("*") if p.is_file()):
                    relative_model = path.relative_to(root_path).as_posix()
                    files[relative_model] = {"sha256": sha_file(path), "bytes": path.stat().st_size}
                model_cache[root_path] = files
            require(model_cache[root_path] == model["files"] and
                    model["tree_sha256"] == sha_bytes(canonical(model["files"])),
                    f"{key}: model tree changed for {model_name}")
        arm = key.split("/")[1]
        runtime_signature = {
            "native_binaries": {name: identity["binaries"][name]["sha256"] for name in ("vlm", "tts", "wan")},
            "models": identity["models"], "python_runtime": identity["python_runtime"],
            "flux_stage_sha256": identity["flux_stage"]["sha256"],
        }
        if arm in runtime_by_arm:
            require(runtime_signature == runtime_by_arm[arm], f"{key}: native/model/runtime drift within {arm}")
        else:
            runtime_by_arm[arm] = runtime_signature
        nonnative_signature = {name: runtime_signature[name] for name in
                               ("models", "python_runtime")}
        if nonnative_reference is None:
            nonnative_reference = nonnative_signature
        else:
            require(nonnative_signature == nonnative_reference,
                    f"{key}: model/Python identity drift across arms")
        if reference_flux is None:
            reference_flux = runtime_signature["flux_stage_sha256"]
        elif integrated_merge is None:
            require(runtime_signature["flux_stage_sha256"] == reference_flux,
                    f"{key}: FLUX identity drift across arms")
        if (reused_before or integrated_merge is not None) and arm == "after":
            require(runtime_signature == candidate["after_runtime"],
                    f"{key}: After runtime differs from candidate after_runtime")
        elif integrated_merge is not None and focused_followup and arm == "before":
            require(runtime_signature == candidate["before_runtime"],
                    f"{key}: Before runtime differs from accepted-parent before_runtime")
        elif not reused_before and integrated_merge is None and runtime_by_arm.get("before") is not None:
            require(runtime_signature == runtime_by_arm["before"],
                    f"{key}: native/model/runtime identity drift")
        server_hash = identity["binaries"]["server"]["sha256"]
        if arm in server_by_arm:
            require(server_by_arm[arm] == server_hash, f"{key}: server binary drift within {arm}")
        else:
            server_by_arm[arm] = server_hash
        population, arm = key.split("/")
        case_class = identity.get("case_class", "all")
        require(case_class in ("all", *CLASSES) and report.get("case_class", "all") == case_class,
                f"{key}: case-class identity mismatch")
        require(case_class == ("followup" if focused_followup else "all"),
                f"{key}: wrong case class for scorecard format")
        initial_history = arm == "after" and candidate.get("first_turn_history") is True
        require(identity.get("initial_history", initial_history) == initial_history,
                f"{key}: initial-history identity mismatch")
        observer_marked = identity.get("vlm_observer", False)
        require(isinstance(observer_marked, bool), f"{key}: vlm_observer must be boolean")
        if focused_followup:
            require("initial_history" in identity and observer_marked is True,
                    f"{key}: focused runs require explicit history and VLM observer identity")
        require_capture = observer_marked or initial_history
        require(report["population"] == population and report["arm"] == arm and
                identity["population"] == population and identity["arm"] == arm,
                f"{key}: population/arm mismatch")
        require(all(report["gpu1"][point]["index"] == 1 and
                    report["gpu1"][point]["memory_used_mib"] <= 16 and
                    report["gpu1"][point]["utilization_percent"] == 0
                    for point in ("before", "after_cleanup")) and
                report["health_final"]["in_flight"] == 0,
                f"{key}: GPU/health cleanup failed")
        require_no_owned_processes(identity)
        reports = {r["case_id"]: r for r in report["cases"]}
        selected = [case for case in suite["cases"] if case["population"] == population and
                    (case_class == "all" or case["class"] == case_class)]
        require(len(selected) == (20 if case_class == "all" else 5) and
                set(reports) == {case["id"] for case in selected},
                f"{key}: report does not contain exactly its selected frozen cases")
        # Native ids are process-local; each independent stack may restart at
        # http-1. Require uniqueness within this run, never across fresh stacks.
        native_request_bindings = {}
        results = []
        for case in selected:
            case = dict(case)
            golden = suite["classes"][case["class"]]
            case["_expected"] = [golden["images"], golden["videos"]]
            setup = None
            if case["class"] == "followup":
                setup = raw_case(case, run_dir, "setup", ffmpeg,
                                 initial_history=initial_history, require_capture=require_capture,
                                 reported_record=reports[case["id"]]["setup"],
                                 native_request_bindings=native_request_bindings)
                setup_streams_verified += 1
                compare_public_record(setup, reports[case["id"]]["setup"], f"{key}/{case['id']}/setup")
            answer = raw_case(case, run_dir, "answer", ffmpeg,
                              initial_history=initial_history, require_capture=require_capture,
                              setup_record=setup, reported_record=reports[case["id"]],
                              native_request_bindings=native_request_bindings)
            require(answer["session_id"] == f"ns03-{arm}-{case['id']}",
                    f"{key}/{case['id']}: unexpected frozen session identity")
            compare_public_record(answer, reports[case["id"]], f"{key}/{case['id']}")
            if case["class"] == "followup":
                derived_same_session = answer["session_id"] == setup["session_id"]
                require(answer["session_history"] is True and derived_same_session and
                        reports[case["id"]]["same_session_as_setup"] is True,
                        f"{key}/{case['id']}: real same-session history not requested")
            result = evaluate(case, answer, setup)
            result["_case"] = case
            result["_record"] = answer
            result["_setup"] = setup
            results.append(result)
        all_results[key] = results
    if focused_followup:
        for key in ("development/before", "heldout/before"):
            require(run_times[key][1] <= evaluation_frozen,
                    f"{key} did not finish before final candidate freeze")
        for key in ("development/after", "heldout/after"):
            require(run_times[key][0] >= evaluation_frozen,
                    f"{key} started before final candidate freeze")
    elif not reused_before:
        require(run_times["development/before"][1] <= evaluation_frozen,
                "development Before did not finish before final candidate freeze")
        for key in ("development/after", "heldout/before", "heldout/after"):
            require(run_times[key][0] >= evaluation_frozen,
                    f"{key} started before final candidate freeze")
    for key, results in all_results.items():
        producer_cases = {r["case_id"]: r for r in scorecard["arms"][key]["cases"]}
        for result in results:
            produced = producer_cases[result["case_id"]]
            current_record = result["_record"]
            setup = result["_setup"]
            require(produced["public_leg_errors"] == current_record["leg_errors"] and
                    produced["all_output_legs_completed"] == (not current_record["leg_errors"]) and
                    produced["media_errors"] == current_record["media_errors"],
                    f"{key}/{result['case_id']}: public leg/media errors were not retained")
            for field in ("routing_correct", "end_to_end_correct", "redundant_visual_case",
                          "redundant_image_case", "redundant_video_case", "redundant_visual_slots",
                          "missing_visual_slots", "wrong_modality", "ambiguous", "voice_present"):
                require(produced[field] == result[field], f"{key}/{result['case_id']}: score mismatch {field}")
            require(produced["clean_text_present"] == bool(current_record["clean_text"].strip()),
                    f"{key}/{result['case_id']}: clean-text presence mismatch")
            if result["_case"]["class"] == "followup":
                require(produced["followup_correct"] == result["followup_correct"],
                        f"{key}/{result['case_id']}: followup score mismatch")
                require(produced["followup"]["setup_raw_response_sha256"] == setup["raw_response_sha256"] and
                        produced["followup"]["setup_clean_text_sha256"] == setup["clean_text_sha256"] and
                        produced["followup"]["setup_request_sha256"] == setup["request_sha256"],
                        f"{key}/{result['case_id']}: setup evidence binding mismatch: "
                        f"produced={produced['followup']} independent="
                        f"{setup['raw_response_sha256']}/{setup['clean_text_sha256']}/{setup['request_sha256']}")
        for population in ("development", "heldout", "all"):
            for kind in (None, *CLASSES):
                expected = aggregate(results, population, kind)
                produced = scorecard["arms"][key]["aggregate"][population]["overall" if kind is None else "classes"][kind] \
                    if kind is not None else scorecard["arms"][key]["aggregate"][population]["overall"]
                require(produced == expected, f"{key}: aggregate mismatch {population}/{kind}")
    for population in ("development", "heldout", "all"):
        before = all_results["development/before"] + (all_results["heldout/before"] if population == "all" else [])
        after = all_results["development/after"] + (all_results["heldout/after"] if population == "all" else [])
        if population == "development":
            before, after = all_results["development/before"], all_results["development/after"]
        elif population == "heldout":
            before, after = all_results["heldout/before"], all_results["heldout/after"]
        b = {r["case_id"]: r["routing_correct"] for r in before}
        a = {r["case_id"]: r["routing_correct"] for r in after}
        improved = sum(not b[i] and a[i] for i in b)
        worsened = sum(b[i] and not a[i] for i in b)
        n = improved + worsened
        p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(improved, worsened) + 1)) / 2 ** n)
        expected = {"cases": len(b), "improved": improved, "worsened": worsened, "discordant": n,
                    "two_sided_exact_p": p, "accuracy_delta_percentage_points": 100 * (sum(a.values()) - sum(b.values())) / len(b)}
        require(scorecard["paired"][population] == expected, f"paired statistic mismatch: {population}")
    before_all_results = all_results["development/before"] + all_results["heldout/before"]
    after_all_results = all_results["development/after"] + all_results["heldout/after"]
    if focused_followup:
        expected_history = followup_history_gate(before_all_results, after_all_results)
        require(scorecard["history_gate"] == expected_history,
                "focused follow-up history gate was not independently reproduced")
        require("thresholds" not in scorecard and scorecard["promotion"] == "COMPARATOR_REQUIRED",
                "focused follow-up scorecard must defer full routing promotion")
    else:
        before_all = aggregate(before_all_results, "all")
        after_all = aggregate(after_all_results, "all")
        before_heldout = aggregate(all_results["heldout/before"], "heldout")
        after_heldout = aggregate(all_results["heldout/after"], "heldout")
        gain = scorecard["paired"]["all"]["accuracy_delta_percentage_points"] > 0
        no_heldout_regression = after_heldout["routing_accuracy"] >= before_heldout["routing_accuracy"]
        owner_all = after_all["routing_accuracy"] >= 0.80 and after_all["redundant_visual_rate"] < 0.05
        owner_heldout = after_heldout["routing_accuracy"] >= 0.80 and after_heldout["redundant_visual_rate"] < 0.05
        expected_thresholds = {
            "owner": {"routing_accuracy_min": 0.80, "redundant_visual_rate_max_exclusive": 0.05,
                      "after_all_met": owner_all, "after_heldout_met": owner_heldout,
                      "positive_paired_gain": gain, "heldout_non_regression": no_heldout_regression,
                      "all_met": owner_all and owner_heldout and gain and no_heldout_regression},
            "rubric_hard_ideal": {"routing_accuracy_min": 0.90,
                                  "after_all_met": after_all["routing_accuracy"] >= 0.90,
                                  "after_heldout_met": after_heldout["routing_accuracy"] >= 0.90},
            "before_all": {"routing_accuracy": before_all["routing_accuracy"],
                           "redundant_visual_rate": before_all["redundant_visual_rate"]},
            "after_all": {"routing_accuracy": after_all["routing_accuracy"],
                          "redundant_visual_rate": after_all["redundant_visual_rate"]},
            "before_heldout": {"routing_accuracy": before_heldout["routing_accuracy"],
                               "redundant_visual_rate": before_heldout["redundant_visual_rate"]},
            "after_heldout": {"routing_accuracy": after_heldout["routing_accuracy"],
                              "redundant_visual_rate": after_heldout["redundant_visual_rate"]},
        }
        require(scorecard["thresholds"] == expected_thresholds, "promotion thresholds were not reproduced")
        expected_promotion = "PROMOTION_ELIGIBLE" if expected_thresholds["owner"]["all_met"] else "NOT_PROMOTED"
        require(scorecard["promotion"] == expected_promotion, "promotion verdict was not independently reproduced")
    expected_misses = []
    for key, results in all_results.items():
        for result in results:
            record = result["_record"]
            if (not result["end_to_end_correct"] or record["errors"] or record["leg_errors"] or
                    record["media_errors"] or result["redundant_visual_case"] or
                    (result["_case"]["class"] == "followup" and not result["followup_correct"])):
                expected_misses.append({"arm": key, "case_id": result["case_id"],
                                        "routing_correct": result["routing_correct"],
                                        "end_to_end_correct": result["end_to_end_correct"],
                                        "redundant_visual_case": result["redundant_visual_case"],
                                        "missing_visual_slots": result["missing_visual_slots"],
                                        "wrong_modality": result["wrong_modality"],
                                        "errors": record["errors"], "public_leg_errors": record["leg_errors"],
                                        "media_errors": record["media_errors"],
                                        "followup_correct": result.get("followup_correct")})
    require(sorted(map(canonical, scorecard["misses"])) == sorted(map(canonical, expected_misses)),
            "miss/error list was not independently reproduced")
    require(scorecard["physical_perceptual_quality"] == "NOT_RUN", "perceptual quality must remain NOT_RUN")
    return {"result": "PASS", "manifest_file_count": len(manifest["files"]),
            "manifest_sha256": sha_file(root / "manifest.json"), "scorecard_sha256": sha_file(root / manifest["scorecard"]),
            "raw_cases_verified": sum(len(v) for v in all_results.values()),
            "followup_setup_streams_verified": setup_streams_verified,
            "model_trees_rehashed": len(model_cache), "physical_perceptual_quality": "NOT_RUN"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.write_manifest:
        require(args.out is None, "--out is not used while writing a manifest")
        write_manifest(args.root)
        return
    result = verify(args.root, args.ffmpeg)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        resolved_root = Path(args.root).resolve()
        resolved_out = out.resolve()
        require(resolved_out != resolved_root and resolved_root not in resolved_out.parents,
                "verification report must be outside the complete frozen manifest root")
        out.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
