# SPDX-License-Identifier: Apache-2.0
"""Stage C Joint SFT & Preference (DPO/GRPO) Dataset Builder and Auditor.

Reads schema-v1 and schema-v2 trajectory JSONL exports (umminfer_rl_trajectory)
strictly read-only, verifies record and content-addressed media integrity, and
builds deterministic, content-addressed training packs:

- SFT Interleaved Modality-Routing Pack (--mode sft): filters only completed
  verified turns, reconstructs canonical director-supervised targets with
  ``[[image: ...]]`` / ``[[video: ...]]`` beat markers aligned to
  ``visual_slots`` (or clean prose for text-only turns), preserves verified
  input/output media SHA-256 references, and deduplicates by canonical
  ``(prompt_sha256, target_sha256)``.
- Preference / GRPO Group Pack (--mode preference): groups completed verified
  rollouts by canonical ``prompt_sha256``, scores each trajectory strictly from
  its exported JSON record via pluggable ``load_reward("module:attr")``, and
  emits deterministic pairwise DPO records (``margin >= --min-margin``) plus
  GRPO group records with group-relative normalized advantages.
- Deterministic Split & Leakage Auditor: assigns each prompt deterministically
  to ``train`` or ``eval`` via ``sha256(seed + prompt_sha256)``, verifies zero
  prompt overlap across splits (``leakage_count == 0``), and emits
  ``manifest.json`` + ``manifest.json.sha256`` with explicit
  ``REAL_TRAINING: NOT_RUN``.

Pure Python 3 stdlib only (zero torch/numpy, zero serving-code imports).
"""

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

# Some interpreters' -I mode omits the script directory from sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from umminfer_rl_reward import load_reward  # noqa: E402
from umminfer_rl_trajectory import (  # noqa: E402
    SUPPORTED_TRAJECTORY_SCHEMA_VERSIONS,
    TERMINAL_CANCELLED,
    TERMINAL_COMPLETED,
    TERMINAL_EVENT_FOR_STATE,
    TERMINAL_FAILED,
    TERMINAL_STATES,
    canonical_json,
    sha256_bytes,
    sha256_text,
)

DATASET_SCHEMA_VERSION = 1
DEFAULT_REWARD_SPEC = "umminfer_ablation_metrics:NarrativeRoutingReward"
DEFAULT_SEED = "umminfer-stage-c-v1"
DEFAULT_EVAL_RATIO = 0.2
DEFAULT_MIN_MARGIN = 1e-6
DEFAULT_MIN_GROUP_SIZE = 2
VALID_MODES = ("sft", "preference", "both")

_FINAL_LEG_STATES = frozenset(("completed", "failed", "aborted"))
_VISUAL_MODALITIES = frozenset(("image", "video"))
_HEX_256 = re.compile(r"[0-9a-f]{64}\Z")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _sha256(value):
    return isinstance(value, str) and _HEX_256.fullmatch(value) is not None


def _fields(event, names):
    return {name: event[name] for name in names if name in event}


def _slot_records(events, *, legacy=False):
    names = ["seq", "server_seq", "stage", "slot_id", "slot_index",
             "media_type", "prompt", "text_offset"]
    if not legacy:
        names.extend(("continuity_mode", "anchor_slot_index", "anchor_explicit"))
    return [_fields(event, names) for event in events
            if event.get("type") == "visual_slot"]


def _leg_error_records(events, *, legacy=False):
    names = ["seq", "server_seq", "stage", "error", "slot_id", "slot_index"]
    if not legacy:
        names.extend(("anchor_slot_index", "anchor_explicit", "anchor_sha256"))
    return [_fields(event, names) for event in events
            if event.get("type") == "leg_error"]


def _artifact_identity(descriptor):
    varying = {"index", "total", "data", "encoding", "bytes", "sha256",
               "chunk_bytes", "chunk_sha256", "chunks_sha256", "ref"}
    return {key: value for key, value in descriptor.items()
            if key not in varying}


def _expected_clean_text(record, terminal):
    events = record["events"]
    parts = [event.get("text", "") for event in events
             if event.get("type") in ("vlm_delta", "output_delta")]
    if parts:
        return "".join(parts)
    if terminal.get("type") == "complete" and not any(
            event.get("type") == "asr_final" for event in events):
        return terminal.get("text", "")
    if any(event.get("type") in ("asr_final", "asr_partial")
           for event in events):
        return ""
    return record["output_text"]


class _IntegrityError(ValueError):
    def __init__(self, category, message):
        super().__init__(message)
        self.category = category


def _verify_cas_ref(artifact_root, ref, expected_sha, expected_bytes, label,
                    *, check_disk):
    if not isinstance(ref, str) or not ref or Path(ref).name != ref:
        raise _IntegrityError(
            "integrity_error", f"{label} has an invalid relative artifact ref")
    if not _sha256(expected_sha) or not ref.startswith(expected_sha + "."):
        raise _IntegrityError(
            "integrity_error", f"{label} ref is not bound to its SHA-256")
    if not check_disk:
        return False
    if artifact_root is None or not Path(artifact_root).is_dir():
        raise _IntegrityError(
            "artifact_missing", f"{label} artifact directory is missing")
    path = Path(artifact_root) / ref
    if not path.is_file():
        raise _IntegrityError(
            "artifact_missing", f"{label} artifact is missing: {ref}")
    raw = path.read_bytes()
    if expected_bytes is not None and (
            not _integer(expected_bytes) or expected_bytes != len(raw)):
        raise _IntegrityError(
            "artifact_tampered", f"{label} artifact byte count mismatch: {ref}")
    if sha256_bytes(raw) != expected_sha:
        raise _IntegrityError(
            "artifact_tampered", f"{label} artifact SHA-256 mismatch: {ref}")
    return True


def _validate_inputs(record, artifact_root, *, check_disk, strict_v2):
    inputs = record.get("inputs")
    if not isinstance(inputs, list):
        if strict_v2 or inputs is not None:
            raise _IntegrityError("integrity_error", "missing inputs list")
        return 0
    checked_media = 0
    for index, ref in enumerate(inputs):
        if not isinstance(ref, dict) or not isinstance(ref.get("kind"), str):
            raise _IntegrityError("integrity_error", f"input {index} is malformed")
        if ref["kind"] == "text":
            if not isinstance(ref.get("text"), str) \
                    or ref.get("sha256") != sha256_text(ref["text"]):
                raise _IntegrityError(
                    "integrity_error",
                    f"input {index} text SHA-256 does not recompute")
            continue
        digest = ref.get("sha256")
        if digest is None:
            if not isinstance(ref.get("ref"), str) \
                    or ref["ref"].startswith("data:"):
                raise _IntegrityError(
                    "integrity_error",
                    f"input {index} has neither bytes nor a remote URL")
            continue
        if _verify_cas_ref(artifact_root, ref.get("ref"), digest,
                           ref.get("bytes"), f"input {index}",
                           check_disk=check_disk):
            checked_media += 1
    return checked_media


def _validate_request_params(record, artifact_root, *, check_disk):
    params = record.get("request_params")
    if not isinstance(params, dict):
        raise _IntegrityError("integrity_error", "request_params must be an object")
    inputs = record["inputs"]
    for section, field in (("image", "image"), ("video", "image"),
                           ("audio", "ref_audio")):
        options = params.get(section)
        if not isinstance(options, dict) or field not in options:
            continue
        ref = options[field]
        if not isinstance(ref, dict):
            raise _IntegrityError(
                "integrity_error",
                f"request_params.{section}.{field} embeds media")
        if ref not in inputs:
            raise _IntegrityError(
                "integrity_error",
                f"request_params.{section}.{field} lacks an input reference")
        _verify_cas_ref(artifact_root, ref.get("ref"), ref.get("sha256"),
                        ref.get("bytes"), f"request_params.{section}.{field}",
                        check_disk=check_disk)


def _validate_terminal(record):
    events = record["events"]
    state = record["terminal_state"]
    terminal_indexes = [
        index for index, event in enumerate(events)
        if event.get("type") in ("complete", "error", "cancelled")
    ]
    if len(terminal_indexes) != 1:
        raise _IntegrityError(
            "integrity_error", "v2 requires exactly one terminal event")
    terminal_index = terminal_indexes[0]
    if terminal_index != len(events) - 1:
        raise _IntegrityError(
            "integrity_error", "terminal event is not final")
    if record.get("terminal_event_seq") != terminal_index:
        raise _IntegrityError(
            "integrity_error", "terminal_event_seq does not identify terminal")
    terminal = events[terminal_index]
    if terminal.get("type") != TERMINAL_EVENT_FOR_STATE[state]:
        raise _IntegrityError(
            "integrity_error", "terminal_state does not match terminal event")
    error = record.get("error")
    if state in (TERMINAL_COMPLETED, TERMINAL_CANCELLED) and error is not None:
        raise _IntegrityError(
            "integrity_error", f"{state} record carries an error")
    if state == TERMINAL_FAILED and (not isinstance(error, str) or not error):
        raise _IntegrityError(
            "integrity_error", "failed record lacks an error message")

    legs = terminal.get("legs", [])
    if not isinstance(legs, list) or record.get("terminal_legs") != legs:
        raise _IntegrityError(
            "integrity_error", "terminal_legs differs from terminal event")
    for leg in legs:
        if not isinstance(leg, dict) or not isinstance(leg.get("stage"), str) \
                or leg.get("state") not in _FINAL_LEG_STATES:
            raise _IntegrityError(
                "integrity_error",
                "terminal leg has an invalid stage or non-final state")
    return terminal


def _validate_slots(record):
    events = record["events"]
    slots = record.get("visual_slots")
    if slots != _slot_records(events) \
            and slots != _slot_records(events, legacy=True):
        raise _IntegrityError(
            "integrity_error", "visual_slots differs from visual_slot events")
    by_id = {}
    for expected_index, slot in enumerate(slots):
        slot_id = slot.get("slot_id")
        if not isinstance(slot_id, str) or not slot_id \
                or slot_id in by_id or slot.get("slot_index") != expected_index:
            raise _IntegrityError(
                "integrity_error",
                "visual slot ids/indices are not unique and gap-free")
        if slot.get("media_type") not in _VISUAL_MODALITIES \
                or slot.get("stage") != slot["media_type"]:
            raise _IntegrityError(
                "integrity_error",
                f"visual slot stage/media mismatch for {slot_id}")
        if not isinstance(slot.get("prompt"), str):
            raise _IntegrityError(
                "integrity_error", f"visual slot prompt is not text for {slot_id}")
        if not _integer(slot.get("text_offset")) or slot["text_offset"] < 0:
            raise _IntegrityError(
                "integrity_error",
                f"visual slot text_offset is invalid for {slot_id}")
        by_id[slot_id] = slot
        matches = [leg for leg in record["terminal_legs"]
                   if leg.get("slot_id") == slot_id]
        if len(matches) != 1 or matches[0]["stage"] != slot["stage"]:
            raise _IntegrityError(
                "integrity_error",
                f"visual slot lacks one matching terminal leg: {slot_id}")
        if not slot["prompt"].strip() and not (
                slot.get("anchor_explicit", False)
                and matches[0]["state"] in ("failed", "aborted")):
            raise _IntegrityError(
                "integrity_error", f"visual slot prompt is empty for {slot_id}")
    if record.get("leg_errors") != _leg_error_records(events) \
            and record.get("leg_errors") != _leg_error_records(events, legacy=True):
        raise _IntegrityError(
            "integrity_error", "leg_errors differs from leg_error events")
    return by_id


def _validate_artifacts(record, artifact_root, slots, *, check_disk):
    groups = {}
    for event in record["events"]:
        if event.get("type") != "artifact":
            continue
        descriptor = event.get("artifact")
        if not isinstance(descriptor, dict) or "data" in descriptor:
            raise _IntegrityError(
                "integrity_error",
                "artifact event is malformed or embeds media bytes")
        artifact_id = descriptor.get("artifact_id")
        index, total = descriptor.get("index"), descriptor.get("total")
        if not isinstance(artifact_id, str) or not artifact_id \
                or not _integer(index) or not _integer(total) or total <= 0 \
                or not 0 <= index < total:
            raise _IntegrityError(
                "integrity_error", "artifact chunk identity/index is invalid")
        if not _integer(descriptor.get("chunk_bytes")) \
                or descriptor["chunk_bytes"] < 0 \
                or not _sha256(descriptor.get("chunk_sha256")):
            raise _IntegrityError(
                "integrity_error",
                f"artifact chunk hash/bytes are invalid: {artifact_id}")
        final = index + 1 == total
        if final and (not _integer(descriptor.get("bytes"))
                      or not _sha256(descriptor.get("sha256"))):
            raise _IntegrityError(
                "integrity_error",
                f"final artifact chunk lacks bytes/SHA-256: {artifact_id}")
        if not final and ("bytes" in descriptor or "sha256" in descriptor):
            raise _IntegrityError(
                "integrity_error",
                f"non-final artifact chunk carries final hash: {artifact_id}")
        group = groups.setdefault(artifact_id, {
            "total": total, "descriptors": {},
            "identity": _artifact_identity(descriptor),
        })
        if group["total"] != total \
                or group["identity"] != _artifact_identity(descriptor):
            raise _IntegrityError(
                "integrity_error",
                f"artifact metadata/total changed: {artifact_id}")
        if index in group["descriptors"]:
            raise _IntegrityError(
                "integrity_error", f"duplicate artifact chunk: {artifact_id}")
        group["descriptors"][index] = descriptor

        slot_id = descriptor.get("slot_id")
        if slot_id is not None:
            if slot_id not in slots:
                raise _IntegrityError(
                    "integrity_error",
                    f"artifact binds to an unknown slot: {slot_id}")
            slot = slots[slot_id]
            if descriptor.get("slot_index") != slot["slot_index"] \
                    or descriptor.get("media_type") != slot["media_type"] \
                    or event.get("stage") != slot["stage"]:
                raise _IntegrityError(
                    "integrity_error",
                    f"artifact does not match its slot: {slot_id}")
        elif slots and descriptor.get("media_type") in _VISUAL_MODALITIES:
            raise _IntegrityError(
                "integrity_error",
                "narrative visual artifact lacks slot attribution")

    outputs = record.get("artifacts")
    if not isinstance(outputs, list):
        raise _IntegrityError("integrity_error", "missing artifacts list")
    output_by_id = {}
    for output in outputs:
        if not isinstance(output, dict) or not isinstance(
                output.get("artifact_id"), str) \
                or output["artifact_id"] in output_by_id:
            raise _IntegrityError(
                "integrity_error", "output artifact ids are missing or duplicated")
        output_by_id[output["artifact_id"]] = output

    complete_ids = []
    checked_outputs = 0
    for artifact_id, group in groups.items():
        total = group["total"]
        descriptors = group["descriptors"]
        complete = sorted(descriptors) == list(range(total))
        if not complete:
            if record["terminal_state"] == TERMINAL_COMPLETED:
                raise _IntegrityError(
                    "integrity_error",
                    f"completed turn has missing chunks: {artifact_id}")
            if artifact_id in output_by_id:
                raise _IntegrityError(
                    "integrity_error",
                    f"partial artifact was published: {artifact_id}")
            continue
        complete_ids.append(artifact_id)
        output = output_by_id.get(artifact_id)
        if output is None:
            raise _IntegrityError(
                "integrity_error",
                f"complete artifact has no output descriptor: {artifact_id}")
        final = descriptors[total - 1]
        if output.get("chunk_count") != total \
                or output.get("bytes") != final.get("bytes") \
                or output.get("sha256") != final.get("sha256"):
            raise _IntegrityError(
                "integrity_error",
                f"output/final chunk descriptor mismatch: {artifact_id}")
        if sum(item["chunk_bytes"] for item in descriptors.values()) \
                != final["bytes"]:
            raise _IntegrityError(
                "integrity_error",
                f"artifact chunk bytes do not sum to output: {artifact_id}")
        chunk_hashes = b"".join(
            bytes.fromhex(descriptors[index]["chunk_sha256"])
            for index in range(total))
        chunks_sha256 = sha256_bytes(chunk_hashes)
        if output.get("chunks_sha256") != chunks_sha256:
            raise _IntegrityError(
                "integrity_error",
                f"artifact chunk hash chain mismatch: {artifact_id}")
        for key, value in group["identity"].items():
            if output.get(key) != value:
                raise _IntegrityError(
                    "integrity_error",
                    f"output metadata mismatch {key}: {artifact_id}")
        for descriptor in descriptors.values():
            if descriptor.get("ref") != output.get("ref"):
                raise _IntegrityError(
                    "integrity_error",
                    f"artifact chunk ref mismatch: {artifact_id}")
        if _verify_cas_ref(artifact_root, output.get("ref"),
                           output.get("sha256"), output.get("bytes"),
                           f"output {artifact_id}", check_disk=check_disk):
            checked_outputs += 1

    if set(output_by_id) != set(complete_ids) \
            or [item["artifact_id"] for item in outputs] != complete_ids:
        raise _IntegrityError(
            "integrity_error",
            "artifacts list does not match complete event groups")
    return checked_outputs


def _validate_linkages(record, slots):
    outputs_by_slot = {}
    for output in record["artifacts"]:
        slot_id = output.get("slot_id")
        if slot_id is not None:
            outputs_by_slot.setdefault(slot_id, []).append(output)
    legs = {leg.get("slot_id"): leg for leg in record["terminal_legs"]
            if leg.get("slot_id") is not None}
    by_index = {slot["slot_index"]: slot for slot in slots.values()}
    for slot_id, slot in slots.items():
        leg = legs[slot_id]
        outputs = outputs_by_slot.get(slot_id, [])
        anchor = slot.get("anchor_slot_index")
        explicit = slot.get("anchor_explicit", False)
        if not isinstance(explicit, bool):
            raise _IntegrityError(
                "integrity_error", f"anchor_explicit is not boolean: {slot_id}")
        if slot.get("continuity_mode") not in (None, "prompt_anchor", "i2v_chain"):
            raise _IntegrityError(
                "integrity_error", f"invalid continuity_mode: {slot_id}")
        if anchor is None:
            if explicit:
                if leg["state"] == "completed" or outputs \
                        or "anchor_slot_index" in leg or "anchor_sha256" in leg:
                    raise _IntegrityError(
                        "integrity_error", f"unresolved explicit has output: {slot_id}")
                continue
            if "anchor_slot_index" in leg or "anchor_sha256" in leg \
                    or any("anchor_slot_index" in output
                           or "anchor_sha256" in output for output in outputs):
                raise _IntegrityError(
                    "integrity_error", f"unanchored slot has metadata: {slot_id}")
            continue
        if not _integer(anchor) or anchor < 0 \
                or slot.get("continuity_mode") != "i2v_chain":
            raise _IntegrityError(
                "integrity_error", f"invalid visual anchor: {slot_id}")
        if explicit or "anchor_slot_index" in leg or "anchor_sha256" in leg:
            if leg.get("anchor_slot_index") != anchor \
                    or leg.get("anchor_explicit", False) != explicit:
                raise _IntegrityError(
                    "integrity_error", f"visual/terminal anchor mismatch: {slot_id}")
        digest = leg.get("anchor_sha256")
        if digest is not None and not _sha256(digest):
            raise _IntegrityError(
                "integrity_error", f"invalid anchor SHA-256: {slot_id}")
        if explicit and leg["state"] == "completed" and not digest:
            raise _IntegrityError(
                "integrity_error", f"completed explicit I2V lacks digest: {slot_id}")
        if explicit and leg["state"] == "completed" and not outputs:
            raise _IntegrityError(
                "integrity_error", f"completed explicit I2V has no artifact: {slot_id}")
        for output in outputs:
            if output.get("media_type") != "video":
                raise _IntegrityError(
                    "integrity_error", f"invalid anchored modality: {slot_id}")
            if explicit or "anchor_slot_index" in output \
                    or "anchor_sha256" in output:
                if output.get("anchor_slot_index") != anchor \
                        or output.get("anchor_explicit", False) != explicit:
                    raise _IntegrityError(
                        "integrity_error", f"invalid anchored artifact: {slot_id}")
            output_digest = output.get("anchor_sha256")
            if output_digest is not None and not _sha256(output_digest):
                raise _IntegrityError(
                    "integrity_error", f"invalid artifact anchor digest: {slot_id}")
            if explicit and output_digest is None:
                raise _IntegrityError(
                    "integrity_error", f"explicit I2V artifact lacks digest: {slot_id}")
            if digest is not None and output_digest != digest:
                raise _IntegrityError(
                    "integrity_error", f"artifact/terminal anchor mismatch: {slot_id}")
        if digest is None and not any("anchor_sha256" in output
                                      for output in outputs):
            if explicit and leg["state"] == "completed":
                raise _IntegrityError(
                    "integrity_error", f"completed explicit I2V has no artifact: {slot_id}")
            continue
        resolved = digest if digest is not None else outputs[0]["anchor_sha256"]
        source = by_index.get(anchor)
        if source is None or anchor >= slot["slot_index"] \
                or source["media_type"] != "image" \
                or slot["media_type"] != "video":
            raise _IntegrityError(
                "integrity_error", f"anchor is not an earlier image: {slot_id}")
        source_outputs = [item for item in outputs_by_slot.get(
                              source["slot_id"], [])
                          if item.get("media_type") == "image"]
        if len(source_outputs) != 1 \
                or source_outputs[0].get("sha256") != resolved:
            raise _IntegrityError(
                "integrity_error", f"anchor CAS artifact mismatch: {slot_id}")


def verify_record_integrity(record, *, artifact_dir=None,
                            require_artifact_dir=False):
    """Verify one schema-v1 or schema-v2 trajectory record read-only.

    Returns ``(ok, category, message, stats)`` without mutating ``record``.
    When ``artifact_dir`` exists (or ``require_artifact_dir`` is True), all
    content-addressed input/request/output media files are re-hashed on disk.
    """
    stats = {
        "checked_disk": False,
        "checked_input_media_refs": 0,
        "checked_output_media_refs": 0,
    }
    try:
        if not isinstance(record, dict):
            raise _IntegrityError("integrity_error", "record must be an object")
        version = record.get("schema_version")
        if not _integer(version) \
                or version not in SUPPORTED_TRAJECTORY_SCHEMA_VERSIONS:
            raise _IntegrityError(
                "integrity_error", "unsupported or missing schema_version")
        if record.get("terminal_state") not in TERMINAL_STATES:
            raise _IntegrityError("integrity_error", "invalid terminal_state")
        events = record.get("events")
        if not isinstance(events, list):
            raise _IntegrityError("integrity_error", "missing events list")
        for event_index, event in enumerate(events):
            if not isinstance(event, dict) \
                    or not _integer(event.get("seq")) \
                    or event["seq"] != event_index:
                raise _IntegrityError(
                    "integrity_error", f"event seq gap at {event_index}")
        output_text = record.get("output_text")
        if not isinstance(output_text, str) \
                or record.get("output_sha256") != sha256_text(output_text):
            raise _IntegrityError(
                "integrity_error", "output_sha256 does not recompute")

        artifact_root = Path(artifact_dir) if artifact_dir is not None else None
        check_disk = bool(
            require_artifact_dir
            or (artifact_root is not None and artifact_root.is_dir())
        )
        stats["checked_disk"] = check_disk

        if version == 1:
            stats["checked_input_media_refs"] = _validate_inputs(
                record, artifact_root, check_disk=check_disk, strict_v2=False)
            return True, None, None, stats

        server_seqs = [event.get("server_seq") for event in events
                       if event.get("server_seq") is not None]
        if any(not _integer(seq) for seq in server_seqs) \
                or server_seqs != list(range(len(server_seqs))):
            raise _IntegrityError(
                "integrity_error", "server_seq is not gap-free from zero")
        terminal = _validate_terminal(record)
        clean_text = record.get("clean_text")
        if not isinstance(clean_text, str) \
                or record.get("clean_text_sha256") != sha256_text(clean_text):
            raise _IntegrityError(
                "integrity_error", "clean_text_sha256 does not recompute")
        if clean_text != _expected_clean_text(record, terminal):
            raise _IntegrityError(
                "integrity_error", "clean_text does not match narrative deltas")
        output_parts = [
            event.get("text", "") for event in events
            if event.get("type") in ("asr_final", "vlm_delta", "output_delta")
        ]
        expected_output = "".join(output_parts) if output_parts else (
            terminal.get("text", "") if terminal.get("type") == "complete"
            else "")
        if output_text != expected_output:
            raise _IntegrityError(
                "integrity_error", "output_text does not match output events")
        stats["checked_input_media_refs"] = _validate_inputs(
            record, artifact_root, check_disk=check_disk, strict_v2=True)
        _validate_request_params(record, artifact_root, check_disk=check_disk)
        slots = _validate_slots(record)
        stats["checked_output_media_refs"] = _validate_artifacts(
            record, artifact_root, slots, check_disk=check_disk)
        _validate_linkages(record, slots)
        return True, None, None, stats
    except _IntegrityError as exc:
        return False, exc.category, str(exc), stats


def _canonical_slot_prompt(prompt):
    return " ".join(prompt.split())


def format_beat_marker(media_type, prompt):
    """Format one canonical visual beat marker matching NarrativeStreamParser."""
    if media_type not in _VISUAL_MODALITIES:
        raise ValueError(f"unsupported visual slot modality: {media_type!r}")
    normalized = _canonical_slot_prompt(prompt)
    if not normalized:
        raise ValueError("visual slot prompt must not be empty")
    return f"[[{media_type}: {normalized}]]"


def _snap_utf8_offset(raw_bytes, offset):
    length = len(raw_bytes)
    clamped = max(0, min(offset, length))
    while clamped < length and (raw_bytes[clamped] & 0xC0) == 0x80:
        clamped += 1
    return clamped


def reconstruct_director_target(record, *, verified_on_disk=False):
    """Reconstruct the director-supervised assistant target from a verified turn.

    For schema-v2 records with ``visual_slots``, inserts canonical
    ``[[image: ...]]`` / ``[[video: ...]]`` beat markers at each slot's
    clean-prose UTF-8 byte offset (exact round-trip with
    ``NarrativeStreamParser``). For text-only v2 turns and v1 turns, returns
    clean prose with no beat markers.
    """
    version = record["schema_version"]
    if version == 1:
        clean_text = record["output_text"]
        return clean_text, clean_text, [], []

    clean_text = record["clean_text"]
    slots = record.get("visual_slots") or []
    artifacts = record.get("artifacts") or []
    terminal_legs = record.get("terminal_legs") or []

    leg_state_by_slot = {
        leg["slot_id"]: leg.get("state")
        for leg in terminal_legs
        if isinstance(leg, dict) and leg.get("slot_id") is not None
    }
    artifact_by_slot = {}
    output_media_refs = []
    for item in artifacts:
        ref_entry = {
            "artifact_id": item["artifact_id"],
            "media_type": item.get("media_type"),
            "mime_type": item.get("mime_type"),
            "ref": item.get("ref"),
            "sha256": item.get("sha256"),
            "bytes": item.get("bytes"),
            "chunk_count": item.get("chunk_count"),
            "chunks_sha256": item.get("chunks_sha256"),
            "verified_on_disk": bool(verified_on_disk),
        }
        if item.get("slot_id") is not None:
            ref_entry["slot_id"] = item["slot_id"]
            ref_entry["slot_index"] = item.get("slot_index")
            for key in ("anchor_slot_index", "anchor_explicit", "anchor_sha256"):
                if key in item:
                    ref_entry[key] = item[key]
            artifact_by_slot[item["slot_id"]] = ref_entry
        output_media_refs.append(ref_entry)

    if not slots:
        return clean_text, clean_text, [], output_media_refs

    clean_bytes = clean_text.encode("utf-8")
    ordered_slots = sorted(
        slots,
        key=lambda slot: (_snap_utf8_offset(clean_bytes, slot["text_offset"]),
                          slot["slot_index"]),
    )

    pieces = []
    annotated_slots = []
    cursor = 0
    for slot in ordered_slots:
        offset = _snap_utf8_offset(clean_bytes, slot["text_offset"])
        if offset > cursor:
            pieces.append(clean_bytes[cursor:offset].decode("utf-8"))
            cursor = offset
        prompt = _canonical_slot_prompt(slot["prompt"])
        leg_state = leg_state_by_slot.get(slot["slot_id"])
        marker = ""
        if prompt or leg_state == "completed":
            marker = format_beat_marker(slot["media_type"], prompt)
        bound = artifact_by_slot.get(slot["slot_id"])
        if bound is not None and bound.get("anchor_explicit", False):
            if slot["media_type"] != "video" \
                    or bound.get("anchor_slot_index") != slot.get("anchor_slot_index") \
                    or not _sha256(bound.get("anchor_sha256")):
                raise ValueError(f"invalid explicit anchor marker for {slot['slot_id']}")
            marker = marker[:-2] + (
                f" | source=slot-{bound['anchor_slot_index']}]]")
        pieces.append(marker)
        annotated_slots.append({
            "slot_id": slot["slot_id"],
            "slot_index": slot["slot_index"],
            "media_type": slot["media_type"],
            "prompt": prompt,
            "text_offset": slot["text_offset"],
            "marker": marker,
            "leg_state": leg_state_by_slot.get(slot["slot_id"]),
            "artifact_id": bound["artifact_id"] if bound else None,
            "artifact_ref": bound["ref"] if bound else None,
            "artifact_sha256": bound["sha256"] if bound else None,
        })
        for key in ("continuity_mode", "anchor_slot_index", "anchor_explicit"):
            if key in slot:
                annotated_slots[-1][key] = slot[key]
        if bound is not None and "anchor_sha256" in bound:
            annotated_slots[-1]["anchor_sha256"] = bound["anchor_sha256"]
    if cursor < len(clean_bytes):
        pieces.append(clean_bytes[cursor:].decode("utf-8"))

    assistant_target = "".join(pieces)
    return assistant_target, clean_text, annotated_slots, output_media_refs


def canonicalize_prompt(record, *, verified_on_disk=False):
    """Extract canonical prompt parts, media refs, and ``prompt_sha256``."""
    raw_inputs = record.get("inputs") or []
    prompt_parts = []
    input_media_refs = []
    text_chunks = []

    if raw_inputs:
        for item in raw_inputs:
            kind = item["kind"]
            if kind == "text":
                text = item["text"]
                text_chunks.append(text)
                prompt_parts.append({
                    "kind": "text",
                    "text": text,
                    "sha256": item["sha256"],
                })
            else:
                part = {
                    "kind": kind,
                    "media_type": item.get("media_type"),
                    "ref": item.get("ref"),
                    "sha256": item.get("sha256"),
                }
                if "bytes" in item:
                    part["bytes"] = item["bytes"]
                if "source" in item:
                    part["source"] = item["source"]
                prompt_parts.append(part)
                media_ref = dict(part)
                media_ref["verified_on_disk"] = bool(
                    verified_on_disk and item.get("sha256") is not None)
                input_media_refs.append(media_ref)
    elif isinstance(record.get("messages"), list):
        for message in record["messages"]:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str):
                text_chunks.append(content)
                prompt_parts.append({
                    "kind": "text",
                    "role": message.get("role", "user"),
                    "text": content,
                    "sha256": sha256_text(content),
                })

    prompt_sha256 = sha256_bytes(canonical_json(prompt_parts))
    prompt_text = "\n".join(text_chunks)
    return {
        "prompt_sha256": prompt_sha256,
        "prompt_text": prompt_text,
        "prompt_parts": prompt_parts,
        "input_media_refs": input_media_refs,
    }


def assign_split(prompt_sha256, *, seed=DEFAULT_SEED,
                 eval_ratio=DEFAULT_EVAL_RATIO):
    """Deterministically map a prompt hash to ``train`` or ``eval``.

    Uses ``sha256(str(seed) + prompt_sha256)`` so identical prompts always land
    in the same split with zero cross-split prompt leakage.
    """
    if not _sha256(prompt_sha256):
        raise ValueError(f"invalid prompt_sha256: {prompt_sha256!r}")
    if not _finite_number(eval_ratio) or not 0.0 <= float(eval_ratio) <= 1.0:
        raise ValueError("eval_ratio must be in [0.0, 1.0]")
    ratio = float(eval_ratio)
    if ratio <= 0.0:
        return "train"
    if ratio >= 1.0:
        return "eval"
    digest = hashlib.sha256(
        (str(seed) + prompt_sha256).encode("utf-8")).hexdigest()
    bucket = int(digest[:16], 16) / float(1 << 64)
    return "eval" if bucket < ratio else "train"


def compute_grpo_advantages(scores):
    """Compute group-relative normalized advantages ``(r - mean) / std``.

    Zero-variance groups receive exact ``0.0`` advantages (never NaN/Inf).
    """
    if not scores:
        return [], 0.0, 0.0
    values = [float(score) for score in scores]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std == 0.0:
        return [0.0 for _ in values], mean, 0.0
    return [(value - mean) / std for value in values], mean, std


def audit_split_leakage(train_records, eval_records):
    """Audit train/eval record collections for prompt-level leakage."""
    train_prompts = {record["prompt_sha256"] for record in train_records}
    eval_prompts = {record["prompt_sha256"] for record in eval_records}
    overlap = sorted(train_prompts & eval_prompts)
    return {
        "train_prompt_count": len(train_prompts),
        "eval_prompt_count": len(eval_prompts),
        "overlap_prompt_sha256s": overlap,
        "leakage_count": len(overlap),
        "passed": len(overlap) == 0,
    }


def _provenance_block(record):
    ablation = record.get("ablation")
    source_prompt_id = record.get("prompt_id")
    if source_prompt_id is None and isinstance(ablation, dict):
        source_prompt_id = ablation.get("prompt_id")
    return {
        "source_schema_version": record["schema_version"],
        "source_session_id": record.get("session_id"),
        "source_request_id": record.get("request_id"),
        "source_turn_index": record.get("turn_index"),
        "source_endpoint": record.get("endpoint"),
        "source_prompt_id": source_prompt_id,
    }


def _candidate_tie_key(candidate):
    prov = candidate["provenance"]
    return (
        candidate["prompt_sha256"],
        candidate["target_sha256"],
        str(prov.get("source_session_id") or ""),
        str(prov.get("source_request_id") or ""),
        prov.get("source_turn_index") if _integer(prov.get("source_turn_index"))
        else -1,
    )


def _write_jsonl_atomic(path, records):
    payload = b"".join(canonical_json(item) + b"\n" for item in records)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)
    return {
        "path": path.name,
        "sha256": sha256_bytes(payload),
        "bytes": len(payload),
        "record_count": len(records),
    }


def _write_manifest_atomic(out_dir, manifest):
    manifest_path = Path(out_dir) / "manifest.json"
    sidecar_path = Path(out_dir) / "manifest.json.sha256"
    payload = canonical_json(manifest) + b"\n"
    tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, manifest_path)
    digest = sha256_bytes(payload)
    tmp_sidecar = sidecar_path.with_name(sidecar_path.name + ".tmp")
    tmp_sidecar.write_text(digest + "\n", encoding="utf-8")
    os.replace(tmp_sidecar, sidecar_path)
    return manifest_path, sidecar_path, digest


def _prepare_out_dir(out_dir, *, overwrite):
    out_path = Path(out_dir)
    if out_path.exists():
        if not out_path.is_dir():
            raise ValueError("output path exists and is not a directory")
        if any(out_path.iterdir()) and not overwrite:
            raise ValueError(
                "output directory must be absent or empty (or pass --overwrite)")
    else:
        out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def _extract_candidate(record, *, verified_on_disk):
    prompt_info = canonicalize_prompt(record, verified_on_disk=verified_on_disk)
    assistant_target, clean_text, annotated_slots, output_media_refs = (
        reconstruct_director_target(record, verified_on_disk=verified_on_disk)
    )
    target_sha256 = sha256_text(assistant_target)
    return {
        "prompt_sha256": prompt_info["prompt_sha256"],
        "prompt_text": prompt_info["prompt_text"],
        "prompt_parts": prompt_info["prompt_parts"],
        "input_media_refs": prompt_info["input_media_refs"],
        "target_sha256": target_sha256,
        "assistant_target": assistant_target,
        "clean_text": clean_text,
        "clean_text_sha256": sha256_text(clean_text),
        "visual_slots": annotated_slots,
        "modality_route": [slot["media_type"] for slot in annotated_slots],
        "is_text_only": len(annotated_slots) == 0,
        "output_media_refs": output_media_refs,
        "provenance": _provenance_block(record),
        "_raw_record": record,
    }


def _build_sft_records(candidates, *, seed, eval_ratio):
    ordered = sorted(candidates, key=_candidate_tie_key)
    seen = set()
    duplicates = 0
    train_records = []
    eval_records = []

    for candidate in ordered:
        key = (candidate["prompt_sha256"], candidate["target_sha256"])
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        split = assign_split(candidate["prompt_sha256"],
                             seed=seed, eval_ratio=eval_ratio)
        example_id = sha256_text(
            f"{candidate['prompt_sha256']}:{candidate['target_sha256']}")[:32]
        entry = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "record_type": "sft_interleaved",
            "example_id": example_id,
            "split": split,
            "prompt_sha256": candidate["prompt_sha256"],
            "target_sha256": candidate["target_sha256"],
            "prompt_text": candidate["prompt_text"],
            "prompt_parts": candidate["prompt_parts"],
            "clean_text": candidate["clean_text"],
            "clean_text_sha256": candidate["clean_text_sha256"],
            "assistant_target": candidate["assistant_target"],
            "messages": [
                {"role": "user", "content": candidate["prompt_text"]},
                {"role": "assistant", "content": candidate["assistant_target"]},
            ],
            "visual_slots": candidate["visual_slots"],
            "modality_route": candidate["modality_route"],
            "is_text_only": candidate["is_text_only"],
            "input_media_refs": candidate["input_media_refs"],
            "output_media_refs": candidate["output_media_refs"],
            "provenance": candidate["provenance"],
        }
        if split == "eval":
            eval_records.append(entry)
        else:
            train_records.append(entry)

    train_records.sort(
        key=lambda item: (item["prompt_sha256"], item["target_sha256"],
                          item["example_id"]))
    eval_records.sort(
        key=lambda item: (item["prompt_sha256"], item["target_sha256"],
                          item["example_id"]))
    return train_records, eval_records, duplicates


def _response_summary(candidate, score, *, rank=None, advantage=None):
    summary = {
        "target_sha256": candidate["target_sha256"],
        "assistant_target": candidate["assistant_target"],
        "clean_text": candidate["clean_text"],
        "visual_slots": candidate["visual_slots"],
        "modality_route": candidate["modality_route"],
        "output_media_refs": candidate["output_media_refs"],
        "score": float(score),
        "provenance": candidate["provenance"],
    }
    if rank is not None:
        summary["rank"] = rank
    if advantage is not None:
        summary["advantage"] = float(advantage)
    return summary


def _build_preference_records(candidates, *, reward, seed, eval_ratio,
                              min_margin, min_group_size):
    ordered = sorted(candidates, key=_candidate_tie_key)
    seen_rollouts = set()
    duplicates = 0
    unscored = 0
    invalid_reward = 0
    by_prompt = {}

    for candidate in ordered:
        rollout_key = _candidate_tie_key(candidate)
        if rollout_key in seen_rollouts:
            duplicates += 1
            continue
        seen_rollouts.add(rollout_key)

        raw_copy = copy.deepcopy(candidate["_raw_record"])
        score = reward.score(raw_copy)
        if score is None:
            unscored += 1
            continue
        if not _finite_number(score):
            invalid_reward += 1
            continue
        by_prompt.setdefault(candidate["prompt_sha256"], []).append(
            (candidate, float(score)))

    dpo_train, dpo_eval = [], []
    grpo_train, grpo_eval = [], []
    skipped_small_groups = 0
    skipped_low_margin_pairs = 0

    for prompt_sha256 in sorted(by_prompt):
        group = by_prompt[prompt_sha256]
        if len(group) < min_group_size:
            skipped_small_groups += 1
            continue
        group.sort(key=lambda item: (-item[1], _candidate_tie_key(item[0])))
        split = assign_split(prompt_sha256, seed=seed, eval_ratio=eval_ratio)
        lead = group[0][0]

        scores = [item[1] for item in group]
        advantages, group_mean, group_std = compute_grpo_advantages(scores)
        rollout_entries = [
            _response_summary(cand, score, rank=rank, advantage=adv)
            for rank, ((cand, score), adv) in enumerate(zip(group, advantages))
        ]
        group_id = sha256_text(
            prompt_sha256 + ":" + reward.name + ":"
            + ",".join(item["target_sha256"] for item in rollout_entries))[:32]
        grpo_entry = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "record_type": "grpo_group",
            "group_id": group_id,
            "split": split,
            "prompt_sha256": prompt_sha256,
            "prompt_text": lead["prompt_text"],
            "prompt_parts": lead["prompt_parts"],
            "input_media_refs": lead["input_media_refs"],
            "reward_name": reward.name,
            "group_size": len(rollout_entries),
            "group_mean": group_mean,
            "group_std": group_std,
            "has_reward_variance": group_std > 0.0,
            "rollouts": rollout_entries,
        }
        if split == "eval":
            grpo_eval.append(grpo_entry)
        else:
            grpo_train.append(grpo_entry)

        # Deduplicate targets for pairwise DPO (keep highest-scoring rollout
        # per distinct target_sha256, deterministically ordered).
        distinct_targets = []
        seen_targets = set()
        for cand, score in group:
            if cand["target_sha256"] in seen_targets:
                continue
            seen_targets.add(cand["target_sha256"])
            distinct_targets.append((cand, score))

        for i in range(len(distinct_targets)):
            chosen_cand, chosen_score = distinct_targets[i]
            for j in range(i + 1, len(distinct_targets)):
                rejected_cand, rejected_score = distinct_targets[j]
                margin = chosen_score - rejected_score
                if chosen_score <= rejected_score \
                        or margin + 1e-12 < min_margin:
                    skipped_low_margin_pairs += 1
                    continue
                pair_id = sha256_text(
                    f"{prompt_sha256}:{reward.name}:"
                    f"{chosen_cand['target_sha256']}:"
                    f"{rejected_cand['target_sha256']}")[:32]
                dpo_entry = {
                    "schema_version": DATASET_SCHEMA_VERSION,
                    "record_type": "dpo_pair",
                    "pair_id": pair_id,
                    "split": split,
                    "prompt_sha256": prompt_sha256,
                    "prompt_text": lead["prompt_text"],
                    "prompt_parts": lead["prompt_parts"],
                    "input_media_refs": lead["input_media_refs"],
                    "reward_name": reward.name,
                    "margin": float(margin),
                    "chosen": _response_summary(chosen_cand, chosen_score),
                    "rejected": _response_summary(rejected_cand, rejected_score),
                }
                if split == "eval":
                    dpo_eval.append(dpo_entry)
                else:
                    dpo_train.append(dpo_entry)

    pair_sort_key = lambda item: (
        item["prompt_sha256"],
        -item["margin"],
        item["chosen"]["target_sha256"],
        item["rejected"]["target_sha256"],
        item["pair_id"],
    )
    dpo_train.sort(key=pair_sort_key)
    dpo_eval.sort(key=pair_sort_key)
    grpo_train.sort(key=lambda item: (item["prompt_sha256"], item["group_id"]))
    grpo_eval.sort(key=lambda item: (item["prompt_sha256"], item["group_id"]))

    stats = {
        "duplicate": duplicates,
        "unscored": unscored,
        "invalid_reward": invalid_reward,
        "skipped_small_groups": skipped_small_groups,
        "skipped_low_margin_pairs": skipped_low_margin_pairs,
    }
    return dpo_train, dpo_eval, grpo_train, grpo_eval, stats


def build_dataset(input_paths, out_dir, *, mode="sft", artifact_dir=None,
                  seed=DEFAULT_SEED, eval_ratio=DEFAULT_EVAL_RATIO,
                  reward_spec=DEFAULT_REWARD_SPEC,
                  min_margin=DEFAULT_MIN_MARGIN,
                  min_group_size=DEFAULT_MIN_GROUP_SIZE,
                  strict_artifacts=False, overwrite=False):
    """Build deterministic SFT and/or Preference (DPO+GRPO) packs with audit.

    Input trajectory JSONL files and CAS artifact directories are accessed
    strictly read-only. Returns the parsed provenance manifest dictionary.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"unsupported mode: {mode!r}")
    if not input_paths:
        raise ValueError("at least one input trajectory JSONL path is required")
    if not _finite_number(eval_ratio) or not 0.0 <= float(eval_ratio) <= 1.0:
        raise ValueError("eval_ratio must be in [0.0, 1.0]")
    if not _finite_number(min_margin) or float(min_margin) < 0.0:
        raise ValueError("min_margin must be a finite number >= 0.0")
    if not _integer(min_group_size) or min_group_size < 2:
        raise ValueError("min_group_size must be an integer >= 2")

    resolved_inputs = [Path(item) for item in input_paths]
    for path in resolved_inputs:
        if not path.is_file():
            raise FileNotFoundError(f"input trajectory not found: {path}")

    explicit_artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
    if explicit_artifact_dir is not None and not explicit_artifact_dir.is_dir():
        raise FileNotFoundError(
            f"artifact directory not found: {explicit_artifact_dir}")

    reward = None
    if mode in ("preference", "both"):
        reward = load_reward(reward_spec)

    out_path = _prepare_out_dir(out_dir, overwrite=overwrite)

    input_entries = []
    rejection_counts = Counter({
        "cancelled": 0,
        "failed": 0,
        "unverified": 0,
        "duplicate": 0,
        "unscored": 0,
    })
    rejection_details = Counter()
    total_records_read = 0
    checked_input_media = 0
    checked_output_media = 0
    any_disk_verified = False
    candidates = []

    for path in resolved_inputs:
        raw_bytes = path.read_bytes()
        lines = [line for line in raw_bytes.decode("utf-8").splitlines()
                 if line.strip()]
        input_entries.append({
            "name": path.name,
            "sha256": sha256_bytes(raw_bytes),
            "bytes": len(raw_bytes),
            "line_count": len(lines),
        })
        effective_artifact_dir = (
            explicit_artifact_dir
            if explicit_artifact_dir is not None
            else path.parent / "artifacts"
        )
        require_disk = bool(strict_artifacts or explicit_artifact_dir is not None)

        for line in lines:
            total_records_read += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                rejection_counts["unverified"] += 1
                rejection_details["parse_error"] += 1
                continue

            ok, category, _message, stats = verify_record_integrity(
                record,
                artifact_dir=effective_artifact_dir,
                require_artifact_dir=require_disk,
            )
            if stats["checked_disk"]:
                any_disk_verified = True
            checked_input_media += stats["checked_input_media_refs"]
            checked_output_media += stats["checked_output_media_refs"]

            if not ok:
                rejection_counts["unverified"] += 1
                rejection_details[category or "integrity_error"] += 1
                continue

            state = record["terminal_state"]
            if state == TERMINAL_CANCELLED:
                rejection_counts["cancelled"] += 1
                continue
            if state == TERMINAL_FAILED:
                rejection_counts["failed"] += 1
                continue

            candidates.append(_extract_candidate(
                record, verified_on_disk=stats["checked_disk"]))

    input_entries.sort(key=lambda item: (item["sha256"], item["name"]))

    outputs = {}
    per_pack_audits = {}
    all_train_records = []
    all_eval_records = []
    pref_extra = {}

    if mode in ("sft", "both"):
        sft_train, sft_eval, sft_dups = _build_sft_records(
            candidates, seed=seed, eval_ratio=float(eval_ratio))
        rejection_counts["duplicate"] += sft_dups
        outputs["sft_train"] = _write_jsonl_atomic(
            out_path / "sft_train.jsonl", sft_train)
        outputs["sft_eval"] = _write_jsonl_atomic(
            out_path / "sft_eval.jsonl", sft_eval)
        per_pack_audits["sft"] = audit_split_leakage(sft_train, sft_eval)
        all_train_records.extend(sft_train)
        all_eval_records.extend(sft_eval)

    if mode in ("preference", "both"):
        dpo_train, dpo_eval, grpo_train, grpo_eval, pref_stats = (
            _build_preference_records(
                candidates,
                reward=reward,
                seed=seed,
                eval_ratio=float(eval_ratio),
                min_margin=float(min_margin),
                min_group_size=min_group_size,
            )
        )
        if mode == "preference":
            rejection_counts["duplicate"] += pref_stats["duplicate"]
        elif pref_stats["duplicate"]:
            rejection_details["preference_duplicate"] += pref_stats["duplicate"]
        rejection_counts["unscored"] += pref_stats["unscored"]
        if pref_stats["invalid_reward"]:
            rejection_counts["unverified"] += pref_stats["invalid_reward"]
            rejection_details["invalid_reward"] += pref_stats["invalid_reward"]
        pref_extra = {
            "skipped_small_groups": pref_stats["skipped_small_groups"],
            "skipped_low_margin_pairs": pref_stats["skipped_low_margin_pairs"],
        }
        outputs["dpo_train"] = _write_jsonl_atomic(
            out_path / "dpo_train.jsonl", dpo_train)
        outputs["dpo_eval"] = _write_jsonl_atomic(
            out_path / "dpo_eval.jsonl", dpo_eval)
        outputs["grpo_train"] = _write_jsonl_atomic(
            out_path / "grpo_train.jsonl", grpo_train)
        outputs["grpo_eval"] = _write_jsonl_atomic(
            out_path / "grpo_eval.jsonl", grpo_eval)
        per_pack_audits["dpo"] = audit_split_leakage(dpo_train, dpo_eval)
        per_pack_audits["grpo"] = audit_split_leakage(grpo_train, grpo_eval)
        all_train_records.extend(dpo_train)
        all_train_records.extend(grpo_train)
        all_eval_records.extend(dpo_eval)
        all_eval_records.extend(grpo_eval)

    overall_audit = audit_split_leakage(all_train_records, all_eval_records)
    if overall_audit["leakage_count"] != 0:
        raise RuntimeError(
            f"prompt leakage detected across train/eval splits: "
            f"{overall_audit['leakage_count']}")

    manifest = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "status": "COMPLETED",
        "REAL_TRAINING": "NOT_RUN",
        "claims": {
            "REAL_TRAINING": "NOT_RUN",
            "optimizer": "NOT_RUN",
            "model_weights_updated": False,
        },
        "mode": mode,
        "seed": str(seed),
        "eval_ratio": float(eval_ratio),
        "min_margin": float(min_margin) if mode in ("preference", "both") else None,
        "min_group_size": min_group_size if mode in ("preference", "both") else None,
        "reward": ({"spec": reward_spec, "name": reward.name}
                   if reward is not None else None),
        "inputs": input_entries,
        "artifact_verification": {
            "mode": "verified_on_disk" if any_disk_verified else "no_artifact_dir",
            "checked_input_media_refs": checked_input_media,
            "checked_output_media_refs": checked_output_media,
        },
        "counts": {
            "total_records_read": total_records_read,
            "accepted_verified_completed": len(candidates),
            "rejected": {
                "cancelled": rejection_counts["cancelled"],
                "failed": rejection_counts["failed"],
                "unverified": rejection_counts["unverified"],
                "duplicate": rejection_counts["duplicate"],
                "unscored": rejection_counts["unscored"],
            },
            "rejection_details": dict(sorted(rejection_details.items())),
            **pref_extra,
        },
        "outputs": outputs,
        "split_audit": {
            **overall_audit,
            "packs": per_pack_audits,
        },
    }
    _write_manifest_atomic(out_path, manifest)
    return manifest


def audit_dataset_pack(out_dir):
    """Independently verify a generated dataset pack directory on disk.

    Recomputes ``manifest.json.sha256``, verifies every output JSONL file's
    SHA-256 and record count, validates split labels, and re-audits prompt
    leakage across train and eval splits.
    """
    root = Path(out_dir)
    manifest_path = root / "manifest.json"
    sidecar_path = root / "manifest.json.sha256"
    if not manifest_path.is_file():
        raise ValueError("missing manifest.json")
    if not sidecar_path.is_file():
        raise ValueError("missing manifest.json.sha256")

    actual_manifest_sha = sha256_file(manifest_path)
    expected_manifest_sha = sidecar_path.read_text(encoding="utf-8").strip()
    if actual_manifest_sha != expected_manifest_sha:
        raise ValueError("manifest.json.sha256 mismatch")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("REAL_TRAINING") != "NOT_RUN":
        raise ValueError("manifest missing explicit REAL_TRAINING: NOT_RUN")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("manifest has no outputs")

    loaded = {}
    for logical_name, meta in outputs.items():
        rel_path = meta.get("path")
        if not isinstance(rel_path, str) or Path(rel_path).name != rel_path:
            raise ValueError(f"invalid output filename for {logical_name}")
        file_path = root / rel_path
        if not file_path.is_file():
            raise ValueError(f"missing output file: {rel_path}")
        raw = file_path.read_bytes()
        if len(raw) != meta.get("bytes") \
                or sha256_bytes(raw) != meta.get("sha256"):
            raise ValueError(f"output hash/size mismatch for {rel_path}")
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()
                if line.strip()]
        if len(rows) != meta.get("record_count"):
            raise ValueError(f"record count mismatch for {rel_path}")
        expected_split = "eval" if logical_name.endswith("_eval") else "train"
        for index, row in enumerate(rows):
            if row.get("split") != expected_split:
                raise ValueError(
                    f"{rel_path} record {index} split={row.get('split')!r} "
                    f"does not match {expected_split!r}")
        loaded[logical_name] = rows

    train_all, eval_all = [], []
    for name, rows in loaded.items():
        if name.endswith("_eval"):
            eval_all.extend(rows)
        else:
            train_all.extend(rows)

    audit = audit_split_leakage(train_all, eval_all)
    if audit["leakage_count"] != 0:
        raise ValueError(
            f"train/eval prompt leakage detected: {audit['leakage_count']} "
            f"overlapping prompts")
    return {
        "status": "PASSED",
        "manifest_sha256": actual_manifest_sha,
        "REAL_TRAINING": manifest["REAL_TRAINING"],
        "split_audit": audit,
    }


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Stage C Joint SFT & Preference (DPO/GRPO) dataset builder")
    parser.add_argument(
        "--input", action="extend", nargs="+", default=[],
        help="Read-only schema-v1/v2 trajectory JSONL path(s)")
    parser.add_argument(
        "--out", required=True,
        help="Output directory for dataset JSONL files and manifest")
    parser.add_argument(
        "--mode", choices=VALID_MODES, default="sft",
        help="Pack mode: sft, preference (DPO+GRPO), or both")
    parser.add_argument(
        "--artifact-dir", default=None,
        help="Optional explicit CAS artifact directory")
    parser.add_argument(
        "--seed", default=DEFAULT_SEED,
        help="Deterministic split seed (sha256(seed + prompt_sha256))")
    parser.add_argument(
        "--eval-ratio", type=float, default=DEFAULT_EVAL_RATIO,
        help="Fraction of prompts routed to eval split [0.0, 1.0]")
    parser.add_argument(
        "--reward", default=DEFAULT_REWARD_SPEC,
        help="Dotted module:attr Reward loader for preference mode")
    parser.add_argument(
        "--min-margin", type=float, default=DEFAULT_MIN_MARGIN,
        help="Minimum score margin (chosen - rejected) for DPO pairs")
    parser.add_argument(
        "--min-group-size", type=int, default=DEFAULT_MIN_GROUP_SIZE,
        help="Minimum scored rollouts per prompt for GRPO/DPO grouping")
    parser.add_argument(
        "--strict-artifacts", action="store_true",
        help="Fail records with CAS refs if artifact directory is absent")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow writing into a non-empty --out directory")
    parser.add_argument(
        "--audit-only", action="store_true",
        help="Verify an existing dataset pack directory instead of building")
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.audit_only:
            result = audit_dataset_pack(args.out)
            sys.stdout.write(
                canonical_json(result).decode("utf-8") + "\n")
            return 0
        if not args.input:
            parser.error("--input is required unless --audit-only is set")
        manifest = build_dataset(
            args.input,
            args.out,
            mode=args.mode,
            artifact_dir=args.artifact_dir,
            seed=args.seed,
            eval_ratio=args.eval_ratio,
            reward_spec=args.reward,
            min_margin=args.min_margin,
            min_group_size=args.min_group_size,
            strict_artifacts=args.strict_artifacts,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"dataset error: {exc}\n")
    sys.stdout.write(canonical_json(manifest).decode("utf-8") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
