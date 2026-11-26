# SPDX-License-Identifier: Apache-2.0
"""Optional deterministic token-id and source-span export for trajectory v2.

The core has no serving, trainer, or ML-framework dependency. A caller supplies
an already constructed fast-tokenizer-compatible object and explicitly declares
its identity. Every token must cover a non-empty, contiguous UTF-8 byte range
and decode back to exactly those source bytes. The additive ``token_export``
block is bound to the complete source record and to hashes of every available
file below the selected tokenizer directory.
"""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re

TOKEN_EXPORT_FORMAT = "umminfer-trajectory-token-export-v1"
TOKEN_EXPORT_FORMAT_VERSION = 1

_HEX_256 = re.compile(r"[0-9a-f]{64}\Z")
_VISUAL_BEAT_RE = re.compile(r"(\[\[(image|video):[ \t]*)(.*?)(\]\])", re.DOTALL)
_SPOKEN_TAG_RE = re.compile(r"<spoken(?:\s+[^>]*)?>.*?</spoken>", re.DOTALL)

# These are the joint_full defaults, duplicated deliberately at the export
# boundary so this module never imports the trainer. Tests compare both mask
# implementations token-for-token; a future weight/profile change needs a new
# export format rather than silently changing existing records.
SFT_SPAN_WEIGHTS = {
    "narrative_prose": 1.0,
    "visual_route_tag": 1.5,
    "visual_prompt_body": 1.0,
    "spoken_style": 1.0,
    "image_form_multiplier": 1.0,
    "video_form_multiplier": 1.0,
}

EXPORT_POLICY = {
    "add_special_tokens": False,
    "padding": "none",
    "truncation": "none",
    "unknown_tokens": "reject",
    "offset_unit": "utf8_byte",
    "offset_interval": "half_open",
    "offset_continuity": "exact_no_gaps_or_overlaps",
    "normalization": "source_bytes_must_equal_token_decode",
    "decode": "per_token_skip_special_false_cleanup_false",
    "sft_mask": "joint_full_codepoint_overlap_weighted_v1",
}


class TokenExportError(ValueError):
    """A tokenizer, token stream, source span, or integrity binding is invalid."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def _sha_text(text):
    return _sha_bytes(text.encode("utf-8"))


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _same_json(left, right):
    """Type-strict JSON equality: bool/float never impersonate an integer."""
    try:
        return _canonical(left) == _canonical(right)
    except (TypeError, ValueError):
        return False


def _require(condition, message):
    if not condition:
        raise TokenExportError(message)


def _nonempty_text(value, label):
    _require(isinstance(value, str) and bool(value.strip()),
             f"{label} must be an explicit non-empty string")
    return value


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_available_tokenizer_files(tokenizer_dir):
    """Hash all regular files below a tokenizer directory, path-free and sorted."""
    root = Path(tokenizer_dir)
    _require(root.is_dir(), "tokenizer_dir must be an existing directory")
    entries = []
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if path.is_symlink():
            raise TokenExportError(
                f"tokenizer directory contains a symlink: {path.relative_to(root)}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        entries.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        })
    _require(bool(entries), "tokenizer_dir contains no available files to hash")
    return entries


def _files_digest(entries):
    return _sha_bytes(_canonical(entries))


def _source_record(record):
    return {key: copy.deepcopy(value) for key, value in record.items()
            if key != "token_export"}


def _source_record_digest(record):
    return _sha_bytes(_canonical(_source_record(record)))


def _config_digest(identity):
    return _sha_bytes(_canonical({
        "tokenizer": identity,
        "policy": EXPORT_POLICY,
    }))


def _tokenizer_identity(tokenizer, tokenizer_dir, *, tokenizer_id,
                        tokenizer_revision, tokenization_version):
    _nonempty_text(tokenizer_id, "tokenizer_id")
    _nonempty_text(tokenizer_revision, "tokenizer_revision")
    _nonempty_text(tokenization_version, "tokenization_version")
    try:
        token_space = len(tokenizer)
    except (TypeError, AttributeError) as exc:
        raise TokenExportError(
            "tokenizer must expose its complete token-id space through len()") from exc
    _require(_integer(token_space) and token_space > 0,
             "tokenizer token-id space must be a positive integer")
    unknown = getattr(tokenizer, "unk_token_id", None)
    _require(unknown is None or (_integer(unknown) and 0 <= unknown < token_space),
             "tokenizer unknown-token id is invalid")
    files = hash_available_tokenizer_files(tokenizer_dir)
    identity = {
        "id": tokenizer_id,
        "revision": tokenizer_revision,
        "tokenization_version": tokenization_version,
        "token_id_space_size": token_space,
        "unknown_token_id": unknown,
        "available_files": files,
        "available_files_sha256": _files_digest(files),
    }
    return identity


def _char_byte_map(text):
    offsets = [0]
    for char in text:
        offsets.append(offsets[-1] + len(char.encode("utf-8")))
    return offsets


def _snap_utf8_offset(raw, offset):
    _require(_integer(offset) and offset >= 0,
             "visual slot text_offset must be a non-negative integer")
    clamped = min(offset, len(raw))
    while clamped < len(raw) and (raw[clamped] & 0xC0) == 0x80:
        clamped += 1
    return clamped


def _canonical_prompt(prompt):
    return " ".join(prompt.split())


def _clean_segments(record):
    """Return exact clean-text pieces and their normalized source event ids."""
    events = record["events"]
    parts = [event for event in events
             if event.get("type") in ("vlm_delta", "output_delta")]
    if not parts:
        terminal = events[-1]
        if terminal.get("type") == "complete" and not any(
                event.get("type") == "asr_final" for event in events):
            parts = [terminal]
    segments = []
    cursor = 0
    for event in parts:
        text = event.get("text", "")
        _require(isinstance(text, str), "clean-text source event is not text")
        raw = text.encode("utf-8")
        if raw:
            segments.append({
                "text": text, "start": cursor, "end": cursor + len(raw),
                "event_index": event["seq"],
            })
            cursor += len(raw)
    _require(b"".join(item["text"].encode("utf-8") for item in segments)
             == record["clean_text"].encode("utf-8"),
             "clean-text source events do not concatenate canonically")
    return segments


def _slice_segments(segments, start, end):
    """Slice a clean byte range while retaining exact source-event identities."""
    if end <= start:
        return []
    output = []
    for segment in segments:
        overlap_start = max(start, segment["start"])
        overlap_end = min(end, segment["end"])
        if overlap_start >= overlap_end:
            continue
        raw = segment["text"].encode("utf-8")
        piece = raw[overlap_start - segment["start"]:
                    overlap_end - segment["start"]]
        output.append({"text": piece.decode("utf-8"),
                       "event_index": segment["event_index"]})
    _require(sum(len(item["text"].encode("utf-8")) for item in output) == end - start,
             "clean-text slice has a source gap")
    return output


def reconstruct_sft_target_segments(record):
    """Reconstruct the canonical SFT target and marker/source provenance.

    This mirrors the existing dataset builder's director-target reconstruction
    without importing it. ``attach_token_export(..., sft_target=...)`` and
    validation's ``expected_sft_target`` provide an independent equality check.
    """
    _require(_integer(record.get("schema_version")) and record["schema_version"] == 2,
             "token export requires schema_version=2")
    clean = record.get("clean_text")
    _require(isinstance(clean, str), "v2 token export requires clean_text")
    clean_segments = _clean_segments(record)
    clean_raw = clean.encode("utf-8")
    slots = sorted(record.get("visual_slots") or [],
                   key=lambda slot: (_snap_utf8_offset(
                       clean_raw, slot.get("text_offset")), slot.get("slot_index")))
    legs = {leg.get("slot_id"): leg for leg in record.get("terminal_legs") or []
            if isinstance(leg, dict)}
    artifacts = {}
    for artifact in record.get("artifacts") or []:
        if artifact.get("slot_id") is not None:
            artifacts.setdefault(artifact["slot_id"], artifact)

    pieces = []
    cursor = 0
    for slot in slots:
        offset = _snap_utf8_offset(clean_raw, slot.get("text_offset"))
        _require(offset >= cursor, "visual slots overlap or are noncanonical")
        pieces.extend(_slice_segments(clean_segments, cursor, offset))
        cursor = offset
        prompt = _canonical_prompt(slot.get("prompt", ""))
        leg = legs.get(slot.get("slot_id"), {})
        marker = ""
        if prompt or leg.get("state") == "completed":
            _require(slot.get("media_type") in ("image", "video"),
                     "visual slot has an unsupported media_type")
            _require(bool(prompt), "completed visual slot has an empty prompt")
            marker = f"[[{slot['media_type']}: {prompt}]]"
        bound = artifacts.get(slot.get("slot_id"))
        if bound is not None and bound.get("anchor_explicit", False):
            _require(slot.get("media_type") == "video"
                     and bound.get("anchor_slot_index") == slot.get("anchor_slot_index")
                     and _integer(bound.get("anchor_slot_index"))
                     and bound.get("anchor_slot_index") >= 0
                     and isinstance(bound.get("anchor_sha256"), str)
                     and _HEX_256.fullmatch(bound["anchor_sha256"]) is not None,
                     "invalid explicit visual-anchor marker")
            marker = marker[:-2] + (
                f" | source=slot-{bound['anchor_slot_index']}]]")
        if marker:
            event_index = slot.get("seq")
            _require(_integer(event_index)
                     and 0 <= event_index < len(record["events"])
                     and record["events"][event_index].get("type") == "visual_slot"
                     and record["events"][event_index].get("slot_id")
                     == slot.get("slot_id"),
                     "visual marker lacks its exact source slot event")
            pieces.append({"text": marker, "event_index": event_index})
    pieces.extend(_slice_segments(clean_segments, cursor, len(clean_raw)))

    target_segments = []
    target_cursor = 0
    for piece in pieces:
        raw = piece["text"].encode("utf-8")
        target_segments.append({
            "text": piece["text"], "start": target_cursor,
            "end": target_cursor + len(raw), "event_index": piece["event_index"],
        })
        target_cursor += len(raw)
    return "".join(piece["text"] for piece in pieces), target_segments


def _new_span(spans, span_type, modality, text, char_start, char_end, char_to_byte):
    spans.append({
        "span_type": span_type,
        "modality": modality,
        "start": char_to_byte[char_start],
        "end": char_to_byte[char_end],
        "char_start": char_start,
        "char_end": char_end,
        "text_sha256": _sha_text(text[char_start:char_end]),
    })


def _parse_spans(text):
    """Lossless typed spans with both UTF-8 byte and Python-codepoint offsets."""
    char_to_byte = _char_byte_map(text)
    spans = []
    cursor = 0
    for match in _SPOKEN_TAG_RE.finditer(text):
        if match.start() > cursor:
            _emit_visual_plain_range(spans, text, char_to_byte, cursor,
                                     match.start(), "narrative_prose", "text")
        _emit_visual_plain_range(spans, text, char_to_byte, match.start(),
                                 match.end(), "spoken_style", "audio")
        cursor = match.end()
    if cursor < len(text):
        _emit_visual_plain_range(spans, text, char_to_byte, cursor, len(text),
                                 "narrative_prose", "text")
    return spans


def _emit_visual_plain_range(spans, text, char_to_byte, start, end,
                             default_type, default_modality):
    # Match within a slice but retain offsets in the original string. This is
    # essential for spoken tags and visual beats separated by multibyte prose.
    local = text[start:end]
    cursor = 0
    for match in _VISUAL_BEAT_RE.finditer(local):
        if match.start() > cursor:
            _new_span(spans, default_type, default_modality, text,
                      start + cursor, start + match.start(), char_to_byte)
        modality = match.group(2)
        _new_span(spans, "visual_route_tag", modality, text,
                  start + match.start(1), start + match.end(1), char_to_byte)
        if match.group(3):
            _new_span(spans, "visual_prompt_body", modality, text,
                      start + match.start(3), start + match.end(3), char_to_byte)
        _new_span(spans, "visual_route_tag", modality, text,
                  start + match.start(4), start + match.end(4), char_to_byte)
        cursor = match.end()
    if cursor < len(local):
        _new_span(spans, default_type, default_modality, text,
                  start + cursor, start + len(local), char_to_byte)


def _effective_weight(span):
    weight = float(SFT_SPAN_WEIGHTS[span["span_type"]])
    if span["modality"] == "image":
        weight *= float(SFT_SPAN_WEIGHTS["image_form_multiplier"])
    elif span["modality"] == "video":
        weight *= float(SFT_SPAN_WEIGHTS["video_form_multiplier"])
    return weight


def _source_indices(token_start, token_end, segments, events):
    event_indices = []
    for segment in segments:
        if min(token_end, segment["end"]) <= max(token_start, segment["start"]):
            continue
        index = segment["event_index"]
        if index not in event_indices:
            event_indices.append(index)
    frame_indices = []
    for index in event_indices:
        _require(_integer(index) and 0 <= index < len(events)
                 and events[index].get("seq") == index,
                 "token source event index is invalid")
        frame = events[index].get("source_event_index")
        if frame is not None:
            _require(_integer(frame) and frame >= 0,
                     "source event frame index is invalid")
            frame_indices.append(frame)
    return event_indices, frame_indices


def _encode(tokenizer, text):
    try:
        encoded = tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True,
            padding=False, truncation=False)
        ids = list(encoded["input_ids"])
        offsets = list(encoded["offset_mapping"])
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise TokenExportError(
            "tokenizer must provide input_ids and fast offset_mapping without "
            "special tokens, padding, or truncation") from exc
    _require(len(ids) == len(offsets), "token ids and offsets differ in length")
    char_to_byte = _char_byte_map(text)
    byte_offsets = []
    for index, offset in enumerate(offsets):
        _require(isinstance(offset, (list, tuple)) and len(offset) == 2
                 and all(_integer(value) for value in offset),
                 f"token {index} offset is not an integer pair")
        start, end = offset
        _require(0 <= start <= end <= len(text),
                 f"token {index} offset is out of range")
        byte_offsets.append((char_to_byte[start], char_to_byte[end]))
    return ids, byte_offsets


def _decode_tokens(tokenizer, ids):
    pieces = []
    for index, token_id in enumerate(ids):
        try:
            piece = tokenizer.decode(
                [token_id], skip_special_tokens=False,
                clean_up_tokenization_spaces=False)
        except (AttributeError, TypeError, ValueError) as exc:
            raise TokenExportError(
                f"tokenizer cannot deterministically decode token {index}") from exc
        _require(isinstance(piece, str),
                 f"tokenizer decode for token {index} is not text")
        pieces.append(piece.encode("utf-8"))
    return pieces


def _build_stream(kind, text, ids, byte_offsets, segments, events, identity,
                  decoded_pieces=None):
    _require(isinstance(text, str), f"{kind} text must be a string")
    raw = text.encode("utf-8")
    _require(len(ids) == len(byte_offsets),
             f"{kind} token ids and offsets differ in length")
    char_to_byte = _char_byte_map(text)
    byte_to_char = {value: index for index, value in enumerate(char_to_byte)}
    spans = _parse_spans(text)
    _require(not spans or (spans[0]["start"] == 0 and spans[-1]["end"] == len(raw)),
             f"{kind} spans do not cover the target")
    for previous, current in zip(spans, spans[1:]):
        _require(previous["end"] == current["start"],
                 f"{kind} spans overlap or contain a gap")

    tokens = []
    annotations = []
    weights = []
    cursor = 0
    for index, (token_id, offset) in enumerate(zip(ids, byte_offsets)):
        _require(_integer(token_id) and 0 <= token_id < identity["token_id_space_size"],
                 f"{kind} token {index} has an invalid token id")
        unknown = identity["unknown_token_id"]
        _require(unknown is None or token_id != unknown,
                 f"{kind} token {index} is the rejected unknown token")
        _require(isinstance(offset, (list, tuple)) and len(offset) == 2
                 and all(_integer(value) for value in offset),
                 f"{kind} token {index} offset is not an integer pair")
        start, end = offset
        _require(start == cursor and end > start and end <= len(raw),
                 f"{kind} token {index} has an overlap, gap, or out-of-range offset")
        _require(start in byte_to_char and end in byte_to_char,
                 f"{kind} token {index} splits a UTF-8 codepoint")
        piece = raw[start:end]
        if decoded_pieces is not None:
            _require(decoded_pieces[index] == piece,
                     f"{kind} token {index} decode differs from source UTF-8 bytes")
        char_start, char_end = byte_to_char[start], byte_to_char[end]
        overlapping = []
        for span_index, span in enumerate(spans):
            byte_overlap = max(0, min(end, span["end"]) - max(start, span["start"]))
            if not byte_overlap:
                continue
            char_overlap = max(0, min(char_end, span["char_end"])
                               - max(char_start, span["char_start"]))
            _require(char_overlap > 0, f"{kind} token {index} has no codepoint overlap")
            overlapping.append((span_index, char_overlap))
        _require(sum(item[1] for item in overlapping) == char_end - char_start,
                 f"{kind} token {index} is not exactly covered by SFT spans")
        weight = sum(_effective_weight(spans[span_index]) * overlap
                     for span_index, overlap in overlapping) / (char_end - char_start)
        _require(math.isfinite(weight) and weight > 0.0,
                 f"{kind} token {index} has no positive joint_full loss weight")
        span_types = sorted({spans[span_index]["span_type"]
                             for span_index, _ in overlapping})
        closing = any(spans[span_index]["span_type"] == "visual_route_tag"
                      and raw[spans[span_index]["start"]:spans[span_index]["end"]] == b"]]"
                      for span_index, _ in overlapping)
        source_events, source_frames = _source_indices(start, end, segments, events)
        tokens.append({
            "index": index,
            "token_id": token_id,
            "start": start,
            "end": end,
            "char_start": char_start,
            "char_end": char_end,
            "piece_sha256": _sha_bytes(piece),
            "source_event_indices": source_events,
            "source_frame_indices": source_frames,
        })
        annotation = {
            "span_types": span_types,
            "weight": weight,
            "includes_terminal_delimiter": closing,
        }
        annotations.append(annotation)
        weights.append(weight)
        cursor = end
    _require(cursor == len(raw), f"{kind} tokens do not exactly cover the text")

    for span_index, span in enumerate(spans):
        overlapping = [index for index, token in enumerate(tokens)
                       if min(token["end"], span["end"])
                       > max(token["start"], span["start"])]
        _require(bool(overlapping), f"{kind} span {span_index} has no tokens")
        span["token_start"] = overlapping[0]
        span["token_end"] = overlapping[-1] + 1

    return {
        "source": kind,
        "text_bytes": len(raw),
        "text_sha256": _sha_bytes(raw),
        "decoded_sha256": _sha_bytes(b"".join(decoded_pieces))
            if decoded_pieces is not None else _sha_bytes(raw),
        "token_count": len(tokens),
        "token_ids": list(ids),
        "token_ids_sha256": _sha_bytes(_canonical(list(ids))),
        "tokens": tokens,
        "tokens_sha256": _sha_bytes(_canonical(tokens)),
        "spans": spans,
        "spans_sha256": _sha_bytes(_canonical(spans)),
        "sft_mask": {
            "ablation_profile": "joint_full",
            "span_weights": copy.deepcopy(SFT_SPAN_WEIGHTS),
            "token_weights": weights,
            "token_weights_sha256": _sha_bytes(_canonical(weights)),
            "annotations_sha256": _sha_bytes(_canonical(annotations)),
            "active_tokens": sum(weight > 0.0 for weight in weights),
            "terminal_delimiter_tokens": sum(
                item["includes_terminal_delimiter"] for item in annotations),
        },
    }


def attach_token_export(record, tokenizer, *, tokenizer_dir, tokenizer_id,
                        tokenizer_revision, tokenization_version, sft_target=None):
    """Return a copied v2 record with an additive, source-bound token export.

    ``sft_target`` is optional but recommended: when supplied by the existing
    dataset builder, any disagreement with this module's reconstruction fails.
    """
    _require(isinstance(record, dict)
             and _integer(record.get("schema_version"))
             and record["schema_version"] == 2,
             "token export is defined only for schema-v2 records")
    _require("token_export" not in record,
             "record already has token_export; refusing silent replacement")
    identity = _tokenizer_identity(
        tokenizer, tokenizer_dir, tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        tokenization_version=tokenization_version)
    clean = record.get("clean_text")
    _require(isinstance(clean, str), "v2 token export requires clean_text")
    target, target_segments = reconstruct_sft_target_segments(record)
    if sft_target is not None:
        _require(target == sft_target,
                 "token export target differs from the SFT dataset target")
    clean_segments = _clean_segments(record)

    streams = {}
    for kind, text, segments in (
            ("clean_text", clean, clean_segments),
            ("sft_target", target, target_segments)):
        ids, offsets = _encode(tokenizer, text)
        decoded = _decode_tokens(tokenizer, ids)
        streams[kind] = _build_stream(
            kind, text, ids, offsets, segments, record["events"], identity,
            decoded_pieces=decoded)

    block = {
        "format": TOKEN_EXPORT_FORMAT,
        "format_version": TOKEN_EXPORT_FORMAT_VERSION,
        "source_record_sha256": _source_record_digest(record),
        "tokenizer": identity,
        "policy": copy.deepcopy(EXPORT_POLICY),
        "config_sha256": _config_digest(identity),
        "streams": streams,
    }
    block["export_sha256"] = _sha_bytes(_canonical(block))
    output = copy.deepcopy(record)
    output["token_export"] = block
    validate_record_token_export(output)
    return output


def _validate_identity(identity, tokenizer, tokenizer_dir,
                       expected_tokenizer_id, expected_tokenizer_revision,
                       expected_tokenization_version):
    _require(isinstance(identity, dict), "tokenizer identity must be an object")
    for key in ("id", "revision", "tokenization_version"):
        _nonempty_text(identity.get(key), f"tokenizer.{key}")
    _require(_integer(identity.get("token_id_space_size"))
             and identity["token_id_space_size"] > 0,
             "tokenizer token-id space is invalid")
    unknown = identity.get("unknown_token_id")
    _require(unknown is None or (_integer(unknown) and 0 <= unknown
                                 < identity["token_id_space_size"]),
             "unknown-token identity is invalid")
    files = identity.get("available_files")
    _require(isinstance(files, list) and bool(files),
             "tokenizer available_files must be non-empty")
    previous = None
    for entry in files:
        _require(isinstance(entry, dict) and set(entry) == {"path", "bytes", "sha256"},
                 "available-file entry is malformed")
        path = entry["path"]
        _require(isinstance(path, str) and path and not path.startswith("/")
                 and ".." not in Path(path).parts and Path(path).as_posix() == path,
                 "available-file path is not a safe relative POSIX path")
        _require(previous is None or previous < path,
                 "available-file paths are not unique and sorted")
        previous = path
        _require(_integer(entry["bytes"]) and entry["bytes"] >= 0
                 and isinstance(entry["sha256"], str)
                 and _HEX_256.fullmatch(entry["sha256"]) is not None,
                 f"available-file hash/bytes are invalid: {path}")
    _require(identity.get("available_files_sha256") == _files_digest(files),
             "tokenizer available-files digest does not recompute")
    for actual, expected, label in (
            (identity["id"], expected_tokenizer_id, "tokenizer id"),
            (identity["revision"], expected_tokenizer_revision, "tokenizer revision"),
            (identity["tokenization_version"], expected_tokenization_version,
             "tokenization version")):
        if expected is not None:
            _require(actual == expected, f"{label} changed")
    if tokenizer_dir is not None:
        _require(_same_json(files, hash_available_tokenizer_files(tokenizer_dir)),
                 "tokenizer available-file hashes or bytes changed")
    if tokenizer is not None:
        try:
            token_space = len(tokenizer)
        except (TypeError, AttributeError) as exc:
            raise TokenExportError("validator tokenizer has no token-id space") from exc
        _require(token_space == identity["token_id_space_size"],
                 "tokenizer token-id space changed")
        actual_unknown = getattr(tokenizer, "unk_token_id", None)
        _require((actual_unknown is None and unknown is None)
                 or (_integer(actual_unknown) and _integer(unknown)
                     and actual_unknown == unknown),
                 "tokenizer unknown-token handling changed")


def validate_record_token_export(record, *, tokenizer=None, tokenizer_dir=None,
                                 expected_tokenizer_id=None,
                                 expected_tokenizer_revision=None,
                                 expected_tokenization_version=None,
                                 expected_sft_target=None):
    """Independently validate an additive block; absent blocks remain legacy.

    Without ``tokenizer`` this validates all ids, offsets, source mappings,
    spans, masks, canonical target reconstruction, and integrity hashes against
    source text. Supplying the tokenizer additionally re-encodes both streams
    and decodes every id, proving exact UTF-8 reconstruction and tokenizer/file
    identity rather than merely structural consistency.
    """
    if not isinstance(record, dict):
        raise TokenExportError("record must be an object")
    if "token_export" not in record:
        return {"status": "ABSENT", "present": False,
                "tokenizer_reencoded": False}
    try:
        _require(_integer(record.get("schema_version")) and record["schema_version"] == 2,
                 "only schema-v2 records may carry token_export")
        block = record["token_export"]
        _require(isinstance(block, dict), "token_export must be an object")
        _require(block.get("format") == TOKEN_EXPORT_FORMAT
                 and _integer(block.get("format_version"))
                 and block["format_version"] == TOKEN_EXPORT_FORMAT_VERSION,
                 "unsupported token-export format/version")
        _require(block.get("source_record_sha256") == _source_record_digest(record),
                 "token export is stale or its source record changed")
        _require(_same_json(block.get("policy"), EXPORT_POLICY),
                 "tokenization padding/truncation/unknown/offset policy changed")
        identity = block.get("tokenizer")
        _validate_identity(
            identity, tokenizer, tokenizer_dir, expected_tokenizer_id,
            expected_tokenizer_revision, expected_tokenization_version)
        _require(block.get("config_sha256") == _config_digest(identity),
                 "tokenizer/config identity digest does not recompute")
        supplied_hash = block.get("export_sha256")
        unsigned = {key: value for key, value in block.items()
                    if key != "export_sha256"}
        _require(isinstance(supplied_hash, str)
                 and _HEX_256.fullmatch(supplied_hash) is not None
                 and supplied_hash == _sha_bytes(_canonical(unsigned)),
                 "token-export hash does not recompute")

        target, target_segments = reconstruct_sft_target_segments(record)
        if expected_sft_target is not None:
            _require(target == expected_sft_target,
                     "reconstructed target differs from the SFT dataset target")
        streams = block.get("streams")
        _require(isinstance(streams, dict)
                 and set(streams) == {"clean_text", "sft_target"},
                 "token export requires exactly clean_text and sft_target streams")
        for kind, text, segments in (
                ("clean_text", record["clean_text"], _clean_segments(record)),
                ("sft_target", target, target_segments)):
            stream = streams[kind]
            _require(isinstance(stream, dict), f"{kind} stream must be an object")
            ids = stream.get("token_ids")
            tokens = stream.get("tokens")
            _require(isinstance(ids, list) and isinstance(tokens, list)
                     and len(ids) == len(tokens),
                     f"{kind} token ids/tokens are missing or differ in length")
            offsets = []
            for index, token in enumerate(tokens):
                _require(isinstance(token, dict)
                         and _integer(token.get("index")) and token["index"] == index
                         and _integer(token.get("token_id"))
                         and token["token_id"] == ids[index],
                         f"{kind} token identity/index mismatch at {index}")
                offsets.append((token.get("start"), token.get("end")))
            decoded = None
            if tokenizer is not None:
                actual_ids, actual_offsets = _encode(tokenizer, text)
                _require(_same_json(ids, actual_ids)
                         and _same_json(offsets, actual_offsets),
                         f"{kind} tokenizer re-encoding changed ids or offsets")
                decoded = _decode_tokens(tokenizer, ids)
            expected = _build_stream(
                kind, text, ids, offsets, segments, record["events"], identity,
                decoded_pieces=decoded)
            _require(_same_json(stream, expected),
                     f"{kind} stream differs from canonical offsets/spans/mask/hashes")
        return {
            "status": "PASS", "present": True,
            "tokenizer_reencoded": tokenizer is not None,
            "clean_text_tokens": streams["clean_text"]["token_count"],
            "sft_target_tokens": streams["sft_target"]["token_count"],
            "sft_target_sha256": _sha_text(target),
        }
    except TokenExportError:
        raise
    except (KeyError, IndexError, TypeError, AttributeError,
            UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TokenExportError(f"malformed token export: {exc}") from exc


def _load_real_tokenizer(tokenizer_dir):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise TokenExportError(
            "real-tokenizer CLI requires optional transformers; structural "
            "validation remains stdlib-only") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir), local_files_only=True, use_fast=True,
        trust_remote_code=False)
    if not getattr(tokenizer, "is_fast", False):
        raise TokenExportError("real-tokenizer export requires a fast tokenizer")
    return tokenizer


def _read_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TokenExportError(f"line {line_number} is not a JSON object")
            records.append(value)
    if not records:
        raise TokenExportError("input trajectory contains no records")
    return records


def _write_new_jsonl(path, records):
    output = Path(path)
    _require(not output.exists(), "output path already exists; refusing overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = b"".join(_canonical(record) + b"\n" for record in records)
    temporary = output.with_name(output.name + ".tmp")
    try:
        temporary.write_bytes(raw)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="trajectory JSONL (read-only)")
    parser.add_argument("--out", help="new JSONL for export; omitted with --validate-only")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--tokenizer-dir")
    parser.add_argument("--tokenizer-id")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--tokenization-version")
    args = parser.parse_args(argv)
    try:
        records = _read_jsonl(args.input)
        tokenizer = None
        identity_args = (args.tokenizer_dir, args.tokenizer_id,
                         args.tokenizer_revision, args.tokenization_version)
        if any(value is not None for value in identity_args):
            if not all(value is not None for value in identity_args):
                raise TokenExportError(
                    "real-tokenizer mode requires --tokenizer-dir, --tokenizer-id, "
                    "--tokenizer-revision, and --tokenization-version together")
            tokenizer = _load_real_tokenizer(args.tokenizer_dir)
        if args.validate_only:
            if args.out is not None:
                raise TokenExportError("--out is not valid with --validate-only")
            counts = {"records": len(records), "legacy_or_absent": 0,
                      "validated": 0, "tokenizer_reencoded": 0}
            for record in records:
                result = validate_record_token_export(
                    record, tokenizer=tokenizer, tokenizer_dir=args.tokenizer_dir,
                    expected_tokenizer_id=args.tokenizer_id,
                    expected_tokenizer_revision=args.tokenizer_revision,
                    expected_tokenization_version=args.tokenization_version)
                if result["present"]:
                    counts["validated"] += 1
                    counts["tokenizer_reencoded"] += int(result["tokenizer_reencoded"])
                else:
                    counts["legacy_or_absent"] += 1
            print(json.dumps({"status": "PASS", **counts}, sort_keys=True))
            return 0
        if tokenizer is None or args.out is None:
            raise TokenExportError(
                "export requires --out and all four real-tokenizer identity options")
        exported = []
        for record in records:
            if record.get("schema_version") == 1:
                exported.append(record)
                continue
            exported.append(attach_token_export(
                record, tokenizer, tokenizer_dir=args.tokenizer_dir,
                tokenizer_id=args.tokenizer_id,
                tokenizer_revision=args.tokenizer_revision,
                tokenization_version=args.tokenization_version))
        for record in exported:
            if record.get("schema_version") == 2:
                validate_record_token_export(
                    record, tokenizer=tokenizer, tokenizer_dir=args.tokenizer_dir,
                    expected_tokenizer_id=args.tokenizer_id,
                    expected_tokenizer_revision=args.tokenizer_revision,
                    expected_tokenization_version=args.tokenization_version)
        _write_new_jsonl(args.out, exported)
        print(json.dumps({
            "status": "PASS", "records": len(records),
            "exported": sum(record.get("schema_version") == 2
                            for record in exported),
            "legacy_preserved": sum(record.get("schema_version") == 1
                                    for record in exported),
        }, sort_keys=True))
        return 0
    except (TokenExportError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"token export error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
