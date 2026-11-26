# SPDX-License-Identifier: Apache-2.0
"""Independent CPU binding of sampled behavior evidence to token_export v1/v2.

The validator consumes retained records only. It never performs inference,
serving, optimization, or numerical logit recomputation. Structural evidence,
raw-SSE/CAS bytes, attestations, source mappings, masks, and sampled/post-hoc id
relations are checked before an optional fast tokenizer is imported. A report
is written only after every requested check and fresh raw replay has passed.
"""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unicodedata

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

import umminfer_online_rollout as online  # noqa: E402
import umminfer_rl_trajectory as traj  # noqa: E402
import umminfer_sse_replay as sse  # noqa: E402
import umminfer_token_export as tokens  # noqa: E402

ALIGNMENT_SCHEMA = "umminfer-behavior-token-alignment-fixture-v1"
BEHAVIOR_FORMAT = "umminfer-behavior-logprobs-v1"
LOADED_TOKENIZER_FORMAT = "umminfer-qwen38-loaded-tokenizer-v1"
BEHAVIOR_KEYS = frozenset(
    ("format", "tokenizer_sha256", "token_ids", "logprobs", "mask"))
VISUAL_SPANS = frozenset(("visual_route_tag", "visual_prompt_body"))
PROSE_SPANS = frozenset(("narrative_prose", "spoken_style"))
_HEX = frozenset("0123456789abcdef")


class AlignmentError(ValueError):
    """Sampled evidence and its supervision/source binding are not aligned."""


def _require(condition, message):
    if not condition:
        raise AlignmentError(message)


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(float(value)))


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    return tokens._canonical(value)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AlignmentError(f"duplicate JSON member: {key}")
        result[key] = value
    return result


def _constant(value):
    raise AlignmentError(f"non-finite JSON number is forbidden: {value}")


def _json(raw, label):
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                          parse_constant=_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AlignmentError(f"{label} is not strict UTF-8 JSON: {exc}") from exc


def _safe_path(root, relative):
    _require(isinstance(relative, str) and relative and not relative.startswith("/")
             and Path(relative).as_posix() == relative
             and ".." not in Path(relative).parts,
             f"unsafe fixture-relative path: {relative!r}")
    path = Path(root) / relative
    _require(not path.is_symlink(), f"fixture path may not be a symlink: {relative}")
    return path


def _read(root, relative, expected_sha=None):
    path = _safe_path(root, relative)
    raw = path.read_bytes()
    if expected_sha is not None:
        _require(_sha(raw) == expected_sha,
                 f"fixture file SHA-256 mismatch: {relative}")
    return raw


def _document(root, relative, expected_sha=None):
    raw = _read(root, relative, expected_sha)
    value = _json(raw, relative)
    _require(raw == _canonical(value) + b"\n",
             f"fixture document is not canonical JSON plus newline: {relative}")
    return value


def _jsonl(raw, label):
    _require(raw and raw.endswith(b"\n"), f"{label} is empty or lacks a final newline")
    records = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        _require(bool(line), f"{label} contains a blank record at {line_number}")
        value = _json(line, f"{label}:{line_number}")
        _require(isinstance(value, dict), f"{label}:{line_number} is not an object")
        _require(_canonical(value) == line,
                 f"{label}:{line_number} is not canonical JSON")
        records.append(value)
    _require(bool(records), f"{label} has no records")
    _require(raw == b"".join(_canonical(record) + b"\n" for record in records),
             f"{label} byte reconstruction failed")
    return records


def _source_record(record):
    return {key: copy.deepcopy(value) for key, value in record.items()
            if key != "token_export"}


def _without_cas(value):
    return online._without_cas_filenames(value)


def _validate_cas(record, artifact_dir):
    for item in list(record.get("inputs", [])) + list(record.get("artifacts", [])):
        if not isinstance(item, dict) or item.get("kind") == "text" \
                or item.get("ref") is None:
            continue
        digest = item.get("sha256")
        ref = item.get("ref")
        _require(isinstance(digest, str) and len(digest) == 64
                 and not set(digest) - _HEX
                 and isinstance(ref, str) and Path(ref).name == ref
                 and ref.startswith(digest + "."),
                 "record contains an invalid content-addressed media reference")
        path = _safe_path(artifact_dir, ref)
        raw = path.read_bytes()
        _require(_sha(raw) == digest, f"CAS media SHA-256 mismatch: {ref}")
        if item.get("bytes") is not None:
            _require(_integer(item["bytes"]) and item["bytes"] == len(raw),
                     f"CAS media byte count mismatch: {ref}")


def _raw_chunks(record, artifact_dir):
    raw = sse.validate_replay_metadata(
        record, artifact_dir=artifact_dir, require_raw=True)
    capture = sse.RawSSECapture.from_metadata(record["sse_replay"], raw)
    chunks = []
    for frame_index, payload in capture.payload_frames:
        if payload == "[DONE]":
            continue
        value = _json(payload.encode("utf-8"), f"raw SSE frame {frame_index}")
        _require(isinstance(value, dict), f"raw SSE frame {frame_index} is not an object")
        chunks.append(value)
    _require(bool(chunks), "raw SSE contains no data chunks")
    return raw, chunks


def _loaded_tokenizer_digest(tokenizer_dir):
    rows = []
    for name in ("merges.txt", "tokenizer_config.json", "vocab.json"):
        raw = (Path(tokenizer_dir) / name).read_bytes()
        rows.append(f"{name}\n{len(raw)}\n{_sha(raw)}\n")
    return _sha((LOADED_TOKENIZER_FORMAT + "\n" + "".join(rows)).encode("ascii"))


def _terminal_ids(tokenizer_dir):
    generation = _json((Path(tokenizer_dir) / "generation_config.json").read_bytes(),
                       "generation_config.json")
    config = _json((Path(tokenizer_dir) / "tokenizer_config.json").read_bytes(),
                   "tokenizer_config.json")
    vocab = _json((Path(tokenizer_dir) / "vocab.json").read_bytes(), "vocab.json")
    eos = generation.get("eos_token_id")
    ids = [eos] if _integer(eos) else eos
    _require(isinstance(ids, list) and ids
             and all(_integer(value) and value >= 0 for value in ids)
             and len(ids) == len(set(ids)),
             "generation_config eos_token_id must be unique non-negative integers")
    declared = config.get("eos_token")
    if isinstance(declared, dict):
        declared = declared.get("content")
    added = config.get("added_tokens_decoder", {})
    added_ids = [entry.get("id", key) for key, entry in added.items()
                 if isinstance(entry, dict) and entry.get("content") == declared] \
        if isinstance(added, dict) else []
    added_ids = [int(value) if isinstance(value, str) and value.isdigit() else value
                 for value in added_ids]
    _require(isinstance(declared, str) and declared and isinstance(vocab, dict)
             and (( _integer(vocab.get(declared)) and vocab[declared] in ids)
                  or any(_integer(value) and value in ids for value in added_ids)),
             "tokenizer_config eos_token is not bound to generation terminal ids")
    return sorted(ids)


def _behavior_from_chunks(chunks, coverage):
    blocks = [chunk["behavior"] for chunk in chunks if "behavior" in chunk]
    if coverage == "not_exposed":
        _require(not blocks, "not_exposed response contains sampled behavior evidence")
        return None
    _require(coverage == "exposed", "behavior coverage must be exposed or not_exposed")
    _require(len(blocks) == 1 and "behavior" in chunks[-1],
             "sampled behavior requires exactly one block in the final data chunk")
    evidence = blocks[0]
    _require(isinstance(evidence, dict) and set(evidence) == BEHAVIOR_KEYS
             and evidence.get("format") == BEHAVIOR_FORMAT,
             "sampled behavior schema or fields differ from v1")
    digest = evidence["tokenizer_sha256"]
    _require(isinstance(digest, str) and len(digest) == 64
             and not set(digest) - _HEX,
             "sampled behavior tokenizer digest is invalid")
    ids, logprobs, mask = (evidence["token_ids"], evidence["logprobs"], evidence["mask"])
    _require(all(isinstance(values, list) and values for values in (ids, logprobs, mask))
             and len(ids) == len(logprobs) == len(mask),
             "sampled ids/logprobs/masks must be non-empty and equal length")
    _require(all(_integer(value) and value >= 0 for value in ids),
             "sampled token ids must be non-negative integers")
    _require(all(_finite(value) and value <= 0.0 for value in logprobs),
             "sampled logprobs must be finite and non-positive")
    _require(all(_integer(value) and value in (0, 1) for value in mask) and any(mask),
             "sampled masks must be binary with at least one supervised token")
    return evidence


def _behavior_summary(evidence):
    return {
        "status": "COVERED", "format": BEHAVIOR_FORMAT,
        "tokenizer_sha256": evidence["tokenizer_sha256"],
        "token_ids": evidence["token_ids"], "logprobs": evidence["logprobs"],
        "mask": evidence["mask"], "token_count": len(evidence["token_ids"]),
        "masked_token_count": sum(evidence["mask"]),
        "evidence_sha256": _sha(_canonical(evidence)),
    }


def _plan_and_item(manifest, summary, body, case):
    provenance = summary["provenance"]
    policy = manifest["behavior_policy"]
    behavior = {
        "policy_id": policy["policy_id"], "revision": policy["revision"],
        "model_id": body["model"], "model_revision": provenance["revision"],
        "model_sha256": provenance["model_sha256"],
        "model_config_sha256": provenance["config_sha256"],
    }
    plan = {
        "batch_id": manifest["batch_id"], "behavior_policy": behavior,
        "sampling": {key: body[key] for key in ("max_tokens", "temperature", "top_p")},
    }
    name = case["name"]
    item = {
        "group_id": "behavior-group-" + name,
        "rollout_id": "behavior-rollout-" + name,
        "rollout_index": 0, "request_sha256": _sha(_canonical(body)),
        "session_id": body["session_id"],
    }
    return plan, item


def _validate_request_response(root, manifest, summary, case, record, chunks):
    name = case["name"]
    body = _document(root, f"requests/{name}.json", case["request_file_sha256"])
    response = _document(root, f"live/responses/{name}.json",
                         case["response_file_sha256"])
    _require(isinstance(body, dict) and body.get("stream") is True
             and body.get("session_id") == record.get("session_id"),
             f"{name}: request and trajectory session/stream identity differ")
    _require(response.get("status") == 200 and isinstance(response.get("headers"), dict),
             f"{name}: retained response is not an attested HTTP 200 response")
    headers = response["headers"]
    exact_length = headers.get("content-length") == str(record["sse_replay"]["raw_bytes"])
    chunked = (headers.get("transfer-encoding") == "chunked"
               and "content-length" not in headers)
    _require((exact_length or chunked)
             and headers.get("content-type") == "text/event-stream",
             f"{name}: response transport headers do not bind raw SSE bytes")
    plan, item = _plan_and_item(manifest, summary, body, case)
    coverage = headers.get("x-umminfer-behavior-evidence")
    _require(coverage in ("exposed", "not_exposed"),
             f"{name}: invalid behavior coverage header")
    expected = online._expected_attestation(
        item, plan, http_status=200, behavior_evidence=coverage, attempt=0)
    actual = online._attestation_from_response(response, item, plan, 0)
    _require(actual == expected
             and case["attestation_sha256"] == expected["attestation_sha256"],
             f"{name}: response attestation drift")
    params, option_inputs = traj.request_parameter_references(body, None)
    inputs = traj.input_references(body["messages"], None) + option_inputs
    _require(_without_cas(record.get("request_params")) == _without_cas(params)
             and _without_cas(record.get("inputs")) == _without_cas(inputs),
             f"{name}: trajectory request/messages differ from the exact request")
    for chunk in chunks:
        _require(chunk.get("id") == record.get("request_id")
                 and chunk.get("model") == body.get("model"),
                 f"{name}: raw SSE contains stale, duplicate, or cross-record identity")
        unified = chunk.get("unified")
        if isinstance(unified, dict):
            if "request_id" in unified:
                _require(unified["request_id"] == record.get("request_id"),
                         f"{name}: raw SSE inner request identity is stale")
            _require(unified.get("session_id") == record.get("session_id"),
                     f"{name}: raw SSE inner session identity is stale")
    return body, response, coverage, expected


def _summary_case(summary, name):
    matches = [case for case in summary.get("cases", []) if case.get("case") == name]
    _require(len(matches) == 1, f"summary has missing or duplicate case: {name}")
    return matches[0]


def _validate_replay_document(root, case, record, evidence, target, summary_case):
    name = case["name"]
    replay = _document(root, f"logs/{name}-replay.stdout", case["replay_file_sha256"])
    _require(isinstance(replay, dict)
             and replay.get("behavior") == evidence
             and replay.get("finish_reason") == "stop"
             and replay.get("clean_text") == record.get("clean_text")
             and replay.get("raw_text") == target,
             f"{name}: sampled replay differs from raw evidence or reconstructed target")
    prompt_ids = replay.get("prompt_token_ids")
    _require(isinstance(prompt_ids, list) and prompt_ids
             and all(_integer(value) and value >= 0 for value in prompt_ids),
             f"{name}: sampled replay lacks an exact prompt prefix")
    forward = summary_case.get("fresh_forward")
    if forward is not None:
        _require(forward.get("returncode") == 0
                 and forward.get("raw_text_sha256") == _sha(target.encode("utf-8"))
                 and forward.get("clean_text_sha256")
                 == _sha(record["clean_text"].encode("utf-8"))
                 and forward.get("prompt_sha256") == _sha(_canonical(prompt_ids))
                 and _finite(forward.get("max_abs_logprob_diff"))
                 and 0.0 <= forward["max_abs_logprob_diff"] <= 1e-6,
                 f"{name}: retained fresh-forward identity/tolerance evidence differs")
    return forward is not None


def _token_types(stream):
    count = stream["token_count"]
    result = [set() for _ in range(count)]
    for span in stream["spans"]:
        start, end = span["token_start"], span["token_end"]
        _require(_integer(start) and _integer(end) and 0 <= start < end <= count,
                 "token_export contains an invalid span token range")
        for index in range(start, end):
            result[index].add(span["span_type"])
    return result


def _target_piece(stream, token, target_raw):
    if stream["source"] == "clean_text":
        raise AlignmentError("internal target stream selection error")
    return target_raw[token["start"]:token["end"]]


def _masked_bytes(stream, mask, target):
    target_raw = target.encode("utf-8")
    is_v2 = "normalized_bytes" in stream
    pieces = []
    for token, supervised in zip(stream["tokens"], mask):
        if not supervised:
            continue
        pieces.append(bytes.fromhex(token["piece_hex"]) if is_v2
                      else _target_piece(stream, token, target_raw))
    return b"".join(pieces)


def _validate_alignment(record, evidence, terminal_token_ids):
    block = record["token_export"]
    target, _ = tokens.reconstruct_sft_target_segments(record)
    stream = block["streams"]["sft_target"]
    clean_stream = block["streams"]["clean_text"]
    ids, mask = evidence["token_ids"], evidence["mask"]
    _require(ids == stream["token_ids"],
             "sampled ids differ from post-hoc re-encoded SFT-target ids; "
             "silent substitution is forbidden")
    _require(not set(ids) & set(terminal_token_ids),
             "terminal/EOS token ids must be excluded from sampled supervision")
    unknown = block["tokenizer"].get("unknown_token_id")
    _require(unknown is None or unknown not in ids,
             "sampled evidence contains the tokenizer unknown token")
    types = _token_types(stream)
    expected_mask = []
    visual_seqs = {event["seq"] for event in record["events"]
                   if event.get("type") == "visual_slot"}
    for index, span_types in enumerate(types):
        _require(span_types and span_types <= (VISUAL_SPANS | PROSE_SPANS),
                 f"token {index} has no supported source-span type")
        _require(not (span_types & VISUAL_SPANS and span_types & PROSE_SPANS),
                 f"token {index} straddles prose and directive/prompt/anchor spans")
        supervised = not bool(span_types & VISUAL_SPANS)
        expected_mask.append(1 if supervised else 0)
        source_events = set(stream["tokens"][index]["source_event_indices"])
        if supervised:
            _require(not source_events & visual_seqs,
                     f"supervised token {index} maps to a visual-slot boundary")
        elif visual_seqs:
            _require(bool(source_events & visual_seqs),
                     f"directive token {index} lacks its visual-slot source mapping")
    _require(mask == expected_mask,
             "behavior mask differs from clean-prose source spans; directive/prompt/"
             "anchor tokens must have mask=0")
    if visual_seqs:
        _require(0 in mask and 1 in mask,
                 "visual sampled evidence must separate directive and prose tokens")
    else:
        _require(all(mask), "text-only sampled evidence contains an unexplained mask=0")
    normalized_clean = (unicodedata.normalize("NFC", record["clean_text"]).encode("utf-8")
                        if "normalized_bytes" in stream
                        else record["clean_text"].encode("utf-8"))
    _require(_masked_bytes(stream, mask, target) == normalized_clean,
             "masked sampled token bytes do not reconstruct exact clean prose")
    sampled_clean_ids = [token for token, supervised in zip(ids, mask) if supervised]
    clean_equal = sampled_clean_ids == clean_stream["token_ids"]
    return target, {
        "sampled_token_count": len(ids), "masked_token_count": sum(mask),
        "sampled_posthoc_target_ids_equal": True,
        "masked_sampled_token_count": len(sampled_clean_ids),
        "clean_posthoc_token_count": clean_stream["token_count"],
        "clean_reencoded_ids_equal": clean_equal,
        "clean_id_relation": ("EXACT" if clean_equal
                              else "CONTEXT_BOUNDARY_DIFFERENCE_BYTES_EXACT"),
        "offline_sft_mask": "joint_full_validated_separately_not_behavior_mask",
        "logprob_numerical_recomputation": "NOT_RUN",
    }


def _validate_excluded(root, manifest, summary, case, source_record, artifact_dir):
    name = case["name"]
    raw, chunks = _raw_chunks(source_record, artifact_dir)
    _, _, coverage, _ = _validate_request_response(
        root, manifest, summary, case, source_record, chunks)
    _require(coverage == "not_exposed"
             and _behavior_from_chunks(chunks, coverage) is None,
             f"{name}: excluded case exposed sampled evidence")
    summary_case = _summary_case(summary, name)
    _require(summary_case.get("behavior") == {
        "status": "NOT_COVERED", "token_ids": None, "logprobs": None, "mask": None,
    } and summary_case.get("raw_sha256") == _sha(raw)
             and case["raw_sha256"] == _sha(raw)
             and case["attestation_sha256"] == summary_case["attestation_sha256"],
             f"{name}: excluded-case summary/raw/attestation binding differs")
    expected_state = case["terminal_state"]
    _require(source_record.get("terminal_state") == expected_state,
             f"{name}: excluded-case terminal state differs")
    finish = chunks[-1].get("choices", [{}])[0].get("finish_reason")
    if expected_state == "cancelled":
        _require(source_record["events"][-1].get("type") == "cancelled",
                 f"{name}: cancelled evidence lacks a cancelled terminal")
    else:
        _require(finish == case["finish_reason"],
                 f"{name}: excluded-case finish_reason differs")
    return {"name": name, "status": "PASS", "coverage": "NOT_COVERED",
            "terminal_state": expected_state, "finish_reason": finish,
            "raw_sha256": _sha(raw)}


def _fresh_replay(source_records, aligned_records, artifact_dir):
    with tempfile.TemporaryDirectory(prefix="umm-behavior-align-replay-") as tmp:
        root = Path(tmp)
        for label, records in (("source", source_records), ("aligned", aligned_records)):
            path = root / label / "trajectories.jsonl"
            recorder = traj.TrajectoryRecorder(path)
            for record in records:
                replayed = sse.replay_record(
                    record, recorder=recorder, artifact_dir=artifact_dir)
                _require(_source_record(replayed) == _source_record(record),
                         f"{label}: fresh replay changed its source record")
            expected = b"".join(_canonical(record) + b"\n" for record in records)
            _require(path.read_bytes() == expected,
                     f"{label}: fresh raw-SSE replay is not byte-identical")


def validate_bundle(root, tokenizer_dir=None, *, tokenizer=None, reencode=False):
    """Validate one pinned fixture bundle and return a path-free JSON report.

    Supplying an already constructed tokenizer is intended for deterministic
    fixture tests. ``reencode=True`` loads a local fast tokenizer only after all
    structural checks pass. Both modes independently re-encode token_export.
    Omitting ``tokenizer_dir`` is permitted only for read-only structural
    validation (``reencode=False`` and ``tokenizer=None``), in which case
    ``tokenizer_files`` is explicitly reported as ``NOT_RUN``.
    """
    root = Path(root)
    if tokenizer_dir is None:
        _require(not reencode and tokenizer is None,
                 "tokenizer_dir is required for tokenizer re-encoding")
    else:
        tokenizer_dir = Path(tokenizer_dir)
    manifest = _document(root, "alignment.json")
    _require(manifest.get("schema") == ALIGNMENT_SCHEMA,
             "unsupported behavior-token alignment fixture schema")
    source_info = manifest["source"]
    source_raw = _read(root, source_info["trajectory"],
                       source_info["trajectory_sha256"])
    aligned_raw = _read(root, source_info["aligned_trajectory"],
                        source_info["aligned_trajectory_sha256"])
    source_records = _jsonl(source_raw, "source trajectories")
    aligned_records = _jsonl(aligned_raw, "aligned trajectories")
    _require(len(source_records) == len(aligned_records),
             "source/aligned trajectory record counts differ")
    identities = set()
    for index, (source, aligned) in enumerate(zip(source_records, aligned_records)):
        _require(_source_record(aligned) == source,
                 f"record {index}: token export changed the source record")
        identity = (source.get("session_id"), source.get("request_id"))
        _require(all(isinstance(value, str) and value for value in identity)
                 and identity not in identities,
                 f"record {index}: duplicate or invalid source identity")
        identities.add(identity)
    artifact_dir = _safe_path(root, source_info["artifact_dir"])
    for record in source_records:
        _validate_cas(record, artifact_dir)

    summary = _document(root, source_info["summary"], source_info["summary_sha256"])
    summary_schema = summary.get("schema")
    retained_task42 = manifest.get("evidence_class") == "RETAINED_REAL_TASK42"
    _require(summary.get("status") == "PASS"
             and (summary_schema == "umminfer-behavior-logprobs-live-evidence-v1"
                  or (retained_task42 and summary_schema is None
                      and summary.get("provenance", {}).get("source_commit")
                      == manifest.get("source_commit"))),
             "retained behavior summary is not an accepted v1 evidence record")
    checksums = _read(root, source_info["checksums"], source_info["checksums_sha256"])
    _require(_sha(summary and _canonical(summary) + b"\n")
             == source_info["summary_sha256"] and bool(checksums),
             "source summary/checksum binding failed")

    tokenizer_info = manifest["tokenizer"]
    _require(isinstance(tokenizer_info, dict),
             "alignment manifest lacks tokenizer metadata")
    expected_identity = tokenizer_info["identity"]
    expected_files = tokenizer_info["files"]
    _require(isinstance(expected_files, list) and bool(expected_files),
             "alignment manifest lacks tokenizer available-file metadata")
    if tokenizer_dir is not None:
        available_files = tokens.hash_available_tokenizer_files(tokenizer_dir)
        terminal_token_ids = _terminal_ids(tokenizer_dir)
        _require(terminal_token_ids == tokenizer_info["terminal_token_ids"],
                 "terminal-token declaration differs from the pinned fixture")
        loaded_digest = _loaded_tokenizer_digest(tokenizer_dir)
        _require(loaded_digest == tokenizer_info["behavior_sha256"],
                 "exact loaded tokenizer files differ from sampled evidence")
    else:
        available_files = expected_files
        terminal_token_ids = tokenizer_info["terminal_token_ids"]
        _require(isinstance(terminal_token_ids, list) and bool(terminal_token_ids)
                 and all(_integer(value) and value >= 0 for value in terminal_token_ids)
                 and len(terminal_token_ids) == len(set(terminal_token_ids))
                 and terminal_token_ids == sorted(terminal_token_ids),
                 "pinned terminal_token_ids are invalid")
        loaded_digest = tokenizer_info["behavior_sha256"]
        _require(isinstance(loaded_digest, str) and len(loaded_digest) == 64
                 and not set(loaded_digest) - _HEX,
                 "pinned behavior tokenizer digest is invalid")
    for index, aligned in enumerate(aligned_records):
        if "token_export" not in aligned:
            continue
        identity = aligned["token_export"]["tokenizer"]
        _require(identity["available_files"] == available_files
                 == expected_files,
                 f"record {index}: tokenizer identity/files are incomplete or changed")
        result = tokens.validate_record_token_export(
            aligned, tokenizer_dir=tokenizer_dir,
            expected_tokenizer_id=expected_identity["id"],
            expected_tokenizer_revision=expected_identity["revision"],
            expected_tokenization_version=expected_identity["tokenization_version"])
        _require(result["status"] == "PASS" and not result["tokenizer_reencoded"],
                 f"record {index}: structural token_export validation failed")

    accepted_reports = []
    retained_forward = []
    # This loop completes every structural rejection before optional ML import.
    for case in manifest["cases"]:
        name = case["name"]
        index = case["record_index"]
        _require(_integer(index) and 0 <= index < len(source_records),
                 f"{name}: invalid record index")
        source, aligned = source_records[index], aligned_records[index]
        raw, chunks = _raw_chunks(source, artifact_dir)
        _, _, coverage, attestation = _validate_request_response(
            root, manifest, summary, case, source, chunks)
        _require(coverage == "exposed", f"{name}: accepted alignment lacks behavior evidence")
        evidence = _behavior_from_chunks(chunks, coverage)
        summary_case = _summary_case(summary, name)
        _require(source.get("terminal_state") == "completed"
                 and chunks[-1].get("choices", [{}])[0].get("finish_reason") == "stop",
                 f"{name}: truncated/cancelled/non-stop evidence cannot be supervised")
        _require(summary_case.get("behavior") == _behavior_summary(evidence)
                 and summary_case.get("raw_sha256") == _sha(raw)
                 and summary_case.get("attestation_sha256")
                 == attestation["attestation_sha256"]
                 and case.get("raw_sha256") == _sha(raw)
                 and case.get("evidence_sha256") == _sha(_canonical(evidence)),
                 f"{name}: raw/summary/manifest evidence hash binding differs")
        _require(evidence["tokenizer_sha256"] == loaded_digest,
                 f"{name}: sampled evidence tokenizer identity differs from exact files")
        _require("token_export" in aligned,
                 f"{name}: accepted sampled evidence lacks token_export supervision")
        target = tokens.reconstruct_sft_target_segments(aligned)[0]
        retained_forward.append(_validate_replay_document(
            root, case, source, evidence, target, summary_case))
        target, alignment = _validate_alignment(aligned, evidence, terminal_token_ids)
        accepted_reports.append({
            "name": name, "status": "PASS", "raw_sha256": _sha(raw),
            "evidence_sha256": _sha(_canonical(evidence)),
            "attestation_sha256": attestation["attestation_sha256"],
            "source_record_sha256": aligned["token_export"]["source_record_sha256"],
            "export_sha256": aligned["token_export"]["export_sha256"],
            **alignment,
        })

    excluded_reports = [
        _validate_excluded(root, manifest, summary, case,
                           source_records[case["record_index"]], artifact_dir)
        for case in manifest.get("excluded_cases", [])]

    if reencode and tokenizer is None:
        tokenizer = tokens._load_real_tokenizer(tokenizer_dir)
    strong = tokenizer is not None
    if strong:
        for index, aligned in enumerate(aligned_records):
            if "token_export" not in aligned:
                continue
            result = tokens.validate_record_token_export(
                aligned, tokenizer=tokenizer, tokenizer_dir=tokenizer_dir,
                expected_tokenizer_id=expected_identity["id"],
                expected_tokenizer_revision=expected_identity["revision"],
                expected_tokenization_version=expected_identity["tokenization_version"])
            _require(result["status"] == "PASS" and result["tokenizer_reencoded"],
                     f"record {index}: strong tokenizer re-encoding failed")

    _fresh_replay(source_records, aligned_records, artifact_dir)
    return {
        "schema": "umminfer-behavior-token-alignment-report-v1",
        "status": "PASS", "evidence_class": manifest["evidence_class"],
        "validation_level": "STRONG" if strong else "STRUCTURAL",
        "tokenizer_files": "PASS" if tokenizer_dir is not None else "NOT_RUN",
        "tokenizer_reencode": "PASS" if strong else "NOT_RUN",
        "raw_sse_fresh_replay": "PASS_BYTE_IDENTICAL",
        "accepted_cases": accepted_reports, "excluded_cases": excluded_reports,
        "claims": {
            "new_model_inference": "NOT_RUN", "new_logprob_recomputation": "NOT_RUN",
            "online_optimizer": "NOT_RUN", "model_quality": "NOT_RUN",
            "serving": "NOT_RUN", "gpu": "NOT_RUN",
            "retained_forward_evidence": (
                "RETAINED_TASK42_PASS_NOT_RERUN" if retained_forward
                and all(retained_forward) else "NOT_COVERED"),
        },
    }


def _write_new(path, payload):
    path = Path(path)
    _require(not path.exists(), "report output already exists; refusing overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path, help="pinned alignment fixture directory")
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--reencode", action="store_true",
                        help="after structural PASS, re-encode with local fast tokenizer")
    parser.add_argument("--report", type=Path, help="new report JSON; no file on failure")
    args = parser.parse_args(argv)
    try:
        report = validate_bundle(args.fixture, args.tokenizer_dir,
                                 reencode=args.reencode)
        payload = _canonical(report) + b"\n"
        if args.report is not None:
            _write_new(args.report, payload)
        sys.stdout.buffer.write(payload)
        return 0
    except (AlignmentError, tokens.TokenExportError, sse.ReplayError,
            OSError, KeyError, IndexError, TypeError, ValueError) as exc:
        print(f"behavior-token alignment error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
