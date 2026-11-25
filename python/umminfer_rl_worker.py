# SPDX-License-Identifier: Apache-2.0
"""Separate training worker: trajectory validation and hash-chained state.

Serving never imports this module. The metadata-only ``train`` verb reports
``optimizer_status=NOT_RUN``; only a successfully executed ``train_step`` may
report ``EXECUTED``. Schema v1 exports remain valid, while schema v2 adds strict
narrative, terminal, chunk, slot, and content-addressed-artifact validation.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

# Some interpreters' -I mode omits the script directory from sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

WORKER_PROTOCOL_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
SUPPORTED_TRAJECTORY_SCHEMA_VERSIONS = frozenset((1, 2))

_TERMINAL_STATES = frozenset(("completed", "failed", "cancelled"))
_TERMINAL_EVENT_FOR_STATE = {
    "completed": "complete", "failed": "error", "cancelled": "cancelled",
}
_FINAL_LEG_STATES = frozenset(("completed", "failed", "aborted"))
_HEX_256 = re.compile(r"[0-9a-f]{64}\Z")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _sha256(value):
    return isinstance(value, str) and _HEX_256.fullmatch(value) is not None


def _text_sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _verify_cas(root, ref, expected_sha, expected_bytes, label):
    if not isinstance(ref, str) or not ref or Path(ref).name != ref:
        raise ValueError(f"{label} has an invalid relative artifact ref")
    if not _sha256(expected_sha) or not ref.startswith(expected_sha + "."):
        raise ValueError(f"{label} ref is not bound to its SHA-256")
    path = root / ref
    if not path.is_file():
        raise ValueError(f"{label} artifact is missing: {ref}")
    raw = path.read_bytes()
    if expected_bytes is not None and (
            not _integer(expected_bytes) or expected_bytes != len(raw)):
        raise ValueError(f"{label} artifact byte count does not recompute")
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError(f"{label} artifact SHA-256 does not recompute")


def _validate_inputs(record, artifact_root):
    inputs = record.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError("missing inputs list")
    for index, ref in enumerate(inputs):
        if not isinstance(ref, dict) or not isinstance(ref.get("kind"), str):
            raise ValueError(f"input {index} is malformed")
        if ref["kind"] == "text":
            if not isinstance(ref.get("text"), str) \
                    or ref.get("sha256") != _text_sha(ref["text"]):
                raise ValueError(f"input {index} text SHA-256 does not recompute")
            continue
        digest = ref.get("sha256")
        if digest is None:
            # Remote URL references are deliberately unfetched and unhashed.
            if not isinstance(ref.get("ref"), str) \
                    or ref["ref"].startswith("data:"):
                raise ValueError(f"input {index} has neither bytes nor a remote URL")
            continue
        _verify_cas(artifact_root, ref.get("ref"), digest,
                    ref.get("bytes"), f"input {index}")


def _validate_request_params(record, artifact_root):
    params = record.get("request_params")
    if not isinstance(params, dict):
        raise ValueError("request_params must be an object")
    inputs = record["inputs"]
    for section, field in (("image", "image"), ("video", "image"),
                           ("audio", "ref_audio")):
        options = params.get(section)
        if not isinstance(options, dict) or field not in options:
            continue
        ref = options[field]
        if not isinstance(ref, dict):
            raise ValueError(f"request_params.{section}.{field} embeds media")
        if ref not in inputs:
            raise ValueError(
                f"request_params.{section}.{field} lacks an input reference")
        _verify_cas(artifact_root, ref.get("ref"), ref.get("sha256"),
                    ref.get("bytes"), f"request_params.{section}.{field}")


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


def _validate_terminal(record):
    events = record["events"]
    state = record["terminal_state"]
    terminal_indexes = [index for index, event in enumerate(events)
                        if event.get("type") in ("complete", "error", "cancelled")]
    if len(terminal_indexes) != 1:
        raise ValueError("v2 requires exactly one terminal event")
    terminal_index = terminal_indexes[0]
    if terminal_index != len(events) - 1:
        raise ValueError("terminal event is not final")
    if record.get("terminal_event_seq") != terminal_index:
        raise ValueError("terminal_event_seq does not identify the terminal")
    terminal = events[terminal_index]
    if terminal.get("type") != _TERMINAL_EVENT_FOR_STATE[state]:
        raise ValueError("terminal_state does not match terminal event")
    error = record.get("error")
    if state in ("completed", "cancelled") and error is not None:
        raise ValueError(f"{state} record carries an error")
    if state == "failed" and (not isinstance(error, str) or not error):
        raise ValueError("failed record lacks an error message")

    legs = terminal.get("legs", [])
    if not isinstance(legs, list) or record.get("terminal_legs") != legs:
        raise ValueError("terminal_legs differs from the terminal event")
    for leg in legs:
        if not isinstance(leg, dict) or not isinstance(leg.get("stage"), str) \
                or leg.get("state") not in _FINAL_LEG_STATES:
            raise ValueError("terminal leg has an invalid stage or non-final state")
    return terminal


def _validate_slots(record):
    events = record["events"]
    slots = record.get("visual_slots")
    expected = _slot_records(events)
    if slots != expected and slots != _slot_records(events, legacy=True):
        raise ValueError("visual_slots differs from visual_slot events")
    by_id = {}
    for expected_index, slot in enumerate(slots):
        slot_id = slot.get("slot_id")
        if not isinstance(slot_id, str) or not slot_id \
                or slot_id in by_id or slot.get("slot_index") != expected_index:
            raise ValueError("visual slot ids/indices are not unique and gap-free")
        if slot.get("media_type") not in ("image", "video") \
                or slot.get("stage") != slot["media_type"]:
            raise ValueError(f"visual slot stage/media mismatch for {slot_id}")
        if not isinstance(slot.get("prompt"), str):
            raise ValueError(f"visual slot prompt is not text for {slot_id}")
        if not _integer(slot.get("text_offset")) or slot["text_offset"] < 0:
            raise ValueError(f"visual slot text_offset is invalid for {slot_id}")
        by_id[slot_id] = slot
        matches = [leg for leg in record["terminal_legs"]
                   if leg.get("slot_id") == slot_id]
        if len(matches) != 1 or matches[0]["stage"] != slot["stage"]:
            raise ValueError(f"visual slot lacks one matching terminal leg: {slot_id}")
        if not slot["prompt"].strip() and not (
                slot.get("anchor_explicit", False)
                and matches[0]["state"] in ("failed", "aborted")):
            raise ValueError(f"visual slot prompt is empty for {slot_id}")
    if record.get("leg_errors") != _leg_error_records(events) \
            and record.get("leg_errors") != _leg_error_records(events, legacy=True):
        raise ValueError("leg_errors differs from leg_error events")
    return by_id


def _validate_artifacts(record, artifact_root, slots):
    groups = {}
    for event in record["events"]:
        if event.get("type") != "artifact":
            continue
        descriptor = event.get("artifact")
        if not isinstance(descriptor, dict) or "data" in descriptor:
            raise ValueError("artifact event is malformed or embeds media bytes")
        artifact_id = descriptor.get("artifact_id")
        index, total = descriptor.get("index"), descriptor.get("total")
        if not isinstance(artifact_id, str) or not artifact_id \
                or not _integer(index) or not _integer(total) or total <= 0 \
                or not 0 <= index < total:
            raise ValueError("artifact chunk identity/index is invalid")
        if not _integer(descriptor.get("chunk_bytes")) \
                or descriptor["chunk_bytes"] < 0 \
                or not _sha256(descriptor.get("chunk_sha256")):
            raise ValueError(f"artifact chunk hash/bytes are invalid: {artifact_id}")
        final = index + 1 == total
        if final and (not _integer(descriptor.get("bytes"))
                      or not _sha256(descriptor.get("sha256"))):
            raise ValueError(f"final artifact chunk lacks bytes/SHA-256: {artifact_id}")
        if not final and ("bytes" in descriptor or "sha256" in descriptor):
            raise ValueError(f"non-final artifact chunk carries final hash: {artifact_id}")
        group = groups.setdefault(artifact_id, {
            "total": total, "descriptors": {},
            "identity": _artifact_identity(descriptor),
        })
        if group["total"] != total \
                or group["identity"] != _artifact_identity(descriptor):
            raise ValueError(f"artifact metadata/total changed: {artifact_id}")
        if index in group["descriptors"]:
            raise ValueError(f"duplicate artifact chunk: {artifact_id}")
        group["descriptors"][index] = descriptor

        slot_id = descriptor.get("slot_id")
        if slot_id is not None:
            if slot_id not in slots:
                raise ValueError(f"artifact binds to an unknown slot: {slot_id}")
            slot = slots[slot_id]
            if descriptor.get("slot_index") != slot["slot_index"] \
                    or descriptor.get("media_type") != slot["media_type"] \
                    or event.get("stage") != slot["stage"]:
                raise ValueError(f"artifact does not match its slot: {slot_id}")
        elif slots and descriptor.get("media_type") in ("image", "video"):
            raise ValueError("narrative visual artifact lacks slot attribution")

    outputs = record.get("artifacts")
    if not isinstance(outputs, list):
        raise ValueError("missing artifacts list")
    output_by_id = {}
    for output in outputs:
        if not isinstance(output, dict) or not isinstance(
                output.get("artifact_id"), str) \
                or output["artifact_id"] in output_by_id:
            raise ValueError("output artifact ids are missing or duplicated")
        output_by_id[output["artifact_id"]] = output

    complete_ids = []
    for artifact_id, group in groups.items():
        total = group["total"]
        descriptors = group["descriptors"]
        complete = sorted(descriptors) == list(range(total))
        if not complete:
            if record["terminal_state"] == "completed":
                raise ValueError(f"completed turn has missing chunks: {artifact_id}")
            if artifact_id in output_by_id:
                raise ValueError(f"partial artifact was published: {artifact_id}")
            continue
        complete_ids.append(artifact_id)
        output = output_by_id.get(artifact_id)
        if output is None:
            raise ValueError(f"complete artifact has no output descriptor: {artifact_id}")
        final = descriptors[total - 1]
        if output.get("chunk_count") != total \
                or output.get("bytes") != final.get("bytes") \
                or output.get("sha256") != final.get("sha256"):
            raise ValueError(f"output/final chunk descriptor mismatch: {artifact_id}")
        if sum(item["chunk_bytes"] for item in descriptors.values()) \
                != final["bytes"]:
            raise ValueError(f"artifact chunk bytes do not sum to output: {artifact_id}")
        chunk_hashes = b"".join(
            bytes.fromhex(descriptors[index]["chunk_sha256"])
            for index in range(total))
        chunks_sha256 = hashlib.sha256(chunk_hashes).hexdigest()
        if output.get("chunks_sha256") != chunks_sha256:
            raise ValueError(f"artifact chunk hash chain mismatch: {artifact_id}")
        for key, value in group["identity"].items():
            if output.get(key) != value:
                raise ValueError(f"output metadata mismatch {key}: {artifact_id}")
        for descriptor in descriptors.values():
            if descriptor.get("ref") != output.get("ref"):
                raise ValueError(f"artifact chunk ref mismatch: {artifact_id}")
        _verify_cas(artifact_root, output.get("ref"), output.get("sha256"),
                    output.get("bytes"), f"output {artifact_id}")

    if set(output_by_id) != set(complete_ids) \
            or [item["artifact_id"] for item in outputs] != complete_ids:
        raise ValueError("artifacts list does not match complete event groups")


def _validate_linkages(record, slots):
    """Validate mainline i2v_chain links and their content addresses.

    Explicit references may be unresolved only on a failed/aborted leg. The
    mainline implicit T2V fallback is also unresolved by design. Any resolved
    digest must bind an earlier image slot's completed CAS artifact.
    """
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
            raise ValueError(f"anchor_explicit is not boolean for {slot_id}")
        if slot.get("continuity_mode") not in (None, "prompt_anchor", "i2v_chain"):
            raise ValueError(f"invalid continuity_mode for {slot_id}")
        if anchor is None:
            if explicit:
                if leg["state"] == "completed" or outputs \
                        or "anchor_slot_index" in leg or "anchor_sha256" in leg:
                    raise ValueError(f"unresolved explicit slot has output: {slot_id}")
                continue
            if "anchor_slot_index" in leg or "anchor_sha256" in leg \
                    or any("anchor_slot_index" in output
                           or "anchor_sha256" in output for output in outputs):
                raise ValueError(f"unanchored slot carries anchor metadata: {slot_id}")
            continue
        if not _integer(anchor) or anchor < 0 \
                or slot.get("continuity_mode") != "i2v_chain":
            raise ValueError(f"invalid visual anchor for {slot_id}")
        if explicit or "anchor_slot_index" in leg or "anchor_sha256" in leg:
            if leg.get("anchor_slot_index") != anchor \
                    or leg.get("anchor_explicit", False) != explicit:
                raise ValueError(f"visual/terminal anchor mismatch for {slot_id}")
        digest = leg.get("anchor_sha256")
        if digest is not None and not _sha256(digest):
            raise ValueError(f"invalid anchor SHA-256 for {slot_id}")
        if explicit and leg["state"] == "completed" and not digest:
            raise ValueError(f"completed explicit I2V slot lacks digest: {slot_id}")
        if explicit and leg["state"] == "completed" and not outputs:
            raise ValueError(f"completed explicit I2V has no artifact: {slot_id}")

        for output in outputs:
            if output.get("media_type") != "video":
                raise ValueError(f"anchored artifact modality is invalid: {slot_id}")
            if explicit or "anchor_slot_index" in output \
                    or "anchor_sha256" in output:
                if output.get("anchor_slot_index") != anchor \
                        or output.get("anchor_explicit", False) != explicit:
                    raise ValueError(f"anchored artifact metadata is invalid: {slot_id}")
            output_digest = output.get("anchor_sha256")
            if output_digest is not None and not _sha256(output_digest):
                raise ValueError(f"invalid artifact anchor digest: {slot_id}")
            if explicit and output_digest is None:
                raise ValueError(f"explicit I2V artifact lacks digest: {slot_id}")
            if digest is not None and output_digest != digest:
                raise ValueError(f"artifact/terminal anchor mismatch: {slot_id}")
        if digest is None and not any("anchor_sha256" in output
                                      for output in outputs):
            if explicit and leg["state"] == "completed":
                raise ValueError(f"completed explicit I2V has no artifact: {slot_id}")
            continue

        resolved = digest if digest is not None else outputs[0]["anchor_sha256"]
        source = by_index.get(anchor)
        if source is None or anchor >= slot["slot_index"] \
                or source["media_type"] != "image" \
                or slot["media_type"] != "video":
            raise ValueError(f"resolved anchor is not an earlier image: {slot_id}")
        source_outputs = [item for item in outputs_by_slot.get(
                              source["slot_id"], [])
                          if item.get("media_type") == "image"]
        if len(source_outputs) != 1 \
                or source_outputs[0].get("sha256") != resolved:
            raise ValueError(f"anchor CAS artifact does not match {slot_id}")


def _validate_v2(record, artifact_root):
    events = record["events"]
    server_seqs = [event.get("server_seq") for event in events
                   if event.get("server_seq") is not None]
    if any(not _integer(seq) for seq in server_seqs) \
            or server_seqs != list(range(len(server_seqs))):
        raise ValueError("server_seq is not gap-free from zero")
    terminal = _validate_terminal(record)
    clean_text = record.get("clean_text")
    if not isinstance(clean_text, str) \
            or record.get("clean_text_sha256") != _text_sha(clean_text):
        raise ValueError("clean_text_sha256 does not recompute")
    if clean_text != _expected_clean_text(record, terminal):
        raise ValueError("clean_text does not match narrative deltas")
    output_parts = [event.get("text", "") for event in events
                    if event.get("type") in
                    ("asr_final", "vlm_delta", "output_delta")]
    expected_output = "".join(output_parts) if output_parts else (
        terminal.get("text", "") if terminal.get("type") == "complete" else "")
    if record["output_text"] != expected_output:
        raise ValueError("output_text does not match output events")
    _validate_inputs(record, artifact_root)
    _validate_request_params(record, artifact_root)
    slots = _validate_slots(record)
    _validate_artifacts(record, artifact_root, slots)
    _validate_linkages(record, slots)


def validate_trajectory(path, *, artifact_dir=None):
    """Parse and validate v1/v2 JSONL, returning its records.

    ``artifact_dir`` defaults to ``artifacts`` beside the export. V2 refs are
    always checked against CAS bytes; v1 keeps its historical validation path.
    """
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("trajectory record must be a JSON object")
                records.append(value)
    if not records:
        raise ValueError("trajectory export contains no records")
    artifact_root = Path(artifact_dir) if artifact_dir is not None \
        else Path(path).parent / "artifacts"
    for index, record in enumerate(records):
        prefix = f"record {index}: "
        try:
            version = record.get("schema_version")
            if not _integer(version) \
                    or version not in SUPPORTED_TRAJECTORY_SCHEMA_VERSIONS:
                raise ValueError("unsupported or missing schema_version")
            if record.get("terminal_state") not in _TERMINAL_STATES:
                raise ValueError("invalid terminal_state")
            events = record.get("events")
            if not isinstance(events, list):
                raise ValueError("missing events list")
            for event_index, event in enumerate(events):
                if not isinstance(event, dict) \
                        or not _integer(event.get("seq")) \
                        or event["seq"] != event_index:
                    raise ValueError(f"event seq gap at {event_index}")
            output_text = record.get("output_text")
            if not isinstance(output_text, str) \
                    or record.get("output_sha256") != _text_sha(output_text):
                raise ValueError("output_sha256 does not recompute")
            if version == 2:
                _validate_v2(record, artifact_root)
        except ValueError as exc:
            raise ValueError(prefix + str(exc)) from exc
    return records


def _reward_name(records):
    names = set()
    for record in records:
        block = record.get("reward")
        if block is None:
            continue
        if not isinstance(block, dict):
            raise ValueError("reward block must be an object or null")
        name = block.get("name")
        if name is not None and not isinstance(name, str):
            raise ValueError("reward name must be a string or null")
        if name is not None:
            names.add(name)
    if len(names) > 1:
        raise ValueError(
            "trajectory mixes reward names: " + ", ".join(sorted(names)))
    return next(iter(names), None)


class CheckpointStore:
    """Monotonic checkpoints with a content hash chain, no mutable state file."""

    def __init__(self, state_dir):
        self.checkpoint_dir = Path(state_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._latest_sequence, self._latest_hash = self._verify_chain()

    @property
    def next_sequence(self):
        return self._latest_sequence + 1

    def _verify_chain(self):
        previous_hash = None
        latest_sequence = -1
        for path in sorted(self.checkpoint_dir.glob("ckpt-*.json")):
            sidecar = Path(str(path) + ".sha256")
            if not sidecar.exists():
                raise ValueError(f"missing checksum sidecar for {path.name}")
            actual = sha256_file(path)
            if sidecar.read_text(encoding="utf-8").strip() != actual:
                raise ValueError(f"checksum mismatch for {path.name}")
            manifest = json.loads(path.read_text(encoding="utf-8"))
            expected_sequence = latest_sequence + 1
            if manifest.get("sequence") != expected_sequence \
                    or path.name != f"ckpt-{expected_sequence:08d}.json":
                raise ValueError(f"checkpoint sequence mismatch at {path.name}")
            if manifest.get("base_checkpoint_sha256") != previous_hash:
                raise ValueError(
                    f"hash chain broken at {path.name}: parent hash mismatch")
            previous_hash = actual
            latest_sequence = manifest["sequence"]
        return latest_sequence, previous_hash

    def write_checkpoint(self, *, trajectory_path, record_count,
                         trajectory_sha256, reward_name,
                         algorithm=None, weights_ref=None,
                         weights_sha256=None, hyperparameters=None,
                         metrics=None, optimizer_status="NOT_RUN"):
        if optimizer_status not in ("NOT_RUN", "EXECUTED"):
            raise ValueError("invalid optimizer_status")
        if optimizer_status == "NOT_RUN" and (
                algorithm is not None or weights_ref is not None):
            raise ValueError("NOT_RUN checkpoint cannot claim algorithm weights")
        sequence = self.next_sequence
        manifest = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "worker_protocol_version": WORKER_PROTOCOL_VERSION,
            "sequence": sequence,
            "optimizer_status": optimizer_status,
            "algorithm": algorithm,
            "weights_ref": weights_ref,
            "weights_sha256": weights_sha256,
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "reward": {"name": reward_name},
            "source": {
                "trajectory_path": str(trajectory_path),
                "record_count": record_count,
                "trajectory_sha256": trajectory_sha256,
            },
            "base_checkpoint_sha256": self._latest_hash,
        }
        path = self.checkpoint_dir / f"ckpt-{sequence:08d}.json"
        if path.exists():
            raise ValueError(f"checkpoint sequence conflict: {path.name}")
        content = json.dumps(manifest, sort_keys=True,
                             ensure_ascii=False, allow_nan=False)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(content + "\n", encoding="utf-8")
        os.replace(tmp, path)
        digest = sha256_file(path)
        Path(str(path) + ".sha256").write_text(digest + "\n",
                                                encoding="utf-8")
        self._latest_sequence = sequence
        self._latest_hash = digest
        return sequence, path


class TrainingWorker:
    """NDJSON request handler; optimizer imports remain lazy."""

    def __init__(self, state_dir):
        self.store = CheckpointStore(state_dir)
        self._trainer = None
        self._trainer_key = None

    def handle(self, request):
        req_type = request.get("type")
        if req_type == "ping":
            return {"type": "pong",
                    "worker_protocol_version": WORKER_PROTOCOL_VERSION}
        if req_type == "train":
            return self._handle_train(request)
        if req_type == "train_step":
            return self._handle_train_step(request)
        return {"type": "error", "task_id": request.get("task_id"),
                "error": f"unknown request type: {req_type!r}"}

    def _handle_train(self, request):
        task_id = request.get("task_id")
        try:
            trajectory_path = request["trajectory_path"]
            records = validate_trajectory(
                trajectory_path, artifact_dir=request.get("artifact_dir"))
            reward_name = _reward_name(records)
            sequence, path = self.store.write_checkpoint(
                trajectory_path=trajectory_path,
                record_count=len(records),
                trajectory_sha256=sha256_file(trajectory_path),
                reward_name=reward_name,
                optimizer_status="NOT_RUN")
        except KeyError as exc:
            return {"type": "error", "task_id": task_id,
                    "optimizer_status": "NOT_RUN",
                    "error": f"missing field: {exc.args[0]}"}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return {"type": "error", "task_id": task_id,
                    "optimizer_status": "NOT_RUN", "error": str(exc)}
        return {"type": "train_result", "task_id": task_id, "status": "ok",
                "optimizer_status": "NOT_RUN", "sequence": sequence,
                "checkpoint_path": str(path),
                "worker_protocol_version": WORKER_PROTOCOL_VERSION}

    def _handle_train_step(self, request):
        task_id = request.get("task_id")
        optimizer_executed = False
        try:
            trajectory_path = request["trajectory_path"]
            model_path = request["model_path"]
            hyperparameters = dict(request.get("hyperparameters") or {})
            records = validate_trajectory(
                trajectory_path, artifact_dir=request.get("artifact_dir"))
            self._require_scores(records)
            reward_name = _reward_name(records)
            trainer_key = (model_path, json.dumps(
                hyperparameters, sort_keys=True))
            if self._trainer is None:
                from umminfer_rl_train import ReinforceTrainer
                self._trainer = ReinforceTrainer(model_path, **hyperparameters)
                self._trainer_key = trainer_key
            elif self._trainer_key != trainer_key:
                raise ValueError(
                    "trainer config is fixed for this state directory; "
                    "open a new one for different model/hyperparameters")
            metrics = self._trainer.train_step(records)
            optimizer_executed = True
            sequence = self.store.next_sequence
            weights_path = (Path(self.store.checkpoint_dir.parent)
                            / "weights" / f"weights-{sequence:08d}.pt")
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            self._trainer.save_weights(weights_path)
            sequence, path = self.store.write_checkpoint(
                trajectory_path=trajectory_path,
                record_count=len(records),
                trajectory_sha256=sha256_file(trajectory_path),
                reward_name=reward_name,
                algorithm=self._trainer.config["algorithm"],
                weights_ref=str(weights_path),
                weights_sha256=sha256_file(weights_path),
                hyperparameters=self._trainer.config,
                metrics=metrics,
                optimizer_status="EXECUTED")
        except KeyError as exc:
            return {"type": "error", "task_id": task_id,
                    "optimizer_status": "EXECUTED" if optimizer_executed else "NOT_RUN",
                    "error": f"missing field: {exc.args[0]}"}
        except (OSError, ValueError, TypeError, json.JSONDecodeError,
                RuntimeError, ImportError) as exc:
            return {"type": "error", "task_id": task_id,
                    "optimizer_status": "EXECUTED" if optimizer_executed else "NOT_RUN",
                    "error": str(exc)}
        return {"type": "train_step_result", "task_id": task_id,
                "status": "ok", "optimizer_status": "EXECUTED",
                "sequence": sequence, "checkpoint_path": str(path),
                "weights_path": str(weights_path), "metrics": metrics,
                "worker_protocol_version": WORKER_PROTOCOL_VERSION}

    @staticmethod
    def _require_scores(records):
        for index, record in enumerate(records):
            block = record.get("reward")
            score = block.get("score") if isinstance(block, dict) else None
            valid = isinstance(score, (int, float)) \
                and not isinstance(score, bool)
            if valid:
                try:
                    valid = math.isfinite(score)
                except OverflowError:
                    valid = False
            if not valid:
                raise ValueError(
                    f"record {index}: train_step needs a finite numeric reward score")


def main(argv=None):
    parser = argparse.ArgumentParser(description="ummInfer RL training worker")
    parser.add_argument("--state-dir", required=True)
    args = parser.parse_args(argv)
    worker = TrainingWorker(args.state_dir)
    sys.stdout.write(json.dumps({
        "type": "hello", "worker_protocol_version": WORKER_PROTOCOL_VERSION,
    }) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            response = {"type": "error", "error": f"invalid JSON: {exc}"}
        else:
            response = worker.handle(request)
        sys.stdout.write(json.dumps(response, sort_keys=True,
                                    allow_nan=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
