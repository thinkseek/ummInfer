#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real public-SSE capture/replay gate for one isolated Qwen3.8 service.

The caller starts the service on an ephemeral loopback port, using physical
GPU 0 only. This script never starts or stops a service and never contacts the
resident port. Visual output bytes must come from an explicitly labelled
contract-fixture stage; they are transport evidence, never model-quality
 evidence. Real model text and image-input processing still use Qwen3.8.

Example (after the isolated service is ready):
  UMMINFER_LIVE_OPT_IN=1 python3 -I tests/sse_replay_live.py \
    --url 127.0.0.1:PORT --model qwen3-native --model-dir /path/to/qwen3.8 \
    --source-commit COMMIT --replay-commit COMMIT \
    --server-binary /path/to/umm-infer \
    --stage-binary /path/to/umm-native-stage-qwen38 \
    --out-dir /tmp/umminfer-sse-replay-live

Without UMMINFER_LIVE_OPT_IN=1 the gate reports NOT_RUN and opens no socket.
"""

import argparse
import base64
import copy
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
from urllib.parse import urlparse
import zlib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import umminfer_rl_trajectory as traj  # noqa: E402
import umminfer_rl_worker as worker  # noqa: E402
import umminfer_stage_c_pipeline as pipeline  # noqa: E402
from umminfer_sse_replay import (  # noqa: E402
    LogicalTickClock, PerfCounterClock, RawSSECapture, ReplayError,
)

EXPECTED_REPLAY_FORMAT = "umminfer-raw-sse-replay-v1"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def canonical(value):
    return traj.canonical_json(value)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(8 << 20)
            if not block:
                break
            size += len(block)
            digest.update(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def file_tree(root):
    root = Path(root)
    return {str(path.relative_to(root)): sha256_file(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def fixture_png():
    """Return a deterministic 64x64 PNG used only as a visual contract input."""
    def chunk(name, body):
        return (struct.pack(">I", len(body)) + name + body
                + struct.pack(">I", zlib.crc32(name + body) & 0xFFFFFFFF))

    rows = bytearray()
    for y in range(64):
        rows.append(0)
        for x in range(64):
            rows.extend((255 if x < 32 else 20,
                         255 if y >= 32 else 30,
                         40 if (x // 8 + y // 8) % 2 else 220))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(rows), 9))
            + chunk(b"IEND", b""))


def model_provenance(model_dir):
    model_dir = Path(model_dir)
    require(model_dir.is_dir(), f"model directory is missing: {model_dir}")
    config = model_dir / "config.json"
    require(config.is_file(), f"model config is missing: {config}")
    parsed = json.loads(config.read_text(encoding="utf-8"))
    config_files = {}
    weights = {}
    for path in sorted(model_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix == ".safetensors":
            weights[path.name] = sha256_file(path)
        elif path.suffix in (".json", ".jinja", ".txt"):
            config_files[path.name] = sha256_file(path)
    require(weights, "no available safetensors weights were found")
    return {
        "model_architectures": parsed.get("architectures", []),
        "model_type": parsed.get("model_type"),
        "dtype": parsed.get("dtype", parsed.get("torch_dtype")),
        "revision": parsed.get("_commit_hash", "UNKNOWN"),
        "config_files": config_files,
        "available_weights": weights,
    }, config.read_bytes()


def request_bodies(model):
    image_url = "data:image/png;base64," + base64.b64encode(
        fixture_png()).decode("ascii")
    text = {
        "model": model,
        "stream": True,
        "messages": [{"role": "user", "content":
                      "Reply with exactly: raw SSE transport capture."}],
        "max_tokens": 32,
        "temperature": 0,
        "seed": 42,
    }
    visual = {
        "model": model,
        "stream": True,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text":
             "Inspect the supplied abstract pattern. Write one short prose "
             "sentence, then one standalone line containing exactly "
             "[[image: high contrast abstract transport contract pattern]]. "
             "Do not add other text."},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]}],
        "modalities": ["text", "image"],
        "image": {"size": "256x256", "steps": 1, "seed": 42,
                  "output_format": "png"},
        "narrative": {"enabled": True, "visual_mode": "image",
                      "max_slots": 1, "allow_text_only": False},
        "max_tokens": 96,
        "temperature": 0,
        "seed": 42,
    }
    cancel = {
        "model": model,
        "stream": True,
        "messages": [{"role": "user", "content":
                      "Count upward from 1, putting one integer on each line. "
                      "Continue until the output token limit."}],
        "max_tokens": 512,
        "temperature": 0,
        "seed": 42,
    }
    return (("text", text), ("visual_contract", visual), ("cancel", cancel))


def capture_public_sse(base_url, body, recorder, session_id, *, cancel=False,
                       timeout=300.0):
    """Capture actual reads, optionally aborting after the first text delta."""
    params, option_inputs = traj.request_parameter_references(
        body, recorder.artifact_dir)
    inputs = traj.input_references(body["messages"], recorder.artifact_dir) \
        + option_inputs
    started_ns = time.perf_counter_ns()
    capture = RawSSECapture(clock=PerfCounterClock(origin_ns=started_ns))
    conn = http.client.HTTPConnection(base_url, timeout=timeout)
    abort_result = None
    observed_request_id = None
    inspected = 0
    try:
        conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
                     headers={"Content-Type": "application/json",
                              "Accept": "text/event-stream"})
        response = conn.getresponse()
        if response.status != 200:
            error_payload = response.read()
            require(False, f"public chat returned HTTP {response.status}: "
                           f"{error_payload!r}")
        require(response.getheader("Content-Type", "").startswith(
            "text/event-stream"), "public chat did not return SSE")
        while True:
            chunk = response.read1(65536) if hasattr(response, "read1") \
                else response.read(65536)
            if not chunk:
                break
            capture.feed(chunk)
            payloads = capture.payload_frames
            for _frame_index, payload in payloads[inspected:]:
                if payload == "[DONE]":
                    continue
                frame = json.loads(payload)
                if frame.get("id"):
                    observed_request_id = frame["id"]
                unified = frame.get("unified") or {}
                has_text = bool(unified.get("kind") == "vlm_delta"
                                and unified.get("text"))
                if not has_text:
                    for choice in frame.get("choices") or []:
                        has_text = bool((choice.get("delta") or {}).get("content"))
                        if has_text:
                            break
                if cancel and abort_result is None and has_text:
                    require(observed_request_id,
                            "cannot cancel before learning the public request id")
                    status, abort_body = traj.abort_request(
                        base_url, observed_request_id, timeout=30.0)
                    require(status == 200 and abort_body.get("success") is True,
                            f"public abort failed: {status} {abort_body}")
                    abort_result = {"status": status, "response": abort_body,
                                    "request_id": observed_request_id}
            inspected = len(payloads)
        try:
            capture.finish()
        except ReplayError as exc:
            failed_raw = recorder.jsonl_path.parent / (
                "failed-" + sha256_bytes(canonical(body))[:16] + ".sse")
            failed_raw.write_bytes(capture.raw_bytes)
            raise ReplayError(
                f"{exc}; preserved {len(capture.raw_bytes)} raw bytes at {failed_raw}") from exc
        details = {}
        request_id, events, output_text, state, error = traj.parse_sse_stream(
            io.BytesIO(capture.raw_bytes), artifact_dir=recorder.artifact_dir,
            details=details, capture=capture)
        record = recorder.record(
            session_id=session_id,
            request_id=request_id or traj._synthetic_request_id(events),
            endpoint="/v1/chat/completions", inputs=inputs, params=params,
            events=events, terminal_state=state, error=error,
            output_text=output_text, artifacts=details.get("artifacts", []),
            capture=capture)
    finally:
        conn.close()
    if cancel:
        require(abort_result is not None, "no in-flight public abort was issued")
        require(state == "cancelled", f"abort produced terminal state {state}")
    else:
        require(state == "completed", f"capture produced terminal state {state}: {error}")
    return record, abort_result


def source_stripped(events):
    varying = {"source_event_index", "source_event_sha256", "source_event_tick"}
    return [{key: value for key, value in event.items() if key not in varying}
            for event in events]


def framing_variant(record, raw, source_artifacts, destination, sizes):
    """Re-read accepted bytes with new boundaries; never mutate the capture."""
    capture = RawSSECapture(clock=LogicalTickClock())
    cursor = 0
    index = 0
    while cursor < len(raw):
        size = sizes[index % len(sizes)]
        capture.feed(raw[cursor:cursor + size])
        cursor += size
        index += 1
    capture.finish()
    recorder = traj.TrajectoryRecorder(destination, artifact_dir=source_artifacts)
    details = {}
    request_id, events, output_text, state, error = traj.parse_sse_stream(
        io.BytesIO(raw), artifact_dir=source_artifacts,
        details=details, capture=capture)
    variant = recorder.record(
        session_id=record["session_id"], request_id=request_id,
        endpoint=record["endpoint"], inputs=copy.deepcopy(record["inputs"]),
        params=copy.deepcopy(record["request_params"]), events=events,
        terminal_state=state, error=error, output_text=output_text,
        artifacts=details.get("artifacts", []), capture=capture)
    require(variant["sse_replay"]["raw_sha256"]
            == record["sse_replay"]["raw_sha256"], "rechunking changed raw bytes")
    require(source_stripped(variant["events"]) == source_stripped(record["events"]),
            "rechunking changed normalized event content")
    require(variant["artifacts"] == record["artifacts"],
            "rechunking changed reassembled artifacts")
    require(variant["terminal_state"] == record["terminal_state"]
            and variant["terminal_event_seq"] == record["terminal_event_seq"],
            "rechunking changed the unique terminal")
    original_frames = [{key: value for key, value in frame.items() if key != "tick"}
                       for frame in record["sse_replay"]["frames"]]
    variant_frames = [{key: value for key, value in frame.items() if key != "tick"}
                      for frame in variant["sse_replay"]["frames"]]
    require(variant_frames == original_frames,
            "rechunking changed frame bytes, offsets, kinds, or hashes")
    return {"chunks": len(variant["sse_replay"]["chunks"]),
            "raw_sha256": variant["sse_replay"]["raw_sha256"],
            "semantic_identity": True}


def fresh_replay(source_jsonl, replay_jsonl):
    replay_jsonl.parent.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-I", str(ROOT / "python" / "umminfer_sse_replay.py"),
               str(source_jsonl), "--out", str(replay_jsonl)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    require(result.returncode == 0,
            f"fresh-process replay failed: {result.stdout}\n{result.stderr}")
    require("RAW_SSE_REPLAY: PASS" in result.stdout, "replay success marker is absent")
    require(replay_jsonl.read_bytes() == source_jsonl.read_bytes(),
            "replayed canonical JSONL is not byte-identical")
    source_tree = file_tree(source_jsonl.parent / "artifacts")
    replay_tree = file_tree(replay_jsonl.parent / "artifacts")
    require(source_tree == replay_tree, "copied raw/media CAS tree differs")
    worker.validate_trajectory(str(replay_jsonl))
    _files, audit = pipeline.validate_trajectory_inputs(
        [replay_jsonl], strict_artifacts=True)
    require(audit["replay_records"] == 3
            and audit["verified_replay_refs"] == 3,
            f"replay input audit did not verify all raw sidecars: {audit}")
    return {"jsonl_sha256": sha256_file(source_jsonl)["sha256"],
            "replayed_jsonl_sha256": sha256_file(replay_jsonl)["sha256"],
            "byte_identical_jsonl": True,
            "verified_cas_files": len(source_tree),
            "cas_files": source_tree,
            "input_audit": audit}


def expect_tamper_rejection(source_jsonl, records, out_dir, *, raw_tamper):
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source_jsonl.parent / "artifacts", root / "artifacts")
    changed = copy.deepcopy(records)
    target = changed[0]
    if raw_tamper:
        replay = target["sse_replay"]
        raw_path = root / "artifacts" / replay["raw_ref"]
        raw = bytearray(raw_path.read_bytes())
        frame = next(item for item in replay["frames"] if item["kind"] == "event")
        raw[frame["offset"]] ^= 1
        raw_path.write_bytes(raw)
        expected = "SHA-256"
        label = "altered_source_frame"
    else:
        target["events"][0]["source_event_tick"] += 1
        target["sse_replay"]["normalized_events_sha256"] = sha256_bytes(
            canonical(target["events"]))
        expected = "source tick"
        label = "altered_source_tick_with_recomputed_event_digest"
    trajectory = root / "trajectories.jsonl"
    trajectory.write_bytes(b"".join(canonical(record) + b"\n" for record in changed))
    result = subprocess.run(
        [sys.executable, "-I", str(ROOT / "python" / "umminfer_sse_replay.py"),
         str(trajectory), "--out", str(root / "replayed" / "trajectories.jsonl")],
        capture_output=True, text=True, timeout=120)
    require(result.returncode != 0 and expected in result.stderr,
            f"{label} was not rejected as expected: {result.stdout}\n{result.stderr}")
    return {"case": label, "rejected": True,
            "returncode": result.returncode, "diagnostic": expected}


def write_checksums(root):
    root = Path(root)
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            info = sha256_file(path)
            entries.append(f"{info['sha256']}  {path.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")
    result = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=root,
                            capture_output=True, text=True, timeout=300)
    require(result.returncode == 0, f"evidence checksum verification failed: {result.stderr}")
    return len(entries)


def run(args):
    require(os.environ.get("UMMINFER_LIVE_OPT_IN") == "1",
            "LIVE_GATE: NOT_RUN (UMMINFER_LIVE_OPT_IN=1 is required)")
    parsed = urlparse(args.url if "://" in args.url else "http://" + args.url)
    require(parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost", "::1")
            and parsed.username is None and parsed.password is None
            and parsed.path in ("", "/") and parsed.query == "" and parsed.fragment == "",
            "live gate requires a plain ephemeral loopback HTTP endpoint")
    base_url = parsed.netloc
    require(args.gpu == 0, "this gate permits physical GPU 0 only")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    provenance, config_bytes = model_provenance(args.model_dir)
    (out_dir / "model-config.json").write_bytes(config_bytes)
    provenance.update({
        "source_commit": args.source_commit,
        "accepted_replay_commit": args.replay_commit,
        "replay_format": EXPECTED_REPLAY_FORMAT,
        "physical_gpu": args.gpu,
        "server_binary": sha256_file(args.server_binary),
        "stage_binary": sha256_file(args.stage_binary),
        "requested_model": args.model,
        "source_files": {name: sha256_file(ROOT / name) for name in (
            "python/umminfer_sse_replay.py",
            "python/umminfer_rl_trajectory.py",
            "tools/umminfer_stage_c_pipeline.py",
            "tests/sse_replay_live.py",
        )},
        "visual_input_evidence_class": "CONTRACT_FIXTURE_NOT_QUALITY_EVIDENCE",
        "visual_output_evidence_class": "CONTRACT_FIXTURE_NOT_REAL_MODEL_OUTPUT",
        "visual_output_backend": "umminfer_flux2_stage.py --backend mock",
        "timing_claim": "observed monotonic arrival provenance only; no performance claim",
    })

    capture_dir = out_dir / "capture"
    recorder = traj.TrajectoryRecorder(capture_dir / "trajectories.jsonl")
    records = []
    requests = {}
    captures = {}
    for name, body in request_bodies(args.model):
        requests[name] = body
        print(f"CAPTURE_BEGIN case={name}", flush=True)
        record, abort_result = capture_public_sse(
            base_url, body, recorder, "sse-live-gate",
            cancel=name == "cancel", timeout=args.timeout)
        records.append(record)
        replay = record["sse_replay"]
        require(replay["clock"]["kind"] == "monotonic_ns",
                "live capture did not use observed monotonic_ns ticks")
        require(replay["format"] == EXPECTED_REPLAY_FORMAT
                and replay["frames"][-1]["kind"] == "done",
                "capture lacks its terminal [DONE] frame")
        require(all(event["source_event_tick"] == replay["frames"][
            event["source_event_index"]]["tick"] for event in record["events"]),
                "event source tick does not match its source frame")
        captures[name] = {
            "request_sha256": sha256_bytes(canonical(body)),
            "request_id": record["request_id"],
            "raw_ref": replay["raw_ref"],
            "raw_sha256": replay["raw_sha256"],
            "raw_bytes": replay["raw_bytes"],
            "transport_chunks": len(replay["chunks"]),
            "sse_frames": len(replay["frames"]),
            "done_frame": replay["done_frame"],
            "normalized_events": len(record["events"]),
            "terminal_state": record["terminal_state"],
            "terminal_event_seq": record["terminal_event_seq"],
            "record_sha256": sha256_bytes(canonical(record)),
            "abort": abort_result,
        }
    require(not records[0]["artifacts"], "text-only case unexpectedly emitted media")
    require(len(records[1]["artifacts"]) == 1
            and records[1]["artifacts"][0]["mime_type"] == "image/png"
            and len(records[1]["visual_slots"]) == 1,
            "visual contract case did not emit one interleaved PNG slot")
    require(records[2]["terminal_state"] == "cancelled",
            "cancel case did not retain a cancelled terminal")
    (out_dir / "requests.jsonl").write_bytes(
        b"".join(canonical({"case": name, "body": body}) + b"\n"
                 for name, body in requests.items()))

    framing = {}
    for name, record in zip(("text", "visual_contract", "cancel"), records):
        raw = (recorder.artifact_dir / record["sse_replay"]["raw_ref"]).read_bytes()
        framing[name] = {
            "split": framing_variant(
                record, raw, recorder.artifact_dir,
                out_dir / "framing" / name / "split.jsonl", (1, 2, 3, 5, 8, 13)),
            "coalesced": framing_variant(
                record, raw, recorder.artifact_dir,
                out_dir / "framing" / name / "coalesced.jsonl", (len(raw),)),
        }

    replay_summary = fresh_replay(
        recorder.jsonl_path, out_dir / "replayed" / "trajectories.jsonl")
    original_artifacts = file_tree(recorder.artifact_dir)
    tampering = [
        expect_tamper_rejection(recorder.jsonl_path, records,
                                out_dir / "tamper-frame", raw_tamper=True),
        expect_tamper_rejection(recorder.jsonl_path, records,
                                out_dir / "tamper-tick", raw_tamper=False),
    ]
    require(file_tree(recorder.artifact_dir) == original_artifacts,
            "tamper checks modified accepted source bytes")

    summary = {
        "schema_version": 1,
        "status": "PASS",
        "evidence_class": "REAL_PUBLIC_SSE_TRANSPORT_WITH_CONTRACT_VISUAL_MEDIA",
        "model_quality_claim": "NOT_RUN",
        "generalized_performance_claim": "NOT_RUN",
        "provenance": provenance,
        "captures": captures,
        "framing": framing,
        "fresh_process_replay": replay_summary,
        "tamper_rejection": tampering,
    }
    (out_dir / "live-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["evidence_checksum_files"] = write_checksums(out_dir)
    print(json.dumps({"status": summary["status"], "captures": captures,
                      "replay_jsonl_sha256": replay_summary["jsonl_sha256"],
                      "verified_cas_files": replay_summary["verified_cas_files"],
                      "tamper_rejection": tampering,
                      "evidence_checksum_files": summary["evidence_checksum_files"]},
                     sort_keys=True))
    print("LIVE_PUBLIC_SSE_CAPTURE_REPLAY: PASS")
    print("VISUAL_MEDIA: CONTRACT_FIXTURE_ONLY")
    print("MODEL_QUALITY: NOT_RUN")
    print("GENERALIZED_PERFORMANCE: NOT_RUN")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="loopback host:port only")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--replay-commit", required=True)
    parser.add_argument("--server-binary", required=True)
    parser.add_argument("--stage-binary", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)
    if os.environ.get("UMMINFER_LIVE_OPT_IN") != "1":
        print("LIVE_PUBLIC_SSE_CAPTURE_REPLAY: NOT_RUN (opt-in missing)")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
