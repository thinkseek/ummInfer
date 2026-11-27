#!/usr/bin/env python3
"""Independent fresh-process verification of native Wan cancel/fault evidence.

This verifier intentionally does not import the live driver. It reconstructs
public events and media from exact transport bytes, decodes every retained
MessagePack message, checks native logs/timing/process/GPU ownership, rehashes
available binaries/model/source files, probes valid MP4s, and seals evidence.
"""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

EXPECTED_FORMAT = "umminfer-native-wan-cancel-fault-v1"
CONTROL_LABELS = {"control", "recovery-after-queued-denoise",
                  "recovery-after-artifact", "terminal-then-cancel",
                  "recovery-after-terminal", "recovery-after-invalid"}
CANCEL_LABELS = {"queued-cancel", "denoise-cancel", "artifact-cancel",
                 "terminal-then-cancel"}
ACTIVE_CANCELS = {"queued-cancel", "denoise-cancel", "artifact-cancel"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def split_frames(raw):
    frames = []
    cursor = 0
    for match in re.finditer(rb"\r?\n\r?\n", raw):
        frames.append(raw[cursor:match.end()])
        cursor = match.end()
    require(cursor == len(raw), "trailing partial SSE frame")
    return frames


def frame_data(frame):
    values = []
    for line in frame.replace(b"\r\n", b"\n").split(b"\n"):
        if line.startswith(b"data:"):
            value = line[5:]
            values.append(value[1:] if value.startswith(b" ") else value)
    return values


def check_event_contract(events):
    terminals = [index for index, event in enumerate(events)
                 if event.get("kind") in {"complete", "error", "cancelled"}]
    require(len(terminals) == 1, f"public terminal count != 1: {terminals}")
    require(terminals[0] == len(events) - 1, "public event arrived after terminal")
    for index, event in enumerate(events):
        require(event.get("seq") == index, f"public seq gap at {index}")
    return events[terminals[0]]


def parse_native_abort(line):
    match = re.search(
        r'video request aborted .*?request_id=(\S+).*?boundary="([^"]+)"', line)
    if not match:
        return None
    result = {"request_id": match.group(1), "boundary": match.group(2)}
    for field in ("abort_epoch_ms", "exit_epoch_ms", "compute_ms"):
        found = re.search(rf"\b{field}=(\d+)", line)
        require(found is not None, f"native abort log lacks {field}: {line}")
        result[field] = int(found.group(1))
    return result


def hash_model(root):
    files = []
    for path in sorted(item for item in Path(root).rglob("*") if item.is_file()):
        files.append({"path": path.relative_to(root).as_posix(),
                      "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return files, sha256_bytes(b"".join(
        f"{item['sha256']}  {item['path']}\n".encode() for item in files))


def probe(ffprobe, path):
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries",
         "stream=codec_name,width,height,nb_frames,r_frame_rate,pix_fmt",
         "-of", "json", str(path)], check=True, capture_output=True, text=True,
        timeout=30)
    streams = json.loads(result.stdout)["streams"]
    require(streams == [{"codec_name": "h264", "width": 352, "height": 256,
                         "nb_frames": "9", "r_frame_rate": "16/1",
                         "pix_fmt": "yuv420p"}],
            f"unexpected valid MP4 stream: {streams}")
    return streams[0]


def verify_existing_manifest(root):
    manifest = root / "SHA256SUMS"
    if not manifest.exists():
        return 0
    lines = manifest.read_text(encoding="ascii").splitlines()
    for line in lines:
        digest, relative = line.split("  ", 1)
        path = root / relative
        require(path.is_file(), f"manifest file missing: {relative}")
        require(sha256_file(path) == digest, f"manifest checksum failed: {relative}")
    return len(lines)


def verify(root, args):
    root = Path(root).resolve()
    summary = load_json(root / "summary.json")
    identities = load_json(root / "identities.json")
    requests_list = load_json(root / "requests.json")
    taps = load_json(root / "taps.json")
    processes = load_json(root / "processes.json")
    requests = {item["label"]: item for item in requests_list}
    require(summary["format"] == EXPECTED_FORMAT and summary["status"] == "PASS",
            "unexpected evidence format/status")
    require(summary["physical_gpu"] == 7, "gate was not confined to physical GPU 7")
    require(summary["base_commit"] == args.base_commit and
            summary["execution_commit"] == args.execution_commit,
            "base/execution identity mismatch")
    require(summary["control_count"] == 1 and summary["valid_generation_count"] == 6 and
            summary["cancellation_count"] == 4 and summary["fault_count"] == 1 and
            summary["recovery_count"] == 4, "case counts do not match the accepted gate")
    require(set(requests) == CONTROL_LABELS | ACTIVE_CANCELS | {"invalid-input"},
            f"unexpected request labels: {sorted(requests)}")
    require(len(requests_list) == len(requests), "duplicate request label")
    verify_existing_manifest(root)

    for key in ("server_binary", "wan_binary"):
        item = identities[key]
        path = root / item["file"]
        require(path.stat().st_size == item["bytes"] and sha256_file(path) == item["sha256"],
                f"retained binary identity mismatch: {key}")
    source_by_path = {item["path"]: item for item in identities["sources"]}
    for name in ("native_wan_cancel_fault_live.py", "native_wan_cancel_fault_verify.py"):
        source = source_by_path[f"tests/{name}"]
        retained = root / "source" / name
        require(retained.stat().st_size == source["bytes"] and
                sha256_file(retained) == source["sha256"],
                f"retained harness source mismatch: {name}")
    if args.source_root:
        for relative, item in source_by_path.items():
            path = Path(args.source_root) / relative
            require(path.stat().st_size == item["bytes"] and sha256_file(path) == item["sha256"],
                    f"source identity mismatch: {relative}")
    model_status = "manifest_bound_only"
    if args.model:
        files, tree_hash = hash_model(args.model)
        require(files == identities["model"]["files"] and
                tree_hash == identities["model"]["tree_sha256"],
                "model tree identity mismatch")
        model_status = "all_files_rehashed"
    require(identities["model"]["file_count"] == len(identities["model"]["files"]) and
            identities["model"]["file_count"] >= 30, "incomplete Wan model manifest")
    require(Path(args.ffprobe).is_file() or os.sep not in args.ffprobe,
            "ffprobe argument is invalid")

    decoded_taps = {}
    all_native_records = []
    for tap_name, records in taps.items():
        require([item["index"] for item in records] == list(range(len(records))),
                f"{tap_name}: tap index gap")
        decoded_taps[tap_name] = []
        for item in records:
            raw = (root / item["packed_file"]).read_bytes()
            require(len(raw) == item["packed_bytes"] and
                    sha256_bytes(raw) == item["packed_sha256"],
                    f"raw tap hash mismatch: {item['packed_file']}")
            message = __import__("msgpack").unpackb(raw, raw=False, strict_map_key=False)
            require(message.get("type") == item["message_type"] and
                    message.get("request_id") == item["request_id"] and
                    message.get("from_stage") == item["from_stage"],
                    f"raw tap semantic mismatch: {item['packed_file']}")
            require(item["observed_monotonic_ns"] <= item["forwarded_monotonic_ns"],
                    "nonmonotonic tap forwarding")
            if message.get("type") == "complete":
                result = message.get("result") or {}
                require(bool(message.get("success")) == item["success"] and
                        message.get("error") == item["error"],
                        "native terminal metadata mismatch")
                if isinstance(result.get("data"), bytes):
                    require(len(result["data"]) == item["media_bytes"] and
                            sha256_bytes(result["data"]) == item["media_sha256"],
                            "native terminal media hash mismatch")
            joined = dict(item)
            joined["message"] = message
            decoded_taps[tap_name].append(joined)
            all_native_records.append(joined)

    native_log = (root / "wan.log").read_text(encoding="utf-8", errors="replace")
    native_aborts = {}
    native_success_logs = {}
    for line in native_log.splitlines():
        abort = parse_native_abort(line)
        if abort:
            require(abort["request_id"] not in native_aborts,
                    f"duplicate native abort log: {abort['request_id']}")
            native_aborts[abort["request_id"]] = abort
        if "video request backend=native-cuda-wan" in line:
            match = re.search(r"\brequest_id=(\S+)", line)
            require(match is not None, "native success log lacks request id")
            require(match.group(1) not in native_success_logs,
                    f"duplicate native generation log: {match.group(1)}")
            native_success_logs[match.group(1)] = line
    require("ready kind=video" in native_log, "native Wan readiness log missing")

    media_hashes = {}
    public_timing = {}
    for label, item in requests.items():
        capture = load_json(root / item["capture_file"])
        request_body = json.loads((root / f"{label}.request.json").read_text(encoding="utf-8"))
        require(sha256_bytes(canonical(request_body)) == item["request_sha256"] ==
                capture["request_sha256"], f"{label}: request hash mismatch")
        require(request_body["session_id"] == item["session_id"] and
                request_body["stream"] is True and request_body["modalities"] == ["text", "video"],
                f"{label}: public request contract mismatch")
        chunks = []
        offset = 0
        previous_tick = 0
        for chunk in capture["chunks"]:
            raw = (root / chunk["file"]).read_bytes()
            require(chunk["offset"] == offset and len(raw) == chunk["bytes"] and
                    sha256_bytes(raw) == chunk["sha256"], f"{label}: transport chunk mismatch")
            require(chunk["observed_monotonic_ns"] >= previous_tick,
                    f"{label}: nonmonotonic transport timing")
            previous_tick = chunk["observed_monotonic_ns"]
            chunks.append(raw)
            offset += len(raw)
        raw_sse = b"".join(chunks)
        require(raw_sse == (root / capture["raw_file"]).read_bytes() and
                len(raw_sse) == capture["raw_bytes"] and
                sha256_bytes(raw_sse) == capture["raw_sha256"],
                f"{label}: raw SSE reconstruction mismatch")
        frames = split_frames(raw_sse)
        require(len(frames) == len(capture["frames"]), f"{label}: raw frame count mismatch")
        rebuilt_events = []
        done = 0
        for index, frame in enumerate(frames):
            frame_record = capture["frames"][index]
            require(frame_record["index"] == index and len(frame) == frame_record["bytes"] and
                    sha256_bytes(frame) == frame_record["sha256"],
                    f"{label}: raw frame hash mismatch")
            for payload in frame_data(frame):
                if payload == b"[DONE]":
                    done += 1
                    continue
                outer = json.loads(payload)
                if isinstance(outer.get("unified"), dict):
                    require(outer.get("id") == item["request_id"],
                            f"{label}: stale public request id")
                    rebuilt_events.append(outer["unified"])
        require(done == 1 == capture["done_count"], f"{label}: [DONE] count mismatch")
        terminal = check_event_contract(rebuilt_events)
        require(terminal["kind"] == item["terminal_kind"], f"{label}: terminal kind mismatch")
        require(all(event.get("session_id") == item["session_id"] for event in rebuilt_events),
                f"{label}: stale session event")
        event_rows = [json.loads(line) for line in
                      (root / item["events_file"]).read_text(encoding="utf-8").splitlines()]
        require([row["event"] for row in event_rows] == rebuilt_events and
                [row["event_index"] for row in event_rows] == list(range(len(rebuilt_events))),
                f"{label}: retained event log differs from raw SSE")
        require(event_rows[-1]["observed_monotonic_ns"] == item["terminal_monotonic_ns"],
                f"{label}: terminal tick mismatch")
        require(item["submit_monotonic_ns"] <= item["terminal_monotonic_ns"] <=
                item["end_monotonic_ns"], f"{label}: nonmonotonic public lifecycle")

        video_chunks = []
        for event in rebuilt_events:
            if event.get("kind") == "artifact" and event.get("stage") == "video":
                artifact = event["artifact"]
                raw = base64.b64decode(artifact["data"], validate=True)
                video_chunks.append((int(artifact["index"]), int(artifact["total"]), raw))
        if item["media"] is None:
            require(not video_chunks, f"{label}: cancelled/failed public media escaped")
        else:
            require(video_chunks and len({total for _, total, _ in video_chunks}) == 1,
                    f"{label}: no consistent public video")
            total = video_chunks[0][1]
            require(sorted(index for index, _, _ in video_chunks) == list(range(total)),
                    f"{label}: public artifact chunk gap")
            media = b"".join(raw for _, _, raw in sorted(video_chunks))
            media_path = root / item["media"]["file"]
            require(media_path.read_bytes() == media and len(media) == item["media"]["bytes"] and
                    sha256_bytes(media) == item["media"]["sha256"],
                    f"{label}: public media reconstruction mismatch")
            require(media[4:8] == b"ftyp", f"{label}: invalid MP4 header")
            probe(args.ffprobe, media_path)
            media_hashes[label] = sha256_bytes(media)

        native_id = item["native_request_id"]
        native = [record for record in all_native_records
                  if record["request_id"] == native_id and
                  (record["tap"] != "native-output" or record.get("from_stage") == "video")]
        submits = [record for record in native if record["tap"] == "native-input" and
                   record["message_type"] == "submit"]
        completes = [record for record in native if record["tap"] == "native-output" and
                     record["message_type"] == "complete"]
        aborts = [record for record in native if record["tap"] == "native-input" and
                  record["message_type"] == "abort"]
        require(len(submits) == 1 and len(completes) == 1 == item["native_terminal_count"],
                f"{label}: native submit/terminal count mismatch")
        complete = completes[0]
        if label in ACTIVE_CANCELS:
            require(item["native_abort_terminal"]["packed_file"] == complete["packed_file"],
                    f"{label}: producer terminal is not the owning video Stage message")
            require(terminal["kind"] == "cancelled" and not complete["success"] and
                    complete["message"].get("error") == "request aborted" and
                    len(aborts) == 1, f"{label}: active native cancellation contract failed")
            require(not (complete["message"].get("result") or {}).get("data"),
                    f"{label}: native cancelled media escaped")
            abort = item["abort"]
            require(abort["http_status"] == 200 and abort["body"] == {"success": True},
                    f"{label}: abort was not acknowledged")
            require(abort["send_monotonic_ns"] <= abort["ack_monotonic_ns"] and
                    abort["ack_monotonic_ns"] - abort["send_monotonic_ns"] <= 1_000_000_000,
                    f"{label}: abort ack exceeded 1 s")
            require(abort["send_monotonic_ns"] <= item["terminal_monotonic_ns"] and
                    item["terminal_monotonic_ns"] - abort["send_monotonic_ns"] <= 1_000_000_000,
                    f"{label}: public cancel terminal exceeded 1 s")
            require(abort["send_monotonic_ns"] <= complete["observed_monotonic_ns"],
                    f"{label}: native terminal precedes abort")
            public_timing[label] = {
                "abort_ack_ms": (abort["ack_monotonic_ns"] - abort["send_monotonic_ns"]) / 1e6,
                "public_terminal_ms": (item["terminal_monotonic_ns"] -
                                       abort["send_monotonic_ns"]) / 1e6,
                "native_terminal_ms": (complete["observed_monotonic_ns"] -
                                       abort["send_monotonic_ns"]) / 1e6}
            if label == "queued-cancel":
                require(native_id not in native_aborts and native_id not in native_success_logs,
                        "queued request reached native compute")
            else:
                require(native_id in native_aborts and native_id not in native_success_logs,
                        f"{label}: missing cooperative abort boundary")
                boundary = native_aborts[native_id]
                for key, value in boundary.items():
                    require(item["native_abort_log"][key] == value,
                            f"{label}: native log field mismatch {key}")
                require(item["native_abort_log"]["observed_monotonic_ns"] >=
                        complete["observed_monotonic_ns"],
                        f"{label}: cooperative exit preceded native abort terminal")
                require(boundary["abort_epoch_ms"] <= boundary["exit_epoch_ms"],
                        f"{label}: nonmonotonic native epoch log")
                public_timing[label]["native_detection_wait_ms"] = (
                    boundary["abort_epoch_ms"] - abort["send_wall_time_ns"] / 1e6)
                public_timing[label]["native_exit_wait_ms"] = (
                    boundary["exit_epoch_ms"] - abort["send_wall_time_ns"] / 1e6)
                public_timing[label]["native_unwind_after_detection_ms"] = (
                    boundary["exit_epoch_ms"] - boundary["abort_epoch_ms"])
                public_timing[label]["native_boundary_observed_ms"] = (
                    item["native_abort_log"]["observed_monotonic_ns"] -
                    abort["send_monotonic_ns"]) / 1e6
                # Native epoch fields are truncated to integer milliseconds. Each
                # wall-bridged duration is therefore an interval [raw, raw + 1 ms),
                # not an exact nonnegative timestamp difference.
                public_timing[label]["native_wall_clock_quantization_ms"] = 1.0
                public_timing[label]["native_detection_wait_interval_ms"] = [
                    public_timing[label]["native_detection_wait_ms"],
                    public_timing[label]["native_detection_wait_ms"] + 1.0]
                public_timing[label]["native_exit_wait_interval_ms"] = [
                    public_timing[label]["native_exit_wait_ms"],
                    public_timing[label]["native_exit_wait_ms"] + 1.0]
                require(-1.0 < public_timing[label]["native_detection_wait_ms"] and
                        -1.0 < public_timing[label]["native_exit_wait_ms"] and
                        public_timing[label]["native_exit_wait_ms"] >=
                        public_timing[label]["native_detection_wait_ms"],
                        f"{label}: invalid wall-bridged native cancellation timing")
                if label == "denoise-cancel":
                    require(re.search(r"\bstep [1-9][0-9]*/20\b", boundary["boundary"]),
                            f"denoise abort is not at a denoise step: {boundary}")
                else:
                    require("MP4" in boundary["boundary"],
                            f"artifact abort is not during MP4 emission: {boundary}")
        elif label == "invalid-input":
            require(terminal["kind"] == "complete" and not complete["success"] and
                    "divisible by 32" in complete["message"].get("error", "") and
                    not aborts and native_id not in native_success_logs,
                    "invalid input did not fail before native generation")
            require(next(leg for leg in item["legs"] if leg["stage"] == "video")["state"]
                    == "failed", "invalid public video leg is not failed")
        else:
            require(label in CONTROL_LABELS and terminal["kind"] == "complete" and
                    complete["success"] and not aborts,
                    f"{label}: valid control/recovery failed")
            native_media = complete["message"]["result"]["data"]
            require(sha256_bytes(native_media) == media_hashes[label],
                    f"{label}: public/native media mismatch")
            require(native_id in native_success_logs, f"{label}: real native log missing")
        if label == "terminal-then-cancel":
            require(item["abort"]["http_status"] == 200 and not aborts and
                    item["post_terminal_no_new_events"] is True,
                    "after-terminal cancellation created native activity")

    control_hash = media_hashes["control"]
    require(control_hash == summary["control_media_sha256"], "summary control hash mismatch")
    require(all(media_hashes[label] == control_hash for label in CONTROL_LABELS),
            f"valid media is not exactly repeated: {media_hashes}")
    initial_control = requests["control"]
    first_abort = min(item["abort"]["send_monotonic_ns"] for item in requests.values()
                      if "abort" in item)
    require(initial_control["end_monotonic_ns"] < first_abort,
            "fault/cancel injection preceded a valid real control")
    recovery_predecessors = {
        "recovery-after-queued-denoise": ["queued-cancel", "denoise-cancel"],
        "recovery-after-artifact": ["artifact-cancel"],
        "recovery-after-terminal": ["terminal-then-cancel"],
        "recovery-after-invalid": ["invalid-input"],
    }
    for recovery, predecessors in recovery_predecessors.items():
        require(requests[recovery]["submit_monotonic_ns"] >= max(
            requests[label]["terminal_monotonic_ns"] for label in predecessors),
            f"{recovery}: submitted before predecessor terminal")
    for recovery, cancelled in (("recovery-after-queued-denoise", "denoise-cancel"),
                                ("recovery-after-artifact", "artifact-cancel")):
        require(requests[recovery]["submit_monotonic_ns"] >=
                requests[cancelled]["native_abort_log"]["observed_monotonic_ns"],
                f"{recovery}: submitted before cooperative native exit")
    require(requests["recovery-after-terminal"]["submit_monotonic_ns"] >=
            requests["terminal-then-cancel"]["abort"]["ack_monotonic_ns"],
            "post-terminal recovery preceded late-abort acknowledgment")

    queued = requests["queued-cancel"]
    running = requests["denoise-cancel"]
    queued_native_submit = next(record for record in all_native_records
        if record["tap"] == "native-input" and record["message_type"] == "submit" and
        record["request_id"] == queued["native_request_id"])
    require(queued["submit_monotonic_ns"] > running["submit_monotonic_ns"] and
            queued_native_submit["observed_monotonic_ns"] <
            running["abort"]["send_monotonic_ns"] and
            queued["terminal_monotonic_ns"] <
            running["native_abort_log"]["observed_monotonic_ns"],
            "queued victim did not settle before the running blocker exited")
    partial = requests["artifact-cancel"]["partial_artifact"]
    partial_raw = (root / partial["file"]).read_bytes()
    require(len(partial_raw) == partial["bytes"] >= 4096 and
            len(partial_raw) <= 48 * 1024 * 1024 and
            sha256_bytes(partial_raw) == partial["sha256"] and
            partial["bounded_by_public_limit"] is True,
            "partial native artifact is missing/unbounded")
    require(not list((root / "native-tmp").iterdir()),
            "task-owned native temporary artifacts remain")

    process_by_name = {item["name"]: item for item in processes}
    require(set(process_by_name) == {"server", "entry", "vlm", "wan"},
            "unexpected process ownership set")
    require(len({item["pid"] for item in processes}) == 4, "process pids are not distinct")
    for item in processes:
        require(item["proc_start_ticks"] > 0 and item["stop_error"] is None and
                item["start_monotonic_ns"] <= item["stop_monotonic_ns"],
                f"process ownership/lifecycle invalid: {item}")
        require(item["log"] and (root / item["log"]).is_file(), "native/process log missing")
    require(process_by_name["wan"]["returncode"] == 0,
            "native Wan did not shut down cleanly")
    initial = load_json(root / "gpu-initial.json")
    final = load_json(root / "gpu-final.json")
    ready = load_json(root / "gpu-ready.json")
    for state in (initial, final):
        require(state["snapshot"]["index"] == 7 and
                state["snapshot"]["memory_used_mib"] <= 16 and
                state["compute_processes"] == [], "GPU 7 is not clean")
    require(ready["wan_container_pid"] == process_by_name["wan"]["pid"] and
            ready["nvml_pid_namespace"] == "driver_host" and
            len(ready["compute_processes"]) == 1 and
            ready["compute_processes"][0]["used_memory_mib"] > 0,
            "isolated GPU 7 did not have the single expected native Wan owner")
    samples = load_json(root / "gpu-samples.json")
    require(samples and all(item["index"] == 7 for item in samples) and
            all(left["observed_monotonic_ns"] <= right["observed_monotonic_ns"]
                for left, right in zip(samples, samples[1:])),
            "GPU samples are nonmonotonic or not GPU 7")
    final_health = load_json(root / "health-final.json")
    require(final_health["status"] == 200 and final_health["body"].get("in_flight") == 0,
            "final public in_flight is not zero")

    return {"format": "umminfer-native-wan-cancel-fault-verification-v1",
            "status": "PASS", "base_commit": args.base_commit,
            "execution_commit": args.execution_commit,
            "verifier_sha256": sha256_file(Path(__file__).resolve()),
            "physical_gpu": 7, "control_count": 1, "valid_generation_count": 6,
            "cancellation_count": 4, "active_native_cancellation_count": 3,
            "fault_count": 1, "recovery_count": 4,
            "control_media_sha256": control_hash, "media_sha256": media_hashes,
            "timing_ms": public_timing,
            "partial_artifact": {"bytes": partial["bytes"], "sha256": partial["sha256"]},
            "model_verification": model_status,
            "native_terminal_count_per_request": 1,
            "late_public_events": 0, "late_cancelled_artifacts": 0,
            "final_in_flight": 0, "gpu7_final_memory_mib": final["snapshot"]["memory_used_mib"],
            "cooperative_cuda_preemption_claimed": False}


def seal(root, result):
    root = Path(root)
    verification = root / "independent-verification.json"
    verification.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    lines = []
    for path in sorted(item for item in root.rglob("*")
                       if item.is_file() and item.name != "SHA256SUMS"):
        lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="ascii")
    verify_existing_manifest(root)
    return sha256_file(root / "SHA256SUMS")


class OfflineTests(unittest.TestCase):
    def test_raw_frames(self):
        raw = b'data: {"unified":{"kind":"complete","seq":0}}\n\ndata: [DONE]\n\n'
        frames = split_frames(raw)
        self.assertEqual(len(frames), 2)
        self.assertEqual(frame_data(frames[1]), [b"[DONE]"])

    def test_terminal_and_stale_rejection(self):
        check_event_contract([{"kind": "cancelled", "seq": 0}])
        with self.assertRaises(RuntimeError):
            check_event_contract([{"kind": "complete", "seq": 0},
                                  {"kind": "artifact", "seq": 1}])
        with self.assertRaises(RuntimeError):
            check_event_contract([{"kind": "complete", "seq": 1}])

    def test_native_wall_timestamp_quantization_interval(self):
        send_wall_ns = 1_000_500_000
        raw_ms = 1000 - send_wall_ns / 1e6
        lower, upper = raw_ms, raw_ms + 1.0
        self.assertLess(lower, 0)
        self.assertLessEqual(lower, 0.2)
        self.assertGreater(upper, 0.2)

    def test_native_abort_fields(self):
        line = ('video request aborted backend=native-cuda-wan request_id=http-2:video '
                'boundary="aborted at step 2/20" abort_epoch_ms=10 exit_epoch_ms=15 compute_ms=30')
        parsed = parse_native_abort(line)
        self.assertEqual(parsed["exit_epoch_ms"] - parsed["abort_epoch_ms"], 5)
        self.assertEqual(parsed["compute_ms"], 30)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--base-commit")
    parser.add_argument("--execution-commit")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(OfflineTests))
        return 0 if result.wasSuccessful() else 1
    require(args.evidence_dir and args.base_commit and args.execution_commit,
            "evidence dir and exact base/execution commits are required")
    result = verify(args.evidence_dir, args)
    if not args.no_write:
        manifest_hash = seal(args.evidence_dir, result)
        result["sha256sums_sha256"] = manifest_hash
    print(json.dumps(result, indent=2, sort_keys=True))
    print("NATIVE_WAN_CANCEL_FAULT_INDEPENDENT_VERIFY: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
