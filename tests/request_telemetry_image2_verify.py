#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent stdlib-only verifier for retained image2 allocator evidence.

This intentionally does not import request_telemetry_live.py. Run with python3 -I
in a fresh process against the raw evidence directory. The original SHA256SUMS
must cover every producer file; this verifier then writes its own report and a
final SHA256SUMS.independent covering both producer and verification records.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import struct
import zlib


class VerifyFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise VerifyFailure(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def read_json(root, name):
    return json.loads((root / name).read_bytes())


def telemetry(body):
    if isinstance(body.get("telemetry"), dict):
        return body["telemetry"]
    unified = body.get("unified")
    if isinstance(unified, dict) and isinstance(unified.get("telemetry"), dict):
        return unified["telemetry"]
    error = body.get("error")
    if isinstance(error, dict) and isinstance(error.get("telemetry"), dict):
        return error["telemetry"]
    return None


def stage_row(envelope, stage, native_stage=None):
    rows = [row for row in envelope.get("stages", [])
            if row.get("stage") == stage and
            (native_stage is None or row.get("native_stage") == native_stage)]
    require(len(rows) == 1, f"expected one {stage}/{native_stage} stage")
    return rows[0]


def null_bytes(row):
    for key in ("allocated_bytes_peak", "reserved_bytes_peak",
                "allocated_bytes_final", "reserved_bytes_final"):
        require(row.get(key) is None, f"NOT_COVERED {key} must be null")


def covered_image(envelope, lifecycle, nonzero):
    row = stage_row(envelope, "image", "image")
    require(row.get("coverage") == "COVERED" and
            row.get("backend") == "qwen-image-2.1" and
            row.get("scope") == "request-native-cuda-runtime" and
            row.get("device_index") == 1 and
            row.get("lifecycle_status") == lifecycle and
            row.get("execution") == {"mode": "serial", "batch_capacity": 1},
            f"invalid covered image identity/lifecycle: {row}")
    epoch = row.get("epoch", {})
    require(type(epoch.get("id")) is int and epoch["id"] > 0 and
            epoch.get("reset") is True and
            epoch.get("semantics") == "new_allocation_ownership_epoch",
            "invalid image ownership epoch")
    for key in ("allocated_bytes_peak", "reserved_bytes_peak",
                "allocated_bytes_final", "reserved_bytes_final"):
        require(type(row.get(key)) is int and row[key] >= 0,
                f"missing integer {key}")
    require(row["allocated_bytes_peak"] == row["reserved_bytes_peak"] and
            row["allocated_bytes_final"] == row["reserved_bytes_final"],
            "direct allocated/reserved bytes differ")
    samples = row.get("sample_coverage", {})
    require(samples.get("terminal_snapshots") == 1,
            "covered image does not have one terminal snapshot")
    if nonzero:
        require(row["allocated_bytes_peak"] > 0 and
                samples.get("allocation_events", 0) > 0 and
                samples.get("free_events") == samples["allocation_events"] and
                row["allocated_bytes_final"] == 0,
                f"successful/cancelled image direct ownership did not close: {row}")
    else:
        require(row["allocated_bytes_peak"] == 0 and
                row["allocated_bytes_final"] == 0 and
                samples.get("allocation_events") == 0 and
                samples.get("free_events") == 0,
                "native validation failure has fabricated allocator events")
    return row


def normalize(value, request_id=None):
    if isinstance(value, dict):
        return {key: normalize(item, request_id) for key, item in value.items()
                if key not in ("created", "session_id", "telemetry",
                               "request_telemetry", "denoise_ms", "encode_ms",
                               "first_frame_ms",
                               "total_ms", "first_audio_ms", "elapsed_ms")}
    if isinstance(value, list):
        return [normalize(item, request_id) for item in value]
    if isinstance(value, str) and request_id:
        return value.replace(request_id, "<request>")
    return value


def normalized_digest(value, request_id):
    return digest(canonical(normalize(value, request_id)))


def parse_sse(raw):
    frames = []
    pending = []
    for raw_line in raw.decode("utf-8").splitlines():
        if raw_line.startswith("data:"):
            pending.append(raw_line[5:].lstrip())
            continue
        if raw_line or not pending:
            continue
        data = "\n".join(pending)
        pending = []
        if data != "[DONE]":
            frames.append(json.loads(data))
    require(not pending, "unterminated SSE data frame")
    return frames


def decode_png(png):
    require(png[:8] == b"\x89PNG\r\n\x1a\n", "bad PNG signature")
    offset = 8
    compressed = bytearray()
    dimensions = None
    saw_end = False
    while offset < len(png):
        require(offset + 12 <= len(png), "truncated PNG chunk")
        size = struct.unpack(">I", png[offset:offset + 4])[0]
        kind = png[offset + 4:offset + 8]
        end = offset + size + 12
        require(end <= len(png), "overflowing PNG chunk")
        payload = png[offset + 8:offset + 8 + size]
        crc = struct.unpack(">I", png[offset + 8 + size:end])[0]
        require((zlib.crc32(kind + payload) & 0xffffffff) == crc,
                "PNG CRC mismatch")
        if kind == b"IHDR":
            values = struct.unpack(">IIBBBBB", payload)
            require(values == (512, 512, 8, 6, 0, 0, 0),
                    "PNG is not 512x512 RGBA8")
            dimensions = values[:2]
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            require(size == 0 and end == len(png), "invalid PNG IEND")
            saw_end = True
        offset = end
    require(saw_end and dimensions == (512, 512) and compressed,
            "incomplete PNG")
    decoded = zlib.decompress(bytes(compressed))
    stride = 512 * 4 + 1
    require(len(decoded) == stride * 512, "wrong decoded PNG byte count")
    rows = [decoded[y * stride:(y + 1) * stride] for y in range(512)]
    require(all(row[0] == 0 for row in rows), "unexpected PNG row filter")
    pixels = b"".join(row[1:] for row in rows)
    require(len(set(pixels)) > 1, "constant image is not a useful media fixture")
    return {"width": 512, "height": 512, "color_type": "RGBA8",
            "png_bytes": len(png), "png_sha256": digest(png),
            "rgba_bytes": len(pixels), "rgba_sha256": digest(pixels)}


def json_media(root, name, body):
    artifacts = [item for item in body.get("artifacts", [])
                 if item.get("media_type") == "image"]
    require(len(artifacts) == 1, f"{name}: expected one JSON image")
    artifact = artifacts[0]
    require("request_telemetry" not in artifact,
            f"{name}: internal telemetry leaked into artifact")
    png = base64.b64decode(artifact["data"], validate=True)
    require(artifact.get("bytes") == len(png) and
            artifact.get("sha256") == digest(png), f"{name}: media descriptor mismatch")
    require((root / f"{name}.png").read_bytes() == png,
            f"{name}: retained PNG differs from raw JSON")
    return decode_png(png)


def sse_media(root, name, frames):
    parts = {}
    totals = {}
    final = {}
    for frame in frames:
        unified = frame.get("unified", {})
        artifact = unified.get("artifact", {})
        if (unified.get("kind") != "artifact" or
                artifact.get("mime_type") != "image/png"):
            continue
        identity = artifact["artifact_id"]
        index, total = artifact["index"], artifact["total"]
        require(type(index) is int and type(total) is int and 0 <= index < total,
                f"{name}: bad chunk index")
        require(totals.setdefault(identity, total) == total,
                f"{name}: chunk total changed")
        require(index not in parts.setdefault(identity, {}),
                f"{name}: duplicate chunk")
        parts[identity][index] = base64.b64decode(artifact["data"], validate=True)
        if "sha256" in artifact:
            require(index + 1 == total, f"{name}: digest before final chunk")
            final[identity] = (artifact["sha256"], artifact["bytes"])
    require(len(parts) == 1 and final.keys() == parts.keys(),
            f"{name}: expected one complete streamed image")
    identity = next(iter(parts))
    require(set(parts[identity]) == set(range(totals[identity])),
            f"{name}: missing image chunks")
    png = b"".join(parts[identity][i] for i in range(totals[identity]))
    require(final[identity] == (digest(png), len(png)),
            f"{name}: streamed media digest mismatch")
    require((root / f"{name}.png").read_bytes() == png,
            f"{name}: retained PNG differs from raw SSE")
    return decode_png(png)


def verify_manifest(root):
    manifest = root / "SHA256SUMS"
    listed = set()
    for line in manifest.read_text().splitlines():
        checksum, relative = line.split("  ", 1)
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts and
                relative not in listed, "unsafe or duplicate manifest path")
        listed.add(relative)
        require(digest((root / path).read_bytes()) == checksum,
                f"producer checksum mismatch: {relative}")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name not in
              ("SHA256SUMS", "SHA256SUMS.independent", "independent-verification.json")}
    require(actual == listed,
            f"producer manifest coverage differs: missing={actual - listed}, extra={listed - actual}")
    return len(listed)


def verify(root):
    require(root.is_dir(), "evidence directory is missing")
    require(not (root / "independent-verification.json").exists() and
            not (root / "SHA256SUMS.independent").exists(),
            "independent output already exists; use a fresh evidence copy")
    producer_file_count = verify_manifest(root)
    summary = read_json(root, "summary.json")
    require(summary.get("suite") == "image2" and summary.get("status") == "PASS" and
            summary.get("device_index") == 1 and summary.get("failures") == [] and
            summary.get("public_request_count") == 9,
            "producer summary is not the passed nine-request GPU 1 image2 gate")

    names = ("image2-warmup-off", "image2-json-off", "image2-json-on",
             "image2-json-warm-on", "image2-sse-off", "image2-sse-on",
             "image2-native-failed", "image2-cancelled", "image2-after-cancel-on")
    requests = {}
    for name in names:
        raw = (root / f"{name}.request.json").read_bytes()
        require(raw.endswith(b"\n"), f"{name}: request framing missing")
        body = json.loads(raw)
        requests[name] = body
        require(canonical(body) + b"\n" == raw, f"{name}: noncanonical request bytes")
        require(summary["canonical_request_sha256"][name] == digest(canonical(body)),
                f"{name}: request hash mismatch")
        require(body["temperature"] == 0 and body["top_p"] == 1 and
                body["seed"] == 42 and body["max_tokens"] == 1 and
                body["modalities"] == ["text", "image"],
                f"{name}: VLM/sampler settings changed")
        require(body["image"]["seed"] == 1234 and
                body["image"]["size"] == "512x512" and
                body["image"]["output_format"] == "png" and
                body["image"]["steps"] == (40 if name == "image2-cancelled" else 2),
                f"{name}: image settings changed")
        require(body.get("stream") is ("sse" in name or name == "image2-cancelled"),
                f"{name}: transport mode changed")
        expected_telemetry = name not in ("image2-warmup-off", "image2-json-off",
                                          "image2-sse-off")
        require(body.get("telemetry") == ({"native_allocator": True}
                if expected_telemetry else None), f"{name}: telemetry opt-in changed")

    json_names = ("image2-warmup-off", "image2-json-off", "image2-json-on",
                  "image2-json-warm-on", "image2-native-failed",
                  "image2-after-cancel-on")
    responses = {name: json.loads((root / f"{name}.response.json").read_bytes())
                 for name in json_names}
    media = {}
    for name in ("image2-warmup-off", "image2-json-off", "image2-json-on",
                 "image2-json-warm-on", "image2-after-cancel-on"):
        media[name] = json_media(root, name, responses[name])
    for name in ("image2-warmup-off", "image2-json-off"):
        require(telemetry(responses[name]) is None and "telemetry" not in responses[name],
                f"{name}: default-off JSON emitted telemetry")

    covered = {}
    for name in ("image2-json-on", "image2-json-warm-on", "image2-after-cancel-on"):
        envelope = telemetry(responses[name])
        require(envelope is not None and
                envelope.get("whole_device_nvml") == "separate_measurement_resources",
                f"{name}: telemetry/NVML separation missing")
        covered[name] = covered_image(envelope, "completed", True)
        stage_row(envelope, "vlm", "qwen38")

    failed = responses["image2-native-failed"]
    require(any(leg.get("stage") == "image" and leg.get("state") == "failed"
                for leg in failed.get("legs", [])),
            "native edit rejection did not retain a failed image leg")
    covered["image2-native-failed"] = covered_image(
        telemetry(failed), "failed", False)

    streams = {}
    for name in ("image2-sse-off", "image2-sse-on", "image2-cancelled"):
        raw = (root / f"{name}.response.sse").read_bytes()
        frames = parse_sse(raw)
        require(frames == read_json(root, f"{name}.frames.json"),
                f"{name}: saved frames differ from raw SSE")
        streams[name] = frames
    require(telemetry(streams["image2-sse-off"][0]) is None and
            all("telemetry" not in frame for frame in streams["image2-sse-off"]),
            "default-off SSE emitted telemetry")
    on_envelopes = [telemetry(frame) for frame in streams["image2-sse-on"]
                    if telemetry(frame) is not None]
    require(len(on_envelopes) == 1, "SSE-on did not emit one terminal envelope")
    covered["image2-sse-on"] = covered_image(on_envelopes[0], "completed", True)
    for name in ("image2-sse-off", "image2-sse-on"):
        media[name] = sse_media(root, name, streams[name])

    cancel_envelopes = [telemetry(frame) for frame in streams["image2-cancelled"]
                        if telemetry(frame) is not None]
    require(len(cancel_envelopes) == 1 and
            cancel_envelopes[0].get("lifecycle_status") == "cancelled",
            "image cancellation did not emit one cancelled terminal")
    cancel_envelope = cancel_envelopes[0]
    cancel_row = stage_row(cancel_envelope, "image")
    if cancel_row.get("native_stage") == "image":
        covered["image2-cancelled"] = covered_image(cancel_envelope, "cancelled", True)
        cancel_coverage = "COVERED"
    else:
        require(cancel_row.get("native_stage") == "unknown" and
                cancel_row.get("coverage") == "NOT_COVERED" and
                cancel_row.get("reason") == "stage_terminal_snapshot_missing",
                "unexpected cancellation attribution")
        null_bytes(cancel_row)
        cancel_coverage = "NOT_COVERED_STAGE_TERMINAL_SNAPSHOT_MISSING"
    request_id = cancel_envelope["request_id"]
    trigger = read_json(root, "image2-cancelled.trace-trigger.json")
    event = trigger.get("event", {})
    require(event.get("request_id") == request_id and
            event.get("event") == "legs_snapshot" and
            event.get("phase") == "outputs_started" and
            any(leg.get("stage") == "image" and leg.get("state") == "running"
                for leg in event.get("legs", [])) and
            trigger.get("abort", {}).get("status") == 200,
            "trace did not prove a running image child was aborted")
    native_log = (root / "image2-native-log-tail.txt").read_bytes()
    marker = ("image request aborted backend=2.1 request_id=" +
              request_id + ":image").encode()
    require(marker in native_log and b"denoise step" in native_log,
            "native log lacks exact image cancellation")
    success_names = ("image2-warmup-off", "image2-json-off", "image2-json-on",
                     "image2-json-warm-on", "image2-sse-off", "image2-sse-on",
                     "image2-after-cancel-on")
    warm_reuse = read_json(root, "image2-warm-reuse.json")
    require(set(warm_reuse) == set(success_names), "warm-reuse request set differs")
    for name in success_names:
        record = warm_reuse[name]
        child = record["request_id"] + ":image"
        match = re.search(
            rb"image request backend=2\.1 request_id=" + re.escape(child.encode()) +
            rb"[^\n]*\bload_ms=([0-9]+)(?:\.[0-9]+)?", native_log)
        require(match is not None and int(match.group(1)) == record["load_ms"],
                f"{name}: native warm-reuse record does not match raw log")
        require((record["load_ms"] > 0) if name == "image2-warmup-off"
                else (record["load_ms"] == 0),
                f"{name}: incorrect pre-request/warm load boundary")
    require(read_json(root, "image2-final-health.json").get("in_flight") == 0,
            "gateway did not drain in-flight requests")

    order = ("image2-json-on", "image2-json-warm-on", "image2-sse-on",
             "image2-native-failed", "image2-cancelled", "image2-after-cancel-on")
    epoch_rows = [(name, covered[name]) for name in order if name in covered]
    epochs = [row["epoch"]["id"] for _, row in epoch_rows]
    require(len(epochs) >= 5 and len(set(epochs)) == len(epochs) and
            all(b > a for a, b in zip(epochs, epochs[1:])),
            f"image epochs are not unique and monotonic in raw request order: {epochs}")

    parity = {
        "json_off_on": normalized_digest(responses["image2-json-off"],
                                          responses["image2-json-off"]["id"]),
        "json_warm": normalized_digest(responses["image2-json-on"],
                                        responses["image2-json-on"]["id"]),
        "json_recovery": normalized_digest(responses["image2-after-cancel-on"],
                                            responses["image2-after-cancel-on"]["id"]),
        "sse_off_on": normalized_digest(streams["image2-sse-off"],
                                         streams["image2-sse-off"][0]["id"]),
    }
    require(parity["json_off_on"] == normalized_digest(
                responses["image2-json-on"], responses["image2-json-on"]["id"]),
            "independent JSON off/on parity failed")
    require(parity["json_warm"] == normalized_digest(
                responses["image2-json-warm-on"], responses["image2-json-warm-on"]["id"]),
            "independent warm parity failed")
    require(parity["json_recovery"] == normalized_digest(
                responses["image2-json-on"], responses["image2-json-on"]["id"]),
            "independent recovery parity failed")
    require(parity["sse_off_on"] == normalized_digest(
                streams["image2-sse-on"], streams["image2-sse-on"][0]["id"]),
            "independent SSE off/on parity failed")

    png_hashes = {media[name]["png_sha256"] for name in success_names}
    rgba_hashes = {media[name]["rgba_sha256"] for name in success_names}
    require(len(png_hashes) == 1 and len(rgba_hashes) == 1,
            "independent exact PNG/RGBA parity failed")
    for name in success_names:
        require(media[name] == summary["media_hashes"][name],
                f"{name}: independently decoded media record differs")

    nvml = read_json(root, "nvml-samples.json")
    samples = nvml.get("samples", [])
    require(nvml.get("scope") == "whole-device-used-memory" and
            nvml.get("poll_interval_ms") == 50 and len(samples) >= 2 and
            all(row.get("device_index") == 1 and
                row.get("scope") == "whole-device-used-memory" and
                type(row.get("memory_used_mib")) is int and row["memory_used_mib"] >= 0
                for row in samples),
            "independent NVML record has wrong scope/device")
    memory = [row["memory_used_mib"] for row in samples]

    return {
        "schema": "umminfer-image2-request-allocator-independent-v1",
        "status": "PASS", "producer_file_count": producer_file_count,
        "public_requests": len(names), "covered_image_epochs": epochs,
        "allocated_bytes_peak": {name: row["allocated_bytes_peak"]
                                 for name, row in epoch_rows},
        "allocated_bytes_final": {name: row["allocated_bytes_final"]
                                  for name, row in epoch_rows},
        "allocation_events": {name: row["sample_coverage"]["allocation_events"]
                              for name, row in epoch_rows},
        "free_events": {name: row["sample_coverage"]["free_events"]
                        for name, row in epoch_rows},
        "cancellation_coverage": cancel_coverage,
        "normalized_protocol_sha256": parity,
        "png_sha256": next(iter(png_hashes)), "rgba_sha256": next(iter(rgba_hashes)),
        "png_bytes": media["image2-json-on"]["png_bytes"],
        "whole_device_nvml": {"sample_count": len(samples),
                              "memory_used_mib_min": min(memory),
                              "memory_used_mib_max": max(memory),
                              "scope": "separate_whole_device_measurement"},
        "producer_imported": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_dir", type=Path)
    args = parser.parse_args(argv)
    report = verify(args.evidence_dir)
    report_bytes = json.dumps(report, indent=2, sort_keys=True).encode() + b"\n"
    (args.evidence_dir / "independent-verification.json").write_bytes(report_bytes)
    rows = []
    for path in sorted(args.evidence_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.independent":
            rows.append(f"{digest(path.read_bytes())}  "
                        f"{path.relative_to(args.evidence_dir).as_posix()}\n")
    (args.evidence_dir / "SHA256SUMS.independent").write_text("".join(rows))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL independent image2 verification: {exc}")
        raise SystemExit(2)
