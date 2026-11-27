#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent, stdlib-first raw-evidence verifier for Task #74 I2V gates.

This file deliberately imports neither the live driver nor any producer helper.
It parses MessagePack itself, reconstructs public/native media bindings, decodes
RGB with FFmpeg, recomputes all statistics, and enforces a complete manifest that
excludes only its own top-level SHA256SUMS entry.
"""

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import struct
import subprocess
import sys

EXPECTED_CASES = {
    "wide": {"source": [704, 480], "target": [704, 480]},
    "tall": {"source": [480, 704], "target": [480, 704]},
    "square": {"source": [512, 512], "target": [704, 480]},
}
REPETITIONS = 3
INTERLEAVING_SEED = 74062
FIRST_FRAME_MAX_MAE = 48.0
MOTION_MIN_MAE = 1.0


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


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


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def size_text(value):
    return f"{value[0]}x{value[1]}"


def close(actual, expected, label):
    require(math.isfinite(actual) and math.isfinite(expected)
            and abs(actual - expected) <= 1e-9 * max(1.0, abs(expected)),
            f"numeric mismatch for {label}: {actual} != {expected}")


class MsgpackReader:
    """Bounded MessagePack decoder for the retained native protocol frames."""

    def __init__(self, raw):
        self.raw = memoryview(raw)
        self.offset = 0

    def take(self, count):
        require(count >= 0 and self.offset + count <= len(self.raw),
                "truncated MessagePack value")
        result = self.raw[self.offset:self.offset + count]
        self.offset += count
        return result

    def byte(self):
        return self.take(1)[0]

    def number(self, fmt, size):
        return struct.unpack(fmt, self.take(size))[0]

    def array(self, count, depth):
        require(count <= len(self.raw), "implausible MessagePack array length")
        return [self.value(depth + 1) for _ in range(count)]

    def mapping(self, count, depth):
        require(count <= len(self.raw), "implausible MessagePack map length")
        result = {}
        for _ in range(count):
            key = self.value(depth + 1)
            require(isinstance(key, (str, int, float, bool)) or key is None,
                    "unsupported MessagePack map key")
            result[key] = self.value(depth + 1)
        return result

    def value(self, depth=0):
        require(depth <= 64, "MessagePack nesting limit exceeded")
        marker = self.byte()
        if marker <= 0x7f:
            return marker
        if marker >= 0xe0:
            return marker - 0x100
        if 0x80 <= marker <= 0x8f:
            return self.mapping(marker & 0x0f, depth)
        if 0x90 <= marker <= 0x9f:
            return self.array(marker & 0x0f, depth)
        if 0xa0 <= marker <= 0xbf:
            return str(self.take(marker & 0x1f), "utf-8")
        if marker == 0xc0:
            return None
        if marker == 0xc2:
            return False
        if marker == 0xc3:
            return True
        if marker in (0xc4, 0xc5, 0xc6):
            size = self.number({0xc4: ">B", 0xc5: ">H", 0xc6: ">I"}[marker],
                               {0xc4: 1, 0xc5: 2, 0xc6: 4}[marker])
            return self.take(size).tobytes()
        if marker in (0xc7, 0xc8, 0xc9) or 0xd4 <= marker <= 0xd8:
            raise ValueError("MessagePack extension values are not used by this protocol")
        if marker == 0xca:
            return self.number(">f", 4)
        if marker == 0xcb:
            return self.number(">d", 8)
        if marker in (0xcc, 0xcd, 0xce, 0xcf):
            sizes = {0xcc: (">B", 1), 0xcd: (">H", 2),
                     0xce: (">I", 4), 0xcf: (">Q", 8)}
            fmt, size = sizes[marker]
            return self.number(fmt, size)
        if marker in (0xd0, 0xd1, 0xd2, 0xd3):
            sizes = {0xd0: (">b", 1), 0xd1: (">h", 2),
                     0xd2: (">i", 4), 0xd3: (">q", 8)}
            fmt, size = sizes[marker]
            return self.number(fmt, size)
        if marker in (0xd9, 0xda, 0xdb):
            sizes = {0xd9: (">B", 1), 0xda: (">H", 2), 0xdb: (">I", 4)}
            fmt, size = sizes[marker]
            length = self.number(fmt, size)
            return str(self.take(length), "utf-8")
        if marker in (0xdc, 0xdd):
            length = self.number(">H" if marker == 0xdc else ">I",
                                 2 if marker == 0xdc else 4)
            return self.array(length, depth)
        if marker in (0xde, 0xdf):
            length = self.number(">H" if marker == 0xde else ">I",
                                 2 if marker == 0xde else 4)
            return self.mapping(length, depth)
        raise ValueError(f"unsupported MessagePack marker 0x{marker:02x}")


def unpack_msgpack(raw):
    reader = MsgpackReader(raw)
    result = reader.value()
    require(reader.offset == len(raw), "trailing bytes after MessagePack value")
    return result


def png_dimensions(raw):
    require(len(raw) >= 24 and raw[:8] == b"\x89PNG\r\n\x1a\n"
            and raw[12:16] == b"IHDR", "invalid PNG/IHDR")
    return list(struct.unpack(">II", raw[16:24]))


def decode_data_url(value):
    require(isinstance(value, str) and value.startswith("data:image/png;base64,"),
            "unexpected native/public image URL")
    return base64.b64decode(value.split(",", 1)[1], validate=True)


def ffmpeg_rgb(ffmpeg, media, width, height, frame=None, evaluation=None):
    filters = []
    if frame is not None:
        filters.append(f"select=eq(n\\,{frame})")
    if evaluation is not None:
        eval_w, eval_h = evaluation
        filters.extend([
            f"scale={eval_w}:{eval_h}:force_original_aspect_ratio=increase",
            f"crop={eval_w}:{eval_h}"])
    command = [ffmpeg, "-v", "error", "-i", str(media)]
    if filters:
        command += ["-vf", ",".join(filters)]
    command += ["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    result = subprocess.run(command, check=True, capture_output=True, timeout=90)
    out_w, out_h = evaluation if evaluation is not None else (width, height)
    require(len(result.stdout) == out_w * out_h * 3, "decoded RGB size mismatch")
    return result.stdout


def probe_video(ffprobe, path, expected):
    result = subprocess.run([
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=codec_name,width,height,nb_frames,r_frame_rate,pix_fmt",
        "-of", "json", str(path)], check=True, capture_output=True, text=True, timeout=30)
    streams = json.loads(result.stdout)["streams"]
    require(len(streams) == 1, "expected exactly one video stream")
    stream = streams[0]
    require(stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
            and [stream["width"], stream["height"]] == expected
            and int(stream["nb_frames"]) == 29 and stream["r_frame_rate"] == "16/1",
            f"unexpected video stream: {stream}")
    return stream


def mean_absolute(first, second):
    require(len(first) == len(second) and bool(first), "RGB buffers differ or empty")
    return sum(abs(a - b) for a, b in zip(first, second)) / len(first)


def quantile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def expected_summary(values):
    median = statistics.median(values)
    q1 = quantile(values, 0.25)
    q3 = quantile(values, 0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return {
        "count": len(values), "all_ms": values, "median_ms": median,
        "p95_ms": quantile(values, 0.95), "iqr_ms": iqr,
        "mad_ms": statistics.median([abs(value - median) for value in values]),
        "iqr_outliers_ms": [value for value in values if value < low or value > high],
        "outlier_policy": "retained_in_all_statistics",
    }


def check_summary(actual, values, label):
    expected = expected_summary(values)
    require(actual.get("count") == len(values)
            and actual.get("outlier_policy") == "retained_in_all_statistics",
            f"summary population changed for {label}")
    require(actual.get("all_ms") == values, f"raw observations changed for {label}")
    for key in ("median_ms", "p95_ms", "iqr_ms", "mad_ms"):
        close(actual[key], expected[key], f"{label}.{key}")
    require(actual.get("iqr_outliers_ms") == expected["iqr_outliers_ms"],
            f"outliers were filtered or changed for {label}")


def check_manifest(root):
    manifest = root / "SHA256SUMS"
    require(manifest.is_file(), "SHA256SUMS is missing")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        require(re.fullmatch(r"[0-9a-f]{64}", digest)
                and relative and relative != "SHA256SUMS"
                and not relative.startswith("/") and ".." not in Path(relative).parts,
                f"invalid manifest row: {line!r}")
        require(relative not in expected, f"duplicate manifest row: {relative}")
        expected[relative] = digest
    actual = {path.relative_to(root).as_posix(): path for path in root.rglob("*")
              if path.is_file() and path != manifest}
    require(set(actual) == set(expected),
            f"manifest set mismatch: missing={sorted(set(expected)-set(actual))}, "
            f"extra={sorted(set(actual)-set(expected))}")
    for relative, digest in expected.items():
        require(sha256_file(actual[relative])["sha256"] == digest,
                f"manifest digest mismatch: {relative}")
    return {"file_count": len(expected), "manifest_sha256": sha256_file(manifest)["sha256"]}


def model_files(root):
    files = {}
    for path in sorted(item for item in Path(root).rglob("*") if item.is_file()):
        files[path.relative_to(root).as_posix()] = sha256_file(path)
    return files


def check_verifier_identity(recorded_identity, historical_sha256, current_identity):
    require(recorded_identity["sha256"]
            == (historical_sha256 or current_identity["sha256"]),
            "independent verifier identity mismatch")


def check_optional_identities(report, args):
    provenance = report["provenance"]
    pairs = [
        ("server_binary", args.server_binary),
        ("baseline_wan_binary", args.baseline_wan_binary),
        ("candidate_wan_binary", args.candidate_wan_binary),
        ("flux_stage", args.flux_stage),
        ("ffmpeg", args.ffmpeg), ("ffprobe", args.ffprobe),
    ]
    for key, path in pairs:
        if path is not None and not args.skip_execution_binary_recheck:
            require(provenance.get(key) == sha256_file(path), f"identity mismatch: {key}")
    harness = Path(__file__).with_name("i2v_geometry_latency_live.py")
    require(provenance["live_harness"] == sha256_file(harness), "live harness identity mismatch")
    check_verifier_identity(provenance["independent_verifier"],
                            args.recorded_verifier_sha256, sha256_file(__file__))
    for key, path in (("wan22", args.wan_model), ("flux2", args.flux_model)):
        if path is not None:
            expected = model_files(path)
            require(report["models"][key]["files"] == expected,
                    f"model file identity mismatch: {key}")
            require(report["models"][key]["file_count"] == len(expected)
                    and report["models"][key]["tree_sha256"] == sha256_bytes(canonical(expected)),
                    f"model tree identity mismatch: {key}")


def native_phase(log_path, request_id):
    matches = []
    for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
        if "video request backend=" in line and f"request_id={request_id}" in line:
            matches.append(line)
    require(len(matches) == 1, f"expected one native generation line for {request_id}")
    fields = dict(re.findall(r"([a-z0-9_]+)=([^\s]+)", matches[0]))
    require(fields.get("backend") == "native-cuda-wan" and fields.get("i2v") == "1",
            f"not native I2V: {matches[0]}")
    for key in ("image_decode_ms", "image_resize_ms", "vae_encode_ms", "vae_decode_ms",
                "latent_init_ms", "umt5_ms", "dit_ms", "vae_ms", "denoise_ms",
                "encode_ms", "elapsed_ms"):
        fields[key] = float(fields[key])
        require(math.isfinite(fields[key]) and fields[key] >= 0.0, f"invalid {key}")
    fields["log_sha256"] = sha256_bytes(matches[0].encode())
    return fields


def expected_schedule():
    result = []
    for repetition in range(REPETITIONS):
        arms = ["baseline", "candidate"]
        random.Random(INTERLEAVING_SEED + repetition).shuffle(arms)
        for arm in arms:
            result.append({"repetition": repetition, "arm": arm,
                           "shapes": list(EXPECTED_CASES), "warmups_per_shape": 1,
                           "measured_per_shape": 1})
    return result


def verify_preregistration(document, profile_sha):
    require(document["format"] == "umminfer-i2v-latency-preregistration-v1"
            and document["profile_report_sha256"] == profile_sha
            and document["primary_metric"] == "median_paired_public_complete_artifact_ms"
            and document["repetitions"] == REPETITIONS
            and document["warmup_per_shape_per_block"] == 1
            and document["interleaving_seed"] == INTERLEAVING_SEED,
            "preregistration identity or population mismatch")
    require(document["settings"] == {"frames": 29, "steps": 20, "fps": 16,
                                      "guidance": 5.0, "seed": 42,
                                      "teacache_thresh": 0.0},
            "preregistered generation settings changed")
    gates = document["regression_gates"]
    require(gates["exact_mp4_parity_with_profile_and_paired_baseline"] is True
            and gates["decoded_dimensions_frames_codec"] is True
            and gates["source_hash_and_native_submit_binding"] is True
            and gates["malformed_dimension_rejection_and_valid_recovery"] is True
            and gates["one_native_terminal_and_final_public_in_flight_zero"] is True
            and gates["first_frame_mae_max"] == FIRST_FRAME_MAX_MAE
            and gates["motion_mae_min"] == MOTION_MIN_MAE
            and gates["maximum_median_latency_regression_percent"] == 5.0,
            "preregistration gates are incomplete")


def verify_bundle(root, args, nested=False):
    root = Path(root).resolve()
    manifest = check_manifest(root)
    report = json.loads((root / "report.json").read_text())
    require(report["format"] == "umminfer-i2v-geometry-latency-v1"
            and report["status"] == "SEALED_AWAITING_INDEPENDENT_VERIFICATION"
            and report["mode"] in ("profile", "candidate")
            and report["geometry_cases"] == EXPECTED_CASES,
            "report format, status, mode, or geometry cases mismatch")
    require(re.fullmatch(r"[0-9a-f]{40}", report["base_commit"])
            and re.fullmatch(r"[0-9a-f]{40}", report["execution_commit"])
            and report["physical_gpu"] in (0, 1, 7)
            and report["port"] != 43301,
            "commit, GPU, or service-isolation contract mismatch")
    require(report["final_health"] == {"http_status": 200, "body": {
        "in_flight": 0, "status": "ok"}}, "final public in-flight state is not zero")
    for key in ("before", "after-all-requests-before-server-stop"):
        state = report["gpu"][key]
        require(state["index"] == report["physical_gpu"]
                and state["memory_used_mib"] <= 16
                and state["compute_processes"] == []
                and (key != "before" or state["utilization_percent"] == 0),
                f"GPU is not idle at {key}")
    for process in report["processes"]:
        require(process["stop_monotonic_ns"] is not None and process["returncode"] is not None
                and process["stop_error"] is None,
                f"task-owned process did not stop cleanly: {process}")
        if process["name"] != "server" and process["name"] != "entry":
            require(process["returncode"] == 0,
                    f"native/image stage returned nonzero: {process}")
    if not nested:
        check_optional_identities(report, args)

    sources = {}
    for shape, case in EXPECTED_CASES.items():
        source = report["sources"][shape]
        require(source["source_size"] == size_text(case["source"]),
                f"source size metadata mismatch: {shape}")
        raw = (root / source["source_file"]).read_bytes()
        require(png_dimensions(raw) == case["source"]
                and sha256_bytes(raw) == source["source_sha256"],
                f"real FLUX source bytes/dimensions mismatch: {shape}")
        target_text = size_text(case["target"])
        evaluation = source["evaluation_rgb"][target_text]
        eval_size = [evaluation["width"], evaluation["height"]]
        require(eval_size == [case["target"][0] // 8, case["target"][1] // 8],
                f"evaluation geometry mismatch: {shape}")
        expected_eval = ffmpeg_rgb(args.ffmpeg, root / source["source_file"],
                                   *case["source"], evaluation=eval_size)
        require((root / evaluation["file"]).read_bytes() == expected_eval
                and evaluation["sha256"] == sha256_bytes(expected_eval),
                f"source center-cover evaluation RGB mismatch: {shape}")
        if "full_rgb" in source:
            expected_full = ffmpeg_rgb(args.ffmpeg, root / source["source_file"],
                                       *case["source"])
            require((root / source["full_rgb"]["file"]).read_bytes() == expected_full
                    and source["full_rgb"]["sha256"] == sha256_bytes(expected_full),
                    f"source full RGB mismatch: {shape}")
        if report["mode"] == "profile":
            public_request = json.loads((root / source["public_request_file"]).read_text())
            response = (root / source["public_response_file"]).read_bytes()
            require(public_request == {"kind": "image", "prompt": (
                        "A single vivid red paper lantern above a misty blue mountain lake at dawn, "
                        "cinematic composition, clear subject, gentle ripples"),
                        "size": size_text(case["source"]), "steps": 4, "seed": 42}
                    and response == raw and source["http_status"] == 200
                    and source["public_response_sha256"] == sha256_bytes(raw),
                    f"public FLUX response binding mismatch: {shape}")
        sources[shape] = (raw, expected_eval)

    request_ids = set()
    successful = {shape: [] for shape in EXPECTED_CASES}
    measured = []
    for run in report["runs"] + report["controls"]:
        shape = run["shape"]
        case = EXPECTED_CASES[shape]
        source, source_eval = sources[shape]
        require(run["source_sha256"] == sha256_bytes(source), "source digest changed")
        body_raw = (root / run["public_request_file"]).read_bytes()
        body = json.loads(body_raw)
        response = (root / run["public_response_file"]).read_bytes()
        require(sha256_bytes(body_raw) == run["public_request_sha256"]
                and sha256_bytes(response) == run["public_response_sha256"]
                and len(response) == run["public_response_bytes"],
                f"public raw request/response hash mismatch: {run['run_id']}")
        require(body["kind"] == "video" and body["image"] is not None
                and decode_data_url(body["image"]) == source
                and body["prompt"] == (
                    "The red lantern sways gently while the camera slowly orbits above the misty lake")
                and body["steps"] == 20 and body["num_frames"] == 29
                and body["fps"] == 16 and body["guidance_scale"] == 5.0
                and body["seed"] == 42 and body["teacache_thresh"] == 0.0,
                f"public video request drift: {run['run_id']}")
        target = [500, 480] if run["role"] == "malformed_dimension" else case["target"]
        require(body["size"] == size_text(target), f"requested target mismatch: {run['run_id']}")
        native_input_raw = (root / run["native_input_file"]).read_bytes()
        native_output_raw = (root / run["native_output_file"]).read_bytes()
        native_input = unpack_msgpack(native_input_raw)
        native_output = unpack_msgpack(native_output_raw)
        require(native_input["type"] == "submit" and native_output["type"] == "complete"
                and native_input["request_id"] == native_output["request_id"]
                == run["native_request_id"],
                f"native terminal identity mismatch: {run['run_id']}")
        require(run["native_request_id"] not in request_ids,
                f"duplicate native request id: {run['native_request_id']}")
        request_ids.add(run["native_request_id"])
        inputs = native_input["data"]["request"]["inputs"]
        native_source = decode_data_url(inputs["image"]) if isinstance(inputs["image"], str) \
            else inputs["image"]
        require(native_source == source and inputs["size"] == body["size"]
                and inputs["steps"] == 20 and inputs["num_frames"] == 29
                and inputs["fps"] == 16 and inputs["guidance_scale"] == 5.0
                and inputs["seed"] == 42 and inputs["teacache_thresh"] == 0.0,
                f"native source/settings binding mismatch: {run['run_id']}")
        submit = run["submit_monotonic_ns"]
        headers = run["response_headers_monotonic_ns"]
        complete = run["complete_monotonic_ns"]
        first_body = run["first_body_monotonic_ns"]
        require(run["source_preparation_ms"] >= 0.0
                and run["native_submit_monotonic_ns"] <= run["native_complete_monotonic_ns"]
                <= headers and submit <= headers <= complete
                and (first_body is None or headers <= first_body <= complete),
                f"nonmonotonic native/public timing: {run['run_id']}")
        close(run["public_first_output_ms"], (headers - submit) / 1e6,
              run["run_id"] + ".first_output")
        close(run["public_complete_artifact_ms"], (complete - submit) / 1e6,
              run["run_id"] + ".complete")
        close(run["native_complete_to_public_first_output_ms"],
              (headers - run["native_complete_monotonic_ns"]) / 1e6,
              run["run_id"] + ".artifact_emission")
        if first_body is not None:
            close(run["public_first_artifact_byte_ms"], (first_body - submit) / 1e6,
                  run["run_id"] + ".first_body")
        else:
            require(run["public_first_artifact_byte_ms"] is None, "missing-body timing mismatch")
        if run["http_status"] != 200:
            require(run["role"] == "malformed_dimension" and target == [500, 480]
                    and run["http_status"] == 500
                    and "divisible by 32" in response.decode("utf-8")
                    and native_output["success"] is False
                    and not (native_output.get("result") or {}).get("data"),
                f"malformed-dimension control did not fail closed: {run['run_id']}")
            continue
        require(native_output["success"] is True, f"native generation failed: {run['run_id']}")
        media = native_output["result"]["data"]
        require(media == response and native_output["result"]["size"] == size_text(target)
                and native_output["result"]["frames"] == 29,
                f"native/public artifact mismatch: {run['run_id']}")
        media_path = root / run["media_file"]
        require(media_path.read_bytes() == media
                and run["media_sha256"] == sha256_bytes(media),
                f"retained media mismatch: {run['run_id']}")
        require(probe_video(args.ffprobe, media_path, target) == run["video_stream"],
                f"ffprobe metadata mismatch: {run['run_id']}")
        phase = native_phase(root / f"video-r{run['block']}-{run['arm']}.log",
                             run["native_request_id"])
        require(phase == run["native_phase"]
                and phase["size"] == size_text(target)
                and phase["source_size"] == size_text(case["source"]),
                f"native phase/stale geometry mismatch: {run['run_id']}")
        first = ffmpeg_rgb(args.ffmpeg, media_path, *target, frame=0)
        last = ffmpeg_rgb(args.ffmpeg, media_path, *target, frame=28)
        eval_size = [target[0] // 8, target[1] // 8]
        first_eval = ffmpeg_rgb(args.ffmpeg, media_path, *target, frame=0,
                                evaluation=eval_size)
        last_eval = ffmpeg_rgb(args.ffmpeg, media_path, *target, frame=28,
                               evaluation=eval_size)
        for label, expected in (("first_full", first), ("last_full", last),
                                ("first_eval", first_eval), ("last_eval", last_eval)):
            retained = run["decoded_rgb"][label]
            require((root / retained["file"]).read_bytes() == expected
                    and retained["sha256"] == sha256_bytes(expected)
                    and retained["bytes"] == len(expected),
                    f"retained {label} RGB mismatch: {run['run_id']}")
        first_mae = mean_absolute(source_eval, first_eval)
        motion_mae = mean_absolute(first_eval, last_eval)
        close(run["first_frame_mae"], first_mae, run["run_id"] + ".first_mae")
        close(run["motion_mae"], motion_mae, run["run_id"] + ".motion_mae")
        require(first_mae <= FIRST_FRAME_MAX_MAE and motion_mae >= MOTION_MIN_MAE,
                f"continuity/motion gate failed: {run['run_id']}")
        successful[shape].append(run)
        if run["role"] == "measured":
            measured.append(run)
    for shape, shape_runs in successful.items():
        require(shape_runs and len({run["media_sha256"] for run in shape_runs}) == 1,
                f"fixed-seed media are not reproducible: {shape}")
    require(successful["wide"][0]["media_sha256"] != successful["square"][0]["media_sha256"],
            "wide and square anchors produced identical wide output")
    recoveries = [run for run in report["controls"] if run["role"] == "post_failure_recovery"]
    require(len(recoveries) == 1 and recoveries[0]["http_status"] == 200
            and recoveries[0]["media_sha256"] == successful["wide"][0]["media_sha256"],
            "post-failure exact recovery control failed")
    require(sum(run["role"] == "malformed_dimension" for run in report["controls"]) == 1,
            "expected one malformed-dimension control")

    profile = report["profile"]
    for shape in EXPECTED_CASES:
        values = [run["public_complete_artifact_ms"] for run in measured
                  if run["shape"] == shape]
        check_summary(profile["warm_by_shape"][shape], values, f"warm.{shape}")
        phases = profile["warm_native_phases_by_shape"][shape]
        for phase in ("image_decode_ms", "image_resize_ms", "vae_encode_ms", "umt5_ms",
                      "latent_init_ms", "dit_ms", "vae_decode_ms", "encode_ms"):
            check_summary(phases[phase], [run["native_phase"][phase] for run in measured
                          if run["shape"] == shape], f"{shape}.{phase}")
    check_summary(profile["source_preparation"],
                  [run["source_preparation_ms"] for run in measured], "source_preparation")
    check_summary(profile["public_first_artifact_byte"],
                  [run["public_first_artifact_byte_ms"] for run in measured], "first_body")
    check_summary(profile["public_complete_artifact"],
                  [run["public_complete_artifact_ms"] for run in measured], "public_complete")
    require(len(profile["cold_or_warmup"]) == sum(
        run["role"] == "cold_or_warmup" for run in report["runs"]),
        "cold/warmup observations were dropped")

    candidate_result = None
    if report["mode"] == "candidate":
        require(not nested, "nested candidate bundles are forbidden")
        baseline_root = root / "baseline-reference"
        baseline_result = verify_bundle(baseline_root, args, nested=True)
        baseline_manifest = baseline_result["manifest"]
        baseline_report = baseline_result["report"]
        prereg_raw = (root / "preregistration.json").read_bytes()
        prereg = json.loads(prereg_raw)
        profile_sha = sha256_file(baseline_root / "report.json")["sha256"]
        verify_preregistration(prereg, profile_sha)
        require(report["preregistration_sha256"] == sha256_bytes(prereg_raw)
                and baseline_report["mode"] == "profile"
                and report["provenance"]["baseline_wan_binary"]
                == baseline_report["provenance"]["baseline_wan_binary"],
                "preregistration/profile identity or instrumented baseline mismatch")
        schedule = json.loads((root / "candidate-schedule.json").read_text())
        require(schedule == expected_schedule(), "candidate arm schedule was not preregistered")
        analysis = report["candidate_analysis"]
        require(analysis["schedule"] == schedule, "report schedule differs from frozen schedule")
        paired = []
        for repetition in range(REPETITIONS):
            for shape in EXPECTED_CASES:
                baseline = [run for run in measured if run["arm"] == "baseline"
                            and run["repetition"] == repetition and run["shape"] == shape]
                candidate = [run for run in measured if run["arm"] == "candidate"
                             and run["repetition"] == repetition and run["shape"] == shape]
                require(len(baseline) == len(candidate) == 1,
                        f"paired population mismatch r{repetition} {shape}")
                baseline = baseline[0]
                candidate = candidate[0]
                profile_matches = [run for run in baseline_report["runs"]
                                   if run["shape"] == shape and run["role"] == "measured"
                                   and run["repetition"] == 0]
                require(len(profile_matches) == 1
                        and candidate["media_sha256"] == baseline["media_sha256"]
                        == profile_matches[0]["media_sha256"],
                        f"exact candidate/baseline/profile parity failed r{repetition} {shape}")
                paired.append({
                    "repetition": repetition, "shape": shape,
                    "baseline_run_id": baseline["run_id"], "candidate_run_id": candidate["run_id"],
                    "baseline_ms": baseline["public_complete_artifact_ms"],
                    "candidate_ms": candidate["public_complete_artifact_ms"],
                    "candidate_minus_baseline_ms": (
                        candidate["public_complete_artifact_ms"]
                        - baseline["public_complete_artifact_ms"]),
                })
        require(analysis["paired"] == paired, "paired raw measurements changed")
        deltas = [item["candidate_minus_baseline_ms"] for item in paired]
        baseline_values = [item["baseline_ms"] for item in paired]
        candidate_values = [item["candidate_ms"] for item in paired]
        check_summary(analysis["paired_delta"], deltas, "paired_delta")
        check_summary(analysis["baseline"], baseline_values, "candidate.baseline")
        check_summary(analysis["candidate"], candidate_values, "candidate.candidate")
        median_delta = statistics.median(deltas)
        median_baseline = statistics.median(baseline_values)
        close(analysis["median_paired_candidate_minus_baseline_ms"], median_delta,
              "median paired delta")
        close(analysis["median_paired_change_percent"], 100.0 * median_delta / median_baseline,
              "median paired change percent")
        require(analysis["latency_verdict"] == (
                    "IMPROVED" if median_delta < 0.0 else "NO_MEASURED_IMPROVEMENT")
                and analysis["regression_gate_pass"] == (median_delta <= 0.05 * median_baseline)
                and analysis["all_observations_retained"] is True,
                "candidate latency verdict or regression gate mismatch")
        candidate_result = {
            "latency_verdict": analysis["latency_verdict"],
            "median_paired_candidate_minus_baseline_ms": median_delta,
            "median_paired_change_percent": analysis["median_paired_change_percent"],
            "regression_gate_pass": analysis["regression_gate_pass"],
            "paired_count": len(paired),
            "baseline_reference_manifest": baseline_manifest,
        }
    return {"report": report, "manifest": manifest, "candidate": candidate_result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--server-binary", type=Path)
    parser.add_argument("--baseline-wan-binary", type=Path)
    parser.add_argument("--candidate-wan-binary", type=Path)
    parser.add_argument("--flux-stage", type=Path)
    parser.add_argument("--wan-model", type=Path)
    parser.add_argument("--flux-model", type=Path)
    parser.add_argument(
        "--skip-execution-binary-recheck", action="store_true",
        help=("for a copied-evidence replay on another host; raw evidence is still fully "
              "verified and the output explicitly marks this provenance recheck skipped"))
    parser.add_argument(
        "--recorded-verifier-sha256",
        help=("explicit historical verifier digest for a documented verifier-only correction; "
              "the current verifier digest is recorded separately in the new output"))
    args = parser.parse_args()
    require(args.recorded_verifier_sha256 is None
            or re.fullmatch(r"[0-9a-f]{64}", args.recorded_verifier_sha256),
            "recorded verifier SHA-256 must be exact lowercase hex")
    root = args.root.resolve()
    out = args.out.resolve()
    require(out != root and root not in out.parents,
            "independent verification output must be outside the sealed evidence root")
    require(not out.exists(), f"verification output already exists: {out}")
    result = verify_bundle(root, args)
    report = result["report"]
    verification = {
        "format": "umminfer-i2v-independent-verification-v1", "status": "PASS",
        "mode": report["mode"], "base_commit": report["base_commit"],
        "execution_commit": report["execution_commit"],
        "report_sha256": sha256_file(root / "report.json")["sha256"],
        **result["manifest"], "candidate": result["candidate"],
        "raw_messagepack_decoder": "independent_stdlib",
        "execution_binary_recheck": (
            "SKIPPED_COPIED_EVIDENCE_REPLAY" if args.skip_execution_binary_recheck
            else "PERFORMED_FOR_ALL_SUPPLIED_BINARIES"),
        "model_file_recheck": [key for key, path in (
            ("wan22", args.wan_model), ("flux2", args.flux_model)) if path is not None],
        "producer_recorded_verifier_sha256": report["provenance"]["independent_verifier"]["sha256"],
        "current_verifier_sha256": sha256_file(__file__)["sha256"],
        "verifier_correction_without_media_regeneration": (
            report["provenance"]["independent_verifier"]["sha256"]
            != sha256_file(__file__)["sha256"]),
        "manifest_policy": "complete recursive file set; only top-level SHA256SUMS self-excluded",
        "all_observations_and_outliers_retained": True,
        "perceptual_quality": "NOT_RUN", "human_review": "NOT_RUN",
        "character_style_consistency": "NOT_RUN",
    }
    out.write_bytes(canonical(verification) + b"\n")
    print(json.dumps(verification, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
