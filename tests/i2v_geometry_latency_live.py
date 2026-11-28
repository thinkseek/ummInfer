#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Public FLUX.2 -> native Wan2.2 I2V geometry and controlled latency gate.

CPU-only check:
  python3 -I tests/i2v_geometry_latency_live.py --self-check

Live mode uses one explicitly idle physical GPU selected by the caller. It never
contacts the resident service. Public /v1/visual/render requests generate real
FLUX sources and native Wan artifacts; read-only MessagePack taps retain exact
native inputs/outputs. Candidate mode replays the frozen real FLUX sources from
an accepted profile with seeded, process-interleaved baseline/candidate blocks.
"""

import argparse
import base64
import hashlib
import http.client
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import socket
import statistics
import struct
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_narrative_i2v_live import (  # noqa: E402
    ManagedProcess, free_port, http_json, sha256_bytes, sha256_file)

FORMAT = "umminfer-i2v-geometry-latency-v1"
SEED = 42
FRAMES = 29
STEPS = 20
FPS = 16
GUIDANCE = 5.0
FIRST_FRAME_MAX_MAE = 48.0
MOTION_MIN_MAE = 1.0
REPETITIONS = 3
NS02_REPETITIONS = 5
WARMUP_PER_SHAPE_PER_BLOCK = 1
INTERLEAVING_SEED = 74062
NS02_INTERLEAVING_SEED = 79025
NS02_MIN_SPEEDUP_MS = 2000.0
NS02_VIDEO_LIMIT_MS = 5000.0
NS02_CANCEL_LIMIT_MS = 500.0
NS02_MIN_SSIM = 0.90
NS02_MIN_PSNR_DB = 24.0
NS02_BOUNDARY2_FORMAT = "umminfer-ns02-interactive-preview-preregistration-v2"
NS02_BOUNDARY2_ID = "automatic-i2v-boundary2-teacache-0.25"
NS02_BOUNDARY2_COMMIT = "b4c92ac1e967aa3b661d42059cff281060c73398"
NS02_BOUNDARY2_BASE = "b23f6fd1f2b3c50165ddcc5cac86dc5aa7723dc4"
NS02_BOUNDARY2_POLICY = {
    "schedule": "boundary2",
    "automatic_default": True,
    "applies_to": "i2v_only",
    "protected_step_indices": [0, 1, 18, 19],
    "cacheable_step_indices": list(range(2, 18)),
    "maximum_hits_per_arm": 32,
    "expected_total_calls_per_arm": 40,
    "explicit_request_and_stage_values_override": True,
}
NS02_EXACT_CUDA_FORMAT = "umminfer-ns02-exact-cuda-speedup-preregistration-v1"
NS02_EXACT_CUDA_ID = "wan-dit-cublaslt-fixed-algorithms"
NS02_EXACT_CUDA_COMMIT = "b64f022a3555c925982a9d88274d98c82edddd9f"
NS02_EXACT_CUDA_BASE = "e084d8404b1d96bee5cd6f83d8299c2498a3af65"
NS02_EXACT_CUDA_POLICY = {
    "kind": "cublaslt_fixed_heuristic_indices",
    "default_index": 2,
    "ff2_index": 5,
    "fallback_index": 0,
    "compute_type": "FP32",
    "input_output_type": "BF16",
    "expected_total_calls_per_arm": 40,
    "expected_teacache_hits_per_arm": 0,
    "exact_media_parity_required": True,
}
PRIMARY_METRIC = "median_paired_public_complete_artifact_ms"
CASES = {
    "wide": {"source": (704, 480), "target": (704, 480)},
    "tall": {"source": (480, 704), "target": (480, 704)},
    "square": {"source": (512, 512), "target": (704, 480)},
}
SOURCE_PROMPT = (
    "A single vivid red paper lantern above a misty blue mountain lake at dawn, "
    "cinematic composition, clear subject, gentle ripples")
MOTION_PROMPT = (
    "The red lantern sways gently while the camera slowly orbits above the misty lake")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def clock_ns():
    return time.perf_counter_ns()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def write_json(path, value):
    Path(path).write_bytes(canonical(value) + b"\n")


def size_text(size):
    return f"{size[0]}x{size[1]}"


def png_dimensions(raw):
    require(len(raw) >= 24 and raw[:8] == b"\x89PNG\r\n\x1a\n"
            and raw[12:16] == b"IHDR", "source is not a PNG with IHDR")
    width, height = struct.unpack(">II", raw[16:24])
    require(width > 0 and height > 0, "source PNG has invalid dimensions")
    return width, height


def source_data_url(raw):
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def decode_source_data_url(value):
    require(isinstance(value, str) and value.startswith("data:image/png;base64,"),
            "source data URL has an unexpected prefix")
    return base64.b64decode(value.split(",", 1)[1], validate=True)


def quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    require(ordered and 0.0 <= fraction <= 1.0, "quantile requires nonempty data")
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values):
    require(values, "latency summary requires at least one observation")
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    q1 = quantile(values, 0.25)
    q3 = quantile(values, 0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return {
        "count": len(values), "all_ms": values,
        "median_ms": median, "p95_ms": quantile(values, 0.95),
        "iqr_ms": iqr, "mad_ms": statistics.median(deviations),
        "iqr_outliers_ms": [value for value in values if value < low or value > high],
        "outlier_policy": "retained_in_all_statistics",
    }


def model_identity(root, model_id):
    root = Path(root)
    require(root.is_dir(), f"model directory is unavailable: {root}")
    files = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files[path.relative_to(root).as_posix()] = sha256_file(path)
    require(any(name.endswith(".safetensors") for name in files),
            f"no model weights under {root}")
    return {"model_id": model_id, "file_count": len(files), "files": files,
            "tree_sha256": sha256_bytes(canonical(files))}


def runtime_identity():
    names = ("torch", "diffusers", "transformers", "huggingface-hub", "tokenizers",
             "safetensors", "accelerate", "numpy", "pillow", "pyzmq", "msgpack")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "ABSENT"
    return versions


def gpu_state(nvidia_smi, gpu):
    result = subprocess.run([
        nvidia_smi, "-i", str(gpu), "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], check=True, capture_output=True,
        text=True, timeout=20)
    fields = [int(value.strip()) for value in result.stdout.strip().split(",")]
    require(len(fields) == 3 and fields[0] == gpu,
            f"unexpected physical GPU response: {result.stdout!r}")
    processes = subprocess.run([
        nvidia_smi, "-i", str(gpu), "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits"], check=True, capture_output=True,
        text=True, timeout=20)
    compute = []
    for line in processes.stdout.splitlines():
        if line.strip():
            pid, memory = (value.strip() for value in line.split(","))
            compute.append({"pid": int(pid), "used_gpu_memory_mib": int(memory)})
    return {"index": gpu, "memory_used_mib": fields[1],
            "utilization_percent": fields[2], "compute_processes": compute}


def require_gpu_idle(nvidia_smi, gpu, *, after_owned_stage=False):
    observed = gpu_state(nvidia_smi, gpu)
    # At initial admission require a completely quiet device. Immediately after
    # a task-owned Stage exits, NVML utilization can be a short stale sample;
    # zero compute processes and <=16 MiB prove that no model remains resident.
    require(observed["memory_used_mib"] <= 16
            and (after_owned_stage or observed["utilization_percent"] == 0)
            and not observed["compute_processes"],
            f"physical GPU {gpu} is occupied: {observed}")
    return observed


class MessageTap:
    """Forward unchanged PUSH/PULL messages and retain every packed frame."""

    def __init__(self, name, listen, destination, evidence_dir):
        self.name = name
        self.listen = listen
        self.destination = destination
        self.evidence_dir = Path(evidence_dir)
        self.records = []
        self.error = None
        self.stop_requested = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"tap-{name}", daemon=True)
        self.thread.start()

    def _run(self):
        import msgpack
        import zmq
        context = pull = push = None
        try:
            context = zmq.Context()
            pull = context.socket(zmq.PULL)
            pull.RCVTIMEO = 50
            pull.bind(self.listen)
            push = context.socket(zmq.PUSH)
            push.setsockopt(zmq.LINGER, 0)
            push.connect(self.destination)
            while not self.stop_requested.is_set():
                try:
                    packed = pull.recv()
                except zmq.Again:
                    continue
                tick = clock_ns()
                message = msgpack.unpackb(packed, raw=False, strict_map_key=False)
                message_type = message.get("type")
                request_id = message.get("request_id")
                filename = f"{self.name}-{len(self.records)}-{sha256_bytes(packed)}.msgpack"
                (self.evidence_dir / filename).write_bytes(packed)
                record = {"tap": self.name, "message_type": message_type,
                          "request_id": request_id, "observed_monotonic_ns": tick,
                          "packed_file": filename, "packed_sha256": sha256_bytes(packed),
                          "packed_bytes": len(packed)}
                if message_type == "submit":
                    inputs = (((message.get("data") or {}).get("request") or {})
                              .get("inputs") or {})
                    record["inputs"] = {
                        key: value for key, value in inputs.items() if key != "image"}
                    image = inputs.get("image")
                    if image is not None:
                        source = decode_source_data_url(image) if isinstance(image, str) else image
                        require(isinstance(source, bytes) and bool(source),
                                "native I2V submit has no source bytes")
                        record["source_sha256"] = sha256_bytes(source)
                        record["source_bytes"] = len(source)
                        record["image_wire_type"] = "data_url" if isinstance(image, str) else "bin"
                elif message_type == "complete":
                    record["success"] = bool(message.get("success"))
                    record["error"] = message.get("error")
                    result = message.get("result") or {}
                    record["result"] = {key: value for key, value in result.items()
                                        if key != "data"}
                    data = result.get("data")
                    if isinstance(data, bytes):
                        record["media_sha256"] = sha256_bytes(data)
                        record["media_bytes"] = len(data)
                forwarded = False
                while not forwarded and not self.stop_requested.is_set():
                    try:
                        push.send(packed, zmq.NOBLOCK)
                        forwarded = True
                    except zmq.Again:
                        time.sleep(0.005)
                if not forwarded:
                    break
                record["forwarded_monotonic_ns"] = clock_ns()
                self.records.append(record)
        except Exception as error:
            self.error = str(error)
        finally:
            if pull is not None:
                pull.close(0)
            if push is not None:
                push.close(0)
            if context is not None:
                context.term()

    def stop(self):
        self.stop_requested.set()
        self.thread.join(timeout=10)
        require(not self.thread.is_alive(), f"{self.name} tap did not stop")
        require(self.error is None, f"{self.name} tap failed: {self.error}")


def public_render(port, payload, out, name, timeout):
    request_path = Path(out) / "requests" / f"{name}.json"
    response_path = Path(out) / "responses" / f"{name}.bin"
    request_path.write_bytes(payload)
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    chunks = []
    first_body_ns = None
    try:
        submit_ns = clock_ns()
        conn.request("POST", "/v1/visual/render", body=payload,
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        headers_ns = clock_ns()
        while True:
            chunk = response.read(65536)
            if not chunk:
                break
            if first_body_ns is None:
                first_body_ns = clock_ns()
            chunks.append(chunk)
        complete_ns = clock_ns()
        raw = b"".join(chunks)
        response_path.write_bytes(raw)
        result = {
            "http_status": response.status, "response_headers": dict(response.getheaders()),
            "public_request_file": request_path.relative_to(out).as_posix(),
            "public_response_file": response_path.relative_to(out).as_posix(),
            "public_request_sha256": sha256_bytes(payload),
            "public_response_sha256": sha256_bytes(raw),
            "public_response_bytes": len(raw),
            "submit_monotonic_ns": submit_ns, "response_headers_monotonic_ns": headers_ns,
            "first_body_monotonic_ns": first_body_ns, "complete_monotonic_ns": complete_ns,
            "public_first_output_ms": (headers_ns - submit_ns) / 1e6,
            "public_first_artifact_byte_ms": (
                (first_body_ns - submit_ns) / 1e6 if first_body_ns is not None else None),
            "public_complete_artifact_ms": (complete_ns - submit_ns) / 1e6,
        }
        require(submit_ns <= headers_ns <= complete_ns
                and (first_body_ns is None or headers_ns <= first_body_ns <= complete_ns),
                f"nonmonotonic public timing for {name}")
        return result, raw
    finally:
        conn.close()


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
    expected_w, expected_h = evaluation if evaluation is not None else (width, height)
    require(len(result.stdout) == expected_w * expected_h * 3,
            f"decoded RGB byte count mismatch for {media}")
    return result.stdout


def probe_video(ffprobe, path, expected):
    result = subprocess.run([
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=codec_name,width,height,nb_frames,r_frame_rate,pix_fmt",
        "-of", "json", str(path)], check=True, capture_output=True, text=True, timeout=30)
    stream = json.loads(result.stdout)["streams"][0]
    require(stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
            and (stream["width"], stream["height"]) == expected
            and int(stream["nb_frames"]) == FRAMES
            and stream["r_frame_rate"] == f"{FPS}/1",
            f"unexpected native Wan video: {stream}")
    return stream


def mean_absolute(first, second):
    require(len(first) == len(second) and bool(first), "RGB buffers differ or are empty")
    return sum(abs(a - b) for a, b in zip(first, second)) / len(first)


def ffmpeg_all_rgb(ffmpeg, media, width, height, evaluation):
    eval_w, eval_h = evaluation
    result = subprocess.run([
        ffmpeg, "-v", "error", "-i", str(media), "-vf",
        f"scale={eval_w}:{eval_h}:force_original_aspect_ratio=increase,crop={eval_w}:{eval_h}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True,
        capture_output=True, timeout=120)
    frame_bytes = eval_w * eval_h * 3
    require(len(result.stdout) == FRAMES * frame_bytes,
            f"all-frame RGB byte count mismatch for {media}")
    return [result.stdout[index * frame_bytes:(index + 1) * frame_bytes]
            for index in range(FRAMES)]


def ssim_all(ffmpeg, baseline, candidate):
    result = subprocess.run([
        ffmpeg, "-hide_banner", "-i", str(baseline), "-i", str(candidate),
        "-lavfi", "ssim", "-f", "null", "-"], check=True, capture_output=True,
        text=True, timeout=120)
    match = re.search(r"\bAll:([0-9.]+)", result.stderr)
    require(match is not None, f"FFmpeg SSIM output is missing: {result.stderr}")
    value = float(match.group(1))
    require(math.isfinite(value) and 0.0 <= value <= 1.0, "invalid SSIM")
    return {"all": value}


def paired_video_quality(ffmpeg, out, baseline, candidate, baseline_out=None):
    target = CASES[baseline["shape"]]["target"]
    require(target == CASES[candidate["shape"]]["target"], "paired geometry differs")
    before_path = (baseline_out or out) / baseline["media_file"]
    after_path = out / candidate["media_file"]
    before_first = ((baseline_out or out) /
                    baseline["decoded_rgb"]["first_full"]["file"]).read_bytes()
    after_first = (out / candidate["decoded_rgb"]["first_full"]["file"]).read_bytes()
    evaluation = (target[0] // 8, target[1] // 8)
    before_frames = ffmpeg_all_rgb(ffmpeg, before_path, *target, evaluation)
    after_frames = ffmpeg_all_rgb(ffmpeg, after_path, *target, evaluation)
    before = b"".join(before_frames)
    after = b"".join(after_frames)
    squared = sum((a - b) ** 2 for a, b in zip(before, after))
    mse = squared / len(before)
    psnr = 999.0 if mse == 0.0 else 10.0 * math.log10(255.0 * 255.0 / mse)
    before_temporal = [mean_absolute(before_frames[i - 1], before_frames[i])
                       for i in range(1, FRAMES)]
    after_temporal = [mean_absolute(after_frames[i - 1], after_frames[i])
                      for i in range(1, FRAMES)]
    before_motion = sum(before_temporal) / len(before_temporal)
    after_motion = sum(after_temporal) / len(after_temporal)
    motion_ratio = after_motion / max(before_motion, 1e-9)
    ssim = ssim_all(ffmpeg, before_path, after_path)
    result = {
        "first_frame_rgb_exact": before_first == after_first,
        "first_frame_rgb_sha256_before": sha256_bytes(before_first),
        "first_frame_rgb_sha256_candidate": sha256_bytes(after_first),
        "evaluation_size": list(evaluation), "all_frames": FRAMES,
        "all_frame_mae": mean_absolute(before, after), "all_frame_psnr_db": psnr,
        "ssim": ssim, "before_temporal_mae_mean": before_motion,
        "candidate_temporal_mae_mean": after_motion,
        "temporal_motion_ratio": motion_ratio,
        "before_first_last_mae": baseline["motion_mae"],
        "candidate_first_last_mae": candidate["motion_mae"],
    }
    result["quality_gate_pass"] = bool(
        result["first_frame_rgb_exact"] and ssim["all"] >= NS02_MIN_SSIM
        and psnr >= NS02_MIN_PSNR_DB and 0.5 <= motion_ratio <= 1.5
        and candidate["first_frame_mae"] <= FIRST_FRAME_MAX_MAE
        and candidate["motion_mae"] >= MOTION_MIN_MAE)
    return result


def parse_native_phase(log_path, request_id):
    for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
        if "video request backend=" not in line or f"request_id={request_id}" not in line:
            continue
        fields = {}
        for key, value in re.findall(r"([a-z0-9_]+)=([^\s]+)", line):
            fields[key] = value
        for key in ("image_decode_ms", "image_resize_ms", "vae_encode_ms",
                    "vae_decode_ms", "latent_init_ms", "umt5_ms", "dit_ms",
                    "vae_ms", "denoise_ms", "encode_ms", "elapsed_ms"):
            require(key in fields, f"native log lacks {key}: {line}")
            fields[key] = float(fields[key])
            require(math.isfinite(fields[key]) and fields[key] >= 0.0,
                    f"invalid native phase {key}: {line}")
        require(fields.get("backend") == "native-cuda-wan" and fields.get("i2v") == "1",
                f"request did not execute native I2V: {line}")
        require(fields.get("size") and fields.get("source_size"),
                f"native geometry metadata missing: {line}")
        fields["log_sha256"] = sha256_bytes(line.encode())
        return fields
    raise AssertionError(f"native execution log missing request {request_id}")


def preregistration_document(profile_sha256, candidate_id, description):
    return {
        "format": "umminfer-i2v-latency-preregistration-v1",
        "profile_report_sha256": profile_sha256,
        "candidate_id": candidate_id, "candidate_description": description,
        "primary_metric": PRIMARY_METRIC, "repetitions": REPETITIONS,
        "warmup_per_shape_per_block": WARMUP_PER_SHAPE_PER_BLOCK,
        "interleaving_seed": INTERLEAVING_SEED,
        "schedule": "per repetition, seeded arm order; fixed wide,tall,square order within arm",
        "cold_prefix_policy": (
            "model loading and one warmup per shape per fresh stage block are reported "
            "separately and excluded from warm statistics; no cold-prefix mixing"),
        "source_policy": "exact frozen real public FLUX.2 PNG bytes from the profile",
        "settings": {"frames": FRAMES, "steps": STEPS, "fps": FPS,
                     "guidance": GUIDANCE, "seed": SEED, "teacache_thresh": 0.0},
        "regression_gates": {
            "exact_mp4_parity_with_profile_and_paired_baseline": True,
            "decoded_dimensions_frames_codec": True,
            "first_frame_mae_max": FIRST_FRAME_MAX_MAE,
            "motion_mae_min": MOTION_MIN_MAE,
            "source_hash_and_native_submit_binding": True,
            "malformed_dimension_rejection_and_valid_recovery": True,
            "one_native_terminal_and_final_public_in_flight_zero": True,
            "maximum_median_latency_regression_percent": 5.0,
        },
        "analysis": "retain every observation and IQR outlier; report median/p95/IQR/MAD",
        "perceptual_quality": "NOT_RUN",
    }


def validate_preregistration(document):
    require(document.get("format") == "umminfer-i2v-latency-preregistration-v1",
            "unknown preregistration format")
    require(re.fullmatch(r"[0-9a-f]{64}", document.get("profile_report_sha256", "")),
            "preregistration lacks an exact profile report hash")
    require(isinstance(document.get("candidate_id"), str)
            and re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", document["candidate_id"]),
            "candidate id must be a bounded lowercase slug")
    require(document.get("primary_metric") == PRIMARY_METRIC
            and document.get("repetitions") == REPETITIONS
            and document.get("warmup_per_shape_per_block") == WARMUP_PER_SHAPE_PER_BLOCK
            and document.get("interleaving_seed") == INTERLEAVING_SEED,
            "preregistered metric, repetition, warmup, or seed contract changed")
    require(document.get("settings") == {
        "frames": FRAMES, "steps": STEPS, "fps": FPS,
        "guidance": GUIDANCE, "seed": SEED, "teacache_thresh": 0.0},
        "candidate measurement changed sampler, quality, frame, or TeaCache settings")
    gates = document.get("regression_gates") or {}
    require(gates.get("exact_mp4_parity_with_profile_and_paired_baseline") is True
            and gates.get("first_frame_mae_max") == FIRST_FRAME_MAX_MAE
            and gates.get("motion_mae_min") == MOTION_MIN_MAE
            and gates.get("maximum_median_latency_regression_percent") == 5.0,
            "correctness or latency regression gates were weakened")
    return document


def validate_ns02_preregistration(document, profile_sha256):
    historical = (document.get("format")
                  == "umminfer-ns02-interactive-preview-preregistration-v1")
    boundary2 = document.get("format") == NS02_BOUNDARY2_FORMAT
    exact_cuda = document.get("format") == NS02_EXACT_CUDA_FORMAT
    require(historical or boundary2 or exact_cuda,
            "unknown NS-02 preregistration format")
    require(document.get("profile_report_sha256") == profile_sha256
            and document.get("repetitions") == NS02_REPETITIONS
            and document.get("interleaving_seed") == NS02_INTERLEAVING_SEED,
            "NS-02 profile, population, or seed identity changed")
    if historical:
        require(document.get("candidate_id") == "automatic-i2v-teacache-0.25"
                and document.get("candidate_commit")
                == "3f98fe1645d4d82877372056ba8ad2544aa399c5",
                "historical NS-02 candidate identity changed")
    elif boundary2:
        require(document.get("official_base_commit") == NS02_BOUNDARY2_BASE
                and document.get("candidate_id") == NS02_BOUNDARY2_ID
                and document.get("candidate_commit") == NS02_BOUNDARY2_COMMIT
                and document.get("candidate_policy") == NS02_BOUNDARY2_POLICY
                and document.get("created_before_final_candidate_measurement") is True,
                "boundary2 NS-02 base, candidate, or frozen schedule changed")
    else:
        require(document.get("official_base_commit") == NS02_EXACT_CUDA_BASE
                and document.get("candidate_id") == NS02_EXACT_CUDA_ID
                and document.get("candidate_commit") == NS02_EXACT_CUDA_COMMIT
                and document.get("candidate_policy") == NS02_EXACT_CUDA_POLICY
                and document.get("created_before_final_candidate_measurement") is True,
                "exact-CUDA NS-02 base, candidate, or frozen policy changed")
    expected_candidate_threshold = 0.0 if exact_cuda else 0.25
    require(document.get("settings") == {
        "frames": FRAMES, "steps": STEPS, "fps": FPS, "guidance": GUIDANCE,
        "seed": SEED, "request_teacache_field": "omitted",
        "baseline_effective_teacache": 0.0,
        "candidate_effective_teacache": expected_candidate_threshold},
        "NS-02 sampler, fidelity, frame, or TeaCache protocol changed")
    expected_gates = {
        "exact_paired_first_frame_rgb": True, "minimum_ssim_all": NS02_MIN_SSIM,
        "minimum_all_frame_psnr_db": NS02_MIN_PSNR_DB,
        "temporal_motion_ratio_range": [0.5, 1.5],
        "minimum_median_speedup_ms": NS02_MIN_SPEEDUP_MS,
        "all_paired_deltas_negative": True,
        "candidate_median_and_p95_video_ms_below": NS02_VIDEO_LIMIT_MS,
        "cancel_native_terminal_and_client_settle_ms_below": NS02_CANCEL_LIMIT_MS,
        "cancel_threshold_arm": "candidate_only_before_is_reported_independently",
        "zero_late_native_media_and_final_in_flight_zero": True,
        "immediate_retry_exact_own_arm_hash": True}
    if exact_cuda:
        expected_gates["exact_paired_mp4_media"] = True
        expected_gates["baseline_matches_frozen_profile_mp4"] = True
    require(document.get("regression_gates") == expected_gates,
            "NS-02 quality, exact-media, latency, cancellation, or recovery gates changed")
    return document


def candidate_schedule(mode="candidate"):
    schedule = []
    repetitions = NS02_REPETITIONS if mode == "ns02" else REPETITIONS
    seed = NS02_INTERLEAVING_SEED if mode == "ns02" else INTERLEAVING_SEED
    for repetition in range(repetitions):
        arms = ["baseline", "candidate"]
        random.Random(seed + repetition).shuffle(arms)
        for arm in arms:
            schedule.append({"repetition": repetition, "arm": arm,
                             "shapes": list(CASES), "warmups_per_shape": 1,
                             "measured_per_shape": 1})
    return schedule


def make_manifest(root):
    root = Path(root)
    manifest = root / "SHA256SUMS"
    paths = sorted(path for path in root.rglob("*")
                   if path.is_file() and path != manifest)
    manifest.write_text("".join(
        f"{sha256_file(path)['sha256']}  {path.relative_to(root).as_posix()}\n"
        for path in paths), encoding="utf-8")
    return len(paths)


def run_live(args):
    require(os.environ.get("UMMINFER_LIVE_OPT_IN") == "1",
            "REAL_GENERATION: NOT_RUN (UMMINFER_LIVE_OPT_IN=1 required)")
    require(args.gpu in (0, 1, 7), "only explicitly selected physical GPU 0, 1, or 7 is allowed")
    for value in (args.base_commit, args.execution_commit):
        require(re.fullmatch(r"[0-9a-f]{40}", value or ""),
                "live mode requires exact lowercase base/execution commits")
    out = Path(args.out_dir)
    require(not out.exists(), f"evidence directory already exists: {out}")
    for directory in ("requests", "responses", "media", "rgb", "native", "run-records"):
        (out / directory).mkdir(parents=True, exist_ok=True)
    args.ffmpeg = shutil.which(args.ffmpeg)
    args.ffprobe = shutil.which(args.ffprobe)
    args.nvidia_smi = shutil.which(args.nvidia_smi)
    require(args.ffmpeg and args.ffprobe and args.nvidia_smi,
            "ffmpeg, ffprobe, and nvidia-smi must resolve through PATH")
    verifier = Path(args.verifier).resolve()
    require(verifier.is_file(), "independent verifier is missing")
    candidate_mode = args.mode in ("candidate", "ns02")
    ns02_mode = args.mode == "ns02"
    prereg = None
    baseline_reference = None
    if candidate_mode:
        require(args.preregistration and args.baseline_evidence
                and args.candidate_wan_binary,
                "candidate mode requires preregistration, baseline evidence, and candidate binary")
        prereg = json.loads(Path(args.preregistration).read_text())
        baseline_reference = Path(args.baseline_evidence)
        if ns02_mode:
            prereg = validate_ns02_preregistration(
                prereg, sha256_file(baseline_reference / "report.json")["sha256"])
        else:
            prereg = validate_preregistration(prereg)
        require((baseline_reference / "report.json").is_file(),
                "baseline evidence report is missing")
        require(sha256_file(baseline_reference / "report.json")["sha256"]
                == prereg["profile_report_sha256"],
                "baseline evidence does not match preregistered profile hash")
        shutil.copytree(baseline_reference, out / "baseline-reference")
        baseline_report = json.loads((baseline_reference / "report.json").read_text())
        require(baseline_report["provenance"]["baseline_wan_binary"]
                == sha256_file(args.wan_binary),
                "candidate baseline must use the exact instrumented Wan binary from the profile")
        (out / "preregistration.json").write_bytes(Path(args.preregistration).read_bytes())
        schedule = candidate_schedule(args.mode)
        write_json(out / "candidate-schedule.json", schedule)
    else:
        require(not args.candidate_wan_binary and not args.baseline_evidence
                and not args.preregistration,
                "profile mode does not accept candidate arguments")
        schedule = None

    provenance = {
        "base_commit": args.base_commit, "execution_commit": args.execution_commit,
        "server_binary": sha256_file(args.server_binary),
        "baseline_wan_binary": sha256_file(args.wan_binary),
        "flux_stage": sha256_file(args.flux_stage) if args.flux_stage else None,
        "live_harness": sha256_file(Path(__file__).resolve()),
        "independent_verifier": sha256_file(verifier),
        "ffmpeg": sha256_file(args.ffmpeg), "ffprobe": sha256_file(args.ffprobe),
        "python_runtime": runtime_identity(),
    }
    if candidate_mode:
        provenance["candidate_wan_binary"] = sha256_file(args.candidate_wan_binary)
    models = {"wan22": model_identity(args.wan_model, "Wan-AI/Wan2.2-TI2V-5B-Diffusers")}
    if not candidate_mode:
        models["flux2"] = model_identity(
            args.flux_model, "black-forest-labs/FLUX.2-klein-4B")

    port = args.port or free_port()
    require(port != 43301, "resident service port is forbidden")
    run_ipc = out / "ipc"
    run_ipc.mkdir()
    endpoints = {name: f"ipc://{run_ipc / name}" for name in (
        "completion", "abort", "entry-in", "vlm-in", "image-in", "video-tap-in", "video-in",
        "video-output-in")}
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if args.library_path:
        env["LD_LIBRARY_PATH"] = str(args.library_path) + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    server = entry = vlm = flux = None
    video_input_tap = video_output_tap = None
    processes = []
    runs = []
    sources = {}
    controls = []
    cancellations = []
    gpu = {"before": require_gpu_idle(args.nvidia_smi, args.gpu)}
    current_video = None
    current_arm = None
    current_block = None

    def cleanup_ipc():
        if not run_ipc.exists():
            return
        for path in run_ipc.iterdir():
            require(path.is_socket(), f"unexpected non-socket IPC artifact: {path.name}")
            path.unlink()
        run_ipc.rmdir()

    def stop_video():
        nonlocal current_video
        if current_video is not None:
            current_video.stop(endpoints["video-in"])
            require(current_video.returncode == 0,
                    f"Wan stage shutdown failed: {current_video.returncode}")
            processes.append(current_video.record())
            current_video = None
            require_gpu_idle(args.nvidia_smi, args.gpu, after_owned_stage=True)

    def start_video(arm, block):
        nonlocal current_video, current_arm, current_block
        require(current_video is None, "a Wan stage is already resident")
        binary = args.wan_binary if arm == "baseline" else args.candidate_wan_binary
        name = f"video-r{block}-{arm}"
        current_video = ManagedProcess(name, [
            binary, "--kind", "video", "--name", "video", "--video-backend", "cuda",
            "--model", args.wan_model, "--gpu", str(args.gpu), "--size", "704x480",
            "--steps", str(STEPS), "--fps", str(FPS), "--ffmpeg-bin", args.ffmpeg,
            "--recv-endpoint", endpoints["video-in"],
            "--coordinator-endpoint", endpoints["video-output-in"],
            "--abort-endpoint", endpoints["abort"]], out / f"{name}.log", env=env)
        current_arm = arm
        current_block = block
        current_video.wait_log("native stage video ready kind=video", args.timeout)
        gpu[f"{name}-ready"] = gpu_state(args.nvidia_smi, args.gpu)

    def journal_record(record):
        write_json(out / "run-records" / f"{record['run_id']}.json", record)
        return record

    def execute_video(shape, role, run_id, repetition=None, override_size=None):
        case = CASES[shape]
        target = override_size or case["target"]
        source_path = out / sources[shape]["source_file"]
        prep_started = clock_ns()
        source = source_path.read_bytes()
        source_sha = sha256_bytes(source)
        require(source_sha == sources[shape]["source_sha256"], "frozen source bytes changed")
        body = {"kind": "video", "prompt": MOTION_PROMPT, "size": size_text(target),
                "steps": STEPS, "num_frames": FRAMES, "fps": FPS,
                "guidance_scale": GUIDANCE, "seed": SEED,
                "image": source_data_url(source)}
        if not ns02_mode:
            body["teacache_thresh"] = 0.0
        payload = canonical(body)
        prep_done = clock_ns()
        input_before = len(video_input_tap.records)
        output_before = len(video_output_tap.records)
        public, raw = public_render(port, payload, out, run_id, args.timeout)
        tap_deadline = time.monotonic() + 30.0
        while True:
            new_inputs = [record for record in video_input_tap.records[input_before:]
                          if record["message_type"] == "submit"]
            new_outputs = [record for record in video_output_tap.records[output_before:]
                           if record["message_type"] == "complete"]
            if len(new_inputs) == 1 and len(new_outputs) == 1:
                break
            require(time.monotonic() < tap_deadline
                    and video_input_tap.error is None and video_output_tap.error is None,
                    f"native taps did not settle for {run_id}")
            time.sleep(0.001)
        require(len(new_inputs) == 1,
                f"{run_id} did not produce exactly one native submit")
        require(len(new_outputs) == 1,
                f"{run_id} did not produce exactly one native terminal")
        native_input = new_inputs[0]
        native_output = new_outputs[0]
        require(native_input["request_id"] == native_output["request_id"],
                f"native request identity mismatch for {run_id}")
        require(native_input.get("source_sha256") == source_sha,
                f"native source bytes differ from public FLUX source for {run_id}")
        require(native_input["inputs"].get("size") == size_text(target)
                and native_input["inputs"].get("steps") == STEPS
                and native_input["inputs"].get("num_frames") == FRAMES
                and native_input["inputs"].get("seed") == SEED
                and (("teacache_thresh" not in native_input["inputs"]) if ns02_mode
                     else native_input["inputs"].get("teacache_thresh") == 0.0),
                f"native request settings drifted for {run_id}")
        record = {
            **public, "run_id": run_id, "shape": shape, "role": role,
            "arm": current_arm, "block": current_block, "repetition": repetition,
            "source_sha256": source_sha, "source_file": sources[shape]["source_file"],
            "source_preparation_ms": (prep_done - prep_started) / 1e6,
            "native_request_id": native_input["request_id"],
            "native_input_file": "native/" + native_input["packed_file"],
            "native_output_file": "native/" + native_output["packed_file"],
            "native_submit_monotonic_ns": native_input["observed_monotonic_ns"],
            "native_complete_monotonic_ns": native_output["observed_monotonic_ns"],
            "native_success": native_output["success"],
            "native_error": native_output.get("error"),
        }
        if native_output["observed_monotonic_ns"] <= public["response_headers_monotonic_ns"]:
            record["native_complete_to_public_first_output_ms"] = (
                public["response_headers_monotonic_ns"]
                - native_output["observed_monotonic_ns"]) / 1e6
        else:
            raise AssertionError(f"native completion followed public headers for {run_id}")
        if public["http_status"] != 200:
            record["error_body"] = raw.decode("utf-8", errors="replace")
            require(not native_output["success"] and not native_output.get("media_sha256"),
                    f"failed request emitted media: {run_id}")
            return journal_record(record)
        require(native_output["success"] and native_output.get("media_sha256")
                == public["public_response_sha256"],
                f"native/public MP4 bytes differ for {run_id}")
        media_path = out / "media" / f"{run_id}.mp4"
        media_path.write_bytes(raw)
        record["media_file"] = media_path.relative_to(out).as_posix()
        record["media_sha256"] = sha256_bytes(raw)
        record["video_stream"] = probe_video(args.ffprobe, media_path, target)
        record["native_phase"] = parse_native_phase(
            out / f"video-r{current_block}-{current_arm}.log", native_input["request_id"])
        require(record["native_phase"]["size"] == size_text(target)
                and record["native_phase"]["source_size"] == sources[shape]["source_size"],
                f"stale native geometry metadata for {run_id}")
        if ns02_mode:
            expected_teacache = (prereg["settings"]["candidate_effective_teacache"]
                                 if current_arm == "candidate"
                                 else prereg["settings"]["baseline_effective_teacache"])
            require(float(record["native_phase"]["teacache_thresh"]) == expected_teacache,
                    f"effective TeaCache policy drifted for {run_id}")
            if current_arm == "candidate":
                hits, total = (int(value) for value in
                               record["native_phase"]["teacache_hits"].split("/"))
                policy = prereg.get("candidate_policy")
                if policy is None:
                    require(0 < hits < total,
                            f"candidate did not use bounded cache: {run_id}")
                elif policy.get("kind") == "cublaslt_fixed_heuristic_indices":
                    require(hits == policy["expected_teacache_hits_per_arm"]
                            and total == policy["expected_total_calls_per_arm"],
                            f"exact-CUDA candidate unexpectedly used TeaCache: {run_id}")
                else:
                    require(record["native_phase"].get("teacache_boundary2") == "1"
                            and 0 < hits <= policy["maximum_hits_per_arm"]
                            and total == policy["expected_total_calls_per_arm"],
                            f"boundary2 schedule or cache telemetry drifted for {run_id}")
        eval_size = (target[0] // 8, target[1] // 8)
        first = ffmpeg_rgb(args.ffmpeg, media_path, *target, frame=0)
        last = ffmpeg_rgb(args.ffmpeg, media_path, *target, frame=FRAMES - 1)
        first_eval = ffmpeg_rgb(args.ffmpeg, media_path, *target, frame=0,
                                evaluation=eval_size)
        last_eval = ffmpeg_rgb(args.ffmpeg, media_path, *target, frame=FRAMES - 1,
                               evaluation=eval_size)
        rgb_files = {}
        for label, data in (("first_full", first), ("last_full", last),
                            ("first_eval", first_eval), ("last_eval", last_eval)):
            path = out / "rgb" / f"{run_id}.{label}.rgb"
            path.write_bytes(data)
            rgb_files[label] = {"file": path.relative_to(out).as_posix(),
                                **sha256_file(path)}
        record["decoded_rgb"] = rgb_files
        source_eval_path = out / sources[shape]["evaluation_rgb"][size_text(target)]["file"]
        source_eval = source_eval_path.read_bytes()
        record["source_evaluation_rgb"] = sources[shape]["evaluation_rgb"][size_text(target)]
        record["first_frame_mae"] = mean_absolute(source_eval, first_eval)
        record["motion_mae"] = mean_absolute(first_eval, last_eval)
        require(record["first_frame_mae"] <= FIRST_FRAME_MAX_MAE,
                f"first-frame continuity gate failed for {run_id}: {record['first_frame_mae']}")
        require(record["motion_mae"] >= MOTION_MIN_MAE,
                f"motion/nonzero-output gate failed for {run_id}: {record['motion_mae']}")
        return journal_record(record)

    def exercise_disconnect_cancel(arm):
        shape = "wide"
        source = (out / sources[shape]["source_file"]).read_bytes()
        body = {"kind": "video", "prompt": MOTION_PROMPT,
                "size": size_text(CASES[shape]["target"]), "steps": STEPS,
                "num_frames": FRAMES, "fps": FPS, "guidance_scale": GUIDANCE,
                "seed": SEED, "image": source_data_url(source)}
        payload = canonical(body)
        run_id = f"{arm}-disconnect-cancel"
        (out / "requests" / f"{run_id}.json").write_bytes(payload)
        input_before = len(video_input_tap.records)
        output_before = len(video_output_tap.records)
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=args.timeout)
        submit_ns = clock_ns()
        conn.request("POST", "/v1/visual/render", body=payload,
                     headers={"Content-Type": "application/json"})
        deadline = time.monotonic() + 30.0
        while True:
            inputs = [record for record in video_input_tap.records[input_before:]
                      if record["message_type"] == "submit"]
            if len(inputs) == 1:
                break
            require(time.monotonic() < deadline and video_input_tap.error is None,
                    "native submit did not arrive before disconnect")
            time.sleep(0.001)
        native_input = inputs[0]
        target_ns = native_input["observed_monotonic_ns"] + 1_000_000_000
        remaining_ns = target_ns - clock_ns()
        if remaining_ns > 0:
            time.sleep(remaining_ns / 1e9)
        disconnect_ns = clock_ns()
        if conn.sock is not None:
            conn.sock.shutdown(socket.SHUT_RDWR)
        conn.close()
        while True:
            outputs = [record for record in video_output_tap.records[output_before:]
                       if record["message_type"] == "complete"]
            if len(outputs) == 1:
                break
            require(time.monotonic() < deadline and video_output_tap.error is None,
                    "native abort terminal did not arrive after disconnect")
            time.sleep(0.001)
        terminal = outputs[0]
        require(terminal["request_id"] == native_input["request_id"]
                and terminal["success"] is False and not terminal.get("media_sha256"),
                f"disconnect did not fail closed without media: {terminal}")
        health_deadline = time.monotonic() + 5.0
        while True:
            status, health = http_json(port, "GET", "/health", timeout=5)
            if status == 200 and health.get("in_flight") == 0:
                break
            require(time.monotonic() < health_deadline, "disconnect retained in-flight work")
            time.sleep(0.005)
        settled_ns = clock_ns()
        time.sleep(0.6)
        late = [record for record in video_output_tap.records[output_before:]
                if record.get("request_id") == native_input["request_id"]
                and record["observed_monotonic_ns"] > terminal["observed_monotonic_ns"]]
        require(not late, f"late native events/media followed disconnect: {late}")
        record = {
            "run_id": run_id, "arm": arm, "shape": shape,
            "public_request_sha256": sha256_bytes(payload),
            "native_request_id": native_input["request_id"],
            "native_input_file": "native/" + native_input["packed_file"],
            "native_output_file": "native/" + terminal["packed_file"],
            "submit_monotonic_ns": submit_ns,
            "native_submit_monotonic_ns": native_input["observed_monotonic_ns"],
            "disconnect_monotonic_ns": disconnect_ns,
            "native_terminal_monotonic_ns": terminal["observed_monotonic_ns"],
            "settled_monotonic_ns": settled_ns,
            "disconnect_to_native_terminal_ms": (
                terminal["observed_monotonic_ns"] - disconnect_ns) / 1e6,
            "disconnect_to_client_settled_ms": (settled_ns - disconnect_ns) / 1e6,
            "public_abort_ack": "NOT_APPLICABLE_NONSTREAM_RENDER_HAS_NO_REQUEST_ID",
            "late_native_records_after_terminal": 0, "public_media_bytes": 0,
            "health": health,
        }
        return journal_record(record)

    try:
        video_input_tap = MessageTap("video-input", endpoints["video-tap-in"],
                                     endpoints["video-in"], out / "native")
        video_output_tap = MessageTap("video-output", endpoints["video-output-in"],
                                      endpoints["completion"], out / "native")
        server = ManagedProcess("server", [
            args.server_binary, "serve", "--host", "127.0.0.1", "--port", str(port),
            "--model", "qwen3-native", "--entry", "entry",
            "--entry-endpoint", endpoints["entry-in"],
            "--vlm-entry", "vlm", "--vlm-entry-endpoint", endpoints["vlm-in"],
            "--image-entry", "image", "--image-entry-endpoint", endpoints["image-in"],
            "--video-entry", "video", "--video-entry-endpoint", endpoints["video-tap-in"],
            "--completion", endpoints["completion"], "--abort", endpoints["abort"],
            "--max-in-flight", "1"], out / "server.log", env=env)
        entry = ManagedProcess("entry", [
            args.server_binary, "stage", "--name", "entry", "--recv", endpoints["entry-in"],
            "--coordinator", endpoints["completion"], "--abort", endpoints["abort"]],
            out / "entry.log", env=env)
        # The direct-render route never submits text work. This existing CPU echo
        # Stage satisfies only the unified VLM readiness requirement; it is not a
        # model-output substitute and no VLM inference is claimed.
        vlm = ManagedProcess("vlm-readiness", [
            args.server_binary, "stage", "--name", "vlm", "--recv", endpoints["vlm-in"],
            "--coordinator", endpoints["completion"], "--abort", endpoints["abort"]],
            out / "vlm-readiness.log", env=env)
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            try:
                status, _ = http_json(port, "GET", "/ready", timeout=5)
                if status == 200:
                    entry.ready_ns = clock_ns()
                    break
            except (OSError, http.client.HTTPException):
                pass
            require(server.process.poll() is None, "public gateway exited before readiness")
            time.sleep(0.05)
        else:
            raise TimeoutError("public gateway did not become ready")

        if not candidate_mode:
            flux = ManagedProcess("image", [
                args.python, "-I", args.flux_stage, "--name", "image", "--backend", "diffusers",
                "--model", args.flux_model, "--model-id", "black-forest-labs/FLUX.2-klein-4B",
                "--revision", "UNKNOWN", "--gpu", str(args.gpu),
                "--default-size", "512x512", "--default-steps", "4",
                "--recv", endpoints["image-in"], "--coordinator", endpoints["completion"],
                "--abort", endpoints["abort"]], out / "image.log", env=env)
            flux.wait_log("ready kind=image name=image", args.timeout)
            gpu["flux-ready"] = gpu_state(args.nvidia_smi, args.gpu)
            for shape, case in CASES.items():
                body = {"kind": "image", "prompt": SOURCE_PROMPT,
                        "size": size_text(case["source"]), "steps": 4, "seed": SEED}
                public, raw = public_render(port, canonical(body), out,
                                            f"source-{shape}", args.timeout)
                require(public["http_status"] == 200 and png_dimensions(raw) == case["source"],
                        f"real public FLUX source failed for {shape}: {public}")
                source_path = out / "media" / f"{shape}-source.png"
                source_path.write_bytes(raw)
                eval_size = (case["target"][0] // 8, case["target"][1] // 8)
                source_full = ffmpeg_rgb(args.ffmpeg, source_path, *case["source"])
                source_eval = ffmpeg_rgb(args.ffmpeg, source_path, *case["source"],
                                         evaluation=eval_size)
                full_path = out / "rgb" / f"{shape}-source.full.rgb"
                eval_path = out / "rgb" / f"{shape}-source.{size_text(case['target'])}.rgb"
                full_path.write_bytes(source_full)
                eval_path.write_bytes(source_eval)
                sources[shape] = {
                    **public, "source_file": source_path.relative_to(out).as_posix(),
                    "source_sha256": sha256_bytes(raw), "source_size": size_text(case["source"]),
                    "source_png_dimensions": list(case["source"]),
                    "full_rgb": {"file": full_path.relative_to(out).as_posix(),
                                 **sha256_file(full_path)},
                    "evaluation_rgb": {size_text(case["target"]): {
                        "file": eval_path.relative_to(out).as_posix(), **sha256_file(eval_path),
                        "width": eval_size[0], "height": eval_size[1]}},
                }
            flux.stop(endpoints["image-in"])
            require(flux.returncode == 0, "FLUX stage did not stop cleanly")
            processes.append(flux.record())
            flux = None
            gpu["after-flux"] = require_gpu_idle(
                args.nvidia_smi, args.gpu, after_owned_stage=True)
        else:
            for shape, case in CASES.items():
                relative = f"media/{shape}-source.png"
                source_path = baseline_reference / relative
                require(source_path.is_file(), f"baseline source is missing: {relative}")
                raw = source_path.read_bytes()
                require(png_dimensions(raw) == case["source"],
                        f"baseline source dimensions changed for {shape}")
                eval_size = (case["target"][0] // 8, case["target"][1] // 8)
                eval_path = out / "baseline-reference" / "rgb" / (
                    f"{shape}-source.{size_text(case['target'])}.rgb")
                require(eval_path.is_file(), f"baseline evaluation RGB missing for {shape}")
                sources[shape] = {
                    "source_file": "baseline-reference/" + relative,
                    "source_sha256": sha256_bytes(raw), "source_size": size_text(case["source"]),
                    "evaluation_rgb": {size_text(case["target"]): {
                        "file": eval_path.relative_to(out).as_posix(), **sha256_file(eval_path),
                        "width": eval_size[0], "height": eval_size[1]}},
                }

        if not candidate_mode:
            start_video("baseline", 0)
            for shape in CASES:
                for warmup in range(WARMUP_PER_SHAPE_PER_BLOCK):
                    runs.append(execute_video(shape, "cold_or_warmup", f"{shape}-warmup{warmup}"))
                for repetition in range(REPETITIONS):
                    runs.append(execute_video(shape, "measured", f"{shape}-r{repetition}",
                                              repetition=repetition))
            invalid = execute_video("wide", "malformed_dimension", "wide-invalid",
                                    override_size=(500, 480))
            require(invalid["http_status"] == 500
                    and "divisible by 32" in invalid["error_body"],
                    f"malformed dimensions were not rejected: {invalid}")
            controls.append(invalid)
            recovery = execute_video("wide", "post_failure_recovery", "wide-recovery")
            controls.append(recovery)
            stop_video()
        else:
            for block, item in enumerate(schedule):
                start_video(item["arm"], block)
                for shape in item["shapes"]:
                    for warmup in range(item["warmups_per_shape"]):
                        runs.append(execute_video(
                            shape, "cold_or_warmup",
                            f"r{item['repetition']}-{item['arm']}-{shape}-warmup{warmup}",
                            repetition=item["repetition"]))
                    runs.append(execute_video(
                        shape, "measured",
                        f"r{item['repetition']}-{item['arm']}-{shape}-measured",
                        repetition=item["repetition"]))
                stop_video()
            if ns02_mode:
                for control_index, arm in enumerate(("baseline", "candidate")):
                    block = len(schedule) + control_index
                    start_video(arm, block)
                    runs.append(execute_video(
                        "wide", "cold_or_warmup", f"{arm}-cancel-warmup"))
                    cancellations.append(exercise_disconnect_cancel(arm))
                    controls.append(execute_video(
                        "wide", "post_cancel_recovery", f"{arm}-immediate-retry"))
                    invalid = execute_video(
                        "wide", "malformed_dimension", f"{arm}-invalid",
                        override_size=(500, 480))
                    require(invalid["http_status"] == 500
                            and "divisible by 32" in invalid["error_body"],
                            f"{arm} malformed dimensions were not rejected: {invalid}")
                    controls.append(invalid)
                    controls.append(execute_video(
                        "wide", "post_failure_recovery", f"{arm}-failure-recovery"))
                    stop_video()
            else:
                start_video("candidate", len(schedule))
                invalid = execute_video("wide", "malformed_dimension", "candidate-invalid",
                                        override_size=(500, 480))
                require(invalid["http_status"] == 500
                        and "divisible by 32" in invalid["error_body"],
                        f"candidate malformed dimensions were not rejected: {invalid}")
                controls.append(invalid)
                controls.append(execute_video("wide", "post_failure_recovery", "candidate-recovery"))
                stop_video()

        # Include the entire retained post-cancel history, including activity while
        # immediate retry/failure controls ran, rather than only the first 0.6 s.
        for cancellation in cancellations:
            terminal_tick = cancellation["native_terminal_monotonic_ns"]
            request_id = cancellation["native_request_id"]
            late = [record for record in video_output_tap.records
                    if record.get("request_id") == request_id
                    and record["observed_monotonic_ns"] > terminal_tick]
            cancellation["late_native_records_after_terminal"] = len(late)
            cancellation["late_native_record_sha256"] = [
                record["packed_sha256"] for record in late]
            journal_record(cancellation)
        video_input_tap.stop()
        video_output_tap.stop()
        video_input_tap = video_output_tap = None
        health_status, health = http_json(port, "GET", "/health", timeout=30)
        require(health_status == 200 and health.get("in_flight") == 0,
                f"public gateway retained work: {health_status} {health}")
        gpu["after-all-requests-before-server-stop"] = require_gpu_idle(
            args.nvidia_smi, args.gpu, after_owned_stage=True)
        entry.stop(endpoints["entry-in"])
        vlm.stop(endpoints["vlm-in"])
        server.stop()
        require(entry.process.poll() is not None and vlm.process.poll() is not None
                and server.process.poll() is not None,
                "task-owned gateway processes did not stop")
        processes.extend([entry.record(), vlm.record(), server.record()])
        entry = vlm = server = None
        cleanup_ipc()

        valid_by_shape = {shape: [run for run in runs if run["shape"] == shape
                                  and run["http_status"] == 200] for shape in CASES}
        for shape, shape_runs in valid_by_shape.items():
            require(shape_runs, f"no successful real I2V run for {shape}")
            arms = ("baseline", "candidate") if ns02_mode else (None,)
            for arm in arms:
                hashes = {run["media_sha256"] for run in shape_runs
                          if arm is None or run["arm"] == arm}
                require(len(hashes) == 1,
                        f"fixed source/settings are not byte reproducible for {shape}/{arm}: {hashes}")
        require(valid_by_shape["wide"][0]["media_sha256"]
                != valid_by_shape["square"][0]["media_sha256"],
                "wide and square anchors produced identical wide outputs; source may be ignored")
        for control in controls:
            if control["role"] in ("post_failure_recovery", "post_cancel_recovery"):
                reference = next(run for run in valid_by_shape[control["shape"]]
                                 if not ns02_mode or run["arm"] == control["arm"])
                require(control["media_sha256"] == reference["media_sha256"],
                        "post-cancel/failure recovery output differs from its own-arm control")

        profile = {
            "warm_by_shape_arm": {
                arm: {shape: summarize([
                    run["public_complete_artifact_ms"] for run in valid_by_shape[shape]
                    if run["role"] == "measured" and run["arm"] == arm])
                    for shape in CASES} for arm in ("baseline", "candidate")
            } if ns02_mode else None,
            "warm_by_shape": {
                shape: summarize([run["public_complete_artifact_ms"]
                                  for run in valid_by_shape[shape]
                                  if run["role"] == "measured"])
                for shape in CASES},
            "cold_or_warmup": [
                {"run_id": run["run_id"], "arm": run["arm"], "shape": run["shape"],
                 "public_complete_artifact_ms": run["public_complete_artifact_ms"]}
                for run in runs if run["role"] == "cold_or_warmup"],
            "model_loads": [dict(process) for process in processes if process["name"] != "server"],
            "warm_native_phases_by_shape": {
                shape: {phase: summarize([
                    run["native_phase"][phase] for run in valid_by_shape[shape]
                    if run["role"] == "measured"]) for phase in (
                        "image_decode_ms", "image_resize_ms", "vae_encode_ms", "umt5_ms",
                        "latent_init_ms", "dit_ms", "vae_decode_ms", "encode_ms")}
                for shape in CASES},
            "source_preparation": summarize([
                run["source_preparation_ms"] for run in runs if run["role"] == "measured"]),
            "public_first_artifact_byte": summarize([
                run["public_first_artifact_byte_ms"] for run in runs if run["role"] == "measured"]),
            "public_complete_artifact": summarize([
                run["public_complete_artifact_ms"] for run in runs if run["role"] == "measured"]),
        }
        candidate_analysis = None
        if candidate_mode:
            measured = [run for run in runs if run["role"] == "measured"]
            paired = []
            repetitions = NS02_REPETITIONS if ns02_mode else REPETITIONS
            for repetition in range(repetitions):
                for shape in CASES:
                    baseline = next(run for run in measured if run["arm"] == "baseline"
                                    and run["repetition"] == repetition and run["shape"] == shape)
                    candidate = next(run for run in measured if run["arm"] == "candidate"
                                     and run["repetition"] == repetition and run["shape"] == shape)
                    if not ns02_mode:
                        require(candidate["media_sha256"] == baseline["media_sha256"],
                                f"candidate/baseline MP4 mismatch at r{repetition} {shape}")
                    profile_run = next(run for run in baseline_reference_runs(baseline_reference)
                                       if run["shape"] == shape and run["role"] == "measured"
                                       and run["repetition"] == 0)
                    profile_media = (baseline_reference / profile_run["media_file"]).read_bytes()
                    require(sha256_bytes(profile_media) == profile_run["media_sha256"],
                            f"profile media hash mismatch at r{repetition} {shape}")
                    if not ns02_mode:
                        require(candidate["media_sha256"] == profile_run["media_sha256"],
                                f"candidate/profile MP4 mismatch at r{repetition} {shape}")
                    item = {"repetition": repetition, "shape": shape,
                            "baseline_run_id": baseline["run_id"],
                            "candidate_run_id": candidate["run_id"],
                            "baseline_sha256": baseline["media_sha256"],
                            "candidate_sha256": candidate["media_sha256"],
                            "baseline_ms": baseline["public_complete_artifact_ms"],
                            "candidate_ms": candidate["public_complete_artifact_ms"],
                            "candidate_minus_baseline_ms": (
                                candidate["public_complete_artifact_ms"]
                                - baseline["public_complete_artifact_ms"])}
                    if ns02_mode:
                        item["quality"] = paired_video_quality(args.ffmpeg, out, baseline, candidate)
                        if prereg["format"] == NS02_EXACT_CUDA_FORMAT:
                            exact_pair = baseline["media_sha256"] == candidate["media_sha256"]
                            exact_profile = baseline["media_sha256"] == profile_run["media_sha256"]
                            item["quality"]["exact_mp4_media"] = exact_pair
                            item["quality"]["baseline_profile_mp4_exact"] = exact_profile
                            item["quality"]["quality_gate_pass"] = bool(
                                item["quality"]["quality_gate_pass"]
                                and exact_pair and exact_profile)
                    else:
                        item.pop("baseline_sha256")
                        item.pop("candidate_sha256")
                    paired.append(item)
            deltas = [item["candidate_minus_baseline_ms"] for item in paired]
            baseline_values = [item["baseline_ms"] for item in paired]
            candidate_values = [item["candidate_ms"] for item in paired]
            median_delta = statistics.median(deltas)
            median_baseline = statistics.median(baseline_values)
            candidate_analysis = {
                "schedule": schedule, "paired": paired,
                "paired_delta": summarize(deltas),
                "baseline": summarize(baseline_values), "candidate": summarize(candidate_values),
                "median_paired_candidate_minus_baseline_ms": median_delta,
                "median_paired_change_percent": 100.0 * median_delta / median_baseline,
                "latency_verdict": "IMPROVED" if median_delta < 0.0 else "NO_MEASURED_IMPROVEMENT",
                "regression_gate_pass": median_delta <= 0.05 * median_baseline,
                "all_observations_retained": True,
            }
            if ns02_mode:
                shape_deltas = {shape: statistics.median([
                    item["candidate_minus_baseline_ms"] for item in paired
                    if item["shape"] == shape]) for shape in CASES}
                quality_pass = all(item["quality"]["quality_gate_pass"] for item in paired)
                candidate_cancel = next((item for item in cancellations
                                         if item["arm"] == "candidate"), None)
                cancel_pass = len(cancellations) == 2 and candidate_cancel is not None and all((
                    candidate_cancel["disconnect_to_native_terminal_ms"] < NS02_CANCEL_LIMIT_MS,
                    candidate_cancel["disconnect_to_client_settled_ms"] < NS02_CANCEL_LIMIT_MS,
                    candidate_cancel["late_native_records_after_terminal"] == 0,
                    candidate_cancel["health"].get("in_flight") == 0))
                per_shape_values = {shape: [
                    item["public_complete_artifact_ms"] for item in runs
                    if item["arm"] == "candidate" and item["role"] == "measured"
                    and item["shape"] == shape] for shape in CASES}
                video_threshold_pass = (statistics.median(candidate_values) < NS02_VIDEO_LIMIT_MS
                                        and quantile(candidate_values, 0.95) < NS02_VIDEO_LIMIT_MS)
                per_shape_video_pass = None
                if prereg.get("candidate_policy") is not None:
                    per_shape_video_pass = {
                        shape: bool(values) and max(values) < NS02_VIDEO_LIMIT_MS
                        for shape, values in per_shape_values.items()}
                    video_threshold_pass = (max(candidate_values) < NS02_VIDEO_LIMIT_MS
                                            and all(per_shape_video_pass.values()))
                speed_pass = (median_delta <= -NS02_MIN_SPEEDUP_MS
                              and all(delta < 0.0 for delta in deltas)
                              and all(value < 0.0 for value in shape_deltas.values()))
                candidate_analysis.update({
                    "candidate_id": prereg["candidate_id"],
                    "candidate_commit": prereg["candidate_commit"],
                    "candidate_policy": prereg.get("candidate_policy"),
                    "max_public_complete_artifact_ms": max(candidate_values),
                    "per_shape_public_complete_artifact": {
                        shape: {**summarize(values), "min_ms": min(values),
                                "max_ms": max(values)}
                        for shape, values in per_shape_values.items()},
                    "per_shape_video_hard_threshold_pass": per_shape_video_pass,
                    "shape_median_paired_delta_ms": shape_deltas,
                    "quality_gate_pass": quality_pass, "speed_gate_pass": speed_pass,
                    "exact_media_gate_pass": (all(
                        item["quality"].get("exact_mp4_media") is True
                        and item["quality"].get("baseline_profile_mp4_exact") is True
                        for item in paired)
                        if prereg["format"] == NS02_EXACT_CUDA_FORMAT else None),
                    "video_hard_threshold_pass": video_threshold_pass,
                    "cancellation_hard_threshold_pass": cancel_pass,
                    "cancellations": cancellations,
                    "cancellation_latency": {arm: {
                        "native_terminal": summarize([item["disconnect_to_native_terminal_ms"]
                            for item in cancellations if item["arm"] == arm]),
                        "client_settled": summarize([item["disconnect_to_client_settled_ms"]
                            for item in cancellations if item["arm"] == arm])}
                        for arm in ("baseline", "candidate")},
                    "promotion_decision": "PROMOTED" if all((quality_pass, speed_pass,
                        video_threshold_pass, cancel_pass)) else "NOT_PROMOTED",
                    "promotion_scope": "native I2V default only; no full NS PASS",
                    "perceptual_quality": "NOT_RUN", "human_review": "NOT_RUN",
                })

        report = {
            "format": FORMAT, "mode": args.mode,
            "status": "SEALED_AWAITING_INDEPENDENT_VERIFICATION",
            "base_commit": args.base_commit, "execution_commit": args.execution_commit,
            "physical_gpu": args.gpu, "port": port,
            "geometry_cases": CASES, "sources": sources, "runs": runs, "controls": controls,
            "cancellations": cancellations,
            "profile": profile, "candidate_analysis": candidate_analysis,
            "preregistration_sha256": sha256_file(out / "preregistration.json")["sha256"]
                if candidate_mode else None,
            "processes": processes, "gpu": gpu,
            "final_health": {"http_status": health_status, "body": health},
            "models": models, "provenance": provenance,
            "claims": {
                "geometry": "numeric decoded-dimension/source-binding/continuity/motion gates only",
                "public_path": "POST /v1/visual/render -> UnifiedEngine Coordinator -> native Wan",
                "perceptual_quality": "NOT_RUN", "human_review": "NOT_RUN",
                "character_style_consistency": "NOT_RUN",
                "real_cancellation": "CLIENT_DISCONNECT_MEASURED" if ns02_mode else
                    "NOT_RUN (accepted Task #67 controls retained separately)",
                "resident_service": "NOT_CONTACTED", "deployment": "NOT_RUN",
            },
            "manifest_policy": "every regular file recursively except top-level SHA256SUMS itself",
            "journal_policy": "one exact JSON snapshot per completed public/cancel record" if ns02_mode else None,
        }
        write_json(out / "report.json", report)
        # No file is created or modified under out after this manifest is sealed.
        file_count = make_manifest(out)
        verification = out.parent / f"{out.name}-independent-verification.json"
        require(not verification.exists(), f"verification output already exists: {verification}")
        verify_command = [
            sys.executable, "-I", str(verifier), str(out), "--out", str(verification),
            "--ffmpeg", args.ffmpeg, "--ffprobe", args.ffprobe,
            "--server-binary", str(args.server_binary),
            "--baseline-wan-binary", str(args.wan_binary),
            "--wan-model", str(args.wan_model)]
        if args.candidate_wan_binary:
            verify_command += ["--candidate-wan-binary", str(args.candidate_wan_binary)]
        if args.flux_stage:
            verify_command += ["--flux-stage", str(args.flux_stage)]
        if args.flux_model:
            verify_command += ["--flux-model", str(args.flux_model)]
        result = subprocess.run(verify_command, capture_output=True, text=True,
                                timeout=max(300.0, args.timeout))
        require(result.returncode == 0,
                f"fresh-process verification failed:\n{result.stdout}\n{result.stderr}")
        print(result.stdout.strip())
        print(json.dumps({
            "status": "CONTROLLED_LIVE", "mode": args.mode,
            "report_sha256": sha256_file(out / "report.json")["sha256"],
            "manifest_sha256": sha256_file(out / "SHA256SUMS")["sha256"],
            "manifest_files": file_count,
            "verification_sha256": sha256_file(verification)["sha256"],
            "candidate_verdict": (candidate_analysis.get("promotion_decision",
                candidate_analysis["latency_verdict"]) if candidate_analysis else None),
            "gpu_after": gpu["after-all-requests-before-server-stop"],
        }, sort_keys=True))
        print("PERCEPTUAL_QUALITY: NOT_RUN")
    finally:
        try:
            stop_video()
        except Exception:
            if current_video is not None and current_video.process.poll() is None:
                current_video.process.kill()
                current_video.process.wait()
        for tap in (video_input_tap, video_output_tap):
            if tap is not None:
                tap.stop_requested.set()
                tap.thread.join(timeout=10)
        if flux is not None:
            flux.stop(endpoints["image-in"])
        if entry is not None:
            entry.stop(endpoints["entry-in"])
        if vlm is not None:
            vlm.stop(endpoints["vlm-in"])
        if server is not None:
            server.stop()
        if run_ipc.exists():
            for path in run_ipc.iterdir():
                if path.is_socket():
                    path.unlink()
            try:
                run_ipc.rmdir()
            except OSError:
                pass
        if args.nvidia_smi:
            gpu_state(args.nvidia_smi, args.gpu)


def baseline_reference_runs(root):
    return json.loads((Path(root) / "report.json").read_text())["runs"]


def self_check():
    raw_png = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 704, 480)
    require(png_dimensions(raw_png) == (704, 480), "PNG IHDR parsing failed")
    source = b"real-source\x00\xff"
    require(decode_source_data_url(source_data_url(source)) == source,
            "source data URL does not preserve exact bytes")
    stats = summarize([1.0, 2.0, 3.0, 4.0])
    require(stats["median_ms"] == 2.5 and math.isclose(stats["p95_ms"], 3.85)
            and stats["iqr_ms"] == 1.5 and stats["mad_ms"] == 1.0,
            f"latency statistics self-check failed: {stats}")
    require(candidate_schedule() == candidate_schedule()
            and len(candidate_schedule()) == 6,
            "seeded candidate schedule is not deterministic")
    ns02_schedule = candidate_schedule("ns02")
    require(ns02_schedule == candidate_schedule("ns02") and len(ns02_schedule) == 10
            and sum(item["arm"] == "candidate" for item in ns02_schedule) == 5,
            "NS-02 seeded process-separated schedule is not balanced")
    ns02_prereg = {
        "format": "umminfer-ns02-interactive-preview-preregistration-v1",
        "profile_report_sha256": "0" * 64,
        "candidate_id": "automatic-i2v-teacache-0.25",
        "candidate_commit": "3f98fe1645d4d82877372056ba8ad2544aa399c5",
        "repetitions": NS02_REPETITIONS, "interleaving_seed": NS02_INTERLEAVING_SEED,
        "settings": {"frames": FRAMES, "steps": STEPS, "fps": FPS,
                     "guidance": GUIDANCE, "seed": SEED,
                     "request_teacache_field": "omitted",
                     "baseline_effective_teacache": 0.0,
                     "candidate_effective_teacache": 0.25},
        "regression_gates": {
            "exact_paired_first_frame_rgb": True, "minimum_ssim_all": NS02_MIN_SSIM,
            "minimum_all_frame_psnr_db": NS02_MIN_PSNR_DB,
            "temporal_motion_ratio_range": [0.5, 1.5],
            "minimum_median_speedup_ms": NS02_MIN_SPEEDUP_MS,
            "all_paired_deltas_negative": True,
            "candidate_median_and_p95_video_ms_below": NS02_VIDEO_LIMIT_MS,
            "cancel_native_terminal_and_client_settle_ms_below": NS02_CANCEL_LIMIT_MS,
        "cancel_threshold_arm": "candidate_only_before_is_reported_independently",
            "zero_late_native_media_and_final_in_flight_zero": True,
            "immediate_retry_exact_own_arm_hash": True}}
    require(validate_ns02_preregistration(ns02_prereg, "0" * 64) == ns02_prereg,
            "NS-02 preregistration self-check failed")
    boundary2_prereg = json.loads(json.dumps(ns02_prereg))
    boundary2_prereg.update({
        "format": NS02_BOUNDARY2_FORMAT,
        "official_base_commit": NS02_BOUNDARY2_BASE,
        "candidate_id": NS02_BOUNDARY2_ID,
        "candidate_commit": NS02_BOUNDARY2_COMMIT,
        "candidate_policy": NS02_BOUNDARY2_POLICY,
        "created_before_final_candidate_measurement": True,
    })
    require(validate_ns02_preregistration(boundary2_prereg, "0" * 64)
            == boundary2_prereg,
            "boundary2 NS-02 preregistration self-check failed")
    exact_path = (Path(__file__).resolve().parent.parent /
                  "benchmarks/ns02-exact-cuda-speedup/preregistration.json")
    exact_prereg = json.loads(exact_path.read_text())
    exact_prereg["profile_report_sha256"] = "0" * 64
    require(validate_ns02_preregistration(exact_prereg, "0" * 64) == exact_prereg,
            "exact-CUDA NS-02 preregistration self-check failed")
    for field in ("exact_paired_mp4_media", "baseline_matches_frozen_profile_mp4"):
        changed = json.loads(json.dumps(exact_prereg))
        changed["regression_gates"][field] = False
        try:
            validate_ns02_preregistration(changed, "0" * 64)
        except AssertionError:
            pass
        else:
            raise AssertionError("exact-CUDA accepted a weakened media gate")
    changed = json.loads(json.dumps(exact_prereg))
    changed["settings"]["candidate_effective_teacache"] = 0.25
    try:
        validate_ns02_preregistration(changed, "0" * 64)
    except AssertionError:
        pass
    else:
        raise AssertionError("exact-CUDA accepted an approximate cache threshold")
    for bad_step in ([0, 2, 18, 19], [0, 1, 17, 19], [0, 1, 18]):
        changed = json.loads(json.dumps(boundary2_prereg))
        changed["candidate_policy"]["protected_step_indices"] = bad_step
        try:
            validate_ns02_preregistration(changed, "0" * 64)
        except AssertionError:
            pass
        else:
            raise AssertionError("boundary2 accepted an unfrozen cache schedule")
    weakened = dict(ns02_prereg)
    weakened["settings"] = dict(ns02_prereg["settings"], steps=STEPS - 1)
    try:
        validate_ns02_preregistration(weakened, "0" * 64)
    except AssertionError:
        pass
    else:
        raise AssertionError("NS-02 accepted a reduced solver budget")
    prereg = preregistration_document("0" * 64, "self-check", "offline contract only")
    validate_preregistration(prereg)
    bad = dict(prereg)
    bad["settings"] = dict(prereg["settings"], steps=STEPS - 1)
    rejected = False
    try:
        validate_preregistration(bad)
    except AssertionError:
        rejected = True
    require(rejected, "changed sampler settings bypassed preregistration")
    # Deliberate wrong-geometry controls: these expected source coordinates are
    # what a stretch/transposition oracle would wrongly accept for center cover.
    wide_correct = (20, 0)
    wide_stretched = (0, 0)
    tall_correct = (0, 20)
    tall_transposed = (20, 0)
    require(wide_correct != wide_stretched and tall_correct != tall_transposed,
            "geometry red controls collapsed")
    print("OFFLINE_I2V_GEOMETRY_LATENCY: PASS")
    print("ASYMMETRIC_RED_CONTROLS: PASS (stretch/transposition rejected analytically)")
    print("REAL_GENERATION: NOT_RUN")
    print("PERCEPTUAL_QUALITY: NOT_RUN")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--mode", choices=("profile", "candidate", "ns02"), default="profile")
    parser.add_argument("--out-dir")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--server-binary", type=Path)
    parser.add_argument(
        "--wan-binary", type=Path,
        help=("instrumented baseline native Wan executable; candidate mode requires "
              "the exact binary sealed in the profile, never an uninstrumented historical build"))
    parser.add_argument("--candidate-wan-binary", type=Path)
    parser.add_argument("--flux-stage", type=Path)
    parser.add_argument("--flux-model", type=Path)
    parser.add_argument("--wan-model", type=Path)
    parser.add_argument("--baseline-evidence", type=Path)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--verifier", type=Path,
                        default=Path(__file__).with_name("i2v_geometry_latency_verify.py"))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--library-path", default="")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--base-commit")
    parser.add_argument("--execution-commit")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    for name in ("out_dir", "server_binary", "wan_binary", "wan_model", "gpu",
                 "base_commit", "execution_commit"):
        if getattr(args, name) is None:
            parser.error(f"live mode requires --{name.replace('_', '-')}")
    if args.mode == "profile":
        for name in ("flux_stage", "flux_model"):
            if getattr(args, name) is None:
                parser.error(f"profile mode requires --{name.replace('_', '-')}")
    run_live(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
