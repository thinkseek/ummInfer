#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage C single-variable narrative-routing and visual-form ablations.

Routing varies only its policy. Visual form varies image/video presentation and
whether a second public render animates the image; image-only and I2V share a
byte-identical first-stage request. Prompt, seed, sampling, and selected
audio/visual option values stay fixed. Schema-v2 recording/validation and CAS are reused from
the RL exporter; latency, labels, and derived animations stay out-of-band.

  python3 -I python/umminfer_ablation_run.py \
    --config benchmarks/ablations/narrative-routing.json \
    --replay-only --out /tmp/umminfer-narrative-routing-ablation

Live mode additionally requires --base-url host:port (or http://host:port).
Replay is deterministic contract evidence, never real model/performance evidence.

An optional config `measurement` block (schema v1) declares warmup/measured
repetitions, deterministic interleaved or randomized arm order with a recorded
seed, exclusive-device preconditions, and cooldown/cleanup checks. Live runs
then execute that contract and earn CONTROLLED_LIVE only when every declared
repetition, exclusivity check, hash, and completion coverage is present;
otherwise they stay UNCONTROLLED_LIVE with explicit refusal reasons. Replay
runs carrying the block stay CONTRACT_REPLAY and only validate the schedule.
See python/umminfer_controlled_measurement.py and docs/narrative-routing-ablation.md.
"""

import argparse
import base64
import copy
from dataclasses import dataclass
import hashlib
import http.client
import io
import json
import math
from pathlib import Path
import re
import statistics
import struct
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
import zlib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import umminfer_ablation_metrics as routing_metrics  # noqa: E402
import umminfer_controlled_measurement as controlled  # noqa: E402
import umminfer_rl_reward as reward_mod  # noqa: E402
import umminfer_rl_trajectory as traj  # noqa: E402
import umminfer_rl_worker as worker  # noqa: E402

ABLATION_SCHEMA_VERSION = 1
REPLAY_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
POLICIES = ("text_only", "forced_image", "forced_video", "auto_tag")
VISUAL_FORMS = ("image_only", "video_only", "image_then_i2v_animate")
REQUEST_NATIVE_BACKENDS = ("qwen38", "wan22")
VARIABLES = ("routing_policy", "visual_form", "request_native_backend")
VISUAL_MODALITIES = ("image", "video")
HEX_256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass
class Experiment:
    config_path: Path
    config_bytes: bytes
    config: dict
    dataset_bytes: bytes
    dataset: dict
    replay_bytes: bytes | None
    replay: dict | None

    @property
    def prompts(self):
        return self.dataset["prompts"]

    @property
    def config_sha256(self):
        return hashlib.sha256(self.config_bytes).hexdigest()

    @property
    def data_sha256(self):
        return hashlib.sha256(self.dataset_bytes).hexdigest()

    @property
    def replay_sha256(self):
        if self.replay_bytes is None:
            return None
        return hashlib.sha256(self.replay_bytes).hexdigest()


def _canonical(value):
    return traj.canonical_json(value)


def _sha256(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b"\n")


def _read_json(path):
    raw = Path(path).read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return raw, value


def _check_hash(raw, expected, label):
    if not isinstance(expected, str) or not HEX_256.fullmatch(expected):
        raise ValueError(f"{label} SHA-256 must be 64 lowercase hex characters")
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def _resolve(base, relative):
    path = Path(relative)
    return path if path.is_absolute() else Path(base) / path


def _load_versioned(base, block, label, expected_version):
    if not isinstance(block, dict) or not isinstance(block.get("path"), str):
        raise ValueError(f"{label} requires path and sha256")
    path = _resolve(base, block["path"])
    raw, value = _read_json(path)
    _check_hash(raw, block.get("sha256"), label)
    if value.get("schema_version") != expected_version:
        raise ValueError(f"{label} has unsupported schema_version")
    return raw, value


def _validate_prompt(prompt):
    if not isinstance(prompt, dict) or not isinstance(prompt.get("id"), str) \
            or not prompt["id"] or not re.fullmatch(r"[a-z0-9_-]+", prompt["id"]):
        raise ValueError("prompt id must be a non-empty lowercase slug")
    if not isinstance(prompt.get("messages"), list) or not prompt["messages"]:
        raise ValueError(f"prompt {prompt['id']} lacks messages")
    if "expected_visual_modalities" in prompt:
        labels = prompt["expected_visual_modalities"]
        if not isinstance(labels, list) or any(
                label not in VISUAL_MODALITIES for label in labels):
            raise ValueError("expected_visual_modalities accepts only image/video")


def _validate_optional_hash(block, key):
    value = block.get(key)
    if value is not None and (not isinstance(value, str) or not HEX_256.fullmatch(value)):
        raise ValueError(f"{key} must be null or lowercase SHA-256")


def _validate_provenance_extras(block, label):
    _validate_optional_hash(block, "config_file_sha256")
    _validate_optional_hash(block, "weights_sha256")
    _validate_optional_hash(block, "binary_sha256")
    if "weights_sha256_scope" in block and not isinstance(
            block["weights_sha256_scope"], str):
        raise ValueError(f"{label}.weights_sha256_scope must be a string")
    files = block.get("config_files_sha256", {})
    if not isinstance(files, dict):
        raise ValueError(f"{label}.config_files_sha256 must be an object")
    for name, digest in files.items():
        if not isinstance(name, str) or not name or not isinstance(digest, str) \
                or not HEX_256.fullmatch(digest):
            raise ValueError(f"{label}.config_files_sha256 contains an invalid entry")


def _validation_policy(experiment):
    value = experiment.config.get("validation", {})
    if not isinstance(value, dict):
        raise ValueError("validation must be an object")
    decode = value.get("decode_media", False)
    first_frame = value.get("i2v_first_frame_crop", False)
    if not isinstance(decode, bool) or not isinstance(first_frame, bool):
        raise ValueError("validation decode_media and i2v_first_frame_crop must be boolean")
    maximum_mae = value.get("first_frame_max_mae", 32.0)
    if isinstance(maximum_mae, bool) or not isinstance(maximum_mae, (int, float)) \
            or not math.isfinite(maximum_mae) or not 0 <= maximum_mae <= 255:
        raise ValueError("validation.first_frame_max_mae must be finite in [0,255]")
    if first_frame and not decode:
        raise ValueError("i2v_first_frame_crop requires decode_media=true")
    return {"decode_media": decode, "i2v_first_frame_crop": first_frame,
            "first_frame_max_mae": float(maximum_mae)}


def _variable(experiment):
    return experiment.config["variable"]


def _variants(experiment):
    variable = _variable(experiment)
    key = {"routing_policy": "routing_policies", "visual_form": "visual_forms",
           "request_native_backend": "request_native_backends"}[variable]
    return experiment.config[key]


def load_experiment(config_path):
    config_path = Path(config_path).resolve()
    config_bytes, config = _read_json(config_path)
    if config.get("ablation_schema_version") != ABLATION_SCHEMA_VERSION \
            or config.get("trajectory_schema_version") != 2 \
            or config.get("variable") not in VARIABLES:
        raise ValueError("config requires ablation v1, trajectory v2, and a supported variable")
    variable = config["variable"]
    if variable == "routing_policy" and config.get("routing_policies") != list(POLICIES):
        raise ValueError("routing_policies must contain exactly the four honest baselines")
    if variable == "visual_form" and config.get("visual_forms") != list(VISUAL_FORMS):
        raise ValueError("visual_forms must contain exactly the three visual forms")
    if variable == "request_native_backend" \
            and config.get("request_native_backends") != list(REQUEST_NATIVE_BACKENDS):
        raise ValueError("request_native_backends must contain exactly qwen38 and wan22")
    if not isinstance(config.get("ablation_id"), str) or not config["ablation_id"]:
        raise ValueError("ablation_id must be a non-empty string")

    dataset_bytes, dataset = _load_versioned(
        config_path.parent, config.get("dataset"), "dataset", 1)
    prompts = dataset.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("dataset prompts must be a non-empty list")
    ids = []
    for prompt in prompts:
        _validate_prompt(prompt)
        ids.append(prompt["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("prompt ids must be unique")

    fixed = config.get("fixed")
    if not isinstance(fixed, dict) or not isinstance(fixed.get("sampling"), dict) \
            or not isinstance(fixed.get("visual"), dict):
        raise ValueError("fixed requires sampling and visual objects")
    if "audio" in fixed and not isinstance(fixed["audio"], dict):
        raise ValueError("fixed.audio must be an object when present")
    visual = fixed["visual"]
    if not isinstance(visual.get("max_slots"), int) \
            or isinstance(visual.get("max_slots"), bool) \
            or not 1 <= visual["max_slots"] <= 6:
        raise ValueError("fixed.visual.max_slots must be in [1,6]")
    for modality in VISUAL_MODALITIES:
        if not isinstance(visual.get(modality), dict):
            raise ValueError(f"fixed.visual.{modality} must be an object")
    forbidden = {"model", "messages", "stream", "narrative", "modalities",
                 "audio", "image", "video"}
    overlap = forbidden & set(fixed["sampling"])
    if overlap:
        raise ValueError(f"sampling contains non-sampling fields: {sorted(overlap)}")
    model = config.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("id"), str) \
            or not model["id"] or not isinstance(model.get("revision"), str) \
            or not model["revision"]:
        raise ValueError("model requires non-empty id and revision")
    weights_hash = model.get("weights_sha256")
    if weights_hash is not None and (
            not isinstance(weights_hash, str) or not HEX_256.fullmatch(weights_hash)):
        raise ValueError("model.weights_sha256 must be null or lowercase SHA-256")
    _validate_provenance_extras(model, "model")
    backends = config.get("visual_backends", {})
    if not isinstance(backends, dict):
        raise ValueError("visual_backends must be an object")
    for name, backend in backends.items():
        if name not in VISUAL_MODALITIES or not isinstance(backend, dict) \
                or not isinstance(backend.get("id"), str) or not backend["id"] \
                or not isinstance(backend.get("revision"), str) or not backend["revision"]:
            raise ValueError("visual_backends accepts image/video id and revision objects")
        _validate_provenance_extras(backend, f"visual_backends.{name}")
    auxiliary = config.get("auxiliary_backends", {})
    if not isinstance(auxiliary, dict):
        raise ValueError("auxiliary_backends must be an object")
    for name, backend in auxiliary.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9_-]+", name) \
                or not isinstance(backend, dict) \
                or not isinstance(backend.get("id"), str) or not backend["id"] \
                or not isinstance(backend.get("revision"), str) or not backend["revision"]:
            raise ValueError("auxiliary_backends accepts named id and revision objects")
        _validate_provenance_extras(backend, f"auxiliary_backends.{name}")
    _validation_policy(Experiment(
        config_path, config_bytes, config, b"", {}, None, None))
    controlled.parse_contract(config.get("measurement"))

    replay_bytes = replay = None
    if "replay" in config:
        replay_bytes, replay = _load_versioned(
            config_path.parent, config["replay"], "replay", REPLAY_SCHEMA_VERSION)
        if not isinstance(replay.get("cases"), list):
            raise ValueError("replay cases must be a list")
    return Experiment(config_path, config_bytes, config, dataset_bytes, dataset,
                      replay_bytes, replay)


def routing_overlay(policy, max_slots, include_audio=True):
    """The only fields allowed to vary between otherwise identical requests."""
    if policy not in POLICIES:
        raise ValueError(f"unknown routing policy: {policy}")
    narrative = {"enabled": policy != "text_only", "visual_mode": "auto",
                 "max_slots": max_slots}
    modalities = ["text", "audio"] if include_audio else ["text"]
    if policy == "forced_image":
        narrative["visual_mode"] = "image"
        modalities.append("image")
    elif policy == "forced_video":
        narrative["visual_mode"] = "video"
        modalities.append("video")
    elif policy == "auto_tag":
        modalities.extend(VISUAL_MODALITIES)
    return {"modalities": modalities, "narrative": narrative}


def visual_form_overlay(form, max_slots):
    """Image-only and I2V deliberately share the same first-stage request."""
    if form not in VISUAL_FORMS:
        raise ValueError(f"unknown visual form: {form}")
    video = form == "video_only"
    return {
        "modalities": ["text", "video" if video else "image"],
        "narrative": {"enabled": True, "visual_mode": "video" if video else "image",
                      "max_slots": max_slots},
    }


def build_request(experiment, prompt, policy):
    fixed = experiment.config["fixed"]
    body = {
        "model": experiment.config["model"]["id"],
        "stream": True,
        "messages": copy.deepcopy(prompt["messages"]),
    }
    body.update(copy.deepcopy(fixed["sampling"]))
    if _variable(experiment) == "routing_policy":
        body.update(routing_overlay(policy, fixed["visual"]["max_slots"],
                                    include_audio="audio" in fixed))
    elif _variable(experiment) == "request_native_backend":
        # Two fully instrumented native paths; never substitute an
        # uninstrumented image arm when request_memory.required=true.
        overlay = "text_only" if policy == "qwen38" else "forced_video"
        body.update(routing_overlay(overlay, fixed["visual"]["max_slots"],
                                    include_audio=False))
    else:
        body.update(visual_form_overlay(policy, fixed["visual"]["max_slots"]))
    # The public parser treats a present options object as enabling its output.
    # Send each fixed object only when the routing overlay selects that modality.
    selected = set(body["modalities"])
    if "audio" in selected:
        body["audio"] = copy.deepcopy(fixed["audio"])
    for modality in VISUAL_MODALITIES:
        if modality in selected:
            body[modality] = copy.deepcopy(fixed["visual"][modality])
    request_memory = experiment.config.get("measurement", {}).get("request_memory")
    if request_memory and request_memory.get("enabled"):
        body["telemetry"] = {"native_allocator": True}
    return body


def build_render_request(experiment, slot, image_bytes):
    """Build the second public request used only by image_then_i2v_animate."""
    if _variable(experiment) != "visual_form":
        raise ValueError("I2V render requests belong to the visual_form ablation")
    video = experiment.config["fixed"]["visual"]["video"]
    body = {
        "kind": "video",
        "prompt": slot["prompt"],
        "image": "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii"),
    }
    for key in ("size", "steps", "num_frames", "fps", "seconds",
                "guidance_scale", "seed"):
        if key in video:
            body[key] = copy.deepcopy(video[key])
    return body


def fixed_render_request(body):
    return {key: copy.deepcopy(value) for key, value in body.items() if key != "image"}


def fixed_render_request_sha256(body):
    return _sha256(fixed_render_request(body))


def fixed_request(body):
    return {key: copy.deepcopy(value) for key, value in body.items()
            if key not in ("modalities", "narrative", "audio", "image", "video")}


def fixed_request_sha256(body):
    return _sha256(fixed_request(body))


def fixed_matrix(experiment):
    rows = []
    variants = _variants(experiment)
    for prompt in experiment.prompts:
        body = build_request(experiment, prompt, variants[0])
        rows.append({"prompt_id": prompt["id"],
                     "messages_sha256": _sha256(prompt["messages"]),
                     "fixed_request_sha256": fixed_request_sha256(body)})
    overlay_fields = ["modalities", "narrative", "audio", "image", "video"]
    if _variable(experiment) == "visual_form":
        overlay_fields.append("post_processing.i2v_animate")
    return {"variable": _variable(experiment),
            "overlay_fields": overlay_fields,
            "fixed_matrix_sha256": _sha256(rows), "prompts": rows}


def _model_provenance(model):
    identity = {"id": model["id"], "revision": model["revision"]}
    result = {
        "id": model["id"], "revision": model["revision"],
        "config_sha256": _sha256(identity),
        "weights_sha256": model.get("weights_sha256"),
        "weights_coverage": "COVERED" if model.get("weights_sha256") else "NOT_COVERED",
    }
    for key in ("weights_sha256_scope", "config_file_sha256", "config_files_sha256",
                "binary_sha256"):
        if key in model:
            result[key] = copy.deepcopy(model[key])
    return result


def _source_provenance():
    """Bind HEAD plus every tracked diff and untracked source byte."""
    def git_bytes(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                              capture_output=True).stdout

    commit = git_bytes("rev-parse", "HEAD").decode("ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("git returned an invalid source commit")
    diff = git_bytes("diff", "--binary", "HEAD")
    untracked_names = git_bytes(
        "ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
    untracked = []
    for encoded in sorted(name for name in untracked_names if name):
        name = encoded.decode("utf-8", "surrogateescape")
        path = ROOT / name
        if not path.is_file():
            raise ValueError(f"untracked source is not a regular file: {name}")
        raw = path.read_bytes()
        untracked.append({"path": name, "bytes": len(raw),
                          "sha256": hashlib.sha256(raw).hexdigest(),
                          "content_base64": base64.b64encode(raw).decode("ascii")})
    state = {"commit": commit, "tracked_diff_base64":
             base64.b64encode(diff).decode("ascii"), "untracked": untracked}
    return {"commit": commit, "dirty": bool(diff or untracked),
            "state_sha256": _sha256(state)}


def _prepare_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"output directory must be absent or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_base_url(value):
    if "://" not in value:
        value = "http://" + value
    parsed = urlsplit(value)
    if parsed.scheme != "http" or not parsed.netloc \
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment \
            or parsed.username is not None or parsed.password is not None:
        raise ValueError("base URL must be plain http://host:port without credentials/path")
    return parsed.netloc


def _elapsed_us(start_ns):
    return max(0, (time.monotonic_ns() - start_ns) // 1000)


class LatencyTracker:
    """Out-of-band monotonic observations; nothing is added to trajectory v2."""

    def __init__(self):
        self.start_ns = time.monotonic_ns()
        self.raw = {"ttft_us": None, "ttfa_us": None, "terminal_us": None,
                    "slots": []}
        self.slots = {}

    def observe(self, event):
        elapsed = _elapsed_us(self.start_ns)
        kind = event.get("type")
        if kind in ("vlm_delta", "output_delta") and event.get("text") \
                and self.raw["ttft_us"] is None:
            self.raw["ttft_us"] = elapsed
        if kind == "artifact":
            descriptor = event.get("artifact", {})
            if descriptor.get("media_type") == "audio" \
                    and self.raw["ttfa_us"] is None:
                self.raw["ttfa_us"] = elapsed
            slot_id = descriptor.get("slot_id")
            if slot_id in self.slots:
                slot = self.slots[slot_id]
                if slot["first_artifact_us"] is None:
                    slot["first_artifact_us"] = elapsed
                if descriptor.get("index", 0) + 1 == descriptor.get("total", 1):
                    slot["settlement_us"] = elapsed
        elif kind == "visual_slot":
            slot_id = event["slot_id"]
            slot = {"slot_id": slot_id, "dispatch_us": elapsed,
                    "first_artifact_us": None, "settlement_us": None}
            self.slots[slot_id] = slot
            self.raw["slots"].append(slot)
        elif kind == "leg_error" and event.get("slot_id") in self.slots:
            self.slots[event["slot_id"]]["settlement_us"] = elapsed
        elif kind in ("complete", "error", "cancelled"):
            self.raw["terminal_us"] = elapsed

    def finish(self):
        if self.raw["terminal_us"] is None:
            self.raw["terminal_us"] = _elapsed_us(self.start_ns)
        return self.raw


def _optional_us(value, label):
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be null or a non-negative integer")
    return value


def _ms(value):
    return None if value is None else value / 1000.0


def _paeth(a, b, c):
    estimate = a + b - c
    distances = (abs(estimate - a), abs(estimate - b), abs(estimate - c))
    return (a, b, c)[distances.index(min(distances))]


def decode_png_rgb8(raw):
    """Decode non-interlaced 8-bit PNG and verify every chunk CRC."""
    if not isinstance(raw, bytes) or not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("PNG signature is missing")
    offset = 8
    header = None
    compressed = bytearray()
    ended = False
    while offset < len(raw):
        if offset + 12 > len(raw):
            raise ValueError("truncated PNG chunk header")
        length = struct.unpack_from(">I", raw, offset)[0]
        kind = raw[offset + 4:offset + 8]
        end = offset + 12 + length
        if end > len(raw):
            raise ValueError("truncated PNG chunk payload")
        payload = raw[offset + 8:offset + 8 + length]
        expected_crc = struct.unpack_from(">I", raw, offset + 8 + length)[0]
        if zlib.crc32(kind + payload) & 0xffffffff != expected_crc:
            raise ValueError("PNG chunk CRC mismatch")
        if kind == b"IHDR":
            if header is not None or length != 13 or offset != 8:
                raise ValueError("invalid PNG IHDR")
            header = struct.unpack(">IIBBBBB", payload)
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            if length != 0 or end != len(raw):
                raise ValueError("invalid PNG IEND")
            ended = True
            offset = end
            break
        offset = end
    if header is None or not ended or not compressed:
        raise ValueError("PNG lacks IHDR, IDAT, or IEND")
    width, height, depth, color, compression, filter_method, interlace = header
    channels_by_color = {0: 1, 2: 3, 4: 2, 6: 4}
    if width <= 0 or height <= 0 or width * height > 100_000_000:
        raise ValueError("invalid or excessive PNG dimensions")
    if depth != 8 or color not in channels_by_color or compression != 0 \
            or filter_method != 0 or interlace != 0:
        raise ValueError("only non-interlaced 8-bit grayscale/RGB/alpha PNG is supported")
    channels = channels_by_color[color]
    stride = width * channels
    try:
        packed = zlib.decompress(bytes(compressed))
    except zlib.error as exc:
        raise ValueError(f"PNG zlib decode failed: {exc}") from exc
    if len(packed) != height * (stride + 1):
        raise ValueError("PNG decoded scanline length mismatch")
    rows = []
    previous = bytearray(stride)
    cursor = 0
    for _ in range(height):
        filter_type = packed[cursor]
        cursor += 1
        source = packed[cursor:cursor + stride]
        cursor += stride
        row = bytearray(stride)
        for index, byte in enumerate(source):
            left = row[index - channels] if index >= channels else 0
            above = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = above
            elif filter_type == 3:
                predictor = (left + above) // 2
            elif filter_type == 4:
                predictor = _paeth(left, above, upper_left)
            else:
                raise ValueError("unsupported PNG scanline filter")
            row[index] = (byte + predictor) & 0xff
        rows.append(row)
        previous = row
    rgb = bytearray(width * height * 3)
    for y, row in enumerate(rows):
        for x in range(width):
            source = x * channels
            target = (y * width + x) * 3
            if color == 2:
                rgb[target:target + 3] = row[source:source + 3]
            elif color == 6:
                rgb[target:target + 3] = row[source:source + 3]
            else:
                gray = row[source]
                rgb[target:target + 3] = bytes((gray, gray, gray))
    return width, height, bytes(rgb)


def _media_command(command, raw, label, timeout_s=120.0):
    try:
        completed = subprocess.run(command, input=raw, capture_output=True,
                                   timeout=timeout_s, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"{label} failed: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace")[-2048:]
        raise ValueError(f"{label} exited {completed.returncode}: {detail}")
    return completed.stdout


def decode_video_mp4(raw, *, expected_frames=None, expected_size=None):
    """Fully decode one video with ffmpeg and count frames with ffprobe."""
    if not raw.startswith(b"\x00\x00\x00") or raw[4:8] != b"ftyp":
        raise ValueError("MP4 ftyp box is missing")
    probe_raw = _media_command([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries",
        "stream=codec_name,width,height,nb_read_frames,avg_frame_rate",
        "-of", "json", "-i", "pipe:0"], raw, "ffprobe")
    try:
        streams = json.loads(probe_raw).get("streams")
        if not isinstance(streams, list) or len(streams) != 1:
            raise ValueError("expected exactly one video stream")
        stream = streams[0]
        frames = int(stream["nb_read_frames"])
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid ffprobe video metadata: {exc}") from exc
    if frames <= 0 or width <= 0 or height <= 0:
        raise ValueError("video decode returned no frames or invalid dimensions")
    if expected_frames is not None and frames != expected_frames:
        raise ValueError(f"video frame count {frames} != requested {expected_frames}")
    if expected_size is not None and (width, height) != expected_size:
        raise ValueError(f"video size {width}x{height} != requested {expected_size}")
    _media_command(["ffmpeg", "-v", "error", "-xerror", "-i", "pipe:0",
                    "-map", "0:v:0", "-f", "null", "-"], raw, "ffmpeg full decode")
    first = _media_command(["ffmpeg", "-v", "error", "-xerror", "-i", "pipe:0",
                            "-map", "0:v:0", "-frames:v", "1", "-pix_fmt", "rgb24",
                            "-f", "rawvideo", "pipe:1"], raw, "ffmpeg first-frame decode")
    if len(first) != width * height * 3:
        raise ValueError("decoded first-frame byte count does not match video dimensions")
    descriptor = {
        "codec": stream.get("codec_name"), "width": width, "height": height,
        "frame_count": frames, "avg_frame_rate": stream.get("avg_frame_rate"),
        "full_decode": "PASSED", "first_frame_sha256": hashlib.sha256(first).hexdigest(),
    }
    return descriptor, first


def _resize_rgb8(source, src_width, src_height, dst_width, dst_height,
                 *, preserve_aspect, centered):
    """Mirror native Wan I2V half-pixel bilinear sampling in RGB8 space."""
    if len(source) != src_width * src_height * 3:
        raise ValueError("source RGB buffer size mismatch")
    scale_x = src_width / dst_width
    scale_y = src_height / dst_height
    if preserve_aspect:
        scale_x = scale_y = min(scale_x, scale_y)
    offset_x = (src_width - scale_x * dst_width) * 0.5 if centered else 0.0
    offset_y = (src_height - scale_y * dst_height) * 0.5 if centered else 0.0
    output = bytearray(dst_width * dst_height * 3)
    for y in range(dst_height):
        sy = min(max(offset_y + (y + 0.5) * scale_y - 0.5, 0.0), src_height - 1.0)
        y0 = int(sy)
        y1 = min(y0 + 1, src_height - 1)
        wy = sy - y0
        for x in range(dst_width):
            sx = min(max(offset_x + (x + 0.5) * scale_x - 0.5, 0.0), src_width - 1.0)
            x0 = int(sx)
            x1 = min(x0 + 1, src_width - 1)
            wx = sx - x0
            target = (y * dst_width + x) * 3
            for channel in range(3):
                def at(px, py):
                    return source[(py * src_width + px) * 3 + channel]
                top = at(x0, y0) * (1.0 - wx) + at(x1, y0) * wx
                bottom = at(x0, y1) * (1.0 - wx) + at(x1, y1) * wx
                output[target + channel] = min(255, max(0, round(
                    top * (1.0 - wy) + bottom * wy)))
    return bytes(output)


def _pixel_error(actual, expected):
    if len(actual) != len(expected) or not actual:
        raise ValueError("pixel comparison buffer size mismatch")
    absolute = [abs(a - b) for a, b in zip(actual, expected)]
    mae = sum(absolute) / len(absolute)
    mse = sum(value * value for value in absolute) / len(absolute)
    return mae, mse


def check_i2v_first_frame(source, source_size, first_frame, video_size, max_mae):
    """Check the known center-crop behavior, not perceptual generation quality."""
    src_width, src_height = source_size
    width, height = video_size
    if len(first_frame) != width * height * 3:
        raise ValueError("first-frame RGB buffer size mismatch")
    center = _resize_rgb8(source, src_width, src_height, width, height,
                          preserve_aspect=True, centered=True)
    stretch = _resize_rgb8(source, src_width, src_height, width, height,
                           preserve_aspect=False, centered=False)
    top_left = _resize_rgb8(source, src_width, src_height, width, height,
                            preserve_aspect=True, centered=False)
    center_mae, center_mse = _pixel_error(first_frame, center)
    stretch_mae, _ = _pixel_error(first_frame, stretch)
    top_left_mae, _ = _pixel_error(first_frame, top_left)
    exact = center_mse == 0
    psnr = None if exact else 10.0 * math.log10(255.0 * 255.0 / center_mse)
    passed = center_mae <= max_mae and center_mae <= stretch_mae \
        and center_mae <= top_left_mae
    return {
        "metric_class": "algorithmic-behavior-proxy",
        "passed": passed, "source_size": [src_width, src_height],
        "video_size": [width, height],
        "center_crop_mae": center_mae, "stretch_mae": stretch_mae,
        "top_left_crop_mae": top_left_mae,
        "center_crop_minus_stretch_mae": center_mae - stretch_mae,
        "center_crop_minus_top_left_mae": center_mae - top_left_mae,
        "first_frame_psnr_db": psnr, "exact_pixel_match": exact,
        "maximum_allowed_mae": max_mae,
        "quality_coverage": "NOT_COVERED",
    }


def _expected_size(options):
    match = re.fullmatch(r"([1-9][0-9]*)x([1-9][0-9]*)", options.get("size", ""))
    if not match:
        raise ValueError("media validation requires fixed WIDTHxHEIGHT size")
    return int(match.group(1)), int(match.group(2))


def _validate_record_media(experiment, record, artifact_dir):
    policy = _validation_policy(experiment)
    rows = []
    for artifact in record["artifacts"]:
        row = {"artifact_id": artifact.get("artifact_id"),
               "sha256": artifact.get("sha256"),
               "media_type": artifact.get("media_type")}
        try:
            raw = (Path(artifact_dir) / artifact["ref"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != artifact["sha256"] \
                    or len(raw) != artifact["bytes"]:
                raise ValueError("artifact CAS size/hash mismatch")
            if not policy["decode_media"]:
                row.update({"status": "NOT_COVERED", "decode_coverage": "NOT_COVERED"})
            elif artifact.get("media_type") == "image":
                width, height, pixels = decode_png_rgb8(raw)
                expected = _expected_size(experiment.config["fixed"]["visual"]["image"])
                if (width, height) != expected:
                    raise ValueError(f"image size {width}x{height} != requested {expected}")
                row.update({"status": "passed", "decode_coverage": "COVERED",
                            "full_decode": "PASSED", "width": width, "height": height,
                            "frame_count": 1,
                            "decoded_rgb_sha256": hashlib.sha256(pixels).hexdigest()})
            elif artifact.get("media_type") == "video":
                visual = experiment.config["fixed"]["visual"]["video"]
                descriptor, _ = decode_video_mp4(
                    raw, expected_frames=visual.get("num_frames"),
                    expected_size=_expected_size(visual))
                row.update(descriptor)
                row.update({"status": "passed", "decode_coverage": "COVERED"})
            else:
                row.update({"status": "NOT_APPLICABLE",
                            "decode_coverage": "NOT_APPLICABLE"})
        except (OSError, KeyError, ValueError) as exc:
            row.update({"status": "failed", "decode_coverage": "COVERED",
                        "error": str(exc)})
        rows.append(row)
    return rows


def _query_nvidia_smi(indexes):
    try:
        output = subprocess.run([
            "nvidia-smi", "-i", ",".join(str(index) for index in indexes),
            "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        raise ValueError(f"nvidia-smi sampling failed: {exc}") from exc
    result = {}
    for line in output.splitlines():
        index, used = line.split(",", 1)
        result[int(index.strip())] = int(used.strip())
    if set(result) != set(indexes):
        raise ValueError("nvidia-smi returned a different GPU set")
    return result


class GpuResourceMonitor:
    """Sample explicitly selected whole-device memory without process discovery."""

    def __init__(self, indexes, sampler=None, interval_s=0.05):
        self.indexes = tuple(sorted(set(indexes)))
        self.sampler = sampler or _query_nvidia_smi
        self.interval_s = interval_s
        self.samples = []
        self.error = None
        self.stop_event = threading.Event()
        self.thread = None

    def _sample(self):
        sample = self.sampler(self.indexes)
        if set(sample) != set(self.indexes):
            raise ValueError("GPU sampler returned a different device set")
        self.samples.append({str(index): int(sample[index]) for index in self.indexes})

    def _run(self):
        try:
            while not self.stop_event.wait(self.interval_s):
                self._sample()
        except Exception as exc:  # surfaced by stop; never silently drop coverage
            self.error = exc
            self.stop_event.set()

    def start(self):
        if not self.indexes:
            raise ValueError("GPU monitor requires at least one device")
        self._sample()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
        if self.error is not None:
            raise ValueError(f"GPU resource sampling failed: {self.error}")
        self._sample()
        rows = {}
        for index in self.indexes:
            values = [sample[str(index)] for sample in self.samples]
            rows[str(index)] = {"baseline_mib": values[0], "peak_mib": max(values),
                                "final_mib": values[-1], "sample_count": len(values)}
        return {"coverage": "COVERED", "scope": "whole-device-used-memory",
                "source": "nvidia-smi-poll", "poll_interval_ms": self.interval_s * 1000,
                "attribution": "includes persistent model baseline; not request-only allocation",
                "gpus": rows}


def _uncovered_resources():
    return {"coverage": "NOT_COVERED", "reason": "no explicit --gpu-index sampler",
            "gpus": {}}


def normalize_latency(raw, record, source):
    if not isinstance(raw, dict):
        raise ValueError("latency observation must be an object")
    expected_slots = {slot["slot_id"]: slot for slot in record["visual_slots"]}
    raw_slots = raw.get("slots")
    if not isinstance(raw_slots, list):
        raise ValueError("latency slots must be a list")
    by_id = {}
    for slot in raw_slots:
        if not isinstance(slot, dict) or not isinstance(slot.get("slot_id"), str) \
                or slot["slot_id"] in by_id:
            raise ValueError("latency slot ids are missing or duplicated")
        by_id[slot["slot_id"]] = slot
    if set(by_id) != set(expected_slots):
        raise ValueError("latency slots do not match trajectory visual_slots")
    slots = []
    for slot_id, event in expected_slots.items():
        observed = by_id[slot_id]
        dispatch = _optional_us(observed.get("dispatch_us"), "slot dispatch_us")
        first = _optional_us(observed.get("first_artifact_us"), "slot first_artifact_us")
        settled = _optional_us(observed.get("settlement_us"), "slot settlement_us")
        if dispatch is None:
            raise ValueError("every visual slot requires dispatch_us")
        slot_to_artifact = None if first is None else first - dispatch
        if slot_to_artifact is not None and slot_to_artifact < 0:
            raise ValueError("slot artifact predates dispatch")
        slots.append({
            "slot_id": slot_id, "slot_index": event["slot_index"],
            "media_type": event["media_type"], "dispatch_ms": _ms(dispatch),
            "first_artifact_ms": _ms(first),
            "slot_to_artifact_ms": _ms(slot_to_artifact),
            "settlement_ms": _ms(settled),
        })
    return {
        "source": source,
        "ttft_ms": _ms(_optional_us(raw.get("ttft_us"), "ttft_us")),
        "ttfa_ms": _ms(_optional_us(raw.get("ttfa_us"), "ttfa_us")),
        "terminal_ms": _ms(_optional_us(raw.get("terminal_us"), "terminal_us")),
        "slots": slots,
    }


def _replay_map(experiment):
    if experiment.replay is None:
        raise ValueError("replay-only mode requires config.replay")
    variable = _variable(experiment)
    expected = {(variant, prompt["id"]) for variant in _variants(experiment)
                for prompt in experiment.prompts}
    result = {}
    for case in experiment.replay["cases"]:
        if not isinstance(case, dict):
            raise ValueError("replay case must be an object")
        key = (case.get(variable), case.get("prompt_id"))
        if key not in expected or key in result:
            raise ValueError(f"unexpected or duplicate replay case: {key}")
        result[key] = case
    if set(result) != expected:
        raise ValueError(f"replay is missing cases: {sorted(expected - set(result))}")
    return result


def _record_replay(case, body, recorder, session_id):
    response_hash = case.get("response_sha256")
    encoded = case.get("response_sse_base64")
    if not isinstance(encoded, str):
        raise ValueError("replay response_sse_base64 must be a string")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid replay response base64: {exc}") from exc
    _check_hash(raw, response_hash, "replay response")
    if case.get("request_sha256") != _sha256(body):
        raise ValueError("replay request_sha256 does not match the fixed config/prompt")
    if case.get("fixed_request_sha256") != fixed_request_sha256(body):
        raise ValueError("replay fixed_request_sha256 mismatch")
    params, option_inputs = traj.request_parameter_references(
        body, recorder.artifact_dir)
    inputs = traj.input_references(body["messages"], recorder.artifact_dir) \
        + option_inputs
    details = {}
    request_id, events, output_text, state, error = traj.parse_sse_stream(
        io.BytesIO(raw), artifact_dir=recorder.artifact_dir, details=details)
    record = recorder.record(
        session_id=session_id, request_id=request_id or "replay-request",
        endpoint="/v1/chat/completions", inputs=inputs, params=params,
        events=events, terminal_state=state, error=error,
        output_text=output_text, artifacts=details.get("artifacts", []))
    latency = normalize_latency(case.get("latency_us"), record,
                                "recorded-contract-observation")
    return record, latency


def _source_image(record, slot, artifact_dir):
    matches = [artifact for artifact in record["artifacts"]
               if artifact.get("slot_id") == slot["slot_id"]
               and artifact.get("media_type") == "image"]
    if len(matches) != 1:
        raise ValueError(f"I2V source slot {slot['slot_id']} requires one image artifact")
    artifact = matches[0]
    raw = (Path(artifact_dir) / artifact["ref"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != artifact["sha256"] \
            or len(raw) != artifact["bytes"]:
        raise ValueError("I2V source image failed CAS verification")
    return artifact, raw


def _animation_artifact(artifact_dir, slot, source, raw):
    digest = hashlib.sha256(raw).hexdigest()
    ref = traj._store_artifact(artifact_dir, digest, ".mp4", raw)
    return {
        "artifact_id": slot["slot_id"] + ":i2v",
        "media_type": "video", "mime_type": "video/mp4", "format": "mp4",
        "source_slot_id": slot["slot_id"],
        "source_artifact_id": source["artifact_id"],
        "source_image_sha256": source["sha256"],
        "ref": ref, "sha256": digest, "bytes": len(raw),
        "chunk_count": 1,
        "chunks_sha256": hashlib.sha256(bytes.fromhex(digest)).hexdigest(),
    }


def _run_one_animation(experiment, record, slot, recorder, *, base_url,
                       timeout_s, replay_case=None):
    source = None
    body = None
    try:
        source, image_bytes = _source_image(record, slot, recorder.artifact_dir)
        body = build_render_request(experiment, slot, image_bytes)
    except (OSError, ValueError, KeyError) as exc:
        return {"source_slot_id": slot["slot_id"], "terminal_state": "failed",
                "error": str(exc), "request_sha256": None,
                "fixed_render_request_sha256": None, "artifact": None,
                "latency": {"source": "monotonic-live" if replay_case is None
                            else "recorded-contract-observation", "terminal_ms": None}}

    result = {
        "source_slot_id": slot["slot_id"],
        "source_artifact_id": source["artifact_id"],
        "source_image_sha256": source["sha256"],
        "request_sha256": _sha256(body),
        "fixed_render_request_sha256": fixed_render_request_sha256(body),
    }
    if replay_case is not None:
        if replay_case.get("source_slot_id") != slot["slot_id"] \
                or replay_case.get("source_image_sha256") != source["sha256"] \
                or replay_case.get("request_sha256") != result["request_sha256"] \
                or replay_case.get("fixed_render_request_sha256") != \
                result["fixed_render_request_sha256"]:
            raise ValueError("replay I2V request/source binding mismatch")
        try:
            raw = base64.b64decode(replay_case["response_base64"], validate=True)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"invalid replay I2V response base64: {exc}") from exc
        _check_hash(raw, replay_case.get("response_sha256"), "replay I2V response")
        status = replay_case.get("http_status", 200)
        state = replay_case.get("terminal_state", "completed")
        error = replay_case.get("error")
        elapsed_us = _optional_us(replay_case.get("latency_us", {}).get("terminal_us"),
                                  "I2V terminal_us")
        source_name = "recorded-contract-observation"
    else:
        started = time.monotonic_ns()
        conn = http.client.HTTPConnection(base_url, timeout=timeout_s)
        try:
            conn.request("POST", "/v1/visual/render", body=json.dumps(body),
                         headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            status = response.status
            raw = response.read()
            content_type = response.getheader("Content-Type", "")
        except (OSError, http.client.HTTPException) as exc:
            status, raw, content_type = 0, str(exc).encode("utf-8"), ""
        finally:
            conn.close()
        elapsed_us = _elapsed_us(started)
        state = "completed" if status == 200 else "failed"
        error = None
        source_name = "monotonic-live"
        if status != 200:
            error = raw.decode("utf-8", "replace")[:4096] or f"HTTP {status}"
        elif not content_type.startswith("video/mp4"):
            state, error = "failed", f"unexpected I2V Content-Type: {content_type}"

    if state == "completed" and not raw:
        state, error = "failed", "I2V render returned an empty video"
    if state not in ("completed", "failed", "cancelled"):
        raise ValueError("invalid I2V terminal state")
    artifact = None
    if state == "completed":
        artifact = _animation_artifact(recorder.artifact_dir, slot, source, raw)
    result.update({
        "http_status": status, "terminal_state": state, "error": error,
        "latency": {"source": source_name, "terminal_ms": _ms(elapsed_us)},
        "artifact": artifact,
    })
    policy = _validation_policy(experiment)
    if state == "completed" and policy["decode_media"]:
        try:
            video = experiment.config["fixed"]["visual"]["video"]
            descriptor, first_frame = decode_video_mp4(
                raw, expected_frames=video.get("num_frames"),
                expected_size=_expected_size(video))
            result["media_validation"] = descriptor
            if policy["i2v_first_frame_crop"]:
                source_size = decode_png_rgb8(image_bytes)
                behavior = check_i2v_first_frame(
                    source_size[2], source_size[:2], first_frame,
                    (descriptor["width"], descriptor["height"]),
                    policy["first_frame_max_mae"])
                result["first_frame_crop"] = behavior
                if not behavior["passed"]:
                    state = "failed"
                    error = "I2V first frame failed center-crop/anchor behavior gate"
                    result["terminal_state"] = state
                    result["error"] = error
        except (OSError, ValueError, KeyError) as exc:
            result["terminal_state"] = "failed"
            result["error"] = f"I2V media validation failed: {exc}"
    return result


def _run_animations(experiment, record, recorder, *, base_url, timeout_s,
                    replay_case=None):
    slots = [slot for slot in record["visual_slots"] if slot["media_type"] == "image"]
    replay_by_slot = {}
    if replay_case is not None:
        rows = replay_case.get("renders", [])
        if not isinstance(rows, list):
            raise ValueError("replay renders must be a list")
        for row in rows:
            slot_id = row.get("source_slot_id")
            if slot_id in replay_by_slot:
                raise ValueError("duplicate replay I2V source slot")
            replay_by_slot[slot_id] = row
        if set(replay_by_slot) != {slot["slot_id"] for slot in slots}:
            raise ValueError("replay I2V renders do not match source image slots")
    return [_run_one_animation(
        experiment, record, slot, recorder, base_url=base_url, timeout_s=timeout_s,
        replay_case=replay_by_slot.get(slot["slot_id"])) for slot in slots]


def _session_id(experiment, variant, prompt_id):
    return f"{experiment.config['ablation_id']}:{variant}:{prompt_id}"


def _finalize_trajectory(path, experiment, variant, animation_rows=None, entries=None):
    records = []
    prompts = experiment.prompts
    if entries is None:
        entries = [{"prompt": prompt, "repetition": None} for prompt in prompts]
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line:
            records.append(json.loads(line))
    if len(records) != len(entries):
        raise ValueError("trajectory record count differs from prompt count")
    if animation_rows is None:
        animation_rows = [[] for _ in entries]
    if len(animation_rows) != len(entries):
        raise ValueError("animation row count differs from prompt count")
    variable = _variable(experiment)
    for record, entry, animations in zip(records, entries, animation_rows):
        prompt = entry["prompt"]
        metadata = {"schema_version": 1,
                    "ablation_id": experiment.config["ablation_id"],
                    variable: variant, "prompt_id": prompt["id"]}
        if entry["repetition"] is not None:
            metadata["repetition"] = entry["repetition"]
        if "expected_visual_modalities" in prompt:
            metadata["expected_visual_modalities"] = copy.deepcopy(
                prompt["expected_visual_modalities"])
        if variable == "visual_form":
            metadata["animations"] = [
                {key: copy.deepcopy(value) for key, value in animation.items()
                 if key not in ("latency", "media_validation", "first_frame_crop")}
                for animation in animations]
        record["ablation"] = metadata
    Path(path).write_bytes(b"".join(_canonical(record) + b"\n" for record in records))
    if variable == "routing_policy":
        reward_mod.apply_reward(path, routing_metrics.NarrativeRoutingReward())
        records = [json.loads(line) for line in Path(path).read_text(
            encoding="utf-8").splitlines() if line]
        Path(path).write_bytes(b"".join(_canonical(record) + b"\n" for record in records))
    return worker.validate_trajectory(path)


def _latency_coverage(value):
    return "COVERED" if value is not None else "NOT_COVERED"


def _case_report(experiment, variant, prompt, body, record, latency, animations=None,
                 media_validation=None, repetition=None, schedule_index=None,
                 request_telemetry=None):
    variable = _variable(experiment)
    observation = routing_metrics.routing_observation(record) \
        if variable == "routing_policy" else None
    form_observation = routing_metrics.visual_form_observation(record) \
        if variable == "visual_form" else None
    slots = latency["slots"]
    settled = [slot["slot_to_artifact_ms"] for slot in slots
               if slot["slot_to_artifact_ms"] is not None]
    if not slots and record["terminal_state"] == "completed":
        slot_coverage = "NOT_APPLICABLE"
    elif len(settled) == len(slots) and slots:
        slot_coverage = "COVERED"
    elif settled:
        slot_coverage = "PARTIAL"
    else:
        slot_coverage = "NOT_COVERED"
    if animations is None:
        animations = record["ablation"].get("animations", [])
    animations = copy.deepcopy(animations)
    result = {
        variable: variant, "prompt_id": prompt["id"],
        "request_sha256": _sha256(body),
        "fixed_request_sha256": fixed_request_sha256(body),
        "terminal_state": record["terminal_state"], "error": record["error"],
        "terminal_legs": copy.deepcopy(record["terminal_legs"]),
        "leg_errors": copy.deepcopy(record["leg_errors"]),
        "latency": latency,
        "animations": animations,
        "reward": copy.deepcopy(record.get("reward")),
        "output_sha256": record["output_sha256"],
        "clean_text_sha256": record["clean_text_sha256"],
        "artifacts": copy.deepcopy(record["artifacts"]),
        "media_validation": copy.deepcopy(media_validation or []),
        "metric_coverage": {
            "ttft": _latency_coverage(latency["ttft_ms"]),
            "ttfa": _latency_coverage(latency["ttfa_ms"]),
            "terminal_latency": _latency_coverage(latency["terminal_ms"]),
            "slot_latency": slot_coverage,
            "media_decode": "COVERED" if _validation_policy(experiment)["decode_media"]
                else "NOT_COVERED",
        },
    }
    if request_telemetry is not None:
        result["request_scoped_telemetry"] = copy.deepcopy(request_telemetry)
        result["metric_coverage"]["request_native_allocator"] = \
            request_telemetry["coverage"]
    if repetition is not None:
        result["phase"] = "measured"
        result["repetition"] = repetition
        result["schedule_index"] = schedule_index
    if "audio" not in record["request_params"].get("modalities", []):
        result["metric_coverage"]["ttfa"] = "NOT_APPLICABLE"
    if observation is not None:
        result["routing"] = observation
        result["metric_coverage"]["routing_f1"] = observation["coverage"]
        result["metric_coverage"]["redundant_visual_slot_rate"] = observation["coverage"]
    elif variable == "visual_form":
        result["visual_form_metrics"] = form_observation
        result["metric_coverage"]["visual_form_completion"] = "COVERED"
        result["metric_coverage"]["animation_success"] = (
            "COVERED" if form_observation["expected_animation_count"]
            else "NOT_APPLICABLE")
        result["metric_coverage"]["i2v_first_frame_crop"] = (
            "COVERED" if variant == "image_then_i2v_animate"
            and _validation_policy(experiment)["i2v_first_frame_crop"]
            else "NOT_APPLICABLE" if variant != "image_then_i2v_animate"
            else "NOT_COVERED")
        result["metric_coverage"]["quality"] = "NOT_COVERED"
    return result


def _case_success(case):
    media_ok = all(row.get("status") in ("passed", "NOT_COVERED", "NOT_APPLICABLE")
                   for row in case.get("media_validation", []))
    terminal_ok = case["terminal_state"] == "completed" \
        and case.get("latency", {}).get("terminal_ms") is not None \
        and not case.get("leg_errors") \
        and all(leg.get("state") == "completed"
                for leg in case.get("terminal_legs", []))
    animations_ok = all(animation["terminal_state"] == "completed"
                        for animation in case.get("animations", []))
    if "visual_form_metrics" in case:
        return terminal_ok and case["visual_form_metrics"]["completion"] == 1.0 \
            and media_ok and animations_ok
    return terminal_ok and media_ok and animations_ok


def _summary(values):
    present = [value for value in values if value is not None]
    coverage = "NOT_COVERED"
    if len(present) == len(values) and values:
        coverage = "COVERED"
    elif present:
        coverage = "PARTIAL"
    result = {"coverage": coverage, "sample_count": len(present),
              "total_count": len(values), "median_ms": None,
              "p95_ms": None, "min_ms": None, "max_ms": None}
    if present:
        ordered = sorted(present)
        result.update({
            "median_ms": statistics.median(ordered),
            "p95_ms": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
            "min_ms": ordered[0], "max_ms": ordered[-1],
        })
    return result


def _default_evidence(mode):
    return controlled.CONTRACT_REPLAY if mode == "replay" \
        else controlled.UNCONTROLLED_LIVE


def _claims(mode, evidence_class=None):
    return {
        "learned_routing": "NOT_RUN",
        "optimizer": "NOT_RUN",
        "quality_improvement": "NOT_COVERED",
        "real_model_quality": "NOT_RUN" if mode == "replay" else "NOT_MEASURED",
        "latency_evidence": evidence_class or _default_evidence(mode),
    }


def _variant_name(case):
    return case.get("routing_policy", case.get("visual_form"))


def _artifact_entries(case_reports):
    entries = []
    for case in case_reports:
        for artifact in case["artifacts"]:
            entries.append({"variant": _variant_name(case),
                            "prompt_id": case["prompt_id"],
                            "artifact": artifact})
        for animation in case.get("animations", []):
            if animation.get("artifact") is not None:
                entries.append({"variant": _variant_name(case),
                                "prompt_id": case["prompt_id"],
                                "source_slot_id": animation["source_slot_id"],
                                "artifact": animation["artifact"]})
    return entries


def _provenance(experiment, source, artifact_entries):
    result = {
        "source_commit": source["commit"],
        "source_tree_dirty": source["dirty"],
        "source_state_sha256": source["state_sha256"],
        "config_sha256": experiment.config_sha256,
        "data_sha256": experiment.data_sha256,
        "replay_data_sha256": experiment.replay_sha256,
        "artifact_manifest_sha256": _sha256(artifact_entries),
        "model": _model_provenance(experiment.config["model"]),
        "trajectory_schema_version": 2,
    }
    backends = experiment.config.get("visual_backends", {})
    if backends:
        result["visual_backends"] = {
            name: _model_provenance(backend) for name, backend in backends.items()}
    auxiliary = experiment.config.get("auxiliary_backends", {})
    if auxiliary:
        result["auxiliary_backends"] = {
            name: _model_provenance(backend) for name, backend in auxiliary.items()}
    return result


def _metric_view(cases, records, variable="routing_policy"):
    slot_values = [slot["slot_to_artifact_ms"] for case in cases
                   for slot in case["latency"]["slots"]]
    ttft = _summary([case["latency"]["ttft_ms"] for case in cases])
    ttfa = _summary([case["latency"]["ttfa_ms"] for case in cases])
    terminal = _summary([case["latency"]["terminal_ms"] for case in cases])
    animation_terminal = _summary([
        animation["latency"]["terminal_ms"] for case in cases
        for animation in case.get("animations", [])])
    slots = _summary(slot_values)
    if not slot_values and all(not case["latency"]["slots"]
                               and case["terminal_state"] == "completed"
                               for case in cases):
        slots["coverage"] = "NOT_APPLICABLE"
    media_rows = [row for case in cases for row in case.get("media_validation", [])]
    animation_media = [animation["media_validation"] for case in cases
                       for animation in case.get("animations", [])
                       if "media_validation" in animation]
    video_frames = [{"artifact_id": row.get("artifact_id"),
                     "frame_count": row["frame_count"]}
                    for row in media_rows + animation_media if "frame_count" in row
                    and row.get("media_type", "video") == "video"]
    first_frame_rows = [animation["first_frame_crop"] for case in cases
                        for animation in case.get("animations", [])
                        if "first_frame_crop" in animation]
    metrics = {"ttft_ms": ttft, "ttfa_ms": ttfa, "terminal_ms": terminal,
               "animation_terminal_ms": animation_terminal,
               "slot_to_artifact_ms": slots,
               "decoded_video_frame_counts": video_frames,
               "media_validation_failure_count": sum(
                   row.get("status") == "failed" for row in media_rows),
               "i2v_first_frame_behavior": {
                   "coverage": "COVERED" if first_frame_rows else "NOT_COVERED",
                   "passed_count": sum(row["passed"] for row in first_frame_rows),
                   "total_count": len(first_frame_rows),
                   "cases": first_frame_rows,
               }}
    coverage = {"ttft": ttft["coverage"], "ttfa": ttfa["coverage"],
                "terminal_latency": terminal["coverage"],
                "slot_latency": slots["coverage"], "quality": "NOT_COVERED"}
    media_states = {case.get("metric_coverage", {}).get("media_decode", "NOT_COVERED")
                    for case in cases}
    coverage["media_decode"] = "COVERED" if media_states == {"COVERED"} \
        else "PARTIAL" if "COVERED" in media_states else "NOT_COVERED"
    coverage["i2v_first_frame_crop"] = "COVERED" if first_frame_rows \
        and all(row["passed"] for row in first_frame_rows) else \
        "FAILED" if first_frame_rows else "NOT_COVERED"
    if variable == "visual_form":
        aggregate = routing_metrics.aggregate_visual_form(records)
        metrics["visual_form_counts"] = aggregate
        metrics["visual_form_completion_rate"] = aggregate["visual_form_completion_rate"]
        metrics["animation_success_rate"] = aggregate["animation_success_rate"]
        metrics["image_text_relevance"] = None
        psnr_values = [row["first_frame_psnr_db"] for row in first_frame_rows
                       if row["first_frame_psnr_db"] is not None]
        metrics["first_frame_consistency"] = None if not first_frame_rows else {
            "metric_class": "algorithmic-center-crop-and-anchor-proxy",
            "passed": all(row["passed"] for row in first_frame_rows),
            "center_crop_mae": {
                "min": min(row["center_crop_mae"] for row in first_frame_rows),
                "mean": statistics.mean(row["center_crop_mae"] for row in first_frame_rows),
                "max": max(row["center_crop_mae"] for row in first_frame_rows)},
            "first_frame_psnr_db": None if not psnr_values else {
                "min_db": min(psnr_values), "max_db": max(psnr_values)},
            "quality_coverage": "NOT_COVERED",
        }
        metrics["user_blind_preference"] = None
        coverage["visual_form_completion"] = aggregate["coverage"]
        coverage["animation_success"] = aggregate["animation_coverage"]
        if not first_frame_rows and cases and all(
                case.get("visual_form") != "image_then_i2v_animate" for case in cases):
            coverage["i2v_first_frame_crop"] = "NOT_APPLICABLE"
        if not cases:
            coverage["ttfa"] = "NOT_COVERED"
        else:
            coverage["ttfa"] = "NOT_APPLICABLE"
    elif variable == "routing_policy":
        aggregate = routing_metrics.aggregate_routing(records)
        metrics["routing_f1"] = aggregate["f1"]
        metrics["redundant_visual_slot_rate"] = aggregate["redundant_slot_rate"]
        metrics["routing_counts"] = aggregate
        coverage["routing_f1"] = aggregate["coverage"]
        coverage["redundant_visual_slot_rate"] = aggregate["coverage"]
        coverage["routing_reward"] = aggregate["coverage"]
    if records and not any(
            "audio" in record["request_params"].get("modalities", [])
            for record in records):
        coverage["ttfa"] = "NOT_APPLICABLE"
    return metrics, coverage


def _policy_report(experiment, source, mode, variant, cases, records,
                   trajectory_path, output_root, resources=None,
                   evidence_class=None, measurement=None):
    entries = _artifact_entries(cases)
    variable = _variable(experiment)
    metrics, metric_coverage = _metric_view(cases, records, variable)
    leg_state_counts = {}
    for case in cases:
        for leg in case["terminal_legs"]:
            state = leg.get("state", "unknown")
            leg_state_counts[state] = leg_state_counts.get(state, 0) + 1
    evidence = evidence_class or _default_evidence(mode)
    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "ablation_id": experiment.config["ablation_id"],
        "mode": mode,
        "evidence_class": evidence,
        variable: variant,
        "single_variable": fixed_matrix(experiment),
        "cases": cases,
        "metrics": metrics,
        "metric_coverage": metric_coverage,
        "terminal_state_counts": {
            state: sum(case["terminal_state"] == state for case in cases)
            for state in ("completed", "failed", "cancelled")},
        "terminal_leg_state_counts": leg_state_counts,
        "animation_terminal_state_counts": {
            state: sum(animation.get("terminal_state") == state for case in cases
                       for animation in case.get("animations", []))
            for state in ("completed", "failed", "cancelled")},
        "resources": resources or _uncovered_resources(),
        "trajectory_file": str(Path(trajectory_path).relative_to(output_root).as_posix()),
        "trajectory_sha256": hashlib.sha256(Path(trajectory_path).read_bytes()).hexdigest(),
        "claims": _claims(mode, evidence),
    }
    if measurement is not None:
        report["measurement"] = measurement
    report.update(_provenance(experiment, source, entries))
    return report


def _replay_measurement(experiment, contract):
    variants = _variants(experiment)
    steps = controlled.build_schedule(variants, contract)
    return {
        "schema_version": controlled.MEASUREMENT_SCHEMA_VERSION,
        "status": "NOT_RUN",
        "reason": "CONTRACT_REPLAY validates the declared schedule only; "
                  "controlled live measurement requires a separate exclusive live gate",
        "contract": contract,
        "schedule": controlled.schedule_descriptor(variants, contract, steps),
        "request_scoped_telemetry": controlled.request_scoped_telemetry(
            [] if contract.get("request_memory", {}).get("enabled") else None),
    }


def _controlled_aggregates(cases):
    """Per-prompt repetition statistics plus pooled per-arm dispersion."""
    per_prompt = {}
    for case in cases:
        per_prompt.setdefault(case["prompt_id"], []).append(case)
    prompts = {}
    for prompt_id, rows in sorted(per_prompt.items()):
        prompts[prompt_id] = {
            "repetitions": len(rows),
            "ttft_ms": controlled.summarize(
                [row["latency"]["ttft_ms"] for row in rows]),
            "ttfa_ms": controlled.summarize(
                [row["latency"]["ttfa_ms"] for row in rows]),
            "terminal_ms": controlled.summarize(
                [row["latency"]["terminal_ms"] for row in rows]),
        }
    return {
        "per_prompt": prompts,
        "pooled": {
            "ttft_ms": controlled.summarize(
                [case["latency"]["ttft_ms"] for case in cases]),
            "ttfa_ms": controlled.summarize(
                [case["latency"]["ttfa_ms"] for case in cases]),
            "terminal_ms": controlled.summarize(
                [case["latency"]["terminal_ms"] for case in cases]),
            "slot_to_artifact_ms": controlled.summarize(
                [slot["slot_to_artifact_ms"] for case in cases
                 for slot in case["latency"]["slots"]]),
        },
    }


def _arm_resources(variant, step_resources, gpu_indexes):
    rows = [row for row in step_resources if row["arm"] == variant]
    if not rows:
        return {"coverage": "NOT_COVERED", "scope": "whole-device-used-memory",
                "reason": "no declared whole-device GPU sampler for controlled steps",
                "gpus": {}}
    gpus = {}
    for index in gpu_indexes:
        key = str(index)
        samples = [row["whole_device_gpus"][key] for row in rows]
        baseline = samples[0]["baseline_mib"]
        gpus[key] = {"baseline_mib": baseline,
                     "peak_mib": max(sample["peak_mib"] for sample in samples),
                     "final_mib": samples[-1]["final_mib"],
                     "sample_count": sum(sample["sample_count"] for sample in samples)}
    return {
        "coverage": "COVERED", "scope": "whole-device-used-memory",
        "source": "nvidia-smi-poll",
        "poll_interval_ms": rows[0]["poll_interval_ms"],
        "attribution": "per-step whole-device samples include persistent model "
                       "baseline and other interleaved arms; not request-only allocation",
        "gpus": gpus,
    }


def _first_stage_image_equivalence(all_cases, variable):
    """Pair every I2V repetition with image-only and require exact source bytes."""
    if variable != "visual_form":
        return {"coverage": "NOT_APPLICABLE", "passed": True, "cases": []}
    rows = []
    for i2v_case in all_cases:
        if i2v_case["visual_form"] != "image_then_i2v_animate":
            continue
        matches = [case for case in all_cases
                   if case["prompt_id"] == i2v_case["prompt_id"]
                   and case["visual_form"] == "image_only"
                   and case.get("repetition") == i2v_case.get("repetition")]
        image_case = matches[0] if len(matches) == 1 else None
        image_hashes = sorted(
            artifact["sha256"] for artifact in (image_case or {"artifacts": []})["artifacts"]
            if artifact.get("media_type") == "image")
        i2v_hashes = sorted(artifact["sha256"] for artifact in i2v_case["artifacts"]
                            if artifact.get("media_type") == "image")
        row = {
            "prompt_id": i2v_case["prompt_id"],
            "paired_image_case_count": len(matches),
            "request_sha256": image_case["request_sha256"] if image_case else None,
            "request_sha256_equal": image_case is not None and
                image_case["request_sha256"] == i2v_case["request_sha256"],
            "image_only_source_sha256": image_hashes,
            "i2v_source_sha256": i2v_hashes,
            "source_bytes_identical": bool(image_hashes) and image_hashes == i2v_hashes,
        }
        if "repetition" in i2v_case:
            row["repetition"] = i2v_case["repetition"]
        rows.append(row)
    passed = bool(rows) and all(
        row["paired_image_case_count"] == 1 and row["request_sha256_equal"]
        and row["source_bytes_identical"] for row in rows)
    return {"coverage": "COVERED", "passed": passed, "cases": rows}


def _run_controlled_experiment(experiment, output_root, source, normalized_url, *,
                               timeout_s, contract, resource_gpu_indexes,
                               gpu_sampler, compute_apps_sampler, health_probe,
                               sleep_fn, require_controlled):
    """Execute the declared measurement contract against a live endpoint.

    The CONTROLLED_LIVE label is an eligibility decision, never a flag: any
    missing repetition, exclusivity, hash, or completion coverage downgrades
    the whole run to UNCONTROLLED_LIVE with explicit reasons.
    """
    request_memory = contract.get("request_memory", {})
    telemetry_enabled = request_memory.get("enabled", False)
    mode = "live"
    variable = _variable(experiment)
    variants = _variants(experiment)
    steps = controlled.build_schedule(variants, contract)
    schedule = controlled.schedule_descriptor(variants, contract, steps)
    sleep = sleep_fn or time.sleep
    exclusive = contract["exclusive_device"]
    gpu_indexes = sorted(set(exclusive["gpu_indexes"]) | set(resource_gpu_indexes))
    sampler = gpu_sampler or _query_nvidia_smi
    baseline = sampler(gpu_indexes) if gpu_indexes else None

    preconditions = []
    if exclusive["required"]:
        apps = compute_apps_sampler or controlled.query_compute_apps
        record = controlled.check_exclusive_device(contract, apps, baseline)
        preconditions.append(record)
        if require_controlled and not record["passed"]:
            raise ValueError("controlled precondition failed before any request: "
                             + "; ".join(record["problems"]))

    def default_health():
        conn = http.client.HTTPConnection(normalized_url, timeout=timeout_s)
        try:
            conn.request("GET", "/health")
            response = conn.getresponse()
            return json.loads(response.read().decode("utf-8"))
        finally:
            conn.close()

    health = health_probe or default_health
    recorders = {variant: traj.TrajectoryRecorder(
        output_root / variant / "trajectory.jsonl") for variant in variants}
    warmup_recorders = {}
    pending = {variant: [] for variant in variants}
    animation_rows = {variant: [] for variant in variants}
    entries = {variant: [] for variant in variants}
    arm_coverage = {
        variant: {"warmups_declared": contract["warmup_repetitions"],
                  "warmups_executed": 0, "warmup_failures": 0,
                  "measured_declared": contract["measured_repetitions"],
                  "measured_completed": 0, "measured_failed": 0}
        for variant in variants}
    warmups = []
    cleanups = []
    step_resources = []

    for step in steps:
        variant = step["arm"]
        coverage = arm_coverage[variant]
        is_warmup = step["phase"] == "warmup"
        if is_warmup and variant not in warmup_recorders:
            warmup_recorders[variant] = traj.TrajectoryRecorder(
                output_root / "warmup" / variant / "trajectory.jsonl")
        recorder = warmup_recorders[variant] if is_warmup else recorders[variant]
        monitor = GpuResourceMonitor(gpu_indexes, sampler=sampler).start() \
            if gpu_indexes else None
        warmup_cases = []
        for prompt in experiment.prompts:
            body = build_request(experiment, prompt, variant)
            session_id = (f"{experiment.config['ablation_id']}:{variant}:"
                          f"{step['phase']}:r{step['repetition']}:{prompt['id']}")
            tracker = LatencyTracker()
            telemetry_events = []

            def observe(event):
                tracker.observe(event)
                if telemetry_enabled and isinstance(event, dict) \
                        and event.get("telemetry") is not None:
                    telemetry_events.append(event["telemetry"])

            record = traj.record_chat_completion(
                normalized_url, body, recorder, session_id,
                timeout=timeout_s, event_observer=observe)
            if telemetry_enabled:
                if len(telemetry_events) > 1:
                    raise ValueError("request emitted more than one telemetry envelope")
                request_telemetry = controlled.request_scoped_telemetry(
                    telemetry_events[0] if telemetry_events else None,
                    request_id=record["request_id"],
                    terminal_state=record["terminal_state"])
            else:
                request_telemetry = None
            latency = normalize_latency(tracker.finish(), record, "monotonic-live")
            animations = []
            if variable == "visual_form" and variant == "image_then_i2v_animate":
                animations = _run_animations(
                    experiment, record, recorder, base_url=normalized_url,
                    timeout_s=timeout_s)
            media_validation = _validate_record_media(
                experiment, record, recorder.artifact_dir)
            if is_warmup:
                warmup_case = {
                    "terminal_state": record["terminal_state"],
                    "latency": latency,
                    "terminal_legs": record["terminal_legs"],
                    "leg_errors": record["leg_errors"],
                    "media_validation": media_validation,
                    "animations": animations,
                    "request_scoped_telemetry": request_telemetry,
                }
                warmup_cases.append({
                    "prompt_id": prompt["id"],
                    "terminal_state": record["terminal_state"],
                    "error": record["error"],
                    "terminal_legs": copy.deepcopy(record["terminal_legs"]),
                    "leg_errors": copy.deepcopy(record["leg_errors"]),
                    "terminal_ms": latency["terminal_ms"],
                })
                if not _case_success(warmup_case):
                    coverage["warmup_failures"] += 1
            else:
                pending[variant].append((prompt, body, record, latency,
                                         media_validation, step["repetition"],
                                         step["index"], request_telemetry))
                animation_rows[variant].append(animations)
                entries[variant].append({"prompt": prompt,
                                         "repetition": step["repetition"]})
        if is_warmup:
            coverage["warmups_executed"] += 1
            warmups.append({"schedule_index": step["index"], "arm": variant,
                            "repetition": step["repetition"], "cases": warmup_cases})
        else:
            coverage["measured_completed"] += 1

        deadline = time.monotonic() + contract["cleanup"]["timeout_s"]
        while True:
            in_flight = health().get("in_flight")
            if not contract["cleanup"]["require_in_flight_zero"] or in_flight == 0:
                break
            if time.monotonic() >= deadline:
                break
            sleep(0.05)
        residual = None
        if monitor is not None:
            observed = monitor.stop()
            final = {index: observed["gpus"][str(index)]["final_mib"]
                     for index in gpu_indexes}
            step_resources.append({
                "schedule_index": step["index"], "arm": variant,
                "phase": step["phase"], "repetition": step["repetition"],
                "poll_interval_ms": observed["poll_interval_ms"],
                "whole_device_gpus": {
                    str(index): {"baseline_mib": baseline[index],
                                 "peak_mib": observed["gpus"][str(index)]["peak_mib"],
                                 "final_mib": final[index],
                                 "sample_count":
                                     observed["gpus"][str(index)]["sample_count"]}
                    for index in gpu_indexes},
            })
            residual = {index: final[index] - baseline[index]
                        for index in gpu_indexes}
        verdict = controlled.evaluate_cleanup(
            contract, in_flight=in_flight, residual_mib=residual)
        verdict["schedule_index"] = step["index"]
        cleanups.append(verdict)
        if contract["cooldown_ms"]:
            sleep(contract["cooldown_ms"] / 1000.0)

    variant_data = {}
    for variant in variants:
        trajectory_path = output_root / variant / "trajectory.jsonl"
        records = _finalize_trajectory(
            trajectory_path, experiment, variant, animation_rows[variant],
            entries[variant])
        cases = [
            _case_report(experiment, variant, prompt, body, record, latency,
                         animations, media_validation, repetition=repetition,
                         schedule_index=schedule_index,
                         request_telemetry=request_telemetry)
            for (prompt, body, _, latency, media_validation, repetition,
                 schedule_index, request_telemetry), record, animations
            in zip(pending[variant], records, animation_rows[variant])]
        arm_coverage[variant]["measured_failed"] = sum(
            not _case_success(case) for case in cases)
        variant_data[variant] = (cases, records, trajectory_path)

    all_cases = [case for variant in variants for case in variant_data[variant][0]]
    provenance = _provenance(experiment, source, _artifact_entries(all_cases))
    equivalence = _first_stage_image_equivalence(all_cases, variable) \
        if variable == "visual_form" else None
    request_telemetry = controlled.request_scoped_telemetry([
        case["request_scoped_telemetry"] for case in all_cases]) \
        if telemetry_enabled else None
    eligibility = controlled.evaluate_eligibility(
        contract, provenance=provenance, model=provenance["model"],
        arms=arm_coverage, preconditions=preconditions, cleanups=cleanups,
        visual_form_equivalence=equivalence, request_telemetry=request_telemetry)
    evidence_class = eligibility["label"]

    policy_reports = []
    all_records = []
    for variant in variants:
        cases, records, trajectory_path = variant_data[variant]
        measurement = dict(arm_coverage[variant])
        measurement.update({
            "schedule_sha256": schedule["schedule_sha256"],
            "aggregates": _controlled_aggregates(cases),
        })
        if telemetry_enabled:
            measurement["request_scoped_telemetry"] = \
                controlled.request_scoped_telemetry([
                    case["request_scoped_telemetry"] for case in cases])
        report = _policy_report(
            experiment, source, mode, variant, cases, records, trajectory_path,
            output_root,
            resources=_arm_resources(variant, step_resources, gpu_indexes),
            evidence_class=evidence_class, measurement=measurement)
        _write_json(output_root / variant / "report.json", report)
        policy_reports.append(report)
        all_records.extend(records)

    measurement = {
        "schema_version": controlled.MEASUREMENT_SCHEMA_VERSION,
        "status": "EXECUTED",
        "contract": contract,
        "schedule": schedule,
        "preconditions": preconditions,
        "cleanup_checks": cleanups,
        "arms": {variant: dict(arm_coverage[variant],
                               aggregates=_controlled_aggregates(
                                   variant_data[variant][0]))
                 for variant in variants},
        "warmups": warmups,
        "resources": {
            "coverage": "COVERED" if step_resources else "NOT_COVERED",
            "scope": "whole-device-used-memory",
            "source": "nvidia-smi-poll" if step_resources else "NOT_COVERED",
            "attribution": "per-step whole-device samples; interleaved arms share "
                           "devices and persistent model baselines",
            "steps": step_resources,
        },
        "request_scoped_telemetry": request_telemetry
        if telemetry_enabled else controlled.request_scoped_telemetry(),
        "eligibility": eligibility,
    }
    return _build_combined(experiment, source, mode, evidence_class, variants,
                           policy_reports, all_cases, all_records, output_root,
                           measurement=measurement)


def run_experiment(experiment, output_root, *, replay_only, base_url=None,
                   timeout_s=120.0, resource_gpu_indexes=(), gpu_sampler=None,
                   require_controlled=False, compute_apps_sampler=None,
                   health_probe=None, sleep_fn=None):
    output_root = _prepare_output(output_root)
    source = _source_provenance()
    mode = "replay" if replay_only else "live"
    contract = controlled.parse_contract(experiment.config.get("measurement"))
    if not replay_only and base_url is None:
        raise ValueError("live mode requires --base-url")
    normalized_url = _normalize_base_url(base_url) if base_url else None
    if require_controlled and (replay_only or contract is None):
        raise ValueError("--require-controlled needs live mode and a measurement block")
    if not replay_only and contract is not None:
        return _run_controlled_experiment(
            experiment, output_root, source, normalized_url,
            timeout_s=timeout_s, contract=contract,
            resource_gpu_indexes=resource_gpu_indexes, gpu_sampler=gpu_sampler,
            compute_apps_sampler=compute_apps_sampler, health_probe=health_probe,
            sleep_fn=sleep_fn, require_controlled=require_controlled)
    replay_cases = _replay_map(experiment) if replay_only else None
    variable = _variable(experiment)
    variants = _variants(experiment)
    policy_reports = []
    all_cases = []
    all_records = []

    if replay_only and resource_gpu_indexes:
        raise ValueError("GPU sampling belongs to live mode, not deterministic replay")

    for variant in variants:
        policy_dir = output_root / variant
        trajectory_path = policy_dir / "trajectory.jsonl"
        recorder = traj.TrajectoryRecorder(trajectory_path)
        pending = []
        animation_rows = []
        monitor = GpuResourceMonitor(resource_gpu_indexes, sampler=gpu_sampler).start() \
            if resource_gpu_indexes else None
        try:
            for prompt in experiment.prompts:
                body = build_request(experiment, prompt, variant)
                session_id = _session_id(experiment, variant, prompt["id"])
                replay_case = replay_cases[(variant, prompt["id"])] if replay_only else None
                if replay_only:
                    record, latency = _record_replay(
                        replay_case, body, recorder, session_id)
                else:
                    tracker = LatencyTracker()
                    record = traj.record_chat_completion(
                        normalized_url, body, recorder, session_id,
                        timeout=timeout_s, event_observer=tracker.observe)
                    latency = normalize_latency(
                        tracker.finish(), record, "monotonic-live")
                animations = []
                if variable == "visual_form" and variant == "image_then_i2v_animate":
                    animations = _run_animations(
                        experiment, record, recorder, base_url=normalized_url,
                        timeout_s=timeout_s, replay_case=replay_case)
                elif replay_case is not None and replay_case.get("renders"):
                    raise ValueError("non-I2V replay case unexpectedly contains renders")
                media_validation = _validate_record_media(
                    experiment, record, recorder.artifact_dir)
                animation_rows.append(animations)
                pending.append((prompt, body, record, latency, media_validation))
            records = _finalize_trajectory(
                trajectory_path, experiment, variant, animation_rows)
            cases = [_case_report(experiment, variant, prompt, body, record, latency,
                                  animations, media_validation)
                     for (prompt, body, _, latency, media_validation), record, animations
                     in zip(pending, records, animation_rows)]
            resources = monitor.stop() if monitor is not None else _uncovered_resources()
            monitor = None
            report = _policy_report(experiment, source, mode, variant, cases,
                                    records, trajectory_path, output_root, resources)
            _write_json(policy_dir / "report.json", report)
            policy_reports.append(report)
            all_cases.extend(cases)
            all_records.extend(records)
        finally:
            if monitor is not None:
                monitor.stop()

    measurement = _replay_measurement(experiment, contract) \
        if replay_only and contract is not None else None
    return _build_combined(experiment, source, mode, _default_evidence(mode),
                           variants, policy_reports, all_cases, all_records,
                           output_root, measurement=measurement)


def _build_combined(experiment, source, mode, evidence_class, variants,
                    policy_reports, all_cases, all_records, output_root,
                    measurement=None):
    variable = _variable(experiment)
    all_entries = _artifact_entries(all_cases)
    summaries = []
    for report in policy_reports:
        summaries.append({
            variable: report[variable],
            "metrics": report["metrics"],
            "metric_coverage": report["metric_coverage"],
            "terminal_state_counts": report["terminal_state_counts"],
            "terminal_leg_state_counts": report["terminal_leg_state_counts"],
            "animation_terminal_state_counts": report["animation_terminal_state_counts"],
            "resources": report["resources"],
            "trajectory_file": report["trajectory_file"],
            "trajectory_sha256": report["trajectory_sha256"],
            "artifact_manifest_sha256": report["artifact_manifest_sha256"],
        })
    overall_metrics, overall_coverage = _metric_view(all_cases, all_records, variable)
    equivalence = _first_stage_image_equivalence(all_cases, variable)
    equivalence_passed = equivalence["passed"]
    combined = {
        "ablation_id": experiment.config["ablation_id"],
        "mode": mode,
        "evidence_class": evidence_class,
        "run_status": "COMPLETED" if equivalence_passed and all(
            _case_success(case) for case in all_cases) else "FAILED",
        {"routing_policy": "routing_policies", "visual_form": "visual_forms",
         "request_native_backend": "request_native_backends"}[variable]: list(variants),
        "single_variable": fixed_matrix(experiment),
        "policy_summaries": summaries,
        "cases": all_cases,
        "metrics": overall_metrics,
        "metric_coverage": overall_coverage,
        "first_stage_image_equivalence": equivalence,
        "resources": {report[variable]: report["resources"] for report in policy_reports},
        "claims": _claims(mode, evidence_class),
    }
    if measurement is not None:
        combined["measurement"] = measurement
    combined.update(_provenance(experiment, source, all_entries))
    _write_json(output_root / "report.json", combined)
    _write_checksums(output_root)
    return combined


def _write_checksums(output_root):
    rows = []
    for path in sorted(Path(output_root).rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        relative = path.relative_to(output_root).as_posix()
        rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}\n")
    (Path(output_root) / "SHA256SUMS").write_text("".join(rows), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--replay-only", action="store_true")
    parser.add_argument("--base-url")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--gpu-index", type=int, action="append", default=[],
                        help="sample this GPU's whole-device memory in live mode; repeatable")
    parser.add_argument("--require-controlled", action="store_true",
                        help="exit 2 unless a live measurement-contract run earns "
                             "CONTROLLED_LIVE; precondition failures abort before any request")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("--timeout-s must be a finite positive number")
    if any(index < 0 for index in args.gpu_index):
        parser.error("--gpu-index must be non-negative")
    try:
        experiment = load_experiment(args.config)
        report = run_experiment(
            experiment, args.out, replay_only=args.replay_only,
            base_url=args.base_url, timeout_s=args.timeout_s,
            resource_gpu_indexes=args.gpu_index,
            require_controlled=args.require_controlled)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f"ablation error: {exc}\n")
    print(json.dumps({"status": report["run_status"],
                      "evidence_class": report["evidence_class"],
                      "report": str(Path(args.out) / "report.json"),
                      "variable": _variable(experiment),
                      "variants": _variants(experiment)}, sort_keys=True))
    if args.require_controlled \
            and report["evidence_class"] != controlled.CONTROLLED_LIVE:
        reasons = report.get("measurement", {}).get("eligibility", {}).get(
            "reasons", ["measurement contract is absent"])
        parser.exit(2, "CONTROLLED_LIVE refused: " + "; ".join(reasons) + "\n")
    return 0 if report["run_status"] == "COMPLETED" else 1


if __name__ == "__main__":
    sys.exit(main())
