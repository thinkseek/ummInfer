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
import subprocess
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import umminfer_ablation_metrics as routing_metrics  # noqa: E402
import umminfer_rl_reward as reward_mod  # noqa: E402
import umminfer_rl_trajectory as traj  # noqa: E402
import umminfer_rl_worker as worker  # noqa: E402

ABLATION_SCHEMA_VERSION = 1
REPLAY_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
POLICIES = ("text_only", "forced_image", "forced_video", "auto_tag")
VISUAL_FORMS = ("image_only", "video_only", "image_then_i2v_animate")
VARIABLES = ("routing_policy", "visual_form")
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


def _variable(experiment):
    return experiment.config["variable"]


def _variants(experiment):
    key = "routing_policies" if _variable(experiment) == "routing_policy" \
        else "visual_forms"
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
    if "weights_sha256_scope" in model and not isinstance(
            model["weights_sha256_scope"], str):
        raise ValueError("model.weights_sha256_scope must be a string")

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


def _model_provenance(experiment):
    model = experiment.config["model"]
    identity = {"id": model["id"], "revision": model["revision"]}
    result = {
        "id": model["id"], "revision": model["revision"],
        "config_sha256": _sha256(identity),
        "weights_sha256": model.get("weights_sha256"),
        "weights_coverage": "COVERED" if model.get("weights_sha256") else "NOT_COVERED",
    }
    if "weights_sha256_scope" in model:
        result["weights_sha256_scope"] = model["weights_sha256_scope"]
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


def _finalize_trajectory(path, experiment, variant, animation_rows=None):
    records = []
    prompts = experiment.prompts
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line:
            records.append(json.loads(line))
    if len(records) != len(prompts):
        raise ValueError("trajectory record count differs from prompt count")
    if animation_rows is None:
        animation_rows = [[] for _ in prompts]
    if len(animation_rows) != len(prompts):
        raise ValueError("animation row count differs from prompt count")
    variable = _variable(experiment)
    for record, prompt, animations in zip(records, prompts, animation_rows):
        metadata = {"schema_version": 1,
                    "ablation_id": experiment.config["ablation_id"],
                    variable: variant, "prompt_id": prompt["id"]}
        if "expected_visual_modalities" in prompt:
            metadata["expected_visual_modalities"] = copy.deepcopy(
                prompt["expected_visual_modalities"])
        if variable == "visual_form":
            metadata["animations"] = [
                {key: copy.deepcopy(value) for key, value in animation.items()
                 if key != "latency"} for animation in animations]
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


def _case_report(experiment, variant, prompt, body, record, latency, animations=None):
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
        "metric_coverage": {
            "ttft": _latency_coverage(latency["ttft_ms"]),
            "ttfa": _latency_coverage(latency["ttfa_ms"]),
            "terminal_latency": _latency_coverage(latency["terminal_ms"]),
            "slot_latency": slot_coverage,
        },
    }
    if "audio" not in record["request_params"].get("modalities", []):
        result["metric_coverage"]["ttfa"] = "NOT_APPLICABLE"
    if observation is not None:
        result["routing"] = observation
        result["metric_coverage"]["routing_f1"] = observation["coverage"]
        result["metric_coverage"]["redundant_visual_slot_rate"] = observation["coverage"]
    else:
        result["visual_form_metrics"] = form_observation
        result["metric_coverage"]["visual_form_completion"] = "COVERED"
        result["metric_coverage"]["animation_success"] = (
            "COVERED" if form_observation["expected_animation_count"]
            else "NOT_APPLICABLE")
        result["metric_coverage"]["quality"] = "NOT_COVERED"
    return result


def _case_success(case):
    if "visual_form_metrics" in case:
        return case["visual_form_metrics"]["completion"] == 1.0
    return case["terminal_state"] == "completed" and all(
        animation["terminal_state"] == "completed"
        for animation in case.get("animations", []))


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


def _claims(mode):
    return {
        "learned_routing": "NOT_RUN",
        "optimizer": "NOT_RUN",
        "quality_improvement": "NOT_COVERED",
        "real_model_quality": "NOT_RUN" if mode == "replay" else "NOT_MEASURED",
        "latency_evidence": "CONTRACT_OBSERVATIONS_ONLY" if mode == "replay"
            else "UNCONTROLLED_LIVE_MEASUREMENT",
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
    return {
        "source_commit": source["commit"],
        "source_tree_dirty": source["dirty"],
        "source_state_sha256": source["state_sha256"],
        "config_sha256": experiment.config_sha256,
        "data_sha256": experiment.data_sha256,
        "replay_data_sha256": experiment.replay_sha256,
        "artifact_manifest_sha256": _sha256(artifact_entries),
        "model": _model_provenance(experiment),
        "trajectory_schema_version": 2,
    }


def _metric_view(cases, records, variable="routing_policy"):
    slot_values = [slot["slot_to_artifact_ms"] for case in cases
                   for slot in case["latency"]["slots"]]
    ttft = _summary([case["latency"]["ttft_ms"] for case in cases])
    ttfa = _summary([case["latency"]["ttfa_ms"] for case in cases])
    terminal = _summary([case["latency"]["terminal_ms"] for case in cases])
    slots = _summary(slot_values)
    if not slot_values and all(not case["latency"]["slots"]
                               and case["terminal_state"] == "completed"
                               for case in cases):
        slots["coverage"] = "NOT_APPLICABLE"
    metrics = {"ttft_ms": ttft, "ttfa_ms": ttfa, "terminal_ms": terminal,
               "slot_to_artifact_ms": slots}
    coverage = {"ttft": ttft["coverage"], "ttfa": ttfa["coverage"],
                "terminal_latency": terminal["coverage"],
                "slot_latency": slots["coverage"], "quality": "NOT_COVERED"}
    if variable == "visual_form":
        aggregate = routing_metrics.aggregate_visual_form(records)
        metrics["visual_form_counts"] = aggregate
        metrics["visual_form_completion_rate"] = aggregate["visual_form_completion_rate"]
        metrics["animation_success_rate"] = aggregate["animation_success_rate"]
        metrics["image_text_relevance"] = None
        metrics["first_frame_consistency"] = None
        metrics["user_blind_preference"] = None
        coverage["visual_form_completion"] = aggregate["coverage"]
        coverage["animation_success"] = aggregate["animation_coverage"]
        if not cases:
            coverage["ttfa"] = "NOT_COVERED"
        else:
            coverage["ttfa"] = "NOT_APPLICABLE"
    else:
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
                   trajectory_path, output_root):
    entries = _artifact_entries(cases)
    variable = _variable(experiment)
    metrics, metric_coverage = _metric_view(cases, records, variable)
    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "ablation_id": experiment.config["ablation_id"],
        "mode": mode,
        "evidence_class": "contract-replay" if mode == "replay" else "live-uncontrolled",
        variable: variant,
        "single_variable": fixed_matrix(experiment),
        "cases": cases,
        "metrics": metrics,
        "metric_coverage": metric_coverage,
        "terminal_state_counts": {
            state: sum(case["terminal_state"] == state for case in cases)
            for state in ("completed", "failed", "cancelled")},
        "trajectory_file": str(Path(trajectory_path).relative_to(output_root).as_posix()),
        "trajectory_sha256": hashlib.sha256(Path(trajectory_path).read_bytes()).hexdigest(),
        "claims": _claims(mode),
    }
    report.update(_provenance(experiment, source, entries))
    return report


def run_experiment(experiment, output_root, *, replay_only, base_url=None,
                   timeout_s=120.0):
    output_root = _prepare_output(output_root)
    source = _source_provenance()
    mode = "replay" if replay_only else "live"
    replay_cases = _replay_map(experiment) if replay_only else None
    if not replay_only and base_url is None:
        raise ValueError("live mode requires --base-url")
    normalized_url = _normalize_base_url(base_url) if base_url else None
    variable = _variable(experiment)
    variants = _variants(experiment)
    policy_reports = []
    all_cases = []
    all_records = []

    for variant in variants:
        policy_dir = output_root / variant
        trajectory_path = policy_dir / "trajectory.jsonl"
        recorder = traj.TrajectoryRecorder(trajectory_path)
        pending = []
        animation_rows = []
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
            animation_rows.append(animations)
            pending.append((prompt, body, record, latency))
        records = _finalize_trajectory(
            trajectory_path, experiment, variant, animation_rows)
        cases = [_case_report(experiment, variant, prompt, body, record, latency, animations)
                 for (prompt, body, _, latency), record, animations
                 in zip(pending, records, animation_rows)]
        report = _policy_report(experiment, source, mode, variant, cases,
                                records, trajectory_path, output_root)
        _write_json(policy_dir / "report.json", report)
        policy_reports.append(report)
        all_cases.extend(cases)
        all_records.extend(records)

    all_entries = _artifact_entries(all_cases)
    summaries = []
    for report in policy_reports:
        summaries.append({
            variable: report[variable],
            "metrics": report["metrics"],
            "metric_coverage": report["metric_coverage"],
            "terminal_state_counts": report["terminal_state_counts"],
            "trajectory_file": report["trajectory_file"],
            "trajectory_sha256": report["trajectory_sha256"],
            "artifact_manifest_sha256": report["artifact_manifest_sha256"],
        })
    overall_metrics, overall_coverage = _metric_view(all_cases, all_records, variable)
    combined = {
        "ablation_id": experiment.config["ablation_id"],
        "mode": mode,
        "evidence_class": "contract-replay" if mode == "replay" else "live-uncontrolled",
        "run_status": "COMPLETED" if all(
            _case_success(case) for case in all_cases) else "FAILED",
        "routing_policies" if variable == "routing_policy" else "visual_forms":
            list(variants),
        "single_variable": fixed_matrix(experiment),
        "policy_summaries": summaries,
        "cases": all_cases,
        "metrics": overall_metrics,
        "metric_coverage": overall_coverage,
        "claims": _claims(mode),
    }
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
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("--timeout-s must be a finite positive number")
    try:
        experiment = load_experiment(args.config)
        report = run_experiment(
            experiment, args.out, replay_only=args.replay_only,
            base_url=args.base_url, timeout_s=args.timeout_s)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f"ablation error: {exc}\n")
    print(json.dumps({"status": report["run_status"],
                      "report": str(Path(args.out) / "report.json"),
                      "variable": _variable(experiment),
                      "variants": _variants(experiment)}, sort_keys=True))
    return 0 if report["run_status"] == "COMPLETED" else 1


if __name__ == "__main__":
    sys.exit(main())
