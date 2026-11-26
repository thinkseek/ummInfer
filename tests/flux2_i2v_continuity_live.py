#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Direct native FLUX.2 -> Wan2.2 I2V continuity gate on physical GPU 1.

CPU-only self-check:
  python3 -I tests/flux2_i2v_continuity_live.py --self-check

Real generation is opt-in and launches only task-owned Stage processes. FLUX is
joined and GPU 1 is checked idle before native Wan loads. No VLM, TTS, realtime,
resident service, Qwen-Image, mock artifact, or public schema is involved.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from narrative_slot_reference_live import (  # noqa: E402
    _decode_ppm, _mean_absolute_rgb, _rgb_frame, measure_continuity,
)

SEED = 42
SOURCE_SIZE = "512x512"
VIDEO_SIZE = "704x480"
FRAMES = 29
STEPS = 20
FPS = 16
GUIDANCE = 5.0
FIRST_FRAME_MAX_MAE = 48.0
MOTION_MIN_MAE = 1.0
METRIC_VERSION = "center-cover-64x40-rgb8-frame0-frame28-v1"
SOURCE_PROMPT = (
    "A single vivid red paper lantern above a misty blue mountain lake at dawn, "
    "cinematic composition, clear subject, gentle ripples")
MOTION_PROMPT = (
    "The red lantern sways gently while the camera slowly orbits above the misty lake")
REPORT_FORMAT = "umminfer-flux2-wan22-i2v-continuity-v1"


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
    return {"sha256": digest.hexdigest(), "bytes": size}


def executable_identity(executable):
    resolved = shutil.which(executable)
    require(resolved is not None, f"executable is unavailable: {executable}")
    return sha256_file(resolved)


def canonical_sha256(value):
    return sha256_bytes(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


def model_identity(root, model_id):
    root = Path(root)
    require(root.is_dir(), f"model directory is unavailable: {root}")
    files = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        files[path.relative_to(root).as_posix()] = sha256_file(path)
    require(any(name.endswith(".safetensors") for name in files),
            f"no model weights under {root}")
    for required in ("model_index.json", "transformer/config.json", "vae/config.json"):
        require(required in files, f"missing model manifest {required}")
    return {"model_id": model_id, "file_count": len(files), "files": files,
            "tree_sha256": canonical_sha256(files)}


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


def gpu1_memory(nvidia_smi):
    result = subprocess.run(
        [nvidia_smi, "-i", "1", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=20)
    fields = [int(value.strip()) for value in result.stdout.strip().split(",")]
    require(len(fields) == 3 and fields[0] == 1,
            f"unexpected physical GPU 1 response: {result.stdout!r}")
    return {"index": fields[0], "memory_used_mib": fields[1],
            "utilization_percent": fields[2]}


def require_gpu1_idle(nvidia_smi):
    observed = gpu1_memory(nvidia_smi)
    require(observed["memory_used_mib"] <= 16,
            f"physical GPU 1 is not idle: {observed}")
    return observed


def full_rgb(ffmpeg, media, selector=None):
    command = [ffmpeg, "-v", "error", "-i", str(media)]
    if selector is not None:
        command += ["-vf", selector]
    command += ["-frames:v", "1", "-f", "image2pipe", "-vcodec", "ppm", "-"]
    result = subprocess.run(command, check=True, capture_output=True, timeout=60)
    return _decode_ppm(result.stdout)


def probe_video(ffprobe, path):
    result = subprocess.run([
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,nb_frames,r_frame_rate,pix_fmt",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True, timeout=30)
    stream = json.loads(result.stdout)["streams"][0]
    require(stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
            and (stream["width"], stream["height"]) == (704, 480)
            and int(stream["nb_frames"]) == FRAMES
            and stream["r_frame_rate"] == f"{FPS}/1",
            f"unexpected Wan video contract: {stream}")
    return stream


def public_request(request_id, inputs):
    return {"type": "submit", "protocol_version": 1, "request_id": request_id,
            "data": {"_type": "StagePayload", "request_id": request_id,
                     "request": {"inputs": inputs}}}


def sanitized_inputs(inputs):
    result = {}
    for key, value in inputs.items():
        if isinstance(value, bytes):
            result[key] = {"encoding": "msgpack-bin", "bytes": len(value),
                           "sha256": sha256_bytes(value)}
        else:
            result[key] = value
    return result


class DirectStage:
    def __init__(self, name, command, out_dir, timeout_s):
        import msgpack
        import zmq
        self.msgpack = msgpack
        self.zmq = zmq
        self.name = name
        self.command = command
        self.timeout_s = timeout_s
        self.out_dir = Path(out_dir)
        self.run_dir = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=self.out_dir))
        self.recv = f"ipc://{self.run_dir}/submit"
        self.done = f"ipc://{self.run_dir}/complete"
        self.abort = f"ipc://{self.run_dir}/abort"
        self.context = None
        self.pull = None
        self.pub = None
        self.push = None
        self.log_path = self.out_dir / f"{name}.stage.log"
        self.log = None
        self.process = None
        self.records = []
        self.stage_record = None

    def __enter__(self):
        self.command = [self.recv if item == "RECV" else self.done if item == "DONE"
                        else self.abort if item == "ABORT" else item
                        for item in self.command]
        self.start_ns = time.monotonic_ns()
        try:
            self.context = self.zmq.Context()
            self.pull = self.context.socket(self.zmq.PULL)
            self.pull.RCVTIMEO = 1000
            self.pull.bind(self.done)
            self.pub = self.context.socket(self.zmq.PUB)
            self.pub.bind(self.abort)
            self.push = self.context.socket(self.zmq.PUSH)
            self.push.connect(self.recv)
            self.log = self.log_path.open("wb")
            self.process = subprocess.Popen(
                self.command, stdout=self.log, stderr=subprocess.STDOUT,
                env={key: value for key, value in os.environ.items()
                     if key != "CUDA_VISIBLE_DEVICES"})
            self.ready_ns = self._wait_ready()
        except Exception as error:
            self.__exit__(type(error), error, error.__traceback__)
            raise
        return self

    def _wait_ready(self):
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"{self.name} exited before ready; see {self.log_path.name}")
            try:
                message = self.msgpack.unpackb(self.pull.recv(), raw=False)
            except self.zmq.Again:
                continue
            if message.get("type") == "stage_ready" \
                    and message.get("from_stage") == self.name:
                return time.monotonic_ns()
        raise TimeoutError(f"{self.name} did not become ready")

    def submit(self, request_id, inputs):
        body = public_request(request_id, inputs)
        packed = self.msgpack.packb(body, use_bin_type=True)
        submit_ns = time.monotonic_ns()
        self.push.send(packed)
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"{self.name} exited during {request_id}")
            try:
                message = self.msgpack.unpackb(self.pull.recv(), raw=False)
            except self.zmq.Again:
                continue
            if message.get("type") == "complete" \
                    and message.get("request_id") == request_id:
                complete_ns = time.monotonic_ns()
                result = message.get("result") or {}
                record = {
                    "request_id": request_id, "inputs": sanitized_inputs(inputs),
                    "request_sha256": sha256_bytes(packed),
                    "success": bool(message.get("success")),
                    "error": message.get("error"),
                    "submit_monotonic_ns": submit_ns,
                    "complete_monotonic_ns": complete_ns,
                    "latency_ms": (complete_ns - submit_ns) / 1e6,
                    "result": {key: value for key, value in result.items()
                               if key != "data"},
                }
                if "data" in result:
                    record["media_bytes"] = len(result["data"])
                    record["media_sha256"] = sha256_bytes(result["data"])
                self.records.append(record)
                with (self.out_dir / "records.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                return message, record
        raise TimeoutError(f"{request_id} did not complete")

    def __exit__(self, exc_type, exc, tb):
        stop_error = None
        if self.process is not None and self.process.poll() is None:
            try:
                self.push.send(self.msgpack.packb({"type": "shutdown"}, use_bin_type=True))
                self.process.wait(timeout=30)
            except Exception as error:
                stop_error = str(error)
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        stop_ns = time.monotonic_ns()
        if self.process is not None:
            self.stage_record = {
                "stage": self.name, "pid": self.process.pid,
                "returncode": self.process.returncode,
                "argv_sha256": sha256_bytes("\0".join(self.command).encode()),
                "start_monotonic_ns": self.start_ns,
                "ready_monotonic_ns": getattr(self, "ready_ns", self.start_ns),
                "stop_monotonic_ns": stop_ns,
                "log": self.log_path.name,
                "stop_error": stop_error,
            }
        if self.log is not None:
            self.log.close()
        for socket in (self.push, self.pub, self.pull):
            if socket is not None:
                socket.close(linger=0)
        if self.context is not None:
            self.context.term()
        for socket in self.run_dir.glob("*"):
            socket.unlink()
        if self.run_dir.exists():
            self.run_dir.rmdir()
        if exc_type is None and stop_error is not None:
            raise RuntimeError(f"{self.name} shutdown failed: {stop_error}")


def save_media(record, result, out_dir):
    media = result["data"]
    suffix = "png" if record["result"]["media_type"] == "image" else "mp4"
    name = f"{record['request_id']}.{suffix}"
    (out_dir / name).write_bytes(media)
    record["media_file"] = name
    require(sha256_file(out_dir / name)["sha256"] == record["media_sha256"],
            f"retained media hash mismatch: {name}")


def save_video_rgb(record, source_eval, args, out_dir, *, enforce_i2v):
    media_path = out_dir / record["media_file"]
    first = _rgb_frame(args.ffmpeg, media_path, "select=eq(n\\,0)")
    last = _rgb_frame(args.ffmpeg, media_path, "select=eq(n\\,28)")
    first_full = full_rgb(args.ffmpeg, media_path, "select=eq(n\\,0)")
    last_full = full_rgb(args.ffmpeg, media_path, "select=eq(n\\,28)")
    stem = record["request_id"]
    outputs = {
        "first_eval": (f"{stem}.first.rgb64x40", first),
        "last_eval": (f"{stem}.last.rgb64x40", last),
        "first_full": (f"{stem}.first.rgb704x480", first_full),
        "last_full": (f"{stem}.last.rgb704x480", last_full),
    }
    record["decoded_rgb"] = {}
    for label, (name, raw) in outputs.items():
        (out_dir / name).write_bytes(raw)
        record["decoded_rgb"][label] = {"file": name, **sha256_file(out_dir / name)}
    record["first_frame_mae"] = round(_mean_absolute_rgb(source_eval, first), 6)
    record["motion_mae"] = round(_mean_absolute_rgb(first, last), 6)
    record["metric_version"] = METRIC_VERSION
    record["video_stream"] = probe_video(args.ffprobe, media_path)
    if enforce_i2v:
        # Call the accepted narrative gate unchanged; no threshold relaxation.
        accepted = measure_continuity(
            args.ffmpeg, out_dir / "source.png", media_path)
        require(accepted["first_frame_mae"] == record["first_frame_mae"]
                and accepted["motion_mae"] == record["motion_mae"],
                "accepted narrative metric disagrees with retained RGB")
        require(record["first_frame_mae"] <= FIRST_FRAME_MAX_MAE
                and record["motion_mae"] >= MOTION_MIN_MAE,
                "I2V numeric continuity gate failed")


def verify_fresh(out_dir, verifier, ffmpeg, ffprobe):
    output = out_dir / "independent-verification.json"
    result = subprocess.run(
        [sys.executable, "-I", str(verifier), str(out_dir), "--out", str(output),
         "--ffmpeg", ffmpeg, "--ffprobe", ffprobe],
        capture_output=True, text=True, timeout=180)
    require(result.returncode == 0,
            f"fresh-process verification failed:\n{result.stdout}\n{result.stderr}")
    return output


def self_check():
    source = b"source-png-bytes"
    inputs = {"prompt": MOTION_PROMPT, "image": source,
              "anchor_explicit": True, "anchor_sha256": sha256_bytes(source)}
    body = public_request("contract-i2v", inputs)
    require(body["data"]["request_id"] == "contract-i2v", "request id binding missing")
    require(sanitized_inputs(inputs)["image"]["sha256"] == sha256_bytes(source),
            "source bytes were not hashed exactly")
    require("image" not in {"prompt": MOTION_PROMPT}, "T2V control unexpectedly anchored")
    require(_mean_absolute_rgb(bytes([0, 0, 0]), bytes([255, 255, 255])) == 255.0,
            "RGB MAE self-check failed")
    require(FIRST_FRAME_MAX_MAE == 48.0 and MOTION_MIN_MAE == 1.0,
            "frozen thresholds changed")
    print("OFFLINE_DIRECT_CONTRACT: PASS (raw-byte binding, T2V shape, frozen metrics)")
    print("REAL_GENERATION: NOT_RUN")
    print("PERCEPTUAL_QUALITY: NOT_RUN")


def live(args):
    require(os.environ.get("UMMINFER_LIVE_OPT_IN") == "1",
            "REAL_GENERATION: NOT_RUN (set UMMINFER_LIVE_OPT_IN=1 explicitly)")
    require(args.gpu == 1, "this gate permits only physical GPU 1")
    require(args.execution_commit is not None and len(args.execution_commit) == 40
            and all(c in "0123456789abcdef" for c in args.execution_commit),
            "live mode requires the exact lowercase --execution-commit")
    args.ffmpeg = shutil.which(args.ffmpeg)
    args.ffprobe = shutil.which(args.ffprobe)
    args.nvidia_smi = shutil.which(args.nvidia_smi)
    require(args.ffmpeg and args.ffprobe and args.nvidia_smi,
            "ffmpeg, ffprobe, and nvidia-smi must resolve through PATH")
    out_dir = Path(args.out_dir)
    require(not out_dir.exists(), f"evidence directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    verifier = Path(args.verifier).resolve()
    require(verifier.is_file(), "independent verifier is missing")
    provenance = {
        "base_commit": args.base_commit,
        "execution_commit": args.execution_commit,
        "flux_stage": sha256_file(args.flux_stage),
        "wan_stage_binary": sha256_file(args.wan_stage),
        "harness": sha256_file(Path(__file__).resolve()),
        "independent_verifier": sha256_file(verifier),
        "ffmpeg": sha256_file(args.ffmpeg),
        "ffprobe": sha256_file(args.ffprobe),
        "runtime": runtime_identity(),
    }
    models = {
        "flux2": model_identity(args.flux_model, "black-forest-labs/FLUX.2-klein-4B"),
        "wan22": model_identity(args.wan_model, "Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
    }
    gpu = {"before": require_gpu1_idle(args.nvidia_smi)}
    stages = []
    flux_command = [
        str(args.python), "-I", str(args.flux_stage), "--name", "image",
        "--backend", "diffusers", "--model", str(args.flux_model),
        "--model-id", "black-forest-labs/FLUX.2-klein-4B",
        "--revision", "UNKNOWN", "--gpu", "1", "--default-size", SOURCE_SIZE,
        "--default-steps", "4", "--recv", "RECV", "--coordinator", "DONE",
        "--abort", "ABORT",
    ]
    # DirectStage owns unique endpoints; replace placeholders without a shell.
    with DirectStage("image", flux_command, out_dir, args.timeout) as stage:
        source_inputs = {
            "prompt": SOURCE_PROMPT, "size": SOURCE_SIZE, "steps": 4,
            "guidance_scale": 4.0, "seed": SEED, "output_format": "png",
        }
        message, source = stage.submit("source-r0", source_inputs)
        require(source["success"], f"FLUX source generation failed: {source['error']}")
        require(source["result"].get("backend") == "diffusers"
                and source["result"].get("model") == "black-forest-labs/FLUX.2-klein-4B"
                and source["result"].get("seed") == SEED
                and source["result"].get("steps") == 4
                and source["result"].get("size") == SOURCE_SIZE,
                f"unexpected real FLUX result: {source['result']}")
        save_media(source, message["result"], out_dir)
        require(source["media_file"] == "source-r0.png", "unexpected source name")
        (out_dir / "source.png").write_bytes((out_dir / "source-r0.png").read_bytes())
        source_full = full_rgb(args.ffmpeg, out_dir / "source.png")
        require(len(source_full) == 512 * 512 * 3, "source RGB dimensions mismatch")
        source_eval = _rgb_frame(args.ffmpeg, out_dir / "source.png")
        (out_dir / "source.rgb512x512").write_bytes(source_full)
        (out_dir / "source.rgb64x40").write_bytes(source_eval)
        source["decoded_rgb"] = {
            "full": {"file": "source.rgb512x512", **sha256_file(out_dir / "source.rgb512x512")},
            "evaluation": {"file": "source.rgb64x40", **sha256_file(out_dir / "source.rgb64x40")},
        }
    stages.append(stage.stage_record)
    require(stage.stage_record["returncode"] == 0, "FLUX stage did not stop cleanly")
    gpu["between"] = require_gpu1_idle(args.nvidia_smi)

    source_bytes = (out_dir / "source.png").read_bytes()
    source_sha = sha256_bytes(source_bytes)
    require(source_sha == source["media_sha256"], "source bytes changed before Wan load")
    wan_command = [
        str(args.wan_stage), "--kind", "video", "--name", "video",
        "--video-backend", "cuda", "--model", str(args.wan_model),
        "--gpu", "1", "--size", VIDEO_SIZE, "--steps", str(STEPS),
        "--max-frames", str(FRAMES), "--fps", str(FPS),
        "--ffmpeg-bin", args.ffmpeg,
        "--recv-endpoint", "RECV", "--coordinator-endpoint", "DONE",
        "--abort-endpoint", "ABORT",
    ]
    negatives = []
    measured = []
    with DirectStage("video", wan_command, out_dir, args.timeout) as stage:
        common = {
            "prompt": MOTION_PROMPT, "size": VIDEO_SIZE, "num_frames": FRAMES,
            "steps": STEPS, "guidance_scale": GUIDANCE, "seed": SEED,
            "output_format": "mp4",
        }
        wrong_inputs = {**common, "image": source_bytes, "anchor_explicit": True,
                        "anchor_slot_index": 0, "anchor_crop": "center_cover",
                        "anchor_sha256": "0" * 64}
        message, wrong = stage.submit("negative-wrong-hash", wrong_inputs)
        wrong["control"] = "wrong_hash"
        negatives.append(wrong)

        # A valid but changed PNG proves that the check binds content, not merely
        # the presence or syntactic validity of an image data field.
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(source_bytes)).convert("RGB")
        old = image.getpixel((0, 0))
        image.putpixel((0, 0), (255 - old[0], old[1], 255 - old[2]))
        changed = io.BytesIO()
        image.save(changed, format="PNG")
        tampered_bytes = changed.getvalue()
        require(sha256_bytes(tampered_bytes) != source_sha, "tamper did not change bytes")
        (out_dir / "tampered-source.png").write_bytes(tampered_bytes)
        tampered_inputs = {**common, "image": tampered_bytes, "anchor_explicit": True,
                           "anchor_slot_index": 0, "anchor_crop": "center_cover",
                           "anchor_sha256": source_sha}
        message, tampered = stage.submit("negative-tampered-source", tampered_inputs)
        tampered["control"] = "tampered_source"
        negatives.append(tampered)

        for control in negatives:
            require(not control["success"]
                    and control["error"] == "video.image does not match anchor_sha256"
                    and "media_file" not in control,
                    f"anchor negative control was not rejected pre-generation: {control}")

        # Fixed order is pre-registered: no run is selected or discarded after
        # seeing pixels, latency, or hashes. All six requests enter the medians.
        for index in range(3):
            for mode in ("i2v", "t2v"):
                inputs = dict(common)
                if mode == "i2v":
                    inputs.update({
                        "image": source_bytes, "anchor_explicit": True,
                        "anchor_slot_index": 0, "anchor_crop": "center_cover",
                        "anchor_sha256": source_sha,
                    })
                request_id = f"{mode}-r{index}"
                message, record = stage.submit(request_id, inputs)
                record["mode"] = mode
                record["ordinal"] = len(measured)
                record["source_sha256"] = source_sha if mode == "i2v" else None
                measured.append(record)
                require(record["success"], f"{request_id} failed: {record['error']}")
                require(record["result"].get("backend") == "native-cuda-wan"
                        and record["result"].get("model") == "Wan2.2-TI2V-5B-Diffusers"
                        and record["result"].get("seed") == SEED
                        and record["result"].get("steps") == STEPS
                        and record["result"].get("frames") == FRAMES,
                        f"unexpected real Wan result: {record['result']}")
                save_media(record, message["result"], out_dir)
                save_video_rgb(record, source_eval, args, out_dir,
                               enforce_i2v=(mode == "i2v"))
    stages.append(stage.stage_record)
    require(stage.stage_record["returncode"] == 0, "Wan stage did not stop cleanly")
    gpu["after"] = require_gpu1_idle(args.nvidia_smi)

    stage_log = (out_dir / "video.stage.log").read_text(encoding="utf-8")
    for record in measured:
        line = next((line for line in stage_log.splitlines()
                     if "video request backend=" in line
                     and f"request_id={record['request_id']}" in line), None)
        require(line is not None, f"native execution log missing: {record['request_id']}")
        expected = "i2v=1" if record["mode"] == "i2v" else "i2v=0"
        require(expected in line, f"native path mismatch for {record['request_id']}: {line}")
        record["native_path"] = expected
        record["native_log_sha256"] = sha256_bytes(line.encode())
    for control in negatives:
        require(not any(f"request_id={control['request_id']}" in line
                        and "video request backend=" in line
                        for line in stage_log.splitlines()),
                f"rejected anchor reached generation: {control['request_id']}")

    i2v = [record for record in measured if record["mode"] == "i2v"]
    t2v = [record for record in measured if record["mode"] == "t2v"]
    require(len({record["media_sha256"] for record in i2v}) == 1,
            "fixed-seed I2V outputs are not byte-reproducible")
    require(len({record["media_sha256"] for record in t2v}) == 1,
            "fixed-seed T2V controls are not byte-reproducible")
    require(i2v[0]["media_sha256"] != t2v[0]["media_sha256"],
            "explicit I2V and T2V control returned identical bytes")
    all_records = [source, *negatives, *measured]
    for record in all_records:
        require(record["submit_monotonic_ns"] <= record["complete_monotonic_ns"],
                f"nonmonotonic request timing: {record['request_id']}")
    require(stages[0]["start_monotonic_ns"] <= stages[0]["ready_monotonic_ns"]
            <= source["submit_monotonic_ns"] <= source["complete_monotonic_ns"]
            <= stages[0]["stop_monotonic_ns"] <= stages[1]["start_monotonic_ns"]
            <= stages[1]["ready_monotonic_ns"] <= measured[0]["submit_monotonic_ns"]
            <= measured[-1]["complete_monotonic_ns"] <= stages[1]["stop_monotonic_ns"],
            "native stages overlapped or stage timing is nonmonotonic")

    with (out_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in all_records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    report = {
        "format": REPORT_FORMAT, "status": "GENERATED_AWAITING_INDEPENDENT_VERIFICATION",
        "base_commit": args.base_commit, "execution_commit": args.execution_commit,
        "physical_gpu": 1,
        "source": source, "measured_runs": measured, "negative_controls": negatives,
        "real_generation_counts": {
            "source_images": 1, "i2v_videos": len(i2v), "t2v_control_videos": len(t2v),
            "successful_requests": 1 + len(measured), "rejected_anchor_requests": len(negatives),
        },
        "latency_median_ms": {
            "source_request": round(statistics.median([source["latency_ms"]]), 6),
            "i2v_request": round(statistics.median(r["latency_ms"] for r in i2v), 6),
            "t2v_request": round(statistics.median(r["latency_ms"] for r in t2v), 6),
            "i2v_denoise": round(statistics.median(r["result"]["denoise_ms"] for r in i2v), 6),
            "t2v_denoise": round(statistics.median(r["result"]["denoise_ms"] for r in t2v), 6),
        },
        "metric_contract": {
            "version": METRIC_VERSION, "source_size": SOURCE_SIZE,
            "evaluation_size": "64x40", "crop": "center_cover",
            "first_frame_index": 0, "last_frame_index": 28,
            "first_frame_mae_max_i2v": FIRST_FRAME_MAX_MAE,
            "motion_mae_min_i2v": MOTION_MIN_MAE,
            "meaning": "numeric RGB continuity/motion proxies only",
        },
        "models": models, "provenance": provenance, "stages": stages, "gpu1": gpu,
        "public_vlm_narrative_orchestration": "NOT_RUN",
        "perceptual_quality": "NOT_RUN",
        "t2v_control_thresholds": "DESCRIPTIVE_ONLY; frozen I2V thresholds are not applied",
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = out_dir / "SHA256SUMS"
    paths = sorted(path for path in out_dir.iterdir() if path.is_file()
                   and path.name not in {"SHA256SUMS", "independent-verification.json"})
    manifest.write_text("".join(
        f"{sha256_file(path)['sha256']}  {path.name}\n" for path in paths), encoding="utf-8")
    verification = verify_fresh(out_dir, verifier, args.ffmpeg, args.ffprobe)
    summary = {
        "status": "CONTROLLED_LIVE", "source_sha256": source_sha,
        "i2v_sha256": i2v[0]["media_sha256"], "t2v_sha256": t2v[0]["media_sha256"],
        "i2v_first_frame_mae": [r["first_frame_mae"] for r in i2v],
        "i2v_motion_mae": [r["motion_mae"] for r in i2v],
        "t2v_first_frame_mae": [r["first_frame_mae"] for r in t2v],
        "t2v_motion_mae": [r["motion_mae"] for r in t2v],
        "latency_median_ms": report["latency_median_ms"],
        "report_sha256": sha256_file(report_path)["sha256"],
        "manifest_sha256": sha256_file(manifest)["sha256"],
        "verification_sha256": sha256_file(verification)["sha256"],
        "gpu1_after": gpu["after"],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    print("REAL_GENERATION: PASS (1 FLUX source + 3 native I2V + 3 native T2V controls)")
    print("PUBLIC_VLM_NARRATIVE: NOT_RUN")
    print("PERCEPTUAL_QUALITY: NOT_RUN (numeric RGB proxies only)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--out-dir", default="/tmp/umm-flux-i2v-62-final")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--flux-stage", type=Path, required=False)
    parser.add_argument("--wan-stage", type=Path)
    parser.add_argument("--flux-model", type=Path)
    parser.add_argument("--wan-model", type=Path)
    parser.add_argument("--verifier", type=Path,
                        default=Path(__file__).with_name("flux2_i2v_continuity_verify.py"))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--base-commit", default="27c584e8da571e16434969e9c1a11f46aabc6e76")
    parser.add_argument("--execution-commit")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    for name in ("flux_stage", "wan_stage", "flux_model", "wan_model"):
        if getattr(args, name) is None:
            parser.error(f"live mode requires --{name.replace('_', '-')}")
    live(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
