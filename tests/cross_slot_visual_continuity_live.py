#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real public three-slot FLUX.2 edit -> native Wan2.2 I2V gate on GPU 7.

Offline check:
  python3 -I tests/cross_slot_visual_continuity_live.py --self-check

The opt-in live mode starts only task-owned processes on physical GPU 7. It
retains raw public SSE/CAS, native MessagePack, exact source bytes, a cancelled
three-slot request, a wrong-reference request, wrong-hash native rejection, and
valid recovery. It never contacts the resident service. Human/perceptual
character similarity is not evaluated by this harness.
"""
import argparse
import base64
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "python"))
import public_narrative_i2v_live as base  # noqa: E402
from umminfer_rl_trajectory import (  # noqa: E402
    TrajectoryRecorder, input_references, parse_sse_stream,
    request_parameter_references,
)
from umminfer_sse_replay import PerfCounterClock, RawSSECapture  # noqa: E402

SEED = 42
SOURCE_SIZE = "512x512"
VIDEO_SIZE = "704x480"
REPORT_FORMAT = "umminfer-cross-slot-visual-continuity-v1"
DEFAULT_LIBRARY_PATH = ""
IMAGE0 = ("[[image: a single vivid red paper lantern above a misty blue mountain "
          "lake at dawn, cinematic composition, clear subject, gentle ripples]]")
IMAGE1 = ("[[image: the same red paper lantern above the same misty blue mountain "
          "lake at night, preserve the lantern design and cinematic style, add "
          "warm lantern glow | source=slot-0]]")
VIDEO2 = ("[[video: the red lantern sways gently while the camera slowly orbits "
          "above the misty lake | source=slot-1]]")
WRONG_VIDEO = ("[[video: the lantern moves | source=slot-0]]")
USER_TEXT = (
    "Create a three-beat visual story. Output one short standalone English prose "
    "sentence, then exactly this standalone marker: " + IMAGE0 +
    " Then output one short standalone English prose sentence and exactly this "
    "standalone marker: " + IMAGE1 +
    " Then output one short standalone English prose sentence and exactly this "
    "standalone marker: " + VIDEO2 +
    " Do not output any other marker or repeat these instructions.")
WRONG_USER_TEXT = (
    "Output one short English sentence and then exactly this standalone marker: "
    + WRONG_VIDEO + " Do not create an earlier image or any other marker.")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def gpu7_memory(nvidia_smi):
    result = subprocess.run(
        [nvidia_smi, "-i", "7", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], check=True, capture_output=True,
        text=True, timeout=20)
    fields = [int(value.strip()) for value in result.stdout.strip().split(",")]
    require(len(fields) == 3 and fields[0] == 7,
            f"unexpected physical GPU 7 response: {result.stdout!r}")
    return {"index": fields[0], "memory_used_mib": fields[1],
            "utilization_percent": fields[2]}


def gpu7_compute_pids(nvidia_smi):
    result = subprocess.run(
        [nvidia_smi, "-i", "7", "--query-compute-apps=pid",
         "--format=csv,noheader,nounits"], check=True, capture_output=True,
        text=True, timeout=20)
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]


def require_gpu7_idle(nvidia_smi):
    observed = gpu7_memory(nvidia_smi)
    pids = gpu7_compute_pids(nvidia_smi)
    require(observed["memory_used_mib"] <= 16
            and observed["utilization_percent"] == 0 and not pids,
            f"physical GPU 7 is not idle: {observed}, compute_pids={pids}")
    observed["compute_processes"] = 0
    return observed


def wait_gpu7_idle(nvidia_smi, timeout=30):
    deadline = time.monotonic() + timeout
    observed = None
    while time.monotonic() < deadline:
        observed = gpu7_memory(nvidia_smi)
        pids = gpu7_compute_pids(nvidia_smi)
        if observed["memory_used_mib"] <= 16 and not pids \
                and observed["utilization_percent"] == 0:
            observed["compute_processes"] = 0
            return observed
        time.sleep(0.1)
    require(False, f"GPU 7 did not settle idle after task cleanup: {observed}")


def request_body(model, mode="main"):
    if mode == "wrong":
        return {
            "model": model, "stream": True, "session_id": "cross-slot-wrong-76",
            "messages": [{"role": "user", "content": WRONG_USER_TEXT}],
            "modalities": ["text", "video"],
            "video": {"size": VIDEO_SIZE, "num_frames": 29, "steps": 20,
                      "fps": 16, "guidance_scale": 5.0, "seed": SEED,
                      "output_format": "mp4"},
            "narrative": {"enabled": True, "visual_mode": "auto", "max_slots": 1,
                          "allow_text_only": False, "continuity_mode": "i2v_chain"},
            "max_tokens": 128, "temperature": 0, "seed": SEED,
        }
    body = {
        "model": model, "stream": True,
        "session_id": f"cross-slot-{mode}-76",
        "messages": [{"role": "user", "content": USER_TEXT}],
        "modalities": ["text", "image", "audio", "video"],
        "image": {"size": SOURCE_SIZE, "steps": 50 if mode == "cancel" else 4,
                  "seed": SEED, "output_format": "png"},
        "audio": {"voice": "vivian", "format": "wav", "stream": True},
        "video": {"size": VIDEO_SIZE, "num_frames": 29, "steps": 20,
                  "fps": 16, "guidance_scale": 5.0, "seed": SEED,
                  "output_format": "mp4"},
        "narrative": {"enabled": True, "visual_mode": "auto", "max_slots": 3,
                      "allow_text_only": False, "continuity_mode": "i2v_chain"},
        "max_tokens": 320, "temperature": 0, "seed": SEED,
    }
    return body


class GpuMonitor:
    def __init__(self, nvidia_smi):
        self.nvidia_smi = nvidia_smi
        self.samples = []
        self.error = None
        self.stop_requested = False
        self.thread = threading.Thread(target=self._run, name="gpu7-monitor", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stop_requested:
                self.samples.append({"observed_monotonic_ns": base.clock_ns(),
                                     **gpu7_memory(self.nvidia_smi)})
                time.sleep(0.05)
        except Exception as error:
            self.error = str(error)

    def stop(self):
        self.stop_requested = True
        self.thread.join(timeout=10)
        require(not self.thread.is_alive() and self.error is None,
                f"GPU 7 monitor failed: {self.error}")


def capture_public(port, body, recorder, session_id, request_id_out=None,
                   observed=None):
    params, option_inputs = request_parameter_references(body, recorder.artifact_dir)
    inputs = input_references(body["messages"], recorder.artifact_dir) + option_inputs
    capture = RawSSECapture(clock=PerfCounterClock(origin_ns=base.clock_ns()))
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    inspected = 0
    try:
        conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
                     headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
        response = conn.getresponse()
        if response.status != 200:
            error_body = response.read()
            raise AssertionError(f"public chat HTTP {response.status}: {error_body!r}")
        require(response.getheader("Content-Type", "").startswith("text/event-stream"),
                "public response is not SSE")
        while True:
            chunk = response.read1(65536)
            if not chunk:
                break
            capture.feed(chunk)
            payloads = capture.payload_frames
            for frame_index, payload in payloads[inspected:]:
                if payload == "[DONE]":
                    continue
                frame = json.loads(payload)
                request_id = frame.get("id")
                if request_id_out is not None and request_id:
                    request_id_out[0] = request_id
                unified = frame.get("unified") or {}
                if observed is not None and unified.get("kind") == "visual_slot":
                    observed.setdefault("visual_slots", []).append(unified.get("slot_id"))
                    if len(observed["visual_slots"]) == 3:
                        observed["all_slots"].set()
                if observed is not None and unified.get("kind") == "artifact":
                    desc = unified["artifact"]
                    slot_id = desc.get("slot_id")
                    if (slot_id in ("slot-0", "slot-1")
                            and desc.get("media_type") == "image"
                            and desc.get("index") + 1 == desc.get("total")):
                        require(desc.get("total") == 1,
                                "the frozen 512px source should fit one public chunk")
                        raw = base64.b64decode(desc["data"], validate=True)
                        require(base.sha256_bytes(raw) == desc.get("sha256"),
                                f"public {slot_id} source digest mismatch")
                        observed[slot_id] = {
                            "bytes": raw, "sha256": desc["sha256"],
                            "frame_index": frame_index,
                            "observed_monotonic_ns": capture._frames[frame_index]["tick"],
                            "observed_absolute_monotonic_ns": base.clock_ns(),
                            "seq": unified.get("seq"),
                        }
                        observed[slot_id + "-ready"].set()
            inspected = len(payloads)
        capture.finish()
        details = {}
        request_id, events, output_text, state, error = parse_sse_stream(
            io.BytesIO(capture.raw_bytes), artifact_dir=recorder.artifact_dir,
            details=details, capture=capture)
        record = recorder.record(
            session_id=session_id, request_id=request_id,
            endpoint="/v1/chat/completions", inputs=inputs, params=params,
            events=events, terminal_state=state, error=error, output_text=output_text,
            artifacts=details.get("artifacts", []), capture=capture)
        return record
    finally:
        conn.close()
        # Preserve even an unterminated/failed response for raw diagnosis. On a
        # successful capture this is byte-identical to the replay CAS object.
        recorder.artifact_dir.mkdir(parents=True, exist_ok=True)
        (recorder.artifact_dir / "raw-capture.sse").write_bytes(capture.raw_bytes)


def observation_state():
    return {"all_slots": threading.Event(),
            "slot-0-ready": threading.Event(), "slot-1-ready": threading.Event()}


def direct_native_controls(image_endpoint, output_tap, source, out):
    """Wrong hash must fail before generation; exact hash then recovers."""
    import msgpack
    import zmq
    context = zmq.Context()
    push = context.socket(zmq.PUSH)
    push.connect(image_endpoint)
    results = []
    try:
        for label, declared in (("wrong-hash", "0" * 64),
                                ("valid-retry", base.sha256_bytes(source))):
            request_id = f"native-76-{label}"
            inputs = {
                "prompt": "preserve the lantern and style; add subtle night glow",
                "size": SOURCE_SIZE, "steps": 4, "seed": SEED,
                "output_format": "png", "image": source,
                "anchor_explicit": True, "anchor_slot_index": 0,
                "anchor_sha256": declared,
            }
            message = {"type": "submit", "request_id": request_id,
                       "data": {"request": {"inputs": inputs}}}
            packed = msgpack.packb(message, use_bin_type=True)
            name = f"{request_id}-submit.msgpack"
            (out / name).write_bytes(packed)
            push.send(packed)
            deadline = time.monotonic() + 120
            complete = None
            output_record = None
            while time.monotonic() < deadline:
                for item in output_tap.records:
                    if item.get("request_id") == request_id:
                        output_record = item
                        complete = msgpack.unpackb(
                            (out / item["packed_file"]).read_bytes(), raw=False)
                        break
                if complete is not None:
                    break
                time.sleep(0.01)
            require(complete is not None and output_record is not None
                    and complete.get("type") == "complete",
                    f"native {label} emitted no complete")
            require(complete.get("success") is (label == "valid-retry"),
                    f"native {label} unexpected terminal: {complete.get('error')}")
            entry = {"request_id": request_id, "submit_file": name,
                     "submit_sha256": base.sha256_bytes(packed),
                     "source_sha256": base.sha256_bytes(source),
                     "declared_anchor_sha256": declared,
                     "output_file": output_record["packed_file"],
                     "output_packed_sha256": output_record["packed_sha256"],
                     "success": complete.get("success"),
                     "error": complete.get("error")}
            if label == "wrong-hash":
                require("does not match anchor_sha256" in (complete.get("error") or ""),
                        f"wrong-hash error changed: {complete.get('error')}")
            else:
                png = complete["result"]["data"]
                require(base.is_valid_png(png) if hasattr(base, "is_valid_png")
                        else png.startswith(b"\x89PNG\r\n\x1a\n"),
                        "valid native recovery did not return PNG")
                (out / "native-control-recovery.png").write_bytes(png)
                entry.update({"output_sha256": base.sha256_bytes(png),
                              "output_bytes": len(png),
                              "output_size": complete["result"].get("size")})
            results.append(entry)
    finally:
        push.close(0)
        context.term()
    return results


def native_submits(records):
    return [item for item in records if item.get("message_type") == "submit"]


def visual_legs(record):
    return [leg for leg in record["terminal_legs"] if leg.get("slot_id") is not None]


def live(args):
    require(os.environ.get("UMMINFER_LIVE_OPT_IN") == "1",
            "REAL_GENERATION: NOT_RUN (explicit UMMINFER_LIVE_OPT_IN=1 required)")
    require(args.gpu == 7, "this gate permits only physical GPU 7")
    require(len(args.base_commit) == 40 and len(args.execution_commit) == 40,
            "full base and execution commits are required")
    out = Path(args.out_dir)
    require(not out.exists(), f"evidence directory already exists: {out}")
    out.mkdir(parents=True)
    run = out / "run"
    run.mkdir()
    provenance = {
        "base_commit": args.base_commit, "execution_commit": args.execution_commit,
        "server_binary": {"path": str(args.server_binary), **base.sha256_file(args.server_binary)},
        "vlm_binary": {"path": str(args.vlm_binary), **base.sha256_file(args.vlm_binary)},
        "tts_binary": {"path": str(args.tts_binary), **base.sha256_file(args.tts_binary)},
        "wan_binary": {"path": str(args.wan_binary), **base.sha256_file(args.wan_binary)},
        "flux_stage": {"path": str(args.flux_stage), **base.sha256_file(args.flux_stage)},
        "live_harness": {"path": str(Path(__file__).resolve()), **base.sha256_file(__file__)},
        "shared_live_helpers": {"path": str(Path(base.__file__).resolve()),
                                **base.sha256_file(base.__file__)},
        "independent_verifier": {"path": str(args.verifier), **base.sha256_file(args.verifier)},
        "ffmpeg": base.sha256_file(shutil.which(args.ffmpeg)),
        "ffprobe": base.sha256_file(shutil.which(args.ffprobe)),
        "python_runtime": base.runtime_identity(),
    }
    models = {
        "vlm": base.model_identity(args.vlm_model, "Qwen/Qwen3.8-27B"),
        "flux2": base.model_identity(args.flux_model, "black-forest-labs/FLUX.2-klein-4B"),
        "tts": base.model_identity(args.tts_model, "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        "wan22": base.model_identity(args.wan_model, "Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
    }
    bodies = {mode: request_body(args.model, mode) for mode in ("wrong", "cancel", "main")}
    (out / "requests.json").write_bytes(base.canonical(bodies) + b"\n")
    gpu = {"before": require_gpu7_idle(args.nvidia_smi)}
    monitor = GpuMonitor(args.nvidia_smi)
    port = args.port or base.free_port()
    require(port != 43301, "resident service port is forbidden")
    endpoints = {name: f"ipc://{run}/{name}" for name in (
        "completion", "abort", "entry-in", "vlm-in", "image-tap-in", "image-in",
        "image-output-in", "tts-tap-in", "tts-in", "tts-output-in",
        "video-tap-in", "video-in")}
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if args.library_path:
        env["LD_LIBRARY_PATH"] = str(args.library_path) + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["UMM_TRACE"] = "1"
    env["UMM_TRACE_FILE"] = str(out / "server.trace.jsonl")
    server = entry = vlm = flux = tts = wan = None
    image_tap = image_output_tap = video_tap = tts_tap = tts_output_tap = None
    manager_error = []
    manager_record = {"native_controls": False, "flux_to_wan": False}
    stage_record = {}
    main_record = wrong_record = cancel_record = None
    main_observed = observation_state()
    native_controls = []
    health_final = None

    def start_wan():
        nonlocal wan
        wan = base.ManagedProcess("video", [
            args.wan_binary, "--kind", "video", "--name", "video", "--video-backend", "cuda",
            "--model", args.wan_model, "--gpu", "7", "--size", VIDEO_SIZE,
            "--steps", "20", "--fps", "16", "--recv-endpoint", endpoints["video-in"],
            "--coordinator-endpoint", endpoints["completion"],
            "--abort-endpoint", endpoints["abort"]], out / "video.log", env=env)
        wan.wait_log("native stage video ready kind=video", args.timeout)
        gpu["wan_ready"] = gpu7_memory(args.nvidia_smi)

    def output_manager():
        try:
            require(main_observed["slot-0-ready"].wait(args.timeout),
                    "main source slot-0 did not complete")
            native_controls.extend(direct_native_controls(
                endpoints["image-in"], image_output_tap,
                main_observed["slot-0"]["bytes"], out))
            manager_record["native_controls"] = True
            require(main_observed["slot-1-ready"].wait(args.timeout),
                    "edited source slot-1 did not complete")
            stage_record["edited_public_complete_ns"] = base.clock_ns()
            flux.stop(endpoints["image-in"])
            require(flux.returncode == 0, f"FLUX shutdown failed: {flux.returncode}")
            gpu["after_flux_stop"] = gpu7_memory(args.nvidia_smi)
            stage_record["flux_stopped_ns"] = base.clock_ns()
            start_wan()
            manager_record["flux_to_wan"] = True
        except Exception as error:
            manager_error.append(str(error))
            try:
                base.http_json(port, "POST", "/abort", {
                    "request_id": main_request_id[0],
                    "session_id": "cross-slot-main-76"}, timeout=30)
            except Exception:
                pass

    try:
        image_tap = base.TapProxy("image", endpoints["image-tap-in"],
                                  endpoints["image-in"], out)
        image_output_tap = base.TapProxy("image-output", endpoints["image-output-in"],
                                         endpoints["completion"], out)
        video_tap = base.TapProxy("video", endpoints["video-tap-in"],
                                  endpoints["video-in"], out)
        tts_tap = base.TapProxy("tts", endpoints["tts-tap-in"], endpoints["tts-in"], out)
        tts_output_tap = base.TapProxy(
            "tts-output", endpoints["tts-output-in"], endpoints["completion"], out)
        server = base.ManagedProcess("server", [
            args.server_binary, "serve", "--host", "127.0.0.1", "--port", str(port),
            "--model", args.model, "--entry", "entry",
            "--entry-endpoint", endpoints["entry-in"], "--vlm-entry", "vlm",
            "--vlm-entry-endpoint", endpoints["vlm-in"],
            "--image-entry", "image", "--image-entry-endpoint", endpoints["image-tap-in"],
            "--tts-entry", "tts", "--tts-entry-endpoint", endpoints["tts-tap-in"],
            "--video-entry", "video", "--video-entry-endpoint", endpoints["video-tap-in"],
            "--completion", endpoints["completion"], "--abort", endpoints["abort"],
            "--max-in-flight", "2"], out / "server.log", env=env)
        entry = base.ManagedProcess("entry", [
            args.server_binary, "stage", "--name", "entry", "--recv", endpoints["entry-in"],
            "--coordinator", endpoints["completion"], "--abort", endpoints["abort"]],
            out / "entry.log", env=env)
        vlm = base.ManagedProcess("vlm", [
            args.vlm_binary, "--kind", "qwen38", "--name", "vlm", "--model", args.vlm_model,
            "--gpu", "7", "--batch-size", "1", "--max-new-tokens", "320",
            "--recv-endpoint", endpoints["vlm-in"],
            "--coordinator-endpoint", endpoints["completion"],
            "--abort-endpoint", endpoints["abort"]], out / "vlm.log", env=env)
        vlm.wait_log("native stage vlm ready kind=qwen38", args.timeout)
        gpu["vlm_ready"] = gpu7_memory(args.nvidia_smi)
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            try:
                status, _ = base.http_json(port, "GET", "/ready", timeout=5)
                if status == 200:
                    entry.ready_ns = base.clock_ns()
                    break
            except (OSError, http.client.HTTPException):
                pass
            require(server.process.poll() is None, "public gateway exited before readiness")
            time.sleep(0.05)
        else:
            raise TimeoutError("public gateway did not become ready")

        wrong_recorder = TrajectoryRecorder(out / "wrong" / "trajectories.jsonl")
        wrong_record = capture_public(port, bodies["wrong"], wrong_recorder,
                                      "cross-slot-wrong-76")
        require(wrong_record["terminal_state"] == "completed"
                and len(wrong_record["visual_slots"]) == 1
                and len(visual_legs(wrong_record)) == 1
                and visual_legs(wrong_record)[0]["state"] == "failed"
                and not wrong_record["artifacts"] and not native_submits(video_tap.records),
                "wrong/self reference did not fail before native video generation")

        flux = base.ManagedProcess("image", [
            args.python, "-I", args.flux_stage, "--name", "image", "--backend", "diffusers",
            "--model", args.flux_model, "--model-id", "black-forest-labs/FLUX.2-klein-4B",
            "--revision", "UNKNOWN", "--gpu", "7", "--default-size", SOURCE_SIZE,
            "--default-steps", "4", "--recv", endpoints["image-in"],
            "--coordinator", endpoints["image-output-in"],
            "--abort", endpoints["abort"]], out / "image.log", env=env)
        flux.wait_log("ready kind=image name=image", args.timeout)
        gpu["flux_ready"] = gpu7_memory(args.nvidia_smi)
        tts = base.ManagedProcess("tts", [
            args.tts_binary, "--kind", "tts", "--name", "tts", "--model", args.tts_model,
            "--gpu", "7", "--recv-endpoint", endpoints["tts-in"],
            "--coordinator-endpoint", endpoints["tts-output-in"],
            "--abort-endpoint", endpoints["abort"]], out / "tts.log", env=env)
        tts.wait_log("native stage tts ready kind=tts", args.timeout)
        gpu["tts_ready"] = gpu7_memory(args.nvidia_smi)

        cancel_observed = observation_state()
        cancel_request_id = [None]
        cancel_result = []
        cancel_error = []
        cancel_recorder = TrajectoryRecorder(out / "cancel" / "trajectories.jsonl")

        def run_cancel():
            try:
                cancel_result.append(capture_public(
                    port, bodies["cancel"], cancel_recorder, "cross-slot-cancel-76",
                    cancel_request_id, cancel_observed))
            except Exception as error:
                cancel_error.append(str(error))

        cancel_thread = threading.Thread(target=run_cancel, name="cancel-capture")
        cancel_thread.start()
        require(cancel_observed["all_slots"].wait(args.timeout),
                "cancel request did not expose all three dependent slots")
        require(cancel_request_id[0], "cancel request has no public id")
        abort_status, _ = base.http_json(port, "POST", "/abort", {
            "request_id": cancel_request_id[0], "session_id": "cross-slot-cancel-76"})
        require(abort_status == 200, "real cancellation was rejected")
        cancel_thread.join(timeout=60)
        require(not cancel_thread.is_alive() and not cancel_error,
                f"cancel capture failed: {cancel_error}")
        cancel_record = cancel_result[0]
        time.sleep(0.2)
        require(cancel_record["terminal_state"] == "cancelled"
                and len(cancel_record["visual_slots"]) == 3
                and len(visual_legs(cancel_record)) == 3
                and all(leg["state"] == "aborted" for leg in visual_legs(cancel_record))
                and not [item for item in cancel_record["artifacts"]
                         if item["media_type"] in ("image", "video")]
                and len(native_submits(image_tap.records)) == 1
                and not native_submits(video_tap.records),
                "cancelled chain generated a dependent or failed to finalize every slot")

        manager = threading.Thread(target=output_manager, name="cross-slot-output-manager",
                                   daemon=True)
        main_recorder = TrajectoryRecorder(out / "main" / "trajectories.jsonl")
        main_request_id = [None]
        main_observed["request_id"] = None
        manager.start()
        main_record = capture_public(port, bodies["main"], main_recorder,
                                     "cross-slot-main-76", main_request_id, main_observed)
        manager.join(timeout=10)
        require(not manager.is_alive() and not manager_error,
                f"output manager failed: {manager_error}")
        require(manager_record == {"native_controls": True, "flux_to_wan": True},
                f"native controls/stage switch incomplete: {manager_record}")
        require(main_record["terminal_state"] == "completed",
                f"valid retry failed: {main_record['error']}")
        video_tap.stop()
        health_status, health_final = base.http_json(port, "GET", "/health", timeout=30)
        require(health_status == 200 and health_final.get("in_flight") == 0,
                f"public gateway retained in-flight work: {health_status} {health_final}")
        gpu["after_complete_before_cleanup"] = gpu7_memory(args.nvidia_smi)
    finally:
        for process, endpoint in ((wan, endpoints["video-in"]),
                                  (tts, endpoints["tts-in"]),
                                  (flux, endpoints["image-in"]),
                                  (vlm, endpoints["vlm-in"]),
                                  (entry, endpoints["entry-in"])):
            if process is not None and process.stop_ns is None:
                process.stop(endpoint)
        if server is not None:
            server.stop(None)
        for tap in (image_tap, image_output_tap, video_tap, tts_tap, tts_output_tap):
            if tap is not None and tap.thread.is_alive():
                tap.stop()
        monitor.stop()
        gpu["after_cleanup"] = wait_gpu7_idle(args.nvidia_smi)
        stage_record["processes"] = [process.record() for process in
                                     (server, entry, vlm, flux, tts, wan) if process is not None]
        (out / "gpu7-monitor.json").write_text(
            json.dumps(monitor.samples, indent=2, sort_keys=True) + "\n")
        require(not manager_error, f"live run failed; raw evidence retained: {manager_error}")

    require(main_record is not None and wrong_record is not None
            and cancel_record is not None, "a raw public control is missing")
    artifacts = main_record["artifacts"]
    images = sorted((item for item in artifacts if item["media_type"] == "image"),
                    key=lambda item: item["slot_index"])
    videos = [item for item in artifacts if item["media_type"] == "video"]
    audio = [item for item in artifacts if item["media_type"] == "audio"]
    require(len(images) == 2 and len(videos) == 1 and audio,
            "main retry did not produce two images, one video, and streaming speech")
    source0, source1 = images
    video = videos[0]
    main_root = out / "main"
    source0_path = main_recorder.artifact_dir / source0["ref"]
    source1_path = main_recorder.artifact_dir / source1["ref"]
    video_path = main_recorder.artifact_dir / video["ref"]
    require(source0_path.read_bytes() == main_observed["slot-0"]["bytes"]
            and source1_path.read_bytes() == main_observed["slot-1"]["bytes"],
            "public source event bytes differ from CAS")
    slots = main_record["visual_slots"]
    require(len(slots) == 3
            and [slot["media_type"] for slot in slots] == ["image", "image", "video"]
            and slots[1].get("anchor_slot_index") == 0
            and slots[2].get("anchor_slot_index") == 1
            and slots[1].get("anchor_explicit") is True
            and slots[2].get("anchor_explicit") is True
            and all("anchor_sha256" not in slot for slot in slots),
            "public three-slot reference identities are wrong or timing-dependent")
    require(source1.get("anchor_slot_index") == 0
            and source1.get("anchor_sha256") == source0["sha256"]
            and video.get("anchor_slot_index") == 1
            and video.get("anchor_sha256") == source1["sha256"],
            "public artifact reference chain does not bind exact source SHA-256")
    require(all(leg["state"] == "completed" for leg in visual_legs(main_record)),
            "main visual terminal legs are not all completed")
    main_image_submits = [item for item in native_submits(image_tap.records)
                          if str(item.get("request_id", "")).startswith(
                              main_record["request_id"] + ":slot:")]
    require(len(main_image_submits) == 2
            and main_image_submits[0]["request_id"].endswith(":slot:0")
            and main_image_submits[0]["image_sha256"] is None
            and main_image_submits[1]["request_id"].endswith(":slot:1")
            and main_image_submits[1]["image_sha256"] == source0["sha256"]
            == main_image_submits[1]["anchor_sha256"]
            and main_image_submits[1]["anchor_slot_index"] == 0
            and main_image_submits[1]["anchor_explicit"] is True
            and main_image_submits[1]["anchor_crop"] is None,
            "native FLUX edit did not consume exact slot-0 bytes")
    main_video_submits = native_submits(video_tap.records)
    require(len(main_video_submits) == 1
            and main_video_submits[0]["image_sha256"] == source1["sha256"]
            == main_video_submits[0]["anchor_sha256"]
            and main_video_submits[0]["anchor_slot_index"] == 1
            and main_video_submits[0]["anchor_crop"] == "center_cover",
            "native Wan did not consume exact slot-1 bytes")

    rgb = {}
    for label, path, selector, evaluation in (
            ("source0_full", source0_path, None, False),
            ("source1_full", source1_path, None, False),
            ("source1_eval", source1_path, None, True),
            ("first_full", video_path, "select=eq(n\\,0)", False),
            ("last_full", video_path, "select=eq(n\\,28)", False),
            ("first_eval", video_path, "select=eq(n\\,0)", True),
            ("last_eval", video_path, "select=eq(n\\,28)", True),
            ("native_recovery_full", out / "native-control-recovery.png", None, False)):
        width, height, raw = base.rgb_frame(args.ffmpeg, path, selector, evaluation)
        name = f"{label}.rgb{width}x{height}"
        (out / name).write_bytes(raw)
        rgb[label] = {"file": name, "width": width, "height": height,
                      **base.sha256_file(out / name)}
    require((rgb["source0_full"]["width"], rgb["source0_full"]["height"]) == (512, 512)
            and (rgb["source1_full"]["width"], rgb["source1_full"]["height"]) == (512, 512)
            and (rgb["native_recovery_full"]["width"],
                 rgb["native_recovery_full"]["height"]) == (512, 512),
            "decoded source/recovery dimensions differ from 512x512")
    first_mae = round(base.mean_absolute_rgb(
        (out / rgb["source1_eval"]["file"]).read_bytes(),
        (out / rgb["first_eval"]["file"]).read_bytes()), 6)
    motion_mae = round(base.mean_absolute_rgb(
        (out / rgb["first_eval"]["file"]).read_bytes(),
        (out / rgb["last_eval"]["file"]).read_bytes()), 6)
    require(first_mae <= base.FIRST_FRAME_MAX_MAE,
            f"first-frame MAE gate failed: {first_mae}")
    require(motion_mae >= base.MOTION_MIN_MAE, f"motion MAE gate failed: {motion_mae}")

    public_audio = []
    pcm = []
    for item in audio:
        raw = (main_recorder.artifact_dir / item["ref"]).read_bytes()
        public_audio.append((item, raw))
        require(len(raw) % 2 == 0 and raw, "invalid PCM16 artifact")
        samples = [int.from_bytes(raw[index:index + 2], "little", signed=True)
                   for index in range(0, len(raw), 2)]
        rms = (sum(sample * sample for sample in samples) / len(samples)) ** 0.5
        require(rms > 1.0, "TTS artifact is silent")
        pcm.append({"ref": item["ref"], "sha256": item["sha256"], "bytes": len(raw),
                    "sample_rate": item.get("sample_rate", 24000),
                    "channels": item.get("channels", 1), "rms": rms})
    prefix = main_record["request_id"] + ":tts"
    main_tts_submits = [item for item in native_submits(tts_tap.records)
                        if str(item.get("request_id", "")).startswith(prefix)]
    main_tts_output = [item for item in tts_output_tap.records
                       if str(item.get("request_id", "")).startswith(prefix)]
    tts_joins = base.join_tts_pcm(main_tts_submits, main_tts_output, public_audio, out)
    require(tts_joins and len(tts_joins) == len(main_tts_submits),
            "native/public TTS join is incomplete")
    tts_text = " ".join(item["text"] for item in main_tts_submits)
    require(base.spoken_text_projection(tts_text)
            == base.spoken_text_projection(main_record["clean_text"]),
            "TTS text differs from clean narrative prose")
    require("[[" not in tts_text and "source=" not in tts_text,
            "directive/source suffix leaked into TTS")

    event_counts = {}
    for event in main_record["events"]:
        event_counts[event["type"]] = event_counts.get(event["type"], 0) + 1
    report = {
        "format": REPORT_FORMAT, "result": "CONTROLLED_LIVE_PASS", "physical_gpu": 7,
        "base_commit": args.base_commit, "execution_commit": args.execution_commit,
        "requests_sha256": base.sha256_bytes(base.canonical(bodies)),
        "public_request_id": main_record["request_id"],
        "public_event_count": len(main_record["events"]),
        "public_event_counts": event_counts, "public_artifact_count": len(artifacts),
        "public_artifact_chunk_count": event_counts.get("artifact", 0),
        "raw_sse": {key: main_record["sse_replay"][key] for key in
                    ("raw_ref", "raw_sha256", "raw_bytes", "event_count", "done_frame")},
        "raw_transport_chunks": len(main_record["sse_replay"]["chunks"]),
        "raw_frames": len(main_record["sse_replay"]["frames"]),
        "source0": source0, "source1": source1, "video": video, "audio": pcm,
        "reference_chain": [
            {"dependent_slot": 1, "source_slot": 0, "source_sha256": source0["sha256"]},
            {"dependent_slot": 2, "source_slot": 1, "source_sha256": source1["sha256"]},
        ],
        "clean_text": main_record["clean_text"],
        "clean_text_sha256": main_record["clean_text_sha256"],
        "visual_slots": slots, "terminal_legs": main_record["terminal_legs"],
        "source_observations": {key: {k: v for k, v in main_observed[key].items()
                                      if k != "bytes"} for key in ("slot-0", "slot-1")},
        "image_submits": main_image_submits, "video_submit": main_video_submits[0],
        "native_controls": native_controls,
        "wrong_reference": {"terminal_state": wrong_record["terminal_state"],
                            "visual_slots": wrong_record["visual_slots"],
                            "terminal_legs": wrong_record["terminal_legs"],
                            "artifact_count": len(wrong_record["artifacts"]),
                            "native_video_submits": 0},
        "cancellation": {"request_id": cancel_record["request_id"],
                         "terminal_state": cancel_record["terminal_state"],
                         "visual_slots": cancel_record["visual_slots"],
                         "terminal_legs": cancel_record["terminal_legs"],
                         "visual_artifact_count": 0,
                         "native_image_submits": 1, "native_video_submits": 0,
                         "valid_retry_request_id": main_record["request_id"]},
        "tts_submits": main_tts_submits, "tts_output": main_tts_output,
        "tts_legs": tts_joins,
        "first_frame_mae": first_mae, "motion_mae": motion_mae,
        "first_frame_max_mae": base.FIRST_FRAME_MAX_MAE,
        "motion_min_mae": base.MOTION_MIN_MAE, "metric_version": base.METRIC_VERSION,
        "decoded_rgb": rgb, "video_stream": base.probe_video(args.ffprobe, video_path),
        "provenance": provenance, "models": models, "gpu7": gpu,
        "stage_lifecycle": stage_record, "manager": manager_record,
        "health_final": health_final,
        "captures": {name: {"record": f"{name}/trajectories.jsonl",
                            "artifact_dir": f"{name}/artifacts"}
                     for name in ("wrong", "cancel", "main")},
        "gpu_monitor": "gpu7-monitor.json", "native_video_log": "video.log",
        "native_tts_log": "tts.log", "native_image_log": "image.log",
        "server_trace": "server.trace.jsonl",
        "perceptual_quality": "NOT_RUN", "human_review": "NOT_RUN",
        "generalization": "NOT_COVERED", "resident_service": "NOT_CONTACTED",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    verify = subprocess.run([
        sys.executable, "-I", str(args.verifier), str(out), "--out",
        str(out / "independent-verification.json"), "--ffmpeg", args.ffmpeg,
        "--ffprobe", args.ffprobe], capture_output=True, text=True, timeout=600)
    require(verify.returncode == 0,
            f"fresh-process verification failed:\n{verify.stdout}\n{verify.stderr}")
    summary = {key: report[key] for key in (
        "format", "result", "public_event_count", "public_artifact_count",
        "public_artifact_chunk_count", "raw_sse", "source0", "source1", "video",
        "reference_chain", "first_frame_mae", "motion_mae", "wrong_reference",
        "cancellation", "perceptual_quality", "human_review", "generalization")}
    summary["independent_verification"] = "PASS"
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    files = {}
    for path in sorted(p for p in out.rglob("*") if p.is_file()
                       and p.name != "SHA256SUMS"
                       and "run" not in p.relative_to(out).parts):
        files[path.relative_to(out).as_posix()] = base.sha256_file(path)
    (out / "SHA256SUMS").write_text("".join(
        f"{item['sha256']}  {name}\n" for name, item in files.items()))
    print(json.dumps(summary, sort_keys=True))
    print("CROSS_SLOT_VISUAL_CONTINUITY: PASS")
    print("PERCEPTUAL_QUALITY: NOT_RUN")


def self_check():
    body = request_body("qwen3-native")
    require(body["narrative"]["max_slots"] == 3
            and body["narrative"]["continuity_mode"] == "i2v_chain",
            "three-slot continuity not opted in")
    require(IMAGE1.endswith("| source=slot-0]]")
            and VIDEO2.endswith("| source=slot-1]]"), "reference chain missing")
    require(request_body("qwen3-native", "cancel")["image"]["steps"] == 50,
            "cancellation window is not extended")
    require(DEFAULT_LIBRARY_PATH == "", "portable loader default must be empty")
    require(base.free_port() != 43301, "unexpected resident port allocation")
    try:
        base.exercise_pcm_join()
    except ModuleNotFoundError as error:
        require(error.name == "msgpack", f"unexpected self-check dependency failure: {error}")
        print("OFFLINE_NATIVE_PUBLIC_WAV_JOIN: NOT_RUN (msgpack unavailable)")
    print("OFFLINE_CROSS_SLOT: PASS (three slots, cancellation, native/public PCM controls)")
    print("REAL_GENERATION: NOT_RUN")
    print("PERCEPTUAL_QUALITY: NOT_RUN")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--out-dir")
    parser.add_argument("--server-binary", type=Path)
    parser.add_argument("--vlm-binary", type=Path)
    parser.add_argument("--tts-binary", type=Path)
    parser.add_argument("--wan-binary", type=Path)
    parser.add_argument("--flux-stage", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--vlm-model", type=Path)
    parser.add_argument("--flux-model", type=Path)
    parser.add_argument("--tts-model", type=Path)
    parser.add_argument("--wan-model", type=Path)
    parser.add_argument("--model", default="qwen3-native")
    parser.add_argument("--verifier", type=Path,
                        default=Path(__file__).with_name("cross_slot_visual_continuity_verify.py"))
    parser.add_argument("--base-commit", default="a4320e7678748a2336f74e3c2477c21b30805fb8")
    parser.add_argument("--execution-commit")
    parser.add_argument("--port", type=int)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--library-path", default=DEFAULT_LIBRARY_PATH,
                        help="optional loader prefix; default inherits environment/system loader")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    if args.out_dir is None:
        parser.error("live mode requires a caller-selected --out-dir")
    if args.execution_commit is None:
        parser.error("live mode requires --execution-commit")
    for name in ("server_binary", "vlm_binary", "tts_binary", "wan_binary", "flux_stage",
                 "vlm_model", "flux_model", "tts_model", "wan_model"):
        if getattr(args, name) is None:
            parser.error(f"live mode requires --{name.replace('_', '-')}")
    live(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
