#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture frozen NS-01 opening pairs on an explicitly idle physical GPU 7.

The capture never converts missing media, terminal failures, or lifecycle errors
into a PASS. Raw SSE, native MessagePack, CAS, logs, and a compact arm record are
retained externally. A separate scorecard/verifier makes every promotion decision.
"""
import argparse
import hashlib
import http.client
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))
import public_narrative_i2v_live as base  # noqa: E402
from umminfer_rl_trajectory import (  # noqa: E402
    TrajectoryRecorder, input_references, parse_sse_stream,
    request_parameter_references,
)
from umminfer_sse_replay import PerfCounterClock, RawSSECapture  # noqa: E402


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def clock_ns():
    return time.perf_counter_ns()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                    ensure_ascii=False, allow_nan=False) + "\n")


def runtime_identity():
    result = {}
    for name in ("torch", "diffusers", "transformers", "huggingface-hub",
                 "tokenizers", "safetensors", "accelerate", "numpy", "pillow",
                 "pyzmq", "msgpack"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "ABSENT"
    return result


def request_body(prereg, case_name):
    case = prereg["golden_cases"][case_name]
    request = prereg["request"]
    audio = {key: value for key, value in request["audio"].items()
             if key in ("voice", "format", "stream")}
    return {
        "model": "qwen3-native", "stream": request["stream"],
        "messages": [{"role": "user", "content": case["user_text"]}],
        "modalities": request["modalities"], "image": request["image"],
        "audio": audio, "video": request["video"],
        "narrative": request["narrative"],
        "max_tokens": request["max_tokens"],
        "temperature": request["temperature"], "seed": request["seed"],
        "session_id": case["scenario"],
    }


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    require(port != 43301, "resident service port is forbidden")
    return port


def http_json(port, method, path, body=None, timeout=30):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        payload = None if body is None else json.dumps(body)
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        conn.request(method, path, body=payload, headers=headers)
        response = conn.getresponse()
        raw = response.read()
        return response.status, json.loads(raw) if raw else {}
    finally:
        conn.close()


def stable_idle(nvidia_smi, gpu):
    samples = []
    stable = 0
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and stable < 3:
        observed = base.gpu_memory(nvidia_smi, gpu)
        samples.append(observed)
        stable = stable + 1 if observed["memory_used_mib"] <= 16 \
            and observed["utilization_percent"] == 0 else 0
        if stable < 3:
            time.sleep(0.5)
    require(stable == 3, f"physical GPU {gpu} is not stably idle: {samples[-1]}")
    return samples


def capture_public(port, body, recorder, source_ready, video_slot_seen,
                   request_ready, id_out):
    params, option_inputs = request_parameter_references(body, recorder.artifact_dir)
    inputs = input_references(body["messages"], recorder.artifact_dir) + option_inputs
    origin_ns = clock_ns()
    capture = RawSSECapture(clock=PerfCounterClock(origin_ns=origin_ns))
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    inspected = 0
    source_final = None
    audio_streams = set()
    first_ticks = {name: None for name in (
        "ttft_ns", "last_vlm_delta_ns", "first_visual_slot_ns", "image_slot_ns",
        "video_slot_ns", "ttfa_ns", "first_image_ns", "video_first_public_chunk_ns",
        "video_public_complete_ns")}
    request_id = None
    raw_live = (recorder.artifact_dir.parent / "raw-live.sse").open("wb")
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
            raw_live.write(chunk)
            raw_live.flush()
            payloads = capture.payload_frames
            for frame_index, payload in payloads[inspected:]:
                if payload == "[DONE]":
                    continue
                frame = json.loads(payload)
                request_id = request_id or frame.get("id")
                if request_id is not None:
                    id_out[0] = request_id
                    request_ready.set()
                unified = frame.get("unified") or {}
                tick = capture._frames[frame_index]["tick"]
                kind = unified.get("kind")
                if kind == "vlm_delta" and unified.get("text"):
                    if first_ticks["ttft_ns"] is None:
                        first_ticks["ttft_ns"] = tick
                    first_ticks["last_vlm_delta_ns"] = tick
                elif kind == "visual_slot":
                    slot_id = unified.get("slot_id")
                    if first_ticks["first_visual_slot_ns"] is None:
                        first_ticks["first_visual_slot_ns"] = tick
                    if unified.get("stage") == "image" or slot_id == "slot-0":
                        first_ticks["image_slot_ns"] = tick
                    if unified.get("stage") == "video" or slot_id == "slot-1":
                        first_ticks["video_slot_ns"] = tick
                        video_slot_seen.set()
                elif kind == "artifact":
                    desc = unified["artifact"]
                    media_type = desc.get("media_type")
                    if media_type == "audio":
                        audio_streams.add(desc.get("stream_index"))
                        if first_ticks["ttfa_ns"] is None:
                            first_ticks["ttfa_ns"] = tick
                    elif media_type == "video":
                        if first_ticks["video_first_public_chunk_ns"] is None:
                            first_ticks["video_first_public_chunk_ns"] = tick
                        if desc.get("index") + 1 == desc.get("total"):
                            first_ticks["video_public_complete_ns"] = tick
                    elif media_type == "image" and desc.get("slot_id") == "slot-0" \
                            and desc.get("index") + 1 == desc.get("total"):
                        first_ticks["first_image_ns"] = tick
                        source_final = {"frame_index": frame_index,
                                        "observed_monotonic_ns": tick,
                                        "observed_absolute_monotonic_ns": clock_ns(),
                                        "seq": unified.get("seq"),
                                        "sha256": desc.get("sha256")}
                        source_ready.set()
            inspected = len(payloads)
        capture.finish()
        details = {}
        request_id, events, output_text, state, error = parse_sse_stream(
            io.BytesIO(capture.raw_bytes), artifact_dir=recorder.artifact_dir,
            details=details, capture=capture)
        record = recorder.record(
            session_id=body["session_id"], request_id=request_id,
            endpoint="/v1/chat/completions", inputs=inputs, params=params,
            events=events, terminal_state=state, error=error,
            output_text=output_text, artifacts=details.get("artifacts", []), capture=capture)
        timings = {key: (None if value is None else value / 1e6)
                   for key, value in first_ticks.items()}
        timings.update({"unit": "ms_from_capture_start",
                        "clock": "python.time.perf_counter_ns",
                        "capture_origin_monotonic_ns": origin_ns})
        return record, source_final, sorted(audio_streams), timings
    finally:
        conn.close()
        raw_live.close()


def model_identity(args):
    if args.model_identity is not None:
        return json.loads(args.model_identity.read_text())
    return {
        "vlm": base.model_identity(args.vlm_model, "Qwen/Qwen3.8-27B"),
        "flux2": base.model_identity(args.flux_model, "black-forest-labs/FLUX.2-klein-4B"),
        "tts": base.model_identity(args.tts_model, "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        "wan22": base.model_identity(args.wan_model, "Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
    }


def live(args):
    for name in ("preregistration", "out_dir", "server_binary", "vlm_binary", "tts_binary",
                 "wan_binary", "flux_stage", "vlm_model", "flux_model", "tts_model",
                 "wan_model", "model_identity"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, Path(value).resolve())
    # Resolve the location without following a virtualenv's interpreter symlink.
    args.python = os.path.abspath(args.python)
    require(os.environ.get("UMMINFER_LIVE_OPT_IN") == "1",
            "REAL_GENERATION: NOT_RUN (explicit UMMINFER_LIVE_OPT_IN=1 required)")
    require(args.gpu == 7, "Task 87 permits only explicitly idle physical GPU 7")
    require(args.case in ("chinese", "english"), "unknown frozen case")
    require(args.arm in ("before", "after"), "unknown arm")
    require(len(args.base_commit) == 40 and len(args.execution_commit) == 40,
            "full base and execution commits are required")
    prereg = json.loads(args.preregistration.read_text())
    require(prereg["format"] == "umminfer-ns01-streaming-opening-preregistration-v1",
            "unexpected preregistration format")
    require(prereg["official_base_commit"] == args.base_commit,
            "execution base differs from frozen official base")
    body = request_body(prereg, args.case)
    out = args.out_dir
    require(not out.exists(), f"evidence directory already exists: {out}")
    out.mkdir(parents=True)
    (out / "run").mkdir()
    (out / "capture").mkdir()
    write_json(out / "request.json", body)
    write_json(out / "preregistration.json", prereg)
    models = model_identity(args)
    write_json(out / "model-identity.json", models)
    provenance = {
        "base_commit": args.base_commit, "execution_commit": args.execution_commit,
        "case": args.case, "repetition": args.repetition, "arm": args.arm,
        "preregistration_sha256": base.sha256_file(args.preregistration)["sha256"],
        "server_binary": {"path": str(args.server_binary), **base.sha256_file(args.server_binary)},
        "vlm_binary": {"path": str(args.vlm_binary), **base.sha256_file(args.vlm_binary)},
        "tts_binary": {"path": str(args.tts_binary), **base.sha256_file(args.tts_binary)},
        "wan_binary": {"path": str(args.wan_binary), **base.sha256_file(args.wan_binary)},
        "flux_stage": {"path": str(args.flux_stage), **base.sha256_file(args.flux_stage)},
        "live_harness": {"path": str(Path(__file__).resolve()), **base.sha256_file(__file__)},
        "ffmpeg": base.sha256_file(shutil.which(args.ffmpeg)),
        "ffprobe": base.sha256_file(shutil.which(args.ffprobe)),
        "python_runtime": runtime_identity(),
    }
    write_json(out / "provenance.json", provenance)
    gpu = {"before": base.require_gpu_idle(args.nvidia_smi, args.gpu)}
    monitor = base.GpuMonitor(args.nvidia_smi, args.gpu)
    port = args.port or free_port()
    require(port != 43301, "resident service port is forbidden")
    endpoints = {name: f"ipc://{out}/run/{name}" for name in (
        "completion", "abort", "entry-in", "vlm-in", "image-in", "tts-tap-in",
        "tts-in", "tts-output-in", "video-tap-in", "video-in")}
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if args.library_path:
        env["LD_LIBRARY_PATH"] = str(args.library_path) + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["UMM_TRACE"] = "1"
    env["UMM_TRACE_FILE"] = str(out / "server.trace.jsonl")
    server = entry = vlm = flux = tts = wan = None
    video_tap = tts_tap = tts_output_tap = None
    source_ready = threading.Event()
    video_slot_seen = threading.Event()
    request_ready = threading.Event()
    capture_done = threading.Event()
    manager_error = []
    manager_record = {"flux_to_tts": False, "tts_to_wan": False,
                      "wan_required": None}
    stage_record = {}
    recorder = TrajectoryRecorder(out / "capture" / "trajectories.jsonl")
    record = source_final = audio_streams = timings = None
    health_final = None
    request_id = [None]

    def start_tts():
        nonlocal tts
        tts = base.ManagedProcess("tts", [
            args.tts_binary, "--kind", "tts", "--name", "tts", "--model", args.tts_model,
            "--gpu", str(args.gpu), "--recv-endpoint", endpoints["tts-in"],
            "--coordinator-endpoint", endpoints["tts-output-in"],
            "--abort-endpoint", endpoints["abort"]], out / "tts.log", env=env)
        tts.wait_log("native stage tts ready kind=tts", args.timeout)
        stage_record["tts_ready_monotonic_ns"] = clock_ns()
        gpu["tts_ready"] = base.gpu_memory(args.nvidia_smi, args.gpu)

    def start_wan():
        nonlocal wan
        wan = base.ManagedProcess("video", [
            args.wan_binary, "--kind", "video", "--name", "video", "--video-backend", "cuda",
            "--model", args.wan_model, "--gpu", str(args.gpu), "--size", "704x480",
            "--steps", "20", "--fps", "16", "--recv-endpoint", endpoints["video-in"],
            "--coordinator-endpoint", endpoints["completion"],
            "--abort-endpoint", endpoints["abort"]], out / "video.log", env=env)
        wan.wait_log("native stage video ready kind=video", args.timeout)
        stage_record["wan_ready_monotonic_ns"] = clock_ns()
        gpu["wan_ready"] = base.gpu_memory(args.nvidia_smi, args.gpu)

    def output_manager():
        try:
            deadline = time.monotonic() + args.timeout
            while not request_ready.wait(0.005):
                # The native parent id is a read-only ownership fallback for stage
                # lifecycle management; public timing/identity still comes from SSE.
                fallback = next((item["request_id"].split(":tts", 1)[0]
                                 for item in tts_tap.records
                                 if item.get("request_id") and ":tts" in item["request_id"]), None)
                if fallback is not None:
                    request_id[0] = fallback
                    manager_record["native_parent_id_fallback"] = True
                    request_ready.set()
                    break
                require(time.monotonic() < deadline and not capture_done.is_set(),
                        "no public or native-parent request id observed")
            while not source_ready.wait(0.05):
                require(not capture_done.is_set(),
                        "public stream ended before a source image completed")
            stage_record["source_public_complete_ns"] = clock_ns()
            flux.stop(endpoints["image-in"])
            require(flux.returncode == 0, f"FLUX shutdown failed: {flux.returncode}")
            stage_record["flux_stopped_ns"] = clock_ns()
            gpu["after_flux_stop_vlm_resident"] = base.gpu_memory(args.nvidia_smi, args.gpu)
            start_tts()
            manager_record["flux_to_tts"] = True
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                traces = base.trace_records(out / "server.trace.jsonl")
                outputs_started = any(
                    row.get("event") == "legs_snapshot" and row.get("phase") == "outputs_started"
                    and row.get("request_id") == request_id[0] for row in traces)
                state = base.coordinator_stage_state(port, "tts")
                settled = state["completed"] + state["failed"] + state["aborted"]
                if outputs_started and state["submitted"] > 0 and settled == state["submitted"]:
                    stage_record["tts_coordinator_complete"] = state
                    break
                require(tts.process.poll() is None, "TTS exited before streaming legs settled")
                time.sleep(0.02)
            else:
                raise TimeoutError("TTS streaming legs did not settle after outputs_started")
            tts.stop(endpoints["tts-in"])
            require(tts.returncode == 0, f"TTS shutdown failed: {tts.returncode}")
            tts_tap.stop()
            tts_output_tap.stop()
            stage_record["tts_stopped_ns"] = clock_ns()
            gpu["after_tts_stop_vlm_resident"] = base.gpu_memory(args.nvidia_smi, args.gpu)
            manager_record["wan_required"] = video_slot_seen.is_set()
            if video_slot_seen.is_set():
                start_wan()
                manager_record["tts_to_wan"] = True
        except Exception as error:
            manager_error.append(str(error))
            if request_id[0] is not None:
                try:
                    http_json(port, "POST", "/abort", {
                        "request_id": request_id[0], "session_id": body["session_id"]}, timeout=30)
                except Exception:
                    pass
            source_ready.set()

    capture_error = None
    try:
        video_tap = base.TapProxy("video", endpoints["video-tap-in"], endpoints["video-in"], out)
        tts_tap = base.TapProxy("tts", endpoints["tts-tap-in"], endpoints["tts-in"], out)
        tts_output_tap = base.TapProxy(
            "tts-output", endpoints["tts-output-in"], endpoints["completion"], out)
        server = base.ManagedProcess("server", [
            args.server_binary, "serve", "--host", "127.0.0.1", "--port", str(port),
            "--model", "qwen3-native", "--entry", "entry",
            "--entry-endpoint", endpoints["entry-in"], "--vlm-entry", "vlm",
            "--vlm-entry-endpoint", endpoints["vlm-in"],
            "--image-entry", "image", "--image-entry-endpoint", endpoints["image-in"],
            "--tts-entry", "tts", "--tts-entry-endpoint", endpoints["tts-tap-in"],
            "--video-entry", "video", "--video-entry-endpoint", endpoints["video-tap-in"],
            "--completion", endpoints["completion"], "--abort", endpoints["abort"],
            "--max-in-flight", "1"], out / "server.log", env=env)
        entry = base.ManagedProcess("entry", [
            args.server_binary, "stage", "--name", "entry", "--recv", endpoints["entry-in"],
            "--coordinator", endpoints["completion"], "--abort", endpoints["abort"]],
            out / "entry.log", env=env)
        vlm = base.ManagedProcess("vlm", [
            args.vlm_binary, "--kind", "qwen38", "--name", "vlm", "--model", args.vlm_model,
            "--gpu", str(args.gpu), "--batch-size", "1", "--max-new-tokens", "768",
            "--recv-endpoint", endpoints["vlm-in"], "--coordinator-endpoint", endpoints["completion"],
            "--abort-endpoint", endpoints["abort"]], out / "vlm.log", env=env)
        vlm.wait_log("native stage vlm ready kind=qwen38", args.timeout)
        stage_record["vlm_ready_monotonic_ns"] = clock_ns()
        gpu["vlm_ready"] = base.gpu_memory(args.nvidia_smi, args.gpu)
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
        flux = base.ManagedProcess("image", [
            args.python, "-I", args.flux_stage, "--name", "image", "--backend", "diffusers",
            "--model", args.flux_model, "--model-id", "black-forest-labs/FLUX.2-klein-4B",
            "--revision", "UNKNOWN", "--gpu", str(args.gpu), "--default-size", "512x512",
            "--default-steps", "4", "--recv", endpoints["image-in"],
            "--coordinator", endpoints["completion"], "--abort", endpoints["abort"]],
            out / "image.log", env=env)
        flux.wait_log("ready kind=image name=image", args.timeout)
        stage_record["flux_ready_monotonic_ns"] = clock_ns()
        gpu["flux_ready"] = base.gpu_memory(args.nvidia_smi, args.gpu)
        manager = threading.Thread(target=output_manager, name="output-manager", daemon=True)
        manager.start()
        try:
            record, source_final, audio_streams, timings = capture_public(
                port, body, recorder, source_ready, video_slot_seen, request_ready, request_id)
        except Exception as error:
            capture_error = str(error)
        finally:
            capture_done.set()
        manager.join(timeout=args.timeout)
        require(not manager.is_alive(), "output manager did not exit")
        health_status, health_final = http_json(port, "GET", "/health", timeout=30)
        require(health_status == 200, f"health returned {health_status}")
    finally:
        for process, endpoint in ((wan, endpoints["video-in"]), (tts, endpoints["tts-in"]),
                                  (flux, endpoints["image-in"]), (vlm, endpoints["vlm-in"]),
                                  (entry, endpoints["entry-in"])):
            if process is not None and process.stop_ns is None:
                process.stop(endpoint)
        if server is not None:
            server.stop(None)
        for tap in (video_tap, tts_tap, tts_output_tap):
            if tap is not None and tap.thread.is_alive():
                tap.stop()
        monitor.stop()
        gpu["after_cleanup_samples"] = stable_idle(args.nvidia_smi, args.gpu)
        stage_record["processes"] = [process.record() for process in
                                     (server, entry, vlm, flux, tts, wan) if process is not None]
        write_json(out / "gpu-monitor.json", monitor.samples)

    failures = list(manager_error)
    if capture_error:
        failures.append(capture_error)
    if record is None:
        write_json(out / "arm.json", {
            "format": "umminfer-ns01-streaming-opening-arm-v1", "result": "CAPTURE_FAILED",
            "case": args.case, "repetition": args.repetition, "arm": args.arm,
            "request_sha256": hashlib.sha256(canonical(body)).hexdigest(),
            "failures": failures, "manager": manager_record, "gpu": gpu,
            "stage_lifecycle": stage_record, "health_final": health_final,
            "perceptual_quality": "NOT_RUN"})
        raise AssertionError(f"public capture failed; raw evidence retained: {failures}")
    write_json(out / "public-record.json", record)
    artifacts = record["artifacts"]
    images = [item for item in artifacts if item["media_type"] == "image"]
    videos = [item for item in artifacts if item["media_type"] == "video"]
    audio = [item for item in artifacts if item["media_type"] == "audio"]
    public_audio = []
    for item in audio:
        raw = (recorder.artifact_dir / item["ref"]).read_bytes()
        public_audio.append((item, raw))
    try:
        tts_joins = base.join_tts_pcm(tts_tap.records, tts_output_tap.records,
                                      public_audio, out)
    except Exception as error:
        tts_joins = []
        failures.append(f"PCM join: {error}")
    for tap in (video_tap, tts_tap, tts_output_tap):
        if tap.error:
            failures.append(f"{tap.name}: {tap.error}")
    if health_final is None or health_final.get("in_flight") != 0:
        failures.append(f"final health is not settled: {health_final}")
    video_stream = None
    if videos:
        try:
            video_stream = base.probe_video(
                args.ffprobe, recorder.artifact_dir / videos[0]["ref"])
        except Exception as error:
            failures.append(f"video decode: {error}")
    first_tts_submit_ms = None
    if tts_tap.records and timings:
        first_tts_submit_ms = (
            tts_tap.records[0]["observed_monotonic_ns"] -
            timings["capture_origin_monotonic_ns"]) / 1e6
    arm = {
        "format": "umminfer-ns01-streaming-opening-arm-v1",
        "result": "CAPTURED_WITH_FAILURES" if failures else "CAPTURED",
        "case": args.case, "repetition": args.repetition, "arm": args.arm,
        "base_commit": args.base_commit, "execution_commit": args.execution_commit,
        "request_sha256": hashlib.sha256(canonical(body)).hexdigest(),
        "public_request_id": record["request_id"], "terminal_state": record["terminal_state"],
        "terminal_error": record["error"], "failures": failures,
        "clean_text": record["clean_text"], "clean_text_sha256": record["clean_text_sha256"],
        "visual_slots": record["visual_slots"], "terminal_legs": record["terminal_legs"],
        "public_event_count": len(record["events"]), "public_artifact_count": len(artifacts),
        "source": images[0] if images else None, "video": videos[0] if videos else None,
        "audio_artifact_count": len(audio), "audio_stream_indices": audio_streams,
        "interactive_timings": timings, "first_tts_submit_ms": first_tts_submit_ms,
        "source_final": source_final, "video_stream": video_stream,
        "video_submits": video_tap.records, "tts_submits": tts_tap.records,
        "tts_legs": tts_joins, "manager": manager_record,
        "stage_lifecycle": stage_record, "gpu": gpu, "health_final": health_final,
        "raw_sse": {key: record["sse_replay"][key] for key in
                    ("raw_ref", "raw_sha256", "raw_bytes", "event_count", "done_frame")},
        "provenance": "provenance.json", "models": "model-identity.json",
        "public_record": "public-record.json", "perceptual_quality": "NOT_RUN",
        "asr": "NOT_APPLICABLE_TEXT_INPUT",
    }
    write_json(out / "arm.json", arm)
    write_json(out / "native-taps.json", {
        "video": video_tap.records, "tts": tts_tap.records,
        "tts_output": tts_output_tap.records})
    files = {}
    for path in sorted(p for p in out.rglob("*") if p.is_file()
                       and p.name != "SHA256SUMS" and "run" not in p.relative_to(out).parts):
        files[path.relative_to(out).as_posix()] = base.sha256_file(path)
    (out / "SHA256SUMS").write_text("".join(
        f"{item['sha256']}  {name}\n" for name, item in files.items()))
    print(json.dumps({key: arm[key] for key in
                      ("result", "case", "repetition", "arm", "terminal_state",
                       "clean_text_sha256", "interactive_timings", "first_tts_submit_ms",
                       "source", "video", "failures")}, sort_keys=True))
    print("NS01_STREAMING_OPENING_CAPTURE: RETAINED (scorecard decides PASS/FAIL)")


def self_check(args):
    prereg = json.loads(args.preregistration.read_text())
    chinese = request_body(prereg, "chinese")
    english = request_body(prereg, "english")
    require(chinese["messages"][0]["content"].startswith(
        "用通俗方式讲清黑洞吸积盘和引力透镜，配示意图和动态演示"), "Chinese request drift")
    require(english["messages"][0]["content"].startswith("Explain black-hole"), "English request drift")
    for body in (chinese, english):
        require(body["max_tokens"] == 768 and body["temperature"] == 0
                and body["narrative"]["visual_mode"] == "auto"
                and body["narrative"]["continuity_mode"] == "i2v_chain"
                and "teacache_thresh" not in body["video"]
                and "[[" not in body["messages"][0]["content"], "request contract drift")
    print("NS01_STREAMING_OPENING_REQUESTS: PASS")
    print("REAL_GENERATION: NOT_RUN")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--preregistration", type=Path,
                        default=ROOT / "benchmarks" / "ns01-streaming-opening" / "preregistration.json")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--case", choices=("chinese", "english"))
    parser.add_argument("--arm", choices=("before", "after"))
    parser.add_argument("--repetition", type=int, choices=(1, 2))
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
    parser.add_argument("--model-identity", type=Path)
    parser.add_argument("--base-commit")
    parser.add_argument("--execution-commit")
    parser.add_argument("--port", type=int)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--library-path", default="")
    args = parser.parse_args()
    if args.self_check:
        self_check(args)
        return 0
    required = ("out_dir", "case", "arm", "repetition", "server_binary", "vlm_binary",
                "tts_binary", "wan_binary", "flux_stage", "vlm_model", "flux_model",
                "tts_model", "wan_model", "base_commit", "execution_commit")
    for name in required:
        if getattr(args, name) is None:
            parser.error(f"live mode requires --{name.replace('_', '-')}")
    live(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
