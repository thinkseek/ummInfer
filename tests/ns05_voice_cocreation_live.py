#!/usr/bin/env python3
"""Real NS-05 voice-interrupted video -> FLUX -> native-I2V journey.

All paths are caller supplied. The same frozen real video accompanies actual
ASR voice turns. No transcript, narrative directive, frame budget, visual slot,
or direct-render fallback is injected. Physical acoustics remain NOT_RUN.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import wave

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
import public_narrative_i2v_live as common  # noqa: E402

SEED = 510075
SESSION = "ns05-video-voice-coc-v1"
EXPECTED_VIDEO = "ffef335edc161a2f6dead512e0fde96cf48fb997ae6b6491067e36c76eec58d0"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def write_json(path, value):
    Path(path).write_bytes(json.dumps(value, indent=2, sort_keys=True,
        ensure_ascii=False, allow_nan=False).encode() + b"\n")


def read_wav(path):
    with wave.open(str(path), "rb") as source:
        require((source.getnchannels(), source.getsampwidth(), source.getframerate()) ==
                (1, 2, 16000), f"{path} is not PCM16 mono 16 kHz")
        pcm = source.readframes(source.getnframes())
    require(pcm and len(pcm) % 2 == 0, f"empty/invalid PCM: {path}")
    return Path(path).read_bytes(), pcm


def body_for(video, wav, max_tokens):
    return {
        "model": "qwen3-native", "stream": True,
        "temperature": 0.0, "top_p": 1.0, "seed": SEED,
        "max_tokens": max_tokens, "session_id": SESSION, "session_history": True,
        "messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": {
                "url": "data:video/mp4;base64," + base64.b64encode(video).decode()}},
            {"type": "input_audio", "input_audio": {
                "data": base64.b64encode(wav).decode(), "format": "wav"}}]}],
        "modalities": ["text", "audio", "image", "video"],
        "image": {"size": "512x512", "steps": 4, "seed": SEED, "output_format": "png"},
        "audio": {"voice": "vivian", "format": "wav", "stream": True},
        "video": {"size": "704x480", "num_frames": 29, "steps": 20,
                  "fps": 16, "guidance_scale": 5.0, "seed": SEED, "output_format": "mp4"},
        "narrative": {"enabled": True, "visual_mode": "auto", "max_slots": 2,
                      "allow_text_only": True, "continuity_mode": "i2v_chain"},
    }


class Capture:
    def __init__(self, port, label, body, out, on_id=None, on_audio=None,
                 on_visual=None, on_image=None):
        self.port, self.label, self.body, self.out = port, label, body, out
        self.on_id, self.on_audio = on_id, on_audio
        self.on_visual, self.on_image = on_visual, on_image
        self.result = None
        self.error = None
        self.first_audio_ns = None
        self.first_text_ns = None
        self.request_id = None
        self.thread = threading.Thread(target=self.run, name=f"capture-{label}", daemon=True)

    def start(self):
        self.thread.start()
        return self

    def run(self):
        started = time.monotonic_ns()
        raw = bytearray()
        frames = []
        texts = []
        transcripts = []
        chunks = {}
        artifacts = []
        visual_slots = []
        terminal = None
        done = False
        inspected = 0
        capture = common.RawSSECapture(clock=common.PerfCounterClock(origin_ns=started))
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=900)
        try:
            conn.request("POST", "/v1/chat/completions", body=json.dumps(self.body),
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
            response = conn.getresponse()
            if response.status != 200:
                error_body = response.read()
                raise AssertionError(f"{self.label} HTTP {response.status}: {error_body!r}")
            while True:
                block = response.read1(65536)
                if not block:
                    break
                raw.extend(block)
                capture.feed(block)
                payloads = capture.payload_frames
                for frame_index, payload in payloads[inspected:]:
                    if payload == "[DONE]":
                        done = True
                        continue
                    frame = json.loads(payload)
                    frames.append(frame)
                    self.request_id = self.request_id or frame.get("id")
                    if self.request_id and self.on_id:
                        self.on_id(self.request_id)
                        self.on_id = None
                    for choice in frame.get("choices", []):
                        text = choice.get("delta", {}).get("content")
                        if isinstance(text, str) and text:
                            if self.first_text_ns is None:
                                self.first_text_ns = time.monotonic_ns()
                            texts.append(text)
                    unified = frame.get("unified") or {}
                    if unified.get("session_id") not in (None, SESSION):
                        raise AssertionError(f"cross-session event: {unified.get('session_id')}")
                    kind = unified.get("kind")
                    if kind == "asr_final" and unified.get("text"):
                        transcripts.append(unified["text"])
                    if kind == "visual_slot":
                        visual_slots.append(unified)
                        if self.on_visual:
                            self.on_visual(self.request_id)
                            self.on_visual = None
                    if kind in ("complete", "cancelled", "error"):
                        terminal = {"kind": kind, "error": unified.get("error"),
                                    "legs": unified.get("legs", [])}
                    if kind != "artifact":
                        continue
                    desc = unified["artifact"]
                    artifact_id = desc.get("artifact_id")
                    index, total = desc.get("index"), desc.get("total")
                    require(isinstance(artifact_id, str) and isinstance(index, int)
                            and isinstance(total, int) and total > 0, "invalid artifact descriptor")
                    group = chunks.setdefault(artifact_id, [])
                    require(index == len(group), f"out-of-order artifact {artifact_id}")
                    data = base64.b64decode(desc.get("data", ""), validate=True)
                    group.append((desc, data))
                    if desc.get("media_type") == "audio" and self.first_audio_ns is None:
                        self.first_audio_ns = time.monotonic_ns()
                        if self.on_audio:
                            self.on_audio(self.request_id)
                    if index + 1 != total:
                        continue
                    joined = b"".join(item[1] for item in group)
                    require(desc.get("bytes") == len(joined) and desc.get("sha256") == sha(joined),
                            f"artifact hash mismatch: {artifact_id}")
                    media_type = desc.get("media_type")
                    suffix = {"audio": ".pcm", "image": ".png", "video": ".mp4"}.get(media_type)
                    require(suffix is not None, f"unexpected media type {media_type}")
                    name = f"{self.label}-{len(artifacts):03d}-{re.sub(r'[^A-Za-z0-9_.-]', '_', artifact_id)}{suffix}"
                    (self.out / name).write_bytes(joined)
                    artifacts.append({"artifact_id": artifact_id, "file": name,
                        "media_type": media_type, "mime_type": desc.get("mime_type"),
                        "stream_index": desc.get("stream_index"), "slot_id": desc.get("slot_id"),
                        "slot_index": desc.get("slot_index"),
                        "anchor_slot_index": desc.get("anchor_slot_index"),
                        "anchor_explicit": desc.get("anchor_explicit", False),
                        "anchor_sha256": desc.get("anchor_sha256"),
                        "bytes": len(joined), "sha256": sha(joined), "descriptor": desc})
                    if media_type == "image" and self.on_image:
                        self.on_image(self.request_id, joined)
                inspected = len(payloads)
            capture.finish()
            require(done and self.request_id and terminal is not None,
                    f"{self.label} lacks DONE/request/terminal")
            self.result = {"label": self.label, "request_id": self.request_id,
                "started_monotonic_ns": started, "first_audio_monotonic_ns": self.first_audio_ns,
                "first_text_monotonic_ns": self.first_text_ns,
                "finished_monotonic_ns": time.monotonic_ns(), "terminal": terminal,
                "text": "".join(texts), "text_sha256": sha("".join(texts).encode()),
                "asr_transcripts": transcripts, "artifacts": artifacts,
                "visual_slots": visual_slots, "request_sha256": sha(canonical(self.body)),
                "raw_sse_sha256": sha(raw), "frames_sha256": sha(canonical(frames))}
            (self.out / f"{self.label}.sse").write_bytes(raw)
            write_json(self.out / f"{self.label}.frames.json", frames)
            write_json(self.out / f"{self.label}.json", self.result)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
            (self.out / f"{self.label}.sse").write_bytes(raw)
            write_json(self.out / f"{self.label}.error.json", {"error": self.error})
        finally:
            conn.close()

    def join(self):
        self.thread.join()
        if self.error:
            raise AssertionError(f"{self.label}: {self.error}")
        return self.result


def abort(port, request_id):
    payload = json.dumps({"request_id": request_id, "session_id": SESSION})
    request = urllib.request.Request(f"http://127.0.0.1:{port}/abort", data=payload.encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.load(response)
        require(response.status == 200 and body.get("success") is True, f"abort failed: {body}")


def health_zero(port, timeout=30):
    deadline = time.monotonic() + timeout
    latest = None
    while time.monotonic() < deadline:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
            latest = json.load(response)
        if latest.get("in_flight") == 0:
            return latest
        time.sleep(0.05)
    raise AssertionError(f"in_flight did not drain: {latest}")


def public_audio(result, out):
    rows = []
    for artifact in result["artifacts"]:
        if artifact["media_type"] == "audio":
            rows.append(({"stream_index": artifact["stream_index"],
                          "sha256": artifact["sha256"], "ref": artifact["file"]},
                         (out / artifact["file"]).read_bytes()))
    return rows


def pcm_joins(result, taps, out):
    prefix = result["request_id"] + ":tts:"
    submits = [row for row in taps["tts"].records
               if row.get("message_type") == "submit" and row.get("request_id", "").startswith(prefix)]
    require(submits, f"no native TTS submits for {result['label']}")
    return common.join_tts_pcm(submits, taps["tts-output"].records, public_audio(result, out), out)


def require_gpu_idle_stable(nvidia_smi, gpu):
    deadline = time.monotonic() + 30
    latest = None
    while time.monotonic() < deadline:
        try:
            return common.require_gpu_idle(nvidia_smi, gpu)
        except AssertionError as error:
            latest = error
            time.sleep(0.5)
    raise AssertionError(f"physical GPU {gpu} did not become stably idle: {latest}")


def run(args):
    require(os.environ.get("UMMINFER_LIVE_OPT_IN") == "1", "live opt-in required")
    require(args.gpu in (0, 7), "this gate permits only explicitly idle physical GPU 0 or 7")
    out = Path(args.out_dir).resolve()
    require(not out.exists(), "refusing to overwrite evidence")
    out.mkdir(parents=True)
    video = Path(args.video).read_bytes()
    require(sha(video) == EXPECTED_VIDEO, "frozen video identity mismatch")
    wavs = {name: read_wav(getattr(args, name))[0] for name in ("phone", "scene", "cyberpunk")}
    bodies = {"phone": body_for(video, wavs["phone"], 64),
              "scene": body_for(video, wavs["scene"], 64),
              "cyberpunk": body_for(video, wavs["cyberpunk"], 384)}
    require("frame_selection" not in json.dumps(bodies) and "[[" not in json.dumps(bodies),
            "manual frame/directive control leaked into request")
    write_json(out / "requests.json", bodies)
    write_json(out / "input-identity.json", {
        "video": {"bytes": len(video), "sha256": sha(video)},
        **{name: {"bytes": len(wavs[name]), "sha256": sha(wavs[name])} for name in wavs}})
    pre = require_gpu_idle_stable(args.nvidia_smi, args.gpu)
    monitor = common.GpuMonitor(args.nvidia_smi, args.gpu)
    run_dir = out / "run"
    run_dir.mkdir()
    endpoints = {name: f"ipc://{run_dir}/{name}" for name in
        ("completion", "abort", "entry-in", "vlm-in", "image-in", "tts-tap-in",
         "tts-in", "tts-output-in", "video-tap-in", "video-in")}
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["UMM_QWEN38_PREFIX_CACHE"] = "0"
    env["UMM_TRACE"] = "1"
    env["UMM_TRACE_FILE"] = str(out / "server.trace.jsonl")
    server = asr = vlm = tts = flux = wan = None
    taps = {}
    results = {}
    joins = {}
    barge_ns = None
    final_health = None
    error = None
    try:
        taps["video"] = common.TapProxy("video", endpoints["video-tap-in"], endpoints["video-in"], out)
        taps["tts"] = common.TapProxy("tts", endpoints["tts-tap-in"], endpoints["tts-in"], out)
        taps["tts-output"] = common.TapProxy(
            "tts-output", endpoints["tts-output-in"], endpoints["completion"], out)
        server = common.ManagedProcess("server", [args.server_binary, "serve",
            "--host", "127.0.0.1", "--port", str(args.port), "--model", "qwen3-native",
            "--entry", "entry", "--entry-endpoint", endpoints["entry-in"],
            "--vlm-entry", "vlm", "--vlm-entry-endpoint", endpoints["vlm-in"],
            "--image-entry", "image", "--image-entry-endpoint", endpoints["image-in"],
            "--tts-entry", "tts", "--tts-entry-endpoint", endpoints["tts-tap-in"],
            "--video-entry", "video", "--video-entry-endpoint", endpoints["video-tap-in"],
            "--completion", endpoints["completion"], "--abort", endpoints["abort"],
            "--session-history", "--max-in-flight", "4"], out / "server.log", env=env)
        asr = common.ManagedProcess("asr", [args.asr_binary, "--kind", "asr", "--name", "entry",
            "--model", args.asr_model, "--gpu", "0", "--recv-endpoint", endpoints["entry-in"],
            "--coordinator-endpoint", endpoints["completion"], "--abort-endpoint", endpoints["abort"],
            ], out / "asr.log", env=env)
        vlm = common.ManagedProcess("vlm", [args.vlm_binary, "--kind", "qwen38", "--name", "vlm",
            "--model", args.vlm_model, "--gpu", "0", "--batch-size", "1",
            "--recv-endpoint", endpoints["vlm-in"], "--coordinator-endpoint", endpoints["completion"],
            "--abort-endpoint", endpoints["abort"]], out / "vlm.log", env=env)
        tts = common.ManagedProcess("tts", [args.tts_binary, "--kind", "tts", "--name", "tts",
            "--model", args.tts_model, "--gpu", "0", "--recv-endpoint", endpoints["tts-in"],
            "--coordinator-endpoint", endpoints["tts-output-in"],
            "--abort-endpoint", endpoints["abort"]], out / "tts.log", env=env)
        for process, marker in ((asr, "ready kind=asr"), (vlm, "ready kind=qwen38"),
                                (tts, "native stage tts ready kind=tts")):
            process.wait_log(marker, args.timeout)
        flux = common.ManagedProcess("image", [args.python, "-I", args.flux_stage,
            "--name", "image", "--backend", "diffusers", "--model", args.flux_model,
            "--model-id", "black-forest-labs/FLUX.2-klein-4B", "--revision", "UNKNOWN",
            "--gpu", "0", "--default-size", "512x512", "--default-steps", "4",
            "--recv", endpoints["image-in"], "--coordinator", endpoints["completion"],
            "--abort", endpoints["abort"]], out / "image.log", env=env)
        flux.wait_log("ready kind=image name=image", args.timeout)

        results["phone"] = Capture(args.port, "phone", bodies["phone"], out).start().join()
        require(results["phone"]["terminal"]["kind"] == "complete", "phone turn failed")
        require(results["phone"]["asr_transcripts"], "phone voice was not transcribed")

        scene_audio = threading.Event()
        scene_id = [None]
        scene = Capture(args.port, "scene", bodies["scene"], out,
            on_id=lambda value: scene_id.__setitem__(0, value),
            on_audio=lambda _: scene_audio.set()).start()
        require(scene_audio.wait(args.timeout), "scene answer produced no voice")
        barge_ns = time.monotonic_ns()
        initial_id = [None]
        initial_id_ready = threading.Event()

        def record_initial_id(value):
            initial_id[0] = value
            initial_id_ready.set()

        initial = Capture(args.port, "cyberpunk-initial", bodies["cyberpunk"], out,
            on_id=record_initial_id,
            on_visual=lambda request_id: abort(args.port, request_id)).start()
        abort(args.port, scene_id[0])
        require(initial_id_ready.wait(args.timeout), "initial co-creation response id was not observed")
        write_json(out / "barge.json", {"barge_start_monotonic_ns": barge_ns,
            "interrupted_request_id": scene_id[0], "initial_request_id": initial_id[0]})
        results["scene"] = scene.join()
        require(results["scene"]["terminal"]["kind"] == "cancelled", "barge did not cancel old answer")
        results["cyberpunk-initial"] = initial.join()
        require(results["cyberpunk-initial"]["terminal"]["kind"] == "cancelled",
                "initial co-creation was not cancelled at its autonomous visual slot")
        health_zero(args.port)

        def start_wan_after_image(_, image_bytes):
            require(image_bytes, "empty retry FLUX image")
            flux.stop(endpoints["image-in"])
            require(flux.returncode == 0, "FLUX did not stop cleanly")
            nonlocal_wan[0] = common.ManagedProcess("video", [args.wan_binary,
                "--kind", "video", "--name", "video", "--video-backend", "cuda",
                "--model", args.wan_model, "--gpu", "0", "--size", "704x480",
                "--steps", "20", "--fps", "16", "--recv-endpoint", endpoints["video-in"],
                "--coordinator-endpoint", endpoints["completion"],
                "--abort-endpoint", endpoints["abort"]], out / "video.log", env=env)
            nonlocal_wan[0].wait_log("native stage video ready kind=video", args.timeout)

        nonlocal_wan = [None]
        retry = Capture(args.port, "cyberpunk-retry", bodies["cyberpunk"], out,
                        on_image=start_wan_after_image).start()
        results["cyberpunk-retry"] = retry.join()
        wan = nonlocal_wan[0]
        require(results["cyberpunk-retry"]["terminal"]["kind"] == "complete",
                "retry did not complete")
        final_health = health_zero(args.port)
        for label in ("phone", "cyberpunk-retry"):
            joins[label] = pcm_joins(results[label], taps, out)
        require(results["cyberpunk-retry"]["asr_transcripts"], "retry voice was not transcribed")
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
    finally:
        for process, endpoint in ((wan, endpoints["video-in"]), (flux, endpoints["image-in"]),
                                  (tts, endpoints["tts-in"]), (vlm, endpoints["vlm-in"]),
                                  (asr, endpoints["entry-in"])):
            if process is not None and process.stop_ns is None:
                process.stop(endpoint)
        if server is not None:
            server.stop(None)
        for tap in taps.values():
            if tap.thread.is_alive():
                tap.stop()
        monitor.stop()
        post = require_gpu_idle_stable(args.nvidia_smi, args.gpu)
        write_json(out / "cleanup.json", {"pre": pre, "post": post,
            "processes": [process.record() for process in (server, asr, vlm, tts, flux, wan)
                          if process is not None], "in_flight": final_health})
    if error:
        write_json(out / "error.json", {"error": error})
        raise AssertionError(error)

    retry_result = results["cyberpunk-retry"]
    images = [row for row in retry_result["artifacts"] if row["media_type"] == "image"]
    videos = [row for row in retry_result["artifacts"] if row["media_type"] == "video"]
    require(len(images) == 1 and len(videos) == 1, "retry lacks exactly one styled image and I2V")
    image, video_row = images[0], videos[0]
    require(video_row["anchor_sha256"] == image["sha256"], "I2V is not bound to exact FLUX bytes")
    video_submits = [row for row in taps["video"].records if row.get("message_type") == "submit"
        and row.get("request_id", "").startswith(retry_result["request_id"] + ":slot:")]
    require(len(video_submits) == 1 and video_submits[0]["image_sha256"] == image["sha256"]
        and video_submits[0]["anchor_sha256"] == image["sha256"]
        and video_submits[0]["anchor_crop"] == "center_cover",
        "native Wan source binding/center-cover failed")
    require("backend=native-cuda-wan" in (out / "video.log").read_text(errors="replace")
        and "i2v=1" in (out / "video.log").read_text(errors="replace"), "native Wan I2V not observed")
    require("[[" not in retry_result["text"] and "]]" not in retry_result["text"]
        and retry_result["text"].strip(), "directive leak or empty narration")
    stream = common.probe_video(args.ffprobe, out / video_row["file"])
    source_eval = common.rgb_frame(args.ffmpeg, out / image["file"], evaluation=True)
    first_eval = common.rgb_frame(args.ffmpeg, out / video_row["file"], "select=eq(n\\,0)", True)
    last_eval = common.rgb_frame(args.ffmpeg, out / video_row["file"], "select=eq(n\\,28)", True)
    first_mae = common.mean_absolute_rgb(source_eval[2], first_eval[2])
    motion_mae = common.mean_absolute_rgb(first_eval[2], last_eval[2])
    require(first_mae <= 48.0 and motion_mae >= 1.0, "first-frame/motion proxy failed")
    full_barge_ms = (retry_result["first_audio_monotonic_ns"] - barge_ns) / 1e6
    report = {"schema_version": 1, "experiment": "umminfer_ns05_duration_budget_voice_cocreation_v1",
        "execution_commit": args.execution_commit, "physical_gpu": args.gpu,
        "source": {"sha256": EXPECTED_VIDEO, "bytes": len(video)},
        "requests": results, "native_pcm_joins": joins,
        "native_video_submit": video_submits[0], "decoded_video": stream,
        "first_frame_mae": first_mae, "motion_mae": motion_mae,
        "barge_start_monotonic_ns": barge_ns,
        "full_barge_to_retry_first_audio_ms": full_barge_ms,
        "timing_ms": {
            "phone_first_text": (results["phone"]["first_text_monotonic_ns"] - results["phone"]["started_monotonic_ns"]) / 1e6,
            "phone_first_audio": (results["phone"]["first_audio_monotonic_ns"] - results["phone"]["started_monotonic_ns"]) / 1e6,
            "retry_first_text": (retry_result["first_text_monotonic_ns"] - retry_result["started_monotonic_ns"]) / 1e6,
            "retry_first_audio": (retry_result["first_audio_monotonic_ns"] - retry_result["started_monotonic_ns"]) / 1e6,
            "barge_to_scene_terminal": (results["scene"]["finished_monotonic_ns"] - barge_ns) / 1e6,
            "barge_to_initial_terminal": (results["cyberpunk-initial"]["finished_monotonic_ns"] - barge_ns) / 1e6,
            "full_barge_to_retry_first_audio": full_barge_ms,
        },
        "response_identity_unique": len({row["request_id"] for row in results.values()}) == len(results),
        "no_stale_or_cross_session_events": True, "final_in_flight": final_health["in_flight"],
        "physical_acoustic_and_perceptual": "NOT_RUN", "status": "PASS"}
    require(report["response_identity_unique"] and report["final_in_flight"] == 0,
            "response identity/lifecycle failed")
    write_json(out / "report.json", report)
    print(json.dumps({"status": "PASS", "first_frame_mae": first_mae,
        "motion_mae": motion_mae, "full_barge_to_retry_first_audio_ms": full_barge_ms,
        "image_sha256": image["sha256"], "video_sha256": video_row["sha256"]}, indent=2))
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("out-dir", "video", "phone", "scene", "cyberpunk", "server-binary",
                 "asr-binary", "vlm-binary", "tts-binary", "wan-binary", "flux-stage",
                 "asr-model", "vlm-model", "tts-model", "wan-model", "flux-model",
                 "python", "execution-commit"):
        result.add_argument("--" + name, required=True)
    result.add_argument("--port", type=int, required=True)
    result.add_argument("--gpu", type=int, default=0)
    result.add_argument("--timeout", type=float, default=600)
    result.add_argument("--ffmpeg", default="ffmpeg")
    result.add_argument("--ffprobe", default="ffprobe")
    result.add_argument("--nvidia-smi", default="nvidia-smi")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return run(args)
    except (OSError, AssertionError, ValueError, subprocess.SubprocessError) as error:
        print(f"NS05 COCREATION FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
