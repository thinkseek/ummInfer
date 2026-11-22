#!/usr/bin/env python3
"""Offline end-to-end contract test for the native C++ video stage.

Starts real `umm-infer serve --demo --video-entry video` and real C++
`umm-native-stage-video --kind video --video-backend contract-stub` over
ZeroMQ+MessagePack IPC and verifies:
  1. Default `--video-backend cuda` refuses silent fallback on non-CUDA builds.
  2. Completed video leg: SSE gap-free seq, session_id, 2 MiB video/mp4 chunk
     reassembly, SHA-256 match, ffprobe H.264/dimensions/frame count check,
     UMM_TRACE_FILE legs_snapshot pending->running->completed, in_flight=0.
  3. Mid-generation /abort: cross-session 403, matching-session 200, C++ stage
     RequestAborted at step boundary, SSE cancelled terminal with video leg
     aborted, legs_snapshot pending->running->aborted, in_flight=0, and clean
     follow-up generation.
  4. Stage validation failure (unaligned 500x480): leg_error + terminal
     complete with video leg failed, legs_snapshot pending->running->failed,
     in_flight=0.

Note: This is an offline contract test (`backend=native-cpp-contract-stub`);
it validates the native C++ Stage/UnifiedEngine vertical slice and NEVER counts
as real GPU Wan2.2 model verification.
"""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def http_json(url, method="GET", body=None, timeout=15.0):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"raw": raw}
        return exc.code, parsed


def stream_chat(base_url, body, timeout=60.0, on_event=None):
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    events = []
    saw_done = False
    request_id = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    saw_done = True
                    break
                frame = json.loads(payload)
                if request_id is None and isinstance(frame.get("id"), str):
                    request_id = frame["id"]
                unified = frame.get("unified")
                if isinstance(unified, dict):
                    events.append(unified)
                    if on_event is not None:
                        on_event(request_id, unified)
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} on /v1/chat/completions: {err_body}") from exc
    return status, request_id, events, saw_done


def wait_for_ready(base_url: str, timeout_s: float = 15.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            st, body = http_json(f"{base_url}/ready", timeout=2.0)
            if st == 200:
                return body
        except Exception:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"server at {base_url} did not become ready within {timeout_s}s")


def wait_for_in_flight_zero(base_url: str, timeout_s: float = 5.0):
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        st, body = http_json(f"{base_url}/health", timeout=2.0)
        last = (st, body)
        if st == 200 and body.get("in_flight") == 0:
            return body
        time.sleep(0.05)
    raise RuntimeError(f"in_flight did not reach 0: {last}")


def ffprobe_mp4(path: Path):
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries",
            "format=format_name,size,duration:stream=codec_type,codec_name,width,height,nb_frames,r_frame_rate,pix_fmt",
            "-of", "json",
            str(path),
        ],
        text=True,
        timeout=15,
    )
    return json.loads(out)


def verify_trace_transitions(trace_file: Path, expected_by_req: dict[str, list[str]]):
    if not trace_file.is_file():
        raise RuntimeError(f"trace file missing: {trace_file}")
    snapshots_by_req: dict[str, list[dict]] = {k: [] for k in expected_by_req}
    for raw in trace_file.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)
        if rec.get("event") != "legs_snapshot":
            continue
        req_id = rec.get("request_id")
        if req_id in snapshots_by_req:
            snapshots_by_req[req_id].append(rec)

    for req_id, expected in expected_by_req.items():
        observed = []
        for snap in snapshots_by_req[req_id]:
            for leg in snap.get("legs", []):
                if leg.get("stage") == "video":
                    observed.append(leg.get("state"))
        if observed != expected:
            raise RuntimeError(
                f"legs_snapshot mismatch for {req_id}: expected {expected}, "
                f"observed {observed} (snapshots={snapshots_by_req[req_id]})"
            )


def terminate_proc(proc: subprocess.Popen, name: str):
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
        raise RuntimeError(f"{name} did not exit cleanly on SIGTERM")


def run_contract(
    server_bin: Path,
    stage_bin: Path,
    video_backend: str = "contract-stub",
    gpu: int = 0,
    model_dir: Path | None = None,
    size: str = "160x128",
    fps: int = 8,
    steps: int = 8,
    step_delay_ms: int = 120,
):
    expected_backend_tag = (
        "native-cuda-scaffold"
        if video_backend in ("cuda-scaffold", "cuda_scaffold")
        else "native-cpp-contract-stub"
    )
    evidence_class = (
        "real-gpu-cuda-scaffold-test"
        if expected_backend_tag == "native-cuda-scaffold"
        else "offline-contract-test"
    )
    expected_w, expected_h = [int(x) for x in size.lower().split("x")]
    with tempfile.TemporaryDirectory(prefix="umminfer-native-video-") as tmp_str:
        tmp = Path(tmp_str)
        if model_dir is not None:
            resolved_model_dir = model_dir
        else:
            resolved_model_dir = tmp / "Wan2.2-TI2V-5B-manifest"
            resolved_model_dir.mkdir(parents=True)
            (resolved_model_dir / "model_index.json").write_text(
                json.dumps({"_class_name": "WanPipeline"}), encoding="utf-8")

        ipc_dir = tmp / "ipc"
        ipc_dir.mkdir(mode=0o700)
        asr_ep = f"ipc://{ipc_dir}/asr"
        vlm_ep = f"ipc://{ipc_dir}/vlm"
        video_ep = f"ipc://{ipc_dir}/video"
        comp_ep = f"ipc://{ipc_dir}/completion"
        abort_ep = f"ipc://{ipc_dir}/abort"

        # 1) Safety gate: default --video-backend cuda must fail loudly.
        gate = subprocess.run(
            [
                str(stage_bin),
                "--kind", "video",
                "--model", str(resolved_model_dir),
                "--gpu", str(gpu),
                "--recv-endpoint", video_ep,
                "--coordinator-endpoint", comp_ep,
                "--abort-endpoint", abort_ep,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if gate.returncode == 0:
            raise RuntimeError("expected default --video-backend cuda to refuse silent fallback")

        port = free_port()
        base_url = f"http://127.0.0.1:{port}"
        trace_file = tmp / "trace.jsonl"
        server_log = tmp / "server.log"
        stage_log = tmp / "stage.log"

        env = os.environ.copy()
        env["UMM_TRACE"] = "1"
        env["UMM_TRACE_FILE"] = str(trace_file)

        with server_log.open("w", encoding="utf-8") as slog, \
             stage_log.open("w", encoding="utf-8") as vlog:
            server = subprocess.Popen(
                [
                    str(server_bin), "serve",
                    "--model", "demo",
                    "--host", "127.0.0.1",
                    "--port", str(port),
                    "--entry", "asr",
                    "--entry-endpoint", asr_ep,
                    "--vlm-entry", "vlm",
                    "--vlm-entry-endpoint", vlm_ep,
                    "--video-entry", "video",
                    "--video-entry-endpoint", video_ep,
                    "--completion", comp_ep,
                    "--abort", abort_ep,
                ],
                stdout=slog,
                stderr=subprocess.STDOUT,
                env=env,
            )
            asr_stage = subprocess.Popen(
                [
                    str(server_bin), "stage",
                    "--name", "asr",
                    "--recv", asr_ep,
                    "--coordinator", comp_ep,
                    "--abort", abort_ep,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            vlm_stage = subprocess.Popen(
                [
                    str(server_bin), "stage",
                    "--name", "vlm",
                    "--recv", vlm_ep,
                    "--coordinator", comp_ep,
                    "--abort", abort_ep,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            stage = subprocess.Popen(
                [
                    str(stage_bin),
                    "--kind", "video",
                    "--name", "video",
                    "--model", str(resolved_model_dir),
                    "--gpu", str(gpu),
                    "--video-backend", video_backend,
                    "--size", size,
                    "--fps", str(fps),
                    "--steps", str(steps),
                    "--step-delay-ms", str(step_delay_ms),
                    "--recv-endpoint", video_ep,
                    "--coordinator-endpoint", comp_ep,
                    "--abort-endpoint", abort_ep,
                ],
                stdout=vlog,
                stderr=subprocess.STDOUT,
                env=env,
            )

            try:
                try:
                    wait_for_ready(base_url, timeout_s=8.0)
                except Exception as exc:
                    raise RuntimeError(
                        f"{exc}\nserver.log:\n{server_log.read_text(encoding='utf-8')}\n"
                        f"stage.log:\n{stage_log.read_text(encoding='utf-8')}"
                    ) from exc
                # Wait briefly for stage_ready handshake to settle.
                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline:
                    if "ready kind=video" in stage_log.read_text(encoding="utf-8"):
                        break
                    time.sleep(0.05)
                else:
                    raise RuntimeError(
                        f"native video stage did not log ready: {stage_log.read_text(encoding='utf-8')}")

                # 2) Completed path
                gen_session = "native-video-ok"
                st, gen_req_id, events, saw_done = stream_chat(
                    base_url,
                    {
                        "model": "demo",
                        "stream": True,
                        "session_id": gen_session,
                        "modalities": ["text", "video"],
                        "messages": [{"role": "user", "content": "Describe a calm pond."}],
                        "video": {
                            "prompt": "A paper boat floating on a calm pond",
                            "size": size,
                            "seconds": 1,
                            "fps": fps,
                            "seed": 20261122,
                            "output_format": "mp4",
                        },
                    },
                )
                assert st == 200 and saw_done, f"unexpected HTTP result: {st} {saw_done}"
                expected_seq = 0
                chunks = {}
                artifact_meta = None
                for ev in events:
                    assert ev["seq"] == expected_seq, f"seq gap: {ev}"
                    expected_seq += 1
                    assert ev["session_id"] == gen_session
                    if ev["kind"] == "artifact" and ev["stage"] == "video":
                        art = ev["artifact"]
                        artifact_meta = {k: v for k, v in art.items() if k != "data"}
                        chunks[int(art["index"])] = base64.b64decode(art["data"])
                terminal = events[-1]
                assert terminal["kind"] == "complete", f"expected complete: {terminal}"
                vleg = next(l for l in terminal["legs"] if l["stage"] == "video")
                assert vleg["state"] == "completed", f"video leg not completed: {vleg}"
                assert artifact_meta is not None
                assert artifact_meta["backend"] == expected_backend_tag, (
                    f"expected backend {expected_backend_tag}, got {artifact_meta['backend']}")
                mp4_bytes = b"".join(chunks[i] for i in range(int(artifact_meta["total"])))
                assert len(mp4_bytes) >= 12 and mp4_bytes[4:8] == b"ftyp"
                mp4_sha = hashlib.sha256(mp4_bytes).hexdigest()
                assert mp4_sha == artifact_meta["sha256"], "MP4 SHA-256 mismatch"
                mp4_path = tmp / "out.mp4"
                mp4_path.write_bytes(mp4_bytes)
                probe = ffprobe_mp4(mp4_path)
                vstream = next(s for s in probe["streams"] if s["codec_type"] == "video")
                assert vstream["codec_name"] == "h264"
                assert int(vstream["width"]) == expected_w and int(vstream["height"]) == expected_h
                expected_frames = max(5, ((max(5, fps) - 1) // 4) * 4 + 1)
                assert int(vstream["nb_frames"]) == expected_frames, (
                    f"expected {expected_frames} frames, got {vstream['nb_frames']}")
                wait_for_in_flight_zero(base_url)

                # 3) Mid-generation /abort path (wrong session 403 -> matching session 200)
                cancel_session = "native-video-cancel"
                abort_info = {}
                stop_watch = threading.Event()

                def watch_and_abort():
                    deadline = time.monotonic() + 10.0
                    while time.monotonic() < deadline and not stop_watch.is_set():
                        if trace_file.is_file():
                            for raw in trace_file.read_text(encoding="utf-8").splitlines():
                                raw = raw.strip()
                                if not raw:
                                    continue
                                try:
                                    rec = json.loads(raw)
                                except Exception:
                                    continue
                                if rec.get("event") != "legs_snapshot":
                                    continue
                                rid = rec.get("request_id")
                                if not rid or rid == gen_req_id:
                                    continue
                                for leg in rec.get("legs", []):
                                    if leg.get("stage") == "video" and leg.get("state") == "running":
                                        # Let denoising enter the step loop briefly.
                                        time.sleep(0.15)
                                        st_bad, _ = http_json(
                                            f"{base_url}/abort",
                                            method="POST",
                                            body={"request_id": rid, "session_id": "wrong-session"},
                                        )
                                        st_ok, _ = http_json(
                                            f"{base_url}/abort",
                                            method="POST",
                                            body={"request_id": rid, "session_id": cancel_session},
                                        )
                                        abort_info["request_id"] = rid
                                        abort_info["wrong"] = st_bad
                                        abort_info["ok"] = st_ok
                                        return
                        time.sleep(0.02)

                watcher = threading.Thread(target=watch_and_abort, daemon=True)
                watcher.start()

                st_c, cancel_req_id, c_events, c_done = stream_chat(
                    base_url,
                    {
                        "model": "demo",
                        "stream": True,
                        "session_id": cancel_session,
                        "modalities": ["text", "video"],
                        "messages": [{"role": "user", "content": "Say ok."}],
                        "video": {
                            "prompt": "Slow motion meadow",
                            "size": "160x128",
                            "seconds": 1,
                            "fps": 8,
                            "seed": 20261123,
                            "output_format": "mp4",
                        },
                    },
                )
                stop_watch.set()
                watcher.join(timeout=5.0)
                assert st_c == 200 and c_done
                assert abort_info.get("wrong") == 403, f"expected 403: {abort_info}"
                assert abort_info.get("ok") == 200, f"expected 200: {abort_info}"
                c_term = c_events[-1]
                assert c_term["kind"] == "cancelled", f"expected cancelled: {c_term}"
                c_vleg = next(l for l in c_term["legs"] if l["stage"] == "video")
                assert c_vleg["state"] == "aborted", f"expected video aborted: {c_vleg}"
                wait_for_in_flight_zero(base_url)

                # 4) Failed leg path (unaligned 500x480 rejected by Wan 32-px alignment)
                fail_session = "native-video-fail"
                st_f, fail_req_id, f_events, f_done = stream_chat(
                    base_url,
                    {
                        "model": "demo",
                        "stream": True,
                        "session_id": fail_session,
                        "modalities": ["text", "video"],
                        "messages": [{"role": "user", "content": "Say ok."}],
                        "video": {
                            "prompt": "invalid alignment",
                            "size": "500x480",
                            "seconds": 1,
                            "fps": 8,
                            "output_format": "mp4",
                        },
                    },
                )
                assert st_f == 200 and f_done
                leg_errs = [e for e in f_events if e["kind"] == "leg_error" and e["stage"] == "video"]
                assert leg_errs, f"expected video leg_error: {f_events}"
                f_term = f_events[-1]
                assert f_term["kind"] == "complete", f"expected complete: {f_term}"
                f_vleg = next(l for l in f_term["legs"] if l["stage"] == "video")
                assert f_vleg["state"] == "failed", f"expected video failed: {f_vleg}"
                wait_for_in_flight_zero(base_url)

                # 5) Verify UMM_TRACE_FILE legs_snapshot transitions for all three lifecycles.
                verify_trace_transitions(
                    trace_file,
                    {
                        gen_req_id: ["pending", "running", "completed"],
                        cancel_req_id: ["pending", "running", "aborted"],
                        fail_req_id: ["pending", "running", "failed"],
                    },
                )

                # Verify stage log captured the step-boundary abort.
                stage_text = stage_log.read_text(encoding="utf-8")
                assert "video request aborted" in stage_text, (
                    f"expected step-boundary abort log in stage stderr:\n{stage_text}"
                )

                summary = {
                    "evidence_class": evidence_class,
                    "backend": expected_backend_tag,
                    "gpu": gpu,
                    "model_dir": str(resolved_model_dir),
                    "cuda_default_gate_rejected": True,
                    "completed": {
                        "request_id": gen_req_id,
                        "mp4_bytes": len(mp4_bytes),
                        "mp4_sha256": mp4_sha,
                        "codec": vstream["codec_name"],
                        "size": f"{vstream['width']}x{vstream['height']}",
                        "frames": int(vstream["nb_frames"]),
                        "r_frame_rate": vstream.get("r_frame_rate"),
                        "duration": probe.get("format", {}).get("duration"),
                    },
                    "cancelled": {
                        "request_id": cancel_req_id,
                        "abort_wrong_session": abort_info["wrong"],
                        "abort_matching_session": abort_info["ok"],
                        "stage_abort_logged": True,
                    },
                    "failed": {
                        "request_id": fail_req_id,
                        "leg_error": leg_errs[0]["error"],
                    },
                    "in_flight_after_all": 0,
                    "legs_snapshot_verified": True,
                }
                print(json.dumps(summary, indent=2))
            finally:
                terminate_proc(stage, "umm-native-stage-video")
                terminate_proc(vlm_stage, "vlm-echo-stage")
                terminate_proc(asr_stage, "asr-echo-stage")
                terminate_proc(server, "umm-infer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", type=Path, required=True)
    parser.add_argument("--stage-bin", type=Path, required=True)
    parser.add_argument("--video-backend", default="contract-stub",
                        choices=["contract-stub", "cuda-scaffold"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--size", default="160x128")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--step-delay-ms", type=int, default=120)
    args = parser.parse_args()
    run_contract(
        args.server_bin,
        args.stage_bin,
        video_backend=args.video_backend,
        gpu=args.gpu,
        model_dir=args.model_dir,
        size=args.size,
        fps=args.fps,
        steps=args.steps,
        step_delay_ms=args.step_delay_ms,
    )


if __name__ == "__main__":
    main()
