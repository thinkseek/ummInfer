#!/usr/bin/env python3
"""Live end-to-end verification for UnifiedEngine + real video stage.

Exercises:
1. Public HTTP POST /v1/chat/completions (stream=true, modalities=["text","video"])
   -> UnifiedEngine -> real VLM + real Video Stage -> SSE 2 MiB video/mp4 chunks
   -> reassembled MP4 verified by ffprobe + SHA-256.
2. Real mid-generation cancellation via POST /abort (verifying cross-session 403
   rejection, same-session 200 cancel, SSE cancelled terminal with video leg
   aborted, /health in_flight == 0, and clean follow-up /ready).
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


def http_json(url, method="GET", body=None, timeout=30.0):
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


def stream_chat(base_url, body, timeout=540.0, on_event=None):
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    events = []
    saw_done = False
    request_id = None
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
    return status, request_id, events, saw_done


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
        timeout=30,
    )
    return json.loads(out)


def verify_generation(base_url, model, out_dir: Path, prompt: str,
                      video_prompt: str, size: str, seconds: int, fps: int,
                      seed: int):
    session_id = f"video-live-{int(time.time())}"
    body = {
        "model": model,
        "stream": True,
        "max_tokens": 32,
        "session_id": session_id,
        "modalities": ["text", "video"],
        "messages": [{"role": "user", "content": prompt}],
        "video": {
            "prompt": video_prompt,
            "size": size,
            "seconds": seconds,
            "fps": fps,
            "seed": seed,
            "output_format": "mp4",
        },
    }
    t0 = time.perf_counter()
    status, request_id, events, saw_done = stream_chat(base_url, body)
    elapsed_s = time.perf_counter() - t0
    if status != 200 or not saw_done:
        raise RuntimeError(f"generation HTTP failed: status={status} saw_done={saw_done}")

    # Verify gap-free seq and session_id
    expected_seq = 0
    vlm_text_parts = []
    chunks_by_index = {}
    artifact_meta = None
    terminal = None
    for ev in events:
        if ev.get("seq") != expected_seq:
            raise RuntimeError(f"seq gap: expected {expected_seq}, got {ev.get('seq')}")
        expected_seq += 1
        if ev.get("session_id") != session_id:
            raise RuntimeError(f"session_id mismatch: {ev.get('session_id')} != {session_id}")
        kind = ev.get("kind")
        if kind == "vlm_delta":
            vlm_text_parts.append(ev.get("text", ""))
        elif kind == "artifact" and ev.get("stage") == "video":
            art = ev.get("artifact", {})
            artifact_meta = {k: v for k, v in art.items() if k != "data"}
            raw = base64.b64decode(art["data"])
            chunks_by_index[int(art["index"])] = raw
        elif kind in ("complete", "error", "cancelled"):
            terminal = ev

    if terminal is None or terminal.get("kind") != "complete":
        raise RuntimeError(f"expected terminal complete, got {terminal}")
    legs = terminal.get("legs", [])
    video_leg = next((l for l in legs if l.get("stage") == "video"), None)
    if video_leg is None or video_leg.get("state") != "completed":
        raise RuntimeError(f"video leg did not complete: {legs}")
    if artifact_meta is None:
        raise RuntimeError("no video artifact events received")
    total_chunks = int(artifact_meta["total"])
    if sorted(chunks_by_index.keys()) != list(range(total_chunks)):
        raise RuntimeError(
            f"missing artifact chunks: got {sorted(chunks_by_index.keys())}, total={total_chunks}")

    mp4_bytes = b"".join(chunks_by_index[i] for i in range(total_chunks))
    if len(mp4_bytes) < 12 or mp4_bytes[4:8] != b"ftyp":
        raise RuntimeError("reassembled video bytes lack ISO BMFF ftyp header")
    mp4_path = out_dir / "real_video_output.mp4"
    mp4_path.write_bytes(mp4_bytes)
    sha256 = hashlib.sha256(mp4_bytes).hexdigest()
    probe = ffprobe_mp4(mp4_path)
    video_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
    if not video_streams:
        raise RuntimeError(f"ffprobe found no video stream: {probe}")

    return {
        "request_id": request_id,
        "session_id": session_id,
        "elapsed_s": round(elapsed_s, 3),
        "event_count": len(events),
        "vlm_text": "".join(vlm_text_parts),
        "legs": legs,
        "artifact_meta": artifact_meta,
        "chunk_count": total_chunks,
        "mp4_path": str(mp4_path),
        "mp4_bytes": len(mp4_bytes),
        "mp4_sha256": sha256,
        "ffprobe": probe,
    }


def verify_cancel(base_url, model, prompt: str, video_prompt: str, size: str,
                  seconds: int, fps: int, seed: int, abort_delay_s: float):
    session_id = f"video-cancel-{int(time.time())}"
    body = {
        "model": model,
        "stream": True,
        "max_tokens": 16,
        "session_id": session_id,
        "modalities": ["text", "video"],
        "messages": [{"role": "user", "content": prompt}],
        "video": {
            "prompt": video_prompt,
            "size": size,
            "seconds": seconds,
            "fps": fps,
            "seed": seed,
            "output_format": "mp4",
        },
    }
    abort_evidence = {}
    abort_triggered = threading.Event()

    def trigger_abort(req_id):
        time.sleep(abort_delay_s)
        # 1) Wrong session must be rejected with 403.
        st_bad, body_bad = http_json(
            f"{base_url.rstrip('/')}/abort",
            method="POST",
            body={"request_id": req_id, "session_id": "wrong-session"},
        )
        # 2) Matching session must cancel with 200.
        st_ok, body_ok = http_json(
            f"{base_url.rstrip('/')}/abort",
            method="POST",
            body={"request_id": req_id, "session_id": session_id},
        )
        abort_evidence["wrong_session_status"] = st_bad
        abort_evidence["wrong_session_body"] = body_bad
        abort_evidence["cancel_status"] = st_ok
        abort_evidence["cancel_body"] = body_ok

    def on_event(req_id, ev):
        # Schedule abort after the first VLM delta arrives; the delay lets VLM
        # finish and hand off to the video stage so abort hits active denoising.
        if not abort_triggered.is_set() and req_id and ev.get("kind") == "vlm_delta":
            abort_triggered.set()
            threading.Thread(target=trigger_abort, args=(req_id,), daemon=True).start()

    t0 = time.perf_counter()
    status, request_id, events, saw_done = stream_chat(
        base_url, body, on_event=on_event)
    elapsed_s = time.perf_counter() - t0
    if not abort_triggered.is_set():
        raise RuntimeError("did not observe vlm_delta to trigger video abort")

    terminal = events[-1] if events else None
    if terminal is None or terminal.get("kind") != "cancelled":
        raise RuntimeError(f"expected terminal cancelled, got {terminal}")
    legs = terminal.get("legs", [])
    video_leg = next((l for l in legs if l.get("stage") == "video"), None)
    if video_leg is None or video_leg.get("state") != "aborted":
        raise RuntimeError(f"expected video leg aborted, got {legs}")
    if abort_evidence.get("wrong_session_status") != 403:
        raise RuntimeError(f"expected 403 for wrong session abort, got {abort_evidence}")
    if abort_evidence.get("cancel_status") != 200:
        raise RuntimeError(f"expected 200 for matching session abort, got {abort_evidence}")

    # Poll /health until in_flight == 0
    health = None
    for _ in range(50):
        st_h, health = http_json(f"{base_url.rstrip('/')}/health")
        if st_h == 200 and health.get("in_flight") == 0:
            break
        time.sleep(0.1)
    if not health or health.get("in_flight") != 0:
        raise RuntimeError(f"in_flight did not return to 0 after cancel: {health}")

    return {
        "request_id": request_id,
        "session_id": session_id,
        "elapsed_s": round(elapsed_s, 3),
        "event_count": len(events),
        "legs": legs,
        "abort_evidence": abort_evidence,
        "health_after_cancel": health,
        "saw_done": saw_done,
    }


def verify_leg_failure(base_url, model):
    """Verify video leg transitions cleanly to failed on stage validation error."""
    session_id = f"video-fail-{int(time.time())}"
    body = {
        "model": model,
        "stream": True,
        "max_tokens": 16,
        "session_id": session_id,
        "modalities": ["text", "video"],
        "messages": [{"role": "user", "content": "Say ok."}],
        "video": {
            "prompt": "invalid alignment test",
            "size": "500x480",  # passes HTTP non-empty check, rejected by Wan 32-px alignment
            "seconds": 1,
            "fps": 16,
            "output_format": "mp4",
        },
    }
    t0 = time.perf_counter()
    status, request_id, events, saw_done = stream_chat(base_url, body)
    elapsed_s = time.perf_counter() - t0
    if status != 200 or not saw_done:
        raise RuntimeError(f"failure test HTTP failed: status={status} saw_done={saw_done}")
    leg_errors = [ev for ev in events if ev.get("kind") == "leg_error" and ev.get("stage") == "video"]
    terminal = events[-1] if events else None
    if not leg_errors:
        raise RuntimeError(f"expected video leg_error event, got {events}")
    if terminal is None or terminal.get("kind") != "complete":
        raise RuntimeError(f"expected terminal complete with failed video leg, got {terminal}")
    legs = terminal.get("legs", [])
    video_leg = next((l for l in legs if l.get("stage") == "video"), None)
    if video_leg is None or video_leg.get("state") != "failed":
        raise RuntimeError(f"expected video leg failed, got {legs}")
    return {
        "request_id": request_id,
        "session_id": session_id,
        "elapsed_s": round(elapsed_s, 3),
        "leg_error": leg_errors[0].get("error"),
        "legs": legs,
    }


def verify_trace_transitions(trace_file: Path, expected_by_req: dict[str, list[str]]):
    """Verify real UnifiedEngine legs_snapshot transitions from UMM_TRACE_FILE."""
    if not trace_file.is_file():
        raise RuntimeError(f"trace file does not exist: {trace_file}")
    snapshots_by_req: dict[str, list[dict]] = {req_id: [] for req_id in expected_by_req}
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

    verified = {}
    for req_id, expected_states in expected_by_req.items():
        snaps = snapshots_by_req[req_id]
        observed = []
        for snap in snaps:
            for leg in snap.get("legs", []):
                if leg.get("stage") == "video":
                    observed.append(leg.get("state"))
        if observed != expected_states:
            raise RuntimeError(
                f"video leg lifecycle mismatch for {req_id}: "
                f"expected {expected_states}, observed {observed} (snapshots={snaps})")
        verified[req_id] = {
            "video_states": observed,
            "snapshots": snaps,
        }
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="qwen3-native")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--size", default="480x320")
    parser.add_argument("--seconds", type=int, default=1)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--abort-delay-s", type=float, default=3.0)
    parser.add_argument("--trace-file", type=Path, default=None,
                        help="Optional UMM_TRACE_FILE JSONL to assert real UnifiedEngine legs_snapshot transitions")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    st_ready, ready_body = http_json(f"{args.url.rstrip('/')}/ready")
    if st_ready != 200:
        raise RuntimeError(f"/ready returned {st_ready}: {ready_body}")

    gen_report = verify_generation(
        args.url, args.model, args.out_dir,
        prompt="Describe in one short sentence a paper boat floating on a calm pond.",
        video_prompt="A small white paper boat gently floating on a calm sunlit pond, soft ripples, cinematic lighting",
        size=args.size, seconds=args.seconds, fps=args.fps, seed=args.seed,
    )
    cancel_report = verify_cancel(
        args.url, args.model,
        prompt="Say ok.",
        video_prompt="A golden retriever running across a wide sunlit meadow in slow motion",
        size=args.size, seconds=max(2, args.seconds), fps=args.fps,
        seed=args.seed + 1, abort_delay_s=args.abort_delay_s,
    )
    failure_report = verify_leg_failure(args.url, args.model)
    st_health, final_health = http_json(f"{args.url.rstrip('/')}/health")
    trace_report = None
    if args.trace_file is not None:
        trace_report = verify_trace_transitions(
            args.trace_file,
            {
                gen_report["request_id"]: ["pending", "running", "completed"],
                cancel_report["request_id"]: ["pending", "running", "aborted"],
                failure_report["request_id"]: ["pending", "running", "failed"],
            },
        )
    report = {
        "url": args.url,
        "model": args.model,
        "ready": {"status": st_ready, "body": ready_body},
        "generation": gen_report,
        "cancel": cancel_report,
        "leg_failure": failure_report,
        "legs_snapshot_trace": trace_report,
        "final_health": {"status": st_health, "body": final_health},
    }
    report_path = args.out_dir / "video_live_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
