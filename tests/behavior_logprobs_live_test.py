#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deterministic transport regression for the real live-gate capture helper."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import behavior_logprobs_live as live  # noqa: E402
import umminfer_online_rollout as online  # noqa: E402
import umminfer_rl_trajectory as traj  # noqa: E402


def main():
    body = dict(live.request_cases("qwen3-native"))["text"]
    provenance = {
        "revision": "UNKNOWN", "model_sha256": "00" * 32,
        "config_sha256": "11" * 32,
    }
    plan, item, _ = live.attestation_for(body, "text", provenance)
    response_attestation = online._expected_attestation(
        item, plan, http_status=200, behavior_evidence="exposed", attempt=0)
    behavior = {
        "format": online.BEHAVIOR_FORMAT, "tokenizer_sha256": "ab" * 32,
        "token_ids": [101, 102], "logprobs": [-0.5, -0.25], "mask": [1, 1],
    }
    unified = {
        "kind": "complete", "stage": "vlm", "seq": 0,
        "session_id": body["session_id"], "is_final": True,
        "text": "红灯笼 🌅", "request_id": "http-1",
    }
    final = {
        "id": "http-1", "object": "chat.completion.chunk", "created": 0,
        "model": "qwen3-native",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "unified": unified, "behavior": behavior,
    }
    raw = b"data: " + online.canonical_json(final) + b"\n\ndata: [DONE]\n\n"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            for key, name in online.RESPONSE_HEADERS.items():
                self.send_header(name, str(response_attestation[key]))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="umm-behavior-capture-test-") as tmp:
            recorder = traj.TrajectoryRecorder(Path(tmp) / "live/trajectories.jsonl")
            captured = live.capture_case(
                f"127.0.0.1:{server.server_port}", "text", body, recorder,
                plan, item, online._request_headers(item, plan, 0))
            assert captured["coverage"] == "exposed"
            assert captured["behavior"]["status"] == "COVERED"
            assert captured["record"]["clean_text"] == "红灯笼 🌅"
            assert captured["record"]["sse_replay"]["raw_bytes"] == len(raw)
            assert (Path(tmp) / "live/raw/text.sse").read_bytes() == raw
            variant = live.replay_tools.framing_variant(
                captured["record"], raw, recorder.artifact_dir,
                Path(tmp) / "framing/trajectories.jsonl", (1, 2, 3, 5))
            assert variant["semantic_identity"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    print("BEHAVIOR_LIVE_HARNESS_TEST: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
