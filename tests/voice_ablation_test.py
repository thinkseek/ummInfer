#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit and contract tests for Stage C Ablation #3: Voice Collaboration Strategy."""

from __future__ import annotations

import ast
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIR = REPO_ROOT / "python"
SCRIPT_PATH = PYTHON_DIR / "umminfer_voice_ablation.py"
CONFIG_PATH = REPO_ROOT / "benchmarks" / "ablations" / "voice-strategy.json"
PROMPTS_PATH = REPO_ROOT / "benchmarks" / "ablations" / "voice-strategy-prompts.json"
REPLAY_PATH = REPO_ROOT / "benchmarks" / "ablations" / "voice-strategy-replay.json"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import umminfer_voice_ablation as va  # noqa: E402


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", str(SCRIPT_PATH), *args],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )


class _LoopbackVoiceHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    replay_cases: dict[tuple[str, str], dict] = {}
    recorded_requests: list[dict] = []

    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length))
        type(self).recorded_requests.append(body)

        is_stream = bool(body.get("audio", {}).get("stream"))
        strategy = "sentence_streaming_tts" if is_stream else "full_response_tts"
        prompt_text = json.dumps(body.get("messages", [])).lower()
        if "speed of light" in prompt_text:
            pid = "short_factual"
        elif "ice floats" in prompt_text:
            pid = "physics_explanation"
        else:
            pid = "vivid_narrative"

        case = self.replay_cases[(strategy, pid)]
        raw_sse = base64.b64decode(case["response_sse_base64"])

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        # Stream frame-by-frame with small deterministic delays so live monotonic
        # timings exercise TTFA speedup and inter-chunk gaps cleanly.
        for block in raw_sse.split(b"\n\n"):
            if not block:
                continue
            payload = block + b"\n\n"
            if not is_stream and b'"kind":"artifact"' in block:
                time.sleep(0.030)
            elif is_stream and b'"kind":"artifact"' in block:
                time.sleep(0.004)
            else:
                time.sleep(0.002)
            self.wfile.write(payload)
            self.wfile.flush()
        self.close_connection = True


class VoiceAblationTest(unittest.TestCase):
    def test_single_variable_invariance_only_audio_stream_toggles(self) -> None:
        exp = va.load_experiment(CONFIG_PATH)
        matrix = va.verify_single_variable_matrix(exp)
        self.assertTrue(matrix["single_variable_verified"])
        self.assertEqual(len(exp.prompts), 3)

        full_items = matrix["by_arm"]["full_response_tts"]
        stream_items = matrix["by_arm"]["sentence_streaming_tts"]
        for full_item, stream_item in zip(full_items, stream_items):
            self.assertEqual(full_item["prompt_id"], stream_item["prompt_id"])
            self.assertEqual(
                full_item["fixed_request_sha256"],
                stream_item["fixed_request_sha256"],
            )
            self.assertNotEqual(full_item["request_sha256"], stream_item["request_sha256"])
            self.assertFalse(full_item["request"]["audio"]["stream"])
            self.assertTrue(stream_item["request"]["audio"]["stream"])

            # Verify literally zero other fields differ between arms
            req_full = json.loads(json.dumps(full_item["request"]))
            req_stream = json.loads(json.dumps(stream_item["request"]))
            req_full["audio"].pop("stream")
            req_stream["audio"].pop("stream")
            self.assertEqual(req_full, req_stream)

    def test_deterministic_byte_identical_replay_and_sha256sums(self) -> None:
        with tempfile.TemporaryDirectory(prefix="voice-ablation-det-") as tmp:
            out_a = Path(tmp) / "run_a"
            out_b = Path(tmp) / "run_b"
            proc_a = _run_cli("--config", str(CONFIG_PATH), "--replay-only", "--out", str(out_a))
            proc_b = _run_cli("--config", str(CONFIG_PATH), "--replay-only", "--out", str(out_b))
            self.assertEqual(proc_a.returncode, 0, f"stderr={proc_a.stderr}\nstdout={proc_a.stdout}")
            self.assertEqual(proc_b.returncode, 0, f"stderr={proc_b.stderr}\nstdout={proc_b.stdout}")
            self.assertIn("CONTRACT_REPLAY", proc_a.stdout)
            self.assertIn("REAL_MODEL_EVIDENCE: NOT_RUN", proc_a.stdout)

            tree_a = _tree_bytes(out_a)
            tree_b = _tree_bytes(out_b)
            self.assertEqual(tree_a, tree_b)
            self.assertEqual(
                set(tree_a.keys()),
                {
                    "SHA256SUMS",
                    "report.json",
                    "arms/full_response_tts/report.json",
                    "arms/sentence_streaming_tts/report.json",
                },
            )
            self.assertEqual(va.verify_sha256sums(out_a), 3)

            report = json.loads((out_a / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["evidence_class"], "CONTRACT_REPLAY")
            self.assertEqual(report["real_model_evidence"], "NOT_RUN")
            self.assertIn("real_model_latency_improvement", report["claims_forbidden"])
            self.assertTrue(report["single_variable"]["single_variable_verified"])

    def test_tamper_detection_fails_closed(self) -> None:
        # 1. Tampered dataset file -> dataset sha256 mismatch
        with tempfile.TemporaryDirectory(prefix="voice-tamper-data-") as tmp:
            work = Path(tmp)
            for src in (CONFIG_PATH, PROMPTS_PATH, REPLAY_PATH):
                shutil.copy2(src, work / src.name)
            p_path = work / PROMPTS_PATH.name
            p_doc = json.loads(p_path.read_text(encoding="utf-8"))
            p_doc["prompts"][0]["messages"][0]["content"] += " (tampered)"
            p_path.write_text(json.dumps(p_doc), encoding="utf-8")
            proc = _run_cli(
                "--config", str(work / CONFIG_PATH.name), "--replay-only", "--out", str(work / "out")
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("dataset sha256 mismatch", proc.stderr)

        # 2. Tampered replay file -> replay sha256 mismatch
        with tempfile.TemporaryDirectory(prefix="voice-tamper-replay-") as tmp:
            work = Path(tmp)
            for src in (CONFIG_PATH, PROMPTS_PATH, REPLAY_PATH):
                shutil.copy2(src, work / src.name)
            r_path = work / REPLAY_PATH.name
            r_doc = json.loads(r_path.read_text(encoding="utf-8"))
            r_doc["cases"][0]["latency_us"]["ttfa_us"] += 1
            r_path.write_text(json.dumps(r_doc), encoding="utf-8")
            proc = _run_cli(
                "--config", str(work / CONFIG_PATH.name), "--replay-only", "--out", str(work / "out")
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("replay sha256 mismatch", proc.stderr)

        # 3. Tampered config sampling (even with re-pinned replay hash) -> case fixed_request_sha256 mismatch
        with tempfile.TemporaryDirectory(prefix="voice-tamper-cfg-") as tmp:
            work = Path(tmp)
            for src in (CONFIG_PATH, PROMPTS_PATH, REPLAY_PATH):
                shutil.copy2(src, work / src.name)
            c_path = work / CONFIG_PATH.name
            c_doc = json.loads(c_path.read_text(encoding="utf-8"))
            c_doc["fixed"]["sampling"]["seed"] = 999
            c_path.write_text(json.dumps(c_doc), encoding="utf-8")
            proc = _run_cli("--config", str(c_path), "--replay-only", "--out", str(work / "out"))
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("fixed_request_sha256 mismatch", proc.stderr)

    def test_metric_correctness_ttfa_gaps_stalls_speedup_and_bytes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="voice-metrics-") as tmp:
            out_dir = Path(tmp)
            exp = va.load_experiment(CONFIG_PATH)
            report = va.run_ablation(exp, out_dir=out_dir, replay_only=True)

            full_arm = json.loads(
                (out_dir / "arms" / "full_response_tts" / "report.json").read_text(encoding="utf-8")
            )
            stream_arm = json.loads(
                (out_dir / "arms" / "sentence_streaming_tts" / "report.json").read_text(encoding="utf-8")
            )

            # Check full_response_tts per-case metrics
            full_by_id = {c["prompt_id"]: c for c in full_arm["cases"]}
            self.assertEqual(full_by_id["short_factual"]["ttft_ms"], 120.0)
            self.assertEqual(full_by_id["short_factual"]["ttfa_ms"], 1180.0)
            self.assertEqual(full_by_id["short_factual"]["ttfa_after_ttft_ms"], 1060.0)
            self.assertEqual(full_by_id["short_factual"]["audio_chunk_count"], 1)
            self.assertIsNone(full_by_id["short_factual"]["inter_chunk_gap_p95_ms"])
            self.assertEqual(full_by_id["short_factual"]["stall_count"], 0)
            self.assertEqual(full_by_id["short_factual"]["audio_bytes_total"], 44 + 240 * 2)
            self.assertEqual(
                full_arm["summary"]["metrics"]["inter_chunk_gap_p95_ms"]["coverage"],
                "NOT_APPLICABLE",
            )

            # Check sentence_streaming_tts per-case metrics (including 1 stall > 500 ms on vivid_narrative)
            stream_by_id = {c["prompt_id"]: c for c in stream_arm["cases"]}
            vivid = stream_by_id["vivid_narrative"]
            self.assertEqual(vivid["ttft_ms"], 140.0)
            self.assertEqual(vivid["ttfa_ms"], 380.0)
            self.assertEqual(vivid["ttfa_after_ttft_ms"], 240.0)
            self.assertEqual(vivid["audio_chunk_count"], 5)
            self.assertEqual(vivid["audio_bytes_total"], 5 * 120 * 2)
            self.assertEqual(vivid["inter_chunk_gaps_ms"], [360.0, 560.0, 360.0, 370.0])
            # Linear-interpolated p95 of [360, 360, 370, 560]: rank=0.95*3=2.85 -> 370 + 0.85*(560-370) = 531.5
            self.assertAlmostEqual(vivid["inter_chunk_gap_p95_ms"], 531.5, places=6)
            self.assertEqual(vivid["stall_count"], 1)
            self.assertAlmostEqual(vivid["stall_rate"], 0.25, places=6)

            # Check cross-arm ttfa_speedup_x:
            # full_response_tts ttfa medians = median([1180, 2520, 3560]) = 2520.0
            # sentence_streaming_tts ttfa medians = median([340, 360, 380]) = 360.0
            # speedup = 2520 / 360 = 7.0x
            self.assertEqual(report["arms"]["full_response_tts"]["metrics"]["ttfa_ms"]["median"], 2520.0)
            self.assertEqual(report["arms"]["sentence_streaming_tts"]["metrics"]["ttfa_ms"]["median"], 360.0)
            self.assertAlmostEqual(report["comparison"]["ttfa_speedup_x"], 7.0, places=6)
            self.assertEqual(report["arms"]["sentence_streaming_tts"]["total_stall_count"], 1)

    def test_loopback_live_http_sse_server_base_url_mode(self) -> None:
        replay_doc = json.loads(REPLAY_PATH.read_text(encoding="utf-8"))
        _LoopbackVoiceHandler.replay_cases = {
            (c["voice_strategy"], c["prompt_id"]): c for c in replay_doc["cases"]
        }
        _LoopbackVoiceHandler.recorded_requests = []

        server = ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackVoiceHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(prefix="voice-live-loopback-") as tmp:
                out_dir = Path(tmp)
                proc = _run_cli(
                    "--config",
                    str(CONFIG_PATH),
                    "--base-url",
                    f"http://127.0.0.1:{server.server_port}",
                    "--out",
                    str(out_dir),
                )
                self.assertEqual(proc.returncode, 0, f"stderr={proc.stderr}\nstdout={proc.stdout}")
                self.assertIn("UNCONTROLLED_LIVE", proc.stdout)
                self.assertEqual(len(_LoopbackVoiceHandler.recorded_requests), 6)

                report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["evidence_class"], "UNCONTROLLED_LIVE")
                self.assertEqual(report["real_model_evidence"], "UNCONTROLLED_LIVE")
                self.assertEqual(report["provenance"]["mode"], "live")
                self.assertIsNone(report["provenance"]["replay_sha256"])
                self.assertEqual(report["arms"]["full_response_tts"]["completed_case_count"], 3)
                self.assertEqual(report["arms"]["sentence_streaming_tts"]["completed_case_count"], 3)
                self.assertGreater(report["comparison"]["ttfa_speedup_x"], 1.0)
                self.assertEqual(va.verify_sha256sums(out_dir), 3)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5.0)

    def test_pure_stdlib_zero_cpp_or_ml_imports(self) -> None:
        tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

        stdlib_names = set(sys.stdlib_module_names)
        non_stdlib = sorted(name for name in imported if name not in stdlib_names)
        self.assertEqual(
            non_stdlib,
            [],
            f"{SCRIPT_PATH.name} must use pure Python stdlib only; found non-stdlib imports: {non_stdlib}",
        )
        forbidden_ml = {"torch", "numpy", "mlx", "transformers", "diffusers", "umminfer", "ctypes"}
        self.assertEqual(imported & forbidden_ml, set())


if __name__ == "__main__":
    unittest.main()
