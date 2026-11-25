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
CONTROLLED_SM90_CONFIG_PATH = (
    REPO_ROOT / "benchmarks" / "ablations" / "voice-strategy-controlled-sm90.json"
)
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
    recorded_order: list[tuple[str, str]] = []
    fail_on_call_number: int | None = None
    corrupt_audio_on_call_number: int | None = None
    delay_scale: float = 1.0

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
        call_num = len(type(self).recorded_requests)

        is_stream = bool(body.get("audio", {}).get("stream"))
        strategy = "sentence_streaming_tts" if is_stream else "full_response_tts"
        prompt_text = json.dumps(body.get("messages", [])).lower()
        if "speed of light" in prompt_text:
            pid = "short_factual"
        elif "ice floats" in prompt_text:
            pid = "physics_explanation"
        else:
            pid = "vivid_narrative"
        type(self).recorded_order.append((strategy, pid))

        if type(self).fail_on_call_number == call_num:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"injected measured failure"}}')
            self.wfile.flush()
            self.close_connection = True
            return

        case = self.replay_cases[(strategy, pid)]
        raw_sse = base64.b64decode(case["response_sse_base64"])
        if type(self).corrupt_audio_on_call_number == call_num:
            raw_sse = raw_sse.replace(b'"sha256":"', b'"sha256":"0000', 1)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        # Stream frame-by-frame with small deterministic delays so live monotonic
        # timings exercise TTFA speedup and inter-chunk gaps cleanly.
        scale = type(self).delay_scale
        for block in raw_sse.split(b"\n\n"):
            if not block:
                continue
            payload = block + b"\n\n"
            if not is_stream and b'"kind":"artifact"' in block:
                time.sleep(0.030 * scale)
            elif is_stream and b'"kind":"artifact"' in block:
                time.sleep(0.004 * scale)
            else:
                time.sleep(0.002 * scale)
            self.wfile.write(payload)
            self.wfile.flush()
        self.close_connection = True


def _sample_provenance(vlm_weights_sha256: str) -> dict:
    return {
        "schema_version": 1,
        "commit_sha": "0123456789abcdef0123456789abcdef01234567",
        "host_profile": {
            "os": "Linux",
            "gpu_arch": "sm_90",
            "driver_version": "550.90.07",
            "cuda_version": "12.4",
        },
        "binary_sha256": {
            "gateway": "a" * 64,
            "vlm": "b" * 64,
            "tts": "c" * 64,
        },
        "model_manifests": {
            "vlm": {
                "id": "qwen38",
                "weights_sha256": vlm_weights_sha256,
                "file_count": 18,
            },
            "tts": {
                "id": "qwen3-tts-12hz-1.7b",
                "weights_sha256": "d" * 64,
                "file_count": 1,
            },
        },
        "gpus": [
            {
                "index": 3,
                "name": "NVIDIA Hopper SM90",
                "exclusive": True,
                "pre_check": {"exclusive": True, "other_compute_processes": 0},
                "post_check": {"exclusive": True, "other_compute_processes": 0},
            },
            {
                "index": 5,
                "name": "NVIDIA Hopper SM90",
                "exclusive": True,
                "pre_check": {"exclusive": True, "other_compute_processes": 0},
                "post_check": {"exclusive": True, "other_compute_processes": 0},
            },
        ],
    }


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
        _LoopbackVoiceHandler.recorded_order = []
        _LoopbackVoiceHandler.fail_on_call_number = None
        _LoopbackVoiceHandler.corrupt_audio_on_call_number = None
        _LoopbackVoiceHandler.delay_scale = 1.0

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

    def test_controlled_sm90_config_schema_and_pinned_dataset_sha256(self) -> None:
        exp = va.load_experiment(CONTROLLED_SM90_CONFIG_PATH)
        self.assertEqual(exp.config["model"]["id"], "qwen38")
        self.assertEqual(exp.config["stall_threshold_ms"], 500.0)
        self.assertEqual(
            exp.config["fixed"]["audio"],
            {"voice": "vivian", "format": "wav", "sample_rate": 24000},
        )
        self.assertEqual(
            exp.config["fixed"]["sampling"],
            {"max_tokens": 256, "seed": 42, "temperature": 0.0, "top_p": 1.0},
        )
        self.assertEqual(
            exp.controlled_measurement,
            {
                "schema_version": 1,
                "warmup_runs": 1,
                "repetitions": 3,
                "schedule_mode": "interleaved_alternating",
                "require_exclusive_gpu": True,
                "max_failed_cases": 0,
            },
        )
        self.assertEqual(exp.dataset_sha256, va.sha256_bytes(PROMPTS_PATH.read_bytes()))
        matrix = va.verify_single_variable_matrix(exp)
        self.assertTrue(matrix["single_variable_verified"])

        # Verify zero workstation/absolute host paths in the versioned controlled SM90 config.
        raw_text = CONTROLLED_SM90_CONFIG_PATH.read_text(encoding="utf-8")
        self.assertNotIn("/home/", raw_text)
        self.assertNotIn("/root/", raw_text)
        self.assertNotIn("172.17.", raw_text)

    def test_interleaved_alternating_arm_schedule_and_validation(self) -> None:
        schedule = va.build_arm_schedule(3, "interleaved_alternating")
        self.assertEqual(
            schedule,
            [
                ["full_response_tts", "sentence_streaming_tts"],
                ["sentence_streaming_tts", "full_response_tts"],
                ["full_response_tts", "sentence_streaming_tts"],
            ],
        )
        self.assertEqual(va.build_arm_schedule(0, "interleaved_alternating"), [])
        with self.assertRaises(ValueError):
            va.build_arm_schedule(3, "unknown_schedule")
        with self.assertRaises(ValueError):
            va.validate_controlled_measurement(
                {
                    "schema_version": 1,
                    "warmup_runs": 1,
                    "repetitions": 3,
                    "schedule_mode": "interleaved_alternating",
                    "require_exclusive_gpu": True,
                    "max_failed_cases": 1,
                }
            )
        with self.assertRaises(ValueError):
            va.validate_controlled_measurement(
                {
                    "schema_version": 2,
                    "warmup_runs": 1,
                    "repetitions": 3,
                    "schedule_mode": "interleaved_alternating",
                    "require_exclusive_gpu": True,
                    "max_failed_cases": 0,
                }
            )

    def test_iqr_and_mad_statistical_accuracy(self) -> None:
        # Odd sample count: [1, 2, 3, 4, 5] -> Q1=2, Q3=4 => IQR=2; median=3, |x-3|=[2,1,0,1,2] => MAD=1
        self.assertEqual(va.compute_iqr([1.0, 2.0, 3.0, 4.0, 5.0]), 2.0)
        self.assertEqual(va.compute_mad([1.0, 2.0, 3.0, 4.0, 5.0]), 1.0)

        # Even sample count: [10, 20, 30, 40] -> Q1=17.5, Q3=32.5 => IQR=15.0; median=25, |x-25|=[15,5,5,15] => MAD=10.0
        self.assertEqual(va.compute_iqr([40.0, 10.0, 30.0, 20.0]), 15.0)
        self.assertEqual(va.compute_mad([40.0, 10.0, 30.0, 20.0]), 10.0)

        # Single sample -> IQR=0.0, MAD=0.0; empty -> None
        self.assertEqual(va.compute_iqr([42.0]), 0.0)
        self.assertEqual(va.compute_mad([42.0]), 0.0)
        self.assertIsNone(va.compute_iqr([]))
        self.assertIsNone(va.compute_mad([]))

        # Replay arm summary exposes iqr & mad on all 5 required latency/gap metrics.
        with tempfile.TemporaryDirectory(prefix="voice-dispersion-") as tmp:
            exp = va.load_experiment(CONFIG_PATH)
            report = va.run_ablation(exp, out_dir=Path(tmp), replay_only=True)
            for arm_name in va.VOICE_STRATEGIES:
                metrics = report["arms"][arm_name]["metrics"]
                for mname in (
                    "ttft_ms",
                    "ttfa_ms",
                    "ttfa_after_ttft_ms",
                    "terminal_ms",
                    "inter_chunk_gap_p95_ms",
                ):
                    self.assertIn("iqr", metrics[mname])
                    self.assertIn("mad", metrics[mname])
                    for base_field in ("median", "p95", "min", "max", "sample_count", "total_count", "coverage"):
                        self.assertIn(base_field, metrics[mname])
            # full_response_tts ttfa_ms samples are [1180.0, 2520.0, 3560.0]:
            # Q1 = 1180 + 0.5*(2520-1180) = 1850; Q3 = 2520 + 0.5*(3560-2520) = 3040; IQR = 1190.0
            # deviations from 2520 = [1340, 0, 1040] -> median = 1040.0
            full_ttfa = report["arms"]["full_response_tts"]["metrics"]["ttfa_ms"]
            self.assertAlmostEqual(full_ttfa["iqr"], 1190.0, places=6)
            self.assertAlmostEqual(full_ttfa["mad"], 1040.0, places=6)
            # full_response_tts inter_chunk_gap_p95_ms has no samples -> None
            full_gap = report["arms"]["full_response_tts"]["metrics"]["inter_chunk_gap_p95_ms"]
            self.assertIsNone(full_gap["iqr"])
            self.assertIsNone(full_gap["mad"])

    def test_controlled_live_end_to_end_interleaved_warmup_and_fail_closed(self) -> None:
        replay_doc = json.loads(REPLAY_PATH.read_text(encoding="utf-8"))
        _LoopbackVoiceHandler.replay_cases = {
            (c["voice_strategy"], c["prompt_id"]): c for c in replay_doc["cases"]
        }
        sm90_cfg = json.loads(CONTROLLED_SM90_CONFIG_PATH.read_text(encoding="utf-8"))
        vlm_sha = sm90_cfg["model"]["weights_sha256"]

        server = ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackVoiceHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}"
        try:
            with tempfile.TemporaryDirectory(prefix="voice-controlled-live-") as tmp:
                work = Path(tmp)
                prov_path = work / "provenance.json"
                prov_path.write_text(json.dumps(_sample_provenance(vlm_sha)), encoding="utf-8")

                # 1. Happy path: --controlled-live with valid provenance -> CONTROLLED_LIVE / COMPLETED
                _LoopbackVoiceHandler.recorded_requests = []
                _LoopbackVoiceHandler.recorded_order = []
                _LoopbackVoiceHandler.fail_on_call_number = None
                _LoopbackVoiceHandler.corrupt_audio_on_call_number = None
                _LoopbackVoiceHandler.delay_scale = 0.25

                out_ctrl = work / "out_ctrl"
                proc = _run_cli(
                    "--config",
                    str(CONTROLLED_SM90_CONFIG_PATH),
                    "--base-url",
                    base_url,
                    "--controlled-live",
                    "--provenance-json",
                    str(prov_path),
                    "--out",
                    str(out_ctrl),
                )
                self.assertEqual(proc.returncode, 0, f"stderr={proc.stderr}\nstdout={proc.stdout}")
                self.assertIn("CONTROLLED_LIVE", proc.stdout)
                self.assertIn("STATUS: COMPLETED", proc.stdout)

                # Warmup (1 * 2 * 3 = 6) + Measured (3 * 2 * 3 = 18) = 24 total HTTP requests.
                self.assertEqual(len(_LoopbackVoiceHandler.recorded_requests), 24)
                # Verify actual interleaved arm ordering on the server for measured reps 0, 1, 2.
                measured_order = _LoopbackVoiceHandler.recorded_order[6:]
                rep0_arms = [s for s, _ in measured_order[0:6]]
                rep1_arms = [s for s, _ in measured_order[6:12]]
                rep2_arms = [s for s, _ in measured_order[12:18]]
                self.assertEqual(rep0_arms, ["full_response_tts"] * 3 + ["sentence_streaming_tts"] * 3)
                self.assertEqual(rep1_arms, ["sentence_streaming_tts"] * 3 + ["full_response_tts"] * 3)
                self.assertEqual(rep2_arms, ["full_response_tts"] * 3 + ["sentence_streaming_tts"] * 3)

                report = json.loads((out_ctrl / "report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["evidence_class"], "CONTROLLED_LIVE")
                self.assertEqual(report["real_model_evidence"], "CONTROLLED_LIVE")
                self.assertEqual(report["controlled_measurement_status"], "COMPLETED")
                self.assertEqual(report["provenance"]["mode"], "controlled_live")
                self.assertEqual(report["controlled_measurement"]["failed_case_count"], 0)
                self.assertTrue(report["controlled_measurement"]["provenance_verified"])

                # Warmup excluded from measured counts (9 measured cases per arm, 6 warmup requests).
                self.assertEqual(report["warmup_summary"]["warmup_runs"], 1)
                self.assertEqual(report["warmup_summary"]["completed_request_count"], 6)
                self.assertTrue(report["warmup_summary"]["excluded_from_measured_aggregates"])
                self.assertEqual(report["arms"]["full_response_tts"]["completed_case_count"], 9)
                self.assertEqual(report["arms"]["sentence_streaming_tts"]["completed_case_count"], 9)
                self.assertEqual(report["arms"]["full_response_tts"]["total_case_count"], 9)
                self.assertEqual(report["arms"]["sentence_streaming_tts"]["total_case_count"], 9)

                # Repetitions and schedule_execution recorded faithfully.
                self.assertEqual(len(report["repetitions"]), 3)
                self.assertEqual(
                    [r["arm_order"] for r in report["repetitions"]],
                    [
                        ["full_response_tts", "sentence_streaming_tts"],
                        ["sentence_streaming_tts", "full_response_tts"],
                        ["full_response_tts", "sentence_streaming_tts"],
                    ],
                )
                self.assertEqual(len(report["schedule_execution"]), 24)
                self.assertEqual(
                    sum(1 for e in report["schedule_execution"] if e["phase"] == "warmup"), 6
                )
                self.assertEqual(
                    sum(1 for e in report["schedule_execution"] if e["phase"] == "measured"), 18
                )

                # Dispersion fields iqr & mad present on required metrics and SHA256SUMS valid.
                for arm_name in va.VOICE_STRATEGIES:
                    for mname in (
                        "ttft_ms",
                        "ttfa_ms",
                        "ttfa_after_ttft_ms",
                        "terminal_ms",
                        "inter_chunk_gap_p95_ms",
                    ):
                        self.assertIn("iqr", report["arms"][arm_name]["metrics"][mname])
                        self.assertIn("mad", report["arms"][arm_name]["metrics"][mname])
                self.assertEqual(va.verify_sha256sums(out_ctrl), 3)

                # 2. --base-url WITHOUT --controlled-live on controlled config stays UNCONTROLLED_LIVE
                _LoopbackVoiceHandler.recorded_requests = []
                _LoopbackVoiceHandler.recorded_order = []
                out_unctrl = work / "out_unctrl"
                proc_u = _run_cli(
                    "--config",
                    str(CONTROLLED_SM90_CONFIG_PATH),
                    "--base-url",
                    base_url,
                    "--out",
                    str(out_unctrl),
                )
                self.assertEqual(proc_u.returncode, 0, f"stderr={proc_u.stderr}\nstdout={proc_u.stdout}")
                self.assertIn("UNCONTROLLED_LIVE", proc_u.stdout)
                self.assertEqual(len(_LoopbackVoiceHandler.recorded_requests), 6)
                u_report = json.loads((out_unctrl / "report.json").read_text(encoding="utf-8"))
                self.assertEqual(u_report["evidence_class"], "UNCONTROLLED_LIVE")
                self.assertEqual(u_report["real_model_evidence"], "UNCONTROLLED_LIVE")
                self.assertNotIn("controlled_measurement_status", u_report)
                self.assertEqual(va.verify_sha256sums(out_unctrl), 3)

                # 3. Fail-closed: missing --provenance-json when require_exclusive_gpu=true
                proc_no_prov = _run_cli(
                    "--config",
                    str(CONTROLLED_SM90_CONFIG_PATH),
                    "--base-url",
                    base_url,
                    "--controlled-live",
                    "--out",
                    str(work / "out_no_prov"),
                )
                self.assertNotEqual(proc_no_prov.returncode, 0)
                self.assertIn("--provenance-json", proc_no_prov.stderr)

                # 4. Fail-closed: non-exclusive GPU provenance (exclusive: false)
                bad_excl = _sample_provenance(vlm_sha)
                bad_excl["gpus"][0]["exclusive"] = False
                bad_excl_path = work / "prov_non_exclusive.json"
                bad_excl_path.write_text(json.dumps(bad_excl), encoding="utf-8")
                proc_bad_excl = _run_cli(
                    "--config",
                    str(CONTROLLED_SM90_CONFIG_PATH),
                    "--base-url",
                    base_url,
                    "--controlled-live",
                    "--provenance-json",
                    str(bad_excl_path),
                    "--out",
                    str(work / "out_bad_excl"),
                )
                self.assertNotEqual(proc_bad_excl.returncode, 0)
                self.assertIn("exclusive", proc_bad_excl.stderr)

                # Also fail-closed when post_check.exclusive is false
                bad_post = _sample_provenance(vlm_sha)
                bad_post["gpus"][1]["post_check"]["exclusive"] = False
                bad_post_path = work / "prov_bad_post.json"
                bad_post_path.write_text(json.dumps(bad_post), encoding="utf-8")
                proc_bad_post = _run_cli(
                    "--config",
                    str(CONTROLLED_SM90_CONFIG_PATH),
                    "--base-url",
                    base_url,
                    "--controlled-live",
                    "--provenance-json",
                    str(bad_post_path),
                    "--out",
                    str(work / "out_bad_post"),
                )
                self.assertNotEqual(proc_bad_post.returncode, 0)
                self.assertIn("post_check", proc_bad_post.stderr)

                # 5. Fail-closed: missing binary_sha256
                bad_bin = _sample_provenance(vlm_sha)
                del bad_bin["binary_sha256"]
                bad_bin_path = work / "prov_no_bin.json"
                bad_bin_path.write_text(json.dumps(bad_bin), encoding="utf-8")
                proc_bad_bin = _run_cli(
                    "--config",
                    str(CONTROLLED_SM90_CONFIG_PATH),
                    "--base-url",
                    base_url,
                    "--controlled-live",
                    "--provenance-json",
                    str(bad_bin_path),
                    "--out",
                    str(work / "out_bad_bin"),
                )
                self.assertNotEqual(proc_bad_bin.returncode, 0)
                self.assertIn("binary_sha256", proc_bad_bin.stderr)

                # 6. Fail-closed: --controlled-live with --replay-only
                proc_replay_ctrl = _run_cli(
                    "--config",
                    str(CONFIG_PATH),
                    "--replay-only",
                    "--controlled-live",
                    "--out",
                    str(work / "out_replay_ctrl"),
                )
                self.assertNotEqual(proc_replay_ctrl.returncode, 0)
                self.assertIn("mutually exclusive", proc_replay_ctrl.stderr)

                # 7. Fail-closed: any measured HTTP failure exits nonzero and writes no CONTROLLED_LIVE report
                _LoopbackVoiceHandler.recorded_requests = []
                _LoopbackVoiceHandler.recorded_order = []
                _LoopbackVoiceHandler.fail_on_call_number = 8  # during measured rep 0
                out_fail = work / "out_fail"
                proc_fail = _run_cli(
                    "--config",
                    str(CONTROLLED_SM90_CONFIG_PATH),
                    "--base-url",
                    base_url,
                    "--controlled-live",
                    "--provenance-json",
                    str(prov_path),
                    "--out",
                    str(out_fail),
                )
                self.assertNotEqual(proc_fail.returncode, 0)
                self.assertFalse((out_fail / "report.json").exists())

                # 8. Fail-closed: corrupt audio artifact digest in measured rep exits nonzero
                _LoopbackVoiceHandler.recorded_requests = []
                _LoopbackVoiceHandler.recorded_order = []
                _LoopbackVoiceHandler.fail_on_call_number = None
                _LoopbackVoiceHandler.corrupt_audio_on_call_number = 9
                out_corrupt = work / "out_corrupt"
                proc_corrupt = _run_cli(
                    "--config",
                    str(CONTROLLED_SM90_CONFIG_PATH),
                    "--base-url",
                    base_url,
                    "--controlled-live",
                    "--provenance-json",
                    str(prov_path),
                    "--out",
                    str(out_corrupt),
                )
                self.assertNotEqual(proc_corrupt.returncode, 0)
                self.assertIn("sha256 mismatch", proc_corrupt.stderr)
                self.assertFalse((out_corrupt / "report.json").exists())
        finally:
            _LoopbackVoiceHandler.fail_on_call_number = None
            _LoopbackVoiceHandler.corrupt_audio_on_call_number = None
            _LoopbackVoiceHandler.delay_scale = 1.0
            server.shutdown()
            server.server_close()
            thread.join(timeout=5.0)


if __name__ == "__main__":
    unittest.main()
