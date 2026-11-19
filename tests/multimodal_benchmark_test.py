#!/usr/bin/env python3
"""Offline contract test for tests/multimodal_benchmark.py.

Starts a local mock server, runs the real harness against it with
``--evidence-class contract-test`` on the real manifest and the real cached
inputs, then checks the evidence layout and the scores. The mock replays the
reference answer of each case, so a correct harness reports near-perfect
metrics. The mock is a fixture: a contract-test run can never be a baseline.

Usage: python3 -I tests/multimodal_benchmark_test.py
"""

import collections
import hashlib
import json
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "tests" / "multimodal_benchmark.py"
MANIFEST = ROOT / "benchmarks" / "multimodal_manifest.json"
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "tests"))
import multimodal_data as md  # noqa: E402
import multimodal_benchmark as mb  # noqa: E402


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def build_index(manifest):
    by_wav, by_image, by_audio, by_question = {}, {}, {}, {}
    for case in manifest["cases"]:
        kind = case["input"]["type"]
        if kind in ("librispeech_utterance", "fleurs_utterance"):
            by_wav[case["input"]["wav_sha256"]] = case
        elif kind == "image_url":
            by_image[case["input"]["sha256"]] = case
        elif kind == "speechcoco_row":
            by_audio[case["input"]["audio_sha256"]] = case
        elif kind == "dataset_row":
            by_question[case["question"]] = case
    return by_wav, by_image, by_audio, by_question


def _answer_for(case):
    task = case["task"]
    if task == "asr":
        return case["reference"]["transcript"], case["reference"]["transcript"]
    if task == "gsm8k":
        return f"#### {case['reference']['answer']}", ""
    if task == "vqa":
        common = collections.Counter(case["reference"]["answers"]).most_common(1)
        return common[0][0], ""
    if task == "joint":
        transcript = case["reference"]["caption"]
        keywords = case["reference"]["expected_keywords"]
        return " ".join(keywords), transcript
    raise AssertionError(f"unknown task {task}")


def _sse(event):
    return f"data: {json.dumps(event)}\n\n".encode()


class MockHandler(BaseHTTPRequestHandler):
    index = None

    def log_message(self, *args):
        pass

    def _body(self):
        return self.rfile.read(int(self.headers["Content-Length"]))

    def _json(self, status, payload):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        by_wav, by_image, by_audio, by_question = self.index
        body = self._body()
        if self.path == "/v1/audio/transcriptions":
            start = body.index(b"Content-Type: audio/wav\r\n\r\n")
            start += len(b"Content-Type: audio/wav\r\n\r\n")
            wav = body[start:body.rindex(b"\r\n--", start)]
            case = by_wav[_sha(wav)]
            text, _ = _answer_for(case)
            self._json(200, {"text": text, "generation_metadata": {
                "finish_reason": "stop",
                "generated_token_count": len(text.split())}})
            return
        payload = json.loads(body)
        content = payload["messages"][0]["content"]
        case = None
        transcript = ""
        if isinstance(content, str):
            case = by_question[content.split("Question: ")[-1]]
        else:
            for part in content:
                if part["type"] == "image_url":
                    raw = part["image_url"]["url"].split(",", 1)[1]
                    import base64
                    case = by_image.get(_sha(base64.b64decode(raw)), case)
                elif part["type"] == "input_audio":
                    import base64
                    audio = base64.b64decode(part["input_audio"]["data"])
                    case = by_audio.get(_sha(audio)) or by_wav.get(_sha(audio))
        assert case is not None, "mock could not identify the case"
        answer, transcript = _answer_for(case)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b": mock\n\n")
        if transcript and "audio" in case["modalities"]:
            self.wfile.write(_sse({"unified": {"kind": "asr_final",
                                               "text": transcript},
                                   "choices": []}))
        words = answer.split()
        for word in words:
            self.wfile.write(_sse({"choices": [{"delta": {
                "content": word + " "}, "finish_reason": None}]}))
        self.wfile.write(_sse({"choices": [{"delta": {},
                                            "finish_reason": "stop"}],
                               "usage": {"completion_tokens": len(words)}}))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def unit_checks():
    """Offline checks: manifest shape, hash rejection, Chinese CER bounds."""
    failures = []
    manifest = json.loads(MANIFEST.read_text())
    if manifest["total_cases"] != len(manifest["cases"]):
        failures.append("total_cases does not match len(cases)")
    zh = [case for case in manifest["cases"] if case.get("language") == "zh"]
    if len(zh) != 16:
        failures.append(f"{len(zh)} Mandarin ASR cases, expected 16")
    if any(case["metric"] != "cer" for case in zh):
        failures.append("a Mandarin case does not use the cer metric")
    speakers = {case["input"]["client_id"] for case in zh}
    if len(speakers) != len(zh):
        failures.append("Mandarin selection repeats a speaker")
    dataset = manifest["datasets"].get("fleurs_zh_cn_test") or {}
    for key in ("url", "license", "license_url", "archive_sha256",
                "revision"):
        if not dataset.get(key):
            failures.append(f"fleurs_zh_cn_test: missing {key}")
    if dataset.get("license") != "CC BY 4.0":
        failures.append("fleurs_zh_cn_test: license is not CC BY 4.0")
    for case in zh:
        for key in ("member", "sha256", "wav_sha256", "wav_bytes"):
            if not case["input"].get(key):
                failures.append(f"{case['id']}: missing input.{key}")

    with tempfile.TemporaryDirectory(prefix="umm-mm-unit-") as tmp:
        payload = b"benchmark-input-that-must-not-be-trusted" * 8
        source = Path(tmp) / "source.bin"
        source.write_bytes(payload)
        try:
            md.download_verified(source.as_uri(), Path(tmp) / "copy.bin",
                                 "0" * 64, len(payload), timeout=5)
        except md.DataError:
            pass
        else:
            failures.append("download_verified accepted a wrong SHA-256")
        if (Path(tmp) / "copy.bin").exists():
            failures.append("a rejected download left a file behind")
        good = md.sha256_bytes(payload)
        try:
            md.download_verified(source.as_uri(), Path(tmp) / "copy.bin",
                                 good, len(payload), timeout=5)
        except md.DataError as error:
            failures.append(f"download_verified rejected a good file: {error}")

    cases = [
        ("今天天气很好", "今天天气很好", 0.0),      # identical
        ("北京上海", "广州深圳", 1.0),              # every character wrong
        ("今天天气很好", "明天天气很坏", 2 / 6),    # two substitutions
        ("今天，天气 很好。", "今天天气很好", 0.0),  # punctuation removed
    ]
    for reference, hypothesis, want in cases:
        value, errors, chars = mb.chinese_character_error_rate(
            reference, hypothesis)
        if value is None or abs(value - want) > 1e-9:
            failures.append(f"CER({reference!r}, {hypothesis!r})="
                            f"{value}, expected {want}")
    value, _, _ = mb.chinese_character_error_rate("", "空")
    if value is not None:
        failures.append("CER with an empty reference must be None")
    return failures


def main():
    manifest = json.loads(MANIFEST.read_text())
    failures = unit_checks()
    MockHandler.index = build_index(manifest)
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory(prefix="umm-mm-contract-") as tmp:
        cmd = [sys.executable, "-I", str(RUNNER),
               "--base-url", f"http://127.0.0.1:{port}",
               "--model", "mock-unified", "--evidence-dir", tmp,
               "--evidence-class", "contract-test"]
        completed = subprocess.run(cmd, capture_output=True, text=True)
        # A contract-test run is deliberately not real evidence: the runner
        # reports NOT_REAL_EVIDENCE and exits 1 by design.
        if completed.returncode not in (0, 1):
            print(completed.stdout[-4000:])
            print(completed.stderr[-4000:])
            failures.append(f"runner exited {completed.returncode}")
        round_dir = Path(tmp) / "round-1"
        summary = json.loads((Path(tmp) / "summary.json").read_text()) \
            if (Path(tmp) / "summary.json").exists() else {}
        if summary.get("stability_verdict") != "NOT_REAL_EVIDENCE":
            failures.append("contract test was not labelled NOT_REAL_EVIDENCE")
        for name in ("report.json", "failures.json", "cases.jsonl",
                     "events", "answers"):
            if not (round_dir / name).exists():
                failures.append(f"missing evidence {name}")
        stability_path = Path(tmp) / "stability.json"
        if not stability_path.exists():
            failures.append("missing stability.json")
        elif json.loads(stability_path.read_text())["verdict"] != \
                "NOT_REAL_EVIDENCE":
            failures.append("a contract-test run claimed a real verdict")
        if (round_dir / "report.json").exists():
            report = json.loads((round_dir / "report.json").read_text())
            aggregates = report["aggregates"]
            counts = aggregates["failure_counts"]
            if counts["infrastructure"] != 0:
                failures.append(f"infrastructure failures: {counts}")
            if counts["public_contract"] != 0:
                failures.append(f"public-contract failures: {counts}")
            rows = (round_dir / "cases.jsonl").read_text().splitlines()
            if len(rows) != manifest["total_cases"]:
                failures.append(f"{len(rows)} case rows, expected "
                                f"{manifest['total_cases']}")
            tasks = aggregates["tasks"]
            expected = {"asr_en": ("wer", 0.0),
                        "asr_zh": ("cer", 0.0),
                        "gsm8k": ("gsm8k_accuracy", 1.0),
                        "vqa": ("vqa_accuracy", 0.0),
                        "joint": ("joint_asr_wer", 0.0)}
            for task, (key, want) in expected.items():
                value = tasks.get(task, {}).get(key)
                if value is None:
                    failures.append(f"{task}: no {key}")
                elif want and abs(value - want) > 1e-9:
                    failures.append(f"{task}: {key}={value}, expected {want}")
            if tasks.get("vqa", {}).get("vqa_accuracy", 0) < 0.9:
                failures.append("vqa: replay of the modal answer scored < 0.9")
            if tasks.get("joint", {}).get("joint_keyword_hit", 0) < 1.0:
                failures.append("joint: keyword_hit < 1.0")
            print(json.dumps({task: {k: v for k, v in values.items()
                                     if k in ("cases", "successes", "wer",
                                              "cer", "vqa_accuracy",
                                              "gsm8k_accuracy",
                                              "joint_asr_wer",
                                              "joint_keyword_hit")}
                              for task, values in tasks.items()},
                             indent=2))
    server.shutdown()
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print(f"PASS contract test: {manifest['total_cases']} cases, "
          "full evidence layout, metrics match the replay")
    return 0


if __name__ == "__main__":
    sys.exit(main())
