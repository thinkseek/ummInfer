#!/usr/bin/env python3
"""Unit and contract tests for the multimodal benchmark harness.

No model is contacted. The HTTP server here is a local contract fixture; the
report it produces is labelled ``contract-test`` and is never real evidence.
"""

import hashlib
import http.server
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import threading
import time
import unittest
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load("multimodal_benchmark", "multimodal_benchmark.py")
data = bench.md


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def wav_bytes(samples=b"\x01\x00" * 1600):
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(samples)
    return out.getvalue()


JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 256


# ---- metric tests ---------------------------------------------------------

class MetricTests(unittest.TestCase):
    def test_wer_exact_and_substitution(self):
        self.assertEqual(
            bench.word_error_rate("HE HOPED THERE WOULD BE STEW",
                                  "HE HOPED THERE WOULD BE STEW"),
            (0.0, 0, 6))
        self.assertEqual(
            bench.word_error_rate("HE HOPED THERE WOULD BE STEW",
                                  "HE HOPED THERE WOULD BE BEEF"),
            (1 / 6, 1, 6))
        self.assertEqual(
            bench.word_error_rate("HE HOPED THERE WOULD BE STEW",
                                  "HE HOPED THERE WAS STEW"),
            (2 / 6, 2, 6))
        self.assertIsNone(bench.word_error_rate("", "anything")[0])

    def test_wer_normalizes_case_and_punctuation(self):
        self.assertEqual(
            bench.word_error_rate("HELLO, WORLD!", "hello world")[0], 0.0)

    def test_insertion_and_deletion(self):
        self.assertEqual(bench.word_error_rate("A B C", "A B X C")[0], 1 / 3)
        self.assertEqual(bench.word_error_rate("A B C", "A C")[0], 1 / 3)

    def test_cer(self):
        self.assertEqual(bench.character_error_rate("ABCD", "ABCD")[0], 0.0)
        self.assertEqual(bench.character_error_rate("ABCD", "ABXD")[0], 0.25)

    def test_vqa_official_accuracy(self):
        answers = ["down"] * 8 + ["table"] * 2
        self.assertEqual(bench.vqa_accuracy("down", answers), (1.0, 8))
        self.assertEqual(bench.vqa_accuracy("Down.", answers), (1.0, 8))
        self.assertEqual(bench.vqa_accuracy("up", answers), (0.0, 0))
        answers = ["two"] + ["2"] * 4 + ["three"] * 5
        self.assertEqual(bench.vqa_accuracy("2", answers), (1.0, 5))

    def test_vqa_normalize_articles_and_numbers(self):
        self.assertEqual(bench.vqa_normalize("The TWO cats!"), "2 cats")

    def test_gsm8k_marker(self):
        self.assertEqual(bench.gsm8k_extract("work\n#### 1,234"), 1234)
        self.assertEqual(bench.gsm8k_extract("#### -7"), -7)
        self.assertIsNone(bench.gsm8k_extract("the answer is 18"))
        self.assertIsNone(bench.gsm8k_extract("#### 18 then more"))

    def test_keyword_hit(self):
        self.assertEqual(
            bench.keyword_hit_rate("A blue room with a sink",
                                   ["room", "blue", "sink", "door"]), 0.75)
        self.assertIsNone(bench.keyword_hit_rate("x", []))

    def test_summary_percentiles_come_from_unified_harness(self):
        self.assertEqual(bench.ub.percentile([10, 20, 30], 50), 20)


# ---- manifest tests -------------------------------------------------------

REAL_MANIFEST = data.load_manifest()


class RealManifestTests(unittest.TestCase):
    def test_at_least_fifty_cases(self):
        self.assertGreaterEqual(len(REAL_MANIFEST["cases"]), 50)
        self.assertEqual(REAL_MANIFEST["total_cases"],
                         len(REAL_MANIFEST["cases"]))

    def test_tasks_cover_asr_gsm8k_and_vqa(self):
        counts = {}
        for case in REAL_MANIFEST["cases"]:
            counts[case["task"]] = counts.get(case["task"], 0) + 1
        self.assertGreaterEqual(counts["asr"], 1)
        self.assertGreaterEqual(counts["gsm8k"], 1)
        self.assertGreaterEqual(counts["vqa"], 1)
        self.assertEqual(counts["joint"], 6)
        self.assertEqual(counts["asr"], 16)
        self.assertEqual(counts["gsm8k"], 16)
        self.assertEqual(counts["vqa"], 16)

    def test_every_dataset_has_license_url_and_pin(self):
        for name, dataset in REAL_MANIFEST["datasets"].items():
            self.assertTrue(dataset["url"], name)
            self.assertTrue(dataset["license"], name)
            self.assertTrue(
                dataset.get("sha256") or dataset.get("archive_sha256")
                or dataset.get("questions_sha256") or dataset.get("revision"),
                name)

    def test_vqa_has_ten_answers(self):
        for case in REAL_MANIFEST["cases"]:
            if case["task"] == "vqa":
                self.assertEqual(len(case["reference"]["answers"]), 10)

    def test_joint_is_real_speech_and_pinned(self):
        joint = [case for case in REAL_MANIFEST["cases"]
                 if case["task"] == "joint"]
        self.assertTrue(joint)
        for case in joint:
            source = case["input"]
            self.assertEqual(len(source["audio_sha256"]), 64)
            self.assertEqual(len(source["image_sha256"]), 64)
            self.assertIn("datasets-server.huggingface.co", source["audio_source"])
            self.assertTrue(case["reference"]["caption"])
            self.assertTrue(case["reference"]["expected_keywords"])

    def test_validator_rejects_missing_license(self):
        broken = json.loads(json.dumps(REAL_MANIFEST))
        del broken["datasets"]["gsm8k_test"]["license"]
        self.assertTrue(data.validate_manifest(broken))

    def test_validator_rejects_duplicate_ids(self):
        broken = json.loads(json.dumps(REAL_MANIFEST))
        broken["cases"][1]["id"] = broken["cases"][0]["id"]
        self.assertTrue(data.validate_manifest(broken))


# ---- downloader tests -----------------------------------------------------

class Fixture:
    """A tiny HTTP server plus a mutable route table."""

    def __init__(self):
        self.routes = {}
        self.hits = {}
        fixture = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                fixture.hits[self.path] = fixture.hits.get(self.path, 0) + 1
                body = fixture.routes.get(self.path)
                if body is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        self.fixture = Fixture()
        self.cache = Path(tempfile.mkdtemp())

    def tearDown(self):
        self.fixture.close()

    def test_download_verifies_and_reuses_cache(self):
        body = b"x" * 100
        self.fixture.routes["/d.bin"] = body
        dest = self.cache / "nested" / "d.bin"
        data.download_verified(self.fixture.base + "/d.bin", dest, sha(body),
                               len(body))
        data.download_verified(self.fixture.base + "/d.bin", dest, sha(body),
                               len(body))
        self.assertEqual(dest.read_bytes(), body)
        self.assertEqual(self.fixture.hits["/d.bin"], 1)

    def test_sha_mismatch_leaves_no_file(self):
        body = b"y" * 50
        self.fixture.routes["/bad.bin"] = body
        dest = self.cache / "bad.bin"
        with self.assertRaises(data.DataError):
            data.download_verified(self.fixture.base + "/bad.bin", dest,
                                   "0" * 64, len(body))
        self.assertFalse(dest.exists())
        self.assertFalse(dest.with_name("bad.bin.part").exists())

    def test_upstream_larger_than_declared_is_rejected(self):
        self.fixture.routes["/big.bin"] = b"z" * 200
        with self.assertRaises(data.DataError):
            data.download_verified(self.fixture.base + "/big.bin",
                                   self.cache / "big.bin", sha(b"z" * 100),
                                   100)
        self.assertFalse((self.cache / "big.bin").exists())

    def test_corrupt_cache_is_replaced(self):
        body = b"q" * 20
        self.fixture.routes["/c.bin"] = body
        dest = self.cache / "c.bin"
        dest.write_bytes(b"corrupt")
        data.download_verified(self.fixture.base + "/c.bin", dest, sha(body),
                               len(body))
        self.assertEqual(dest.read_bytes(), body)

    def test_extract_member_and_missing(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            info = tarfile.TarInfo("root/a.txt")
            info.size = 3
            archive.addfile(info, io.BytesIO(b"abc"))
        path = self.cache / "a.tar.gz"
        path.write_bytes(buffer.getvalue())
        self.assertEqual(data.extract_members(path, ["root/a.txt"]),
                         {"root/a.txt": b"abc"})
        with self.assertRaises(data.DataError):
            data.extract_members(path, ["root/missing.txt"])

    def test_canonical_wav_rejects_wrong_shape(self):
        good = wav_bytes()
        self.assertEqual(data.canonical_wav_bytes(good), good)
        out = io.BytesIO()
        with wave.open(out, "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x00\x00" * 32)
        with self.assertRaises(data.DataError):
            data.canonical_wav_bytes(out.getvalue())

    def _speechcoco_fixture(self):
        audio, image = wav_bytes(), JPEG
        rev = "rev1"
        case = {
            "id": "joint-0",
            "input": {
                "type": "speechcoco_row", "row_index": 0, "revision": rev,
                "audio_sha256": sha(audio), "audio_bytes": len(audio),
                "image_sha256": sha(image), "image_bytes": len(image),
                "audio_source": self.fixture.base + "/0.wav",
                "image_source": self.fixture.base + "/0.jpg",
            },
            "reference": {"caption": "a caption"},
        }
        manifest = {"datasets": {"speechcoco_val": {
            "revision": rev, "rows_api": self.fixture.base + "/rows"}}}
        return case, manifest, audio, image

    def test_speechcoco_warm_cache_skips_rows_api(self):
        case, manifest, audio, image = self._speechcoco_fixture()
        d = self.cache / "speechcoco_val"
        d.mkdir(parents=True)
        (d / "row-0000.wav").write_bytes(audio)
        (d / "row-0000.jpg").write_bytes(image)
        got_audio, got_image = data._speechcoco(manifest, self.cache, case)
        self.assertEqual(got_audio, data.canonical_wav_bytes(audio))
        self.assertEqual(got_image, image)
        # No route is registered, so any HTTP use would have failed; hits
        # proves the verified cache path stayed fully offline.
        self.assertEqual(self.fixture.hits, {})

    def test_speechcoco_cold_cache_falls_back_to_rows_api(self):
        case, manifest, audio, image = self._speechcoco_fixture()
        self.fixture.routes["/0.wav"] = audio
        self.fixture.routes["/0.jpg"] = image
        self.fixture.routes["/rows?revision=rev1&offset=0&length=1"] = \
            json.dumps({"rows": [{"row": {
                "text": "a caption",
                "audio": [{"src": self.fixture.base + "/0.wav"}],
                "image": {"src": self.fixture.base + "/0.jpg"},
            }}]}).encode()
        got_audio, got_image = data._speechcoco(manifest, self.cache, case)
        self.assertEqual(got_image, image)
        self.assertEqual(got_audio, data.canonical_wav_bytes(audio))


# ---- mock model server ----------------------------------------------------

class MockModel:
    """Contract fixture only. Never evidence of model quality."""

    def __init__(self):
        self.requests = []
        self.audio_bodies = []
        self.mode = "ok"
        mock = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    self._send(200, b"{}", "application/json")
                    return
                if self.path in mock.routes:
                    body = mock.routes[self.path]
                    self._send(200, body, "application/octet-stream")
                    return
                self._send(404, b"{}", "application/json")

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                if self.path.endswith("/audio/transcriptions"):
                    mock.audio_bodies.append(raw)
                    payload = {"text": "HE HOPED THERE WOULD BE STEW",
                               "generation_metadata":
                               {"generated_token_count": 6,
                                "finish_reason": "stop"}}
                    self._send(200, json.dumps(payload).encode(),
                               "application/json")
                    return
                if not self.path.endswith("/chat/completions"):
                    self._send(404, b"{}", "application/json")
                    return
                body = json.loads(raw)
                mock.requests.append(body)
                content = body["messages"][0]["content"]
                text = content if isinstance(content, str) else " ".join(
                    part.get("text", "") for part in content
                    if part.get("type") == "text")
                if "MOCK_500" in text:
                    self._send(500, b"boom", "text/plain")
                    return
                if "MOCK_400" in text:
                    self._send(400, b'{"error":"bad request"}',
                               "application/json")
                    return
                answer = ("The total is 18.\n#### 18"
                          if "MOCK_GSM" in text else "down")
                events = []
                if "input_audio" in raw.decode(errors="ignore") or (
                        isinstance(content, list)
                        and any(part.get("type") == "input_audio"
                                for part in content)):
                    events.append({"choices": [{"delta": {}}],
                                   "unified": {"kind": "asr_final",
                                               "text": "Well dressed man "
                                                       "and woman pose.",
                                               "seq": 0}})
                events.append({"choices": [{"delta": {"content": answer},
                                            "finish_reason": None}]})
                events.append({"choices": [{"delta": {},
                                            "finish_reason": "stop"}],
                               "usage": {"completion_tokens": 7,
                                         "prompt_tokens": 11,
                                         "total_tokens": 18}})
                if "MOCK_NODONE" in text:
                    events = events[:1]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                if "MOCK_SLOW" in text:
                    time.sleep(0.2)
                for event in events:
                    self.wfile.write(
                        b"data: " + json.dumps(event).encode() + b"\n\n")
                if "MOCK_NODONE" not in text:
                    self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

            def _send(self, status, body, content_type):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.routes = {}
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def gsm8k_line(question):
    return (json.dumps({"question": question,
                        "answer": "work\n#### 18"}) + "\n").encode()


def write_fixture_manifest(root, model, questions):
    rows = [gsm8k_line(question) for question in questions]
    gsm = b"".join(rows)
    model.routes["/gsm8k.jsonl"] = gsm
    model.routes["/image.jpg"] = JPEG
    hashes = []
    for line in rows:
        hashes.append(sha(line.rstrip(b"\n")))
    manifest = {
        "schema_version": data.SCHEMA_VERSION,
        "manifest_version": "contract-test-1",
        "created": "2026-11-15",
        "total_cases": len(questions) + 1,
        "selection": {},
        "prompt_templates": {
            "gsm8k": "Question: {question}",
            "vqa": "Question: {question}\nAnswer:",
        },
        "metrics": {},
        "datasets": {
            "gsm8k_test": {"url": model.base + "/gsm8k.jsonl",
                           "license": "contract-test fixture",
                           "sha256": sha(gsm), "bytes": len(gsm)},
            "vqav2_val": {"url": model.base + "/image.jpg",
                          "license": "contract-test fixture",
                          "sha256": sha(JPEG), "bytes": len(JPEG)},
        },
        "cases": [],
    }
    for index, question in enumerate(questions):
        manifest["cases"].append({
            "id": f"gsm8k-test-{index:04d}", "task": "gsm8k",
            "dataset": "gsm8k_test", "modalities": ["text"],
            "metric": "gsm8k_exact", "language": "en",
            "tags": ["gsm8k", "contract"],
            "input": {"type": "dataset_row", "row_index": index,
                      "row_sha256": hashes[index]},
            "question": question,
            "reference": {"answer": 18},
        })
    manifest["cases"].append({
        "id": "vqa2-val-contract", "task": "vqa",
        "dataset": "vqav2_val", "modalities": ["image", "text"],
        "metric": "vqa_accuracy", "language": "en",
        "tags": ["vqa", "contract"],
        "input": {"type": "image_url", "url": model.base + "/image.jpg",
                  "sha256": sha(JPEG), "bytes": len(JPEG), "image_id": 1},
        "question": "Where is he looking?",
        "reference": {"answers": ["down"] * 10, "answer_type": "other"},
    })
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


# ---- end-to-end contract test ---------------------------------------------

class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.model = MockModel()
        self.manifest = write_fixture_manifest(
            self.tmp, self.model, ["MOCK_GSM one", "MOCK_GSM two"])

    def tearDown(self):
        self.model.close()

    def run_harness(self, evidence, *extra):
        return bench.main([
            "--base-url", self.model.base, "--model", "mock-model",
            "--manifest", str(self.manifest),
            "--cache-dir", str(self.tmp / "cache"),
            "--evidence-dir", str(evidence),
            "--evidence-class", "contract-test",
            "--max-tokens", "32", "--timeout", "10",
            *extra,
        ])

    def test_clean_contract_run_writes_all_evidence(self):
        evidence = self.tmp / "evidence"
        code = self.run_harness(evidence)
        self.assertEqual(code, 1)  # no baseline: cannot claim stability
        report = json.loads((evidence / "round-1/report.json").read_text())
        self.assertEqual(report["evidence_class"], "contract-test")
        self.assertFalse(report["real_model_evidence"])
        self.assertEqual(report["run"]["selected_cases"], 3)
        self.assertFalse(report["run"]["subset"])
        self.assertEqual(report["aggregates"]["failure_counts"],
                         {"infrastructure": 0, "public_contract": 0,
                          "quality": 0, "performance": 0})
        self.assertEqual(report["aggregates"]["tasks"]["gsm8k"]
                         ["gsm8k_accuracy"], 1.0)
        self.assertEqual(report["aggregates"]["tasks"]["vqa"]
                         ["vqa_accuracy"], 1.0)
        self.assertIsNotNone(report["aggregates"]["tasks"]["gsm8k"]["p95_ms"])
        self.assertIsNotNone(report["aggregates"]["tasks"]["gsm8k"]
                             ["tokens_per_s_p50"])
        rows = [json.loads(line) for line in
                (evidence / "round-1/cases.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertIsNotNone(row["input_sha256"])
            self.assertEqual(row["tokens_source"], "usage")
        self.assertTrue((evidence / "round-1/events/"
                         "gsm8k-test-0000.jsonl").exists())
        self.assertTrue((evidence / "round-1/answers/"
                         "gsm8k-test-0000.txt").exists())
        stability = json.loads((evidence / "stability.json").read_text())
        self.assertEqual(stability["verdict"], "NOT_REAL_EVIDENCE")
        self.assertTrue((evidence / "summary.json").exists())
        self.assertLessEqual((evidence / "run.log").stat().st_size, bench.LOG_CAP)

    def test_failures_are_classified_per_category(self):
        questions = ["MOCK_GSM ok", "MOCK_500", "MOCK_400", "MOCK_NODONE",
                     "MOCK_SLOW"]
        self.model.close()
        self.model = MockModel()
        self.manifest = write_fixture_manifest(self.tmp, self.model, questions)
        evidence = self.tmp / "evidence"
        self.run_harness(evidence, "--case-latency-cap-ms", "1")
        report = json.loads((evidence / "round-1/report.json").read_text())
        failures = json.loads((evidence / "round-1/failures.json").read_text())
        codes = {failure["code"] for failure in failures["failures"]}
        self.assertEqual(failures["counts"]["infrastructure"],
                         report["aggregates"]["failure_counts"]["infrastructure"])
        self.assertIn("http_5xx", codes)
        self.assertIn("http_4xx_valid_request", codes)
        self.assertIn("missing_done", codes)
        self.assertIn("case_latency_cap", codes)
        counts = report["aggregates"]["failure_counts"]
        self.assertGreaterEqual(counts["infrastructure"], 1)
        self.assertGreaterEqual(counts["public_contract"], 2)
        self.assertGreaterEqual(counts["performance"], 1)
        stability = json.loads((evidence / "stability.json").read_text())
        self.assertEqual(stability["verdict"], "NOT_REAL_EVIDENCE")

    def test_explicit_subset_is_reported_and_marked(self):
        evidence = self.tmp / "evidence"
        self.run_harness(evidence, "--cases", "gsm8k-test-0000")
        report = json.loads((evidence / "round-1/report.json").read_text())
        self.assertTrue(report["run"]["subset"])
        self.assertEqual(report["run"]["selected_cases"], 1)
        self.assertEqual(report["run"]["manifest_cases"], 3)
        self.assertEqual(report["run"]["excluded_case_ids"],
                         ["gsm8k-test-0001", "vqa2-val-contract"])
        checks = bench.evaluate_round(report, bench.baseline_from_report(report))
        self.assertFalse(checks["eligible"])

    def test_unknown_case_id_is_an_error(self):
        with self.assertRaises(SystemExit):
            self.run_harness(self.tmp / "evidence", "--cases", "no-such-case")

    def test_asr_client_reads_public_transcription(self):
        client = bench.PublicClient(self.model.base, "mock-model", 10)
        record = client.transcribe(wav_bytes())
        self.assertEqual(record["status"], 200)
        self.assertEqual(record["transcript"], "HE HOPED THERE WOULD BE STEW")
        self.assertEqual(record["output_tokens"], 6)
        self.assertEqual(record["endpoint"], "/v1/audio/transcriptions")

    def test_asr_route_forwards_cli_max_new_tokens(self):
        case = {
            "id": "asr-test", "task": "asr", "dataset": "x",
            "modalities": ["audio"], "metric": "wer", "language": "en",
            "tags": ["asr"],
            "reference": {"transcript": "HE HOPED THERE WOULD BE STEW"},
        }
        resolved = {"asr-test": {"type": "audio", "audio": wav_bytes(),
                                 "input_sha256": "input-sha"}}
        args = bench.parse_args([
            "--asr-route", "audio", "--asr-max-new-tokens", "777",
            "--max-tokens", "16"])[0]
        args.round = 1
        args.rounds = 1
        args.selection = {"subset": True, "selected_cases": 1,
                          "manifest_cases": 1, "excluded_case_ids": []}
        args.evidence_class = "contract-test"
        args.case_latency_cap_ms = 1000
        args.p95_tolerance = 0.1
        directory = self.tmp / "asr"
        log = bench.RunLog(self.tmp / "asr.log")
        client = bench.PublicClient(self.model.base, "mock-model", 10)
        report = bench.run_round(
            {"manifest_version": "contract-test-1", "cases": [case]},
            [case], resolved, args, directory, log, None, client)
        log.close()
        self.assertEqual(len(self.model.audio_bodies), 1)
        self.assertIn(b'name="max_new_tokens"\r\n\r\n777',
                      self.model.audio_bodies[0])
        self.assertEqual(report["config"]["asr_max_new_tokens"], 777)
        self.assertEqual(report["aggregates"]["tasks"]["asr"]["wer"], 0.0)

    def test_p95_tolerance_is_recorded_from_cli(self):
        evidence = self.tmp / "evidence-p95"
        self.run_harness(evidence, "--p95-tolerance", "0.05")
        report = json.loads((evidence / "round-1/report.json").read_text())
        self.assertEqual(report["config"]["p95_tolerance"], 0.05)
        reference = synthetic_report(p95_tolerance=0.05)
        baseline = {
            "quality": bench.quality_snapshot(reference["aggregates"]),
            "p95_ms": bench.p95_snapshot(reference["aggregates"]),
            "median_ms": bench.median_snapshot(reference["aggregates"]),
        }
        checks = bench.evaluate_round(report, baseline)
        self.assertEqual(checks["p95"]["gsm8k"]["tolerance"], 0.05)

    def test_unified_stream_captures_asr_final_and_order(self):
        client = bench.PublicClient(self.model.base, "mock-model", 10)
        case = {"id": "joint-test"}
        resolved = {"type": "audio_image", "audio": wav_bytes(),
                    "image": JPEG, "prompt": "describe",
                    "input_sha256": {"image": "i", "sent_audio": "a"}}
        payload = bench.build_request(case, resolved, "mock-model", 16)
        record = client.chat_stream(payload)
        self.assertEqual(record["transcript"],
                         "Well dressed man and woman pose.")
        self.assertTrue(record["asr_final_before_delta"])
        self.assertTrue(record["done"])
        self.assertEqual(record["finish_reason"], "stop")
        self.assertIsNotNone(record["usage"])

    def test_joint_case_writes_human_review_without_quality_score(self):
        case = {
            "id": "joint-review", "task": "joint", "dataset": "x",
            "modalities": ["audio", "image"], "metric":
            "joint_asr_wer_keyword_hit", "language": "en", "tags": ["joint"],
            "reference": {"caption": "Well dressed man and woman pose.",
                          "expected_keywords": ["man", "woman", "pose"]},
            "review": {"human_review_required": True,
                       "reason": "free-form answer"},
        }
        resolved = {"joint-review": {
            "type": "audio_image", "audio": wav_bytes(), "image": JPEG,
            "prompt": "describe",
            "input_sha256": {"image": sha(JPEG),
                             "sent_audio": sha(wav_bytes())}}}
        manifest = {"manifest_version": "contract-test-1", "cases": [case]}
        args = bench.parse_args(["--max-tokens", "16"])[0]
        args.round = 1
        args.rounds = 1
        args.selection = {"subset": True, "selected_cases": 1,
                          "manifest_cases": 1, "excluded_case_ids": []}
        args.evidence_class = "contract-test"
        args.model = "mock-model"
        args.asr_route = "audio"
        args.case_latency_cap_ms = 1000
        args.p95_tolerance = 0.1
        log = bench.RunLog(self.tmp / "joint.log")
        client = bench.PublicClient(self.model.base, "mock-model", 10)
        directory = self.tmp / "joint"
        report = bench.run_round(manifest, [case], resolved, args, directory,
                                 log, None, client)
        log.close()
        review = json.loads((directory / "human_review"
                             / "joint-review.json").read_text())
        self.assertIn("answer", review)
        self.assertIn("scored", review)
        self.assertIn("no fabricated quality score", review["note"])
        self.assertEqual(report["human_review_required"], ["joint-review"])
        self.assertIsNotNone(report["aggregates"]["tasks"]["joint"]
                             ["joint_keyword_hit"])


# ---- failure classification and gate tests --------------------------------

def synthetic_report(subset=False, infra=0, contract=0, wer=0.1,
                     vqa=0.8, gsm=0.7, p95=100.0, p50=100.0,
                     joint_wer=0.1, joint_hit=0.9, p95_tolerance=0.10):
    failures = []
    for index in range(infra):
        failures.append({"case_id": f"i{index}", "category": "infrastructure",
                         "code": "x", "severity": "P1", "message": "x"})
    for index in range(contract):
        failures.append({"case_id": f"c{index}", "category": "public_contract",
                         "code": "x", "severity": "P0", "message": "x"})
    return {
        "schema_version": "umm-multimodal-benchmark-report/v1",
        "manifest_version": "contract-test-1",
        "evidence_class": "real",
        "run": {"subset": subset, "model": "mock-model"},
        "config": {"p95_tolerance": p95_tolerance},
        "cases": [],
        "failures": failures,
        "aggregates": {
            "failure_counts": {"infrastructure": infra,
                               "public_contract": contract,
                               "quality": 0, "performance": 0},
            "tasks": {
                "asr": {"wer": wer, "cer": wer, "p95_ms": p95, "p50_ms": p50},
                "vqa": {"vqa_accuracy": vqa, "p95_ms": p95, "p50_ms": p50},
                "gsm8k": {"gsm8k_accuracy": gsm, "p95_ms": p95, "p50_ms": p50},
                "joint": {"joint_asr_wer": joint_wer,
                          "joint_keyword_hit": joint_hit,
                          "p95_ms": p95, "p50_ms": p50},
            },
        },
    }


class GateTests(unittest.TestCase):
    def setUp(self):
        report = synthetic_report()
        self.baseline = {
            "manifest_version": "contract-test-1",
            "evidence_class": "real",
            "model": "mock-model",
            "created_at": "2026-11-15T00:00:00+0000",
            "case_ids": [],
            "quality": bench.quality_snapshot(report["aggregates"]),
            "p95_ms": bench.p95_snapshot(report["aggregates"]),
            "median_ms": bench.median_snapshot(report["aggregates"]),
            "success_rate": 1.0,
        }

    def test_first_round_without_baseline_cannot_pass(self):
        checks = bench.evaluate_round(synthetic_report(), None)
        self.assertFalse(checks["passed"])
        self.assertFalse(checks["baseline_ok"])
        self.assertEqual(bench.stability_verdict([checks], False, 1),
                         "BASELINE_REQUIRED")

    def test_joint_aggregate_tolerates_skipped_case_metric(self):
        # A case whose input failed to resolve keeps metric value None.
        rows = [
            {"metric": {"name": "joint_asr_wer_keyword_hit",
                        "value": {"asr_wer": 0.0, "keyword_hit": 1.0},
                        "detail": {"errors": 0, "reference_words": 10}},
             "latency": {"total_ms": 5.0}, "failures": [],
             "tokens_per_s": None},
            {"metric": {"name": "joint_asr_wer_keyword_hit", "value": None,
                        "detail": {}},
             "latency": {"total_ms": None}, "failures": [],
             "tokens_per_s": None},
        ]
        agg = bench._task_aggregate("joint", rows)
        self.assertEqual(agg["joint_asr_wer"], 0.0)
        self.assertEqual(agg["joint_keyword_hit"], 1.0)

    def test_three_consecutive_clean_rounds_pass(self):
        checks = [bench.evaluate_round(synthetic_report(), self.baseline)
                  for _ in range(3)]
        self.assertTrue(all(item["passed"] for item in checks))
        self.assertEqual(bench.stability_verdict(checks, True, 3), "PASS")

    def test_two_rounds_are_insufficient(self):
        checks = [bench.evaluate_round(synthetic_report(), self.baseline)
                  for _ in range(2)]
        self.assertEqual(bench.stability_verdict(checks, True, 2),
                         "INSUFFICIENT_ROUNDS")

    def test_infrastructure_failure_blocks_pass(self):
        checks = [bench.evaluate_round(synthetic_report(), self.baseline)
                  for _ in range(2)]
        checks.append(bench.evaluate_round(
            synthetic_report(infra=1), self.baseline))
        self.assertEqual(bench.stability_verdict(checks, True, 3), "FAIL")

    def test_contract_p0_blocks_pass(self):
        checks = [bench.evaluate_round(synthetic_report(), self.baseline)
                  for _ in range(3)]
        checks[1] = bench.evaluate_round(
            synthetic_report(contract=1), self.baseline)
        self.assertEqual(bench.stability_verdict(checks, True, 3), "FAIL")

    def test_p95_observation_still_flagged_but_not_blocking(self):
        outside = bench.evaluate_round(
            synthetic_report(p95=111.0), self.baseline)
        self.assertFalse(outside["p95"]["asr"]["ok"])
        self.assertFalse(outside["p95"]["asr"]["blocking"])
        # Median unchanged -> the round still passes; p95 is observation only.
        self.assertTrue(outside["median_ok"])
        self.assertTrue(outside["passed"])

    def test_median_outside_ten_percent_blocks(self):
        inside = bench.evaluate_round(
            synthetic_report(p50=110.0), self.baseline)
        self.assertTrue(inside["median"]["asr"]["ok"])
        self.assertTrue(inside["passed"])
        slower = bench.evaluate_round(
            synthetic_report(p50=111.0), self.baseline)
        self.assertFalse(slower["median"]["asr"]["ok"])
        self.assertTrue(slower["median"]["asr"]["blocking"])
        self.assertFalse(slower["median_ok"])
        self.assertFalse(slower["passed"])
        faster = bench.evaluate_round(
            synthetic_report(p50=89.0), self.baseline)
        self.assertFalse(faster["median"]["asr"]["ok"])
        self.assertFalse(faster["passed"])
        self.assertEqual(slower["latency_gate"], "median_v1")

    def test_report_p95_tolerance_controls_the_observation(self):
        inside = synthetic_report(p95=105.0, p95_tolerance=0.05)
        self.assertTrue(
            bench.evaluate_round(inside, self.baseline)["p95"]["asr"]["ok"])
        outside = synthetic_report(p95=105.0, p95_tolerance=0.04)
        self.assertFalse(
            bench.evaluate_round(outside, self.baseline)["p95"]["asr"]["ok"])
        # The global default is 0.10: a report-set 0.04 must still win.
        self.assertGreater(bench.P95_TOLERANCE, 0.04)
        self.assertEqual(
            bench.evaluate_round(outside, self.baseline)["p95"]["asr"]
            ["tolerance"], 0.04)

    def test_quality_drop_beyond_tolerance_fails(self):
        inside = bench.evaluate_round(
            synthetic_report(wer=0.12), self.baseline)
        self.assertTrue(inside["quality"]["wer"]["ok"])
        outside = bench.evaluate_round(
            synthetic_report(wer=0.121), self.baseline)
        self.assertFalse(outside["quality"]["wer"]["ok"])
        accuracy = bench.evaluate_round(
            synthetic_report(vqa=0.75), self.baseline)
        self.assertTrue(accuracy["quality"]["vqa_accuracy"]["ok"])
        regressed = bench.evaluate_round(
            synthetic_report(vqa=0.74), self.baseline)
        self.assertFalse(regressed["quality"]["vqa_accuracy"]["ok"])

    def test_subset_round_is_ineligible(self):
        checks = [bench.evaluate_round(
            synthetic_report(subset=True), self.baseline) for _ in range(3)]
        self.assertFalse(checks[0]["eligible"])
        self.assertEqual(bench.stability_verdict(checks, True, 3),
                         "INELIGIBLE_SUBSET")

    def test_baseline_must_match_case_set(self):
        self.assertTrue(bench.baseline_matches(
            {"manifest_version": "contract-test-1",
             "evidence_class": "real", "case_ids": ["a", "b"]},
            {"manifest_version": "contract-test-1"},
            [{"id": "b"}, {"id": "a"}]))
        self.assertFalse(bench.baseline_matches(
            {"manifest_version": "contract-test-1",
             "evidence_class": "real", "case_ids": ["a"]},
            {"manifest_version": "contract-test-1"},
            [{"id": "a"}, {"id": "b"}]))


class LogTests(unittest.TestCase):
    def test_log_is_capped_at_64_kib(self):
        tmp = Path(tempfile.mkdtemp())
        log = bench.RunLog(tmp / "run.log")
        for index in range(400):
            log.write(f"line {index} " + "x" * 1000)
        log.close()
        size = (tmp / "run.log").stat().st_size
        self.assertLessEqual(size, bench.LOG_CAP)
        self.assertIn("line 399", (tmp / "run.log").read_text())


class SelectTests(unittest.TestCase):
    def test_default_is_full_and_unknown_is_rejected(self):
        cases, selection = bench.select_cases(REAL_MANIFEST)
        self.assertEqual(len(cases), len(REAL_MANIFEST["cases"]))
        self.assertFalse(selection["subset"])
        with self.assertRaises(ValueError):
            bench.select_cases(REAL_MANIFEST, "unknown-id")

    def test_tag_selection(self):
        cases, selection = bench.select_cases(REAL_MANIFEST, None, "vqa")
        self.assertEqual(len(cases), 16)
        self.assertTrue(selection["subset"])


if __name__ == "__main__":
    sys.exit(unittest.main())
