#!/usr/bin/env python3
"""Real multimodal benchmark and evaluation loop for ummInfer.

The harness sends public HTTP requests to a running ``umm-infer serve``:
``POST /v1/chat/completions`` (SSE) and ``POST /v1/audio/transcriptions``.
It never starts workers or servers and never writes a response. Every input is
pinned in ``benchmarks/multimodal_manifest.json`` by URL and SHA-256; the
downloader in ``benchmarks/multimodal_data.py`` verifies the bytes.

Evidence per round:
  cases.jsonl            one line per case, with metrics and timings
  events/<id>.jsonl      every SSE data payload, exactly as received
  answers/<id>.txt       final answer
  transcripts/<id>.txt   ASR transcript
  human_review/<id>.json free-form output with no objective reference
  failures.json          infrastructure / public_contract / quality / performance
  report.json            aggregates and verdict
  run.log                capped at 64 KiB

Without --evidence-dir the script prints help. A local mock is never accepted
as real evidence: pass --evidence-class contract-test for unit tests.
"""

import argparse
import base64
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import multimodal_data as md  # noqa: E402
import bench_common as ub  # noqa: E402  (percentile, capped_append)

LOG_CAP = 64 * 1024
MAX_ROUNDS = 30
REQUIRED_CONSECUTIVE = 3
MAX_EVENTS = 200000
MAX_EVENT_BYTES = 64 << 20
MAX_RESPONSE_BYTES = 8 << 20
MAX_LINE_BYTES = 1 << 20
MAX_METRIC_WORDS = 4096
CASE_LATENCY_CAP_MS = 120000
P95_TOLERANCE = 0.10
# Blocking latency gate (median_v1): nearest-rank p95 equals the per-task max
# at n<=16 and is dominated by host scheduling stalls on the shared box, so the
# gate compares per-task medians instead. p95/max stay recorded as observations.
LATENCY_GATE = "median_v2"
MEDIAN_TOLERANCE = 0.10
# n=6 joint has no representative single-round median (it averages only the
# middle two values and steps coarsely), so its latency is recorded but never
# blocks. asr/gsm8k/vqa (n=16) stay blocking under the median band.
LATENCY_OBSERVATION_TASKS = frozenset({"joint"})
QUALITY_TOLERANCE = {
    "wer": 0.02,
    "cer": 0.02,
    "vqa_accuracy": 0.05,
    "gsm8k_accuracy": 0.05,
    "joint_asr_wer": 0.05,
    "joint_keyword_hit": 0.05,
}
GAPS = [
    {"item": "zh_asr_cer", "status": "NOT_COVERED",
     "reason": ("no small, clearly licensed Mandarin test set was added. "
                "google/fleurs zh_cn exceeds the datasets-server scan limit "
                "and AISHELL-1 is about 15 GB; neither is downloaded by this "
                "manifest.")},
    {"item": "joint_objective_answer", "status": "PARTIAL",
     "reason": ("joint cases score ASR WER against the real spoken caption "
                "and caption-keyword hit in the answer. The free-form answer "
                "is saved for human review, not given a fabricated quality "
                "score.")},
]
FAILURE_CATEGORIES = ("infrastructure", "public_contract", "quality",
                      "performance")


# ---- log with a hard 64 KiB cap -------------------------------------------

class RunLog:
    """Streaming log whose on-disk size never exceeds LOG_CAP bytes."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "w", encoding="utf-8")
        self.size = 0

    def write(self, message):
        line = message.rstrip("\n") + "\n"
        self.handle.write(line)
        self.size += len(line.encode("utf-8"))
        self.handle.flush()
        if self.size > LOG_CAP:
            self._trim()

    def _trim(self):
        self.handle.close()
        raw = self.path.read_bytes()
        tail = bytearray()
        ub.capped_append(tail, raw, cap=LOG_CAP)
        while tail and tail[0] & 0xC0 == 0x80:  # do not start mid-UTF-8
            del tail[0]
        self.handle = open(self.path, "wb")
        self.handle.write(bytes(tail))
        self.handle.close()
        self.handle = open(self.path, "a", encoding="utf-8")
        self.size = len(tail)

    def close(self):
        self.handle.close()


# ---- metric helpers (unit-tested, no model) -------------------------------

def normalize_asr(text):
    text = re.sub(r"[^A-Z0-9' ]+", " ", (text or "").upper())
    return " ".join(text.split())


def _edit_distance(reference, hypothesis):
    if len(reference) > MAX_METRIC_WORDS or len(hypothesis) > MAX_METRIC_WORDS:
        raise ValueError("metric input exceeds MAX_METRIC_WORDS")
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def word_error_rate(reference, hypothesis):
    """Return (wer, errors, reference_words). None when the reference is empty."""
    ref = normalize_asr(reference).split()
    hyp = normalize_asr(hypothesis).split()
    if not ref:
        return None, None, 0
    errors = _edit_distance(ref, hyp)
    return errors / len(ref), errors, len(ref)


def character_error_rate(reference, hypothesis):
    ref = " ".join((reference or "").upper().split())
    hyp = " ".join((hypothesis or "").upper().split())
    if not ref:
        return None, None, 0
    errors = _edit_distance(list(ref), list(hyp))
    return errors / len(ref), errors, len(ref)


_VQA_ARTICLES = {"a", "an", "the"}
_VQA_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10",
}


def vqa_normalize(text):
    """Lowercase, strip punctuation/articles, map 0..10 number words."""
    text = re.sub(r"[^\w\s]", " ", (text or "").lower())
    tokens = [_VQA_NUMBERS.get(token, token) for token in text.split()
              if token not in _VQA_ARTICLES]
    return " ".join(tokens)


def vqa_accuracy(prediction, answers):
    """Official VQA accuracy: min(1, matches/3), using the 10 answers."""
    normalized = vqa_normalize(prediction)
    matches = sum(1 for answer in answers
                  if vqa_normalize(answer) == normalized)
    return min(1.0, matches / 3.0), matches


_GSM8K_MARKER = re.compile(r"####\s*(-?[\d,]+)\s*$")


def gsm8k_extract(text):
    match = _GSM8K_MARKER.search((text or "").strip())
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def keyword_hit_rate(answer, keywords):
    if not keywords:
        return None
    tokens = set(re.findall(r"[a-z0-9]+", (answer or "").lower()))
    return sum(1 for word in keywords if word in tokens) / len(keywords)


def token_count(tokenizer, text):
    if tokenizer is None or not text:
        return None, "unavailable"
    return len(tokenizer.encode(text).ids), "tokenizer"


# ---- public HTTP client ----------------------------------------------------

def normalize_base_url(base_url):
    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base


class PublicClient:
    def __init__(self, base_url, model, timeout_s=120, api_key=None):
        self.base_url = normalize_base_url(base_url)
        parts = urlsplit(self.base_url)
        if parts.scheme not in ("http", "https"):
            raise ValueError("base URL must be http or https")
        self.scheme = parts.scheme
        self.host = parts.hostname
        self.port = parts.port
        self.prefix = parts.path.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.api_key = api_key

    def _connect(self):
        factory = (http.client.HTTPSConnection if self.scheme == "https"
                   else http.client.HTTPConnection)
        return factory(self.host, self.port, timeout=self.timeout_s)

    def _headers(self, content_type):
        headers = {"Content-Type": content_type}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def health(self):
        connection = self._connect()
        try:
            connection.request("GET", "/health")
            response = connection.getresponse()
            response.read(4096)
            return response.status
        finally:
            connection.close()

    def transcribe(self, wav_bytes, max_new_tokens=256):
        """Public /v1/audio/transcriptions. Returns a transport record."""
        boundary = "----umminferbench" + hashlib.sha256(
            wav_bytes[-64:]).hexdigest()[:16]
        body = bytearray()
        for name, value in (("model", self.model),
                            ("max_new_tokens", str(max_new_tokens)),
                            ("temperature", "0"),
                            ("include_generation_metadata", "true")):
            body += f"--{boundary}\r\n".encode()
            body += (f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                     ).encode()
            body += value.encode() + b"\r\n"
        body += f"--{boundary}\r\n".encode()
        body += (b'Content-Disposition: form-data; name="file"; '
                 b'filename="audio.wav"\r\n')
        body += b"Content-Type: audio/wav\r\n\r\n"
        body += wav_bytes + b"\r\n"
        body += f"--{boundary}--\r\n".encode()

        started = time.perf_counter()
        status, raw, error = None, b"", None
        connection = self._connect()
        try:
            connection.request(
                "POST", self.path("/audio/transcriptions"), bytes(body),
                self._headers(f"multipart/form-data; boundary={boundary}"))
            response = connection.getresponse()
            status = response.status
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        except (OSError, http.client.HTTPException) as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            connection.close()
        finished = time.perf_counter()
        record = {
            "endpoint": "/v1/audio/transcriptions",
            "status": status, "error": error, "events": 0,
            "answer": "", "transcript": "", "regions": [],
            "usage": None, "finish_reason": None,
            "total_ms": (finished - started) * 1000.0,
            "first_delta_ms": None, "asr_final_ms": None, "stream_ms": None,
        }
        if error is None and len(raw) > MAX_RESPONSE_BYTES:
            record["error"] = "response exceeded the byte cap"
            return record
        if error is None and status == 200:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                record["error"] = f"response is not JSON: {exc}"
                return record
            text = parsed.get("text", "")
            metadata = parsed.get("generation_metadata") or {}
            record.update({
                "transcript": text,
                "usage": metadata or None,
                "finish_reason": metadata.get("finish_reason"),
            })
            if isinstance(metadata.get("generated_token_count"), int):
                record["output_tokens"] = metadata["generated_token_count"]
                record["tokens_source"] = "generation_metadata"
        elif error is None:
            record["error_body"] = raw[:2000].decode(errors="replace")
        return record

    def path(self, suffix):
        return self.prefix + suffix

    def chat_stream(self, payload, capture_events=None):
        """POST /v1/chat/completions with SSE. No retry, no fake terminal."""
        started = time.perf_counter()
        record = {
            "endpoint": "/v1/chat/completions",
            "status": None, "error": None, "events": 0,
            "answer": "", "transcript": "", "regions": [],
            "usage": None, "finish_reason": None,
            "total_ms": None, "first_delta_ms": None, "asr_final_ms": None,
            "stream_ms": None, "done": False,
            "asr_final_before_delta": None,
        }
        first_delta = asr_final = last_delta = None
        answer, transcript, usage, finish = [], "", None, None
        event_bytes = 0
        malformed = None
        data_lines = []
        connection = self._connect()
        try:
            body = json.dumps(payload).encode()
            connection.request(
                "POST", self.path("/chat/completions"), body,
                self._headers("application/json"))
            response = connection.getresponse()
            record["status"] = response.status
            if response.status != 200:
                record["error_body"] = response.read(4096).decode(errors="replace")
                return record
            while True:
                line = response.readline(MAX_LINE_BYTES)
                if not line:
                    break
                if len(line) >= MAX_LINE_BYTES and not line.endswith(b"\n"):
                    malformed = "SSE line exceeded the byte cap"
                    break
                if record["events"] >= MAX_EVENTS:
                    malformed = "SSE event count exceeded the cap"
                    break
                text = line.decode("utf-8", errors="replace").rstrip("\r\n")
                if not text:
                    if data_lines:
                        payload_text = "\n".join(data_lines)
                        data_lines = []
                        record["events"] += 1
                        event_bytes += len(payload_text)
                        if event_bytes > MAX_EVENT_BYTES:
                            malformed = "SSE payload exceeded the byte cap"
                            break
                        if capture_events is not None:
                            capture_events.append(payload_text)
                        if payload_text == "[DONE]":
                            record["done"] = True
                            break
                        try:
                            event = json.loads(payload_text)
                        except json.JSONDecodeError as exc:
                            malformed = f"event is not JSON: {exc}"
                            break
                        record["raw_event_count"] = record.get(
                            "raw_event_count", 0) + 1
                        if "error" in event:
                            record["error_body"] = json.dumps(event["error"])[:2000]
                            continue
                        if event.get("usage"):
                            usage = event["usage"]
                        unified = event.get("unified") or {}
                        if unified.get("kind") == "asr_final":
                            asr_final = time.perf_counter()
                            transcript = unified.get("text") or ""
                            record["asr_final_before_delta"] = first_delta is None
                        for choice in event.get("choices", []):
                            delta = choice.get("delta") or {}
                            content = delta.get("content")
                            if content:
                                if first_delta is None:
                                    first_delta = time.perf_counter()
                                last_delta = time.perf_counter()
                                answer.append(content)
                            if choice.get("finish_reason") is not None:
                                finish = choice["finish_reason"]
                    continue
                if text.startswith("data:"):
                    data_lines.append(text[5:].lstrip())
            if data_lines and not record["done"]:
                # A final event without a trailing blank line is still an event.
                payload_text = "\n".join(data_lines)
                record["events"] += 1
                if capture_events is not None and len(payload_text) <= MAX_EVENT_BYTES:
                    capture_events.append(payload_text)
                if payload_text == "[DONE]":
                    record["done"] = True
                else:
                    try:
                        event = json.loads(payload_text)
                        for choice in event.get("choices", []):
                            content = (choice.get("delta") or {}).get("content")
                            if content:
                                if first_delta is None:
                                    first_delta = time.perf_counter()
                                last_delta = time.perf_counter()
                                answer.append(content)
                            if choice.get("finish_reason") is not None:
                                finish = choice["finish_reason"]
                        unified = event.get("unified") or {}
                        if unified.get("kind") == "asr_final":
                            asr_final = time.perf_counter()
                            transcript = unified.get("text") or ""
                    except json.JSONDecodeError as exc:
                        malformed = f"event is not JSON: {exc}"
            finished = time.perf_counter()
        except (OSError, http.client.HTTPException) as exc:
            finished = time.perf_counter()
            record["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            connection.close()
        record["total_ms"] = (finished - started) * 1000.0
        if first_delta is not None:
            record["first_delta_ms"] = (first_delta - started) * 1000.0
        if asr_final is not None:
            record["asr_final_ms"] = (asr_final - started) * 1000.0
        if first_delta is not None and last_delta is not None:
            record["stream_ms"] = (last_delta - first_delta) * 1000.0
        record.update({"answer": "".join(answer), "transcript": transcript,
                       "usage": usage, "finish_reason": finish})
        if isinstance(usage, dict) and isinstance(
                usage.get("completion_tokens"), int):
            record["output_tokens"] = usage["completion_tokens"]
            record["tokens_source"] = "usage"
        if malformed and record["error"] is None:
            record["error"] = malformed
        return record


# ---- request bodies -------------------------------------------------------

def _data_url(image, mime):
    return f"data:{mime};base64," + base64.b64encode(image).decode("ascii")


def _image_mime(image):
    return "image/png" if image.startswith(b"\x89PNG") else "image/jpeg"


def build_request(case, resolved, model, max_tokens):
    """Return the exact public payload for one case."""
    kind = resolved["type"]
    if kind == "text":
        content = resolved["prompt"]
    elif kind == "image":
        content = [
            {"type": "text", "text": resolved["prompt"]},
            {"type": "image_url",
             "image_url": {"url": _data_url(resolved["image"],
                                            _image_mime(resolved["image"]))}},
        ]
    elif kind == "audio_image":
        content = [
            {"type": "text", "text": resolved["prompt"]},
            {"type": "image_url",
             "image_url": {"url": _data_url(resolved["image"],
                                            _image_mime(resolved["image"]))}},
            {"type": "input_audio",
             "input_audio": {"data": base64.b64encode(
                 resolved["audio"]).decode("ascii"), "format": "wav"}},
        ]
    elif kind == "audio":
        content = [
            {"type": "input_audio",
             "input_audio": {"data": base64.b64encode(
                 resolved["audio"]).decode("ascii"), "format": "wav"}},
        ]
    else:
        raise ValueError(f"{case['id']}: no chat body for {kind}")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
    }
    return payload


def request_summary(payload, resolved):
    """A small, auditable view of the request; payload bytes stay in the cache."""
    parts = []
    hashes = resolved.get("input_sha256")
    hashes = hashes if isinstance(hashes, dict) else {}
    content = payload["messages"][0]["content"]
    if isinstance(content, str):
        parts.append({"type": "text", "chars": len(content)})
    else:
        for part in content:
            if part["type"] == "text":
                parts.append({"type": "text", "chars": len(part["text"])})
            elif part["type"] == "image_url":
                parts.append({"type": "image_url", "data_url": True,
                              "sha256": hashes.get("image")})
            elif part["type"] == "input_audio":
                parts.append({"type": "input_audio", "format": "wav",
                              "bytes": len(resolved["audio"]),
                              "sha256": hashes.get("sent_audio")})
    return {"model": payload["model"], "stream": True,
            "max_tokens": payload["max_tokens"], "parts": parts}


# ---- scoring and classification ------------------------------------------

def score_case(case, record):
    task = case["task"]
    if task == "asr":
        reference = case["reference"]["transcript"]
        hypothesis = record.get("transcript") or ""
        wer, errors, words = word_error_rate(reference, hypothesis)
        cer, char_errors, chars = character_error_rate(reference, hypothesis)
        return {"name": "wer", "value": wer,
                "detail": {"errors": errors, "reference_words": words,
                           "cer": cer, "character_errors": char_errors,
                           "reference_characters": chars}}
    if task == "vqa":
        value, matches = vqa_accuracy(record.get("answer", ""),
                                      case["reference"]["answers"])
        return {"name": "vqa_accuracy", "value": value,
                "detail": {"matches": matches, "answers":
                           case["reference"]["answers"]}}
    if task == "gsm8k":
        predicted = gsm8k_extract(record.get("answer", ""))
        gold = case["reference"]["answer"]
        return {"name": "gsm8k_exact", "value": float(predicted == gold),
                "detail": {"predicted": predicted, "gold": gold,
                           "marker_present": predicted is not None}}
    if task == "joint":
        reference = case["reference"]["caption"]
        wer, errors, words = word_error_rate(
            reference, record.get("transcript", ""))
        hit = keyword_hit_rate(record.get("answer", ""),
                               case["reference"]["expected_keywords"])
        return {"name": "joint_asr_wer_keyword_hit",
                "value": {"asr_wer": wer, "keyword_hit": hit},
                "detail": {"errors": errors, "reference_words": words,
                           "keywords": case["reference"]["expected_keywords"]}}
    raise ValueError(f"{case['id']}: unknown task {task}")


def classify(case, record, latency_cap_ms=CASE_LATENCY_CAP_MS):
    """Return failure dicts. Transport problems never become quality failures."""
    failures = []

    def add(category, code, severity, message):
        failures.append({"case_id": case["id"], "category": category,
                         "code": code, "severity": severity,
                         "message": message})

    status = record.get("status")
    if record.get("error") and (status is None or status >= 500):
        add("infrastructure", "transport_error", "P1", record["error"])
    elif status is not None and status >= 500:
        add("infrastructure", "http_5xx", "P1",
            f"server returned {status}")
    if status is not None and 400 <= status < 500:
        add("public_contract", "http_4xx_valid_request", "P0",
            f"valid manifest request rejected with {status}: "
            + (record.get("error_body") or "")[:500])
    if (record.get("error") and status == 200
            and record["endpoint"] == "/v1/audio/transcriptions"):
        add("public_contract", "bad_transcription_response", "P0",
            record["error"])
    elif record.get("error") and status == 200:
        add("public_contract", "malformed_stream", "P0", record["error"])
    if status == 200 and record["endpoint"] == "/v1/chat/completions":
        if not record.get("done"):
            add("public_contract", "missing_done", "P1",
                "SSE ended without [DONE]")
        if record.get("finish_reason") is None and record.get("done"):
            add("public_contract", "missing_finish_reason", "P1",
                "stream completed without finish_reason")
        if "audio" in case["modalities"] and not record.get("transcript"):
            add("public_contract", "missing_asr_final", "P0",
                "unified audio request produced no asr_final transcript")
        if record.get("asr_final_before_delta") is False:
            add("public_contract", "asr_final_after_delta", "P1",
                "asr_final arrived after a VLM delta")
    if not any(f["category"] in ("infrastructure", "public_contract")
               for f in failures):
        empty = not (record.get("answer") or "").strip()
        no_transcript = (record.get("transcript") is not None
                         and not record["transcript"].strip()
                         and "audio" in case["modalities"])
        if case["task"] in ("vqa", "gsm8k", "joint") and empty:
            add("quality", "empty_answer", "P2", "the answer is empty")
        elif no_transcript:
            add("quality", "empty_transcript", "P2",
                "the ASR transcript is empty")
    metric = record.get("metric") or {}
    if not any(f["category"] in ("infrastructure", "public_contract")
               for f in failures):
        value = metric.get("value")
        if case["task"] == "asr" and isinstance(value, float) and value > 0.5:
            add("quality", "asr_wer_high", "P2", f"WER {value:.3f} > 0.50")
        elif case["task"] == "vqa" and value == 0.0:
            add("quality", "vqa_wrong", "P2",
                "answer matches none of the 10 reference answers")
        elif case["task"] == "gsm8k" and value == 0.0:
            add("quality", "gsm8k_wrong", "P2",
                f"answer is not the gold integer: {metric.get('detail')}")
        elif case["task"] == "joint" and isinstance(value, dict):
            wer = value.get("asr_wer")
            hit = value.get("keyword_hit")
            if (isinstance(wer, float) and wer > 0.5) or (
                    isinstance(hit, float) and hit < 0.5):
                add("quality", "joint_metric_low", "P2",
                    f"asr_wer={wer} keyword_hit={hit}")
    total = record.get("total_ms")
    if total is not None and total > latency_cap_ms:
        add("performance", "case_latency_cap", "P2",
            f"latency {total:.0f} ms > {latency_cap_ms} ms")
    return failures


# ---- aggregation ----------------------------------------------------------

def _task_aggregate(task, rows):
    metrics = [row["metric"] for row in rows if row.get("metric")]
    latencies = [row["latency"]["total_ms"] for row in rows
                 if row["latency"].get("total_ms") is not None]
    tokens = [row["tokens_per_s"] for row in rows
              if row.get("tokens_per_s") is not None]
    usable = [row for row in rows
              if not any(f["category"] in ("infrastructure", "public_contract")
                         for f in row["failures"])]
    out = {
        "cases": len(rows),
        "successes": len(usable),
        "success_rate": len(usable) / len(rows) if rows else None,
        "p50_ms": ub.percentile(latencies, 50),
        "p95_ms": ub.percentile(latencies, 95),
        "tokens_per_s_p50": ub.percentile(tokens, 50),
        "tokens_per_s_p95": ub.percentile(tokens, 95),
    }
    if task == "asr":
        errors = sum(m.get("detail", {}).get("errors") or 0 for m in metrics)
        words = sum(m.get("detail", {}).get("reference_words") or 0
                    for m in metrics)
        char_errors = sum(m.get("detail", {}).get("character_errors") or 0
                          for m in metrics)
        chars = sum(m.get("detail", {}).get("reference_characters") or 0
                    for m in metrics)
        out["wer"] = errors / words if words else None
        out["cer"] = char_errors / chars if chars else None
    elif task == "vqa":
        values = [m["value"] for m in metrics if isinstance(m.get("value"), float)]
        out["vqa_accuracy"] = statistics.fmean(values) if values else None
    elif task == "gsm8k":
        values = [m["value"] for m in metrics if isinstance(m.get("value"), float)]
        out["gsm8k_accuracy"] = statistics.fmean(values) if values else None
    elif task == "joint":
        # A skipped case (for example an input that failed to resolve) keeps
        # metric value None; it must not crash the aggregate.
        joint_metrics = [m for m in metrics if isinstance(m.get("value"), dict)]
        wers = [(m["value"].get("asr_wer"), m.get("detail", {}).get("errors"),
                 m.get("detail", {}).get("reference_words"))
                for m in joint_metrics]
        errors = sum(e or 0 for _, e, _ in wers)
        words = sum(w or 0 for _, _, w in wers)
        hits = [m["value"].get("keyword_hit") for m in joint_metrics
                if isinstance(m["value"].get("keyword_hit"), float)]
        out["joint_asr_wer"] = errors / words if words else None
        out["joint_keyword_hit"] = statistics.fmean(hits) if hits else None
    return out


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["task"], []).append(row)
    aggregates = {task: _task_aggregate(task, group)
                  for task, group in sorted(groups.items())}
    latencies = [row["latency"]["total_ms"] for row in rows
                 if row["latency"].get("total_ms") is not None]
    usable = [row for row in rows
              if not any(f["category"] in ("infrastructure", "public_contract")
                         for f in row["failures"])]
    counts = {category: 0 for category in FAILURE_CATEGORIES}
    for row in rows:
        for failure in row["failures"]:
            counts[failure["category"]] += 1
    return {
        "cases": len(rows),
        "successes": len(usable),
        "success_rate": len(usable) / len(rows) if rows else None,
        "p50_ms": ub.percentile(latencies, 50),
        "p95_ms": ub.percentile(latencies, 95),
        "tasks": aggregates,
        "failure_counts": counts,
    }


def quality_snapshot(aggregates):
    snapshot = {}
    for task, values in aggregates["tasks"].items():
        for key in ("wer", "cer", "vqa_accuracy", "gsm8k_accuracy",
                    "joint_asr_wer", "joint_keyword_hit"):
            if key in values:
                snapshot[key] = values[key]
    return snapshot


def p95_snapshot(aggregates):
    return {task: values.get("p95_ms")
            for task, values in aggregates["tasks"].items()}


def median_snapshot(aggregates):
    return {task: values.get("p50_ms")
            for task, values in aggregates["tasks"].items()}


# ---- stability gate -------------------------------------------------------

def _regression(baseline_value, current_value, tolerance, lower_is_better):
    if baseline_value is None or current_value is None:
        return None
    delta = (current_value - baseline_value if lower_is_better
             else baseline_value - current_value)
    return {"baseline": baseline_value, "current": current_value,
            "delta": delta, "tolerance": tolerance,
            "ok": delta <= tolerance + 1e-9}


def evaluate_round(report, baseline):
    """Judge one full round against the accepted baseline."""
    aggregates = report["aggregates"]
    checks = {"subset": report["run"]["subset"],
              "infrastructure": aggregates["failure_counts"]["infrastructure"],
              "contract_p0_p1": sum(
                  1 for failure in report["failures"]
                  if failure["category"] == "public_contract"
                  and failure["severity"] in ("P0", "P1"))}
    checks["infrastructure_ok"] = checks["infrastructure"] == 0
    checks["contract_ok"] = checks["contract_p0_p1"] == 0
    checks["quality"] = {}
    checks["p95"] = {}
    checks["median"] = {}
    checks["baseline_ok"] = baseline is not None
    p95_tolerance = report["config"]["p95_tolerance"]
    if baseline:
        current_quality = quality_snapshot(aggregates)
        for key, tolerance in QUALITY_TOLERANCE.items():
            lower = key in ("wer", "cer", "joint_asr_wer")
            comparison = _regression(baseline["quality"].get(key),
                                     current_quality.get(key), tolerance,
                                     lower_is_better=lower)
            if comparison is not None:
                checks["quality"][key] = comparison
        # Observation only: p95 == max at n<=16 and tracks host stalls. Recorded
        # for visibility, never flips the verdict under the median_v1 gate.
        current_p95 = p95_snapshot(aggregates)
        for task, value in current_p95.items():
            base = baseline["p95_ms"].get(task)
            if base is None or value is None or base == 0:
                continue
            delta = (value - base) / base
            checks["p95"][task] = {
                "baseline_ms": base, "current_ms": value,
                "delta": delta, "tolerance": p95_tolerance,
                "ok": abs(delta) <= p95_tolerance + 1e-9,
                "blocking": False}
        # Blocking latency statistic: per-task median within a symmetric band.
        # n=6 joint is observation-only (see LATENCY_OBSERVATION_TASKS).
        current_median = median_snapshot(aggregates)
        for task, value in current_median.items():
            base = baseline["median_ms"].get(task)
            if base is None or value is None or base == 0:
                continue
            delta = (value - base) / base
            blocking = task not in LATENCY_OBSERVATION_TASKS
            checks["median"][task] = {
                "baseline_ms": base, "current_ms": value,
                "delta": delta, "tolerance": MEDIAN_TOLERANCE,
                "ok": abs(delta) <= MEDIAN_TOLERANCE + 1e-9,
                "blocking": blocking}
    quality_ok = all(item["ok"] for item in checks["quality"].values())
    p95_ok = all(item["ok"] for item in checks["p95"].values())
    median_ok = all(item["ok"] for task, item in checks["median"].items()
                    if item["blocking"])
    eligible = not checks["subset"]
    passed = bool(eligible and checks["infrastructure_ok"]
                  and checks["contract_ok"] and quality_ok and median_ok
                  and checks["baseline_ok"])
    checks["quality_ok"] = quality_ok
    checks["p95_ok"] = p95_ok
    checks["median_ok"] = median_ok
    checks["latency_gate"] = LATENCY_GATE
    checks["eligible"] = eligible
    checks["passed"] = passed
    return checks


def stability_verdict(round_checks, baseline_accepted, rounds_requested):
    """PASS needs 3 consecutive full, clean rounds against an accepted baseline."""
    if baseline_accepted is False or not round_checks:
        return "BASELINE_REQUIRED"
    if any(not item["eligible"] for item in round_checks):
        return "INELIGIBLE_SUBSET"
    if any(not item["passed"] for item in round_checks):
        return "FAIL"
    consecutive = 0
    for item in reversed(round_checks):
        if item["passed"]:
            consecutive += 1
        else:
            break
    if len(round_checks) < REQUIRED_CONSECUTIVE:
        return "INSUFFICIENT_ROUNDS"
    return "PASS" if consecutive >= REQUIRED_CONSECUTIVE else "FAIL"


# ---- case selection -------------------------------------------------------

def select_cases(manifest, case_ids=None, tags=None):
    """Default is the full manifest. A subset must be explicit."""
    cases = manifest["cases"]
    if case_ids:
        wanted = [item.strip() for item in case_ids.split(",") if item.strip()]
        known = {case["id"] for case in cases}
        unknown = [item for item in wanted if item not in known]
        if unknown:
            raise ValueError("unknown case id: " + ", ".join(unknown))
        selected = [case for case in cases if case["id"] in set(wanted)]
    elif tags:
        wanted = {item.strip() for item in tags.split(",") if item.strip()}
        selected = [case for case in cases
                    if wanted & set(case.get("tags", []))]
        if not selected:
            raise ValueError("no case has a requested tag")
    else:
        selected = list(cases)
    subset = len(selected) != len(cases)
    excluded = [case["id"] for case in cases
                if case["id"] not in {item["id"] for item in selected}]
    return selected, {"subset": subset, "selected_cases": len(selected),
                      "manifest_cases": len(cases),
                      "excluded_case_ids": excluded}


# ---- evidence writing -----------------------------------------------------

def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---- one round ------------------------------------------------------------

def run_round(manifest, cases, resolved, args, round_dir, log, tokenizer,
              client):
    round_dir.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    for index, case in enumerate(cases, 1):
        case_id = case["id"]
        preflight = []
        if case_id not in resolved:
            preflight.append({"case_id": case_id, "category": "infrastructure",
                              "code": "input_unavailable", "severity": "P1",
                              "message": "input bytes were not resolved"})
        record = {"answer": "", "transcript": "", "usage": None,
                  "finish_reason": None, "status": None, "error": None,
                  "endpoint": None, "events": 0, "error_body": "",
                  "total_ms": None, "first_delta_ms": None,
                  "asr_final_ms": None, "stream_ms": None, "done": False,
                  "asr_final_before_delta": None}
        metric = {"name": case["metric"], "value": None, "detail": {}}
        request = None
        if preflight:
            record["error"] = "input_unavailable"
            failures.extend(preflight)
        else:
            try:
                item = resolved[case_id]
                if case["task"] == "asr" and args.asr_route == "audio":
                    log.write(f"[{index}/{len(cases)}] {case_id}: "
                              "/v1/audio/transcriptions")
                    record = client.transcribe(
                        item["audio"], args.asr_max_new_tokens)
                else:
                    payload = build_request(case, item, args.model,
                                            args.max_tokens)
                    request = request_summary(payload, item)
                    log.write(f"[{index}/{len(cases)}] {case_id}: "
                              "/v1/chat/completions")
                    events = []
                    record = client.chat_stream(payload, capture_events=events)
                    write_jsonl(round_dir / "events" / f"{case_id}.jsonl",
                                [{"index": i, "data": payload_text}
                                 for i, payload_text in enumerate(events)])
                if record.get("transcript") is None:
                    record["transcript"] = ""
                if case["task"] == "asr":
                    record["transcript"] = (record.get("transcript")
                                            or record.get("answer") or "")
                metric = score_case(case, record)
                record["metric"] = metric
                tokens = record.get("output_tokens")
                source = record.get("tokens_source")
                if tokens is None:
                    tokens, source = token_count(tokenizer,
                                                 record.get("answer"))
                    if tokens is None and record.get("answer"):
                        tokens = len(record["answer"].split())
                        source = "word_estimate"
                record["output_tokens"] = tokens
                record["tokens_source"] = source
                if tokens is not None:
                    # A single delta makes first_delta and last_delta the same
                    # clock reading, so stream_ms is near zero. Fall back to
                    # the total request time instead of dividing by it.
                    basis = record.get("stream_ms") or 0
                    if basis < 1.0:
                        basis = record.get("total_ms") or 0
                    if basis:
                        record["tokens_per_s"] = tokens / (basis / 1000.0)
                failures.extend(classify(case, record,
                                          args.case_latency_cap_ms))
            except Exception as exc:  # a harness bug must not be a model miss
                record["error"] = f"{type(exc).__name__}: {exc}"
                failures.append({"case_id": case_id,
                                 "category": "infrastructure",
                                 "code": "harness_error", "severity": "P1",
                                 "message": record["error"]})
        answer = record.get("answer") or ""
        transcript = record.get("transcript") or ""
        if answer:
            write_text(round_dir / "answers" / f"{case_id}.txt", answer)
        if transcript and case["modalities"].count("audio"):
            write_text(round_dir / "transcripts" / f"{case_id}.txt", transcript)
        case_failures = [f for f in failures if f["case_id"] == case_id]
        row = {
            "case_id": case_id,
            "task": case["task"],
            "modalities": case["modalities"],
            "metric_name": case["metric"],
            "metric": metric,
            "endpoint": record.get("endpoint"),
            "http_status": record.get("status"),
            "answer": answer,
            "transcript": transcript,
            "usage": record.get("usage"),
            "finish_reason": record.get("finish_reason"),
            "events": record.get("events"),
            "done": record.get("done"),
            "input_sha256": resolved.get(case_id, {}).get("input_sha256"),
            "request": request,
            "latency": {
                "total_ms": record.get("total_ms"),
                "first_delta_ms": record.get("first_delta_ms"),
                "asr_final_ms": record.get("asr_final_ms"),
                "stream_ms": record.get("stream_ms"),
            },
            "output_tokens": record.get("output_tokens"),
            "tokens_source": record.get("tokens_source"),
            "tokens_per_s": record.get("tokens_per_s"),
            "failures": case_failures,
            "human_review_required": bool(
                case.get("review", {}).get("human_review_required")
                or (case["task"] == "asr" and args.asr_route == "chat")),
        }
        rows.append(row)
        if row["human_review_required"]:
            write_json(round_dir / "human_review" / f"{case_id}.json", {
                "case_id": case_id,
                "reason": case.get("review", {}).get("reason",
                                                     "no objective reference"),
                "question": case.get("question"),
                "transcript": transcript,
                "answer": answer,
                "scored": metric,
                "note": "Scores above cover only verifiable submetrics. "
                        "The free-form answer has no fabricated quality score.",
            })
    write_jsonl(round_dir / "cases.jsonl", rows)
    aggregates = aggregate(rows)
    failure_report = {
        "schema_version": "umm-multimodal-failures/v1",
        "counts": {category: sum(
            1 for failure in failures if failure["category"] == category)
            for category in FAILURE_CATEGORIES},
        "failures": failures,
    }
    write_json(round_dir / "failures.json", failure_report)
    report = {
        "schema_version": "umm-multimodal-benchmark-report/v1",
        "manifest_version": manifest["manifest_version"],
        "evidence_class": args.evidence_class,
        "real_model_evidence": args.evidence_class == "real",
        "run": {
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "base_url": client.base_url,
            "model": args.model,
            "asr_route": args.asr_route,
            "round": args.round,
            "rounds_requested": args.rounds,
            **args.selection,
        },
        "config": {
            "p95_tolerance": args.p95_tolerance,
            "quality_tolerance": QUALITY_TOLERANCE,
            "case_latency_cap_ms": args.case_latency_cap_ms,
            "max_tokens": args.max_tokens,
            "asr_max_new_tokens": args.asr_max_new_tokens,
            "max_rounds": MAX_ROUNDS,
            "required_consecutive_rounds": REQUIRED_CONSECUTIVE,
            "log_cap_bytes": LOG_CAP,
        },
        "coverage": md.coverage(manifest),
        "gaps": GAPS,
        "cases": rows,
        "aggregates": aggregates,
        "failures": failures,
        "human_review_required": [row["case_id"] for row in rows
                                  if row["human_review_required"]],
    }
    write_json(round_dir / "report.json", report)
    return report


# ---- baseline -------------------------------------------------------------

def baseline_from_report(report):
    aggregates = report["aggregates"]
    return {
        "schema_version": "umm-multimodal-baseline/v1",
        "manifest_version": report["manifest_version"],
        "model": report.get("run", {}).get("model"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "evidence_class": report.get("evidence_class"),
        "case_ids": sorted(row["case_id"] for row in report["cases"]),
        "quality": quality_snapshot(aggregates),
        "p95_ms": p95_snapshot(aggregates),
        "median_ms": median_snapshot(aggregates),
        "success_rate": aggregates["success_rate"],
    }


def baseline_matches(baseline, manifest, cases):
    if baseline.get("manifest_version") != manifest["manifest_version"]:
        return False
    if baseline.get("evidence_class") != "real":
        return False
    return sorted(case["id"] for case in cases) == baseline.get("case_ids")


# ---- entry point ----------------------------------------------------------

def load_tokenizer(path):
    if not path:
        return None
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--manifest")
    parser.add_argument("--cache-dir",
                        default=os.environ.get(
                            "UMMINFER_BENCH_CACHE",
                            str(Path.home() /
                                ".cache/umminfer-multimodal-benchmark")))
    parser.add_argument("--evidence-dir")
    parser.add_argument("--cases",
                        help="explicit comma-separated case ids; default is all")
    parser.add_argument("--tags",
                        help="explicit comma-separated tags; default is all")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--baseline", help="accepted baseline JSON to compare")
    parser.add_argument("--accept-baseline", action="store_true",
                        help="write baseline.json when this full round is clean")
    parser.add_argument("--evidence-class", choices=("real", "contract-test"),
                        default="real")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--asr-max-new-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--asr-route", choices=("audio", "chat"),
                        default="audio")
    parser.add_argument("--case-latency-cap-ms", type=int,
                        default=CASE_LATENCY_CAP_MS)
    parser.add_argument("--p95-tolerance", type=float, default=P95_TOLERANCE)
    parser.add_argument("--tokenizer")
    parser.add_argument("--verify-data", action="store_true",
                        help="download and verify every input; no server")
    parser.add_argument("--list", action="store_true",
                        help="print the selected case ids and exit")
    parser.add_argument("--api-key-env", default="UMMINFER_BENCH_API_KEY",
                        help="environment variable that holds the key")
    args = parser.parse_args(argv)
    if args.rounds < 1 or args.rounds > MAX_ROUNDS:
        parser.error(f"--rounds must be 1..{MAX_ROUNDS}")
    if args.max_tokens < 1 or args.asr_max_new_tokens < 1:
        parser.error("token limits must be positive")
    return args, parser


def main(argv=None):
    args, parser = parse_args(argv)
    manifest = md.load_manifest(args.manifest)
    try:
        cases, selection = select_cases(manifest, args.cases, args.tags)
    except ValueError as error:
        parser.error(str(error))
    args.selection = selection
    print(json.dumps({"manifest_version": manifest["manifest_version"],
                      "coverage": md.coverage(manifest),
                      "selection": selection}, indent=2, ensure_ascii=False))
    if args.list:
        for case in cases:
            print(case["id"])
        return 0
    if args.verify_data:
        log = lambda message: print(message)
        try:
            md.resolve_cases(manifest, cases, args.cache_dir, log=log)
        except md.DataError as error:
            print(f"DATA_ERROR: {error}")
            return 1
        print(f"PASS data: {len(cases)} cases verified "
              f"(manifest {manifest['manifest_version']})")
        return 0
    if not (args.base_url and args.model and args.evidence_dir):
        parser.print_help()
        print("\nThis harness needs --base-url, --model and --evidence-dir. "
              "Use --verify-data to check inputs only.")
        return 0

    evidence = Path(args.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    log = RunLog(evidence / "run.log")
    tokenizer = load_tokenizer(args.tokenizer)
    api_key = os.environ.get(args.api_key_env) or None
    client = PublicClient(args.base_url, args.model, args.timeout, api_key)
    log.write(f"manifest={manifest['manifest_version']} "
              f"cases={len(cases)} subset={selection['subset']} "
              f"evidence_class={args.evidence_class}")
    try:
        status = client.health()
    except OSError as error:
        status = None
        log.write(f"health check failed: {error}")
    if status != 200:
        log.write(f"server is not healthy (status={status})")
    if args.accept_baseline:
        if args.baseline:
            parser.error("--accept-baseline and --baseline are mutually exclusive")
        if args.rounds != 1:
            parser.error("--accept-baseline establishes one baseline round; "
                         "run the gate afterwards with --baseline")
        if selection["subset"]:
            parser.error("--accept-baseline requires the full case set")
        if args.evidence_class != "real":
            parser.error("a contract test cannot establish a real baseline")

    asr_archive = None
    if any(case["input"]["type"] == "librispeech_utterance" for case in cases):
        try:
            asr_archive = md.ensure_librispeech(manifest, args.cache_dir)
            log.write(f"librispeech archive verified: {asr_archive}")
        except (md.DataError, OSError) as error:
            log.write(f"librispeech archive unavailable: {error}")
    resolved = {}
    for case in cases:
        if (case["input"]["type"] == "librispeech_utterance"
                and asr_archive is None):
            log.write(f"input unavailable: {case['id']}: archive is missing")
            continue
        try:
            resolved[case["id"]] = md.resolve_case(
                manifest, case, args.cache_dir, asr_archive=asr_archive)
            log.write(f"input verified: {case['id']}")
        except (md.DataError, OSError) as error:
            log.write(f"input unavailable: {case['id']}: {error}")

    baseline = None
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        if not baseline_matches(baseline, manifest, cases):
            log.write("baseline does not match this manifest or case set")
            print("BASELINE_MISMATCH: manifest version or case set differs")
            return 2

    round_checks = []
    reports = []
    for number in range(1, args.rounds + 1):
        args.round = number
        round_dir = evidence / f"round-{number}"
        report = run_round(manifest, cases, resolved, args, round_dir, log,
                           tokenizer, client)
        reports.append(report)
        checks = evaluate_round(report, baseline)
        round_checks.append(checks)
        log.write(f"round {number}: passed={checks['passed']} "
                  f"infra={checks['infrastructure']} "
                  f"contract={checks['contract_p0_p1']}")
        if args.accept_baseline and number == 1:
            if checks["subset"] or not checks["infrastructure_ok"] or \
                    not checks["contract_ok"]:
                log.write("baseline refused: round is not a clean full run")
                print("BASELINE_REFUSED: the round is not a clean full run")
                return 2
            baseline = baseline_from_report(report)
            write_json(evidence / "baseline.json", baseline)
            log.write("baseline accepted")
        if not checks["passed"] and number == 1 and not args.accept_baseline \
                and baseline is None:
            break  # no baseline: further rounds cannot prove stability
        if any(not item["passed"] for item in round_checks):
            break  # a failed round blocks PASS for this run; do not burn rounds
        if len(round_checks) >= REQUIRED_CONSECUTIVE:
            break  # the requirement is satisfied; no extra spending

    verdict = ("BASELINE_ESTABLISHED" if args.accept_baseline
               else stability_verdict(round_checks,
                                      baseline is not None, args.rounds))
    if args.evidence_class != "real":
        verdict = "NOT_REAL_EVIDENCE"
    stability = {
        "schema_version": "umm-multimodal-stability/v1",
        "manifest_version": manifest["manifest_version"],
        "model": args.model,
        "rounds_requested": args.rounds,
        "rounds_run": len(round_checks),
        "required_consecutive_rounds": REQUIRED_CONSECUTIVE,
        "latency_gate": LATENCY_GATE,
        "p95_tolerance": args.p95_tolerance,
        "median_tolerance": MEDIAN_TOLERANCE,
        "quality_tolerance": QUALITY_TOLERANCE,
        "baseline_accepted": baseline is not None,
        "baseline": {key: baseline[key] for key in
                     ("created_at", "manifest_version", "model")
                     if key in baseline} if baseline else None,
        "rounds": round_checks,
        "verdict": verdict,
    }
    write_json(evidence / "stability.json", stability)
    write_json(evidence / "summary.json", {
        "manifest_version": manifest["manifest_version"],
        "evidence_class": args.evidence_class,
        "real_model_evidence": args.evidence_class == "real",
        "selection": selection,
        "rounds": [report["aggregates"] for report in reports],
        "stability_verdict": verdict,
        "gaps": GAPS,
    })
    log.write(f"verdict={verdict}")
    log.close()
    print(json.dumps({
        "verdict": verdict,
        "rounds_run": len(round_checks),
        "aggregates": reports[-1]["aggregates"],
        "stability": stability,
        "evidence_dir": str(evidence),
    }, indent=2, ensure_ascii=False))
    return 0 if verdict in ("PASS", "BASELINE_ESTABLISHED") else 1


if __name__ == "__main__":
    sys.exit(main())
