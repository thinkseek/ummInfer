#!/usr/bin/env python3
"""Offline negative and arithmetic tests for the combined real-evidence audit."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import multimodal_coverage_audit as audit  # noqa: E402


def expect_error(call, label, failures):
    try:
        call()
    except audit.AuditError:
        return
    failures.append(f"{label}: auditor accepted invalid evidence")


def row(task="joint", metric=None, failures=None):
    return {
        "case_id": "synthetic", "task": task, "language": "en",
        "modalities": ["audio", "image"] if task == "joint" else ["text"],
        "metric": metric, "failures": failures or [],
        "latency": {"total_ms": 3.0, "first_delta_ms": 1.0,
                    "terminal_ms": 2.0, "asr_final_ms": None},
        "tokens_per_s": None,
    }


def main():
    failures = []
    en, errors, words = audit.ratio_metric(
        "Two words", "Two words", audit.english_words)
    if (en, errors, words) != (0.0, 0, 2):
        failures.append("English WER zero was not preserved")
    zh, errors, chars = audit.ratio_metric(
        "今天，天气 很好。", "今天天气很好", audit.chinese_characters)
    if (zh, errors, chars) != (0.0, 0, 6):
        failures.append("Chinese CER normalization changed")
    if audit.vqa_normalize("The Two, FOODIEBAKER.COM") != "2 foodiebakercom":
        failures.append("VQA normalization changed")
    if audit.keyword_hit("A room with blue walls", ["room", "zebra"]) != 0.5:
        failures.append("joint keyword arithmetic changed")

    missing = row(metric={"name": "joint_asr_wer_keyword_hit",
                          "value": None, "detail": {}})
    zero = row(metric={"name": "joint_asr_wer_keyword_hit",
                       "value": {"asr_wer": 0.0, "keyword_hit": 0.0},
                       "detail": {"errors": 0, "reference_words": 2,
                                  "keywords": ["room"]}})
    aggregate = audit.recompute_aggregate([missing, zero])
    joint = aggregate["tasks"]["joint"]
    if joint["joint_keyword_hit"] != 0.0 or joint["joint_asr_wer"] != 0.0:
        failures.append("numeric zero was lost beside a missing joint metric")
    if aggregate["cases"] != 2 or aggregate["completed_terminals"] != 0:
        failures.append("aggregate case or terminal counts changed")

    with tempfile.TemporaryDirectory(prefix="umm-coverage-audit-test-") as tmp:
        root = Path(tmp)
        payload = root / "payload.bin"
        payload.write_bytes(b"real-evidence-test")
        digest = hashlib.sha256(payload.read_bytes()).hexdigest()
        (root / "SHA256SUMS").write_text(f"{digest}  payload.bin\n")
        try:
            audit.verify_sha256sums(root)
        except audit.AuditError as error:
            failures.append(f"valid checksum rejected: {error}")
        payload.write_bytes(b"corrupt")
        expect_error(lambda: audit.verify_sha256sums(root),
                     "corrupt checksum", failures)
        (root / "SHA256SUMS").write_text(f"{digest}  ../outside\n")
        expect_error(lambda: audit.verify_sha256sums(root),
                     "unsafe checksum path", failures)

        round_dir = root / "round-1"
        (round_dir / "answers").mkdir(parents=True)
        (round_dir / "transcripts").mkdir()
        (round_dir / "events").mkdir()
        case_id = "synthetic"
        answer = "#### 45"
        events = [
            {"choices": [{"delta": {"content": answer},
                          "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}],
             "unified": {"kind": "complete", "stage": "vlm"},
             "usage": {"completion_tokens": 2}},
        ]
        payloads = [json.dumps(event) for event in events] + ["[DONE]"]
        (round_dir / "events" / f"{case_id}.jsonl").write_text(
            "".join(json.dumps({"index": index, "data": data}) + "\n"
                    for index, data in enumerate(payloads)))
        (round_dir / "answers" / f"{case_id}.txt").write_text(answer)
        valid = row("gsm8k", {"name": "gsm8k_exact", "value": 1.0,
                              "detail": {"predicted": 45, "gold": 45,
                                         "marker_present": True}})
        valid.update({"case_id": case_id, "answer": answer, "transcript": "",
                      "events": 3, "done": True, "finish_reason": "stop",
                      "usage": {"completion_tokens": 2}, "terminal_count": 1,
                      "terminal_kind": "complete", "terminal_stage": "vlm",
                      "terminal_legs": None})
        try:
            audit.rebuild_sse(round_dir, valid)
        except audit.AuditError as error:
            failures.append(f"valid SSE rejected: {error}")
        invalid = dict(valid)
        invalid["terminal_count"] = 0
        expect_error(lambda: audit.rebuild_sse(round_dir, invalid),
                     "missing terminal", failures)
        invalid = dict(valid)
        invalid["answer"] = "handwritten replacement"
        expect_error(lambda: audit.rebuild_sse(round_dir, invalid),
                     "rewritten answer", failures)

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print("PASS multimodal coverage audit: metrics, missing-vs-zero, raw SSE, "
          "terminal and checksum negative checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
