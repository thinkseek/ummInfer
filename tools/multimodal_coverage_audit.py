#!/usr/bin/env python3
"""Independently audit disjoint real multimodal evidence as one 70-case set.

This tool does not import the benchmark runner's scoring or aggregation code. It
rebuilds responses from raw SSE, recomputes every metric, verifies the pinned
input bytes and media contracts, checks each source SHA256SUMS, and emits a
combined coverage record. Combining bounded rounds is coverage evidence only;
it is never a stability PASS.
"""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import sys
import unicodedata
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import multimodal_data as data  # noqa: E402

CATEGORIES = ("infrastructure", "public_contract", "quality", "performance")
VQA_ARTICLES = {"a", "an", "the"}
VQA_NUMBERS = dict(zip(
    ("zero", "one", "two", "three", "four", "five", "six", "seven",
     "eight", "nine", "ten"),
    tuple(str(value) for value in range(11))))
GSM8K_MARKER = re.compile(r"####\s*(-?[\d,]+)\s*$")


class AuditError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256sums(root):
    """Verify a relative checksum manifest without following its paths outside."""
    manifest = root / "SHA256SUMS"
    require(manifest.is_file(), f"missing {manifest}")
    entries = []
    seen = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64}) ([ *])(.+)", line)
        require(match is not None, f"invalid checksum line in {manifest}")
        expected, _, name = match.groups()
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts,
                f"unsafe checksum path {name!r}")
        require(name not in seen, f"duplicate checksum path {name!r}")
        seen.add(name)
        target = root.joinpath(*relative.parts)
        require(target.is_file(), f"checksum target missing: {target}")
        actual = sha256_file(target)
        require(actual == expected,
                f"checksum mismatch: {target}: {actual} != {expected}")
        entries.append(name)
    require(entries, f"empty checksum manifest: {manifest}")
    return entries


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def english_words(text):
    normalized = re.sub(r"[^A-Z0-9' ]+", " ", (text or "").upper())
    return " ".join(normalized.split()).split()


def english_characters(text):
    return list(" ".join((text or "").upper().split()))


def chinese_characters(text):
    normalized = unicodedata.normalize("NFKC", (text or "").lower())
    return [char for char in normalized
            if unicodedata.category(char)[0] not in "PSZC"]


def chinese_words(text):
    import jieba
    normalized = "".join(chinese_characters(text))
    if not normalized:
        return []
    return [token for token in jieba.cut(normalized, HMM=False) if token.strip()]


def ratio_metric(reference, hypothesis, tokenizer):
    ref = tokenizer(reference)
    hyp = tokenizer(hypothesis)
    if not ref:
        return None, None, 0
    errors = edit_distance(ref, hyp)
    return errors / len(ref), errors, len(ref)


def vqa_normalize(text):
    text = re.sub(r"[^\w\s]", "", (text or "").lower())
    return " ".join(VQA_NUMBERS.get(token, token) for token in text.split()
                    if token not in VQA_ARTICLES)


def keyword_hit(answer, keywords):
    if not keywords:
        return None
    tokens = set(re.findall(r"[a-z0-9]+", (answer or "").lower()))
    return sum(word in tokens for word in keywords) / len(keywords)


def recompute_metric(case, row):
    """Return the runner-compatible metric shape from independent arithmetic."""
    task = case["task"]
    reference = case["reference"]
    if task == "asr":
        hypothesis = row.get("transcript") or ""
        transcript = reference["transcript"]
        if case["metric"] == "cer":
            cer, char_errors, chars = ratio_metric(
                transcript, hypothesis, chinese_characters)
            wer, errors, words = ratio_metric(
                transcript, hypothesis, chinese_words)
            return {"name": "cer", "value": cer,
                    "detail": {"character_errors": char_errors,
                               "reference_characters": chars, "wer": wer,
                               "errors": errors, "reference_words": words}}
        wer, errors, words = ratio_metric(
            transcript, hypothesis, english_words)
        cer, char_errors, chars = ratio_metric(
            transcript, hypothesis, english_characters)
        return {"name": "wer", "value": wer,
                "detail": {"errors": errors, "reference_words": words,
                           "cer": cer, "character_errors": char_errors,
                           "reference_characters": chars}}
    if task == "gsm8k":
        match = GSM8K_MARKER.search((row.get("answer") or "").strip())
        predicted = int(match.group(1).replace(",", "")) if match else None
        gold = int(reference["answer"])
        return {"name": "gsm8k_exact", "value": float(predicted == gold),
                "detail": {"predicted": predicted, "gold": gold,
                           "marker_present": match is not None}}
    if task == "vqa":
        normalized = vqa_normalize(row.get("answer"))
        matches = sum(vqa_normalize(answer) == normalized
                      for answer in reference["answers"])
        return {"name": "vqa_accuracy", "value": min(1.0, matches / 3.0),
                "detail": {"matches": matches,
                           "answers": reference["answers"]}}
    if task == "joint":
        wer, errors, words = ratio_metric(
            reference["caption"], row.get("transcript") or "", english_words)
        return {"name": "joint_asr_wer_keyword_hit",
                "value": {"asr_wer": wer,
                          "keyword_hit": keyword_hit(
                              row.get("answer"), reference["expected_keywords"])},
                "detail": {"errors": errors, "reference_words": words,
                           "keywords": reference["expected_keywords"]}}
    raise AuditError(f"unknown task {task!r}")


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percent / 100.0 * len(ordered)))
    return ordered[rank - 1]


def recompute_aggregate(rows):
    """Recompute corpus metrics, macro means, timing, and missing-vs-zero."""
    groups = {}
    for row in rows:
        key = ("asr_" + (row.get("language") or "unknown")
               if row["task"] == "asr" else row["task"])
        groups.setdefault(key, []).append(row)

    def common(group):
        totals = [row["latency"].get("total_ms") for row in group
                  if row["latency"].get("total_ms") is not None]
        first = [row["latency"].get("first_delta_ms") for row in group
                 if row["latency"].get("first_delta_ms") is not None]
        terminals = [row["latency"].get("terminal_ms") for row in group
                     if row["latency"].get("terminal_ms") is not None]
        token_rates = [row.get("tokens_per_s") for row in group
                       if row.get("tokens_per_s") is not None]
        usable = [row for row in group if not any(
            failure["category"] in CATEGORIES[:2]
            for failure in row.get("failures", []))]
        return {
            "cases": len(group), "successes": len(usable),
            "success_rate": len(usable) / len(group) if group else None,
            "p50_ms": percentile(totals, 50), "p95_ms": percentile(totals, 95),
            "first_delta_p50_ms": percentile(first, 50),
            "first_delta_p95_ms": percentile(first, 95),
            "terminal_p50_ms": percentile(terminals, 50),
            "terminal_p95_ms": percentile(terminals, 95),
            "completed_terminals": sum(
                row.get("terminal_kind") == "complete" for row in group),
            "tokens_per_s_p50": percentile(token_rates, 50),
            "tokens_per_s_p95": percentile(token_rates, 95),
        }

    tasks = {}
    for task, group in sorted(groups.items()):
        aggregate = common(group)
        metrics = [row["metric"] for row in group if row.get("metric")]
        if task.startswith("asr"):
            errors = sum(metric["detail"].get("errors") or 0
                         for metric in metrics)
            words = sum(metric["detail"].get("reference_words") or 0
                        for metric in metrics)
            char_errors = sum(metric["detail"].get("character_errors") or 0
                              for metric in metrics)
            chars = sum(metric["detail"].get("reference_characters") or 0
                        for metric in metrics)
            aggregate["wer"] = errors / words if words else None
            aggregate["cer"] = char_errors / chars if chars else None
        elif task in ("vqa", "gsm8k"):
            values = [metric["value"] for metric in metrics
                      if type(metric.get("value")) is float]
            key = "vqa_accuracy" if task == "vqa" else "gsm8k_accuracy"
            aggregate[key] = statistics.fmean(values) if values else None
        elif task == "joint":
            present = [metric for metric in metrics
                       if isinstance(metric.get("value"), dict)]
            errors = sum(metric["detail"].get("errors") or 0
                         for metric in present)
            words = sum(metric["detail"].get("reference_words") or 0
                        for metric in present)
            hits = [metric["value"].get("keyword_hit") for metric in present
                    if type(metric["value"].get("keyword_hit")) in (int, float)]
            aggregate["joint_asr_wer"] = errors / words if words else None
            aggregate["joint_keyword_hit"] = (
                statistics.fmean(hits) if hits else None)
        tasks[task] = aggregate

    combined = common(rows)
    # The public overall aggregate intentionally omits token-rate summaries;
    # those remain per task because standalone ASR and VLM rates differ.
    combined.pop("tokens_per_s_p50")
    combined.pop("tokens_per_s_p95")
    combined["tasks"] = tasks
    combined["failure_counts"] = {
        category: sum(failure["category"] == category
                      for row in rows for failure in row.get("failures", []))
        for category in CATEGORIES}
    return combined


def assert_same(actual, expected, path):
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"{path}: expected object")
        require(set(actual) == set(expected),
                f"{path}: keys {set(actual)} != {set(expected)}")
        for key in expected:
            assert_same(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected),
                f"{path}: array mismatch")
        for index, value in enumerate(expected):
            assert_same(actual[index], value, f"{path}[{index}]")
    elif type(expected) in (int, float) and not isinstance(expected, bool):
        require(type(actual) in (int, float) and not isinstance(actual, bool),
                f"{path}: non-numeric {actual!r}")
        require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9),
                f"{path}: {actual!r} != {expected!r}")
    else:
        require(actual == expected, f"{path}: {actual!r} != {expected!r}")


def rebuild_sse(round_dir, row):
    case_id = row["case_id"]
    events = load_jsonl(round_dir / "events" / f"{case_id}.jsonl")
    answers = []
    transcript = ""
    usage = None
    finish_reason = None
    done = False
    terminals = []
    asr_positions = []
    first_delta_position = None
    for position, item in enumerate(events):
        require(item["index"] == position, f"{case_id}: event index gap")
        payload = item["data"]
        if payload == "[DONE]":
            require(not done, f"{case_id}: duplicate [DONE]")
            done = True
            continue
        event = json.loads(payload)
        unified = event.get("unified") or {}
        kind = unified.get("kind")
        if kind == "asr_final":
            transcript = unified.get("text") or ""
            asr_positions.append(position)
        if kind in ("complete", "error", "cancelled"):
            terminals.append(unified)
        for choice in event.get("choices", []):
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                if first_delta_position is None:
                    first_delta_position = position
                answers.append(content)
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
        if event.get("usage"):
            usage = event["usage"]
    answer = "".join(answers)
    require(answer == row["answer"], f"{case_id}: answer differs from raw SSE")
    require(transcript == row["transcript"],
            f"{case_id}: transcript differs from raw SSE")
    require(done == row["done"], f"{case_id}: [DONE] mismatch")
    require(finish_reason == row["finish_reason"],
            f"{case_id}: finish_reason mismatch")
    require(usage == row["usage"], f"{case_id}: usage mismatch")
    require(len(events) == row["events"], f"{case_id}: event count mismatch")
    require(len(terminals) == row["terminal_count"],
            f"{case_id}: terminal count mismatch")
    require(len(terminals) == 1 and terminals[0].get("kind") == "complete",
            f"{case_id}: no unique completed terminal")
    require(terminals[0].get("stage") == row["terminal_stage"] == "vlm",
            f"{case_id}: terminal stage mismatch")
    legs = terminals[0].get("legs")
    require(legs == row["terminal_legs"], f"{case_id}: terminal legs mismatch")
    require(legs is None or (isinstance(legs, list) and all(
        isinstance(leg, dict) and isinstance(leg.get("stage"), str)
        and leg.get("state") == "completed" for leg in legs)),
        f"{case_id}: invalid terminal legs")
    if "audio" in row["modalities"]:
        require(len(asr_positions) == 1, f"{case_id}: expected one asr_final")
        require(first_delta_position is None or
                asr_positions[0] < first_delta_position,
                f"{case_id}: ASR did not precede VLM")
        require(row["latency"].get("asr_final_ms") is not None,
                f"{case_id}: missing ASR-final timing")
    for key in ("first_delta_ms", "terminal_ms", "total_ms"):
        require(row["latency"].get(key) is not None,
                f"{case_id}: missing {key}")
    require(0 <= row["latency"]["first_delta_ms"] <=
            row["latency"]["terminal_ms"] <= row["latency"]["total_ms"] + 1e-6,
            f"{case_id}: non-monotonic timing")
    saved_answer = round_dir / "answers" / f"{case_id}.txt"
    require(saved_answer.is_file() and
            saved_answer.read_text(encoding="utf-8") == answer,
            f"{case_id}: saved answer mismatch")
    if transcript:
        saved = round_dir / "transcripts" / f"{case_id}.txt"
        require(saved.is_file() and saved.read_text(encoding="utf-8") == transcript,
                f"{case_id}: saved transcript mismatch")
    return {"raw_event_count": len(events),
            "completed_terminal": True,
            "terminal_legs_missing_not_empty": legs is None,
            "answer_sha256": sha256_bytes(answer.encode()),
            "transcript_sha256": sha256_bytes(transcript.encode())}


def verify_input(case, row, resolved):
    item = resolved[case["id"]]
    require(item.get("source") != "mock", f"{case['id']}: mock input")
    require(row["input_sha256"] == item["input_sha256"],
            f"{case['id']}: resolved input hash mismatch")
    kind = item["type"]
    media = {}
    if kind == "audio":
        media["audio"] = verify_wav(item["audio"])
    elif kind == "text":
        require(item["prompt"], f"{case['id']}: empty prompt")
        media["text_characters"] = len(item["prompt"])
    elif kind == "image":
        media["image"] = verify_image(item["image"])
    elif kind == "audio_image":
        media["audio"] = verify_wav(item["audio"])
        media["image"] = verify_image(item["image"])
        sent = item["input_sha256"].get("sent_audio")
        require(sent == sha256_bytes(item["audio"]),
                f"{case['id']}: sent-audio hash mismatch")
    else:
        raise AuditError(f"{case['id']}: unknown resolved type {kind}")
    return {"type": kind, "input_sha256": item["input_sha256"], **media}


def verify_wav(payload):
    with wave.open(io.BytesIO(payload), "rb") as wav:
        require(wav.getnchannels() == 1 and wav.getsampwidth() == 2 and
                wav.getframerate() == 16000 and wav.getnframes() > 0,
                "audio is not non-empty PCM16 mono 16 kHz")
        return {"bytes": len(payload), "sha256": sha256_bytes(payload),
                "frames": wav.getnframes(), "sample_rate": wav.getframerate(),
                "channels": 1, "sample_width_bytes": 2}


def verify_image(payload):
    if payload.startswith(b"\xff\xd8\xff"):
        image_format = "jpeg"
    elif payload.startswith(b"\x89PNG\r\n\x1a\n"):
        image_format = "png"
    else:
        raise AuditError("input image is neither JPEG nor PNG")
    require(len(payload) > 32, "truncated input image")
    return {"bytes": len(payload), "sha256": sha256_bytes(payload),
            "format": image_format}


def source_name(argument):
    label, separator, directory = argument.partition("=")
    require(separator and re.fullmatch(r"[a-z0-9_-]+", label),
            "--source must be LABEL=EVIDENCE_DIR")
    root = Path(directory).resolve()
    require(root.is_dir(), f"evidence directory not found: {root}")
    return label, root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(
        ROOT / "benchmarks" / "multimodal_manifest.json"))
    parser.add_argument("--source", action="append", required=True,
                        metavar="LABEL=DIR")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    import jieba
    require(jieba.__version__ == "0.42.1",
            f"Mandarin WER requires pinned jieba 0.42.1, got {jieba.__version__}")

    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    manifest_cases = {case["id"]: case for case in manifest["cases"]}
    require(manifest["total_cases"] == len(manifest_cases) == 70,
            "the declared manifest must contain 70 unique cases")
    labels = set()
    rows = []
    sources = []
    case_audit = []
    for argument in args.source:
        label, root = source_name(argument)
        require(label not in labels, f"duplicate source label {label}")
        labels.add(label)
        checksum_entries = verify_sha256sums(root)
        summary = load_json(root / "summary.json")
        report = load_json(root / "round-1" / "report.json")
        require(summary.get("evidence_class") == "real" and
                summary.get("real_model_evidence") is True and
                report.get("evidence_class") == "real" and
                report.get("real_model_evidence") is True,
                f"{label}: not real-model evidence")
        require(summary.get("stability_verdict") == "BASELINE_REQUIRED",
                f"{label}: expected BASELINE_REQUIRED")
        require(summary.get("manifest_version") == manifest["manifest_version"] ==
                report.get("manifest_version"), f"{label}: manifest mismatch")
        source_rows = load_jsonl(root / "round-1" / "cases.jsonl")
        require(report.get("cases") == source_rows,
                f"{label}: report and cases.jsonl differ")
        require(summary["selection"]["selected_cases"] == len(source_rows),
                f"{label}: selection count mismatch")
        ids = [row["case_id"] for row in source_rows]
        require(len(ids) == len(set(ids)), f"{label}: duplicate case")
        require(set(summary["selection"]["excluded_case_ids"]) ==
                set(manifest_cases) - set(ids),
                f"{label}: excluded case set mismatch")
        for row in source_rows:
            require(row["case_id"] in manifest_cases,
                    f"{label}: unknown case {row['case_id']}")
            case = manifest_cases[row["case_id"]]
            require(row["task"] == case["task"] and
                    row["modalities"] == case["modalities"] and
                    row.get("language") == case.get("language"),
                    f"{row['case_id']}: case identity changed")
            metric = recompute_metric(case, row)
            assert_same(row["metric"], metric, f"{row['case_id']}.metric")
            raw = rebuild_sse(root / "round-1", row)
            case_audit.append({"case_id": row["case_id"], "source": label,
                               "task": row["task"], "language": row["language"],
                               "metric": metric, "latency": row["latency"],
                               "failure_codes": [f["code"] for f in row["failures"]],
                               "case_row_sha256": sha256_bytes(
                                   json.dumps(row, ensure_ascii=False,
                                              sort_keys=True).encode()),
                               **raw})
        independent = recompute_aggregate(source_rows)
        assert_same(report["aggregates"], independent, f"{label}.aggregates")
        assert_same(summary["rounds"][0], independent,
                    f"{label}.summary.rounds[0]")
        sources.append({"label": label, "cases": len(source_rows),
                        "case_ids": ids, "sha256sums_entries": len(checksum_entries),
                        "summary_sha256": sha256_file(root / "summary.json"),
                        "report_sha256": sha256_file(root / "round-1/report.json"),
                        "sha256sums_sha256": sha256_file(root / "SHA256SUMS"),
                        "stability_verdict": "BASELINE_REQUIRED",
                        "aggregates": independent})
        rows.extend(source_rows)

    ids = [row["case_id"] for row in rows]
    require(len(rows) == len(set(ids)), "sources overlap or repeat a case")
    require(set(ids) == set(manifest_cases),
            "real evidence does not cover exactly the 70 manifest cases: " +
            json.dumps({"missing": sorted(set(manifest_cases) - set(ids)),
                        "unknown": sorted(set(ids) - set(manifest_cases))}))

    # Re-resolve every pinned input. This validates all archives, source members,
    # reconstructed WAVs, text prompts, and image bytes against the manifest.
    resolved = data.resolve_cases(manifest, list(manifest["cases"]),
                                  Path(args.cache_dir).resolve())
    input_audit = {}
    for row in rows:
        case = manifest_cases[row["case_id"]]
        input_audit[row["case_id"]] = verify_input(case, row, resolved)

    combined = recompute_aggregate(rows)
    output = {
        "schema_version": "umm-multimodal-combined-coverage-audit/v1",
        "status": "PASS",
        "real_model_evidence": True,
        "manifest_version": manifest["manifest_version"],
        "manifest_sha256": sha256_file(manifest_path),
        "declared_cases": 70,
        "real_cases": len(rows),
        "unique_real_cases": len(set(ids)),
        "coverage_complete": True,
        "completed_terminals": combined["completed_terminals"],
        "http_200": sum(row["http_status"] == 200 for row in rows),
        "missing_metric_case_ids": [row["case_id"] for row in rows
                                    if row["metric"].get("value") is None],
        "stability_verdict": "BASELINE_REQUIRED",
        "stability_pass_claim": False,
        "stability_limit": (
            "Three disjoint one-round real subsets establish 70-case coverage, "
            "not three consecutive full rounds or a regression baseline."),
        "sources": sources,
        "aggregates": combined,
        "case_results": sorted(case_audit, key=lambda item: item["case_id"]),
        "input_verification": {
            "cases": len(input_audit),
            "audio_cases": sum("audio" in item for item in input_audit.values()),
            "image_cases": sum("image" in item for item in input_audit.values()),
            "text_cases": sum(item["type"] == "text"
                              for item in input_audit.values()),
            "all_manifest_hashes_match": True,
            "all_media_contracts_pass": True,
            "zh_segmenter": "jieba 0.42.1, jieba.cut(HMM=False), default dictionary",
            "cases_detail": input_audit,
        },
        "human_review_required": [row["case_id"] for row in rows
                                  if row["human_review_required"]],
        "quality_limit": (
            "Objective submetrics do not grade free-form joint answers or the "
            "auxiliary one-token VLM output on standalone ASR requests."),
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"status": output["status"], "real_cases": len(rows),
                      "completed_terminals": combined["completed_terminals"],
                      "stability_verdict": output["stability_verdict"],
                      "tasks": combined["tasks"],
                      "failure_counts": combined["failure_counts"]},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AuditError, data.DataError) as error:
        print(f"AUDIT_ERROR: {error}", file=sys.stderr)
        sys.exit(1)
