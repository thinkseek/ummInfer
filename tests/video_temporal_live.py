#!/usr/bin/env python3
"""Opt-in, limited-fixture Qwen3.8 uniform-vs-temporal live video gate.

Three arms run for every clip: omitted frame_selection (default), explicit
uniform, and temporal. Model, prompt, video bytes, seed, sampling, stream and
token budget are otherwise identical. The native stage's request-keyed
qwen38_video_evidence log supplies selected frames, timestamps, temporal-2
packing and MRoPE diagnostics; this harness never fabricates model answers.

Offline:
  python3 -I tests/video_temporal_live.py --self-check
Live (isolated loopback service on physical GPU 0 only):
  UMMINFER_LIVE_OPT_IN=1 python3 -I tests/video_temporal_live.py \
    --url http://127.0.0.1:PORT --manifest fixtures.json \
    --provenance provenance.json --source-revision COMMIT \
    --native-log qwen38-stage.log --run-mode serial --out-dir evidence/serial

A small fixture set measures only those cases. It cannot establish generalized
video-quality improvement. Expected-answer misses are recorded, not hidden or
converted into a transport/gate failure. Clean latency requires the stage to run
with UMM_QWEN38_PREFIX_CACHE=0 and provenance qwen38_prefix_cache=disabled;
every measured arm must log prefix_reused=0.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request

MAX_VIDEO_BYTES = 32 << 20
POLICIES = ("default", "uniform", "temporal")
NATIVE_PREFIX = "qwen38_video_evidence "
NATIVE_PARITY_FIELDS = (
    "policy", "decoded_frame_count", "selected_frame_count",
    "selected_candidate_indices", "selected_source_frame_indices",
    "source_indices_exact", "selected_frame_timestamps",
    "selected_frame_content_fnv1a64",
    "has_nominal_timestamps", "temporal2_pair_first_timestamps",
    "max_frames", "max_candidate_frames", "motion_threshold",
    "scene_threshold", "grid_t", "grid_h", "grid_w",
    "premerge_patch_rows", "merged_tokens_per_slice", "video_token_count",
    "prompt_token_count", "media_hash", "mrope_next", "mrope_fnv1a64",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_video(path: Path) -> tuple[str, str, str]:
    size = path.stat().st_size
    if size <= 0 or size > MAX_VIDEO_BYTES:
        raise ValueError(f"video must contain 1..{MAX_VIDEO_BYTES} bytes: {path}")
    data = path.read_bytes()
    suffix = path.suffix.lower()
    mime = {".mp4": "video/mp4", ".webm": "video/webm",
            ".mov": "video/quicktime"}.get(suffix)
    if mime is None:
        raise ValueError("video extension must be .mp4, .webm, or .mov")
    url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    return url, sha256_bytes(data), mime


def build_body(url: str, question: str, policy: str, model: str,
               motion_threshold: float = 0.02, scene_threshold: float = 0.25,
               seed: int = 42, max_tokens: int = 128) -> dict:
    if policy not in POLICIES:
        raise ValueError("policy must be default, uniform, or temporal")
    if not question:
        raise ValueError("question must be nonempty")
    if not 1 <= max_tokens <= 4096:
        raise ValueError("max_tokens must be in [1,4096]")
    for threshold in (motion_threshold, scene_threshold):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("thresholds must be in [0,1]")
    video = {"url": url}
    if policy == "uniform":
        video["frame_selection"] = {"policy": "uniform"}
    elif policy == "temporal":
        video["frame_selection"] = {
            "policy": "temporal",
            "motion_threshold": motion_threshold,
            "scene_threshold": scene_threshold,
        }
    return {
        "model": model,
        "stream": True,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": seed,
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "video_url", "video_url": video},
            ],
        }],
    }


def without_selection(body: dict) -> dict:
    result = copy.deepcopy(body)
    result["messages"][0]["content"][1]["video_url"].pop("frame_selection", None)
    result.pop("canonical_request_sha256", None)
    return result


def request_summary(body: dict, video_sha256: str) -> dict:
    result = copy.deepcopy(body)
    result["messages"][0]["content"][1]["video_url"]["url"] = {
        "data_url_sha256": video_sha256
    }
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["canonical_request_sha256"] = sha256_bytes(encoded)
    return result


def stream_chat(base_url: str, body: dict, timeout_s: float) -> tuple[dict, list]:
    payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    started = time.monotonic()
    first_event_s = None
    ttft_s = None
    text_parts = []
    terminal = None
    request_id = None
    done = False
    frames = []
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        status = response.status
        for raw in response:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line.startswith("data:"):
                continue
            data = line[5:].lstrip()
            if data == "[DONE]":
                done = True
                break
            frame = json.loads(data)
            frames.append(frame)
            if first_event_s is None:
                first_event_s = time.monotonic() - started
            request_id = request_id or frame.get("id")
            for choice in frame.get("choices", []):
                delta = choice.get("delta", {}).get("content")
                if isinstance(delta, str) and delta:
                    if ttft_s is None:
                        ttft_s = time.monotonic() - started
                    text_parts.append(delta)
                if choice.get("finish_reason") is not None:
                    terminal = choice["finish_reason"]
            unified = frame.get("unified")
            if isinstance(unified, dict) and unified.get("kind") in ("error", "cancelled"):
                terminal = unified["kind"]
    total_s = time.monotonic() - started
    if status != 200 or not done:
        raise RuntimeError(f"SSE status/done failure: status={status} done={done}")
    answer = "".join(text_parts)
    if terminal != "stop" or not answer.strip() or not request_id:
        raise RuntimeError(
            f"invalid terminal/answer/id: terminal={terminal!r} "
            f"chars={len(answer)} request_id={request_id!r}")
    return {"request_id": request_id, "http_status": status, "answer": answer,
            "answer_sha256": sha256_bytes(answer.encode()),
            "first_event_s": first_event_s, "ttft_s": ttft_s,
            "total_s": total_s, "finish_reason": terminal,
            "request_bytes": len(payload)}, frames


def parse_native_log(path: Path) -> dict[str, dict]:
    found = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            start = line.find(NATIVE_PREFIX)
            if start < 0:
                continue
            record = json.loads(line[start + len(NATIVE_PREFIX):])
            request_id = record.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("native evidence has no request_id")
            if request_id in found:
                raise ValueError(f"duplicate native evidence for {request_id}")
            found[request_id] = record
    return found


def parse_native_execution_log(path: Path) -> dict[str, dict]:
    """Parse the existing native request summary used to prove cold prefill."""
    found = {}
    integer_fields = ("prompt_tokens", "prefill_tokens", "prefix_reused",
                      "image_tokens", "generated_tokens", "elapsed_ms")
    float_fields = ("vision_ms", "prefill_ms", "decode_ms", "decode_tok_per_s")
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("qwen38 request request_id="):
                continue
            fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
            request_id = fields.get("request_id")
            if not request_id:
                raise ValueError("native execution record has no request_id")
            if request_id in found:
                raise ValueError(f"duplicate native execution record for {request_id}")
            try:
                record = {"request_id": request_id}
                record.update({key: int(fields[key]) for key in integer_fields})
                record.update({key: float(fields[key]) for key in float_fields})
            except (KeyError, ValueError) as error:
                raise ValueError(f"malformed native execution record for {request_id}") from error
            found[request_id] = record
    return found


def native_for_request(records: dict[str, dict], request_id: str) -> dict:
    matches = [record for key, record in records.items()
               if key == request_id or key == request_id + ":vlm"]
    if len(matches) != 1:
        raise ValueError(
            f"expected one native video record for {request_id}:vlm, got {len(matches)}")
    return matches[0]


def execution_for_request(records: dict[str, dict], request_id: str) -> dict:
    matches = [record for key, record in records.items()
               if key == request_id or key == request_id + ":vlm"]
    if len(matches) != 1:
        raise ValueError(
            f"expected one native execution record for {request_id}, got {len(matches)}")
    return matches[0]


def validate_native(record: dict, policy: str) -> None:
    required = set(NATIVE_PARITY_FIELDS) | {"request_id", "engine_path"}
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"native evidence missing fields: {missing}")
    expected_policy = "temporal" if policy == "temporal" else "uniform"
    if record["policy"] != expected_policy:
        raise ValueError(f"native policy mismatch: {record['policy']} != {expected_policy}")
    selected = record["selected_frame_count"]
    decoded = record["decoded_frame_count"]
    if record["max_frames"] not in (4, 8, 16) or not 1 <= selected <= record["max_frames"]:
        raise ValueError("native selected frame budget was violated")
    if decoded < selected or decoded > record["max_candidate_frames"] or decoded > 64:
        raise ValueError("native candidate frame budget was violated")
    if len(record["selected_candidate_indices"]) != selected:
        raise ValueError("native selected candidate count mismatch")
    if len(record["selected_source_frame_indices"]) != selected:
        raise ValueError("native selected source index count mismatch")
    if len(record["selected_frame_timestamps"]) != selected:
        raise ValueError("native selected timestamp count mismatch")
    if len(record["selected_frame_content_fnv1a64"]) != selected or any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{16}", value)
            for value in record["selected_frame_content_fnv1a64"]):
        raise ValueError("native selected RGB fingerprint count/format mismatch")
    if record["grid_t"] != (selected + 1) // 2:
        raise ValueError("temporal-2 grid_t does not match selected frame count")
    if record["premerge_patch_rows"] != (
            record["grid_t"] * record["grid_h"] * record["grid_w"]):
        raise ValueError("native premerge packing mismatch")
    if record["video_token_count"] != (
            record["grid_t"] * record["merged_tokens_per_slice"]):
        raise ValueError("native merged video token count mismatch")
    pair_timestamps = record["temporal2_pair_first_timestamps"]
    expected_pairs = [record["selected_frame_timestamps"][min(2 * i, selected - 1)]
                      for i in range(record["grid_t"])]
    if pair_timestamps != expected_pairs:
        raise ValueError("temporal-2 pair-first timestamp mismatch")
    if policy == "temporal" and not record["source_indices_exact"]:
        raise ValueError("temporal gate requires exact source frame indices")


def native_parity(default: dict, uniform: dict) -> bool:
    return all(default.get(field) == uniform.get(field)
               for field in NATIVE_PARITY_FIELDS)


def first_frame_sha256(path: Path) -> str:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        check=True, capture_output=True, timeout=60)
    if not result.stdout:
        raise ValueError(f"FFmpeg decoded no first frame: {path}")
    return sha256_bytes(result.stdout)


def probe_video(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height,nb_frames,avg_frame_rate:format=duration,size",
         "-of", "json", str(path)], check=True, capture_output=True, timeout=60)
    return json.loads(result.stdout)


def load_cases(args: argparse.Namespace) -> tuple[list[dict], dict | None, str | None]:
    if args.manifest:
        manifest_path = Path(args.manifest).resolve()
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
        if not isinstance(manifest.get("cases"), list) or not manifest["cases"]:
            raise ValueError("manifest must contain a nonempty cases array")
        cases = []
        for item in manifest["cases"]:
            case = dict(item)
            video = Path(case["video"])
            case["video"] = str(video if video.is_absolute()
                                else manifest_path.parent / video)
            if not case.get("id") or not case.get("question"):
                raise ValueError("each manifest case requires id and question")
            cases.append(case)
        return cases, manifest, sha256_bytes(raw)
    if not args.video or not args.question:
        raise ValueError("live mode requires --manifest or --video plus --question")
    if len(args.question) not in (1, len(args.video)):
        raise ValueError("provide one --question per --video, or one shared question")
    if args.expect and len(args.expect) != len(args.video):
        raise ValueError("provide no --expect or exactly one per --video")
    cases = []
    for index, video in enumerate(args.video):
        cases.append({
            "id": Path(video).stem,
            "video": str(Path(video).resolve()),
            "question": args.question[0] if len(args.question) == 1 else args.question[index],
            "expect_any": [args.expect[index]] if args.expect else [],
        })
    return cases, None, None


def quality_hit(answer: str, expectations: list[str]) -> bool | None:
    if not expectations:
        return None
    folded = answer.casefold()
    return any(expected.casefold() in folded for expected in expectations)


def run_live(args: argparse.Namespace) -> int:
    if (not args.url or not args.out_dir or not args.native_log or not args.provenance
            or not args.source_revision):
        raise ValueError("live mode requires --url, --out-dir, --native-log, "
                         "--provenance and --source-revision")
    parsed_url = urllib.parse.urlsplit(args.url)
    if parsed_url.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("live gate requires an isolated loopback URL")
    if parsed_url.port == 43301:
        raise ValueError("resident service port is forbidden")
    if args.run_mode == "serial" and args.concurrency != 1:
        raise ValueError("serial mode requires --concurrency 1")
    if args.run_mode == "batched" and args.concurrency < 2:
        raise ValueError("batched mode requires --concurrency 2 or greater")
    if not 0 <= args.warmup <= 3:
        raise ValueError("--warmup must be in [0,3]")
    cases, manifest, manifest_sha256 = load_cases(args)
    provenance_path = Path(args.provenance).resolve()
    provenance_raw = provenance_path.read_bytes()
    provenance = json.loads(provenance_raw)
    if provenance.get("physical_gpu") != args.expected_gpu:
        raise ValueError("provenance does not pin the explicitly allowed physical GPU")
    cold_prefix = not args.allow_prefix_reuse
    if cold_prefix and provenance.get("qwen38_prefix_cache") != "disabled":
        raise ValueError("clean latency requires provenance qwen38_prefix_cache=disabled "
                         "(or explicit --allow-prefix-reuse)")
    provenance["source_revision_reported"] = args.source_revision
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "video_temporal_report.json"
    if report_path.exists():
        raise ValueError(f"refusing to overwrite existing report: {report_path}")

    encoded = {}
    first_hashes = {}
    for case in cases:
        path = Path(case["video"])
        case["video"] = path.name
        case["video_sha256"] = sha256_file(path)
        case["video_bytes"] = path.stat().st_size
        case["ffprobe"] = probe_video(path)
        case["first_frame_rgb_sha256"] = first_frame_sha256(path)
        expected_first = case.get("expected_first_frame_rgb_sha256")
        if expected_first and case["first_frame_rgb_sha256"] != expected_first:
            raise ValueError(f"first-frame digest mismatch for {case['id']}")
        if case.get("pair"):
            previous = first_hashes.setdefault(case["pair"], case["first_frame_rgb_sha256"])
            if previous != case["first_frame_rgb_sha256"]:
                raise ValueError(f"paired clips do not share a first frame: {case['pair']}")
        encoded[case["id"]] = encode_video(path)

    jobs = [(case, policy) for case in cases for policy in POLICIES]

    def execute(job):
        case, policy = job
        url, digest, mime = encoded[case["id"]]
        body = build_body(url, case["question"], policy, args.model,
                          args.motion_threshold, args.scene_threshold,
                          args.seed, args.max_tokens)
        result, frames = stream_chat(args.url, body, args.timeout_s)
        frame_path = out_dir / f"{case['id']}-{policy}-sse.json"
        frame_path.write_text(json.dumps(frames, ensure_ascii=False) + "\n", encoding="utf-8")
        result["request"] = request_summary(body, digest)
        result["mime"] = mime
        result["expected_substring_hit"] = quality_hit(
            result["answer"], case.get("expect_any", []))
        return case["id"], policy, result

    # Warm model kernels, dynamic libraries, FFmpeg and the OS file cache before
    # any measured arm. Without this, the historical default arm always pays a
    # first-request penalty that explicit uniform does not, biasing TTFT.
    warmups = []
    for case in cases:
        for iteration in range(args.warmup):
            _, _, warmup = execute((case, "default"))
            warmups.append({"case": case["id"], "iteration": iteration,
                            "request_id": warmup["request_id"],
                            "ttft_s": warmup["ttft_s"], "total_s": warmup["total_s"]})

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(execute, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            case_id, policy, result = future.result()
            results[(case_id, policy)] = result

    native_log_path = Path(args.native_log).resolve()
    native_records = parse_native_log(native_log_path)
    execution_records = parse_native_execution_log(native_log_path)
    report_cases = []
    default_uniform_ok = True
    native_ok = True
    for case in cases:
        case_result = {key: value for key, value in case.items()
                       if key not in ("video", "expect_any")}
        case_result["video"] = case["video"]
        case_result["expect_any"] = case.get("expect_any", [])
        case_result["policies"] = {}
        for policy in POLICIES:
            result = results[(case["id"], policy)]
            try:
                native = native_for_request(native_records, result["request_id"])
                validate_native(native, policy)
                execution = execution_for_request(execution_records, result["request_id"])
                if cold_prefix and execution["prefix_reused"] != 0:
                    raise ValueError(
                        f"prefix reuse contaminated measured arm: {execution['prefix_reused']} tokens")
                result["native"] = native
                result["native_execution"] = execution
                result["native_evidence_valid"] = True
            except (ValueError, TypeError) as error:
                result["native_evidence_valid"] = False
                result["native_evidence_error"] = str(error)
                native_ok = False
            case_result["policies"][policy] = result
        default_result = case_result["policies"]["default"]
        uniform_result = case_result["policies"]["uniform"]
        requests_match = (without_selection(uniform_result["request"]) ==
                          without_selection(default_result["request"]))
        answers_match = default_result["answer"] == uniform_result["answer"]
        native_match = (default_result.get("native_evidence_valid") and
                        uniform_result.get("native_evidence_valid") and
                        native_parity(default_result["native"], uniform_result["native"]))
        case_result["default_uniform_compatibility"] = {
            "request_after_removing_explicit_uniform_selection": requests_match,
            "answer_byte_identical": answers_match,
            "native_selection_packing_mrope_identical": native_match,
        }
        default_uniform_ok &= requests_match and answers_match and native_match
        case_result["answers_identical_default_temporal"] = (
            default_result["answer"] == case_result["policies"]["temporal"]["answer"])
        report_cases.append(case_result)

    quality = {}
    for policy in POLICIES:
        checks = [results[(case["id"], policy)]["expected_substring_hit"] for case in cases]
        covered = [check for check in checks if check is not None]
        quality[policy] = {"correct": sum(check is True for check in covered),
                           "covered": len(covered),
                           "hits": checks}
    report = {
        "schema_version": 2,
        "run_mode": args.run_mode,
        "concurrency": args.concurrency,
        "model": args.model,
        "source_revision": args.source_revision,
        "source_revision_object": args.source_revision_object,
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "provenance": provenance,
        "provenance_sha256": sha256_bytes(provenance_raw),
        "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": args.seed,
                     "max_tokens": args.max_tokens, "stream": True},
        "thresholds": {"motion_threshold": args.motion_threshold,
                       "scene_threshold": args.scene_threshold},
        "warmups_per_case": args.warmup,
        "warmups": warmups,
        "cases": report_cases,
        "correctness_coverage": quality,
        "default_uniform_compatibility": default_uniform_ok,
        "native_evidence_complete": native_ok,
        "prefix_cache_policy": "disabled" if cold_prefix else "allowed_by_request",
        "clean_policy_latency": cold_prefix and native_ok,
        "real_model_quality": "MEASURED_LIMITED_FIXTURE" if any(
            item["covered"] for item in quality.values()) else "REVIEW_REQUIRED",
        "generalized_quality_claim": False,
        "live_latency": "MEASURED",
    }

    comparison_ok = True
    if args.baseline_report:
        baseline_path = Path(args.baseline_report).resolve()
        baseline_raw = baseline_path.read_bytes()
        baseline = json.loads(baseline_raw)
        baseline_cases = {case["id"]: case for case in baseline["cases"]}
        comparisons = []
        for case in report_cases:
            other = baseline_cases.get(case["id"])
            for policy in POLICIES:
                current_arm = case["policies"][policy]
                other_arm = other["policies"][policy] if other else {}
                answer_same = other_arm.get("answer") == current_arm["answer"]
                native_same = bool(other_arm.get("native")) and native_parity(
                    other_arm["native"], current_arm.get("native", {}))
                comparisons.append({"case": case["id"], "policy": policy,
                                    "answer_byte_identical": answer_same,
                                    "native_selection_packing_mrope_identical": native_same})
                comparison_ok &= answer_same and native_same
        report["serial_batched_comparison"] = {
            "baseline_report_sha256": sha256_bytes(baseline_raw),
            "baseline_run_mode": baseline.get("run_mode"),
            "current_run_mode": args.run_mode,
            "all_answers_and_native_inputs_identical": comparison_ok,
            "arms": comparisons,
        }
    report["gate_pass"] = bool(default_uniform_ok and native_ok and comparison_ok)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"cases": len(report_cases), "run_mode": args.run_mode,
                      "default_uniform_compatibility": default_uniform_ok,
                      "native_evidence_complete": native_ok,
                      "clean_policy_latency": report["clean_policy_latency"],
                      "serial_batched_comparison": comparison_ok if args.baseline_report else "NOT_REQUESTED",
                      "correctness_coverage": quality,
                      "real_model_quality": report["real_model_quality"],
                      "generalized_quality_claim": False,
                      "gate_pass": report["gate_pass"], "report": str(report_path)},
                     indent=2, ensure_ascii=False))
    return 0 if report["gate_pass"] else 1


def self_check() -> int:
    url = "data:video/mp4;base64,AAAAIGZ0eXBpc29t"
    bodies = {policy: build_body(url, "What moves?", policy, "qwen3-native")
              for policy in POLICIES}
    assert "frame_selection" not in bodies["default"]["messages"][0]["content"][1]["video_url"]
    assert bodies["uniform"]["messages"][0]["content"][1]["video_url"]["frame_selection"] == {
        "policy": "uniform"}
    assert without_selection(bodies["uniform"]) == bodies["default"]
    assert without_selection(bodies["temporal"]) == bodies["default"]
    assert without_selection(request_summary(bodies["default"], "abc")) == without_selection(
        request_summary(bodies["uniform"], "abc"))
    for bad in ("other", "omitted"):
        try:
            build_body(url, "bad", bad, "qwen3-native")
        except ValueError:
            pass
        else:
            raise AssertionError("unknown policy was accepted")
    native = {
        "request_id": "req-1:vlm", "policy": "temporal", "engine_path": "serial",
        "decoded_frame_count": 4, "selected_frame_count": 3,
        "selected_candidate_indices": [0, 2, 3],
        "selected_source_frame_indices": [0, 4, 6], "source_indices_exact": True,
        "selected_frame_timestamps": [0.0, 0.5, 0.75],
        "selected_frame_content_fnv1a64": ["0123456789abcdef"] * 3,
        "has_nominal_timestamps": False,
        "temporal2_pair_first_timestamps": [0.0, 0.75],
        "max_frames": 8, "max_candidate_frames": 64,
        "motion_threshold": 0.02, "scene_threshold": 0.25,
        "grid_t": 2, "grid_h": 8, "grid_w": 8,
        "premerge_patch_rows": 128, "merged_tokens_per_slice": 16,
        "video_token_count": 32, "prompt_token_count": 80,
        "media_hash": "01", "mrope_next": 90, "mrope_fnv1a64": "02",
    }
    validate_native(native, "temporal")
    assert native_for_request({"req-1:vlm": native}, "req-1") == native
    with tempfile.TemporaryDirectory() as temp_dir:
        log_path = Path(temp_dir) / "stage.log"
        log_path.write_text(
            "qwen38 request request_id=req-1 prompt_tokens=80 prefill_tokens=80 "
            "prefix_reused=0 image_tokens=32 generated_tokens=1 finish_reason=stop "
            "chars=4 vision_ms=1.0 prefill_ms=2.0 decode_ms=3.0 "
            "decode_tok_per_s=4.0 elapsed_ms=6\n")
        execution = execution_for_request(parse_native_execution_log(log_path), "req-1")
        assert execution["prefix_reused"] == 0 and execution["prefill_tokens"] == 80
    assert quality_hit("The liquid rises.", ["rise"])
    assert quality_hit("No scene change.", ["fall"]) is False
    print("SELF-CHECK PASS: three-arm request, native temporal-2/MRoPE, and quality contracts")
    print("REAL_MODEL_QUALITY: NOT_RUN")
    print("LIVE_LATENCY: NOT_RUN")
    print("SERIAL_BATCHED_COMPARISON: NOT_RUN")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--url", default=os.environ.get("UMMINFER_URL", ""))
    parser.add_argument("--model", default="qwen3-native")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument("--question", action="append", default=[])
    parser.add_argument("--expect", action="append", default=[])
    parser.add_argument("--motion-threshold", type=float, default=0.02)
    parser.add_argument("--scene-threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--native-log", default="")
    parser.add_argument("--provenance", default="")
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--source-revision-object", default="")
    parser.add_argument("--expected-gpu", type=int, default=0)
    parser.add_argument("--run-mode", choices=("serial", "batched"), default="serial")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--allow-prefix-reuse", action="store_true",
                        help="permit cache-enabled measurements (not clean policy latency)")
    parser.add_argument("--baseline-report", default="")
    args = parser.parse_args(argv)
    if args.self_check:
        return self_check()
    if os.environ.get("UMMINFER_LIVE_OPT_IN") != "1":
        print("REAL_MODEL_QUALITY: NOT_RUN")
        print("LIVE_LATENCY: NOT_RUN")
        print("SERIAL_BATCHED_COMPARISON: NOT_RUN")
        print("Set UMMINFER_LIVE_OPT_IN=1 and provide the isolated live arguments.")
        return 0
    try:
        return run_live(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"LIVE FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
