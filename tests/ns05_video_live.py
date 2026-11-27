#!/usr/bin/env python3
"""Bounded, preregistered NS-05 real natural-video Before/After gate.

The caller supplies an external reviewed CC-BY source and all binary/model paths.
Only physical GPU 1 is used. This measures public text and streaming PCM first
response; physical playback, barge-in, and cross-modal generation are NOT_RUN.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
import video_decode_prefill_live as common  # noqa: E402
import video_temporal_live as short  # noqa: E402

SEED = 510075
CASE_IDS = ("phone-action", "driver-scene-change")
METRICS = ("decode_prefill", "first_text", "first_voice", "public_total")
PARITY = short.NATIVE_PARITY_FIELDS
EXPECTED_SOURCE = "ffef335edc161a2f6dead512e0fde96cf48fb997ae6b6491067e36c76eec58d0"
EXPECTED_LICENSE = "e7edaaaeb0cd6da6a3dd97c54f3e8443ef027be685253fab64bf6eed2419ee7e"


def schedule():
    rng = random.Random(SEED)
    warm = [{"phase": "warmup", "case_id": case, "repetition": 0} for case in CASE_IDS]
    rng.shuffle(warm)
    measured = [{"phase": "measured", "case_id": case, "repetition": repetition}
                for repetition in range(8) for case in CASE_IDS]
    rng.shuffle(measured)
    result = warm + measured
    for index, row in enumerate(result):
        row["schedule_index"] = index
    return result


def body_for(source, case):
    data_url = "data:video/mp4;base64," + base64.b64encode(source).decode("ascii")
    return {"model": "qwen3-native", "stream": True, "temperature": 0.0,
            "top_p": 1.0, "seed": SEED, "max_tokens": 64,
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": data_url}},
                {"type": "text", "text": case["question"]}]}],
            "modalities": ["text", "audio"],
            "audio": {"voice": "vivian", "format": "wav", "stream": True}}


def hit(answer, case):
    folded = answer.casefold()
    return all(term.casefold() in folded for term in case["expect_all"]) and (
        not case["expect_any"] or any(term.casefold() in folded for term in case["expect_any"]))


def stream_request(url, body, timeout):
    payload = json.dumps(body, separators=(",", ":")).encode()
    request = urllib.request.Request(url + "/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    started = time.monotonic()
    first_text = None
    first_voice = None
    answer_parts = []
    request_id = None
    terminal = None
    done = False
    frames = []
    raw_payloads = []
    chunks = {}
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = response.status
        for raw in response:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line.startswith("data:"):
                continue
            data = line[5:].lstrip()
            if data == "[DONE]":
                done = True
                raw_payloads.append(data)
                break
            frame = json.loads(data)
            raw_payloads.append(data)
            frames.append(frame)
            request_id = request_id or frame.get("id")
            for choice in frame.get("choices", []):
                text = choice.get("delta", {}).get("content")
                if isinstance(text, str) and text:
                    if first_text is None:
                        first_text = time.monotonic() - started
                    answer_parts.append(text)
                if choice.get("finish_reason") is not None:
                    terminal = choice["finish_reason"]
            unified = frame.get("unified", {})
            if unified.get("kind") == "artifact":
                artifact = unified.get("artifact", {})
                if artifact.get("mime_type") == "audio/pcm":
                    if first_voice is None:
                        first_voice = time.monotonic() - started
                    artifact_id = artifact.get("artifact_id")
                    index, total = artifact.get("index"), artifact.get("total")
                    if not isinstance(artifact_id, str) or not isinstance(index, int) \
                            or not isinstance(total, int) or total <= 0 or index != len(chunks.get(artifact_id, [])):
                        raise ValueError("invalid/out-of-order public PCM chunk")
                    chunks.setdefault(artifact_id, []).append(base64.b64decode(
                        artifact["data"], validate=True))
                    if index + 1 == total:
                        raw = b"".join(chunks[artifact_id])
                        if artifact.get("bytes") != len(raw) or artifact.get("sha256") != \
                                hashlib.sha256(raw).hexdigest():
                            raise ValueError("public PCM byte/hash mismatch")
    total = time.monotonic() - started
    answer = "".join(answer_parts)
    if status != 200 or not done or terminal not in ("stop", "length") or not answer.strip() \
            or not request_id or first_text is None or first_voice is None or not chunks:
        raise RuntimeError(f"invalid NS05 stream status={status} done={done} terminal={terminal} "
                           f"chars={len(answer)} text={first_text} voice={first_voice}")
    pcm = b"".join(part for key in sorted(chunks) for part in chunks[key])
    return {"request_id": request_id, "http_status": status, "answer": answer,
            "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
            "finish_reason": terminal, "first_text_s": first_text,
            "first_voice_s": first_voice, "total_s": total,
            "pcm_bytes": len(pcm), "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
            "pcm_artifact_count": len(chunks), "request_bytes": len(payload)}, frames, raw_payloads


def wait_tts(log, process, timeout=240):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("TTS exited before ready")
        if "native stage tts ready kind=tts" in log.read_text(encoding="utf-8", errors="replace"):
            return
        time.sleep(0.1)
    raise TimeoutError("TTS readiness timeout")


def stop_all(processes):
    records = []
    for name in ("gateway", "vlm", "asr", "tts"):
        process = processes.get(name)
        if process is None:
            continue
        record = {"name": name, "pid": process.pid, "sigterm_sent": False,
                  "sigkill_sent": False}
        if process.poll() is None:
            process.terminate()
            record["sigterm_sent"] = True
        try:
            record["returncode"] = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            record["sigkill_sent"] = True
            record["returncode"] = process.wait(timeout=10)
        records.append(record)
    return records


def run_stack(args):
    if os.environ.get("UMMINFER_LIVE_OPT_IN") != "1":
        raise ValueError("NS05 requires UMMINFER_LIVE_OPT_IN=1")
    if args.implementation not in ("baseline", "candidate") or args.block not in "ABCD":
        raise ValueError("invalid implementation/block")
    out = Path(args.out_dir).resolve()
    if out.exists():
        raise ValueError("refusing to overwrite NS05 evidence")
    out.mkdir(parents=True)
    prereg_path = Path(args.preregistration).resolve()
    prereg = json.loads(prereg_path.read_bytes())
    source_path, license_path = Path(args.source).resolve(), Path(args.license).resolve()
    if common.sha256_file(source_path) != EXPECTED_SOURCE or common.sha256_file(license_path) != EXPECTED_LICENSE:
        raise ValueError("reviewed NS05 source/license identity mismatch")
    identity = common.source_identity(args.source_root)
    pre = common.query_gpu()
    if pre["compute_apps"] or pre["memory_used_mib"] > 16:
        raise ValueError(f"GPU 1 not idle: {pre}")
    gateway = Path(args.gateway).resolve()
    binaries = {"vlm": Path(args.native_stage).resolve(), "asr": Path(args.asr_stage).resolve(),
                "tts": Path(args.tts_stage).resolve()}
    models = {"vlm": Path(args.model).resolve(), "asr": Path(args.asr_model).resolve(),
              "tts": Path(args.tts_model).resolve()}
    ipc = out / "ipc"
    ipc.mkdir()
    url = f"http://127.0.0.1:{args.port}"
    if args.port == 43301:
        raise ValueError("resident port forbidden")
    processes = {}
    logs = {}
    monitor = None
    error = None
    records = []
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "1"
        env["UMM_QWEN38_PREFIX_CACHE"] = "0"
        for name, kind in (("vlm", "qwen38"), ("asr", "asr"), ("tts", "tts")):
            log_path = out / f"{name}.log"
            logs[name] = log_path
            endpoint = {"vlm": "vlm", "asr": "entry", "tts": "tts"}[name]
            command = [str(binaries[name]), "--kind", kind, "--name",
                       "qwen" if name == "vlm" else name,
                       "--model", str(models[name]), "--gpu", "0",
                       "--recv-endpoint", f"ipc://{ipc}/{endpoint}",
                       "--coordinator-endpoint", f"ipc://{ipc}/completion",
                       "--abort-endpoint", f"ipc://{ipc}/abort"]
            if name == "vlm":
                command += ["--batch-size", "1"]
            with log_path.open("wb") as log:
                processes[name] = subprocess.Popen(command, cwd=args.source_root, env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        gateway_log = out / "gateway.log"
        logs["gateway"] = gateway_log
        command = [str(gateway), "serve", "--host", "127.0.0.1", "--port", str(args.port),
            "--model", "qwen3-native", "--entry", "asr", "--entry-endpoint", f"ipc://{ipc}/entry",
            "--vlm-entry", "qwen", "--vlm-entry-endpoint", f"ipc://{ipc}/vlm",
            "--tts-entry", "tts", "--tts-entry-endpoint", f"ipc://{ipc}/tts",
            "--completion", f"ipc://{ipc}/completion", "--abort", f"ipc://{ipc}/abort",
            "--max-in-flight", "4"]
        with gateway_log.open("wb") as log:
            processes["gateway"] = subprocess.Popen(command, cwd=args.source_root,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        common.wait_ready(url, processes, timeout=300)
        wait_tts(logs["tts"], processes["tts"])
        ready = common.query_gpu()
        if len(ready["compute_apps"]) != 3 or ready["memory_used_mib"] > 80000:
            raise ValueError(f"expected exactly three owned GPU1 stage processes: {ready}")
        source_bytes = source_path.read_bytes()
        monitor = common.ResourceMonitor()
        monitor.thread.start()
        for step in schedule():
            case = next(case for case in prereg["cases"] if case["id"] == step["case_id"])
            body = body_for(source_bytes, case)
            if "frame_selection" in json.dumps(body):
                raise ValueError("manual frame selection leaked into default NS05 request")
            response, frames, raw_payloads = stream_request(url, body, 60)
            row = {**step, "status": "COMPLETED", "response": response,
                   "canonical_request_sha256": hashlib.sha256(json.dumps(body,
                       sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                   "expected_answer_hit": hit(response["answer"], case)}
            records.append(row)
            relative = Path("records") / f"schedule-{step['schedule_index']:03d}.json"
            common.write_json(out / relative, row)
            common.write_json(out / relative.with_suffix(".frames.json"), frames)
            common.write_json(out / relative.with_suffix(".sse.json"), raw_payloads)
            health = common_health(url)
            if health.get("in_flight") != 0:
                raise ValueError("in_flight nonzero after NS05 request")
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
    finally:
        resources = monitor.stop() if monitor is not None else None
        process_records = stop_all(processes)
        deadline = time.monotonic() + 15
        post = common.query_gpu()
        while (post["compute_apps"] or post["memory_used_mib"] > 16) and time.monotonic() < deadline:
            time.sleep(0.1)
            post = common.query_gpu()
        cleanup = {"owned_processes": process_records, "post_stop": post,
                   "passed": not post["compute_apps"] and post["memory_used_mib"] <= 16}
        common.write_json(out / "cleanup.json", cleanup)
    if error:
        common.write_json(out / "error.json", {"error": error, "completed": len(records)})
        raise RuntimeError(error)
    native_records = short.parse_native_log(logs["vlm"])
    execution_records = common.parse_execution(logs["vlm"])
    for row in records:
        request_id = row["response"]["request_id"]
        native = short.native_for_request(native_records, request_id)
        execution = short.execution_for_request(execution_records, request_id)
        if native["max_frames"] != 8 or native["selected_frame_count"] != 8 \
                or native["source_indices_exact"] or not native["has_nominal_timestamps"] \
                or execution["prefix_reused"] != 0:
            raise ValueError("default-eight/cold-prefix NS05 native contract failed")
        row["native_evidence"] = native
        row["native_execution"] = execution
        row["latency_ms"] = {"decode_prefill": execution["video_input_ms"] +
                execution["vision_ms"] + execution["prefill_ms"],
            "first_text": row["response"]["first_text_s"] * 1000,
            "first_voice": row["response"]["first_voice_s"] * 1000,
            "public_total": row["response"]["total_s"] * 1000}
        common.write_json(out / "records" / f"schedule-{row['schedule_index']:03d}.json", row)
    report = {"schema_version": 1, "implementation": args.implementation, "block": args.block,
        "run_id": args.run_id, "preregistration_sha256": common.sha256_file(prereg_path),
        "source_sha256": EXPECTED_SOURCE, **identity,
        "binary": {"gateway_sha256": common.sha256_file(gateway),
                   **{name + "_sha256": common.sha256_file(path) for name, path in binaries.items()}},
        "model_config": {"vlm": common.sha256_file(models["vlm"] / "config.json"),
                         "asr": common.sha256_file(models["asr"] / "config.json"),
                         "tts": common.sha256_file(models["tts"] / "config.json")},
        "physical_gpu": 1, "pre_stack": pre, "post_ready": ready,
        "resources": resources, "records": records, "post_stop_cleanup": cleanup,
        "physical_playback": "NOT_RUN", "barge_in": "NOT_RUN", "cross_modal_generation": "NOT_RUN"}
    common.write_json(out / "mode-result.json", report)
    print(json.dumps({"run_id": args.run_id, "records": len(records),
                      "cleanup": cleanup["passed"]}, indent=2))
    return 0 if cleanup["passed"] else 2


def common_health(url):
    with urllib.request.urlopen(url + "/health", timeout=5) as response:
        return json.load(response)


def outlier_rows(rows):
    result = []
    for implementation in ("baseline", "candidate"):
        for case in CASE_IDS:
            selected = [row for row in rows if row["phase"] == "measured"
                        and row["implementation"] == implementation and row["case_id"] == case]
            for metric in METRICS:
                values = [row["latency_ms"][metric] for row in selected]
                q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
                low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
                for row in selected:
                    value = row["latency_ms"][metric]
                    if value < low or value > high:
                        result.append({"run_id": row["run_id"], "case_id": case,
                            "schedule_index": row["schedule_index"], "implementation": implementation,
                            "metric": metric, "value_ms": value, "lower_fence_ms": low,
                            "upper_fence_ms": high, "retained": True})
    return result


def analyze(args):
    runs = [json.loads((Path(path).resolve() / "mode-result.json").read_bytes()) for path in args.run]
    expected = [("baseline", "A"), ("candidate", "B"), ("candidate", "C"), ("baseline", "D")]
    if [(run["implementation"], run["block"]) for run in runs] != expected:
        raise ValueError("NS05 ABBA run order mismatch")
    if any(len(run["records"]) != 18 or not run["post_stop_cleanup"]["passed"] for run in runs):
        raise ValueError("incomplete NS05 run")
    if len({run["preregistration_sha256"] for run in runs}) != 1 \
            or len({json.dumps(run["binary"], sort_keys=True) for run in runs}) != 2:
        raise ValueError("protocol/binary identity drift")
    rows = [{key: run[key] for key in ("implementation", "block", "run_id")} | record
            for run in runs for record in run["records"]]
    pairs = []
    for left, right in (("A", "B"), ("D", "C")):
        baseline = next(run for run in runs if run["block"] == left)
        candidate = next(run for run in runs if run["block"] == right)
        for brow in baseline["records"]:
            crow = next(row for row in candidate["records"] if row["schedule_index"] == brow["schedule_index"])
            native_equal = all(brow["native_evidence"][field] == crow["native_evidence"][field]
                               for field in PARITY)
            answer_equal = brow["response"]["answer"] == crow["response"]["answer"]
            pcm_equal = brow["response"]["pcm_sha256"] == crow["response"]["pcm_sha256"] \
                and brow["response"]["pcm_bytes"] == crow["response"]["pcm_bytes"]
            pairs.append({"baseline_block": left, "candidate_block": right,
                "schedule_index": brow["schedule_index"], "case_id": brow["case_id"],
                "phase": brow["phase"], "repetition": brow["repetition"],
                "answer_exact": answer_equal, "native_exact": native_equal, "pcm_exact": pcm_equal,
                **{metric + "_delta_ms": crow["latency_ms"][metric] - brow["latency_ms"][metric]
                   for metric in METRICS}})
    if len(pairs) != 36 or any(not row["answer_exact"] or not row["native_exact"] or not row["pcm_exact"]
                               for row in pairs):
        raise ValueError("NS05 exact answer/native/PCM parity failed")
    distributions = {}
    paired_stats = {}
    quality = {}
    for implementation in ("baseline", "candidate"):
        for case in CASE_IDS:
            selected = [row for row in rows if row["phase"] == "measured"
                        and row["implementation"] == implementation and row["case_id"] == case]
            if len(selected) != 16:
                raise ValueError("NS05 measured population mismatch")
            key = f"{implementation}/{case}"
            distributions[key] = {metric: common.summarize([row["latency_ms"][metric]
                for row in selected]) for metric in METRICS}
            hits = sum(row["expected_answer_hit"] for row in selected)
            quality[key] = {"hits": hits, "total": 16, "misses": 16 - hits,
                "missed_answers": [row["response"]["answer"] for row in selected
                                   if not row["expected_answer_hit"]]}
    for case in CASE_IDS:
        for metric in METRICS:
            paired_stats[f"{case}/{metric}"] = common.paired_stats([
                row[metric + "_delta_ms"] for row in pairs if row["case_id"] == case
                and row["phase"] == "measured"])
    threshold_failures = []
    for key, metrics in distributions.items():
        for metric in ("first_text", "first_voice"):
            if metrics[metric]["p95_ms"] >= 3000:
                threshold_failures.append({"cell": key, "metric": metric,
                                           "p95_ms": metrics[metric]["p95_ms"]})
    primary = paired_stats["phone-action/first_voice"]
    other_primary = paired_stats["driver-scene-change/first_voice"]
    primary_pass = all(item["median_delta_ms"] < 0 and item["bootstrap_95_median_delta_ms"][1] < 0
        and item["exact_two_sided_sign_test_p"] <= 0.05 for item in (primary, other_primary))
    regressions = []
    for case in CASE_IDS:
        for metric in METRICS:
            baseline = distributions[f"baseline/{case}"][metric]
            candidate = distributions[f"candidate/{case}"][metric]
            for stat in ("median_ms", "p95_ms"):
                change = candidate[stat] / baseline[stat] - 1 if baseline[stat] else 0
                if change > 0.01:
                    regressions.append({"case_id": case, "metric": metric,
                                        "statistic": stat, "relative_change": change})
    promotion = primary_pass and not threshold_failures and not regressions
    report = {"schema_version": 1, "experiment": "umminfer_ns05_video_decode_prefill_before_after_v1",
        "preregistration_sha256": runs[0]["preregistration_sha256"],
        "requests": len(rows), "pairs": pairs, "distributions": distributions,
        "paired_comparisons": paired_stats, "quality": quality,
        "outliers": outlier_rows(rows), "threshold_failures": threshold_failures,
        "regressions_over_1_percent": regressions, "candidate_promotion": promotion,
        "candidate_disposition": "PROMOTE" if promotion else "NOT_PROMOTED",
        "historical_public_total_gate": "FAIL_2.3266_PERCENT_UNCHANGED",
        "physical_playback": "NOT_RUN", "perceptual_quality": "NOT_RUN",
        "barge_in_and_cross_modal_generation": "NOT_RUN"}
    common.write_json(args.output, report)
    print(json.dumps({"candidate_promotion": promotion, "threshold_failures": threshold_failures,
                      "regressions": regressions, "quality": quality}, indent=2))
    return 0 if promotion else 2


def self_check():
    steps = schedule()
    assert len(steps) == 18 and steps == schedule()
    assert hit("He holds the phone to his ear to make a call.",
               {"expect_all": ["phone"], "expect_any": ["ear", "call", "talk"]})
    assert not hit("He looks at the phone.", {"expect_all": ["phone"], "expect_any": ["ear"]})
    print("SELF-CHECK PASS: NS05 schedule and frozen scoring; REAL_GENERATION NOT_RUN")
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-stack")
    for name in ("implementation", "block", "run-id", "gateway", "native-stage", "asr-stage",
                 "tts-stage", "model", "asr-model", "tts-model", "source", "license",
                 "source-root", "preregistration", "out-dir"):
        run.add_argument("--" + name, required=True)
    run.add_argument("--port", required=True, type=int)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--run", action="append", required=True)
    analysis.add_argument("--output", required=True)
    sub.add_parser("self-check")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "self-check":
            return self_check()
        if args.command == "run-stack":
            return run_stack(args)
        return analyze(args)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"NS05 LIVE FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
