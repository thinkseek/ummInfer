#!/usr/bin/env python3
"""Preregistered real natural-video Qwen3.8 decode/prefill ABBA gate.

Each run-stack invocation owns exactly the gateway/native processes it starts.
Only physical GPU 1 and an isolated loopback/IPC stack are permitted. Raw media,
SSE, and request records stay in the caller-provided external evidence directory.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
import video_temporal_live as short  # noqa: E402

ARMS = ("default-8", "uniform-16", "temporal-16")
EXPECTED_SOURCE = "81178f8af6c859d908cd6fa4a5348ed9bbe271c071a94a196ff2488c63f0db2e"
QUESTION = "Describe the visible natural scenes and their order in one concise English sentence."
SEED = 510075
NATIVE_PARITY_FIELDS = short.NATIVE_PARITY_FIELDS
METRICS = ("total", "ttft", "native_elapsed", "video_input", "prompt_prep",
           "vision", "prefill", "decode", "unattributed")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def query_gpu():
    memory = subprocess.run([
        "nvidia-smi", "-i", "1", "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True,
        timeout=10).stdout.strip()
    parts = [part.strip() for part in memory.split(",")]
    if len(parts) != 3 or int(parts[0]) != 1:
        raise ValueError("GPU 1 memory query returned an invalid record")
    apps_text = subprocess.run([
        "nvidia-smi", "-i", "1", "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True,
        timeout=10).stdout
    apps = []
    for line in apps_text.splitlines():
        if line.strip():
            pid, used = [part.strip() for part in line.split(",")]
            apps.append({"pid": int(pid), "used_mib": int(used)})
    return {"physical_gpu": 1, "memory_used_mib": int(parts[1]),
            "utilization_percent": int(parts[2]), "compute_apps": apps}


def source_identity(root):
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, text=True, timeout=30).stdout.strip()
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError("source root has tracked modifications")
    tree = git("ls-tree", "-r", "HEAD")
    return {"source_commit": git("rev-parse", "HEAD"),
            "source_tree_sha256": sha256_bytes(tree.encode()),
            "source_tree_clean": True}


def build_body(data_url, arm):
    selection = None
    if arm == "temporal-16":
        selection = {"policy": "temporal", "motion_threshold": 0.02,
                     "scene_threshold": 0.25, "max_frames": 16}
    elif arm == "uniform-16":
        selection = {"policy": "uniform", "max_frames": 16}
    video = {"url": data_url}
    if selection is not None:
        video["frame_selection"] = selection
    return {"model": "qwen3-native", "stream": True, "temperature": 0.0,
            "top_p": 1.0, "seed": SEED, "max_tokens": 32,
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": video},
                {"type": "text", "text": QUESTION}]}]}


def stream_chat(base_url, body, timeout_s):
    """Throughput transport accepting either real stop or length terminal."""
    payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(base_url + "/v1/chat/completions", data=payload,
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
    answer = "".join(text_parts)
    if status != 200 or not done or terminal not in ("stop", "length") \
            or not answer.strip() or not request_id:
        raise RuntimeError(f"invalid native SSE status/done/terminal/answer: {status} "
                           f"{done} {terminal!r} {len(answer)} {request_id!r}; "
                           f"frames={json.dumps(frames, sort_keys=True)}")
    return {"request_id": request_id, "http_status": status, "answer": answer,
            "answer_sha256": sha256_bytes(answer.encode()), "first_event_s": first_event_s,
            "ttft_s": ttft_s, "total_s": total_s, "finish_reason": terminal,
            "request_bytes": len(payload)}, frames


def schedule():
    rng = random.Random(SEED)
    warmup = [{"phase": "warmup", "arm": arm, "repetition": 0, "padding": False}
              for arm in ARMS]
    warmup.append({"phase": "warmup", "arm": "default-8", "repetition": 1,
                   "padding": True})
    rng.shuffle(warmup)
    measured = [{"phase": "measured", "arm": arm, "repetition": repetition,
                 "padding": False}
                for repetition in range(8) for arm in ARMS]
    rng.shuffle(measured)
    for index, step in enumerate(warmup + measured):
        step["schedule_index"] = index
    return warmup + measured


def parse_execution(path):
    found = {}
    required_int = ("prompt_tokens", "prefill_tokens", "prefix_reused", "image_tokens",
                    "generated_tokens", "elapsed_ms")
    required_float = ("vision_ms", "prefill_ms", "decode_ms", "decode_tok_per_s",
                      "video_input_ms", "prompt_prep_ms", "unattributed_ms")
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("qwen38 request request_id="):
            continue
        fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
        request_id = fields.get("request_id")
        if not request_id or request_id in found:
            raise ValueError("missing/duplicate native execution request id")
        record = {"request_id": request_id}
        record.update({key: int(fields[key]) for key in required_int})
        record.update({key: float(fields[key]) for key in required_float})
        if record["prefix_reused"] != 0 or record["unattributed_ms"] < -0.001:
            raise ValueError("cold-prefix or timing decomposition failed")
        found[request_id] = record
    return found


class ResourceMonitor:
    def __init__(self):
        self.samples = []
        self.error = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def poll(self):
        while not self.stop_event.is_set():
            try:
                self.samples.append(query_gpu()["memory_used_mib"])
            except Exception as error:  # fail closed below
                self.error = str(error)
                return
            self.stop_event.wait(0.05)

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        if self.error or not self.samples:
            raise ValueError(f"GPU resource monitor failed: {self.error}")
        return {"scope": "whole-device-used-memory", "physical_gpu": 1,
                "sample_count": len(self.samples), "baseline_mib": self.samples[0],
                "peak_mib": max(self.samples), "final_mib": self.samples[-1],
                "samples_mib": self.samples}


def wait_ready(url, processes, timeout=240):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for name, process in processes.items():
            if process.poll() is not None:
                raise RuntimeError(f"{name} exited before ready with {process.returncode}")
        try:
            with urllib.request.urlopen(url + "/ready", timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.1)
    raise TimeoutError("isolated native stack did not become ready")


def stop_owned(processes):
    records = []
    for name in ("gateway", "native"):
        process = processes.get(name)
        if process is None:
            continue
        record = {"name": name, "pid": process.pid,
                  "pre_stop_returncode": process.poll(), "sigterm_sent": False,
                  "sigkill_sent": False}
        if process.poll() is None:
            process.terminate()
            record["sigterm_sent"] = True
        try:
            record["returncode"] = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()  # only this invocation's recorded process
            record["sigkill_sent"] = True
            record["returncode"] = process.wait(timeout=10)
        records.append(record)
    return records


def run_stack(args):
    if os.environ.get("UMMINFER_LIVE_OPT_IN") != "1":
        raise ValueError("live execution requires UMMINFER_LIVE_OPT_IN=1")
    if args.implementation not in ("baseline", "candidate") or args.mode not in ("serial", "batch4"):
        raise ValueError("invalid implementation/mode")
    out = Path(args.out_dir).resolve()
    if out.exists():
        raise ValueError("refusing to overwrite a stack evidence directory")
    out.mkdir(parents=True)
    source = Path(args.source).resolve()
    if sha256_file(source) != EXPECTED_SOURCE:
        raise ValueError("natural-full-330 source identity mismatch")
    prereg_path = Path(args.preregistration).resolve()
    prereg = json.loads(prereg_path.read_bytes())
    if prereg["experiment"] != "qwen38_natural_video_decode_prefill_throughput_v1" \
            or prereg["live_workload"]["requests_total"] != 224:
        raise ValueError("throughput preregistration identity/schedule mismatch")
    gateway, native = Path(args.gateway).resolve(), Path(args.native_stage).resolve()
    model = Path(args.model).resolve()
    identity = source_identity(args.source_root)
    pre = query_gpu()
    if pre["compute_apps"] or pre["memory_used_mib"] > 16:
        raise ValueError(f"GPU 1 is not idle before stack: {pre}")
    ipc = out / "ipc"
    ipc.mkdir()
    url = f"http://127.0.0.1:{args.port}"
    if args.port == 43301:
        raise ValueError("resident service port is forbidden")
    native_log = out / "native-stage.log"
    gateway_log = out / "gateway.log"
    processes = {}
    monitor = None
    records = []
    cleanup = None
    error = None
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "1"
        env["UMM_QWEN38_PREFIX_CACHE"] = "0"
        native_cmd = [str(native), "--kind", "qwen38", "--name", "qwen",
                      "--model", str(model), "--gpu", "0", "--batch-size",
                      "1" if args.mode == "serial" else "4",
                      "--recv-endpoint", f"ipc://{ipc}/entry",
                      "--coordinator-endpoint", f"ipc://{ipc}/completion",
                      "--abort-endpoint", f"ipc://{ipc}/abort"]
        with native_log.open("wb") as log:
            processes["native"] = subprocess.Popen(native_cmd, cwd=args.source_root,
                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        gateway_cmd = [str(gateway), "serve", "--host", "127.0.0.1", "--port",
                       str(args.port), "--model", "qwen3-native", "--entry", "qwen",
                       "--entry-endpoint", f"ipc://{ipc}/entry",
                       "--completion", f"ipc://{ipc}/completion",
                       "--abort", f"ipc://{ipc}/abort", "--max-in-flight", "4"]
        with gateway_log.open("wb") as log:
            processes["gateway"] = subprocess.Popen(gateway_cmd, cwd=args.source_root,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        wait_ready(url, processes)
        ready_device = query_gpu()
        native_pid = processes["native"].pid
        if len(ready_device["compute_apps"]) != 1:
            raise ValueError(f"GPU 1 does not have exactly one owned native compute process: {ready_device}")
        # nvidia-smi reports the host-namespace PID while Popen reports the
        # container PID. With an empty preflight and one post-load process, retain
        # both identities without pretending the namespace values are equal.
        native_compute_pid = ready_device["compute_apps"][0]["pid"]
        encoded = base64.b64encode(source.read_bytes()).decode("ascii")
        data_url = "data:video/mp4;base64," + encoded
        monitor = ResourceMonitor()
        monitor.thread.start()

        def execute(step):
            body = build_body(data_url, step["arm"])
            response, frames = stream_chat(url, body, 30)
            return step, body, response, frames

        steps = schedule()
        for begin in range(0, len(steps), 4 if args.mode == "batch4" else 1):
            group = steps[begin:begin + (4 if args.mode == "batch4" else 1)]
            if args.mode == "batch4" and len(group) != 4:
                raise ValueError("incomplete batch-four group")
            if args.mode == "serial":
                results = [execute(group[0])]
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    results = list(pool.map(execute, group))
            for step, body, response, frames in results:
                row = {**step, "status": "COMPLETED", "response": response,
                       "canonical_request_sha256": sha256_bytes(json.dumps(
                           body, sort_keys=True, separators=(",", ":")).encode())}
                records.append(row)
                relative = Path("records") / f"schedule-{step['schedule_index']:03d}.json"
                write_json(out / relative, row)
                write_json(out / relative.with_suffix(".sse.json"), frames)
            with urllib.request.urlopen(url + "/health", timeout=5) as health_response:
                health = json.load(health_response)
            if health.get("in_flight") != 0:
                raise ValueError("in_flight did not return to zero after schedule group")
        if len(records) != 28:
            raise ValueError("stack did not complete 28 requests")
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
    finally:
        if monitor is not None:
            resources = monitor.stop()
        else:
            resources = None
        cleanup_records = stop_owned(processes)
        deadline = time.monotonic() + 15
        post = query_gpu()
        while (post["compute_apps"] or post["memory_used_mib"] > 16) \
                and time.monotonic() < deadline:
            time.sleep(0.1)
            post = query_gpu()
        cleanup = {"owned_processes": cleanup_records, "post_stop": post,
                   "passed": not post["compute_apps"] and post["memory_used_mib"] <= 16}
        write_json(out / "cleanup.json", cleanup)
    if error:
        write_json(out / "error.json", {"error": error, "completed_records": len(records)})
        raise RuntimeError(error)
    native_records = short.parse_native_log(native_log)
    execution_records = parse_execution(native_log)
    for row in records:
        request_id = row["response"]["request_id"]
        native_record = short.native_for_request(native_records, request_id)
        execution = short.execution_for_request(execution_records, request_id)
        expected_budget = 8 if row["arm"] == "default-8" else 16
        if native_record["selected_frame_count"] != expected_budget \
                or native_record["max_frames"] != expected_budget \
                or native_record["engine_path"] != ("serial" if args.mode == "serial" else "batched"):
            raise ValueError("native budget/engine mismatch")
        row["native_evidence"] = native_record
        row["native_execution"] = execution
        response = row["response"]
        row["latency_ms"] = {"total": response["total_s"] * 1000,
            "ttft": response["ttft_s"] * 1000, "native_elapsed": execution["elapsed_ms"],
            "video_input": execution["video_input_ms"],
            "prompt_prep": execution["prompt_prep_ms"], "vision": execution["vision_ms"],
            "prefill": execution["prefill_ms"], "decode": execution["decode_ms"],
            "unattributed": execution["unattributed_ms"]}
        relative = Path("records") / f"schedule-{row['schedule_index']:03d}.json"
        write_json(out / relative, row)
    report = {"schema_version": 1, "implementation": args.implementation,
              "mode": args.mode, "block": args.block, "run_id": args.run_id,
              "preregistration_sha256": sha256_file(prereg_path),
              "source_sha256": EXPECTED_SOURCE, **identity,
              "binary": {"gateway_sha256": sha256_file(gateway),
                         "native_stage_sha256": sha256_file(native)},
              "model": {"id": "qwen3-native",
                        "config_sha256": sha256_file(model / "config.json"),
                        "manifest_sha256": args.model_manifest_sha256},
              "physical_gpu": 1, "batch_size": 1 if args.mode == "serial" else 4,
              "process_identity": {"native_container_pid": native_pid,
                                   "native_device_compute_pid": native_compute_pid},
              "prefix_cache": "disabled", "pre_stack": pre,
              "post_ready": ready_device, "resources": resources,
              "records": sorted(records, key=lambda row: row["schedule_index"]),
              "post_stop_cleanup": cleanup, "h2d_transfer": "NOT_COVERED_SEPARATELY"}
    write_json(out / "mode-result.json", report)
    print(json.dumps({"run_id": args.run_id, "records": len(records),
                      "cleanup_passed": cleanup["passed"], "out_dir": str(out)}, indent=2))
    return 0 if cleanup["passed"] else 2


def percentile95(values):
    return values[math.ceil(0.95 * len(values)) - 1]


def summarize(values):
    ordered = sorted(values)
    median = statistics.median(ordered)
    mean = statistics.fmean(ordered)
    return {"count": len(ordered), "median_ms": median, "p95_ms": percentile95(ordered),
            "iqr_ms": (lambda q: q[2] - q[0])(statistics.quantiles(ordered, n=4, method="inclusive")),
            "mad_ms": statistics.median(abs(value - median) for value in ordered),
            "mean_ms": mean, "cv": statistics.pstdev(ordered) / mean if mean else 0.0,
            "min_ms": ordered[0], "max_ms": ordered[-1]}


def outliers(rows):
    result = []
    for mode in ("serial", "batch4"):
        for implementation in ("baseline", "candidate"):
            for arm in ARMS:
                selected = [row for row in rows if row["phase"] == "measured"
                            and row["mode"] == mode and row["implementation"] == implementation
                            and row["arm"] == arm]
                if len(selected) != 16:
                    raise ValueError("measured population mismatch")
                for metric in ("total", "native_elapsed"):
                    values = [row["latency_ms"][metric] for row in selected]
                    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
                    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
                    for row in selected:
                        value = row["latency_ms"][metric]
                        if value < low or value > high:
                            result.append({"run_id": row["run_id"], "schedule_index": row["schedule_index"],
                                "request_id": row["response"]["request_id"], "mode": mode,
                                "implementation": implementation, "arm": arm, "metric": metric,
                                "value_ms": value, "lower_fence_ms": low, "upper_fence_ms": high,
                                "retained": True})
    return result


def sign_test(improved, worsened):
    count = improved + worsened
    if not count:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(count, k)
               for k in range(min(improved, worsened) + 1)) / 2 ** count)


def paired_stats(deltas):
    rng = random.Random(SEED)
    boots = []
    for _ in range(20000):
        boots.append(statistics.median(rng.choice(deltas) for _ in deltas))
    boots.sort()
    negative, positive = sum(value < 0 for value in deltas), sum(value > 0 for value in deltas)
    return {"count": len(deltas), "median_delta_ms": statistics.median(deltas),
            "mean_delta_ms": statistics.fmean(deltas),
            "mad_delta_ms": statistics.median(abs(value - statistics.median(deltas)) for value in deltas),
            "negative_pairs": negative, "positive_pairs": positive,
            "zero_pairs": len(deltas) - negative - positive,
            "exact_two_sided_sign_test_p": sign_test(positive, negative),
            "bootstrap_95_median_delta_ms": [boots[math.floor(0.025 * (len(boots) - 1))],
                                             boots[math.ceil(0.975 * (len(boots) - 1))]],
            "deltas_ms": deltas}


def analyze(args):
    runs = []
    for value in args.run:
        report_path = Path(value).resolve() / "mode-result.json"
        report = json.loads(report_path.read_bytes())
        if report["physical_gpu"] != 1 or report["prefix_cache"] != "disabled" \
                or not report["post_stop_cleanup"]["passed"] or len(report["records"]) != 28:
            raise ValueError("invalid/incomplete stack report")
        runs.append(report)
    rows = []
    for run in runs:
        for record in run["records"]:
            rows.append({key: run[key] for key in ("implementation", "mode", "block", "run_id")} | record)
    expected = [("serial", "A", "baseline"), ("serial", "B", "candidate"),
                ("serial", "C", "candidate"), ("serial", "D", "baseline"),
                ("batch4", "A", "baseline"), ("batch4", "B", "candidate"),
                ("batch4", "C", "candidate"), ("batch4", "D", "baseline")]
    actual = [(run["mode"], run["block"], run["implementation"]) for run in runs]
    if actual != expected:
        raise ValueError(f"ABBA run order/topology mismatch: {actual}")
    if len({run["preregistration_sha256"] for run in runs}) != 1 \
            or len({run["source_sha256"] for run in runs}) != 1 \
            or len({run["model"]["config_sha256"] for run in runs}) != 1 \
            or len({run["binary"]["gateway_sha256"] for run in runs}) != 1:
        raise ValueError("protocol/source/model/gateway identity drift")
    paired = []
    for mode in ("serial", "batch4"):
        for left_block, right_block in (("A", "B"), ("D", "C")):
            baseline = next(run for run in runs if run["mode"] == mode
                            and run["block"] == left_block)
            candidate = next(run for run in runs if run["mode"] == mode
                             and run["block"] == right_block)
            for brow in baseline["records"]:
                crow = next(row for row in candidate["records"]
                            if row["schedule_index"] == brow["schedule_index"])
                for field in ("phase", "arm", "repetition", "padding", "canonical_request_sha256"):
                    if crow[field] != brow[field]:
                        raise ValueError("paired request identity mismatch")
                answer_equal = crow["response"]["answer"] == brow["response"]["answer"]
                native_equal = all(crow["native_evidence"][field] == brow["native_evidence"][field]
                                   for field in NATIVE_PARITY_FIELDS)
                paired.append({"mode": mode, "baseline_block": left_block,
                    "candidate_block": right_block, "schedule_index": brow["schedule_index"],
                    "phase": brow["phase"], "arm": brow["arm"], "repetition": brow["repetition"],
                    "answer_exact": answer_equal, "native_semantics_exact": native_equal,
                    "total_delta_ms": crow["latency_ms"]["total"] - brow["latency_ms"]["total"]})
    if len(paired) != 112 or any(not row["answer_exact"] or not row["native_semantics_exact"]
                                 for row in paired):
        raise ValueError("exact output/native parity failed")
    repeated = []
    for mode in ("serial", "batch4"):
        for implementation in ("baseline", "candidate"):
            for arm in ARMS:
                selected = [row for row in rows if row["mode"] == mode
                            and row["implementation"] == implementation and row["arm"] == arm
                            and row["phase"] == "measured"]
                answers = {row["response"]["answer"] for row in selected}
                native = {json.dumps({field: row["native_evidence"][field]
                                      for field in NATIVE_PARITY_FIELDS}, sort_keys=True)
                          for row in selected}
                repeated.append({"mode": mode, "implementation": implementation, "arm": arm,
                                 "count": len(selected), "answer_exact": len(answers) == 1,
                                 "native_semantics_exact": len(native) == 1})
    cross_mode = []
    for implementation in ("baseline", "candidate"):
        for arm in ARMS:
            selected = [row for row in rows if row["implementation"] == implementation
                        and row["arm"] == arm and row["phase"] == "measured"]
            cross_mode.append({"implementation": implementation, "arm": arm,
                "answer_exact": len({row["response"]["answer"] for row in selected}) == 1,
                "native_semantics_exact": len({json.dumps({field: row["native_evidence"][field]
                    for field in NATIVE_PARITY_FIELDS}, sort_keys=True) for row in selected}) == 1})
    if any(not row["answer_exact"] or not row["native_semantics_exact"]
           for row in repeated + cross_mode):
        raise ValueError("repetition or serial/batch-four parity failed")
    distributions = {}
    paired_comparisons = {}
    for mode in ("serial", "batch4"):
        for arm in ARMS:
            for implementation in ("baseline", "candidate"):
                selected = [row for row in rows if row["phase"] == "measured"
                            and row["mode"] == mode and row["implementation"] == implementation
                            and row["arm"] == arm]
                distributions[f"{mode}/{implementation}/{arm}"] = {
                    metric: summarize([row["latency_ms"][metric] for row in selected])
                    for metric in METRICS}
            deltas = [row["total_delta_ms"] for row in paired if row["mode"] == mode
                      and row["arm"] == arm and row["phase"] == "measured"]
            paired_comparisons[f"{mode}/{arm}"] = paired_stats(deltas)
    primary_dist = distributions["serial/baseline/temporal-16"]["total"]
    primary_candidate = distributions["serial/candidate/temporal-16"]["total"]
    primary_paired = paired_comparisons["serial/temporal-16"]
    primary_delta = primary_paired["median_delta_ms"]
    relative = primary_delta / primary_dist["median_ms"]
    primary_pass = (primary_delta <= -25 and relative <= -0.05
        and primary_paired["bootstrap_95_median_delta_ms"][1] < 0
        and primary_paired["exact_two_sided_sign_test_p"] <= 0.05
        and primary_dist["cv"] <= 0.10 and primary_candidate["cv"] <= 0.10)
    secondary_regressions = []
    for mode in ("serial", "batch4"):
        for arm in ARMS:
            if mode == "serial" and arm == "temporal-16":
                continue
            base = distributions[f"{mode}/baseline/{arm}"]["total"]
            cand = distributions[f"{mode}/candidate/{arm}"]["total"]
            for metric in ("median_ms", "p95_ms"):
                change = cand[metric] / base[metric] - 1
                if change > 0.05:
                    secondary_regressions.append({"mode": mode, "arm": arm, "statistic": metric,
                                                  "relative_change": change})
    report = {"schema_version": 1, "experiment": "qwen38_natural_video_decode_prefill_throughput_v1",
        "preregistration_sha256": runs[0]["preregistration_sha256"], "runs": [{
            "run_id": run["run_id"], "mode": run["mode"], "block": run["block"],
            "implementation": run["implementation"], "source_commit": run["source_commit"],
            "source_tree_sha256": run["source_tree_sha256"], "binary": run["binary"],
            "resources": {key: value for key, value in run["resources"].items() if key != "samples_mib"},
            "cleanup_passed": run["post_stop_cleanup"]["passed"]} for run in runs],
        "request_count": len(rows), "measured_count": sum(row["phase"] == "measured" for row in rows),
        "pair_count": len(paired), "paired_requests": paired,
        "repetition_parity": repeated, "serial_batch4_parity": cross_mode,
        "distributions": distributions, "paired_comparisons": paired_comparisons,
        "outliers": outliers(rows), "secondary_regressions_over_5_percent": secondary_regressions,
        "primary": {"metric": "serial temporal-16 public total", "paired_delta_ms": primary_delta,
                    "relative_median_delta": relative, "passed": primary_pass},
        "claims": {"throughput_improvement": primary_pass and not secondary_regressions,
                   "output_and_native_semantics_parity": True,
                   "natural_fixture_accuracy": "SEPARATE_TASK72_CONTROL",
                   "generalized_quality": "NOT_COVERED", "perceptual_quality": "NOT_RUN",
                   "isolated_h2d_transfer": "NOT_COVERED"}}
    write_json(args.output, report)
    print(json.dumps({"primary": report["primary"],
                      "secondary_regressions": secondary_regressions,
                      "outliers": len(report["outliers"]), "claims": report["claims"]}, indent=2))
    return 0 if primary_pass and not secondary_regressions else 2


def self_check():
    steps = schedule()
    assert len(steps) == 28 and len(steps[4:]) == 24
    assert all(len(steps[i:i + 4]) == 4 for i in range(0, 28, 4))
    assert steps == schedule()
    values = summarize([1, 2, 3, 4, 100])
    assert values["median_ms"] == 3 and values["p95_ms"] == 100
    print("SELF-CHECK PASS: deterministic complete schedule and statistics")
    print("REAL_GENERATION: NOT_RUN; PERCEPTUAL_QUALITY: NOT_RUN")
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-stack")
    for name in ("implementation", "mode", "block", "run-id", "gateway", "native-stage",
                 "model", "source", "source-root", "out-dir", "preregistration",
                 "model-manifest-sha256"):
        run.add_argument("--" + name, required=True)
    run.add_argument("--port", type=int, required=True)
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
        print(f"VIDEO DECODE/PREFILL LIVE FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
