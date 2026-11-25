#!/usr/bin/env python3
"""CPU-only contracts for the Stage C narrative-routing ablation runner.

Usage: python3 -I tests/ablation_runner_test.py
No real model, optimizer, GPU, or resident service is used.
"""

import base64
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import umminfer_ablation_metrics as metrics  # noqa: E402
import umminfer_ablation_run as runner  # noqa: E402

CONFIG = ROOT / "benchmarks" / "ablations" / "narrative-routing.json"
REPLAY = ROOT / "benchmarks" / "ablations" / "narrative-routing-replay.json"
VISUAL_CONFIG = ROOT / "benchmarks" / "ablations" / "visual-form.json"
VISUAL_REPLAY = ROOT / "benchmarks" / "ablations" / "visual-form-replay.json"


def tree_bytes(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in Path(root).rglob("*") if path.is_file()}


def run_cli(out, config=CONFIG, extra=()):
    return subprocess.run(
        [sys.executable, "-I", str(ROOT / "python" / "umminfer_ablation_run.py"),
         "--config", str(config), "--replay-only", "--out", str(out), *extra],
        capture_output=True, text=True, timeout=60)


class ReplayHandler(BaseHTTPRequestHandler):
    cases = {}
    requests = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length))
        type(self).requests.append(body)
        narrative = body["narrative"]
        if not narrative["enabled"]:
            policy = "text_only"
        elif narrative["visual_mode"] == "image":
            policy = "forced_image"
        elif narrative["visual_mode"] == "video":
            policy = "forced_video"
        else:
            policy = "auto_tag"
        prompt_id = "conservation" if "momentum" in json.dumps(body["messages"]) \
            else "lantern"
        raw = base64.b64decode(self.cases[(policy, prompt_id)]["response_sse_base64"])
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)
        self.close_connection = True


class VisualFormReplayHandler(BaseHTTPRequestHandler):
    cases = {}
    renders = {}
    requests = []
    image_requests = {}

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length))
        type(self).requests.append((self.path, body))
        if self.path == "/v1/visual/render":
            case = self.renders[runner._sha256(body)]
            raw = base64.b64decode(case["response_base64"])
            content_type = "video/mp4"
        else:
            prompt_id = "conservation" if "momentum" in json.dumps(body["messages"]) \
                else "lantern"
            if body["narrative"]["visual_mode"] == "video":
                form = "video_only"
            else:
                count = self.image_requests.get(prompt_id, 0)
                type(self).image_requests[prompt_id] = count + 1
                form = "image_only" if count == 0 else "image_then_i2v_animate"
            case = self.cases[(form, prompt_id)]
            raw = base64.b64decode(case["response_sse_base64"])
            content_type = "text/event-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)
        self.close_connection = True


class AblationRunnerTests(unittest.TestCase):
    def test_replay_is_byte_reproducible_and_reports_required_provenance(self):
        with tempfile.TemporaryDirectory(prefix="umm-ablation-replay-") as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            for out in (first, second):
                completed = run_cli(out)
                self.assertEqual(completed.returncode, 0,
                                 completed.stdout + completed.stderr)
            self.assertEqual(tree_bytes(first), tree_bytes(second))

            report = json.loads((first / "report.json").read_text())
            self.assertEqual(report["run_status"], "COMPLETED")
            self.assertEqual(report["routing_policies"], list(runner.POLICIES))
            self.assertEqual(len(report["cases"]), 8)
            self.assertEqual(report["config_sha256"],
                             hashlib.sha256(CONFIG.read_bytes()).hexdigest())
            self.assertEqual(report["data_sha256"], hashlib.sha256(
                (CONFIG.parent / "narrative-routing-prompts.json").read_bytes()).hexdigest())
            self.assertEqual(report["replay_data_sha256"],
                             hashlib.sha256(REPLAY.read_bytes()).hexdigest())
            self.assertRegex(report["artifact_manifest_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(report["model"]["config_sha256"], r"^[0-9a-f]{64}$")
            self.assertIsNone(report["model"]["weights_sha256"])
            self.assertEqual(report["model"]["weights_coverage"], "NOT_COVERED")
            self.assertEqual(report["claims"]["learned_routing"], "NOT_RUN")
            self.assertEqual(report["claims"]["optimizer"], "NOT_RUN")
            self.assertEqual(report["claims"]["quality_improvement"], "NOT_COVERED")
            self.assertEqual(report["claims"]["latency_evidence"],
                             "CONTRACT_OBSERVATIONS_ONLY")
            source = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                    check=True, capture_output=True, text=True).stdout.strip()
            self.assertEqual(report["source_commit"], source)
            self.assertRegex(report["source_state_sha256"], r"^[0-9a-f]{64}$")
            self.assertIn("metric_coverage", report)
            self.assertEqual(report["metric_coverage"]["terminal_latency"], "COVERED")

            fixed_by_prompt = {}
            for case in report["cases"]:
                fixed_by_prompt.setdefault(case["prompt_id"], set()).add(
                    case["fixed_request_sha256"])
                self.assertEqual(case["terminal_state"], "completed")
                self.assertIsInstance(case["terminal_legs"], list)
                self.assertIsNotNone(case["latency"]["ttft_ms"])
                self.assertIsNotNone(case["latency"]["ttfa_ms"])
                self.assertIsNotNone(case["latency"]["terminal_ms"])
                self.assertEqual(case["metric_coverage"]["terminal_latency"], "COVERED")
                self.assertEqual(case["metric_coverage"]["routing_f1"], "NOT_COVERED")
                self.assertEqual(case["routing"]["f1"], None)
                self.assertEqual(case["routing"]["redundant_slot_rate"], None)
                self.assertIsNone(case["reward"]["score"])
                for artifact in case["artifacts"]:
                    self.assertRegex(artifact["sha256"], r"^[0-9a-f]{64}$")
                    self.assertTrue(artifact["ref"].startswith(artifact["sha256"] + "."))
                if case["routing_policy"] == "text_only":
                    self.assertEqual(case["latency"]["slots"], [])
                    self.assertEqual(case["metric_coverage"]["slot_latency"],
                                     "NOT_APPLICABLE")
                else:
                    self.assertTrue(case["latency"]["slots"])
                    for slot in case["latency"]["slots"]:
                        self.assertIsNotNone(slot["dispatch_ms"])
                        self.assertIsNotNone(slot["slot_to_artifact_ms"])
            self.assertEqual({len(values) for values in fixed_by_prompt.values()}, {1})

            expected = {
                "text_only": [], "forced_image": ["image"],
                "forced_video": ["video"], "auto_tag": ["image", "video"],
            }
            lantern = [case for case in report["cases"]
                       if case["prompt_id"] == "lantern"]
            for case in lantern:
                self.assertEqual(case["routing"]["predicted_visual_modalities"],
                                 expected[case["routing_policy"]])
                per_policy = json.loads((first / case["routing_policy"] /
                                         "report.json").read_text())
                self.assertEqual(per_policy["source_commit"], source)
                self.assertEqual(per_policy["source_state_sha256"],
                                 report["source_state_sha256"])
                self.assertEqual(per_policy["metric_coverage"]["terminal_latency"],
                                 "COVERED")
                self.assertEqual(per_policy["config_sha256"], report["config_sha256"])
                self.assertEqual(per_policy["data_sha256"], report["data_sha256"])
                records = [json.loads(line) for line in
                           (first / case["routing_policy"] /
                            "trajectory.jsonl").read_text().splitlines()]
                self.assertTrue(all(record["schema_version"] == 2 for record in records))

    def test_only_routing_and_selected_option_presence_vary(self):
        experiment = runner.load_experiment(CONFIG)
        for prompt in experiment.prompts:
            bodies = [runner.build_request(experiment, prompt, policy)
                      for policy in runner.POLICIES]
            projections = [runner.fixed_request(body) for body in bodies]
            self.assertTrue(all(projection == projections[0]
                                for projection in projections[1:]))
            self.assertEqual(len({runner.fixed_request_sha256(body)
                                  for body in bodies}), 1)
            self.assertEqual(len({runner._sha256(body) for body in bodies}), 4)
            self.assertEqual(bodies[0]["modalities"], ["text", "audio"])
            self.assertFalse(bodies[0]["narrative"]["enabled"])
            self.assertEqual(bodies[1]["narrative"]["visual_mode"], "image")
            self.assertEqual(bodies[2]["narrative"]["visual_mode"], "video")
            self.assertEqual(bodies[3]["narrative"]["visual_mode"], "auto")
            for body in bodies[1:]:
                self.assertTrue(body["narrative"]["enabled"])
            for body in bodies:
                self.assertEqual(body["seed"], 42)
                selected = set(body["modalities"])
                self.assertEqual("audio" in body, "audio" in selected)
                self.assertEqual("image" in body, "image" in selected)
                self.assertEqual("video" in body, "video" in selected)
                if "image" in body:
                    self.assertEqual(body["image"],
                                     experiment.config["fixed"]["visual"]["image"])
                if "video" in body:
                    self.assertEqual(body["video"],
                                     experiment.config["fixed"]["visual"]["video"])
                if "audio" in body:
                    self.assertEqual(body["audio"], experiment.config["fixed"]["audio"])

    def test_sm90_config_disables_audio_uniformly_without_changing_fixed_matrix(self):
        experiment = runner.load_experiment(
            ROOT / "benchmarks" / "ablations" / "narrative-routing-sm90.json")
        self.assertNotIn("audio", experiment.config["fixed"])
        for prompt in experiment.prompts:
            bodies = [runner.build_request(experiment, prompt, policy)
                      for policy in runner.POLICIES]
            for body in bodies:
                self.assertNotIn("audio", body)
                self.assertNotIn("audio", body["modalities"])
                self.assertEqual("image" in body, "image" in body["modalities"])
                self.assertEqual("video" in body, "video" in body["modalities"])
            projections = [runner.fixed_request(body) for body in bodies]
            self.assertTrue(all(projection == projections[0] for projection in projections[1:]))
            self.assertEqual(len({runner.fixed_request_sha256(body) for body in bodies}), 1)
        default = runner.load_experiment(CONFIG)
        self.assertIn("audio", default.config["fixed"])
        self.assertIn("audio", runner.build_request(
            default, default.prompts[0], "text_only")["modalities"])

    def test_labels_gate_f1_redundancy_and_export_only_reward(self):
        record = {"schema_version": 2, "visual_slots": [
            {"media_type": "image"}, {"media_type": "image"},
            {"media_type": "video"},
        ]}
        unlabelled = metrics.routing_observation(record)
        self.assertEqual(unlabelled["coverage"], "NOT_COVERED")
        self.assertIsNone(unlabelled["f1"])
        self.assertIsNone(unlabelled["redundant_slot_rate"])
        self.assertIsNone(metrics.NarrativeRoutingReward().score(record))

        labelled = copy.deepcopy(record)
        labelled["ablation"] = {"expected_visual_modalities": ["image"]}
        observation = metrics.routing_observation(labelled)
        self.assertEqual((observation["true_positive"],
                          observation["false_positive"],
                          observation["false_negative"]), (1, 1, 0))
        self.assertAlmostEqual(observation["f1"], 2 / 3)
        self.assertEqual(observation["redundant_slot_count"], 2)
        self.assertAlmostEqual(observation["redundant_slot_rate"], 2 / 3)
        self.assertAlmostEqual(metrics.NarrativeRoutingReward().score(labelled), 0.0)
        aggregate = metrics.aggregate_routing([record, labelled])
        self.assertEqual(aggregate["coverage"], "PARTIAL")
        self.assertEqual(aggregate["labelled_case_count"], 1)
        self.assertEqual(aggregate["total_case_count"], 2)
        self.assertAlmostEqual(aggregate["f1"], 2 / 3)
        self.assertAlmostEqual(aggregate["redundant_slot_rate"], 2 / 3)

        abstention = {"schema_version": 2, "visual_slots": [],
                      "ablation": {"expected_visual_modalities": []}}
        self.assertEqual(metrics.routing_observation(abstention)["f1"], 1.0)
        self.assertEqual(metrics.NarrativeRoutingReward().score(abstention), 1.0)

    def test_labelled_replay_scores_every_case_without_changing_requests(self):
        with tempfile.TemporaryDirectory(prefix="umm-ablation-labelled-") as tmp:
            case_dir = Path(tmp) / "config"
            shutil.copytree(CONFIG.parent, case_dir)
            dataset_path = case_dir / "narrative-routing-prompts.json"
            dataset = json.loads(dataset_path.read_text())
            labels = {"lantern": ["image", "video"], "conservation": ["image"]}
            for prompt in dataset["prompts"]:
                prompt["expected_visual_modalities"] = labels[prompt["id"]]
            dataset_path.write_text(json.dumps(dataset, sort_keys=True) + "\n")
            config_path = case_dir / CONFIG.name
            config = json.loads(config_path.read_text())
            config["dataset"]["sha256"] = hashlib.sha256(
                dataset_path.read_bytes()).hexdigest()
            config_path.write_text(json.dumps(config, sort_keys=True) + "\n")
            completed = run_cli(Path(tmp) / "out", config_path)
            self.assertEqual(completed.returncode, 0,
                             completed.stdout + completed.stderr)
            report = json.loads((Path(tmp) / "out" / "report.json").read_text())
            self.assertEqual(report["metric_coverage"]["routing_f1"], "COVERED")
            self.assertEqual(report["metric_coverage"]["routing_reward"], "COVERED")
            self.assertIsNotNone(report["metrics"]["routing_f1"])
            for case in report["cases"]:
                self.assertEqual(case["metric_coverage"]["routing_f1"], "COVERED")
                self.assertIsNotNone(case["routing"]["f1"])
                self.assertIsNotNone(case["routing"]["redundant_slot_rate"])
                self.assertIsInstance(case["reward"]["score"], float)
            default = runner.load_experiment(CONFIG)
            labelled = runner.load_experiment(config_path)
            self.assertEqual(runner.fixed_matrix(default),
                             runner.fixed_matrix(labelled))

    def test_latency_edge_coverage_and_plain_text_ttft(self):
        tracker = runner.LatencyTracker()
        tracker.observe({"type": "output_delta", "text": "plain"})
        self.assertIsNotNone(tracker.finish()["ttft_us"])
        case = {"terminal_state": "completed", "latency": {
            "ttft_ms": 1.0, "ttfa_ms": None, "terminal_ms": None, "slots": []}}
        _, coverage = runner._metric_view([case], [])
        self.assertEqual(coverage["terminal_latency"], "NOT_COVERED")
        self.assertEqual(coverage["slot_latency"], "NOT_APPLICABLE")

    def test_hash_policy_and_replay_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="umm-ablation-tamper-") as tmp:
            source = CONFIG.parent
            for index, mode in enumerate(("dataset-hash", "policy", "response")):
                with self.subTest(mode=mode):
                    case_dir = Path(tmp) / mode
                    shutil.copytree(source, case_dir)
                    config_path = case_dir / CONFIG.name
                    config = json.loads(config_path.read_text())
                    if mode == "dataset-hash":
                        config["dataset"]["sha256"] = "0" * 64
                    elif mode == "policy":
                        config["routing_policies"] = ["auto_tag"]
                    else:
                        replay_path = case_dir / REPLAY.name
                        replay = json.loads(replay_path.read_text())
                        raw = bytearray(base64.b64decode(
                            replay["cases"][0]["response_sse_base64"]))
                        raw[-3] ^= 1
                        replay["cases"][0]["response_sse_base64"] = base64.b64encode(raw).decode()
                        replay_path.write_text(json.dumps(replay, sort_keys=True) + "\n")
                        config["replay"]["sha256"] = hashlib.sha256(
                            replay_path.read_bytes()).hexdigest()
                    config_path.write_text(json.dumps(config, sort_keys=True) + "\n")
                    completed = run_cli(Path(tmp) / f"out-{index}", config_path)
                    self.assertEqual(completed.returncode, 2,
                                     completed.stdout + completed.stderr)
                    self.assertIn("ablation error:", completed.stderr)

    def test_live_observer_measures_out_of_band_without_changing_trajectories(self):
        replay = json.loads(REPLAY.read_text())
        ReplayHandler.cases = {(case["routing_policy"], case["prompt_id"]): case
                               for case in replay["cases"]}
        ReplayHandler.requests = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), ReplayHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(prefix="umm-ablation-live-") as tmp:
                replay_out = Path(tmp) / "replay"
                live_out = Path(tmp) / "live"
                self.assertEqual(run_cli(replay_out).returncode, 0)
                experiment = runner.load_experiment(CONFIG)
                report = runner.run_experiment(
                    experiment, live_out, replay_only=False,
                    base_url=f"127.0.0.1:{server.server_port}", timeout_s=5)
                self.assertEqual(report["run_status"], "COMPLETED")
                self.assertEqual(report["claims"]["latency_evidence"],
                                 "UNCONTROLLED_LIVE_MEASUREMENT")
                for case in report["cases"]:
                    self.assertEqual(case["latency"]["source"], "monotonic-live")
                    self.assertIsNotNone(case["latency"]["ttft_ms"])
                    self.assertIsNotNone(case["latency"]["ttfa_ms"])
                    for slot in case["latency"]["slots"]:
                        self.assertIsNotNone(slot["slot_to_artifact_ms"])
                for policy in runner.POLICIES:
                    self.assertEqual(
                        (replay_out / policy / "trajectory.jsonl").read_bytes(),
                        (live_out / policy / "trajectory.jsonl").read_bytes())
                for prompt_id in ("lantern", "conservation"):
                    requests = [body for body in ReplayHandler.requests
                                if ("momentum" in json.dumps(body["messages"]))
                                == (prompt_id == "conservation")]
                    self.assertEqual(len(requests), 4)
                    projections = [runner.fixed_request(body) for body in requests]
                    self.assertTrue(all(item == projections[0]
                                        for item in projections[1:]))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_visual_form_replay_is_reproducible_and_binds_i2v_to_source_image(self):
        with tempfile.TemporaryDirectory(prefix="umm-visual-form-replay-") as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            for out in (first, second):
                completed = run_cli(out, VISUAL_CONFIG)
                self.assertEqual(completed.returncode, 0,
                                 completed.stdout + completed.stderr)
            self.assertEqual(tree_bytes(first), tree_bytes(second))
            report = json.loads((first / "report.json").read_text())
            self.assertEqual(report["run_status"], "COMPLETED")
            self.assertEqual(report["visual_forms"], list(runner.VISUAL_FORMS))
            self.assertEqual(len(report["cases"]), 6)
            self.assertEqual(report["metrics"]["visual_form_completion_rate"], 1.0)
            self.assertEqual(report["metrics"]["animation_success_rate"], 1.0)
            self.assertIsNone(report["metrics"]["image_text_relevance"])
            self.assertIsNone(report["metrics"]["first_frame_consistency"])
            self.assertEqual(report["metric_coverage"]["quality"], "NOT_COVERED")
            self.assertEqual(report["metric_coverage"]["ttfa"], "NOT_APPLICABLE")

            by_prompt = {}
            for case in report["cases"]:
                by_prompt.setdefault(case["prompt_id"], {})[case["visual_form"]] = case
                self.assertIsNone(case["latency"]["ttfa_ms"])
                self.assertEqual(case["metric_coverage"]["quality"], "NOT_COVERED")
                if case["visual_form"] == "image_then_i2v_animate":
                    self.assertEqual(len(case["animations"]), 1)
                    animation = case["animations"][0]
                    self.assertEqual(animation["terminal_state"], "completed")
                    self.assertIsNotNone(animation["latency"]["terminal_ms"])
                    self.assertEqual(animation["artifact"]["media_type"], "video")
                    self.assertEqual(animation["artifact"]["mime_type"], "video/mp4")
                    source = [artifact for artifact in case["artifacts"]
                              if artifact.get("slot_id") == animation["source_slot_id"]]
                    self.assertEqual(len(source), 1)
                    self.assertEqual(source[0]["sha256"],
                                     animation["artifact"]["source_image_sha256"])
                    self.assertTrue((first / case["visual_form"] / "artifacts" /
                                     animation["artifact"]["ref"]).is_file())
                else:
                    self.assertEqual(case["animations"], [])
            for cases in by_prompt.values():
                self.assertEqual(len({case["fixed_request_sha256"]
                                      for case in cases.values()}), 1)
                self.assertEqual(cases["image_only"]["request_sha256"],
                                 cases["image_then_i2v_animate"]["request_sha256"])
                self.assertNotEqual(cases["image_only"]["request_sha256"],
                                    cases["video_only"]["request_sha256"])

    def test_visual_form_completion_is_not_a_quality_metric(self):
        record = {
            "schema_version": 2,
            "terminal_state": "completed",
            "visual_slots": [{"slot_id": "slot-0", "media_type": "image"}],
            "artifacts": [{"media_type": "image", "sha256": "a" * 64}],
            "ablation": {"visual_form": "image_then_i2v_animate", "animations": []},
        }
        observation = metrics.visual_form_observation(record)
        self.assertEqual(observation["completion"], 0.0)
        self.assertEqual(observation["expected_animation_count"], 1)
        self.assertEqual(observation["completed_animation_count"], 0)
        self.assertEqual(observation["quality_coverage"], "NOT_COVERED")
        aggregate = metrics.aggregate_visual_form([record])
        self.assertEqual(aggregate["visual_form_completion_rate"], 0.0)
        self.assertEqual(aggregate["animation_success_rate"], 0.0)
        self.assertFalse(runner._case_success({
            "terminal_state": "completed", "animations": [],
            "visual_form_metrics": {"completion": 0.0}}))

    def test_visual_form_live_matches_replay_and_records_render_latency(self):
        replay = json.loads(VISUAL_REPLAY.read_text())
        VisualFormReplayHandler.cases = {
            (case["visual_form"], case["prompt_id"]): case for case in replay["cases"]}
        VisualFormReplayHandler.renders = {
            render["request_sha256"]: render
            for case in replay["cases"] for render in case.get("renders", [])}
        VisualFormReplayHandler.requests = []
        VisualFormReplayHandler.image_requests = {}
        server = ThreadingHTTPServer(("127.0.0.1", 0), VisualFormReplayHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(prefix="umm-visual-form-live-") as tmp:
                replay_out = Path(tmp) / "replay"
                live_out = Path(tmp) / "live"
                self.assertEqual(run_cli(replay_out, VISUAL_CONFIG).returncode, 0)
                experiment = runner.load_experiment(VISUAL_CONFIG)
                report = runner.run_experiment(
                    experiment, live_out, replay_only=False,
                    base_url=f"127.0.0.1:{server.server_port}", timeout_s=5)
                self.assertEqual(report["run_status"], "COMPLETED")
                for form in runner.VISUAL_FORMS:
                    self.assertEqual(
                        (replay_out / form / "trajectory.jsonl").read_bytes(),
                        (live_out / form / "trajectory.jsonl").read_bytes())
                for case in report["cases"]:
                    for animation in case["animations"]:
                        self.assertEqual(animation["latency"]["source"], "monotonic-live")
                        self.assertIsNotNone(animation["latency"]["terminal_ms"])
                render_requests = [body for path, body in VisualFormReplayHandler.requests
                                   if path == "/v1/visual/render"]
                self.assertEqual(len(render_requests), 2)
                for body in render_requests:
                    self.assertEqual(body["kind"], "video")
                    self.assertTrue(body["image"].startswith("data:image/png;base64,"))
                    self.assertEqual(body["num_frames"], 13)
                    self.assertEqual(body["steps"], 4)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_visual_form_render_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="umm-visual-form-tamper-") as tmp:
            case_dir = Path(tmp) / "config"
            shutil.copytree(VISUAL_CONFIG.parent, case_dir)
            replay_path = case_dir / VISUAL_REPLAY.name
            replay = json.loads(replay_path.read_text())
            render = next(render for case in replay["cases"]
                          for render in case.get("renders", []))
            raw = bytearray(base64.b64decode(render["response_base64"]))
            raw[-1] ^= 1
            render["response_base64"] = base64.b64encode(raw).decode()
            replay_path.write_text(json.dumps(replay, sort_keys=True) + "\n")
            config_path = case_dir / VISUAL_CONFIG.name
            config = json.loads(config_path.read_text())
            config["replay"]["sha256"] = hashlib.sha256(replay_path.read_bytes()).hexdigest()
            config_path.write_text(json.dumps(config, sort_keys=True) + "\n")
            completed = run_cli(Path(tmp) / "out", config_path)
            self.assertEqual(completed.returncode, 2,
                             completed.stdout + completed.stderr)
            self.assertIn("replay I2V response SHA-256 mismatch", completed.stderr)

    def test_serving_has_no_ablation_or_optimizer_dependency(self):
        for directory in (ROOT / "src", ROOT / "include"):
            for path in directory.rglob("*"):
                if path.is_file():
                    content = path.read_text(errors="ignore")
                    self.assertNotIn("umminfer_ablation", content, str(path))
                    self.assertNotIn("umminfer_rl_", content, str(path))


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=2).result
    print("REAL MODEL ROUTING QUALITY: NOT_RUN")
    print("REAL OPTIMIZER: NOT_RUN")
    sys.exit(0 if result.wasSuccessful() else 1)
