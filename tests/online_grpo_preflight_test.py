#!/usr/bin/env python3
"""CPU-only online-GRPO preflight, identity, objective, and tamper tests.

Usage: python3 -I tests/online_grpo_preflight_test.py
No GPU, model inference, real optimizer, or resident service is used.
"""

import builtins
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import behavior_token_alignment_test as behavior_fixture  # noqa: E402
import narrative_rl_fixtures as fixture  # noqa: E402
import online_rollout_test as online_fixture  # noqa: E402
import umminfer_behavior_token_alignment as alignment  # noqa: E402
import umminfer_online_grpo_preflight as preflight  # noqa: E402
import umminfer_online_rollout as online  # noqa: E402
import umminfer_rl_trajectory as trajectory  # noqa: E402
import umminfer_sse_replay as replay  # noqa: E402


def canonical(value):
    return online.canonical_json(value)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(
        encoding="utf-8").splitlines() if line]


def write_jsonl(path, records):
    Path(path).write_bytes(b"".join(canonical(record) + b"\n" for record in records))


def plan_with_groups():
    plan = online_fixture.fixture_plan()
    plan["batch_id"] = "online-preflight-deterministic-v1"
    plan["groups"].append({
        "prompt_id": "zero-variance-multilingual",
        "messages": [{"role": "user", "content": "零方差：灯笼🏮与动态水面。"}],
        "expected_visual_modalities": ["image", "video"],
        "seeds": [301, 302, 303],
    })
    return online.prepare_plan(plan)


def group_events(group_index, rollout_index):
    """Full/partial/redundant routes plus multilingual and explicit-anchor cases."""
    route_index = 0 if group_index == 2 else rollout_index
    events = online_fixture.response_events(group_index, route_index)
    if group_index == 1:
        first = "<spoken>黎明🌅 é 与灯笼🏮。 "
        second = f"候选 {rollout_index}：水面的倒影 é 尾🧪。"
        first_visual = next(index for index, event in enumerate(events)
                            if event["kind"] == "visual_slot")
        for event in events:
            if event["kind"] == "visual_slot":
                event["text"] = "灯笼🏮 é 动态水面"
                event["text_offset"] = len(first.encode("utf-8")) \
                    if event["slot_index"] == 0 else len((first + second).encode("utf-8"))
            elif event["kind"] == "artifact":
                event["text"] = ""
        events[0]["text"] = first
        events[first_visual - 1]["text"] = first
        deltas = [event for event in events if event["kind"] == "vlm_delta"]
        deltas[-1]["text"] = second
        events[-1]["text"] = first + second
    if group_index == 2:
        # Different targets with identical full routes produce exact zero variance.
        suffix = f" 零方差候选 {rollout_index}🌊。"
        for event in events:
            if event["kind"] == "vlm_delta":
                event["text"] += suffix if event is events[-2] else ""
        events[-1]["text"] += suffix
    return events


def make_online_record(root, item, events, evidence, finish_reason="stop"):
    """Task #48 capture path with exact online-scheduled request/model identity."""
    root = Path(root)
    request_id = "request-" + item["rollout_id"]
    raw = online_fixture.response_bytes(events, item, request_id, evidence)
    frames = raw.split(b"\n\n")
    for index in range(len(frames) - 1, -1, -1):
        if frames[index].startswith(b"data: ") and frames[index] != b"data: [DONE]":
            chunk = json.loads(frames[index][6:])
            chunk["choices"][0]["finish_reason"] = finish_reason
            frames[index] = b"data: " + canonical(chunk)
            break
    raw = b"\n\n".join(frames)
    recorder = trajectory.TrajectoryRecorder(root / "live/trajectories.jsonl")
    capture = trajectory.capture_sse_response(
        io.BytesIO(raw), clock=replay.LogicalTickClock())
    details = {}
    request_id, normalized, output, state, error = trajectory.parse_sse_stream(
        io.BytesIO(capture.raw_bytes), artifact_dir=recorder.artifact_dir,
        details=details, capture=capture)
    params, option_inputs = trajectory.request_parameter_references(item["body"], None)
    inputs = trajectory.input_references(item["body"]["messages"], None) + option_inputs
    record = recorder.record(
        session_id=item["body"]["session_id"], request_id=request_id,
        endpoint="/v1/chat/completions", inputs=inputs, params=params,
        events=normalized, terminal_state=state, error=error, output_text=output,
        artifacts=details["artifacts"], capture=capture, turn_index=item["schedule_index"],
        extra_fields={"ablation": {
            "prompt_id": item["prompt_id"],
            "expected_visual_modalities": item["expected_visual_modalities"],
        }})
    return record


def _consolidate_artifacts(work, bundle):
    source = work / "live/artifacts"
    destination = bundle / "artifacts"
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        target = destination / path.name
        if target.exists():
            if target.read_bytes() != path.read_bytes():
                raise AssertionError("fixture CAS collision")
        else:
            shutil.copyfile(path, target)


def make_online_bundle(root, *, version=1, exports=True, covered=True,
                       finish_reason="stop"):
    """Build a replayable online bundle from raw SSE and deterministic tokenizers."""
    root = Path(root)
    root.mkdir(parents=True)
    plan = plan_with_groups()
    schedule, schedule_sha = online.build_schedule(plan)
    tokenizer_dir = behavior_fixture.tokenizer_directory(root, version)
    tokenizer = behavior_fixture.V1Tokenizer() if version == 1 \
        else behavior_fixture.V2Tokenizer()
    kwargs = dict(
        tokenizer_dir=tokenizer_dir, tokenizer_id=f"online-fixture-v{version}",
        tokenizer_revision="online-fixture-revision",
        tokenization_version=f"online-tokenization-v{version}", format_version=version)
    records = []
    for item in schedule:
        work = root / "work" / item["rollout_id"]
        events = group_events(item["group_index"], item["rollout_index"])
        evidence = None
        if covered:
            if exports:
                base = make_online_record(
                    work / "base", item, copy.deepcopy(events), None)
                base = tokens_attach(base, tokenizer, kwargs)
                target = base["token_export"]["streams"]["sft_target"]
                evidence = {
                    "format": online.BEHAVIOR_FORMAT,
                    "tokenizer_sha256": alignment._loaded_tokenizer_digest(tokenizer_dir),
                    "token_ids": target["token_ids"],
                    "logprobs": [-(index + 1) / 4.0
                                 for index in range(target["token_count"])],
                    "mask": behavior_fixture.expected_mask(base),
                }
            else:
                evidence = {
                    "format": online.BEHAVIOR_FORMAT,
                    "tokenizer_sha256": alignment._loaded_tokenizer_digest(tokenizer_dir),
                    "token_ids": [11, 12, 13, 14],
                    "logprobs": [-0.25, -0.5, -1.0, -2.0],
                    "mask": [0, 1, 1, 1],
                }
        record = make_online_record(
            work, item, events, evidence, finish_reason=finish_reason)
        response = online._expected_attestation(
            item, plan, http_status=200,
            behavior_evidence="exposed" if covered else "not_exposed", attempt=0)
        behavior = online._behavior_evidence(
            record, response["behavior_evidence"], work / "live/artifacts")
        record["rollout_provenance"] = {
            **online._provenance(item, plan, schedule_sha, 0),
            "response": response, "behavior": behavior,
        }
        online.validate_online_record(
            record, item, plan, schedule_sha, work / "live/artifacts")
        records.append(record)
        _consolidate_artifacts(work, root / "bundle")

    # Reward is part of the source record before token_export binding, matching
    # the normal export workflow.  _export_replayed recomputes the same values.
    online._score_records(plan, records)
    if exports:
        records = [tokens_attach(record, tokenizer, kwargs) for record in records]
    bundle = root / "bundle"
    bundle.mkdir(exist_ok=True)
    (bundle / "plan.json").write_bytes(canonical(plan) + b"\n")
    online._export_replayed(plan, schedule, schedule_sha, records, bundle)
    return bundle, tokenizer_dir


def tokens_attach(record, tokenizer, kwargs):
    # Local import alias keeps the fixture builder's tokenizer doubles explicit.
    import umminfer_token_export as tokens
    return tokens.attach_token_export(record, tokenizer, **kwargs)


def make_config(bundle, tokenizer_dir, *, groups_per_step=1):
    rows = read_jsonl(Path(bundle) / "pack/grpo_train.jsonl") \
        + read_jsonl(Path(bundle) / "pack/grpo_eval.jsonl")
    rows.sort(key=lambda row: row["rollout_provenance"]["group_index"])
    plan = json.loads((Path(bundle) / "plan.json").read_text(encoding="utf-8"))
    behavior = plan["behavior_policy"]
    tokenizer_sha = alignment._loaded_tokenizer_digest(tokenizer_dir)
    current = {
        "policy_id": "current-online-policy", "revision": "current-revision-001",
        "model_id": behavior["model_id"], "model_revision": "current-model-revision-001",
        "model_sha256": "cc" * 32,
        "model_config_sha256": behavior["model_config_sha256"],
        "tokenizer_sha256": tokenizer_sha,
    }
    reference = {
        "policy_id": "frozen-reference-policy", "revision": "reference-revision-000",
        "model_id": behavior["model_id"], "model_revision": "reference-model-revision-000",
        "model_sha256": "dd" * 32,
        "model_config_sha256": behavior["model_config_sha256"],
        "tokenizer_sha256": tokenizer_sha,
    }
    return preflight._validate_config({
        "schema": preflight.CONFIG_SCHEMA,
        "current_policy": current, "reference_policy": reference,
        "objective": {
            "clip_eps": 0.2, "kl_coef": 0.04, "log_ratio_clamp": 10.0,
            "current_logprob_offsets": [0.0, math.log(1.25), math.log(0.75)],
            "reference_logprob_offsets": [0.0, 0.1, -0.1],
        },
        "prompt_token_ids": {
            row["prompt_sha256"]: [1000 + 2 * index, 1001 + 2 * index]
            for index, row in enumerate(rows)
        },
        "groups_per_step": groups_per_step,
    })


def refresh_rollout_manifest(bundle):
    bundle = Path(bundle)
    path = bundle / "rollout_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    trajectory = bundle / manifest["trajectory"]["path"]
    manifest["trajectory"].update({
        "sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
        "bytes": trajectory.stat().st_size,
    })
    write_json(path, manifest)
    Path(str(path) + ".sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "\n", encoding="ascii")


def refresh_checksums_for_consistent_tamper(bundle):
    root = Path(bundle)
    path = root / "SHA256SUMS"
    lines = []
    for name in path.read_text(encoding="ascii").splitlines():
        _, relative = name.split("  ", 1)
        lines.append(hashlib.sha256((root / relative).read_bytes()).hexdigest()
                     + "  " + relative)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def run_cli(arguments, *, block_ml=False):
    if not block_ml:
        return subprocess.run(
            [sys.executable, "-I", str(ROOT / "python/umminfer_online_grpo_preflight.py"),
             *map(str, arguments)], cwd=ROOT, text=True, capture_output=True, timeout=120)
    program = r'''
import builtins, runpy, sys
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split(".")[0] in {"torch", "transformers", "numpy", "mlx"}:
        raise AssertionError("forbidden ML import: " + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
'''
    return subprocess.run(
        [sys.executable, "-I", "-c", program,
         str(ROOT / "python/umminfer_online_grpo_preflight.py"), *map(str, arguments)],
        cwd=ROOT, text=True, capture_output=True, timeout=120)


class OnlineGrpoPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="umm-online-grpo-preflight-test-")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self, name="case", **kwargs):
        bundle, tokenizer = make_online_bundle(self.root / name, **kwargs)
        config = make_config(bundle, tokenizer)
        output = self.root / (name + "-preflight")
        result = preflight.prepare_preflight(bundle, output, config, tokenizer_dir=tokenizer)
        return bundle, tokenizer, config, output, result

    def test_multilingual_visual_anchor_zero_variance_and_exact_objective(self):
        bundle, tokenizer, config, output, result = self.prepare(version=1, exports=True)
        self.assertEqual(config, preflight.load_config(
            ROOT / "configs/training/online_grpo_preflight_fixture.json"))
        self.assertEqual((result["group_count"], result["rollout_count"]), (3, 9))
        self.assertFalse(result["torch_imported"])
        schedule = json.loads((output / "schedule.json").read_text(encoding="utf-8"))
        objectives = json.loads((output / "objective_fixtures.json").read_text(encoding="utf-8"))
        self.assertEqual(len(schedule["steps"]), 3)
        zero = objectives["groups"][2]
        self.assertFalse(zero["has_reward_variance"])
        self.assertEqual(zero["group_std"], 0.0)
        self.assertTrue(all(member["metrics"]["advantage"] == 0.0
                            for member in zero["members"]))
        for group in objectives["groups"]:
            for member in group["members"]:
                self.assertEqual(member["independent_token_alignment"]["status"], "PASS")
                self.assertTrue(member["independent_token_alignment"][
                    "sampled_posthoc_target_ids_equal"])
                packing = member["packing"]
                evidence = member["behavior_policy"]
                self.assertEqual(packing["input_ids"], packing["prompt_token_ids"]
                                 + packing["sampled_response_token_ids"])
                for token, supervised, label in zip(
                        packing["sampled_response_token_ids"],
                        packing["response_mask"][len(packing["prompt_token_ids"]):],
                        packing["labels"][len(packing["prompt_token_ids"]):]):
                    self.assertEqual(label, token if supervised else -100)
                self.assertNotEqual(member["current_policy"]["identity"]["revision"],
                                    member["behavior_policy"]["identity"]["revision"])
                self.assertEqual(member["current_policy"]["forward"], "NOT_RUN")
                self.assertEqual(member["reference_policy"]["forward"], "NOT_RUN")
        # Explicit high-ratio positive/negative clipping and independent reference KL.
        objective = {"clip_eps": 0.2, "kl_coef": 0.04, "log_ratio_clamp": 10.0}
        high = preflight._objective_metrics(math.log(2.0), 0.0, math.log(2.0), 1.0, objective)
        self.assertEqual(high["ratio"], 2.0)
        self.assertEqual(high["selected_surrogate"], 1.2)
        self.assertEqual(high["schulman_kl"], 0.0)
        negative = preflight._objective_metrics(
            math.log(2.0), 0.0, math.log(2.0) - 1.0, -1.0, objective)
        self.assertEqual(negative["selected_surrogate"], -2.0)
        self.assertEqual(negative["schulman_kl"], math.exp(-1.0))
        self.assertNotEqual(negative["behavior_mean_logp"],
                            negative["reference_mean_logp"])
        self.assertNotIn("torch", sys.modules)

    def test_v2_fresh_process_byte_verification_and_tamper_rejection(self):
        bundle, tokenizer, _config, output, _result = self.prepare(version=2, exports=True)
        rebuilt = self.root / "fresh-reconstruction"
        verified = run_cli([
            "--verify-bundle", output, "--online-pack", bundle,
            "--tokenizer-dir", tokenizer, "--out", rebuilt], block_ml=True)
        self.assertEqual(verified.returncode, 0, verified.stderr)
        report = json.loads(verified.stdout)
        self.assertEqual(report["fresh_process_reconstruction"], "BYTE_IDENTICAL")
        original_files = sorted(path.relative_to(output) for path in output.rglob("*")
                                if path.is_file())
        for relative in original_files:
            self.assertEqual((output / relative).read_bytes(),
                             (rebuilt / relative).read_bytes())

        for consistent in (False, True):
            tampered = self.root / ("tampered-consistent" if consistent else "tampered-hash")
            shutil.copytree(output, tampered)
            path = tampered / "objective_fixtures.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["groups"][0]["members"][0]["current_policy"]["logprobs"][0] -= 0.01
            write_json(path, value)
            if consistent:
                refresh_checksums_for_consistent_tamper(tampered)
            rejected = self.root / ("rejected-consistent" if consistent else "rejected-hash")
            result = run_cli([
                "--verify-bundle", tampered, "--online-pack", bundle,
                "--tokenizer-dir", tokenizer, "--out", rejected], block_ml=True)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(rejected.exists())

    def test_cli_without_exports_and_no_ml_import(self):
        root = self.root / "cli"
        bundle, tokenizer = make_online_bundle(root, exports=False)
        config = make_config(bundle, tokenizer)
        config_path = root / "config.json"
        write_json(config_path, config)
        output = root / "preflight"
        result = run_cli([
            "--preflight-only", "--online-pack", bundle, "--config", config_path,
            "--out", output], block_ml=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["schema"], preflight.RESULT_SCHEMA)
        self.assertFalse(report["torch_imported"])
        objective = json.loads((output / "objective_fixtures.json").read_text(encoding="utf-8"))
        self.assertTrue(all(member["independent_token_alignment"]["status"] == "NOT_COVERED"
                            for group in objective["groups"] for member in group["members"]))

    def test_invalid_inputs_fail_before_output_or_ml(self):
        # A valid online export with NOT_COVERED behavior is still not optimizer-ready.
        uncovered_root = self.root / "uncovered"
        uncovered, tokenizer = make_online_bundle(uncovered_root, exports=False, covered=False)
        config = make_config(uncovered, tokenizer)
        uncovered_config_path = uncovered_root / "config.json"
        write_json(uncovered_config_path, config)
        uncovered_output = self.root / "uncovered-output"
        rejected = run_cli([
            "--preflight-only", "--online-pack", uncovered, "--config", uncovered_config_path,
            "--out", uncovered_output], block_ml=True)
        self.assertEqual(rejected.returncode, 2)
        self.assertEqual(rejected.stdout, "")
        self.assertFalse(uncovered_output.exists())

        bundle, tokenizer, config, _output, _result = self.prepare("negatives", exports=True)
        cases = {}

        def missing_behavior(work):
            records = read_jsonl(work / "trajectories.jsonl")
            records[0]["rollout_provenance"]["behavior"]["logprobs"] = None
            write_jsonl(work / "trajectories.jsonl", records)
            refresh_rollout_manifest(work)
        cases["missing-behavior"] = missing_behavior

        def reordered(work):
            for name in ("grpo_train.jsonl", "grpo_eval.jsonl"):
                path = work / "pack" / name
                rows = read_jsonl(path)
                for row in rows:
                    row["rollouts"] = list(reversed(row["rollouts"]))
                write_jsonl(path, rows)
            online_fixture.resign_grpo_pack(work / "pack")
        cases["reordered-group"] = reordered

        def duplicate(work):
            path = work / "pack/grpo_train.jsonl"
            rows = read_jsonl(path)
            rows[0]["rollouts"][1] = copy.deepcopy(rows[0]["rollouts"][0])
            write_jsonl(path, rows)
            online_fixture.resign_grpo_pack(work / "pack")
        cases["duplicate-member"] = duplicate

        def stale(work):
            path = work / "pack/grpo_train.jsonl"
            rows = read_jsonl(path)
            rows[0]["rollouts"][0]["rollout_provenance"]["rollout_id"] = "0" * 32
            write_jsonl(path, rows)
            online_fixture.resign_grpo_pack(work / "pack")
        cases["stale-member"] = stale

        def cross_group(work):
            path = work / "pack/grpo_train.jsonl"
            rows = read_jsonl(path)
            other = next(row["rollout_provenance"]["schedule_group_id"]
                         for row in rows[1:])
            rows[0]["rollouts"][0]["rollout_provenance"]["group_id"] = other
            write_jsonl(path, rows)
            online_fixture.resign_grpo_pack(work / "pack")
        cases["cross-group"] = cross_group

        def nonfinite(work):
            path = work / "trajectories.jsonl"
            raw = path.read_text(encoding="utf-8")
            path.write_text(raw.replace('"logprobs":[-0.25', '"logprobs":[NaN', 1),
                            encoding="utf-8")
            refresh_rollout_manifest(work)
        cases["nonfinite-logprob"] = nonfinite

        def raw_tamper(work):
            records = read_jsonl(work / "trajectories.jsonl")
            raw_name = records[0]["sse_replay"]["raw_ref"]
            path = work / "artifacts" / raw_name
            payload = bytearray(path.read_bytes())
            payload[-3] ^= 1
            path.write_bytes(payload)
        cases["raw-cas"] = raw_tamper

        def cancelled(work):
            records = read_jsonl(work / "trajectories.jsonl")
            records[0]["terminal_state"] = "cancelled"
            write_jsonl(work / "trajectories.jsonl", records)
            refresh_rollout_manifest(work)
        cases["cancelled"] = cancelled

        def offline(work):
            path = work / "pack/grpo_train.jsonl"
            rows = read_jsonl(path)
            rows[0].pop("rollout_provenance")
            write_jsonl(path, rows)
            online_fixture.resign_grpo_pack(work / "pack")
        cases["frozen-offline"] = offline

        def partial_export(work):
            records = read_jsonl(work / "trajectories.jsonl")
            records[0].pop("token_export")
            write_jsonl(work / "trajectories.jsonl", records)
            refresh_rollout_manifest(work)
        cases["partial-supervision"] = partial_export

        for name, mutation in cases.items():
            with self.subTest(name=name):
                work = self.root / ("negative-" + name)
                shutil.copytree(bundle, work)
                mutation(work)
                output = self.root / ("negative-output-" + name)
                with self.assertRaises(ValueError):
                    preflight.prepare_preflight(work, output, config, tokenizer_dir=tokenizer)
                self.assertFalse(output.exists())
                self.assertNotIn("torch", sys.modules)

        for name, policy, key in (
                ("current-tokenizer-drift", "current_policy", "tokenizer_sha256"),
                ("reference-model-drift", "reference_policy", "model_config_sha256")):
            drifted = copy.deepcopy(config)
            drifted[policy][key] = "ef" * 32
            output = self.root / name
            with self.assertRaisesRegex(ValueError, "differs"):
                preflight.prepare_preflight(bundle, output, drifted, tokenizer_dir=tokenizer)
            self.assertFalse(output.exists())

        drifted_tokenizer = self.root / "drifted-tokenizer"
        shutil.copytree(tokenizer, drifted_tokenizer)
        with (drifted_tokenizer / "vocab.json").open("a", encoding="utf-8") as stream:
            stream.write(" ")
        drift_output = self.root / "tokenizer-file-drift-output"
        with self.assertRaisesRegex(ValueError, "tokenizer"):
            preflight.prepare_preflight(
                bundle, drift_output, config, tokenizer_dir=drifted_tokenizer)
        self.assertFalse(drift_output.exists())

        truncated_root = self.root / "truncated"
        truncated, truncated_tokenizer = make_online_bundle(
            truncated_root, exports=False, finish_reason="length")
        truncated_config = make_config(truncated, truncated_tokenizer)
        with self.assertRaisesRegex(ValueError, "truncated"):
            preflight.prepare_preflight(
                truncated, self.root / "truncated-output", truncated_config,
                tokenizer_dir=truncated_tokenizer)
        self.assertFalse((self.root / "truncated-output").exists())

    def test_split_resume_and_policy_revision_identities(self):
        bundle, tokenizer, config, output, _ = self.prepare("identity", exports=False)
        schedule = json.loads((output / "schedule.json").read_text(encoding="utf-8"))
        run = json.loads((output / "run_identity.json").read_text(encoding="utf-8"))
        self.assertEqual(len(schedule["checkpoints"]), 4)
        self.assertEqual(schedule["checkpoints"][0]["schedule_position"], 0)
        self.assertEqual(schedule["checkpoints"][-1]["schedule_position"], 3)
        self.assertEqual(schedule["checkpoints"][-1]["remaining_group_ids"], [])
        for index, step in enumerate(schedule["steps"]):
            self.assertEqual(step["checkpoint_before"],
                             schedule["checkpoints"][index]["checkpoint_id"])
            self.assertEqual(step["checkpoint_after"],
                             schedule["checkpoints"][index + 1]["checkpoint_id"])
        for checkpoint in schedule["checkpoints"]:
            self.assertEqual(checkpoint["run_id"], run["run_id"])
            self.assertEqual(checkpoint["optimizer_state"], "NOT_RUN")

        for label, policy, key in (
                ("behavior", None, None),
                ("current", "current_policy", "revision"),
                ("reference", "reference_policy", "model_sha256")):
            if policy is None:
                changed_bundle = self.root / "behavior-revision"
                shutil.copytree(bundle, changed_bundle)
                plan = json.loads((changed_bundle / "plan.json").read_text(encoding="utf-8"))
                # Changing only the source plan cannot bypass exact replay rejection.
                plan["behavior_policy"]["revision"] = "changed-behavior-revision"
                write_json(changed_bundle / "plan.json", plan)
                with self.assertRaises(ValueError):
                    preflight.prepare_preflight(
                        changed_bundle, self.root / "behavior-output", config,
                        tokenizer_dir=tokenizer)
                continue
            changed = copy.deepcopy(config)
            changed[policy][key] = "changed-" + label if key == "revision" else "ee" * 32
            changed_output = self.root / (label + "-identity")
            preflight.prepare_preflight(
                bundle, changed_output, changed, tokenizer_dir=tokenizer)
            other = json.loads((changed_output / "run_identity.json").read_text(encoding="utf-8"))
            other_schedule = json.loads((changed_output / "schedule.json").read_text(encoding="utf-8"))
            self.assertNotEqual(other["run_id"], run["run_id"])
            self.assertNotEqual(other_schedule["checkpoints"][0]["checkpoint_id"],
                                schedule["checkpoints"][0]["checkpoint_id"])

        split = copy.deepcopy(config)
        split["groups_per_step"] = 2
        split_output = self.root / "split-identity"
        preflight.prepare_preflight(bundle, split_output, split, tokenizer_dir=tokenizer)
        split_schedule = json.loads((split_output / "schedule.json").read_text(encoding="utf-8"))
        self.assertEqual(len(split_schedule["steps"]), 2)
        self.assertEqual(split_schedule["checkpoints"][1]["schedule_position"], 2)
        self.assertNotEqual(split_schedule["run_id"], run["run_id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
