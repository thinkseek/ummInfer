#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU preflight tests plus the opt-in real SM90 joint-GRPO acceptance gate.

Default: python3 -I tests/joint_grpo_live_test.py
Live:    CUDA_VISIBLE_DEVICES=<idle-sm90-gpu> python3 -I tests/joint_grpo_live_test.py \
             --live --model-path DIR --evidence-dir DIR --source-commit COMMIT

The caller selects one physical GPU. The adapter uses visible logical cuda:0,
loads only local model files, and never starts or contacts a serving process.
"""

import argparse
import builtins
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import joint_trainer_test as fixture_helpers  # noqa: E402
import umminfer_joint_grpo_live as live  # noqa: E402
import umminfer_joint_trainer as stage_c  # noqa: E402
import umminfer_rl_trajectory as traj  # noqa: E402
import umminfer_sft_dataset as ds  # noqa: E402

FIXTURE_SEED = "joint-grpo-live-fixture-v1"
GROUP_SIZE = 3


def build_fixture_pack(root):
    """Build two complete three-rollout groups through the real recorder/builder."""
    root = Path(root)
    export_dir = root / "export"
    export_dir.mkdir(parents=True)
    recorder = traj.TrajectoryRecorder(export_dir / "trajectories.jsonl")
    cases = [
        (
            "Explain a lantern with static and moving illustrations.",
            ["A lantern glows at dawn. ", "It rises slowly."],
            "a red paper lantern glowing at dawn",
            "the lantern rising slowly into the dawn sky",
        ),
        (
            "Show an accretion disk and its dynamic plasma motion.",
            ["Superheated plasma forms a bright disk. ", "Matter spirals inward."],
            "labelled black hole accretion disk cross section",
            "plasma spiraling around a black hole",
        ),
    ]
    for case_index, (prompt, clean_parts, image_prompt, video_prompt) in enumerate(cases):
        split = len(clean_parts[0].encode("utf-8"))
        variants = [
            (
                "best",
                [("image", image_prompt, 0), ("video", video_prompt, split)],
            ),
            (
                "partial",
                [("image", image_prompt + " detailed still frame", 0)],
            ),
            (
                "redundant",
                [("image", image_prompt + " first redundant view", 0),
                 ("image", image_prompt + " second redundant view", split)],
            ),
        ]
        for variant, slots in variants:
            fixture_helpers._make_v2_record(
                recorder, prompt_text=prompt, clean_parts=clean_parts, slots=slots,
                session_id=f"grpo-live-{case_index}-{variant}",
                request_id=f"grpo-live-{case_index}-{variant}-request",
                expected_modalities=["image", "video"])
    pack_dir = root / "pack"
    manifest = ds.build_dataset(
        [recorder.jsonl_path], pack_dir, mode="preference",
        seed=FIXTURE_SEED, eval_ratio=0.0, min_margin=0.1,
        min_group_size=GROUP_SIZE, strict_artifacts=True)
    if manifest["outputs"]["grpo_train"]["record_count"] != 2:
        raise RuntimeError("deterministic GRPO fixture did not produce exactly two groups")
    return {
        "trajectory_path": recorder.jsonl_path,
        "trajectory_sha256": stage_c.sha256_file(recorder.jsonl_path),
        "pack_dir": pack_dir,
        "manifest": manifest,
    }


def _fake_model_dir(root):
    root = Path(root)
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({
        "model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"],
    }), encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"preflight-only-weight-bytes")
    return root


def _json_bytes(row, *, allow_nan=False):
    return json.dumps(
        row, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=allow_nan).encode("utf-8")


def _resign_pack_group(pack_dir, mutate, *, row_index=0, allow_nan=False):
    pack_path = Path(pack_dir) / "grpo_train.jsonl"
    rows = [json.loads(line) for line in pack_path.read_text(encoding="utf-8").splitlines()]
    mutate(rows, row_index)
    payload = b"".join(_json_bytes(row, allow_nan=allow_nan) + b"\n" for row in rows)
    pack_path.write_bytes(payload)
    manifest_path = Path(pack_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["grpo_train"].update({
        "bytes": len(payload), "sha256": stage_c.sha256_bytes(payload),
        "record_count": len(rows),
    })
    manifest_payload = stage_c.canonical_json(manifest) + b"\n"
    manifest_path.write_bytes(manifest_payload)
    Path(str(manifest_path) + ".sha256").write_text(
        stage_c.sha256_bytes(manifest_payload) + "\n", encoding="ascii")


def _prepare_without_ml(pack_dir, root, trajectory_path=None):
    real_import = builtins.__import__

    def no_ml(name, *args, **kwargs):
        if name == "torch" or name.startswith("transformers"):
            raise AssertionError(f"ML import during preflight: {name}")
        return real_import(name, *args, **kwargs)

    source = Path(trajectory_path or Path(pack_dir).parent / "export/trajectories.jsonl")
    with mock.patch("builtins.__import__", side_effect=no_ml):
        return live.prepare_live_run(
            dataset_dir=pack_dir, trajectory_paths=[source],
            output_dir=Path(root) / "run",
            config_path=ROOT / "configs/training/joint_grpo_live_gate.json",
            model_path=_fake_model_dir(Path(root) / "model"),
            source_commit="0" * 40, max_steps=2)


class JointGrpoLivePreflightTests(unittest.TestCase):
    def test_complete_groups_rewards_masks_and_provenance(self):
        with tempfile.TemporaryDirectory(prefix="umm-joint-grpo-preflight-") as tmp:
            root = Path(tmp)
            fixture = build_fixture_pack(root)
            prepared = _prepare_without_ml(fixture["pack_dir"], root)
            self.assertEqual(len(prepared["rows"]), 2)
            self.assertEqual(prepared["plan"]["active_stages"], ["grpo"])
            self.assertEqual(prepared["plan"]["REAL_TRAINING"], "NOT_RUN")
            for audit in prepared["group_audits"]:
                self.assertEqual(audit["group_size"], GROUP_SIZE)
                self.assertTrue(audit["has_reward_variance"])
                self.assertEqual(len(set(audit["rollout_order"])), GROUP_SIZE)
                scores = [item["reward"]["score"] for item in audit["rollouts"]]
                self.assertTrue(math.isclose(scores[0], 1.0, abs_tol=1e-12))
                self.assertTrue(math.isclose(scores[1], 2.0 / 3.0, abs_tol=1e-12))
                self.assertTrue(math.isclose(scores[2], 1.0 / 6.0, abs_tol=1e-12))
                for rollout in audit["rollouts"]:
                    self.assertGreater(rollout["span_counts"]["visual_route_tag"], 0)
                    self.assertGreater(rollout["span_counts"]["visual_prompt_body"], 0)
                    self.assertGreater(rollout["terminal_legs_verified"], 0)
                    self.assertTrue(rollout["export_integrity"]["cas_verified_on_disk"])
                    components = rollout["reward"]["components"]
                    self.assertIn("f1", components)
                    self.assertIn("redundant_slot_rate", components)
            self.assertNotIn("torch", sys.modules)
            self.assertNotIn("transformers", sys.modules)

    def test_resigned_invalid_groups_fail_before_ml_import(self):
        mutations = [
            ("invalid-group", lambda rows, i: rows[i].update(group_size=2), False),
            ("missing-rollout", lambda rows, i: rows[i]["rollouts"].pop(), False),
            ("duplicate-rollout", lambda rows, i: rows[i]["rollouts"].__setitem__(
                1, {**copy.deepcopy(rows[i]["rollouts"][0]), "rank": 1}), False),
            ("reordered-rollouts", lambda rows, i: rows[i].update(rollouts=[
                {**rows[i]["rollouts"][1], "rank": 0},
                {**rows[i]["rollouts"][0], "rank": 1},
                rows[i]["rollouts"][2],
            ]), False),
            ("nan-reward", lambda rows, i: rows[i]["rollouts"][0].update(
                score=float("nan")), True),
            ("infinite-advantage", lambda rows, i: rows[i]["rollouts"][0].update(
                advantage=float("inf")), True),
            ("tampered-reward", lambda rows, i: rows[i]["rollouts"][0].update(
                score=rows[i]["rollouts"][0]["score"] + 0.125), False),
            ("tampered-advantage", lambda rows, i: rows[i]["rollouts"][0].update(
                advantage=rows[i]["rollouts"][0]["advantage"] + 0.125), False),
            ("terminal-inconsistency", lambda rows, i: rows[i]["rollouts"][0][
                "visual_slots"][0].update(leg_state="failed"), False),
            ("target-hash-inconsistency", lambda rows, i: rows[i]["rollouts"][0].update(
                assistant_target=rows[i]["rollouts"][0]["assistant_target"] + "bad"), False),
            ("prompt-inconsistency", lambda rows, i: rows[i].update(
                prompt_text=rows[i]["prompt_text"] + "tampered"), False),
        ]
        for name, mutate, allow_nan in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory(
                    prefix="umm-joint-grpo-invalid-") as tmp:
                root = Path(tmp)
                fixture = build_fixture_pack(root)
                bad_pack = root / "bad-pack"
                shutil.copytree(fixture["pack_dir"], bad_pack)
                # The copied pack still binds to the original immutable exports.
                _resign_pack_group(bad_pack, mutate, allow_nan=allow_nan)
                with self.assertRaises(ValueError):
                    _prepare_without_ml(
                        bad_pack, root, trajectory_path=fixture["trajectory_path"])
                self.assertFalse((root / "run/live_manifest.json").exists())
                self.assertFalse((root / "run/checkpoints").exists())

    def test_cas_and_export_tampering_fail_before_ml_import(self):
        for mode in ("cas", "export"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(
                    prefix="umm-joint-grpo-source-invalid-") as tmp:
                root = Path(tmp)
                fixture = build_fixture_pack(root)
                if mode == "cas":
                    artifact = next((root / "export/artifacts").iterdir())
                    artifact.write_bytes(artifact.read_bytes() + b"tamper")
                else:
                    path = fixture["trajectory_path"]
                    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
                    record["terminal_state"] = "failed"
                    path.write_bytes(stage_c.canonical_json(record) + b"\n")
                with self.assertRaises(ValueError):
                    _prepare_without_ml(
                        fixture["pack_dir"], root,
                        trajectory_path=fixture["trajectory_path"])

    def test_objective_starts_at_unit_ratio_and_zero_kl(self):
        for advantage in (-1.25, 0.0, 1.75):
            with self.subTest(advantage=advantage):
                result = live._objective_metrics(
                    -3.5, -3.5, advantage, clip_eps=0.2, kl_coef=0.04, clamp=20.0)
                self.assertEqual(result["ratio"], 1.0)
                self.assertEqual(result["kl"], 0.0)
                self.assertTrue(math.isclose(
                    result["objective_loss"], -advantage, abs_tol=1e-12))
        changed = live._objective_metrics(
            -3.0, -3.5, 1.0, clip_eps=0.2, kl_coef=0.04, clamp=20.0)
        self.assertGreater(changed["ratio"], 1.0)
        self.assertTrue(changed["clip_active"])
        self.assertGreaterEqual(changed["kl"], 0.0)


def _load_manifest(run_dir):
    path = Path(run_dir) / "live_manifest.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    sidecar = Path(str(path) + ".sha256").read_text(encoding="ascii").strip()
    if sidecar != hashlib.sha256(path.read_bytes()).hexdigest():
        raise RuntimeError("live manifest SHA-256 mismatch")
    return result


def _run_adapter(args, run_name, *, resume=None, dataset_dir=None, stop_after=None):
    evidence = Path(args.evidence_dir)
    output = evidence / run_name
    command = [
        sys.executable, "-I", str(ROOT / "python/umminfer_joint_grpo_live.py"),
        "--dataset-dir", str(dataset_dir or Path(args.dataset_dir)),
        "--trajectory-path", str(args.trajectory_path),
        "--config", str(ROOT / "configs/training/joint_grpo_live_gate.json"),
        "--model-path", str(args.model_path),
        "--output-dir", str(output),
        "--source-commit", args.source_commit,
        "--max-steps", "2",
    ]
    if resume is not None:
        command.extend(["--resume-from", str(resume)])
    if stop_after is not None:
        command.extend(["--stop-after", str(stop_after)])
    proc = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    (evidence / f"{run_name}.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (evidence / f"{run_name}.stderr.log").write_text(proc.stderr, encoding="utf-8")
    return proc, output


def _reject_before_ml(args, fixture, name, mutate, *, allow_nan=False):
    evidence = Path(args.evidence_dir)
    bad_pack = evidence / f"{name}-pack"
    shutil.copytree(fixture["pack_dir"], bad_pack)
    _resign_pack_group(bad_pack, mutate, allow_nan=allow_nan)
    proc, output = _run_adapter(args, name, dataset_dir=bad_pack)
    rejected = proc.returncode != 0 \
        and not (output / "live_manifest.json").exists() \
        and not (output / "checkpoints").exists() \
        and "GRPO_ML_IMPORT_BEGIN" not in proc.stderr
    return {
        "status": "PASS" if rejected else "FAIL",
        "returncode": proc.returncode,
        "ml_import_marker_observed": "GRPO_ML_IMPORT_BEGIN" in proc.stderr,
        "live_manifest_created": (output / "live_manifest.json").exists(),
        "checkpoint_created": (output / "checkpoints").exists(),
        "optimizer_executed": False if rejected else "UNKNOWN",
        "stderr_sha256": stage_c.sha256_text(proc.stderr),
    }


def _assert_finite_json(value, path="root"):
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite evidence at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_json(item, f"{path}[{index}]")


def run_live_gate(args):
    evidence = Path(args.evidence_dir)
    if evidence.exists() and any(evidence.iterdir()):
        raise ValueError("live evidence directory must be absent or empty")
    evidence.mkdir(parents=True, exist_ok=True)
    fixture = build_fixture_pack(evidence / "fixture")
    args.dataset_dir = str(fixture["pack_dir"])
    args.trajectory_path = str(fixture["trajectory_path"])

    uninterrupted_proc, uninterrupted_dir = _run_adapter(args, "uninterrupted")
    if uninterrupted_proc.returncode:
        raise RuntimeError("uninterrupted live run failed; see captured stderr")
    uninterrupted = _load_manifest(uninterrupted_dir)
    first_proc, first_dir = _run_adapter(args, "split-first", stop_after=1)
    if first_proc.returncode:
        raise RuntimeError("first split run failed; see captured stderr")
    first = _load_manifest(first_dir)
    checkpoint = first_dir / "checkpoints/step-000001.pt"
    resumed_proc, resumed_dir = _run_adapter(args, "split-resumed", resume=checkpoint)
    if resumed_proc.returncode:
        raise RuntimeError("resumed live run failed; see captured stderr")
    resumed = _load_manifest(resumed_dir)

    exact_fields = (
        "loss", "post_update_loss", "post_update_loss_delta", "grad_norm",
        "record_ids", "rollout_ids", "record_order_sha256", "membership_order_sha256",
        "parameter_sha256_before", "parameter_sha256_after", "optimizer_state_sha256",
        "behavior_parameter_sha256", "policy_ratio_kl_evidence",
        "changed_parameter_tensors", "changed_parameters", "max_abs_parameter_delta",
        "group_audit",
    )
    first_step = uninterrupted["steps"][0]
    uninterrupted_second = uninterrupted["steps"][1]
    resumed_second = resumed["steps"][0]
    initial_mismatches = {
        key: {"uninterrupted": first_step[key], "split_first": first["steps"][0][key]}
        for key in exact_fields if first_step[key] != first["steps"][0][key]
    }
    mismatches = {
        key: {"uninterrupted": uninterrupted_second[key], "resumed": resumed_second[key]}
        for key in exact_fields if uninterrupted_second[key] != resumed_second[key]
    }
    first_checkpoint_equal = (
        uninterrupted["checkpoints"][0]["sha256"] == first["checkpoints"][0]["sha256"])
    second_checkpoint_equal = (
        uninterrupted["checkpoints"][1]["sha256"] == resumed["checkpoints"][-1]["sha256"])
    equivalence = not initial_mismatches and not mismatches \
        and first_checkpoint_equal and second_checkpoint_equal \
        and uninterrupted["final_parameter_sha256"] == resumed["final_parameter_sha256"] \
        and uninterrupted["final_optimizer_state_sha256"] == resumed["final_optimizer_state_sha256"]

    mechanics_errors = []
    for step in uninterrupted["steps"]:
        try:
            _assert_finite_json(step)
        except RuntimeError as exc:
            mechanics_errors.append(str(exc))
        if step["changed_parameters"] <= 0 or not step["parameters_finite_after"] \
                or step["nonzero_gradient_tensors"] <= 0:
            mechanics_errors.append(f"step {step['step']}: no finite nonzero update")
        if len(set(step["rollout_ids"])) != GROUP_SIZE:
            mechanics_errors.append(f"step {step['step']}: incomplete rollout membership")
        for rollout in step["group_audit"]["rollouts"]:
            if rollout["prompt_masked_tokens"] <= 0 \
                    or rollout["supervised_target_tokens"] <= 0 \
                    or rollout["visual_route_tag_tokens"] <= 0 \
                    or rollout["visual_prompt_body_tokens"] <= 0 \
                    or rollout["terminal_delimiter_tokens"] <= 0 \
                    or rollout["terminal_legs_verified"] <= 0 \
                    or not rollout["export_integrity"]["cas_verified_on_disk"]:
                mechanics_errors.append(f"step {step['step']}: invalid rollout mask/CAS audit")
    for rollout in first_step["policy_ratio_kl_evidence"]["pre_update"]:
        if rollout["ratio"] != 1.0 or rollout["kl"] != 0.0:
            mechanics_errors.append("initial policy/old ratio is not one or KL is nonzero")
    if not any(item["ratio"] != 1.0
               for item in first_step["policy_ratio_kl_evidence"]["post_update"]):
        mechanics_errors.append("post-update policy ratios did not change")
    improving_steps = sum(step["post_update_loss_delta"] < 0.0
                          for step in uninterrupted["steps"])
    if improving_steps == 0:
        mechanics_errors.append("no measured post-update GRPO objective decrease")

    rejections = {
        "invalid_group": _reject_before_ml(
            args, fixture, "invalid-group", lambda rows, i: rows[i].update(group_size=2)),
        "missing_rollout": _reject_before_ml(
            args, fixture, "missing-rollout", lambda rows, i: rows[i]["rollouts"].pop()),
        "duplicate_rollout": _reject_before_ml(
            args, fixture, "duplicate-rollout", lambda rows, i: rows[i]["rollouts"].__setitem__(
                1, {**copy.deepcopy(rows[i]["rollouts"][0]), "rank": 1})),
        "nonfinite_reward": _reject_before_ml(
            args, fixture, "nonfinite-reward", lambda rows, i: rows[i]["rollouts"][0].update(
                score=float("nan")), allow_nan=True),
        "tampered_reward": _reject_before_ml(
            args, fixture, "tampered-reward", lambda rows, i: rows[i]["rollouts"][0].update(
                score=rows[i]["rollouts"][0]["score"] + 0.125)),
        "terminal_hash_inconsistency": _reject_before_ml(
            args, fixture, "terminal-hash-inconsistency",
            lambda rows, i: rows[i]["rollouts"][0]["visual_slots"][0].update(leg_state="failed")),
    }
    passed = equivalence and not mechanics_errors and improving_steps > 0 \
        and all(item["status"] == "PASS" for item in rejections.values())
    report = {
        "schema": "umminfer-joint-grpo-live-acceptance-v1",
        "status": "PASS" if passed else "FAIL",
        "source_commit": args.source_commit,
        "source_test_sha256": stage_c.sha256_file(Path(__file__).resolve()),
        "fixture": {
            "schema_version": 2,
            "trajectory_sha256": fixture["trajectory_sha256"],
            "dataset_manifest_sha256": stage_c.sha256_file(fixture["pack_dir"] / "manifest.json"),
            "dataset_builder_sha256": stage_c.sha256_file(Path(ds.__file__).resolve()),
            "seed": FIXTURE_SEED,
            "group_count": 2,
            "group_size": GROUP_SIZE,
            "rollouts": "image+video / image-only / two redundant images; all clean prose with CAS-backed completed legs",
            "group_order_sha256": uninterrupted["group_order_sha256"],
        },
        "real_optimizer": {
            "status": "EXECUTED",
            "uninterrupted": uninterrupted,
            "split_first": first,
            "resumed": resumed,
            "mechanics_errors": mechanics_errors,
            "post_update_objective_decrease_steps": improving_steps,
        },
        "checkpoint_resume_equivalence": {
            "status": "PASS" if equivalence else "FAIL",
            "comparison_fields": list(exact_fields),
            "initial_step_mismatches": initial_mismatches,
            "mismatches": mismatches,
            "first_checkpoint_sha256_equal": first_checkpoint_equal,
            "second_checkpoint_sha256_equal": second_checkpoint_equal,
            "final_parameter_sha256_equal": (
                uninterrupted["final_parameter_sha256"] == resumed["final_parameter_sha256"]),
            "final_optimizer_state_sha256_equal": (
                uninterrupted["final_optimizer_state_sha256"] == resumed["final_optimizer_state_sha256"]),
        },
        "pre_ml_rejections": rejections,
        "SFT": "EXISTING_REAL_ACCEPTANCE_PRESERVED",
        "DPO": "EXISTING_REAL_ACCEPTANCE_PRESERVED",
        "online_rollout_generation": "NOT_RUN",
        "model_quality": "NOT_RUN",
        "generalized_improvement": "NOT_RUN",
        "production_model_training": "NOT_RUN",
        "upstream_revision_verification": "NOT_RUN",
        "vision_or_media_forward": "NOT_RUN",
    }
    report_path = evidence / "acceptance.json"
    report_path.write_bytes(stage_c.canonical_json(report) + b"\n")
    Path(str(report_path) + ".sha256").write_text(
        stage_c.sha256_file(report_path) + "\n", encoding="ascii")
    if not passed:
        raise RuntimeError("joint GRPO acceptance failed; see acceptance.json")
    print(json.dumps({
        "status": report["status"],
        "optimizer_steps": uninterrupted["optimizer_step_count"],
        "losses": [step["loss"] for step in uninterrupted["steps"]],
        "post_update_losses": [step["post_update_loss"] for step in uninterrupted["steps"]],
        "changed_parameters": [step["changed_parameters"] for step in uninterrupted["steps"]],
        "final_parameter_sha256": uninterrupted["final_parameter_sha256"],
        "checkpoint_resume_equivalence": "PASS",
        "pre_ml_rejections": sorted(rejections),
        "model_quality": "NOT_RUN", "generalized_improvement": "NOT_RUN",
    }, sort_keys=True))
    return 0


def _live_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--source-commit", required=True)
    return parser


if __name__ == "__main__":
    if "--live" in sys.argv:
        raise SystemExit(run_live_gate(_live_parser().parse_args()))
    result = unittest.main(exit=False, verbosity=2)
    print("REAL JOINT GRPO OPTIMIZER: NOT_RUN (default CPU preflight suite)")
    print("MODEL QUALITY / GENERALIZED IMPROVEMENT / ONLINE ROLLOUTS: NOT_RUN")
    raise SystemExit(0 if result.result.wasSuccessful() else 1)
