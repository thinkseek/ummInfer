#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only online optimizer identity, arithmetic, raw-byte, and tamper tests."""

import ast
import copy
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import umminfer_online_grpo_optimizer as optimizer  # noqa: E402
import umminfer_online_evidence_audit as independent_audit  # noqa: E402
import umminfer_rl_reward as rewards  # noqa: E402
import online_grpo_optimizer_verify as verify  # noqa: E402

TARGET_A = "model.language_model.layers.3.self_attn.q_proj.lora_A.weight"
TARGET_B = "model.language_model.layers.3.self_attn.q_proj.lora_B.weight"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def write_json(path, value):
    Path(path).write_bytes(canonical(value) + b"\n")


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def tensor_doc(run, name, values, shapes):
    payload = struct.pack("<" + "f" * len(values), *values)
    path = run / "tensors" / (name + ".f32")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    layout = []
    offset = 0
    for tensor_name, shape in zip((TARGET_A, TARGET_B), shapes):
        count = math.prod(shape)
        layout.append({"name": tensor_name, "shape": shape, "offset": offset,
                       "count": count, "dtype": "float32", "endianness": "little"})
        offset += count
    return {"path": path.name, "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(), "scalar_count": len(values),
            "dtype": "float32", "endianness": "little", "layout": layout}


def plain_metrics(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {key: plain_metrics(item) for key, item in value.items()}
    return value


def make_update(root):
    run = root / "run"
    run.mkdir()
    before = [0.0] * 17408
    gradient = [0.0] * 17408
    after = [0.0] * 17408
    gradient[5120] = -0.25
    after[5120] = f32(0.001)
    delta = [f32(a - b) for a, b in zip(after, before)]
    exp_avg = [0.0] * 17408
    exp_avg_sq = [0.0] * 17408
    exp_avg[5120] = f32(0.1 * -0.25)
    exp_avg_sq[5120] = f32(0.001 * f32(-0.25 * -0.25))
    documents = {
        "before": tensor_doc(run, "before", before, [[1, 5120], [12288, 1]]),
        "gradient": tensor_doc(run, "gradient", gradient, [[1, 5120], [12288, 1]]),
        "postclip": tensor_doc(run, "postclip", gradient, [[1, 5120], [12288, 1]]),
        "after": tensor_doc(run, "after", after, [[1, 5120], [12288, 1]]),
        "delta": tensor_doc(run, "delta", delta, [[1, 5120], [12288, 1]]),
        "exp_avg": tensor_doc(run, "exp_avg", exp_avg, [[1, 5120], [12288, 1]]),
        "exp_avg_sq": tensor_doc(run, "exp_avg_sq", exp_avg_sq, [[1, 5120], [12288, 1]]),
    }
    objective = {"clip_eps": 0.2, "kl_coef": 0.04, "log_ratio_clamp": 10.0}
    members = []
    reference_members = []
    preflight_members = []
    for index, advantage in enumerate((0.5, -0.5)):
        current = [-0.2 - index * 0.1, -0.4 - index * 0.1]
        behavior = [-0.25 - index * 0.1, -0.5 - index * 0.1]
        reference = [-0.3 - index * 0.1, -0.55 - index * 0.1]
        mask = [1, 1] if index == 0 else [1, 0]
        rollout_id = f"rollout-{index}"
        metrics = plain_metrics(verify.decimal_metrics(
            current, behavior, reference, mask, advantage, objective))
        members.append({"rollout_id": rollout_id, "reward": 1.0 - index,
                        "advantage": advantage, "current_logprobs": current,
                        "sampled_behavior_logprobs": behavior,
                        "reference_logprobs": reference, "response_mask": mask,
                        "metrics": metrics, "raw_sse_sha256": "a" * 64,
                        "source_record_sha256": "b" * 64,
                        "packed_ids_labels_masks_sha256": "c" * 64})
        reference_members.append({"rollout_id": rollout_id, "logprobs": reference})
        preflight_members.append({"rollout_id": rollout_id, "advantage": advantage,
                                  "reward": 1.0 - index, "raw_sse_sha256": "a" * 64,
                                  "source_record_sha256": "b" * 64,
                                  "packed_ids_labels_masks_sha256": "c" * 64})
    update = {
        "schema": "umminfer-online-grpo-lora-update-v1", "run_identity_sha256": "d" * 64,
        "step": 1, "schedule_step_index": 0, "group_id": "group-0",
        "group_export_sha256": "e" * 64, "group_mean": 0.5, "group_std": 0.5,
        "has_reward_variance": True, "rollout_ids": ["rollout-0", "rollout-1"],
        "members": members, "objective_loss": math.fsum(
            member["metrics"]["objective_loss"] for member in members) / 2,
        "objective": objective,
        "gradient": {**documents["gradient"], "grad_norm": 0.25, "nonzero_scalars": 1,
                     "postclip": documents["postclip"]},
        "adamw": {"learning_rate": 0.001, "weight_decay": 0.0, "beta1": 0.9,
                  "beta2": 0.999, "epsilon": 1e-8, "max_grad_norm": 1.0,
                  "foreach": False},
        "moments_after": {"step": 1, "exp_avg": documents["exp_avg"],
                          "exp_avg_sq": documents["exp_avg_sq"]},
        "parameters": {"trainable_count": 17408, "trainable_tensors": 2,
                       "before": documents["before"], "after": documents["after"],
                       "delta": documents["delta"], "changed_scalars": 1,
                       "max_abs_delta": delta[5120], "delta_l2": abs(delta[5120])},
        "base_weights": "FROZEN_NO_GRAD_NO_OPTIMIZER_NO_VERSION_CHANGE",
    }
    write_json(run / "update-000001.json", update)
    reference = {"groups": [{"group_id": "group-0", "members": reference_members}]}
    preflight = [{"group_id": "group-0", "group_export_sha256": "e" * 64,
                  "members": preflight_members}]
    return run, update, reference, preflight


class VerifiableArithmeticRewardTests(unittest.TestCase):
    def test_exact_wrong_and_unknown_answers_match_independent_audit(self):
        reward = rewards.VerifiableArithmeticReward()
        spec = "umminfer_rl_reward:VerifiableArithmeticReward"
        for text, expected in (("340679", 1.0), (" 340679\n", 1.0),
                               ("340678", 0.0), ("340679.", 0.0)):
            record = {"schema_version": 2, "clean_text": text,
                      "ablation": {"prompt_id": "online-arithmetic-739x461"}}
            block = {"schema_version": 1, "name": reward.name, "score": expected}
            self.assertEqual(reward.score(record), expected)
            self.assertEqual(independent_audit.independent_reward(record, spec), block)
        for prompt_id, answer in (("online-arithmetic-7389x4621", "34144569"),
                                  ("online-arithmetic-4597x6839", "31438883"),
                                  ("online-arithmetic-3274x8957", "29325218"),
                                  ("online-arithmetic-8642x7351", "63527342"),
                                  ("online-arithmetic-6287x5399", "33943513")):
            record = {"schema_version": 2, "clean_text": answer,
                      "ablation": {"prompt_id": prompt_id}}
            self.assertEqual(reward.score(record), 1.0)
            self.assertEqual(independent_audit.independent_reward(record, spec)["score"], 1.0)
        unknown = {"schema_version": 2, "clean_text": "0",
                   "ablation": {"prompt_id": "unknown"}}
        self.assertIsNone(reward.score(unknown))


class OptimizerRawEvidenceTests(unittest.TestCase):
    def test_raw_gradient_delta_and_decimal_objective_pass(self):
        with tempfile.TemporaryDirectory(prefix="umm-online-optimizer-unit-") as tmp:
            run, _update, reference, groups = make_update(Path(tmp))
            result = verify.verify_update(run, 1, "d" * 64, reference, groups)
            self.assertEqual(result["nonzero_gradient_scalars"], 1)
            self.assertEqual(result["changed_parameter_scalars"], 1)
            self.assertLessEqual(result["objective_max_abs_delta"], 1e-8)

    def test_consistently_rehashed_gradient_and_delta_tamper_fail(self):
        for case in ("gradient", "delta"):
            with self.subTest(case=case), tempfile.TemporaryDirectory(
                    prefix="umm-online-optimizer-tamper-") as tmp:
                run, update, reference, groups = make_update(Path(tmp))
                if case == "gradient":
                    document = update["gradient"]
                    key = "gradient"
                else:
                    document = update["parameters"]["delta"]
                    key = "delta"
                path = run / "tensors" / document["path"]
                values = list(struct.unpack("<17408f", path.read_bytes()))
                values[5120] += 0.125
                payload = struct.pack("<17408f", *values)
                path.write_bytes(payload)
                document["sha256"] = hashlib.sha256(payload).hexdigest()
                write_json(run / "update-000001.json", update)
                with self.assertRaises(verify.VerificationError, msg=case):
                    verify.verify_update(run, 1, "d" * 64, reference, groups)

    def test_norm_preserving_gradient_sign_tamper_fails_adamw(self):
        with tempfile.TemporaryDirectory(prefix="umm-online-gradient-sign-") as tmp:
            run, update, reference, groups = make_update(Path(tmp))
            for document in (update["gradient"], update["gradient"]["postclip"]):
                path = run / "tensors" / document["path"]
                values = list(struct.unpack("<17408f", path.read_bytes()))
                values[5120] = 0.25
                payload = struct.pack("<17408f", *values)
                path.write_bytes(payload)
                document["sha256"] = hashlib.sha256(payload).hexdigest()
            write_json(run / "update-000001.json", update)
            with self.assertRaisesRegex(verify.VerificationError, "first moment"):
                verify.verify_update(run, 1, "d" * 64, reference, groups)

    def test_objective_tamper_fails(self):
        with tempfile.TemporaryDirectory(prefix="umm-online-objective-tamper-") as tmp:
            run, update, reference, groups = make_update(Path(tmp))
            update["members"][0]["current_logprobs"][0] += 0.01
            write_json(run / "update-000001.json", update)
            with self.assertRaises(verify.VerificationError):
                verify.verify_update(run, 1, "d" * 64, reference, groups)

    def test_verifier_is_stdlib_only(self):
        tree = ast.parse((ROOT / "tests/online_grpo_optimizer_verify.py").read_text(
            encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        forbidden = {"torch", "transformers", "numpy", "umminfer_online_rollout",
                     "umminfer_online_grpo_preflight", "umminfer_online_grpo_optimizer",
                     "umminfer_online_evidence_audit", "umminfer_rl_reward",
                     "umminfer_ablation_metrics"}
        self.assertTrue(forbidden.isdisjoint(imported))


class OptimizerPreparationTests(unittest.TestCase):
    def _documents(self, model_dir, *, zero_variance=True, eval_group=False):
        weight_map = {f"model-{index:02d}.safetensors": {
            "bytes": 1, "sha256": hashlib.sha256(bytes([index])).hexdigest()}
            for index in range(18)}
        model_sha = hashlib.sha256(canonical(weight_map)).hexdigest()
        config_sha = hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest()
        tokenizer_sha = "ab" * 32
        behavior = {"policy_id": "behavior", "revision": "0" * 40,
                    "model_id": "qwen3-native", "model_revision": "UNKNOWN",
                    "model_sha256": model_sha, "model_config_sha256": config_sha}
        policy = {"policy_id": "policy", "revision": "0" * 40,
                  "model_id": "qwen3-native", "model_revision": "UNKNOWN",
                  "model_sha256": model_sha, "model_config_sha256": config_sha,
                  "tokenizer_sha256": tokenizer_sha}
        groups = []
        fixtures = []
        steps = []
        for index in range(2):
            group = {"group_id": f"group-{index}", "schedule_group_id": f"schedule-{index}",
                     "group_export_sha256": "e" * 64, "split": "train",
                     "group_mean": 0.5, "group_std": 0.0 if zero_variance else 0.5,
                     "has_reward_variance": not zero_variance, "members": []}
            if index == 0 and eval_group:
                group["split"] = "eval"
            groups.append(group)
            fixtures.append({"group_id": group["group_id"], "split": group["split"],
                             "group_export_sha256": group["group_export_sha256"],
                             "has_reward_variance": group["has_reward_variance"],
                             "members": []})
            steps.append({"step_index": index, "group_ids": [group["group_id"]]})
        identity = {"mode": "online", "sampled_ids_replaced_by_posthoc_ids": False,
                    "token_export_alignment": "PASS_INDEPENDENT_STRUCTURAL",
                    "behavior_policy": behavior, "behavior_tokenizer_sha256": tokenizer_sha,
                    "current_policy": policy, "reference_policy": policy,
                    "groups": groups, "packed_input_order_sha256": "f" * 64,
                    "objective": {"clip_eps": 0.2, "kl_coef": 0.04,
                                  "log_ratio_clamp": 10.0}}
        manifest = {"status": "PASS", "step_count": 2, "group_count": 2, "run_id": "run"}
        documents = {"run_identity.json": {"run_id": "run", "identity": identity},
                     "schedule.json": {"run_id": "run", "groups_per_step": 1, "steps": steps},
                     "objective_fixtures.json": {"run_id": "run", "groups": fixtures},
                     "config.json": {"groups_per_step": 1}}
        return manifest, documents, weight_map

    def _prepare(self, root, *, zero_variance=True, eval_group=False):
        model = root / "model"
        model.mkdir()
        write_json(model / "config.json", {
            "model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {"vocab_size": 248320, "max_position_embeddings": 8192}})
        manifest, documents, weights = self._documents(
            model, zero_variance=zero_variance, eval_group=eval_group)
        pack = root / "pack"
        bundle = root / "preflight"
        pack.mkdir()
        bundle.mkdir()
        (pack / "rollout_manifest.json").write_bytes(b"{}\n")
        (bundle / "manifest.json").write_bytes(b"{}\n")
        args = types.SimpleNamespace(
            stop_after=2, out=root / "new-output", online_pack=pack,
            preflight_bundle=bundle, learning_rate=1e-3, max_grad_norm=1.0,
            seed_text="unit", model_dir=model, model_id="qwen3-native",
            model_revision="UNKNOWN", source_commit="0" * 40, resume_from=None)
        with mock.patch.dict(os.environ, {"UMMINFER_LIVE_OPT_IN": "1"}), \
                mock.patch.object(optimizer.preflight, "verify_preflight",
                                  return_value={"status": "PASS",
                                                "verification": "BYTE_IDENTICAL"}), \
                mock.patch.object(optimizer.preflight, "_read_verified_bundle",
                                  return_value={"manifest.json": canonical(manifest),
                                                **{name: canonical(value)
                                                   for name, value in documents.items()}}), \
                mock.patch.object(optimizer, "_model_files", return_value=([], weights)), \
                mock.patch.object(optimizer.alignment, "_loaded_tokenizer_digest",
                                  return_value="ab" * 32):
            optimizer.prepare_run(args)
        self.assertFalse(args.out.exists())

    def test_zero_variance_and_eval_groups_fail_before_ml_or_output(self):
        for kwargs in ({"zero_variance": True}, {"zero_variance": False, "eval_group": True}):
            with self.subTest(**kwargs), tempfile.TemporaryDirectory(
                    prefix="umm-online-prepare-reject-") as tmp:
                with self.assertRaises(ValueError):
                    self._prepare(Path(tmp), **kwargs)
                self.assertNotIn("torch", sys.modules)

    def test_existing_output_is_not_overwritten_on_rejected_launch(self):
        with tempfile.TemporaryDirectory(prefix="umm-online-existing-output-") as tmp:
            out = Path(tmp) / "existing"
            out.mkdir()
            sentinel = b'{"status":"RETAINED"}\n'
            (out / "summary.json").write_bytes(sentinel)
            command = [
                sys.executable, "-I", str(ROOT / "tests/online_grpo_optimizer_live.py"),
                "--model-dir", str(Path(tmp) / "model"), "--server-binary",
                str(Path(tmp) / "server"), "--stage-binary", str(Path(tmp) / "stage"),
                "--replay-binary", str(Path(tmp) / "replay"), "--trainer-python",
                str(Path(tmp) / "python"), "--source-commit", "0" * 40,
                "--batch-id", "existing", "--out-dir", str(out)]
            env = os.environ.copy()
            env["UMMINFER_LIVE_OPT_IN"] = "1"
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                    text=True, timeout=30, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((out / "summary.json").read_bytes(), sentinel)
            self.assertFalse((out / "SHA256SUMS").exists())

    def test_default_live_entry_points_are_not_run(self):
        with tempfile.TemporaryDirectory(prefix="umm-online-default-not-run-") as tmp:
            out = Path(tmp) / "out"
            commands = [[
                sys.executable, "-I", str(ROOT / "python/umminfer_online_grpo_optimizer.py"),
                "--online-pack", str(Path(tmp) / "pack"), "--preflight-bundle",
                str(Path(tmp) / "preflight"), "--model-dir", str(Path(tmp) / "model"),
                "--source-commit", "0" * 40, "--out", str(out)], [
                sys.executable, "-I", str(ROOT / "tests/online_grpo_optimizer_live.py"),
                "--model-dir", str(Path(tmp) / "model"), "--server-binary",
                str(Path(tmp) / "server"), "--stage-binary", str(Path(tmp) / "stage"),
                "--replay-binary", str(Path(tmp) / "replay"), "--trainer-python",
                str(Path(tmp) / "python"), "--source-commit", "0" * 40,
                "--batch-id", "not-run", "--out-dir", str(out)]]
            for command in commands:
                env = os.environ.copy()
                env.pop("UMMINFER_LIVE_OPT_IN", None)
                result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                        text=True, timeout=30, check=False)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("NOT_RUN", result.stdout)
                self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
