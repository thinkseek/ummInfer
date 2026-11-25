#!/usr/bin/env python3
"""Deterministic worker/checkpoint contracts; no real optimizer is executed.

Usage: python3 -I tests/rl_phase2_test.py
"""

import builtins
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import narrative_rl_fixtures as fixture  # noqa: E402
import rl_architecture_test as architecture  # noqa: E402
import umminfer_rl_reward as reward_mod  # noqa: E402
import umminfer_rl_worker as worker_mod  # noqa: E402


class Phase2WorkerTests(unittest.TestCase):
    def make_export(self, root):
        recorder, record = architecture.record_response(
            root, fixture.narrative_events())
        return recorder, record

    def test_metadata_train_is_not_run_and_checkpoint_chain_reopens(self):
        with tempfile.TemporaryDirectory(prefix="umm-rl-phase2-chain-") as tmp:
            recorder, _ = self.make_export(Path(tmp) / "export")
            worker = worker_mod.TrainingWorker(Path(tmp) / "state")
            first = worker.handle({"type": "train", "task_id": "metadata-0",
                                   "trajectory_path": str(recorder.jsonl_path)})
            second = worker.handle({"type": "train", "task_id": "metadata-1",
                                    "trajectory_path": str(recorder.jsonl_path)})
            self.assertEqual((first["sequence"], second["sequence"]), (0, 1))
            self.assertEqual(first["optimizer_status"], "NOT_RUN")
            self.assertEqual(second["optimizer_status"], "NOT_RUN")
            first_path = Path(first["checkpoint_path"])
            second_path = Path(second["checkpoint_path"])
            first_manifest = json.loads(first_path.read_text())
            second_manifest = json.loads(second_path.read_text())
            self.assertIsNone(first_manifest["algorithm"])
            self.assertIsNone(first_manifest["weights_ref"])
            self.assertEqual(first_manifest["optimizer_status"], "NOT_RUN")
            self.assertEqual(second_manifest["base_checkpoint_sha256"],
                             worker_mod.sha256_file(first_path))
            self.assertEqual(Path(str(second_path) + ".sha256").read_text().strip(),
                             worker_mod.sha256_file(second_path))
            reopened = worker_mod.TrainingWorker(Path(tmp) / "state")
            self.assertEqual(reopened.store.next_sequence, 2)

    def test_invalid_train_step_never_imports_trainer_or_consumes_sequence(self):
        with tempfile.TemporaryDirectory(prefix="umm-rl-phase2-gate-") as tmp:
            recorder, valid = self.make_export(Path(tmp) / "export")
            worker = worker_mod.TrainingWorker(Path(tmp) / "state")
            real_import = builtins.__import__

            def no_trainer(name, *args, **kwargs):
                if name in ("umminfer_rl_train", "torch", "transformers"):
                    raise AssertionError(f"optimizer import attempted: {name}")
                return real_import(name, *args, **kwargs)

            for score in (None, True, "1.0", float("nan"), float("inf"), 10 ** 1000):
                with self.subTest(score=score):
                    bad = copy.deepcopy(valid)
                    bad["reward"] = {"schema_version": 1, "name": "contract",
                                     "score": score}
                    recorder.jsonl_path.write_text(json.dumps(bad) + "\n")
                    with mock.patch("builtins.__import__", side_effect=no_trainer):
                        response = worker.handle({
                            "type": "train_step", "task_id": "must-not-run",
                            "trajectory_path": str(recorder.jsonl_path),
                            "model_path": str(Path(tmp) / "unused-model"),
                            "hyperparameters": {"lr": 1e-5, "seed": 1},
                        })
                    self.assertEqual(response["type"], "error")
                    self.assertEqual(response["optimizer_status"], "NOT_RUN")
                    self.assertIn("finite numeric reward", response["error"])
                    self.assertEqual(worker.store.next_sequence, 0)
                    self.assertEqual(list(worker.store.checkpoint_dir.glob("ckpt-*.json")), [])
            self.assertNotIn("torch", sys.modules)
            self.assertNotIn("transformers", sys.modules)

    def test_tampered_export_and_mixed_rewards_consume_no_sequence(self):
        with tempfile.TemporaryDirectory(prefix="umm-rl-phase2-reject-") as tmp:
            recorder, valid = self.make_export(Path(tmp) / "export")
            worker = worker_mod.TrainingWorker(Path(tmp) / "state")
            bad = copy.deepcopy(valid)
            bad["terminal_state"] = "cancelled"
            recorder.jsonl_path.write_text(json.dumps(bad) + "\n")
            response = worker.handle({"type": "train", "task_id": "bad-terminal",
                                      "trajectory_path": str(recorder.jsonl_path)})
            self.assertEqual(response["type"], "error")
            self.assertEqual(response["optimizer_status"], "NOT_RUN")
            self.assertEqual(worker.store.next_sequence, 0)

            mixed = copy.deepcopy(valid)
            mixed["reward"] = {"name": "one", "score": 1.0}
            second = copy.deepcopy(mixed)
            second["reward"] = {"name": "two", "score": 0.0}
            recorder.jsonl_path.write_text(
                json.dumps(mixed) + "\n" + json.dumps(second) + "\n")
            response = worker.handle({"type": "train", "task_id": "mixed-reward",
                                      "trajectory_path": str(recorder.jsonl_path)})
            self.assertEqual(response["type"], "error")
            self.assertIn("mixes reward names", response["error"])
            self.assertEqual(worker.store.next_sequence, 0)

    def test_checkpoint_sidecar_parent_and_sequence_tampering_are_rejected(self):
        for mode in ("sidecar", "checksum", "parent", "sequence"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(
                    prefix=f"umm-rl-phase2-{mode}-") as tmp:
                recorder, _ = self.make_export(Path(tmp) / "export")
                worker = worker_mod.TrainingWorker(Path(tmp) / "state")
                for index in range(2):
                    response = worker.handle({
                        "type": "train", "task_id": f"metadata-{index}",
                        "trajectory_path": str(recorder.jsonl_path)})
                    self.assertEqual(response["status"], "ok")
                second_path = worker.store.checkpoint_dir / "ckpt-00000001.json"
                sidecar = Path(str(second_path) + ".sha256")
                if mode == "sidecar":
                    sidecar.unlink()
                else:
                    manifest = json.loads(second_path.read_text())
                    if mode == "checksum":
                        manifest["reward"]["name"] = "tampered"
                    elif mode == "parent":
                        manifest["base_checkpoint_sha256"] = "0" * 64
                    else:
                        manifest["sequence"] = 9
                    second_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
                    if mode in ("parent", "sequence"):
                        sidecar.write_text(worker_mod.sha256_file(second_path) + "\n")
                with self.assertRaises(ValueError):
                    worker_mod.TrainingWorker(Path(tmp) / "state")

    def test_not_run_checkpoint_cannot_claim_weights(self):
        with tempfile.TemporaryDirectory(prefix="umm-rl-phase2-no-weights-") as tmp:
            store = worker_mod.CheckpointStore(tmp)
            with self.assertRaisesRegex(ValueError, "cannot claim"):
                store.write_checkpoint(
                    trajectory_path="unused.jsonl", record_count=1,
                    trajectory_sha256="0" * 64, reward_name=None,
                    algorithm="not-executed", weights_ref="unused.pt",
                    optimizer_status="NOT_RUN")
            self.assertEqual(store.next_sequence, 0)

    def test_stdio_protocol_and_unknown_fields(self):
        with tempfile.TemporaryDirectory(prefix="umm-rl-phase2-stdio-") as tmp:
            recorder, _ = self.make_export(Path(tmp) / "export")
            requests = [
                {"type": "ping", "future_optional_field": True},
                {"type": "unknown", "task_id": "unknown"},
                {"type": "train", "task_id": "metadata",
                 "trajectory_path": str(recorder.jsonl_path),
                 "future_optional_field": True},
            ]
            completed = subprocess.run(
                [sys.executable, "-I", str(ROOT / "python" / "umminfer_rl_worker.py"),
                 "--state-dir", str(Path(tmp) / "state")],
                input="".join(json.dumps(request) + "\n" for request in requests),
                text=True, capture_output=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            responses = [json.loads(line) for line in completed.stdout.splitlines()]
            self.assertEqual(responses[0]["type"], "hello")
            self.assertEqual(responses[1]["type"], "pong")
            self.assertEqual(responses[2]["type"], "error")
            self.assertEqual(responses[3]["type"], "train_result")
            self.assertEqual(responses[3]["optimizer_status"], "NOT_RUN")


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=2).result
    print("REAL OPTIMIZER: NOT_RUN (CPU-only deterministic protocol/checkpoint tests)")
    print("REAL MODEL QUALITY RESULT: NOT_RUN")
    sys.exit(0 if result.wasSuccessful() else 1)
