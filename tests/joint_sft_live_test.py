#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU preflight tests plus the opt-in real SM90 joint-SFT acceptance gate.

Default: python3 -I tests/joint_sft_live_test.py
Live:    python3 -I tests/joint_sft_live_test.py --live \
             --model-path DIR --evidence-dir DIR --source-commit COMMIT

The caller selects the physical GPU with CUDA_VISIBLE_DEVICES.  The adapter
always uses visible logical cuda:0 and never starts a serving process.
"""

import argparse
import builtins
import hashlib
import json
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
import umminfer_joint_sft_live as live  # noqa: E402
import umminfer_joint_trainer as stage_c  # noqa: E402
import umminfer_rl_trajectory as traj  # noqa: E402
import umminfer_sft_dataset as ds  # noqa: E402


def build_fixture_pack(root):
    """Build two deterministic schema-v2 turns through the real recorder/builder."""
    root = Path(root)
    export_dir = root / "export"
    export_dir.mkdir(parents=True)
    recorder = traj.TrajectoryRecorder(export_dir / "trajectories.jsonl")
    fixture_helpers._make_v2_record(
        recorder,
        prompt_text="Give a one-sentence explanation of gravity.",
        clean_parts=["Gravity pulls masses together."],
        slots=(),
        session_id="live-gate-text",
        request_id="live-text-0",
    )
    clean_parts = ["A lantern glows at dawn. ", "It rises slowly."]
    fixture_helpers._make_v2_record(
        recorder,
        prompt_text="Explain a lantern with static and moving illustrations.",
        clean_parts=clean_parts,
        slots=[
            ("image", "a red paper lantern glowing at dawn", 0),
            ("video", "the lantern rising slowly into the dawn sky",
             len(clean_parts[0].encode("utf-8"))),
        ],
        session_id="live-gate-visual",
        request_id="live-visual-0",
    )
    pack_dir = root / "pack"
    manifest = ds.build_dataset(
        [recorder.jsonl_path], pack_dir, mode="sft",
        seed="joint-sft-live-fixture-v1", eval_ratio=0.0,
        strict_artifacts=True)
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
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
    }), encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"preflight-only-weight-bytes")
    return root


def _resign_pack_row(pack_dir, mutate):
    pack_path = Path(pack_dir) / "sft_train.jsonl"
    rows = [json.loads(line) for line in pack_path.read_text(
        encoding="utf-8").splitlines()]
    row = next((item for item in rows if item.get("visual_slots")), rows[0])
    mutate(row)
    payload = b"".join(stage_c.canonical_json(row) + b"\n" for row in rows)
    pack_path.write_bytes(payload)
    manifest_path = Path(pack_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["sft_train"].update({
        "bytes": len(payload),
        "sha256": stage_c.sha256_bytes(payload),
        "record_count": len(rows),
    })
    manifest_payload = stage_c.canonical_json(manifest) + b"\n"
    manifest_path.write_bytes(manifest_payload)
    Path(str(manifest_path) + ".sha256").write_text(
        stage_c.sha256_bytes(manifest_payload) + "\n", encoding="ascii")


class JointSftLivePreflightTests(unittest.TestCase):
    def test_fixture_preflight_and_provenance(self):
        with tempfile.TemporaryDirectory(prefix="umm-joint-live-preflight-") as tmp:
            root = Path(tmp)
            fixture = build_fixture_pack(root)
            model = _fake_model_dir(root / "model")
            real_import = builtins.__import__

            def no_ml(name, *args, **kwargs):
                if name == "torch" or name.startswith("transformers"):
                    raise AssertionError(f"ML import during preflight: {name}")
                return real_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=no_ml):
                prepared = live.prepare_live_run(
                    dataset_dir=fixture["pack_dir"],
                    output_dir=root / "run",
                    config_path=ROOT / "configs/training/joint_sft_live_gate.json",
                    model_path=model, source_commit="0" * 40, max_steps=2)
            self.assertEqual(len(prepared["rows"]), 2)
            self.assertTrue(any(a["text_only"] for a in prepared["row_audits"]))
            visual = [a for a in prepared["row_audits"] if a["visual_modalities"]]
            self.assertEqual(visual[0]["visual_modalities"], ["image", "video"])
            self.assertEqual(visual[0]["terminal_legs_verified"], 2)
            self.assertEqual(prepared["plan"]["REAL_TRAINING"], "NOT_RUN")
            self.assertEqual(len(prepared["model"]["available_weight_files"]), 1)
            self.assertNotIn("torch", sys.modules)
            self.assertNotIn("transformers", sys.modules)

    def test_invalid_targets_and_terminal_legs_fail_before_ml_import(self):
        mutations = [
            lambda row: row.update(assistant_target=row["assistant_target"] + "tamper"),
            lambda row: [slot.update(leg_state="failed") for slot in row["visual_slots"]],
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory(
                    prefix="umm-joint-live-invalid-") as tmp:
                root = Path(tmp)
                fixture = build_fixture_pack(root)
                bad_pack = root / "bad-pack"
                shutil.copytree(fixture["pack_dir"], bad_pack)
                _resign_pack_row(bad_pack, mutate)
                real_import = builtins.__import__

                def no_ml(name, *args, **kwargs):
                    if name == "torch" or name.startswith("transformers"):
                        raise AssertionError(f"ML import after invalid input: {name}")
                    return real_import(name, *args, **kwargs)

                with mock.patch("builtins.__import__", side_effect=no_ml):
                    with self.assertRaises(ValueError):
                        live.prepare_live_run(
                            dataset_dir=bad_pack, output_dir=root / "bad-run",
                            config_path=ROOT / "configs/training/joint_sft_live_gate.json",
                            model_path=_fake_model_dir(root / "model"),
                            source_commit="0" * 40, max_steps=2)
                self.assertFalse((root / "bad-run/live_manifest.json").exists())
                self.assertFalse((root / "bad-run/checkpoints").exists())

    def test_hf_offset_loss_mask_covers_text_slots_and_terminal_delimiters(self):
        target = "Plain [[image: red lantern]] tail"
        prompt = "Explain"
        full = prompt + "\n" + target
        target_start = len(prompt) + 1
        mask = stage_c.build_span_loss_mask(target)
        offsets = [(0, target_start)]
        for span in mask["spans"]:
            offsets.append((target_start + span["start"], target_start + span["end"]))
        ids = list(range(len(offsets)))
        labels, weights, annotations = live._token_loss_mask(
            offsets, target_start=target_start, target=target,
            token_ids=ids, mask_info=mask)
        self.assertEqual(labels[0], -100)
        self.assertEqual(weights[0], 0.0)
        self.assertTrue(all(value > 0 for value in weights[1:]))
        self.assertTrue(any(item["includes_terminal_delimiter"]
                            for item in annotations))
        route_weights = [item["weight"] for item in annotations
                         if "visual_route_tag" in item["span_types"]]
        self.assertTrue(route_weights)
        self.assertTrue(all(value == 1.5 for value in route_weights))
        with self.assertRaisesRegex(ValueError, "crosses the prompt/target"):
            live._token_loss_mask(
                [(target_start - 1, target_start + 1)],
                target_start=target_start, target=target,
                token_ids=[1], mask_info=mask)


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
        sys.executable, "-I", str(ROOT / "python/umminfer_joint_sft_live.py"),
        "--dataset-dir", str(dataset_dir or Path(args.dataset_dir)),
        "--config", str(ROOT / "configs/training/joint_sft_live_gate.json"),
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


def run_live_gate(args):
    evidence = Path(args.evidence_dir)
    if evidence.exists() and any(evidence.iterdir()):
        raise ValueError("live evidence directory must be absent or empty")
    evidence.mkdir(parents=True, exist_ok=True)
    fixture = build_fixture_pack(evidence / "fixture")
    args.dataset_dir = str(fixture["pack_dir"])

    uninterrupted_proc, uninterrupted_dir = _run_adapter(args, "uninterrupted")
    if uninterrupted_proc.returncode:
        raise RuntimeError("uninterrupted live run failed; see captured stderr")
    uninterrupted = _load_manifest(uninterrupted_dir)

    # Keep the same two-step run identity but stop after one real step, then
    # resume its independently written checkpoint in a fresh Python process.
    first_proc, first_dir = _run_adapter(args, "split-first", stop_after=1)
    if first_proc.returncode:
        raise RuntimeError("first split run failed; see captured stderr")
    first = _load_manifest(first_dir)
    checkpoint = first_dir / "checkpoints/step-000001.pt"

    resumed_proc, resumed_dir = _run_adapter(
        args, "split-resumed", resume=checkpoint)
    if resumed_proc.returncode:
        raise RuntimeError("resumed live run failed; see captured stderr")
    resumed = _load_manifest(resumed_dir)

    first_step = uninterrupted["steps"][0]
    resumed_second = resumed["steps"][0]
    uninterrupted_second = uninterrupted["steps"][1]
    exact_fields = (
        "loss", "grad_norm", "record_ids", "record_order_sha256",
        "parameter_sha256_before", "parameter_sha256_after",
        "optimizer_state_sha256", "active_loss_weight",
        "changed_parameter_tensors", "changed_parameters",
        "max_abs_parameter_delta", "target_audit",
    )
    initial_mismatches = {
        key: {"uninterrupted": first_step[key], "split_first": first["steps"][0][key]}
        for key in exact_fields
        if first_step[key] != first["steps"][0][key]
    }
    mismatches = {
        key: {"uninterrupted": uninterrupted_second[key],
              "resumed": resumed_second[key]}
        for key in exact_fields
        if uninterrupted_second[key] != resumed_second[key]
    }
    equivalence = not initial_mismatches and not mismatches \
        and uninterrupted["final_parameter_sha256"] == resumed["final_parameter_sha256"] \
        and uninterrupted["final_optimizer_state_sha256"] == resumed["final_optimizer_state_sha256"]

    invalid_pack = evidence / "invalid-pack"
    shutil.copytree(fixture["pack_dir"], invalid_pack)
    _resign_pack_row(invalid_pack,
                     lambda row: row.update(assistant_target=row["assistant_target"] + "bad"))
    invalid_proc, invalid_dir = _run_adapter(
        args, "invalid-input", dataset_dir=invalid_pack)
    invalid_rejected = invalid_proc.returncode != 0 \
        and not (invalid_dir / "live_manifest.json").exists() \
        and not (invalid_dir / "checkpoints").exists()
    if not invalid_rejected:
        raise RuntimeError("invalid-input process did not fail before optimizer output")

    report = {
        "schema": "umminfer-joint-sft-live-acceptance-v1",
        "status": "PASS" if equivalence else "FAIL",
        "source_commit": args.source_commit,
        "source_test_sha256": stage_c.sha256_file(Path(__file__).resolve()),
        "fixture": {
            "schema_version": 2,
            "trajectory_sha256": fixture["trajectory_sha256"],
            "dataset_manifest_sha256": stage_c.sha256_file(
                fixture["pack_dir"] / "manifest.json"),
            "dataset_builder_sha256": stage_c.sha256_file(Path(ds.__file__).resolve()),
            "seed": "joint-sft-live-fixture-v1",
            "records": ["text-only", "image+video-with-completed-terminal-legs"],
        },
        "real_optimizer": {
            "status": "EXECUTED",
            "uninterrupted": uninterrupted,
            "split_first": first,
            "resumed": resumed,
        },
        "checkpoint_resume_equivalence": {
            "status": "PASS" if equivalence else "FAIL",
            "comparison_fields": list(exact_fields),
            "initial_step_mismatches": initial_mismatches,
            "mismatches": mismatches,
            "final_parameter_sha256_equal": (
                uninterrupted["final_parameter_sha256"] == resumed["final_parameter_sha256"]),
            "final_optimizer_state_sha256_equal": (
                uninterrupted["final_optimizer_state_sha256"]
                == resumed["final_optimizer_state_sha256"]),
        },
        "invalid_input": {
            "status": "PASS" if invalid_rejected else "FAIL",
            "returncode": invalid_proc.returncode,
            "live_manifest_created": (invalid_dir / "live_manifest.json").exists(),
            "checkpoint_created": (invalid_dir / "checkpoints").exists(),
            "optimizer_executed": False,
        },
        "quality": "NOT_RUN",
        "generalized_improvement": "NOT_RUN",
        "DPO": "NOT_RUN",
        "GRPO": "NOT_RUN",
        "upstream_revision_verification": "NOT_RUN",
    }
    report_path = evidence / "acceptance.json"
    report_path.write_bytes(stage_c.canonical_json(report) + b"\n")
    Path(str(report_path) + ".sha256").write_text(
        stage_c.sha256_file(report_path) + "\n", encoding="ascii")
    if not equivalence:
        raise RuntimeError("checkpoint/resume equivalence failed; see acceptance.json")
    print(json.dumps({
        "status": report["status"],
        "optimizer_steps": uninterrupted["optimizer_step_count"],
        "losses": [step["loss"] for step in uninterrupted["steps"]],
        "final_parameter_sha256": uninterrupted["final_parameter_sha256"],
        "checkpoint_resume_equivalence": "PASS",
        "invalid_input": "PASS",
        "DPO": "NOT_RUN", "GRPO": "NOT_RUN", "quality": "NOT_RUN",
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
    print("REAL JOINT SFT OPTIMIZER: NOT_RUN (default CPU preflight suite)")
    print("MODEL QUALITY / DPO / GRPO: NOT_RUN")
    raise SystemExit(0 if result.result.wasSuccessful() else 1)
