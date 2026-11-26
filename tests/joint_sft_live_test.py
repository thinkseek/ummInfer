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
import copy
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
import token_export_test as token_fixture  # noqa: E402
import umminfer_joint_sft_live as live  # noqa: E402
import umminfer_joint_trainer as stage_c  # noqa: E402
import umminfer_rl_trajectory as traj  # noqa: E402
import umminfer_sft_dataset as ds  # noqa: E402
import umminfer_token_export as tokens  # noqa: E402


def build_fixture_pack(root, *, export_version=None, rich=False,
                       real_tokenizer=None):
    """Build deterministic schema-v2 turns, optionally with additive exports."""
    root = Path(root)
    export_dir = root / "export"
    export_dir.mkdir(parents=True)
    recorder = traj.TrajectoryRecorder(export_dir / "trajectories.jsonl")
    if rich:
        fixture_helpers._make_v2_record(
            recorder,
            prompt_text="用一句中文和 English explain gravity 🌅.",
            clean_parts=[
                "<spoken>黎明🌅 é 字。</spoken> "
                "Gravity 引力 pulls masses; café é 🧪.",
            ],
            slots=(), session_id="live-gate-text", request_id="live-text-0")
        _, linked = token_fixture._record(
            root / "linked", token_fixture.fixture.linked_events())
        trajectory_paths = [recorder.jsonl_path, root / "linked/trajectory.jsonl"]
    else:
        fixture_helpers._make_v2_record(
            recorder,
            prompt_text="Give a one-sentence explanation of gravity.",
            clean_parts=["Gravity pulls masses together."],
            slots=(), session_id="live-gate-text", request_id="live-text-0")
        clean_parts = ["A lantern glows at dawn. ", "It rises slowly."]
        fixture_helpers._make_v2_record(
            recorder,
            prompt_text="Explain a lantern with static and moving illustrations.",
            clean_parts=clean_parts,
            slots=[
                ("image", "a red paper lantern glowing at dawn", 0),
                ("video", "the lantern rising slowly into the dawn sky",
                 len(clean_parts[0].encode("utf-8"))),
            ], session_id="live-gate-visual", request_id="live-visual-0")
        trajectory_paths = [recorder.jsonl_path]
        linked = None

    identity = None
    if export_version is not None:
        exported_paths = []
        if real_tokenizer is not None:
            tokenizer, tokenizer_dir, tokenizer_id, revision, version = real_tokenizer
        elif export_version == 2:
            tokenizer = token_fixture.V2FixtureTokenizer()
            tokenizer_dir = token_fixture._v2_tokenizer_dir(root / "fixture-tokenizer")
            tokenizer_id, revision, version = (
                "fixture-bytelevel-nfc", "fixture-revision-2", "fixture-tokenization-v2")
        else:
            tokenizer = token_fixture.fixture_tokenizer()
            tokenizer_dir = token_fixture._tokenizer_dir(root / "fixture-tokenizer")
            tokenizer_id, revision, version = (
                "fixture-unicode-bpe", "fixture-revision-1", "fixture-tokenization-v1")
        identity = {
            "tokenizer": tokenizer, "tokenizer_dir": tokenizer_dir,
            "tokenizer_id": tokenizer_id, "tokenizer_revision": revision,
            "tokenization_version": version,
        }
        for source_path in trajectory_paths:
            source_path = Path(source_path)
            records = [json.loads(line) for line in source_path.read_text(
                encoding="utf-8").splitlines() if line.strip()]
            exported = [tokens.attach_token_export(
                record, tokenizer, tokenizer_dir=tokenizer_dir,
                tokenizer_id=tokenizer_id, tokenizer_revision=revision,
                tokenization_version=version,
                sft_target=ds.reconstruct_director_target(record)[0],
                format_version=export_version) for record in records]
            output = source_path.with_name(source_path.stem + ".tokens.jsonl")
            output.write_bytes(b"".join(
                stage_c.canonical_json(record) + b"\n" for record in exported))
            exported_paths.append(output)
        trajectory_paths = exported_paths

    pack_dir = root / "pack"
    manifest = ds.build_dataset(
        trajectory_paths, pack_dir, mode="sft", seed="joint-sft-live-fixture-v1",
        eval_ratio=0.0, strict_artifacts=True)
    return {
        "trajectory_path": trajectory_paths[0],
        "trajectory_paths": trajectory_paths,
        "trajectory_sha256": stage_c.sha256_file(trajectory_paths[0]),
        "pack_dir": pack_dir,
        "manifest": manifest,
        "export_identity": identity,
        "linked_record": linked,
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


    def test_legacy_pack_bytes_and_v1_v2_export_packing_equivalence(self):
        with tempfile.TemporaryDirectory(prefix="umm-joint-legacy-bytes-") as tmp:
            fixture = build_fixture_pack(Path(tmp))
            self.assertEqual(
                stage_c.sha256_file(fixture["trajectory_path"]),
                "f6e2496a2f62b5cfe13288855d10fa2c4bb19b571e921346749f3eb5ba9496fc")
            self.assertEqual(
                stage_c.sha256_file(fixture["pack_dir"] / "sft_train.jsonl"),
                "9bf15b0fb13d25c1d7a94084174d759e58351dddcf327514d969ca63d227f24e")
            rows = [json.loads(line) for line in (
                fixture["pack_dir"] / "sft_train.jsonl").read_text(
                    encoding="utf-8").splitlines()]
            self.assertTrue(all("token_export" not in row for row in rows))

        for version in (1, 2):
            with self.subTest(export_version=version), tempfile.TemporaryDirectory(
                    prefix=f"umm-joint-export-v{version}-") as tmp:
                fixture = build_fixture_pack(
                    Path(tmp), export_version=version, rich=True)
                rows = [json.loads(line) for line in (
                    fixture["pack_dir"] / "sft_train.jsonl").read_text(
                        encoding="utf-8").splitlines()]
                self.assertEqual(len(rows), 2)
                self.assertTrue(all(row.get("token_export") for row in rows))
                identity = fixture["export_identity"]
                result = live._pack_export_rows(
                    rows, identity["tokenizer"],
                    tokenizer_dir=identity["tokenizer_dir"],
                    tokenizer_id=identity["tokenizer_id"],
                    tokenizer_revision=identity["tokenizer_revision"],
                    tokenization_version=identity["tokenization_version"],
                    span_loss_weights=stage_c.DEFAULT_SPAN_WEIGHTS, max_seq_len=256)
                self.assertEqual(len(result["packed"]), 2)
                self.assertTrue(all(item["legacy_equality"] == "PASS"
                                    for item in result["records"]))
                annotations = [annotation for item in result["packed"]
                               for annotation in item["annotations"]]
                self.assertTrue(any("spoken_style" in item["span_types"]
                                    for item in annotations))
                self.assertTrue(any("visual_route_tag" in item["span_types"]
                                    for item in annotations))
                self.assertTrue(any(item["includes_terminal_delimiter"]
                                    for item in annotations))
                self.assertTrue(any("source=slot-0" in row["assistant_target"]
                                    for row in rows))
                for item in result["packed"]:
                    legacy = live._legacy_pack_row(
                        item["row"], identity["tokenizer"],
                        span_loss_weights=stage_c.DEFAULT_SPAN_WEIGHTS,
                        max_seq_len=256)
                    for key in ("input_ids", "labels", "weights", "annotations", "audit"):
                        self.assertEqual(
                            stage_c.canonical_json(item[key]),
                            stage_c.canonical_json(legacy[key]))

    def test_partial_stale_and_malformed_exports_fail_before_output_or_ml(self):
        mutations = {
            "partial": lambda row: (row.pop("token_export"),
                                    row.pop("token_export_source")),
            "unsupported": lambda row: row["token_export"].update(
                format="umminfer-unsupported", format_version=3),
            "stale-source": lambda row: row["token_export_source"].update(
                clean_text=row["token_export_source"]["clean_text"] + "stale"),
            "token-id": lambda row: row["token_export"]["streams"]["sft_target"]
                ["token_ids"].__setitem__(0, 1),
            "raw-offset": lambda row: row["token_export"]["streams"]["sft_target"]
                ["tokens"][0].__setitem__("offset_char_end", 999),
            "weight": lambda row: row["token_export"]["streams"]["sft_target"]
                ["sft_mask"]["token_weights"].__setitem__(0, 0.0),
            "annotation": lambda row: row["token_export"]["streams"]["sft_target"]
                ["sft_mask"].__setitem__("annotations_sha256", "0" * 64),
        }
        for name, mutate in mutations.items():
            with self.subTest(mutation=name), tempfile.TemporaryDirectory(
                    prefix="umm-joint-export-invalid-") as tmp:
                root = Path(tmp)
                fixture = build_fixture_pack(root, export_version=2, rich=True)
                _resign_pack_row(fixture["pack_dir"], mutate)
                real_import = builtins.__import__

                def no_ml(import_name, *args, **kwargs):
                    if import_name == "torch" or import_name.startswith("transformers"):
                        raise AssertionError(f"ML import after invalid export: {import_name}")
                    return real_import(import_name, *args, **kwargs)

                with mock.patch("builtins.__import__", side_effect=no_ml), \
                        mock.patch.object(live, "_run_export_preflight",
                                          side_effect=AssertionError("preflight executed")):
                    with self.assertRaises(ValueError):
                        live.prepare_live_run(
                            dataset_dir=fixture["pack_dir"], output_dir=root / "bad-run",
                            config_path=ROOT / "configs/training/joint_sft_live_gate.json",
                            model_path=_fake_model_dir(root / "model"),
                            source_commit="0" * 40, max_steps=2,
                            token_source="token_export")
                self.assertFalse((root / "bad-run/plan").exists())
                self.assertFalse((root / "bad-run/live_manifest.json").exists())
                self.assertFalse((root / "bad-run/checkpoints").exists())

    def test_export_path_requires_exact_tokenizer_and_mixed_legacy_opt_in_is_explicit(self):
        with tempfile.TemporaryDirectory(prefix="umm-joint-export-identity-") as tmp:
            root = Path(tmp)
            fixture = build_fixture_pack(root, export_version=2, rich=True)
            rows = [json.loads(line) for line in (
                fixture["pack_dir"] / "sft_train.jsonl").read_text(
                    encoding="utf-8").splitlines()]
            identity = fixture["export_identity"]
            for field, wrong in (("tokenizer_id", "wrong"),
                                 ("tokenizer_revision", "wrong"),
                                 ("tokenization_version", "wrong")):
                options = dict(identity)
                options[field] = wrong
                with self.subTest(field=field), self.assertRaises(ValueError):
                    live._pack_export_rows(
                        rows, options["tokenizer"],
                        tokenizer_dir=options["tokenizer_dir"],
                        tokenizer_id=options["tokenizer_id"],
                        tokenizer_revision=options["tokenizer_revision"],
                        tokenization_version=options["tokenization_version"],
                        span_loss_weights=stage_c.DEFAULT_SPAN_WEIGHTS,
                        max_seq_len=256)
            prepared = live.prepare_live_run(
                dataset_dir=fixture["pack_dir"], output_dir=root / "legacy-run",
                config_path=ROOT / "configs/training/joint_sft_live_gate.json",
                model_path=_fake_model_dir(root / "model"),
                source_commit="0" * 40, max_steps=2, token_source="legacy")
            self.assertEqual(prepared["token_source"], "legacy")
            self.assertIsNone(prepared["export_packed"])
            self.assertNotIn("token_source", prepared["identity"])


def _load_manifest(run_dir):
    path = Path(run_dir) / "live_manifest.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    sidecar = Path(str(path) + ".sha256").read_text(encoding="ascii").strip()
    if sidecar != hashlib.sha256(path.read_bytes()).hexdigest():
        raise RuntimeError("live manifest SHA-256 mismatch")
    return result


def _run_adapter(args, run_name, *, resume=None, dataset_dir=None, stop_after=None,
                 token_source="legacy"):
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
        "--token-source", token_source,
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

    import tokenizers
    import transformers
    tokenizer = tokens._load_real_tokenizer(args.model_path)
    version = (f"transformers-{transformers.__version__}_"
               f"tokenizers-{tokenizers.__version__}")
    fixture = build_fixture_pack(
        evidence / "fixture", export_version=2, rich=True,
        real_tokenizer=(
            tokenizer, args.model_path, "Qwen/Qwen2.5-0.5B-Instruct",
            "UNKNOWN", version))
    args.dataset_dir = str(fixture["pack_dir"])

    legacy_proc, legacy_dir = _run_adapter(args, "legacy-uninterrupted")
    if legacy_proc.returncode:
        raise RuntimeError("legacy live run failed; see captured stderr")
    legacy = _load_manifest(legacy_dir)

    export_proc, export_dir = _run_adapter(
        args, "export-uninterrupted", token_source="token_export")
    if export_proc.returncode:
        raise RuntimeError("export-driven live run failed; see captured stderr")
    exported = _load_manifest(export_dir)

    first_proc, first_dir = _run_adapter(
        args, "export-split-first", stop_after=1, token_source="token_export")
    if first_proc.returncode:
        raise RuntimeError("export split-first run failed; see captured stderr")
    first = _load_manifest(first_dir)
    checkpoint = first_dir / "checkpoints/step-000001.pt"

    resumed_proc, resumed_dir = _run_adapter(
        args, "export-split-resumed", resume=checkpoint, token_source="token_export")
    if resumed_proc.returncode:
        raise RuntimeError("export resumed run failed; see captured stderr")
    resumed = _load_manifest(resumed_dir)

    exact_fields = (
        "loss", "grad_norm", "gradient_sha256", "parameter_update_sha256",
        "record_ids", "record_order_sha256", "parameter_sha256_before",
        "parameter_sha256_after", "optimizer_state_sha256", "active_loss_weight",
        "changed_parameter_tensors", "changed_parameters", "max_abs_parameter_delta",
        "parameter_delta_l2", "target_audit",
    )
    path_mismatches = []
    for index, (legacy_step, export_step) in enumerate(
            zip(legacy["steps"], exported["steps"]), 1):
        mismatches = {
            key: {"legacy": legacy_step[key], "token_export": export_step[key]}
            for key in exact_fields if legacy_step[key] != export_step[key]
        }
        if mismatches:
            path_mismatches.append({"step": index, "fields": mismatches})
    path_equivalence = not path_mismatches \
        and len(legacy["steps"]) == len(exported["steps"]) == 2 \
        and legacy["final_parameter_sha256"] == exported["final_parameter_sha256"] \
        and legacy["final_optimizer_state_sha256"] \
            == exported["final_optimizer_state_sha256"] \
        and all(record["legacy_equality"] == "PASS"
                for record in exported["token_source"]["records"])

    initial_mismatches = {
        key: {"uninterrupted": exported["steps"][0][key],
              "split_first": first["steps"][0][key]}
        for key in exact_fields
        if exported["steps"][0][key] != first["steps"][0][key]
    }
    resume_mismatches = {
        key: {"uninterrupted": exported["steps"][1][key],
              "resumed": resumed["steps"][0][key]}
        for key in exact_fields
        if exported["steps"][1][key] != resumed["steps"][0][key]
    }
    checkpoint_bytes_equal = (
        exported["checkpoints"][0]["sha256"] == first["checkpoints"][0]["sha256"]
        and exported["checkpoints"][1]["sha256"] == resumed["checkpoints"][0]["sha256"])
    resume_equivalence = not initial_mismatches and not resume_mismatches \
        and checkpoint_bytes_equal \
        and exported["final_parameter_sha256"] == resumed["final_parameter_sha256"] \
        and exported["final_optimizer_state_sha256"] \
            == resumed["final_optimizer_state_sha256"]

    invalid_results = {}
    invalid_mutations = {
        "target-binding": lambda row: row.update(
            assistant_target=row["assistant_target"] + "bad"),
        "partial-export": lambda row: (row.pop("token_export"),
                                       row.pop("token_export_source")),
    }
    for name, mutate in invalid_mutations.items():
        invalid_pack = evidence / f"invalid-{name}-pack"
        shutil.copytree(fixture["pack_dir"], invalid_pack)
        _resign_pack_row(invalid_pack, mutate)
        proc, run_dir = _run_adapter(
            args, f"invalid-{name}", dataset_dir=invalid_pack,
            token_source="token_export")
        rejected = proc.returncode != 0 \
            and not (run_dir / "plan").exists() \
            and not (run_dir / "live_manifest.json").exists() \
            and not (run_dir / "checkpoints").exists() \
            and not (run_dir / "token_export_preflight.json").exists()
        invalid_results[name] = {
            "status": "PASS" if rejected else "FAIL",
            "returncode": proc.returncode,
            "plan_created": (run_dir / "plan").exists(),
            "live_manifest_created": (run_dir / "live_manifest.json").exists(),
            "checkpoint_created": (run_dir / "checkpoints").exists(),
            "preflight_output_created": (run_dir / "token_export_preflight.json").exists(),
            "torch_imported": False,
            "optimizer_executed": False,
        }
    invalid_rejected = all(item["status"] == "PASS"
                           for item in invalid_results.values())
    overall = path_equivalence and resume_equivalence and invalid_rejected

    report = {
        "schema": "umminfer-joint-sft-export-live-acceptance-v1",
        "status": "PASS" if overall else "FAIL",
        "source_commit": args.source_commit,
        "source_test_sha256": stage_c.sha256_file(Path(__file__).resolve()),
        "python_binary_sha256": hashlib.sha256(
            Path(sys.executable).read_bytes()).hexdigest(),
        "fixture": {
            "schema_version": 2,
            "trajectory_files": [{
                "bytes": path.stat().st_size,
                "sha256": stage_c.sha256_file(path),
            } for path in fixture["trajectory_paths"]],
            "dataset_manifest_sha256": stage_c.sha256_file(
                fixture["pack_dir"] / "manifest.json"),
            "sft_train_sha256": stage_c.sha256_file(
                fixture["pack_dir"] / "sft_train.jsonl"),
            "dataset_builder_sha256": stage_c.sha256_file(Path(ds.__file__).resolve()),
            "token_export_source_sha256": stage_c.sha256_file(Path(tokens.__file__).resolve()),
            "seed": "joint-sft-live-fixture-v1",
            "records": [
                "multilingual+NFC-combining+emoji-byte-fallback+spoken",
                "image+video+explicit-source-slot-0+completed-terminal-legs",
            ],
            "token_export_format_version": 2,
        },
        "real_optimizer": {
            "status": "EXECUTED",
            "legacy_uninterrupted": legacy,
            "export_uninterrupted": exported,
            "export_split_first": first,
            "export_resumed": resumed,
        },
        "legacy_export_equivalence": {
            "status": "PASS" if path_equivalence else "FAIL",
            "comparison_fields": list(exact_fields),
            "mismatches": path_mismatches,
            "cross_path_checkpoint_bytes_expected_equal": False,
            "reason": "token_source is deliberately bound into each checkpoint run identity",
        },
        "checkpoint_resume_equivalence": {
            "status": "PASS" if resume_equivalence else "FAIL",
            "comparison_fields": list(exact_fields),
            "initial_step_mismatches": initial_mismatches,
            "mismatches": resume_mismatches,
            "checkpoint_bytes_equal": checkpoint_bytes_equal,
            "final_parameter_sha256_equal": (
                exported["final_parameter_sha256"] == resumed["final_parameter_sha256"]),
            "final_optimizer_state_sha256_equal": (
                exported["final_optimizer_state_sha256"]
                == resumed["final_optimizer_state_sha256"]),
        },
        "invalid_input": {
            "status": "PASS" if invalid_rejected else "FAIL",
            "cases": invalid_results,
        },
        "sampled_online_behavior_ids": "NOT_CLAIMED",
        "quality": "NOT_RUN",
        "generalized_improvement": "NOT_RUN",
        "DPO": "NOT_RUN",
        "GRPO": "NOT_RUN",
        "upstream_revision_verification": "NOT_RUN",
    }
    report_path = evidence / "acceptance.json"
    report_path.write_bytes(stage_c.canonical_json(report) + b"\n")
    Path(str(report_path) + ".sha256").write_text(
        hashlib.sha256(report_path.read_bytes()).hexdigest() + "\n", encoding="ascii")
    checksum_lines = []
    for path in sorted(evidence.rglob("*")):
        if not path.is_file() or path.suffix == ".pt" or path.name == "SHA256SUMS":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksum_lines.append(f"{digest}  {path.relative_to(evidence).as_posix()}\n")
    (evidence / "SHA256SUMS").write_text("".join(checksum_lines), encoding="ascii")
    if not overall:
        raise RuntimeError("live equality/resume/invalid-input gate failed; see acceptance.json")
    print(json.dumps({
        "status": report["status"],
        "optimizer_steps_per_path": exported["optimizer_step_count"],
        "losses": [step["loss"] for step in exported["steps"]],
        "final_parameter_sha256": exported["final_parameter_sha256"],
        "legacy_export_equivalence": "PASS",
        "checkpoint_resume_equivalence": "PASS",
        "invalid_input": "PASS",
        "sampled_online_behavior_ids": "NOT_CLAIMED",
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
