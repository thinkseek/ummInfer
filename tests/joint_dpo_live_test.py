#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU preflight tests plus the opt-in real SM90 joint-DPO acceptance gate.

Default: python3 -I tests/joint_dpo_live_test.py
Live:    CUDA_VISIBLE_DEVICES=<idle-gpu> python3 -I tests/joint_dpo_live_test.py \
             --live --model-path DIR --evidence-dir DIR --source-commit COMMIT

The caller selects one physical GPU.  The adapter uses visible logical cuda:0
and never starts or contacts a serving process.
"""

import argparse
import builtins
import copy
import hashlib
import io
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
import token_export_test as token_fixture  # noqa: E402
import umminfer_joint_dpo_live as live  # noqa: E402
import umminfer_joint_trainer as stage_c  # noqa: E402
import umminfer_rl_trajectory as traj  # noqa: E402
import umminfer_sft_dataset as ds  # noqa: E402
import umminfer_token_export as tokens  # noqa: E402

FIXTURE_SEED = "joint-dpo-live-fixture-v1"


def build_fixture_pack(root, *, export_version=None, rich=False):
    """Build two schema-v2 preference pairs with structured targets on both sides."""
    root = Path(root)
    export_dir = root / "export"
    export_dir.mkdir(parents=True)
    recorder = traj.TrajectoryRecorder(export_dir / "trajectories.jsonl")
    cases = [
        (
            "Explain a lantern with static and moving illustrations.",
            ["A lantern glows at dawn. ", "It rises slowly."],
            [
                ("image", "a red paper lantern glowing at dawn", 0),
                ("video", "the lantern rising slowly into the dawn sky",
                 len("A lantern glows at dawn. ".encode("utf-8"))),
            ],
            ["A lantern glows at dawn. ", "The paper frame is visible."],
            [("image", "a detailed red paper lantern at dawn", 0)],
        ),
        (
            "Show an accretion disk and its dynamic plasma motion.",
            ["Superheated plasma forms a bright disk. ", "Matter spirals inward."],
            [
                ("image", "labelled black hole accretion disk cross section", 0),
                ("video", "plasma spiraling around a black hole",
                 len("Superheated plasma forms a bright disk. ".encode("utf-8"))),
            ],
            ["Superheated plasma forms a bright disk. ", "A static sketch labels it."],
            [("image", "static labelled accretion disk sketch", 0)],
        ),
    ]
    if rich:
        cases = [
            (
                "用中文和 English explain a lantern 🌅 with café physics.",
                ["<spoken>黎明🌅 café 灯笼。</spoken> ",
                 "Gravity 引力 makes it rise 🧪."],
                [
                    ("image", "红灯笼 🏮 glowing at dawn", 0),
                    ("video", "灯笼随 🌅 rises slowly",
                     len("<spoken>黎明🌅 café 灯笼。</spoken> ".encode("utf-8"))),
                ],
                ["<spoken>黎明🌅 café 灯笼。</spoken> Static 静图 🧪."],
                [("image", "静态红灯笼 🏮 at dawn", 0)],
            ),
            (
                "Présentez un disque d’accrétion 黑洞 with spoken 🎙️ detail.",
                ["<spoken>Le plasma café 黑洞 🎙️.</spoken> ",
                 "Matter spirals inward 🌌."],
                [
                    ("image", "disque 黑洞 café cross section", 0),
                    ("video", "plasma 🌌 spiraling 黑洞",
                     len("<spoken>Le plasma café 黑洞 🎙️.</spoken> ".encode("utf-8"))),
                ],
                ["<spoken>Le plasma café 黑洞 🎙️.</spoken> Static sketch."],
                [("image", "static disque 黑洞 sketch", 0)],
            ),
        ]
    for index, (prompt, chosen_parts, chosen_slots,
                rejected_parts, rejected_slots) in enumerate(cases):
        fixture_helpers._make_v2_record(
            recorder, prompt_text=prompt, clean_parts=chosen_parts,
            slots=chosen_slots, session_id=f"dpo-live-{index}-chosen",
            request_id=f"dpo-live-{index}-chosen-request",
            expected_modalities=["image", "video"], explicit_anchor=rich)
        fixture_helpers._make_v2_record(
            recorder, prompt_text=prompt, clean_parts=rejected_parts,
            slots=rejected_slots, session_id=f"dpo-live-{index}-rejected",
            request_id=f"dpo-live-{index}-rejected-request",
            expected_modalities=["image", "video"])

    identity = None
    trajectory_path = recorder.jsonl_path
    if export_version is not None:
        if export_version == 2:
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
        records = [json.loads(line) for line in trajectory_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        exported = [tokens.attach_token_export(
            record, tokenizer, tokenizer_dir=tokenizer_dir,
            tokenizer_id=tokenizer_id, tokenizer_revision=revision,
            tokenization_version=version,
            sft_target=ds.reconstruct_director_target(record)[0],
            format_version=export_version) for record in records]
        trajectory_path = trajectory_path.with_name(
            trajectory_path.stem + ".tokens.jsonl")
        trajectory_path.write_bytes(b"".join(
            stage_c.canonical_json(record) + b"\n" for record in exported))

    pack_dir = root / "pack"
    manifest = ds.build_dataset(
        [trajectory_path], pack_dir, mode="preference",
        seed=FIXTURE_SEED, eval_ratio=0.0, min_margin=0.1,
        min_group_size=2, strict_artifacts=True)
    if manifest["outputs"]["dpo_train"]["record_count"] != 2:
        raise RuntimeError("deterministic DPO fixture did not produce exactly two pairs")
    return {
        "trajectory_path": trajectory_path,
        "trajectory_sha256": stage_c.sha256_file(trajectory_path),
        "pack_dir": pack_dir,
        "manifest": manifest,
        "export_identity": identity,
    }


def _fake_model_dir(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
    }), encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"preflight-only-weight-bytes")
    return root


def _resign_pack_pair(pack_dir, mutate, *, row_index=0):
    pack_path = Path(pack_dir) / "dpo_train.jsonl"
    rows = [json.loads(line) for line in pack_path.read_text(
        encoding="utf-8").splitlines()]
    mutate(rows, row_index)
    payload = b"".join(stage_c.canonical_json(row) + b"\n" for row in rows)
    pack_path.write_bytes(payload)
    manifest_path = Path(pack_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["dpo_train"].update({
        "bytes": len(payload),
        "sha256": stage_c.sha256_bytes(payload),
        "record_count": len(rows),
    })
    manifest_payload = stage_c.canonical_json(manifest) + b"\n"
    manifest_path.write_bytes(manifest_payload)
    Path(str(manifest_path) + ".sha256").write_text(
        stage_c.sha256_bytes(manifest_payload) + "\n", encoding="ascii")


def _prepare_without_ml(pack_dir, root):
    real_import = builtins.__import__

    def no_ml(name, *args, **kwargs):
        if name == "torch" or name.startswith("transformers"):
            raise AssertionError(f"ML import during preflight: {name}")
        return real_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=no_ml):
        return live.prepare_live_run(
            dataset_dir=pack_dir, output_dir=Path(root) / "run",
            config_path=ROOT / "configs/training/joint_dpo_live_gate.json",
            model_path=_fake_model_dir(Path(root) / "model"),
            source_commit="0" * 40, max_steps=2)


class JointDpoLivePreflightTests(unittest.TestCase):
    def test_fixture_preflight_pair_identity_and_provenance(self):
        with tempfile.TemporaryDirectory(prefix="umm-joint-dpo-preflight-") as tmp:
            root = Path(tmp)
            fixture = build_fixture_pack(root)
            prepared = _prepare_without_ml(fixture["pack_dir"], root)
            self.assertEqual(len(prepared["rows"]), 2)
            self.assertEqual(prepared["plan"]["REAL_TRAINING"], "NOT_RUN")
            self.assertEqual(prepared["plan"]["active_stages"], ["dpo"])
            self.assertEqual(prepared["identity"]["dpo"]["beta"], 0.1)
            self.assertEqual(
                prepared["identity"]["dpo"]["reference_policy"],
                "frozen_initial_base_model")
            for audit in prepared["pair_audits"]:
                self.assertGreater(audit["chosen"]["score"], audit["rejected"]["score"])
                for side in ("chosen", "rejected"):
                    self.assertIn("image", audit[side]["visual_modalities"])
                    self.assertGreater(audit[side]["terminal_legs_verified"], 0)
                    self.assertGreater(audit[side]["verified_cas_refs"], 0)
                    self.assertGreater(
                        audit[side]["span_counts"]["visual_route_tag"], 0)
                    self.assertGreater(
                        audit[side]["span_counts"]["visual_prompt_body"], 0)

    def test_malformed_crossed_and_cas_broken_pairs_fail_before_ml_import(self):
        mutations = [
            lambda rows, i: rows[i].update(
                rejected=copy.deepcopy(rows[1 - i]["chosen"])),
            lambda rows, i: rows[i].update(
                chosen=rows[i]["rejected"], rejected=rows[i]["chosen"]),
            lambda rows, i: rows[i]["rejected"].update(
                assistant_target=rows[i]["rejected"]["assistant_target"] + "bad"),
            lambda rows, i: rows[i]["chosen"]["visual_slots"][0].update(
                leg_state="failed"),
            lambda rows, i: rows[i]["rejected"]["output_media_refs"][0].update(
                verified_on_disk=False),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory(
                    prefix="umm-joint-dpo-invalid-") as tmp:
                root = Path(tmp)
                fixture = build_fixture_pack(root)
                bad_pack = root / "bad-pack"
                shutil.copytree(fixture["pack_dir"], bad_pack)
                _resign_pack_pair(bad_pack, mutate)
                with self.assertRaises(ValueError):
                    _prepare_without_ml(bad_pack, root)
                self.assertFalse((root / "run/live_manifest.json").exists())
                self.assertFalse((root / "run/checkpoints").exists())

    def test_prompt_masks_and_structured_weights_are_independent_for_both_sides(self):
        for target in (
            "Plain [[image: red lantern]] tail [[video: lantern rises]] done",
            "Plain [[image: static disk]] tail",
        ):
            with self.subTest(target=target):
                prompt = "Explain"
                target_start = len(prompt) + 1
                mask = stage_c.build_span_loss_mask(target)
                offsets = [(0, target_start)]
                for span in mask["spans"]:
                    offsets.append((target_start + span["start"],
                                    target_start + span["end"]))
                ids = list(range(len(offsets)))
                labels, weights, annotations = live.sft_live._token_loss_mask(
                    offsets, target_start=target_start, target=target,
                    token_ids=ids, mask_info=mask)
                self.assertEqual(labels[0], -100)
                self.assertEqual(weights[0], 0.0)
                self.assertTrue(all(value > 0 for value in weights[1:]))
                self.assertTrue(any(item["includes_terminal_delimiter"]
                                    for item in annotations))
                self.assertTrue(any("visual_route_tag" in item["span_types"]
                                    for item in annotations))
                self.assertTrue(any("visual_prompt_body" in item["span_types"]
                                    for item in annotations))
                with self.assertRaisesRegex(ValueError, "crosses the prompt/target"):
                    live.sft_live._token_loss_mask(
                        [(target_start - 1, target_start + 1)],
                        target_start=target_start, target=target,
                        token_ids=[1], mask_info=mask)


    def test_legacy_pack_bytes_and_v1_v2_export_pair_equivalence(self):
        with tempfile.TemporaryDirectory(prefix="umm-dpo-legacy-bytes-") as tmp:
            fixture = build_fixture_pack(Path(tmp))
            self.assertEqual(
                stage_c.sha256_file(fixture["pack_dir"] / "dpo_train.jsonl"),
                "2e7e15b5143bdba21cb597371849c35b911c07ba03b647ad8b184491617e4ead")
            rows = [json.loads(line) for line in (
                fixture["pack_dir"] / "dpo_train.jsonl").read_text().splitlines()]
            self.assertTrue(all("token_export" not in row[side]
                                for row in rows for side in ("chosen", "rejected")))

        config = json.loads((ROOT / "configs/training/joint_dpo_live_gate.json")
                            .read_text(encoding="utf-8"))
        for version in (1, 2):
            with self.subTest(export_version=version), tempfile.TemporaryDirectory(
                    prefix=f"umm-dpo-export-v{version}-") as tmp:
                fixture = build_fixture_pack(Path(tmp), export_version=version, rich=True)
                rows = [json.loads(line) for line in (
                    fixture["pack_dir"] / "dpo_train.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), 2)
                self.assertTrue(all(row[side].get("token_export")
                                    for row in rows for side in ("chosen", "rejected")))
                self.assertTrue(any("source=slot-0" in row["chosen"]["assistant_target"]
                                    for row in rows))
                identity = fixture["export_identity"]
                result = live._pack_export_pairs(
                    rows, identity["tokenizer"],
                    tokenizer_dir=identity["tokenizer_dir"],
                    tokenizer_id=identity["tokenizer_id"],
                    tokenizer_revision=identity["tokenizer_revision"],
                    tokenization_version=identity["tokenization_version"],
                    span_loss_weights=config["span_loss_weights"],
                    dpo_config=config["dpo"])
                self.assertEqual(result["status"], "PASS")
                self.assertEqual(result["claims"]["real_dpo_optimizer"], "NOT_RUN")
                annotations = [annotation for pair in result["packed"]
                               for side in ("chosen", "rejected")
                               for annotation in pair[side]["annotations"]]
                self.assertTrue(any("spoken_style" in item["span_types"]
                                    for item in annotations))
                self.assertTrue(any("visual_route_tag" in item["span_types"]
                                    for item in annotations))
                self.assertTrue(any(item["includes_terminal_delimiter"]
                                    for item in annotations))
                for row, pair in zip(rows, result["packed"]):
                    audit = live._audit_pair(row)
                    for side in ("chosen", "rejected"):
                        legacy = live._pack_response(
                            identity["tokenizer"], row, row[side], audit[side],
                            side=side, prepared={"identity": {
                                "dpo": config["dpo"],
                                "span_loss_weights": config["span_loss_weights"],
                            }})
                        for key in ("input_ids", "labels", "weights", "annotations", "audit"):
                            self.assertEqual(
                                stage_c.canonical_json(pair[side][key]),
                                stage_c.canonical_json(legacy[key]))
                        self.assertEqual(pair[side]["token_export"]["legacy_equality"], "PASS")
                grpo_rows = [json.loads(line) for line in (
                    fixture["pack_dir"] / "grpo_train.jsonl").read_text().splitlines()]
                self.assertTrue(all("token_export" not in rollout
                                    for group in grpo_rows for rollout in group["rollouts"]))

    def test_exact_dpo_loss_and_independent_reference_masks(self):
        pair = {
            "chosen": {
                "input_ids": [5, 11, 12, 13],
                "labels": [-100, 11, 12, 13],
                "weights": [0.0, 1.0, 1.5, 0.5],
            },
            "rejected": {
                "input_ids": [5, 21, 22, 23],
                "labels": [-100, 21, -100, 23],
                "weights": [0.0, 2.0, 0.0, 1.0],
            },
        }
        result = live._stdlib_dpo_fixture(
            pair,
            policy_logps={
                "chosen": [0.0, -0.5, -0.25, -2.0],
                "rejected": [0.0, -4.0, -100.0, -1.0],
            },
            reference_logps={
                "chosen": [0.0, -0.25, -0.5, -0.75],
                "rejected": [0.0, -1.0, -100.0, -2.0],
            }, beta=0.1)
        self.assertEqual(result["policy"]["chosen"]["weighted_logp_sum"], -1.875)
        self.assertEqual(result["policy"]["chosen"]["mean_logp"], -0.625)
        self.assertEqual(result["policy"]["rejected"]["mean_logp"], -3.0)
        self.assertEqual(result["reference"]["chosen"]["active_loss_weight"], 3.0)
        self.assertEqual(result["reference"]["rejected"]["active_loss_weight"], 3.0)
        self.assertAlmostEqual(result["scaled_margin"], 0.15, places=15)
        self.assertEqual(result["loss"], 0.6209570477895321)
        self.assertEqual(result["real_optimizer"], "NOT_RUN")
        broken = copy.deepcopy(pair)
        broken["rejected"]["weights"][2] = 1.0
        with self.assertRaisesRegex(ValueError, "masked DPO reference token"):
            live._stdlib_sequence_logprob(
                broken["rejected"], [0.0, -1.0, -2.0, -3.0])

    def test_invalid_exports_fail_before_output_planner_or_ml(self):
        mutations = {
            "missing-chosen": lambda rows, i: (
                rows[i]["chosen"].pop("token_export"),
                rows[i]["chosen"].pop("token_export_source")),
            "partial-rejected": lambda rows, i: rows[i]["rejected"].pop(
                "token_export_source"),
            "stale-source": lambda rows, i: rows[i]["chosen"][
                "token_export_source"].update(clean_text="stale"),
            "malformed-token": lambda rows, i: rows[i]["rejected"]["token_export"][
                "streams"]["sft_target"]["token_ids"].__setitem__(0, 1),
            "cross-pair": lambda rows, i: rows[i].update(
                chosen=copy.deepcopy(rows[1 - i]["chosen"])),
            "prompt-mismatch": lambda rows, i: rows[i].update(
                prompt_text=rows[i]["prompt_text"] + " mismatched"),
            "cross-side-export": lambda rows, i: rows[i]["chosen"].update(
                token_export=copy.deepcopy(rows[i]["rejected"]["token_export"])),
            "duplicate-pair": lambda rows, i: rows.append(copy.deepcopy(rows[i])),
            "unsupported": lambda rows, i: rows[i]["chosen"]["token_export"].update(
                format="umminfer-unsupported", format_version=3),
        }
        for name, mutate in mutations.items():
            with self.subTest(mutation=name), tempfile.TemporaryDirectory(
                    prefix="umm-dpo-export-invalid-") as tmp:
                root = Path(tmp)
                fixture = build_fixture_pack(root, export_version=2, rich=True)
                _resign_pack_pair(fixture["pack_dir"], mutate)
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
                            config_path=ROOT / "configs/training/joint_dpo_live_gate.json",
                            model_path=_fake_model_dir(root / "model"),
                            source_commit="0" * 40, max_steps=2,
                            token_source="token_export")
                self.assertFalse((root / "bad-run/plan").exists())
                self.assertFalse((root / "bad-run/token_export_preflight.json").exists())
                self.assertFalse((root / "bad-run/live_manifest.json").exists())
                self.assertFalse((root / "bad-run/checkpoints").exists())

    def test_export_tokenizer_identity_and_mixed_batches_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix="umm-dpo-export-identity-") as tmp:
            fixture = build_fixture_pack(Path(tmp), export_version=2, rich=True)
            rows = [json.loads(line) for line in (
                fixture["pack_dir"] / "dpo_train.jsonl").read_text().splitlines()]
            identity = fixture["export_identity"]
            config = json.loads((ROOT / "configs/training/joint_dpo_live_gate.json")
                                .read_text(encoding="utf-8"))
            for field, wrong in (("tokenizer_id", "wrong"),
                                 ("tokenizer_revision", "wrong"),
                                 ("tokenization_version", "wrong")):
                options = dict(identity)
                options[field] = wrong
                with self.subTest(field=field), self.assertRaises(ValueError):
                    live._pack_export_pairs(
                        rows, options["tokenizer"],
                        tokenizer_dir=options["tokenizer_dir"],
                        tokenizer_id=options["tokenizer_id"],
                        tokenizer_revision=options["tokenizer_revision"],
                        tokenization_version=options["tokenization_version"],
                        span_loss_weights=config["span_loss_weights"],
                        dpo_config=config["dpo"])
            changed = copy.deepcopy(rows[0]["chosen"]["token_export"])
            changed["tokenizer"]["id"] = "cross-pair-tokenizer"
            changed["config_sha256"] = tokens._config_digest(
                changed["tokenizer"], tokens.EXPORT_POLICY_V2)
            unsigned = {key: value for key, value in changed.items() if key != "export_sha256"}
            changed["export_sha256"] = stage_c.sha256_bytes(stage_c.canonical_json(unsigned))
            rows[0]["chosen"]["token_export"] = changed
            with self.assertRaisesRegex(ValueError, "mixed DPO token-export tokenizer"):
                live._pack_export_pairs(
                    rows, identity["tokenizer"],
                    tokenizer_dir=identity["tokenizer_dir"],
                    tokenizer_id=identity["tokenizer_id"],
                    tokenizer_revision=identity["tokenizer_revision"],
                    tokenization_version=identity["tokenization_version"],
                    span_loss_weights=config["span_loss_weights"],
                    dpo_config=config["dpo"])

    def test_export_identity_split_resume_and_legacy_isolation(self):
        with tempfile.TemporaryDirectory(prefix="umm-dpo-export-resume-") as tmp:
            root = Path(tmp)
            fixture = build_fixture_pack(root / "rich", export_version=2, rich=True)
            other = build_fixture_pack(root / "other", export_version=2, rich=False)
            config_path = ROOT / "configs/training/joint_dpo_live_gate.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))

            def raw(item):
                rows = [json.loads(line) for line in (
                    item["pack_dir"] / "dpo_train.jsonl").read_text().splitlines()]
                identity = item["export_identity"]
                return live._pack_export_pairs(
                    rows, identity["tokenizer"],
                    tokenizer_dir=identity["tokenizer_dir"],
                    tokenizer_id=identity["tokenizer_id"],
                    tokenizer_revision=identity["tokenizer_revision"],
                    tokenization_version=identity["tokenization_version"],
                    span_loss_weights=config["span_loss_weights"],
                    dpo_config=config["dpo"])

            preflight = raw(fixture)
            with mock.patch.object(live, "_run_export_preflight", return_value=preflight):
                full = live.prepare_live_run(
                    dataset_dir=fixture["pack_dir"], output_dir=root / "full",
                    config_path=config_path, model_path=_fake_model_dir(root / "model"),
                    source_commit="0" * 40, max_steps=2, token_source="token_export")
                split = live.prepare_live_run(
                    dataset_dir=fixture["pack_dir"], output_dir=root / "split",
                    config_path=config_path, model_path=_fake_model_dir(root / "model"),
                    source_commit="0" * 40, max_steps=2, stop_after=1,
                    token_source="token_export")
            legacy = live.prepare_live_run(
                dataset_dir=fixture["pack_dir"], output_dir=root / "legacy",
                config_path=config_path, model_path=_fake_model_dir(root / "model"),
                source_commit="0" * 40, max_steps=2)
            with mock.patch.object(live, "_run_export_preflight", return_value=raw(other)):
                different = live.prepare_live_run(
                    dataset_dir=other["pack_dir"], output_dir=root / "different",
                    config_path=config_path, model_path=_fake_model_dir(root / "model"),
                    source_commit="0" * 40, max_steps=2, token_source="token_export")
            self.assertEqual(full["identity_sha256"], split["identity_sha256"])
            self.assertEqual(full["export_packed"], split["export_packed"])
            self.assertNotEqual(full["identity_sha256"], legacy["identity_sha256"])
            self.assertNotEqual(full["identity_sha256"], different["identity_sha256"])
            self.assertNotIn("token_source", legacy["identity"])
            self.assertIsNone(legacy["export_packed"])
            payload = {
                "format": live.CHECKPOINT_FORMAT,
                "run_identity_sha256": full["identity_sha256"],
                "next_step": 1,
                "schedule_position": 1,
                "reference_policy_inputs_sha256": stage_c.sha256_bytes(
                    stage_c.canonical_json(full["reference_policy_inputs"])),
            }
            self.assertEqual(live._validate_resume_payload(full, payload), 1)
            with self.assertRaises(ValueError):
                live._validate_resume_payload(legacy, payload)
            with self.assertRaises(ValueError):
                live._validate_resume_payload(different, payload)
            payload["reference_policy_inputs_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "reference-policy input"):
                live._validate_resume_payload(full, payload)

    def test_cpu_preflight_only_writes_no_optimizer_output(self):
        with tempfile.TemporaryDirectory(prefix="umm-dpo-cpu-only-") as tmp:
            root = Path(tmp)
            fixture = build_fixture_pack(root, export_version=2, rich=True)
            preflight = _fixture_raw_bundle(fixture)
            output = root / "cpu-run"
            argv = [
                "--dataset-dir", str(fixture["pack_dir"]),
                "--config", str(ROOT / "configs/training/joint_dpo_live_gate.json"),
                "--model-path", str(_fake_model_dir(root / "model")),
                "--output-dir", str(output), "--source-commit", "0" * 40,
                "--max-steps", "2", "--token-source", "token_export",
                "--preflight-only",
            ]
            with mock.patch.object(live, "_run_export_preflight", return_value=preflight), \
                    mock.patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(live.main(argv), 0)
            bundle_path = output / "token_export_preflight.json"
            self.assertTrue(bundle_path.is_file())
            self.assertTrue(Path(str(bundle_path) + ".sha256").is_file())
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertEqual(bundle["claims"]["real_dpo_optimizer"], "NOT_RUN")
            self.assertEqual(bundle["schedule"]["stop_after"], 2)
            self.assertNotIn("torch", sys.modules)
            self.assertNotIn("transformers", sys.modules)
            self.assertFalse((output / "live_manifest.json").exists())
            self.assertFalse((output / "checkpoints").exists())
            bad_output = root / "bad-cli"
            bad_argv = list(argv)
            bad_argv[bad_argv.index("--output-dir") + 1] = str(bad_output)
            bad_argv.remove("--token-source")
            bad_argv.remove("token_export")
            with mock.patch("sys.stderr", new=io.StringIO()), \
                    self.assertRaisesRegex(SystemExit, "2"):
                live.main(bad_argv)
            self.assertFalse((bad_output / "plan").exists())
            self.assertFalse((bad_output / "token_export_preflight.json").exists())

    def test_independent_fresh_process_v1_v2_bundle_verification(self):
        for version in (1, 2):
            with self.subTest(export_version=version), tempfile.TemporaryDirectory(
                    prefix=f"umm-dpo-bundle-v{version}-") as tmp:
                root = Path(tmp)
                write = subprocess.run([
                    sys.executable, "-I", str(Path(__file__).resolve()),
                    "--fixture-bundle-write", str(root), str(version),
                ], capture_output=True, text=True, timeout=120)
                self.assertEqual(write.returncode, 0, write.stderr)
                output = root / "verification.json"
                verify_command = [
                    sys.executable, "-I", str(Path(__file__).resolve()),
                    "--fixture-bundle-verify", str(root), str(version), str(output),
                ]
                verify = subprocess.run(
                    verify_command, capture_output=True, text=True, timeout=120)
                self.assertEqual(verify.returncode, 0, verify.stderr)
                report = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "PASS")
                self.assertEqual(report["fresh_process"], "BYTE_IDENTICAL")
                self.assertEqual(report["real_dpo_optimizer"], "NOT_RUN")
                bundle_path = root / "token_export_preflight.json"
                bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
                bundle["pairs"][0]["chosen"]["labels_sha256"] = "0" * 64
                bundle_path.write_bytes(stage_c.canonical_json(bundle) + b"\n")
                Path(str(bundle_path) + ".sha256").write_text(
                    stage_c.sha256_file(bundle_path) + "\n", encoding="ascii")
                tampered_output = root / "tampered-verification.json"
                verify_command[-1] = str(tampered_output)
                tampered = subprocess.run(
                    verify_command, capture_output=True, text=True, timeout=120)
                self.assertNotEqual(tampered.returncode, 0)
                self.assertFalse(tampered_output.exists())


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
        sys.executable, "-I", str(ROOT / "python/umminfer_joint_dpo_live.py"),
        "--dataset-dir", str(dataset_dir or Path(args.dataset_dir)),
        "--config", str(ROOT / "configs/training/joint_dpo_live_gate.json"),
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


def _reject_before_ml(args, fixture, name, mutate):
    evidence = Path(args.evidence_dir)
    bad_pack = evidence / f"{name}-pack"
    shutil.copytree(fixture["pack_dir"], bad_pack)
    _resign_pack_pair(bad_pack, mutate)
    proc, output = _run_adapter(args, name, dataset_dir=bad_pack)
    rejected = proc.returncode != 0 \
        and not (output / "live_manifest.json").exists() \
        and not (output / "checkpoints").exists() \
        and "DPO_ML_IMPORT_BEGIN" not in proc.stderr
    return {
        "status": "PASS" if rejected else "FAIL",
        "returncode": proc.returncode,
        "ml_import_marker_observed": "DPO_ML_IMPORT_BEGIN" in proc.stderr,
        "live_manifest_created": (output / "live_manifest.json").exists(),
        "checkpoint_created": (output / "checkpoints").exists(),
        "optimizer_executed": False if rejected else "UNKNOWN",
        "stderr_sha256": stage_c.sha256_text(proc.stderr),
    }


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

    exact_fields = (
        "loss", "beta", "grad_norm", "record_ids", "record_order_sha256",
        "parameter_sha256_before", "parameter_sha256_after",
        "optimizer_state_sha256", "reference_parameter_sha256",
        "policy_reference_logprob_evidence", "changed_parameter_tensors",
        "changed_parameters", "max_abs_parameter_delta", "pair_audit",
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
        evidence_block = step["policy_reference_logprob_evidence"]
        values = [step["loss"], step["grad_norm"], step["max_abs_parameter_delta"]]
        for phase in ("pre_update", "post_update"):
            for family in ("policy", "reference"):
                for side in ("chosen", "rejected"):
                    values.append(evidence_block[phase][family][side]["mean_logp"])
        if not all(math.isfinite(value) for value in values):
            mechanics_errors.append(f"step {step['step']}: non-finite metric")
        if step["changed_parameters"] <= 0 or not step["parameters_finite_after"]:
            mechanics_errors.append(f"step {step['step']}: no finite changed parameters")
        for side in ("chosen", "rejected"):
            audit = step["pair_audit"][side]
            if audit["prompt_masked_tokens"] <= 0 \
                    or audit["supervised_target_tokens"] <= 0 \
                    or audit["visual_route_tag_tokens"] <= 0 \
                    or audit["visual_prompt_body_tokens"] <= 0 \
                    or audit["terminal_delimiter_tokens"] <= 0:
                mechanics_errors.append(f"step {step['step']}: invalid {side} target mask")
    initial = first_step["policy_reference_logprob_evidence"]["pre_update"]
    for side in ("chosen", "rejected"):
        if initial["policy"][side]["mean_logp"] != initial["reference"][side]["mean_logp"]:
            mechanics_errors.append(f"initial {side} policy/reference weights differ")
    if not math.isclose(first_step["loss"], math.log(2.0), abs_tol=1e-6):
        mechanics_errors.append("initial sigmoid-DPO loss is not log(2)")
    margin_increases = sum(
        step["policy_reference_logprob_evidence"]["post_update"]["scaled_margin_delta"] > 0.0
        for step in uninterrupted["steps"])
    if margin_increases == 0:
        mechanics_errors.append("no measured DPO scaled-margin increase")

    invalid = _reject_before_ml(
        args, fixture, "invalid-target",
        lambda rows, i: rows[i]["chosen"].update(
            assistant_target=rows[i]["chosen"]["assistant_target"] + "bad"))
    crossed = _reject_before_ml(
        args, fixture, "crossed-pair",
        lambda rows, i: rows[i].update(
            rejected=copy.deepcopy(rows[1 - i]["chosen"])))

    passed = equivalence and not mechanics_errors and margin_increases > 0 \
        and invalid["status"] == "PASS" and crossed["status"] == "PASS"
    report = {
        "schema": "umminfer-joint-dpo-live-acceptance-v1",
        "status": "PASS" if passed else "FAIL",
        "source_commit": args.source_commit,
        "source_test_sha256": stage_c.sha256_file(Path(__file__).resolve()),
        "fixture": {
            "schema_version": 2,
            "trajectory_sha256": fixture["trajectory_sha256"],
            "dataset_manifest_sha256": stage_c.sha256_file(
                fixture["pack_dir"] / "manifest.json"),
            "dataset_builder_sha256": stage_c.sha256_file(Path(ds.__file__).resolve()),
            "seed": FIXTURE_SEED,
            "pair_count": 2,
            "pair_order_sha256": uninterrupted["pair_order_sha256"],
            "sides": "chosen image+video; rejected image-only; clean prose and CAS/terminal bindings on both",
        },
        "real_optimizer": {
            "status": "EXECUTED",
            "uninterrupted": uninterrupted,
            "split_first": first,
            "resumed": resumed,
            "mechanics_errors": mechanics_errors,
            "scaled_margin_increase_steps": margin_increases,
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
                uninterrupted["final_optimizer_state_sha256"]
                == resumed["final_optimizer_state_sha256"]),
        },
        "invalid_input": invalid,
        "crossed_pair": crossed,
        "SFT": "EXISTING_REAL_ACCEPTANCE_PRESERVED",
        "GRPO": "NOT_RUN",
        "model_quality": "NOT_RUN",
        "generalized_improvement": "NOT_RUN",
        "production_model_training": "NOT_RUN",
        "upstream_revision_verification": "NOT_RUN",
    }
    report_path = evidence / "acceptance.json"
    report_path.write_bytes(stage_c.canonical_json(report) + b"\n")
    Path(str(report_path) + ".sha256").write_text(
        stage_c.sha256_file(report_path) + "\n", encoding="ascii")
    if not passed:
        raise RuntimeError("joint DPO acceptance failed; see acceptance.json")
    print(json.dumps({
        "status": report["status"],
        "optimizer_steps": uninterrupted["optimizer_step_count"],
        "losses": [step["loss"] for step in uninterrupted["steps"]],
        "changed_parameters": [step["changed_parameters"] for step in uninterrupted["steps"]],
        "scaled_margin_increase_steps": margin_increases,
        "final_parameter_sha256": uninterrupted["final_parameter_sha256"],
        "checkpoint_resume_equivalence": "PASS",
        "invalid_input": "PASS", "crossed_pair": "PASS",
        "GRPO": "NOT_RUN", "model_quality": "NOT_RUN",
    }, sort_keys=True))
    return 0


def _fixture_raw_bundle(fixture):
    rows = [json.loads(line) for line in (
        fixture["pack_dir"] / "dpo_train.jsonl").read_text().splitlines()]
    identity = fixture["export_identity"]
    config = json.loads((ROOT / "configs/training/joint_dpo_live_gate.json")
                        .read_text(encoding="utf-8"))
    return live._pack_export_pairs(
        rows, identity["tokenizer"], tokenizer_dir=identity["tokenizer_dir"],
        tokenizer_id=identity["tokenizer_id"],
        tokenizer_revision=identity["tokenizer_revision"],
        tokenization_version=identity["tokenization_version"],
        span_loss_weights=config["span_loss_weights"], dpo_config=config["dpo"])


def _fixture_bundle_write(root, version):
    root = Path(root)
    fixture = build_fixture_pack(
        root / "fixture", export_version=int(version), rich=True)
    bundle = _fixture_raw_bundle(fixture)
    path = root / "token_export_preflight.json"
    live.sft_live._atomic_json(path, bundle)
    print(json.dumps({"status": "PASS", "pairs": len(bundle["pairs"]),
                      "real_dpo_optimizer": "NOT_RUN"}, sort_keys=True))
    return 0


def _fixture_bundle_verify(root, version, output):
    root = Path(root)
    fixture_root = root / "fixture"
    if int(version) == 2:
        tokenizer = token_fixture.V2FixtureTokenizer()
        tokenizer_dir = fixture_root / "fixture-tokenizer/tokenizer-v2"
        tokenizer_id, revision, tokenization_version = (
            "fixture-bytelevel-nfc", "fixture-revision-2", "fixture-tokenization-v2")
    else:
        tokenizer = token_fixture.fixture_tokenizer()
        tokenizer_dir = fixture_root / "fixture-tokenizer/tokenizer"
        tokenizer_id, revision, tokenization_version = (
            "fixture-unicode-bpe", "fixture-revision-1", "fixture-tokenization-v1")
    fixture = {
        "pack_dir": fixture_root / "pack",
        "export_identity": {
            "tokenizer": tokenizer, "tokenizer_dir": tokenizer_dir,
            "tokenizer_id": tokenizer_id, "tokenizer_revision": revision,
            "tokenization_version": tokenization_version,
        },
    }
    expected = _fixture_raw_bundle(fixture)
    bundle_path = root / "token_export_preflight.json"
    sidecar = Path(str(bundle_path) + ".sha256")
    if sidecar.read_text(encoding="ascii").strip() != stage_c.sha256_file(bundle_path):
        raise ValueError("fixture bundle SHA-256 mismatch")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if stage_c.canonical_json(bundle) != stage_c.canonical_json(expected):
        raise ValueError("fresh-process fixture bundle reconstruction differs")
    if "torch" in sys.modules or "transformers" in sys.modules:
        raise RuntimeError("ML import during fixture bundle verification")
    result = {
        "schema": "umminfer-joint-dpo-fixture-bundle-verification-v1",
        "status": "PASS", "fresh_process": "BYTE_IDENTICAL",
        "bundle_sha256": stage_c.sha256_file(bundle_path),
        "pair_count": len(bundle["pairs"]), "torch_imported": False,
        "real_dpo_optimizer": "NOT_RUN", "model_quality": "NOT_RUN",
        "sampled_online_behavior_ids": "NOT_CLAIMED",
    }
    live.sft_live._atomic_json(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


def _live_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--source-commit", required=True)
    return parser


if __name__ == "__main__":
    if "--fixture-bundle-write" in sys.argv:
        index = sys.argv.index("--fixture-bundle-write")
        raise SystemExit(_fixture_bundle_write(sys.argv[index + 1], sys.argv[index + 2]))
    if "--fixture-bundle-verify" in sys.argv:
        index = sys.argv.index("--fixture-bundle-verify")
        try:
            status = _fixture_bundle_verify(
                sys.argv[index + 1], sys.argv[index + 2], sys.argv[index + 3])
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            print(f"fixture bundle verification error: {exc}", file=sys.stderr)
            status = 2
        raise SystemExit(status)
    if "--live" in sys.argv:
        raise SystemExit(run_live_gate(_live_parser().parse_args()))
    result = unittest.main(exit=False, verbosity=2)
    print("REAL JOINT DPO OPTIMIZER: NOT_RUN (default CPU preflight suite)")
    print("MODEL QUALITY / GENERALIZED IMPROVEMENT / GRPO: NOT_RUN")
    raise SystemExit(0 if result.result.wasSuccessful() else 1)
