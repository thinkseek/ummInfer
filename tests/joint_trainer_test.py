#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deterministic CPU-only tests for Stage C Joint Trainer & Ablation Driver.

Usage: python3 -m unittest tests/joint_trainer_test.py -v
No GPU, real model, torch, numpy, transformers, or serving process is used.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import narrative_rl_fixtures as fixture  # noqa: E402
import umminfer_joint_trainer as trainer  # noqa: E402
import umminfer_rl_trajectory as traj  # noqa: E402
import umminfer_sft_dataset as ds  # noqa: E402

# Assembled dynamically so literal internal tokens never appear in source.
FORBIDDEN_TOKENS = [
    "h" + "20b",
    "lin" + "yu",
    "172." + "17.0.2",
    "/" + "Users/",
    "/home/",
]


def _make_v2_record(recorder, *, prompt_text, clean_parts, slots=(),
                    session_id="sess-0", request_id="req-0",
                    expected_modalities=None, explicit_anchor=False):
    """Record a verified schema-v2 trajectory with CAS media artifacts."""
    inputs = [{
        "kind": "text",
        "ref": None,
        "media_type": None,
        "sha256": traj.sha256_text(prompt_text),
        "text": prompt_text,
    }]
    events = []
    legs = []
    artifacts = []
    seq = 0

    for part in clean_parts:
        events.append({
            "seq": seq,
            "server_seq": seq,
            "type": "vlm_delta",
            "stage": "vlm",
            "is_final": False,
            "text": part,
        })
        seq += 1

    for idx, slot in enumerate(slots):
        media_type, slot_prompt, text_offset = slot
        slot_id = f"slot-{idx}"
        event = {
            "seq": seq,
            "server_seq": seq,
            "type": "visual_slot",
            "stage": media_type,
            "is_final": False,
            "text": "",
            "slot_id": slot_id,
            "slot_index": idx,
            "media_type": media_type,
            "prompt": slot_prompt,
            "text_offset": text_offset,
        }
        if explicit_anchor:
            event["continuity_mode"] = "i2v_chain"
            if idx == 1 and media_type == "video":
                event["anchor_slot_index"] = 0
                event["anchor_explicit"] = True
            elif idx != 0 or media_type != "image":
                raise ValueError("explicit-anchor fixture requires image slot 0 then video slot 1")
        events.append(event)
        seq += 1

        raw = fixture.PNG if media_type == "image" else fixture.MP4
        mime = "image/png" if media_type == "image" else "video/mp4"
        suffix = ".png" if media_type == "image" else ".mp4"
        digest = traj.sha256_bytes(raw)
        recorder.artifact_dir.mkdir(parents=True, exist_ok=True)
        ref = digest + suffix
        (recorder.artifact_dir / ref).write_bytes(raw)
        art_id = f"art-{media_type}-{idx}"
        chunks_sha = traj.sha256_bytes(bytes.fromhex(digest))
        artifact_event = {
            "seq": seq,
            "server_seq": seq,
            "type": "artifact",
            "stage": media_type,
            "is_final": True,
            "text": "",
            "artifact": {
                "artifact_id": art_id,
                "media_type": media_type,
                "mime_type": mime,
                "format": "png" if media_type == "image" else "mp4",
                "index": 0,
                "total": 1,
                "chunk_bytes": len(raw),
                "chunk_sha256": digest,
                "bytes": len(raw),
                "sha256": digest,
                "ref": ref,
                "slot_id": slot_id,
                "slot_index": idx,
            },
        }
        if explicit_anchor and idx == 1:
            artifact_event["artifact"].update({
                "anchor_slot_index": 0,
                "anchor_explicit": True,
                "anchor_sha256": artifacts[0]["sha256"],
            })
        events.append(artifact_event)
        seq += 1
        artifact_record = {
            "artifact_id": art_id,
            "media_type": media_type,
            "mime_type": mime,
            "format": "png" if media_type == "image" else "mp4",
            "ref": ref,
            "sha256": digest,
            "bytes": len(raw),
            "chunk_count": 1,
            "chunks_sha256": chunks_sha,
            "slot_id": slot_id,
            "slot_index": idx,
        }
        if explicit_anchor and idx == 1:
            artifact_record.update({
                "anchor_slot_index": 0,
                "anchor_explicit": True,
                "anchor_sha256": artifacts[0]["sha256"],
            })
        artifacts.append(artifact_record)
        leg = {
            "stage": media_type,
            "slot_id": slot_id,
            "slot_index": idx,
            "state": "completed",
        }
        if explicit_anchor and idx == 1:
            leg.update({
                "anchor_slot_index": 0,
                "anchor_explicit": True,
                "anchor_sha256": artifacts[0]["sha256"],
            })
        legs.append(leg)

    output_text = "".join(clean_parts)
    events.append({
        "seq": seq,
        "server_seq": seq,
        "type": "complete",
        "stage": None,
        "is_final": True,
        "text": output_text,
        "legs": legs,
    })

    record = recorder.record(
        session_id=session_id,
        request_id=request_id,
        endpoint="/v1/chat/completions",
        inputs=inputs,
        params={"model": "fixture-vlm", "stream": True},
        events=events,
        terminal_state="completed",
        error=None,
        output_text=output_text,
        artifacts=artifacts,
    )
    if expected_modalities is not None:
        lines = recorder.jsonl_path.read_text(encoding="utf-8").splitlines()
        enriched = json.loads(lines[-1])
        enriched["ablation"] = {
            "expected_visual_modalities": list(expected_modalities),
        }
        lines[-1] = ds.canonical_json(enriched).decode("utf-8")
        recorder.jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return enriched
    return record


def _build_synthetic_stage_c_pack(root, *, eval_ratio=0.25):
    """Build a complete verified SFT + DPO + GRPO pack via umminfer_sft_dataset."""
    rec_path = root / "trajectories.jsonl"
    recorder = traj.TrajectoryRecorder(rec_path)

    # Multiple prompts with varying visual modalities, spoken tags, and margins.
    prompts = [
        (
            "Explain black hole accretion disks and relativistic jets.",
            ["image", "video"],
            [
                (
                    ["<spoken>Watch how superheated plasma spirals inward.</spoken> ",
                     "Accretion disks radiate strongly across X-ray bands."],
                    [
                        ("image", "cross section of glowing accretion disk", 59),
                        ("video", "plasma spiraling into event horizon with polar jets", 111),
                    ],
                ),
                (
                    ["<spoken>Plasma heats up near the horizon.</spoken> ",
                     "Only static description without dynamic motion."],
                    [
                        ("image", "static ring sketch", 53),
                    ],
                ),
                (
                    ["Accretion disks are luminous structures around compact objects."],
                    [],
                ),
            ],
        ),
        (
            "Illustrate plate tectonics and subduction zones.",
            ["image", "video"],
            [
                (
                    ["Oceanic crust descends beneath continental plates. ",
                     "<spoken>Notice the volcanic arc forming above the trench.</spoken>"],
                    [
                        ("image", "labelled subduction zone cross section", 51),
                        ("video", "oceanic slab sinking into mantle over time", 118),
                    ],
                ),
                (
                    ["Tectonic plates collide and form mountain belts."],
                    [
                        ("image", "tectonic map view one", 15),
                        ("image", "tectonic map view two", 48),
                    ],
                ),
                (
                    ["Plates move slowly across the asthenosphere."],
                    [],
                ),
            ],
        ),
        (
            "Show how an optical prism disperses white light.",
            ["image"],
            [
                (
                    ["<spoken>Different wavelengths refract at different angles.</spoken> ",
                     "Violet bends more sharply than red."],
                    [
                        ("image", "glass prism splitting white beam into rainbow spectrum", 68),
                    ],
                ),
                (
                    ["White light separates into colors inside glass."],
                    [],
                ),
            ],
        ),
        (
            "Compute 19 * 27 concisely without visuals.",
            [],
            [
                # Zero-variance GRPO group: both rollouts correctly omit visual slots (score=1.0).
                (["<spoken>Nineteen times twenty seven is 513.</spoken>"], []),
                (["19 * 27 = 513."], []),
            ],
        ),
        (
            "用中文讲解红灯笼随风飘动的画面",
            ["image", "video"],
            [
                (
                    ["黎明时分，", "红灯笼在微风中缓缓升起。"],
                    [
                        ("image", "glowing red lantern at dawn", len("黎明时分，".encode("utf-8"))),
                        ("video", "red lantern gently swaying in morning breeze",
                         len("黎明时分，红灯笼在微风中缓缓升起。".encode("utf-8"))),
                    ],
                ),
                (
                    ["红灯笼高高挂起。"],
                    [],
                ),
            ],
        ),
    ]

    for p_idx, (prompt_text, expected_mods, rollouts) in enumerate(prompts):
        for r_idx, (parts, slots) in enumerate(rollouts):
            _make_v2_record(
                recorder,
                prompt_text=prompt_text,
                clean_parts=parts,
                slots=slots,
                session_id=f"sess-{p_idx}-{r_idx}",
                request_id=f"req-{p_idx}-{r_idx}",
                expected_modalities=expected_mods,
            )

    pack_dir = root / "dataset_pack"
    manifest = ds.build_dataset(
        [rec_path],
        pack_dir,
        mode="both",
        seed="joint-trainer-test-seed",
        eval_ratio=eval_ratio,
        min_margin=0.1,
        min_group_size=2,
    )
    return pack_dir, manifest


class SpanLossMaskerTests(unittest.TestCase):
    def test_lossless_span_partition_across_modalities_utf8_and_edge_cases(self):
        cases = [
            "",
            "Plain prose with no visual or spoken tags.",
            "At dawn, [[image: a glowing red lantern]][[video: lantern drifts downstream]]a lantern rises.",
            "<spoken>Listen carefully.</spoken> Here is [[image: accretion disk]] and [[video: plasma swirl]].",
            "黎明时分，[[image: red lantern at dawn]]<spoken>缓缓升起。</spoken>尾声。",
            "<spoken>Before [[image: nested diagram]] after.</spoken>",
            "Unclosed [[image: not finished yet and trailing text.",
            "Empty body [[image: ]] followed by [[video: orbiting planets]]!",
        ]
        for text in cases:
            with self.subTest(text=text):
                spans = trainer.parse_interleaved_spans(text)
                reconstructed = "".join(s["text"] for s in spans)
                self.assertEqual(reconstructed, text)
                cursor = 0
                for span in spans:
                    self.assertEqual(span["start"], cursor)
                    self.assertEqual(text[span["start"]:span["end"]], span["text"])
                    self.assertIn(span["span_type"], trainer.SPAN_TYPES)
                    cursor = span["end"]
                self.assertEqual(cursor, len(text))

        # Verify exact span type breakdown on interleaved image + video + spoken text.
        sample = (
            "<spoken>Intro voice.</spoken> "
            "Prose lead [[image: star field]] middle [[video: supernova wave]] end."
        )
        spans = trainer.parse_interleaved_spans(sample)
        types = [(s["span_type"], s["modality"], s["text"]) for s in spans]
        self.assertEqual(types, [
            ("spoken_style", "audio", "<spoken>Intro voice.</spoken>"),
            ("narrative_prose", "text", " Prose lead "),
            ("visual_route_tag", "image", "[[image: "),
            ("visual_prompt_body", "image", "star field"),
            ("visual_route_tag", "image", "]]"),
            ("narrative_prose", "text", " middle "),
            ("visual_route_tag", "video", "[[video: "),
            ("visual_prompt_body", "video", "supernova wave"),
            ("visual_route_tag", "video", "]]"),
            ("narrative_prose", "text", " end."),
        ])

    def test_ablation_profiles_deterministically_modify_span_weights(self):
        sample = (
            "<spoken>Spoken cue.</spoken> "
            "Prose [[image: static diagram]] and [[video: dynamic motion]] done."
        )
        full = trainer.build_span_loss_mask(sample, ablation_profile="joint_full")
        self.assertEqual(full["masked_tokens"], 0)
        self.assertGreater(full["active_tokens"], 0)

        no_route = trainer.build_span_loss_mask(
            sample, ablation_profile="ablate_visual_routing")
        for s in no_route["spans"]:
            if s["span_type"] == "visual_route_tag":
                self.assertEqual(s["weight"], 0.0)
                self.assertFalse(s["active"])
            else:
                self.assertGreater(s["weight"], 0.0)
        self.assertGreater(no_route["masked_tokens"], 0)
        self.assertLess(no_route["active_tokens"], full["active_tokens"])

        no_prompt = trainer.build_span_loss_mask(
            sample, ablation_profile="ablate_visual_prompt")
        for s in no_prompt["spans"]:
            if s["span_type"] == "visual_prompt_body":
                self.assertEqual(s["weight"], 0.0)
            else:
                self.assertGreater(s["weight"], 0.0)

        no_video = trainer.build_span_loss_mask(
            sample, ablation_profile="ablate_video_form")
        for s in no_video["spans"]:
            if s["modality"] == "video":
                self.assertEqual(s["weight"], 0.0)
            else:
                self.assertGreater(s["weight"], 0.0)

        no_spoken = trainer.build_span_loss_mask(
            sample, ablation_profile="ablate_spoken_style")
        for s in no_spoken["spans"]:
            if s["span_type"] == "spoken_style":
                self.assertEqual(s["weight"], 0.0)
            else:
                self.assertGreater(s["weight"], 0.0)


class JointTrainerEndToEndTests(unittest.TestCase):
    def test_tampered_manifest_sidecar_and_jsonl_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix="umm-jt-tamper-") as tmp:
            root = Path(tmp)
            pack_dir, _ = _build_synthetic_stage_c_pack(root, eval_ratio=0.25)

            # 1) Tamper manifest.json.sha256 -> fails closed.
            sidecar = pack_dir / "manifest.json.sha256"
            orig_sidecar = sidecar.read_text(encoding="utf-8")
            sidecar.write_text("0" * 64 + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest.json.sha256 mismatch"):
                trainer.prepare_training_run(
                    mode="joint",
                    dataset_dir=pack_dir,
                    output_dir=root / "out_bad_sidecar",
                )
            sidecar.write_text(orig_sidecar, encoding="utf-8")

            # 2) Tamper a JSONL pack -> fails closed.
            sft_train = pack_dir / "sft_train.jsonl"
            orig_sft = sft_train.read_bytes()
            sft_train.write_bytes(orig_sft + b"\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 or byte length mismatch"):
                trainer.prepare_training_run(
                    mode="sft",
                    dataset_dir=pack_dir,
                    output_dir=root / "out_bad_jsonl",
                )
            sft_train.write_bytes(orig_sft)

            # 3) Cross-split prompt leakage fails closed even when hashes are re-signed.
            leaked_pack = root / "leaked_pack"
            leaked_pack.mkdir()
            for item in pack_dir.iterdir():
                (leaked_pack / item.name).write_bytes(item.read_bytes())
            train_lines = (leaked_pack / "sft_train.jsonl").read_text(
                encoding="utf-8").splitlines()
            eval_lines = (leaked_pack / "sft_eval.jsonl").read_text(
                encoding="utf-8").splitlines()
            leaked_row = json.loads(train_lines[0])
            leaked_row["split"] = "eval"
            eval_lines.append(ds.canonical_json(leaked_row).decode("utf-8"))
            eval_bytes = ("\n".join(eval_lines) + "\n").encode("utf-8")
            (leaked_pack / "sft_eval.jsonl").write_bytes(eval_bytes)
            m_obj = json.loads((leaked_pack / "manifest.json").read_text(
                encoding="utf-8"))
            m_obj["outputs"]["sft_eval"]["bytes"] = len(eval_bytes)
            m_obj["outputs"]["sft_eval"]["sha256"] = ds.sha256_bytes(eval_bytes)
            m_obj["outputs"]["sft_eval"]["record_count"] = len(eval_lines)
            m_bytes = ds.canonical_json(m_obj) + b"\n"
            (leaked_pack / "manifest.json").write_bytes(m_bytes)
            (leaked_pack / "manifest.json.sha256").write_text(
                ds.sha256_bytes(m_bytes) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prompt leakage detected"):
                trainer.prepare_training_run(
                    mode="sft",
                    dataset_dir=leaked_pack,
                    output_dir=root / "out_leaked",
                )

            # 4) Incompatible mode + ablation -> fails closed.
            with self.assertRaisesRegex(ValueError, "disables dpo stage"):
                trainer.prepare_training_run(
                    mode="dpo",
                    ablation="sft_only",
                    dataset_dir=pack_dir,
                    output_dir=root / "out_bad_combo",
                )

    def test_modes_ablations_micro_steps_and_numerical_properties(self):
        with tempfile.TemporaryDirectory(prefix="umm-jt-e2e-") as tmp:
            root = Path(tmp)
            pack_dir, ds_manifest = _build_synthetic_stage_c_pack(
                root, eval_ratio=0.0)
            self.assertEqual(ds_manifest["split_audit"]["leakage_count"], 0)
            self.assertGreater(ds_manifest["outputs"]["sft_train"]["record_count"], 0)
            self.assertGreater(ds_manifest["outputs"]["dpo_train"]["record_count"], 0)
            self.assertGreater(ds_manifest["outputs"]["grpo_train"]["record_count"], 0)

            sft_cfg = ROOT / "configs" / "training" / "joint_sft_default.json"
            pref_cfg = ROOT / "configs" / "training" / "joint_dpo_grpo_default.json"

            # 1) SFT: joint_full vs ablate_visual_routing vs ablate_video_form
            # Run 6 steps so every mini-batch (including image+video+spoken turns) is exercised.
            sft_full = trainer.prepare_training_run(
                mode="sft",
                ablation="joint_full",
                dataset_dir=pack_dir,
                config_path=sft_cfg,
                output_dir=root / "sft_full",
                dry_run_steps=6,
                seed="num-seed-1",
            )
            sft_no_route = trainer.prepare_training_run(
                mode="sft",
                ablation="ablate_visual_routing",
                dataset_dir=pack_dir,
                config_path=sft_cfg,
                output_dir=root / "sft_no_route",
                dry_run_steps=6,
                seed="num-seed-1",
            )
            sft_no_video = trainer.prepare_training_run(
                mode="sft",
                ablation="ablate_video_form",
                dataset_dir=pack_dir,
                config_path=sft_cfg,
                output_dir=root / "sft_no_video",
                dry_run_steps=6,
                seed="num-seed-1",
            )
            sft_no_spoken = trainer.prepare_training_run(
                mode="sft",
                ablation="ablate_spoken_style",
                dataset_dir=pack_dir,
                config_path=sft_cfg,
                output_dir=root / "sft_no_spoken",
                dry_run_steps=6,
                seed="num-seed-1",
            )

            self.assertEqual(sft_full["mode"], "dry_run_micro_step")
            self.assertEqual(
                sft_full["REAL_TRAINING"],
                "NOT_RUN (CPU_REFERENCE_MICRO_STEP)",
            )
            self.assertFalse(sft_full["claims"]["model_weights_updated"])
            full_sft_hist = sft_full["micro_step_verification"]["stages"]["sft"]
            no_route_hist = sft_no_route["micro_step_verification"]["stages"]["sft"]
            no_video_hist = sft_no_video["micro_step_verification"]["stages"]["sft"]
            no_spoken_hist = sft_no_spoken["micro_step_verification"]["stages"]["sft"]

            self.assertTrue(math.isfinite(full_sft_hist["initial_loss"]))
            self.assertTrue(math.isfinite(no_route_hist["initial_loss"]))
            # Loss strictly responds to ablation mask changes across visual/spoken samples.
            self.assertNotEqual(
                full_sft_hist["final_loss"], no_route_hist["final_loss"])
            self.assertNotEqual(
                full_sft_hist["final_loss"], no_video_hist["final_loss"])
            self.assertNotEqual(
                full_sft_hist["final_loss"], no_spoken_hist["final_loss"])
            self.assertGreater(
                sft_no_route["statistics"]["sft_train"]["masked_tokens"],
                sft_full["statistics"]["sft_train"]["masked_tokens"],
            )
            self.assertEqual(
                sft_no_route["statistics"]["sft_train"]["weighted_token_mass"]["visual_route_tag"],
                0.0,
            )
            self.assertGreater(
                sft_no_video["statistics"]["sft_train"]["masked_tokens"],
                sft_full["statistics"]["sft_train"]["masked_tokens"],
            )
            self.assertEqual(
                sft_no_spoken["statistics"]["sft_train"]["weighted_token_mass"]["spoken_style"],
                0.0,
            )

            # 2) DPO: Bradley-Terry log-sigmoid loss strictly decreases on chosen > rejected.
            dpo_run = trainer.prepare_training_run(
                mode="dpo",
                ablation="joint_full",
                dataset_dir=pack_dir,
                config_path=pref_cfg,
                output_dir=root / "dpo_run",
                dry_run_steps=6,
                seed="dpo-seed-1",
            )
            dpo_metrics = dpo_run["micro_step_verification"]["stages"]["dpo"]
            self.assertEqual(dpo_metrics["steps_executed"], 6)
            self.assertAlmostEqual(
                dpo_metrics["initial_loss"], math.log(2.0), places=6)
            self.assertLess(
                dpo_metrics["final_loss"], dpo_metrics["initial_loss"])
            self.assertGreater(dpo_metrics["final_implicit_reward_margin"], 0.0)

            # 3) GRPO: finite clipped objective + KL penalty, handles zero-variance groups safely.
            grpo_run = trainer.prepare_training_run(
                mode="grpo",
                ablation="joint_full",
                dataset_dir=pack_dir,
                config_path=pref_cfg,
                output_dir=root / "grpo_run",
                dry_run_steps=6,
                seed="grpo-seed-1",
            )
            grpo_metrics = grpo_run["micro_step_verification"]["stages"]["grpo"]
            self.assertEqual(grpo_metrics["steps_executed"], 6)
            self.assertGreater(grpo_metrics["zero_variance_groups_handled"], 0)
            for entry in grpo_metrics["history"]:
                self.assertTrue(math.isfinite(entry["loss"]))
                self.assertTrue(math.isfinite(entry["mean_kl"]))
                self.assertGreaterEqual(entry["mean_kl"], 0.0)
                self.assertTrue(math.isfinite(entry["grad_norm"]))

            # Also verify single-member GRPO group advantage helper directly (zero NaN/Inf).
            singleton_adv, is_zv = trainer._compute_safe_group_advantages({
                "rollouts": [{"assistant_target": "only one", "score": 0.8}],
            })
            self.assertEqual(singleton_adv, [0.0])
            self.assertTrue(is_zv)

            # 4) Joint mode + stage ablation arms: sft_only, sft_plus_dpo, sft_plus_grpo, joint_full.
            joint_full = trainer.prepare_training_run(
                mode="joint",
                ablation="joint_full",
                dataset_dir=pack_dir,
                config_path=pref_cfg,
                output_dir=root / "joint_full",
                dry_run_steps=3,
                seed="joint-seed",
            )
            self.assertEqual(joint_full["active_stages"], ["sft", "dpo", "grpo"])
            self.assertEqual(
                sorted(joint_full["micro_step_verification"]["stages"].keys()),
                ["dpo", "grpo", "sft"],
            )

            arm_sft_only = trainer.prepare_training_run(
                mode="joint",
                ablation="sft_only",
                dataset_dir=pack_dir,
                config_path=pref_cfg,
                output_dir=root / "arm_sft_only",
                dry_run_steps=2,
            )
            self.assertEqual(arm_sft_only["active_stages"], ["sft"])

            arm_sft_dpo = trainer.prepare_training_run(
                mode="joint",
                ablation="sft_plus_dpo",
                dataset_dir=pack_dir,
                config_path=pref_cfg,
                output_dir=root / "arm_sft_dpo",
                dry_run_steps=2,
            )
            self.assertEqual(arm_sft_dpo["active_stages"], ["sft", "dpo"])

            arm_sft_grpo = trainer.prepare_training_run(
                mode="joint",
                ablation="sft_plus_grpo",
                dataset_dir=pack_dir,
                config_path=pref_cfg,
                output_dir=root / "arm_sft_grpo",
                dry_run_steps=2,
            )
            self.assertEqual(arm_sft_grpo["active_stages"], ["sft", "grpo"])

            # 5) Plan mode & GPU launch script mode + shell syntax validation.
            plan_run = trainer.prepare_training_run(
                mode="joint",
                ablation="ablate_spoken_style",
                dataset_dir=pack_dir,
                output_dir=root / "plan_out",
                dry_run_steps=0,
            )
            self.assertEqual(plan_run["mode"], "plan")
            self.assertEqual(plan_run["REAL_TRAINING"], "NOT_RUN")
            self.assertIsNone(plan_run["micro_step_verification"])

            gpu_script_run = trainer.prepare_training_run(
                mode="joint",
                ablation="ablate_visual_prompt",
                dataset_dir=pack_dir,
                output_dir=root / "gpu_script_out",
                gpu_launch_script=True,
            )
            self.assertEqual(gpu_script_run["mode"], "gpu_launch_script")
            self.assertEqual(gpu_script_run["REAL_TRAINING"], "NOT_RUN")

            launch_sh = root / "gpu_script_out" / "launch_train.sh"
            self.assertTrue(launch_sh.is_file())
            syntax = subprocess.run(
                ["bash", "-n", str(launch_sh)],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(syntax.returncode, 0, syntax.stderr)
            script_text = launch_sh.read_text(encoding="utf-8")
            self.assertIn(ds_manifest["outputs"]["sft_train"]["path"], script_text)
            for token in FORBIDDEN_TOKENS:
                self.assertNotIn(token.lower(), script_text.lower())

    def test_byte_determinism_and_cli_execution(self):
        with tempfile.TemporaryDirectory(prefix="umm-jt-det-") as tmp:
            root = Path(tmp)
            pack_dir, _ = _build_synthetic_stage_c_pack(root, eval_ratio=0.25)
            out_a = root / "run_a"
            out_b = root / "run_b"
            cfg = ROOT / "configs" / "training" / "joint_dpo_grpo_default.json"

            for out_dir in (out_a, out_b):
                cmd = [
                    sys.executable, "-I",
                    str(ROOT / "python" / "umminfer_joint_trainer.py"),
                    "--mode", "joint",
                    "--ablation", "joint_full",
                    "--dataset-dir", str(pack_dir),
                    "--config", str(cfg),
                    "--output-dir", str(out_dir),
                    "--dry-run-steps", "3",
                    "--seed", "deterministic-seed-42",
                ]
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=60)
                self.assertEqual(
                    proc.returncode, 0, proc.stdout + proc.stderr)
                parsed = json.loads(proc.stdout)
                self.assertEqual(parsed["mode"], "dry_run_micro_step")
                self.assertEqual(
                    parsed["REAL_TRAINING"],
                    "NOT_RUN (CPU_REFERENCE_MICRO_STEP)",
                )

            # Byte-identical run_manifest.json, .sha256, and launch_train.sh.
            for fname in ("run_manifest.json", "run_manifest.json.sha256",
                          "launch_train.sh"):
                bytes_a = (out_a / fname).read_bytes()
                bytes_b = (out_b / fname).read_bytes()
                self.assertEqual(bytes_a, bytes_b, f"mismatch in {fname}")

            actual_sha = hashlib.sha256(
                (out_a / "run_manifest.json").read_bytes()).hexdigest()
            sidecar_sha = (
                out_a / "run_manifest.json.sha256"
            ).read_text(encoding="utf-8").strip()
            self.assertEqual(actual_sha, sidecar_sha)

    def test_configs_hygiene_valid_json_and_serving_isolation(self):
        config_dir = ROOT / "configs" / "training"
        expected_files = (
            "joint_sft_default.json",
            "joint_dpo_grpo_default.json",
            "ablation_matrix.json",
        )
        for fname in expected_files:
            path = config_dir / fname
            self.assertTrue(path.is_file(), f"missing {path}")
            raw = path.read_text(encoding="utf-8")
            parsed = json.loads(raw)
            self.assertIsInstance(parsed, dict)
            self.assertEqual(parsed.get("config_schema_version"), 1)
            lowered = raw.lower()
            for token in FORBIDDEN_TOKENS:
                self.assertNotIn(
                    token.lower(), lowered,
                    f"forbidden token {token!r} found in {fname}",
                )

        # Verify ablation_matrix.json arms align 1-to-1 with trainer profiles.
        matrix = json.loads(
            (config_dir / "ablation_matrix.json").read_text(encoding="utf-8"))
        self.assertEqual(matrix["baseline_arm"], "joint_full")
        self.assertEqual(
            set(matrix["arms"].keys()),
            set(trainer.VALID_ABLATION_PROFILES),
        )
        for arm_id, arm in matrix["arms"].items():
            self.assertEqual(arm["arm_id"], arm_id)
            self.assertTrue(arm["hypothesis"])
            self.assertIsInstance(arm["expected_metric_movement"], dict)
            self.assertTrue(arm["expected_metric_movement"])

        # Verify zero C++ serving references and zero ML framework imports.
        for directory in (ROOT / "src", ROOT / "include"):
            if not directory.is_dir():
                continue
            for path in directory.rglob("*"):
                if path.is_file():
                    self.assertNotIn(
                        "umminfer_joint_trainer",
                        path.read_text(encoding="utf-8", errors="ignore"),
                        str(path),
                    )
        for mod_name in ("torch", "numpy", "transformers", "trl", "peft"):
            self.assertNotIn(mod_name, sys.modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
