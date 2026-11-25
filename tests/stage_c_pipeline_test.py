#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deterministic CPU-only offline unit & E2E tests for Stage C Pipeline Driver.

Usage: python3 -m unittest tests/stage_c_pipeline_test.py -v
Zero network calls, zero GPU/torch/numpy, zero serving processes.
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
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import narrative_rl_fixtures as fixture  # noqa: E402
import umminfer_rl_trajectory as traj  # noqa: E402
import umminfer_sft_dataset as ds  # noqa: E402
import umminfer_stage_c_pipeline as pipeline  # noqa: E402

# Assembled dynamically so literal internal tokens never appear in source.
FORBIDDEN_TOKENS = (
    "h" + "20b",
    "lin" + "yu",
    "172." + "17.0.2",
    "/" + "Users/",
    "/" + "home/",
)


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(Path(root).rglob("*"))
        if path.is_file()
    }


def _make_v2_record(
    recorder: traj.TrajectoryRecorder,
    *,
    prompt_text: str,
    clean_parts: list[str],
    slots=(),
    session_id: str = "sess-0",
    request_id: str = "req-0",
    expected_modalities=None,
    latency_ms=None,
    terminal_state: str = "completed",
    error: str | None = None,
):
    """Write a verified schema-v2 trajectory with CAS media and ablation labels."""
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

    for idx, (media_type, slot_prompt, text_offset) in enumerate(slots):
        slot_id = f"slot-{idx}"
        events.append({
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
        })
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
        events.append({
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
        })
        seq += 1
        artifacts.append({
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
        })
        legs.append({
            "stage": media_type,
            "slot_id": slot_id,
            "slot_index": idx,
            "state": "completed",
        })

    output_text = "".join(clean_parts) if terminal_state == "completed" else ""
    if terminal_state == "completed":
        events.append({
            "seq": seq,
            "server_seq": seq,
            "type": "complete",
            "stage": None,
            "is_final": True,
            "text": output_text,
            "legs": legs,
        })
    elif terminal_state == "cancelled":
        events.append({
            "seq": seq,
            "server_seq": seq,
            "type": "cancelled",
            "stage": None,
            "is_final": True,
            "text": "",
            "legs": legs,
        })
    else:
        events.append({
            "seq": seq,
            "server_seq": seq,
            "type": "error",
            "stage": None,
            "is_final": True,
            "text": error or "failed",
            "error": error or "failed",
            "legs": legs,
        })

    record = recorder.record(
        session_id=session_id,
        request_id=request_id,
        endpoint="/v1/chat/completions",
        inputs=inputs,
        params={"model": "fixture-vlm", "stream": True},
        events=events,
        terminal_state=terminal_state,
        error=error,
        output_text=output_text,
        artifacts=artifacts,
    )

    if expected_modalities is not None or latency_ms is not None:
        lines = recorder.jsonl_path.read_text(encoding="utf-8").splitlines()
        enriched = json.loads(lines[-1])
        ablation = {}
        if expected_modalities is not None:
            ablation["expected_visual_modalities"] = list(expected_modalities)
        if latency_ms is not None:
            ablation["latency"] = dict(latency_ms)
        enriched["ablation"] = ablation
        lines[-1] = traj.canonical_json(enriched).decode("utf-8")
        recorder.jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return enriched
    return record


def _build_fixture_trajectories(root: Path) -> tuple[Path, Path]:
    """Create two realistic schema-v2 JSONL files with CAS dirs and reward margins."""
    file_a = root / "export_a" / "trajectories_a.jsonl"
    dir_b = root / "export_b"
    file_b = dir_b / "trajectories_b.jsonl"
    rec_a = traj.TrajectoryRecorder(file_a)
    rec_b = traj.TrajectoryRecorder(file_b)

    # Prompts with diverse modalities, spoken tags, and clear CompositeUmmReward margins.
    cases_a = [
        (
            "Prove concisely that sqrt(2) is irrational using integer parity.",
            [],
            [
                (
                    ["Assume sqrt(2) = p/q in lowest terms. ",
                     "Then p^2 = 2q^2, so both p and q are even, contradicting coprimality."],
                    [],
                    {"ttft_ms": 140.0, "terminal_ms": 1200.0},
                ),
                (
                    ["Assume sqrt(2) = p/q in lowest terms with an unneeded diagram."],
                    [("image", "redundant chalkboard equation", 32)],
                    {"ttft_ms": 220.0, "terminal_ms": 4800.0},
                ),
            ],
        ),
        (
            "Explain black hole accretion disks and relativistic jets.",
            ["image", "video"],
            [
                (
                    ["<spoken>Watch superheated plasma spiral inward.</spoken> ",
                     "Accretion disks glow brightly across X-ray bands."],
                    [
                        ("image", "cross section of glowing accretion disk", 59),
                        ("video", "plasma spiraling into event horizon with polar jets", 108),
                    ],
                    {"ttft_ms": 180.0, "terminal_ms": 3400.0},
                ),
                (
                    ["<spoken>Plasma heats near the horizon.</spoken> ",
                     "Static description without dynamic motion."],
                    [("image", "static ring sketch", 50)],
                    {"ttft_ms": 250.0, "terminal_ms": 5100.0},
                ),
                (
                    ["Accretion disks form luminous rings around compact objects."],
                    [],
                    {"ttft_ms": 300.0, "terminal_ms": 6200.0},
                ),
            ],
        ),
        (
            "Illustrate a sunlit botanical glass conservatory.",
            ["image"],
            [
                (
                    ["<spoken>Sunlight streams through arched ironwork.</spoken> ",
                     "Tropical ferns thrive beneath the glass dome."],
                    [("image", "sunlit victorian botanical glass conservatory", 60)],
                    {"ttft_ms": 160.0, "terminal_ms": 2100.0},
                ),
                (
                    ["Sunlight streams through a botanical conservatory."],
                    [],
                    {"ttft_ms": 260.0, "terminal_ms": 3900.0},
                ),
            ],
        ),
    ]

    cases_b = [
        (
            "Narrate an ocean wave breaking along a rocky cliff in motion.",
            ["video"],
            [
                (
                    ["<spoken>The swell steepens into foam.</spoken> ",
                     "White spray cascades across dark basalt."],
                    [("video", "slow motion ocean wave cresting over basalt rocks", 48)],
                    {"ttft_ms": 175.0, "terminal_ms": 3900.0},
                ),
                (
                    ["The swell steepens and breaks on rocks."],
                    [("image", "frozen ocean wave snapshot", 20)],
                    {"ttft_ms": 310.0, "terminal_ms": 6800.0},
                ),
            ],
        ),
        (
            "Describe plate tectonics and volcanic arcs above subduction zones.",
            ["image", "video"],
            [
                (
                    ["Oceanic crust descends beneath continental plates. ",
                     "<spoken>Magma rises to build coastal volcanoes.</spoken>"],
                    [
                        ("image", "geological cross section of subduction trench", 51),
                        ("video", "subducting slab triggering rising magma plumes", 107),
                    ],
                    {"ttft_ms": 190.0, "terminal_ms": 3600.0},
                ),
                (
                    ["Plates collide and form mountain trenches."],
                    [],
                    {"ttft_ms": 280.0, "terminal_ms": 5900.0},
                ),
            ],
        ),
        (
            "Summarize binary search time complexity in pure text.",
            [],
            [
                (
                    ["Binary search halves the sorted interval on every comparison, "
                     "yielding O(log n) worst-case time complexity."],
                    [],
                    {"ttft_ms": 120.0, "terminal_ms": 950.0},
                ),
                (
                    ["Binary search halves the interval repeatedly with extra visual."],
                    [("image", "unnecessary array diagram", 30)],
                    {"ttft_ms": 240.0, "terminal_ms": 4200.0},
                ),
            ],
        ),
    ]

    req_counter = 0
    for prompt, expected, rollouts in cases_a:
        for idx, (parts, slots, latency) in enumerate(rollouts):
            _make_v2_record(
                rec_a,
                prompt_text=prompt,
                clean_parts=parts,
                slots=slots,
                session_id=f"sess-a-{req_counter}",
                request_id=f"req-a-{req_counter}-{idx}",
                expected_modalities=expected,
                latency_ms=latency,
            )
            req_counter += 1

    # Also include one structurally valid cancelled turn to verify clean filtering.
    _make_v2_record(
        rec_a,
        prompt_text="Interrupted voice query.",
        clean_parts=[],
        slots=[],
        session_id="sess-cancelled",
        request_id="req-cancelled-0",
        expected_modalities=[],
        latency_ms={"terminal_ms": 180.0},
        terminal_state="cancelled",
    )

    for prompt, expected, rollouts in cases_b:
        for idx, (parts, slots, latency) in enumerate(rollouts):
            _make_v2_record(
                rec_b,
                prompt_text=prompt,
                clean_parts=parts,
                slots=slots,
                session_id=f"sess-b-{req_counter}",
                request_id=f"req-b-{req_counter}-{idx}",
                expected_modalities=expected,
                latency_ms=latency,
            )
            req_counter += 1

    return file_a, dir_b


class StageCPipelineTest(unittest.TestCase):
    def test_e2e_ablation_matrix_dry_run_micro_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file_a, dir_b = _build_fixture_trajectories(root)
            out_dir = root / "stage_c_out"

            report = pipeline.run_stage_c_pipeline(
                output_dir=out_dir,
                trajectory_paths=[file_a, dir_b],
                seed="stage-c-e2e-seed-v1",
                eval_ratio=0.34,
                reward_spec="umminfer_rl_reward:CompositeUmmReward",
                ablation_matrix_path=ROOT / "configs" / "training" / "ablation_matrix.json",
                execution_mode="dry_run_micro_step",
                dry_run_steps=2,
            )

            # Report file and SHA-256 sidecar verification
            report_path = out_dir / "stage_c_e2e_report.json"
            sidecar_path = out_dir / "stage_c_e2e_report.json.sha256"
            self.assertTrue(report_path.is_file())
            self.assertTrue(sidecar_path.is_file())
            actual_sha = ds.sha256_file(report_path)
            self.assertEqual(sidecar_path.read_text(encoding="utf-8").strip(), actual_sha)
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8")), report)

            # Honest claim boundary
            self.assertEqual(report["status"], "COMPLETED")
            self.assertEqual(report["source_mode"], "offline_trajectories")
            self.assertEqual(report["REAL_TRAINING"], "NOT_RUN")
            self.assertEqual(report["claims"]["REAL_TRAINING"], "NOT_RUN")
            self.assertFalse(report["claims"]["model_weights_updated"])
            self.assertFalse(report["claims"]["real_gpu_training_executed"])
            self.assertTrue(report["claims"]["decoupled_from_cpp_serving"])

            # Input trajectory audit
            inp = report["input_trajectories"]
            self.assertEqual(inp["file_count"], 2)
            self.assertEqual(inp["total_records"], 14)
            self.assertEqual(inp["completed_records"], 13)
            self.assertEqual(inp["cancelled_records"], 1)
            self.assertEqual(inp["schema_versions"], {"1": 0, "2": 14})
            self.assertEqual(inp["artifact_verification_mode"], "verified_on_disk")
            self.assertGreater(inp["verified_output_media_refs"], 0)
            for fmeta in inp["files"]:
                self.assertEqual(len(fmeta["sha256"]), 64)

            # SFT & Preference pack checks + zero leakage
            sft_pack = report["sft_pack"]
            self.assertEqual(sft_pack["record_count"], 13)
            self.assertEqual(
                sft_pack["split_counts"]["train"] + sft_pack["split_counts"]["eval"], 13
            )
            self.assertEqual(sft_pack["leakage_count"], 0)
            self.assertTrue(sft_pack["audit_passed"])
            self.assertTrue((out_dir / "sft" / "manifest.json").is_file())
            self.assertTrue((out_dir / "sft" / "manifest.json.sha256").is_file())

            pref_pack = report["preference_pack"]
            self.assertEqual(pref_pack["reward_spec"], "umminfer_rl_reward:CompositeUmmReward")
            self.assertEqual(pref_pack["reward_name"], "umm-composite-multiobjective-v1")
            self.assertGreater(pref_pack["dpo_pair_count"], 0)
            self.assertGreater(pref_pack["grpo_group_count"], 0)
            self.assertEqual(pref_pack["leakage_count"], 0)
            self.assertTrue(pref_pack["audit_passed"])
            self.assertTrue((out_dir / "preference" / "dpo_train.jsonl").is_file())
            self.assertTrue((out_dir / "preference" / "grpo_train.jsonl").is_file())
            self.assertTrue((out_dir / "preference" / "manifest.json.sha256").is_file())

            self.assertEqual(report["leakage_audit"]["total_leakage_count"], 0)
            self.assertTrue(report["leakage_audit"]["passed"])
            self.assertTrue(report["joint_pack"]["cross_pack_sha256_verified"])

            # Full 8-arm ablation matrix verification
            matrix = json.loads(
                (ROOT / "configs" / "training" / "ablation_matrix.json").read_text(encoding="utf-8")
            )
            expected_arms = set(matrix["arms"].keys())
            self.assertEqual(set(report["trainer_runs"].keys()), expected_arms)

            for arm_id, run in report["trainer_runs"].items():
                self.assertEqual(run["execution_mode"], "dry_run_micro_step")
                self.assertEqual(run["REAL_TRAINING"], "NOT_RUN (CPU_REFERENCE_MICRO_STEP)")
                self.assertTrue(run["all_losses_finite"])
                self.assertGreater(run["step_count"], 0)
                self.assertEqual(len(run["manifest_sha256"]), 64)
                self.assertEqual(
                    run["bound_sft_manifest_sha256"], sft_pack["manifest_sha256"]
                )
                self.assertEqual(
                    run["bound_preference_manifest_sha256"], pref_pack["manifest_sha256"]
                )
                self.assertEqual(
                    run["bound_dataset_manifest_sha256"],
                    report["joint_pack"]["manifest_sha256"],
                )
                for st_name, st_metrics in run["stage_metrics"].items():
                    self.assertTrue(st_metrics["loss_finite"], f"{arm_id}:{st_name}")
                    if st_metrics["steps_executed"] > 0:
                        self.assertTrue(math.isfinite(st_metrics["initial_loss"]))
                        self.assertTrue(math.isfinite(st_metrics["final_loss"]))

            # Single-variable ablation masking check: ablate_visual_routing masks tokens vs joint_full
            full_cov = report["trainer_runs"]["joint_full"]["span_mask_coverage"]
            route_cov = report["trainer_runs"]["ablate_visual_routing"]["span_mask_coverage"]
            self.assertEqual(full_cov["masked_tokens"], 0)
            self.assertGreater(route_cov["masked_tokens"], 0)
            self.assertLess(
                route_cov["active_token_ratio"], full_cov["active_token_ratio"]
            )
            self.assertEqual(
                report["trainer_runs"]["sft_only"]["active_stages"], ["sft"]
            )
            self.assertEqual(
                report["trainer_runs"]["sft_plus_dpo"]["active_stages"], ["sft", "dpo"]
            )
            self.assertEqual(
                report["trainer_runs"]["sft_plus_grpo"]["active_stages"], ["sft", "grpo"]
            )

    def test_single_config_and_plan_and_gpu_launch_script_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file_a, dir_b = _build_fixture_trajectories(root)

            # Single SFT config in plan mode
            plan_out = root / "sft_plan_out"
            plan_report = pipeline.run_stage_c_pipeline(
                output_dir=plan_out,
                trajectory_paths=[file_a, dir_b],
                config_path=ROOT / "configs" / "training" / "joint_sft_default.json",
                execution_mode="plan",
            )
            self.assertEqual(plan_report["REAL_TRAINING"], "NOT_RUN")
            self.assertEqual(list(plan_report["trainer_runs"].keys()), ["joint_full"])
            run = plan_report["trainer_runs"]["joint_full"]
            self.assertEqual(run["training_mode"], "sft")
            self.assertEqual(run["execution_mode"], "plan")
            self.assertEqual(run["step_count"], 0)
            self.assertEqual(
                run["bound_dataset_manifest_sha256"],
                plan_report["sft_pack"]["manifest_sha256"],
            )

            # Single preference config in gpu_launch_script mode
            gpu_out = root / "gpu_script_out"
            gpu_report = pipeline.run_stage_c_pipeline(
                output_dir=gpu_out,
                trajectory_paths=[file_a, dir_b],
                config_path=ROOT / "configs" / "training" / "joint_dpo_grpo_default.json",
                execution_mode="gpu_launch_script",
            )
            self.assertEqual(gpu_report["REAL_TRAINING"], "NOT_RUN")
            run_gpu = gpu_report["trainer_runs"]["joint_full"]
            self.assertEqual(run_gpu["execution_mode"], "gpu_launch_script")
            self.assertEqual(run_gpu["step_count"], 0)
            self.assertTrue(
                (gpu_out / "trainer_runs" / "joint_full" / "launch_train.sh").is_file()
            )

    def test_corrupt_and_tampered_trajectories_rejected_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file_a, _ = _build_fixture_trajectories(root)
            original_lines = file_a.read_text(encoding="utf-8").splitlines()

            # 1. Malformed JSON line
            bad_json = root / "bad_json.jsonl"
            bad_json.write_text(original_lines[0] + "\n{not valid json\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                pipeline.run_stage_c_pipeline(
                    output_dir=root / "out_bad_json",
                    trajectory_paths=[bad_json],
                )
            self.assertFalse((root / "out_bad_json" / "stage_c_e2e_report.json").exists())

            # 2. Unsupported schema_version
            bad_ver_dir = root / "bad_ver"
            bad_ver_dir.mkdir()
            rec = json.loads(original_lines[0])
            rec["schema_version"] = 99
            bad_ver = bad_ver_dir / "traj.jsonl"
            bad_ver.write_text(json.dumps(rec) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported schema_version"):
                pipeline.run_stage_c_pipeline(
                    output_dir=root / "out_bad_ver",
                    trajectory_paths=[bad_ver],
                )

            # 3. Tampered clean_text_sha256
            tampered_hash = root / "tampered_hash.jsonl"
            rec2 = json.loads(original_lines[0])
            rec2["clean_text_sha256"] = "0" * 64
            tampered_hash.write_text(json.dumps(rec2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "integrity failed"):
                pipeline.run_stage_c_pipeline(
                    output_dir=root / "out_tampered_hash",
                    trajectory_paths=[tampered_hash],
                )

            # 4. Tampered CAS artifact on disk
            cas_dir = file_a.parent / "artifacts"
            png_files = list(cas_dir.glob("*.png"))
            self.assertTrue(png_files)
            png_files[0].write_bytes(b"corrupted-png-bytes")
            with self.assertRaisesRegex(ValueError, "artifact_tampered|SHA-256"):
                pipeline.run_stage_c_pipeline(
                    output_dir=root / "out_tampered_cas",
                    trajectory_paths=[file_a],
                    strict_artifacts=True,
                )

    def test_sse_to_schema_v2_converter_pure_text_and_multimodal_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl_path = root / "captured.jsonl"
            cas_dir = root / "artifacts"
            recorder = traj.TrajectoryRecorder(jsonl_path, artifact_dir=cas_dir)

            # Case 1: Pure-text 0-slot SSE turn (allow_text_only: true)
            pure_text_events = [
                fixture.envelope(0, "vlm_delta", "Proof: assume sqrt(2) = p/q. "),
                fixture.envelope(1, "vlm_delta", "Both p and q are even: contradiction."),
                fixture.envelope(
                    2,
                    "complete",
                    "Proof: assume sqrt(2) = p/q. Both p and q are even: contradiction.",
                    legs=[],
                ),
            ]
            pure_req = {
                "model": "qwen3-native",
                "stream": True,
                "messages": [{"role": "user", "content": "Prove sqrt(2) is irrational."}],
                "modalities": ["text", "image", "video"],
                "narrative": {
                    "enabled": True,
                    "visual_mode": "auto",
                    "max_slots": 2,
                    "allow_text_only": True,
                },
            }
            rec_text = pipeline.sse_bytes_to_v2_record(
                fixture.sse_bytes(pure_text_events),
                request_body=pure_req,
                recorder=recorder,
                session_id="live-pure-text-0",
                ablation_metadata={
                    "prompt_id": "pure_text",
                    "expected_visual_modalities": [],
                    "latency": {"ttft_ms": 85.2, "terminal_ms": 420.5},
                },
            )
            self.assertEqual(rec_text["schema_version"], 2)
            self.assertEqual(rec_text["terminal_state"], "completed")
            self.assertEqual(rec_text["visual_slots"], [])
            self.assertEqual(rec_text["artifacts"], [])
            self.assertEqual(
                rec_text["clean_text"],
                "Proof: assume sqrt(2) = p/q. Both p and q are even: contradiction.",
            )

            # Case 2: Multi-chunk visual slots (image + video) + streaming audio turn
            multi_req = {
                "model": "qwen3-native",
                "stream": True,
                "messages": [{"role": "user", "content": "Tell a lantern story at dawn."}],
                "modalities": ["text", "image", "video", "audio"],
                "narrative": {
                    "enabled": True,
                    "visual_mode": "auto",
                    "max_slots": 3,
                    "allow_text_only": False,
                },
                "audio": {"voice": "Chelsie", "format": "wav", "stream": True},
            }
            rec_multi = pipeline.sse_bytes_to_v2_record(
                fixture.sse_bytes(fixture.narrative_events()),
                request_body=multi_req,
                recorder=recorder,
                session_id="live-multi-0",
                ablation_metadata={
                    "prompt_id": "lantern_story",
                    "expected_visual_modalities": ["image", "video"],
                    "latency": {"ttft_ms": 120.0, "ttfa_ms": 340.0, "terminal_ms": 2150.0},
                },
            )
            self.assertEqual(rec_multi["schema_version"], 2)
            self.assertEqual(rec_multi["terminal_state"], "completed")
            self.assertEqual(rec_multi["clean_text"], fixture.CLEAN_TEXT)
            self.assertEqual(len(rec_multi["visual_slots"]), 3)
            self.assertEqual(len(rec_multi["artifacts"]), 3)

            # Case 3: explicit same-request image -> I2V linkage. The source
            # digest stays out of visual_slot and resolves on artifact/terminal.
            linked_req = dict(multi_req)
            linked_req["narrative"] = {
                "enabled": True,
                "visual_mode": "auto",
                "max_slots": 2,
                "continuity_mode": "i2v_chain",
            }
            rec_linked = pipeline.sse_bytes_to_v2_record(
                fixture.sse_bytes(fixture.linked_events(delayed=True)),
                request_body=linked_req,
                recorder=recorder,
                session_id="live-linked-0",
                ablation_metadata={
                    "prompt_id": "linked_lantern",
                    "expected_visual_modalities": ["image", "video"],
                    "latency": {"ttft_ms": 100.0, "terminal_ms": 2400.0},
                },
            )
            self.assertEqual(rec_linked["visual_slots"][1]["anchor_slot_index"], 0)
            self.assertTrue(rec_linked["visual_slots"][1]["anchor_explicit"])
            self.assertNotIn("anchor_sha256", rec_linked["visual_slots"][1])
            linked_video = next(item for item in rec_linked["artifacts"]
                                if item["media_type"] == "video")
            linked_image = next(item for item in rec_linked["artifacts"]
                                if item["media_type"] == "image")
            self.assertEqual(linked_video["anchor_sha256"], linked_image["sha256"])

            # Verify zero embedded base64 data and exact CAS SHA-256 on disk
            for event in rec_multi["events"]:
                if event["type"] == "artifact":
                    self.assertNotIn("data", event["artifact"])
                    self.assertIn("ref", event["artifact"])
            media_by_type = {item["media_type"]: item for item in rec_multi["artifacts"]}
            self.assertEqual(set(media_by_type.keys()), {"image", "video", "audio"})
            expected_raws = {
                "image": fixture.PNG,
                "video": fixture.MP4,
                "audio": fixture.PCM,
            }
            for mtype, expected_raw in expected_raws.items():
                art = media_by_type[mtype]
                cas_file = cas_dir / art["ref"]
                self.assertTrue(cas_file.is_file(), f"missing CAS file {cas_file}")
                disk_bytes = cas_file.read_bytes()
                self.assertEqual(disk_bytes, expected_raw)
                self.assertEqual(
                    hashlib.sha256(disk_bytes).hexdigest(), art["sha256"]
                )

            # Strictly verify the entire captured JSONL via pipeline validator
            _, summary = pipeline.validate_trajectory_inputs(
                [jsonl_path], artifact_dir=cas_dir, strict_artifacts=True
            )
            self.assertEqual(summary["total_records"], 3)
            self.assertEqual(summary["schema_versions"], {"1": 0, "2": 3})
            self.assertEqual(summary["verified_output_media_refs"], 5)

            # Feed converted live-SSE JSONL straight into the Stage C pipeline!
            out_dir = root / "converted_pipeline_out"
            report = pipeline.run_stage_c_pipeline(
                output_dir=out_dir,
                trajectory_paths=[jsonl_path],
                artifact_dir=cas_dir,
                strict_artifacts=True,
                config_path=ROOT / "configs" / "training" / "joint_sft_default.json",
                execution_mode="dry_run_micro_step",
                dry_run_steps=1,
            )
            self.assertEqual(report["sft_pack"]["record_count"], 3)
            self.assertEqual(report["leakage_audit"]["total_leakage_count"], 0)
            self.assertEqual(report["REAL_TRAINING"], "NOT_RUN")

    def test_determinism_across_repeated_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file_a, dir_b = _build_fixture_trajectories(root)
            out1 = root / "run1"
            out2 = root / "run2"

            kwargs = dict(
                trajectory_paths=[file_a, dir_b],
                seed="deterministic-stage-c-seed",
                eval_ratio=0.34,
                reward_spec="umminfer_rl_reward:CompositeUmmReward",
                ablation_matrix_path=ROOT / "configs" / "training" / "ablation_matrix.json",
                execution_mode="dry_run_micro_step",
                dry_run_steps=2,
            )
            report1 = pipeline.run_stage_c_pipeline(output_dir=out1, **kwargs)
            report2 = pipeline.run_stage_c_pipeline(output_dir=out2, **kwargs)

            self.assertEqual(report1, report2)
            self.assertEqual(_tree_bytes(out1), _tree_bytes(out2))

    def test_url_hygiene_and_cli_isolated_mode_and_no_forbidden_tokens(self):
        # Transport hygiene on validate_server_url
        self.assertEqual(
            pipeline.validate_server_url("http://127.0.0.1:43301/"),
            "http://127.0.0.1:43301",
        )
        self.assertEqual(
            pipeline.validate_server_url("https://example.invalid/v1/"),
            "https://example.invalid/v1",
        )
        with self.assertRaises(ValueError):
            pipeline.validate_server_url("http://user:pass@127.0.0.1:43301")
        with self.assertRaises(ValueError):
            pipeline.validate_server_url("http://198.51.100.10:43301")
        self.assertEqual(
            pipeline.validate_server_url(
                "http://198.51.100.10:43301", allow_remote_http=True
            ),
            "http://198.51.100.10:43301",
        )

        # CLI isolated mode execution (python3 -I)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file_a, dir_b = _build_fixture_trajectories(root)
            out_dir = root / "cli_out"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(ROOT / "tools" / "umminfer_stage_c_pipeline.py"),
                    "--from-trajectories",
                    str(file_a),
                    str(dir_b),
                    "--out",
                    str(out_dir),
                    "--ablation-matrix",
                    str(ROOT / "configs" / "training" / "ablation_matrix.json"),
                    "--ablation",
                    "joint_full",
                    "--execution-mode",
                    "dry_run_micro_step",
                    "--dry-run-steps",
                    "1",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            parsed = json.loads(proc.stdout)
            self.assertEqual(parsed["status"], "COMPLETED")
            self.assertEqual(parsed["REAL_TRAINING"], "NOT_RUN")

        # Source hygiene: zero internal hostnames, IPs, usernames, or local home paths
        for rel in (
            "tools/umminfer_stage_c_pipeline.py",
            "tests/stage_c_pipeline_test.py",
            "docs/joint-training.md",
        ):
            text = (ROOT / rel).read_text(encoding="utf-8")
            for token in FORBIDDEN_TOKENS:
                self.assertNotIn(token, text, f"forbidden token {token!r} in {rel}")


if __name__ == "__main__":
    unittest.main()
