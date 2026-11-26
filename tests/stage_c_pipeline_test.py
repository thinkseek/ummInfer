#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deterministic CPU-only offline unit & E2E tests for Stage C Pipeline Driver.

Usage: python3 -m unittest tests/stage_c_pipeline_test.py -v
Zero network calls, zero GPU/torch/numpy, zero serving processes.
"""

from contextlib import contextmanager
import ast
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import narrative_rl_fixtures as fixture  # noqa: E402
import token_export_test as token_fixture  # noqa: E402
import umminfer_rl_trajectory as traj  # noqa: E402
import umminfer_sft_dataset as ds  # noqa: E402
import umminfer_stage_c_pipeline as pipeline  # noqa: E402
import umminfer_token_export as token_export  # noqa: E402

REAL_TASK42_BUNDLE = ROOT / "tests" / "fixtures" / "behavior_token_alignment" / "task42"
REAL_TASK42_ACCEPTED = ROOT / "docs" / "evidence" / "behavior-token-alignment-real-cpu.json"

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

    def test_default_off_parity_no_optional_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file_a, dir_b = _build_fixture_trajectories(root)
            out_dir = root / "default_off_out"
            report = pipeline.run_stage_c_pipeline(
                output_dir=out_dir,
                trajectory_paths=[file_a, dir_b],
                ablation_matrix_path=ROOT / "configs" / "training" / "ablation_matrix.json",
                ablation_override="joint_full",
                execution_mode="dry_run_micro_step",
                dry_run_steps=1,
            )
            self.assertNotIn("behavior_logprobs_capture", report)
            self.assertNotIn("behavior_token_alignment", report)
            self.assertNotIn("dpo_export_preflight", report)
            self.assertNotIn("online_grpo_preflight", report)
            self.assertNotIn("independent_online_audit", report)
            self.assertNotIn("independent_online_audit_task42", report)
            self.assertFalse((out_dir / "independent_online_audit").exists())
            self.assertNotIn("umminfer_online_evidence_audit", sys.modules)
            self.assertNotIn("torch", sys.modules)
            self.assertNotIn("transformers", sys.modules)
            self.assertNotIn("numpy", sys.modules)
            self.assertNotIn("mlx", sys.modules)

            # Static AST check: pipeline must NEVER import umminfer_online_evidence_audit or ML libs.
            tree = ast.parse(
                (ROOT / "tools" / "umminfer_stage_c_pipeline.py").read_text(encoding="utf-8")
            )
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            for forbidden_mod in (
                "umminfer_online_evidence_audit",
                "torch",
                "transformers",
                "numpy",
                "mlx",
            ):
                self.assertNotIn(forbidden_mod, imported)

            # Isolated fresh process: default-off must not import online-GRPO, auditor, or ML modules.
            probe = (
                "import json, sys\n"
                f"sys.path.insert(0, {str(ROOT / 'tools')!r})\n"
                "import umminfer_stage_c_pipeline as p\n"
                f"r = p.run_stage_c_pipeline(output_dir={str(root / 'probe_out')!r}, "
                f"trajectory_paths=[{str(file_a)!r}, {str(dir_b)!r}], "
                f"ablation_matrix_path={str(ROOT / 'configs/training/ablation_matrix.json')!r}, "
                "ablation_override='joint_full', execution_mode='dry_run_micro_step', "
                "dry_run_steps=1)\n"
                "blocked = [m for m in ('umminfer_online_grpo_preflight', "
                "'umminfer_online_evidence_audit', 'torch', "
                "'transformers', 'numpy', 'mlx') if m in sys.modules]\n"
                "print(json.dumps({'has_key': 'online_grpo_preflight' in r, "
                "'has_audit': 'independent_online_audit' in r, "
                "'has_audit_task42': 'independent_online_audit_task42' in r, "
                "'blocked': blocked}))\n"
            )
            proc = subprocess.run(
                [sys.executable, "-I", "-c", probe],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            probe_res = json.loads(proc.stdout)
            self.assertFalse(probe_res["has_key"])
            self.assertFalse(probe_res["has_audit"])
            self.assertFalse(probe_res["has_audit_task42"])
            self.assertEqual(probe_res["blocked"], [])

    def test_capture_behavior_logprobs_fail_closed_and_mock_live_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file_a, _ = _build_fixture_trajectories(root)

            # 1. Fail-closed without --capture-live (API and CLI exit 2)
            with self.assertRaisesRegex(ValueError, "--capture-behavior-logprobs requires --capture-live"):
                pipeline.run_stage_c_pipeline(
                    output_dir=root / "bad_api",
                    trajectory_paths=[file_a],
                    capture_behavior_logprobs=True,
                )
            self.assertFalse((root / "bad_api" / "stage_c_e2e_report.json").exists())

            proc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(ROOT / "tools" / "umminfer_stage_c_pipeline.py"),
                    "--from-trajectories",
                    str(file_a),
                    "--out",
                    str(root / "bad_cli"),
                    "--capture-behavior-logprobs",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 2, proc.stdout)
            self.assertFalse((root / "bad_cli" / "stage_c_e2e_report.json").exists())

            # 2. Mock live HTTP SSE server with exposed + not_exposed rollouts
            exposed_behavior = {
                "format": "umminfer-behavior-logprobs-v1",
                "tokenizer_sha256": "a" * 64,
                "token_ids": [11, 12, 13],
                "logprobs": [-0.125, -0.25, -0.5],
                "mask": [1, 1, 1],
            }

            def _make_sse_with_behavior(text_parts, behavior_block=None):
                events = [
                    fixture.envelope(i, "vlm_delta", part, request_id=f"req-mock-{len(text_parts)}")
                    for i, part in enumerate(text_parts)
                ]
                events.append(
                    fixture.envelope(
                        len(text_parts),
                        "complete",
                        "".join(text_parts),
                        legs=[],
                        request_id=f"req-mock-{len(text_parts)}",
                    )
                )
                frames = fixture.sse_bytes(events).split(b"\n\n")
                if behavior_block is not None:
                    for idx in range(len(frames) - 1, -1, -1):
                        if frames[idx].startswith(b"data: ") and frames[idx] != b"data: [DONE]":
                            chunk = json.loads(frames[idx][6:].decode("utf-8"))
                            chunk["behavior"] = copy.deepcopy(behavior_block)
                            frames[idx] = b"data: " + traj.canonical_json(chunk)
                            break
                return b"\n\n".join(frames)

            @contextmanager
            def _mock_server(responses):
                recorded_bodies = []
                state = {"idx": 0}

                class _Handler(BaseHTTPRequestHandler):
                    def log_message(self, *args):
                        pass

                    def do_POST(self):
                        length = int(self.headers.get("Content-Length") or 0)
                        body = json.loads(self.rfile.read(length).decode("utf-8"))
                        recorded_bodies.append(body)
                        hdr_val, payload = responses[state["idx"]]
                        state["idx"] += 1
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Content-Length", str(len(payload)))
                        if hdr_val is not None:
                            self.send_header("X-UmmInfer-Behavior-Evidence", hdr_val)
                        self.send_header("Connection", "close")
                        self.end_headers()
                        self.wfile.write(payload)
                        self.close_connection = True

                server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    yield f"http://127.0.0.1:{server.server_port}", recorded_bodies
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)

            mini_suite = (
                {
                    "prompt_id": "mock_behavior_case",
                    "prompt": "Prove concisely that sqrt(2) is irrational.",
                    "expected_visual_modalities": [],
                    "rollouts": (
                        {
                            "rollout_label": "exposed_rollout",
                            "temperature": 0.0,
                            "seed": 42,
                            "max_tokens": 64,
                            "modalities": ["text"],
                            "narrative": {
                                "enabled": True,
                                "visual_mode": "auto",
                                "max_slots": 1,
                                "allow_text_only": True,
                            },
                        },
                        {
                            "rollout_label": "not_exposed_rollout",
                            "temperature": 0.7,
                            "seed": 43,
                            "max_tokens": 64,
                            "modalities": ["text"],
                            "narrative": {
                                "enabled": True,
                                "visual_mode": "auto",
                                "max_slots": 1,
                                "allow_text_only": True,
                            },
                        },
                    ),
                },
            )

            sse_exposed = _make_sse_with_behavior(["Proof line one. ", "QED."], exposed_behavior)
            sse_not_exposed = _make_sse_with_behavior(["Alternate proof text."], None)

            with _mock_server([
                ("exposed", sse_exposed),
                ("not_exposed", sse_not_exposed),
            ]) as (base_url, seen_bodies):
                report = pipeline.run_stage_c_pipeline(
                    output_dir=root / "live_behavior_on",
                    capture_live_url=base_url,
                    live_prompt_suite=mini_suite,
                    capture_behavior_logprobs=True,
                    config_path=ROOT / "configs" / "training" / "joint_sft_default.json",
                    execution_mode="dry_run_micro_step",
                    dry_run_steps=1,
                )
                self.assertEqual(len(seen_bodies), 2)
                for req_body in seen_bodies:
                    self.assertEqual(req_body.get("behavior"), {"logprobs": True})

                bc = report["behavior_logprobs_capture"]
                self.assertTrue(bc["requested"])
                self.assertEqual(bc["exposed_count"], 1)
                self.assertEqual(bc["not_exposed_count"], 1)
                self.assertEqual(bc["absent_count"], 0)
                self.assertEqual(bc["total_requests"], 2)
                self.assertEqual(bc["cases"][0]["behavior_evidence_header"], "exposed")
                self.assertEqual(bc["cases"][0]["behavior"], exposed_behavior)
                self.assertEqual(
                    bc["cases"][0]["behavior_sha256"],
                    traj.sha256_bytes(traj.canonical_json(exposed_behavior)),
                )
                # Zero fabrication on not_exposed!
                self.assertEqual(bc["cases"][1]["behavior_evidence_header"], "not_exposed")
                self.assertIsNone(bc["cases"][1]["behavior"])
                self.assertIsNone(bc["cases"][1]["behavior_sha256"])

            # Default live capture (flag OFF) sends NO behavior field and omits report section
            with _mock_server([
                (None, sse_not_exposed),
                (None, sse_not_exposed),
            ]) as (base_url, seen_bodies_off):
                report_off = pipeline.run_stage_c_pipeline(
                    output_dir=root / "live_behavior_off",
                    capture_live_url=base_url,
                    live_prompt_suite=mini_suite,
                    capture_behavior_logprobs=False,
                    config_path=ROOT / "configs" / "training" / "joint_sft_default.json",
                    execution_mode="dry_run_micro_step",
                    dry_run_steps=1,
                )
                self.assertTrue(all("behavior" not in b for b in seen_bodies_off))
                self.assertNotIn("behavior_logprobs_capture", report_off)

            # Fail closed if server says exposed without final behavior block
            with _mock_server([
                ("exposed", sse_not_exposed),
                ("not_exposed", sse_not_exposed),
            ]) as (base_url, _):
                with self.assertRaisesRegex(ValueError, "requires exactly one behavior block"):
                    pipeline.run_stage_c_pipeline(
                        output_dir=root / "live_bad_exposed",
                        capture_live_url=base_url,
                        live_prompt_suite=mini_suite,
                        capture_behavior_logprobs=True,
                        config_path=ROOT / "configs" / "training" / "joint_sft_default.json",
                    )
                self.assertFalse((root / "live_bad_exposed" / "stage_c_e2e_report.json").exists())

            # Fail closed if server says not_exposed yet emits a behavior block
            with _mock_server([
                ("not_exposed", sse_exposed),
                ("not_exposed", sse_not_exposed),
            ]) as (base_url, _):
                with self.assertRaisesRegex(ValueError, "unexpectedly contained a behavior block"):
                    pipeline.run_stage_c_pipeline(
                        output_dir=root / "live_bad_not_exposed",
                        capture_live_url=base_url,
                        live_prompt_suite=mini_suite,
                        capture_behavior_logprobs=True,
                        config_path=ROOT / "configs" / "training" / "joint_sft_default.json",
                    )
                self.assertFalse((root / "live_bad_not_exposed" / "stage_c_e2e_report.json").exists())

    def test_verify_behavior_alignment_real_task42_and_tamper_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file_a, dir_b = _build_fixture_trajectories(root)

            # 1. Combined pipeline + real Task #42 alignment bundle (stdlib-only, zero GPU/ML)
            out_combined = root / "combined_align"
            report = pipeline.run_stage_c_pipeline(
                output_dir=out_combined,
                trajectory_paths=[file_a, dir_b],
                config_path=ROOT / "configs" / "training" / "joint_sft_default.json",
                execution_mode="dry_run_micro_step",
                dry_run_steps=1,
                verify_behavior_alignment=REAL_TASK42_BUNDLE,
            )
            align_section = report["behavior_token_alignment"]
            self.assertEqual(align_section["status"], "PASS")
            self.assertEqual(align_section["evidence_class"], "RETAINED_REAL_TASK42")
            self.assertEqual(align_section["validation_level"], "STRUCTURAL")
            self.assertEqual(align_section["tokenizer_files"], "NOT_RUN")
            self.assertEqual(align_section["tokenizer_reencode"], "NOT_RUN")
            self.assertEqual(align_section["raw_sse_fresh_replay"], "PASS_BYTE_IDENTICAL")
            self.assertEqual(align_section["accepted_case_count"], 2)
            self.assertEqual(align_section["excluded_case_count"], 3)
            self.assertEqual(align_section["sampled_token_count"], 61)
            self.assertEqual(align_section["masked_token_count"], 49)
            self.assertEqual(len(align_section["bundle_sha256"]), 64)
            self.assertEqual(align_section["claims"]["new_inference"], "NOT_RUN")
            self.assertEqual(align_section["claims"]["optimizer_step"], "NOT_RUN")
            self.assertEqual(align_section["claims"]["new_model_inference"], "NOT_RUN")
            self.assertEqual(align_section["claims"]["online_optimizer"], "NOT_RUN")
            self.assertNotIn("torch", sys.modules)
            self.assertNotIn("transformers", sys.modules)

            # 2. Standalone CLI invocation against real Task #42 bundle (python3 -I)
            out_cli = root / "standalone_align_cli"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(ROOT / "tools" / "umminfer_stage_c_pipeline.py"),
                    "--verify-behavior-alignment",
                    str(REAL_TASK42_BUNDLE),
                    "--out",
                    str(out_cli),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            cli_report = json.loads(proc.stdout)
            self.assertEqual(cli_report["status"], "COMPLETED")
            self.assertEqual(cli_report["source_mode"], "behavior_alignment_only")
            self.assertEqual(cli_report["REAL_TRAINING"], "NOT_RUN")
            self.assertEqual(cli_report["behavior_token_alignment"]["status"], "PASS")
            self.assertEqual(cli_report["behavior_token_alignment"]["sampled_token_count"], 61)
            self.assertEqual(
                cli_report["behavior_token_alignment"]["claims"]["new_inference"], "NOT_RUN"
            )
            self.assertEqual(
                cli_report["behavior_token_alignment"]["claims"]["optimizer_step"], "NOT_RUN"
            )
            self.assertTrue((out_cli / "stage_c_e2e_report.json").is_file())
            self.assertTrue((out_cli / "stage_c_e2e_report.json.sha256").is_file())

            # 3. Fail-closed on --alignment-reencode without --alignment-tokenizer-dir
            proc_bad_reenc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(ROOT / "tools" / "umminfer_stage_c_pipeline.py"),
                    "--verify-behavior-alignment",
                    str(REAL_TASK42_BUNDLE),
                    "--alignment-reencode",
                    "--out",
                    str(root / "bad_reenc"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_bad_reenc.returncode, 2, proc_bad_reenc.stdout)
            self.assertFalse((root / "bad_reenc" / "stage_c_e2e_report.json").exists())

            # 4. Fail-closed rejection on copied tampered Task #42 bundle (CAS PNG + raw SSE)
            for tamper_kind in ("cas_png", "raw_sse"):
                with self.subTest(tamper=tamper_kind):
                    tampered_dir = root / f"tampered_{tamper_kind}"
                    shutil.copytree(REAL_TASK42_BUNDLE, tampered_dir)
                    if tamper_kind == "cas_png":
                        png = next((tampered_dir / "live" / "artifacts").glob("*.png"))
                        raw = bytearray(png.read_bytes())
                        raw[-1] ^= 1
                        png.write_bytes(raw)
                    else:
                        sse_file = next((tampered_dir / "live" / "artifacts").glob("*.sse"))
                        raw = bytearray(sse_file.read_bytes())
                        raw[10] ^= 1
                        sse_file.write_bytes(raw)

                    bad_out = root / f"out_tampered_{tamper_kind}"
                    with self.assertRaisesRegex(ValueError, "behavior token alignment"):
                        pipeline.run_stage_c_pipeline(
                            output_dir=bad_out,
                            verify_behavior_alignment=tampered_dir,
                        )
                    self.assertFalse((bad_out / "stage_c_e2e_report.json").exists())

                    cli_bad_out = root / f"cli_tampered_{tamper_kind}"
                    proc_tamper = subprocess.run(
                        [
                            sys.executable,
                            "-I",
                            str(ROOT / "tools" / "umminfer_stage_c_pipeline.py"),
                            "--verify-behavior-alignment",
                            str(tampered_dir),
                            "--out",
                            str(cli_bad_out),
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(proc_tamper.returncode, 2, proc_tamper.stdout)
                    self.assertFalse((cli_bad_out / "stage_c_e2e_report.json").exists())

    def test_dpo_export_preflight_pass_and_missing_export_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file_a, dir_b = _build_fixture_trajectories(root)

            # 1. Fail-closed when token_export metadata is missing (never falls back to legacy!)
            missing_out = root / "dpo_missing_api"
            with self.assertRaisesRegex(ValueError, "refusing per-side legacy fallback"):
                pipeline.run_stage_c_pipeline(
                    output_dir=missing_out,
                    trajectory_paths=[file_a, dir_b],
                    config_path=ROOT / "configs" / "training" / "joint_dpo_grpo_default.json",
                    execution_mode="dry_run_micro_step",
                    dry_run_steps=1,
                    dpo_export_preflight=True,
                )
            self.assertFalse((missing_out / "stage_c_e2e_report.json").exists())

            missing_cli_out = root / "dpo_missing_cli"
            proc_missing = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(ROOT / "tools" / "umminfer_stage_c_pipeline.py"),
                    "--from-trajectories",
                    str(file_a),
                    str(dir_b),
                    "--out",
                    str(missing_cli_out),
                    "--config",
                    str(ROOT / "configs" / "training" / "joint_dpo_grpo_default.json"),
                    "--dpo-export-preflight",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_missing.returncode, 2, proc_missing.stdout)
            self.assertFalse((missing_cli_out / "stage_c_e2e_report.json").exists())

            # 2. Build deterministic v2 token_export trajectories and run CPU preflight PASS
            tok = token_fixture.V2FixtureTokenizer()
            tok_dir = token_fixture._v2_tokenizer_dir(root / "v2_tokenizer")
            exported_file = file_a.with_name("trajectories_a.tokens.jsonl")
            raw_records = [
                json.loads(line)
                for line in file_a.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            exported_records = [
                token_export.attach_token_export(
                    rec,
                    tok,
                    tokenizer_dir=tok_dir,
                    tokenizer_id="fixture-bytelevel-nfc",
                    tokenizer_revision="fixture-revision-2",
                    tokenization_version="fixture-tokenization-v2",
                    sft_target=ds.reconstruct_director_target(rec)[0],
                    format_version=2,
                )
                for rec in raw_records
            ]
            exported_file.write_bytes(
                b"".join(traj.canonical_json(rec) + b"\n" for rec in exported_records)
            )

            # API pass (with strong fixture tokenizer check + zero torch import)
            out_api = root / "dpo_preflight_api"
            report = pipeline.run_stage_c_pipeline(
                output_dir=out_api,
                token_export_file=exported_file,
                artifact_dir=file_a.parent / "artifacts",
                strict_artifacts=True,
                eval_ratio=0.0,
                config_path=ROOT / "configs" / "training" / "joint_dpo_grpo_default.json",
                execution_mode="dry_run_micro_step",
                dry_run_steps=1,
                dpo_export_preflight=True,
                dpo_preflight_tokenizer_dir=tok_dir,
                dpo_preflight_tokenizer=tok,
            )
            pf = report["dpo_export_preflight"]
            self.assertEqual(pf["status"], "PASS")
            self.assertEqual(pf["token_source"], "token_export")
            self.assertEqual(pf["EXPORT_DPO_OPTIMIZER"], "NOT_RUN")
            self.assertEqual(pf["REAL_TRAINING"], "NOT_RUN")
            self.assertFalse(pf["torch_imported"])
            self.assertEqual(pf["validation_level"], "STRONG")
            self.assertTrue(pf["tokenizer_files_verified"])
            self.assertTrue(pf["tokenizer_reencoded"])
            self.assertEqual(pf["format_versions"], [2])
            self.assertGreater(pf["validated_pair_count"], 0)
            self.assertEqual(len(pf["preflight_sha256"]), 64)
            self.assertEqual(pf["claims"]["EXPORT_DPO_OPTIMIZER"], "NOT_RUN")
            self.assertEqual(pf["claims"]["real_dpo_optimizer"], "NOT_RUN")
            self.assertFalse(pf["claims"]["legacy_fallback_used"])
            self.assertNotIn("torch", sys.modules)
            self.assertNotIn("transformers", sys.modules)

            # CLI pass (pure stdlib via --token-export-file + --dpo-preflight-tokenizer-dir)
            out_cli = root / "dpo_preflight_cli"
            proc_ok = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(ROOT / "tools" / "umminfer_stage_c_pipeline.py"),
                    "--token-export-file",
                    str(exported_file),
                    "--artifact-dir",
                    str(file_a.parent / "artifacts"),
                    "--strict-artifacts",
                    "--eval-ratio",
                    "0.0",
                    "--config",
                    str(ROOT / "configs" / "training" / "joint_dpo_grpo_default.json"),
                    "--execution-mode",
                    "dry_run_micro_step",
                    "--dry-run-steps",
                    "1",
                    "--dpo-export-preflight",
                    "--dpo-preflight-tokenizer-dir",
                    str(tok_dir),
                    "--out",
                    str(out_cli),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_ok.returncode, 0, proc_ok.stderr)
            cli_pf = json.loads(proc_ok.stdout)["dpo_export_preflight"]
            self.assertEqual(cli_pf["status"], "PASS")
            self.assertEqual(cli_pf["EXPORT_DPO_OPTIMIZER"], "NOT_RUN")
            self.assertFalse(cli_pf["torch_imported"])
            self.assertTrue(cli_pf["tokenizer_files_verified"])

            # 3. Fail-closed on partial token_export (one record stripped -> zero fallback!)
            partial_records = copy.deepcopy(exported_records)
            partial_records[0].pop("token_export")
            partial_file = file_a.with_name("trajectories_a.partial.jsonl")
            partial_file.write_bytes(
                b"".join(traj.canonical_json(rec) + b"\n" for rec in partial_records)
            )
            with self.assertRaisesRegex(ValueError, "refusing per-side legacy fallback"):
                pipeline.run_stage_c_pipeline(
                    output_dir=root / "dpo_partial_fail",
                    token_export_file=partial_file,
                    artifact_dir=file_a.parent / "artifacts",
                    strict_artifacts=True,
                    eval_ratio=0.0,
                    config_path=ROOT / "configs" / "training" / "joint_dpo_grpo_default.json",
                    dpo_export_preflight=True,
                )
            self.assertFalse((root / "dpo_partial_fail" / "stage_c_e2e_report.json").exists())

    def test_online_grpo_preflight_cli_validation_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dummy_bundle = root / "bundle"
            dummy_source = root / "source"
            dummy_tok = root / "tok"
            for d in (dummy_bundle, dummy_source, dummy_tok):
                d.mkdir()

            # 1. --verify-online-grpo-preflight without --online-grpo-source-dir
            with self.assertRaisesRegex(
                ValueError, "--verify-online-grpo-preflight requires --online-grpo-source-dir"
            ):
                pipeline.run_stage_c_pipeline(
                    output_dir=root / "api_no_src",
                    verify_online_grpo_preflight=dummy_bundle,
                )
            self.assertFalse((root / "api_no_src" / "stage_c_e2e_report.json").exists())

            proc_no_src = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(ROOT / "tools" / "umminfer_stage_c_pipeline.py"),
                    "--verify-online-grpo-preflight",
                    str(dummy_bundle),
                    "--out",
                    str(root / "cli_no_src"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_no_src.returncode, 2, proc_no_src.stdout)
            self.assertFalse((root / "cli_no_src" / "stage_c_e2e_report.json").exists())

            # 2. --online-grpo-source-dir without --verify-online-grpo-preflight
            file_a, _ = _build_fixture_trajectories(root)
            with self.assertRaisesRegex(
                ValueError, "--online-grpo-source-dir and --online-grpo-tokenizer-dir require"
            ):
                pipeline.run_stage_c_pipeline(
                    output_dir=root / "api_src_only",
                    trajectory_paths=[file_a],
                    online_grpo_source_dir=dummy_source,
                )
            self.assertFalse((root / "api_src_only" / "stage_c_e2e_report.json").exists())

            proc_src_only = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(ROOT / "tools" / "umminfer_stage_c_pipeline.py"),
                    "--from-trajectories",
                    str(file_a),
                    "--online-grpo-source-dir",
                    str(dummy_source),
                    "--out",
                    str(root / "cli_src_only"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_src_only.returncode, 2, proc_src_only.stdout)
            self.assertFalse((root / "cli_src_only" / "stage_c_e2e_report.json").exists())

            # 3. --online-grpo-tokenizer-dir without --verify-online-grpo-preflight
            proc_tok_only = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(ROOT / "tools" / "umminfer_stage_c_pipeline.py"),
                    "--from-trajectories",
                    str(file_a),
                    "--online-grpo-tokenizer-dir",
                    str(dummy_tok),
                    "--out",
                    str(root / "cli_tok_only"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_tok_only.returncode, 2, proc_tok_only.stdout)
            self.assertFalse((root / "cli_tok_only" / "stage_c_e2e_report.json").exists())

    def test_online_grpo_preflight_standalone_and_combined_pass(self):
        import online_grpo_preflight_test as grpo_fixture  # noqa: PLC0415
        import umminfer_online_grpo_preflight as online_grpo  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_bundle, tok_dir = grpo_fixture.make_online_bundle(
                root / "src_exports", version=1, exports=True
            )
            cfg = grpo_fixture.make_config(source_bundle, tok_dir)
            preflight_bundle = root / "preflight_bundle"
            prep = online_grpo.prepare_preflight(
                source_bundle, preflight_bundle, cfg, tokenizer_dir=tok_dir
            )

            before_bundle_bytes = _tree_bytes(preflight_bundle)
            before_source_bytes = _tree_bytes(source_bundle)

            # 1. Standalone CLI pass with --online-grpo-tokenizer-dir (python3 -I)
            out_standalone_1 = root / "standalone_1"
            proc1 = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(ROOT / "tools" / "umminfer_stage_c_pipeline.py"),
                    "--verify-online-grpo-preflight",
                    str(preflight_bundle),
                    "--online-grpo-source-dir",
                    str(source_bundle),
                    "--online-grpo-tokenizer-dir",
                    str(tok_dir),
                    "--out",
                    str(out_standalone_1),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc1.returncode, 0, proc1.stderr)
            report1 = json.loads(proc1.stdout)
            self.assertEqual(report1["status"], "COMPLETED")
            self.assertEqual(report1["source_mode"], "online_grpo_preflight_only")
            self.assertEqual(report1["REAL_TRAINING"], "NOT_RUN")

            og = report1["online_grpo_preflight"]
            self.assertEqual(og["status"], "PASS")
            self.assertEqual(og["verification"], "BYTE_IDENTICAL")
            self.assertEqual(og["run_id"], prep["run_id"])
            self.assertEqual(
                og["manifest_sha256"], ds.sha256_file(preflight_bundle / "manifest.json")
            )
            self.assertEqual(og["reconstructed_file_count"], 7)
            self.assertEqual(len(og["reconstructed_files"]), 7)
            self.assertFalse(og["torch_imported"])
            self.assertTrue(og["tokenizer_dir_provided"])
            self.assertEqual(len(og["report_sha256"]), 64)
            for fname, digest in og["reconstructed_files"].items():
                self.assertNotIn("/", fname)
                self.assertEqual(digest, ds.sha256_file(preflight_bundle / fname))

            claims = og["claims"]
            for key in (
                "real_online_generation",
                "current_reference_forward",
                "online_optimizer",
                "gpu_execution",
                "model_quality",
                "REAL_TRAINING",
            ):
                self.assertEqual(claims[key], "NOT_RUN", key)
            self.assertFalse(claims["model_weights_updated"])
            self.assertFalse(claims["torch_imported"])

            # Path-free check & sidecar verification
            report1_path = out_standalone_1 / "stage_c_e2e_report.json"
            report1_sha_path = out_standalone_1 / "stage_c_e2e_report.json.sha256"
            self.assertTrue(report1_path.is_file())
            self.assertTrue(report1_sha_path.is_file())
            self.assertEqual(
                report1_sha_path.read_text(encoding="utf-8").strip(),
                ds.sha256_file(report1_path),
            )
            self.assertNotIn(str(root), report1_path.read_text(encoding="utf-8"))

            # Determinism across repeated standalone runs
            out_standalone_2 = root / "standalone_2"
            report2 = pipeline.run_stage_c_pipeline(
                output_dir=out_standalone_2,
                verify_online_grpo_preflight=preflight_bundle,
                online_grpo_source_dir=source_bundle,
                online_grpo_tokenizer_dir=tok_dir,
            )
            self.assertEqual(report1, report2)
            self.assertEqual(
                report1_path.read_bytes(),
                (out_standalone_2 / "stage_c_e2e_report.json").read_bytes(),
            )

            # Input bundles must NEVER be mutated
            self.assertEqual(_tree_bytes(preflight_bundle), before_bundle_bytes)
            self.assertEqual(_tree_bytes(source_bundle), before_source_bytes)

            # 2. Standalone without --online-grpo-tokenizer-dir (exports=False bundle)
            no_exp_src, no_exp_tok = grpo_fixture.make_online_bundle(
                root / "src_no_exports", version=1, exports=False
            )
            no_exp_cfg = grpo_fixture.make_config(no_exp_src, no_exp_tok)
            no_exp_bundle = root / "preflight_no_exports"
            no_exp_prep = online_grpo.prepare_preflight(
                no_exp_src, no_exp_bundle, no_exp_cfg
            )
            report_no_tok = pipeline.run_stage_c_pipeline(
                output_dir=root / "standalone_no_tok",
                verify_online_grpo_preflight=no_exp_bundle,
                online_grpo_source_dir=no_exp_src,
            )
            self.assertEqual(report_no_tok["online_grpo_preflight"]["status"], "PASS")
            self.assertEqual(
                report_no_tok["online_grpo_preflight"]["run_id"], no_exp_prep["run_id"]
            )
            self.assertFalse(
                report_no_tok["online_grpo_preflight"]["tokenizer_dir_provided"]
            )

            # 3. Combined trajectory + behavior-alignment + online-GRPO preflight run
            file_a, dir_b = _build_fixture_trajectories(root)
            combined_report = pipeline.run_stage_c_pipeline(
                output_dir=root / "combined_all",
                trajectory_paths=[file_a, dir_b],
                config_path=ROOT / "configs" / "training" / "joint_sft_default.json",
                execution_mode="dry_run_micro_step",
                dry_run_steps=1,
                verify_behavior_alignment=REAL_TASK42_BUNDLE,
                verify_online_grpo_preflight=preflight_bundle,
                online_grpo_source_dir=source_bundle,
                online_grpo_tokenizer_dir=tok_dir,
            )
            self.assertEqual(combined_report["status"], "COMPLETED")
            self.assertEqual(combined_report["source_mode"], "offline_trajectories")
            self.assertEqual(combined_report["behavior_token_alignment"]["status"], "PASS")
            self.assertEqual(combined_report["online_grpo_preflight"]["status"], "PASS")
            self.assertEqual(
                combined_report["online_grpo_preflight"]["verification"], "BYTE_IDENTICAL"
            )
            self.assertEqual(
                combined_report["online_grpo_preflight"]["run_id"], prep["run_id"]
            )
            self.assertEqual(_tree_bytes(preflight_bundle), before_bundle_bytes)
            for mod in ("torch", "transformers", "numpy", "mlx"):
                self.assertNotIn(mod, sys.modules)

    def test_online_grpo_preflight_tampered_bundle_rejected_fail_closed(self):
        import online_grpo_preflight_test as grpo_fixture  # noqa: PLC0415
        import umminfer_online_grpo_preflight as online_grpo  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_bundle, tok_dir = grpo_fixture.make_online_bundle(
                root / "src_tamper", version=2, exports=True
            )
            cfg = grpo_fixture.make_config(source_bundle, tok_dir)
            preflight_bundle = root / "preflight_valid"
            online_grpo.prepare_preflight(
                source_bundle, preflight_bundle, cfg, tokenizer_dir=tok_dir
            )

            # 1. Both raw-hash tamper and consistently-rehashed tamper must fail closed!
            for consistent in (False, True):
                label = "consistent_rehash" if consistent else "raw_hash"
                with self.subTest(tamper=label):
                    tampered = root / f"tampered_{label}"
                    shutil.copytree(preflight_bundle, tampered)
                    obj_path = tampered / "objective_fixtures.json"
                    data = json.loads(obj_path.read_text(encoding="utf-8"))
                    data["groups"][0]["members"][0]["current_policy"]["logprobs"][0] -= 0.01
                    grpo_fixture.write_json(obj_path, data)
                    if consistent:
                        grpo_fixture.refresh_checksums_for_consistent_tamper(tampered)

                    api_out = root / f"api_out_{label}"
                    with self.assertRaisesRegex(ValueError, "online-GRPO preflight"):
                        pipeline.run_stage_c_pipeline(
                            output_dir=api_out,
                            verify_online_grpo_preflight=tampered,
                            online_grpo_source_dir=source_bundle,
                            online_grpo_tokenizer_dir=tok_dir,
                        )
                    self.assertFalse((api_out / "stage_c_e2e_report.json").exists())

                    cli_out = root / f"cli_out_{label}"
                    proc = subprocess.run(
                        [
                            sys.executable,
                            "-I",
                            str(ROOT / "tools" / "umminfer_stage_c_pipeline.py"),
                            "--verify-online-grpo-preflight",
                            str(tampered),
                            "--online-grpo-source-dir",
                            str(source_bundle),
                            "--online-grpo-tokenizer-dir",
                            str(tok_dir),
                            "--out",
                            str(cli_out),
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(proc.returncode, 2, proc.stdout)
                    self.assertFalse((cli_out / "stage_c_e2e_report.json").exists())

            # 2. Missing --online-grpo-tokenizer-dir when source has token_export -> exit 2
            missing_tok_out = root / "cli_missing_tok"
            proc_missing_tok = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(ROOT / "tools" / "umminfer_stage_c_pipeline.py"),
                    "--verify-online-grpo-preflight",
                    str(preflight_bundle),
                    "--online-grpo-source-dir",
                    str(source_bundle),
                    "--out",
                    str(missing_tok_out),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_missing_tok.returncode, 2, proc_missing_tok.stdout)
            self.assertFalse((missing_tok_out / "stage_c_e2e_report.json").exists())

            # 3. Mismatched source bundle -> exit 2, no report written
            wrong_source = root / "wrong_source"
            wrong_source.mkdir()
            wrong_src_out = root / "cli_wrong_src"
            proc_wrong_src = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(ROOT / "tools" / "umminfer_stage_c_pipeline.py"),
                    "--verify-online-grpo-preflight",
                    str(preflight_bundle),
                    "--online-grpo-source-dir",
                    str(wrong_source),
                    "--online-grpo-tokenizer-dir",
                    str(tok_dir),
                    "--out",
                    str(wrong_src_out),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_wrong_src.returncode, 2, proc_wrong_src.stdout)
            self.assertFalse((wrong_src_out / "stage_c_e2e_report.json").exists())

    def test_independent_online_audit_task42_standalone_cli_and_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # 1. Standalone CLI with --independent-online-audit-accepted (python3 -I)
            cli_out = root / "task42_cli_accepted"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(ROOT / "tools" / "umminfer_stage_c_pipeline.py"),
                    "--independent-online-audit-task42",
                    str(REAL_TASK42_BUNDLE),
                    "--independent-online-audit-accepted",
                    str(REAL_TASK42_ACCEPTED),
                    "--out",
                    str(cli_out),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            cli_report = json.loads(proc.stdout)
            self.assertEqual(cli_report["status"], "COMPLETED")
            self.assertEqual(
                cli_report["source_mode"], "independent_online_audit_task42_only"
            )
            self.assertEqual(cli_report["REAL_TRAINING"], "NOT_RUN")
            self.assertNotIn("independent_online_audit", cli_report)

            audit = cli_report["independent_online_audit_task42"]
            self.assertEqual(audit["status"], "PASS")
            self.assertEqual(audit["sampled_token_count"], 61)
            self.assertEqual(audit["supervised_token_count"], 49)
            self.assertTrue(audit["process_isolated"])
            self.assertTrue(audit["accepted_evidence_checked"])
            self.assertEqual(
                audit["report_path"], "independent_online_audit/task42_audit.json"
            )
            audit_file = cli_out / "independent_online_audit" / "task42_audit.json"
            self.assertTrue(audit_file.is_file())
            self.assertEqual(audit["report_sha256"], ds.sha256_file(audit_file))
            raw_audit = json.loads(audit_file.read_text(encoding="utf-8"))
            self.assertEqual(raw_audit["status"], "PASS")
            self.assertEqual(raw_audit["sampled_token_count"], 61)
            self.assertEqual(raw_audit["supervised_token_count"], 49)

            for claim_key in (
                "new_model_inference",
                "new_logprob_recomputation",
                "online_optimizer",
                "gpu",
                "serving",
                "model_quality",
            ):
                self.assertEqual(audit["claims"][claim_key], "NOT_RUN", claim_key)

            report_file = cli_out / "stage_c_e2e_report.json"
            self.assertNotIn(str(root), report_file.read_text(encoding="utf-8"))

            # 2. Standalone Python API with accepted evidence -> byte-identical determinism
            api_out = root / "task42_api_accepted"
            api_report = pipeline.run_stage_c_pipeline(
                output_dir=api_out,
                independent_online_audit_task42=REAL_TASK42_BUNDLE,
                independent_online_audit_accepted=REAL_TASK42_ACCEPTED,
            )
            self.assertEqual(api_report, cli_report)
            self.assertEqual(
                audit_file.read_bytes(),
                (api_out / "independent_online_audit" / "task42_audit.json").read_bytes(),
            )
            self.assertNotIn("umminfer_online_evidence_audit", sys.modules)
            for mod in ("torch", "transformers", "numpy", "mlx"):
                self.assertNotIn(mod, sys.modules)

            # 3. Standalone Python API without accepted evidence
            api_no_acc = pipeline.run_stage_c_pipeline(
                output_dir=root / "task42_api_no_accepted",
                independent_online_audit_task42=REAL_TASK42_BUNDLE,
            )
            audit_no_acc = api_no_acc["independent_online_audit_task42"]
            self.assertEqual(audit_no_acc["status"], "PASS")
            self.assertEqual(audit_no_acc["sampled_token_count"], 61)
            self.assertEqual(audit_no_acc["supervised_token_count"], 49)
            self.assertTrue(audit_no_acc["process_isolated"])
            self.assertFalse(audit_no_acc["accepted_evidence_checked"])
            self.assertEqual(
                audit_no_acc["report_sha256"],
                ds.sha256_file(
                    root
                    / "task42_api_no_accepted"
                    / "independent_online_audit"
                    / "task42_audit.json"
                ),
            )

    def test_independent_online_audit_preflight_e2e_pass(self):
        import online_grpo_preflight_test as grpo_fixture  # noqa: PLC0415
        import umminfer_online_grpo_preflight as online_grpo  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_bundle, tok_dir = grpo_fixture.make_online_bundle(
                root / "src_exports", version=2, exports=True
            )
            cfg = grpo_fixture.make_config(source_bundle, tok_dir)
            preflight_bundle = root / "preflight_bundle"
            prep = online_grpo.prepare_preflight(
                source_bundle, preflight_bundle, cfg, tokenizer_dir=tok_dir
            )
            before_bundle_bytes = _tree_bytes(preflight_bundle)
            before_source_bytes = _tree_bytes(source_bundle)

            # 1. Standalone CLI: --verify-online-grpo-preflight + --independent-online-audit
            out_cli = root / "preflight_audit_cli"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(ROOT / "tools" / "umminfer_stage_c_pipeline.py"),
                    "--verify-online-grpo-preflight",
                    str(preflight_bundle),
                    "--online-grpo-source-dir",
                    str(source_bundle),
                    "--online-grpo-tokenizer-dir",
                    str(tok_dir),
                    "--independent-online-audit",
                    "--out",
                    str(out_cli),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            cli_report = json.loads(proc.stdout)
            self.assertEqual(cli_report["status"], "COMPLETED")
            self.assertEqual(cli_report["source_mode"], "online_grpo_preflight_only")
            self.assertEqual(cli_report["REAL_TRAINING"], "NOT_RUN")

            og = cli_report["online_grpo_preflight"]
            ind = cli_report["independent_online_audit"]
            self.assertEqual(og["status"], "PASS")
            self.assertEqual(ind["status"], "PASS")
            self.assertEqual(ind["checkpoint_disagreements_count"], 0)
            self.assertEqual(ind["run_id"], og["run_id"])
            self.assertEqual(ind["run_id"], prep["run_id"])
            self.assertTrue(ind["process_isolated"])
            self.assertFalse(ind["accepted_evidence_checked"])
            self.assertEqual(
                ind["report_path"], "independent_online_audit/preflight_audit.json"
            )
            audit_file = out_cli / "independent_online_audit" / "preflight_audit.json"
            self.assertTrue(audit_file.is_file())
            self.assertEqual(ind["report_sha256"], ds.sha256_file(audit_file))
            raw_audit = json.loads(audit_file.read_text(encoding="utf-8"))
            self.assertEqual(raw_audit["status"], "PASS")
            self.assertEqual(raw_audit["checkpoint_disagreements"], [])
            self.assertEqual(raw_audit["run_id"], prep["run_id"])
            self.assertEqual(ind["claims"]["REAL_TRAINING"], "NOT_RUN")
            self.assertEqual(ind["claims"]["optimizer"], "NOT_RUN")
            self.assertEqual(ind["claims"]["gpu"], "NOT_RUN")
            self.assertFalse(ind["claims"]["model_weights_updated"])
            self.assertFalse(ind["claims"]["torch_imported"])
            self.assertEqual(ind["boundaries"]["optimizer"], "NOT_RUN")
            self.assertNotIn(
                str(root), (out_cli / "stage_c_e2e_report.json").read_text(encoding="utf-8")
            )

            # 2. Combined trajectories + producer preflight + independent preflight + task42
            file_a, dir_b = _build_fixture_trajectories(root)
            out_combined = root / "combined_both_audits"
            combined = pipeline.run_stage_c_pipeline(
                output_dir=out_combined,
                trajectory_paths=[file_a, dir_b],
                config_path=ROOT / "configs" / "training" / "joint_sft_default.json",
                execution_mode="dry_run_micro_step",
                dry_run_steps=1,
                verify_online_grpo_preflight=preflight_bundle,
                online_grpo_source_dir=source_bundle,
                online_grpo_tokenizer_dir=tok_dir,
                independent_online_audit=True,
                independent_online_audit_task42=REAL_TASK42_BUNDLE,
            )
            self.assertEqual(combined["status"], "COMPLETED")
            self.assertEqual(combined["source_mode"], "offline_trajectories")
            self.assertEqual(combined["independent_online_audit"]["status"], "PASS")
            self.assertEqual(
                combined["independent_online_audit"]["checkpoint_disagreements_count"], 0
            )
            self.assertEqual(
                combined["independent_online_audit"]["run_id"],
                combined["online_grpo_preflight"]["run_id"],
            )
            self.assertTrue(combined["independent_online_audit"]["process_isolated"])
            self.assertEqual(
                combined["independent_online_audit_task42"]["status"], "PASS"
            )
            self.assertEqual(
                combined["independent_online_audit_task42"]["sampled_token_count"], 61
            )
            self.assertEqual(
                combined["independent_online_audit_task42"]["supervised_token_count"], 49
            )
            self.assertTrue(
                combined["independent_online_audit_task42"]["process_isolated"]
            )
            self.assertTrue(
                (out_combined / "independent_online_audit" / "preflight_audit.json").is_file()
            )
            self.assertTrue(
                (out_combined / "independent_online_audit" / "task42_audit.json").is_file()
            )

            # Input bundles untouched and auditor never imported in-process.
            self.assertEqual(_tree_bytes(preflight_bundle), before_bundle_bytes)
            self.assertEqual(_tree_bytes(source_bundle), before_source_bytes)
            self.assertNotIn("umminfer_online_evidence_audit", sys.modules)
            for mod in ("torch", "transformers", "numpy", "mlx"):
                self.assertNotIn(mod, sys.modules)

    def test_independent_online_audit_fail_closed_flags_and_tampering(self):
        import online_grpo_preflight_test as grpo_fixture  # noqa: PLC0415
        import umminfer_online_grpo_preflight as online_grpo  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file_a, _ = _build_fixture_trajectories(root)
            source_bundle, tok_dir = grpo_fixture.make_online_bundle(
                root / "src_fail", version=1, exports=True
            )
            cfg = grpo_fixture.make_config(source_bundle, tok_dir)
            preflight_bundle = root / "preflight_valid"
            online_grpo.prepare_preflight(
                source_bundle, preflight_bundle, cfg, tokenizer_dir=tok_dir
            )

            # 1. Invalid flag combinations must fail BEFORE creating output_dir
            invalid_api_cases = [
                (
                    "audit_without_preflight",
                    dict(trajectory_paths=[file_a], independent_online_audit=True),
                    "--independent-online-audit requires --verify-online-grpo-preflight",
                ),
                (
                    "audit_without_source_dir",
                    dict(
                        verify_online_grpo_preflight=preflight_bundle,
                        independent_online_audit=True,
                    ),
                    "--verify-online-grpo-preflight requires --online-grpo-source-dir",
                ),
                (
                    "accepted_without_audit",
                    dict(
                        trajectory_paths=[file_a],
                        independent_online_audit_accepted=REAL_TASK42_ACCEPTED,
                    ),
                    "--independent-online-audit-accepted requires",
                ),
                (
                    "accepted_with_both_audits",
                    dict(
                        verify_online_grpo_preflight=preflight_bundle,
                        online_grpo_source_dir=source_bundle,
                        online_grpo_tokenizer_dir=tok_dir,
                        independent_online_audit=True,
                        independent_online_audit_task42=REAL_TASK42_BUNDLE,
                        independent_online_audit_accepted=REAL_TASK42_ACCEPTED,
                    ),
                    "cannot be combined with both",
                ),
            ]
            for label, kwargs, pattern in invalid_api_cases:
                with self.subTest(invalid_api=label):
                    out_dir = root / f"invalid_api_{label}"
                    with self.assertRaisesRegex(ValueError, pattern):
                        pipeline.run_stage_c_pipeline(output_dir=out_dir, **kwargs)
                    self.assertFalse(out_dir.exists())

            invalid_cli_cases = [
                (
                    "audit_without_preflight",
                    ["--from-trajectories", str(file_a), "--independent-online-audit"],
                ),
                (
                    "accepted_without_audit",
                    [
                        "--from-trajectories",
                        str(file_a),
                        "--independent-online-audit-accepted",
                        str(REAL_TASK42_ACCEPTED),
                    ],
                ),
                (
                    "accepted_with_both_audits",
                    [
                        "--verify-online-grpo-preflight",
                        str(preflight_bundle),
                        "--online-grpo-source-dir",
                        str(source_bundle),
                        "--online-grpo-tokenizer-dir",
                        str(tok_dir),
                        "--independent-online-audit",
                        "--independent-online-audit-task42",
                        str(REAL_TASK42_BUNDLE),
                        "--independent-online-audit-accepted",
                        str(REAL_TASK42_ACCEPTED),
                    ],
                ),
            ]
            for label, extra_args in invalid_cli_cases:
                with self.subTest(invalid_cli=label):
                    out_dir = root / f"invalid_cli_{label}"
                    proc = subprocess.run(
                        [
                            sys.executable,
                            "-I",
                            str(ROOT / "tools" / "umminfer_stage_c_pipeline.py"),
                            *extra_args,
                            "--out",
                            str(out_dir),
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(proc.returncode, 2, proc.stdout)
                    self.assertFalse(out_dir.exists())

            # 2. Tampered Task #42 fixture rejected fail-closed before creating output_dir
            tampered_task42 = root / "tampered_task42"
            shutil.copytree(REAL_TASK42_BUNDLE, tampered_task42)
            sse_file = next((tampered_task42 / "live" / "artifacts").glob("*.sse"))
            raw_sse = bytearray(sse_file.read_bytes())
            raw_sse[12] ^= 1
            sse_file.write_bytes(raw_sse)

            out_tampered_t42_api = root / "out_tampered_t42_api"
            with self.assertRaisesRegex(ValueError, "independent online Task #42 audit"):
                pipeline.run_stage_c_pipeline(
                    output_dir=out_tampered_t42_api,
                    independent_online_audit_task42=tampered_task42,
                )
            self.assertFalse(out_tampered_t42_api.exists())

            out_tampered_t42_cli = root / "out_tampered_t42_cli"
            proc_t42 = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(ROOT / "tools" / "umminfer_stage_c_pipeline.py"),
                    "--independent-online-audit-task42",
                    str(tampered_task42),
                    "--out",
                    str(out_tampered_t42_cli),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_t42.returncode, 2, proc_t42.stdout)
            self.assertFalse(out_tampered_t42_cli.exists())

            # 3. Mismatched accepted evidence for Task #42 -> auditor returns FAIL (exit 1) -> fail closed
            bad_accepted = root / "bad_task42_accepted.json"
            bad_doc = json.loads(REAL_TASK42_ACCEPTED.read_text(encoding="utf-8"))
            bad_doc["accepted_cases"][0]["raw_sha256"] = "0" * 64
            bad_accepted.write_text(json.dumps(bad_doc) + "\n", encoding="utf-8")

            out_bad_acc = root / "out_bad_acc_t42"
            with self.assertRaisesRegex(ValueError, "independent online Task #42 audit"):
                pipeline.run_stage_c_pipeline(
                    output_dir=out_bad_acc,
                    independent_online_audit_task42=REAL_TASK42_BUNDLE,
                    independent_online_audit_accepted=bad_accepted,
                )
            self.assertFalse(out_bad_acc.exists())

            # 4. Tampered preflight bundle rejected fail-closed before creating output_dir
            tampered_preflight = root / "tampered_preflight"
            shutil.copytree(preflight_bundle, tampered_preflight)
            sched_path = tampered_preflight / "schedule.json"
            sched_doc = json.loads(sched_path.read_text(encoding="utf-8"))
            sched_doc["checkpoints"][0]["remaining_group_exports_sha256"] = "0" * 64
            grpo_fixture.write_json(sched_path, sched_doc)
            grpo_fixture.refresh_checksums_for_consistent_tamper(tampered_preflight)

            out_tampered_pf = root / "out_tampered_pf"
            with self.assertRaisesRegex(ValueError, "online-GRPO preflight"):
                pipeline.run_stage_c_pipeline(
                    output_dir=out_tampered_pf,
                    verify_online_grpo_preflight=tampered_preflight,
                    online_grpo_source_dir=source_bundle,
                    online_grpo_tokenizer_dir=tok_dir,
                    independent_online_audit=True,
                )
            self.assertFalse(out_tampered_pf.exists())

            # 5. Valid preflight bundle + mismatched accepted evidence -> producer passes,
            #    independent subprocess fails closed with exit 1 before creating output_dir!
            bad_pf_accepted = root / "bad_pf_accepted.json"
            bad_pf_accepted.write_text(
                json.dumps({"run_id": "0" * 64}) + "\n", encoding="utf-8"
            )
            out_bad_pf_acc = root / "out_bad_pf_acc"
            with self.assertRaisesRegex(ValueError, "independent online preflight audit"):
                pipeline.run_stage_c_pipeline(
                    output_dir=out_bad_pf_acc,
                    verify_online_grpo_preflight=preflight_bundle,
                    online_grpo_source_dir=source_bundle,
                    online_grpo_tokenizer_dir=tok_dir,
                    independent_online_audit=True,
                    independent_online_audit_accepted=bad_pf_accepted,
                )
            self.assertFalse(out_bad_pf_acc.exists())

            out_bad_pf_acc_cli = root / "out_bad_pf_acc_cli"
            proc_bad_pf = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(ROOT / "tools" / "umminfer_stage_c_pipeline.py"),
                    "--verify-online-grpo-preflight",
                    str(preflight_bundle),
                    "--online-grpo-source-dir",
                    str(source_bundle),
                    "--online-grpo-tokenizer-dir",
                    str(tok_dir),
                    "--independent-online-audit",
                    "--independent-online-audit-accepted",
                    str(bad_pf_accepted),
                    "--out",
                    str(out_bad_pf_acc_cli),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_bad_pf.returncode, 2, proc_bad_pf.stdout)
            self.assertFalse(out_bad_pf_acc_cli.exists())


if __name__ == "__main__":
    unittest.main()
